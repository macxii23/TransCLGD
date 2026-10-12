import os, json, torch, numpy as np #interchange dataset (geo: xy 2D continuous + attr: lanes/speed discrete classes, adj_flat for diffusion edge conditioning, node_type conditioning)
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.graphgym.config import cfg
from lgd.transform.segment_graph import build_segment_graph #segment coarse-graph construction (optional, active when dataset.segment.enable=True)
from lgd.loader.dataset.graph_store import ensure_jsonl_index, iter_graphs, jsonl_path, list_graph_ids, load_graph #torch-free storage API (re-exported)

NODE_TYPE_MAP = {"mainline": 0, "ramp": 1} #node type mapping (consistent with GraphMaker)
SPEED_VOCAB = [8.33, 9.72, 11.11, 13.89, 16.67, 19.44, 22.22, 27.78, 33.33, 39.44] #10 discrete speed classes (m/s), corresponding to ≈30/35/40/50/60/70/80/100/120/142 km/h
N_LANE_UNI, N_SPEED_UNI = 3, 3 #unified-mode reduced classes: lane(L1/L2/L34) speed(slow/mid/fast), following GeoLDM
UNIFIED_FEAT_DIM = 2 + N_LANE_UNI + N_SPEED_UNI #unified feature dim: xy(2)+lane_oh(3)+speed_oh(3)=8
DEFAULT_COST_BINS = [8400, 14500, 20000, 22800] #cost fallback bins (old 17976-sample 20% equal-frequency quantiles); used when adaptive is off or no persisted file exists
DEFAULT_throughput_BINS = [9960, 12860, 17190, 21430] #throughput fallback bins (old 17976-sample 20% equal-frequency quantiles)
COST_BINS = list(DEFAULT_COST_BINS) #currently active cost boundaries (module-global, overwritten in place from this dataset's distribution when adaptive, imported references stay in sync)
throughput_BINS = list(DEFAULT_throughput_BINS) #currently active throughput boundaries (same as above)
N_COST_CLS, N_throughput_CLS = len(COST_BINS) + 1, len(throughput_BINS) + 1 #number of classes = number of boundaries + 1
Y_MEAN = [20000.0, 17000.0] #regression normalization stats [cost,throughput] (training-set mean; refreshed in place by process, persisted in the bins file, eval/generate share the same denormalization convention)
Y_STD = [5000.0, 3000.0] #regression normalization std [cost,throughput] (default used only as a fallback when no persisted file exists)
_BINS_FILE = '_property_bins.json' #persistence filename for adaptive bins + regression normalization stats (stored in graph_data_dir, shared across the train/eval/generate pipeline under the same convention)

def _set_bins(cost_bins, through_bins): #uniformly update module-global bins: modify lists in place so imported references stay in sync, recompute number of classes
    global N_COST_CLS, N_throughput_CLS
    COST_BINS[:] = [float(b) for b in cost_bins]
    throughput_BINS[:] = [float(b) for b in through_bins]
    N_COST_CLS, N_throughput_CLS = len(COST_BINS) + 1, len(throughput_BINS) + 1

def _equal_freq_bins(vals, n_cls): #equal-frequency bins: n_cls classes -> n_cls-1 internal quantile boundaries (np.quantile is naturally monotonic non-decreasing, always returns n_cls-1 to align with the classification head dim)
    qs = np.linspace(0, 1, n_cls + 1)[1:-1]
    return [float(np.quantile(vals, q)) for q in qs]

def _scan_props(graph_data_dir, max_nodes): #scan dataset graphs to collect (cost,throughput) (N<=max_nodes), used to set boundaries from the reference dataset when "decoupling training data from the binning reference"
    cs, ts = [], []
    for _, g in iter_graphs(graph_data_dir):
        if len(g.get('nodes', [])) > max_nodes: continue
        cs.append(float(g.get('cost', 0.0))); ts.append(float(g.get('throughput', 0.0)))
    return cs, ts

def load_bins(graph_data_dir=None): #load from graph_data_dir/_property_bins.json and refresh global bins, keep current (default) values if no file
    gd = graph_data_dir or cfg.dataset.get('graph_data_dir', 'graph_data')
    fp = os.path.join(gd, _BINS_FILE)
    if os.path.isfile(fp):
        with open(fp, 'r', encoding='utf-8') as f: b = json.load(f)
        _set_bins(b['cost_bins'], b['throughput_bins'])
        if 'y_mean' in b and 'y_std' in b: Y_MEAN[:], Y_STD[:] = [float(v) for v in b['y_mean']], [float(v) for v in b['y_std']] #sync regression normalization stats (modify in place so imported references stay in sync)
    return COST_BINS, throughput_BINS

def speed_to_class(speed_val): #map raw speed to the nearest-matching SPEED_VOCAB class index
    return int(np.argmin([abs(speed_val - v) for v in SPEED_VOCAB]))

def lane_to_class_uni(num_lanes): #unified-mode lane mapping: 1->0(L1), 2->1(L2), 3/4->2(L34)
    return min(max(num_lanes, 1) - 1, 2)

def speed_to_class_uni(speed_val): #unified-mode speed mapping: m/s->km/h->coarse 3 levels (slow/mid/fast), following GeoLDM
    spd_raw = min(int(speed_val * 3.6 / 20), 7)
    return 0 if spd_raw <= 2 else (1 if spd_raw <= 4 else 2)

def _val_to_bin(val, bins): #generic binning: return the interval index that val falls into among the bins (0~len(bins))
    for i, b in enumerate(bins):
        if val < b: return i
    return len(bins)

def cost_to_class(cost_val): return _val_to_bin(cost_val, COST_BINS) #discretize cost -> 5 classes
def throughput_to_class(throughput_val): return _val_to_bin(throughput_val, throughput_BINS) #discretize throughput -> 5 classes

def compute_affinity_target(x_geo, ntype, adj, K=10): #compute the N×K affinity matrix: same-type kNN, +1=edge -1=no edge 0=padding
    n = x_geo.shape[0]
    pos = x_geo.numpy()
    nt = ntype.numpy()
    aff = np.zeros((n, K), dtype=np.float32)
    knn_idx = np.zeros((n, K), dtype=np.int64)
    for i in range(n):
        same = np.where(nt == nt[i])[0]
        same = same[same != i]
        if len(same) == 0: continue
        dists = np.linalg.norm(pos[same] - pos[i], axis=1)
        order = np.argsort(dists)
        topk = same[order[:K]]
        for k, j in enumerate(topk):
            knn_idx[i, k] = j
            aff[i, k] = 1.0 if adj[i, j] > 0 else -1.0
    return torch.tensor(aff, dtype=torch.float), torch.tensor(knn_idx, dtype=torch.long)

class InterchangeDataset(InMemoryDataset):
    """Load interchange graphs from graph_data/graphs.jsonl(.gz) or {id}/graph.json; x=[x,y]*scale 2D continuous, x_attr=[lanes_cls,speed_cls] discrete classes"""
    def __init__(self, stage, root, transform=None, pre_transform=None, pre_filter=None):
        self.stage = stage
        self.file_idx = {'train': 0, 'val': 1, 'test': 2}[stage]
        self.graph_data_dir = cfg.dataset.get('graph_data_dir', 'graph_data')
        self.ctrl_scale = cfg.dataset.get('ctrl_scale', 1.0)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])
        load_bins(self.graph_data_dir) #refresh module-global bins to this dataset's persisted boundaries (so eval labels/repr match the cached y_*_cls)

    @property
    def raw_file_names(self):
        return ['meta.json']

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        pass

    def process(self):
        ctrl_scale = cfg.dataset.get('ctrl_scale', 1.0)
        data_list = []
        for gid, gdata in iter_graphs(self.graph_data_dir): #jsonl(.gz) preferred; folder layout still supported
            nodes, edges = gdata['nodes'], gdata['edges']
            n = len(nodes)
            id_map = {}
            x_geo = torch.zeros(n, 2, dtype=torch.float) #xy coordinates (2D continuous)
            node_lanes = torch.zeros(n, dtype=torch.long) #lanes discrete classes (0-3, corresponding to 1-4 lanes)
            node_speed = torch.zeros(n, dtype=torch.long) #speed discrete classes (0-9, corresponding to SPEED_VOCAB)
            ntype = torch.zeros(n, dtype=torch.long)
            for i, nd in enumerate(nodes):
                id_map[nd['id']] = i
                x_geo[i, 0], x_geo[i, 1] = nd['x'], nd['y']
                node_lanes[i] = max(0, min(nd.get('num_lanes', 1) - 1, 3)) #lanes 1-4 → class 0-3
                node_speed[i] = speed_to_class(nd.get('speed', 0))
                ntype[i] = NODE_TYPE_MAP.get(nd.get('node_type', 'mainline'), 0)
            x_geo = x_geo * ctrl_scale
            x_attr = torch.stack([node_lanes, node_speed], dim=-1).float() #[N,2] input for the attr encoder
            lane_uni = torch.zeros(n, dtype=torch.long) #unified-mode 3-class lanes
            speed_uni = torch.zeros(n, dtype=torch.long) #unified-mode 3-class speed
            for i, nd in enumerate(nodes):
                lane_uni[i] = lane_to_class_uni(nd.get('num_lanes', 1))
                speed_uni[i] = speed_to_class_uni(nd.get('speed', 0))
            lane_oh = torch.nn.functional.one_hot(lane_uni, N_LANE_UNI).float() #[N,3]
            speed_oh = torch.nn.functional.one_hot(speed_uni, N_SPEED_UNI).float() #[N,3]
            x_unified = torch.cat([x_geo, lane_oh, speed_oh], dim=-1) #[N,8] unified feature: xy+lane_oh+speed_oh
            adj = torch.zeros(n, n, dtype=torch.long)
            for e in edges:
                s, t = id_map.get(e['source'], -1), id_map.get(e['target'], -1)
                if s >= 0 and t >= 0:
                    adj[s, t] = 1
                    adj[t, s] = 1
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            edge_attr = torch.ones(edge_index.shape[1], dtype=torch.long)
            adj_flat = adj.flatten().float()
            _cost_raw, _through_raw = gdata.get('cost', 0.0), gdata.get('throughput', 0.0)
            y = torch.tensor([[_cost_raw, _through_raw]], dtype=torch.float) #graph-level label [1,2] (cost,throughput), 2D ensures PyG Batch collates to [B,2]
            if n > cfg.dataset.get('max_nodes', 200): continue # point-level N filter (attribute stage and topology stage share this)
            aff_k = (cfg.get('diffusion', {}) or {}).get('affinity_k', 10)
            x_affinity, affinity_knn_idx = compute_affinity_target(x_geo, ntype, adj, K=aff_k)
            d = Data(x=x_geo, edge_index=edge_index, edge_attr=edge_attr, y=y,
                     num_node_per_graph=torch.tensor(n, dtype=torch.long),
                     adj_flat=adj_flat, node_type=ntype, pos_orig=x_geo.clone(),
                     x_attr=x_attr, node_lanes=node_lanes, node_speed=node_speed,
                     x_affinity=x_affinity, affinity_knn_idx=affinity_knn_idx,
                     x_unified=x_unified, lane_uni=lane_uni, speed_uni=speed_uni) #y_cost_cls/y_throughput_cls are labeled uniformly later once the bins are determined
            seg_cfg = cfg.dataset.get('segment', {}) or {}
            if seg_cfg.get('enable', False): # attach seg_* fields (SegmentRewire promotes them to primary fields during topology training)
                for k, v in build_segment_graph(x_geo, adj, ntype).items(): d[k] = v
                max_m = seg_cfg.get('max_nodes', None) # optional coarse-graph M cap (independent of point-level max_nodes)
                if max_m is not None and int(d.seg_num_nodes) > int(max_m): continue
            data_list.append(d)
        if not data_list:
            raise RuntimeError(f'InterchangeDataset.process: no graphs kept from {self.graph_data_dir} '
                               f'(max_nodes={cfg.dataset.get("max_nodes", 200)}, segment.enable={bool((cfg.dataset.get("segment", {}) or {}).get("enable", False))}). '
                               f'Clear {self.processed_dir} after changing filters, and ensure the data directory is not empty.')
        all_y = torch.cat([d.y for d in data_list], dim=0) #[N_total, 2] compute graph-level label mean/std for normalization
        y_mean, y_std = all_y.mean(0, keepdim=True), all_y.std(0, keepdim=True).clamp(min=1e-6) #[1,2] keep 2D to broadcast-align with batch.y
        Y_MEAN[:], Y_STD[:] = y_mean.flatten().tolist(), y_std.flatten().tolist() #refresh regression normalization stats in place (training-set convention), persisted together with the bins
        if cfg.dataset.get('adaptive_bins', True): #adaptive equal-frequency binning: recompute boundaries from this dataset's cost/throughput distribution and persist (shared by eval/generate)
            enc = cfg.get('encoder', None) #number of classes uses encoder as the single source (classification head dim), default 5
            ncc = int(enc.get('num_cost_classes', 5)) if enc is not None else 5
            ntc = int(enc.get('num_throughput_classes', 5)) if enc is not None else 5
            src = cfg.dataset.get('bins_source_dir', '') or self.graph_data_dir #binning reference dir: default = the training data itself; setting bins_source_dir (e.g. four-way graph_data) decouples training data from binning
            if os.path.abspath(src) == os.path.abspath(self.graph_data_dir): #reference is the training set -> use the already-aggregated all_y directly, saving one scan
                cvals, tvals = all_y[:, 0].numpy(), all_y[:, 1].numpy()
            else:
                cvals, tvals = _scan_props(src, cfg.dataset.get('max_nodes', 200)) #set boundaries from the reference dataset's distribution (same max_nodes convention)
            _set_bins(_equal_freq_bins(cvals, ncc), _equal_freq_bins(tvals, ntc))
            with open(os.path.join(self.graph_data_dir, _BINS_FILE), 'w', encoding='utf-8') as f:
                json.dump({'cost_bins': COST_BINS, 'throughput_bins': throughput_BINS, 'n_cost_cls': N_COST_CLS,
                           'n_throughput_cls': N_throughput_CLS, 'num_graphs': int(all_y.shape[0]),
                           'bins_source': os.path.abspath(src), 'bins_source_n': len(cvals),
                           'y_mean': Y_MEAN, 'y_std': Y_STD}, f, indent=2, ensure_ascii=False) #y_mean/y_std shared across the pipeline for regression denormalization
        for d in data_list:
            d.y_mean, d.y_std = y_mean, y_std
            d.y_cost_cls = torch.tensor([cost_to_class(d.y[0, 0].item())], dtype=torch.long) #generate the cost class [1] from the (adaptive) bins
            d.y_throughput_cls = torch.tensor([throughput_to_class(d.y[0, 1].item())], dtype=torch.long) #throughput class [1]
        np.random.seed(42)
        perm = np.random.permutation(len(data_list))
        n_train = int(len(data_list) * 0.8)
        n_val = int(len(data_list) * 0.1)
        splits = [perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]]
        for i, idx in enumerate(splits):
            split_list = [data_list[j] for j in idx]
            torch.save(self.collate(split_list), self.processed_paths[i])
