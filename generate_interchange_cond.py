#!/usr/bin/env python
"""Task A - property-conditioned de novo interchange generation: the topology stage diffusion directly generates the coarse graph (uncoarsened to the full adjacency A), then the attribute stage jointly generates node attributes (coordinates + lanes + speed) conditioned on that topology; Top-K reference graphs only seed the coarse-graph size M; outputs training-compatible JSON + _summary.csv"""
import os, json, argparse, random, logging, yaml, numpy as np, torch
import lgd  # noqa register custom modules/encoders/datasets
from torch_geometric.data import Data, Batch
from torch_geometric.graphgym.config import cfg, set_cfg
from lgd.transform.posenc_stats import compute_posenc_stats
from lgd.transform.transforms import no_virtual_node_edge, pretransform_pe, segment_rewire  # dense edge + pestat_node (same order as master_loader); segment_rewire feeds the topology stage
from lgd.ddpm.LGD import LatentDiffusion
from lgd.loader.dataset.interchange import InterchangeDataset, N_LANE_UNI, N_SPEED_UNI

LANE_UNI_TO_NUM = lambda c: int(min(int(c) + 1, 3))                  # unified lane class 0/1/2 -> 1/2/3 lanes
SPEED_UNI_TO_MS = lambda c: (11.11, 19.44, 27.78)[min(int(c), 2)]   # unified speed class -> representative m/s (slow/mid/fast)


def _reset_load_cfg(cfg_path):                                       # reset the cfg singleton and load a new yaml (clear() first to avoid cross-stage residue like adj_cond/unified_cond)
    if cfg.is_frozen(): cfg.defrost()
    cfg.clear()
    set_cfg(cfg); cfg.set_new_allowed(True); cfg.merge_from_file(cfg_path)


def _build_model(cfg_path, ckpt):                                    # build LatentDiffusion from cfg + load weights + eval
    _reset_load_cfg(cfg_path)
    m = LatentDiffusion(timesteps=cfg.diffusion.get('timesteps', 1000), conditioning_key=cfg.diffusion.conditioning_key,
                        hid_dim=cfg.diffusion.hid_dim, parameterization=cfg.diffusion.get('parameterization', 'x0'),
                        cond_stage_key=cfg.diffusion.cond_stage_key, first_stage_config=cfg.diffusion.first_stage_config,
                        cond_stage_config=cfg.diffusion.cond_stage_config,
                        edge_factor=cfg.diffusion.get('edge_factor', 1.0), graph_factor=cfg.diffusion.get('graph_factor', 1.0)).to(torch.device(cfg.accelerator))
    sd = torch.load(ckpt, map_location='cpu')
    sd = sd.get('model_state', sd.get('state_dict', sd))            # graphgym(model_state)/lightning(state_dict)/bare state_dict
    missing, unexpected = m.load_state_dict(sd, strict=False)
    logging.info(f'[LoadCkpt] {ckpt} missing={len(missing)} unexpected={len(unexpected)}')
    if missing: logging.warning(f'[LoadCkpt] MISSING keys (first 20): {missing[:20]}')
    if unexpected: logging.warning(f'[LoadCkpt] UNEXPECTED keys (first 20): {unexpected[:20]}')
    m.eval(); return m


def _load_splits(root):                                             # load the three InterchangeDataset splits (preprocessed, carries y_mean/y_std)
    return {k: InterchangeDataset(stage=k, root=root) for k in ('train', 'val', 'test')}


def _prepare_infer_cfg(graph_data_dir, len_scale=10.0):            # point the dataset at the inference source, enable segment fields (needed by the topology stage), and disable adaptive-bin file writes
    from yacs.config import CfgNode
    cfg.dataset.graph_data_dir = graph_data_dir; cfg.dataset.adaptive_bins = False
    if 'segment' not in cfg.dataset: cfg.dataset.segment = CfgNode(new_allowed=True)
    cfg.dataset.segment.enable = True; cfg.dataset.segment.len_scale = float(len_scale)


def _load_train_ystats(train_stats_dir):                          # y_mean/y_std from the training-set statistics (inference normalization must match training), read from _property_bins.json
    with open(os.path.join(train_stats_dir, '_property_bins.json'), 'r', encoding='utf-8') as f: b = json.load(f)
    return np.asarray(b['y_mean'], np.float32).reshape(-1), np.asarray(b['y_std'], np.float32).reshape(-1)


def _scan_meta(cache, root):                                        # scan graph_data (jsonl.gz or {id}/graph.json) for (gid, cost, throughput, N), with an npz cache
    if os.path.isfile(cache):
        z = np.load(cache, allow_pickle=True); return list(z['ids']), z['y'].astype(np.float32), z['N'].astype(np.int32)
    from lgd.loader.dataset.interchange import iter_graphs
    ids, ys, Ns = [], [], []
    for gid, g in iter_graphs(root):
        ids.append(gid); ys.append([g.get('cost', 0.0), g.get('throughput', 0.0)]); Ns.append(len(g.get('nodes', [])))
    ids = np.asarray(ids); ys = np.asarray(ys, dtype=np.float32); Ns = np.asarray(Ns, dtype=np.int32)
    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True); np.savez(cache, ids=ids, y=ys, N=Ns)
    return list(ids), ys, Ns


def _build_id2idx(ids_all, Ns_all, max_n, seed=42):                 # reproduce InterchangeDataset.process filter + random.seed(42) permutation split: gid -> (split_name, local_idx)
    kept_ids = [gid for gid, N in zip(ids_all, Ns_all) if N <= max_n]
    np.random.seed(seed); perm = np.random.permutation(len(kept_ids))
    n_tr = int(len(kept_ids) * 0.8); n_va = int(len(kept_ids) * 0.1)
    m = {}
    for li, pi in enumerate(perm[:n_tr]):                  m[kept_ids[pi]] = ('train', li)
    for li, pi in enumerate(perm[n_tr:n_tr + n_va]):       m[kept_ids[pi]] = ('val', li)
    for li, pi in enumerate(perm[n_tr + n_va:]):           m[kept_ids[pi]] = ('test', li)
    return m


def _pick_pool(ids_all, ys_all, id2idx, target, topk, pool_source, metric='rel_chebyshev'):  # Top-K reference graphs by (cost,throughput) proximity, used only as coarse-graph size-M seeds
    t = np.asarray(target, dtype=np.float32)
    if metric == 'zscore_l2':
        yn = (ys_all - ys_all.mean(0)) / np.clip(ys_all.std(0), 1e-6, None); tn = (t - ys_all.mean(0)) / np.clip(ys_all.std(0), 1e-6, None)
        dist = np.sqrt(((yn - tn) ** 2).sum(-1))
    else:
        rel = np.abs(ys_all - t) / np.clip(np.abs(t), 1e-6, None)   # per-axis relative error (target as denominator, same as MAPE)
        dist = rel.max(-1) if metric == 'rel_chebyshev' else np.sqrt((rel ** 2).sum(-1))
    pool = []
    for k in np.argsort(dist):
        gid = ids_all[k]
        if gid not in id2idx: continue
        if pool_source != 'all' and id2idx[gid][0] != pool_source: continue
        pool.append(gid)
        if len(pool) >= topk: break
    return pool


def _sample_no_repl(pool, n):                                       # sample n gids without replacement (reshuffle-loop if n exceeds pool size)
    if n <= len(pool): return random.sample(pool, n)
    res = []
    while len(res) < n:
        p = pool[:]; random.shuffle(p); res += p
    return res[:n]


def _apply_transforms(data):                                       # reproduce master_loader pretransform: RRWP posenc -> dense edge [N^2] -> pestat_node
    data = compute_posenc_stats(data, pe_types=['RRWP'], is_undirected=True, cfg=cfg)
    data = no_virtual_node_edge(data, format=cfg.dataset.format)
    data = pretransform_pe(data, add_virtual_node=False)
    return data


_TRANS_CACHE = {}                                                  # gid -> transformed Data cache


def _fetch_data(gid, id2idx, splits):                             # get transformed full-graph Data by gid (lazy transform + cache), returns a clone
    if gid not in _TRANS_CACHE:
        sp, li = id2idx[gid]
        _TRANS_CACHE[gid] = _apply_transforms(splits[sp][li].clone())
    return _TRANS_CACHE[gid].clone()


def _build_topo_data(gid, id2idx, splits):                       # segment coarse-graph Data: rewire seg_* to primary fields -> transform, feeds the topology diffusion
    sp, li = id2idx[gid]
    return _apply_transforms(segment_rewire(splits[sp][li].clone()))


def _build_node_mask(batch, hid_dim, device):                    # mainline=1 (known/locked) / ramp=0 (to be generated), shape [N, hid]
    return (batch.node_type.to(device) == 0).float().unsqueeze(-1).expand(-1, hid_dim).contiguous()


def _make_unified(xy_n, lane_uni, speed_uni):                    # [N,2] xy(normalized) + lane one-hot(3) + speed one-hot(3) -> unified feature [N,8]
    N = int(np.asarray(xy_n).reshape(-1, 2).shape[0])
    lo = torch.nn.functional.one_hot(torch.as_tensor(lane_uni, dtype=torch.long), N_LANE_UNI).float()
    so = torch.nn.functional.one_hot(torch.as_tensor(speed_uni, dtype=torch.long), N_SPEED_UNI).float()
    return torch.cat([torch.as_tensor(xy_n, dtype=torch.float).reshape(N, 2), lo, so], dim=-1)


def _parse_unified_outputs(gp):                                  # unified inference item = ('unified', xy, adj, lanes_uni, speed_uni); returns per-graph (xy, adj, lane_uni, speed_uni)
    out = []
    for item in gp:
        _, xy, adj, lu, su = item
        out.append((xy.detach().float().cpu(), adj.detach().long().cpu(), lu.detach().long().cpu().numpy(), su.detach().long().cpu().numpy()))
    return out


@torch.no_grad()
def _run_stage_topo(model, tasks, splits, id2idx, device, batch_size, ddim_steps, ddim_eta, use_ddpm_steps, y_mean, y_std):  # topology (property-conditioned): sample (W,L,type) -> uncoarsen to full A; tasks=[(ti,gid,target)] -> [(ti,gid,target,full_adj,node_type)] (gid only fixes the segment count M; W/lengths/types are sampled from noise)
    ym = torch.as_tensor(y_mean).float().reshape(1, -1); ys = torch.as_tensor(y_std).float().reshape(1, -1)
    res = []
    for s in range(0, len(tasks), batch_size):
        grp = tasks[s:s + batch_size]
        dl = [_build_topo_data(gid, id2idx, splits) for _, gid, _ in grp]
        for d, (_, _, tg) in zip(dl, grp): d.y = torch.tensor([tg], dtype=torch.float); d.y_mean = ym.clone(); d.y_std = ys.clone()  # normalize the target with training statistics
        batch = Batch.from_data_list(dl).to(device)
        _, gp = model.inference(batch, ddim_steps=ddim_steps, ddim_eta=ddim_eta, use_ddpm_steps=use_ddpm_steps)
        for (ti, gid, tg), item in zip(grp, gp):
            meta = item[3]; res.append((ti, gid, tg, meta['full_adj'].long().cpu(), meta['full_node_type'].long().cpu()))
        logging.info(f'[Topo] {s + len(grp)}/{len(tasks)} N0={res[-1][3].shape[0]}')
    return res


def _feature_skeleton(full_adj, node_type, target, y_mean, y_std, x_unified=None):  # downstream skeleton for the attribute stage: adjacency condition + node_type + x_unified (placeholder for generation, or reference values for RePaint/perturbation)
    N = int(full_adj.shape[0]); ei = full_adj.nonzero(as_tuple=False).t().contiguous().long()
    xu = x_unified if x_unified is not None else torch.zeros(N, 2 + N_LANE_UNI + N_SPEED_UNI)
    d = Data(x=torch.zeros(N, 2), x_unified=xu, edge_index=ei, edge_attr=torch.ones(ei.shape[1], dtype=torch.long),
             node_type=node_type.long(), pos_orig=torch.zeros(N, 2), adj_flat=full_adj.flatten().float(),
             num_node_per_graph=torch.tensor(N, dtype=torch.long), y=torch.tensor([target], dtype=torch.float),
             y_mean=torch.as_tensor(y_mean).float().reshape(1, -1), y_std=torch.as_tensor(y_std).float().reshape(1, -1))
    return _apply_transforms(d)


@torch.no_grad()
def _run_stage_feature(model, skels, device, batch_size, ddim_steps, ddim_eta, use_ddpm_steps, lock_mainline=False, start_t0=None, noise_temp=1.0):  # attribute stage: given topology, jointly generate coordinates+lanes+speed; lock_mainline=RePaint hard-replace mainline latent each step; start_t0=perturb reference latent from step t0 (task C); returns per-skeleton (xy, adj, lane_uni, speed_uni, node_type)
    res = []; hid = int(cfg.diffusion.hid_dim)
    for s in range(0, len(skels), batch_size):
        dl = [skels[j].clone() for j in range(s, min(s + batch_size, len(skels)))]
        batch = Batch.from_data_list(dl).to(device)
        kw = dict(ddim_steps=ddim_steps, ddim_eta=ddim_eta, use_ddpm_steps=use_ddpm_steps)
        if lock_mainline: kw['mask'] = _build_node_mask(batch, hid, device)
        if start_t0 is not None: kw['start_t0'] = int(start_t0)
        if float(noise_temp) != 1.0: kw['noise_temp'] = float(noise_temp)
        _, gp = model.inference(batch, **kw)
        for j, (xy, adj, lu, su) in enumerate(_parse_unified_outputs(gp)):
            d = dl[j]; ntype = d.node_type.numpy()
            if lock_mainline:                                       # hard-overwrite mainline nodes with reference (from skeleton x_unified)
                im = (d.node_type == 0).nonzero(as_tuple=False).flatten().numpy(); xu = d.x_unified.numpy()
                xy[im] = torch.as_tensor(xu[im, :2]).float(); lu[im] = xu[im, 2:2 + N_LANE_UNI].argmax(1); su[im] = xu[im, 2 + N_LANE_UNI:].argmax(1)
            res.append((xy, adj, lu, su, ntype))
        logging.info(f'[Feature] {min(s + batch_size, len(skels))}/{len(skels)}')
    return res


def _denorm_xy(pos_n, ctrl_scale, diag):                          # de-normalize: undo ctrl_scale then multiply by diag (center-origin meters, no cx/cy shift)
    return (pos_n.numpy() / ctrl_scale) * diag


def _load_tpl_json(gid, root):                                    # read a reference graph (jsonl.gz or {id}/graph.json): norm_params/edges/node_type
    from lgd.loader.dataset.interchange import load_graph
    return load_graph(root, gid)


def _dump_json(save_p, tpl_g, pos_m, pos_n, adj_np, num_lanes, speed_ms, target, z_fill, diag, denorm_on, gen_ntype=None):  # write training-compatible JSON (nodes x/y normalized + x_m/y_m meters); gen_ntype non-null uses generated node_type (0 mainline / 1 ramp)
    nodes_t = tpl_g.get('nodes', []); N = pos_n.shape[0]; nodes = []
    for i in range(N):
        nt_str = ('mainline' if int(gen_ntype[i]) == 0 else 'ramp') if gen_ntype is not None else (nodes_t[i].get('node_type', 'mainline') if i < len(nodes_t) else 'mainline')
        nd = {'id': i, 'x': round(float(pos_n[i, 0]), 6), 'y': round(float(pos_n[i, 1]), 6), 'z': float(z_fill),
              'node_type': nt_str, 'num_lanes': int(num_lanes[i]), 'speed': round(float(speed_ms[i]), 2)}
        if denorm_on: nd['x_m'], nd['y_m'] = round(float(pos_m[i, 0]), 3), round(float(pos_m[i, 1]), 3)
        nodes.append(nd)
    edges = []; a = adj_np.numpy() if isinstance(adj_np, torch.Tensor) else adj_np
    ref_types = {(e['source'], e['target']): e.get('edge_type', 'type1') for e in tpl_g.get('edges', [])}
    ref_types.update({(e['target'], e['source']): e.get('edge_type', 'type1') for e in tpl_g.get('edges', [])})
    for i in range(N):
        for j in range(i + 1, N):
            if a[i, j] > 0 or a[j, i] > 0:
                dist_m = float(np.linalg.norm(pos_m[i] - pos_m[j]))
                edges.append({'source': i, 'target': j, 'distance': round(dist_m / diag if not denorm_on else dist_m, 3), 'edge_type': ref_types.get((i, j), ref_types.get((j, i), 'type1'))})
    np_out = {**tpl_g.get('norm_params', {}), 'diag': diag}
    with open(save_p, 'w', encoding='utf-8') as f:
        json.dump({'norm_params': np_out, 'nodes': nodes, 'edges': edges, 'cost': float(target[0]), 'throughput': float(target[1])}, f, ensure_ascii=False, indent=2)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--cfg', required=True)
    pa.add_argument('--targets', default=None, help='override yaml: "c1,t1;c2,t2"')
    pa.add_argument('--n', type=int, default=None, help='override total_per_target')
    pa.add_argument('--topk', type=int, default=None, help='override topk_pool')
    pa.add_argument('--batch-size', type=int, default=None, help='override yaml batch_size')
    pa.add_argument('--out', default=None)
    a = pa.parse_args()
    with open(a.cfg, 'r', encoding='utf-8') as f: gc = yaml.safe_load(f)['generate']
    if a.targets: gc['targets'] = [[float(v) for v in pair.split(',')] for pair in a.targets.split(';')]
    if a.n is not None: gc['total_per_target'] = int(a.n)
    if a.topk is not None: gc['topk_pool'] = int(a.topk)
    if a.batch_size is not None: gc['batch_size'] = int(a.batch_size)
    if a.out: gc['out_dir'] = a.out
    random.seed(gc['seed']); np.random.seed(gc['seed']); torch.manual_seed(gc['seed'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    os.makedirs(gc['out_dir'], exist_ok=True)

    ids_all, ys_all, Ns_all = _scan_meta(gc['meta_cache'], gc['graph_data_dir'])   # 1. inference-source metadata (four-leg subset)
    _reset_load_cfg(gc['attribute_cfg']); _prepare_infer_cfg(gc['graph_data_dir'])  # 2. attribute-stage cfg, pointed at the inference source with segment fields enabled
    max_n = cfg.dataset.get('max_nodes', 200); ctrl_scale = float(cfg.dataset.get('ctrl_scale', 1.0))
    splits = _load_splits(gc['dataset_root']); id2idx = _build_id2idx(ids_all, Ns_all, max_n, seed=gc['seed'])
    y_mean, y_std = _load_train_ystats(gc.get('train_stats_dir', gc['graph_data_dir']))  # normalization from training statistics
    logging.info(f'[Meta] ids={len(ids_all)} kept={len(id2idx)} y_mean={y_mean} y_std={y_std}')

    tasks = []                                                                     # 3. build tasks: Top-K reference graphs seed the coarse-graph size M (adjacency is generated, not copied)
    pool_metric = gc.get('pool_metric', 'rel_chebyshev')
    for ti, tg in enumerate(gc['targets']):
        pool = _pick_pool(ids_all, ys_all, id2idx, tg, gc['topk_pool'], gc['pool_source'], pool_metric)
        if not pool: logging.warning(f'target={tg} has no usable reference'); continue
        for gid in _sample_no_repl(pool, gc['total_per_target']): tasks.append((ti, gid, tg))

    mt = _build_model(gc['topology_cfg'], gc['topology_ckpt']); device = torch.device(cfg.accelerator)   # 4. topology stage: property-conditioned segment diffusion -> full adjacency A
    topo_res = _run_stage_topo(mt, tasks, splits, id2idx, device, gc['batch_size'], gc.get('ddim_steps'), gc.get('ddim_eta', 0.0), gc.get('use_ddpm_steps', False), y_mean, y_std)
    del mt; torch.cuda.empty_cache()

    mf = _build_model(gc['attribute_cfg'], gc['attribute_ckpt']); device = torch.device(cfg.accelerator)  # 5. attribute stage: unified diffusion -> coordinates + lanes + speed jointly
    ctrl_scale = float(cfg.dataset.get('ctrl_scale', ctrl_scale))
    skels = [_feature_skeleton(adj, nt, tg, y_mean, y_std) for (_, _, tg, adj, nt) in topo_res]
    feat = _run_stage_feature(mf, skels, device, gc['batch_size'], gc.get('ddim_steps'), gc.get('ddim_eta', 0.0), gc.get('use_ddpm_steps', False))
    del mf; torch.cuda.empty_cache()

    diag = float(gc['global_diag']); denorm_on = bool(gc.get('denorm', True))      # 6. de-normalize + write JSON
    summary_p = os.path.join(gc['out_dir'], '_summary.csv'); per_t = {}
    with open(summary_p, 'w', encoding='utf-8') as fsum:
        fsum.write('idx,target_cost,target_throughput,seed_gid,N,lanes_mean,speed_mean_ms\n')
        for (ti, gid, tg, _, _), (xy, adj, lu, su, ntype) in zip(topo_res, feat):
            num_lanes = np.array([LANE_UNI_TO_NUM(c) for c in lu]); speed_ms = np.array([SPEED_UNI_TO_MS(c) for c in su])
            pos_m = _denorm_xy(xy, ctrl_scale, diag)
            sub = os.path.join(gc['out_dir'], f'c{int(tg[0])}_t{int(tg[1])}'); os.makedirs(sub, exist_ok=True)
            idx = per_t.get(ti, 0); per_t[ti] = idx + 1
            base = os.path.join(sub, f'{idx:04d}_seed{gid}')
            tpl_g = _load_tpl_json(gid, gc['graph_data_dir'])
            _dump_json(base + '.json', tpl_g, pos_m, xy.numpy() / ctrl_scale, adj, num_lanes, speed_ms, tg, gc['z_fill'], diag, denorm_on, gen_ntype=ntype)
            fsum.write(f'{idx},{tg[0]:.2f},{tg[1]:.2f},{gid},{xy.shape[0]},{float(num_lanes.mean()):.2f},{float(speed_ms.mean()):.2f}\n')
    logging.info(f'[Done] {len(feat)} samples -> {gc["out_dir"]} (summary: {summary_p})')


if __name__ == '__main__':
    main()
