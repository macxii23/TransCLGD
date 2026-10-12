#!/usr/bin/env python
"""Multi-stage feasibility evaluation of generated interchanges: 1) quality checking, 2) rationality checking, 3) property matching."""
import argparse, csv, json, logging, os, random, statistics
from collections import defaultdict
from glob import glob
import numpy as np, torch
import lgd  # noqa
from torch_geometric.data import Batch, Data
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything
from lgd.loader.dataset.interchange import (N_LANE_UNI, N_SPEED_UNI, NODE_TYPE_MAP, Y_MEAN, Y_STD,
                                            compute_affinity_target, lane_to_class_uni, load_bins, speed_to_class_uni)
from lgd.transform.posenc_stats import compute_posenc_stats
from lgd.transform.transforms import no_virtual_node_edge, pretransform_pe
from interchange_validity import _xy_nodes, _adj_edges, graph_valid as _graph_valid  #stage 1/2 criteria centralized in interchange_validity
from eval_property_predictor import load_model

QUALITY_REASONS = ('too_few_nodes', 'too_many_nodes', 'nonfinite_coord', 'degenerate_coord', 'bad_lane', 'no_edge') #stage 1 = malformed graph / abnormal node attributes; any other rejection reason belongs to stage 2
CSV_FIELDS = ('path', 'stage', 'reason', 'target_cost', 'target_throughput', 'cost_est', 'cost_abs_err', 'cost_rel_err',
              'throughput_est', 'throughput_abs_err', 'throughput_rel_err')


def _load_cfg(cfg_path):
    cfg.set_new_allowed(True); set_cfg(cfg); cfg.set_new_allowed(True)
    class A: pass
    A.cfg_file = cfg_path; A.opts = []
    load_cfg(cfg, A())
    cfg.out_dir = os.path.join(cfg.out_dir, os.path.splitext(os.path.basename(cfg_path))[0]); cfg.run_dir = os.path.join(cfg.out_dir, '0')


def _json_paths(gen_dir, limit=0):
    paths = sorted(p for p in glob(os.path.join(gen_dir, '**', '*.json'), recursive=True) if not os.path.basename(p).startswith('_'))
    return paths[:limit] if limit > 0 else paths


def _read_graph(path):
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)


def _cost_from_graph(g, unit_price): #closed-form cost: Σ_edges ||xa-xb||·diag·min(la,lb)·unit_price (the predictor has no cost head)
    nodes, edges = g.get('nodes', []), g.get('edges', []); diag = float(g.get('norm_params', {}).get('diag', 1.0))
    id_map = {n.get('id', i): i for i, n in enumerate(nodes)}; s = 0.0
    for e in edges:
        si, ti = id_map.get(e.get('source'), -1), id_map.get(e.get('target'), -1)
        if si < 0 or ti < 0: continue
        a, b = nodes[si], nodes[ti]
        d = ((float(a.get('x', 0)) - float(b.get('x', 0))) ** 2 + (float(a.get('y', 0)) - float(b.get('y', 0))) ** 2) ** 0.5 * diag
        s += d * min(int(a.get('num_lanes', 1)), int(b.get('num_lanes', 1)))
    return s * unit_price


def _attr_classes(g):
    lanes = np.asarray([lane_to_class_uni(int(n.get('num_lanes', 1))) for n in g.get('nodes', [])], dtype=np.int64)
    speeds = np.asarray([speed_to_class_uni(float(n.get('speed', 0.0))) for n in g.get('nodes', [])], dtype=np.int64)
    return lanes, speeds


def _to_data(g, ctrl_scale): #generated JSON -> predictor input (x_unified = xy + lane one-hot + speed one-hot)
    nodes = g.get('nodes', []); n = len(nodes)
    xy = torch.tensor(_xy_nodes(g), dtype=torch.float) * ctrl_scale
    adj = torch.tensor(_adj_edges(g), dtype=torch.long)
    lanes, speeds = _attr_classes(g)
    lane_oh = torch.nn.functional.one_hot(torch.tensor(lanes), N_LANE_UNI).float()
    speed_oh = torch.nn.functional.one_hot(torch.tensor(speeds), N_SPEED_UNI).float()
    node_type = torch.tensor([NODE_TYPE_MAP.get(nd.get('node_type', 'mainline'), 0) for nd in nodes], dtype=torch.long)
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    y = torch.tensor([[float(g.get('cost', 0.0)), float(g.get('throughput', 0.0))]], dtype=torch.float) #target vector the sample was generated for
    d = Data(x=xy, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[1], dtype=torch.long), y=y,
             num_node_per_graph=torch.tensor(n, dtype=torch.long), adj_flat=adj.flatten().float(), node_type=node_type,
             pos_orig=xy.clone(), x_unified=torch.cat([xy, lane_oh, speed_oh], -1))
    k = cfg.diffusion.get('affinity_k', 10) if hasattr(cfg, 'diffusion') else 10
    d.x_affinity, d.affinity_knn_idx = compute_affinity_target(xy, node_type, adj, K=k)
    d = compute_posenc_stats(d, pe_types=['RRWP'], is_undirected=True, cfg=cfg)
    return pretransform_pe(no_virtual_node_edge(d, format=cfg.dataset.format), add_virtual_node=cfg.dataset.add_virtual_node_edge)


@torch.no_grad()
def _predict(paths, graphs, model, dev, batch_size, ctrl_scale, unit_price): #stage 3: throughput from the predictor, cost from the closed-form formula, both compared against the raw target vector
    rows = []
    for s in range(0, len(paths), batch_size):
        bp, bg = paths[s:s + batch_size], graphs[s:s + batch_size]
        data = [_to_data(g, ctrl_scale) for g in bg]
        batch = Batch.from_data_list(data).to(dev); batch.x = batch.x_unified
        _, _, graph_pred = model.decode(model.forward(batch))
        thr = (graph_pred[1].cpu() * Y_STD[1] + Y_MEAN[1]).tolist() #de-normalize with the training-set statistics
        for p, d, g, pthr in zip(bp, data, bg, thr):
            tg_cost, tg_thr = float(d.y[0, 0]), float(d.y[0, 1])
            ce = _cost_from_graph(g, unit_price)
            rows.append({'path': p, 'stage': 'property_matching', 'reason': 'ok', 'target_cost': tg_cost, 'target_throughput': tg_thr,
                         'cost_est': ce, 'cost_abs_err': abs(ce - tg_cost), 'cost_rel_err': abs(ce - tg_cost) / max(abs(tg_cost), 1e-6),
                         'throughput_est': pthr, 'throughput_abs_err': abs(pthr - tg_thr), 'throughput_rel_err': abs(pthr - tg_thr) / max(abs(tg_thr), 1e-6)})
        logging.info('[Predict] %d/%d', min(s + batch_size, len(paths)), len(paths))
    return rows


def _err_stats(rows, name): #MAE + MAPE (mean/median/p90) of one property over the given rows
    ae = [r[f'{name}_abs_err'] for r in rows]; rel = [r[f'{name}_rel_err'] for r in rows]
    return {f'{name}_mae': float(np.mean(ae)), f'{name}_mape': float(np.mean(rel)),
            f'{name}_mape_median': float(statistics.median(rel)), f'{name}_mape_p90': float(np.percentile(rel, 90))}


def _match_stats(rows): #stage-3 aggregate: cost/throughput errors vs target + mean generated properties
    return {**_err_stats(rows, 'cost'), **_err_stats(rows, 'throughput'),
            'gen_cost_mean': float(np.mean([r['cost_est'] for r in rows])),
            'gen_throughput_mean': float(np.mean([r['throughput_est'] for r in rows]))}


def main():
    pa = argparse.ArgumentParser(description='multi-stage feasibility evaluation of generated interchanges: quality checking, rationality checking, property matching')
    pa.add_argument('--gen_dir', default='results/generated'); pa.add_argument('--predictor_cfg', default='cfg/interchange-property-predictor.yaml')
    pa.add_argument('--predictor_ckpt', required=True); pa.add_argument('--graph_data_dir', default='graph_data_all', help='dataset providing the predictor normalization statistics (_property_bins.json)')
    pa.add_argument('--batch_size', type=int, default=16); pa.add_argument('--limit', type=int, default=0, help='evaluate only the first N generated graphs (0=all)')
    pa.add_argument('--generated_labels', default='ramp', help='comma-separated node_type strings treated as the generated part (the rest counts as kept, e.g. the mainline fixed by tasks B/C)')
    pa.add_argument('--cost_unit_price', type=float, default=0.962, help='closed-form cost coefficient k for Σd_eu·min(la,lb)')
    pa.add_argument('--out_prefix', default=None); pa.add_argument('--seed', type=int, default=42)
    pa.add_argument('--min_ramp_nodes', type=int, default=2, help='rationality: lower bound on generated (ramp) node count; below this the sample degenerated to mainline only')
    pa.add_argument('--valid_lcc', type=float, default=0.9, help='rationality: lower bound on largest-connected-component fraction (drops fragmented graphs)')
    pa.add_argument('--valid_ramp_meanang', type=float, default=50.0, help='rationality (smoothness): upper bound on the mean internal turn angle of ramp chains (deg)')
    pa.add_argument('--valid_ramp_kinkfrac', type=float, default=0.25, help='rationality (smoothness): upper bound on the fraction of internal ramp turns >90 deg')
    pa.add_argument('--allow_isolated', action='store_true', help='rationality: allow degree-0 isolated nodes (rejected by default)')
    pa.add_argument('--no_require_ramp_on_mainline', dest='require_ramp_on_mainline', action='store_false', default=True, help='rationality: disable the "ramp must be reachable from the mainline" rule')
    args = pa.parse_args(); logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    seed_everything(args.seed); random.seed(args.seed); np.random.seed(args.seed); _load_cfg(args.predictor_cfg); auto_select_device()
    if not cfg.encoder.get('regression', True): raise ValueError('only the regression predictor is supported; a classification head would force target values onto bin representatives')
    load_bins(args.graph_data_dir) #normalization statistics of the predictor training set (de-normalizes the throughput head)
    logging.info('[Norm] Y_MEAN=%s Y_STD=%s', Y_MEAN, Y_STD)
    paths = _json_paths(args.gen_dir, args.limit)
    if not paths: raise FileNotFoundError(f'no generated JSON found under: {args.gen_dir}')
    gen_labels = set(s.strip() for s in args.generated_labels.split(',') if s.strip())
    num_generated = len(paths)

    max_nodes = cfg.dataset.get('max_nodes', 200)                                      #stages 1-2: one pass over interchange_validity, rejection reason routed to the stage it belongs to
    q_fail, r_fail = defaultdict(int), defaultdict(int); rejected, ok_paths, ok_graphs = [], [], []
    for p in paths:
        g = _read_graph(p)
        ok, why = _graph_valid(g, gen_labels, max_nodes, args.min_ramp_nodes, args.valid_lcc, args.valid_ramp_meanang, args.valid_ramp_kinkfrac,
                               allow_isolated=args.allow_isolated, require_ramp_on_mainline=args.require_ramp_on_mainline)
        if ok: ok_paths.append(p); ok_graphs.append(g); continue
        stage = 'quality_checking' if why in QUALITY_REASONS else 'rationality_checking'
        (q_fail if stage == 'quality_checking' else r_fail)[why] += 1
        rejected.append({'path': p, 'stage': stage, 'reason': why, 'target_cost': g.get('cost', ''), 'target_throughput': g.get('throughput', '')}) #target vector the sample was generated for, straight from the JSON
    n_q_fail, n_r_fail, n_pass = sum(q_fail.values()), sum(r_fail.values()), len(ok_paths)
    logging.info('[Stage1] quality checking: passed %d/%d (rejected %d: %s)', num_generated - n_q_fail, num_generated, n_q_fail, dict(q_fail) or 'none')
    logging.info('[Stage2] rationality checking: passed %d/%d (rejected %d: %s)', n_pass, num_generated - n_q_fail, n_r_fail, dict(r_fail) or 'none')
    if not ok_paths: raise RuntimeError(f'all {num_generated} generated graphs were rejected by the quality/rationality checks')

    dev = torch.device(cfg.accelerator); model = load_model(args.predictor_ckpt, dev)  #stage 3: property matching
    ctrl_scale = float(cfg.dataset.get('ctrl_scale', 1.0))
    rows = _predict(ok_paths, ok_graphs, model, dev, args.batch_size, ctrl_scale, args.cost_unit_price)
    stats = _match_stats(rows)
    logging.info('[Stage3] property matching: cost MAPE=%.1f%% (median %.1f%%) | throughput MAPE=%.1f%% (median %.1f%%)',
                 stats['cost_mape'] * 100, stats['cost_mape_median'] * 100, stats['throughput_mape'] * 100, stats['throughput_mape_median'] * 100)

    by_t = defaultdict(list)
    for r in rows: by_t[(r['target_cost'], r['target_throughput'])].append(r)
    summary = {'num_generated': num_generated,
               'quality_checking': {'num_passed': num_generated - n_q_fail, 'num_rejected': n_q_fail, 'reasons': dict(q_fail)},
               'rationality_checking': {'num_passed': n_pass, 'num_rejected': n_r_fail, 'reasons': dict(r_fail)},
               'num_feasible': n_pass, 'feasible_rate': n_pass / max(num_generated, 1),
               'property_matching': stats,
               'per_target': [{'target': [float(k[0]), float(k[1])], 'n': len(v), **_match_stats(v)} for k, v in sorted(by_t.items())],
               'settings': {'cost_unit_price': args.cost_unit_price, 'generated_labels': sorted(gen_labels), 'max_nodes': max_nodes,
                            'min_ramp_nodes': args.min_ramp_nodes, 'valid_lcc': args.valid_lcc, 'valid_ramp_meanang': args.valid_ramp_meanang,
                            'valid_ramp_kinkfrac': args.valid_ramp_kinkfrac, 'allow_isolated': args.allow_isolated,
                            'require_ramp_on_mainline': args.require_ramp_on_mainline}}
    prefix = args.out_prefix or os.path.join(args.gen_dir, '_metrics')
    with open(prefix + '.csv', 'w', newline='', encoding='utf-8') as f:                #per-sample record: rejected samples keep their stage/reason, evaluated ones carry the property errors
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore', restval=''); w.writeheader(); w.writerows(rejected + rows)
    with open(prefix + '_summary.json', 'w', encoding='utf-8') as f: json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info('[Done] csv=%s summary=%s', prefix + '.csv', prefix + '_summary.json')
    logging.shutdown(); os._exit(0)


if __name__ == '__main__':
    main()
