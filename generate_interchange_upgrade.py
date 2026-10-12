#!/usr/bin/env python
"""Task C - reference-guided existing interchange upgrading: upgrade an existing interchange (reference) while preserving the mainline and the basic topology. Like task B the mainline is anchored (masked latent replacement at every step), but instead of starting from pure noise the attribute latent is perturbed from an intermediate diffusion step (default 50%) of the reference latent and then denoised, so ramp coordinates/attributes are refined locally around the reference to match the upgraded target."""
import os, argparse, random, logging, yaml, numpy as np, torch
import lgd  # noqa
from torch_geometric.data import Batch
from torch_geometric.graphgym.config import cfg
from generate_interchange_cond import (_build_model, _load_splits, _scan_meta, _build_id2idx, _reset_load_cfg, _prepare_infer_cfg, _load_train_ystats,
                                        _fetch_data, _denorm_xy, _load_tpl_json, _dump_json,
                                        _run_stage_feature, LANE_UNI_TO_NUM, SPEED_UNI_TO_MS)


def _ref_data(anchor_gid, id2idx, splits, target, y_mean, y_std):              # reference interchange clone + target override (x_start comes from the encoder; start_t0 perturbs it); normalize with training statistics
    d = _fetch_data(anchor_gid, id2idx, splits)
    d.y = torch.tensor([target], dtype=torch.float)
    d.y_mean = torch.as_tensor(y_mean).float().reshape(1, -1); d.y_std = torch.as_tensor(y_std).float().reshape(1, -1)
    return d


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--cfg', required=True)
    pa.add_argument('--targets', default=None, help='"c1,t1;c2,t2" upgraded targets (default: read the reference cost/throughput)')
    pa.add_argument('--anchor-template', required=True, help='reference interchange gid (existing full design; mainline and basic topology preserved)')
    pa.add_argument('--perturb-frac', type=float, default=0.5, help='intermediate perturbation fraction (default 0.5 = perturb the reference latent from 50% of steps then denoise; smaller = closer to reference / larger = more freedom)')
    pa.add_argument('--k', type=int, default=None, help='samples per target (overrides cfg.total_per_target)')
    pa.add_argument('--batch-size', type=int, default=None, help='overrides cfg.batch_size (lower it on small devices)')
    pa.add_argument('--noise-temp', type=float, default=1.0, help='initial randn multiplier (>1 amplifies ramp perturbation for diversity)')
    pa.add_argument('--out', default=None)
    a = pa.parse_args()
    with open(a.cfg, 'r', encoding='utf-8') as f: gc = yaml.safe_load(f)['generate']
    if a.targets: gc['targets'] = [[float(v) for v in p.split(',')] for p in a.targets.split(';')]
    if a.batch_size is not None: gc['batch_size'] = int(a.batch_size)
    if a.out: gc['out_dir'] = a.out
    random.seed(gc['seed']); np.random.seed(gc['seed']); torch.manual_seed(gc['seed'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if gc.get('ddim_steps') is not None: logging.warning(f'start_t0 truncated sampling is incompatible with DDIM, forcing ddim_steps=null (was {gc.get("ddim_steps")})'); gc['ddim_steps'] = None
    os.makedirs(gc['out_dir'], exist_ok=True)

    ids_all, ys_all, Ns_all = _scan_meta(gc['meta_cache'], gc['graph_data_dir'])   # reference interchange source (four-leg subset)
    _reset_load_cfg(gc['attribute_cfg']); _prepare_infer_cfg(gc['graph_data_dir'])
    max_n = cfg.dataset.get('max_nodes', 200); ctrl_scale = float(cfg.dataset.get('ctrl_scale', 1.0))
    T = int(cfg.diffusion.get('timesteps', 1000)); start_t0 = max(1, min(int(a.perturb_frac * T), T - 1))
    splits = _load_splits(gc['dataset_root'])
    id2idx = _build_id2idx(ids_all, Ns_all, max_n, seed=gc['seed'])
    y_mean, y_std = _load_train_ystats(gc.get('train_stats_dir', gc['graph_data_dir']))  # normalization from training statistics
    diag = float(gc['global_diag']); denorm_on = bool(gc.get('denorm', True))
    anc_gid = str(a.anchor_template)
    if anc_gid not in id2idx: raise SystemExit(f'anchor_gid={anc_gid} not in id2idx (filtered by max_nodes or missing)')
    anc_tpl = _load_tpl_json(anc_gid, gc['graph_data_dir'])
    targets = gc['targets'] if a.targets else [[float(anc_tpl.get('cost', 0.0)), float(anc_tpl.get('throughput', 0.0))]]  # default = reference's own properties
    K = int(a.k) if a.k is not None else int(gc.get('total_per_target', 1))
    logging.info(f'[Upgrade] anchor={anc_gid} targets={targets} K={K} start_t0={start_t0}/{T}({a.perturb_frac:.0%})')

    mf = _build_model(gc['attribute_cfg'], gc['attribute_ckpt']); device = torch.device(cfg.accelerator)  # attribute stage: mainline anchored (RePaint) + ramp perturbed from the reference latent
    ctrl_scale = float(cfg.dataset.get('ctrl_scale', ctrl_scale))
    tasks = [(ci, si, list(tg)) for ci, tg in enumerate(targets) for si in range(K)]
    skels = [_ref_data(anc_gid, id2idx, splits, tg, y_mean, y_std) for (_, _, tg) in tasks]
    feat = _run_stage_feature(mf, skels, device, gc['batch_size'], gc.get('ddim_steps'), gc.get('ddim_eta', 0.0), gc.get('use_ddpm_steps', False), lock_mainline=True, start_t0=start_t0, noise_temp=float(a.noise_temp))
    del mf; torch.cuda.empty_cache()

    summary_p = os.path.join(gc['out_dir'], '_summary.csv')
    with open(summary_p, 'w', encoding='utf-8') as fsum:
        fsum.write('case,sample,target_cost,target_throughput,anchor_gid,N,lanes_mean,speed_mean_ms\n')
        for (ci, si, tg), (xy, adj, lu, su, ntype) in zip(tasks, feat):
            num_lanes = np.array([LANE_UNI_TO_NUM(c) for c in lu]); speed_ms = np.array([SPEED_UNI_TO_MS(c) for c in su])
            pos_m = _denorm_xy(xy, ctrl_scale, diag)
            sub = os.path.join(gc['out_dir'], f'anc{anc_gid}_c{int(tg[0])}_t{int(tg[1])}'); os.makedirs(sub, exist_ok=True)
            base = os.path.join(sub, f'{si:04d}')
            _dump_json(base + '.json', anc_tpl, pos_m, xy.numpy() / ctrl_scale, adj, num_lanes, speed_ms, tg, gc['z_fill'], diag, denorm_on, gen_ntype=ntype.astype(np.int64))
            fsum.write(f'{ci},{si},{tg[0]:.2f},{tg[1]:.2f},{anc_gid},{xy.shape[0]},{float(num_lanes.mean()):.2f},{float(speed_ms.mean()):.2f}\n')
    logging.info(f'[Done] {len(feat)} samples -> {gc["out_dir"]} (summary: {summary_p})')


if __name__ == '__main__':
    main()
