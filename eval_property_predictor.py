"""Property predictor evaluation/inference script: load ckpt, randomly sample from dataset, predict cost/throughput classes and report acc/confusion matrix"""
import lgd  # noqa, register custom modules
import argparse, logging, os, random
import torch
import numpy as np
from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.register import network_dict
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric import seed_everything
from lgd.loader.dataset.interchange import COST_BINS, throughput_BINS, Y_MEAN, Y_STD, load_bins

def _bin_labels(bins, prec): #build interval display labels from current (adaptive) bin edges: <b0 / b0-b1 / ... / >=b_last
    return [f'<{bins[0]/1000:.{prec}f}k'] + [f'{bins[i]/1000:.{prec}f}-{bins[i+1]/1000:.{prec}f}k' for i in range(len(bins)-1)] + [f'≥{bins[-1]/1000:.{prec}f}k']


def load_model(ckpt_path, dev): #load PropertyPredictor: ckpt keys carry a model. prefix (GraphGymModule_custom wrapper), strip before loading
    model_cls = network_dict[cfg.model.type]
    model = model_cls(dim_in=cfg.share.dim_in, dim_out=cfg.share.dim_out, cfg=cfg.encoder)
    ckpt = torch.load(ckpt_path, map_location=dev)
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    stripped = {k.replace('model.', '', 1) if k.startswith('model.') else k: v for k, v in state.items()} #strip GraphGymModule_custom's model. prefix
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing: logging.warning(f'missing keys ({len(missing)}): {missing[:5]}...')
    if unexpected: logging.warning(f'unexpected keys ({len(unexpected)}): {unexpected[:5]}...')
    model.to(dev).eval()
    return model


def print_confusion_matrix(cm, labels, title): #print confusion matrix (row=GT, col=Pred)
    n = len(labels)
    max_lw = max(len(l) for l in labels)
    header = ' ' * (max_lw + 2) + ''.join(f'{l:>{max_lw+1}}' for l in labels)
    print(f'\n{title}')
    print(header)
    for i in range(n):
        row = ''.join(f'{cm[i][j]:>{max_lw+1}}' for j in range(n))
        print(f'{labels[i]:>{max_lw}}  {row}')


@torch.no_grad()
def _eval_regression(model, loader, dev, sample_ratio, predict_cost): #regression eval: de-normalize to real units, report MAE/RMSE/R2/MAPE over every sample
    ym, ys = torch.tensor(Y_MEAN), torch.tensor(Y_STD)
    cp, ct, tp, tt = [], [], [], []
    for batch in loader:
        batch.to(dev); batch.x = batch.x_unified
        c_out, t_out = model.decode(model.forward(batch))[2]
        cp.append((c_out.cpu() * ys[0] + ym[0])); ct.append(batch.y[:, 0].cpu())
        tp.append((t_out.cpu() * ys[1] + ym[1])); tt.append(batch.y[:, 1].cpu())
    cp, ct, tp, tt = torch.cat(cp), torch.cat(ct), torch.cat(tp), torch.cat(tt)
    n_total = len(cp)
    if sample_ratio < 1.0:
        idx = torch.tensor(random.sample(range(n_total), max(1, int(n_total * sample_ratio)))); cp, ct, tp, tt = cp[idx], ct[idx], tp[idx], tt[idx]; n_total = len(cp)
    def _stat(p, t): #real units: MAE/RMSE/R2 on all samples; MAPE (mean/median/p90) excludes samples with true value ~0 (else 1/0 blows up)
        ae = (p - t).abs(); rmse = ((p - t) ** 2).mean().sqrt().item(); r2 = (1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum().clamp_min(1e-12)).item(); m = t.abs() > 1.0
        mp = ((ae[m] / t[m].abs() * 100).mean().item(), (ae[m] / t[m].abs() * 100).median().item(), np.percentile((ae[m] / t[m].abs() * 100).numpy(), 90)) if bool(m.any()) else (float('nan'),) * 3 #set MAPE to nan when no valid sample (all true values ~0)
        return (ae.mean().item(), rmse, r2) + mp
    print(f'\n{"="*72}\n  samples: {n_total}  (sample ratio: {sample_ratio*100:.0f}%, regression real units)')
    out = {}
    for name, p, t, on in [('cost', cp, ct, predict_cost), ('throughput', tp, tt, True)]:
        if not on: print(f'  (cost prediction disabled, predict_cost=False)'); continue
        mae, rmse, r2, mape, med, p90 = _stat(p, t); out.update({f'{name}_mae': mae, f'{name}_rmse': rmse, f'{name}_r2': r2, f'{name}_mape': mape})
        print(f'  {name:>10s}:  MAE={mae:8.0f}  RMSE={rmse:8.0f}  R2={r2:6.3f}  MAPE={mape:5.1f}%  MAPE_median={med:5.1f}%  MAPE_p90={p90:5.1f}%')
    print(f'{"="*72}')
    return out


@torch.no_grad()
def evaluate(model, loader, dev, sample_ratio=1.0, predict_cost=True): #regression (default) reports real-unit errors; classification reports acc/confusion matrix
    if cfg.encoder.get('regression', True):
        return _eval_regression(model, loader, dev, sample_ratio, predict_cost)
    cost_labels, through_labels = _bin_labels(COST_BINS, 1), _bin_labels(throughput_BINS, 2) #build labels from the currently active bins
    n_cost_cls, n_through_cls = len(COST_BINS) + 1, len(throughput_BINS) + 1
    all_cost_pred, all_cost_true, all_through_pred, all_through_true = [], [], [], []
    for batch in loader:
        batch.to(dev)
        batch.x = batch.x_unified
        pred = model.forward(batch)
        _, _, graph_pred = model.decode(pred)
        cost_logits, through_logits = graph_pred
        all_cost_pred.append(cost_logits.argmax(dim=1).cpu())
        all_cost_true.append(batch.y_cost_cls.squeeze(-1).cpu())
        all_through_pred.append(through_logits.argmax(dim=1).cpu())
        all_through_true.append(batch.y_throughput_cls.squeeze(-1).cpu())
    cp, ct = torch.cat(all_cost_pred), torch.cat(all_cost_true)
    tp, tt = torch.cat(all_through_pred), torch.cat(all_through_true)
    n_total = len(cp)
    if sample_ratio < 1.0: #random sampling
        n_sample = max(1, int(n_total * sample_ratio))
        idx = torch.tensor(random.sample(range(n_total), n_sample))
        cp, ct, tp, tt = cp[idx], ct[idx], tp[idx], tt[idx]
        n_total = n_sample
    cost_acc = (cp == ct).float().mean().item() * 100
    through_acc = (tp == tt).float().mean().item() * 100
    both_acc = ((cp == ct) & (tp == tt)).float().mean().item() * 100
    print(f'\n{"="*60}')
    print(f'  samples: {n_total}  (sample ratio: {sample_ratio*100:.0f}%)')
    if predict_cost:
        print(f'  cost accuracy:       {cost_acc:.1f}%')
    print(f'  throughput accuracy: {through_acc:.1f}%')
    if predict_cost:
        print(f'  both targets correct: {both_acc:.1f}%')
    else:
        print(f'  (cost prediction disabled, predict_cost=False)')
    print(f'{"="*60}')
    targets = ([('cost', cp, ct, cost_labels, n_cost_cls)] if predict_cost else []) + [('throughput', tp, tt, through_labels, n_through_cls)]
    for name, pred, true, labels, n_cls in targets: #detailed per-target report
        cm = torch.zeros(n_cls, n_cls, dtype=torch.long)
        for i in range(len(pred)): cm[true[i], pred[i]] += 1
        per_cls_acc = [(cm[i, i].item() / max(cm[i].sum().item(), 1)) * 100 for i in range(n_cls)]
        per_cls_n = [cm[i].sum().item() for i in range(n_cls)]
        print(f'\n--- {name} per-class accuracy ---')
        for i in range(n_cls):
            print(f'  class {i} ({labels[i]:>16s}): {per_cls_acc[i]:5.1f}%  (N={per_cls_n[i]})')
        print_confusion_matrix(cm.tolist(), [f'C{i}' for i in range(n_cls)], f'{name} confusion matrix (row=GT, col=Pred)')
    return {'cost_acc': cost_acc, 'throughput_acc': through_acc, 'both_acc': both_acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Property predictor evaluation/inference')
    parser.add_argument('--cfg', type=str, default='cfg/interchange-property-predictor.yaml', help='config file path')
    parser.add_argument('--ckpt', type=str, default=None, help='ckpt path (default: the highest-numbered epoch under cfg.run_dir/ckpt)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test', 'all'], help='dataset split to evaluate (all=run train/val/test in one go)')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='random sample ratio (0~1; 1=evaluate the whole split)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    cfg.set_new_allowed(True)
    set_cfg(cfg)
    cfg.set_new_allowed(True)
    class FakeArgs: #match the args format required by load_cfg
        cfg_file = args.cfg
        opts = []
    load_cfg(cfg, FakeArgs())
    cfg.out_dir = os.path.join(cfg.out_dir, os.path.splitext(os.path.basename(args.cfg))[0])
    cfg.run_dir = os.path.join(cfg.out_dir, '0')
    seed_everything(args.seed)
    random.seed(args.seed)
    auto_select_device()
    dev = torch.device(cfg.accelerator)
    logging.basicConfig(level=logging.INFO)
    logging.info(f'loading dataset...')
    loaders = create_loader()
    load_bins(cfg.dataset.get('graph_data_dir', 'graph_data')) #refresh to this dataset's adaptive bins (consistent with cached y_*_cls)
    split_map = {'train': 0, 'val': 1, 'test': 2}
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split] #all=evaluate the three splits in turn
    ckpt_path = args.ckpt
    if ckpt_path is None: #no ckpt given: take the latest epoch on disk (pass --ckpt explicitly to evaluate the best-val one)
        ckpt_dir = os.path.join(cfg.run_dir, 'ckpt')
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        if not ckpts:
            raise FileNotFoundError(f'ckpt not found: {ckpt_dir}')
        ckpt_path = os.path.join(ckpt_dir, sorted(ckpts, key=lambda x: int(x.split('.')[0]))[-1]) #default to the latest
        logging.info(f'auto-selected ckpt: {ckpt_path}')
    else:
        logging.info(f'using specified ckpt: {ckpt_path}')
    model = load_model(ckpt_path, dev)
    logging.info(f'model loaded, param count: {sum(p.numel() for p in model.parameters())}')
    pc = cfg.train.pretrain.get('predict_cost', True) #read the predict_cost switch; if off, do not report cost
    if cfg.encoder.get('regression', True): logging.info(f'evaluating splits={splits}, sample_ratio={args.sample_ratio}, predict_cost={pc}, regression mode Y_MEAN={Y_MEAN}, Y_STD={Y_STD}')
    else: logging.info(f'evaluating splits={splits}, sample_ratio={args.sample_ratio}, predict_cost={pc}, classification mode COST_BINS={COST_BINS}, throughput_BINS={throughput_BINS}')
    for sp in splits: #evaluate per split (with --split all, output train/val/test in turn)
        print(f'\n{"#"*72}\n#  split = {sp}\n{"#"*72}'); evaluate(model, loaders[split_map[sp]], dev, sample_ratio=args.sample_ratio, predict_cost=pc)
