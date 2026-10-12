import logging, time, os #property prediction network training pipeline (graph-level classification cost_cls+throughput_cls, strictly reuses pretrain_encoder data/network/training chain)
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from lgd.asset.utils import cfg_to_dict, flatten_dict, make_wandb_name
from lgd.loader.dataset.interchange import COST_BINS, throughput_BINS, Y_MEAN, Y_STD #bins (classification)/normalization stats (regression), printed at training start
from utils import random_mask


def _fmt_bins(bins): #bin boundaries -> per-class interval string (C0:<b0 / Ci:bi-1~bi / Cn:>=b_last)
    return '  '.join([f'C0:<{bins[0]:.0f}'] + [f'C{i+1}:{bins[i]:.0f}~{bins[i+1]:.0f}' for i in range(len(bins) - 1)] + [f'C{len(bins)}:>={bins[-1]:.0f}'])


def _yn(batch, dev): #graph-level label z-score normalized tensor (mean/std from training set, persisted in bins file): returns ((y-mean)/std[B,2], mean[2], std[2])
    ym, ys = torch.tensor(Y_MEAN, device=dev), torch.tensor(Y_STD, device=dev)
    return (batch.y - ym) / ys, ym, ys


def _train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    dev = torch.device(cfg.accelerator)
    reg = cfg.encoder.get('regression', True) #True=regression (SmoothL1 on normalized values, default); False=classification (CE)
    ce_cost, ce_through, sl1 = nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.SmoothL1Loss() #classification CE / regression SmoothL1
    pc = cfg.train.pretrain.get('predict_cost', True) #whether to predict cost (cost also has a closed-form formula; if off, only trains the throughput head, default on)
    mi = 0 if pc else 1 #primary target column index (MAE reference): cost column when predicting cost, otherwise throughput column
    for iter, batch in enumerate(loader):
        batch.split = 'train'
        batch.to(dev)
        batch.x = batch.x_unified #unified feature [N,8]: xy(2)+lane_oh(3)+speed_oh(3), consistent with the unified encoder path
        batch, masked_node_idx, masked_edge_idx = random_mask(batch, cfg.train.pretrain.mask_node_prob, cfg.train.pretrain.mask_edge_prob) #keep the mask flow exactly identical to pretrain_encoder (prob=0 means no masking)
        pred = model(batch, label=None) #GraphTransformerEncoder.forward: node_emb→posenc→prefix→node_in_mlp→BN→GTE_layers→final_layer→pool→graph_attr
        node_pred, edge_pred, graph_pred = model.model.decode(pred) #regression: graph_pred=(cost_pred[B], throughput_pred[B]); classification: (cost_logits, throughput_logits)
        cp_out, tp_out = graph_pred
        if reg:
            yn, ym, ys = _yn(batch, dev)
            loss = sl1(tp_out, yn[:, 1]) + (sl1(cp_out, yn[:, 0]) if pc else 0.) #main loss: throughput(+cost) normalized regression
            _pred = ((cp_out if pc else tp_out) * ys[mi] + ym[mi]).detach().cpu().float() #denormalize to real units for logger MAE computation
            _true = batch.y[:, mi].detach().cpu().float()
        else:
            cost_label, through_label = batch.y_cost_cls.squeeze(-1), batch.y_throughput_cls.squeeze(-1)
            loss = ce_through(tp_out, through_label) + (ce_cost(cp_out, cost_label) if pc else 0.)
            _m_logits, _m_label = (cp_out, cost_label) if pc else (tp_out, through_label)
            _true, _pred = _m_label.detach().cpu().float(), _m_logits.argmax(dim=1).detach().cpu().float()
        loss = loss * cfg.train.pretrain.get('graph_factor', 1.0)
        loss.backward()
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)): #gradient accumulation + clipping exactly identical to pretrain_encoder
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        logger.update_stats(true=_true, pred=_pred, loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0], time_used=time.time() - time_start,
                            params=cfg.params, dataset_name=cfg.dataset.name)
        time_start = time.time()


@torch.no_grad()
def _eval_epoch(logger, loader, model, split='val', cur_epoch=0):
    model.eval()
    time_start = time.time()
    dev = torch.device(cfg.accelerator)
    reg = cfg.encoder.get('regression', True)
    ce_cost, ce_through, sl1 = nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.SmoothL1Loss()
    pc = cfg.train.pretrain.get('predict_cost', True) #whether to predict cost (same as _train_epoch)
    mi = 0 if pc else 1
    all_cp, all_ct, all_tp, all_tt = [], [], [], [] #(pred,true) for cost/throughput: real values for regression, class indices for classification
    for batch in loader:
        batch.split = split
        batch.to(dev)
        batch.x = batch.x_unified
        pred = model(batch, label=None)
        node_pred, edge_pred, graph_pred = model.model.decode(pred)
        cp_out, tp_out = graph_pred
        if reg:
            yn, ym, ys = _yn(batch, dev)
            loss = sl1(tp_out, yn[:, 1]) + (sl1(cp_out, yn[:, 0]) if pc else 0.)
            all_cp.append((cp_out * ys[0] + ym[0]).cpu()); all_ct.append(batch.y[:, 0].cpu()) #denormalized real values
            all_tp.append((tp_out * ys[1] + ym[1]).cpu()); all_tt.append(batch.y[:, 1].cpu())
            _pred = (all_cp[-1] if pc else all_tp[-1]).float(); _true = (all_ct[-1] if pc else all_tt[-1]).float()
        else:
            cost_label, through_label = batch.y_cost_cls.squeeze(-1), batch.y_throughput_cls.squeeze(-1)
            loss = ce_through(tp_out, through_label) + (ce_cost(cp_out, cost_label) if pc else 0.)
            all_cp.append(cp_out.argmax(1).cpu()); all_ct.append(cost_label.cpu())
            all_tp.append(tp_out.argmax(1).cpu()); all_tt.append(through_label.cpu())
            _m_logits, _m_label = (cp_out, cost_label) if pc else (tp_out, through_label)
            _true, _pred = _m_label.detach().cpu().float(), _m_logits.argmax(dim=1).detach().cpu().float()
        logger.update_stats(true=_true, pred=_pred, loss=loss.cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params, dataset_name=cfg.dataset.name)
        time_start = time.time()
    cp, ct, tp, tt = torch.cat(all_cp), torch.cat(all_ct), torch.cat(all_tp), torch.cat(all_tt)
    if reg: #real-unit MAE/RMSE/R2/MAPE (intuitive), no longer depends on binning
        def _mm(p, t): #MAE/RMSE/R2 over all samples; MAPE excludes samples with true≈0 (otherwise 1/0 blows up), returns nan when no valid samples
            m = t.abs() > 1.0
            return (p - t).abs().mean().item(), ((p - t) ** 2).mean().sqrt().item(), (1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum().clamp_min(1e-12)).item(), (((p[m] - t[m]).abs() / t[m].abs()).mean().item() * 100 if m.any() else float('nan'))
        c_mae, c_rmse, c_r2, c_mape = _mm(cp, ct); t_mae, t_rmse, t_r2, t_mape = _mm(tp, tt)
        msg = f'throughput_mae={t_mae:.0f} throughput_rmse={t_rmse:.0f} throughput_r2={t_r2:.3f} throughput_mape={t_mape:.1f}%'
        if pc: msg = f'cost_mae={c_mae:.0f} cost_rmse={c_rmse:.0f} cost_r2={c_r2:.3f} cost_mape={c_mape:.1f}% ' + msg
        logging.info(f'[PropertyPredictor] {split} epoch={cur_epoch} {msg}')
    else:
        cost_acc, through_acc = (cp == ct).float().mean().item() * 100, (tp == tt).float().mean().item() * 100
        if pc: logging.info(f'[PropertyPredictor] {split} epoch={cur_epoch} cost_acc={cost_acc:.1f}% throughput_acc={through_acc:.1f}% both_acc={((cp == ct) & (tp == tt)).float().mean().item() * 100:.1f}%')
        else: logging.info(f'[PropertyPredictor] {split} epoch={cur_epoch} throughput_acc={through_acc:.1f}% (cost prediction disabled, predict_cost=False)')


@register_train('property_predictor')
def custom_train_property_predictor(loggers, loaders, model, optimizer, scheduler): #main training loop, structurally aligned with pretrain_encoder.custom_pretrain_encoder
    pc = cfg.train.pretrain.get('predict_cost', True) #print task type and conventions at start, to verify whether they refresh with the dataset
    reg = cfg.encoder.get('regression', True)
    if not pc: logging.info('[PropertyPredictor] predict_cost=False, skipping cost prediction (throughput only)')
    if reg: logging.info(f'[PropertyPredictor] regression mode normalization stats cost(mean={Y_MEAN[0]:.0f},std={Y_STD[0]:.0f}) throughput(mean={Y_MEAN[1]:.0f},std={Y_STD[1]:.0f})')
    else:
        if pc: logging.info(f'[PropertyPredictor] classification mode cost bins ({len(COST_BINS)+1} classes): {_fmt_bins(COST_BINS)}')
        logging.info(f'[PropertyPredictor] classification mode throughput bins ({len(throughput_BINS)+1} classes): {_fmt_bins(throughput_BINS)}')
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)
    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        wandb_name = cfg.wandb.name if cfg.wandb.name != '' else make_wandb_name(cfg)
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=wandb_name)
        run.config.update(cfg_to_dict(cfg))
    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        _train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))
        torch.cuda.empty_cache()
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                _eval_epoch(loggers[i], loaders[i], model, split=split_names[i - 1], cur_epoch=cur_epoch)
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])
        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        if cfg.train.enable_ckpt and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]), cfg.metric_agg)()
                best_train = f"train_{m}: {perf[0][best_epoch].get(m, 0):.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch].get(m, 0):.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch].get(m, 0):.4f}"
                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = perf[i][best_epoch][m]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:
                    clean_ckpt()
            logging.info(f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                         f"(avg {np.mean(full_epoch_times):.1f}s) | "
                         f"Best so far: epoch {best_epoch}\t"
                         f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                         f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                         f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}")
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    if cfg.wandb.use:
        run.finish()
    logging.info('Task done, results saved in %s', cfg.run_dir)
