import logging, time #interchange unconditional generation training pipeline
import numpy as np
import torch
import torch.distributed as dist
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from lgd.asset.utils import cfg_to_dict, flatten_dict, make_wandb_name

def _amp_ctx(): #BF16 mixed-precision context (A100/H100+), halves activation memory
    use = cfg.train.get('amp', False)
    return torch.cuda.amp.autocast(enabled=use, dtype=torch.bfloat16 if use else None)

def _is_ddp(): return dist.is_initialized()
def _is_main(): return not _is_ddp() or dist.get_rank() == 0
def _world_size(): return dist.get_world_size() if _is_ddp() else 1

def _sync_grads(model): #DDP manual gradient sync: all_reduce then average
    ws = _world_size()
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(ws)


def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    if hasattr(loader, '_ddp_sampler'): #DDP: reset sampler each epoch to ensure data shuffle
        loader._ddp_sampler.set_epoch(cfg.get('_cur_epoch', 0))
    for iter, batch in enumerate(loader):
        batch.split = 'train'
        batch.to(torch.device(cfg.accelerator))
        batch.x_masked = batch.x.clone().detach()
        batch.edge_attr_masked = batch.edge_attr.clone().detach()
        graph_label = batch.y
        with _amp_ctx():
            loss, loss_task, graph_pred, loss_node, loss_edge, loss_graph, loss_encoder = model.training_step(batch)
            if cfg.diffusion.cond_stage_key != 'unconditional':
                loss = loss + loss_task * cfg.diffusion.get("task_factor", 0.0)
        loss.backward()
        is_step = ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader))
        if is_step:
            if _is_ddp():
                _sync_grads(model)
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        _true = graph_label.detach().to('cpu', non_blocking=True)
        _pred = graph_label.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true, pred=_pred, loss=loss.detach().cpu().item(),
                            loss_node=loss_node.detach().cpu().item(), loss_edge=loss_edge.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0], time_used=time.time() - time_start,
                            params=cfg.params, dataset_name=cfg.dataset.name)
        time_start = time.time()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val', repeat=1, ensemble_mode='none', evaluate=False, current_epoch=0): #evaluate/current_epoch kept for call-site compat; no val figures during training
    model.eval()
    time_start = time.time()
    max_samples = cfg.train.get('eval_max_samples', 0) #0=no limit, >0=evaluate at most N samples
    n_seen = 0
    for batch in loader:
        if max_samples > 0 and n_seen >= max_samples:
            break
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        batch.x_masked = batch.x.clone().detach()
        batch.edge_attr_masked = batch.edge_attr.clone().detach()
        ddim_steps = cfg.diffusion.get('ddim_steps', None)
        ddim_eta = cfg.diffusion.get('ddim_eta', 0.0)
        use_ddpm_steps = cfg.diffusion.get('use_ddpm_steps', False)
        with _amp_ctx():
            loss_t, graph_pred = model.inference(batch, ddim_steps=ddim_steps, ddim_eta=ddim_eta, use_ddpm_steps=use_ddpm_steps)
        n_seen += batch.num_graphs
        _true = batch.y.detach().to('cpu', non_blocking=True)
        _pred = batch.y.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true, pred=_pred, loss=loss_t.detach().cpu().item() if isinstance(loss_t, torch.Tensor) else 0.,
                            lr=0, time_used=time.time() - time_start, params=cfg.params, dataset_name=cfg.dataset.name)
        time_start = time.time()


@register_train('interchange_unconditional')
@register_train('interchange_conditional') #conditional generation reuses the same pipeline, behavior controlled by the cond_stage_key config
def custom_train_interchange(loggers, loaders, model, optimizer, scheduler):
    """Interchange diffusion generation training pipeline (unconditional/conditional), supports multi-GPU DDP"""
    is_main, is_distributed = _is_main(), _is_ddp()
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    if is_main:
        if start_epoch == cfg.optim.max_epoch:
            logging.info('Checkpoint found, Task already done')
        else:
            logging.info('Start from epoch %s', start_epoch)
    if cfg.wandb.use and is_main:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        wandb_name = cfg.wandb.name if cfg.wandb.name != '' else make_wandb_name(cfg)
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=wandb_name)
        run.config.update(cfg_to_dict(cfg))
    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times, perf = [], [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        cfg._cur_epoch = cur_epoch
        start_time = time.perf_counter()
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, cfg.optim.batch_accumulation)
        if is_main: #only rank0 writes logs/eval/saves
            perf[0].append(loggers[0].write_epoch(cur_epoch))
        else:
            perf[0].append({'loss': 0.})
        if cur_epoch > cfg.train.get('start_eval_epoch', 0):
            if is_eval_epoch(cur_epoch):
                if is_main:
                    for i in range(1, num_splits):
                        eval_epoch(loggers[i], loaders[i], model, split=split_names[i - 1], current_epoch=cur_epoch)
                        perf[i].append(loggers[i].write_epoch(cur_epoch))
                else:
                    for i in range(1, num_splits):
                        perf[i].append({'loss': 0.})
                if is_distributed:
                    dist.barrier()
            else:
                for i in range(1, num_splits):
                    perf[i].append(perf[i][-1])
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1] if perf[i] else {'loss': float('inf')})
        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1].get('loss', float('inf')))
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        if is_main and cfg.train.enable_ckpt and not cfg.train.ckpt_best and (is_ckpt_epoch(cur_epoch) or ((cur_epoch + 1) % 10 == 0)):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
            if cfg.train.ckpt_clean:
                clean_ckpt()
        if cfg.wandb.use and is_main:
            run.log(flatten_dict(perf), step=cur_epoch)
        if is_eval_epoch(cur_epoch) and is_main:
            best_epoch = np.array([vp.get('loss', float('inf')) for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                m = cfg.metric_best
                _default = float('inf') if cfg.metric_agg == 'argmin' else float('-inf')
                best_epoch = getattr(np.array([vp.get(m, _default) for vp in val_perf]), cfg.metric_agg)()
                best_train = f"train_{m}: {perf[0][best_epoch].get(m, 0):.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch].get(m, 0):.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch].get(m, 0):.4f}"
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:
                    clean_ckpt()
            logging.info(f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s (avg {np.mean(full_epoch_times):.1f}s) | "
                         f"Best: epoch {best_epoch} train_loss: {perf[0][best_epoch].get('loss',0):.4f} {best_train} "
                         f"val_loss: {perf[1][best_epoch].get('loss',0):.4f} {best_val} "
                         f"test_loss: {perf[2][best_epoch].get('loss',0):.4f} {best_test}")
    if is_main:
        logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
        logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if is_main and cfg.train.ckpt_clean:
        clean_ckpt()
    if cfg.wandb.use and is_main:
        run.finish()
    if is_main:
        logging.info('Task done, results saved in %s', cfg.run_dir)
