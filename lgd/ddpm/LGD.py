import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
# from pytorch_lightning.utilities.distributed import rank_zero_only
from torch_geometric.graphgym.config import cfg
import logging
from torch_geometric.graphgym.loss import compute_loss

from lgd.ddpm.ema import LitEma
from lgd.ddpm.distributions import normal_kl, DiagonalGaussianDistribution
from lgd.model.utils import make_beta_schedule, extract_into_tensor, extract_into_sparse_tensor, noise_like, num2batch, \
    exists, default, ismap, mean_flat, count_params, instantiate_from_config, symmetrize
from lgd.ddpm.ddim import DDIMSampler
from lgd.model.Dictionary import *
from lgd.model.GraphTransformerEncoder import *
from lgd.model.DenoisingTransformer import DenoisingTransformer


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 gt_config=None,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=False,   # originally True
                 first_stage_key=None,  # TODO: need to change this?
                 image_size=256,
                 hid_dim=3,
                 log_every_t=100,
                 clip_denoised=False,  # originally True
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="x0",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization  # TODO: now we only support "x0", i.e. output graph without noise
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.hid_dim = hid_dim
        self.force_undirected = cfg.diffusion.get('force_undirected', False)
        self.adj_cond = cfg.diffusion.get('adj_cond', False) #adjacency matrix as a known condition rather than a generation target
        self.adj_only = cfg.diffusion.get('adj_only', False) #diffuse the adjacency matrix only, node features not involved
        self.geo_cond = cfg.diffusion.get('geo_cond', False) #geo latent as condition, diffuse attr latent (dual-latent sequential mode)
        self.fc_cond = cfg.diffusion.get('fc_cond', False) #fully-connected condition: same logic as adj_cond but edge_attr all ones (topology not encoded)
        self.affinity_mode = cfg.diffusion.get('affinity_mode', False) #affinity diffusion: diffuse N×K affinity latent, geo latent condition
        self.unified_cond = cfg.diffusion.get('unified_cond', False) #unified encoding: xy+lanes+speed merged and VAE-compressed, then latent diffusion, adjacency condition
        self.segment_mode = cfg.diffusion.get('segment_mode', False) #segment refined-graph mode (works with adj_only): diffuse over M×M adjacency after chain folding
        assert sum([self.adj_cond, self.adj_only, self.geo_cond, self.fc_cond, self.affinity_mode, self.unified_cond]) <= 1, 'modes mutually exclusive'
        assert (not self.segment_mode) or sum([self.adj_cond, self.adj_only, self.geo_cond, self.fc_cond, self.affinity_mode, self.unified_cond]) == 0, 'segment_mode uses the joint W+L mode (node=segment length+type, edge=segment connections); other mode flags must be disabled' #segment mode must be enabled together with adj_only
        if self.adj_only:
            self.node_type_embed = nn.Embedding(cfg.diffusion.get('node_type_num', 2), hid_dim)
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(gt_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        # if self.learn_logvar:
        self.logvar = nn.Parameter(self.logvar, requires_grad=self.learn_logvar)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t, batch_idx):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_sparse_tensor(self.sqrt_alphas_cumprod, t, batch_idx) * x_start)
        variance = extract_into_sparse_tensor(1.0 - self.alphas_cumprod, t, batch_idx)
        log_variance = extract_into_sparse_tensor(self.log_one_minus_alphas_cumprod, t, batch_idx)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise, batch_idx):
        return (
                extract_into_sparse_tensor(self.sqrt_recip_alphas_cumprod, t, batch_idx) * x_t -
                extract_into_sparse_tensor(self.sqrt_recipm1_alphas_cumprod, t, batch_idx) * noise
        )

    def q_posterior(self, x_start, x_t, t, batch_idx):
        posterior_mean = (
                extract_into_sparse_tensor(self.posterior_mean_coef1, t, batch_idx) * x_start +
                extract_into_sparse_tensor(self.posterior_mean_coef2, t, batch_idx) * x_t
        )
        posterior_variance = extract_into_sparse_tensor(self.posterior_variance, t, batch_idx)
        posterior_log_variance_clipped = extract_into_sparse_tensor(self.posterior_log_variance_clipped, t, batch_idx)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, batch_idx, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out, batch_idx=batch_idx)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t, batch_idx=batch_idx)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, batch_idx, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, batch_idx=batch_idx)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, batch_idx, return_intermediates=False):
        device = self.betas.device
        b = int(torch.max(batch_idx)) + 1
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), batch_idx=batch_idx,
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_idx, batch_size=16, return_intermediates=False):
        hid_dim = self.hid_dim
        return self.p_sample_loop((batch_idx.shape[0], hid_dim), batch_idx=batch_idx,
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, batch_idx=None, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if (t >= self.num_timesteps).any():
            return noise
        if batch_idx is None:
            return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        else:
            return (extract_into_sparse_tensor(self.sqrt_alphas_cumprod, t, batch_idx) * x_start +
                extract_into_sparse_tensor(self.sqrt_one_minus_alphas_cumprod, t, batch_idx) * noise)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, batch, t, noise=None, task_id=None):
        batch_idx = batch.batch_idx
        x_start = torch.cat([batch.x, batch.edge_attr], dim=0)
        # TODO: this class does not support conditional generation and masked graph,
        #  so x_start should not be masked
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, batch_idx=batch_idx, noise=noise)
        batch.x = x_noisy[:batch.num_nodes]
        batch.edge_attr = x_noisy[batch.num_nodes:]
        model_out = self.model(batch, t, task_id)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, batch, task_id=None, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (batch.num_graphs,), device=self.device).long()
        return self.p_losses(batch, t, task_id=task_id, *args, **kwargs)

    # def get_input(self, batch, k):
    #     x = batch[k]
    #     if len(x.shape) == 3:
    #         x = x[..., None]
    #     x = rearrange(x, 'b h w c -> b c h w')
    #     x = x.to(memory_format=torch.contiguous_format).float()
    #     return x

    def get_input(self, batch):
        num_node = batch.num_node_per_graph
        if self.adj_only:
            batch_idx = num2batch(num_node ** 2)
        elif self.adj_cond or self.geo_cond or self.fc_cond or self.affinity_mode or self.unified_cond: #node-level diffusion mode
            batch_idx = batch.batch
        else:
            batch_idx = torch.cat([batch.batch, num2batch(num_node ** 2)], dim=0)
        batch.batch_idx = batch_idx
        batch.batch_size = batch.num_graphs
        return batch

    def shared_step(self, batch, task_id=None):
        # x = self.get_input(batch, self.first_stage_key)
        batch = self.get_input(batch)
        if task_id is not None:
            batch.task_id = task_id
        loss, loss_dict = self(batch, task_id)
        return loss, loss_dict

    def training_step(self, batch, task_id=None):
        loss, loss_dict = self.shared_step(batch, task_id)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, task_id=None):
        _, loss_dict_no_ema = self.shared_step(batch, task_id)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch, task_id)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,  # actually the ckpt path
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="unconditional",
                 first_stage_trainable=False,
                 cond_stage_trainable=False,
                 concat_mode=False,  # default change to cross attention
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 node_factor=1.0,
                 edge_factor=1.0,
                 recon_factor=0.0,
                 graph_factor=1.0,
                 scale_by_std=False,
                 train_mode='sample',
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        # assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        # assert cond_stage_config in cond_stage_config_list
        assert cond_stage_key in cond_stage_key_list
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        assert conditioning_key in conditioning_key_list
        self.cond_stage_key = cond_stage_key
        if self.cond_stage_key in ['masked_graph', 'prompt_graph', 'pe']:
            cond_stage_config = "__is_first_stage__"  # TODO: better to do this
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.first_stage_trainable = first_stage_trainable
        self.cond_stage_trainable = cond_stage_trainable
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.node_factor = cfg.diffusion.get("node_factor", node_factor)
        self.edge_factor = cfg.diffusion.get("edge_factor", edge_factor)
        self.graph_factor = cfg.diffusion.get("graph_factor", graph_factor)
        self.recon_factor = cfg.diffusion.get("recon_factor", recon_factor)
        assert train_mode in ['sample', 'half-sample', 'complete-noise']
        self.train_mode = train_mode
        if self.unified_cond: #unified: structured-latent scale normalization buffers (following GeoLDM z_x_scale/z_h_scale)
            self._coord_dim = cfg.encoder.get('coord_dim', 24) #matches the encoder's coord_dim
            self.register_buffer('z_coord_scale', torch.ones(1)) #coordinate latent normalization
            self.register_buffer('z_attr_scale', torch.ones(1)) #attribute latent normalization
            self.register_buffer('z_edge_scale', torch.ones(1)) #edge latent normalization
            self._coord_weight = cfg.diffusion.get('coord_weight', 3.0) #weight of coordinate dims in the diffusion loss (following GeoLDM)

        if self.geo_cond or self.affinity_mode: #geo_cond/affinity: load the frozen geo encoder to provide the geo_latent condition
            geo_cfg_path = cfg.diffusion.get('geo_stage_config', None)
            if geo_cfg_path:
                self.instantiate_geo_stage(geo_cfg_path)
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    # @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            batch = super().get_input(batch)  # TODO: check this
            batch = batch.to(self.device)
            encoder_posterior = self.encode_first_stage(batch)  # TODO: implement encoder
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = eval(cfg.encoder.get("model_type", "GraphTransformerEncoder"))(cfg=cfg.encoder)
        self.first_stage_model = model
        sd = torch.load(config, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        if "model_state" in list(sd.keys()):
            sd = sd["model_state"]

        state_dict = {}
        for k, v in sd.items():
            # new_k = k.replace('model.', '') if 'model' in k else k
            new_k = k[6:] if k.startswith('model.') else k
            state_dict[new_k] = v
        missing, unexpected = self.first_stage_model.load_state_dict(state_dict, strict=False)
        logging.info(f"Restored from {config} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            logging.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            logging.info(f"Unexpected Keys: {unexpected}")
        if not self.first_stage_trainable:
            self.first_stage_model = model.eval()
            self.first_stage_model.train = disabled_train
            for param in self.first_stage_model.parameters():
                param.requires_grad = False

        # model = instantiate_from_config(config)
        # if not self.first_stage_trainable:
        #     self.first_stage_model = model.eval()
        #     self.first_stage_model.train = disabled_train
        #     for param in self.first_stage_model.parameters():
        #         param.requires_grad = False
        # else:
        #     self.first_stage_model = model

    def instantiate_geo_stage(self, config): #load the frozen geo encoder (GraphTransformerEncoder) to provide the latent_geo condition in geo_cond mode
        from torch_geometric.graphgym.register import network_dict as register_net
        geo_model_type = cfg.diffusion.get('geo_model_type', 'GraphTransformerEncoder')
        geo_enc_cfg = copy.deepcopy(cfg.encoder)
        geo_enc_cfg.model_type = geo_model_type
        geo_enc_cfg.node_encoder_name = 'InterchangeNode'
        geo_enc_cfg.node_encoder_num_types = 2 #geo encoder uses xy 2-dim only
        model = register_net[geo_model_type](cfg=geo_enc_cfg)
        sd = torch.load(config, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        if "model_state" in sd: sd = sd["model_state"]
        state_dict = {(k[6:] if k.startswith('model.') else k): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logging.info(f"[GeoStage] Restored from {config} with {len(missing)} missing / {len(unexpected)} unexpected")
        self.geo_stage_model = model.eval()
        self.geo_stage_model.train = disabled_train
        for p in self.geo_stage_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_geo_stage(self, batch, **kwargs): #encode xy with the frozen geo encoder → latent_geo + edge_latent
        return self.geo_stage_model.encode(batch, **kwargs)

    def _seg_meta(self, batch, i): #segment mode: extract raw_pos/raw_adj/seg_chains of the i-th graph for visualization (returned on CPU)
        if not (self.segment_mode and hasattr(batch, 'raw_num_node_per_graph')): return None
        rn_list = batch.raw_num_node_per_graph.detach().cpu().tolist() #raw node count per graph
        rn = int(rn_list[i]); rn_off_n = sum(rn_list[:i]); rn_off_e = sum(r * r for r in rn_list[:i]) #node/edge offsets
        raw_pos = batch.raw_pos_orig[rn_off_n: rn_off_n + rn].detach().cpu()
        raw_adj = batch.raw_adj_flat[rn_off_e: rn_off_e + rn * rn].reshape(rn, rn).long().detach().cpu()
        meta = dict(is_segment=True, raw_pos=raw_pos, raw_adj=raw_adj)
        if hasattr(batch, 'seg_chain_lens'): #chain lengths + flattened node ids, for reconstructing the original graph or chain visualization
            m_list = batch.num_node_per_graph.detach().cpu().tolist()
            m_off = sum(m_list[:i]); m = int(m_list[i])
            cl = batch.seg_chain_lens[m_off: m_off + m].detach().cpu()
            cf_off = int(batch.seg_chain_lens[:m_off].sum().item()) if m_off > 0 else 0 #total chain length of the first m_off segments, used as chain_flat offset
            cf = batch.seg_chain_flat[cf_off: cf_off + int(cl.sum().item())].detach().cpu()
            meta['seg_chain_lens'] = cl; meta['seg_chain_flat'] = cf
        return meta

    def _seg_decode(self, batch, i, node_decode, edge_decode, an, ae, m): #segment joint decode: node→(segment length L, type), edge→segment connections W, uncoarsen→full adjacency Â into meta
        from lgd.transform.segment_graph import split_seg_x, uncoarsen_segment_graph
        seg_x = node_decode[an: an + m].detach().cpu() #[M,1+C] segment features (length + type)
        W = torch.argmax(edge_decode[ae: ae + m * m], dim=1).reshape(m, m).detach().cpu() #segment connections W
        W_gt = batch.adj_flat[ae: ae + m * m].reshape(m, m).long().detach().cpu()
        lens, typ = split_seg_x(seg_x)
        full_adj, full_nt = (None, None) if self.training else uncoarsen_segment_graph(W, lens, typ) #training steps don't need the decoded result, skip networkx uncoarsening to save time; only eval/inference uncoarsens to node-level full adjacency
        meta = self._seg_meta(batch, i) or {}
        meta.update(lens_pred=lens, type_pred=typ, full_adj=full_adj, full_node_type=full_nt)
        return (batch.pos_orig[an: an + m].detach().cpu(), W, W_gt, meta) #(segment midpoint coordinates, W prediction, W ground truth, meta)

    def instantiate_cond_stage(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__":
            print(f"Training {self.__class__.__name__} as an unconditional model.")
            self.cond_stage_model = None
            # self.be_unconditional = True
        else:
            # TODO: for other conditions, e.g. multi-model and label, specialize the cond_stage_model
            # model = instantiate_from_config(config)
            model = eval(cfg.cond.get("model_type", "GraphTransformerEncoder"))(cfg=cfg.cond)
            self.cond_stage_model = model
            sd = torch.load(config, map_location="cpu")
            if "state_dict" in list(sd.keys()):
                sd = sd["state_dict"]
            if "model_state" in list(sd.keys()):
                sd = sd["model_state"]

            state_dict = {}
            for k, v in sd.items():
                new_k = k.replace('model.', '') if 'model' in k else k
                state_dict[new_k] = v
            missing, unexpected = self.cond_stage_model.load_state_dict(state_dict, strict=False)
            # keys = list(sd.keys())
            # missing, unexpected = self.cond_stage_model.load_state_dict(sd, strict=False)
            print(f"Restored from {config} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if len(missing) > 0:
                print(f"Missing Keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected Keys: {unexpected}")
        if not self.cond_stage_trainable and self.cond_stage_model is not None:
            self.cond_stage_model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
        # else:
        #     assert config != '__is_first_stage__'  # TODO: may not be necessary
        #     assert config != '__is_unconditional__'
        #     # model = instantiate_from_config(config)
        #     model = DenoisingTransformer(cfg=cfg)
        #     self.cond_stage_model = model
        #     sd = torch.load(config, map_location="cpu")
        #     if "state_dict" in list(sd.keys()):
        #         sd = sd["state_dict"]
        #     keys = list(sd.keys())
        #     missing, unexpected = self.cond_stage_model.load_state_dict(sd, strict=False)
        #     print(f"Restored from {config} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        #     if len(missing) > 0:
        #         print(f"Missing Keys: {missing}")
        #     if len(unexpected) > 0:
        #         print(f"Unexpected Keys: {unexpected}")

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()  # TODO: what is the role of DiagonalGaussian? this sample needs batch_idx?
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    @torch.no_grad()
    def get_input(self, batch, prompt_graph_batch=None, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False):
        batch = super().get_input(batch)
        batch = batch.to(self.device)
        batch.x_0 = batch.x.clone().detach()
        batch.edge_attr_0 = batch.edge_attr.clone().detach()

        input_label = batch.y.clone().detach() if cfg.train.pretrain.input_target else None
        batch_z = copy.deepcopy(batch)
        if self.unified_cond and hasattr(batch, 'x_unified'): #unified: xy+lane_oh+speed_oh unified encoding
            batch_z.x = batch.x_unified.clone().detach()
        elif self.geo_cond and hasattr(batch, 'x_attr'): #geo_cond: attr encoder needs x_attr as input
            batch_z.x = batch.x_attr.clone().detach()
        elif self.affinity_mode and hasattr(batch, 'x_affinity'): #affinity: affinity matrix as encoder input
            batch_z.x = batch.x_affinity.clone().detach()
        if batch.get("prefix", None) is not None:
            batch_z.prefix = batch.prefix
        batch_z = self.encode_first_stage(batch_z, label=input_label)  # original space stored in x_0, edge_attr_0
        # batch_z = self.get_first_stage_encoding(encoder_posterior)
        if not self.first_stage_trainable:
            batch_z.x = batch_z.x.detach()
            batch_z.edge_attr = batch_z.edge_attr.detach()
            batch_z.graph_attr = batch_z.graph_attr.detach()
        if self.unified_cond and self.training and self.z_coord_scale.item() == 1.0: #calibrate structured-latent scales on the first batch → each part std≈1
            cd = self._coord_dim
            self.z_coord_scale.fill_(batch_z.x[:, :cd].flatten().std().clamp(min=1e-6))
            self.z_attr_scale.fill_(batch_z.x[:, cd:].flatten().std().clamp(min=1e-6))
            self.z_edge_scale.fill_(batch_z.edge_attr.flatten().std().clamp(min=1e-6))
            logging.info(f'[UnifiedCond] z_coord_scale={self.z_coord_scale.item():.4f} z_attr_scale={self.z_attr_scale.item():.4f} z_edge_scale={self.z_edge_scale.item():.4f}')

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key == 'masked_graph':
                batch_masked = copy.deepcopy(batch)
                batch_masked.x = batch.x_masked.clone().detach()  # TODO (done): operate masking in main function
                batch_masked.edge_attr = batch.edge_attr_masked.clone().detach()
                if batch.get("prefix", None) is not None:
                    batch_masked.prefix = batch.prefix
                batch_masked = self.get_learned_conditioning(batch_masked)
                if not self.first_stage_trainable or not self.cond_stage_trainable:
                    batch_masked.x = batch_masked.x.detach()
                    batch_masked.edge_attr = batch_masked.edge_attr.detach()
                    batch_masked.graph_attr = batch_masked.graph_attr.detach()
                xc = (batch.x_masked, batch.edge_attr_masked)
                c = (batch_masked.x, batch_masked.edge_attr, batch_masked.graph_attr)
            elif cond_key == 'unconditional':
                xc = None
                c = None
            elif cond_key == 'pe':  # undone
                pos_node = batch.get('pestat_node', torch.zeros([batch.num_nodes, self.model.diffusion_model.cond_dim], dtype=torch.float, device=batch.x.device))
                pos_edge = batch.get('pestat_edge', torch.zeros([batch.edge_index.shape[1], self.model.diffusion_model.cond_dim], dtype=torch.float, device=batch.x.device))
                pos_graph = batch.get('pestat_graph', torch.zeros([batch.num_graphs, self.model.diffusion_model.cond_dim], dtype=torch.float, device=batch.x.device))
                if not hasattr(batch, 'pestat_node'):
                    logging.info('No pe was found for node')
                if not hasattr(batch, 'pestat_edge'):
                    logging.info('No pe was found for edge')
                xc = (pos_node, pos_edge, pos_graph)
                c = (pos_node, pos_edge, pos_graph)
            elif cond_key == 'prompt_graph':
                assert prompt_graph_batch is not None
                prompt_graph_batch.task_id = batch.get('task_id', None)
                xc = (prompt_graph_batch.x, prompt_graph_batch.edge_attr)
                c = self.get_learned_conditioning(prompt_graph_batch).graph_attr  # TODO: here use graph_attr as prompt
                if not self.first_stage_trainable or not self.cond_stage_trainable:
                    c = c.detach()
            elif cond_key == 'class_label':
                xc = batch.y
                if hasattr(batch, 'y_mean') and hasattr(batch, 'y_std'): #Interchange: standardize graph-level labels
                    xc = (xc - batch.y_mean) / batch.y_std
                if cfg.dt.get("cond_dim", 1) == cfg.encoder.hid_dim:
                    c = self.cond_stage_model.encode_label(xc)
                else: #cond_dim matches label dim: pass the raw normalized values directly (e.g. cond_dim=2 corresponds to cost+throughput)
                    c = xc
                if len(c.shape) == 1:
                    c = c.unsqueeze(1).unsqueeze(2)
                if len(c.shape) == 2:
                    c = c.unsqueeze(1)
                if not self.cond_stage_trainable:
                    c = c.detach()
            elif cond_key in ['prefix', 'multi-modal']:
                xc = batch[cond_key]
                c = self.get_learned_conditioning(xc)
                if not self.cond_stage_trainable:
                    c = c.detach()
            else:
                raise NotImplementedError

        else:
            c = None
            xc = None

        if self.adj_only: #adj_only: diffuse edge latent only, node_type as prior condition
            batch.x_start = batch_z.edge_attr
            batch.node_pos_orig = batch.pos_orig.clone().detach() if hasattr(batch, 'pos_orig') else batch.x_0[:, :2].clone().detach()
            batch.node_latent_clean = batch_z.x.clone().detach()
        elif self.geo_cond: #geo_cond: attr encoder encodes discrete attributes→latent_attr (denoising target), geo encoder encodes xy→latent_geo (condition)
            batch_geo = copy.deepcopy(batch)
            batch_geo.x = batch.x_0.clone().detach() #x_0 is already [N,2] xy coordinates
            batch_geo = self.encode_geo_stage(batch_geo)
            batch.x_start = batch_z.x #attr latent as denoising target
            batch.geo_latent_cond = batch_geo.x.clone().detach() #geo latent as node-level condition
            batch.edge_latent_cond = batch_geo.edge_attr.clone().detach() #edge latent obtained from the geo encoder
        elif self.adj_cond or self.fc_cond or self.unified_cond: #adj_cond/fc_cond/unified: diffuse node latent only, edge latent as condition
            if self.unified_cond: #unified: per-segment normalize coord/attr/edge→each std≈1 (following GeoLDM z_x_scale/z_h_scale)
                cd = self._coord_dim
                z_norm = torch.cat([batch_z.x[:, :cd] / self.z_coord_scale, batch_z.x[:, cd:] / self.z_attr_scale], dim=-1)
                batch.x_start = z_norm
                batch.edge_latent_cond = batch_z.edge_attr.clone().detach() / self.z_edge_scale
            else:
                batch.x_start = batch_z.x
                batch.edge_latent_cond = batch_z.edge_attr.clone().detach()
        elif self.affinity_mode: #affinity: same structure as geo_cond, diffuse affinity latent, geo+edge latent condition
            batch_geo = copy.deepcopy(batch)
            batch_geo.x = batch.x_0.clone().detach()
            batch_geo = self.encode_geo_stage(batch_geo)
            batch.x_start = batch_z.x
            batch.geo_latent_cond = batch_geo.x.clone().detach()
            batch.edge_latent_cond = batch_geo.edge_attr.clone().detach()
        else:
            batch.x_start = torch.cat([batch_z.x, batch_z.edge_attr], dim=0)
        batch.graph_start = batch_z.graph_attr.clone().detach()
        batch.c = c
        if return_first_stage_outputs:
            xrec_x, xrec_e, xrec_g = self.decode_first_stage(batch_z)  # TODO: implement decode_first_stage
            batch.x_rec = xrec_x
            batch.edge_attr_rec = xrec_e
            batch.graph_attr_rec = xrec_g
        if return_original_cond:
            batch.xc = xc
        return batch

    @torch.no_grad()
    def decode_first_stage(self, batch_z):
        return self.first_stage_model.decode(batch_z)  # TODO: implement .decode() method for first_stage_model

    @torch.no_grad()
    def encode_first_stage(self, batch_x, label=None, prefix=None):
        if hasattr(self.first_stage_model, 'encode') and callable(self.first_stage_model.encode):
            return self.first_stage_model.encode(batch_x, prefix=prefix, label=label)
        else:
            return self.first_stage_model(batch_x, prefix=prefix, label=label)

    # TODO: use this shared_step in training; the return is loss_value, loss_dict
    def shared_step(self, batch, t=None, prompt_graph_batch=None, task_id=None, **kwargs):
        if task_id is not None:
            batch.task_id = task_id
        batch = self.get_input(batch, prompt_graph_batch=prompt_graph_batch)
        loss = self(batch, t=t)
        return loss

    def training_step(self, batch, prompt_graph_batch=None, task_id=None):
        if self.train_mode == 'sample':
            t = torch.randint(0, self.num_timesteps, (batch.num_graphs,), device=self.device).long()
        elif self.train_mode == 'half-sample':
            t = torch.randint(0, 2 * self.num_timesteps, (batch.num_graphs,), device=self.device).long()
            t[t >= self.num_timesteps] = self.num_timesteps - 1
        else:
            t = torch.ones((batch.num_graphs, ), device=self.device).long() * (self.num_timesteps - 1)
        loss, loss_dict, loss_task, graph_pred, loss_node, loss_edge, loss_graph, loss_encoder = self.shared_step(batch, t=t, prompt_graph_batch=prompt_graph_batch, task_id=task_id)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss, loss_task, graph_pred, loss_node, loss_edge, loss_graph, loss_encoder

    @torch.no_grad()
    def validation_step(self, batch, prompt_graph_batch=None, task_id=None):
        t = torch.ones((batch.num_graphs, ), device=self.device).long() * (self.num_timesteps)
        loss, loss_dict_no_ema, loss_graph, graph_pred = self.shared_step(batch, t=t, prompt_graph_batch=prompt_graph_batch, task_id=task_id)
        # with self.ema_scope():
        #     _, loss_dict_ema = self.shared_step(batch, task_id)
        #     loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        return loss, loss_graph, graph_pred

    def forward(self, batch, t=None, *args, **kwargs):
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch.num_graphs,), device=self.device).long()
        c = batch.get("c", None)
        # TODO: reconsider the loss
        # x_start = batch.x_start
        if self.model.conditioning_key is not None:
            assert c is not None
        return self.p_losses(batch, c, t, batch.batch_idx, *args, **kwargs)

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart, batch_idx):
        return (extract_into_sparse_tensor(self.sqrt_recip_alphas_cumprod, t, batch_idx) * x_t - pred_xstart) / \
               extract_into_sparse_tensor(self.sqrt_recipm1_alphas_cumprod, t, batch_idx)

    def _prior_bpd(self, x_start, batch_idx):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t, batch_idx)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    # TODO: reconsider how to construct the loss; also note for different levels/task types
    def p_losses(self, batch, cond, t, batch_idx, noise=None):
        x_start = batch.x_start.clone().detach()
        graph_attr_label = batch.graph_start.clone().detach()
        noise = default(noise, lambda: torch.randn_like(x_start))
        if self.adj_only: #adj_only: add noise to edge latent only, inject node_type as prior condition
            if self.force_undirected:
                noise = symmetrize(batch.edge_index, batch.batch, noise)
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, batch_idx=batch_idx)
            batch_noisy = copy.deepcopy(batch)
            batch_noisy.edge_attr = x_noisy
            batch_noisy.x = self.node_type_embed(batch.node_type.to(x_start.device))
        elif self.geo_cond or self.affinity_mode: #geo_cond/affinity: add noise to latent only, geo+edge latent as condition
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, batch_idx=batch_idx)
            batch_noisy = copy.deepcopy(batch)
            batch_noisy.x = x_noisy
            batch_noisy.edge_attr = batch.edge_latent_cond
            batch_noisy.geo_cond = batch.geo_latent_cond
        elif self.adj_cond or self.fc_cond or self.unified_cond: #adj_cond/fc_cond/unified: add noise to node latent only, edge latent condition
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, batch_idx=batch_idx)
            batch_noisy = copy.deepcopy(batch)
            batch_noisy.x = x_noisy
            batch_noisy.edge_attr = batch.edge_latent_cond
        else:
            if self.force_undirected:
                noise_symmetrized_edge = symmetrize(batch.edge_index, batch.batch, noise[batch.num_nodes:, :])
                noise[batch.num_nodes:, :] = noise_symmetrized_edge
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, batch_idx=batch_idx)
            batch_noisy = copy.deepcopy(batch)
            batch_noisy.x = x_noisy[:batch.num_nodes, :]
            batch_noisy.edge_attr = x_noisy[batch.num_nodes:, :]
        batch_output = self.model(batch_noisy, t, cond)
        if self.adj_only: model_output = batch_output.edge_attr
        elif self.adj_cond or self.geo_cond or self.fc_cond or self.affinity_mode or self.unified_cond: model_output = batch_output.x #node-level diffusion: node prediction only
        else: model_output = torch.cat([batch_output.x, batch_output.edge_attr], dim=0)
        if self.unified_cond: #unified: per-segment denormalize latent before decode (model_output stays normalized for loss)
            cd = self._coord_dim
            batch_output.x = torch.cat([batch_output.x[:, :cd] * self.z_coord_scale, batch_output.x[:, cd:] * self.z_attr_scale], dim=-1)
            batch_output.edge_attr = batch_output.edge_attr * self.z_edge_scale
        node_decode, edge_decode, graph_decode = self.decode_first_stage(batch_output)

        _y = batch.y.clone().detach()
        if graph_decode.dim() == _y.dim() and graph_decode.shape == _y.shape: #compute task loss when dims match, otherwise skip (Interchange: decode=[B] vs y=[B,2])
            loss_task, graph_decode = compute_loss(graph_decode, _y)
        else:
            loss_task = torch.tensor(0., device=graph_decode.device)

        if cfg.train.mode in ['interchange_unconditional']: #interchange: continuous node values + adjacency (adj_only uses raw coordinates + generated adjacency, adj_cond uses GT, otherwise argmax)
            interchange_graphs = []
            accumulated_node, accumulated_edge = 0, 0
            for i in range(batch.num_graphs):
                nn_i = batch.num_node_per_graph[i]
                if self.adj_only: #adj_only: raw coordinates + generated adjacency + GT adjacency (+optional meta: carries raw_* under segment_mode)
                    ctrl_pts = batch.node_pos_orig[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    adj = torch.argmax(edge_decode[accumulated_edge: accumulated_edge + nn_i ** 2], dim=1, keepdim=False).reshape(nn_i, nn_i).detach().cpu()
                    gt_adj = batch.adj_flat[accumulated_edge: accumulated_edge + nn_i ** 2].reshape(nn_i, nn_i).long().detach().cpu()
                    meta = self._seg_meta(batch, i) if self.segment_mode else None
                    interchange_graphs.append((ctrl_pts, adj, gt_adj, meta))
                elif self.geo_cond: #geo_cond: simple placeholder in p_losses
                    ctrl_pts = batch.pos_orig[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    adj = batch.adj_flat[accumulated_edge: accumulated_edge + nn_i ** 2].reshape(nn_i, nn_i).long().detach().cpu()
                    interchange_graphs.append((ctrl_pts, adj, None, None))
                elif self.affinity_mode: #affinity: use GT coordinates, derive adjacency from affinity decode
                    ctrl_pts = batch.pos_orig[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    aff_pred = node_decode[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    gt_adj = batch.adj_flat[accumulated_edge: accumulated_edge + nn_i ** 2].reshape(nn_i, nn_i).long().detach().cpu()
                    knn_idx = batch.affinity_knn_idx[accumulated_node: accumulated_node + nn_i].detach().cpu() if hasattr(batch, 'affinity_knn_idx') else None
                    interchange_graphs.append(('affinity', ctrl_pts, aff_pred, gt_adj, knn_idx))
                elif self.fc_cond: #fc_cond: show scatter points only, no edges
                    ctrl_pts = node_decode[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    interchange_graphs.append(('scatter', ctrl_pts))
                elif self.unified_cond: #unified: triplet (xy,lanes,speed) + GT adjacency
                    xy_dec, lanes_dec, speed_dec = node_decode
                    ctrl_pts = xy_dec[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    lp = lanes_dec[accumulated_node: accumulated_node + nn_i].argmax(dim=1).detach().cpu()
                    sp = speed_dec[accumulated_node: accumulated_node + nn_i].argmax(dim=1).detach().cpu()
                    adj = batch.adj_flat[accumulated_edge: accumulated_edge + nn_i ** 2].reshape(nn_i, nn_i).long().detach().cpu()
                    interchange_graphs.append(('unified', ctrl_pts, adj, lp, sp))
                elif self.segment_mode: #segment coarse graph: decode segment length L + type + segment connections W, uncoarsen to full adjacency Â into meta
                    interchange_graphs.append(self._seg_decode(batch, i, node_decode, edge_decode, accumulated_node, accumulated_edge, nn_i))
                elif self.adj_cond:
                    ctrl_pts = node_decode[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    adj = batch.adj_flat[accumulated_edge: accumulated_edge + nn_i ** 2].reshape(nn_i, nn_i).long().detach().cpu()
                    interchange_graphs.append((ctrl_pts, adj))
                else:
                    ctrl_pts = node_decode[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    adj = torch.argmax(edge_decode[accumulated_edge: accumulated_edge + nn_i ** 2], dim=1, keepdim=False).reshape(nn_i, nn_i).detach().cpu()
                    interchange_graphs.append((ctrl_pts, adj))
                accumulated_node += nn_i
                accumulated_edge += nn_i ** 2
            graph_decode = interchange_graphs

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        # graph_criterion = nn.MSELoss()
        # loss_task = graph_criterion(graph_decode, batch.y)
        # encoder_criterion = nn.L1Loss() if cfg.model.loss_fun == 'l1' else nn.CrossEntropyLoss()
        # loss_graph = torch.nn.functional.mse_loss(batch_output.graph_attr, graph_attr_label).mean()
        # encoder_pred = self.first_stage_model.decode_graph(graph_attr_label).flatten()
        # if cfg.dataset.format == 'PyG-QM9':
        #     encoder_pred = encoder_pred * batch.get('y_std', 1.) + batch.get('y_mean', 0.)
        # loss_graph_encoder, _ = compute_loss(encoder_pred, batch.y.clone().detach())
        loss_graph_encoder = 0.
        # TODO: complete loss_graph_encoder for class GraphTransformerDecoder
        # print(loss_graph_encoder)

        _raw_loss = self.get_loss(model_output, target, mean=False) #[N, D]
        if self.unified_cond: #unified: coord_weight weighting → higher loss weight on coordinate dims (following GeoLDM compute_error)
            _w = torch.ones(1, _raw_loss.shape[1], device=_raw_loss.device)
            _w[:, :self._coord_dim] = self._coord_weight
            loss_simple = (_raw_loss * _w).mean([1])
        else:
            loss_simple = _raw_loss.mean([1])
        loss_graph = self.get_loss(batch_output.graph_attr, graph_attr_label, mean=False).mean([1])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        loss_dict.update({f'{prefix}/loss_graph': loss_graph.mean()})

        if (t >= self.num_timesteps).any():
            t = t - 1
        logvar_t = self.logvar[t].to(self.device)
        logvar_t = logvar_t[batch_idx]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss
        if self.adj_only: #adj_only: compute edge loss only, no node loss, supports edge_pos_weight + adjacency CE auxiliary loss
            loss_node = torch.tensor(0.0, device=loss.device)
            _ew = None
            _epw = cfg.diffusion.get('edge_pos_weight', 50.0)
            if hasattr(batch, 'adj_flat'):
                _ew = 1.0 + (_epw - 1.0) * batch.adj_flat.to(loss.device)
            loss_edge = ((loss * _ew).mean() if _ew is not None else loss.mean()) * self.edge_factor
            loss_graph = loss_graph.mean() * self.graph_factor
            loss = loss_edge + loss_graph
            _adj_rf = cfg.diffusion.get('adj_recon_factor', 0.0) #adjacency CE auxiliary loss: supervise binary classification directly through the frozen decoder, gradients flow back to DT only
            if _adj_rf > 0 and hasattr(batch, 'adj_flat'):
                _edge_logits = self.first_stage_model.decode_edge(batch_output.edge_attr)
                _adj_target = batch.adj_flat.long().to(_edge_logits.device)
                _cls_w = torch.tensor([1.0, _epw], device=_edge_logits.device)
                loss_adj = F.cross_entropy(_edge_logits, _adj_target, weight=_cls_w)
                loss = loss + loss_adj * _adj_rf
                loss_dict[f'{prefix}/loss_adj_ce'] = loss_adj.detach()
                with torch.no_grad(): #adjacency accuracy monitoring (not part of gradient)
                    _pred_adj = _edge_logits.argmax(dim=1)
                    loss_dict[f'{prefix}/adj_acc'] = (_pred_adj == _adj_target).float().mean()
                    loss_dict[f'{prefix}/adj_prec'] = (_pred_adj[_adj_target == 1] == 1).float().mean() if (_adj_target == 1).any() else torch.tensor(0.0)
            loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=1)
            _vlb_edge = (self.lvlb_weights[t][batch_idx] * loss_vlb * _ew).mean() if _ew is not None else (self.lvlb_weights[t][batch_idx] * loss_vlb).mean()
            loss_vlb = _vlb_edge * self.edge_factor \
                + (self.get_loss(batch_output.graph_attr, graph_attr_label, mean=False).mean(dim=1) * self.lvlb_weights[t]).mean() * self.graph_factor
        elif self.adj_cond or self.geo_cond or self.fc_cond or self.affinity_mode or self.unified_cond: #node-level diffusion: compute node latent loss only, no edge loss
            loss_node = loss.mean() * self.node_factor
            loss_edge = torch.tensor(0.0, device=loss.device)
            loss_graph = loss_graph.mean() * self.graph_factor
            loss = loss_node + loss_graph
            _vlb_raw = self.get_loss(model_output, target, mean=False) #[N, D]
            if self.unified_cond: #unified: vlb also weighted by coord_weight
                _w = torch.ones(1, _vlb_raw.shape[1], device=_vlb_raw.device)
                _w[:, :self._coord_dim] = self._coord_weight
                _vlb_raw = _vlb_raw * _w
            loss_vlb = _vlb_raw.mean(dim=1)
            loss_vlb = (self.lvlb_weights[t][batch_idx] * loss_vlb).mean() * self.node_factor \
                + (self.get_loss(batch_output.graph_attr, graph_attr_label, mean=False).mean(dim=1) * self.lvlb_weights[t]).mean() * self.graph_factor
        else:
            loss_node = loss[:batch.num_nodes].mean() * self.node_factor
            _ew = None #edge weighting: amplify true-edge weight for sparse interchange graphs, alleviating the imbalance where no-edge dominates the loss
            if cfg.train.mode in ['interchange_unconditional'] and hasattr(batch, 'adj_flat'):
                _epw = cfg.diffusion.get('edge_pos_weight', 5.0)
                _ew = 1.0 + (_epw - 1.0) * batch.adj_flat.to(loss.device)
            loss_edge = ((loss[batch.num_nodes:] * _ew).mean() if _ew is not None else loss[batch.num_nodes:].mean()) * self.edge_factor
            loss_graph = loss_graph.mean() * self.graph_factor
            loss = loss_node + loss_edge + loss_graph
            loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=1)
            loss_vlb = self.lvlb_weights[t][batch_idx] * loss_vlb
            _vlb_edge = (loss_vlb[batch.num_nodes:] * _ew).mean() if _ew is not None else loss_vlb[batch.num_nodes:].mean()
            loss_vlb = loss_vlb[:batch.num_nodes].mean() * self.node_factor + _vlb_edge * self.edge_factor \
                + (self.get_loss(batch_output.graph_attr, graph_attr_label, mean=False).mean(dim=1) * self.lvlb_weights[t]).mean() * self.graph_factor
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})
        loss_dict.update({f'{prefix}/loss_task': loss_task})

        # TODO: these lines do not apply for generic graphs because the decoding method is different
        if hasattr(self.first_stage_model, 'decode_recon') and callable(self.first_stage_model.decode_recon) and self.recon_factor > 0:
            node_recon, edge_recon = self.first_stage_model.decode_recon(batch_output)
            criterion_node = nn.CrossEntropyLoss()
            criterion_edge = nn.CrossEntropyLoss()
            loss_structure_recon = criterion_node(node_recon, batch.x_0.flatten()) * self.node_factor \
                                   + criterion_edge(edge_recon, batch.edge_attr_0.flatten()) * self.edge_factor
            loss += loss_structure_recon * self.recon_factor

        return loss, loss_dict, loss_task, graph_decode, loss_node, loss_edge, loss_graph, loss_graph_encoder

    def p_mean_variance(self, batch_x, c, t, batch_idx, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        if self.adj_only:
            x = batch_x.edge_attr.clone()
        elif self.adj_cond or self.fc_cond or self.geo_cond or self.affinity_mode or self.unified_cond:
            x = batch_x.x.clone()
        else:
            x = torch.cat([batch_x.x.clone(), batch_x.edge_attr.clone()], dim=0)
        batch_model_out = self.model(batch_x, t_in, c)
        if score_corrector is not None:
            assert self.parameterization == "eps"
            batch_model_out = score_corrector.modify_score(self, batch_model_out, batch_x, t, c, **corrector_kwargs)
        if return_codebook_ids:
            batch_model_out, logits = batch_model_out
        if self.parameterization == "eps":
            if self.adj_only:
                x_recon = self.predict_start_from_noise(x, t=t, noise=batch_model_out.edge_attr, batch_idx=batch_idx)
            elif self.adj_cond or self.fc_cond or self.geo_cond or self.affinity_mode or self.unified_cond:
                x_recon = self.predict_start_from_noise(x, t=t, noise=batch_model_out.x, batch_idx=batch_idx)
            else:
                x_recon = self.predict_start_from_noise(x, t=t, noise=torch.cat([batch_model_out.x, batch_model_out.edge_attr], dim=0), batch_idx=batch_idx)
        elif self.parameterization == "x0":
            if self.adj_only: x_recon = batch_model_out.edge_attr
            elif self.adj_cond or self.fc_cond or self.geo_cond or self.affinity_mode or self.unified_cond: x_recon = batch_model_out.x
            else: x_recon = torch.cat([batch_model_out.x, batch_model_out.edge_attr], dim=0)
        else:
            raise NotImplementedError()
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t, batch_idx=batch_idx)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, batch_x, c, t, batch_idx, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        device = batch_x.x.device
        b = batch_idx.shape[0]
        outputs = self.p_mean_variance(batch_x=batch_x, c=c, t=t, batch_idx=batch_idx, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs
        noise = noise_like(model_mean.shape, device, repeat_noise) * temperature
        _node_only = self.adj_cond or self.fc_cond or self.geo_cond or self.affinity_mode or self.unified_cond
        if self.adj_only and self.force_undirected: #adj_only: noise covers all edges, symmetrize directly
            noise = symmetrize(batch_x.edge_index, batch_x.batch, noise)
        elif not _node_only and self.force_undirected: #symmetrize the edge part during joint generation
            symmetrized = symmetrize(batch_x.edge_index, batch_x.batch, noise[batch_x.num_nodes:])
            noise[batch_x.num_nodes:] = symmetrized
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        nonzero_mask = (1 - (t == 0).float())[batch_idx].reshape(-1, 1)
        p_sampled_x = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        if self.adj_only: #adj_only: update edges only, nodes use node_type prior condition
            batch_x.edge_attr = p_sampled_x
            batch_x.x = self.node_type_embed(batch_x.node_type.to(device))
        elif self.geo_cond or self.affinity_mode: #geo_cond/affinity: update nodes, keep edge+geo conditions
            batch_x.x = p_sampled_x
            batch_x.edge_attr = batch_x.edge_latent_cond
            if hasattr(batch_x, 'geo_latent_cond'): batch_x.geo_cond = batch_x.geo_latent_cond
        elif self.adj_cond or self.fc_cond or self.unified_cond: #adj_cond/fc_cond/unified: update nodes only, edge_attr restored to clean condition
            batch_x.x = p_sampled_x
            batch_x.edge_attr = batch_x.edge_latent_cond
        else:
            batch_x.x = p_sampled_x[:batch_x.num_nodes]
            batch_x.edge_attr = p_sampled_x[batch_x.num_nodes:]
        if return_codebook_ids:
            return batch_x, logits.argmax(dim=1)
        if return_x0:
            return batch_x, x0
        else:
            return batch_x

    @torch.no_grad()
    def p_sample_loop(self, batch, cond, shape, batch_idx, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None, noise_temp=1.0):    #noise_temp: starting randn multiplier >1 amplifies the ramp initial noise to increase diversity; default 1.0 keeps behavior unchanged
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = int(torch.max(batch_idx)) + 1
        if x_T is None:
            img = torch.randn(shape, device=device, dtype=torch.float) * float(noise_temp)   #noise_temp only amplifies the starting randn; mask=1 regions are reset by q_sample each step and are unaffected
        else:
            img = x_T
        if self.adj_only: #adj_only: noise initializes edges only, nodes use node_type prior condition
            if self.force_undirected:
                img = symmetrize(batch.edge_index, batch.batch, img)
            batch.edge_attr = img
            batch.x = self.node_type_embed(batch.node_type.to(device))
        elif self.geo_cond or self.affinity_mode: #geo_cond/affinity: noise initializes latent, geo+edge conditions
            batch.x = img
            batch.edge_attr = batch.edge_latent_cond
            batch.geo_cond = batch.geo_latent_cond
        elif self.adj_cond or self.fc_cond or self.unified_cond: #adj_cond/fc_cond/unified: noise initializes nodes only, edge condition
            batch.x = img
            batch.edge_attr = batch.edge_latent_cond
        else:
            if self.force_undirected:
                symmetrized = symmetrize(batch.edge_index, batch.batch, img[batch.num_nodes:])
                img[batch.num_nodes:] = symmetrized
            batch.x = img[: batch.num_nodes]
            batch.edge_attr = img[batch.num_nodes:]
        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if mask is not None:
            assert x0 is not None
        _nan_logged = False
        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond), batch_idx=None)
            batch = self.p_sample(batch, cond, ts, batch_idx, clip_denoised=self.clip_denoised, quantize_denoised=quantize_denoised)
            if self.adj_only: img = batch.edge_attr
            elif self.adj_cond or self.geo_cond or self.fc_cond or self.affinity_mode or self.unified_cond: img = batch.x
            else: img = torch.cat([batch.x, batch.edge_attr], dim=0)
            if not _nan_logged and torch.isnan(img).any():
                logging.info(f'[NaN-DBG] p_sample_loop: NaN first at step t={i}/{timesteps} nan={torch.isnan(img).sum().item()}/{img.numel()} dtype={img.dtype}')
                _nan_logged = True
            if mask is not None:
                img_orig = self.q_sample(x0, ts, batch_idx=batch_idx)
                img = img_orig * mask + (1. - mask) * img
                if self.adj_only:
                    batch.edge_attr = img
                    batch.x = self.node_type_embed(batch.node_type.to(device))
                elif self.adj_cond or self.fc_cond or self.geo_cond or self.affinity_mode or self.unified_cond:
                    batch.x = img
                    batch.edge_attr = batch.edge_latent_cond
                else:
                    batch.x = img[: batch.num_nodes]
                    batch.edge_attr = img[batch.num_nodes:]
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        if return_intermediates:
            return batch, intermediates
        return batch

    @torch.no_grad()
    def sample(self, batch, cond, batch_size=16, batch_idx=None, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_idx.shape[0], self.hid_dim)
        # TODO: check if this part is necessary for multiple conditions
        # if cond is not None:
        #     if isinstance(cond, dict):
        #         cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
        #         list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
        #     else:
        #         cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]  # TODO: check whether this obtain correct cond
        return self.p_sample_loop(batch,
                                  cond,
                                  shape,
                                  batch_idx,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0, **kwargs)                  #pass through inference-time options such as noise_temp / x_T

    @torch.no_grad()
    def sample_log(self, batch, cond, batch_size, batch_idx, ddim, ddim_steps, eta, use_ddpm_steps, **kwargs):

        if ddim:  # TODO: further implement DDIM
            ddim_sampler = DDIMSampler(self)
            shape = (batch_idx.shape[0], self.hid_dim)
            samples = ddim_sampler.sample(batch, ddim_steps, batch_size,
                                                        shape, cond, verbose=False, eta=eta, return_intermediates=False, **kwargs)

        else:
            samples = self.sample(batch=batch, cond=cond, batch_size=batch_size, batch_idx=batch_idx,
                                                 return_intermediates=False, **kwargs)

        return samples


    @torch.no_grad()
    def inference(self, batch, sample=True, ddim_steps=None, ddim_eta=0., use_ddpm_steps=False, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, visualize=False, mask=None, x0=None, noise_temp=1.0, start_t0=None, **kwargs):  #mask/x0: RePaint; noise_temp>1: amplify starting randn to increase ramp diversity; start_t0: task C controlled latent-space perturbation — partially add noise to the reference latent up to t0, then denoise from there (small t0→near reconstruction / large t0→near regeneration)

        use_ddim = ddim_steps is not None
        N = batch.num_graphs
        batch = self.get_input(batch, return_first_stage_outputs=True, force_c_encode=True, return_original_cond=True)
        z, c, x, xrec, xc = (batch.x_start, batch.graph_start), batch.get('c', None), (batch.x_0, batch.edge_attr_0), (batch.x_rec, batch.edge_attr_rec, batch.graph_attr_rec), batch.get('xc', None)
        batch_idx = batch.batch_idx
        _extra_kw = {'noise_temp': float(noise_temp)} if float(noise_temp) != 1.0 else {}
        if self.adj_only: #adj_only: denoise starting from the val raw adjacency with noise added (t=T-1)
            t_start = torch.full((N,), self.num_timesteps - 1, device=self.device, dtype=torch.long)
            _extra_kw['x_T'] = self.q_sample(x_start=batch.x_start, t=t_start, batch_idx=batch_idx)
        elif start_t0 is not None: #task C: z_t0 = sqrt(ᾱ_{t0})·z_ref + sqrt(1-ᾱ_{t0})·ε; start_T truncates the iterator to only iterate t0→0
            assert not use_ddim, 'start_t0 is incompatible with ddim_steps (x0-parameterized truncated sampling diverges); please set ddim_steps=None'
            _t0 = max(0, min(int(start_t0), self.num_timesteps - 1))
            t_vec = torch.full((N,), _t0, device=self.device, dtype=torch.long)
            _extra_kw['x_T'] = self.q_sample(x_start=batch.x_start, t=t_vec, batch_idx=batch_idx)
            _extra_kw['start_T'] = _t0 + 1
        if mask is not None and x0 is None: x0 = batch.x_start                                                                                                          #task 1 inpainting: by default use the encoder-output x_start as the latent reference (the script only needs to provide node_mask and batch.x with mainline xy written in)
        samples = self.sample_log(batch=batch, cond=c, batch_size=N, batch_idx=batch_idx, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta, use_ddpm_steps=use_ddpm_steps, mask=mask, x0=x0, **_extra_kw)  #pass through RePaint mask/x0 + noise_temp / start_t0
        if self.unified_cond: #unified: per-segment denormalize latent before decode
            cd = self._coord_dim
            samples.x = torch.cat([samples.x[:, :cd] * self.z_coord_scale, samples.x[:, cd:] * self.z_attr_scale], dim=-1)
            samples.edge_attr = samples.edge_attr * self.z_edge_scale
        if visualize: pass
        node_decode, edge_decode, graph_decode = self.decode_first_stage(samples)
        _y_inf = batch.y.clone().detach()
        if graph_decode.dim() == _y_inf.dim() and graph_decode.shape == _y_inf.shape: #compute graph loss when dims match, otherwise skip (Interchange: decode=[B] vs y=[B,2])
            loss_graph, graph_decode = compute_loss(graph_decode, _y_inf)
        else:
            loss_graph = torch.tensor(0., device=graph_decode.device)

        if cfg.train.mode in ['interchange_unconditional']: #interchange decode at inference (adj_only uses raw coordinates + generated adjacency, adj_cond uses GT adjacency)
            interchange_graphs = []
            accumulated_node, accumulated_edge = 0, 0
            for i in range(batch.num_graphs):
                nn_i = batch.num_node_per_graph[i]
                if self.adj_only: #adj_only: raw coordinates + generated adjacency + GT adjacency (+optional meta: carries raw_* for visualization under segment_mode)
                    ctrl_pts = batch.node_pos_orig[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    adj = torch.argmax(edge_decode[accumulated_edge: accumulated_edge + nn_i ** 2], dim=1, keepdim=False).reshape(nn_i, nn_i).detach().cpu()
                    gt_adj = batch.adj_flat[accumulated_edge: accumulated_edge + nn_i ** 2].reshape(nn_i, nn_i).long().detach().cpu()
                    meta = self._seg_meta(batch, i) if self.segment_mode else None #segment mode: extract the corresponding graph's raw_pos/raw_adj etc. for comparison plotting
                    interchange_graphs.append((ctrl_pts, adj, gt_adj, meta))
                elif self.geo_cond: #geo_cond: node_decode=(lanes_logits, speed_logits), use GT coordinates + GT adjacency + generated attributes
                    if isinstance(node_decode, tuple):
                        lanes_logits, speed_logits = node_decode
                        lanes_pred = lanes_logits[accumulated_node: accumulated_node + nn_i].argmax(dim=1).detach().cpu()
                        speed_pred = speed_logits[accumulated_node: accumulated_node + nn_i].argmax(dim=1).detach().cpu()
                    else:
                        lanes_pred = speed_pred = None
                    ctrl_pts = batch.pos_orig[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    adj = batch.adj_flat[accumulated_edge: accumulated_edge + nn_i ** 2].reshape(nn_i, nn_i).long().detach().cpu()
                    interchange_graphs.append((ctrl_pts, adj, lanes_pred, speed_pred))
                elif self.affinity_mode: #affinity: GT coordinates + affinity decode → adjacency
                    ctrl_pts = batch.pos_orig[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    aff_pred = node_decode[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    gt_adj = batch.adj_flat[accumulated_edge: accumulated_edge + nn_i ** 2].reshape(nn_i, nn_i).long().detach().cpu()
                    knn_idx = batch.affinity_knn_idx[accumulated_node: accumulated_node + nn_i].detach().cpu() if hasattr(batch, 'affinity_knn_idx') else None
                    interchange_graphs.append(('affinity', ctrl_pts, aff_pred, gt_adj, knn_idx))
                elif self.fc_cond: #fc_cond: show scatter points only, no edges
                    ctrl_pts = node_decode[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    interchange_graphs.append(('scatter', ctrl_pts))
                elif self.unified_cond: #unified: triplet (xy,lanes,speed) + GT adjacency
                    xy_dec, lanes_dec, speed_dec = node_decode
                    ctrl_pts = xy_dec[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    lp = lanes_dec[accumulated_node: accumulated_node + nn_i].argmax(dim=1).detach().cpu()
                    sp = speed_dec[accumulated_node: accumulated_node + nn_i].argmax(dim=1).detach().cpu()
                    adj = batch.adj_flat[accumulated_edge: accumulated_edge + nn_i ** 2].reshape(nn_i, nn_i).long().detach().cpu()
                    interchange_graphs.append(('unified', ctrl_pts, adj, lp, sp))
                elif self.segment_mode: #segment coarse graph: decode segment length L + type + segment connections W, uncoarsen to full adjacency Â into meta
                    interchange_graphs.append(self._seg_decode(batch, i, node_decode, edge_decode, accumulated_node, accumulated_edge, nn_i))
                elif self.adj_cond:
                    ctrl_pts = node_decode[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    adj = batch.adj_flat[accumulated_edge: accumulated_edge + nn_i ** 2].reshape(nn_i, nn_i).long().detach().cpu()
                    interchange_graphs.append((ctrl_pts, adj))
                else:
                    ctrl_pts = node_decode[accumulated_node: accumulated_node + nn_i].detach().cpu()
                    adj = torch.argmax(edge_decode[accumulated_edge: accumulated_edge + nn_i ** 2], dim=1, keepdim=False).reshape(nn_i, nn_i).detach().cpu()
                    interchange_graphs.append((ctrl_pts, adj))
                accumulated_node += nn_i
                accumulated_edge += nn_i ** 2
            graph_decode = interchange_graphs

        return loss_graph, graph_decode


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        # self.diffusion_model = instantiate_from_config(diff_model_config)
        self.diffusion_model = DenoisingTransformer(cfg)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    # TODO: haven't implement other conditioning_key forward, including in get_batch()
    def forward(self, x, t, c_crossattn=None, c_concat=None):  # originally c_crossattn and c_concat are list
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            # cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, c_crossattn)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out
