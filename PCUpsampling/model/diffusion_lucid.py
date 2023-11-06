import math
from collections import namedtuple
from functools import partial
from random import random

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from utils.losses import get_scaling, projection_loss

try:
    from .unet_mink import MinkUnet
except:
    pass

from .unet_pointvoxel import PVCLionSmall
from gecco_torch.models.linear_lift import LinearLift
from gecco_torch.models.set_transformer import SetTransformer

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])
from loguru import logger


# gaussian diffusion trainer class
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def identity(t, *args, **kwargs):
    return t


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=0.9, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def normalize_to_neg_one_to_one(x):
    return x * 2.0 - 1.0


def unnormalize_to_zero_to_one(x):
    return (x + 1.0) / 2.0


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def dynamic_threshold_percentile(x, threshold=0.975):
    """
    dynamic thresholding, based on percentile
    """
    s = torch.quantile(rearrange(x, "b ... -> b (...)").abs(), threshold, dim=-1)
    s.clamp_(min=1.0)
    s = right_pad_dims_to(x, s)
    x = x.clamp(-s, s) / s

    return x


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        cfg,
        schedule_fn_kwargs=dict(),
        auto_normalize=False,
        offset_noise_strength=0.0,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight=False,  # https://arxiv.org/abs/2303.09556
    ):
        super().__init__()

        # load configs
        beta_schedule = cfg.diffusion.schedule
        timesteps = cfg.diffusion.timesteps
        sampling_timesteps = cfg.diffusion.sampling_timesteps
        ddim_sampling_eta = cfg.diffusion.ddim_sampling_eta
        min_snr_gamma = cfg.diffusion.min_snr_gamma
        self.reg_scale = cfg.diffusion.reg_scale

        # setup networks
        if cfg.model.type == "PVD":
            self.model = PVCLionSmall(
                out_dim=cfg.model.out_dim,
                input_dim=cfg.model.in_dim,
                npoints=cfg.data.npoints,
                embed_dim=cfg.model.time_embed_dim,
                use_att=cfg.model.use_attention,
                dropout=cfg.model.dropout,
                extra_feature_channels=cfg.model.extra_feature_channels,
            ).cuda()
        elif cfg.model.type == "Mink":
            self.model = MinkUnet(
                dim=cfg.model.time_embed_dim,
                init_ds_factor=cfg.model.Mink.init_ds_factor,
                D=cfg.model.Mink.D,
                in_shape=[cfg.training.bs, cfg.model.in_dim, cfg.data.npoints],
                out_dim=cfg.model.out_dim,
                in_channels=cfg.model.in_dim + cfg.model.extra_feature_channels,
                dim_mults=cfg.model.Mink.dim_mults,
                downsampfactors=cfg.model.Mink.downsampfactors,
                use_attention=cfg.model.use_attention,
            ).cuda()
        elif cfg.model.type == "SetTransformer":
            set_transformer = SetTransformer(
                n_layers=cfg.model.ST.layers,
                feature_dim=cfg.model.ST.fdim,
                num_inducers=cfg.model.ST.inducers,
                t_embed_dim=1,
            ).cuda()
            self.model = LinearLift(
                inner=set_transformer,
                feature_dim=cfg.model.ST.fdim,
                in_dim=cfg.model.in_dim + cfg.model.extra_feature_channels,
                out_dim=cfg.model.out_dim,
            ).cuda()
        else:
            raise NotImplementedError(cfg.unet)

        logger.info(
            f"Generated model with following number of params (M): {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f}"
        )

        # dimensions
        self.channels = cfg.data.nc
        self.npoints = cfg.data.npoints
        self.amp = cfg.training.amp

        objective = cfg.diffusion.objective
        self.objective = objective
        self.self_condition = self.model.self_condition
        self.dynamic_threshold = cfg.diffusion.dynamic_threshold
        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.timesteps_clip = int(cfg.diffusion.timesteps_clip)

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            register_buffer("loss_weight", maybe_clipped_snr / snr)
        elif objective == "pred_x0":
            register_buffer("loss_weight", maybe_clipped_snr)
        elif objective == "pred_v":
            register_buffer("loss_weight", maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        x,
        t,
        cond=None,
        x_self_cond=None,
        clip_x_start=False,
        rederive_pred_noise=True,
    ):
        model_output = self.model(x, t, cond=cond, x_self_cond=x_self_cond)

        pred_noise, x_start = self.to_noise_and_xstart(
            model_output,
            x,
            t,
            clip_x_start=clip_x_start,
            rederive_pred_noise=rederive_pred_noise,
        )

        return ModelPrediction(pred_noise, x_start)

    def to_noise_and_xstart(self, model_output, x, t, clip_x_start=False, rederive_pred_noise=False):
        # setup clipping
        if not self.dynamic_threshold:
            maybe_clip = partial(torch.clamp, min=-3.0, max=3.0) if clip_x_start else identity
        else:
            maybe_clip = dynamic_threshold_percentile if clip_x_start else identity

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(self, x, t, cond=None, x_self_cond=None, clip_denoised=False):
        preds = self.model_predictions(x, t, cond=cond, x_self_cond=x_self_cond, clip_x_start=clip_denoised)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        elif self.dynamic_threshold:
            x_start = dynamic_threshold_percentile(x_start)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, cond=None, x_self_cond=None, clip=False):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, cond=cond, x_self_cond=x_self_cond, clip_denoised=clip
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, hint=None, save_every=0, clip=False, *args, **kwargs):
        img = torch.randn(shape, device=self.device)

        total_steps = min(self.num_timesteps, self.timesteps_clip)

        # generate start by hint if hint is not None
        # this is done by diffusing the hint the same way as trainig samples
        if hint is not None:
            with torch.no_grad():
                t = total_steps - 1
                t = torch.tensor([t], device=self.device).long()
                t = repeat(t, "1 -> b", b=shape[0])
                img, alphas, sigmas = self.q_sample(x_start=hint, noise=img, t=t, return_alphas_simgas=True)

        imgs = [img]

        x_start = None  # TODO add back self conditioning
        sample_step = 0

        for t in tqdm(
            reversed(range(0, total_steps)),
            desc="sampling loop time step",
            total=total_steps,
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, cond=cond, x_self_cond=self_cond, clip=clip)
            if save_every and ((sample_step + 1) % save_every == 0):
                imgs.append(img)
            sample_step += 1

        ret = img if not save_every else imgs

        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, cond=None, hint=None, save_every=0, clip=False, *args, **kwargs):
        # extract potential timestep clipping
        total_steps = min(self.num_timesteps, self.timesteps_clip)

        times = torch.linspace(-1, total_steps - 1, steps=total_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=self.device)

        # generate start by hint if hint is not None
        # this is done by diffusing the hint the same way as trainig samples
        if hint is not None:
            with torch.no_grad():
                t = torch.tensor([total_steps - 1], device=self.device).long()
                t = repeat(t, "1 -> b", b=shape[0])
                img = self.q_sample(x_start=hint, noise=img, t=t)

        x_start = None
        # set saving
        imgs = [img]
        sample_step = 0

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((shape[0],), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                x=img, t=time_cond, cond=cond, x_self_cond=self_cond, clip_x_start=clip
            )

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            if save_every and ((sample_step + 1) % save_every == 0):
                imgs.append(img)

            sample_step += 1

        ret = img if not save_every else imgs
        # ret = self.unnormalize(ret)

        return ret

    @torch.inference_mode()
    def sample(self, shape, freq=0, cond=None, clip=False, hint=None, *args, **kwargs):
        steps = min(self.num_timesteps, self.timesteps_clip)
        if freq > 0 and freq <= 1:
            save_every = int(steps * freq)
            if save_every < steps:
                save_every = 1
        else:
            save_every = 0

        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        # savety batch size check
        if cond is not None:
            if shape[0] != cond.shape[0]:
                min_bs = min(shape[0], cond.shape[0])
                shape = (min_bs, *shape[1:])
                cond = cond[:min_bs] if cond is not None else None
        if hint is not None:
            if shape[0] != hint.shape[0]:
                min_bs = min(shape[0], hint.shape[0])
                shape = (min_bs, *shape[1:])
                hint = hint[:min_bs] if hint is not None else None
                cond = cond[:min_bs] if cond is not None else None

        return sample_fn(
            shape,
            cond=cond,
            hint=hint,
            save_every=save_every,
            clip=clip,
            *args,
            **kwargs,
        )

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc="interpolation sample time step", total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None, return_alphas_simgas=False):
        noise = default(noise, lambda: torch.randn_like(x_start))
        alphas = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sigmas = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        noised_x = alphas * x_start + sigmas * noise
        if return_alphas_simgas:
            return noised_x, alphas, sigmas
        else:
            return noised_x

    def p_losses(self, x_start, t, cond=None, noise=None, offset_noise_strength=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, "b c -> b c 1 1")

        # noise sample
        x, alphas, sigmas = self.q_sample(x_start=x_start, t=t, noise=noise, return_alphas_simgas=True)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, cond=cond).pred_x_start
                x_self_cond.detach_()

        with autocast(enabled=self.amp):
            model_out = self.model(x, t, cond=cond, x_self_cond=x_self_cond)

            if self.objective == "pred_noise":
                target = noise
            elif self.objective == "pred_x0":
                target = x_start
            elif self.objective == "pred_v":
                v = self.predict_v(x_start, t, noise)
                target = v
            else:
                raise ValueError(f"unknown objective {self.objective}")

            mse_loss = F.mse_loss(model_out, target, reduction="none")
            mse_loss = reduce(mse_loss, "b ... -> b", "mean")
            mse_loss = mse_loss * extract(self.loss_weight, t, mse_loss.shape)
            mse_loss = mse_loss.mean()
            loss = mse_loss

            if self.reg_scale > 0:
                # calculate additional projectiom loss on the x0
                model_pred_noise, model_pred_x0 = self.to_noise_and_xstart(
                    model_out, x, t, clip_x_start=False, rederive_pred_noise=False
                )
                proj_loss = projection_loss(x_start, model_pred_x0)
                proj_loss_scale = get_scaling(alphas / sigmas)
                proj_loss_scale = torch.where(t >= 800, torch.zeros_like(proj_loss_scale), proj_loss_scale)
                proj_loss = proj_loss * proj_loss_scale
                proj_loss = proj_loss.mean()
                loss += self.reg_scale * proj_loss

        return loss

    def forward(self, input, cond=None, *args, **kwargs):
        B, D, N, device = *input.shape, input.device

        # sample timestep randomly from [0,T] or [0,T_clip]. The second option is to keep the noisy sample close the the target
        t = torch.randint(0, min(self.num_timesteps, self.timesteps_clip), (B,), device=device).long()

        return self.p_losses(x_start=input, t=t, cond=cond, *args, **kwargs)
