import math
from collections import namedtuple
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import Tensor
from torch.cuda.amp import autocast
from tqdm import tqdm

from training.dpm_sampler import DPM_Solver, NoiseScheduleVP, model_wrapper
from training.train_utils import (DiffusionModel, default,
                                  dynamic_threshold_percentile, extract,
                                  identity, normalize_to_neg_one_to_one,
                                  unnormalize_to_zero_to_one)
from utils.losses import projection_loss

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])

from loguru import logger

from .loss import get_loss

try:
    pass
except:
    logger.error("MinkUnet not found, please install MinkowskiEngine")


# gaussian diffusion trainer class
def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02) -> Tensor:
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * beta_start
    beta_end = scale * beta_end
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008) -> Tensor:
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


def adapted_sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02, a=10, b=-5):
    """
    Generate a sigmoid-shaped beta schedule for diffusion process.

    Args:
    - timesteps (int): Number of timesteps in the diffusion process.
    - beta_start (float): Starting noise level.
    - beta_end (float): Maximum noise level.
    - a (float): Steepness parameter for sigmoid function.
    - b (float): Center shift parameter for sigmoid function.

    Returns:
    - numpy array: Array of beta values for each timestep.
    """
    t = torch.linspace(0, timesteps, timesteps, dtype=torch.float64)
    sigmoid_function = 1 / (1 + torch.exp(-a * (t / timesteps) + b))
    beta_schedule = beta_start + (beta_end - beta_start) * sigmoid_function
    return torch.clip(beta_schedule, 0, 0.999)  # Clipping to avoid very high values


class GaussianDiffusion(DiffusionModel):
    def __init__(
        self,
        cfg,
        model,
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
        self.cfg = cfg
        self.model = model

        # setup loss
        self.loss = get_loss(cfg.diffusion.loss_type)

        # dimensions
        self.channels = cfg.data.nc
        self.npoints = cfg.data.npoints
        self.amp = cfg.training.amp

        objective = cfg.diffusion.objective
        self.objective = objective
        self.dynamic_threshold = cfg.diffusion.dynamic_threshold
        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
            schedule_fn_kwargs["beta_start"] = cfg.diffusion.beta_start
            schedule_fn_kwargs["beta_end"] = cfg.diffusion.beta_end
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = adapted_sigmoid_beta_schedule
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

        self.sampling_timesteps = default(sampling_timesteps, timesteps)

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

    def model_conditioned(self, x, t, cond=None):
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
        return self.model(x, t)

    def model_predictions(
        self,
        x,
        t,
        cond=None,
        clip_x_start=False,
        rederive_pred_noise=True,
    ):
        model_output = self.model_conditioned(x, t, cond)

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

    def p_mean_variance(self, x, t, cond=None, clip_denoised=False):
        preds = self.model_predictions(
            x,
            t,
            cond=cond,
            clip_x_start=clip_denoised,
        )
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)
        elif self.dynamic_threshold:
            x_start = dynamic_threshold_percentile(x_start)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, cond=None, clip: bool = False):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, cond=cond, clip_denoised=clip
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def generate_x_start(self, shape, steps, x_start=None, add_hint_noise=False):
        # sample gaussian noise
        noise = torch.randn(shape, device=self.device)

        if x_start is None:
            x_start = noise
        elif add_hint_noise:
            with torch.no_grad():
                t = steps - 1
                t = torch.tensor([t], device=self.device).long()
                t = repeat(t, "1 -> b", b=shape[0])
                x_start, *_ = self.q_sample(x_start=x_start, noise=noise, t=t, return_alphas_simgas=True)
        return x_start

    @torch.inference_mode()
    def p_sample_loop(
        self,
        shape,
        x_start=None,
        add_hint_noise=True,
        return_noised_hint=False,
        save_every=0,
        cond=None,
        clip=False,
    ):
        total_steps = min(self.sampling_timesteps, self.timesteps_clip)

        x_start = self.generate_x_start(
            shape=shape,
            steps=self.num_timesteps,
            x_start=x_start,
            add_hint_noise=add_hint_noise,
        )

        x = x_start
        x_list = [x_start]

        sample_step = 0

        for t in tqdm(
            reversed(range(0, total_steps)),
            desc="sampling loop time step",
            total=total_steps,
        ):
            x = self.p_sample(x, t, cond=cond, clip=clip)[0]
            if save_every and ((sample_step + 1) % save_every == 0):
                x_list.append(x)
            sample_step += 1

        ret = x if not save_every else x_list

        if return_noised_hint:
            return ret, x_start
        else:
            return ret

    @torch.inference_mode()
    def sample(
        self,
        shape,
        freq=0,
        clip=False,
        x_start=None,
        cond=None,
        add_x_start_noise=True,
        return_noised_hint=False,
    ):
        # logging intermediate outputs
        steps = min(self.num_timesteps, self.timesteps_clip)

        if freq > 0 and freq <= 1:
            save_every = int(steps * freq)
            if save_every < steps:
                save_every = 1
        else:
            save_every = 0

        if self.cfg.diffusion.sampling_strategy == "DDPM":
            return self.p_sample_loop(
                shape,
                x_start=x_start,
                save_every=save_every,
                clip=clip,
                cond=cond,
                add_hint_noise=add_x_start_noise,
                return_noised_hint=return_noised_hint,
            )
        elif self.cfg.diffusion.sampling_strategy == "DPM++":
            logger.info("Using DPM++ sampling strategy")
            logger.info("Wrapping model with DPM++ solver")
            vp_schedule = NoiseScheduleVP(
                schedule="discrete",
                betas=self.betas,
            )
            wrapped_model = model_wrapper(
                model=self.model_conditioned,
                noise_schedule=vp_schedule,
                model_type="v",
                model_kwargs={"cond": cond},
            )
            dpm_solver = DPM_Solver(
                model_fn=wrapped_model,
                noise_schedule=vp_schedule,
                algorithm_type="dpmsolver++",
            )
            x_start = self.generate_x_start(
                shape=shape,
                steps=self.num_timesteps,
                x_start=x_start,
                add_hint_noise=add_x_start_noise,
            )
            logger.info("Sampling from DPM++ solver using {} steps".format(self.sampling_timesteps))
            out, chain = dpm_solver.sample(
                x=x_start.clone(),
                steps=self.sampling_timesteps,
                order=3,
                method="singlestep",
                denoise_to_zero=True,
                return_intermediate=True,
            )

            data = {
                "x_pred": out,
                "x_chain": chain,
                "x_start": x_start,
            }

            return data

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

    def p_losses(self, x_start, t, noise=None, cond=None, offset_noise_strength=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, "b c -> b c 1 1")

        # noise sample
        x, alphas, sigmas = self.q_sample(x_start=x_start, t=t, noise=noise, return_alphas_simgas=True)

        with autocast(enabled=self.amp):
            model_out = self.model_conditioned(x, t, cond)

            if self.objective == "pred_noise":
                target = noise
            elif self.objective == "pred_x0":
                target = x_start
            elif self.objective == "pred_v":
                v = self.predict_v(x_start, t, noise)
                target = v
            else:
                raise ValueError(f"unknown objective {self.objective}")

            loss = self.loss(model_out, target)
            loss = loss * extract(self.loss_weight, t, loss.shape)  # SNR weighted loss
            loss = loss.mean()
            total_loss = loss

            if self.reg_scale > 0:
                # calculate additional projectiom loss on the x0
                model_pred_noise, model_pred_x0 = self.to_noise_and_xstart(
                    model_out, x, t, clip_x_start=False, rederive_pred_noise=False
                )
                proj_loss = projection_loss(x_start, model_pred_x0)
                proj_loss = reduce(proj_loss, "b ... -> b", "mean")
                proj_loss_scale = torch.ones_like(t) - ((1 - 0.75) / self.timesteps_clip) * t
                proj_loss = proj_loss * proj_loss_scale
                proj_loss = proj_loss.mean()
                proj_loss = proj_loss * self.reg_scale
                total_loss += proj_loss

        return total_loss

    def forward(self, input, cond=None, *args, **kwargs):
        B, D, N, device = *input.shape, input.device

        # sample timestep randomly from [0,T] or [0,T_clip]. The second option is to keep the noisy sample close the the target
        t = torch.randint(0, min(self.num_timesteps, self.timesteps_clip), (B,), device=device).long()

        return self.p_losses(x_start=input, t=t, cond=cond, *args, **kwargs)
