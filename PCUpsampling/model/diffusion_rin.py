import math
from collections import namedtuple
from functools import partial, wraps
from random import random
import torch
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange, reduce, repeat
from torch import einsum, nn
from torch.special import expm1
from torchvision import transforms as T
from torchvision import utils
from tqdm.auto import tqdm
from .rin import RIN

FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers functions
def exists(x):
    return x is not None

def identity(x):
    return x

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def safe_div(numer, denom, eps = 1e-10):
    return numer / denom.clamp(min = eps)

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    num_sqrt = math.sqrt(num)
    return int(num_sqrt) == num_sqrt

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# normalize and unnormalize image

def normalize_batch(x):
    return x * 2 - 1

def unnormalize_batch(x):
    return (x + 1) * 0.5

# normalize variance of noised image, if scale is not 1

def normalize_pc_variance(x, eps = 1e-5):
    std = reduce(x, 'b n c -> b 1 1', partial(torch.std, unbiased = False))
    return x / std.clamp(min = eps)

# helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# noise schedules
def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

# converting gamma to alpha, sigma or logsnr

def gamma_to_alpha_sigma(gamma, scale = 1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

def gamma_to_log_snr(gamma, scale = 1, eps = 1e-5):
    return log(gamma * (scale ** 2) / (1 - gamma), eps = eps)

# gaussian diffusion

@beartype
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        cfg,
        schedule_kwargs: dict = dict(),
        time_difference = 0.,
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        train_prob_self_cond = 0.9,
        scale = 1.                      # this will be set to < 1. for better convergence when training on higher resolution images
    ):
        super().__init__()

        self.model = RIN(
            in_dim=cfg.model.in_dim*2 + cfg.model.extra_feature_channels,
            out_dim=cfg.model.out_dim,
            lift_dimension=128,
            depth=4,
            latent_self_attn_depth=2,
            dim_latent=256,
            num_latents=256,
            flash=True,
            )
        
        # extract config
        timesteps = cfg.diffusion.timesteps
        npoints = cfg.data.npoints
        use_ddim = cfg.diffusion.sampling_strategy == "DDIM"
        noise_schedule = cfg.diffusion.schedule
        cfg_convert = {'pred_v': 'v', 'pred_noise': 'eps', 'pred_x0': 'x0'}
        objective = cfg.diffusion.objective
        objective = cfg_convert[objective]
        self.out_dim = self.model.out_dim

        assert objective in {'x0', 'eps', 'v'}, 'objective must be either predict x0 or noise'
        self.objective = objective

        self.npoints = npoints

        if noise_schedule == "linear":
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == "cosine":
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == "sigmoid":
            self.gamma_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        # the main finding presented in Ting Chen's paper - that higher resolution images requires more noise for better training

        assert scale <= 1, 'scale must be less than or equal to 1'
        self.scale = scale
        self.maybe_normalize_variance = normalize_pc_variance if scale < 1 else identity

        # gamma schedules

        self.gamma_schedule = partial(self.gamma_schedule, **schedule_kwargs)

        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400

        self.time_difference = time_difference

        # probability for self conditioning during training

        self.train_prob_self_cond = train_prob_self_cond

        # min snr loss weight

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)
        
    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, shape, cond=None, time_difference = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        img = torch.randn(shape, device=device)

        x_start = None
        last_latents = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.timesteps):

            # add the time delay

            time_next = (time_next - self.time_difference).clamp(min = 0.)

            noise_cond = time

            # get predicted x0

            maybe_normalized_img = self.maybe_normalize_variance(img)
            model_output, last_latents = self.model(maybe_normalized_img, noise_cond, cond=cond, x_self_cond=x_start, latent_self_cond=last_latents, return_latents = True)

            # get log(snr)

            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)
            gamma, gamma_next = map(partial(right_pad_dims_to, img), (gamma, gamma_next))

            # get alpha sigma of time and next time

            alpha, sigma = gamma_to_alpha_sigma(gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next, self.scale)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = safe_div(img - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * img - sigma * model_output

            # clip x0

            x_start.clamp_(-1., 1.)

            # derive posterior mean and variance

            log_snr, log_snr_next = map(gamma_to_log_snr, (gamma, gamma_next))

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise

            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            img = mean + (0.5 * log_variance).exp() * noise

        return unnormalize_batch(img)

    @torch.no_grad()
    def ddim_sample(self, shape, cond = None, sample_every = 0, time_difference = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        img = torch.randn(shape, device = device)

        x_start = None
        last_latents = None
        samples_list = []
        sample_step = 0
        
        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step'):

            # get times and noise levels

            gamma = self.gamma_schedule(times)
            gamma_next = self.gamma_schedule(times_next)

            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, img), (gamma, gamma_next))

            alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next, self.scale)

            # add the time delay

            times_next = (times_next - time_difference).clamp(min = 0.)

            # predict x0

            maybe_normalized_img = self.maybe_normalize_variance(img)
            model_output, last_latents = self.model(maybe_normalized_img, times, cond=cond, x_self_cond=x_start, latent_self_cond=last_latents, return_latents = True)

            # calculate x0 and noise

            if self.objective == 'x0':
                x_start = model_output

            elif self.objective == 'eps':
                x_start = safe_div(img - sigma * model_output, alpha)

            elif self.objective == 'v':
                x_start = alpha * img - sigma * model_output

            # clip x0

            x_start.clamp_(-1., 1.)

            # get predicted noise

            pred_noise = safe_div(img - alpha * x_start, sigma)

            # calculate x next

            img = x_start * alpha_next + pred_noise * sigma_next
            
            samples_list.append(rearrange(unnormalize_batch(img), 'b n c -> b c n'))
            sample_step += 1
        
        ret = unnormalize_batch(img) if sample_every == 0 else samples_list[::sample_every]
        ret = rearrange(ret, 'b n c-> b c n') if sample_every == 0 else ret
        return ret

    @torch.no_grad()
    def sample(self, shape = 16, cond = None, freq = 0, *args, **kwargs):
        n, d = self.npoints, self.out_dim
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        sample_every = int(self.timesteps / (self.timesteps * (1 -freq))) if freq > 0 else 0
        return sample_fn((shape[0], d, n), cond=cond, sample_every=sample_every)

    def forward(self, x, cond=None, *args, **kwargs):
        batch, n, d, device, = *x.shape, x.device

        x = rearrange(x, 'b c n -> b n c')
        cond = rearrange(cond, 'b c n -> b n c') if exists(cond) else None

        # sample random times
        times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)

        # convert image to bit representation

        x = normalize_batch(x)

        # noise sample

        noise = torch.randn_like(x)

        gamma = self.gamma_schedule(times)
        padded_gamma = right_pad_dims_to(x, gamma)
        alpha, sigma =  gamma_to_alpha_sigma(padded_gamma, self.scale)

        noised_x = alpha * x + sigma * noise

        noised_x = self.maybe_normalize_variance(noised_x)

        # in the paper, they had to use a really high probability of latent self conditioning, up to 90% of the time
        # slight drawback

        self_cond = self_latents = None

        if random() < self.train_prob_self_cond:
            with torch.no_grad():
                model_output, self_latents = self.model(noised_x, times, cond=cond, return_latents = True)
                self_latents = self_latents.detach()

                if self.objective == 'x0':
                    self_cond = model_output

                elif self.objective == 'eps':
                    self_cond = safe_div(noised_x - sigma * model_output, alpha)

                elif self.objective == 'v':
                    self_cond = alpha * noised_x - sigma * model_output

                self_cond.clamp_(-1., 1.)
                self_cond = self_cond.detach()

        # predict and take gradient step

        pred = self.model(noised_x, times, cond=cond, x_self_cond=self_cond, latent_self_cond=self_latents)

        if self.objective == 'eps':
            target = noise

        elif self.objective == 'x0':
            target = x

        elif self.objective == 'v':
            target = alpha * noise - sigma * x

        loss = F.mse_loss(pred, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # min snr loss weight

        snr = (alpha * alpha) / (sigma * sigma)
        maybe_clipped_snr = snr.clone()

        if self.min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = self.min_snr_gamma)

        if self.objective == 'eps':
            loss_weight = maybe_clipped_snr / snr

        elif self.objective == 'x0':
            loss_weight = maybe_clipped_snr

        elif self.objective == 'v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        return (loss * loss_weight).mean()