from math import sqrt
from random import random

import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import einsum, nn
from tqdm import tqdm
import math
from torch import Tensor

try:
    from .unet_mink import MinkUnet
except:
    pass
from .unet_pointvoxel import PVCLionSmall
from gecco_torch.models.linear_lift import LinearLift
from gecco_torch.models.set_transformer import SetTransformer

# from .unet_torchsparse import TSUnet
# helpers


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# tensor helpers


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def ones(n: int):
    return (1,) * n


class LogUniformSchedule(nn.Module):
    """
    LogUniform noise schedule which seems to work better in our (GECCO) context.
    """

    def __init__(self, max: float, min: float = 0.002, low_discrepancy: bool = True):
        super().__init__()

        self.sigma_min = min
        self.sigma_max = max
        self.log_sigma_min = math.log(min)
        self.log_sigma_max = math.log(max)
        self.low_discrepancy = low_discrepancy

    def extra_repr(self) -> str:
        return f"sigma_min={self.sigma_min}, sigma_max={self.sigma_max}, low_discrepancy={self.low_discrepancy}"

    def forward(self, data: Tensor) -> Tensor:
        u = torch.rand(data.shape[0], device=data.device)

        if self.low_discrepancy:
            div = 1 / data.shape[0]
            u = div * u
            u = u + div * torch.arange(data.shape[0], device=data.device)

        sigma = (u * (self.log_sigma_max - self.log_sigma_min) + self.log_sigma_min).exp()
        return sigma.reshape(-1, *ones(data.ndim - 1))


# main class
class ElucidatedDiffusion(nn.Module):
    def __init__(
        self,
        args,
        sigma_min=0.002,  # min noise level
        sigma_max=80,  # max noise level
        sigma_data=0.5,  # standard deviation of data distribution
        rho=7,  # controls the sampling schedule
        P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std=1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin=0.05,
        S_tmax=50,
        S_noise=1.003,
    ):
        super().__init__()

        # setup networks
        if args.model.type == "PVD":
            self.net = PVCLionSmall(
                out_dim=args.data.nc,
                input_dim=args.data.nc,
                npoints=args.data.npoints,
                embed_dim=args.model.time_embed_dim,
                use_att=args.model.use_attention,
                dropout=args.model.dropout,
                extra_feature_channels=args.model.extra_feature_channels,
            )
        elif args.model.type == "Mink":
            self.net = MinkUnet(
                dim=args.model.time_embed_dim,
                D=1,
                out_dim=args.model.out_dim,
                in_channels=args.model.in_dim + args.model.extra_feature_channels,
                dim_mults=(1, 2, 4, 8),
                use_attention=args.model.use_attention,
                init_ds_factor=2,
            )
        elif args.model.type == "SetTransformer":
            set_transformer = SetTransformer(
                n_layers=args.model.ST.layers,
                feature_dim=args.model.ST.fdim,
                num_inducers=args.model.SR.inducers,
                t_embed_dim=1,
            ).cuda()
            self.net = LinearLift(
                inner=set_transformer,
                feature_dim=args.model.ST.fdim,
                in_dim=args.model.in_dim + args.model.extra_feature_channels,
                out_dim=args.model.out_dim,
            ).cuda()
        else:
            raise NotImplementedError(args.model.type)

        # dimensions
        self.channels = args.data.nc
        self.npoints = args.data.npoints

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = args.diffusion.sampling_timesteps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.net.parameters()).device

    # methods for training script
    def multi_gpu_wrapper(self, f):
        self.net = f(self.net)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma**2 + self.sigma_data**2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(self, noised_images, sigma, cond=None, clamp=False):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        net_in = self.c_in(padded_sigma) * noised_images
        cond = self.c_in(padded_sigma) * cond if exists(cond) else None

        net_out = self.net(
            net_in,  # the data (can have more than 3 channels)
            self.c_noise(sigma),  # time cond
            cond=cond,  #
        )

        out = self.c_skip(padded_sigma) * noised_images + self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1.0, 1.0)

        return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(self, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (
            self.sigma_max**inv_rho + steps / (N - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def sample(self, shape, num_sample_steps=None, cond=None, clip_denoised=False, freq=0, **kwargs):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        if freq > 0 and freq <= 1:
            save_every = int(num_sample_steps * freq)
        else:
            save_every = None

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.0,
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # images is noise at the beginning

        init_sigma = sigmas[0]

        images = init_sigma * torch.randn(shape, device=self.device)

        # for saving intermediate steps
        output_list = []
        samp_idx = 0

        # gradually denoise
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc="sampling time step"):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device=self.device)  # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat**2 - sigma**2) * eps

            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, cond=cond, clamp=clip_denoised)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(
                    images_next, sigma_next, cond=cond, clamp=clip_denoised
                )
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (
                    denoised_over_sigma + denoised_prime_over_sigma
                )

            images = images_next

            if save_every is not None and samp_idx % save_every == 0:
                output_list.append(images)

            samp_idx += 1

        # images = images.clamp(-1., 1.)
        # images = unnormalize_to_zero_to_one(images)

        return images if freq == 0 else output_list

    @torch.no_grad()
    def sample_using_dpmpp(self, batch_size=16, cond=None, num_sample_steps=None):
        """
        thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095
        """

        device, num_sample_steps = self.device, default(num_sample_steps, self.num_sample_steps)

        sigmas = self.sample_schedule(num_sample_steps)

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        images = sigmas[0] * torch.randn(shape, device=device)

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            denoised = self.preconditioned_network_forward(images, sigmas[i].item(), cond=cond)
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = -1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            images = (sigma_fn(t_next) / sigma_fn(t)) * images - (-h).expm1() * denoised_d
            old_denoised = denoised

        images = images.clamp(-1.0, 1.0)
        return unnormalize_to_zero_to_one(images)

    # training

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()

    def forward(self, input, cond=None, noises=None):
        batch_size = input.shape[0]

        # images = normalize_to_neg_one_to_one(images)

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        noise = torch.randn_like(input)

        noised_input = input + padded_sigmas * noise  # alphas are 1. in the paper

        denoised = self.preconditioned_network_forward(noised_input, sigmas, cond=cond)

        losses = F.mse_loss(denoised, input, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()
