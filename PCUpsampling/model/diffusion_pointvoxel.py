import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from .unet_mink import MinkUnet
from .unet_pointvoxel import PVCLion
import math


class PVD(nn.Module):
    def __init__(
        self, args, loss_type: str, model_mean_type: str, model_var_type: str, device=None,
    ):
        super(PVD, self).__init__()
        betas = get_betas(args.diffusion.schedule, args.diffusion.beta_start, args.diffusion.beta_end, args.diffusion.timesteps)
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type, device=device, amp=args.training.amp)
        self.num_sample_steps = args.diffusion.sampling_timesteps
        self.ddim = args.diffusion.sampling_strategy == "DDIM"

        if args.model.type == "PVD":
            self.model = PVCLion(
                out_dim=args.data.nc,
                input_dim=args.data.nc,
                npoints=args.data.npoints,
                embed_dim=args.model.time_embed_dim,
                use_att=args.model.use_attention,
                dropout=args.model.dropout,
                extra_feature_channels=args.model.extra_feature_channels,
            )
        elif args.model.type == "Mink":
            self.model = MinkUnet(
                dim=args.model.time_embed_dim, D=1, out_dim=args.model.out_dim, in_channels=args.model.in_dim + args.model.extra_feature_channels, dim_mults=(1, 2, 4, 8), use_attention=args.model.use_attention
            )
        else:
            raise NotImplementedError(args.model.type)

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, cond=None, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt = self.diffusion.calc_bpd_loop(
            self._denoise, x0, cond=cond, clip_denoised=clip_denoised
        )

        return {
            "total_bpd_b": total_bpd_b,
            "terms_bpd": vals_bt,
            "prior_bpd_b": prior_bpd_b,
            "mse_bt": mse_bt,
        }

    def _denoise(self, data, t, cond=None):
        B, D, N = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        if cond is not None:
            data = torch.cat([data, cond], dim=1)
        out = self.model(data, t)

        assert out.shape == torch.Size([B, D, N])
        return out

    def forward(self, data, cond=None, noises=None):
        B, D, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t != 0] = torch.randn((t != 0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(denoise_fn=self._denoise, data_start=data, t=t, cond=cond, noise=noises)
        assert losses.shape == t.shape == torch.Size([B])
        return losses.mean()

    def sample(
        self,
        shape,
        device=None,
        freq=0,
        noise_fn=torch.randn,
        cond=None,
        clip_denoised=True,
        keep_running=False,
    ):
        if freq == 0:
            if self.ddim:
                return self.diffusion.ddim_sample(
                    shape=shape,
                    denoise_fn=self._denoise,
                    cond=cond,
                    clip=False,
                )
            else:
                return self.diffusion.p_sample_loop(
                    self._denoise,
                    shape=shape,
                    device=device,
                    noise_fn=noise_fn,
                    cond=cond,
                    clip_denoised=clip_denoised,
                    keep_running=keep_running,
                )
        elif freq > 0 and freq <= 1:
            save_every = int(self.num_sample_steps * freq)
            if self.ddim:
                return self.diffusion.ddim_sample(
                    shape=shape,
                    denoise_fn=self._denoise,
                    save_every=save_every,
                    cond=cond,
                    clip=False,
                )
            else:
                return self.diffusion.p_sample_loop_trajectory(
                    self._denoise,
                    shape=shape,
                    device=device,
                    noise_fn=noise_fn,
                    freq=save_every,
                    cond=cond,
                    clip_denoised=clip_denoised,
                    keep_running=keep_running,
                )

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == "linear":
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == "warm0.1":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == "warm0.2":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == "warm0.5":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == "sigmoid":
        betas = sigmoid_beta_schedule(time_num,tau=1.).numpy()
    elif schedule_type == "cosine":
        betas = cosine_beta_schedule(time_num).numpy()
    else:
        raise NotImplementedError(schedule_type)
    return betas


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2) ** 2 * torch.exp(-logvar2)
    )


class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type, device = None, amp=False):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.amp = amp
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(
            np.float64
        )  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.device = device
        self.self_condition = False

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1.0 - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(
            np.append(1.0, alphas_cumprod[:-1])
        ).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance))
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        (bs,) = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = (
            self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape)
            * x_start
        )
        variance = self._extract(
            1.0 - self.alphas_cumprod.to(x_start.device), t, x_start.shape
        )
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape)
            * x_start
            + self._extract(
                self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape
            )
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape)
            * x_start
            + self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape)
            * x_t
        )
        posterior_variance = self._extract(
            self.posterior_variance.to(x_start.device), t, x_t.shape
        )
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_prediction(self, denoise_fn, data, t, cond=None, clip_denoised=False, return_xstart=True):
        model_output = denoise_fn(data, t, cond)

        if self.model_mean_type == "eps":
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -0.5, 0.5)
        else:
            raise NotImplementedError(self.loss_type)
        if return_xstart:
            return model_output, x_recon
        else:
            return model_output
    
    def p_mean_variance(
        self,
        denoise_fn,
        data,
        t,
        cond=None,
        clip_denoised: bool = False,
        return_pred_xstart: bool = False,
    ):
        model_output = denoise_fn(data, t, cond)

        if self.model_var_type in ["fixedsmall", "fixedlarge"]:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                "fixedlarge": (
                    self.betas.to(data.device),
                    torch.log(
                        torch.cat([self.posterior_variance[1:2], self.betas[1:]])
                    ).to(data.device),
                ),
                "fixedsmall": (
                    self.posterior_variance.to(data.device),
                    self.posterior_log_variance_clipped.to(data.device),
                ),
            }[self.model_var_type]
            model_variance = self._extract(
                model_variance, t, data.shape
            ) * torch.ones_like(data)
            model_log_variance = self._extract(
                model_log_variance, t, data.shape
            ) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == "eps":
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -0.5, 0.5)

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=x_recon, x_t=data, t=t
            )
        else:
            raise NotImplementedError(self.loss_type)

        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape)
            * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape)
            * eps
        )

    """ samples """
    @torch.inference_mode()
    def ddim_sample(self, shape, denoise_fn, cond = None, save_every = 0, sampling_timesteps=50, clip = False, *args, **kwargs):
        batch, total_timesteps, eta = shape[0], self.num_timesteps, 0.

        device = "cuda" if self.device is None else self.device

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None
        # set saving
        imgs = [img]
        sample_step = 0
        
        for time, time_next in tqdm(time_pairs, desc = 'DDIM sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start = self.model_prediction(denoise_fn=denoise_fn, data=img, t=time_cond, cond=cond)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            if save_every and ((sample_step + 1) % save_every == 0):
                imgs.append(img)
            
            sample_step += 1

        ret = img if not save_every else imgs
        #ret = self.unnormalize(ret)

        return ret
    
    def p_sample(
        self,
        denoise_fn,
        data,
        t,
        noise_fn,
        cond=None,
        clip_denoised=False,
        return_pred_xstart=False,
    ):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn,
            data=data,
            t=t,
            cond=cond,
            clip_denoised=clip_denoised,
            return_pred_xstart=True,
        )
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample

    def p_sample_loop(
        self,
        denoise_fn,
        shape,
        device,
        cond=None,
        noise_fn=torch.randn,
        clip_denoised=True,
        keep_running=False,
        use_tqdm=True,
    ):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)

        for t in tqdm(
            reversed(
                range(0, self.num_timesteps if not keep_running else len(self.betas))
            ),
            desc="DDPM sampling",
            total=self.num_timesteps if not keep_running else len(self.betas),
            disable=not use_tqdm,
        ):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(
                denoise_fn=denoise_fn,
                data=img_t,
                t=t_,
                noise_fn=noise_fn,
                cond=cond,
                clip_denoised=clip_denoised,
                return_pred_xstart=False,
            )

        assert img_t.shape == shape
        return img_t

    def p_sample_loop_trajectory(
        self,
        denoise_fn,
        shape,
        device,
        freq,
        cond=None,
        noise_fn=torch.randn,
        clip_denoised=True,
        keep_running=False,
        use_tqdm=True,
    ):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps = self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in tqdm(
            reversed(range(0, total_steps)),
            desc="p_sample_loop_trajectory",
            disable=not use_tqdm,
        ):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(
                denoise_fn=denoise_fn,
                data=img_t,
                t=t_,
                noise_fn=noise_fn,
                cond=cond,
                clip_denoised=clip_denoised,
                return_pred_xstart=False,
            )
            if t % freq == 0 or t == total_steps - 1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs


    """losses"""

    def _vb_terms_bpd(
        self,
        denoise_fn,
        data_start,
        data_t,
        t,
        cond=None,
        clip_denoised: bool = False,
        return_pred_xstart: bool = False,
    ):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=data_start, x_t=data_t, t=t
        )
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn,
            data=data_t,
            t=t,
            cond=cond,
            clip_denoised=clip_denoised,
            return_pred_xstart=True,
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, model_mean, model_log_variance
        )
        kl = kl.mean(dim=list(range(1, len(data_start.shape)))) / np.log(2.0)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, t, cond=None, noise=None):
        """
        Training loss calculation
        """
        B, D, N = data_start.shape
        assert t.shape == torch.Size([B])

        if noise is None:
            noise = torch.randn(
                data_start.shape, dtype=data_start.dtype, device=data_start.device
            )

        assert noise.shape == data_start.shape and noise.dtype == data_start.dtype

        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)

        if self.loss_type == "mse":
            with autocast(enabled=self.amp, dtype=torch.float16):
                eps_recon = denoise_fn(data_t, t, cond=cond)
                assert (
                    data_t.shape == data_start.shape
                    and eps_recon.shape == torch.Size([B, D, N])
                    and eps_recon.shape == data_start.shape
                )
                losses = ((noise - eps_recon) ** 2).mean(
                    dim=list(range(1, len(data_start.shape)))
                )
        elif self.loss_type == "kl":
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn,
                data_start=data_start,
                data_t=data_t,
                t=t,
                cond=cond,
                clip_denoised=False,
                return_pred_xstart=False,
            )
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])
        return losses

    """debug"""

    def _prior_bpd(self, x_start):
        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T - 1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(
                mean1=qt_mean,
                logvar1=qt_log_variance,
                mean2=torch.tensor([0.0]).to(qt_mean),
                logvar2=torch.tensor([0.0]).to(qt_log_variance),
            )
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.0)

    def calc_bpd_loop(self, denoise_fn, x_start, cond=None, clip_denoised=True):
        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_ = torch.zeros([B, T], device=x_start.device), torch.zeros(
                [B, T], device=x_start.device
            )
            for t in reversed(range(T)):
                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                new_vals_b, pred_xstart = self._vb_terms_bpd(
                    denoise_fn,
                    data_start=x_start,
                    data_t=self.q_sample(x_start=x_start, t=t_b),
                    t=t_b,
                    cond=cond,
                    clip_denoised=clip_denoised,
                    return_pred_xstart=True,
                )
                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start.shape
                new_mse_b = ((pred_xstart - x_start) ** 2).mean(
                    dim=list(range(1, len(x_start.shape)))
                )
                assert new_vals_b.shape == new_mse_b.shape == torch.Size([B])
                # Insert the calculated term into the tensor of all terms
                mask_bt = (
                    t_b[:, None] == torch.arange(T, device=t_b.device)[None, :].float()
                )
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert (
                    mask_bt.shape
                    == vals_bt_.shape
                    == vals_bt_.shape
                    == torch.Size([B, T])
                )

            prior_bpd_b = self._prior_bpd(x_start)
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size(
                [B, T]
            ) and total_bpd_b.shape == prior_bpd_b.shape == torch.Size([B])
            return (
                total_bpd_b.mean(),
                vals_bt_.mean(),
                prior_bpd_b.mean(),
                mse_bt_.mean(),
            )
