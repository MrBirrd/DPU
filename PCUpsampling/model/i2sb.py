from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm
from torch import Tensor
from typing import Optional, Tuple, Union, List, Dict
from model.modules import DiffusionModel


def space_indices(num_steps: int, count: int):
    """
    Generate a list of indices that evenly space out over a given number of steps.

    Args:
        num_steps (int): The total number of steps.
        count (int): The number of indices to generate.

    Returns:
        list: A list of indices that evenly space out over the given number of steps.
    """
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps


def unsqueeze_xdim(z: Tensor, xdim: Tuple[int, ...]) -> Tensor:
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def compute_gaussian_product_coef(sigma1, sigma2):
    """Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
    return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var)"""

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
    return betas.numpy()


class I2SB(DiffusionModel):
    def __init__(self, cfg, model):
        super().__init__()
        # setup config
        device = cfg.gpu if cfg.gpu is not None else torch.device("cuda")
        self.device = device
        self.timesteps = cfg.diffusion.timesteps
        self.ot_ode = cfg.diffusion.ot_ode
        self.cfg = cfg

        # load model
        self.model = model.to(device)
        # setup betas
        betas = make_beta_schedule(
            n_timestep=cfg.diffusion.timesteps, linear_end=cfg.diffusion.beta_max / cfg.diffusion.timesteps
        )
        betas = np.concatenate([betas[: cfg.diffusion.timesteps // 2], np.flip(betas[: cfg.diffusion.timesteps // 2])])

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """Given network output, recover x0. This should be the inverse of Eq 12"""
        std_fwd = self.get_std_fwd(step, xdim=xt.shape[1:])

        if std_fwd.ndim != net_out.ndim:
            std_fwd = std_fwd.squeeze(-1)

        pred_x0 = xt - std_fwd * net_out

        if clip_denoise:
            pred_x0.clamp_(-1.0, 1.0)
        return pred_x0

    def compute_gt(self, step, x0, xt):
        """Eq 12"""
        std_fwd = self.get_std_fwd(step, xdim=x0.shape[1:])
        gt = (xt - x0) / std_fwd
        return gt.detach()

    def q_sample(self, step, x0, x1, ot_ode=False):
        """Sample q(x_t | x_0, x_1), i.e. eq 11"""

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0 = unsqueeze_xdim(self.mu_x0[step], xdim)
        mu_x1 = unsqueeze_xdim(self.mu_x1[step], xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False):
        """Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def sample_ddpm(self, steps, pred_x0_fn, x1, cond=None, features=None, ot_ode=False, log_steps=None, verbose=True):
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc="DDPM sampling", total=len(steps) - 1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"
            pred_x0 = pred_x0_fn(xt, step, features=features, cond=cond)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)

    @torch.no_grad()
    def ddpm_sampling(self, x1, cond=None, features=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):
        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or self.timesteps - 1
        assert 0 < nfe < self.timesteps == len(self.betas)
        steps = space_indices(self.timesteps, nfe + 1)

        # create log steps
        log_count = min(len(steps) - 1, log_count)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0
        logger.info(f"[DDPM Sampling] steps={self.timesteps}, {nfe=}, {log_steps=}!")

        x1 = x1.to(self.device)
        if cond is not None:
            cond = cond.to(self.device)

        self.model.eval()

        def pred_x0_fn(xt, step, features=None, cond=None):
            step = torch.full((xt.shape[0],), step, device=self.device, dtype=torch.long)

            step_features = None
            xt_cond = None

            if features is not None:
                step_features = torch.cat([step.unsqueeze(-1), features.unsqueeze(-1)], dim=1)
            if cond is not None:
                xt_cond = torch.cat([xt, cond], dim=1)

            out = self.model(xt if xt_cond is None else xt_cond, step if step_features is None else step_features)
            return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

        xs, pred_x0 = self.sample_ddpm(
            steps,
            pred_x0_fn,
            x1,
            cond=cond,
            features=features,
            ot_ode=self.ot_ode,
            log_steps=log_steps,
            verbose=verbose,
        )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        self.model.train()
        return xs, pred_x0

    @torch.no_grad()
    def sample(
        self,
        features: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
        x1: Optional[Tensor] = None,
        clip: bool = False,
        *args,
        **kwargs,
    ) -> Dict:
        if self.cfg.diffusion.sampling_strategy == "DDPM":
            xs, _ = self.ddpm_sampling(
                x1=x1,
                features=features,
                cond=cond,
                clip_denoise=clip,
                nfe=self.cfg.diffusion.sampling_timesteps,
                verbose=False,
            )
            data = {
                "x_chain": xs,
                "x_pred": xs[:, 0, ...],
                "x_start": x1,
            }
            return data
        else:
            raise NotImplementedError()

    def loss(self, pred: Tensor, gt: Tensor) -> Tensor:
        pred = pred.to(self.device)
        gt = gt.to(self.device)
        return F.mse_loss(pred, gt)

    def forward(
        self, x0: Tensor, x1: Tensor, cond: Optional[Tensor] = None, features: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        step = torch.randint(0, self.timesteps, (x0.shape[0],)).to(self.device)
        xt = self.q_sample(step, x0, x1, ot_ode=self.ot_ode)
        gt = self.compute_gt(step, x0, xt)

        # timestep features
        if features is not None:
            step = torch.cat([step.unsqueeze(-1), features.unsqueeze(-1)], dim=1)

        # conditional features
        if cond is not None:
            xt = torch.cat([xt, cond], dim=1)

        pred = self.model(xt, step)

        loss = F.mse_loss(pred, gt)
        return loss
