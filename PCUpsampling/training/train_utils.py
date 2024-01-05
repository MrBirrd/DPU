import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from loguru import logger
from torch import Tensor, nn


class DiffusionModel(ABC, nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(
        self,
        x0: Tensor,
        x_start: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, shape, cond=None, x_start=None, return_noised_hint=False, clip=False) -> Tensor:
        raise NotImplementedError()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


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


def set_seed(opt):
    if opt.training.seed is None:
        opt.training.seed = 42

    # different seed per gpu
    opt.training.seed += opt.rank

    logger.info("Random Seed: {}", opt.training.seed)
    random.seed(opt.training.seed)
    torch.manual_seed(opt.training.seed)
    np.random.seed(opt.training.seed)
    if opt.gpu is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.training.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def to_cuda(data, device) -> Union[Tensor, List, Tuple, Dict, None]:
    """
    Moves the input data to the specified device (GPU) if available.

    Args:
        data: The input data to be moved to the device.
        device: The device (GPU) to move the data to.

    Returns:
        The input data moved to the specified device.

    """
    if data is None:
        return None
    if isinstance(data, (list, tuple)):
        return [to_cuda(d, device) for d in data]
    if isinstance(data, dict):
        return {k: to_cuda(v, device) for k, v in data.items()}
    if device is None:
        return data.cuda(non_blocking=True)
    else:
        return data.to(device, non_blocking=True)


def ensure_size(x: Tensor) -> Tensor:
    """
    Ensures that the input tensor has the correct size and dimensions.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The input tensor with the correct size and dimensions.
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)
    assert x.dim() == 3
    if x.size(1) > x.size(2):
        x = x.transpose(1, 2)
    return x


def get_data_batch(batch, cfg):
    """
    Get a batch of data for training or testing.

    Args:
        batch (dict): A dictionary containing the batch data.
        cfg (dict): A dictionary containing the configuration settings.

    Returns:
        dict: A dictionary containing the processed batch data.
    """
    hr_points = batch["hr_points"].transpose(1, 2)

    # load conditioning features
    if not cfg.data.unconditional:
        features = batch["features"] if "features" in batch else None
        lr_points = batch["lr_points"] if "lr_points" in batch else None
    else:
        features, lr_points = None, None

    hr_points = ensure_size(hr_points)

    features = ensure_size(features) if features is not None else None
    lr_points = ensure_size(lr_points) if lr_points is not None else None

    lr_colors = ensure_size(batch["lr_colors"]) if "lr_colors" in batch else None
    hr_colors = ensure_size(batch["hr_colors"]) if "hr_colors" in batch else None

    # concatenate colors to features
    if lr_colors is not None and lr_colors.shape[-1] > 0 and cfg.data.use_rgb_features:
        features = torch.cat([lr_colors, features], dim=1) if features is not None else lr_colors

    # unconditionals training (no features) at all
    if cfg.data.unconditional:
        features = None
        lr_points = None

    if lr_points is not None:
        assert hr_points.shape == lr_points.shape

    return {
        "hr_points": hr_points,
        "lr_points": lr_points,
        "features": features,
    }


def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p**2) for p in net.parameters() if p.requires_grad))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad**2) for p in net.parameters() if p.requires_grad))
    return pNorm, gradNorm
