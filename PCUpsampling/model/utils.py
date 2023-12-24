from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from einops import rearrange
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
