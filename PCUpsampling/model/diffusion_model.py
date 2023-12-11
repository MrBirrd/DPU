import torch
from torch import Tensor
from torch import nn
from abc import ABC, abstractmethod

class DiffusionModel(ABC, nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, x0: Tensor, x_start: Tensor = None, cond: Tensor = None,) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def sample(self, shape, cond=None, x_start=None, return_noised_hint=False, clip=False) -> Tensor:
        raise NotImplementedError()