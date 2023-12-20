from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List
from torch import nn, Tensor


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
