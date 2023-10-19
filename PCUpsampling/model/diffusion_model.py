import torch
from torch import nn

class DiffusionModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        raise NotImplementedError()

    def sample(self, x: torch.Tensor, t: int) -> torch.Tensor:
        raise NotImplementedError()