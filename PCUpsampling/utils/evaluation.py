from torch import Tensor
import torch
from loguru import logger


def print_stats(x: Tensor, name: str):
    xmean = torch.mean(x)
    xstd = torch.std(x)
    xmin = torch.min(x)
    xmax = torch.max(x)
    logger.info(f"{name} mean: {xmean}, std: {xstd}, min: {xmin}, max: {xmax}")


def calculate_stats(x: Tensor):
    xmean = torch.mean(x)
    xstd = torch.std(x)
    xmin = torch.min(x)
    xmax = torch.max(x)
    stats = {"mean": xmean, "std": xstd, "min": xmin, "max": xmax}
    return stats
