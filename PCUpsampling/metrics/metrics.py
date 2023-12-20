import numpy as np
import torch
from loguru import logger
from point_cloud_utils import chamfer_distance
from torch import Tensor


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


def calculate_cd(pred, gt):
    cds = []

    try:
        for x_pred, x_gt in zip(pred, gt):
            cd = chamfer_distance(
                x_pred.cpu().permute(1, 0).numpy(),
                x_gt.cpu().permute(1, 0).numpy(),
            )
            cds.append(cd)

        cd = np.mean(cds)

    except:
        # switch row major to col major
        xgnp = pred.cpu().numpy()
        xgnp = np.asfortranarray(xgnp)
        pred = torch.from_numpy(xgnp).cuda()
        # evaluate again
        for x_pred, x_gt in zip(pred, gt):
            print(x_pred.shape, x_gt.shape)
            cd = chamfer_distance(
                x_pred.cpu().permute(1, 0).numpy(),
                x_gt.cpu().permute(1, 0).numpy(),
            )

            cds.append(cd)

        cd = np.mean(cds)
    return cd
