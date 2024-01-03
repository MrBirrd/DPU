import torch
from einops import reduce
from torch.nn.functional import l1_loss, mse_loss

from metrics.emd_ import emd_module as EMD


def mean_squared_error(pred, gt):
    loss = mse_loss(pred, gt, reduction="none")
    loss = reduce(loss, "b ... -> b", "mean")
    return loss


def l1(pred, gt):
    loss = l1_loss(pred, gt, reduction="none")
    loss = reduce(loss, "b ... -> b", "mean")
    return loss


class EmdLoss:
    def __init__(self):
        self.emd = EMD.emdModule()

    def __call__(self, pred, gt):
        # make shure the pointclouds are in the right shape
        if pred.shape[-1] != 3:
            pred = pred.transpose(1, 2)
        if gt.shape[-1] != 3:
            gt = gt.transpose(1, 2)
        # calculate approximate EMD
        distances, _ = self.emd(pred, gt, eps=0.005, iters=50)
        # calculate L2 distance by taking the square root
        loss = torch.sqrt(distances)
        loss = reduce(loss, "b ... -> b", "mean")
        return loss


def get_loss(type="mse"):
    if type == "mse":
        return mean_squared_error
    if type == "l1":
        return l1
    if type == "emd":
        return EmdLoss()
