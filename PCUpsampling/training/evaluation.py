import os
import numpy as np
import torch
from loguru import logger
import wandb
from metrics.emd_ import emd_module as EMD
from metrics.metrics import calculate_cd
from training.train_utils import get_data_batch, to_cuda
from utils.visualize import visualize_pointcloud_batch


def new_x_chain(x, num_chain):
    return torch.randn(num_chain, *x.shape[1:], device=x.device)


def save_visualizations(items, out_dir, step):
    for item in items:
        if item is None:
            continue
        ptc, name = item
        visualize_pointcloud_batch(
            "%s/%03d_%s.png" % (out_dir, step, name),
            ptc.transpose(1, 2),
        )


def log_wandb(name, out_dir, step):
    out_path = "%s/%03d_%s.png" % (out_dir, step, name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wandb_img = wandb.Image(out_path)
    wandb.log({name: wandb_img}, step=step)


def save_ptc(name, ptc, out_dir, step):
    np.save("%s/%03d_%s.npy" % (out_dir, step, name), ptc.cpu().numpy())


def evaluate(model, eval_iter, cfg, step, sampling=False, save_npy=False, debug=False):
    if sampling:
        out_dir = cfg.out_sampling
    else:
        out_dir = cfg.outf_syn

    eval_data = next(eval_iter)
    eval_data = to_cuda(eval_data, cfg.gpu)

    data_batch = get_data_batch(batch=eval_data, cfg=cfg)
    x0 = data_batch["hr_points"]
    x1 = data_batch["lr_points"] if data_batch["lr_points"] is not None else None
    features = data_batch["features"] if data_batch["features"] is not None else None

    # get the right x_start
    if cfg.diffusion.sampling_hint:
        x_start = x0
    elif cfg.diffusion.formulation.lower() == "i2sb":
        x_start = x1
    else:
        x_start = None

    with torch.no_grad():
        sample_data = model.sample(
            shape=x0.shape,
            cond=features,
            x_start=x_start,
            clip=False,
        )

    pred = sample_data["x_pred"]
    chain = sample_data["x_chain"][0]
    x_start = sample_data["x_start"]

    # visualize the pointclouds
    save_visualizations(
        [
            (pred, "pred"),
            (x_start, "start") if x_start is not None else None,
            (x0, "gt"),
            (chain, "chain"),
        ],
        out_dir,
        step,
    )

    # calculate stats
    # subsample cloud to closest multiple of 128
    n_points = pred.shape[-1]
    n_points = n_points - n_points % 128

    pred = pred[..., :n_points]
    x0 = x0[..., :n_points]
    x_start = x_start[..., :n_points] if x_start is not None else None

    cd, emd, eval_loss = get_metrics(model, x0, pred)
    # print the stats
    batch_metrics = {
        "CD": cd,
        "EMD": emd,
        "eval_loss_unweighted": eval_loss,
    }

    if x_start is not None:
        cd_hint, emd_hint, eval_loss_hint = get_metrics(model, x0, x_start)
        batch_metrics["CD_hint"] = cd_hint
        batch_metrics["EMD_hint"] = emd_hint
        batch_metrics["eval_loss_hint_unweighted"] = eval_loss_hint

    logger.info(batch_metrics)

    if not sampling:
        wandb.log(batch_metrics, step=step)
        log_wandb("pred", out_dir, step)
        log_wandb("start", out_dir, step)
        log_wandb("gt", out_dir, step)
    elif save_npy:
        save_ptc("pred", pred, out_dir, step)
        save_ptc("start", x_start, out_dir, step)
        save_ptc("gt", x0, out_dir, step)

    return batch_metrics


def get_metrics(model, gt, pred):
    """
    Calculate evaluation metrics for the given model predictions.

    Args:
        model (torch.nn.Module): The model used for predictions.
        gt (torch.Tensor): Ground truth point cloud.
        pred (torch.Tensor): Predicted point cloud.

    Returns:
        tuple: A tuple containing the evaluation metrics (cd, emd, eval_loss).
            - cd (float): Chamfer distance between the ground truth and predicted point clouds.
            - emd (float): Earth Mover's Distance between the ground truth and predicted point clouds.
            - eval_loss (float): Evaluation loss of the model predictions.
    """
    emd = EMD.emdModule()
    cd = calculate_cd(pred, gt)
    eval_loss = model.loss(pred, gt).mean().item()
    distance, _ = emd(gt.permute(0, 2, 1), pred.permute(0, 2, 1), 0.05, 3000)
    emd = torch.sqrt(distance).mean().item()
    return cd, emd, eval_loss
