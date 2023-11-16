from torch import Tensor
import torch
from loguru import logger
import numpy as np
import wandb
from utils.visualize import visualize_pointcloud_batch
from point_cloud_utils import chamfer_distance
import json


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


# helper for chain of samples
def new_x_chain(x, num_chain):
    return torch.randn(num_chain, *x.shape[1:], device=x.device)


def evaluate(model, eval_iter, cfg, step, sampling=False):
    if sampling:
        out_dir = cfg.out_sampling
    else:
        out_dir = cfg.outf_syn

    model.eval()

    eval_data = next(eval_iter)

    gt_pointcloud = eval_data["train_points"].transpose(1, 2)
    lowres_pointcloud = (
        eval_data["train_points_lowres"].transpose(1, 2)
        if "train_points_lowres" in eval_data and not cfg.data.unconditional
        else None
    )

    # move data to gpu
    if cfg.distribution_type == "multi":
        gt_pointcloud = gt_pointcloud.cuda(cfg.gpu)
        lowres_pointcloud = lowres_pointcloud.cuda(cfg.gpu) if lowres_pointcloud is not None else None
    elif cfg.distribution_type == "single":
        gt_pointcloud = gt_pointcloud.cuda()
        lowres_pointcloud = lowres_pointcloud.cuda() if lowres_pointcloud is not None else None

    with torch.no_grad():
        if cfg.sampling.bs == 1:
            cond = lowres_pointcloud[0].unsqueeze(0) if lowres_pointcloud is not None else None
        else:
            cond = lowres_pointcloud[: cfg.sampling.bs] if lowres_pointcloud is not None else None

        x_gen_eval = model.sample(
            shape=new_x_chain(gt_pointcloud, cfg.sampling.bs).shape,
            device=gt_pointcloud.device,
            cond=cond,
            hint=gt_pointcloud if cfg.diffusion.sampling_hint else None,
            clip_denoised=False,
        )

        x_gen_list = model.sample(
            shape=new_x_chain(gt_pointcloud, 1).shape,
            device=gt_pointcloud.device,
            cond=lowres_pointcloud[0].unsqueeze(0) if lowres_pointcloud is not None else None,
            hint=gt_pointcloud if cfg.diffusion.sampling_hint else None,
            freq=0.1,
            clip_denoised=False,
        )

        x_gen_all = torch.cat(x_gen_list, dim=0)

    # calculate metrics such as min, max, mean, std, etc.
    print_stats(x_gen_eval, "x_gen_eval")
    print_stats(x_gen_all, "x_gen_all")

    # calculate the CD
    cds = []

    try:
        for x_pred, x_gt in zip(x_gen_eval, gt_pointcloud):
            cd = chamfer_distance(
                x_pred.cpu().permute(1, 0).numpy(),
                x_gt.cpu().permute(1, 0).numpy(),
            )
            cds.append(cd)

        cd = np.mean(cds)

    except Exception as e:
        # switch row major to col major
        xgnp = x_gen_eval.cpu().numpy()
        xgnp = np.asfortranarray(xgnp)
        x_gen_eval = torch.from_numpy(xgnp).cuda()
        # evaluate again
        for x_pred, x_gt in zip(x_gen_eval, gt_pointcloud):
            cd = chamfer_distance(
                x_pred.cpu().permute(1, 0).numpy(),
                x_gt.cpu().permute(1, 0).numpy(),
            )

            cds.append(cd)

        cd = np.mean(cds)

    loss = model.loss(x_gen_eval, gt_pointcloud).mean().item()
    logger.info("CD: {} \t eval_loss_unweighted: {}", cd, loss)
    stats = {"CD": cd, "eval_loss_unweighted": loss}

    # visualize the pointclouds
    visualize_pointcloud_batch(
        "%s/%03d_pred.png" % (out_dir, step),
        x_gen_eval.transpose(1, 2),
    )

    visualize_pointcloud_batch(
        "%s/%03d_pred_all.png" % (out_dir, step),
        x_gen_all.transpose(1, 2),
    )

    if lowres_pointcloud is not None:
        visualize_pointcloud_batch(
            "%s/%03d_low_quality.png" % (out_dir, step),
            lowres_pointcloud.transpose(1, 2),
        )

    visualize_pointcloud_batch(
        "%s/%03d_high_quality.png" % (out_dir, step),
        gt_pointcloud.transpose(1, 2),
    )

    if not sampling:
        wandb.log(stats, step=step)

        samps_eval = wandb.Image("%s/%03d_pred.png" % (out_dir, step))
        samps_eval_all = wandb.Image("%s/%03d_pred_all.png" % (out_dir, step))
        samps_lowres = (
            wandb.Image("%s/%03d_low_quality.png" % (out_dir, step)) if lowres_pointcloud is not None else None
        )
        samps_highres = wandb.Image("%s/%03d_high_quality.png" % (out_dir, step))
        wandb.log(
            {
                "samples_eval": samps_eval,
                "samples_eval_all": samps_eval_all,
                "samples_lowres": samps_lowres,
                "samples_highres": samps_highres,
            },
            step=step,
        )
    else:
        np.save("%s/%03d_pred.npy" % (out_dir, step), x_gen_eval.cpu().numpy())
        np.save("%s/%03d_pred_all.npy" % (out_dir, step), x_gen_all.cpu().numpy())
        if gt_pointcloud is not None:
            np.save("%s/%03d_gt_highres.npy" % (out_dir, step), gt_pointcloud.cpu().numpy())
        if lowres_pointcloud is not None:
            np.save("%s/%03d_gt_lowres.npy" % (out_dir, step), lowres_pointcloud.cpu().numpy())

    model.train()
    return stats
