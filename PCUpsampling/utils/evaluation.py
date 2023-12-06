from torch import Tensor
import torch
from loguru import logger
import numpy as np
import wandb
from utils.visualize import visualize_pointcloud_batch
from point_cloud_utils import chamfer_distance
import json
from modules.functional import furthest_point_sample
from scipy import spatial
from tqdm import tqdm
from utils.utils import get_data_batch, to_cuda
from metrics.metrics import print_stats, calculate_cd
from metrics.emd_ import emd_module as EMD


def new_x_chain(x, num_chain):
    return torch.randn(num_chain, *x.shape[1:], device=x.device)


def save_visualizations(items, out_dir, step):
    for item in items:
        ptc, name = item
        visualize_pointcloud_batch(
            "%s/%03d_%s.png" % (out_dir, step, name),
            ptc.transpose(1, 2),
        )


def log_wandb(name, out_dir, step):
    wandb_img = wandb.Image("%s/%03d_%s.png" % (out_dir, step, name))
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
    
    gt, features = get_data_batch(batch=eval_data, cfg=cfg, device=cfg.gpu)

    with torch.no_grad():
        pred, hints = model.sample(
            shape=new_x_chain(gt, cfg.sampling.bs).shape,
            device=gt.device,
            cond=features,
            hint=gt if cfg.diffusion.sampling_hint else None,
            return_noised_hint=True,
            clip_denoised=False,
        )

        if debug:
            pred_trajectory = torch.cat(
                model.sample(
                    shape=new_x_chain(gt, 1).shape,
                    device=gt.device,
                    cond=features[0].unsqueeze(0) if features is not None else None,
                    hint=gt if cfg.diffusion.sampling_hint else None,
                    freq=0.1,
                    clip_denoised=False,
                ),
                dim=0,
            )
            # log and save the trajectory
            print_stats(pred, "x_gen_eval")
            print_stats(pred_trajectory, "x_gen_all")
            save_visualizations((pred_trajectory, "trajectory"), out_dir, step)
            save_ptc("trajectory", pred_trajectory, out_dir, step)
            if not sampling:
                log_wandb("trajectory", out_dir, step)

    # visualize the pointclouds
    save_visualizations(
        [
            (pred, "pred"),
            (hints, "hints"),
            (gt, "gt"),
        ],
        out_dir,
        step,
    )

    # calculate stats
    # subsample cloud to closest multiple of 128
    n_points = pred.shape[-1]
    n_points = n_points - n_points % 128

    pred = pred[..., :n_points]
    gt = gt[..., :n_points]
    hints = hints[..., :n_points]
    
    emd = EMD.emdModule()
    cd = calculate_cd(pred, gt)
    eval_loss = model.loss(pred, gt).mean().item()
    distance, _ = emd(gt.permute(0, 2, 1), pred.permute(0, 2, 1), 0.05, 3000)
    emd = torch.sqrt(distance).mean().item()

    # print the stats
    batch_metrics = {
        "CD": cd,
        "EMD": emd,
        "eval_loss_unweighted": eval_loss,
    }
    logger.info(batch_metrics)

    if not sampling:
        wandb.log(batch_metrics, step=step)
        log_wandb("pred", out_dir, step)
        log_wandb("hints", out_dir, step)
        log_wandb("gt", out_dir, step)
    elif save_npy:
        save_ptc("pred", pred, out_dir, step)
        save_ptc("hints", hints, out_dir, step)
        save_ptc("gt", gt, out_dir, step)

    return batch_metrics


def upsample_big_pointcloud(model, pointcloud, batch_size=8192, n_batches=32):
    """Upsamples big pointcloud of shape [N, 3] by generating batches of smaller pointclouds and stitching them together."""
    # numerate the pointcloud points
    numbering = torch.arange(pointcloud.shape[0], device=pointcloud.device).unsqueeze(-1)
    pointcloud = torch.cat([pointcloud, numbering], dim=-1)
    # make tree to generate local sample batches
    pcd_tree = spatial.cKDTree(pointcloud[:, :3])
    # find center points of batches by farthest point sampling
    n_centers = pointcloud.shape[0] // batch_size * 2
    center_points = furthest_point_sample(pointcloud[:, :3].T.unsqueeze(0).cuda(), n_centers).squeeze().T

    upsampled_points = torch.tensor([]).cuda()
    upsampling_batch = torch.tensor([]).cuda()

    total_pbar = tqdm(total=np.ceil(center_points.shape[0] / n_batches), desc="Upsampling pointcloud")

    # upsample batches
    for n_center, center in enumerate(center_points):
        # sample
        _, idx = pcd_tree.query(center.cpu(), k=batch_size)
        batch = pointcloud[idx].cuda()
        # TODO normalize the batch and sav enormalizer constants to redo the normalizing
        upsampling_batch = torch.cat([upsampling_batch, batch.unsqueeze(0)], dim=0)
        if upsampling_batch.shape[0] == n_batches:
            # upsample
            low_res_cloud = upsampling_batch[:, :, :3].permute(0, 2, 1).cuda()
            with torch.no_grad():
                upsampled_batch = model.sample(
                    shape=low_res_cloud.shape,
                    device=low_res_cloud.device,
                    cond=None,
                    hint=low_res_cloud,
                    add_hint_noise=False,
                    clip_denoised=False,
                )
            total_pbar.update(1)
            # reshape
            upsampled_batch = upsampled_batch.permute(0, 2, 1)  # B, 3, N => B, N 3
            # add back indices
            upsampled_batch = torch.cat([upsampled_batch, upsampling_batch[:, :, 3:]], dim=-1)
            upsampled_points = torch.cat([upsampled_points, upsampled_batch.squeeze(0)], dim=0)
            upsampling_batch = torch.tensor([]).cuda()
    # do final upsampling with last batch
    if upsampling_batch.shape[0] != 0:
        # upsample
        low_res_cloud = upsampling_batch[:, :, :3].permute(0, 2, 1).cuda()
        with torch.no_grad():
            upsampled_batch = model.sample(
                shape=low_res_cloud.shape,
                device=low_res_cloud.device,
                cond=None,
                hint=low_res_cloud,
                add_hint_noise=False,
                clip_denoised=False,
            )
        total_pbar.update(1)
        # reshape
        upsampled_batch = upsampled_batch.permute(0, 2, 1)  # B, 3, N => B, N 3
        # add back indices
        upsampled_batch = torch.cat([upsampled_batch, upsampling_batch[:, :, 3:]], dim=-1)
        upsampled_points = torch.cat([upsampled_points, upsampled_batch.squeeze(0)], dim=0)
        upsampling_batch = torch.tensor([]).cuda()

    # check if all points were upsampled
    return upsampled_points
