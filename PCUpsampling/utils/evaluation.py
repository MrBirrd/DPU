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
from PCUpsampling.utils.utils import get_data_batch, to_cuda


# helper for chain of samples
def new_x_chain(x, num_chain):
    return torch.randn(num_chain, *x.shape[1:], device=x.device)


def evaluate(model, eval_iter, cfg, step, sampling=False, debug=False):
    if sampling:
        out_dir = cfg.out_sampling
    else:
        out_dir = cfg.outf_syn

    model.eval()

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
            pred_trajectory = torch.cat(model.sample(
                shape=new_x_chain(gt, 1).shape,
                device=gt.device,
                cond=features[0].unsqueeze(0) if features is not None else None,
                hint=gt if cfg.diffusion.sampling_hint else None,
                freq=0.1,
                clip_denoised=False,
            ), dim=0)
            
            print_stats(pred, "x_gen_eval")
            print_stats(pred_trajectory, "x_gen_all")
        

    # calculate the CD
    cds = []

    try:
        for x_pred, x_gt in zip(pred, gt):
            cd = chamfer_distance(
                x_pred.cpu().permute(1, 0).numpy(),
                x_gt.cpu().permute(1, 0).numpy(),
            )
            cds.append(cd)

        cd = np.mean(cds)

    except Exception as e:
        # switch row major to col major
        xgnp = pred.cpu().numpy()
        xgnp = np.asfortranarray(xgnp)
        pred = torch.from_numpy(xgnp).cuda()
        # evaluate again
        for x_pred, x_gt in zip(pred, gt):
            cd = chamfer_distance(
                x_pred.cpu().permute(1, 0).numpy(),
                x_gt.cpu().permute(1, 0).numpy(),
            )

            cds.append(cd)

        cd = np.mean(cds)

    loss = model.loss(pred, gt).mean().item()

    logger.info("CD: {} \t eval_loss_unweighted: {}", cd, loss)
    stats = {"CD": cd, "eval_loss_unweighted": loss}

    # visualize the pointclouds
    visualize_pointcloud_batch(
        "%s/%03d_pred.png" % (out_dir, step),
        pred.transpose(1, 2),
    )

    visualize_pointcloud_batch(
        "%s/%03d_pred_all.png" % (out_dir, step),
        pred_trajectory.transpose(1, 2),
    )

    if lowres_pointcloud is not None:
        visualize_pointcloud_batch(
            "%s/%03d_low_quality.png" % (out_dir, step),
            lowres_pointcloud.transpose(1, 2),
        )

    visualize_pointcloud_batch(
        "%s/%03d_high_quality.png" % (out_dir, step),
        gt.transpose(1, 2),
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
        np.save("%s/%03d_pred.npy" % (out_dir, step), pred.cpu().numpy())
        np.save("%s/%03d_pred_all.npy" % (out_dir, step), pred_trajectory.cpu().numpy())
        np.save("%s/%03d_hints.npy" % (out_dir, step), hints.cpu().numpy())

        if gt is not None:
            np.save("%s/%03d_gt_highres.npy" % (out_dir, step), gt.cpu().numpy())
        if lowres_pointcloud is not None:
            np.save("%s/%03d_gt_lowres.npy" % (out_dir, step), lowres_pointcloud.cpu().numpy())

    model.train()
    return stats


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
