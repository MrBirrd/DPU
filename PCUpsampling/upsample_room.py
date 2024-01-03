from functools import partial

import numpy as np
import open3d as o3d
import torch
from loguru import logger
from scipy.spatial import cKDTree
from tqdm import tqdm

from pvcnn.functional.sampling import furthest_point_sample
from training.evaluation import get_metrics
from training.model_loader import load_diffusion
from utils.args import args_to_string, parse_args
from utils.utils import create_room_batches_iphone


def main():
    cfg = parse_args()
    cfg.gpu = None

    model, ckpt = load_diffusion(cfg)
    model.eval()

    room_path = "../test_room/02455b3d20"
    # point clouds
    faro_path = room_path + "/scans/mesh_aligned_0.05.ply"
    iphone_path = room_path + "/scans/iphone.ply"

    # features
    faro_dino_f_path = room_path + "/features/dino_faro.npy"
    iphone_dino_f_path = room_path + "/features/dino_iphone.npy"

    faro = o3d.io.read_point_cloud(faro_path)
    iphone = o3d.io.read_point_cloud(iphone_path)

    points_faro = np.asarray(faro.points, dtype=np.float32)
    points_iphone = np.asarray(iphone.points, dtype=np.float32)
    rgb_faro = np.asarray(faro.colors, dtype=np.float32)
    rgb_iphone = np.asarray(iphone.colors, dtype=np.float32)

    npoints = cfg.data.npoints

    logger.info(f"Upsamling {room_path} using config: {args_to_string(cfg)}")
    torch.backends.cudnn.benchmark = True

    if cfg.data.dataset == "ScanNetPP_iPhone":
        n_batches = int(points_iphone.shape[0] // npoints * 10)

        features = None
        points_key = "iphone"
        features_key = "dino"

        if cfg.data.point_features == "dino":
            features = np.load(iphone_dino_f_path).T

        if cfg.data.use_rgb_features:
            if features is not None:
                features = np.concatenate([rgb_iphone, features], axis=1)
            else:
                features = rgb_iphone

        upsampling_batches = create_room_batches_iphone(
            pcd_faro=points_faro,
            pcd_iphone=points_iphone,
            rgb_faro=rgb_faro,
            rgb_iphone=rgb_iphone,
            features=features,
            n_batches=n_batches,
            min_points_faro=1024,
            npoints=cfg.data.npoints,
        )

        sampling_fn = partial(model.sample, clip=False)

    elif cfg.data.dataset == "ScanNetPP_Faro":
        raise NotImplementedError
        points_hr = np.asarray(faro.points)
        feats_dino_faro = np.load(faro_dino_f_path)

        points_room = points_hr
        n_batches = int(points_room.shape[0] // npoints * 1.3)
        centers = furthest_point_sample(torch.tensor(points_room.T).float().unsqueeze(0).cuda(), n_batches)
        tree = cKDTree(points_room)
        features = feats_dino_faro
        upsampling_batches = None
        sampling_fn = partial(model.sample, clip=False, add_x_start_noise=True)

    n_gpu_batches = int(np.ceil(len(upsampling_batches) / cfg.sampling.bs))

    denoised_batches = []
    noisy_batches = []
    gt_batches = []

    for gpu_batch in tqdm(range(0, n_gpu_batches)):
        batch_data = upsampling_batches[gpu_batch * cfg.sampling.bs : (gpu_batch + 1) * cfg.sampling.bs]

        batch_points = torch.tensor([]).cuda()
        batch_gt = torch.tensor([]).cuda()
        batch_features = torch.tensor([]).cuda()
        scales = []

        # create gpu batches
        for data in batch_data:
            # TODO check the feature generation with rgb + dino!!
            batch_points = torch.cat(
                [
                    batch_points,
                    torch.from_numpy(data["iphone"][:, :3].T).unsqueeze(0).cuda(),
                ],
                dim=0,
            )
            batch_gt = torch.cat(
                [batch_gt, torch.from_numpy(data["faro"][:, :3].T).unsqueeze(0).cuda()],
                dim=0,
            )
            batch_features = torch.cat(
                [
                    batch_features,
                    torch.from_numpy(data[features_key].T).unsqueeze(0).cuda(),
                ],
                dim=0,
            )
            scales.append((data["center"], data["scale"]))

        with torch.inference_mode():
            sample = model.sample(
                shape=batch_points.shape,
                cond=batch_features if not cfg.data.unconditional else None,
                x_start=batch_points,
                add_x_start_noise=True,
                return_noised_hint=True,
                clip=False,
            )

        x_pred, x_start = sample["x_pred"], sample["x_start"]

        # calculate metrics for the batch using gpu tensors
        cd, emd, _ = get_metrics(model, batch_gt, x_pred)
        logger.info(f"CD: {cd:.4f}, EMD: {emd:.4f}")
        cd, emd, _ = get_metrics(model, batch_gt, x_start)
        logger.info(f"CD: {cd:.4f}, EMD: {emd:.4f}")

        x_pred = x_pred.cpu().detach().permute(0, 2, 1).numpy()
        x_start = x_start.cpu().detach().permute(0, 2, 1).numpy()
        x_gt = batch_gt.cpu().detach().permute(0, 2, 1).numpy()

        # renormalize
        for idx, (center, scale) in enumerate(scales):
            x_pred[idx] *= scale
            x_pred[idx] += center
            x_start[idx] *= scale
            x_start[idx] += center
            x_gt[idx] *= scale
            x_gt[idx] += center

        denoised_batches.append(x_pred)
        noisy_batches.append(x_start)
        gt_batches.append(x_gt)

    denoised_room = np.concatenate(denoised_batches, axis=0).reshape(-1, 3)
    noisy_room = np.concatenate(noisy_batches, axis=0).reshape(-1, 3)
    gt_batches = np.concatenate(gt_batches, axis=0).reshape(-1, 3)

    model_name = cfg.model_path.split("/")[-2]

    # center and scale the point clouds
    center = np.mean(noisy_room, axis=0)
    gt_batches -= center
    noisy_room -= center
    denoised_room -= center

    np.save(f"{room_path}/denoised_{model_name}.npy", denoised_room)
    np.save(f"{room_path}/noised_{model_name}.npy", noisy_room)
    np.save(f"{room_path}/gt_{model_name}.npy", gt_batches)


if __name__ == "__main__":
    main()
