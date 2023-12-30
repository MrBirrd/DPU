import os

import numpy as np
import open3d as o3d
import pyminiply
import pyviz3d.visualizer as viz
import torch
from model.loader import load_diffusion
from modules.functional import furthest_point_sample
from scipy.spatial import cKDTree
from tqdm import tqdm
from utils.args import parse_args


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

    npoints = cfg.data.npoints
    
    if cfg.data.dataset == "ScanNetPP_iPhone":
        points_lr = np.asarray(iphone.points, dtype=np.float32)
        feats_dino_iphone = np.load(iphone_dino_f_path)
        
        points_room = points_lr
        n_batches = int(points_lr.shape[0] // npoints * 1.3)
        centers = furthest_point_sample(torch.tensor(points_room.T).float().unsqueeze(0).cuda(), n_batches)
        tree = cKDTree(points_room)
        if cfg.data.use_rgb_features:
            features = torch.from_numpy(np.asarray(iphone.colors, dtype=np.float32).T)
        else:
            raise NotImplementedError
            
    elif cfg.data.dataset == "ScanNetPP_Faro":
        points_hr = np.asarray(faro.points)
        feats_dino_faro = np.load(faro_dino_f_path)
        
        points_room = points_hr
        n_batches = int(points_room.shape[0] // npoints * 1.3)
        centers = furthest_point_sample(torch.tensor(points_room.T).float().unsqueeze(0).cuda(), n_batches)
        tree = cKDTree(points_room)
        features = feats_dino_faro

    _, idxs = tree.query(centers.squeeze().cpu().numpy().T, k=npoints, p=1.0)

    n_gpu_batches = int(np.ceil(len(idxs) / cfg.sampling.bs))

    denoised_batches = []
    noisy_batches = []
    gt_batches = []

    for gpu_batch in tqdm(range(0, n_gpu_batches)):
        batch_idxs = idxs[gpu_batch * cfg.sampling.bs : (gpu_batch + 1) * cfg.sampling.bs]

        batch_features = torch.tensor([], dtype=torch.float).cuda()
        batch_points = torch.tensor([], dtype=torch.float).cuda()
        scales = []

        for idx in batch_idxs:
            batch_features = torch.cat([batch_features, features[:, idx].unsqueeze(0).cuda()], dim=0)
            points = points_room[idx]
            center = np.mean(points, axis=0)
            points -= center
            scale = np.max(np.linalg.norm(points, axis=1))
            points /= scale
            batch_points = torch.cat([batch_points, torch.tensor(points.T).unsqueeze(0).cuda()], dim=0)
            scales.append((center, scale))

        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                sample = model.sample(
                    shape=batch_points.shape,
                    cond=batch_features if not cfg.data.unconditional else None,
                    x_start=batch_points,
                    add_x_start_noise=True,
                    return_noised_hint=True,
                    clip=False,
                )

            x_pred, x_start = sample["x_pred"], sample["x_start"]
            
            x_pred = x_pred.cpu().detach().permute(0, 2, 1).numpy()
            x_start = x_start.cpu().detach().permute(0, 2, 1).numpy()
            x_gt = batch_points.cpu().detach().permute(0, 2, 1).numpy()

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
    np.save(f"{room_path}/denoised_{model_name}.npy", denoised_room)
    np.save(f"{room_path}/noised_{model_name}.npy", noisy_room)
    np.save(f"{room_path}/gt_{model_name}.npy", gt_batches)


if __name__ == "__main__":
    main()
