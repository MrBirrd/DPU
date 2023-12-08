import torch
from omegaconf import OmegaConf
from model.diffusion_lucid import GaussianDiffusion
from model.loader import load_model
import pyminiply
import numpy as np
from modules.functional import furthest_point_sample
import argparse
import cudf
from cuml.neighbors import NearestNeighbors
from tqdm import tqdm
from utils.args import parse_args
from scipy.spatial import cKDTree
import pyviz3d.visualizer as viz
from einops import rearrange


def main():
    cfg = parse_args()
    cfg.gpu = None

    model, _ = load_model(cfg, None, smart=False)
    model.eval()

    room_path = "../datasets/scannetpp/02455b3d20/"
    room_points = room_path + "/scans/mesh_aligned_0.05.ply"
    room_features = room_path + "/features/dino.npy"

    pts_npy, *_ = pyminiply.read(room_points)
    feats = np.load(room_features)

    npoints = cfg.data.npoints
    n_batches = int(pts_npy.shape[0] // npoints * 1.3)

    centers = furthest_point_sample(torch.tensor(pts_npy.T).unsqueeze(0).cuda(), n_batches)

    tree = cKDTree(pts_npy)

    _, idxs = tree.query(centers.squeeze().cpu().numpy().T, k=npoints, p=2)

    n_gpu_batches = int(np.ceil(len(idxs) / cfg.diffusion.sampling.bs))

    denoised_batches = []
    noisy_batches = []

    for gpu_batch in tqdm(range(0, n_gpu_batches)):
        batch_idxs = idxs[gpu_batch * cfg.diffusion.sampling.bs : (gpu_batch + 1) * cfg.diffusion.sampling.bs]

        batch_features = []
        batch_points = []
        scales = []

        for idx in batch_idxs:
            batch_features.append(feats[:, idx])
            points = pts_npy[idx]
            center = np.mean(points, axis=0)
            points -= center
            scale = np.max(np.linalg.norm(points, axis=1))
            points /= scale
            batch_points.append(points.T)
            scales.append((center, scale))

        batch_features = torch.from_numpy(np.array(batch_features)).cuda()
        batch_points = torch.from_numpy(np.array(batch_points)).cuda()

        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                out, out_noises = model.sample(
                    shape=batch_points.shape,
                    cond=batch_features,
                    x_start=batch_points,
                    add_x_start_noise=True,
                    return_noised_hint=True,
                    clip=False,
                )

            out = out.cpu().detach().permute(0, 2, 1).numpy()
            out_noises = out_noises.cpu().detach().permute(0, 2, 1).numpy()

            # renormalize
            for idx, (center, scale) in enumerate(scales):
                out[idx] *= scale
                out[idx] += center
                out_noises[idx] *= scale
                out_noises[idx] += center

        denoised_batches.append(out)
        noisy_batches.append(out_noises)

    denoised_room = np.concatenate(denoised_batches, axis=0).reshape(-1, 3)
    noisy_room = np.concatenate(noisy_batches, axis=0).reshape(-1, 3)

    room_center = np.mean(pts_npy, axis=0)
    denoised_room -= room_center
    noisy_room -= room_center

    v = viz.Visualizer()
    v.add_points("denoised", denoised_room, point_size=2, visible=False)
    v.add_points("noised", noisy_room, point_size=2, visible=False)
    v.save(".viz")


if __name__ == "__main__":
    main()
