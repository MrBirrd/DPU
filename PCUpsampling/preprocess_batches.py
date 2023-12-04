from tqdm import tqdm
import numpy as np
import os
import argparse
import torch
import pyminiply
from scipy import spatial
from modules.functional import furthest_point_sample
from einops import rearrange


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--target_root",
        type=str,
        required=True,
        help="Path to the target directory.",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=8192,
        help="Number of batches per scene.",
    )
    args = parser.parse_args()
    
    data_folders = os.listdir(args.data_root)
    data_folders = [f for f in data_folders if os.path.isdir(os.path.join(args.data_root, f))]
    
    factor = np.random.uniform(0.5, 2.5)
    
    for data_folder in tqdm(data_folders):
        scan_path = os.path.join(args.data_root, data_folder, "scans", "mesh_aligned_0.05.ply")
        dino_feature_path = os.path.join(args.data_root, data_folder, "features", "dino.npy")
        if not os.path.exists(scan_path) or not os.path.exists(dino_feature_path):
            print(scan_path, dino_feature_path)
            continue
        else:
            pointcloud, *_ = pyminiply.read(scan_path)
            pcd_tree = spatial.cKDTree(pointcloud)
            features = np.load(dino_feature_path)
            
            if features.shape[-1] != pointcloud.shape[0]:
                print("Scene {} has {} points but {} features".format(data_folder, pointcloud.shape[0], features.shape[0]))
                continue
            
            # calculate number of center points
            n_batches = pointcloud.shape[0] // args.npoints
            
            n_batches = int(n_batches * factor) # add 30% overlap
            
            # get center points
            pointcloud = torch.from_numpy(pointcloud).float().cuda()
            pointcloud = rearrange(pointcloud, 'n d -> 1 d n')
            center_points = furthest_point_sample(pointcloud, n_batches).squeeze().cpu().numpy().T
            
            _, idxs = pcd_tree.query(center_points, k=args.npoints, p=2)
            
            # target scene path
            target_scene_path = os.path.join(args.target_root, data_folder)
            os.makedirs(target_scene_path, exist_ok=True)
            
            # check if already processed
            existing_files = os.listdir(target_scene_path)
            existing_files = [f for f in existing_files if f.startswith("points")]
            latest_idx = 0
            if len(existing_files) != 0:
                latest_idx = max([int(f.split("_")[-1].split(".")[0]) for f in existing_files])
                print("appending {} batches to scene {} with {} already existing batches".format(n_batches, data_folder, latest_idx))
                        
            # save the batches
            for i, idx in enumerate(idxs):
                batch = pointcloud[:, :, idx].squeeze().cpu().numpy()
                batch_features = features[:, idx]
                
                # save the batch
                batch_path = os.path.join(target_scene_path, "points_{}.npy".format(i+latest_idx))
                features_path = os.path.join(target_scene_path, "dino_{}.npy".format(i+latest_idx))
                np.save(batch_path, batch)
                np.save(features_path, batch_features.astype(np.float16))

if __name__ == "__main__":
    main()