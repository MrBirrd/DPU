import argparse
import os

import numpy as np
import open3d as o3d
import torch
from cuml.neighbors import NearestNeighbors
from sklearn import neighbors
from tqdm import tqdm

from utils.utils import create_room_batches_training_faro, create_room_batches_training_iphone_v1, create_room_batches_training_iphone_v2, filter_iphone_scan

FACTOR = 3


def find_closest_neighbors(A, B, k=5):
    """
    For each point in A, efficiently find the k closest points in B using NearestNeighbors from scikit-learn.

    Parameters:
    A (np.array): Nx3 matrix representing points in 3D.
    B (np.array): Nx3 matrix representing points in 3D.
    k (int): Number of closest neighbors to find.

    Returns:
    np.array: Indices of the k closest points in B for each point in A.
    """
    # Using NearestNeighbors to find k closest points
    neigh = neighbors.NearestNeighbors(n_neighbors=k, n_jobs=-1, leaf_size=40)
    neigh.fit(B)
    distances, indices = neigh.kneighbors(A)

    return indices


def find_closest_neighbors_cuml(A, B, k=5):
    """
    For each point in A, efficiently find the k closest points in B using NearestNeighbors from scikit-learn.

    Parameters:
    A (np.array): Nx3 matrix representing points in 3D.
    B (np.array): Nx3 matrix representing points in 3D.
    k (int): Number of closest neighbors to find.

    Returns:
    np.array: Indices of the k closest points in B for each point in A.
    """
    # Using NearestNeighbors to find k closest points
    neigh = NearestNeighbors(n_neighbors=k, metric="l2")
    neigh.fit(B)
    distances, indices = neigh.kneighbors(A)

    return indices


def optimize_assignments(A, B, closest_neighbors):
    """
    Optimize the assignments from A to B, maximizing unique mappings in B while minimizing total distance.

    Parameters:
    A (np.array): Nx3 matrix representing points in 3D.
    B (np.array): Nx3 matrix representing points in 3D.
    closest_neighbors (list of list): Indices of the closest neighbors in B for each point in A.

    Returns:
    np.array: Array of indices in B to which each point in A is assigned.
    """
    N = A.shape[0]
    assigned_B_indices = -1 * np.ones(N, dtype=int)  # Initialize with -1 (unassigned)
    available_B_points = set(range(B.shape[0]))  # Set of available points in B

    for i, neighbors in enumerate(closest_neighbors):
        # Try to assign to the closest available neighbor
        for neighbor in neighbors:
            if neighbor in available_B_points:
                assigned_B_indices[i] = neighbor
                available_B_points.remove(neighbor)
                break

        # If all neighbors are already assigned, assign to the closest regardless of uniqueness
        if assigned_B_indices[i] == -1:
            assigned_B_indices[i] = neighbors[0]

    return assigned_B_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--name_suffix",
        type=str,
        default="",
        help="Suffix to append to the name of the points and features.",
    )
    parser.add_argument(
        "--target_root",
        type=str,
        help="Path to the target directory.",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=8192,
        help="Number of batches per scene.",
    )
    parser.add_argument(
        "--upsampling_rate",
        type=int,
        default=4,
        help="Upsampling ratio.",
    )
    parser.add_argument(
        "--centers_amount",
        type=float,
        default=2e-3,
        help="Amount of centers to sample from the point cloud.",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="dino",
        help="Features to use.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="unconditional",
        choices=["unconditional", "conditional"],
        help="Whether to use conditional or unconditional model.",
    )
    args = parser.parse_args()

    if args.target_root is None:
        args.target_root = args.data_root

    # set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    data_folders = os.listdir(args.data_root)
    data_folders = [f for f in data_folders if os.path.isdir(os.path.join(args.data_root, f))]
    pbar = tqdm(data_folders, total=len(data_folders))

    for data_folder in pbar:
        pbar.set_description(data_folder)
        faro_scan_path = os.path.join(args.data_root, data_folder, "scans", "mesh_aligned_0.05.ply")
        iphone_scan_path = os.path.join(args.data_root, data_folder, "scans", f"iphone{args.name_suffix}.ply")

        if not os.path.exists(faro_scan_path) or (args.mode == "conditional" and not os.path.exists(iphone_scan_path)):
            continue

        # unconditional mode
        if args.mode == "unconditional":
            # feature paths creation and check
            fpath = os.path.join(args.data_root, data_folder, "features", f"{args.feature_type}{args.name_suffix}.npy")
            if os.path.exists(fpath):
                features = np.load(fpath)
            else:
                print("Skipping", data_folder, "because of missing features")

            # target scene path
            target_scene_path = os.path.join(args.target_root, data_folder)
            os.makedirs(target_scene_path, exist_ok=True)

            existing_batches = [
                f for f in os.listdir(target_scene_path) if f.startswith("points") and f.endswith(".npz")
            ]

            pointcloud = np.array(o3d.io.read_point_cloud(faro_scan_path).points)

            if features.shape[-1] != pointcloud.shape[0]:
                print(
                    "Scene {} has {} points but {} {} features".format(
                        data_folder,
                        pointcloud.shape[0],
                        features[feature].shape[-1],
                        feature,
                    )
                )
                continue

            # calculate number of center points
            n_batches = int(pointcloud.shape[0] * args.centers_amount)

            if len(existing_batches) == n_batches:
                if os.path.exists(os.path.join(target_scene_path, f"points_{n_batches-1}.npz")):
                    print("Skipping", data_folder)
                    continue

            
            batches = create_room_batches_training_faro(
                pointcloud=pointcloud,
                features=features,
                n_batches=n_batches,
                target_scene_path=target_scene_path,
                args=args.npoints,
                existing_batches=existing_batches,
            )
            for batch_idx in batches:
                np.savez(
                    os.path.join(target_scene_path, "points_{}.npz".format(batch_idx)),
                    **batches[batch_idx],
                )

        elif args.mode == "conditional":
            # feature paths creation and check
            fpath = os.path.join(
                args.data_root,
                data_folder,
                "features",
                f"{args.feature_type}_iphone{args.name_suffix}.npy",
            )
            if os.path.exists(fpath):
                features = np.load(fpath).T
            else:
                print("Skipping", data_folder, "because of missing features")
                continue

            # target scene path
            target_scene_path = os.path.join(args.target_root, data_folder)
            os.makedirs(target_scene_path, exist_ok=True)

            existing_batches = [
                f for f in os.listdir(target_scene_path) if f.startswith("points") and f.endswith(".npz")
            ]

            scan_iphone = o3d.io.read_point_cloud(iphone_scan_path)
            scan_faro = o3d.io.read_point_cloud(faro_scan_path).voxel_down_sample(0.01)

            pcd_iphone = np.array(scan_iphone.points)
            rgb_iphone = np.array(scan_iphone.colors)

            pcd_faro = np.array(scan_faro.points)
            rgb_faro = np.array(scan_faro.colors)

            # filter iphone scan
            pcd_iphone, rgb_iphone, features = filter_iphone_scan(pcd_iphone, rgb_iphone, features, pcd_faro)
            
            n_points_iphone = pcd_iphone.shape[0]
            n_points_faro = pcd_faro.shape[0]

            if features.shape[0] != n_points_iphone:
                print(
                    "Scene {} has {} points but {} features".format(
                        data_folder,
                        n_points_iphone,
                        features.shape[0],
                    )
                )
                continue

            # get batches
            batches = create_room_batches_training_iphone_v2(
                pcd_faro=pcd_faro,
                pcd_iphone=pcd_iphone,
                rgb_faro=rgb_faro,
                rgb_iphone=rgb_iphone,
                features=features,
                args=args
            )
            
            # save batches
            for batch_idx in range(len(batches)):
                np.savez(
                    os.path.join(target_scene_path, "points_{}.npz".format(batch_idx)),
                    **batches[batch_idx],
                )


if __name__ == "__main__":
    main()
