import argparse
import os
import numpy as np
import pyminiply
import torch
from sklearn import neighbors
from einops import rearrange
from tqdm import tqdm
import open3d as o3d
from modules.functional import furthest_point_sample
from cuml.neighbors import NearestNeighbors
import cudf

FEATURES = ["dino"]
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
        "--centers_amount",
        type=float,
        default=2e-3,
        help="Amount of centers to sample from the point cloud.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Whether to append to existing batches.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Whether to fix missing batches.",
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
        iphone_scan_path = os.path.join(args.data_root, data_folder, "scans", "iphone.ply")

        if not os.path.exists(faro_scan_path) or (args.mode == "conditional" and not os.path.exists(iphone_scan_path)):
            continue

        # unconditional mode
        if args.mode == "unconditional":
            # feature paths creation and check
            features = {}
            for feature_type in FEATURES:
                fpath = os.path.join(args.data_root, data_folder, "features", f"{feature_type}.npy")
                if os.path.exists(fpath):
                    features[feature_type] = np.load(fpath)
                else:
                    continue
            if len(features) != len(FEATURES):
                continue

            # target scene path
            target_scene_path = os.path.join(args.target_root, data_folder)
            os.makedirs(target_scene_path, exist_ok=True)

            existing_batches = [
                f for f in os.listdir(target_scene_path) if f.startswith("points") and f.endswith(".npz")
            ]

            if args.mode == "conditional":
                pointcloud = np.array(o3d.io.read_point_cloud(iphone_scan_path).points)
            else:
                pointcloud = np.array(o3d.io.read_point_cloud(faro_scan_path).points)

            for feature in features:
                if features[feature].shape[-1] != pointcloud.shape[0]:
                    print(
                        "Scene {} has {} points but {} {} features".format(
                            data_folder, pointcloud.shape[0], features[feature].shape[-1], feature
                        )
                    )
                    continue

            # calculate number of center points
            n_batches = int(pointcloud.shape[0] * args.centers_amount)

            if len(existing_batches) == n_batches:
                if os.path.exists(os.path.join(target_scene_path, f"points_{n_batches-1}.npz")):
                    print("Skipping", data_folder)
                    continue

            # get center points
            pointcloud_torch = torch.from_numpy(pointcloud).float().cuda()
            pointcloud_torch = rearrange(pointcloud_torch, "n d -> 1 d n")
            center_points = furthest_point_sample(pointcloud_torch, n_batches).squeeze().cpu().numpy().T

            pointcloud_tree_cuml = NearestNeighbors(metric="l1" if args.mode == "conditional" else "l2")
            pointcloud_tree_cuml.fit(pointcloud)

            idxs = pointcloud_tree_cuml.kneighbors(center_points, n_neighbors=args.npoints, return_distance=False)
            latest_idx = 0

            if args.append:
                if len(existing_batches) > 0:
                    latest_idx = max([int(f.split("_")[-1].split(".")[0]) for f in existing_batches])

            # generate new data
            for i in range(n_batches):
                batch_path = os.path.join(target_scene_path, "points_{}.npz".format(i + latest_idx))
                if os.path.exists(batch_path):
                    continue

                # extract the indices for the current batch
                neighbor_indices = idxs[i]

                # create npz file to append to and later save
                batch_data = {}

                # handle the points
                batch_points = pointcloud[neighbor_indices]
                batch_data["points"] = batch_points
                batch_data["indices"] = neighbor_indices

                # handle features
                for feature in features:
                    batch_features = features[feature][:, neighbor_indices.ravel()].astype(np.float16)
                    batch_data[feature] = batch_features

                np.savez(batch_path, **batch_data)
        elif args.mode == "conditional":
            # feature paths creation and check
            features = {}
            for feature_type in FEATURES:
                fpath = os.path.join(args.data_root, data_folder, "features", f"{feature_type}_iphone.npy")
                if os.path.exists(fpath):
                    features[feature_type] = np.load(fpath).T
                else:
                    continue
            if len(features) != len(FEATURES):
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

            n_points_iphone = pcd_iphone.shape[0]
            n_points_faro = pcd_faro.shape[0]

            for feature in features:
                if features[feature].shape[0] != n_points_iphone:
                    print(
                        "Scene {} has {} points but {} {} features".format(
                            data_folder, n_points_iphone, features[feature].shape[0], feature
                        )
                    )
                    continue

            # calculate number of center points
            n_batches = int(n_points_iphone * args.centers_amount)

            tree_faro = neighbors.KDTree(pcd_faro, metric="l1")
            tree_iphone = neighbors.KDTree(pcd_iphone, metric="l1")

            # get center points of batches
            pointcloud_torch = torch.from_numpy(pcd_iphone).float().cuda()
            pointcloud_torch = rearrange(pointcloud_torch, "n d -> 1 d n")
            center_points = furthest_point_sample(pointcloud_torch, n_batches).squeeze().cpu().numpy().T

            # first query points in radius
            idxs_faro = tree_faro.query_radius(center_points, r=1, return_distance=False)
            idxs_iphone = tree_iphone.query_radius(center_points, r=1, return_distance=False)

            assert (
                len(idxs_faro) == len(idxs_iphone) == n_batches
            ), "Number of batches is not equal to number of indices"

            batch_idx = 0
            for idx in range(len(idxs_iphone)):
                faro_batch_points = pcd_faro[idxs_faro[idx]]
                iphone_batch_points = pcd_iphone[idxs_iphone[idx]]
                faro_batch_colors = rgb_faro[idxs_faro[idx]]
                iphone_batch_colors = rgb_iphone[idxs_iphone[idx]]
                iphone_batch_dino = features["dino"][idxs_iphone[idx]]

                # skip if the batch is too small
                if len(faro_batch_points) < args.npoints:
                    continue

                # center the points
                center = faro_batch_points.mean(axis=0)
                faro_batch_points -= center
                iphone_batch_points -= center

                diff = args.npoints - len(iphone_batch_points)
                if diff > 0:
                    rand_idx = np.random.randint(0, len(iphone_batch_points), diff)
                    iphone_additional_xyz = iphone_batch_points[rand_idx]
                    iphone_additional_rgb = iphone_batch_colors[rand_idx]
                    iphone_additional_dino = features["dino"][idxs_iphone[idx]][rand_idx]
                    iphone_additional_xyz += np.random.normal(0, 1e-2, iphone_additional_xyz.shape)

                    iphone_batch_points = np.concatenate([iphone_batch_points, iphone_additional_xyz])
                    iphone_batch_colors = np.concatenate([iphone_batch_colors, iphone_additional_rgb])
                    iphone_batch_dino = np.concatenate([iphone_batch_dino, iphone_additional_dino])
                else:
                    rand_idx = np.random.randint(0, len(iphone_batch_points), args.npoints)
                    iphone_batch_points = iphone_batch_points[rand_idx]
                    iphone_batch_colors = iphone_batch_colors[rand_idx]
                    iphone_batch_dino = features["dino"][idxs_iphone[idx]][rand_idx]

                # assign points using NN
                cn = find_closest_neighbors_cuml(iphone_batch_points, faro_batch_points, k=200)
                assignment = optimize_assignments(iphone_batch_points, faro_batch_points, cn)

                faro_batch_points_assigned = faro_batch_points[assignment]
                faro_batch_colors_assigned = faro_batch_colors[assignment]

                batch_data = {}
                batch_data["faro"] = np.concatenate([faro_batch_points_assigned, faro_batch_colors_assigned], axis=1)
                batch_data["iphone"] = np.concatenate([iphone_batch_points, iphone_batch_colors], axis=1)
                batch_data["dino"] = iphone_batch_dino
                np.savez(os.path.join(target_scene_path, "points_{}.npz".format(batch_idx)), **batch_data)
                batch_idx += 1


if __name__ == "__main__":
    main()
