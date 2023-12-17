import argparse
import os

import cudf
import numpy as np
import pyminiply
import torch
from cuml.neighbors import NearestNeighbors
from einops import rearrange
from tqdm import tqdm

from modules.functional import furthest_point_sample

FEATURES = ["dino"]
FACTOR = 3


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
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    data_folders = os.listdir(args.data_root)
    data_folders = [f for f in data_folders if os.path.isdir(os.path.join(args.data_root, f))]
    pbar = tqdm(data_folders, total=len(data_folders))

    for data_folder in pbar:
        pbar.set_description(data_folder)
        scan_path = os.path.join(args.data_root, data_folder, "scans", "mesh_aligned_0.05.ply")
        if not os.path.exists(scan_path):
            continue

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
            f for f in os.listdir(target_scene_path) if f.startswith("points") and f.endswith(".npy")
        ]  # TODO adapt to npz files afterwards

        pointcloud, *_ = pyminiply.read(scan_path)
        df_points = cudf.DataFrame(pointcloud, columns=["x", "y", "z"])
        nn_model = NearestNeighbors()
        nn_model.fit(df_points)

        for feature in features:
            if features[feature].shape[-1] != pointcloud.shape[0]:
                print(
                    "Scene {} has {} points but {} {} features".format(
                        data_folder, pointcloud.shape[0], features[feature].shape[-1], feature
                    )
                )
                continue

        # calculate number of center points
        n_batches = pointcloud.shape[0] // args.npoints
        n_batches = int(n_batches * FACTOR)

        # handle folders with npy files and no npz files
        if len(existing_batches) == n_batches:
            if os.path.exists(os.path.join(target_scene_path, f"points_{n_batches-1}.npz")):
                print("Skipping", data_folder)
                continue
            # open all seperate files and add them to an uncompressed npz file
            for b in existing_batches:
                batch_path = os.path.join(target_scene_path, b)
                batch_points = np.load(batch_path)
                batch_indices = np.load(batch_path.replace("points", "indices"))
                batch_data = {"points": batch_points, "indices": batch_indices}
                for feature in features:
                    feature_path = batch_path.replace("points", feature)
                    batch_data[feature] = np.load(feature_path)

                # save all data in seperate files using the index
                np.savez(batch_path.replace(".npy", ".npz"), **batch_data)

                # remove the old files
                os.remove(batch_path)
                os.remove(batch_path.replace("points", "indices"))
                for feature in features:
                    feature_path = batch_path.replace("points", feature)
                    os.remove(feature_path)
            continue

        # get center points
        pointcloud = torch.from_numpy(pointcloud).float().cuda()
        pointcloud = rearrange(pointcloud, "n d -> 1 d n")
        center_points = furthest_point_sample(pointcloud, n_batches).squeeze().cpu().numpy().T
        _, idxs = nn_model.kneighbors(center_points, args.npoints)

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
            batch_points = df_points.iloc[neighbor_indices].values
            batch_data["points"] = batch_points
            batch_data["indices"] = neighbor_indices

            # handle features
            for feature in features:
                batch_features = features[feature][:, neighbor_indices.ravel()].astype(np.float16)
                batch_data[feature] = batch_features

            np.savez(batch_path, **batch_data)

        if args.fix:
            # fixing missing files TODO
            existing_batches = [f for f in os.listdir(target_scene_path) if f.startswith("points")]
            for batch in existing_batches:
                batch_idx = int(batch.split("_")[-1].split(".")[0])
                for feature in features:
                    feature_path = os.path.join(target_scene_path, f"{feature}_{batch_idx}.npy")
                    if not os.path.exists(feature_path):
                        missing_indices = np.load(os.path.join(target_scene_path, f"indices_{batch_idx}.npy"))
                        print("Missing feature", feature_path)
                        np.save(feature_path, features[feature][:, missing_indices].astype(np.float16))


if __name__ == "__main__":
    main()
