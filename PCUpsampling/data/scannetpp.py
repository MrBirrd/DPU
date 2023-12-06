from .utils import *
from loguru import logger
import os
from scipy import spatial
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
import pyminiply
from utils.ops import random_rotate_pointcloud_horizontally
EULER_FEATURE_ROOT = "/cluster/scratch/matvogel/scannetpp"
import time

class ScanNetPPProcessed(Dataset):
    def __init__(self, root, mode="training", features=None, augment=False) -> None:
        super().__init__()
        self.root = root
        self.mode = mode
        if features not in ["dino", "rgb", "dino_svd64"]:
            features = None
        
        self.features = features
        self.augment = augment
        
        splits_path = os.path.join(root, "splits")
        with open(os.path.join(splits_path, "train.txt"), "r") as f:
            train_scans = f.read().splitlines()
        with open(os.path.join(splits_path, "val.txt"), "r") as f:
            val_scans = f.read().splitlines()
            
        # setup the splits
        if mode == "training":
            scans = train_scans
        elif mode == "validation":
            scans = val_scans
        else:
            raise NotImplementedError(f"Mode {mode} not implemented!")

        # scan paths for ply files
        folders = os.listdir(self.root)
        logger.info(f"Setting up preprocessed {mode} scannet dataset")
        folders = [f for f in folders if os.path.isdir(os.path.join(self.root, f))]
        folders = [f for f in folders if f in scans]
        
        self.scene_batches = []
        
        for idx, folder in enumerate(folders):
            folder_files = os.listdir(os.path.join(self.root, folder))
            points_paths = sorted([f for f in folder_files if f.startswith("points") and f.endswith(".npz")])
            for points in points_paths:
                data = {
                    "scene": folder,
                    "npz": os.path.join(self.root, folder, points),
                }
                self.scene_batches.append(data)
        
        logger.info(f"Loaded {len(self.scene_batches)} batches")
        
    def __len__(self):
        return len(self.scene_batches)

    def __getitem__(self, index):
        
        batch_data = {}
        
        # try to load the data and retry if it fails
        while True:
            try:
                data = self.scene_batches[index]
                data_dict = np.load(data["npz"])
                points = data_dict["points"]
                # append the features if they are available
                if self.features is not None:
                    features = data_dict[self.features]
                    batch_data["features"] = torch.from_numpy(features).float()
                break
            except Exception as e:
                logger.error(f"Failed to load data {data}")
                self.scene_batches.pop(index)
                index = np.random.randint(0, len(self.scene_batches))
                
        # normalize the point coordinates
        center = np.mean(points, axis=0)
        points -= center
        scale = np.max(np.linalg.norm(points, axis=1))
        points /= scale
        # random rotation augmentation
        if self.augment and np.random.rand() < 0.2:
            points, theta = random_rotate_pointcloud_horizontally(points)
        
        batch_data["idx"] = index
        batch_data["train_points"] = torch.from_numpy(points).float()
        batch_data["train_points_center"] = center
        batch_data["train_points_scale"] = scale
        
        return batch_data
        
class ScanNetPPCut(Dataset):
    def __init__(self, root, npoints, mode="training", features=None) -> None:
        super().__init__()
        self.root = root
        self.npoints = npoints

        # setup the pcd trees
        self.data = []

        # load the splits which are located in the splits folder at same level
        script_path = os.path.dirname(os.path.realpath(__file__))
        splits_path = os.path.join(script_path, "splits")
        with open(os.path.join(splits_path, "scannetpp_train.txt"), "r") as f:
            train_scans = f.read().splitlines()
        with open(os.path.join(splits_path, "scannetpp_eval.txt"), "r") as f:
            val_scans = f.read().splitlines()

        # setup the splits
        if mode == "training":
            scans = train_scans
        elif mode == "validation":
            scans = val_scans
        else:
            raise NotImplementedError(f"Mode {mode} not implemented!")

        # scan paths for ply files
        folders = os.listdir(self.root)
        logger.info("Setting up scannet dataset")
        folders = [f for f in folders if os.path.isdir(os.path.join(self.root, f))]
        folders = [f for f in folders if f in scans]

        for idx, f in enumerate(tqdm(folders, desc=f"Loading {mode} scans")):
            file = os.path.join(self.root, f, "scans", "mesh_aligned_0.05.ply")
            feature_file = os.path.join(EULER_FEATURE_ROOT, f, "features", f"{features}.npy") if features is not None else None

            # check if the files exists
            valid_scan = (
                os.path.exists(file) if features is None else os.path.exists(file) and os.path.exists(feature_file)
            )

            if valid_scan:
                # read the data
                pointcloud, *_ = pyminiply.read(file)

                # remove nans or infs
                #mask = ~np.isnan(pointcloud).any(axis=1) | ~np.isinf(pointcloud).any(axis=1)

                # filter
                #pointcloud = pointcloud[mask]
                # generate the tree
                pcd_tree = spatial.cKDTree(pointcloud)

                data = {
                    "tree": pcd_tree,
                    "scene": f,
                    "feature_path": feature_file,
                }
                self.data.append(data)

        logger.info(f"Loaded {len(self.data)} scans")

    def __len__(self):
        return len(self.data) * 128

    def __getitem__(self, index):
        iteration_data = self.data[index // 128]

        pcd_tree = iteration_data["tree"]
        feature_file = iteration_data["feature_path"]

        # extract points and features
        points = pcd_tree.data

        # sample k points around randomly chosen point
        rand_idx = np.random.randint(0, len(points))
        rand_point = points[rand_idx]
        _, idx = pcd_tree.query(rand_point, k=self.npoints, p=2)
        points = points[idx]

        # load the features
        if feature_file is not None:
            features = np.load(feature_file, mmap_mode="r")
            features = features[:, idx]
            np.nan_to_num(features, copy=False, nan=0.0)
            features = torch.from_numpy(features).float()
        else:
            features = None

        # normalize the point coordinates
        center = np.mean(points, axis=0)
        points -= center
        scale = np.max(np.linalg.norm(points, axis=1))
        points /= scale

        data = {
            "idx": index,
            "train_points": torch.from_numpy(points).float(),
            "train_points_center": center,
            "train_points_scale": scale,
        }

        # append the features if they are available
        if features is not None:
            data["features"] = features

        return data
