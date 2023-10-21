from .utils import *
from loguru import logger
import os
from scipy import spatial
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
import pyminiply


class ScanNetPPCut(Dataset):
    def __init__(self, root, npoints, mode="training") -> None:
        super().__init__()
        self.root = root
        self.npoints = npoints

        # setup the pcd trees
        self.trees = []

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
        for idx, f in enumerate(tqdm(folders, desc="Loading scans")):
            # skip if not in split
            if f not in scans:
                continue
            # else read the ply file and generate a tree
            file = os.path.join(self.root, f, "scans", "mesh_aligned_0.05.ply")
            if os.path.exists(file):
                ply, *_ = pyminiply.read(file)
                # remove nans or infs
                ply = ply[~np.isnan(ply).any(axis=1)]
                ply = ply[~np.isinf(ply).any(axis=1)]
                # generate the tree
                pcd_tree = spatial.cKDTree(ply)
                self.trees.append(pcd_tree)

            if idx > 100:
                break

        logger.info(f"Loaded {len(self.trees)} scans")

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        pcd_tree = self.trees[index]
        points = pcd_tree.data

        # sample k points around randomly chosen point
        rand_idx = np.random.randint(0, len(points))
        rand_point = points[rand_idx]
        _, idx = pcd_tree.query(rand_point, k=self.npoints, p=2)
        points = points[idx]

        # normalize the points
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

        return data
