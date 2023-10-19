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
    def __init__(self, root, npoints) -> None:
        super().__init__()
        self.root = root
        self.npoints = npoints
        
        # setup the pcd trees
        self.trees = []
        
        # scan paths for ply files
        folders = os.listdir(self.root)
        logger.info("Setting up scannet dataset")
        folders = [f for f in folders if os.path.isdir(os.path.join(self.root, f))]
        for f in tqdm(folders, desc="Loading scans"):
            file = os.path.join(self.root, f, "scans", "mesh_aligned_0.05.ply")
            if os.path.exists(file):
                ply, *_ = pyminiply.read(file)
                pcd_tree = spatial.cKDTree(ply)
                self.trees.append(pcd_tree)
    
    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        pcd_tree = self.trees[index]
        points = pcd_tree.data
        
        # sample k points around randomly chosen point
        rand_idx = np.random.randint(0, len(points))
        rand_point = points[rand_idx]
        _, idx = pcd_tree.query(rand_point, k=self.npoints)
        points = points[idx]
        
        data = {
            "idx": index,
            "train_points": torch.from_numpy(points).float(),
        }
        
        return data