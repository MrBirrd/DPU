from torch.utils.data import Dataset
import torch
import numpy as np
import open3d as o3d
from .utils import concat_nn, cut_by_bounding_box, normalize_lowres_hires_pair, random_local_sample
from loguru import logger
import os
from tqdm import tqdm

def ply_to_np(pcd):
    """Converts a ply file to a numpy array with points and colors"""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    stacked = np.hstack((points, colors))
    return stacked

def apply_transform(array, transformation):
    """Transforms a numpy array of points with a transformation matrix"""
    ones = np.ones((array.shape[0], 1))
    stacked = np.hstack((array, ones))
    transformed = np.dot(transformation, stacked.T)
    transformed = transformed.T[..., :3]
    return transformed

def fp_to_color(array):
    colors = array * 255
    colors = colors.astype(np.uint8)
    return colors

def inverse_T(transformation):
    R = transformation[:3, :3]
    t = transformation[:3, 3]
    inv = np.zeros((4, 4))
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    inv[3, 3] = 1
    return inv


class IndoorScenes(Dataset):
    def __init__(self, root_dir, npoints=1_000_000, voxel_size=0.03, normalize=False):
        self.root = root_dir
        self.npoints = int(npoints)
        self.voxel_size = voxel_size
        self.normalize = normalize


        # specific paths
        arkit_ply = self.root + "42445028_3dod_mesh.ply"
        transformation_matrix = self.root + "42445028_estimated_transform.npy"
        faro_ply = self.root + "421378.ply"
        
        arkit_pcd = o3d.io.read_point_cloud(arkit_ply)
        T_faro_arkit = np.load(transformation_matrix)
        faro_pcd = o3d.io.read_point_cloud(faro_ply)
        
        # downsample to voxel size
        arkit_pcd = arkit_pcd.voxel_down_sample(self.voxel_size)
        faro_pcd = faro_pcd.voxel_down_sample(self.voxel_size)
        logger.info("Downsampled to voxel size: {}".format(self.voxel_size))
        logger.info("Number of points in arkit: {}".format(len(arkit_pcd.points)))
        logger.info("Number of points in faro: {}".format(len(faro_pcd.points)))
        
        arkit_npy = ply_to_np(arkit_pcd)[..., :3]
        faro_npy = ply_to_np(faro_pcd)[..., :3]
        
        faro_npy = apply_transform(faro_npy, T_faro_arkit)
        
        # cut faro outliers w.r.t arkit
        faro_npy = cut_by_bounding_box(arkit_npy, faro_npy)

        # cut the artifacts
        faro_npy = faro_npy[faro_npy[:, 1] < 4.5]
        arkit_npy = arkit_npy[arkit_npy[:, 1] < 4.5]
        
        # normalize to be centered at origin and inside unid sphere
        if self.normalize:
            arkit_npy, faro_npy = normalize_lowres_hires_pair(arkit_npy, faro_npy)

        self.lowres = arkit_npy
        self.hires = faro_npy
        
    def __len__(self):
        return 128

    def __getitem__(self, idx):
        
        # subsample if needed
        if self.npoints < self.hires.shape[0]:
            idxs = np.random.choice(self.hires.shape[0], self.npoints)
            hires = self.hires[idxs, :]
        if self.npoints < self.lowres.shape[0]:
            idxs = np.random.choice(self.lowres.shape[0], self.npoints)
            lowres = self.lowres[idxs, :]
        
        # upsample if needed
        if self.npoints > self.hires.shape[0]:
            points_difference = self.npoints - self.hires.shape[0]
            idxs = np.random.choice(self.hires.shape[0], points_difference)
            hires = np.concatenate((self.hires, self.hires[idxs, :]), axis=0)
        if self.npoints > self.lowres.shape[0]:
            points_difference = self.npoints - self.lowres.shape[0]
            idxs = np.random.choice(self.lowres.shape[0], points_difference)
            lowres = np.concatenate((self.lowres, self.lowres[idxs, :]), axis=0)
        
        lowres = torch.from_numpy(lowres).float()
        hires = torch.from_numpy(hires).float()

        # shuffle points
        idxs = np.random.permutation(self.npoints)
        lowres = lowres[idxs, :]
        hires = hires[idxs, :]

        out = {
            'idx': idx,
            'train_points': hires,
            'train_points_lowres': lowres,
        }
        
        return out

class IndoorScenesCut(Dataset):
    def __init__(self, root_dir, npoints=10000, voxel_size=0.03, normalize=False):
        self.root = root_dir
        self.npoints = int(npoints)
        self.voxel_size = voxel_size
        self.normalize = normalize


        # specific paths
        arkit_ply = self.root + "42445028_3dod_mesh.ply"
        transformation_matrix = self.root + "42445028_estimated_transform.npy"
        faro_ply = self.root + "421378.ply"
        
        arkit_pcd = o3d.io.read_point_cloud(arkit_ply)
        T_faro_arkit = np.load(transformation_matrix)
        faro_pcd = o3d.io.read_point_cloud(faro_ply)
        
        # downsample to voxel size
        arkit_pcd = arkit_pcd.voxel_down_sample(self.voxel_size)
        faro_pcd = faro_pcd.voxel_down_sample(self.voxel_size)
        logger.info("Downsampled to voxel size: {}".format(self.voxel_size))
        logger.info("Number of points in arkit: {}".format(len(arkit_pcd.points)))
        logger.info("Number of points in faro: {}".format(len(faro_pcd.points)))
        
        arkit_npy = ply_to_np(arkit_pcd)[..., :3]
        faro_npy = ply_to_np(faro_pcd)[..., :3]
        
        faro_npy = apply_transform(faro_npy, T_faro_arkit)
        
        # cut faro outliers w.r.t arkit
        faro_npy = cut_by_bounding_box(arkit_npy, faro_npy)

        # cut the outliers manually
        faro_npy = faro_npy[faro_npy[:, 1] < 4.5]
        arkit_npy = arkit_npy[arkit_npy[:, 1] < 4.5]
        
        # normalize to be centered at origin and inside unit sphere
        if self.normalize:
            arkit_npy, faro_npy = normalize_lowres_hires_pair(arkit_npy, faro_npy)

        self.lowres = arkit_npy
        self.hires = faro_npy
        
    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        
        # take a random cube around point
        lowres, hires, center = random_local_sample(self.lowres, self.hires, radius=0.2)
        
        # substract center
        hires -= center
        lowres -= center
        
        # subsample if needed
        if self.npoints < hires.shape[0]:
            idxs = np.random.choice(hires.shape[0], self.npoints)
            hires = hires[idxs, :]
        if self.npoints < lowres.shape[0]:
            idxs = np.random.choice(lowres.shape[0], self.npoints)
            lowres = lowres[idxs, :]
        
        # upsample if needed
        if self.npoints > hires.shape[0]:
            points_difference = self.npoints - hires.shape[0]
            idxs = np.random.choice(hires.shape[0], points_difference)
            hires = np.concatenate((hires, hires[idxs, :]), axis=0)
        if self.npoints > lowres.shape[0]:
            points_difference = self.npoints - lowres.shape[0]
            idxs = np.random.choice(lowres.shape[0], points_difference)
            lowres = np.concatenate((lowres, lowres[idxs, :]), axis=0)
        
        lowres = torch.from_numpy(lowres).float()
        hires = torch.from_numpy(hires).float()

        # shuffle points
        idxs = np.random.permutation(self.npoints)
        lowres = lowres[idxs, :]
        hires = hires[idxs, :]

        out = {
            'idx': idx,
            'train_points': hires,
            'train_points_lowres': lowres,
        }
        
        return out


class ArkitScans(Dataset):
    def __init__(self, root_dir, npoints=10000, voxel_size=0.03, normalize=False):
        self.root = root_dir
        self.npoints = int(npoints)
        self.voxel_size = voxel_size
        self.normalize = normalize


        # specific paths
        folders = os.listdir(self.root)
        logger.info("Setting up arkit scans dataset")
        folders = [f for f in folders if os.path.isdir(os.path.join(self.root, f))]
        self.scans = []
        for f in tqdm(folders, desc="Loading scans"):
            if os.path.exists(os.path.join(self.root, f, "arkit.npy")):
                scan = np.load(os.path.join(self.root, f, "arkit.npy"))
                self.scans.append(scan)

        logger.info("Found {} folders with arkit scans".format(len(self.scans)))
        
    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        # read ply
        scan = self.scans[idx]
        
        # subsample if needed
        if self.npoints < scan.shape[0]:
            idxs = np.random.choice(scan.shape[0], self.npoints)
            scan = scan[idxs, :]
        
        # substract center
        center = scan.mean(axis=0)
        scan -= center
        # put in unit sphere
        scan /= np.max(np.abs(scan))

        # upsample if needed
        if self.npoints > scan.shape[0]:
            points_difference = self.npoints - scan.shape[0]
            idxs = np.random.choice(scan.shape[0], points_difference)
            scan = np.concatenate((scan, scan[idxs, :]), axis=0)

        scan = torch.from_numpy(scan).float()

        out = {
            'idx': idx,
            'train_points': scan,
        }
        
        return out

if __name__ == "__main__":
    ds = IndoorScenes(root_dir="../../3d")
    for data in ds:
        idx = data['idx']
        tp = data['train_points']
        tplr = data['train_points_lowres']
        print("idx: ", idx)
        print("tp: ", tp.shape)
        print("tplr: ", tplr.shape)
        print("stats: ", tp.min(), tp.max(), tplr.min(), tplr.max())
        break