import os
import random
from typing import Tuple

import numpy as np
import torch
from scipy import spatial
from torch import nn

from modules.functional.devoxelization import trilinear_devoxelize
from modules.voxelization import Voxelization


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


class FeatureVoxelConcatenation(nn.Module):
    """
    FeatureVoxelConcatenation
    assumes tensors of shape (B, C, N)
    """

    def __init__(self, resolution, normalize=True):
        super().__init__()
        self.resolution = resolution
        self.vox = Voxelization(resolution=resolution, normalize=normalize, eps=0)

    def forward(self, x1_features, x2_features, x1_coords, x2_coords):
        vox_x1, nc_x1 = self.vox(features=x1_features, coords=x1_coords)
        vox_x2, nc_x2 = self.vox(features=x2_features, coords=x2_coords)

        devox_mixed = trilinear_devoxelize(vox_x2, nc_x1, self.resolution)

        return torch.cat([x1_features, devox_mixed], dim=1)


def random_local_sample(x1, x2, k=None, radius=None):
    """
    Takes a random point and then takes k closest points within around that point.
    """
    center = random.choice(x1)
    # build KDTree for x1 and x2
    tree1 = spatial.cKDTree(x1)
    tree2 = spatial.cKDTree(x2)
    # query tree for x1
    indices1 = tree1.query_ball_point(center, r=radius, p=1)
    # query tree for x2
    indices2 = tree2.query_ball_point(center, r=radius, p=1)
    # extract points
    x1_sample = x1[indices1]
    x2_sample = x2[indices2]
    return x1_sample, x2_sample, center


def concat_nn(x1, x2):
    """
    Concatenate two point clouds by nearest neighbor search. This means that for each point in x1 it searches it's closest point in x2 and appends the features of that point to the features of the point in x1.
    """
    assert x1.shape[1] == x2.shape[1], "Point clouds must have same dimension!"
    # build KDTree for x2
    tree = spatial.cKDTree(x2)
    # query tree for x1
    dists, indices = tree.query(x1)
    # concatenate x1 and x2
    return np.concatenate((x1, x2[indices]), axis=1)


def cut_by_bounding_box(reference, target):
    """
    Cut the target point cloud by the bounding box of the reference point cloud.

    Args:
        reference (np.ndarray): The reference point cloud.
        target (np.ndarray): The target point cloud.

    Returns:
        np.ndarray: The target point cloud cut by the bounding box of the reference point cloud.
    """
    assert reference.shape[1] == target.shape[1], "Reference and target point clouds must have same dimension!"
    # calculate bounding box of reference
    mins = np.min(reference, axis=0)
    maxs = np.max(reference, axis=0)
    # cut target by bounding box of reference
    target = target[
        (target[:, 0] >= mins[0])
        & (target[:, 0] <= maxs[0])
        & (target[:, 1] >= mins[1])
        & (target[:, 1] <= maxs[1])
        & (target[:, 2] >= mins[2])
        & (target[:, 2] <= maxs[2])
    ]
    return target


def normalize_lowres_hires_pair(lowres: np.ndarray, hires: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize a pair of low-resolution and high-resolution point clouds by centering them around their centroid and scaling them by their maximum distance from the centroid.

    Args:
        lowres (np.ndarray): The low-resolution point cloud.
        hires (np.ndarray): The high-resolution point cloud.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the normalized low-resolution and high-resolution point clouds.
    """
    assert lowres.shape[1] == hires.shape[1], "Lowres and hires point clouds must have same dimension!"
    # calculate centroid of lowres
    centroid = np.mean(lowres, axis=0)
    # subtract centroid from both
    lowres -= centroid
    hires -= centroid
    # calculate max distance from centroid
    max_dist = max(np.max(np.linalg.norm(lowres, axis=1)), np.max(np.linalg.norm(hires, axis=1)))
    # divide both by max dist
    lowres /= max_dist
    hires /= max_dist
    return lowres, hires


def load_npz_folder(folder):
    """Load a folder of .npz files."""
    data = []
    for file in os.listdir(folder):
        if file.endswith(".npz"):
            data.append(load_npz(os.path.join(folder, file)))
    return data


def load_npz(path):
    """Load a .npz file."""
    data = np.load(path)
    return data
