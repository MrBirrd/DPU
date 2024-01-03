from typing import Dict, List, Tuple, Union
import torch
import torch.nn.init as init
from torch import Tensor
from cuml.neighbors import NearestNeighbors
from einops import rearrange
from sklearn import neighbors
import numpy as np
from modules.functional.sampling import furthest_point_sample
import os


def to_cuda(data, device) -> Union[Tensor, List, Tuple, Dict, None]:
    """
    Moves the input data to the specified device (GPU) if available.

    Args:
        data: The input data to be moved to the device.
        device: The device (GPU) to move the data to.

    Returns:
        The input data moved to the specified device.

    """
    if data is None:
        return None
    if isinstance(data, (list, tuple)):
        return [to_cuda(d, device) for d in data]
    if isinstance(data, dict):
        return {k: to_cuda(v, device) for k, v in data.items()}
    if device is None:
        return data.cuda(non_blocking=True)
    else:
        return data.to(device, non_blocking=True)


def ensure_size(x: Tensor) -> Tensor:
    """
    Ensures that the input tensor has the correct size and dimensions.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The input tensor with the correct size and dimensions.
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)
    assert x.dim() == 3
    if x.size(1) > x.size(2):
        x = x.transpose(1, 2)
    return x


def get_data_batch(batch, cfg):
    """
    Get a batch of data for training or testing.

    Args:
        batch (dict): A dictionary containing the batch data.
        cfg (dict): A dictionary containing the configuration settings.

    Returns:
        dict: A dictionary containing the processed batch data.
    """
    hr_points = batch["hr_points"].transpose(1, 2)

    # load conditioning features
    if not cfg.data.unconditional:
        features = batch["features"] if "features" in batch else None
        lr_points = batch["lr_points"] if "lr_points" in batch else None
    else:
        features, lr_points = None, None

    hr_points = ensure_size(hr_points)
    
    features = ensure_size(features) if features is not None else None
    lr_points = ensure_size(lr_points) if lr_points is not None else None
    
    lr_colors = ensure_size(batch["lr_colors"]) if "lr_colors" in batch else None
    hr_colors = ensure_size(batch["hr_colors"]) if "hr_colors" in batch else None

    # concatenate colors to features
    if lr_colors is not None and lr_colors.shape[-1] > 0 and cfg.data.use_rgb_features:
        features = torch.cat([lr_colors, features], dim=1) if features is not None else lr_colors

    # unconditionals training (no features) at all
    if cfg.data.unconditional:
        features = None
        lr_points = None
    
    assert hr_points.shape == lr_points.shape
    
    return {
        "hr_points": hr_points,
        "lr_points": lr_points,
        "features": features,
    }


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


def create_room_batches_faro(pointcloud, features, n_batches, args):
    # get center points
    pointcloud_torch = torch.from_numpy(pointcloud).float().cuda()
    pointcloud_torch = rearrange(pointcloud_torch, "n d -> 1 d n")
    center_points = furthest_point_sample(pointcloud_torch, n_batches).squeeze().cpu().numpy().T

    pointcloud_tree_cuml = NearestNeighbors(metric="l1" if args.mode == "conditional" else "l2")
    pointcloud_tree_cuml.fit(pointcloud)

    idxs = pointcloud_tree_cuml.kneighbors(center_points, n_neighbors=args.npoints, return_distance=False)

    # generate new data
    data = {}
    for i in range(n_batches):
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
        data[i] = batch_data
    return data        

def create_room_batches_iphone(
    pcd_faro,
    pcd_iphone,
    rgb_faro,
    rgb_iphone,
    features,
    n_batches,
    min_points_faro,
    npoints
    ):
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

    data = []
    
    for idx in range(len(idxs_iphone)):
        faro_batch_points = pcd_faro[idxs_faro[idx]]
        iphone_batch_points = pcd_iphone[idxs_iphone[idx]]
        faro_batch_colors = rgb_faro[idxs_faro[idx]]
        iphone_batch_colors = rgb_iphone[idxs_iphone[idx]]
        iphone_batch_dino = features[idxs_iphone[idx]]

        # skip if the batch is too small
        if len(faro_batch_points) < min_points_faro:
            print(f"Skipping batch {idx} because it is too small ({len(faro_batch_points)} points)")
            continue
        
        diff = npoints - len(iphone_batch_points)
        if diff > 0:
            rand_idx = np.random.randint(0, len(iphone_batch_points), diff)
            iphone_additional_xyz = iphone_batch_points[rand_idx]
            iphone_additional_rgb = iphone_batch_colors[rand_idx]
            iphone_additional_dino = iphone_batch_dino[rand_idx]
            iphone_additional_xyz += np.random.normal(0, 1e-2, iphone_additional_xyz.shape)

            iphone_batch_points = np.concatenate([iphone_batch_points, iphone_additional_xyz])
            iphone_batch_colors = np.concatenate([iphone_batch_colors, iphone_additional_rgb])
            iphone_batch_dino = np.concatenate([iphone_batch_dino, iphone_additional_dino])
        else:
            rand_idx = np.random.randint(0, len(iphone_batch_points), npoints)
            iphone_batch_points = iphone_batch_points[rand_idx]
            iphone_batch_colors = iphone_batch_colors[rand_idx]
            iphone_batch_dino = iphone_batch_dino[rand_idx]

        # assign points using NN
        cn = find_closest_neighbors_cuml(iphone_batch_points, faro_batch_points, k=200)
        assignment = optimize_assignments(iphone_batch_points, faro_batch_points, cn)

        faro_batch_points_assigned = faro_batch_points[assignment]
        faro_batch_colors_assigned = faro_batch_colors[assignment]

        # center the points
        center = iphone_batch_points.mean(axis=0)
        faro_batch_points_assigned -= center
        iphone_batch_points -= center

        # scale the points
        scale = np.max(np.linalg.norm(iphone_batch_points, axis=1))
        faro_batch_points_assigned /= scale
        iphone_batch_points /= scale
        
        batch_data = {}
        batch_data["faro"] = np.concatenate([faro_batch_points_assigned, faro_batch_colors_assigned], axis=1)
        batch_data["iphone"] = np.concatenate([iphone_batch_points, iphone_batch_colors], axis=1)
        batch_data["dino"] = iphone_batch_dino
        batch_data["center"] = center
        batch_data["scale"] = scale
        data.append(batch_data)
    return data

def smart_load_model_weights(model, pretrained_dict):
    # Get the model's state dict
    model_dict = model.state_dict()

    # New state dict
    new_state_dict = {}
    device = model.device

    for name, param in model_dict.items():
        if name in pretrained_dict:
            # Load the pretrained weight
            pretrained_param = pretrained_dict[name]

            if param.size() == pretrained_param.size():
                # If sizes match, load the pretrained weights as is
                new_state_dict[name] = pretrained_param
            else:
                # Handle size mismatch
                # Resize pretrained_param to match the size of param
                reshaped_param = resize_weight(param.size(), pretrained_param, device=device, layer_name=name)
                new_state_dict[name] = reshaped_param
        else:
            # If no pretrained weight, use the model's original weights
            new_state_dict[name] = param

    # Update the model's state dict
    model.load_state_dict(new_state_dict)


def resize_weight(target_size, weight, layer_name="", device="cpu"):
    """
    Resize the weight tensor to the target size.
    Handles different layer types including attention layers.
    Uses Xavier or He initialization for new weights.
    Args:
        target_size: The desired size of the tensor.
        weight: The original weight tensor.
        layer_name: Name of the layer (used to determine initialization strategy).
        device: The target device ('cpu', 'cuda', etc.)
    """
    # Initialize the target tensor on the specified device
    target_tensor = torch.zeros(target_size, device=device)

    # Copy existing weights
    min_shape = tuple(min(s1, s2) for s1, s2 in zip(target_size, weight.shape))
    slice_objects = tuple(slice(0, min_dim) for min_dim in min_shape)
    target_tensor[slice_objects] = weight[slice_objects].to(device)

    # Mask to identify new weights (those that are still zero)
    mask = (target_tensor == 0).type(torch.float32)

    # Initialize new weights
    if "attention" in layer_name or "conv" in layer_name:
        # He initialization for layers typically followed by ReLU
        new_weights = torch.empty(target_size, device=device)
        init.kaiming_uniform_(new_weights, a=0, mode="fan_in", nonlinearity="relu")
    else:
        # Xavier initialization for other layers
        new_weights = torch.empty(target_size, device=device)
        init.xavier_uniform_(new_weights, gain=init.calculate_gain("linear"))

    # Apply the initialization only to new weights
    target_tensor = target_tensor * (1 - mask) + new_weights * mask

    return target_tensor
