import os

import numpy as np
import torch
import torch.nn.init as init
from cuml.neighbors import NearestNeighbors
from einops import rearrange
from sklearn import neighbors

from pvcnn.functional.sampling import furthest_point_sample


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


def create_room_batches_inference(pointcloud, features, args):
    pass

def create_room_batches_training_faro(pointcloud, features, n_batches, args):
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


def create_room_batches_training_iphone_v1(
    pcd_faro,
    pcd_iphone,
    rgb_faro,
    rgb_iphone,
    features,
    args
):
    tree_faro = neighbors.KDTree(pcd_faro, metric="l2")
    tree_iphone = neighbors.KDTree(pcd_iphone, metric="l2")

    # calculate number of center points
    n_batches = int(pcd_iphone.shape[0] * args.centers_amount)
    
    # get center points of batches
    pointcloud_torch = torch.from_numpy(pcd_iphone).float().cuda()
    pointcloud_torch = rearrange(pointcloud_torch, "n d -> 1 d n")
    center_points = furthest_point_sample(pointcloud_torch, n_batches).squeeze().cpu().numpy().T

    # first query points in radius
    idxs_faro = tree_faro.query_radius(center_points, r=1, return_distance=False)
    idxs_iphone = tree_iphone.query_radius(center_points, r=1, return_distance=False)

    assert len(idxs_faro) == len(idxs_iphone) == n_batches, "Number of batches is not equal to number of indices"

    data = []

    for idx in range(len(idxs_iphone)):        
        faro_batch_points = pcd_faro[idxs_faro[idx]]
        iphone_batch_points = pcd_iphone[idxs_iphone[idx]]
        faro_batch_colors = rgb_faro[idxs_faro[idx]]
        iphone_batch_colors = rgb_iphone[idxs_iphone[idx]]
        iphone_batch_dino = features[idxs_iphone[idx]]

        # skip if the batch is too small
        if len(faro_batch_points) < args.npoints:
            print(f"Skipping batch {idx} because it is too small ({len(faro_batch_points)} points)")
            continue

        diff = args.npoints - len(iphone_batch_points)
        
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
            rand_idx = np.random.randint(0, len(iphone_batch_points), args.npoints)
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

        # create output data to save as npz later
        batch_data = {}
        batch_data["faro"] = np.concatenate([faro_batch_points_assigned, faro_batch_colors_assigned], axis=1)
        batch_data["iphone"] = np.concatenate([iphone_batch_points, iphone_batch_colors], axis=1)
        batch_data["dino"] = iphone_batch_dino
        batch_data["center"] = center
        batch_data["scale"] = scale
        data.append(batch_data)
    return data


def create_room_batches_training_iphone_v2(
    pcd_faro,
    pcd_iphone,
    rgb_faro,
    rgb_iphone,
    features,
    args
):
    tree_faro = neighbors.KDTree(pcd_faro, metric="l2")
    tree_iphone = neighbors.KDTree(pcd_iphone, metric="l2")

    n_batches  = pcd_faro.shape[0] // args.npoints
    
    # get center points of batches from faro, such that we don't get empty correspondences
    pointcloud_torch = torch.from_numpy(pcd_faro).float().cuda()
    pointcloud_torch = rearrange(pointcloud_torch, "n d -> 1 d n")
    center_points = furthest_point_sample(pointcloud_torch, n_batches).squeeze().cpu().numpy().T

    # query points from iphone scan around the center points from faro
    dists_lr, idxs_iphone = tree_iphone.query(center_points, k=args.npoints // args.upsampling_rate)
    max_dist_lr = dists_lr.max(axis=-1, keepdims=True).squeeze()
    idxs_faro = tree_faro.query_radius(center_points, max_dist_lr)
    
    data = []
    
    for idx in range(len(idxs_iphone)):        
        faro_batch_points = pcd_faro[idxs_faro[idx]]
        iphone_batch_points = pcd_iphone[idxs_iphone[idx]]
        faro_batch_colors = rgb_faro[idxs_faro[idx]]
        iphone_batch_colors = rgb_iphone[idxs_iphone[idx]]
        iphone_batch_dino = features[idxs_iphone[idx]]

        # skip if the batch is too small
        if len(faro_batch_points) < args.npoints:
            print(f"Skipping batch {idx} because it is too small ({len(faro_batch_points)} points)")
            continue
        
        # randomly downsample the faro scan to npoints
        rand_idx = np.random.randint(0, len(faro_batch_points), args.npoints)
        faro_batch_points = faro_batch_points[rand_idx]
        faro_batch_colors = faro_batch_colors[rand_idx]
        
        # randomly upsample the iphone scan to npoints
        diff = args.npoints - len(iphone_batch_points)

        rand_idx = np.random.randint(0, len(iphone_batch_points), diff)
        iphone_additional_xyz = iphone_batch_points[rand_idx]
        iphone_additional_rgb = iphone_batch_colors[rand_idx]
        iphone_additional_dino = iphone_batch_dino[rand_idx]
        
        # add noise to the points and features
        iphone_additional_xyz += np.random.normal(0, 1e-2, iphone_additional_xyz.shape)
        iphone_additional_dino += np.random.normal(0, 1e-2, iphone_additional_dino.shape)

        iphone_batch_points = np.concatenate([iphone_batch_points, iphone_additional_xyz])
        iphone_batch_colors = np.concatenate([iphone_batch_colors, iphone_additional_rgb])
        iphone_batch_dino = np.concatenate([iphone_batch_dino, iphone_additional_dino])

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

        # create output data to save as npz later
        batch_data = {}
        batch_data["faro"] = np.concatenate([faro_batch_points_assigned, faro_batch_colors_assigned], axis=1)
        batch_data["iphone"] = np.concatenate([iphone_batch_points, iphone_batch_colors], axis=1)
        batch_data["dino"] = iphone_batch_dino
        batch_data["center"] = center
        batch_data["scale"] = scale
        data.append(batch_data)
    return data


def filter_iphone_scan(iphone_scan, iphone_colors, features, faro_scan, threshold=1.0):
    """
    Filter the iPhone scan to remove points that are too far from the Faro scan.
    This is done by calculating the nearest neighbor distances between the two scans,
    and then removing points that are more than the specified threshold times the std away from the mean.
    """
    tree_faro = neighbors.KDTree(faro_scan, metric="l2")
    
    distances = tree_faro.query(iphone_scan, k=1, return_distance=True)[0].ravel()
    mean = distances.mean()
    std = distances.std()
    threshold = mean + threshold * std
    mask = distances < threshold
    return iphone_scan[mask], iphone_colors[mask], features[mask]


def smart_load_model_weights(model, pretrained_dict):
    """
    Loads pretrained weights into a model's state dictionary, handling size mismatches if necessary.

    Args:
        model (nn.Module): The model to load the weights into.
        pretrained_dict (dict): A dictionary containing the pretrained weights.

    Returns:
        None
    """
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


def setup_output_subdirs(output_dir, *subfolders):
    """
    Creates the output directory and subdirectories if they don't exist.

    Args:
        output_dir (str): The path to the output directory.
        *subfolders (str): Variable number of subfolder names.

    Returns:
        list: A list of the created subdirectory paths.
    """
    output_subdirs = output_dir
    try:
        os.makedirs(output_subdirs)
    except OSError:
        pass

    subfolder_list = []
    for sf in subfolders:
        curr_subf = os.path.join(output_subdirs, sf)
        try:
            os.makedirs(curr_subf)
        except OSError:
            pass
        subfolder_list.append(curr_subf)

    return subfolder_list
