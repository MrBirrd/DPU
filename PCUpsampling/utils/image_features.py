import torch
import torchvision.transforms as transforms
from typing import Tuple
from torch import Tensor
from einops import rearrange
from third_party.scannetpp.common.utils.colmap import read_model
from third_party.scannetpp.common.scene_release import ScannetppScene_Release
import pyminiply
import numpy as np
import json
from tqdm import tqdm
from decord import VideoReader
from decord import cpu
from sklearn.neighbors import KDTree
import os


def load_dino(model_name):
    model = torch.hub.load("facebookresearch/dinov2", model_name).cuda()
    model.eval()
    return model


def make_transform(smaller_edge_size: int) -> transforms.Compose:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    interpolation_mode = transforms.InterpolationMode.BILINEAR

    return transforms.Compose(
        [
            transforms.Resize(size=smaller_edge_size, interpolation=interpolation_mode, antialias=True),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def prepare_image_batched(images, smaller_edge_size: float, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    transform = make_transform(int(smaller_edge_size))

    image_tensor = transform(images)
    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[2:]  # B x C x H x W
    cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
    image_tensor = image_tensor[:, :, :cropped_height, :cropped_width]

    grid_size = (cropped_height // patch_size, cropped_width // patch_size)  # h x w
    return image_tensor, grid_size


def normalize_t(t):
    t_min = t.min(dim=0, keepdim=True).values
    t_max = t.max(dim=0, keepdim=True).values
    normalized_t = (t - t_min) / (t_max - t_min)
    return normalized_t


@torch.inference_mode()
def get_dino_features(model, image, patch_size=14):
    B, C, H, W = image.shape
    smaller_edge_size = min(H, W)

    image_batch, grid_size = prepare_image_batched(image, smaller_edge_size, patch_size)

    t = model.get_intermediate_layers(image_batch)[0].squeeze().detach()

    features = normalize_t(t).reshape(B, grid_size[0], grid_size[1], -1)
    features = rearrange(features, "b h w c -> b c h w")

    # features = torch.cat([torch.nn.functional.interpolate(f.unsqueeze(0), size=(H, W), mode="bilinear") for f in features])
    features = torch.nn.functional.interpolate(
        features, size=(H, W), mode="bilinear", recompute_scale_factor=False, antialias=False
    )
    return features


# Pixel Feature Lookup


def project_point_cloud(points, translation, rotation, camera_matrix):
    # Assuming points is already a NumPy array
    points_transformed = np.dot(rotation, points.T) + translation
    points_projected = np.dot(camera_matrix, points_transformed)
    points_projected[:2, :] /= points_projected[2, :]
    return points_projected.T


def project_point_cloud_batch(points, translation, rotation, camera_matrix):
    # Points: BxNx3
    # Translation: Bx3x1
    # Rotation: Bx3x3
    # Camera Matrix: Bx3x3

    # Apply rotation: Bx3x3 dot BxNx3 -> BxNx3
    points_rotated = np.matmul(rotation, points.transpose(0, 2, 1))

    # Apply translation: BxNx3 + Bx3x1 -> BxNx3
    points_transformed = points_rotated + translation

    # Apply camera matrix: Bx3x3 dot BxNx3 -> BxNx3
    points_projected = np.matmul(camera_matrix, points_transformed)

    # Normalize: Divide by the z-coordinate to project onto the image plane
    points_projected = points_projected.transpose(0, 2, 1)
    points_projected[..., :2] /= points_projected[..., 2:3]

    return points_projected


def filter_points_batch(points_projected, image_width, image_height, min_depth=0.1, max_depth=1000):
    # points_projected is of shape BxNx3, where the last dimension is [x, y, depth]

    # Check within image bounds for the entire batch
    in_image_bounds = (
        (points_projected[:, :, 0] >= 0)
        & (points_projected[:, :, 0] < image_width)
        & (points_projected[:, :, 1] >= 0)
        & (points_projected[:, :, 1] < image_height)
    )

    # Check valid depth for the entire batch
    valid_depth = (points_projected[:, :, 2] > min_depth) & (points_projected[:, :, 2] < max_depth)

    # Combine the conditions for the entire batch
    valid_indices = in_image_bounds & valid_depth

    return valid_indices


def map_image_features_to_filtered_ptc_batch(images, projected_points_batch, valid_indices_batch):
    # images: BxHxWxC - a batch of images
    # projected_points_batch: BxNx3 - a batch of projected 3D points
    # valid_indices_batch: BxN - a batch of boolean valid indices

    # Initialize a list to store the mapped features for each batch
    mapped_features_batch = []

    # Process each batch
    for image, projected_points, valid_indices in zip(images, projected_points_batch, valid_indices_batch):
        # Filter out valid 2D points using valid indices
        valid_points_2d = projected_points[valid_indices, :2]

        # Extract x and y coordinates, ensuring they are within the image bounds
        x, y = valid_points_2d[:, 0].astype(int), valid_points_2d[:, 1].astype(int)
        x = np.clip(x, 0, image.shape[1] - 1)
        y = np.clip(y, 0, image.shape[0] - 1)

        # Extract the features (e.g., RGB values) from the image at these coordinates
        rgb_values = image[y, x]

        # Append the features to the batch list
        mapped_features_batch.append(rgb_values)

    return mapped_features_batch


def update_features_batched(feature_array, count_array, new_features, valid_indices):
    # feature_array: NxF - Pre-initialized feature array for running mean
    # count_array: N - Pre-initialized array for counts
    # new_features: BxMxF - Array of new features for each batch
    # valid_indices: BxN - Boolean array of valid indices for each batch

    for batch_new_features, batch_valid_indices in zip(new_features, valid_indices):
        # Get indices where valid_indices is True
        indices = np.where(batch_valid_indices)[0]

        # Increment counts
        count_array[indices] += 1

        # Extract the current counts and features for valid indices
        current_counts = count_array[indices]
        current_means = feature_array[indices]

        # Update the running mean
        # Calculate the increment for the mean
        increment = (batch_new_features - current_means) / current_counts

        # Apply the increment
        feature_array[indices] += increment

    return feature_array, count_array


def interpolate_missing_features(ptc_feats, ptc_feats_count, points, f_shape, batch_size=128):
    # ptc_feats: NxF - Feature array
    # ptc_feats_count: N - Array tracking the count of each feature
    # points: Nx3 - Array of point cloud coordinates

    # Find indices of missing features (where count is 0)
    missing_idx = np.where(ptc_feats_count == 0)[0]

    if len(missing_idx) == 0:
        return ptc_feats
    
    # Create a KDTree for nearest neighbor search
    tree = KDTree(points)

    # Optimize batch size based on the size of missing_idx
    # Adjust this value based on memory constraints and dataset size
    batch_size = min(batch_size, len(missing_idx))

    # Create batches of missing indices
    batches = np.array_split(missing_idx, max(len(missing_idx) // batch_size, 1))

    # Perform KNN interpolation in batches
    for batch in tqdm(batches, total=len(batches), desc="KNN interpolation"):
        # Find nearest neighbors (10 nearest neighbors)
        _, idx = tree.query(points[batch], k=10)

        # Filter out zero features among neighbors
        # This has to be done in a loop because the number of neighbors is different for each point

        for batch_idx, neighbor_batch in enumerate(idx):
            nonzero_neighbors = ptc_feats[neighbor_batch, :]
            nonzero_neighbors_mask = np.any(nonzero_neighbors != np.zeros(f_shape), axis=-1)
            nonzero_neighbors = nonzero_neighbors[nonzero_neighbors_mask]
            if nonzero_neighbors.shape[0] == 0:
                neighbors_agg = np.zeros(f_shape)
            else:
                neighbors_agg = np.median(nonzero_neighbors, axis=-2)
            ptc_feats[batch[batch_idx]] = neighbors_agg

    return ptc_feats


def process_scene(
    scene_id,
    data_root,
    movie_path,
    target_path,
    feature_type,
    skip_scans: int = 5,
    image_width: int = 1920,
    image_height: int = 1440,
    dino_model_name: str = "dinov2_vits14",
    overwrite: bool = False,
):
    if os.path.exists(target_path + ".npy") and not overwrite:
        print("Already processed scene", scene_id)
        return

    # load up scene configuration
    scene = ScannetppScene_Release(scene_id, data_root=data_root)
    mesh_path = scene.scan_mesh_path

    colmap_dir = scene.iphone_colmap_dir
    cameras, images, points3D = read_model(colmap_dir, ".txt")
    iphone_intrinsics_path = scene.iphone_pose_intrinsic_imu_path
    iphone_intrinsics = json.load(open(iphone_intrinsics_path))

    ply, *_ = pyminiply.read(str(mesh_path))
    # remove nans or infs
    ply = ply[~np.isnan(ply).any(axis=1)]
    ply = ply[~np.isinf(ply).any(axis=1)]
    points = ply[:, :3]

    # create videoreader and read the frames
    vr = VideoReader(movie_path, ctx=cpu(0))

    # create extractors
    if feature_type == "rgb":
        f_shape = 3
        pass
    elif feature_type == "dino":
        f_shape = 384
        print("Loading DINO model")
        model = load_dino(dino_model_name)

    # initialize features and set count to 0
    ptc_feats = np.zeros((len(points), f_shape), dtype=np.float16)
    ptc_feats_count = np.zeros((len(points), 1), dtype=np.int32)

    # calculate features in batches, first skip every nth scan
    batch_size = 2
    total_data = len(images)
    
    if total_data < 30:
        skip_scans = 1
    elif total_data < 60:
        skip_scans = 2
    elif total_data < 90:
        skip_scans = 3
    elif total_data < 160:
        skip_scans = 4
    else:
        skip_scans = 5
    
    images = list(images.values())
    images = images[::skip_scans]

    # recalculate total data
    total_data = len(images)
    
    # split into batches of maximum shape of batch_size
    num_batches = int(np.ceil(total_data / batch_size))
    batches = np.array_split(np.arange(total_data), num_batches)
    
    for batch in tqdm(batches, total=len(batches), desc="Processing images"):
        # expand dims to batch size
        points_batch = np.expand_dims(points, axis=0).repeat(len(batch), axis=0)
        
        # create batch of frames
        frame_names = [images[i].name for i in batch]
        frames = [int(frame_name.split("_")[-1].split(".")[0]) for frame_name in frame_names]

        # seek the video frames
        videoframes = vr.get_batch(frames).asnumpy() / 255.0

        # get intrinsics and extrinsics
        intrinsic_matrices = np.array(
            [
                iphone_data["intrinsic"]
                for iphone_data in [iphone_intrinsics[frame_name.split(".")[0]] for frame_name in frame_names]
            ]
        )
        
        world_to_cameras = [images[i].world_to_camera for i in batch]
        Rs = np.array([world_to_camera[:3, :3] for world_to_camera in world_to_cameras])
        ts = np.array([world_to_camera[:-1, -1:] for world_to_camera in world_to_cameras])

        points_projected = project_point_cloud_batch(points_batch, ts, Rs, intrinsic_matrices)
        valid_indices = filter_points_batch(points_projected, image_width, image_height)

        # extract features
        if feature_type == "rgb":
            features = map_image_features_to_filtered_ptc_batch(videoframes, points_projected, valid_indices)
        elif feature_type == "dino":
            with torch.cuda.amp.autocast(enabled=True, cache_enabled=False):
                videoframes = torch.tensor(videoframes).permute(0, 3, 1, 2).type(torch.float16).cuda()
                dino_feats = get_dino_features(model, videoframes)

            dino_feats = rearrange(dino_feats, "b c h w -> b h w c").cpu().numpy()
            features = map_image_features_to_filtered_ptc_batch(dino_feats, points_projected, valid_indices)

        update_features_batched(ptc_feats, ptc_feats_count, features, valid_indices)

    ptc_feats = interpolate_missing_features(ptc_feats, ptc_feats_count, points, f_shape)
    np.nan_to_num(ptc_feats, copy=False)
    np.save(target_path, ptc_feats.astype(np.float16))
