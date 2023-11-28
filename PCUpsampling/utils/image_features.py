import torch
import torchvision.transforms as transforms
from typing import Tuple
from torch import Tensor
from einops import rearrange
from third_party.scannetpp.common.utils.colmap import read_model
from third_party.scannetpp.common.scene_release import ScannetppScene_Release
import open3d as o3d
import numpy as np
import json
from tqdm import tqdm
from decord import VideoReader
from decord import cpu
from sklearn.neighbors import KDTree


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


def prepare_image(image, smaller_edge_size: float, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    transform = make_transform(int(smaller_edge_size))

    image_tensor = transform(image)
    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[1:]  # C x H x W
    cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    grid_size = (cropped_height // patch_size, cropped_width // patch_size)  # h x w
    return image_tensor, grid_size


def get_dino_features(model, image, patch_size=14):
    C, H, W = image.shape
    interpolation = transforms.Resize(size=(H, W), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)

    smaller_edge_size = min(H, W)
    image_tensor, grid_size = prepare_image(image, smaller_edge_size, patch_size)

    with torch.inference_mode(), torch.cuda.amp.autocast():
        image_batch = image_tensor.cuda().unsqueeze(0)
        t = model.get_intermediate_layers(image_batch)[0].squeeze().cpu()

    t_min = t.min(dim=0, keepdim=True).values
    t_max = t.max(dim=0, keepdim=True).values
    normalized_t = (t - t_min) / (t_max - t_min)

    features = normalized_t.reshape(*grid_size, -1)
    features = rearrange(features, "h w c -> 1 c h w")
    features = interpolation(features)
    return features


# Pixel Feature Lookup


def project_point_cloud(points, translation, rotation, camera_matrix):
    # Assuming points is already a NumPy array
    points_transformed = np.dot(rotation, points.T) + translation
    points_projected = np.dot(camera_matrix, points_transformed)
    points_projected[:2, :] /= points_projected[2, :]
    return points_projected.T


def filter_points(points_projected, image_width, image_height, min_depth=0.1, max_depth=1000):
    # Check within image bounds and positive depth
    in_image_bounds = (
        (points_projected[:, 0] >= 0)
        & (points_projected[:, 0] < image_width)
        & (points_projected[:, 1] >= 0)
        & (points_projected[:, 1] < image_height)
    )
    valid_depth = (points_projected[:, 2] > min_depth) & (points_projected[:, 2] < max_depth)
    valid_indices = in_image_bounds & valid_depth
    return valid_indices


def map_image_features_to_ptc(image, valid_points_2d, valid_indices):
    # Assuming valid_points_2d is a NumPy array
    x, y = valid_points_2d[:, 0].astype(int), valid_points_2d[:, 1].astype(int)
    rgb_values = image[y, x]  # Efficient bulk operation
    return rgb_values, valid_indices


def update_running_mean(current_mean, current_count, new_value):
    new_count = current_count + 1
    new_mean = current_mean + (new_value - current_mean) / new_count
    return new_mean, new_count


def update_features(feature_list, feature_values, valid_indices):
    # convert valid_indices to indexes
    valid_indices = np.where(valid_indices)[0]
    for feature, valid_index in zip(feature_values, valid_indices):
        if feature_list[valid_index] == []:
            feature_list[valid_index] = (feature, 1)
        else:
            feature_mean, count = feature_list[valid_index]
            feature_mean, count = update_running_mean(feature_mean, count, feature)
            feature_list[valid_index] = (feature_mean, count)
    return feature_list


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
):
    # load up scene configuration
    scene = ScannetppScene_Release(scene_id, data_root=data_root)
    mesh_path = scene.scan_mesh_path

    colmap_dir = scene.iphone_colmap_dir
    cameras, images, points3D = read_model(colmap_dir, ".txt")
    iphone_intrinsics_path = scene.iphone_pose_intrinsic_imu_path
    iphone_intrinsics = json.load(open(iphone_intrinsics_path))

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    pointcloud = o3d.geometry.PointCloud(mesh.vertices)
    points = np.array(pointcloud.points)

    # create videoreader and read the frames
    vr = VideoReader(movie_path, ctx=cpu(0))
    skip_frames = 10  # we only have intrinsics for every 10th frame
    frame_idxs = np.linspace(
        0, vr._num_frame // skip_frames * skip_frames, round(vr._num_frame // skip_frames + 1), dtype=np.int32
    )
    videoframes = vr.get_batch(frame_idxs).asnumpy()

    # initialize list
    ptc_feats = [[] for _ in range(len(points))]

    # create extractors
    if feature_type == "rgb":
        f_shape = 3
        pass
    elif feature_type == "dino":
        f_shape = 384
        model = load_dino(dino_model_name)

    # calculate features
    for image_idx, image in tqdm(images.items(), total=len(images), desc="Processing images"):
        if image_idx % skip_scans != 0:
            continue
        world_to_camera = image.world_to_camera

        # extract video frame
        frame_name = image.name
        frame = int(frame_name.split("_")[-1].split(".")[0]) // 10
        videoframe = videoframes[frame] / 255.0

        # project the mesh on the camera and extract the rgb values from the videoframe
        iphone_data = iphone_intrinsics[frame_name.split(".")[0]]
        intrinsic_matrix = iphone_data["intrinsic"]

        R = world_to_camera[:3, :3]
        t = world_to_camera[:-1, -1:]

        # project points
        points_projected = project_point_cloud(points, t, R, intrinsic_matrix)
        valid_indices = filter_points(points_projected, image_width, image_height)

        # extract features
        if feature_type == "rgb":
            features, valid_indices = map_image_features_to_ptc(
                videoframe, points_projected[valid_indices], valid_indices
            )
        elif feature_type == "dino":
            videoframe = torch.tensor(np.array(videoframe) / 255.0).permute(2, 0, 1).float()
            dino_feats = get_dino_features(model, videoframe).squeeze()
            dino_feats = rearrange(dino_feats, "c h w -> h w c").cpu().numpy()
            features, valid_indices = map_image_features_to_ptc(
                dino_feats, points_projected[valid_indices], valid_indices
            )

        # add the features to the pointcloud
        update_features(ptc_feats, features, valid_indices)

    # clean the misssing features my KNN interpolation
    missing_idx = []
    for idx, f in tqdm(enumerate(ptc_feats), total=len(ptc_feats), desc="Cleaning features"):
        if f != []:
            feat, count = f
            ptc_feats[idx] = np.array(feat)
        else:
            ptc_feats[idx] = np.zeros(f_shape)
            missing_idx.append(idx)

    ptc_feats = np.array(ptc_feats)

    tree = KDTree(points)

    # make batches
    batch_size = 100
    batches = np.array_split(missing_idx, len(missing_idx) // batch_size)

    for batch in batches:
        # find nearest neighbors
        dist, idx = tree.query(points[batch], k=10)
        # replace missing values with mean over nearest neighbors
        neighbors = ptc_feats[idx]
        neighbors_agg = np.median(neighbors, axis=1)
        ptc_feats[batch] = neighbors_agg

    np.save(target_path, ptc_feats)
