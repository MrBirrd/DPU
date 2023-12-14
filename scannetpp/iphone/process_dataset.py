import os
import subprocess
import argparse


import argparse
from pathlib import Path
import yaml
from munch import Munch
from tqdm import tqdm
import json
import sys
import subprocess
import zlib
import numpy as np
import imageio as iio
import lz4.block
from PIL import Image
from iphone.arkit_pcl import backproject, outlier_removal, voxel_down_sample, save_point_cloud
from common.scene_release import ScannetppScene_Release
from common.utils.utils import run_command, load_yaml_munch, load_json, read_txt_list
import plyfile


def extract_rgb(scene):
    scene.iphone_rgb_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {scene.iphone_video_path} -start_number 0 -q:v 1 {scene.iphone_rgb_dir}/frame_%06d.jpg"
    run_command(cmd, verbose=True)


def extract_masks(scene):
    scene.iphone_video_mask_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {str(scene.iphone_video_mask_path)} -pix_fmt gray -start_number 0 {scene.iphone_video_mask_dir}/frame_%06d.png"
    run_command(cmd, verbose=True)


def extract_depth(scene, sample_rate=1):
    # global compression with zlib
    height, width = 192, 256
    scene.iphone_depth_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(scene.iphone_depth_path, "rb") as infile:
            data = infile.read()
            data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
            depth = np.frombuffer(data, dtype=np.float32).reshape(-1, height, width)

        for frame_id in tqdm(range(0, depth.shape[0], sample_rate), desc="decode_depth"):
            iio.imwrite(f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png", (depth * 1000).astype(np.uint16))

    # per frame compression with lz4/zlib
    except Exception as e:
        frame_id = 0
        with open(scene.iphone_depth_path, "rb") as infile:
            while True:
                size = infile.read(4)  # 32-bit integer
                if len(size) == 0:
                    break
                size = int.from_bytes(size, byteorder="little")
                if frame_id % sample_rate != 0:
                    infile.seek(size, 1)
                    frame_id += 1
                    continue

                # read the whole file
                data = infile.read(size)
                try:
                    # try using lz4
                    data = lz4.block.decompress(data, uncompressed_size=height * width * 2)  # UInt16 = 2bytes
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
                except:
                    # try using zlib
                    data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                    depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
                    depth = (depth * 1000).astype(np.uint16)

                # 6 digit frame id = 277 minute video at 60 fps
                iio.imwrite(f"{scene.iphone_depth_dir}/frame_{frame_id:06}.png", depth)
                frame_id += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--sr", type=int, default=30)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--grid_size", type=float, default=0.05, help="Grid size for voxel downsampling.")
    args = parser.parse_args()

    scenes_root = os.path.join(args.data_root, "data")
    scene_ids = [item for item in os.listdir(scenes_root) if os.path.isdir(os.path.join(scenes_root, item))]

    scenes_filtered = []

    # check if we have a faro scan inside
    for scene_id in scene_ids:
        scene_path = os.path.join(scenes_root, scene_id)
        faro_path = os.path.join(scene_path, "scans", "mesh_aligned_0.05.ply")
        if os.path.exists(faro_path):
            scenes_filtered.append(scene_id)

    scenes_filtered.sort()

    # process the scenes
    for scene_id in tqdm(scenes_filtered, desc="Scenes"):
        # extract the frames and depth
        print("#" * 50)
        print("Processing scene: ", scene_id)
        print("#" * 50)
        scene = ScannetppScene_Release(scene_id, data_root=scenes_root)
        extract_rgb(scene)
        print("Extracted RGB")
        print("#" * 50)
        extract_depth(scene, sample_rate=args.sr)
        print("Extracted Depth")
        print("#" * 50)

        iphone_rgb_dir = scene.iphone_rgb_dir
        iphone_depth_dir = scene.iphone_depth_dir

        with open(scene.iphone_pose_intrinsic_imu_path, "r") as f:
            json_data = json.load(f)
        frame_data = [(frame_id, data) for frame_id, data in json_data.items()]
        frame_data.sort()

        all_xyz = []
        all_rgb = []

        for frame_id, data in tqdm(frame_data[:: args.sr]):
            camera_to_world = np.array(data["aligned_pose"]).reshape(4, 4)
            intrinsic = np.array(data["intrinsic"]).reshape(3, 3)
            rgb = np.array(Image.open(os.path.join(iphone_rgb_dir, frame_id + ".jpg")), dtype=np.uint8)
            depth = np.array(Image.open(os.path.join(iphone_depth_dir, frame_id + ".png")), dtype=np.float32) / 1000.0

            xyz, rgb = backproject(
                rgb,
                depth,
                camera_to_world,
                intrinsic,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                use_point_subsample=False,
                use_voxel_subsample=True,
                voxel_grid_size=args.grid_size,
            )
            all_xyz.append(xyz)
            all_rgb.append(rgb)

        all_xyz = np.concatenate(all_xyz, axis=0)
        all_rgb = np.concatenate(all_rgb, axis=0)

        # Voxel downsample again
        all_xyz, all_rgb, _ = outlier_removal(all_xyz, all_rgb, nb_points=10, radius=0.1)
        all_xyz, all_rgb = voxel_down_sample(all_xyz, all_rgb, voxel_size=args.grid_size)

        iphone_scan_path = os.path.join(scene.data_root, scene_id, "scans", "iphone.ply")
        save_point_cloud(
            filename=iphone_scan_path,
            points=all_xyz,
            rgb=all_rgb,
            binary=True,
            verbose=True,
        )
        print("Saved point cloud with {} points tp {}".format(all_xyz.shape[0], iphone_scan_path))
        # remove the extracted frames and depth
        os.system("rm -rf {}".format(iphone_rgb_dir))
        os.system("rm -rf {}".format(iphone_depth_dir))


if __name__ == "__main__":
    main()
