import argparse
import gc
import os
import traceback

import torch

from utils.image_features import process_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--scenes_files",
        type=str,
        default=None,
        help="Path to the scenes files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing features.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="rgb",
        help="Type of features to extract.",
        choices=["rgb", "dino", "clip"],
    )
    parser.add_argument(
        "--feature_suffix",
        type=str,
        default="",
        help="Suffix to add to the feature name.",
    )
    parser.add_argument(
        "--source_cloud",
        type=str,
        default="iphone",
        help="Source of the pointcloud coordinates.",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=20,
        help="Number of scans to skip between each scan.",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=1920,
        help="Width of the images in the video.",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=1440,
        help="Height of the images in the video.",
    )
    parser.add_argument(
        "--dino_model_name",
        type=str,
        default="dinov2_vits14",
        help="Name of the DINO model to use.",
    )
    args = parser.parse_args()

    # deactivate xformers
    os.environ["XFORMERS_DISABLED"] = "0"

    if args.scenes_files is not None:
        with open(args.scenes_files, "r") as f:
            scenes = f.read().splitlines()
    else:
        root_dir = os.listdir(args.data_root)
        scenes = [f for f in root_dir if os.path.isdir(os.path.join(args.data_root, f))]

    if args.output_dir is None:
        args.output_dir = args.data_root

    print("Processing", len(scenes), "scenes")

    for scene_id in scenes:
        # construct paths
        target_path = os.path.join(
            args.output_dir,
            scene_id,
            "features",
            f"{args.feature_type}_{args.source_cloud}{args.feature_suffix}",
        )
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        movie_path = os.path.join(args.data_root, scene_id, "iphone", "rgb.mp4")

        try:
            print("Processing scene", scene_id)
            process_scene(
                scene_id=scene_id,
                data_root=args.data_root,
                movie_path=movie_path,
                target_path=target_path,
                feature_type=args.feature_type,
                feature_suffix=args.feature_suffix,
                sampling_rate=args.sampling_rate,
                image_width=args.image_width,
                image_height=args.image_height,
                dino_model_name=args.dino_model_name,
                overwrite=args.overwrite,
                pointcloud_source=args.source_cloud,
                downscale=True,
                autoskip=True,
                batch_size=8,
            )
            print("Done with scene", scene_id)
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
