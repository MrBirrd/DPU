from utils.image_features import process_scene
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--feature_type", type=str, default="rgb", help="Type of features to extract.", choices=["rgb", "dino"]
    )
    parser.add_argument("--skip_scans", type=int, default=5, help="Number of scans to skip between each scan.")
    parser.add_argument("--image_width", type=int, default=1920, help="Width of the images in the video.")
    parser.add_argument("--image_height", type=int, default=1440, help="Height of the images in the video.")
    parser.add_argument("--dino_model_name", type=str, default="dinov2_vits14", help="Name of the DINO model to use.")
    args = parser.parse_args()

    root_dir = os.listdir(args.data_root)
    scenes = [f for f in root_dir if os.path.isdir(os.path.join(args.data_root, f))]

    if args.output_dir is None:
        args.output_dir = args.data_root

    for scene_id in scenes:
        # construct paths
        target_path = os.path.join(args.output_dir, scene_id, "features", f"{args.feature_type}")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        movie_path = os.path.join(args.data_root, scene_id, "iphone", "rgb.mp4")

        try:
            process_scene(
                scene_id=scene_id,
                data_root=args.data_root,
                movie_path=movie_path,
                target_path=target_path,
                feature_type=args.feature_type,
                skip_scans=args.skip_scans,
                image_width=args.image_width,
                image_height=args.image_height,
                dino_model_name=args.dino_model_name,
            )
            print("Done with scene", scene_id)
            break
        except Exception as e:
            print(e)
            pass


if __name__ == "__main__":
    main()