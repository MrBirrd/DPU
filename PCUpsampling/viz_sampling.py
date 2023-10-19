from pyviz3d import visualizer
import numpy as np
import os
import argparse
import subprocess
from einops import rearrange

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--point_size", type=int, default=5)
    args = parser.parse_args()

    # find all npys
    npys = []
    for root, dirs, files in os.walk(args.dir):
        for file in files:
            if file.endswith(".npy"):
                npys.append(os.path.join(root, file))

    vis = visualizer.Visualizer()

    for cloud in npys:
        pc = np.load(cloud)
        name = os.path.basename(cloud).split(".")[0]
        if "highres" in name:
            name = "gt"
        elif "eval_all" in name:
            continue
        elif "eval" in name:
            name = "pred"

        for idx, cloud in enumerate(pc):
            cloud = rearrange(cloud, "n d -> d n")
            vis.add_points(f"{name}_{idx}", cloud, point_size=args.point_size, visible=False)

    vis.save(".sampling_viz")

    # run http.server 6008 in viz directory
    subprocess.run(["python", "-m", "http.server", "6008"], cwd=".sampling_viz")
