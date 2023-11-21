import os
import pandas as pd
import numpy as np
from metrics.emd_ import emd_module as EMD
from metrics.chamferdist import ChamferDistance
import torch
import matplotlib.pyplot as plt
from point_cloud_utils import chamfer_distance
from glob import glob
from tqdm import tqdm

stats_columns = ["model", "ckpt", "t", "cd", "emd"]


def calculate_chamfer_distance(xyz1, xyz2):
    assert xyz1.shape == xyz2.shape, "Shape mismatch"
    if xyz1.ndim == 2:
        xyz1 = xyz1.unsqueeze(0)
        xyz2 = xyz2.unsqueeze(0)

    cds = 0.0

    for batch_idx in range(xyz1.shape[0]):
        cds += chamfer_distance(xyz1[batch_idx].cpu().numpy(), xyz2[batch_idx].cpu().numpy())

    cds /= xyz1.shape[0]
    return cds


if __name__ == "__main__":
    # Load the data
    models = [
        # "scannet_cut_pvd",
        # "scannet_cut_mink",
        # "scannet_cut_pvd_att",
        # "scannet_cut_st",
        #"scannet_cut_pvd_X"
        "scannet_cut_small_mink",
        "scannet_cut_small_st",
        "scannet_cut_small_pvd_large_mse",
        "scannet_cut_small_pvd_st"
    ]

    models = ["checkpoints/" + model for model in models]
    emd = EMD.emdModule()

    for model in models:
        path = os.path.join(model, "sampling")
        ckpts = os.listdir(path)
        for ckpt in ckpts:
            ckpt_path = os.path.join(path, ckpt)

            stats_path = os.path.join(ckpt_path, "stats.csv")
            try:
                stats = pd.read_csv(stats_path)
            except:
                stats = pd.DataFrame(columns=stats_columns)

            configs = os.listdir(ckpt_path)
            # go through all configs
            for config in tqdm(configs):
                t = config.split("T=")[-1].split(")")[0]
                new_row = {"model": model.split("/")[-1], "ckpt": ckpt, "t": t}

                # check if stats already exist
                if (
                    len(
                        stats[
                            (stats["model"] == new_row["model"])
                            & (stats["ckpt"] == new_row["ckpt"])
                            & (stats["t"] == new_row["t"])
                        ]
                    )
                    > 0
                ):
                    print("Stats already exist for ", new_row)
                    continue

                predictions = glob(os.path.join(ckpt_path, config, "*pred.npy"))
                n_predictions = len(predictions)

                if n_predictions == 0:
                    print("No predictions found for ", new_row)
                    continue

                cd_total = 0.0
                emd_total = 0.0

                print("Calculating stats for ", new_row)

                for pred in tqdm(predictions):
                    gt = pred.replace("pred", "gt_highres")
                    if os.path.exists(gt):
                        gt = torch.from_numpy(np.load(gt)).cuda().float()
                        pred = torch.from_numpy(np.load(pred)).cuda().float()

                        # subsample cloud to closest multiple of 128
                        n_points = pred.shape[-1]
                        n_points = n_points - n_points % 128
                        pred = pred[..., :n_points]
                        gt = gt[..., :n_points]

                        # reshape to BND
                        gt = gt.permute(0, 2, 1)
                        pred = pred.permute(0, 2, 1)

                        # calculate chamfer distance
                        try:
                            chamfer_dist = calculate_chamfer_distance(gt, pred)
                        except:
                            # switch row major to col major
                            xgnp = pred.cpu().numpy()
                            xgnp = np.asfortranarray(xgnp)
                            pred = torch.from_numpy(xgnp).cuda()
                            chamfer_dist = calculate_chamfer_distance(gt, pred)

                        cd_total += chamfer_dist.item() * 1000

                        # calculate earth mover's distance
                        dis, assigment = emd(gt, pred, 0.05, 100)
                        dis = torch.sqrt(dis).mean()
                        emd_total += dis.item()

                new_row["cd"] = cd_total / n_predictions
                new_row["emd"] = emd_total / n_predictions

                stats = pd.concat([stats, pd.DataFrame(new_row, index=[0])])

            stats.to_csv(stats_path, index=False)
