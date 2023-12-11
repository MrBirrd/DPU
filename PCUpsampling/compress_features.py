import argparse
import gc
import os
from glob import glob

import cudf
import numpy as np
import torch
from cuml import TruncatedSVD
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--ftype",
        type=str,
        required=True,
        help="Type of the features.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing features.",
    )
    parser.add_argument(
        "--fdim",
        type=int,
        default=64,
        help="Dimension of the features.",
    )
    
    # find all features in root
    args = parser.parse_args()
    root = args.root
    ftype = args.ftype

    feature_files = glob(os.path.join(root, "*", "features", f"{ftype}.npy"))
    print("Found", len(feature_files), "feature files.")
    
    for feature_path in tqdm(feature_files):
        feats = np.load(feature_path).T
        data_cudf = cudf.DataFrame(feats)
        svd = TruncatedSVD(n_components=args.fdim)
        transformed_data = svd.fit_transform(data_cudf)
        
        # convert to numpy array
        transformed_data = transformed_data.to_numpy().T
        target_path = feature_path.replace(".npy", f"_svd{args.fdim}.npy")
        if os.path.exists(target_path):
            pass
        
        np.save(target_path, transformed_data.astype(np.float16))
        del feats, data_cudf, transformed_data
        torch.cuda.empty_cache()
        gc.collect()