import argparse
import torch.distributed as dist
import os
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data
from data.dataloader import get_dataloader
from einops import repeat
from model.diffusion_elucidated import ElucidatedDiffusion
from model.diffusion_lucid import GaussianDiffusion as LUCID
from model.diffusion_pointvoxel import PVD
from omegaconf import DictConfig, OmegaConf
from utils.file_utils import set_seed
from utils.evaluation import evaluate
from loguru import logger
from utils.args import parse_args
import pandas as pd
import numpy as np
from model.loader import load_model

@torch.no_grad()
def sample(gpu, cfg, output_dir):
    set_seed(cfg)
    
    if cfg.distribution_type == "multi":
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])

        base_rank = cfg.rank * cfg.ngpus_per_node
        cfg.rank = base_rank + gpu
        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank,
        )

        cfg.sampling.bs = int(cfg.sampling.bs / cfg.ngpus_per_node)

    # get the loaders
    _, test_loader, _, _ = get_dataloader(cfg, sampling=True)

    model, _ = load_model(cfg, gpu, smart=False)
    
    ds_iter = iter(test_loader)
    model.eval()

    # first generate samples and save to disk
    # TODO
    
    # then evaluate the samples and accumulate the stats
    # TODO

    # finally save the stats
    # TODO


    # cleanup
    if cfg.distribution_type == "multi":
        dist.destroy_process_group()


def main():
    opt = parse_args()

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == "multi":
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(sample, nprocs=opt.ngpus_per_node, args=(opt, opt.out_sampling))
    else:
        opt.gpu = None
        sample(opt.gpu, opt, opt.out_sampling)


if __name__ == "__main__":
    main()
