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


def sample(gpu, cfg, output_dir):
    set_seed(cfg)
    torch.cuda.empty_cache()

    # apply timestep clipping
    timestep = min(cfg.diffusion.timesteps_clip, cfg.diffusion.sampling_timesteps)

    if cfg.distribution_type == "multi":
        is_main_process = gpu == 0
    else:
        is_main_process = True
    
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

        cfg.bs = int(cfg.bs / cfg.ngpus_per_node)
        cfg.workers = 0

        cfg.saveIter = int(cfg.saveIter / cfg.ngpus_per_node)
        cfg.diagIter = int(cfg.diagIter / cfg.ngpus_per_node)
        cfg.vizIter = int(cfg.vizIter / cfg.ngpus_per_node)

    """ data """
    train_loader, test_loader, _, _ = get_dataloader(cfg, sampling=True)

    """
    create networks
    """

    if cfg.diffusion.formulation == "PVD":
        model = PVD(
            cfg,
            loss_type=cfg.diffusion.loss_type,
            model_mean_type="eps",
            model_var_type="fixedsmall",
        )
    elif cfg.diffusion.formulation == "EDM":
        model = ElucidatedDiffusion(args=cfg)
    elif cfg.diffusion.formulation == "LUCID":
        model = LUCID(cfg=cfg)

    if cfg.distribution_type == "multi":  # Multiple processes, single GPU per process

        def _transform_(m):
            return nn.parallel.DistributedDataParallel(m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)

    elif cfg.distribution_type == "single":

        def _transform_(m):
            return nn.parallel.DataParallel(m)

        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError("distribution_type = multi | single | None")

    if is_main_process:
        logger.info(cfg)

    if cfg.model_path != "":
        ckpt = torch.load(cfg.model_path)
        model.load_state_dict(ckpt["model_state"])
    else:
        raise ValueError("model_path must be specified")

    ds_iter = iter(test_loader)
    model.eval()

    cds = []
    for sampling_iter in range(cfg.sampling.num_iter):
        stats = evaluate(model, ds_iter, cfg, sampling_iter, sampling=True)
        cd = stats["CD"]
        cds.append(cd)

    cd = np.mean(cds)
    stats = pd.DataFrame(columns=["CD"], data=[cd])
    stats.to_csv(os.path.join(cfg.out_sampling, "stats.csv"))
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
