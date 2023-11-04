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

# from metrics.evaluation_metrics import compute_all_metrics


def sample(gpu, cfg, output_dir):
    set_seed(cfg)
    torch.cuda.empty_cache()

    # apply timestep clipping
    timestep = min(cfg.diffusion.timesteps_clip, cfg.diffusion.sampling_timesteps)

    if cfg.distribution_type == "multi":
        is_main_process = gpu == 0
    else:
        is_main_process = True
    if is_main_process:
        scheduler_info = f"{cfg.diffusion.sampling_strategy}(T={str(timestep)})"

        # add clipping information to scheduler info
        if cfg.diffusion.clip:
            if cfg.diffusion.dynamic_threshold:
                clip = "_clip_dynamic"
            else:
                clip = "_clip"
        else:
            clip = ""
        scheduler_info += clip

        cfg.out_sampling = os.path.join(output_dir, "sampling", scheduler_info)
        # os.makedirs(output_dir, out_sampling, exist_ok=True)

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

    for sampling_iter in range(cfg.sampling.num_iter):
        evaluate(model, ds_iter, cfg, sampling_iter, sampling=True)

    if cfg.distribution_type == "multi":
        dist.destroy_process_group()


def main():
    opt = parse_args()

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == "multi":
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(sample, nprocs=opt.ngpus_per_node, args=(opt, opt.output_dir))
    else:
        opt.gpu = None
        sample(opt.gpu, opt, opt.output_dir)


def parse_args():
    # make parser which accepts optinal arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, help="Path to the config file.")
    parser.add_argument("--name", type=str, default="", help="Name of the experiment.")
    parser.add_argument("--save_dir", default=".")
    parser.add_argument("--model_path", default="", help="path to model (to continue training)")

    """distributed"""
    parser.add_argument("--world_size", default=1, type=int, help="Number of distributed nodes.")
    parser.add_argument(
        "--dist_url",
        default="tcp://127.0.0.1:9991",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--distribution_type",
        default="single",
        choices=["multi", "single", None],
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")

    args, remaining_argv = parser.parse_known_args()
    # load config from eperiment folder
    experiment_path = os.path.dirname(args.model_path)
    opt_path = os.path.join(experiment_path, "opt.yaml")
    cfg = OmegaConf.load(opt_path)

    # merge config with command line arguments
    opt = OmegaConf.merge(cfg, OmegaConf.create(vars(args)))
    opt.output_dir = experiment_path

    if remaining_argv:
        for i in range(0, len(remaining_argv), 2):
            key = remaining_argv[i].lstrip("--")
            value = remaining_argv[i + 1]

            # Convert numerical strings to appropriate number types handling scientific notation
            try:
                if "." in remaining_argv[i + 1] or "e" in remaining_argv[i + 1]:
                    value = float(value)
                # handle bools
                elif value in ["True", "False", "true", "false"]:
                    value = value.lower() == "true"
                else:
                    value = int(value)
            except ValueError:
                pass

            # Update the config using OmegaConf's select and set methods
            OmegaConf.update(opt, key, value, merge=False)

    return opt


if __name__ == "__main__":
    main()
