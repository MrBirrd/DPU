import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data
from data.dataloader import get_dataloader
from einops import repeat
from model.diffusion_elucidated import ElucidatedDiffusion
from model.diffusion_lucid import GaussianDiffusion as LUCID
from model.diffusion_pointvoxel import PVD
from omegaconf import DictConfig, OmegaConf
from utils.file_utils import *
from utils.ops import *
from utils.visualize import *

import wandb


def sample(gpu, cfg, output_dir, noises_init):
    set_seed(cfg)
    torch.cuda.empty_cache()

    logger = setup_logging(output_dir)
    if cfg.distribution_type == "multi":
        is_main_process = gpu == 0
    else:
        is_main_process = True
    if is_main_process:
        scheduler_info = "_".join([cfg.diffusion.sampling_strategy, str(cfg.diffusion.sampling_timesteps), f"dts_{cfg.diffusion.dynamic_threshold}"])
        out_sampling = os.path.join(output_dir, "sampling", scheduler_info)
        #os.makedirs(output_dir, out_sampling, exist_ok=True)

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
    dataloader, _, train_sampler, _ = get_dataloader(cfg)

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
        wandb.init(project="pvdup", config=cfg, entity="matvogel")

    if cfg.model_path != "":
        ckpt = torch.load(cfg.model_path)
        model.load_state_dict(ckpt["model_state"])
    else:
        raise ValueError("model_path must be specified")

    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)

    ds_iter = iter(dataloader)
    model.eval()

    for sampling_iter in range(cfg.sampling.num_iter):
        data = next(ds_iter)
        x = data["train_points"].transpose(1, 2)
        noises_batch = noises_init[data["idx"]].transpose(1, 2)
        lowres = data["train_points_lowres"].transpose(1, 2)


        if cfg.distribution_type == "multi" or (
            cfg.distribution_type is None and gpu is not None
        ):
            x = x.cuda(gpu)
            noises_batch = noises_batch.cuda(gpu)
            lowres = lowres.cuda(gpu)
        elif cfg.distribution_type == "single":
            x = x.cuda()
            noises_batch = noises_batch.cuda()
            lowres = lowres.cuda()

        with torch.no_grad():
            x_gen_eval = model.sample(
                shape=new_x_chain(x, lowres.shape[0] if not cfg.training.overfit else 1).shape,
                device=x.device,
                cond=lowres if not cfg.training.overfit else lowres[0].unsqueeze(0),
                clip_denoised=False,
            )
            x_gen_list = model.sample(
                shape=new_x_chain(x, 1).shape,
                device=x.device,
                cond=lowres[0].unsqueeze(0),
                freq=0.1,
                clip_denoised=False,
            )
            x_gen_all = torch.cat(x_gen_list, dim=0)

            gen_stats = [x_gen_eval.mean(), x_gen_eval.std()]
            gen_eval_range = [x_gen_eval.min().item(), x_gen_eval.max().item()]

            # normalize if overfit
            if cfg.training.overfit:
                mean, std = dataloader.dataset.mean, dataloader.dataset.std
                B, C, N = x_gen_eval.shape
                std = repeat(std, "c -> b c n", b=B, n=N)
                mean = repeat(mean, "c -> b c n", b=B, n=N)
                x_gen_eval = x_gen_eval.cpu() * std + mean
                x_gen_all = x_gen_all.cpu() * std + mean
                x = x.cpu() * std + mean
                lowres = lowres.cpu() * std + mean

            visualize_pointcloud_batch(
                "%s/epoch_%03d_samples_eval.png" % (out_sampling, sampling_iter),
                x_gen_eval.transpose(1, 2),
                None,
                None,
                None,
            )

            visualize_pointcloud_batch(
                "%s/epoch_%03d_samples_eval_all.png" % (out_sampling, sampling_iter),
                x_gen_all.transpose(1, 2),
                None,
                None,
                None,
            )

            visualize_pointcloud_batch(
                "%s/epoch_%03d_lowres.png" % (out_sampling, sampling_iter),
                lowres.transpose(1, 2),
                None,
                None,
                None,
            )

            visualize_pointcloud_batch(
                "%s/epoch_%03d_highres.png" % (out_sampling, sampling_iter),
                x.transpose(1, 2),
                None,
                None,
                None,
            )

    if cfg.distribution_type == "multi":
        dist.destroy_process_group()

def main():
    opt = parse_args()

    """ workaround TODO inspect this for future versions"""
    loader, _, _, _ = get_dataloader(opt)
    noises_init = torch.randn(len(loader.dataset), opt.data.npoints, opt.data.nc)

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == "multi":
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(sample, nprocs=opt.ngpus_per_node, args=(opt, opt.output_dir, noises_init))
    else:
        opt.gpu = None
        sample(opt.gpu, opt, opt.output_dir, noises_init)


def parse_args():
    # make parser which accepts optinal arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument("--config", type=str, help="Path to the config file.")
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
            key = remaining_argv[i].lstrip('--')
            value = remaining_argv[i + 1]
            
            # Convert numerical strings to appropriate number types handling scientific notation
            try:
                if '.' in remaining_argv[i + 1] or 'e' in remaining_argv[i + 1]:
                    value = float(value)
                # handle bools
                elif value in ['True', 'False', 'true', 'false']:
                    value = value.capitalize() == 'True'
                else:
                    value = int(value)
            except ValueError:
                pass

            # Update the config using OmegaConf's select and set methods
            OmegaConf.update(opt, key, value, merge=False)
    
    return opt


if __name__ == "__main__":
    main()
