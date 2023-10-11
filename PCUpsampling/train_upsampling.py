import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from data.dataloader import get_dataloader, save_iter
from einops import repeat
from model.diffusion_elucidated import ElucidatedDiffusion
from model.diffusion_lucid import GaussianDiffusion as LUCID
from model.diffusion_pointvoxel import PVD
from omegaconf import DictConfig, OmegaConf
from utils.file_utils import *
from utils.ops import *
from utils.visualize import *
from lion_pytorch import Lion
from loguru import logger
import sys
import wandb
import json

def train(gpu, cfg, output_dir, noises_init):
    set_seed(cfg)
    torch.cuda.empty_cache()

    if cfg.distribution_type == "multi":
        is_main_process = gpu == 0
    else:
        is_main_process = True
    if is_main_process:
        (outf_syn,) = setup_output_subdirs(output_dir, "syn")

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

    if cfg.diffusion.formulation == "PVD":
        model = PVD(
            cfg,
            loss_type=cfg.diffusion.loss_type,
            model_mean_type="eps",
            model_var_type="fixedsmall",
            device="cuda" if gpu == 0 else gpu,
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
        pretty_cfg = json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4, sort_keys=False)
        logger.info("Configuration used:\n{}", pretty_cfg)
        wandb.init(project="pvdup", config=cfg, entity="matvogel")

    if cfg.training.optimizer.type == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay, betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2)
        )
    elif cfg.training.optimizer.type == "Lion":
        optimizer = Lion(model.parameters(), lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay, betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2))
    elif cfg.training.optimizer.type == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay, betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2)
        )
    
    if cfg.training.scheduler.type == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, cfg.training.scheduler.lr_gamma)
    else:
        lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    if cfg.model_path != "":
        ckpt = torch.load(cfg.model)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if cfg.model_path != "":
        start_step = torch.load(cfg.model_path)["step"] + 1
    else:
        start_step = 0

    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)

    train_iter = save_iter(dataloader, train_sampler)

    for step in range(start_step, cfg.training.steps):
        
        # chek if we have a new epoch
        if cfg.distribution_type == "multi":
            train_sampler.set_epoch(step // len(dataloader))

        # update scheduler
        if (step+1) % len(dataloader) == 0:
            lr_scheduler.step()

        for accum_iter in range(cfg.training.accumulation_steps):
            # get next batch
            data = next(train_iter)

            x = data["train_points"].transpose(1, 2)
            noises_batch = noises_init[data["idx"]].transpose(1, 2)
            lowres = data["train_points_lowres"].transpose(1, 2)

            # move data to gpu
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

            loss = model(x, cond=lowres, noises=noises_batch) / cfg.training.accumulation_steps
            loss.backward()

        # get gradient norms for debugging and logging
        if not cfg.model.type == "Mink":
            netpNorm, netgradNorm = getGradNorm(model)
        else:
            netpNorm, netgradNorm = 0, 0

        if cfg.training.grad_clip.enabled:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip.value)

        optimizer.step()
        optimizer.zero_grad()

        if step % cfg.training.log_interval == 0 and is_main_process:
            logger.info(
                "[{:>3d}/{:>3d}]\tloss: {:>10.4f},\t"
                "netpNorm: {:>10.2f},\tnetgradNorm: {:>10.2f}\t",
                    step,
                    cfg.training.steps,
                    loss.item(),
                    netpNorm,
                    netgradNorm,
                
            )
            wandb.log(
                {
                    "loss": loss.item(),
                    "netpNorm": netpNorm,
                    "netgradNorm": netgradNorm,
                },
                step=step,
            )

        if (step+1) % cfg.training.viz_interval == 0 and is_main_process:
            logger.info("Generation: eval")

            model.eval()
            with torch.no_grad():
                x_gen_eval = model.sample(
                    shape=new_x_chain(x, cfg.sampling.bs).shape,
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

                logger.info(
                    "[{:>3d}/{:>3d}]\t"
                    "eval_gen_range: [{:>10.4f}, {:>10.4f}]\t"
                    "eval_gen_stats: [mean={:>10.4f}, std={:>10.4f}]\t",
                        step,
                        cfg.training.steps,
                        *gen_eval_range,
                        *gen_stats,

                )

            visualize_pointcloud_batch(
                "%s/step_%03d_samples_eval.png" % (outf_syn, step),
                x_gen_eval.transpose(1, 2),
                None,
                None,
                None,
            )

            visualize_pointcloud_batch(
                "%s/step_%03d_samples_eval_all.png" % (outf_syn, step),
                x_gen_all.transpose(1, 2),
                None,
                None,
                None,
            )

            visualize_pointcloud_batch(
                "%s/step_%03d_lowres.png" % (outf_syn, step),
                lowres.transpose(1, 2),
                None,
                None,
                None,
            )

            visualize_pointcloud_batch(
                "%s/step_%03d_highres.png" % (outf_syn, step),
                x.transpose(1, 2),
                None,
                None,
                None,
            )
            # log the saved images to wandb
            samps_eval = wandb.Image("%s/step_%03d_samples_eval.png" % (outf_syn, step))
            samps_eval_all = wandb.Image("%s/step_%03d_samples_eval_all.png" % (outf_syn, step))
            samps_lowres = wandb.Image("%s/step_%03d_lowres.png" % (outf_syn, step))
            samps_highres = wandb.Image("%s/step_%03d_highres.png" % (outf_syn, step))
            wandb.log(
                {
                    "samples_eval": samps_eval,
                    "samples_eval_all": samps_eval_all,
                    "samples_lowres": samps_lowres,
                    "samples_highres": samps_highres,
                },
                step=step,
            )

            logger.info("Generation: train")
            model.train()

        if (step + 1) % cfg.training.save_interval == 0:
            if is_main_process:
                save_dict = {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }

                torch.save(save_dict, "%s/step_%d.pth" % (output_dir, step))

            if cfg.distribution_type == "multi":
                dist.barrier()
                map_location = {"cuda:%d" % 0: "cuda:%d" % gpu}
                model.load_state_dict(
                    torch.load(
                        "%s/step_%d.pth" % (output_dir, step),
                        map_location=map_location,
                    )["model_state"]
                )

    if cfg.distribution_type == "multi":
        dist.destroy_process_group()
        wandb.finish()

def main():
    opt = parse_args()

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)

    # generating output dir
    output_dir = get_output_dir(dir_id, exp_id, opt)
    output_dir = os.path.join(opt.save_dir, output_dir)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # save the opt to output_dir
    OmegaConf.save(opt, os.path.join(output_dir, "opt.yaml"))
    
    #logger.remove() # remove stderr handler
    #logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
    #logger.add("file_{time}.log")

    """ workaround TODO inspect this for future versions"""
    loader, _, _, _ = get_dataloader(opt)
    noises_init = torch.randn(len(loader.dataset), opt.data.npoints, opt.data.nc)

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == "multi":
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir, noises_init))
    else:
        opt.gpu = None
        train(opt.gpu, opt, output_dir, noises_init)


def parse_args():
    # make parser which accepts optinal arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config file.")
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
    # load config
    cfg = OmegaConf.load(args.config)

    # merge config with command line arguments
    opt = OmegaConf.merge(cfg, OmegaConf.create(vars(args)))

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

    # set name
    if opt.name == "":
        opt.name = os.path.splitext(os.path.basename(opt.config))[0]
    
    return opt


if __name__ == "__main__":
    main()
