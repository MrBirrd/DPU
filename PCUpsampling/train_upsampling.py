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
from model.diffusion_rin import GaussianDiffusion as RINDIFFUSION
from omegaconf import DictConfig, OmegaConf
from utils.evaluation import *
from utils.file_utils import *
from utils.ops import *
from utils.visualize import *
from lion_pytorch import Lion
from loguru import logger
import sys
import wandb
import json
from point_cloud_utils import chamfer_distance


def train(gpu, cfg, output_dir, noises_init=None):
    # set gpu
    if cfg.distribution_type == "multi":
        cfg.gpu = gpu

    logger.info("CUDA available: {}", torch.cuda.is_available())
    
    # set seed
    set_seed(cfg)
    torch.cuda.empty_cache()

    # evaluate main process and set output dir
    if cfg.distribution_type == "multi":
        is_main_process = gpu == 0
    else:
        is_main_process = True
    if is_main_process:
        (outf_syn,) = setup_output_subdirs(output_dir, "output")

    # set multi gpu training variables
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

        cfg.training.bs = int(cfg.training.bs / cfg.ngpus_per_node)
        cfg.sampling.bs = cfg.training.bs
        cfg.data.workers = 0

        cfg.training.save_interval = int(cfg.training.save_interval / cfg.ngpus_per_node)
        cfg.training.viz_interval = int(cfg.training.viz_interval / cfg.ngpus_per_node)

    # setup data loader and sampler
    train_loader, val_loader, train_sampler, val_sampler = get_dataloader(cfg)

    # setup model
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
    elif cfg.diffusion.formulation == "RIN":
        model = RINDIFFUSION(cfg=cfg)

    # setup DDP model
    if cfg.distribution_type == "multi":

        def _transform_(m):
            return nn.parallel.DistributedDataParallel(m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)

    # setup data parallel model
    elif cfg.distribution_type == "single":

        def _transform_(m):
            return nn.parallel.DataParallel(m)

        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    # setup single gpu model
    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError("distribution_type = multi | single | None")

    # initialize config and wandb
    if is_main_process:
        pretty_cfg = json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4, sort_keys=False)
        logger.info("Configuration used:\n{}", pretty_cfg)
        wandb.init(
            project="pvdup",
            config=OmegaConf.to_container(cfg, resolve=True),
            entity="matvogel",
            #settings=wandb.Settings(start_method="fork"),
        )

    # setup optimizers
    if cfg.training.optimizer.type == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
            betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2),
        )
    elif cfg.training.optimizer.type == "Lion":
        optimizer = Lion(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
            betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2),
        )
    elif cfg.training.optimizer.type == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
            betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2),
        )

    # setup lr scheduler
    if cfg.training.scheduler.type == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, cfg.training.scheduler.lr_gamma)
    else:
        lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    # setup amp scaler
    ampscaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)

    # load model
    if cfg.model_path != "":
        ckpt = torch.load(cfg.model_path)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

    # set start step
    if cfg.model_path != "":
        start_step = torch.load(cfg.model_path)["step"] + 1
    else:
        start_step = 0

    # helper for chain of samples
    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)

    # train loop
    train_iter = save_iter(train_loader, train_sampler)
    eval_iter = save_iter(val_loader, val_sampler)

    for step in range(start_step, cfg.training.steps):
        # chek if we have a new epoch
        if cfg.distribution_type == "multi":
            train_sampler.set_epoch(step // len(train_loader))

        # update scheduler
        if (step + 1) % len(train_loader) == 0:
            lr_scheduler.step()

        loss_accum = 0.0
        for accum_iter in range(cfg.training.accumulation_steps):
            # get next batch
            data = next(train_iter)

            x = data["train_points"].transpose(1, 2)
            lowres = (
                data["train_points_lowres"].transpose(1, 2)
                if "train_points_lowres" in data and not cfg.data.unconditional
                else None
            )

            # move data to gpu
            if cfg.distribution_type == "multi":
                x = x.cuda(gpu)
                lowres = lowres.cuda(gpu) if lowres is not None else None
            elif cfg.distribution_type == "single":
                x = x.cuda()
                lowres = lowres.cuda() if lowres is not None else None

            # forward pass
            loss = model(x, cond=lowres) / cfg.training.accumulation_steps
            loss_accum += loss.item()
            ampscaler.scale(loss).backward()

        # get gradient norms for debugging and logging
        netpNorm, netgradNorm = getGradNorm(model)

        if cfg.training.grad_clip.enabled:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip.value)

        ampscaler.step(optimizer)
        ampscaler.update()
        optimizer.zero_grad()

        if step % cfg.training.log_interval == 0 and is_main_process:
            logger.info(
                "[{:>3d}/{:>3d}]\tloss: {:>10.4f},\t" "netpNorm: {:>10.2f},\tnetgradNorm: {:>10.2f}\t",
                step,
                cfg.training.steps,
                loss_accum,
                netpNorm,
                netgradNorm,
            )
            wandb.log(
                {
                    "loss": loss_accum,
                    "netpNorm": netpNorm,
                    "netgradNorm": netgradNorm,
                },
                step=step,
            )

        if (step + 1) % cfg.training.viz_interval == 0 and is_main_process:
            logger.info("Starting evaluation...")

            model.eval()

            eval_data = next(eval_iter)

            x_eval = eval_data["train_points"].transpose(1, 2)
            lowres_eval = (
                eval_data["train_points_lowres"].transpose(1, 2)
                if "train_points_lowres" in eval_data and not cfg.data.unconditional
                else None
            )

            # move data to gpu
            if cfg.distribution_type == "multi":
                x_eval = x_eval.cuda(gpu)
                lowres_eval = lowres_eval.cuda(gpu) if lowres_eval is not None else None
            elif cfg.distribution_type == "single":
                x_eval = x_eval.cuda()
                lowres_eval = lowres_eval.cuda() if lowres_eval is not None else None

            with torch.no_grad():
                if cfg.sampling.bs == 1:
                    cond = lowres_eval[0].unsqueeze(0) if lowres_eval is not None else None
                else:
                    cond = lowres_eval[: cfg.sampling.bs] if lowres_eval is not None else None

                x_gen_eval = model.sample(
                    shape=new_x_chain(x_eval, cfg.sampling.bs).shape,
                    device=x_eval.device,
                    cond=cond,
                    hint=x_eval if cfg.diffusion.sampling_hint else None,
                    clip_denoised=False,
                )

                x_gen_list = model.sample(
                    shape=new_x_chain(x_eval, 1).shape,
                    device=x_eval.device,
                    cond=lowres_eval[0].unsqueeze(0) if lowres_eval is not None else None,
                    hint=x_eval if cfg.diffusion.sampling_hint else None,
                    freq=0.1,
                    clip_denoised=False,
                )

                x_gen_all = torch.cat(x_gen_list, dim=0)

            # calculate metrics such as min, max, mean, std, etc.
            print_stats(x_gen_eval, "x_gen_eval")
            print_stats(x_gen_all, "x_gen_all")

            # calculate the CD
            cds = []

            if cfg.model.type == "Mink":
                xgnp = x_gen_eval.cpu().numpy()
                xgnp = np.asfortranarray(xgnp)
                x_gen_eval = torch.from_numpy(xgnp).cuda()

            for x_pred, x_gt in zip(x_gen_eval, x_eval):
                cd = chamfer_distance(
                    x_pred.cpu().permute(1, 0).numpy(),
                    x_gt.cpu().permute(1, 0).numpy(),
                )
                cds.append(cd)
            
            cd = np.mean(cds)

            logger.info("CD: {}", cd)
            wandb.log({"CD": cd}, step=step)

            # visualize the pointclouds
            visualize_pointcloud_batch(
                "%s/step_%03d_samples_eval.png" % (outf_syn, step),
                x_gen_eval.transpose(1, 2),
            )

            visualize_pointcloud_batch(
                "%s/step_%03d_samples_eval_all.png" % (outf_syn, step),
                x_gen_all.transpose(1, 2),
            )

            if lowres_eval is not None:
                visualize_pointcloud_batch(
                    "%s/step_%03d_lowres.png" % (outf_syn, step),
                    lowres_eval.transpose(1, 2),
                )

            visualize_pointcloud_batch(
                "%s/step_%03d_highres.png" % (outf_syn, step),
                x_eval.transpose(1, 2),
            )

            # log the saved images to wandb
            samps_eval = wandb.Image("%s/step_%03d_samples_eval.png" % (outf_syn, step))
            samps_eval_all = wandb.Image("%s/step_%03d_samples_eval_all.png" % (outf_syn, step))
            samps_lowres = (
                wandb.Image("%s/step_%03d_lowres.png" % (outf_syn, step)) if lowres_eval is not None else None
            )
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

    # generating output dir
    output_dir = get_output_dir(opt.save_dir, opt)

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # save the opt to output_dir
    OmegaConf.save(opt, os.path.join(output_dir, "opt.yaml"))

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == "multi":
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, output_dir))
    else:
        opt.gpu = None
        train(opt.gpu, opt, output_dir)


def parse_args():
    # make parser which accepts optinal arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config file.")
    parser.add_argument("--name", type=str, default="", help="Name of the experiment.")
    parser.add_argument("--save_dir", default="checkpoints", help="path to save models")
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
            key = remaining_argv[i].lstrip("--")
            value = remaining_argv[i + 1]

            # Convert numerical strings to appropriate number types handling scientific notation
            try:
                if "." in remaining_argv[i + 1] or "e" in remaining_argv[i + 1]:
                    value = float(value)
                # handle bools
                elif value in ["True", "False", "true", "false"]:
                    value = value.capitalize() == "True"
                else:
                    value = int(value)
            except ValueError:
                pass

            # Update the config using OmegaConf's select and set methods
            OmegaConf.update(opt, key, value, merge=False)

    # set name
    if opt.name == "":
        opt.name = os.path.splitext(os.path.basename(opt.config))[0]

    # fix values for DDPM sampling steps
    if opt.diffusion.sampling_strategy == "DDPM":
        opt.diffusion.sampling_timesteps = opt.diffusion.timesteps

    return opt


if __name__ == "__main__":
    main()
