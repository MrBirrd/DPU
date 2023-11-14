import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from data.dataloader import get_dataloader, save_iter
from model.diffusion_elucidated import ElucidatedDiffusion
from model.diffusion_lucid import GaussianDiffusion as LUCID
from model.diffusion_pointvoxel import PVD
from model.diffusion_rin import GaussianDiffusion as RINDIFFUSION
from omegaconf import OmegaConf
from utils.evaluation import evaluate
from utils.file_utils import *
from utils.ops import *
from utils.args import parse_args
from lion_pytorch import Lion
from loguru import logger
import wandb
import json


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
        cfg.outf_syn = outf_syn

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
            # settings=wandb.Settings(start_method="fork"),
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
            loss = model(x, cond=lowres)

            loss /= cfg.training.accumulation_steps

            loss_accum += loss.item()
            ampscaler.scale(loss).backward()

        # get gradient norms for debugging and logging
        ampscaler.unscale_(optimizer)
        netpNorm, netgradNorm = getGradNorm(model)

        if cfg.training.grad_clip.enabled:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip.value)

        ampscaler.step(optimizer)
        ampscaler.update()
        optimizer.zero_grad()

        # do ema update
        if model.model_ema is not None:
            model.model_ema.update()

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
            try:
                evaluate(model, eval_iter, cfg, step)
            except Exception as e:
                logger.error("Error during evaluation: {}", e)

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

        if (step + 1) % cfg.training.viz_interval == 0 and is_main_process:
            evaluate(model, eval_iter, cfg, step)

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


if __name__ == "__main__":
    main()
