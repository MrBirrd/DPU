import json
import os

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
from loguru import logger
from omegaconf import OmegaConf

import wandb
from data.dataloader import get_dataloader, save_iter
from training.evaluation import evaluate
from training.model_loader import load_diffusion, load_optim_sched
from training.train_utils import get_data_batch, set_seed, to_cuda
from utils.args import parse_args
from utils.ops import *
from utils.utils import setup_output_subdirs


def train(gpu, cfg, output_dir, noises_init=None):
    # set gpu
    if cfg.distribution_type == "multi":
        cfg.gpu = gpu

    logger.info("CUDA available: {}", torch.cuda.is_available())

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

        global_batch_size = cfg.training.bs
        cfg.training.bs = int(cfg.training.bs / cfg.ngpus_per_node)
        cfg.sampling.bs = cfg.training.bs
        logger.info(
            "Distributed training with {} GPUs. Rank: {}, World size: {}, Global Batch size: {}, Minibatch size: {}",
            cfg.ngpus_per_node,
            cfg.rank,
            cfg.world_size,
            global_batch_size,
            cfg.training.bs,
        )

    # set seed
    set_seed(cfg)
    torch.cuda.empty_cache()

    # setup data loader and sampler
    train_loader, val_loader, train_sampler, val_sampler = get_dataloader(cfg)

    # initialize config and wandb
    if is_main_process:
        pretty_cfg = json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4, sort_keys=False)
        logger.info("Configuration used:\n{}", pretty_cfg)
        wandb.init(
            project="pvdup",
            config=OmegaConf.to_container(cfg, resolve=True),
            entity="matvogel",
        )

    model, ckpt = load_diffusion(cfg)
    optimizer, lr_scheduler = load_optim_sched(cfg, model, ckpt)

    # setup amp scaler
    ampscaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)

    # train loop
    train_iter = save_iter(train_loader, train_sampler)
    eval_iter = save_iter(val_loader, val_sampler)

    torch.cuda.empty_cache()

    # sample first batch
    next_batch = next(train_iter)
    next_batch = to_cuda(next_batch, gpu)

    for step in range(cfg.start_step, cfg.training.steps):
        # chek if we have a new epoch
        if cfg.distribution_type == "multi":
            train_sampler.set_epoch(step // len(train_loader))

        # update scheduler
        if (step + 1) % len(train_loader) == 0:
            lr_scheduler.step()

        loss_accum = 0.0
        for accum_iter in range(cfg.training.accumulation_steps):
            data = next_batch
            # prepare next batch
            next_batch = next(train_iter)
            next_batch = to_cuda(next_batch, gpu)

            data_batch = get_data_batch(batch=data, cfg=cfg)
            x0 = data_batch["hr_points"]
            x1 = data_batch["lr_points"] if data_batch["lr_points"] is not None else None
            features = data_batch["features"] if data_batch["features"] is not None else None

            # forward pass
            loss = model(x0, x1=x1, cond=features)

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

        if step % cfg.training.log_interval == 0 and is_main_process:
            logger.info(
                "[{:>3d}/{:>3d}]\tloss: {:>10.6f},\t" "netpNorm: {:>10.2f},\tnetgradNorm: {:>10.4f}\t",
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
            model.eval()
            evaluate(model, eval_iter, cfg, step + 1)
            model.train()

        if (step + 1) % cfg.training.save_interval == 0:
            if is_main_process:
                save_dict = {
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
                os.makedirs(output_dir, exist_ok=True)
                torch.save(save_dict, "%s/step_%d.pth" % (output_dir, step + 1))

            if cfg.distribution_type == "multi":
                dist.barrier()
                map_location = {"cuda:%d" % 0: "cuda:%d" % gpu}
                model.load_state_dict(
                    torch.load(
                        "%s/step_%d.pth" % (output_dir, step + 1),
                        map_location=map_location,
                    )["model_state"]
                )

    if cfg.distribution_type == "multi":
        dist.destroy_process_group()
        wandb.finish()


def main():
    opt = parse_args()

    # generating output dir
    output_dir = os.path.join(opt.save_dir, opt.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
