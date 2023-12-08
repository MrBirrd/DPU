from .diffusion_pointvoxel import PVD
from .diffusion_lucid import GaussianDiffusion as LUCID
import torch
from loguru import logger
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch import optim
from utils.utils import smart_load_model_weights


def load_optim_sched(cfg, model, model_ckpt=None):
    # setup optimizers
    if cfg.training.optimizer.type == "Adam":
        optimizer = optim.Adam(
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

    if model_ckpt is not None:
        optimizer.load_state_dict(model_ckpt["optimizer_state"])

    return optimizer, lr_scheduler


def load_model(cfg, gpu=None, smart=False):
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
            return DistributedDataParallel(m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)

    # setup data parallel model
    elif cfg.distribution_type == "single":

        def _transform_(m):
            return DataParallel(m)

        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    # setup single gpu model
    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError("distribution_type = multi | single | None")

    # load the model weights
    cfg.start_step = 0
    if cfg.model_path != "":
        ckpt = torch.load(cfg.model_path, map_location=torch.device("cpu"))
        if not cfg.restart:
            cfg.start_step = ckpt["step"] + 1
            model.load_state_dict(ckpt["model_state"])
        else:
            # only load the model parameters and let rest start from scratch
            model_keys = {k.replace("model.", ""): v for k, v in ckpt["model_state"].items() if k.startswith("model.")}
            model.model.load_state_dict(model_keys)

        logger.info("Loaded model from %s" % cfg.model_path)
    else:
        ckpt = None

    torch.cuda.empty_cache()

    return model, ckpt
