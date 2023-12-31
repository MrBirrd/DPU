import torch
from loguru import logger
from torch import optim
from torch.nn.parallel import DataParallel, DistributedDataParallel

from training.i2sb import I2SB

try:
    from training.unet_mink import MinkUnet
except ImportError:
    logger.warning("Minkowski Engine not installed. Minkowski models will not be available.")

from third_party.gecco_torch.models.linear_lift import LinearLift
from third_party.gecco_torch.models.set_transformer import SetTransformer
from training.diffusion_lucid import GaussianDiffusion as LUCID
from training.unet_pointvoxel import PVCNN2Unet


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
    else:
        raise NotImplementedError(cfg.training.optimizer.type)

    # setup lr scheduler
    if cfg.training.scheduler.type == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, cfg.training.scheduler.lr_gamma)
    else:
        lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    if model_ckpt is not None:
        optimizer.load_state_dict(model_ckpt["optimizer_state"])

    return optimizer, lr_scheduler


def load_model(cfg):
    if cfg.model.type == "PVD":
        if cfg.model.PVD.size == "small":
            raise NotImplementedError(cfg.model.PVD.size)
        elif cfg.model.PVD.size == "large":
            model = PVCNN2Unet(cfg)
        else:
            raise NotImplementedError(cfg.model.PVD.size)
    elif cfg.model.type == "Mink":
        model = MinkUnet(
            dim=cfg.model.time_embed_dim,
            init_ds_factor=cfg.model.Mink.init_ds_factor,
            D=cfg.model.Mink.D,
            in_shape=[cfg.training.bs, cfg.model.in_dim, cfg.data.npoints],
            out_dim=cfg.model.out_dim,
            in_channels=cfg.model.in_dim + cfg.model.extra_feature_channels,
            dim_mults=cfg.model.Mink.dim_mults,
            downsampfactors=cfg.model.Mink.downsampfactors,
            use_attention=cfg.model.Mink.use_attention,
        )
    elif cfg.model.type == "SetTransformer":
        set_transformer = SetTransformer(
            n_layers=cfg.model.ST.layers,
            feature_dim=cfg.model.ST.fdim,
            num_inducers=cfg.model.ST.inducers,
            t_embed_dim=1,
        )
        model = LinearLift(
            inner=set_transformer,
            feature_dim=cfg.model.ST.fdim,
            in_dim=cfg.model.in_dim + cfg.model.extra_feature_channels,
            out_dim=cfg.model.out_dim,
        )
    else:
        model = None
        raise NotImplementedError(cfg.unet)
    logger.info(
        f"Generated model with following number of params (M): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}"
    )
    return model


def load_diffusion(cfg, smart=False):
    # setup model
    backbone = load_model(cfg).to(cfg.gpu)
    if cfg.diffusion.formulation.lower() == "lucid":
        model = LUCID(cfg=cfg, model=backbone)
    elif cfg.diffusion.formulation.lower() == "i2sb":
        model = I2SB(cfg=cfg, model=backbone)
    else:
        raise NotImplementedError(cfg.diffusion.formulation)

    gpu = cfg.gpu

    # setup DDP model
    if cfg.distribution_type == "multi":

        def ddp_transform(m):
            return DistributedDataParallel(m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(ddp_transform)

    # setup data parallel model
    elif cfg.distribution_type == "single":

        def dp_transform(m):
            return DataParallel(m)

        model = model.cuda()
        model.multi_gpu_wrapper(dp_transform)

    # setup single gpu model
    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError("distribution_type = multi | single")

    # load the model weights
    cfg.start_step = 0
    if cfg.model_path != "":
        ckpt = torch.load(cfg.model_path, map_location=torch.device("cpu"))
        if not cfg.restart:
            cfg.start_step = ckpt["step"] + 1
            try:
                model.load_state_dict(ckpt["model_state"])
            except RuntimeError:
                logger.warning("Could not load model state dict. Trying to load only the model parameters.")
                model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            # only load the model parameters and let rest start from scratch
            model_keys = {k.replace("model.", ""): v for k, v in ckpt["model_state"].items() if k.startswith("model.")}
            model.model.load_state_dict(model_keys)

        logger.info("Loaded model from %s" % cfg.model_path)
    else:
        ckpt = None

    torch.cuda.empty_cache()

    return model, ckpt
