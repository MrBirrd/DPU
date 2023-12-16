import torch
from loguru import logger
from torch import optim
from torch.nn.parallel import DataParallel, DistributedDataParallel

try:
    from model.unet_mink import MinkUnet
except ImportError:
    logger.warning("Minkowski Engine not installed. Minkowski models will not be available.")

from model.unet_pointvoxel import PVCAdaptive, PVCLionSmall
from third_party.gecco_torch.models.linear_lift import LinearLift
from third_party.gecco_torch.models.set_transformer import SetTransformer
from .diffusion_lucid import GaussianDiffusion as LUCID


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


def load_model(cfg):
    if cfg.model.type == "PVD":
        if cfg.model.PVD.size == "small":
            model = PVCLionSmall(
                out_dim=cfg.model.out_dim,
                input_dim=cfg.model.in_dim,
                npoints=cfg.data.npoints,
                embed_dim=cfg.model.time_embed_dim,
                use_att=cfg.model.use_attention,
                use_st=cfg.model.PVD.use_st,
                dropout=cfg.model.dropout,
                extra_feature_channels=cfg.model.extra_feature_channels,
            )
        if cfg.model.PVD.size == "large":
            model = PVCAdaptive(
                out_dim=cfg.model.out_dim,
                input_dim=cfg.model.in_dim,
                npoints=cfg.data.npoints,
                channels=cfg.model.PVD.channels,
                embed_dim=cfg.model.time_embed_dim,
                use_att=cfg.model.PVD.use_attention,
                use_st=cfg.model.PVD.use_st,
                st_params=cfg.model.ST,
                dropout=cfg.model.dropout,
                extra_feature_channels=cfg.model.extra_feature_channels,
            )
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
        raise NotImplementedError(cfg.unet)
    logger.info(
            f"Generated model with following number of params (M): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}"
        )
    return model

def load_diffusion(cfg):
    # setup model
    backbone = load_model(cfg).to(cfg.gpu)
    model = LUCID(cfg=cfg, model=backbone)
    gpu = cfg.gpu

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
