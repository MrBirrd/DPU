import warnings

import mmcv
import torch
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from omegaconf import DictConfig
from third_party.ZegCLIP.models.segmentor.zegclip import ZegCLIP


def predict(model, img):
    B, C, H, W = img.shape

    img = preprocess_image(img)

    with torch.no_grad():
        out = model.extract_feat(img)

    # extract features
    numpy_arrays = []
    for item in out:  # Assume 'out' is a list of items
        if isinstance(item, tuple):
            # Process each tensor in the tuple
            for element in item:
                if isinstance(element, torch.Tensor):
                    numpy_arrays.append(element.detach().cpu().numpy())
        elif isinstance(item, torch.Tensor):
            # Directly convert the tensor
            numpy_arrays.append(item.detach().cpu().numpy())

    features = torch.from_numpy(numpy_arrays[0]).cuda().half().contiguous()

    # interpolate back
    features = torch.nn.functional.interpolate(features, (H, W), mode="bilinear")

    return features.cpu().numpy()


def preprocess_image(img):
    img = torch.nn.functional.interpolate(img, (512, 512), mode="bilinear")
    # make mean zero std 1, img is of shape B C H W
    img = img - torch.mean(img, dim=(2, 3), keepdim=True)
    img = img / torch.std(img, dim=(2, 3), keepdim=True)
    return img


def get_model():
    args = DictConfig(
        {
            "config": "third_party/ZegCLIP/configs/coco/vpt_seg_fully_vit-b_512x512_80k_12_100_multi.py",
            # checkpoint is from same folder this file is in
            "checkpoint": "/cluster/scratch/matvogel/models/coco_fully_512_vit_base.pth",
        }
    )
    args.gpu_id = 0
    args.launcher = "none"

    cfg = mmcv.Config.fromfile(args.config)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    cfg.gpu_ids = [args.gpu_id]
    distributed = False
    if len(cfg.gpu_ids) > 1:
        warnings.warn(
            f"The gpu-ids is reset from {cfg.gpu_ids} to "
            f"{cfg.gpu_ids[0:1]} to avoid potential error in "
            "non-distribute testing time."
        )
        cfg.gpu_ids = cfg.gpu_ids[0:1]

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    if "CLIP" in cfg.model.type:
        cfg.model.class_names = [0] * 10  # just dummy class names

    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    if hasattr(model, "presegmentor"):
        model.presegmentor.init_weights()  # to initial conv1/2 to get pseudo mask
    _ = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()

    model.eval()

    return model
