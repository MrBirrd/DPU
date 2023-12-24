import os
import random
from shutil import copyfile

import numpy as np
import torch
from loguru import logger


def set_global_gpu_env(opt):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    torch.cuda.set_device(opt.gpu)


def copy_source(file, output_dir):
    copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def get_output_dir(prefix, cfg):
    output_dir = os.path.join(prefix, cfg.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def set_seed(opt):
    if opt.training.seed is None:
        opt.training.seed = 42

    # different seed per gpu
    opt.training.seed += opt.rank

    logger.info("Random Seed: {}", opt.training.seed)
    random.seed(opt.training.seed)
    torch.manual_seed(opt.training.seed)
    np.random.seed(opt.training.seed)
    if opt.gpu is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.training.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def setup_output_subdirs(output_dir, *subfolders):
    output_subdirs = output_dir
    try:
        os.makedirs(output_subdirs)
    except OSError:
        pass

    subfolder_list = []
    for sf in subfolders:
        curr_subf = os.path.join(output_subdirs, sf)
        try:
            os.makedirs(curr_subf)
        except OSError:
            pass
        subfolder_list.append(curr_subf)

    return subfolder_list
