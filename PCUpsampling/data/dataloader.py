import torch
from .arkitscenes import IndoorScenes, IndoorScenesCut, ArkitScans
from .scannetpp import ScanNetPPCut
from .shapenet import get_dataset_shapenet
from loguru import logger
import os
from torch.utils.data import Dataset, DataLoader


def save_iter(dataloader, sampler):
    """Return a save iterator over the loader, which supports multi-gpu training using a distributed sampler."""
    iterator = iter(dataloader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            if sampler is not None:
                sampler.set_epoch(sampler.epoch + 1)
            yield next(iterator)


def get_dataloader(opt, sampling=False):
    """
    Returns train and test dataloaders along with their respective samplers.

    Args:
        opt (argparse.Namespace): Command line arguments parsed by argparse.

    Returns:
        tuple: A tuple containing:
            - train_dataloader (torch.utils.data.DataLoader): Dataloader for training dataset.
            - test_dataloader (torch.utils.data.DataLoader): Dataloader for testing dataset.
            - train_sampler (torch.utils.data.distributed.DistributedSampler): Sampler for training dataset.
            - test_sampler (torch.utils.data.distributed.DistributedSampler): Sampler for testing dataset.
    """
    test_dataset = None
    if opt.data.dataset == "ShapeNet":
        train_dataset, _ = get_dataset_shapenet(
            dataroot=opt.data.data_dir,
            npoints=opt.data.npoints,
            category="chair",
            upsample_frac=opt.data.upsample_frac,
            overfit=opt.training.overfit,
        )
    elif opt.data.dataset == "Indoor":
        logger.info("Loading IndoorScenes dataset, which is currently only for overfitting on one scene!")
        train_dataset = IndoorScenes(
            opt.data.data_dir,
            opt.data.npoints,
            voxel_size=opt.data.voxel_size,
            normalize=opt.data.normalize,
        )
    elif opt.data.dataset == "IndoorCut":
        train_dataset = IndoorScenesCut(
            opt.data.data_dir,
            opt.data.npoints,
            voxel_size=opt.data.voxel_size,
            normalize=opt.data.normalize,
        )
    elif opt.data.dataset == "Arkit":
        train_dataset = ArkitScans(
            os.path.join(opt.data.data_dir, "Training"),
            opt.data.npoints,
            voxel_size=opt.data.voxel_size,
            normalize=opt.data.normalize,
            unconditional=opt.data.unconditional,
        )
    elif opt.data.dataset == "ScanNetPP":
        train_dataset = (
            ScanNetPPCut(npoints=opt.data.npoints, root=opt.data.data_dir, mode="training") if not sampling else None
        )
        test_dataset = ScanNetPPCut(npoints=opt.data.npoints, root=opt.data.data_dir, mode="validation")
    else:
        raise NotImplementedError(f"Dataset {opt.data.dataset} not implemented!")

    # setup the samplers
    if opt.distribution_type == "multi":
        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=opt.world_size, rank=opt.rank)
            if train_dataset is not None
            else None
        )
        test_sampler = (
            torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=opt.world_size, rank=opt.rank)
            if test_dataset is not None
            else None
        )
    else:
        train_sampler = None
        test_sampler = None

    # setup the dataloaders
    train_dataloader = (
        DataLoader(
            train_dataset,
            batch_size=opt.training.bs if not sampling else opt.sampling.bs,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            num_workers=int(opt.data.workers),
            prefetch_factor=4 if int(opt.data.workers) > 0 else None,
            drop_last=True,
        )
        if train_dataset is not None
        else None
    )

    test_dataloader = (
        DataLoader(
            test_dataset,
            batch_size=opt.training.bs if not sampling else opt.sampling.bs,
            sampler=test_sampler,
            shuffle=False,
            num_workers=int(opt.data.workers),
            drop_last=False,
        )
        if test_dataset is not None
        else None
    )

    return train_dataloader, test_dataloader, train_sampler, test_sampler
