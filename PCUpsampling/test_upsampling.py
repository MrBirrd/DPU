import os

import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data

from data.dataloader import get_dataloader, get_npz_loader
from model.loader import load_model, load_diffusion
from utils.args import parse_args
from utils.evaluation import evaluate
from utils.file_utils import set_seed


@torch.no_grad()
def sample(gpu, cfg, output_dir):
    set_seed(cfg)

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

        cfg.sampling.bs = int(cfg.sampling.bs / cfg.ngpus_per_node)

    # get the loaders
    if "eval_folder" in cfg:
        test_loader = get_npz_loader(cfg.eval_folder, cfg)
        cfg.out_sampling = cfg.out_sampling.replace("sampling", "sampling_npz")
    else:
        test_loader = get_dataloader(cfg, sampling=True)[1]

    model, _ = load_diffusion(cfg)

    ds_iter = iter(test_loader)
    model.eval()

    # run evaluatoin for each iteration and accumulate the stats
    metrics_df = pd.DataFrame()
    for eval_iter in range(cfg.sampling.num_iter):
        metrics = evaluate(model, ds_iter, cfg, eval_iter, sampling=True, save_npy=True, debug=False)
        if metrics_df.empty:
            metrics_df = pd.DataFrame.from_dict(metrics, orient="index").T
        else:
            metrics_df = pd.concat([metrics_df, pd.DataFrame.from_dict(metrics, orient="index").T])

    # save the metrics
    metrics_df.to_csv(os.path.join(cfg.out_sampling, "metrics.csv"))

    # cleanup
    if cfg.distribution_type == "multi":
        dist.destroy_process_group()


def main():
    opt = parse_args()

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == "multi":
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(sample, nprocs=opt.ngpus_per_node, args=(opt, opt.out_sampling))
    else:
        opt.gpu = None
        sample(opt.gpu, opt, opt.out_sampling)


if __name__ == "__main__":
    main()
