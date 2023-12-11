import argparse
import os
from omegaconf import OmegaConf


def parse_args():
    # make parser which accepts optinal arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file.")
    parser.add_argument("--name", type=str, default="", help="Name of the experiment.")
    parser.add_argument("--save_dir", default="checkpoints", help="path to save models")
    parser.add_argument("--model_path", type=str, default="", help="path to model (to continue training)")
    parser.add_argument("--restart", action="store_true", help="restart training from scratch")

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
    if args.config is not None:
        cfg = OmegaConf.load(args.config)
    elif args.model_path != "":
        try:
            cfg = OmegaConf.load(os.path.join(os.path.dirname(args.model_path), "opt.yaml"))
        except FileNotFoundError:
            cfg = OmegaConf.load(args.config)
    else:
        raise ValueError("config file must be specified or model path must be specified")

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
    if opt.name == "" and opt.config is not None:
        opt.name = os.path.splitext(os.path.basename(opt.config))[0]

    # set sampling output dir
    if opt.model_path != "":
        model_name = opt.model_path.split("/")[-1].split(".")[0].split("_")[-1]
        steps = min(opt.diffusion.sampling_timesteps, opt.diffusion.timesteps_clip)
        scheduler_info = f"{opt.diffusion.sampling_strategy}(T={str(steps)})"
        if opt.diffusion.clip:
            if opt.diffusion.dynamic_threshold:
                clip = "_clip_dynamic"
            else:
                clip = "_clip"
        else:
            clip = ""
        scheduler_info += clip
        opt.out_sampling = os.path.join(os.path.dirname(opt.model_path), "sampling", model_name, scheduler_info)

    return opt
