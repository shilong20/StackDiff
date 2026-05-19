"""
Purpose: Generate unconditional STEM-like image samples from a trained StackDiff checkpoint and write PNG files to disk. YAML configs under configs/sample provide model, runtime, checkpoint, and output settings.
Related files: configs/sample/*.yml, src/core/guided_diffusion/script_util.py, and src/core/guided_diffusion/dist_util.py.
CLI usage: python src/core/scripts/image_sample.py --config configs/sample/ReS2.yml (arguments: --config selects the sampling YAML; legacy guided-diffusion arguments remain available).
"""

import argparse
import os
import sys
import yaml


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    parser = create_argparser()
    parser.add_argument("--config", type=str, default="", help="Path to a YAML sampling config; values override command-line defaults")
    args = parser.parse_args()


    yaml_cfg = {}
    if getattr(args, "config", ""):
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)


    sample_cfg = yaml_cfg.get("sample", {})
    model_cfg = yaml_cfg.get("model", {})
    paths_cfg = yaml_cfg.get("paths", {})
    runtime_cfg = yaml_cfg.get("runtime", {})

    for k, v in sample_cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    for k, v in model_cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)

    if "steps" in sample_cfg and sample_cfg["steps"] is not None:
        args.timestep_respacing = str(sample_cfg["steps"])  # e.g., "200"
    if paths_cfg.get("model_path"):
        args.model_path = paths_cfg["model_path"]

    gpu_value = getattr(args, "gpu", -1)
    if sample_cfg.get("gpu") is not None:
        gpu_value = sample_cfg.get("gpu")
    if runtime_cfg.get("gpu") is not None:
        gpu_value = runtime_cfg.get("gpu")
    try:
        gpu_index = int(gpu_value) if gpu_value is not None else -1
    except (TypeError, ValueError):
        gpu_index = -1
    setattr(args, "gpu", gpu_index)
    if gpu_index is not None and gpu_index >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        print(f"[GPU] Using nvidia-smi GPU index: {gpu_index} -> process-local cuda:0")

    dist_util.setup_dist()
    out_dir = paths_cfg.get("output_dir", "./sample")

    logger.configure(dir=out_dir, format_strs=["stdout"])

    logger.log("creating model and diffusion...")
    if th.cuda.is_available():
        try:
            cur = th.cuda.current_device()
            logger.log(f"Current GPU: cuda:{cur} ({th.cuda.get_device_name(cur)})")
            if getattr(args, "gpu", -1) >= 0 and cur == 0:
                logger.log(f"Note: CUDA_VISIBLE_DEVICES is set to {args.gpu}，cuda:0 maps to nvidia-smi:{args.gpu}")
        except Exception:
            pass
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")

    saved = 0

    save_dir = os.path.join(out_dir, "raw")
    os.makedirs(save_dir, exist_ok=True)

    while saved < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)


        sample = sample.squeeze(1)

        sample = sample.contiguous()


        b = sample.shape[0]
        take = min(b, args.num_samples - saved)
        arr = sample[:take].cpu().numpy()

        logger.log("saving individual images...")
        from PIL import Image
        for i in range(take):
            img_path = os.path.join(save_dir, f"sample_{saved + i:05d}.png")
            img = Image.fromarray(arr[i])
            img.save(img_path)
        saved += take
        logger.log(f"saved {saved}/{args.num_samples} images to {save_dir}")


    if dist.is_available() and dist.is_initialized():

        try:
            dist.destroy_process_group()
        except Exception:
            pass
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=4,
        batch_size=4,
        use_ddim=True,
        model_path="",
        gpu=-1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
