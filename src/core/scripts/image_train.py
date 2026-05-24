# -*- coding: utf-8 -*-
"""
Purpose: Train a StackDiff diffusion checkpoint from local single-layer/source STEM images with on-the-fly augmentation. Training writes checkpoints and logs under train.output_dir and can save augmented preview patches when save_samples.enable is true.
Related files: configs/train/*.yml, src/core/datasets/augment_dataset.py, src/core/augmentations/stem.py, and src/core/guided_diffusion/train_util.py.
CLI usage: python src/core/scripts/image_train.py --config configs/train/ReS2.yml (arguments: --config selects the training YAML; source_data.data_root and source_data.mask_path point to local source images and mask).
"""

import argparse
import sys
import os
import yaml


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from guided_diffusion import dist_util, logger
from datasets.augment_dataset import load_training_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import torch


def main():
    parser = create_argparser()
    parser.add_argument("--config", type=str, default="", help="Path to a YAML training config; values override command-line defaults")
    args = parser.parse_args()


    yaml_cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)



    # train: { output_dir, batch_size, lr, ema_rate, log_interval, save_interval, lr_anneal_steps, max_steps, microbatch, schedule_sampler, use_fp16 }

    train_cfg = yaml_cfg.get("train", {})
    source_data_cfg = yaml_cfg.get("source_data", {})
    augment_cfg = yaml_cfg.get("augment", {})
    save_samples_cfg = yaml_cfg.get("save_samples", {})
    model_cfg = yaml_cfg.get("model", {})


    for k, v in train_cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)

    for k, v in model_cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)



    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or not hasattr(args, dest):
            continue
        expected_type = getattr(action, "type", None)

        if expected_type is None:
            continue
        cur_val = getattr(args, dest)

        try:
            if cur_val is not None and not isinstance(cur_val, expected_type):

                if expected_type is bool and isinstance(cur_val, bool):
                    continue

                setattr(args, dest, expected_type(cur_val))
        except Exception:

            pass


    gpu = getattr(args, "gpu", -1)
    if isinstance(gpu, str):
        try:
            gpu = int(gpu)
        except Exception:
            gpu = -1
    if gpu is not None and int(gpu) >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
        print(f"[GPU] Using nvidia-smi GPU index: {int(gpu)} -> process-local cuda:0")


    dist_util.setup_dist()
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    out_dir = train_cfg.get("output_dir", os.path.join(os.getcwd(), "train_logs"))
    logger.configure(dir=out_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    if torch.cuda.is_available():
        try:
            cur = torch.cuda.current_device()
            logger.log(f"CUDA device index: {cur}, name: {torch.cuda.get_device_name(cur)}")
            if int(gpu) >= 0 and cur == 0:
                logger.log("CUDA_VISIBLE_DEVICES remaps the selected physical GPU to process-local cuda:0.")
        except Exception:
            pass

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if not bool(source_data_cfg.get("enable", True)):
        raise ValueError("This public training entry expects source_data.enable=true in the YAML config.")
    data_root = str(source_data_cfg.get("data_root", "data/training_source")).strip()
    mask_path = str(source_data_cfg.get("mask_path", os.path.join(data_root, "mask.png"))).strip()
    data = load_training_data(
        data_root=data_root,
        mask_path=mask_path,
        mask_cache=source_data_cfg.get("mask_cache"),
        batch_size=args.batch_size,
        image_size=args.image_size,
        augment_cfg=augment_cfg,
        save_samples_cfg=save_samples_cfg,
        num_workers=int(source_data_cfg.get("num_workers", 4)),
        pin_memory=bool(source_data_cfg.get("pin_memory", True)),
        prefetch_factor=int(source_data_cfg.get("prefetch_factor", 2)),
        persistent_workers=bool(source_data_cfg.get("persistent_workers", True)),
    )
    logger.log(f"source data enabled. root={data_root}, mask={mask_path}")

    logger.log("training...")
    train_loop = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        max_steps=getattr(args, "max_steps", 0),
    )
    train_loop.run_loop()


def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        max_steps=0,
        batch_size=32,
        microbatch=-1,  # -1 disables microbatches
        #microbatch=8,
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        # resume_checkpoint="MoS2_STEM_total_plus_atom_loss_single_s_mdsave/model200000.pt",
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,

        gpu=-1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
