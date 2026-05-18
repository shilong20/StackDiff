# -*- coding: utf-8 -*-

"""
Train a diffusion model on images.

支持两种用法：
1) 传统命令行参数（兼容原逻辑）
2) YAML 配置（推荐）：--config configs/train/<name>.yml
"""

import argparse
import sys
import os
import yaml

# 将当前脚本所在目录添加到Python模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 便于导入 src/data_prep 下的在线增强模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from guided_diffusion import dist_util, logger
from datasets.online_augment_dataset import load_online_data
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
    parser.add_argument("--config", type=str, default="", help="YAML 训练配置路径，可覆盖命令行参数")
    args = parser.parse_args()

    # 加载 YAML 配置（若提供）
    yaml_cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)

    # 训练与模型参数从 YAML 中读取并回填到 args
    # YAML 结构建议：
    # train: { data_dir, output_dir, batch_size, lr, ema_rate, log_interval, save_interval, lr_anneal_steps, max_steps, microbatch, schedule_sampler, use_fp16 }
    # model: 与 model_and_diffusion_defaults 对齐的字段
    train_cfg = yaml_cfg.get("train", {})
    online_cfg = yaml_cfg.get("online", {})
    augment_cfg = yaml_cfg.get("augment", {})
    save_samples_cfg = yaml_cfg.get("save_samples", {})
    model_cfg = yaml_cfg.get("model", {})

    # 回填 train 配置
    for k, v in train_cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    # 回填 model 配置
    for k, v in model_cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)

    # 按 argparse 中声明的类型进行强制类型转换（修复 YAML 将数值读成字符串的问题，如 lr: 1e-4）
    # 例：--lr 应为 float，但 YAML 可能解析为字符串 '1e-4'，需转回 float
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest or not hasattr(args, dest):
            continue
        expected_type = getattr(action, "type", None)
        # argparse 的 type 可能为 None（默认字符串），或自定义函数（如 str2bool）
        if expected_type is None:
            continue
        cur_val = getattr(args, dest)
        # 跳过 None 或已是目标类型的值
        try:
            if cur_val is not None and not isinstance(cur_val, expected_type):
                # 避免对已经是 bool 的再次转换
                if expected_type is bool and isinstance(cur_val, bool):
                    continue
                # 将字符串等转换为目标类型（float/int/bool 等）
                setattr(args, dest, expected_type(cur_val))
        except Exception:
            # 若转换失败则保留原值，以避免非关键字段导致崩溃
            pass

    # 优先通过 YAML/参数指定单卡；设置 CUDA_VISIBLE_DEVICES 后再初始化分布式
    gpu = getattr(args, "gpu", -1)
    if isinstance(gpu, str):
        try:
            gpu = int(gpu)
        except Exception:
            gpu = -1
    if gpu is not None and int(gpu) >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
        print(f"[GPU] 锁定 nvidia-smi 索引: {int(gpu)} -> 进程内 cuda:0")

    # 分布式与日志
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
    # 记录当前使用的 GPU 信息，便于排查绑定的设备
    if torch.cuda.is_available():
        try:
            cur = torch.cuda.current_device()
            logger.log(f"CUDA device index: {cur}, name: {torch.cuda.get_device_name(cur)}")
            if int(gpu) >= 0 and cur == 0:
                logger.log(f"注意：cuda:0 对应 nvidia-smi:{int(gpu)}（CUDA_DEVICE_ORDER=PCI_BUS_ID）")
        except Exception:
            pass

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data_root = online_cfg.get("data_root", "data/仿真数据集")
    mask_path = online_cfg.get("mask_path", os.path.join(data_root, "mask.png"))
    if not bool(online_cfg.get("enable", True)):
        raise ValueError("当前仓库仅支持在线训练，请在 YAML 中将 online.enable 设为 true。")
    data = load_online_data(
        data_root=data_root,
        mask_path=mask_path,
        mask_cache=online_cfg.get("mask_cache"),
        batch_size=args.batch_size,
        image_size=args.image_size,
        augment_cfg=augment_cfg,
        save_samples_cfg=save_samples_cfg,
        num_workers=int(online_cfg.get("num_workers", 4)),
        pin_memory=bool(online_cfg.get("pin_memory", True)),
        prefetch_factor=int(online_cfg.get("prefetch_factor", 2)),
        persistent_workers=bool(online_cfg.get("persistent_workers", True)),
    )
    logger.log(f"online data enabled. root={data_root}, mask={mask_path}")

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
        data_dir="",
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
        # 指定单卡 GPU 索引（与 nvidia-smi 一致），-1 表示不强制
        gpu=-1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
