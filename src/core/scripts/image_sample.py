"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys
import yaml

# # 在导入torch之前设置
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 将当前脚本所在目录添加到Python模块搜索路径中
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
    parser.add_argument("--config", type=str, default="", help="YAML 采样配置路径，可覆盖命令行参数")
    args = parser.parse_args()

    # 加载 YAML 配置（若提供）
    yaml_cfg = {}
    if getattr(args, "config", ""):
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f)

    # 从 YAML 中回填采样与模型配置
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
    # 采样步数统一入口：sample.steps（若提供则覆盖 timestep_respacing）
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
        # 提前记录映射关系，避免用户被进程内索引（cuda:0）误导
        print(f"[GPU] 锁定 nvidia-smi 索引: {gpu_index} -> 进程内 cuda:0")

    dist_util.setup_dist()
    out_dir = paths_cfg.get("output_dir", "./sample")
    # 仅控制台输出，不写入 log/csv/json 等文件
    logger.configure(dir=out_dir, format_strs=["stdout"])

    logger.log("creating model and diffusion...")
    if th.cuda.is_available():
        try:
            cur = th.cuda.current_device()
            logger.log(f"当前使用 GPU: cuda:{cur} ({th.cuda.get_device_name(cur)})")
            if getattr(args, "gpu", -1) >= 0 and cur == 0:
                logger.log(f"注意：由于 CUDA_VISIBLE_DEVICES 设为 {args.gpu}，cuda:0 对应 nvidia-smi:{args.gpu}")
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
    # 仅保存 PNG，不保存 npz/日志文件
    saved = 0
    # 为单张图片输出准备目录
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
        # 对于灰度图，值范围从[-1,1]转换到[0,255]
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # 对于灰度图 shape为(batch_size, 1, height, width)
        # 如果需要去掉通道维度，可以使用squeeze
        sample = sample.squeeze(1)  # 结果shape为(batch_size, height, width)
        # 如果需要保持通道维度，可以直接使用contiguous()
        sample = sample.contiguous()

        # 保存本批次 PNG（裁剪到需要的数量）
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

    # 分布式善后，避免 NCCL 资源泄漏告警
    if dist.is_available() and dist.is_initialized():
        # 直接销毁进程组，避免 NCCL 资源泄漏告警
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
        use_ddim=True,  # 默认使用 DDIM 采样
        model_path="",
        gpu=-1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
