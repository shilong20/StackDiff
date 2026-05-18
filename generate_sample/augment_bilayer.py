"""
【作用概述】对已生成的双层仿真图（通常是 debug1024 或其他输入目录）复用在线增强配置，输出 128×128 PNG；主要用于调试/对照，不是主生成入口。
【关联说明】文件/模块：generate_sample/batch_config.json（读取 augment/mask_path）；src/data_prep/online_augmentor.py；generate_sample/batch_runner.py（主生成/更推荐）。
【命令行用法】python generate_sample/augment_bilayer.py --config generate_sample/batch_config.json --input_dir <in> --output_dir <out>（参数：--num=限制数量；--overwrite=覆盖输出）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
SYS_SRC = ROOT_DIR / "src"
if str(SYS_SRC) not in sys.path:
    sys.path.append(str(SYS_SRC))

from data_prep.online_augmentor import (  # type: ignore  # noqa: E402
    AugmentConfig,
    OnlineAugmentor,
    distance_transform,
    to_float01,
)


TARGET_SIZE = 128


def _as_path(base_dir: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description="对双层仿真图像做在线增强并输出 128×128 PNG（命名保持不变）")
    ap.add_argument("--config", default="generate_sample/batch_config.json", help="包含 augment/mask_path 的 JSON 配置")
    ap.add_argument("--input_dir", default=None, help="覆盖配置中的 input_dir（默认用 output_dir_debug1024 或 output_dir）")
    ap.add_argument("--output_dir", default=None, help="覆盖配置中的 output_dir")
    ap.add_argument("--num", type=int, default=None, help="限制输出数量（默认处理所有输入图）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（覆盖配置）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_dir = cfg_path.parent

    augment = data.get("augment")
    if not isinstance(augment, dict):
        raise SystemExit("配置缺少 augment 字段")

    seed = int(data.get("seed", 0)) if args.seed is None else int(args.seed)
    np.random.seed(seed)

    output_dir = _as_path(base_dir, str(data.get("output_dir", "aug_out"))) if args.output_dir is None else Path(args.output_dir).resolve()
    mask_path = _as_path(base_dir, str(data.get("mask_path", "mask/ReS2_mask.png")))

    # 默认输入：如果存在 debug1024 目录就优先用它，否则用 output_dir 本身
    if args.input_dir is None:
        default_debug = output_dir.parent / f"{output_dir.name}_debug1024"
        input_dir = default_debug if default_debug.exists() else output_dir
    else:
        input_dir = Path(args.input_dir).resolve()

    if not input_dir.exists():
        raise SystemExit(f"input_dir 不存在: {input_dir}")
    if not mask_path.exists():
        raise SystemExit(f"mask_path 不存在: {mask_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    mask_img = Image.open(mask_path).convert("L")
    mask01 = to_float01(mask_img)
    dist_map0 = distance_transform(mask01)
    aug_cfg = AugmentConfig(image_size=TARGET_SIZE, cfg=augment)
    augmentor = OnlineAugmentor(mask01=mask01, dist_map=dist_map0, aug_cfg=aug_cfg)

    paths = sorted([p for p in input_dir.rglob("*.png") if p.is_file()])
    if args.num is not None:
        paths = paths[: max(0, int(args.num))]

    for p in paths:
        out = output_dir / p.name
        if out.exists() and not args.overwrite:
            continue
        with Image.open(p) as im:
            im = im.convert("L")
            aug = augmentor(im)
            img01 = (np.clip(aug[0], -1.0, 1.0) + 1.0) * 0.5
            img_u8 = (np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(img_u8, mode="L").save(out, format="PNG")

    print(f"✅ 完成，输出目录：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
