#!/usr/bin/env python3
"""
【作用概述】批量生成不同 Re 缺陷率（defect_rates_re）的数据集：读取材料 JSON 模板，逐个修改 defect_rates_re 与 output_dir，
并调用 `generate_sample/Batch_generate.py` 生成图片；最终得到多个输出文件夹（每个缺陷率一个）。
默认启用完整噪声模型（scan+poisson+gaussian），高斯噪声使用 GT 级别 sigma（0.129）。
【关联说明】文件/模块：generate_sample/ReS2.json；generate_sample/Batch_generate.py；generate_sample/batch_runner.py；
内置噪声参数方案。
【命令行用法】python tools/generate_defect_datasets.py --config generate_sample/ReS2.json
（参数：--rates=0.05,0.10,...；--output-parent=data/experiments；--no-noise=不加噪声；--keep-config=不恢复原配置）。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import List

# GT 噪声参数（与论文噪声鲁棒性实验参数一致）
GT_NOISE_CONFIG = {
    "mode": "full",
    "pixel_size_A_orig": 0.12,
    "beam_current_A": 3.0e-11,
    "dwell_time_s": 3.0e-6,
    "width_orig_px": 3289,
    "sigma_jitter_A": 0.02,
    "line_freq_hz": 60.0,
    "phase_x": 0.0,
    "phase_y": math.pi / 2,
    "gaussian_sigma": 0.129,
}


def _parse_rates(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("rates 为空，请提供形如 0.05,0.10,0.15,0.20")
    return [float(p) for p in parts]


def _fmt_rate(rate: float) -> str:
    return f"{float(rate):.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="批量生成不同 defect_rates_re 的数据集（每个缺陷率一个文件夹）")
    ap.add_argument("--config", default="generate_sample/ReS2.json", help="材料 JSON 配置模板路径")
    ap.add_argument("--rates", default="0.05,0.10,0.15,0.20", help="缺陷率列表（逗号分隔）")
    ap.add_argument(
        "--output-parent",
        default=None,
        help="输出父目录（默认使用 config.output_dir 的父目录；例如 data/experiments）",
    )
    ap.add_argument(
        "--output-prefix",
        default="ReS2_defect",
        help="输出目录名前缀（默认 ReS2_defect，将生成 ReS2_defect0.05 等）",
    )
    ap.add_argument("--no-noise", action="store_true", help="不添加噪声（默认添加 GT 级别的完整噪声模型）")
    ap.add_argument("--keep-config", action="store_true", help="兼容旧参数；当前脚本只写临时配置，不会修改原材料 JSON")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要生成的目录，不实际运行")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"config 不存在：{cfg_path}")

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    rates = _parse_rates(str(args.rates))

    original_text = cfg_path.read_text(encoding="utf-8")

    if args.output_parent is None:
        original_output_dir = data.get("output_dir", "")
        if not isinstance(original_output_dir, str) or not original_output_dir.strip():
            raise SystemExit("config 缺少 output_dir，无法推断 output_parent。")
        output_parent = Path(original_output_dir).parent
    else:
        output_parent = Path(args.output_parent)

    if args.dry_run:
        for r in rates:
            out_dir = output_parent / f"{args.output_prefix}{_fmt_rate(r)}"
            noise_info = "no noise" if args.no_noise else f"full noise (sigma={GT_NOISE_CONFIG['gaussian_sigma']})"
            print(f"rate={r:.3f} -> output_dir={out_dir}  [{noise_info}]")
        return 0

    import tempfile

    for r in rates:
        run_data = copy.deepcopy(data)
        run_data["defect_rates_re"] = [float(r)]
        run_data["dopant_rates_re"] = [0.0]
        out_dir = output_parent / f"{args.output_prefix}{_fmt_rate(r)}"
        run_data["output_dir"] = str(out_dir)

        augment = run_data.setdefault("augment", {})
        disable_list = list(augment.get("disable", []))

        if args.no_noise:
            if "noise" not in disable_list:
                disable_list.append("noise")
            if "display" not in disable_list:
                disable_list.append("display")
        else:
            disable_list = [d for d in disable_list if d not in ("noise",)]
            if "display" not in disable_list:
                disable_list.append("display")
            augment["noise"] = copy.deepcopy(GT_NOISE_CONFIG)

        augment["disable"] = disable_list

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", dir=cfg_path.parent, delete=False, encoding="utf-8",
        ) as tmp:
            tmp.write(json.dumps(run_data, ensure_ascii=False, indent=2) + "\n")
            tmp_path = Path(tmp.name)

        try:
            noise_info = "no noise" if args.no_noise else f"full noise (sigma={GT_NOISE_CONFIG['gaussian_sigma']})"
            cmd = [sys.executable, "generate_sample/Batch_generate.py", "--config", str(tmp_path)]
            print(f"\n[RUN] defect_rates_re={[float(r)]} output_dir={out_dir}  [{noise_info}]")
            subprocess.run(cmd, check=True)
        finally:
            tmp_path.unlink(missing_ok=True)

    print("\n完成。输出目录：")
    for r in rates:
        out_dir = output_parent / f"{args.output_prefix}{_fmt_rate(r)}"
        print(f"- {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
