#!/usr/bin/env python3
"""
【作用概述】
在“仿真 slip 双层数据”上端到端评估当前 pred_dadb pipeline 的正确性：
1) 调用 generate_sample 批量生成 ReS2 双层仿真图（slip），并导出 labels（npz）。
2) 从 labels 中导出两张单层图（*_0.png/*_1.png），构造成与 HighT1 类似的文件夹结构。
3) 调用 tools/pred_dadb/pipeline_bilayer_root.py 对单层对进行分类并估计每层的 (da,db)/origin/atoms（512 坐标系）。
4) 从 labels 的 GT（t1/t2 与几何增强 meta）推导 GT 的 (da,db)（512 坐标系），与 pred_dadb 输出做对比：
   - 分类是否全为 slip（期望：全部 slip；不应出现 twist/flip_*）
   - (da,db) 的方向误差（轴向夹角，[0,90]）与长度相对误差

【关联说明】文件/模块：
- generate_sample/Batch_generate.py（批量生成；需设置环境变量 MOIRE_SAVE_LABELS=1）
- generate_sample/export_monolayer_pngs_from_labels.py（从 labels 快速导出单层 PNG，避免额外 incostem 成像）
- tools/pred_dadb/pipeline_bilayer_root.py（当前 dadb/原点/分类入口，输出 512 坐标系）

【命令行用法】
python src/tools/analysis/eval_pred_dadb_on_synth_slip.py --num 50 --seed 123
（参数：--template_config=生成配置模板；--work_dir=输出工作目录；--incostem_monolayer=用 incostem 额外导出单层图（更接近实验图，但更慢）；默认从 labels 光栅化导出单层图；--no_highpass=仿真图建议关闭高通；--bina_thre/--min_area=原子检测参数）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _axis_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """轴向夹角（[0,90]）：v 与 -v 等价。"""
    uu = np.asarray(u, dtype=np.float64).reshape(2)
    vv = np.asarray(v, dtype=np.float64).reshape(2)
    nu = float(np.hypot(uu[0], uu[1]))
    nv = float(np.hypot(vv[0], vv[1]))
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")
    c = float(np.dot(uu, vv) / (nu * nv))
    c = abs(max(-1.0, min(1.0, c)))
    return float(np.degrees(np.arccos(c)))


def _rot2_from_cv2_angle_deg(angle_deg: float) -> np.ndarray:
    """
    复现 cv2.getRotationMatrix2D 的 2x2 部分：
      x' = cos*x + sin*y + ...
      y' = -sin*x + cos*y + ...
    对向量只需 2x2。
    """
    a = math.radians(float(angle_deg))
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def _gt_vec_px512_from_labels(npz: Dict[str, Any], which: str) -> np.ndarray:
    """
    从 labels 的 v1A/v2A + meta 计算 GT 的像素向量（512 坐标系）。
    约定（与对话一致）：
    - db == t1 == v1A
    - da == t2 == v2A（本次把 t2 修正为与 t1 夹角约 120° 的基矢）
    """
    if which not in {"da", "db"}:
        raise ValueError(which)
    v1A = np.asarray(npz["v1A"], dtype=np.float64).reshape(2)
    v2A = np.asarray(npz["v2A"], dtype=np.float64).reshape(2)
    if which == "db":
        vA = v1A
    else:
        vA = v2A
    # 注意：batch_runner 保存到 npz 的 v1A/v2A 已经同步应用了 rotate/flip（在最终图像坐标系里，以 Å 表示）。
    # 因此这里只需要把 Å -> 像素：A_per_px = sideA / 128。
    sideA_A = float(npz["sideA"])  # Å，最终 128×128 视野边长
    A_per_px = sideA_A / 128.0
    v128 = (vA / max(A_per_px, 1e-12)).astype(np.float64)
    return (v128 * 4.0).astype(np.float64)  # 128 -> 512


def _parse_vec(s: str) -> Optional[np.ndarray]:
    ss = str(s).strip()
    if not ss:
        return None
    try:
        v = json.loads(ss)
    except Exception:
        return None
    if not isinstance(v, list) or len(v) != 2:
        return None
    return np.array([float(v[0]), float(v[1])], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--template_config", default="generate_sample/batch_config_test_res2_2p8nm_slip.json")
    ap.add_argument("--work_dir", default="", help="输出工作目录（默认 data/experiments/ReS2_eval_slip_<stamp>/）")
    ap.add_argument("--incostem_monolayer", action="store_true", help="生成时额外用 incostem 导出单层图（更慢，但更像真实 HAADF）")
    ap.add_argument("--mono_sigma_px", type=float, default=0.6, help="从 labels 光栅化单层图时的高斯点 sigma（像素）")

    # pred_dadb atom detect params
    ap.add_argument("--use_highpass", action="store_true")
    ap.add_argument("--no_highpass", action="store_true")
    ap.add_argument("--bg_sigma", type=float, default=6.0)
    ap.add_argument("--bina_thre", type=float, default=1.8)
    ap.add_argument("--min_area", type=float, default=5.0, help="仿真点云图通常较小，默认更低")
    ap.add_argument("--min_dist", type=float, default=2.0)
    ap.add_argument("--max_points", type=int, default=5000)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.work_dir).strip():
        work_dir = Path(str(args.work_dir)).resolve()
    else:
        work_dir = (_REPO_ROOT / "data" / "experiments" / f"ReS2_eval_slip_{stamp}").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1) 生成 config（在 work_dir 下）
    template_path = (_REPO_ROOT / str(args.template_config)).resolve()
    cfg = json.loads(template_path.read_text(encoding="utf-8"))
    cfg["num"] = int(args.num)
    cfg["seed"] = int(args.seed)
    cfg["output_dir"] = str((work_dir / "bilayer_png").as_posix())
    # 由于本脚本把 config 写到 work_dir 下，需把模板里的相对路径改成绝对路径（否则会相对 work_dir 解析）。
    for k, base in [
        ("structure_path", _REPO_ROOT / "generate_sample"),
        ("incostem_path", _REPO_ROOT / "generate_sample"),
        ("mask_path", _REPO_ROOT),
    ]:
        if k in cfg and isinstance(cfg[k], str) and cfg[k]:
            p = Path(cfg[k])
            if not p.is_absolute():
                cfg[k] = str((base / p).resolve().as_posix())
    gen_cfg_path = work_dir / "batch_config_eval.json"
    gen_cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) 生成 bilayer + labels（不额外跑 incostem 导单层；单层从 labels 光栅化导出）
    env = dict(os.environ)
    env["MOIRE_SAVE_LABELS"] = "1"
    if bool(args.incostem_monolayer):
        env["MOIRE_SAVE_MONOLAYER"] = "1"
    else:
        env.pop("MOIRE_SAVE_MONOLAYER", None)
    subprocess.run(
        [sys.executable, str(_REPO_ROOT / "generate_sample" / "Batch_generate.py"), "--config", str(gen_cfg_path)],
        check=True,
        env=env,
        cwd=str(_REPO_ROOT),
    )

    bilayer_dir = Path(cfg["output_dir"]).resolve()
    labels_dir = bilayer_dir.parent / f"{bilayer_dir.name}_labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(str(labels_dir))

    # 3) 导出单层图像
    if bool(args.incostem_monolayer):
        mono_root = bilayer_dir.parent / f"{bilayer_dir.name}_monolayer"
        if not mono_root.is_dir():
            raise FileNotFoundError(str(mono_root))
    else:
        mono_root = work_dir / "monolayer_root"
        mono_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "generate_sample" / "export_monolayer_pngs_from_labels.py"),
                "--labels_dir",
                str(labels_dir),
                "--out_root",
                str(mono_root),
                "--sigma_px",
                str(float(args.mono_sigma_px)),
                "--overwrite",
            ],
            check=True,
            cwd=str(_REPO_ROOT),
        )

    # 4) 运行 pred_dadb pipeline（得到分类与 dadb/origin/atoms）
    pred_csv = work_dir / "pred_dadb.csv"
    pred_atoms_dir = work_dir / "pred_atoms"
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "tools" / "pred_dadb" / "pipeline_bilayer_root.py"),
        "--root",
        str(mono_root),
        "--out_csv",
        str(pred_csv),
        "--atoms_out_dir",
        str(pred_atoms_dir),
        "--bg_sigma",
        str(float(args.bg_sigma)),
        "--bina_thre",
        str(float(args.bina_thre)),
        "--min_area",
        str(float(args.min_area)),
        "--min_dist",
        str(float(args.min_dist)),
        "--max_points",
        str(int(args.max_points)),
    ]
    if bool(args.no_highpass):
        cmd.append("--no_highpass")
    elif bool(args.use_highpass):
        cmd.append("--use_highpass")
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))

    # 5) 评估：分类 + dadb
    npz_map: Dict[str, Path] = {p.stem: p for p in labels_dir.glob("*.npz")}

    rows_out: List[Dict[str, str]] = []
    n_total = 0
    n_class_ok = 0
    n_dadb_ok = 0
    bad_examples: List[str] = []

    with pred_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n_total += 1
            folder = str(r.get("folder", ""))
            cls = str(r.get("class", ""))
            expected_cls = "slip"
            class_ok = cls == expected_cls
            if class_ok:
                n_class_ok += 1

            img0 = str(r.get("img0", ""))
            stem = img0[:-6] if img0.endswith("_0.png") else Path(img0).stem
            npz_path = npz_map.get(stem)
            if npz_path is None:
                rows_out.append(
                    {
                        "folder": folder,
                        "class": cls,
                        "expected_class": expected_cls,
                        "class_ok": "1" if class_ok else "0",
                        "dadb_ok": "0",
                        "reason": "missing_npz",
                    }
                )
                bad_examples.append(folder)
                continue

            data = np.load(npz_path, allow_pickle=True)
            gt_da = _gt_vec_px512_from_labels(data, "da")
            gt_db = _gt_vec_px512_from_labels(data, "db")

            da0 = _parse_vec(str(r.get("da0_512", "")))
            db0 = _parse_vec(str(r.get("db0_512", "")))
            da1 = _parse_vec(str(r.get("da1_512", "")))
            db1 = _parse_vec(str(r.get("db1_512", "")))

            if any(v is None for v in (da0, db0, da1, db1)):
                rows_out.append(
                    {
                        "folder": folder,
                        "class": cls,
                        "expected_class": expected_cls,
                        "class_ok": "1" if class_ok else "0",
                        "dadb_ok": "0",
                        "reason": "missing_pred_vec",
                    }
                )
                bad_examples.append(folder)
                continue

            def _len(v: np.ndarray) -> float:
                return float(np.hypot(float(v[0]), float(v[1])))

            # per-layer errors (axis-angle + relative length)
            err = {}
            for tag, da_p, db_p in [("0", da0, db0), ("1", da1, db1)]:
                err[f"db_ang_deg_{tag}"] = f"{_axis_angle_deg(db_p, gt_db):.4f}"
                err[f"da_ang_deg_{tag}"] = f"{_axis_angle_deg(da_p, gt_da):.4f}"
                err[f"db_len_rel_{tag}"] = f"{abs(_len(db_p) - _len(gt_db)) / max(_len(gt_db), 1e-9):.4f}"
                err[f"da_len_rel_{tag}"] = f"{abs(_len(da_p) - _len(gt_da)) / max(_len(gt_da), 1e-9):.4f}"

            # dadb_ok: both layers within tolerances
            tol_ang = 5.0
            tol_len = 0.15
            dadb_ok = True
            for tag in ("0", "1"):
                if float(err[f"db_ang_deg_{tag}"]) > tol_ang or float(err[f"da_ang_deg_{tag}"]) > tol_ang:
                    dadb_ok = False
                if float(err[f"db_len_rel_{tag}"]) > tol_len or float(err[f"da_len_rel_{tag}"]) > tol_len:
                    dadb_ok = False

            if dadb_ok:
                n_dadb_ok += 1
            else:
                bad_examples.append(folder)

            rows_out.append(
                {
                    "folder": folder,
                    "class": cls,
                    "expected_class": expected_cls,
                    "class_ok": "1" if class_ok else "0",
                    "dadb_ok": "1" if dadb_ok else "0",
                    **err,
                }
            )

    eval_csv = work_dir / "eval_pred_dadb_vs_gt.csv"
    fieldnames = sorted({k for row in rows_out for k in row.keys()})
    with eval_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    summary = {
        "work_dir": str(work_dir.as_posix()),
        "bilayer_dir": str(bilayer_dir.as_posix()),
        "labels_dir": str(labels_dir.as_posix()),
        "mono_root": str(mono_root.as_posix()),
        "pred_csv": str(pred_csv.as_posix()),
        "pred_atoms_dir": str(pred_atoms_dir.as_posix()),
        "eval_csv": str(eval_csv.as_posix()),
        "n_total": int(n_total),
        "n_class_ok": int(n_class_ok),
        "n_dadb_ok": int(n_dadb_ok),
        "class_acc": float(n_class_ok / max(1, n_total)),
        "dadb_pass_rate": float(n_dadb_ok / max(1, n_total)),
        "bad_examples_top10": bad_examples[:10],
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
