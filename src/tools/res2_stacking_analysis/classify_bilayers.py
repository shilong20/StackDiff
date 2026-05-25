#!/usr/bin/env python3
"""
Purpose: Batch-process folders of separated bilayer ReS2 images, estimate each layer origin and da/db vectors, and classify samples as slip, twist, flip_slip, flip_twist, or unknown. Outputs are CSV summaries and optional atom-point JSON files; when paths lack a physical-size tag, the da/db template-growth step uses the --sideA_A fallback.
Related files: src/tools/res2_stacking_analysis/single_layer_lattice.py, src/tools/res2_stacking_analysis/atoms.py, src/tools/res2_stacking_analysis/cycles.py, and src/tools/res2_stacking_analysis/analyze_interlayer.py.
CLI usage: python src/tools/res2_stacking_analysis/classify_bilayers.py --root outputs/ReS2 --out_csv outputs/res2_stacking.csv (arguments: --root=input folders; --out_csv=summary CSV; --atoms_out_dir=optional atom JSON directory; --sideA_A=field-of-view side length in Angstrom, default 27.9).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.tools.res2_stacking_analysis.atoms import AtomDetectConfig, detect_atoms_from_image  # noqa: E402
from src.tools.res2_stacking_analysis.single_layer_lattice import run_single_layer_lattice_pipeline  # noqa: E402


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def _cross_z(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    return float(a[0] * b[1] - a[1] * b[0])


def _handedness_sign(da: np.ndarray, db: np.ndarray) -> int:
    da = np.asarray(da, dtype=np.float64).reshape(2)
    db = np.asarray(db, dtype=np.float64).reshape(2)
    cz = _cross_z(da, db)
    eps = 1e-6 * float(np.hypot(da[0], da[1])) * float(np.hypot(db[0], db[1]))
    if abs(cz) <= max(eps, 1e-9):
        return 0
    return 1 if cz > 0 else -1


def _angle_deg_mod180(u: np.ndarray, v: np.ndarray) -> float:
    """Internal helper."""
    a = np.asarray(u, dtype=np.float64).reshape(2)
    b = np.asarray(v, dtype=np.float64).reshape(2)
    na = float(np.hypot(a[0], a[1]))
    nb = float(np.hypot(b[0], b[1]))
    if na <= 1e-9 or nb <= 1e-9:
        return float("nan")
    c = float((a[0] * b[0] + a[1] * b[1]) / (na * nb))
    c = abs(c)
    c = max(-1.0, min(1.0, c))
    return float(math.degrees(math.acos(c)))


def _reflect_vec_about_axis(v: np.ndarray, axis_dir: np.ndarray) -> np.ndarray:
    """Internal helper."""
    vv = np.asarray(v, dtype=np.float64).reshape(2)
    n = _unit(np.asarray(axis_dir, dtype=np.float64).reshape(2))
    return (2.0 * float(np.dot(vv, n)) * n - vv).astype(np.float64)


def _align_to_ref(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    r = np.asarray(ref, dtype=np.float64).reshape(2)
    if float(np.dot(v, r)) < 0.0:
        return -v
    return v


def _find_layer_images(folder: Path) -> Tuple[Optional[Path], Optional[Path]]:
    p0 = next(iter(sorted(folder.glob("*_0.png"))), None)
    p1 = next(iter(sorted(folder.glob("*_1.png"))), None)
    return p0, p1


def _safe_id_for_image(img_path: Path) -> str:
    rel = str(img_path.as_posix())
    h = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    stem = img_path.stem
    return f"{h}_{stem}"


def _write_atoms_json(atoms_out_dir: Path, img_path: Path, pts: np.ndarray, *, out_scale: float) -> str:
    atoms_out_dir.mkdir(parents=True, exist_ok=True)
    sid = _safe_id_for_image(img_path)
    out = atoms_out_dir / f"{sid}_atoms.json"
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    pts512 = (pts * float(out_scale)).astype(np.float64)
    data = [{"x": float(x), "y": float(y)} for (x, y) in pts512.tolist()]
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out.name


def _detect_atoms_for_image(img_path: Path, cfg: AtomDetectConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    img = Image.open(str(img_path)).convert("L")
    arr = np.array(img)
    pts, dbg = detect_atoms_from_image(arr, cfg=cfg, return_debug=True)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    dbg = dbg if isinstance(dbg, dict) else {}
    return pts, dbg


def _analyze_one_folder(
    folder: Path,
    *,
    slip_thresh_deg: float,
    sideA_A: Optional[float],
    atom_cfg: AtomDetectConfig,
    out_scale: float,
    atoms_out_dir: Optional[Path],
    atoms_index: List[Dict[str, Any]],
) -> Dict[str, Any]:
    folder = Path(folder)
    img0, img1 = _find_layer_images(folder)
    out: Dict[str, Any] = {
        "folder": str(folder.as_posix()),
        "img0": "" if img0 is None else img0.name,
        "img1": "" if img1 is None else img1.name,
        "class": "unknown",
        "ok": False,
        "reason": "",
    }

    if img0 is None or img1 is None:
        out["reason"] = "missing__0_or__1"
        return out


    pts0, dbg0 = _detect_atoms_for_image(img0, atom_cfg)
    pts1, dbg1 = _detect_atoms_for_image(img1, atom_cfg)
    if atoms_out_dir is not None:
        fn0 = _write_atoms_json(atoms_out_dir, img0, pts0, out_scale=float(out_scale))
        fn1 = _write_atoms_json(atoms_out_dir, img1, pts1, out_scale=float(out_scale))
        atoms_index.append(
            {
                "image": str(img0.as_posix()),
                "atoms_json": fn0,
                "n_points": int(pts0.shape[0]),
                "coord_sys": "scaled",
                "out_scale": float(out_scale),
                "thr": dbg0.get("thr"),
                "area_thr": dbg0.get("area_thr"),
            }
        )
        atoms_index.append(
            {
                "image": str(img1.as_posix()),
                "atoms_json": fn1,
                "n_points": int(pts1.shape[0]),
                "coord_sys": "scaled",
                "out_scale": float(out_scale),
                "thr": dbg1.get("thr"),
                "area_thr": dbg1.get("area_thr"),
            }
        )

    r0 = run_single_layer_lattice_pipeline(
        img0,
        out_scale=float(out_scale),
        atom_cfg=atom_cfg,
        use_highpass=atom_cfg.use_highpass,
        atoms_points_px=pts0,
        fallback_sideA_A=sideA_A,
        include_debug=False,
    )
    r1 = run_single_layer_lattice_pipeline(
        img1,
        out_scale=float(out_scale),
        atom_cfg=atom_cfg,
        use_highpass=atom_cfg.use_highpass,
        atoms_points_px=pts1,
        fallback_sideA_A=sideA_A,
        include_debug=False,
    )
    if not bool(r0.get("ok", False)) or not bool(r1.get("ok", False)):
        out["reason"] = f"dadb_failed:0={r0.get('reason','')} 1={r1.get('reason','')}"
        return out

    # always output per-layer origin/dadb (512)
    out["origin0_512"] = json.dumps(r0.get("origin_xy_512", []), ensure_ascii=False)
    out["origin1_512"] = json.dumps(r1.get("origin_xy_512", []), ensure_ascii=False)
    out["da0_512"] = json.dumps(r0.get("da_vec_512", []), ensure_ascii=False)
    out["db0_512"] = json.dumps(r0.get("db_vec_512", []), ensure_ascii=False)
    out["da1_512"] = json.dumps(r1.get("da_vec_512", []), ensure_ascii=False)
    out["db1_512"] = json.dumps(r1.get("db_vec_512", []), ensure_ascii=False)

    da0 = np.asarray(r0["da_vec_512"], dtype=np.float64).reshape(2)
    db0 = np.asarray(r0["db_vec_512"], dtype=np.float64).reshape(2)
    da1 = np.asarray(r1["da_vec_512"], dtype=np.float64).reshape(2)
    db1 = np.asarray(r1["db_vec_512"], dtype=np.float64).reshape(2)

    h0 = int(_handedness_sign(da0, db0))
    h1 = int(_handedness_sign(da1, db1))
    out["handedness_0"] = str(h0)
    out["handedness_1"] = str(h1)
    if h0 == 0 or h1 == 0:
        out["reason"] = "degenerate_cross"
        return out

    is_flip = bool(h0 != h1)
    out["is_flip"] = "1" if is_flip else "0"

    if is_flip:
        axis_dir = np.array([-db0[1], db0[0]], dtype=np.float64)  # ⟂ db0
        db0_cmp = _reflect_vec_about_axis(db0, axis_dir)
        theta = _angle_deg_mod180(db0_cmp, db1)
        out["theta_deg"] = f"{float(theta):.6f}"
        out["theta_method"] = "flip_reflect_db0_then_mod180"
        cls = "flip_slip" if float(theta) <= float(slip_thresh_deg) else "flip_twist"
    else:
        theta = _angle_deg_mod180(db0, db1)
        out["theta_deg"] = f"{float(theta):.6f}"
        out["theta_method"] = "noflip_mod180"
        cls = "slip" if float(theta) <= float(slip_thresh_deg) else "twist"

    out["class"] = cls
    out["ok"] = True

    # extra dadb mean (512) only for slip / flip_slip
    if cls == "slip":
        db1a = _align_to_ref(db1, db0)
        da1a = _align_to_ref(da1, da0)
        da_mean = 0.5 * (da0 + da1a)
        db_mean = 0.5 * (db0 + db1a)
        out["da_mean_512"] = json.dumps([float(da_mean[0]), float(da_mean[1])], ensure_ascii=False)
        out["db_mean_512"] = json.dumps([float(db_mean[0]), float(db_mean[1])], ensure_ascii=False)
    elif cls == "flip_slip":
        axis_dir = np.array([-db0[1], db0[0]], dtype=np.float64)
        da0r = _reflect_vec_about_axis(da0, axis_dir)
        db0r = _reflect_vec_about_axis(db0, axis_dir)
        db1a = _align_to_ref(db1, db0r)
        da1a = _align_to_ref(da1, da0r)
        da_mean = 0.5 * (da0r + da1a)
        db_mean = 0.5 * (db0r + db1a)
        out["da_mean_512"] = json.dumps([float(da_mean[0]), float(da_mean[1])], ensure_ascii=False)
        out["db_mean_512"] = json.dumps([float(db_mean[0]), float(db_mean[1])], ensure_ascii=False)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing bilayer sample subfolders.")
    ap.add_argument("--out_csv", default="", help="Output CSV path. If omitted, a timestamped CSV is written under src/tools/res2_stacking_analysis/.")
    ap.add_argument("--slip_thresh_deg", type=float, default=5.0)
    ap.add_argument("--sideA_A", type=float, default=27.9, help="Fallback field-of-view side length in Angstrom when image paths do not contain a size tag such as 2.79x2.79.")
    ap.add_argument("--out_scale", type=float, default=4.0, help="Coordinate scale for outputs. The default maps 128 px coordinates to 512 px.")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N subfolders; 0 means all.")

    # atoms output
    ap.add_argument("--atoms_out_dir", default="", help="Directory for atom point-cloud JSON files.")
    ap.add_argument("--no_atoms_out", action="store_true", help="Disable atom point-cloud JSON output.")

    # atom detect config
    ap.add_argument("--use_highpass", action="store_true", help="Force high-pass preprocessing")
    ap.add_argument("--no_highpass", action="store_true", help="Disable high-pass preprocessing")
    ap.add_argument("--bg_sigma", type=float, default=6.0)
    ap.add_argument("--bina_thre", type=float, default=1.8)
    ap.add_argument("--min_area", type=float, default=100.0)
    ap.add_argument("--min_dist", type=float, default=5.0)
    ap.add_argument("--max_points", type=int, default=4000)
    args = ap.parse_args()

    root = Path(str(args.root))
    if not root.exists():
        raise FileNotFoundError(str(root))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.out_csv).strip():
        out_csv = Path(str(args.out_csv))
    else:
        out_csv = _REPO_ROOT / "src" / "tools" / "res2_stacking_analysis" / f"res2_stacking_{stamp}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    atoms_out_dir: Optional[Path] = None
    if not bool(args.no_atoms_out):
        if str(args.atoms_out_dir).strip():
            atoms_out_dir = Path(str(args.atoms_out_dir))
        else:
            atoms_out_dir = out_csv.parent / f"{out_csv.stem}_atoms"
        atoms_out_dir.mkdir(parents=True, exist_ok=True)

    use_hp = True
    if bool(args.no_highpass):
        use_hp = False
    if bool(args.use_highpass):
        use_hp = True

    atom_cfg = AtomDetectConfig(
        use_highpass=bool(use_hp),
        bg_sigma_px=float(args.bg_sigma),
        bina_thre=float(args.bina_thre),
        min_area_threshold=float(args.min_area),
        min_distance_px=float(args.min_dist),
        max_points=int(args.max_points),
    )

    subdirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if int(args.limit) > 0:
        subdirs = subdirs[: int(args.limit)]

    atoms_index: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for d in subdirs:
        rows.append(
            _analyze_one_folder(
                d,
                slip_thresh_deg=float(args.slip_thresh_deg),
                sideA_A=float(args.sideA_A),
                atom_cfg=atom_cfg,
                out_scale=float(args.out_scale),
                atoms_out_dir=atoms_out_dir,
                atoms_index=atoms_index,
            )
        )

    fieldnames = [
        "folder",
        "img0",
        "img1",
        "ok",
        "class",
        "is_flip",
        "handedness_0",
        "handedness_1",
        "theta_deg",
        "theta_method",
        "origin0_512",
        "origin1_512",
        "da0_512",
        "db0_512",
        "da1_512",
        "db1_512",
        "da_mean_512",
        "db_mean_512",
        "reason",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    if atoms_out_dir is not None:
        (atoms_out_dir / "atoms_index.json").write_text(json.dumps(atoms_index, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"out_csv={out_csv}")
    if atoms_out_dir is not None:
        print(f"atoms_out_dir={atoms_out_dir}")
    print(f"folders={len(subdirs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
