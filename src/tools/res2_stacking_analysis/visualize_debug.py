#!/usr/bin/env python3
"""
Purpose: Generate visual debug overlays for ReS2 stacking-analysis outputs. The
script draws detected atom positions, optional per-layer lattice vectors, and
optional slip/flip-slip alignment checks from classifier/interlayer CSV files.
It writes PNG overlays and JSON diagnostics without changing analysis results,
using --sideA_A as the default field-of-view side length when paths do not
contain a physical-size tag.
Related files: src/tools/res2_stacking_analysis/classify_bilayers.py,
src/tools/res2_stacking_analysis/analyze_interlayer.py,
src/tools/res2_stacking_analysis/single_layer_lattice.py, and atoms JSON files
under a res2_stacking atoms directory.
CLI usage: python src/tools/res2_stacking_analysis/visualize_debug.py --root
outputs/separation/ReS2_0 --stacking_csv outputs/res2_stacking.csv
--atoms_dir outputs/res2_stacking_atoms --interlayer_csv outputs/interlayer.csv
--out_dir outputs/res2_debug (arguments: --root=input folders;
--stacking_csv=classifier CSV; --atoms_dir=atom JSON directory;
--interlayer_csv=optional interlayer CSV used for alignment checks).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.tools.res2_stacking_analysis.atoms import AtomDetectConfig  # noqa: E402
from src.tools.res2_stacking_analysis.single_layer_lattice import run_single_layer_lattice_pipeline  # noqa: E402


def _read_csv_rows(path: Path, key: str = "folder") -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    if not path or not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[str(row.get(key, ""))] = dict(row)
    return rows


def _load_atoms_index(atoms_dir: Path) -> Tuple[Dict[str, Path], float]:
    index_path = atoms_dir / "atoms_index.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    img_to_json: Dict[str, Path] = {}
    out_scale = 4.0
    for item in data:
        img = str(item.get("image", ""))
        atoms_json = str(item.get("atoms_json", ""))
        if img and atoms_json:
            img_to_json[img] = atoms_dir / atoms_json
        if item.get("out_scale") is not None:
            out_scale = float(item["out_scale"])
    return img_to_json, out_scale


def _load_points(path: Optional[Path]) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros((0, 2), dtype=np.float64)
    data = json.loads(path.read_text(encoding="utf-8"))
    pts = [[float(p["x"]), float(p["y"])] for p in data]
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def _parse_vec(text: str) -> Optional[np.ndarray]:
    if not text:
        return None
    try:
        val = json.loads(text)
        arr = np.asarray(val, dtype=np.float64).reshape(2)
        if np.all(np.isfinite(arr)):
            return arr
    except Exception:
        return None
    return None


def _row_float(row: Dict[str, str], name: str) -> Optional[float]:
    try:
        text = str(row.get(name, "")).strip()
        if not text:
            return None
        val = float(text)
        return val if math.isfinite(val) else None
    except Exception:
        return None


def _as_rgb_canvas(image_path: Path, scale: float = 4.0) -> Image.Image:
    img = Image.open(image_path).convert("L")
    if abs(float(scale) - 1.0) > 1e-9:
        w, h = img.size
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.BILINEAR)
    return img.convert("RGB")


def _draw_points(
    draw: ImageDraw.ImageDraw,
    pts: np.ndarray,
    *,
    color: Tuple[int, int, int],
    radius: float = 2.4,
) -> None:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    r = float(radius)
    for x, y in pts:
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    origin: np.ndarray,
    vec: np.ndarray,
    *,
    color: Tuple[int, int, int],
    label: str,
    scale: float = 1.0,
) -> None:
    o = np.asarray(origin, dtype=np.float64).reshape(2)
    v = np.asarray(vec, dtype=np.float64).reshape(2) * float(scale)
    p = o + v
    draw.line((float(o[0]), float(o[1]), float(p[0]), float(p[1])), fill=color, width=3)
    n = float(np.hypot(v[0], v[1]))
    if n > 1e-9:
        u = v / n
        q = np.array([-u[1], u[0]], dtype=np.float64)
        head = 10.0
        a = p - head * u + 0.45 * head * q
        b = p - head * u - 0.45 * head * q
        draw.polygon([(float(p[0]), float(p[1])), (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))], fill=color)
    draw.text((float(p[0]) + 4.0, float(p[1]) + 4.0), label, fill=color)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.hypot(v[0], v[1]))
    if n <= 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def _reflect_points_about_axis(points: np.ndarray, *, origin: np.ndarray, axis_dir: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    o = np.asarray(origin, dtype=np.float64).reshape(2)
    n = _unit(axis_dir)
    v = pts - o.reshape(1, 2)
    return o.reshape(1, 2) + 2.0 * np.sum(v * n.reshape(1, 2), axis=1, keepdims=True) * n.reshape(1, 2) - v


def _write_layer_debug_json(
    *,
    img_path: Path,
    pts512: np.ndarray,
    out_path: Path,
    out_scale: float,
    atom_cfg: AtomDetectConfig,
    sideA_A: float,
) -> None:
    pts_img = np.asarray(pts512, dtype=np.float64).reshape(-1, 2) / float(out_scale)
    result = run_single_layer_lattice_pipeline(
        img_path,
        out_scale=float(out_scale),
        atom_cfg=atom_cfg,
        atoms_points_px=pts_img,
        use_highpass=bool(atom_cfg.use_highpass),
        fallback_sideA_A=float(sideA_A),
        include_debug=True,
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_layer_overlay(
    *,
    img_path: Path,
    pts512: np.ndarray,
    row: Dict[str, str],
    layer_idx: int,
    out_path: Path,
    out_scale: float,
) -> None:
    canvas = _as_rgb_canvas(img_path, scale=float(out_scale))
    draw = ImageDraw.Draw(canvas)
    _draw_points(draw, pts512, color=(255, 80, 80), radius=2.4)
    origin = _parse_vec(row.get(f"origin{layer_idx}_512", ""))
    da = _parse_vec(row.get(f"da{layer_idx}_512", ""))
    db = _parse_vec(row.get(f"db{layer_idx}_512", ""))
    if origin is not None:
        _draw_points(draw, origin.reshape(1, 2), color=(80, 220, 255), radius=4.5)
    if origin is not None and da is not None:
        _draw_arrow(draw, origin, da, color=(80, 220, 80), label="da", scale=0.9)
    if origin is not None and db is not None:
        _draw_arrow(draw, origin, db, color=(80, 140, 255), label="db", scale=0.9)
    canvas.save(out_path)


def _write_alignment_overlay(
    *,
    img0_path: Path,
    pts0: np.ndarray,
    pts1: np.ndarray,
    stacking_row: Dict[str, str],
    interlayer_row: Dict[str, str],
    out_path: Path,
    out_scale: float,
) -> bool:
    cls = str(stacking_row.get("class", ""))
    sx = _row_float(interlayer_row, "shift_px_x")
    sy = _row_float(interlayer_row, "shift_px_y")
    if cls not in {"slip", "flip_slip"} or sx is None or sy is None:
        return False

    pts0_use = np.asarray(pts0, dtype=np.float64).reshape(-1, 2)
    if cls == "flip_slip":
        origin0 = _parse_vec(stacking_row.get("origin0_512", ""))
        db_mean = _parse_vec(stacking_row.get("db_mean_512", ""))
        if db_mean is None:
            db_mean = _parse_vec(stacking_row.get("db0_512", ""))
        if origin0 is None or db_mean is None:
            return False
        axis_dir = np.array([-db_mean[1], db_mean[0]], dtype=np.float64)
        pts0_use = _reflect_points_about_axis(pts0_use, origin=origin0, axis_dir=axis_dir)

    shift = np.array([float(sx), float(sy)], dtype=np.float64)
    pts1_aligned = np.asarray(pts1, dtype=np.float64).reshape(-1, 2) - shift.reshape(1, 2)
    canvas = _as_rgb_canvas(img0_path, scale=float(out_scale))
    draw = ImageDraw.Draw(canvas)
    _draw_points(draw, pts0_use, color=(80, 160, 255), radius=2.6)
    _draw_points(draw, pts1_aligned, color=(255, 170, 60), radius=2.2)
    draw.text((8, 8), f"{cls}: layer1 - shift over layer0", fill=(255, 255, 255))
    canvas.save(out_path)
    return True


def _iter_sample_dirs(root: Path, limit: int) -> Iterable[Path]:
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if int(limit) > 0:
        dirs = dirs[: int(limit)]
    return dirs


def main() -> int:
    ap = argparse.ArgumentParser(description="Write ReS2 stacking-analysis debug overlays.")
    ap.add_argument("--root", required=True, help="Root directory containing bilayer sample subfolders.")
    ap.add_argument("--stacking_csv", required=True, help="CSV produced by classify_bilayers.py.")
    ap.add_argument("--atoms_dir", required=True, help="Atoms directory produced by classify_bilayers.py.")
    ap.add_argument("--interlayer_csv", default="", help="Optional CSV produced by analyze_interlayer.py.")
    ap.add_argument("--out_dir", required=True, help="Directory for debug PNG/JSON outputs.")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N subfolders; 0 means all.")
    ap.add_argument("--sideA_A", type=float, default=27.9, help="Fallback field-of-view side length in Angstrom when image paths do not contain a size tag such as 2.79x2.79.")
    ap.add_argument("--use_highpass", action="store_true", help="Use high-pass preprocessing in per-layer debug JSON.")
    ap.add_argument("--no_highpass", action="store_true", help="Disable high-pass preprocessing in per-layer debug JSON.")
    ap.add_argument("--bg_sigma", type=float, default=6.0)
    ap.add_argument("--bina_thre", type=float, default=1.8)
    ap.add_argument("--min_area", type=float, default=100.0)
    ap.add_argument("--min_dist", type=float, default=5.0)
    ap.add_argument("--max_points", type=int, default=4000)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stacking_rows = _read_csv_rows(Path(args.stacking_csv))
    interlayer_rows = _read_csv_rows(Path(args.interlayer_csv)) if str(args.interlayer_csv).strip() else {}
    img_to_json, out_scale = _load_atoms_index(Path(args.atoms_dir))

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

    summary: List[Dict[str, Any]] = []
    for folder in _iter_sample_dirs(root, int(args.limit)):
        row = stacking_rows.get(str(folder.as_posix()), {})
        inter_row = interlayer_rows.get(str(folder.as_posix()), {})
        img0 = next(iter(sorted(folder.glob("*_0.png"))), None)
        img1 = next(iter(sorted(folder.glob("*_1.png"))), None)
        if img0 is None or img1 is None:
            continue
        sample_dir = out_dir / folder.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        pts0 = _load_points(img_to_json.get(str(img0.as_posix())))
        pts1 = _load_points(img_to_json.get(str(img1.as_posix())))
        _write_layer_overlay(img_path=img0, pts512=pts0, row=row, layer_idx=0, out_path=sample_dir / f"{img0.stem}_atoms_lattice.png", out_scale=out_scale)
        _write_layer_overlay(img_path=img1, pts512=pts1, row=row, layer_idx=1, out_path=sample_dir / f"{img1.stem}_atoms_lattice.png", out_scale=out_scale)
        _write_layer_debug_json(img_path=img0, pts512=pts0, out_path=sample_dir / f"{img0.stem}_lattice_debug.json", out_scale=out_scale, atom_cfg=atom_cfg, sideA_A=float(args.sideA_A))
        _write_layer_debug_json(img_path=img1, pts512=pts1, out_path=sample_dir / f"{img1.stem}_lattice_debug.json", out_scale=out_scale, atom_cfg=atom_cfg, sideA_A=float(args.sideA_A))
        wrote_align = _write_alignment_overlay(
            img0_path=img0,
            pts0=pts0,
            pts1=pts1,
            stacking_row=row,
            interlayer_row=inter_row,
            out_path=sample_dir / "alignment_check.png",
            out_scale=out_scale,
        )
        summary.append(
            {
                "folder": str(folder.as_posix()),
                "class": row.get("class", ""),
                "reason": row.get("reason", ""),
                "n_atoms_layer0": int(pts0.shape[0]),
                "n_atoms_layer1": int(pts1.shape[0]),
                "alignment_check": bool(wrote_align),
            }
        )

    (out_dir / "debug_index.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"debug_out_dir={out_dir}")
    print(f"samples={len(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
