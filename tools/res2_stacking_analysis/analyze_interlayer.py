#!/usr/bin/env python3
"""
Purpose: Analyze interlayer geometry for folders of separated bilayer ReS2 outputs. It reuses or runs the ReS2 stacking classifier, then reports slip shifts, da/db coordinate projections, and fine twist angles in a CSV file.
Related files: tools/res2_stacking_analysis/classify_bilayers.py, tools/res2_stacking_analysis/single_layer_lattice.py, and README.md.
CLI usage: python tools/res2_stacking_analysis/analyze_interlayer.py --root outputs/ReS2 --out_csv outputs/res2_interlayer.csv (arguments: --root=input folders; --out_csv=analysis CSV; --force_stacking reruns stacking classification).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_STACKING_PIPELINE = _REPO_ROOT / "tools" / "res2_stacking_analysis" / "classify_bilayers.py"


def _parse_sideA_A_from_folder(name: str) -> Optional[float]:
    """Internal helper."""
    m = re.search(r"_([0-9]+(?:\.[0-9]+)?)x([0-9]+(?:\.[0-9]+)?)\s*$", name)
    if not m:
        return None
    a_nm = float(m.group(1))
    b_nm = float(m.group(2))
    return float(0.5 * (a_nm + b_nm) * 10.0)


def _cross_z(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    return float(a[0] * b[1] - a[1] * b[0])


def _angle_deg_mod180(u: np.ndarray, v: np.ndarray) -> float:
    """Internal helper."""
    u = np.asarray(u, dtype=np.float64).reshape(2)
    v = np.asarray(v, dtype=np.float64).reshape(2)
    nu = float(np.hypot(u[0], u[1]))
    nv = float(np.hypot(v[0], v[1]))
    if nu <= 1e-9 or nv <= 1e-9:
        return float("nan")
    c = float((u[0] * v[0] + u[1] * v[1]) / (nu * nv))
    c = abs(c)
    c = max(-1.0, min(1.0, c))
    return float(math.degrees(math.acos(c)))


def _angle_deg_directed_0_180(u: np.ndarray, v: np.ndarray) -> float:
    """Internal helper."""
    u = np.asarray(u, dtype=np.float64).reshape(2)
    v = np.asarray(v, dtype=np.float64).reshape(2)
    nu = float(np.hypot(u[0], u[1]))
    nv = float(np.hypot(v[0], v[1]))
    if nu <= 1e-9 or nv <= 1e-9:
        return float("nan")
    c = float((u[0] * v[0] + u[1] * v[1]) / (nu * nv))
    c = max(-1.0, min(1.0, c))
    return float(math.degrees(math.acos(c)))


def _trimmed_mean(values: np.ndarray, *, trim_ratio: float, min_keep: int = 120) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    if v.size == 0:
        return float("inf")
    tr = float(max(0.0, min(0.49, trim_ratio)))
    keep = int(max(min_keep, math.floor(float(v.size) * (1.0 - tr))))
    keep = int(min(keep, v.size))
    if keep <= 0:
        return float(np.mean(v))
    return float(np.mean(v[:keep]))


def _score_align_trimmed(tree_ref: KDTree, pts_mov: np.ndarray, *, trim_ratio: float) -> float:
    dists, _ = tree_ref.query(np.asarray(pts_mov, dtype=np.float64))
    return _trimmed_mean(dists, trim_ratio=float(trim_ratio))


def estimate_translation_multiscale_robust(
    tree_ref: KDTree,
    pts_mov: np.ndarray,
    *,
    origin: Tuple[int, int] = (0, 0),
    max_shift: int,
    coarse_step: int = 8,
    trim_ratio: float = 0.20,
) -> Optional[Tuple[np.ndarray, float]]:
    """Internal helper."""
    pts_mov = np.asarray(pts_mov, dtype=np.float32).reshape(-1, 2)
    if pts_mov.size == 0:
        return None

    def _search(dx0: int, dy0: int, step: int, radius: int) -> Tuple[Tuple[int, int], float]:
        best_xy: Optional[Tuple[int, int]] = None
        best_score: Optional[float] = None
        for dx in range(dx0 - radius, dx0 + radius + 1, step):
            for dy in range(dy0 - radius, dy0 + radius + 1, step):
                shifted = pts_mov + np.array([dx, dy], dtype=np.float32).reshape(1, 2)
                score = _score_align_trimmed(tree_ref, shifted, trim_ratio=float(trim_ratio))
                if best_score is None or score < best_score:
                    best_score = float(score)
                    best_xy = (int(dx), int(dy))
        assert best_xy is not None and best_score is not None
        return best_xy, float(best_score)

    coarse_step = int(max(1, coarse_step))
    max_shift = int(max(1, max_shift))
    ox, oy = int(origin[0]), int(origin[1])

    best_xy, best_score = _search(ox, oy, coarse_step, max_shift)
    fine_radius = max(2, coarse_step)
    best_xy, best_score = _search(best_xy[0], best_xy[1], 1, fine_radius)
    return np.array(best_xy, dtype=np.float32), float(best_score)


def _project_coeffs(da: np.ndarray, db: np.ndarray, vec: np.ndarray) -> Optional[np.ndarray]:
    da = np.asarray(da, dtype=np.float64).reshape(2)
    db = np.asarray(db, dtype=np.float64).reshape(2)
    v = np.asarray(vec, dtype=np.float64).reshape(2)
    P = np.stack([da, db], axis=1)  # 2x2 columns
    det = float(np.linalg.det(P))
    if abs(det) < 1e-9:
        return None
    c = np.linalg.solve(P, v.reshape(2, 1)).reshape(2)
    return c.astype(np.float64)


def _align_basis_sign_by_db(da: np.ndarray, db: np.ndarray, *, db_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Internal helper."""
    da = np.asarray(da, dtype=np.float64).reshape(2)
    db = np.asarray(db, dtype=np.float64).reshape(2)
    ref = np.asarray(db_ref, dtype=np.float64).reshape(2)
    if float(db @ ref) < 0.0:
        return (-da).astype(np.float64), (-db).astype(np.float64), -1
    return da.astype(np.float64), db.astype(np.float64), 1


def _canonicalize_basis_by_db_halfplane(da: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Internal helper."""
    da = np.asarray(da, dtype=np.float64).reshape(2)
    db = np.asarray(db, dtype=np.float64).reshape(2)
    if float(db[0]) < 0.0 or (abs(float(db[0])) <= 1e-12 and float(db[1]) < 0.0):
        return (-da).astype(np.float64), (-db).astype(np.float64), -1
    return da.astype(np.float64), db.astype(np.float64), 1


def _align_coeffs_mod1(c_ref: np.ndarray, c_other: np.ndarray) -> np.ndarray:
    """Internal helper."""
    c_ref = np.asarray(c_ref, dtype=np.float64).reshape(2)
    c_other = np.asarray(c_other, dtype=np.float64).reshape(2)
    delta = c_other - c_ref
    k = np.round(delta).astype(np.float64)
    return (c_other - k).astype(np.float64)


def _reduce_coeffs_res2(c: np.ndarray) -> np.ndarray:
    """Internal helper."""
    x = np.asarray(c, dtype=np.float64).reshape(2)
    x = x - np.floor(x)  # [0,1)
    if float(x[0] + x[1]) > 1.0:
        x = 1.0 - x
    x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float64)


def _displacement_vectors_from_points(
    pts: np.ndarray,
    *,
    k_neighbors: int,
    r_min_px: float,
    r_max_px: float,
) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < max(10, int(k_neighbors) + 1):
        return np.zeros((0, 2), dtype=np.float64)
    tree = KDTree(pts)
    _, idx = tree.query(pts, k=int(k_neighbors) + 1)
    V: List[np.ndarray] = []
    for i in range(pts.shape[0]):
        pi = pts[i]
        for j in idx[i, 1:]:
            pj = pts[int(j)]
            d = pj - pi
            r = float(np.hypot(float(d[0]), float(d[1])))
            if r < float(r_min_px) or r > float(r_max_px):
                continue
            V.append(d.reshape(1, 2))
    if not V:
        return np.zeros((0, 2), dtype=np.float64)
    vv = np.vstack(V).astype(np.float64)
    return vv


def _augment_vecs_by_sign(vecs: np.ndarray) -> np.ndarray:
    v = np.asarray(vecs, dtype=np.float64).reshape(-1, 2)
    if v.size == 0:
        return v.reshape(0, 2)
    return np.vstack([v, -v]).astype(np.float64)


def _rotate_vecs_deg(vecs_xy: np.ndarray, angle_deg: float) -> np.ndarray:
    a = math.radians(float(angle_deg))
    c = math.cos(a)
    s = math.sin(a)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(vecs_xy, dtype=np.float64) @ R.T


def estimate_twist_deg_from_vecs_search(
    vecs_ref: np.ndarray,
    vecs_mov: np.ndarray,
    *,
    center_deg: Optional[float] = None,
    radius_deg: Optional[float] = None,
    coarse_step_deg: float,
    trim_ratio: float,
) -> Optional[Tuple[float, float]]:
    """Internal helper."""
    ref = np.asarray(vecs_ref, dtype=np.float64).reshape(-1, 2)
    mov = np.asarray(vecs_mov, dtype=np.float64).reshape(-1, 2)
    if ref.shape[0] < 80 or mov.shape[0] < 80:
        return None
    step0 = float(coarse_step_deg)
    if not np.isfinite(step0) or step0 <= 0.0:
        return None

    tree_ref = KDTree(ref)
    best_deg = 0.0
    best_score = float("inf")

    if center_deg is None or radius_deg is None or not np.isfinite(float(center_deg)) or not np.isfinite(float(radius_deg)):
        deg = 0.0
        while deg < 180.0 - 1e-12:
            dists, _ = tree_ref.query(_rotate_vecs_deg(mov, deg))
            s = _trimmed_mean(dists, trim_ratio=float(trim_ratio), min_keep=200)
            if s < best_score:
                best_score = float(s)
                best_deg = float(deg)
            deg += step0
    else:
        c0 = float(center_deg) % 180.0
        rad = float(max(0.0, min(90.0, float(radius_deg))))
        start = c0 - rad
        end = c0 + rad
        deg = start
        while deg <= end + 1e-12:
            d = float(deg) % 180.0
            dists, _ = tree_ref.query(_rotate_vecs_deg(mov, d))
            s = _trimmed_mean(dists, trim_ratio=float(trim_ratio), min_keep=200)
            if s < best_score:
                best_score = float(s)
                best_deg = float(d)
            deg += step0

    for step, radius in [(0.2, 2.0), (0.02, 0.4), (0.005, 0.1)]:
        start = best_deg - radius
        end = best_deg + radius
        deg = start
        while deg <= end + 1e-12:
            d = float(deg) % 180.0
            dists, _ = tree_ref.query(_rotate_vecs_deg(mov, d))
            s = _trimmed_mean(dists, trim_ratio=float(trim_ratio), min_keep=200)
            if s < best_score:
                best_score = float(s)
                best_deg = float(d)
            deg += step

    return float(best_deg % 180.0), float(best_score)


@dataclass(frozen=True)
class PredLayer:
    points_px: np.ndarray  # (N,2) in 512 coord
    origin_px: np.ndarray  # (2,) in 512 coord
    da_px: np.ndarray  # (2,) in 512 coord
    db_px: np.ndarray  # (2,) in 512 coord


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return (v / n).astype(np.float64)


def _reflect_vec_about_axis(v: np.ndarray, axis_dir: np.ndarray) -> np.ndarray:
    """Internal helper."""
    vv = np.asarray(v, dtype=np.float64).reshape(2)
    n = _unit(np.asarray(axis_dir, dtype=np.float64).reshape(2))
    return (2.0 * float(np.dot(vv, n)) * n - vv).astype(np.float64)


def _reflect_points_about_axis(points: np.ndarray, *, origin: np.ndarray, axis_dir: np.ndarray) -> np.ndarray:
    """Internal helper."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    o = np.asarray(origin, dtype=np.float64).reshape(2)
    n = _unit(np.asarray(axis_dir, dtype=np.float64).reshape(2))
    v = P - o.reshape(1, 2)
    proj = (v @ n.reshape(2, 1)).reshape(-1, 1)
    v_ref = 2.0 * proj * n.reshape(1, 2) - v
    return (o.reshape(1, 2) + v_ref).astype(np.float64)


def _default_stacking_paths(out_csv: Path) -> Tuple[Path, Path]:
    stacking_csv = out_csv.with_name(f"{out_csv.stem}__res2_stacking.csv")
    stacking_atoms_dir = out_csv.with_name(f"{out_csv.stem}__res2_stacking_atoms")
    return stacking_csv, stacking_atoms_dir


def _run_stacking_pipeline_if_needed(
    *,
    root: Path,
    stacking_csv: Path,
    stacking_atoms_dir: Path,
    limit: int,
    slip_thresh_deg: float,
    out_scale: float,
    force_stacking: bool,
    use_highpass: Optional[bool],
    bg_sigma: float,
    bina_thre: float,
    min_area: float,
    min_dist: float,
    max_points: int,
) -> None:
    stacking_csv = Path(stacking_csv)
    stacking_atoms_dir = Path(stacking_atoms_dir)
    idx = stacking_atoms_dir / "atoms_index.json"
    if (not bool(force_stacking)) and stacking_csv.exists() and idx.exists():
        return

    cmd = [
        sys.executable,
        str(_STACKING_PIPELINE),
        "--root",
        str(root),
        "--out_csv",
        str(stacking_csv),
        "--atoms_out_dir",
        str(stacking_atoms_dir),
        "--slip_thresh_deg",
        str(float(slip_thresh_deg)),
        "--out_scale",
        str(float(out_scale)),
        "--bg_sigma",
        str(float(bg_sigma)),
        "--bina_thre",
        str(float(bina_thre)),
        "--min_area",
        str(float(min_area)),
        "--min_dist",
        str(float(min_dist)),
        "--max_points",
        str(int(max_points)),
    ]
    if int(limit) > 0:
        cmd.extend(["--limit", str(int(limit))])
    if use_highpass is True:
        cmd.append("--use_highpass")
    elif use_highpass is False:
        cmd.append("--no_highpass")

    subprocess.run(cmd, check=True)


def _load_stacking_outputs(stacking_csv: Path, stacking_atoms_dir: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Path], float]:
    stacking_rows: Dict[str, Dict[str, str]] = {}
    with stacking_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            stacking_rows[str(r.get("folder", ""))] = dict(r)

    atoms_index_path = stacking_atoms_dir / "atoms_index.json"
    atoms_index = json.loads(atoms_index_path.read_text(encoding="utf-8"))
    img2json: Dict[str, Path] = {}
    out_scale = None
    for it in atoms_index:
        img = str(it.get("image", ""))
        fn = str(it.get("atoms_json", ""))
        if img and fn:
            img2json[img] = stacking_atoms_dir / fn
        if out_scale is None and it.get("out_scale") is not None:
            try:
                out_scale = float(it["out_scale"])
            except Exception:
                out_scale = None
    if out_scale is None:
        out_scale = 4.0
    return stacking_rows, img2json, float(out_scale)


def _load_atoms_points_from_json(atoms_json: Path) -> np.ndarray:
    data = json.loads(Path(atoms_json).read_text(encoding="utf-8"))
    pts = np.array([[float(d["x"]), float(d["y"])] for d in data], dtype=np.float64).reshape(-1, 2)
    return pts


def _parse_vec_field(row: Dict[str, str], key: str) -> Optional[np.ndarray]:
    s = str(row.get(key, "")).strip()
    if not s:
        return None
    try:
        v = json.loads(s)
    except Exception:
        return None
    if not isinstance(v, list) or len(v) != 2:
        return None
    return np.array([float(v[0]), float(v[1])], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing bilayer sample subfolders.")
    ap.add_argument("--out_csv", required=True, help="Output CSV path for interlayer analysis.")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N subfolders; 0 means all.")

    # res2_stacking reuse / cache
    ap.add_argument("--force_stacking", action="store_true", help="Force rerunning the ReS2 stacking classifier instead of reusing existing outputs.")
    ap.add_argument("--stacking_csv", default="", help="Existing or target res2_stacking CSV path. Inferred from out_csv by default.")
    ap.add_argument("--stacking_atoms_dir", default="", help="res2_stacking atoms directory; inferred from out_csv by default")
    ap.add_argument("--stacking_out_scale", type=float, default=4.0, help="Coordinate scale for res2_stacking outputs. The default maps 128 px to 512 px.")

    # atom detect params passed through to the stacking classifier
    ap.add_argument("--use_highpass", action="store_true", help="Force high-pass preprocessing")
    ap.add_argument("--no_highpass", action="store_true", help="Disable high-pass preprocessing")
    ap.add_argument("--bg_sigma", type=float, default=6.0)
    ap.add_argument("--bina_thre", type=float, default=1.8)
    ap.add_argument("--min_area", type=float, default=100.0)
    ap.add_argument("--min_dist", type=float, default=5.0)
    ap.add_argument("--max_points", type=int, default=4000)

    # slip/twist rule (kept)
    ap.add_argument("--slip_thresh_deg", type=float, default=5.0)
    ap.add_argument("--sideA_A", type=float, default=27.9, help="Default field-of-view side length in Angstrom when it cannot be parsed from folder names")

    # slip translation search (512 coord)
    ap.add_argument("--slip_max_shift_px", type=int, default=48, help="Maximum slip-search shift in 512-coordinate pixels")
    ap.add_argument("--slip_coarse_step", type=int, default=8)
    ap.add_argument("--slip_trim_ratio", type=float, default=0.20)

    # twist (512 coord)
    ap.add_argument("--twist_k_neighbors", type=int, default=12)
    ap.add_argument("--twist_r_min_px", type=float, default=8.0, help="Minimum displacement-vector radius in 512-coordinate pixels")
    ap.add_argument("--twist_r_max_px", type=float, default=160.0, help="Maximum displacement-vector radius in 512-coordinate pixels")
    ap.add_argument("--twist_trim_ratio", type=float, default=0.20)
    ap.add_argument("--twist_coarse_step_deg", type=float, default=1.0)
    ap.add_argument(
        "--twist_search_radius_deg",
        type=float,
        default=30.0,
        help="Half-width of the fine twist-angle search around the coarse angle, in degrees.",
    )
    ap.add_argument(
        "--twist_augment_sign",
        action="store_true",
        default=False,
        help="Also evaluate the opposite twist-angle sign during fine search.",
    )
    args = ap.parse_args()

    root = Path(str(args.root))
    if not root.exists():
        raise FileNotFoundError(str(root))

    out_csv = Path(str(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    stacking_csv_arg = str(args.stacking_csv).strip()
    stacking_atoms_dir_arg = str(args.stacking_atoms_dir).strip()
    stacking_csv, stacking_atoms_dir = _default_stacking_paths(out_csv)
    if stacking_csv_arg:
        stacking_csv = Path(stacking_csv_arg)
    if stacking_atoms_dir_arg:
        stacking_atoms_dir = Path(stacking_atoms_dir_arg)

    use_hp: Optional[bool] = None
    if bool(args.no_highpass):
        use_hp = False
    if bool(args.use_highpass):
        use_hp = True

    _run_stacking_pipeline_if_needed(
        root=root,
        stacking_csv=stacking_csv,
        stacking_atoms_dir=stacking_atoms_dir,
        limit=int(args.limit),
        slip_thresh_deg=float(args.slip_thresh_deg),
        out_scale=float(args.stacking_out_scale),
        force_stacking=bool(args.force_stacking),
        use_highpass=use_hp,
        bg_sigma=float(args.bg_sigma),
        bina_thre=float(args.bina_thre),
        min_area=float(args.min_area),
        min_dist=float(args.min_dist),
        max_points=int(args.max_points),
    )

    stacking_rows, img2json, stacking_out_scale = _load_stacking_outputs(stacking_csv, stacking_atoms_dir)

    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if int(args.limit) > 0:
        dirs = dirs[: int(args.limit)]

    rows: List[Dict[str, str]] = []
    n_flip_twist = n_flip_slip = n_slip = n_twist = n_unknown = 0

    for d in dirs:
        folder_key = str(d.as_posix())
        pr = stacking_rows.get(folder_key)
        if pr is None:
            continue

        cls = str(pr.get("class", "unknown"))
        is_flip = str(pr.get("is_flip", ""))
        reason = str(pr.get("reason", ""))
        coarse_twist = str(pr.get("theta_deg", ""))
        theta_method = str(pr.get("theta_method", ""))

        img0 = d / str(pr.get("img0", ""))
        img1 = d / str(pr.get("img1", ""))
        if (not img0.exists()) or (not img1.exists()):
            cls = "unknown"
            reason = "missing__0_or__1"

        origin0 = _parse_vec_field(pr, "origin0_512")
        origin1 = _parse_vec_field(pr, "origin1_512")
        da0 = _parse_vec_field(pr, "da0_512")
        db0 = _parse_vec_field(pr, "db0_512")
        da1 = _parse_vec_field(pr, "da1_512")
        db1 = _parse_vec_field(pr, "db1_512")
        db_mean = _parse_vec_field(pr, "db_mean_512")

        pts0 = pts1 = None
        if img0.exists():
            fn0 = img2json.get(str(img0.as_posix()))
            if fn0 is not None and fn0.exists():
                pts0 = _load_atoms_points_from_json(fn0)
        if img1.exists():
            fn1 = img2json.get(str(img1.as_posix()))
            if fn1 is not None and fn1.exists():
                pts1 = _load_atoms_points_from_json(fn1)

        # tolerate old caches that stored 128 coords
        def _maybe_scale_pts(pts: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if pts is None or pts.size == 0:
                return pts
            m = float(np.max(pts))
            if m <= 140.0 and float(stacking_out_scale) > 1.0:
                return (pts * float(stacking_out_scale)).astype(np.float64)
            return pts.astype(np.float64)

        pts0 = _maybe_scale_pts(pts0)
        pts1 = _maybe_scale_pts(pts1)

        sideA = _parse_sideA_A_from_folder(d.name)
        sideA_A = float(args.sideA_A) if sideA is None else float(sideA)
        A_per_px = float(sideA_A) / 512.0

        twist_fine = ""
        twist_score = ""
        shift_px_x = shift_px_y = ""
        shift_A_x = shift_A_y = ""
        shift_align_score = ""
        ab0_raw_a = ab0_raw_b = ""
        ab1_raw_a = ab1_raw_b = ""
        ab_mean_raw_a = ab_mean_raw_b = ""
        ab_mean_red_a = ab_mean_red_b = ""

        ok_for_downstream = (
            origin0 is not None
            and origin1 is not None
            and da0 is not None
            and db0 is not None
            and da1 is not None
            and db1 is not None
            and pts0 is not None
            and pts1 is not None
            and pts0.shape[0] >= 10
            and pts1.shape[0] >= 10
        )

        if cls == "unknown" or (not ok_for_downstream):
            n_unknown += 1
        elif cls == "flip_twist":
            n_flip_twist += 1
        elif cls == "twist":
            n_twist += 1

            _da0c, db0c, _ = _canonicalize_basis_by_db_halfplane(da0, db0)
            _da1c, db1c, _ = _canonicalize_basis_by_db_halfplane(da1, db1)
            theta_db_dir = _angle_deg_directed_0_180(db0c, db1c)
            center = None
            radius = None
            if np.isfinite(theta_db_dir) and float(args.twist_search_radius_deg) > 0.0:
                center = float(theta_db_dir)
                radius = float(args.twist_search_radius_deg)

            v0 = _displacement_vectors_from_points(
                pts0,
                k_neighbors=int(args.twist_k_neighbors),
                r_min_px=float(args.twist_r_min_px),
                r_max_px=float(args.twist_r_max_px),
            )
            v1 = _displacement_vectors_from_points(
                pts1,
                k_neighbors=int(args.twist_k_neighbors),
                r_min_px=float(args.twist_r_min_px),
                r_max_px=float(args.twist_r_max_px),
            )
            if bool(args.twist_augment_sign):
                v0 = _augment_vecs_by_sign(v0)
                v1 = _augment_vecs_by_sign(v1)

            out_tw = estimate_twist_deg_from_vecs_search(
                v0,
                v1,
                center_deg=center,
                radius_deg=radius,
                coarse_step_deg=float(args.twist_coarse_step_deg),
                trim_ratio=float(args.twist_trim_ratio),
            )
            if out_tw is None:
                cls = "unknown"
                reason = "twist_failed"
                n_unknown += 1
                n_twist -= 1
            else:
                tw_deg, tw_s = out_tw
                twist_fine = f"{float(tw_deg):.6f}"
                twist_score = f"{float(tw_s):.6f}"

        elif cls in ("slip", "flip_slip"):
            if cls == "flip_slip":
                n_flip_slip += 1


                axis_db = db_mean if db_mean is not None else db0
                axis_dir = np.array([-axis_db[1], axis_db[0]], dtype=np.float64)  # ⟂ db_mean
                pts0_use = _reflect_points_about_axis(pts0, origin=origin0, axis_dir=axis_dir)
                da0_use = _reflect_vec_about_axis(da0, axis_dir)
                db0_use = _reflect_vec_about_axis(db0, axis_dir)
            else:
                n_slip += 1
                pts0_use = pts0
                da0_use = da0
                db0_use = db0

            da1_al, db1_al, _s1 = _align_basis_sign_by_db(da1, db1, db_ref=db0_use)
            tree0 = KDTree(np.asarray(pts0_use, dtype=np.float64))
            t_out = estimate_translation_multiscale_robust(
                tree0,
                np.asarray(pts1, dtype=np.float64),
                origin=(0, 0),
                max_shift=int(args.slip_max_shift_px),
                coarse_step=int(args.slip_coarse_step),
                trim_ratio=float(args.slip_trim_ratio),
            )
            if t_out is None:
                cls = "unknown"
                reason = "shift_failed"
                n_unknown += 1
                if cls == "flip_slip":
                    n_flip_slip -= 1
                else:
                    n_slip -= 1
            else:
                t_align, score = t_out
                shift_px = (-t_align.astype(np.float64)).reshape(2)
                shift_px_x = f"{float(shift_px[0]):.3f}"
                shift_px_y = f"{float(shift_px[1]):.3f}"
                shift_A_x = f"{float(shift_px[0] * A_per_px):.4f}"
                shift_A_y = f"{float(shift_px[1] * A_per_px):.4f}"
                shift_align_score = f"{float(score):.6f}"

                best_pack = None
                for cand_shift in [shift_px, -shift_px]:
                    c0 = _project_coeffs(da0_use, db0_use, cand_shift)
                    c1 = _project_coeffs(da1_al, db1_al, cand_shift)
                    if c0 is None or c1 is None:
                        continue
                    c1_adj = _align_coeffs_mod1(c0, c1)
                    c_mean = 0.5 * (c0 + c1_adj)
                    c_red = _reduce_coeffs_res2(c_mean)
                    norm = float(np.sum(c_red**2))
                    pack = (norm, cand_shift, c0, c1_adj, c_mean, c_red)
                    if best_pack is None or pack[0] < best_pack[0]:
                        best_pack = pack
                if best_pack is None:
                    cls = "unknown"
                    reason = "project_failed"
                    n_unknown += 1
                    if cls == "flip_slip":
                        n_flip_slip -= 1
                    else:
                        n_slip -= 1
                else:
                    _norm, best_shift, c0, c1_adj, c_mean, c_red = best_pack
                    shift_px_x = f"{float(best_shift[0]):.3f}"
                    shift_px_y = f"{float(best_shift[1]):.3f}"
                    shift_A_x = f"{float(best_shift[0] * A_per_px):.4f}"
                    shift_A_y = f"{float(best_shift[1] * A_per_px):.4f}"
                    ab0_raw_a = f"{float(c0[0]):.6f}"
                    ab0_raw_b = f"{float(c0[1]):.6f}"
                    ab1_raw_a = f"{float(c1_adj[0]):.6f}"
                    ab1_raw_b = f"{float(c1_adj[1]):.6f}"
                    ab_mean_raw_a = f"{float(c_mean[0]):.6f}"
                    ab_mean_raw_b = f"{float(c_mean[1]):.6f}"
                    ab_mean_red_a = f"{float(c_red[0]):.6f}"
                    ab_mean_red_b = f"{float(c_red[1]):.6f}"

        else:
            n_unknown += 1

        row = {
            "folder": str(d),
            "img0": "" if not img0.exists() else img0.name,
            "img1": "" if not img1.exists() else img1.name,
            "sideA_A": f"{sideA_A:.4f}",
            "stacking_csv": str(stacking_csv.as_posix()),
            "stacking_atoms_dir": str(stacking_atoms_dir.as_posix()),
            "stacking_theta_deg": coarse_twist,
            "stacking_theta_method": theta_method,
            "class": cls,
            "reason": reason,
            "is_flip": is_flip,
            "n0": "" if pts0 is None else str(int(pts0.shape[0])),
            "n1": "" if pts1 is None else str(int(pts1.shape[0])),
            "da0_x": "" if da0 is None else f"{float(da0[0]):.6f}",
            "da0_y": "" if da0 is None else f"{float(da0[1]):.6f}",
            "db0_x": "" if db0 is None else f"{float(db0[0]):.6f}",
            "db0_y": "" if db0 is None else f"{float(db0[1]):.6f}",
            "da1_x": "" if da1 is None else f"{float(da1[0]):.6f}",
            "da1_y": "" if da1 is None else f"{float(da1[1]):.6f}",
            "db1_x": "" if db1 is None else f"{float(db1[0]):.6f}",
            "db1_y": "" if db1 is None else f"{float(db1[1]):.6f}",
            # slip
            "shift_px_x": shift_px_x,
            "shift_px_y": shift_px_y,
            "shift_A_x": shift_A_x,
            "shift_A_y": shift_A_y,
            "shift_align_score": shift_align_score,
            "ab0_raw_a": ab0_raw_a,
            "ab0_raw_b": ab0_raw_b,
            "ab1_raw_a": ab1_raw_a,
            "ab1_raw_b": ab1_raw_b,
            "ab_mean_raw_a": ab_mean_raw_a,
            "ab_mean_raw_b": ab_mean_raw_b,
            "ab_mean_reduced_a": ab_mean_red_a,
            "ab_mean_reduced_b": ab_mean_red_b,
            # twist
            "twist_fine_deg": twist_fine,
            "twist_score": twist_score,
        }
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(
        f"done: folders={len(rows)} slip={n_slip} twist={n_twist} flip_slip={n_flip_slip} flip_twist={n_flip_twist} unknown={n_unknown} out_csv={out_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
