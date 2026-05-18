"""
Purpose: Estimate Re-chain directions from Re4-cycle centroids and infer da/db pixel vectors from centroid-neighbor geometry. It returns direction vectors and diagnostic statistics for downstream classification.
Related files: tools/pred_dadb/cycles.py, tools/pred_dadb/pipeline_single.py, and tools/pred_dadb/pipeline_bilayer_root.py.
CLI usage: This module is imported by the da/db pipelines and is not intended to be executed directly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


def _axis_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Internal helper."""
    uu = np.asarray(u, dtype=np.float64).reshape(2)
    vv = np.asarray(v, dtype=np.float64).reshape(2)
    nu = float(np.hypot(uu[0], uu[1]))
    nv = float(np.hypot(vv[0], vv[1]))
    if nu < 1e-12 or nv < 1e-12:
        return 90.0
    c = float(np.dot(uu, vv) / (nu * nv))
    c = abs(max(-1.0, min(1.0, c)))  # axial
    return float(np.degrees(np.arccos(c)))


def _kmeans_1d_two_clusters(x: np.ndarray, *, n_iter: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Internal helper."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros((0,), dtype=np.int64), np.array([0.0, 0.0], dtype=np.float64)
    if x.size == 1:
        return np.zeros((1,), dtype=np.int64), np.array([float(x[0]), float(x[0])], dtype=np.float64)


    c0 = float(np.quantile(x, 0.20))
    c1 = float(np.quantile(x, 0.80))
    if abs(c1 - c0) < 1e-9:
        c1 = float(np.max(x))
    centers = np.array([c0, c1], dtype=np.float64)

    labels = np.zeros((x.size,), dtype=np.int64)
    for _ in range(int(n_iter)):
        d0 = np.abs(x - centers[0])
        d1 = np.abs(x - centers[1])
        new_labels = (d1 < d0).astype(np.int64)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in (0, 1):
            m = x[labels == k]
            if m.size:
                centers[k] = float(np.mean(m))
    return labels, centers


def _select_two_main_axes(candidates_deg: Sequence[float], strengths: Optional[Sequence[Tuple[float, int]]] = None) -> List[float]:
    """Internal helper."""
    cand = [float(a) for a in candidates_deg]
    if not cand:
        return []
    wmap: Dict[float, float] = {}
    if strengths:
        for a, w in strengths:
            wmap[float(a)] = float(w)

    def w(a: float) -> float:
        return float(wmap.get(float(a), 1.0))

    cand_sorted = sorted(cand, key=lambda a: w(a), reverse=True)
    a1 = cand_sorted[0]

    best = None
    best_key = None
    for a2 in cand_sorted[1:]:
        sep = abs(a2 - a1)
        sep = min(sep, 180.0 - sep)
        if sep < 20.0:
            continue
        key = (abs(sep - 60.0), -w(a2))
        if best_key is None or key < best_key:
            best_key = key
            best = a2
    if best is None:
        best = cand_sorted[1] if len(cand_sorted) >= 2 else None
    return [float(a1)] + ([float(best)] if best is not None else [])


def _select_topk_candidate_axes(
    candidates_deg: Sequence[float],
    *,
    strengths: Optional[Sequence[Tuple[float, int]]] = None,
    k: int = 3,
    min_sep_deg: float = 10.0,
) -> List[float]:
    """Internal helper."""
    cand = [float(a) for a in candidates_deg]
    if not cand:
        return []

    wmap: Dict[float, float] = {}
    if strengths:
        for a, w in strengths:
            wmap[float(a)] = float(w)

    def w(a: float) -> float:
        return float(wmap.get(float(a), 1.0))

    order = sorted(cand, key=lambda a: w(a), reverse=True) if strengths else list(cand)

    picked: List[float] = []
    for a in order:
        if any(min(abs(a - b), 180.0 - abs(a - b)) < float(min_sep_deg) for b in picked):
            continue
        picked.append(float(a))
        if len(picked) >= int(max(1, k)):
            break
    return picked


def _drop_least_parallel_to_seed_edges(
    cand_angles_deg: Sequence[float],
    *,
    seed_edge_dirs: Tuple[np.ndarray, np.ndarray],
) -> Tuple[List[float], Optional[float], Dict]:
    """Internal helper."""
    cand = [float(a) for a in cand_angles_deg]
    if len(cand) <= 2:
        return list(cand), None, {"reason": "no_need_drop", "n": int(len(cand))}

    e0 = np.asarray(seed_edge_dirs[0], dtype=np.float64).reshape(2)
    e1 = np.asarray(seed_edge_dirs[1], dtype=np.float64).reshape(2)
    if float(np.hypot(e0[0], e0[1])) < 1e-9 or float(np.hypot(e1[0], e1[1])) < 1e-9:
        return list(cand), None, {"reason": "bad_seed_edges"}

    scores = []
    for a in cand:
        r = float(np.radians(float(a)))
        v = np.array([float(np.cos(r)), float(np.sin(r))], dtype=np.float64)
        s0 = float(_axis_angle_deg(v, e0))
        s1 = float(_axis_angle_deg(v, e1))
        scores.append((float(min(s0, s1)), float(s0), float(s1)))

    worst_i = int(np.argmax([s[0] for s in scores]))
    dropped = float(cand[worst_i])
    kept = [float(a) for i, a in enumerate(cand) if i != worst_i]
    dbg = {
        "reason": "drop_least_parallel",
        "cand_angles_deg": [float(a) for a in cand],
        "scores_min_s0_s1": [{"min": float(m), "to_e0": float(s0), "to_e1": float(s1)} for (m, s0, s1) in scores],
        "dropped_angle_deg": float(dropped),
        "kept_angles_deg": [float(a) for a in kept],
    }
    return kept, dropped, dbg


def _pick_candidate_angles_from_centroids(
    centroids: np.ndarray,
    *,
    k_neighbors: int = 6,
    max_candidates: int = 6,
    min_sep_deg: float = 10.0,
) -> Tuple[List[float], Dict]:
    """Internal helper."""
    C = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
    if C.shape[0] < 2:
        return [], {"reason": "too_few_centroids"}

    if C.shape[0] < 4:
        vecs = []
        for i in range(int(C.shape[0])):
            for j in range(i + 1, int(C.shape[0])):
                v = (C[int(j)] - C[int(i)]).astype(np.float64)
                if float(np.hypot(v[0], v[1])) < 1e-9:
                    continue
                vecs.append(v)
        if not vecs:
            return [], {"reason": "no_vectors_small_n"}
        vecs = np.stack(vecs, axis=0)
        vecs_short = vecs
        keep = np.ones((vecs.shape[0],), dtype=bool)
        centers_len = np.asarray([], dtype=np.float64)
        k_short = 0
    else:
        k = int(max(2, min(k_neighbors + 1, C.shape[0])))
        tree = cKDTree(C)
        _d, idx = tree.query(C, k=k)

        vecs = []
        for i in range(C.shape[0]):
            for j in idx[i, 1:]:
                v = (C[int(j)] - C[int(i)]).astype(np.float64)
                if float(np.hypot(v[0], v[1])) < 1e-9:
                    continue
                vecs.append(v)
        if not vecs:
            return [], {"reason": "no_vectors"}

        vecs = np.stack(vecs, axis=0)
        lens = np.hypot(vecs[:, 0], vecs[:, 1]).astype(np.float64)
        labels_len, centers_len = _kmeans_1d_two_clusters(lens)
        k_short = int(np.argmin(centers_len))
        keep = labels_len == k_short
        vecs_short = vecs[keep]
        if vecs_short.shape[0] < max(8, int(0.25 * vecs.shape[0])):
            vecs_short = vecs
            keep = np.ones((vecs.shape[0],), dtype=bool)

    ang = (np.degrees(np.arctan2(vecs_short[:, 1], vecs_short[:, 0])) % 180.0).astype(np.float64)
    hist, edges = np.histogram(ang, bins=180, range=(0.0, 180.0))

    peaks, _props = find_peaks(hist.astype(np.float64), prominence=max(2.0, float(np.max(hist)) * 0.05))
    peak_bins = [int(p) for p in np.asarray(peaks).tolist()]
    top_bins = [int(x) for x in np.argsort(hist)[::-1][: max(12, int(max_candidates) * 2)]]

    merged = list(dict.fromkeys(peak_bins + top_bins))
    merged.sort(key=lambda b: int(hist[int(b)]), reverse=True)

    picked: List[float] = []
    for b in merged:
        a = float((edges[int(b)] + edges[int(b) + 1]) * 0.5)
        if any(min(abs(a - bb), 180.0 - abs(a - bb)) < float(min_sep_deg) for bb in picked):
            continue
        picked.append(a)
        if len(picked) >= int(max_candidates):
            break

    angle_strengths = []
    for a in picked:
        b = int(np.clip(np.searchsorted(edges, a, side="right") - 1, 0, len(hist) - 1))
        angle_strengths.append((float(a), int(hist[b])))

    dbg = {
        "n_centroids": int(C.shape[0]),
        "k_neighbors": int(k_neighbors),
        "n_vectors": int(vecs.shape[0]),
        "n_vectors_short": int(vecs_short.shape[0]),
        "len_centers": [float(centers_len[0]), float(centers_len[1])] if centers_len.size >= 2 else None,
        "len_short_center": float(centers_len[k_short]) if centers_len.size else None,
        "len_short_keep_ratio": float(np.mean(keep)) if keep.size else None,
        "picked_angles_deg": [float(x) for x in picked],
        "picked_angle_strengths": angle_strengths,
    }
    return picked, dbg


def _neighbor_edges_for_two_axes(
    centroids: np.ndarray,
    *,
    ang0_deg: float,
    ang1_deg: float,
    k_neighbors: int = 6,
    len_ref: str = "nn_min",
    min_len_factor: float = 0.6,
    max_len_factor: float = 1.6,
    angle_sim_tol_deg: float = 18.0,
    min_edges: int = 3,
) -> Tuple[List[float], List[float], Dict]:
    """Internal helper."""
    C = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
    if C.shape[0] < 2:
        return [], [], {"reason": "too_few_centroids"}

    tree = cKDTree(C)
    d, idx = tree.query(C, k=min(int(max(2, k_neighbors + 1)), int(C.shape[0])))
    nn_med = float(np.median(d[:, 1])) if d.ndim == 2 and d.shape[1] >= 2 else float("nan")
    nn_min = float(np.min(d[:, 1])) if d.ndim == 2 and d.shape[1] >= 2 else float("nan")
    if not np.isfinite(nn_min) or nn_min <= 0:
        return [], [], {"reason": "bad_nn_med"}

    lr = str(len_ref).strip().lower()
    if lr in ("nn_med", "med", "median"):
        if not np.isfinite(nn_med) or nn_med <= 0:
            return [], [], {"reason": "bad_nn_med"}
        lo = float(min_len_factor) * float(nn_med)
        hi = float(max_len_factor) * float(nn_med)
        len_ref_used = "nn_med"
    else:
        lo = float(nn_min)
        hi = float(max_len_factor) * float(nn_min)
        len_ref_used = "nn_min"

    edge_set = set()
    for i in range(int(C.shape[0])):
        neigh = idx[i, 1:] if idx.ndim == 2 else []
        for j in neigh:
            a = int(i)
            b = int(j)
            if a == b:
                continue
            edge_set.add((min(a, b), max(a, b)))

    edges_lenok: List[Tuple[int, int, float]] = []
    for a, b in sorted(edge_set):
        v = (C[int(b)] - C[int(a)]).astype(np.float64)
        L = float(np.hypot(v[0], v[1]))
        if L < lo or L > hi:
            continue
        edges_lenok.append((int(a), int(b), float(L)))

    r0 = float(np.radians(float(ang0_deg)))
    r1 = float(np.radians(float(ang1_deg)))
    u0 = np.array([float(np.cos(r0)), float(np.sin(r0))], dtype=np.float64)
    u1 = np.array([float(np.cos(r1)), float(np.sin(r1))], dtype=np.float64)

    tol = float(max(1.0, angle_sim_tol_deg))
    lens0: List[float] = []
    lens1: List[float] = []
    picked0: Dict[Tuple[int, int], int] = {}
    picked1: Dict[Tuple[int, int], int] = {}

    n_angle_reject = 0
    for a, b, L in edges_lenok:
        v = (C[int(b)] - C[int(a)]).astype(np.float64)
        a0 = float(_axis_angle_deg(v, u0))
        a1 = float(_axis_angle_deg(v, u1))
        m = float(min(a0, a1))
        if m > tol:
            n_angle_reject += 1
            continue
        if a0 <= a1:
            lens0.append(float(L))
            picked0[(int(a), int(b))] = int(picked0.get((int(a), int(b)), 0) + 1)
        else:
            lens1.append(float(L))
            picked1[(int(a), int(b))] = int(picked1.get((int(a), int(b)), 0) + 1)

    dbg = {
        "method": "knn_edges_length_filter_and_parallel_cluster",
        "n_centroids": int(C.shape[0]),
        "k_neighbors": int(k_neighbors),
        "nn_med": float(nn_med),
        "nn_min": float(nn_min),
        "len_ref": str(len_ref_used),
        "len_lo": float(lo),
        "len_hi": float(hi),
        "n_edges_lenok": int(len(edges_lenok)),
        "edges_kept": [[int(a), int(b)] for (a, b, _) in edges_lenok],
        "angle_sim_tol_deg": float(tol),
        "n_edges_raw": int(len(edge_set)),
        "n_edges_angle_reject": int(n_angle_reject),
        "lens0_n": int(len(lens0)),
        "lens1_n": int(len(lens1)),
        "picked_edges0": {f"{x}-{y}": int(c) for (x, y), c in picked0.items()},
        "picked_edges1": {f"{x}-{y}": int(c) for (x, y), c in picked1.items()},
        "min_edges": int(min_edges),
    }
    return lens0, lens1, dbg


def _choose_by_neighbor_min_edge(
    centroids: np.ndarray,
    *,
    ang0_deg: float,
    ang1_deg: float,
    k_neighbors: int = 6,
    len_ref: str = "nn_min",
    min_len_factor: float = 0.6,
    max_len_factor: float = 1.6,
    angle_sim_tol_deg: float = 18.0,
    min_edges: int = 3,
) -> Tuple[Optional[float], Dict]:
    """Internal helper."""
    l0, l1, dbg0 = _neighbor_edges_for_two_axes(
        centroids,
        ang0_deg=float(ang0_deg),
        ang1_deg=float(ang1_deg),
        k_neighbors=int(k_neighbors),
        len_ref=str(len_ref),
        min_len_factor=float(min_len_factor),
        max_len_factor=float(max_len_factor),
        angle_sim_tol_deg=float(angle_sim_tol_deg),
        min_edges=int(min_edges),
    )
    if len(l0) < int(min_edges) or len(l1) < int(min_edges):
        return None, {"reason": "too_few_samples", "dbg": dbg0, "lens0": [float(x) for x in l0], "lens1": [float(x) for x in l1]}
    m0 = float(np.mean(np.asarray(l0, dtype=np.float64)))
    m1 = float(np.mean(np.asarray(l1, dtype=np.float64)))
    best = float(ang0_deg) if m0 < m1 else float(ang1_deg)
    dbg = {
        **dbg0,
        "lens0": [float(x) for x in l0],
        "lens1": [float(x) for x in l1],
        "mean_len0": float(m0),
        "mean_len1": float(m1),
        "best_angle_deg": float(best),
    }
    return best, dbg


def estimate_dadb_from_centroids_edges_and_chain(
    *,
    main_axes_deg: Sequence[float],
    best_angle_deg: float,
    mean_len0: float,
    mean_len1: float,
    chain_dir: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Internal helper."""
    axes = [float(a) for a in main_axes_deg]
    if len(axes) < 2:
        return np.zeros((2,), dtype=np.float64), np.zeros((2,), dtype=np.float64), {"reason": "too_few_axes"}

    a0, a1 = float(axes[0]), float(axes[1])

    def sep(u: float, v: float) -> float:
        d = abs(float(u) - float(v)) % 180.0
        return float(min(d, 180.0 - d))

    is0 = sep(float(best_angle_deg), a0) <= sep(float(best_angle_deg), a1)
    ang_db = a0 if is0 else a1
    ang_da = a1 if is0 else a0
    len_db = float(mean_len0) if is0 else float(mean_len1)
    len_da = float(mean_len1) if is0 else float(mean_len0)

    db_u = np.asarray(chain_dir, dtype=np.float64).reshape(2)
    ndb = float(np.hypot(db_u[0], db_u[1]))
    db_u = db_u / ndb if ndb > 1e-12 else np.array([float(np.cos(np.radians(ang_db))), float(np.sin(np.radians(ang_db)))], dtype=np.float64)

    r_da = float(np.radians(ang_da))
    da_u0 = np.array([float(np.cos(r_da)), float(np.sin(r_da))], dtype=np.float64)

    def angle_between(u: np.ndarray, v: np.ndarray) -> float:
        uu = np.asarray(u, dtype=np.float64).reshape(2)
        vv = np.asarray(v, dtype=np.float64).reshape(2)
        nu = float(np.hypot(uu[0], uu[1]))
        nv = float(np.hypot(vv[0], vv[1]))
        if nu < 1e-12 or nv < 1e-12:
            return 0.0
        c = float(np.dot(uu, vv) / (nu * nv))
        c = max(-1.0, min(1.0, c))
        return float(np.degrees(np.arccos(c)))

    ang0 = angle_between(db_u, da_u0)
    ang1 = angle_between(db_u, -da_u0)
    da_u = da_u0 if abs(ang0 - 120.0) <= abs(ang1 - 120.0) else -da_u0

    db = db_u * float(len_db)
    da = da_u * float(len_da)

    dbg = {
        "ang_db_deg": float(ang_db),
        "ang_da_deg": float(ang_da),
        "len_db_px": float(len_db),
        "len_da_px": float(len_da),
        "angle_db_da_deg": float(angle_between(db, da)),
        "db_unit": [float(db_u[0]), float(db_u[1])],
        "da_unit": [float(da_u[0]), float(da_u[1])],
    }
    return da.astype(np.float64), db.astype(np.float64), dbg


def solve_re_chain_from_centroids(
    centroids: np.ndarray,
    *,
    seed_edge_dirs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    k_neighbors: int = 6,
    max_candidates: int = 6,
) -> Tuple[np.ndarray, Dict]:
    """Internal helper."""
    cand, dbg0 = _pick_candidate_angles_from_centroids(
        centroids, k_neighbors=int(k_neighbors), max_candidates=int(max_candidates), min_sep_deg=10.0
    )
    if not cand:
        return np.array([1.0, 0.0], dtype=np.float64), {"reason": "no_candidates", **dbg0}

    cand3 = _select_topk_candidate_axes(cand, strengths=dbg0.get("picked_angle_strengths"), k=3, min_sep_deg=10.0)
    drop_dbg = None
    dropped = None
    if seed_edge_dirs is not None and len(cand3) >= 3:
        main_axes, dropped, drop_dbg = _drop_least_parallel_to_seed_edges(cand3, seed_edge_dirs=seed_edge_dirs)
    else:
        main_axes = list(cand3)

    if len(main_axes) < 2:
        main_axes = _select_two_main_axes(cand, strengths=dbg0.get("picked_angle_strengths"))
        if len(main_axes) < 2:
            main_axes = [float(x) for x in cand[:2]]
    elif len(main_axes) > 2:
        main_axes2 = _select_two_main_axes(main_axes, strengths=dbg0.get("picked_angle_strengths"))
        main_axes = main_axes2 if len(main_axes2) >= 2 else [float(x) for x in main_axes[:2]]

    neighbor_best = None
    neighbor_dbg = None
    try:
        neighbor_best, neighbor_dbg = _choose_by_neighbor_min_edge(
            centroids,
            ang0_deg=float(main_axes[0]),
            ang1_deg=float(main_axes[1]),
            k_neighbors=int(k_neighbors),
            len_ref="nn_min",
            min_len_factor=0.6,
            max_len_factor=1.2,
            angle_sim_tol_deg=18.0,
            min_edges=1,
        )
    except Exception as e:  # pragma: no cover
        neighbor_best, neighbor_dbg = None, {"reason": "exception", "err": str(e)}

    best_ang = float(neighbor_best) if neighbor_best is not None else float(main_axes[0])
    chosen_by = "neighbor_min_edge" if neighbor_best is not None else "neighbor_failed_fallback_to_axis0"

    r = np.radians(best_ang)
    v = np.array([np.cos(r), np.sin(r)], dtype=np.float64)

    dadb_dbg = None
    da_vec = None
    db_vec = None
    try:
        if len(main_axes) >= 2:
            if isinstance(neighbor_dbg, dict) and isinstance(neighbor_dbg.get("mean_len0"), (int, float)) and isinstance(neighbor_dbg.get("mean_len1"), (int, float)):
                m0 = float(neighbor_dbg["mean_len0"])
                m1 = float(neighbor_dbg["mean_len1"])
                da_vec, db_vec, dadb_dbg = estimate_dadb_from_centroids_edges_and_chain(
                    main_axes_deg=[float(main_axes[0]), float(main_axes[1])],
                    best_angle_deg=float(best_ang),
                    mean_len0=float(m0),
                    mean_len1=float(m1),
                    chain_dir=v,
                )
                dadb_dbg = {
                    **dadb_dbg,
                    "len_source": "neighbor_min_edge",
                    "len_estimation_edges": neighbor_dbg,
                    "mean_len0": float(m0),
                    "mean_len1": float(m1),
                }
            else:
                l0, l1, edbg = _neighbor_edges_for_two_axes(
                    centroids,
                    ang0_deg=float(main_axes[0]),
                    ang1_deg=float(main_axes[1]),
                    k_neighbors=int(k_neighbors),
                    len_ref="nn_med",
                    min_len_factor=0.6,
                    max_len_factor=2.2,
                    angle_sim_tol_deg=18.0,
                    min_edges=1,
                )
                if l0 and l1:
                    m0 = float(np.mean(np.asarray(l0, dtype=np.float64)))
                    m1 = float(np.mean(np.asarray(l1, dtype=np.float64)))
                    da_vec, db_vec, dadb_dbg = estimate_dadb_from_centroids_edges_and_chain(
                        main_axes_deg=[float(main_axes[0]), float(main_axes[1])],
                        best_angle_deg=float(best_ang),
                        mean_len0=float(m0),
                        mean_len1=float(m1),
                        chain_dir=v,
                    )
                    dadb_dbg = {
                        **dadb_dbg,
                        "len_source": "nn_med_fallback",
                        "len_estimation_edges": edbg,
                        "mean_len0": float(m0),
                        "mean_len1": float(m1),
                    }
                else:
                    dadb_dbg = {
                        "reason": "too_few_edge_samples_for_dadb",
                        "len_source": "nn_med_fallback",
                        "len_estimation_edges": edbg,
                        "lens0_n": int(len(l0)),
                        "lens1_n": int(len(l1)),
                    }
    except Exception:  # pragma: no cover
        da_vec = None
        db_vec = None
        dadb_dbg = {"reason": "exception"}

    dbg = {
        **dbg0,
        "candidates_deg": [float(a) for a in cand],
        "cand3_deg": [float(a) for a in cand3],
        "seed_edge_dirs": None
        if seed_edge_dirs is None
        else [[float(seed_edge_dirs[0][0]), float(seed_edge_dirs[0][1])], [float(seed_edge_dirs[1][0]), float(seed_edge_dirs[1][1])]],
        "dropped_candidate_deg": None if dropped is None else float(dropped),
        "drop_debug": drop_dbg,
        "main_axes_deg": [float(a) for a in main_axes[:2]],
        "best_angle_deg": float(best_ang),
        "chosen_by": str(chosen_by),
        "neighbor_best_angle_deg": None if neighbor_best is None else float(neighbor_best),
        "neighbor_debug": neighbor_dbg,
        "dadb_from_centroid_edges": None
        if dadb_dbg is None
        else {
            **dadb_dbg,
            "da_vec_px": None if da_vec is None else [float(da_vec[0]), float(da_vec[1])],
            "db_vec_px": None if db_vec is None else [float(db_vec[0]), float(db_vec[1])],
        },
        "max_candidates": int(max_candidates),
    }
    return v, dbg


__all__ = [
    "_axis_angle_deg",
    "_kmeans_1d_two_clusters",
    "_pick_candidate_angles_from_centroids",
    "estimate_dadb_from_centroids_edges_and_chain",
    "solve_re_chain_from_centroids",
]
