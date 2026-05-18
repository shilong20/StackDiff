"""
【作用概述】提供 da/db 的基础运算：规范化、角度/长度误差、shift 向量在晶格基下的投影等。
核心输入/输出：
- 输入 da/db（2D 向量，像素或归一化坐标均可）与点云/shift；
- 输出误差指标、(u,v) 晶格坐标、以及重建/检查所需的中间量。
【关联说明】文件/模块：tools/pred_dadb/dataset_online.py（标签生成）；tools/pred_dadb/train.py（训练指标）；tools/pred_dadb/predict.py（推理与投影）。
【命令行用法】本文件不直接运行。
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def vec_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(np.hypot(v[0], v[1]) + eps)


def angle_deg(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    """返回两向量夹角（度），范围 [0, 180]。"""
    u = np.asarray(u, dtype=np.float64).reshape(2)
    v = np.asarray(v, dtype=np.float64).reshape(2)
    nu = vec_norm(u, eps=eps)
    nv = vec_norm(v, eps=eps)
    c = float(np.dot(u, v) / (nu * nv))
    c = max(-1.0, min(1.0, c))
    return float(math.degrees(math.acos(c)))


def det2(da: np.ndarray, db: np.ndarray) -> float:
    da = np.asarray(da, dtype=np.float64).reshape(2)
    db = np.asarray(db, dtype=np.float64).reshape(2)
    return float(da[0] * db[1] - da[1] * db[0])


def canonicalize_global_sign(da: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    仅允许对 (da,db) 施加“整体取负”，用于消除 180° 等价带来的全局符号歧义。
    规则：若 da 的方向落在“负半平面”（da_x<0 或 da_x==0 且 da_y<0），则整体乘 -1。
    注意：整体取负不会改变 det 符号（不会破坏 handedness/flip 判别）。
    """
    da = np.asarray(da, dtype=np.float64).reshape(2)
    db = np.asarray(db, dtype=np.float64).reshape(2)
    if (da[0] < 0.0) or (abs(float(da[0])) < 1e-12 and da[1] < 0.0):
        return (-da).astype(np.float64), (-db).astype(np.float64)
    return da.astype(np.float64), db.astype(np.float64)


def canonicalize_order_by_length(da: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    规范化 da/db 的“顺序语义”：强制 |da| >= |db|。
    若需要交换，则使用 (da,db) -> (db,-da) 保持 det 符号不变（handedness 保留）。
    """
    da = np.asarray(da, dtype=np.float64).reshape(2)
    db = np.asarray(db, dtype=np.float64).reshape(2)
    if vec_norm(db) > vec_norm(da):
        return db, -da
    return da, db


def canonicalize_by_reference(da: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 (da,db) 规范化到一个“固定参考系”，用于让输出在跨样本比较时保持一致。

    规则（针对 ReS2 经验选择，便于与 generate_sample/ReS2.json 的定义一致）：
    - 允许用 det=+1 的基变换： (da,db) 与 (db,-da) 视为同一晶格的等价基；
    - 允许整体取负： (da,db) 与 (-da,-db) 等价；
    - 在这些候选中，选取使 da 最接近 +x 方向（最大化 da_x/||da||）的那一组；
      若分数相同，优先使 db 的 y 分量为负（更接近 ReS2.json 中 db_y<0 的习惯）。

    注意：该规范化不会改变 handedness（det 符号保持不变），因此可用于后续 flip 判别。
    """
    da = np.asarray(da, dtype=np.float64).reshape(2)
    db = np.asarray(db, dtype=np.float64).reshape(2)

    cands = [
        (da, db),
        (-da, -db),
        (db, -da),
        (-db, da),
    ]

    best = None
    best_score = -1e30
    best_tie = 1e30

    for a, b in cands:
        na = vec_norm(a)
        # 主目标：da 对齐 +x
        score = float(a[0] / max(na, 1e-12))
        # 次目标：db_y 越负越好（tie-break）
        tie = float(b[1])  # 越小越好
        if score > best_score + 1e-12:
            best_score = score
            best_tie = tie
            best = (a, b)
        elif abs(score - best_score) <= 1e-12 and tie < best_tie:
            best_tie = tie
            best = (a, b)

    assert best is not None
    return best[0], best[1]


def gauss_reduce_2d(da: np.ndarray, db: np.ndarray, *, max_iter: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 2D 晶格基做 Gauss/Minkowski 风格的约化（只用 det=±1 的整数变换，不改变晶格）。

    目标（经验上更稳定）：
    - ||da|| <= ||db||
    - |da·db| <= 0.5*||da||^2

    注意：该约化不会固定“基矢的物理命名语义”，只是把基变得更短、更接近正交，便于后续规则化。
    """
    a = np.asarray(da, dtype=np.float64).reshape(2).copy()
    b = np.asarray(db, dtype=np.float64).reshape(2).copy()

    def n2(x: np.ndarray) -> float:
        return float(np.dot(x, x))

    for _ in range(int(max_iter)):
        if n2(b) < n2(a):
            a, b = b, a
        mu = float(np.dot(a, b) / max(n2(a), 1e-12))
        k = float(np.round(mu))
        if abs(k) <= 1e-12:
            # 已经接近约化条件
            if abs(float(np.dot(a, b))) <= 0.5 * max(n2(a), 1e-12) + 1e-9:
                break
            # 否则继续下一轮（swap 后可能会更好）
            continue
        b = b - k * a

    return a.astype(np.float64), b.astype(np.float64)


def flip_y_points(points_xy: np.ndarray, *, size_px: int) -> np.ndarray:
    """
    在 [0,size) 的 patch 坐标里做 y 轴翻转：
      y -> (size-1)-y
    常用于把“图像坐标(y 向下)”与“数学坐标(y 向上)”互相转换。
    """
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2).copy()
    pts[:, 1] = float(size_px - 1) - pts[:, 1]
    return pts.astype(np.float32)


def flip_y_vecs(vecs: np.ndarray) -> np.ndarray:
    """对 2D 向量做 y 分量取反（坐标系翻转下的向量变换）。"""
    v = np.asarray(vecs, dtype=np.float64).reshape(-1, 2).copy()
    v[:, 1] = -v[:, 1]
    return v.astype(np.float64)


def res2_user_dadb_from_res2json_yup(
    t1_t2_yup: np.ndarray,
) -> np.ndarray:
    """
    将 `generate_sample/ReS2.json` 语义下的 (t1,t2)（y 向上坐标）映射到用户定义的 (da,db)（图像坐标 y 向下）。

    约定（与用户蓝色定义对齐）：
    - 先把向量从 y-up 转到 y-down：v_down = (vx, -vy)
    - 再定义：
        da = t2_down
        db = -t1_down
    这样在“未旋转、未翻转”的标准取向下：
      t1=(6.42,0), t2=(3.15,-5.71)  ->  da=(3.15,5.71), db=(-6.42,0)
    """
    tt = np.asarray(t1_t2_yup, dtype=np.float64).reshape(2, 2)
    t_down = flip_y_vecs(tt)
    da = t_down[1]
    db = -t_down[0]
    out = np.stack([da, db], axis=0)
    return out.astype(np.float64)


def best_det_plus1_variant_to_match(
    da: np.ndarray, db: np.ndarray, *, ref_da: np.ndarray, ref_db: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 det=+1 的等价变换集合中选择与参考 (ref_da,ref_db) 最接近的一组。
    候选集合（不改变 handedness）：
      (da,db), (-da,-db), (db,-da), (-db,da)
    距离度量：对单位向量的 L2 距离（方向优先）。
    """
    a = np.asarray(da, dtype=np.float64).reshape(2)
    b = np.asarray(db, dtype=np.float64).reshape(2)
    ra = np.asarray(ref_da, dtype=np.float64).reshape(2)
    rb = np.asarray(ref_db, dtype=np.float64).reshape(2)

    def _unit(x: np.ndarray) -> np.ndarray:
        n = vec_norm(x)
        return (x / max(n, 1e-12)).astype(np.float64)

    ua, ub = _unit(a), _unit(b)
    ura, urb = _unit(ra), _unit(rb)

    cands: Iterable[Tuple[np.ndarray, np.ndarray]] = (
        (ua, ub),
        (-ua, -ub),
        (ub, -ua),
        (-ub, ua),
    )

    best = None
    best_d = 1e30
    for ca, cb in cands:
        d = float(np.linalg.norm(ca - ura) + np.linalg.norm(cb - urb))
        if d < best_d:
            best_d = d
            best = (ca, cb)

    assert best is not None
    # 把选择结果映射回原始尺度（保留长度）
    # 注意：候选是在单位向量空间选的，因此这里用“选择对应的变换”而不是直接返回 best。
    # 为简洁，这里再比较一次以找回变换。
    raw_cands = [
        (a, b),
        (-a, -b),
        (b, -a),
        (-b, a),
    ]
    # 用同一个距离定义选 raw
    best_raw = raw_cands[0]
    best_raw_d = 1e30
    for ca, cb in raw_cands:
        d = float(np.linalg.norm(_unit(ca) - ura) + np.linalg.norm(_unit(cb) - urb))
        if d < best_raw_d:
            best_raw_d = d
            best_raw = (ca, cb)
    return best_raw[0].astype(np.float64), best_raw[1].astype(np.float64)


def project_shift_to_lattice(shift_xy: np.ndarray, da: np.ndarray, db: np.ndarray, *, eps: float = 1e-9) -> Tuple[float, float]:
    """
    将像素坐标系下的 shift 向量投影到 (da, db) 张成的基下，解线性方程：
      [da db] [u v]^T = shift
    若矩阵接近奇异则用最小二乘。
    """
    shift = np.asarray(shift_xy, dtype=np.float64).reshape(2)
    A = np.stack([np.asarray(da, dtype=np.float64).reshape(2), np.asarray(db, dtype=np.float64).reshape(2)], axis=1)  # 2x2
    if abs(float(np.linalg.det(A))) < eps:
        uv, *_ = np.linalg.lstsq(A, shift, rcond=None)
    else:
        uv = np.linalg.solve(A, shift)
    return float(uv[0]), float(uv[1])


def reconstruct_shift_from_lattice(u: float, v: float, da: np.ndarray, db: np.ndarray) -> np.ndarray:
    da = np.asarray(da, dtype=np.float64).reshape(2)
    db = np.asarray(db, dtype=np.float64).reshape(2)
    return da * float(u) + db * float(v)


def dadb_errors(
    da_pred: np.ndarray,
    db_pred: np.ndarray,
    da_gt: np.ndarray,
    db_gt: np.ndarray,
) -> dict:
    """返回角度/长度/面积（det）误差的简单统计。"""
    da_pred = np.asarray(da_pred, dtype=np.float64).reshape(2)
    db_pred = np.asarray(db_pred, dtype=np.float64).reshape(2)
    da_gt = np.asarray(da_gt, dtype=np.float64).reshape(2)
    db_gt = np.asarray(db_gt, dtype=np.float64).reshape(2)
    out = {
        "ang_da_deg": angle_deg(da_pred, da_gt),
        "ang_db_deg": angle_deg(db_pred, db_gt),
        "len_da": vec_norm(da_pred),
        "len_db": vec_norm(db_pred),
        "len_da_gt": vec_norm(da_gt),
        "len_db_gt": vec_norm(db_gt),
        "det": det2(da_pred, db_pred),
        "det_gt": det2(da_gt, db_gt),
    }
    out["len_da_abs_err"] = abs(out["len_da"] - out["len_da_gt"])
    out["len_db_abs_err"] = abs(out["len_db"] - out["len_db_gt"])
    out["det_abs_err"] = abs(out["det"] - out["det_gt"])
    return out
