"""
geometry.py
────────────────────────────────────────────────────────────────────
座標変換・カメラモデル・時系列の補間など，全プログラムで共有する計算．

  座標系の記号
    B : ボード座標系   原点 = 内部コーナー#0，X右，Y下，Z ボード奥 [mm]
    C : カラー光学座標系（X右，Y下，Z前方）[mm]
    M : Motive 座標系 [mm]
    T_XY : Y 座標系の点を X 座標系へ移す 4×4 同次変換（p_X = T_XY · p_Y）
────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import cv2


# ══════════════════════════════════════════════════
#  同次変換
# ══════════════════════════════════════════════════
def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def apply(T, P):
    """点群 P (..., 3) に T を掛ける．"""
    return np.asarray(P) @ T[:3, :3].T + T[:3, 3]


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def mean_R(Rs):
    """回転行列の平均（SVD で最も近い回転に射影）．"""
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def kabsch(A, B):
    """B ≈ T·A となる剛体変換 T (4×4) を最小二乗で求める．A, B は N×3 の対応点．"""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T   # det で鏡映を防ぐ
    return make_T(R, cb - R @ ca)


# ══════════════════════════════════════════════════
#  カメラ
# ══════════════════════════════════════════════════
def K_D(intr):
    """rs_device.intrinsics_to_dict() の dict → OpenCV の K, D．"""
    K = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]], float)
    return K, np.array(intr["coeffs"], float)


def project(P_C, K, D):
    """カメラ座標の点群 (N×3) → 画素座標 (N×2)．"""
    return cv2.projectPoints(np.asarray(P_C, float).reshape(-1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)


def deproject(uv, z, K, D):
    """画素座標 (N×2) と奥行き z (N,) [mm] → カメラ座標 (N×3) [mm]．z が NaN なら結果も NaN．"""
    xy = cv2.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2), K, D).reshape(-1, 2)   # 正規化座標
    z = np.asarray(z, float)
    return np.column_stack([xy * z[:, None], z])


def split_ray(P_C, Q_C):
    """カメラ座標の差 P−Q を，視線方向（Q を見る方向）とそれに垂直な方向に分ける．
    返り値: (視線方向の成分 [mm]（正 = P が遠い）, 垂直な成分のベクトル (…, 3))"""
    u = Q_C / np.linalg.norm(Q_C, axis=-1, keepdims=True)
    D = P_C - Q_C
    along = np.sum(D * u, axis=-1)
    return along, D - along[..., None] * u


# ══════════════════════════════════════════════════
#  時系列
# ══════════════════════════════════════════════════
def interp_nan(tq, t, y, max_gap):
    """欠損（NaN）を max_gap [s] 以上またいで補間しない線形補間．y: (n, ...) → (len(tq), ...)"""
    y2 = np.asarray(y, float).reshape(len(t), -1)
    ok = np.isfinite(y2).all(axis=1)
    out = np.full((len(tq), y2.shape[1]), np.nan)
    if ok.sum() < 2:
        return out.reshape((len(tq),) + np.shape(y)[1:])
    tv, yv = np.asarray(t)[ok], y2[ok]
    i = np.clip(np.searchsorted(tv, tq), 1, len(tv) - 1)
    t0, t1 = tv[i - 1], tv[i]
    w = ((tq - t0) / np.where(t1 > t0, t1 - t0, 1))[:, None]
    good = (t1 - t0 <= max_gap) & (tq >= tv[0]) & (tq <= tv[-1])
    out[good] = (yv[i - 1] * (1 - w) + yv[i] * w)[good]
    return out.reshape((len(tq),) + np.shape(y)[1:])


def velocity(t, P, smooth_s=0.05):
    """位置 (n, ..., 3) の時間微分 [mm/s]．等間隔でなくてよい．smooth_s の移動平均でノイズを抑える．"""
    V = np.gradient(np.asarray(P, float), t, axis=0)
    k = max(1, int(round(smooth_s / np.median(np.diff(t)))))
    flat = pd.DataFrame(V.reshape(len(t), -1)).rolling(k, center=True, min_periods=1).mean().to_numpy()
    return flat.reshape(V.shape)
