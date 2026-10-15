"""
calib_common.py
────────────────────────────────────────────────────────────────────
較正の各プログラムで共有する幾何計算．

  座標系の記号（scripts/ と同じ）
    B : ボード座標系   原点 = 内部コーナー#0，X右，Y下，Z ボード奥 [mm]
    C : カラー光学座標系（X右，Y下，Z前方）[mm]
    M : Motive 座標系 [mm]
    T_XY : Y 座標系の点を X 座標系へ移す 4×4 同次変換（p_X = T_XY · p_Y）
────────────────────────────────────────────────────────────────────
"""

import numpy as np
import cv2

import config

# ボード上の内部コーナー座標 [mm]（行優先，findChessboardCornersSB の並びと同じ）
OBJ = np.array([[j * config.SQUARE, i * config.SQUARE, 0.0]
                for i in range(config.PATTERN[1]) for j in range(config.PATTERN[0])], np.float32)
MARKERS_B = np.array(list(config.MARKERS_B.values()), float)   # S1, S2, S3


# ══════════════════════════════════════════════════
#  同次変換
# ══════════════════════════════════════════════════
def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def apply(T, P):
    """点群 P (N×3) に T を掛ける．"""
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
    """カメラ座標の点群 → 画素座標．"""
    return cv2.projectPoints(np.asarray(P_C, float), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)


def find_corners(gray):
    ok, c = cv2.findChessboardCornersSB(gray, config.PATTERN,
                                        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    return c if ok else None


def solve_board(gray, K, D, max_err=config.RS_MAX_ERR_PX):
    """1 枚の画像からボード姿勢を求める．検出失敗・誤差過大なら None．
    返り値: dict(R, t [mm], err [px], corners)"""
    c = find_corners(gray)
    if c is None:
        return None

    def pnp(c):
        _, r, t = cv2.solvePnP(OBJ, c, K, D, flags=cv2.SOLVEPNP_IPPE)
        return cv2.solvePnPRefineLM(OBJ, c, K, D, r, t)

    r, t = pnp(c)
    # コーナー列の 180° 反転チェック: 原点の左上マスは黒のはず．白なら逆順にして解き直す
    u, v = project(apply(make_T(cv2.Rodrigues(r)[0], t), [[-config.SQUARE / 2, -config.SQUARE / 2, 0]]),
                   K, D)[0].astype(int)
    if 3 <= u < gray.shape[1] - 4 and 3 <= v < gray.shape[0] - 4 and gray[v - 3:v + 4, u - 3:u + 4].mean() >= 128:
        c = c[::-1].copy()
        r, t = pnp(c)

    R = cv2.Rodrigues(r)[0]
    err = float(np.linalg.norm(cv2.projectPoints(OBJ, r, t, K, D)[0].reshape(-1, 2) - c.reshape(-1, 2), axis=1).mean())
    if R[2, 2] <= 0 or err > max_err:          # 裏向き解・誤差過大は捨てる
        return None
    return dict(R=R, t=t.ravel(), err=err, corners=c.reshape(-1, 2))
