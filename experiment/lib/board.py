"""
board.py
────────────────────────────────────────────────────────────────────
較正・同期用ボード（チェッカー 7×4 + 反射マーカー S1〜S3）の検出．

  RealSense 側: solve_board()   画像 → ボード姿勢 T_CB（向きの 180° 取り違えを修正済み）
  MoCap 側    : identify()      ラベルなしマーカー → S1, S2, S3
                board_frame()   S1〜S3 → T_MB

【向きの判定】
  7×4 の内部コーナーは 180° 回しても同じ格子なので，検出器が返すコーナー順は逆になりうる．
  8×5 マスのうち「白のはずのマス」と「黒のはずのマス」の明るさの差（contrast）で決める．
  差を見るので露出に依らない（1 点の明るさを固定のしきい値で見ると，露出が暗いとき
  紙の白が 127 前後になり，2026-10-15 の試行では 2 割のフレームで向きを誤った）．
────────────────────────────────────────────────────────────────────
"""

import numpy as np
import cv2

import config
from lib.geometry import make_T

# ボード上の内部コーナー座標 [mm]（行優先，findChessboardCornersSB の並びと同じ）
OBJ = np.array([[j * config.SQUARE, i * config.SQUARE, 0.0]
                for i in range(config.PATTERN[1]) for j in range(config.PATTERN[0])], np.float32)
MARKERS_B = np.array(list(config.MARKERS_B.values()), float)   # S1, S2, S3

# 8×5 マスの中心（ボード座標）と色．原点側の角のマスが黒，(i+j) が奇数のマスが白
_SQ = [(i, j) for i in range(config.PATTERN[0] + 1) for j in range(config.PATTERN[1] + 1)]
SQ_CENTERS = np.array([[(i - 0.5) * config.SQUARE, (j - 0.5) * config.SQUARE, 0.0] for i, j in _SQ])
SQ_WHITE = np.array([(i + j) % 2 == 1 for i, j in _SQ])

# 設計上のマーカー間距離（S1S2, S1S3, S2S3）
PAIRS = [(0, 1), (0, 2), (1, 2)]
D_REF = [np.linalg.norm(MARKERS_B[b] - MARKERS_B[a]) for a, b in PAIRS]


# ══════════════════════════════════════════════════
#  RealSense: 画像 → T_CB
# ══════════════════════════════════════════════════
def find_corners(gray, exhaustive=True):
    """exhaustive=False は速いが見落としが増える（動いているボードを全フレーム調べる同期用）．"""
    flags = cv2.CALIB_CB_ACCURACY | (cv2.CALIB_CB_EXHAUSTIVE if exhaustive else 0)
    ok, c = cv2.findChessboardCornersSB(gray, config.PATTERN, flags=flags)
    return c if ok else None


def flip_pose(R, t):
    """ボードを面内で 180° 回した（コーナー順を逆にした）姿勢．"""
    F = np.diag([-1.0, -1.0, 1.0])
    t_off = np.array([(config.PATTERN[0] - 1) * config.SQUARE, (config.PATTERN[1] - 1) * config.SQUARE, 0.0])
    return R @ F, np.ravel(t) + R @ t_off


def square_contrast(gray, r, t, K, D, corners):
    """「白のはずのマス − 黒のはずのマス」の明るさの差．正しい向きなら正，逆なら負．"""
    uv = cv2.projectPoints(SQ_CENTERS, r, t, K, D)[0].reshape(-1, 2)
    step = np.median(np.linalg.norm(np.diff(corners[:config.PATTERN[0]], axis=0), axis=1))   # 1 マスの画素数
    half = max(1, int(0.15 * step))
    h, w = gray.shape
    vals = np.full(len(uv), np.nan)
    for k, (u, v) in enumerate(np.round(uv).astype(int)):
        if half <= u < w - half and half <= v < h - half:
            vals[k] = gray[v - half:v + half + 1, u - half:u + half + 1].mean()
    return float(np.nanmean(vals[SQ_WHITE]) - np.nanmean(vals[~SQ_WHITE]))


def solve_board(gray, K, D, max_err=config.RS_MAX_ERR_PX, exhaustive=True):
    """1 枚の画像からボード姿勢を求める．検出失敗・誤差過大なら None．
    返り値: dict(R, t [mm], err [px], corners, contrast)
      contrast : 白黒の明るさの差（向きを決めた後なので常に ≥ 0）．小さいフレームは向きが怪しい．"""
    c = find_corners(gray, exhaustive)
    if c is None:
        return None
    c = c.reshape(-1, 2)
    _, r, t = cv2.solvePnP(OBJ, c, K, D, flags=cv2.SOLVEPNP_IPPE)
    r, t = cv2.solvePnPRefineLM(OBJ, c, K, D, r, t)
    R = cv2.Rodrigues(r)[0]
    err = float(np.linalg.norm(cv2.projectPoints(OBJ, r, t, K, D)[0].reshape(-1, 2) - c, axis=1).mean())
    if R[2, 2] <= 0 or err > max_err:          # 裏向き解・誤差過大は捨てる
        return None

    contrast = square_contrast(gray, r, t, K, D, c)
    t = t.ravel()
    if contrast < 0:                           # コーナー順が逆 → 180° 回す（解き直しと同じ結果）
        R, t = flip_pose(R, t)
        c = c[::-1].copy()
    return dict(R=R, t=t, err=err, corners=c, contrast=abs(contrast))


# ══════════════════════════════════════════════════
#  MoCap: ラベルなしマーカー → S1〜S3 → T_MB
# ══════════════════════════════════════════════════
def identify(P, tol=config.MC_TOL):
    """見えている点 P (N×3) から，マーカー間距離が設計値に最も合う 3 点を S1, S2, S3 の順で返す．なければ None．"""
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    best, cost = None, np.inf
    n = len(P)
    for a in range(n):
        for b in range(n):
            if b == a or abs(d[a, b] - D_REF[0]) >= tol:      # S1S2 が合わない組は早めに外す（点が多いときの高速化）
                continue
            for c in range(n):
                if c in (a, b):
                    continue
                e = (abs(d[a, b] - D_REF[0]), abs(d[a, c] - D_REF[1]), abs(d[b, c] - D_REF[2]))
                if max(e) < tol and sum(e) < cost:
                    best, cost = P[[a, b, c]], sum(e)
    return best


def identify_all(X, tol=config.MC_TOL):
    """全フレームで S1〜S3 を識別する．X: (frames, markers, 3) → (frames, 3, 3)，見つからなければ NaN．"""
    S = np.full((len(X), 3, 3), np.nan)
    for i, x in enumerate(X):
        P = x[np.isfinite(x[:, 0])]
        if len(P) >= 3 and (res := identify(P, tol)) is not None:
            S[i] = res
    return S


def board_frame(S1, S2, S3):
    """S1→S2 を X，S1→S3 を Y（直交化）とする T_MB．"""
    x = (S2 - S1) / np.linalg.norm(S2 - S1)
    v = S3 - S1
    y = v - (v @ x) * x
    y /= np.linalg.norm(y)
    R = np.column_stack([x, y, np.cross(x, y)])
    return make_T(R, S1 - R @ MARKERS_B[0])      # 原点 = 内部コーナー#0
