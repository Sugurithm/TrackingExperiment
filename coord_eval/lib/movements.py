"""1 動作（ストローク）の切り出し

課題は連続追従（正弦波の和）なので「A から B へ動かす 1 回の動作」は記録に区切りとして無い。
ここでは **マウス（カメラに依らない真値）の速さの谷**で区切る:
  1. マウスの変位（5 Hz 低域通過済み、解析グリッド 60 Hz）の速さ v(t)
  2. 速さの谷（深さ ≥ MOVE_MIN_PROM × 速さの中央値）を区切りにする
  3. 区切りの間を 1 動作とし、長さ・正味の移動量・直線性で絞る
     （直線性 = 正味の移動量 / 道のり。往復を含む区間は「A → B」ではないので除く）
"""
import numpy as np
from scipy.signal import find_peaks

from .. import config as K


def segment(mouse, fs=K.FS_U):
    """マウスの変位 (N,2) [mm] → 動作の一覧 [{i0, i1, D(2,), amp, dir_deg, dur, straight, vpeak}]。"""
    X = np.asarray(mouse, float)
    ok = np.isfinite(X).all(1)
    if ok.sum() < 30:
        return []
    V = np.gradient(X, axis=0) * fs
    sp = np.linalg.norm(V, axis=1)
    sp_f = np.where(np.isfinite(sp), sp, 0.0)
    med = np.nanmedian(sp[ok])
    if not np.isfinite(med) or med <= 0:
        return []
    mins, _ = find_peaks(-sp_f, prominence=K.MOVE_MIN_PROM * med, distance=max(1, int(0.1 * fs)))
    edges = np.r_[0, mins, len(X) - 1]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if not (ok[a] and ok[b]) or not ok[a:b + 1].all():
            continue
        dur = (b - a) / fs
        if not (K.MOVE_DUR_S[0] <= dur <= K.MOVE_DUR_S[1]):
            continue
        D = X[b] - X[a]
        amp = float(np.linalg.norm(D))
        path = float(np.sum(np.linalg.norm(np.diff(X[a:b + 1], axis=0), axis=1)))
        straight = amp / path if path > 0 else 0.0
        if amp < K.MOVE_MIN_MM or straight < K.MOVE_MIN_STRAIGHT:
            continue
        out.append({"i0": int(a), "i1": int(b), "D": D, "amp": amp,
                    "dir_deg": float(np.degrees(np.arctan2(D[0], D[1]))),   # 0 = 奥、+90 = 右
                    "dur": dur, "straight": straight, "vpeak": float(np.nanmax(sp[a:b + 1]))})
    return out


def edge_mean(X, i, n=K.MOVE_EDGE_N):
    """i の前後 ±n フレームの平均（端点の位置のノイズを減らす）。"""
    a, b = max(0, i - n), min(len(X), i + n + 1)
    return np.nanmean(X[a:b], axis=0)


def direction_class(dir_deg):
    """動作の向き: 左右（x）/ 前後（y）/ 斜め。"""
    a = abs(((dir_deg + 90) % 180) - 90)          # 0 = 前後、90 = 左右
    if a >= 60:
        return "x"
    if a <= 30:
        return "y"
    return "diag"
