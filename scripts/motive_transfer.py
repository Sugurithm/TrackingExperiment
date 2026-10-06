# Motive のテイク (.csv, 複数姿勢を連続記録) から静止区間を切り出す -> 姿勢ごとの T^M_B（ボード座標 -> Motive 座標, mm）
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====== 設定 =====
FILEPATH = Path("recordings\Take_2026-08-31_025049PM.csv")
TOL = 3.0             # マーカー間距離の許容 [mm]
WIN = 0.5             # 静止判定の窓幅 [s]
STILL = 4.0           # 窓内の振れ幅がこれ未満なら静止 [mm]
MIN_LEN = 5.0         # 静止区間の最短長 [s]
MERGE = 3.0           # この時間以内の途切れは同じ姿勢として結合 [s]
TRIM = 0.25           # 静止区間の最初と最後を捨てる [s]
OUTLIER = 3.0         # 中央値からこれ以上離れたフレームを捨てる [mm]

SCALE = 0.9588        # 印刷倍率（MoCap のマーカー間距離から推定。定規で実測したら置き換える。RealSense 側の SQUARE にも掛ける）
MARKERS_B = {k: (x * SCALE, y * SCALE, 0) for k, (x, y) in {"S1": (-40, -40), "S2": (190, -40), "S3": (-40, 115)}.items()}
PB = np.array(list(MARKERS_B.values()), float)
D12, D13, D23 = (np.linalg.norm(PB[b] - PB[a]) for a, b in [(0, 1), (0, 2), (1, 2)])

# ===== S1, S2, S3 を距離で識別 =====
def identify(P):
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    best, cost = None, np.inf
    for s1, s2, s3 in ((a, b, c) for a in range(len(P)) for b in range(len(P)) for c in range(len(P)) if len({a, b, c}) == 3):
        e = [abs(d[s1, s2] - D12), abs(d[s1, s3] - D13), abs(d[s2, s3] - D23)]
        if max(e) < TOL and sum(e) < cost:
            best, cost = P[[s1, s2, s3]], sum(e)
    return best

# ===== ボード座標系 =====
def board_frame(S1, S2, S3):
    x = (S2 - S1) / np.linalg.norm(S2 - S1)
    v = S3 - S1
    y = v - (v @ x) * x; y /= np.linalg.norm(y)
    R = np.column_stack([x, y, np.cross(x, y)])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = S1 - R @ PB[0]     # 原点 = S1 + 40x + 40y
    return T


# ===== CSV 読み込み =====
with open(FILEPATH, encoding="utf-8-sig", errors="replace") as f:
    head = [r for r, _ in zip(csv.reader(f), range(7))]
meta = dict(zip(head[0][0::2], head[0][1::2]))
types, kinds, axes = head[2], head[5], head[6]
cols = [j for j in range(len(types)) if types[j] == "Marker" and kinds[j] == "Position" and axes[j] == "X"]
df = pd.read_csv(FILEPATH, skiprows=7, header=None, names=range(len(types)), dtype=np.float32)
t = df[1].to_numpy()
X = df[[j + k for j in cols for k in range(3)]].to_numpy().reshape(len(df), len(cols), 3)
X *= {"Meters": 1000, "Millimeters": 1}[meta["Length Units"]]
fps = float(meta["Export Frame Rate"])
print(f"{len(t)} frames  {len(cols)} markers  {fps:.0f} fps  {meta['Length Units']}")

# ===== 毎フレーム識別 =====
S = np.full((len(t), 3, 3), np.nan)
for i, x in enumerate(X):
    P = x[~np.isnan(x[:, 0])].astype(float)
    if len(P) >= 3 and (res := identify(P)) is not None:
        S[i] = res
print(f"identified {np.isfinite(S[:, 0, 0]).sum()} / {len(t)} frames")

# ===== 静止区間（窓内の振れ幅で判定）=====
w = int(WIN * fps)
roll = pd.DataFrame(S.reshape(-1, 9)).rolling(w, center=True, min_periods=int(0.8 * w))
still = ((roll.max() - roll.min()).max(axis=1) < STILL).to_numpy()
edge = np.diff(np.r_[0, still.astype(int), 0])
segs = []
for a, b in zip(np.where(edge == 1)[0], np.where(edge == -1)[0] - 1):
    if segs and t[a] - t[segs[-1][1]] < MERGE and \
            np.nanmax(abs(np.nanmedian(S[a:b + 1], 0) - np.nanmedian(S[segs[-1][0]:segs[-1][1] + 1], 0))) < OUTLIER:
        segs[-1][1] = b
    else:
        segs.append([a, b])
n = int(TRIM * fps)
segs = [(a + n, b - n) for a, b in segs if t[b] - t[a] - 2 * TRIM >= MIN_LEN]
print(f"{len(segs)} poses")

# ===== 姿勢ごとに 1 つの値にまとめる =====
poses = []
for a, b in segs:
    seg = S[a:b + 1][np.isfinite(S[a:b + 1, 0, 0])]
    seg = seg[np.linalg.norm(seg - np.median(seg, 0), axis=2).max(axis=1) < OUTLIER]
    med = np.median(seg, axis=0)
    T = board_frame(*med)
    err = [np.linalg.norm(med[j] - med[i]) - D for (i, j), D in zip([(0, 1), (0, 2), (1, 2)], [D12, D13, D23])]
    std = seg.std(axis=0)
    poses.append(dict(T_MB=T.tolist(), n_frames=len(seg), marker_std=std.tolist(), dist_err=err, t_start=float(t[a]), t_end=float(t[b])))
    print(f"t={t[a]:6.2f}-{t[b]:6.2f}s  used {len(seg)} frames  origin={T[:3, 3].round(2)}  "
        f"std={std.max():.3f} mm  dist_err={np.round(err, 2)} mm  det={np.linalg.det(T[:3, :3]):.6f}")

# ===== 保存 =====
# T_MB : ボード座標系（Board）から Motive 座標系への4×4 同次変換行列 [mm]
# marker_std : S1, S2, S3 の位置の標準偏差 [mm]
# dist_err : マーカー間距離（S1S2, S1S3, S2S3）の実測 - 設計 [mm]
# t_start, t_end : 姿勢推定に使用した区間の最初と最後の時刻（テイク先頭から）[s]
out = FILEPATH.parent / "calib_mocap"
out.mkdir(exist_ok=True)
json.dump(dict(poses=poses, scale=SCALE, markers_B=MARKERS_B, take=meta["Take Name"],
        capture_start=meta["Capture Start Time"]),
        open(out / "T_MB.json", "w"), indent=2, ensure_ascii=False)

# 目視確認: S1 の軌跡と切り出した静止区間
plt.figure(figsize=(14, 4))
plt.plot(t, S[:, 0], lw=0.8)
for k, (a, b) in enumerate(segs):
    plt.axvspan(t[a], t[b], color="orange", alpha=0.3)
    plt.text((t[a] + t[b]) / 2, np.nanmax(S[:, 0]), str(k), ha="center")
plt.legend(["S1 X", "S1 Y", "S1 Z"])
plt.xlabel("time [s]")
plt.ylabel("[mm]")
plt.savefig(out / "pose_check.png", dpi=120, bbox_inches="tight")
print(f"saved -> {out}")