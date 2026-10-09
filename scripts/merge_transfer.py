# T_MB（motive_transfer.py）と T_CB（realsense_transfer.py）を結合した T_MC = T_MB · T_CB^-1 を目視で確認する
#   overlay_all.png : MoCap の生マーカー（赤）と MoCap 由来のチェッカー交点（シアン）を RGB 画像に投影
#                     T_MC はその姿勢を除いて求める（LOO）ので自己採点にならない
#   overlay_err.csv : MoCap 由来の交点と RealSense の検出交点の距離 [px]
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ====== 設定 =====
ROOT = Path("recordings")
TAKE = ROOT / "Take_2026-08-31_025049PM.csv"
OUT = ROOT / "verify"
SQUARE = 23.97
PATTERN = (7, 4)
OBJ = np.array([[j * SQUARE, i * SQUARE, 0] for i in range(PATTERN[1]) for j in range(PATTERN[0])], float)


def load_json(p):
    raw = Path(p).read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:            # Windows で書いた T_MB.json は cp932
        return json.loads(raw.decode("cp932"))


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def kabsch(A, B):                         # B ≈ T·A となる 4x4
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cb - R @ ca
    return T


def project(P, K, D):
    return cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)


# ===== 読み込み =====
mb = load_json(ROOT / "calib_mocap" / "T_MB.json")
PB = np.array(list(mb["markers_B"].values()), float)

with open(TAKE, encoding="utf-8-sig", errors="replace") as f:
    head = [r for r, _ in zip(csv.reader(f), range(7))]
meta = dict(zip(head[0][0::2], head[0][1::2]))
cols = [j for j in range(len(head[2])) if head[2][j] == "Marker" and head[5][j] == "Position" and head[6][j] == "X"]
df = pd.read_csv(TAKE, skiprows=7, header=None, names=range(len(head[2])), dtype=np.float32)
t_take = df[1].to_numpy()
X = df[[j + k for j in cols for k in range(3)]].to_numpy().reshape(len(df), len(cols), 3).astype(float)
X *= {"Meters": 1000, "Millimeters": 1}[meta["Length Units"]]

# ===== 録画と MoCap 姿勢を撮影順で対応づけ（RealSense と Motive は別 PC で時計が合っていないので時刻は使わない）=====
recs = sorted(ROOT.glob("rec_*"))
assert len(recs) == len(mb["poses"]), f"録画 {len(recs)} 本と MoCap 姿勢 {len(mb['poses'])} 個の数が合わない"
pairs = []
for k, (rec, p) in enumerate(zip(recs, mb["poses"])):
    cb = load_json(rec / "calib_rs" / "T_CB.json")
    Xp = X[(t_take >= p["t_start"]) & (t_take <= p["t_end"])]
    Xp = Xp[:, np.isfinite(Xp[:, :, 0]).mean(0) > 0.9]      # 区間中ほぼずっと見えている列だけ（一瞬のゴーストを除く）
    pairs.append(dict(rec=rec, cb=cb, pose=k, T_CB=np.array(cb["T_CB"]), T_MB=np.array(p["T_MB"]),
                      P_M=np.nanmedian(Xp, axis=0)))       # 静止区間の生マーカー（中央値）
    print(f"{rec.name} -> pose {k}")


def solve_T_MC(ps):
    return kabsch(np.vstack([apply(p["T_CB"], PB) for p in ps]), np.vstack([apply(p["T_MB"], PB) for p in ps]))


print("camera position in Motive [mm]:", solve_T_MC(pairs)[:3, 3].round(1))

# ===== 画像への投影 =====
tiles, rows = [], []
for p in pairs:
    rec, K, D = p["rec"], np.array(p["cb"]["K"]), np.array(p["cb"]["D"])
    T_CM = np.linalg.inv(solve_T_MC([q for q in pairs if q is not p]))   # LOO
    ts = pd.read_csv(rec / "frame_timestamps.csv")["timestamp_ms"].to_numpy() / 1000   # RealSense の時計の中だけで使う
    k = int(np.argmin(abs(ts - (p["cb"]["t_start"] + p["cb"]["t_end"]) / 2)))
    img = cv2.imread(str(sorted((rec / "rgb").glob("*.jpg"))[k]))

    q_mk = project(apply(T_CM, p["P_M"]), K, D)
    q_bd = project(apply(T_CM @ p["T_MB"], OBJ), K, D)
    ok, c = cv2.findChessboardCornersSB(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), PATTERN,
                                        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    e = np.linalg.norm(q_bd[:, None] - c.reshape(1, -1, 2), axis=2).min(1) if ok else np.full(1, np.nan)
    rows.append(dict(rec=rec.name, pose=p["pose"], err_mean_px=e.mean(), err_max_px=e.max()))

    for u in q_bd:
        cv2.circle(img, tuple(np.round(u).astype(int)), 3, (255, 255, 0), 1)
    for u in q_mk:
        cv2.circle(img, tuple(np.round(u).astype(int)), 7, (0, 0, 255), 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, f"{rec.name} pose {p['pose']}  corner err mean {e.mean():.1f} max {e.max():.1f} px",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(img)
    print(f"  pose {p['pose']}: corner err mean {e.mean():.2f} max {e.max():.2f} px")

OUT.mkdir(exist_ok=True)
tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
cv2.imwrite(str(OUT / "overlay_all.png"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
pd.DataFrame(rows).to_csv(OUT / "overlay_err.csv", index=False)

print(f"saved -> {OUT}")