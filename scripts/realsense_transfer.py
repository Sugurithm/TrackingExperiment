# rs_recorder.py の記録 (.db3, 1姿勢を約5秒録画) を pyrealsense2 で再生 -> T^C_B（ボード座標 -> カラー光学座標, mm）
import json
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs

# ====== 設定 =====
FILEPATH = Path("recordings/rec_20260831_145053/session.db3")
STRIDE = 2            # n フレームに1枚処理
MAX_ERR = 0.5         # 再投影誤差 [px]
TRIM = 0.5            # 録画の最初と最後を捨てる [s]
MIN_FRAMES = 5

SQUARE = 23.97
PATTERN = (7, 4)
MARKERS_B = {"S1": (-40, -40, 0), "S2": (190, -40, 0), "S3": (-40, 115, 0)}
OBJ = np.array([[j * SQUARE, i * SQUARE, 0] for i in range(PATTERN[1]) for j in range(PATTERN[0])], np.float32)

# ===== solvePnP =====
def solve(gray, K, D):
    ok, c = cv2.findChessboardCornersSB(gray, PATTERN, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ok:
        return None

    def pnp(c):
        _, r, t = cv2.solvePnP(OBJ, c, K, D, flags=cv2.SOLVEPNP_IPPE)
        return cv2.solvePnPRefineLM(OBJ, c, K, D, r, t)

    r, t = pnp(c)
    u, v = cv2.projectPoints(np.array([[-SQUARE / 2, -SQUARE / 2, 0.]]), r, t, K, D)[0].ravel().astype(int)
    if gray[v - 3:v + 4, u - 3:u + 4].mean() >= 128:   # 左上マスが白なら向きが逆
        c = c[::-1].copy()
        r, t = pnp(c)
    R = cv2.Rodrigues(r)[0]
    proj = cv2.projectPoints(OBJ, r, t, K, D)[0].reshape(-1, 2)
    err = np.linalg.norm(proj - c.reshape(-1, 2), axis=1).mean()
    if R[2, 2] <= 0 or err > MAX_ERR:
        return None
    return dict(R=R, t=t.ravel(), err=err)

# ===== 回転行列の平均化 =====
def mean_R(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


# ===== フレーム取得 =====
frames, imgs = [], []
cfg = rs.config()
rs.config.enable_device_from_file(cfg, str(FILEPATH), repeat_playback=False)
pipeline = rs.pipeline()
profile = pipeline.start(cfg)
profile.get_device().as_playback().set_real_time(False)   # 全フレームを落とさず再生

cs = profile.get_stream(rs.stream.color).as_video_stream_profile()
it = cs.get_intrinsics()
K = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1]])
D = np.array(it.coeffs)
print(cs.format(), it.width, it.height, K, D)

i = 0
while True:
    ok, fs = pipeline.try_wait_for_frames(timeout_ms=3000)
    if not ok:
        break
    cf = fs.get_color_frame()
    i += 1
    if not cf or i % STRIDE:
        continue
    img = np.asanyarray(cf.get_data()).copy()
    if cs.format() == rs.format.rgb8:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    res = solve(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), K, D)
    if res:
        res["stamp"] = cf.get_timestamp() / 1000.0   # [s]
        frames.append(res)
        imgs.append(img)
pipeline.stop()
print(f"detected {len(frames)} / {i} frames")

# ===== 両端を捨てて 1 姿勢にまとめる =====
t0, t1 = frames[0]["stamp"] + TRIM, frames[-1]["stamp"] - TRIM
s = [k for k, f in enumerate(frames) if t0 <= f["stamp"] <= t1]
assert len(s) >= MIN_FRAMES, f"フレーム不足: {len(s)}"
ts = np.array([frames[k]["t"] for k in s])
T = np.eye(4)
T[:3, :3] = mean_R([frames[k]["R"] for k in s])
T[:3, 3] = np.median(ts, axis=0)
rms = float(np.mean([frames[k]["err"] for k in s]))
print(f"used {len(s)} frames  tvec={T[:3, 3].round(2)}  std={ts.std(axis=0).round(3)} mm  rms={rms:.3f}px")

# ===== 保存 =====
# T_CB : ボード座標系（Board）からカメラ光学座標系（Color Camera）への4×4 同次変換行列
# K : カメラ内部パラメータ行列
# D : カメラ歪み係数
# tvec_std : 平均姿勢の推定誤差（標準偏差）[mm]
# rms_px : 平均姿勢の再投影誤差 [px]
# t_start, t_end : 姿勢推定に使用したフレームの最初と最後のタイムスタンプ [s]
out = FILEPATH.parent / "calib_rs"
out.mkdir(exist_ok=True)
json.dump(dict(
    T_CB=T.tolist(), K=K.tolist(), D=D.tolist(), n_frames=len(s),
    tvec_std=ts.std(axis=0).tolist(), rms_px=rms,
    t_start=frames[s[0]]["stamp"], t_end=frames[s[-1]]["stamp"]),
    open(out / "T_CB.json", "w"), indent=2)

# 目視確認: 軸とマーカー位置を投影
vis = imgs[s[len(s) // 2]].copy()
rv, tv = cv2.Rodrigues(T[:3, :3])[0], T[:3, 3].reshape(3, 1)
cv2.drawFrameAxes(vis, K, D, rv, tv, 50)
for name, p in MARKERS_B.items():
    u, v = cv2.projectPoints(np.array([p], float), rv, tv, K, D)[0].ravel()
    cv2.circle(vis, (int(u), int(v)), 10, (0, 0, 255), 2)
    cv2.putText(vis, name, (int(u) + 12, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imwrite(str(out / "pose_check.png"), vis)
print(f"saved -> {out}")