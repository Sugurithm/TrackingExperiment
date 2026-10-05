# Motive(MoCap) と RealSense の座標系を合わせる変換行列 T_mocap_cam を求める．
#   p_mocap = T_mocap_cam @ p_cam   （同次座標, 単位 m）
#
# 入力: ボードを静止させた姿勢ごとに Motive の .csv と RealSense の .db3 を1組ずつ．
# 手順:
#   1. Motive : 各フレームの3点を距離で S1,S2,S3 に識別し，フレーム方向の中央値をとる．
#   2. RealSense: 各フレームで PnP → ボード座標の S1,S2,S3 をカメラ座標へ写し，中央値をとる．
#   3. 全姿勢の対応点（3点 × 姿勢数）から剛体変換を最小二乗で推定（Kabsch）．
#   4. 残差・Leave-one-pose-out 誤差・静止度・マーカ間距離を表示して JSON に保存．
# マーカ位置そのもの（Motive の生の計測値）を対応点にするので，MoCap 側でボード座標系を
# 組み立てる必要はない．
#
# POSES を編集して実行する．

import datetime
import json
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

import board_lib as bl

RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"
CALIB_DIR = RECORDINGS_DIR / "calib"

# 姿勢ごとの (Motive CSV, RealSense db3)．CALIB_DIR からの相対パスでも絶対パスでもよい．
POSES = [
    {"csv": "pose1.csv", "db3": "pose1/session.db3"},
    {"csv": "pose2.csv", "db3": "pose2/session.db3"},
    {"csv": "pose3.csv", "db3": "pose3/session.db3"},
    {"csv": "pose4.csv", "db3": "pose4/session.db3"},
    {"csv": "pose5.csv", "db3": "pose5/session.db3"},
]
OUTPATH = CALIB_DIR / "T_mocap_cam.json"

MARKER_TOL = 0.010     # 公称マーカ間距離からの許容ずれ [m]（識別用）
RS_STRIDE = 1          # RealSense は何フレームおきに処理するか
STATIC_WARN = 0.002    # フレーム間ばらつき(std)がこれを超えたら「静止していない」警告 [m]


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else CALIB_DIR / p


def mocap_markers(path):
    """戻り値: (S (3,3) 中央値, std (3,), 採用フレーム数, 全フレーム数)"""
    _, frames, _ = bl.load_motive_csv(path)
    ids = [bl.identify_markers(f, MARKER_TOL) for f in frames]
    ids = np.array([s for s in ids if s is not None])
    if len(ids) == 0:
        raise RuntimeError(f"{path}: S1-S3 を識別できたフレームがない")
    return np.median(ids, 0), ids.std(0).max(1), len(ids), len(frames)


def rs_markers(path):
    """戻り値: (S_cam (3,3) 中央値, std (3,), 検出フレーム数, 全フレーム数, rms[px], dZ[m])
    dZ は depth 実測 − PnP 予測 の中央値（depth のバイアス確認用，推定には使わない）．"""
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, str(path), repeat_playback=False)
    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)
    profile.get_device().as_playback().set_real_time(False)   # フレームを落とさない
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics

    S, rms, dz, n = [], [], [], 0
    try:
        while True:
            ok, frames = pipeline.try_wait_for_frames(timeout_ms=3000)
            if not ok:
                break                                 # ファイル終端
            n += 1
            if (n - 1) % RS_STRIDE:
                continue
            frames = align.process(frames)
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color:
                continue
            res = bl.board_pose(np.asanyarray(color.get_data()), intr)
            if res is None:
                continue
            R, t, px, r = res
            S.append(bl.MARKERS_BOARD @ R.T + t)
            rms.append(r)
            if depth:
                P = bl.OBJP @ R.T + t
                d = [depth.get_distance(int(round(u)), int(round(v))) - p[2]
                     for (u, v), p in zip(px, P)
                     if depth.get_distance(int(round(u)), int(round(v))) > 0.1]
                if d:
                    dz.append(np.median(d))
    finally:
        pipeline.stop()
    if not S:
        raise RuntimeError(f"{path}: チェッカーボードを検出できたフレームがない")
    S = np.array(S)
    return (np.median(S, 0), S.std(0).max(1), len(S), n,
            float(np.median(rms)), float(np.median(dz)) if dz else float("nan"))


def main():
    cam, moc, info = [], [], []
    for i, pose in enumerate(POSES, 1):
        csv_path, db3_path = resolve(pose["csv"]), resolve(pose["db3"])
        print(f"[POSE {i}] {csv_path.name} / {db3_path}")
        Sm, sd_m, nm, Nm = mocap_markers(csv_path)
        Sc, sd_c, nc, Nc, rms, dz = rs_markers(db3_path)

        dm = bl.nominal_distances(Sm) * 1e3
        dc = bl.nominal_distances(Sc) * 1e3
        print(f"  Motive   : {nm}/{Nm} frames  std(max)={sd_m.max()*1e3:.2f}mm  "
              f"|S1S2|,|S1S3|,|S2S3|=({dm[0]:.1f}, {dm[1]:.1f}, {dm[2]:.1f})mm")
        print(f"  RealSense: {nc}/{Nc} frames  std(max)={sd_c.max()*1e3:.2f}mm  "
              f"rms={rms:.2f}px  dZ(depth-PnP)={dz*1e3:+.1f}mm")
        if sd_m.max() > STATIC_WARN or sd_c.max() > STATIC_WARN:
            print("  [WARN] 記録中にボードが動いている可能性あり")
        cam.append(Sc)
        moc.append(Sm)
        info.append({"csv": str(csv_path), "db3": str(db3_path),
                     "mocap_frames": [nm, Nm], "rs_frames": [nc, Nc],
                     "mocap_std_mm": (sd_m * 1e3).tolist(), "rs_std_mm": (sd_c * 1e3).tolist(),
                     "mocap_dist_mm": dm.tolist(), "rs_dist_mm": dc.tolist(),
                     "rs_reproj_rms_px": rms, "rs_depth_minus_pnp_mm": dz * 1e3})

    cam, moc = np.array(cam), np.array(moc)               # (姿勢数, 3, 3)
    A, B = cam.reshape(-1, 3), moc.reshape(-1, 3)
    T = bl.fit_rigid(A, B)
    err = np.linalg.norm(bl.transform(T, A) - B, axis=1).reshape(len(POSES), 3)

    # 鏡像チェック: Motive 側が左手系（エクスポート設定の誤り等）なら鏡像の方がよく合う
    Am = A * [-1, 1, 1]
    err_m = np.linalg.norm(bl.transform(bl.fit_rigid(Am, B), Am) - B, axis=1)
    if np.sqrt((err_m ** 2).mean()) < 0.5 * np.sqrt((err ** 2).mean()):
        print("[WARN] 鏡像変換の方がよく合う．Motive の座標系が左手系になっていないか確認する")

    # Leave-one-pose-out: 1姿勢を抜いて推定し，抜いた姿勢で評価（汎化誤差の目安）
    loo = []
    for k in range(len(POSES)):
        m = np.arange(len(POSES)) != k
        Tk = bl.fit_rigid(cam[m].reshape(-1, 3), moc[m].reshape(-1, 3))
        loo.append(np.linalg.norm(bl.transform(Tk, cam[k]) - moc[k], axis=1))
    loo = np.array(loo)

    print("\nT_mocap_cam (p_mocap = T @ p_cam, [m]) =")
    print(np.array2string(T, precision=5, suppress_small=True))
    print(f"camera position in MoCap = {np.array2string(T[:3, 3], precision=4)} m")
    print("\nresidual [mm]       " + "  ".join(f"{s:>5}" for s in bl.MARKER_NAMES) + "  | LOO")
    for k in range(len(POSES)):
        print(f"  pose{k + 1}             " + "  ".join(f"{e * 1e3:5.1f}" for e in err[k])
              + "  | " + "  ".join(f"{e * 1e3:5.1f}" for e in loo[k]))
    print(f"  RMS (fit) = {np.sqrt((err ** 2).mean()) * 1e3:.2f} mm   "
          f"RMS (LOO) = {np.sqrt((loo ** 2).mean()) * 1e3:.2f} mm")

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPATH.write_text(json.dumps({
        "T_mocap_cam": T.tolist(),
        "convention": "p_mocap = T_mocap_cam @ [p_cam; 1], unit m, "
                      "cam = RealSense color optical frame",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "marker_center_z_m": bl.MARKER_CENTER_Z,
        "residual_mm": (err * 1e3).tolist(),
        "residual_rms_mm": float(np.sqrt((err ** 2).mean()) * 1e3),
        "loo_mm": (loo * 1e3).tolist(),
        "loo_rms_mm": float(np.sqrt((loo ** 2).mean()) * 1e3),
        "points_cam_m": cam.tolist(),
        "points_mocap_m": moc.tolist(),
        "poses": info,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE] -> {OUTPATH}")


if __name__ == "__main__":
    main()
