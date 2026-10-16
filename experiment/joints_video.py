"""
joints_video.py
────────────────────────────────────────────────────────────────────
rs_joints.py の結果を RGB 画像に重ねた確認動画を作る（MediaPipe はやり直さない）．
sync_check.py と mocap_joints.py の後なら，MoCap の関節もカメラ画像に投影して重ねる．

  python experiment/joints_video.py                         # 最新の trial_*
  python experiment/joints_video.py recordings/trial_XXXX
  python experiment/joints_video.py recordings/trial_XXXX --show            # 書き出さずに画面で見る
  python experiment/joints_video.py recordings/trial_XXXX --start 30 --end 40   # 記録の 30〜40 s だけ

入力:  trial_*/session.db3（なければ rgb/ + frame_timestamps.csv）
       trial_*/joints/rs_joints.npz
       （あれば）trial_*/joints/mc_joints.npz と sync_check/sync_result.json
出力:  trial_*/joints/joints_check.mp4

【画面の見方】
  ● 橙・塗りつぶし : RealSense の関節（3 次元座標あり）
  ○ 橙・輪郭だけ   : 2 次元では見つかったが深度が取れなかった
  × 緑             : MoCap の関節を画像に投影した位置
  左上             : 記録開始からの秒数，区間（SYNC = ボードを振った区間 / OP = 動作区間 / -- = それ以外），
                     RealSense と MoCap の 3 次元の差 [mm]
  --show のとき: スペースで一時停止，q / Esc で終了
────────────────────────────────────────────────────────────────────
"""

import argparse

import numpy as np
import cv2

import config
from lib import io_utils, sync_io
from lib.geometry import apply, interp_nan, project
from lib.rs_device import open_color
from lib.viz import draw_joints

MATCH_TOL = 0.002         # 動画のフレームと rs_joints.npz の時刻をこの差以内で対応づける [s]


def mocap_in_camera(trial_dir, rs, t):
    """MoCap の関節をカメラ座標にして返す {RS 側の関節名: (frames × 3)}．同期前なら空．"""
    sync, calib, T_MC = sync_io.load_sync(trial_dir)
    mc_path = trial_dir / "joints" / "mc_joints.npz"
    if sync is None or not mc_path.is_file():
        print("[INFO] sync_check.py / mocap_joints.py の結果がないので MoCap は重ねません")
        return {}, None
    mc = np.load(mc_path)
    mc_names = list(mc["names"])
    take_t = t + sync_io.take_offset(sync, t)
    T_CM = np.linalg.inv(T_MC)
    out = {}
    for a, b in config.COMPARE_PAIRS:
        if a in rs["names"] and b in mc_names and a not in out:
            out[a] = apply(T_CM, interp_nan(take_t, mc["t"], mc["P_M"][:, mc_names.index(b)], config.CMP_MAX_GAP))
    print(f"[MoCap] {calib.name} の T_MC で投影: {', '.join(out)}")
    return out, sync


def label_of(sync, t):
    """各フレームの区間名．"""
    if sync is None:
        return np.full(len(t), "", dtype=object)
    lab = np.where(sync_io.operation_mask(sync, t), "OP", "--").astype(object)   # "SYNC" が切れないように
    for s in sync["segments"]:
        lab[(t >= s["rs_t0"]) & (t <= s["rs_t1"])] = "SYNC"
    return lab


def main():
    ap = argparse.ArgumentParser(description="rs_joints の結果を重ねた確認動画")
    ap.add_argument("trial", nargs="?", help="trial_* フォルダ（省略すると最新）")
    ap.add_argument("--show", action="store_true", help="書き出さずに画面で見る")
    ap.add_argument("--start", type=float, default=0.0, help="記録開始からこの秒数以降 [s]")
    ap.add_argument("--end", type=float, default=np.inf, help="記録開始からこの秒数まで [s]")
    args = ap.parse_args()

    trial_dir = io_utils.resolve_session_dir(args.trial, config.TRIAL_PREFIX)
    path = trial_dir / "joints" / "rs_joints.npz"
    if not path.is_file():
        raise SystemExit(f"[ERROR] {path} がありません．先に rs_joints.py を実行してください")
    rs = {k: v for k, v in np.load(path).items()}
    rs["names"] = list(rs["names"])
    t = rs["t"]
    print(f"[TRIAL] {trial_dir}  {len(t)} frames")
    mc_C, sync = mocap_in_camera(trial_dir, rs, t)
    labels = label_of(sync, t)

    writer, out = None, trial_dir / "joints" / "joints_check.mp4"
    n = 0
    paused = False
    for stamp, img, K, D in open_color(trial_dir)():
        rel = stamp - t[0]
        if rel < args.start:
            continue
        if rel > args.end:
            break
        k = int(np.clip(np.searchsorted(t, stamp), 1, len(t) - 1))
        k = k if abs(t[k] - stamp) < abs(t[k - 1] - stamp) else k - 1
        if abs(t[k] - stamp) > MATCH_TOL:          # rs_joints で深度がなく飛ばしたフレーム
            uv, has3d, errs, mc_uv = np.full((len(rs["names"]), 2), np.nan), np.zeros(len(rs["names"]), bool), "", {}
        else:
            uv, P = rs["uv"][k], rs["P_C"][k]
            has3d = np.isfinite(P[:, 0])
            mc_uv = {a: project(Q[k][None], K, D)[0] if np.isfinite(Q[k, 0]) else np.full(2, np.nan)
                     for a, Q in mc_C.items()}
            errs = "  ".join(f"{a} {np.linalg.norm(P[rs['names'].index(a)] - Q[k]):.0f}"
                             for a, Q in mc_C.items() if np.isfinite(Q[k, 0]) and has3d[rs["names"].index(a)])
        lines = [f"{rel:6.2f} s  {labels[k]}", "err[mm] " + errs if errs else ""]
        vis = draw_joints(img, rs["names"], uv, has3d, mc_uv, lines)

        if args.show:
            cv2.imshow("joints_video (space: pause, q: quit)", vis)
            key = cv2.waitKey(0 if paused else max(1, int(1000 / config.FPS))) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
        else:
            if writer is None:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), config.FPS,
                                         (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        n += 1
        if n % 300 == 0:
            print(f"\r  {n} frames", end="", flush=True)
    print()
    if writer is not None:
        writer.release()
        print(f"[DONE] {n} frames -> {out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
