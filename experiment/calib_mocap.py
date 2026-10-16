"""
calib_mocap.py
────────────────────────────────────────────────────────────────────
較正フォルダの Motive Take CSV（全姿勢を 1 本で連続記録）から静止区間を切り出し，
姿勢ごとの T_MB（ボード → Motive 座標, mm）を求める．
（scripts/motive_transfer.py を較正フォルダ単位で動くようにしたもの）

  python experiment/calib_mocap.py                       # 最新の calib_*
  python experiment/calib_mocap.py recordings/calib_XXXX

入力:  calib_*/mocap/*.csv（1 つだけ置く）
出力:  calib_*/mocap/T_MB.json      … 静止区間ごとの T_MB とマーカー位置
                    /pose_check.png  … S1 の軌跡と切り出した静止区間（区間の数を目視で確認）
────────────────────────────────────────────────────────────────────
"""

import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # 画面のない環境でも保存だけできるようにする
import matplotlib.pyplot as plt

import config
from lib import io_utils
from lib.board import D_REF, PAIRS, board_frame, identify_all
from lib.mocap_io import Take



# ══════════════════════════════════════════════════
#  静止区間（窓内の振れ幅で判定）
# ══════════════════════════════════════════════════
def still_segments(S, t, fps):
    w = int(config.MC_WIN * fps)
    roll = pd.DataFrame(S.reshape(-1, 9)).rolling(w, center=True, min_periods=int(0.8 * w))
    still = ((roll.max() - roll.min()).max(axis=1) < config.MC_STILL).to_numpy()
    edge = np.diff(np.r_[0, still.astype(int), 0])

    segs = []
    for a, b in zip(np.where(edge == 1)[0], np.where(edge == -1)[0] - 1):
        # 短い途切れで位置も変わっていなければ前の区間と結合（誤識別フレームによる分断を防ぐ）
        if segs and t[a] - t[segs[-1][1]] < config.MC_MERGE and \
                np.nanmax(abs(np.nanmedian(S[a:b + 1], 0) - np.nanmedian(S[segs[-1][0]:segs[-1][1] + 1], 0))) < config.MC_OUTLIER:
            segs[-1][1] = b
        else:
            segs.append([a, b])
    n = int(config.MC_TRIM * fps)
    return [(a + n, b - n) for a, b in segs if t[b] - t[a] - 2 * config.MC_TRIM >= config.MC_MIN_LEN]


def summarize_segment(S, t, a, b):
    seg = S[a:b + 1][np.isfinite(S[a:b + 1, 0, 0])]
    seg = seg[np.linalg.norm(seg - np.median(seg, 0), axis=2).max(axis=1) < config.MC_OUTLIER]   # 外れフレーム除去
    med = np.median(seg, axis=0)
    T = board_frame(*med)
    err = [float(np.linalg.norm(med[j] - med[i]) - D) for (i, j), D in zip(PAIRS, D_REF)]
    # T_MB      : ボード座標 → Motive 座標 [mm]
    # markers_M : 静止区間での S1, S2, S3 の位置（中央値）[mm]
    # dist_err  : マーカー間距離の実測 - 設計 [mm]
    # t_start, t_end : 区間の時刻（テイク先頭から）[s]
    return dict(T_MB=T.tolist(), markers_M=med.tolist(), n_frames=len(seg),
                marker_std=seg.std(axis=0).tolist(), dist_err=err,
                t_start=float(t[a]), t_end=float(t[b]))


def save_check_plot(path, t, S, segs):
    plt.figure(figsize=(14, 4))
    plt.plot(t, S[:, 0], lw=0.8)
    for k, (a, b) in enumerate(segs):
        plt.axvspan(t[a], t[b], color="orange", alpha=0.3)
        plt.text((t[a] + t[b]) / 2, np.nanmax(S[:, 0]), str(k), ha="center")
    plt.legend(["S1 X", "S1 Y", "S1 Z"])
    plt.xlabel("time [s]")
    plt.ylabel("[mm]")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════
#  メイン
# ══════════════════════════════════════════════════
def main():
    calib_dir = io_utils.resolve_session_dir(sys.argv[1] if len(sys.argv) > 1 else None, config.CALIB_PREFIX)
    path = io_utils.find_one(calib_dir / "mocap", "*.csv")
    print(f"[CALIB] {calib_dir}\n[TAKE]  {path.name}")

    take = Take(path)
    t, fps, X = take.t, take.fps, take.markers()
    print(f"  {len(t)} frames  {X.shape[1]} markers  {fps:.0f} fps")

    S = identify_all(X)                          # 毎フレーム S1, S2, S3 を識別（見つからなければ NaN）
    print(f"  identified {np.isfinite(S[:, 0, 0]).sum()} / {len(t)} frames")

    segs = still_segments(S, t, fps)
    if not segs:                                 # 閾値（config の MC_*）が合っていない可能性
        raise SystemExit("[ERROR] 静止区間が見つかりません．config.py の MC_STILL / MC_MIN_LEN を確認してください")

    poses = []
    for k, (a, b) in enumerate(segs):
        p = summarize_segment(S, t, a, b)
        poses.append(p)
        print(f"  [{k}] t={p['t_start']:6.1f}-{p['t_end']:6.1f} s  {p['n_frames']} frames  "
              f"std={np.max(p['marker_std']):.2f} mm  dist_err={np.round(p['dist_err'], 2)} mm")

    out = calib_dir / "mocap"
    io_utils.save_json(out / "T_MB.json", dict(
        take=take.meta.get("Take Name"), csv=path.name, capture_start=take.meta.get("Capture Start Time"),
        print_scale=config.PRINT_SCALE, markers_B=config.MARKERS_B, poses=poses))
    save_check_plot(out / "pose_check.png", t, S, segs)
    print(f"[DONE] {len(poses)} 静止区間 -> {out / 'T_MB.json'}")


if __name__ == "__main__":
    main()
