#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xcorr_trial_simple.py — 関節の変位の図だけを出力する簡易版

  python joint_analysis/xcorr_trial_simple.py recordings/aim_xxx/trial_01_slow_s1
  python joint_analysis/xcorr_trial_simple.py recordings/aim_xxx            # セッション内の全試行
  python joint_analysis/xcorr_trial_simple.py recordings                    # 全セッションの全試行

出力: <out>/<セッション名>/<試行名>/
  combined_disp.png             全関節の変位（X, Y, Z ごとに重ねる）
  combined_disp_norm.png        全関節の変位の大きさ（ノルム）
  pair_relative_disp_norm.png   隣り合う関節の相対変位の大きさ
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ---------- 設定（数値はここで変更する） ----------
DEFAULT_PATH = "recordings"                                   # 入力フォルダ
DEFAULT_OUT = "analysis"                                      # 出力フォルダ
DEFAULT_JOINTS = ["shoulder", "elbow", "wrist", "mcp3"]       # 関節（連結の順）

DEFAULT_FC = 5.0          # 低域通過の遮断周波数 [Hz]
FILTER_ORDER = 4          # フィルタの次数
MAX_NORM_FREQ = 0.99      # 遮断周波数の上限（ナイキスト周波数に対する比）

Y_RANGE_DISP = (-150, 150)   # combined_disp の Y 軸 [mm]
Y_RANGE_NORM = (0, 200)      # combined_disp_norm の Y 軸 [mm]
Y_RANGE_PAIR = (0, 120)      # pair_relative_disp_norm の Y 軸 [mm]

FIGSIZE_DISP = (12, 7.5)     # combined_disp の図の大きさ
FIGSIZE_NORM = (12, 4)       # combined_disp_norm の図の大きさ
PAIR_FIG_WIDTH = 12          # pair 図の幅
PAIR_FIG_ROW_HEIGHT = 2.5    # pair 図の 1 段あたりの高さ
PAIR_FIG_MARGIN = 0.8        # pair 図の余白分の高さ

DPI = 110                    # 画像の解像度
GRID_ALPHA = 0.3             # グリッド線の濃さ


# ---------- 試行フォルダを探す ----------
def find_trials(path):
    p = Path(path)
    if (p / "joints" / "rs_joints.npz").exists():      # 試行そのもの
        return [p]
    trials = []
    for d in sorted(p.iterdir()):
        if not d.is_dir():
            continue
        if (d / "joints" / "rs_joints.npz").exists():  # p がセッション
            trials.append(d)
        elif d.name.lower().startswith("aim"):         # p が recordings
            trials += [t for t in sorted(d.iterdir()) if (t / "joints" / "rs_joints.npz").exists()]
    return trials


# ---------- データの読み込みと前処理 ----------
def load_displacement(trial, joints, fc):
    """各関節の変位（平均を引いた位置）を返す。 戻り値: 時間, {関節名: (N,3)}"""
    z = np.load(trial / "joints" / "rs_joints.npz")
    names = [str(n) for n in z["names"]]
    t = z["t"].astype(float)
    P = z["P_C"].astype(float)

    # 時間順に並べる
    order = np.argsort(t)
    t = t[order]
    P = P[order]

    fs = 1.0 / np.median(np.diff(t))              # 標本化周波数
    grid = np.arange(t[0], t[-1], 1.0 / fs)        # 等間隔の時間

    b, a = butter(FILTER_ORDER, min(fc / (fs / 2), MAX_NORM_FREQ))     # 低域通過フィルタ

    disp = {}
    for j in joints:
        X = P[:, names.index(j), :]
        ok = np.isfinite(X).all(axis=1)            # 欠損(NaN)でないフレーム
        # 等間隔の時間に補間
        Y = np.column_stack([np.interp(grid, t[ok], X[ok, k]) for k in range(3)])
        Y = filtfilt(b, a, Y, axis=0)
        disp[j] = Y - Y.mean(axis=0)               # 平均を引いて変位にする
    return grid - grid[0], disp


# ---------- 図 ----------
def plot_combined_disp(path, joints, disp, tt):
    fig, axs = plt.subplots(3, 1, figsize=FIGSIZE_DISP, sharex=True)
    for k, label in enumerate("XYZ"):
        for j in joints:
            axs[k].plot(tt, disp[j][:, k], label=j)
        axs[k].set_ylabel(f"{label} [mm]")
        axs[k].set_ylim(*Y_RANGE_DISP)
        axs[k].grid(alpha=GRID_ALPHA)
    axs[0].legend(ncol=len(joints), loc="upper right")
    axs[-1].set_xlabel("time [s]")
    fig.suptitle("Displacement of all joints")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_combined_disp_norm(path, joints, disp, tt):
    fig, ax = plt.subplots(figsize=FIGSIZE_NORM)
    for j in joints:
        ax.plot(tt, np.linalg.norm(disp[j], axis=1), label=j)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("|displacement| [mm]")
    ax.set_ylim(*Y_RANGE_NORM)
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(ncol=len(joints), loc="upper right")
    fig.suptitle("Displacement magnitude of all joints")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_pair_relative_disp_norm(path, pairs, disp, tt):
    fig, axs = plt.subplots(len(pairs), 1, figsize=(PAIR_FIG_WIDTH, PAIR_FIG_ROW_HEIGHT * len(pairs) + PAIR_FIG_MARGIN), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, (a, b) in zip(axs, pairs):
        rel = disp[b] - disp[a]                    # 例: 肘の変位 - 肩の変位
        ax.plot(tt, np.linalg.norm(rel, axis=1), color="tab:purple")
        ax.set_ylabel(f"|{b} - {a}| [mm]")
        ax.set_ylim(*Y_RANGE_PAIR)
        ax.grid(alpha=GRID_ALPHA)
    axs[-1].set_xlabel("time [s]")
    fig.suptitle("Relative displacement magnitude between joints")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------- メイン ----------
def main():
    ap = argparse.ArgumentParser(description="関節の変位の図を出力する")
    ap.add_argument("path", nargs="?", default=DEFAULT_PATH, help="recordings / セッション / 試行のフォルダ")
    ap.add_argument("--out", default=DEFAULT_OUT, help="出力先")
    ap.add_argument("--joints", nargs="+", default=DEFAULT_JOINTS)
    ap.add_argument("--fc", type=float, default=DEFAULT_FC, help="低域通過 [Hz]")
    args = ap.parse_args()

    trials = find_trials(args.path)
    if not trials:
        raise SystemExit(f"試行が見つかりません: {args.path}")

    pairs = list(zip(args.joints[:-1], args.joints[1:]))   # 隣どうしのペア

    for trial in trials:
        out = Path(args.out) / trial.parent.name / trial.name
        out.mkdir(parents=True, exist_ok=True)

        tt, disp = load_displacement(trial, args.joints, args.fc)

        plot_combined_disp(out / "combined_disp.png", args.joints, disp, tt)
        plot_combined_disp_norm(out / "combined_disp_norm.png", args.joints, disp, tt)
        plot_pair_relative_disp_norm(out / "pair_relative_disp_norm.png", pairs, disp, tt)
        print(f"完了: {trial.parent.name}/{trial.name}")


if __name__ == "__main__":
    main()