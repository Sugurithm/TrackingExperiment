#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_joint_displacement.py — 試行ごとの関節の変位の時系列を図にする（解析区間 40 s）

対象の試行・遮断周波数・やり直しの扱いは analysis_config（SESSIONS, FC, ATTEMPT）に従う。

出力: <cfg.OUT>/plot_joint_displacement/<セッション名>/<試行名>/
  combined_disp.png             全関節の変位（X, Y, Z ごとに重ねる）
  combined_disp_norm.png        全関節の変位の大きさ（ノルム）
  pair_relative_disp_norm.png   隣り合う関節の相対変位の大きさ
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import analysis_config as cfg  # noqa: E402
from utilities import iter_config_trials, load_trial, save_fig  # noqa: E402

# ---------- 設定 ----------
OUT_DIR = HERE / cfg.OUT / "plot_joint_displacement"

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


# ---------- 図 ----------
def plot_combined_disp(path, data):
    fig, axs = plt.subplots(3, 1, figsize=FIGSIZE_DISP, sharex=True)
    for k, label in enumerate("XYZ"):
        for j in data.joints:
            axs[k].plot(data.tt, data.disp[j][:, k], label=j)
        axs[k].set_ylabel(f"{label} [mm]")
        axs[k].set_ylim(*Y_RANGE_DISP)
        axs[k].grid(alpha=GRID_ALPHA)
    axs[0].legend(ncol=len(data.joints), loc="upper right")
    axs[-1].set_xlabel("time [s]")
    fig.suptitle("Displacement of all joints")
    save_fig(fig, path, DPI)


def plot_combined_disp_norm(path, data):
    fig, ax = plt.subplots(figsize=FIGSIZE_NORM)
    for j in data.joints:
        ax.plot(data.tt, data.norm(j), label=j)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("|displacement| [mm]")
    ax.set_ylim(*Y_RANGE_NORM)
    ax.grid(alpha=GRID_ALPHA)
    ax.legend(ncol=len(data.joints), loc="upper right")
    fig.suptitle("Displacement magnitude of all joints")
    save_fig(fig, path, DPI)


def plot_pair_relative_disp_norm(path, data):
    pairs = data.pairs()
    fig, axs = plt.subplots(len(pairs), 1, sharex=True,
                            figsize=(PAIR_FIG_WIDTH, PAIR_FIG_ROW_HEIGHT * len(pairs) + PAIR_FIG_MARGIN))
    axs = np.atleast_1d(axs)
    for ax, (a, b) in zip(axs, pairs):
        ax.plot(data.tt, np.linalg.norm(data.relative(a, b), axis=1), color="tab:purple")
        ax.set_ylabel(f"|{b} - {a}| [mm]")
        ax.set_ylim(*Y_RANGE_PAIR)
        ax.grid(alpha=GRID_ALPHA)
    axs[-1].set_xlabel("time [s]")
    fig.suptitle("Relative displacement magnitude between joints")
    save_fig(fig, path, DPI)


# ---------- メイン ----------
def main():
    for ses, trial in iter_config_trials(cfg, ROOT):
        out = OUT_DIR / ses.dir.name / trial.name
        out.mkdir(parents=True, exist_ok=True)

        data = load_trial(trial.dir, fc=cfg.FC)
        plot_combined_disp(out / "combined_disp.png", data)
        plot_combined_disp_norm(out / "combined_disp_norm.png", data)
        plot_pair_relative_disp_norm(out / "pair_relative_disp_norm.png", data)
        print(f"完了: {ses.label}/{trial.name}")


if __name__ == "__main__":
    main()