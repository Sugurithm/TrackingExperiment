#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
link_share_whole_trial.py — 手先(MCP3)の変位に対する各リンクの射影寄与率（解析区間 40 s 全体）

  手先の変位 x_tip = x_shoulder + x_upper_arm + x_forearm + x_hand
    shoulder  = 肩の変位
    upper_arm = 肘 - 肩、forearm = 手首 - 肘、hand = MCP3 - 手首 の相対変位
  寄与率 share_disp_k = Σ(x_k · x_tip) / Σ|x_tip|²   （4 リンクの合計 = 1）
  リンクの定義と計算本体は utilities（LINKS, TrialData.link_disp, projection_shares）。
  方向別版（joint_use_by_*_direction.py）の dir == "all" と同じ値になる。

出力: <cfg.OUT>/link_share_whole_trial/
  all_link_share.csv     試行ごとの寄与率（share_disp_*）と RMS
  person_link_share.csv  被験者 × 帯域ごとの平均・SD
  link_share.png         被験者ごとの寄与率（行 = slow/fast，列 = リンク，色 = 群，点 = 各試行）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import analysis_config as cfg  # noqa: E402
from utilities import (BANDS, DPI, GROUP_COLOR, LINKS, TIP, iter_config_trials,  # noqa: E402
                       load_trial, person_order, projection_shares, save_fig)

# ---------- 設定 ----------
OUT_DIR = HERE / cfg.OUT / "link_share_whole_trial"
SHARE_COLS = [f"share_disp_{k}" for k in LINKS]


# ---------- 解析手法 ----------
def link_share(trial_dir):
    """1 試行の各リンクの寄与率（変位）と RMS [mm]"""
    data = load_trial(trial_dir, fc=cfg.FC)
    X = data.link_disp()
    x_tip = data.disp[TIP]                       # = 4 リンクの和

    row = {"fs": data.fs, "tip_rms_mm": float(np.sqrt(np.mean(np.sum(x_tip ** 2, axis=1))))}
    for k, s in projection_shares(X, x_tip).items():
        row[f"share_disp_{k}"] = s
        row[f"rms_{k}_mm"] = float(np.sqrt(np.mean(np.sum(X[k] ** 2, axis=1))))
    return row


def summarize(df):
    """被験者 × 帯域ごとの寄与率の平均・SD"""
    person = df.groupby(["person", "group", "band"])[SHARE_COLS].agg(["mean", "std"]).reset_index()
    person.columns = ["_".join(c).strip("_") for c in person.columns]
    return person


# ---------- 収集 ----------
def collect():
    rows = []
    for ses, trial in iter_config_trials(cfg, ROOT):
        rows.append({"person": ses.person, "group": ses.group, "band": trial.band,
                     "seed": trial.seed, "trial": trial.name, **link_share(trial.dir)})
        print(f"完了: {ses.person}/{trial.name}")
    return pd.DataFrame(rows)


# ---------- 図（方向別版と同じ割り付け：行 = 帯域，列 = リンク，色 = 群） ----------
def plot_link_share(path, df):
    """横軸 = 被験者。棒 = 試行の平均（色 = 群），点 = 各試行"""
    order, group_of = person_order(df)
    x_of = {p: i for i, p in enumerate(order)}
    colors = [GROUP_COLOR.get(group_of[p], "0.5") for p in order]

    fig, axs = plt.subplots(len(BANDS), len(LINKS), figsize=(3.6 * len(LINKS), 3.3 * len(BANDS)),
                            sharex=True, sharey=True, squeeze=False)
    for r, band in enumerate(BANDS):
        d = df[df.band == band]
        for c, (name, col) in enumerate(zip(LINKS, SHARE_COLS)):
            ax = axs[r, c]
            mean = d.groupby("person")[col].mean().reindex(order)
            ax.bar(range(len(order)), mean, 0.7, color=colors, alpha=0.8)
            ax.scatter([x_of[p] for p in d.person], d[col], s=10, color="k", alpha=0.6, zorder=3)
            ax.axhline(0, color="gray", lw=0.6)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(name.replace("_", " "))
            if c == 0:
                ax.set_ylabel(f"{band}\nprojection share")
    for ax in axs[-1]:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=8)
    axs[0, -1].legend(handles=[Patch(color=v, label=k) for k, v in GROUP_COLOR.items()], fontsize=8)
    fig.supxlabel("participant")
    save_fig(fig, path, DPI)


# ---------- メイン ----------
def main():
    df = collect()
    person = summarize(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "all_link_share.csv", index=False)
    person.to_csv(OUT_DIR / "person_link_share.csv", index=False)
    plot_link_share(OUT_DIR / "link_share.png", df)

    print(df.groupby("band")[SHARE_COLS + ["fs"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()