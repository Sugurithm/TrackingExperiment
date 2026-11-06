#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
link_share_whole_trial.py — 手先(MCP3)の変位に対する各リンクの射影寄与率（解析区間 40 s 全体）

  手先の変位 tip = shoulder + upper_arm + forearm + hand
    shoulder  = 肩の変位
    upper_arm = 肘 - 肩、forearm = 手首 - 肘、hand = MCP3 - 手首 の相対変位
  寄与率 share_k = Σ(v_k · tip) / Σ|tip|²   （4 リンクの合計 = 1）

出力: <cfg.OUT>/link_share_whole_trial/
  all_link_share.csv     試行ごとの寄与率と RMS
  person_link_share.csv  被験者 × 帯域ごとの平均・SD
  link_share.png         被験者ごとの寄与率（slow / fast）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import analysis_config as cfg  # noqa: E402
from utilities import iter_config_trials, load_trial, person_order, save_fig  # noqa: E402

# ---------- 設定 ----------
OUT_DIR = HERE / cfg.OUT / "link_share_whole_trial"
# リンク名: (近位の関節, 遠位の関節)。shoulder だけは肩そのものの変位（近位なし）
LINKS = {
    "shoulder":  (None, "shoulder"),
    "upper_arm": ("shoulder", "elbow"),
    "forearm":   ("elbow", "wrist"),
    "hand":      ("wrist", "mcp3"),
}
SHARE_COLS = [f"share_{k}" for k in LINKS]
DPI = 120


# ---------- 解析手法 ----------
def link_share(trial_dir):
    """1 試行の各リンクの寄与率と RMS [mm]"""
    data = load_trial(trial_dir, fc=cfg.FC)
    link = {k: data.disp[b] if a is None else data.relative(a, b) for k, (a, b) in LINKS.items()}
    tip = data.disp["mcp3"]                      # = 4 リンクの和
    den = np.sum(tip * tip)

    row = {"fs": data.fs, "tip_rms_mm": np.sqrt(den / len(tip))}
    for k, v in link.items():
        row[f"share_{k}"] = np.sum(v * tip) / den
        row[f"rms_{k}_mm"] = np.sqrt(np.mean(np.sum(v ** 2, axis=1)))
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


# ---------- 図 ----------
def plot_link_share(path, df):
    """人ごとの寄与率（左: slow、右: fast）"""
    order, group_of = person_order(df)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, band in zip(axs, ["slow", "fast"]):
        g = df[df.band == band].groupby("person")[SHARE_COLS].agg(["mean", "std"]).reindex(order)
        for i, (name, col) in enumerate(zip(LINKS, SHARE_COLS)):
            ax.bar(np.arange(len(order)) + (i - 1.5) * 0.2, g[(col, "mean")], 0.2, yerr=g[(col, "std")], capsize=2, label=name.replace("_", " "))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{p}\n({group_of[p]})" for p in order])
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(band)
        ax.grid(alpha=0.3)
    for ax in axs:
        ax.set_xlabel("participant")
    axs[0].set_ylabel("share of tip displacement")
    axs[0].legend(ncol=4)
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