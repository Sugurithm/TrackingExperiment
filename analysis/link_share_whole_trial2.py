#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
link_share_whole_trial2.py — 手先(MCP3)の速度に対する各リンクの射影寄与率（解析区間 40 s 全体）

  手先の速度 v_tip = v_shoulder + v_upper_arm + v_forearm + v_hand
    shoulder  = 肩の速度
    upper_arm = 肘 - 肩、forearm = 手首 - 肘、hand = MCP3 - 手首 の相対速度
  寄与率 share_vel_k = Σ(v_k · v_tip) / Σ|v_tip|²   （4 リンクの合計 = 1）
  リンクの定義と計算本体は utilities（LINKS, TrialData.link_vel, projection_shares）。
  方向別版（joint_use_by_*_direction.py）の dir == "all" の share_proj_* と同じ値になる。

  速度にする理由
    - 方向別版（③）と同じ量になり、②と③をそのまま比べられる
    - 変位は「試行区間の平均位置」からのずれなので、基準の取り方（姿勢のゆっくりした漂いなど）に影響される。
      速度は基準によらない
    - 速度は速い成分の重みが大きくなる（周波数 f の成分は振幅が 2πf 倍）。
      変位と結果が違えば「遅い動きと速い動きで使うリンクが違う」ことを示す

  比較用に、変位の寄与率 share_disp_k = Σ(x_k · x_tip) / Σ|x_tip|² も CSV に残す（図は SIGNAL で選ぶ）。

出力: <cfg.OUT>/link_share_whole_trial/
  all_link_share.csv     試行ごとの寄与率（share_vel_*, share_disp_*）と RMS
  person_link_share.csv  被験者 × 帯域ごとの平均・SD
  link_share_vel.png     被験者ごとの寄与率（左 = slow，右 = fast，色 = リンク，棒 = 試行の平均，黒点 = 各試行）
  link_share_disp.png    同じ図の変位版（SIGNAL に "disp" を入れたとき）
  vel_vs_disp.png        速度と変位の寄与率の比較（横軸 = 変位，縦軸 = 速度，点 = 被験者 × 帯域の平均）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import analysis_config as cfg
from utilities import (BANDS, DPI, LINKS, TIP, iter_config_trials, load_trial, person_order, projection_shares, save_fig)

# ---------- 設定 ----------
OUT_DIR = HERE / cfg.OUT / "link_share_whole_trial"
SIGNALS = ["vel", "disp"]      # 図を出す量．"vel" = 速度（主），"disp" = 変位（比較用）
YLIM = (-0.1, 0.8)             # 寄与率の図の縦軸．None = 自動
SHARE_COLS = {s: [f"share_{s}_{k}" for k in LINKS] for s in ["vel", "disp"]}


# ---------- 解析手法 ----------
def link_share(trial_dir):
    """1 試行の各リンクの寄与率（速度・変位）と RMS"""
    data = load_trial(trial_dir, fc=cfg.FC)
    V, v_tip = data.link_vel(), data.vel(TIP)     # 速度 [mm/s]．Σ V = v_tip
    X, x_tip = data.link_disp(), data.disp[TIP]   # 変位 [mm]．Σ X = x_tip

    def rms(a):
        return float(np.sqrt(np.mean(np.sum(a ** 2, axis=1))))

    row = {"fs": data.fs, "tip_rms_mm_s": rms(v_tip), "tip_rms_mm": rms(x_tip)}
    share_v, share_x = projection_shares(V, v_tip), projection_shares(X, x_tip)
    for k in LINKS:
        row[f"share_vel_{k}"] = share_v[k]
        row[f"share_disp_{k}"] = share_x[k]
        row[f"rms_vel_{k}_mm_s"] = rms(V[k])
        row[f"rms_disp_{k}_mm"] = rms(X[k])
    return row


def summarize(df):
    """被験者 × 帯域ごとの寄与率の平均・SD"""
    cols = SHARE_COLS["vel"] + SHARE_COLS["disp"]
    person = df.groupby(["person", "group", "band"])[cols].agg(["mean", "std"]).reset_index()
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
def plot_link_share(path, df, signal):
    """人ごとの寄与率。棒 = 試行の平均、黒点 = 各試行。signal = "vel" / "disp" """
    order, group_of = person_order(df)
    x_of = {p: i for i, p in enumerate(order)}
    width = 0.2
    fig, axs = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, band in zip(axs, BANDS):
        d = df[df.band == band]
        for i, (name, col) in enumerate(zip(LINKS, SHARE_COLS[signal])):
            off = (i - 1.5) * width
            mean = d.groupby("person")[col].mean().reindex(order)
            ax.bar(np.arange(len(order)) + off, mean, width, label=name.replace("_", " "))
            ax.scatter([x_of[p] + off for p in d.person], d[col], s=8, color="k", alpha=0.7, zorder=3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{p}\n({group_of[p]})" for p in order])
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(band)
        ax.grid(alpha=0.3)
        ax.set_xlabel("participant")
        if YLIM is not None:
            ax.set_ylim(*YLIM)
    axs[0].set_ylabel(f"Link Share ({'velocity' if signal == 'vel' else 'displacement'})")
    axs[0].legend(ncol=4)
    save_fig(fig, path, DPI)


def plot_vel_vs_disp(path, person):
    """横軸 = 変位の寄与率，縦軸 = 速度の寄与率。点 = 被験者 × 帯域の平均，色 = リンク，形 = 帯域"""
    fig, ax = plt.subplots(figsize=(5, 5))
    marker = {"slow": "o", "fast": "^"}
    for i, k in enumerate(LINKS):
        for band in BANDS:
            d = person[person.band == band]
            ax.scatter(d[f"share_disp_{k}_mean"], d[f"share_vel_{k}_mean"], color=f"C{i}",
                       marker=marker[band], s=28, alpha=0.8,
                       label=f"{k.replace('_', ' ')} ({band})")
    lim = YLIM if YLIM is not None else ax.get_xlim()
    ax.plot(lim, lim, color="gray", lw=0.8, ls="--")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("share (displacement)")
    ax.set_ylabel("share (velocity)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    save_fig(fig, path, DPI)


# ---------- メイン ----------
def main():
    df = collect()
    person = summarize(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "all_link_share.csv", index=False)
    person.to_csv(OUT_DIR / "person_link_share.csv", index=False)
    for s in SIGNALS:
        plot_link_share(OUT_DIR / f"link_share_{s}.png", df, s)
    plot_vel_vs_disp(OUT_DIR / "vel_vs_disp.png", person)

    print(df.groupby("band")[SHARE_COLS["vel"] + SHARE_COLS["disp"]].mean().round(3).T.to_string())


if __name__ == "__main__":
    main()