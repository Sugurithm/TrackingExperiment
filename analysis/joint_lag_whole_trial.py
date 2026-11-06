#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_lag_whole_trial.py — 隣り合う関節の遅れ（変位ノルムの相互相関，解析区間 40 s 全体）

出力: <cfg.OUT>/joint_lag_whole_trial/
  all_lag_norm.csv   試行 × ペアごとの遅れとピーク相関
  person_lag.csv     被験者 × 帯域ごとの平均・SD
  summary.csv        ペアごとの ICC など
  lag_by_person.png  被験者ごとの遅れのばらつき
  slow_vs_fast.png   slow と fast の平均遅れの散布図
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
from utilities import (adjacent_pairs, icc_oneway, iter_config_trials, load_trial, person_order, save_fig)

# ---------- 設定 ----------
OUT_DIR = HERE / cfg.OUT / "joint_lag_whole_trial"

MAX_LAG = 0.5          # 相互相関でずらす最大範囲 [s]（±）
MIN_R = 0.0            # 有効判定: ピーク相関がこれ以上（正の相関のみ対象）
EDGE_MS = 10.0         # 有効判定: ピークが範囲の端からこの ms 以上内側にあること

BAND_COLOR = {"slow": "tab:blue", "fast": "tab:orange"}
DPI = 120


# ---------- 1. 解析手法 ----------
def xcorr_direct(a, b, max_lag):
    """
    1 次元時系列 a, b の正規化相互相関。ピークの位置（放物線補間でサブフレーム）と相関を返す。
    プラスのラグ = b（遠位）が a（近位）より遅れている。
    """
    n = len(a)
    a_norm = (a - np.mean(a)) / np.std(a)     # z スコア
    b_norm = (b - np.mean(b)) / np.std(b)

    lags = np.arange(-max_lag, max_lag + 1)   # ずらし量 [フレーム]
    corr = np.zeros(len(lags), dtype=float)
    for i, lag in enumerate(lags):
        if lag > 0:                           # b を後ろにずらす
            corr[i] = np.sum(a_norm[:-lag] * b_norm[lag:]) / n
        elif lag < 0:                         # b を前にずらす
            k = -lag
            corr[i] = np.sum(a_norm[k:] * b_norm[:-k]) / n
        else:
            corr[i] = np.sum(a_norm * b_norm) / n

    # ノルム同士なので正のピークのみを探す
    peak_idx = int(np.argmax(corr))
    peak_r = float(corr[peak_idx])

    shift = 0.0
    if 0 < peak_idx < len(corr) - 1:
        y0, y1, y2 = corr[peak_idx - 1], corr[peak_idx], corr[peak_idx + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            shift = 0.5 * (y0 - y2) / denom

    peak_lag = float(lags[peak_idx]) + shift
    return lags, corr, peak_lag, peak_r


def lag_norm_trial(trial_dir):
    """1 試行のペアごとの (ペア名, 遅れ [ms], ピーク相関)"""
    data = load_trial(trial_dir, fc=cfg.FC)
    max_lag = int(round(MAX_LAG * data.fs))

    out = []
    for a, b in adjacent_pairs(data.joints):
        _, _, peak, peak_r = xcorr_direct(data.norm(a), data.norm(b), max_lag)
        out.append((f"{a}-{b}", peak / data.fs * 1000, peak_r))
    return out




# ---------- 2. 収集と集計 ----------
def collect():
    """全試行の遅れを DataFrame にまとめ、有効判定の列 valid を付ける"""
    rows = []
    for ses, trial in iter_config_trials(cfg, ROOT):
        for pair, lag_ms, peak_r in lag_norm_trial(trial.dir):
            rows.append({
                "label": ses.label, 
                "person": ses.person, 
                "group": ses.group, 
                "dpi": ses.dpi, 
                "trial": trial.name, 
                "band": trial.band, 
                "seed": trial.seed, 
                "pair": pair, 
                "lag_ms": lag_ms, 
                "peak_r": peak_r
                })
        print(f"完了: {ses.label}/{trial.name}")

    df = pd.DataFrame(rows)
    # 正の相関が基準以上 かつ ピークが探索範囲の端に張り付いていない
    edge = MAX_LAG * 1000 - EDGE_MS
    df["valid"] = (df.peak_r >= MIN_R) & (df.lag_ms.abs() < edge)
    return df


def summarize(df):
    """被験者ごとの平均・SD と、ペアごとの統計指標"""
    rows_person, rows_sum = [], []
    for pair, d in df.groupby("pair", sort=False):
        ok = d[d.valid]
        for band in ["all", "slow", "fast"]:
            b = ok if band == "all" else ok[ok.band == band]
            for person, g in b.groupby("person"):
                rows_person.append({
                    "pair": pair, "band": band, "person": person,
                    "group": g.group.iloc[0], "n": len(g),
                    "mean_ms": g.lag_ms.mean(), "sd_ms": g.lag_ms.std(ddof=1),
                    "mean_r": g.peak_r.mean(), "sd_r": g.peak_r.std(ddof=1),
                })
        rows_sum.append({
            "pair": pair, "n_valid": len(ok), "n_excluded": len(d) - len(ok),
            "mean_r": ok.peak_r.mean(),
            "icc": icc_oneway(ok.lag_ms, ok.person) if ok.person.nunique() >= 2 else np.nan,
        })
    return pd.DataFrame(rows_person), pd.DataFrame(rows_sum)


# ---------- 3. 図 ----------
def _pair_title(ax, pair):
    a, b = pair.split("-")
    ax.set_title(f"{a} -> {b}")
    return b


def plot_by_person(path, df):
    """被験者ごとの遅れのばらつきと平均 ± SD"""
    persons, group_of = person_order(df)
    pairs = list(dict.fromkeys(df.pair))
    x = {p: i for i, p in enumerate(persons)}

    fig, axs = plt.subplots(len(pairs), 1, figsize=(max(10, 1.2 * len(persons)), 3.2 * len(pairs)),
                            sharex=True, squeeze=False)
    for ax, pair in zip(axs[:, 0], pairs):
        d = df[(df.pair == pair) & df.valid]
        for band, off in [("slow", -0.12), ("fast", 0.12)]:
            b = d[d.band == band]
            ax.scatter([x[p] + off for p in b.person], b.lag_ms, s=22, color=BAND_COLOR[band], alpha=0.8, label=band)
        g = d.groupby("person").lag_ms
        ax.errorbar([x[p] for p in g.mean().index], g.mean(), yerr=g.std(ddof=1), fmt="o",
                    color="black", ms=6, capsize=4, label="mean ± SD")
        ax.axhline(0, color="gray", lw=0.8)
        distal = _pair_title(ax, pair)
        ax.set_ylabel(f"lag [ms]\n(+ = {distal} later)")
        ax.grid(alpha=0.3)

    axs[0, 0].legend(loc="upper right", ncol=3, fontsize=9)
    axs[-1, 0].set_xlabel("participant")
    axs[-1, 0].set_xticks(range(len(persons)))
    axs[-1, 0].set_xticklabels([f"{p}\n({group_of[p]})" for p in persons])
    save_fig(fig, path, DPI)


def plot_slow_vs_fast(path, person_lag):
    """横軸 slow・縦軸 fast の被験者平均遅れ"""
    pairs = list(dict.fromkeys(person_lag.pair))
    fig, axs = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4.2), squeeze=False)
    for ax, pair in zip(axs[0], pairs):
        d = person_lag[person_lag.pair == pair]
        m = d.pivot(index="person", columns="band", values="mean_ms")
        m = m.dropna(subset=[c for c in ["slow", "fast"] if c in m])

        if {"slow", "fast"} <= set(m.columns) and len(m):
            ax.scatter(m.slow, m.fast, color="black")
            for p, r in m.iterrows():
                ax.annotate(p, (r.slow, r.fast), fontsize=8, xytext=(3, 3), textcoords="offset points")
            lim = [min(m.slow.min(), m.fast.min()) - 10, max(m.slow.max(), m.fast.max()) + 10]
            ax.plot(lim, lim, color="gray", lw=0.8, ls="--")

        _pair_title(ax, pair)
        ax.set_xlabel("slow: mean lag [ms]")
        ax.set_ylabel("fast: mean lag [ms]")
        ax.grid(alpha=0.3)
    save_fig(fig, path, DPI)


# ---------- メイン ----------
def main():
    df = collect()
    person_lag, summary = summarize(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "all_lag_norm.csv", index=False)
    person_lag.to_csv(OUT_DIR / "person_lag.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    plot_by_person(OUT_DIR / "lag_by_person.png", df)
    plot_slow_vs_fast(OUT_DIR / "slow_vs_fast.png", person_lag)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()