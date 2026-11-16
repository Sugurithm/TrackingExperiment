#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_lag_whole_trial.py — 隣り合う関節の遅れ（変位ノルムの相互相関，解析区間 40 s 全体）

出力: <cfg.OUT>/joint_lag_whole_trial/
  all_lag_norm.csv   試行 × ペアごとの遅れとピーク相関
  person_lag.csv     被験者 × 帯域ごとの平均・SD
  summary.csv        ペアごとの ICC など
  lag_by_person.png  被験者ごとの遅れのばらつき（縦軸の範囲は YLIM でペアごとに指定できる）
  slow_vs_fast.png   slow と fast の平均遅れの散布図
  xcorr_curve.png    相互相関の曲線（横軸 = ずらし量 [ms]，縦軸 = 相関係数）．行 = ペア，列 = 被験者
  all_xcorr_curve.csv  上の曲線の数値（試行 × ペア × ずらし量）

相関の曲線の見方
  山がはっきりしていれば，ピークの位置を「遅れ」と読める．
  曲線がほぼ平ら（ピークの相関とずらし量 0 の相関がほとんど同じ）なら，ピークの位置はノイズで決まっており，遅れとは言えない．
  all_lag_norm.csv の r_at_0（ずらし量 0 の相関）と r_gain（= peak_r − r_at_0）で数値でも確認できる．
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
MIN_R = 0.5          # 有効判定: ピーク相関がこれ以上（正の相関のみ対象）
EDGE_MS = 10.0         # 有効判定: ピークが範囲の端からこの ms 以上内側にあること

# lag_by_person.png の縦軸の範囲 [ms]．ペアごとに (下限, 上限) を書く．None = 自動
#   例: 3 つとも同じにしたいときは 3 つに同じ値を書く
#   範囲の外に出た点は，枠の端に三角（▲▼）で表示する（点が消えたことに気づけるように）
YLIM = {
    "shoulder-elbow": (-250, 100),
    "elbow-wrist":    (-250, 50),
    "wrist-mcp3":     (-250, 50),
}

# xcorr_curve.png の縦軸（相関係数）の範囲．ペアごとに (下限, 上限)．None = 自動（同じ行で共通）
XCORR_YLIM = {
    "shoulder-elbow": None,
    "elbow-wrist":    None,
    "wrist-mcp3":     None,
}

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
    """
    1 試行のペアごとの結果のリスト．各要素は dict:
      pair, lag_ms（遅れ）, peak_r（ピークの相関）, r_at_0（ずらし量 0 の相関）,
      lags_ms・corr（相関の曲線．横軸 = ずらし量 [ms]，縦軸 = 相関係数）
    """
    data = load_trial(trial_dir, fc=cfg.FC)
    max_lag = int(round(MAX_LAG * data.fs))

    out = []
    for a, b in adjacent_pairs(data.joints):
        lags, corr, peak, peak_r = xcorr_direct(data.norm(a), data.norm(b), max_lag)
        out.append({
            "pair": f"{a}-{b}",
            "lag_ms": peak / data.fs * 1000,
            "peak_r": peak_r,
            "r_at_0": float(corr[max_lag]),          # lags の中央がずらし量 0
            "lags_ms": lags / data.fs * 1000,
            "corr": corr,
        })
    return out




# ---------- 2. 収集と集計 ----------
def collect():
    """
    全試行の遅れを DataFrame にまとめ、有効判定の列 valid を付ける。
    相関の曲線は別のリスト curves（1 要素 = 試行 × ペア）で返す。
    """
    rows, curves = [], []
    for ses, trial in iter_config_trials(cfg, ROOT):
        for res in lag_norm_trial(trial.dir):
            info = {
                "label": ses.label,
                "person": ses.person,
                "group": ses.group,
                "dpi": ses.dpi,
                "trial": trial.name,
                "band": trial.band,
                "seed": trial.seed,
                "pair": res["pair"],
            }
            rows.append({**info, "lag_ms": res["lag_ms"], "peak_r": res["peak_r"],
                         "r_at_0": res["r_at_0"], "r_gain": res["peak_r"] - res["r_at_0"]})
            curves.append({**info, "lag_ms": res["lag_ms"], "peak_r": res["peak_r"],
                           "lags_ms": res["lags_ms"], "corr": res["corr"]})
        print(f"完了: {ses.label}/{trial.name}")

    df = pd.DataFrame(rows)
    # 正の相関が基準以上 かつ ピークが探索範囲の端に張り付いていない
    edge = MAX_LAG * 1000 - EDGE_MS
    df["valid"] = (df.peak_r >= MIN_R) & (df.lag_ms.abs() < edge)
    return df, curves


def curves_table(curves):
    """相関の曲線を縦長の表にする（1 行 = 試行 × ペア × ずらし量）"""
    parts = []
    for c in curves:
        parts.append(pd.DataFrame({
            "person": c["person"], "group": c["group"], "band": c["band"], "seed": c["seed"],
            "trial": c["trial"], "pair": c["pair"], "shift_ms": c["lags_ms"], "r": c["corr"],
        }))
    return pd.concat(parts, ignore_index=True)


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
                    "mean_r_at_0": g.r_at_0.mean(), "mean_r_gain": g.r_gain.mean(),
                })
        rows_sum.append({
            "pair": pair, "n_valid": len(ok), "n_excluded": len(d) - len(ok),
            "mean_r": ok.peak_r.mean(),
            "mean_r_at_0": ok.r_at_0.mean(), "mean_r_gain": ok.r_gain.mean(),
            "icc": icc_oneway(ok.lag_ms, ok.person) if ok.person.nunique() >= 2 else np.nan,
        })
    return pd.DataFrame(rows_person), pd.DataFrame(rows_sum)


# ---------- 3. 図 ----------
def _pair_title(ax, pair):
    a, b = pair.split("-")
    ax.set_title(f"{a} -> {b}")
    return b


def _scatter_clipped(ax, x, y, ylim, **kw):
    """
    散布図。ylim = (下限, 上限) の外に出た点は、枠の端に三角で描く（▲ = 上に外れ，▼ = 下に外れ）。
    ylim が None なら普通の散布図。
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if ylim is None:
        ax.scatter(x, y, **kw)
        return
    lo, hi = ylim
    inside = (y >= lo) & (y <= hi)
    ax.scatter(x[inside], y[inside], **kw)
    kw_out = {k: v for k, v in kw.items() if k != "label"}
    ax.scatter(x[y > hi], np.full((y > hi).sum(), hi), marker="^", clip_on=False, **kw_out)
    ax.scatter(x[y < lo], np.full((y < lo).sum(), lo), marker="v", clip_on=False, **kw_out)


def plot_by_person(path, df):
    """被験者ごとの遅れのばらつきと平均 ± SD。縦軸の範囲は設定の YLIM（ペアごと）"""
    persons, group_of = person_order(df)
    pairs = list(dict.fromkeys(df.pair))
    x = {p: i for i, p in enumerate(persons)}

    fig, axs = plt.subplots(len(pairs), 1, figsize=(max(10, 1.2 * len(persons)), 3.2 * len(pairs)),
                            sharex=True, squeeze=False)
    for ax, pair in zip(axs[:, 0], pairs):
        d = df[(df.pair == pair) & df.valid]
        ylim = YLIM.get(pair)
        for band, off in [("slow", -0.12), ("fast", 0.12)]:
            b = d[d.band == band]
            _scatter_clipped(ax, [x[p] + off for p in b.person], b.lag_ms, ylim,
                             s=22, color=BAND_COLOR[band], alpha=0.8, label=band)
        g = d.groupby("person").lag_ms
        ax.errorbar([x[p] for p in g.mean().index], g.mean(), yerr=g.std(ddof=1), fmt="o",
                    color="black", ms=6, capsize=4, label="mean ± SD")
        ax.axhline(0, color="gray", lw=0.8)
        distal = _pair_title(ax, pair)
        ax.set_ylabel(f"lag [ms]\n(+ = {distal} later)")
        ax.grid(alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)

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


def plot_xcorr_curves(path, curves):
    """
    相互相関の曲線。横軸 = ずらし量 [ms]（+ = 遠位が遅れる），縦軸 = 相関係数。
    行 = ペア，列 = 被験者。線 1 本 = 1 試行（色 = slow / fast），● = ピーク（遅れとして採用した位置）。
    """
    cv = pd.DataFrame(curves)
    persons, group_of = person_order(cv)
    pairs = list(dict.fromkeys(cv.pair))

    fig, axs = plt.subplots(len(pairs), len(persons), figsize=(2.3 * len(persons), 2.6 * len(pairs)),
                            sharex=True, sharey="row", squeeze=False)
    for r, pair in enumerate(pairs):
        for c, person in enumerate(persons):
            ax = axs[r, c]
            for _, row in cv[(cv.pair == pair) & (cv.person == person)].iterrows():
                color = BAND_COLOR[row.band]
                ax.plot(row.lags_ms, row["corr"], color=color, lw=1, alpha=0.8)
                ax.plot(row.lag_ms, row.peak_r, "o", color=color, ms=4)
            ax.axvline(0, color="gray", lw=0.8)
            ax.grid(alpha=0.3)
            if XCORR_YLIM.get(pair) is not None:
                ax.set_ylim(*XCORR_YLIM[pair])
            if r == 0:
                ax.set_title(f"{person} ({group_of[person]})", fontsize=9)
            if c == 0:
                a, b = pair.split("-")
                ax.set_ylabel(f"{a} -> {b}\ncorrelation")
    handles = [plt.Line2D([], [], color=v, label=k) for k, v in BAND_COLOR.items()]
    handles.append(plt.Line2D([], [], color="k", marker="o", ls="", ms=4, label="peak"))
    axs[0, -1].legend(handles=handles, fontsize=7, loc="lower right")
    fig.supxlabel("shift [ms]  (+ = distal later)")
    save_fig(fig, path, DPI)


# ---------- メイン ----------
def main():
    df, curves = collect()
    person_lag, summary = summarize(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "all_lag_norm.csv", index=False)
    person_lag.to_csv(OUT_DIR / "person_lag.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    plot_by_person(OUT_DIR / "lag_by_person.png", df)
    plot_slow_vs_fast(OUT_DIR / "slow_vs_fast.png", person_lag)
    curves_table(curves).to_csv(OUT_DIR / "all_xcorr_curve.csv", index=False)
    plot_xcorr_curves(OUT_DIR / "xcorr_curve.png", curves)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()