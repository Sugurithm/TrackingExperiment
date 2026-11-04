#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_lag_person.py — 関節ペアの遅れ（相互相関）を人ごとに比べる

  python temp/joint_basic.py          # 先にこれで all_xcorr.csv を作る
  python temp/joint_lag_person.py

入力: temp/<OUT>/joint_basic/all_xcorr.csv
出力: temp/<OUT>/joint_lag_person/
  person_lag.csv     人 × ペア × 帯域（all / slow / fast）の 平均・SD・試行数
  summary.csv        ペアごとの 人の差の大きさ（ICC, F, 並べ替え p）と slow↔fast の順位相関
  lag_by_person.png  人ごとの遅れ（点 = 試行、黒 = 平均 ± SD）
  slow_vs_fast.png   人ごとの平均の slow と fast の比較
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_config as cfg      # noqa: E402
import joint_basic as jb          # noqa: E402  （MAX_LAG と関節の並びを共有する）

# ---------- 設定 ----------
IN_CSV = HERE / cfg.OUT / "joint_basic" / "all_xcorr.csv"
OUT_DIR = HERE / cfg.OUT / "joint_lag_person"

AXES = ["v"]          # 比べる向き。横 h は肘・肩がほぼ動かないので既定では外す
MIN_R = 0.5           # ピークの相関の絶対値がこれ未満の試行は使わない
EDGE_MS = 10.0        # ピークがずらす範囲の端からこの ms 以内なら使わない（端に張り付いた結果）
N_PERM = 10000        # 並べ替え検定の回数
SEED = 0

BAND_COLOR = {"slow": "tab:blue", "fast": "tab:orange"}


# ---------- 有効な試行だけ残す ----------
def load():
    df = pd.read_csv(IN_CSV)
    edge = jb.MAX_LAG * 1000 - EDGE_MS
    df["valid"] = (df.peak_r.abs() >= MIN_R) & (df.lag_ms.abs() < edge)
    return df[df.axis.isin(AXES)]


def person_order(df):
    """グループ → 名前 の順に並べる（どの図でも同じ順にする）"""
    p = df[["person", "group"]].drop_duplicates().sort_values(["group", "person"])
    return list(p.person), dict(zip(p.person, p.group))


# ---------- 人の差の大きさ ----------
def icc_oneway(values, groups):
    """一元配置の ICC(1)。人の間の分散 /（人の間 + 人の中）。人ごとの試行数が違ってもよい"""
    values, groups = np.asarray(values, float), np.asarray(groups)
    labels = np.unique(groups)
    n_i = np.array([np.sum(groups == g) for g in labels])
    m_i = np.array([values[groups == g].mean() for g in labels])
    N, k = len(values), len(labels)
    ss_b = np.sum(n_i * (m_i - values.mean()) ** 2)
    ss_w = sum(np.sum((values[groups == g] - m) ** 2) for g, m in zip(labels, m_i))
    ms_b, ms_w = ss_b / (k - 1), ss_w / (N - k)
    n0 = (N - np.sum(n_i ** 2) / N) / (k - 1)          # 試行数が違うときの平均的な試行数
    icc = (ms_b - ms_w) / (ms_b + (n0 - 1) * ms_w)
    return icc, ms_b / ms_w


def perm_p(values, groups, f_obs, rng):
    """人のラベルを並べ替えて F が観測値以上になる割合"""
    count = 0
    for _ in range(N_PERM):
        if icc_oneway(values, rng.permutation(groups))[1] >= f_obs:
            count += 1
    return (count + 1) / (N_PERM + 1)


# ---------- 集計 ----------
def summarize(df):
    rng = np.random.default_rng(SEED)
    rows_person, rows_sum = [], []
    for (axis, pair), d in df.groupby(["axis", "pair"], sort=False):
        ok = d[d.valid]
        for band in ["all", "slow", "fast"]:
            b = ok if band == "all" else ok[ok.band == band]
            for person, g in b.groupby("person"):
                rows_person.append({"axis": axis, "pair": pair, "band": band, "person": person,
                                    "group": g.group.iloc[0], "n": len(g),
                                    "mean_ms": g.lag_ms.mean(), "sd_ms": g.lag_ms.std(ddof=1)})

        icc, F = icc_oneway(ok.lag_ms, ok.person)
        p = perm_p(ok.lag_ms.values, ok.person.values, F, rng)
        m = ok.groupby(["person", "band"]).lag_ms.mean().unstack()
        m = m.dropna()
        rho = spearmanr(m["slow"], m["fast"])[0] if len(m) >= 3 else np.nan
        rows_sum.append({"axis": axis, "pair": pair, "n_valid": len(ok), "n_excluded": len(d) - len(ok),
                         "icc": icc, "F": F, "p_perm": p, "rho_slow_fast": rho})
    return pd.DataFrame(rows_person), pd.DataFrame(rows_sum)


# ---------- 図 ----------
def plot_by_person(path, df, summary, persons, group_of):
    keys = list(summary[["axis", "pair"]].itertuples(index=False, name=None))
    fig, axs = plt.subplots(len(keys), 1, figsize=(10, 3.2 * len(keys)), sharex=True, squeeze=False)
    x = {p: i for i, p in enumerate(persons)}
    for ax, (axis, pair) in zip(axs[:, 0], keys):
        d = df[(df.axis == axis) & (df.pair == pair) & df.valid]
        for band, off in [("slow", -0.12), ("fast", 0.12)]:
            b = d[d.band == band]
            ax.scatter([x[p] + off for p in b.person], b.lag_ms, s=22,
                       color=BAND_COLOR[band], alpha=0.8, label=band)
        g = d.groupby("person").lag_ms
        mean, sd = g.mean(), g.std(ddof=1)
        ax.errorbar([x[p] for p in mean.index], mean, yerr=sd, fmt="o", color="black",
                    ms=6, capsize=4, label="mean ± SD")
        ax.axhline(0, color="gray", lw=0.8)
        a, b_ = pair.split("-")
        ax.set_title(f"{a} - {b_}")
        ax.set_ylabel(f"lag [ms]\n(+ = {b_} later)")
        ax.grid(alpha=0.3)
    axs[0, 0].legend(loc="upper right", ncol=3, fontsize=9)
    axs[-1, 0].set_xlabel("participant")
    axs[-1, 0].set_xticks(range(len(persons)))
    axs[-1, 0].set_xticklabels([f"{p}\n({group_of[p]})" for p in persons])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_slow_vs_fast(path, person_lag, summary):
    keys = list(summary[["axis", "pair"]].itertuples(index=False, name=None))
    fig, axs = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.2), squeeze=False)
    for ax, (axis, pair) in zip(axs[0], keys):
        d = person_lag[(person_lag.axis == axis) & (person_lag.pair == pair)]
        m = d.pivot(index="person", columns="band", values="mean_ms").dropna(subset=["slow", "fast"])
        ax.scatter(m.slow, m.fast, color="black")
        for p, r in m.iterrows():
            ax.annotate(p, (r.slow, r.fast), fontsize=8, xytext=(3, 3), textcoords="offset points")
        lim = [min(m.slow.min(), m.fast.min()) - 10, max(m.slow.max(), m.fast.max()) + 10]
        ax.plot(lim, lim, color="gray", lw=0.8, ls="--")
        a, b_ = pair.split("-")
        ax.set_title(f"{a} - {b_}")
        ax.set_xlabel("slow: mean lag [ms]")
        ax.set_ylabel("fast: mean lag [ms]")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------- メイン ----------
def main():
    df = load()
    persons, group_of = person_order(df)
    person_lag, summary = summarize(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    person_lag.to_csv(OUT_DIR / "person_lag.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    plot_by_person(OUT_DIR / "lag_by_person.png", df, summary, persons, group_of)
    plot_slow_vs_fast(OUT_DIR / "slow_vs_fast.png", person_lag, summary)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()