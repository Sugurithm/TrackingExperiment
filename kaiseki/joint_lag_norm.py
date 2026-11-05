#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_lag_norm.py — 変位ノルムの相互相関で、隣り合う関節の遅れを人ごとに比べる（試行 40 s 全体）
※ joint_basic.py に依存しない完全独立版
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import spearmanr

# 実行ファイル自身のディレクトリとリポジトリルートを取得
HERE = Path(__file__).resolve().parent        # temp/
ROOT = HERE.parent                            # リポジトリ直下
sys.path.insert(0, str(HERE))
import compare_config as cfg                  # noqa: E402

# ---------- 設定 ----------
OUT_DIR = HERE / cfg.OUT / "joint_lag_norm"

JOINTS = ["shoulder", "elbow", "wrist", "mcp3"]  # 関節（根元から末端への連結順）
FILTER_ORDER = 4       # ローパスフィルタの次数

MAX_LAG = 0.3          # 相互相関でずらす最大範囲 [s]（±）
MIN_R = 0.0            # 有効判定: ピーク相関がこれ以上（正の相関のみ対象）
EDGE_MS = 10.0         # 有効判定: ピークが範囲の端からこの ms 以上内側にあること

BAND_COLOR = {"slow": "tab:blue", "fast": "tab:orange"}


# ---------- 1. 試行データの探索と読み込み（旧 joint_basic の機能） ----------
def session_info(s):
    """config の SESSIONS の 1 行から、フォルダと人・群・dpi・呼び名を辞書にまとめる"""
    d = Path(s["dir"])
    d = d if d.is_absolute() else ROOT / d
    m = re.match(r"aim_([A-Za-z]+)(\d+)_", d.name)
    return {
        "dir": d,
        "label": s.get("label", d.name.split("_")[1] if "_" in d.name else d.name),
        "person": s.get("person", m.group(1) if m else d.name),
        "group": s.get("group", ""),
        "dpi": s.get("dpi", int(m.group(2)) if m else ""),
    }


def trial_key(name):
    """フォルダ名（例: trial_01_slow_s1_2）から (帯域, seed番号, やり直し番号) を抽出"""
    m = re.search(r"(slow|fast)_s(\d+)(?:_(\d+))?", name.lower())
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3) or 1)


def find_trials(session_dir):
    """セッション内の試行フォルダを集め、やり直し試行は cfg.ATTEMPT (first/last) に従って選ぶ"""
    groups = {}
    for d in sorted(session_dir.iterdir()):
        key = trial_key(d.name)
        if key and (d / "joints" / "rs_joints.npz").exists():
            groups.setdefault(key[:2], []).append((key[2], d))
            
    trials = []
    for key in sorted(groups):
        tries = sorted(groups[key])
        if cfg.ATTEMPT == "first":
            tries = tries[:1]
        elif cfg.ATTEMPT == "last":
            tries = tries[-1:]
        trials += [(key[0], key[1], att, d) for att, d in tries]
    return trials


def load_trial(trial_dir):
    """
    npzファイルから3次元関節座標を読み込み、等間隔リサンプリング・ローパスフィルタ・平均引きを行う。
    戻り値: 時間配列 tt [s], サンプリング周波数 fs [Hz], 平均位置 mean, 変位データ disp
    """
    z = np.load(trial_dir / "joints" / "rs_joints.npz")
    names = [str(n) for n in z["names"]]
    t = z["t"].astype(float)
    P = z["P_C"].astype(float)  # カメラ座標系での3次元位置 (フレーム数, 関節数, 3)

    # 時刻順にソート
    order = np.argsort(t)
    t, P = t[order], P[order]

    # 時間間隔の中央値からサンプリング周波数 fs を求め、等間隔の時間グリッドを作成
    fs = 1.0 / np.median(np.diff(t))
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    
    # バターワース型ローパスフィルタの設計（カットオフ周波数 cfg.FC）
    b, a = butter(FILTER_ORDER, min(cfg.FC / (fs / 2), 0.99))

    mean, disp = {}, {}
    for j in JOINTS:
        X = P[:, names.index(j), :]
        ok = np.isfinite(X).all(axis=1)  # 欠損値（NaN/Inf）でないフレームを抽出
        
        # 欠損を線形補間しつつ等間隔グリッドにリサンプリング（x, y, z の3軸それぞれ）
        Y = np.column_stack([np.interp(grid, t[ok], X[ok, k]) for k in range(3)])
        
        # ゼロ位相ローパスフィルタを適用して高周波ノイズを除去
        Y = filtfilt(b, a, Y, axis=0)
        
        # 試行全体の平均位置を引き、基準からの「変位ベクトル」にする
        mean[j] = Y.mean(axis=0)
        disp[j] = Y - mean[j]
        
    return grid - grid[0], fs, mean, disp


# ---------- 2. 相互相関の直接実装 ----------
def xcorr_direct(a, b, max_lag):
    """
    2つの1次元時系列 a, b の正規化相互相関を計算し、ピークの位置と相関係数を返す。
    ※ プラスのラグ = 「b（遠位）が a（近位）より遅れている」と定義
    """
    n = len(a)
    
    # 1. Zスコア正規化（平均を0、標準偏差を1に揃える）
    a_norm = (a - np.mean(a)) / np.std(a)
    b_norm = (b - np.mean(b)) / np.std(b)
    
    # 2. -max_lag から +max_lag までのずらし量（フレーム数）の配列を作成
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.zeros(len(lags), dtype=float)
    
    # 3. 1フレームずつずらしながら内積（掛け合わせて平均）を計算
    for i, lag in enumerate(lags):
        if lag > 0:
            # b を後ろにずらす（＝ a の前半と b の後半を比較）
            corr[i] = np.sum(a_norm[:-lag] * b_norm[lag:]) / n
        elif lag < 0:
            # b を前にずらす（＝ a の後半と b の前半を比較）
            k = -lag
            corr[i] = np.sum(a_norm[k:] * b_norm[:-k]) / n
        else:
            # ずらしなし（同時相関）
            corr[i] = np.sum(a_norm * b_norm) / n
            
    # 4. 最も相関が高くなるインデックスを特定（※ノルム同士なので正のピークのみを探す）
    peak_idx = int(np.argmax(corr))
    peak_r = float(corr[peak_idx])
    
    # 5. 放物線補間（パラボラフィッティング）で1フレーム未満の細かいピーク位置を推定
    shift = 0.0
    if 0 < peak_idx < len(corr) - 1:
        y0, y1, y2 = corr[peak_idx - 1], corr[peak_idx], corr[peak_idx + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            shift = 0.5 * (y0 - y2) / denom
            
    peak_lag = float(lags[peak_idx]) + shift
    return lags, corr, peak_lag, peak_r


# ---------- 3. 1試行の計算と全体収集 ----------
def lag_norm_trial(trial_dir):
    """戻り値: ペアごとの (遅れ [ms], ピークの相関)"""
    tt, fs, mean, disp = load_trial(trial_dir)
    
    # 各時刻の3次元変位 (dx, dy, dz) からユークリッド距離（変位ノルム） √(dx²+dy²+dz²) を計算
    norm = {j: np.linalg.norm(disp[j], axis=1) for j in JOINTS}
    max_lag = int(round(MAX_LAG * fs))

    out = []
    for a, b in zip(JOINTS[:-1], JOINTS[1:]):
        _, _, peak, peak_r = xcorr_direct(norm[a], norm[b], max_lag)
        out.append((f"{a}-{b}", peak / fs * 1000, peak_r))
    return out


def collect():
    """全セッション・全試行を走査して遅れ時間を計算し、DataFrameにまとめる"""
    rows = []
    for s in cfg.SESSIONS:
        ses = session_info(s)
        if not ses["dir"].exists():
            print(f"見つかりません: {ses['dir']}")
            continue
        for band, seed, attempt, trial in find_trials(ses["dir"]):
            for pair, lag_ms, peak_r in lag_norm_trial(trial):
                rows.append({"label": ses["label"], "person": ses["person"], "group": ses["group"],
                             "dpi": ses["dpi"], "trial": trial.name, "band": band, "seed": seed,
                             "pair": pair, "lag_ms": lag_ms, "peak_r": peak_r})
            print(f"完了: {ses['label']}/{trial.name}")
            
    df = pd.DataFrame(rows)
    
    # 有効判定: 正の相関が基準以上 かつ ピークが探索範囲の端に張り付いていないかチェック
    edge = MAX_LAG * 1000 - EDGE_MS
    df["valid"] = (df.peak_r >= MIN_R) & (df.lag_ms.abs() < edge)
    return df


# ---------- 4. 集計 ----------
def icc_oneway(values, groups):
    """一元配置の級内相関係数 ICC(1) を算出（個人差の大きさを 0〜1 で評価）"""
    values, groups = np.asarray(values, float), np.asarray(groups)
    labels = np.unique(groups)
    n_i = np.array([np.sum(groups == g) for g in labels])
    m_i = np.array([values[groups == g].mean() for g in labels])
    N, k = len(values), len(labels)
    
    ss_b = np.sum(n_i * (m_i - values.mean()) ** 2)
    ss_w = sum(np.sum((values[groups == g] - m) ** 2) for g, m in zip(labels, m_i))
    
    ms_b, ms_w = ss_b / (k - 1), ss_w / (N - k)
    n0 = (N - np.sum(n_i ** 2) / N) / (k - 1)
    return (ms_b - ms_w) / (ms_b + (n0 - 1) * ms_w)


def summarize(df):
    """被験者ごとの平均・標準偏差（ラグおよび相関係数）と、全体の統計指標を集計"""
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
        m = ok.groupby(["person", "band"]).lag_ms.mean().unstack().dropna()
        rows_sum.append({
            "pair": pair, "n_valid": len(ok), "n_excluded": len(d) - len(ok),
            "mean_r": ok.peak_r.mean(),
            "icc": icc_oneway(ok.lag_ms, ok.person) if ok.person.nunique() >= 2 else np.nan,
            "rho_slow_fast": spearmanr(m["slow"], m["fast"])[0] if len(m) >= 3 and {"slow", "fast"} <= set(m.columns) else np.nan,
        })
    return pd.DataFrame(rows_person), pd.DataFrame(rows_sum)


# ---------- 5. 図の作成 ----------
def person_order(df):
    """グラフの横軸用に、グループ順・名前順でソートした被験者リストを作成"""
    p = df[["person", "group"]].drop_duplicates().sort_values(["group", "person"])
    return list(p.person), dict(zip(p.person, p.group))


def plot_by_person(path, df):
    """被験者ごとの遅れ時間のばらつきと平均±SDをプロット"""
    persons, group_of = person_order(df)
    pairs = list(dict.fromkeys(df.pair))
    x = {p: i for i, p in enumerate(persons)}
    
    fig, axs = plt.subplots(len(pairs), 1, figsize=(max(10, 1.2 * len(persons)), 3.2 * len(pairs)),
                            sharex=True, squeeze=False)
    for ax, pair in zip(axs[:, 0], pairs):
        d = df[(df.pair == pair) & df.valid]
        for band, off in [("slow", -0.12), ("fast", 0.12)]:
            b = d[d.band == band]
            ax.scatter([x[p] + off for p in b.person], b.lag_ms, s=22,
                       color=BAND_COLOR[band], alpha=0.8, label=band)
        g = d.groupby("person").lag_ms
        ax.errorbar([x[p] for p in g.mean().index], g.mean(), yerr=g.std(ddof=1), fmt="o",
                    color="black", ms=6, capsize=4, label="mean ± SD")
        ax.axhline(0, color="gray", lw=0.8)
        a, b_ = pair.split("-")
        ax.set_title(f"{a} -> {b_}")
        ax.set_ylabel(f"lag [ms]\n(+ = {b_} later)")
        ax.grid(alpha=0.3)
        
    axs[0, 0].legend(loc="upper right", ncol=3, fontsize=9)
    axs[-1, 0].set_xlabel("participant")
    axs[-1, 0].set_xticks(range(len(persons)))
    axs[-1, 0].set_xticklabels([f"{p}\n({group_of[p]})" for p in persons])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_slow_vs_fast(path, person_lag):
    """横軸にslow時の平均遅れ、縦軸にfast時の平均遅れをとった散布図を作成"""
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
            
        a, b_ = pair.split("-")
        ax.set_title(f"{a} -> {b_}")
        ax.set_xlabel("slow: mean lag [ms]")
        ax.set_ylabel("fast: mean lag [ms]")
        ax.grid(alpha=0.3)
        
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


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