#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_use_by_cursor_direction.py — カーソルの移動方向（8 方向）ごとの関節の使い方（解析区間 40 s）

  問い: その向きの動きを「実際に出した」とき、どの関節を使ったか
  ラベル: 画面上のカーソル速度の向き（マウス入力の写像。関節の動きと同じ時刻）

  link_share.py と joint_lag_norm.py の計算を、カーソルの移動方向で仕分けたフレームごとにやり直す。
  方向を使うのはラベル付けだけ。関節はカメラ座標の 3 次元のまま扱う（成分に分けない）。

  1. 方向ラベル   d(t) = round(atan2(q̇_y, q̇_x) / 45°) mod 8
       q̇ = 画面上のカーソルの速度（x 右・y 上）
       名前は画面上の向き（R, UR, U, UL, L, DL, D, DR）。マウスの実空間の向きとは限らない
  2. 寄与率（u = 手先 MCP3 の速度，w_k = リンクの相対速度）
       射影（主）    share_k = Σ_{t∈S_d} w_k·u / Σ_{t∈S_d} |u|²
       パワー（補助） share_k = Σ_{t∈S_d} |w_k|² / Σ_j Σ_{t∈S_d} |w_j|²   （ノイズ補正なし）
  3. 遅れ（マスク付き相互相関）
       r_d(τ) = Pearson{ a(t), b(t+τ) : t ∈ S_d } のピークの τ
       a, b = 隣り合う関節の速さ |v|（主）／変位のノルム |x|（元の信号，比較用）

  静止試行（trial_*_static_pre）は使わない（find_trials が slow / fast の試行だけを拾う）。

出力: <cfg.OUT>/joint_use_by_cursor_direction/
  all_share.csv          試行 × 方向（all + 8 方向）の寄与率。agree_target = ターゲットのラベルと一致した割合
  all_lag.csv            試行 × 信号 × 方向 × ペアの遅れ
  person_share.csv       被験者 × 帯域 × 方向の平均・SD
  person_lag.csv         被験者 × 帯域 × 方向 × ペアの遅れの平均・SD（信号 = 速さ）
  summary_lag.csv        信号 × ペア × 方向の ICC
  share_proj_by_dir.png  射影の寄与率（主）
  share_pow_by_dir.png   パワーの寄与率（補助）
  lag_by_dir.png         遅れ（速さ）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import compare_config as cfg  # noqa: E402
from utilities import adjacent_pairs, iter_config_trials, load_trial, save_fig  # noqa: E402
from joint_lag_norm import icc_oneway  # noqa: E402

# ---------- 設定 ----------
OUT_NAME = "joint_use_by_cursor_direction"
SOURCE = "cursor"   # ラベルに使う 2D 速度

# 方向（画面上の 2D 速度の向き）。0° = 右，90° = 上，反時計回りに 45° ずつ，各 ±22.5°
DIRS = ["R", "UR", "U", "UL", "L", "DL", "D", "DR"]
DIR_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
N_DIR = len(DIRS)

# リンク名: (近位の関節, 遠位の関節)。shoulder だけは肩そのものの速度（link_share.py と同じ）
LINKS = {
    "shoulder":  (None, "shoulder"),
    "upper_arm": ("shoulder", "elbow"),
    "forearm":   ("elbow", "wrist"),
    "hand":      ("wrist", "mcp3"),
}
TIP = "mcp3"

# 遅れ（joint_lag_norm.py と同じ値）
MAX_LAG = 0.5          # ずらす最大範囲 [s]（±）
MIN_R = 0.0            # 有効判定: ピーク相関がこれ以上
EDGE_MS = 10.0         # 有効判定: ピークが範囲の端からこの ms 以上内側
SIGNALS = ["speed", "disp_norm"]   # speed = |v|（主），disp_norm = |x|（元の信号，比較用）
MAIN_SIGNAL = "speed"

GROUP_COLOR = {"FPS": "#2a78d6", "nonFPS": "#eb6834"}
DPI = 120


# =====================================================================
# 1. 方向ラベル
# =====================================================================
def direction_labels(vel2d):
    """2D 速度 (N, 2) → 方向の番号 0–7（0 = 右，反時計回りに 45° ずつ）"""
    ang = np.arctan2(vel2d[:, 1], vel2d[:, 0])
    return np.round(ang / (2 * np.pi / N_DIR)).astype(int) % N_DIR


def screen_vel(data, source):
    """ラベルに使う 2D 速度 [cm/s]。source = "cursor" / "target" """
    v = data.cursor_vel() if source == "cursor" else data.target_vel()
    if v is None:
        raise ValueError("task.csv がないので方向を決められません")
    return v


def link_velocities(data):
    """各リンクの速度 (N, 3) [mm/s]。合計 = 手先の速度"""
    return {k: data.vel(b) if a is None else data.relative_vel(a, b) for k, (a, b) in LINKS.items()}


# =====================================================================
# 2. 寄与率
# =====================================================================
def shares(W, u, mask):
    """フレーム集合 mask での射影とパワーの寄与率"""
    out = {"n_frames": int(mask.sum())}
    if mask.sum() == 0:
        return out
    um = u[mask]
    den = np.sum(um * um)
    out["tip_rms_mm_s"] = float(np.sqrt(den / mask.sum()))
    pw = {k: float(np.sum(w[mask] ** 2)) for k, w in W.items()}
    for k, w in W.items():
        out[f"share_proj_{k}"] = float(np.sum(w[mask] * um) / den)
        out[f"share_pow_{k}"] = pw[k] / sum(pw.values())
    return out


def disp_share(data):
    """元の link_share.py と同じ変位の寄与率（全フレームのみ）"""
    tip = data.disp[TIP]
    den = np.sum(tip * tip)
    return {f"share_disp_{k}": float(np.sum((data.disp[b] if a is None else data.relative(a, b)) * tip) / den)
            for k, (a, b) in LINKS.items()}


# =====================================================================
# 3. 遅れ（マスク付き相互相関）
# =====================================================================
def xcorr_masked(a, b, mask, max_lag):
    """
    近位 a(t)（t ∈ mask）と遠位 b(t+τ) の組だけでピアソン相関をとり、τ をずらす。
    b(t+τ) は mask の外でもよい（区間を切るのではなく、試行全体の連続した信号をずらす）。
    ピーク位置は joint_lag_norm.xcorr_direct と同じく放物線補間。プラスのラグ = b が遅れる。
    """
    idx = np.flatnonzero(mask)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.full(len(lags), np.nan)
    for i, L in enumerate(lags):
        j = idx + L
        ok = (j >= 0) & (j < len(b))
        if ok.sum() < 3:
            continue
        x, y = a[idx[ok]], b[j[ok]]
        if x.std() == 0 or y.std() == 0:
            continue
        corr[i] = np.corrcoef(x, y)[0, 1]
    if np.all(np.isnan(corr)):
        return lags, corr, np.nan, np.nan

    p = int(np.nanargmax(corr))
    shift = 0.0
    if 0 < p < len(corr) - 1 and np.isfinite(corr[p - 1]) and np.isfinite(corr[p + 1]):
        y0, y1, y2 = corr[p - 1], corr[p], corr[p + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            shift = 0.5 * (y0 - y2) / denom
    return lags, corr, float(lags[p]) + shift, float(corr[p])


# =====================================================================
# 4. 1 試行の解析
# =====================================================================
def analyze_trial(trial_dir, source):
    data = load_trial(trial_dir, fc=cfg.FC)
    lab = direction_labels(screen_vel(data, source))
    # もう一方の 2D 速度のラベル（どれだけ一致するかを記録する）
    other = "target" if source == "cursor" else "cursor"
    lab_other = direction_labels(screen_vel(data, other))
    masks = {"all": np.ones(len(lab), bool), **{d: lab == i for i, d in enumerate(DIRS)}}

    W = link_velocities(data)
    u = data.vel(TIP)
    share_rows = []
    for dname, m in masks.items():
        row = {"dir": dname, "time_s": m.sum() / data.fs,
               f"agree_{other}": float(np.mean(lab[m] == lab_other[m])) if m.any() else np.nan,
               **shares(W, u, m)}
        if dname == "all":
            row.update(disp_share(data))
        share_rows.append(row)

    max_lag = int(round(MAX_LAG * data.fs))
    sig = {"speed": {j: np.linalg.norm(data.vel(j), axis=1) for j in data.joints},
           "disp_norm": {j: data.norm(j) for j in data.joints}}
    lag_rows = []
    for dname, m in masks.items():
        for sname in SIGNALS:
            for a, b in adjacent_pairs(data.joints):
                _, _, peak, r = xcorr_masked(sig[sname][a], sig[sname][b], m, max_lag)
                lag_rows.append({"signal": sname, "dir": dname, "pair": f"{a}-{b}",
                                 "n_frames": int(m.sum()), "lag_ms": peak / data.fs * 1000, "peak_r": r})
    return share_rows, lag_rows


# =====================================================================
# 5. 収集と集計
# =====================================================================
def collect(source):
    shares_all, lags_all = [], []
    for ses, trial in iter_config_trials(cfg, ROOT):
        info = {"label": ses.label, "person": ses.person, "group": ses.group, "dpi": ses.dpi,
                "trial": trial.name, "band": trial.band, "seed": trial.seed}
        s_rows, l_rows = analyze_trial(trial.dir, source)
        shares_all += [{**info, **r} for r in s_rows]
        lags_all += [{**info, **r} for r in l_rows]
        print(f"完了: {ses.label}/{trial.name}")

    share = pd.DataFrame(shares_all)
    lag = pd.DataFrame(lags_all)
    edge = MAX_LAG * 1000 - EDGE_MS
    lag["valid"] = (lag.peak_r >= MIN_R) & (lag.lag_ms.abs() < edge)
    return share, lag


def summarize(share, lag):
    cols = [c for c in share.columns if c.startswith(("share_proj_", "share_pow_", "agree_"))]
    g = share.groupby(["person", "group", "band", "dir"], sort=False)
    person_share = g[cols].agg(["mean", "std"])
    person_share.columns = ["_".join(c) for c in person_share.columns]
    person_share.insert(0, "n_frames", g.n_frames.sum())
    person_share = person_share.reset_index()

    main = lag[(lag.signal == MAIN_SIGNAL) & lag.valid]
    person_lag = (main.groupby(["pair", "band", "dir", "person", "group"], sort=False)
                  .agg(n=("lag_ms", "size"), mean_ms=("lag_ms", "mean"), sd_ms=("lag_ms", "std"),
                       mean_r=("peak_r", "mean"), n_frames=("n_frames", "sum")).reset_index())

    rows = []
    for (sname, pair, dname), d in lag.groupby(["signal", "pair", "dir"], sort=False):
        ok = d[d.valid]
        rows.append({
            "signal": sname, "pair": pair, "dir": dname,
            "n_valid": len(ok), "n_excluded": len(d) - len(ok),
            "mean_frames": d.n_frames.mean(), "mean_lag_ms": ok.lag_ms.mean(), "mean_r": ok.peak_r.mean(),
            "icc": icc_oneway(ok.lag_ms, ok.person) if ok.person.nunique() >= 2 else np.nan,
        })
    return person_share, person_lag, pd.DataFrame(rows)


# =====================================================================
# 6. 図
# =====================================================================
def _wrap(v):
    """8 方向の値の最後に最初の値を足して 0° と 360° をつなぐ"""
    return np.append(v, v[0])


def plot_by_dir(path, df, value_cols, titles, ylabel, xlabel, hline=None):
    """
    行 = 帯域，列 = value_cols，横軸 = 8 方向。被験者ごとの細線と、グループ平均の太線。
    hline: 方向で分けない値（dir == "all"）の表。グループ平均を破線で示す。
    """
    bands = ["slow", "fast"]
    fig, axs = plt.subplots(len(bands), len(value_cols), figsize=(3.6 * len(value_cols), 3.3 * len(bands)),
                            sharex=True, sharey=True, squeeze=False)
    x = np.arange(N_DIR + 1)
    for r, band in enumerate(bands):
        d = df[(df.band == band) & df.dir.isin(DIRS)]
        for c, col in enumerate(value_cols):
            ax = axs[r, c]
            for (person, group), g in d.groupby(["person", "group"]):
                v = g.set_index("dir")[col].reindex(DIRS).to_numpy(float)
                ax.plot(x, _wrap(v), color=GROUP_COLOR.get(group, "0.5"), lw=1, alpha=0.35)
            for group, g in d.groupby("group"):
                v = g.groupby("dir")[col].mean().reindex(DIRS).to_numpy(float)
                ax.plot(x, _wrap(v), color=GROUP_COLOR.get(group, "0.3"), lw=2.2, marker="o", ms=5,
                        label=f"{group} mean")
                if hline is not None:
                    h = hline[(hline.band == band) & (hline.group == group)][col].mean()
                    ax.axhline(h, color=GROUP_COLOR.get(group, "0.3"), lw=1, ls="--")
            ax.axhline(0, color="gray", lw=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{n}\n{a}°" for n, a in zip(DIRS + DIRS[:1], DIR_DEG + [360])], fontsize=8)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(titles[c])
            if c == 0:
                ax.set_ylabel(f"{band}\n{ylabel}")
    axs[0, -1].legend(fontsize=8, loc="best")
    fig.supxlabel(f"{xlabel} (R = right, U = up)  thin = participant, dashed = all directions")
    save_fig(fig, path, DPI)


# =====================================================================
# メイン
# =====================================================================
def run(source=SOURCE):
    """出力は <cfg.OUT>/<OUT_NAME>/"""
    out_dir = HERE / cfg.OUT / OUT_NAME
    xlabel = f"{source} direction on screen"

    share, lag = collect(source)
    person_share, person_lag, summary = summarize(share, lag)

    out_dir.mkdir(parents=True, exist_ok=True)
    share.to_csv(out_dir / "all_share.csv", index=False)
    lag.to_csv(out_dir / "all_lag.csv", index=False)
    person_share.to_csv(out_dir / "person_share.csv", index=False)
    person_lag.to_csv(out_dir / "person_lag.csv", index=False)
    summary.to_csv(out_dir / "summary_lag.csv", index=False)

    names = [k.replace("_", " ") for k in LINKS]
    all_rows = person_share[person_share.dir == "all"]
    for kind, label in [("proj", "projection share"), ("pow", "power share")]:
        cols = [f"share_{kind}_{k}_mean" for k in LINKS]
        plot_by_dir(out_dir / f"share_{kind}_by_dir.png", person_share, cols, names, label, xlabel,
                    hline=all_rows)

    pairs = list(dict.fromkeys(person_lag.pair))
    wide = person_lag.pivot_table(index=["person", "group", "band", "dir"], columns="pair",
                                  values="mean_ms").reset_index()
    plot_by_dir(out_dir / "lag_by_dir.png", wide, pairs, [p.replace("-", " -> ") for p in pairs],
                "lag [ms] (+ = distal later)", xlabel, hline=wide[wide.dir == "all"])

    agree = [c for c in person_share.columns if c.startswith("agree_") and c.endswith("_mean")][0]
    print(person_share[person_share.dir != "all"]
          .groupby(["band", "dir"], sort=False)[[f"share_proj_{k}_mean" for k in LINKS] + [agree]]
          .mean().round(3).to_string())
    print(summary[summary.signal == MAIN_SIGNAL].round(3).to_string(index=False))


if __name__ == "__main__":
    run()