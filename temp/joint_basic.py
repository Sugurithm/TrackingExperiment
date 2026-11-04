#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
joint_basic.py — 関節の時系列から 3 つの基本解析を行う
  1. 平均と分散    （どれだけ動かしたか）
  2. 大きさと位相  （ターゲットの周波数ごとに、いつ動いたか）
  3. 相互相関      （2 で求めた遅れを時間領域で確認）

  python temp/joint_basic.py

対象のセッション・低域通過・出力先・やり直し試行の扱いは compare_config.py から読む。
出力: temp/<OUT>/joint_basic/<label>/<試行>/  stats.csv, phase.csv, xcorr.csv, xcorr.png
      temp/<OUT>/joint_basic/all_stats.csv, all_phase.csv, all_xcorr.csv（全試行をまとめたもの）

座標の向き:
  カメラ座標のままでは画面の横・縦と一致しないので、試行ごとに
  「手先（MCP3）がターゲットの x の周波数で動く向き」を h（横）、
  「y の周波数で動く向き」を v（縦）、両方に垂直な向きを n とし、全関節をこの 3 軸に直す。
  符号（右が正か左が正か）は決めていない。大きさ・比・遅れには影響しない。
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

HERE = Path(__file__).resolve().parent        # temp/
ROOT = HERE.parent                            # リポジトリ直下
sys.path.insert(0, str(HERE))
import compare_config as cfg                  # noqa: E402

# ---------- この解析だけの設定（config にないもの） ----------
JOINTS = ["shoulder", "elbow", "wrist", "mcp3"]   # 関節（連結の順）
REF = "mcp3"                                      # 基準の関節（手先）

FILTER_ORDER = 4
T_TRIAL = 40.0                                    # ターゲットの周波数 f = n / T_TRIAL
FREQ_N = {
    "slow": {"h": [2, 5, 11, 17], "v": [3, 7, 13, 19]},
    "fast": {"h": [3, 7, 13, 19, 29, 41], "v": [5, 11, 17, 23, 31, 43]},
}
AMP_MIN = 5.0        # 手先の振幅がこれ未満の周波数は位相を信用しない [mm]
MAX_LAG = 0.3        # 相互相関でずらす範囲 [s]（±）

OUT_DIR = HERE / cfg.OUT / "joint_basic"


# ---------- 対象の試行を集める ----------
def session_info(s):
    """config の SESSIONS の 1 行から、フォルダと人・群・dpi・呼び名を決める"""
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
    """trial_01_slow_s1_2 → ("slow", 1, 2)（帯域, seed, やり直しの番号）"""
    m = re.search(r"(slow|fast)_s(\d+)(?:_(\d+))?", name.lower())
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3) or 1)


def find_trials(session_dir):
    """セッション内の試行を集め、やり直した試行は cfg.ATTEMPT に従って選ぶ"""
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


# ---------- 読み込みと前処理 ----------
def load(trial):
    """戻り値: 時間 tt [s], 標本化周波数 fs, 平均位置 {関節: (3,)}, 変位 {関節: (N,3)}（カメラ座標）"""
    z = np.load(trial / "joints" / "rs_joints.npz")
    names = [str(n) for n in z["names"]]
    t = z["t"].astype(float)
    P = z["P_C"].astype(float)

    order = np.argsort(t)
    t, P = t[order], P[order]

    fs = 1.0 / np.median(np.diff(t))
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    b, a = butter(FILTER_ORDER, min(cfg.FC / (fs / 2), 0.99))

    mean, disp = {}, {}
    for j in JOINTS:
        X = P[:, names.index(j), :]
        ok = np.isfinite(X).all(axis=1)
        Y = np.column_stack([np.interp(grid, t[ok], X[ok, k]) for k in range(3)])
        Y = filtfilt(b, a, Y, axis=0)
        mean[j] = Y.mean(axis=0)
        disp[j] = Y - mean[j]
    return grid - grid[0], fs, mean, disp


def fit_complex(tt, X, freqs):
    """X(t) ≈ Re(Z e^{i 2πf t}) となる複素振幅 Z を最小二乗で求める。戻り値 (周波数数, 列数)"""
    cols = [np.ones_like(tt)]
    for f in freqs:
        w = 2 * np.pi * f * tt
        cols += [np.cos(w), np.sin(w)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return coef[1::2] - 1j * coef[2::2]


def screen_axes(tt, X_ref, band):
    """手先がターゲットの x / y の周波数で動く向きから、横 h・縦 v・垂直 n の単位ベクトルを作る"""
    def main_dir(freqs):
        Z = fit_complex(tt, X_ref, freqs)                  # (周波数数, 3)
        M = np.vstack([Z.real, Z.imag])                    # 実部と虚部をまとめて主成分
        return np.linalg.svd(M, full_matrices=False)[2][0]
    h = main_dir([n / T_TRIAL for n in FREQ_N[band]["h"]])
    v = main_dir([n / T_TRIAL for n in FREQ_N[band]["v"]])
    v = v - (v @ h) * h                                     # h と直交させる
    v /= np.linalg.norm(v)
    n = np.cross(h, v)
    return np.vstack([h, v, n])                            # (3, 3)。行が h, v, n


def to_screen(mean, disp, R):
    """カメラ座標 → (h, v, n)"""
    return {j: R @ mean[j] for j in JOINTS}, {j: disp[j] @ R.T for j in JOINTS}


# ---------- 1. 平均と分散 ----------
def analyze_stats(mean, disp):
    rows = []
    for j in JOINTS:
        sd = disp[j].std(axis=0)
        rows.append({
            "joint": j,
            "mean_h": mean[j][0], "mean_v": mean[j][1], "mean_n": mean[j][2],
            "sd_h": sd[0], "sd_v": sd[1], "sd_n": sd[2],
            "sd_total": np.sqrt((sd ** 2).sum()),
        })
    return rows


# ---------- 2. 大きさと位相 ----------
def analyze_phase(tt, disp, band):
    """横の周波数は h 成分、縦の周波数は v 成分だけを見る"""
    rows = []
    for k_axis, axis in enumerate(["h", "v"]):
        ns = FREQ_N[band][axis]
        freqs = [n / T_TRIAL for n in ns]
        Z = {j: fit_complex(tt, disp[j][:, k_axis], freqs) for j in JOINTS}   # (周波数数,)
        for k, f in enumerate(freqs):
            Zr = Z[REF][k]
            for j in JOINTS:
                c = Z[j][k] / Zr                                 # 手先に対する比
                rows.append({
                    "axis": axis, "n": ns[k], "freq_hz": f, "joint": j,
                    "amp_mm": np.abs(Z[j][k]),                   # この周波数での動きの大きさ
                    "gain": np.abs(c),                           # 手先に対する大きさの比
                    "phase_deg": np.degrees(np.angle(c)),        # 手先に対する位相
                    "lag_ms": -np.angle(c) / (2 * np.pi * f) * 1000,   # 正 = 手先より遅れる
                    "valid": np.abs(Zr) >= AMP_MIN,              # 手先が十分動いた周波数か
                })
    return rows


# ---------- 3. 相互相関 ----------
def xcorr(a, b, max_lag):
    """ずらすごとの相関係数。戻り値: ずらし量 [標本], 相関, ピークのずらし量（補間）, ピークの相関
       ずらし量が正 = b が a より遅れる"""
    n = len(a)
    lags = np.arange(-max_lag, max_lag + 1)
    r = []
    for k in lags:
        x, y = (a[:n - k], b[k:]) if k >= 0 else (a[-k:], b[:n + k])   # 重なる部分だけ
        r.append(np.corrcoef(x, y)[0, 1])
    r = np.array(r)

    i = np.argmax(np.abs(r))                 # 逆向き（負の相関）のピークも拾う
    shift = 0.0
    if 0 < i < len(r) - 1:                   # 放物線補間で 1 標本より細かく
        y0, y1, y2 = np.abs(r[i - 1:i + 2])
        shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return lags, r, lags[i] + shift, r[i]


def analyze_xcorr(disp, fs):
    """隣り合う関節の変位を、横 h と縦 v に分けて相互相関をとる
       （速度にすると高い周波数のノイズが強調され、遅れが偏るので変位を使う）"""
    rows, curves = [], []
    for k_axis, axis in enumerate(["h", "v"]):
        for a, b in zip(JOINTS[:-1], JOINTS[1:]):
            lags, r, peak, peak_r = xcorr(disp[a][:, k_axis], disp[b][:, k_axis], int(round(MAX_LAG * fs)))
            rows.append({"axis": axis, "pair": f"{a}-{b}", "lag_ms": peak / fs * 1000, "peak_r": peak_r})
            curves.append((axis, f"{a}-{b}", lags / fs * 1000, r))
    return rows, curves


def plot_xcorr(path, curves):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, axis in zip(axs, ["h", "v"]):
        for ax_name, label, lag_ms, r in curves:
            if ax_name == axis:
                ax.plot(lag_ms, r, label=label)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(f"axis {axis}")
        ax.set_xlabel("lag [ms]  (+ = later joint lags)")
        ax.set_ylim(-1, 1)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("correlation")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------- 出力 ----------
def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def tag(rows, info):
    """各行の先頭にセッション・試行などの情報を付ける"""
    return [{**info, **r} for r in rows]


# ---------- メイン ----------
def main():
    all_stats, all_phase, all_xcorr = [], [], []
    for s in cfg.SESSIONS:
        ses = session_info(s)
        if not ses["dir"].exists():
            print(f"見つかりません: {ses['dir']}")
            continue
        for band, seed, attempt, trial in find_trials(ses["dir"]):
            info = {"label": ses["label"], "person": ses["person"], "group": ses["group"], "dpi": ses["dpi"],
                    "trial": trial.name, "band": band, "seed": seed, "attempt": attempt}
            out = OUT_DIR / ses["label"] / trial.name
            out.mkdir(parents=True, exist_ok=True)

            tt, fs, mean, disp = load(trial)
            R = screen_axes(tt, disp[REF], band)
            mean, disp = to_screen(mean, disp, R)

            stats = tag(analyze_stats(mean, disp), info)
            phase = tag(analyze_phase(tt, disp, band), info)
            xc_rows, curves = analyze_xcorr(disp, fs)
            xc = tag(xc_rows, info)

            write_csv(out / "stats.csv", stats)
            write_csv(out / "phase.csv", phase)
            write_csv(out / "xcorr.csv", xc)
            plot_xcorr(out / "xcorr.png", curves)

            all_stats += stats
            all_phase += phase
            all_xcorr += xc
            print(f"完了: {ses['label']}/{trial.name}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "all_stats.csv", all_stats)
    write_csv(OUT_DIR / "all_phase.csv", all_phase)
    write_csv(OUT_DIR / "all_xcorr.csv", all_xcorr)


if __name__ == "__main__":
    main()