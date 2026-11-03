#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xcorr_trial.py — 試行ごとの関節間の相互相関（変位・速度・加速度）

  python joint_analysis/xcorr_trial.py                                  # recordings/aim*/ の全試行（引数なし = recordings）
  python joint_analysis/xcorr_trial.py recordings                        # 同上
  python joint_analysis/xcorr_trial.py recordings/aim_kakeru400_xxx      # 1 セッション
  python joint_analysis/xcorr_trial.py "recordings/aim_kakeru*"          # パターン（* ? [ は自分で展開する）
  python joint_analysis/xcorr_trial.py recordings/aim_kakeru400_xxx/trial_01_slow_s1   # 1 試行
  対象は recordings/aim*/<試行>/joints/rs_joints.npz の形だけ（calib_*, trial_* など aim 以外は見ない）。
  オプション: --out analysis  --joints shoulder elbow wrist mcp3  --pairs chain|all
              --fc 5  --max-lag 1.0  --include-static  --no-crop
              --fig-width 12.0  --fig-height-scale 1.0  --xlim 0 20  --ext pdf

出力: <out>/<セッション名>/<試行名>/xcorr/
  summary.csv            ペア × 信号（変位・速度・加速度）ごとの r(0)，ピークのラグ・相関
  curves.csv             R(τ) の曲線そのもの（ラグ [ms] × ペア・信号）
  info.json              使った設定，標本化周波数，区間，各関節の有効率，左右の取り違えの割合
  timeseries_<信号>.<ext>  全関節（X, Y, Z）の時系列
  combined_disp.<ext>      全関節の変位を軸（X, Y, Z）ごとに重ね合わせた時系列
  combined_disp_norm.<ext> 全関節の「変位の大きさ（ノルム）」を一つに重ね合わせた時系列
  pair_disp_norm.<ext>     全ペアごとの「変位の大きさ（ノルム）」の比較時系列
  pair_relative_disp_norm.<ext> ペア間の相対変位（例: 肘の変位 - 肩の変位）の大きさ
  pair_<a>-<b>.<ext>       ペアごと: 共通の主軸方向への射影と相互相関
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt

SIGS = ("disp", "vel", "acc")
UNIT = {"disp": "mm", "vel": "mm/s", "acc": "mm/s²"}
NAME_JA = {"disp": "変位", "vel": "速度", "acc": "加速度"}
NAME_EN = {"disp": "displacement", "vel": "velocity", "acc": "acceleration"}
AXC = ("tab:red", "tab:green", "tab:blue")
JA = False   # setup_font() で決まる

JOINT_JA = {
    "shoulder": "肩",
    "elbow": "肘",
    "wrist": "手首",
    "mcp3": "中指の中手指節関節"
}

# ------------------------------------------------------------------ 補助
def setup_font():
    global JA
    import matplotlib
    from matplotlib import font_manager as fm
    
    # グラフの文字サイズを全体的に大きく、太文字に設定
    matplotlib.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',           # 全体の文字を太く
        'axes.labelweight': 'bold',      # 軸ラベルを太く
        'axes.titleweight': 'bold',      # タイトルを太く
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'legend.title_fontsize': 14
    })
    
    have = {f.name for f in fm.fontManager.ttflist}
    for c in ("Yu Gothic", "Meiryo", "IPAexGothic", "IPAGothic", "Noto Sans CJK JP", "Noto Sans JP",
              "Hiragino Sans", "MS Gothic", "TakaoGothic"):
        if c in have:
            matplotlib.rcParams["font.family"] = c
            matplotlib.rcParams["axes.unicode_minus"] = False
            JA = True
            break


def L(ja, en):
    return ja if JA else en


def _trials_in_session(sess):
    return sorted(q for q in sess.iterdir() if q.is_dir() and (q / "joints" / "rs_joints.npz").exists())


def discover(path):
    p = Path(path)
    if not p.is_dir():
        return []
    if (p / "joints" / "rs_joints.npz").exists():
        return [p]
    if p.name.lower().startswith("aim"):
        return _trials_in_session(p)
    found = []
    for d in sorted(p.iterdir()):
        if d.is_dir() and d.name.lower().startswith("aim"):
            found += _trials_in_session(d)
    return found


def collect(paths):
    trials = []
    for pat in paths:
        hits = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        for h in hits:
            trials += discover(h)
    return list(dict.fromkeys(trials))


def trial_kind(trial):
    f = trial / "meta.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8")).get("task", {}).get("kind")
        except Exception:
            return None
    return None


def read_span(trial):
    f = trial / "task.csv"
    if not f.exists():
        return None
    d = np.genfromtxt(f, delimiter=",", names=True)
    t, ep = np.atleast_1d(d["t"]).astype(float), np.atleast_1d(d["epoch"]).astype(float)
    T = 40.0
    sj = trial.parent / "session.json"
    if sj.exists():
        try:
            T = float(json.loads(sj.read_text(encoding="utf-8")).get("config", {}).get("trial_s", T))
        except Exception:
            pass
    T = min(T, float(t.max()))
    return float(np.interp(0.0, t, ep)), float(np.interp(T, t, ep)), T


def hampel_flags(x, k=4, n=3.0):
    pad = np.pad(x, (k, k), mode="edge")
    win = sliding_window_view(pad, 2 * k + 1)
    med = np.median(win, axis=1)
    mad = 1.4826 * np.median(np.abs(win - med[:, None]), axis=1)
    return (mad > 1e-9) & (np.abs(x - med) > n * mad)


def despike(X, k=4, n=3.0):
    X = X.copy()
    ok = np.isfinite(X).all(1)
    if ok.sum() < 2 * k + 2:
        return X, 0
    sub = X[ok]
    bad = np.zeros(len(sub), bool)
    for c in range(3):
        bad |= hampel_flags(sub[:, c], k, n)
    idx = np.flatnonzero(ok)[bad]
    X[idx] = np.nan
    return X, int(bad.sum())


def lowpass(X, fs, fc):
    if len(X) < 40:
        return X.copy()
    b, a = butter(4, min(fc / (fs / 2), 0.99))
    return filtfilt(b, a, X, axis=0)


def to_grid(t, X, grid, gap_limit, fs):
    ok = np.isfinite(X).all(1)
    if ok.sum() < 8:
        return np.full((len(grid), 3), np.nan), np.zeros(len(grid), bool)
    tv, Xv = t[ok], X[ok]
    Y = np.column_stack([np.interp(grid, tv, Xv[:, k]) for k in range(3)])
    i = np.searchsorted(tv, grid)
    prev = tv[np.clip(i - 1, 0, len(tv) - 1)]
    nxt = tv[np.clip(i, 0, len(tv) - 1)]
    near = np.minimum(np.abs(grid - prev), np.abs(nxt - grid))
    inside = (grid >= tv[0] - 0.5 / fs) & (grid <= tv[-1] + 0.5 / fs)
    valid = inside & (((nxt - prev) <= gap_limit) | (near <= 0.5 / fs))
    return Y, valid


def erode(m, k):
    out = m.copy()
    for s in range(1, k + 1):
        out &= np.roll(m, s) & np.roll(m, -s)
    out[:k] = False
    out[len(out) - k:] = False
    return out


def make_signals(X, valid, fs):
    d = X - X[valid].mean(0)
    v = np.gradient(d, axis=0) * fs
    a = np.gradient(v, axis=0) * fs
    return {"disp": (d, valid), "vel": (v, erode(valid, 1)), "acc": (a, erode(valid, 2))}


# ------------------------------------------------------------------ 相互相関
def xcorr(a, ma, b, mb, L_, min_overlap=0.3):
    lags = np.arange(-L_, L_ + 1)
    r = np.full(len(lags), np.nan)
    na, nb = int(ma.sum()), int(mb.sum())
    if na < 10 or nb < 10:
        return lags, r
    N = len(a)
    nfft = 1 << int(np.ceil(np.log2(2 * N)))
    A = np.where(ma[:, None], a - a[ma].mean(0), 0.0)
    B = np.where(mb[:, None], b - b[mb].mean(0), 0.0)
    va, vb = (A ** 2).sum() / na, (B ** 2).sum() / nb
    if va <= 0 or vb <= 0:
        return lags, r
    FA, FB = np.fft.rfft(A, nfft, axis=0), np.fft.rfft(B, nfft, axis=0)
    c = np.fft.irfft((FA * np.conj(FB)).sum(1), nfft)
    cnt = np.fft.irfft(np.fft.rfft(ma.astype(float), nfft) * np.conj(np.fft.rfft(mb.astype(float), nfft)), nfft)
    ii = lags % nfft
    n_ov = np.rint(cnt[ii])
    ok = n_ov >= max(10.0, min_overlap * min(na, nb))
    r[ok] = c[ii][ok] / (n_ov[ok] * np.sqrt(va * vb))
    return lags, r


def refine(lags, r, i, fs):
    if 0 < i < len(r) - 1 and np.isfinite(r[i - 1:i + 2]).all():
        y0, y1, y2 = r[i - 1], r[i], r[i + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            dl = float(np.clip(0.5 * (y0 - y2) / den, -1, 1))
            return (lags[i] + dl) / fs * 1000.0, float(y1 - 0.25 * (y0 - y2) * dl)
    return lags[i] / fs * 1000.0, float(r[i])


def peaks(lags, r, fs):
    if not np.isfinite(r).any():
        return dict(tau=np.nan, rpk=np.nan, tau_pos=np.nan, rpos=np.nan, r0=np.nan, edge=False)
    i = int(np.nanargmax(np.abs(r)))
    j = int(np.nanargmax(r))
    tau, rpk = refine(lags, r, i, fs)
    tp, rp = refine(lags, r, j, fs)
    r0 = float(r[len(r) // 2])
    return dict(tau=tau, rpk=rpk, tau_pos=tp, rpos=rp, r0=r0, edge=bool(i in (0, len(r) - 1)))


# ------------------------------------------------------------------ 図
def plot_all(path, joints, sig, S, tt, width=12.0, h_scale=1.0, xlim=(0.0, 20.0)):
    import matplotlib.pyplot as plt
    fig_height = (2.1 * len(joints) + 0.8) * h_scale
    # 複数グラフ間の上下の隙間をゼロにする (hspace: 0)
    fig, axs = plt.subplots(len(joints), 1, figsize=(width, fig_height), sharex=True, gridspec_kw={"hspace": 0})
    axs = np.atleast_1d(axs)
    for ax, j in zip(axs, joints):
        x, m = S[j][sig]
        y = np.where(m[:, None], x, np.nan)
        for k, lab in enumerate("XYZ"):
            ax.plot(tt, y[:, k], lw=1.2, color=AXC[k], label=lab)
        ja_j = JOINT_JA.get(j, j)
        ax.set_ylabel(f"{ja_j}\n[{UNIT[sig]}]")
        ax.grid(alpha=0.3)
    axs[0].legend(ncol=3, loc="upper right", title=L("カメラ座標の軸", "camera axes"))
    axs[-1].set_xlabel(L("時間 [s]", "time [s]"))
    axs[-1].set_xlim(xlim[0], xlim[1])
    # h_pad=0 で余白を徹底的に削る
    fig.tight_layout(h_pad=0)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_combined_disp(path, joints, S, tt, width=12.0, h_scale=1.0, xlim=(0.0, 20.0)):
    import matplotlib.pyplot as plt
    # 複数グラフ間の上下の隙間をゼロにする (hspace: 0)
    fig, axs = plt.subplots(3, 1, figsize=(width, 7.5 * h_scale), sharex=True, gridspec_kw={"hspace": 0})
    labels = ["X", "Y", "Z"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(joints)))
    for ax, k, lab in zip(axs, range(3), labels):
        for idx, j in enumerate(joints):
            x, m = S[j]["disp"]
            y = np.where(m, x[:, k], np.nan)
            ja_j = JOINT_JA.get(j, j)
            ax.plot(tt, y, lw=1.5, color=colors[idx], label=ja_j, alpha=0.8)
        ax.set_ylabel(f"{lab} {L('変位', 'Disp.')}\n[mm]")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(ncol=len(joints), loc="upper right", title=L("関節", "Joints"))
    axs[-1].set_xlabel(L("時間 [s]", "time [s]"))
    axs[-1].set_xlim(xlim[0], xlim[1])
    fig.tight_layout(h_pad=0)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_combined_disp_norm(path, joints, S, tt, width=12.0, h_scale=1.0, xlim=(0.0, 20.0)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width, 4.0 * h_scale))
    colors = plt.cm.tab10(np.linspace(0, 1, len(joints)))
    for idx, j in enumerate(joints):
        x, m = S[j]["disp"]
        norm = np.linalg.norm(x, axis=1)
        y = np.where(m, norm, np.nan)
        ja_j = JOINT_JA.get(j, j)
        ax.plot(tt, y, lw=1.5, color=colors[idx], label=ja_j, alpha=0.8)
    ax.set_ylabel(L("変位の大きさ\n[mm]", "Disp. Magnitude\n[mm]"))
    ax.set_xlabel(L("時間 [s]", "time [s]"))
    ax.set_xlim(xlim[0], xlim[1])
    ax.grid(alpha=0.3)
    ax.legend(ncol=len(joints), loc="upper right", title=L("関節", "Joints"))
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_pair_disp_norm(path, pairs, S, tt, width=12.0, h_scale=1.0, xlim=(0.0, 20.0)):
    import matplotlib.pyplot as plt
    if not pairs:
        return
    fig_height = (2.5 * len(pairs) + 0.8) * h_scale
    fig, axs = plt.subplots(len(pairs), 1, figsize=(width, fig_height), sharex=True, gridspec_kw={"hspace": 0})
    axs = np.atleast_1d(axs)
    for ax, (a, b) in zip(axs, pairs):
        xa, ma = S[a]["disp"]
        xb, mb = S[b]["disp"]
        norm_a = np.where(ma, np.linalg.norm(xa, axis=1), np.nan)
        norm_b = np.where(mb, np.linalg.norm(xb, axis=1), np.nan)
        ja_a, ja_b = JOINT_JA.get(a, a), JOINT_JA.get(b, b)
        ax.plot(tt, norm_a, lw=1.5, color="tab:blue", label=ja_a, alpha=0.8)
        ax.plot(tt, norm_b, lw=1.5, color="tab:orange", label=ja_b, alpha=0.8)
        ax.set_ylabel(f"{ja_a} & {ja_b}\n[mm]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axs[-1].set_xlabel(L("時間 [s]", "time [s]"))
    axs[-1].set_xlim(xlim[0], xlim[1])
    fig.tight_layout(h_pad=0)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_pair_relative_disp_norm(path, pairs, S, tt, width=12.0, h_scale=1.0, xlim=(0.0, 20.0)):
    import matplotlib.pyplot as plt
    if not pairs:
        return
    fig_height = (2.5 * len(pairs) + 0.8) * h_scale
    fig, axs = plt.subplots(len(pairs), 1, figsize=(width, fig_height), sharex=True, gridspec_kw={"hspace": 0})
    axs = np.atleast_1d(axs)
    for ax, (a, b) in zip(axs, pairs):
        xa, ma = S[a]["disp"]
        xb, mb = S[b]["disp"]
        
        valid = ma & mb
        rel_disp = xb - xa  
        norm_rel = np.where(valid, np.linalg.norm(rel_disp, axis=1), np.nan)
        
        ja_a, ja_b = JOINT_JA.get(a, a), JOINT_JA.get(b, b)
        
        ax.plot(tt, norm_rel, lw=1.5, color="tab:purple", alpha=0.9)
        ax.set_ylabel(f"|| {ja_b} - {ja_a} ||\n[mm]")
        ax.grid(alpha=0.3)
        ax.legend([L(f"{ja_b} と {ja_a} の相対変位の大きさ", f"Relative disp. mag. ({b}-{a})")], loc="upper right")
        
    axs[-1].set_xlabel(L("時間 [s]", "time [s]"))
    axs[-1].set_xlim(xlim[0], xlim[1])
    fig.tight_layout(h_pad=0)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_pair(path, a, b, S, res, tt, fs, max_lag_ms, width=12.0, h_scale=1.0, xlim=(0.0, 20.0)):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(width * (14.0 / 12.0), 8.5 * h_scale), 
                            gridspec_kw={"width_ratios": [3, 1.35], "hspace": 0}, sharex='col')
    ja_a, ja_b = JOINT_JA.get(a, a), JOINT_JA.get(b, b)
    
    for row, sig in enumerate(SIGS):
        (xa, ma), (xb, mb) = S[a][sig], S[b][sig]
        both = np.vstack([xa[ma] - xa[ma].mean(0), xb[mb] - xb[mb].mean(0)]) if ma.any() and mb.any() else None
        ax, ar = axs[row]
        if both is not None and len(both) > 5:
            u = np.linalg.svd(both, full_matrices=False)[2][0]
            ya = np.where(ma, (xa - xa[ma].mean(0)) @ u, np.nan)
            yb = np.where(mb, (xb - xb[mb].mean(0)) @ u, np.nan)
            ax.plot(tt, ya, lw=1.2, color="tab:blue", label=ja_a)
            ax.plot(tt, yb, lw=1.2, color="tab:orange", label=ja_b)
            if row == 0:
                ax.legend(loc="upper right", ncol=2)
        ax.set_ylabel(f"{L(NAME_JA[sig], NAME_EN[sig])}\n[{UNIT[sig]}]")
        ax.grid(alpha=0.3)
        ax.set_xlim(xlim[0], xlim[1])
        
        if row == 2:
            ax.set_xlabel(L("時間 [s]（共通の主軸方向への射影）", "time [s] (projection on the common principal axis)"))
        else:
            ax.tick_params(labelbottom=False)
        
        R = res[sig]
        lag_ms = R["lags"] / fs * 1000.0
        ar.plot(lag_ms, R["r"], color="k", lw=1.5)
        ar.axhline(0, color="gray", lw=1.0)
        ar.axvline(0, color="gray", lw=1.0, ls=":")
        p = R["pk"]
        if np.isfinite(p["tau"]):
            ar.axvline(p["tau"], color="tab:red", lw=1.5)
            who = (L(f"{ja_a} が先行", f"{ja_a} leads") if p["tau"] < 0 else L(f"{ja_b} が先行", f"{ja_b} leads")) \
                if abs(p["tau"]) > 1e-9 else L("同時", "simultaneous")
            ar.set_title(f"τ = {p['tau']:+.0f} ms, r = {p['rpk']:+.2f}（{who}）", fontsize=14, weight='bold')
        ar.set_xlim(-max_lag_ms, max_lag_ms)
        ar.set_ylim(-1.02, 1.02)
        ar.set_ylabel("r(τ)")
        ar.grid(alpha=0.3)
        if row == 2:
            ar.set_xlabel(L("ラグ τ [ms]  （+ = a が遅れ，− = a が先行）", "lag τ [ms]  (+ = a lags, − = a leads)"))
        else:
            ar.tick_params(labelbottom=False)
            
    fig.tight_layout(h_pad=0, w_pad=1.0)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------ 試行
def process_trial(trial, out_root, args):
    session = trial.parent.name
    out = Path(out_root) / session / trial.name / "xcorr"
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(trial / "joints" / "rs_joints.npz", allow_pickle=False)
    names = [str(n) for n in z["names"]]
    miss = [j for j in args.joints if j not in names]
    if miss:
        raise SystemExit(f"[ERROR] 関節 {miss} が rs_joints.npz にない（ある関節: {names}）")
    te = z["t"].astype(float)
    P = z["P_C"].astype(float)
    swapped = z["lr_swapped"].astype(bool) if "lr_swapped" in z.files else None

    span = None if args.no_crop else read_span(trial)
    if span:
        e0, e1, T = span
        sel = (te >= e0) & (te < e1)
        crop = f"task.csv の [0, {T:g}) s"
    else:
        sel = np.isfinite(te)
        e0, e1, T = float(te[sel][0]), float(te[sel][-1]), float(te[sel][-1] - te[sel][0])
        crop = "全フレーム（task.csv なし or --no-crop）"
    te_s = te[sel]
    order = np.argsort(te_s)
    te_s = te_s[order]
    keep = np.r_[True, np.diff(te_s) > 1e-6]
    te_s = te_s[keep]
    if len(te_s) < 30:
        info = {"status": "skipped", "reason": f"フレームが少ない（{len(te_s)}）"}
        (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        return info
    fs = 1.0 / float(np.median(np.diff(te_s)))
    if args.fc > 0.9 * fs / 2:
        print(f"[WARN] {trial.name}: fs = {fs:.1f} Hz に対し fc = {args.fc:g} Hz はナイキスト（{fs / 2:.1f} Hz）に近く、"
              f"低域通過がほとんど効かない（ノイズが残る）")
    n = int((e1 - e0) * fs)
    grid = e0 + np.arange(n) / fs
    tt = grid - e0

    S, valid_frac, raw_frac, n_spikes = {}, {}, {}, {}
    for j in args.joints:
        X = P[sel][order][keep][:, names.index(j), :]
        raw_frac[j] = float(np.isfinite(X).all(1).mean())
        X, n_spike = despike(X)
        n_spikes[j] = n_spike
        Y, valid = to_grid(te_s, X, grid, args.gap, fs)
        if valid.sum() < 10:
            S[j] = None
            valid_frac[j] = 0.0
            continue
        Yf = lowpass(Y, fs, args.fc)
        S[j] = make_signals(Yf, valid, fs)
        valid_frac[j] = float(valid.mean())
    usable = [j for j in args.joints if S[j] is not None]
    if len(usable) < 2:
        info = {"status": "skipped", "reason": "有効な関節が 2 つ未満", "valid_frac": valid_frac}
        (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        return info

    if args.pairs == "all":
        pairs = [(a, b) for a, b in itertools.combinations(args.joints, 2)]
    else:
        pairs = [(a, b) for a, b in zip(args.joints[:-1], args.joints[1:])]
    pairs = [(a, b) for a, b in pairs if S[a] is not None and S[b] is not None]

    L_ = int(min(args.max_lag * fs, n // 4))
    rows, curves = [], {}
    results = {}
    for a, b in pairs:
        results[(a, b)] = {}
        for sig in SIGS:
            xa, ma = S[a][sig]
            xb, mb = S[b][sig]
            lags, r = xcorr(xa, ma, xb, mb, L_)
            pk = peaks(lags, r, fs)
            results[(a, b)][sig] = dict(lags=lags, r=r, pk=pk)
            curves[f"{a}-{b}_{sig}"] = r
            lead = ("" if not np.isfinite(pk["tau"]) else
                    "同時" if abs(pk["tau"]) < 1e-9 else (f"{a} が先行" if pk["tau"] < 0 else f"{b} が先行"))
            rows.append(dict(session=session, trial=trial.name, a=a, b=b, signal=sig,
                             n_overlap=int(np.sum(ma & mb)), r0=pk["r0"],
                             tau_peak_ms=pk["tau"], r_peak=pk["rpk"], lead=lead,
                             tau_pos_ms=pk["tau_pos"], r_pos=pk["rpos"],
                             at_window_edge=pk["edge"]))

    with open(out / "summary.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r_ in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r_.items()})
    lag_ms = np.arange(-L_, L_ + 1) / fs * 1000.0
    with open(out / "curves.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lag_ms"] + list(curves.keys()))
        for i in range(len(lag_ms)):
            w.writerow([round(lag_ms[i], 3)] + [("" if not np.isfinite(c[i]) else round(float(c[i]), 5))
                                                for c in curves.values()])

    if not args.no_plots:
        setup_font()
        w, h, xl, ext = args.fig_width, args.fig_height_scale, tuple(args.xlim), args.ext
        for sig in SIGS:
            plot_all(out / f"timeseries_{sig}.{ext}", usable, sig, S, tt, w, h, xl)
        plot_combined_disp(out / f"combined_disp.{ext}", usable, S, tt, w, h, xl)
        plot_combined_disp_norm(out / f"combined_disp_norm.{ext}", usable, S, tt, w, h, xl)
        plot_pair_disp_norm(out / f"pair_disp_norm.{ext}", pairs, S, tt, w, h, xl)
        plot_pair_relative_disp_norm(out / f"pair_relative_disp_norm.{ext}", pairs, S, tt, w, h, xl)
        
        for a, b in pairs:
            plot_pair(out / f"pair_{a}-{b}.{ext}", a, b, S, results[(a, b)], tt, fs, L_ / fs * 1000.0, w, h, xl)

    info = dict(status="ok", session=session, trial=trial.name, kind=trial_kind(trial), crop=crop,
                span_s=[0.0, round(T, 3)] if span else [0.0, round(float(tt[-1]), 3)],
                fs_hz=round(fs, 3), n_samples=n, joints=args.joints, pairs=[f"{a}-{b}" for a, b in pairs],
                fc_hz=args.fc, max_lag_s=args.max_lag, gap_limit_s=args.gap,
                valid_frac_raw=raw_frac, valid_frac_grid=valid_frac, n_spikes_removed=n_spikes,
                lr_swapped_frac=(None if swapped is None else round(float(swapped[sel].mean()), 4)),
                sign_convention="R(tau)=sum a(t)b(t-tau); tau>0 -> a lags b (b leads); tau<0 -> a leads")
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def main():
    ap = argparse.ArgumentParser(description="試行ごとの関節間の相互相関")
    ap.add_argument("path", nargs="*", default=["recordings"],
                    help="recordings（既定。中の aim* の全試行）/ aim* のセッション / 試行のフォルダ。"
                         "recordings/aim_kakeru* のようなパターンも可")
    ap.add_argument("--out", default="analysis", help="出力のルート（既定 analysis）")
    ap.add_argument("--joints", nargs="+", default=["shoulder", "elbow", "wrist", "mcp3"],
                    help="関節（連結の順）。手首を Pose 側にするなら wrist_pose")
    ap.add_argument("--pairs", choices=["chain", "all"], default="chain", help="chain = 隣どうし, all = 全組み合わせ")
    ap.add_argument("--fc", type=float, default=5.0, help="低域通過 [Hz]（既定 5）")
    ap.add_argument("--max-lag", type=float, default=1.0, help="ラグの範囲 ±[s]")
    ap.add_argument("--gap", type=float, default=0.5, help="これより長く欠けた区間は欠損のまま [s]")
    ap.add_argument("--include-static", action="store_true", help="静止の試行も処理する")
    ap.add_argument("--no-crop", action="store_true", help="task.csv があっても全フレームを使う")
    ap.add_argument("--no-plots", action="store_true")
    # 図の描画設定オプション
    ap.add_argument("--fig-width", type=float, default=12.0, help="グラフの基準幅（インチ）。既定: 12.0")
    ap.add_argument("--fig-height-scale", type=float, default=1.0, help="グラフの高さの倍率。既定: 1.0 (例: 0.5で高さを半減)")
    ap.add_argument("--xlim", type=float, nargs=2, default=[0.0, 20.0], help="グラフのX軸（時間）の描画範囲[s]（開始 終了）。既定: 0.0 20.0")
    ap.add_argument("--ext", choices=["png", "pdf", "svg"], default="png", help="画像の出力フォーマット (既定: png)")
    args = ap.parse_args()

    trials = collect(args.path)
    if not trials:
        sys.exit(f"[ERROR] 対象の試行が見つからない: {args.path}\n"
                 f"  recordings/aim*/<試行>/joints/rs_joints.npz の形で置かれているか確認（実行した場所: {Path.cwd()}）")
    by_session = {}
    for tr in trials:
        by_session.setdefault(tr.parent.name, []).append(tr.name)
    print(f"[対象] {len(by_session)} セッション, {len(trials)} 試行")
    for s, ts in by_session.items():
        print(f"  {s}: {len(ts)} 試行")
    t0 = time.time()
    done = skipped = 0
    for k, tr in enumerate(trials, 1):
        if not args.include_static and ("static" in tr.name.lower() or trial_kind(tr) == "static"):
            print(f"[{k}/{len(trials)}] {tr.parent.name}/{tr.name}: 静止の試行なので飛ばす（--include-static で処理）")
            skipped += 1
            continue
        try:
            info = process_trial(tr, args.out, args)
        except SystemExit:
            raise
        except Exception as e:
            print(f"[{k}/{len(trials)}] {tr.parent.name}/{tr.name}: 失敗 {type(e).__name__}: {e}")
            skipped += 1
            continue
        if info["status"] == "ok":
            print(f"[{k}/{len(trials)}] {tr.parent.name}/{tr.name}: fs {info['fs_hz']:.1f} Hz, "
                  f"有効率 " + " ".join(f"{j}={v:.0%}" for j, v in info["valid_frac_grid"].items()))
            done += 1
        else:
            print(f"[{k}/{len(trials)}] {tr.parent.name}/{tr.name}: 飛ばした（{info['reason']}）")
            skipped += 1
    print(f"[DONE] 処理 {done} 試行，飛ばした {skipped} 試行，{time.time() - t0:.0f} s -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()