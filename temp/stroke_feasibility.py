#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stroke_feasibility.py — 「ターゲットの動きの大きさ」で関節の使い分けを調べられるか（実現可能性チェック）

  python temp/stroke_feasibility.py                         # recordings/aim_*/ の全試行
  python temp/stroke_feasibility.py recordings/aim_X_...    # 1 セッションだけ
  dpi: フォルダ名の数字（aim_Suguru400_…）を本当の dpi とし，session.json の dpi が違えば cm を直す．--trust-json で無効
  オプション: --out temp/out  --fc 3  --bins 5 10 20 40 80
              --min-strokes 15  --snr 2  --min-cov 0.8

目的（統計的な有意性は見ない）
  1. ターゲットの 1 ストローク（目標の速さの谷から谷まで）の大きさ A [mm] の分布は，
     関節の位置のノイズより十分大きいか．
  2. 人 × 条件（dpi・帯域）× 大きさのビンごとに，使えるストロークが何本あるか．
  3. A と最大速度は，どれくらい重なっているか（大きさと速さを分けられるか）．

方法
  - ストロークは task.csv のターゲット位置（t >= 0）から切る．成績に依らない．
    単位は cm（マウスの移動量）なので A_mm = 10 * A_cm．
  - 手の応答の遅れ tau は，cursor が target を tau だけ遅れて追う，と当てはめて 1 試行に 1 つ求め，
    関節の窓を tau だけずらす．（task.csv の epoch と rs_joints.npz の t は同じ PC の時計）
  - 手先 = mcp3 の変位 D を，次の 5 つに足し算で分ける（足すと D になる）:
        S 肩の点   U 上腕 (elbow - shoulder)   F 前腕 (wrist_pose - elbow)
        M モデルのずれ (wrist - wrist_pose)    H 手 (mcp3 - wrist)
    各リンクの変位を D の向きに射影した量 s_k [mm]，割合 s_k / |D| を出す．
  - ノイズは，記録の高い周波数帯（既定 5 Hz 以上）を白色とみなして軸ごとに見積もる．
    ストロークの変位のノイズ = sqrt(2 * 位置のノイズ分散)．|s_k| > snr * ノイズ のとき「検出可能」とする．
    （MediaPipe のノイズは完全な白色ではないので，ノイズは少なめに見積もられる点に注意）

入力: recordings/aim_*/trial_NN_<slow|fast>_sS[_k]/{task.csv, joints/rs_joints.npz}
      （練習・静止の試行は対象外）
出力: <out>/strokes.csv      ストロークごと（A, 速さ, 各リンクの変位・ノイズ・割合，有効かどうか）
      <out>/trials.csv       試行ごと（遅れ，ノイズ，被覆率，ストローク数）
      <out>/counts.csv       人 × dpi × 帯域 × 大きさのビンごとの本数
      <out>/report.txt       要約（判定つき）
      <out>/feasibility.png  図
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import spearmanr

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
LINKS = ["S", "U", "F", "M", "H"]
LINK_NAME = {"S": "shoulder pt", "U": "upper arm", "F": "forearm", "M": "model gap", "H": "hand"}
TRIAL_RE = re.compile(r"^trial_(\d+)_(slow|fast)_s(\d+)(?:_(\d+))?$")


# ---------------------------------------------------------------- 読み込み
def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def session_info(sdir, trust_json=False):
    """(参加者, 本当の dpi, 記録時に宣言された dpi)．

    フォルダ名 aim_<名前><dpi>_<日付>_<時刻> に dpi があれば，それを本当の dpi とする
    （記録時に --dpi を付け忘れると session.json は既定値のままになるため）．--trust-json で session.json を優先．
    """
    name = os.path.basename(os.path.normpath(sdir))
    sj = read_json(os.path.join(sdir, "session.json"))
    m = re.match(r"^aim_(.+?)(\d{3,4})?_\d{8}_\d{6}$", name)
    person = (m.group(1) if m else None) or re.sub(r"\d+$", "", str(sj.get("participant") or name))
    declared = None
    sens = sj.get("sensitivity") or {}
    for key in ("mouse_dpi", "dpi"):
        if isinstance(sens, dict) and sens.get(key):
            declared = float(sens[key])
            break
    from_name = float(m.group(2)) if (m and m.group(2)) else None
    true = declared if (trust_json or from_name is None) else from_name
    if true is None:
        true = declared
    if declared is None:
        declared = true
    return str(person), (true if true is not None else float("nan")), (declared if declared is not None else float("nan"))


def discover(paths):
    """[(session_dir, trial_dir, band, seed, attempt)]"""
    out = []
    for p in paths:
        cands = []
        for q in glob.glob(p):
            if os.path.isdir(q):
                cands.append(q)
        for q in cands:
            sessions = [q] if os.path.basename(q).startswith("aim_") else sorted(glob.glob(os.path.join(q, "aim_*")))
            for s in sessions:
                for t in sorted(glob.glob(os.path.join(s, "trial_*"))):
                    m = TRIAL_RE.match(os.path.basename(t))
                    if not m:
                        continue
                    if not (os.path.exists(os.path.join(t, "task.csv"))
                            and os.path.exists(os.path.join(t, "joints", "rs_joints.npz"))):
                        continue
                    out.append((s, t, m.group(2), int(m.group(3)), int(m.group(4) or 1)))
    return out


def load_task(path):
    a = np.genfromtxt(path, delimiter=",", names=True)
    m = a["t"] >= 0
    d = {k: np.asarray(a[k], float)[m] for k in
         ("t", "epoch", "target_x_cm", "target_y_cm", "cursor_x_cm", "cursor_y_cm")}
    return d


# ---------------------------------------------------------------- 信号処理
def lowpass(x, fs, fc, order=2):
    nyq = fs / 2.0
    if fc >= nyq * 0.95 or len(x) < 30:
        return np.asarray(x, float)
    b, a = butter(order, fc / nyq)
    return filtfilt(b, a, x, axis=0)


def target_strokes(t, x, y, min_dur=0.25, prom=0.03):
    """ターゲットの速さの谷から谷まで．戻り値: (stroke のリスト, 格子の dt, 速さ, 格子 t, x, y)"""
    dt = 0.005
    g = np.arange(0.0, t[-1], dt)
    gx, gy = np.interp(g, t, x), np.interp(g, t, y)
    s = np.hypot(np.gradient(gx, dt), np.gradient(gy, dt))           # cm/s
    mins, _ = find_peaks(-s, distance=max(1, int(min_dur / dt)), prominence=prom * s.max())
    strokes = []
    for i0, i1 in zip(mins[:-1], mins[1:]):
        A = float(np.hypot(gx[i1] - gx[i0], gy[i1] - gy[i0])) * 10.0      # mm
        path = float(np.sum(np.hypot(np.diff(gx[i0:i1 + 1]), np.diff(gy[i0:i1 + 1])))) * 10.0
        strokes.append(dict(t0=float(g[i0]), t1=float(g[i1]), A_mm=A, path_mm=path,
                            vmax_mm_s=float(s[i0:i1 + 1].max()) * 10.0, dur_s=float(g[i1] - g[i0]),
                            dx_mm=float(gx[i1] - gx[i0]) * 10.0, dy_mm=float(gy[i1] - gy[i0]) * 10.0))
    return strokes


def estimate_lag(d, max_lag=0.8):
    """cursor(t) ≈ target(t - tau) となる tau（遅れ）[s]．"""
    t = d["t"]
    best, tau_best = np.inf, 0.0
    for tau in np.arange(0.0, max_lag + 1e-9, 0.01):
        m = t >= tau
        if m.sum() < 50:
            continue
        tx = np.interp(t[m] - tau, t, d["target_x_cm"])
        ty = np.interp(t[m] - tau, t, d["target_y_cm"])
        e = np.mean((d["cursor_x_cm"][m] - tx) ** 2 + (d["cursor_y_cm"][m] - ty) ** 2)
        if e < best:
            best, tau_best = e, float(tau)
    return tau_best, float(np.sqrt(best))


def prepare_joints(npz, fc, max_gap, noise_split, min_noise=0.0):
    """関節を一様な格子に載せ，低域通過した位置とノイズ分散（軸ごと）を返す．"""
    r = np.load(npz, allow_pickle=False)
    names = [str(n) for n in r["names"]]
    t = np.asarray(r["t"], float)
    P = np.asarray(r["P_C"], float)                      # frames x joints x 3 [mm]
    order = np.argsort(t)
    t, P = t[order], P[order]
    ok = np.isfinite(t)
    t, P = t[ok], P[ok]
    need = ["shoulder", "elbow", "wrist", "mcp3"]
    if any(n not in names for n in need):
        return None
    if "wrist_pose" not in names:
        names.append("wrist_pose")
        P = np.concatenate([P, P[:, [names.index("wrist")]]], axis=1)
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    grid = np.arange(t[0], t[-1], dt)
    pos, var, valid = {}, {}, np.ones(len(grid), bool)
    for n in ["shoulder", "elbow", "wrist_pose", "wrist", "mcp3"]:
        X = P[:, names.index(n), :]
        fin = np.all(np.isfinite(X), axis=1)
        if fin.sum() < 60:
            return None
        tf, Xf = t[fin], X[fin]
        G = np.stack([np.interp(grid, tf, Xf[:, k]) for k in range(3)], axis=1)
        idx = np.clip(np.searchsorted(tf, grid), 1, len(tf) - 1)
        near = np.minimum(np.abs(grid - tf[idx - 1]), np.abs(tf[idx] - grid))
        valid &= near <= max_gap
        pos[n] = lowpass(G, fs, fc)
        # ノイズ: 高い周波数帯（noise_split 以上）の分散を白色として，0..fc の分に直す
        nyq = fs / 2.0
        if nyq - noise_split >= 2.0:
            Gs = lowpass(G, fs, noise_split)
            res = Xf - np.stack([np.interp(tf, grid, Gs[:, k]) for k in range(3)], axis=1)
            var[n] = np.var(res, axis=0) * fc / (nyq - noise_split)
        else:
            var[n] = np.full(3, np.nan)
        var[n] = np.maximum(var[n], min_noise ** 2)
    return dict(grid=grid, fs=fs, pos=pos, var=var, valid=valid,
                lr_swapped=float(np.mean(r["lr_swapped"])) if "lr_swapped" in r.files else float("nan"))


def at(grid, X, tt):
    return np.array([np.interp(tt, grid, X[:, k]) for k in range(3)])


def link_vectors(J, tt):
    p = {n: at(J["grid"], J["pos"][n], tt) for n in J["pos"]}
    return {"S": p["shoulder"], "U": p["elbow"] - p["shoulder"], "F": p["wrist_pose"] - p["elbow"],
            "M": p["wrist"] - p["wrist_pose"], "H": p["mcp3"] - p["wrist"]}


def link_var(J):
    v = J["var"]
    return {"S": v["shoulder"], "U": v["shoulder"] + v["elbow"], "F": v["elbow"] + v["wrist_pose"],
            "M": v["wrist_pose"] + v["wrist"], "H": v["wrist"] + v["mcp3"]}


# ---------------------------------------------------------------- 1 試行
def process_trial(sdir, tdir, band, seed, attempt, args):
    person, dpi, dpi_decl = session_info(sdir, args.trust_json)
    d = load_task(os.path.join(tdir, "task.csv"))
    if len(d["t"]) < 100:
        return None, None
    # cm は「宣言した dpi」で counts から換算されている → 本当の dpi に直す（dpi が低いほど実際の移動量は大きい）
    ov = (getattr(args, "dpi_override", None) or {}).get(os.path.normpath(os.path.abspath(sdir)))
    if ov:
        dpi = float(ov)
    scale = dpi_decl / dpi if (np.isfinite(dpi) and np.isfinite(dpi_decl) and dpi > 0) else 1.0
    if abs(scale - 1.0) > 1e-6:
        for k in ("target_x_cm", "target_y_cm", "cursor_x_cm", "cursor_y_cm"):
            d[k] = d[k] * scale
    J = prepare_joints(os.path.join(tdir, "joints", "rs_joints.npz"), args.fc, args.max_gap, args.noise_split, args.min_noise)
    if J is None:
        return None, None
    epoch0 = float(np.median(d["epoch"] - d["t"]))
    tau, rms_err = estimate_lag(d)
    strokes = target_strokes(d["t"], d["target_x_cm"], d["target_y_cm"])
    lv = link_var(J)
    rows = []
    for k, s in enumerate(strokes):
        w0, w1 = epoch0 + s["t0"] + tau, epoch0 + s["t1"] + tau
        row = dict(person=person, dpi=dpi, band=band, seed=seed, attempt=attempt, trial=os.path.basename(tdir),
                   stroke=k, **{key: round(val, 4) for key, val in s.items()}, lag_s=tau)
        if w0 < J["grid"][0] or w1 > J["grid"][-1]:
            row.update(valid=0, why="outside_joint_record")
            rows.append(row)
            continue
        m = (J["grid"] >= w0) & (J["grid"] <= w1)
        cov = float(np.mean(J["valid"][m])) if m.any() else 0.0
        row["coverage"] = round(cov, 3)
        if cov < args.min_cov:
            row.update(valid=0, why="coverage")
            rows.append(row)
            continue
        a, b = link_vectors(J, w0), link_vectors(J, w1)
        dl = {n: b[n] - a[n] for n in LINKS}
        D = sum(dl.values())
        nD = float(np.linalg.norm(D))
        row["D_mm"] = round(nD, 3)
        row["gain"] = round(nD / s["A_mm"], 3) if s["A_mm"] > 0 else float("nan")
        if nD < 1e-6 or not all(np.all(np.isfinite(lv[n])) for n in LINKS):
            row.update(valid=0, why="no_hand_motion_or_noise")
            rows.append(row)
            continue
        u = D / nD
        for n in LINKS:
            sk = float(dl[n] @ u)
            sd = float(np.sqrt(2.0 * np.sum(u ** 2 * lv[n])))
            row[f"s_{n}"] = round(sk, 3)
            row[f"sd_{n}"] = round(sd, 3)
            row[f"share_{n}"] = round(sk / nD, 4)
            row[f"det_{n}"] = int(abs(sk) > args.snr * sd)
        sdD = float(np.sqrt(2.0 * np.sum(u ** 2 * J["var"]["mcp3"])))
        row["sd_D"] = round(sdD, 3)
        row["hand_detectable"] = int(nD > args.snr * sdD)
        row.update(valid=1, why="")
        rows.append(row)
    tr = dict(person=person, dpi=dpi, band=band, seed=seed, attempt=attempt, trial=os.path.basename(tdir),
              dpi_declared=dpi_decl, cm_scale=round(scale, 3),
              lag_s=tau, track_rms_cm=round(rms_err, 3), fs_joints=round(J["fs"], 2),
              lr_swapped_frac=round(J["lr_swapped"], 3), n_strokes=len(strokes),
              n_valid=sum(r.get("valid", 0) for r in rows),
              **{f"noise_{n}_mm": round(float(np.sqrt(np.nansum(J["var"][n]))), 2) for n in J["var"]})
    return rows, tr


# ---------------------------------------------------------------- 集計
def bin_of(A, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= A < edges[i + 1]:
            return i
    return len(edges) - 2


def bin_label(edges, i):
    lo, hi = edges[i], edges[i + 1]
    return f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+"


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, edges, args):
    groups = {}
    for r in rows:
        key = (r["person"], r["dpi"], r["band"])
        groups.setdefault(key, []).append(r)
    counts = []
    for key, rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
        for i in range(len(edges) - 1):
            sub = [r for r in rs if bin_of(r["A_mm"], edges) == i]
            val = [r for r in sub if r.get("valid") == 1]
            hd = [r for r in val if r.get("hand_detectable") == 1]
            c = dict(person=key[0], dpi=key[1], band=key[2], bin_mm=bin_label(edges, i),
                     n_total=len(sub), n_valid=len(val), n_hand_detectable=len(hd))
            for n in LINKS:
                c[f"n_det_{n}"] = sum(r.get(f"det_{n}", 0) for r in hd)
            c["enough"] = int(len(hd) >= args.min_strokes)
            counts.append(c)
    return counts


def report(rows, trials, counts, edges, args):
    L = []
    L.append("stroke_feasibility report")
    L.append(f"trials: {len(trials)}   strokes: {len(rows)}   valid: {sum(r.get('valid', 0) for r in rows)}")
    why = {}
    for r in rows:
        if not r.get("valid"):
            why[r.get("why", "?")] = why.get(r.get("why", "?"), 0) + 1
    if why:
        L.append("invalid strokes: " + ", ".join(f"{k}={v}" for k, v in sorted(why.items())))
    val = [r for r in rows if r.get("valid") == 1]
    if val:
        g = np.array([r["gain"] for r in val if np.isfinite(r["gain"])])
        L.append(f"hand gain |D|/A (sanity: ~1 if units, lag and windows are right): "
                 f"median {np.median(g):.2f}, IQR {np.percentile(g, 25):.2f}-{np.percentile(g, 75):.2f}")
        A = np.array([r["A_mm"] for r in val])
        v = np.array([r["vmax_mm_s"] for r in val])
        rho = spearmanr(np.log(A), np.log(v))[0]
        L.append(f"Spearman rho(log A, log vmax) = {rho:.2f}  (|rho| near 1 means amplitude and speed cannot be separated)")
        L.append("noise (sd of stroke displacement along the hand direction, median over valid strokes) [mm]: " +
                 ", ".join(f"{n}={np.nanmedian([r[f'sd_{n}'] for r in val]):.1f}" for n in LINKS) +
                 f", hand={np.nanmedian([r['sd_D'] for r in val]):.1f}")
    L.append("")
    L.append(f"bins of target stroke amplitude A [mm]: {[bin_label(edges, i) for i in range(len(edges) - 1)]}")
    L.append(f"'enough' = hand-detectable strokes >= {args.min_strokes} (|D| > {args.snr} x noise)")
    L.append("")
    L.append("person  dpi  band | " + " ".join(f"{bin_label(edges, i):>9}" for i in range(len(edges) - 1)) + "   (n_hand_detectable / n_total)")
    seen = []
    for c in counts:
        k = (c["person"], c["dpi"], c["band"])
        if k not in seen:
            seen.append(k)
    for k in seen:
        cs = [c for c in counts if (c["person"], c["dpi"], c["band"]) == k]
        cells = [f"{c['n_hand_detectable']:>3}/{c['n_total']:<3}{'*' if c['enough'] else ' '}" for c in cs]
        L.append(f"{k[0]:<7} {k[1]:>4g} {k[2]:<4} | " + " ".join(f"{x:>9}" for x in cells))
    L.append("  (* = enough)")
    L.append("")
    L.append("link detectability by amplitude bin (pooled; fraction of hand-detectable strokes whose link motion > snr x noise):")
    for i in range(len(edges) - 1):
        hd = [r for r in val if bin_of(r["A_mm"], edges) == i and r.get("hand_detectable") == 1]
        if hd:
            L.append(f"  {bin_label(edges, i):>8} mm (n={len(hd):>4}): " +
                     " ".join(f"{n}={np.mean([r[f'det_{n}'] for r in hd]):.2f}" for n in LINKS))
    bad = [t for t in trials if np.isfinite(t.get("lr_swapped_frac", np.nan)) and t["lr_swapped_frac"] > 0.05]
    if bad:
        L.append("")
        L.append("trials with >5% left/right swapped frames (check before trusting): " +
                 ", ".join(f"{t['person']}/{t['trial']}" for t in bad))
    return "\n".join(L)


def plot(rows, edges, path, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    val = [r for r in rows if r.get("valid") == 1]
    if not val:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    dpis = sorted({r["dpi"] for r in val})
    allA = np.array([r["A_mm"] for r in val])
    lo, hi = max(allA.min(), 0.5), allA.max() * 1.05
    bins = np.logspace(np.log10(lo), np.log10(hi), 30)
    for dpi in dpis:
        A = [r["A_mm"] for r in val if r["dpi"] == dpi]
        ax[0].hist(A, bins=bins, alpha=0.5, label=f"{dpi:g} dpi (n={len(A)})")
    for n, col in (("U", "k"), ("H", "r")):
        sd = np.nanmedian([r[f"sd_{n}"] for r in val])
        ax[0].axvline(args.snr * sd, color=col, ls="--", lw=1, label=f"{args.snr:g} x noise {LINK_NAME[n]}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("target stroke amplitude A [mm of mouse travel]")
    ax[0].set_ylabel("strokes")
    ax[0].set_title("(a) amplitude distribution vs noise")
    ax[0].legend(fontsize=7)
    xs = np.arange(len(edges) - 1)
    for n in LINKS:
        fr = []
        for i in xs:
            hd = [r for r in val if bin_of(r["A_mm"], edges) == i and r.get("hand_detectable") == 1]
            fr.append(np.mean([r[f"det_{n}"] for r in hd]) if hd else np.nan)
        ax[1].plot(xs, fr, "o-", label=LINK_NAME[n])
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([bin_label(edges, i) for i in xs])
    ax[1].set_xlabel("A bin [mm]")
    ax[1].set_ylabel(f"fraction with |link motion| > {args.snr:g} x noise")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("(b) which links are measurable")
    ax[1].legend(fontsize=7)
    v = np.array([r["vmax_mm_s"] for r in val])
    ax[2].loglog(allA, v, ".", ms=3, alpha=0.5)
    rho = spearmanr(np.log(allA), np.log(v))[0]
    ax[2].set_xlabel("A [mm]")
    ax[2].set_ylabel("peak target speed [mm/s]")
    ax[2].set_title(f"(c) amplitude vs speed (Spearman rho = {rho:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="ターゲットのストロークの大きさで関節の使い分けを調べられるかの確認")
    ap.add_argument("path", nargs="*", default=["recordings"])
    ap.add_argument("--out", default=os.path.join(HERE, "out"), help="出力先（既定: このファイルと同じ temp/ の out/）")
    ap.add_argument("--fc", type=float, default=3.0, help="関節位置の低域通過 [Hz]")
    ap.add_argument("--noise-split", type=float, default=5.0, help="これ以上の周波数をノイズとみなす [Hz]")
    ap.add_argument("--bins", type=float, nargs="+", default=[5, 10, 20, 40, 80], help="A のビンの境界 [mm]")
    ap.add_argument("--min-noise", type=float, default=2.0,
                    help="1 関節・1 軸の位置ノイズ(sd)の下限 [mm]．高周波から見積もったノイズは MediaPipe の平滑化で小さく出るため")
    ap.add_argument("--trust-json", action="store_true", help="dpi はフォルダ名でなく session.json を信じる")
    ap.add_argument("--min-strokes", type=int, default=15)
    ap.add_argument("--snr", type=float, default=2.0)
    ap.add_argument("--min-cov", type=float, default=0.8, help="窓内で関節が取れている割合の下限")
    ap.add_argument("--max-gap", type=float, default=0.5, help="これより長い欠損は埋めない [s]")
    args = ap.parse_args()
    edges = [0.0] + sorted(args.bins) + [np.inf]

    paths = [p if (os.path.exists(p) or p != "recordings") else os.path.join(os.path.dirname(HERE), "recordings")
             for p in args.path]
    found = discover(paths)
    if not found:
        print("no trials found (need recordings/aim_*/trial_NN_<slow|fast>_sS/{task.csv, joints/rs_joints.npz})")
        return
    rows, trials = [], []
    for sdir, tdir, band, seed, attempt in found:
        r, t = process_trial(sdir, tdir, band, seed, attempt, args)
        if r is None:
            print(f"skip {os.path.basename(sdir)}/{os.path.basename(tdir)}")
            continue
        rows += r
        trials.append(t)
        print(f"{os.path.basename(sdir)}/{os.path.basename(tdir)}: strokes {t['n_strokes']}, valid {t['n_valid']}, lag {t['lag_s']:.2f} s")
    os.makedirs(args.out, exist_ok=True)
    counts = summarize(rows, edges, args)
    write_csv(os.path.join(args.out, "strokes.csv"), rows)
    write_csv(os.path.join(args.out, "trials.csv"), trials)
    write_csv(os.path.join(args.out, "counts.csv"), counts)
    txt = report(rows, trials, counts, edges, args)
    with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    plot(rows, edges, os.path.join(args.out, "feasibility.png"), args)
    print("\n" + txt)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()