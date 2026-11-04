#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
freq_phase.py — ターゲットの周波数ごとに，リンクの寄与（振幅）とタイミング（位相）の個人差を調べる

  python temp/freq_phase.py --list-config                 # (compare_config.py を使う．stroke_compare.py と同じ設定)
  python temp/freq_phase.py                               # COMPARES を全部
  python temp/freq_phase.py --compare slow_all            # 1 つだけ
  オプション: --perm 2000  --snr 5  --correct-model  --out temp/out  --config temp/compare_config.py

考え方
  ターゲットは 既知の周波数 f_k = n_k / 40 s の正弦波の和．どの 40 s の窓にも各成分がちょうど整数周期入るので，
  各信号を「f_k の cos / sin の最小二乗フィット」にすると，成分ごとの複素振幅 Z_k = A_k e^{i phi_k} が互いに混ざらずに取れる．
  （ストロークの切り出しは要らない．40 s 分が積分されるので，ノイズが 3 mm でも成分の振幅の誤差は約 0.1 mm，
   位相の誤差は 1 フレーム (17-33 ms) より細かい．）
  同じ帯域の周波数は seed によらず共通（位相だけ seed で変わる）→ 1 人 3 試行は「同じ周波数の繰り返し」になる．

  1. 各関節の 3 次元位置を，手先 (mcp3) の動く向き（x 成分用・y 成分用に別々）に射影してフィットする．
  2. リンクの複素振幅は関節の差: S = 肩，U = 肘 - 肩，F = 手首(pose) - 肘，M = 手首(hand) - 手首(pose)，H = MCP3 - 手首(hand)．
     足すと手先 (mcp3) の複素振幅になる．
  3. c_link = Z_link / Z_hand（成分 k ごと．足すと 1）．
        share_re = Re(c)   … 同相の寄与（これが「この周波数の動きを，そのリンクがどれだけ作っているか」）
        lead_ms  = 1000 * arg(c) / (2 pi f)  … リンクが手先より何 ms 先に動くか（+ = 先行，- = 遅れ）．
                   リンクの振幅がノイズの SNR 倍を超えた成分だけ出す．
  4. マウス（cursor）とターゲットの比 = ゲインと遅れ（伝達関数）．手（mcp3）とマウスの比 = カメラ側の遅れ（システムの遅れ）．
  5. 個人差: 人 × 試行（seed）を繰り返しとみなして，周波数 × リンク × 指標ごとに 1 元配置の分散分析と ICC(1)．
        F の p 値は F 分布と並べ替え（--perm 回）の両方．検定の数が多いので BH 法の q 値も付ける（指標ごと）．

注意
  - 肩・肘・手首(pose) は pose モデル，手首(hand)・MCP3 は hand モデル由来．モデルごとに処理の遅れが違うと，
    M・H のタイミングに全員共通のずれが入る．手首(hand) と 手首(pose) は同じ場所なので，この 2 つの遅れの差が
    「モデルの遅れの差」の目安になる（model_offset_ms）．--correct-model で hand モデル側をその分だけ戻す．
  - 共通のずれは人の違いには影響しない（人どうしの差を見る限り）．絶対的な「何 ms 先」は割り引いて読むこと．
  - ICC は 9 人 × 3 試行ほどの推定．多数の指標を同時に見るので，q 値と個数（期待される偶然の個数）を一緒に見る．

出力: <out>/<比較の名前>/freq/
  trial_freq.csv   試行 × 周波数 × リンクの値（長い表）
  person_freq.csv  人 × 周波数 × リンクの平均（試行をまとめたもの）
  icc.csv          周波数 × リンク × 指標ごとの ICC・F・p・q
  trials.csv       試行ごとの診断（モデルの遅れ，システムの遅れ，測れた成分数）
  report.txt  freq_share.png  freq_lead.png  transfer.png  icc_heatmap.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from scipy.stats import f as fdist
from scipy.stats import ttest_1samp

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stroke_compare as SC  # noqa: E402
import stroke_feasibility as SF  # noqa: E402

JOINTS = ["shoulder", "elbow", "wrist_pose", "wrist", "mcp3"]
LINKS = ["S", "U", "F", "M", "H"]


# ---------------------------------------------------------------- フィット
def get_freqs(tdir, band, seed):
    """meta.json の軌道から周波数を読む．なければ tasks/targets.py から作る"""
    meta = SF.read_json(os.path.join(tdir, "meta.json"))
    tr = (meta.get("task") or {}).get("trajectory")
    if not (tr and "axes" in tr):
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tasks"))
        from targets import make_trajectory
        tr = make_trajectory(band, seed)
    return {ax: [float(f) for f in tr["axes"][ax]["f_hz"]] for ax in ("x", "y")}, float(tr.get("T", 40.0))


def fit_components(tt, Y, freqs, T):
    """Y (N x C) を f_k の cos / sin + 定数 + 直線で最小二乗．複素振幅 Z (K x C) = a - i b（A cos(wt + phi) = a cos + b sin）と，残差の SD (C)"""
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    cols = [np.ones_like(tt), (tt - T / 2) / T]
    for f in freqs:
        w = 2 * np.pi * f
        cols += [np.cos(w * tt), np.sin(w * tt)]
    X = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ coef
    dof = max(len(tt) - X.shape[1], 1)
    return coef[2::2] - 1j * coef[3::2], np.sqrt((res ** 2).sum(axis=0) / dof)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- 1 試行
def process_trial(sdir, tdir, band, seed, attempt, person, args):
    _, dpi, dpi_decl = SF.session_info(sdir, args.trust_json)
    ov = (args.dpi_override or {}).get(os.path.normpath(os.path.abspath(sdir)))
    if ov:
        dpi = float(ov)
    scale = dpi_decl / dpi if (np.isfinite(dpi) and np.isfinite(dpi_decl) and dpi > 0) else 1.0
    d = SF.load_task(os.path.join(tdir, "task.csv"))
    if len(d["t"]) < 100:
        return None
    fr, T = get_freqs(tdir, band, seed)
    m = d["t"] < T
    t = d["t"][m]
    cur = {"x": d["cursor_x_cm"][m] * scale * 10.0, "y": d["cursor_y_cm"][m] * scale * 10.0}      # mm
    tgt = {"x": d["target_x_cm"][m] * scale * 10.0, "y": d["target_y_cm"][m] * scale * 10.0}
    freqs = fr["x"] + fr["y"]
    axis_of = ["x"] * len(fr["x"]) + ["y"] * len(fr["y"])
    K = len(freqs)
    idx = {ax: [k for k in range(K) if axis_of[k] == ax] for ax in ("x", "y")}
    om = 2 * np.pi * np.array(freqs)

    # マウスとターゲット
    Zt, Zc = np.zeros(K, complex), np.zeros(K, complex)
    for ax in ("x", "y"):
        zt, _ = fit_components(t, tgt[ax], freqs, T)
        zc, _ = fit_components(t, cur[ax], freqs, T)
        for k in idx[ax]:
            Zt[k], Zc[k] = zt[k, 0], zc[k, 0]

    # 関節
    J = SF.prepare_joints(os.path.join(tdir, "joints", "rs_joints.npz"), args.fc, args.max_gap, args.noise_split, args.min_noise)
    if J is None:
        return None
    epoch0 = float(np.median(d["epoch"] - d["t"]))
    tt = J["grid"] - epoch0
    w = (tt >= 0) & (tt < T) & J["valid"]
    if w.sum() < 0.7 * T * J["fs"]:
        return None
    N = int(w.sum())
    Z3, sd3 = {}, {}
    for j in JOINTS:
        Z3[j], sd3[j] = fit_components(tt[w], J["pos"][j][w], freqs, T)           # (K, 3), (3,)
    # 手先の動く向き（x 成分用・y 成分用）
    u = {}
    for ax in ("x", "y"):
        Zm = Z3["mcp3"][idx[ax]]
        M = np.concatenate([Zm.real, Zm.imag], axis=0)
        u[ax] = np.linalg.svd(M, full_matrices=False)[2][0]
    Zp, se = {}, {}
    for j in JOINTS:
        Zp[j], se[j] = np.zeros(K, complex), np.zeros(K)
        for k in range(K):
            uu = u[axis_of[k]]
            Zp[j][k] = Z3[j][k] @ uu
            se[j][k] = np.sqrt(np.sum(uu ** 2 * sd3[j] ** 2)) * np.sqrt(2.0 / N)
    for ax in ("x", "y"):                                                           # 符号をマウスに合わせる
        r = np.sum(Zp["mcp3"][idx[ax]] * np.conj(Zc[idx[ax]]))
        if r.real < 0:
            for j in JOINTS:
                Zp[j][idx[ax]] *= -1
    # モデルの遅れの差（手首 hand - 手首 pose）
    ok_w = np.abs(Zp["wrist_pose"]) > args.snr * se["wrist_pose"]
    if ok_w.any():
        lead_w = np.angle(Zp["wrist"] / Zp["wrist_pose"]) / om
        wt = np.abs(Zp["wrist_pose"]) * ok_w
        offset = float(np.sum(lead_w * wt) / np.sum(wt))                             # s（+ = hand モデルが先）
    else:
        offset = 0.0
    if args.correct_model:
        for j in ("wrist", "mcp3"):
            Zp[j] = Zp[j] * np.exp(-1j * om * offset)
    Zl = {"S": Zp["shoulder"], "U": Zp["elbow"] - Zp["shoulder"], "F": Zp["wrist_pose"] - Zp["elbow"],
          "M": Zp["wrist"] - Zp["wrist_pose"], "H": Zp["mcp3"] - Zp["wrist"]}
    sel = {"S": se["shoulder"], "U": np.hypot(se["elbow"], se["shoulder"]), "F": np.hypot(se["wrist_pose"], se["elbow"]),
           "M": np.hypot(se["wrist"], se["wrist_pose"]), "H": np.hypot(se["mcp3"], se["wrist"])}
    Zh = Zp["mcp3"]
    ok_h = np.abs(Zh) > args.snr * se["mcp3"]

    rows = []
    base = dict(person=person, band=band, seed=seed, attempt=attempt, trial=os.path.basename(tdir))

    def add(k, link, **kw):
        r = dict(base, f_hz=round(freqs[k], 4), axis=axis_of[k], n=int(round(freqs[k] * T)), link=link,
                 A_target_mm=abs(Zt[k]), A_hand_mm=abs(Zh[k]), ok_hand=int(ok_h[k]))
        r.update(kw)
        rows.append(r)

    nan = float("nan")
    for k in range(K):
        H = Zc[k] / Zt[k]
        add(k, "cursor", gain=abs(H), lag_ms=-1000 * np.angle(H) / om[k], share_re=nan, lead_ms=nan, ok_link=1)
        Hh = Zh[k] / Zc[k]
        add(k, "hand", gain=abs(Hh), lag_ms=-1000 * np.angle(Hh) / om[k], share_re=nan, lead_ms=nan, ok_link=int(ok_h[k]))
        for l in LINKS:
            c = Zl[l][k] / Zh[k]
            ok_l = bool(ok_h[k] and abs(Zl[l][k]) > args.snr * sel[l][k])
            add(k, l, share_re=c.real if ok_h[k] else nan, share_im=c.imag if ok_h[k] else nan,
                lead_ms=1000 * np.angle(c) / om[k] if ok_l else nan, gain=nan, lag_ms=nan, ok_link=int(ok_l),
                A_link_mm=abs(Zl[l][k]))
    hc = np.angle(Zh / Zc) / om
    wt = np.abs(Zh) * ok_h
    tr = dict(base, dpi=dpi, cm_scale=round(scale, 3), N=N, fs=round(J["fs"], 1),
              model_offset_ms=1000 * offset, hand_vs_cursor_lag_ms=(-1000 * float(np.sum(hc * wt) / np.sum(wt)) if wt.sum() > 0 else nan),
              n_ok_hand=int(ok_h.sum()), n_components=K,
              sum_links_check=float(np.max(np.abs(sum(Zl[l] for l in LINKS) - Zh))))
    return rows, tr


# ---------------------------------------------------------------- 統計
def anova1(vals, lab, a):
    N = len(vals)
    n = np.bincount(lab, minlength=a).astype(float)
    s = np.bincount(lab, weights=vals, minlength=a)
    ssb = (s ** 2 / n).sum() - vals.sum() ** 2 / N
    ssw = (vals ** 2).sum() - (s ** 2 / n).sum()
    msb, msw = ssb / (a - 1), ssw / (N - a)
    return msb, msw, n


def icc_test(groups, rng, nperm):
    """groups: {人: [値...]}．ICC(1)，F，p（F 分布と並べ替え）"""
    groups = {p: np.array([x for x in v if np.isfinite(x)]) for p, v in groups.items()}
    groups = {p: v for p, v in groups.items() if len(v) >= 1}
    if not groups:
        return None
    a = len(groups)
    vals = np.concatenate(list(groups.values()))
    N = len(vals)
    if a < 4 or N - a < 3 or sum(len(v) >= 2 for v in groups.values()) < 3:
        return None
    lab = np.concatenate([[i] * len(v) for i, v in enumerate(groups.values())]).astype(int)
    msb, msw, n = anova1(vals, lab, a)
    k0 = (N - (n ** 2).sum() / N) / (a - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw) if (msb + (k0 - 1) * msw) > 0 else float("nan")
    F = msb / msw if msw > 0 else float("inf")
    p_par = float(fdist.sf(F, a - 1, N - a)) if np.isfinite(F) else 0.0
    cnt = 0
    for _ in range(nperm):
        mp, wp, _ = anova1(vals, rng.permutation(lab), a)
        cnt += (mp / wp if wp > 0 else np.inf) >= F - 1e-12
    return dict(n_persons=a, N=N, k0=k0, ICC=icc, F=F, p_param=p_par, p_perm=(cnt + 1) / (nperm + 1),
                between_sd=float(np.sqrt(max((msb - msw) / k0, 0))), within_sd=float(np.sqrt(msw)))


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    prev = 1.0
    for rank, i in zip(range(len(p), 0, -1), o[::-1]):
        prev = min(prev, p[i] * len(p) / rank)
        q[i] = prev
    return q


# ---------------------------------------------------------------- 出力
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


MEASURES = [("share_re", LINKS), ("lead_ms", LINKS), ("gain", ["cursor", "hand"]), ("lag_ms", ["cursor", "hand"])]


def plots(out, PF, persons, bands, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("tab10")
    col = {p: cm(i % 10) for i, p in enumerate(persons)}
    for fname, meas, links, ylab, scale in (("freq_share.png", "share_re", LINKS, "in-phase share of hand motion [%]", 100.0),
                                            ("freq_lead.png", "lead_ms", LINKS, "lead relative to hand [ms] (+ = link moves first)", 1.0)):
        fig, axs = plt.subplots(2, len(links), figsize=(3.1 * len(links), 6.2), sharex="row", squeeze=False)
        for ri, ax_name in enumerate(("x", "y")):
            for ci, l in enumerate(links):
                ax = axs[ri][ci]
                allv = {}
                for p in persons:
                    pts = sorted((f, v) for (pp, b, f, a_, ll), v in PF.items() if pp == p and a_ == ax_name and ll == l and np.isfinite(v.get(meas, np.nan)))
                    if pts:
                        ax.plot([x[0] for x in pts], [scale * x[1][meas] for x in pts], "-o", ms=3, lw=0.8, color=col[p], alpha=0.8, label=p)
                    for f, v in pts:
                        allv.setdefault(f, []).append(scale * v[meas])
                if allv:
                    fs_ = sorted(allv)
                    ax.plot(fs_, [np.mean(allv[f]) for f in fs_], "k-", lw=2)
                ax.axhline(0, color="#bbb", lw=0.6)
                ax.set_xscale("log")
                ax.set_title(f"{l} ({'horizontal' if ax_name == 'x' else 'vertical'} components)", fontsize=9)
                if ci == 0:
                    ax.set_ylabel(ylab, fontsize=8)
                ax.set_xlabel("target frequency [Hz]", fontsize=8)
        axs[0][-1].legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=120)
        plt.close(fig)
    fig, axs = plt.subplots(2, 4, figsize=(15, 6), squeeze=False)
    for ri, meas in enumerate(("gain", "lag_ms")):
        for ci, (link, ax_name) in enumerate((("cursor", "x"), ("cursor", "y"), ("hand", "x"), ("hand", "y"))):
            ax = axs[ri][ci]
            allv = {}
            for p in persons:
                pts = sorted((f, v[meas]) for (pp, b, f, a_, ll), v in PF.items() if pp == p and a_ == ax_name and ll == link and np.isfinite(v.get(meas, np.nan)))
                if pts:
                    ax.plot([x[0] for x in pts], [x[1] for x in pts], "-o", ms=3, lw=0.8, color=col[p], alpha=0.8)
                for f, v in pts:
                    allv.setdefault(f, []).append(v)
            if allv:
                fs_ = sorted(allv)
                ax.plot(fs_, [np.mean(allv[f]) for f in fs_], "k-", lw=2)
            ax.set_xscale("log")
            ax.set_title(f"{'mouse vs target' if link == 'cursor' else 'hand (mcp3) vs mouse'} / {ax_name}", fontsize=9)
            ax.set_ylabel("gain" if meas == "gain" else "lag [ms] (+ = later)", fontsize=8)
            ax.set_xlabel("frequency [Hz]", fontsize=8)
            if meas == "gain":
                ax.axhline(1, color="#bbb", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "transfer.png"), dpi=120)
    plt.close(fig)


def plot_icc(out, icc_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bands = sorted({r["band"] for r in icc_rows})
    fig, axs = plt.subplots(2, 2 * len(bands), figsize=(5 * 2 * len(bands), 6), squeeze=False)
    for ri, meas in enumerate(("share_re", "lead_ms")):
        for bi, band in enumerate(bands):
            for ai, ax_name in enumerate(("x", "y")):
                ax = axs[ri][bi * 2 + ai]
                rr = [r for r in icc_rows if r["measure"] == meas and r["band"] == band and r["axis"] == ax_name]
                fs_ = sorted({r["f_hz"] for r in rr})
                lk = [l for l in LINKS if any(r["link"] == l for r in rr)]
                M = np.full((len(lk), len(fs_)), np.nan)
                S = [[""] * len(fs_) for _ in lk]
                for r in rr:
                    i, j = lk.index(r["link"]), fs_.index(r["f_hz"])
                    M[i, j] = r["ICC"]
                    S[i][j] = "*" if r["q"] < 0.05 else ""
                im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
                for i in range(len(lk)):
                    for j in range(len(fs_)):
                        if np.isfinite(M[i, j]):
                            ax.text(j, i, f"{M[i, j]:.2f}{S[i][j]}", ha="center", va="center", fontsize=7, color="w" if M[i, j] < 0.6 else "k")
                ax.set_xticks(range(len(fs_)))
                ax.set_xticklabels([f"{f:g}" for f in fs_], fontsize=7)
                ax.set_yticks(range(len(lk)))
                ax.set_yticklabels(lk, fontsize=8)
                ax.set_title(f"{meas} / {band} / {ax_name}  (ICC, * q<0.05)", fontsize=8)
                ax.set_xlabel("Hz", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "icc_heatmap.png"), dpi=120)
    plt.close(fig)


def run_compare(cmp, cfg, sessions, found_all, cache, args):
    name = cmp["name"]
    policy = cmp.get("attempt", cfg["ATTEMPT"])
    tr_sel = cmp.get("trials", "all")
    tr_sel = None if tr_sel == "all" else ([tr_sel] if isinstance(tr_sel, str) else list(tr_sel))
    sess = {s["dir"]: s for s in sessions if SC.session_match(s, cmp.get("sessions", "all"))}
    found = SC.pick_attempts([f for f in found_all if os.path.normpath(os.path.abspath(f[0])) in sess], policy)
    found = [f for f in found if tr_sel is None or (SC.trial_tokens(f[2], f[3], f[4], f[1]) & set(tr_sel))]
    if not found:
        print(f"[{name}] 対象の試行がありません")
        return
    rows, trials = [], []
    for sd, td, band, seed, att in found:
        key = os.path.normpath(os.path.abspath(td))
        s = sess[os.path.normpath(os.path.abspath(sd))]
        if key not in cache:
            try:
                res = process_trial(sd, td, band, seed, att, s["person"], args)
            except Exception as e:      # 1 試行の失敗で全体を止めない
                res = None
                print(f"  {os.path.basename(sd)}/{os.path.basename(td)}: error {type(e).__name__}: {e}")
            cache[key] = res
            print(f"  {os.path.basename(sd)}/{os.path.basename(td)}: " + ("skip" if res is None else f"ok components {res[1]['n_ok_hand']}/{res[1]['n_components']}, model offset {res[1]['model_offset_ms']:.0f} ms"))
        if cache[key] is None:
            continue
        r, t = cache[key]
        for x in r:
            x = dict(x)
            x.update(person=s["person"], group=s["group"])
            rows.append(x)
        trials.append(dict(t, person=s["person"], group=s["group"]))
    if not rows:
        print(f"[{name}] 使える試行がありません")
        return
    out = os.path.join(args.out, name, "freq")
    os.makedirs(out, exist_ok=True)
    persons = sorted({r["person"] for r in rows})
    bands = sorted({r["band"] for r in rows})
    write_csv(os.path.join(out, "trial_freq.csv"), rows)
    write_csv(os.path.join(out, "trials.csv"), trials)

    # 人 × 周波数 の平均
    PF = {}
    acc = {}
    for r in rows:
        acc.setdefault((r["person"], r["band"], r["f_hz"], r["axis"], r["link"]), []).append(r)
    pf_rows = []
    for (p, b, f, ax, l), rs in acc.items():
        v = {m: (float(np.nanmean([x[m] for x in rs])) if np.any(np.isfinite([x[m] for x in rs])) else float("nan")) for m in ("share_re", "lead_ms", "gain", "lag_ms")}
        v["n"] = len(rs)
        PF[(p, b, f, ax, l)] = v
        pf_rows.append(dict(person=p, band=b, f_hz=f, axis=ax, link=l, n_trials=len(rs), **v))
    write_csv(os.path.join(out, "person_freq.csv"), sorted(pf_rows, key=lambda r: (r["person"], r["band"], r["axis"], r["f_hz"], r["link"])))

    # ICC
    rng = np.random.default_rng(args.seed)
    icc_rows = []
    keys = sorted({(r["band"], r["f_hz"], r["axis"], r["link"]) for r in rows})
    for meas, links in MEASURES:
        fam = []
        for b, f, ax, l in keys:
            if l not in links:
                continue
            g = {}
            for r in rows:
                if r["band"] == b and r["f_hz"] == f and r["axis"] == ax and r["link"] == l and np.isfinite(r.get(meas, np.nan)):
                    g.setdefault(r["person"], []).append(r[meas])
            res = icc_test(g, rng, args.perm)
            if res:
                fam.append(dict(band=b, f_hz=f, axis=ax, link=l, measure=meas, **res))
        if fam:
            q = bh([x["p_perm"] for x in fam])
            for x, qq in zip(fam, q):
                x["q"] = float(qq)
            icc_rows += fam
    write_csv(os.path.join(out, "icc.csv"), icc_rows)

    # レポート
    L = [f"[{name}] freq_phase: persons {persons}", f"bands {bands}; trials {len(trials)}; snr {args.snr:g}; model-offset correction {'on' if args.correct_model else 'off'}", ""]
    t0 = [r for r in rows if r["link"] == "cursor"]
    L.append("=== components (target amplitude along mouse axis [mm], median over trials; hand SNR) ===")
    for b in bands:
        for ax in ("x", "y"):
            fs_ = sorted({r["f_hz"] for r in t0 if r["band"] == b and r["axis"] == ax})
            L.append(f"{b:<5} {ax}: " + "  ".join(f"{f:g} Hz: {np.median([r['A_target_mm'] for r in t0 if r['band'] == b and r['axis'] == ax and r['f_hz'] == f]):.0f} mm "
                                                  f"(hand ok {100 * np.mean([r['ok_hand'] for r in t0 if r['band'] == b and r['axis'] == ax and r['f_hz'] == f]):.0f}%)" for f in fs_))
    L.append("")
    L.append("=== system checks per person (mean over trials) ===")
    L.append("person   model_offset_ms (hand-model minus pose-model wrist; + = hand model earlier)   hand_vs_cursor_lag_ms (+ = camera later)")
    for p in persons:
        ts = [t for t in trials if t["person"] == p]
        L.append(f"{p:<8} {np.mean([t['model_offset_ms'] for t in ts]):>8.0f}   {np.nanmean([t['hand_vs_cursor_lag_ms'] for t in ts]):>8.0f}")
    L.append(f"max |sum of links - hand| over all trials: {max(t['sum_links_check'] for t in trials):.2e} (should be ~0)")
    L.append("")

    def table(title, meas, links, scale, fmt, valid_only=False):
        L.append(title)
        for b in bands:
            for ax in ("x", "y"):
                fs_ = sorted({k[2] for k in PF if k[1] == b and k[3] == ax})
                if not fs_:
                    continue
                L.append(f"-- {b} / {'horizontal (x)' if ax == 'x' else 'vertical (y)'}   Hz: " + " ".join(f"{f:>7g}" for f in fs_))
                for l in links:
                    cells = []
                    for f in fs_:
                        v = [PF[(p, b, f, ax, l)][meas] for p in persons if (p, b, f, ax, l) in PF and np.isfinite(PF[(p, b, f, ax, l)][meas])]
                        cells.append(f"{scale * np.mean(v):{fmt}}" if len(v) >= 2 else "      -")
                    sds = []
                    for f in fs_:
                        v = [PF[(p, b, f, ax, l)][meas] for p in persons if (p, b, f, ax, l) in PF and np.isfinite(PF[(p, b, f, ax, l)][meas])]
                        sds.append(f"{scale * np.std(v, ddof=1):{fmt}}" if len(v) >= 3 else "      -")
                    L.append(f"   {l:<7} mean " + " ".join(f"{c:>7}" for c in cells))
                    L.append(f"   {'':<7} SD   " + " ".join(f"{c:>7}" for c in sds) + "   (SD across persons)")
        L.append("")

    table("=== in-phase share of the hand motion per frequency [%] (mean across persons) ===", "share_re", LINKS, 100.0, ".0f")
    table("=== lead of the link relative to the hand [ms] (+ = link first; only where the link amplitude > snr x noise) ===", "lead_ms", LINKS, 1.0, ".0f")
    table("=== mouse vs target: gain and lag [ms] (+ = mouse later) ===", "gain", ["cursor"], 1.0, ".2f")
    table("", "lag_ms", ["cursor"], 1.0, ".0f")

    # 同時性: 手先に対するリンクの先行が 0 か（人を単位に）
    L.append("=== simultaneity: is the link lead different from 0 ms? (one-sample t-test across persons; mean lead per person) ===")
    L.append("band  axis  Hz      link | persons  mean lead [ms]  SD    persons leading(+)  p")
    for b in bands:
        for ax in ("x", "y"):
            for f in sorted({k[2] for k in PF if k[1] == b and k[3] == ax}):
                for l in ("S", "U", "F", "H"):
                    v = [PF[(p, b, f, ax, l)]["lead_ms"] for p in persons if (p, b, f, ax, l) in PF and np.isfinite(PF[(p, b, f, ax, l)]["lead_ms"])]
                    if len(v) >= 4:
                        L.append(f"{b:<5} {ax}    {f:<6g}  {l:<4} | {len(v):>4}     {np.mean(v):>8.0f}    {np.std(v, ddof=1):>5.0f}   {sum(x > 0 for x in v):>4}/{len(v):<4}          {ttest_1samp(v, 0).pvalue:.3f}")
    L.append("")

    # ICC のまとめ
    L.append("=== person differences: ICC(1) over trials (persons x trials as repeats of the same frequency) ===")
    L.append("measure   tests  ICC median  p_perm<0.05 (expected by chance)  q<0.05")
    for meas, _ in MEASURES:
        rr = [r for r in icc_rows if r["measure"] == meas]
        if rr:
            L.append(f"{meas:<9} {len(rr):>5}  {np.nanmedian([r['ICC'] for r in rr]):>8.2f}   {sum(r['p_perm'] < 0.05 for r in rr):>4} ({0.05 * len(rr):.1f})                 {sum(r['q'] < 0.05 for r in rr):>4}")
    L.append("")
    for meas in ("share_re", "lead_ms", "gain", "lag_ms"):
        rr = sorted([r for r in icc_rows if r["measure"] == meas], key=lambda r: -r["ICC"])
        if rr:
            L.append(f"highest ICC ({meas}): " + ", ".join(f"{r['band']} {r['axis']} {r['f_hz']:g}Hz {r['link']} {r['ICC']:.2f}(q={r['q']:.3f})" for r in rr[:5]))
    txt = "\n".join(L)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    plots(out, PF, persons, bands, args)
    if icc_rows:
        plot_icc(out, icc_rows)
    print("\n" + txt)
    print(f"\n-> {out}")


def main():
    ap = argparse.ArgumentParser(description="周波数ごとのリンクの寄与とタイミングの個人差")
    ap.add_argument("--config", default=os.path.join(HERE, "compare_config.py"))
    ap.add_argument("--compare", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--snr", type=float, default=5.0, help="成分の振幅がノイズの何倍を超えたら測れたとするか")
    ap.add_argument("--correct-model", action="store_true", help="hand モデル側の遅れを，手首 pose との差の分だけ戻す")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-split", type=float, default=5.0)
    ap.add_argument("--max-gap", type=float, default=0.5)
    ap.add_argument("--trust-json", action="store_true")
    args = ap.parse_args()
    cfg = SC.load_config(args.config)
    if args.out is None:
        args.out = cfg["OUT"] if os.path.isabs(cfg["OUT"]) else os.path.join(HERE, cfg["OUT"])
    args.fc, args.min_noise = cfg["FC"], cfg["MIN_NOISE"]
    sessions = SC.resolve_sessions(cfg)
    if not sessions:
        print("SESSIONS が空，またはフォルダが見つかりません（compare_config.py を編集）")
        return
    args.dpi_override = {s["dir"]: s["dpi"] for s in sessions if s["dpi"]}
    found_all = []
    for s in sessions:
        found_all += SF.discover([s["dir"]])
    cache = {}
    for c in [c for c in cfg["COMPARES"] if not args.compare or c["name"] in args.compare]:
        print(f"\n##### {c['name']}")
        run_compare(c, cfg, sessions, found_all, cache, args)


if __name__ == "__main__":
    main()