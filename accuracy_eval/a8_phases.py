"""a8 フェーズごとの解析（合図音の区間: still / count / bpmNN / rest / board …）

何をするか
  試行を合図音のフェーズ（phases.py: meta.json の cue，なければ予定表を MoCap の動きに合わせて復元）に分け，
  フェーズごとに a1〜a5 の指標と，そのフェーズに固有の指標を出す。

  [全フェーズ]   使える率，左右の取り違え率，その他の誤検出率（a6/a7）
  [静止フェーズ] still・rest#k・count#k
                 ジッタ σ（RS − MoCap の SD，生データ，成分別），フレーム間ノイズ（a1）
                 偏り（成分別）と，そのときの腕の位置 → 静止フェーズ間で偏りがどれだけ変わるか（a3）
                 肘の角度の偏り（a5）
  [動作フェーズ] bpmNN
                 正確度: 偏り，偏りを除いた SD，RMSE（a2）
                 still で測った偏りで補正した RMSE（a3: 補正値の転用）
                 動き: 遅れ，振幅比，速さの r，最大の速さの比，形の誤差/範囲（a4）
                 肘の角度の偏り・SD・r・変化幅の比，骨長の SD（a5）
                 実際のテンポ・振幅・最大の速さ（被験者が指示どおり動いたか）
                 ストロークごと: 振幅比，折り返しの時刻のずれ（RS − MoCap），最大の速さの比，
                                 ストロークの時間 / 指示（60/bpm）
                 切り替え: 動き始め・止まりの時刻のずれ（RS − MoCap）
わかること
  静止・各テンポ・切り替えのどこで，どの種類の誤差が出るか。
  特に「折り返しの時刻」「動き始め・止まり」のずれは，関節の協調（タイミング）を解析するときの誤差に直結する。
"""
import csv
import numpy as np
from scipy.signal import find_peaks
import common as C
import phases as PH
from a4_motion import _shift

MOTION_SD_MIN = 20.0     # これ未満しか動かない関節（肩）は動きの指標を出さない
EXT_WIN = 0.20           # 折り返しの時刻を探す範囲 ±[s]
ONSET_MM = 10.0          # 動き始め・止まり: 静止位置からこれ以上離れたら「動いている」[mm]
ONSET_K = 5.0            #   ただし RS の静止時の揺れ（SD）の 5 倍の方が大きければそちら（RS と MoCap に同じ閾値）
HOLD_S = 0.3


def _angle(S, E, W):
    a = S - E; b = W - E
    c = np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _parabolic(y, i):
    if 0 < i < len(y) - 1:
        d = y[i - 1] - 2 * y[i] + y[i + 1]
        return i + (0.5 * (y[i - 1] - y[i + 1]) / d if d != 0 else 0.0)
    return float(i)


def _first_hold_below(t, v, thr, t_from, hold=HOLD_S):
    """t_from 以降で v < thr が hold 秒続く最初の時刻"""
    i0 = np.searchsorted(t, t_from); n = max(1, int(hold / np.median(np.diff(t))))
    below = v < thr
    for i in range(i0, len(t) - n):
        if below[i:i + n].all(): return t[i]
    return np.nan


def _first_above(t, v, thr, t_from, t_to):
    m = (t >= t_from) & (t <= t_to) & (v > thr)
    idx = np.flatnonzero(m)
    if not len(idx): return np.nan
    i = idx[0]
    if i == 0: return t[i]
    return t[i - 1] + (thr - v[i - 1]) / (v[i] - v[i - 1]) * (t[i] - t[i - 1])   # 線形補間


def _last_above(t, v, thr, t_from, t_to):
    """[t_from, t_to] で v > thr となる最後の時刻（その後は閾値内に収まる）。線形補間。"""
    m = (t >= t_from) & (t <= t_to) & (v > thr)
    idx = np.flatnonzero(m)
    if not len(idx): return np.nan
    i = idx[-1]
    if i + 1 >= len(t): return t[i]
    return t[i] + (v[i] - thr) / (v[i] - v[i + 1]) * (t[i + 1] - t[i]) if v[i] != v[i + 1] else t[i]


def run(tr):
    if not getattr(tr, "phases", None):
        return "## a8 フェーズごとの解析\n\nフェーズが求まらなかった（meta.json の cue がなく，予定表も記録と合わない）。\n"
    J = tr.compared(); fs = tr.fs; t = tr.t
    E = {n: tr.rs(n) - tr.ref(n) for n in J}
    Eraw = {n: tr.rs(n, False) - tr.ref(n, False) for n in J}
    still = [p for p in tr.phases if p[2] == "still"]
    st_mask = np.zeros(len(t), bool)
    for p in still: st_mask |= (t >= p[0]) & (t < p[1])
    bias_still = {n: np.nanmean(E[n][st_mask], 0) for n in J}
    # 全動作フェーズの主な方向（静止位置の表現用）
    allmove = np.zeros(len(t), bool)
    for p in tr.phases:
        if p[2].startswith("bpm"): allmove |= (t >= p[0]) & (t < p[1])
    W = tr.M["wrist"][allmove]; W = W[~np.isnan(W).any(1)]
    gax = np.linalg.svd(W - W.mean(0), full_matrices=False)[2][0]; gc = W.mean(0)
    if gax[0] < 0: gax = -gax

    # MoCap で実際に止まっているフレーム（手首・肘の 1 秒の振れ幅 < 4 mm）
    held = (C._range_1s(t, tr.M_raw["wrist"]) < 4.0) & (C._range_1s(t, tr.M_raw["elbow"]) < 4.0)

    rows = []; strokes = []; trans = []
    for t0, t1, name, inst in tr.phases:
        mk = (t >= t0) & (t < t1)
        if mk.sum() < 10: continue
        kind = "動作" if name.startswith("bpm") else ("静止" if name in PH.STATIC else "課題外")
        held_pct = 100 * held[mk].mean()
        if kind == "静止":            # 静止フェーズは実際に止まっているフレームだけで評価（ストップ後の動きすぎを除く）
            mk = mk & held
            if mk.sum() < 30: continue
        base = {"phase": inst, "name": name, "kind": kind, "t0": t0, "t1": t1, "dur": t1 - t0,
                "moving_pct": tr.phase_info.get("moving_pct", {}).get(inst, np.nan), "held_pct": held_pct}
        Wm = tr.M["wrist"][mk]
        base["wrist_pos_mm"] = float(np.nanmean((Wm - gc) @ gax))
        # 角度
        ang = None
        if {"shoulder", "elbow", "wrist"} <= set(tr.rs_names):
            ar = _angle(tr.rs("shoulder")[mk], tr.rs("elbow")[mk], tr.rs("wrist")[mk])
            am = _angle(tr.ref("shoulder")[mk], tr.ref("elbow")[mk], tr.ref("wrist")[mk])
            g = ~(np.isnan(ar) | np.isnan(am)); ang = (ar[g], am[g])
        for n in J:
            j = tr.rs_names.index(n)
            r = dict(base, joint=n, label=C.LABEL[n])
            valid = ~np.isnan(tr.P_raw[mk, j, 0])
            r["usable_pct"] = 100 * (valid & ~tr.gross[n][mk]).mean()
            r["gross_lr_pct"] = 100 * tr.gross_lr[n][mk].mean()
            r["gross_other_pct"] = 100 * (tr.gross[n][mk] & ~tr.gross_lr[n][mk]).mean()
            X = tr.rs(n)[mk]; ok = ~np.isnan(E[n][mk]).any(1)
            if ok.sum() < 0.5 * mk.sum() or ok.sum() < 30 or kind == "課題外":
                rows.append(r); continue
            D = C.decompose(E[n][mk][ok], X[ok]); b = D.mean(0)
            r.update(bias_ray=b[0], bias_h=b[1], bias_v=b[2], bias3d=float(np.linalg.norm(b)),
                     sd3d=float(np.sqrt(np.sum(D.var(0)))),
                     rmse=float(np.sqrt(np.mean(np.sum(D ** 2, 1)))))
            es = E[n][mk][ok] - bias_still[n]
            r["rmse_still_corr"] = float(np.sqrt(np.mean(np.sum(es ** 2, 1))))
            if kind == "静止":
                Er = Eraw[n][mk]; okr = ~np.isnan(Er).any(1)
                if okr.sum() > 30:
                    Dr = C.decompose(Er[okr] - Er[okr].mean(0), tr.rs(n, False)[mk][okr])
                    s = Dr.std(0)
                    r.update(jit3d=float(np.sqrt(np.sum(s ** 2))), jit_ray=s[0], jit_h=s[1], jit_v=s[2],
                             jit3d_filt=float(np.sqrt(np.sum(E[n][mk][ok].var(0)))))
                    Xr = tr.rs(n, False)[mk][okr]
                    r["f2f"] = float(np.sqrt(np.mean(np.sum(np.diff(Xr, axis=0) ** 2, 1)) / 2))
            else:
                R = tr.ref(n)[mk][ok]; Xo = X[ok]
                Rc = R - R.mean(0); ax = np.linalg.svd(Rc, full_matrices=False)[2][0]
                a_mc = Rc @ ax; a_rs = (Xo - Xo.mean(0)) @ ax
                rng = np.percentile(a_mc, 97) - np.percentile(a_mc, 3)
                sR = C.speed(Xo, fs); sM = C.speed(R, fs)
                r.update(motion_sd=float(a_mc.std()), range_mm=float(rng),
                         peak_speed_mc=float(np.percentile(sM, 99) / 1000), peak_speed_ratio=float(np.percentile(sR, 99) / np.percentile(sM, 99)))
                if a_mc.std() >= MOTION_SD_MIN:
                    r["amp_ratio"] = float((np.percentile(a_rs, 97) - np.percentile(a_rs, 3)) / rng)
                    r["speed_r"] = float(np.corrcoef(sR, sM)[0, 1])
                    r["nrmse_pct"] = 100 * r["sd3d"] / rng
                    Rfull = np.nan_to_num(tr.ref(n)); best = None
                    for L in np.arange(-0.15, 0.1501, 0.002):
                        e = Xo - _shift(tr, Rfull, L)[mk][ok]; e -= e.mean(0); v = np.mean(np.sum(e ** 2, 1))
                        if best is None or v < best[1]: best = (L, v)
                    r["lag_ms"] = best[0] * 1000
            if ang is not None and len(ang[0]) > 30:
                d = ang[0] - ang[1]
                r.update(ang_bias=float(d.mean()), ang_sd=float(d.std()))
                if np.ptp(ang[1]) > 5:
                    r["ang_r"] = float(np.corrcoef(*ang)[0, 1])
                    r["ang_range_ratio"] = float((np.percentile(ang[0], 97) - np.percentile(ang[0], 3)) / (np.percentile(ang[1], 97) - np.percentile(ang[1], 3)))
            if n in ("elbow",) and "shoulder" in tr.rs_names:
                L = np.linalg.norm(tr.rs("shoulder", False)[mk] - tr.rs("elbow", False)[mk], axis=1); r["upper_sd"] = float(np.nanstd(L))
            if n in ("wrist", "wrist_pose"):
                L = np.linalg.norm(tr.rs("elbow", False)[mk] - tr.rs(n, False)[mk], axis=1); r["fore_sd"] = float(np.nanstd(L))
            rows.append(r)

        # ---------------- ストロークと切り替え（動作フェーズ）
        if kind != "動作": continue
        bpm = int(name[3:]); Tn = 60.0 / bpm
        ext = (t >= t0 - 0.5) & (t < t1 + 1.0)
        Wx = tr.M["wrist"][ext]; okw = ~np.isnan(Wx).any(1)
        c0 = Wx[okw].mean(0); ax0 = np.linalg.svd(Wx[okw] - c0, full_matrices=False)[2][0]
        s = np.nan_to_num((Wx - c0) @ ax0); te = t[ext]; rng = np.percentile(s, 98) - np.percentile(s, 2)
        pk, _ = find_peaks(s, distance=max(1, int(0.6 * Tn * fs)), prominence=0.3 * rng)
        tg, _ = find_peaks(-s, distance=max(1, int(0.6 * Tn * fs)), prominence=0.3 * rng)
        rev = sorted([(i, +1) for i in pk] + [(i, -1) for i in tg])
        rev = [x for k, x in enumerate(rev) if k == 0 or x[1] != rev[k - 1][1]]     # 交互になるように
        # 関節ごとの主な方向（このブロック）
        jax = {}
        for n in J:
            R = tr.ref(n)[ext]; okr = ~np.isnan(R).any(1)
            Rc = R[okr] - R[okr].mean(0); a = np.linalg.svd(Rc, full_matrices=False)[2][0]
            if (Wx[okr] - c0) @ ax0 @ (Rc @ a) < 0: a = -a                         # 手首と同じ向き
            jax[n] = (a, R[okr].mean(0), float((Rc @ a).std()))
        rev_t = {}
        for n in J:
            a, c, sd = jax[n]
            if sd < MOTION_SD_MIN: continue
            pm = np.nan_to_num((tr.ref(n)[ext] - c) @ a); pr = np.nan_to_num((tr.rs(n)[ext] - c) @ a)
            lst = []
            for i, sg in rev:
                w = (te >= te[i] - EXT_WIN) & (te <= te[i] + EXT_WIN); idx = np.flatnonzero(w)
                if len(idx) < 5: lst.append((np.nan, np.nan, np.nan, np.nan)); continue
                im = idx[np.argmax(sg * pm[idx])]; ir = idx[np.argmax(sg * pr[idx])]
                if im in (idx[0], idx[-1]) or ir in (idx[0], idx[-1]):
                    lst.append((np.nan, np.nan, np.nan, np.nan)); continue
                tm_ = np.interp(_parabolic(sg * pm, im), np.arange(len(te)), te)
                trs = np.interp(_parabolic(sg * pr, ir), np.arange(len(te)), te)
                lst.append((tm_, trs, pm[im], pr[ir]))
            rev_t[n] = lst
        for k in range(len(rev) - 1):
            ia, ib = rev[k][0], rev[k + 1][0]
            row = {"phase": inst, "bpm": bpm, "stroke": k + 1, "t_start": float(te[ia]), "t_end": float(te[ib]),
                   "duration_s": float(te[ib] - te[ia]), "nominal_s": Tn, "tempo_ratio": float((te[ib] - te[ia]) / Tn),
                   "wrist_amp_mm": float(abs(s[ib] - s[ia]))}
            for n in rev_t:
                (ma, ra, pma, pra), (mb, rb, pmb, prb) = rev_t[n][k], rev_t[n][k + 1]
                amc = abs(pmb - pma); ars = abs(prb - pra)
                row[f"{n}_amp_mc"] = amc; row[f"{n}_amp_ratio"] = ars / amc if amc > 10 else np.nan
                row[f"{n}_rev_dt_ms"] = (rb - mb) * 1000 if np.isfinite(rb) and np.isfinite(mb) else np.nan
                w = (t >= te[ia]) & (t <= te[ib])
                vm = C.speed(np.nan_to_num(tr.ref(n)), fs)[w]; vr = C.speed(np.nan_to_num(tr.rs(n)), fs)[w]
                row[f"{n}_peak_mc"] = float(vm.max() / 1000) if len(vm) else np.nan
                row[f"{n}_peak_ratio"] = float(vr.max() / vm.max()) if len(vm) and vm.max() > 0 else np.nan
            strokes.append(row)
        # 切り替え（動き始め・止まり）: 静止位置から閾値以上離れた時刻（RS・MoCap それぞれ自分の静止位置が基準）
        nxt = [p for p in tr.phases if p[0] >= t1 - 1e-6]
        rest_after = nxt[0] if nxt and nxt[0][2] == "rest" else None
        tr_row = {"phase": inst, "bpm": bpm}
        for n in J:
            a, c, sd = jax[n]
            if sd < MOTION_SD_MIN: continue
            pm = (tr.ref(n) - c) @ a; pr = (tr.rs(n) - c) @ a
            pre = (t >= t0 - 1.5) & (t < t0 - 0.2)
            if rest_after: post = (t >= rest_after[1] - 1.2) & (t < rest_after[1] - 0.2)
            else: post = (t >= t1 + 1.0) & (t < t1 + 2.0)
            if np.isnan(pm[pre]).all() or np.isnan(pr[pre]).all() or np.isnan(pm[post]).all() or np.isnan(pr[post]).all():
                continue
            thr = max(ONSET_MM, ONSET_K * np.nanstd(pr[pre]))
            dm = np.abs(pm - np.nanmean(pm[pre])); dr = np.abs(pr - np.nanmean(pr[pre]))
            on_m = _first_above(t, np.nan_to_num(dm), thr, t0 - 1.0, t0 + 3.0)
            on_r = _first_above(t, np.nan_to_num(dr), thr, t0 - 1.0, t0 + 3.0)
            em = np.abs(pm - np.nanmean(pm[post])); er = np.abs(pr - np.nanmean(pr[post]))
            t_end = rest_after[1] - 1.2 if rest_after else t1 + 1.0
            off_m = _last_above(t, np.nan_to_num(em), thr, t1 - 2.0, t_end)
            off_r = _last_above(t, np.nan_to_num(er), thr, t1 - 2.0, t_end)
            tr_row[f"{n}_onset_ms"] = (on_r - on_m) * 1000; tr_row[f"{n}_offset_ms"] = (off_r - off_m) * 1000
            tr_row[f"{n}_onset_mc"] = on_m; tr_row[f"{n}_offset_mc"] = off_m; tr_row[f"{n}_thr_mm"] = thr
            if n == "wrist":
                tr_row["onset_after_cue_ms"] = (on_m - t0) * 1000       # 被験者の反応（予定表の位置合わせ込み）
                tr_row["overrun_s"] = off_m - t1                          # 「ストップ」の後も動いていた時間
        trans.append(tr_row)

    # ---------------- 静止フェーズ間の偏りの変化
    drift = []
    for n in J:
        sr = [r for r in rows if r["joint"] == n and r["kind"] == "静止" and r["name"] in PH.TASK_STATIC and "bias3d" in r]
        if len(sr) < 2: continue
        B = np.array([[r["bias_ray"], r["bias_h"], r["bias_v"]] for r in sr]); pos = np.array([r["wrist_pos_mm"] for r in sr])
        rngv = B.max(0) - B.min(0)
        A_ = np.column_stack([np.ones(len(pos)), pos / 100]); co = np.linalg.lstsq(A_, B, rcond=None)[0][1]
        drift.append({"joint": n, "label": C.LABEL[n], "n": len(sr), "range_ray": rngv[0], "range_h": rngv[1], "range_v": rngv[2],
                      "sd3d": float(np.sqrt(np.sum(B.var(0)))), "per100_3d": float(np.linalg.norm(co)),
                      "pos_range": float(np.ptp(pos))})

    tr.save_json("a8_phases.json", {"source": tr.phase_source, "info": tr.phase_info, "rows": rows,
                                    "strokes": strokes, "transitions": trans, "static_drift": drift})
    keys = sorted({k for r in rows for k in r})
    with open(f"{tr.out}/a8_phase_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow(r) for r in rows]
    if strokes:
        keys = list(strokes[0].keys()) + sorted({k for r in strokes for k in r} - set(strokes[0].keys()))
        with open(f"{tr.out}/a8_strokes.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow(r) for r in strokes]

    _figures(tr, J, rows, strokes, trans)
    return _markdown(tr, J, rows, strokes, trans, drift)


# ======================================================================= 図
def _figures(tr, J, rows, strokes, trans):
    plt = C.setup_plot()
    ph = [p[3] for p in tr.phases if p[2] not in PH.OFF_TASK]
    xs = {p: i for i, p in enumerate(ph)}
    fig, ax = plt.subplots(4, 1, figsize=(16, 13), sharex=True)
    panels = [("bias3d", "|偏り| [mm]"), ("sd3d", "偏りを除いた SD [mm]（静止 = ジッタ）"),
              ("rmse_still_corr", "still の偏りで補正した RMSE [mm]"), ("usable_pct", "使える率 [%]")]
    for a, (k, title) in zip(ax, panels):
        for n in J:
            rr = [r for r in rows if r["joint"] == n and r["phase"] in xs]
            a.plot([xs[r["phase"]] for r in rr], [r.get(k, np.nan) for r in rr], "-o", color=C.COLOR[n], label=C.LABEL[n], lw=1.6, ms=5)
        a.set_ylabel(title, fontsize=9)
        for p in ph:
            if p.startswith("bpm"): a.axvspan(xs[p] - .4, xs[p] + .4, color="#2a78d6", alpha=.07, lw=0)
    ax[0].legend(fontsize=8, ncol=4); ax[-1].set_xticks(range(len(ph))); ax[-1].set_xticklabels(ph, rotation=60, fontsize=8)
    fig.suptitle(f"フェーズごとの誤差（{tr.phase_source}）", y=1.0, fontsize=10)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a8_phases.png", dpi=120); plt.close(fig)

    if not strokes: return
    bpms = sorted({r["bpm"] for r in strokes}); Jm = [n for n in J if any(f"{n}_amp_ratio" in r for r in strokes)]
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
    def box(a, key, title, ref=None):
        for k, n in enumerate(Jm):
            for bi, b in enumerate(bpms):
                v = np.array([r.get(f"{n}_{key}", np.nan) for r in strokes if r["bpm"] == b], float); v = v[np.isfinite(v)]
                if not len(v): continue
                x = bi + (k - (len(Jm) - 1) / 2) * 0.22
                a.plot(np.full(len(v), x) + np.random.uniform(-.05, .05, len(v)), v, ".", color=C.COLOR[n], alpha=.45, ms=4)
                a.errorbar(x, v.mean(), yerr=v.std(), fmt="o", color=C.COLOR[n], ms=6, capsize=3, label=C.LABEL[n] if bi == 0 else None)
        if ref is not None: a.axhline(ref, color="#888", lw=1)
        a.set_xticks(range(len(bpms))); a.set_xticklabels([f"{b}" for b in bpms]); a.set_xlabel("テンポ [bpm]"); a.set_title(title)
    box(ax[0, 0], "amp_ratio", "ストロークごとの振幅比 RS/MoCap", 1.0)
    box(ax[0, 1], "rev_dt_ms", "折り返しの時刻のずれ [ms]（+ = RS が遅い）", 0.0)
    box(ax[0, 2], "peak_ratio", "ストロークごとの最大の速さの比", 1.0)
    ax[0, 0].legend(fontsize=8)
    # 被験者の動き
    for bi, b in enumerate(bpms):
        v = np.array([r["tempo_ratio"] for r in strokes if r["bpm"] == b]); ax[1, 0].plot(np.full(len(v), bi) + np.random.uniform(-.08, .08, len(v)), v, ".", color="#444", alpha=.5)
        ax[1, 0].errorbar(bi, v.mean(), yerr=v.std(), fmt="o", color="#eb6834", capsize=3)
        v = np.array([r["wrist_amp_mm"] for r in strokes if r["bpm"] == b]); ax[1, 1].plot(np.full(len(v), bi) + np.random.uniform(-.08, .08, len(v)), v, ".", color="#444", alpha=.5)
        ax[1, 1].errorbar(bi, v.mean(), yerr=v.std(), fmt="o", color="#eb6834", capsize=3)
    ax[1, 0].axhline(1, color="#888", lw=1); ax[1, 0].set_title("ストロークの時間 / 指示（60/bpm）"); ax[1, 1].set_title("手首のストロークの振幅（MoCap）[mm]")
    for a in ax[1, :2]: a.set_xticks(range(len(bpms))); a.set_xticklabels(bpms); a.set_xlabel("テンポ [bpm]")
    for k, n in enumerate(Jm):
        on = [r.get(f"{n}_onset_ms", np.nan) for r in trans]; off = [r.get(f"{n}_offset_ms", np.nan) for r in trans]
        bb = [r["bpm"] for r in trans]
        ax[1, 2].plot(bb, on, "-o", color=C.COLOR[n], label=f"{C.LABEL[n]} 動き始め", lw=1.6, ms=5)
        ax[1, 2].plot(bb, off, "--s", color=C.COLOR[n], label=f"{C.LABEL[n]} 止まり", lw=1.2, ms=4, alpha=.8)
    ax[1, 2].axhline(0, color="#888", lw=1); ax[1, 2].set_title("動き始め・止まりの時刻のずれ [ms]（+ = RS が遅い）")
    ax[1, 2].set_xlabel("テンポ [bpm]"); ax[1, 2].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a8_strokes.png", dpi=120); plt.close(fig)

    # 静止フェーズの偏りの変化
    sr_all = [r for r in rows if r["kind"] == "静止" and r["name"] in PH.TASK_STATIC and "bias3d" in r]
    if sr_all:
        fig, ax = plt.subplots(1, len(J), figsize=(4.2 * len(J), 3.8), sharey=False)
        for a, n in zip(np.atleast_1d(ax), J):
            sr = [r for r in sr_all if r["joint"] == n]
            for k, (key, lab, col) in enumerate([("bias_ray", "奥行き", "#2a78d6"), ("bias_h", "横", "#1baf7a"), ("bias_v", "縦", "#eda100")]):
                a.plot([r["wrist_pos_mm"] for r in sr], [r[key] for r in sr], "o", color=col, label=lab, ms=6)
                for r in sr: a.annotate(r["phase"].replace("rest#", "r"), (r["wrist_pos_mm"], r[key]), fontsize=6, color="#666", xytext=(3, 2), textcoords="offset points")
            a.set_title(C.LABEL[n]); a.set_xlabel("静止時の手首の位置（動きの方向）[mm]"); a.set_ylabel("偏り [mm]")
        np.atleast_1d(ax)[0].legend(fontsize=8)
        fig.suptitle("静止フェーズ（still, rest#k）ごとの偏りと，そのときの腕の位置", y=1.02, fontsize=10)
        fig.tight_layout(); fig.savefig(f"{tr.out}/a8_static_bias.png", dpi=120, bbox_inches="tight"); plt.close(fig)


# ======================================================================= 表
def _markdown(tr, J, rows, strokes, trans, drift):
    md = f"## a8 フェーズごとの解析\n\nフェーズの出どころ: {tr.phase_source}\n\n"
    # フェーズ一覧
    lst = []
    for t0, t1, name, inst in tr.phases:
        d = {"phase": inst, "t": f"{t0:.1f}–{t1:.1f}", "dur": t1 - t0, "mov": tr.phase_info.get("moving_pct", {}).get(inst, np.nan)}
        ss = [s for s in strokes if s["phase"] == inst]
        if ss:
            d["tempo"] = 60.0 / np.mean([s["duration_s"] for s in ss]); d["amp"] = np.mean([s["wrist_amp_mm"] for s in ss])
            d["ns"] = len(ss)
        u = [r for r in rows if r["phase"] == inst]
        d["usable"] = np.mean([r["usable_pct"] for r in u]) if u else np.nan
        d["lr"] = max([r["gross_lr_pct"] for r in u], default=np.nan)
        d["held"] = u[0]["held_pct"] if u and name in PH.STATIC else np.nan
        lst.append(d)
    md += "フェーズ一覧（Take 時刻。実テンポは MoCap の手首のストロークから）\n\n"
    md += C.md_table(lst, [("フェーズ", "phase"), ("時刻 [s]", "t"), ("長さ [s]", "dur"), ("動いている [%]", "mov"),
                           ("ストローク数", "ns", "{:.0f}"), ("実テンポ [bpm]", "tempo"), ("手首の振幅 [mm]", "amp", "{:.0f}"),
                           ("止まれていた [%]", "held"), ("使える率 [%]（関節平均）", "usable"), ("左右の取り違え [%]", "lr")])

    # 静止フェーズ
    md += ("\n### 静止フェーズ（still / rest / count）\n\n実際に止まっているフレーム（MoCap の手首・肘の 1 秒の振れ幅 < 4 mm）だけで評価。"
           "使える率はそのうち誤検出でないフレームの割合。\nジッタ σ（生データ，RS − MoCap の SD，3D）[mm] と |偏り| [mm]\n\n")
    sp = [p[3] for p in tr.phases if p[2] in PH.STATIC]
    tab = []
    for p in sp:
        d = {"phase": p}
        for n in J:
            r = next((x for x in rows if x["phase"] == p and x["joint"] == n), {})
            d[n + "_j"] = r.get("jit3d", np.nan); d[n + "_b"] = r.get("bias3d", np.nan); d[n + "_a"] = r.get("ang_bias", np.nan) if n == "elbow" else np.nan
        tab.append(d)
    cols = [("フェーズ", "phase")]
    for n in J: cols += [(f"{C.LABEL[n]} σ", n + "_j", "{:.2f}"), ("|偏り|", n + "_b")]
    cols += [("肘の角度の偏り [°]", "elbow_a")]
    md += C.md_table(tab, cols)
    if drift:
        md += "\n静止フェーズ間の偏りの変化（still + rest#1〜，同じ人・同じ姿勢のはず）\n\n"
        md += C.md_table(drift, [("関節", "label"), ("フェーズ数", "n", "{:.0f}"), ("奥行きの幅", "range_ray"), ("横の幅", "range_h"),
                                 ("縦の幅", "range_v"), ("SD 3D", "sd3d"), ("静止位置の幅 [mm]", "pos_range", "{:.0f}")])
        md += ("\n静止位置がほぼ同じ（幅が小さい）なら，偏りの変化は腕の位置ではなく，姿勢の微妙な違いや"
               "MediaPipe の当てはまりの揺らぎによるもの。この SD が「静止で測った補正値」の不確かさの目安になる。\n")

    # 動作フェーズ
    mp = [p[3] for p in tr.phases if p[2].startswith("bpm")]
    md += "\n### 動作フェーズ（bpmNN）\n"
    for key, title, f in [("sd3d", "偏りを除いた SD（ばらつき）[mm]", "{:.1f}"), ("bias3d", "|偏り| [mm]", "{:.1f}"),
                          ("rmse_still_corr", "still の偏りで補正した RMSE [mm]", "{:.1f}"),
                          ("lag_ms", "遅れ [ms]（+ = RS が遅い）", "{:+.0f}"), ("amp_ratio", "振幅比 RS/MoCap", "{:.3f}"),
                          ("speed_r", "速さの相関 r", "{:.3f}"), ("peak_speed_ratio", "最大の速さの比", "{:.2f}"),
                          ("nrmse_pct", "形の誤差 / 動きの範囲 [%]", "{:.1f}")]:
        tab = []
        for n in J:
            d = {"label": C.LABEL[n]}
            for p in mp:
                r = next((x for x in rows if x["phase"] == p and x["joint"] == n), {}); d[p] = r.get(key, np.nan)
            tab.append(d)
        md += f"\n{title}\n\n" + C.md_table(tab, [("関節", "label")] + [(p, p, f) for p in mp])
    tab = []
    for p in mp:
        r = next((x for x in rows if x["phase"] == p and x["joint"] == "elbow"), {})
        rw = next((x for x in rows if x["phase"] == p and x["joint"] == "wrist"), {})
        tab.append({"phase": p, "b": r.get("ang_bias", np.nan), "s": r.get("ang_sd", np.nan), "r": r.get("ang_r", np.nan),
                    "rr": r.get("ang_range_ratio", np.nan), "u": r.get("upper_sd", np.nan), "f": rw.get("fore_sd", np.nan)})
    md += "\n肘の角度 [°] と骨長の SD [mm]\n\n" + C.md_table(tab, [("フェーズ", "phase"), ("角度の偏り", "b"), ("角度の SD", "s"),
                                                         ("r", "r", "{:.2f}"), ("変化幅の比", "rr", "{:.2f}"),
                                                         ("上腕長の SD", "u"), ("前腕長(hand)の SD", "f")])
    # ストローク
    if strokes:
        Jm = [n for n in J if any(f"{n}_amp_ratio" in r for r in strokes)]
        tab = []
        for b in sorted({r["bpm"] for r in strokes}):
            ss = [r for r in strokes if r["bpm"] == b]
            d = {"bpm": f"bpm{b}", "n": len(ss), "tr": np.mean([r["tempo_ratio"] for r in ss]),
                 "amp": np.mean([r["wrist_amp_mm"] for r in ss]),
                 "cv": 100 * np.std([r["wrist_amp_mm"] for r in ss]) / np.mean([r["wrist_amp_mm"] for r in ss])}
            for n in Jm:
                for k in ("amp_ratio", "rev_dt_ms", "peak_ratio"):
                    v = np.array([r.get(f"{n}_{k}", np.nan) for r in ss], float); v = v[np.isfinite(v)]
                    d[f"{n}_{k}"] = f"{v.mean():.2f} ± {v.std():.2f}" if k != "rev_dt_ms" and len(v) else (f"{v.mean():+.0f} ± {v.std():.0f}" if len(v) else "–")
            tab.append(d)
        md += "\n### ストロークごと（1 拍 = 片道 1 回。平均 ± SD）\n\n"
        md += C.md_table(tab, [("テンポ", "bpm"), ("本数", "n", "{:.0f}"), ("時間/指示", "tr", "{:.2f}"),
                               ("手首の振幅 [mm]", "amp", "{:.0f}"), ("振幅の CV [%]", "cv")])
        cols = [("テンポ", "bpm")]
        for n in Jm:
            cols += [(f"{C.LABEL[n]} 振幅比", f"{n}_amp_ratio"), ("折り返しのずれ [ms]", f"{n}_rev_dt_ms"), ("最大の速さの比", f"{n}_peak_ratio")]
        md += "\n" + C.md_table(tab, cols)
    if trans:
        Jm = [n for n in J if any(f"{n}_onset_ms" in r for r in trans)]
        tab = [{"phase": r["phase"], **{k: v for k, v in r.items() if k != "phase"}} for r in trans]
        cols = [("フェーズ", "phase"), ("合図→動き始め [ms]", "onset_after_cue_ms", "{:.0f}"), ("ストップ後も動いた [s]", "overrun_s", "{:+.2f}")]
        for n in Jm:
            cols += [(f"{C.LABEL[n]} 動き始めのずれ [ms]", f"{n}_onset_ms", "{:+.0f}"), ("止まりのずれ [ms]", f"{n}_offset_ms", "{:+.0f}")]
        md += "\n### 切り替え（静止 → 動作 → 静止）\n\n「合図→動き始め」は予定表を MoCap に合わせた場合，位置合わせのずれを引いた残り。\n\n" + C.md_table(tab, cols)
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
