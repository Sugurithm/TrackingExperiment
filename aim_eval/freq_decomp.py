"""freq_decomp.py — 入力周波数ごとの分解（ターゲットは既知の正弦波の和なので、周波数ごとに読み分ける）

  python aim_eval/freq_decomp.py --config analysis_out/20261022/config_used.py \
         --out analysis_out/20261022 [--prelim prelim_relaxed] [--n-boot 5000] [--seed 0]

前提: run.py --steps analyze（desk_frame.json）、prelim_inference.py（除外の一覧）を先に回す。既存コードは書き換えない。

課題の設計（session.json / meta.json で確認）
  T = 40 s。周波数は帯域・軸ごとに固定で、seed は位相だけを変える（振幅 ∝ f^-0.5 は全 seed 共通の形）。
    slow x: 2, 5, 11, 17 / 40 Hz   y: 3, 7, 13, 19 / 40 Hz   （0.05〜0.475 Hz）
    fast x: 3, 7, 13, 19, 29, 41   y: 5, 11, 17, 23, 31, 43   （0.075〜1.075 Hz）
  → 同じ周波数が条件ごとに seed 3 本ぶん繰り返される。x と y の周波数は重ならない。

1. 入力周波数ごと（試行 × 軸 × 周波数、per_bin.csv）
   追従   H = カーソル / ターゲット（そのビンの複素振幅の比）: ゲイン |H|、位相、遅れ [ms] = −位相 / (360 f)、
          相対誤差 |カーソル − ターゲット| / |ターゲット|（そのビンでの失敗の大きさ。0 = 完全）
   分担   成分（体幹・上腕・前腕・手）の手先（MCP3）への射影シェア Re(P·conj(E)) / |E|²。和は 1、
          負 = 手先の動きと逆向き（打ち消し）。遠位 = 前腕 + 手（手首の点を使わない）、近位 = 体幹 + 上腕
   位相   各成分の手先に対する位相 ∠(P / E) [deg]（+ = 成分が手先より進む）。近位 → 遠位の位相差
   計測   カメラの手先（MCP3）/ マウス のゲインと遅れ（その周波数でカメラが手の動きを正しく捉えているか）
2. 入力周波数以外（試行ごと、nonin.csv）
   課題外の帯域の下端 lo = 最高の入力周波数 + 0.4 Hz（slow 1.2 Hz、fast 1.5 Hz）
   カーソル速度の Welch PSD: lo–2 / 2–4 / 4–8 Hz の帯域パワー、lo–8 Hz のピーク周波数（修正のリズム）
   成分の速度 PSD の lo–4 Hz の帯域パワーから、同じセッションの静止試行（ノイズ床）を引いた「課題外の動き」が
   どの成分から出ているか（シェア）。課題の帯域（入力周波数）のシェアと比べる
3. 統計（人 = 単位）
   周波数への依存   人 × 帯域 × DPI × 軸ごとに、指標を log2 f に回帰した傾き（1 オクターブあたりの変化）→ 人をまたいで検定
   DPI × 周波数     d(f) = 800 − 400 を log2 f に回帰した傾き（DPI の効果が速い成分ほど大きいか）
   再現性           同じ条件の seed どうしで周波数プロファイル（周波数ごとの値の並び）の相関
   個人差           人の傾きの 400 と 800 での順位の一致
   協調 ↔ 追従      周波数ごとに、(人 × DPI) と seed を除いた残差で 遠位シェア・上腕シェア・手シェア【参考】 と
                    相対誤差・遅れ の相関。周波数をまたいで Fisher z で平均し、向きがそろうかを見る
手首・MCP3 は MoCap 未照合なので、手の成分を含む指標（手シェア、前腕シェア）は【参考】。遠位シェアは手首の点を使わない。
"""
import argparse, csv, importlib.util, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common as C                                      # noqa: E402
from scipy import stats                                 # noqa: E402
from scipy.signal import welch, butter, filtfilt        # noqa: E402

SKILL = {"Syou": 3, "Yuya": 3, "Suguru": 3, "kakeru": 2, "xin": 2, "Kosuke": 1, "Yuuki": 1, "Suzuka": 1}
PARTS = ["trunk", "upperarm", "forearm", "hand"]
PLAB = {"trunk": "体幹", "upperarm": "上腕", "forearm": "前腕【参考】", "hand": "手【参考】",
        "distal": "遠位（肘→MCP3）", "proximal": "近位（肘の位置）"}
FS_TASK = 250.0
WELCH_S = 4.0
NONIN_LO_MARGIN = 0.4          # 課題外の帯域の下端 = 最高の入力周波数 + これ [Hz]（Welch 4 s の分解能 0.25 Hz の漏れを避ける）
NONIN_EDGES = (2.0, 4.0, 8.0)  # 下端〜2 / 2〜4 / 4〜8 Hz
JOINT_HI = 4.0                 # 関節は 5 Hz の低域通過済みなので 4 Hz まで


class Args:
    def __init__(self, trial, **kw):
        self.trial = trial
        self.rs = self.tmc = self.axes = self.out = None
        for k, v in kw.items():
            setattr(self, k, v)


def load_config(path):
    spec = importlib.util.spec_from_file_location("aim_config", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def resolve(d):
    return os.path.normpath(d if os.path.isabs(d) else os.path.join(ROOT, d))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def circ_mean_deg(a, w=None):
    a = np.radians(np.asarray(a, float)); ok = np.isfinite(a)
    if not ok.any():
        return np.nan
    w = np.ones_like(a) if w is None else np.asarray(w, float)
    return float(np.degrees(np.angle(np.sum(w[ok] * np.exp(1j * a[ok])))))


# ------------------------------------------------------------------ 1. 入力周波数ごと
def per_bin(tr, key, meta):
    tg, cu, mo = tr.task_grid("target"), tr.task_grid("cursor"), tr.task_grid("mouse")
    parts = C.contributions(tr, end="mcp3")
    rows = []
    if parts is None:
        return rows
    for k, ax in enumerate(C.AX):
        nb, f, _ = tr.freqs(ax)
        n, FT = C.spectrum(tg[:, k], tr.T); _, FC = C.spectrum(cu[:, k], tr.T); _, FM = C.spectrum(mo[:, k], tr.T)
        T_, Cu, Mo = C.at_bins(n, FT, nb), C.at_bins(n, FC, nb), C.at_bins(n, FM, nb)
        nC = C.bin_noise(FC, nb)
        _, FE = C.spectrum(parts["_total"][:, k], tr.T)
        E = C.at_bins(n, FE, nb); nE = C.bin_noise(FE, nb)
        P, nP = {}, {}
        for p in PARTS:
            _, FP = C.spectrum(parts[p][:, k], tr.T)
            P[p] = C.at_bins(n, FP, nb); nP[p] = C.bin_noise(FP, nb)
        P["proximal"] = P["trunk"] + P["upperarm"]; P["distal"] = P["forearm"] + P["hand"]
        for i in range(len(nb)):
            H = Cu[i] / T_[i]
            r = {**meta, "key": key, "axis": ax, "n": int(nb[i]), "f": float(f[i]),
                 "amp_target_mm": float(np.abs(T_[i])),
                 "gain": float(np.abs(H)), "phase_deg": float(np.degrees(np.angle(H))),
                 "rel_err": float(np.abs(Cu[i] - T_[i]) / np.abs(T_[i])),
                 "H_re": float(H.real), "H_im": float(H.imag),
                 "snr_cursor": float(np.abs(Cu[i]) / nC[i]) if nC[i] > 0 else np.nan,
                 "snr_end": float(np.abs(E[i]) / nE[i]) if nE[i] > 0 else np.nan}
            r["lag_ms"] = -r["phase_deg"] / (360 * r["f"]) * 1000
            G = E[i] / Mo[i] if np.abs(Mo[i]) > 0 else np.nan
            r["cam_gain"] = float(np.abs(G)); r["cam_lag_ms"] = float(-np.degrees(np.angle(G)) / (360 * r["f"]) * 1000)
            for p in PARTS + ["proximal", "distal"]:
                Pi = P[p][i]
                r[f"share_{p}"] = float(np.real(Pi * np.conj(E[i])) / np.abs(E[i]) ** 2) if np.abs(E[i]) > 0 else np.nan
                r[f"phase_{p}"] = float(np.degrees(np.angle(Pi / E[i]))) if np.abs(E[i]) > 0 and np.abs(Pi) > 0 else np.nan
                if p in nP:
                    r[f"snr_{p}"] = float(np.abs(Pi) / nP[p][i]) if nP[p][i] > 0 else np.nan
            pd_ = P["distal"][i] / P["proximal"][i] if np.abs(P["proximal"][i]) > 0 else np.nan
            r["phase_dist_minus_prox"] = float(np.degrees(np.angle(pd_)))
            rows.append(r)
    return rows


# ------------------------------------------------------------------ 2. 入力周波数以外
def comp_vel(tr):
    parts = C.contributions(tr, end="mcp3")
    if parts is None:
        return None
    V = {p: np.gradient(parts[p], axis=0) * C.FS_U for p in PARTS}
    V["proximal"] = V["trunk"] + V["upperarm"]; V["distal"] = V["forearm"] + V["hand"]
    return V


def band_power(x, fs, band):
    """x (N,2) の各軸の Welch PSD を足し、band の積分 [(mm/s)²]。NaN は最長の連続区間だけ使う。"""
    ok = np.isfinite(x).all(1)
    if ok.sum() < fs * WELCH_S:
        return np.nan, None, None
    idx = np.flatnonzero(ok); brk = np.flatnonzero(np.diff(idx) > 1)
    segs = np.split(idx, brk + 1); seg = max(segs, key=len)
    if len(seg) < fs * WELCH_S:
        return np.nan, None, None
    y = x[seg]
    f, Pxx = welch(y - y.mean(0), fs=fs, nperseg=int(fs * WELCH_S), axis=0)
    P = Pxx.sum(1)
    m = (f >= band[0]) & (f < band[1])
    return float(np.trapezoid(P[m], f[m])), f, P


def nonin_lo(tr):
    fmax = max(float(np.max(tr.freqs(ax)[1])) for ax in C.AX)
    return max(1.2, round((fmax + NONIN_LO_MARGIN) * 4) / 4)


def nonin(tr, static_V):
    out = {}
    lo_ = nonin_lo(tr)
    out["nonin_lo_hz"] = lo_
    t = np.arange(int(round(tr.T * FS_TASK))) / FS_TASK
    g = np.isfinite(tr.task_t)
    cu = np.column_stack([np.interp(t, tr.task_t[g], tr.cur[g, k]) for k in range(2)])
    vc = np.gradient(cu, axis=0) * FS_TASK
    b, a = butter(4, 20 / (FS_TASK / 2)); vc = filtfilt(b, a, vc, axis=0)
    f, Pxx = welch(vc - vc.mean(0), fs=FS_TASK, nperseg=int(FS_TASK * WELCH_S), axis=0)
    P = Pxx.sum(1)
    for name, lo, hi in (("lo_2", lo_, NONIN_EDGES[0]), ("2_4", NONIN_EDGES[0], NONIN_EDGES[1]),
                         ("4_8", NONIN_EDGES[1], NONIN_EDGES[2])):
        m = (f >= lo) & (f < hi)
        out[f"cur_pow_{name}"] = float(np.trapezoid(P[m], f[m]))
    m = (f >= lo_) & (f < NONIN_EDGES[2])
    out["cur_peak_hz"] = float(f[m][np.argmax(P[m])])
    out["cur_centroid_hz"] = float(np.sum(f[m] * P[m]) / np.sum(P[m]))
    out["_cur_psd"] = (f, P)
    V = comp_vel(tr)
    if V is not None:
        ex = {}
        for p in PARTS + ["proximal", "distal"]:
            bp, _, _ = band_power(V[p], C.FS_U, (lo_, JOINT_HI))
            fl = static_V.get((p, lo_), np.nan) if static_V else np.nan
            out[f"j_pow_{p}"] = bp; out[f"j_floor_{p}"] = fl
            ex[p] = max(bp - fl, 0.0) if np.isfinite(bp) and np.isfinite(fl) else np.nan
            out[f"j_excess_{p}"] = ex[p]
            out[f"j_snr_{p}"] = float(np.sqrt(bp / fl)) if np.isfinite(bp) and np.isfinite(fl) and fl > 0 else np.nan
        tot = np.nansum([ex[p] for p in PARTS])
        for p in PARTS:
            out[f"j_exshare_{p}"] = float(ex[p] / tot) if tot > 0 and np.isfinite(ex[p]) else np.nan
        tot2 = np.nansum([ex[p] for p in ("proximal", "distal")])
        out["j_exshare_distal"] = float(ex["distal"] / tot2) if tot2 > 0 else np.nan
    return out


def static_powers(sess_dir, axes, tmp):
    cand = [d for d, m in C.list_trials(sess_dir, kinds=("static",))
            if os.path.exists(os.path.join(d, "joints", "rs_joints.npz"))]
    if not cand:
        return None
    st = C.Trial(Args(cand[0], axes=axes, out=tmp))
    V = comp_vel(st)
    if V is None:
        return None
    return {(p, lo): band_power(V[p], C.FS_U, (lo, JOINT_HI))[0] for p in V for lo in (1.2, 1.5)}


# ------------------------------------------------------------------ 統計の道具
def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]; o = np.argsort(pv); n = len(pv)
    r = pv[o] * n / (np.arange(n) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(n); qq[o] = np.minimum(r, 1); q[ok] = qq
    return q


def boot_mean_ci(v, rng, n_boot):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan
    i = rng.integers(0, len(v), (n_boot, len(v)))
    m = v[i].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def one_sample(d, rng, n_boot):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) < 4:
        return {"n": len(d), "mean": np.nan, "lo": np.nan, "hi": np.nan, "dz": np.nan, "n_pos": 0, "n_neg": 0, "p": np.nan}
    lo, hi = boot_mean_ci(d, rng, n_boot)
    try:
        p = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p = np.nan
    return {"n": int(len(d)), "mean": float(d.mean()), "lo": lo, "hi": hi,
            "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan,
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "p": p}


def residualize(rows, k, factors):
    y = np.array([fnum(r.get(k)) for r in rows]); ok = np.isfinite(y)
    cols = [np.ones(len(rows))]
    for fac in factors:
        lv = sorted({tuple(r[f] for f in fac) for r in rows})
        for l in lv[1:]:
            cols.append(np.array([tuple(r[f] for f in fac) == l for r in rows], float))
    X = np.column_stack(cols); res = np.full(len(rows), np.nan)
    if ok.sum() <= X.shape[1]:
        return res, X.shape[1]
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    res[ok] = y[ok] - X[ok] @ beta
    return res, int(np.linalg.matrix_rank(X[ok]))


def fmt(x, f="{:.2f}"):
    return f.format(x) if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else "–"


def md(rows, cols):
    h = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        h += "| " + " | ".join((fmt(r.get(c[1]), c[2]) if len(c) > 2 else str(r.get(c[1], "–"))) for c in cols) + " |\n"
    return h


def save(L, path):
    keys = []
    for r in L:
        for k in r:
            if k not in keys and not k.startswith("_"):
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for r in L:
            w.writerow(r)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prelim", default="prelim_relaxed")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = resolve(a.out)
    pout = os.path.join(out, "freq_decomp"); os.makedirs(pout, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    logf = open(os.path.join(pout, "freq_log.md"), "w", encoding="utf-8")

    def lg(s=""):
        print(s); logf.write(s + "\n"); logf.flush()

    lg(f"# freq_decomp（{time.strftime('%Y-%m-%d %H:%M:%S')}）\n")
    lg(f"config: {a.config} / out: {out} / prelim: {a.prelim} / n_boot {a.n_boot} / seed {a.seed}\n")

    keep = {r["key"] for r in csv.DictReader(open(os.path.join(out, a.prelim, "trials_prelim.csv"), encoding="utf-8"))}
    trials = []
    for e in cfg.SESSIONS:
        for t, meta in C.list_trials(resolve(e["dir"]), kinds=("track",)):
            if cfg.BANDS and meta["task"].get("band") not in cfg.BANDS:
                continue
            key = f"{e['participant']}_{float(e['dpi']):g}_{meta['task']['band']}_s{int(meta['task']['seed'])}"
            if key in keep:
                trials.append((key, t, e))
    lg(f"試行: {len(trials)} 本\n")

    # ---- 試行ごとの計算
    B, NI, psd = [], [], {}
    st_cache = {}
    tmp = os.path.join(out, "_freq_tmp")
    t0 = time.time()
    for key, d, e in trials:
        name = e["participant"]
        axes = os.path.join(out, name, "desk_frame.json")
        axes = axes if os.path.exists(axes) else None
        tr = C.Trial(Args(d, axes=axes, out=tmp))
        meta = {"participant": name, "dpi": float(e["dpi"]), "band": tr.band, "seed": int(tr.seed),
                "skill": SKILL.get(name, np.nan)}
        B += per_bin(tr, key, meta)
        sd = tr.sess_dir
        if sd not in st_cache:
            st_cache[sd] = static_powers(sd, axes, tmp)
        r = nonin(tr, st_cache[sd])
        psd[key] = r.pop("_cur_psd")
        NI.append({**meta, "key": key, **r})
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    lg(f"計算 {time.time() - t0:.0f} s（周波数ビン {len(B)} 行、静止試行 {sum(v is not None for v in st_cache.values())} / {len(st_cache)} セッション）\n")
    save(B, os.path.join(pout, "per_bin.csv")); save(NI, os.path.join(pout, "nonin.csv"))

    people = sorted({r["participant"] for r in B}, key=str.lower)
    bands = [b for b in ("slow", "fast") if any(r["band"] == b for r in B)]
    dpis = sorted({r["dpi"] for r in B})
    freqs = {(b, ax): sorted({r["f"] for r in B if r["band"] == b and r["axis"] == ax}) for b in bands for ax in C.AX}

    # ---- 人 × 条件 × 軸 × 周波数 の平均（seed 3 本）。H は複素平均してからゲイン・位相
    METRICS = ["gain", "lag_ms", "rel_err", "cam_gain", "cam_lag_ms",
               "share_trunk", "share_upperarm", "share_forearm", "share_hand", "share_distal",
               "phase_dist_minus_prox", "phase_trunk", "phase_upperarm", "phase_forearm", "phase_hand"]
    CIRC = {"phase_dist_minus_prox", "phase_trunk", "phase_upperarm", "phase_forearm", "phase_hand"}
    PM = []
    for p in people:
        for b in bands:
            for dpi in dpis:
                for ax in C.AX:
                    for f in freqs[(b, ax)]:
                        rr = [r for r in B if r["participant"] == p and r["band"] == b and r["dpi"] == dpi
                              and r["axis"] == ax and r["f"] == f]
                        if not rr:
                            continue
                        m = {"participant": p, "band": b, "dpi": dpi, "axis": ax, "f": f, "n_seed": len(rr),
                             "skill": SKILL.get(p, np.nan)}
                        H = np.mean([r["H_re"] + 1j * r["H_im"] for r in rr])
                        m["gain_H"] = float(np.abs(H)); m["lag_H_ms"] = float(-np.degrees(np.angle(H)) / (360 * f) * 1000)
                        for k in METRICS:
                            v = [r[k] for r in rr]
                            m[k] = circ_mean_deg(v) if k in CIRC else float(np.nanmean(v))
                        PM.append(m)
    save(PM, os.path.join(pout, "person_freq_means.csv"))

    def pm(p, b, dpi, ax, f, k):
        r = [x for x in PM if x["participant"] == p and x["band"] == b and x["dpi"] == dpi and x["axis"] == ax and x["f"] == f]
        return r[0][k] if r else np.nan

    # ---- 1. 群の周波数プロファイル
    lg("## 1. 入力周波数ごとのプロファイル（9 人の平均 [人のブートストラップ 95 % CI]）\n")
    lg("gain = |カーソル/ターゲット|（seed を複素平均）。lag = 位相からの遅れ [ms]。rel_err = そのビンの相対誤差。"
       "share_* = 手先（MCP3）への射影シェア（和 1、負 = 逆向き）。cam_gain / cam_lag = カメラの手先 / マウス。\n")
    prof = []
    PROF_K = ["gain_H", "lag_H_ms", "rel_err", "share_trunk", "share_upperarm", "share_forearm", "share_hand",
              "share_distal", "phase_dist_minus_prox", "cam_gain", "cam_lag_ms"]
    for b in bands:
        for dpi in dpis:
            for ax in C.AX:
                for f in freqs[(b, ax)]:
                    row = {"band": b, "dpi": dpi, "axis": ax, "f": f}
                    for k in PROF_K:
                        v = np.array([pm(p, b, dpi, ax, f, k) for p in people], float)
                        if k == "phase_dist_minus_prox":
                            row[k] = circ_mean_deg(v)
                            row[k + "_R"] = float(np.abs(np.nanmean(np.exp(1j * np.radians(v)))))
                        else:
                            row[k] = float(np.nanmean(v))
                            row[k + "_lo"], row[k + "_hi"] = boot_mean_ci(v, rng, a.n_boot)
                    prof.append(row)
    save(prof, os.path.join(pout, "group_profile.csv"))
    lg(md(prof, [("帯域", "band"), ("DPI", "dpi", "{:.0f}"), ("軸", "axis"), ("f [Hz]", "f", "{:.3f}"),
                 ("gain", "gain_H", "{:.2f}"), ("lag [ms]", "lag_H_ms", "{:+.0f}"), ("rel_err", "rel_err", "{:.2f}"),
                 ("体幹", "share_trunk", "{:+.2f}"), ("上腕", "share_upperarm", "{:+.2f}"),
                 ("前腕", "share_forearm", "{:+.2f}"), ("手", "share_hand", "{:+.2f}"), ("遠位", "share_distal", "{:+.2f}"),
                 ("遠−近 位相", "phase_dist_minus_prox", "{:+.0f}"), ("位相のそろい R", "phase_dist_minus_prox_R", "{:.2f}"),
                 ("cam gain", "cam_gain", "{:.2f}"), ("cam lag [ms]", "cam_lag_ms", "{:+.0f}")]))

    # ---- 2. 周波数への依存（人ごとの傾き / オクターブ）
    lg("\n## 2. 周波数への依存（1 オクターブ = 周波数 2 倍あたりの変化。人 × 帯域 × DPI × 軸ごとの回帰の傾き → 9 人で検定）\n")
    SL_K = ["gain_H", "lag_H_ms", "rel_err", "share_trunk", "share_upperarm", "share_forearm", "share_hand",
            "share_distal", "phase_dist_minus_prox", "cam_gain", "cam_lag_ms"]
    slopes, slope_tests = {}, []
    for b in bands:
        for dpi in dpis:
            for ax in C.AX:
                fs = np.array(freqs[(b, ax)])
                for k in SL_K:
                    d = []
                    for p in people:
                        y = np.array([pm(p, b, dpi, ax, f, k) for f in fs], float)
                        if k == "phase_dist_minus_prox":
                            y = np.degrees(np.unwrap(np.radians(y)))
                        ok = np.isfinite(y)
                        s = np.polyfit(np.log2(fs[ok]), y[ok], 1)[0] if ok.sum() >= 3 else np.nan
                        slopes[(p, b, dpi, ax, k)] = s; d.append(s)
                    slope_tests.append({"band": b, "dpi": dpi, "axis": ax, "k": k, **one_sample(d, rng, a.n_boot)})
    for b in bands:
        idx = [i for i, r in enumerate(slope_tests) if r["band"] == b]
        for i, q in zip(idx, bh([slope_tests[i]["p"] for i in idx])):
            slope_tests[i]["q"] = float(q)
    save(slope_tests, os.path.join(pout, "freq_slopes.csv"))
    lg(md(slope_tests, [("帯域", "band"), ("DPI", "dpi", "{:.0f}"), ("軸", "axis"), ("指標", "k"), ("人", "n", "{:.0f}"),
                        ("傾き/oct", "mean", "{:+.3f}"), ("CI 下", "lo", "{:+.3f}"), ("CI 上", "hi", "{:+.3f}"),
                        ("dz", "dz", "{:+.2f}"), ("+", "n_pos", "{:.0f}"), ("−", "n_neg", "{:.0f}"),
                        ("p", "p", "{:.3f}"), ("q", "q", "{:.3f}")]))

    # 個人差: 人の傾きは 400 と 800 で順位が保たれるか
    lg("\n### 傾きの個人差（人の傾きの 400 と 800 での Spearman ρ）\n")
    ind = []
    if len(dpis) == 2:
        for b in bands:
            for ax in C.AX:
                for k in SL_K:
                    x = [slopes[(p, b, dpis[0], ax, k)] for p in people]; y = [slopes[(p, b, dpis[1], ax, k)] for p in people]
                    ok = np.isfinite(x) & np.isfinite(y)
                    s = stats.spearmanr(np.array(x)[ok], np.array(y)[ok]) if ok.sum() >= 5 else None
                    ind.append({"band": b, "axis": ax, "k": k, "rho": float(s.statistic) if s else np.nan,
                                "p": float(s.pvalue) if s else np.nan})
        save(ind, os.path.join(pout, "slope_individual.csv"))
        lg(md(ind, [("帯域", "band"), ("軸", "axis"), ("指標", "k"), ("ρ(400,800)", "rho", "{:+.2f}"), ("p", "p", "{:.3f}")]))

    # ---- 3. DPI × 周波数
    lg("\n## 3. DPI × 周波数（d(f) = 800 − 400 の、周波数への傾き / オクターブ。+ = 速い成分ほど 800 で大きい）\n")
    dxf = []
    if len(dpis) == 2:
        for b in bands:
            for ax in C.AX:
                fs = np.array(freqs[(b, ax)])
                for k in SL_K:
                    d, dm = [], []
                    for p in people:
                        y0 = np.array([pm(p, b, dpis[0], ax, f, k) for f in fs], float)
                        y1 = np.array([pm(p, b, dpis[1], ax, f, k) for f in fs], float)
                        dd = y1 - y0
                        if k == "phase_dist_minus_prox":
                            dd = (dd + 180) % 360 - 180
                        ok = np.isfinite(dd)
                        d.append(np.polyfit(np.log2(fs[ok]), dd[ok], 1)[0] if ok.sum() >= 3 else np.nan)
                        dm.append(np.nanmean(dd))
                    t1 = one_sample(dm, rng, a.n_boot); t2 = one_sample(d, rng, a.n_boot)
                    dxf.append({"band": b, "axis": ax, "k": k, "d_mean": t1["mean"], "d_lo": t1["lo"], "d_hi": t1["hi"],
                                "d_pos": t1["n_pos"], "d_p": t1["p"], "slope": t2["mean"], "s_lo": t2["lo"], "s_hi": t2["hi"],
                                "s_dz": t2["dz"], "s_pos": t2["n_pos"], "s_neg": t2["n_neg"], "s_p": t2["p"]})
        for b in bands:
            idx = [i for i, r in enumerate(dxf) if r["band"] == b]
            for key in ("d_p", "s_p"):
                for i, q in zip(idx, bh([dxf[i][key] for i in idx])):
                    dxf[i][key.replace("_p", "_q")] = float(q)
        save(dxf, os.path.join(pout, "dpi_x_freq.csv"))
        lg(md(dxf, [("帯域", "band"), ("軸", "axis"), ("指標", "k"), ("d の平均", "d_mean", "{:+.3f}"),
                    ("d+ 人", "d_pos", "{:.0f}"), ("d p", "d_p", "{:.3f}"), ("d q", "d_q", "{:.3f}"),
                    ("d の傾き/oct", "slope", "{:+.3f}"), ("CI 下", "s_lo", "{:+.3f}"), ("CI 上", "s_hi", "{:+.3f}"),
                    ("dz", "s_dz", "{:+.2f}"), ("+", "s_pos", "{:.0f}"), ("−", "s_neg", "{:.0f}"),
                    ("傾き p", "s_p", "{:.3f}"), ("傾き q", "s_q", "{:.3f}")]))

    # ---- 4. 周波数プロファイルの再現性（同じ条件の seed どうし）
    lg("\n## 4. 周波数プロファイルの seed 間の再現性\n")
    lg("同じ人 × 帯域 × DPI × 軸で、seed のペアごとに周波数ごとの値の並びの相関を取り、中央値を出す。"
       "比較として、別の人の同じ条件の seed とのペア（人をまたぐ）の中央値も出す。\n")
    rel = []
    for k in ["rel_err", "lag_ms", "gain", "share_upperarm", "share_forearm", "share_hand", "share_distal", "share_trunk"]:
        for b in bands:
            same, other = [], []
            for dpi in dpis:
                for ax in C.AX:
                    prof_ = {}
                    for r in B:
                        if r["band"] == b and r["dpi"] == dpi and r["axis"] == ax:
                            prof_.setdefault((r["participant"], r["seed"]), {})[r["f"]] = r[k]
                    keys = sorted(prof_)
                    for i in range(len(keys)):
                        for j in range(i + 1, len(keys)):
                            fs = sorted(set(prof_[keys[i]]) & set(prof_[keys[j]]))
                            x = np.array([prof_[keys[i]][f] for f in fs]); y = np.array([prof_[keys[j]][f] for f in fs])
                            ok = np.isfinite(x) & np.isfinite(y)
                            if ok.sum() < 3 or x[ok].std() == 0 or y[ok].std() == 0:
                                continue
                            rr = np.corrcoef(x[ok], y[ok])[0, 1]
                            (same if keys[i][0] == keys[j][0] else other).append(rr)
            rel.append({"k": k, "band": b, "r_same": float(np.median(same)) if same else np.nan, "n_same": len(same),
                        "r_other": float(np.median(other)) if other else np.nan, "n_other": len(other),
                        "p": float(stats.mannwhitneyu(same, other).pvalue) if same and other else np.nan})
    save(rel, os.path.join(pout, "profile_reliability.csv"))
    lg(md(rel, [("指標", "k"), ("帯域", "band"), ("同じ人の seed 間 r（中央値）", "r_same", "{:+.2f}"), ("組", "n_same", "{:.0f}"),
                ("別の人との r（中央値）", "r_other", "{:+.2f}"), ("組", "n_other", "{:.0f}"), ("p (Mann-Whitney)", "p", "{:.4f}")]))
    lg("\n同じ人の r が別の人の r を上回る = 周波数プロファイルに個人特有の形がある"
       "（どちらも高い = 全員に共通の形、どちらも低い = 周波数ごとの値はノイズに埋もれている）。\n")

    # ---- 5. 協調 ↔ 追従（周波数ごと、条件内の残差相関）
    lg("\n## 5. 周波数ごとの協調 ↔ 追従（同じ人・同じ DPI の試行間。(人 × DPI) と seed を除いた残差相関）\n")
    lg("周波数ごとに r を出し、帯域 × 軸ごとに Fisher z で平均（周波数の数で重みなし）して、向きがそろうかを見る。"
       "+ = その協調の指標が大きい試行ほど、そのビンの値（相対誤差・遅れ）が大きい（= 悪い）。\n")
    link = []
    COORD = [("share_distal", "遠位シェア"), ("share_upperarm", "上腕シェア"), ("share_trunk", "体幹シェア"),
             ("share_hand", "手シェア【参考】")]
    PERF = [("rel_err", "相対誤差"), ("lag_ms", "遅れ [ms]")]
    for stratum, sd in [(f"{d:.0f}", [d]) for d in dpis] + [("pooled", dpis)]:
        for b in bands:
            for ax in C.AX:
                for ck, cl in COORD:
                    for pk, pl in PERF:
                        rs, ns = [], []
                        for f in freqs[(b, ax)]:
                            rows = [r for r in B if r["band"] == b and r["axis"] == ax and r["f"] == f and r["dpi"] in sd]
                            fac = [("participant", "dpi"), ("seed",)]
                            rx, k1 = residualize(rows, ck, fac); ry, k2 = residualize(rows, pk, fac)
                            ok = np.isfinite(rx) & np.isfinite(ry)
                            if ok.sum() - max(k1, k2) < 5:
                                continue
                            rr = float(np.corrcoef(rx[ok], ry[ok])[0, 1])
                            rs.append(rr); ns.append(int(ok.sum()) - max(k1, k2) - 1)
                            link.append({"stratum": stratum, "band": b, "axis": ax, "f": f, "coord": ck, "clab": cl,
                                         "perf": pk, "plab": pl, "r": rr, "df": ns[-1]})
                        if rs:
                            z = np.arctanh(np.clip(rs, -0.999, 0.999))
                            zbar = float(np.mean(z))
                            se = float(np.sqrt(np.mean(1 / (np.array(ns) - 1))) / np.sqrt(len(z)))  # 周波数を独立とみなした近似
                            link.append({"stratum": stratum, "band": b, "axis": ax, "f": "all", "coord": ck, "clab": cl,
                                         "perf": pk, "plab": pl, "r": float(np.tanh(zbar)),
                                         "n_f": len(rs), "n_pos": int(np.sum(np.array(rs) > 0)),
                                         "p": float(2 * stats.norm.sf(abs(zbar) / se)) if se > 0 else np.nan})
    summ = [r for r in link if r["f"] == "all"]
    for key in sorted({r["stratum"] for r in summ}):
        idx = [i for i, r in enumerate(summ) if r["stratum"] == key]
        for i, q in zip(idx, bh([summ[i]["p"] for i in idx])):
            summ[i]["q"] = float(q)
    save(link, os.path.join(pout, "freq_link.csv"))
    lg(md(sorted(summ, key=lambda r: (r["stratum"] != "pooled", r["p"]))[:40],
          [("層", "stratum"), ("帯域", "band"), ("軸", "axis"), ("協調", "clab"), ("追従", "plab"),
           ("周波数平均 r", "r", "{:+.2f}"), ("周波数", "n_f", "{:.0f}"), ("r>0 の周波数", "n_pos", "{:.0f}"),
           ("p（近似）", "p", "{:.4f}"), ("q", "q", "{:.3f}")]))
    lg("\n（p は周波数どうしを独立とみなした近似。同じ試行のビンなので実際には相関があり、p は小さめに出る）\n")

    # ---- 6. 入力周波数以外（課題外の動き）
    lg("\n## 6. 入力周波数以外の動き（課題に無い動き = 修正・揺れ）\n")
    NIK = ["cur_pow_lo_2", "cur_pow_2_4", "cur_pow_4_8", "cur_peak_hz", "cur_centroid_hz",
           "j_exshare_trunk", "j_exshare_upperarm", "j_exshare_forearm", "j_exshare_hand", "j_exshare_distal",
           "j_snr_trunk", "j_snr_upperarm", "j_snr_forearm", "j_snr_hand", "j_snr_distal"]
    nim = []
    for p in people:
        for b in bands:
            for dpi in dpis:
                rr = [r for r in NI if r["participant"] == p and r["band"] == b and r["dpi"] == dpi]
                m = {"participant": p, "band": b, "dpi": dpi, "skill": SKILL.get(p, np.nan)}
                for k in NIK:
                    m[k] = float(np.nanmean([r.get(k, np.nan) for r in rr]))
                nim.append(m)
    save(nim, os.path.join(pout, "nonin_person.csv"))
    grp = []
    for b in bands:
        for dpi in dpis:
            row = {"band": b, "dpi": dpi}
            for k in NIK:
                v = [r[k] for r in nim if r["band"] == b and r["dpi"] == dpi]
                row[k] = float(np.nanmean(v)); row[k + "_lo"], row[k + "_hi"] = boot_mean_ci(v, rng, a.n_boot)
            grp.append(row)
    lg("### 群の平均（9 人）\n")
    lg(md(grp, [("帯域", "band"), ("DPI", "dpi", "{:.0f}"), ("カーソル 下端–2 Hz", "cur_pow_lo_2", "{:.0f}"),
                ("2–4 Hz", "cur_pow_2_4", "{:.0f}"), ("4–8 Hz", "cur_pow_4_8", "{:.0f}"),
                ("ピーク [Hz]", "cur_peak_hz", "{:.2f}"), ("重心 [Hz]", "cur_centroid_hz", "{:.2f}"),
                ("関節 SNR 体幹", "j_snr_trunk", "{:.1f}"), ("上腕", "j_snr_upperarm", "{:.1f}"),
                ("前腕", "j_snr_forearm", "{:.1f}"), ("手", "j_snr_hand", "{:.1f}"), ("遠位", "j_snr_distal", "{:.1f}"),
                ("課題外シェア 体幹", "j_exshare_trunk", "{:.2f}"), ("上腕", "j_exshare_upperarm", "{:.2f}"),
                ("前腕", "j_exshare_forearm", "{:.2f}"), ("手", "j_exshare_hand", "{:.2f}"), ("遠位(2 成分)", "j_exshare_distal", "{:.2f}")]))
    lg("\n課題外の帯域の下端は slow 1.2 Hz、fast 1.5 Hz（最高の入力周波数 + 0.4 Hz）。"
       "カーソルの帯域パワーは [(mm/s)²]（x + y）。関節 SNR = sqrt(下端–4 Hz のパワー / 静止試行の同じ帯域のパワー)。"
       "課題外シェア = 静止のパワーを引いた残りの成分ごとの割合。\n")

    # 課題外シェア vs 課題の帯域のシェア（人ごとの差）
    lg("\n### 課題外の動きを担う成分は、課題の動きを担う成分と同じか（人ごとに 課題外シェア − 課題帯域のパワーシェア）\n")
    pre = list(csv.DictReader(open(os.path.join(out, a.prelim, "trials_prelim.csv"), encoding="utf-8")))
    cmp_ = []
    for b in bands:
        for dpi in dpis:
            for part in PARTS + ["distal"]:
                d = []
                for p in people:
                    tk = "v2_share_distal" if part == "distal" else f"v4_share_{part}"
                    task_sh = np.nanmean([fnum(r[tk]) for r in pre if r["participant"] == p and r["band"] == b
                                          and fnum(r["dpi"]) == dpi])
                    ex = [r[f"j_exshare_{part}"] for r in nim if r["participant"] == p and r["band"] == b and r["dpi"] == dpi][0]
                    d.append(ex - task_sh)
                cmp_.append({"band": b, "dpi": dpi, "part": PLAB.get(part, part), **one_sample(d, rng, a.n_boot)})
    for b in bands:
        idx = [i for i, r in enumerate(cmp_) if r["band"] == b]
        for i, q in zip(idx, bh([cmp_[i]["p"] for i in idx])):
            cmp_[i]["q"] = float(q)
    save(cmp_, os.path.join(pout, "nonin_vs_task_share.csv"))
    lg(md(cmp_, [("帯域", "band"), ("DPI", "dpi", "{:.0f}"), ("成分", "part"), ("差の平均", "mean", "{:+.3f}"),
                 ("CI 下", "lo", "{:+.3f}"), ("CI 上", "hi", "{:+.3f}"), ("dz", "dz", "{:+.2f}"),
                 ("+", "n_pos", "{:.0f}"), ("−", "n_neg", "{:.0f}"), ("p", "p", "{:.3f}"), ("q", "q", "{:.3f}")]))

    # DPI 効果・練度・個人差
    lg("\n### 課題外の動きの DPI 効果・練度・個人差\n")
    nt = []
    for b in bands:
        for k in ["cur_pow_lo_2", "cur_pow_2_4", "cur_pow_4_8", "cur_peak_hz", "cur_centroid_hz",
                  "j_exshare_distal", "j_exshare_hand", "j_exshare_upperarm"]:
            x0 = np.array([[r[k] for r in nim if r["participant"] == p and r["band"] == b and r["dpi"] == dpis[0]][0] for p in people])
            x1 = np.array([[r[k] for r in nim if r["participant"] == p and r["band"] == b and r["dpi"] == dpis[-1]][0] for p in people])
            d = (np.log2(x1) - np.log2(x0)) if k.startswith("cur_pow") else (x1 - x0)
            t = one_sample(d, rng, a.n_boot)
            sk = [i for i, p in enumerate(people) if p in SKILL]
            xm = ((x0 + x1) / 2)[sk]; ys = np.array([SKILL[people[i]] for i in sk], float)
            ok = np.isfinite(xm)
            tau = stats.kendalltau(xm[ok], ys[ok]) if ok.sum() >= 5 else None
            rho = stats.spearmanr(x0, x1) if np.isfinite(x0).all() and np.isfinite(x1).all() else None
            nt.append({"band": b, "k": k + (" [log2比]" if k.startswith("cur_pow") else ""), "d": t["mean"], "lo": t["lo"],
                       "hi": t["hi"], "dz": t["dz"], "n_pos": t["n_pos"], "n_neg": t["n_neg"], "p": t["p"],
                       "rho_400_800": float(rho.statistic) if rho else np.nan,
                       "tau_skill": float(tau.statistic) if tau else np.nan, "p_tau": float(tau.pvalue) if tau else np.nan})
    save(nt, os.path.join(pout, "nonin_tests.csv"))
    lg(md(nt, [("帯域", "band"), ("指標", "k"), ("d (800−400)", "d", "{:+.3f}"), ("CI 下", "lo", "{:+.3f}"), ("CI 上", "hi", "{:+.3f}"),
               ("dz", "dz", "{:+.2f}"), ("+", "n_pos", "{:.0f}"), ("−", "n_neg", "{:.0f}"), ("p", "p", "{:.3f}"),
               ("ρ(400,800)", "rho_400_800", "{:+.2f}"), ("τ 練度", "tau_skill", "{:+.2f}"), ("p(τ)", "p_tau", "{:.3f}")]))
    lg("\n（τ の p は Kendall の漸近 p。8 人で並べ替えではないので参考）\n")

    plots(prof, PM, people, bands, dpis, freqs, psd, NI, nim, link, pout)
    lg(f"\n→ {pout}")
    logf.close()


def _logx(a_):
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
    a_.set_xscale("log")
    a_.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1.0, 2, 5, 10]))
    a_.xaxis.set_major_formatter(FormatStrFormatter("%g")); a_.xaxis.set_minor_formatter(NullFormatter())


def plots(prof, PM, people, bands, dpis, freqs, psd, NI, nim, link, pout):
    plt = C.setup_plot()
    col = {dpis[0]: "#2a78d6", dpis[-1]: "#eb6834"}
    mk = {"x": "o", "y": "s"}

    def series(k, b, dpi, ax):
        r = sorted([x for x in prof if x["band"] == b and x["dpi"] == dpi and x["axis"] == ax], key=lambda x: x["f"])
        f = np.array([x["f"] for x in r]); m = np.array([x[k] for x in r])
        lo = np.array([x.get(k + "_lo", np.nan) for x in r]); hi = np.array([x.get(k + "_hi", np.nan) for x in r])
        return f, m, lo, hi

    # 図 1: 追従（gain, lag, rel_err）と計測（cam gain, cam lag）
    keys = [("gain_H", "ゲイン |カーソル/ターゲット|"), ("lag_H_ms", "遅れ [ms]"), ("rel_err", "相対誤差"),
            ("cam_gain", "カメラ MCP3 / マウス"), ("cam_lag_ms", "カメラの遅れ [ms]")]
    fig, axes = plt.subplots(len(bands), len(keys), figsize=(3.3 * len(keys), 3.2 * len(bands)), squeeze=False)
    for i, b in enumerate(bands):
        for j, (k, lab) in enumerate(keys):
            a_ = axes[i][j]
            for dpi in dpis:
                for ax in C.AX:
                    f, m, lo, hi = series(k, b, dpi, ax)
                    a_.errorbar(f, m, yerr=[m - lo, hi - m], marker=mk[ax], ms=4, lw=1, capsize=2, color=col[dpi],
                                ls="-" if ax == "x" else "--", label=f"{dpi:.0f} {ax}")
            _logx(a_); a_.set_title(f"{b}: {lab}", fontsize=8.5); a_.set_xlabel("周波数 [Hz]", fontsize=7.5)
            if k in ("gain_H", "cam_gain"):
                a_.axhline(1, color="#888", ls=":", lw=.8)
            if k.endswith("ms"):
                a_.axhline(0, color="#888", ls=":", lw=.8)
    axes[0][0].legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(pout, "fig1_tracking_by_freq.png"), dpi=130); plt.close(fig)

    # 図 2: 成分の射影シェア × 周波数
    comps = [("share_trunk", "体幹"), ("share_upperarm", "上腕"), ("share_forearm", "前腕【参考】"),
             ("share_hand", "手【参考】"), ("share_distal", "遠位（肘→MCP3）")]
    fig, axes = plt.subplots(len(bands) * 2, len(comps), figsize=(3.2 * len(comps), 2.9 * len(bands) * 2), squeeze=False)
    for i, b in enumerate(bands):
        for a_i, ax in enumerate(C.AX):
            for j, (k, lab) in enumerate(comps):
                a_ = axes[i * 2 + a_i][j]
                for dpi in dpis:
                    f, m, lo, hi = series(k, b, dpi, ax)
                    a_.errorbar(f, m, yerr=[m - lo, hi - m], marker=mk[ax], ms=4, lw=1, capsize=2, color=col[dpi],
                                label=f"{dpi:.0f}")
                a_.axhline(0, color="#888", ls=":", lw=.8)
                _logx(a_); a_.set_title(f"{b} {ax}: {lab}", fontsize=8.5)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(pout, "fig2_shares_by_freq.png"), dpi=130); plt.close(fig)

    # 図 3: 人ごとの遠位シェアと遅れのプロファイル（個人差の形）
    fig, axes = plt.subplots(len(bands), 4, figsize=(15, 3.3 * len(bands)), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for i, b in enumerate(bands):
        for j, (k, ax, lab) in enumerate([("share_distal", "x", "遠位シェア x"), ("share_distal", "y", "遠位シェア y"),
                                          ("lag_H_ms", "x", "遅れ x [ms]"), ("lag_H_ms", "y", "遅れ y [ms]")]):
            a_ = axes[i][j]
            for n, p in enumerate(people):
                r = sorted([x for x in PM if x["participant"] == p and x["band"] == b and x["axis"] == ax], key=lambda x: x["f"])
                fs = sorted({x["f"] for x in r})
                y = [np.nanmean([x[k] for x in r if x["f"] == f]) for f in fs]
                a_.plot(fs, y, "o-", ms=3, lw=1, color=cmap(n % 10), label=p)
            _logx(a_); a_.set_title(f"{b}: {lab}（DPI 平均）", fontsize=8.5)
    axes[0][-1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, .5))
    fig.tight_layout(); fig.savefig(os.path.join(pout, "fig3_person_profiles.png"), dpi=130); plt.close(fig)

    # 図 4: カーソル速度の PSD（入力周波数の外）
    fig, axes = plt.subplots(1, len(bands), figsize=(7 * len(bands), 4), squeeze=False)
    for a_, b in zip(axes[0], bands):
        for dpi in dpis:
            ks = [r["key"] for r in NI if r["band"] == b and r["dpi"] == dpi]
            f = psd[ks[0]][0]
            P = np.exp(np.mean([np.log(psd[k][1] + 1e-12) for k in ks], axis=0))
            a_.plot(f, P, color=col[dpi], lw=1.3, label=f"{dpi:.0f} DPI（幾何平均）")
        for ax in C.AX:
            for fq in freqs[(b, ax)]:
                a_.axvline(fq, color="#bbb", lw=.6)
        lo_ = [r["nonin_lo_hz"] for r in NI if r["band"] == b][0]
        a_.axvspan(lo_, 8, color="#f3e6d8", alpha=.5, label=f"課題外の帯域（{lo_:g}–8 Hz）")
        _logx(a_); a_.set_yscale("log"); a_.set_xlim(0.04, 20); a_.set_ylim(1e-2, 3e4)
        a_.set_xlabel("周波数 [Hz]"); a_.set_ylabel("カーソル速度の PSD [(mm/s)²/Hz]")
        a_.set_title(f"{b}: カーソル速度のスペクトル（灰色の線 = 入力周波数）", fontsize=9); a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(pout, "fig4_cursor_psd.png"), dpi=130); plt.close(fig)

    # 図 5: 周波数ごとの協調 ↔ 追従の r
    sel = [r for r in link if r["f"] != "all" and r["stratum"] == "pooled"]
    pairs = sorted({(r["coord"], r["clab"], r["perf"], r["plab"]) for r in sel})
    fig, axes = plt.subplots(len(bands), len(pairs), figsize=(2.6 * len(pairs), 3 * len(bands)), squeeze=False)
    for i, b in enumerate(bands):
        for j, (ck, cl, pk, pl) in enumerate(pairs):
            a_ = axes[i][j]
            for ax in C.AX:
                r = sorted([x for x in sel if x["band"] == b and x["axis"] == ax and x["coord"] == ck and x["perf"] == pk],
                           key=lambda x: x["f"])
                a_.plot([x["f"] for x in r], [x["r"] for x in r], marker=mk[ax], ms=4, lw=1, label=ax)
            a_.axhline(0, color="#888", ls=":", lw=.8); a_.set_ylim(-1, 1); _logx(a_)
            a_.set_title(f"{b}: {cl} ↔ {pl}", fontsize=7.5)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(pout, "fig5_link_by_freq.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
