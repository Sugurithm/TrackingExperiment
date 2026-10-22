"""corrections.py — 修正の動きはどの関節から来るか（C-1: ターゲットから外れた瞬間 / C-4: 2〜4 Hz の揺れ）

  python aim_eval/corrections.py --config analysis_out/20261022/config_used.py \
         --out analysis_out/20261022 [--prelim prelim_relaxed] [--n-boot 5000] [--seed 0]

前提: run.py --steps analyze（between/trials.csv、desk_frame.json）、prelim_inference.py（除外）、
freq_decomp.py（freq_decomp/per_bin.csv: 試行ごとの課題の帯域での分担とカメラ/マウスのゲイン）を先に回す。
既存コードは書き換えない。結果はすべて DPI の層（400 だけ / 800 だけ / 400+800）に分けて出す。

C-4  2〜4 Hz の修正の揺れはどの関節から来るか
     マウス（カメラと独立に測った手の動き）とのクロススペクトルで、各成分のうち**マウスと一緒に動いている部分**だけを取り出す。
       B_k = Σ_band CSD(マウス, 成分 k)、B_E = Σ_band CSD(マウス, 手先 MCP3) = Σ_k B_k
       シェア_k = Re(B_k · conj(B_E)) / |B_E|²（和は 1。負 = 手先と逆向き）
     カメラのノイズはマウスと相関しないので平均で消える（静止試行の床を引く方法より頑健）。
     帯域: 課題の帯域（0.2 Hz〜最高の入力周波数）、課題外の低い側（下端〜2 Hz）、2〜4 Hz。軸は x / y / x+y。
     比較のため、ノイズ込みのパワーの割合（素朴なシェア）と、手先とマウスのコヒーレンスも出す。

C-1  ターゲットから外れた瞬間の前後 1 秒
     出た瞬間 = on-target が 1 → 0（20 ms の中央値フィルタ後、直前に 0.2 s 以上ターゲット内、試行の端 1 s は除く）。
     u = 出た瞬間の誤差の向き（ターゲット → カーソル）。速度をすべて u に投影する（+ = 外へ、− = 戻る）。
       誤差の速度（手の座標）  e_hand = マウス·u − ターゲット·u / k     （k = カーソル mm / 手 mm、試行ごとの回帰。カメラ不要）
       成分の速度              P_k·u（カメラ。試行ごとのカメラの遅れ = a0 の mouse_lag_ms だけ前にずらす）
       成分の「役割からのずれ」 P_k·u − s_k(u) · g(u) · ターゲット·u / k
           s_k(u) = その試行の課題の帯域での成分 k のシェア（x・y を u² で重み付け、freq_decomp/per_bin.csv）
           g(u)   = その試行のカメラ MCP3 / マウス のゲイン（同上）
         → 完全に追従していたらその関節が出していたはずの速度からのずれ。全成分の和 ≈ g · e_hand
     窓: 基準 [−1.0, −0.6] s、出る前 [−0.3, 0] s、出た後 [0.1, 0.5] s。
       外へ押した分担 = 出る前の (ずれ − 基準) の成分ごとの割合、戻した分担 = 出た後の同じ割合
       戻しのピーク時刻 = 出た後 0〜0.8 s でずれが最も負になる時刻（成分ごと）
     出方の種類: 置いていかれ（ターゲットが u と逆向きに動いていた）/ 行き過ぎ（ターゲットは u 向き、カーソルがそれより速い）
     誤差の向きが主に x か y か でも分ける（分担が方向で違うため）。
手首・MCP3 は MoCap 未照合なので、前腕・手は【参考】。遠位（前腕 + 手）・近位（体幹 + 上腕）は手首の点を使わない。
"""
import argparse, csv, importlib.util, json, os, sys, time, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common as C                                      # noqa: E402
from scipy import stats                                 # noqa: E402
from scipy.signal import csd, welch, medfilt            # noqa: E402

PARTS = ["trunk", "upperarm", "forearm", "hand"]
GROUPS = PARTS + ["proximal", "distal"]
PLAB = {"trunk": "体幹", "upperarm": "上腕", "forearm": "前腕【参考】", "hand": "手【参考】",
        "proximal": "近位（体幹+上腕）", "distal": "遠位（肘→MCP3）", "end": "手先 MCP3", "mouse": "マウス（手）"}
PCOL = {"trunk": "#7a5ea8", "upperarm": "#2a78d6", "forearm": "#eb6834", "hand": "#1baf7a",
        "proximal": "#4b5fb0", "distal": "#d0573a"}
SKILL = {"Syou": 3, "Yuya": 3, "Suguru": 3, "kakeru": 2, "xin": 2, "Kosuke": 1, "Yuuki": 1, "Suzuka": 1}
FSG = C.FS_U
WELCH_S = 4.0
WIN_S = 1.0
W_BASE, W_PRE, W_POST, W_LAT = (-1.0, -0.6), (-0.3, 0.0), (0.1, 0.5), (0.0, 0.8)
MIN_ON_S = 0.2


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


def filled(X):
    X = np.asarray(X, float).copy()
    for j in range(X.shape[1]):
        X[:, j], _ = C.nanfill_grid(X[:, j])
    return X


def nonin_lo(tr):
    fmax = max(float(np.max(tr.freqs(ax)[1])) for ax in C.AX)
    return max(1.2, round((fmax + 0.4) * 4) / 4), fmax


# ------------------------------------------------------------------ C-4
def c4_trial(tr, meta):
    parts = C.contributions(tr, end="mcp3")
    if parts is None:
        return []
    P = {p: np.gradient(filled(parts[p]), axis=0) * FSG for p in PARTS}
    P["proximal"] = P["trunk"] + P["upperarm"]; P["distal"] = P["forearm"] + P["hand"]
    E = np.gradient(filled(parts["_total"]), axis=0) * FSG
    M = np.gradient(filled(tr.task_grid("mouse")), axis=0) * FSG
    lo, fmax = nonin_lo(tr)
    bands = {"task": (0.2, fmax + 0.05), "lo_2": (lo, 2.0), "2_4": (2.0, 4.0)}
    nper = int(WELCH_S * FSG)
    kw = dict(fs=FSG, nperseg=nper, noverlap=nper // 2, detrend="constant")
    spec = {}
    for k in range(2):
        f, Smm = welch(M[:, k], **kw)
        _, See = welch(E[:, k], **kw)
        _, Bem = csd(M[:, k], E[:, k], **kw)
        spec[k] = {"Smm": Smm, "See": See, "BE": Bem,
                   "B": {p: csd(M[:, k], P[p][:, k], **kw)[1] for p in GROUPS},
                   "Spp": {p: welch(P[p][:, k], **kw)[1] for p in GROUPS}}
    rows = []
    for bname, (b0, b1) in bands.items():
        m = (f >= b0) & (f < b1)
        for axn, ks in (("x", [0]), ("y", [1]), ("xy", [0, 1])):
            BE = sum(spec[k]["BE"][m].sum() for k in ks)
            Smm = sum(spec[k]["Smm"][m].sum() for k in ks); See = sum(spec[k]["See"][m].sum() for k in ks)
            r = {**meta, "band_f": bname, "axis": axn, "f_lo": b0, "f_hi": b1,
                 "coh": float(np.abs(BE) ** 2 / (See * Smm)) if See > 0 and Smm > 0 else np.nan,
                 "coh_gain": float(np.abs(BE) / Smm) if Smm > 0 else np.nan,
                 "mouse_pow": float(Smm), "end_pow": float(See)}
            naive_tot = sum(sum(spec[k]["Spp"][p][m].sum() for k in ks) for p in PARTS)
            for p in GROUPS:
                Bp = sum(spec[k]["B"][p][m].sum() for k in ks)
                r[f"share_{p}"] = float(np.real(Bp * np.conj(BE)) / np.abs(BE) ** 2) if np.abs(BE) > 0 else np.nan
                if p in PARTS:
                    r[f"naive_{p}"] = float(sum(spec[k]["Spp"][p][m].sum() for k in ks) / naive_tot)
            rows.append(r)
    return rows


# ------------------------------------------------------------------ C-1
def exits(tr):
    g = np.isfinite(tr.task_t) & (tr.task_t >= 0) & (tr.task_t < tr.T)
    t = tr.task_t[g]; on = medfilt(tr.on_target[g].astype(float), 5) > 0.5
    d = np.diff(on.astype(int))
    ev = []
    for i in np.flatnonzero(d == -1):
        j = i
        while j > 0 and on[j - 1]:
            j -= 1
        if t[i + 1] - t[j] < MIN_ON_S:
            continue
        t0 = t[i + 1]
        if WIN_S <= t0 < tr.T - WIN_S:
            ev.append(t0)
    return ev


def c1_trial(tr, meta, roles, lag_ms):
    parts = C.contributions(tr, end="mcp3")
    if parts is None:
        return []
    tg = tr.grid_t
    sh = (lag_ms / 1000.0) if np.isfinite(lag_ms) else 0.0
    P = {}
    for p in PARTS + ["_total"]:
        X = filled(parts[p])
        X = np.column_stack([np.interp(tg + sh, tg, X[:, j]) for j in range(2)])   # カメラの遅れを補正
        P[p] = np.gradient(X, axis=0) * FSG
    P["proximal"] = P["trunk"] + P["upperarm"]; P["distal"] = P["forearm"] + P["hand"]
    E = P.pop("_total")
    tgt, cur = filled(tr.task_grid("target")), filled(tr.task_grid("cursor"))
    mo = filled(tr.task_grid("mouse"))
    vt, vc, vm = (np.gradient(x, axis=0) * FSG for x in (tgt, cur, mo))
    e = cur - tgt
    k = float(np.sum(vc * vm) / np.sum(vm * vm))                     # カーソル mm / 手 mm
    W = int(round(WIN_S * FSG)); lags = np.arange(-W, W + 1) / FSG
    out = []
    for t0 in exits(tr):
        i0 = int(round(t0 * FSG))
        if i0 - W < 0 or i0 + W >= len(tg):
            continue
        u = e[i0] / max(np.linalg.norm(e[i0]), 1e-9)
        sl = slice(i0 - W, i0 + W + 1)
        tv = vt[sl] @ u / k                                          # ターゲットの u 向きの速度（手の mm/s）
        w2 = u ** 2
        g = w2[0] * roles["gain"][0] + w2[1] * roles["gain"][1]
        r = {"t0": t0, "u": u, "type": "lag" if (vt[i0] @ u) < 0 else "over",
             "dom": "x" if abs(u[0]) >= abs(u[1]) else "y",
             "err": np.linalg.norm(e[sl], axis=1), "e_hand": vm[sl] @ u - tv, "tgt": tv, "mouse": vm[sl] @ u,
             "end": E[sl] @ u, "end_dev": E[sl] @ u - g * tv}
        for p in GROUPS:
            s = w2[0] * roles[p][0] + w2[1] * roles[p][1]
            r[p] = P[p][sl] @ u
            r[p + "_dev"] = r[p] - s * g * tv
        out.append(r)
    return out, lags


def window(x, lags, w):
    m = (lags >= w[0]) & (lags <= w[1])
    return float(np.nanmean(x[m]))


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


def boot_ci(v, rng, n_boot):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan
    i = rng.integers(0, len(v), (n_boot, len(v)))
    m = v[i].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def test(d, rng, n_boot):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) < 4:
        return {"n": len(d), "mean": np.nan, "lo": np.nan, "hi": np.nan, "dz": np.nan, "n_pos": 0, "n_neg": 0, "p": np.nan}
    lo, hi = boot_ci(d, rng, n_boot)
    try:
        p = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p = np.nan
    return {"n": int(len(d)), "mean": float(d.mean()), "lo": lo, "hi": hi,
            "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan,
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "p": p}


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
            w.writerow({k: v for k, v in r.items() if not isinstance(v, np.ndarray)})


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
    pout = os.path.join(out, "corrections"); os.makedirs(pout, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    logf = open(os.path.join(pout, "corrections_log.md"), "w", encoding="utf-8")

    def lg(s=""):
        print(s); logf.write(s + "\n"); logf.flush()

    lg(f"# corrections（{time.strftime('%Y-%m-%d %H:%M:%S')}）\n")
    lg(f"config: {a.config} / out: {out} / prelim: {a.prelim} / n_boot {a.n_boot} / seed {a.seed}\n")

    keep = {r["key"] for r in csv.DictReader(open(os.path.join(out, a.prelim, "trials_prelim.csv"), encoding="utf-8"))}
    lagmap = {}
    for r in csv.DictReader(open(os.path.join(out, "between", "trials.csv"), encoding="utf-8")):
        if r.get("kind") == "track":
            lagmap[f"{r['participant']}_{fnum(r['dpi']):g}_{r['band']}_s{int(fnum(r['seed']))}"] = fnum(r.get("mouse_lag_ms"))
    roles_raw = {}
    for r in csv.DictReader(open(os.path.join(out, "freq_decomp", "per_bin.csv"), encoding="utf-8")):
        roles_raw.setdefault((r["key"], r["axis"]), []).append(r)
    roles = {}
    for (key, ax), rr in roles_raw.items():
        j = 0 if ax == "x" else 1
        d = roles.setdefault(key, {p: [np.nan, np.nan] for p in GROUPS + ["gain"]})
        for p in GROUPS:
            d[p][j] = float(np.nanmean([fnum(x[f"share_{p}"]) for x in rr]))
        d["gain"][j] = float(np.nanmean([fnum(x["cam_gain"]) for x in rr]))

    trials = []
    for e in cfg.SESSIONS:
        for t, meta in C.list_trials(resolve(e["dir"]), kinds=("track",)):
            if cfg.BANDS and meta["task"].get("band") not in cfg.BANDS:
                continue
            key = f"{e['participant']}_{float(e['dpi']):g}_{meta['task']['band']}_s{int(meta['task']['seed'])}"
            if key in keep and key in roles:
                trials.append((key, t, e))
    lg(f"試行: {len(trials)} 本\n")

    C4, EV = [], []
    tmp = os.path.join(out, "_corr_tmp")
    t0 = time.time()
    for key, d, e in trials:
        name = e["participant"]
        axes = os.path.join(out, name, "desk_frame.json")
        tr = C.Trial(Args(d, axes=axes if os.path.exists(axes) else None, out=tmp))
        meta = {"key": key, "participant": name, "dpi": float(e["dpi"]), "band": tr.band, "seed": int(tr.seed),
                "skill": SKILL.get(name, np.nan)}
        C4 += c4_trial(tr, meta)
        evs, lags = c1_trial(tr, meta, roles[key], lagmap.get(key, np.nan))
        for r in evs:
            EV.append({**meta, **r})
    shutil.rmtree(tmp, ignore_errors=True)
    lg(f"計算 {time.time() - t0:.0f} s。C-4: {len(C4)} 行、C-1: 出た瞬間 {len(EV)} 回\n")
    save(C4, os.path.join(pout, "c4_trials.csv"))
    save([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} | {"u_x": r["u"][0], "u_y": r["u"][1]} for r in EV],
         os.path.join(pout, "c1_events.csv"))

    people = sorted({r["participant"] for r in C4}, key=str.lower)
    bands = [b for b in ("slow", "fast") if any(r["band"] == b for r in C4)]
    dpis = sorted({r["dpi"] for r in C4})
    strata = [(f"{d:.0f}", [d]) for d in dpis] + [("pooled", list(dpis))]

    # ================================================================== C-4
    lg("# C-4 2〜4 Hz の修正の揺れはどの関節から来るか\n")
    lg("シェア = マウスと一緒に動く部分（クロススペクトル）の成分ごとの割合（和 1、負 = 手先と逆向き）。"
       "素朴なシェア = ノイズ込みのパワーの割合。coh = 手先（カメラ）とマウスのコヒーレンス（1 = カメラの動きがすべて本物）。"
       "人ごとに seed（と層によっては DPI）を平均してから 9 人で集計。\n")

    def c4_person(p, b, sd, bf, ax, k):
        v = [r[k] for r in C4 if r["participant"] == p and r["band"] == b and r["dpi"] in sd
             and r["band_f"] == bf and r["axis"] == ax]
        return float(np.nanmean(v)) if v else np.nan

    c4g = []
    for sname, sd in strata:
        for b in bands:
            for ax in ("xy", "x", "y"):
                for bf in ("task", "lo_2", "2_4"):
                    row = {"stratum": sname, "band": b, "axis": ax, "band_f": bf}
                    for k in ["coh", "coh_gain"] + [f"share_{p}" for p in GROUPS] + [f"naive_{p}" for p in PARTS]:
                        v = [c4_person(p, b, sd, bf, ax, k) for p in people]
                        row[k] = float(np.nanmean(v)); row[k + "_lo"], row[k + "_hi"] = boot_ci(v, rng, a.n_boot)
                    c4g.append(row)
    save(c4g, os.path.join(pout, "c4_group.csv"))
    lg("## C-4.1 群の平均（9 人）\n")
    lg(md([r for r in c4g], [("層", "stratum"), ("帯域", "band"), ("軸", "axis"), ("周波数帯", "band_f"),
                             ("coh", "coh", "{:.2f}"), ("体幹", "share_trunk", "{:+.2f}"), ("上腕", "share_upperarm", "{:+.2f}"),
                             ("前腕", "share_forearm", "{:+.2f}"), ("手", "share_hand", "{:+.2f}"),
                             ("近位", "share_proximal", "{:+.2f}"), ("遠位", "share_distal", "{:+.2f}"),
                             ("素朴 体幹", "naive_trunk", "{:.2f}"), ("素朴 上腕", "naive_upperarm", "{:.2f}"),
                             ("素朴 前腕", "naive_forearm", "{:.2f}"), ("素朴 手", "naive_hand", "{:.2f}")]))

    lg("\n## C-4.2 2〜4 Hz の分担は課題の帯域の分担と違うか（人ごとの差 = 2〜4 Hz − 課題の帯域）\n")
    c4t = []
    for sname, sd in strata:
        for b in bands:
            for ax in ("xy", "x", "y"):
                for p in GROUPS:
                    d = [c4_person(q, b, sd, "2_4", ax, f"share_{p}") - c4_person(q, b, sd, "task", ax, f"share_{p}")
                         for q in people]
                    c4t.append({"stratum": sname, "band": b, "axis": ax, "part": PLAB[p], **test(d, rng, a.n_boot)})
    for key in sorted({(r["stratum"], r["band"]) for r in c4t}):
        idx = [i for i, r in enumerate(c4t) if (r["stratum"], r["band"]) == key]
        for i, q in zip(idx, bh([c4t[i]["p"] for i in idx])):
            c4t[i]["q"] = float(q)
    save(c4t, os.path.join(pout, "c4_vs_task.csv"))
    lg(md(c4t, [("層", "stratum"), ("帯域", "band"), ("軸", "axis"), ("成分", "part"), ("差", "mean", "{:+.3f}"),
                ("CI 下", "lo", "{:+.3f}"), ("CI 上", "hi", "{:+.3f}"), ("dz", "dz", "{:+.2f}"),
                ("+", "n_pos", "{:.0f}"), ("−", "n_neg", "{:.0f}"), ("p", "p", "{:.3f}"), ("q", "q", "{:.3f}")]))

    lg("\n## C-4.3 DPI の効果（2〜4 Hz のシェア、800 − 400）\n")
    c4d = []
    for b in bands:
        for ax in ("xy", "x", "y"):
            for k in ["coh"] + [f"share_{p}" for p in GROUPS]:
                d = [c4_person(q, b, [dpis[-1]], "2_4", ax, k) - c4_person(q, b, [dpis[0]], "2_4", ax, k) for q in people]
                c4d.append({"band": b, "axis": ax, "k": k, **test(d, rng, a.n_boot)})
    for b in bands:
        idx = [i for i, r in enumerate(c4d) if r["band"] == b]
        for i, q in zip(idx, bh([c4d[i]["p"] for i in idx])):
            c4d[i]["q"] = float(q)
    save(c4d, os.path.join(pout, "c4_dpi.csv"))
    lg(md(c4d, [("帯域", "band"), ("軸", "axis"), ("指標", "k"), ("d", "mean", "{:+.3f}"), ("CI 下", "lo", "{:+.3f}"),
                ("CI 上", "hi", "{:+.3f}"), ("dz", "dz", "{:+.2f}"), ("+", "n_pos", "{:.0f}"), ("−", "n_neg", "{:.0f}"),
                ("p", "p", "{:.3f}"), ("q", "q", "{:.3f}")]))

    # ================================================================== C-1
    lg("\n# C-1 ターゲットから外れた瞬間の前後 1 秒\n")
    nev = []
    for sname, sd in strata:
        for b in bands:
            rr = [r for r in EV if r["band"] == b and r["dpi"] in sd]
            nev.append({"stratum": sname, "band": b, "n": len(rr),
                        "per_person_min": min(sum(1 for r in rr if r["participant"] == p) for p in people),
                        "lag_frac": float(np.mean([r["type"] == "lag" for r in rr])) if rr else np.nan,
                        "x_frac": float(np.mean([r["dom"] == "x" for r in rr])) if rr else np.nan})
    lg(md(nev, [("層", "stratum"), ("帯域", "band"), ("出た回数", "n", "{:.0f}"), ("1 人の最少", "per_person_min", "{:.0f}"),
                ("置いていかれの割合", "lag_frac", "{:.2f}"), ("誤差が主に x の割合", "x_frac", "{:.2f}")]))

    SIG = ["err", "e_hand", "tgt", "mouse", "end", "end_dev"] + GROUPS + [p + "_dev" for p in GROUPS]
    curves = {}
    for p in people:
        for b in bands:
            for dpi in dpis:
                for typ in ("all", "lag", "over"):
                    for dom in ("all", "x", "y"):
                        rr = [r for r in EV if r["participant"] == p and r["band"] == b and r["dpi"] == dpi
                              and (typ == "all" or r["type"] == typ) and (dom == "all" or r["dom"] == dom)]
                        if len(rr) >= 3:
                            curves[(p, b, dpi, typ, dom)] = ({s: np.mean([r[s] for r in rr], axis=0) for s in SIG}, len(rr))

    def pcurve(p, b, sd, typ, dom, s):
        cs = [curves[(p, b, d, typ, dom)][0][s] for d in sd if (p, b, d, typ, dom) in curves]
        return np.mean(cs, axis=0) if cs else None

    # 窓の値（人ごと）
    PW = []
    for sname, sd in strata:
        for b in bands:
            for typ in ("all", "lag", "over"):
                for dom in ("all", "x", "y"):
                    for p in people:
                        c = {s: pcurve(p, b, sd, typ, dom, s) for s in SIG}
                        if c["end"] is None:
                            continue
                        row = {"stratum": sname, "band": b, "type": typ, "dom": dom, "participant": p}
                        for s in SIG:
                            base = window(c[s], lags, W_BASE)
                            row[f"{s}_pre"] = window(c[s], lags, W_PRE) - base
                            row[f"{s}_post"] = window(c[s], lags, W_POST) - base
                            seg = (lags >= W_LAT[0]) & (lags <= W_LAT[1])
                            row[f"{s}_tmin"] = float(lags[seg][np.argmin(c[s][seg])])
                        for ph in ("pre", "post"):
                            tot = sum(row[f"{q}_dev_{ph}"] for q in PARTS)
                            for q in GROUPS:
                                row[f"sh_{q}_{ph}"] = row[f"{q}_dev_{ph}"] / tot if abs(tot) > 1e-9 else np.nan
                        # 課題の帯域の分担（比べる基準）: その人・条件の試行の平均、x・y を出方の向きで重み付け
                        ev = [r for r in EV if r["participant"] == p and r["band"] == b and r["dpi"] in sd
                              and (typ == "all" or r["type"] == typ) and (dom == "all" or r["dom"] == dom)]
                        for q in GROUPS:
                            row[f"task_{q}"] = float(np.mean([r["u"][0] ** 2 * roles[r["key"]][q][0]
                                                              + r["u"][1] ** 2 * roles[r["key"]][q][1] for r in ev]))
                        PW.append(row)
    save(PW, os.path.join(pout, "c1_person_windows.csv"))

    def pw(sname, b, typ, dom, k):
        return [r[k] for r in PW if r["stratum"] == sname and r["band"] == b and r["type"] == typ and r["dom"] == dom]

    lg("\n## C-1.1 誤差と手の動き（カメラ不要: マウスとターゲットから）\n")
    lg("値は基準 [−1, −0.6] s からの差。e_hand = 手の座標での誤差の速度（+ = 外へ）。tmin = 出た後 0〜0.8 s で最も戻る向きに速い時刻。\n")
    t1 = []
    for sname, _ in strata:
        for b in bands:
            for typ in ("all", "lag", "over"):
                row = {"stratum": sname, "band": b, "type": typ}
                for k in ("e_hand_pre", "e_hand_post", "e_hand_tmin", "mouse_pre", "mouse_post", "tgt_pre", "tgt_post"):
                    v = pw(sname, b, typ, "all", k); row[k] = float(np.nanmean(v)); row["n"] = len(v)
                t1.append(row)
    lg(md(t1, [("層", "stratum"), ("帯域", "band"), ("出方", "type"), ("人", "n", "{:.0f}"),
               ("誤差の速度 出る前 [mm/s]", "e_hand_pre", "{:+.1f}"), ("出た後", "e_hand_post", "{:+.1f}"),
               ("戻しのピーク [s]", "e_hand_tmin", "{:.2f}"), ("マウス 前", "mouse_pre", "{:+.1f}"), ("後", "mouse_post", "{:+.1f}"),
               ("ターゲット 前", "tgt_pre", "{:+.1f}"), ("後", "tgt_post", "{:+.1f}")]))

    lg("\n## C-1.2 外へ押した分担・戻した分担（役割からのずれの成分ごとの割合）と、課題の帯域の分担との差\n")
    lg("sh_pre = 出る前 [−0.3, 0] に外へ押したずれの割合、sh_post = 出た後 [0.1, 0.5] に戻したずれの割合（和 1）。"
       "task = 同じ出方の向きでの課題の帯域の分担。差 = 分担 − task を人ごとに出して 9 人で検定。\n")
    c1s = []
    for sname, _ in strata:
        for b in bands:
            for typ in ("all", "lag", "over"):
                for dom in ("all", "x", "y"):
                    for q in GROUPS:
                        task = pw(sname, b, typ, dom, f"task_{q}")
                        for ph in ("pre", "post"):
                            v = pw(sname, b, typ, dom, f"sh_{q}_{ph}")
                            t = test(np.array(v) - np.array(task), rng, a.n_boot)
                            c1s.append({"stratum": sname, "band": b, "type": typ, "dom": dom, "part": PLAB[q], "phase": ph,
                                        "share": float(np.nanmean(v)), "task": float(np.nanmean(task)),
                                        "diff": t["mean"], "lo": t["lo"], "hi": t["hi"], "dz": t["dz"],
                                        "n_pos": t["n_pos"], "n_neg": t["n_neg"], "n": t["n"], "p": t["p"]})
    for key in sorted({(r["stratum"], r["band"]) for r in c1s}):
        idx = [i for i, r in enumerate(c1s) if (r["stratum"], r["band"]) == key]
        for i, q in zip(idx, bh([c1s[i]["p"] for i in idx])):
            c1s[i]["q"] = float(q)
    save(c1s, os.path.join(pout, "c1_shares.csv"))
    lg(md([r for r in c1s if r["type"] == "all"],
          [("層", "stratum"), ("帯域", "band"), ("誤差の向き", "dom"), ("成分", "part"), ("窓", "phase"),
           ("分担", "share", "{:+.2f}"), ("課題の帯域", "task", "{:+.2f}"), ("差", "diff", "{:+.3f}"),
           ("CI 下", "lo", "{:+.3f}"), ("CI 上", "hi", "{:+.3f}"), ("dz", "dz", "{:+.2f}"),
           ("+", "n_pos", "{:.0f}"), ("−", "n_neg", "{:.0f}"), ("p", "p", "{:.3f}"), ("q", "q", "{:.3f}")]))
    lg("\n（出方別 = 置いていかれ / 行き過ぎ の表は c1_shares.csv）\n")

    lg("\n## C-1.3 どの関節が先に戻すか（戻しのピーク時刻の差、人ごと）\n")
    lg("+ = 前に書いた成分のほうが遅い。カメラの成分どうしは同じ遅れの補正なので、差には計測の遅れが入らない。"
       "手先 − マウス は補正の残り（0 に近いほど遅れの補正が合っている）。\n")
    PAIRS = [("upperarm_dev", "forearm_dev"), ("upperarm_dev", "hand_dev"), ("proximal_dev", "distal_dev"),
             ("trunk_dev", "upperarm_dev"), ("forearm_dev", "hand_dev"), ("end", "mouse")]
    c1l = []
    for sname, _ in strata:
        for b in bands:
            for dom in ("all", "x", "y"):
                for s1, s2 in PAIRS:
                    d = np.array(pw(sname, b, "all", dom, f"{s1}_tmin")) - np.array(pw(sname, b, "all", dom, f"{s2}_tmin"))
                    c1l.append({"stratum": sname, "band": b, "dom": dom, "pair": f"{s1} − {s2}",
                                "t1": float(np.nanmean(pw(sname, b, "all", dom, f"{s1}_tmin"))),
                                "t2": float(np.nanmean(pw(sname, b, "all", dom, f"{s2}_tmin"))),
                                **{k: v for k, v in test(d * 1000, rng, a.n_boot).items()}})
    for key in sorted({(r["stratum"], r["band"]) for r in c1l}):
        idx = [i for i, r in enumerate(c1l) if (r["stratum"], r["band"]) == key]
        for i, q in zip(idx, bh([c1l[i]["p"] for i in idx])):
            c1l[i]["q"] = float(q)
    save(c1l, os.path.join(pout, "c1_latency.csv"))
    lg(md(c1l, [("層", "stratum"), ("帯域", "band"), ("誤差の向き", "dom"), ("組", "pair"),
                ("時刻 1 [s]", "t1", "{:.2f}"), ("時刻 2 [s]", "t2", "{:.2f}"), ("差 [ms]", "mean", "{:+.0f}"),
                ("CI 下", "lo", "{:+.0f}"), ("CI 上", "hi", "{:+.0f}"), ("+", "n_pos", "{:.0f}"), ("−", "n_neg", "{:.0f}"),
                ("p", "p", "{:.3f}"), ("q", "q", "{:.3f}")]))

    plots(curves, lags, people, bands, dpis, strata, c4g, c1s, pcurve, pout)
    json.dump({"c4_group": c4g, "c4_vs_task": c4t, "c4_dpi": c4d, "c1_shares": c1s, "c1_latency": c1l},
              open(os.path.join(pout, "corrections.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=C._jsonable)
    lg(f"\n→ {pout}")
    logf.close()


def plots(curves, lags, people, bands, dpis, strata, c4g, c1s, pcurve, pout):
    plt = C.setup_plot()
    # C-1 図: 層（行）× 信号（列）。帯域ごとに 1 枚
    for b in bands:
        for dom in ("all", "x", "y"):
            fig, axes = plt.subplots(len(strata), 4, figsize=(17, 3.4 * len(strata)), squeeze=False)
            for i, (sname, sd) in enumerate(strata):
                def grp(s):
                    cs = [pcurve(p, b, sd, "all", dom, s) for p in people]
                    cs = np.array([c for c in cs if c is not None])
                    return cs.mean(0), cs.std(0, ddof=1) / np.sqrt(len(cs)), len(cs)
                ax = axes[i][0]
                m, se, n = grp("err"); ax.plot(lags, m, color="k"); ax.fill_between(lags, m - se, m + se, color="k", alpha=.15)
                ax.axhline(8, color="#c55", ls=":", lw=.8); ax.set_ylabel("誤差 |e| [mm]（画面）")
                ax.set_title(f"{b} / {sname} / 誤差の向き {dom}: 誤差（点線 = ターゲット半径）", fontsize=8.5)
                ax = axes[i][1]
                for s, c_, lab in (("e_hand", "k", "誤差の速度（手）"), ("mouse", "#888", "マウス"), ("tgt", "#c33", "ターゲット")):
                    m, se, _ = grp(s); ax.plot(lags, m, color=c_, label=lab); ax.fill_between(lags, m - se, m + se, color=c_, alpha=.12)
                ax.set_ylabel("u 向きの速度 [mm/s]（手）"); ax.set_title("カメラ不要（マウス・ターゲット）", fontsize=8.5)
                ax.legend(fontsize=6.5)
                ax = axes[i][2]
                for q in PARTS:
                    m, se, _ = grp(q); ax.plot(lags, m, color=PCOL[q], label=PLAB[q]); ax.fill_between(lags, m - se, m + se, color=PCOL[q], alpha=.12)
                ax.set_title("各成分の u 向きの速度（そのまま）", fontsize=8.5); ax.legend(fontsize=6.5)
                ax = axes[i][3]
                for q in PARTS:
                    m, se, _ = grp(q + "_dev"); ax.plot(lags, m, color=PCOL[q], label=PLAB[q]); ax.fill_between(lags, m - se, m + se, color=PCOL[q], alpha=.12)
                m, se, _ = grp("end_dev"); ax.plot(lags, m, color="k", lw=1.6, label="手先（和）")
                ax.set_title(f"役割からのずれ（+ 外へ押す / − 戻す）{n} 人", fontsize=8.5); ax.legend(fontsize=6.5)
                for ax in axes[i]:
                    ax.axvline(0, color="#888", lw=.8)
                    for w, c_ in ((W_PRE, "#f3d9d9"), (W_POST, "#d9e8f3")):
                        ax.axvspan(*w, color=c_, alpha=.4, lw=0)
                    if ax is not axes[i][0]:
                        ax.axhline(0, color="#888", ls=":", lw=.8)
                    ax.set_xlabel("出た瞬間からの時刻 [s]", fontsize=7.5)
            fig.tight_layout(); fig.savefig(os.path.join(pout, f"fig_c1_{b}_{dom}.png"), dpi=120); plt.close(fig)

    # C-1 分担の棒グラフ
    fig, axes = plt.subplots(len(bands), len(strata), figsize=(5.2 * len(strata), 3.8 * len(bands)), squeeze=False)
    for i, b in enumerate(bands):
        for j, (sname, _) in enumerate(strata):
            ax = axes[i][j]; x = np.arange(3)
            for k, q in enumerate(PARTS):
                vals = []
                for ph in ("task", "pre", "post"):
                    r = next(r for r in c1s if r["stratum"] == sname and r["band"] == b and r["type"] == "all"
                             and r["dom"] == "all" and r["part"] == PLAB[q] and r["phase"] == ("pre" if ph == "task" else ph))
                    vals.append(r["task"] if ph == "task" else r["share"])
                ax.bar(x + (k - 1.5) * .2, vals, width=.2, color=PCOL[q], label=PLAB[q])
            ax.axhline(0, color="#888", lw=.8)
            ax.set_xticks(x); ax.set_xticklabels(["課題の帯域", "外へ押す（前）", "戻す（後）"], fontsize=8)
            ax.set_title(f"{b} / {sname}", fontsize=9)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(pout, "fig_c1_shares.png"), dpi=130); plt.close(fig)

    # C-4 図: 帯域ごとのシェア（コヒーレント vs 素朴）
    fig, axes = plt.subplots(len(bands) * 3, len(strata), figsize=(5.2 * len(strata), 3.0 * len(bands) * 3), squeeze=False)
    for i, b in enumerate(bands):
        for a_i, ax_ in enumerate(("xy", "x", "y")):
            for j, (sname, _) in enumerate(strata):
                ax = axes[i * 3 + a_i][j]
                cats = [("task", "share"), ("lo_2", "share"), ("2_4", "share"), ("2_4", "naive")]
                for k, q in enumerate(PARTS):
                    v, lo, hi = [], [], []
                    for bf, kind in cats:
                        r = next(r for r in c4g if r["stratum"] == sname and r["band"] == b and r["axis"] == ax_ and r["band_f"] == bf)
                        v.append(r[f"{kind}_{q}"]); lo.append(r[f"{kind}_{q}_lo"]); hi.append(r[f"{kind}_{q}_hi"])
                    v, lo, hi = map(np.array, (v, lo, hi))
                    ax.bar(np.arange(4) + (k - 1.5) * .2, v, width=.2, color=PCOL[q], yerr=[v - lo, hi - v], capsize=2,
                           label=PLAB[q], error_kw={"lw": .7})
                cohs = [next(r for r in c4g if r["stratum"] == sname and r["band"] == b and r["axis"] == ax_ and r["band_f"] == bf)["coh"]
                        for bf, _ in cats[:3]]
                ax.set_xticks(range(4))
                ax.set_xticklabels([f"課題\ncoh {cohs[0]:.2f}", f"下端–2 Hz\ncoh {cohs[1]:.2f}", f"2–4 Hz\ncoh {cohs[2]:.2f}",
                                    "2–4 Hz\n（素朴）"], fontsize=7)
                ax.axhline(0, color="#888", lw=.8)
                ax.set_title(f"{b} / {sname} / 軸 {ax_}", fontsize=9)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(pout, "fig_c4_shares.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
