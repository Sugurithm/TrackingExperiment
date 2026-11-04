#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
perf_link.py — 成績（task.csv）と動き方（freq_phase.py の出力）の関係

  python temp/freq_phase.py --compare slow_all fps_vs_non      # 先に実行（trial_freq.csv を作る）
  python temp/perf_link.py --compare slow_all fps_vs_non

1. 試行ごとの成績を task.csv から計算:  rms_mm（ターゲットとカーソルの距離の RMS），on_pct（on_target の割合），
   rms_lagfree_mm（カーソルを最適な時間だけずらした RMS = 遅れを除いた形の誤差）
2. 成績の個人差が試行をまたいで再現するか（ICC）．再現しなければ，動き方との関係を見る意味が薄い．
3. 動き方の特徴（試行ごと）: x 方向の F・H の寄与率，y 方向の U・S・F の寄与率（周波数平均），
   マウス追従のゲイン・遅れ（高周波側 / 全体）
   (a) 人の間: 人ごとの平均どうしの Spearman（n = 人数，並べ替え検定）
   (b) 人の中: 人×帯域の平均を引いた値どうしの Spearman（n = 試行数，人の中で並べ替え）
       「同じ人が上手くいった試行では，動き方が違うか」
出力: <out>/<比較名>/perf/  perf_trial.csv  corr.csv  report.txt
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np
from scipy.stats import spearmanr

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stroke_compare as SC      # noqa: E402
import stroke_feasibility as SF  # noqa: E402
import freq_phase as FP          # noqa: E402


def trial_perf(sdir, tdir, dpi_override):
    _, dpi, dpi_decl = SF.session_info(sdir, False)
    if dpi_override:
        dpi = float(dpi_override)
    scale = dpi_decl / dpi if (np.isfinite(dpi) and np.isfinite(dpi_decl) and dpi > 0) else 1.0
    a = np.genfromtxt(os.path.join(tdir, "task.csv"), delimiter=",", names=True)
    m = a["t"] >= 0
    t = np.asarray(a["t"], float)[m]
    tx, ty = (np.asarray(a[k], float)[m] * scale * 10 for k in ("target_x_cm", "target_y_cm"))
    cx, cy = (np.asarray(a[k], float)[m] * scale * 10 for k in ("cursor_x_cm", "cursor_y_cm"))
    on = np.asarray(a["on_target"], float)[m]
    if len(t) < 100:
        return None
    err = np.hypot(cx - tx, cy - ty)
    best = float("inf")
    for lag in np.arange(0, 0.5, 0.01):                  # カーソルを lag 秒戻して誤差を測る
        ct_x, ct_y = np.interp(t + lag, t, cx), np.interp(t + lag, t, cy)
        k = t + lag <= t[-1]
        best = min(best, float(np.sqrt(np.mean((ct_x[k] - tx[k]) ** 2 + (ct_y[k] - ty[k]) ** 2))))
    return dict(rms_mm=float(np.sqrt(np.mean(err ** 2))), on_pct=100 * float(np.nanmean(on)), rms_lagfree_mm=best)


def perm_spearman(x, y, groups, rng, n):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[ok], y[ok], np.asarray(groups)[ok]
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return None
    r = spearmanr(x, y).statistic
    idx = {g: np.where(groups == g)[0] for g in set(groups)}
    cnt = 0
    for _ in range(n):
        yp = y.copy()
        for g, ii in idx.items():
            yp[ii] = y[rng.permutation(ii)]
        cnt += abs(spearmanr(x, yp).statistic) >= abs(r) - 1e-12
    return r, (cnt + 1) / (n + 1), len(x)


def features(rows):
    """trial_freq.csv の行 → 試行ごとの特徴"""
    tr = {}
    for r in rows:
        k = (r["person"], r["band"], int(r["seed"]), int(r["attempt"]))
        tr.setdefault(k, []).append(r)
    out = {}
    for k, rs in tr.items():
        f = {}

        def mean(link, ax, key, top=None):
            v = [(float(r["f_hz"]), float(r[key])) for r in rs if r["link"] == link and r["axis"] == ax and r[key] not in ("", "nan")]
            v = [(a, b) for a, b in v if np.isfinite(b)]
            if top:
                v = sorted(v)[-top:]
            return float(np.mean([b for _, b in v])) if v else float("nan")
        for link, ax in (("F", "x"), ("H", "x"), ("U", "y"), ("S", "y"), ("F", "y")):
            f[f"{link}_{ax}_share"] = mean(link, ax, "share_re")
        f["gain_high"] = np.nanmean([mean("cursor", ax, "gain", 2) for ax in ("x", "y")])
        f["lag_high_ms"] = np.nanmean([mean("cursor", ax, "lag_ms", 2) for ax in ("x", "y")])
        f["lag_all_ms"] = np.nanmean([mean("cursor", ax, "lag_ms") for ax in ("x", "y")])
        out[k] = f
    return out


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p); q = np.empty_like(p); prev = 1.0
    for rank, i in zip(range(len(p), 0, -1), o[::-1]):
        prev = min(prev, p[i] * len(p) / rank); q[i] = prev
    return q


def run(cmp, cfg, sessions, found_all, args):
    name = cmp["name"]
    fdir = os.path.join(args.out, name, "freq")
    tf = os.path.join(fdir, "trial_freq.csv")
    policy = cmp.get("attempt", cfg["ATTEMPT"])
    tr_sel = cmp.get("trials", "all")
    tr_sel = None if tr_sel == "all" else ([tr_sel] if isinstance(tr_sel, str) else list(tr_sel))
    sess = {s["dir"]: s for s in sessions if SC.session_match(s, cmp.get("sessions", "all"))}
    found = SC.pick_attempts([f for f in found_all if os.path.normpath(os.path.abspath(f[0])) in sess], policy)
    found = [f for f in found if tr_sel is None or (SC.trial_tokens(f[2], f[3], f[4], f[1]) & set(tr_sel))]
    rows = []
    for sd, td, band, seed, att in found:
        s = sess[os.path.normpath(os.path.abspath(sd))]
        p = trial_perf(sd, td, s["dpi"])
        if p:
            rows.append(dict(person=s["person"], group=s["group"], band=band, seed=seed, attempt=att, trial=os.path.basename(td), **p))
    if not rows:
        print(f"[{name}] 試行なし"); return
    out = os.path.join(args.out, name, "perf"); os.makedirs(out, exist_ok=True)
    feats = {}
    if os.path.exists(tf):
        feats = features(list(csv.DictReader(open(tf, encoding="utf-8"))))
    for r in rows:
        r.update(feats.get((r["person"], r["band"], r["seed"], r["attempt"]), {}))
    FP.write_csv(os.path.join(out, "perf_trial.csv"), rows)
    persons = sorted({r["person"] for r in rows}); bands = sorted({r["band"] for r in rows})
    rng = np.random.default_rng(0)
    L = [f"[{name}] perf_link: persons {len(persons)}, trials {len(rows)}", ""]
    L.append("=== performance by person (mean over trials) ===")
    L.append("person   group     " + "  ".join(f"{b}:rms_mm  on%  lagfree" for b in bands))
    for p in persons:
        g = [r for r in rows if r["person"] == p]
        L.append(f"{p:<8} {g[0]['group']:<8}  " + "  ".join(
            (lambda rr: f"{np.mean([r['rms_mm'] for r in rr]):>8.1f} {np.mean([r['on_pct'] for r in rr]):>4.0f} {np.mean([r['rms_lagfree_mm'] for r in rr]):>7.1f}" if rr else "   -")([r for r in g if r["band"] == b]) for b in bands))
    L.append("")
    L.append("=== is performance a stable person trait? ICC(1) over trials ===")
    for b in bands:
        for m in ("rms_mm", "on_pct", "rms_lagfree_mm"):
            gdict = {}
            for r in rows:
                if r["band"] == b:
                    gdict.setdefault(r["person"], []).append(r[m])
            res = FP.icc_test(gdict, rng, args.perm)
            L.append(f"{b:<5} {m:<15} " + (f"ICC {res['ICC']:.2f}  between SD {res['between_sd']:.1f}  within SD {res['within_sd']:.1f}  p_perm {res['p_perm']:.3f}" if res else "n/a"))
    L.append("")
    SHARES = ["F_x_share", "H_x_share", "U_y_share", "S_y_share", "F_y_share"]
    LAGS = ("lag_all_ms", "lag_high_ms")
    TARGETS = ["rms_mm", "on_pct", "rms_lagfree_mm", "lag_all_ms", "lag_high_ms"]
    fnames = ["F_x_share", "H_x_share", "U_y_share", "S_y_share", "F_y_share", "gain_high", "lag_high_ms", "lag_all_ms"]
    if not feats:
        L.append(f"(動き方の特徴がありません: 先に freq_phase.py --compare {name} を実行)")
    else:
        crow = []
        for b in bands:
            R = [r for r in rows if r["band"] == b and np.isfinite(r.get("F_x_share", np.nan))]
            for m in TARGETS:
                for fn in fnames:
                    if m in LAGS and fn not in SHARES:
                        continue          # 遅れ自体は遅れの特徴とは比べない
                    # (a) between persons
                    pp = sorted({r["person"] for r in R})
                    xs = np.array([np.nanmean([r[fn] for r in R if r["person"] == p]) for p in pp])
                    ys = np.array([np.mean([r[m] for r in R if r["person"] == p]) for p in pp])
                    a = perm_spearman(xs, ys, ["all"] * len(pp), rng, args.perm)   # 人の間: 全体で並べ替え
                    # (b) within persons
                    x = np.array([r[fn] for r in R]); y = np.array([r[m] for r in R])
                    gp = np.array([r["person"] for r in R])
                    for p in set(gp):
                        ii = gp == p
                        x[ii] -= np.nanmean(x[ii]); y[ii] -= np.nanmean(y[ii])
                    w = perm_spearman(x, y, gp, rng, args.perm)
                    for kind, res in (("between", a), ("within", w)):
                        if res:
                            crow.append(dict(band=b, perf=m, feature=fn, kind=kind, rho=res[0], p=res[1], n=res[2]))
        if crow:
            for kind in ("between", "within"):
                sub = [c for c in crow if c["kind"] == kind]
                for c, q in zip(sub, bh([c["p"] for c in sub])):
                    c["q"] = float(q)
            FP.write_csv(os.path.join(out, "corr.csv"), crow)
            for kind, title in (("between", "人の間（人ごとの平均 n=人数）"), ("within", "人の中（人の平均を引いた試行 n=試行数）")):
                L.append(f"=== correlation, {title}: Spearman rho (p) ; rms_mm/rms_lagfree_mm は小さいほど良い，on_pct は大きいほど良い，lag_*_ms はマウスの遅れ（小さいほど速い）===")
                for b in bands:
                    L.append(f"-- {b}")
                    L.append(f"{'feature':<12}" + "".join(f"{m:>22}" for m in TARGETS))
                    for fn in fnames:
                        cells = []
                        for m in TARGETS:
                            c = [c for c in crow if c["band"] == b and c["perf"] == m and c["feature"] == fn and c["kind"] == kind]
                            cells.append(f"{c[0]['rho']:>+7.2f} (p={c[0]['p']:.3f}, q={c[0]['q']:.2f})" if c else "-")
                        L.append(f"{fn:<12}" + "".join(f"{c:>22}" for c in cells))
                sub = [c for c in crow if c["kind"] == kind]
                L.append(f"p<0.05: {sum(c['p'] < 0.05 for c in sub)} / {len(sub)} (偶然の期待 {0.05 * len(sub):.1f}) ; q<0.05: {sum(c['q'] < 0.05 for c in sub)}")
                L.append("")
    txt = "\n".join(L)
    open(os.path.join(out, "report.txt"), "w", encoding="utf-8").write(txt + "\n")
    print(txt); print(f"\n-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "compare_config.py"))
    ap.add_argument("--compare", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--perm", type=int, default=2000)
    args = ap.parse_args()
    cfg = SC.load_config(args.config)
    if args.out is None:
        args.out = cfg["OUT"] if os.path.isabs(cfg["OUT"]) else os.path.join(HERE, cfg["OUT"])
    sessions = SC.resolve_sessions(cfg)
    found_all = []
    for s in sessions:
        found_all += SF.discover([s["dir"]])
    for c in [c for c in cfg["COMPARES"] if not args.compare or c["name"] in args.compare]:
        print(f"\n##### {c['name']}")
        run(c, cfg, sessions, found_all, args)


if __name__ == "__main__":
    main()