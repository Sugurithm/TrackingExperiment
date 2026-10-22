"""prelim_inference.py — 中間審査用の予備解析（既存の run.py の結果を読んで推論する）

  python aim_eval/prelim_inference.py --config analysis_out/20261022/config_used.py \
         --out analysis_out/20261022 [--n-boot 5000] [--seed 0]

前提: 先に `run.py --config ... --steps analyze` を回し、OUT/between/trials.csv と
各試行の eval/、OUT/<参加者>/desk_frame.json ができていること。既存コードは書き換えない。

何をするか（出力は OUT/prelim/）
  0. 除外: config.GATE で引っかかった試行を除き、理由を excluded_trials.csv に残す
  1. 頑健性の再計算（試行ごと、common.py の関数をそのまま使う）
       v4   … a1 と同じ 4 成分（体幹・上腕・前腕・手）。a1 の値が再現できるかの確認を兼ねる
       v3   … 手首の点を使わない 3 成分（体幹・上腕・肘→MCP3）。手首は MoCap 未照合のため
       v2   … 近位（肩の並進 + 上腕 = 肘の位置）と遠位（肘→MCP3）
     それぞれ N_eff・S・C（そのまま）・C_corr（ビンのノイズパワーを引いた振幅で作る C）。
     成分どうしの入力周波数での位相差 → 時間差 [ms]（+ = 遠位が遅れる = 近位→遠位の順）と、
     逆相（|Δφ| > 90°）のビンの割合（打ち消し合いの直接の証拠）
  2. 関節依存性: 成分シェアの人ごとの平均、主役の成分、N_eff
  3. 協調様式: S・R・C_corr・位相の時間差
  4. DPI 効果（人物内）: 800 − 400 の差を人ごとに出し、dz・Wilcoxon・符号の一致数・
     帯域（slow / fast）をまたいだ再現・seed 間の SD と比べた人物内の z
  5. 個人差（人物間）: ICC(1,1)（条件平均を引いてから。DPI × 人の交互作用は人物内のばらつきに入れる
     ＝「DPI をまたいで個人特有か」の保守的な見方）、並べ替え検定、人をリサンプルするブートストラップ CI、
     400 と 800 の人の順位の一致（Spearman）
  6. 練度（ユーザーから与えられた順位、3 群）との Kendall τ（探索的。因果は読まない）
"""
import argparse, csv, importlib.util, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common as C                                      # noqa: E402
from scipy import stats                                 # noqa: E402

# ユーザーから与えられた練度（Syou, Yuuya, Suguru > Kakeru, Xin >> Kosuke, Yuuki, Suzuka）
# config の参加者名に合わせる（Yuuya → Yuya, Kakeru → kakeru, Xin → xin）。Keito は順位なし。
SKILL = {"Syou": 3, "Yuya": 3, "Suguru": 3, "kakeru": 2, "xin": 2, "Kosuke": 1, "Yuuki": 1, "Suzuka": 1}

VARIANTS = {
    "v4": {"trunk": ["shoulder", None], "upperarm": ["elbow", "shoulder"],
           "forearm": ["wrist", "elbow"], "hand": ["mcp3", "wrist"]},
    "v3": {"trunk": ["shoulder", None], "upperarm": ["elbow", "shoulder"], "distal": ["mcp3", "elbow"]},
    "v2": {"proximal": ["elbow", None], "distal": ["mcp3", "elbow"]},
}
PHASE_PAIRS = [("v4", "trunk", "upperarm"), ("v4", "upperarm", "forearm"),
               ("v4", "forearm", "hand"), ("v2", "proximal", "distal")]
MAIN_IDX = ["N_eff", "S", "C", "C_corr"]


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
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v


# ------------------------------------------------------------------ 1. 再計算
def components(tr, spec):
    """spec = {名前: [遠位の関節, 近位の関節 or None]} → 各成分の速度 (N,3)、手先の速度、有効フレーム。"""
    need = sorted({j for a, b in spec.values() for j in (a, b) if j})
    if any(n not in tr.rs_names for n in need):
        return None, None, None
    G = {n: tr.joint_grid(n, True) for n in need}
    ok = np.all([np.isfinite(G[n]).all(1) for n in need], axis=0)
    V = {}
    for p, (a, b) in spec.items():
        X = G[a] - (G[b] if b else 0.0)
        X = np.where(ok[:, None], X, np.nan)
        V[p] = np.gradient(X - np.nanmean(X, axis=0), axis=0) * C.FS_U
    E = np.where(ok[:, None], G["mcp3"], np.nan)
    E = np.gradient(E - np.nanmean(E, axis=0), axis=0) * C.FS_U
    return V, E, ok


def indices(tr, spec):
    """common.joint_indices と同じ手順で N_eff・S・C を作り、C_corr と位相も返す。"""
    V, E, ok = components(tr, spec)
    if V is None:
        return None
    names = list(spec)
    pw = {p: 0.0 for p in names}
    Cs, Cw, Cc, Ccw, per_bin = [], [], [], [], []
    amps = {p: [] for p in names}; csnr = {p: [] for p in names}; f_all, snrE_all, ax_all = [], [], []
    for k, ax in enumerate(C.AX):
        nb, f, _ = tr.freqs(ax)
        amp, pwr = {}, {}
        for p in names:
            sig, noi, A = C.bin_power(V[p][:, k], tr.T, nb)
            pwr[p] = np.maximum(sig - noi, 0)
            pw[p] += float(np.nansum(pwr[p]))
            amp[p] = A
            amps[p].append(A)
            with np.errstate(invalid="ignore", divide="ignore"):
                csnr[p].append(np.sqrt(sig / noi))             # 成分ごとのビンの SNR（振幅比）
        sigE, noiE, AE = C.bin_power(E[:, k], tr.T, nb)
        f_all.append(f); ax_all.append(np.full(len(f), ax))
        for i in range(len(nb)):
            den = float(np.abs(AE[i]))
            snr = float(np.sqrt(sigE[i] / noiE[i])) if noiE[i] > 0 else np.nan
            snrE_all.append(snr)
            if den > 0 and np.isfinite(snr):
                Cs.append(float(np.sum([np.abs(amp[p][i]) for p in names])) / den); Cw.append(den)
            denc = np.sqrt(max(sigE[i] - noiE[i], 0))
            if denc > 0 and np.isfinite(snr) and snr > C.SNR_MIN:
                Cc.append(float(np.sum([np.sqrt(pwr[p][i]) for p in names])) / denc); Ccw.append(denc)
            v = np.array([pwr[p][i] for p in names], float)
            per_bin.append((snr, v))
    tot = sum(pw.values())
    if tot <= 0:
        return None
    share = {p: pw[p] / tot for p in names}
    out = {"share": share, "N_eff": float(1 / np.sum([s ** 2 for s in share.values()])),
           "N_max": len(names), "pw": pw,
           "C": float(np.average(Cs, weights=Cw)) if Cs else np.nan,
           "C_corr": float(np.average(Cc, weights=Ccw)) if Cc else np.nan}
    nb_eff, nb_w, pw_sel = [], [], np.zeros(len(names))
    for snr, v in per_bin:
        if np.isfinite(snr) and snr > C.SNR_MIN and v.sum() > 0:
            q = v / v.sum()
            nb_eff.append(1 / np.sum(q ** 2)); nb_w.append(snr); pw_sel += v
    if len(nb_eff) >= 2 and pw_sel.sum() > 0:
        q = pw_sel / pw_sel.sum()
        out["S"] = float(np.average(nb_eff, weights=nb_w) / (1 / np.sum(q ** 2)))
    else:
        out["S"] = np.nan
    # 周波数の低い半分 / 高い半分でのシェア（S < 1 のとき、どの成分が速い成分を受け持つかを見る）
    fa = np.concatenate(f_all); fm = np.median(fa)
    V_ = np.array([v for _, v in per_bin])
    aa = np.concatenate(ax_all)
    # x = 左右、y = 前後。S < 1 が「速さ」でなく「方向」による分担から来ていないかを見る
    for tag, m in (("lo", fa <= fm), ("hi", fa > fm), ("x", aa == "x"), ("y", aa == "y")):
        tot_ = V_[m].sum()
        for j, p in enumerate(names):
            out[f"share_{tag}_{p}"] = float(V_[m, j].sum() / tot_) if tot_ > 0 else np.nan
    out["_amps"] = {p: np.concatenate(amps[p]) for p in names}
    out["_csnr"] = {p: np.concatenate(csnr[p]) for p in names}
    out["_f"] = np.concatenate(f_all)
    out["_snrE"] = np.array(snrE_all)
    return out


def phase_stats(res, a, b):
    """成分 a（近位）→ b（遠位）の位相差。SNR の高いビンだけ。+ ms = b が a より遅れる。"""
    Aa, Ab, f, snr = res["_amps"][a], res["_amps"][b], res["_f"], res["_snrE"]
    # 位相はノイズで一様に散る（逆相の割合 → 0.5）ので、手先だけでなく **両方の成分**が SNR > 2 のビンに限る
    sa, sb = res["_csnr"][a], res["_csnr"][b]
    ok = (np.isfinite(snr) & (snr > C.SNR_MIN) & np.isfinite(sa) & (sa > C.SNR_MIN)
          & np.isfinite(sb) & (sb > C.SNR_MIN) & (np.abs(Aa) > 0) & (np.abs(Ab) > 0))
    # 弱い成分の位相は不安定なので、振幅の積で重みを付ける（強いビンほど効く）
    if not ok.any():
        return {"n_bins": 0}
    d = np.angle(Ab[ok] / Aa[ok])                       # 遠位 − 近位 [rad]
    w = np.abs(Aa[ok]) * np.abs(Ab[ok])
    anti = np.abs(d) > np.pi / 2
    inph = ~anti
    lag = -d / (2 * np.pi * f[ok]) * 1000               # 遅れ [ms]
    return {"n_bins": int(ok.sum()),
            "anti_frac": float(np.sum(w[anti]) / np.sum(w)),
            "cos_mean": float(np.sum(w * np.cos(d)) / np.sum(w)),
            # 重み付きの円平均の位相 [deg]（+ = 遠位が進む）。低い周波数では小さな位相差でも ms が大きくなるので併記
            "phase_deg": float(np.degrees(np.angle(np.sum(w * np.exp(1j * d))))),
            "lag_ms": float(np.average(lag[inph], weights=w[inph])) if inph.any() else np.nan}


def recompute(cfg, out, trial_rows, lg):
    rows = []
    tmp = os.path.join(out, "_prelim_trial_tmp")
    for tr_row in trial_rows:
        d, name = tr_row["_dir"], tr_row["participant"]
        axes = os.path.join(out, name, "desk_frame.json")
        try:
            tr = C.Trial(Args(d, axes=axes if os.path.exists(axes) else None, out=tmp))
        except Exception as e:
            lg(f"- [SKIP] {d}: {e}"); continue
        r = {k: tr_row[k] for k in ("key", "participant", "dpi", "band", "seed")}
        for v, spec in VARIANTS.items():
            res = indices(tr, spec)
            if res is None:
                continue
            for k in ("N_eff", "S", "C", "C_corr"):
                r[f"{v}_{k}"] = res[k]
            for p, s in res["share"].items():
                r[f"{v}_share_{p}"] = s
                r[f"{v}_lo_share_{p}"] = res[f"share_lo_{p}"]
                r[f"{v}_hi_share_{p}"] = res[f"share_hi_{p}"]
                r[f"{v}_x_share_{p}"] = res[f"share_x_{p}"]
                r[f"{v}_y_share_{p}"] = res[f"share_y_{p}"]
                r[f"{v}_amp_{p}"] = float(np.sqrt(res["pw"][p]))   # ノイズを引いた入力ビンの振幅 [mm/s]
            r[f"_{v}"] = res
        for v, a, b in PHASE_PAIRS:
            if f"_{v}" in r:
                for k, x in phase_stats(r[f"_{v}"], a, b).items():
                    r[f"ph_{a}_{b}_{k}"] = x
        for v in VARIANTS:
            r.pop(f"_{v}", None)
        rows.append(r)
    return rows


# ------------------------------------------------------------------ 統計の道具
def icc1(groups):
    """ICC(1,1)（一元配置の変量モデル。群の大きさが違っても k0 で補正）。groups = [array, ...]"""
    g = [np.asarray(x, float)[np.isfinite(x)] for x in groups]
    g = [x for x in g if len(x) >= 2]
    if len(g) < 3:
        return np.nan
    n = np.array([len(x) for x in g]); N = n.sum(); a = len(g)
    grand = np.concatenate(g).mean()
    msb = np.sum(n * (np.array([x.mean() for x in g]) - grand) ** 2) / (a - 1)
    msw = np.sum([np.sum((x - x.mean()) ** 2) for x in g]) / (N - a)
    k0 = (N - np.sum(n ** 2) / N) / (a - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw)) if (msb + (k0 - 1) * msw) > 0 else np.nan


def boot_ci(fn, items, n_boot, rng, alpha=0.05):
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(items), len(items))
        v = fn([items[i] for i in idx])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))))


def fmt(x, f="{:.2f}"):
    return f.format(x) if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else "–"


def md(rows, cols):
    h = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        h += "| " + " | ".join(
            (fmt(r.get(c[1]), c[2]) if len(c) > 2 else str(r.get(c[1], "–"))) for c in cols) + " |\n"
    return h


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", choices=["relaxed", "strict"], default="relaxed",
                    help="strict = config.GATE そのまま / relaxed = ゲインと遅れだけ DPI の指定ミスを捉える幅に広げる")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = resolve(a.out)
    pout = os.path.join(out, f"prelim_{a.gate}"); os.makedirs(pout, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    logf = open(os.path.join(pout, "prelim_log.md"), "w", encoding="utf-8")

    def lg(s=""):
        print(s); logf.write(s + "\n"); logf.flush()

    lg(f"# prelim_inference（{time.strftime('%Y-%m-%d %H:%M:%S')}）\n")
    lg(f"config: {a.config} / out: {out} / n_boot {a.n_boot} / n_perm {a.n_perm} / seed {a.seed} / gate {a.gate}\n")

    # ---- 試行の一覧（config の SESSIONS と BANDS）と pipeline の trials.csv
    tc = list(csv.DictReader(open(os.path.join(out, "between", "trials.csv"), encoding="utf-8")))
    by_trial = {}
    for e in cfg.SESSIONS:
        d = resolve(e["dir"])
        for t, meta in C.list_trials(d, kinds=("track",)):
            if cfg.BANDS and meta["task"].get("band") not in cfg.BANDS:
                continue
            by_trial[(e["participant"], float(e["dpi"]), meta["task"]["band"], int(meta["task"]["seed"]))] = t
    trials = []
    for r in tc:
        if r.get("kind") != "track":
            continue
        key = (r["participant"], fnum(r["dpi"]), r["band"], int(fnum(r["seed"])))
        if key not in by_trial:
            continue
        r = dict(r); r["_dir"] = by_trial[key]
        r["key"] = f"{key[0]}_{key[1]:g}_{key[2]}_s{key[3]}"
        r["dpi"] = key[1]; r["seed"] = key[3]
        trials.append(r)
    lg(f"本番試行（track）: {len(trials)} 本 / config 上の期待値 {len(by_trial)} 本\n")
    missing = sorted(set(f"{k[0]}_{k[1]:g}_{k[2]}_s{k[3]}" for k in by_trial) - {t['key'] for t in trials})

    # ---- 0. 除外
    g = dict(cfg.GATE)
    if a.gate == "relaxed":
        # ゲインは 0.64〜0.90 の連続な分布（中央値 0.80）。DPI の指定ミスなら 0.4 か 1.6 付近に離れて出るので、
        # その離れた値だけを捉える幅にする。遅れも MediaPipe の遅れ（+25 ms 前後）から大きく外れたものだけ。
        g["mouse_gain"] = (0.5, 1.5)
        g["mouse_lag_ms"] = (-100, 150)
    excl, keep = [], []
    for r in trials:
        why = []
        gain, lag = fnum(r.get("mouse_gain_x")), fnum(r.get("mouse_lag_ms"))
        val, lr, snr = fnum(r.get("valid_wrist")), fnum(r.get("lr_swapped_pct")), fnum(r.get("snr_ok_pct"))
        if not np.isfinite(gain) or not (g["mouse_gain"][0] <= gain <= g["mouse_gain"][1]):
            why.append(f"ゲイン {gain:.2f}")
        if np.isfinite(lag) and not (g["mouse_lag_ms"][0] <= lag <= g["mouse_lag_ms"][1]):
            why.append(f"遅れ {lag:+.0f} ms")
        if np.isfinite(val) and val < g["valid_pct"]:
            why.append(f"手首の有効率 {val:.0f} %")
        if np.isfinite(lr) and lr > g["lr_swapped_pct"]:
            why.append(f"左右の取り違え {lr:.1f} %")
        if np.isfinite(snr) and snr < g["snr_ok_pct"]:
            why.append(f"SNR≥2 の割合 {snr:.0f} %")
        if not np.isfinite(fnum(r.get("N_eff"))):
            why.append("a1 の指標なし")
        (excl if why else keep).append(r)
        if why:
            r["why"] = "，".join(why)
    for k in missing:
        excl.append({"key": k, "why": "pipeline の trials.csv に無い（解析できなかった）"})
    with open(os.path.join(pout, "excluded_trials.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["trial", "reason"])
        for r in excl:
            w.writerow([r["key"], r["why"]])
    lg(f"## 0. 除外\n\n除外 {len(excl)} 本 / 使う {len(keep)} 本（GATE = {g}）\n")
    lg(md(excl, [("試行", "key"), ("理由", "why")]) if excl else "（なし）\n")

    # ---- 1. 再計算
    lg("\n## 1. 頑健性の再計算\n")
    t0 = time.time()
    R = recompute(cfg, out, keep, lg)
    lg(f"{len(R)} 本を再計算（{time.time() - t0:.0f} s）\n")
    base = {r["key"]: r for r in keep}
    for r in R:
        b = base[r["key"]]
        for k in ("N_eff", "S", "C", "R", "R_raw", "rms_2d", "on_target_pct", "mouse_gain_x"):
            r[f"a1_{k}" if k in ("N_eff", "S", "C", "R", "R_raw") else k] = fnum(b.get(k))
    # a1 との一致（v4 は a1 と同じ定義）
    chk = {k: np.nanmax(np.abs([r[f"v4_{k}"] - r[f"a1_{k}"] for r in R])) for k in ("N_eff", "S", "C")}
    lg(f"v4 と a1 の最大差: " + "，".join(f"{k} {v:.2e}" for k, v in chk.items())
       + "（0 に近ければ再計算は pipeline と同じ）\n")
    keys = sorted({k for r in R for k in r})
    with open(os.path.join(pout, "trials_prelim.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in R:
            w.writerow(r)

    people = sorted({r["participant"] for r in R}, key=str.lower)
    bands = [b for b in ("slow", "fast") if any(r["band"] == b for r in R)]
    dpis = sorted({r["dpi"] for r in R})

    def sel(p=None, band=None, dpi=None):
        return [r for r in R if (p is None or r["participant"] == p) and (band is None or r["band"] == band)
                and (dpi is None or r["dpi"] == dpi)]

    def pmean(p, band, dpi, k):
        v = [r.get(k, np.nan) for r in sel(p, band, dpi)]
        v = np.array(v, float)
        return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan

    def nmean(p, band, dpi, k):
        return int(np.isfinite(np.array([r.get(k, np.nan) for r in sel(p, band, dpi)], float)).sum())

    # ---- 2/3. 人ごとの平均（帯域 × DPI）
    lg("\n## 2. 関節依存性と 3. 協調様式（人ごとの平均。seed 3 本の平均）\n")
    prof = []
    IDX = ["v4_share_trunk", "v4_share_upperarm", "v4_share_forearm", "v4_share_hand",
           "v4_N_eff", "v3_N_eff", "v2_share_distal", "v4_S", "v4_C", "v4_C_corr", "a1_R",
           "ph_proximal_distal_lag_ms", "ph_proximal_distal_phase_deg", "ph_proximal_distal_anti_frac",
           "ph_upperarm_forearm_anti_frac", "ph_forearm_hand_anti_frac", "rms_2d",
           "v4_lo_share_upperarm", "v4_hi_share_upperarm", "v4_lo_share_forearm", "v4_hi_share_forearm",
           "v4_lo_share_hand", "v4_hi_share_hand", "v2_lo_share_distal", "v2_hi_share_distal",
           "ph_proximal_distal_n_bins", "v4_x_share_trunk", "v4_y_share_trunk", "v4_x_share_upperarm",
           "v4_y_share_upperarm", "v4_x_share_forearm", "v4_y_share_forearm", "v4_x_share_hand", "v4_y_share_hand"]
    for band in bands:
        for dpi in dpis:
            for p in people:
                row = {"participant": p, "band": band, "dpi": dpi, "n": len(sel(p, band, dpi)),
                       "skill": SKILL.get(p, np.nan)}
                for k in IDX:
                    row[k] = pmean(p, band, dpi, k)
                sh = {c: row[f"v4_share_{c}"] for c in ("trunk", "upperarm", "forearm", "hand")}
                row["dominant"] = max(sh, key=lambda c: sh[c] if np.isfinite(sh[c]) else -1)
                prof.append(row)
    with open(os.path.join(pout, "person_means.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(prof[0])); w.writeheader()
        for r in prof:
            w.writerow(r)
    cols = [("参加者", "participant"), ("帯域", "band"), ("DPI", "dpi", "{:.0f}"), ("本", "n", "{:.0f}"),
            ("体幹", "v4_share_trunk", "{:.0%}"), ("上腕", "v4_share_upperarm", "{:.0%}"),
            ("前腕", "v4_share_forearm", "{:.0%}"), ("手", "v4_share_hand", "{:.0%}"),
            ("主役", "dominant"), ("N_eff(4)", "v4_N_eff", "{:.2f}"), ("N_eff(3,手首なし)", "v3_N_eff", "{:.2f}"),
            ("S", "v4_S", "{:.2f}"), ("C", "v4_C", "{:.2f}"), ("C_corr", "v4_C_corr", "{:.2f}"), ("R", "a1_R", "{:+.2f}"),
            ("近→遠 [ms]", "ph_proximal_distal_lag_ms", "{:+.0f}"),
            ("遠−近 位相 [deg]", "ph_proximal_distal_phase_deg", "{:+.0f}"),
            ("近/遠 逆相", "ph_proximal_distal_anti_frac", "{:.0%}")]
    lg(md(prof, cols))

    # 群の要約（人を単位に。帯域 × DPI ごと）
    lg("\n### 群の要約（人 = 単位。平均 [95 % CI, 人をリサンプル]、範囲）\n")
    grp = []
    for band in bands:
        for dpi in dpis:
            P = [r for r in prof if r["band"] == band and r["dpi"] == dpi]
            for k in IDX:
                v = np.array([r[k] for r in P], float); v = v[np.isfinite(v)]
                if len(v) < 2:
                    continue
                lo, hi = boot_ci(lambda s: float(np.mean(s)), list(v), a.n_boot, rng)
                grp.append({"band": band, "dpi": dpi, "idx": k, "n": len(v), "mean": float(v.mean()),
                            "lo": lo, "hi": hi, "min": float(v.min()), "max": float(v.max())})
    lg(md(grp, [("帯域", "band"), ("DPI", "dpi", "{:.0f}"), ("指標", "idx"), ("人", "n", "{:.0f}"),
                ("平均", "mean", "{:.3f}"), ("CI 下", "lo", "{:.3f}"), ("CI 上", "hi", "{:.3f}"),
                ("最小", "min", "{:.3f}"), ("最大", "max", "{:.3f}")]))
    dom = {}
    for r in prof:
        dom.setdefault((r["band"], r["dpi"]), []).append(r["dominant"])
    lg("\n主役の成分（人数）: " + "；".join(
        f"{b} {d:.0f}: " + "，".join(f"{c} {v.count(c)}" for c in sorted(set(v))) for (b, d), v in dom.items()))

    # ---- 4. DPI 効果（人物内）
    lg("\n\n## 4. DPI 効果（人物内: 800 − 400）\n")
    lg("d = 人ごとの (800 の seed 平均) − (400 の seed 平均)。dz = mean(d)/SD(d)（人をまたいだ効果量）。"
       "z_within = d / √(s²₄₀₀/n + s²₈₀₀/n)（seed 間のばらつきに対する大きさ。|z| ≥ 2 の人数）。"
       "再現 = slow と fast で d の符号が同じ人数。\n")
    DPI_IDX = ["v4_N_eff", "v3_N_eff", "v4_S", "v4_C_corr", "a1_R", "v4_share_trunk", "v4_share_upperarm",
               "v4_share_forearm", "v4_share_hand", "v2_share_distal", "ph_proximal_distal_lag_ms", "ph_proximal_distal_phase_deg",
               "ph_proximal_distal_anti_frac", "rms_2d"]
    AMP = [f"v4_amp_{c}" for c in ("trunk", "upperarm", "forearm", "hand")] + ["v2_amp_proximal", "v2_amp_distal"]
    dres, dper = [], {}
    if len(dpis) == 2:
        lo_d, hi_d = dpis
        for band in bands:
            for k in DPI_IDX + AMP:
                d, z, ps = [], [], []
                for p in people:
                    x0 = np.array([r.get(k, np.nan) for r in sel(p, band, lo_d)], float)
                    x1 = np.array([r.get(k, np.nan) for r in sel(p, band, hi_d)], float)
                    x0, x1 = x0[np.isfinite(x0)], x1[np.isfinite(x1)]
                    if len(x0) < 2 or len(x1) < 2:
                        continue
                    if k in AMP:                       # 振幅は比の log2（−1 = DPI に比例して半分）
                        if (x0 <= 0).any() or (x1 <= 0).any():
                            continue
                        x0, x1 = np.log2(x0), np.log2(x1)
                    dd = x1.mean() - x0.mean()
                    se = np.sqrt(x0.var(ddof=1) / len(x0) + x1.var(ddof=1) / len(x1))
                    d.append(dd); z.append(dd / se if se > 0 else np.nan); ps.append(p)
                    dper[(band, k, p)] = dd
                d = np.array(d); z = np.array(z)
                if len(d) < 3:
                    continue
                try:
                    pw = float(stats.wilcoxon(d).pvalue)
                except ValueError:
                    pw = np.nan
                lo, hi = boot_ci(lambda s: float(np.mean(s)), list(d), a.n_boot, rng)
                dres.append({"band": band, "idx": k + (" [log2比]" if k in AMP else ""), "n": len(d),
                             "mean_d": float(d.mean()), "lo": lo, "hi": hi,
                             "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan,
                             "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
                             "n_zsig": int((np.abs(z) >= 2).sum()), "p_wilcoxon": pw})
        for r in dres:
            k = r["idx"].replace(" [log2比]", "")
            if r["band"] != bands[0] or len(bands) < 2:
                continue
            same = [np.sign(dper.get((bands[0], k, p), np.nan)) == np.sign(dper.get((bands[1], k, p), np.nan))
                    for p in people if (bands[0], k, p) in dper and (bands[1], k, p) in dper]
            r["rep_bands"] = f"{int(np.sum(same))}/{len(same)}"
            x = [dper[(bands[0], k, p)] for p in people if (bands[0], k, p) in dper and (bands[1], k, p) in dper]
            y = [dper[(bands[1], k, p)] for p in people if (bands[0], k, p) in dper and (bands[1], k, p) in dper]
            r["r_bands"] = float(stats.spearmanr(x, y).statistic) if len(x) >= 4 else np.nan
    lg(md(dres, [("帯域", "band"), ("指標", "idx"), ("人", "n", "{:.0f}"), ("平均 d", "mean_d", "{:+.3f}"),
                 ("CI 下", "lo", "{:+.3f}"), ("CI 上", "hi", "{:+.3f}"), ("dz", "dz", "{:+.2f}"),
                 ("+ の人", "n_pos", "{:.0f}"), ("− の人", "n_neg", "{:.0f}"),
                 ("|z_within|≥2", "n_zsig", "{:.0f}"), ("p (Wilcoxon)", "p_wilcoxon", "{:.3f}"),
                 ("slow/fast 符号一致", "rep_bands"), ("d の順位相関 slow-fast", "r_bands", "{:+.2f}")]))
    with open(os.path.join(pout, "dpi_effect.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in dres for k in r})); w.writeheader()
        for r in dres:
            w.writerow(r)

    # ---- 5. 個人差（人物間）
    lg("\n\n## 5. 個人差（人物間）\n")
    lg("ICC(1,1): 帯域ごとに、各試行から (帯域 × DPI) の平均を引いた値を人ごとにまとめる（1 人 6 本）。"
       "DPI × 人の交互作用は人物内のばらつきに入るので「DPI をまたいで個人特有か」の保守的な値。"
       "p = 条件内で人のラベルを入れ替えた並べ替え検定。CI = 人をリサンプル。"
       "ρ(400, 800) = 人の平均の順位が DPI をまたいで保たれるか。D = SD(人の平均)/√(人物内の平均分散)。\n")
    BIDX = ["v4_N_eff", "v3_N_eff", "v4_S", "v4_C_corr", "a1_R", "v4_share_trunk", "v4_share_upperarm",
            "v4_share_forearm", "v4_share_hand", "v2_share_distal", "ph_proximal_distal_lag_ms", "ph_proximal_distal_phase_deg",
            "ph_proximal_distal_anti_frac", "rms_2d"]
    bres = []
    for band in bands:
        for k in BIDX:
            cen = {}
            for dpi in dpis:
                rr = sel(None, band, dpi)
                m = np.nanmean([r.get(k, np.nan) for r in rr]) if rr else np.nan
                for r in rr:
                    cen.setdefault(r["participant"], []).append(r.get(k, np.nan) - m)
            G = [np.array(cen[p], float) for p in people if p in cen]
            G = [x[np.isfinite(x)] for x in G]
            G = [x for x in G if len(x) >= 2]
            if len(G) < 3:
                continue
            icc = icc1(G)
            allv = np.concatenate(G); sizes = [len(x) for x in G]
            null = []
            for _ in range(a.n_perm):
                perm = rng.permutation(allv); i0 = 0; gg = []
                for s in sizes:
                    gg.append(perm[i0:i0 + s]); i0 += s
                null.append(icc1(gg))
            null = np.array(null)
            p_perm = float((np.sum(null >= icc) + 1) / (len(null) + 1))
            lo, hi = boot_ci(icc1, G, a.n_boot, rng)
            Dv = float(np.std([x.mean() for x in G], ddof=1) / np.sqrt(np.mean([x.var(ddof=1) for x in G])))
            m0 = [pmean(p, band, dpis[0], k) for p in people]
            m1 = [pmean(p, band, dpis[-1], k) for p in people]
            okm = np.isfinite(m0) & np.isfinite(m1)
            rho = stats.spearmanr(np.array(m0)[okm], np.array(m1)[okm]) if okm.sum() >= 4 else None
            bres.append({"band": band, "idx": k, "n_people": len(G), "n_trials": int(sum(sizes)),
                         "ICC": icc, "lo": lo, "hi": hi, "p_perm": p_perm, "D": Dv,
                         "rho": float(rho.statistic) if rho else np.nan,
                         "p_rho": float(rho.pvalue) if rho else np.nan})
    lg(md(bres, [("帯域", "band"), ("指標", "idx"), ("人", "n_people", "{:.0f}"), ("試行", "n_trials", "{:.0f}"),
                 ("ICC", "ICC", "{:+.2f}"), ("CI 下", "lo", "{:+.2f}"), ("CI 上", "hi", "{:+.2f}"),
                 ("p (perm)", "p_perm", "{:.4f}"), ("D", "D", "{:.2f}"),
                 ("ρ(400,800)", "rho", "{:+.2f}"), ("p(ρ)", "p_rho", "{:.3f}")]))
    with open(os.path.join(pout, "between_icc.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bres[0])); w.writeheader()
        for r in bres:
            w.writerow(r)

    # 帯域をまたいだ個人の一貫性（slow の人の平均と fast の人の平均）
    if len(bands) == 2:
        lg("\n### 帯域をまたいだ個人の順位の一致（DPI を平均した人の値、Spearman）\n")
        cr = []
        for k in BIDX:
            x = [np.nanmean([pmean(p, bands[0], d, k) for d in dpis]) for p in people]
            y = [np.nanmean([pmean(p, bands[1], d, k) for d in dpis]) for p in people]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 4:
                s = stats.spearmanr(np.array(x)[ok], np.array(y)[ok])
                cr.append({"idx": k, "n": int(ok.sum()), "rho": float(s.statistic), "p": float(s.pvalue)})
        lg(md(cr, [("指標", "idx"), ("人", "n", "{:.0f}"), ("ρ(slow, fast)", "rho", "{:+.2f}"), ("p", "p", "{:.3f}")]))

    # ---- 6. 練度との関係（探索的）
    lg("\n\n## 6. 練度（3 群: 3 = Syou, Yuya, Suguru / 2 = kakeru, xin / 1 = Kosuke, Yuuki, Suzuka）との Kendall τ\n")
    lg("人の平均（DPI 2 条件を平均）と練度の群の順位相関。p = 練度ラベルの並べ替え（両側）。"
       "**8 人・探索的・多重比較の補正なし。因果は読まない**（練度 → 協調の向きも、"
       "協調 → 成績の向きも、この相関からは決まらない）。rms_2d は練度の操作確認（成績が練度と並ぶか）。\n")
    sres = []
    sk_people = [p for p in people if p in SKILL]
    for band in bands:
        for k in BIDX:
            x = np.array([np.nanmean([pmean(p, band, d, k) for d in dpis]) for p in sk_people])
            y = np.array([SKILL[p] for p in sk_people], float)
            ok = np.isfinite(x)
            if ok.sum() < 5:
                continue
            tau = stats.kendalltau(x[ok], y[ok]).statistic
            null = [stats.kendalltau(x[ok], rng.permutation(y[ok])).statistic for _ in range(2000)]
            p = float((np.sum(np.abs(null) >= abs(tau) - 1e-12) + 1) / (len(null) + 1))
            hi = x[ok][y[ok] == 3]; lo = x[ok][y[ok] == 1]
            sp = np.sqrt((hi.var(ddof=1) + lo.var(ddof=1)) / 2) if len(hi) > 1 and len(lo) > 1 else np.nan
            sres.append({"band": band, "idx": k, "n": int(ok.sum()), "tau": float(tau), "p": p,
                         "hi_mean": float(hi.mean()) if len(hi) else np.nan,
                         "lo_mean": float(lo.mean()) if len(lo) else np.nan,
                         "g_hi_lo": float((hi.mean() - lo.mean()) / sp) if sp and np.isfinite(sp) and sp > 0 else np.nan})
    lg(md(sres, [("帯域", "band"), ("指標", "idx"), ("人", "n", "{:.0f}"), ("τ", "tau", "{:+.2f}"),
                 ("p (perm)", "p", "{:.3f}"), ("上位群 平均", "hi_mean", "{:.3f}"),
                 ("下位群 平均", "lo_mean", "{:.3f}"), ("効果量 g（上位−下位）", "g_hi_lo", "{:+.2f}")]))
    with open(os.path.join(pout, "skill_corr.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sres[0])); w.writeheader()
        for r in sres:
            w.writerow(r)

    # ---- 図
    plots(R, prof, people, bands, dpis, pout)
    json.dump({"group": grp, "dpi": dres, "between": bres, "skill": sres, "check_v4_vs_a1": chk,
               "excluded": [{"trial": r["key"], "reason": r["why"]} for r in excl]},
              open(os.path.join(pout, "prelim.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=C._jsonable)
    lg(f"\n→ {pout}")
    logf.close()


def plots(R, prof, people, bands, dpis, pout):
    plt = C.setup_plot()
    comps = ["trunk", "upperarm", "forearm", "hand"]
    # 図 1: 成分シェア（人 × DPI、帯域ごと）
    fig, axes = plt.subplots(1, len(bands), figsize=(7 * len(bands), 4.6), squeeze=False)
    for ax, band in zip(axes[0], bands):
        x, labels = 0, []
        for p in people:
            for dpi in dpis:
                r = [q for q in prof if q["participant"] == p and q["band"] == band and q["dpi"] == dpi]
                if not r:
                    continue
                bottom = 0
                for c in comps:
                    v = r[0][f"v4_share_{c}"]
                    v = v if np.isfinite(v) else 0
                    ax.bar(x, v, bottom=bottom, color=C.PARTCOLOR[c], width=.8,
                           label=C.PARTLABEL[c] if x == 0 else None)
                    bottom += v
                labels.append(f"{p}\n{dpi:.0f}")
                x += 1
            x += .5
        ax.set_xticks([i + (i // len(dpis)) * .5 for i in range(len(labels))])
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel("寄与シェア（入力周波数、ノイズ差引）"); ax.set_ylim(0, 1)
        ax.set_title(f"{band}: 成分シェア（seed 3 本の平均）")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.13), ncol=4)
    fig.tight_layout(); fig.savefig(os.path.join(pout, "fig1_shares.png"), dpi=130); plt.close(fig)

    # 図 5: 方向（x = 左右、y = 前後）ごとの成分シェア（人の平均の平均 ± 人の SD）
    fig, axes = plt.subplots(1, len(bands), figsize=(6 * len(bands), 4.2), squeeze=False)
    for ax, band in zip(axes[0], bands):
        x = 0; ticks, labs = [], []
        for dirn, dl in (("x", "x（左右）"), ("y", "y（前後）")):
            for dpi in dpis:
                bottom = 0
                for c in comps:
                    v = [np.nanmean([r.get(f"v4_{dirn}_share_{c}", np.nan) for r in R if r["participant"] == p
                                     and r["band"] == band and r["dpi"] == dpi]) for p in people]
                    m = float(np.nanmean(v))
                    ax.bar(x, m, bottom=bottom, color=C.PARTCOLOR[c], width=.8,
                           label=C.PARTLABEL[c] if x == 0 else None)
                    ax.errorbar(x, bottom + m, yerr=np.nanstd(v, ddof=1), color="k", lw=.7, capsize=2)
                    bottom += m
                ticks.append(x); labs.append(f"{dl}\n{dpi:.0f} DPI"); x += 1
            x += .6
        ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=8); ax.set_ylim(0, 1.05)
        ax.set_ylabel("寄与シェア"); ax.set_title(f"{band}: 方向ごとの分担（{len(people)} 人の平均 ± SD）")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=4)
    fig.tight_layout(); fig.savefig(os.path.join(pout, "fig5_direction.png"), dpi=130); plt.close(fig)

    # 図 2: 指標の DPI 対（人ごとの線）
    keys = [("v4_N_eff", "N_eff（4 成分）"), ("v3_N_eff", "N_eff（3 成分・手首なし）"), ("v4_S", "S（同時性）"),
            ("v4_C_corr", "C_corr（相殺比・ノイズ補正）"), ("v2_share_distal", "遠位（肘→MCP3）のシェア"),
            ("ph_proximal_distal_lag_ms", "近位→遠位の時間差 [ms]")]
    fig, axes = plt.subplots(len(bands), len(keys), figsize=(3.1 * len(keys), 3.2 * len(bands)), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for i, band in enumerate(bands):
        for j, (k, lab) in enumerate(keys):
            ax = axes[i][j]
            for n, p in enumerate(people):
                ys, es = [], []
                for dpi in dpis:
                    v = np.array([r.get(k, np.nan) for r in R if r["participant"] == p and r["band"] == band
                                  and r["dpi"] == dpi], float)
                    ys.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
                    es.append(np.nanstd(v, ddof=1) if np.isfinite(v).sum() > 1 else np.nan)
                ax.errorbar(np.arange(len(dpis)) + (n - len(people) / 2) * .02, ys, yerr=es, marker="o", ms=3.5,
                            lw=1, capsize=2, color=cmap(n % 10), label=p)
            ax.set_xticks(range(len(dpis))); ax.set_xticklabels([f"{d:.0f}" for d in dpis])
            ax.set_title(f"{band}: {lab}", fontsize=8.5)
            if k == "v4_S":
                ax.axhline(1, color="#888", ls="--", lw=.8); ax.axhline(.85, color="#c55", ls=":", lw=.8)
            if k == "v4_C_corr":
                ax.axhline(1, color="#888", ls="--", lw=.8)
            if k == "ph_proximal_distal_lag_ms":
                ax.axhline(0, color="#888", ls="--", lw=.8)
    axes[0][-1].legend(fontsize=6, ncol=1, loc="center left", bbox_to_anchor=(1.02, .5))
    fig.tight_layout(); fig.savefig(os.path.join(pout, "fig2_dpi_pairs.png"), dpi=130); plt.close(fig)

    # 図 3: 頑健性（a1 の C と C_corr、4 成分と 3 成分の N_eff）
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.array([r.get("v4_C", np.nan) for r in R]); y = np.array([r.get("v4_C_corr", np.nan) for r in R])
    axes[0].scatter(x, y, s=10, alpha=.6); axes[0].plot([1, 3], [1, 3], "k--", lw=.8)
    axes[0].set_xlabel("C（a1、ノイズ込み）"); axes[0].set_ylabel("C_corr（ノイズ差引）"); axes[0].set_title("相殺比の補正")
    x = np.array([r.get("v4_N_eff", np.nan) for r in R]); y = np.array([r.get("v3_N_eff", np.nan) for r in R])
    axes[1].scatter(x, y, s=10, alpha=.6); axes[1].plot([1, 4], [1, 4], "k--", lw=.8)
    axes[1].set_xlabel("N_eff 4 成分"); axes[1].set_ylabel("N_eff 3 成分（手首なし）"); axes[1].set_title("手首の点に依存するか")
    for band, mk in zip(bands, "os"):
        rr = [r for r in R if r["band"] == band]
        axes[2].scatter([r.get("v4_S", np.nan) for r in rr], [r.get("a1_R", np.nan) for r in rr], s=12,
                        marker=mk, alpha=.6, label=band)
    axes[2].axvline(.85, color="#c55", ls=":", lw=.8); axes[2].axvline(.95, color="#888", ls="--", lw=.8)
    axes[2].axhline(0, color="#888", ls="--", lw=.8)
    axes[2].set_xlabel("S"); axes[2].set_ylabel("R（信頼できる試行のみ）"); axes[2].legend(fontsize=8)
    axes[2].set_title("同時性: 周波数（S）と時間（R）")
    fig.tight_layout(); fig.savefig(os.path.join(pout, "fig3_robustness.png"), dpi=130); plt.close(fig)

    # 図 4: 練度と主要指標
    sk = [p for p in people if p in SKILL]
    keys4 = [("v4_N_eff", "N_eff"), ("v2_share_distal", "遠位のシェア"), ("v4_S", "S"),
             ("v4_C_corr", "C_corr"), ("rms_2d", "追従誤差 RMS（操作確認）")]
    fig, axes = plt.subplots(len(bands), len(keys4), figsize=(3 * len(keys4), 3 * len(bands)), squeeze=False)
    for i, band in enumerate(bands):
        for j, (k, lab) in enumerate(keys4):
            ax = axes[i][j]
            for p in sk:
                v = np.nanmean([q[k] for q in prof if q["participant"] == p and q["band"] == band])
                ax.scatter(SKILL[p] + (hash(p) % 7 - 3) * .03, v, s=18)
                ax.annotate(p, (SKILL[p], v), fontsize=6, xytext=(3, 0), textcoords="offset points")
            ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["下位", "中位", "上位"])
            ax.set_title(f"{band}: {lab}", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(pout, "fig4_skill.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
