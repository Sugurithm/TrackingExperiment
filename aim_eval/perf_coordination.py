"""perf_coordination.py — エイムの質（スコア）と関節協調の関係（中間審査用の予備解析）

  python aim_eval/perf_coordination.py --config analysis_out/20261022/config_used.py \
         --out analysis_out/20261022 [--prelim prelim_relaxed] [--n-boot 5000] [--n-perm 5000] [--seed 0]

前提: run.py --steps analyze と prelim_inference.py を先に回しておく
（OUT/between/trials.csv、OUT/<prelim>/trials_prelim.csv を読む）。既存コードは書き換えない。

1. エイムの質の指標（試行ごと。task.csv の 250 Hz のカーソル・ターゲットから。単位は画面上の mm）
   カーソルは DPI によらず 1 count あたり同じだけ動くので、カーソル空間の指標は DPI をまたいで比べられる。
   正確さ
     rms_2d         … RMS 誤差 [mm]（a6）
     on_target_pct  … ターゲット内にいた時間の割合 [%]（a6）
   滑らかさ（ターゲット自体の滑らかさで割る / 引く。ターゲットは 1.08 Hz 以下の正弦波の和）
     log_jerk_ratio … log10( RMS 躍度(カーソル) / RMS 躍度(ターゲット) )。0 = ターゲットと同じ滑らかさ、大きいほどぎくしゃく
     hf_frac        … カーソル速度のパワーのうち 1.5 Hz より上（課題に無い帯域）の割合。小さいほど滑らか
                       （SPARC も計算して CSV に残すが解析には使わない。40 s の追従では速さのスペクトルが
                        ターゲットへの追従ゲインに支配され、遅れて小さく追う試行ほど「滑らか」に出たため）
     submove_rate   … 速さの山の数 [1/s] のターゲットに対する超過分（断続的な修正動作の頻度）
   安定性
     hf_jitter      … 2〜10 Hz のカーソル速度の RMS [mm/s]（課題に無い速い揺れ = ふらつき・修正）
     remnant        … 入力周波数以外に出たカーソルのパワーの割合（a5、x・y の平均）
     exit_rate      … ターゲットから外れる回数 [1/s]（20 ms の中央値フィルタで瞬間的な出入りは除く）
     dwell_s        … ターゲット内に連続していた時間の平均 [s]（捕まえたら離さない度合い）
     err_cv         … 2 s 窓ごとの RMS 誤差の変動係数（試行内で出来がぶれるか）
   追従の動特性（a5、x・y の平均）
     delay_ms       … 実効遅れ [ms]
     （a5 の追従帯域 fb は外挿で約 20 Hz に張り付くので使わない。CSV には残す）

2. 協調の指標との関係（協調は prelim_inference.py の値。手首に依存する指標は【参考】）
   B1 人物間   … 人の平均（DPI 2 条件 × seed 3 本の平均）どうしの Spearman ρ。9 人。
                  人をリサンプルした CI、並べ替え p、帯域ごとに BH-FDR の q
   B2 条件内   … 同じ人・同じ DPI・同じ帯域の中での試行間（seed 間）のばらつきどうしの相関。
                  (人 × DPI × 帯域) と (seed × 帯域 = 軌道と練習順) の効果を最小二乗で取り除いた残差の相関
                  （repeated-measures correlation の拡張）。「同じ人が、ある関節をより使った試行で成績が良いか」
   B3 DPI 変化 … 人ごとの Δ協調（800 − 400）と Δ成績の Spearman ρ。「DPI で使い方を大きく変えた人ほど成績が変わるか」
   3 つのどれも相関であって、**因果の向き（協調 → 成績 / 成績 → 協調 / 第 3 の要因）は決まらない**。
"""
import argparse, csv, importlib.util, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common as C                                      # noqa: E402
from scipy import stats                                 # noqa: E402
from scipy.signal import butter, filtfilt, find_peaks, medfilt  # noqa: E402

SKILL = {"Syou": 3, "Yuya": 3, "Suguru": 3, "kakeru": 2, "xin": 2, "Kosuke": 1, "Yuuki": 1, "Suzuka": 1}
FS = 250.0
FC_SMOOTH = 10.0           # 滑らかさの前処理の低域通過 [Hz]（マウスの量子化と描画の揺れを落とす）
HF_BAND = (2.0, 10.0)      # 課題に無い速い揺れの帯域 [Hz]（課題の最高周波数は 1.08 Hz）
WIN_S = 2.0

PERF = [  # (キー, 見出し, 良い向き: +1 = 大きいほど良い / −1 = 小さいほど良い)
    ("on_target_pct", "on-target [%]", +1), ("rms_2d", "RMS 誤差 [mm]", -1),
    ("log_jerk_ratio", "躍度比 log10", -1), ("hf_frac", "1.5 Hz 超の割合", -1), ("submove_rate", "修正の頻度 [1/s]", -1),
    ("hf_jitter", "2–10 Hz の揺れ [mm/s]", -1), ("remnant", "remnant", -1), ("exit_rate", "外れる頻度 [1/s]", -1),
    ("dwell_s", "滞在時間 [s]", +1), ("err_cv", "誤差の変動係数", -1),
    ("delay_ms", "実効遅れ [ms]", -1),
]
# 追従帯域（a5 の fb_fit_hz）は全条件で約 20 Hz に張り付いた（入力周波数の範囲 ≤ 1.08 Hz からの外挿）ので使わない
COORD = [  # (キー, 見出し, 手首に依存するか)
    ("v3_N_eff", "N_eff（手首なし）", False), ("v2_share_distal", "遠位シェア", False),
    ("v4_share_trunk", "体幹シェア", False), ("v4_share_upperarm", "上腕シェア", False),
    ("v4_share_forearm", "前腕シェア", True), ("v4_S", "S", True), ("v4_C_corr", "C_corr", True),
    ("ph_proximal_distal_anti_frac", "近/遠の逆相", False),
    ("v4_share_hand", "手シェア【参考】", True), ("v4_N_eff", "N_eff（4 成分）【参考】", True),
]


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


# ------------------------------------------------------------------ 1. エイムの質
def sparc(speed, fs, fc=FC_SMOOTH, amp_th=0.05, pad=4):
    """SPARC（Balasubramanian et al. 2015）。速さの振幅スペクトルの弧長（負、0 に近いほど滑らか）。"""
    N = len(speed)
    nfft = int(2 ** (np.ceil(np.log2(N)) + pad))
    f = np.arange(nfft) / nfft * fs
    V = np.abs(np.fft.fft(speed, nfft))
    V = V / V.max()
    sel = f <= fc
    f, V = f[sel], V[sel]
    idx = np.flatnonzero(V >= amp_th)
    f, V = f[:idx[-1] + 1], V[:idx[-1] + 1]
    return float(-np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(V) ** 2)))


def perf_metrics(tr):
    t = np.arange(int(round(tr.T * FS))) / FS
    g = np.isfinite(tr.task_t)
    tg = np.column_stack([np.interp(t, tr.task_t[g], tr.tgt[g, k]) for k in range(2)])
    cu = np.column_stack([np.interp(t, tr.task_t[g], tr.cur[g, k]) for k in range(2)])
    on = np.interp(t, tr.task_t[g], tr.on_target[g].astype(float)) > 0.5
    b, a = butter(4, FC_SMOOTH / (FS / 2))
    tgf, cuf = filtfilt(b, a, tg, axis=0), filtfilt(b, a, cu, axis=0)

    def deriv(X, n):
        for _ in range(n):
            X = np.gradient(X, axis=0) * FS
        return X

    vt, vc = deriv(tgf, 1), deriv(cuf, 1)
    jt, jc = deriv(tgf, 3), deriv(cuf, 3)
    st, sc = np.linalg.norm(vt, axis=1), np.linalg.norm(vc, axis=1)
    e = cu - tg
    m = {}
    m["log_jerk_ratio"] = float(np.log10(np.sqrt(np.mean(np.sum(jc ** 2, 1))) / np.sqrt(np.mean(np.sum(jt ** 2, 1)))))
    m["sparc_cursor"] = sparc(sc, FS); m["sparc_target"] = sparc(st, FS)
    m["dsparc"] = m["sparc_cursor"] - m["sparc_target"]
    F = np.abs(np.fft.rfft(vc - vc.mean(0), axis=0)) ** 2
    fr = np.fft.rfftfreq(len(vc), 1 / FS)
    m["hf_frac"] = float(F[fr > 1.5].sum() / F[fr > 0].sum())
    prom_c, prom_t = 0.1 * sc.std(), 0.1 * st.std()
    m["submove_rate"] = (len(find_peaks(sc, prominence=prom_c)[0]) - len(find_peaks(st, prominence=prom_t)[0])) / tr.T
    bb, ab = butter(4, [HF_BAND[0] / (FS / 2), HF_BAND[1] / (FS / 2)], btype="band")
    m["hf_jitter"] = float(np.sqrt(np.mean(np.sum(filtfilt(bb, ab, deriv(cu, 1), axis=0) ** 2, 1))))
    onf = medfilt(on.astype(float), 5) > 0.5                      # 20 ms 未満の出入りは除く
    d = np.diff(onf.astype(int))
    m["exit_rate"] = float(np.sum(d == -1) / tr.T)
    starts = np.flatnonzero(d == 1) + 1; ends = np.flatnonzero(d == -1) + 1
    if onf[0]:
        starts = np.r_[0, starts]
    if onf[-1]:
        ends = np.r_[ends, len(onf)]
    dur = (ends[:len(starts)] - starts[:len(ends)]) / FS
    m["dwell_s"] = float(np.mean(dur)) if len(dur) else 0.0
    w = int(WIN_S * FS)
    wr = [np.sqrt(np.mean(np.sum(e[i:i + w] ** 2, 1))) for i in range(0, len(e) - w + 1, w)]
    m["err_cv"] = float(np.std(wr) / np.mean(wr))
    return m


# ------------------------------------------------------------------ 統計の道具
def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]; o = np.argsort(pv); n = len(pv)
    r = pv[o] * n / (np.arange(n) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(n); qq[o] = np.minimum(r, 1)
    q[ok] = qq
    return q


def _rowcorr(A, B):
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    den = np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (A * B).sum(1) / den


def spearman_full(x, y, rng, n_boot, n_perm):
    """Spearman ρ、並べ替え p（両側）、人をリサンプルしたブートストラップ 95 % CI（行列でまとめて計算）。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return dict(n=n, rho=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    perm = np.argsort(rng.random((n_perm, n)), axis=1)
    null = _rowcorr(np.broadcast_to(rx, (n_perm, n)), ry[perm])
    p = float((np.sum(np.abs(null) >= abs(rho) - 1e-12) + 1) / (n_perm + 1))
    idx = rng.integers(0, n, (n_boot, n))
    bs = _rowcorr(stats.rankdata(x[idx], axis=1), stats.rankdata(y[idx], axis=1))
    bs = bs[np.isfinite(bs)]
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) > n_boot / 2 else (np.nan, np.nan))
    return dict(n=n, rho=rho, lo=float(lo), hi=float(hi), p=p)


def residualize(rows, k, factors):
    """rows の k を、factors（列名のタプルの一覧）のダミー変数で最小二乗回帰した残差。"""
    y = np.array([fnum(r.get(k)) for r in rows])
    ok = np.isfinite(y)
    cols = [np.ones(len(rows))]
    for fac in factors:
        lv = sorted({tuple(r[f] for f in fac) for r in rows})
        for l in lv[1:]:
            cols.append(np.array([tuple(r[f] for f in fac) == l for r in rows], float))
    X = np.column_stack(cols)
    res = np.full(len(rows), np.nan)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    res[ok] = y[ok] - X[ok] @ beta
    rank = np.linalg.matrix_rank(X[ok])
    return res, rank


def partial_r(rx, ry, rank):
    ok = np.isfinite(rx) & np.isfinite(ry)
    n = int(ok.sum())
    if n - rank < 5:
        return dict(n=n, r=np.nan, lo=np.nan, hi=np.nan, p=np.nan, df=n - rank)
    r = float(np.corrcoef(rx[ok], ry[ok])[0, 1])
    df = n - rank - 1                                   # 取り除いた自由度 + 傾き 1
    tval = r * np.sqrt(df / max(1 - r ** 2, 1e-12))
    p = float(2 * stats.t.sf(abs(tval), df))
    z = np.arctanh(np.clip(r, -0.999999, 0.999999)); se = 1 / np.sqrt(max(df - 1, 1))
    return dict(n=n, r=r, lo=float(np.tanh(z - 1.96 * se)), hi=float(np.tanh(z + 1.96 * se)), p=p, df=df)


def icc1(groups):
    g = [np.asarray(x, float)[np.isfinite(x)] for x in groups]
    g = [x for x in g if len(x) >= 2]
    if len(g) < 3:
        return np.nan
    n = np.array([len(x) for x in g]); N = n.sum(); a = len(g)
    grand = np.concatenate(g).mean()
    msb = np.sum(n * (np.array([x.mean() for x in g]) - grand) ** 2) / (a - 1)
    msw = np.sum([np.sum((x - x.mean()) ** 2) for x in g]) / (N - a)
    k0 = (N - np.sum(n ** 2) / N) / (a - 1)
    den = msb + (k0 - 1) * msw
    return float((msb - msw) / den) if den > 0 else np.nan


def fmt(x, f="{:.2f}"):
    return f.format(x) if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else "–"


def md(rows, cols):
    h = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        h += "| " + " | ".join((fmt(r.get(c[1]), c[2]) if len(c) > 2 else str(r.get(c[1], "–"))) for c in cols) + " |\n"
    return h


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prelim", default="prelim_relaxed")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = resolve(a.out)
    pout = os.path.join(out, "perf_coordination"); os.makedirs(pout, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    logf = open(os.path.join(pout, "perf_log.md"), "w", encoding="utf-8")

    def lg(s=""):
        print(s); logf.write(s + "\n"); logf.flush()

    lg(f"# perf_coordination（{time.strftime('%Y-%m-%d %H:%M:%S')}）\n")
    lg(f"config: {a.config} / out: {out} / prelim: {a.prelim} / n_boot {a.n_boot} / n_perm {a.n_perm} / seed {a.seed}\n")

    # ---- 試行の対応づけ
    pre = {r["key"]: r for r in csv.DictReader(open(os.path.join(out, a.prelim, "trials_prelim.csv"), encoding="utf-8"))}
    tc = {}
    for r in csv.DictReader(open(os.path.join(out, "between", "trials.csv"), encoding="utf-8")):
        if r.get("kind") == "track":
            tc[f"{r['participant']}_{fnum(r['dpi']):g}_{r['band']}_s{int(fnum(r['seed']))}"] = r
    dirs = {}
    for e in cfg.SESSIONS:
        for t, meta in C.list_trials(resolve(e["dir"]), kinds=("track",)):
            if cfg.BANDS and meta["task"].get("band") not in cfg.BANDS:
                continue
            dirs[f"{e['participant']}_{float(e['dpi']):g}_{meta['task']['band']}_s{int(meta['task']['seed'])}"] = t
    keys = sorted(set(pre) & set(dirs))
    lg(f"試行: {len(keys)} 本（{a.prelim} の除外を引き継ぐ）\n")

    # ---- 1. エイムの質
    R = []
    tmp = os.path.join(out, "_perf_tmp")
    t0 = time.time()
    for k in keys:
        p = pre[k]
        tr = C.Trial(Args(dirs[k], out=tmp))
        m = perf_metrics(tr)
        b = tc.get(k, {})
        row = {"key": k, "participant": p["participant"], "dpi": fnum(p["dpi"]), "band": p["band"],
               "seed": int(fnum(p["seed"])), "skill": SKILL.get(p["participant"], np.nan),
               "rms_2d": fnum(b.get("rms_2d")), "on_target_pct": fnum(b.get("on_target_pct")),
               "remnant": np.nanmean([fnum(b.get("remnant_x")), fnum(b.get("remnant_y"))]),
               "delay_ms": np.nanmean([fnum(b.get("delay_x_ms")), fnum(b.get("delay_y_ms"))]),
               "fb_hz": np.nanmean([fnum(b.get("fb_x_hz")), fnum(b.get("fb_y_hz"))]), **m}
        for ck, _, _ in COORD:
            row[ck] = fnum(p.get(ck))
        R.append(row)
    lg(f"エイムの質を計算（{time.time() - t0:.0f} s）\n")
    with open(os.path.join(pout, "trials_perf.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(R[0])); w.writeheader()
        for r in R:
            w.writerow(r)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    people = sorted({r["participant"] for r in R}, key=str.lower)
    bands = [b for b in ("slow", "fast") if any(r["band"] == b for r in R)]
    dpis = sorted({r["dpi"] for r in R})

    def cell(p=None, band=None, dpi=None):
        return [r for r in R if (p is None or r["participant"] == p) and (band is None or r["band"] == band)
                and (dpi is None or r["dpi"] == dpi)]

    def pm(p, band, dpi, k):
        v = np.array([r[k] for r in cell(p, band, dpi)], float)
        return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan

    # ---- 1a. 記述・DPI 効果・個人差・練度
    lg("## 1. エイムの質の指標\n")
    lg("値は 9 人の平均（各人は seed 3 本の平均）。d = 800 − 400（人ごと）、dz = mean(d)/SD(d)。"
       "「良」= 800 で良くなった人数。ICC は (帯域 × DPI) の平均を引いた 1 人 6 本で計算。"
       "τ = 練度（3 群、8 人）との Kendall τ を『良い向き』にそろえた値（+ = 練度が高いほど良い）。\n")
    desc = []
    for band in bands:
        for k, lab, good in PERF:
            v4 = [pm(p, band, dpis[0], k) for p in people]; v8 = [pm(p, band, dpis[-1], k) for p in people]
            d = np.array(v8) - np.array(v4)
            G = []
            for p in people:
                x = []
                for dpi in dpis:
                    mu = np.nanmean([r[k] for r in cell(None, band, dpi)])
                    x += [r[k] - mu for r in cell(p, band, dpi)]
                G.append(np.array(x, float))
            sk = [p for p in people if p in SKILL]
            xs = np.array([np.nanmean([pm(p, band, dd, k) for dd in dpis]) for p in sk])
            ys = np.array([SKILL[p] for p in sk], float)
            tau = stats.kendalltau(xs, ys).statistic * good
            null = [stats.kendalltau(xs, rng.permutation(ys)).statistic * good for _ in range(2000)]
            ptau = float((np.sum(np.abs(null) >= abs(tau) - 1e-12) + 1) / 2001)
            desc.append({"band": band, "k": k, "lab": lab, "m400": float(np.nanmean(v4)), "m800": float(np.nanmean(v8)),
                         "d": float(np.nanmean(d)), "dz": float(np.nanmean(d) / np.nanstd(d, ddof=1)),
                         "n_better": int(np.sum(np.sign(d) == good)),
                         "p_w": float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else np.nan,
                         "icc": icc1(G), "tau": float(tau), "p_tau": ptau})
    lg(md(desc, [("帯域", "band"), ("指標", "lab"), ("400", "m400", "{:.3g}"), ("800", "m800", "{:.3g}"),
                 ("d", "d", "{:+.3g}"), ("dz", "dz", "{:+.2f}"), ("800 で良", "n_better", "{:.0f}"),
                 ("p (Wilcoxon)", "p_w", "{:.3f}"), ("ICC", "icc", "{:+.2f}"),
                 ("τ（練度, 良い向き）", "tau", "{:+.2f}"), ("p(τ)", "p_tau", "{:.3f}"), ("q(τ)", "q_tau", "{:.3f}")]))
    for band in bands:                                    # 練度の τ の BH-FDR（帯域ごと、質の指標の数だけ）
        idx = [i for i, r in enumerate(desc) if r["band"] == band]
        for i, qq in zip(idx, bh([desc[i]["p_tau"] for i in idx])):
            desc[i]["q_tau"] = float(qq)
    with open(os.path.join(pout, "perf_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(desc[0])); w.writeheader()
        for r in desc:
            w.writerow(r)

    # 指標どうしの相関（条件内の残差、試行レベル）: 別の次元を測っているかの確認
    lg("\n### エイムの質の指標どうしの相関（条件と seed を除いた試行レベルの残差、Pearson r）\n")
    fac = [("participant", "band", "dpi"), ("band", "seed")]
    res = {k: residualize(R, k, fac) for k, _, _ in PERF}
    rank = max(v[1] for v in res.values())
    hdr = "| | " + " | ".join(lab for _, lab, _ in PERF) + " |\n|" + "---|" * (len(PERF) + 1) + "\n"
    cm = np.full((len(PERF), len(PERF)), np.nan)
    for i, (ki, li, _) in enumerate(PERF):
        hdr += f"| {li} |"
        for j, (kj, _, _) in enumerate(PERF):
            x, y = res[ki][0], res[kj][0]; ok = np.isfinite(x) & np.isfinite(y)
            cm[i, j] = np.corrcoef(x[ok], y[ok])[0, 1]
            hdr += f" {cm[i, j]:+.2f} |"
        hdr += "\n"
    lg(hdr)

    # ---- 2. 協調との関係（DPI の層: 400 だけ / 800 だけ / 400+800 をまとめて）
    strata = [(f"{d:.0f}", [d]) for d in dpis] + ([("pooled", list(dpis))] if len(dpis) > 1 else [])
    all_b1, all_b2, all_b3, skill_rows = [], [], [], []
    for band in bands:
        for sname, sd in strata:
            # B1 人物間（その層の DPI だけで人の平均を作る）
            for ck, cl, wd in COORD:
                x = [np.nanmean([pm(p, band, d, ck) for d in sd]) for p in people]
                for pk, pl, good in PERF:
                    y = [np.nanmean([pm(p, band, d, pk) for d in sd]) for p in people]
                    s = spearman_full(x, y, rng, a.n_boot, a.n_perm)
                    all_b1.append({"stratum": sname, "band": band, "coord": ck, "clab": cl, "perf": pk, "plab": pl,
                                   "good": good, **s, "rho_good": s["rho"] * good})
            # B2 条件内（その層の試行だけ。(人 × DPI) と seed を除く）
            rows = [r for r in cell(None, band, None) if r["dpi"] in sd]
            fac = [("participant", "dpi"), ("seed",)]
            rp = {pk: residualize(rows, pk, fac) for pk, _, _ in PERF}
            for ck, cl, wd in COORD:
                rx, rk = residualize(rows, ck, fac)
                for pk, pl, good in PERF:
                    s = partial_r(rx, rp[pk][0], max(rk, rp[pk][1]))
                    all_b2.append({"stratum": sname, "band": band, "coord": ck, "clab": cl, "perf": pk, "plab": pl,
                                   "good": good, **s, "r_good": s["r"] * good})
            # 練度（質の指標、その層の DPI の人の平均）
            sk = [p for p in people if p in SKILL]
            ys = np.array([SKILL[p] for p in sk], float)
            for pk, pl, good in PERF:
                xs = np.array([np.nanmean([pm(p, band, d, pk) for d in sd]) for p in sk])
                tau = stats.kendalltau(xs, ys).statistic * good
                null = [stats.kendalltau(xs, rng.permutation(ys)).statistic * good for _ in range(2000)]
                skill_rows.append({"stratum": sname, "band": band, "perf": pk, "plab": pl, "tau": float(tau),
                                   "p": float((np.sum(np.abs(null) >= abs(tau) - 1e-12) + 1) / 2001)})
        # B3 DPI による変化どうし（定義上 400 と 800 の両方を使う）
        for ck, cl, wd in COORD:
            dx = [pm(p, band, dpis[-1], ck) - pm(p, band, dpis[0], ck) for p in people]
            for pk, pl, good in PERF:
                dy = [pm(p, band, dpis[-1], pk) - pm(p, band, dpis[0], pk) for p in people]
                s = spearman_full(dx, dy, rng, a.n_boot, a.n_perm)
                all_b3.append({"stratum": "delta", "band": band, "coord": ck, "clab": cl, "perf": pk, "plab": pl,
                               "good": good, **s, "rho_good": s["rho"] * good})
    for L in (all_b1, all_b2, all_b3, skill_rows):          # BH-FDR は (層 × 帯域) ごと
        for key in sorted({(r["stratum"], r["band"]) for r in L}):
            idx = [i for i, r in enumerate(L) if (r["stratum"], r["band"]) == key]
            for i, qq in zip(idx, bh([L[i]["p"] for i in idx])):
                L[i]["q"] = float(qq)

    # 400 と 800 で関係が違うか（異質性）
    #   B2: 別々の試行から作った r どうしの Fisher z 検定（同じ人を含むので近似）
    #   B1: 人を対にしてリサンプルし、ρ(800) − ρ(400) の分布から CI と p
    het = []
    if len(dpis) == 2:
        d0, d1 = f"{dpis[0]:.0f}", f"{dpis[1]:.0f}"
        for band in bands:
            for ck, cl, wd in COORD:
                x0 = np.array([pm(p, band, dpis[0], ck) for p in people]); x1 = np.array([pm(p, band, dpis[1], ck) for p in people])
                for pk, pl, good in PERF:
                    g = lambda L, st: next(r for r in L if r["stratum"] == st and r["band"] == band
                                           and r["coord"] == ck and r["perf"] == pk)
                    b0, b1_, bp = g(all_b2, d0), g(all_b2, d1), g(all_b2, "pooled")
                    z = np.nan; pz = np.nan
                    if np.isfinite(b0["r"]) and np.isfinite(b1_["r"]) and b0["df"] > 2 and b1_["df"] > 2:
                        z = (np.arctanh(b1_["r"]) - np.arctanh(b0["r"])) / np.sqrt(1 / (b0["df"] - 1) + 1 / (b1_["df"] - 1))
                        pz = float(2 * stats.norm.sf(abs(z)))
                    y0 = np.array([pm(p, band, dpis[0], pk) for p in people]); y1 = np.array([pm(p, band, dpis[1], pk) for p in people])
                    ok = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(y0) & np.isfinite(y1)
                    i = rng.integers(0, ok.sum(), (a.n_boot, ok.sum()))
                    X0, X1, Y0, Y1 = x0[ok][i], x1[ok][i], y0[ok][i], y1[ok][i]
                    rk = lambda M: stats.rankdata(M, axis=1)
                    dd = _rowcorr(rk(X1), rk(Y1)) - _rowcorr(rk(X0), rk(Y0))
                    dd = dd[np.isfinite(dd)]
                    c0, c1, cp = g(all_b1, d0), g(all_b1, d1), g(all_b1, "pooled")
                    het.append({"band": band, "coord": ck, "clab": cl, "perf": pk, "plab": pl, "good": good,
                                "b1_rho400": c0["rho"] * good, "b1_rho800": c1["rho"] * good, "b1_pooled": cp["rho"] * good,
                                "b1_diff": (c1["rho"] - c0["rho"]) * good,
                                "b1_diff_p": float(min(1, 2 * min(np.mean(dd <= 0), np.mean(dd >= 0)))) if len(dd) else np.nan,
                                "b2_r400": b0["r"] * good, "b2_r800": b1_["r"] * good, "b2_pooled": bp["r"] * good,
                                "b2_diff": (b1_["r"] - b0["r"]) * good if np.isfinite(z) else np.nan, "b2_diff_p": pz,
                                "b2_p400": b0["p"], "b2_p800": b1_["p"], "b2_ppool": bp["p"]})
        for band in bands:
            for key in ("b1_diff_p", "b2_diff_p"):
                idx = [i for i, r in enumerate(het) if r["band"] == band]
                for i, qq in zip(idx, bh([het[i][key] for i in idx])):
                    het[i][key.replace("_p", "_q")] = float(qq)

    def save(L, name):
        with open(os.path.join(pout, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(L[0])); w.writeheader()
            for r in L:
                w.writerow(r)
    save(all_b1, "b1_between.csv"); save(all_b2, "b2_within_condition.csv"); save(all_b3, "b3_dpi_change.csv")
    save(skill_rows, "skill_by_dpi.csv")
    if het:
        save(het, "dpi_heterogeneity.csv")

    lg("\n## 2. 協調とエイムの質の関係（DPI の層ごと）\n")
    lg("層: 400 = 400 の試行だけ、800 = 800 の試行だけ、pooled = 400 と 800 をまとめる（DPI の差そのものは除く）。"
       "符号は「良い向き」にそろえた（+ = その協調の指標が大きいほどエイムが良い）。"
       f"q = (層 × 帯域) ごとの BH-FDR（協調 {len(COORD)} × 質 {len(PERF)} = {len(COORD) * len(PERF)} 検定）。\n")
    lg("\n### 層ごとの要約（p < 0.05 の組の数 / 偶然に期待される数、最小の q）\n")
    summ = []
    for lev, L in (("B1 人物間", all_b1), ("B2 条件内", all_b2), ("B3 Δ", all_b3)):
        for key in sorted({(r["stratum"], r["band"]) for r in L}, key=lambda k: (k[1] != "slow", k[0])):
            x = [r for r in L if (r["stratum"], r["band"]) == key and np.isfinite(r["p"])]
            summ.append({"lev": lev, "stratum": key[0], "band": key[1], "n_tests": len(x),
                         "n_sig": sum(r["p"] < 0.05 for r in x), "expect": 0.05 * len(x),
                         "min_q": min(r["q"] for r in x), "n_obs": x[0]["n"] if x else np.nan})
    lg(md(summ, [("水準", "lev"), ("層", "stratum"), ("帯域", "band"), ("観測数", "n_obs", "{:.0f}"),
                 ("検定", "n_tests", "{:.0f}"), ("p<0.05", "n_sig", "{:.0f}"), ("期待", "expect", "{:.1f}"),
                 ("最小 q", "min_q", "{:.3f}")]))
    for title, L, rk in (("B1 人物間（人の平均どうし、9 人）", all_b1, "rho_good"),
                         ("B2 条件内（試行間のばらつきどうし、残差相関）", all_b2, "r_good"),
                         ("B3 DPI による変化どうし（Δ800−400、9 人）", all_b3, "rho_good")):
        lg(f"\n### {title}\n")
        for key in sorted({(r["stratum"], r["band"]) for r in L}, key=lambda k: (k[1] != "slow", k[0])):
            lg(f"\n**{key[1]} / 層 {key[0]}**（行 = 協調、列 = エイムの質。* = p < 0.05、** = q < 0.10）\n")
            hdr = "| 協調 \\ 質 | " + " | ".join(pl for _, pl, _ in PERF) + " |\n|" + "---|" * (len(PERF) + 1) + "\n"
            for ck, cl, _ in COORD:
                hdr += f"| {cl} |"
                for pk, _, _ in PERF:
                    r = next(x for x in L if (x["stratum"], x["band"]) == key and x["coord"] == ck and x["perf"] == pk)
                    mark = "**" if r.get("q", 1) < 0.10 else ("*" if r["p"] < 0.05 else "")
                    hdr += f" {fmt(r[rk], '{:+.2f}')}{mark} |"
                hdr += "\n"
            lg(hdr)
        top = sorted([x for x in L if np.isfinite(x["p"])], key=lambda x: x["p"])[:15]
        lg("\n上位（全層、p の小さい順。CI は良い向きにそろえた）:\n")
        disp = []
        for r in top:
            lo, hi = (r["lo"], r["hi"]) if r["good"] > 0 else (-r["hi"], -r["lo"])
            disp.append({**r, "lo_g": lo, "hi_g": hi})
        lg(md(disp, [("層", "stratum"), ("帯域", "band"), ("協調", "clab"), ("質", "plab"), ("n", "n", "{:.0f}"),
                     ("相関（良い向き）", rk, "{:+.2f}"), ("CI 下", "lo_g", "{:+.2f}"), ("CI 上", "hi_g", "{:+.2f}"),
                     ("p", "p", "{:.4f}"), ("q", "q", "{:.3f}")]))

    if het:
        lg("\n### 400 と 800 で関係が違うか（異質性）\n")
        lg("B2: r(800) − r(400) の Fisher z 検定（同じ人を含むので近似）。B1: 人を対にしたブートストラップ。"
           "q = 帯域ごとの BH-FDR（110 検定）。下は B2 の差の p が小さい順の 15 組。\n")
        hs = sorted([h for h in het if np.isfinite(h["b2_diff_p"])], key=lambda h: h["b2_diff_p"])[:15]
        lg(md(hs, [("帯域", "band"), ("協調", "clab"), ("質", "plab"),
                   ("B2 r400", "b2_r400", "{:+.2f}"), ("B2 r800", "b2_r800", "{:+.2f}"), ("B2 pooled", "b2_pooled", "{:+.2f}"),
                   ("差 p", "b2_diff_p", "{:.4f}"), ("差 q", "b2_diff_q", "{:.3f}"),
                   ("B1 ρ400", "b1_rho400", "{:+.2f}"), ("B1 ρ800", "b1_rho800", "{:+.2f}"), ("B1 差 p", "b1_diff_p", "{:.3f}")]))
        n2 = sum(h["b2_diff_p"] < 0.05 for h in het if np.isfinite(h["b2_diff_p"]))
        n1 = sum(h["b1_diff_p"] < 0.05 for h in het if np.isfinite(h["b1_diff_p"]))
        lg(f"\n差の p < 0.05: B2 {n2} / {len(het)} 組、B1 {n1} / {len(het)} 組（偶然に期待される数 {0.05 * len(het):.0f}）。"
           f"最小 q: B2 {min(h['b2_diff_q'] for h in het):.3f}、B1 {min(h['b1_diff_q'] for h in het):.3f}\n")

        lg("\n### 練度（Kendall τ、良い向き）の層ごとの値\n")
        st = []
        for band in bands:
            for pk, pl, good in PERF:
                row = {"band": band, "plab": pl}
                for sname, _ in strata:
                    r = next(x for x in skill_rows if x["stratum"] == sname and x["band"] == band and x["perf"] == pk)
                    row[f"t_{sname}"] = r["tau"]; row[f"p_{sname}"] = r["p"]; row[f"q_{sname}"] = r["q"]
                st.append(row)
        cols = [("帯域", "band"), ("質", "plab")]
        for sname, _ in strata:
            cols += [(f"τ {sname}", f"t_{sname}", "{:+.2f}"), (f"p {sname}", f"p_{sname}", "{:.3f}"), (f"q {sname}", f"q_{sname}", "{:.2f}")]
        lg(md(st, cols))

    plots(R, all_b1, all_b2, all_b3, people, bands, dpis, pout, cm, [s for s, _ in strata])
    json.dump({"perf_summary": desc, "b1": all_b1, "b2": all_b2, "b3": all_b3, "skill": skill_rows, "heterogeneity": het},
              open(os.path.join(pout, "perf_coordination.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=C._jsonable)
    lg(f"\n→ {pout}")
    logf.close()


def plots(R, b1, b2, b3, people, bands, dpis, pout, cm, strata):
    plt = C.setup_plot()
    # 図 1〜3: ヒートマップ（行 = 層、列 = 帯域）
    for name, L, rk, sts in (("fig1_between", b1, "rho_good", strata), ("fig2_within_condition", b2, "r_good", strata),
                             ("fig3_dpi_change", b3, "rho_good", ["delta"])):
        fig, axes = plt.subplots(len(sts), len(bands), figsize=(8.5 * len(bands), 4.6 * len(sts)), squeeze=False)
        for i, st in enumerate(sts):
            for ax, band in zip(axes[i], bands):
                sel = lambda ck, pk: next(x for x in L if x["stratum"] == st and x["band"] == band
                                          and x["coord"] == ck and x["perf"] == pk)
                M = np.array([[sel(ck, pk)[rk] for pk, _, _ in PERF] for ck, _, _ in COORD])
                im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
                for a_, (ck, _, _) in enumerate(COORD):
                    for b_, (pk, _, _) in enumerate(PERF):
                        r = sel(ck, pk)
                        mark = "**" if r.get("q", 1) < 0.10 else ("*" if r["p"] < 0.05 else "")
                        ax.text(b_, a_, f"{M[a_, b_]:+.2f}{mark}" if np.isfinite(M[a_, b_]) else "–",
                                ha="center", va="center", fontsize=6.5)
                ax.set_xticks(range(len(PERF)))
                ax.set_xticklabels([pl for _, pl, _ in PERF] if i == len(sts) - 1 else [""] * len(PERF),
                                   rotation=40, ha="right", fontsize=7)
                ax.set_yticks(range(len(COORD))); ax.set_yticklabels([cl for _, cl, _ in COORD], fontsize=7.5)
                lab = {"pooled": "400+800", "delta": "Δ800−400"}.get(st, f"{st} DPI のみ")
                ax.set_title(f"{band} / {lab}（+ = 協調の指標が大きいほど良い。* p<.05、** q<.10）", fontsize=9)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.6)
        fig.savefig(os.path.join(pout, f"{name}.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

    # 図 4: 質の指標の人 × DPI（帯域ごと）
    fig, axes = plt.subplots(len(bands), len(PERF), figsize=(2.2 * len(PERF), 2.6 * len(bands)), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for i, band in enumerate(bands):
        for j, (pk, pl, good) in enumerate(PERF):
            ax = axes[i][j]
            for n, p in enumerate(people):
                ys = [np.nanmean([r[pk] for r in R if r["participant"] == p and r["band"] == band and r["dpi"] == d])
                      for d in dpis]
                ax.plot(range(len(dpis)), ys, "o-", ms=3, lw=1, color=cmap(n % 10), label=p)
            ax.set_xticks(range(len(dpis))); ax.set_xticklabels([f"{d:.0f}" for d in dpis], fontsize=7)
            ax.set_title(f"{band}: {pl}{'↑' if good > 0 else '↓'}", fontsize=7.5)
            ax.tick_params(labelsize=6.5)
    axes[0][-1].legend(fontsize=5.5, loc="center left", bbox_to_anchor=(1.02, .5))
    fig.tight_layout(); fig.savefig(os.path.join(pout, "fig4_perf_by_dpi.png"), dpi=130); plt.close(fig)

    # 図 5: 質の指標どうしの相関
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(PERF)):
        for j in range(len(PERF)):
            ax.text(j, i, f"{cm[i, j]:+.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_xticks(range(len(PERF))); ax.set_xticklabels([pl for _, pl, _ in PERF], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(PERF))); ax.set_yticklabels([pl for _, pl, _ in PERF], fontsize=7)
    ax.set_title("エイムの質の指標どうしの相関（試行レベルの残差）", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(os.path.join(pout, "fig5_perf_corr.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
