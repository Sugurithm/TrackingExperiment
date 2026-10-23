"""統計（人 = 単位）。prelim_inference / perf_coordination / b_between / skill_eval.stats の重複をまとめた。"""
import itertools
import numpy as np
from scipy import stats as ss

from .. import config as K


def finite(v):
    v = np.asarray(v, float)
    return v[np.isfinite(v)]


def bh(p):
    """Benjamini–Hochberg の q 値。"""
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(len(pv))
    qq[o] = np.minimum(r, 1)
    q[ok] = qq
    return q


def boot_ci(v, rng, n=K.N_BOOT, stat=np.mean, alpha=0.05):
    v = finite(v)
    if len(v) < 3:
        return np.nan, np.nan
    b = np.array([stat(rng.choice(v, len(v))) for _ in range(n)])
    return float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2)))


def tost(d, delta, alpha=K.ALPHA):
    """1 標本（人ごとの差 d）の同等性検定（TOST）。

    H0: |平均の差| ≥ delta。両側の t 検定の p の大きい方を返す。p < alpha なら「±delta の範囲で同じ」と言える。
    あわせて 90 % CI（TOST に対応）と 95 % CI、通常の t 検定の p も返す。
    """
    d = finite(d)
    n = len(d)
    if n < 3 or not np.isfinite(delta):
        return {"n": n, "mean": float(np.mean(d)) if n else np.nan, "p_tost": np.nan, "p_diff": np.nan,
                "ci90": (np.nan, np.nan), "ci95": (np.nan, np.nan), "dz": np.nan}
    m, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    se = sd / np.sqrt(n) if sd > 0 else 1e-12
    df = n - 1
    p_lo = 1 - ss.t.cdf((m + delta) / se, df)      # H0: m ≤ −delta
    p_hi = ss.t.cdf((m - delta) / se, df)          # H0: m ≥ +delta
    t90, t95 = ss.t.ppf(0.95, df), ss.t.ppf(0.975, df)
    return {"n": n, "mean": m, "sd": sd, "p_tost": float(max(p_lo, p_hi)),
            "p_diff": float(2 * (1 - ss.t.cdf(abs(m) / se, df))),
            "ci90": (m - t90 * se, m + t90 * se), "ci95": (m - t95 * se, m + t95 * se),
            "dz": m / sd if sd > 0 else np.nan, "n_pos": int(np.sum(d > 0)), "n_neg": int(np.sum(d < 0))}


def verdict(t, delta):
    """TOST と差の検定から: 同等 / 差あり / 差あり（ただし幅の内）/ 判定不能。"""
    if not np.isfinite(t.get("p_tost", np.nan)):
        return "–"
    eq = t["p_tost"] < K.ALPHA
    lo, hi = t["ci95"]
    diff = (lo > 0) or (hi < 0)
    if eq and diff:
        return "差はあるが小さい（幅の内）"
    if eq:
        return "同等（変わらない）"
    if diff and (abs(t["mean"]) >= delta):
        return "変化あり"
    if diff:
        return "差あり（同等性は示せず）"
    return "判定不能（ばらつき大）"


def icc1(groups):
    """ICC(1,1)（一元配置、群の大きさが違ってもよい）。groups = [配列, ...]（人ごとの試行の値）。"""
    g = [finite(x) for x in groups]
    g = [x for x in g if len(x) >= 1]
    k = len(g)
    N = sum(len(x) for x in g)
    if k < 2 or N - k < 1:
        return np.nan
    grand = np.mean(np.concatenate(g))
    msb = sum(len(x) * (x.mean() - grand) ** 2 for x in g) / (k - 1)
    msw = sum(((x - x.mean()) ** 2).sum() for x in g) / (N - k)
    n0 = (N - sum(len(x) ** 2 for x in g) / N) / (k - 1)
    den = msb + (n0 - 1) * msw
    return float((msb - msw) / den) if den > 0 else np.nan


def icc3(M):
    """ICC(3,1) 一貫性（人 × 条件の行列。各セル = その条件での人の平均）。欠けた行は除く。

    条件ごとの平均の差（DPI の主効果）は除いたうえで、「人の順序と間隔が条件をまたいで保たれるか」。
    """
    M = np.asarray(M, float)
    M = M[np.isfinite(M).all(1)]
    n, k = M.shape if M.ndim == 2 else (0, 0)
    if n < 3 or k < 2:
        return np.nan
    gm = M.mean()
    ssr = k * np.sum((M.mean(1) - gm) ** 2)
    ssc = n * np.sum((M.mean(0) - gm) ** 2)
    sse = np.sum((M - gm) ** 2) - ssr - ssc
    msr, mse = ssr / (n - 1), sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse)) if (msr + (k - 1) * mse) > 0 else np.nan


def var_components(rows, field, person="participant", cond="cond"):
    """試行の値を 人・条件・人×条件・残差（seed）に分ける（セルの大きさがそろっていなくてもよい近似）。

    返り値: 各成分の分散と割合。「人×条件 / 残差」が 1 前後なら、条件に対する反応の個人差は
    試行ごとのばらつき以上には無い（＝条件が変わっても人ごとの差の構造は保たれる）。
    """
    data = [(r[person], r[cond], float(r[field])) for r in rows
            if r.get(field) is not None and np.isfinite(float(r[field]))]
    if len(data) < 6:
        return {}
    P = sorted({d[0] for d in data}); C = sorted({d[1] for d in data})
    cell = {}
    for p, c, v in data:
        cell.setdefault((p, c), []).append(v)
    # 残差（セル内の分散のプール）
    dfw = sum(len(v) - 1 for v in cell.values())
    s2e = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in cell.values()) / dfw if dfw > 0 else np.nan
    nbar = np.mean([len(v) for v in cell.values()])
    M = np.full((len(P), len(C)), np.nan)
    for (p, c), v in cell.items():
        M[P.index(p), C.index(c)] = np.mean(v)
    M = M[np.isfinite(M).all(1)]
    if len(M) < 3:
        return {}
    gm = M.mean()
    r_eff = M.mean(1) - gm
    c_eff = M.mean(0) - gm
    inter = M - gm - r_eff[:, None] - c_eff[None, :]
    n, k = M.shape
    s2_pc_obs = np.sum(inter ** 2) / ((n - 1) * (k - 1))
    s2_pc = max(s2_pc_obs - s2e / nbar, 0.0)
    s2_p = max(np.sum(r_eff ** 2) / (n - 1) - s2_pc_obs / k, 0.0)
    s2_c = float(np.sum(c_eff ** 2) / (k - 1)) if k > 1 else 0.0
    tot = s2_p + s2_c + s2_pc + s2e
    return {"var_person": s2_p, "var_cond": s2_c, "var_person_x_cond": s2_pc, "var_resid": s2e,
            "pct_person": 100 * s2_p / tot if tot > 0 else np.nan,
            "pct_cond": 100 * s2_c / tot if tot > 0 else np.nan,
            "pct_person_x_cond": 100 * s2_pc / tot if tot > 0 else np.nan,
            "pct_resid": 100 * s2e / tot if tot > 0 else np.nan,
            "pxc_over_resid": float(s2_pc_obs / (s2e / nbar)) if s2e > 0 else np.nan, "n_people": n}


def perm_group(a, b, rng, n_perm=K.N_PERM):
    """2 群の平均の差の並べ替え検定（両側）。組み合わせが少なければ全列挙（正確な p）。"""
    a, b = finite(a), finite(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    allv = np.r_[a, b]
    obs = a.mean() - b.mean()
    n, na = len(allv), len(a)
    from math import comb
    if comb(n, na) <= n_perm:
        diffs = []
        for idx in itertools.combinations(range(n), na):
            m = np.zeros(n, bool); m[list(idx)] = True
            diffs.append(allv[m].mean() - allv[~m].mean())
        diffs = np.array(diffs)
    else:
        diffs = np.empty(n_perm)
        for i in range(n_perm):
            pidx = rng.permutation(n)
            diffs[i] = allv[pidx[:na]].mean() - allv[pidx[na:]].mean()
    p = float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12))
    return float(obs), p


def hedges_g(a, b):
    a, b = finite(a), finite(b)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if sp <= 0:
        return np.nan
    J = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    return float((a.mean() - b.mean()) / sp * J)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan, np.nan
    r, p = ss.spearmanr(x[ok], y[ok])
    return float(r), float(p)


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.ptp(x[ok]) <= 0:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])
