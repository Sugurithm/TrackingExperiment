"""人を単位にしたまとめ（H3 不変性・個人差・FPS / 非 FPS）

旧 prelim_inference（DPI 効果・ICC）、b_between（ICC・判別比・並べ替え）、w_within（DPI の傾き）、
perf_coordination（群・相関）、skill_eval.stats（群の検定）で重複していた部分を 1 つにした。
"""
import itertools
import numpy as np

from .. import config as K
from ..lib import stats as S
from . import metrics as MT


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return v


def cells(rows, cols):
    """人 × 条件（DPI × 帯域）ごとの平均・SD・本数。"""
    out = {}
    for r in rows:
        k = (r["participant"], r["dpi"], r["band"])
        out.setdefault(k, []).append(r)
    res = []
    for (p, d, b), rs in sorted(out.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        row = {"participant": p, "group": rs[0]["group"], "dpi": d, "band": b, "cond": f"{int(d)}_{b}", "n": len(rs)}
        for c in cols:
            v = S.finite([_f(r.get(c)) for r in rs])
            row[c] = float(np.mean(v)) if len(v) else np.nan
            row[c + "__sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
            row[c + "__n"] = len(v)
        res.append(row)
    return res


def _cell(cell_rows, p, dpi, band, c):
    for r in cell_rows:
        if r["participant"] == p and r["dpi"] == dpi and r["band"] == band:
            return r.get(c, np.nan), r.get(c + "__sd", np.nan), r.get(c + "__n", 0)
    return np.nan, np.nan, 0


def bias_table(bias_rows, cols):
    """擬似データの偏り（擬似 − 元）を 人 × 帯域 × (元の DPI → 先の DPI) で平均。"""
    out = {}
    for r in bias_rows:
        k = (r["participant"], r["band"], r["dpi_from"], r["dpi_to"])
        out.setdefault(k, []).append(r)
    res = {}
    for k, rs in out.items():
        res[k] = {c: float(np.mean(S.finite([_f(x.get(c)) for x in rs]))) if len(S.finite([_f(x.get(c)) for x in rs]))
                  else np.nan for c in cols}
    return res


def invariance(cell_rows, bias, cols):
    """DPI（帯域ごと）と帯域（DPI ごと）で、人ごとの差 d を出して同等性を検定する。"""
    people = sorted({r["participant"] for r in cell_rows})
    dpis = sorted({r["dpi"] for r in cell_rows})
    bands = [b for b in K.BANDS if b in {r["band"] for r in cell_rows}]
    out = []
    comps = []
    for b in bands:
        for d0, d1 in itertools.combinations(dpis, 2):
            comps.append(("DPI", f"{int(d1)} − {int(d0)}", b, ((d0, b), (d1, b))))
    for d in dpis:
        if len(bands) == 2:
            comps.append(("帯域", f"{bands[1]} − {bands[0]}", f"{int(d)} DPI", ((d, bands[0]), (d, bands[1]))))
    for c in cols:
        delta = MT.sesoi(c)
        for fac, contrast, stratum, ((da, ba), (db, bb)) in comps:
            dd, draw, bs, sd_pool, a_v, b_v = [], [], [], [], [], []
            for p in people:
                va, sa, na = _cell(cell_rows, p, da, ba, c)
                vb, sb, nb = _cell(cell_rows, p, db, bb, c)
                if not (np.isfinite(va) and np.isfinite(vb)):
                    continue
                bias_p = np.nan
                if fac == "DPI" and bias:
                    bias_p = bias.get((p, ba, da, db), {}).get(c, np.nan)
                corr = bias_p if (K.BIAS_CORRECT and np.isfinite(bias_p)) else 0.0
                draw.append(vb - va)
                dd.append(vb - va - corr)
                bs.append(bias_p)
                a_v.append(va); b_v.append(vb)
                sp = np.nanmean([sa ** 2, sb ** 2]) if np.isfinite(sa) or np.isfinite(sb) else np.nan
                n_ = max(min(na, nb), 1)
                sd_pool.append(np.sqrt(sp) * np.sqrt(2 / n_) * 1.96 if np.isfinite(sp) else np.nan)
            t = S.tost(dd, delta)
            mdc = np.array(sd_pool)
            dd_a = np.array(dd)
            n_changed = int(np.sum(np.abs(dd_a) > mdc)) if len(dd_a) else 0
            rho, prho = S.spearman(a_v, b_v)
            out.append({"metric": c, "label": MT.label(c), "hyp": MT.HYP.get(c, ""), "main": c in MT.MAIN,
                        "factor": fac, "contrast": contrast, "stratum": stratum, "n": t["n"],
                        "mean_a": float(np.mean(a_v)) if a_v else np.nan,
                        "mean_b": float(np.mean(b_v)) if b_v else np.nan,
                        "d": t["mean"], "d_raw": float(np.mean(draw)) if draw else np.nan,
                        "bias": float(np.nanmean(bs)) if bs and np.isfinite(bs).any() else np.nan,
                        "ci95_lo": t["ci95"][0], "ci95_hi": t["ci95"][1],
                        "ci90_lo": t["ci90"][0], "ci90_hi": t["ci90"][1], "dz": t.get("dz", np.nan),
                        "p_diff": t["p_diff"], "p_tost": t["p_tost"], "sesoi": delta,
                        "n_pos": t.get("n_pos", 0), "n_neg": t.get("n_neg", 0),
                        "n_changed_mdc": n_changed, "rho_rank": rho, "p_rho": prho,
                        "verdict": S.verdict(t, delta)})
    # 多重比較: 主要な指標 × 比較 の中で BH
    idx = [i for i, r in enumerate(out) if r["main"]]
    for key in ("p_diff", "p_tost"):
        q = S.bh([out[i][key] for i in idx])
        for j, i in enumerate(idx):
            out[i]["q_" + key[2:]] = q[j]
    return out


def person_changes(rows, cell_rows, bias, cols):
    """人ごとの判定: その人の中で DPI（帯域ごと）・帯域（DPI ごと）を変えたとき指標が変わったか。

    SE は _seed_adjusted_sd の SD から作る（DPI の比較は同じ軌道どうしの対応をとる）。
    同等 = 差の 90 % CI が ±幅 に収まる / 変化 = 95 % CI が 0 を含まず |d| ≥ 幅 /
    差あり（小）= 0 を含まないが幅の内 / 判定不能 = それ以外。
    """
    from scipy import stats as ss
    people = sorted({r["participant"] for r in cell_rows})
    dpis = sorted({r["dpi"] for r in cell_rows})
    bands = [b for b in K.BANDS if b in {r["band"] for r in cell_rows}]
    comps = [("DPI", b, (dpis[0], b), (dpis[-1], b)) for b in bands] if len(dpis) >= 2 else []
    if len(bands) == 2:
        comps += [("帯域", f"{int(d)} DPI", (d, bands[0]), (d, bands[1])) for d in dpis]
    out = []
    for p in people:
        for c in cols:
            delta = MT.sesoi(c)
            prow = [r for r in rows if r["participant"] == p]
            sds = {"DPI": _seed_adjusted_sd(prow, c, True), "帯域": _seed_adjusted_sd(prow, c, False)}
            for fac, st, (da, ba), (db, bb) in comps:
                va, _, na = _cell(cell_rows, p, da, ba, c)
                vb, _, nb = _cell(cell_rows, p, db, bb, c)
                if not (np.isfinite(va) and np.isfinite(vb)) or na < 1 or nb < 1:
                    continue
                corr = 0.0
                if fac == "DPI" and K.BIAS_CORRECT and bias:
                    bb_ = bias.get((p, ba, da, db), {}).get(c, np.nan)
                    corr = bb_ if np.isfinite(bb_) else 0.0
                d = vb - va - corr
                sd, df = sds[fac]
                se = sd * np.sqrt(1 / na + 1 / nb) if np.isfinite(sd) else np.nan
                if np.isfinite(se) and se > 0:
                    t90, t95 = ss.t.ppf(0.95, df), ss.t.ppf(0.975, df)
                    lo90, hi90, lo95, hi95 = d - t90 * se, d + t90 * se, d - t95 * se, d + t95 * se
                else:
                    lo90 = hi90 = lo95 = hi95 = np.nan
                if not np.isfinite(lo90):
                    v = "–"
                elif lo90 > -delta and hi90 < delta:
                    v = "同等" if (lo95 <= 0 <= hi95) else "差あり（小）"
                elif (lo95 > 0 or hi95 < 0) and abs(d) >= delta:
                    v = "変化"
                elif lo95 > 0 or hi95 < 0:
                    v = "差あり（小）"
                else:
                    v = "判定不能"
                out.append({"participant": p, "group": group_of_row(cell_rows, p), "metric": c, "label": MT.label(c),
                            "main": c in MT.MAIN, "factor": fac, "stratum": st, "a": va, "b": vb, "d": d,
                            "se": se, "ci95_lo": lo95, "ci95_hi": hi95, "sesoi": delta, "verdict": v})
    return out


def _seed_adjusted_sd(prow, c, paired=True):
    """1 人ぶんの試行から、比較に使う試行のばらつき（SD）と自由度。

    paired = True（DPI の比較）: 帯域ごとに DPI × seed の表を作り、DPI と seed の効果を除いた残差をプールする。
      同じ帯域・同じ seed は DPI が違っても同じ軌道なので、軌道ごとの癖を除いて比べられる
      （自由度 = Σ_帯域 (DPI 数 − 1)(seed 数 − 1)）。
    paired = False（帯域の比較。slow と fast は軌道が違う）: セル内の SD をプールする。
    """
    def pooled(groups):
        ss_, df = 0.0, 0
        for v in groups:
            v = np.asarray([x for x in v if np.isfinite(x)])
            if len(v) > 1:
                ss_ += float(np.sum((v - v.mean()) ** 2)); df += len(v) - 1
        return (float(np.sqrt(ss_ / df)) if df > 0 else np.nan), df

    if not paired:
        cells_ = {}
        for r in prow:
            cells_.setdefault(r["cond"], []).append(_f(r.get(c)))
        return pooled(cells_.values())
    ss_, df = 0.0, 0
    for band in sorted({r["band"] for r in prow}):
        rb = [r for r in prow if r["band"] == band]
        dpis = sorted({r["dpi"] for r in rb}); seeds = sorted({str(r["seed"]) for r in rb})
        M = np.full((len(dpis), len(seeds)), np.nan)
        for r in rb:
            v = _f(r.get(c))
            if np.isfinite(v):
                M[dpis.index(r["dpi"]), seeds.index(str(r["seed"]))] = v
        M = M[:, np.isfinite(M).all(0)]
        if M.shape[0] >= 2 and M.shape[1] >= 2:
            res = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
            ss_ += float(np.sum(res ** 2)); df += (M.shape[0] - 1) * (M.shape[1] - 1)
    if df >= 2:
        return float(np.sqrt(ss_ / df)), df
    return _seed_adjusted_sd(prow, c, paired=False)


def group_of_row(cell_rows, p):
    return next((r["group"] for r in cell_rows if r["participant"] == p), None)


def individuality(rows, cell_rows, cols):
    """個人差: 分散成分（人・条件・人×条件・残差）、条件をまたいだ ICC(3,1)、条件内の ICC(1,1)。"""
    conds = sorted({r["cond"] for r in cell_rows})
    people = sorted({r["participant"] for r in cell_rows})
    out = []
    for c in cols:
        vc = S.var_components(rows, c)
        Mx = np.array([[_cell(cell_rows, p, int(cc.split("_")[0]), cc.split("_")[1], c)[0] for cc in conds]
                       for p in people])
        iccs = []
        for cc in conds:
            g = [[_f(r.get(c)) for r in rows if r["participant"] == p and r["cond"] == cc] for p in people]
            iccs.append(S.icc1(g))
        out.append({"metric": c, "label": MT.label(c), "hyp": MT.HYP.get(c, ""), "main": c in MT.MAIN,
                    "icc1_within_cond": float(np.nanmean(iccs)) if np.isfinite(iccs).any() else np.nan,
                    "icc3_across_cond": S.icc3(Mx), **vc})
    return out


def signature_match(cell_rows, cols, rng):
    """個人の「型」が条件をまたいで保たれるか（多変量）。

    条件 A の各人のプロファイル（指標ベクトル、指標ごとに人の間の SD で標準化）に最も近い
    条件 B のプロファイルが本人かどうか。正解数を、B のラベルを並べ替えた帰無分布と比べる。
    """
    people = sorted({r["participant"] for r in cell_rows})
    conds = sorted({r["cond"] for r in cell_rows})
    out = []
    for a, b in itertools.combinations(conds, 2):
        A = np.array([[_cell(cell_rows, p, int(a.split("_")[0]), a.split("_")[1], c)[0] for c in cols] for p in people])
        B = np.array([[_cell(cell_rows, p, int(b.split("_")[0]), b.split("_")[1], c)[0] for c in cols] for p in people])
        ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
        A, B = A[ok], B[ok]
        n = len(A)
        if n < 4:
            continue
        sd = np.nanstd(np.vstack([A, B]), axis=0)
        sd[sd == 0] = 1
        A, B = A / sd, B / sd
        D = np.linalg.norm(A[:, None] - B[None], axis=2)
        hit = int(np.sum(np.argmin(D, 1) == np.arange(n)))
        null = np.array([np.sum(np.argmin(D[:, rng.permutation(n)], 1) == np.arange(n)) for _ in range(K.N_PERM)])
        out.append({"cond_a": a, "cond_b": b, "n": n, "hits": hit, "chance": 1.0,
                    "p_perm": float(np.mean(null >= hit)), "metrics": ",".join(cols)})
    return out


def person_means(cell_rows, cols):
    people = sorted({r["participant"] for r in cell_rows})
    res = []
    for p in people:
        rs = [r for r in cell_rows if r["participant"] == p]
        row = {"participant": p, "group": rs[0]["group"], "n_cells": len(rs)}
        for c in cols:
            v = S.finite([r.get(c) for r in rs])
            row[c] = float(np.mean(v)) if len(v) else np.nan
        res.append(row)
    return res


def group_tests(cell_rows, cols, rng):
    """FPS vs 非 FPS: 人の平均（全条件 / 条件ごと）で並べ替え検定（正確な p）と Hedges g。

    あわせて「DPI を変えたときの変化量 d が群で違うか」も検定する（不変性が群で違うか）。
    """
    pm = person_means(cell_rows, cols)
    g1, g2 = list(K.GROUPS)[:2]
    strata = [("全条件の平均", pm)]
    for cond in sorted({r["cond"] for r in cell_rows}):
        strata.append((cond, [r for r in cell_rows if r["cond"] == cond]))
    out = []
    for c in cols:
        for st, rs in strata:
            a = [r[c] for r in rs if r["group"] == g1]
            b = [r[c] for r in rs if r["group"] == g2]
            diff, p = S.perm_group(a, b, rng)
            out.append({"metric": c, "label": MT.label(c), "hyp": MT.HYP.get(c, ""), "main": c in MT.MAIN,
                        "stratum": st, f"mean_{g1}": float(np.mean(S.finite(a))) if len(S.finite(a)) else np.nan,
                        f"mean_{g2}": float(np.mean(S.finite(b))) if len(S.finite(b)) else np.nan,
                        "n1": len(S.finite(a)), "n2": len(S.finite(b)), "diff": diff, "p_perm": p,
                        "g": S.hedges_g(a, b)})
        # DPI の変化量（帯域ごと）
        dpis = sorted({r["dpi"] for r in cell_rows})
        if len(dpis) >= 2:
            for band in K.BANDS:
                da, db = {}, {}
                for r in cell_rows:
                    if r["band"] != band:
                        continue
                    if r["dpi"] == dpis[0]:
                        da[r["participant"]] = (r[c], r["group"])
                    if r["dpi"] == dpis[-1]:
                        db[r["participant"]] = (r[c], r["group"])
                d = {p: (db[p][0] - da[p][0], da[p][1]) for p in da if p in db}
                a = [v for v, g in d.values() if g == g1]
                b = [v for v, g in d.values() if g == g2]
                diff, p = S.perm_group(a, b, rng)
                out.append({"metric": c, "label": MT.label(c), "hyp": MT.HYP.get(c, ""), "main": c in MT.MAIN,
                            "stratum": f"DPI の変化（{band}）", f"mean_{g1}": float(np.mean(S.finite(a))) if len(S.finite(a)) else np.nan,
                            f"mean_{g2}": float(np.mean(S.finite(b))) if len(S.finite(b)) else np.nan,
                            "n1": len(S.finite(a)), "n2": len(S.finite(b)), "diff": diff, "p_perm": p,
                            "g": S.hedges_g(a, b)})
    idx = [i for i, r in enumerate(out) if r["main"] and r["stratum"] == "全条件の平均"]
    q = S.bh([out[i]["p_perm"] for i in idx])
    for j, i in enumerate(idx):
        out[i]["q_bh"] = q[j]
    return out, pm


def amp_profile(move_rows, parts=("mshare_prox", "mshare_hand", "N_move", "C_move3")):
    """振幅の区分ごと（全 DPI 共通の区分）の 1 動作の指標: 人 × DPI × 帯域 × 区分。

    同じ振幅の動作どうしで 400 と 800 を比べられる（DPI の効果と振幅の効果を分ける）。
    """
    edges = K.AMP_BINS_MM
    out = {}
    for r in move_rows:
        if not r.get("valid"):
            continue
        a = _f(r["amp_mm"])
        k = int(np.searchsorted(edges, a, side="right")) - 1
        if k < 0 or k >= len(edges) - 1:
            continue
        key = (r["participant"], r["group"], r["dpi"], r["band"], k)
        out.setdefault(key, []).append(r)
    res = []
    for (p, g, d, b, k), rs in sorted(out.items(), key=lambda x: x[0]):
        if len(rs) < K.MIN_MOVES_PER_CELL:
            continue
        row = {"participant": p, "group": g, "dpi": d, "band": b, "amp_lo": edges[k], "amp_hi": edges[k + 1],
               "amp_mid": float(np.sqrt(edges[k] * edges[k + 1])), "n_moves": len(rs)}
        for c in parts:
            v = S.finite([_f(r.get(c)) for r in rs])
            row[c] = float(np.mean(v)) if len(v) else np.nan
        res.append(row)
    return res


def matched_amp_dpi(prof, cols=("mshare_prox", "mshare_hand", "N_move", "C_move3")):
    """同じ振幅の区分で 800 − 400（人ごと）→ 区分をまたいで平均した差と同等性。"""
    people = sorted({r["participant"] for r in prof})
    dpis = sorted({r["dpi"] for r in prof})
    out = []
    if len(dpis) < 2:
        return out
    for band in K.BANDS:
        for c in cols:
            d_p = []
            for p in people:
                ds = []
                for k in sorted({r["amp_lo"] for r in prof}):
                    a = [r[c] for r in prof if r["participant"] == p and r["band"] == band and r["dpi"] == dpis[0] and r["amp_lo"] == k]
                    b = [r[c] for r in prof if r["participant"] == p and r["band"] == band and r["dpi"] == dpis[-1] and r["amp_lo"] == k]
                    if a and b and np.isfinite(a[0]) and np.isfinite(b[0]):
                        ds.append(b[0] - a[0])
                if ds:
                    d_p.append(np.mean(ds))
            t = S.tost(d_p, MT.sesoi(c))
            out.append({"metric": c, "label": MT.label(c), "band": band, "n": t["n"], "d": t["mean"],
                        "ci95_lo": t["ci95"][0], "ci95_hi": t["ci95"][1], "p_diff": t["p_diff"],
                        "p_tost": t["p_tost"], "sesoi": MT.sesoi(c), "verdict": S.verdict(t, MT.sesoi(c))})
    return out
