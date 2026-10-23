"""練度グループの比較（参加者 = 単位。試行は seed 3 本を参加者ごとに平均してから比べる）

人数が少ない（A 3 / B 2 / C 3 人）ので、検定は探索的な目安として扱う。
  rho / p_perm   練度の順位（A=3, B=2, C=1）と参加者平均の Spearman ρ。p は練度ラベルの全並べ替え
                 （8 人なら 560 通り、Syou を除く 7 人なら 210 通り）による正確な両側 p
  q_bh           指標の族（追従 / 協調）ごとの Benjamini–Hochberg 補正
  H / p_kw       Kruskal–Wallis（3 群）
  g_AC           Hedges の g（A − C）
  D              判別比 = 参加者間 SD / 参加者内（seed 間）SD の平均。1 付近 = 人の違いが試行のばらつきと同程度
  syou_z         Syou の値の、他 7 人に対するロバスト z（(x − 中央値) / (1.4826·MAD)）
  flip           Syou を除くと ρ の符号が変わる / |Δρ| ≥ 0.3 のとき True
"""
import itertools
import numpy as np
from scipy import stats as ss

from . import config as K

# (キー, 見出し, 族, 良い向き: +1 = 大きいほど良い / −1 = 小さいほど良い / 0 = 良し悪しなし)
METRICS = [
    # ① 追従精度
    ("rms_2d", "RMS 誤差 [mm]", "追従精度", -1),
    ("on_target_pct", "on-target [%]", "追従精度", +1),
    ("vaf_pct", "追従率 VAF [%]", "追従精度", +1),
    ("lin_track_pct", "線形追従率 [%]", "追従精度", +1),
    ("gain_mean", "追従ゲイン", "追従精度", 0),
    ("delay_ms", "実効遅れ（位相の傾き）[ms]", "追従精度", -1),
    ("lag_mean_ms", "位相遅れの平均 [ms]", "追従精度", -1),
    ("gain_hi", "高周波でのゲイン", "追従精度", 0),
    # ① 滑らかさ
    ("log_jerk_ratio", "躍度比 log10", "滑らかさ", -1),
    ("ldlj_diff", "LDLJ 差（カーソル−ターゲット）", "滑らかさ", +1),
    ("hf_frac", "課題外帯域のパワー割合", "滑らかさ", -1),
    # ① 軌道修正
    ("corr_rate", "修正の頻度 [1/s]", "軌道修正", 0),
    ("corr_amp_mm", "修正の振幅 [mm]", "軌道修正", -1),
    ("corr_peak_mm", "修正開始時の誤差 [mm]", "軌道修正", -1),
    ("corr_time_s", "修正にかかる時間 [s]", "軌道修正", -1),
    ("submove_excess", "サブムーブメント超過 [1/s]", "軌道修正", 0),
    ("reversal_excess", "向きの反転の超過 [1/s]", "軌道修正", 0),
    ("exit_rate", "ターゲットから外れる頻度 [1/s]", "軌道修正", -1),
    ("out_dur_s", "外れてから戻るまで [s]", "軌道修正", -1),
    ("hf_jitter", "2–10 Hz の揺れ [mm/s]", "軌道修正", -1),
    ("remnant_mm", "ターゲットで説明できない動き [mm]", "軌道修正", -1),
    # ② 寄与・分配
    ("share_trunk", "体幹のシェア", "関節の寄与", 0),
    ("share_upperarm", "肩関節のシェア", "関節の寄与", 0),
    ("share_forearm", "肘関節のシェア", "関節の寄与", 0),
    ("share_hand", "手首のシェア【参考】", "関節の寄与", 0),
    ("distal_share", "遠位（肘+手首）のシェア", "関節の寄与", 0),
    ("N_eff", "関与関節数 N_eff", "関節の寄与", 0),
    ("share_x_forearm", "x（左右）: 肘関節のシェア", "関節の寄与", 0),
    ("share_x_hand", "x（左右）: 手首のシェア【参考】", "関節の寄与", 0),
    ("share_y_upperarm", "y（前後）: 肩関節のシェア", "関節の寄与", 0),
    ("share_y_forearm", "y（前後）: 肘関節のシェア", "関節の寄与", 0),
    ("share_y_trunk", "y（前後）: 体幹のシェア", "関節の寄与", 0),
    ("proj_upperarm", "肩関節の射影シェア", "関節の寄与", 0),
    ("proj_hand", "手首の射影シェア【参考】", "関節の寄与", 0),
    ("C", "相殺比 C", "時間の役割分担", 0),
    ("S", "周波数間の分担のそろい S", "時間の役割分担", 0),
    ("fshift_upperarm", "肩のシェアの周波数変化（高−低）", "時間の役割分担", 0),
    ("fshift_forearm", "肘のシェアの周波数変化（高−低）", "時間の役割分担", 0),
    ("fshift_hand", "手首のシェアの周波数変化（高−低）【参考】", "時間の役割分担", 0),
    ("lead_upperarm_ms", "肩の手先に対する進み [ms]", "時間の役割分担", 0),
    ("lead_forearm_ms", "肘の手先に対する進み [ms]", "時間の役割分担", 0),
    ("lead_hand_ms", "手首の手先に対する進み [ms]【参考】", "時間の役割分担", 0),
    ("xr_upperarm_forearm", "r(肩, 肘)", "時間の役割分担", 0),
    ("xlag_upperarm_forearm_ms", "ラグ(肩→肘) [ms]", "時間の役割分担", 0),
    ("xr_forearm_hand", "r(肘, 手首)", "時間の役割分担", 0),
    ("xlag_forearm_hand_ms", "ラグ(肘→手首) [ms]", "時間の役割分担", 0),
    ("opp_upperarm_forearm", "肩と肘が逆向きの割合", "時間の役割分担", 0),
    ("opp_forearm_hand", "肘と手首が逆向きの割合", "時間の役割分担", 0),
    # ② 動き出しの順序（手先が向きを変える瞬間に、どの関節が先に向きを変えるか）
    ("init_tau_trunk_ms", "動き出し: 体幹の手先に対する時刻 [ms]", "動き出しの順序", 0),
    ("init_tau_upperarm_ms", "動き出し: 肩の手先に対する時刻 [ms]", "動き出しの順序", 0),
    ("init_tau_forearm_ms", "動き出し: 肘の手先に対する時刻 [ms]", "動き出しの順序", 0),
    ("init_tau_hand_ms", "動き出し: 手首の手先に対する時刻 [ms]【参考】", "動き出しの順序", 0),
    ("init_first_upperarm", "最初に動く割合: 肩", "動き出しの順序", 0),
    ("init_first_forearm", "最初に動く割合: 肘", "動き出しの順序", 0),
    ("init_first_hand", "最初に動く割合: 手首【参考】", "動き出しの順序", 0),
    ("init_prec_upperarm_forearm", "肩が肘より先の割合", "動き出しの順序", 0),
    ("init_prec_forearm_hand", "肘が手首より先の割合【参考】", "動き出しの順序", 0),
    ("init_rel_upperarm_forearm_ms", "肩 − 肘の時刻差 [ms]", "動き出しの順序", 0),
    ("init_rel_forearm_hand_ms", "肘 − 手首の時刻差 [ms]【参考】", "動き出しの順序", 0),
    ("elbow_range", "肘角の範囲 [deg]", "関節の寄与", 0),
    ("wrist_range", "手首角の範囲 [deg]【参考】", "関節の寄与", 0),
]
MLAB = {k: lab for k, lab, _, _ in METRICS}
FAMILY = {k: fam for k, _, fam, _ in METRICS}
BETTER = {k: b for k, _, _, b in METRICS}


def person_table(trials):
    """試行の表 → 参加者ごとの平均と SD（seed 間）。trials は dict のリスト。"""
    people = {}
    for r in trials:
        people.setdefault(r["participant"], []).append(r)
    out = []
    for name, rows in people.items():
        row = {"participant": name, "group": rows[0]["group"], "n_trials": len(rows)}
        for k, *_ in METRICS:
            v = np.array([r.get(k, np.nan) for r in rows], float)
            v = v[np.isfinite(v)]
            row[k] = float(v.mean()) if len(v) else np.nan
            row[k + "__sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
        out.append(row)
    order = {n: i for i, n in enumerate(sum(K.GROUPS.values(), []))}
    return sorted(out, key=lambda r: order.get(r["participant"], 99))


def _rho(x, y):
    return ss.spearmanr(x, y).statistic


_LABEL_CACHE = {}


def _label_ranks(levels):
    """練度ラベルの異なる並べ方すべて（の順位、中心化・正規化済み）。ラベルの組ごとに 1 回だけ作る。"""
    key = tuple(sorted(levels))
    if key not in _LABEL_CACHE:
        labs = np.array(sorted(set(itertools.permutations(key))), float)
        R = ss.rankdata(labs, axis=1)
        R -= R.mean(1, keepdims=True)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        _LABEL_CACHE[key] = R
    return _LABEL_CACHE[key]


def _perm_rho(levels, y):
    """練度ラベルの全並べ替えによる Spearman ρ の正確な両側 p。

    Spearman ρ = 順位どうしの Pearson 相関なので、ラベルの並べ方すべての順位を行列にしておき、
    y の順位との内積で一度に計算する（spearmanr を並べ方の数だけ呼ぶのと同じ値）。
    """
    levels, y = np.asarray(levels), np.asarray(y, float)
    obs = _rho(levels, y)
    if not np.isfinite(obs):
        return np.nan, np.nan
    ry = ss.rankdata(y) - (len(y) + 1) / 2
    ry /= np.linalg.norm(ry)
    null = _label_ranks(levels) @ ry
    p = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
    return float(obs), p


def _hedges(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if sp == 0:
        return np.nan
    J = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    return float((a.mean() - b.mean()) / sp * J)


def _bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    ps = p[ok]; o = np.argsort(ps); m = len(ps)
    qq = ps[o] * m / (np.arange(m) + 1)
    qq = np.minimum.accumulate(qq[::-1])[::-1]
    tmp = np.empty(m); tmp[o] = np.minimum(qq, 1)
    q[ok] = tmp
    return q


def _robust_z(x, others):
    others = np.asarray(others, float); others = others[np.isfinite(others)]
    if len(others) < 3 or not np.isfinite(x):
        return np.nan
    med = np.median(others); mad = 1.4826 * np.median(np.abs(others - med))
    if mad == 0:
        mad = others.std(ddof=1) or np.nan
    return float((x - med) / mad)


def group_tests(people, trials):
    """指標ごとの群比較（Syou あり / なし）。行のリスト。"""
    rows = []
    out_names = set(K.OUTLIER_CANDIDATES)
    for k, lab, fam, better in METRICS:
        row = {"metric": k, "label": lab, "family": fam, "better": better}
        for tag, sel in (("all", people), ("noSyou", [p for p in people if p["participant"] not in out_names])):
            ok = [p for p in sel if np.isfinite(p.get(k, np.nan))]
            lv = [K.GROUP_LEVEL[p["group"]] for p in ok]
            y = [p[k] for p in ok]
            for g in K.GROUPS:
                v = [p[k] for p in ok if p["group"] == g]
                row[f"mean_{g}_{tag}"] = float(np.mean(v)) if v else np.nan
                row[f"sd_{g}_{tag}"] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
            if len(set(lv)) >= 2 and len(ok) >= 4:
                rho, p = _perm_rho(lv, y)
                grp = [[p_[k] for p_ in ok if p_["group"] == g] for g in K.GROUPS]
                grp = [g_ for g_ in grp if len(g_)]
                try:
                    H, pk = ss.kruskal(*grp)
                except ValueError:
                    H, pk = np.nan, np.nan
            else:
                rho = p = H = pk = np.nan
            row[f"rho_{tag}"], row[f"p_{tag}"] = rho, p
            row[f"H_{tag}"], row[f"p_kw_{tag}"] = float(H), float(pk)
            row[f"g_AC_{tag}"] = _hedges([p_[k] for p_ in ok if p_["group"] == "A"],
                                         [p_[k] for p_ in ok if p_["group"] == "C"])
            row[f"n_{tag}"] = len(ok)
        # 判別比 D（参加者間 SD / 参加者内 SD）
        means = np.array([p[k] for p in people], float)
        sds = np.array([p[k + "__sd"] for p in people], float)
        within = np.nanmean(sds) if np.isfinite(sds).any() else np.nan
        row["D"] = float(np.nanstd(means, ddof=1) / within) if within and within > 0 else np.nan
        # Syou の位置
        for nm in K.OUTLIER_CANDIDATES:
            me = [p[k] for p in people if p["participant"] == nm]
            others = [p[k] for p in people if p["participant"] != nm]
            row[f"{nm}_value"] = me[0] if me else np.nan
            row[f"{nm}_z"] = _robust_z(me[0], others) if me else np.nan
        d = row["rho_noSyou"] - row["rho_all"]
        row["delta_rho"] = d
        row["flip"] = bool(np.isfinite(d) and (np.sign(row["rho_noSyou"]) != np.sign(row["rho_all"]) or abs(d) >= 0.3))
        rows.append(row)
    for tag in ("all", "noSyou"):
        for fam in sorted(set(FAMILY.values())):
            idx = [i for i, r in enumerate(rows) if r["family"] == fam]
            q = _bh([rows[i][f"p_{tag}"] for i in idx])
            for i, qq in zip(idx, q):
                rows[i][f"q_{tag}"] = float(qq)
    return rows


def freq_group_table(freq_rows, value_keys):
    """周波数 × 軸ごとの行（試行単位）→ 参加者平均 → 群平均。"""
    by = {}
    for r in freq_rows:
        by.setdefault((r["participant"], r["group"], r["axis"], r["n"], r["f_hz"]), []).append(r)
    person = []
    for (pp, g, ax, n, f), rows in by.items():
        row = {"participant": pp, "group": g, "axis": ax, "n": n, "f_hz": f}
        for k in value_keys:
            v = np.array([r.get(k, np.nan) for r in rows], float)
            row[k] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        person.append(row)
    out = []
    for tag, excl in (("all", set()), ("noSyou", set(K.OUTLIER_CANDIDATES))):
        keys = sorted({(r["group"], r["axis"], r["n"], r["f_hz"]) for r in person})
        for g, ax, n, f in keys:
            rows = [r for r in person if (r["group"], r["axis"], r["n"]) == (g, ax, n) and r["participant"] not in excl]
            if not rows:
                continue
            row = {"set": tag, "group": g, "axis": ax, "n": n, "f_hz": f, "n_people": len(rows)}
            for k in value_keys:
                v = np.array([r[k] for r in rows], float)
                row[k] = float(np.nanmean(v)); row[k + "__sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else np.nan
            out.append(row)
    return person, out
