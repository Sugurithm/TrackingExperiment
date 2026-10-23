"""
pca_coherence.py — 時系列そのものに対する主成分分析（PCA）とコヒーレンス解析
────────────────────────────────────────────────────────────────────
練度（上級 / 中級 / 初級）やスコアとの関係を、「差が出るように作った指標」ではなく
汎用の手法でそのまま見る。差が無かった結果もすべて表に残す。

  python skill_analysis/pca_coherence.py                        # dpi400_slow
  python skill_analysis/pca_coherence.py --all                  # 4 条件それぞれ独立に
  python skill_analysis/pca_coherence.py --recordings D:/rec --out D:/res_ts

入力: recordings/aim_*（run_skill.py と同じ。読むだけ）
出力: skill_analysis/results_ts/<条件>/
        ts_trials.csv     試行ごとの全指標
        ts_persons.csv    参加者ごと（3 試行の平均、__sd = 試行間 SD）
        ts_tests.csv      全指標の 練度との ρ / スコア（RMS 誤差）との ρ / 試行間の r / q（BH）
        coh_spectra.csv   群平均のコヒーレンス・スペクトル
        pca_loadings.csv  参加者ごとの第 1 主成分の負荷量
        figures/*.png     図（参加者名は出さない）
        report.md         数値の要約（自動）

■ 時系列 PCA（協調の次元）
  手先の速度 = 体幹 + 肩関節（上腕）+ 肘関節（前腕）+ 手首（手）の 4 成分（aim_eval と同じ分解）。
  各成分の x・y 速度 = 8 列の時系列 V(t) を、帯域通過してから共分散の PCA にかける（単位が同じ mm/s なので標準化しない）。
    帯域: 課題 = 0.03 Hz〜最高入力周波数 + 0.2 Hz、修正 = 2〜4 Hz（課題に無い帯域）
    VAF1   第 1 主成分が説明する分散の割合（1 に近い = 全成分が 1 つのパターンでまとまって動く）
    PR     参加比 (Σλ)² / Σλ²（実効的な次元数。1〜8）
    n90    累積で 90 % に達する主成分の数
    負荷量 第 1 主成分の 8 要素（手先方向への寄与が正になるよう符号をそろえる）
  手首は MoCap 未照合なので、手首を除いた 6 列版（_nohand）も出す。

■ コヒーレンス（Welch, 窓 8 s・50 % 重なり、60 Hz）
  |S_ab(f)|² / (S_aa(f) S_bb(f))。2 つの信号が周波数 f で線形に結びついている度合い（0〜1）。
    tc_   ターゲット → カーソル（各軸）。課題の帯域で平均 = 線形に追えている度合い
    jj_   成分どうしの速度（各軸）。課題 / 修正の帯域で平均。位相（角度の平均）も出す
          位相 ≈ 0° = 同じ向きに一緒に動く、≈ 180° = 逆向きに動いて打ち消し合う
    jm_   各成分の速度 → マウスの速度（カメラと独立）。修正の帯域で、マウスの揺れがどの成分と
          線形に一緒に動いているか
  カメラのノイズはマウス・ターゲットと無相関なので、tc_ と jm_ は計測ノイズで上がらない（下がる側に効く）。
  jj_ はノイズどうしが関節の点を共有するため、修正の帯域では見かけの結びつき（特に逆相）が出うる → 解釈に注意。

■ 統計
  ρ_skill  練度の順位（上級 3・中級 2・初級 1）と参加者平均の Spearman ρ。p はラベルの全並べ替え（8 人 560 通り）
  ρ_score  参加者平均の RMS 誤差（スコア）との Spearman ρ。p は 8! 通りの全並べ替え
  r_within 同じ人の試行どうし（参加者平均を引いた残差）での、指標と RMS 誤差の Pearson r（試行間のゆらぎ）
  q        指標の族ごとの BH 補正。外れ値候補（config.OUTLIER_CANDIDATES）を除いた ρ も併記
────────────────────────────────────────────────────────────────────
"""
import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skill_eval import config as K                               # noqa: E402
from skill_eval import tracking                                    # noqa: E402
from skill_eval.io import C, find_sessions, trial_dirs, load_trial  # noqa: E402

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from scipy.signal import butter, csd, filtfilt, welch              # noqa: E402
from scipy.stats import pearsonr, rankdata                         # noqa: E402

plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans",
                                     "IPAexGothic", "sans-serif"],
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 12})

PARTS = ["trunk", "upperarm", "forearm", "hand"]
PJ = {"trunk": "体幹", "upperarm": "肩関節", "forearm": "肘関節", "hand": "手首"}
PAIRS = [("trunk", "upperarm"), ("upperarm", "forearm"), ("forearm", "hand"), ("upperarm", "hand")]
GN = {"A": "上級者", "B": "中級者", "C": "初級者"}
GC = {"A": "#104281", "B": "#2a78d6", "C": "#86b6ef"}
GM = {"A": "o", "B": "s", "C": "D"}
LV = {"A": 3, "B": 2, "C": 1}
FS = C.FS_U
NPERSEG_S = 8.0
CORR_BAND = (2.0, 4.0)
INK2, GRID = "#5f5f5a", "#e6e6e3"


# ───────────────────────────────────────── 信号処理
def bandpass(X, lo, hi, fs=FS):
    b, a = butter(2, [lo / (fs / 2), min(hi / (fs / 2), 0.99)], btype="band")
    return filtfilt(b, a, X, axis=0)


def deriv(X):
    return np.gradient(X, axis=0) * FS


def pca(X):
    """共分散 PCA。(固有値の降順, 固有ベクトル[:, k])"""
    Xc = X - X.mean(0)
    lam, W = np.linalg.eigh(np.cov(Xc, rowvar=False))
    o = np.argsort(lam)[::-1]
    return np.clip(lam[o], 0, None), W[:, o]


def pca_metrics(V, cols, endpoint, tag):
    """V: {col: (N,) 速度}。endpoint: (N,2) 手先の速度（負荷量の符号合わせ用）。"""
    X = np.column_stack([V[c] for c in cols])
    lam, W = pca(X)
    tot = lam.sum()
    out = {}
    if tot <= 0:
        return out, None
    ev = lam / tot
    out[f"pca_{tag}_vaf1"] = float(ev[0])
    out[f"pca_{tag}_vaf2"] = float(ev[1]) if len(ev) > 1 else np.nan
    out[f"pca_{tag}_pr"] = float(tot ** 2 / np.sum(lam ** 2))
    out[f"pca_{tag}_n90"] = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
    w = W[:, 0]
    score = (X - X.mean(0)) @ w
    # 符号: 第 1 主成分の時系列が手先の速さの大きい向き（x・y の和）と正に相関するように
    ep = endpoint[:, 0] + endpoint[:, 1]
    if np.corrcoef(score, ep)[0, 1] < 0:
        w = -w
    return out, dict(zip(cols, w))


def band_mean(f, y, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.nanmean(y[m])) if m.any() else np.nan


def coh_and_phase(a, b, fs=FS):
    nper = int(NPERSEG_S * fs)
    f, Sab = csd(a, b, fs=fs, nperseg=nper)
    _, Saa = welch(a, fs=fs, nperseg=nper)
    _, Sbb = welch(b, fs=fs, nperseg=nper)
    coh = np.abs(Sab) ** 2 / (Saa * Sbb + 1e-30)
    return f, coh, Sab


def band_phase(f, Sab, lo, hi):
    """帯域内のクロススペクトルを足し合わせた位相 [deg]（|位相| 0〜180）。"""
    m = (f >= lo) & (f <= hi)
    return float(abs(np.degrees(np.angle(np.sum(Sab[m]))))) if m.any() else np.nan


# ───────────────────────────────────────── 1 試行
def analyze_trial(tr):
    V, E, ok = C.vel_contributions(tr)
    if V is None:
        return None, None
    fmax = tracking.max_input_freq(tr)
    task_band = (0.03, fmax + 0.2)
    res, spec = {}, {}

    # 速度（x, y）。欠損は補間
    comp = {}
    for p in PARTS:
        X, _ = C.fill_nan(V[p][:, :2])
        comp[p] = X
    Ep, _ = C.fill_nan(E[:, :2])
    tg = C.fill_nan(tr.task_grid("target"))[0]
    cu = C.fill_nan(tr.task_grid("cursor"))[0]
    mo = C.fill_nan(tr.task_grid("mouse"))[0]
    vt, vc, vm = deriv(tg), deriv(cu), deriv(mo)

    # ── 時系列 PCA（課題の帯域 / 修正の帯域）
    for tag, (lo, hi) in (("task", task_band), ("corr", CORR_BAND)):
        Vb = {f"{p}_{a}": bandpass(comp[p][:, k], lo, hi) for p in PARTS for k, a in enumerate("xy")}
        Eb = bandpass(Ep, lo, hi)
        cols8 = [f"{p}_{a}" for p in PARTS for a in "xy"]
        cols6 = [c for c in cols8 if not c.startswith("hand")]
        m8, w8 = pca_metrics(Vb, cols8, Eb, tag)
        m6, w6 = pca_metrics(Vb, cols6, Eb, tag + "_nohand")
        res.update(m8); res.update(m6)
        if w8:
            for c, v in w8.items():
                res[f"load_{tag}_{c}"] = float(v)

    # ── コヒーレンス
    for k, a in enumerate("xy"):
        f, coh, Sab = coh_and_phase(vt[:, k], vc[:, k])
        res[f"tc_{a}_task"] = band_mean(f, coh, 0.04, fmax + 0.05)
        spec[f"tc_{a}"] = (f, coh)
        for p, q in PAIRS:
            f, coh, Sab = coh_and_phase(comp[p][:, k], comp[q][:, k])
            for tag, (lo, hi) in (("task", (0.04, fmax + 0.05)), ("corr", CORR_BAND)):
                res[f"jj_{p}_{q}_{a}_{tag}"] = band_mean(f, coh, lo, hi)
                res[f"jjph_{p}_{q}_{a}_{tag}"] = band_phase(f, Sab, lo, hi)
            spec[f"jj_{p}_{q}_{a}"] = (f, coh)
        for p in PARTS:
            f, coh, Sab = coh_and_phase(comp[p][:, k], vm[:, k])
            res[f"jm_{p}_{a}_corr"] = band_mean(f, coh, *CORR_BAND)
            res[f"jmph_{p}_{a}_corr"] = band_phase(f, Sab, *CORR_BAND)
            spec[f"jm_{p}_{a}"] = (f, coh)
        # 手先（カメラ）とマウスのコヒーレンス = カメラが修正の揺れを捉えているか（品質の確認）
        f, coh, _ = coh_and_phase(Ep[:, k], vm[:, k])
        res[f"em_{a}_corr"] = band_mean(f, coh, *CORR_BAND)
        res[f"em_{a}_task"] = band_mean(f, coh, 0.04, fmax + 0.05)
    # x・y の平均（主な指標）
    for key in [k for k in list(res) if k.endswith(("_x_task", "_x_corr"))]:
        ky = key.replace("_x_", "_y_")
        if ky in res:
            res[key.replace("_x_", "_xy_")] = float(np.nanmean([res[key], res[ky]]))
    for a in "xy":
        res[f"tc_xy_task"] = float(np.nanmean([res["tc_x_task"], res["tc_y_task"]]))
    return res, spec


# ───────────────────────────────────────── 統計
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return np.nan
    rx, ry = rankdata(x[m]), rankdata(y[m])
    return float(np.corrcoef(rx, ry)[0, 1])


_PERMS = {}


def perm_p(x, y, exact_labels=True):
    """Spearman ρ の全並べ替えによる両側 p（x の並べ方をすべて試す。n ≤ 9 を想定）。
    同順位（練度ラベル）があっても、全 n! 通りで数えれば区別できる並べ方の数で数えたのと同じ割合になる。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 4 or n > 9:
        return np.nan
    if n not in _PERMS:
        _PERMS[n] = np.array(list(itertools.permutations(range(n))))
    rx, ry = rankdata(x), rankdata(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12); ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    obs = abs(np.mean(rx * ry))
    null = np.abs((rx[_PERMS[n]] * ry).mean(1))
    return float(np.mean(null >= obs - 1e-12))


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    m = np.isfinite(p)
    if not m.any():
        return q
    ps = p[m]; o = np.argsort(ps); n = len(ps)
    adj = ps[o] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(adj, 1)
    q[m] = out
    return q


FAMILY = [("pca_task", "時系列 PCA（課題の帯域）"), ("pca_corr", "時系列 PCA（修正の帯域）"),
          ("tc_", "ターゲット→カーソルのコヒーレンス"), ("jj_", "成分どうしのコヒーレンス"),
          ("jjph_", "成分どうしの位相"), ("jm_", "成分→マウスのコヒーレンス（修正の帯域）"),
          ("jmph_", "成分→マウスの位相"), ("em_", "計測の確認（手先→マウス）")]


def family_of(k):
    for pre, lab in FAMILY:
        if k.startswith(pre):
            return lab
    return "その他"


def run_tests(people, trials):
    keys = [k for k in people[0] if not k.startswith(("participant", "group", "load_", "rms_2d", "on_target"))
            and not k.endswith("__sd") and isinstance(people[0][k], (float, int))]
    lv = np.array([LV[p["group"]] for p in people], float)
    score = np.array([p["rms_2d"] for p in people], float)
    keep = np.array([p["participant"] not in K.OUTLIER_CANDIDATES for p in people])
    rows = []
    for k in keys:
        v = np.array([p.get(k, np.nan) for p in people], float)
        if not np.isfinite(v).any():
            continue
        # 試行間: 参加者平均を引いた残差どうし
        tv = np.array([t.get(k, np.nan) for t in trials], float)
        ts = np.array([t["rms_2d"] for t in trials], float)
        who = np.array([t["participant"] for t in trials])
        rv, rs = tv.copy(), ts.copy()
        for w in set(who):
            i = who == w
            rv[i] -= np.nanmean(tv[i]); rs[i] -= np.nanmean(ts[i])
        mm = np.isfinite(rv) & np.isfinite(rs)
        rw, pw = (pearsonr(rv[mm], rs[mm]) if mm.sum() > 5 and np.std(rv[mm]) > 0 else (np.nan, np.nan))
        sd_in = np.nanmean([p.get(k + "__sd", np.nan) for p in people])
        rows.append({"metric": k, "family": family_of(k),
                     "mean_A": np.nanmean(v[lv == 3]), "mean_B": np.nanmean(v[lv == 2]), "mean_C": np.nanmean(v[lv == 1]),
                     "rho_skill": spearman(lv, v), "p_skill": perm_p(lv, v),
                     "rho_skill_excl": spearman(lv[keep], v[keep]),
                     "rho_score": spearman(score, v), "p_score": perm_p(score, v, exact_labels=False),
                     "r_within": float(rw), "p_within": float(pw),
                     "D": float(np.nanstd(v, ddof=1) / sd_in) if sd_in and sd_in > 0 else np.nan})
    for fam in set(r["family"] for r in rows):
        idx = [i for i, r in enumerate(rows) if r["family"] == fam]
        for col in ("skill", "score", "within"):
            q = bh([rows[i][f"p_{col}"] for i in idx])
            for j, i in enumerate(idx):
                rows[i][f"q_{col}"] = float(q[j])
    return rows


def verdict(r):
    """差の有無を言葉で（探索的な目安）。"""
    s = []
    if np.isfinite(r["p_skill"]) and r["p_skill"] < 0.05:
        s.append("練度と関係あり" if r["q_skill"] < 0.1 else "練度と関係あり（補正後は残らない）")
    if np.isfinite(r["p_score"]) and r["p_score"] < 0.05:
        s.append("スコアと関係あり" if r["q_score"] < 0.1 else "スコアと関係あり（補正後は残らない）")
    if np.isfinite(r["p_within"]) and r["p_within"] < 0.05:
        s.append("試行間でスコアと連動")
    return "、".join(s) if s else "差なし"


# ───────────────────────────────────────── 図
def strip(ax, people, k, title):
    for i, g in enumerate("ABC"):
        v = [p[k] for p in people if p["group"] == g and np.isfinite(p.get(k, np.nan))]
        xs = i + (np.arange(len(v)) - (len(v) - 1) / 2) * .14
        ax.scatter(xs, v, s=80, marker=GM[g], color=GC[g], edgecolor="white", zorder=3)
        if v:
            ax.hlines(np.mean(v), i - .3, i + .3, color=INK2, lw=2)
    ax.set_xticks(range(3), [GN[g] for g in "ABC"])
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)


def figures(out, people, tests, spectra, loadings):
    d = os.path.join(out, "figures"); os.makedirs(d, exist_ok=True)
    T = {r["metric"]: r for r in tests}

    def ttl(k, lab):
        r = T.get(k)
        return f"{lab}\nρ練度 {r['rho_skill']:+.2f}（p {r['p_skill']:.2f}）ρスコア {r['rho_score']:+.2f}" if r else lab

    # 1 PCA の次元
    ks = [("pca_task_vaf1", "第 1 主成分の説明率（課題の帯域）"), ("pca_task_pr", "実効次元 PR（課題の帯域）"),
          ("pca_task_nohand_vaf1", "第 1 主成分の説明率（手首除く）"), ("pca_corr_vaf1", "第 1 主成分の説明率（修正 2–4 Hz）"),
          ("pca_corr_pr", "実効次元 PR（修正 2–4 Hz）")]
    fig, axs = plt.subplots(1, len(ks), figsize=(4 * len(ks), 4.6))
    for ax, (k, lab) in zip(axs, ks):
        strip(ax, people, k, ttl(k, lab))
    fig.tight_layout(); fig.savefig(os.path.join(d, "ts_pca_dimension.png"), dpi=140); plt.close(fig)

    # 2 第 1 主成分の負荷量（参加者 × 8 列、群で並べる。名前は出さない）
    cols = [f"{p}_{a}" for p in PARTS for a in "xy"]
    order = [p for g in "ABC" for p in people if p["group"] == g]
    M = np.array([[loadings.get(p["participant"], {}).get(c, np.nan) for c in cols] for p in order])
    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(M, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [f"{PJ[c.split('_')[0]]} {c.split('_')[1]}" for c in cols], rotation=30)
    ax.set_yticks(range(len(order)), [GN[p["group"]] for p in order])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(M[i, j]) > .6 else "#1f1f1d")
    ax.set_title("第 1 主成分の負荷量（課題の帯域、3 試行の平均。1 行 = 1 人）")
    fig.colorbar(im, ax=ax, fraction=.03)
    fig.tight_layout(); fig.savefig(os.path.join(d, "ts_pca_loadings.png"), dpi=140); plt.close(fig)

    # 3 コヒーレンス・スペクトル（群平均）
    def spec_plot(keys, titles, fname, xlim=(0, 6)):
        fig, axs = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.2), sharey=True)
        axs = np.atleast_1d(axs)
        for ax, k, t in zip(axs, keys, titles):
            for g in "ABC":
                S = [spectra[(p["participant"], k)] for p in people if p["group"] == g and (p["participant"], k) in spectra]
                if not S:
                    continue
                f = S[0][0]; y = np.nanmean([s[1] for s in S], 0)
                ax.plot(f, y, color=GC[g], lw=2, label=GN[g])
            ax.axvspan(*CORR_BAND, color="#f6efe6", zorder=0)
            ax.set_xlim(*xlim); ax.set_ylim(0, 1); ax.set_title(t, fontsize=11)
            ax.set_xlabel("周波数 [Hz]", color=INK2); ax.grid(color=GRID, lw=.6)
        axs[0].set_ylabel("コヒーレンス", color=INK2); axs[0].legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(d, fname), dpi=140); plt.close(fig)

    spec_plot(["tc_x", "tc_y"], ["ターゲット→カーソル（左右）", "ターゲット→カーソル（前後）"], "ts_coh_target_cursor.png", (0, 2))
    spec_plot([f"jj_{p}_{q}_x" for p, q in PAIRS], [f"{PJ[p]}–{PJ[q]}（左右）" for p, q in PAIRS], "ts_coh_joints_x.png")
    spec_plot([f"jj_{p}_{q}_y" for p, q in PAIRS], [f"{PJ[p]}–{PJ[q]}（前後）" for p, q in PAIRS], "ts_coh_joints_y.png")
    spec_plot([f"jm_{p}_x" for p in PARTS], [f"{PJ[p]}→マウス（左右）" for p in PARTS], "ts_coh_joint_mouse_x.png")
    spec_plot([f"jm_{p}_y" for p in PARTS], [f"{PJ[p]}→マウス（前後）" for p in PARTS], "ts_coh_joint_mouse_y.png")

    # 4 主な指標の練度別
    ks = [("tc_xy_task", "ターゲット→カーソル（課題の帯域）"), ("jj_upperarm_forearm_xy_task", "肩–肘 コヒーレンス（課題）"),
          ("jj_forearm_hand_xy_task", "肘–手首 コヒーレンス（課題）"), ("jj_upperarm_forearm_xy_corr", "肩–肘 コヒーレンス（修正）"),
          ("jm_forearm_xy_corr", "肘→マウス（修正）"), ("jm_hand_xy_corr", "手首→マウス（修正）")]
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 8.4))
    for ax, (k, lab) in zip(axs.ravel(), ks):
        strip(ax, people, k, ttl(k, lab))
    fig.tight_layout(); fig.savefig(os.path.join(d, "ts_coh_metrics.png"), dpi=140); plt.close(fig)

    # 5 全指標の ρ（練度 / スコア）の分布: 偶然と比べて多いか
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for r in tests:
        ax.scatter(r["rho_skill"], r["rho_score"], s=22, color="#2a78d6" if r["p_skill"] < .05 or r["p_score"] < .05 else "#b8b8b3")
    ax.axhline(0, color=INK2, lw=.8); ax.axvline(0, color=INK2, lw=.8)
    ax.set_xlabel("練度との ρ", color=INK2); ax.set_ylabel("スコア（RMS 誤差）との ρ", color=INK2)
    n = len(tests); n_sig = sum(r["p_skill"] < .05 for r in tests)
    ax.set_title(f"全 {n} 指標（青 = p < 0.05）。練度で p < 0.05 は {n_sig} 個（偶然の期待 {0.05 * n:.1f} 個）", fontsize=11)
    ax.grid(color=GRID, lw=.6)
    fig.tight_layout(); fig.savefig(os.path.join(d, "ts_all_metrics.png"), dpi=140); plt.close(fig)


# ───────────────────────────────────────── 出力
def write_csv(path, rows, first=()):
    keys = list(first)
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (v if not isinstance(v, float) or np.isfinite(v) else "") for k, v in r.items()})


def report(out, cond, people, tests):
    L = [f"# 時系列 PCA・コヒーレンス解析 — {cond['name']}", "",
         f"参加者 {len(people)} 人（上級 {sum(p['group'] == 'A' for p in people)} / 中級 {sum(p['group'] == 'B' for p in people)} / "
         f"初級 {sum(p['group'] == 'C' for p in people)}）。各指標は 3 試行の平均。p は全並べ替えの正確な値、q は族ごとの BH 補正。", "",
         "**探索的な解析。多数の指標を見ているので、p < 0.05 でも q が大きいものは偶然の可能性がある。差が無かった指標もすべて載せる。**", ""]
    n = len(tests)
    L.append(f"- 全 {n} 指標のうち、練度と p < 0.05: {sum(r['p_skill'] < .05 for r in tests)} 個 / スコアと p < 0.05: "
             f"{sum(r['p_score'] < .05 for r in tests)} 個 / 試行間でスコアと p < 0.05: {sum(r['p_within'] < .05 for r in tests)} 個"
             f"（偶然の期待はそれぞれ約 {0.05 * n:.1f} 個）")
    L.append("")
    for _, fam in FAMILY:
        rows = [r for r in tests if r["family"] == fam]
        if not rows:
            continue
        L += [f"## {fam}", "", "| 指標 | 上級 | 中級 | 初級 | ρ練度（p / q） | 外れ値除外 ρ | ρスコア（p） | 試行間 r（p） | D | 判定 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['metric']} | {r['mean_A']:.3f} | {r['mean_B']:.3f} | {r['mean_C']:.3f} | "
                     f"{r['rho_skill']:+.2f}（{r['p_skill']:.2f} / {r['q_skill']:.2f}） | {r['rho_skill_excl']:+.2f} | "
                     f"{r['rho_score']:+.2f}（{r['p_score']:.2f}） | {r['r_within']:+.2f}（{r['p_within']:.2f}） | {r['D']:.1f} | {verdict(r)} |")
        L.append("")
    L += ["## 図", ""] + [f"![{f}](figures/{f})" for f in sorted(os.listdir(os.path.join(out, "figures")))]
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write("\n".join(L))


def run(cond, out_root, log=print):
    out = os.path.join(out_root, cond["name"]); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    log(f"== {cond['name']} → {out}")
    sessions = find_sessions(cond["dpi"])
    trials, spectra_tr = [], {}
    for g, members in K.GROUPS.items():
        for person in members:
            sess = sessions.get(person)
            if not sess:
                log(f"  ! {person}: セッションなし"); continue
            for td, meta in trial_dirs(sess, cond["band"]):
                tr = load_trial(td, os.path.join(out, "_tmp"))
                if not tr.has_rs:
                    log(f"  ! {person} {os.path.basename(td)}: 関節なし"); continue
                tres = tracking.compute(tr)[0]
                res, spec = analyze_trial(tr)
                if res is None:
                    continue
                trials.append({"participant": person, "group": g, "trial": os.path.basename(td), "seed": tr.seed,
                               "rms_2d": tres["rms_2d"], "on_target_pct": tres["on_target_pct"], **res})
                for k, (f, y) in spec.items():
                    spectra_tr.setdefault((person, k), []).append((f, y))
                log(f"  {person:7s} ({g}) {os.path.basename(td)}: VAF1 {res.get('pca_task_vaf1', np.nan):.2f}  "
                    f"PR {res.get('pca_task_pr', np.nan):.2f}  tc {res.get('tc_xy_task', np.nan):.2f}")
    if not trials:
        log("  対象の試行がありません"); return
    # 参加者ごと
    people, loadings = [], {}
    for person in dict.fromkeys(t["participant"] for t in trials):
        rows = [t for t in trials if t["participant"] == person]
        p = {"participant": person, "group": rows[0]["group"]}
        for k in rows[0]:
            if k in ("participant", "group", "trial", "seed"):
                continue
            v = np.array([r.get(k, np.nan) for r in rows], float)
            p[k] = float(np.nanmean(v)); p[k + "__sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else np.nan
        people.append(p)
        loadings[person] = {c.replace("load_task_", ""): p[c] for c in p if c.startswith("load_task_") and not c.endswith("__sd")}
    spectra = {}
    for key, lst in spectra_tr.items():
        spectra[key] = (lst[0][0], np.nanmean([y for _, y in lst], 0))
    tests = run_tests(people, trials)
    # 群平均のスペクトルを CSV に
    srows = []
    for (person, k), (f, y) in spectra.items():
        g = next(p["group"] for p in people if p["participant"] == person)
        for fi, yi in zip(f, y):
            if fi <= 6:
                srows.append({"group": g, "participant": person, "spectrum": k, "f_hz": float(fi), "coh": float(yi)})
    write_csv(os.path.join(out, "ts_trials.csv"), trials, ("participant", "group", "trial", "seed"))
    write_csv(os.path.join(out, "ts_persons.csv"), people, ("participant", "group"))
    for r in tests:
        r["verdict"] = verdict(r)
    write_csv(os.path.join(out, "ts_tests.csv"), tests, ("metric", "family"))
    write_csv(os.path.join(out, "coh_spectra.csv"), srows)
    write_csv(os.path.join(out, "pca_loadings.csv"),
              [{"participant": k, "group": next(p["group"] for p in people if p["participant"] == k), **v}
               for k, v in loadings.items()])
    figures(out, people, tests, spectra, loadings)
    report(out, cond, people, tests)
    log(f"  → {out}（{time.time() - t0:.0f} s）")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cond", default="dpi400_slow")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--recordings", default=None, help="記録フォルダ（既定: config.RECORDINGS）")
    ap.add_argument("--out", default=os.path.join(K.PKG_ROOT, "results_ts"))
    a = ap.parse_args()
    if a.recordings:
        K.RECORDINGS = a.recordings
    conds = K.CONDITIONS if a.all else [c for c in K.CONDITIONS if c["name"] == a.cond]
    for c in conds:
        run(c, a.out)


if __name__ == "__main__":
    main()