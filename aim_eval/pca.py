"""
feature_pca.py — 参加者 × 指標 の表に対する主成分分析（教師なし）
────────────────────────────────────────────────────────────────────
  python skill_analysis/feature_pca.py [results_dir] [out_dir]
    既定: skill_analysis/results → skill_analysis/results_pca

run_skill.py が出した data/persons.csv（参加者 8 人 × 指標）を、
  ① 追従の成績（スコア）の指標だけ   ② 関節協調の指標だけ   ③ 両方
の 3 通りで、標準化（平均 0・SD 1）してから PCA にかける。練度のラベルは使わない（教師なし）。
その後で、各主成分のスコアが 練度の順位 / RMS 誤差（スコア）とどれだけ並ぶかを Spearman ρ で見る。

「練度で差が出るように選んだ指標」ではなく、「データの中で一番大きくばらつく方向」が
練度と関係するかどうかを確かめるのが目的。差が無い主成分もそのまま出す。

注意: 8 人に対して指標が 19〜44 個あるので、主成分は最大 7 個（人数 − 1）しか取れず、
負荷量は人の入れ替わりで大きく変わりうる（記述的な目安）。体幹型の 1 名を除いた版も併記する。
────────────────────────────────────────────────────────────────────
"""
import itertools
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans",
                                     "IPAexGothic", "sans-serif"],
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 12})
HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results_pca")
os.makedirs(OUT, exist_ok=True)
CONDS = ["dpi400_slow", "dpi400_fast", "dpi800_slow", "dpi800_fast"]
OUTLIER = "Syou"
LV = {"A": 3, "B": 2, "C": 1}
GN = {"A": "上級者", "B": "中級者", "C": "初級者"}
GC = {"A": "#104281", "B": "#2a78d6", "C": "#86b6ef"}
GM = {"A": "o", "B": "s", "C": "D"}
INK2, GRID = "#5f5f5a", "#e6e6e3"

TRK = {"rms_2d": "RMS 誤差", "on_target_pct": "on-target", "vaf_pct": "追従率 VAF", "lin_track_pct": "線形追従率",
       "lag_mean_ms": "位相遅れ平均", "delay_ms": "実効遅れ", "gain_hi": "高周波ゲイン", "log_jerk_ratio": "躍度比",
       "ldlj_diff": "LDLJ 差", "hf_frac": "課題外パワー割合", "corr_rate": "修正の頻度", "corr_peak_mm": "修正開始時の誤差",
       "corr_time_s": "修正にかかる時間", "submove_excess": "サブムーブメント超過", "reversal_excess": "反転の超過",
       "exit_rate": "外れる頻度", "out_dur_s": "外れてから戻るまで", "hf_jitter": "2–10 Hz の揺れ", "remnant_mm": "説明できない動き"}
CRD = {"share_trunk": "体幹シェア", "share_upperarm": "肩関節シェア", "share_forearm": "肘関節シェア",
       "share_hand": "手首シェア", "N_eff": "関与関節数", "share_x_hand": "x 手首シェア", "share_y_upperarm": "y 肩関節シェア",
       "share_y_forearm": "y 肘関節シェア", "share_y_trunk": "y 体幹シェア", "C": "相殺比 C", "S": "分担のそろい S",
       "fshift_upperarm": "肩の周波数変化", "fshift_forearm": "肘の周波数変化", "fshift_hand": "手首の周波数変化",
       "lead_upperarm_ms": "肩の進み", "lead_forearm_ms": "肘の進み", "lead_hand_ms": "手首の進み",
       "xr_upperarm_forearm": "r(肩,肘)", "xlag_upperarm_forearm_ms": "ラグ 肩→肘", "xr_forearm_hand": "r(肘,手首)",
       "xlag_forearm_hand_ms": "ラグ 肘→手首", "opp_upperarm_forearm": "肩肘 逆向き割合", "opp_forearm_hand": "肘手首 逆向き割合",
       "elbow_range": "肘角の範囲", "wrist_range": "手首角の範囲"}
SETS = {"score": ("追従の成績の指標", TRK), "coord": ("関節協調の指標", CRD), "all": ("両方", {**TRK, **CRD})}


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_p(x, y):
    n = len(x)
    P = np.array(list(itertools.permutations(range(n))))
    rx, ry = rankdata(x), rankdata(y)
    rx = (rx - rx.mean()) / rx.std(); ry = (ry - ry.mean()) / ry.std()
    obs = abs(np.mean(rx * ry))
    return float(np.mean(np.abs((rx[P] * ry).mean(1)) >= obs - 1e-12))


def pca(df, cols):
    X = df[list(cols)].astype(float)
    X = X.loc[:, X.std(ddof=1) > 0]
    Z = ((X - X.mean()) / X.std(ddof=1)).fillna(0).values
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    ev = s ** 2 / np.sum(s ** 2)
    sc = U * s
    L = pd.DataFrame(Vt.T, index=X.columns)
    # 符号: 各主成分のスコアが RMS 誤差と正に相関する向き（解釈しやすく）
    for k in range(len(ev)):
        if np.corrcoef(sc[:, k], df["rms_2d"])[0, 1] < 0:
            sc[:, k] *= -1; L[k] *= -1
    return ev, sc, L


rows = []
for cond in CONDS:
    f = os.path.join(RES, cond, "data", "persons.csv")
    if not os.path.exists(f):
        continue
    P = pd.read_csv(f, encoding="utf-8-sig")
    for excl in (False, True):
        Q = P[P.participant != OUTLIER].reset_index(drop=True) if excl else P
        lv = Q.group.map(LV).values
        for key, (lab, cols) in SETS.items():
            ev, sc, L = pca(Q, cols)
            for k in range(3):
                top = L[k].abs().sort_values(ascending=False).index[:5]
                rows.append({"condition": cond, "outlier_excluded": excl, "set": lab, "pc": k + 1,
                             "explained": ev[k], "rho_skill": spearman(lv, sc[:, k]), "p_skill": perm_p(lv, sc[:, k]),
                             "rho_score": spearman(Q.rms_2d.values, sc[:, k]), "p_score": perm_p(Q.rms_2d.values, sc[:, k]),
                             "top_loadings": " / ".join(f"{cols[t]} {L.loc[t, k]:+.2f}" for t in top)})
            if cond == "dpi400_slow" and not excl:
                # 図: PC1 × PC2 と負荷量
                fig = plt.figure(figsize=(15, 5.6))
                ax = fig.add_subplot(1, 3, 1)
                for i, r in Q.iterrows():
                    ax.scatter(sc[i, 0], sc[i, 1], s=120, marker=GM[r.group], color=GC[r.group], edgecolor="white", zorder=3)
                ax.axhline(0, color=GRID); ax.axvline(0, color=GRID)
                ax.set_xlabel(f"PC1（{ev[0]*100:.0f} %）", color=INK2); ax.set_ylabel(f"PC2（{ev[1]*100:.0f} %）", color=INK2)
                ax.set_title(f"{lab}の PCA（400 slow、1 点 = 1 人）", fontsize=12)
                h = [plt.Line2D([], [], ls="", marker=GM[g], color=GC[g], ms=10, label=GN[g]) for g in "ABC"]
                ax.legend(handles=h, frameon=False, fontsize=10)
                for j in (0, 1):
                    ax2 = fig.add_subplot(1, 3, j + 2)
                    s = L[j].sort_values()
                    s = pd.concat([s.head(5), s.tail(5)]).drop_duplicates()
                    ax2.barh([cols[t] for t in s.index], s.values,
                             color=["#d1453b" if v < 0 else "#2a78d6" for v in s.values])
                    ax2.axvline(0, color=INK2, lw=.8)
                    r1, p1 = spearman(lv, sc[:, j]), perm_p(lv, sc[:, j])
                    r2 = spearman(Q.rms_2d.values, sc[:, j])
                    ax2.set_title(f"PC{j+1} の負荷量（上位）\nρ練度 {r1:+.2f}（p {p1:.2f}）、ρスコア {r2:+.2f}", fontsize=12)
                    ax2.grid(axis="x", color=GRID)
                fig.tight_layout(); fig.savefig(os.path.join(OUT, f"feature_pca_{key}_400slow.png"), dpi=140); plt.close(fig)

T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "feature_pca_summary.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220)
print(T[T.pc <= 2].round(2).to_string(index=False))