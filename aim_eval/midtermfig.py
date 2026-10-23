"""
midterm_figs.py — 中間審査スライドの図（1, 2, 2', 6, 6', 7, 8）
────────────────────────────────────────────────────────────────────
  python midterm_figs.py skill_analysis/results figures_midterm

入力: skill_analysis/results/<条件>/data/ の CSV（persons / group_tests /
      freq_tracking_trials / freq_joint_trials）。主軸は dpi400_slow。
出力: fig1_score.png, fig2_freq_tracking.png, fig2b_joint_timing.png,
      fig6_ternary.png, fig6b_style_vs_skill.png, fig7_timing.png, fig8_wrist_conditions.png

・参加者名は図に出さない（群 = 上級者 / 中級者 / 初級者 で表示）
・ρ は練度の順位（上級 3・中級 2・初級 1）と参加者平均の Spearman 順位相関
・体幹で前後を作る 1 名（体幹型）を除いた ρ も併記する（名前は出さない）
────────────────────────────────────────────────────────────────────
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
from scipy.stats import spearmanr

# 日本語フォント（環境にあるものを順に使う）
plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "IPAexGothic", "sans-serif"],
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 13, "axes.titlesize": 14,
})

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "skill_analysis/results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "figures_midterm")
OUT.mkdir(parents=True, exist_ok=True)

MAIN = "dpi400_slow"
CONDS = ["dpi400_slow", "dpi400_fast", "dpi800_slow", "dpi800_fast"]
CN = {"dpi400_slow": "400 DPI\nslow", "dpi400_fast": "400 DPI\nfast",
      "dpi800_slow": "800 DPI\nslow", "dpi800_fast": "800 DPI\nfast"}
GROUPS = ["A", "B", "C"]
GN = {"A": "上級者", "B": "中級者", "C": "初級者"}
GC = {"A": "#104281", "B": "#2a78d6", "C": "#86b6ef"}     # 単色の濃淡（濃い = 上級）
GM = {"A": "o", "B": "s", "C": "D"}                        # 色に頼らないよう形も変える
LV = {"A": 3, "B": 2, "C": 1}
OUTLIER = "Syou"                                           # 体幹型の 1 名（図には名前を出さない）
INK2, GRID = "#5f5f5a", "#e6e6e3"
JC = {"体幹": "#7d5ba6", "肩関節": "#2a78d6", "肘関節": "#eb6834"}


def rd(cond, name):
    return pd.read_csv(RES / cond / "data" / name, encoding="utf-8-sig")


def rho(df, col):
    """(全員の ρ, 体幹型 1 名を除いた ρ)"""
    r1 = spearmanr(df.group.map(LV), df[col])[0]
    d2 = df[df.participant != OUTLIER]
    r2 = spearmanr(d2.group.map(LV), d2[col])[0]
    return r1, r2


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=.6)
    ax.set_axisbelow(True)


def log_x(ax, ticks):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))


def group_legend(fig, y=0.0):
    h = [plt.Line2D([], [], ls="", marker=GM[g], color=GC[g], ms=10, label=GN[g]) for g in GROUPS]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, y))


def strip(ax, P, col, title, note):
    """群ごとに参加者の点（3 試行の平均）と群平均の横線。縦線 = その人の 3 試行の SD。"""
    for k, g in enumerate(GROUPS):
        d = P[P.group == g].reset_index(drop=True)
        xs = k + (np.arange(len(d)) - (len(d) - 1) / 2) * 0.14
        sd = d[col + "__sd"] if col + "__sd" in d else None
        if sd is not None:
            ax.errorbar(xs, d[col], sd, fmt="none", ecolor=GC[g], alpha=.45, lw=1.5)
        ax.scatter(xs, d[col], s=95, marker=GM[g], color=GC[g], edgecolor="white", lw=1.2, zorder=3)
        ax.hlines(d[col].mean(), k - .32, k + .32, color=INK2, lw=2.2, zorder=2)
    r1, r2 = rho(P, col)
    ax.set_xticks(range(3), [GN[g] for g in GROUPS])
    ax.set_xlim(-.6, 2.6)
    ax.set_title(f"{title}{note}\nρ = {r1:+.2f}（体幹型 1 名除外 {r2:+.2f}）", fontsize=13)
    grid(ax)


P = rd(MAIN, "persons.csv")

# ─────────────────────────── 図 1 スコアと練度
fig, axs = plt.subplots(1, 4, figsize=(16, 5.4))
for ax, (c, t, n) in zip(axs, [("rms_2d", "RMS 誤差 [mm]", "（↓良）"), ("on_target_pct", "on-target [%]", "（↑良）"),
                               ("exit_rate", "外れる頻度 [回/s]", "（↓良）"), ("hf_jitter", "2〜10 Hz の揺れ [mm/s]", "（↓良）")]):
    strip(ax, P, c, t, n)
group_legend(fig)
fig.tight_layout(rect=(0, .07, 1, 1))
fig.savefig(OUT / "fig1_score.png", dpi=150); plt.close(fig)

# ─────────────────────────── 図 2 周波数ごとの追従（カーソル）
T = rd(MAIN, "freq_tracking_trials.csv")
fig, axs = plt.subplots(2, 2, figsize=(13, 8.6))
for r, axis in enumerate(("x", "y")):
    an = "左右 x" if axis == "x" else "前後 y"
    d = T[T.axis == axis].groupby(["participant", "group", "f_hz"])[["lag_ms", "rel_err"]].mean().reset_index()
    for c, (col, lab) in enumerate((("lag_ms", "位相遅れ [ms]"), ("rel_err", "相対誤差 |C−T|/|T|"))):
        ax = axs[r, c]
        for g in GROUPS:
            m = d[d.group == g].groupby("f_hz")[col].mean()
            ax.plot(m.index, m.values, color=GC[g], marker=GM[g], ms=8, lw=2.4, label=GN[g])
        ax.set_xticks(sorted(d.f_hz.unique()))
        ax.set_xlabel(f"ターゲットの周波数 [Hz]（{an}）", color=INK2)
        ax.set_title(f"{an}: {lab}")
        grid(ax, "both")
axs[0, 0].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig2_freq_tracking.png", dpi=150); plt.close(fig)

# ─────────────────────────── 図 2' 関節ごとのタイミング（左右 x）
J = rd(MAIN, "freq_joint_trials.csv")
M = J.merge(T[["participant", "seed", "axis", "f_hz", "lag_ms"]], on=["participant", "seed", "axis", "f_hz"])
for p in ("forearm", "hand"):
    # ターゲットに対する関節の遅れ = カーソルの遅れ − 関節の手先に対する進み
    M["abs_" + p] = M.lag_ms - M["lead_" + p + "_ms"]
d = M[M.axis == "x"]
fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
for ax, (col, t) in zip(axs, (("lag_ms", "カーソル（手全体の出力）"), ("abs_forearm", "肘関節成分（前腕の振り）"),
                              ("abs_hand", "手首成分（手の振り）【参考】"))):
    for g in GROUPS:
        m = d[d.group == g].groupby("f_hz")[col].median()
        ax.plot(m.index, m.values, color=GC[g], marker=GM[g], ms=8, lw=2.4, label=GN[g])
    ax.axhline(0, color=INK2, lw=.9)
    ax.axvspan(0.2, 0.45, color="#eef3fa", zorder=0)
    ax.set_xticks(sorted(d.f_hz.unique()))
    ax.set_title(t)
    ax.set_xlabel("ターゲットの周波数 [Hz]（左右 x）", color=INK2)
    grid(ax, "both")
axs[0].set_ylabel("ターゲットに対する遅れ [ms]\n（0 より下 = ターゲットより先に動く）", color=INK2)
axs[0].legend(frameon=False)
fig.text(.5, .01, "試行ごとの値の群中央値。青帯 = 位相が安定に推定できる 0.275・0.425 Hz"
         "（0.05 Hz では位相 1° が 56 ms になり不安定）", ha="center", fontsize=11, color=INK2)
fig.tight_layout(rect=(0, .05, 1, 1))
fig.savefig(OUT / "fig2b_joint_timing.png", dpi=150); plt.close(fig)

# ─────────────────────────── 図 6 前後の動きの作り方（三角図）
TOP, LEFT, RIGHT = np.array([.5, np.sqrt(3) / 2]), np.array([0., 0.]), np.array([1., 0.])   # 体幹 / 肩 / 肘


def tern(trunk, shoulder, elbow):
    v = np.clip(np.array([trunk, shoulder, elbow], float), 0, None)
    v = v / v.sum()
    return v[0] * TOP + v[1] * LEFT + v[2] * RIGHT


def ternary_axes(ax):
    ax.plot(*np.array([LEFT, RIGHT, TOP, LEFT]).T, color="#1f1f1d", lw=1.4)
    for t in (.2, .4, .6, .8):                     # 目盛り線（各成分 20 % ごと）
        for a, b in ((tern(t, 1 - t, 0), tern(t, 0, 1 - t)), (tern(0, t, 1 - t), tern(1 - t, t, 0)),
                     (tern(0, 1 - t, t), tern(1 - t, 0, t))):
            ax.plot(*np.array([a, b]).T, color=GRID, lw=.8, zorder=0)
    for t in (.2, .4, .6, .8):
        ax.text(*(tern(t, 1 - t, 0) + [-.035, 0]), f"{int(t*100)}", ha="right", va="center", fontsize=10, color="#7d5ba6")
        ax.text(*(tern(0, t, 1 - t) + [0, -.035]), f"{int(t*100)}", ha="center", va="top", fontsize=10, color="#2a78d6")
        ax.text(*(tern(1 - t, 0, t) + [.03, 0]), f"{int(t*100)}", ha="left", va="center", fontsize=10, color="#eb6834")
    ax.text(*(TOP + [0, .05]), "体幹（肩ごと）100 %", ha="center", fontsize=13, color="#7d5ba6", weight="bold")
    ax.text(*(LEFT + [-.02, -.09]), "肩関節（上腕）100 %", ha="center", fontsize=13, color="#2a78d6", weight="bold")
    ax.text(*(RIGHT + [.02, -.09]), "肘関節（前腕）100 %", ha="center", fontsize=13, color="#eb6834", weight="bold")
    ax.set_xlim(-.15, 1.15); ax.set_ylim(-.16, 1.0); ax.set_aspect("equal"); ax.axis("off")


P8 = rd("dpi800_slow", "persons.csv").set_index("participant")
fig, ax = plt.subplots(figsize=(10.5, 9))
ternary_axes(ax)
for _, r in P.iterrows():
    a = tern(r.share_y_trunk, r.share_y_upperarm, r.share_y_forearm)
    if r.participant in P8.index:                  # 800 DPI slow（白抜き）と線で結ぶ
        q = P8.loc[r.participant]
        b = tern(q.share_y_trunk, q.share_y_upperarm, q.share_y_forearm)
        ax.plot(*np.array([a, b]).T, color=GC[r.group], lw=1.2, alpha=.6, zorder=2)
        ax.scatter(*b, s=110, marker=GM[r.group], facecolor="white", edgecolor=GC[r.group], lw=1.8, zorder=3)
    ax.scatter(*a, s=150, marker=GM[r.group], color=GC[r.group], edgecolor="white", lw=1.4, zorder=4)
for txt, xy in (("肩型", tern(.05, .85, .10)), ("肘型", tern(.05, .30, .65)), ("体幹型", tern(.80, .12, .08))):
    ax.annotate(txt, xy, xytext=(-70, 10) if txt != "肘型" else (20, 25), textcoords="offset points",
                fontsize=13, color=INK2, arrowprops=dict(arrowstyle="-", color="#b0b0ab"))
h = [plt.Line2D([], [], ls="", marker=GM[g], color=GC[g], ms=11, label=GN[g]) for g in GROUPS]
h += [plt.Line2D([], [], ls="", marker="o", color="#1f1f1d", ms=10, label="400 DPI（塗り）"),
      plt.Line2D([], [], ls="", marker="o", mfc="white", mec="#1f1f1d", ms=10, label="800 DPI（白抜き）")]
ax.legend(handles=h, loc="upper right", frameon=False, fontsize=12)
ax.set_title("前後（y）の動きを 3 つの成分がどう分担したか（slow、1 点 = 1 人）", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "fig6_ternary.png", dpi=150); plt.close(fig)

# ─────────────────────────── 図 6' 前後 = 個性、左右 = 技量
COMP = {"share_x_hand": ("左右: 手首", "#eb6834", "o", 1), "share_x_forearm": ("左右: 肘（符号反転）", "#f2a07a", "o", -1),
        "share_y_upperarm": ("前後: 肩", "#2a78d6", "s", 1), "share_y_forearm": ("前後: 肘", "#86b6ef", "s", 1),
        "share_y_trunk": ("前後: 体幹", "#7d5ba6", "s", 1)}
CS = {"dpi400_slow": "400s", "dpi400_fast": "400f", "dpi800_slow": "800s", "dpi800_fast": "800f"}
fig, ax = plt.subplots(figsize=(12.5, 6.6))
for c in CONDS:
    Q = rd(c, "persons.csv")
    Q = Q[Q.participant != OUTLIER]               # 体幹型 1 名が前後の個人差を独占しないよう除く
    for k, (lab, col, mk, s) in COMP.items():
        Dv = Q[k].std() / Q[k + "__sd"].mean()   # 判別比 D
        r = s * spearmanr(Q.group.map(LV), Q[k])[0]
        ax.scatter(Dv, r, s=130, color=col, marker=mk, edgecolor="white", zorder=3)
        ax.annotate(CS[c], (Dv, r), xytext=(6, 4), textcoords="offset points", fontsize=9, color=INK2)
h = [plt.Line2D([], [], ls="", marker=mk, color=col, ms=10, label=lab) for lab, col, mk, _ in COMP.values()]
ax.legend(handles=h, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
log_x(ax, [0.5, 1, 2, 3, 5])
ax.axvline(1, color=INK2, lw=.8, ls=":"); ax.axhline(0, color=INK2, lw=.8)
ax.set_xlabel("判別比 D = 人の間の SD / 同じ人の試行間 SD（対数、> 1 で個人特有）", color=INK2)
ax.set_ylabel("練度との順位相関 ρ", color=INK2)
ax.set_title("左右（○）は 4 条件とも練度と同じ向き、前後（□）は向きがばらばら（体幹型 1 名除外）")
grid(ax, "both")
fig.tight_layout()
fig.savefig(OUT / "fig6b_style_vs_skill.png", dpi=150); plt.close(fig)

# ─────────────────────────── 図 7 時系列での協調度
fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
for ax, (c, t, n) in zip(axs, [("xr_upperarm_forearm", "r(肩関節, 肘関節)", "（− = 補い合う）"),
                               ("opp_upperarm_forearm", "肩と肘が逆向きに動いた割合", ""),
                               ("xlag_forearm_hand_ms", "ラグ 肘→手首 [ms]【参考】", "（− = 手首が先）")]):
    strip(ax, P, c, t, n)
    if c != "opp_upperarm_forearm":
        ax.axhline(0, color=INK2, lw=.8)
group_legend(fig)
fig.tight_layout(rect=(0, .07, 1, 1))
fig.savefig(OUT / "fig7_timing.png", dpi=150); plt.close(fig)

# ─────────────────────────── 図 8 左右の手首シェア：4 条件
fig, ax = plt.subplots(figsize=(12, 5.8))
for k, c in enumerate(CONDS):
    Q = rd(c, "persons.csv")
    for g in GROUPS:
        dd = Q[Q.group == g]
        x0 = k + {"A": -.24, "B": 0, "C": .24}[g]
        xs = x0 + (np.arange(len(dd)) - (len(dd) - 1) / 2) * .06
        ax.scatter(xs, dd.share_x_hand, s=60, marker=GM[g], color=GC[g], alpha=.6, edgecolor="white", zorder=3)
        ax.hlines(dd.share_x_hand.mean(), x0 - .1, x0 + .1, color=GC[g], lw=4, zorder=4)
    r1, r2 = rho(Q, "share_x_hand")
    ax.text(k, .615, f"ρ = {r1:+.2f}\n（除外 {r2:+.2f}）", ha="center", va="top", fontsize=11, color=INK2)
ax.set_xticks(range(4), [CN[c] for c in CONDS])
ax.set_ylim(0, .62)
ax.set_ylabel("左右（x）の動きを手首が作った割合【参考】", color=INK2)
ax.set_title("どの条件でも、上級者ほど左右の動きに手首を使う（横線 = 群平均）")
grid(ax)
group_legend(fig)
fig.tight_layout(rect=(0, .07, 1, 1))
fig.savefig(OUT / "fig8_wrist_conditions.png", dpi=150); plt.close(fig)

print("出力:", OUT.resolve())
for f in sorted(OUT.glob("fig*.png")):
    print("  ", f.name)