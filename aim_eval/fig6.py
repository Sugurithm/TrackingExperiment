import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 日本語フォント設定とベースフォントサイズの拡大
plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "IPAexGothic", "sans-serif"],
    "axes.spines.top": False, "axes.spines.right": False, 
    "font.size": 18,
})

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "skill_analysis/results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "figures_midterm")
OUT.mkdir(parents=True, exist_ok=True)

MAIN = "dpi400_slow"
GROUPS = ["A", "B", "C"]
GN = {"A": "上級者", "B": "中級者", "C": "初級者"}
GC = {"A": "#104281", "B": "#2a78d6", "C": "#86b6ef"}
GM = {"A": "o", "B": "s", "C": "D"}
INK2, GRID = "#5f5f5a", "#e6e6e3"

def rd(cond, name):
    return pd.read_csv(RES / cond / "data" / name, encoding="utf-8-sig")

P = rd(MAIN, "persons.csv")

# ─────────────────────────── 図 6 動きの作り方（三角図）
TOP, LEFT, RIGHT = np.array([.5, np.sqrt(3) / 2]), np.array([0., 0.]), np.array([1., 0.])

def tern(trunk, shoulder, elbow):
    v = np.clip(np.array([trunk, shoulder, elbow], float), 0, None)
    v = v / v.sum()
    return v[0] * TOP + v[1] * LEFT + v[2] * RIGHT

def ternary_axes(ax):
    ax.plot(*np.array([LEFT, RIGHT, TOP, LEFT]).T, color="#1f1f1d", lw=1.4)
    
    # グリッド線の描画
    for t in (.2, .4, .6, .8):
        for a, b in ((tern(t, 1 - t, 0), tern(t, 0, 1 - t)), (tern(0, t, 1 - t), tern(1 - t, t, 0)),
                     (tern(0, 1 - t, t), tern(1 - t, 0, t))):
            ax.plot(*np.array([a, b]).T, color=GRID, lw=.8, zorder=0)
            
    # 軸の目盛り（数値を拡大し、色を黒に統一）
    for t in (.2, .4, .6, .8):
        ax.text(*(tern(t, 1 - t, 0) + [-.035, 0]), f"{int(t*100)}", ha="right", va="center", fontsize=16, color="black")
        ax.text(*(tern(0, t, 1 - t) + [0, -.040]), f"{int(t*100)}", ha="center", va="top", fontsize=16, color="black")
        ax.text(*(tern(1 - t, 0, t) + [.03, 0]), f"{int(t*100)}", ha="left", va="center", fontsize=16, color="black")
    
    # 各辺の中点座標を計算
    mid_left = (TOP + LEFT) / 2
    mid_right = (TOP + RIGHT) / 2
    mid_bottom = (LEFT + RIGHT) / 2
    
    # 辺ラベル（各辺に沿って配置、角度と矢印を指定）
    # 左辺
    ax.text(mid_left[0] - 0.12, mid_left[1] + 0.02, "肩の移動 [%]", 
            ha="center", va="center", rotation=60, fontsize=20, color="black", weight="bold")
    # 右辺
    ax.text(mid_right[0] + 0.12, mid_right[1] + 0.02, "手首の移動 [%]", 
            ha="center", va="center", rotation=-60, fontsize=20, color="black", weight="bold")
    # 底辺
    ax.text(mid_bottom[0], mid_bottom[1] - 0.12, "肘の移動 [%]", 
            ha="center", va="center", rotation=0, fontsize=20, color="black", weight="bold")
    
    # ラベルが切れないように描画範囲を少し広げる
    ax.set_xlim(-.25, 1.25)
    ax.set_ylim(-.20, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

P8 = rd("dpi800_slow", "persons.csv").set_index("participant")
fig, ax = plt.subplots(figsize=(10.5, 9))
ternary_axes(ax)

# データのプロット
for _, r in P.iterrows():
    a = tern(r.share_y_trunk, r.share_y_upperarm, r.share_y_forearm)
    if r.participant in P8.index:
        q = P8.loc[r.participant]
        b = tern(q.share_y_trunk, q.share_y_upperarm, q.share_y_forearm)
        ax.plot(*np.array([a, b]).T, color=GC[r.group], lw=1.2, alpha=.6, zorder=2)
        ax.scatter(*b, s=110, marker=GM[r.group], facecolor="white", edgecolor=GC[r.group], lw=1.8, zorder=3)
    ax.scatter(*a, s=150, marker=GM[r.group], color=GC[r.group], edgecolor="white", lw=1.4, zorder=4)

# 凡例の作成
h = [plt.Line2D([], [], ls="", marker=GM[g], color=GC[g], ms=13, label=GN[g]) for g in GROUPS]
h += [plt.Line2D([], [], ls="", marker="o", color="#1f1f1d", ms=12, label="400 DPI（塗り）"),
      plt.Line2D([], [], ls="", marker="o", mfc="white", mec="#1f1f1d", ms=12, label="800 DPI（白抜き）")]
ax.legend(handles=h, loc="upper right", frameon=False, fontsize=16)

fig.tight_layout()
fig.savefig(OUT / "fig6_ternary.png", dpi=150)
plt.close(fig)

print("出力:", OUT.resolve() / "fig6_ternary.png")