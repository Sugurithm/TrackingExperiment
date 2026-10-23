import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 文字サイズを大きくし、全体的に「太く」する設定
plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "IPAexGothic", "sans-serif"],
    "font.weight": "bold",          # フォント全体を太字に
    "axes.labelweight": "bold",     # 軸ラベルを太字に
    "axes.titleweight": "bold",     # タイトルを太字に
    "axes.spines.top": False, 
    "axes.spines.right": False, 
    "axes.linewidth": 2.5,          # 軸の線を太く（デフォルトは0.8）
    "xtick.major.width": 2.5,       # x軸の目盛り線を太く
    "ytick.major.width": 2.5,       # y軸の目盛り線を太く
    "xtick.major.size": 8,          # 目盛りの長さも少し長く
    "ytick.major.size": 8,
    "font.size": 24,                # 文字サイズ
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
})

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "skill_analysis/results")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "figures_midterm")
OUT.mkdir(parents=True, exist_ok=True)

CONDS = ["dpi400_slow", "dpi400_fast", "dpi800_slow", "dpi800_fast"]
CN = {"dpi400_slow": "400 DPI\nslow", "dpi400_fast": "400 DPI\nfast",
      "dpi800_slow": "800 DPI\nslow", "dpi800_fast": "800 DPI\nfast"}
GROUPS = ["A", "B", "C"]
GN = {"A": "上級者", "B": "中級者", "C": "初級者"}
GC = {"A": "#104281", "B": "#2a78d6", "C": "#86b6ef"}
GM = {"A": "o", "B": "s", "C": "D"}
INK2, GRID = "#5f5f5a", "#e6e6e3"

def rd(cond, name):
    return pd.read_csv(RES / cond / "data" / name, encoding="utf-8-sig")

def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=2.0) # グリッド線も太く
    ax.set_axisbelow(True)

def group_legend(fig, y=0.0):
    # 凡例のマーカーサイズと枠線を太く
    h = [plt.Line2D([], [], ls="", marker=GM[g], color=GC[g], ms=18, markeredgewidth=2.0, label=GN[g]) for g in GROUPS]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, y), fontsize=24)

# ─────────────────────────── 図 8 左右の手首シェア：4 条件
fig, ax = plt.subplots(figsize=(12, 7.5))

for k, c in enumerate(CONDS):
    Q = rd(c, "persons.csv")
    for g in GROUPS:
        dd = Q[Q.group == g]
        x0 = k + {"A": -.24, "B": 0, "C": .24}[g]
        xs = x0 + (np.arange(len(dd)) - (len(dd) - 1) / 2) * .06
        
        # 点の大きさ(s)と枠線の太さ(linewidth)を拡大
        ax.scatter(xs, dd.share_x_hand, s=180, marker=GM[g], color=GC[g], 
                   alpha=.7, edgecolor="white", linewidth=2.0, zorder=3)
        
        # 平均値の横線をさらに太く（lw=6）
        ax.hlines(dd.share_x_hand.mean(), x0 - .1, x0 + .1, color=GC[g], lw=6, zorder=4)

ax.set_xticks(range(4), [CN[c] for c in CONDS])
ax.set_ylim(0, .65)
ax.set_ylabel("左右の動きに手首を使った割合", color="black")

# グリッドの描画
grid(ax)

# 凡例の追加
group_legend(fig, y=0.01)

# レイアウト調整
fig.tight_layout(rect=(0, .12, 1, 1))
fig.savefig(OUT / "fig8_wrist_conditions.png", dpi=150)
plt.close(fig)

print("出力:", OUT.resolve() / "fig8_wrist_conditions.png")