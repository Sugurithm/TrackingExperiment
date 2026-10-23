"""
arm_usage3.py — 腕の使い方を 3 つに分けて確かめる（上腕のひねり と 肘の曲げ伸ばし を同じ「前腕の振り」として扱う版）
────────────────────────────────────────────────────────────────────
  python skill_analysis/arm_usage3.py                     # 名前を伏せて表示（上級1, 上級2, …）
  python skill_analysis/arm_usage3.py --names             # 参加者名で表示
  python skill_analysis/arm_usage3.py --csv analysis_out/20261022/between/trials.csv --out figures_arm3

入力: aim_eval の解析結果を集めた trials.csv（between/trials.csv）の dofmm_* 列。
      aim_eval/a8_dof.py が出す「その動きだけを実測どおり動かし、他を平均姿勢に止めたとき、
      手先が何 mm 動くか（変位の SD）」を使う。生データは要らない。

腕の使い方（3 つ）
    上腕の振り   肩・仰角 + 肩・方位          上腕（肩→肘）を振る。肘の位置が動く           … 肩エイム
    前腕の振り   肩・回旋 + 肘・屈曲          肘の位置はそのままで前腕の向きが変わる        … 肘エイム
                 （上腕のひねり + 肘の曲げ伸ばし）
    手首         手首・屈曲 + 手首・偏位      手（手首→指の付け根）を振る【参考】           … 手首エイム
  割合 = それぞれの mm² ÷ 3 つの mm² の合計（複数の角度をまとめるものは mm² を足す）。
  体幹（肩の位置そのものの移動）はこの分解に入らない（肩を原点にしている）。多くの人で全体の 1〜4 %。

まとめる理由
  上腕のひねりと肘の曲げ伸ばしは、どちらも「肘を支点に前腕が動く」動きで、データ上も強く逆相関
  （r ≈ −0.7〜−0.86）しており、カメラの肘点の誤差で分け方が不安定になりうる。まとめると安定する。
  ただし mm² の足し算なので、2 つが打ち消し合う分（逆相関の分）は差し引かれず、前腕の振りは
  実際より大きめに出る。

注意
  ・手首は MoCap で未照合。肩の回旋も未照合。
  ・各動きを単独で動かしたときの量なので、互いに打ち消し合う分があると合計は実際の手先の動きより大きくなる。
  ・課題の周波数以外（ノイズ・修正）も含んだ量。
出力: arm_usage3.png（参加者 × 条件の割合）、arm_usage3_by_group.png（群平均）、arm_usage3.csv
────────────────────────────────────────────────────────────────────
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans",
                                     "IPAexGothic", "sans-serif"],
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 13})

GROUPS = {"A": ["Syou", "Yuuya", "Suguru"], "B": ["Kakeru", "Xin"], "C": ["Kosuke", "Yuuki", "Suzuka", "Keito"]}
ALIAS = {"yuya": "Yuuya", "yuuya": "Yuuya", "kakeru": "Kakeru", "xin": "Xin", "syou": "Syou", "suguru": "Suguru",
         "kosuke": "Kosuke", "yuuki": "Yuuki", "suzuka": "Suzuka", "keito": "Keito"}
GN = {"A": "上級", "B": "中級", "C": "初級", "-": "未分類"}
USAGE = {  # 名前: (使う列, 色)  青・緑・オレンジ
    "上腕の振り": (["dofmm_sh_elev", "dofmm_sh_azim"], "#2a78d6"),
    "前腕の振り（ひねり＋曲げ伸ばし）": (["dofmm_hum_rot", "dofmm_el_flex"], "#1baa7d"),
    "手首【参考】": (["dofmm_wr_flex", "dofmm_wr_dev"], "#eb6834"),
}
CONDS = [(400, "slow"), (400, "fast"), (800, "slow"), (800, "fast")]


def group_of(name):
    return next((g for g, m in GROUPS.items() if name in m), "-")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(here, "..", "analysis_out", "20261022", "between", "trials.csv"))
    ap.add_argument("--out", default=os.path.join(here, "results_arm3"))
    ap.add_argument("--names", action="store_true", help="参加者名で表示する（既定は伏せる）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = pd.read_csv(a.csv)
    d["name"] = d.participant.astype(str).str.lower().map(ALIAS).fillna(d.participant)
    d["group"] = d.name.map(group_of)
    for lab, (cols, _) in USAGE.items():
        d[lab] = (d[cols] ** 2).sum(1)                       # mm²
    labs = list(USAGE)
    tot = d[labs].sum(1)
    for lab in labs:
        d[lab] = d[lab] / tot
    # 参加者 × 条件で 3 試行を平均
    m = d.groupby(["name", "group", "dpi", "band"])[labs].mean().reset_index()
    m["主な使い方"] = m[labs].idxmax(1).str.replace(r"[（【].*", "", regex=True)

    # 表示名（伏せる場合は 上級1, 上級2 …）
    order = [n for g in ("A", "B", "C") for n in GROUPS[g] if n in set(m.name)] + \
            sorted(n for n in set(m.name) if group_of(n) == "-")
    disp = {}
    cnt = {}
    for n in order:
        g = group_of(n)
        cnt[g] = cnt.get(g, 0) + 1
        disp[n] = n if a.names else f"{GN[g]}{cnt[g]}"
    m["表示名"] = m.name.map(disp)
    m.sort_values(["dpi", "band", "group", "name"]).to_csv(os.path.join(a.out, "arm_usage3.csv"),
                                                           index=False, encoding="utf-8-sig")

    # 図 1: 参加者 × 条件
    fig, axs = plt.subplots(1, 4, figsize=(19, 6.6), sharey=True)
    for ax, (dpi, band) in zip(axs, CONDS):
        s = m[(m.dpi == dpi) & (m.band == band)].set_index("name")
        rows = [n for n in order if n in s.index]
        y = np.arange(len(rows))[::-1]
        left = np.zeros(len(rows))
        for lab in labs:
            v = s.loc[rows, lab].values
            ax.barh(y, v, left=left, color=USAGE[lab][1], height=.7, edgecolor="white", lw=1.5,
                    label=lab if (dpi, band) == CONDS[0] else None)
            for yi, l0, vi in zip(y, left, v):
                if vi >= .08:
                    ax.text(l0 + vi / 2, yi, f"{vi*100:.0f}", ha="center", va="center", fontsize=10, color="white")
            left += v
        ax.set_yticks(y, [disp[n] for n in rows])
        ax.set_xlim(0, 1); ax.set_title(f"{dpi} DPI・{band}")
        ax.set_xlabel("手先の動きの割合", color="#5f5f5a")
    fig.legend(loc="lower center", ncol=3, frameon=False)
    fig.suptitle("腕の使い方（各動きだけで手先が動く量の割合、3 試行の平均）", fontsize=15)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(os.path.join(a.out, "arm_usage3.png"), dpi=150); plt.close(fig)

    # 図 2: 群平均
    g = m[m.group != "-"].groupby(["group", "dpi", "band"])[labs].mean().reset_index()
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.2), sharey=True)
    for ax, (dpi, band) in zip(axs, CONDS):
        s = g[(g.dpi == dpi) & (g.band == band)].set_index("group")
        rows = [x for x in "ABC" if x in s.index]
        y = np.arange(len(rows))[::-1]
        left = np.zeros(len(rows))
        for lab in labs:
            v = s.loc[rows, lab].values
            ax.barh(y, v, left=left, color=USAGE[lab][1], height=.6, edgecolor="white", lw=1.5,
                    label=lab if (dpi, band) == CONDS[0] else None)
            for yi, l0, vi in zip(y, left, v):
                if vi >= .08:
                    ax.text(l0 + vi / 2, yi, f"{vi*100:.0f}", ha="center", va="center", fontsize=11, color="white")
            left += v
        ax.set_yticks(y, [GN[x] + "者" for x in rows])
        ax.set_xlim(0, 1); ax.set_title(f"{dpi} DPI・{band}")
    fig.legend(loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(os.path.join(a.out, "arm_usage3_by_group.png"), dpi=150); plt.close(fig)

    # 画面にも要約
    pd.set_option("display.width", 200)
    show = m.copy()
    for lab in labs:
        show[lab] = (show[lab] * 100).round(0).astype(int)
    print(show[["表示名", "dpi", "band"] + labs + ["主な使い方"]].sort_values(["dpi", "band", "表示名"]).to_string(index=False))
    print("出力:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()