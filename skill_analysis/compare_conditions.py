"""条件どうしの比較表（解析済みの条件だけを並べる。データは統合せず、各条件の group_tests.csv を並べるだけ）

  python skill_analysis/compare_conditions.py [--out skill_analysis/results]

出力: results/_compare/
  condition_matrix.csv   指標 × 条件の ρ・p（全員 / Syou 除外）と「練度差あり」の印
  condition_matrix.png   指標 × 条件の ρ のヒートマップ（印 = p < 0.10）
  compare.md             「特定の条件でだけ見られる / どの条件でも見られる / どの条件でも見られない」特徴の一覧
"""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skill_eval import config as K               # noqa: E402
from skill_eval.stats import METRICS, MLAB       # noqa: E402
from skill_eval.figures import _plt, _save       # noqa: E402

P_FLAG = 0.10
RHO_CONS = 0.4     # 「どの条件でも同じ向き」とみなす |ρ| の下限
RHO_REV = 0.5      # 「条件で向きが逆」とみなす |ρ| の下限


def _read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=K.RESULTS)
    a = ap.parse_args()
    conds = [c["name"] for c in K.CONDITIONS if os.path.exists(os.path.join(a.out, c["name"], "data", "group_tests.csv"))]
    if not conds:
        sys.exit("解析済みの条件がありません（先に run_skill.py を実行）")
    tab = {c: {r["metric"]: r for r in _read(os.path.join(a.out, c, "data", "group_tests.csv"))} for c in conds}
    od = os.path.join(a.out, "_compare"); os.makedirs(od, exist_ok=True)

    rows = []
    for k, lab, fam, _ in METRICS:
        row = {"metric": k, "label": lab, "family": fam}
        for c in conds:
            r = tab[c].get(k, {})
            for tag in ("all", "noSyou"):
                row[f"{c}_rho_{tag}"] = _num(r.get(f"rho_{tag}"))
                row[f"{c}_p_{tag}"] = _num(r.get(f"p_{tag}"))
            row[f"{c}_flag"] = bool(row[f"{c}_p_all"] < P_FLAG and row[f"{c}_p_noSyou"] < P_FLAG)
        rows.append(row)
    with open(os.path.join(od, "condition_matrix.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ヒートマップ（発散: 青 = 負、橙 = 正、灰 = 0）
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#e6e5e0", "#f2a27f", "#c04a1c"])
    cols = [(c, t) for c in conds for t in ("all", "noSyou")]
    M = np.array([[r[f"{c}_rho_{t}"] for c, t in cols] for r in rows], float)
    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 4, 0.26 * len(rows) + 1.5))
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i, r in enumerate(rows):
        for j, (c, t) in enumerate(cols):
            p = r[f"{c}_p_{t}"]
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}" + ("*" if p < P_FLAG else ""), ha="center", va="center", fontsize=6.5,
                        color="#0b0b0b")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c}\n{'全員' if t == 'all' else 'Syou除外'}" for c, t in cols], fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["label"] for r in rows], fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=.5, label="練度との ρ")
    ax.set_title(f"条件 × 指標: 練度との順位相関（* = p < {P_FLAG}）", loc="left", fontsize=10)
    fig.tight_layout(); _save(fig, od, "condition_matrix.png"); plt.close(fig)

    md = f"# 条件間の比較（解析済み: {', '.join(conds)}）\n\n"
    md += f"「練度差あり」= 全員でも Syou 除外でも p < {P_FLAG}（探索的な目安）。各条件は独立に解析した結果を並べただけです。\n\n"
    if os.path.exists(os.path.join(od, "discussion.md")):
        md += "**考察（手書き）: [discussion.md](discussion.md)**（この表は再実行で作り直されますが、考察は上書きされません）\n\n"
    if len(conds) < 2:
        md += "**まだ 1 条件しか解析していないため、条件間の比較はできません。** 他の条件を `run_skill.py --all` で解析してから再実行してください。\n\n"
    every = [r for r in rows if all(r[f"{c}_flag"] for c in conds)]
    some = [r for r in rows if any(r[f"{c}_flag"] for c in conds) and not all(r[f"{c}_flag"] for c in conds)]
    md += "## どの条件でも練度差あり\n\n" + ("\n".join(f"- {r['label']}" for r in every) or "- なし") + "\n\n"
    md += "## 特定の条件でだけ練度差あり\n\n" + ("\n".join(
        f"- {r['label']}: " + "、".join(c for c in conds if r[f"{c}_flag"]) for r in some) or "- なし") + "\n\n"
    # 向きの一貫性（p によらず ρ の符号と大きさで見る。人数が少ないので p だけで分けると取りこぼす）
    def rhos(r):
        return np.array([r[f"{c}_rho_{t}"] for c in conds for t in ("all", "noSyou")], float)
    cons = [r for r in rows if np.all(np.isfinite(rhos(r))) and
            (np.all(rhos(r) >= RHO_CONS) or np.all(rhos(r) <= -RHO_CONS))]
    rev = [r for r in rows if np.nanmax(rhos(r)) >= RHO_REV and np.nanmin(rhos(r)) <= -RHO_REV]
    md += (f"## どの条件でも同じ向き（全条件・全員 / Syou 除外とも |ρ| ≥ {RHO_CONS} で同符号）\n\n"
           + ("\n".join(f"- {r['label']}: " + ("高練度ほど大きい" if np.mean(rhos(r)) > 0 else "高練度ほど小さい")
                        + f"（ρ {np.min(rhos(r)):+.2f}〜{np.max(rhos(r)):+.2f}）" for r in cons) or "- なし") + "\n\n")
    md += (f"## 条件によって向きが逆になる（ある条件で ρ ≥ +{RHO_REV}、別の条件で ≤ −{RHO_REV}）\n\n"
           + ("\n".join(f"- {r['label']}: " + "、".join(
               f"{c} {r[f'{c}_rho_all']:+.2f}" for c in conds) for r in rev) or "- なし") + "\n\n")
    md += "## ρ の一覧\n\n| 指標 | " + " | ".join(f"{c} 全員 | {c} 除外" for c in conds) + " |\n|---|" + "---|---|" * len(conds) + "\n"
    for r in rows:
        cells = []
        for c in conds:
            for t in ("all", "noSyou"):
                v, p = r[f"{c}_rho_{t}"], r[f"{c}_p_{t}"]
                cells.append(("–" if not np.isfinite(v) else f"{v:+.2f}" + ("*" if p < P_FLAG else "")))
        md += f"| {r['label']} | " + " | ".join(cells) + " |\n"
    md += "\n![条件 × 指標](condition_matrix.png)\n"
    open(os.path.join(od, "compare.md"), "w", encoding="utf-8").write(md)
    print(f"→ {od}")


if __name__ == "__main__":
    main()
