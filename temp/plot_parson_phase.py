#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
plot_person_phase.py — 「人による違いがあるか」を 1 枚で見る図（寄与率 と 位相ずれ を並べる）

  python temp/plot_person_phase.py --compare slow_all fps_vs_non [--fmin 0.275] [--band slow]
  (先に freq_phase.py を実行して trial_freq.csv を作る)

各点 = 1 試行（周波数 fmin 以上の成分を平均）．横軸 = 人，黒線 = 人の平均．
上段 = 同相の寄与率 [%]，下段 = 位相ずれ [ms]（+ = リンクが手先より先）．
人の間の広がりが試行のばらつきより大きければ「人による違い」．タイトルに ICC と p（並べ替え）．
出力: <out>/<比較名>/freq/person_phase_<band>_<x|y>.png
"""
import argparse, csv, os, sys
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stroke_compare as SC   # noqa: E402
import freq_phase as FP       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "compare_config.py"))
    ap.add_argument("--compare", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fmin", type=float, default=0.275, help="この周波数以上の成分だけ平均（低周波の位相は不安定）")
    ap.add_argument("--perm", type=int, default=2000)
    args = ap.parse_args()
    cfg = SC.load_config(args.config)
    out = args.out or (cfg["OUT"] if os.path.isabs(cfg["OUT"]) else os.path.join(HERE, cfg["OUT"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    for c in [c for c in cfg["COMPARES"] if not args.compare or c["name"] in args.compare]:
        d = os.path.join(out, c["name"], "freq")
        f = os.path.join(d, "trial_freq.csv")
        if not os.path.exists(f):
            print(f"[{c['name']}] trial_freq.csv がありません（先に freq_phase.py）"); continue
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        persons = sorted({r["person"] for r in rows})
        grp = {r["person"]: r.get("group", "") for r in rows}
        gcol = {g: col for g, col in zip(sorted(set(grp.values())), ("tab:red", "tab:blue", "tab:green"))}
        for band in sorted({r["band"] for r in rows}):
            for ax_name in ("x", "y"):
                fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
                for ci, link in enumerate(("U", "F", "H")):
                    for ri, (key, scale, ylab) in enumerate((("share_re", 100, "in-phase share [%]"), ("lead_ms", 1, "lead vs hand [ms] (+ = link first)"))):
                        a = axs[ri][ci]
                        per = {}
                        for r in rows:
                            if r["band"] != band or r["axis"] != ax_name or r["link"] != link or float(r["f_hz"]) < args.fmin:
                                continue
                            if key == "lead_ms" and r["ok_link"] != "1":
                                continue
                            v = r[key]
                            if v in ("", "nan"):
                                continue
                            per.setdefault((r["person"], r["seed"], r["attempt"]), []).append(float(v))
                        tri = {}
                        for (p, s, at), v in per.items():
                            tri.setdefault(p, []).append(scale * float(np.mean(v)))
                        for i, p in enumerate(persons):
                            v = tri.get(p, [])
                            a.scatter([i + 0.0] * len(v), v, s=22, color=gcol.get(grp[p], "gray"), alpha=0.75, zorder=3)
                            if v:
                                a.hlines(np.mean(v), i - 0.3, i + 0.3, color="k", lw=2, zorder=4)
                        res = FP.icc_test(tri, rng, args.perm) if tri else None
                        a.set_title(f"{link}  " + (f"ICC {res['ICC']:.2f}, p={res['p_perm']:.3f}" if res else "ICC n/a"), fontsize=10)
                        a.axhline(0, color="#bbb", lw=0.7)
                        a.set_ylabel(ylab, fontsize=8)
                        a.set_xticks(range(len(persons)))
                        a.set_xticklabels(persons, rotation=45, fontsize=8)
                fig.suptitle(f"{c['name']} / {band} / {'horizontal (x)' if ax_name == 'x' else 'vertical (y)'} components, f >= {args.fmin:g} Hz   (dots = trials, bar = person mean; colour = group)", fontsize=10)
                fig.tight_layout()
                path = os.path.join(d, f"person_phase_{band}_{ax_name}.png")
                fig.savefig(path, dpi=130)
                plt.close(fig)
                print("->", path)


if __name__ == "__main__":
    main()