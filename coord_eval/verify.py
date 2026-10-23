"""合成データで coord_eval の指標を検証する（実データの前に 1 回、コードを変えたら毎回）

  python -m coord_eval.verify [--work <作業フォルダ>]

1. make_synthetic.py で答えの分かっている 6 人（× 400 / 800 DPI × slow / fast × seed 3）を作る
   （ノイズあり = 実データと同じ条件 / ノイズなし = 指標の真値）
2. 両方を run.py で解析し、
   (a) ノイズありの値がノイズなしの値（真値）をどれだけ再現するか（試行ごとの MAE と相関）
   (b) 入れた「型」が指標に現れるか（向きの安定化・握り・振幅に応じた関節の使い分け・DPI で使い方を変える人）
   を確かめて、OK / NG の表を出す。
"""
import argparse, csv, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load(p, key=("participant", "dpi", "trial")):
    with open(p, encoding="utf-8-sig") as f:
        return {tuple(r[k] for k in key): r for r in csv.DictReader(f)}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    work = a.work or tempfile.mkdtemp(prefix="coord_verify_")
    syn, cln = os.path.join(work, "syn"), os.path.join(work, "clean")
    py = sys.executable
    for d, extra in ((syn, []), (cln, ["--clean"])):
        if not os.path.exists(os.path.join(d, "groups.json")):
            subprocess.run([py, os.path.join(HERE, "make_synthetic.py"), d] + extra, check=True, stdout=subprocess.DEVNULL)
    for d, o, extra in ((syn, "out_syn", []), (cln, "out_clean", ["--no-bias"])):
        subprocess.run([py, "-m", "coord_eval.run", "--recordings", d, "--groups", os.path.join(d, "groups.json"),
                        "--out", os.path.join(work, o)] + extra, check=True, stdout=subprocess.DEVNULL, cwd=ROOT)
    A = load(os.path.join(work, "out_clean", "data", "trials.csv"))
    B = load(os.path.join(work, "out_syn", "data", "trials.csv"))
    PA = load(os.path.join(work, "out_clean", "data", "person_means.csv"), ("participant",))
    PB = load(os.path.join(work, "out_syn", "data", "person_means.csv"), ("participant",))
    INV = list(csv.DictReader(open(os.path.join(work, "out_syn", "data", "invariance_person.csv"), encoding="utf-8-sig")))

    res = []

    def check(name, ok, detail):
        res.append((name, bool(ok), detail))

    # ---- (a) 真値の再現（試行ごと）
    tol = {"gshare_upperarm": .03, "gshare_forearm": .03, "gshare_hand": .03, "eshare3_distal": .03,
           "N_eff3": .08, "C_glob3": .05, "mshare_prox": .04, "mshare_hand": .04, "N_move": .10, "C_move3": .08,
           "rslope_prox": .05, "K_orient": .10, "psi_sdc_per100": 1.5, "psis_sd_per100": 2.5,
           "psis_vs_psi_slope": .10, "dofshare_hum_rot": .03}
    for c, t in tol.items():
        x = np.array([f(A[k].get(c)) for k in A]); y = np.array([f(B[k].get(c)) if k in B else np.nan for k in A])
        ok = np.isfinite(x) & np.isfinite(y)
        mae = float(np.mean(np.abs(x[ok] - y[ok]))) if ok.any() else np.nan
        r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 3 else np.nan
        check(f"真値の再現: {c}", mae <= t, f"MAE {mae:.3f}（許容 {t}）、r {r:.3f}、n {int(ok.sum())}")

    # ---- (b) 入れた型が現れるか（ノイズありのデータで）
    g = lambda p, c: f(PB[(p,)].get(c))
    check("向きの安定化: SynForearm（κ = 1）の K ≈ 1", g("SynForearm", "K_orient") > 0.85, f"K = {g('SynForearm', 'K_orient'):.2f}")
    check("向きの安定化: SynStab（κ = 0.7）> SynWhole（κ = 0）+ 0.3",
          g("SynStab", "K_orient") > g("SynWhole", "K_orient") + 0.3,
          f"{g('SynStab', 'K_orient'):.2f} vs {g('SynWhole', 'K_orient'):.2f}")
    check("握り: SynLoose（マウスの回転 = 0.3 × 手の回転）の傾き 0.15〜0.45",
          0.15 <= g("SynLoose", "psis_vs_psi_slope") <= 0.45, f"{g('SynLoose', 'psis_vs_psi_slope'):.2f}")
    check("握り: SynWhole（固い握り）の傾き > 0.7", g("SynWhole", "psis_vs_psi_slope") > 0.7,
          f"{g('SynWhole', 'psis_vs_psi_slope'):.2f}")
    check("振幅による使い分け: SynEcon の rslope 近位 > SynWhole + 0.05",
          g("SynEcon", "rslope_prox") > g("SynWhole", "rslope_prox") + 0.05,
          f"{g('SynEcon', 'rslope_prox'):+.3f} vs {g('SynWhole', 'rslope_prox'):+.3f}")
    for band in ("slow", "fast"):
        r = next((x for x in INV if x["participant"] == "SynShift" and x["metric"] == "gshare_hand"
                  and x["factor"] == "DPI" and x["stratum"] == band), None)
        check(f"DPI で使い方を変える人: SynShift の手のシェアが 800 で増える（{band}）",
              r is not None and r["verdict"] == "変化" and f(r["d"]) > 0, f"{r['verdict']} d = {f(r['d']):+.3f}" if r else "–")
        for p in ("SynWhole", "SynForearm", "SynStab"):
            r = next((x for x in INV if x["participant"] == p and x["metric"] == "gshare_hand"
                      and x["factor"] == "DPI" and x["stratum"] == band), None)
            check(f"DPI で使い方を変えない人: {p} の手のシェアは同等（{band}）",
                  r is not None and r["verdict"] in ("同等", "差あり（小）"), f"{r['verdict']} d = {f(r['d']):+.3f}" if r else "–")
    # 真値とノイズありで人の値が同じ順序か
    for c in ("gshare_upperarm", "N_eff3", "mshare_prox", "K_orient"):
        x = [f(PA[k].get(c)) for k in PA]; y = [f(PB[k].get(c)) for k in PA]
        from scipy.stats import spearmanr
        rho = spearmanr(x, y)[0]
        check(f"人の順序の再現: {c}", rho >= 0.9, f"ρ = {rho:.2f}")

    n_ok = sum(1 for _, ok, _ in res if ok)
    print(f"\n検証: {n_ok} / {len(res)} 項目 OK（作業フォルダ {work}）\n")
    for name, ok, det in res:
        print(f"  {'OK' if ok else 'NG'}  {name:60s} {det}")
    with open(os.path.join(work, "verify.md"), "w", encoding="utf-8") as fo:
        fo.write(f"# coord_eval の検証（合成データ）\n\n{n_ok} / {len(res)} 項目 OK\n\n| 結果 | 項目 | 値 |\n|---|---|---|\n")
        for name, ok, det in res:
            fo.write(f"| {'OK' if ok else '**NG**'} | {name} | {det} |\n")
    return 0 if n_ok == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
