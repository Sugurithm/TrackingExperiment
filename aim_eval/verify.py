"""合成データで指標を検証する（実データに使う前に 1 回通す）

make_synthetic.py で「答えの分かっているセッション」を 4 通りの型で作り、
指標がその型を復元できるかを確かめる。

  型（--style = 空間的な型，--timing = 時間的な型）
    手首依存        wrist / uni    … 手だけが動く            → N_eff ≈ 1
    全体・同時      mixed / uni    … 全体が同じ分担で動く    → N_eff ≈ 4，S ≈ 1
    全体・交代      mixed / seq    … 一つずつ持ち回る        → S < 1
    全体・相殺      mixed / comp   … 同時かつ打ち消し合う    → C > 1
  あわせて、計測系の値（人の遅れ・MediaPipe の遅れ・DPI による縮小）も復元できるか見る。

  python verify.py              # 400 と 1600 DPI で検証（数分）
  python verify.py --keep /tmp/v
"""
import argparse, glob, json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C                                      # noqa: E402

CASES = [("W", "wrist", "uni", "手首依存"), ("M", "mixed", "uni", "全体・同時"),
         ("Q", "mixed", "seq", "全体・交代"), ("P", "mixed", "comp", "全体・相殺")]


class Args:
    def __init__(self, t):
        self.trial = t; self.rs = self.tmc = self.axes = None
        self.out = os.path.join(t, "eval")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep")
    ap.add_argument("--dpi", nargs="+", default=["400", "1600"])
    a = ap.parse_args()
    root = a.keep or tempfile.mkdtemp(prefix="aim_verify_")
    for name, style, timing, _ in CASES:
        subprocess.run([sys.executable, os.path.join(HERE, "make_synthetic.py"), root,
                        "--name", name, "--style", style, "--timing", timing,
                        "--dpi", *a.dpi, "--seeds", "1"],
                       check=True, cwd=HERE, stdout=subprocess.DEVNULL)

    got = {}
    for dpi in a.dpi:
        for name, _, _, lab in CASES:
            d = glob.glob(os.path.join(root, f"aim_{name}_dpi{dpi}_*", "trial_01_slow_s1"))[0]
            tr = C.Trial(Args(d))
            r = C.joint_indices(tr)
            got[(dpi, name)] = (r, tr)

    checks = []

    def chk(name, ok, got_s, want_s):
        checks.append((name, got_s, want_s, bool(ok)))

    for dpi in a.dpi:
        g = {n: got[(dpi, n)][0] for n, _, _, _ in CASES}
        chk(f"[{dpi}] 手首依存で N_eff が小さい", g["W"]["N_eff"] < 2.0,
            f"{g['W']['N_eff']:.2f}", "< 2.0")
        chk(f"[{dpi}] 全体で N_eff が大きい", g["M"]["N_eff"] > 3.0,
            f"{g['M']['N_eff']:.2f}", "> 3.0")
        chk(f"[{dpi}] 手首依存の主役が「手の回転」", g["W"]["dominant"] == "hand",
            C.PARTLABEL.get(g["W"]["dominant"], "?"), "手の回転（手首）")
        chk(f"[{dpi}] 交代で S が 1 を下回る", g["Q"]["S"] < 0.95, f"{g['Q']['S']:.2f}", "< 0.95")
        chk(f"[{dpi}] 同時で S が 1 付近", g["M"]["S"] > 0.95, f"{g['M']['S']:.2f}", "> 0.95")
        chk(f"[{dpi}] 相殺で C が 1 を大きく超える", g["P"]["C"] > 2.0, f"{g['P']['C']:.2f}", "> 2.0")
        chk(f"[{dpi}] 同時で C が 1 付近", g["M"]["C"] < 1.3, f"{g['M']['C']:.2f}", "< 1.3")

    # DPI によらず指標が同じか（400 と 1600 の差）
    if len(a.dpi) >= 2:
        d0, d1 = a.dpi[0], a.dpi[-1]
        for key, tol, lab in (("N_eff", 0.5, "N_eff"), ("S", 0.08, "S"), ("C", 0.5, "C")):
            diff = max(abs(got[(d0, n)][0][key] - got[(d1, n)][0][key]) for n, _, _, _ in CASES)
            chk(f"{lab} が DPI でほぼ変わらない", diff <= tol, f"最大差 {diff:.2f}", f"≤ {tol}")

    # 計測系
    tr = got[(a.dpi[0], "M")][1]
    TRUE = json.load(open(os.path.join(tr.sess_dir, "session.json"), encoding="utf-8"))["true_values"]
    import a0_quality
    mm = a0_quality.mouse_match(tr, "mcp3" if "mcp3" in tr.rs_names else "wrist")
    chk("MediaPipe の遅れ（マウス → 手先）", abs(mm["lag_ms"] - TRUE["tau_rs_ms"]) < 15,
        f"{mm['lag_ms']:+.0f} ms", f"{TRUE['tau_rs_ms']:.0f} ± 15 ms")
    chk("手先 / マウス のゲイン", abs(mm["gain_x"] - 1) < 0.15, f"{mm['gain_x']:.2f}", "1.00 ± 0.15")
    import a5_describing
    a5_describing.run(tr)
    d = json.load(open(os.path.join(tr.out, "a5_describing.json"), encoding="utf-8"))["summary"]["x"]
    want = TRUE["tau_human_ms"] + 1000 * TRUE.get("human_lag_tau_s", 0.0)
    chk("人の遅れ（ターゲット → カーソル）", abs(d["delay_ms"] - want) < max(30, .1 * want),
        f"{d['delay_ms']:+.0f} ms", f"{want:.0f} ± {max(30, .1 * want):.0f} ms")

    w = max(len(c[0]) for c in checks)
    print("\n=== 検証結果 ===")
    for name, g, want, ok in checks:
        print(f"{'OK ' if ok else 'NG '} {name:<{w}}  得た値 {g:>16}   期待 {want}")
    bad = [c for c in checks if not c[3]]
    print(f"\n{len(checks) - len(bad)} / {len(checks)} 件 OK")
    print(f"データ: {root}")

    print("\n--- 指標の一覧（参考）---")
    print(f"{'DPI':>5} {'型':12s} {'N_eff':>6} {'S':>6} {'C':>6} {'R':>7} {'R信頼':>6}")
    for dpi in a.dpi:
        for name, _, _, lab in CASES:
            r = got[(dpi, name)][0]
            R = f"{r['R']:+.2f}" if np.isfinite(r.get("R", np.nan)) else "  –"
            print(f"{dpi:>5} {lab:12s} {r['N_eff']:6.2f} {r['S']:6.2f} {r['C']:6.2f} {R:>7} "
                  f"{str(r.get('R_reliable')):>6}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
