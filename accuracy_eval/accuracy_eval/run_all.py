"""a0〜a6 をまとめて実行し，eval/summary.md（要点の表 + 各解析の表）を書く。

使い方:  python run_all.py recordings/trial_20261019_015509
"""
import json, os, time
import numpy as np
import common as C
import a0_reference, a1_jitter, a2_accuracy, a3_bias_stability, a4_motion, a5_angles_segments, a6_availability, a7_conditions
import a8_phases

MODULES = [a0_reference, a1_jitter, a2_accuracy, a3_bias_stability, a4_motion, a5_angles_segments, a6_availability,
           a7_conditions, a8_phases]


def headline(tr):
    L = lambda f: json.load(open(os.path.join(tr.out, f), encoding="utf-8"))
    a1 = {r["joint"]: r for r in L("a1_jitter.json")["summary"]}
    a2 = {r["joint"]: r for r in L("a2_accuracy.json") if r["segment"] == "動作全体"}
    a3 = {r["joint"]: r for r in L("a3_bias_stability.json")["summary"]}
    a4 = L("a4_motion.json"); a6 = L("a6_availability.json")["rows"]
    fast = tr.blocks[-1]["name"] if tr.blocks else None
    rows = []
    for n in tr.compared():
        f = next((r for r in a4 if r["joint"] == n and r["block"] == fast), {})
        lags = [r["lag_ms"] for r in a4 if r["joint"] == n and r["lag_ms"] is not None and np.isfinite(r["lag_ms"])]
        u = next((r for r in a6 if r["joint"] == n and r["segment"] == "解析範囲"), {})
        rows.append({"label": C.LABEL[n], "jit": a1[n]["sd3d_filt"], "rmse": a2[n]["rmse"], "bias": a2[n]["bias3d"],
                     "btv": a3[n]["between_sd3d"], "rest": a3[n]["rmse_minus_rest"], "glob": a3[n]["rmse_minus_global"],
                     "nr": f.get("nrmse_pct", np.nan), "lag": f"{min(lags):+.0f}〜{max(lags):+.0f}" if lags else "–",
                     "sr": f.get("speed_r", np.nan), "use": u.get("usable_pct", np.nan)})
    md = "## 要点（中間審査用）\n\n"
    md += C.md_table(rows, [("関節", "label"), ("静止ジッタ σ [mm]", "jit"), ("絶対 RMSE [mm]", "rmse"),
                            ("一定の偏り [mm]", "bias"), ("偏りの時間変動 SD [mm]", "btv"),
                            ("静止で補正後 RMSE", "rest"), ("最適補正後 RMSE", "glob"),
                            (f"形の誤差 / 範囲 [%]（{fast}）", "nr"), ("遅れ [ms]", "lag"),
                            (f"速さの r（{fast}）", "sr", "{:.3f}"), ("使える率 [%]", "use")])
    return md


def main():
    args = C.parser(__doc__).parse_args()
    tr = C.Trial(args)
    parts = []
    for m in MODULES:
        t0 = time.time()
        parts.append(m.run(tr))
        print(f"{m.__name__:22s} {time.time() - t0:5.1f} s")
    head = (f"# 精度評価 {os.path.basename(os.path.abspath(args.trial))}\n\n"
            f"- 解析範囲（Take 時刻）: {tr.span[0]:.1f}–{tr.span[1]:.1f} s（同期ボード区間 ±1 s を除く）\n"
            f"- 静止区間: " + "，".join(f"{a:.1f}–{b:.1f}" for a, b in tr.rest) + "\n"
            f"- 動作ブロック: " + "，".join(f"{b['name']}（{b['t0']:.0f}–{b['t1']:.0f} s，最大 {b['vmax_mps']:.2f} m/s）" for b in tr.blocks) + "\n"
            f"- 前処理: RS = Hampel（9 フレーム，3·MAD）+ 5 Hz ゼロ位相低域通過，MoCap = 5 Hz。大外れ（>{C.GROSS_MM:.0f} mm）は除外して a6 で数える\n\n")
    md = head + headline(tr) + "\n\n" + "\n\n".join(parts)
    path = os.path.join(tr.out, "summary.md")
    open(path, "w", encoding="utf-8").write(md)
    tr.save_json("segments.json", tr.header())
    print("→", path)


if __name__ == "__main__":
    main()
