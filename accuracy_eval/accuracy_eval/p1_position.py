"""p1 フェーズ別の位置の誤差

何をするか
  静止・各テンポ・休みのフェーズごとに，関節ごとに RS − MoCap を集計する（カメラ座標，奥行き/横/縦）。
  - 偏り（成分別）と |偏り|，SD，RMSE，95 パーセンタイル
  - 偏りを除いた RMSE を 2 通り:
      全体補正 = 動作フェーズ全体で 1 つの偏りを引く（実際の補正に近い）
      フェーズ補正 = そのフェーズ自身の偏りを引く（= ばらつきだけ）
  - 一致限界（偏り ± 1.96 SD）
わかること
  静止 → 遅い → 速い と条件を変えたとき，誤差のどの部分（偏り・ばらつき）が変わるか。
  全体補正とフェーズ補正の差が大きければ，偏りがフェーズ（姿勢・速さ）で変わっている。
"""
import numpy as np
import common as C
import phase_common as P


def run(tr, phs=None):
    phs = phs or P.phases(tr)
    J = tr.compared()
    use = [p for p in phs if p["kind"] in ("static", "move")]
    move = [p for p in use if p["kind"] == "move"]
    gbias = {}
    for n in J:
        Es = [];
        for p in move:
            X, R, _, _ = P.pair(tr, n, p); Es.append(X - R)
        gbias[n] = np.concatenate(Es).mean(0) if Es else np.zeros(3)
    rows = []
    for p in use:
        for n in J:
            X, R, _, _ = P.pair(tr, n, p)
            if len(X) < 30: continue
            E = X - R; D = C.decompose(E, X); b = D.mean(0); s = D.std(0)
            Eg = E - gbias[n]; e3g = np.linalg.norm(Eg, axis=1)
            Xr, Rr, _, _ = P.pair(tr, n, p, filtered=False)
            rows.append({"phase": p["name"], "joint": n, "n": len(X),
                         "bias_ray": b[0], "bias_h": b[1], "bias_v": b[2], "bias3d": float(np.linalg.norm(b)),
                         "sd3d": float(np.sqrt(np.sum(s ** 2))), "sd_ray": s[0], "sd_h": s[1], "sd_v": s[2],
                         "rmse": float(np.sqrt(np.mean(np.sum(E ** 2, 1)))),
                         "rmse_raw": float(np.sqrt(np.mean(np.sum((Xr - Rr) ** 2, 1)))),
                         "rmse_global_corr": float(np.sqrt(np.mean(e3g ** 2))), "p95_global_corr": float(np.percentile(e3g, 95)),
                         "bias_shift": float(np.linalg.norm(E.mean(0) - gbias[n])),
                         "loa_ray": (b[0] - 1.96 * s[0], b[0] + 1.96 * s[0]), "loa_h": (b[1] - 1.96 * s[1], b[1] + 1.96 * s[1]),
                         "loa_v": (b[2] - 1.96 * s[2], b[2] + 1.96 * s[2])})
    P.save(tr, "p1_position.json", {"phases": P.strip_masks(use), "rows": rows, "global_bias_cam": gbias})

    plt = C.setup_plot()
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
    P.plot_by_phase(ax[0, 0], rows, "rmse", J, use, "RMSE（偏りを含む）", "[mm]", ylim0=True)
    P.plot_by_phase(ax[0, 1], rows, "rmse_global_corr", J, use, "全体の偏りで補正後の RMSE", "[mm]", ylim0=True)
    P.plot_by_phase(ax[0, 2], rows, "sd3d", J, use, "フェーズ自身の偏りで補正後（ばらつき）", "[mm]", ylim0=True)
    for a, k in zip(ax[1], C.AXES):
        P.plot_by_phase(a, rows, f"bias_{k}", J, use, f"偏り: {C.AXLABEL[k]}", "RS − MoCap [mm]", hline=0)
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{P.outdir(tr)}/p1_position.png", dpi=120); plt.close(fig)

    md = "## p1 フェーズ別の位置の誤差\n\n"
    for k, t in [("rmse", "RMSE（偏りを含む）[mm]"), ("rmse_global_corr", "全体の偏りで補正後の RMSE [mm]"),
                 ("sd3d", "フェーズ自身の偏りで補正後（ばらつき）[mm]"), ("bias_shift", "フェーズの偏りと全体の偏りの差 [mm]"),
                 ("bias_ray", "偏り: 奥行き [mm]（+ = RS が奥）"), ("bias_h", "偏り: 横 [mm]"), ("bias_v", "偏り: 縦 [mm]")]:
        md += f"\n{t}\n\n" + P.pivot(rows, k, J, use)
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
