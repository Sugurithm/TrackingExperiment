"""a3 偏りの一定性（表面と関節中心のずれは一定か）

何をするか
  解析範囲を 1 秒の窓に区切り，窓ごとの偏り（RS − MoCap の平均）を求める。
  - 偏りの時間変動 σ_between: 窓ごとの偏りの SD。小さいほど「一定」
  - 誤差のうち一定の偏りが占める割合: |全体の偏り|² / 平均二乗誤差
  - 偏りを引いた後の RMSE を 3 通りで比較
      (1) 動作全体の偏りを引く（最適）
      (2) 静止区間の偏りを動作に当てる（転用。実際の使い方: 最初に静止して偏りを測る）
      (3) 各ブロック自身の偏りを引く
    (2) ≈ (1) なら「静止で 1 回測った補正値がそのまま使える」
  - 偏りが何に依存するか: 誤差（各成分）を「腕の位置（主な動きの方向，100 mm あたり）」と
    「速さ（1 m/s あたり）」に回帰し，傾きと説明率 R² を出す
わかること
  RMSE が大きくても偏りが一定なら，動き方（変位・速度・関節の協調）の解析には使える。
  逆に偏りが姿勢や速さで変わるなら，その分は動きの誤差として残る。
"""
import numpy as np
import common as C


def _err(tr, n, mk):
    X = tr.rs(n)[mk]; R = tr.ref(n)[mk]
    ok = ~(np.isnan(X).any(1) | np.isnan(R).any(1))
    return C.decompose(X - R, X), ok, R


def _mask_union(tr, segs):
    mk = np.zeros(len(tr.t), bool)
    for s in segs: mk |= tr.mask(s)
    return mk


def run(tr, win=1.0):
    rest_mk = _mask_union(tr, tr.rest)
    blocks = [(b["t0"], b["t1"]) for b in tr.blocks]
    move_mk = _mask_union(tr, blocks)
    # 腕の主な動きの方向（MoCap 手首，カメラ座標）
    W = tr.M["wrist"][move_mk]; W = W[~np.isnan(W).any(1)]
    axis = np.linalg.svd(W - W.mean(0), full_matrices=False)[2][0]; center = W.mean(0)
    wspeed = C.speed(np.nan_to_num(tr.M["wrist"]), tr.fs)

    rows = []; series = {}
    for n in tr.compared():
        D, ok, R = _err(tr, n, slice(None))
        span = tr.in_span & ok
        # 窓ごとの偏り
        edges = np.arange(tr.span[0], tr.span[1], win); wb = []; wt = []; wsd = []
        for a in edges:
            m = span & (tr.t >= a) & (tr.t < a + win)
            if m.sum() < 0.8 * win * tr.fs: continue
            wb.append(D[m].mean(0)); wsd.append(D[m].std(0)); wt.append(a + win / 2)
        wb = np.array(wb); wsd = np.array(wsd)
        series[n] = {"t": wt, "bias": wb.tolist()}
        mm = move_mk & ok; rm = rest_mk & ok
        b_move = D[mm].mean(0); b_rest = D[rm].mean(0)
        mse = np.mean(np.sum(D[mm] ** 2, 1))
        rmse0 = np.sqrt(mse)
        rmse_glob = np.sqrt(np.mean(np.sum((D[mm] - b_move) ** 2, 1)))
        rmse_rest = np.sqrt(np.mean(np.sum((D[mm] - b_rest) ** 2, 1)))
        own = []
        for s in blocks:
            m = tr.mask(s) & ok; own.append(D[m] - D[m].mean(0))
        rmse_own = np.sqrt(np.mean(np.sum(np.concatenate(own) ** 2, 1)))
        # 前半ブロックで求めた偏りを後半に
        h = len(blocks) // 2
        m1 = _mask_union(tr, blocks[:h]) & ok; m2 = _mask_union(tr, blocks[h:]) & ok
        rmse_half = np.sqrt(np.mean(np.sum((D[m2] - D[m1].mean(0)) ** 2, 1)))
        # 位置・速さへの依存（動作中）。説明変数は全関節共通で「手首（MoCap）の位置と速さ」
        Rm = R[mm]; pos = (tr.M["wrist"][mm] - center) @ axis / 100.0
        spd = wspeed[mm] / 1000.0
        A = np.column_stack([np.ones(mm.sum()), pos, spd])
        reg = {}
        for k, a in enumerate(C.AXES):
            coef, *_ = np.linalg.lstsq(A, D[mm][:, k], rcond=None)
            pred = A @ coef; r2 = 1 - np.var(D[mm][:, k] - pred) / np.var(D[mm][:, k])
            reg[a] = {"per_100mm": coef[1], "per_mps": coef[2], "r2": r2}
        resid = D[mm] - A @ np.linalg.lstsq(A, D[mm], rcond=None)[0]
        rmse_reg = np.sqrt(np.mean(np.sum(resid ** 2, 1)))
        pos_range = (np.percentile(pos, 97) - np.percentile(pos, 3)) * 100
        vmax = np.percentile(spd, 99)
        rows.append({"joint": n, "label": C.LABEL[n],
                     "bias_ray": b_move[0], "bias_h": b_move[1], "bias_v": b_move[2], "bias3d": np.linalg.norm(b_move),
                     "between_sd3d": float(np.sqrt(np.sum(wb.std(0) ** 2))),
                     "between_sd_ray": wb.std(0)[0], "between_sd_h": wb.std(0)[1], "between_sd_v": wb.std(0)[2],
                     "within_sd3d": float(np.sqrt(np.mean(np.sum(wsd ** 2, 1)))),
                     "rest_vs_move_bias_diff": float(np.linalg.norm(b_rest - b_move)),
                     "const_fraction_pct": 100 * np.sum(b_move ** 2) / mse,
                     "rmse_raw": rmse0, "rmse_minus_global": rmse_glob, "rmse_minus_rest": rmse_rest,
                     "rmse_minus_block": rmse_own, "rmse_minus_firsthalf": rmse_half, "rmse_minus_pos_speed": rmse_reg,
                     "pos_range_mm": pos_range, "regression": reg,
                     "vmax_mps": vmax,
                     "pos_effect_mm": float(np.sqrt(sum((reg[a]["per_100mm"] * pos_range / 100) ** 2 for a in C.AXES))),
                     "speed_effect_mm": float(np.sqrt(sum((reg[a]["per_mps"] * vmax) ** 2 for a in C.AXES)))})
    tr.save_json("a3_bias_stability.json", {"summary": rows, "window_bias": series, "motion_axis_cam": axis})

    plt = C.setup_plot()
    J = tr.compared()
    fig, ax = plt.subplots(len(J), 1, figsize=(15, 2.4 * len(J)), sharex=True)
    for i, n in enumerate(J):
        s = series[n]; b = np.array(s["bias"])
        for k, a in enumerate(C.AXES):
            ax[i].plot(s["t"], b[:, k], "-", lw=1.5, color=["#2a78d6", "#1baf7a", "#eda100"][k], label=C.AXLABEL[a])
        C.shade_segments(ax[i], tr); ax[i].set_ylabel(f"{C.LABEL[n]}\n[mm]")
    ax[0].legend(fontsize=8, ncol=3, loc="lower right"); ax[-1].set_xlabel("Take 時刻 [s]")
    fig.suptitle("1 秒ごとの偏り（RS − MoCap）。灰 = 静止，青 = 動作ブロック", y=1.0)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a3_bias_timeseries.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    xs = np.arange(len(rows)); w = .16
    keys = [("rmse_raw", "そのまま"), ("rmse_minus_rest", "静止の偏りを転用"), ("rmse_minus_firsthalf", "前半の偏りを後半に"),
            ("rmse_minus_global", "全体の偏り（最適）"), ("rmse_minus_pos_speed", "位置・速さ依存も除去")]
    cols = ["#999", "#2a78d6", "#1baf7a", "#eda100", "#e87ba4"]
    for i, (k, lab) in enumerate(keys):
        ax[0].bar(xs + (i - 2) * w, [r[k] for r in rows], w, label=lab, color=cols[i])
    ax[0].set_xticks(xs); ax[0].set_xticklabels([r["label"] for r in rows]); ax[0].set_ylabel("RMSE [mm]（動作中）")
    ax[0].set_title("偏りの補正方法ごとの RMSE"); ax[0].legend(fontsize=8)
    n = "wrist" if "wrist" in J else J[0]
    D, ok, R = _err(tr, n, slice(None)); mm = move_mk & ok
    pos = (R[mm] - center) @ axis
    for k, a in enumerate(C.AXES):
        ax[1].plot(pos[::5], D[mm][::5, k], ".", ms=2, alpha=.4, color=["#2a78d6", "#1baf7a", "#eda100"][k], label=C.AXLABEL[a])
    ax[1].set_xlabel("腕の位置（主な動きの方向）[mm]"); ax[1].set_ylabel("RS − MoCap [mm]")
    ax[1].set_title(f"{C.LABEL[n]}: 位置による偏りの変化"); ax[1].legend(fontsize=8, markerscale=4)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a3_bias_correction.png", dpi=120); plt.close(fig)

    md = "## a3 偏りの一定性\n\n"
    md += C.md_table(rows, [("関節", "label"), ("|偏り| [mm]", "bias3d"), ("偏りの時間変動 SD", "between_sd3d"),
                            ("静止と動作の偏りの差", "rest_vs_move_bias_diff"), ("一定の偏りの割合 [%]", "const_fraction_pct")])
    md += "\n偏りの補正方法ごとの RMSE（動作中，mm）\n\n"
    md += C.md_table(rows, [("関節", "label"), ("そのまま", "rmse_raw"), ("静止の偏りを転用", "rmse_minus_rest"),
                            ("前半→後半", "rmse_minus_firsthalf"), ("全体の偏り", "rmse_minus_global"),
                            ("各ブロックの偏り", "rmse_minus_block"), ("+位置・速さ依存", "rmse_minus_pos_speed")])
    rr = rows[0]
    md += (f"\n偏りの位置・速さ依存（説明変数 = 手首の左右位置と速さ。手首の動きの範囲 {rr['pos_range_mm']:.0f} mm，"
           f"最大速 {rr['vmax_mps']:.2f} m/s での偏りの変化量，3 成分の合成）\n\n")
    reg_rows = []
    for r in rows:
        g = r["regression"]; ap = max(C.AXES, key=lambda a: abs(g[a]["per_100mm"]))
        reg_rows.append({"label": r["label"], "eff": r["pos_effect_mm"], "main": f"{C.AXLABEL[ap]}（R²={g[ap]['r2']:.2f}）",
                         "spd": r["speed_effect_mm"]})
    md += C.md_table(reg_rows, [("関節", "label"), ("位置による変化 [mm]", "eff"), ("主な成分", "main"),
                                ("速さによる変化 [mm]", "spd")])
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
