"""c3 関節ごとの周波数応答（= 周波数の分業）

何をするか
  入力周波数ごとに、各関節が「手の動き（マウスの積算変位 = 真値）」に対してどれだけ動いたかを見る。
  - ゲイン |関節 / マウス| と位相（入力周波数ごと、軸別）
  - **相対ゲイン**: 最低周波数のゲインを 1 としたときの落ち方 → 関節ごとの「カットオフ周波数」
    （ゲインが最低周波数の 1/√2 になる周波数。低周波は肩・肘、高周波は手首、という分業の指標）
  - 位相の傾きからの遅れ: 近位（肩）が先か遠位（手首）が先か
  - 各関節の動きのうち入力周波数で説明できる割合（1 − remnant）
わかること
  **「どの速さの動きをどの関節が担っているか」**。slow / fast の 2 帯域と DPI 3 水準で直接比べられるので、
  「人によって関節の使い方が違う」を周波数軸で示せる。
注意
  c0 の SNR < 2 の組は ok = False とし、図では薄く描く（高 DPI の高周波は手の動きが数 mm しかない）。
"""
import numpy as np
import common as C

REF = "mouse"   # 基準の入力（マウスの積算変位 = 机の上の手の変位の真値）


def run(tr):
    if tr.kind != "track" or not tr.traj or not tr.has_rs:
        return "## c3 関節ごとの周波数応答\n\n静止の試行 / 関節データなしなので省略。\n"
    ref = tr.task_grid(REF)
    rows, summary = [], []
    for n in tr.rs_names:
        J = tr.joint_grid(n)
        for k, ax in enumerate(C.AX):
            nb, f, _ = tr.freqs(ax)
            d = C.describing(ref[:, k], J[:, k], tr.T, nb)
            ok = d["snr"] > C.SNR_MIN
            g = d["gain"].copy()
            g0 = g[ok][0] if ok.any() else np.nan
            rel = g / g0 if np.isfinite(g0) else g * np.nan
            # カットオフ: 相対ゲインが 1/√2 を下回る周波数（対数で補間）
            fc = np.nan
            for i in range(len(f) - 1):
                if rel[i] >= 1 / np.sqrt(2) > rel[i + 1]:
                    w = (rel[i] - 1 / np.sqrt(2)) / (rel[i] - rel[i + 1])
                    fc = float(np.exp(np.log(f[i]) + w * (np.log(f[i + 1]) - np.log(f[i])))); break
            tau, pres = C.delay_from_phase(d["f_hz"], d["phase_deg"], snr=d["snr"])
            fb, extrap, _ = C.bandwidth_fit(d["f_hz"], g, snr=d["snr"])
            summary.append({"joint": n, "label": C.LABEL[n], "axis": ax, "gain_lo": float(g0),
                            "gain_hi": float(g[-1]), "rel_hi": float(rel[-1]), "cutoff_hz": fc,
                            "fb_fit_hz": fb, "fb_extrapolated": extrap,
                            "delay_ms": tau, "phase_resid_deg": pres, "remnant": d["remnant"],
                            "n_ok": int(ok.sum()), "n_bins": len(nb)})
            for i in range(len(nb)):
                rows.append({"joint": n, "label": C.LABEL[n], "axis": ax, "n": int(nb[i]),
                             "f_hz": float(f[i]), "amp_mm": float(d["amp_out"][i]),
                             "gain": float(g[i]), "rel_gain": float(rel[i]),
                             "phase_deg": float(d["phase_deg"][i]), "snr": float(d["snr"][i]),
                             "ok": bool(ok[i])})
    tr.save_json("a4_joint_freq.json", {"summary": summary, "rows": rows, "ref": REF})

    plt = C.setup_plot()
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex="col")
    for k, ax in enumerate(C.AX):
        for n in tr.rs_names:
            s = [r for r in rows if r["axis"] == ax and r["joint"] == n]
            f = np.array([r["f_hz"] for r in s]); good = np.array([r["ok"] for r in s])
            for key, row in ((("amp_mm"), 0), ("rel_gain", 1), ("phase_deg", 2)):
                y = np.array([r[key] for r in s])
                axes[row, k].plot(f, y, "-", color=C.COLOR[n], lw=1.0, alpha=.35)
                axes[row, k].plot(f[good], y[good], "o-", color=C.COLOR[n], lw=1.6, ms=4,
                                  label=C.LABEL[n] if row == 0 else None)
        axes[0, k].set_yscale("log"); axes[0, k].set_xscale("log")
        axes[0, k].set_title(f"{ax} 軸  関節の振幅 [mm]")
        axes[1, k].axhline(1 / np.sqrt(2), color="#888", ls="--", lw=1)
        axes[1, k].set_yscale("log"); axes[1, k].set_title("相対ゲイン（最低周波数 = 1）")
        axes[2, k].set_title("位相 [deg]（マウス基準）"); axes[2, k].set_xlabel("周波数 [Hz]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{tr.name}（{tr.tag()}，DPI {tr.dpi:g}）関節の周波数応答", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a4_joint_freq.png", dpi=120); plt.close(fig)

    md = "## c3 関節ごとの周波数応答（周波数の分業）\n\n"
    md += f"基準 = マウスの積算変位。薄い線は SNR < {C.SNR_MIN:.0f}（測れていない成分）。\n\n"
    md += C.md_table(summary, [("関節", "label"), ("軸", "axis"), ("最低周波数のゲイン", "gain_lo", "{:.3f}"),
                               ("最高周波数のゲイン", "gain_hi", "{:.3f}"),
                               ("相対ゲイン（最高周波数）", "rel_hi", "{:.2f}"),
                               ("カットオフ [Hz]", "cutoff_hz", "{:.2f}"),
                               ("当てはめ fb [Hz]", "fb_fit_hz", "{:.2f}"),
                               ("外挿", "fb_extrapolated"),
                               ("遅れ [ms]", "delay_ms", "{:+.0f}"),
                               ("remnant [%]", "remnant", "{:.0%}"),
                               ("使えた成分", "n_ok", "{:.0f}")])
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
