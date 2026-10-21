"""c2 記述関数（ターゲット → カーソル）と remnant

何をするか
  ターゲットは 1/T（T = 40 s）の素数倍の正弦波の和で、x と y で周波数が重ならない。
  解析区間をちょうど 1 周期として窓なし FFT をかけ、入力周波数のビンだけを読む。
  - ゲイン |カーソル / ターゲット| と位相（入力周波数ごと）= ボード線図
  - 位相の傾きから**実効的な遅れ** τ [ms]、ゲインが 1 を下回る**交差周波数**（帯域幅）
  - **remnant**: 入力周波数以外に漏れたパワーの割合 = 線形には説明できない成分
  - 軸間の漏れ（x の入力周波数に y がどれだけ応答したか）
わかること
  命中率や RMS より細かい「追従の中身」。
  ゲインが落ちる周波数 = 付いていけなくなる速さ、位相の傾き = 人の反応の遅れ、
  remnant = ばらつき・修正動作・ノイズ。DPI や帯域でどれが変わるかを見る。
"""
import numpy as np
import common as C


def run(tr):
    if tr.kind != "track" or not tr.traj:
        return "## c2 記述関数（ターゲット → カーソル）\n\n静止の試行なので省略。\n"
    tg, cu = tr.task_grid("target"), tr.task_grid("cursor")
    rows, summary = [], {}
    for k, ax in enumerate(C.AX):
        nb, f, _ = tr.freqs(ax)
        d = C.describing(tg[:, k], cu[:, k], tr.T, nb)
        tau, resid = C.delay_from_phase(d["f_hz"], d["phase_deg"], snr=d["snr"])
        fc = C.crossover(d["f_hz"], d["gain"], snr=d["snr"])
        bw = C.bandwidth(d["f_hz"], d["gain"], snr=d["snr"])
        fb, extrap, fres = C.bandwidth_fit(d["f_hz"], d["gain"], snr=d["snr"])
        # 軸間の漏れ: この軸の入力周波数に、もう一方の軸の出力がどれだけ出ているか
        other = 1 - k
        _, Fo = C.spectrum(cu[:, other], tr.T)
        leak = np.abs(C.at_bins(np.arange(len(Fo)), Fo, nb)) / np.maximum(d["amp_out"], 1e-9)
        summary[ax] = {"delay_ms": tau, "phase_resid_deg": resid, "crossover_hz": fc, "bandwidth_hz": bw, "fb_fit_hz": fb,
                       "fb_extrapolated": extrap, "fb_fit_resid": fres,
                       "remnant": d["remnant"], "gain_lo": float(d["gain"][0]),
                       "gain_hi": float(d["gain"][-1]), "leak_mean": float(np.nanmean(leak))}
        for i in range(len(nb)):
            rows.append({"axis": ax, "n": int(nb[i]), "f_hz": float(f[i]),
                         "amp_in_mm": float(d["amp_in"][i]), "amp_out_mm": float(d["amp_out"][i]),
                         "gain": float(d["gain"][i]), "phase_deg": float(d["phase_deg"][i]),
                         "snr": float(d["snr"][i]), "leak": float(leak[i]),
                         "ok": bool(d["snr"][i] > C.SNR_MIN)})
    tr.save_json("a5_describing.json", {"summary": summary, "rows": rows})

    plt = C.setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
    for k, ax in enumerate(C.AX):
        s = [r for r in rows if r["axis"] == ax]
        f = [r["f_hz"] for r in s]
        axes[0, k].plot(f, [r["gain"] for r in s], "o-", color="#2a78d6", lw=1.4)
        axes[0, k].axhline(1, color="#888", ls="--", lw=1)
        axes[0, k].set_yscale("log"); axes[0, k].set_xscale("log")
        axes[0, k].set_title(f"{ax} 軸  ゲイン（fb {summary[ax]['fb_fit_hz']:.2f} Hz，"
                            f"remnant {100 * summary[ax]['remnant']:.0f} %）")
        axes[1, k].plot(f, [r["phase_deg"] for r in s], "o-", color="#eb6834", lw=1.4)
        axes[1, k].set_xscale("log")
        axes[1, k].set_xlabel("周波数 [Hz]")
        axes[1, k].set_title(f"位相（遅れ {summary[ax]['delay_ms']:+.0f} ms）")
    axes[0, 0].set_ylabel("|カーソル / ターゲット|"); axes[1, 0].set_ylabel("位相 [deg]")
    fig.suptitle(f"{tr.name}（{tr.tag()}，DPI {tr.dpi:g}）記述関数", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a5_describing.png", dpi=120); plt.close(fig)

    md = "## c2 記述関数（ターゲット → カーソル）\n\n"
    md += C.md_table([{"axis": a, **summary[a]} for a in C.AX],
                     [("軸", "axis"), ("実効遅れ [ms]", "delay_ms", "{:+.0f}"),
                      ("位相の残差 [deg]", "phase_resid_deg", "{:.1f}"),
                      ("帯域幅 −3dB [Hz]", "bandwidth_hz", "{:.2f}"),
                      ("1 次遅れの当てはめ fb [Hz]", "fb_fit_hz", "{:.2f}"),
                      ("外挿", "fb_extrapolated"),
                      ("交差周波数 [Hz]", "crossover_hz", "{:.2f}"),
                      ("remnant [%]", "remnant", "{:.1%}"),
                      ("最低周波数のゲイン", "gain_lo", "{:.2f}"),
                      ("最高周波数のゲイン", "gain_hi", "{:.2f}"),
                      ("軸間の漏れ", "leak_mean", "{:.2f}")])
    md += "\n"
    md += C.md_table(rows, [("軸", "axis"), ("n", "n", "{:.0f}"), ("f [Hz]", "f_hz", "{:.3f}"),
                            ("入力振幅 [mm]", "amp_in_mm"), ("出力振幅 [mm]", "amp_out_mm"),
                            ("ゲイン", "gain", "{:.3f}"), ("位相 [deg]", "phase_deg", "{:+.1f}"),
                            ("SNR", "snr", "{:.1f}")])
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
