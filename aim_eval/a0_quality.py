"""c0 データの健全性と、この試行で測れる限界

何をするか
  1. 関節ごとの有効率・status の内訳・左右の取り違え（lr_swapped）・最長の欠損
  2. RealSense のフレーム時刻（周期・欠落）と課題のフレームレート
  3. **マウスを真値にした照合**: マウスの積算変位（counts × cm_per_count）と手首の机座標 x, y を比べ、
     ゲイン（手 / マウス）・残差・遅れ（相互相関）を出す。MoCap がなくてもこれで
     「追従の正確さ」と「MediaPipe + 露光の遅れ」が試行ごとに測れる
  4. 静止（trial_00_static_pre）または動きの少ない区間から求めたジッタ σ を使い、
     **周波数成分ごとの SNR 表**（成分の振幅 / ノイズ）を作る
わかること
  その試行がそもそも解析に使えるか、どの周波数成分まで信用できるか。
  SNR < 2 の成分は以降の解析で「測れていない」として扱う（c2〜c4 の表で印を付ける）。
"""
import numpy as np
import common as C


def mouse_match(tr, joint="wrist", lags_ms=np.arange(-150, 151, 2)):
    """マウス（真値）と関節の机座標 xy を比べる。"""
    if joint not in tr.rs_names:
        return {}
    X = tr.joint_grid(joint)[:, :2]
    M = tr.task_grid("mouse")
    ok = np.isfinite(X).all(1) & np.isfinite(M).all(1)
    if ok.sum() < 200:
        return {"n": int(ok.sum())}
    Xc = X - np.nanmean(X[ok], 0); Mc = M - np.nanmean(M[ok], 0)
    gx = float(np.polyfit(Mc[ok, 0], Xc[ok, 0], 1)[0])
    gy = float(np.polyfit(Mc[ok, 1], Xc[ok, 1], 1)[0])
    resid = float(np.sqrt(np.nanmean(np.sum((Xc[ok] - Mc[ok]) ** 2, 1))))
    # 遅れ: マウスを +L ずらして残差が最小になる L（+ = RS が遅れている）
    # 残差はゲインを合わせたうえで評価する（手首は手の回転のぶんだけゲインが 1 を下回るため）
    t = tr.grid_t
    Mg = Mc * np.array([gx, gy])
    best, vals = np.nan, []
    for L in lags_ms / 1000.0:
        Ms = np.stack([np.interp(t, t + L, Mg[:, k]) for k in range(2)], 1)
        o = ok & np.isfinite(Ms).all(1)
        vals.append(np.sqrt(np.mean(np.sum((Xc[o] - Ms[o]) ** 2, 1))))
    vals = np.array(vals)
    i = int(np.argmin(vals))
    if 0 < i < len(vals) - 1:                                   # 放物線補間
        d = (vals[i - 1] - vals[i + 1]) / (2 * (vals[i - 1] - 2 * vals[i] + vals[i + 1]))
        best = float(lags_ms[i] + d * (lags_ms[1] - lags_ms[0]))
    else:
        best = float(lags_ms[i])
    r = [float(np.corrcoef(Xc[ok, k], Mc[ok, k])[0, 1]) for k in range(2)]
    return {"gain_x": gx, "gain_y": gy, "resid_mm": resid, "lag_ms": best,
            "r_x": r[0], "r_y": r[1], "resid_at_lag_mm": float(vals[i]), "n": int(ok.sum())}


def jitter(tr, joint, win_s=1.0, still_mm=6.0):
    """動きの少ない 1 秒窓での高域成分の SD = ジッタ σ（3D と成分別）。静止試行ではほぼ全区間が対象。"""
    X = tr.rs(joint, False)
    t = tr.t
    ok = tr.in_span & np.isfinite(X).all(1)
    if ok.sum() < 120:
        return {}
    Xf = tr.rs(joint, True)
    hp = X - Xf                                   # 5 Hz より上 = 揺れ
    w = max(3, int(round(win_s * tr.fs)))
    sds, comps = [], []
    idx = np.where(ok)[0]
    for s in range(0, len(idx) - w, w):
        sl = idx[s:s + w]
        if not np.isfinite(X[sl]).all():
            continue
        if np.ptp(Xf[sl], axis=0).max() > still_mm * 10:        # 大きく動く窓も高域は使える（甘めに）
            pass
        d = hp[sl]
        sds.append(np.sqrt(np.sum(d.var(0))))
        comps.append(d.std(0))
    if not sds:
        return {}
    comps = np.array(comps)
    return {"sd3d": float(np.median(sds)), "sd_x": float(np.median(comps[:, 0])),
            "sd_y": float(np.median(comps[:, 1])), "sd_z": float(np.median(comps[:, 2])),
            "n_win": len(sds)}


def run(tr):
    md = "## c0 データの健全性と測れる限界\n\n"
    rows = []
    N = int(tr.in_span.sum())
    for j, n in enumerate(tr.rs_names):
        X = tr.P_cam[:, j]
        ok = tr.in_span & np.isfinite(X).all(1)
        # 最長の欠損
        bad = tr.in_span & ~np.isfinite(X).all(1)
        longest, run_len = 0.0, 0
        for b in bad[tr.in_span]:
            run_len = run_len + 1 if b else 0
            longest = max(longest, run_len / tr.fs)
        row = {"joint": n, "label": C.LABEL[n], "valid_pct": 100 * ok.sum() / max(N, 1),
               "gap_max_s": longest}
        row.update({f"jit_{k}": v for k, v in jitter(tr, n).items()})
        if tr.status is not None and tr.status_names is not None:
            st = tr.status[tr.in_span, j]
            for k, sn in enumerate(tr.status_names):
                if sn != "OK":
                    row[f"st_{sn}"] = 100 * float(np.mean(st == k))
        rows.append(row)
    lr = float(100 * np.mean(tr.lr_swapped[tr.in_span])) if tr.lr_swapped is not None else np.nan
    # マウスとの照合は関節ごとに出す。ゲインは「手先に近い関節ほど 1 に近い」のが正しい
    # （手首は手の回転のぶんだけ 1 を下回る。その下回り方は人の型によって変わる）。
    # 代表値 mouse_match には最も手先に近い関節を使う
    mm_by = {n: mouse_match(tr, n) for n in ("wrist", "mcp3", "wrist_pose") if n in tr.rs_names}
    ref_joint = "mcp3" if mm_by.get("mcp3", {}).get("n") else "wrist"
    mm = dict(mm_by.get(ref_joint, {})); mm["joint"] = ref_joint

    # --- フレーム時刻
    dt = np.diff(tr.t[tr.in_span])
    timing = {"fs_rs": tr.fs, "period_ms": float(1000 * np.median(dt)),
              "drop": int(np.sum(dt > 1.8 / tr.fs)), "fs_task": tr.fs_task,
              "dt_max_task_ms": float(1000 * np.max(np.diff(tr.task_t)))}

    # --- 周波数成分ごとの SNR
    snr_rows = []
    if tr.traj:
        parts = {n: tr.joint_grid(n) for n in tr.rs_names}
        for ax, k in zip(C.AX, (0, 1)):
            nb, f, _ = tr.freqs(ax)
            for n in tr.rs_names:
                x = parts[n][:, k]
                bins, F = C.spectrum(x, tr.T)
                amp = np.abs(C.at_bins(bins, F, nb))
                noise = C.bin_noise(F, nb)
                for i in range(len(nb)):
                    snr_rows.append({"axis": ax, "n": int(nb[i]), "f_hz": float(f[i]), "joint": n,
                                     "label": C.LABEL[n], "amp_mm": float(amp[i]),
                                     "noise_mm": float(noise[i]), "snr": float(amp[i] / noise[i])})

    tr.save_json("a0_quality.json", {"header": tr.header(), "joints": rows, "lr_swapped_pct": lr,
                                     "mouse_match": mm, "mouse_match_by_joint": mm_by,
                                     "timing": timing, "snr": snr_rows})

    md += (f"- 解析区間 {N} フレーム（{N / tr.fs:.1f} s，{tr.fs:.1f} fps，周期 {timing['period_ms']:.2f} ms，"
           f"欠落 {timing['drop']} 回）。課題は {tr.fs_task:.0f} fps（最大 {timing['dt_max_task_ms']:.1f} ms）\n")
    md += f"- 左右の取り違え: {lr:.1f} %\n"
    md += f"- 机の座標系: {tr.frame_source}"
    if np.isfinite(tr.frame.get("gain_x", np.nan)):
        md += (f"（手 / マウス のゲイン x {tr.frame['gain_x']:.3f}，y {tr.frame['gain_y']:.3f}，"
               f"2 軸のなす角 {tr.frame.get('axis_angle_deg', float('nan')):.1f}°，残差 {tr.frame['resid_mm']:.1f} mm）")
    md += "\n\n"
    if mm:
        md += (f"**マウスを真値にした照合（{C.LABEL.get(ref_joint, ref_joint)}）**: "
               f"ゲイン x {mm.get('gain_x', np.nan):.3f} / y {mm.get('gain_y', np.nan):.3f}，"
               f"r x {mm.get('r_x', np.nan):.3f} / y {mm.get('r_y', np.nan):.3f}，"
               f"残差 {mm.get('resid_mm', np.nan):.1f} mm（遅れを合わせると {mm.get('resid_at_lag_mm', np.nan):.1f} mm），"
               f"**遅れ {mm.get('lag_ms', np.nan):+.0f} ms**（+ = RS が遅れ）\n\n")
    cols = [("関節", "label"), ("有効率 [%]", "valid_pct"), ("最長欠損 [s]", "gap_max_s", "{:.2f}"),
            ("ジッタ σ3D [mm]", "jit_sd3d", "{:.2f}"), ("σx", "jit_sd_x", "{:.2f}"),
            ("σy", "jit_sd_y", "{:.2f}"), ("σz", "jit_sd_z", "{:.2f}")]
    for sn in ("NO_POSE", "NO_HAND", "REJECT", "OUT", "NO_DEPTH"):
        if any(f"st_{sn}" in r for r in rows):
            cols.append((sn, f"st_{sn}", "{:.1f}"))
    md += C.md_table(rows, cols) + "\n"

    if snr_rows:
        plt = C.setup_plot()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
        for a, ax in zip(axes, C.AX):
            for n in tr.rs_names:
                s = [r for r in snr_rows if r["axis"] == ax and r["joint"] == n]
                a.plot([r["f_hz"] for r in s], [r["snr"] for r in s], "o-", color=C.COLOR[n],
                       label=C.LABEL[n], lw=1.2, ms=4)
            a.axhline(C.SNR_MIN, color="#c00", ls="--", lw=1)
            a.set_xscale("log"); a.set_yscale("log")
            a.set_xlabel("周波数 [Hz]"); a.set_title(f"{ax} 軸の入力周波数での SNR")
        axes[0].set_ylabel("振幅 / 近傍ビンのノイズ"); axes[0].legend(fontsize=8)
        fig.suptitle(f"{tr.name}（{tr.tag()}，DPI {tr.dpi:g}）", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{tr.out}/c0_snr.png", dpi=120); plt.close(fig)

        bad = [r for r in snr_rows if r["snr"] < C.SNR_MIN]
        names = sorted({"{} {:.2f} Hz".format(r["label"], r["f_hz"]) for r in bad})
        md += (f"\n入力周波数 × 関節 {len(snr_rows)} 組のうち，SNR < {C.SNR_MIN:.0f} は {len(bad)} 組"
               f"（{'，'.join(names)[:300] if names else 'なし'}）。"
               "この組は以降の解析では参考値として扱う。\n")
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
