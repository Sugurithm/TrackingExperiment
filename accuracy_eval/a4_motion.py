"""a4 動きの再現性（変位・遅れ・振幅・速さ）

何をするか
  各テンポのブロックで，一定の偏りを除いた「動きそのもの」が MoCap と一致するかを見る。
  - 形の誤差: ブロックの偏りを引いた RMSE と，それを動きの範囲で割った値（NRMSE [%]）
  - 遅れ: MoCap を ±150 ms ずらして誤差が最小になる時間（+ = RS が遅れる）
  - 振幅比・ゲイン・位相: 主な動きの方向で，振れ幅の比（p97−p3）と基本周波数での比・位相差
  - 速さ: 相関 r，差の RMS，最大の速さの比（p99）
わかること
  動き方の解析（速度・変位・タイミング）に使うときの信頼性と，速くなると何が崩れるか。
  遅れはそのまま「関節間のタイミング（協調）」の誤差になるので重要。
"""
import numpy as np
import common as C

MIN_MOTION_SD = 20.0   # 動きの SD がこれ未満の関節は遅れ・ゲインを求めない


def _shift(tr, R, L):
    """R(t − L) を RS 時刻で返す（+L = RS が遅れている仮定）。"""
    out = np.empty_like(R)
    for k in range(3): out[:, k] = np.interp(tr.t - L, tr.t, R[:, k])
    return out


def run(tr):
    rows = []
    for b in tr.blocks:
        mk = tr.mask((b["t0"], b["t1"]))
        for n in tr.compared():
            X = tr.rs(n)[mk]; Rfull = np.nan_to_num(tr.ref(n)); R = Rfull[mk]
            ok = ~(np.isnan(X).any(1) | np.isnan(tr.ref(n)[mk]).any(1))
            if ok.sum() < 0.8 * mk.sum(): continue
            X, R = X[ok], R[ok]
            E = X - R; Eb = E - E.mean(0)
            Rc = R - R.mean(0); ax = np.linalg.svd(Rc, full_matrices=False)[2][0]
            a_mc = Rc @ ax; a_rs = (X - X.mean(0)) @ ax
            rng = np.percentile(a_mc, 97) - np.percentile(a_mc, 3)
            row = {"block": b["name"], "bpm": b["bpm_oneway"], "joint": n, "label": C.LABEL[n],
                   "motion_sd_mm": float(a_mc.std()), "range_mm": float(rng),
                   "shape_rmse": float(np.sqrt(np.mean(np.sum(Eb ** 2, 1)))),
                   "pos_r": float(np.corrcoef(a_rs, a_mc)[0, 1])}
            row["nrmse_pct"] = 100 * row["shape_rmse"] / rng if rng > 0 else np.nan
            row["amp_ratio"] = float((np.percentile(a_rs, 97) - np.percentile(a_rs, 3)) / rng)
            sR = C.speed(X, tr.fs); sM = C.speed(R, tr.fs)
            row["speed_r"] = float(np.corrcoef(sR, sM)[0, 1]); row["speed_rmsd_mmps"] = float(np.sqrt(np.mean((sR - sM) ** 2)))
            row["peak_speed_ratio"] = float(np.percentile(sR, 99) / np.percentile(sM, 99))
            if a_mc.std() >= MIN_MOTION_SD:
                Ls = np.arange(-0.15, 0.1501, 0.001); v = []
                for L in Ls:
                    Rs = _shift(tr, Rfull, L)[mk][ok]; e = X - Rs; e -= e.mean(0); v.append(np.mean(np.sum(e ** 2, 1)))
                v = np.array(v); i = int(v.argmin()); L = Ls[i]
                if 0 < i < len(v) - 1:
                    L += 0.001 * 0.5 * (v[i - 1] - v[i + 1]) / (v[i - 1] - 2 * v[i] + v[i + 1])
                row["lag_ms"] = float(L * 1000); row["shape_rmse_lagfree"] = float(np.sqrt(v.min()))
                w = np.hanning(len(a_mc)); nf = len(a_mc) * 8
                Fa = np.fft.rfft(a_mc * w, nf); Fb = np.fft.rfft(a_rs * w, nf)
                k = int(np.abs(Fa[1:]).argmax()) + 1
                row["gain"] = float(np.abs(Fb[k]) / np.abs(Fa[k])); row["phase_deg"] = float(np.degrees(np.angle(Fb[k] / Fa[k])))
            else:  # ほぼ動かない関節: 比や相関はノイズで決まるので出さない
                row.update(lag_ms=np.nan, shape_rmse_lagfree=np.nan, gain=np.nan, phase_deg=np.nan,
                           amp_ratio=np.nan, speed_r=np.nan, nrmse_pct=np.nan)
            rows.append(row)
    tr.save_json("a4_motion.json", rows)

    plt = C.setup_plot()
    J = tr.compared()
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    panels = [("shape_rmse", "形の誤差（偏り除去後 RMSE）[mm]"), ("nrmse_pct", "形の誤差 / 動きの範囲 [%]"),
              ("lag_ms", "遅れ [ms]（+ = RS が遅い）"), ("amp_ratio", "振幅比 RS/MoCap"),
              ("speed_r", "速さの相関 r"), ("peak_speed_ratio", "最大の速さの比 RS/MoCap")]
    for a, (k, title) in zip(ax.ravel(), panels):
        for n in J:
            rr = [r for r in rows if r["joint"] == n]
            a.plot([r["bpm"] for r in rr], [r[k] for r in rr], "-o", color=C.COLOR[n], label=C.LABEL[n], lw=2, ms=6)
        a.set_title(title); a.set_xlabel("テンポ [bpm]")
        if k in ("amp_ratio", "peak_speed_ratio"): a.axhline(1, color="#888", lw=1)
        if k == "lag_ms": a.axhline(0, color="#888", lw=1)
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a4_motion.png", dpi=120); plt.close(fig)

    # 最速ブロックの波形
    if tr.blocks:
        b = tr.blocks[-1]; mk = tr.mask((b["t0"], b["t1"]))
        fig, ax = plt.subplots(len(J), 2, figsize=(15, 2.5 * len(J)), sharex="col")
        for i, n in enumerate(J):
            X = tr.rs(n)[mk]; R = tr.ref(n)[mk]; Xr = tr.rs(n, False)[mk]
            Rc = R - np.nanmean(R, 0); axv = np.linalg.svd(np.nan_to_num(Rc), full_matrices=False)[2][0]
            off = np.nanmean(X - R, 0); tt = tr.t[mk] - b["t0"]
            ax[i, 0].plot(tt, Rc @ axv, color="#333", lw=2, label="MoCap")
            ax[i, 0].plot(tt, (Xr - off - np.nanmean(R, 0)) @ axv, color=C.COLOR[n], lw=.6, alpha=.4, label="RS 生")
            ax[i, 0].plot(tt, (X - off - np.nanmean(R, 0)) @ axv, color=C.COLOR[n], lw=1.5, label="RS 5 Hz")
            ax[i, 0].set_ylabel(f"{C.LABEL[n]}\n位置 [mm]")
            ax[i, 1].plot(tt, C.speed(np.nan_to_num(R), tr.fs) / 1000, color="#333", lw=2, label="MoCap")
            ax[i, 1].plot(tt, C.speed(np.nan_to_num(X), tr.fs) / 1000, color=C.COLOR[n], lw=1.5, label="RS 5 Hz")
            ax[i, 1].set_ylabel("速さ [m/s]")
        ax[0, 0].legend(fontsize=8, ncol=3); ax[0, 1].legend(fontsize=8)
        ax[0, 0].set_xlim(0, 4); ax[0, 1].set_xlim(0, 4)
        ax[-1, 0].set_xlabel("時間 [s]"); ax[-1, 1].set_xlabel("時間 [s]")
        fig.suptitle(f"最速ブロック {b['name']}（偏りを除いて重ねた主な動きの方向と速さ）", y=1.0)
        fig.tight_layout(); fig.savefig(f"{tr.out}/a4_waveform_fastest.png", dpi=120); plt.close(fig)

    md = "## a4 動きの再現性\n\n"
    for k, title, f in [("shape_rmse", "形の誤差（偏り除去後 RMSE）[mm]", "{:.1f}"), ("nrmse_pct", "形の誤差 / 動きの範囲 [%]", "{:.1f}"),
                        ("lag_ms", "遅れ [ms]（+ = RS が遅い）", "{:+.0f}"), ("amp_ratio", "振幅比 RS/MoCap", "{:.3f}"),
                        ("speed_r", "速さの相関 r", "{:.3f}"), ("peak_speed_ratio", "最大の速さの比", "{:.2f}")]:
        tab = []
        for n in J:
            d = {"label": C.LABEL[n]}
            for r in rows:
                if r["joint"] == n: d[r["block"]] = r[k]
            tab.append(d)
        md += f"\n{title}\n\n" + C.md_table(tab, [("関節", "label")] + [(b["name"], b["name"]) for b in tr.blocks], f)
    md += (f"\n動きの SD が {MIN_MOTION_SD:.0f} mm 未満の関節（主に肩）は遅れ・振幅比・相関を求めていない（–）。"
           "その場合の最大の速さの比は，ノイズが速さを水増しする程度を表す。\n")
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
