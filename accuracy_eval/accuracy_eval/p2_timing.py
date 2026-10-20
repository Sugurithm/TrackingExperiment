"""p2 フェーズ別の時間の一致（遅れ・折り返しの時刻）

何をするか（動作フェーズのみ。時計のずれはボードで補正済みなので，残るのは MediaPipe とフィルタの遅れ）
  - 変位の遅れ: MoCap を ±150 ms ずらし，偏りを除いた誤差が最小になる時間
  - 速度の遅れ: 主な方向の速度（符号つき）の相互相関のピーク位置と，そのときの相関係数
  - 実効的な遅れ: 動きの方向の誤差 ≈ −遅れ × 速さ の傾き
  - 折り返しの時刻の誤差: MoCap の各折り返し（ストロークの端）に最も近い RS の折り返しとの時間差の平均・SD，
    見つかった割合（±150 ms 以内）
  いずれも + = RS が遅れる。
わかること
  条件（テンポ）ごとの時間のずれ。関節ごとに遅れが違うと，関節間のタイミング（協調）の評価がずれる。
  折り返し時刻の SD は「動きのイベントの時刻をどれだけ正確に測れるか」。
"""
import numpy as np
import common as C
import phase_common as P

MAXLAG = 0.15
MIN_SD = 20.0


def lag_disp(tr, n, ph):
    X, R, t, ok = P.pair(tr, n, ph, fill=True)
    Rfull = C.fill_nan(tr.ref(n))[0]
    Ls = np.arange(-MAXLAG, MAXLAG + 1e-9, 0.001); v = []
    for L in Ls:
        Rs = np.stack([np.interp(t - L, tr.t, Rfull[:, k]) for k in range(3)], 1)
        e = (X - Rs)[ok]; e -= e.mean(0); v.append(np.mean(np.sum(e ** 2, 1)))
    v = np.array(v); i = int(v.argmin())
    return float((Ls[0] + 0.001 * P.sub_frame_peak(v, i)) * 1000) if 0 < i < len(v) - 1 else np.nan


def run(tr, phs=None):
    phs = phs or P.phases(tr)
    J = tr.compared(); move = [p for p in phs if p["kind"] == "move"]
    rows = []
    for p in move:
        for n in J:
            X, R, t, ok = P.pair(tr, n, p, fill=True)
            ax = P.main_axis(R); sR = (R - R.mean(0)) @ ax; sX = (X - X.mean(0)) @ ax
            row = {"phase": p["name"], "joint": n, "motion_sd": float(sR.std())}
            if sR.std() < MIN_SD:
                rows.append(row); continue
            row["lag_disp_ms"] = lag_disp(tr, n, p)
            vR = np.gradient(sR) * tr.fs; vX = np.gradient(sX) * tr.fs
            maxk = int(MAXLAG * tr.fs); cc = []
            for k in range(-maxk, maxk + 1):
                a = vX[max(0, k): len(vX) + min(0, k)]; b = vR[max(0, -k): len(vR) + min(0, -k)]
                cc.append(np.corrcoef(a, b)[0, 1])
            cc = np.array(cc); i = int(cc.argmax())
            row["lag_vel_ms"] = float((P.sub_frame_peak(cc, i) - maxk) / tr.fs * 1000); row["xcorr_peak"] = float(cc.max())
            # 実効的な遅れ（3D）
            vel = np.gradient(R, axis=0) * tr.fs; sp = np.linalg.norm(vel, axis=1)
            Ev = (X - R) - (X - R)[ok].mean(0); along = np.sum(Ev * vel / np.maximum(sp, 1e-9)[:, None], 1)
            m = ok & (sp > 50)
            if m.sum() > 50:
                row["lag_reg_ms"] = float(-np.polyfit(sp[m], along[m], 1)[0] * 1000)
            # 折り返し
            tR = P.turnarounds(sR, tr.fs); tX = P.turnarounds(sX, tr.fs)
            d = []
            for i_ in tR:
                if len(tX) == 0: break
                j = tX[np.abs(tX - i_).argmin()]
                if abs(j - i_) <= MAXLAG * tr.fs:
                    # 放物線補間で極値の時刻を細かく
                    pi = P.sub_frame_peak(sR * np.sign(sR[i_]), i_); pj = P.sub_frame_peak(sX * np.sign(sX[j]), j)
                    d.append((pj - pi) / tr.fs * 1000)
            if d:
                row.update(turn_mean_ms=float(np.mean(d)), turn_sd_ms=float(np.std(d)), turn_found_pct=100 * len(d) / max(len(tR), 1),
                           n_turn=len(tR))
            rows.append(row)
    P.save(tr, "p2_timing.json", {"phases": P.strip_masks(move), "rows": rows})

    plt = C.setup_plot()
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
    P.plot_by_phase(ax[0, 0], rows, "lag_disp_ms", J, move, "変位の遅れ [ms]", "+ = RS が遅い", hline=0)
    P.plot_by_phase(ax[0, 1], rows, "lag_vel_ms", J, move, "速度の相互相関の遅れ [ms]", "+ = RS が遅い", hline=0)
    P.plot_by_phase(ax[0, 2], rows, "lag_reg_ms", J, move, "実効的な遅れ（誤差 vs 速さの傾き）[ms]", "", hline=0)
    P.plot_by_phase(ax[1, 0], rows, "turn_mean_ms", J, move, "折り返し時刻の差の平均 [ms]", "+ = RS が遅い", hline=0)
    P.plot_by_phase(ax[1, 1], rows, "turn_sd_ms", J, move, "折り返し時刻の差の SD [ms]", "", ylim0=True)
    P.plot_by_phase(ax[1, 2], rows, "xcorr_peak", J, move, "速度の相互相関の最大値", "")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{P.outdir(tr)}/p2_timing.png", dpi=120); plt.close(fig)

    md = "## p2 フェーズ別の時間の一致（+ = RS が遅れる）\n\n"
    for k, t, f in [("lag_disp_ms", "変位の遅れ [ms]", "{:+.0f}"), ("lag_vel_ms", "速度の相互相関の遅れ [ms]", "{:+.0f}"),
                    ("xcorr_peak", "速度の相互相関の最大値", "{:.3f}"), ("lag_reg_ms", "実効的な遅れ [ms]", "{:+.0f}"),
                    ("turn_mean_ms", "折り返し時刻の差の平均 [ms]", "{:+.0f}"), ("turn_sd_ms", "折り返し時刻の差の SD [ms]", "{:.0f}"),
                    ("turn_found_pct", "折り返しが ±150 ms 以内に見つかった割合 [%]", "{:.0f}")]:
        md += f"\n{t}\n\n" + P.pivot(rows, k, J, move, f)
    md += f"\n動きの SD が {MIN_SD:.0f} mm 未満の関節（肩）は求めない（–）。\n"
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
