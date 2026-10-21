"""c5 関節間の協調（時間のずれ・位相・自由度の数）

何をするか
  1. **相互相関**: 関節どうしの動き（机座標 x, y の変位と、速さ）の相関が最大になる時間のずれ [ms]。
     + = 後ろに書いた関節が遅れている。近位（肩）→ 遠位（手首）の順序があるかを見る
  2. **入力周波数ごとの相対位相**: 関節の位相差（c3 と同じ FFT から）。周波数ごとに
     同相（一緒に動く）か逆相（打ち消し合う）かが分かる。平均と、周波数間のばらつき
  3. **PCA**: 全関節の変位（机座標、関節ごとに平均を引く）を 1 つの行列にして主成分分析。
     PC1 の寄与率 = どれだけ「1 つの動き」にまとまっているか、90 % に必要な PC 数 = 使っている自由度の数、
     PC1 のローディング = どの関節が主役か
  4. 関節ごとの動きの大きさ（RMS）と、手先との相関
わかること
  空間的な分担（c4）に対する**時間的な協調**。
  逆相の成分があれば、関節どうしが互いの動きを打ち消して手先を安定させている（冗長性の利用）と読める。
"""
import numpy as np
import common as C

MAX_LAG_MS = 200.0


def safe_r(a, b):
    """相関係数（定数・NaN なら NaN を返す。静止の試行で警告を出さないため）。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    o = np.isfinite(a) & np.isfinite(b)
    if o.sum() < 50 or a[o].std() == 0 or b[o].std() == 0:
        return np.nan
    return float(np.corrcoef(a[o], b[o])[0, 1])


def xcorr_lag(a, b, fs, max_lag_ms=MAX_LAG_MS):
    """a, b（1 次元、NaN 可）の相互相関が最大になる b の遅れ [ms] と、そのときの r。"""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 200:
        return np.nan, np.nan
    x = a[ok] - a[ok].mean(); y = b[ok] - b[ok].mean()
    if x.std() == 0 or y.std() == 0:
        return np.nan, np.nan
    m = int(round(max_lag_ms / 1000 * fs))
    lags = np.arange(-m, m + 1)
    c = np.correlate(x, y, mode="full") / (len(x) * x.std() * y.std())
    mid = len(x) - 1
    c = c[mid - m: mid + m + 1]
    i = int(np.argmax(c))
    return float(-lags[i] / fs * 1000), float(c[i])


def run(tr):
    if not tr.has_rs:
        return "## c5 関節間の協調\n\n関節データなしのため省略。\n"
    names = [n for n in tr.rs_names if n != "wrist_pose"]
    G = {n: tr.joint_grid(n) for n in names}
    fs = C.FS_U
    spd = {n: np.linalg.norm(np.gradient(G[n], axis=0), axis=1) * fs for n in names}

    # --- 相互相関
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            lx, rx = xcorr_lag(G[a][:, 0], G[b][:, 0], fs)
            ly, ry = xcorr_lag(G[a][:, 1], G[b][:, 1], fs)
            ls, rs = xcorr_lag(spd[a], spd[b], fs)
            pairs.append({"pair": f"{C.LABEL[a]} → {C.LABEL[b]}", "a": a, "b": b,
                          "lag_x_ms": lx, "r_x": rx, "lag_y_ms": ly, "r_y": ry,
                          "lag_speed_ms": ls, "r_speed": rs})

    # --- 入力周波数ごとの相対位相（手首を基準）
    phase_rows = []
    base = "wrist" if "wrist" in names else names[-1]
    if tr.kind == "track" and tr.traj:
        for k, ax in enumerate(C.AX):
            nb, f, _ = tr.freqs(ax)
            Fb = C.spectrum(G[base][:, k], tr.T)
            Bk = C.at_bins(*Fb, nb)
            for n in names:
                Fn = C.spectrum(G[n][:, k], tr.T)
                Xk = C.at_bins(*Fn, nb)
                noise = C.bin_noise(Fn[1], nb)
                rel = np.degrees(np.angle(Xk / Bk))
                for i in range(len(nb)):
                    phase_rows.append({"axis": ax, "joint": n, "label": C.LABEL[n], "n": int(nb[i]),
                                       "f_hz": float(f[i]), "rel_phase_deg": float(rel[i]),
                                       "amp_mm": float(np.abs(Xk[i])),
                                       "snr": float(np.abs(Xk[i]) / noise[i]),
                                       "ok": bool(np.abs(Xk[i]) / noise[i] > C.SNR_MIN)})

    # --- PCA（全関節の変位をまとめる）
    M = np.concatenate([G[n] - np.nanmean(G[n], axis=0) for n in names], axis=1)
    ok = np.isfinite(M).all(1)
    pca = {}
    if ok.sum() > 200:
        X = M[ok]
        U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        ev = S ** 2 / np.sum(S ** 2)
        cum = np.cumsum(ev)
        pca = {"explained": ev[:6].tolist(), "pc1": float(ev[0]), "pc12": float(cum[1]),
               "n_for_90pct": int(np.searchsorted(cum, 0.90) + 1), "n_frames": int(ok.sum())}
        # PC1 の関節ごとの大きさ（3 軸のノルム）
        load = {}
        for i, n in enumerate(names):
            load[n] = float(np.linalg.norm(Vt[0][3 * i:3 * i + 3]))
        pca["pc1_loading"] = load

    # --- 関節の大きさと手先との相関
    jrows = []
    endn = "mcp3" if "mcp3" in names else "wrist"
    for n in names:
        X = G[n]; o = np.isfinite(X).all(1) & np.isfinite(G[endn]).all(1)
        jrows.append({"joint": n, "label": C.LABEL[n],
                      "rms_mm": float(np.sqrt(np.nanmean(np.sum((X - np.nanmean(X, 0)) ** 2, 1)))),
                      "range_x_mm": float(np.nanmax(X[:, 0]) - np.nanmin(X[:, 0])),
                      "range_y_mm": float(np.nanmax(X[:, 1]) - np.nanmin(X[:, 1])),
                      "range_z_mm": float(np.nanmax(X[:, 2]) - np.nanmin(X[:, 2])),
                      "r_end_x": safe_r(X[:, 0], G[endn][:, 0]),
                      "speed_p95": float(np.nanpercentile(spd[n], 95))})
    tr.save_json("a3_coupling.json", {"pairs": pairs, "phase": phase_rows, "pca": pca, "joints": jrows,
                                      "phase_base": base})

    plt = C.setup_plot()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    # 遅れ
    xs = np.arange(len(pairs))
    axes[0].bar(xs - .2, [p["lag_x_ms"] for p in pairs], .4, label="x 変位", color="#2a78d6")
    axes[0].bar(xs + .2, [p["lag_speed_ms"] for p in pairs], .4, label="速さ", color="#eb6834")
    axes[0].set_xticks(xs); axes[0].set_xticklabels([p["pair"] for p in pairs], fontsize=7, rotation=20)
    axes[0].axhline(0, color="#888", lw=1)
    axes[0].set_ylabel("遅れ [ms]（+ = 後者が遅い）"); axes[0].set_title("関節間の時間のずれ")
    axes[0].legend(fontsize=8)
    # 相対位相
    if phase_rows:
        for n in names:
            s = [r for r in phase_rows if r["joint"] == n and r["axis"] == "x"]
            f = np.array([r["f_hz"] for r in s]); y = np.array([r["rel_phase_deg"] for r in s])
            g = np.array([r["ok"] for r in s])
            axes[1].plot(f, y, "-", color=C.COLOR[n], lw=1, alpha=.35)
            axes[1].plot(f[g], y[g], "o-", color=C.COLOR[n], lw=1.6, ms=4, label=C.LABEL[n])
        axes[1].axhline(0, color="#888", lw=1); axes[1].axhline(180, color="#888", ls=":", lw=1)
        axes[1].axhline(-180, color="#888", ls=":", lw=1)
        axes[1].set_xscale("log"); axes[1].set_xlabel("周波数 [Hz]")
        axes[1].set_ylabel(f"{C.LABEL[base]} に対する位相 [deg]")
        axes[1].set_title("x 軸の相対位相（0 = 同相，±180 = 逆相）"); axes[1].legend(fontsize=7)
    # PCA
    if pca:
        axes[2].bar(np.arange(1, len(pca["explained"]) + 1), pca["explained"], color="#2a78d6")
        axes[2].set_xlabel("主成分"); axes[2].set_ylabel("寄与率")
        axes[2].set_title(f"PCA（PC1 {pca['pc1']:.0%}，90 % に {pca['n_for_90pct']} 成分）")
    fig.suptitle(f"{tr.name}（{tr.tag()}，DPI {tr.dpi:g}）関節間の協調", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a3_coupling.png", dpi=120); plt.close(fig)

    md = "## c5 関節間の協調\n\n"
    md += C.md_table(jrows, [("関節", "label"), ("変位 RMS [mm]", "rms_mm"), ("x 範囲", "range_x_mm"),
                             ("y 範囲", "range_y_mm"), ("z 範囲", "range_z_mm"),
                             ("手先 x との r", "r_end_x", "{:.3f}"), ("速さ 95% [mm/s]", "speed_p95", "{:.0f}")])
    md += "\n"
    md += C.md_table(pairs, [("組", "pair"), ("x の遅れ [ms]", "lag_x_ms", "{:+.0f}"), ("r", "r_x", "{:.3f}"),
                             ("y の遅れ [ms]", "lag_y_ms", "{:+.0f}"), ("r", "r_y", "{:.3f}"),
                             ("速さの遅れ [ms]", "lag_speed_ms", "{:+.0f}"), ("r", "r_speed", "{:.3f}")])
    if pca:
        md += (f"\nPCA: PC1 {pca['pc1']:.1%}，PC1+PC2 {pca['pc12']:.1%}，90 % に **{pca['n_for_90pct']} 成分**。"
               "PC1 のローディング（関節ごとの大きさ）: "
               + "，".join(f"{C.LABEL[n]} {v:.2f}" for n, v in pca["pc1_loading"].items()) + "\n")
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
