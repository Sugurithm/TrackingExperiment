"""a1 静止時のジッタ（RealSense 単体の精密さ）

何をするか
  腕を止めている区間（MoCap の手首・肘の 1 秒の振れ幅 < 4 mm を自動検出）で，
  RS の位置がどれだけ揺れるかを測る。
  - σ（3D と 視線/横/縦 の成分）: RS − MoCap の SD（体の微小な動きを MoCap で差し引く）。
    真値のない関節（MCP3）は RS 単体の SD。
  - フレーム間ノイズ: 連続フレームの差の RMS / √2（ゆっくりした体の揺れに影響されない）
  - 画像上のジッタ [px]: MediaPipe の 2D 位置（uv）の SD → 深度センサと MediaPipe のどちらが揺れの原因か
  - 5 Hz 低域通過後の σ（解析で実際に使う精度）
わかること
  真値がなくても測れる「再現性（精密さ）」。動き方の解析で検出できる最小の変位の目安になる
  （最小検出変位 ≈ 2√2·σ ≈ 2.8σ）。
"""
import numpy as np
import common as C


def _uv_sd(tr, j, mk):
    if tr.uv is None: return np.nan
    U = tr.uv[mk, j]; U = U[~np.isnan(U).any(1)]
    return float(np.sqrt(np.sum(U.var(0)))) if len(U) > 20 else np.nan


def run(tr):
    rows = []; per_window = []
    for n in tr.rs_names:
        j = tr.rs_names.index(n)
        acc = {k: [] for k in ["sd3d", "sd_ray", "sd_h", "sd_v", "f2f", "sd3d_filt", "uv_px", "rs_only_sd3d", "mc_sd3d"]}
        for a, b in tr.rest:
            mk = tr.mask((a, b))
            Xr = tr.rs(n, False)[mk]; ref_r = tr.ref(n, False)
            ok = ~np.isnan(Xr).any(1)
            if ref_r is not None: ok &= ~np.isnan(ref_r[mk]).any(1)
            if ok.sum() < 0.8 * mk.sum() or ok.sum() < 60: continue
            Xr = Xr[ok]
            E = Xr - ref_r[mk][ok] if ref_r is not None else Xr
            D = C.decompose(E - E.mean(0), Xr)
            sd = D.std(0)
            Xf = tr.rs(n, True)[mk][ok]
            Ef = Xf - tr.ref(n, True)[mk][ok] if ref_r is not None else Xf
            f2f = np.sqrt(np.mean(np.sum(np.diff(Xr, axis=0) ** 2, 1)) / 2)
            vals = dict(sd3d=np.sqrt(np.sum(sd ** 2)), sd_ray=sd[0], sd_h=sd[1], sd_v=sd[2], f2f=f2f,
                        sd3d_filt=np.sqrt(np.sum(Ef.var(0))), uv_px=_uv_sd(tr, j, mk),
                        rs_only_sd3d=np.sqrt(np.sum(Xr.var(0))),
                        mc_sd3d=np.sqrt(np.sum(ref_r[mk][ok].var(0))) if ref_r is not None else np.nan)
            for k, v in vals.items(): acc[k].append(v)
            per_window.append({"joint": n, "t0": a, "t1": b, **vals})
        row = {"joint": n, "label": C.LABEL.get(n, n), "n_windows": len(acc["sd3d"]),
               "ref": "RS−MC" if tr.ref(n) is not None else "RS のみ"}
        for k, v in acc.items():
            v = [x for x in v if np.isfinite(x)]
            row[k] = float(np.median(v)) if v else np.nan
        row["mdd_mm"] = 2 * np.sqrt(2) * row["sd3d_filt"]
        # 1 px が奥行き Z で何 mm か
        Z = np.nanmedian(tr.rs(n)[tr.in_span][:, 2]); fx = tr.calib.get("color_intrinsics", {}).get("fx", 611.4)
        row["mm_per_px"] = Z / fx
        rows.append(row)
    tr.save_json("a1_jitter.json", {"summary": rows, "windows": per_window, "rest": tr.rest})

    plt = C.setup_plot()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    xs = np.arange(len(rows)); w = .2
    for i, (k, lab) in enumerate([("sd_ray", "奥行き(視線)"), ("sd_h", "横(画像)"), ("sd_v", "縦(画像)"), ("sd3d_filt", "3D（5 Hz 後）")]):
        ax[0].bar(xs + (i - 1.5) * w, [r[k] for r in rows], w, label=lab,
                  color=["#2a78d6", "#1baf7a", "#eda100", "#444"][i])
    ax[0].set_xticks(xs); ax[0].set_xticklabels([r["label"] for r in rows]); ax[0].set_ylabel("SD [mm]")
    ax[0].set_title("静止時のジッタ（生データの成分と 5 Hz 後の 3D）"); ax[0].legend(fontsize=8)
    # 例: 最長の静止区間の手首
    if tr.rest:
        a, b = max(tr.rest, key=lambda s: s[1] - s[0]); mk = tr.mask((a, b))
        for n in ["wrist", "elbow"]:
            if n not in tr.rs_names: continue
            E = tr.rs(n, False)[mk] - tr.ref(n, False)[mk]
            D = C.decompose(E - np.nanmean(E, 0), tr.rs(n, False)[mk])
            ax[1].plot(tr.t[mk] - a, D[:, 0], lw=.8, color=C.COLOR[n], label=f"{C.LABEL[n]} 奥行き")
            ax[1].plot(tr.t[mk] - a, D[:, 1], lw=.8, color=C.COLOR[n], alpha=.45, label=f"{C.LABEL[n]} 横")
        ax[1].set_xlabel("時間 [s]"); ax[1].set_ylabel("RS − MoCap（平均を引いた値）[mm]")
        ax[1].set_title(f"静止区間の例（{a:.0f}–{b:.0f} s，生データ）"); ax[1].legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a1_jitter.png", dpi=120); plt.close(fig)

    md = "## a1 静止時のジッタ（RealSense 単体の精密さ）\n\n"
    md += f"静止区間 {len(tr.rest)} 個（計 {sum(b - a for a, b in tr.rest):.0f} s）。各区間の値の中央値。\n\n"
    md += C.md_table(rows, [("関節", "label"), ("基準", "ref"), ("σ3D 生 [mm]", "sd3d"), ("σ奥行き", "sd_ray"),
                            ("σ横", "sd_h"), ("σ縦", "sd_v"), ("フレーム間 [mm]", "f2f"),
                            ("σ3D 5Hz後", "sd3d_filt"), ("最小検出変位", "mdd_mm"), ("画像 σ [px]", "uv_px"),
                            ("1px=mm", "mm_per_px")], "{:.2f}")
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
