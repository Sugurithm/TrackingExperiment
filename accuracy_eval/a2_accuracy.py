"""a2 正確度（MoCap に対する絶対誤差）

何をするか
  較正 T_MC で MoCap をカメラ座標に移し，RS − MoCap を関節 × 区間（静止・各テンポ）で集計する。
  誤差は「奥行き（視線方向，+ = RS が奥）」「画像の横」「画像の縦」に分ける。
  - 偏り（平均誤差）: 皮膚表面と関節中心の差など「一定のずれ」
  - SD: ばらつき
  - RMSE（3D）と 95 パーセンタイル: 総合的な誤差の大きさ
  - 一致限界（Bland–Altman, 偏り ± 1.96 SD）: 95% のフレームが入る範囲
わかること
  RS の値を「そのまま絶対位置として」使ったときにどれだけ外れるか。
  偏りとばらつきのどちらが主かがわかり，a3（偏りの一定性）につながる。
"""
import numpy as np
import common as C


def stats(tr, n, mk, filtered=True):
    X = tr.rs(n, filtered)[mk]; R = tr.ref(n, filtered)[mk]
    ok = ~(np.isnan(X).any(1) | np.isnan(R).any(1))
    if ok.sum() < 30: return None
    E = X[ok] - R[ok]; D = C.decompose(E, X[ok]); e3 = np.linalg.norm(E, axis=1)
    b = D.mean(0); s = D.std(0)
    return {"n": int(ok.sum()), "bias_ray": b[0], "bias_h": b[1], "bias_v": b[2], "bias3d": float(np.linalg.norm(b)),
            "sd_ray": s[0], "sd_h": s[1], "sd_v": s[2], "sd3d": float(np.sqrt(np.sum(s ** 2))),
            "rmse": float(np.sqrt(np.mean(e3 ** 2))), "p95": float(np.percentile(e3, 95)),
            "loa_ray": (b[0] - 1.96 * s[0], b[0] + 1.96 * s[0]),
            "loa_h": (b[1] - 1.96 * s[1], b[1] + 1.96 * s[1]), "loa_v": (b[2] - 1.96 * s[2], b[2] + 1.96 * s[2]),
            "bias_cam_xyz": (X[ok] - R[ok]).mean(0)}


def segments(tr):
    segs = [("静止", None)] + [(b["name"], (b["t0"], b["t1"])) for b in tr.blocks] + [("動作全体", "move")]
    out = []
    for name, s in segs:
        if s is None:
            mk = np.zeros(len(tr.t), bool)
            for r in tr.rest: mk |= tr.mask(r)
        elif s == "move":
            mk = np.zeros(len(tr.t), bool)
            for b in tr.blocks: mk |= tr.mask((b["t0"], b["t1"]))
        else:
            mk = tr.mask(s)
        out.append((name, mk))
    return out


def run(tr):
    rows = []
    for n in tr.compared():
        for name, mk in segments(tr):
            s = stats(tr, n, mk); raw = stats(tr, n, mk, False)
            if s is None: continue
            rows.append({"joint": n, "label": C.LABEL[n], "segment": name, **s,
                         "rmse_raw": raw["rmse"] if raw else np.nan})
    tr.save_json("a2_accuracy.json", rows)

    plt = C.setup_plot()
    J = tr.compared()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    allm = [r for r in rows if r["segment"] == "動作全体"]
    xs = np.arange(len(allm)); w = .25
    for i, a in enumerate(C.AXES):
        b = np.array([r[f"bias_{a}"] for r in allm]); lo = np.array([r[f"loa_{a}"][0] for r in allm]); hi = np.array([r[f"loa_{a}"][1] for r in allm])
        ax[0].bar(xs + (i - 1) * w, b, w, color=["#2a78d6", "#1baf7a", "#eda100"][i], label=C.AXLABEL[a])
        ax[0].errorbar(xs + (i - 1) * w, b, yerr=[b - lo, hi - b], fmt="none", ecolor="#333", capsize=3, lw=1)
    ax[0].axhline(0, color="#888", lw=1); ax[0].set_xticks(xs); ax[0].set_xticklabels([r["label"] for r in allm])
    ax[0].set_ylabel("RS − MoCap [mm]"); ax[0].set_title("偏りと 95% 一致限界（動作全体）"); ax[0].legend(fontsize=8)
    bpm = [b["bpm_oneway"] for b in tr.blocks]
    for n in J:
        y = [next(r["rmse"] for r in rows if r["joint"] == n and r["segment"] == b["name"]) for b in tr.blocks]
        y2 = [next(r["sd3d"] for r in rows if r["joint"] == n and r["segment"] == b["name"]) for b in tr.blocks]
        ax[1].plot(bpm, y, "-o", color=C.COLOR[n], label=C.LABEL[n], lw=2, ms=6)
        ax[2].plot(bpm, y2, "-o", color=C.COLOR[n], label=C.LABEL[n], lw=2, ms=6)
    ax[1].set_title("RMSE（偏りを含む）"); ax[2].set_title("ばらつき（偏りを除いた SD, 3D）")
    for a in ax[1:]:
        a.set_xlabel("テンポ [bpm]"); a.set_ylabel("[mm]"); a.set_ylim(bottom=0); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a2_accuracy.png", dpi=120); plt.close(fig)

    md = "## a2 正確度（MoCap に対する絶対誤差，カメラ座標）\n\n"
    md += "奥行き + = RS が奥。前処理後（Hampel + 5 Hz）。RMSE(生) は前処理なし。\n\n"
    md += C.md_table([r for r in rows if r["segment"] in ("静止", "動作全体")],
                     [("関節", "label"), ("区間", "segment"), ("偏り奥行き", "bias_ray"), ("偏り横", "bias_h"),
                      ("偏り縦", "bias_v"), ("|偏り|", "bias3d"), ("SD 3D", "sd3d"), ("RMSE", "rmse"),
                      ("95%", "p95"), ("RMSE(生)", "rmse_raw")])
    md += "\nテンポ別 RMSE [mm]\n\n"
    tab = []
    for n in tr.compared():
        d = {"label": C.LABEL[n]}
        for b in tr.blocks:
            d[b["name"]] = next(r["rmse"] for r in rows if r["joint"] == n and r["segment"] == b["name"])
        tab.append(d)
    md += C.md_table(tab, [("関節", "label")] + [(b["name"], b["name"]) for b in tr.blocks])
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
