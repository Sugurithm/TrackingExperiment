"""a0 基準（MoCap・較正・同期）の不確かさ

何をするか
  RealSense を評価する「ものさし」自体の誤差を並べる。
  - 較正 T_MC の残差（T_MC.json の rms / LOO）
  - 時刻同期の残差（sync_result.json のボードの横方向の誤差，前後の区間の δ の差＝drift）
  - MoCap 関節の静止時のゆらぎ（静止区間の SD）
  - MoCap 骨長のばらつき（関節中心をソルバーが推定する不確かさ）
わかること
  RealSense の誤差のうち，どこまでが「真値側の誤差」で説明できてしまうか（評価の下限）。
  例: 較正 2 mm・同期 1 mm なら，RS の誤差 3 mm 未満は区別できない。
"""
import numpy as np
import common as C


def run(tr):
    cal = tr.calib; sy = tr.sync
    loo = [p["loo_mm"] for p in cal.get("poses", []) if "loo_mm" in p]
    out = {"calib": {"rms_mm": cal.get("rms_mm"), "max_mm": cal.get("max_mm"),
                     "loo_min_mm": min(loo) if loo else None, "loo_max_mm": max(loo) if loo else None,
                     "n_poses": len(loo), "scale_cam_over_mocap": cal.get("scale_cam_over_mocap")}}
    segs = []
    for i, s in enumerate(sy["segments"]):
        segs.append({"segment": i, "delta_s": s.get("delta"), "corr": s.get("corr"),
                     "lateral_median_mm": s.get("median_mm"), "lateral_p95_mm": s.get("p95_mm"),
                     "err3d_median_mm": s.get("median_3d_mm"), "depth_range_mm": s.get("depth_range_mm")})
    drift = sy.get("drift_ms")
    vmax = max([b["vmax_mps"] for b in tr.blocks], default=np.nan)
    out["sync"] = {"segments": segs, "drift_ms": drift,
                   "pos_err_if_drift_unmodelled_mm": (abs(drift) * vmax if drift is not None else None),
                   "note": "c は区間中央で直線補間しているので drift の大部分は補正済み。上の値は補正しなかった場合の上限"}

    # MoCap の静止時のゆらぎ（120 fps のまま，カメラ座標）
    jit = {}
    for j, n in enumerate(tr.mc_names):
        sds = []
        for a, b in tr.rest:
            mk = (tr.t_mc >= a) & (tr.t_mc <= b)
            X = tr.M_cam_native[mk, j]; X = X[~np.isnan(X).any(1)]
            if len(X) > 50: sds.append(np.sqrt(np.sum(X.var(0))))
        jit[n] = float(np.median(sds)) if sds else np.nan
    out["mocap_rest_sd3d_mm"] = jit

    # MoCap 骨長
    def seglen(a, b):
        mk = (tr.t_mc > tr.span[0]) & (tr.t_mc < tr.span[1])
        ia, ib = tr.mc_names.index(a), tr.mc_names.index(b)
        L = np.linalg.norm(tr.M_cam_native[mk, ia] - tr.M_cam_native[mk, ib], axis=1); L = L[~np.isnan(L)]
        return {"mean_mm": float(L.mean()), "sd_mm": float(L.std()), "p5_mm": float(np.percentile(L, 5)),
                "p95_mm": float(np.percentile(L, 95))}
    bones = {}
    if {"shoulder", "elbow"} <= set(tr.mc_names): bones["upper_arm"] = seglen("shoulder", "elbow")
    if {"elbow", "wrist"} <= set(tr.mc_names): bones["forearm"] = seglen("elbow", "wrist")
    out["mocap_bone"] = bones

    budget = np.sqrt((cal.get("rms_mm") or 0) ** 2 + (segs[0]["lateral_median_mm"] or 0) ** 2)
    out["reference_floor_mm"] = float(budget)
    tr.save_json("a0_reference.json", out)

    md = "## a0 基準（MoCap・較正・同期）の不確かさ\n\n"
    md += f"- 較正 T_MC: 残差 RMS {cal.get('rms_mm', np.nan):.2f} mm（最大 {cal.get('max_mm', np.nan):.2f}），LOO {out['calib']['loo_min_mm']:.1f}–{out['calib']['loo_max_mm']:.1f} mm（{len(loo)} 姿勢），尺度比 {cal.get('scale_cam_over_mocap', np.nan):.4f}\n"
    for s in segs:
        md += f"- 同期 区間{s['segment']}: δ = {s['delta_s']:+.4f} s，相関 {s['corr']:.3f}，ボードの横方向誤差 中央値 {s['lateral_median_mm']:.2f} mm（95% {s['lateral_p95_mm']:.2f}），3D 中央値 {s['err3d_median_mm']:.2f} mm\n"
    if drift is not None:
        md += f"- 時計のずれの変化（drift）: {drift:.1f} ms（区間間）。補正しなければ最大速 {vmax:.2f} m/s で {abs(drift) * vmax:.1f} mm 相当 → 直線補間で補正済み\n"
    md += "- MoCap 静止時のゆらぎ（3D SD）: " + "，".join(f"{C.LABEL.get(k, k)} {v:.2f} mm" for k, v in jit.items()) + "\n"
    for k, v in bones.items():
        md += f"- MoCap {'上腕' if k == 'upper_arm' else '前腕'}長: {v['mean_mm']:.1f} mm，SD {v['sd_mm']:.2f} mm（5–95%: {v['p5_mm']:.1f}–{v['p95_mm']:.1f}）\n"
    md += f"\n**基準側の誤差の目安 ≈ {budget:.1f} mm**（較正 RMS と同期の横方向誤差の二乗和平方根）。これより小さい RS の誤差は区別できない。\n"
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
