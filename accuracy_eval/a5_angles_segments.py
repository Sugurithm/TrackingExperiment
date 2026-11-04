"""a5 関節角度とセグメント（骨長・向き）

何をするか
  3 次元座標から計算する量の誤差を見る（研究で実際に使う量）。
  - 骨長（上腕 = 肩–肘，前腕 = 肘–手首，手 = 手首–MCP3）の平均・SD・変動係数:
    剛体なので一定のはず → RS の SD は真値なしで測れる「内部の一貫性」
  - 肘の角度（肩–肘–手首のなす角）: 偏り，SD，相関 r，一致限界，変化幅の比
  - セグメントの向きの誤差: RS と MoCap の上腕・前腕ベクトルのなす角（平均と，平均を引いた後の変動）
わかること
  位置の偏りが角度・向きにどう伝わるか。角度の偏りが一定（SD が小さい）なら
  関節の動き方（角度変化・協調）の解析に使える。
"""
import numpy as np
import common as C


def angle(S, E, W):
    a = S - E; b = W - E
    c = np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def vec_angle(a, b):
    c = np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _segs(tr):
    return [("静止", tr.rest)] + [(b["name"], [(b["t0"], b["t1"])]) for b in tr.blocks] + \
           [("動作全体", [(b["t0"], b["t1"]) for b in tr.blocks])]


def run(tr):
    have = set(tr.rs_names)
    pairs = [("上腕", "shoulder", "elbow"), ("前腕(hand)", "elbow", "wrist"), ("前腕(pose)", "elbow", "wrist_pose"),
             ("手(手首–MCP3)", "wrist", "mcp3")]
    move = np.zeros(len(tr.t), bool)
    for b in tr.blocks: move |= tr.mask((b["t0"], b["t1"]))
    bones = []
    for lab, a, b in pairs:
        if not {a, b} <= have: continue
        L = np.linalg.norm(tr.rs(a, False)[move] - tr.rs(b, False)[move], axis=1); L = L[~np.isnan(L)]
        Lf = np.linalg.norm(tr.rs(a)[move] - tr.rs(b)[move], axis=1); Lf = Lf[~np.isnan(Lf)]
        row = {"bone": lab, "rs_mean": L.mean(), "rs_sd": L.std(), "rs_cv_pct": 100 * L.std() / L.mean(), "rs_sd_filt": Lf.std()}
        ra, rb = tr.ref(a), tr.ref(b)
        if ra is not None and rb is not None and C.PAIRS[a] != C.PAIRS[b]:
            M = np.linalg.norm(ra[move] - rb[move], axis=1); M = M[~np.isnan(M)]
            row.update(mc_mean=M.mean(), mc_sd=M.std(), diff=L.mean() - M.mean())
        bones.append(row)

    ang_rows = []; dir_rows = []
    ok_ang = {"shoulder", "elbow", "wrist"} <= have
    for name, segs in _segs(tr):
        mk = np.zeros(len(tr.t), bool)
        for s in segs: mk |= tr.mask(s)
        if ok_ang:
            ar = angle(tr.rs("shoulder")[mk], tr.rs("elbow")[mk], tr.rs("wrist")[mk])
            am = angle(tr.ref("shoulder")[mk], tr.ref("elbow")[mk], tr.ref("wrist")[mk])
            g = ~(np.isnan(ar) | np.isnan(am)); ar, am = ar[g], am[g]; d = ar - am
            rng_m = np.percentile(am, 97) - np.percentile(am, 3); rng_r = np.percentile(ar, 97) - np.percentile(ar, 3)
            ang_rows.append({"segment": name, "mc_mean": am.mean(), "mc_range": rng_m, "bias": d.mean(), "sd": d.std(),
                             "rmse": np.sqrt(np.mean(d ** 2)), "loa_lo": d.mean() - 1.96 * d.std(), "loa_hi": d.mean() + 1.96 * d.std(),
                             "r": np.corrcoef(ar, am)[0, 1] if rng_m > 3 else np.nan,
                             "range_ratio": rng_r / rng_m if rng_m > 3 else np.nan})
        for lab, a, b in pairs[:2]:
            if not {a, b} <= have: continue
            vr = tr.rs(b)[mk] - tr.rs(a)[mk]; vm = tr.ref(b)[mk] - tr.ref(a)[mk]
            g = ~(np.isnan(vr).any(1) | np.isnan(vm).any(1)); th = vec_angle(vr[g], vm[g])
            # 一定の回転を除いた変動: RS ベクトルを平均回転で合わせてから比較
            ur = vr[g] / np.linalg.norm(vr[g], axis=1, keepdims=True); um = vm[g] / np.linalg.norm(vm[g], axis=1, keepdims=True)
            H = ur.T @ um; U, _, Vt = np.linalg.svd(H); Dm = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
            Rot = Vt.T @ Dm @ U.T; th2 = vec_angle(ur @ Rot.T, um)
            dir_rows.append({"segment": name, "bone": lab, "mean_deg": th.mean(), "p95_deg": np.percentile(th, 95),
                             "var_deg": np.sqrt(np.mean(th2 ** 2))})
    tr.save_json("a5_angles_segments.json", {"bones": bones, "elbow_angle": ang_rows, "direction": dir_rows})

    plt = C.setup_plot()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    if ok_ang and tr.blocks:
        b = tr.blocks[-1]; mk = tr.mask((b["t0"], b["t1"])); tt = tr.t[mk] - b["t0"]
        ax[0].plot(tt, angle(tr.ref("shoulder")[mk], tr.ref("elbow")[mk], tr.ref("wrist")[mk]), color="#333", lw=2, label="MoCap")
        ax[0].plot(tt, angle(tr.rs("shoulder")[mk], tr.rs("elbow")[mk], tr.rs("wrist")[mk]), color=C.COLOR["elbow"], lw=1.5, label="RS 5 Hz")
        ax[0].set_xlim(0, 4); ax[0].set_xlabel("時間 [s]"); ax[0].set_ylabel("肘の角度 [°]"); ax[0].set_title(f"肘の角度（{b['name']}）"); ax[0].legend(fontsize=8)
        mk = np.zeros(len(tr.t), bool)
        for bb in tr.blocks: mk |= tr.mask((bb["t0"], bb["t1"]))
        ar = angle(tr.rs("shoulder")[mk], tr.rs("elbow")[mk], tr.rs("wrist")[mk]); am = angle(tr.ref("shoulder")[mk], tr.ref("elbow")[mk], tr.ref("wrist")[mk])
        g = ~(np.isnan(ar) | np.isnan(am)); ar, am = ar[g], am[g]; d = ar - am
        ax[1].plot(((ar + am) / 2)[::4], d[::4], ".", ms=2, alpha=.3, color=C.COLOR["elbow"])
        for y, ls in [(d.mean(), "-"), (d.mean() - 1.96 * d.std(), "--"), (d.mean() + 1.96 * d.std(), "--")]: ax[1].axhline(y, color="#333", ls=ls, lw=1)
        ax[1].set_xlabel("(RS + MoCap) / 2 [°]"); ax[1].set_ylabel("RS − MoCap [°]"); ax[1].set_title("肘の角度の Bland–Altman（動作全体）")
    xs = np.arange(len(bones))
    ax[2].bar(xs - .2, [r["rs_sd"] for r in bones], .4, color="#2a78d6", label="RS（生）")
    ax[2].bar(xs + .2, [r.get("mc_sd", np.nan) for r in bones], .4, color="#444", label="MoCap")
    ax[2].set_xticks(xs); ax[2].set_xticklabels([r["bone"] for r in bones], fontsize=8); ax[2].set_ylabel("骨長の SD [mm]")
    ax[2].set_title("骨長のばらつき（剛体なら 0）"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a5_angles_segments.png", dpi=120); plt.close(fig)

    md = "## a5 関節角度とセグメント\n\n骨長（動作中）\n\n"
    md += C.md_table(bones, [("骨", "bone"), ("RS 平均 [mm]", "rs_mean"), ("RS SD", "rs_sd"), ("RS CV [%]", "rs_cv_pct"),
                             ("RS SD 5Hz後", "rs_sd_filt"), ("MoCap 平均", "mc_mean"), ("MoCap SD", "mc_sd"), ("差 RS−MC", "diff")])
    if ang_rows:
        md += "\n肘の角度 [°]\n\n" + C.md_table(ang_rows, [("区間", "segment"), ("MoCap 平均", "mc_mean"), ("変化幅", "mc_range"),
                                                       ("偏り", "bias"), ("SD", "sd"), ("RMSE", "rmse"), ("一致限界 下", "loa_lo"),
                                                       ("上", "loa_hi"), ("r", "r"), ("変化幅の比", "range_ratio")], "{:.2f}")
    md += "\nセグメントの向きの誤差 [°]（平均 = 一定分を含む，変動 = 一定の回転を除いた RMS）\n\n"
    md += C.md_table([r for r in dir_rows if r["segment"] in ("静止", "動作全体")],
                     [("区間", "segment"), ("骨", "bone"), ("平均", "mean_deg"), ("95%", "p95_deg"), ("変動", "var_deg")])
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
