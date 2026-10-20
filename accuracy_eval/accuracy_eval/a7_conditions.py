"""a7 試行条件ごとの誤差（区間・速さ・ストロークの位置・ブロック開始からの時間）

何をするか
  A. 区間ごと（meta.json の cue: ready / board / away / still / count / bpmNN / rest / tail。なければ自動検出）に
     有効率・大外れ率を「原因別」（左右の取り違え／それ以外の誤検出）に分け，大外れを除いた誤差を出す。
     → 区間全体の RMSE が数百 mm になる原因が，速さなのか誤検出なのかを切り分ける。
  B. 動作中（bpmNN）のフレームを条件で層別し，「一定の偏りを除いた誤差」の RMS を比べる。
     - 瞬間の速さ（その関節の MoCap の速さ）: 0–0.1, 0.1–0.3, …, >1.1 m/s
     - 動きの方向の誤差（+ = RS が先行，− = 遅れ）と速さの関係 → 傾きから「実効的な遅れ [ms]」
       （遅れ τ があると 動きの方向の誤差 ≈ −τ·v）
     - ストロークの位置: 中央（速い）〜 折り返し（止まって向きが変わる）
     - ブロック開始からの時間（最初の数秒だけ悪い＝追従の立ち上がり）
  フレームごとの条件と誤差を a7_frames.csv に書くので，別の条件でも自分で層別できる。
わかること
  誤差が「試行のどの条件で」大きくなるか。速さで増える成分（遅れ・ブレ）と，
  速さと無関係な成分（誤検出・左右の取り違え・一定の偏り）を分けて報告できる。
"""
import csv
import numpy as np
import common as C

SPEED_BINS = [0, .1, .3, .5, .7, .9, 1.1, 9]      # m/s
PHASE_BINS = [0, .2, .4, .6, .8, 1.01]            # |位置 − 中心| / 片振幅（0 = 中央，1 = 折り返し）
TIME_BINS = [0, 1, 2, 4, 1e9]                     # ブロック開始からの秒


def _order(names):
    def key(n):
        base = ["ready", "board", "away", "still", "count"]
        if n.startswith("bpm"): return (5, int(n[3:]) if n[3:].isdigit() else 0)
        if n in base: return (base.index(n), 0)
        return (9, 0)
    return sorted(names, key=key)


def run(tr):
    J = tr.compared(); lab = tr.labels
    task = np.array([str(x).startswith("bpm") for x in lab])
    # 動作中の一定の偏り（関節ごと）
    E = {}; bias = {}
    for n in J:
        E[n] = tr.rs(n) - tr.ref(n)
        ok = task & ~np.isnan(E[n]).any(1)
        bias[n] = E[n][ok].mean(0)

    # ---------------- A. 区間ごと（原因別）
    seg_rows = []
    for name in _order(set(map(str, lab))):
        mk = lab == name
        for n in J:
            j = tr.rs_names.index(n)
            valid = ~np.isnan(tr.P_raw[mk, j, 0])
            g = tr.gross[n][mk]; glr = tr.gross_lr[n][mk]
            Ev = E[n][mk] - bias[n]; ok = ~np.isnan(Ev).any(1)
            Eraw = tr.P_raw[mk, j] - tr.ref(n, False)[mk]; okr = ~np.isnan(Eraw).any(1)
            seg_rows.append({"segment": name, "joint": n, "label": C.LABEL[n], "n": int(mk.sum()),
                             "valid_pct": 100 * valid.mean(), "gross_lr_pct": 100 * glr.mean(),
                             "gross_other_pct": 100 * (g & ~glr).mean(),
                             "rmse_all_raw": float(np.sqrt(np.mean(np.sum(Eraw[okr] ** 2, 1)))) if okr.any() else np.nan,
                             "rmse_clean": float(np.sqrt(np.mean(np.sum(Ev[ok] ** 2, 1)))) if ok.sum() > 30 else np.nan})

    # ---------------- B. 動作中の層別
    # ストロークの位置（ブロックごとに MoCap 手首の主な方向）
    phase = np.full(len(tr.t), np.nan); tsb = np.full(len(tr.t), np.nan); blk = np.full(len(tr.t), "", dtype=object)
    for name in set(lab[task]):
        mk = lab == name
        idx = np.flatnonzero(mk)
        W = tr.M["wrist"][mk]; okw = ~np.isnan(W).any(1)
        if okw.sum() < 30: continue
        c = W[okw].mean(0); ax = np.linalg.svd(W[okw] - c, full_matrices=False)[2][0]
        s = (W - c) @ ax; A = (np.nanpercentile(s, 98) - np.nanpercentile(s, 2)) / 2
        phase[mk] = np.clip(np.abs(s - (np.nanpercentile(s, 98) + np.nanpercentile(s, 2)) / 2) / A, 0, 1)
        tsb[mk] = tr.t[mk] - tr.t[idx[0]]; blk[mk] = name
    rows_speed = []; rows_phase = []; rows_time = []; lag_rows = []; frames = {}
    for n in J:
        R = tr.ref(n); v = C.speed(np.nan_to_num(R), tr.fs) / 1000.0
        vel = np.gradient(np.nan_to_num(R), axis=0) * tr.fs
        vhat = vel / np.maximum(np.linalg.norm(vel, axis=1, keepdims=True), 1e-9)
        Ev = E[n] - bias[n]
        along = np.sum(Ev * vhat, 1); perp = np.linalg.norm(Ev - along[:, None] * vhat, axis=1)
        e3 = np.linalg.norm(Ev, axis=1)
        ok = task & ~np.isnan(e3)
        frames[n] = dict(v=v, along=along, perp=perp, e3=e3)
        for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
            m = ok & (v >= lo) & (v < hi)
            if m.sum() < 30: continue
            rows_speed.append({"joint": n, "label": C.LABEL[n], "bin": f"{lo:.1f}–{hi:.1f}" if hi < 9 else f">{lo:.1f}",
                               "v_mid": float(np.median(v[m])), "n": int(m.sum()), "rms3d": float(np.sqrt(np.mean(e3[m] ** 2))),
                               "along_mean": float(along[m].mean()), "perp_rms": float(np.sqrt(np.mean(perp[m] ** 2)))})
        for lo, hi in zip(PHASE_BINS[:-1], PHASE_BINS[1:]):
            m = ok & (phase >= lo) & (phase < hi)
            if m.sum() < 30: continue
            rows_phase.append({"joint": n, "label": C.LABEL[n], "bin": f"{lo:.1f}–{min(hi, 1):.1f}", "mid": (lo + min(hi, 1)) / 2,
                               "n": int(m.sum()), "rms3d": float(np.sqrt(np.mean(e3[m] ** 2))),
                               "speed": float(np.median(v[m]))})
        for lo, hi in zip(TIME_BINS[:-1], TIME_BINS[1:]):
            m = ok & (tsb >= lo) & (tsb < hi)
            if m.sum() < 30: continue
            rows_time.append({"joint": n, "label": C.LABEL[n], "bin": f"{lo:.0f}–{hi:.0f} s" if hi < 1e8 else f"{lo:.0f} s–",
                              "n": int(m.sum()), "rms3d": float(np.sqrt(np.mean(e3[m] ** 2)))})
        m = ok & (v > 0.05)
        if m.sum() > 100 and np.std(v[m]) > 0.05:
            A_ = np.column_stack([np.ones(m.sum()), v[m] * 1000]); coef, *_ = np.linalg.lstsq(A_, along[m], rcond=None)
            r = np.corrcoef(v[m], along[m])[0, 1]
            # 速さで増える 3D 誤差: RMS² = a + (b·v)²
            B_ = np.column_stack([np.ones(m.sum()), (v[m]) ** 2]); cb, *_ = np.linalg.lstsq(B_, e3[m] ** 2, rcond=None)
            lag_rows.append({"joint": n, "label": C.LABEL[n], "lag_eff_ms": float(-coef[1] * 1000), "r": float(r),
                             "err_at_0": float(np.sqrt(max(cb[0], 0))), "err_at_1mps": float(np.sqrt(max(cb[0] + cb[1], 0))),
                             "vmax": float(np.percentile(v[m], 99))})
    tr.save_json("a7_conditions.json", {"label_source": tr.label_source, "segments": seg_rows, "speed": rows_speed,
                                        "phase": rows_phase, "time_since_start": rows_time, "lag": lag_rows})
    with open(f"{tr.out}/a7_frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["take_s", "segment", "block_t_s", "stroke_pos"] + [f"{n}_{k}" for n in J for k in ("speed_mps", "err3d_mm", "along_mm", "perp_mm", "gross", "gross_lr")])
        for i in range(len(tr.t)):
            row = [f"{tr.t[i]:.4f}", lab[i], f"{tsb[i]:.3f}", f"{phase[i]:.3f}"]
            for n in J:
                fr = frames[n]
                row += [f"{fr['v'][i]:.4f}", f"{fr['e3'][i]:.2f}", f"{fr['along'][i]:.2f}", f"{fr['perp'][i]:.2f}",
                        int(tr.gross[n][i]), int(tr.gross_lr[n][i])]
            w.writerow(row)

    # ---------------- 図
    plt = C.setup_plot()
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    segs = [s for s in _order(set(map(str, lab)))]
    xs = np.arange(len(segs)); w = 0.8 / len(J)
    for k, n in enumerate(J):
        rr = {r["segment"]: r for r in seg_rows if r["joint"] == n}
        lr = np.array([rr[s]["gross_lr_pct"] for s in segs]); ot = np.array([rr[s]["gross_other_pct"] for s in segs])
        inv = np.array([100 - rr[s]["valid_pct"] for s in segs])
        x = xs + (k - (len(J) - 1) / 2) * w
        ax[0, 0].bar(x, inv, w, color="#bbb", label="欠損" if k == 0 else None)
        ax[0, 0].bar(x, lr, w, bottom=inv, color="#d03b3b", label="左右の取り違え" if k == 0 else None)
        ax[0, 0].bar(x, ot, w, bottom=inv + lr, color="#eda100", label="その他の誤検出" if k == 0 else None)
    ax[0, 0].set_xticks(xs); ax[0, 0].set_xticklabels(segs, rotation=45, fontsize=8); ax[0, 0].set_ylabel("[% of frames]")
    ax[0, 0].set_title("区間ごとの「使えないフレーム」の原因（関節ごとの棒）"); ax[0, 0].legend(fontsize=8)
    for n in J:
        rr = {r["segment"]: r for r in seg_rows if r["joint"] == n}
        ax[0, 1].plot(xs, [rr[s]["rmse_clean"] for s in segs], "-o", color=C.COLOR[n], label=C.LABEL[n], lw=2, ms=5)
    ax[0, 1].set_xticks(xs); ax[0, 1].set_xticklabels(segs, rotation=45, fontsize=8); ax[0, 1].set_ylabel("[mm]")
    ax[0, 1].set_title("誤検出を除き，一定の偏りを引いた RMSE"); ax[0, 1].legend(fontsize=8); ax[0, 1].set_ylim(bottom=0)
    for n in J:
        rr = [r for r in rows_speed if r["joint"] == n]
        ax[0, 2].plot([r["v_mid"] for r in rr], [r["rms3d"] for r in rr], "-o", color=C.COLOR[n], label=C.LABEL[n], lw=2, ms=5)
    ax[0, 2].set_xlabel("その関節の瞬間の速さ [m/s]"); ax[0, 2].set_ylabel("RMS [mm]"); ax[0, 2].set_title("速さ別の誤差（動作中，偏り除去後）")
    ax[0, 2].legend(fontsize=8); ax[0, 2].set_ylim(bottom=0)
    for n in J:
        rr = [r for r in rows_speed if r["joint"] == n]
        ax[1, 0].plot([r["v_mid"] for r in rr], [r["along_mean"] for r in rr], "-o", color=C.COLOR[n], label=C.LABEL[n], lw=2, ms=5)
    ax[1, 0].axhline(0, color="#888", lw=1)
    ax[1, 0].set_xlabel("瞬間の速さ [m/s]"); ax[1, 0].set_ylabel("動きの方向の誤差の平均 [mm]（− = RS が遅れ）")
    ax[1, 0].set_title("遅れによる誤差（傾き = 実効的な遅れ）"); ax[1, 0].legend(fontsize=8)
    for n in J:
        rr = [r for r in rows_phase if r["joint"] == n]
        ax[1, 1].plot([r["mid"] for r in rr], [r["rms3d"] for r in rr], "-o", color=C.COLOR[n], label=C.LABEL[n], lw=2, ms=5)
    ax[1, 1].set_xlabel("ストロークの位置（0 = 中央・速い，1 = 折り返し）"); ax[1, 1].set_ylabel("RMS [mm]")
    ax[1, 1].set_title("ストロークの位置別の誤差"); ax[1, 1].legend(fontsize=8); ax[1, 1].set_ylim(bottom=0)
    tb = [r["bin"] for r in rows_time if r["joint"] == J[0]]
    for n in J:
        rr = [r for r in rows_time if r["joint"] == n]
        ax[1, 2].plot(range(len(rr)), [r["rms3d"] for r in rr], "-o", color=C.COLOR[n], label=C.LABEL[n], lw=2, ms=5)
    ax[1, 2].set_xticks(range(len(tb))); ax[1, 2].set_xticklabels(tb); ax[1, 2].set_xlabel("ブロック開始からの時間")
    ax[1, 2].set_ylabel("RMS [mm]"); ax[1, 2].set_title("ブロックの始めだけ悪いか"); ax[1, 2].legend(fontsize=8); ax[1, 2].set_ylim(bottom=0)
    fig.suptitle(f"試行条件ごとの誤差（区間: {tr.label_source}）", y=1.0)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a7_conditions.png", dpi=120); plt.close(fig)

    # ---------------- 表
    md = f"## a7 試行条件ごとの誤差\n\n区間の定義: {tr.label_source}\n\n"
    md += "区間ごと（大外れの原因別。RMSE(全部) は除外なしの生データ，RMSE(除外後) は誤検出を除き一定の偏りを引いた値）\n\n"
    tab = []
    for s in segs:
        d = {"segment": s}
        for n in J:
            r = next(x for x in seg_rows if x["segment"] == s and x["joint"] == n)
            d[n + "_all"] = r["rmse_all_raw"]; d[n + "_clean"] = r["rmse_clean"]
            d[n + "_lr"] = r["gross_lr_pct"]; d[n + "_ot"] = r["gross_other_pct"]
        tab.append(d)
    cols = [("区間", "segment")]
    for n in J:
        cols += [(f"{C.LABEL[n]} RMSE(全部)", n + "_all", "{:.0f}"), ("除外後", n + "_clean", "{:.1f}"),
                 ("取り違え%", n + "_lr", "{:.1f}"), ("他の誤検出%", n + "_ot", "{:.1f}")]
    md += C.md_table(tab, cols)
    md += "\n速さ別（動作中，偏り除去後の 3D RMS [mm]）\n\n"
    bins = []
    for r in rows_speed:
        if r["bin"] not in bins: bins.append(r["bin"])
    tab = []
    for n in J:
        d = {"label": C.LABEL[n]}
        for r in rows_speed:
            if r["joint"] == n: d[r["bin"]] = r["rms3d"]
        tab.append(d)
    md += C.md_table(tab, [("関節", "label")] + [(b + " m/s", b) for b in bins])
    md += "\n速さとの関係（動きの方向の誤差 ≈ −遅れ × 速さ，3D 誤差² ≈ a + (b·速さ)²）\n\n"
    md += C.md_table(lag_rows, [("関節", "label"), ("実効的な遅れ [ms]", "lag_eff_ms", "{:+.1f}"), ("r", "r", "{:+.2f}"),
                                ("止まっているときの誤差 [mm]", "err_at_0"), ("1 m/s のときの誤差 [mm]", "err_at_1mps"),
                                ("最大の速さ [m/s]", "vmax", "{:.2f}")])
    md += "\nストロークの位置別（0 = 中央，1 = 折り返し）の 3D RMS [mm]\n\n"
    tab = []
    pb = []
    for r in rows_phase:
        if r["bin"] not in pb: pb.append(r["bin"])
    for n in J:
        d = {"label": C.LABEL[n]}
        for r in rows_phase:
            if r["joint"] == n: d[r["bin"]] = r["rms3d"]
        tab.append(d)
    md += C.md_table(tab, [("関節", "label")] + [(b, b) for b in pb])
    md += "\nブロック開始からの時間別の 3D RMS [mm]\n\n"
    tab = []
    for n in J:
        d = {"label": C.LABEL[n]}
        for r in rows_time:
            if r["joint"] == n: d[r["bin"]] = r["rms3d"]
        tab.append(d)
    md += C.md_table(tab, [("関節", "label")] + [(b, b) for b in tb])
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
