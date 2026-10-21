"""w プレイヤー内の解析（センシが変わると関節の動きはどう変わるか）

何をするか
  1 人ぶんの全試行（a0〜a7 の JSON）を集めて 1 行 = 1 試行の表（trials.csv）にし、
  DPI（= センシ）× 帯域 × seed で比べる。

  - **協調の指標の変化**: N_eff・R・C が DPI でどう動くか。
    変わらない = センシによらず同じ型で動く。変わる = センシに応じて使い方を変えている
  - **関節ごとの寄与の変化**: 絶対値（mm/s）とシェア。
    必要な手の移動量は DPI に反比例（400:800:1600 = 4:2:1）なので、log(寄与)/log(DPI) の傾きが
    **−1 = 必要な分だけ相似に縮める**、**0 に近い = DPI を上げても動かし続ける関節**
  - 追従の帯域・遅れ・remnant・課題成績・肘の角度の範囲も同じ表で並べる
  - **再現性**: 同一条件 3 本（seed）の SD と「条件間の分散 / 条件内の分散」。
    この比が 1 前後の指標は、条件の差として読んではいけない
わかること
  その人がセンシに対してどう適応するか。プレイヤー間の比較（b）の前提になる個人内の安定性も分かる。
"""
import csv, glob, json, os
import numpy as np
import common as C

JOINT_KEYS = ["shoulder", "elbow", "wrist", "mcp3"]
# まとめの表に出す指標（キー, 見出し, 書式）
MAIN = [
    ("N_eff", "N_eff（関与関節数）", "{:.2f}"),
    ("S", "S（同時性）", "{:.2f}"),
    ("R", "R（時間の交代・補助）", "{:+.2f}"),
    ("C", "C（相殺比）", "{:.2f}"),
    ("dominant_share", "主役のシェア", "{:.0%}"),
    ("rms_2d", "RMS 誤差 [mm]", "{:.1f}"),
    ("on_target_pct", "on-target [%]", "{:.1f}"),
    ("fb_x_hz", "追従の帯域 fb x [Hz]", "{:.2f}"),
    ("delay_x_ms", "実効遅れ x [ms]", "{:+.0f}"),
    ("remnant_x", "remnant x [%]", "{:.1%}"),
    ("mouse_range_x", "マウスの範囲 x [mm]", "{:.0f}"),
    ("total_rms_mm", "手先の変位 RMS [mm]", "{:.1f}"),
    ("pc1", "PC1 の寄与率", "{:.2f}"),
    ("n_for_90pct", "90 % に必要な成分数", "{:.1f}"),
    ("ang_elbow_range", "肘の角度の範囲 [deg]", "{:.1f}"),
    ("mouse_lag_ms", "マウスとの遅れ [ms]", "{:+.0f}"),
]


def _load(d, name):
    f = os.path.join(d, "eval", name)
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None


def collect(trial_dirs):
    """試行フォルダの一覧 → 1 行 / 試行 の辞書の一覧。"""
    rows = []
    for d in trial_dirs:
        q = _load(d, "a0_quality.json")
        if q is None:
            continue
        h = q["header"]
        r = {"trial": h["trial"], "participant": h["participant"], "dpi": h["dpi"], "kind": h["kind"],
             "band": h["band"], "seed": h["seed"], "fs_rs": h["fs_rs"],
             "mouse_lag_ms": q["mouse_match"].get("lag_ms", np.nan),
             "mouse_gain_x": q["mouse_match"].get("gain_x", np.nan),
             "mouse_resid_mm": q["mouse_match"].get("resid_at_lag_mm", np.nan),
             "lr_swapped_pct": q.get("lr_swapped_pct", np.nan)}
        for j in q["joints"]:
            r[f"valid_{j['joint']}"] = j["valid_pct"]
            r[f"jit_{j['joint']}"] = j.get("jit_sd3d", np.nan)
        snr = q.get("snr", [])
        if snr:
            r["snr_ok_pct"] = 100 * np.mean([s["snr"] > C.SNR_MIN for s in snr])
        # --- 核の指標
        i1 = _load(d, "a1_indices.json")
        if i1:
            r.update({"N_eff": i1["N_eff"], "N_max": i1["N_max"], "S": i1.get("S", np.nan),
                      "C": i1["C"], "dominant": i1["dominant"],
                      "dominant_share": i1["dominant_share"],
                      "R": i1["R"] if i1.get("R_reliable") else np.nan,
                      "R_raw": i1.get("R", np.nan), "R_reliable": i1.get("R_reliable"),
                      "R_n_active": i1.get("R_n_active", 0)})
            for p, v in i1.get("snr_time", {}).items():
                r[f"isnrt_{p}"] = v
            for p, v in i1["share"].items():
                r[f"ishare_{p}"] = v
            for p, v in i1["snr_part"].items():
                r[f"isnr_{p}"] = v
        t4 = _load(d, "a2_contribution.json")
        if t4:
            r["total_rms_mm"] = t4["summary"]["total_rms_mm"]
            r["style_distal_share"] = t4["summary"]["style_distal_share"]
            r["rms_sum_ratio"] = t4["summary"]["rms_sum_ratio"]
            for p in t4["parts"]:
                r[f"share_{p['part']}"] = p["share"]; r[f"rms_{p['part']}"] = p["rms_mm"]
        t5 = _load(d, "a3_coupling.json")
        if t5:
            if t5.get("pca"):
                r["pc1"] = t5["pca"]["pc1"]; r["n_for_90pct"] = t5["pca"]["n_for_90pct"]
            for p in t5["pairs"]:
                r[f"lag_{p['a']}_{p['b']}_ms"] = p["lag_speed_ms"]
            for j in t5["joints"]:
                r[f"disp_{j['joint']}_mm"] = j["rms_mm"]
        t3 = _load(d, "a4_joint_freq.json")
        if t3:
            for s in t3["summary"]:
                if s["axis"] == "x":
                    r[f"cutoff_{s['joint']}_hz"] = s["cutoff_hz"]
                    r[f"fb_{s['joint']}_hz"] = s.get("fb_fit_hz", np.nan)
                    r[f"jdelay_{s['joint']}_ms"] = s["delay_ms"]
        t2 = _load(d, "a5_describing.json")
        if t2:
            for ax in C.AX:
                s = t2["summary"][ax]
                r[f"delay_{ax}_ms"] = s["delay_ms"]; r[f"crossover_{ax}_hz"] = s["crossover_hz"]
                r[f"bandwidth_{ax}_hz"] = s.get("bandwidth_hz", np.nan)
                r[f"fb_{ax}_hz"] = s.get("fb_fit_hz", np.nan)
                r[f"remnant_{ax}"] = s["remnant"]
        t1 = _load(d, "a6_task.json")
        if t1:
            r.update({"rms_2d": t1["rms_2d"], "on_target_pct": t1["on_target_pct"],
                      "rms_first": t1["rms_first_half"], "rms_second": t1["rms_second_half"],
                      "mouse_range_x": t1["mouse_range_x_mm"], "mouse_path_mm": t1["mouse_path_mm"]})
        t6 = _load(d, "a7_angles.json")
        if t6:
            for s in t6["segments"]:
                r[f"len_{s['seg']}_mm"] = s["len_mm"]; r[f"lensd_{s['seg']}"] = s["len_sd_mm"]
                r[f"azrange_{s['seg']}"] = s["az_range_deg"]
            for k, v in t6.get("angles", {}).items():
                r[f"ang_{k}_range"] = v["range_deg"]; r[f"ang_{k}_sd"] = v["sd_deg"]
        rows.append(r)
    return rows


def _vals(rows, field, **where):
    out = []
    for r in rows:
        if any(r.get(k) != v for k, v in where.items()):
            continue
        v = r.get(field, np.nan)
        if isinstance(v, (int, float)) and np.isfinite(v):
            out.append(float(v))
    return np.array(out)


def _agg(rows, keys, field):
    out = {}
    for r in rows:
        k = tuple(r.get(x) for x in keys)
        v = r.get(field, np.nan)
        if isinstance(v, (int, float)) and np.isfinite(v):
            out.setdefault(k, []).append(float(v))
    res = []
    for k, v in sorted(out.items(), key=lambda kv: [str(x) for x in kv[0]]):
        v = np.array(v)
        res.append({**{x: k[i] for i, x in enumerate(keys)}, "n": len(v), "mean": float(v.mean()),
                    "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan})
    return res


def _fratio(rows, field, keys=("dpi", "band")):
    """条件間の分散 / 条件内の分散。"""
    g = {}
    for r in rows:
        v = r.get(field, np.nan)
        if isinstance(v, (int, float)) and np.isfinite(v):
            g.setdefault(tuple(r.get(k) for k in keys), []).append(float(v))
    g = {k: np.array(v) for k, v in g.items() if len(v) >= 2}
    if len(g) < 2:
        return np.nan
    within = np.mean([v.var(ddof=1) for v in g.values()])
    between = np.var([v.mean() for v in g.values()], ddof=1)
    return float(between / within) if within > 0 else np.nan


def dpi_slope(rows, field, band=None):
    """log(指標) を log(DPI) に回帰した傾き。"""
    x, y = [], []
    for r in rows:
        if band and r.get("band") != band:
            continue
        v = r.get(field, np.nan); d = r.get("dpi", np.nan)
        if isinstance(v, (int, float)) and np.isfinite(v) and v > 0 and np.isfinite(d):
            x.append(np.log(d)); y.append(np.log(v))
    if len(set(x)) < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def run(rows, out, title=""):
    track = [r for r in rows if r.get("kind") == "track"]
    if not track:
        return "## w プレイヤー内の解析\n\n本番の試行がありません。\n"
    os.makedirs(out, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    head = ["trial", "participant", "dpi", "band", "seed"]
    with open(os.path.join(out, "trials.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=head + [k for k in keys if k not in head])
        w.writeheader(); w.writerows(rows)

    dpis = sorted({r["dpi"] for r in track})
    bands = sorted({r["band"] for r in track})
    md = f"## w プレイヤー内の解析{('（' + title + '）') if title else ''}\n\n"
    md += (f"本番 {len(track)} 試行（DPI {', '.join(f'{d:g}' for d in dpis)} × "
           f"{', '.join(bands)} × seed {len({r['seed'] for r in track})} 本）。\n\n")

    # --- 1. 指標の条件別
    md += "### 1. 条件ごとの指標（平均 ± 同一条件 3 本の SD）\n\n"
    tab = []
    for f, lab, fmt in MAIN:
        row = {"metric": lab}
        for d in dpis:
            for b in bands:
                v = _vals(track, f, dpi=d, band=b)
                row[f"{d:g}_{b}"] = ((fmt.format(v.mean()) + (f" ± {v.std(ddof=1):.2g}" if len(v) > 1 else ""))
                                     if len(v) else "–")
        fr = _fratio(track, f)
        row["F"] = f"{fr:.1f}" if np.isfinite(fr) else "–"
        row["slope"] = f"{dpi_slope(track, f):+.2f}" if np.isfinite(dpi_slope(track, f)) else "–"
        tab.append(row)
    cols = ([("指標", "metric")] + [(f"{d:g} / {b}", f"{d:g}_{b}") for d in dpis for b in bands]
            + [("条件間/条件内", "F"), ("log-log 傾き", "slope")])
    md += C.md_table(tab, cols)
    md += ("\n「条件間/条件内」が 1 前後の指標は、条件の差ではなく試行ごとのばらつきです。"
           "「log-log 傾き」は log(指標) を log(DPI) に回帰したもの（指標が正の値のときだけ）。\n\n")

    # --- 2. 関節ごとの寄与と DPI
    md += "### 2. 関節ごとの寄与と DPI\n\n"
    jr = []
    for p in C.PARTS:
        if f"ishare_{p}" not in keys:
            continue
        row = {"label": C.PARTLABEL[p]}
        for d in dpis:
            row[f"sh{d:g}"] = float(np.mean(_vals(track, f"ishare_{p}", dpi=d))) if len(
                _vals(track, f"ishare_{p}", dpi=d)) else np.nan
            row[f"snr{d:g}"] = float(np.mean(_vals(track, f"isnr_{p}", dpi=d))) if len(
                _vals(track, f"isnr_{p}", dpi=d)) else np.nan
        row["slope"] = dpi_slope(track, f"rms_{p}")
        jr.append(row)
    md += C.md_table(jr, [("成分", "label")] +
                     [(f"シェア {d:g}", f"sh{d:g}", "{:.0%}") for d in dpis] +
                     [(f"SNR {d:g}", f"snr{d:g}", "{:.0f}") for d in dpis] +
                     [("log(寄与)/log(DPI)", "slope", "{:+.2f}")])
    md += ("\n必要な手の移動量は DPI に反比例（傾き −1）。"
           "**0 に近い成分 = DPI を上げても動かし続けている**。"
           "SNR が 2 を下回る条件のシェアは参考値。\n\n")

    jr2 = []
    for j in JOINT_KEYS:
        if f"disp_{j}_mm" not in keys:
            continue
        row = {"label": C.LABEL.get(j, j), "slope": dpi_slope(track, f"disp_{j}_mm")}
        for d in dpis:
            v = _vals(track, f"disp_{j}_mm", dpi=d)
            row[f"d{d:g}"] = float(v.mean()) if len(v) else np.nan
        v = _vals(track, f"jit_{j}")
        row["jit"] = float(v.mean()) if len(v) else np.nan
        jr2.append(row)
    md += "関節そのものの変位（参考。ジッタの床と比べて読む）\n\n"
    md += C.md_table(jr2, [("関節", "label")] + [(f"{d:g} DPI [mm]", f"d{d:g}") for d in dpis] +
                     [("log-log 傾き", "slope", "{:+.2f}"), ("ジッタ σ [mm]", "jit", "{:.1f}")])

    # --- 図
    plt = C.setup_plot()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    mk = {b: m for b, m in zip(bands, "os^v")}

    def scat(a, field, ylab, hline=None):
        for b in bands:
            xs = [r["dpi"] for r in track if r["band"] == b and np.isfinite(r.get(field, np.nan))]
            ys = [r[field] for r in track if r["band"] == b and np.isfinite(r.get(field, np.nan))]
            a.plot(xs, ys, mk[b], ms=6, alpha=.55, label=b)
            m = _agg([r for r in track if r["band"] == b], ["dpi"], field)
            a.plot([x["dpi"] for x in m], [x["mean"] for x in m], "-", lw=1.6)
        if hline is not None:
            a.axhline(hline, color="#888", ls="--", lw=1)
        a.set_xscale("log"); a.set_xticks(dpis); a.set_xticklabels([f"{d:g}" for d in dpis])
        a.set_xlabel("DPI（センシ）"); a.set_ylabel(ylab); a.legend(fontsize=8)

    scat(axes[0, 0], "N_eff", "N_eff")
    axes[0, 0].set_title("関与関節数（1 = 一関節依存）")
    scat(axes[0, 1], "S", "S", hline=1)
    axes[0, 1].set_title("同時性（1 = 同時，<1 = 一つずつ交代）")
    scat(axes[0, 2], "C", "C", hline=1)
    axes[0, 2].set_title("相殺比（>1 = 打ち消し合い）")
    # シェアの積み上げ
    a = axes[1, 0]
    cols2 = [(d, b) for d in dpis for b in bands]
    xs = np.arange(len(cols2)); bottoms = np.zeros(len(cols2))
    for p in C.PARTS:
        if f"ishare_{p}" not in keys:
            continue
        v = np.array([np.mean(_vals(track, f"ishare_{p}", dpi=d, band=b)) if len(
            _vals(track, f"ishare_{p}", dpi=d, band=b)) else 0.0 for d, b in cols2])
        a.bar(xs, v, bottom=bottoms, color=C.PARTCOLOR[p], label=C.PARTLABEL[p], width=.7)
        bottoms += v
    a.set_xticks(xs); a.set_xticklabels([f"{d:g}\n{b}" for d, b in cols2], fontsize=8)
    a.set_ylabel("寄与シェア"); a.set_title("成分ごとの寄与"); a.legend(fontsize=7)
    # 寄与の絶対値（log-log）
    a = axes[1, 1]
    for p in C.PARTS:
        if f"rms_{p}" not in keys:
            continue
        m = _agg(track, ["dpi"], f"rms_{p}")
        if m:
            a.plot([x["dpi"] for x in m], [x["mean"] for x in m], "o-", color=C.PARTCOLOR[p],
                   label=C.PARTLABEL[p])
    ref = _agg(track, ["dpi"], "total_rms_mm")
    if ref:
        a.plot([x["dpi"] for x in ref], [ref[0]["mean"] * dpis[0] / x["dpi"] for x in ref],
               "k--", lw=1, label="DPI に反比例")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xticks(dpis)
    a.set_xticklabels([f"{d:g}" for d in dpis])
    a.set_xlabel("DPI"); a.set_ylabel("寄与 RMS [mm]"); a.set_title("寄与の絶対値"); a.legend(fontsize=7)
    # 成績
    scat(axes[1, 2], "rms_2d", "RMS 誤差 [mm]")
    axes[1, 2].set_title("課題の成績")
    fig.suptitle(f"プレイヤー内（センシの効果）{(' — ' + title) if title else ''}", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "w_within.png"), dpi=120); plt.close(fig)

    json.dump({"rows": rows}, open(os.path.join(out, "w_within.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=C._jsonable)
    return md
