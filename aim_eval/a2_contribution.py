"""c4 寄与の分解（どの関節が手先を動かしているか）

何をするか
  手先（手首、MCP3 があれば手先も）の変位を、足し算で厳密に分ける:
      手首 = 肩 + (肘 − 肩) + (手首 − 肘)
    → 変位 = 肩の並進（体幹）+ 上腕の回転（肩関節）+ 前腕の回転（肘関節）[+ 手の回転（手首）]
  ヤコビアンもモデルも要らず、3 次元座標だけで決まる（合計は必ず実際の手先変位に一致する）。
  - 成分ごとの RMS [mm] と**寄与率**（合計への射影。和が 1 になる）、軸別（x = 左右、y = 奥行き、z = 上下）
  - **入力周波数ごとの寄与率**: 低周波は体幹・上腕、高周波は前腕・手、という分業を 1 枚の図で
  - 成分どうしの打ち消し合い（寄与率の和が 1 でも、各成分の RMS の和が合計を大きく超えるなら互いに相殺している
    = 冗長な関節が補償し合っている証拠）
わかること
  「手首エイム / 腕エイム」を数値にしたもの。**スタイル指標** = 前腕 + 手の寄与 / 全体。
  DPI を上げて手の移動量が 1/4 になったとき、どの成分が減ってどれが残るかで関節の使い分けが見える。
"""
import numpy as np
import common as C


def run(tr):
    if not tr.has_rs:
        return "## c4 寄与の分解\n\n関節データなしなので省略。\n"
    end = "mcp3" if "mcp3" in tr.rs_names else "wrist"
    parts = C.contributions(tr, end=end)
    if parts is None:
        return "## c4 寄与の分解\n\n必要な関節（肩・肘・手首）が揃っていないので省略。\n"
    total = parts["_total"]; ok = parts["_ok"]
    comp = {k: v for k, v in parts.items() if not k.startswith("_")}
    vs = C.var_share(comp, total)

    rows = []
    for k in C.PARTS:
        if k not in vs:
            continue
        rows.append({"part": k, "label": C.PARTLABEL[k], **vs[k]})
    style = sum(vs[k]["share"] for k in ("forearm", "hand") if k in vs)
    sum_rms = sum(vs[k]["rms_mm"] for k in comp)
    res = {"end": end, "total_rms_mm": vs["_total_rms_mm"], "n": vs["_n"],
           "style_distal_share": float(style), "rms_sum_ratio": float(sum_rms / vs["_total_rms_mm"])}

    # --- 入力周波数ごとの寄与率
    freq_rows = []
    if tr.kind == "track" and tr.traj:
        for k, ax in enumerate(C.AX):
            nb, f, _ = tr.freqs(ax)
            sh = C.share_at_bins({p: comp[p][:, k] for p in comp}, total[:, k], tr.T, nb)
            amp_tot = np.abs(C.at_bins(*C.spectrum(total[:, k], tr.T), nb))
            noise = C.bin_noise(C.spectrum(total[:, k], tr.T)[1], nb)
            for i in range(len(nb)):
                r = {"axis": ax, "n": int(nb[i]), "f_hz": float(f[i]),
                     "amp_total_mm": float(amp_tot[i]), "snr": float(amp_tot[i] / noise[i]),
                     "ok": bool(amp_tot[i] / noise[i] > C.SNR_MIN)}
                for p in comp:
                    r[p] = float(sh[p][i]); r[p + "_amp"] = float(sh[p + "_amp"][i])
                freq_rows.append(r)
    tr.save_json("a2_contribution.json", {"summary": res, "parts": rows, "freq": freq_rows})

    plt = C.setup_plot()
    nc = 2 if freq_rows else 1
    fig, axes = plt.subplots(1, 1 + nc, figsize=(5.5 * (1 + nc), 4.3))
    axes = np.atleast_1d(axes)
    xs = np.arange(len(rows)); w = .25
    for i, (key, lab) in enumerate([("rms_x", "x（左右）"), ("rms_y", "y（奥行き）"), ("rms_z", "z（上下）")]):
        axes[0].bar(xs + (i - 1) * w, [r[key] for r in rows], w, label=lab,
                    color=["#2a78d6", "#1baf7a", "#eda100"][i])
    axes[0].set_xticks(xs); axes[0].set_xticklabels([C.PARTLABEL[r["part"]] for r in rows], fontsize=8, rotation=12)
    axes[0].set_ylabel("RMS [mm]"); axes[0].legend(fontsize=8)
    axes[0].set_title(f"成分ごとの大きさ（合計 {vs['_total_rms_mm']:.0f} mm，遠位の寄与 {style:.0%}）")
    if freq_rows:
        for k, ax in enumerate(C.AX):
            a = axes[1 + k]
            s = [r for r in freq_rows if r["axis"] == ax]
            f = [r["f_hz"] for r in s]
            bottom = np.zeros(len(s))
            for p in C.PARTS:
                if p not in comp:
                    continue
                y = np.clip([r[p] for r in s], 0, None)
                a.bar(range(len(s)), y, bottom=bottom, color=C.PARTCOLOR[p], label=C.PARTLABEL[p], width=.7)
                bottom += y
            a.set_xticks(range(len(s)))
            a.set_xticklabels([f"{x:.2f}" for x in f], fontsize=7, rotation=45)
            a.set_xlabel("周波数 [Hz]"); a.set_ylabel("寄与率")
            a.set_title(f"{ax} 軸：周波数ごとの寄与率")
            for i, r in enumerate(s):
                if not r["ok"]:
                    a.text(i, 0.02, "×", ha="center", color="#c00", fontsize=8)
        axes[1].legend(fontsize=7)
    fig.suptitle(f"{tr.name}（{tr.tag()}，DPI {tr.dpi:g}）寄与の分解（手先 = {end}）", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a2_contribution.png", dpi=120); plt.close(fig)

    md = "## c4 寄与の分解\n\n"
    md += (f"手先 = {C.LABEL[end]}、合計の変位 RMS {vs['_total_rms_mm']:.1f} mm（{vs['_n']} フレーム）。"
           f"**遠位（前腕 + 手）の寄与率 {style:.0%}**。"
           f"各成分の RMS の和 / 合計 = {res['rms_sum_ratio']:.2f}"
           "（1 より大きいほど成分どうしが打ち消し合っている）\n\n")
    md += C.md_table(rows, [("成分", "label"), ("RMS [mm]", "rms_mm"), ("寄与率", "share", "{:.0%}"),
                            ("x 成分 [mm]", "rms_x"), ("y 成分", "rms_y"), ("z 成分", "rms_z")])
    if freq_rows:
        md += "\n周波数ごとの寄与率（× = SNR 不足）\n\n"
        cols = [("軸", "axis"), ("f [Hz]", "f_hz", "{:.3f}"), ("合計振幅 [mm]", "amp_total_mm")]
        cols += [(C.PARTLABEL[p], p, "{:.0%}") for p in C.PARTS if p in comp]
        cols += [("SNR", "snr", "{:.1f}")]
        md += C.md_table(freq_rows, cols)
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
