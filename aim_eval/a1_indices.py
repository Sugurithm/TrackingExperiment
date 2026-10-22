"""a1 協調の指標（この研究の核）

何をするか
  手先の速度を 4 成分（肩の並進・上腕の回転・前腕の回転・手の回転）に足し算で厳密に分け、
  3 つの数で「上肢動作がどう組み立てられているか」を表す。

    N_eff（関与関節数）= 1 / Σp²     p = 各成分の寄与シェア
        1 に近い … どこかの関節に依存している
        4 に近い … 全体が動いている
    S（同時性）= 入力周波数ごとの N_eff の平均 / 同じビンから作った全体の N_eff
        1 に近い（> 0.95） … どのビンも同じ分担（全体が同時に動いている）
        小さい（< 0.85）   … ビンごとに担当が替わる（関節が一つずつ交代で動いている）
        合成データでは 同時 1.00 / 交代 0.58。分子と分母は必ず同じビンから作る
        （片方だけ SNR で絞ると、絞ったビンのパワーが分母に残って S が偏る）
    C（相殺比）= 入力周波数ごとの Σ|成分| / |合計|
        1  … 成分がそのまま足し合わさっている
        >1 … 互いに打ち消しながら手先を作っている（協調し合っている証拠）
    R（補助。速さの包絡の相関）= 時間領域での同時性。+ = 同時，− = 交代
        ノイズが必ず「同時」側に偏らせるので、動いている成分が 2 つ以上あるときだけ出す

  仕様の 4 つの問いとの対応:
    一関節依存  → N_eff ≈ 1
    全体が動く  → N_eff 大
    一つずつ    → S < 1（補助: R < 0）
    同時に協調  → S ≈ 1 かつ C > 1

精度について（合成データで確認した結果）
  シェアは **入力周波数のビンだけ**（ターゲットは 1/T の素数倍の正弦波の和）で作り、
  近傍ビンのノイズを引く。時間領域の瞬間シェアで作るとジッタが全関節に等しく乗るため、
  一関節依存の動きでも N_eff が最大値に張り付く（1.3 → 3.6 になることを確認）。
  - N_eff と C は 400 → 1600 DPI でほぼ変わらない（1.46 → 1.46，4.88 → 4.88）。そのまま使える
  - S は交代を 0.58、同時を 1.00 前後で返し、DPI によらず安定（0.58 → 0.60）
  - R は 1600 DPI で符号まで反転する（ノイズの包絡が成分間で相関するため）。
    低 DPI の補助指標として扱い、R_reliable が False の試行では使わない
  成分ごとの SNR（周波数領域・時間領域）も出すので、「動いていない」と「測れていない」を区別できる。
"""
import numpy as np
import common as C


def run(tr):
    if tr.kind != "track" or not tr.has_rs or not tr.traj:
        return "## a1 協調の指標\n\n静止の試行 / 関節データなしのため省略。\n"
    res = C.joint_indices(tr)
    if res is None:
        return "## a1 協調の指標\n\n必要な関節（肩・肘・手首）が揃っていないため省略。\n"
    slim = dict(res)
    slim["per_bin"] = [{k: v for k, v in r.items() if k != "_pwr"} for r in res["per_bin"]]
    tr.save_json("a1_indices.json", slim)

    names = res["parts"]
    rows = [{"part": p, "label": C.PARTLABEL[p], "share": res["share"][p],
             "snr": res["snr_part"][p], "snr_t": res.get("snr_time", {}).get(p, np.nan),
             "vel": res.get("vel_rms", {}).get(p, np.nan)} for p in names]

    plt = C.setup_plot()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # シェア
    a = axes[0]
    a.bar(range(len(names)), [res["share"][p] for p in names],
          color=[C.PARTCOLOR[p] for p in names], width=.65)
    for i, p in enumerate(names):
        a.text(i, res["share"][p], f"SNR {res['snr_part'][p]:.0f}", ha="center", va="bottom", fontsize=7)
    a.set_xticks(range(len(names)))
    a.set_xticklabels([C.PARTLABEL[p] for p in names], fontsize=8, rotation=12)
    a.set_ylabel("寄与シェア")
    a.set_title(f"N_eff = {res['N_eff']:.2f} / {res['N_max']}（主役: {C.PARTLABEL[res['dominant']]}）")
    # 包絡（時系列）
    a = axes[1]
    V, E, ok = C.vel_contributions(tr)
    from scipy.signal import butter, filtfilt
    b, aa = butter(2, 0.5 / (C.FS_U / 2))
    t = tr.grid_t
    for p in names:
        m, _ = C.nanfill_grid(np.linalg.norm(V[p], axis=1))
        a.plot(t, filtfilt(b, aa, m), lw=1.1, color=C.PARTCOLOR[p], label=C.PARTLABEL[p])
    a.set_xlim(0, min(20, tr.T)); a.set_xlabel("時間 [s]"); a.set_ylabel("活動量の包絡 [mm/s]")
    a.set_title(f"S = {res['S']:.2f}（1 = 同時，<1 = 交代）"
                + (f"　R = {res['R']:+.2f}" if res.get("R_reliable") else "　R = 参考値"))
    a.legend(fontsize=7)
    # ビンごとの C
    a = axes[2]
    pb = [r for r in res["per_bin"] if np.isfinite(r.get("C", np.nan))]
    for ax_name, mk in zip(C.AX, "os"):
        s = [r for r in pb if r["axis"] == ax_name]
        a.plot([r["f_hz"] for r in s], [r["C"] for r in s], mk + "-", lw=1.2, ms=5, label=f"{ax_name} 軸")
    a.axhline(1, color="#888", ls="--", lw=1)
    a.set_xscale("log"); a.set_xlabel("周波数 [Hz]"); a.set_ylabel("Σ|成分| / |合計|")
    a.set_title(f"C = {res['C']:.2f}（1 = 単純な和，>1 = 相殺）"); a.legend(fontsize=8)
    fig.suptitle(f"{tr.name}（{tr.tag()}，DPI {tr.dpi:g}，参加者 {tr.participant}）協調の指標", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a1_indices.png", dpi=120); plt.close(fig)

    md = "## a1 協調の指標\n\n"
    md += (f"**N_eff = {res['N_eff']:.2f} / {res['N_max']}**（主役は {C.PARTLABEL[res['dominant']]}，"
           f"シェア {res['dominant_share']:.0%}）　**S = {res['S']:.2f}**　**C = {res['C']:.2f}**"
           + (f"　R = {res['R']:+.2f}" if np.isfinite(res.get("R", np.nan)) else "　R = –")
           + (f"（信頼できる，動いている成分 {res['R_n_active']}）" if res.get("R_reliable")
              else "（参考値: 動いている成分が 2 つ未満）")
           + f"　手先 = {C.LABEL[res['end']]}\n\n")
    md += C.md_table(rows, [("成分", "label"), ("寄与シェア", "share", "{:.1%}"),
                            ("速度 RMS [mm/s]", "vel", "{:.1f}"),
                            ("SNR（周波数）", "snr", "{:.1f}"), ("SNR（時間）", "snr_t", "{:.1f}")])
    md += "\n読み方: "
    md += ("N_eff が 1 に近いので **一関節に依存**している。" if res["N_eff"] < 1.6 else
           ("N_eff が大きいので **複数の関節が分担**している。" if res["N_eff"] > 2.5 else
            "N_eff は中間（2 関節ぶんの分担）。"))
    if res["N_eff"] >= 1.6 and np.isfinite(res["S"]):
        md += ("S が小さいので **担当が入れ替わっている**（関節が一つずつ動く傾向）。" if res["S"] < 0.85 else
               ("S が 1 に近いので **同時に動いている**。" if res["S"] > 0.95 else
                "S は中間（同時と交代の間）。"))
    if np.isfinite(res["C"]):
        md += (f"C = {res['C']:.2f} で、成分が互いに**打ち消し合いながら**手先を作っている。"
               if res["C"] > 1.3 else "C は 1 に近く、成分はほぼそのまま足し合わさっている。")
    md += ("\n\nSNR が 2 を下回る成分は「動いていない」のではなく「測れていない」可能性がある"
           "（その成分のシェアは参考値）。\n")
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))