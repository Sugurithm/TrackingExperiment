"""c6 関節角度とセグメント（3 次元座標から計算できる量）

何をするか
  - 骨長（上腕 = 肩–肘、前腕 = 肘–手首、手 = 手首–MCP3）の中央値と SD。
    SD は RS 単体の信頼性の指標（本来は一定）
  - **肘の角度**（肩–肘–手首）の中央値・可動域・SD・角速度
  - セグメントの**向きの変化**: 上腕・前腕・手のベクトルが試行中に何度動いたか（机座標での方位角・仰角）。
    手の向き（手首 → MCP3）は手首の偏位（撓屈・尺屈）に相当する成分
  - 角度から見た寄与: 各セグメントの角度の振れ幅 × セグメント長 ≈ そのセグメントが生む手先の移動量
わかること
  「どの関節をどれだけ曲げているか」を角度で示す。c4 の寄与（mm）と合わせると、
  手首だけ小さく回しているのか、腕全体を振っているのかが角度で言える。
注意
  前腕の回内外（軸まわりの回旋）は 3 点では求まらない。手の向きの変化として部分的にしか見えない。
"""
import numpy as np
import common as C

SEGS = [("upperarm", "shoulder", "elbow", "上腕"), ("forearm", "elbow", "wrist", "前腕"),
        ("hand", "wrist", "mcp3", "手")]


def _dir_angles(V):
    """ベクトル列 → 方位角（机の xy 面内）と仰角 [deg]。"""
    az = np.degrees(np.arctan2(V[:, 1], V[:, 0]))
    el = np.degrees(np.arcsin(np.clip(V[:, 2] / np.linalg.norm(V, axis=1), -1, 1)))
    return az, el


def run(tr):
    if not tr.has_rs:
        return "## c6 関節角度とセグメント\n\n関節データなしのため省略。\n"
    G = {n: tr.joint_grid(n) for n in tr.rs_names}
    rows = []
    for key, a, b, lab in SEGS:
        if a not in G or b not in G:
            continue
        V = G[b] - G[a]
        L = np.linalg.norm(V, axis=1)
        az, el = _dir_angles(V)
        azu = np.unwrap(np.radians(az)) * 180 / np.pi
        rows.append({"seg": key, "label": lab, "len_mm": float(np.nanmedian(L)),
                     "len_sd_mm": float(np.nanstd(L)),
                     "az_range_deg": float(np.nanpercentile(azu, 97) - np.nanpercentile(azu, 3)),
                     "el_range_deg": float(np.nanpercentile(el, 97) - np.nanpercentile(el, 3)),
                     "az_sd_deg": float(np.nanstd(azu)), "el_sd_deg": float(np.nanstd(el)),
                     "swept_mm": float(np.nanmedian(L) * np.radians(
                         np.nanpercentile(azu, 97) - np.nanpercentile(azu, 3)))})
    ang = {}
    if all(n in G for n in ("shoulder", "elbow", "wrist")):
        e = C.angle_deg(G["shoulder"], G["elbow"], G["wrist"])
        de = np.gradient(e) * C.FS_U
        ang["elbow"] = {"median_deg": float(np.nanmedian(e)), "sd_deg": float(np.nanstd(e)),
                        "range_deg": float(np.nanpercentile(e, 97) - np.nanpercentile(e, 3)),
                        "vel_p95_dps": float(np.nanpercentile(np.abs(de), 95))}
    if all(n in G for n in ("elbow", "wrist", "mcp3")):
        w = C.angle_deg(G["elbow"], G["wrist"], G["mcp3"])
        ang["wrist"] = {"median_deg": float(np.nanmedian(w)), "sd_deg": float(np.nanstd(w)),
                        "range_deg": float(np.nanpercentile(w, 97) - np.nanpercentile(w, 3)),
                        "vel_p95_dps": float(np.nanpercentile(np.abs(np.gradient(w) * C.FS_U), 95))}
    tr.save_json("a7_angles.json", {"segments": rows, "angles": ang})

    plt = C.setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    t = tr.grid_t
    for key, a, b, lab in SEGS:
        if a not in G or b not in G:
            continue
        az, _ = _dir_angles(G[b] - G[a])
        axes[0].plot(t, np.unwrap(np.radians(az)) * 180 / np.pi, lw=.8, label=f"{lab}の方位角")
    axes[0].set_xlabel("時間 [s]"); axes[0].set_ylabel("方位角 [deg]")
    axes[0].set_title("セグメントの向き（机の xy 面内）"); axes[0].legend(fontsize=8)
    if "elbow" in ang:
        axes[1].plot(t, C.angle_deg(G["shoulder"], G["elbow"], G["wrist"]), lw=.8, color="#eb6834",
                     label="肘の角度")
        if "wrist" in ang:
            axes[1].plot(t, C.angle_deg(G["elbow"], G["wrist"], G["mcp3"]), lw=.8, color="#1baf7a",
                         label="手首の角度")
        axes[1].set_xlabel("時間 [s]"); axes[1].set_ylabel("角度 [deg]")
        axes[1].set_title("関節角度"); axes[1].legend(fontsize=8)
    fig.suptitle(f"{tr.name}（{tr.tag()}，DPI {tr.dpi:g}）", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a7_angles.png", dpi=120); plt.close(fig)

    md = "## c6 関節角度とセグメント\n\n"
    md += C.md_table(rows, [("セグメント", "label"), ("長さ [mm]", "len_mm"), ("長さ SD [mm]", "len_sd_mm", "{:.1f}"),
                            ("方位角の範囲 [deg]", "az_range_deg"), ("仰角の範囲 [deg]", "el_range_deg"),
                            ("方位角 SD", "az_sd_deg", "{:.1f}"),
                            ("方位角で掃いた距離 [mm]", "swept_mm", "{:.0f}")])
    if ang:
        md += "\n"
        md += C.md_table([{"j": "肘（肩–肘–手首）" if k == "elbow" else "手首（肘–手首–MCP3）", **v}
                          for k, v in ang.items()],
                         [("角度", "j"), ("中央値 [deg]", "median_deg"), ("範囲 [deg]", "range_deg"),
                          ("SD [deg]", "sd_deg", "{:.1f}"), ("角速度 95% [deg/s]", "vel_p95_dps", "{:.0f}")])
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
