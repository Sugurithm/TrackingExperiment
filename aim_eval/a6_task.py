"""c1 課題の成績（身体とは別に、まず行動として）

何をするか
  ターゲットとカーソルのずれを軸別・時間別に集計する。
  - RMS 誤差（x, y, 2D）、on-target 率、誤差の分布（95 %）
  - 誤差の時間変化（前半 / 後半 = 慣れ）と、ターゲットの速さ別の誤差
  - マウスの移動量（実際に手が動いた距離）: DPI の効果が出ているかの確認
  - 軌跡の重ね描き（ターゲット vs カーソル）
わかること
  その試行の難しさと出来。以降の協調の指標を「うまくできているとき / いないとき」で切り分ける材料。
"""
import numpy as np
import common as C


def run(tr):
    if tr.kind != "track":
        return "## c1 課題の成績\n\n静止の試行なので省略。\n"
    tg, cu = tr.task_grid("target"), tr.task_grid("cursor")
    t = tr.grid_t
    e = cu - tg
    v_t = np.linalg.norm(np.gradient(tg, axis=0), axis=1) * C.FS_U      # ターゲットの速さ [mm/s]
    on = np.interp(t, tr.task_t, tr.on_target)

    half = t < tr.T / 2
    bins = [0, 100, 200, 400, 1e9]
    res = {"rms_x": float(np.sqrt(np.mean(e[:, 0] ** 2))), "rms_y": float(np.sqrt(np.mean(e[:, 1] ** 2))),
           "rms_2d": float(np.sqrt(np.mean(np.sum(e ** 2, 1)))),
           "p95_2d": float(np.percentile(np.linalg.norm(e, axis=1), 95)),
           "bias_x": float(e[:, 0].mean()), "bias_y": float(e[:, 1].mean()),
           "on_target_pct": float(100 * on.mean()),
           "rms_first_half": float(np.sqrt(np.mean(np.sum(e[half] ** 2, 1)))),
           "rms_second_half": float(np.sqrt(np.mean(np.sum(e[~half] ** 2, 1)))),
           "target_speed_mean": float(v_t.mean()), "target_speed_p99": float(np.percentile(v_t, 99)),
           "mouse_path_mm": float(np.sum(np.linalg.norm(np.diff(tr.mouse, axis=0), axis=1))),
           "mouse_range_x_mm": float(np.ptp(tr.mouse[:, 0])), "mouse_range_y_mm": float(np.ptp(tr.mouse[:, 1]))}
    by_speed = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (v_t >= a) & (v_t < b)
        if m.sum() > 30:
            by_speed.append({"bin": f"{a:.0f}–{b:.0f}" if b < 1e8 else f"{a:.0f}+",
                             "n": int(m.sum()), "rms_2d": float(np.sqrt(np.mean(np.sum(e[m] ** 2, 1)))),
                             "on_pct": float(100 * on[m].mean())})
    res["by_target_speed"] = by_speed
    tr.save_json("a6_task.json", res)

    plt = C.setup_plot()
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(tg[:, 0], tg[:, 1], lw=.8, color="#c33", label="ターゲット")
    ax.plot(cu[:, 0], cu[:, 1], lw=.8, color="#333", alpha=.7, label="カーソル")
    ax.set_aspect("equal"); ax.set_xlabel("画面 x [mm]"); ax.set_ylabel("画面 y [mm]")
    ax.set_title("軌跡"); ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 2, 2)
    for k, lab in enumerate(("x", "y")):
        ax.plot(t, e[:, k], lw=.7, label=f"誤差 {lab}")
    ax.set_xlabel("時間 [s]"); ax.set_ylabel("カーソル − ターゲット [mm]")
    ax.set_title(f"誤差の時系列（RMS {res['rms_2d']:.1f} mm）"); ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t, tg[:, 0], lw=.8, color="#c33"); ax.plot(t, cu[:, 0], lw=.8, color="#333", alpha=.7)
    ax.set_xlim(0, min(12, tr.T)); ax.set_xlabel("時間 [s]"); ax.set_ylabel("x [mm]")
    ax.set_title("x 成分の追従（先頭 12 s）")
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t, tr.on_grid(tr.task_t, tr.mouse)[:, 0], lw=.8, label="マウス x")
    ax.plot(t, tr.on_grid(tr.task_t, tr.mouse)[:, 1], lw=.8, label="マウス y")
    ax.set_xlabel("時間 [s]"); ax.set_ylabel("積算変位 [mm]")
    ax.set_title(f"マウス（手）の移動 範囲 x {res['mouse_range_x_mm']:.0f} / y {res['mouse_range_y_mm']:.0f} mm")
    ax.legend(fontsize=8)
    fig.suptitle(f"{tr.name}（{tr.tag()}，DPI {tr.dpi:g}）", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a6_task.png", dpi=120); plt.close(fig)

    md = "## c1 課題の成績\n\n"
    md += (f"- RMS 誤差 2D **{res['rms_2d']:.1f} mm**（x {res['rms_x']:.1f}，y {res['rms_y']:.1f}），"
           f"95 % {res['p95_2d']:.1f} mm，on-target {res['on_target_pct']:.1f} %\n")
    md += f"- 前半 {res['rms_first_half']:.1f} → 後半 {res['rms_second_half']:.1f} mm\n"
    md += (f"- ターゲットの速さ 平均 {res['target_speed_mean']:.0f} / 99 % {res['target_speed_p99']:.0f} mm/s，"
           f"マウスの移動量 {res['mouse_path_mm'] / 10:.0f} cm（範囲 x {res['mouse_range_x_mm']:.0f} mm）\n\n")
    if by_speed:
        md += C.md_table(by_speed, [("ターゲットの速さ [mm/s]", "bin"), ("フレーム", "n", "{:.0f}"),
                                    ("RMS 誤差 [mm]", "rms_2d"), ("on-target [%]", "on_pct")])
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
