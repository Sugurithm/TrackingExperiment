"""a6 計測の可用性（取れているか・誤検出はないか）

何をするか
  記録全体と解析範囲（静止・各テンポ）で，関節ごとに次を数える。
  - 有効率: 座標が得られたフレームの割合（status の内訳つき）
  - 大外れ率: status は OK なのに MoCap から 100 mm 以上離れたフレーム（骨格の取り違え・誤検出）
  - 外れ値率: 前後 9 フレームの中央値から max(3·MAD, 15 mm) 以上飛んだフレームの割合（瞬間的な飛び）
  - 最長の欠損: 連続して使えなかった最長時間
  - フレームの時刻: 実効 fps，取りこぼし（周期の 1.5 倍を超える間隔）
わかること
  精度以前に「いつ・どれだけ測れないか」。誤検出は RMSE を大きく壊すので別に数える。
"""
import numpy as np
import common as C

OUTLIER_MIN_MM = 15.0   # 深度は 1 mm 単位で MAD が小さくなりがちなので，これ未満の飛びは外れ値と数えない


def _longest(bad, fs):
    best = cur = 0
    for b in bad:
        cur = cur + 1 if b else 0; best = max(best, cur)
    return best / fs


def run(tr):
    dt = np.diff(tr.t_rs)
    timing = {"fps": float(1 / np.median(dt)), "n_frames": int(len(tr.t_rs)), "duration_s": float(tr.t_rs[-1] - tr.t_rs[0]),
              "dropped": int((dt > 1.5 * np.median(dt)).sum()), "max_gap_ms": float(dt.max() * 1000)}
    segs = [("記録全体", np.ones(len(tr.t), bool)), ("解析範囲", tr.in_span)]
    rmk = np.zeros(len(tr.t), bool)
    for r in tr.rest: rmk |= tr.mask(r)
    segs.append(("静止", rmk))
    for b in tr.blocks: segs.append((b["name"], tr.mask((b["t0"], b["t1"]))))
    rows = []
    for n in tr.rs_names:
        j = tr.rs_names.index(n)
        valid = ~np.isnan(tr.P_raw[:, j, 0])
        flags, _ = C.hampel_flags(tr.Praw[n], min_dev=OUTLIER_MIN_MM)
        for name, mk in segs:
            N = mk.sum()
            if N == 0: continue
            st = {}
            if tr.status is not None and tr.status_names:
                cnt = np.bincount(tr.status[mk, j], minlength=len(tr.status_names))
                st = {tr.status_names[k]: int(c) for k, c in enumerate(cnt) if c and tr.status_names[k] != "OK"}
            usable = valid & ~tr.gross[n]
            rows.append({"joint": n, "label": C.LABEL.get(n, n), "segment": name, "n": int(N),
                         "valid_pct": 100 * valid[mk].mean(), "gross_pct": 100 * tr.gross[n][mk].mean(),
                         "usable_pct": 100 * usable[mk].mean(),
                         "outlier_pct": 100 * flags[mk & usable].mean() if (mk & usable).any() else np.nan,
                         "longest_gap_s": _longest(~usable[mk], tr.fs), "status": st})
    # 大外れの時間帯
    events = []
    g = np.zeros(len(tr.t), bool)
    for n in tr.compared(): g |= tr.gross[n]
    g &= tr.in_span
    st = None
    for i, v in enumerate(g):
        if v and st is None: st = i
        if (not v or i == len(g) - 1) and st is not None:
            if events and tr.t[st] - events[-1][1] < 0.5: events[-1] = (events[-1][0], float(tr.t[i]))
            else: events.append((float(tr.t[st]), float(tr.t[i])))
            st = None
    tr.save_json("a6_availability.json", {"timing": timing, "rows": rows, "gross_events_take_s": events})

    plt = C.setup_plot()
    fig, ax = plt.subplots(figsize=(15, 3.2))
    for i, n in enumerate(tr.rs_names):
        j = tr.rs_names.index(n); valid = ~np.isnan(tr.P_raw[:, j, 0])
        y = len(tr.rs_names) - i
        ax.scatter(tr.t[~valid], np.full((~valid).sum(), y), s=2, color="#999", marker="|")
        ax.scatter(tr.t[tr.gross[n]], np.full(tr.gross[n].sum(), y), s=4, color="#d03b3b", marker="|")
    ax.set_yticks(range(len(tr.rs_names), 0, -1)); ax.set_yticklabels([C.LABEL.get(n, n) for n in tr.rs_names])
    C.shade_segments(ax, tr); ax.axvspan(tr.span[0], tr.span[1], color="#1baf7a", alpha=.04)
    for a, b in tr.sync_windows: ax.axvspan(a, b, color="#eda100", alpha=.15)
    ax.set_xlabel("Take 時刻 [s]"); ax.set_title("欠損（灰）と大外れ（赤）。黄 = 同期ボード区間，青 = 動作，灰帯 = 静止")
    ax.set_ylim(.3, len(tr.rs_names) + .7)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a6_availability.png", dpi=120); plt.close(fig)

    md = "## a6 計測の可用性\n\n"
    md += (f"RS: {timing['fps']:.1f} fps，{timing['n_frames']} フレーム（{timing['duration_s']:.0f} s），"
           f"取りこぼし {timing['dropped']} 回（最大間隔 {timing['max_gap_ms']:.0f} ms）\n\n")
    md += C.md_table([r for r in rows if r["segment"] in ("記録全体", "解析範囲")],
                     [("関節", "label"), ("区間", "segment"), ("有効率 [%]", "valid_pct"), ("大外れ [%]", "gross_pct"),
                      ("使える [%]", "usable_pct"), ("外れ値 [%]", "outlier_pct"), ("最長欠損 [s]", "longest_gap_s")])
    if events:
        md += "\n解析範囲内の大外れの時間帯（Take 時刻）: " + "，".join(f"{a:.1f}–{b:.1f} s" for a, b in events[:10]) + "\n"
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))
