"""図（matplotlib、PNG）。色は成分 → 固定の順（青・橙・緑・黄）、群 → 青・橙。"""
import numpy as np

from .. import config as K

PCOL = {"trunk": "#2a78d6", "upperarm": "#eb6834", "forearm": "#1baf7a", "hand": "#eda100",
        "distal": "#6250d6", "prox": "#2a78d6"}
GCOL = {"FPS": "#2a78d6", "nonFPS": "#eb6834", None: "#888888"}
CCOL = {"400_slow": "#2a78d6", "800_slow": "#6250d6", "400_fast": "#eb6834", "800_fast": "#e34948"}
CMARK = {"400_slow": "o", "800_slow": "s", "400_fast": "^", "800_fast": "D"}
PJA = {"trunk": "体幹", "upperarm": "上腕", "forearm": "前腕", "hand": "手"}
INK, MUTED = "#0b0b0b", "#6b6a66"


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    want = ["Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic"]
    have = {f.name for f in fm.fontManager.ttflist}
    if not any(w in have for w in want):
        for f in fm.findSystemFonts():
            if "NotoSansCJK" in f or "ipaexg" in f.lower():
                try:
                    fm.fontManager.addfont(f)
                except Exception:
                    pass
        have = {f.name for f in fm.fontManager.ttflist}
    fam = [w for w in want if w in have] + ["DejaVu Sans"]
    plt.rcParams.update({"font.family": fam, "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "axes.edgecolor": "#9a9893", "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "font.size": 9})
    return plt


def order_people(people_rows):
    """群 → 名前の順。"""
    rank = {g: i for i, g in enumerate(K.GROUPS)}
    return sorted({(rank.get(r["group"], 9), r["participant"], r["group"]) for r in people_rows})


def global_shares(cell_rows, path):
    plt = setup()
    ppl = order_people(cell_rows)
    conds = sorted({r["cond"] for r in cell_rows}, key=lambda c: (c.split("_")[1] != "slow", int(c.split("_")[0])))
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.32 * len(ppl) * len(conds) + 1.6), sharey=True)
    for ax, tag, title in ((axs[0], "gshare_", "全体の運動量シェア（どの成分がどれだけ動いたか）"),
                           (axs[1], "eshare_", "手先への寄与（マウス基準、負 = 逆向きに動いて打ち消す）")):
        y, ticks, labels = 0, [], []
        for _, p, g in ppl:
            for c in conds:
                r = [x for x in cell_rows if x["participant"] == p and x["cond"] == c]
                if not r:
                    continue
                r = r[0]
                left_pos, left_neg = 0.0, 0.0
                for part in K.PARTS:
                    v = r.get(tag + part, np.nan)
                    if not np.isfinite(v):
                        continue
                    if v >= 0:
                        ax.barh(y, v, left=left_pos, color=PCOL[part], height=.78, edgecolor="white", linewidth=1)
                        left_pos += v
                    else:
                        ax.barh(y, v, left=left_neg, color=PCOL[part], height=.78, edgecolor="white", linewidth=1)
                        left_neg += v
                ticks.append(y); labels.append(f"{p}（{'FPS' if g == 'FPS' else '非'}） {c}")
                y += 1
            y += 0.6
        ax.set_title(title, fontsize=10, loc="left")
        ax.axvline(0, color="#9a9893", lw=.8)
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("シェア")
    axs[0].invert_yaxis()                      # sharey なので 1 回だけ
    handles = [plt.Rectangle((0, 0), 1, 1, color=PCOL[p]) for p in K.PARTS]
    fig.legend(handles, [PJA[p] for p in K.PARTS], loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def movement_amp(prof, path, col="mshare_prox", ylabel="1 動作の近位（体幹+上腕）の寄与"):
    plt = setup()
    ppl = order_people(prof)
    n = len(ppl)
    nc = min(5, n)
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.3 * nr + .6), sharex=True, sharey=True, squeeze=False)
    for i, (_, p, g) in enumerate(ppl):
        ax = axs[i // nc, i % nc]
        for cond in CCOL:
            d, b = cond.split("_")
            rs = sorted([r for r in prof if r["participant"] == p and int(r["dpi"]) == int(d) and r["band"] == b],
                        key=lambda r: r["amp_mid"])
            if rs:
                ax.plot([r["amp_mid"] for r in rs], [r[col] for r in rs], color=CCOL[cond], lw=2,
                        marker=CMARK[cond], ms=5, label=cond)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{p}（{'FPS' if g == 'FPS' else '非 FPS'}）", fontsize=9, loc="left")
    for j in range(n, nr * nc):
        axs[j // nc, j % nc].axis("off")
    fig.supxlabel("動作の振幅（マウス）[mm]")
    fig.supylabel(ylabel)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def person_lines(cell_rows, cols, labels, path):
    """主要な指標を人ごとの線で 4 条件に並べる（線が平行 = 個人差の構造が条件で保たれる）。"""
    plt = setup()
    conds = [c for c in ("400_slow", "800_slow", "400_fast", "800_fast") if c in {r["cond"] for r in cell_rows}]
    nc = 4
    nr = int(np.ceil(len(cols) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.5 * nr + .5), squeeze=False)
    for i, c in enumerate(cols):
        ax = axs[i // nc, i % nc]
        for _, p, g in order_people(cell_rows):
            v = [next((r[c] for r in cell_rows if r["participant"] == p and r["cond"] == cc), np.nan) for cc in conds]
            ax.plot(range(len(conds)), v, color=GCOL[g], lw=1.2, marker="o", ms=4, alpha=.85)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, fontsize=7)
        ax.set_title(labels[c], fontsize=8, loc="left")
    for j in range(len(cols), nr * nc):
        axs[j // nc, j % nc].axis("off")
    handles = [plt.Line2D([], [], color=GCOL[g], lw=2) for g in K.GROUPS]
    fig.legend(handles, [K.GROUP_LABEL[g] for g in K.GROUPS], loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def forest(inv_rows, path, title):
    """DPI・帯域での人ごとの差の平均と 95 % CI。灰色の帯 = 同等とみなす幅（SESOI）。"""
    import textwrap
    plt = setup()
    rows = [r for r in inv_rows if r["main"]]
    metrics = list(dict.fromkeys(r["metric"] for r in rows))
    comps = list(dict.fromkeys((r["factor"], r["stratum"]) for r in rows))
    nc = 4
    nr = int(np.ceil(len(metrics) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, (0.32 * len(comps) + 1.1) * nr + .6), squeeze=False)
    for i, m in enumerate(metrics):
        ax = axs[i // nc, i % nc]
        delta = next(r["sesoi"] for r in rows if r["metric"] == m)
        ax.axvspan(-delta, delta, color="#e7e6e2", zorder=0)
        ax.axvline(0, color="#9a9893", lw=.8)
        lim = delta * 1.5
        for k, comp in enumerate(comps):
            r = next((x for x in rows if x["metric"] == m and (x["factor"], x["stratum"]) == comp), None)
            if r is None or not np.isfinite(r["d"]):
                continue
            col = "#2a78d6" if r["factor"] == "DPI" else "#eb6834"
            ax.plot([r["ci95_lo"], r["ci95_hi"]], [k, k], color=col, lw=2)
            ax.plot([r["d"]], [k], "o", color=col, ms=6, mec="white", mew=1.5)
            lim = max([lim, abs(r["d"])] + [abs(v) for v in (r["ci95_lo"], r["ci95_hi"]) if np.isfinite(v)])
        ax.set_xlim(-lim * 1.05, lim * 1.05)
        lab = next(r["label"] for r in rows if r["metric"] == m).replace("【参考】", "*")
        ax.set_title("\n".join(textwrap.wrap(lab, 20)), fontsize=8, loc="left")
        ax.set_yticks(range(len(comps)))
        ax.set_yticklabels([f"{f}: {s_}" for f, s_ in comps] if i % nc == 0 else [], fontsize=7)
        ax.set_ylim(len(comps) - .5, -.5)
        ax.tick_params(axis="x", labelsize=7)
    for j in range(len(metrics), nr * nc):
        axs[j // nc, j % nc].axis("off")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def group_dots(pm, cols, labels, path):
    plt = setup()
    nc = 4
    nr = int(np.ceil(len(cols) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.3 * nr + .4), squeeze=False)
    rng = np.random.default_rng(0)
    for i, c in enumerate(cols):
        ax = axs[i // nc, i % nc]
        for k, g in enumerate(K.GROUPS):
            v = [r[c] for r in pm if r["group"] == g and np.isfinite(r[c])]
            x = k + rng.uniform(-.12, .12, len(v))
            ax.plot(x, v, "o", color=GCOL[g], ms=7, mec="white", mew=1.5)
            if v:
                ax.plot([k - .25, k + .25], [np.mean(v)] * 2, color=INK, lw=2)
        ax.set_xticks(range(len(K.GROUPS))); ax.set_xticklabels([K.GROUP_LABEL[g] for g in K.GROUPS], fontsize=8)
        ax.set_xlim(-.6, len(K.GROUPS) - .4)
        ax.set_title(labels[c], fontsize=8, loc="left")
    for j in range(len(cols), nr * nc):
        axs[j // nc, j % nc].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def orientation(cell_rows, path):
    plt = setup()
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    ppl = order_people(cell_rows)
    names = [p for _, p, _ in ppl]
    for c in CCOL:
        rs = [r for r in cell_rows if r["cond"] == c]
        if not rs:
            continue
        xs = [names.index(r["participant"]) + (list(CCOL).index(c) - 1.5) * .15 for r in rs]
        axs[0].plot(xs, [r.get("K_orient", np.nan) for r in rs], CMARK[c], color=CCOL[c], ms=6, label=c, mec="white")
        axs[1].plot([r.get("psi_sdc_deg", np.nan) for r in rs], [r.get("psis_sd_deg", np.nan) for r in rs],
                    CMARK[c], color=CCOL[c], ms=6, mec="white", label=c)
        axs[2].plot(xs, [r.get("psis_per100mm_x", np.nan) for r in rs], CMARK[c], color=CCOL[c], ms=6, mec="white")
    for ax in (axs[0], axs[2]):
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, fontsize=8)
    axs[0].axhline(0, color="#9a9893", lw=.8); axs[0].axhline(1, color="#9a9893", lw=.8, ls="--")
    axs[0].set_title("向きの安定化 K（0 = 手首と一緒に回る、1 = 向きを保つ）", fontsize=9, loc="left")
    lim = np.nanmax([[r.get("psi_sdc_deg", np.nan), r.get("psis_sd_deg", np.nan)] for r in cell_rows] + [[1, 1]])
    axs[1].plot([0, lim], [0, lim], color="#9a9893", lw=.8, ls="--")
    axs[1].set_xlabel("手の向き ψ の SD（ランドマーク）[deg]"); axs[1].set_ylabel("センサの向きの SD [deg]")
    axs[1].set_title("手の向き vs マウスのセンサの向き", fontsize=9, loc="left")
    axs[2].axhline(0, color="#9a9893", lw=.8)
    axs[2].set_title("センサの向きの位置依存 [deg / 左右 100 mm]", fontsize=9, loc="left")
    axs[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)
    return path
