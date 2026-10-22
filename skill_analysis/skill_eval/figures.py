"""図（条件ごと）。すべて results/<条件>/figures/ に PNG で書き出す。

色の約束
  練度グループ … 順序があるので 1 色相の濃淡（灰: A 濃 → C 淡）＋ マーカーの形（A ● / B ■ / C ▲）
  関節成分     … カテゴリ色（体幹 青 / 肩 橙 / 肘 緑青 / 手首 黄）。どの図でも同じ成分は同じ色
  Syou         … 赤い輪と名前のラベル（外れ値候補として常に個別に見えるようにする）
"""
import os
import numpy as np

from . import config as K
from .io import C
from .coordination import PARTS, PLAB, PSHORT, PAIRS
from .stats import METRICS, MLAB, FAMILY

GCOL = {"A": "#262626", "B": "#6f6e69", "C": "#a3a29b"}
GMARK = {"A": "o", "B": "s", "C": "^"}
PCOL = {"trunk": "#2a78d6", "upperarm": "#eb6834", "forearm": "#1baf7a", "hand": "#eda100"}
HL = "#e34948"
INK2 = "#52514e"


def _plt():
    plt = C.setup_plot()
    plt.rcParams.update({"axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "lines.linewidth": 2})
    return plt


def _freq_ticks(ax, fs):
    """対数軸の目盛りを入力周波数そのものにする（既定の 10^x 表記は重なるため）。"""
    from matplotlib.ticker import NullLocator
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(fs)
    ax.set_xticklabels([f"{f:.3g}" for f in fs], rotation=0)


def _save(fig, out, name):
    p = os.path.join(out, name)
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
    return p


def _strip(ax, people, key, tests_row):
    groups = list(K.GROUPS)
    for gi, g in enumerate(groups):
        rows = [p for p in people if p["group"] == g and np.isfinite(p[key])]
        xs = gi + np.linspace(-.12, .12, max(len(rows), 1))[:len(rows)]
        for x, p in zip(xs, rows):
            sd = p.get(key + "__sd", np.nan)
            if np.isfinite(sd):
                ax.plot([x, x], [p[key] - sd, p[key] + sd], color=GCOL[g], lw=1, alpha=.6, zorder=2)
            ax.scatter([x], [p[key]], s=34, marker=GMARK[g], color=GCOL[g], edgecolor="white", lw=1, zorder=3)
            if p["participant"] in K.OUTLIER_CANDIDATES:
                ax.scatter([x], [p[key]], s=110, facecolor="none", edgecolor=HL, lw=1.6, zorder=4)
                ax.annotate(p["participant"], (x, p[key]), xytext=(6, 0), textcoords="offset points",
                            fontsize=7, color=HL, va="center")
        if rows:
            m = np.mean([p[key] for p in rows])
            ax.plot([gi - .28, gi + .28], [m, m], color=GCOL[g], lw=2.2, zorder=1, solid_capstyle="round")
            rest = [p[key] for p in rows if p["participant"] not in K.OUTLIER_CANDIDATES]
            if len(rest) != len(rows) and rest:
                ax.plot([gi - .28, gi + .28], [np.mean(rest)] * 2, color=GCOL[g], lw=1.4, ls=(0, (2, 2)), zorder=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([K.GROUP_LABEL[g].split()[0] for g in groups])
    ax.set_xlim(-.5, len(groups) - .5)
    r = tests_row
    ax.set_title(MLAB[key], loc="left")
    if r is not None:
        ax.text(0.0, -0.2, f"ρ={r['rho_all']:+.2f} (p={r['p_all']:.2f}) / Syou除外 ρ={r['rho_noSyou']:+.2f} (p={r['p_noSyou']:.2f})",
                transform=ax.transAxes, fontsize=6.6, color=INK2)


def metric_grid(people, tests, keys, title, out, name, ncol=4):
    plt = _plt()
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.75 * nrow), squeeze=False)
    tmap = {t["metric"]: t for t in tests}
    for i, k in enumerate(keys):
        _strip(axs.flat[i], people, k, tmap.get(k))
    for j in range(len(keys), nrow * ncol):
        axs.flat[j].set_visible(False)
    fig.suptitle(title + "\n点 = 参加者（seed 3 本の平均、縦線 = seed 間 SD）、実線 = 群平均、破線 = Syou を除いた A 群平均",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(h_pad=2.2, rect=(0, 0, 1, 1 - 0.5 / (2.75 * nrow)))
    p = _save(fig, out, name); plt.close(fig); return p


def tracking_freq(fgroup, out, name="tracking_by_freq.png"):
    plt = _plt()
    cols = [("gain", "追従ゲイン |カーソル/ターゲット|"), ("lag_ms", "位相遅れ [ms]"), ("rel_err", "相対誤差 |誤差|/|ターゲット|")]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2), squeeze=False)
    for i, ax_name in enumerate(C.AX):
        for j, (k, lab) in enumerate(cols):
            ax = axs[i, j]
            for g in K.GROUPS:
                for tag, ls in (("all", "-"), ("noSyou", (0, (2, 2)))):
                    if tag == "noSyou" and g != "A":
                        continue
                    rows = sorted([r for r in fgroup if r["set"] == tag and r["group"] == g and r["axis"] == ax_name],
                                  key=lambda r: r["f_hz"])
                    if not rows:
                        continue
                    f = [r["f_hz"] for r in rows]; v = [r[k] for r in rows]
                    lab_ = K.GROUP_LABEL[g] + ("（Syou 除外）" if tag == "noSyou" else "")
                    ax.plot(f, v, ls=ls, color=GCOL[g], marker=GMARK[g] if tag == "all" else None, ms=6,
                            mec="white", lw=1.8 if tag == "all" else 1.3, label=lab_)
            ax.set_xscale("log")
            _freq_ticks(ax, sorted({r["f_hz"] for r in fgroup if r["axis"] == ax_name}))
            ax.set_xlabel(f"入力周波数 [Hz]（{ax_name} 軸）")
            ax.set_title(lab, loc="left")
            if k == "gain":
                ax.axhline(1, color="#8a8984", lw=.8)
    axs[0, 0].legend(loc="best")
    fig.suptitle("周波数ごとの追従特性（群平均）", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    p = _save(fig, out, name); plt.close(fig); return p


def joint_shares(people, out, name="joint_shares.png"):
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    # 左: 参加者ごと
    ax = axs[0]
    ys, labels = [], []
    y = 0
    for g in K.GROUPS:
        for p in [p for p in people if p["group"] == g]:
            left = 0
            for part in PARTS:
                v = max(p[f"share_{part}"], 0)
                ax.barh(y, v, left=left, color=PCOL[part], edgecolor="white", lw=2, height=.72,
                        label=PLAB[part] if y == 0 else None)
                left += v
            lab = f"{p['participant']}（{g}）"
            labels.append(lab); ys.append(y); y += 1
        y += .6
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    for t in ax.get_yticklabels():
        if t.get_text().split("（")[0] in K.OUTLIER_CANDIDATES:
            t.set_color(HL)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("入力周波数での速度パワーのシェア")
    ax.set_title("参加者ごとの関節の寄与（seed 平均）", loc="left")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.14))
    # 右: 群平均（A は Syou あり / なし）
    ax = axs[1]
    sets = []
    for g in K.GROUPS:
        sets.append((K.GROUP_LABEL[g], [p for p in people if p["group"] == g]))
        if g == "A":
            sets.append(("A（Syou 除外）", [p for p in people if p["group"] == g and p["participant"] not in K.OUTLIER_CANDIDATES]))
    for i, (lab, rows) in enumerate(sets):
        left = 0
        for part in PARTS:
            v = max(np.mean([r[f"share_{part}"] for r in rows]), 0)
            ax.barh(i, v, left=left, color=PCOL[part], edgecolor="white", lw=2, height=.72)
            if v > .07:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5, color="#0b0b0b")
            left += v
    ax.set_yticks(range(len(sets))); ax.set_yticklabels([s[0] for s in sets]); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("シェア（群平均）"); ax.set_title("練度グループの平均", loc="left")
    fig.tight_layout()
    p = _save(fig, out, name); plt.close(fig); return p


def share_by_freq(fperson, out, name="share_by_freq.png"):
    """列 = A（Syou 除外）/ Syou / B / C、行 = 軸。線 = 成分のシェア。"""
    plt = _plt()
    cols = [("A（Syou 除外）", lambda r: r["group"] == "A" and r["participant"] not in K.OUTLIER_CANDIDATES)]
    cols += [(nm, (lambda nm_: (lambda r: r["participant"] == nm_))(nm)) for nm in K.OUTLIER_CANDIDATES]
    cols += [(K.GROUP_LABEL[g], (lambda g_: (lambda r: r["group"] == g_))(g)) for g in ("B", "C")]
    fig, axs = plt.subplots(2, len(cols), figsize=(3.4 * len(cols), 6), squeeze=False, sharey=True)
    for i, ax_name in enumerate(C.AX):
        for j, (lab, sel) in enumerate(cols):
            ax = axs[i, j]
            rows = [r for r in fperson if r["axis"] == ax_name and sel(r)]
            fs = sorted({r["f_hz"] for r in rows})
            for part in PARTS:
                v = [np.nanmean([r[f"share_{part}"] for r in rows if r["f_hz"] == f]) for f in fs]
                ax.plot(fs, v, color=PCOL[part], marker="o", ms=5, mec="white", label=PLAB[part])
            ax.set_xscale("log"); ax.set_ylim(0, 1); _freq_ticks(ax, fs)
            ax.set_title(f"{lab}　{ax_name} 軸", loc="left", color=HL if lab in K.OUTLIER_CANDIDATES else "#0b0b0b")
            ax.set_xlabel("入力周波数 [Hz]")
            if j == 0:
                ax.set_ylabel("速度パワーのシェア")
    axs[0, 0].legend(loc="upper left")
    fig.suptitle("周波数ごとの関節の分担（遅い成分と速い成分で担当が変わるか）", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    p = _save(fig, out, name); plt.close(fig); return p


def timing(people, tests, out, name="timing.png"):
    keys = ["lead_upperarm_ms", "lead_forearm_ms", "lead_hand_ms",
            "xr_upperarm_forearm", "xlag_upperarm_forearm_ms", "opp_upperarm_forearm",
            "xr_forearm_hand", "xlag_forearm_hand_ms", "opp_forearm_hand"]
    return metric_grid(people, tests, keys, "時間軸の役割分担（先行・位相差・支え合い）", out, name, ncol=3)


def syou_sensitivity(tests, out, name="syou_sensitivity.png"):
    plt = _plt()
    rows = [t for t in tests if np.isfinite(t["rho_all"]) or np.isfinite(t["rho_noSyou"])]
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.27 * len(rows) + 1.6), sharey=True,
                            gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(rows))
    ax = axs[0]
    for i, r in enumerate(rows):
        ax.plot([r["rho_all"], r["rho_noSyou"]], [i, i], color="#c3c2b7", lw=1.5, zorder=1)
    ax.scatter([r["rho_all"] for r in rows], y, s=30, color="#262626", label="全 8 人", zorder=2)
    ax.scatter([r["rho_noSyou"] for r in rows], y, s=30, facecolor="white", edgecolor="#262626", lw=1.4,
               label="Syou 除外 7 人", zorder=3)
    for i, r in enumerate(rows):
        if r["flip"]:
            ax.text(1.04, i, "変化大", fontsize=6.8, color=HL, va="center", transform=ax.get_yaxis_transform())
    ax.axvline(0, color="#8a8984", lw=.8)
    ax.set_xlim(-1.05, 1.05); ax.set_xlabel("練度との順位相関 ρ（+ = 高練度ほど大きい）")
    ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in rows], fontsize=7.2)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax = axs[1]
    nm = K.OUTLIER_CANDIDATES[0]
    z = np.array([r.get(f"{nm}_z", np.nan) for r in rows], float)
    ax.barh(y, np.clip(z, -8, 8), color=["#e34948" if abs(v) >= 2 else "#c3c2b7" for v in np.nan_to_num(z)],
            height=.65, edgecolor="white")
    for v in (-2, 2):
        ax.axvline(v, color="#8a8984", lw=.8, ls=(0, (2, 2)))
    ax.axvline(0, color="#8a8984", lw=.8)
    ax.set_xlabel(f"{nm} のロバスト z（他 7 人に対して。|z| ≥ 2 を赤、±8 で打ち切り）")
    fig.suptitle(f"{nm} を含める / 除くで結論が変わる指標", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    p = _save(fig, out, name); plt.close(fig); return p


def example_traces(examples, out, name="example_traces.png"):
    """examples: [(見出し, group, t, target(N,2), cursor(N,2), events)]"""
    plt = _plt()
    n = len(examples)
    fig, axs = plt.subplots(n, 2, figsize=(13, 2.2 * n), squeeze=False, gridspec_kw={"width_ratios": [1, 1.3]})
    for i, (title, g, t, tg, cu, ev) in enumerate(examples):
        m = t < 10
        ax = axs[i, 0]
        ax.plot(t[m], tg[m, 0], color="#8a8984", lw=2.2, label="ターゲット x")
        ax.plot(t[m], cu[m, 0], color=GCOL[g], lw=1.2, label="カーソル x")
        ax.set_title(title + "　x の追従（先頭 10 s）", loc="left", color=HL if K.OUTLIER_CANDIDATES[0] in title else "#0b0b0b")
        ax.set_ylabel("mm")
        if i == 0: ax.legend(loc="upper right", ncol=2)
        ax = axs[i, 1]
        m2 = t < 20
        em = np.linalg.norm(cu - tg, axis=1)
        ax.plot(t[m2], em[m2], color=GCOL[g], lw=1)
        et = [e for e in ev if e["t_s"] < 20]
        ax.scatter([e["t_s"] for e in et], [e["peak_mm"] for e in et], s=14, color=HL, zorder=3,
                   label="修正イベント（誤差の山）")
        ax.axhline(K_TARGET_R_MM, color="#8a8984", lw=.8, ls=(0, (2, 2)))
        ax.set_title(f"誤差の大きさ |e|（先頭 20 s、破線 = ターゲット半径）", loc="left")
        ax.set_ylabel("mm")
        if i == 0: ax.legend(loc="upper right")
    axs[-1, 0].set_xlabel("時間 [s]"); axs[-1, 1].set_xlabel("時間 [s]")
    fig.tight_layout()
    p = _save(fig, out, name); plt.close(fig); return p


K_TARGET_R_MM = 8.0   # 既定の target_radius_cm 0.8（report 側で試行の設定値に上書きする）


def correction_ecdf(events, out, name="correction_amplitude.png"):
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, (key, lab) in enumerate((("amp_mm", "修正の振幅（|e| の山の突出度）[mm]"), ("time_s", "修正にかかる時間 [s]"))):
        ax = axs[k]
        sets = [(K.GROUP_LABEL[g] + ("（Syou 除外）" if g == "A" else ""), g,
                 [e[key] for e in events if e["group"] == g and e["participant"] not in K.OUTLIER_CANDIDATES])
                for g in K.GROUPS]
        for nm in K.OUTLIER_CANDIDATES:
            sets.append((nm, None, [e[key] for e in events if e["participant"] == nm]))
        for lab_, g, v in sets:
            if not v:
                continue
            v = np.sort(v); y = np.arange(1, len(v) + 1) / len(v)
            ax.step(v, y, where="post", color=GCOL[g] if g else HL, lw=1.8,
                    ls="-" if g else (0, (3, 2)), label=f"{lab_}（{len(v)} 件）")
        ax.set_xscale("log"); ax.set_xlabel(lab); ax.set_ylabel("累積割合")
        ax.legend(loc="lower right")
    fig.suptitle("修正イベントの分布（全試行のイベントをまとめた累積分布）", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    p = _save(fig, out, name); plt.close(fig); return p
