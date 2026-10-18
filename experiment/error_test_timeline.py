"""
joints_timeline.py
────────────────────────────────────────────────────────────────────
rs_joints.py の結果を，横軸 = 時刻の図 1 枚（2 段）にまとめる（MediaPipe はやり直さない）．

  python experiment/joints_timeline.py                         # 最新の trial_*
  python experiment/joints_timeline.py recordings/trial_XXXX

入力:  trial_*/joints/rs_joints.npz（必須）
       （あれば）trial_*/joints/mc_joints.npz と sync_check/sync_result.json → 誤差とタスクの区間も描く
出力:  trial_*/joints/timeline.png        … 図
       trial_*/joints/timeline_blocks.csv … 区間ごとの認識率と誤差の表

【図】横軸 = 記録の開始からの秒（rs_joints の --start に渡す秒と同じ）．
  背景の灰色の帯 = タスクの区間（上の文字: board = ボード，still = 静止，数字 = テンポ [bpm]）．帯のない所は準備・移動．
  上段: 関節ごとに，3 次元座標が取れなかった時刻に赤い印．印がなければすべて取れている．
  下段: 関節ごとの 3 次元の誤差（タスクの区間での偏りを引いたもの，1 s の中央値）．線が切れている所は取れなかった時刻．
────────────────────────────────────────────────────────────────────
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from lib import io_utils, sync_io
from lib.geometry import apply, interp_nan
from scipy.ndimage import uniform_filter1d

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]     # 関節の色（この順で固定）
FAIL = "#e34948"                                          # 取れなかった印
BAND = "#e9e8e4"                                          # タスクの区間の帯
INK, INK2 = "#0b0b0b", "#52514e"
BREAK_MM, TOP_MM = 50.0, 200.0                            # 下段の縦軸: BREAK_MM より上を圧縮，TOP_MM 以上は上端に張り付け [mm]

# ── 区間の検出（compare_protocol.py と同じもの．このファイルだけで動くように写した）──
NOMINAL_BPM = [20, 40, 60, 80, 100, 120]
ACT_THR, STILL_THR = 40.0, 8.0          # ブロック / 静止の判定（2 s RMS 速さ）[mm/s]
MIN_BLOCK_S, MERGE_S = 5.0, 2.0


def runs(mask):
    e = np.diff(np.r_[0, np.asarray(mask, np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def detect_blocks(t, W, sync_windows):
    """MoCap の手首 W（t は Take 時刻）から静止区間とテンポのブロックを検出する．"""
    fs = 1.0 / np.median(np.diff(t))
    ok = np.isfinite(W).all(axis=1)
    Wi = interp_nan(t, t[ok], W[ok], 0.5)
    good = np.isfinite(Wi).all(axis=1)
    Wf = np.where(good[:, None], Wi, 0.0)
    v = np.linalg.norm(np.gradient(uniform_filter1d(Wf, max(1, int(0.05 * fs)), axis=0), t, axis=0), axis=1)
    v[~good] = 0.0
    vr = np.sqrt(np.maximum(uniform_filter1d(v ** 2, int(2.0 * fs)), 0.0))
    t_lo = sync_windows[0][1] + 1.0 if sync_windows else t[0]
    t_hi = sync_windows[-1][0] - 1.0 if len(sync_windows) > 1 else t[-1]
    inner = (t > t_lo) & (t < t_hi) & good
    merged = []
    for a, b in runs(inner & (vr > ACT_THR)):
        if merged and t[a] - t[merged[-1][1] - 1] < MERGE_S:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    blocks = []
    for a, b in merged:
        if t[b - 1] - t[a] < MIN_BLOCK_S:
            continue
        seg = Wi[a:b][good[a:b]]
        seg = seg - seg.mean(0)
        axis = np.linalg.svd(seg, full_matrices=False)[2][0]
        x = seg @ axis
        n_fft = 1 << int(np.ceil(np.log2(len(x) * 8)))
        F = np.abs(np.fft.rfft(x * np.hanning(len(x)), n_fft))
        f = np.fft.rfftfreq(n_fft, 1 / fs)
        band = (f > 0.05) & (f < 3.0)
        f0 = float(f[band][np.argmax(F[band])])
        bpm = f0 * 120.0
        blocks.append(dict(t0=float(t[a]), t1=float(t[b - 1]), f0_hz=f0, bpm=bpm,
                           nominal_bpm=min(NOMINAL_BPM, key=lambda B: abs(B - bpm)),
                           amp_mm=float(np.percentile(x, 97.5) - np.percentile(x, 2.5)),
                           mean_speed=float(v[a:b].mean()), peak_speed=float(np.percentile(v[a:b], 99)),
                           axis_motive=axis.tolist()))
    static = None
    if blocks:
        cand = [(t[a], t[b - 1]) for a, b in runs(inner & (t < blocks[0]["t0"]) & (vr < STILL_THR))]
        if cand:
            s0, s1 = max(cand, key=lambda r: r[1] - r[0])
            static = dict(t0=float(s0), t1=float(s1))
    return static, blocks


def segments(take_t, mc, sync):
    """タスクの区間 [(表示名, take_t0, take_t1)]：ボード・静止・テンポのブロック．"""
    windows = sorted((s["rs_t0"] + s["c"], s["rs_t1"] + s["c"]) for s in sync["segments"])
    segs = [("board", a, b) for a, b in windows]
    names = list(mc["names"])
    if "wrist" in names:
        static, blocks = detect_blocks(mc["t"], mc["P_M"][:, names.index("wrist")], windows)
        if static:
            segs.append(("still", static["t0"], static["t1"]))
        segs += [(str(b["nominal_bpm"]), b["t0"], b["t1"]) for b in blocks]
    return sorted(segs, key=lambda s: s[1])


def rolling_median(rel, e, win_s=1.0):
    s = pd.Series(e, index=pd.to_timedelta(rel, unit="s"))
    out = s.rolling(f"{int(win_s * 1000)}ms", center=True, min_periods=5).median().to_numpy().copy()
    out[~np.isfinite(e)] = np.nan                              # 取れなかった所は線を切る
    return out


def main():
    ap = argparse.ArgumentParser(description="rs_joints の認識と誤差を時刻の図にまとめる")
    ap.add_argument("trial", nargs="?", help="trial_* フォルダ（省略すると最新）")
    args = ap.parse_args()

    trial_dir = io_utils.resolve_session_dir(args.trial, config.TRIAL_PREFIX)
    rs = dict(np.load(trial_dir / "joints" / "rs_joints.npz"))
    mc_path = trial_dir / "joints" / "mc_joints.npz"
    mc = dict(np.load(mc_path)) if mc_path.is_file() else None
    sync, _, T_MC = sync_io.load_sync(trial_dir)
    t = rs["t"]
    rel = t - t[0]
    names = [str(n) for n in rs["names"]]
    ok = rs["status"] == 0 if "status" in rs else np.isfinite(rs["P_C"][:, :, 0])
    have_mc = mc is not None and sync is not None

    # 比べる関節（MoCap がなければ RS の全関節を上段だけに）
    pairs = [(a, b) for a, b in config.COMPARE_PAIRS if a in names and have_mc and b in list(mc["names"])] \
        if have_mc else [(a, None) for a in names]
    pairs = pairs[:len(SERIES)]
    segs, seg_of = [], np.full(len(t), "", dtype=object)
    errs = {}
    if have_mc:
        take_t = t + sync_io.take_offset(sync, t)
        segs = segments(take_t, mc, sync)
        for name, a, b in segs:
            seg_of[(take_t >= a) & (take_t <= b)] = name
        T_CM = np.linalg.inv(T_MC)
        mn = list(mc["names"])
        task = (seg_of != "") & (seg_of != "board")
        for a, b in pairs:
            Q = apply(T_CM, interp_nan(take_t, mc["t"], mc["P_M"][:, mn.index(b)], config.CMP_MAX_GAP))
            E = rs["P_C"][:, names.index(a)] - Q
            good = np.isfinite(E).all(axis=1)
            bias = np.median(E[good & task], axis=0) if (good & task).any() else np.zeros(3)
            errs[(a, b)] = np.linalg.norm(E - bias, axis=1)

    # ── 図 ────────────────────────────────────────
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    n_rows = 2 if have_mc else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.0 + 0.35 * len(pairs) + (4.2 if have_mc else 0)),
                             sharex=True, layout="constrained", gridspec_kw=dict(height_ratios=[0.35 * len(pairs) + 0.6, 4.2][:n_rows]))
    axes = np.atleast_1d(axes)
    x = lambda tt: np.interp(tt, take_t, rel)                 # Take 時刻 → 記録の秒

    for ax in axes:
        for name, a, b in segs:
            ax.axvspan(x(a), x(b), color=BAND, lw=0, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    # 上段: 取れなかった時刻
    ax = axes[0]
    for i, (a, _) in enumerate(pairs):
        miss = ~ok[:, names.index(a)]
        ax.scatter(rel[miss], np.full(miss.sum(), i), marker="|", s=90, color=FAIL, lw=1.2, zorder=2)
        pct = 100 * ok[:, names.index(a)].mean()
        ax.text(1.005, i, f"{pct:.0f}%", transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([a for a, _ in pairs])
    ax.set_ylim(len(pairs) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("not recognized (red ticks)  —  % = recognized over the whole recording", loc="left",
                 fontsize=10, color=INK, pad=22)
    for name, a, b in segs:                                   # タスクの区間の名前（帯の上）
        ax.text((x(a) + x(b)) / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color=INK2)

    # 下段: 誤差
    if have_mc:
        ax = axes[1]
        ys = {a: rolling_median(rel, e) for (a, b), e in errs.items()}
        # 縦軸: 0–BREAK_MM は等間隔，それより上は圧縮（上 1/5 に BREAK_MM〜最大値を詰める）
        ymax = max(np.nanmax(y) for y in ys.values() if np.isfinite(y).any())
        if ymax > BREAK_MM * 1.1:
            top = TOP_MM
            ys = {a: np.minimum(y, top) for a, y in ys.items()}   # 200 mm 以上は上端に描く
            k = (BREAK_MM / 4) / (top - BREAK_MM)                # 圧縮率（上の部分は 1 mm が k 倍に縮む）
            fwd = lambda y: np.where(y <= BREAK_MM, y, BREAK_MM + (y - BREAK_MM) * k)
            inv = lambda z: np.where(z <= BREAK_MM, z, BREAK_MM + (z - BREAK_MM) / k)
            ax.set_yscale("function", functions=(fwd, inv))
            ax.set_ylim(0, top)
            ticks = list(range(0, int(BREAK_MM) + 1, 10)) + [100, int(top)]
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(v) for v in ticks[:-1]] + [f"≥{int(top)}"])
            ax.axhspan(BREAK_MM, top, color="#f6f5f2", lw=0, zorder=0.5)   # 圧縮した範囲は薄く塗る
            ax.axhline(BREAK_MM, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
            ax.text(1.005, 75, f"compressed\n(> {BREAK_MM:.0f} mm)",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK2)
        else:
            ax.set_ylim(0, max(40, 1.15 * ymax))
        for i, (a, y) in enumerate(ys.items()):
            ax.plot(rel, y, lw=1.8, color=SERIES[i], label=a, zorder=3)
        ax.set_ylabel("3D error after bias [mm]\n(1 s median)")
        ax.grid(axis="y", color="#dddcd8", lw=0.6)
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=len(errs), frameon=False, fontsize=9)
        ax.set_title("error vs MoCap", loc="left", fontsize=10, color=INK)
    axes[-1].set_xlabel("time from recording start [s]")
    axes[-1].set_xlim(0, rel[-1])
    out = trial_dir / "joints" / "timeline.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"[DONE] -> {out}")

    # ── 区間ごとの表 ───────────────────────────────
    if have_mc:
        rows = []
        order = [s[0] for s in segs] + ["(other)"]
        lab = np.where(seg_of == "", "(other)", seg_of)
        for (a, b), e in errs.items():
            for name in dict.fromkeys(order):
                m = lab == name
                if m.any():
                    g = m & np.isfinite(e)
                    rows.append(dict(segment=name, joint=a, valid_pct=100 * ok[m, names.index(a)].mean(),
                                     rmse_mm=float(np.sqrt(np.mean(e[g] ** 2))) if g.any() else np.nan))
        df = pd.DataFrame(rows)
        df.to_csv(trial_dir / "joints" / "timeline_blocks.csv", index=False, float_format="%.1f")
        print("\n区間ごとの 認識率 [%] / 偏りを引いた RMSE [mm]   （(other) = 準備・移動など区間の外）")
        joints = list(dict.fromkeys(df.joint))
        print(f"  {'区間':8s}" + "".join(f"{j:>16s}" for j in joints))
        for name in dict.fromkeys(df.segment):
            s = df[df.segment == name].set_index("joint")
            print(f"  {name:8s}" + "".join(f"{s.valid_pct[j]:7.0f} / {s.rmse_mm[j]:5.1f} " for j in joints))


if __name__ == "__main__":
    main()