"""トラッキング課題の解析（マウスのみ。RealSense の関節は同じ関数で後から足せる）

使い方:
  python aim_task/analyze_tracking.py                                  # recordings/aim_* をすべて
  python aim_task/analyze_tracking.py recordings/aim_P01_xxx recordings/aim_P02_xxx
  -> recordings/analysis_aim/ に trials.csv, freq_response.csv, freq_response_mean.csv,
     icc.csv, bode.png, repeat.png

各セッションの session.json から，トラッキングの試行（R でやり直した試行は最後の 1 回だけ）を読む．

試行ごと（練習は除く）・軸ごとに、解析区間 [0, T) を一定間隔に補間して FFT し、
軌道に含めた周波数（1/T の素数倍）で「カーソル ÷ ターゲット」の
  ゲイン（1 = ぴったり追う）、位相 [deg]、位相から求めた遅れ [ms]
を求める。区間でちょうど周期が閉じるので窓関数は要らない（漏れがない）。
加えて、軌道に含めていない周波数の成分の割合（remnant：ターゲットと無関係な動き＝ノイズ・修正運動）、
他方の軸の周波数への漏れ（x を追うときに y がどれだけ動くか）を出す。

個人差の確認: 指標ごとに ICC(1)（＝人の間の分散 /（人の間 + 人の中の分散））を帯域ごとに計算する。
1 に近いほど「人による違いが試行間のばらつきより大きく、安定している」。

関節の寄与（RealSense を合わせたあと）:
  freq_response(t, joint_disp, t, mouse_disp, T, n) で、マウスの変位に対する
  各関節（例: 手首点の机上の変位）の比を周波数ごとに出せる（multi-segment の寄与の割合）。
"""
import argparse
import glob
import json
import os

import numpy as np

FS = 100.0  # 解析の補間周波数 [Hz]


# ---------------- 周波数応答 ----------------
def resample(t, y, T, fs=FS):
    tt = np.arange(0, T, 1 / fs)
    return tt, np.interp(tt, t, y)


def spectrum(y):
    return np.fft.rfft(y) / len(y) * 2


def freq_response(t_out, y_out, t_in, y_in, T, n, fs=FS):
    """y_out / y_in の周波数応答を、基本周波数 1/T の n 倍の周波数で返す。
    戻り値: dict(f_hz, gain, phase_deg, delay_ms, coh_like)"""
    _, a = resample(t_out, y_out, T, fs)
    _, b = resample(t_in, y_in, T, fs)
    A, B = spectrum(a - a.mean()), spectrum(b - b.mean())
    n = np.asarray(n, int)
    H = A[n] / B[n]
    f = n / T
    ph = np.angle(H)
    return {"f_hz": f, "gain": np.abs(H), "phase_deg": np.degrees(ph),
            "delay_ms": -ph / (2 * np.pi * f) * 1000}


def remnant_ratio(t, y, T, n_excited, fmax=2.0, fs=FS):
    """0 < f ≤ fmax のパワーのうち、軌道に含めていない周波数の割合"""
    _, a = resample(t, y, T, fs)
    P = np.abs(spectrum(a - a.mean())) ** 2
    k = np.arange(len(P))
    band = (k > 0) & (k / T <= fmax)
    exc = np.isin(k, n_excited)
    return float(P[band & ~exc].sum() / P[band].sum())


def lag_xcorr_ms(t, v_out, v_in, T, max_ms=600, fs=FS):
    """速さの相互相関のピークの遅れ [ms]（正 = カーソルが遅れる）"""
    _, a = resample(t, v_out, T, fs)
    _, b = resample(t, v_in, T, fs)
    a, b = a - a.mean(), b - b.mean()
    L = int(max_ms / 1000 * fs)
    lags = np.arange(-L, L + 1)
    c = np.array([np.dot(a[max(l, 0):len(a) + min(l, 0)], b[max(-l, 0):len(b) + min(-l, 0)]) for l in lags])
    i = int(np.argmax(c))
    if 0 < i < len(c) - 1:   # 放物線補間
        d = (c[i - 1] - c[i + 1]) / (2 * (c[i - 1] - 2 * c[i] + c[i + 1]))
    else:
        d = 0.0
    return float((lags[i] + d) / fs * 1000)


# ---------------- 読み込み ----------------
def load_session(d):
    """session.json → (session, [(試行の情報, task.csv の配列), ...])．
    やり直した試行（redone = True）は除き，トラッキングの試行だけを返す．軌道は試行の meta.json から読む．"""
    sess = json.load(open(os.path.join(d, "session.json"), encoding="utf-8"))
    trials = []
    for tr in sess["trials"]:
        if tr.get("kind") != "track" or tr.get("redone"):
            continue
        tdir = os.path.join(d, tr["dir"])
        meta = json.load(open(os.path.join(tdir, "meta.json"), encoding="utf-8"))
        a = np.genfromtxt(os.path.join(tdir, "task.csv"), delimiter=",", names=True)
        trials.append(({**tr, "trajectory": meta["task"]["trajectory"]}, a))
    return sess, trials


def analyze_trial(tr, a):
    traj = tr["trajectory"]
    T = traj["T"]
    m = (a["t"] >= -0.2) & (a["t"] < T + 0.2)
    t = a["t"][m]
    out = {"participant": tr.get("participant"), "band": tr["band"], "seed": tr["seed"],
           "index": tr["index"], "repeat": tr["repeat"]}
    w = (t >= 0) & (t < T)
    ex, ey = a["cursor_x_cm"][m] - a["target_x_cm"][m], a["cursor_y_cm"][m] - a["target_y_cm"][m]
    out["rms_err_cm"] = float(np.sqrt(np.mean(ex[w] ** 2 + ey[w] ** 2)))
    out["on_target_pct"] = float(100 * a["on_target"][m][w].mean())
    fr_rows = []
    for ax, other in (("x", "y"), ("y", "x")):
        n = traj["axes"][ax]["n"]
        tgt, cur = a[f"target_{ax}_cm"][m], a[f"cursor_{ax}_cm"][m]
        r = freq_response(t, cur, t, tgt, T, n)
        for i in range(len(n)):
            fr_rows.append({**{k: out[k] for k in ("participant", "band", "seed", "index", "repeat")},
                            "axis": ax, "n": n[i], "f_hz": r["f_hz"][i], "gain": r["gain"][i],
                            "phase_deg": r["phase_deg"][i], "delay_ms": r["delay_ms"][i]})
        out[f"gain_mean_{ax}"] = float(np.mean(r["gain"]))
        out[f"gain_low_{ax}"] = float(r["gain"][0])
        out[f"gain_high_{ax}"] = float(r["gain"][-1])
        out[f"delay_low_{ax}_ms"] = float(r["delay_ms"][0])
        out[f"delay_high_{ax}_ms"] = float(r["delay_ms"][-1])
        out[f"remnant_{ax}"] = remnant_ratio(t, cur, T, n)
        # 他方の軸の周波数への漏れ: 他方のターゲットに対するこの軸のカーソルの「ゲイン」
        n_o = traj["axes"][other]["n"]
        cross = freq_response(t, cur, t, a[f"target_{other}_cm"][m], T, n_o)
        out[f"cross_{other}to{ax}"] = float(np.mean(cross["gain"]))
        v_c = np.gradient(cur, t)
        v_t = np.gradient(tgt, t)
        out[f"lag_xcorr_{ax}_ms"] = lag_xcorr_ms(t, v_c, v_t, T)
    return out, fr_rows


# ---------------- 個人差 ----------------
def icc1(values, groups):
    """一元配置の ICC(1)。groups ごとの試行数が違っても可（平均の n を使う）"""
    values, groups = np.asarray(values, float), np.asarray(groups)
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    ug = np.unique(groups)
    k = len(ug)
    if k < 2:
        return np.nan, np.nan, np.nan
    ns = np.array([np.sum(groups == g) for g in ug])
    means = np.array([values[groups == g].mean() for g in ug])
    grand = values.mean()
    N = len(values)
    msb = np.sum(ns * (means - grand) ** 2) / (k - 1)
    msw = sum(np.sum((values[groups == g] - values[groups == g].mean()) ** 2) for g in ug) / (N - k)
    n0 = (N - np.sum(ns ** 2) / N) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    return float(icc), float(np.sqrt(max(msb - msw, 0) / n0)), float(np.sqrt(msw))


def write_csv(path, rows):
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})


# ---------------- 図 ----------------
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def plot_bode(fr, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parts = sorted({r["participant"] for r in fr})
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, ax_name in enumerate(["x", "y"]):
        for pi, p in enumerate(parts):
            c = COLORS[pi % len(COLORS)]
            for band, ls in (("slow", "--"), ("fast", "-")):
                rows = [r for r in fr if r["participant"] == p and r["axis"] == ax_name and r["band"] == band]
                if not rows:
                    continue
                f = sorted({r["f_hz"] for r in rows})
                g = [[r["gain"] for r in rows if r["f_hz"] == fi] for fi in f]
                d = [[r["phase_deg"] for r in rows if r["f_hz"] == fi] for fi in f]
                gm, gs = [np.mean(x) for x in g], [np.std(x) for x in g]
                dm, ds = [np.mean(x) for x in d], [np.std(x) for x in d]
                lab = f"{p} {band}"
                axs[0, j].errorbar(f, gm, yerr=gs, color=c, ls=ls, lw=2, marker="o", ms=5, capsize=2, label=lab)
                axs[1, j].errorbar(f, dm, yerr=ds, color=c, ls=ls, lw=2, marker="o", ms=5, capsize=2, label=lab)
        axs[0, j].set_title(f"axis {ax_name}")
        axs[0, j].axhline(1, color="0.6", lw=1)
        axs[1, j].axhline(0, color="0.6", lw=1)
        axs[1, j].set_xlabel("frequency [Hz]")
        for a in axs[:, j]:
            a.set_xscale("log")
            a.grid(True, color="0.9", lw=0.8)
            a.spines[["top", "right"]].set_visible(False)
    axs[0, 0].set_ylabel("gain (cursor / target)")
    axs[1, 0].set_ylabel("phase [deg]")
    axs[0, 1].legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Frequency response per participant (mean ± SD over trials; dashed = slow, solid = fast)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_repeat(trials, out_png):
    """同じ軌道（seed_main[0]）の最初と最後、と全試行の RMS 誤差"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parts = sorted({r["participant"] for r in trials})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for bi, band in enumerate(["slow", "fast"]):
        ax = axs[bi]
        for pi, p in enumerate(parts):
            rows = sorted([r for r in trials if r["participant"] == p and r["band"] == band], key=lambda r: r["index"])
            if not rows:
                continue
            c = COLORS[pi % len(COLORS)]
            x = np.arange(len(rows))
            ax.plot(x, [r["rms_err_cm"] for r in rows], color=c, lw=2, marker="o", ms=6, label=p)
            rep = [i for i, r in enumerate(rows) if r["seed"] == rows[0]["seed"]]
            ax.plot(rep, [rows[i]["rms_err_cm"] for i in rep], ls="none", marker="o", ms=11,
                    mfc="none", mec=c, mew=2)
        ax.set_title(f"{band}: RMS error per trial (ring = same trajectory, first & last)")
        ax.set_xlabel("trial order in band")
        ax.set_ylabel("RMS error [cm]")
        ax.grid(True, color="0.9", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axs[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="aim_* フォルダ（またはそれを含むフォルダ）．省略すると recordings/aim_*")
    ap.add_argument("--out", default=None, help="出力先（省略すると recordings/analysis_aim）")
    ap.add_argument("--include-practice", action="store_true")
    args = ap.parse_args()
    from task_config import out_root
    root = str(out_root())
    args.out = args.out or os.path.join(root, "analysis_aim")
    dirs = []
    for p in (args.paths or [root]):
        if os.path.exists(os.path.join(p, "session.json")):
            dirs.append(p)
        else:
            dirs += sorted(os.path.dirname(m) for m in glob.glob(os.path.join(p, "aim_*", "session.json")))
    if not dirs:
        raise SystemExit("[ERROR] session.json のあるフォルダが見つかりません")
    os.makedirs(args.out, exist_ok=True)
    trials, fr = [], []
    for d in dirs:
        meta, ts = load_session(d)
        for tr, a in ts:
            if tr["practice"] and not args.include_practice:
                continue
            tr = {**tr, "participant": meta["participant"]}
            o, rows = analyze_trial(tr, a)
            o["session"] = os.path.basename(d.rstrip("/"))
            trials.append(o)
            fr += rows
        print(f"{d}: {sum(1 for t in trials if t['session'] == os.path.basename(d.rstrip('/')))} trials")
    write_csv(os.path.join(args.out, "trials.csv"), trials)
    write_csv(os.path.join(args.out, "freq_response.csv"), fr)

    # 周波数ごとの平均（参加者 × 帯域 × 軸 × 周波数）
    agg = []
    for key in sorted({(r["participant"], r["band"], r["axis"], r["n"]) for r in fr}):
        rows = [r for r in fr if (r["participant"], r["band"], r["axis"], r["n"]) == key]
        g = np.array([r["gain"] for r in rows])
        dl = np.array([r["delay_ms"] for r in rows])
        agg.append({"participant": key[0], "band": key[1], "axis": key[2], "f_hz": rows[0]["f_hz"],
                    "gain_mean": g.mean(), "gain_sd": g.std(ddof=1) if len(g) > 1 else np.nan,
                    "delay_ms_mean": dl.mean(), "delay_ms_sd": dl.std(ddof=1) if len(dl) > 1 else np.nan,
                    "n_trials": len(rows)})
    write_csv(os.path.join(args.out, "freq_response_mean.csv"), agg)

    # ICC
    metrics = [k for k in trials[0] if k not in ("participant", "band", "seed", "index", "repeat", "session")]
    icc_rows = []
    for band in ("slow", "fast"):
        sub = [t for t in trials if t["band"] == band]
        if len({t["participant"] for t in sub}) < 2:
            continue
        for mname in metrics:
            icc, sb, sw = icc1([t[mname] for t in sub], [t["participant"] for t in sub])
            icc_rows.append({"band": band, "metric": mname, "icc1": icc, "sd_between": sb, "sd_within": sw,
                             "n_participants": len({t['participant'] for t in sub}), "n_trials": len(sub)})
    write_csv(os.path.join(args.out, "icc.csv"), icc_rows)

    plot_bode(fr, os.path.join(args.out, "bode.png"))
    plot_repeat(trials, os.path.join(args.out, "repeat.png"))

    # 画面への要約
    print(f"\n{'participant':12} {'band':5} {'RMS cm':>7} {'on %':>6} {'gain x lo/hi':>13} {'delay x lo/hi ms':>17} {'xcorr lag x':>11}")
    for p in sorted({t["participant"] for t in trials}):
        for band in ("slow", "fast"):
            s = [t for t in trials if t["participant"] == p and t["band"] == band]
            if not s:
                continue
            mean = lambda k: np.mean([t[k] for t in s])
            print(f"{p:12} {band:5} {mean('rms_err_cm'):7.2f} {mean('on_target_pct'):6.1f} "
                  f"{mean('gain_low_x'):6.2f}/{mean('gain_high_x'):5.2f} "
                  f"{mean('delay_low_x_ms'):8.0f}/{mean('delay_high_x_ms'):6.0f} {mean('lag_xcorr_x_ms'):11.0f}")
    if icc_rows:
        print("\nICC(1)（主な指標）")
        for r in icc_rows:
            if r["metric"] in ("rms_err_cm", "gain_high_x", "delay_high_x_ms", "remnant_x", "lag_xcorr_x_ms"):
                print(f"  {r['band']:5} {r['metric']:18} {r['icc1']:6.2f}  (SD between {r['sd_between']:.3g}, within {r['sd_within']:.3g})")
    print(f"\nsaved {args.out}/")


if __name__ == "__main__":
    main()
