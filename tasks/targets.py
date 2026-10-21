"""擬似乱数ターゲット軌道（正弦波の和）と試行の計画表

軌道: p(t) = Σ_k a_k sin(2π f_k t + φ_k)
  - f_k = n_k / T（n_k は素数）→ 互いに整数倍でなく周期を覚えられない。解析区間 [0, T) でちょうど周期が閉じる
  - x と y は別の素数 → FFT で軸ごとに分離できる
  - a_k ∝ f_k^(-amp_exponent)（既定 0.5）。全体を最大振幅 amp_cm に合わせる
  - φ_k は seed から決まる。同じ seed は誰がやっても同じ軌道
  - t < 0 は立ち上がり（振幅を 0→1 に滑らかに上げる。t = −ramp で画面中央）

単体で実行すると，軌道を作って画像に出し，一覧（最大の速さなど）を表示する:
  python task/targets.py                    # 本番の軌道（帯域 × seed）を 1 枚の図に → task/trajectories.png
  python task/targets.py --each             # 1 本ずつの図も出す（task/trajectories/<band>_s<seed>.png，時系列つき）
  python task/targets.py --practice         # 練習の軌道も図に入れる
  python task/targets.py --plan P01         # その参加者の試行の順番を表示する
  python task/targets.py --out figs/traj.png
"""
import numpy as np

from task_config import CFG


def make_trajectory(band, seed, cfg=CFG):
    T = cfg["trial_s"]
    rng = np.random.default_rng([seed, 0 if band == "slow" else 1])
    traj = {"band": band, "seed": int(seed), "T": T, "ramp_s": cfg["ramp_s"], "axes": {}}
    tt = np.arange(0, T, 0.001)
    for ax in ("x", "y"):
        n = np.array(cfg["bands"][band][ax], float)
        f = n / T
        a = f ** (-cfg["amp_exponent"])
        ph = rng.uniform(0, 2 * np.pi, len(f))
        p = (a[:, None] * np.sin(2 * np.pi * f[:, None] * tt + ph[:, None])).sum(0)
        a = a * cfg["amp_cm"][ax] / np.abs(p).max()
        traj["axes"][ax] = {"n": n.astype(int).tolist(), "f_hz": f.tolist(),
                            "amp_cm": a.tolist(), "phase_rad": ph.tolist()}
    return traj


def _ramp(t, R):
    t = np.asarray(t, float)
    w = np.ones_like(t)
    m = t < 0
    w[m] = 0.5 * (1 - np.cos(np.pi * np.clip(t[m] + R, 0, R) / R))
    return w


def target_position(traj, t):
    """t [s]（解析区間の先頭が 0、立ち上がりは負）→ (x_cm, y_cm)。y は上が正"""
    t = np.asarray(t, float)
    w = _ramp(t, traj["ramp_s"])
    out = []
    for ax in ("x", "y"):
        A = traj["axes"][ax]
        f, a, ph = (np.array(A[k]) for k in ("f_hz", "amp_cm", "phase_rad"))
        p = (a[:, None] * np.sin(2 * np.pi * f[:, None] * np.atleast_1d(t) + ph[:, None])).sum(0)
        out.append((w * p).reshape(t.shape))
    return out[0], out[1]


def trajectory_stats(traj):
    t = np.arange(0, traj["T"], 0.001)
    x, y = target_position(traj, t)
    v = np.hypot(np.gradient(x, t), np.gradient(y, t))
    return {"peak_speed_cm_s": float(v.max()), "mean_speed_cm_s": float(v.mean()),
            "path_cm": float(np.sum(np.hypot(np.diff(x), np.diff(y)))),
            "x_range_cm": float(x.max() - x.min()), "y_range_cm": float(y.max() - y.min())}


def build_plan(participant, cfg=CFG):
    """試行の順番。帯域の順は task_config の band_order（既定は全員 slow → fast）、
    帯域内の seed の順は seed_order（既定 "fixed" = 全員同じ。"shuffle" = seed 2–5 を参加者ごとに並べ替える）。
    seed_main[0] は帯域の最初と最後に置く（再現性と慣れの確認）"""
    pid = int("".join(c for c in str(participant) if c.isdigit()) or 0)
    rng = np.random.default_rng(pid)
    order = cfg.get("band_order", ["slow", "fast"])
    if order == "alternate":          # 参加者番号の偶奇で入れ替える（順番の効果を相殺したいとき）
        bands = ["slow", "fast"] if pid % 2 == 0 else ["fast", "slow"]
    else:
        bands = list(order)
    plan = []
    for b in bands:
        for s in cfg["seeds_practice"]:
            plan.append({"band": b, "seed": s, "practice": True, "repeat": 0})
    for b in bands:
        s0, rest = cfg["seeds_main"][0], list(cfg["seeds_main"][1:])
        if cfg.get("seed_order", "fixed") == "shuffle":
            rng.shuffle(rest)
        seq = [s0] + rest + ([s0] if cfg.get("repeat_first", True) else [])
        for i, s in enumerate(seq):
            plan.append({"band": b, "seed": int(s), "practice": False,
                         "repeat": 1 if (i > 0 and s == s0) else 0})
    for i, p in enumerate(plan):
        p["index"] = i
    return plan


def _plot_one(ax_xy, ax_t, tr):
    """1 本の軌道: 左 = 机の上のマウスの道筋，右 = x, y の時系列（灰色 = 立ち上がり）"""
    t = np.arange(-tr["ramp_s"], tr["T"], 0.005)
    x, y = target_position(tr, t)
    ax_xy.plot(x[t < 0], y[t < 0], color="0.75", lw=0.8)
    ax_xy.plot(x[t >= 0], y[t >= 0], color="#2a78d6", lw=0.8)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlim(-CFG["amp_cm"]["x"] * 1.05, CFG["amp_cm"]["x"] * 1.05)
    ax_xy.set_ylim(-CFG["amp_cm"]["y"] * 1.05, CFG["amp_cm"]["y"] * 1.05)
    if ax_t is not None:
        ax_t.axvspan(-tr["ramp_s"], 0, color="0.92")
        ax_t.plot(t, x, color="#2a78d6", lw=1.2, label="x")
        ax_t.plot(t, y, color="#eb6834", lw=1.2, label="y")
        ax_t.set_xlabel("t [s]")
        ax_t.set_ylabel("mouse [cm]")
        ax_t.legend(frameon=False, loc="upper right")
        ax_t.spines[["top", "right"]].set_visible(False)


def main():
    import argparse
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="軌道（正弦波の和）を作って画像に出す")
    ap.add_argument("--out", default=str(here / "trajectories.png"), help="一覧の図の保存先")
    ap.add_argument("--each", action="store_true", help="1 本ずつの図も出す（--out と同じ場所の trajectories/）")
    ap.add_argument("--practice", action="store_true", help="練習の軌道も図に入れる")
    ap.add_argument("--plan", metavar="ID", help="この参加者の試行の順番を表示する（例 P01）")
    args = ap.parse_args()

    seeds = list(CFG["seeds_main"]) + (list(CFG["seeds_practice"]) if args.practice else [])
    bands = ["slow", "fast"]
    rows = []
    fig, axs = plt.subplots(len(bands), len(seeds), figsize=(3.2 * len(seeds), 3.3 * len(bands)), squeeze=False)
    for r, band in enumerate(bands):
        for c, seed in enumerate(seeds):
            tr = make_trajectory(band, seed)
            st = trajectory_stats(tr)
            rows.append((band, seed, st))
            _plot_one(axs[r, c], None, tr)
            tag = " (practice)" if seed in CFG["seeds_practice"] else ""
            axs[r, c].set_title(f"{band} seed {seed}{tag}\npeak {st['peak_speed_cm_s']:.0f} cm/s", fontsize=9)
    fig.supxlabel("mouse x [cm]")
    fig.supylabel("mouse y [cm]")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")

    if args.each:
        d = out.parent / "trajectories"
        d.mkdir(exist_ok=True)
        for band, seed, st in rows:
            tr = make_trajectory(band, seed)
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1.8]})
            _plot_one(a1, a2, tr)
            a1.set_xlabel("mouse x [cm]")
            a1.set_ylabel("mouse y [cm]")
            fig.suptitle(f"{band} seed {seed}   peak {st['peak_speed_cm_s']:.0f} cm/s, "
                         f"mean {st['mean_speed_cm_s']:.1f} cm/s, path {st['path_cm']:.0f} cm", fontsize=10)
            fig.tight_layout()
            fig.savefig(d / f"{band}_s{seed}.png", dpi=110)
            plt.close(fig)
        print(f"saved {d}/ ({len(rows)} files)")

    print(f"\n{'band':5} {'seed':>4} {'peak cm/s':>10} {'mean cm/s':>10} {'path cm':>9} {'x range':>8} {'y range':>8}")
    for b, s, st in rows:
        print(f"{b:5} {s:4d} {st['peak_speed_cm_s']:10.1f} {st['mean_speed_cm_s']:10.1f} "
              f"{st['path_cm']:9.0f} {st['x_range_cm']:8.1f} {st['y_range_cm']:8.1f}")
    if args.plan:
        print(f"\nplan {args.plan}:")
        for p in build_plan(args.plan):
            print(f"  {p['index']:2d}  {'practice' if p['practice'] else 'main':8} {p['band']:4} seed {p['seed']}"
                  + ("  (repeat)" if p["repeat"] else ""))


if __name__ == "__main__":
    main()