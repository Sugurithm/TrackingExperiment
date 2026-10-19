"""擬似乱数ターゲット軌道（正弦波の和）と試行の計画表

軌道: p(t) = Σ_k a_k sin(2π f_k t + φ_k)
  - f_k = n_k / T（n_k は素数）→ 互いに整数倍でなく周期を覚えられない。解析区間 [0, T) でちょうど周期が閉じる
  - x と y は別の素数 → FFT で軸ごとに分離できる
  - a_k ∝ f_k^(-amp_exponent)（既定 0.5）。全体を最大振幅 amp_cm に合わせる
  - φ_k は seed から決まる。同じ seed は誰がやっても同じ軌道
  - t < 0 は立ち上がり（振幅を 0→1 に滑らかに上げる。t = −ramp で画面中央）

単体実行で軌道の一覧（最大の速さなど）と図を出す:  python aim_task/targets.py
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
    """試行の順番。帯域の順は参加者番号の偶奇で入れ替え、帯域内の seed 2–5 は参加者ごとに並べ替える。
    seed_main[0] は帯域の最初と最後に置く（再現性と慣れの確認）"""
    pid = int("".join(c for c in str(participant) if c.isdigit()) or 0)
    rng = np.random.default_rng(pid)
    bands = ["slow", "fast"] if pid % 2 == 0 else ["fast", "slow"]
    plan = []
    for b in bands:
        for s in cfg["seeds_practice"]:
            plan.append({"band": b, "seed": s, "practice": True, "repeat": 0})
    for b in bands:
        s0, rest = cfg["seeds_main"][0], list(cfg["seeds_main"][1:])
        rng.shuffle(rest)
        seq = [s0] + rest + [s0]
        for i, s in enumerate(seq):
            plan.append({"band": b, "seed": int(s), "practice": False,
                         "repeat": 1 if (s == s0 and i == len(seq) - 1) else 0})
    for i, p in enumerate(plan):
        p["index"] = i
    return plan


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    fig, axs = plt.subplots(2, 5, figsize=(16, 6.5))
    for r, band in enumerate(["slow", "fast"]):
        for c, seed in enumerate(CFG["seeds_main"]):
            tr = make_trajectory(band, seed)
            st = trajectory_stats(tr)
            rows.append((band, seed, st))
            t = np.arange(-tr["ramp_s"], tr["T"], 0.005)
            x, y = target_position(tr, t)
            ax = axs[r, c]
            ax.plot(x[t < 0], y[t < 0], color="0.75", lw=0.8)
            ax.plot(x[t >= 0], y[t >= 0], lw=0.8)
            ax.set_aspect("equal")
            ax.set_xlim(-CFG["amp_cm"]["x"] * 1.05, CFG["amp_cm"]["x"] * 1.05)
            ax.set_ylim(-CFG["amp_cm"]["y"] * 1.05, CFG["amp_cm"]["y"] * 1.05)
            ax.set_title(f"{band} seed {seed}\npeak {st['peak_speed_cm_s']:.0f} cm/s", fontsize=9)
    fig.supxlabel("mouse x [cm]")
    fig.supylabel("mouse y [cm]")
    fig.tight_layout()
    from pathlib import Path
    out = Path(__file__).resolve().parent / "trajectories.png"
    fig.savefig(out, dpi=120)
    print(f"{'band':5} {'seed':>4} {'peak cm/s':>10} {'mean cm/s':>10} {'path cm':>9} {'x range':>8} {'y range':>8}")
    for b, s, st in rows:
        print(f"{b:5} {s:4d} {st['peak_speed_cm_s']:10.1f} {st['mean_speed_cm_s']:10.1f} "
              f"{st['path_cm']:9.0f} {st['x_range_cm']:8.1f} {st['y_range_cm']:8.1f}")
    print("plan P01:", [(p['band'], p['seed']) for p in build_plan('P01')])
    print(f"saved {out}")
