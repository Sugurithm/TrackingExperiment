"""slow seed 2 のターゲット軌道の全体（机の上のマウスの道筋）を PDF に出す．タイトルなし

  python tasks/plot_slow2.py                      # → tasks/slow2_trajectory.pdf
  python tasks/plot_slow2.py --no-ramp            # 立ち上がり（灰色）を描かない
  python tasks/plot_slow2.py --out figs/slow2.pdf
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from task_config import CFG
from targets import make_trajectory, target_position


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="slow seed 2 の軌道の全体を PDF に出す")
    ap.add_argument("--band", default="slow")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default=str(here / "slow2_trajectory.pdf"), help="保存先（.pdf）")
    ap.add_argument("--no-ramp", action="store_true", help="立ち上がり（t < 0）を描かない")
    args = ap.parse_args()

    tr = make_trajectory(args.band, args.seed)
    t = np.arange(-tr["ramp_s"], tr["T"], 0.005)
    x, y = target_position(tr, t)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    if not args.no_ramp:
        ax.plot(x[t < 0], y[t < 0], color="0.75", lw=1.0)
    ax.plot(x[t >= 0], y[t >= 0], color="#2a78d6", lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlim(-CFG["amp_cm"]["x"] * 1.05, CFG["amp_cm"]["x"] * 1.05)
    ax.set_ylim(-CFG["amp_cm"]["y"] * 1.05, CFG["amp_cm"]["y"] * 1.05)
    ax.set_xlabel("mouse x [cm]")
    ax.set_ylabel("mouse y [cm]")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
