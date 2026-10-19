"""自動プレイ（記録・解析・カメラ連携の確認用．実験には使わない）

視覚の遅れ delay [s] + 1 次遅れ tau [s] + 運動ノイズでターゲットを追い，マウスのカウントを返す．
lead > 0 なら予告線を見て lead 秒先のターゲットを目標にする．
"""
import math

import numpy as np


class SimPlayer:
    def __init__(self, delay=0.15, tau=0.12, noise=0.3, seed=0, traj_lead=0.0):
        self.delay, self.tau, self.noise = delay, tau, noise
        self.lead = traj_lead
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.hist = []
        self.rem = np.zeros(2)

    def step(self, t, tx, ty, cur, dt, cm_per_count):
        self.hist.append((t, tx, ty))
        td = t - self.delay
        while len(self.hist) > 2 and self.hist[1][0] <= td:
            self.hist.pop(0)       # 遅れより古い分は捨てる
        (t0, x0, y0), (t1, x1, y1) = self.hist[0], self.hist[min(1, len(self.hist) - 1)]
        u = 0.0 if t1 == t0 else min(max((td - t0) / (t1 - t0), 0.0), 1.0)
        goal = np.array([x0 + u * (x1 - x0), y0 + u * (y1 - y0)])
        move = (goal - cur) * (1 - math.exp(-dt / self.tau)) + self.rng.normal(0, self.noise, 2) * math.sqrt(dt)
        c = move / cm_per_count + self.rem
        ci = np.round(c)
        self.rem = c - ci
        return int(ci[0]), int(ci[1]), 1
