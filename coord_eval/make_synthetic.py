"""答えが分かっている合成データを作る（coord_eval の指標が正しく復元できるかの検証用）

aim_eval/make_synthetic.py は成分の変位を直接足していたため、セグメント長が保たれず
「手の向き（ヨー）」が意味を持たなかった。ここでは**関節角で動く腕**（体幹の並進 2 + 肩・肘・手首の
水平面内の回転 3 = 5 自由度、セグメント長は一定）を重み付き逆運動学で動かして、
マウスのセンサの軌道を追わせる。参加者ごとに次の「型」を入れておく:

  w_lo / w_hi   低い周波数（大きくゆっくりした動き）/ 高い周波数での関節の使い分けの重み
                （同じなら振幅によらず同じ分担、違えば「大きい動きは肩、小さい動きは前腕」など）
  kappa         マウスの向きの安定化（0 = 手の向きは自由、1 = 前腕を逆に回して手の向きを一定に保つ）
  grip          握りの固さ（マウスの回転 = grip × 手の回転。1 = 手と一緒に回る）
  w_hi_800      800 DPI での高周波の重み（None = 400 と同じ = DPI で使い方を変えない）

  python coord_eval/make_synthetic.py <出力フォルダ>       # recordings と同じ形の aim_* を作る
"""
import argparse, json, os
import numpy as np
from scipy.signal import butter, filtfilt

FS_TASK = 240.0
FS_RS = 30.0
T = 40.0
RAMP = 3.0
BANDS = {"slow": {"x": [2, 5, 11, 17], "y": [3, 7, 13, 19]},
         "fast": {"x": [3, 7, 13, 19, 29, 41], "y": [5, 11, 17, 23, 31, 43]}}
AMP_CM = {"x": 12.0, "y": 6.0}
HAND_SCALE = {400: 0.85, 800: 0.425}     # 画面の道筋 → 机の上の手の移動量（DPI に反比例）
TAU_HUMAN = 0.20
REMNANT_MM = 1.5
TAU_CAM = 0.03
CAM_GAIN = 0.88                          # カメラが動きを小さめに捉える（実データのゲイン 0.8 前後）
JITTER = {"shoulder": 3.8, "elbow": 3.1, "wrist": 1.9, "wrist_pose": 1.8, "mcp3": 2.2}
GEOM = {"S0": np.array([170.0, -380.0, 330.0]), "L1": 290.0, "e1": np.radians(55), "L2": 250.0,
        "e2": np.radians(8), "L3": 68.0, "sensor": 0.6, "a0": np.radians(15), "b0": np.radians(-40),
        "p0": np.radians(-10)}
FC_SPLIT = 0.22                          # 低い / 高い周波数の境 [Hz]

# 参加者の型（重みは「手先の動きの何割をその自由度で作るか」の目安。順に 体幹 x,y・肩・肘・手首）
PERSONAS = {
    "SynWhole":   dict(group="FPS", w_lo=[.06, .06, .36, .42, .10], w_hi=[.06, .06, .36, .42, .10], kappa=0.0, grip=1.0),
    "SynForearm": dict(group="FPS", w_lo=[.02, .02, .14, .75, .07], w_hi=[.02, .02, .14, .75, .07], kappa=1.0, grip=1.0),
    "SynStab":    dict(group="FPS", w_lo=[.06, .06, .36, .42, .10], w_hi=[.06, .06, .36, .42, .10], kappa=0.7, grip=1.0),
    "SynEcon":    dict(group="nonFPS", w_lo=[.20, .20, .45, .12, .03], w_hi=[.01, .01, .08, .72, .18], kappa=0.0, grip=1.0),
    "SynShift":   dict(group="nonFPS", w_lo=[.06, .06, .36, .42, .10], w_hi=[.06, .06, .36, .42, .10], kappa=0.0, grip=1.0,
                       w_800=[.03, .03, .20, .34, .40]),
    "SynLoose":   dict(group="nonFPS", w_lo=[.02, .02, .14, .67, .15], w_hi=[.02, .02, .14, .67, .15], kappa=0.0, grip=0.3),
}


def traj(band, seed):
    rng = np.random.default_rng([seed, 0 if band == "slow" else 1])
    tt = np.arange(0, T, 0.001)
    out = {"band": band, "seed": int(seed), "T": T, "ramp_s": RAMP, "axes": {}}
    for ax in ("x", "y"):
        n = np.array(BANDS[band][ax], float); f = n / T
        a = f ** -0.5; ph = rng.uniform(0, 2 * np.pi, len(f))
        p = (a[:, None] * np.sin(2 * np.pi * f[:, None] * tt + ph[:, None])).sum(0)
        a = a * AMP_CM[ax] / np.abs(p).max()
        out["axes"][ax] = {"n": n.astype(int).tolist(), "f_hz": f.tolist(), "amp_cm": a.tolist(),
                           "phase_rad": ph.tolist()}
    return out


def eval_traj(tj, t, tau=0.0):
    out = []
    for ax in ("x", "y"):
        A = tj["axes"][ax]
        f, a, ph = (np.array(A[k]) for k in ("f_hz", "amp_cm", "phase_rad"))
        out.append((a[:, None] * 10 * np.sin(2 * np.pi * f[:, None] * (t - tau)[None] + ph[:, None])).sum(0))
    return np.stack(out, 1)


u = lambda a: np.stack([np.sin(a), np.cos(a)], -1)          # 方位 a（0 = +y、+ = +x 側）の単位ベクトル
n_ = lambda a: np.stack([np.cos(a), -np.sin(a)], -1)        # du/da


def lengths():
    g = GEOM
    return g["L1"] * np.cos(g["e1"]), g["L2"] * np.cos(g["e2"]), g["L3"] * g["sensor"]


def sensor_pos(q):
    L1h, L2h, Ls = lengths()
    return GEOM["S0"][:2] + q[:2] + L1h * u(q[2]) + L2h * u(q[3]) + Ls * u(q[4])


def ik_step(q, v, w, kappa):
    """センサの速度 v (2,) を作る関節速度。重み付き擬似逆行列 + 手の向きの安定化（kappa）。"""
    L1h, L2h, Ls = lengths()
    J = np.column_stack([[1, 0], [0, 1], L1h * n_(q[2]), L2h * n_(q[3]), Ls * n_(q[4])])
    lev = np.array([1, 1, L1h, L2h, Ls])
    W = np.diag(np.asarray(w) / lev ** 2)
    dq_free = W @ J.T @ np.linalg.solve(J @ W @ J.T, v)
    if kappa <= 0:
        return dq_free
    Ja = np.vstack([J, [0, 0, 0, 0, 1]])                    # 追加の条件: 手の向きを変えない
    dq_stab = W @ Ja.T @ np.linalg.solve(Ja @ W @ Ja.T, np.r_[v, 0.0])
    return (1 - kappa) * dq_free + kappa * dq_stab


def simulate(path, fs, p, dpi):
    """センサの道筋 path (N,2) [mm] → 関節角の時系列 q (N,5)。"""
    b, a = butter(2, FC_SPLIT / (fs / 2))
    lo = filtfilt(b, a, path, axis=0)
    hi = path - lo
    vlo, vhi = np.gradient(lo, axis=0) * fs, np.gradient(hi, axis=0) * fs
    w_lo = p["w_lo"]
    w_hi = p.get("w_800") if (dpi == 800 and p.get("w_800")) else p["w_hi"]
    if dpi == 800 and p.get("w_800"):
        w_lo = p["w_800"]
    q = np.zeros((len(path), 5))
    q[0] = [0, 0, GEOM["a0"], GEOM["b0"], GEOM["p0"]]
    off = sensor_pos(q[0]) - path[0]
    for i in range(len(path) - 1):
        err = (path[i] + off) - sensor_pos(q[i])
        dq = ik_step(q[i], vlo[i] + 5.0 * err, w_lo, p["kappa"]) + ik_step(q[i], vhi[i], w_hi, p["kappa"])
        q[i + 1] = q[i] + dq / fs
    return q, off


def joints_from_q(q):
    g = GEOM
    S = np.column_stack([g["S0"][0] + q[:, 0], g["S0"][1] + q[:, 1], np.full(len(q), g["S0"][2])])
    E = S + np.column_stack([g["L1"] * np.cos(g["e1"]) * u(q[:, 2]), np.full(len(q), -g["L1"] * np.sin(g["e1"]))])
    W = E + np.column_stack([g["L2"] * np.cos(g["e2"]) * u(q[:, 3]), np.full(len(q), -g["L2"] * np.sin(g["e2"]))])
    M = W + np.column_stack([g["L3"] * u(q[:, 4]), np.zeros(len(q))])
    return {"shoulder": S, "elbow": E, "wrist": W, "wrist_pose": W.copy(), "mcp3": M}


def cam_transform():
    th, ph = np.radians(20), np.radians(-50)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(ph), -np.sin(ph)], [0, np.sin(ph), np.cos(ph)]])
    return Rx @ Rz, np.array([60.0, -40.0, 1050.0])


def write_trial(d, idx, kind, band, seed, dpi, p, rng, epoch0, clean=False):
    os.makedirs(os.path.join(d, "joints"), exist_ok=True)
    nt = int((T + RAMP) * FS_TASK)
    t = np.arange(nt) / FS_TASK - RAMP                       # 立ち上がりを含む課題時刻
    scale = HAND_SCALE[dpi]
    if kind == "track":
        tj = traj(band, seed)
        ramp = np.clip((t + RAMP) / RAMP, 0, 1)[:, None]
        tgt = eval_traj(tj, t) * ramp
        cur = eval_traj(tj, t, TAU_HUMAN) * np.clip((t + RAMP - TAU_HUMAN) / RAMP, 0, 1)[:, None]
        cur = cur + filtfilt(*butter(2, 2.0 / (FS_TASK / 2)), rng.normal(0, REMNANT_MM * 3, cur.shape), axis=0)
    else:
        tj = None
        tgt = np.zeros((nt, 2))
        cur = filtfilt(*butter(2, 0.5 / (FS_TASK / 2)), rng.normal(0, 0.5, (nt, 2)), axis=0)
    hand = cur * scale                                       # センサの机の上の道筋 [mm]
    q, off = simulate(hand, FS_TASK, p, dpi)
    J = joints_from_q(q)
    # ---- マウス（センサの座標系 = 手の向きの回転 × 握り）
    psi = q[:, 4] - q[0, 4]
    yaw_m = p["grip"] * psi                                  # マウスの向きの変化（+ = 右回り）
    Ls = GEOM["L3"] * GEOM["sensor"]
    sens = J["wrist"][:, :2] + Ls * u(q[:, 4])
    v = np.gradient(sens, axis=0)
    c, s = np.cos(yaw_m), np.sin(yaw_m)
    # 机 → マウスの座標系: マウスが右回りに yaw 回っている → 机のベクトルを左回りに yaw 回す
    vm = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1)
    cm_per_count = 2.54 / dpi
    counts = vm / (cm_per_count * 10.0)
    acc = np.cumsum(counts, 0)
    qcnt = np.diff(np.vstack([[0, 0], np.round(acc)]), axis=0)          # 整数のカウント
    mouse_cm = np.cumsum(qcnt, 0) * cm_per_count
    # ---- カメラ（30 fps、遅れ、ゲイン、ジッタ）
    t_rs = np.arange(int((T + RAMP) * FS_RS)) / FS_RS - RAMP
    R, org = cam_transform()
    names = ["shoulder", "elbow", "wrist", "wrist_pose", "mcp3"]
    P = np.zeros((len(t_rs), len(names), 3))
    center = J["mcp3"].mean(0)
    for k, nm in enumerate(names):
        X = np.column_stack([np.interp(t_rs - (0 if clean else TAU_CAM), t, J[nm][:, i]) for i in range(3)])
        if not clean:
            X = center + CAM_GAIN * (X - center)
            X = X + rng.normal(0, JITTER[nm] / np.sqrt(3), X.shape)
        P[:, k] = X @ R.T + org
    np.savez_compressed(os.path.join(d, "joints", "rs_joints.npz"), t=epoch0 + RAMP + t_rs, names=np.array(names),
                        P_C=P, status=np.zeros((len(t_rs), len(names)), np.int8),
                        status_names=np.array(["OK"]), lr_swapped=np.zeros(len(t_rs), bool))
    with open(os.path.join(d, "task.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("frame,t,epoch,target_x_cm,target_y_cm,cursor_x_cm,cursor_y_cm,mouse_dx_counts,mouse_dy_counts,n_events,on_target\n")
        for i in range(nt):
            on = int(np.hypot(*(cur[i] - tgt[i])) < 8.0)
            f.write(f"{i},{t[i]:.6f},{epoch0 + RAMP + t[i]:.6f},{tgt[i, 0] * scale / 10:.5f},{tgt[i, 1] * scale / 10:.5f},"
                    f"{mouse_cm[i, 0]:.5f},{mouse_cm[i, 1]:.5f},{qcnt[i, 0]:.0f},{qcnt[i, 1]:.0f},1,{on}\n")
    meta = {"kind": "trial", "task": {"kind": kind, "band": band, "seed": seed, "practice": False, "index": idx,
                                      "attempt": 1, "name": "static_pre" if kind == "static" else None,
                                      "trajectory": tj}}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return {"psi_sd_true": float(np.degrees(np.std(q[:, 4]))), "mouse_yaw_sd_true": float(np.degrees(np.std(yaw_m)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--people", nargs="+", default=list(PERSONAS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--bands", nargs="+", default=["slow", "fast"])
    ap.add_argument("--clean", action="store_true", help="ノイズ・遅れ・ゲインなし（指標の真値を作る）")
    a = ap.parse_args()
    groups = {}
    for pi, name in enumerate(a.people):
        p = PERSONAS[name]
        groups.setdefault(p["group"], []).append(name)
        for dpi in (400, 800):
            rng = np.random.default_rng([pi, dpi])
            sd = os.path.join(a.out, f"aim_{name}{dpi}_20261023_120000")
            os.makedirs(sd, exist_ok=True)
            epoch = 1.75e9 + pi * 1e4 + dpi
            info = {}
            write_trial(os.path.join(sd, "trial_00_static_pre"), 0, "static", None, None, dpi, p, rng, epoch, a.clean)
            k = 1
            for band in a.bands:
                for seed in a.seeds:
                    td = os.path.join(sd, f"trial_{k:02d}_{band}_s{seed}")
                    epoch += 60
                    info[os.path.basename(td)] = write_trial(td, k, "track", band, seed, dpi, p, rng, epoch, a.clean)
                    k += 1
            json.dump({"kind": "aim_session", "participant": name, "cm_per_count": 2.54 / dpi,
                       "config": {"trial_s": T, "ramp_s": RAMP, "mouse_dpi": dpi, "bands": BANDS},
                       "synthetic": {"persona": p, "trials": info}},
                      open(os.path.join(sd, "session.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1,
                      default=float)
            print("→", sd)
    json.dump(groups, open(os.path.join(a.out, "groups.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
