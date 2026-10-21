"""既知の答えを入れた合成データを作る（解析プログラムの検証用）

run_experiment.py が作るのと同じ形のセッションを作り、
  - 人の遅れ τ_human（ターゲット → カーソル）
  - MediaPipe の遅れ τ_rs（マウス → 関節）
  - 周波数ごとの関節の分担（低周波 = 体幹・上腕、高周波 = 前腕・手）
  - 関節ごとのジッタ σ
を入れておく。run_all.py がこれらを復元できるかを確かめる。

  python make_synthetic.py out_dir --dpi 400 800 1600 --seeds 1 2 3
"""
import argparse, json, os
import numpy as np

FS_TASK = 240.0
FS_RS = 60.0
BANDS = {"slow": {"x": [2, 5, 11, 17], "y": [3, 7, 13, 19]},
         "fast": {"x": [3, 7, 13, 19, 29, 41], "y": [5, 11, 17, 23, 31, 43]}}
AMP_CM = {"x": 12.0, "y": 6.0}
AMP_EXP = 0.5
T = 40.0
RAMP = 3.0

# 真の値（これを復元できるかを見る）
TRUE = {"tau_human_ms": 220.0, "tau_rs_ms": 30.0, "remnant_mm": 2.0, "human_lag_tau_s": 0.12,
        "cutoff_hz": {"trunk": 0.10, "upperarm": 0.25, "forearm": 0.60, "hand": 1e9},
        "jitter_mm": {"shoulder": 3.8, "elbow": 3.1, "wrist": 1.9, "wrist_pose": 1.8, "mcp3": 2.2},
        "seg_mm": {"upperarm": 300.0, "forearm": 250.0, "hand": 80.0}}


def traj(band, seed):
    rng = np.random.default_rng([seed, 0 if band == "slow" else 1])
    tt = np.arange(0, T, 0.001)
    out = {"band": band, "seed": seed, "T": T, "ramp_s": RAMP, "axes": {}}
    for ax in ("x", "y"):
        n = np.array(BANDS[band][ax], float); f = n / T
        a = f ** (-AMP_EXP); ph = rng.uniform(0, 2 * np.pi, len(f))
        p = (a[:, None] * np.sin(2 * np.pi * f[:, None] * tt + ph[:, None])).sum(0)
        a = a * AMP_CM[ax] / np.abs(p).max()
        out["axes"][ax] = {"n": n.astype(int).tolist(), "f_hz": f.tolist(),
                           "amp_cm": a.tolist(), "phase_rad": ph.tolist()}
    return out


def eval_axis(A, t, tau=0.0, weight=None, lag_tau=0.0):
    """正弦波の和を時刻 t で評価。

    tau   : むだ時間 [s]（位相が周波数に比例して遅れる）
    lag_tau: 1 次遅れの時定数 [s]（ゲインが 1/√(1+(ωτ)²) に落ち、位相も −atan(ωτ) 遅れる）
    weight: 成分ごとの重み weight(f)
    """
    f = np.array(A["f_hz"]); a = np.array(A["amp_cm"]) * 10.0; ph = np.array(A["phase_rad"])
    w = np.ones_like(f) if weight is None else weight(f)
    if lag_tau:
        om = 2 * np.pi * f
        w = w / np.sqrt(1 + (om * lag_tau) ** 2)
        ph = ph - np.arctan(om * lag_tau)
    return (w[:, None] * a[:, None] * np.sin(2 * np.pi * f[:, None] * (t - tau)[None, :] + ph[:, None])).sum(0)


# 参加者ごとの「型」: 各部位の担当を重みで増減させる（1 = 既定）
STYLES = {
    "arm":   {"trunk": 2.5, "upperarm": 2.0, "forearm": 1.0, "hand": 0.35},   # 腕エイム
    "wrist": {"trunk": 0.4, "upperarm": 0.5, "forearm": 1.0, "hand": 2.5},    # 手首エイム
    "mixed": {"trunk": 1.0, "upperarm": 1.0, "forearm": 1.0, "hand": 1.0},    # 中間
}
STYLE = {"name": "mixed", "timing": "sim"}

# 時間的な型: sim = 全体が同時，seq = 一つずつ交代，comp = 同時かつ互いに打ち消し合う
TIMINGS = ("sim", "seq", "comp", "uni")
# uni = 周波数の分業なし（どの速さも同じ分担で同時に動く）。sim = 周波数で分業（近位が遅い動き担当）
SEQ_BLOCK_S = 2.5     # seq のとき、1 つの成分が担当する時間 [s]
COMP_GAIN = 1.2       # comp のとき、打ち消し合う成分の大きさ（手先には出ない）


def part_weights(f):
    """周波数ごとの分担（和が 1）。1 次の低域通過で近位ほど低い周波数だけを担う。"""
    g = STYLES[STYLE["name"]]
    w = {}
    for p, fc in TRUE["cutoff_hz"].items():
        band = 1.0 if STYLE.get("timing") == "uni" else 1.0 / np.sqrt(1 + (f / fc) ** 2)
        w[p] = g[p] * (band if np.ndim(band) else np.full_like(f, band))
    tot = sum(w.values())
    return {p: w[p] / tot for p in w}


def make_trial(dirpath, band, seed, dpi, cm_per_count, index, epoch0, rng, kind="track"):
    os.makedirs(os.path.join(dirpath, "joints"), exist_ok=True)
    tj = traj(band, seed) if kind == "track" else None
    n_task = int(T * FS_TASK)
    t = np.arange(n_task) / FS_TASK
    scale = 400.0 / dpi                      # 400 DPI を 1 とした手の移動量の倍率

    if kind == "track":
        tgt = np.stack([eval_axis(tj["axes"][ax], t) for ax in ("x", "y")], 1)        # 画面 [mm]
        tau = TRUE["tau_human_ms"] / 1000
        lag = TRUE.get("human_lag_tau_s", 0.0)
        cur = np.stack([eval_axis(tj["axes"][ax], t, tau=tau, lag_tau=lag) for ax in ("x", "y")], 1)
        cur += rng.normal(0, TRUE["remnant_mm"], cur.shape)
    else:
        tgt = np.zeros((n_task, 2)); cur = rng.normal(0, 0.3, (n_task, 2))
    hand = cur * scale                       # 机の上の手の変位 [mm]
    dm = np.diff(np.vstack([hand[:1] * 0, hand]), axis=0) / (cm_per_count * 10.0)

    # ---- 関節（RS の時刻）
    t_rs = np.arange(int(T * FS_RS)) / FS_RS
    tau_rs = TRUE["tau_rs_ms"] / 1000
    parts = {}
    if kind == "track":
        names4 = list(TRUE["cutoff_hz"])
        for p in names4:
            parts[p] = np.stack([eval_axis(tj["axes"][ax], t_rs, tau=tau_rs + TRUE["tau_human_ms"] / 1000,
                                           lag_tau=TRUE.get("human_lag_tau_s", 0.0),
                                           weight=lambda f, p=p: part_weights(f)[p]) * scale
                                 for ax in ("x", "y")], 1)
        timing = STYLE.get("timing", "sim")
        if timing == "seq":
            # 一つずつ交代: 各成分が担当する時間だけ動かし、担当外は止める（手先の合計は保つ）
            blk = (np.floor(t_rs / SEQ_BLOCK_S).astype(int)) % len(names4)
            total = sum(parts[p] for p in names4)
            for i, p in enumerate(names4):
                g = (blk == i).astype(float)[:, None]
                parts[p] = g * total
        elif timing == "comp":
            # 同時かつ相殺: 手先に出ない成分（符号が打ち消す）を足す
            extra = np.stack([eval_axis(tj["axes"][ax], t_rs, tau=tau_rs, lag_tau=0.0,
                                        weight=lambda f: np.ones_like(f) * COMP_GAIN) * scale
                              for ax in ("x", "y")], 1)
            sgn = [+1, -1, +1, -1]
            for i, p in enumerate(names4):
                parts[p] = parts[p] + sgn[i] * extra
    else:
        for p in TRUE["cutoff_hz"]:
            parts[p] = np.zeros((len(t_rs), 2))
    z = np.zeros((len(t_rs), 1))
    p3 = lambda P: np.hstack([P, z])
    S0 = np.array([0.0, 0.0, 0.0])
    U0 = np.array([0.0, TRUE["seg_mm"]["upperarm"], 60.0])
    F0 = np.array([0.0, TRUE["seg_mm"]["forearm"], -40.0])
    H0 = np.array([0.0, TRUE["seg_mm"]["hand"], 0.0])
    shoulder = S0 + p3(parts["trunk"])
    elbow = shoulder + U0 + p3(parts["upperarm"])
    wrist = elbow + F0 + p3(parts["forearm"])
    mcp3 = wrist + H0 + p3(parts["hand"])
    J = {"shoulder": shoulder, "elbow": elbow, "wrist": wrist, "wrist_pose": wrist.copy(), "mcp3": mcp3}

    # 机座標 → カメラ座標（適当な回転・並進）
    th, ph = np.radians(25), np.radians(-15)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(ph), -np.sin(ph)], [0, np.sin(ph), np.cos(ph)]])
    R = Rx @ Rz
    org = np.array([120.0, -80.0, 900.0])
    names = ["shoulder", "elbow", "wrist", "wrist_pose", "mcp3"]
    P = np.stack([(J[n] @ R.T) + org for n in names], 1)
    for k, n in enumerate(names):
        P[:, k] += rng.normal(0, TRUE["jitter_mm"][n] / np.sqrt(3), P[:, k].shape)
    status = np.zeros((len(t_rs), len(names)), np.int8)
    np.savez_compressed(os.path.join(dirpath, "joints", "rs_joints.npz"),
                        t=epoch0 + t_rs, names=np.array(names), P_C=P,
                        uv=np.zeros((len(t_rs), len(names), 2)), status=status,
                        status_names=np.array(["OK", "NO_POSE", "NO_HAND", "REJECT", "OUT", "NO_DEPTH"]),
                        vis=np.full((len(t_rs), len(names)), 0.99),
                        P_C_left=P + 300.0, lr_swapped=np.zeros(len(t_rs), bool),
                        lr_how=np.zeros(len(t_rs), np.int8))

    with open(os.path.join(dirpath, "task.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("frame,t,epoch,target_x_cm,target_y_cm,cursor_x_cm,cursor_y_cm,"
                "mouse_dx_counts,mouse_dy_counts,n_events,on_target\n")
        for i in range(n_task):
            on = int(np.hypot(*(cur[i] - tgt[i])) < 8.0)
            f.write(f"{i},{t[i]:.6f},{epoch0 + t[i]:.6f},{tgt[i, 0] / 10:.5f},{tgt[i, 1] / 10:.5f},"
                    f"{cur[i, 0] / 10:.5f},{cur[i, 1] / 10:.5f},{dm[i, 0]:.3f},{dm[i, 1]:.3f},1,{on}\n")
    err = np.hypot(cur[:, 0] - tgt[:, 0], cur[:, 1] - tgt[:, 1]) / 10
    meta = {"kind": "trial", "session": os.path.basename(dirpath), "subject": os.path.basename(dirpath), "trial": index,
            "calib": None, "status": "ok", "frames": len(t_rs),
            "task": {"kind": kind, "band": band, "seed": seed, "practice": False, "repeat": 0,
                     "index": index, "attempt": 1, "name": "static_pre" if kind == "static" else None,
                     "dir": os.path.basename(dirpath), "epoch_start": epoch0,
                     "epoch_end": epoch0 + T, "camera": "on", "trajectory": tj,
                     "frames": n_task, "fps_median": FS_TASK, "dt_max_ms": 5.0,
                     "rms_err_cm": float(np.sqrt(np.mean(err ** 2))),
                     "on_target_pct": 100.0 * float(np.mean(np.hypot(cur[:, 0] - tgt[:, 0],
                                                                     cur[:, 1] - tgt[:, 1]) < 8))}}
    json.dump(meta, open(os.path.join(dirpath, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--name", default="SYN", help="参加者 ID")
    ap.add_argument("--style", default="mixed", choices=list(STYLES), help="空間的な型（どの関節が主役か）")
    ap.add_argument("--timing", default="sim", choices=TIMINGS, help="時間的な型（同時 / 交代 / 相殺）")
    ap.add_argument("--dpi", nargs="+", type=float, default=[400, 800, 1600])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--bands", nargs="+", default=["slow", "fast"])
    a = ap.parse_args()
    STYLE["name"] = a.style; STYLE["timing"] = a.timing
    rng = np.random.default_rng(abs(hash(a.name)) % 10000)
    epoch = 1.7e9
    for dpi in a.dpi:
        sdir = os.path.join(a.out, f"aim_{a.name}_dpi{dpi:g}_20261021_120000")
        os.makedirs(sdir, exist_ok=True)
        cm_per_count = 2.54 / dpi
        trials, idx = [], 0
        make_trial(os.path.join(sdir, "trial_00_static_pre"), None, None, dpi, cm_per_count, 0, epoch,
                   rng, kind="static")
        epoch += T + 5
        idx = 1
        for band in a.bands:
            for seed in a.seeds:
                d = os.path.join(sdir, f"trial_{idx:02d}_{band}_s{seed}")
                make_trial(d, band, seed, dpi, cm_per_count, idx, epoch, rng)
                trials.append({"index": idx, "band": band, "seed": seed, "dir": os.path.basename(d)})
                epoch += T + 5; idx += 1
        json.dump({"kind": "aim_session", "participant": a.name, "created": "2026-10-21 12:00:00",
                   "calib": None, "camera": True, "screen_px": [1920, 1080], "px_per_cm": 40.0,
                   "cm_per_count": cm_per_count, "sensitivity": {"mode": "ingame", "edpi": dpi * 0.4},
                   "config": {"trial_s": T, "ramp_s": RAMP, "mouse_dpi": dpi, "bands": BANDS,
                              "amp_cm": AMP_CM, "amp_exponent": AMP_EXP, "sens_mode": "ingame"},
                   "trials": trials, "true_values": TRUE, "style": a.style, "timing": a.timing,
                   "style_weights": STYLES[a.style]},
                  open(os.path.join(sdir, "session.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("→", sdir)


if __name__ == "__main__":
    main()
