"""joint_usage.py — 上級者（FPS プレイヤー）・エイムの良い人は関節をどう使っているか（2026-10-23）

  python aim_eval/joint_usage.py --config analysis_out/20261022/config_used.py \
         --frames analysis_out/20261022 --perf analysis_out/20261022/perf_coordination/trials_perf.csv \
         --out analysis_out/20261023_joint_usage [--n-perm 20000] [--seed 0] [--stats-only]

前提: run.py --steps joints（rs_joints.npz）と run.py --steps analyze（<frames>/<参加者>/desk_frame.json）、
perf_coordination.py（trials_perf.csv）が済んでいること。既存コードは書き換えない。

仮説（依頼の優先順）
  H1 各関節の寄与率を「全体を通して」と「一つの動作の中で」に分けて測る
     G（全体）… 試行全体の関節の動きの量のうち、その関節が占める割合
                 a_k = sqrt(RMS速度_k² − 静止ノイズ床_k²)、share_k = a_k / Σa
     M（動作）… マウスの速さの極小で区切った 1 回の移動（A → B）ごとに、成分の変位 Δc_k を出す
                 p_k = Δc_k·Δp / |Δp|²   … 到達への寄与（和が 1。負 = 逆向きに動いて打ち消す）
                 q_k = |Δc_k| / Σ|Δc|      … 動いた量の内訳 → N_eff_move = 1/Σq²（1 動作で何関節ぶん使うか）
                 E   = Σ|Δc_k| / |Δp|      … 余分な動き（1 = どの関節もまっすぐ目標方向，>1 = 打ち消し合い）
                 single = p_max ≥ 0.8      … 一関節でほぼ済ませた動作の割合
                 S_move = mean(N_eff_move) / N_eff(全動作をまとめた q)
                          1 = どの動作も同じ配分で同時に使う，<1 = 動作ごとに担当が入れ替わる
  H2 マウスの向き（手のヨー角）: ψ = (MCP3 − 手首) の机平面での向き ≈ マウスの向き
     φ = 前腕（肘 → 手首）の向き，ω = ψ − φ（手首の左右の曲げ）
       SD_ψ   … マウスの向きの振れ [deg]
       β      … ψ を φ に回帰した傾き。0 = 手首が前腕の回転を打ち消してマウスの向きを一定に保つ，
                 1 = 手首は固定（マウスは前腕と一緒に回る），>1 = 手首が同じ向きに上乗せして回す
       r_φω   … φ と ω の相関（− = 打ち消す，+ = 上乗せ）
       wiper  … ψ をマウスの左右位置に回帰した傾き [deg/cm]（横に動かすとワイパーのように回るか）
     検証: 手の掌（手首と MCP3 の中点）の速度とマウスの速度（マウス自身の座標系）の回転角 α（1.5 s 窓）を
           ψ と突き合わせる。α はランドマークの角度を使わない独立なマウスの回転の推定
  H3 DPI（400/800）・帯域（slow/fast）が変わっても同じ人の寄与率・協調度は変わらないか
     人 × 条件の平均で ICC(3,1)（一致性）と ICC(2,1)（絶対一致），条件間の差の大きさを人の間のばらつきと比べる，
     条件 A のプロファイルから条件 B の本人を当てる「指紋」照合（並べ替え検定）
  群と成績: FPS（kakeru, Suguru, Syou, xin, Yuya）vs 非 FPS（Keito, Kosuke, Suzuka, Yuuki）。
     成績 = 条件内で z 化した on-target と −RMS 誤差の平均（perf_coordination の trials_perf.csv）
"""
import argparse, csv, importlib.util, json, os, sys, time
from itertools import combinations
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common as C                                      # noqa: E402
from scipy import stats                                 # noqa: E402
from scipy.signal import find_peaks                     # noqa: E402

FPS = ["kakeru", "Suguru", "Syou", "xin", "Yuya"]
NONFPS = ["Keito", "Kosuke", "Suzuka", "Yuuki"]
PARTS = C.PARTS
PL = {"trunk": "体幹", "upperarm": "上腕(肩)", "forearm": "前腕(肘)", "hand": "手(手首)"}
CONDS = [(400, "slow"), (800, "slow"), (400, "fast"), (800, "fast")]
GCOL = {"FPS": "#2a78d6", "非FPS": "#eb6834"}
GMK = {"FPS": "o", "非FPS": "^"}

FC = 2.0              # 動作単位・全体の寄与・角度の低域通過 [Hz]（課題 ≤ 1.08 Hz。静止ノイズ床が 5 Hz の 1/4〜1/6 になる）
MIN_AMP = 15.0        # 動作として数えるマウスの移動量の下限 [mm]（関節のジッタ 1〜2 mm の 10 倍）
DUR = (0.2, 3.0)      # 動作の長さ [s]
SINGLE = 0.8          # p_max がこれ以上なら「一関節で済ませた動作」
RECRUIT = 0.15        # q_k がこれ以上の成分を「使った関節」と数える
ROT_WIN = 1.5         # マウスの回転（速度の回転角）の窓 [s]
ROT_MINSPD = 20.0     # その窓のマウスの平均の速さの下限 [mm/s]


class Args:
    def __init__(self, trial, **kw):
        self.trial = trial
        self.rs = self.tmc = self.axes = self.out = None
        for k, v in kw.items():
            setattr(self, k, v)


def group_of(p):
    return "FPS" if p in FPS else ("非FPS" if p in NONFPS else "?")


def load_config(path):
    spec = importlib.util.spec_from_file_location("cfg", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ 試行ごと
def lp(X):
    return C.lowpass(X, C.FS_U, FC)


def comps_xy(tr):
    """成分（体幹・上腕・前腕・手）と手先（MCP3）の机平面 xy の位置 [mm]（2 Hz 低域通過）。"""
    P = C.contributions(tr, end="mcp3")
    if P is None:
        return None
    X = {k: lp(P[k][:, :2]) for k in PARTS + ["_total"]}
    X["_ok"] = P["_ok"]
    return X


def rms_speed(X):
    v = np.gradient(X, axis=0) * C.FS_U
    return float(np.sqrt(np.nanmean(np.sum(v ** 2, 1))))


def floor_of(static_dir, axes, tmp):
    """静止試行の成分ごとの RMS 速さ [mm/s]（同じ前処理）。"""
    st = C.Trial(Args(static_dir, axes=axes, out=tmp))
    X = comps_xy(st)
    if X is None:
        return None
    return {k: rms_speed(X[k]) for k in PARTS}


def global_share(X, floor):
    a = {}
    for k in PARTS:
        r = rms_speed(X[k]); f = floor.get(k, 0.0)
        a[k] = float(np.sqrt(max(r ** 2 - f ** 2, 0.0)))
    s = sum(a.values())
    out = {f"G_share_{k}": a[k] / s if s > 0 else np.nan for k in PARTS}
    q = np.array([out[f"G_share_{k}"] for k in PARTS])
    out["G_N_eff"] = float(1 / np.sum(q ** 2)) if s > 0 else np.nan
    out.update({f"G_amp_{k}": a[k] for k in PARTS})
    # 手首を使わない 3 成分（体幹・上腕・前腕+手 = 肘 → MCP3 の遠位）
    return out


def movements(tr, X):
    """マウスの速さの極小で区切った動作ごとの寄与。"""
    M = lp(tr.task_grid("mouse"))
    vm = np.gradient(M, axis=0) * C.FS_U
    sm = np.linalg.norm(vm, axis=1)
    sm_f = np.where(np.isfinite(sm), sm, np.nanmax(sm))
    mins, _ = find_peaks(-sm_f, distance=int(0.15 * C.FS_U), prominence=np.nanmedian(sm) * 0.2)
    ok = X["_ok"]
    rows = []
    for a, b in zip(mins[:-1], mins[1:]):
        dur = (b - a) / C.FS_U
        if not (DUR[0] <= dur <= DUR[1]) or not ok[a:b + 1].all():
            continue
        dm = M[b] - M[a]
        amp_m = float(np.linalg.norm(dm))
        if amp_m < MIN_AMP:
            continue
        dp = X["_total"][b] - X["_total"][a]
        n2 = float(dp @ dp)
        dc = np.array([X[k][b] - X[k][a] for k in PARTS])
        if not np.isfinite(dc).all() or n2 < 1:
            continue
        p = dc @ dp / n2
        mag = np.linalg.norm(dc, axis=1)
        q = mag / mag.sum()
        # 経路長（動作の途中の行き過ぎ・戻りも含めた動きの量）
        path = np.array([np.sum(np.linalg.norm(np.diff(X[k][a:b + 1], axis=0), axis=1)) for k in PARTS])
        path_end = float(np.sum(np.linalg.norm(np.diff(X["_total"][a:b + 1], axis=0), axis=1)))
        r = {"t0": a / C.FS_U, "dur_s": dur, "amp_mouse_mm": amp_m, "amp_hand_mm": float(np.sqrt(n2)),
             "dir_deg": float(np.degrees(np.arctan2(dm[1], dm[0]))),
             "xness": float(abs(dm[0]) / amp_m),           # 1 = 左右，0 = 前後
             "peak_speed": float(np.nanmax(sm[a:b + 1])),
             "N_eff_move": float(1 / np.sum(q ** 2)), "E_excess": float(mag.sum() / np.sqrt(n2)),
             "E_path": float(path.sum() / path_end) if path_end > 0 else np.nan,
             "p_max": float(p.max()), "dominant": PARTS[int(np.argmax(p))],
             "n_recruit": int(np.sum(q >= RECRUIT))}
        for i, k in enumerate(PARTS):
            r[f"p_{k}"] = float(p[i]); r[f"q_{k}"] = float(q[i]); r[f"mag_{k}"] = float(mag[i])
        rows.append(r)
    return rows, M, vm


def unwrap_angle(V):
    a = np.arctan2(V[:, 1], V[:, 0])
    ok = np.isfinite(a)
    out = np.full(len(a), np.nan)
    if ok.sum() > 10:
        out[ok] = np.unwrap(a[ok])
    return np.degrees(out)


def heading(tr, M, vm, mov):
    G = {n: lp(tr.joint_grid(n)[:, :2]) for n in ["shoulder", "elbow", "wrist", "mcp3"]}
    psi = unwrap_angle(G["mcp3"] - G["wrist"])        # 手（≈ マウス）の向き
    phi = unwrap_angle(G["wrist"] - G["elbow"])       # 前腕の向き
    tha = unwrap_angle(G["elbow"] - G["shoulder"])    # 上腕の向き
    om = psi - phi                                     # 手首の左右の曲げ
    ok = np.isfinite(psi) & np.isfinite(phi) & np.isfinite(M).all(1)
    out = {}
    if ok.sum() < 300:
        return out, None
    d = lambda x: x[ok] - np.mean(x[ok])
    ps, ph, o, th = d(psi), d(phi), d(om), d(tha)
    out["SD_psi"] = float(ps.std()); out["SD_phi"] = float(ph.std())
    out["SD_omega"] = float(o.std()); out["SD_theta_upper"] = float(np.nanstd(th))
    out["beta_psi_phi"] = float(np.polyfit(ph, ps, 1)[0]) if ph.std() > 0 else np.nan
    out["r_phi_omega"] = float(np.corrcoef(ph, o)[0, 1])
    # 同じ手首の回転量に対してマウスの向きがどれだけ振れるか（1 = 手首の回転がそのままマウスの回転）
    out["psi_per_omega"] = float(ps.std() / o.std()) if o.std() > 0 else np.nan
    mx = M[ok, 0] - np.mean(M[ok, 0])
    out["wiper_deg_per_cm"] = float(np.polyfit(mx / 10.0, ps, 1)[0]) if mx.std() > 0 else np.nan
    out["r_psi_mousex"] = float(np.corrcoef(mx, ps)[0, 1])
    # --- 独立な検証: 掌の速度（机座標）とマウスの速度（マウス座標）の回転角
    h = (G["wrist"] + G["mcp3"]) / 2
    vh = np.gradient(h, axis=0) * C.FS_U
    g = np.isfinite(vh).all(1) & np.isfinite(vm).all(1)
    A, *_ = np.linalg.lstsq(vm[g], vh[g], rcond=None)    # 全体の 2×2 写像（軸の伸び縮み・傾きを吸収）
    vmA = vm @ A
    W = int(ROT_WIN * C.FS_U)
    alpha = np.full(len(vm), np.nan)
    for i in range(W // 2, len(vm) - W // 2, 3):
        s = slice(i - W // 2, i + W // 2); gg = g[s]
        if gg.sum() < 0.8 * W:
            continue
        m = vmA[s][gg]; hh = vh[s][gg]
        if np.mean(np.linalg.norm(m, axis=1)) < ROT_MINSPD:
            continue
        alpha[i] = np.degrees(np.arctan2(np.sum(m[:, 0] * hh[:, 1] - m[:, 1] * hh[:, 0]), np.sum(m * hh)))
    k = np.isfinite(alpha) & np.isfinite(psi)
    if k.sum() > 50:
        a_ = alpha[k]; p_ = psi[k] - np.mean(psi[k])
        out["rot_r_alpha_psi"] = float(np.corrcoef(a_, p_)[0, 1])
        out["rot_slope_alpha_psi"] = float(np.polyfit(p_, a_, 1)[0])
        out["SD_alpha"] = float(np.std(a_))
        # 手のランドマークの角度を使わない β とワイパー傾き（α は窓で平滑されるので絶対値はやや小さめに出る）
        f_ = phi[k] - np.mean(phi[k]); x_ = M[k, 0] - np.mean(M[k, 0])
        out["beta_alpha_phi"] = float(np.polyfit(f_, a_, 1)[0]) if f_.std() > 0 else np.nan
        out["wiper_alpha_deg_per_cm"] = float(np.polyfit(x_ / 10.0, a_, 1)[0]) if x_.std() > 0 else np.nan
    # 動作ごとの向きの変化
    for r in mov:
        a = int(round(r["t0"] * C.FS_U)); b = a + int(round(r["dur_s"] * C.FS_U))
        r["dpsi_deg"] = float(psi[b] - psi[a]); r["dphi_deg"] = float(phi[b] - phi[a])
        r["domega_deg"] = float(om[b] - om[a])
    series = {"t": tr.grid_t, "psi": psi, "phi": phi, "omega": om, "alpha": alpha, "mouse_x": M[:, 0]}
    return out, series


def move_summary(mov):
    """試行の動作をまとめる。"""
    out = {"n_moves": len(mov)}
    if len(mov) < 5:
        return out
    get = lambda k: np.array([r[k] for r in mov], float)
    for k in ["N_eff_move", "E_excess", "E_path", "p_max", "n_recruit", "amp_hand_mm", "amp_mouse_mm"]:
        out[f"M_{k}"] = float(np.median(get(k)))
    out["M_single_frac"] = float(np.mean(get("p_max") >= SINGLE))
    for k in PARTS:
        out[f"M_p_{k}"] = float(np.median(get(f"p_{k}")))
        out[f"M_q_{k}"] = float(np.mean(get(f"q_{k}")))
        out[f"M_neg_{k}"] = float(np.mean(get(f"p_{k}") < -0.05))   # 逆向きに動いた動作の割合
    Q = np.array([[r[f"mag_{k}"] for k in PARTS] for r in mov])
    qa = Q.sum(0) / Q.sum()
    out["M_N_eff_pooled"] = float(1 / np.sum(qa ** 2))
    out["M_S_move"] = float(np.mean(get("N_eff_move")) / out["M_N_eff_pooled"])
    # 動作間で配分がどれだけ揃っているか（q ベクトルの平均ペアのコサイン類似度）
    qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    cs = qn @ qn.T
    out["M_consistency"] = float(cs[np.triu_indices(len(qn), 1)].mean())
    # 近位（体幹+上腕）の関与と動作の大きさの関係: log(振幅) に対する傾き
    prox = get("q_trunk") + get("q_upperarm")
    la = np.log2(get("amp_hand_mm"))
    out["M_prox_slope"] = float(np.polyfit(la, prox, 1)[0]) if la.std() > 0 else np.nan
    # 方向で分けた前腕の到達寄与（左右の動作 / 前後の動作）
    xn = get("xness")
    for lab, sel in [("lr", xn >= 0.8), ("fb", xn <= 0.6)]:
        if sel.sum() >= 3:
            for k in PARTS:
                out[f"M_p_{k}_{lab}"] = float(np.median(get(f"p_{k}")[sel]))
    if "dpsi_deg" in mov[0]:
        dps = get("dpsi_deg"); amp = get("amp_hand_mm")
        out["M_abs_dpsi_per_cm"] = float(np.median(np.abs(dps) / (amp / 10)))
        dph = get("dphi_deg")
        out["M_beta_move"] = float(np.polyfit(dph, dps, 1)[0]) if dph.std() > 0 else np.nan
    return out


def analyze_trials(cfg, frames, out, perf):
    tmp = os.path.join(out, "_tmp"); os.makedirs(tmp, exist_ok=True)
    trials, moves, examples = [], [], {}
    floors = {}
    sessions = [s for s in cfg.SESSIONS]
    # 1) 静止ノイズ床（セッションごと。極端な値は全体の中央値に置き換える）
    for s in sessions:
        sd = os.path.join(ROOT, s["dir"])
        name = s.get("participant")
        axes = os.path.join(ROOT, frames, name, "desk_frame.json")
        st = [d for d, m in C.list_trials(sd, kinds=("static",))
              if os.path.exists(os.path.join(d, "joints", "rs_joints.npz"))]
        floors[s["dir"]] = floor_of(st[0], axes, tmp) if st else None
    med = {k: float(np.median([f[k] for f in floors.values() if f])) for k in PARTS}
    floor_used = {}
    for d, f in floors.items():
        floor_used[d] = {k: (f[k] if f and f[k] <= 2.5 * med[k] else med[k]) for k in PARTS}
        floor_used[d]["_replaced"] = [k for k in PARTS if not f or f[k] > 2.5 * med[k]]
        floor_used[d]["_raw"] = f
    # 2) 本番試行
    for s in sessions:
        sd = os.path.join(ROOT, s["dir"]); name = s["participant"]; dpi = int(s["dpi"])
        axes = os.path.join(ROOT, frames, name, "desk_frame.json")
        for d, meta in C.list_trials(sd, kinds=("track",)):
            band = meta["task"].get("band"); seed = meta["task"].get("seed")
            if band not in ("slow", "fast") or not os.path.exists(os.path.join(d, "joints", "rs_joints.npz")):
                continue
            t0 = time.time()
            tr = C.Trial(Args(d, axes=axes, out=tmp))
            X = comps_xy(tr)
            if X is None:
                continue
            key = f"{name}_{dpi}_{band}_s{seed}"
            row = {"key": key, "participant": name, "group": group_of(name), "dpi": dpi, "band": band, "seed": seed}
            row.update(global_share(X, floor_used[s["dir"]]))
            mov, M, vm = movements(tr, X)
            hd, series = heading(tr, M, vm, mov)
            row.update(hd)
            row.update(move_summary(mov))
            row.update(perf.get(key, {}))
            trials.append(row)
            for r in mov:
                moves.append({"key": key, "participant": name, "group": group_of(name), "dpi": dpi,
                              "band": band, "seed": seed, **r})
            if seed == 1 and series is not None:
                examples[key] = series
            print(f"  {key}: 動作 {len(mov)}，G前腕 {row.get('G_share_forearm', np.nan):.2f}，"
                  f"N_eff_move {row.get('M_N_eff_move', np.nan):.2f}，β {row.get('beta_psi_phi', np.nan):.2f}"
                  f"（{time.time() - t0:.1f} s）", flush=True)
    return trials, moves, floor_used, examples


# ------------------------------------------------------------------ 統計
def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    J = 1 - 3 / (4 * (na + nb) - 9)
    return float((a.mean() - b.mean()) / sp * J) if sp > 0 else np.nan


def exact_group_p(vals, labels):
    """群の平均差の正確な並べ替え p（9 人なら 126 通りすべて）。"""
    vals = np.asarray(vals, float); lab = np.asarray(labels)
    n1 = int(np.sum(lab == "FPS")); n = len(vals)
    obs = vals[lab == "FPS"].mean() - vals[lab != "FPS"].mean()
    diffs = []
    for idx in combinations(range(n), n1):
        m = np.zeros(n, bool); m[list(idx)] = True
        diffs.append(vals[m].mean() - vals[~m].mean())
    diffs = np.array(diffs)
    return float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12))


def boot_diff_ci(a, b, rng, n=5000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def spearman_perm(x, y, rng, n_perm):
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = np.isfinite(x) & np.isfinite(y)
    x, y = x[k], y[k]
    if len(x) < 5:
        return np.nan, np.nan
    r = stats.spearmanr(x, y)[0]
    null = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)])
    return float(r), float((np.sum(np.abs(null) >= abs(r) - 1e-12) + 1) / (n_perm + 1))


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan)
    ok = np.isfinite(p); pv = p[ok]
    if not len(pv):
        return q
    o = np.argsort(pv); m = len(pv)
    adj = np.minimum.accumulate((pv[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    r = np.empty(m); r[o] = np.minimum(adj, 1)
    q[ok] = r
    return q


def icc(Y):
    """Y: 人 × 条件。ICC(3,1)（一致性）と ICC(2,1)（絶対一致）。"""
    Y = np.asarray(Y, float)
    Y = Y[np.isfinite(Y).all(1)]
    n, k = Y.shape
    if n < 3:
        return np.nan, np.nan
    gm = Y.mean()
    ssr = k * np.sum((Y.mean(1) - gm) ** 2); ssc = n * np.sum((Y.mean(0) - gm) ** 2)
    sse = np.sum((Y - gm) ** 2) - ssr - ssc
    msr = ssr / (n - 1); msc = ssc / (k - 1); mse = sse / ((n - 1) * (k - 1))
    i3 = (msr - mse) / (msr + (k - 1) * mse)
    i2 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    return float(i3), float(i2)


def person_cond_means(trials, keys):
    people = sorted({t["participant"] for t in trials}, key=lambda p: (group_of(p) != "FPS", p.lower()))
    out = {}
    for p in people:
        for c in CONDS:
            rr = [t for t in trials if t["participant"] == p and (t["dpi"], t["band"]) == c]
            out[(p, c)] = {k: float(np.nanmean([r.get(k, np.nan) for r in rr])) if rr else np.nan for k in keys}
    return people, out


def fingerprint(people, pcm, keys, cA, cB, rng, n_perm=5000):
    """条件 A の人ごとのプロファイル（keys を z 化）から、条件 B の本人を最近傍で当てる。"""
    A = np.array([[pcm[(p, cA)][k] for k in keys] for p in people])
    B = np.array([[pcm[(p, cB)][k] for k in keys] for p in people])
    ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    A, B = A[ok], B[ok]
    allv = np.vstack([A, B]); mu, sd = allv.mean(0), allv.std(0); sd[sd == 0] = 1
    A, B = (A - mu) / sd, (B - mu) / sd
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    hit = float(np.mean(np.argmin(D, axis=1) == np.arange(len(A))))
    rank = np.mean([np.sum(D[i] < D[i, i]) + 1 for i in range(len(A))])
    # 帰無: 本人の対応をでたらめにしたときの的中率（最近傍は同じで、正解のラベルだけを入れ替える）
    nn = np.argmin(D, axis=1)
    null = [np.mean(nn == rng.permutation(len(A))) for _ in range(n_perm)]
    p = float((np.sum(np.array(null) >= hit - 1e-12) + 1) / (n_perm + 1))
    return {"hit_rate": hit, "mean_rank": float(rank), "n": int(len(A)), "chance": 1 / len(A), "p": p}


# 解析する指標（キー，見出し，手首の点に依存するか）
METRICS = [
    ("G_share_trunk", "G 体幹シェア", False), ("G_share_upperarm", "G 上腕シェア", False),
    ("G_share_forearm", "G 前腕シェア", True), ("G_share_hand", "G 手シェア", True),
    ("G_N_eff", "G N_eff（全体の関与関節数）", True),
    ("M_N_eff_move", "M N_eff_move（1 動作の関与関節数）", True),
    ("M_S_move", "M S_move（動作内の同時性）", True),
    ("M_consistency", "M 配分の一貫性", True),
    ("M_single_frac", "M 一関節で済ませた動作の割合", True),
    ("M_E_excess", "M 余分な動き E", True),
    ("M_p_forearm", "M 前腕の到達寄与", True), ("M_p_upperarm", "M 上腕の到達寄与", False),
    ("M_p_hand", "M 手の到達寄与", True), ("M_p_trunk", "M 体幹の到達寄与", False),
    ("M_neg_hand", "M 手が逆向きに動いた動作の割合", True),
    ("M_prox_slope", "M 近位の関与 vs 動作の大きさ", False),
    ("SD_psi", "H マウスの向きの振れ SD_ψ [deg]", True),
    ("beta_psi_phi", "H β（ψ の φ への追従）", True),
    ("r_phi_omega", "H r(φ, ω)", True),
    ("psi_per_omega", "H SD_ψ / SD_ω", True),
    ("wiper_deg_per_cm", "H ワイパー傾き [deg/cm]", True),
    ("M_abs_dpsi_per_cm", "H 1 動作の向きの変化 [deg/cm]", True),
    ("SD_alpha", "H マウス回転（速度から）SD [deg]", False),
    ("beta_alpha_phi", "H β（速度から: α の φ への追従）", False),
    ("wiper_alpha_deg_per_cm", "H ワイパー傾き（速度から）[deg/cm]", False),
]
PRIMARY = ["G_share_forearm", "G_share_upperarm", "G_N_eff", "M_N_eff_move", "M_S_move",
           "M_single_frac", "M_E_excess", "SD_psi", "beta_psi_phi"]


def run_stats(trials, moves, rng, n_perm):
    keys = [m[0] for m in METRICS] + ["perf_score", "on_target_pct", "rms_2d"]
    # 成績スコア: 条件内で z 化（on-target と −RMS）
    for c in CONDS:
        rr = [t for t in trials if (t["dpi"], t["band"]) == c]
        for k, sgn in [("on_target_pct", 1), ("rms_2d", -1)]:
            v = np.array([r.get(k, np.nan) for r in rr], float)
            z = sgn * (v - np.nanmean(v)) / np.nanstd(v)
            for r, zz in zip(rr, z):
                r[f"z_{k}"] = float(zz)
        for r in rr:
            r["perf_score"] = float(np.nanmean([r["z_on_target_pct"], r["z_rms_2d"]]))
    people, pcm = person_cond_means(trials, keys)
    pm = {p: {k: float(np.nanmean([pcm[(p, c)][k] for c in CONDS])) for k in keys} for p in people}
    groups = [group_of(p) for p in people]
    res = {"people": people, "groups": groups, "group": [], "perf": [], "invariance": [], "fingerprint": []}

    # --- 群の比較と成績との相関（人の平均 = 4 条件 × 3 seed）
    for k, lab, wr in METRICS + [("perf_score", "成績スコア（z）", False)]:
        v = np.array([pm[p][k] for p in people])
        a = v[np.array(groups) == "FPS"]; b = v[np.array(groups) != "FPS"]
        same = 0
        for c in CONDS:
            vc = np.array([pcm[(p, c)][k] for p in people])
            dc = np.nanmean(vc[np.array(groups) == "FPS"]) - np.nanmean(vc[np.array(groups) != "FPS"])
            same += int(np.sign(dc) == np.sign(a.mean() - b.mean()))
        res["group"].append({"key": k, "label": lab, "wrist": wr, "FPS_mean": float(a.mean()), "FPS_sd": float(a.std(ddof=1)),
                             "non_mean": float(b.mean()), "non_sd": float(b.std(ddof=1)),
                             "diff": float(a.mean() - b.mean()), "ci": boot_diff_ci(a, b, rng),
                             "g": hedges_g(a, b), "p_exact": exact_group_p(v, groups), "same_dir_conds": same,
                             "per_cond": {f"{c[0]}_{c[1]}": [float(np.nanmean([pcm[(p, c)][k] for p in people if group_of(p) == g]))
                                                             for g in ("FPS", "非FPS")] for c in CONDS}})
        if k != "perf_score":
            r, p = spearman_perm(v, [pm[pp]["perf_score"] for pp in people], rng, n_perm)
            res["perf"].append({"key": k, "label": lab, "wrist": wr, "rho": r, "p": p})
    for fam in ("group", "perf"):
        pk = "p_exact" if fam == "group" else "p"
        prim = [i for i, r in enumerate(res[fam]) if r["key"] in PRIMARY]
        q = bh([res[fam][i][pk] for i in prim])
        for i, qq in zip(prim, q):
            res[fam][i]["q_primary"] = float(qq)
        qa = bh([r[pk] for r in res[fam]])
        for r, qq in zip(res[fam], qa):
            r["q_all"] = float(qq)

    # --- 試行レベルでの成績との関係（同じ人の中で、ある使い方をした試行ほど成績が良いか）
    #     (人 × 条件) と (seed × 条件 = 軌道) の効果を最小二乗で除いた残差どうしの相関
    res["within_perf"] = []
    pc_lab = [f"{t['participant']}|{t['dpi']}|{t['band']}" for t in trials]
    sc_lab = [f"{t['seed']}|{t['dpi']}|{t['band']}" for t in trials]
    for k, lab, wr in METRICS:
        x = np.array([t.get(k, np.nan) for t in trials], float)
        y = np.array([t.get("perf_score", np.nan) for t in trials], float)
        g = np.isfinite(x) & np.isfinite(y)
        if g.sum() < 20:
            continue
        cols = []
        for labs in (pc_lab, sc_lab):
            u = sorted(set(np.array(labs)[g]))
            cols += [np.array(labs)[g] == v for v in u]
        D = np.column_stack(cols).astype(float)
        rx = x[g] - D @ np.linalg.lstsq(D, x[g], rcond=None)[0]
        ry = y[g] - D @ np.linalg.lstsq(D, y[g], rcond=None)[0]
        if rx.std() == 0 or ry.std() == 0:
            continue
        r = float(np.corrcoef(rx, ry)[0, 1])
        dfree = int(g.sum() - np.linalg.matrix_rank(D) - 1)
        t = r * np.sqrt(dfree / (1 - r ** 2))
        res["within_perf"].append({"key": k, "label": lab, "r": r, "df": dfree, "p": float(2 * stats.t.sf(abs(t), dfree))})
    q = bh([r["p"] for r in res["within_perf"]])
    for r, qq in zip(res["within_perf"], q):
        r["q"] = float(qq)

    # --- 不変性（H3）
    for k, lab, wr in METRICS:
        Y = np.array([[pcm[(p, c)][k] for c in CONDS] for p in people])
        i3, i2 = icc(Y)
        between_sd = float(np.nanstd(np.nanmean(Y, 1), ddof=1))
        row = {"key": k, "label": lab, "wrist": wr, "ICC3": i3, "ICC2": i2, "between_sd": between_sd}
        for nm, (ca, cb) in {"dpi_slow": ((400, "slow"), (800, "slow")), "dpi_fast": ((400, "fast"), (800, "fast")),
                             "band_400": ((400, "slow"), (400, "fast")), "band_800": ((800, "slow"), (800, "fast"))}.items():
            ia, ib = CONDS.index(ca), CONDS.index(cb)
            d = Y[:, ib] - Y[:, ia]; d = d[np.isfinite(d)]
            row[f"{nm}_d"] = float(d.mean())
            row[f"{nm}_absd_over_between"] = float(np.mean(np.abs(d)) / between_sd) if between_sd > 0 else np.nan
            row[f"{nm}_p"] = float(stats.wilcoxon(d)[1]) if len(d) >= 5 and np.any(d != 0) else np.nan
            row[f"{nm}_npos"] = int(np.sum(d > 0))
            ra = stats.spearmanr(Y[:, ia], Y[:, ib])[0]
            row[f"{nm}_rho"] = float(ra)
        res["invariance"].append(row)
    prof_sets = {
        "全体の寄与（G 4 シェア）": ["G_share_trunk", "G_share_upperarm", "G_share_forearm", "G_share_hand"],
        "動作の協調（M）": ["M_N_eff_move", "M_S_move", "M_single_frac", "M_E_excess", "M_consistency"],
        "マウスの向き（H）": ["SD_psi", "beta_psi_phi", "r_phi_omega", "wiper_deg_per_cm"],
        "手首を使わない（G 体幹・上腕 + M 上腕・体幹）": ["G_share_trunk", "G_share_upperarm", "M_p_upperarm", "M_p_trunk"],
    }
    for nm, ks in prof_sets.items():
        for ca, cb in [((400, "slow"), (800, "slow")), ((400, "fast"), (800, "fast")),
                       ((400, "slow"), (400, "fast")), ((800, "slow"), (800, "fast")), ((400, "slow"), (800, "fast"))]:
            fp = fingerprint(people, pcm, ks, ca, cb, rng)
            res["fingerprint"].append({"set": nm, "A": f"{ca[0]} {ca[1]}", "B": f"{cb[0]} {cb[1]}", **fp})

    # --- 動作の大きさをそろえた DPI 比較（800 は同じ画面の動きに半分の手の動きで足りるため）
    bins = [(15, 30), (30, 60), (60, 120)]
    res["amp_matched"] = []
    for band in ("slow", "fast"):
        for lo, hi in bins:
            for k in ["N_eff_move", "p_forearm", "p_upperarm", "p_hand", "p_trunk", "E_excess"]:
                d = []
                for p in people:
                    v = {}
                    for dpi in (400, 800):
                        mm = [m[k] for m in moves if m["participant"] == p and m["dpi"] == dpi and m["band"] == band
                              and lo <= m["amp_hand_mm"] < hi]
                        v[dpi] = (np.median(mm), len(mm)) if len(mm) >= 8 else (np.nan, len(mm))
                    if np.isfinite(v[400][0]) and np.isfinite(v[800][0]):
                        d.append(v[800][0] - v[400][0])
                d = np.array(d)
                res["amp_matched"].append({"band": band, "bin": f"{lo}-{hi}", "key": k, "n_people": len(d),
                                           "d_mean": float(d.mean()) if len(d) else np.nan,
                                           "npos": int(np.sum(d > 0)),
                                           "p": float(stats.wilcoxon(d)[1]) if len(d) >= 5 and np.any(d != 0) else np.nan})
    # 全体（大きさをそろえない）の DPI 差を同じ形で
    for band in ("slow", "fast"):
        for k in ["N_eff_move", "p_forearm", "p_upperarm", "p_hand", "p_trunk", "E_excess"]:
            d = []
            for p in people:
                v = [np.median([m[k] for m in moves if m["participant"] == p and m["dpi"] == dpi and m["band"] == band])
                     for dpi in (400, 800)]
                d.append(v[1] - v[0])
            d = np.array(d)
            res["amp_matched"].append({"band": band, "bin": "全部", "key": k, "n_people": len(d), "d_mean": float(d.mean()),
                                       "npos": int(np.sum(d > 0)), "p": float(stats.wilcoxon(d)[1])})
    res["pcm"] = {f"{p}|{c[0]}|{c[1]}": v for (p, c), v in pcm.items()}
    res["pm"] = pm
    return res


# ------------------------------------------------------------------ 図
def figures(trials, moves, res, examples, out):
    plt = C.setup_plot()
    people = res["people"]; groups = res["groups"]
    pcm = {tuple([k.split("|")[0], (int(k.split("|")[1]), k.split("|")[2])]): v for k, v in res["pcm"].items()}
    pm = res["pm"]
    labels = [f"{p}\n({'FPS' if g == 'FPS' else '非'})" for p, g in zip(people, groups)]

    # fig1: 全体 vs 動作 の寄与（人ごと、4 条件平均）
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, (pref, title) in zip(axes, [("G_share_", "全体を通した寄与率 G（動きの量の内訳）"),
                                         ("M_p_", "1 動作の到達への寄与 p（中央値）")]):
        bottom = np.zeros(len(people))
        for k in PARTS:
            v = np.array([pm[p][pref + k] for p in people])
            ax.bar(range(len(people)), v, bottom=bottom, color=C.PARTCOLOR[k], width=.7, label=PL[k],
                   edgecolor="white", linewidth=1.5)
            bottom += np.clip(v, 0, None)
        ax.axvline(len(FPS) - .5, color="#888", lw=1, ls="--")
        ax.set_xticks(range(len(people))); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("シェア（4 条件 × 3 seed の平均）")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1))
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_global_vs_movement.png"), dpi=130); plt.close(fig)

    # fig2: 動作単位の協調（N_eff_move の分布、G N_eff との散布、近位の関与と動作の大きさ）
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    data = [[m["N_eff_move"] for m in moves if m["participant"] == p] for p in people]
    bp = ax.boxplot(data, widths=.6, showfliers=False, patch_artist=True)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(GCOL[g]); patch.set_alpha(.35)
    ax.set_xticks(range(1, len(people) + 1)); ax.set_xticklabels(people, fontsize=8, rotation=30)
    ax.set_ylabel("1 動作の N_eff_move"); ax.set_title("1 動作で使う関節の数（全動作）", fontsize=10)
    ax = axes[1]
    for p, g in zip(people, groups):
        x = [pcm[(p, c)]["G_N_eff"] for c in CONDS]; y = [pcm[(p, c)]["M_N_eff_move"] for c in CONDS]
        ax.scatter(x, y, color=GCOL[g], marker=GMK[g], s=36, alpha=.8)
        ax.annotate(p, (np.mean(x), np.mean(y)), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lim = [1.5, 4]; ax.plot(lim, lim, "--", color="#888", lw=1)
    ax.set_xlabel("全体の N_eff（G）"); ax.set_ylabel("1 動作の N_eff（M、中央値）")
    ax.set_title("全体の分担 vs 1 動作の中の分担（点 = 人 × 条件）", fontsize=10)
    for g in GCOL:
        ax.scatter([], [], color=GCOL[g], marker=GMK[g], label=g)
    ax.legend(fontsize=8)
    ax = axes[2]
    edges = np.array([15, 22, 32, 45, 64, 90, 128, 180])
    for p, g in zip(people, groups):
        mm = [m for m in moves if m["participant"] == p]
        amp = np.array([m["amp_hand_mm"] for m in mm]); pr = np.array([m["q_trunk"] + m["q_upperarm"] for m in mm])
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (amp >= lo) & (amp < hi)
            if s.sum() >= 10:
                xs.append(np.sqrt(lo * hi)); ys.append(np.median(pr[s]))
        ax.plot(xs, ys, marker=GMK[g], color=GCOL[g], lw=1.2, ms=5, alpha=.85)
        if xs:
            ax.annotate(p, (xs[-1], ys[-1]), fontsize=7, xytext=(3, 0), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("動作の大きさ（手先の変位）[mm]"); ax.set_ylabel("近位（体幹+上腕）の動きの割合 q")
    ax.set_title("大きな動作ほど肩から動かすか", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_movement_coordination.png"), dpi=130); plt.close(fig)

    # fig3: マウスの向き
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ex = next((k for k in examples if k.startswith("Syou_400_slow")), next(iter(examples)))
    s = examples[ex]
    ax.plot(s["t"], s["psi"] - np.nanmean(s["psi"]), color="#1baf7a", lw=1.2, label="ψ 手（≈マウス）の向き")
    ax.plot(s["t"], s["phi"] - np.nanmean(s["phi"]), color="#eb6834", lw=1.2, label="φ 前腕の向き")
    a = s["alpha"]; k = np.isfinite(a)
    ax.plot(s["t"][k], a[k] - np.mean(a[k]), ".", color="#444", ms=2, label="α マウスの回転（速度から）")
    ax.set_xlim(0, 20); ax.set_xlabel("時間 [s]"); ax.set_ylabel("角度（平均からの差）[deg]")
    ax.set_title(f"例: {ex}", fontsize=10); ax.legend(fontsize=7)
    ax = axes[1]
    for p, g in zip(people, groups):
        x = [pcm[(p, c)]["beta_psi_phi"] for c in CONDS]; y = [pcm[(p, c)]["SD_psi"] for c in CONDS]
        ax.scatter(x, y, color=GCOL[g], marker=GMK[g], s=36, alpha=.8)
        ax.annotate(p, (np.mean(x), np.mean(y)), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axvline(0, color="#888", lw=.8, ls=":"); ax.axvline(1, color="#888", lw=.8, ls="--")
    ax.set_xlabel("β = dψ/dφ（0 = 手首が打ち消して向き一定，1 = 前腕と一緒に回る）")
    ax.set_ylabel("SD_ψ マウスの向きの振れ [deg]"); ax.set_title("マウスの向きの保ち方（点 = 人 × 条件）", fontsize=10)
    ax = axes[2]
    r = [t.get("rot_r_alpha_psi", np.nan) for t in trials]
    ax.hist([x for x in r if np.isfinite(x)], bins=20, color="#2a78d6", alpha=.8)
    ax.set_xlabel("r(α, ψ)（試行ごと）"); ax.set_ylabel("試行数")
    ax.set_title(f"検証: ランドマークの向き ψ と速度から求めた回転 α（中央値 r = {np.nanmedian(r):.2f}）", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_mouse_heading.png"), dpi=130); plt.close(fig)

    # fig4: 不変性（人ごとの条件間の線）
    keys = ["G_share_forearm", "G_share_upperarm", "M_N_eff_move", "M_S_move", "SD_psi", "beta_psi_phi"]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.8))
    for ax, k in zip(axes, keys):
        for p, g in zip(people, groups):
            ax.plot(range(4), [pcm[(p, c)][k] for c in CONDS], marker=GMK[g], color=GCOL[g], lw=1, ms=4, alpha=.8)
        ax.set_xticks(range(4)); ax.set_xticklabels([f"{c[0]}\n{c[1]}" for c in CONDS], fontsize=7)
        inv = next(r for r in res["invariance"] if r["key"] == k)
        ax.set_title(f"{dict((m[0], m[1]) for m in METRICS)[k]}\nICC(3,1) {inv['ICC3']:.2f}", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_invariance.png"), dpi=130); plt.close(fig)

    # fig5: 群差（g）と成績との相関（ρ）
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    G = [r for r in res["group"] if r["key"] != "perf_score"]
    y = np.arange(len(G))[::-1]
    axes[0].barh(y, [r["g"] for r in G], color=["#2a78d6" if not r["wrist"] else "#86b6ef" for r in G], height=.6)
    axes[0].set_yticks(y); axes[0].set_yticklabels([r["label"] for r in G], fontsize=8)
    for yy, r in zip(y, G):
        axes[0].text(r["g"], yy, f" p={r['p_exact']:.2f}", va="center", fontsize=6.5,
                     ha="left" if r["g"] >= 0 else "right")
    axes[0].axvline(0, color="#888", lw=.8); axes[0].set_xlabel("Hedges g（FPS − 非FPS）")
    axes[0].set_title("FPS 群と非 FPS 群の差（人 5 vs 4）", fontsize=10)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color="#2a78d6", label="手首の点を使わない指標"),
                            Patch(color="#86b6ef", label="手首・MCP3 の点に依存する指標")], fontsize=7, loc="lower right")
    P = {r["key"]: r for r in res["perf"]}
    axes[1].barh(y, [P[r["key"]]["rho"] for r in G], color="#1baf7a", height=.6)
    for yy, r in zip(y, G):
        pr = P[r["key"]]
        axes[1].text(pr["rho"], yy, f" p={pr['p']:.2f}", va="center", fontsize=6.5, ha="left" if pr["rho"] >= 0 else "right")
    axes[1].axvline(0, color="#888", lw=.8); axes[1].set_xlabel("Spearman ρ（成績スコアと、人 9）")
    axes[1].set_title("エイム成績との関係（+ = その指標が大きい人ほど成績が良い）", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_group_perf.png"), dpi=130); plt.close(fig)


# ------------------------------------------------------------------ 入出力
def read_perf(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = {}
            for k in ["rms_2d", "on_target_pct", "hf_frac", "exit_rate", "delay_ms"]:
                try:
                    d[k] = float(r[k])
                except (KeyError, ValueError):
                    d[k] = np.nan
            out[r["key"].replace("_400.0_", "_400_").replace("_800.0_", "_800_")] = d
    return out


def write_csv(rows, path):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if k not in ("seed", "dpi", "n_moves", "n_recruit") else int(float(v))
            except (TypeError, ValueError):
                pass
    return rows


def main():
    global FC, MIN_AMP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="analysis_out/20261022/config_used.py")
    ap.add_argument("--frames", default="analysis_out/20261022")
    ap.add_argument("--perf", default="analysis_out/20261022/perf_coordination/trials_perf.csv")
    ap.add_argument("--out", default="analysis_out/20261023_joint_usage")
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-only", action="store_true", help="trials.csv / movements.csv を読み直して統計と図だけ")
    ap.add_argument("--fc", type=float, default=FC, help="低域通過 [Hz]（感度解析用）")
    ap.add_argument("--min-amp", type=float, default=MIN_AMP, help="動作の下限 [mm]（感度解析用）")
    a = ap.parse_args()
    FC, MIN_AMP = a.fc, a.min_amp
    out = os.path.join(ROOT, a.out); os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ex_path = os.path.join(out, "examples.npz")
    if a.stats_only:
        trials = read_csv(os.path.join(out, "trials.csv")); moves = read_csv(os.path.join(out, "movements.csv"))
        z = np.load(ex_path, allow_pickle=True); examples = z["examples"].item()
    else:
        cfg = load_config(os.path.join(ROOT, a.config))
        perf = read_perf(os.path.join(ROOT, a.perf))
        trials, moves, floors, examples = analyze_trials(cfg, a.frames, out, perf)
        write_csv(trials, os.path.join(out, "trials.csv")); write_csv(moves, os.path.join(out, "movements.csv"))
        json.dump(floors, open(os.path.join(out, "noise_floor.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        np.savez_compressed(ex_path, examples=np.array(examples, dtype=object))
    res = run_stats(trials, moves, rng, a.n_perm)
    json.dump(res, open(os.path.join(out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=C._jsonable)
    write_csv([{k: v for k, v in r.items() if k != "per_cond"} | {f"{c}_{g}": r["per_cond"][c][i]
               for c in r["per_cond"] for i, g in enumerate(("FPS", "non"))} for r in res["group"]],
              os.path.join(out, "group_diff.csv"))
    write_csv(res["perf"], os.path.join(out, "perf_corr.csv"))
    write_csv(res["within_perf"], os.path.join(out, "within_perf.csv"))
    write_csv(res["invariance"], os.path.join(out, "invariance.csv"))
    write_csv(res["fingerprint"], os.path.join(out, "fingerprint.csv"))
    write_csv(res["amp_matched"], os.path.join(out, "dpi_amp_matched.csv"))
    write_csv([{"participant": p, "group": group_of(p), **res["pm"][p]} for p in res["people"]],
              os.path.join(out, "person_means.csv"))
    figures(trials, moves, res, examples, out)
    print("done:", out)


if __name__ == "__main__":
    main()
