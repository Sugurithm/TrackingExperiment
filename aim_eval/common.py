"""aim_eval 共通部分（読込・時刻合わせ・机座標系・前処理・周波数解析・寄与の分解）

入力（試行フォルダ trial_XX_* から自動で探す）
  task.csv              課題のログ（t, epoch, target_x/y_cm, cursor_x/y_cm, mouse_dx/dy_counts, on_target）
  meta.json             試行の情報（"task" に band / seed / trajectory / epoch_start …）
  joints/rs_joints.npz  RealSense + MediaPipe（カメラ座標 [mm]）
  ../session.json       セッションの設定（cm_per_count, sensitivity, config）

方針
  * 時刻: task.csv の epoch と RealSense のフレーム時刻は同じ PC の時計なので，
    epoch → 課題時刻 t（解析区間の先頭 = 0 s）の対応をそのまま使う。残る遅れ（MediaPipe + 露光）は
    c0 でマウスと手首を突き合わせて測るだけで，補正はしない（accuracy_eval と同じ考え方）。
  * 座標: **机の座標系**（x = マウスの右，y = マウスの奥，z = 上）で解析する。
    MoCap がないので，マウスの積算変位（真値）に手首の 3 次元変位を回帰して軸を作る。
    較正に T_MC.json があれば `--tmc` でそちらを使える。
  * 周波数: ターゲットは 1/T（T = 40 s）の素数倍の正弦波の和。解析区間 [0, T) をちょうど 1 周期として
    一様グリッド（60 Hz）に載せ替え，窓なしの FFT で入力周波数のビンを直接読む。
  * 前処理: RS は Hampel（9 フレーム，3·MAD）+ 5 Hz ゼロ位相低域通過（accuracy_eval と同じ）。
"""
import argparse, glob, json, os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _swv
from scipy.signal import butter, filtfilt

JOINTS = ["shoulder", "elbow", "wrist", "wrist_pose", "mcp3"]
LABEL = {"shoulder": "肩", "elbow": "肘", "wrist": "手首(hand)", "wrist_pose": "手首(pose)", "mcp3": "MCP3",
         "cursor": "カーソル", "mouse": "マウス", "target": "ターゲット"}
COLOR = {"shoulder": "#2a78d6", "elbow": "#eb6834", "wrist": "#1baf7a",
         "wrist_pose": "#eda100", "mcp3": "#e87ba4", "cursor": "#444", "mouse": "#888"}

# 手先の変位を足し算で分解したときの成分（common.contributions）
#   手先 = 肩 + 上腕ベクトル + 前腕ベクトル（+ 手ベクトル）なので，変位も同じ和に分かれる
PARTS = ["trunk", "upperarm", "forearm", "hand"]
PARTLABEL = {"trunk": "肩の並進（体幹）", "upperarm": "上腕の回転（肩関節）",
             "forearm": "前腕の回転（肘関節）", "hand": "手の回転（手首）"}
PARTCOLOR = {"trunk": "#7a5ea8", "upperarm": "#2a78d6", "forearm": "#eb6834", "hand": "#1baf7a"}

OVERRIDE_NAME = "aim_eval_override.json"   # セッションごとの上書き（参加者名・DPI）

FC_LP = 5.0        # 低域通過のカットオフ [Hz]
HAMPEL_K = 4       # 窓の片側（9 フレーム）
HAMPEL_N = 3.0     # 3·MAD
FS_U = 60.0        # 一様グリッドの標本化周波数 [Hz]
# R（同時性）を評価する包絡の時間尺度 [Hz]（1/fc が「持ち回りの周期」の目安）
ENV_FCS = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0)

SNR_MIN = 2.0      # これ未満の成分は「測れていない」とみなす（振幅 / 近傍ビンのノイズ）
AX = ("x", "y")    # 机の座標（x = 右，y = 奥）。画面の x, y と対応する


# ------------------------------------------------------------------ 引数・探索
def parser(desc):
    p = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trial", nargs="?", default=".", help="試行フォルダ（trial_XX_*）")
    p.add_argument("--rs", help="rs_joints.npz")
    p.add_argument("--tmc", help="T_MC.json（あれば机座標にこれを使う）")
    p.add_argument("--axes", help="desk_frame.json（他の試行で作った机の座標系を使う）")
    p.add_argument("--out", help="出力先（既定: <trial>/eval）")
    return p


def _find(trial, given, patterns, required=True):
    if given:
        return given
    for pat in patterns:
        hit = sorted(glob.glob(os.path.join(trial, pat), recursive=True))
        if hit:
            return hit[0]
    if required:
        raise FileNotFoundError(f"{patterns} が {trial} に見つかりません")
    return None


def session_dir(trial):
    d = os.path.dirname(os.path.abspath(trial))
    return d if os.path.exists(os.path.join(d, "session.json")) else None


def list_trials(session, kinds=("track", "static")):
    """セッションの試行フォルダ（やり直しは最後の attempt だけ）。(dir, meta) の一覧。"""
    out = {}
    for d in sorted(glob.glob(os.path.join(session, "trial_*"))):
        f = os.path.join(d, "meta.json")
        if not os.path.exists(f):
            continue
        meta = json.load(open(f, encoding="utf-8"))
        task = meta.get("task", {})
        if task.get("kind") not in kinds or task.get("practice"):
            continue
        key = (task.get("kind"), task.get("band"), task.get("seed"), task.get("name"))
        if key not in out or task.get("attempt", 1) >= out[key][1].get("task", {}).get("attempt", 1):
            out[key] = (d, meta)
    return [out[k] for k in sorted(out, key=lambda k: out[k][1].get("task", {}).get("index", 0))]


# ------------------------------------------------------------------ 試行
class Trial:
    """1 試行ぶん。関節は机座標 [mm]，時刻は課題時刻（解析区間の先頭 = 0 s）。"""

    def __init__(self, args):
        tr = args.trial.rstrip("/\\")
        self.dir = tr
        self.name = os.path.basename(os.path.abspath(tr))
        self.out = args.out or os.path.join(tr, "eval")
        os.makedirs(self.out, exist_ok=True)

        self.meta = json.load(open(os.path.join(tr, "meta.json"), encoding="utf-8"))
        self.task = self.meta.get("task", {})
        self.kind = self.task.get("kind", "track")
        self.band = self.task.get("band")
        self.seed = self.task.get("seed")
        self.traj = self.task.get("trajectory")
        self.sess_dir = session_dir(tr)
        self.session = json.load(open(os.path.join(self.sess_dir, "session.json"), encoding="utf-8")) if self.sess_dir else {}
        cfg = self.session.get("config", {})
        self.cfg = cfg
        self.T = float(cfg.get("trial_s", 40.0))
        self.cm_per_count = float(self.session.get("cm_per_count", np.nan))
        self.dpi = float(cfg.get("mouse_dpi", np.nan))
        self.participant = self.session.get("participant", self.meta.get("subject", "?"))
        self.override = {}
        ovf = os.path.join(self.sess_dir, OVERRIDE_NAME) if self.sess_dir else None
        if ovf and os.path.exists(ovf):
            # pipeline.py が書く上書き（記録時に --dpi を付け忘れた、参加者名を揃えたい、など）
            self.override = json.load(open(ovf, encoding="utf-8"))
            self.participant = self.override.get("participant", self.participant)
            d = self.override.get("dpi")
            if d and np.isfinite(self.dpi) and abs(d - self.dpi) > 1e-6:
                # 記録時の DPI が違っていたら counts → cm の換算もその比で直す
                self.cm_per_count = self.cm_per_count * self.dpi / float(d)
                self.dpi_recorded = self.dpi
                self.dpi = float(d)
            elif d:
                self.dpi = float(d)

        # --- 課題のログ
        (self.task_t, self._epoch, self.tgt, self.cur,
         self.dmouse, self.on_target) = _read_task(os.path.join(tr, "task.csv"))
        self.fs_task = 1.0 / np.median(np.diff(self.task_t))
        # マウスの積算変位 [mm]（机の上の手の 2 次元変位の真値）
        self.mouse = np.cumsum(self.dmouse, axis=0) * self.cm_per_count * 10.0

        # --- 関節（カメラ座標 [mm]）
        self.f_rs = _find(tr, args.rs, ["joints/rs_joints.npz", "**/rs_joints.npz"], required=False)
        self.has_rs = self.f_rs is not None
        if self.has_rs:
            r = np.load(self.f_rs, allow_pickle=False)
            self.rs_names = [str(n) for n in r["names"]]
            t_epoch = r["t"].astype(float)
            self.P_cam = r["P_C"].astype(float)
            self.uv = r["uv"].astype(float) if "uv" in r.files else None
            self.status = r["status"] if "status" in r.files else None
            self.status_names = [str(s) for s in r["status_names"]] if "status_names" in r.files else None
            self.vis = r["vis"].astype(float) if "vis" in r.files else None
            self.P_other = r["P_C_left"].astype(float) if "P_C_left" in r.files else None
            self.lr_swapped = r["lr_swapped"].astype(bool) if "lr_swapped" in r.files else None
            self.lr_how = r["lr_how"].astype(int) if "lr_how" in r.files else None
            # 時刻: epoch → 課題時刻（同じ PC の時計）
            self.t = np.interp(t_epoch, self._epoch, self.task_t, left=np.nan, right=np.nan)
            self.fs = 1.0 / np.median(np.diff(self.t[np.isfinite(self.t)]))
            self.in_span = np.isfinite(self.t) & (self.t >= 0) & (self.t < self.T)
            self._make_desk_frame(args)
            self._preprocess()
        else:
            self.rs_names = []
            self.t = np.array([]); self.in_span = np.array([], bool); self.fs = np.nan

    def _make_desk_frame(self, args):
        """机の座標系（x = マウスの右，y = 奥，z = 上）をカメラ座標の中で決める。"""
        src = None
        if getattr(args, "axes", None):
            fr = json.load(open(args.axes, encoding="utf-8")); src = args.axes
        elif getattr(args, "tmc", None):
            T_MC = np.array(json.load(open(args.tmc, encoding="utf-8"))["T_MC"])
            fr = {"R": np.array(T_MC)[:3, :3].T.tolist(), "origin": np.array(T_MC)[:3, 3].tolist(),
                  "source": "T_MC", "gain_x": np.nan, "gain_y": np.nan, "resid_mm": np.nan}
            src = args.tmc
        else:
            fr = fit_desk_frame(self)
        self.frame = fr
        self.frame_source = fr.get("source", src)
        self.R_desk = np.array(fr["R"], float)        # 列 = 机の x, y, z 軸（カメラ座標）
        self.origin = np.array(fr["origin"], float)

    def to_desk(self, P):
        return (P - self.origin) @ self.R_desk

    def _preprocess(self):
        self.Praw, self.P = {}, {}
        for j, n in enumerate(self.rs_names):
            X = self.to_desk(self.P_cam[:, j])
            self.Praw[n] = X
            self.P[n] = lowpass(hampel(X), self.fs)

    def rs(self, name, filtered=True):
        return (self.P if filtered else self.Praw)[name]

    # --- 一様グリッド（解析区間 [0, T) をちょうど 1 周期にする）
    @property
    def grid_t(self):
        N = int(round(self.T * FS_U))
        return np.arange(N) / FS_U

    def on_grid(self, t, X):
        """時系列 (t, X) を解析区間の一様グリッドに載せ替える。欠損は直線補間（0.2 s 超の穴は NaN）。"""
        X = np.atleast_2d(np.asarray(X, float).T).T
        tg = self.grid_t
        out = np.full((len(tg), X.shape[1]), np.nan)
        for k in range(X.shape[1]):
            g = np.isfinite(t) & np.isfinite(X[:, k])
            if g.sum() < 10:
                continue
            out[:, k] = np.interp(tg, t[g], X[g, k], left=np.nan, right=np.nan)
            idx = np.clip(np.searchsorted(t[g], tg), 1, g.sum() - 1)
            out[(t[g][idx] - t[g][idx - 1]) > 0.2, k] = np.nan
        return out.squeeze() if out.shape[1] == 1 else out

    def joint_grid(self, name, filtered=True):
        return self.on_grid(self.t, self.rs(name, filtered))

    def task_grid(self, what):
        X = {"target": self.tgt, "cursor": self.cur, "mouse": self.mouse,
             "error": self.cur - self.tgt}[what]
        return self.on_grid(self.task_t, X)

    # --- ターゲットの周波数
    def freqs(self, ax):
        if not self.traj:
            return np.array([], int), np.array([]), np.array([])
        A = self.traj["axes"][ax]
        return np.array(A["n"], int), np.array(A["f_hz"], float), np.array(A["amp_cm"], float)

    def save_json(self, name, obj):
        path = os.path.join(self.out, name)
        json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=_jsonable)
        return path

    def header(self):
        return {"trial": self.name, "participant": self.participant, "dpi": self.dpi,
                "dpi_recorded": getattr(self, "dpi_recorded", self.dpi), "kind": self.kind,
                "band": self.band, "seed": self.seed, "T_s": self.T, "fs_rs": self.fs, "fs_task": self.fs_task,
                "cm_per_count": self.cm_per_count, "frame": {k: v for k, v in self.frame.items() if k != "R"},
                "rms_err_cm": self.task.get("rms_err_cm"), "on_target_pct": self.task.get("on_target_pct")}

    def tag(self):
        return f"{self.band} s{self.seed}" if self.kind == "track" else self.task.get("name", self.kind)


def _read_task(path):
    import csv as _csv
    with open(path, encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    g = lambda k: np.array([float(r[k]) for r in rows])
    tgt = np.stack([g("target_x_cm"), g("target_y_cm")], 1) * 10.0        # mm
    cur = np.stack([g("cursor_x_cm"), g("cursor_y_cm")], 1) * 10.0        # mm
    dm = np.stack([g("mouse_dx_counts"), g("mouse_dy_counts")], 1)
    return g("t"), g("epoch"), tgt, cur, dm, g("on_target")


def fit_desk_frame(tr, joint="wrist"):
    """マウスの積算変位（真値）に手首の 3 次元変位を回帰して机の 2 軸を求める。

    手首 ≈ 原点 + ex·mx + ey·my。|ex|, |ey| は「マウス 1 mm あたり手が何 mm 動いたか」で，
    1 に近いほど深度の尺度とマウスの換算が合っている（この試行だけで測れる健全性の指標）。
    """
    j = tr.rs_names.index(joint if joint in tr.rs_names else "wrist_pose")
    X = tr.P_cam[:, j]                       # 回帰はカメラ座標のまま行う
    M = np.stack([np.interp(tr.t, tr.task_t, tr.mouse[:, 0]),
                  np.interp(tr.t, tr.task_t, tr.mouse[:, 1])], 1)
    ok = tr.in_span & np.isfinite(X).all(1) & np.isfinite(M).all(1)
    if ok.sum() < 100 or np.ptp(M[ok, 0]) < 5:
        # 動きが足りない（静止試行など）: カメラ座標をそのまま使う
        return {"R": np.eye(3).tolist(), "origin": [0, 0, 0], "source": "カメラ座標（回帰できず）",
                "gain_x": np.nan, "gain_y": np.nan, "resid_mm": np.nan, "n": int(ok.sum())}
    A = np.column_stack([M[ok], np.ones(ok.sum())])
    sol, *_ = np.linalg.lstsq(A, X[ok], rcond=None)
    ex, ey, o = sol[0], sol[1], sol[2]
    gx, gy = float(np.linalg.norm(ex)), float(np.linalg.norm(ey))
    resid = float(np.sqrt(np.mean(np.sum((X[ok] - A @ sol) ** 2, 1))))
    ex = ex / gx
    ey = ey - (ey @ ex) * ex; ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    ang = float(np.degrees(np.arccos(np.clip((sol[0] / gx) @ (sol[1] / gy), -1, 1))))
    return {"R": np.column_stack([ex, ey, ez]).tolist(), "origin": o.tolist(),
            "source": f"マウス回帰（{joint}）", "gain_x": gx, "gain_y": gy, "axis_angle_deg": ang,
            "resid_mm": resid, "n": int(ok.sum())}


def _jsonable(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


# ------------------------------------------------------------------ 信号処理
def fill_nan(X):
    X = np.array(X, float, copy=True)
    if X.ndim == 1: X = X[:, None]
    bad = np.isnan(X).any(axis=1); i = np.arange(len(X))
    if (~bad).sum() < 2:
        return X, bad
    for k in range(X.shape[1]):
        X[bad, k] = np.interp(i[bad], i[~bad], X[~bad, k])
    return X, bad


def hampel_flags(X, k=HAMPEL_K, n=HAMPEL_N):
    Xf, bad = fill_nan(X)
    flags = np.zeros(len(Xf), bool); med_all = Xf.copy()
    for a in range(Xf.shape[1]):
        pad = np.pad(Xf[:, a], k, mode="edge")
        w = _swv(pad, 2 * k + 1)
        med = np.median(w, axis=1); mad = 1.4826 * np.median(np.abs(w - med[:, None]), axis=1)
        flags |= (np.abs(Xf[:, a] - med) > n * mad) & (mad > 0)
        med_all[:, a] = med
    return flags & ~bad, med_all


def hampel(X):
    flags, med = hampel_flags(X)
    Y = np.array(X, float, copy=True)
    Y[flags] = med[flags]
    return Y


def lowpass(X, fs, fc=FC_LP):
    X = np.asarray(X, float)
    Xf, bad = fill_nan(X)
    if (~bad).sum() < 30 or not np.isfinite(fs):
        return X.copy()
    b, a = butter(4, min(fc / (fs / 2), 0.99))
    Y = filtfilt(b, a, Xf, axis=0)
    Y[bad] = np.nan
    return Y.reshape(X.shape)


def speed(X, fs):
    X = np.asarray(X, float)
    return np.linalg.norm(np.gradient(X, axis=0), axis=1) * fs


def nanfill_grid(x):
    """FFT の前に欠損を埋める（埋めた割合も返す）。"""
    x = np.asarray(x, float)
    bad = ~np.isfinite(x)
    if bad.all():
        return x, 1.0
    y = x.copy(); i = np.arange(len(x))
    y[bad] = np.interp(i[bad], i[~bad], x[~bad])
    return y, float(bad.mean())


# ------------------------------------------------------------------ 周波数解析
def spectrum(x, T):
    """一様グリッドの信号 → 複素フーリエ係数（片側，振幅の単位）とビン番号 n（周波数 = n/T）。"""
    y, _ = nanfill_grid(x)
    y = y - np.nanmean(y)
    N = len(y)
    F = np.fft.rfft(y) * 2.0 / N
    n = np.arange(len(F))
    return n, F


def at_bins(n, F, bins):
    return np.array([F[b] if b < len(F) else np.nan + 0j for b in bins])


def bin_noise(F, bins, half=3, skip=1):
    """各入力ビンの近傍（入力ビン以外）の振幅の RMS = そのビンのノイズの目安。"""
    taken = set(int(b) for b in bins)
    out = []
    for b in bins:
        idx = [i for i in range(max(1, b - half - skip), min(len(F), b + half + skip + 1))
               if abs(i - b) > skip and i not in taken]
        out.append(np.sqrt(np.mean(np.abs(F[idx]) ** 2)) if idx else np.nan)
    return np.array(out)


def describing(u, y, T, bins):
    """入力 u に対する出力 y の記述関数（ビンごとのゲイン・位相）と remnant。"""
    n, U = spectrum(u, T)
    _, Y = spectrum(y, T)
    Uk, Yk = at_bins(n, U, bins), at_bins(n, Y, bins)
    H = Yk / Uk
    noise = bin_noise(Y, bins)
    snr = np.abs(Yk) / noise
    f = np.array(bins) / T
    # remnant: 入力ビン以外のパワー / 全パワー（入力の帯域内のみ）
    hi = int(max(bins)) + 3
    tot = np.sum(np.abs(Y[1:hi]) ** 2)
    sig = np.sum(np.abs(Yk) ** 2)
    rem = float(1 - sig / tot) if tot > 0 else np.nan
    return {"f_hz": f, "bins": np.array(bins), "gain": np.abs(H), "phase_deg": np.degrees(np.angle(H)),
            "amp_in": np.abs(Uk), "amp_out": np.abs(Yk), "noise": noise, "snr": snr,
            "H": H, "remnant": rem}


def delay_from_phase(f, phase_deg, gain=None, snr=None, min_snr=SNR_MIN):
    """位相の傾き（deg/Hz）から実効的な遅れ [ms]。位相は連続になるように展開する。"""
    f = np.asarray(f, float); p = np.asarray(phase_deg, float)
    ok = np.isfinite(f) & np.isfinite(p)
    if snr is not None: ok &= (snr > min_snr)
    if ok.sum() < 3:
        return np.nan, np.nan
    f, p = f[ok], p[ok]
    o = np.argsort(f); f, p = f[o], p[o]
    pu = p.copy()
    for i in range(1, len(pu)):                      # 傾きが単調になるように 360° 単位で下げる
        while pu[i] - pu[i - 1] > 180: pu[i] -= 360
        while pu[i] - pu[i - 1] < -180: pu[i] += 360
    A = np.column_stack([f, np.ones(len(f))])
    sol, *_ = np.linalg.lstsq(A, pu, rcond=None)
    tau_ms = -sol[0] / 360.0 * 1000.0
    r = pu - A @ sol
    return float(tau_ms), float(np.sqrt(np.mean(r ** 2)))


def bandwidth(f, gain, snr=None, min_snr=SNR_MIN, ratio=1 / np.sqrt(2)):
    """ゲインが最低周波数の ratio 倍（既定 1/√2 = −3 dB）になる周波数 [Hz]。

    追従の「帯域幅」。むだ時間だけでゲインが落ちない追従でも必ず定義できるので、
    交差周波数（ゲイン = 1）より安定した指標として主に使う。
    """
    f = np.asarray(f, float); g = np.asarray(gain, float)
    ok = np.isfinite(g) & (g > 0)
    if snr is not None: ok &= (snr > min_snr)
    f, g = f[ok], g[ok]
    if len(f) < 2:
        return np.nan
    o = np.argsort(f); f, g = f[o], g[o]
    r = g / g[0]
    for i in range(len(f) - 1):
        if r[i] >= ratio > r[i + 1]:
            w = (np.log(r[i]) - np.log(ratio)) / (np.log(r[i]) - np.log(r[i + 1]))
            return float(np.exp(np.log(f[i]) + w * (np.log(f[i + 1]) - np.log(f[i]))))
    return np.nan


def bandwidth_fit(f, gain, snr=None, min_snr=SNR_MIN):
    """ゲイン曲線に 1 次遅れ g0/√(1+(f/fb)²) を当てはめて帯域 fb [Hz] を求める。

    −3 dB 点が測った周波数の外にあるときでも推定できる（外挿なので extrapolated = True を付ける）。
    返り値: (fb, extrapolated, 当てはめの残差 [対数ゲイン])
    """
    f = np.asarray(f, float); g = np.asarray(gain, float)
    ok = np.isfinite(g) & (g > 0)
    if snr is not None: ok &= (snr > min_snr)
    f, g = f[ok], g[ok]
    if len(f) < 3:
        return np.nan, True, np.nan
    best = (np.inf, np.nan)
    for fb in np.exp(np.linspace(np.log(0.02), np.log(20), 400)):
        model = 1 / np.sqrt(1 + (f / fb) ** 2)
        g0 = np.exp(np.mean(np.log(g) - np.log(model)))
        r = np.mean((np.log(g) - np.log(g0 * model)) ** 2)
        if r < best[0]:
            best = (r, fb)
    fb = float(best[1])
    return fb, bool(fb > f.max()), float(np.sqrt(best[0]))


def crossover(f, gain, snr=None, min_snr=SNR_MIN, tol=0.95):
    """ゲインが 1 を下回る周波数（対数で直線補間）。

    最後の下向きの交差を使う。最高周波数でもゲインが 1 近く（> tol）残っている場合は
    「まだ付いていけている」なので NaN を返す（遅れだけでゲインが落ちない追従では交差が定義できない）。
    """
    f = np.asarray(f, float); g = np.asarray(gain, float)
    ok = np.isfinite(g) & (g > 0)
    if snr is not None: ok &= (snr > min_snr)
    f, g = f[ok], g[ok]
    if len(f) < 2:
        return np.nan
    o = np.argsort(f); f, g = f[o], g[o]
    if g[-1] > tol:
        return np.nan
    for i in range(len(f) - 2, -1, -1):
        if g[i] >= 1 > g[i + 1]:
            w = np.log(g[i]) / (np.log(g[i]) - np.log(g[i + 1]))
            return float(np.exp(np.log(f[i]) + w * (np.log(f[i + 1]) - np.log(f[i]))))
    return float(f[0]) if g[0] < 1 else np.nan


# ------------------------------------------------------------------ 寄与の分解
def contributions(tr, end="wrist", filtered=True):
    """手先の変位を 肩の並進 + 上腕の回転 + 前腕の回転 (+ 手の回転) に分ける（足し算で厳密）。

    手先 = 肩 + (肘 − 肩) + (手首 − 肘) [+ (MCP3 − 手首)] なので，各項の変位がそのまま寄与になる。
    返り値: {部位: (N,3) 机座標の変位 [mm]}（平均を引いた値）と，合計との一致。
    """
    need = {"wrist": ["shoulder", "elbow", "wrist"], "mcp3": ["shoulder", "elbow", "wrist", "mcp3"]}[end]
    if any(n not in tr.rs_names for n in need):
        return None
    G = {n: tr.joint_grid(n, filtered) for n in need}
    ok = np.all([np.isfinite(G[n]).all(1) for n in need], axis=0)
    out = {}
    out["trunk"] = G["shoulder"]
    out["upperarm"] = G["elbow"] - G["shoulder"]
    out["forearm"] = G["wrist"] - G["elbow"]
    if end == "mcp3":
        out["hand"] = G["mcp3"] - G["wrist"]
    for k in out:
        X = out[k].copy()
        X[~ok] = np.nan
        out[k] = X - np.nanmean(X, axis=0)
    tot = G[end].copy(); tot[~ok] = np.nan
    out["_total"] = tot - np.nanmean(tot, axis=0)
    out["_ok"] = ok
    return out


def share_at_bins(parts, total, T, bins):
    """各成分の寄与を入力周波数ごとに「合計への射影」で分ける（位相を考慮し，和が 1 になる）。"""
    n, Ft = spectrum(total, T)
    Tk = at_bins(n, Ft, bins)
    res = {}
    for k, x in parts.items():
        _, Fk = spectrum(x, T)
        Xk = at_bins(n, Fk, bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            res[k] = np.real(Xk * np.conj(Tk)) / np.abs(Tk) ** 2
        res[k + "_amp"] = np.abs(Xk)
    return res


def var_share(parts, total):
    """分散（RMS）での寄与: 各成分の RMS と，合計への射影による説明率。"""
    ok = np.isfinite(total).all(1)
    for x in parts.values():
        ok &= np.isfinite(x).all(1)
    tv = total[ok]
    den = float(np.sum(tv ** 2))
    out = {}
    for k, x in parts.items():
        v = x[ok]
        out[k] = {"rms_mm": float(np.sqrt(np.mean(np.sum(v ** 2, 1)))),
                  "share": float(np.sum(v * tv) / den) if den > 0 else np.nan,
                  "rms_x": float(np.sqrt(np.mean(v[:, 0] ** 2))),
                  "rms_y": float(np.sqrt(np.mean(v[:, 1] ** 2))),
                  "rms_z": float(np.sqrt(np.mean(v[:, 2] ** 2)))}
    out["_total_rms_mm"] = float(np.sqrt(np.mean(np.sum(tv ** 2, 1))))
    out["_n"] = int(ok.sum())
    return out


# ------------------------------------------------------------------ 角度
def angle_deg(a, b, c):
    """b を頂点とする角（a-b-c）[deg]。"""
    u = a - b; v = c - b
    cos = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def seg_len(a, b):
    return np.linalg.norm(a - b, axis=1)


# ------------------------------------------------------------------ 図・表
def setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    want = ["Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic"]
    have = {f.name for f in fm.fontManager.ttflist}
    if not any(w in have for w in want):
        for f in fm.findSystemFonts():
            if "NotoSansCJK" in f or "ipaexg" in f.lower():
                try: fm.fontManager.addfont(f)
                except Exception: pass
        have = {f.name for f in fm.fontManager.ttflist}
    fam = [w for w in want if w in have] + ["DejaVu Sans"]
    plt.rcParams.update({"font.family": fam, "axes.grid": True, "grid.alpha": .3,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    return plt


def md_table(rows, cols, fmt="{:.1f}"):
    cols = [c if len(c) == 3 else (c[0], c[1], fmt) for c in cols]
    head = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        cells = []
        for _, k, f in cols:
            v = r.get(k, "")
            if isinstance(v, (float, np.floating)):
                cells.append(f.format(v) if np.isfinite(v) else "–")
            else:
                cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body

# ------------------------------------------------------------------ 協調の指標（核）
def vel_contributions(tr, end=None, filtered=True):
    """手先の「速度」を 4 成分に分ける（contributions の時間微分）。

    手先 = 肩 + (肘 − 肩) + (手首 − 肘) [+ (MCP3 − 手首)] の各項を微分したもの。
    返り値: ({部位: (N,3) 速度 [mm/s]}, 手先の速度 (N,3), 有効フレーム)
    """
    end = end or ("mcp3" if "mcp3" in tr.rs_names else "wrist")
    parts = contributions(tr, end=end, filtered=filtered)
    if parts is None:
        return None, None, None
    ok = parts["_ok"]
    V = {}
    for k, x in parts.items():
        if k.startswith("_"):
            continue
        V[k] = np.gradient(np.where(np.isfinite(x), x, np.nan), axis=0) * FS_U
    E = np.gradient(np.where(np.isfinite(parts["_total"]), parts["_total"], np.nan), axis=0) * FS_U
    return V, E, ok


def bin_power(x, T, bins, half=4, skip=1):
    """入力周波数ビンの信号パワー（近傍ビンのノイズを引く）と、そのビンの複素振幅。"""
    n, F = spectrum(x, T)
    taken = set(int(b) for b in bins)
    sig, noise = [], []
    for b in bins:
        idx = [i for i in range(max(1, b - half - skip), min(len(F), b + half + skip + 1))
               if abs(i - b) > skip and i not in taken]
        sig.append(np.abs(F[b]) ** 2 if b < len(F) else np.nan)
        noise.append(np.mean(np.abs(F[idx]) ** 2) if idx else np.nan)
    return np.array(sig), np.array(noise), at_bins(n, F, bins)


def joint_indices(tr, end=None, env_fc=0.5):
    """協調の 3 指標を返す。

    N_eff … 関与関節数 = 1/Σp²（p = 入力周波数での寄与シェア）。1 = 一関節依存，4 = 全体
    R     … 成分の活動量（速さの包絡）どうしの平均ペア相関。+ = 同時，− = 一つずつ交代
    C     … 入力周波数ごとの Σ|成分| / |合計|。1 = 単純な足し算，>1 = 互いに打ち消しながら手先を作る

    シェアは **入力周波数のビンだけ**で作り、近傍ビンのノイズを引く（時間領域の瞬間シェアは
    ジッタで N_eff が最大値へ張り付くため使わない）。
    """
    V, E, ok = vel_contributions(tr, end=end)
    if V is None or not tr.traj:
        return None
    names = [p for p in PARTS if p in V]
    res = {"parts": names, "end": end or ("mcp3" if "mcp3" in tr.rs_names else "wrist"),
           "n_frames": int(np.sum(ok))}

    # --- N_eff と C（入力周波数のビン）
    pw = {p: 0.0 for p in names}; pw_noise = {p: 0.0 for p in names}
    Cs, Cw = [], []
    per_bin = []
    for k, ax in enumerate(AX):
        nb, f, _ = tr.freqs(ax)
        amp = {}
        for p in names:
            sig, noi, A = bin_power(V[p][:, k], tr.T, nb)
            pw[p] += float(np.nansum(np.maximum(sig - noi, 0)))
            pw_noise[p] += float(np.nansum(noi))
            amp[p] = A
        sigE, noiE, AE = bin_power(E[:, k], tr.T, nb)
        for i in range(len(nb)):
            num = float(np.sum([np.abs(amp[p][i]) for p in names]))
            den = float(np.abs(AE[i]))
            snr = float(np.sqrt(sigE[i] / noiE[i])) if noiE[i] > 0 else np.nan
            if den > 0 and np.isfinite(snr):
                Cs.append(num / den); Cw.append(den)
            per_bin.append({"axis": ax, "n": int(nb[i]), "f_hz": float(f[i]), "snr": snr,
                            "C": num / den if den > 0 else np.nan,
                            **{p: float(np.abs(amp[p][i])) for p in names}})
    tot = sum(pw.values())
    share = {p: (pw[p] / tot if tot > 0 else np.nan) for p in names}
    res["share"] = share
    res["snr_part"] = {p: float(np.sqrt(pw[p] / pw_noise[p])) if pw_noise[p] > 0 else np.nan
                       for p in names}
    res["N_eff"] = float(1 / np.sum([share[p] ** 2 for p in names])) if tot > 0 else np.nan
    res["N_max"] = len(names)
    res["dominant"] = max(share, key=lambda p: share[p]) if tot > 0 else None
    res["dominant_share"] = float(share[res["dominant"]]) if res["dominant"] else np.nan
    res["C"] = float(np.average(Cs, weights=Cw)) if Cs else np.nan
    res["per_bin"] = per_bin

    # --- S（分業の指標）: 周波数ごとの分担ベクトルがどれだけ揃っているか
    # 入力周波数ごとに成分のシェア p(f) を作り、ビン対のコサイン類似度を平均する。
    # 1 に近い … どの速さの動きも同じ分担（全体が同じように動いている）
    # 小さい   … 速さごとに担当が分かれる（遅い動きは肩、速い動きは手首 のような役割分担）
    # 入力ビンの振幅だけを使うので SNR が高く、時間領域の R より安定して測れる。
    nb_eff, nb_w = [], []
    for r in per_bin:
        v = np.array([r[p] for p in names], float)
        if np.isfinite(r.get("snr", np.nan)) and r["snr"] > SNR_MIN and v.sum() > 0:
            q = v ** 2 / np.sum(v ** 2)                  # そのビンでの分担（パワー比）
            nb_eff.append(1 / np.sum(q ** 2)); nb_w.append(r["snr"])
            r["N_eff_bin"] = float(1 / np.sum(q ** 2))
    if len(nb_eff) >= 2 and np.isfinite(res["N_eff"]) and res["N_eff"] > 0:
        res["N_eff_bin_mean"] = float(np.average(nb_eff, weights=nb_w))
        res["S"] = float(res["N_eff_bin_mean"] / res["N_eff"])
        res["S_n_bins"] = len(nb_eff)
    else:
        res["S"] = res["N_eff_bin_mean"] = np.nan; res["S_n_bins"] = len(nb_eff)

    # --- R の前に、成分ごとの時間領域 SNR（静止試行のノイズ床と比べる）
    floor = static_floor(tr)
    snr_t, rms_t = {}, {}
    for p in names:
        rms_t[p] = float(np.sqrt(np.nanmean(np.linalg.norm(V[p], axis=1) ** 2)))
        f0 = floor.get(p, np.nan) if floor else np.nan
        snr_t[p] = float(rms_t[p] / f0) if f0 and np.isfinite(f0) and f0 > 0 else np.nan
    res["snr_time"] = snr_t
    res["vel_rms"] = rms_t
    res["floor_trial"] = floor.get("_trial") if floor else None
    # R は「実際に動いている成分」だけで計算する。動いていない/測れない成分を混ぜると、
    # その成分のノイズの包絡が相関に入り込んで符号まで変わる（合成データで確認済み）。
    active = [p for p in names if np.isfinite(snr_t[p]) and snr_t[p] >= SNR_MIN] if floor else list(names)
    res["R_parts"] = active
    res["R_n_active"] = len(active)
    res["R_reliable"] = (len(active) >= 2) if floor else None

    # --- R（速さの包絡の相関）
    # 「一つずつ交代」の時間尺度は分からないので、複数の尺度で評価して
    # 一番負に振れた値を R とする（どの尺度でも負にならなければ「同時」と読める）。
    # 尺度を 1 つに決め打ちすると、その尺度の交代しか見えない。
    mag = {}
    for p in names:
        m, _ = nanfill_grid(np.linalg.norm(V[p], axis=1))
        mag[p] = m
    scales, rows = [], []
    for fc in ENV_FCS:
        b, a = butter(2, min(fc / (FS_U / 2), 0.99))
        env = {p: filtfilt(b, a, mag[p]) for p in active}
        pairs, rs = [], []
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                x, y = env[active[i]], env[active[j]]
                r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan
                pairs.append({"a": active[i], "b": active[j], "r": r})
                if np.isfinite(r):
                    rs.append(r)
        m = float(np.mean(rs)) if rs else np.nan
        scales.append({"fc_hz": fc, "period_s": 1.0 / fc, "R": m, "pairs": pairs})
        rows.append(m)
    rows = np.array(rows, float)
    if len(active) < 2:
        res["R"] = res["R_scale_hz"] = res["R_mean"] = np.nan
        res["R_pairs"] = []
    elif np.isfinite(rows).any():
        i = int(np.nanargmin(rows))
        res["R"] = float(rows[i])                     # 一番負に振れた値（交代の証拠）
        res["R_scale_hz"] = float(ENV_FCS[i])         # その時間尺度
        res["R_mean"] = float(np.nanmean(rows))       # 全尺度の平均（同時性の目安）
        res["R_pairs"] = scales[i]["pairs"]
    else:
        res["R"] = res["R_scale_hz"] = res["R_mean"] = np.nan
        res["R_pairs"] = []
    res["R_scales"] = scales
    return res

def static_floor(tr, cache=True):
    """同じセッションの静止試行から、成分ごとの「速度のノイズ床」[mm/s] を測る。

    静止試行では本当の動きがほぼ 0 なので、同じ手順（寄与の分解 → 微分 → 大きさ）を通した
    値がそのままノイズの大きさになる。R（同時性）が信用できるかの判断に使う。
    結果はセッションに static_floor.json として保存し、2 回目以降は読み直す。
    """
    if not tr.sess_dir:
        return {}
    path = os.path.join(tr.sess_dir, "static_floor.json")
    if cache and os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            pass
    cand = [d for d, m in list_trials(tr.sess_dir, kinds=("static",))
            if os.path.exists(os.path.join(d, "joints", "rs_joints.npz"))]
    if not cand:
        return {}
    import argparse as _ap
    ns = _ap.Namespace(trial=cand[0], rs=None, tmc=None, axes=None,
                       out=os.path.join(cand[0], "eval"))
    try:
        st = Trial(ns)
        V, E, ok = vel_contributions(st)
    except Exception:
        return {}
    if V is None:
        return {}
    floor = {p: float(np.sqrt(np.nanmean(np.linalg.norm(V[p], axis=1) ** 2))) for p in V}
    floor["_total"] = float(np.sqrt(np.nanmean(np.linalg.norm(E, axis=1) ** 2)))
    floor["_trial"] = os.path.basename(cand[0])
    if cache:
        try:
            json.dump(floor, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        except Exception:
            pass
    return floor

