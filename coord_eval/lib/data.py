"""セッションの探索と試行の読み込み（記録フォルダは読むだけ）

aim_eval/common.py の Trial・fit_desk_frame・list_trials と skill_analysis/skill_eval/io.py の
セッション探索を 1 つにした。変更点:
  * 机の座標系はセッション（= 参加者 × DPI）の本番試行をまとめて 1 回だけ回帰で決める
    （試行ごとの切片を入れた最小二乗。1 本で決めるより安定し、同じセッションの全試行で同じ軸になる）
  * 回帰に使う点は手首と MCP3 の中点（palm ≒ マウスのセンサの位置）
  * DPI はフォルダ名からも読む（記録時に --dpi を付け忘れたセッションの換算ミスを防ぐ）
"""
import csv, glob, json, os, re
import numpy as np

from .. import config as K
from . import sig

OVERRIDE_NAME = "aim_eval_override.json"
JOINTS = ["shoulder", "elbow", "wrist", "wrist_pose", "mcp3"]


# ------------------------------------------------------------------ 名前・群
def display_name(raw):
    return K.ALIASES.get(str(raw).lower(), str(raw))


def group_of(name):
    for g, members in K.GROUPS.items():
        if name in members:
            return g
    return None


# ------------------------------------------------------------------ セッションの探索
def _dpi_from_name(path):
    m = re.match(r"aim_[A-Za-z]+?_?(\d{3,4})(?:_|dpi)", os.path.basename(path))
    if m:
        return float(m.group(1))
    m = re.search(r"dpi(\d{3,4})", os.path.basename(path))
    return float(m.group(1)) if m else None


def _name_from_folder(path):
    m = re.match(r"aim_([A-Za-z]+)", os.path.basename(path))
    return m.group(1) if m else None


def find_sessions(log=print):
    """解析するセッションの一覧 [{dir, participant, dpi, dpi_recorded, group, source}]。"""
    if K.SESSIONS:
        cand = [dict(s) for s in K.SESSIONS]
    else:
        cand = [dict(dir=d) for d in sorted(glob.glob(os.path.join(K.RECORDINGS, "aim_*")))]
    out = []
    for s in cand:
        d = s["dir"] if os.path.isabs(s["dir"]) else os.path.join(K.ROOT, s["dir"])
        sj = os.path.join(d, "session.json")
        if not os.path.exists(sj):
            log(f"  ! {d}: session.json が無いので飛ばします")
            continue
        sess = json.load(open(sj, encoding="utf-8"))
        ov = {}
        if os.path.exists(os.path.join(d, OVERRIDE_NAME)):
            ov = json.load(open(os.path.join(d, OVERRIDE_NAME), encoding="utf-8"))
        rec = float(sess.get("config", {}).get("mouse_dpi", np.nan))
        name = s.get("participant") or ov.get("participant") or _name_from_folder(d) or sess.get("participant")
        dpi, src = None, None
        for v, how in ((s.get("dpi"), "config"), (ov.get("dpi"), "override"),
                       (_dpi_from_name(d), "folder"), (rec, "session.json")):
            if v is not None and np.isfinite(float(v)):
                dpi, src = float(v), how
                break
        name = display_name(name)
        row = {"dir": os.path.normpath(d), "session": os.path.basename(d), "participant": name,
               "group": group_of(name), "dpi": dpi, "dpi_recorded": rec, "dpi_source": src}
        if K.DPIS and dpi not in [float(x) for x in K.DPIS]:
            continue
        if row["group"] is None:
            log(f"  ! {row['session']}: 参加者 {name} が config.GROUPS に無いので群なしで扱います")
        if np.isfinite(rec) and abs(rec - dpi) > 1e-6:
            log(f"  - {row['session']}: 記録時 DPI {rec:g} → 実際 {dpi:g}（{src}）。counts→mm を {rec / dpi:.3f} 倍に直します")
        out.append(row)
    # 同じ人・同じ DPI が複数あれば新しい方（フォルダ名の時刻が後）
    keep = {}
    for r in out:
        k = (r["participant"], r["dpi"])
        if k in keep:
            log(f"  ! {r['participant']} DPI {r['dpi']:g}: セッションが複数 → {r['session']} を使います")
        keep[k] = r
    return sorted(keep.values(), key=lambda r: (r["participant"], r["dpi"]))


def list_trials(session_dir, kinds=("track", "static")):
    """セッションの試行 [(dir, meta)]。やり直しは最後の attempt だけ。練習は除く。"""
    best = {}
    for d in sorted(glob.glob(os.path.join(session_dir, "trial_*"))):
        f = os.path.join(d, "meta.json")
        if not os.path.exists(f):
            continue
        meta = json.load(open(f, encoding="utf-8"))
        task = meta.get("task", {})
        if task.get("kind") not in kinds or task.get("practice"):
            continue
        key = (task.get("kind"), task.get("band"), task.get("seed"), task.get("name"))
        if key not in best or task.get("attempt", 1) >= best[key][1].get("task", {}).get("attempt", 1):
            best[key] = (d, meta)
    return sorted(best.values(), key=lambda r: r[1].get("task", {}).get("index", 0))


# ------------------------------------------------------------------ 試行
def _read_task(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    g = lambda k: np.array([float(r[k]) for r in rows])
    return (g("t"), g("epoch"),
            np.stack([g("target_x_cm"), g("target_y_cm")], 1) * 10.0,
            np.stack([g("cursor_x_cm"), g("cursor_y_cm")], 1) * 10.0,
            np.stack([g("mouse_dx_counts"), g("mouse_dy_counts")], 1),
            g("on_target"))


class Trial:
    """1 試行。時刻は課題時刻（解析区間の先頭 = 0 s）、関節はカメラ座標 [mm] のまま持ち、
    set_frame() で机座標（x = マウスの右、y = 奥、z = 上）に移して前処理する。"""

    def __init__(self, trial_dir, sess):
        self.dir = trial_dir
        self.name = os.path.basename(trial_dir)
        self.sess = sess
        self.participant, self.group, self.dpi = sess["participant"], sess["group"], sess["dpi"]
        self.session = sess["session"]
        self.meta = json.load(open(os.path.join(trial_dir, "meta.json"), encoding="utf-8"))
        task = self.meta.get("task", {})
        self.kind = task.get("kind", "track")
        self.band, self.seed, self.traj = task.get("band"), task.get("seed"), task.get("trajectory")
        sj = json.load(open(os.path.join(sess["dir"], "session.json"), encoding="utf-8"))
        cfg = sj.get("config", {})
        self.T = float(cfg.get("trial_s", 40.0))
        cm_per_count = float(sj.get("cm_per_count", 2.54 / float(cfg.get("mouse_dpi", self.dpi))))
        rec = sess.get("dpi_recorded", np.nan)
        if np.isfinite(rec) and abs(rec - self.dpi) > 1e-6:
            cm_per_count *= rec / self.dpi
        self.cm_per_count = cm_per_count

        # ---- 課題のログ
        t, ep, tgt, cur, dm, on = _read_task(os.path.join(trial_dir, "task.csv"))
        self.task_t, self.task_epoch = t, ep
        self.fs_task = 1.0 / np.median(np.diff(t))
        if self.kind == "static":                   # 静止試行は 10 s 程度
            self.T = float(np.floor(t[-1]))
        self.tg = np.arange(int(round(self.T * K.FS_U))) / K.FS_U          # 解析グリッド
        mouse = np.cumsum(dm, axis=0) * cm_per_count * 10.0                 # マウスの積算変位 [mm]
        mouse_lp = sig.lowpass(mouse, self.fs_task)
        self.mouse = sig.resample(t, mouse_lp, self.tg)                     # (N,2) マウス座標系 [mm]
        self.mouse_raw_t = t
        self.mouse_raw = mouse
        self.tgt = sig.resample(t, tgt, self.tg)
        self.cur = sig.resample(t, cur, self.tg)
        self.on_target = sig.resample(t, on.astype(float), self.tg) > 0.5
        self.rms_err_mm = float(np.sqrt(np.nanmean(np.sum((self.cur - self.tgt) ** 2, 1))))
        self.on_target_pct = float(100 * np.nanmean(self.on_target))

        # ---- 関節（カメラ座標）
        f = os.path.join(trial_dir, "joints", "rs_joints.npz")
        self.has_rs = os.path.exists(f)
        self.frame = None
        if not self.has_rs:
            return
        r = np.load(f, allow_pickle=False)
        names = [str(n) for n in r["names"]]
        P = r["P_C"].astype(float)
        st = r["status"] if "status" in r.files else None
        if st is not None:
            P[st != 0] = np.nan
        self.cam = {n: P[:, names.index(n)] for n in JOINTS if n in names}
        self.rs_t = np.interp(r["t"].astype(float), ep, t, left=np.nan, right=np.nan)
        ok_t = np.isfinite(self.rs_t)
        self.fs_rs = 1.0 / np.median(np.diff(self.rs_t[ok_t])) if ok_t.sum() > 2 else np.nan
        span = ok_t & (self.rs_t >= 0) & (self.rs_t < self.T)
        self.in_span = span
        self.valid_pct = {n: float(100 * np.mean(np.isfinite(self.cam[n][span]).all(1))) for n in self.cam}
        self.lr_swapped_pct = (float(100 * np.mean(r["lr_swapped"][span].astype(bool)))
                               if "lr_swapped" in r.files else np.nan)

    # ---- 机座標
    def palm_cam(self):
        return 0.5 * (self.cam["wrist"] + self.cam["mcp3"])

    def set_frame(self, frame, scale=None):
        """机座標に移して前処理（Hampel + 5 Hz 低域通過 → 解析グリッド）。scale は擬似データ用。"""
        self.frame = frame
        R, o = np.asarray(frame["R"]), np.asarray(frame["origin"])
        self.J, self.Jraw = {}, {}
        cam = self.cam if scale is None else scale
        for n, X in cam.items():
            D = (X - o) @ R
            self.Jraw[n] = sig.resample(self.rs_t, D, self.tg)
            Y = sig.lowpass(sig.hampel(D), self.fs_rs)
            self.J[n] = sig.resample(self.rs_t, Y, self.tg)
        self.J["palm"] = 0.5 * (self.J["wrist"] + self.J["mcp3"])

    # ---- 入力周波数
    def bins(self, ax=None):
        """入力周波数のビン番号（ax = "x" / "y" / None = 両方）。"""
        if not self.traj:
            return np.array([], int)
        axes = [ax] if ax else ["x", "y"]
        return np.array(sorted(int(n) for a in axes for n in self.traj["axes"][a]["n"]), int)

    def fmax(self):
        return max(self.bins()) / self.T if self.traj else np.nan

    def key(self):
        return f"{self.participant}/{self.session}/{self.name}"

    def header(self):
        return {"participant": self.participant, "group": self.group, "dpi": self.dpi, "band": self.band,
                "seed": self.seed, "session": self.session, "trial": self.name,
                "cond": f"{int(self.dpi)}_{self.band}"}


def fit_desk_frame(trials):
    """セッションの本番試行をまとめて、マウスの積算変位に palm の 3 次元変位を回帰し机の 2 軸を求める。

    palm(t) = o_試行 + ex·mx(t) + ey·my(t)。|ex|, |ey| = マウス 1 mm あたりカメラで何 mm 動いたか
    （1 に近いほど深度の尺度とマウスの換算が合っている）。
    """
    Xs, Ms, ids = [], [], []
    for i, tr in enumerate(trials):
        if not tr.has_rs or tr.kind != "track":
            continue
        X = tr.palm_cam()
        M = np.column_stack([np.interp(tr.rs_t, tr.mouse_raw_t, tr.mouse_raw[:, k]) for k in range(2)])
        ok = tr.in_span & np.isfinite(X).all(1) & np.isfinite(M).all(1)
        if ok.sum() < 100:
            continue
        Xs.append(X[ok]); Ms.append(M[ok]); ids.append(np.full(ok.sum(), len(Xs) - 1))
    if not Xs:
        return {"R": np.eye(3).tolist(), "origin": [0, 0, 0], "source": "カメラ座標（回帰できず）",
                "gain_x": np.nan, "gain_y": np.nan, "resid_mm": np.nan, "axis_angle_deg": np.nan, "n": 0}
    X = np.vstack(Xs); M = np.vstack(Ms); ids = np.concatenate(ids)
    D = np.zeros((len(X), len(Xs)))
    D[np.arange(len(X)), ids] = 1
    A = np.column_stack([M, D])
    sol, *_ = np.linalg.lstsq(A, X, rcond=None)
    ex, ey = sol[0], sol[1]
    gx, gy = float(np.linalg.norm(ex)), float(np.linalg.norm(ey))
    resid = float(np.sqrt(np.mean(np.sum((X - A @ sol) ** 2, 1))))
    ang = float(np.degrees(np.arccos(np.clip((ex / gx) @ (ey / gy), -1, 1))))
    ex = ex / gx
    ey = ey - (ey @ ex) * ex
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)                   # マウスの y は奥が正 → ez = 上
    return {"R": np.column_stack([ex, ey, ez]).tolist(), "origin": sol[2:].mean(0).tolist(),
            "source": f"マウス回帰（{K.DESK_POINT}、{len(Xs)} 試行）", "gain_x": gx, "gain_y": gy,
            "axis_angle_deg": ang, "resid_mm": resid, "n": int(len(X))}


def load_session(sess, log=print):
    """セッションの試行を読み、机座標を決めて前処理まで済ませる。(本番試行の一覧, 静止試行 or None, frame)"""
    rows = list_trials(sess["dir"])
    tracks, static = [], None
    for d, meta in rows:
        task = meta.get("task", {})
        if task.get("kind") == "static":
            tr = Trial(d, sess)
            if tr.has_rs:
                static = tr
            continue
        if K.BANDS and task.get("band") not in K.BANDS:
            continue
        tr = Trial(d, sess)
        if f"{tr.participant}/{tr.session}/{tr.name}" in K.EXCLUDE:
            log(f"  - {tr.key()}: config.EXCLUDE により除外")
            continue
        if not tr.has_rs:
            log(f"  ! {tr.key()}: joints/rs_joints.npz が無いので飛ばします")
            continue
        tracks.append(tr)
    frame = fit_desk_frame(tracks)
    for tr in tracks:
        tr.set_frame(frame)
    if static is not None:
        static.set_frame(frame)
    return tracks, static, frame
