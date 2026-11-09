#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utilities.py — 関節解析スクリプトの共通部品

  1. 試行データ   : TrialData（変位・ノルム・相対変位・リンクを取り出す）と load_trial()
  2. 試行の探索   : Session / Trial と、config（SESSIONS, ATTEMPT）からの探し方
  3. リンク寄与率 : 射影寄与率・パワー寄与率（変位版・速度版で共通）
  4. 統計・図・表の補助 : ICC、被験者の並び順、図の保存

記号（スクリプト・資料で共通）
  x_k, x_tip : リンク k / 手先 MCP3 の変位（TrialData.disp 由来）
  v_k, v_tip : リンク k / 手先 MCP3 の速度（TrialData.vel 由来）
  4 リンクの和は手先になる：Σ_k x_k = x_tip,  Σ_k v_k = v_tip

解析スクリプト側には「何を計算して何を出力するか」だけを書き、
データの読み込み・前処理・探索はすべてここに置く。
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ---------- 共通の設定 ----------
JOINTS = ["shoulder", "elbow", "wrist", "mcp3"]   # 関節（根元から末端への連結順）
NPZ_PATH = Path("joints") / "rs_joints.npz"       # 試行フォルダ内の関節データ
TASK_CSV = "task.csv"                             # 試行フォルダ内の課題ログ（epoch で RealSense と時刻合わせ）
FILTER_ORDER = 4                                  # ローパスフィルタの次数
MAX_NORM_FREQ = 0.99                              # 遮断周波数の上限（ナイキスト比）

# リンク名: (近位の関節, 遠位の関節)。shoulder だけは肩そのもの（近位なし）
LINKS = {
    "shoulder":  (None, "shoulder"),
    "upper_arm": ("shoulder", "elbow"),
    "forearm":   ("elbow", "wrist"),
    "hand":      ("wrist", "mcp3"),
}
TIP = "mcp3"                                      # 手先（マウスを握る手の中心）

# 図の共通設定
BANDS = ["slow", "fast"]                          # 図の行の順
GROUP_COLOR = {"FPS": "#2a78d6", "nonFPS": "#eb6834"}
DPI = 120

_TRIAL_RE = re.compile(r"(slow|fast)_s(\d+)(?:_(\d+))?")   # trial_01_slow_s1_2 など
_SESSION_RE = re.compile(r"aim_([A-Za-z]+)(\d+)_")         # aim_tanaka800_... など


def adjacent_pairs(joints=JOINTS):
    """隣り合う関節のペア [(近位, 遠位), ...]"""
    return list(zip(joints[:-1], joints[1:]))


def has_joint_data(d):
    return (Path(d) / NPZ_PATH).exists()


# =====================================================================
# 1. 試行データ
# =====================================================================
@dataclass
class TrialData:
    """前処理済みの 1 試行。disp[関節] は平均を引いた変位 (N, 3) [mm]"""
    tt: np.ndarray            # 時間 [s]（切り出したときは課題時刻 t: 解析区間の先頭 = 0）
    fs: float                 # サンプリング周波数 [Hz]
    mean: dict                # 関節ごとの平均位置 (3,)
    disp: dict                # 関節ごとの変位 (N, 3)
    joints: list = field(default_factory=lambda: list(JOINTS))
    cursor: np.ndarray = None  # カーソル位置 (N, 2) [cm]（画面上の x 右・y 上）。task.csv がないと None
    target: np.ndarray = None  # ターゲット位置 (N, 2) [cm]（同上）
    cropped: bool = False     # 解析区間（task.csv の 0 ≤ t ≤ 最後）に切り出したか

    def norm(self, joint):
        """変位の大きさ √(dx²+dy²+dz²)  (N,)"""
        return np.linalg.norm(self.disp[joint], axis=1)

    def relative(self, proximal, distal):
        """相対変位 = 遠位の変位 - 近位の変位  (N, 3)"""
        return self.disp[distal] - self.disp[proximal]

    def vel(self, joint):
        """関節の速度 (N, 3) [mm/s]（中心差分）"""
        return np.gradient(self.disp[joint], 1.0 / self.fs, axis=0)

    def relative_vel(self, proximal, distal):
        """相対速度 = 遠位の速度 - 近位の速度  (N, 3) [mm/s]"""
        return self.vel(distal) - self.vel(proximal)

    def cursor_vel(self):
        """カーソルの速度 (N, 2) [cm/s]"""
        return None if self.cursor is None else np.gradient(self.cursor, 1.0 / self.fs, axis=0)

    def target_vel(self):
        """ターゲットの速度 (N, 2) [cm/s]"""
        return None if self.target is None else np.gradient(self.target, 1.0 / self.fs, axis=0)

    def link_disp(self):
        """各リンクの変位 x_k {名前: (N, 3)} [mm]。合計 = 手先の変位"""
        return {k: self.disp[b] if a is None else self.relative(a, b) for k, (a, b) in LINKS.items()}

    def link_vel(self):
        """各リンクの速度 v_k {名前: (N, 3)} [mm/s]。合計 = 手先の速度"""
        return {k: self.vel(b) if a is None else self.relative_vel(a, b) for k, (a, b) in LINKS.items()}

    def pairs(self):
        return adjacent_pairs(self.joints)


def read_task(trial_dir):
    """task.csv を読む。なければ None"""
    p = Path(trial_dir) / TASK_CSV
    return pd.read_csv(p) if p.exists() else None


def load_trial(trial_dir, joints=JOINTS, fc=5.0, crop=True):
    """
    npz から 3 次元関節座標を読み、次の順に前処理して TrialData を返す。
      時刻順に並べる → 等間隔グリッドへ線形補間（欠損も埋まる）
      → ゼロ位相バターワース・ローパス（遮断 fc [Hz]）
      → crop=True かつ task.csv があれば解析区間だけを切り出す
         （npz の t と task.csv の epoch はどちらも PC の時刻 [s]。
           記録にはカウントダウン・立ち上がり・終わりの余分が入っているので除く）
      → 区間の平均を引いて変位にする
    フィルタは切り出す前の全区間にかける（区間の端でフィルタの端の影響を受けないため）。
    task.csv があれば、カーソルとターゲットの位置も同じグリッド・同じフィルタで並べて
    cursor / target に入れる。
    """
    z = np.load(Path(trial_dir) / NPZ_PATH)
    names = [str(n) for n in z["names"]]
    t = z["t"].astype(float)
    P = z["P_C"].astype(float)            # カメラ座標系の位置 (フレーム数, 関節数, 3)

    order = np.argsort(t)
    t, P = t[order], P[order]

    fs = 1.0 / np.median(np.diff(t))      # 時間間隔の中央値から求める
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    b, a = butter(FILTER_ORDER, min(fc / (fs / 2), MAX_NORM_FREQ))

    Y = {}
    for j in joints:
        X = P[:, names.index(j), :]
        ok = np.isfinite(X).all(axis=1)   # 欠損（NaN/Inf）でないフレーム
        Yj = np.column_stack([np.interp(grid, t[ok], X[ok, k]) for k in range(3)])
        Y[j] = filtfilt(b, a, Yj, axis=0)

    tt = grid - grid[0]
    keep = np.ones(len(grid), bool)
    screen = {"cursor": None, "target": None}
    task = read_task(trial_dir) if crop else None
    if task is not None:
        # 課題時刻 t = epoch - offset（task_core: epoch = epoch0 + 経過時間，t = 経過時間 - 定数）
        offset = float(np.median(task["epoch"] - task["t"]))
        tt = grid - offset
        keep = (tt >= 0) & (tt <= task["t"].max())
        if keep.sum() < 2:
            raise ValueError(f"{trial_dir}: 関節データと task.csv の時刻が重なりません")
        for name in screen:
            C = np.column_stack([np.interp(tt, task["t"], task[f"{name}_{ax}_cm"]) for ax in "xy"])
            screen[name] = filtfilt(b, a, C, axis=0)[keep]
        tt = tt[keep]
    elif crop:
        print(f"[注意] task.csv がないので切り出しません: {trial_dir}")

    mean, disp = {}, {}
    for j in joints:
        Yj = Y[j][keep]
        mean[j] = Yj.mean(axis=0)
        disp[j] = Yj - mean[j]
    return TrialData(tt=tt, fs=fs, mean=mean, disp=disp, joints=list(joints),
                     cursor=screen["cursor"], target=screen["target"], cropped=task is not None)


# =====================================================================
# 2. 試行の探索
# =====================================================================
@dataclass
class Session:
    """config の SESSIONS の 1 行に対応するセッション"""
    dir: Path
    label: str
    person: str
    group: str
    dpi: object


@dataclass
class Trial:
    """1 試行（帯域・seed・やり直し番号とフォルダ）"""
    band: str
    seed: int
    attempt: int
    dir: Path

    @property
    def name(self):
        return self.dir.name


def session_info(entry, root):
    """SESSIONS の 1 行から Session を作る。相対パスは root（リポジトリ直下）基準"""
    d = Path(entry["dir"])
    d = d if d.is_absolute() else Path(root) / d
    m = _SESSION_RE.match(d.name)
    return Session(
        dir=d,
        label=entry.get("label", d.name.split("_")[1] if "_" in d.name else d.name),
        person=entry.get("person", m.group(1) if m else d.name),
        group=entry.get("group", ""),
        dpi=entry.get("dpi", int(m.group(2)) if m else ""),
    )


def trial_key(name):
    """フォルダ名から (帯域, seed, やり直し番号)。該当しなければ None"""
    m = _TRIAL_RE.search(name.lower())
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3) or 1)


def find_trials(session_dir, attempt="all"):
    """
    セッション内の試行を (帯域, seed) 順に返す。
    同じ (帯域, seed) のやり直しは attempt = "first" / "last" で 1 つに絞る（それ以外は全部）。
    """
    groups = {}
    for d in sorted(Path(session_dir).iterdir()):
        key = trial_key(d.name)
        if key and has_joint_data(d):
            groups.setdefault(key[:2], []).append((key[2], d))

    trials = []
    for (band, seed) in sorted(groups):
        tries = sorted(groups[(band, seed)])
        if attempt == "first":
            tries = tries[:1]
        elif attempt == "last":
            tries = tries[-1:]
        trials += [Trial(band, seed, att, d) for att, d in tries]
    return trials


def iter_config_trials(cfg, root):
    """config（SESSIONS, ATTEMPT）に載っている全試行を (Session, Trial) で順に返す"""
    for entry in cfg.SESSIONS:
        ses = session_info(entry, root)
        if not ses.dir.exists():
            print(f"見つかりません: {ses.dir}")
            continue
        for trial in find_trials(ses.dir, cfg.ATTEMPT):
            yield ses, trial



# =====================================================================
# 3. リンク寄与率（変位版・速度版で共通）
# =====================================================================
def projection_shares(W, tip, mask=None):
    """
    射影寄与率  share_k = Σ_{t∈mask} w_k·tip / Σ_{t∈mask} |tip|²   （4 リンクの合計 = 1）
    W: {リンク名: (N, 3)}（変位 x_k でも速度 v_k でもよい），tip: (N, 3)（同じ量の手先）
    mask: bool (N,) または None（全フレーム）。手先と同じ向きの動きは正、直交は 0、逆向きは負。
    """
    m = slice(None) if mask is None else mask
    t = tip[m]
    den = np.sum(t * t)
    return {k: float(np.sum(w[m] * t) / den) for k, w in W.items()}


def power_shares(W, mask=None):
    """パワー寄与率  share_k = Σ|w_k|² / Σ_j Σ|w_j|²   （ノイズ補正なしの補助指標。常に 0 以上）"""
    m = slice(None) if mask is None else mask
    pw = {k: float(np.sum(w[m] ** 2)) for k, w in W.items()}
    total = sum(pw.values())
    return {k: v / total for k, v in pw.items()}


# =====================================================================
# 4. 統計・図・表の補助
# =====================================================================
def icc_oneway(values, groups):
    """一元配置の級内相関係数 ICC(1)（個人差の大きさ）"""
    values, groups = np.asarray(values, float), np.asarray(groups)
    labels = np.unique(groups)
    n_i = np.array([np.sum(groups == g) for g in labels])
    m_i = np.array([values[groups == g].mean() for g in labels])
    N, k = len(values), len(labels)

    ss_b = np.sum(n_i * (m_i - values.mean()) ** 2)
    ss_w = sum(np.sum((values[groups == g] - m) ** 2) for g, m in zip(labels, m_i))

    ms_b, ms_w = ss_b / (k - 1), ss_w / (N - k)
    n0 = (N - np.sum(n_i ** 2) / N) / (k - 1)
    return (ms_b - ms_w) / (ms_b + (n0 - 1) * ms_w)


def person_order(df):
    """グループ順 → 名前順に並べた被験者リストと {被験者: グループ}"""
    p = df[["person", "group"]].drop_duplicates().sort_values(["group", "person"])
    return list(p.person), dict(zip(p.person, p.group))


def save_fig(fig, path, dpi):
    """レイアウトを整えて保存し、図を閉じる"""
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)