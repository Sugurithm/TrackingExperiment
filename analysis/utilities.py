#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utilities.py — 関節解析スクリプトの共通部品

  1. 試行データ   : TrialData（変位・ノルム・相対変位を取り出す）と load_trial()
  2. 試行の探索   : Session / Trial と、config（SESSIONS, ATTEMPT）からの探し方
  3. 図・表の補助 : 被験者の並び順、図の保存

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
# 3. 図・表の補助
# =====================================================================
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