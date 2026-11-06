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
from scipy.signal import butter, filtfilt

# ---------- 共通の設定 ----------
JOINTS = ["shoulder", "elbow", "wrist", "mcp3"]   # 関節（根元から末端への連結順）
NPZ_PATH = Path("joints") / "rs_joints.npz"       # 試行フォルダ内の関節データ
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
    tt: np.ndarray            # 時間 [s]（先頭を 0 とする）
    fs: float                 # サンプリング周波数 [Hz]
    mean: dict                # 関節ごとの平均位置 (3,)
    disp: dict                # 関節ごとの変位 (N, 3)
    joints: list = field(default_factory=lambda: list(JOINTS))

    def norm(self, joint):
        """変位の大きさ √(dx²+dy²+dz²)  (N,)"""
        return np.linalg.norm(self.disp[joint], axis=1)

    def relative(self, proximal, distal):
        """相対変位 = 遠位の変位 - 近位の変位  (N, 3)"""
        return self.disp[distal] - self.disp[proximal]

    def pairs(self):
        return adjacent_pairs(self.joints)


def load_trial(trial_dir, joints=JOINTS, fc=5.0):
    """
    npzから3次元関節座標を読み，次の順に前処理して TrialData を返す．
    ・時刻順に並べる → 等間隔グリッドへ線形補間(欠損も埋める) → ゼロ位相バターワース・ローパス(遮断 fc [Hz])→ 試行平均を引いて変位にする
    """
    z = np.load(Path(trial_dir) / NPZ_PATH)
    names = [str(n) for n in z["names"]]
    t = z["t"].astype(float)
    P = z["P_C"].astype(float)            # カメラ座標系の位置 (フレーム数, 関節数, 3)

    # 時刻順に並べる
    order = np.argsort(t)
    t, P = t[order], P[order]

    # サンプリング周波数を求める．
    fs = 1.0 / np.median(np.diff(t))
    # 等間隔グリッドへ線形補間(欠損も埋める)
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    b, a = butter(FILTER_ORDER, min(fc / (fs / 2), MAX_NORM_FREQ))

    mean, disp = {}, {}
    for j in joints:
        # 関節jの3次元座標を取り出す．
        X = P[:, names.index(j), :]
        # 欠損（NaN/Inf）でないフレームを抽出する．
        ok = np.isfinite(X).all(axis=1)
        # 線形補間で等間隔グリッドに変換する．
        Y = np.column_stack([np.interp(grid, t[ok], X[ok, k]) for k in range(3)])
        Y = filtfilt(b, a, Y, axis=0)

        # 試行平均を引いて変位にする．
        mean[j] = Y.mean(axis=0)
        disp[j] = Y - mean[j]
    return TrialData(tt=grid - grid[0], fs=fs, mean=mean, disp=disp, joints=list(joints))


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