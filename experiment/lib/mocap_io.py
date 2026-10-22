"""
mocap_io.py
────────────────────────────────────────────────────────────────────
Motive の書き出し CSV（Format Version 1.23）の読み込み．

  take = Take(path)
  take.t                … Take 先頭からの時刻 [s]
  take.start_epoch      … Capture Start Time（epoch 秒）．RealSense の時刻と比べるときに使う
  take.names()          … 列の (Type, Name) の一覧（Marker / Rigid Body / Bone など）
  take.position(name)   … その列の位置 (frames × 3) [mm]．隠れたフレームは NaN
  take.markers()        … ラベルの有無を問わず全マーカーの位置 (frames × markers × 3) [mm]
  take.rotation(name)   … その列の回転 (frames × 3 or 4)．Quaternion なら [X,Y,Z,W]，
                          Euler なら書き出し順の 3 角度 [deg]．無ければ None
  take.rotation_mat(nm) … 回転行列 (frames × 3 × 3)．回転が無ければ None
  take.rotation_names() … 回転を持つ列の (Type, Name) の一覧
  take.rotation_type    … "Quaternion" / "Euler"
  take.rotation_order   … Euler のときの軸順（例 "ZXY"）

  CSV の先頭 7 行:
    1 行目: Take の情報（Capture Start Time, Length Units など）
    3 行目: Type  4 行目: Name  5 行目: ID  6 行目: Position / Rotation  7 行目: X / Y / Z
────────────────────────────────────────────────────────────────────
"""

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

TZ_HOURS = 9          # Motive PC のタイムゾーン（Capture Start Time は現地時刻で書かれる）
UNIT_MM = {"Meters": 1000.0, "Centimeters": 10.0, "Millimeters": 1.0}


def parse_capture_start(s):
    """'2026-10-15 07.39.46.497 午後' / '... PM' / 24 時間表記 → epoch [s]"""
    pm = any(x in s for x in ("午後", "PM", "pm"))
    am = any(x in s for x in ("午前", "AM", "am"))
    for x in ("午後", "午前", "PM", "AM", "pm", "am"):
        s = s.replace(x, "")
    date, clock = s.split()[:2]
    h, m, sec, *ms = clock.replace(":", ".").split(".")
    h = int(h) + (12 if pm and int(h) < 12 else 0) - (12 if am and int(h) == 12 else 0)
    y, mo, d = map(int, date.replace("/", "-").split("-"))
    t = dt.datetime(y, mo, d, h, int(m), int(sec), int(ms[0].ljust(3, "0")[:3]) * 1000 if ms else 0,
                    tzinfo=dt.timezone(dt.timedelta(hours=TZ_HOURS)))
    return t.timestamp()


class Take:
    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, encoding="utf-8-sig", errors="replace") as f:   # 「午後」などの日本語を含む
            head = [r for r, _ in zip(csv.reader(f), range(7))]
        self.meta = dict(zip(head[0][0::2], head[0][1::2]))
        self.types, self.col_names, self.kinds, self.axes = head[2], head[3], head[5], head[6]
        self.fps = float(self.meta["Export Frame Rate"])
        self.start_epoch = parse_capture_start(self.meta["Capture Start Time"])
        self.scale = UNIT_MM[self.meta["Length Units"]]
        if self.meta.get("Coordinate Space", "Global") != "Global":      # ローカル座標では比較できない
            raise SystemExit(f"[ERROR] {self.path.name} は Coordinate Space = Global で書き出してください")

        # データ本体（ヘッダ 7 行の後）．列数が多いので float32 で読む
        df = pd.read_csv(self.path, skiprows=7, header=None, names=range(len(self.types)), dtype=np.float32)
        self.t = df[1].to_numpy(float)
        self._data = df

    # ── 列の検索 ──────────────────────────────────
    def _xyz_cols(self, j):
        return [j, j + 1, j + 2]

    def _position_starts(self, types):
        """指定した Type の Position 列のうち X の列番号の一覧．"""
        return [j for j in range(len(self.types))
                if self.types[j] in types and self.kinds[j] == "Position" and self.axes[j] == "X"]

    def names(self):
        """(Type, Name) の一覧（位置を持つ列だけ）．"""
        return [(self.types[j], self.col_names[j])
                for j in self._position_starts({"Marker", "Rigid Body", "Rigid Body Marker", "Bone", "Bone Marker"})]

    # ── 回転 ──────────────────────────────────────
    @property
    def rotation_type(self):
        """"Quaternion" / "Euler"．メタに無ければ列の並び（W があるか）から決める．"""
        t = self.meta.get("Rotation Type")
        if t:
            return t
        return "Quaternion" if any(self.kinds[j] == "Rotation" and self.axes[j] == "W"
                                   for j in range(len(self.types))) else "Euler"

    @property
    def rotation_order(self):
        """Euler のときの軸順（例 "ZXY"）．Quaternion なら None．"""
        if self.rotation_type.lower().startswith("quat"):
            return None
        for j in self._rotation_starts({"Bone", "Rigid Body"}):
            return "".join(self.axes[j + k] for k in range(3))
        return "ZXY"

    def _rotation_starts(self, types):
        """指定した Type の Rotation 列のうち最初の軸の列番号の一覧．"""
        out = []
        for j in range(len(self.types)):
            if self.types[j] not in types or self.kinds[j] != "Rotation":
                continue
            if j > 0 and self.kinds[j - 1] == "Rotation" and self.types[j - 1] == self.types[j] \
                    and self.col_names[j - 1] == self.col_names[j]:
                continue                       # 2 軸目以降は飛ばす
            out.append(j)
        return out

    def rotation_names(self):
        """(Type, Name) の一覧（回転を持つ列だけ）．"""
        return [(self.types[j], self.col_names[j])
                for j in self._rotation_starts({"Bone", "Rigid Body"})]

    def rotation(self, name, types=("Bone", "Rigid Body")):
        """Name が name の回転．Quaternion なら (N,4)[X,Y,Z,W]，Euler なら (N,3)[deg]．"""
        n_ax = 4 if self.rotation_type.lower().startswith("quat") else 3
        for j in self._rotation_starts(set(types)):
            full = self.col_names[j]
            if full == name or full.split(":")[-1] == name:
                return self._data[[j + k for k in range(n_ax)]].to_numpy(float)
        return None

    def rotation_mat(self, name, types=("Bone", "Rigid Body")):
        """回転行列 (N,3,3)．回転が書き出されていなければ None．"""
        from . import angles as _an
        Rr = self.rotation(name, types)
        if Rr is None:
            return None
        if Rr.shape[1] == 4:
            return _an.quat_to_mat(Rr)
        return _an.euler_to_mat(Rr, self.rotation_order)

    def position(self, name, types=("Marker", "Bone", "Bone Marker", "Rigid Body", "Rigid Body Marker")):
        """Name が name（"Skeleton:RUArm" の "Skeleton:" は省略可）の列の位置 [mm]．なければ None．

        ボーン（関節中心）でもラベル付きマーカーでもボーンマーカーでも同じ名前で引ける．
        """
        for j in self._position_starts(set(types)):
            full = self.col_names[j]
            if full == name or full.split(":")[-1] == name:
                return self._data[self._xyz_cols(j)].to_numpy(float) * self.scale
        return None

    def markers(self):
        """全マーカー（Type = Marker）の位置 (frames × markers × 3) [mm]．"""
        cols = self._position_starts({"Marker"})
        if not cols:                     # 書き出し設定でマーカーが含まれていない
            raise SystemExit(f"[ERROR] {self.path.name} に Marker の Position 列がありません（Motive の書き出し設定を確認）")
        X = self._data[[c + k for c in cols for k in range(3)]].to_numpy(float)
        return X.reshape(len(self.t), len(cols), 3) * self.scale