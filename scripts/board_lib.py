# キャリブレーションボード（チェッカー + 反射マーカ3点）の共通処理．
# calib_mocap_rs.py と，後で作る誤差検証スクリプトから import して使う．
#
# 座標系と単位
#   ボード  : 内側コーナー#0 が原点．X右, Y下, Zボード奥（カメラから離れる向き）．[m]
#   カメラ  : RealSense カラー光学座標系（X右, Y下, Z前方）．[m]
#             rs_estimate.py の骨格も depth を color に align して color の内部パラメータで
#             deproject しているので，同じ座標系．
#   MoCap   : Motive のグローバル座標系（CSV の値をそのまま使い，単位だけ m に揃える）．

import csv
from itertools import combinations

import numpy as np
import cv2
import pyrealsense2 as rs

# ── ボード寸法 ────────────────────────────────────
PATTERN = (7, 4)     # 内部コーナー数（X方向, Y方向）
SQ = 0.025           # マス目 [m]

OBJP = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float64)
OBJP[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQ

# マーカ中心のボード座標 [m]．Z はマーカ中心の板面からの高さ（カメラ側が負）．
# 平らなステッカーなら ≒ 0（厚みの半分）．半球/球マーカに替えたら -半径 にする．
MARKER_CENTER_Z = 0.0
MARKERS_BOARD = np.array([
    [-0.040, -0.040, MARKER_CENTER_Z],   # S1 左上
    [ 0.190, -0.040, MARKER_CENTER_Z],   # S2 右上
    [-0.040,  0.115, MARKER_CENTER_Z],   # S3 左下
])
MARKER_NAMES = ("S1", "S2", "S3")


# ── RealSense: PnP によるボード姿勢（board_calcAxis.py と同じ手順） ──
K0, D0 = np.eye(3), np.zeros(5)


def to_normalized(intr, px):
    """画素 -> 正規化像座標．RealSense のカラーは inverse_brown_conrady のことが多く，
    係数をそのまま OpenCV の distCoeffs に渡すと歪みを逆向きに掛けてしまう．
    deproject に任せて，以降は K=I, dist=0 で扱う．"""
    return np.array([rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], 1.0)[:2]
                     for u, v in px])


def to_pixels(intr, pts):
    return np.array([rs.rs2_project_point_to_pixel(intr, list(map(float, p))) for p in pts])


def solve(intr, px):
    n = to_normalized(intr, px).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(OBJP, n, K0, D0, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None, None
    return cv2.solvePnPRefineLM(OBJP, n, K0, D0, rvec, tvec)


def cam_pts(rvec, tvec, pts):
    return (cv2.Rodrigues(rvec)[0] @ pts.T).T + tvec.reshape(3)


def is_upright(gray, intr, rvec, tvec):
    """原点の左上マスは黒，右上マスは白であるはず．逆ならコーナー列が180度反転している．"""
    probe = np.array([[-SQ / 2, -SQ / 2, 0.0], [SQ / 2, -SQ / 2, 0.0]])
    px = to_pixels(intr, cam_pts(rvec, tvec, probe)).astype(int)
    H, W = gray.shape
    vals = []
    for u, v in px:
        if not (2 <= u < W - 2 and 2 <= v < H - 2):
            return True                      # 画面外なら判定不能．そのまま採用
        vals.append(gray[v - 2:v + 3, u - 2:u + 3].mean())
    return vals[0] < vals[1]


def board_pose(img, intr):
    """カラー画像からボード姿勢を求める．
    戻り値: (R, t, px, rms_px) または None．p_cam = R @ p_board + t"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, PATTERN)
    if not found:
        return None
    px = corners.reshape(-1, 2).astype(np.float64)
    rvec, tvec = solve(intr, px)
    if rvec is not None and not is_upright(gray, intr, rvec, tvec):
        px = px[::-1].copy()                 # 逆順 = 180度回転に相当
        rvec, tvec = solve(intr, px)
    if rvec is None:
        return None
    rms = np.sqrt(((to_pixels(intr, cam_pts(rvec, tvec, OBJP)) - px) ** 2).sum(1).mean())
    return cv2.Rodrigues(rvec)[0], tvec.reshape(3), px, rms


# ── Motive: CSV 読み込みとマーカ識別 ──────────────
UNIT_SCALE = {"meters": 1.0, "centimeters": 0.01, "millimeters": 0.001}


def load_motive_csv(path, marker_types=("Marker",)):
    """Motive の CSV エクスポート（Format Version 1.2x 以降）からマーカ位置を読む．
    戻り値: (times[s] (N,), frames: N 個の (k,3) 配列 [m], header dict)
    各フレームには XYZ が揃っているマーカだけを入れる（ラベルは使わない）．
    "Rigid Body Marker" は剛体から逆算した理論位置なので既定では使わない．"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    first = rows[0]
    header = {first[i].strip(): first[i + 1].strip() for i in range(0, len(first) - 1, 2)}
    units = header.get("Length Units", "").lower()
    if units not in UNIT_SCALE:
        print(f"[WARN] {path}: Length Units='{units}' を解釈できないので m とみなす")
    scale = UNIT_SCALE.get(units, 1.0)
    if header.get("Coordinate Space", "Global") != "Global":
        print(f"[WARN] {path}: Coordinate Space={header['Coordinate Space']}（Global を推奨）")

    h = next(i for i, r in enumerate(rows) if r and r[0].strip() == "Frame")
    types = next(r for r in rows[:h] if len(r) > 1 and r[1].strip() == "Type")
    names = next(r for r in rows[:h] if len(r) > 1 and r[1].strip() == "Name")
    cats, axes = rows[h - 1], rows[h]

    cols = {}   # name -> {"X": col, "Y": col, "Z": col}
    for j in range(2, len(axes)):
        if (j < len(types) and types[j].strip() in marker_types
                and cats[j].strip() == "Position" and axes[j].strip() in "XYZ"):
            cols.setdefault(names[j].strip(), {})[axes[j].strip()] = j
    cols = {k: (v["X"], v["Y"], v["Z"]) for k, v in cols.items() if len(v) == 3}
    if not cols:
        raise ValueError(f"{path}: Type={marker_types} の Position 列が見つからない")

    times, frames = [], []
    for r in rows[h + 1:]:
        if len(r) < 2 or not r[1].strip():
            continue
        pts = []
        for cx, cy, cz in cols.values():
            try:
                pts.append([float(r[cx]), float(r[cy]), float(r[cz])])
            except (ValueError, IndexError):
                pass                          # 欠損（空欄）
        times.append(float(r[1]))
        frames.append(np.array(pts).reshape(-1, 3) * scale)
    return np.array(times), frames, header


def nominal_distances(markers=MARKERS_BOARD):
    """(|S1S2|, |S1S3|, |S2S3|) [m]"""
    return np.array([np.linalg.norm(markers[a] - markers[b])
                     for a, b in ((0, 1), (0, 2), (1, 2))])


def identify_markers(pts, tol=0.010):
    """点群から S1,S2,S3 を距離だけで識別する．
    三辺の長さが全て異なる（155 < 230 < 277 mm）ことを利用する:
      最短辺 = S1-S3，最長辺 = S2-S3．両方に含まれる点が S3．
    3点より多い場合は公称距離に最も合う3点を選ぶ．
    戻り値: (3,3) [S1,S2,S3] または None（公称距離から tol 以上ずれる場合）"""
    if len(pts) < 3:
        return None
    nom = nominal_distances()
    best, best_err = None, np.inf
    for tri in combinations(range(len(pts)), 3):
        P = pts[list(tri)]
        pairs = [(0, 1), (0, 2), (1, 2)]
        d = np.array([np.linalg.norm(P[a] - P[b]) for a, b in pairs])
        short, long_ = pairs[int(np.argmin(d))], pairs[int(np.argmax(d))]
        s3 = (set(short) & set(long_)).pop()
        s1 = (set(short) - {s3}).pop()
        s2 = (set(long_) - {s3}).pop()
        Q = P[[s1, s2, s3]]
        err = np.abs(nominal_distances(Q) - nom).max()
        if err < best_err:
            best, best_err = Q, err
    return best if best_err < tol else None


# ── 剛体変換の推定 ────────────────────────────────
def fit_rigid(A, B):
    """B ≈ R @ A + t となる R (det=+1), t を最小二乗で求める（Kabsch）．
    A, B: (N,3)．戻り値: (4,4) 同次変換．"""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cb - R @ ca
    return T


def transform(T, P):
    return P @ T[:3, :3].T + T[:3, 3]
