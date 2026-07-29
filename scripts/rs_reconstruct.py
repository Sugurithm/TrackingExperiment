# rs_estimate3d.py で保存した skeleton3d.npz を読み込み，
# Open3D で骨格を3D再構成して再生する．
# タイムスタンプに従って記録時と同じ速さで再生する．

import time
from pathlib import Path

import numpy as np
import open3d as o3d

RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"
FILEPATH = RECORDINGS_DIR / "rec_20260729_113731" / "skeleton3d.npz"

# ── 表示パラメータ ────────────────────────────────
JOINT_RADIUS_HAND = 0.006   # [m] 手指関節の球半径
JOINT_RADIUS_ARM  = 0.012   # [m] 腕関節の球半径
SPHERE_RESOLUTION = 4       # 球の分割数（大きいほど滑らか・重い）

# ── 色定義 ────────────────────────────────────────
COLOR_THUMB  = [1.00, 0.35, 0.35]   # 親指：赤
COLOR_INDEX  = [1.00, 0.78, 0.15]   # 人差指：黄
COLOR_MIDDLE = [0.35, 1.00, 0.45]   # 中指：緑
COLOR_RING   = [0.30, 0.70, 1.00]   # 薬指：青
COLOR_PINKY  = [0.80, 0.45, 1.00]   # 小指：紫
COLOR_PALM   = [0.85, 0.85, 0.92]   # 手のひら（横方向の連結）：白
COLOR_ARM_POINT = [1.00, 0.55, 0.10]
COLOR_ARM_BONE  = [0.90, 0.40, 0.00]
COLOR_GRID_FLOOR = [0.3, 0.3, 0.35]
COLOR_GRID_BACK  = [0.25, 0.25, 0.30]

# ── 骨格接続定義 ──────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
ARM_CONNECTIONS = [(0, 1), (1, 2)]   # 肩-肘，肘-手首

# ── 指ごとの色割り当て ────────────────────────────
FINGER_LANDMARKS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}
FINGER_COLOR = {
    "thumb": COLOR_THUMB, "index": COLOR_INDEX, "middle": COLOR_MIDDLE,
    "ring": COLOR_RING, "pinky": COLOR_PINKY,
}
FINGER_OF = {i: name for name, ids in FINGER_LANDMARKS.items() for i in ids}

HAND_POINT_COLORS = np.tile(COLOR_PALM, (21, 1))   # 0番（手首）は白のまま
for _name, _ids in FINGER_LANDMARKS.items():
    HAND_POINT_COLORS[_ids] = FINGER_COLOR[_name]
ARM_POINT_COLORS = np.tile(COLOR_ARM_POINT, (3, 1))


def bone_color(i, j):
    """骨（線分）の色．手首から伸びる中手骨は指の色，指をまたぐ連結は白．"""
    fi, fj = FINGER_OF.get(i), FINGER_OF.get(j)
    if i == 0 and fj is not None:
        return FINGER_COLOR[fj]
    if j == 0 and fi is not None:
        return FINGER_COLOR[fi]
    if fi is not None and fi == fj:
        return FINGER_COLOR[fi]
    return COLOR_PALM


HAND_BONE_COLORS = np.array([bone_color(i, j) for i, j in HAND_CONNECTIONS])
ARM_BONE_COLORS  = np.tile(COLOR_ARM_BONE, (len(ARM_CONNECTIONS), 1))


# ── グリッド ──────────────────────────────────────
def build_grid(pts, lines, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return ls


def build_floor_grid(center_y=0.5, z_near=0.3, z_far=2.5, x_range=1.2, step=0.2):
    pts, lines = [], []
    for x in np.arange(-x_range, x_range + 1e-9, step):
        i = len(pts)
        pts += [[x, center_y, z_near], [x, center_y, z_far]]
        lines.append([i, i + 1])
    for z in np.arange(z_near, z_far + 1e-9, step):
        i = len(pts)
        pts += [[-x_range, center_y, z], [x_range, center_y, z]]
        lines.append([i, i + 1])
    return build_grid(pts, lines, COLOR_GRID_FLOOR)


def build_back_grid(center_z=1.0, x_range=1.2, y_range=1.0, step=0.2):
    pts, lines = [], []
    for x in np.arange(-x_range, x_range + 1e-9, step):
        i = len(pts)
        pts += [[x, -y_range, center_z], [x, y_range, center_z]]
        lines.append([i, i + 1])
    for y in np.arange(-y_range, y_range + 1e-9, step):
        i = len(pts)
        pts += [[-x_range, y, center_z], [x_range, y, center_z]]
        lines.append([i, i + 1])
    return build_grid(pts, lines, COLOR_GRID_BACK)


# ── 関節（球メッシュ） ────────────────────────────
class JointSpheres:
    """関節数ぶんの球をまとめて1つのメッシュとして保持し，頂点移動で更新する．
    点群（四角いスプライト）ではなく球なので，どの角度から見ても丸く見える．"""

    def __init__(self, colors, radius, resolution=SPHERE_RESOLUTION):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius,
                                                         resolution=resolution)
        sv = np.asarray(sphere.vertices, dtype=np.float64)
        st = np.asarray(sphere.triangles, dtype=np.int32)
        self.n_vert = len(sv)
        n_joint = len(colors)

        self.base = np.tile(sv, (n_joint, 1))
        tris = np.vstack([st + k * self.n_vert for k in range(n_joint)])
        cols = np.repeat(np.asarray(colors, dtype=np.float64), self.n_vert, axis=0)

        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices  = o3d.utility.Vector3dVector(self.base.copy())
        self.mesh.triangles = o3d.utility.Vector3iVector(tris)
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(cols)
        self.mesh.compute_vertex_normals()   # 以降は平行移動のみなので法線は不変

    def update(self, pts, valid):
        offs = np.repeat(np.nan_to_num(pts), self.n_vert, axis=0)
        v = self.base + offs
        v[np.repeat(~np.asarray(valid, dtype=bool), self.n_vert)] = 0.0  # 潰して非表示
        self.mesh.vertices = o3d.utility.Vector3dVector(v)


def set_lineset(ls, pts, connections, colors):
    if len(pts) == 0 or len(connections) == 0:
        ls.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        ls.lines  = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
    else:
        ls.points = o3d.utility.Vector3dVector(np.nan_to_num(np.asarray(pts, dtype=np.float64)))
        ls.lines  = o3d.utility.Vector2iVector(np.array(connections, dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))


def estimate_center(hand_pts, hand_valid, arm_pts, arm_valid,
                    n_use=10, default=(0.0, 0.0, 1.0)):
    """最初に関節が推定されたフレーム群（最大 n_use 個）から表示中心を求める．
    有効な関節が1つも無い場合は default を返す．"""
    centers = []
    for f in range(len(hand_valid)):
        pts = []
        if hand_valid[f].any():
            pts.append(hand_pts[f][hand_valid[f]])
        if arm_valid[f].any():
            pts.append(arm_pts[f][arm_valid[f]])
        if pts:
            centers.append(np.vstack(pts).mean(axis=0))
            if len(centers) >= n_use:
                break
    if not centers:
        return np.asarray(default, dtype=np.float64)
    return np.mean(centers, axis=0)


# ── データ読み込み ────────────────────────────────
data       = np.load(FILEPATH)
timestamps = data["timestamps"]   # (N,)   [ms]
hand_pts   = data["hand_pts"]     # (N,21,3)
hand_valid = data["hand_valid"]   # (N,21)
arm_pts    = data["arm_pts"]      # (N,3,3)
arm_valid  = data["arm_valid"]    # (N,3)
n_frames   = len(timestamps)

# ── ビジュアライザ構築 ────────────────────────────
vis = o3d.visualization.Visualizer()
vis.create_window("3D Skeleton (playback)", width=800, height=600)

hand_joints = JointSpheres(HAND_POINT_COLORS, JOINT_RADIUS_HAND)
arm_joints  = JointSpheres(ARM_POINT_COLORS, JOINT_RADIUS_ARM)
hand_ls = o3d.geometry.LineSet()
arm_ls  = o3d.geometry.LineSet()

geometries = [hand_joints.mesh, arm_joints.mesh, hand_ls, arm_ls]
for geo in geometries:
    vis.add_geometry(geo)

vis.add_geometry(build_floor_grid())
vis.add_geometry(build_back_grid())
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))

opt = vis.get_render_option()
opt.background_color = np.array([0.1, 0.1, 0.15])
opt.line_width = 2.0
opt.light_on = True

vis.reset_view_point(True)
ctr = vis.get_view_control()
ctr.set_zoom(4.0)
ctr.set_front([0.0, -0.5, -1.0])
ctr.set_up([0.0, -1.0, 0.0])

# 最初に推定された関節群の中心を注視点にする（原点中心だと手が見切れるため）
view_center = estimate_center(hand_pts, hand_valid, arm_pts, arm_valid)
ctr.set_lookat(view_center)

# ── 再生ループ ────────────────────────────────────
print(f"[RECONSTRUCT] {FILEPATH}  ({n_frames} frames, close window to quit)")
print(f"[VIEW] lookat = {np.round(view_center, 3)}")
print("[COLOR] thumb:red  index:yellow  middle:green  ring:blue  pinky:purple")
t0_wall = time.monotonic()
t0_rec  = timestamps[0]
running = True

for f in range(n_frames):
    # 記録タイムスタンプに合わせて待機（実時間再生）
    target = (timestamps[f] - t0_rec) / 1000.0
    while time.monotonic() - t0_wall < target:
        if not vis.poll_events():
            running = False
            break
        vis.update_renderer()
        time.sleep(0.001)
    if not running:
        break

    # 手指：有効点のみ表示，骨格線は両端が有効なもののみ
    hv = hand_valid[f]
    hand_joints.update(hand_pts[f], hv)
    hand_mask = np.array([bool(hv[i] and hv[j]) for i, j in HAND_CONNECTIONS])
    hand_conn = [c for c, m in zip(HAND_CONNECTIONS, hand_mask) if m]
    set_lineset(hand_ls, hand_pts[f], hand_conn, HAND_BONE_COLORS[hand_mask])

    # 右腕
    av = arm_valid[f]
    arm_joints.update(arm_pts[f], av)
    arm_mask = np.array([bool(av[a] and av[b]) for a, b in ARM_CONNECTIONS])
    arm_conn = [c for c, m in zip(ARM_CONNECTIONS, arm_mask) if m]
    set_lineset(arm_ls, arm_pts[f], arm_conn, ARM_BONE_COLORS[arm_mask])

    for geo in geometries:
        vis.update_geometry(geo)
    if not vis.poll_events():
        break
    vis.update_renderer()

vis.destroy_window()
print("[DONE]")