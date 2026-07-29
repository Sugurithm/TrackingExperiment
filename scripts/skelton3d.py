"""
realsense_skeleton_3d.py
────────────────────────────────────────────────────────────────────
RealSense + MediaPipe のランドマーク座標を3D空間に再構成し，
Open3D の別ウィンドウにリアルタイム表示するスクリプト．

【追加機能】
  - RealSense 内部パラメータを使って 2D ピクセル → 3D 点に逆投影
  - Open3D Visualizer で骨格を非ブロッキング更新
  - 's'キーで2D(RGB)と3D(Open3D)の現在の画面を画像として保存
────────────────────────────────────────────────────────────────────
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d
from pathlib import Path
import datetime  # ファイル名用に日時を取得するため追加
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, HandLandmarksConnections,
    PoseLandmarker, PoseLandmarkerOptions,
    drawing_utils, drawing_styles,
)
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode


# ── カメラ設定 ────────────────────────────────────
STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
DEPTH_FPS     = 90
COLOR_FPS     = 60
FRAME_INTERVAL_MS = int(1000 / COLOR_FPS)

# ── 描画設定 ──────────────────────────────────────
ARM_POINT_RADIUS   = 8
ARM_POINT_BORDER   = 2
ARM_BONE_THICKNESS = 2
ARM_LABEL_OFFSET_X = 10
FONT               = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SMALL   = 0.5
FONT_SCALE_NORMAL  = 0.6
FONT_THICKNESS     = 2

# ── 色定義 ────────────────────────────────────────
COLOR_RIGHT_ARM  = (0, 200, 255)
COLOR_HAND_LABEL = (0, 255,   0)
COLOR_WHITE      = (255, 255, 255)

# ── MediaPipe 信頼度 ──────────────────────────────
HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE  = 0.5
POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE  = 0.5
NUM_HANDS = 1
NUM_POSES = 1

QUIT_KEY      = ord('q')
SAVE_KEY      = ord('s')  # 保存キーを追加
WAITKEY_DELAY = 1

# ── ランドマーク定義 ──────────────────────────────
HAND_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP",    "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP",   "INDEX_TIP",
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP",  "MIDDLE_TIP",
    "RING_MCP",  "RING_PIP",  "RING_DIP",    "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP",   "PINKY_TIP",
]
HAND_WRIST_ID = 0

# 手指の骨格線（MediaPipe 接続定義と同じ構造を手動展開）
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

RIGHT_ARM_IDS   = {12: "R_SHOULDER", 14: "R_ELBOW", 16: "R_WRIST"}
RIGHT_ARM_BONES = [(12, 14), (14, 16)]

# ── モデルパス ────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker_heavy.task"


# ══════════════════════════════════════════════════
#  3D 骨格ビジュアライザ
# ══════════════════════════════════════════════════
class SkeletonVisualizer3D:
    """
    Open3D を使ったリアルタイム 3D 骨格表示クラス．
    """
    COLOR_HAND_POINT  = [0.0, 1.0, 0.4]
    COLOR_HAND_BONE   = [0.0, 0.8, 0.3]
    COLOR_ARM_POINT   = [1.0, 0.5, 0.0]
    COLOR_ARM_BONE    = [0.9, 0.4, 0.0]

    # グリッドの色（暗めの灰色で背景に溶け込まないよう調整）
    COLOR_GRID_FLOOR  = [0.3, 0.3, 0.35]   # 床グリッド
    COLOR_GRID_BACK   = [0.25, 0.25, 0.30]  # 背景グリッド

    def __init__(self, window_name: str = "3D Skeleton", width: int = 800, height: int = 600):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name, width=width, height=height)

        self.hand_pcd  = o3d.geometry.PointCloud()
        self.hand_ls   = o3d.geometry.LineSet()
        self.arm_pcd   = o3d.geometry.PointCloud()
        self.arm_ls    = o3d.geometry.LineSet()

        for geo in [self.hand_pcd, self.hand_ls, self.arm_pcd, self.arm_ls]:
            self.vis.add_geometry(geo)

        # ── グリッドと座標軸を追加 ───────────────────
        floor_grid = self._build_floor_grid(
            center_y=0.5,        # カメラから約0.5m下（床面の目安）
            z_near=0.3, z_far=2.5,
            x_range=1.2,
            step=0.2,
        )
        back_grid = self._build_back_grid(
            center_z=1.0,        # カメラから約1m前方（被験者の正面目安）
            x_range=1.2,
            y_range=1.0,
            step=0.2,
        )
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.15, origin=[0.0, 0.0, 0.0]
        )

        for geo in [floor_grid, back_grid, axes]:
            self.vis.add_geometry(geo)

        self._setup_render()
        self._initialized = False

    # ── グリッド生成 ────────────────────────────────

    def _build_floor_grid(
        self,
        center_y: float = 0.5,
        z_near: float = 0.3,
        z_far: float = 2.5,
        x_range: float = 1.2,
        step: float = 0.2,
    ) -> o3d.geometry.LineSet:
        """XZ平面（床面）グリッドを生成する．"""
        pts, lines = [], []

        xs = np.arange(-x_range, x_range + 1e-9, step)
        zs = np.arange(z_near,   z_far   + 1e-9, step)

        # Z方向の線（X位置を固定）
        for x in xs:
            i = len(pts)
            pts.append([x, center_y, z_near])
            pts.append([x, center_y, z_far])
            lines.append([i, i + 1])

        # X方向の線（Z位置を固定）
        for z in zs:
            i = len(pts)
            pts.append([-x_range, center_y, z])
            pts.append([ x_range, center_y, z])
            lines.append([i, i + 1])

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
        ls.lines  = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(self.COLOR_GRID_FLOOR, (len(lines), 1))
        )
        return ls

    def _build_back_grid(
        self,
        center_z: float = 1.0,
        x_range: float = 1.2,
        y_range: float = 1.0,
        step: float = 0.2,
    ) -> o3d.geometry.LineSet:
        """XY平面（背景面）グリッドを生成する．"""
        pts, lines = [], []

        xs = np.arange(-x_range, x_range + 1e-9, step)
        ys = np.arange(-y_range, y_range + 1e-9, step)   # カメラ座標系はY下向き

        # Y方向の線（X位置を固定）
        for x in xs:
            i = len(pts)
            pts.append([x, -y_range, center_z])
            pts.append([x,  y_range, center_z])
            lines.append([i, i + 1])

        # X方向の線（Y位置を固定）
        for y in ys:
            i = len(pts)
            pts.append([-x_range, y, center_z])
            pts.append([ x_range, y, center_z])
            lines.append([i, i + 1])

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
        ls.lines  = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(self.COLOR_GRID_BACK, (len(lines), 1))
        )
        return ls

    def _setup_render(self):
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.15])
        opt.point_size = 7.0
        opt.line_width = 2.0

    def _set_initial_viewpoint(self):
        """初回フレームのみ視点を設定"""
        self.vis.reset_view_point(True)
        
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.6)
        ctr.set_front([0.0, -0.5, -1.0])
        ctr.set_up([0.0, -1.0, 0.0])
        self._initialized = True

    def _update_pointcloud(self, pcd: o3d.geometry.PointCloud, pts: list, color: list):
        if not pts:
            pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        else:
            pcd.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)).astype(np.float64))
        self.vis.update_geometry(pcd)

    def _update_lineset(self, ls: o3d.geometry.LineSet, pts: list, connections: list, color: list):
        if not pts or not connections:
            ls.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            ls.lines  = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
            ls.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        else:
            ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
            ls.lines  = o3d.utility.Vector2iVector(np.array(connections, dtype=np.int32))
            ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(connections), 1)).astype(np.float64))
        self.vis.update_geometry(ls)

    def update(self, hand_pts_3d: list, hand_valid: list, arm_pts_3d: list, arm_valid: list, arm_connections: list):
        valid_hand_pts = [p for p, v in zip(hand_pts_3d, hand_valid) if v]
        valid_arm_pts  = [p for p, v in zip(arm_pts_3d,  arm_valid)  if v]

        self._update_pointcloud(self.hand_pcd, valid_hand_pts, self.COLOR_HAND_POINT)

        if len(hand_pts_3d) == 21:
            valid_hand_conn = [
                (i, j) for i, j in HAND_CONNECTIONS
                if hand_valid[i] and hand_valid[j]
            ]
            self._update_lineset(self.hand_ls, hand_pts_3d, valid_hand_conn, self.COLOR_HAND_BONE)
        else:
            self._update_lineset(self.hand_ls, [], [], self.COLOR_HAND_BONE)

        self._update_pointcloud(self.arm_pcd, valid_arm_pts, self.COLOR_ARM_POINT)
        self._update_lineset(self.arm_ls, arm_pts_3d, arm_connections, self.COLOR_ARM_BONE)

        if not self._initialized and (len(valid_hand_pts) > 0 or len(valid_arm_pts) > 0):
            self._set_initial_viewpoint()

        self.vis.poll_events()
        self.vis.update_renderer()

    def capture_screen(self, filename: str):
        """現在の3D画面を画像として保存する"""
        self.vis.capture_screen_image(filename, do_render=True)

    def close(self):
        self.vis.destroy_window()


# ══════════════════════════════════════════════════
#  2D ピクセル → 3D 点 逆投影
# ══════════════════════════════════════════════════
def deproject(intrinsics, px: int, py: int, depth_m: float) -> list:
    if depth_m <= 0.01 or not np.isfinite(depth_m):
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth_m)


# ══════════════════════════════════════════════════
#  MediaPipe セットアップ
# ══════════════════════════════════════════════════
def build_hand_landmarker():
    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=running_mode.VisionTaskRunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
    )
    return HandLandmarker.create_from_options(opts)

def build_pose_landmarker():
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=running_mode.VisionTaskRunningMode.VIDEO,
        num_poses=NUM_POSES,
        min_pose_detection_confidence=POSE_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_TRACKING_CONFIDENCE,
    )
    return PoseLandmarker.create_from_options(opts)


# ══════════════════════════════════════════════════
#  RealSense セットアップ
# ══════════════════════════════════════════════════
def build_pipeline():
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT,
                         rs.format.z16,  DEPTH_FPS)
    config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT,
                         rs.format.bgr8, COLOR_FPS)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align


# ══════════════════════════════════════════════════
#  ヘルパー
# ══════════════════════════════════════════════════
def to_pixel(lm, w, h):
    return (max(0, min(int(lm.x * w), w - 1)),
            max(0, min(int(lm.y * h), h - 1)))

def draw_arm_point(img, px, py, depth_m, name):
    cv2.circle(img, (px, py), ARM_POINT_RADIUS, COLOR_RIGHT_ARM, -1)
    cv2.circle(img, (px, py), ARM_POINT_RADIUS, COLOR_WHITE, ARM_POINT_BORDER)
    cv2.putText(img, f"{name} Z:{depth_m:.2f}m",
                (px + ARM_LABEL_OFFSET_X, py),
                FONT, FONT_SCALE_SMALL, COLOR_RIGHT_ARM, FONT_THICKNESS)


# ══════════════════════════════════════════════════
#  メインループ
# ══════════════════════════════════════════════════
def main():
    hand_landmarker = build_hand_landmarker()
    pose_landmarker = build_pose_landmarker()
    pipeline, align = build_pipeline()
    vis3d           = SkeletonVisualizer3D("3D Skeleton", width=800, height=600)

    latest_depth_frame = None
    last_color_ts      = -1.0
    frame_ts_ms        = 0
    intrinsics         = None

    hand_pts_cache: list = [None] * 21
    arm_pts_cache:  dict = {}

    try:
        while True:
            frameset = pipeline.wait_for_frames(timeout_ms=5000)
            aligned  = align.process(frameset)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if depth_frame:
                latest_depth_frame = depth_frame

            if not color_frame:
                continue
            color_ts = color_frame.get_timestamp()
            if color_ts == last_color_ts:
                continue
            last_color_ts = color_ts
            if latest_depth_frame is None:
                continue

            if intrinsics is None:
                intrinsics = (color_frame.profile
                              .as_video_stream_profile()
                              .intrinsics)

            color_img = np.asanyarray(color_frame.get_data())
            h, w      = color_img.shape[:2]

            rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            mp_img  = Image(image_format=ImageFormat.SRGB, data=rgb_img)

            hand_res = hand_landmarker.detect_for_video(mp_img, frame_ts_ms)
            pose_res = pose_landmarker.detect_for_video(mp_img, frame_ts_ms)
            frame_ts_ms += FRAME_INTERVAL_MS

            # ══════════════════════════════════════
            #  手指処理
            # ══════════════════════════════════════
            hand_pts_3d: list = []
            hand_valid:  list = []

            if hand_res.hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hand_res.hand_landmarks):
                    handedness = hand_res.handedness[hand_idx][0].category_name

                    drawing_utils.draw_landmarks(
                        color_img, hand_landmarks,
                        HandLandmarksConnections.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style(),
                    )

                    for idx, lm in enumerate(hand_landmarks):
                        px, py  = to_pixel(lm, w, h)
                        depth_m = latest_depth_frame.get_distance(px, py)
                        pt3d    = deproject(intrinsics, px, py, depth_m)

                        if pt3d is not None:
                            hand_pts_cache[idx] = pt3d
                            hand_pts_3d.append(pt3d)
                            hand_valid.append(True)
                        elif hand_pts_cache[idx] is not None:
                            hand_pts_3d.append(hand_pts_cache[idx])
                            hand_valid.append(False)
                        else:
                            hand_pts_3d.append([0.0, 0.0, 0.0])
                            hand_valid.append(False)

                        if idx == HAND_WRIST_ID:
                            cv2.putText(
                                color_img,
                                f"{handedness} Hand Z:{depth_m:.2f}m",
                                (px - 30, py - 15),
                                FONT, FONT_SCALE_NORMAL,
                                COLOR_HAND_LABEL, FONT_THICKNESS,
                            )

            # ══════════════════════════════════════
            #  上肢処理（肩・肘・手首）
            # ══════════════════════════════════════
            arm_pts_3d:     list = []
            arm_valid:      list = []
            arm_connections:list = []
            arm_id_to_idx        = {}

            if pose_res.pose_landmarks:
                pose_lms = pose_res.pose_landmarks[0]

                for lm_id, name in RIGHT_ARM_IDS.items():
                    lm      = pose_lms[lm_id]
                    px, py  = to_pixel(lm, w, h)
                    depth_m = latest_depth_frame.get_distance(px, py)
                    pt3d    = deproject(intrinsics, px, py, depth_m)

                    draw_arm_point(color_img, px, py, depth_m, name)

                    arm_id_to_idx[lm_id] = len(arm_pts_3d)

                    if pt3d is not None:
                        arm_pts_cache[lm_id] = pt3d
                        arm_pts_3d.append(pt3d)
                        arm_valid.append(True)
                    elif lm_id in arm_pts_cache:
                        arm_pts_3d.append(arm_pts_cache[lm_id])
                        arm_valid.append(False)
                    else:
                        arm_pts_3d.append([0.0, 0.0, 0.0])
                        arm_valid.append(False)

                for a, b in RIGHT_ARM_BONES:
                    if a in arm_id_to_idx and b in arm_id_to_idx:
                        cv2.line(color_img,
                                 to_pixel(pose_lms[a], w, h),
                                 to_pixel(pose_lms[b], w, h),
                                 COLOR_RIGHT_ARM, ARM_BONE_THICKNESS)
                        ia, ib = arm_id_to_idx[a], arm_id_to_idx[b]
                        if arm_valid[ia] and arm_valid[ib]:
                            arm_connections.append([ia, ib])

            # ── 3D ビジュアライザ更新 ─────────────
            vis3d.update(hand_pts_3d, hand_valid, arm_pts_3d, arm_valid, arm_connections)

            # ── 2D ウィンドウ表示 (深度カメラを非表示化) ───
            cv2.imshow("RealSense + MediaPipe (2D)", color_img)

            # ── キー入力処理 ──────────────────────
            key = cv2.waitKey(WAITKEY_DELAY) & 0xFF
            if key == QUIT_KEY:
                break
            elif key == SAVE_KEY:
                # sキーが押されたら現在時刻で画像を保存
                now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                rgb_filename = f"rgb_{now_str}.png"
                o3d_filename = f"3d_{now_str}.png"
                
                cv2.imwrite(rgb_filename, color_img)
                vis3d.capture_screen(o3d_filename)
                
                print(f"[Saved] {rgb_filename} & {o3d_filename}")

    finally:
        hand_landmarker.close()
        pose_landmarker.close()
        pipeline.stop()
        cv2.destroyAllWindows()
        vis3d.close()

if __name__ == "__main__":
    main()