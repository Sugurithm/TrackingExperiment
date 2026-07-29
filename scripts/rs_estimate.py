# rs_recorder.py で記録した .db3 を再生しながら MediaPipe で骨格推定し，
# 関節3D座標（手指21点 + 右腕3点）を skeleton3d.npz に保存する．
# 3D表示は行わない．再構成は rs_reconstruct3d.py で行う．

from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions,
)
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode

# ── 入出力 ────────────────────────────────────────
RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"
FILEPATH = RECORDINGS_DIR / "rec_20260729_113731" / "session.db3"
OUTPATH  = FILEPATH.parent / "skeleton3d.npz"

# ── MediaPipe 設定 ────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker_heavy.task"

COLOR_FPS         = 60
FRAME_INTERVAL_MS = int(1000 / COLOR_FPS)

HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE  = 0.5
POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE  = 0.5
NUM_HANDS = 1
NUM_POSES = 1

NUM_HAND_LANDMARKS = 21
RIGHT_ARM_IDS = [12, 14, 16]   # R_SHOULDER, R_ELBOW, R_WRIST


def to_pixel(lm, w, h):
    return (max(0, min(int(lm.x * w), w - 1)),
            max(0, min(int(lm.y * h), h - 1)))


def deproject(intrinsics, px, py, depth_m):
    if depth_m <= 0.01 or not np.isfinite(depth_m):
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth_m)


# ── MediaPipe セットアップ ────────────────────────
hand_landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
    running_mode=running_mode.VisionTaskRunningMode.VIDEO,
    num_hands=NUM_HANDS,
    min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
    min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
))
pose_landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
    running_mode=running_mode.VisionTaskRunningMode.VIDEO,
    num_poses=NUM_POSES,
    min_pose_detection_confidence=POSE_DETECTION_CONFIDENCE,
    min_tracking_confidence=POSE_TRACKING_CONFIDENCE,
))

# ── RealSense 再生セットアップ ────────────────────
cfg = rs.config()
rs.config.enable_device_from_file(cfg, str(FILEPATH), repeat_playback=False)
pipeline = rs.pipeline()
profile = pipeline.start(cfg)
# 実時間再生を切り，フレーム落ちなしで全フレームを処理する
profile.get_device().as_playback().set_real_time(False)
align = rs.align(rs.stream.color)

# ── フレームごとの推定結果を貯めるリスト ──────────
rec_timestamps = []   # (N,)      color フレームのタイムスタンプ [ms]
rec_hand_pts   = []   # (N,21,3)  手指3D座標 [m]
rec_hand_valid = []   # (N,21)    当該フレームで深度が有効だったか
rec_arm_pts    = []   # (N,3,3)   右腕3D座標 [m]（肩・肘・手首）
rec_arm_valid  = []   # (N,3)

# 深度が無効なフレームは直近の有効値で補完する（skelton3d.py と同じ方式）
hand_pts_cache = [None] * NUM_HAND_LANDMARKS
arm_pts_cache  = {}

last_color_ts = -1.0
frame_ts_ms   = 0
intrinsics    = None
latest_depth_frame = None

print(f"[ESTIMATE] {FILEPATH}")
while True:
    ok, frameset = pipeline.try_wait_for_frames(timeout_ms=3000)
    if not ok:
        break   # ファイル終端
    aligned     = align.process(frameset)
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
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

    color_img = np.asanyarray(color_frame.get_data())
    h, w      = color_img.shape[:2]
    rgb_img   = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    mp_img    = Image(image_format=ImageFormat.SRGB, data=rgb_img)

    hand_res = hand_landmarker.detect_for_video(mp_img, frame_ts_ms)
    pose_res = pose_landmarker.detect_for_video(mp_img, frame_ts_ms)
    frame_ts_ms += FRAME_INTERVAL_MS

    # ── 手指21点 ──────────────────────────────
    hand_pts   = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float64)
    hand_valid = np.zeros(NUM_HAND_LANDMARKS, dtype=bool)
    if hand_res.hand_landmarks:
        for idx, lm in enumerate(hand_res.hand_landmarks[0]):
            px, py  = to_pixel(lm, w, h)
            depth_m = latest_depth_frame.get_distance(px, py)
            pt3d    = deproject(intrinsics, px, py, depth_m)
            if pt3d is not None:
                hand_pts_cache[idx] = pt3d
                hand_pts[idx]   = pt3d
                hand_valid[idx] = True
            elif hand_pts_cache[idx] is not None:
                hand_pts[idx] = hand_pts_cache[idx]

    # ── 右腕3点（肩・肘・手首）─────────────────
    arm_pts   = np.zeros((len(RIGHT_ARM_IDS), 3), dtype=np.float64)
    arm_valid = np.zeros(len(RIGHT_ARM_IDS), dtype=bool)
    if pose_res.pose_landmarks:
        pose_lms = pose_res.pose_landmarks[0]
        for i, lm_id in enumerate(RIGHT_ARM_IDS):
            lm      = pose_lms[lm_id]
            px, py  = to_pixel(lm, w, h)
            depth_m = latest_depth_frame.get_distance(px, py)
            pt3d    = deproject(intrinsics, px, py, depth_m)
            if pt3d is not None:
                arm_pts_cache[lm_id] = pt3d
                arm_pts[i]   = pt3d
                arm_valid[i] = True
            elif lm_id in arm_pts_cache:
                arm_pts[i] = arm_pts_cache[lm_id]

    rec_timestamps.append(color_ts)
    rec_hand_pts.append(hand_pts)
    rec_hand_valid.append(hand_valid)
    rec_arm_pts.append(arm_pts)
    rec_arm_valid.append(arm_valid)
    print(f"\r  frame {len(rec_timestamps)}", end="")

pipeline.stop()
hand_landmarker.close()
pose_landmarker.close()

np.savez(
    OUTPATH,
    timestamps=np.array(rec_timestamps),           # (N,)
    hand_pts=np.array(rec_hand_pts),               # (N,21,3)
    hand_valid=np.array(rec_hand_valid),           # (N,21)
    arm_pts=np.array(rec_arm_pts),                 # (N,3,3)
    arm_valid=np.array(rec_arm_valid),             # (N,3)
    arm_ids=np.array(RIGHT_ARM_IDS),               # (3,)
)
print(f"\n[DONE] {len(rec_timestamps)} frames -> {OUTPATH}")