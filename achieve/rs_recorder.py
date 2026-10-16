# RealSense D435 RGB-D レコーダ (クローズ保護なし)
# 最小限のパイプラインのみ．時間経過で終了
# そのため，途中で Ctrl+C すると pipeline.stop() を通らず不完全なまま残る．

import datetime
import time
from pathlib import Path

import pyrealsense2 as rs

WIDTH, HEIGHT = 640, 480
FPS = 60
DURATION = 5.0    # 記録時間 [s]

RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"
session_dir = RECORDINGS_DIR / ("rec_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
session_dir.mkdir(parents=True, exist_ok=True)
path = session_dir / "session.db3"

cfg = rs.config()
cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
cfg.enable_record_to_file(str(path))

# デバイス起動とセンサー最適化
pipeline = rs.pipeline()
profile = pipeline.start(cfg) # PCとカメラの通信確立

# データ品質に影響する2オプションのみ設定
for sensor in profile.get_device().query_sensors():
    for option, value in (
        (rs.option.global_time_enabled, 1), # センサーのタイムスタンプをPCの時刻に同期
        (rs.option.auto_exposure_priority, 0) # 自動露出優先を無効化してフレームレートを安定化
        ):
        if sensor.supports(option):
            sensor.set_option(option, value)

print(f"[REC] {path}  ({DURATION:.0f}s)")
t0 = time.monotonic()
while time.monotonic() - t0 < DURATION:
    pipeline.wait_for_frames()

pipeline.stop()
print("[DONE]")