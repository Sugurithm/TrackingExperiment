# player.pyで記録した .db3 を再生する．
# FILE PATH を編集して実行する．None のままなら recordings/ 内の最新セッションを自動選択する．

from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs

RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"
FILEPATH = RECORDINGS_DIR / "rec_20260729_113731" / "session.db3"

DEPTH_VIS_ALPHA = 0.03   # z16 -> 8bit 圧縮係数

cfg = rs.config()
rs.config.enable_device_from_file(cfg, str(FILEPATH), repeat_playback=False)
pipeline = rs.pipeline()
profile = pipeline.start(cfg)
profile.get_device().as_playback().set_real_time(True)

print(f"[PLAY] {FILEPATH}  (q or Esc to quit)")
while True:
    ok, frames = pipeline.try_wait_for_frames(timeout_ms=3000)
    if not ok:
        break   # ファイル終端
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if color:
        cv2.imshow("color", np.asanyarray(color.get_data()))
    if depth:
        vis = cv2.convertScaleAbs(
            np.asanyarray(depth.get_data()),
            alpha=DEPTH_VIS_ALPHA)
        cv2.imshow("depth", cv2.applyColorMap(vis, cv2.COLORMAP_JET))
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

pipeline.stop()
cv2.destroyAllWindows()
print("[DONE]")