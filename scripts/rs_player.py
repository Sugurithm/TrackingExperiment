# rs_recorder.py で記録した .db3 を再生する．
# FILEPATH を編集して実行する．None のままなら recordings/ 内の最新セッションを自動選択する．

from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs

RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"
FILEPATH = None   # None なら recordings/ 内の最新 session.db3 を自動選択
DEPTH_VIS_ALPHA = 0.03   # z16 -> 8bit 圧縮係数


def resolve_filepath(filepath, name="session.db3"):
    """FILEPATH が None なら recordings/*/name のうち最も新しいものを返す．"""
    if filepath is not None:
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"指定されたファイルがありません: {p}")
        return p
    cands = list(RECORDINGS_DIR.glob(f"*/{name}"))
    if not cands:
        raise FileNotFoundError(f"{RECORDINGS_DIR}/*/{name} が見つかりません")
    return max(cands, key=lambda p: p.stat().st_mtime)


FILEPATH = resolve_filepath(FILEPATH)

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