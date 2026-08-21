# RealSense のカラー映像でチェスボードが検出できているかをその場で確認する．

import numpy as np
import cv2
import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 640, 480, 60
PATTERN = (7, 4)   # 内部コーナー数（X方向, Y方向）

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
pipeline.start(cfg)

print("[CHECK] q or Esc to quit")
try:
    while True:
        color = pipeline.wait_for_frames().get_color_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCornersSB(gray, PATTERN) # チェスボード検出
        if found:
            cv2.drawChessboardCorners(img, PATTERN, corners, found) # 検出結果を描画
            cv2.circle(img, tuple(corners[0].ravel().astype(int)), 8, (0, 0, 255), 2)  # 原点コーナー

        cv2.putText(img, "FOUND" if found else "NOT FOUND", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("board check", img)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()