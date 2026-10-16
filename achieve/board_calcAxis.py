# RealSense のカラー映像からボード座標系 T_cam_board を推定し，軸を重畳表示する．
# 座標系: カメラ = カラー光学座標系（X右, Y下, Z前方）
#         ボード = 内部コーナー#0 が原点（X右, Y下, Zボード奥）
# depth は姿勢推定には使わず，PnP の結果が妥当かの検算にだけ使う．
# q または Esc で終了．s で現在の T_cam_board を表示．

import numpy as np
import cv2
import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 1280, 720, 30
PATTERN = (7, 4)     # 内部コーナー数（X方向, Y方向）
SQ = 0.025           # マス目 [m]

objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float64)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQ

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
    ok, rvec, tvec = cv2.solvePnP(objp, n, K0, D0, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None, None
    return cv2.solvePnPRefineLM(objp, n, K0, D0, rvec, tvec)


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


pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
profile = pipeline.start(cfg)
align = rs.align(rs.stream.color)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics

print("[FRAME] q/Esc: quit   s: print T_cam_board")
last_T = None
try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color:
            continue
        img = np.asanyarray(color.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCornersSB(gray, PATTERN)
        info = "NOT FOUND"
        if found:
            px = corners.reshape(-1, 2).astype(np.float64)
            rvec, tvec = solve(intr, px)
            if rvec is not None and not is_upright(gray, intr, rvec, tvec):
                px = px[::-1].copy()                     # 逆順 = 180度回転に相当
                rvec, tvec = solve(intr, px)

            if rvec is not None:
                P = cam_pts(rvec, tvec, objp)
                rms = np.sqrt(((to_pixels(intr, P) - px) ** 2).sum(1).mean())

                # depth による検算: PnP が予測する Z と実測 Z の差
                dz = []
                if depth:
                    for (u, v), p in zip(px, P):
                        d = depth.get_distance(int(round(u)), int(round(v)))
                        if d > 0.1:
                            dz.append(d - p[2])
                dz_txt = f"  dZ={np.median(dz)*1e3:+.0f}mm" if dz else "  dZ=n/a"

                axes = np.float64([[0, 0, 0], [.05, 0, 0], [0, .05, 0], [0, 0, .05]])
                a = to_pixels(intr, cam_pts(rvec, tvec, axes)).astype(int)
                for k, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):   # X:赤 Y:緑 Z:青
                    cv2.arrowedLine(img, tuple(a[0]), tuple(a[k + 1]), c, 3, tipLength=0.2)

                o = tvec.reshape(3)
                info = (f"O=({o[0]:+.3f},{o[1]:+.3f},{o[2]:+.3f})m  "
                        f"rms={rms:.2f}px{dz_txt}")
                last_T = np.vstack([np.hstack([cv2.Rodrigues(rvec)[0], o.reshape(3, 1)]),
                                    [0, 0, 0, 1]])

        cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("board frame", img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s") and last_T is not None:
            print("T_cam_board =\n", np.array2string(last_T, precision=4, suppress_small=True))
finally:
    pipeline.stop()
    cv2.destroyAllWindows()