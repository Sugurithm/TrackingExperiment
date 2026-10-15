"""
rs_preview.py
────────────────────────────────────────────────────────────────────
RealSense の設置位置・画角を確認するためだけのプレビュー．
保存は一切しない．PREVIEW_SEC 秒経過，または q / Esc で終了する．

  - color と depth を横並びで表示
  - 画面中央に十字線と中央点の距離 [m] を表示
  - 残り時間を表示

【エラー処理】
  - 起動時にハードウェアリセットを行い，前回異常終了した状態を解消する
    （USB を抜き差ししなくても再実行できるようにするため）
  - フレームが一定回数連続で届かなければ，パイプライン停止
    → ハードウェアリセット → 再起動 を自動で行う
────────────────────────────────────────────────────────────────────
"""

import time

import numpy as np
import cv2
import pyrealsense2 as rs

# ── 設定 ──────────────────────────────────────────
PREVIEW_SEC     = 60      # 表示時間 [s]．None なら無制限（q / Esc で終了）
STREAM_WIDTH    = 640
STREAM_HEIGHT   = 480
FPS             = 30      # depth / color で揃える
DEPTH_VIS_ALPHA = 0.03    # z16 -> 8bit 圧縮係数
ALIGN_TO_COLOR  = True    # depth を color 画角に合わせて表示する
WINDOW_NAME     = "RealSense Preview (q or Esc to quit)"

# ── エラー処理設定 ────────────────────────────────
RESET_ON_START      = True   # 起動時にハードウェアリセットする
FRAME_TIMEOUT_MS    = 1000   # 1 回の待ち時間
MAX_TIMEOUTS        = 3      # 連続タイムアウトがこの回数に達したら再起動
MAX_RESTARTS        = 3      # 再起動の上限回数（超えたら諦めて終了）
RECONNECT_WAIT_SEC  = 10     # リセット後，デバイス再認識を待つ最大時間


# ══════════════════════════════════════════════════
#  デバイス操作
# ══════════════════════════════════════════════════
def wait_for_device(timeout_s: float = RECONNECT_WAIT_SEC) -> bool:
    """デバイスが認識されるまで待つ．"""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if len(rs.context().query_devices()) > 0:
            return True
        time.sleep(0.5)
    return False


def hardware_reset():
    """接続中の全 RealSense をハードウェアリセットし，再認識を待つ．"""
    devices = rs.context().query_devices()
    if len(devices) == 0:
        print("[RESET] デバイスが見つかりません")
        return False
    for dev in devices:
        print(f"[RESET] {dev.get_info(rs.camera_info.name)} をリセットします")
        dev.hardware_reset()
    time.sleep(2.0)   # 一度切断されるのを待つ
    ok = wait_for_device()
    print("[RESET] 再認識しました" if ok else "[RESET] 再認識できませんでした")
    time.sleep(1.0)   # 認識直後は start に失敗しやすいので少し待つ
    return ok


def start_pipeline():
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT, rs.format.z16,  FPS)
    cfg.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(cfg)

    dev = profile.get_device()
    print(f"[DEVICE] {dev.get_info(rs.camera_info.name)}  "
          f"USB: {dev.get_info(rs.camera_info.usb_type_descriptor)}")
    return pipeline


def safe_stop(pipeline):
    """stop 自体が失敗しても落ちないようにする．"""
    if pipeline is None:
        return
    try:
        pipeline.stop()
    except RuntimeError as e:
        print(f"[WARN] pipeline.stop() 失敗: {e}")


def restart_pipeline(pipeline):
    """停止 → ハードウェアリセット → 再起動．失敗したら None を返す．"""
    safe_stop(pipeline)
    if not hardware_reset():
        return None
    try:
        return start_pipeline()
    except RuntimeError as e:
        print(f"[ERROR] 再起動に失敗: {e}")
        return None


# ══════════════════════════════════════════════════
#  描画
# ══════════════════════════════════════════════════
def draw_overlay(img, center_dist_m, remaining_s):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
    cv2.line(img, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)
    cv2.putText(img, f"center: {center_dist_m:.2f} m", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if remaining_s is not None:
        cv2.putText(img, f"remain: {remaining_s:4.1f} s", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


# ══════════════════════════════════════════════════
#  メイン
# ══════════════════════════════════════════════════
def main():
    if RESET_ON_START:
        hardware_reset()

    pipeline = None
    try:
        pipeline = start_pipeline()
    except RuntimeError as e:
        print(f"[ERROR] 起動に失敗: {e}")
        return

    align = rs.align(rs.stream.color) if ALIGN_TO_COLOR else None
    t_start  = time.time()
    timeouts = 0
    restarts = 0
    print(f"[PREVIEW] {PREVIEW_SEC if PREVIEW_SEC else '∞'} s  (q or Esc to quit)")

    try:
        while True:
            elapsed = time.time() - t_start
            if PREVIEW_SEC is not None and elapsed >= PREVIEW_SEC:
                break
            remaining = None if PREVIEW_SEC is None else PREVIEW_SEC - elapsed

            # ── フレーム取得（タイムアウト時は再起動を試みる） ──
            try:
                ok, frames = pipeline.try_wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
            except RuntimeError as e:
                print(f"[WARN] フレーム取得エラー: {e}")
                ok = False

            if not ok:
                timeouts += 1
                print(f"[WARN] フレームが届きません ({timeouts}/{MAX_TIMEOUTS})")
                # 待っている間もウィンドウを固まらせない
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if timeouts < MAX_TIMEOUTS:
                    continue

                if restarts >= MAX_RESTARTS:
                    print("[ERROR] 再起動の上限に達しました．USB 接続を確認してください")
                    break
                restarts += 1
                print(f"[RESTART] パイプラインを再起動します ({restarts}/{MAX_RESTARTS})")
                pipeline = restart_pipeline(pipeline)
                if pipeline is None:
                    print("[ERROR] 復帰できませんでした．USB を抜き差ししてください")
                    break
                timeouts = 0
                continue

            timeouts = 0

            if align is not None:
                frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            color_img = np.asanyarray(color.get_data()).copy()
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(np.asanyarray(depth.get_data()), alpha=DEPTH_VIS_ALPHA),
                cv2.COLORMAP_JET)

            center_dist = depth.get_distance(depth.get_width() // 2, depth.get_height() // 2)
            draw_overlay(color_img, center_dist, remaining)
            draw_overlay(depth_vis, center_dist, remaining)

            cv2.imshow(WINDOW_NAME, np.hstack([color_img, depth_vis]))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        safe_stop(pipeline)
        cv2.destroyAllWindows()
        print("[DONE]")


if __name__ == "__main__":
    main()