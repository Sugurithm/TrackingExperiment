# RealSense D435 RGB-D レコーダ（TCPなし）
#
# Enterキーで録画START，もう一度Enterで録画STOP．
# Ctrl+Cで抜けても録画中なら必ず pipeline.stop() を通す．
# 保存されるのは RGB + Depth のみを含む .db3 ファイル（session.db3）．
#
# 構成:
#   メインスレッド … 標準入力待ち（Enterのたびにstart/stopを切り替える）
#   ワーカスレッド … 録画本体（wait_for_framesを回し続ける）

import datetime
import threading
from pathlib import Path

import pyrealsense2 as rs

WIDTH, HEIGHT = 640, 480
FPS = 60

RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"

stop_event = threading.Event()
worker = None


def record_loop(path):
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    cfg.enable_record_to_file(str(path))

    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)  # PCとカメラの通信確立

    # データ品質に影響する2オプションのみ設定
    for sensor in profile.get_device().query_sensors():
        for option, value in (
            (rs.option.global_time_enabled, 1),    # センサーのタイムスタンプをPCの時刻に同期
            (rs.option.auto_exposure_priority, 0)  # 自動露出優先を無効化してフレームレートを安定化
        ):
            if sensor.supports(option):
                sensor.set_option(option, value)

    n = 0
    try:
        while not stop_event.is_set():
            pipeline.wait_for_frames()
            n += 1
    finally:
        pipeline.stop()
        print(f"[REC] stop   {n} frames -> {path}")


def start_recording():
    global worker
    if worker is not None and worker.is_alive():
        print("[WARN] already recording")
        return

    session_dir = RECORDINGS_DIR / ("rec_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "session.db3"

    stop_event.clear()
    worker = threading.Thread(target=record_loop, args=(path,), daemon=True)
    worker.start()
    print(f"[REC] start  {path}")


def stop_recording():
    global worker
    if worker is None or not worker.is_alive():
        print("[WARN] not recording")
        return
    stop_event.set()
    worker.join()  # pipeline.stop() が完了するまで待つ
    worker = None


def main():
    print("Enterキーで START / STOP を切り替えます（Ctrl+Cで終了）")
    try:
        while True:
            input()
            if worker is not None and worker.is_alive():
                stop_recording()
            else:
                start_recording()
    except KeyboardInterrupt:
        print("\n[MAIN] interrupted")
    finally:
        stop_recording()  # 録画中なら安全に停止してから終了
        print("[DONE]")


if __name__ == "__main__":
    main()