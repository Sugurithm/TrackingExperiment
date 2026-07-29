# RealSense D435 RGB-D レコーダ（TCPサーバ版）
#
# Unity(RecordTrigger.cs) から 127.0.0.1:6001 に届く「1行JSON」で録画を制御する．
#   {"cmd":"START","epoch_ms":1753...,"subject":"s01","trial":3}
#   {"cmd":"STOP" ,"epoch_ms":1753...}
#
# 時刻同期の方針:
#   global_time_enabled=1 により frame.get_timestamp() は
#   PCのシステム時刻(epoch ms) を返す．Unity が送る epoch_ms と同じ軸なので，
#   TCP の遅延は同期精度に影響しない．対応づけは meta.json だけで足りる．
#
# 構成:
#   メインスレッド … TCP待受（0.5秒ごとにタイムアウトして Ctrl+C を受け付ける）
#   ワーカスレッド … 録画本体（wait_for_frames を回し続ける）
#
# 最小構成．例外処理・再接続・複数クライアント対応は後から加える．
# 終了は Ctrl+C（録画中なら停止してから pipeline.stop() を通る）．

import datetime
import json
import socket
import threading
import time
from pathlib import Path

import pyrealsense2 as rs

HOST, PORT = "127.0.0.1", 6001
WIDTH, HEIGHT = 640, 480
FPS = 60
SOCKET_TIMEOUT = 0.5

RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"

stop_event = threading.Event()
rec_thread = None
cur_meta = None
cur_dir = None


def epoch_ms():
    return int(time.time() * 1000)


def write_meta():
    (cur_dir / "meta.json").write_text(
        json.dumps(cur_meta, indent=2, ensure_ascii=False), encoding="utf-8")


# ── 録画本体（ワーカスレッド） ────────────────────
def record_loop(session_dir, meta):
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    cfg.enable_record_to_file(str(session_dir / "session.db3"))

    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)

    for sensor in profile.get_device().query_sensors():
        for option, value in (
            (rs.option.global_time_enabled, 1),    # タイムスタンプをPCの絶対時刻に揃える
            (rs.option.auto_exposure_priority, 0)  # 自動露出優先を無効化しFPSを安定化
        ):
            if sensor.supports(option):
                sensor.set_option(option, value)

    meta["rec_start_epoch_ms"] = epoch_ms()
    write_meta()

    # 解析の基準はUnityのSTART時刻ではなく，この first_frame_epoch_ms を使う
    n = 0
    first_ts = last_ts = None
    while not stop_event.is_set():
        frames = pipeline.wait_for_frames()
        last_ts = frames.get_timestamp()
        if first_ts is None:
            first_ts = last_ts
        n += 1

    pipeline.stop()
    meta["rec_stop_epoch_ms"] = epoch_ms()
    meta["frames"] = n
    meta["first_frame_epoch_ms"] = first_ts
    meta["last_frame_epoch_ms"] = last_ts
    write_meta()
    print(f"[REC] stop   {n} frames -> {session_dir}")


def stop_recording():
    """録画中なら停止して meta.json の書き出し完了まで待つ．"""
    global rec_thread
    if rec_thread is not None and rec_thread.is_alive():
        stop_event.set()
        rec_thread.join()
    rec_thread = None


# ── コマンド処理 ──────────────────────────────────
def send_ack(conn, obj):
    conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def on_start(msg, conn):
    global rec_thread, cur_meta, cur_dir

    if rec_thread is not None and rec_thread.is_alive():
        print("[WARN] already recording; START ignored")
        send_ack(conn, {"ack": "START", "ok": False, "reason": "already_recording"})
        return

    name = "rec_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cur_dir = RECORDINGS_DIR / name
    cur_dir.mkdir(parents=True, exist_ok=True)
    cur_meta = {
        "session": name,
        "subject": msg.get("subject", ""),
        "trial": msg.get("trial", -1),
        "unity_start_epoch_ms": msg.get("epoch_ms"),
        "width": WIDTH, "height": HEIGHT, "fps": FPS,
    }

    stop_event.clear()
    rec_thread = threading.Thread(target=record_loop, args=(cur_dir, cur_meta), daemon=True)
    rec_thread.start()
    print(f"[REC] start  {cur_dir}")
    send_ack(conn, {"ack": "START", "ok": True, "session": name})


def on_stop(msg, conn):
    if rec_thread is None or not rec_thread.is_alive():
        print("[WARN] not recording; STOP ignored")
        send_ack(conn, {"ack": "STOP", "ok": False, "reason": "not_recording"})
        return

    cur_meta["unity_stop_epoch_ms"] = msg.get("epoch_ms")
    stop_recording()
    send_ack(conn, {"ack": "STOP", "ok": True, "frames": cur_meta["frames"]})


def handle_line(line, conn):
    msg = json.loads(line)
    cmd = msg.get("cmd")
    print(f"[RECV] {line}")
    if cmd == "START":
        on_start(msg, conn)
    elif cmd == "STOP":
        on_stop(msg, conn)


# ── クライアント1接続ぶんの受信ループ ─────────────
def serve_client(conn):
    """タイムアウト付き recv で1行ずつ処理する．
    makefile のイテレーションだと Windows で Ctrl+C を受け付けないため自前で分割する．"""
    conn.settimeout(SOCKET_TIMEOUT)
    buf = ""
    while True:
        try:
            chunk = conn.recv(4096)
        except socket.timeout:
            continue
        if not chunk:
            break
        buf += chunk.decode("utf-8")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if line:
                handle_line(line, conn)


# ── TCP待受（メインスレッド） ─────────────────────
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((HOST, PORT))
srv.listen(1)
srv.settimeout(SOCKET_TIMEOUT)   # accept をブロックさせない（Ctrl+C 用）
print(f"[SERVER] listening {HOST}:{PORT}  (Ctrl+C to quit)")

try:
    while True:
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            continue

        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[SERVER] connected {addr}")
        with conn:
            serve_client(conn)

        # 録画中に Unity が落ちた場合は録画を畳んでおく
        if rec_thread is not None and rec_thread.is_alive():
            print("[WARN] client gone while recording; stopping")
            stop_recording()
        print("[SERVER] disconnected")

except KeyboardInterrupt:
    print("\n[SERVER] interrupted")

finally:
    stop_recording()     # 録画中なら pipeline.stop() まで通してから終了する
    srv.close()
    print("[SERVER] closed")