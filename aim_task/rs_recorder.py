"""
rs_recorder.py
────────────────────────────────────────────────────────────────────
RealSense の記録を別プロセスで行う（run_experiment.py から使う）．

課題（pygame）は画面を 240 fps で回し続けるので，同じプロセスで RealSense のフレームを
受け取ると，どちらかが詰まる．そこで記録は別プロセスに任せ，課題からは
「この .db3 に記録を始めて」「止めて」をキューで送るだけにする．

カメラの設定と記録の中身は experiment/ と同じ:
  - experiment/config.py の STREAM_WIDTH / STREAM_HEIGHT / FPS / COLOR_EXPOSURE（較正と同じ設定）
  - experiment/lib/rs_device.py の _start_record_pipeline()
    （global_time_enabled = 1 → フレームの時刻が PC の epoch ms になる．課題の epoch と同じ軸）
  - 起動時のハードウェアリセット，フレームが MAX_TIMEOUTS 回続けて届かなければ失敗 → リセット
  - 1 試行 = 1 回の接続 → 記録 → 切断（.db3 が途中で分かれないように，記録途中での自動再起動はしない）

課題側の使い方:
    rec = RSRecorder()            # 子プロセスを起動（ハードウェアリセットを含む）
    rec.wait_ready()
    rec.request_start(db3_path)   # 記録開始を頼む
    rec.poll()                    # 毎フレーム呼ぶ．rec.state が "starting" → "recording" → "done"
    rec.request_stop()            # 止めるよう頼む → rec.state == "done" になったら rec.result に meta 用の dict
    rec.close()

子プロセスから届くイベント:
    ("ready", info)        起動完了（info = カメラの設定）
    ("started", info)      最初のフレームが届いた（info["first_frame_epoch_ms"]）
    ("stopped", res)       記録が終わった（res = rs_device.record() と同じ形の dict．status で成否）
    ("error", text)        子プロセスでの予期しないエラー
────────────────────────────────────────────────────────────────────
"""

import multiprocessing as mp
import queue
import sys
import time
import traceback
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "experiment"


# ══════════════════════════════════════════════════
#  子プロセス
# ══════════════════════════════════════════════════
def _parent_alive():
    p = mp.parent_process()
    return p is None or p.is_alive()


def _record_until_stop(db3_path, cmd_q, ev_q, rs_device, config, rs):
    """接続 → stop が届くまで記録 → 切断．rs_device.record() と同じ形の dict を返す．
    返り値の 2 つめは記録中に quit が届いたか．"""
    res = dict(status="ok", reason="", duration_s=None, frames=0,
               width=config.STREAM_WIDTH, height=config.STREAM_HEIGHT, fps=config.FPS,
               color_exposure=getattr(config, "COLOR_EXPOSURE", None),
               rec_start_epoch_ms=None, rec_stop_epoch_ms=None,
               first_frame_epoch_ms=None, last_frame_epoch_ms=None,
               color_intrinsics=None)
    pipeline, quit_req = None, False
    try:
        try:
            pipeline, profile = rs_device._start_record_pipeline(db3_path)
        except RuntimeError as e:
            res.update(status="failed", reason=f"起動に失敗: {e}")
            return res, quit_req

        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        res["color_intrinsics"] = rs_device.intrinsics_to_dict(intr)
        res["rec_start_epoch_ms"] = int(time.time() * 1000)
        t0, timeouts = time.monotonic(), 0
        while True:
            try:                                   # 止める合図（記録中は待たずに覗くだけ）
                c = cmd_q.get_nowait()
                if c.get("cmd") in ("stop", "quit"):
                    quit_req = c["cmd"] == "quit"
                    break
            except queue.Empty:
                pass
            if not _parent_alive():                # 課題が落ちたら .db3 を閉じて終わる
                res.update(status="interrupted", reason="課題のプロセスが終了した")
                quit_req = True
                break
            try:
                ok, frames = pipeline.try_wait_for_frames(timeout_ms=config.FRAME_TIMEOUT_MS)
            except RuntimeError as e:
                print(f"[RS][WARN] フレーム取得エラー: {e}", flush=True)
                ok = False
            if not ok:
                timeouts += 1
                print(f"[RS][WARN] フレームが届きません ({timeouts}/{config.MAX_TIMEOUTS})", flush=True)
                if timeouts >= config.MAX_TIMEOUTS:
                    res.update(status="failed", reason="フレームが届かなくなった")
                    break
                continue
            timeouts = 0
            ts = frames.get_timestamp()
            if res["first_frame_epoch_ms"] is None:
                res["first_frame_epoch_ms"] = ts
                ev_q.put(("started", {"first_frame_epoch_ms": ts}))
            res["last_frame_epoch_ms"] = ts
            res["frames"] += 1
        res["duration_s"] = round(time.monotonic() - t0, 3)
    finally:
        rs_device.safe_stop(pipeline)
        res["rec_stop_epoch_ms"] = int(time.time() * 1000)
    return res, quit_req


def _child_main(cmd_q, ev_q, experiment_dir):
    try:
        sys.path.insert(0, str(experiment_dir))
        import config                                     # experiment/config.py
        import pyrealsense2 as rs
        from lib import rs_device                         # experiment/lib/rs_device.py

        rs_device.reset_on_start()
        ev_q.put(("ready", dict(width=config.STREAM_WIDTH, height=config.STREAM_HEIGHT, fps=config.FPS,
                                color_exposure=getattr(config, "COLOR_EXPOSURE", None),
                                devices=len(rs.context().query_devices()))))
        while True:
            try:
                cmd = cmd_q.get(timeout=0.5)
            except queue.Empty:
                if not _parent_alive():
                    return
                continue
            if cmd.get("cmd") == "quit":
                return
            if cmd.get("cmd") != "start":
                continue                                  # 記録していないときの stop は無視
            print(f"[RS] 記録開始 {cmd['db3']}", flush=True)
            res, quit_req = _record_until_stop(cmd["db3"], cmd_q, ev_q, rs_device, config, rs)
            print(f"[RS] 記録終了 {res['status']}  {res['frames']} frames"
                  + (f"（{res['reason']}）" if res["reason"] else ""), flush=True)
            ev_q.put(("stopped", res))
            if res["status"] == "failed":                 # 次の試行に備えてデバイスを立て直す
                rs_device.hardware_reset()
            if quit_req:
                return
    except Exception:
        ev_q.put(("error", traceback.format_exc()))


# ══════════════════════════════════════════════════
#  課題側
# ══════════════════════════════════════════════════
class RSRecorder:
    enabled = True

    def __init__(self, experiment_dir=EXPERIMENT_DIR):
        if not (Path(experiment_dir) / "lib" / "rs_device.py").is_file():
            raise SystemExit(f"[ERROR] {experiment_dir}/lib/rs_device.py がありません（リポジトリの中で実行してください）")
        ctx = mp.get_context("spawn")                     # Windows と同じ起動方法に揃える
        self.cmd_q, self.ev_q = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_child_main, args=(self.cmd_q, self.ev_q, str(experiment_dir)),
                                name="rs_recorder", daemon=False)
        self.proc.start()
        self.state = "booting"      # booting → idle → starting → recording → (stopping) → done → idle …
        self.info = None
        self.result = None
        self.first_frame_epoch_ms = None
        self.error = None

    # ---- イベント ----
    def poll(self):
        """届いたイベントを処理して state を更新する（毎フレーム呼んでよい）．"""
        while True:
            try:
                kind, data = self.ev_q.get_nowait()
            except queue.Empty:
                break
            if kind == "ready":
                self.info, self.state = data, "idle"
            elif kind == "started":
                self.first_frame_epoch_ms = data["first_frame_epoch_ms"]
                if self.state == "starting":
                    self.state = "recording"
            elif kind == "stopped":
                self.result, self.state = data, "done"
            elif kind == "error":
                self.error, self.state = data, "dead"
                print("[RS][ERROR] 記録プロセスでエラー:\n" + data, flush=True)
        if self.state not in ("dead",) and not self.proc.is_alive():
            self.error = self.error or "記録プロセスが終了した"
            self.state = "dead"
        return self.state

    def wait_ready(self, timeout=60.0, tick=None):
        """子プロセスの起動（ハードウェアリセットを含む）を待つ．tick は待つ間に呼ぶ関数（画面の更新など）．"""
        t0 = time.monotonic()
        while self.poll() == "booting":
            if time.monotonic() - t0 > timeout:
                return False
            (tick or (lambda: time.sleep(0.05)))()
        return self.state == "idle"

    # ---- 操作 ----
    def request_start(self, db3_path):
        self.result, self.first_frame_epoch_ms = None, None
        self.state = "starting"
        self.cmd_q.put({"cmd": "start", "db3": str(db3_path)})

    def request_stop(self):
        if self.state in ("starting", "recording"):
            self.state = "stopping"
            self.cmd_q.put({"cmd": "stop"})

    def take_result(self):
        """done の結果を受け取り idle に戻す．"""
        res, self.result = self.result, None
        if self.state == "done":
            self.state = "idle"
        return res

    def close(self, timeout=20.0):
        try:
            self.cmd_q.put({"cmd": "quit"})
        except (OSError, ValueError):
            pass
        self.proc.join(timeout)
        if self.proc.is_alive():
            print("[RS][WARN] 記録プロセスが終わらないので強制終了します", flush=True)
            self.proc.terminate()
            self.proc.join(5)


class NullRecorder:
    """カメラなし（tracking_task.py）．課題側のコードを同じにするための空の記録器．"""
    enabled = False
    state = "idle"
    info = None
    error = None

    def poll(self):
        return self.state

    def close(self):
        pass
