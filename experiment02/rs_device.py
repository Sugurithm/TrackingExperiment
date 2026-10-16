"""
rs_device.py
────────────────────────────────────────────────────────────────────
RealSense の接続/切断と .db3 の記録・再生．

  record(db3_path, duration)  … 接続 → duration 秒記録 → 切断 を 1 回行う
  Playback(db3_path)          … .db3 を with 文で開き，カラーフレームを順に読む

【エラー処理】（scripts/rs_review.py と同じ考え方）
  - プログラム起動時に reset_on_start() でハードウェアリセットし，
    前回異常終了した状態を解消する（USB を抜き差ししなくても再実行できる）
  - フレームが MAX_TIMEOUTS 回連続で届かなければ記録失敗とし，
    パイプライン停止 → ハードウェアリセットまで行って次の記録に備える
    （記録途中での自動再起動はしない．.db3 が途中で分かれてしまうため）
  - どの経路で抜けても safe_stop() を通し，.db3 を閉じてから戻る
────────────────────────────────────────────────────────────────────
"""

import time

import numpy as np
import cv2
import pyrealsense2 as rs

import config


# ══════════════════════════════════════════════════
#  デバイス操作（rs_review.py と同じ）
# ══════════════════════════════════════════════════
def wait_for_device(timeout_s=config.RECONNECT_WAIT_SEC):
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


def reset_on_start():
    """各プログラムの最初に 1 回呼ぶ．"""
    if config.RESET_ON_START:
        hardware_reset()


def safe_stop(pipeline):
    """stop 自体が失敗しても落ちないようにする．"""
    if pipeline is None:
        return
    try:
        pipeline.stop()
    except RuntimeError as e:
        print(f"[WARN] pipeline.stop() 失敗: {e}")


def intrinsics_to_dict(intr):
    return dict(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
                ppx=intr.ppx, ppy=intr.ppy, model=str(intr.model), coeffs=list(intr.coeffs))


# ══════════════════════════════════════════════════
#  記録
# ══════════════════════════════════════════════════
def _start_record_pipeline(db3_path):
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, config.STREAM_WIDTH, config.STREAM_HEIGHT, rs.format.z16, config.FPS)
    cfg.enable_stream(rs.stream.color, config.STREAM_WIDTH, config.STREAM_HEIGHT, rs.format.bgr8, config.FPS)
    cfg.enable_record_to_file(str(db3_path))

    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)

    # データ品質に影響する 2 オプションのみ設定（scripts/rs_recorder*.py と同じ）
    for sensor in profile.get_device().query_sensors():
        for option, value in (
            (rs.option.global_time_enabled, 1),     # タイムスタンプを PC の絶対時刻(epoch ms)に揃える
            (rs.option.auto_exposure_priority, 0),  # 自動露出優先を無効化して FPS を安定化
        ):
            if sensor.supports(option):
                sensor.set_option(option, value)
    return pipeline, profile


class _StartFailed(Exception):
    """record() 内で接続失敗を記録ループの後ろへ飛ばすための目印．"""


def record(db3_path, duration):
    """接続 → duration 秒記録 → 切断．結果を dict で返す（meta.json にそのまま書ける）．

    status: "ok" / "failed"（起動失敗・フレーム途絶） / "interrupted"（Ctrl+C）
    last_color は確認用の最終カラー画像（meta.json には書かない）．
    """
    res = dict(status="ok", reason="", duration_s=duration, frames=0,
               width=config.STREAM_WIDTH, height=config.STREAM_HEIGHT, fps=config.FPS,
               rec_start_epoch_ms=None, rec_stop_epoch_ms=None,
               first_frame_epoch_ms=None, last_frame_epoch_ms=None,
               color_intrinsics=None, last_color=None)
    pipeline = None
    try:
        # ── 接続（失敗したら記録せず，下の finally → リセットへ進む） ──
        try:
            pipeline, profile = _start_record_pipeline(db3_path)
        except RuntimeError as e:
            res.update(status="failed", reason=f"起動に失敗: {e}")
            raise _StartFailed

        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        res["color_intrinsics"] = intrinsics_to_dict(intr)
        res["rec_start_epoch_ms"] = int(time.time() * 1000)
        t0, timeouts, shown = time.monotonic(), 0, -1

        # ── 記録ループ（duration 秒経過で抜ける） ──
        while (elapsed := time.monotonic() - t0) < duration:
            try:
                ok, frames = pipeline.try_wait_for_frames(timeout_ms=config.FRAME_TIMEOUT_MS)
            except RuntimeError as e:           # USB 切断などで例外になる場合もタイムアウト扱い
                print(f"\n[WARN] フレーム取得エラー: {e}")
                ok = False

            if not ok:                          # 連続で届かなければ記録失敗として打ち切る
                timeouts += 1
                print(f"\n[WARN] フレームが届きません ({timeouts}/{config.MAX_TIMEOUTS})")
                if timeouts >= config.MAX_TIMEOUTS:
                    res.update(status="failed", reason="フレームが届かなくなった")
                    break
                continue
            timeouts = 0

            ts = frames.get_timestamp()
            res["first_frame_epoch_ms"] = res["first_frame_epoch_ms"] or ts
            res["last_frame_epoch_ms"] = ts
            res["frames"] += 1

            # 1 秒ごとに残り秒数を表示し，確認用のカラー画像を 1 枚コピーしておく
            # （フレーム自体を保持し続けると librealsense のフレームプールを圧迫するため）
            if int(duration - elapsed) != shown:
                shown = int(duration - elapsed)
                print(f"\r[REC] 残り {shown + 1:3d} s", end="", flush=True)
                color = frames.get_color_frame()
                if color:
                    res["last_color"] = np.asanyarray(color.get_data()).copy()

    except _StartFailed:                        # 起動失敗（status は設定済み）
        pass
    except KeyboardInterrupt:                   # Ctrl+C でも .db3 を閉じてから戻る
        res.update(status="interrupted", reason="Ctrl+C")
        res["last_color"] = None
    finally:
        safe_stop(pipeline)
        res["rec_stop_epoch_ms"] = int(time.time() * 1000)
        print()

    if res["status"] == "failed":               # 次の記録に備えてデバイスを立て直す
        hardware_reset()
    return res


# ══════════════════════════════════════════════════
#  再生
# ══════════════════════════════════════════════════
class Playback:
    """with Playback(path) as pb: で開き，pb.color_frames() でカラー画像を順に読む．
    with を抜けると（例外でも）必ず safe_stop() を通る．"""

    def __init__(self, db3_path):
        self.path = db3_path
        self.pipeline = None

    def __enter__(self):
        cfg = rs.config()
        rs.config.enable_device_from_file(cfg, str(self.path), repeat_playback=False)
        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(cfg)              # ファイル破損・形式違いはここで RuntimeError
        profile.get_device().as_playback().set_real_time(False)   # 全フレームを落とさず再生
        self.stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = intrinsics_to_dict(self.stream.get_intrinsics())
        return self

    def __exit__(self, *exc):
        safe_stop(self.pipeline)
        return False                                    # 例外は呼び出し側へそのまま流す

    def color_frames(self, stride=1):
        """(timestamp [s], BGR 画像) を stride 枚に 1 枚ずつ返す．ファイル終端で終わる．"""
        i = 0
        while True:
            ok, fs = self.pipeline.try_wait_for_frames(timeout_ms=3000)
            if not ok:                                  # ファイル終端（再生ではこれが正常終了）
                return
            cf = fs.get_color_frame()
            if not cf:
                continue
            i += 1
            if (i - 1) % stride:
                continue
            img = np.asanyarray(cf.get_data()).copy()
            if self.stream.format() == rs.format.rgb8:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            yield cf.get_timestamp() / 1000.0, img
