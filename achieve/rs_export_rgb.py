# session.db3 から RGB フレームとタイムスタンプを書き出す（深度は含めない）．
# Motive .csv との同期やチェッカーボードキャリブレーション用．
#
# 使い方:
#   python scripts/rs_export_rgb.py              # recordings/rec_* をすべて処理
#   python scripts/rs_export_rgb.py --session recordings/rec_.../session.db3
#   python scripts/rs_export_rgb.py --stride 2   # 2 フレームに 1 枚

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"


def intrinsics_to_dict(intr):
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
    }


def export_session(db3_path: Path, stride: int, jpeg_quality: int) -> int:
    db3_path = db3_path.resolve()
    if not db3_path.is_file():
        raise FileNotFoundError(db3_path)

    out_dir = db3_path.parent / "rgb"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, str(db3_path), repeat_playback=False)
    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr_path = db3_path.parent / "color_intrinsics.json"
    if not intr_path.exists():
        intr_path.write_text(
            json.dumps(intrinsics_to_dict(color_stream.get_intrinsics()), indent=2),
            encoding="utf-8",
        )

    ts_path = db3_path.parent / "frame_timestamps.csv"
    n_written = 0
    frame_idx = 0

    with ts_path.open("w", newline="", encoding="utf-8") as tf:
        writer = csv.writer(tf)
        writer.writerow(
            [
                "frame_index",
                "color_frame_number",
                "timestamp_ms",
                "domain",
            ]
        )
        while True:
            ok, frames = pipeline.try_wait_for_frames(timeout_ms=3000)
            if not ok:
                break
            color = frames.get_color_frame()
            if not color:
                continue
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            img = np.asanyarray(color.get_data())
            name = f"frame_{n_written:06d}.jpg"
            cv2.imwrite(
                str(out_dir / name),
                img,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            writer.writerow(
                [
                    n_written,
                    color.get_frame_number(),
                    color.get_timestamp(),
                    color.get_frame_timestamp_domain().name,
                ]
            )
            n_written += 1
            frame_idx += 1

    pipeline.stop()
    print(f"[EXPORT] {db3_path.parent.name}: {n_written} frames -> {out_dir}")
    return n_written


def discover_sessions(session_arg: str | None) -> list[Path]:
    if session_arg:
        p = Path(session_arg)
        if p.is_dir():
            p = p / "session.db3"
        return [p]
    return sorted(RECORDINGS_DIR.glob("rec_*/session.db3"))


def main():
    ap = argparse.ArgumentParser(description="Export RGB JPEGs from RealSense .db3")
    ap.add_argument("--session", help="session.db3 or rec_* directory (default: all rec_*)")
    ap.add_argument("--stride", type=int, default=1, help="save every Nth color frame")
    ap.add_argument("--quality", type=int, default=92, help="JPEG quality 0-100")
    args = ap.parse_args()

    sessions = discover_sessions(args.session)
    if not sessions:
        raise SystemExit(f"no session.db3 under {RECORDINGS_DIR}")

    total = 0
    for db3 in sessions:
        total += export_session(db3, args.stride, args.quality)
    print(f"[DONE] {len(sessions)} session(s), {total} image(s)")


if __name__ == "__main__":
    main()
