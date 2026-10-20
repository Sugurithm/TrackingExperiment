"""
tracking_task.py
────────────────────────────────────────────────────────────────────
2D エイムトラッキング課題（カメラなし．課題だけを動かす・練習する・DPI を測るとき）．
RealSense と一緒に記録するときは run_experiment.py を使う（画面と保存の形は同じ）．

  python task/tracking_task.py P01                  # 静止 → 練習 2 本 → 本番 12 本
  python task/tracking_task.py P01 --only fast:1    # 1 本だけ（--no-static で静止も飛ばす）
  python task/tracking_task.py --measure-dpi        # マウスの counts/inch を定規で実測
  python task/tracking_task.py P99 --simulate --fast --window   # 自動プレイ（動作確認）

マウスは相対モード（カーソルを隠して画面に閉じ込める．Windows では Raw Input）で読み，
counts → cm（task_config の mouse_dpi）→ 画面の px で動かす．
Windows の「ポインターの精度を高める」は切っておくこと．

保存先・ファイルの形は task_core.py の先頭を参照．
────────────────────────────────────────────────────────────────────
"""

import argparse

import pygame

from rs_recorder import NullRecorder
from task_core import App, Quit, add_common_args


def main():
    ap = argparse.ArgumentParser(description="2D エイムトラッキング課題（カメラなし）")
    add_common_args(ap)
    ap.add_argument("--measure-dpi", action="store_true", help="マウスの counts/inch を定規で実測する")
    args = ap.parse_args()
    if args.fast and not args.simulate:
        ap.error("--fast は --simulate と一緒に使う")
    if not args.measure_dpi and not args.participant:
        ap.error("参加者 ID を指定してください（例 P01）")
    pygame.init()
    app = App(args, NullRecorder())
    try:
        if args.measure_dpi:
            app.measure_dpi()
        else:
            app.run_session(args.participant, note=args.note)
    except Quit:
        pass
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()