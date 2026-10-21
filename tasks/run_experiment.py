"""
run_experiment.py
────────────────────────────────────────────────────────────────────
2D エイムトラッキング課題 + RealSense の記録（実験協力者の本番用）．

  python tasks/run_experiment.py P01 --calib calib_20261020_1400
  python tasks/run_experiment.py P01 --only fast:1 --no-practice      # 1 本だけ（確認用）
  python tasks/run_experiment.py P01 --dpi 400 --calib calib_...   # PC の DPI を 400 にしたとき（800, 1600 も同様）
      → フォルダ名に dpi400 が入り，eDPI と必要なマウスの移動量が session.json の "sensitivity" に残る

流れ（1 試行ごと）:
  SPACE → カメラの記録を開始（最初のフレームが届くまで待つ）→ カウントダウン → 課題 →
  0.5 s 余分に記録して停止 → 保存 → 結果の画面（SPACE: 次へ / R: この試行をやり直す）

  - 試行ごとに 1 つの .db3（trial_XX_*/session.db3）．experiment/rec_trial.py と同じく
    1 試行 = 接続 → 記録 → 切断．カメラの設定は experiment/config.py（較正と同じ）．
  - RealSense のフレーム時刻（global_time_enabled）と task.csv の epoch は同じ PC の時計なので，
    同じ時間軸でそのまま対応づけられる．
    残る小さなずれ（フレーム時刻の変換・露光のタイミング）は，後でマウスの速さと手首の速さを比べて確かめる．
  - 記録の開始に失敗したら画面で R（もう一度）/ S（カメラなしで続ける）を選ぶ．
    記録が途中で止まった・fps が落ちた・課題の全体を覆っていないときは結果の画面に出るので R でやり直す．
  - 静止・練習・本番のすべてを記録する（練習を記録しないときは task_config の rs_record_practice = False）．

試行のフォルダの meta.json は experiment/ の trial_* と同じ形なので，関節の推定はそのまま:
  python experiment/rs_joints.py recordings/aim_P01_YYYYMMDD_HHMMSS/trial_03_fast_s1

--calib には，この日に机の上のボードを撮った較正フォルダ（recordings/calib_*）の名前を書く．
meta.json の "calib" に入り，後で机の座標系に直すときに使う．
────────────────────────────────────────────────────────────────────
"""

import argparse

import pygame

from rs_recorder import RSRecorder
from task_core import App, Quit, add_common_args


def main():
    ap = argparse.ArgumentParser(description="2D エイムトラッキング課題 + RealSense の記録")
    add_common_args(ap)
    ap.add_argument("--calib", default=None, help="この日の較正フォルダ名（recordings/calib_*）")
    args = ap.parse_args()
    if args.fast and not args.simulate:
        ap.error("--fast は --simulate と一緒に使う")
    if not args.participant:
        ap.error("参加者 ID を指定してください（例 P01）")
    if not args.calib:
        print("[WARN] --calib がありません．後で meta.json の \"calib\" に較正フォルダ名を書いてください")

    rec = RSRecorder()                 # 子プロセスで RealSense を開く（ハードウェアリセットを含む）
    pygame.init()
    try:
        app = App(args, rec)
        app.boot_camera()
        app.run_session(args.participant, calib=args.calib, note=args.note)
    except Quit:
        pass
    finally:
        pygame.quit()
        rec.close()                    # 記録中でも .db3 を閉じてから終わる


if __name__ == "__main__":
    main()