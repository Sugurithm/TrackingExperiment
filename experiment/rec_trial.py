"""
rec_trial.py
────────────────────────────────────────────────────────────────────
マウス操作中の上肢を RealSense で記録する（Unity 連携なし版）．

  python experiment/rec_trial.py                         # 被験者・試行番号なし
  python experiment/rec_trial.py --subject s01 --trial 1 # 試行番号は記録ごとに +1

  Enter … 接続 → TRIAL_DURATION 秒記録して自動停止 → 切断（1 試行）
  q     … 終了（記録中の Ctrl+C は，その試行を止めて保存してから終了）

保存先（1 試行 = 1 フォルダ）:
  recordings/trial_YYYYMMDD_HHMMSS/session.db3, meta.json

meta.json の時刻（*_epoch_ms）は MoCap との時刻合わせに使う．
"calib" には後で比較に使う calib_* フォルダ名を書く（比較プログラムが読む）．
────────────────────────────────────────────────────────────────────
"""

import argparse
import shutil

import config
import io_utils
import rs_device


def record_trial(subject, trial):
    """1 試行を記録して meta.json を書く．失敗・中断でもデータは残し status で区別する．
    返り値: record() の status"""
    trial_dir = io_utils.new_session_dir(config.TRIAL_PREFIX)
    print(f"[REC] {trial_dir.name}  {config.TRIAL_DURATION:.0f} s")
    res = rs_device.record(trial_dir / config.DB3_NAME, config.TRIAL_DURATION)
    res.pop("last_color")                        # 画像は meta.json に書かない

    if res["frames"] == 0:                       # 1 枚も取れていなければ空フォルダを残さない
        shutil.rmtree(trial_dir, ignore_errors=True)
        print(f"[ERROR] 記録できませんでした（{res['reason']}）．もう一度 Enter で記録できます")
        return res["status"]

    io_utils.save_json(trial_dir / config.META_NAME,
                       dict(kind="trial", session=trial_dir.name, subject=subject, trial=trial, calib=None, **res))

    if res["status"] == "ok":
        print(f"[DONE] {trial_dir.name}  {res['frames']} frames")
    else:                                        # 途中までのデータも解析できるよう消さずに残す
        print(f"[ERROR] {trial_dir.name} は途中で止まりました（{res['reason']}）．meta.json の status で区別できます")
    return res["status"]


def main():
    ap = argparse.ArgumentParser(description="マウス操作の RealSense 記録")
    ap.add_argument("--subject", default="", help="被験者 ID（例 s01）")
    ap.add_argument("--trial", type=int, default=None, help="最初の試行番号（記録ごとに +1）")
    args = ap.parse_args()

    rs_device.reset_on_start()
    trial = args.trial
    print("Enter: 記録開始 / q: 終了")

    try:
        while True:
            label = f"trial {trial}" if trial is not None else "trial"
            cmd = input(f"[{args.subject or '-'} {label}] > ").strip().lower()
            if cmd == "q":
                break
            if cmd:                              # 打ち間違いで記録が始まらないようにする
                print("[WARN] Enter / q のいずれかを入力してください")
                continue
            status = record_trial(args.subject, trial)
            if status == "interrupted":
                break
            if status == "ok" and trial is not None:
                trial += 1
    except (KeyboardInterrupt, EOFError):         # 入力待ち中の Ctrl+C / Ctrl+Z
        print()
    finally:
        print("[END]")


if __name__ == "__main__":
    main()
