"""
rec_trial.py
────────────────────────────────────────────────────────────────────
マウス操作中の上肢を RealSense で記録する（Unity 連携なし版）．

  python experiment/rec_trial.py                         # 被験者・試行番号なし
  python experiment/rec_trial.py --subject s01 --trial 1 # 試行番号は記録ごとに +1
  python experiment/rec_trial.py --no-cue                # 合図音なし（TRIAL_DURATION 秒）
  python experiment/rec_trial.py --cue-only              # 合図音だけ鳴らす（練習用．記録しない）

  Enter … 接続 → TRIAL_DURATION 秒記録して自動停止 → 切断（1 試行）
  q     … 終了（記録中の Ctrl+C は，その試行を止めて保存してから終了）

保存先（1 試行 = 1 フォルダ）:
  recordings/trial_YYYYMMDD_HHMMSS/session.db3, meta.json
                                  /mocap/   ← Motive の Take CSV をここに置く（sync_check.py などが読む）

meta.json の時刻（*_epoch_ms）は MoCap との時刻合わせに使う．
合図音は既定で鳴らす（記録時間は合図の長さ）．meta.json の "cue" に再生開始の時刻と区間（still, bpm20, ...）を残す
（compare_joints.py が区間ごとに誤差を出す）．合図の中身は config.CUE_*，音は lib/cue.py．
"calib" には後で比較に使う calib_* フォルダ名を書く（比較プログラムが読む）．
────────────────────────────────────────────────────────────────────
"""

import argparse
import shutil
import tempfile
import time

from pathlib import Path

import config
from lib import io_utils
from lib import cue as cue_mod
from lib import rs_device


def record_trial(subject, trial, cue_wav=None, cue_len=None):
    """1 試行を記録して meta.json を書く．失敗・中断でもデータは残し status で区別する．
    cue_wav があれば記録開始と同時に合図音を鳴らし，記録時間は合図の長さにする．
    返り値: record() の status"""
    trial_dir = io_utils.new_session_dir(config.TRIAL_PREFIX)
    (trial_dir / "mocap").mkdir()
    duration = cue_len if cue_wav else config.TRIAL_DURATION
    player = cue_mod.Player(cue_wav) if cue_wav else None
    print(f"[REC] {trial_dir.name}  {duration:.0f} s" + ("（合図音あり）" if player else ""))
    try:
        res = rs_device.record(trial_dir / config.DB3_NAME, duration, on_start=player.start if player else None)
    finally:
        if player:
            player.stop()
    res.pop("last_color")                        # 画像は meta.json に書かない

    if res["frames"] == 0:                       # 1 枚も取れていなければ空フォルダを残さない
        shutil.rmtree(trial_dir, ignore_errors=True)
        print(f"[ERROR] 記録できませんでした（{res['reason']}）．もう一度 Enter で記録できます")
        return res["status"]

    cue = dict(start_epoch=player.start_epoch, segments=cue_mod.schedule(),
               bpms=config.CUE_BPMS, beats=config.CUE_BEATS) if player and hasattr(player, "start_epoch") else None
    io_utils.save_json(trial_dir / config.META_NAME,
                       dict(kind="trial", session=trial_dir.name, subject=subject, trial=trial, calib=None,
                            cue=cue, **res))

    if res["status"] == "ok":
        print(f"[DONE] {trial_dir.name}  {res['frames']} frames  （Take CSV は {trial_dir / 'mocap'} へ）")
    else:                                        # 途中までのデータも解析できるよう消さずに残す
        print(f"[ERROR] {trial_dir.name} は途中で止まりました（{res['reason']}）．meta.json の status で区別できます")
    return res["status"]


def main():
    ap = argparse.ArgumentParser(description="マウス操作の RealSense 記録")
    ap.add_argument("--subject", default="", help="被験者 ID（例 s01）")
    ap.add_argument("--trial", type=int, default=None, help="最初の試行番号（記録ごとに +1）")
    ap.add_argument("--no-cue", action="store_true", help="合図音なしで記録する（TRIAL_DURATION 秒）")
    ap.add_argument("--cue-only", action="store_true", help="合図音だけ鳴らす（練習用）")
    args = ap.parse_args()

    cue_wav = cue_len = None
    if not args.no_cue or args.cue_only:                # 合図音は最初に 1 回だけ作る
        cue_wav = Path(tempfile.gettempdir()) / "trial_cue.wav"
        cue_len = cue_mod.render_wav(cue_wav)
        print("[CUE] " + " → ".join(f"{n}({b - a:.0f}s)" for a, b, n in cue_mod.schedule()))
        print(f"[CUE] 全体 {cue_len:.0f} s")
    if args.cue_only:
        player = cue_mod.Player(cue_wav)
        player.start()
        try:
            time.sleep(cue_len + 0.5)
        except KeyboardInterrupt:
            pass
        player.stop()
        return

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
            status = record_trial(args.subject, trial, cue_wav, cue_len)
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