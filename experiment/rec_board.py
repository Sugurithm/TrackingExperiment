"""
rec_board.py
────────────────────────────────────────────────────────────────────
較正用チェッカーボードを RealSense で記録する．

  python experiment/rec_board.py                          # 新しい較正フォルダを作る
  python experiment/rec_board.py recordings/calib_XXXX    # 既存の較正フォルダに追記（中断後の続き）

  Enter … 接続 → BOARD_DURATION 秒記録 → 切断（1 姿勢）
  r     … 直前の姿勢を消して同じ番号で撮り直す
  q     … 終了（Ctrl+C でも可）

保存先（1 回の起動 = 1 回の較正）:
  recordings/calib_YYYYMMDD_HHMM/
    board/pose_00/session.db3, meta.json
    board/pose_01/...
    mocap/            ← Motive の Take CSV をここに置く（calib_mocap.py が読む）

記録後に最後の画像でボードが検出できたかを表示するので，NG ならその場で r で撮り直す．
────────────────────────────────────────────────────────────────────
"""

import shutil
import sys

import cv2

import config
from lib import io_utils
from lib import rs_device
from lib.board import solve_board
from lib.geometry import K_D


# ══════════════════════════════════════════════════
#  フォルダ
# ══════════════════════════════════════════════════
def open_calib_dir():
    """引数があれば既存フォルダに追記，なければ新規作成．"""
    if len(sys.argv) > 1:
        calib_dir = io_utils.resolve_session_dir(sys.argv[1], config.CALIB_PREFIX)
    else:
        calib_dir = io_utils.new_session_dir(config.CALIB_PREFIX, fmt="%Y%m%d_%H%M")
    (calib_dir / "board").mkdir(exist_ok=True)
    (calib_dir / "mocap").mkdir(exist_ok=True)
    return calib_dir


def pose_dirs(calib_dir):
    return sorted((calib_dir / "board").glob("pose_*"))


def next_pose_dir(calib_dir):
    dirs = pose_dirs(calib_dir)
    k = int(dirs[-1].name.split("_")[1]) + 1 if dirs else 0
    return calib_dir / "board" / f"pose_{k:02d}"


# ══════════════════════════════════════════════════
#  1 姿勢の記録
# ══════════════════════════════════════════════════
def record_pose(pose_dir):
    """記録して meta.json を書く．失敗したらフォルダごと消して番号を空けておく．
    返り値: record() の status"""
    pose_dir.mkdir(parents=True)
    print(f"[REC] {pose_dir.name}  {config.BOARD_DURATION:.0f} s  ボードを静止させてください")
    res = rs_device.record(pose_dir / config.DB3_NAME, config.BOARD_DURATION)

    if res["status"] != "ok":                    # 書きかけの .db3 を残さない（calib_rs.py が拾わないように）
        shutil.rmtree(pose_dir, ignore_errors=True)
        print(f"[ERROR] {pose_dir.name} を記録できませんでした（{res['reason']}）．もう一度 Enter で撮り直せます")
        return res["status"]

    # その場の確認: 最後の画像でボードが解けるか（誤差の閾値は緩めにして検出の有無だけ見る）
    found = False
    if res["last_color"] is not None:
        K, D = K_D(res["color_intrinsics"])
        found = solve_board(cv2.cvtColor(res["last_color"], cv2.COLOR_BGR2GRAY), K, D, max_err=2.0) is not None
    res["board_found_last_frame"] = found

    res.pop("last_color")                        # 画像は meta.json に書かない
    io_utils.save_json(pose_dir / config.META_NAME, dict(kind="board", pose=pose_dir.name, **res))
    print(f"[DONE] {pose_dir.name}  {res['frames']} frames  ボード検出: {'OK' if found else 'NG（r で撮り直し推奨）'}")
    return "ok"


# ══════════════════════════════════════════════════
#  メイン
# ══════════════════════════════════════════════════
def main():
    calib_dir = open_calib_dir()
    print(f"[CALIB] {calib_dir}")
    rs_device.reset_on_start()
    print("Enter: 記録 / r: 直前を撮り直し / q: 終了")

    try:
        while True:
            cmd = input(f"[{next_pose_dir(calib_dir).name}] > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "r":
                dirs = pose_dirs(calib_dir)
                if not dirs:                     # 消すものがなければ何もしない
                    print("[WARN] 撮り直す姿勢がありません")
                    continue
                shutil.rmtree(dirs[-1])
                print(f"[REDO] {dirs[-1].name} を削除しました")
            elif cmd:                            # 打ち間違いで記録が始まらないようにする
                print("[WARN] Enter / r / q のいずれかを入力してください")
                continue
            if record_pose(next_pose_dir(calib_dir)) == "interrupted":
                break
    except (KeyboardInterrupt, EOFError):         # 入力待ち中の Ctrl+C / Ctrl+Z でも下の表示まで通す
        print()
    finally:
        print(f"[END] {calib_dir}  {len(pose_dirs(calib_dir))} 姿勢")
        print(f"      Motive の Take CSV を {calib_dir / 'mocap'} に置いてから calib_*.py を実行してください")


if __name__ == "__main__":
    main()
