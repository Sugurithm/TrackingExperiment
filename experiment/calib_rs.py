"""
calib_rs.py
────────────────────────────────────────────────────────────────────
較正フォルダ内の各姿勢の .db3 から T_CB（ボード → カラー光学座標, mm）を求める．
（scripts/realsense_transfer.py を較正フォルダ単位で動くようにしたもの）

  python experiment/calib_rs.py                          # 最新の calib_*
  python experiment/calib_rs.py recordings/calib_XXXX

入力:  calib_*/board/pose_*/session.db3
出力:  calib_*/board/pose_*/T_CB.json   … T_CB, K, D, 誤差など
                          /frame.png    … 代表フレーム（calib_solve.py の重ね合わせに使う）
                          /pose_check.png … 軸とマーカー位置を投影した確認画像
────────────────────────────────────────────────────────────────────
"""

import sys

import numpy as np
import cv2

import config
from lib import io_utils
from lib.board import MARKERS_B, solve_board
from lib.geometry import K_D, apply, make_T, mean_R, project
from lib.rs_device import Playback


# ══════════════════════════════════════════════════
#  .db3 → 毎フレームのボード姿勢
# ══════════════════════════════════════════════════
def detect_all(db3_path):
    """全フレームでボード姿勢を解く．返り値: (intrinsics, 検出結果のリスト, {stamp: PNG バイト列})"""
    results, images = [], {}
    with Playback(db3_path) as pb:               # with を抜けると必ず停止する
        K, D = K_D(pb.intrinsics)
        n = 0
        for stamp, img in pb.color_frames(stride=config.RS_STRIDE):
            n += 1
            res = solve_board(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), K, D)
            if res:
                res["stamp"] = stamp
                results.append(res)
                images[stamp] = cv2.imencode(".png", img)[1]   # 代表フレーム保存用（PNG で圧縮して保持）
    print(f"  detected {len(results)} / {n} frames")
    return pb.intrinsics, results, images


# ══════════════════════════════════════════════════
#  1 姿勢にまとめる
# ══════════════════════════════════════════════════
def summarize(results):
    """先頭・末尾を捨て，回転は平均，並進は中央値で 1 姿勢にまとめる．足りなければ None．"""
    if not results:
        return None
    t0 = results[0]["stamp"] + config.RS_TRIM_HEAD
    t1 = results[-1]["stamp"] - config.RS_TRIM_TAIL
    used = [r for r in results if t0 <= r["stamp"] <= t1]
    if len(used) < config.RS_MIN_FRAMES:
        return None
    ts = np.array([r["t"] for r in used])
    T = make_T(mean_R([r["R"] for r in used]), np.median(ts, axis=0))
    rep = used[len(used) // 2]                   # 代表フレーム（使用区間の中央）
    return dict(T=T, used=used, rep=rep, tvec_std=ts.std(axis=0), rms=float(np.mean([r["err"] for r in used])))


def save_check_image(path, img, T, K, D):
    """軸（50 mm）と S1〜S3 の位置を投影した確認画像．"""
    vis = img.copy()
    cv2.drawFrameAxes(vis, K, D, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], 50)
    for name, (u, v) in zip(config.MARKERS_B, project(apply(T, MARKERS_B), K, D).astype(int)):
        cv2.circle(vis, (u, v), 10, (0, 0, 255), 2)
        cv2.putText(vis, name, (u + 12, v), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(path), vis)


# ══════════════════════════════════════════════════
#  1 姿勢分の処理
# ══════════════════════════════════════════════════
def process_pose(pose_dir):
    out_json = pose_dir / "T_CB.json"
    out_json.unlink(missing_ok=True)             # 前回の結果が残っていると失敗時に古い値を使ってしまうため消す

    db3 = pose_dir / config.DB3_NAME
    if not db3.is_file():                        # 記録失敗で .db3 がない姿勢は飛ばす
        print("  [SKIP] session.db3 がありません")
        return False
    try:
        intr, results, images = detect_all(db3)
    except RuntimeError as e:                    # .db3 の破損・形式違い
        print(f"  [ERROR] 再生に失敗: {e}")
        return False

    s = summarize(results)
    if s is None:                                # ボードが写っていない・ぶれている
        print(f"  [ERROR] 使えるフレームが {config.RS_MIN_FRAMES} 枚未満です（撮り直してください）")
        return False

    K, D = K_D(intr)
    img = cv2.imdecode(images[s["rep"]["stamp"]], cv2.IMREAD_COLOR)
    cv2.imwrite(str(pose_dir / "frame.png"), img)
    save_check_image(pose_dir / "pose_check.png", img, s["T"], K, D)

    # T_CB      : ボード座標 → カラー光学座標 [mm]
    # tvec_std  : 使用フレーム間の並進のばらつき [mm]
    # rms_px    : 再投影誤差の平均 [px]
    # t_start, t_end : 使用フレームの最初と最後の時刻 [s]（RealSense の時計，PC の epoch）
    # frame_stamp, frame_corners : frame.png の時刻と検出コーナー [px]（calib_solve.py の検証用）
    io_utils.save_json(out_json, dict(
        T_CB=s["T"].tolist(), K=K.tolist(), D=D.tolist(), intrinsics=intr,
        square_mm=config.SQUARE, pattern=list(config.PATTERN),
        n_frames=len(s["used"]), tvec_std=s["tvec_std"].tolist(), rms_px=s["rms"],
        t_start=s["used"][0]["stamp"], t_end=s["used"][-1]["stamp"],
        frame_stamp=s["rep"]["stamp"], frame_corners=s["rep"]["corners"].tolist()))
    print(f"  used {len(s['used'])} frames  t={s['T'][:3, 3].round(1)} mm  "
          f"std={s['tvec_std'].round(2)} mm  rms={s['rms']:.3f} px")
    return True


def main():
    calib_dir = io_utils.resolve_session_dir(sys.argv[1] if len(sys.argv) > 1 else None, config.CALIB_PREFIX)
    pose_dirs = sorted((calib_dir / "board").glob("pose_*"))
    print(f"[CALIB] {calib_dir}  {len(pose_dirs)} 姿勢")
    if not pose_dirs:
        raise SystemExit("[ERROR] board/pose_* がありません")

    ok = {}
    for pose_dir in pose_dirs:
        print(f"[{pose_dir.name}]")
        ok[pose_dir.name] = process_pose(pose_dir)

    print("[SUMMARY] " + "  ".join(f"{k}:{'OK' if v else 'NG'}" for k, v in ok.items()))
    if not all(ok.values()):
        print("[WARN] NG の姿勢は rec_board.py で撮り直すか，フォルダごと削除してから calib_solve.py を実行してください")


if __name__ == "__main__":
    main()
