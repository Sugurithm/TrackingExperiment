"""
rs_joints.py
────────────────────────────────────────────────────────────────────
試行の .db3 から MediaPipe で右上肢の関節を推定し，位置合わせ済みの深度で
3 次元座標（カラー光学座標, mm）にする．
（scripts/rs_estimate.py を試行フォルダ単位にし，比較に必要な形で保存するようにしたもの）

  python experiment/rs_joints.py                         # 最新の trial_*
  python experiment/rs_joints.py recordings/trial_XXXX
  python experiment/rs_joints.py recordings/trial_XXXX --show     # 処理しながら推定結果を画面に出す（q で表示だけ止める）
  python experiment/rs_joints.py recordings/trial_XXXX --summary  # 保存済みの結果から有効率などを出し直すだけ

入力:  trial_*/session.db3
       models/pose_landmarker_heavy.task, models/hand_landmarker.task
出力:  trial_*/joints/rs_joints.npz  … t（RealSense 時刻, epoch s），names，
                                        P_C（frames × joints × 3, mm, 取れなければ NaN），uv（画素）
                     rs_check.png     … 記録の中ほどのフレームに関節を重ねた確認画像
       画面: 関節ごとの有効率（全体 / 5 秒ごと / 動作区間），上腕・前腕の長さのばらつき
             （動作区間 = 最初の同期区間より後でボードを振った区間を除く．sync_check.py の後なら出る）
       確認動画は joints_video.py で後から作る（MediaPipe をやり直さない）

  関節の対応（名前 → MediaPipe のランドマーク）は config.RS_JOINTS で決める．

【rs_estimate.py からの変更点】
  - 深度が取れないフレームは NaN のまま（直前の値で埋めない．誤差の評価で失敗を隠さないため）
  - 深度は 1 画素ではなく ±DEPTH_PATCH px の中央値（関節が腕の縁にあると背景を拾うため）
  - 手は 2 本まで検出し，Pose の右手首に最も近い手を使う（左手を右手と取り違えないため）
  - MediaPipe に渡す時刻は実際のタイムスタンプ（フレーム落ちがあっても追跡が乱れない）
────────────────────────────────────────────────────────────────────
"""

import argparse

import numpy as np
import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, PoseLandmarker, PoseLandmarkerOptions,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

import config
from lib import io_utils, sync_io
from lib.geometry import K_D, deproject
from lib.rs_device import Playback
from lib.viz import draw_joints

NAMES = list(config.RS_JOINTS)
POSE_WRIST, POSE_ELBOW = 16, 14          # 右手首・右肘（手の選択に使う）


# ══════════════════════════════════════════════════
#  MediaPipe
# ══════════════════════════════════════════════════
def make_landmarkers():
    for path in (config.POSE_MODEL, config.HAND_MODEL):
        if not path.is_file():           # models/ は Git に載せていない
            raise SystemExit(f"[ERROR] {path} がありません（MediaPipe のモデルを models/ に置いてください）")
    pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.POSE_MODEL)),
        running_mode=VisionTaskRunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=config.POSE_CONFIDENCE, min_tracking_confidence=config.POSE_CONFIDENCE))
    hand = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.HAND_MODEL)),
        running_mode=VisionTaskRunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=config.HAND_CONFIDENCE, min_tracking_confidence=config.HAND_CONFIDENCE))
    return pose, hand


def pick_right_hand(hands, pose_uv):
    """検出された手（それぞれ 21×2 の画素座標）から，Pose の右手首に最も近い手を返す．
    Pose がなく手が 2 本ある・最も近い手でも前腕の長さより遠い，なら None（取り違えを避ける）．"""
    if not hands:
        return None
    if pose_uv is None:
        return hands[0] if len(hands) == 1 else None
    wrist, forearm = pose_uv[POSE_WRIST], np.linalg.norm(pose_uv[POSE_WRIST] - pose_uv[POSE_ELBOW])
    best = min(hands, key=lambda H: np.linalg.norm(H[0] - wrist))
    return best if np.linalg.norm(best[0] - wrist) <= forearm else None


def landmarks_uv(pose, hand, img, ts_ms):
    """1 枚の画像 → 関節ごとの画素座標 (joints × 2)．見つからない関節は NaN．"""
    h, w = img.shape[:2]
    mp_img = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pose_res = pose.detect_for_video(mp_img, ts_ms)
    hand_res = hand.detect_for_video(mp_img, ts_ms)

    pose_uv = (np.array([[lm.x * w, lm.y * h] for lm in pose_res.pose_landmarks[0]])
               if pose_res.pose_landmarks else None)
    hands = [np.array([[lm.x * w, lm.y * h] for lm in lms]) for lms in hand_res.hand_landmarks]
    hand_uv = pick_right_hand(hands, pose_uv)

    uv = np.full((len(NAMES), 2), np.nan)
    for k, name in enumerate(NAMES):
        model, idx = config.RS_JOINTS[name]
        src = pose_uv if model == "pose" else hand_uv
        if src is not None:
            uv[k] = src[idx]
    return uv


# ══════════════════════════════════════════════════
#  深度 → 3 次元
# ══════════════════════════════════════════════════
def depth_at(depth, uv, r=config.DEPTH_PATCH):
    """各点の周り ±r px の有効な深度の中央値 [mm]．取れなければ NaN．"""
    h, w = depth.shape
    lo, hi = config.DEPTH_RANGE_MM
    z = np.full(len(uv), np.nan)
    for k, (u, v) in enumerate(uv):
        if not np.isfinite(u):
            continue
        u, v = int(round(u)), int(round(v))
        patch = depth[max(v - r, 0):min(v + r + 1, h), max(u - r, 0):min(u + r + 1, w)]
        patch = patch[(patch > lo) & (patch < hi)]
        if patch.size:
            z[k] = np.median(patch)
    return z


# ══════════════════════════════════════════════════
#  確認
# ══════════════════════════════════════════════════
def save_check_image(path, img, uv):
    vis = img.copy()
    for name, (u, v) in zip(NAMES, uv):
        if np.isfinite(u):
            cv2.circle(vis, (int(u), int(v)), 6, (0, 200, 255), 2)
            cv2.putText(vis, name, (int(u) + 8, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    cv2.imwrite(str(path), vis)


def valid_line(ok):
    return "  ".join(f"{n}: {100 * ok[:, k].mean():3.0f}%" for k, n in enumerate(NAMES))


def print_summary(trial_dir, t, P):
    """有効率（3 次元座標が取れたフレームの割合）と，上腕・前腕の長さ．"""
    ok = np.isfinite(P[:, :, 0])
    print(f"[VALID 全体]     {valid_line(ok)}")
    rel = t - t[0]
    print("[VALID 5 秒ごと] " + "  ".join(NAMES) + "（%）")
    for s0 in np.arange(0, rel[-1], 5.0):
        m = (rel >= s0) & (rel < s0 + 5)
        if m.any():
            print(f"    {s0:5.0f}–{s0 + 5:3.0f} s  " + "  ".join(f"{100 * ok[m, k].mean():5.0f}" for k in range(len(NAMES))))
    sync, _, _ = sync_io.load_sync(trial_dir)
    use = sync_io.operation_mask(sync, t) if sync else np.ones(len(t), bool)
    if sync:
        print(f"[VALID 動作区間] {valid_line(ok[use])}  （{use.sum()} frames）")
    else:
        print("[VALID 動作区間] sync_check.py の後に実行すると，同期区間を除いた有効率も出ます")
    for a, b in config.SEGMENTS:                 # 上腕・前腕の長さは一定のはず．ばらつきが精度の目安
        if a in NAMES and b in NAMES:
            L = np.linalg.norm(P[use, NAMES.index(a)] - P[use, NAMES.index(b)], axis=1)
            L = L[np.isfinite(L)]
            if L.size:
                q1, q3 = np.percentile(L, [25, 75])
                print(f"[LENGTH] {a}–{b}: 中央値 {np.median(L):.0f} mm  四分位範囲 {q3 - q1:.1f} mm  （{L.size} frames）")


# ══════════════════════════════════════════════════
#  メイン
# ══════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=".db3 → MediaPipe → 関節の 3 次元座標")
    ap.add_argument("trial", nargs="?", help="trial_* フォルダ（省略すると最新）")
    ap.add_argument("--show", action="store_true", help="処理しながら推定結果を画面に出す")
    ap.add_argument("--summary", action="store_true", help="保存済みの rs_joints.npz から集計を出し直すだけ")
    args = ap.parse_args()

    trial_dir = io_utils.resolve_session_dir(args.trial, config.TRIAL_PREFIX)
    if args.summary:
        z = np.load(trial_dir / "joints" / "rs_joints.npz")
        print(f"[TRIAL] {trial_dir}")
        print_summary(trial_dir, z["t"], z["P_C"])
        return
    db3 = trial_dir / config.DB3_NAME
    if not db3.is_file():
        raise SystemExit(f"[ERROR] {db3} がありません")
    n_total = io_utils.load_json(trial_dir / config.META_NAME).get("frames", 0)
    out = trial_dir / "joints"
    out.mkdir(exist_ok=True)
    print(f"[TRIAL] {trial_dir}  関節: {', '.join(NAMES)}")

    pose, hand = make_landmarkers()
    T, UV, P = [], [], []
    last_ms = -1
    show = args.show
    with Playback(db3) as pb:
        K, D = K_D(pb.intrinsics)
        for k, (stamp, img, depth) in enumerate(pb.rgbd_frames()):
            ts_ms = max(int(round((stamp - (T[0] if T else stamp)) * 1000)), last_ms + 1)   # 単調増加が必要
            last_ms = ts_ms
            uv = landmarks_uv(pose, hand, img, ts_ms)
            p = deproject(uv, depth_at(depth, uv), K, D)
            T.append(stamp)
            UV.append(uv)
            P.append(p)
            if k == n_total // 2:
                save_check_image(out / "rs_check.png", img, uv)
            if show:
                cv2.imshow("rs_joints (q: stop display)", draw_joints(
                    img, NAMES, uv, np.isfinite(p[:, 0]), lines=[f"{ts_ms / 1000:6.2f} s  frame {k}"]))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):    # 表示だけ止めて処理は続ける
                    show = False
                    cv2.destroyAllWindows()
            if k % 300 == 0:
                print(f"\r  {k} frames", end="", flush=True)
    pose.close()
    hand.close()
    cv2.destroyAllWindows()
    print()

    T, P = np.array(T), np.array(P)
    # t      : RealSense 時刻（epoch s）．sync_check.py の δ で Take 時刻に直す
    # names  : 関節名（config.RS_JOINTS の順）
    # P_C    : カラー光学座標 [mm]（frames × joints × 3）．検出・深度の失敗は NaN
    # uv     : 画素座標（frames × joints × 2）．2 次元で見つからなければ NaN
    np.savez(out / "rs_joints.npz", t=T, names=np.array(NAMES), P_C=P, uv=np.array(UV))
    print(f"[DONE] {len(T)} frames -> {out / 'rs_joints.npz'}")
    print_summary(trial_dir, T, P)


if __name__ == "__main__":
    main()
