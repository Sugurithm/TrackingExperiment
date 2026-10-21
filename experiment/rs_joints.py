"""
rs_joints.py
────────────────────────────────────────────────────────────────────
試行の .db3 から MediaPipe で左右の上肢（肩・肘・手首・手）を推定し，そのうち本当の右側を選んで，
位置合わせ済みの深度で 3 次元座標（カラー光学座標, mm）にする．左側も別に保存する．
（scripts/rs_estimate.py を試行フォルダ単位にし，比較に必要な形で保存するようにしたもの）

  python experiment/rs_joints.py                         # 最新の trial_*
  python experiment/rs_joints.py recordings/trial_XXXX
  python experiment/rs_joints.py recordings/trial_XXXX --show              # 処理中のフレームと推定・失敗の理由を画面に出す
  python experiment/rs_joints.py recordings/trial_XXXX --show --start 42   # 記録の 42 s から表示（そこで一時停止）
  python experiment/rs_joints.py recordings/trial_XXXX --save-video        # 同じ画面を joints/rs_diag.mp4 にも保存
  python experiment/rs_joints.py recordings/trial_XXXX --summary           # 保存済みの結果から有効率・失敗の内訳を出し直すだけ

  表示中のキー: スペース = 一時停止 / 再開，n = （一時停止中）1 フレーム進める，q・Esc = 表示だけ止める（処理は続く）
  --start より前のフレームも推定はする（MediaPipe の追跡を途中から始めると結果が変わるため）

【左右の決め方】
  Pose は両腕を，Hand は両手（最大 2 本）を毎フレーム推定する．そのうえで:
  1. Pose の左右: MediaPipe は人の向きを取り違えると左右のラベルを丸ごと入れ替えて出すことがある．
     両肩がはっきり見えていれば画像上の両肩の並びで本当の右を決める．並びの向きは
     config.RIGHT_SHOULDER_IMAGE_SIDE（"left" / "right" = 本当の右肩が画像の左 / 右に映る）．
     未設定なら，試行の中で MediaPipe のラベルの多数決から学習する．
     両肩の並びで決まらないフレームは，直前のフレームの右腕に近いほうを右とする．
  2. 手の左右: 検出した手を，左右の手首（1 で直した Pose）との距離の合計が最小になるように割り当てる．
     右手首に割り当てた手が右手．見つかった手が左手だけなら，右手は「なし」（左手を右手として使わない）．
     Pose がないフレームでは手の左右が決められないので，右手は使わない．

【確認画面の見方】
  左: カラー画像の周りに灰色の余白（画像の外）．関節が余白に描かれていれば画角外．
      右腕（白い線）と右手（水色），比較に使う右の関節は状態の色の丸．
      左腕・左手は青で描く（"L" 付き）．左右どちらにも割り当てなかった手は灰色．顔・腰・脚は描かない．
      （Pose は画像の外や隠れた所の点も体の形から「推測」して出す．visibility が低い関節は推測とみなす）
  右: 時刻・フレーム番号，Pose・手の検出の有無，左右の決め方，右の関節ごとの状態・visibility・深度・画素座標．
  下: 直近 10 秒の状態の帯（pose / 右手 / 左右の入れ替え / 右の関節ごと．右端が現在）．
  状態: OK        3 次元座標が取れた
        NO_POSE   Pose が人を見つけていない
        NO_HAND   右手が検出されていない（見つかった手が左手だけの場合を含む）
        REJECT    手は検出されたが右手として採用しなかった（右手首から遠い，Pose がない）
        OUT       2 次元座標が画像の外（画角外）
        NO_DEPTH  画像内だが深度が取れない（穴・範囲外）

入力:  trial_*/session.db3
       models/pose_landmarker_heavy.task, models/hand_landmarker.task
出力:  trial_*/joints/rs_joints.npz  … t（RealSense 時刻, epoch s），names，
                                        P_C（frames × joints × 3, mm, 取れなければ NaN），uv（画素）… 右
                                        status（frames × joints, 上の状態の番号），status_names，
                                        vis（Pose の visibility，手の関節は NaN），pose_found，hand_code
                                        P_C_left, uv_left, status_left, vis_left  … 同じ関節の左側
                                        lr_swapped（MediaPipe の左右のラベルを入れ替えたフレーム），
                                        lr_how（-1 = Pose なし，0 = ラベルのまま，1 = 両肩の並び，2 = 直前との連続性）
                     rs_check.png     … 記録の中ほどのフレームに関節を重ねた確認画像
                     rs_diag.mp4      … --save-video のとき．確認画面の動画
       画面: 関節ごとの有効率（全体 / 5 秒ごと / 動作区間），失敗の内訳と 0.3 s 以上続いた時間帯，
             左右の入れ替えの回数，上腕・前腕の長さのばらつき
             （動作区間 = 最初の同期区間より後でボードを振った区間を除く．sync_check.py の後なら出る）
       確認動画は joints_video.py で後から作る（MediaPipe をやり直さない）

  関節の対応（名前 → MediaPipe のランドマーク）は config.RS_JOINTS で決める（右側の番号で書く．
  左側は自動で鏡像の番号にする）．

【rs_estimate.py からの変更点】
  - 深度が取れないフレームは NaN のまま（直前の値で埋めない．誤差の評価で失敗を隠さないため）
  - 深度は 1 画素ではなく ±DEPTH_PATCH px の中央値（関節が腕の縁にあると背景を拾うため）
  - 両腕・両手を推定し，本当の右を選ぶ（左手・左腕を右として評価しないため．上の【左右の決め方】）
  - MediaPipe に渡す時刻は実際のタイムスタンプ（フレーム落ちがあっても追跡が乱れない）
────────────────────────────────────────────────────────────────────
"""

import argparse
import time
from collections import defaultdict

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

NAMES = list(config.RS_JOINTS)

# 関節の状態（status）．OK 以外は 3 次元座標が NaN のフレーム
STATUS = ["OK", "NO_POSE", "NO_HAND", "REJECT", "OUT", "NO_DEPTH"]
OK, NO_POSE, NO_HAND, REJECT, OUT, NO_DEPTH = range(len(STATUS))
STATUS_BGR = np.array([(90, 190, 90), (60, 60, 230), (40, 40, 150),
                       (0, 150, 255), (210, 70, 210), (0, 215, 230)], np.uint8)
EMPTY = len(STATUS)                      # 帯で「データなし」に使う番号
EMPTY_BGR = (45, 45, 45)
MIN_RUN_S = 0.3                          # 失敗の時間帯として表示する最短の長さ [s]

# Pose の左右の対応（左右を入れ替えるときの並べ替え）
MIRROR = np.arange(33)
for _a, _b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
               (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]:
    MIRROR[_a], MIRROR[_b] = _b, _a
R_ARM, L_ARM = [12, 14, 16], [11, 13, 15]          # 肩・肘・手首（右 / 左）
R_SHOULDER, L_SHOULDER, R_ELBOW, L_ELBOW, R_WRIST, L_WRIST = 12, 11, 14, 13, 16, 15
SIDE_VIS = 0.5                           # 両肩の並びで左右を決めるのに必要な visibility
SIDE_MIN_PX = 15.0                       # 両肩の左右の差がこれより小さければ並びでは決めない [px]
HOW = {None: -1, "label": 0, "shoulders": 1, "continuity": 2}


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
        running_mode=VisionTaskRunningMode.VIDEO, num_hands=2,          # 両手を推定してから右手を選ぶ
        min_hand_detection_confidence=config.HAND_CONFIDENCE, min_tracking_confidence=config.HAND_CONFIDENCE))
    return pose, hand


class SideChooser:
    """MediaPipe Pose の左右のラベルを入れ替えるべきかを決める（フレームをまたいで状態を持つ）．"""

    def __init__(self):
        side = getattr(config, "RIGHT_SHOULDER_IMAGE_SIDE", None)
        self.prior = {"right": 1.0, "left": -1.0}.get(side)   # 本当の右肩の u − 左肩の u の符号
        self.votes = 0                   # prior がないとき: MediaPipe のラベルでの符号の多数決
        self.prev = None                 # 直前のフレームの本当の右腕（肩・肘・手首の画素座標）

    def decide(self, pa):
        """pa: MediaPipe のラベルのままの 33 × (u, v, visibility)．戻り値 (入れ替えるか, 決め方)．"""
        du = pa[R_SHOULDER, 0] - pa[L_SHOULDER, 0]
        clear = bool(min(pa[R_SHOULDER, 2], pa[L_SHOULDER, 2]) >= SIDE_VIS and abs(du) >= SIDE_MIN_PX)
        sign = self.prior if self.prior is not None else (np.sign(self.votes) if self.votes else None)
        if clear and sign is not None:
            swap, how = bool(np.sign(du) != sign), "shoulders"
        elif self.prev is not None:
            d_r = np.mean(np.linalg.norm(pa[R_ARM, :2] - self.prev, axis=1))
            d_l = np.mean(np.linalg.norm(pa[L_ARM, :2] - self.prev, axis=1))
            swap, how = bool(d_l < d_r), "continuity"
        else:
            swap, how = False, "label"
        if clear and self.prior is None:
            self.votes += 1 if du > 0 else -1
        self.prev = pa[L_ARM if swap else R_ARM, :2].copy()
        return swap, how


def assign_hands(hands, pa):
    """検出した手（それぞれ 21×2）を，本当の左右に直した Pose の左右の手首に割り当てる．
    戻り値 (右手の番号 or -1, 左手の番号 or -1, 右手の理由)．
    理由 = ok / none（手なし）/ left_only（見つかった手は左手だけ）/
           far（右手首に割り当てた手が前腕の長さより遠い）/ no_pose（Pose がなく左右が決められない）．"""
    if not hands:
        return -1, -1, "none"
    if pa is None:
        return -1, -1, "no_pose"
    wr, wl = pa[R_WRIST, :2], pa[L_WRIST, :2]
    d = np.array([[np.linalg.norm(H[0] - wr), np.linalg.norm(H[0] - wl)] for H in hands])   # 手 × (右, 左)
    if len(hands) == 1:
        r, l = (0, -1) if d[0, 0] <= d[0, 1] else (-1, 0)
    else:
        r, l = (0, 1) if d[0, 0] + d[1, 1] <= d[1, 0] + d[0, 1] else (1, 0)
    forearm_r = np.linalg.norm(pa[R_WRIST, :2] - pa[R_ELBOW, :2])
    forearm_l = np.linalg.norm(pa[L_WRIST, :2] - pa[L_ELBOW, :2])
    if l >= 0 and d[l, 1] > forearm_l:
        l = -1
    if r < 0:
        return -1, l, "left_only"
    if d[r, 0] > forearm_r:
        return -1, l, "far"
    return r, l, "ok"


def side_joints(side, pa, hands, hand_i, hand_why):
    """左右どちらか（"R" / "L"）の関節の画素座標・visibility・取れなかった理由．"""
    uv = np.full((len(NAMES), 2), np.nan)
    vis = np.full(len(NAMES), np.nan)
    miss = np.full(len(NAMES), -1, np.int8)      # 2 次元で取れなかった理由（-1 = 取れた）
    for k, name in enumerate(NAMES):
        model, idx = config.RS_JOINTS[name]
        if model == "pose":
            if pa is None:
                miss[k] = NO_POSE
                continue
            j = idx if side == "R" else MIRROR[idx]
            uv[k], vis[k] = pa[j, :2], pa[j, 2]
        else:
            if hand_i < 0:
                miss[k] = (NO_HAND if hand_why in ("none", "left_only") else REJECT) if side == "R" else NO_HAND
                continue
            uv[k] = hands[hand_i][idx]
    return dict(uv=uv, vis=vis, miss=miss)


def detect(pose, hand, img, ts_ms, chooser=None, timer=None):
    """1 枚の画像を MediaPipe に通し，左右の関節の画素座標と確認用の情報を返す．
    chooser（SideChooser）はフレームをまたいで同じものを渡す．
    timer（dict）を渡すと Pose・Hand それぞれにかかった時間 [s] を足し込む．"""
    chooser = chooser or SideChooser()
    h, w = img.shape[:2]
    mp_img = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    t0 = time.perf_counter()
    pose_res = pose.detect_for_video(mp_img, ts_ms)
    t1 = time.perf_counter()
    hand_res = hand.detect_for_video(mp_img, ts_ms)
    if timer is not None:
        timer["pose"] += t1 - t0
        timer["hand"] += time.perf_counter() - t1

    pa, swap, how = None, False, None    # pa: 本当の左右に直した 33 × (u, v, visibility)
    if pose_res.pose_landmarks:
        raw = np.array([[lm.x * w, lm.y * h, np.nan if lm.visibility is None else lm.visibility]
                        for lm in pose_res.pose_landmarks[0]])
        swap, how = chooser.decide(raw)
        pa = raw[MIRROR] if swap else raw
    hands = [np.array([[lm.x * w, lm.y * h] for lm in lms]) for lms in hand_res.hand_landmarks]
    hand_r, hand_l, hand_why = assign_hands(hands, pa)
    return dict(R=side_joints("R", pa, hands, hand_r, hand_why),
                L=side_joints("L", pa, hands, hand_l, hand_why),
                pose_all=pa, swap=swap, how=how, hands=hands,
                hand_r=hand_r, hand_l=hand_l, hand_why=hand_why, size=(w, h))


def landmarks_uv(pose, hand, img, ts_ms):
    """1 枚の画像 → 右の関節ごとの画素座標 (joints × 2)．見つからない関節は NaN．（旧インターフェース）"""
    return detect(pose, hand, img, ts_ms)["R"]["uv"]


def joint_status(sd, size, p):
    """関節ごとの状態．3 次元座標が取れていれば OK，取れていなければ理由．sd = detect()["R"] など．"""
    w, h = size
    st = sd["miss"].copy()
    for k in range(len(st)):
        if np.isfinite(p[k, 0]):
            st[k] = OK
        elif st[k] < 0:
            u, v = sd["uv"][k]
            st[k] = NO_DEPTH if (0 <= u < w and 0 <= v < h) else OUT
    return st


def frame_codes(det, st):
    """下の帯に描く 1 フレーム分の状態（pose 行，右手の行，左右の入れ替えの行，右の関節…）．"""
    pose_code = OK if det["pose_all"] is not None else NO_POSE
    hand_code = {"ok": OK, "none": NO_HAND, "left_only": NO_HAND}.get(det["hand_why"], REJECT)
    swap_code = EMPTY if det["pose_all"] is None else (REJECT if det["swap"] else OK)
    return np.r_[pose_code, hand_code, swap_code, st].astype(np.int8)


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
#  確認画面
# ══════════════════════════════════════════════════
WIN = "rs_joints  [space] pause  [n] step  [q] stop display"
FONT = cv2.FONT_HERSHEY_SIMPLEX
PAD, INFO_W, ROW_H, LABEL_W = 80, 350, 16, 72      # 余白・右の情報欄・帯の行の高さ・帯のラベル幅 [px]
HISTORY_FRAMES = 600                                # 帯に出すフレーム数（約 10 s）
R_COL, L_COL = (230, 230, 230), (230, 150, 70)      # 右腕（白）・左腕（青）
R_HAND_COL, L_HAND_COL, OTHER_HAND_COL = (255, 220, 0), (230, 150, 70), (140, 140, 140)
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def _pt(xy, off=PAD):
    x, y = np.clip(np.asarray(xy, float) + off, -10000, 10000)
    return int(round(x)), int(round(y))


def _bgr(code):
    return tuple(int(c) for c in STATUS_BGR[code])


def _draw_hand(view, H, col):
    for a, b in HAND_EDGES:
        cv2.line(view, _pt(H[a]), _pt(H[b]), col, 1, cv2.LINE_AA)
    cv2.circle(view, _pt(H[0]), 4, col, -1, cv2.LINE_AA)


def draw_view(img, det, st):
    """カラー画像＋周りの余白に，左右の腕と手・比較に使う右の関節を描く．"""
    h, w = img.shape[:2]
    view = np.full((h + 2 * PAD, w + 2 * PAD, 3), 70, np.uint8)
    view[PAD:PAD + h, PAD:PAD + w] = img
    cv2.rectangle(view, (PAD - 1, PAD - 1), (PAD + w, PAD + h), (255, 255, 255), 1)
    cv2.putText(view, "outside the image (gray)", (6, 18), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    pa = det["pose_all"]
    if pa is not None:                   # 腕の骨格線（肩–肘–手首）．左は青
        for arm, col in ((L_ARM, L_COL), (R_ARM, R_COL)):
            for a, b in zip(arm[:-1], arm[1:]):
                cv2.line(view, _pt(pa[a, :2]), _pt(pa[b, :2]), col, 1, cv2.LINE_AA)
    for i, H in enumerate(det["hands"]):
        col = (R_HAND_COL if i == det["hand_r"] else L_HAND_COL if i == det["hand_l"] else OTHER_HAND_COL)
        _draw_hand(view, H, col)
    for k, name in enumerate(NAMES):     # 左の関節は小さな青い丸
        u, v = det["L"]["uv"][k]
        if np.isfinite(u):
            cv2.circle(view, _pt((u, v)), 6, L_COL, 1, cv2.LINE_AA)
            cv2.putText(view, "L " + name, _pt((u + 9, v + 14)), FONT, 0.4, L_COL, 1, cv2.LINE_AA)
    for k, name in enumerate(NAMES):     # 右の関節は状態の色の丸
        u, v = det["R"]["uv"][k]
        if np.isfinite(u):
            cv2.circle(view, _pt((u, v)), 10, _bgr(st[k]), 2, cv2.LINE_AA)
            cv2.putText(view, name, _pt((u + 13, v - 8)), FONT, 0.5, _bgr(st[k]), 1, cv2.LINE_AA)
    return view


def draw_info(height, det, st, z, t_rel, k):
    """右の情報欄．"""
    info = np.full((height, INFO_W, 3), 30, np.uint8)
    white, gray = (235, 235, 235), (170, 170, 170)
    pose_ok = det["pose_all"] is not None
    if not pose_ok:
        lr, lr_col = "L/R  : -", gray
    else:
        lr = f"L/R  : {'SWAPPED' if det['swap'] else 'as labeled'} ({det['how']})"
        lr_col = _bgr(REJECT) if det["swap"] else _bgr(OK)
    hand_col = _bgr({"ok": OK, "none": NO_HAND, "left_only": NO_HAND}.get(det["hand_why"], REJECT))
    lines = [(f"t {t_rel:8.2f} s   frame {k}", white),
             (f"pose : {'found' if pose_ok else 'NOT FOUND'}", _bgr(OK if pose_ok else NO_POSE)),
             (lr, lr_col),
             (f"hands: {len(det['hands'])}   right hand: {det['hand_why']}", hand_col),
             ("", white)]
    for j, name in enumerate(NAMES):
        u, v = det["R"]["uv"][j]
        lines.append((f"R {name:<10}{STATUS[st[j]]}", _bgr(st[j])))
        vis = "  -  " if not np.isfinite(det["R"]["vis"][j]) else f"{det['R']['vis'][j]:.2f}"
        zz = "   -   " if not np.isfinite(z[j]) else f"{z[j]:5.0f} mm"
        uvs = "-" if not np.isfinite(u) else f"({u:.0f}, {v:.0f})"
        lines.append((f"   vis {vis}  z {zz}  uv {uvs}", gray))
    y = 24
    for text, col in lines:
        cv2.putText(info, text, (10, y), FONT, 0.5, col, 1, cv2.LINE_AA)
        y += 21
    y = max(y + 10, height - 21 * len(STATUS) - 8)       # 凡例
    for c, s in enumerate(STATUS):
        if y > height - 6:
            break
        cv2.rectangle(info, (10, y - 11), (24, y + 1), _bgr(c), -1)
        cv2.putText(info, s, (32, y), FONT, 0.45, gray, 1, cv2.LINE_AA)
        y += 21
    return info


def draw_timeline(width, rows):
    """直近 HISTORY_FRAMES フレームの状態の帯（右端が現在）．L/R 行: 緑 = ラベルのまま，橙 = 入れ替え．"""
    labels = ["pose", "R hand", "L/R swap"] + ["R " + n for n in NAMES]
    n, tw = len(labels), width - LABEL_W
    tl = np.full((n * ROW_H + 22, width, 3), 30, np.uint8)
    codes = np.full((n, HISTORY_FRAMES), EMPTY, np.int16)
    recent = np.array(rows[-HISTORY_FRAMES:]).T
    codes[:, HISTORY_FRAMES - recent.shape[1]:] = recent
    lut = np.vstack([STATUS_BGR, np.array([EMPTY_BGR], np.uint8)])
    strip = cv2.resize(lut[codes], (tw, n * ROW_H), interpolation=cv2.INTER_NEAREST)
    tl[:n * ROW_H, LABEL_W:] = strip
    for i, lab in enumerate(labels):
        cv2.putText(tl, lab, (6, i * ROW_H + 12), FONT, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.line(tl, (0, (i + 1) * ROW_H - 1), (width, (i + 1) * ROW_H - 1), (30, 30, 30), 1)
    secs = HISTORY_FRAMES / 60.0
    for s in range(int(secs) + 1):                                     # 1 秒ごとの目盛り（60 fps 換算）
        x = LABEL_W + int(round(tw * (1 - s / secs))) - 1
        cv2.line(tl, (x, n * ROW_H), (x, n * ROW_H + 5), (200, 200, 200), 1)
        if s % 2 == 0:
            cv2.putText(tl, "now" if s == 0 else f"-{s}s", (max(x - 28, LABEL_W), n * ROW_H + 18),
                        FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return tl


def draw_frame(img, det, st, z, t_rel, k, rows):
    view = draw_view(img, det, st)
    top = np.hstack([view, draw_info(view.shape[0], det, st, z, t_rel, k)])
    return np.vstack([top, draw_timeline(top.shape[1], rows)])


def handle_keys(paused):
    """表示中のキー操作．戻り値 (表示を続けるか, 一時停止中か)．"""
    while True:
        key = cv2.waitKey(30 if paused else 1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            return False, False
        if key == ord(" "):
            paused = not paused
            if not paused:
                return True, False
        elif key == ord("n") and paused:
            return True, True                 # 1 フレームだけ進めて，また止まる
        if not paused:
            return True, False


def save_check_image(path, img, uv):
    vis = img.copy()
    for name, (u, v) in zip(NAMES, uv):
        if np.isfinite(u):
            cv2.circle(vis, (int(u), int(v)), 6, (0, 200, 255), 2)
            cv2.putText(vis, name, (int(u) + 8, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    cv2.imwrite(str(path), vis)


# ══════════════════════════════════════════════════
#  集計
# ══════════════════════════════════════════════════
def valid_line(ok):
    return "  ".join(f"{n}: {100 * ok[:, k].mean():3.0f}%" for k, n in enumerate(NAMES))


def runs_of(bad):
    """True が続く区間 [開始, 終了) のリスト．"""
    edges = np.diff(np.r_[0, bad.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def print_breakdown(t, status, use, has_sync):
    """失敗の理由の内訳と，失敗が続いた時間帯（記録の開始からの秒）．"""
    rel = t - t[0]
    head = "動作区間" if has_sync else "全体"
    print(f"[失敗の内訳 {head}]（全フレームに対する割合）")
    for k, name in enumerate(NAMES):
        s = status[use, k]
        parts = [f"{STATUS[c]} {100 * np.mean(s == c):.1f}%" for c in range(1, len(STATUS)) if np.any(s == c)]
        print(f"    {name:<10}" + ("  ".join(parts) if parts else "失敗なし"))
    print(f"[失敗が {MIN_RUN_S} s 以上続いた時間帯]（記録の秒．* = 動作区間にかかる）")
    for k, name in enumerate(NAMES):
        runs = [(a, b) for a, b in runs_of(status[:, k] != OK) if rel[b - 1] - rel[a] >= MIN_RUN_S]
        print(f"    {name}: {len(runs)} か所")
        for a, b in runs[:12]:
            why = STATUS[np.bincount(status[a:b, k], minlength=len(STATUS)).argmax()]
            mark = "*" if (has_sync and use[a:b].any()) else " "
            print(f"      {mark} {rel[a]:7.2f}–{rel[b - 1]:7.2f} s  ({rel[b - 1] - rel[a]:5.2f} s)  主な理由 {why}")
        if len(runs) > 12:
            print(f"        …ほか {len(runs) - 12} か所")


def print_lr(t, swapped, how):
    """MediaPipe の左右のラベルを入れ替えたフレームの数と時間帯．"""
    has_pose = how >= 0
    n = has_pose.sum()
    if not n:
        return
    by = "  ".join(f"{k} {100 * np.mean(how[has_pose] == v):.0f}%" for k, v in HOW.items() if v >= 0)
    print(f"[左右] Pose のあるフレームのうち 入れ替え {100 * swapped[has_pose].mean():.1f}%  （決め方: {by}）")
    rel = t - t[0]
    runs = runs_of(swapped)
    for a, b in runs[:8]:
        print(f"        入れ替え {rel[a]:7.2f}–{rel[b - 1]:7.2f} s")
    if len(runs) > 8:
        print(f"        …ほか {len(runs) - 8} か所")


def print_summary(trial_dir, t, P, status=None, P_left=None, swapped=None, how=None):
    """有効率（3 次元座標が取れたフレームの割合），失敗の内訳，左右の入れ替え，上腕・前腕の長さ．"""
    ok = np.isfinite(P[:, :, 0])
    print(f"[VALID 全体]     {valid_line(ok)}")
    if P_left is not None:
        print(f"[VALID 左 全体]  {valid_line(np.isfinite(P_left[:, :, 0]))}  （参考）")
    rel = t - t[0]
    print("[VALID 5 秒ごと] " + "  ".join(NAMES) + "（%）")
    for s0 in np.arange(0, rel[-1], 5.0):
        m = (rel >= s0) & (rel < s0 + 5)
        if m.any():
            print(f"    {s0:5.0f}-{s0 + 5:3.0f} s  " + "  ".join(f"{100 * ok[m, k].mean():5.0f}" for k in range(len(NAMES))))
    sync, _, _ = sync_io.load_sync(trial_dir)
    use = sync_io.operation_mask(sync, t) if sync else np.ones(len(t), bool)
    if sync:
        print(f"[VALID 動作区間] {valid_line(ok[use])}  （{use.sum()} frames）")
    else:
        print("[VALID 動作区間] sync_check.py の後に実行すると，同期区間を除いた有効率も出ます")
    if status is not None:
        print_breakdown(t, status, use, bool(sync))
    else:
        print("[失敗の内訳] この rs_joints.npz には status がありません（推定をやり直すと出ます）")
    if swapped is not None:
        print_lr(t, swapped, how)
    for a, b in config.SEGMENTS:                 # 上腕・前腕の長さは一定のはず．ばらつきが精度の目安
        if a in NAMES and b in NAMES:
            L = np.linalg.norm(P[use, NAMES.index(a)] - P[use, NAMES.index(b)], axis=1)
            L = L[np.isfinite(L)]
            if L.size:
                q1, q3 = np.percentile(L, [25, 75])
                print(f"[LENGTH] {a}–{b}: 中央値 {np.median(L):.0f} mm  四分位範囲 {q3 - q1:.1f} mm  （{L.size} frames）")


def print_timing(timer, n, total):
    """段階ごとの 1 フレームあたりの時間．どこを速くすればよいかの目安．"""
    labels = [("read", "読み込み・位置合わせ"), ("pose", "Pose"), ("hand", "Hand"),
              ("depth", "深度→3 次元"), ("show", "描画・表示・動画")]
    parts = "  ".join(f"{lab} {1000 * timer[key] / n:5.1f}" for key, lab in labels)
    print(f"[TIME] 1 フレームあたり [ms]: {parts}  （合計 {total / 60:.1f} 分，{n / total:.1f} fps）")


# ══════════════════════════════════════════════════
#  メイン
# ══════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=".db3 → MediaPipe → 関節の 3 次元座標")
    ap.add_argument("trial", nargs="?", help="trial_* フォルダ（省略すると最新）")
    ap.add_argument("--show", action="store_true", help="処理中のフレームと推定・失敗の理由を画面に出す")
    ap.add_argument("--start", type=float, default=0.0, metavar="SEC",
                    help="--show で記録の SEC 秒から表示する（そこで一時停止）")
    ap.add_argument("--save-video", action="store_true", help="確認画面を joints/rs_diag.mp4 に保存する")
    ap.add_argument("--summary", action="store_true", help="保存済みの rs_joints.npz から集計を出し直すだけ")
    ap.add_argument("--stride", type=int, default=getattr(config, "JOINT_STRIDE", 1), metavar="N",
                    help="N フレームに 1 枚だけ処理する（既定は config.JOINT_STRIDE）．"
                         "2 なら 60 fps の記録を 30 fps で処理し，時間が半分になる")
    args = ap.parse_args()

    trial_dir = io_utils.resolve_session_dir(args.trial, config.TRIAL_PREFIX)
    if args.summary:
        z = np.load(trial_dir / "joints" / "rs_joints.npz")
        get = lambda key: z[key] if key in z.files else None
        print(f"[TRIAL] {trial_dir}")
        print_summary(trial_dir, z["t"], z["P_C"], get("status"), get("P_C_left"), get("lr_swapped"), get("lr_how"))
        return
    db3 = trial_dir / config.DB3_NAME
    if not db3.is_file():
        raise SystemExit(f"[ERROR] {db3} がありません")
    n_total = io_utils.load_json(trial_dir / config.META_NAME).get("frames", 0)
    out = trial_dir / "joints"
    out.mkdir(exist_ok=True)
    side = getattr(config, "RIGHT_SHOULDER_IMAGE_SIDE", None)
    print(f"[TRIAL] {trial_dir}  関節: {', '.join(NAMES)}（左右とも推定し右を比較に使う）")
    print(f"[L/R] 本当の右肩が映る側: {side or '未設定（MediaPipe のラベルの多数決から学習）'}")

    stride = max(1, int(args.stride))
    saved_check = False
    eff_fps = (getattr(config, "FPS", 60) or 60) / stride
    if stride > 1:
        print(f"[STRIDE] {stride} フレームに 1 枚だけ処理します（実効 {eff_fps:.0f} fps）")
    pose, hand = make_landmarkers()
    chooser = SideChooser()
    keys = ("uv", "P", "st", "vis")
    R, L = {k: [] for k in keys}, {k: [] for k in keys}
    T, ROWS, SWAP, HOWS = [], [], [], []
    last_ms = -1
    show, paused, shown_once = args.show, False, False
    writer = None
    timer = defaultdict(float)            # 処理の段階ごとの合計時間 [s]（どこが遅いかを見るため）
    t_start = time.perf_counter()
    with Playback(db3) as pb:
        K, D = K_D(pb.intrinsics)
        t_mark = time.perf_counter()
        for k, (stamp, img, depth) in enumerate(pb.rgbd_frames()):
            now = time.perf_counter()
            timer["read"] += now - t_mark                              # .db3 の読み込み・深度の位置合わせ
            if stride > 1 and k % stride:                              # 間引き（MediaPipe に渡さない）
                t_mark = time.perf_counter()
                continue
            t_rel = stamp - (T[0] if T else stamp)
            ts_ms = max(int(round(t_rel * 1000)), last_ms + 1)        # 単調増加が必要
            last_ms = ts_ms
            det = detect(pose, hand, img, ts_ms, chooser, timer)
            t0 = time.perf_counter()
            res = {}
            for name_s, store in (("R", R), ("L", L)):
                sd = det[name_s]
                z = depth_at(depth, sd["uv"])
                p = deproject(sd["uv"], z, K, D)
                st = joint_status(sd, det["size"], p)
                for key, val in zip(keys, (sd["uv"], p, st, sd["vis"])):
                    store[key].append(val)
                res[name_s] = (z, st)
            timer["depth"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            z_r, st_r = res["R"]
            T.append(stamp)
            ROWS.append(frame_codes(det, st_r))
            SWAP.append(det["swap"])
            HOWS.append(HOW[det["how"]])
            if not saved_check and k >= n_total // 2:
                saved_check = True
                save_check_image(out / "rs_check.png", img, det["R"]["uv"])

            display = show and t_rel >= args.start
            if display or args.save_video:
                canvas = draw_frame(img, det, st_r, z_r, t_rel, k, ROWS)
                if args.save_video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(out / "rs_diag.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 float(eff_fps), (canvas.shape[1], canvas.shape[0]))
                    writer.write(canvas)
                if display:
                    cv2.imshow(WIN, canvas)
                    if not shown_once:
                        shown_once, paused = True, args.start > 0      # --start の位置で止めて待つ
                    show, paused = handle_keys(paused)
                    if not show:
                        cv2.destroyAllWindows()                         # 表示だけ止めて処理は続ける
            timer["show"] += time.perf_counter() - t0
            if k % 300 == 0:
                el = time.perf_counter() - t_start
                fps = (k + 1) / el
                eta = f"  残り約 {(n_total - k - 1) / fps / 60:4.1f} 分" if n_total > k else ""
                print(f"\r  {k} frames  {fps:5.1f} fps{eta}   ", end="", flush=True)
            t_mark = time.perf_counter()
    pose.close()
    hand.close()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print()

    T = np.array(T)
    ROWS = np.array(ROWS, np.int8)
    P, ST = np.array(R["P"]), np.array(R["st"], np.int8)
    SWAP, HOWS = np.array(SWAP, bool), np.array(HOWS, np.int8)
    # t           : RealSense 時刻（epoch s）．sync_check.py の δ で Take 時刻に直す
    # names       : 関節名（config.RS_JOINTS の順）
    # P_C         : 右．カラー光学座標 [mm]（frames × joints × 3）．検出・深度の失敗は NaN
    # uv          : 右．画素座標（frames × joints × 2）．2 次元で見つからなければ NaN（画像の外の値もありうる）
    # status      : 右．関節ごとの状態の番号（status_names の順．0 = OK）
    # vis         : 右．Pose の visibility（手の関節は NaN）
    # *_left      : 同じ関節の左側（参考．比較には使わない）
    # pose_found  : Pose が人を見つけたか
    # hand_code   : 右手の状態（0 = 採用，2 = 検出なし・左手だけ，3 = 検出したが不採用）
    # lr_swapped  : MediaPipe の左右のラベルを入れ替えたか
    # lr_how      : 左右の決め方（-1 = Pose なし，0 = ラベルのまま，1 = 両肩の並び，2 = 直前との連続性）
    # stride      : 何フレームに 1 枚処理したか（1 = 全フレーム）
    # pose_model  : 使った Pose モデルのファイル名（heavy / full / lite の区別）
    np.savez(out / "rs_joints.npz", t=T, names=np.array(NAMES),
             P_C=P, uv=np.array(R["uv"]), status=ST, vis=np.array(R["vis"]),
             P_C_left=np.array(L["P"]), uv_left=np.array(L["uv"]),
             status_left=np.array(L["st"], np.int8), vis_left=np.array(L["vis"]),
             status_names=np.array(STATUS), pose_found=ROWS[:, 0] == OK, hand_code=ROWS[:, 1],
             lr_swapped=SWAP, lr_how=HOWS,
             stride=np.int32(stride), pose_model=np.array(str(config.POSE_MODEL.name)))
    print(f"[DONE] {len(T)} frames -> {out / 'rs_joints.npz'}")
    print_timing(timer, len(T), time.perf_counter() - t_start)
    if args.save_video:
        print(f"[VIDEO] {out / 'rs_diag.mp4'}")
    print_summary(trial_dir, T, P, ST, np.array(L["P"]), SWAP, HOWS)


if __name__ == "__main__":
    main()