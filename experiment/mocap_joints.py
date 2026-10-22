"""
mocap_joints.py
────────────────────────────────────────────────────────────────────
試行の Motive CSV から，比較の参照値にする関節位置と**関節角**（Motive 座標, mm / deg）を取り出す．

  python experiment/mocap_joints.py                         # 最新の trial_*
  python experiment/mocap_joints.py recordings/trial_XXXX
  python experiment/mocap_joints.py recordings/trial_XXXX --list   # CSV の列名を一覧表示するだけ

入力:  trial_*/mocap/*.csv（1 つだけ置く．Take 全体を Global 座標で書き出したもの）
出力:  trial_*/joints/mc_joints.npz
         t, start_epoch, fps
         names, P_M        … 関節位置（frames × joints × 3, mm）
         ang_names, ANG    … 関節角（frames × angles, deg）
         pron              … 前腕の回内外（frames,  deg）．ボーンの回転があるときだけ

  関節の定義（名前 → CSV の列名のリスト）は config.MC_JOINTS で決める．
    - 列名が 1 つ  : その列の位置（ボーン = 根元側の関節中心 / ラベル付きマーカー / ボーンマーカー）
    - 列名が複数   : その平均（例: 肘 = 内側・外側上顆マーカーの中点）
  最初に --list で実際の列名を確かめてから config.MC_JOINTS を書くこと．

関節角について
  **手首の角度を出すには，手首より遠位の点（手のマーカー）が要る．**
  config.MC_JOINTS に "mcp3" があればそれを使い，無ければ config.MC_HAND_FALLBACKS の
  名前を順に探す（RHandOut など）．見つからなければ手首の角度は NaN になる．

  角度の式は lib/angles.py の 1 本だけで，RealSense 側（aim_eval/a8_dof.py）と**完全に同じ**．
  点の取り方（MediaPipe の MCP3 と MoCap の手マーカー）が違うので絶対値は一致しないが，
  時間変化（SD・可動域・波形）は同じ定義なので比較できる．

  **前腕の回内外**は点が 3 つでは原理的に求まらないが，ボーンの回転が書き出されていれば
  前腕ボーンと手ボーンの相対回転から求められる（RealSense では測れない自由度）．
────────────────────────────────────────────────────────────────────
"""

import argparse
from collections import defaultdict

import numpy as np

import config
from lib import angles as AN
from lib import io_utils
from lib.mocap_io import Take

NAMES = list(config.MC_JOINTS)
# 手首の角度に要る「手首より遠位の点」の候補（config に無いときに順に探す）
HAND_FALLBACKS = getattr(config, "MC_HAND_FALLBACKS",
                         ["RHandOut", "RHand_end", "RHandIn", "RHandThumb", "RIndexFinger1"])
UP_MOTIVE = np.array([0.0, 1.0, 0.0])      # Motive は Y-up
ANG_NAMES = ["el_flex", "wr_flex", "wr_dev", "sh_elev", "sh_azim", "hum_rot"]


def list_columns(take):
    """Type ごとに列名を表示する（Unlabeled のマーカーは数だけ）．"""
    groups = defaultdict(list)
    for typ, name in take.names():
        groups[typ].append(name)
    print("― 位置を持つ列 ―")
    for typ, names in groups.items():
        labeled = [n for n in names if not n.startswith("Unlabeled")]
        print(f"[{typ}] {len(names)} 列"
              + (f"（うち Unlabeled {len(names) - len(labeled)}）" if len(labeled) < len(names) else ""))
        for n in labeled:
            print(f"    {n}")
    rot = take.rotation_names()
    print(f"\n― 回転を持つ列 ― （{take.rotation_type}"
          + (f" / {take.rotation_order}" if take.rotation_order else "") + "）")
    if not rot:
        print("    （なし）Motive の書き出しで Rotation にチェックを入れると回内外が出せます")
    else:
        groups = defaultdict(list)
        for typ, name in rot:
            groups[typ].append(name)
        for typ, names in groups.items():
            print(f"[{typ}] {len(names)} 列")
            for n in names:
                print(f"    {n}")
    print("\n手首の角度に使える「手首より遠位の点」の候補:")
    have = {n.split(":")[-1] for _, n in take.names()}
    cand = [c for c in HAND_FALLBACKS if c in have]
    print("    " + ("，".join(cand) if cand else
                    "見つかりません。手のマーカーを貼るか、config.MC_HAND_FALLBACKS に名前を足してください"))


def joint_positions(take, names, required=True):
    """config.MC_JOINTS に従って関節位置 (frames × joints × 3) を作る．"""
    P = np.full((len(take.t), len(names), 3), np.nan)
    missing = []
    for k, name in enumerate(names):
        cols = [take.position(c) for c in config.MC_JOINTS[name]]
        miss = [c for c, x in zip(config.MC_JOINTS[name], cols) if x is None]
        missing += miss
        if not miss:
            P[:, k] = np.mean(cols, axis=0)          # どれかが NaN ならそのフレームは NaN
    if missing and required:
        raise SystemExit(f"[ERROR] CSV に列がありません: {missing}\n"
                         f"        --list で列名を確かめて config.MC_JOINTS を直してください")
    return P, missing


def find_hand_point(take):
    """手首より遠位の点を探す．(名前, 位置) か (None, None)．"""
    if "mcp3" in config.MC_JOINTS:
        cols = [take.position(c) for c in config.MC_JOINTS["mcp3"]]
        if all(x is not None for x in cols):
            return "+".join(config.MC_JOINTS["mcp3"]), np.mean(cols, axis=0)
    for nm in HAND_FALLBACKS:
        X = take.position(nm)
        if X is not None:
            return nm, X
    return None, None


def pronation(take):
    """前腕ボーンと手ボーンの相対回転から回内外 [deg]．無ければ None．"""
    fa = getattr(config, "MC_FOREARM_BONE", "RFArm")
    hd = getattr(config, "MC_HAND_BONE", "RHand")
    axis = int(getattr(config, "MC_BONE_LONG_AXIS", 1))
    Mf = take.rotation_mat(fa)
    Mh = take.rotation_mat(hd)
    if Mf is None or Mh is None:
        return None
    return AN.pronation_from_mat(Mf, Mh, axis=axis)


def compute_angles(P, names, hand_xyz):
    """関節角 (frames × len(ANG_NAMES)) [deg]．手の点が無ければ手首の 2 列は NaN．"""
    need = ("shoulder", "elbow", "wrist")
    if any(n not in names for n in need):
        return None
    sh, el, wr = (P[:, names.index(n)] for n in need)
    A6 = AN.dof6(sh, el, wr, hand_xyz, up=UP_MOTIVE)       # [sh_elev, sh_azim, hum_rot, el_flex, wr_flex, wr_dev]
    return np.column_stack([A6[:, 3], A6[:, 4], A6[:, 5], A6[:, 0], A6[:, 1], A6[:, 2]])


def main():
    ap = argparse.ArgumentParser(description="Motive CSV → 関節位置と関節角")
    ap.add_argument("trial", nargs="?", help="trial_* フォルダ（省略すると最新）")
    ap.add_argument("--list", action="store_true", help="CSV の列名を表示するだけ")
    args = ap.parse_args()

    trial_dir = io_utils.resolve_session_dir(args.trial, config.TRIAL_PREFIX)
    take = Take(io_utils.find_one(trial_dir / "mocap", "*.csv"))
    print(f"[TRIAL] {trial_dir}\n[TAKE]  {take.path.name}  {len(take.t)} frames  {take.fps:.0f} fps")
    if args.list:
        list_columns(take)
        return

    # ── 位置（mcp3 は任意扱い：無くても止めない）
    base = [n for n in NAMES if n != "mcp3"]
    P_base, _ = joint_positions(take, base, required=True)
    hand_name, hand_xyz = find_hand_point(take)
    if hand_xyz is None:
        names = base
        P = P_base
        print("[WARN]  手首より遠位の点が見つかりません → 手首の角度は出せません（NaN）。")
        print("        --list で列名を確認し、config.MC_JOINTS['mcp3'] に書いてください。")
    else:
        names = base + ["mcp3"]
        P = np.concatenate([P_base, hand_xyz[:, None, :]], axis=1)
        print(f"[HAND]  手の点 = {hand_name}")

    ANG = compute_angles(P, names, hand_xyz)
    pron = pronation(take)
    if pron is None:
        print("[WARN]  ボーンの回転がありません → 前腕の回内外は出せません。")
        print("        Motive の書き出しで Rotation にチェックを入れると出せます。")
    else:
        print(f"[PRON]  回内外: 中央値 {np.nanmedian(pron):+.1f}°  "
              f"SD {np.nanstd(pron):.1f}°  可動域 "
              f"{np.nanpercentile(pron, 95) - np.nanpercentile(pron, 5):.1f}°")

    out = trial_dir / "joints"
    out.mkdir(exist_ok=True)
    # t           : Take 先頭からの時刻 [s]
    # start_epoch : Capture Start Time（epoch s）
    # P_M         : Motive 座標 [mm]（frames × joints × 3）．隠れたフレームは NaN
    # ANG         : 関節角 [deg]（frames × angles）．角度の定義は lib/angles.py
    np.savez(out / "mc_joints.npz", t=take.t, start_epoch=take.start_epoch, fps=take.fps,
             names=np.array(names), P_M=P,
             ang_names=np.array(ANG_NAMES),
             ANG=ANG if ANG is not None else np.zeros((len(take.t), 0)),
             pron=pron if pron is not None else np.full(len(take.t), np.nan),
             hand_point=np.array(hand_name if hand_name else ""))

    print("[VALID] " + "  ".join(f"{n}: {100 * np.isfinite(P[:, k, 0]).mean():.0f}%"
                                 for k, n in enumerate(names)))
    for a, b in config.SEGMENTS:
        if a in names and b in names:
            L = np.linalg.norm(P[:, names.index(a)] - P[:, names.index(b)], axis=1)
            L = L[np.isfinite(L)]
            if L.size:
                print(f"[LENGTH] {a}–{b}: 中央値 {np.median(L):.0f} mm  標準偏差 {L.std():.1f} mm")
    if ANG is not None:
        print("[ANGLE] " + "  ".join(
            f"{n}: {np.nanmedian(ANG[:, k]):+.1f}±{np.nanstd(ANG[:, k]):.1f}°"
            for k, n in enumerate(ANG_NAMES) if np.isfinite(ANG[:, k]).any()))
    print(f"[DONE] -> {out / 'mc_joints.npz'}")


if __name__ == "__main__":
    main()