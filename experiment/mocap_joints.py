"""
mocap_joints.py
────────────────────────────────────────────────────────────────────
試行の Motive CSV から，比較の参照値にする関節位置（Motive 座標, mm）を取り出す．

  python experiment/mocap_joints.py                         # 最新の trial_*
  python experiment/mocap_joints.py recordings/trial_XXXX
  python experiment/mocap_joints.py recordings/trial_XXXX --list   # CSV の列名を一覧表示するだけ

入力:  trial_*/mocap/*.csv（1 つだけ置く．Take 全体を Global 座標・mm で書き出したもの）
出力:  trial_*/joints/mc_joints.npz  … t（Take 時刻 s），start_epoch，names，P_M（frames × joints × 3, mm）

  関節の定義（名前 → CSV の列名のリスト）は config.MC_JOINTS で決める．
    - 列名が 1 つ  : その列の位置（ラベル付きマーカー，またはスケルトンのボーン = 根元側の関節中心）
    - 列名が複数   : その平均（例: 肘 = 内側・外側上顆マーカーの中点）．どれか 1 つでも隠れたフレームは NaN
  最初に --list で実際の列名を確かめてから config.MC_JOINTS を書くこと．
────────────────────────────────────────────────────────────────────
"""

import argparse
from collections import defaultdict

import numpy as np

import config
from lib import io_utils
from lib.mocap_io import Take

NAMES = list(config.MC_JOINTS)


def list_columns(take):
    """Type ごとに列名を表示する（Unlabeled のマーカーは数だけ）．"""
    groups = defaultdict(list)
    for typ, name in take.names():
        groups[typ].append(name)
    for typ, names in groups.items():
        labeled = [n for n in names if not n.startswith("Unlabeled")]
        print(f"[{typ}] {len(names)} 列" + (f"（うち Unlabeled {len(names) - len(labeled)}）" if len(labeled) < len(names) else ""))
        for n in labeled:
            print(f"    {n}")


def joint_positions(take):
    """config.MC_JOINTS に従って関節位置 (frames × joints × 3) を作る．見つからない列名があれば止める．"""
    P = np.full((len(take.t), len(NAMES), 3), np.nan)
    missing = []
    for k, name in enumerate(NAMES):
        cols = [take.position(c) for c in config.MC_JOINTS[name]]
        missing += [c for c, x in zip(config.MC_JOINTS[name], cols) if x is None]
        if all(x is not None for x in cols):
            P[:, k] = np.mean(cols, axis=0)          # どれかが NaN ならそのフレームは NaN
    if missing:
        raise SystemExit(f"[ERROR] CSV に列がありません: {missing}\n"
                         f"        --list で列名を確かめて config.MC_JOINTS を直してください")
    return P


def main():
    ap = argparse.ArgumentParser(description="Motive CSV → 関節位置")
    ap.add_argument("trial", nargs="?", help="trial_* フォルダ（省略すると最新）")
    ap.add_argument("--list", action="store_true", help="CSV の列名を表示するだけ")
    args = ap.parse_args()

    trial_dir = io_utils.resolve_session_dir(args.trial, config.TRIAL_PREFIX)
    take = Take(io_utils.find_one(trial_dir / "mocap", "*.csv"))
    print(f"[TRIAL] {trial_dir}\n[TAKE]  {take.path.name}  {len(take.t)} frames  {take.fps:.0f} fps")
    if args.list:
        list_columns(take)
        return

    P = joint_positions(take)
    out = trial_dir / "joints"
    out.mkdir(exist_ok=True)
    # t           : Take 先頭からの時刻 [s]
    # start_epoch : Capture Start Time（epoch s）
    # P_M         : Motive 座標 [mm]（frames × joints × 3）．隠れたフレームは NaN
    np.savez(out / "mc_joints.npz", t=take.t, start_epoch=take.start_epoch, fps=take.fps,
             names=np.array(NAMES), P_M=P)
    print("[VALID] " + "  ".join(f"{n}: {100 * np.isfinite(P[:, k, 0]).mean():.0f}%" for k, n in enumerate(NAMES)))
    for a, b in config.SEGMENTS:
        if a in NAMES and b in NAMES:
            L = np.linalg.norm(P[:, NAMES.index(a)] - P[:, NAMES.index(b)], axis=1)
            L = L[np.isfinite(L)]
            if L.size:
                print(f"[LENGTH] {a}–{b}: 中央値 {np.median(L):.0f} mm  標準偏差 {L.std():.1f} mm")
    print(f"[DONE] -> {out / 'mc_joints.npz'}")


if __name__ == "__main__":
    main()
