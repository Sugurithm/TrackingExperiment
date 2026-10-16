"""
calib_solve.py
────────────────────────────────────────────────────────────────────
T_CB（calib_rs.py）と T_MB（calib_mocap.py）を結合して T_MC（カメラ → Motive, mm）を求め，
画像に重ねて目視確認する．（scripts/merge_transfer.py を較正フォルダ単位にしたもの）

  python experiment/calib_solve.py                       # 最新の calib_*
  python experiment/calib_solve.py recordings/calib_XXXX

入力:  calib_*/board/pose_*/T_CB.json, frame.png
       calib_*/mocap/T_MB.json
出力:  calib_*/T_MC.json        … 本番の比較で使う変換行列
       calib_*/overlay_all.png  … MoCap の点を RGB 画像に投影（赤: 生マーカー，シアン: チェッカー交点）
       calib_*/overlay_err.csv  … 投影した交点と RealSense の検出交点の距離 [px]

【姿勢の対応づけ】
  RealSense と Motive は時計がずれているので時刻は使わない．撮影順に並べ，
  姿勢間の相対回転が両者で一致する組み合わせを選ぶ（静止区間が RealSense の姿勢数より
  多い場合も，余分な区間を自動で外せる）．
【求め方】
  全姿勢の S1〜S3 を積み上げた Kabsch（閉形式）．各姿勢を 1 つ抜いて求めた T_MC で
  その姿勢を予測した誤差（LOO）も出す．LOO が全体の残差と同程度なら健全．
────────────────────────────────────────────────────────────────────
"""

import sys
from itertools import combinations

import numpy as np
import pandas as pd
import cv2

import config
from lib import io_utils
from lib.board import MARKERS_B, OBJ
from lib.geometry import apply, kabsch, project, rot_angle_deg

ROT_WARN_DEG = 2.0           # 相対回転の不一致がこれを超えたら対応づけを疑う
SCALE_WARN   = 0.02          # 姿勢間移動距離の比が 1 ± これを外れたら SQUARE / PRINT_SCALE を疑う


# ══════════════════════════════════════════════════
#  読み込み
# ══════════════════════════════════════════════════
def load_inputs(calib_dir):
    rs_poses = []
    for d in sorted((calib_dir / "board").glob("pose_*")):
        if not (d / "T_CB.json").is_file():      # calib_rs.py で NG だった姿勢は使わない
            print(f"[WARN] {d.name}: T_CB.json がないので除外します")
            continue
        cb = io_utils.load_json(d / "T_CB.json")
        rs_poses.append(dict(name=d.name, dir=d, cb=cb, T_CB=np.array(cb["T_CB"])))

    mb_path = calib_dir / "mocap" / "T_MB.json"
    if not mb_path.is_file():                    # 実行順の間違い（calib_mocap.py が先）
        raise SystemExit(f"[ERROR] {mb_path} がありません．先に calib_mocap.py を実行してください")
    mc_poses = [dict(T_MB=np.array(p["T_MB"]), markers_M=np.array(p["markers_M"]), t=(p["t_start"], p["t_end"]))
                for p in io_utils.load_json(mb_path)["poses"]]
    return rs_poses, mc_poses


# ══════════════════════════════════════════════════
#  姿勢の対応づけ
# ══════════════════════════════════════════════════
def rel_rot_mismatch(T_CBs, T_MBs):
    """姿勢 i→j のボードの回転は，カメラから見ても Motive から見ても同じはず．その不一致 [deg] の一覧．"""
    out = []
    for i, j in combinations(range(len(T_CBs)), 2):
        Rc = T_CBs[i][:3, :3].T @ T_CBs[j][:3, :3]
        Rm = T_MBs[i][:3, :3].T @ T_MBs[j][:3, :3]
        out.append(rot_angle_deg(Rc.T @ Rm))
    return np.array(out) if out else np.zeros(1)


def match_poses(rs_poses, mc_poses):
    """撮影順を保ったまま，相対回転の不一致が最小になる静止区間の組を選ぶ．"""
    n, m = len(rs_poses), len(mc_poses)
    if n < 3:                                    # 3 姿勢未満では回転の対応も T_MC も不安定
        raise SystemExit(f"[ERROR] 使える RealSense の姿勢が {n} 個しかありません（3 個以上必要）")
    if m < n:                                    # MoCap 側の区間が足りない = 閾値か CSV の取り違え
        raise SystemExit(f"[ERROR] MoCap の静止区間 {m} 個 < RealSense の姿勢 {n} 個．"
                         f"mocap/pose_check.png と config.py の MC_* を確認してください")
    T_CBs = [p["T_CB"] for p in rs_poses]
    best = min(combinations(range(m), n),        # 区間数が同じなら候補は撮影順の 1 通りだけ
               key=lambda idx: rel_rot_mismatch(T_CBs, [mc_poses[k]["T_MB"] for k in idx]).mean())
    mis = rel_rot_mismatch(T_CBs, [mc_poses[k]["T_MB"] for k in best])
    print("[MATCH] " + "  ".join(f"{p['name']}->MoCap[{k}]" for p, k in zip(rs_poses, best)))
    print(f"        相対回転の不一致 mean {mis.mean():.2f}°  max {mis.max():.2f}°")
    if mis.max() > ROT_WARN_DEG:
        print("[WARN] 相対回転が一致しません．姿勢の撮り漏れ・撮り直しで順番がずれていないか確認してください")
    return [dict(**p, mocap_index=k, **mc_poses[k]) for p, k in zip(rs_poses, best)]


def check_scale(pairs):
    """姿勢間のボード移動距離の比（カメラ / MoCap）．1 からずれていれば SQUARE が実寸と違う．"""
    r = [np.linalg.norm(a["T_CB"][:3, 3] - b["T_CB"][:3, 3]) / np.linalg.norm(a["T_MB"][:3, 3] - b["T_MB"][:3, 3])
         for a, b in combinations(pairs, 2)
         if np.linalg.norm(a["T_MB"][:3, 3] - b["T_MB"][:3, 3]) > 20]      # 動きが小さい組は比が不安定なので除く
    if not r:
        return None
    med = float(np.median(r))
    print(f"[SCALE] 姿勢間移動距離の比 Cam/MoCap = {med:.3f}")
    if abs(med - 1) > SCALE_WARN:
        print("[WARN] 比が 1 から離れています．config.py の PRINT_SCALE（SQUARE）を実測値にしてください")
    return med


# ══════════════════════════════════════════════════
#  T_MC
# ══════════════════════════════════════════════════
def solve_T_MC(pairs):
    """全姿勢の S1〜S3 を積み上げた Kabsch: T_MB·p ≈ T_MC·T_CB·p"""
    return kabsch(np.vstack([apply(p["T_CB"], MARKERS_B) for p in pairs]),
                  np.vstack([apply(p["T_MB"], MARKERS_B) for p in pairs]))


def residual_mm(T_MC, p):
    return np.linalg.norm(apply(T_MC @ p["T_CB"], MARKERS_B) - apply(p["T_MB"], MARKERS_B), axis=1)


# ══════════════════════════════════════════════════
#  重ね合わせ（LOO の T_MC を使い自己採点にしない）
# ══════════════════════════════════════════════════
def overlay(p, T_MC_loo):
    img = cv2.imread(str(p["dir"] / "frame.png"))
    if img is None:                              # frame.png がない（旧形式の T_CB.json など）
        return None, np.nan, np.nan
    K, D = np.array(p["cb"]["K"]), np.array(p["cb"]["D"])
    T_CM = np.linalg.inv(T_MC_loo)
    q_mk = project(apply(T_CM, p["markers_M"]), K, D)                 # MoCap の生マーカー
    q_bd = project(apply(T_CM @ p["T_MB"], OBJ), K, D)                # MoCap から見たチェッカー交点
    e = np.linalg.norm(q_bd - np.array(p["cb"]["frame_corners"]), axis=1)

    for u in q_bd:
        cv2.circle(img, tuple(np.round(u).astype(int)), 3, (255, 255, 0), 1)
    for u in q_mk:
        cv2.circle(img, tuple(np.round(u).astype(int)), 7, (0, 0, 255), 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, f"{p['name']} -> MoCap[{p['mocap_index']}]  corner err mean {e.mean():.1f} max {e.max():.1f} px",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img, float(e.mean()), float(e.max())


def tile(images, cols=3):
    images = images + [np.zeros_like(images[0])] * (-len(images) % cols)
    return np.vstack([np.hstack(images[i:i + cols]) for i in range(0, len(images), cols)])


# ══════════════════════════════════════════════════
#  メイン
# ══════════════════════════════════════════════════
def main():
    calib_dir = io_utils.resolve_session_dir(sys.argv[1] if len(sys.argv) > 1 else None, config.CALIB_PREFIX)
    print(f"[CALIB] {calib_dir}")
    rs_poses, mc_poses = load_inputs(calib_dir)
    pairs = match_poses(rs_poses, mc_poses)
    scale = check_scale(pairs)

    T_MC = solve_T_MC(pairs)
    res = np.concatenate([residual_mm(T_MC, p) for p in pairs])
    rms = float(np.sqrt(np.mean(res ** 2)))

    rows, tiles = [], []
    for p in pairs:
        T_loo = solve_T_MC([q for q in pairs if q is not p])
        loo = float(np.sqrt(np.mean(residual_mm(T_loo, p) ** 2)))
        img, e_mean, e_max = overlay(p, T_loo)
        if img is not None:
            tiles.append(img)
        rows.append(dict(pose=p["name"], mocap_index=p["mocap_index"],
                         resid_mm=float(np.sqrt(np.mean(residual_mm(T_MC, p) ** 2))), loo_mm=loo,
                         corner_err_mean_px=e_mean, corner_err_max_px=e_max))
        print(f"  {p['name']}: resid {rows[-1]['resid_mm']:.2f} mm  LOO {loo:.2f} mm  corner {e_mean:.2f}/{e_max:.2f} px")

    cam = T_MC[:3, 3]
    print(f"[T_MC] 残差 RMS {rms:.2f} mm  max {res.max():.2f} mm   カメラ位置(Motive) {cam.round(1)} mm")

    # T_MC : カラー光学座標 → Motive 座標 [mm]（本番の比較: p_M = T_MC · p_C，p_C は mm に直してから掛ける）
    # T_CM : その逆（MoCap の点を画像に投影するとき使う）
    io_utils.save_json(calib_dir / "T_MC.json", dict(
        calib=calib_dir.name, created=io_utils.now_str(), method="stacked_kabsch",
        T_MC=T_MC.tolist(), T_CM=np.linalg.inv(T_MC).tolist(), camera_position_M=cam.tolist(),
        rms_mm=rms, max_mm=float(res.max()), scale_cam_over_mocap=scale,
        square_mm=config.SQUARE, print_scale=config.PRINT_SCALE,
        color_intrinsics=pairs[0]["cb"].get("intrinsics"), poses=rows))
    pd.DataFrame(rows).to_csv(calib_dir / "overlay_err.csv", index=False)
    if tiles:
        cv2.imwrite(str(calib_dir / "overlay_all.png"), tile(tiles))
    print(f"[DONE] -> {calib_dir / 'T_MC.json'}")


if __name__ == "__main__":
    main()
