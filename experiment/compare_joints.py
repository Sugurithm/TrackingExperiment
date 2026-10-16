"""
compare_joints.py
────────────────────────────────────────────────────────────────────
RealSense + MediaPipe の関節位置を，MoCap の関節位置（参照値）と比べる．

  python experiment/compare_joints.py                       # 最新の trial_*
  python experiment/compare_joints.py recordings/trial_XXXX

入力:  trial_*/joints/rs_joints.npz      （rs_joints.py）
       trial_*/joints/mc_joints.npz      （mocap_joints.py）
       trial_*/sync_check/sync_result.json（sync_check.py．時刻ずれ c と，使った較正）
       較正の T_MC.json
出力:  trial_*/compare/summary.json  … 関節ごとの誤差・MediaPipe の遅れ・上腕/前腕の長さ
                      frames.csv    … フレームごとの誤差（カメラ座標の x, y, z と，視線方向・垂直方向）
                      compare.png   … 軌跡の重ね描き・誤差の時系列・速さと誤差

【時刻の合わせ方】
  Take 時刻 = RealSense 時刻 + c．c は sync_result.json の区間ごとの値を，区間が 2 つ以上なら
  時間で直線補間する（記録中の時計のずれも吸収）．
  比較は動作区間だけ: 最初の同期区間より後（録画開始直後の体が写っていない時間を除く）で，
  ボードを振った区間 ± CMP_SYNC_MARGIN を除く．有効率 valid_pct もこの区間が分母．
【誤差の分け方】（sync_check.py の結果から）
  - 偏り（平均のずれ）とばらつき: 皮膚上のマーカーと関節中心の違いで偏りは必ず出る．
    角度・速度の解析に効くのはばらつきの方
  - 視線方向（奥行き）と視線に垂直な方向: 深度センサの誤差は奥行き方向に出やすく性質が違う
  - 速さ別: ブレ・ローリングシャッターの影響は速いときにだけ出る
【MediaPipe の遅れ】
  δ はボードで決めているので，MediaPipe に時間方向の遅れがあれば関節の軌跡のずれとして残る．
  補正はせず，ずらして一番合う時間を「遅れ」として報告する（正 = RealSense が遅れている）．
────────────────────────────────────────────────────────────────────
"""

import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from lib import io_utils, sync_io
from lib.geometry import apply, interp_nan, split_ray, velocity

LAG_STEP = 0.005          # 遅れを探す刻み [s]
LAG_MIN_MOTION = 20.0     # 動きの大きさ（位置の標準偏差）がこれ未満の関節は遅れを求めない [mm]


# ══════════════════════════════════════════════════
#  読み込み
# ══════════════════════════════════════════════════
def load_inputs(trial_dir):
    paths = dict(rs=trial_dir / "joints" / "rs_joints.npz", mc=trial_dir / "joints" / "mc_joints.npz",
                 sync=trial_dir / "sync_check" / "sync_result.json")
    for key, script in (("rs", "rs_joints.py"), ("mc", "mocap_joints.py"), ("sync", "sync_check.py")):
        if not paths[key].is_file():             # 実行順の間違い
            raise SystemExit(f"[ERROR] {paths[key]} がありません．先に {script} を実行してください")
    rs = dict(np.load(paths["rs"]))
    mc = dict(np.load(paths["mc"]))
    sync, calib, T_MC = sync_io.load_sync(trial_dir)
    return rs, mc, sync, calib, T_MC


# ══════════════════════════════════════════════════
#  1 関節の比較
# ══════════════════════════════════════════════════
def compare_pair(t, take_t, P_C, mc_t, Q_M, T_MC, use):
    """RealSense の関節 P_C（カメラ座標）と MoCap の関節 Q_M（Motive 座標）を比べる．"""
    T_CM = np.linalg.inv(T_MC)
    Q_C = apply(T_CM, interp_nan(take_t, mc_t, Q_M, config.CMP_MAX_GAP))
    ok = use & np.isfinite(P_C[:, 0]) & np.isfinite(Q_C[:, 0])
    d = P_C - Q_C                                          # カメラ座標の差（x 右, y 下, z 奥）
    ray, lat = split_ray(P_C, Q_C)
    spd = np.linalg.norm(interp_nan(take_t, mc_t, velocity(mc_t, Q_M), config.CMP_MAX_GAP), axis=1)   # MoCap の速さ

    frames = pd.DataFrame(dict(t_rs=t, take_time=take_t, speed_mm_s=spd, ex=d[:, 0], ey=d[:, 1], ez=d[:, 2],
                               e_lateral=np.linalg.norm(lat, axis=1), e_ray=ray,
                               e_3d=np.linalg.norm(d, axis=1)))[ok]
    if len(frames) < 10:
        return None, frames
    bias = frames[["ex", "ey", "ez"]].mean().to_numpy()
    resid = frames[["ex", "ey", "ez"]].to_numpy() - bias
    stats = dict(
        n=len(frames), valid_pct=float(100 * ok.sum() / max(use.sum(), 1)),
        bias_mm=bias.tolist(), sd_mm=resid.std(axis=0).tolist(),
        rms_3d_mm=float(np.sqrt(np.mean(frames.e_3d ** 2))), median_3d_mm=float(frames.e_3d.median()),
        rms_without_bias_mm=float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1)))),
        lateral_median_mm=float(frames.e_lateral.median()),
        ray_mean_mm=float(frames.e_ray.mean()), ray_sd_mm=float(frames.e_ray.std()),
        by_speed=[dict(lo=lo, hi=hi, n=int(m.sum()),
                       median_3d_mm=float(frames.e_3d[m].median()) if m.any() else None,
                       lateral_median_mm=float(frames.e_lateral[m].median()) if m.any() else None)
                  for lo, hi in zip(config.CMP_SPEED_BINS[:-1], config.CMP_SPEED_BINS[1:])
                  for m in [(frames.speed_mm_s >= lo) & (frames.speed_mm_s < hi)]],
        motion_mm=float(np.linalg.norm(np.nanstd(Q_C[ok], axis=0))),
        lag_ms=None, rms_without_bias_lag_mm=None)
    if stats["motion_mm"] >= LAG_MIN_MOTION:        # ほとんど動かない関節（肩など）では遅れが決まらない
        stats["lag_ms"] = estimate_lag(take_t, P_C, mc_t, Q_M, T_CM, ok)
    if stats["lag_ms"] is not None:
        # 参考: 遅れの分だけ MoCap をずらした後の，偏りを除いた RMS（空間の誤差だけ．補正した結果ではない）
        Q_L = apply(T_CM, interp_nan(take_t - stats["lag_ms"] / 1000, mc_t, Q_M, config.CMP_MAX_GAP))
        m = ok & np.isfinite(Q_L[:, 0])
        r = (P_C[m] - Q_L[m]) - (P_C[m] - Q_L[m]).mean(0)
        stats["rms_without_bias_lag_mm"] = float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))
    return stats, frames


def estimate_lag(take_t, P_C, mc_t, Q_M, T_CM, ok):
    """MoCap をずらして，偏りを除いた視線に垂直な差が最小になる時間 [ms]（正 = RealSense が遅れている）．"""
    lags = np.arange(-config.CMP_MAX_LAG, config.CMP_MAX_LAG + LAG_STEP / 2, LAG_STEP)
    cost = []
    for L in lags:
        Q_C = apply(T_CM, interp_nan(take_t + L, mc_t, Q_M, config.CMP_MAX_GAP))
        m = ok & np.isfinite(Q_C[:, 0])
        bias = (P_C[m] - Q_C[m]).mean(0)
        _, lat = split_ray(P_C[m] - bias, Q_C[m])                     # 偏りを除いた差の，視線に垂直な成分
        cost.append(np.sqrt(np.mean(np.sum(lat ** 2, axis=1))) if m.sum() > 10 else np.nan)
    k = int(np.nanargmin(cost))
    if k in (0, len(lags) - 1):                  # 探索範囲の端 = 遅れが決まらない（動きが小さい・雑音が大きい）
        return None
    return float(-lags[k] * 1000)


def segment_lengths(rs, mc, use):
    """上腕・前腕の長さ（RealSense / MoCap）．一定のはずなので，ばらつきが精度の目安になる．"""
    out = []
    rs_names, mc_names = list(rs["names"]), list(mc["names"])
    for a, b in config.SEGMENTS:
        row = dict(segment=f"{a}-{b}")
        for key, names, P in (("rs", rs_names, rs["P_C"][use]), ("mc", mc_names, mc["P_M"])):
            if a in names and b in names:
                L = np.linalg.norm(P[:, names.index(a)] - P[:, names.index(b)], axis=1)
                L = L[np.isfinite(L)]
                row[key] = dict(median_mm=float(np.median(L)), sd_mm=float(L.std())) if L.size else None
        out.append(row)
    return out


# ══════════════════════════════════════════════════
#  表示・保存
# ══════════════════════════════════════════════════
def report(results, lengths):
    print("\n関節（RS / MoCap）     n   3次元RMS  偏りを除くRMS  垂直(中央値)  奥行き 平均±SD    遅れ   遅れも除くRMS")
    for (a, b), s in results.items():
        if s is None:
            print(f"  {a:>10s} / {b:<8s}  比較できるフレームがありません")
            continue
        print(f"  {a:>10s} / {b:<8s} {s['n']:5d}  {s['rms_3d_mm']:7.1f}   {s['rms_without_bias_mm']:9.1f}"
              f"    {s['lateral_median_mm']:9.1f}    {s['ray_mean_mm']:+6.1f} ± {s['ray_sd_mm']:4.1f}  "
              + (f"{s['lag_ms']:+5.0f} ms   {s['rms_without_bias_lag_mm']:7.1f}" if s["lag_ms"] is not None else "決まらず"))
    print("  偏り（カメラ座標 x 右, y 下, z 奥）:")
    for (a, b), s in results.items():
        if s is not None:
            print(f"    {a:>10s}: {np.round(s['bias_mm'], 1)} mm   ばらつき {np.round(s['sd_mm'], 1)} mm")
    for row in lengths:
        rs_, mc_ = row.get("rs"), row.get("mc")
        if rs_ and mc_:
            print(f"[LENGTH] {row['segment']}: RealSense {rs_['median_mm']:.0f} ± {rs_['sd_mm']:.1f} mm  "
                  f"MoCap {mc_['median_mm']:.0f} ± {mc_['sd_mm']:.1f} mm")


def save_plot(path, frames_all, series):
    pairs = list(series)
    fig, ax = plt.subplots(len(pairs), 3, figsize=(18, 3.6 * len(pairs)), squeeze=False)
    for k, key in enumerate(pairs):
        tt, P_M, Q_M = series[key]
        f = frames_all[frames_all.joint == f"{key[0]}/{key[1]}"]
        for j, col in enumerate("rgb"):
            ax[k, 0].plot(tt, Q_M[:, j], color=col, lw=0.8)
            ax[k, 0].plot(tt, P_M[:, j], ".", color=col, ms=1.5)
        ax[k, 0].set(title=f"{key[0]} / {key[1]}  (line: MoCap, dots: RealSense)", xlabel="Take time [s]", ylabel="X/Y/Z [mm]")
        ax[k, 1].plot(f.take_time, f.e_3d, ".", ms=1.5, label="3D")
        ax[k, 1].plot(f.take_time, f.e_ray, ".", ms=1.5, label="depth (ray)")
        ax[k, 1].plot(f.take_time, f.e_lateral, ".", ms=1.5, label="lateral")
        ax[k, 1].axhline(0, color="k", lw=0.5)
        ax[k, 1].set(xlabel="Take time [s]", ylabel="error [mm]")
        ax[k, 1].legend(fontsize=8, markerscale=5)
        ax[k, 2].plot(f.speed_mm_s, f.e_lateral, ".", ms=1.5, label="lateral")
        ax[k, 2].plot(f.speed_mm_s, f.e_ray - f.e_ray.mean(), ".", ms=1.5, label="depth - mean")
        ax[k, 2].set(xlabel="speed (MoCap) [mm/s]", ylabel="error [mm]")
        ax[k, 2].legend(fontsize=8, markerscale=5)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ══════════════════════════════════════════════════
#  メイン
# ══════════════════════════════════════════════════
def main():
    trial_dir = io_utils.resolve_session_dir(sys.argv[1] if len(sys.argv) > 1 else None, config.TRIAL_PREFIX)
    rs, mc, sync, calib, T_MC = load_inputs(trial_dir)
    print(f"[TRIAL] {trial_dir}\n[CALIB] {calib.name}  同期区間 {len(sync['segments'])} 個")

    t = rs["t"]
    take_t = t + sync_io.take_offset(sync, t)
    use = sync_io.operation_mask(sync, t)
    print(f"  比較に使う RealSense フレーム {use.sum()} / {len(t)}（動作区間: 最初の同期より後，ボードを振った区間を除く）")

    rs_names, mc_names = list(rs["names"]), list(mc["names"])
    results, frames, series = {}, [], {}
    for a, b in config.COMPARE_PAIRS:
        if a not in rs_names or b not in mc_names:
            print(f"[SKIP] {a} / {b}: どちらかの関節がありません（config.RS_JOINTS / MC_JOINTS を確認）")
            continue
        P_C, Q_M = rs["P_C"][:, rs_names.index(a)], mc["P_M"][:, mc_names.index(b)]
        stats, f = compare_pair(t, take_t, P_C, mc["t"], Q_M, T_MC, use)
        results[(a, b)] = stats
        frames.append(f.assign(joint=f"{a}/{b}"))
        series[(a, b)] = (take_t[use], apply(T_MC, P_C[use]), interp_nan(take_t[use], mc["t"], Q_M, config.CMP_MAX_GAP))

    lengths = segment_lengths(rs, mc, use)
    report(results, lengths)

    out = trial_dir / "compare"
    out.mkdir(exist_ok=True)
    frames_all = pd.concat(frames)
    io_utils.save_json(out / "summary.json", dict(
        trial=trial_dir.name, calib=calib.name, n_rs_frames=int(len(t)), n_used=int(use.sum()),
        note="誤差はカメラ座標（x 右, y 下, z 奥）．e_ray: 視線方向（正 = RealSense が遠い）．lag_ms: 正 = RealSense が遅れている",
        joints={f"{a}/{b}": s for (a, b), s in results.items()}, segments=lengths))
    frames_all.to_csv(out / "frames.csv", index=False, float_format="%.3f")
    save_plot(out / "compare.png", frames_all, series)
    print(f"[DONE] -> {out}")


if __name__ == "__main__":
    main()