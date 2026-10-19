"""
check_sync.py
────────────────────────────────────────────────────────────────────
課題（task.csv の epoch）と RealSense（rs_joints.npz の t）の時刻が合っているかを確かめる．

マウスの速度（生のカウントから）を，RealSense の手首（Hand の手首．遅れがほぼ 0）の速度の 1 次式で
表したときに最もよく当てはまる時間差を求める（カメラと机の向き・倍率の違いは 1 次式が吸収する）．同じ PC の時計なので，正しく記録できていれば
時間差は MediaPipe の遅れ程度（数十 ms 以内）になる．大きくずれていれば記録か時計に問題がある．

  python experiment/rs_joints.py recordings/aim_P01_xxx/trial_03_fast_s1   # 先に関節を推定
  python aim_task/check_sync.py recordings/aim_P01_xxx/trial_03_fast_s1      # 1 試行
  python aim_task/check_sync.py recordings/aim_P01_xxx                       # セッションの全試行（推定済みのもの）

出力: 試行の joints/sync_mouse.json（時間差・相関），joints/sync_mouse.png（速さの重ね描き）
  lag_ms > 0 = RealSense の手首がマウスより遅れて動いている
手首より先（手首の屈曲・指）の動きの分だけ当てはまりは 1 より下がる（相関はその目安にもなる）．
────────────────────────────────────────────────────────────────────
"""

import argparse
import json
from pathlib import Path

import numpy as np

FS = 200.0            # 比べるときのサンプリング [Hz]
MAX_LAG_S = 0.3       # 時間差を探す範囲 ±[s]
MAX_GAP_S = 0.1       # これより長い欠損は補間しない
SMOOTH_S = 0.1        # 位置と速さの移動平均の幅 [s]（中心をそろえるので時間差は変わらない）


def fill_gaps(t, y, max_gap):
    """NaN を直線補間する（max_gap より長い欠損は NaN のまま）．"""
    y = y.copy()
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return y
    yi = np.interp(t, t[ok], y[ok])
    idx = np.where(ok)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 1 and t[b] - t[a] > max_gap:
            yi[a + 1:b] = np.nan
    yi[:idx[0]] = np.nan
    yi[idx[-1] + 1:] = np.nan
    return yi


def smooth(y, n):
    if n <= 1:
        return y
    k = np.ones(n) / n
    return np.convolve(y, k, mode="same")


def velocity_on_grid(t, P, grid, max_gap):
    """位置（frames × d）→ grid 上の速度（grid × d）．欠損の区間は NaN．
    微分は雑音を強めるので，位置をならしてから ±SMOOTH_S/2 離れた 2 点の差で速度にする（中心がそろうので時間差は変わらない）．"""
    n = int(round(SMOOTH_S * FS))
    cols = [fill_gaps(t, P[:, j], max_gap) for j in range(P.shape[1])]
    Pg = np.stack([smooth(np.interp(grid, t, c, left=np.nan, right=np.nan), n) for c in cols], 1)
    h = max(n // 2, 1)
    V = np.full_like(Pg, np.nan)
    V[h:-h] = (Pg[2 * h:] - Pg[:-2 * h]) / (2 * h / FS)
    return V


def fit_r2(Vm, Vr):
    """マウスの速度（n × 2）を関節の速度（n × 3）の 1 次式で表したときの決定係数．
    カメラと机の向き・倍率の違い，関節とマウスの動きの向きの違いを吸収する．"""
    m = np.isfinite(Vm).all(1) & np.isfinite(Vr).all(1)
    if m.sum() < FS * 5:
        return np.nan
    A = np.c_[Vr[m], np.ones(m.sum())]
    B, *_ = np.linalg.lstsq(A, Vm[m], rcond=None)
    res = Vm[m] - A @ B
    return 1 - (res ** 2).sum() / ((Vm[m] - Vm[m].mean(0)) ** 2).sum()


def best_lag(Vm, Vr, max_lag):
    """Vr（関節）が Vm（マウス）より何秒遅れているか（正 = 関節が遅れ）と，その時の決定係数．"""
    L = int(round(max_lag * FS))
    ks = np.arange(-L, L + 1)
    r2 = np.array([fit_r2(Vm[max(0, -k):len(Vm) - max(0, k)], Vr[max(0, k):len(Vr) - max(0, -k)]) for k in ks])
    if not np.isfinite(r2).any():
        return np.nan, np.nan, ks / FS, r2
    i = int(np.nanargmax(r2))
    k = float(ks[i])
    if 0 < i < len(r2) - 1 and np.isfinite(r2[i - 1]) and np.isfinite(r2[i + 1]):   # 放物線補間
        den = r2[i - 1] - 2 * r2[i] + r2[i + 1]
        k += 0.5 * (r2[i - 1] - r2[i + 1]) / den if den != 0 else 0
    return k / FS, r2[i], ks / FS, r2


def check_trial(tdir, joint="wrist", plot=True):
    tdir = Path(tdir)
    npz = tdir / "joints" / "rs_joints.npz"
    if not npz.is_file():
        print(f"[SKIP] {tdir.name}: joints/rs_joints.npz がありません（先に experiment/rs_joints.py）")
        return None
    sess = json.loads((tdir.parent / "session.json").read_text(encoding="utf-8"))
    meta = json.loads((tdir / "meta.json").read_text(encoding="utf-8"))
    a = np.genfromtxt(tdir / "task.csv", delimiter=",", names=True)
    z = np.load(npz, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    if joint not in names:
        raise SystemExit(f"[ERROR] {joint} が rs_joints.npz にありません（{names}）")
    t_rs, P = z["t"], z["P_C"][:, names.index(joint), :]

    # マウス: 生のカウントを積算した位置 [mm]（カウントダウン中の動きも含む）
    mm = sess["cm_per_count"] * 10.0
    t_m = a["epoch"]
    M = np.stack([np.cumsum(a["mouse_dx_counts"]), np.cumsum(a["mouse_dy_counts"])], 1) * mm
    # 比べる区間: 課題の立ち上がり以降（ターゲットが動いている）かつ両方の記録がある範囲
    t0 = max(t_m[a["t"] >= -meta["task"].get("trajectory", {}).get("ramp_s", 0)][0], t_rs[0])
    t1 = min(t_m[-1], t_rs[-1])
    grid = np.arange(t0, t1, 1 / FS)
    Vm = velocity_on_grid(t_m, M, grid, max_gap=1.0)
    Vr = velocity_on_grid(t_rs, P, grid, max_gap=MAX_GAP_S)
    lag, r2, lags, curve = best_lag(Vm, Vr, MAX_LAG_S)
    r = float(np.sqrt(max(r2, 0))) if np.isfinite(r2) else np.nan
    v_m, v_r = np.linalg.norm(Vm, axis=1), np.linalg.norm(Vr, axis=1)
    ok = np.isfinite(v_r)
    ratio = float(np.nanmedian(v_r[ok & (v_m > 20)]) / np.nanmedian(v_m[ok & (v_m > 20)])) if (ok & (v_m > 20)).any() else np.nan
    res = {"trial": tdir.name, "joint": joint, "lag_ms": round(1000 * lag, 1), "r": round(r, 3),
           "valid_pct": round(float(100 * ok.mean()), 1), "speed_ratio_joint_over_mouse": round(ratio, 3),
           "duration_s": round(float(t1 - t0), 1),
           "note": "lag_ms > 0 = RealSense の関節がマウスより遅れて動いている"}
    (tdir / "joints").mkdir(exist_ok=True)
    (tdir / "joints" / "sync_mouse.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    flag = "" if abs(res["lag_ms"]) < 60 and r > 0.7 else "  ← 要確認"
    print(f"[SYNC] {tdir.name}: 時間差 {res['lag_ms']:+.0f} ms，相関 {r:.2f}，"
          f"関節の有効 {res['valid_pct']:.0f}%，速さの比 {ratio:.2f}{flag}")
    if plot:
        _plot(tdir, grid - grid[0], v_m, v_r, lag, lags, curve, res)
    return res


def _plot(tdir, tt, v_m, v_r, lag, lags, curve, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [2, 1]})
    w = (tt > tt[-1] / 2 - 5) & (tt < tt[-1] / 2 + 5)        # 中ほどの 10 s
    ax[0].plot(tt[w], v_m[w], color="#2a78d6", lw=1.6, label="mouse")
    ax[0].plot(tt[w] - lag, v_r[w], color="#eb6834", lw=1.6, label=f"RealSense {res['joint']} (shifted by {-1000 * lag:+.0f} ms)")
    ax[0].set_ylabel("speed [mm/s]")
    ax[0].set_xlabel("time [s]")
    ax[0].legend(frameon=False)
    ax[1].plot(1000 * lags, curve, color="#2a78d6", lw=1.6)
    ax[1].axvline(1000 * lag, color="0.5", lw=1, ls="--")
    ax[1].set_xlabel("lag of RealSense [ms]")
    ax[1].set_ylabel("R² (mouse velocity from joint velocity)")
    for a in ax:
        a.grid(True, color="0.9", lw=0.8)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{res['trial']}: lag {res['lag_ms']:+.0f} ms, r = {res['r']:.2f}")
    fig.tight_layout()
    fig.savefig(tdir / "joints" / "sync_mouse.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="課題と RealSense の時刻合わせの確認")
    ap.add_argument("path", help="試行のフォルダ（trial_*）またはセッションのフォルダ（aim_*）")
    ap.add_argument("--joint", default="wrist", help="比べる関節（config.RS_JOINTS の名前．既定 wrist = Hand の手首）")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    p = Path(args.path)
    if (p / "task.csv").is_file():
        check_trial(p, args.joint, not args.no_plot)
        return
    res = [r for d in sorted(p.glob("trial_*")) if (d / "task.csv").is_file()
           for r in [check_trial(d, args.joint, not args.no_plot)] if r]
    if res:
        lags = np.array([r["lag_ms"] for r in res])
        print(f"[SYNC] {len(res)} 試行: 時間差の中央値 {np.median(lags):+.0f} ms（範囲 {lags.min():+.0f}〜{lags.max():+.0f}）")


if __name__ == "__main__":
    main()
