"""
sync_check.py
────────────────────────────────────────────────────────────────────
試行の冒頭（と末尾）でボードを振った区間から，RealSense と Motive の時刻ずれ δ を求める．

  python experiment/sync_check.py                          # 最新の trial_*
  python experiment/sync_check.py recordings/trial_XXXX
  python experiment/sync_check.py recordings/trial_XXXX --redetect   # ボード検出をやり直す
  python experiment/sync_check.py recordings/trial_XXXX --no-tmc     # T_MC を使わない（較正の確認にならない）
  python experiment/sync_check.py recordings/trial_XXXX --calib calib_20261015_1914   # 較正を指定する

    Motive 時刻 [epoch s] = RealSense 時刻 [epoch s] + δ
    Take 時刻 [s]         = RealSense 時刻 + c         （c = δ − Capture Start Time）

入力:  trial_*/session.db3（なければ rgb/ + frame_timestamps.csv）
       trial_*/mocap/*.csv（1 つだけ置く．Take 全体を Global 座標で書き出したもの）
       較正の T_MC.json（--calib で指定．なければ meta.json の "calib"，それもなければ試行より前の最新の calib_*）
出力:  trial_*/sync_check/rs_board.npz    … ボード検出結果（2 回目以降は再利用）
                         sync_result.json … 区間ごとの δ・c・残差（compare_joints.py が読む）
                         sync_frames.csv  … フレームごとの誤差
                         sync_check.png   … 相関・誤差曲線・軌跡・奥行きの差

【手順】
  1. RealSense: 各フレームでボードを検出 → S1〜S3 のカメラ座標
  2. Motive   : ラベルなしマーカーから S1〜S3 をマーカー間距離で識別
  3. 検出が途切れる所で区間に分ける（開始付近と終了付近で 2 区間になる想定）
  4. 区間ごとに
       粗い探索  : ボード中心の「速さ」の相関が最大になる c（回転に依らないので T_MC 不要）
       細かい探索: S1〜S3 の「視線に垂直な方向」の差が最小になる c（1 ms 刻み + 放物線補間）
  5. 相関のピークが最もはっきりした区間を基準にし，ほかの区間はその近くで探し直す
     （同じような動きを 2 回すると，別の区間の動きと取り違えることがあるため）

【視線に垂直な方向だけで δ を決める理由】
  ボード 1 枚の PnP では奥行きは「見かけの大きさ」から決まるので弱い（1 px の誤差が 1.3 m 先で約 2 cm）．
  さらにブレやローリングシャッターで系統的にずれる．時刻のずれは画像上の動きに正確に出るので，
  δ はそちらで決め，奥行き方向の差は別に報告する．
────────────────────────────────────────────────────────────────────
"""

import argparse

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from lib import io_utils
from lib.board import MARKERS_B, identify_all, solve_board, flip_pose
from lib.geometry import apply, interp_nan, kabsch, split_ray
from lib.mocap_io import Take
from lib.rs_device import open_color

# 解析の内部パラメータ（通常は変えない）
COARSE_STRIDE  = 6        # ボードの写っている時間帯を探すときの間引き（検出は 1 枚 0.1 s かかる）
CONTRAST_MIN   = 0.3      # 白黒の差が典型値のこの割合未満なら，向きは隣のフレームから引き継ぐ
MAX_LINK       = 0.1      # この時間以内の隣り合う検出は向きがつながっているとみなす [s]
JUMP_MM        = 15.0     # 前後の中央値からこれ以上飛んだフレームは誤検出として捨てる [mm]
MIN_SEG_FRAMES = 30       # 区間の最少フレーム数
MIN_PATH_MM    = 100.0    # 区間内のボードの移動距離がこれ未満なら同期に使わない [mm]
GRID_HZ        = 120.0    # 速さを比べる共通の時間軸 [Hz]
SMOOTH_S       = 0.05     # 速さの平滑化の窓 [s]
FINE_RANGE     = 0.15     # 細かい探索の範囲 ±[s]
FINE_STEP      = 0.001    # 細かい探索の刻み [s]
NEXT_SEARCH    = 2.0      # 2 つ目以降の区間は，基準区間の c ±この範囲で探す [s]
KEEP           = 0.9      # 細かい探索は誤差の小さい方から 90% のフレームで評価する
OUTLIER_MM     = 30.0     # これ以上の誤差のフレームを「外れ」として数える [mm]


# ══════════════════════════════════════════════════
#  RealSense: ボード検出
# ══════════════════════════════════════════════════
def detect_board(trial_dir, cache, redetect):
    """全フレームでボードを検出する．1 回目は間引いて写っている時間帯を探し，2 回目にその周りを全部調べる．"""
    if cache.is_file() and not redetect:
        z = np.load(cache)
        if "contrast" in z:
            print(f"[RS] 検出結果を再利用: {cache}（やり直すときは --redetect）")
            return {k: z[k] for k in ("t", "R", "tvec", "err", "contrast")}
        print("[RS] 古い形式の検出結果なので検出し直します")

    frames = open_color(trial_dir)
    hits = {}

    def detect(k, stamp, img, K, D):
        res = solve_board(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), K, D,
                          max_err=config.SYNC_MAX_ERR_PX, exhaustive=False)
        if res:
            hits[k] = (stamp, res["R"], res["t"], res["err"], res["contrast"])

    n = 0
    for k, (stamp, img, K, D) in enumerate(frames()):
        n += 1
        if k % COARSE_STRIDE == 0:
            detect(k, stamp, img, K, D)
    print(f"  [1/2] {n} frames，ボードあり {len(hits)}（{COARSE_STRIDE} 枚おき）")
    near = np.array([v[0] for v in hits.values()])
    for k, (stamp, img, K, D) in enumerate(frames()):
        if k not in hits and len(near) and np.min(np.abs(near - stamp)) <= COARSE_STRIDE / config.FPS + 0.5:
            detect(k, stamp, img, K, D)
    print(f"  [2/2] ボードあり {len(hits)} frames")

    order = sorted(hits)
    out = {key: np.array([hits[k][i] for k in order]) for i, key in enumerate(("t", "R", "tvec", "err", "contrast"))}
    if len(order) == 0:
        raise SystemExit("[ERROR] ボードが 1 枚も検出できません（露出・ボードの向き・画角を確認）")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def fix_unsure(rs):
    """白黒の差が小さい（向きが怪しい）フレームは，近くの確実なフレームと x 軸がそろうように向きを決める．
    近くに確実なフレームがなければ捨てる．"""
    sure = rs["contrast"] >= CONTRAST_MIN * np.median(rs["contrast"])
    keep = sure.copy()
    for k in np.where(~sure)[0]:
        cand = np.where(sure & (np.abs(rs["t"] - rs["t"][k]) <= MAX_LINK))[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmin(np.abs(rs["t"][cand] - rs["t"][k]))]
        if rs["R"][j][:, 0] @ rs["R"][k][:, 0] < 0:          # x 軸が逆向き = 180° 取り違え
            rs["R"][k], rs["tvec"][k] = flip_pose(rs["R"][k], rs["tvec"][k])
        keep[k] = True
    print(f"[RS] 向き: 白黒の差が小さく隣から決めたフレーム {int((~sure).sum())}，決まらず除外 {int((~keep).sum())}")
    return {k: v[keep] for k, v in rs.items()}


def markers_cam(rs, idx):
    """S1〜S3 のカメラ座標 (n, 3, 3)．"""
    return np.einsum("nij,kj->nki", rs["R"][idx], MARKERS_B) + rs["tvec"][idx][:, None]


def reject_jumps(P, win):
    """重心が前後 win フレームの中央値から JUMP_MM 以上離れたフレームを True で返す（1 フレームの誤識別・誤検出）．"""
    c = P.mean(1)
    med = pd.DataFrame(c).rolling(win, center=True, min_periods=3).median().to_numpy()
    return np.linalg.norm(c - med, axis=1) > JUMP_MM


def split_segments(rs):
    """検出が SYNC_SEG_GAP 以上途切れる所で区間に分け，短い区間・ほとんど動いていない区間は除く．"""
    t = rs["t"]
    edges = np.r_[0, np.where(np.diff(t) > config.SYNC_SEG_GAP)[0] + 1, len(t)]
    segs = []
    for a, b in zip(edges[:-1], edges[1:]):
        idx = np.arange(a, b)
        path = np.linalg.norm(np.diff(markers_cam(rs, idx).mean(1), axis=0), axis=1).sum()
        if len(idx) >= MIN_SEG_FRAMES and path >= MIN_PATH_MM:
            segs.append(idx)
        else:
            print(f"[skip] 記録の {t[a] - t[0]:.1f} s 付近の区間（{len(idx)} frames，移動 {path:.0f} mm）")
    return segs


# ══════════════════════════════════════════════════
#  MoCap
# ══════════════════════════════════════════════════
def load_mocap(trial_dir, rs_t, window_s):
    """Take から，RealSense の記録時間 ± window_s の範囲の S1〜S3 を取り出す．"""
    take = Take(io_utils.find_one(trial_dir / "mocap", "*.csv"))
    sel = (take.t >= rs_t[0] - take.start_epoch - window_s) & (take.t <= rs_t[-1] - take.start_epoch + window_s)
    if not sel.any():                               # 2 台の時計が大きくずれている
        sel[:] = True
    S = identify_all(take.markers()[sel])
    bad = reject_jumps(S, 9)
    S[bad] = np.nan
    print(f"[MoCap] {take.path.name}  {sel.sum()} frames，S1〜S3 識別 {np.isfinite(S[:, 0, 0]).sum()}（飛びで除外 {bad.sum()}）")
    return dict(t=take.t[sel], S=S, fps=take.fps, start=take.start_epoch, meta=take.meta)


# ══════════════════════════════════════════════════
#  δ の推定
# ══════════════════════════════════════════════════
def speed(t_grid, P):
    v = np.linalg.norm(np.gradient(P, t_grid, axis=0), axis=1)
    return pd.Series(v).rolling(max(1, int(SMOOTH_S * GRID_HZ)), center=True, min_periods=1).mean().to_numpy()


def trimmed_rms(e):
    e = np.sort(e[np.isfinite(e)])
    return float(np.sqrt(np.mean(e[:max(1, int(np.ceil(KEEP * len(e))))] ** 2)))


def lateral_err(P_C, Q_M, T_MC):
    """フレームごとの誤差: 視線に垂直な成分 [mm]，視線方向の成分 [mm]（正 = RS が遠い）．3 点の平均．"""
    along, lat = split_ray(P_C, apply(np.linalg.inv(T_MC), Q_M))
    return np.linalg.norm(lat, axis=2).mean(1), along.mean(1)


def fit_T(P_C, Q_M):
    """T_MC を使わないとき: 誤差の大きいフレームを除きながら剛体変換を当てはめる．"""
    keep = np.ones(len(P_C), bool)
    for _ in range(3):
        T = kabsch(P_C[keep].reshape(-1, 3), Q_M[keep].reshape(-1, 3))
        e = np.linalg.norm(apply(T, P_C) - Q_M, axis=2).mean(1)
        keep = e <= np.quantile(e, KEEP)
    return T


def sync_segment(rs, idx, mc, T_MC, c_range):
    """1 区間について c（Take 時刻 = RS 時刻 + c）を求める．"""
    P_C = markers_cam(rs, idx)
    keep = ~reject_jumps(P_C, 5)
    idx, P_C = idx[keep], P_C[keep]
    t = rs["t"][idx]
    gap = 2.5 / mc["fps"]

    # ── 粗い探索: 速さの相関 ──
    g = np.arange(t[0], t[-1], 1 / GRID_HZ)
    v_rs = speed(g, interp_nan(g, t, P_C.mean(1), 0.1))
    gm = np.arange(mc["t"][0], mc["t"][-1], 1 / GRID_HZ)
    v_mc = speed(gm, interp_nan(gm, mc["t"], mc["S"].mean(1), 0.05))
    lags = np.arange(c_range[0], c_range[1], 1 / GRID_HZ)
    corr = np.full(len(lags), np.nan)
    for k, c in enumerate(lags):
        q = np.interp(g + c, gm, v_mc, left=np.nan, right=np.nan)
        m = np.isfinite(v_rs) & np.isfinite(q)
        if m.sum() > max(20, 0.8 * np.isfinite(v_rs).sum()) and q[m].std() > 0:
            corr[k] = np.corrcoef(v_rs[m], q[m])[0, 1]
    if np.all(np.isnan(corr)):
        return None
    kb = int(np.nanargmax(corr))
    c0 = lags[kb]
    far = np.abs(lags - c0) > 0.5
    second = float(np.nanmax(corr[far])) if np.any(far & np.isfinite(corr)) else np.nan

    # ── 細かい探索: 視線に垂直な差 ──
    cs = np.arange(c0 - FINE_RANGE, c0 + FINE_RANGE + FINE_STEP / 2, FINE_STEP)
    cost = np.full(len(cs), np.nan)
    for k, c in enumerate(cs):
        Q = interp_nan(t + c, mc["t"], mc["S"], gap)
        m = np.isfinite(Q[:, 0, 0])
        if m.sum() >= 10:
            T = T_MC if T_MC is not None else fit_T(P_C[m], Q[m])
            cost[k] = trimmed_rms(lateral_err(P_C[m], Q[m], T)[0])
    kf = int(np.nanargmin(cost))
    c1 = cs[kf]
    if 0 < kf < len(cs) - 1 and np.all(np.isfinite(cost[kf - 1:kf + 2])):
        y0, y1, y2 = cost[kf - 1:kf + 2] ** 2         # 二乗誤差に放物線を当てる
        if y0 - 2 * y1 + y2 > 0:
            c1 += 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * FINE_STEP

    # ── 最良の c での残差 ──
    Q = interp_nan(t + c1, mc["t"], mc["S"], gap)
    m = np.isfinite(Q[:, 0, 0])
    T = T_MC if T_MC is not None else fit_T(P_C[m], Q[m])
    e_lat, e_ray = lateral_err(P_C, Q, T)
    e_3d = np.linalg.norm(apply(T, P_C) - Q, axis=2).mean(1)
    depth = np.linalg.norm(P_C.mean(1), axis=1)
    v = np.gradient(interp_nan(gm, mc["t"], mc["S"].mean(1), 0.05), gm, axis=0)
    spd = np.linalg.norm(interp_nan(t + c1, gm, v, 0.05), axis=1)
    ok = m & np.isfinite(e_ray) & (np.abs(e_ray) < 100)
    slope = float(np.polyfit(depth[ok], e_ray[ok], 1)[0] * 1000) if ok.sum() > 20 and np.ptp(depth[ok]) > 100 else np.nan

    frames = pd.DataFrame(dict(t_rs=t, take_time=t + c1, speed_mm_s=spd, depth_mm=depth,
                               err_lateral_mm=e_lat, err_ray_mm=e_ray, err_3d_mm=e_3d,
                               contrast=rs["contrast"][idx], reproj_px=rs["err"][idx]))
    e = e_lat[m]
    return dict(c_coarse=float(c0), c=float(c1), corr=float(corr[kb]), corr_second=second,
                median_mm=float(np.median(e)), trimmed_rms_mm=trimmed_rms(e), p95_mm=float(np.percentile(e, 95)),
                outlier_frac=float(np.mean(e >= OUTLIER_MM)),
                ray_mean_mm=float(np.mean(e_ray[ok])), ray_sd_mm=float(np.std(e_ray[ok])), ray_slope_mm_per_m=slope,
                median_3d_mm=float(np.median(e_3d[m])), depth_range_mm=[float(depth.min()), float(depth.max())],
                n_rs=int(len(idx)), n_used=int(m.sum()), at_edge=bool(kf in (0, len(cs) - 1)),
                rs_t0=float(t[0]), rs_t1=float(t[-1]), T_fit=None if T_MC is not None else T.tolist(),
                _frames=frames, _plot=dict(lags=lags, corr=corr, cs=cs, cost=cost, t=t, P_M=apply(T, P_C), Q=Q))


def sync_all(rs, segs, mc, T_MC):
    """各区間を広く探し，相関のピークが最もはっきりした区間を基準に，他の区間はその近くで探し直す．"""
    c_prior = -mc["start"]                          # 2 台の時計が合っていれば c ≈ −Capture Start
    first = {}
    for k, idx in enumerate(segs):
        r = sync_segment(rs, idx, mc, T_MC, (c_prior - config.SYNC_SEARCH, c_prior + config.SYNC_SEARCH))
        if r is not None:
            first[k] = r
    if not first:
        raise SystemExit("[ERROR] MoCap と重なる区間がありません（CSV の取り違え・Take の範囲を確認）")

    anchor = max(first, key=lambda k: first[k]["corr"] - np.nan_to_num(first[k]["corr_second"]))
    results = []
    for k in sorted(first):
        r = first[k]
        if k != anchor:
            c_a = first[anchor]["c"]
            r2 = sync_segment(rs, segs[k], mc, T_MC, (c_a - NEXT_SEARCH, c_a + NEXT_SEARCH))
            if r2 is not None and abs(r2["c"] - r["c"]) > 0.05:
                print(f"[info] 区間 {len(results)} は単独だと {r['c'] - c_a:+.2f} s ずれた所が最大でした"
                      "（似た動きとの取り違え）．基準区間の近くで探し直しました")
            r = r2 or r
        r["anchor"] = k == anchor
        r["delta"] = r["c"] + mc["start"]
        results.append(r)
    return results


# ══════════════════════════════════════════════════
#  表示・保存
# ══════════════════════════════════════════════════
def report(results, t0):
    print("\n========== 結果 ==========")
    print("  Motive 時刻 = RealSense 時刻 + δ")
    for k, r in enumerate(results):
        print(f"[区間 {k}] 記録の {r['rs_t0'] - t0:5.1f}–{r['rs_t1'] - t0:5.1f} s  δ = {r['delta']:+.4f} s"
              f"  （Take の {r['rs_t0'] + r['c']:.2f} s〜）{'  ← 基準' if r['anchor'] else ''}")
        print(f"          速さの相関 {r['corr']:.3f}（次点 {r['corr_second']:.3f}）  使用 {r['n_used']}/{r['n_rs']} frames")
        print(f"          視線に垂直な差: 中央値 {r['median_mm']:.1f} mm  95% {r['p95_mm']:.1f} mm  "
              f"{OUTLIER_MM:.0f} mm 以上 {100 * r['outlier_frac']:.1f}%")
        print(f"          奥行き方向の差: 平均 {r['ray_mean_mm']:+.1f} mm  ばらつき {r['ray_sd_mm']:.1f} mm  "
              f"距離 1 m あたり {r['ray_slope_mm_per_m']:+.1f} mm（δ には使っていない）")
        if r["anchor"] and r["corr"] - np.nan_to_num(r["corr_second"]) < 0.15:
            print("          [WARN] 相関のピークがはっきりしません．ボードをもっと不規則に大きく動かしてください")
        if r["at_edge"]:
            print("          [WARN] 細かい探索の端で最小になりました．FINE_RANGE を広げてください")
    if len(results) >= 2:
        drift = (results[-1]["c"] - results[0]["c"]) * 1000
        print(f"\n  最初と最後の区間の差: {drift:+.1f} ms  ← 1 フレーム（{1000 / config.FPS:.1f} ms）以内なら OK")


def save_plot(path, results, start):
    fig, ax = plt.subplots(len(results), 3, figsize=(17, 4 * len(results)), squeeze=False)
    for k, r in enumerate(results):
        p = r["_plot"]
        ax[k, 0].plot(p["lags"] + start, p["corr"], lw=0.8)
        ax[k, 0].axvline(r["c_coarse"] + start, color="r", lw=0.8)
        ax[k, 0].set(xlabel="δ [s]", ylabel="speed correlation", title=f"segment {k}: coarse")
        ax[k, 1].plot((p["cs"] - r["c"]) * 1000, p["cost"], lw=1)
        ax[k, 1].axvline(0, color="r", lw=0.8)
        ax[k, 1].set(xlabel="offset from best [ms]", ylabel="lateral error [mm]",
                     title=f"fine: min {r['trimmed_rms_mm']:.1f} mm")
        tt = p["t"] + r["c"]
        Q = p["Q"].mean(axis=1)                         # S1〜S3 がそろったフレームだけ（欠けたら NaN）
        for j, col in enumerate("rgb"):
            ax[k, 2].plot(tt, Q[:, j], color=col, lw=0.8)
            ax[k, 2].plot(tt, p["P_M"].mean(1)[:, j], ".", color=col, ms=2)
        ax[k, 2].set(xlabel="Take time [s]", ylabel="board X/Y/Z [mm]", title="after sync (line: MoCap, dots: RealSense)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ══════════════════════════════════════════════════
#  メイン
# ══════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="ボードを振った区間から RealSense と Motive の時刻ずれを求める")
    ap.add_argument("trial", nargs="?", help="trial_* フォルダ（省略すると最新）")
    ap.add_argument("--redetect", action="store_true", help="ボード検出をやり直す")
    ap.add_argument("--no-tmc", action="store_true", help="T_MC を使わず，区間ごとに剛体変換を当てはめる")
    ap.add_argument("--calib", help="使う較正（calib_* フォルダ，または名前）．省略すると meta.json → 試行より前の最新")
    args = ap.parse_args()

    trial_dir = io_utils.resolve_session_dir(args.trial, config.TRIAL_PREFIX)
    out = trial_dir / "sync_check"
    print(f"[TRIAL] {trial_dir}")

    T_MC, calib = None, None
    if not args.no_tmc:
        calib = io_utils.calib_for_trial(trial_dir, args.calib)
        T_MC = np.array(io_utils.load_json(calib / "T_MC.json")["T_MC"])
        print(f"[CALIB] {calib.name}")

    rs = fix_unsure(detect_board(trial_dir, out / "rs_board.npz", args.redetect))
    mc = load_mocap(trial_dir, rs["t"], config.SYNC_SEARCH + 2)
    segs = split_segments(rs)
    if not segs:
        raise SystemExit("[ERROR] 同期に使える区間がありません（ボードを大きく動かした区間が必要）")
    results = sync_all(rs, segs, mc, T_MC)
    report(results, rs["t"][0])

    io_utils.save_json(out / "sync_result.json", dict(
        trial=trial_dir.name, csv=io_utils.find_one(trial_dir / "mocap", "*.csv").name,
        calib=calib.name if calib else None, capture_start=mc["meta"]["Capture Start Time"],
        capture_start_epoch=mc["start"],
        note="Motive 時刻 = RealSense 時刻 + delta / Take 時刻 = RealSense 時刻 + c",
        drift_ms=(results[-1]["c"] - results[0]["c"]) * 1000 if len(results) >= 2 else None,
        segments=[{k: v for k, v in r.items() if not k.startswith("_")} for r in results]))
    pd.concat([r["_frames"].assign(segment=k) for k, r in enumerate(results)]).to_csv(
        out / "sync_frames.csv", index=False, float_format="%.4f")
    save_plot(out / "sync_check.png", results, mc["start"])
    print(f"[DONE] -> {out}")


if __name__ == "__main__":
    main()
