"""② 上肢関節の協調・分担（RealSense + MediaPipe の 3 次元関節位置、机座標 [mm]、60 Hz グリッド）

手先（MCP3）の動きを足し算で厳密に 4 成分に分ける（aim_eval と同じ分解）:
    手先 = 肩 + (肘 − 肩) + (手首 − 肘) + (MCP3 − 手首)
         = 体幹（肩の並進）+ 肩関節（上腕の回転）+ 肘関節（前腕の回転）+ 手首（手の回転）

寄与・運動量の分配
  share_<p>        入力周波数ビンでの速度パワーのシェア（近傍ビンのノイズを引く。aim_eval の a1 と同じ）
  N_eff            関与関節数 1/Σshare²（1 = 一関節依存、4 = 全体）
  S                周波数ごとの分担のそろい方（1 = どの速さも同じ分担、< 1 = 速さで担当が入れ替わる）
  C                Σ|成分| / |手先|（1 = 素直な足し算、> 1 = 打ち消し合いながら手先を作る）
  proj_<p>         変位の手先への射影シェア（符号付き。負 = 手先と逆向きに動いて他を打ち消す）
  path_<p>         各成分の速度の大きさの積分の割合（ノイズ床を引く前の「運動量」。参考）
  distal_share     肘関節 + 手首 のシェア（遠位への依存）
  share_<軸>_<p>   軸ごとのシェア（x = 左右、y = 前後）。軸によって担当の関節が違うか
時間軸の役割分担
  share_lo_<p> / share_hi_<p> / fshift_<p>
                   低い入力周波数（各軸の下 2 つ）と高い入力周波数（上 2 つ）でのシェアとその差（高 − 低）
  lead_<p>_ms      入力周波数での手先に対する位相の進み [ms]（+ = 手先より先に動く）
  xr_<a>_<b> / xlag_<a>_<b>_ms
                   成分 a, b の速度の相互相関（課題の帯域、x・y の平均）の最大 |r| の値と、そのラグ
                   （+ = a が先行）。r が負 = 互いに逆向きに動いて支え合う（補償）
  opp_<a>_<b>      速度ベクトルが逆向き（内積 < 0）だった運動量の割合
  R / R_reliable   成分の活動（速さの包絡）の同時性（aim_eval の R。動いている成分が 2 つ以上のときだけ有効）
関節角度（参考）
  elbow_range / wrist_range   肘角（肩–肘–手首）・手首角（肘–手首–MCP3）の 5–95 % 範囲 [deg]
計測の確認
  mouse_gain_x / y  手首の 3 次元変位 / マウスの変位（机座標を決めた回帰のゲイン。1 から大きく外れたら注意）
"""
import numpy as np
from scipy.signal import butter, filtfilt

from . import config as K
from .io import C, floor_cached

PARTS = ["trunk", "upperarm", "forearm", "hand"]
PLAB = {"trunk": "体幹（肩の並進）", "upperarm": "肩関節（上腕）",
        "forearm": "肘関節（前腕）", "hand": "手首（手）"}
PSHORT = {"trunk": "体幹", "upperarm": "肩", "forearm": "肘", "hand": "手首"}
PAIRS = [("trunk", "upperarm"), ("upperarm", "forearm"), ("forearm", "hand"), ("upperarm", "hand")]

# 既存のノイズ床キャッシュを読むだけにする（記録フォルダに書き込まない）
C.static_floor = lambda tr, cache=True: floor_cached(tr)


def _bandpass(X, lo, hi, fs):
    b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, X, axis=0)


def _xcorr(a, b, max_lag):
    """正規化相互相関。最大 |r| の (r, lag)。lag > 0 = a が b より先に動く。"""
    a = (a - a.mean()) / (a.std() + 1e-12); b = (b - b.mean()) / (b.std() + 1e-12)
    n = len(a)
    best = (0.0, 0)
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            r = np.mean(a[:n - L] * b[L:])        # a(t) と b(t + L): a が L だけ先行
        else:
            r = np.mean(a[-L:] * b[:n + L])
        if abs(r) > abs(best[0]):
            best = (float(r), L)
    return best


def compute(tr, fmax):
    """1 試行ぶんの協調指標。(スカラーの dict, 周波数 × 成分の行のリスト) または (None, None)"""
    if not tr.has_rs:
        return None, None
    ji = C.joint_indices(tr)
    if ji is None:
        return None, None
    res = {"N_eff": ji["N_eff"], "S": ji["S"], "C": ji["C"], "R": ji.get("R", np.nan),
           "R_reliable": ji.get("R_reliable"), "dominant": ji["dominant"], "dominant_share": ji["dominant_share"]}
    for p in PARTS:
        res[f"share_{p}"] = ji["share"].get(p, np.nan)
        res[f"snr_{p}"] = ji["snr_part"].get(p, np.nan)
    res["distal_share"] = res["share_forearm"] + res["share_hand"]

    V, E, ok = C.vel_contributions(tr)
    parts_disp = C.contributions(tr, end="mcp3")
    fs = C.FS_U

    # ---- 変位の射影シェア・運動量（速度の大きさの積分）
    vs = C.var_share({p: parts_disp[p] for p in PARTS}, parts_disp["_total"])
    for p in PARTS:
        res[f"proj_{p}"] = vs[p]["share"]
        res[f"rms_{p}_mm"] = vs[p]["rms_mm"]
    res["endpoint_rms_mm"] = vs["_total_rms_mm"]
    path = {p: float(np.nansum(np.linalg.norm(V[p], axis=1)) / fs) for p in PARTS}
    tot = sum(path.values())
    for p in PARTS:
        res[f"path_{p}"] = path[p] / tot if tot > 0 else np.nan

    # ---- 周波数ごとの分担・位相（入力周波数ビン）
    freq_rows = []
    for k, ax in enumerate(C.AX):
        nb, f, _ = tr.freqs(ax)
        AE = C.bin_power(E[:, k], tr.T, nb)[2]
        A = {p: C.bin_power(V[p][:, k], tr.T, nb)[2] for p in PARTS}
        prow = [r for r in ji["per_bin"] if r["axis"] == ax]
        for i in range(len(nb)):
            pw = np.array([prow[i]["_pwr"][p] for p in PARTS])
            row = {"axis": ax, "n": int(nb[i]), "f_hz": float(f[i]), "snr": prow[i]["snr"]}
            for j, p in enumerate(PARTS):
                row[f"share_{p}"] = float(pw[j] / pw.sum()) if pw.sum() > 0 else np.nan
                row[f"proj_{p}"] = float(np.real(A[p][i] * np.conj(AE[i])) / np.abs(AE[i]) ** 2)
                ph = float(np.degrees(np.angle(A[p][i] / AE[i])))
                row[f"phase_{p}_deg"] = ph
                row[f"lead_{p}_ms"] = ph / 360.0 / f[i] * 1000.0
                row[f"amp_{p}"] = float(np.abs(A[p][i]))
            freq_rows.append(row)

    # 軸ごとのシェア（x = 左右、y = 前後。軸で担当の関節が違うか）
    for k, ax in enumerate(C.AX):
        pw = {p: sum(r["_pwr"][p] for r in ji["per_bin"] if r["axis"] == ax) for p in PARTS}
        tot_ax = sum(pw.values())
        for p in PARTS:
            res[f"share_{ax}_{p}"] = pw[p] / tot_ax if tot_ax > 0 else np.nan

    # 低い / 高い入力周波数でのシェア（各軸の下 2 つ / 上 2 つ、SNR ≥ 2 のビンだけ）
    good = [r for r in freq_rows if np.isfinite(r["snr"]) and r["snr"] >= C.SNR_MIN]
    for p in PARTS:
        lo_v, hi_v = [], []
        for ax in C.AX:
            rows = sorted([r for r in good if r["axis"] == ax], key=lambda r: r["f_hz"])
            if len(rows) >= 3:
                lo_v += [r[f"share_{p}"] for r in rows[:2]]
                hi_v += [r[f"share_{p}"] for r in rows[-2:]]
        res[f"share_lo_{p}"] = float(np.mean(lo_v)) if lo_v else np.nan
        res[f"share_hi_{p}"] = float(np.mean(hi_v)) if hi_v else np.nan
        res[f"fshift_{p}"] = res[f"share_hi_{p}"] - res[f"share_lo_{p}"]
        # 手先に対する位相の進み: 成分の振幅で重み付けした平均（位相が ±90° を超えるビンは逆向きの補償なので除く）
        w, v = [], []
        for r in good:
            ph = r[f"phase_{p}_deg"]
            if abs(ph) < 90:
                w.append(r[f"amp_{p}"]); v.append(r[f"lead_{p}_ms"])
        res[f"lead_{p}_ms"] = float(np.average(v, weights=w)) if w and np.sum(w) > 0 else np.nan

    # ---- 時間領域: 成分どうしの相互相関（課題の帯域に絞る）と逆向きの運動量
    Vf = {}
    for p in PARTS:
        X, _ = C.fill_nan(V[p])
        Vf[p] = _bandpass(X, 0.03, fmax + K.NONIN_MARGIN, fs)
    ml = int(round(K.XCORR_MAX_LAG_S * fs))
    for a, b in PAIRS:
        rs, ls = [], []
        for k in range(2):
            r, L = _xcorr(Vf[a][:, k], Vf[b][:, k], ml)
            rs.append(r); ls.append(L / fs * 1000.0)
        res[f"xr_{a}_{b}"] = float(np.mean(rs))
        res[f"xlag_{a}_{b}_ms"] = float(np.mean(ls))
        dot = np.sum(Vf[a] * Vf[b], 1)
        mag = np.linalg.norm(Vf[a], axis=1) * np.linalg.norm(Vf[b], axis=1)
        res[f"opp_{a}_{b}"] = float(mag[dot < 0].sum() / mag.sum()) if mag.sum() > 0 else np.nan

    # ---- 関節角度（参考）
    G = {n: tr.joint_grid(n) for n in ["shoulder", "elbow", "wrist", "mcp3"]}
    for nm, (a, b, c) in {"elbow": ("shoulder", "elbow", "wrist"), "wrist": ("elbow", "wrist", "mcp3")}.items():
        ang = C.angle_deg(G[a], G[b], G[c])
        ang = ang[np.isfinite(ang)]
        res[f"{nm}_range"] = float(np.percentile(ang, 95) - np.percentile(ang, 5)) if len(ang) else np.nan

    # ---- 計測の確認
    res["mouse_gain_x"] = tr.frame.get("gain_x", np.nan)
    res["mouse_gain_y"] = tr.frame.get("gain_y", np.nan)
    res["valid_pct"] = float(100 * np.mean(ok))
    return res, freq_rows
