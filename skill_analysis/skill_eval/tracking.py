"""① ターゲット追従特性（task.csv の 250 Hz のカーソル・ターゲットから。単位は画面上の mm）

カーソルは DPI によらず画面上で同じ尺度なので、ここの指標は DPI をまたいで比べられる。

追従精度
  rms_2d / rms_x / rms_y   RMS 誤差 [mm]
  on_target_pct            ターゲット円の中にいた時間の割合 [%]
  vaf_pct                  追従率 = 1 − Σ|誤差|² / Σ|ターゲット − 平均|²（ターゲットの動きのうち再現できた割合）[%]
  lin_track_pct            カーソルのパワーのうち入力周波数に乗った割合（= 1 − remnant）[%]
  gain_mean / delay_ms     記述関数（ターゲット → カーソル）の平均ゲインと位相の傾きから求めた実効遅れ [ms]
  lag_mean_ms              入力周波数ごとの位相遅れ（−位相 / 360f）の平均 [ms]（周波数によらない一定の遅れ + 位相のずれ）
  gain_hi                  高い入力周波数（各軸の上 2 つ）でのゲイン（> 1 = 行き過ぎ、< 1 = 追いつけていない）
滑らかさ
  jerk_rms                 カーソルの RMS 躍度 [mm/s³]
  log_jerk_ratio           log10(RMS 躍度(カーソル) / RMS 躍度(ターゲット))。0 = ターゲットと同じ滑らかさ
  ldlj_diff                2 s 窓の LDLJ（速度版）のカーソル − ターゲット。負に大きいほどぎくしゃく
  hf_frac                  カーソル速度のパワーのうち課題外の帯域（最高入力周波数 + 0.5 Hz 以上）の割合
軌道修正
  corr_rate                修正イベントの頻度 [1/s]（誤差の大きさ |e| の山 = ずれが最大になって戻り始めた点）
  corr_amp_mm              修正イベントの振幅（|e| の山の突出度の中央値）[mm]
  corr_peak_mm             修正を始めたときの誤差（|e| の山の高さの中央値）[mm]
  corr_time_s              山から次の谷まで（誤差を戻すのにかかった時間）の中央値 [s]
  submove_excess           速さの山（サブムーブメント）の頻度のターゲットに対する超過 [1/s]
  reversal_excess          軸ごとの速度の向きの反転頻度のターゲットに対する超過 [1/s]（行き過ぎ → 戻し）
  exit_rate / out_dur_s    ターゲットから外れる頻度 [1/s] と外れてから戻るまでの平均時間 [s]
  hf_jitter                2〜10 Hz のカーソル速度の RMS [mm/s]（微小な修正・ふらつきの大きさ）
  remnant_mm               入力周波数以外のカーソル変位の RMS [mm]（ターゲットで説明できない動きの大きさ）
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, medfilt, welch

from . import config as K
from .io import C


def _uniform(tr):
    """解析区間 [0, T) の 250 Hz 一様グリッドに target / cursor / on_target を載せる。"""
    t = np.arange(int(round(tr.T * K.FS_TASK))) / K.FS_TASK
    tg = np.column_stack([np.interp(t, tr.task_t, tr.tgt[:, k]) for k in range(2)])
    cu = np.column_stack([np.interp(t, tr.task_t, tr.cur[:, k]) for k in range(2)])
    on = np.interp(t, tr.task_t, tr.on_target) > 0.5
    return t, tg, cu, on


def _lp(X, fc, fs):
    b, a = butter(4, fc / (fs / 2))
    return filtfilt(b, a, X, axis=0)


def _deriv(X, fs, n):
    for _ in range(n):
        X = np.gradient(X, axis=0) * fs
    return X


def _ldlj(V, fs):
    """速度版の log dimensionless jerk（1 区間）。-ln(D³/v_peak² ∫|j|² dt)"""
    sp = np.linalg.norm(V, axis=1)
    vp = sp.max()
    if vp <= 0:
        return np.nan
    J = _deriv(V, fs, 2)
    D = len(V) / fs
    return float(-np.log(D ** 3 / vp ** 2 * np.sum(np.sum(J ** 2, 1)) / fs))


def _reversals(v, thr):
    """速度の符号の反転回数（|v| が thr を超えたときだけ状態を更新するヒステリシス付き）。"""
    state, n = 0, 0
    for x in v:
        if x > thr:
            if state == -1: n += 1
            state = 1
        elif x < -thr:
            if state == 1: n += 1
            state = -1
    return n


def max_input_freq(tr):
    return max(max(tr.freqs(ax)[1]) for ax in C.AX)


def compute(tr):
    """1 試行ぶんの追従特性。(スカラーの dict, 周波数ごとの行のリスト, 修正イベントのリスト)"""
    fs = K.FS_TASK
    t, tg_raw, cu_raw, on = _uniform(tr)
    tg, cu = _lp(tg_raw, K.FC_TASK, fs), _lp(cu_raw, K.FC_TASK, fs)
    e = cu_raw - tg_raw
    em = np.linalg.norm(e, axis=1)
    res = {}

    # ---- 追従精度
    res["rms_2d"] = float(np.sqrt(np.mean(em ** 2)))
    res["rms_x"] = float(np.sqrt(np.mean(e[:, 0] ** 2)))
    res["rms_y"] = float(np.sqrt(np.mean(e[:, 1] ** 2)))
    res["mae_2d"] = float(em.mean())
    res["p95_2d"] = float(np.percentile(em, 95))
    res["on_target_pct"] = float(100 * on.mean())
    tv = tg_raw - tg_raw.mean(0)
    res["vaf_pct"] = float(100 * (1 - np.sum(em ** 2) / np.sum(tv ** 2)))

    # 記述関数（60 Hz の解析グリッド。aim_eval と同じ FFT）
    tg_g, cu_g = tr.task_grid("target"), tr.task_grid("cursor")
    freq_rows, gains, delays, lin = [], [], [], []
    for k, ax in enumerate(C.AX):
        nb, f, _ = tr.freqs(ax)
        d = C.describing(tg_g[:, k], cu_g[:, k], tr.T, nb)
        n_, Fe = C.spectrum(cu_g[:, k] - tg_g[:, k], tr.T)
        _, Ft = C.spectrum(tg_g[:, k], tr.T)
        Ek, Tk = C.at_bins(n_, Fe, nb), C.at_bins(n_, Ft, nb)
        tau, _ = C.delay_from_phase(d["f_hz"], d["phase_deg"], snr=d["snr"])
        delays.append(tau); gains.append(np.mean(d["gain"])); lin.append(1 - d["remnant"])
        for i in range(len(nb)):
            ph = float(d["phase_deg"][i])
            freq_rows.append({"axis": ax, "n": int(nb[i]), "f_hz": float(f[i]), "gain": float(d["gain"][i]),
                              "phase_deg": ph, "lag_ms": float(-ph / 360.0 / f[i] * 1000.0),
                              "rel_err": float(np.abs(Ek[i]) / np.abs(Tk[i])), "snr": float(d["snr"][i])})
    res["gain_mean"] = float(np.nanmean(gains))
    res["delay_ms"] = float(np.nanmean(delays))
    res["lag_mean_ms"] = float(np.mean([r["lag_ms"] for r in freq_rows]))
    hi = []
    for ax in C.AX:
        rows = sorted([r for r in freq_rows if r["axis"] == ax], key=lambda r: r["f_hz"])
        hi += [r["gain"] for r in rows[-2:]]
    res["gain_hi"] = float(np.mean(hi))
    res["lin_track_pct"] = float(100 * np.nanmean(lin))

    # ---- 滑らかさ
    Vc, Vt = _deriv(cu, fs, 1), _deriv(tg, fs, 1)
    Jc, Jt = _deriv(cu, fs, 3), _deriv(tg, fs, 3)
    jc = float(np.sqrt(np.mean(np.sum(Jc ** 2, 1))))
    jt = float(np.sqrt(np.mean(np.sum(Jt ** 2, 1))))
    res["jerk_rms"] = jc
    res["log_jerk_ratio"] = float(np.log10(jc / jt))
    w = int(K.JERK_WIN_S * fs)
    lc, lt = [], []
    for s in range(0, len(t) - w + 1, w):
        lc.append(_ldlj(Vc[s:s + w], fs)); lt.append(_ldlj(Vt[s:s + w], fs))
    res["ldlj_cursor"] = float(np.nanmean(lc))
    res["ldlj_diff"] = float(np.nanmean(np.array(lc) - np.array(lt)))
    fmax = max_input_freq(tr)
    lo = fmax + K.NONIN_MARGIN
    fw, P = welch(Vc, fs=fs, nperseg=int(4 * fs), axis=0)
    P = P.sum(1)
    res["hf_frac"] = float(P[fw >= lo].sum() / P[fw > 0].sum())

    # ---- 軌道修正
    # (1) 修正イベント: |e| の山（ずれが最大になって戻り始めた点）
    em_s = _lp(em, K.FC_TASK, fs)
    pk, pr = find_peaks(em_s, prominence=K.ERR_PEAK_PROM_MM, distance=int(K.ERR_PEAK_MIN_S * fs))
    tr_idx, _ = find_peaks(-em_s)
    events = []
    for i, p in enumerate(pk):
        nxt = tr_idx[tr_idx > p]
        j = nxt[0] if len(nxt) else len(em_s) - 1
        events.append({"t_s": float(t[p]), "peak_mm": float(em_s[p]), "amp_mm": float(pr["prominences"][i]),
                       "drop_mm": float(em_s[p] - em_s[j]), "time_s": float((j - p) / fs)})
    res["corr_n"] = len(events)
    res["corr_rate"] = len(events) / tr.T
    res["corr_amp_mm"] = float(np.median([ev["amp_mm"] for ev in events])) if events else np.nan
    res["corr_peak_mm"] = float(np.median([ev["peak_mm"] for ev in events])) if events else np.nan
    res["corr_time_s"] = float(np.median([ev["time_s"] for ev in events])) if events else np.nan
    res["corr_speed"] = float(np.median([ev["drop_mm"] / ev["time_s"] for ev in events if ev["time_s"] > 0])) \
        if events else np.nan

    # (2) サブムーブメント（速さの山）の超過
    sc, st = np.linalg.norm(Vc, axis=1), np.linalg.norm(Vt, axis=1)
    prom = K.SPEED_PEAK_PROM * float(np.sqrt(np.mean(st ** 2)))
    nc = len(find_peaks(sc, prominence=prom)[0]); nt = len(find_peaks(st, prominence=prom)[0])
    res["submove_rate"] = nc / tr.T
    res["submove_excess"] = (nc - nt) / tr.T

    # (3) 軸ごとの向きの反転（行き過ぎ → 戻し）の超過
    rev = 0.0
    for k in range(2):
        thr = K.REVERSAL_DEADBAND * float(np.sqrt(np.mean(Vt[:, k] ** 2)))
        rev += _reversals(Vc[:, k], thr) - _reversals(Vt[:, k], thr)
    res["reversal_excess"] = rev / tr.T

    # (4) ターゲットから外れる / 戻る
    kf = int(K.ONTARGET_MEDFILT_S * fs) | 1
    onf = medfilt(on.astype(float), kf) > 0.5
    d = np.diff(onf.astype(int))
    exits, entries = np.where(d == -1)[0], np.where(d == 1)[0]
    outs = [(entries[entries > x][0] - x) / fs for x in exits if np.any(entries > x)]
    res["exit_rate"] = len(exits) / tr.T
    res["out_dur_s"] = float(np.mean(outs)) if outs else np.nan

    # (5) 微小修正の大きさ
    b, a = butter(4, [K.HF_BAND[0] / (fs / 2), K.HF_BAND[1] / (fs / 2)], btype="band")
    Vhf = filtfilt(b, a, _deriv(cu_raw, fs, 1), axis=0)
    res["hf_jitter"] = float(np.sqrt(np.mean(np.sum(Vhf ** 2, 1))))
    rem = []
    for k, ax in enumerate(C.AX):
        nb, _, _ = tr.freqs(ax)
        n_, F = C.spectrum(cu_g[:, k], tr.T)
        hi = int(round(lo * tr.T))
        idx = [i for i in range(1, min(len(F), hi + 1)) if i not in set(int(x) for x in nb)]
        rem.append(np.sum(np.abs(F[idx]) ** 2) / 2)     # 片側振幅 → 分散
    res["remnant_mm"] = float(np.sqrt(np.sum(rem)))
    return res, freq_rows, events
