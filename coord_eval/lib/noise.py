"""ノイズ床（静止試行）と、振幅を縮めた擬似試行（DPI 比較の偏りの補正用）

RealSense + MediaPipe のジッタは関節の動きの大きさに依らずほぼ一定なので、
DPI を上げて動きが小さくなると SNR が下がり、ノイズに敏感な指標（N_eff・相殺比など）が
系統的にずれる（coordination_analysis.md §6.4）。これは試行数を増やしても消えないので、
400 の試行を 1/2 に縮めて静止試行のノイズを入れ直した擬似 800 を作り、指標のずれを測って引く。
"""
import copy
import numpy as np

from .. import config as K
from . import kin, sig


def static_floor(static):
    """静止試行から、成分ごとの位置のノイズ σ [mm]（3 次元、前処理後）と角度のノイズ [deg]。"""
    if static is None or not static.has_rs:
        return {}
    parts, end, ok = kin.components(static.J, K.PARTS + ["distal"])
    out = {}
    for p, X in list(parts.items()) + [("end", end)]:
        Y = X[ok] - np.nanmean(X[ok], 0)
        out[p] = float(np.sqrt(np.nanmean(np.sum(Y ** 2, 1)))) if ok.sum() > 30 else np.nan
    ori = kin.orientation(static.J)
    for k, a in ori.items():
        a = a[np.isfinite(a)]
        out[k] = float(np.std(a)) if len(a) > 30 else np.nan
    out["_trial"] = static.name
    return out


def noise_samples(static, tracks):
    """カメラ座標の生のノイズ列 {関節: (n,3)}。静止試行が無ければ本番試行の 1.5 Hz 以上の残差で代用。"""
    src = {}
    if static is not None and static.has_rs:
        for n, X in static.cam.items():
            Xf, bad = sig.fill_nan(X[static.in_span])
            src[n] = Xf - np.median(Xf, 0)
        src["_source"] = "static"
        return src
    tr = tracks[0]
    for n, X in tr.cam.items():
        Xf, _ = sig.fill_nan(X[tr.in_span])
        src[n] = Xf - sig.lowpass(Xf, tr.fs_rs, fc=1.5)
    src["_source"] = "highpass"
    return src


def pseudo_trial(tr, noise, scale, rng):
    """振幅を scale 倍にし、失われたノイズ（√(1 − scale²) 倍）を足した擬似試行。"""
    ps = copy.copy(tr)
    cam = {}
    for n, X in tr.cam.items():
        mu = np.nanmedian(X[tr.in_span], 0)
        Y = mu + scale * (X - mu)
        Nz = noise.get(n)
        if Nz is not None and len(Nz) > 10:
            idx = (np.arange(len(X)) + rng.integers(len(Nz))) % len(Nz)
            Y = Y + np.sqrt(max(1 - scale ** 2, 0)) * Nz[idx]
        cam[n] = Y
    ps.cam = cam
    ps.mouse = tr.mouse * scale
    ps.mouse_raw = tr.mouse_raw * scale
    ps.set_frame(tr.frame)
    return ps
