"""信号処理（欠損の補間・Hampel・低域通過・入力周波数ビンの読み取り）

aim_eval/common.py・accuracy_eval/common.py・skill_analysis でそれぞれ書かれていたものを 1 つにした。
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _swv
from scipy.signal import butter, filtfilt

from .. import config as K


def fill_nan(X):
    """欠損を直線補間で埋める。(埋めた配列, 欠損フラグ)"""
    X = np.array(X, float, copy=True)
    one = X.ndim == 1
    if one:
        X = X[:, None]
    bad = ~np.isfinite(X).all(axis=1)
    i = np.arange(len(X))
    if (~bad).sum() >= 2:
        for k in range(X.shape[1]):
            X[bad, k] = np.interp(i[bad], i[~bad], X[~bad, k])
    return (X[:, 0] if one else X), bad


def hampel(X, k=K.HAMPEL_K, n=K.HAMPEL_N):
    """Hampel フィルタ（窓 2k+1、n·MAD を超える点を中央値に置き換える）。"""
    X = np.asarray(X, float)
    Xf, bad = fill_nan(X)
    one = Xf.ndim == 1
    if one:
        Xf = Xf[:, None]
    Y = np.array(X, float, copy=True).reshape(Xf.shape)
    if len(Xf) <= 2 * k + 1:
        return X.copy()
    for a in range(Xf.shape[1]):
        w = _swv(np.pad(Xf[:, a], k, mode="edge"), 2 * k + 1)
        med = np.median(w, axis=1)
        mad = 1.4826 * np.median(np.abs(w - med[:, None]), axis=1)
        flag = (np.abs(Xf[:, a] - med) > n * mad) & (mad > 0) & ~bad
        Y[flag, a] = med[flag]
    return Y[:, 0] if one else Y


def lowpass(X, fs, fc=K.FC_LP, order=4):
    """ゼロ位相の Butterworth 低域通過。欠損は補間してからかけ、元の欠損位置は NaN に戻す。"""
    X = np.asarray(X, float)
    Xf, bad = fill_nan(X)
    if (~bad).sum() < 30 or not np.isfinite(fs):
        return X.copy()
    b, a = butter(order, min(fc / (fs / 2), 0.99))
    Y = filtfilt(b, a, Xf, axis=0)
    Y[bad] = np.nan
    return Y


def resample(t, X, tg, gap_max=K.GAP_MAX_S):
    """(t, X) を一様グリッド tg に載せる。gap_max より長い穴と範囲外は NaN。"""
    X = np.asarray(X, float)
    one = X.ndim == 1
    X2 = X[:, None] if one else X
    out = np.full((len(tg), X2.shape[1]), np.nan)
    for k in range(X2.shape[1]):
        g = np.isfinite(t) & np.isfinite(X2[:, k])
        if g.sum() < 10:
            continue
        tt, xx = t[g], X2[g, k]
        out[:, k] = np.interp(tg, tt, xx, left=np.nan, right=np.nan)
        idx = np.clip(np.searchsorted(tt, tg), 1, len(tt) - 1)
        out[(tt[idx] - tt[idx - 1]) > gap_max, k] = np.nan
    return out[:, 0] if one else out


# ------------------------------------------------------------------ 入力周波数ビン
def spectrum(x):
    """一様グリッドの信号（解析区間ちょうど 1 周期）→ 片側の複素振幅（単位は元の振幅）。

    x: (N,) または (N, d)。欠損は補間して埋め、平均を引く。
    """
    y, _ = fill_nan(x)
    y = y - np.mean(y, axis=0)
    return np.fft.rfft(y, axis=0) * 2.0 / len(y)


def neighbors(b, taken, nbin, half=K.NOISE_HALF, skip=1):
    """ビン b のノイズを測る近傍ビン（入力ビンと隣接 1 つは除く）。"""
    return [i for i in range(max(1, b - half - skip), min(nbin, b + half + skip + 1))
            if abs(i - b) > skip and i not in taken]


def bin_read(F, bins):
    """複素スペクトル F (nf,) or (nf, d) から、入力ビンの値と近傍のノイズパワーを読む。

    返り値: (値 (nb, d) 複素, ノイズパワー (nb, d))。ノイズパワーは近傍ビンの |F|² の平均。
    """
    F = np.asarray(F)
    one = F.ndim == 1
    F2 = F[:, None] if one else F
    taken = set(int(b) for b in bins)
    val = np.full((len(bins), F2.shape[1]), np.nan + 0j)
    noi = np.full((len(bins), F2.shape[1]), np.nan)
    for i, b in enumerate(bins):
        if b >= len(F2):
            continue
        val[i] = F2[b]
        idx = neighbors(int(b), taken, len(F2))
        if idx:
            noi[i] = np.mean(np.abs(F2[idx]) ** 2, axis=0)
    if one:
        return val[:, 0], noi[:, 0]
    return val, noi


def grad(X, fs):
    """欠損を保ったまま時間微分（中心差分）。"""
    X = np.asarray(X, float)
    return np.gradient(X, axis=0) * fs
