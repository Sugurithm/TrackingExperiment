"""② 補足: 動き出しの順序（どの関節から動かすか）

連続的な追従には「静止 → 動き出し」がほとんど無いので、**手先（MCP3）が向きを変える瞬間**を「新しい動きの始まり」とみなす。
机座標の軸（x = 左右、y = 前後）ごとに:
  1. 手先の速度 E_k(t) の 0 交差のうち、前後 0.3 s の |E_k| の最大がどちらも 0.3·RMS(E_k) を超えるもの（本当の切り返し）をイベントとする
  2. 各成分（体幹・肩・肘・手首）の速度 V_p,k(t) の、同じ向きの 0 交差のうちイベントに一番近いもの（±0.4 s 以内）を探し、
     τ_p = t_p − t_0 [ms] とする（負 = 手先より先に向きを変えた = その関節から動き出した）。0 交差は線形補間で 1 フレーム未満まで求める
  3. その軸でほとんど動いていない成分（RMS が手先の 15 % 未満）と、イベントの前後で振幅が小さい成分は「関与なし」として除く
指標（試行ごと、x・y のイベントをまとめる）
  init_tau_<p>_ms    τ_p の中央値（負 = 手先より先）
  init_first_<p>     関与した成分の中で最初に向きを変えた割合（関与成分が 2 つ以上のイベントだけ）
  init_prec_<a>_<b>  a と b がどちらも関与したイベントで、a が b より先だった割合（0.5 = 同時 / 順序なし）
  init_rel_<a>_<b>_ms  τ_a − τ_b の中央値（負 = a が先）
速度は 3 Hz の低域通過をかけてから 0 交差を探す（課題の最高周波数は 1.08 Hz。それより速い揺れの交差を拾わないため）。
注意: 肩・肘は MediaPipe Pose、手首・MCP3 は Hand で推定しており、モデルの時間平滑化の違いが成分間の見かけの時間差になりうる。
      全員に同じだけ乗るので練度の比較には効かないが、「手首が先」などの絶対的な順序は割り引いて読むこと。
"""
import numpy as np
from scipy.signal import butter, filtfilt

from .io import C

PARTS = ["trunk", "upperarm", "forearm", "hand"]
PAIRS = [("upperarm", "forearm"), ("forearm", "hand"), ("upperarm", "hand"), ("trunk", "upperarm")]
FC = 3.0
WIN_AMP_S = 0.3
AMP_FRAC = 0.3
SEARCH_S = 0.4
ACTIVE_FRAC = 0.15


def _crossings(v, fs):
    """0 交差の時刻 [s]（線形補間）と向き（+1 = 負 → 正）。"""
    s = np.sign(v)
    idx = np.where((s[:-1] != 0) & (s[:-1] != s[1:]))[0]
    t = (idx + v[idx] / (v[idx] - v[idx + 1])) / fs
    return t, np.sign(v[idx + 1] - v[idx])


def compute(tr):
    """(スカラーの dict, イベントの行のリスト)。関節データが無ければ (None, None)。"""
    V, E, ok = C.vel_contributions(tr)
    if V is None:
        return None, None
    fs = C.FS_U
    b, a = butter(2, FC / (fs / 2))
    lp = lambda x: filtfilt(b, a, C.fill_nan(x)[0], axis=0)
    E = lp(E); V = {p: lp(V[p]) for p in PARTS}
    w = int(WIN_AMP_S * fs); smax = SEARCH_S
    events = []
    for k, ax in enumerate(C.AX):
        e = E[:, k]
        rms_e = np.sqrt(np.mean(e ** 2))
        active = [p for p in PARTS if np.sqrt(np.mean(V[p][:, k] ** 2)) >= ACTIVE_FRAC * rms_e]
        cr = {p: _crossings(V[p][:, k], fs) for p in active}
        rms_p = {p: np.sqrt(np.mean(V[p][:, k] ** 2)) for p in active}
        te, de = _crossings(e, fs)
        for t0, d0 in zip(te, de):
            i0 = int(round(t0 * fs))
            if i0 - w < 0 or i0 + w >= len(e):
                continue
            if min(np.abs(e[i0 - w:i0]).max(), np.abs(e[i0:i0 + w]).max()) < AMP_FRAC * rms_e:
                continue                                        # 揺れの交差（本当の切り返しではない）
            row = {"axis": ax, "t_s": float(t0), "dir": int(d0)}
            for p in active:
                v = V[p][:, k]
                if min(np.abs(v[i0 - w:i0]).max(), np.abs(v[i0:i0 + w]).max()) < AMP_FRAC * rms_p[p]:
                    continue                                    # この成分はこの切り返しに関与していない
                tp, dp = cr[p]
                m = (dp == d0) & (np.abs(tp - t0) <= smax)
                if m.any():
                    j = np.argmin(np.abs(tp[m] - t0))
                    row[f"tau_{p}_ms"] = float((tp[m][j] - t0) * 1000.0)
            events.append(row)

    res = {"init_n_events": len(events)}
    for p in PARTS:
        v = [r[f"tau_{p}_ms"] for r in events if f"tau_{p}_ms" in r]
        res[f"init_tau_{p}_ms"] = float(np.median(v)) if len(v) >= 5 else np.nan
        res[f"init_n_{p}"] = len(v)
    first = {p: 0 for p in PARTS}; n_multi = 0
    for r in events:
        taus = {p: r[f"tau_{p}_ms"] for p in PARTS if f"tau_{p}_ms" in r}
        if len(taus) >= 2:
            n_multi += 1
            first[min(taus, key=taus.get)] += 1
            r["first"] = min(taus, key=taus.get)
    for p in PARTS:
        res[f"init_first_{p}"] = first[p] / n_multi if n_multi else np.nan
    for a_, b_ in PAIRS:
        d = [r[f"tau_{a_}_ms"] - r[f"tau_{b_}_ms"] for r in events if f"tau_{a_}_ms" in r and f"tau_{b_}_ms" in r]
        res[f"init_prec_{a_}_{b_}"] = float(np.mean(np.array(d) < 0) + 0.5 * np.mean(np.array(d) == 0)) if len(d) >= 5 else np.nan
        res[f"init_rel_{a_}_{b_}_ms"] = float(np.median(d)) if len(d) >= 5 else np.nan
        res[f"init_npair_{a_}_{b_}"] = len(d)
    return res, events
