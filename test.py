# sync_check.py
# ─────────────────────────────────────────────────────────────────────────────
# RealSense（.db3）と Motive（.csv）の時刻ずれ δ を，ボードを動かした区間から求める．
#
#     t_Motive [epoch s] = t_RealSense [epoch s] + δ
#     Take 内の時刻 [s]  = t_RealSense [epoch s] + δ − Capture Start Time [epoch s]
#
# 手順
#   1. RealSense: 各フレームでチェッカーボードを検出 → ボード上の反射マーカー S1〜S3 の位置（カメラ座標）
#   2. Motive   : 生マーカーから S1〜S3 をマーカー間距離で識別（ラベル付けは不要）
#   3. RealSense の検出が途切れる所で区間に分ける（開始付近と終了付近で 2 区間になる想定）
#   4. 区間ごとに
#        粗い探索: ボード中心の「速さ」の相関が最大になる δ（回転に依らないので T_MC 不要）
#        細かい探索: マーカー 3 点の 3 次元位置の差（RMS）が最小になる δ（1 ms 刻み → 放物線補間）
#   5. 区間ごとの δ，残差，δ の差（記録中のずれ）を表示・保存
#
# 使い方（TrackingExperiment のルートで）
#   python sync_check.py --rs recordings/trial_20261015_201500 --csv recordings/Take_xxx.csv \
#                        --tmc recordings/calib_20261015_1800/T_MC.json
#   --rs  : session.db3 か，それを含むフォルダ（rgb/ と frame_timestamps.csv があるフォルダでも可）
#   --tmc : 省略すると，δ ごとに剛体変換を当てはめて残差を出す（T_MC の確認はできない）
#   2 回目以降はボード検出結果（sync_check/rs_board.npz）を再利用する．やり直すときは --redetect
#
# 出力（--rs のフォルダ /sync_check/）
#   sync_result.json : 区間ごとの δ・残差など
#   sync_check.png   : 相関・残差の曲線と，合わせた後の軌跡の重ね描き
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
_JP = [f for f in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic")
       if f in {x.name for x in font_manager.fontManager.ttflist}]
plt.rcParams["font.family"] = _JP[:1] + ["DejaVu Sans"]   # 日本語フォントがあれば使う

# ====== 設定（config.py と値を揃えること）=====
SQUARE = 23.97                  # チェッカー 1 マス [mm]（印刷倍率を反映した実測値）
PATTERN = (7, 4)                # 内部コーナー数
PRINT_SCALE = 0.9588            # 印刷倍率（マーカー位置に掛ける）
MARKERS_NOMINAL = {"S1": (-40, -40), "S2": (190, -40), "S3": (-40, 115)}   # 設計値 [mm]
MARKER_Z = 0.0                  # マーカー中心の板面からの高さ（平らなシールなら 0）
TZ_HOURS = 9                    # Motive PC のタイムゾーン（Capture Start Time は現地時刻）

MAX_ERR_PX = 1.0                # 再投影誤差の上限 [px]（動いているので較正時より緩め）
TOL_MM = 4.0                    # MoCap のマーカー間距離の許容 [mm]
JUMP_MM = 15.0                  # 前後のフレームの中央値からこれ以上飛んだフレームは誤識別・誤検出として捨てる [mm]
SEG_GAP = 2.0                   # RealSense の検出がこれ以上途切れたら別区間 [s]
MIN_SEG_FRAMES = 30             # 区間の最少フレーム数
MIN_PATH_MM = 100.0             # 区間内でボードがこれ以上動いていなければ同期に使わない [mm]
SEARCH = 30.0                   # 粗い探索の範囲 ±[s]（時計の粗い合わせからのずれ）
GRID_HZ = 120.0                 # 速さを比べる共通の時間軸 [Hz]
SMOOTH = 0.05                   # 速さの平滑化の窓 [s]
FINE_RANGE = 0.15               # 細かい探索の範囲 ±[s]
FINE_STEP = 0.001               # 細かい探索の刻み [s]
COARSE_STRIDE = 6               # ボードの写っている時間帯を探すときの間引き（検出を速くするため）
NEXT_SEARCH = 2.0               # 2 つ目以降の区間は，最も確からしい区間の δ ±この範囲で探す [s]

OBJ = np.array([[j * SQUARE, i * SQUARE, 0] for i in range(PATTERN[1]) for j in range(PATTERN[0])], np.float32)
PB = np.array([(x * PRINT_SCALE, y * PRINT_SCALE, MARKER_Z) for x, y in MARKERS_NOMINAL.values()], float)
D12, D13, D23 = (np.linalg.norm(PB[b] - PB[a]) for a, b in [(0, 1), (0, 2), (1, 2)])


# ════════════════════════════════════════════════════════════════════════════
#  RealSense: ボード検出
# ════════════════════════════════════════════════════════════════════════════
def detect_board(gray, K, D):
    """チェッカーボードを検出して (R, t, 再投影誤差) を返す．realsense_transfer.py と同じ手順．"""
    ok, c = cv2.findChessboardCornersSB(gray, PATTERN, flags=cv2.CALIB_CB_ACCURACY)
    if not ok:
        return None

    def pnp(c):
        _, r, t = cv2.solvePnP(OBJ, c, K, D, flags=cv2.SOLVEPNP_IPPE)
        return cv2.solvePnPRefineLM(OBJ, c, K, D, r, t)

    r, t = pnp(c)
    u, v = cv2.projectPoints(np.array([[-SQUARE / 2, -SQUARE / 2, 0.]]), r, t, K, D)[0].ravel().astype(int)
    h, w = gray.shape
    if 3 <= u < w - 3 and 3 <= v < h - 3 and gray[v - 3:v + 4, u - 3:u + 4].mean() >= 128:
        c = c[::-1].copy()                 # 左上マスが白なら向きが逆
        r, t = pnp(c)
    R = cv2.Rodrigues(r)[0]
    proj = cv2.projectPoints(OBJ, r, t, K, D)[0].reshape(-1, 2)
    err = float(np.linalg.norm(proj - c.reshape(-1, 2), axis=1).mean())
    if R[2, 2] <= 0 or err > MAX_ERR_PX:
        return None
    return R, t.ravel(), err


def frames_from_db3(path):
    """session.db3 を全フレーム再生し (時刻[ms], 時刻の種類, BGR画像, K, D) を順に返す．"""
    import pyrealsense2 as rs
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, str(path), repeat_playback=False)
    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)
    profile.get_device().as_playback().set_real_time(False)      # フレームを落とさない
    cs = profile.get_stream(rs.stream.color).as_video_stream_profile()
    it = cs.get_intrinsics()
    K = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1]])
    D = np.array(it.coeffs)
    last = None
    try:
        while True:
            ok, fs = pipeline.try_wait_for_frames(timeout_ms=3000)
            if not ok:
                break
            cf = fs.get_color_frame()
            if not cf or cf.get_frame_number() == last:          # 同じカラーフレームの重複を除く
                continue
            last = cf.get_frame_number()
            img = np.asanyarray(cf.get_data()).copy()
            if cs.format() == rs.format.rgb8:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            yield cf.get_timestamp(), cf.get_frame_timestamp_domain().name, img, K, D
    finally:
        pipeline.stop()


def frames_from_jpg(folder):
    """rs_export_rgb.py の書き出し（rgb/*.jpg, frame_timestamps.csv, color_intrinsics.json）から読む．"""
    it = json.loads((folder / "color_intrinsics.json").read_text(encoding="utf-8"))
    K = np.array([[it["fx"], 0, it["ppx"]], [0, it["fy"], it["ppy"]], [0, 0, 1]])
    D = np.array(it["coeffs"])
    ts = pd.read_csv(folder / "frame_timestamps.csv")
    imgs = sorted((folder / "rgb").glob("*.jpg"))
    last = None
    for row, p in zip(ts.itertuples(), imgs):
        if row.color_frame_number == last:
            continue
        last = row.color_frame_number
        yield row.timestamp_ms, row.domain, cv2.imread(str(p)), K, D


def load_rs(rs_path, cache, redetect, stride):
    if cache.exists() and not redetect:
        z = np.load(cache, allow_pickle=False)
        print(f"[RS] 検出結果を再利用: {cache}（やり直すときは --redetect）")
        return dict(t=z["t"], R=z["R"], tvec=z["tvec"], err=z["err"], epoch=bool(z["epoch"]), n_all=int(z["n_all"]))

    p = Path(rs_path)
    if p.is_dir() and (p / "session.db3").exists():
        open_frames = lambda: frames_from_db3(p / "session.db3")
    elif p.suffix == ".db3":
        open_frames = lambda: frames_from_db3(p)
    elif p.is_dir() and (p / "rgb").is_dir():
        open_frames = lambda: frames_from_jpg(p)
    else:
        raise SystemExit(f"RealSense の記録が見つかりません: {p}")

    # 検出は 1 フレーム 0.1 s ほどかかるので 2 段階にする
    #   1 回目: COARSE_STRIDE フレームおきに検出して，ボードが写っている時間帯を見つける
    #   2 回目: その時間帯（前後 0.5 s を含む）の残りのフレームを検出する
    hits, domains, n = {}, set(), 0
    for k, (ts_ms, dom, img, K, D) in enumerate(open_frames()):
        n += 1
        domains.add(dom)
        if k % COARSE_STRIDE == 0:
            res = detect_board(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), K, D)
            if res:
                hits[k] = (ts_ms / 1000.0,) + res
        if n % 600 == 0:
            print(f"  [1/2] {n} frames, ボードあり {len(hits)}")
    ts_hit = np.array([v[0] for v in hits.values()])
    if len(ts_hit):
        for k, (ts_ms, dom, img, K, D) in enumerate(open_frames()):
            if k in hits or k % stride:
                continue
            if np.min(np.abs(ts_hit - ts_ms / 1000.0)) > COARSE_STRIDE / 60 + 0.5:
                continue
            res = detect_board(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), K, D)
            if res:
                hits[k] = (ts_ms / 1000.0,) + res
            if k % 600 == 0:
                print(f"  [2/2] {k} / {n} frames, ボードあり {len(hits)}")
    order = sorted(hits)
    t = [hits[k][0] for k in order]
    R = [hits[k][1] for k in order]
    tvec = [hits[k][2] for k in order]
    err = [hits[k][3] for k in order]
    epoch = bool(domains) and domains <= {"system_time", "global_time"} and (not t or t[0] > 1e9)
    if not epoch:
        print(f"[WARN] タイムスタンプが PC 時刻ではありません（{domains}）．δ の粗い範囲は Take 全体から探します．")
    out = dict(t=np.array(t), R=np.array(R).reshape(-1, 3, 3), tvec=np.array(tvec).reshape(-1, 3),
               err=np.array(err), epoch=epoch, n_all=n)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    print(f"[RS] {n} frames, ボード検出 {len(t)} frames")
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Motive: CSV 読み込みと S1〜S3 の識別
# ════════════════════════════════════════════════════════════════════════════
def parse_capture_start(s):
    """'2026-08-31 02.50.50.277 午後' / '... PM' / 24 時間表記 → epoch [s]"""
    s = s.strip()
    pm = any(x in s for x in ("午後", "PM", "pm"))
    am = any(x in s for x in ("午前", "AM", "am"))
    for x in ("午後", "午前", "PM", "AM", "pm", "am"):
        s = s.replace(x, "")
    date, clock = s.split()[:2]
    parts = clock.replace(":", ".").split(".")
    h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    ms = int(parts[3].ljust(3, "0")[:3]) if len(parts) > 3 else 0
    if pm and h < 12:
        h += 12
    if am and h == 12:
        h = 0
    y, mo, d = map(int, date.replace("/", "-").split("-"))
    t = dt.datetime(y, mo, d, h, m, sec, ms * 1000, tzinfo=dt.timezone(dt.timedelta(hours=TZ_HOURS)))
    return t.timestamp()


def read_header(path):
    raw = Path(path).read_bytes()[:200000]
    for enc in ("utf-8-sig", "cp932"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    return [r for r, _ in zip(csv.reader(text.splitlines()), range(7))]


def identify(P):
    """生マーカーの集合から S1, S2, S3 を距離で識別する（motive_transfer.py と同じ）．"""
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    best, cost = None, np.inf
    n = len(P)
    for a in range(n):
        for b in range(n):
            if b == a or abs(d[a, b] - D12) >= TOL_MM:
                continue
            for c in range(n):
                if c in (a, b):
                    continue
                e = (abs(d[a, b] - D12), abs(d[a, c] - D13), abs(d[b, c] - D23))
                if max(e) < TOL_MM and sum(e) < cost:
                    best, cost = P[[a, b, c]], sum(e)
    return best


def reject_jumps(P, win):
    """位置 (n,3,3) の重心が前後 win フレームの中央値から JUMP_MM 以上離れたフレームを True で返す．
    1 フレームだけの誤識別でも速さが跳ね，相関を壊すため．"""
    c = P.mean(1)
    med = pd.DataFrame(c).rolling(win, center=True, min_periods=3).median().to_numpy()
    return np.linalg.norm(c - med, axis=1) > JUMP_MM


def load_mocap(path, window=None):
    head = read_header(path)
    meta = dict(zip(head[0][0::2], head[0][1::2]))
    types, kinds, axes = head[2], head[5], head[6]
    cols = [j for j in range(len(types)) if types[j] == "Marker" and kinds[j] == "Position" and axes[j] == "X"]
    use = [0, 1] + [j + k for j in cols for k in range(3)]
    df = pd.read_csv(path, skiprows=7, header=None, usecols=use, dtype=np.float64,
                     encoding="utf-8", encoding_errors="replace")
    t = df[1].to_numpy()
    X = df[[j + k for j in cols for k in range(3)]].to_numpy().reshape(len(df), len(cols), 3)
    X *= {"Meters": 1000, "Millimeters": 1, "Centimeters": 10}[meta["Length Units"]]
    if window is not None:
        sel = (t >= window[0]) & (t <= window[1])
        t, X = t[sel], X[sel]
    S = np.full((len(t), 3, 3), np.nan)
    for i, x in enumerate(X):
        P = x[~np.isnan(x[:, 0])]
        if len(P) >= 3 and (res := identify(P)) is not None:
            S[i] = res
    bad = reject_jumps(S, 9)
    S[bad] = np.nan
    ok = np.isfinite(S[:, 0, 0])
    print(f"[MoCap] {len(t)} frames（{t[0]:.1f}–{t[-1]:.1f} s），S1〜S3 識別 {ok.sum()} frames"
          f"（飛びとして除外 {bad.sum()}）")
    return dict(t=t, S=S, meta=meta, start=parse_capture_start(meta["Capture Start Time"]),
                fps=float(meta["Export Frame Rate"]))


# ════════════════════════════════════════════════════════════════════════════
#  δ の推定
# ════════════════════════════════════════════════════════════════════════════
def interp_nan(tq, t, y, max_gap):
    """欠損を max_gap 以上またいで補間しない線形補間．y: (n, ...) → (len(tq), ...)"""
    y2 = y.reshape(len(t), -1)
    ok = np.isfinite(y2).all(axis=1)
    out = np.full((len(tq), y2.shape[1]), np.nan)
    if ok.sum() < 2:
        return out.reshape((len(tq),) + y.shape[1:])
    tv, yv = t[ok], y2[ok]
    i = np.clip(np.searchsorted(tv, tq), 1, len(tv) - 1)
    t0, t1 = tv[i - 1], tv[i]
    w = ((tq - t0) / np.where(t1 > t0, t1 - t0, 1))[:, None]
    good = (t1 - t0 <= max_gap) & (tq >= tv[0]) & (tq <= tv[-1])
    out[good] = (yv[i - 1] * (1 - w) + yv[i] * w)[good]
    return out.reshape((len(tq),) + y.shape[1:])


def speed(t_grid, P):
    """一定間隔の位置 (n,3) → 平滑化した速さ [mm/s]"""
    v = np.linalg.norm(np.gradient(P, t_grid, axis=0), axis=1)
    k = max(1, int(round(SMOOTH * GRID_HZ)))
    return pd.Series(v).rolling(k, center=True, min_periods=1).mean().to_numpy() if k > 1 else v


def kabsch(A, B):                         # B ≈ T·A となる 4x4
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cb - R @ ca
    return T


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def split_segments(t):
    if len(t) == 0:
        return []
    cut = np.where(np.diff(t) > SEG_GAP)[0]
    edges = np.r_[0, cut + 1, len(t)]
    return [np.arange(a, b) for a, b in zip(edges[:-1], edges[1:])]


def sync_segment(idx, rs, mc, T_MC, c_range):
    """1 区間について c（Take 時刻 = RS 時刻 + c）を求める．"""
    P_cam = np.einsum("nij,kj->nki", rs["R"][idx], PB) + rs["tvec"][idx][:, None]   # (n, 3, 3) カメラ座標
    keep = ~reject_jumps(P_cam, 5)
    idx, P_cam = idx[keep], P_cam[keep]
    t_rs = rs["t"][idx]
    cen = P_cam.mean(1)

    # ── 共通の時間軸で速さを作る（両方を同じ処理にして遅れの差を作らない）──
    g = np.arange(t_rs[0], t_rs[-1], 1 / GRID_HZ)
    v_rs = speed(g, interp_nan(g, t_rs, cen, 0.1))
    tm = mc["t"]
    gm = np.arange(tm[0], tm[-1], 1 / GRID_HZ)
    v_mc = speed(gm, interp_nan(gm, tm, mc["S"].mean(1), 0.05))

    # ── 粗い探索：速さの相関 ──
    lags = np.arange(c_range[0], c_range[1], 1 / GRID_HZ)
    corr = np.full(len(lags), np.nan)
    ok_rs = np.isfinite(v_rs)
    for k, c in enumerate(lags):
        q = np.interp(g + c, gm, v_mc, left=np.nan, right=np.nan)
        m = ok_rs & np.isfinite(q)
        if m.sum() > 0.8 * ok_rs.sum() and m.sum() > 20:
            a, b = v_rs[m], q[m]
            if a.std() > 0 and b.std() > 0:
                corr[k] = np.corrcoef(a, b)[0, 1]
    if np.all(np.isnan(corr)):
        return None
    kb = int(np.nanargmax(corr))
    c0 = lags[kb]
    far = np.abs(lags - c0) > 0.5
    second = np.nanmax(corr[far]) if np.any(far & np.isfinite(corr)) else np.nan

    # ── 細かい探索：3 次元位置の RMS ──
    cs = np.arange(c0 - FINE_RANGE, c0 + FINE_RANGE + FINE_STEP / 2, FINE_STEP)
    rms = np.full(len(cs), np.nan)
    gap = 2.5 / mc["fps"]
    for k, c in enumerate(cs):
        Q = interp_nan(t_rs + c, tm, mc["S"], gap)          # (n, 3, 3) Motive 座標
        m = np.isfinite(Q[:, 0, 0])
        if m.sum() < 10:
            continue
        A, B = P_cam[m].reshape(-1, 3), Q[m].reshape(-1, 3)
        T = T_MC if T_MC is not None else kabsch(A, B)
        rms[k] = np.sqrt(np.mean(np.sum((apply(T, A) - B) ** 2, axis=1)))
    if np.all(np.isnan(rms)):
        return None
    kf = int(np.nanargmin(rms))
    c1 = cs[kf]
    if 0 < kf < len(cs) - 1 and np.all(np.isfinite(rms[kf - 1:kf + 2])):
        y0, y1, y2 = rms[kf - 1:kf + 2] ** 2                   # 二乗誤差に放物線を当てる
        den = y0 - 2 * y1 + y2
        if den > 0:
            c1 += 0.5 * (y0 - y2) / den * FINE_STEP
    at_edge = kf in (0, len(cs) - 1)

    # ── 最良の c での残差 ──
    Q = interp_nan(t_rs + c1, tm, mc["S"], gap)
    m = np.isfinite(Q[:, 0, 0])
    T = T_MC if T_MC is not None else kabsch(P_cam[m].reshape(-1, 3), Q[m].reshape(-1, 3))
    P_m = apply(T, P_cam.reshape(-1, 3)).reshape(P_cam.shape)
    e = np.linalg.norm(P_m - Q, axis=2)[m]                    # (n_ok, 3)
    bias = np.nanmean((P_m - Q)[m].reshape(-1, 3), axis=0)
    path = float(np.nansum(np.linalg.norm(np.diff(cen, axis=0), axis=1)))

    return dict(c_coarse=float(c0), c=float(c1), corr=float(corr[kb]), corr_second=float(second),
                rms_mm=float(np.sqrt(np.mean(e ** 2))), median_mm=float(np.median(e)),
                p95_mm=float(np.percentile(e, 95)), bias_mm=bias.tolist(),
                n_rs=int(len(idx)), n_used=int(m.sum()), path_mm=path, at_edge=bool(at_edge),
                rs_t0=float(t_rs[0]), rs_t1=float(t_rs[-1]),
                _plot=dict(lags=lags, corr=corr, cs=cs, rms=rms, t_rs=t_rs, P_m=P_m, Q=Q,
                           tm=tm, S=mc["S"]))


def load_tmc(path):
    if path is None:
        return None
    j = json.loads(Path(path).read_bytes().decode("utf-8", errors="replace"))
    if isinstance(j, dict):
        for key in ("T_MC", "T_mocap_cam", "T"):
            if key in j:
                j = j[key]
                break
    T = np.array(j, float)
    assert T.shape == (4, 4), f"T_MC が 4x4 ではありません: {T.shape}"
    return T


# ════════════════════════════════════════════════════════════════════════════
#  メイン
# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="RealSense と Motive の時刻ずれをボードの動きから求める")
    ap.add_argument("--rs", required=True, help="session.db3 か，その記録フォルダ")
    ap.add_argument("--csv", required=True, help="Motive の書き出し CSV")
    ap.add_argument("--tmc", help="T_MC.json（省略可）")
    ap.add_argument("--out", help="出力先（既定: 記録フォルダ/sync_check）")
    ap.add_argument("--redetect", action="store_true", help="ボード検出をやり直す")
    ap.add_argument("--stride", type=int, default=1, help="n フレームに 1 枚だけ検出（速くしたいとき）")
    args = ap.parse_args()

    rs_dir = Path(args.rs) if Path(args.rs).is_dir() else Path(args.rs).parent
    out = Path(args.out) if args.out else rs_dir / "sync_check"
    out.mkdir(parents=True, exist_ok=True)

    rs = load_rs(args.rs, out / "rs_board.npz", args.redetect, args.stride)
    if len(rs["t"]) < MIN_SEG_FRAMES:
        raise SystemExit("ボードがほとんど検出されていません．露光・ボードの向き・画角を確認してください．")
    T_MC = load_tmc(args.tmc)

    head = read_header(args.csv)
    meta = dict(zip(head[0][0::2], head[0][1::2]))
    start = parse_capture_start(meta["Capture Start Time"])
    if rs["epoch"]:
        prior = -start                                         # 時計が合っていれば c ≈ −Capture Start
        window = (rs["t"][0] + prior - SEARCH - 2, rs["t"][-1] + prior + SEARCH + 2)
        print(f"[時計] 最初にボードを検出したのは Take の {rs['t'][0] - start:.1f} s 付近（2 台の時計が合っていれば）")
    else:
        prior, window = None, None
    mc = load_mocap(args.csv, window)

    # ── 同期に使う区間を選ぶ ──
    segs = []
    for idx in split_segments(rs["t"]):
        if len(idx) < MIN_SEG_FRAMES:
            continue
        cen = (np.einsum("nij,kj->nki", rs["R"][idx], PB) + rs["tvec"][idx][:, None]).mean(1)
        path = np.nansum(np.linalg.norm(np.diff(cen, axis=0), axis=1))
        if path < MIN_PATH_MM:
            print(f"[skip] {rs['t'][idx[0]] - rs['t'][0]:.1f} s 付近の検出区間はボードがほとんど動いていません")
            continue
        segs.append(idx)

    # ── 1 回目: 各区間を広い範囲で探す ──
    first = {}
    for k, idx in enumerate(segs):
        t_rs = rs["t"][idx]
        c_range = (prior - SEARCH, prior + SEARCH) if rs["epoch"] else (mc["t"][0] - t_rs[0], mc["t"][-1] - t_rs[-1])
        r = sync_segment(idx, rs, mc, T_MC, c_range)
        if r is None:
            print(f"[skip] {t_rs[0] - rs['t'][0]:.1f} s 付近: MoCap 側と重なる区間がありません")
            continue
        first[k] = r

    # ── 2 回目: 相関のピークが最もはっきりした区間を基準にし，ほかの区間はその近くだけで探す ──
    #    （同じような動きを 2 回すると，別の区間の動きと取り違えることがあるため）
    results = []
    if first:
        anchor = max(first, key=lambda k: first[k]["corr"] - np.nan_to_num(first[k]["corr_second"]))
        a = first[anchor]
        for k in sorted(first):
            r = first[k]
            if k != anchor:
                r2 = sync_segment(segs[k], rs, mc, T_MC, (a["c"] - NEXT_SEARCH, a["c"] + NEXT_SEARCH))
                if r2 is not None:
                    if abs(r2["c"] - r["c"]) > 0.05:
                        print(f"[info] 区間 {len(results)} は単独だと {r['c'] - a['c']:+.2f} s ずれた所が最大でした"
                              "（似た動きとの取り違え）．基準区間の近くで探し直しました．")
                    r = r2
                r["constrained"] = True
            else:
                r["constrained"] = False
            r["anchor"] = k == anchor
            r["delta"] = r["c"] + start if rs["epoch"] else None
            results.append(r)

    if not results:
        raise SystemExit("同期に使える区間がありませんでした．")

    # ── 表示 ──
    print("\n========== 結果 ==========")
    print("  δ: Motive 時刻 = RealSense 時刻 + δ   /  Take 時刻 = RealSense 時刻 + δ − Capture Start")
    for k, r in enumerate(results):
        rel = r["rs_t0"] - rs["t"][0]
        d = f"δ = {r['delta']:+.4f} s" if r["delta"] is not None else f"c = {r['c']:+.4f} s"
        print(f"[区間 {k}] 記録の {rel:5.1f}–{r['rs_t1'] - rs['t'][0]:5.1f} s  {d}  "
              f"（Take の {r['rs_t0'] + r['c']:.2f} s〜）")
        print(f"          速さの相関 {r['corr']:.3f}（次点 {r['corr_second']:.3f}）  "
              f"位置の残差 RMS {r['rms_mm']:.2f} mm（中央値 {r['median_mm']:.2f}, 95% {r['p95_mm']:.2f}）  "
              f"使用 {r['n_used']}/{r['n_rs']} frames  移動 {r['path_mm']:.0f} mm")
        print(f"          平均のずれ（Motive 座標）{np.round(r['bias_mm'], 2)} mm")
        if r["anchor"] and r["corr"] - np.nan_to_num(r["corr_second"]) < 0.15:
            print("          [WARN] 相関のピークがはっきりしません．ボードをもっと不規則に大きく動かしてください．")
        if r["at_edge"]:
            print("          [WARN] 細かい探索の端で最小になりました．FINE_RANGE を広げてください．")
    if len(results) >= 2:
        a, b = results[0], results[-1]
        drift = (b["c"] - a["c"]) * 1000
        span = b["rs_t0"] - a["rs_t0"]
        print(f"\n  最初と最後の区間の差: {drift:+.1f} ms（{span:.0f} s 間）  ← 1 フレーム（60 fps で 16.7 ms）以内なら OK")

    # ── 保存 ──
    save = dict(
        rs=str(args.rs), csv=str(args.csv), tmc=args.tmc, capture_start=meta["Capture Start Time"],
        capture_start_epoch=start, rs_epoch=rs["epoch"],
        note="Motive 時刻 = RealSense 時刻 + delta / Take 時刻 = RealSense 時刻 + c（c = delta - capture_start_epoch）",
        segments=[{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    if len(results) >= 2:
        save["drift_ms"] = (results[-1]["c"] - results[0]["c"]) * 1000
    (out / "sync_result.json").write_text(json.dumps(save, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 図 ──
    n = len(results)
    fig, ax = plt.subplots(n, 3, figsize=(17, 4 * n), squeeze=False)
    for k, r in enumerate(results):
        p = r["_plot"]
        off = start if rs["epoch"] else 0.0
        ax[k, 0].plot(p["lags"] + off, p["corr"], lw=0.8)
        ax[k, 0].axvline(r["c_coarse"] + off, color="r", lw=0.8)
        ax[k, 0].set_xlabel("δ [s]" if rs["epoch"] else "c [s]")
        ax[k, 0].set_ylabel("速さの相関")
        ax[k, 0].set_title(f"区間 {k}: 粗い探索")
        ax[k, 1].plot((p["cs"] - r["c"]) * 1000, p["rms"], lw=1)
        ax[k, 1].axvline(0, color="r", lw=0.8)
        ax[k, 1].set_xlabel("最良値からのずれ [ms]")
        ax[k, 1].set_ylabel("位置の RMS [mm]")
        ax[k, 1].set_title(f"細かい探索: 最小 {r['rms_mm']:.2f} mm")
        tt = p["t_rs"] + r["c"]
        sel = (p["tm"] >= tt[0] - 1) & (p["tm"] <= tt[-1] + 1)
        with np.errstate(all="ignore"):
            cm = p["S"][sel].mean(axis=1)                    # 3 点そろったフレームだけ
        for j, col in enumerate("rgb"):
            ax[k, 2].plot(p["tm"][sel], cm[:, j], color=col, lw=0.8)
            ax[k, 2].plot(tt, np.nanmean(p["P_m"], axis=1)[:, j], ".", color=col, ms=2)
        ax[k, 2].set_xlabel("Take 時刻 [s]")
        ax[k, 2].set_ylabel("ボード中心 X/Y/Z [mm]")
        ax[k, 2].set_title("合わせた後（線: MoCap，点: RealSense）")
    fig.tight_layout()
    fig.savefig(out / "sync_check.png", dpi=110)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()