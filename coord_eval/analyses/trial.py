"""試行ごとの指標（H1 全体の寄与 / H1 1 動作の寄与 / H2 マウスの向き / 補助）

すべて 1 試行（40 s、解析グリッド 60 Hz、机座標 [mm]）から計算する。値は試行ごとの行と、
1 動作ごとの行（movements）で返し、run.py が CSV にまとめる。
"""
import numpy as np

from .. import config as K
from ..lib import kin, sig
from ..lib.movements import segment, edge_mean, direction_class

P4 = K.PARTS
P3 = K.PARTS3
FS = K.FS_U


# ====================================================================== 点検
def xcorr_lag(a, b, max_s=0.2):
    """b が a に対して何 s 遅れているか（正規化相互相関の山を放物線で補間）。"""
    a = (a - a.mean()) / (a.std() + 1e-12); b = (b - b.mean()) / (b.std() + 1e-12)
    L = int(round(max_s * FS))
    lags = np.arange(-L, L + 1)
    r = np.array([np.mean(a[max(0, -k):len(a) - max(0, k)] * b[max(0, k):len(b) - max(0, -k)]) for k in lags])
    i = int(np.argmax(r))
    d = 0.0
    if 0 < i < len(r) - 1:
        den = r[i - 1] - 2 * r[i] + r[i + 1]
        d = 0.5 * (r[i - 1] - r[i + 1]) / den if den != 0 else 0.0
    return (lags[i] + d) / FS


def quality(tr):
    out = {"valid_wrist_pct": tr.valid_pct.get("wrist", np.nan),
           "valid_elbow_pct": tr.valid_pct.get("elbow", np.nan),
           "valid_shoulder_pct": tr.valid_pct.get("shoulder", np.nan),
           "valid_mcp3_pct": tr.valid_pct.get("mcp3", np.nan),
           "lr_swapped_pct": tr.lr_swapped_pct, "fs_rs": tr.fs_rs,
           "frame_gain_x": tr.frame.get("gain_x", np.nan), "frame_gain_y": tr.frame.get("gain_y", np.nan),
           "rms_err_mm": tr.rms_err_mm, "on_target_pct": tr.on_target_pct}
    # この試行での カメラ palm / マウス のゲインと遅れ（x 軸、速度の相互相関）
    pc = tr.J["palm"][:, 0]
    mx = tr.mouse[:, 0]
    ok = np.isfinite(pc) & np.isfinite(mx)
    if ok.sum() > 200:
        g = np.polyfit(mx[ok], pc[ok], 1)[0]
        a, _ = sig.fill_nan(np.gradient(pc)); b, _ = sig.fill_nan(np.gradient(mx))
        a = (a - a.mean()) / (a.std() + 1e-12); b = (b - b.mean()) / (b.std() + 1e-12)
        out["mouse_gain"] = float(g)
        out["mouse_lag_ms"] = float(xcorr_lag(b, a) * 1000)              # + = カメラが遅れる
    else:
        out["mouse_gain"] = out["mouse_lag_ms"] = np.nan
    flags = []
    lo, hi = K.GATE["mouse_gain"]
    if np.isfinite(out["mouse_gain"]) and not (lo <= out["mouse_gain"] <= hi):
        flags.append("gain")
    if out["valid_wrist_pct"] < K.GATE["valid_pct"]:
        flags.append("valid")
    if np.isfinite(out["lr_swapped_pct"]) and out["lr_swapped_pct"] > K.GATE["lr_swapped_pct"]:
        flags.append("lr")
    out["flags"] = ",".join(flags)
    return out


# ====================================================================== H1a 全体の寄与
def _amp(val, noi):
    """入力ビンの 3 軸の複素振幅とノイズから、ノイズを引いた振幅の 2 乗（ビンごと）。"""
    return np.maximum(np.nansum(np.abs(val) ** 2, axis=1) - np.nansum(noi, axis=1), 0.0)


def h1_global(tr, floor):
    parts, end, ok = kin.components(tr.J, P4 + ["distal"])
    bins = tr.bins()
    f = bins / tr.T
    if len(bins) == 0 or ok.mean() < 0.3:
        return {}
    out = {"valid_all_pct": float(100 * ok.mean())}
    V, N = {}, {}
    for p, X in list(parts.items()) + [("end", end)]:
        V[p], N[p] = sig.bin_read(sig.spectrum(X), bins)
    pw = {p: _amp(V[p], N[p]) for p in V}                     # (nb,) 変位の振幅²（ノイズを引いた値）
    lo = np.argsort(f)[: len(f) // 2]
    hi = np.argsort(f)[len(f) // 2:]
    for tag, parts_ in (("gshare_", P4), ("share3_", P3)):
        tot = sum(pw[p].sum() for p in parts_)
        for p in parts_:
            out[tag + p] = float(pw[p].sum() / tot) if tot > 0 else np.nan
        q = np.array([out[tag + p] for p in parts_])
        out["N_eff" if tag == "gshare_" else "N_eff3"] = float(1 / np.sum(q ** 2)) if tot > 0 else np.nan
    for tag, idx in (("lo", lo), ("hi", hi)):
        tot = sum(pw[p][idx].sum() for p in P4)
        for p in P4:
            out[f"gshare_{tag}_{p}"] = float(pw[p][idx].sum() / tot) if tot > 0 else np.nan
    # 速度の重み（旧 a1 と同じ定義。速い動きほど重くなる）
    w = (2 * np.pi * f) ** 2
    tot = sum((pw[p] * w).sum() for p in P4)
    for p in P4:
        out["vshare_" + p] = float((pw[p] * w).sum() / tot) if tot > 0 else np.nan
    out["gshare_prox"] = out["gshare_trunk"] + out["gshare_upperarm"]
    out["gshare_dist"] = out["gshare_forearm"] + out["gshare_hand"]

    # 手先への寄与（マウスを基準にした射影。カメラのノイズはマウスと無相関なので平均で消える）
    FM, _ = sig.bin_read(sig.spectrum(tr.mouse), bins)        # (nb, 2)
    for tag, parts_ in (("eshare_", P4), ("eshare3_", P3)):
        e = {p: float(np.nansum(np.real(V[p][:, :2] * np.conj(FM)))) for p in parts_}
        tot = sum(e.values())
        for p in parts_:
            out[tag + p] = e[p] / tot if tot != 0 else np.nan
    for k, ax in enumerate(("x", "y")):
        bx = np.isin(bins, tr.bins(ax))
        e = {p: float(np.nansum(np.real(V[p][bx, k] * np.conj(FM[bx, k])))) for p in P4}
        tot = sum(e.values())
        for p in P4:
            out[f"eshare_{ax}_{p}"] = e[p] / tot if tot != 0 else np.nan

    # 打ち消し合い（ノイズを引いた振幅で作る。1 = 素直な足し算、> 1 = 成分どうしが逆向きに動いて打ち消す）
    snr_end = np.sqrt(np.nansum(np.abs(V["end"]) ** 2, 1) / np.maximum(np.nansum(N["end"], 1), 1e-12))
    good = (snr_end >= K.SNR_MIN) & (pw["end"] > 0)
    for tag, parts_ in (("C_glob", P4), ("C_glob3", P3)):
        if good.sum() >= 2:
            c = sum(np.sqrt(pw[p][good]) for p in parts_) / np.sqrt(pw["end"][good])
            out[tag] = float(np.average(c, weights=np.sqrt(pw["end"][good])))
        else:
            out[tag] = np.nan
    out["n_bins_snr"] = int(good.sum())

    # 時間領域の振幅（ノイズ床を引いた RMS）
    for p, X in list(parts.items()) + [("end", end)]:
        Y = X[ok] - np.nanmean(X[ok], 0)
        r = float(np.sqrt(np.nanmean(np.sum(Y ** 2, 1))))
        fl = floor.get(p, np.nan)
        out[f"rms_{p}_mm"] = r
        out[f"rmsc_{p}_mm"] = float(np.sqrt(max(r ** 2 - fl ** 2, 0))) if np.isfinite(fl) else np.nan
        out[f"snr_{p}"] = r / fl if np.isfinite(fl) and fl > 0 else np.nan
    M = tr.mouse - np.nanmean(tr.mouse, 0)
    out["rms_mouse_mm"] = float(np.sqrt(np.nanmean(np.sum(M ** 2, 1))))

    # 近位 → 遠位の時間差（入力ビンでの位相差。+ = 遠位が遅れる = 近位 → 遠位の順）
    prox = parts["trunk"] + parts["upperarm"]                # = 肘の位置
    Vp, Np_ = sig.bin_read(sig.spectrum(prox), bins)
    lags, phs, wts, wph = [], [], [], []
    for k, ax in enumerate(("x", "y")):
        for i, b in enumerate(bins):
            if b not in tr.bins(ax):
                continue
            a_p, a_d = Vp[i, k], V["distal"][i, k]
            snr_p = np.abs(a_p) / np.sqrt(max(Np_[i, k], 1e-12))
            snr_d = np.abs(a_d) / np.sqrt(max(N["distal"][i, k], 1e-12))
            if snr_p >= K.SNR_MIN and snr_d >= K.SNR_MIN:
                dph = np.angle(a_p * np.conj(a_d))           # 遠位の遅れ（rad）
                if abs(dph) < np.pi / 2:                     # 逆相（打ち消し）は時間差として読まない
                    phs.append(np.degrees(dph)); wph.append(np.abs(a_p) * np.abs(a_d))
                    if f[i] >= 0.25:                         # 低い周波数では数度の位相差が数百 ms になるので使わない
                        lags.append(dph / (2 * np.pi * f[i]) * 1000); wts.append(np.abs(a_p) * np.abs(a_d))
    out["phase_glob_deg"] = float(np.average(phs, weights=wph)) if phs else np.nan
    out["lag_glob_ms"] = float(np.average(lags, weights=wts)) if lags else np.nan
    out["lag_glob_nbins"] = len(lags)
    return out


# ====================================================================== H1b 1 動作の寄与
def h1_movement(tr, floor, ori):
    parts, end, ok = kin.components(tr.J, P4 + ["distal"])
    prox = parts["trunk"] + parts["upperarm"]
    moves = segment(tr.mouse)
    sig_p = {p: floor.get(p, np.nan) for p in P4 + ["distal", "end"]}
    sig_p["prox"] = np.sqrt(sig_p["trunk"] ** 2 + sig_p["upperarm"] ** 2) if np.isfinite(sig_p["trunk"]) else np.nan
    if not all(np.isfinite(sig_p[p]) for p in P4):
        # 静止試行が無い: 1 フレームごとの差のばらつきから粗く見積もる
        for p in P4 + ["distal", "end", "prox"]:
            X = (end if p == "end" else prox if p == "prox" else parts[p])
            d = np.diff(tr.Jraw["mcp3"] if p == "end" else X, axis=0)
            sig_p[p] = float(np.sqrt(np.nanmean(np.sum(d ** 2, 1)) / 2))
    rows = []
    for mv in moves:
        a, b = mv["i0"], mv["i1"]
        if not ok[a:b + 1].all():
            continue
        u = np.r_[mv["D"] / mv["amp"], 0.0]
        d = {p: edge_mean(parts[p], b) - edge_mean(parts[p], a) for p in P4 + ["distal"]}
        d["prox"] = edge_mean(prox, b) - edge_mean(prox, a)
        Dc = edge_mean(end, b) - edge_mean(end, a)
        along = float(Dc @ u)
        Dc_amp = float(np.sqrt(max(Dc @ Dc - 2 * sig_p["end"] ** 2, 1e-9)))
        r = {"i0": a, "i1": b, "t0": a / FS, "dur_s": mv["dur"], "amp_mm": mv["amp"], "dir_deg": mv["dir_deg"],
             "dir_class": direction_class(mv["dir_deg"]), "straight": mv["straight"], "vpeak": mv["vpeak"],
             "cam_amp_mm": Dc_amp, "cam_along_mm": along}
        valid = along > 3 * np.sqrt(2) * sig_p["end"]
        r["valid"] = bool(valid)
        exc = {}
        for p in P4 + ["distal", "prox"]:
            proj = float(d[p] @ u)
            r[f"proj_{p}_mm"] = proj
            e2 = float(d[p] @ d[p]) - 2 * sig_p[p] ** 2
            exc[p] = np.sqrt(max(e2, 0.0))
            r[f"exc_{p}_mm"] = exc[p]
            r[f"active_{p}"] = bool(np.sqrt(d[p] @ d[p]) > K.MOVE_ACTIVE_Z * np.sqrt(2) * sig_p[p])
        if valid:
            for p in P4:
                r["mshare_" + p] = r[f"proj_{p}_mm"] / along
            r["mshare_prox"] = r["mshare_trunk"] + r["mshare_upperarm"]
            r["mshare_dist"] = r["mshare_forearm"] + r["mshare_hand"]
            r["mshare3_distal"] = r["proj_distal_mm"] / along
            e = np.array([exc[p] for p in P4])
            # 動いた関節の実効数 = 1 / Σq²、q = 成分の変位² の割合（1 = 一関節だけ、4 = 全成分が同じだけ）
            r["N_move"] = float(np.sum(e ** 2) ** 2 / np.sum(e ** 4)) if np.sum(e ** 4) > 0 else np.nan
            r["C_move"] = float(e.sum() / Dc_amp)
            e3 = np.array([exc[p] for p in ("trunk", "upperarm", "distal")])
            r["C_move3"] = float(e3.sum() / Dc_amp)
            r["nondom"] = float(1 - e.max() / e.sum()) if e.sum() > 0 else np.nan
            r["n_active"] = int(sum(r[f"active_{p}"] for p in P4))
            # 時間差: 近位（肘の位置）と遠位（肘 → MCP3）の速度の山（動作の方向への成分）
            lo_, hi_ = max(0, a - 6), min(len(end) - 1, b + 6)
            pk = {}
            for p in ("prox", "distal"):
                X = prox if p == "prox" else parts["distal"]
                v = sig.grad(X[lo_:hi_ + 1], FS) @ u
                v = v if r[f"proj_{p}_mm"] >= 0 else -v
                if r[f"active_{p}"] and np.isfinite(v).all():
                    i = int(np.argmax(v))
                    dd = 0.0
                    if 0 < i < len(v) - 1 and (v[i - 1] - 2 * v[i] + v[i + 1]) != 0:
                        dd = 0.5 * (v[i - 1] - v[i + 1]) / (v[i - 1] - 2 * v[i] + v[i + 1])
                    pk[p] = (lo_ + i + dd - a) / FS
            r["lag_move_ms"] = (pk["distal"] - pk["prox"]) * 1000 if len(pk) == 2 else np.nan
            # 向きの変化
            for k in ("psi", "phi", "theta"):
                r[f"d{k}_deg"] = float(edge_mean(ori[k], b) - edge_mean(ori[k], a))
        rows.append(r)

    out = {"n_moves": len(moves), "n_moves_valid": int(sum(r["valid"] for r in rows))}
    V = [r for r in rows if r["valid"]]
    if len(V) < 5:
        return out, rows
    g = lambda k: np.array([r.get(k, np.nan) for r in V], float)
    amp = g("amp_mm")
    out["move_amp_med_mm"] = float(np.median(amp))
    out["move_dur_med_s"] = float(np.median(g("dur_s")))
    for p in P4 + ["prox", "dist"]:
        out["mshare_" + p] = float(np.nanmean(g("mshare_" + p)))
    out["mshare3_distal"] = float(np.nanmean(g("mshare3_distal")))
    for k in ("N_move", "C_move", "C_move3", "nondom", "n_active"):
        out[k] = float(np.nanmean(g(k)))
    out["lag_move_ms"] = float(np.nanmedian(g("lag_move_ms"))) if np.isfinite(g("lag_move_ms")).sum() >= 5 else np.nan
    # 振幅への依存（log2 振幅に対する傾き: 動作が 2 倍大きいと何ポイント変わるか）
    la = np.log2(amp)
    out["rslope_prox"] = float(np.polyfit(la, g("mshare_prox"), 1)[0]) if np.ptp(la) > 0.5 else np.nan
    out["rslope_hand"] = float(np.polyfit(la, g("mshare_hand"), 1)[0]) if np.ptp(la) > 0.5 else np.nan
    out["rslope_Nmove"] = float(np.polyfit(la, g("N_move"), 1)[0]) if np.ptp(la) > 0.5 else np.nan
    # 向きごと
    for dc in ("x", "y"):
        m = np.array([r["dir_class"] == dc for r in V])
        if m.sum() >= 5:
            for p in P4:
                out[f"mshare_{dc}_{p}"] = float(np.nanmean(g("mshare_" + p)[m]))
    # 1 動作ごとのマウスの向きの安定化（Δψ を Δθ に回帰。原点を通す）
    dps, dth = g("dpsi_deg"), g("dtheta_deg")
    okk = np.isfinite(dps) & np.isfinite(dth)
    if okk.sum() >= 5 and np.sum(dth[okk] ** 2) > 0:
        n2 = 2 * floor.get("theta", 0.0) ** 2
        den = max(np.sum(dth[okk] ** 2) - okk.sum() * n2, 0.25 * np.sum(dth[okk] ** 2))
        out["mK_orient"] = float(1 - np.sum(dps[okk] * dth[okk]) / den)
    return out, rows


# ====================================================================== H2 マウスの向き
def _band_reg(y, xs, bins):
    """入力ビンで y ≈ Σ c_i x_i（実数の係数）を最小二乗。返り値: (係数, R², 係数の SE)。"""
    Y, _ = sig.bin_read(sig.spectrum(y), bins)
    Xs = [sig.bin_read(sig.spectrum(x), bins)[0] for x in xs]
    A = np.column_stack([np.r_[x.real, x.imag] for x in Xs])
    yy = np.r_[Y.real, Y.imag]
    ok = np.isfinite(A).all(1) & np.isfinite(yy)
    if ok.sum() < len(xs) + 2:
        return np.full(len(xs), np.nan), np.nan
    c, *_ = np.linalg.lstsq(A[ok], yy[ok], rcond=None)
    res = yy[ok] - A[ok] @ c
    r2 = 1 - np.sum(res ** 2) / np.sum(yy[ok] ** 2) if np.sum(yy[ok] ** 2) > 0 else np.nan
    return c, float(r2)


def sensor_yaw(tr, lag_s=0.0):
    """マウスのセンサ自身の向き（窓ごと）: マウスの座標系での速度とカメラで測った手（palm）の
    机の上の速度を 2 次元の回転で重ねたときの角度 [deg]（ψ と同じ向き: + = 右回り）。

    手のランドマークを使わずに「マウスがどれだけ回ったか」を測る（握りの点の取り方に依らない）。
    カメラの軸ごとのゲインの違いは、机座標を決めた回帰のゲインで割って直してから比べる。
    """
    vm = sig.grad(tr.mouse, FS)
    gx, gy = tr.frame.get("gain_x", 1.0), tr.frame.get("gain_y", 1.0)
    gx = gx if np.isfinite(gx) and gx > 0 else 1.0
    gy = gy if np.isfinite(gy) and gy > 0 else 1.0
    palm = tr.J["palm"][:, :2]
    if lag_s and np.isfinite(lag_s):                         # カメラの遅れを戻す（マウスと同じ時刻にそろえる）
        palm = np.column_stack([np.interp(tr.tg + lag_s, tr.tg, palm[:, k], left=np.nan, right=np.nan)
                                for k in range(2)])
    vc = sig.grad(palm, FS) / np.array([gx, gy])
    w, st = int(K.YAW_WIN_S * FS), int(K.YAW_STEP_S * FS)
    out = []
    for a in range(0, len(vm) - w, st):
        A, B = vm[a:a + w], vc[a:a + w]
        ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
        if ok.sum() < 0.8 * w:
            continue
        path = np.sum(np.linalg.norm(A[ok], axis=1)) / FS
        if path < K.YAW_MIN_PATH_MM:
            continue
        cross = np.sum(A[ok, 0] * B[ok, 1] - A[ok, 1] * B[ok, 0])
        dot = np.sum(np.sum(A[ok] * B[ok], 1))
        rho = np.arctan2(cross, dot)                         # マウス → 机 の回転（反時計回りが正）
        # 窓の角度の推定誤差: 1 サンプルごとの角度のずれのばらつき / √(独立なサンプル数)
        ci = A[ok, 0] * B[ok, 1] - A[ok, 1] * B[ok, 0]
        di = np.sum(A[ok] * B[ok], 1)
        wi = np.sqrt(ci ** 2 + di ** 2)
        dev = np.angle(np.exp(1j * (np.arctan2(ci, di) - rho)))
        # 独立なサンプル数はフレーム数とした（合成データで、ノイズだけの場合に実際の誤差と一致することを確認。
        # 実データのノイズが時間的に相関していれば誤差を小さく見積もる＝ノイズが残る側に偏る）
        n_ind = max(float(ok.sum()), 1.0)
        var = float(np.average(dev ** 2, weights=wi) / n_ind) if wi.sum() > 0 else np.nan
        out.append((a + w / 2, -np.degrees(rho), path, np.degrees(np.sqrt(var))))
    return np.array(out) if out else np.zeros((0, 4))


def h2_orientation(tr, floor, lag_s=0.0):
    ori = kin.orientation(tr.J)
    bins = tr.bins()
    out = {}
    for k, a in ori.items():
        v = a[np.isfinite(a)]
        if len(v) < 100:
            return out, ori
        sd = float(np.std(v))
        fl = floor.get(k, np.nan)
        out[f"{k}_sd_deg"] = sd
        out[f"{k}_sdc_deg"] = float(np.sqrt(max(sd ** 2 - fl ** 2, 0))) if np.isfinite(fl) else np.nan
        out[f"{k}_range_deg"] = float(np.percentile(v, 95) - np.percentile(v, 5))
        out[f"{k}_mean_deg"] = float(np.mean(v))
    # 向きの安定化: 手首が水平面内で θ 回ったとき、手（≒ マウス）の向き ψ は何度回るか。
    #   ψ = φ + θ なので dψ/dθ = 1 + dφ/dθ。K_orient = 1 − dψ/dθ
    #   K ≈ 0: マウスは手首と一緒に回る（向きが変わる）
    #   K ≈ 1: 前腕が逆に回って打ち消し、マウスの向きは一定のまま
    #   K < 0: 前腕も同じ向きに回り、マウスは手首以上に回る
    Th, Nt = sig.bin_read(sig.spectrum(ori["theta"]), bins)
    Ps, _ = sig.bin_read(sig.spectrum(ori["psi"]), bins)
    num = float(np.nansum(np.real(Ps * np.conj(Th))))
    den_raw = float(np.nansum(np.abs(Th) ** 2))
    den = max(den_raw - float(np.nansum(Nt)), 0.25 * den_raw)
    out["K_orient"] = 1 - num / den if den > 0 else np.nan
    snr_th = np.sqrt(den_raw / max(float(np.nansum(Nt)), 1e-12))
    out["theta_band_snr"] = float(snr_th)
    # 手の向きは位置で決まるか（ψ をマウスの位置 x, y に回帰。入力周波数のビンだけ）
    c, r2 = _band_reg(ori["psi"], [tr.mouse[:, 0], tr.mouse[:, 1]], bins)
    out["psi_per100mm_x"] = float(c[0] * 100)
    out["psi_per100mm_y"] = float(c[1] * 100)
    out["psi_pos_r2"] = r2
    # センサの向き
    sy = sensor_yaw(tr, lag_s)
    out["n_yaw_windows"] = len(sy)
    if len(sy) >= 10:
        ys = sy[:, 1]
        idx = sy[:, 0].astype(int)
        out["psis_sd_raw_deg"] = float(np.std(ys))
        # 窓ごとの推定誤差を引く（隣り合う窓は半分以上重なるので誤差の分散は窓の数で割らない）
        out["psis_noise_deg"] = float(np.sqrt(np.nanmean(sy[:, 3] ** 2)))
        out["psis_sd_deg"] = float(np.sqrt(max(np.var(ys) - out["psis_noise_deg"] ** 2, 0)))
        out["psis_range_deg"] = float(np.percentile(ys, 95) - np.percentile(ys, 5))
        mx = tr.mouse[idx, 0]
        my = tr.mouse[idx, 1]
        okk = np.isfinite(mx) & np.isfinite(my)
        if okk.sum() >= 10:
            A = np.column_stack([mx[okk], my[okk], np.ones(okk.sum())])
            cc, *_ = np.linalg.lstsq(A, ys[okk], rcond=None)
            out["psis_per100mm_x"] = float(cc[0] * 100)
            out["psis_per100mm_y"] = float(cc[1] * 100)
        # 手のランドマークの向き（同じ窓の平均）との一致
        w = int(K.YAW_WIN_S * FS)
        pm = np.array([np.nanmean(ori["psi"][max(0, i - w // 2): i + w // 2]) for i in idx])
        okk = np.isfinite(pm)
        if okk.sum() >= 10 and np.std(pm[okk]) > 0.5:
            out["psis_vs_psi_r"] = float(np.corrcoef(ys[okk], pm[okk])[0, 1])
            out["psis_vs_psi_slope"] = float(np.polyfit(pm[okk], ys[okk], 1)[0])
    return out, ori


# ====================================================================== 補助: 6 自由度（関節固定法）
def x_dof(tr):
    A, L = kin.dof6(tr.J)
    if A is None:
        return {}
    ok = np.isfinite(A).all(1)
    if ok.sum() < 200:
        return {}
    A = A[ok]
    Am = A.mean(0)
    full = np.sqrt(kin.from_dof(A, L).var(0).sum())
    out = {"dof_endpoint_mm": float(full)}
    tot = 0.0
    for j, d in enumerate(kin.DOF):
        B = np.broadcast_to(Am, A.shape).copy()
        B[:, j] = A[:, j]
        v = float(np.sqrt(kin.from_dof(B, L).var(0).sum()))
        out[f"dofmm_{d}"] = v
        out[f"dofsd_{d}"] = float(np.std(A[:, j]))
        tot += v ** 2
    for d in kin.DOF:
        out[f"dofshare_{d}"] = out[f"dofmm_{d}"] ** 2 / tot if tot > 0 else np.nan
    out["dof_cancel"] = float(np.sqrt(tot) / full) if full > 0 else np.nan
    return out


# ====================================================================== 補助: 追従の成績
def x_tracking(tr):
    bins = {ax: tr.bins(ax) for ax in ("x", "y")}
    out = {"rms_err_mm": tr.rms_err_mm, "on_target_pct": tr.on_target_pct}
    lags, gains, wts = [], [], []
    for k, ax in enumerate(("x", "y")):
        Tg, _ = sig.bin_read(sig.spectrum(tr.tgt[:, k]), bins[ax])
        Cu, _ = sig.bin_read(sig.spectrum(tr.cur[:, k]), bins[ax])
        H = Cu / Tg
        f = bins[ax] / tr.T
        ph = np.angle(H)
        lags += list(-ph / (2 * np.pi * f) * 1000)
        gains += list(np.abs(H)); wts += list(np.abs(Tg))
    out["track_gain"] = float(np.average(gains, weights=wts))
    out["track_lag_ms"] = float(np.average(lags, weights=wts))
    return out


# ====================================================================== まとめ
def compute(tr, floor, with_extra=True):
    """1 試行ぶんの全指標。(試行の行, 動作の行の一覧)"""
    row = dict(tr.header())
    row.update(quality(tr))
    row.update(h1_global(tr, floor))
    lag = row.get("mouse_lag_ms", np.nan) / 1000
    o, ori = h2_orientation(tr, floor, lag if np.isfinite(lag) and abs(lag) < 0.15 else 0.0)
    row.update(o)
    m, mrows = h1_movement(tr, floor, ori)
    row.update(m)
    # 向きの大きさを動きの大きさで割る（DPI を上げると動きが小さくなり角度も小さくなるため）
    rm = row.get("rms_mouse_mm", np.nan)
    for k in ("psi_sdc_deg", "theta_sdc_deg", "psis_sd_deg"):
        if k in row and np.isfinite(rm) and rm > 0:
            row[k.replace("_deg", "_per100")] = row[k] / rm * 100
    if with_extra:
        row.update(x_dof(tr))
        row.update(x_tracking(tr))
    mrows = [{**tr.header(), **r} for r in mrows]
    return row, mrows
