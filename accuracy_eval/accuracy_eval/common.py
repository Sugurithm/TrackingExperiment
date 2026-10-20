"""精度評価プログラム群の共通部分（読込・時刻同期・座標変換・前処理・区間検出）。

入力（試行フォルダから自動で探す。--rs などで個別指定も可）
  joints/rs_joints.npz   RealSense + MediaPipe（カメラ座標 [mm]）
  joints/mc_joints.npz   MoCap 関節（Motive 座標 [mm]，t = Take 時刻）
  sync_check/sync_result.json   Take 時刻 = RS 時刻 + c（区間ごと）
  T_MC.json              較正（T_CM: Motive → カメラ）

方針
  * 比較はすべて「カメラ座標」で行う（MoCap を T_CM でカメラ座標に移す）。
    こうすると誤差を「視線方向（奥行き）」と「画像の横・縦」に分けられる。
  * c は sync の各区間の中央時刻で直線補間（時計のずれの変化 = drift を吸収）。
  * 同期に使ったボードの区間 ±1 s は解析から除く。
"""
import argparse, glob, json, os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _swv
from scipy.signal import butter, filtfilt

# RS の関節名 → 比較する MoCap の関節名（None = 真値なし）
PAIRS = {"shoulder": "shoulder", "elbow": "elbow", "wrist": "wrist",
         "wrist_pose": "wrist", "mcp3": None}
LABEL = {"shoulder": "肩", "elbow": "肘", "wrist": "手首(hand)",
         "wrist_pose": "手首(pose)", "mcp3": "MCP3"}
COLOR = {"shoulder": "#2a78d6", "elbow": "#eb6834", "wrist": "#1baf7a",
         "wrist_pose": "#eda100", "mcp3": "#e87ba4"}
AXES = ("ray", "h", "v")          # 視線方向（+ = RS が奥）、画像の横、画像の縦
AXLABEL = {"ray": "奥行き(視線)", "h": "横(画像)", "v": "縦(画像)"}

GROSS_MM = 100.0   # これ以上真値から離れたフレームは「大外れ（誤検出）」として別扱い
FC_LP = 5.0        # 低域通過のカットオフ [Hz]
HAMPEL_K = 4       # 窓の片側（9 フレーム）
HAMPEL_N = 3.0     # 3·MAD


# ------------------------------------------------------------------ 引数・読込
def parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("trial", nargs="?", default=".", help="試行フォルダ（trial_*）")
    p.add_argument("--rs"); p.add_argument("--mc")
    p.add_argument("--sync"); p.add_argument("--tmc")
    p.add_argument("--meta", help="meta.json（合図音の区間 \"cue\" を使う。既定: <trial>/meta.json があれば使う）")
    p.add_argument("--out", help="出力先（既定: <trial>/eval）")
    return p


def _find(trial, given, patterns):
    if given:
        return given
    for pat in patterns:
        hit = sorted(glob.glob(os.path.join(trial, pat), recursive=True))
        if hit:
            return hit[0]
    raise FileNotFoundError(f"{patterns} が {trial} に見つかりません（オプションで指定してください）")


def _find_tmc(trial, given, sync):
    if given:
        return given
    for pat in ("T_MC.json", "*/T_MC.json"):
        hit = glob.glob(os.path.join(trial, pat))
        if hit:
            return hit[0]
    calib = sync.get("calib")
    if calib:  # recordings/trial_x と同じ階層の recordings/calib_x/T_MC.json
        cand = os.path.join(os.path.dirname(os.path.abspath(trial)), calib, "T_MC.json")
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("T_MC.json が見つかりません（--tmc で指定してください）")


class Trial:
    """1 試行ぶんのデータ。MoCap は RS の各フレーム時刻に補間済み（カメラ座標）。"""

    def __init__(self, args):
        tr = args.trial
        self.f_rs = _find(tr, args.rs, ["joints/rs_joints.npz", "**/rs_joints.npz"])
        self.f_mc = _find(tr, args.mc, ["joints/mc_joints.npz", "**/mc_joints.npz"])
        self.f_sync = _find(tr, args.sync, ["sync_check/sync_result.json", "**/sync_result.json"])
        self.sync = json.load(open(self.f_sync, encoding="utf-8"))
        self.f_tmc = _find_tmc(tr, args.tmc, self.sync)
        self.calib = json.load(open(self.f_tmc, encoding="utf-8"))
        self.out = args.out or os.path.join(tr, "eval")
        os.makedirs(self.out, exist_ok=True)
        self.f_meta = getattr(args, "meta", None) or (os.path.join(tr, "meta.json") if os.path.exists(os.path.join(tr, "meta.json")) else None)
        self.meta = json.load(open(self.f_meta, encoding="utf-8")) if self.f_meta else {}

        r = np.load(self.f_rs, allow_pickle=False)
        # 左右の判定（rs_joints が保存していれば）。反対側の腕の推定値も保存されている
        self.P_other = r["P_C_left"].astype(float) if "P_C_left" in r.files else None
        self.lr_swapped = r["lr_swapped"].astype(bool) if "lr_swapped" in r.files else None
        m = np.load(self.f_mc, allow_pickle=False)
        self.rs_names = [str(n) for n in r["names"]]
        self.t_rs = r["t"].astype(float)                     # epoch [s]
        self.P_raw = r["P_C"].astype(float)                  # (N,J,3) カメラ座標 [mm]
        self.uv = r["uv"].astype(float) if "uv" in r.files else None
        self.status = r["status"] if "status" in r.files else None
        self.status_names = [str(s) for s in r["status_names"]] if "status_names" in r.files else None
        self.fs = 1.0 / np.median(np.diff(self.t_rs))

        self.t_mc = m["t"].astype(float)                     # Take 時刻 [s]
        self.mc_names = [str(n) for n in m["names"]]
        M = m["P_M"].astype(float)
        T_CM = np.array(self.calib["T_CM"])
        self.M_cam_native = M @ T_CM[:3, :3].T + T_CM[:3, 3]  # MoCap をカメラ座標へ（120 fps のまま）
        self.fs_mc = float(m["fps"]) if "fps" in m.files else 1 / np.median(np.diff(self.t_mc))

        # --- 時刻: Take 時刻 = RS 時刻 + c(t)。c は区間中央で直線補間
        segs = self.sync["segments"]
        mids = np.array([(s["rs_t0"] + s["rs_t1"]) / 2 for s in segs])
        cs = np.array([s["c"] for s in segs])
        if len(segs) == 1:
            c = np.full_like(self.t_rs, cs[0])
        else:
            o = np.argsort(mids); mids, cs = mids[o], cs[o]
            slope = (cs[-1] - cs[0]) / (mids[-1] - mids[0])
            c = cs[0] + slope * (self.t_rs - mids[0])          # 両端の外側も同じ傾きで延長
        self.take = self.t_rs + c                              # RS 各フレームの Take 時刻
        self.epoch_to_take = lambda e: e + np.interp(e, self.t_rs, c)

        # --- 合図音の区間（meta.json の cue）→ Take 時刻の [(t0, t1, 名前)]
        self.cue = []
        cue = self.meta.get("cue") if self.meta else None
        if cue:
            for a, b, name in cue["segments"]:
                self.cue.append((float(self.epoch_to_take(cue["start_epoch"] + a)),
                                 float(self.epoch_to_take(cue["start_epoch"] + b)), str(name)))
        self.t = self.take                                     # 以下、時刻はすべて Take 時刻
        self.sync_windows = [(s["rs_t0"] + s["c"] - 1.0, s["rs_t1"] + s["c"] + 1.0) for s in segs]

        # --- MoCap を RS の時刻へ（線形補間。欠損は NaN のまま）
        self.M_raw = {}
        for j, n in enumerate(self.mc_names):
            self.M_raw[n] = _interp_nan(self.t_mc, self.M_cam_native[:, j], self.take)

        # --- 解析範囲: 最初の同期区間の後 〜 次の同期区間の前（なければ最後まで）
        sw = sorted(self.sync_windows)
        a = sw[0][1]
        b = sw[1][0] if len(sw) > 1 else self.t[-1]
        self.span = (a, b)
        self.in_span = (self.t > a) & (self.t < b)

        # --- 大外れ（status は OK なのに真値から GROSS_MM 以上離れる＝誤検出）は除外して記録
        #     そのうち「反対側の腕の推定値なら合っていた」フレームは左右の取り違えとして別に記録する（診断用）
        self.gross = {}; self.gross_lr = {}
        for j, n in enumerate(self.rs_names):
            ref = self.ref(n, filtered=False)
            g = np.zeros(len(self.t), bool); s = np.zeros(len(self.t), bool)
            if ref is not None:
                with np.errstate(invalid="ignore"):
                    g = np.linalg.norm(self.P_raw[:, j] - ref, axis=1) > GROSS_MM
                    if self.P_other is not None:
                        s = g & (np.linalg.norm(self.P_other[:, j] - ref, axis=1) < GROSS_MM)
            self.gross[n] = g; self.gross_lr[n] = s
        # 肘・手首が大外れのフレームは骨格ごと取り違えている → 真値のない関節（MCP3 など）も除外
        any_g = np.zeros(len(self.t), bool)
        for n in self.gross: any_g |= self.gross[n]
        for n in self.rs_names:
            if self.ref(n, filtered=False) is None: self.gross[n] = any_g & ~np.isnan(self.P_raw[:, self.rs_names.index(n)][:, 0])

        # --- 前処理済み（RS: Hampel + 5 Hz，MoCap: 5 Hz）。Praw は大外れを NaN にした生データ
        self.P = {}; self.Praw = {}
        for j, n in enumerate(self.rs_names):
            X = self.P_raw[:, j].copy(); X[self.gross[n]] = np.nan
            self.Praw[n] = X
            self.P[n] = lowpass(hampel(X), self.fs)
        self.M = {n: lowpass(self.M_raw[n], self.fs) for n in self.mc_names}

        self.rest, self.blocks = detect_segments(self)
        self.labels, self.label_source = self._labels()

    def _labels(self):
        """各フレームの区間名。meta.json の cue（合図音）があればそれ，なければ自動検出の区間から作る。"""
        lab = np.full(len(self.t), "(other)", dtype=object)
        # フェーズ（番号付き）: meta.json の cue → 合図音の予定表を MoCap に合わせる → なし
        import phases as PH
        res = PH.from_trial(self)
        self.phases, self.phase_info = (res[0], res[2]) if res else ([], {})
        self.phase_source = res[1] if res else None
        if self.phases:
            for t0, t1, name, _ in self.phases:
                lab[(self.t >= t0) & (self.t < t1)] = name
            return lab, self.phase_source
        for a, b in self.sync_windows: lab[(self.t >= a) & (self.t <= b)] = "board"
        for a, b in self.rest: lab[(self.t >= a) & (self.t <= b)] = "still"
        for bl in self.blocks:
            lab[(self.t >= bl["t0"]) & (self.t <= bl["t1"])] = f"bpm{int(round(bl['bpm_oneway'] / 10) * 10)}"
        return lab, "自動検出（meta.json の cue なし）"

    # 真値（RS 時刻・カメラ座標）
    def ref(self, rs_name, filtered=True):
        mn = PAIRS.get(rs_name)
        if mn is None or mn not in self.mc_names:
            return None
        return (self.M if filtered else self.M_raw)[mn]

    def rs(self, rs_name, filtered=True):
        return (self.P if filtered else self.Praw)[rs_name]

    def mask(self, seg):
        return (self.t >= seg[0]) & (self.t <= seg[1]) & self.in_span

    def compared(self):
        return [n for n in self.rs_names if self.ref(n) is not None]

    def save_json(self, name, obj):
        path = os.path.join(self.out, name)
        json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=_jsonable)
        return path

    def header(self):
        return {"rs": self.f_rs, "mc": self.f_mc, "sync": self.f_sync, "T_MC": self.f_tmc,
                "fs_rs": self.fs, "span_take_s": self.span,
                "rest": self.rest, "blocks": self.blocks}


def _jsonable(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


# ------------------------------------------------------------------ 信号処理
def _interp_nan(t_src, X, t_dst):
    out = np.full((len(t_dst), X.shape[1]), np.nan)
    for k in range(X.shape[1]):
        g = ~np.isnan(X[:, k])
        if g.sum() < 2: continue
        out[:, k] = np.interp(t_dst, t_src[g], X[g, k], left=np.nan, right=np.nan)
        # 欠損が 0.1 s を超える区間は補間しない
        idx = np.searchsorted(t_src[g], t_dst)
        idx = np.clip(idx, 1, g.sum() - 1)
        gap = t_src[g][idx] - t_src[g][idx - 1]
        out[gap > 0.1, k] = np.nan
    return out


def fill_nan(X):
    """NaN を直線補間（フィルタ用）。元の NaN の位置は返り値 2 つ目。"""
    X = X.copy(); bad = np.isnan(X).any(axis=1); i = np.arange(len(X))
    if (~bad).sum() < 2: return X, bad
    for k in range(X.shape[1]):
        X[bad, k] = np.interp(i[bad], i[~bad], X[~bad, k])
    return X, bad


def hampel_flags(X, k=HAMPEL_K, n=HAMPEL_N, min_dev=0.0):
    """外れ値フラグ (N,) — どれかの軸が窓の中央値から max(n·MAD, min_dev) 以上離れたフレーム。"""
    Xf, bad = fill_nan(X)
    flags = np.zeros(len(X), bool); med_all = Xf.copy()
    for a in range(X.shape[1]):
        pad = np.pad(Xf[:, a], k, mode="edge")
        w = _swv(pad, 2 * k + 1)
        med = np.median(w, axis=1); mad = 1.4826 * np.median(np.abs(w - med[:, None]), axis=1)
        f = (np.abs(Xf[:, a] - med) > np.maximum(n * mad, min_dev)) & (mad > 0)
        flags |= f; med_all[:, a] = med
    return flags & ~bad, med_all


def hampel(X):
    flags, med = hampel_flags(X)
    Y = X.copy(); Y[flags] = med[flags]
    return Y


def lowpass(X, fs, fc=FC_LP):
    Xf, bad = fill_nan(X)
    if (~bad).sum() < 30: return X.copy()
    b, a = butter(4, fc / (fs / 2))
    Y = filtfilt(b, a, Xf, axis=0)
    Y[bad] = np.nan
    return Y


def speed(X, fs):
    return np.linalg.norm(np.gradient(X, axis=0), axis=1) * fs


# ------------------------------------------------------------------ 座標の分解
def ray_basis(P):
    """各フレームの (視線, 画像横, 画像縦) 単位ベクトル。P はカメラ座標の RS 位置。"""
    u = P / np.linalg.norm(P, axis=1, keepdims=True)
    ex = np.array([1.0, 0, 0])
    h = ex - (u @ ex)[:, None] * u; h /= np.linalg.norm(h, axis=1, keepdims=True)
    v = np.cross(u, h)
    return u, h, v


def decompose(E, P):
    """誤差 E (N,3) を (視線, 横, 縦) 成分 (N,3) に分解。"""
    u, h, v = ray_basis(P)
    return np.stack([np.sum(E * u, 1), np.sum(E * h, 1), np.sum(E * v, 1)], 1)


# ------------------------------------------------------------------ 区間検出
def _range_1s(t, X, win_s=1.0):
    """各フレーム中心 ±0.5 s の 3 軸の振れ幅（最大）。"""
    fs = 1 / np.median(np.diff(t)); w = max(3, int(round(win_s * fs)))
    Xf, bad = fill_nan(X)
    out = np.full(len(t), np.nan)
    for a in range(3):
        v = _swv(Xf[:, a], w); r = v.max(1) - v.min(1)
        cur = np.full(len(t), np.nan); cur[w // 2: w // 2 + len(r)] = r
        out = np.fmax(out, cur)
    out[bad] = np.nan
    return out


def _runs(mask):
    runs, st = [], None
    for i, m in enumerate(mask):
        if m and st is None: st = i
        if (not m) and st is not None: runs.append((st, i - 1)); st = None
    if st is not None: runs.append((st, len(mask) - 1))
    return runs


def detect_segments(tr, still_mm=4.0, move_mm=30.0, min_rest=2.0, min_block=5.0, merge_gap=3.0, trim=0.3):
    """MoCap の手首・肘から「静止区間」と「動作ブロック」を自動検出（Take 時刻）。"""
    t = tr.t
    W = tr.M_raw.get("wrist"); E = tr.M_raw.get("elbow")
    rW = _range_1s(t, W); rE = _range_1s(t, E) if E is not None else rW
    span = tr.in_span
    rest_m = (rW < still_mm) & (rE < still_mm) & span
    rest = []
    for a, b in _runs(rest_m):
        t0, t1 = t[a] + trim, t[b] - trim
        if t1 - t0 >= min_rest: rest.append((float(t0), float(t1)))
    move_m = (rW > move_mm) & span
    segs = [(t[a], t[b]) for a, b in _runs(move_m)]
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < merge_gap: merged[-1] = (merged[-1][0], s[1])
        else: merged.append(s)
    blocks = []
    for a, b in merged:
        if b - a < min_block: continue
        mk = (t >= a) & (t <= b)
        z = W[mk]; z = z[~np.isnan(z).any(1)]
        X = z - z.mean(0); ax = np.linalg.svd(X, full_matrices=False)[2][0]; s = X @ ax
        n = len(s) * 8; F = np.abs(np.fft.rfft(s * np.hanning(len(s)), n)); f = np.fft.rfftfreq(n, 1 / tr.fs)
        f0 = f[1:][F[1:].argmax()]
        vmax = np.nanpercentile(speed(tr.M["wrist"][mk], tr.fs), 99)
        blocks.append({"name": f"{f0 * 120:.0f}bpm", "t0": float(a), "t1": float(b), "f0_hz": float(f0),
                       "bpm_oneway": float(f0 * 120), "amp_mm": float(np.percentile(s, 97) - np.percentile(s, 3)),
                       "vmax_mps": float(vmax / 1000)})
    return rest, blocks


# ------------------------------------------------------------------ 図
def setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    want = ["Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic"]
    have = {f.name for f in fm.fontManager.ttflist}
    if not any(w in have for w in want):
        for f in fm.findSystemFonts():
            if "NotoSansCJK" in f or "ipaexg" in f.lower():
                try: fm.fontManager.addfont(f)
                except Exception: pass
        have = {f.name for f in fm.fontManager.ttflist}
    fam = [w for w in want if w in have] + ["DejaVu Sans"]
    plt.rcParams.update({"font.family": fam, "axes.grid": True, "grid.alpha": .3,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    return plt


def shade_segments(ax, tr, alpha=.08):
    for a, b in tr.rest: ax.axvspan(a, b, color="#888", alpha=alpha, lw=0)
    for bl in tr.blocks:
        ax.axvspan(bl["t0"], bl["t1"], color="#2a78d6", alpha=alpha, lw=0)
        ax.text((bl["t0"] + bl["t1"]) / 2, 1.0, bl["name"], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color="#555")


def md_table(rows, cols, fmt="{:.1f}"):
    """rows: list of dict → Markdown 表。"""
    cols = [c if len(c) == 3 else (c[0], c[1], fmt) for c in cols]   # (見出し, キー[, 書式])
    head = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        cells = []
        for _, k, f in cols:
            v = r.get(k, "")
            cells.append(f.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                         ("–" if isinstance(v, (float, np.floating)) else str(v)))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body
