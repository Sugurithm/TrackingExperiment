"""フェーズ別解析（p1〜p7）の共通部分。

フェーズ = 試行の区間（静止 → 20 bpm → 休み → 40 bpm → … → 120 bpm）。
  * meta.json の cue（合図音の区間）があればその名前と時刻をそのまま使う
    （ready / board / away / still / count / bpmNN / rest / tail）
  * なければ MoCap から自動検出（同期ボード = board，最初の静止 = still，ブロック間 = rest，動作 = bpmNN）
  * 各フェーズの前後 TRIM 秒は切り替わり（動き出し・止まり際）として除く
種類（kind）: static（still / rest / count），move（bpmNN），board，other（ready / away / tail など）
"""
import os
import numpy as np
from scipy.signal import find_peaks
import common as C

TRIM = 0.5          # フェーズの前後で除く秒数
KIND_JA = {"static": "静止", "move": "動作", "board": "ボード", "other": "その他"}


def _kind(name):
    if name.startswith("bpm"): return "move"
    if name.startswith(("still", "rest", "count")): return "static"
    if name.startswith("board"): return "board"
    return "other"


def _label(name):
    if name.startswith("bpm"): return f"{name[3:]} bpm"
    for k, ja in (("rest", "休み"), ("count", "カウント"), ("board", "ボード"), ("still", "静止")):
        if name.startswith(k) and name[len(k):].isdigit(): return ja + name[len(k):]
    return {"still": "静止", "count": "カウント", "board": "ボード", "ready": "準備", "away": "ボード片付け",
            "tail": "終わり"}.get(name, name)


def phases(tr, trim=TRIM):
    """[{name, label, kind, t0, t1, mask, bpm, f0_hz, vmax_mps}]（時刻順，Take 時刻）。"""
    raw = []
    if tr.cue:
        seen = {}
        for t0, t1, name in tr.cue:
            seen[name] = seen.get(name, 0) + 1
            nm = name if name.startswith("bpm") or seen[name] == 1 else f"{name}{seen[name]}"
            raw.append((nm, t0, t1))
        source = "cue（meta.json）"
    else:
        for i, (a, b) in enumerate(sorted(tr.sync_windows)):
            raw.append(("board" if i == 0 else f"board{i + 1}", a + 1.0, b - 1.0))
        first = tr.blocks[0]["t0"] if tr.blocks else np.inf
        k = 0
        for a, b in tr.rest:
            if b <= first: raw.append(("still", a, b))
            else:
                k += 1; raw.append((f"rest{k}", a, b))
        for bl in tr.blocks:
            raw.append((f"bpm{int(round(bl['bpm_oneway'] / 10) * 10)}", bl["t0"], bl["t1"]))
        source = "自動検出（MoCap）"
    raw.sort(key=lambda r: r[1])
    out = []
    for name, t0, t1 in raw:
        kind = _kind(name)
        a, b = t0 + trim, t1 - trim
        if b - a < 1.0: continue
        mk = (tr.t >= a) & (tr.t <= b)
        if mk.sum() < 30: continue
        ph = {"name": name, "label": _label(name), "kind": kind, "t0": float(a), "t1": float(b), "mask": mk,
              "bpm": int(name[3:]) if kind == "move" and name[3:].isdigit() else None}
        if kind == "move":
            W = tr.M["wrist"][mk]; ok = ~np.isnan(W).any(1)
            s = (W[ok] - W[ok].mean(0)) @ main_axis(W[ok])
            n = len(s) * 8; F = np.abs(np.fft.rfft(s * np.hanning(len(s)), n)); f = np.fft.rfftfreq(n, 1 / tr.fs)
            ph["f0_hz"] = float(f[1:][F[1:].argmax()])
            ph["vmax_mps"] = float(np.nanpercentile(C.speed(np.nan_to_num(tr.M["wrist"][mk]), tr.fs), 99) / 1000)
            ph["amp_mm"] = float(np.percentile(s, 97) - np.percentile(s, 3))
        out.append(ph)
    tr.phase_source = source
    return out


def outdir(tr):
    d = os.path.join(tr.out, "phases"); os.makedirs(d, exist_ok=True); return d


def main_axis(X):
    X = X[~np.isnan(X).any(1)]
    return np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0]


def pair(tr, n, ph, filtered=True, fill=False):
    """フェーズ内の RS・MoCap（カメラ座標）。fill=True なら NaN を直線補間して全フレーム返す。"""
    mk = ph["mask"]
    X = tr.rs(n, filtered)[mk]; R = tr.ref(n, filtered)[mk]; t = tr.t[mk]
    ok = ~(np.isnan(X).any(1) | np.isnan(R).any(1))
    if fill:
        return C.fill_nan(X)[0], C.fill_nan(R)[0], t, ok
    return X[ok], R[ok], t[ok], ok


def turnarounds(s, fs, min_frac=0.3):
    """主な方向の位置 s の折り返し（極大・極小）のインデックス。振幅の min_frac 以上の振れのみ。"""
    amp = np.percentile(s, 97) - np.percentile(s, 3)
    pk, _ = find_peaks(s, prominence=min_frac * amp, distance=int(0.15 * fs))
    tr_, _ = find_peaks(-s, prominence=min_frac * amp, distance=int(0.15 * fs))
    return np.sort(np.r_[pk, tr_])


def strokes(tr, ph):
    """MoCap 手首の折り返しで区切った片道ストローク [(i0, i1)]（フェーズ内のインデックス）と主な方向。"""
    W = C.fill_nan(tr.M["wrist"][ph["mask"]])[0]
    ax = main_axis(W); s = (W - W.mean(0)) @ ax
    ta = turnarounds(s, tr.fs)
    return [(int(a), int(b)) for a, b in zip(ta[:-1], ta[1:]) if b - a > 0.15 * tr.fs], ax


def sub_frame_peak(v, i):
    if 0 < i < len(v) - 1:
        d = v[i - 1] - 2 * v[i] + v[i + 1]
        return i + (0.5 * (v[i - 1] - v[i + 1]) / d if d != 0 else 0.0)
    return float(i)


def pivot(rows, key, joints, phs, fmt="{:.1f}", label_key="label"):
    """行 = 関節，列 = フェーズ の Markdown 表。"""
    names = [p["name"] for p in phs]
    tab = []
    for n in joints:
        d = {"j": C.LABEL.get(n, n)}
        for r in rows:
            if r["joint"] == n and r["phase"] in names: d[r["phase"]] = r.get(key, np.nan)
        tab.append(d)
    return C.md_table(tab, [("関節", "j")] + [(p["label"], p["name"], fmt) for p in phs])


def plot_by_phase(ax, rows, key, joints, phs, title, ylabel="", hline=None, ylim0=False):
    names = [p["name"] for p in phs]; xs = np.arange(len(names))
    for n in joints:
        y = [next((r.get(key, np.nan) for r in rows if r["joint"] == n and r["phase"] == nm), np.nan) for nm in names]
        y = np.array([np.nan if v is None else v for v in y], float)
        ax.plot(xs, y, "-o", color=C.COLOR.get(n, "#444"), label=C.LABEL.get(n, n), lw=2, ms=5)
    ax.set_xticks(xs); ax.set_xticklabels([p["label"] for p in phs], rotation=40, fontsize=8)
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel)
    if hline is not None: ax.axhline(hline, color="#888", lw=1)
    if ylim0: ax.set_ylim(bottom=0)


def header_md(tr, phs):
    md = f"フェーズの定義: {tr.phase_source}（前後 {TRIM} s を除く）\n\n"
    rows = []
    for p in phs:
        rows.append({"label": p["label"], "kind": KIND_JA[p["kind"]], "t": f"{p['t0']:.1f}–{p['t1']:.1f}",
                     "dur": p["t1"] - p["t0"], "f0": p.get("f0_hz", np.nan), "v": p.get("vmax_mps", np.nan),
                     "a": p.get("amp_mm", np.nan)})
    md += C.md_table(rows, [("フェーズ", "label"), ("種類", "kind"), ("Take 時刻 [s]", "t"), ("長さ [s]", "dur"),
                            ("基本周波数 [Hz]", "f0", "{:.3f}"), ("手首の最大の速さ [m/s]", "v", "{:.2f}"),
                            ("振幅 [mm]", "a", "{:.0f}")])
    return md


def save(tr, name, obj):
    import json
    path = os.path.join(outdir(tr), name)
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=C._jsonable)
    return path


def strip_masks(phs):
    return [{k: v for k, v in p.items() if k != "mask"} for p in phs]
