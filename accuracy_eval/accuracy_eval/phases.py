"""試行のフェーズ（合図音の区間）を求める。

優先順位
  1. meta.json の "cue"（記録時の合図音の開始時刻と区間）があればそのまま使う
  2. なければ合図音の予定表（experiment/config.py の CUE_* と同じ値）を作り，
     RS の記録開始を起点に ±3 s ずらして MoCap の手首の動きと最もよく合う位置に置く
     （bpmNN の区間で動いている・still / rest / count で止まっている割合が最大になるずれ）
     → このずれには「合図を聞いてから動き出すまでの遅れ」も含まれる
  3. 予定表の長さが記録と合わなければ None（common が自動検出に切り替える）

フェーズの名前: ready, board, away, still, count, bpmNN, rest, tail
各フェーズは出現順に番号を付けて区別する（rest#1, count#2 など）。
"""
import numpy as np

# experiment/config.py の CUE_* と同じ値（変えたらここも合わせる）
CUE_READY_S = 8.0
CUE_BOARD_S = 5.0
CUE_AWAY_S = 5.0
CUE_STILL_S = 10.0
CUE_COUNT = 3
CUE_BEATS = 16
CUE_REST_S = 4.0
CUE_BPMS = [20, 40, 60, 80, 100, 120]
CUE_TAIL_S = 3.0

MOVE_MM_S = 40.0        # 「動いている」の閾値（MoCap 手首の速さ）
SEARCH_S = 3.0          # 予定表をずらす範囲 ±[s]

STATIC = ("still", "rest", "count")       # 止まっているはずのフェーズ
TASK_STATIC = ("still", "rest")           # 解析で静止の代表として使うフェーズ（count は動き出す直前）
OFF_TASK = ("ready", "board", "away", "tail")


def schedule(bpms=None, beats=None):
    """lib/cue.py の schedule() と同じ: [(開始 s, 終了 s, 名前)]"""
    bpms = bpms or CUE_BPMS; beats = beats or CUE_BEATS
    segs, t = [], 0.0

    def add(d, n):
        nonlocal t
        segs.append((round(t, 3), round(t + d, 3), n)); t += d
    add(CUE_READY_S, "ready"); add(CUE_BOARD_S, "board"); add(CUE_AWAY_S, "away"); add(CUE_STILL_S, "still")
    for k, b in enumerate(bpms):
        if k: add(CUE_REST_S, "rest")
        add(CUE_COUNT, "count"); add(beats * 60.0 / b, f"bpm{b}")
    add(CUE_REST_S, "rest"); add(CUE_BOARD_S, "board"); add(CUE_TAIL_S, "tail")
    return segs


def number(segs):
    """同じ名前のフェーズに出現順の番号を付ける → [(t0, t1, 名前, 番号付きの名前)]"""
    cnt = {}; out = []
    for a, b, n in segs:
        cnt[n] = cnt.get(n, 0) + 1
        out.append((a, b, n, n if n.startswith("bpm") or n == "still" else f"{n}#{cnt[n]}"))
    return out


def from_trial(tr):
    """→ (フェーズ [(Take t0, Take t1, 名前, 番号付き)], 出どころの説明, 情報 dict) または None"""
    cue = tr.meta.get("cue") if getattr(tr, "meta", None) else None
    if cue:
        segs = [(cue["start_epoch"] + a, cue["start_epoch"] + b, n) for a, b, n in cue["segments"]]
        conv = lambda e: float(e + np.interp(e, tr.t_rs, tr.take - tr.t_rs))
        out = [(conv(a), conv(b), n, k) for a, b, n, k in number(segs)]
        return out, "meta.json の cue", {"offset_s": None}
    bpms = tr.meta.get("cue", {}).get("bpms") if tr.meta else None
    S = schedule(bpms)
    total = S[-1][1]; rec = tr.t_rs - tr.t_rs[0]
    if abs(rec[-1] - total) > 5.0:
        return None
    W = tr.M.get("wrist")
    if W is None: return None
    import common as C
    v = C.speed(np.nan_to_num(W), tr.fs); ok = ~np.isnan(W).any(1)
    mov = v > MOVE_MM_S
    best = None
    for o in np.arange(-SEARCH_S, SEARCH_S + 1e-9, 0.01):
        inb = np.zeros(len(rec), bool); st = np.zeros(len(rec), bool)
        for a, b, n in S:
            m = (rec >= a + o) & (rec < b + o) & ok
            if n.startswith("bpm"): inb |= m
            elif n in STATIC: st |= m
        if inb.sum() == 0 or st.sum() == 0: continue
        sc = mov[inb].mean() + (~mov[st]).mean()
        if best is None or sc > best[1]: best = (float(o), float(sc))
    if best is None or best[1] < 1.5:          # 合わない（別の予定表で記録した等）
        return None
    o = best[0]; t0 = tr.t_rs[0]
    conv = lambda r: float(np.interp(t0 + r + o, tr.t_rs, tr.take, left=np.nan, right=np.nan)) \
        if t0 + r + o <= tr.t_rs[-1] else float(tr.take[-1] + (t0 + r + o - tr.t_rs[-1]))
    out = [(conv(a), conv(b), n, k) for a, b, n, k in number(S)]
    info = {"offset_s": o, "score": best[1],
            "moving_pct": {k: float(100 * mov[(tr.take >= a) & (tr.take < b)].mean()) for a, b, n, k in out
                           if ((tr.take >= a) & (tr.take < b)).any()}}
    return out, f"合図音の予定表（config.CUE_*）を MoCap の動きに合わせて配置（ずれ {o:+.2f} s，反応の遅れを含む）", info
