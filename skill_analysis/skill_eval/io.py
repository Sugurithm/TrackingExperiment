"""セッションの探索と試行の読み込み（aim_eval/common.py を読むだけで使う）

記録フォルダには何も書かない。Trial の出力先は results/<条件>/trials/ に向ける。
静止試行のノイズ床（static_floor.json）は既存のキャッシュを読むだけにする（無ければ作らずに NaN）。
"""
import glob, json, os, sys
import argparse

from . import config as K

if K.AIM_EVAL not in sys.path:
    sys.path.insert(0, K.AIM_EVAL)
import common as C  # noqa: E402  aim_eval/common.py


def display_name(raw):
    return K.ALIASES.get(str(raw).lower(), raw)


def group_of(name):
    for g, members in K.GROUPS.items():
        if name in members:
            return g
    return None


def find_sessions(dpi):
    """recordings/aim_* から、指定 DPI のセッションを参加者ごとに 1 つずつ返す。{表示名: フォルダ}"""
    out = {}
    for d in sorted(glob.glob(os.path.join(K.RECORDINGS, "aim_*"))):
        ov = os.path.join(d, C.OVERRIDE_NAME)
        if not os.path.exists(os.path.join(d, "session.json")) or not os.path.exists(ov):
            continue
        o = json.load(open(ov, encoding="utf-8"))
        if float(o.get("dpi", -1)) != float(dpi):
            continue
        name = display_name(o.get("participant", ""))
        if group_of(name) is None:
            continue
        if name in out:      # 同じ人・同じ DPI が 2 つあれば新しい方（フォルダ名の時刻が後）
            print(f"  ! {name} DPI {dpi} のセッションが複数: {os.path.basename(out[name])} → {os.path.basename(d)}")
        out[name] = d
    return out


def trial_dirs(session, band):
    """セッション内の本番の追従試行（指定帯域）を seed 順に返す。[(dir, meta)]"""
    rows = [(d, m) for d, m in C.list_trials(session, kinds=("track",))
            if m.get("task", {}).get("band") == band]
    return sorted(rows, key=lambda r: r[1]["task"].get("seed", 0))


def load_trial(trial_dir, out_dir):
    ns = argparse.Namespace(trial=trial_dir, rs=None, tmc=None, axes=None, out=out_dir)
    return C.Trial(ns)


def floor_cached(tr):
    """既存の static_floor.json を読むだけ（無ければ空）。"""
    p = os.path.join(tr.sess_dir or "", "static_floor.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}
