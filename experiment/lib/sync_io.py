"""
sync_io.py
────────────────────────────────────────────────────────────────────
sync_check.py の結果（sync_check/sync_result.json）を使う側の共通処理．

  load_sync(trial_dir)       … sync_result.json と，そこに記録された較正の T_MC（なければ None）
  take_offset(sync, t)       … RealSense 時刻 t → c（Take 時刻 = t + c）
  operation_mask(sync, t)    … 動作区間のフレームなら True
                                （最初の同期区間より後で，どの同期区間 ± CMP_SYNC_MARGIN にも入らない）
────────────────────────────────────────────────────────────────────
"""

import numpy as np

import config
from lib import io_utils


def load_sync(trial_dir):
    """(sync, calib フォルダ, T_MC)．sync_check.py をまだ実行していなければ (None, None, None)．"""
    path = trial_dir / "sync_check" / "sync_result.json"
    if not path.is_file():
        return None, None, None
    sync = io_utils.load_json(path)
    calib = config.RECORDINGS_DIR / sync["calib"] if sync.get("calib") else io_utils.calib_for_trial(trial_dir)
    T_MC = np.array(io_utils.load_json(calib / "T_MC.json")["T_MC"])
    return sync, calib, T_MC


def take_offset(sync, t):
    """区間が 2 つ以上なら，区間の中央の時刻で c を直線補間する（記録中の時計のずれも吸収）．"""
    segs = sync["segments"]
    mids = np.array([(s["rs_t0"] + s["rs_t1"]) / 2 for s in segs])
    cs = np.array([s["c"] for s in segs])
    return np.interp(t, mids, cs) if len(segs) >= 2 else np.full(len(t), cs[0])


def operation_mask(sync, t):
    """録画開始〜最初の同期の前（まだ体が写っていない）と，ボードを振った区間の前後を除く．"""
    segs = sync["segments"]
    ok = t > min(s["rs_t1"] for s in segs) + config.CMP_SYNC_MARGIN
    for s in segs:
        ok &= (t < s["rs_t0"] - config.CMP_SYNC_MARGIN) | (t > s["rs_t1"] + config.CMP_SYNC_MARGIN)
    return ok
