"""60 fps で関節推定をやり直す単独スクリプト（既存のコードは変更しない・コミット不要）。

  python aim_eval/fps60_joints.py --plan    # 対象の試行と、今ある結果の状態だけ表示
  python aim_eval/fps60_joints.py           # 推定を回す（60 fps の結果があるものは飛ばす）

各試行で行うこと
  1. joints/rs_joints.npz（既存の 30 fps の結果）を一時的に退避する
  2. experiment/rs_joints.py を --stride 1（全フレーム）で実行 → joints/rs_joints.npz ができる
  3. それを joints/rs_joints_fps60.npz に改名する
  4. 退避しておいた 30 fps の結果を joints/rs_joints.npz に戻す
よって、既存の解析が読む rs_joints.npz は変わらず、60 fps は rs_joints_fps60.npz として残る。
"""
import argparse
import glob
import importlib.util
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OLD = "rs_joints.npz"             # 既存（30 fps）
NEW = "rs_joints_fps60.npz"       # 今回作る（60 fps）
TMP = "rs_joints_fps30_tmp.npz"   # 退避用


def resolve(d):
    return os.path.normpath(d if os.path.isabs(d) else os.path.join(ROOT, d))


def load_sessions(path):
    spec = importlib.util.spec_from_file_location("aim_config", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.SESSIONS


def find_trials(sessions):
    trials = []
    for e in sessions:
        d = resolve(e["dir"])
        for db3 in sorted(glob.glob(os.path.join(d, "*", "session.db3"))):
            trials.append(os.path.dirname(db3))
    return trials


def process(trial, python):
    joints = os.path.join(trial, "joints")
    old = os.path.join(joints, OLD)
    new = os.path.join(joints, NEW)
    tmp = os.path.join(joints, TMP)
    if os.path.exists(new):
        return "スキップ（60 fps の結果あり）"
    if os.path.exists(tmp):
        return f"中止（一時ファイル {TMP} が残っています。確認してください）"
    os.makedirs(joints, exist_ok=True)
    had_old = os.path.exists(old)
    if had_old:
        os.rename(old, tmp)
    try:
        p = subprocess.run(
            [python, os.path.join(ROOT, "experiment", "rs_joints.py"), trial, "--stride", "1"],
            cwd=ROOT, capture_output=True, text=True, errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        made = os.path.exists(old)          # rs_joints.py が新しく作った 60 fps の結果
        if made:
            os.rename(old, new)
        if made:
            status = "OK"
        else:
            lines = (p.stderr or "").strip().splitlines()
            status = "失敗: " + (lines[-1] if lines else "原因不明")
    finally:
        # 30 fps の結果を必ず元の名前に戻す（失敗したときも）
        if had_old and os.path.exists(tmp):
            os.rename(tmp, old)
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.py"))
    ap.add_argument("--plan", action="store_true", help="対象と状態だけ表示して終わる")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    trials = find_trials(load_sessions(a.config))
    print(f"対象の試行: {len(trials)} 本")
    if a.plan:
        for t in trials:
            j = os.path.join(t, "joints")
            s60 = "60fps あり" if os.path.exists(os.path.join(j, NEW)) else "60fps なし"
            s30 = "30fps あり" if os.path.exists(os.path.join(j, OLD)) else "30fps なし"
            print(f"  {os.path.relpath(t, ROOT)}  [{s60} / {s30}]")
        return

    for i, t in enumerate(trials, 1):
        t0 = time.time()
        r = process(t, a.python)
        print(f"[{i}/{len(trials)}] {os.path.relpath(t, ROOT)}: {r}（{time.time() - t0:.0f} s）", flush=True)


if __name__ == "__main__":
    main()