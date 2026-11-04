"""60 fps で関節推定をやり直す単独スクリプト（既存のコードは変更しない・コミット不要）。

  python aim_eval/60fps_joints.py --plan    # 対象の試行と、今ある結果の状態だけ表示
  python aim_eval/60fps_joints.py           # 推定を回す（すでに 60 fps の試行は飛ばす）

各試行で行うこと
  1. joints/rs_joints.npz（既存の 30 fps の結果）を joints/rs_joints_fps30.npz に改名して残す
  2. experiment/rs_joints.py を --stride 1（全フレーム）で実行 → joints/rs_joints.npz（60 fps）ができる
  3. 失敗したときは 30 fps の結果を rs_joints.npz に戻す
よって、成功した試行では rs_joints.npz が 60 fps、30 fps は rs_joints_fps30.npz として残る。

60 fps か 30 fps かは npz に保存された stride（1 = 60 fps）で判断するので、
途中で止めても、同じコマンドで続きから進む（終わった試行は飛ばす）。
"""
import argparse
import glob
import importlib.util
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CUR = "rs_joints.npz"             # 解析が読む名前（今回 60 fps にする）
OLD30 = "rs_joints_fps30.npz"     # 30 fps の結果の退避先（残す）


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


def stride_of(path):
    """npz に保存された stride を返す（無ければ None）。"""
    try:
        with np.load(path, allow_pickle=True) as z:
            return int(z["stride"]) if "stride" in z.files else None
    except Exception:
        return None


def state(trial):
    j = os.path.join(trial, "joints")
    cur, o30 = os.path.join(j, CUR), os.path.join(j, OLD30)
    if os.path.exists(cur):
        s = stride_of(cur)
        if s == 1:
            return "done60", "60 fps 済み" + ("（30 fps も保存済み）" if os.path.exists(o30) else "")
        return "have30", f"rs_joints.npz は 30 fps（stride={s}）" + ("、fps30 ファイルも既にある" if os.path.exists(o30) else "")
    return "none", "rs_joints.npz なし"


def process(trial, python):
    joints = os.path.join(trial, "joints")
    cur = os.path.join(joints, CUR)
    o30 = os.path.join(joints, OLD30)
    st, _ = state(trial)
    if st == "done60":
        return "スキップ（60 fps の結果あり）"
    os.makedirs(joints, exist_ok=True)
    moved = False
    if st == "have30":
        if os.path.exists(o30):
            return f"中止（{OLD30} が既にあります。上書きを避けるため確認してください）"
        os.rename(cur, o30)
        moved = True
    ok = False
    try:
        p = subprocess.run(
            [python, os.path.join(ROOT, "experiment", "rs_joints.py"), trial, "--stride", "1"],
            cwd=ROOT, capture_output=True, text=True, errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        ok = os.path.exists(cur) and stride_of(cur) == 1
        if ok:
            return "OK"
        lines = (p.stderr or "").strip().splitlines()
        return "失敗: " + (lines[-1] if lines else "原因不明")
    finally:
        # 失敗したときは 30 fps の結果を必ず元の名前に戻す
        if moved and not ok and os.path.exists(o30):
            if os.path.exists(cur):
                os.remove(cur)          # 中途半端な出力があれば消す（30 fps 版は o30 にある）
            os.rename(o30, cur)


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
            print(f"  {os.path.relpath(t, ROOT)}  [{state(t)[1]}]")
        todo = sum(1 for t in trials if state(t)[0] != "done60")
        print(f"未処理: {todo} 本（1 本 約 9 分として 約 {todo * 9 / 60:.1f} 時間）")
        return

    for i, t in enumerate(trials, 1):
        t0 = time.time()
        r = process(t, a.python)
        print(f"[{i}/{len(trials)}] {os.path.relpath(t, ROOT)}: {r}（{time.time() - t0:.0f} s）", flush=True)


if __name__ == "__main__":
    main()