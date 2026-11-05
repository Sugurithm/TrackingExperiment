"""選んだ試行すべてを 60 fps（全フレーム）で関節推定し直す．確認動画（20 fps）と進捗の記録つき．

  python experiment/60fps_joints.py --plan              # 対象と状態だけ表示（時間の目安つき）
  python experiment/60fps_joints.py                     # 対象すべてを実行（60 fps 済みは飛ばす）
  python experiment/60fps_joints.py --config-only       # recordings/ の自動追加をやめ，設定ファイルの SESSIONS だけ
  python experiment/60fps_joints.py --compare slow_s1_all   # その比較で使う試行だけ（自動追加なし）
  python experiment/60fps_joints.py --jobs 2            # 2 本ずつ並列（PC による．先に --plan で本数を確認）
  python experiment/60fps_joints.py --video-fps 0       # 確認動画を作らない

対象 = temp/compare_config.py の SESSIONS に並べたセッション
      + recordings/aim_* にあって SESSIONS に無く，フォルダ名の dpi が 400 / 800 のセッション（既定で自動追加．1600 などは対象外）
の試行すべて（trial_00_static_pre ややり直し _2, _3 も含む）．
--config-only で自動追加をやめる．--compare NAME を付けると，COMPARES のその比較が使う試行
（sessions と trials の条件に合うもの）だけに絞る（自動追加はしない）．別の設定ファイルは --config．

各試行で行うこと
  1. joints/rs_joints.npz が 30 fps（stride > 1）なら joints/rs_joints_fps30.npz に改名して残す
  2. experiment/rs_joints.py を --stride 1（全フレーム）で実行 → joints/rs_joints.npz が 60 fps になる
  3. 失敗したら 30 fps の結果を rs_joints.npz に戻す
  4. 成功したら関節を重ねた確認動画 joints/rs_joints_20fps.mp4 を作る（--video-fps で変更，0 で無し）
     （動画の失敗は推定の成否に影響しない．60 fps 済みの試行も動画が無ければ動画だけ作る）
60 fps かどうかは npz に保存された stride（1 = 60 fps）で判断するので，
途中で止めても同じコマンドで続きから進む（終わった試行は飛ばす）．

進捗（実行中に別の窓で開いておける）
  recordings/60fps_progress.txt … 数十秒おきに上書き．全体の本数・経過・残り時間の目安，
                                  実行中の試行のフレーム数，終わった試行の結果の一覧
  recordings/60fps_log.txt      … 試行が終わるたびに 1 行追記（時刻・結果・所要時間）
  端末には試行が終わるたびに 1 行，5 分おきに実行中の試行のフレーム数を表示する．
"""
import argparse
import datetime
import glob
import importlib.util
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_CONFIG = os.path.join(ROOT, "temp", "compare_config.py")
CUR = "rs_joints.npz"             # 解析が読む名前（60 fps にする）
OLD30 = "rs_joints_fps30.npz"     # 30 fps の結果の退避先（残す）
VIDEO_NAME = "rs_joints_{fps}fps.mp4"
MIN_PER_TRIAL = 9                 # 1 本あたりの目安 [分]（stride 1，1 本 約 2800 フレーム）
TRIAL_RE = re.compile(r"^trial_(\d+)_(slow|fast)_s(\d+)(?:_(\d+))?$")
LABEL_RE = re.compile(r"^aim_(.+?)_\d{8}_\d{6}$")
DPI_RE = re.compile(r"^aim_(.+?)(\d{3,4})?_\d{8}_\d{6}$")
AUTO_DPI = (400, 800)             # 自動追加する dpi（フォルダ名の数字で判断．1600 などは対象外）
MATCH_TOL = 0.002                 # 動画のフレームと rs_joints.npz の時刻をこの差以内で対応づける [s]
PROGRESS_EVERY = 20               # 進捗ファイルを更新する間隔 [s]
HEARTBEAT_EVERY = 300             # 端末に実行中の様子を出す間隔 [s]
LOCK = threading.Lock()

# 進捗の共有状態
LIVE = {}                         # 実行中の試行 → 最後の進捗行（rs_joints.py の表示）
LIVE_SINCE = {}                   # 実行中の試行 → 開始時刻
RESULTS = []                      # (試行, 結果, 所要秒, 実際に推定したか)
RUN = dict(total=0, todo=0, jobs=1, t0=0.0, done=0, worked=[], paths=None)


# ---------------------------------------------------------------- 設定の読み込み
def load_config(path):
    spec = importlib.util.spec_from_file_location("compare_config", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return list(getattr(m, "SESSIONS", [])), list(getattr(m, "COMPARES", []))


def resolve_dir(p):
    if os.path.isabs(p) and os.path.isdir(p):
        return os.path.normpath(p)
    for base in (os.getcwd(), ROOT):
        q = os.path.join(base, p)
        if os.path.isdir(q):
            return os.path.normpath(os.path.abspath(q))
    return None


def default_label(sdir):
    name = os.path.basename(os.path.normpath(sdir))
    m = LABEL_RE.match(name)
    return m.group(1) if m else name


def session_match(s, spec):
    if spec in ("all", None):
        return True
    spec = [spec] if isinstance(spec, str) else list(spec)
    return any(x in (s["label"], s["person"], s["group"]) for x in spec)


def trial_match(tdir, spec):
    """compare の trials 条件（"all" か，slow_s1 / trial_01_slow_s1 のリスト）に合うか"""
    m = TRIAL_RE.match(os.path.basename(tdir))
    if not m:
        return False                      # 静止試行など（比較の対象外）
    if spec in ("all", None):
        return True
    spec = [spec] if isinstance(spec, str) else list(spec)
    return f"{m.group(2)}_s{int(m.group(3))}" in spec or os.path.basename(tdir) in spec


def discover_aim(sessions):
    """recordings/aim_* のうち，sessions に無く，フォルダ名の dpi が AUTO_DPI のものを返す．
    返り値は (追加するもの, 対象外にしたフォルダ名)"""
    known = {resolve_dir(e["dir"]) for e in sessions}
    extra, skipped = [], []
    for d in sorted(glob.glob(os.path.join(ROOT, "recordings", "aim_*"))):
        d = os.path.normpath(os.path.abspath(d))
        if not os.path.isdir(d) or d in known:
            continue
        m = DPI_RE.match(os.path.basename(d))
        dpi = int(m.group(2)) if m and m.group(2) else None
        if dpi in AUTO_DPI:
            extra.append(dict(dir=d))
        else:
            skipped.append(f"{os.path.basename(d)}（dpi={dpi}）")
    return extra, skipped


# ---------------------------------------------------------------- 対象の試行
def find_trials(sessions, compare=None):
    trials, seen = [], set()
    for e in sessions:
        d = resolve_dir(e["dir"])
        if d is None:
            print(f"[WARN] フォルダがありません: {e['dir']}")
            continue
        s = dict(person=e.get("person", ""), group=e.get("group", ""),
                 label=e.get("label") or default_label(d))
        if compare is not None and not session_match(s, compare.get("sessions", "all")):
            continue
        for db3 in sorted(glob.glob(os.path.join(d, "*", "session.db3"))):
            t = os.path.dirname(db3)
            if compare is not None and not trial_match(t, compare.get("trials", "all")):
                continue
            if t not in seen:
                seen.add(t)
                trials.append(t)
    return trials


# ---------------------------------------------------------------- 状態
def stride_of(path):
    """npz に保存された stride を返す（読めない・無ければ None）"""
    try:
        with np.load(path, allow_pickle=True) as z:
            return int(z["stride"]) if "stride" in z.files else None
    except Exception:
        return None


def video_path(trial, fps):
    return os.path.join(trial, "joints", VIDEO_NAME.format(fps=int(fps)))


def state(trial):
    j = os.path.join(trial, "joints")
    cur, o30 = os.path.join(j, CUR), os.path.join(j, OLD30)
    if os.path.exists(cur):
        s = stride_of(cur)
        if s == 1:
            return "done60", "60 fps 済み" + ("（30 fps も保存済み）" if os.path.exists(o30) else "")
        return "have30", f"rs_joints.npz は 30 fps（stride={s}）" + ("、fps30 ファイルも既にある" if os.path.exists(o30) else "")
    return "none", "rs_joints.npz なし"


# ---------------------------------------------------------------- rs_joints.py の実行（進捗を拾う）
def run_rs_joints(trial, python):
    """rs_joints.py --stride 1 を走らせ，進捗の行（\\r 区切り）を LIVE に入れる．(終了コード, 末尾の行) を返す"""
    p = subprocess.Popen(
        [python, os.path.join(ROOT, "experiment", "rs_joints.py"), trial, "--stride", "1"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    tail, buf = [], b""
    while True:
        chunk = os.read(p.stdout.fileno(), 4096)      # 来ている分だけ読む（溜まるまで待たない）
        if not chunk:
            break
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()                       # 区切りの前の途中の行は次に持ち越す
        for raw in parts:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            tail.append(line)
            tail = tail[-8:]
            with LOCK:
                LIVE[trial] = line
    p.wait()
    return p.returncode, tail


# ---------------------------------------------------------------- 確認動画（20 fps）
def make_video(trial, fps):
    """rs_joints.npz の関節を RGB に重ねた動画を fps で書く（時間で間引く．MediaPipe はやり直さない）"""
    sys.path.insert(0, HERE)                     # experiment/ の config と lib を使う
    import cv2
    from lib.rs_device import open_color
    from lib.viz import draw_joints
    from pathlib import Path

    z = np.load(os.path.join(trial, "joints", CUR), allow_pickle=True)
    names, t, UV, P = list(z["names"]), z["t"], z["uv"], z["P_C"]
    out = video_path(trial, fps)
    tmp = out + ".part.mp4"
    writer, n, next_t = None, 0, 0.0
    try:
        for stamp, img, K, D in open_color(Path(trial))():
            rel = stamp - t[0]
            if rel < next_t:
                continue
            next_t += 1.0 / fps
            k = int(np.clip(np.searchsorted(t, stamp), 1, len(t) - 1))
            k = k if abs(t[k] - stamp) < abs(t[k - 1] - stamp) else k - 1
            if abs(t[k] - stamp) > MATCH_TOL:    # 深度が無く rs_joints で飛ばしたフレーム
                uv, has3d = np.full((len(names), 2), np.nan), np.zeros(len(names), bool)
            else:
                uv, has3d = UV[k], np.isfinite(P[k][:, 0])
            vis = draw_joints(img, names, uv, has3d, {}, [f"{rel:6.2f} s  {fps:g} fps  stride 1", ""])
            if writer is None:
                writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), float(fps),
                                         (vis.shape[1], vis.shape[0]))
            writer.write(vis)
            n += 1
    finally:
        if writer is not None:
            writer.release()
    if n == 0:
        raise RuntimeError("動画に書けるフレームがありません")
    os.replace(tmp, out)
    return n


def try_video(trial, fps):
    if not fps:
        return ""
    try:
        n = make_video(trial, fps)
        return f"、動画 {fps:g} fps（{n} 枚）"
    except Exception as e:                        # 動画の失敗で推定の結果は消さない
        return f"、動画は失敗: {type(e).__name__}: {e}"


# ---------------------------------------------------------------- 1 試行
def process(trial, python, video_fps):
    joints = os.path.join(trial, "joints")
    cur = os.path.join(joints, CUR)
    o30 = os.path.join(joints, OLD30)
    st, _ = state(trial)
    if st == "done60":
        if video_fps and not os.path.exists(video_path(trial, video_fps)):
            with LOCK:
                LIVE[trial] = "動画だけ作成中"
            return "OK（60 fps 済み）" + try_video(trial, video_fps), False
        return "スキップ（60 fps の結果あり）", False
    os.makedirs(joints, exist_ok=True)
    moved = False
    if st == "have30":
        if os.path.exists(o30):
            return f"中止（{OLD30} が既にあります．上書きを避けるため確認してください）", False
        os.rename(cur, o30)
        moved = True
    ok = False
    try:
        code, tail = run_rs_joints(trial, python)
        ok = os.path.exists(cur) and stride_of(cur) == 1
        if ok:
            with LOCK:
                LIVE[trial] = "動画を作成中"
            return "OK" + try_video(trial, video_fps), True
        return "失敗: " + (tail[-1] if tail else f"終了コード {code}"), True
    finally:
        # 失敗したときは 30 fps の結果を必ず元の名前に戻す
        if moved and not ok and os.path.exists(o30):
            if os.path.exists(cur):
                os.remove(cur)          # 中途半端な出力があれば消す（30 fps 版は o30 にある）
            os.rename(o30, cur)


# ---------------------------------------------------------------- 進捗の記録
def fmt_dur(sec):
    sec = max(0, int(sec))
    return f"{sec // 3600}時間{sec % 3600 // 60:02d}分" if sec >= 3600 else f"{sec // 60}分{sec % 60:02d}秒"


def eta_text():
    """実際に推定した試行の平均から，残り時間を見積もる（LOCK を持って呼ぶ）"""
    w = RUN["worked"]
    left = RUN["todo"] - len([r for r in RESULTS if r[3]])
    if left <= 0:
        return "0分00秒"
    if not w:
        return "—（1 本目が終わるまで未定）"
    return fmt_dur(sum(w) / len(w) * left / RUN["jobs"])


def write_progress(final=False):
    now = time.time()
    with LOCK:
        done_w = len([r for r in RESULTS if r[3]])
        lines = [
            f"60 fps 推定の進捗  更新 {datetime.datetime.now():%Y-%m-%d %H:%M:%S}" + ("（終了）" if final else ""),
            f"全体: {RUN['done']}/{RUN['total']} 本 処理済み（飛ばした分を含む）／要処理 {RUN['todo']} 本のうち {done_w} 本完了",
            f"経過: {fmt_dur(now - RUN['t0'])}   残りの目安: {eta_text()}   並列: {RUN['jobs']}",
            "",
            "実行中:",
        ]
        if LIVE:
            for t, s in LIVE.items():
                lines.append(f"  {os.path.relpath(t, ROOT)}  [{fmt_dur(now - LIVE_SINCE.get(t, now))}]  {s}")
        else:
            lines.append("  （なし）")
        lines += ["", "終わった試行:"]
        for t, r, sec, _ in RESULTS:
            lines.append(f"  {os.path.relpath(t, ROOT)}: {r}（{fmt_dur(sec)}）")
        text = "\n".join(lines) + "\n"
    try:
        tmp = RUN["paths"][0] + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, RUN["paths"][0])
    except OSError:
        pass                                      # 窓で開いていて置き換えられないときは次回に


def reporter(stop):
    last_beat = time.time()
    while not stop.wait(PROGRESS_EVERY):
        write_progress()
        if time.time() - last_beat >= HEARTBEAT_EVERY:
            last_beat = time.time()
            with LOCK:
                rows = [f"    {os.path.relpath(t, ROOT)}: {s}" for t, s in LIVE.items()]
                eta = eta_text()
            print(f"[進行中] 経過 {fmt_dur(time.time() - RUN['t0'])}  残りの目安 {eta}", flush=True)
            for r in rows:
                print(r, flush=True)


def log_line(trial, result, sec):
    try:
        with open(RUN["paths"][1], "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\t{os.path.relpath(trial, ROOT)}\t{result}\t{sec:.0f} s\n")
    except OSError:
        pass


# ---------------------------------------------------------------- メイン
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="SESSIONS を持つ設定ファイル（既定 temp/compare_config.py）")
    ap.add_argument("--config-only", action="store_true",
                    help="recordings/aim_* の自動追加（800 dpi など）をせず，SESSIONS だけを対象にする")
    ap.add_argument("--compare", default=None, metavar="NAME", help="COMPARES のその比較が使う試行だけに絞る")
    ap.add_argument("--plan", action="store_true", help="対象と状態だけ表示して終わる")
    ap.add_argument("--jobs", type=int, default=1, help="並列に回す試行の数（既定 1）")
    ap.add_argument("--video-fps", type=float, default=20.0, metavar="FPS",
                    help="確認動画の fps（既定 20．0 で作らない）")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    if not os.path.isfile(a.config):
        raise SystemExit(f"[ERROR] 設定ファイルがありません: {a.config}")
    sessions, compares = load_config(a.config)
    cmp_ = None
    if a.compare:
        cmp_ = next((c for c in compares if c.get("name") == a.compare), None)
        if cmp_ is None:
            names = ", ".join(c.get("name", "?") for c in compares) or "（なし）"
            raise SystemExit(f"[ERROR] 比較 '{a.compare}' が COMPARES にありません．あるもの: {names}")
    if cmp_ is None and not a.config_only:
        extra, skipped = discover_aim(sessions)
        if skipped:
            print(f"[対象外] dpi が {'/'.join(map(str, AUTO_DPI))} でない recordings/aim_* を {len(skipped)} 件除外:")
            for n in skipped:
                print(f"    {n}")
        if extra:
            print(f"[追加] SESSIONS に無い recordings/aim_* を {len(extra)} セッション追加:")
            for e in extra:
                print(f"    {os.path.basename(e['dir'])}")
            sessions += extra
    trials = find_trials(sessions, cmp_)
    todo = [t for t in trials if state(t)[0] != "done60"]
    jobs = max(1, a.jobs)
    print(f"対象の試行: {len(trials)} 本（未処理 {len(todo)} 本）  設定: {os.path.relpath(a.config, ROOT)}"
          + (f"  比較: {a.compare}" if a.compare else ""))
    if not trials:
        print("対象がありません．SESSIONS のフォルダ（recordings/…）がこの PC にあるか確認してください")
        return
    if a.plan:
        for t in trials:
            print(f"  {os.path.relpath(t, ROOT)}  [{state(t)[1]}]")
    eta_h = len(todo) * MIN_PER_TRIAL / 60 / jobs
    print(f"目安: 1 本 約 {MIN_PER_TRIAL} 分 × {len(todo)} 本 ÷ 並列 {jobs} ≒ {eta_h:.1f} 時間"
          "（並列は線形には速くならない．動画を作る分は別に少し増える）")
    if a.plan:
        return

    rec = os.path.join(ROOT, "recordings")
    os.makedirs(rec, exist_ok=True)
    RUN.update(total=len(trials), todo=len(todo), jobs=jobs, t0=time.time(),
               paths=(os.path.join(rec, "60fps_progress.txt"), os.path.join(rec, "60fps_log.txt")))
    print(f"進捗: {os.path.relpath(RUN['paths'][0], ROOT)}（{PROGRESS_EVERY} 秒おきに更新），"
          f"ログ: {os.path.relpath(RUN['paths'][1], ROOT)}", flush=True)

    def run(t):
        t0 = time.time()
        with LOCK:
            LIVE[t] = "開始"
            LIVE_SINCE[t] = t0
        try:
            r, worked = process(t, a.python, a.video_fps)
        except Exception as e:                    # 1 本の想定外の失敗で全体を止めない
            r, worked = f"失敗: {type(e).__name__}: {e}", True
        sec = time.time() - t0
        with LOCK:
            LIVE.pop(t, None)
            RESULTS.append((t, r, sec, worked))
            RUN["done"] += 1
            if worked and r.startswith("OK"):
                RUN["worked"].append(sec)
            done, eta = RUN["done"], eta_text()
        print(f"[{done}/{len(trials)}] {os.path.relpath(t, ROOT)}: {r}（{fmt_dur(sec)}）"
              f"  経過 {fmt_dur(time.time() - RUN['t0'])}  残りの目安 {eta}", flush=True)
        log_line(t, r, sec)
        write_progress()
        return r

    stop = threading.Event()
    th = threading.Thread(target=reporter, args=(stop,), daemon=True)
    th.start()
    write_progress()
    try:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(run, trials))
    finally:
        stop.set()
        th.join(timeout=2)
    write_progress(final=True)
    ng = [(t, r) for t, r in zip(trials, results) if r.startswith(("失敗", "中止"))]
    print(f"\n完了 {len(trials) - len(ng)}/{len(trials)} 本（{fmt_dur(time.time() - RUN['t0'])}）")
    for t, r in ng:
        print(f"  要確認: {os.path.relpath(t, ROOT)}: {r}")


if __name__ == "__main__":
    main()