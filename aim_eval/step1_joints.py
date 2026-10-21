"""step1 関節位置の推定（.db3 → rs_joints.npz）と目視確認用の動画

何をするか
  設定に並べたセッションの試行を順に見て、**joints/rs_joints.npz が無いものだけ**
  experiment/rs_joints.py を回す（時間がかかるのはここだけ）。
  そのあと確認動画（既定 20 fps）を作り、欠落・誤認識を目視で確かめられるようにする。
  最後に試行ごとの一覧（有効率・最長欠損・左右の取り違え・status の内訳）を表にする。

  同じコマンドを何度実行しても、終わっている試行は飛ばして続きから進む。
"""
import json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import common as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def estimate(trial, python=None, env=None, stride=None):
    """experiment/rs_joints.py を 1 試行に回す。(状態, 秒, 出したい行) を返す。

    成否は **.npz が新しくできたか** で判断する（終了コードだけでは見ない）。
    rs_joints.py は .npz を保存したあとに要約を印字するので、印字で落ちても結果は残る。
    とくに Windows の既定（cp932）では、要約の en ダッシュ「–」を印字できずに
    UnicodeEncodeError で落ちる。PYTHONIOENCODING=utf-8 を渡してこれを防ぐ。
    """
    script = os.path.join(ROOT, "experiment", "rs_joints.py")
    npz = os.path.join(trial, "joints", "rs_joints.npz")
    if not os.path.exists(script):
        return "失敗", 0.0, [f"experiment/rs_joints.py がありません（{ROOT}）"]
    before = os.path.getmtime(npz) if os.path.exists(npz) else -1
    t0 = time.time()
    cmd = [python or sys.executable, script, trial]
    if stride and int(stride) > 1:
        cmd += ["--stride", str(int(stride))]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                            **(env or {})})
    dt = time.time() - t0
    made = os.path.exists(npz) and os.path.getmtime(npz) > before
    lines = [l for l in (p.stdout or "").strip().splitlines()[-4:]]
    if made and p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()
        lines.append("※ 結果は保存されています（印字だけ失敗）: " + (tail[-1] if tail else ""))
        return "OK（印字で警告）", dt, lines
    if made:
        return "OK", dt, lines
    lines += [l for l in (p.stderr or "").strip().splitlines()[-3:]]
    return "失敗", dt, lines


def estimate_all(todo, lg, python=None, workers=1, threads_per_worker=None, stride=None):
    """複数の試行を並列に推定する。試行どうしは独立なので結果は変わらない。

    MediaPipe の CPU 推論は内部でも複数スレッドを使うので、worker を増やしすぎると
    かえって遅くなる。threads_per_worker を指定すると 1 本あたりのスレッド数を抑える。
    """
    env = {}
    if threads_per_worker:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "XNNPACK_NUM_THREADS"):
            env[k] = str(int(threads_per_worker))
    n, done, t0 = len(todo), [0], time.time()

    def one(item):
        k, trial = item
        state, dt, lines = estimate(trial, python, env, stride)
        done[0] += 1
        el = time.time() - t0
        rest = (el / done[0]) * (n - done[0])
        lg(f"- [{done[0]}/{n}] {os.path.basename(trial)} … "
           f"{'**失敗**' if state == '失敗' else state}"
           f"（{dt / 60:.1f} 分，経過 {el / 60:.0f} 分，残り目安 {rest / 60:.0f} 分）")
        for line in lines:
            lg(f"      {line}")
        return state != "失敗"

    if workers <= 1:
        return [one((k, t)) for k, t in enumerate(todo, 1)]
    lg(f"（{workers} 本を並列に処理します）\n")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, list(enumerate(todo, 1))))


def video(trial, fps, lg, python=None, force=False):
    """確認動画を作る（既定 20 fps）。

    experiment/lib を直接使って **はじめから fps を落として** 書き出すので ffmpeg は要らない
    （joints_video.py は config.FPS = 60 で書き出すため、そのままでは 60 fps になる）。
    60 fps の記録から等間隔にフレームを間引くので、動画は実時間で進む。
    細かい一瞬の誤検出は下の一覧（status の内訳・最長欠損）で確認すること。
    import に失敗したときは joints_video.py を呼ぶ（その場合 fps は 60 のまま）。
    """
    fps = fps or 20
    out = os.path.join(trial, "joints", f"joints_check_{int(fps)}fps.mp4")
    if os.path.exists(out) and not force:
        lg(f"      確認動画: {os.path.relpath(out, ROOT)}（すでにあります）"); return out
    npz = os.path.join(trial, "joints", "rs_joints.npz")
    if not os.path.exists(npz):
        lg("      確認動画: rs_joints.npz がありません（先に --steps joints）"); return None
    if not os.path.exists(os.path.join(ROOT, "experiment", "lib", "viz.py")):
        lg(f"      確認動画: experiment/lib/viz.py が見つかりません（ROOT = {ROOT}）。"
           "aim_eval はリポジトリ直下に置いてください"); return None
    if not (os.path.exists(os.path.join(trial, "session.db3"))
            or os.path.isdir(os.path.join(trial, "rgb"))):
        lg("      確認動画: session.db3 も rgb/ もないので作れません"); return None
    try:
        return _write_video(trial, out, fps, lg)
    except Exception as e:
        lg(f"      確認動画: 直接の書き出しに失敗（{type(e).__name__}: {e}）。joints_video.py を試します")
        return _video_fallback(trial, fps, lg, python)


def _write_video(trial, out, fps, lg):
    """experiment/lib を使って fps を落とした確認動画を書き出す。"""
    import importlib
    exp = os.path.join(ROOT, "experiment")
    if exp not in sys.path:
        sys.path.insert(0, exp)
    import cv2
    viz = importlib.import_module("lib.viz")          # 描画だけ（cv2 と numpy のみ）
    try:
        ecfg = importlib.import_module("config"); src_fps = float(getattr(ecfg, "FPS", 60) or 60)
    except Exception:
        src_fps = 60.0
    frames = _color_frames(trial)

    r = np.load(os.path.join(trial, "joints", "rs_joints.npz"), allow_pickle=False)
    names = [str(n) for n in r["names"]]
    t = r["t"].astype(float)
    uv, P = r["uv"].astype(float), r["P_C"].astype(float)
    st = r["status"] if "status" in r.files else None
    sn = [str(x) for x in r["status_names"]] if "status_names" in r.files else None
    swapped = r["lr_swapped"].astype(bool) if "lr_swapped" in r.files else None
    step = max(1, int(round(src_fps / fps)))          # 何フレームに 1 枚使うか
    writer, n_in, n_out = None, 0, 0
    for stamp, img in frames:
        n_in += 1
        if (n_in - 1) % step:
            continue
        k = int(np.clip(np.searchsorted(t, stamp), 1, len(t) - 1))
        k = k if abs(t[k] - stamp) < abs(t[k - 1] - stamp) else k - 1
        if abs(t[k] - stamp) > 0.002:                 # rs_joints が飛ばしたフレーム
            u, has3d, info = np.full((len(names), 2), np.nan), np.zeros(len(names), bool), "no estimate"
        else:
            u, has3d = uv[k], np.isfinite(P[k][:, 0])
            miss = [names[i] for i in range(len(names)) if not has3d[i]]
            info = ("3D: " + ("all OK" if not miss else "NG " + ",".join(miss)))
            if st is not None and sn is not None:
                bad = sorted({sn[st[k, i]] for i in range(len(names)) if sn[st[k, i]] != "OK"})
                if bad:
                    info += "  " + ",".join(bad)
            if swapped is not None and swapped[k]:
                info += "  L/R SWAPPED"
        lines = [f"{stamp - t[0]:6.2f} s  frame {k}", info]
        vis = viz.draw_joints(img, names, u, has3d, None, lines)
        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), float(fps),
                                     (vis.shape[1], vis.shape[0]))
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter を開けません（コーデック mp4v）")
        writer.write(vis); n_out += 1
    if writer is None:
        raise RuntimeError("画像を 1 枚も読めませんでした")
    writer.release()
    lg(f"      確認動画: {os.path.relpath(out, ROOT)}（{int(fps)} fps，{n_out} 枚 / {n_in} フレーム）")
    return out


def _color_frames(trial):
    """(時刻 [s], BGR 画像) を順に返す。session.db3 は pyrealsense2 経由、無ければ rgb/ を読む。"""
    import importlib
    db3 = os.path.join(trial, "session.db3")
    if os.path.exists(db3):
        rs_device = importlib.import_module("lib.rs_device")   # pyrealsense2 が必要
        from pathlib import Path
        for stamp, img, _K, _D in rs_device.open_color(Path(trial))():
            yield stamp, img
        return
    rgb = os.path.join(trial, "rgb")
    ts = os.path.join(trial, "frame_timestamps.csv")
    if not (os.path.isdir(rgb) and os.path.exists(ts)):
        raise RuntimeError("session.db3 も rgb/ もありません")
    import csv as _csv
    import cv2
    imgs = sorted(f for f in os.listdir(rgb) if f.lower().endswith((".jpg", ".png")))
    seen = set()
    with open(ts, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            key = row.get("color_frame_number")
            if key in seen:
                continue
            seen.add(key)
            i = int(float(row["frame_index"]))
            if 0 <= i < len(imgs):
                yield float(row["timestamp_ms"]) / 1000.0, cv2.imread(os.path.join(rgb, imgs[i]))


def _video_fallback(trial, fps, lg, python=None):
    script = os.path.join(ROOT, "experiment", "joints_video.py")
    src = os.path.join(trial, "joints", "joints_check.mp4")
    if not os.path.exists(script):
        lg("      joints_video.py もありません"); return None
    if not os.path.exists(src):
        p = subprocess.run([python or sys.executable, script, trial], cwd=ROOT,
                           capture_output=True, text=True, errors="replace")
        if p.returncode != 0 or not os.path.exists(src):
            for line in (p.stderr or "").strip().splitlines()[-2:]:
                lg(f"        {line}")
            return None
    dst = os.path.join(trial, "joints", f"joints_check_{int(fps)}fps.mp4")
    if shutil.which("ffmpeg"):
        q = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-r", str(int(fps)),
                            "-vcodec", "libx264", "-crf", "28", dst], capture_output=True, text=True)
        if q.returncode == 0:
            lg(f"      確認動画: {os.path.relpath(dst, ROOT)}（{int(fps)} fps）"); return dst
    lg(f"      確認動画: {os.path.relpath(src, ROOT)}（fps は 60 のまま）")
    return src


def qc_row(trial, meta):
    """rs_joints.npz から目視確認用の数値を拾う（解析は a0 で行う）。"""
    f = os.path.join(trial, "joints", "rs_joints.npz")
    if not os.path.exists(f):
        return None
    r = np.load(f, allow_pickle=False)
    names = [str(n) for n in r["names"]]
    P = r["P_C"].astype(float)
    row = {"trial": os.path.basename(trial), "frames": len(P),
           "band": meta.get("task", {}).get("band") or meta.get("task", {}).get("kind", ""),
           "seed": meta.get("task", {}).get("seed", "")}
    fs = 1.0 / np.median(np.diff(r["t"].astype(float)))
    for n in ("shoulder", "elbow", "wrist", "mcp3"):
        if n not in names:
            continue
        X = P[:, names.index(n)]
        ok = np.isfinite(X).all(1)
        row[f"v_{n}"] = 100 * float(ok.mean())
        gap, run = 0, 0
        for b in ~ok:
            run = run + 1 if b else 0
            gap = max(gap, run)
        row[f"g_{n}"] = gap / fs
    if "lr_swapped" in r.files:
        row["lr"] = 100 * float(r["lr_swapped"].astype(bool).mean())
    if "status" in r.files and "status_names" in r.files:
        st = r["status"]; sn = [str(x) for x in r["status_names"]]
        j = names.index("wrist") if "wrist" in names else 0
        for k, s in enumerate(sn):
            if s != "OK":
                row[f"st_{s}"] = 100 * float(np.mean(st[:, j] == k))
    return row


def _targets(sessions, include_static):
    for s in sessions:
        for trial, meta in s["trials"]:
            if not include_static and meta.get("task", {}).get("kind") != "track":
                continue
            yield s, trial, meta


def run(sessions, lg, fps=20, skip_done=True, include_static=True, video_mode="all",
        python=None, steps=("estimate", "video", "list"), force_video=False,
        workers=0, threads_per_worker=None, video_workers=0, stride=None):
    """sessions: step1 用の情報（run.py の plan が作る）。

    steps で「推定」「動画」「一覧」を別々に実行できる（run.py の --steps joints / video）。
    動画は **推定を今回したかどうかに関係なく**、rs_joints.npz があって動画が無い試行に作る。
    """
    lg("\n## step1 関節位置の推定と確認動画\n")
    if "estimate" in steps:
        todo = []
        for s, trial, meta in _targets(sessions, include_static):
            if os.path.exists(os.path.join(trial, "joints", "rs_joints.npz")) and skip_done:
                continue
            if not os.path.exists(os.path.join(trial, "session.db3")):
                lg(f"- [skip] {os.path.basename(trial)}: session.db3 がありません"); continue
            todo.append((s, trial, meta))
        import multiprocessing
        cpu = multiprocessing.cpu_count()
        w = workers if workers and workers > 0 else max(1, min(4, cpu - 1))
        est = len(todo) * (9.3 / max(1, int(stride or 1))) / max(1, w)
        per = 9.3 / max(1, int(stride or 1))
        lg(f"### 推定する試行: **{len(todo)} 本**"
           + (f"（CPU {cpu} コア，並列 {w}"
              + (f"，stride {int(stride)}" if stride and int(stride) > 1 else "")
              + f"。1 本 {per:.1f} 分なので目安 "
              f"{int(est) // 60} 時間 {int(est) % 60} 分）\n" if todo else "（すべて済み）\n"))
        if todo:
            estimate_all([t for _, t, _ in todo], lg, python, workers=w,
                         threads_per_worker=threads_per_worker, stride=stride)

    if "video" in steps and video_mode != "none":
        vids = []
        made = set()
        for s, trial, meta in _targets(sessions, include_static):
            if not os.path.exists(os.path.join(trial, "joints", "rs_joints.npz")):
                continue
            if video_mode == "first":
                if s["dir"] in made or meta.get("task", {}).get("kind") != "track":
                    continue
                made.add(s["dir"])
            vids.append(trial)
        vw = video_workers if video_workers and video_workers > 0 else 1
        lg(f"\n### 確認動画（{int(fps or 20)} fps）: {len(vids)} 本"
           + (f"（並列 {vw}）" if vw > 1 else "") + "\n")
        if vw > 1:
            with ThreadPoolExecutor(max_workers=vw) as ex:
                list(ex.map(lambda tr: video(tr, fps, lg, python, force=force_video), vids))
        else:
            for k, trial in enumerate(vids, 1):
                lg(f"- [{k}/{len(vids)}] {os.path.basename(trial)}")
                video(trial, fps, lg, python, force=force_video)

    if "list" not in steps:
        return []
    # --- 一覧
    lg("\n### 目視確認用の一覧\n")
    rows = []
    for s in sessions:
        for trial, meta in s["trials"]:
            r = qc_row(trial, meta)
            if r:
                r["session"] = os.path.basename(s["dir"]); rows.append(r)
    if not rows:
        lg("（rs_joints.npz がまだありません）"); return rows
    cols = [("セッション", "session"), ("試行", "trial"), ("フレーム", "frames", "{:.0f}")]
    for n, lab in (("shoulder", "肩"), ("elbow", "肘"), ("wrist", "手首"), ("mcp3", "MCP3")):
        if any(f"v_{n}" in r for r in rows):
            cols.append((f"{lab} 有効率 [%]", f"v_{n}", "{:.0f}"))
    for n, lab in (("shoulder", "肩"), ("elbow", "肘"), ("wrist", "手首"), ("mcp3", "MCP3")):
        if any(f"g_{n}" in r for r in rows):
            cols.append((f"{lab} 最長欠損 [s]", f"g_{n}", "{:.2f}"))
    cols.append(("左右の取り違え [%]", "lr", "{:.1f}"))
    for s in ("NO_POSE", "NO_HAND", "REJECT", "OUT", "NO_DEPTH"):
        if any(f"st_{s}" in r for r in rows):
            cols.append((s, f"st_{s}", "{:.1f}"))
    lg(C.md_table(rows, cols))
    bad = [r for r in rows if min([r.get(f"v_{n}", 100) for n in
                                   ("shoulder", "elbow", "wrist", "mcp3")]) < 70 or r.get("lr", 0) > 10]
    if bad:
        lg(f"\n> 有効率が 70 % 未満、または左右の取り違えが 10 % を超えた試行が {len(bad)} 本あります。"
           "**確認動画を目視して**、座る向き（config.RIGHT_SHOULDER_IMAGE_SIDE）や画角を見直してください。")
    return rows
