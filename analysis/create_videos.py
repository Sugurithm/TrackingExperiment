# -*- coding: utf-8 -*-
"""
trial_videos.py
────────────────────────────────────────────────────────────────────
recordings/aim_* の各セッションから，本番 6 試行（slow1〜3，fast1〜3）の
RGB 動画を書き出す（目視で動きの違いを見比べるため．MediaPipe は使わない）．

  python experiment/trial_videos.py               # SESSIONS をすべて
  python experiment/trial_videos.py --plan        # 対象と書き出し先だけ表示
  python experiment/trial_videos.py --task-only   # 課題の区間（task.csv の t = 0 〜 最後）だけ
  python experiment/trial_videos.py --overwrite   # 既にある動画も作り直す
  python experiment/trial_videos.py --scale 0.5   # 半分の大きさで書く（容量を減らす）

入力:  recordings/aim_*/trial_XX_<slow|fast>_s<1-3>[_k]/session.db3
       （あれば）同じフォルダの task.csv … 画面左上に課題の時刻 t を出す・--task-only で使う
出力:  recordings/videos/<帯域>_trial<seed>/dpi<dpi>/<参加者>.mp4
         例: videos/slow_trial1/dpi400/kakeru.mp4, videos/fast_trial3/dpi800/Yuya.mp4
       同じ参加者・dpi のセッションが複数あるときは末尾に日時を付ける（kakeru_20261022_123951.mp4）
       フォルダ名から dpi が読めないセッションは dpi_unknown/ に入れる

【やり直し試行（_2, _3 …）】
  同じ帯域・seed が複数あるときは，番号がいちばん大きいもの（最後にやり直したもの）を使う．
  使わなかった試行は --plan で表示する．

【フレームレートと落ちたフレーム】
  mp4 のフレームレートは .db3 のカラーストリームの fps をそのまま使う．
  記録中に落ちたフレームは，タイムスタンプから位置を求めて直前の画像で埋める
  （再生時間 = 実時間になる．埋めた枚数は終わりにまとめて表示する）．

【画面】
  左上: セッション名  試行名 / 課題の時刻 t [s]（t < 0 はカウントダウン）/ 記録開始からの秒数
────────────────────────────────────────────────────────────────────
"""

import argparse
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # experiment/ の config と lib を使う

import numpy as np
import pandas as pd
import cv2

import config

# ══════════════════════════════════════════════════
#  対象のセッション（ここだけを編集する）
#   リポジトリ直下からの相対パス，またはフルパス．外したいものは行頭に # を付ける．
#   空リストにすると recordings/aim_* をすべて対象にする．
# ══════════════════════════════════════════════════
SESSIONS = [
    # ---- 400 dpi ----
    "recordings/aim_kakeru400_20261022_123951",
    "recordings/aim_Keito400_20261021_170947",
    "recordings/aim_Kosuke400_20261021_172948",
    "recordings/aim_Suguru400_20261021_175327",
    "recordings/aim_Suzuka400_20261022_155937",
    "recordings/aim_Syou400_20261022_135724",
    "recordings/aim_xin400_20261022_125957",
    "recordings/aim_Yuuki400_20261022_154046",
    "recordings/aim_Yuya400_20261022_134407",
    # ---- 800 dpi ----
    "recordings/aim_kakeru800_20261022_123338",
    "recordings/aim_Keito800_20261021_171605",
    "recordings/aim_Kosuke800_20261021_173700",
    "recordings/aim_Suguru800_20261021_175931",
    "recordings/aim_Suzuka800_20261022_155314",
    "recordings/aim_Syou800_20261022_135037",
    "recordings/aim_xin800_20261022_125327",
    "recordings/aim_Yuuki800_20261022_153317",
    "recordings/aim_Yuya800_20261022_133738",
]

# BANDS = ["slow", "fast"]
BANDS = ["fast"]
# SEEDS = [1, 2, 3]
SEEDS = [1]
OUT_DIR = config.RECORDINGS_DIR / "videos"
SESSION_RE = re.compile(r"^aim_(.+?)(\d{3,4})_(\d{8}_\d{6})$")   # 参加者, dpi, 日時
MAX_FILL_S = 2.0          # これより長い時刻の空白は記録の落ちではなく時計の異常とみなし，埋めない [s]
TRIAL_RE = re.compile(r"^trial_(\d+)_(slow|fast)_s(\d+)(?:_(\d+))?$")   # 60fps_joints.py と同じ


# ══════════════════════════════════════════════════
#  対象の試行を探す
# ══════════════════════════════════════════════════
def resolve_dir(p):
    path = Path(p)
    if not path.is_absolute():
        path = config.ROOT_DIR / path
    return path if path.is_dir() else None


def session_dirs():
    if SESSIONS:
        dirs = []
        for s in SESSIONS:
            d = resolve_dir(s)
            if d is None:
                print(f"[WARN] フォルダがありません: {s}")
            else:
                dirs.append(d)
        return dirs
    return sorted(p for p in config.RECORDINGS_DIR.glob("aim_*") if p.is_dir())


def pick_trials(sdir):
    """{(band, seed): trial_dir} と，やり直しで使わなかった試行のリストを返す．"""
    cands = {}
    for t in sorted(sdir.iterdir()):
        m = TRIAL_RE.match(t.name)
        if not m or not (t / config.DB3_NAME).is_file():
            continue
        key = (m.group(2), int(m.group(3)))
        attempt = int(m.group(4) or 1)
        cands.setdefault(key, []).append((attempt, t))
    chosen, unused = {}, []
    for key, lst in cands.items():
        lst.sort()
        chosen[key] = lst[-1][1]                       # 最後にやり直したもの
        unused += [t for _, t in lst[:-1]]
    return chosen, unused


def file_label(sdir):
    """aim_kakeru400_20261022_123951 → ("dpi400", "kakeru")．
    形が違えば ("dpi_unknown", aim_ を除いたフォルダ名)．"""
    m = SESSION_RE.match(sdir.name)
    if not m:
        name = sdir.name[len("aim_"):] if sdir.name.startswith("aim_") else sdir.name
        return "dpi_unknown", name
    return f"dpi{m.group(2)}", m.group(1)


def file_labels(sdirs):
    """{セッション: (dpi フォルダ, ファイル名)}．同じ参加者・dpi のセッションが複数あれば日時も付けて区別する．"""
    base = {d: file_label(d) for d in sdirs}
    labels = {}
    for d, lab in base.items():
        if list(base.values()).count(lab) > 1:      # 重複したものは全部に日時を付ける
            m = SESSION_RE.match(d.name)
            labels[d] = (lab[0], f"{lab[1]}_{m.group(3)}" if m else d.name)
        else:
            labels[d] = lab
    return labels


def out_path(label, band, seed):
    dpi, name = label
    return OUT_DIR / f"{band}_trial{seed}" / dpi / f"{name}.mp4"


# ══════════════════════════════════════════════════
#  task.csv（課題の時刻）
# ══════════════════════════════════════════════════
def task_clock(trial_dir):
    """課題の t = 0 の epoch [s] と t の最後 [s]．task.csv が無ければ (None, None)．"""
    path = trial_dir / "task.csv"
    if not path.is_file():
        return None, None
    df = pd.read_csv(path, usecols=["t", "epoch"]).dropna()
    if df.empty:
        return None, None
    t, ep = df["t"].to_numpy(float), df["epoch"].to_numpy(float)
    epoch0 = float(np.interp(0.0, t, ep)) if t[0] <= 0.0 <= t[-1] else float(ep[0] - t[0])
    return epoch0, float(t[-1])


# ══════════════════════════════════════════════════
#  動画の書き出し
# ══════════════════════════════════════════════════
def put_text(img, lines):
    y = 24
    for s in lines:
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def make_video(trial_dir, out, title, task_only, scale):
    from experiment.lib.rs_device import Playback             # pyrealsense2 は書き出すときだけ読む

    epoch0, t_end = task_clock(trial_dir)
    if task_only and epoch0 is None:
        print(f"  [WARN] task.csv が無いので全区間を書きます: {trial_dir.name}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")    # 途中で止めても壊れた動画を残さない
    writer, n, filled = None, 0, 0
    first = base = prev = None                     # 記録の先頭 / 動画の先頭の時刻 [s]，直前に書いた画像
    try:
        with Playback(trial_dir / config.DB3_NAME) as pb:
            fps = float(pb.stream.fps())           # .db3 に記録されたカラーのフレームレート
            for stamp, img in pb.color_frames():   # 重複フレームは Playback が除く
                first = stamp if first is None else first
                lines = [title]
                if epoch0 is not None:
                    tt = stamp - epoch0
                    if task_only and not (0.0 <= tt <= t_end):
                        continue
                    lines.append(f"t = {tt:+7.2f} s")
                lines.append(f"rec {stamp - first:6.2f} s")
                if scale != 1.0:
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                put_text(img, lines)
                if writer is None:
                    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (img.shape[1], img.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"動画ファイルを開けません: {tmp}")
                    base = stamp

                # このフレームが入るべき位置（動画の先頭から何枚目か）．
                # 落ちたフレームの分は直前の画像で埋め，再生時間を実時間に合わせる
                slot = int(round((stamp - base) * fps))
                gap = slot - n
                if gap > MAX_FILL_S * fps:         # 時刻が飛びすぎ（時計の異常）．埋めずに続ける
                    print(f"\n  [WARN] {gap / fps:.1f} s の空白（{stamp - first:.2f} s 付近）．埋めません", end="")
                    base += gap / fps
                elif gap > 0:
                    for _ in range(gap):
                        writer.write(prev)
                    n += gap
                    filled += gap
                writer.write(img)                  # 時刻の揺れで slot < n のときもそのまま次に書く
                n += 1
                prev = img
    finally:
        if writer is not None:
            writer.release()
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("書けるフレームがありません")
    tmp.replace(out)
    return n, filled, fps


# ══════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="aim_* の slow1〜3・fast1〜3 の RGB 動画を書き出す")
    ap.add_argument("--plan", action="store_true", help="対象と書き出し先だけ表示して終わる")
    ap.add_argument("--task-only", action="store_true", help="課題の区間（task.csv の t = 0〜最後）だけ書く")
    ap.add_argument("--overwrite", action="store_true", help="既にある動画も作り直す")
    ap.add_argument("--scale", type=float, default=1.0, help="画像の拡大率（0.5 で半分）")
    args = ap.parse_args()

    jobs = []
    sdirs = session_dirs()
    labels = file_labels(sdirs)
    dup = [d.name for d in sdirs if labels[d] != file_label(d)]
    if dup:
        print("[INFO] 同じ参加者・dpi のセッションが複数あるので，ファイル名に日時を付けます: " + ", ".join(dup))
    for sdir in sdirs:
        chosen, unused = pick_trials(sdir)
        print(f"\n[{sdir.name}]")
        for band in BANDS:
            for seed in SEEDS:
                t = chosen.get((band, seed))
                if t is None:
                    print(f"  {band}{seed}: [無し] trial_*_{band}_s{seed}/session.db3 が見つかりません")
                    continue
                out = out_path(labels[sdir], band, seed)
                done = out.is_file() and not args.overwrite
                print(f"  {band}{seed}: {t.name}  →  {out.relative_to(config.ROOT_DIR)}"
                      + ("  （作成済み）" if done else ""))
                if not done:
                    jobs.append((sdir, band, seed, t, out))
        for t in unused:
            print(f"  [使わない] {t.name}（同じ帯域・seed のやり直しがあるため）")

    print(f"\n作成する動画: {len(jobs)} 本")
    if args.plan or not jobs:
        return

    ok, failed, dropped, t0 = 0, [], [], time.time()
    for k, (sdir, band, seed, t, out) in enumerate(jobs, 1):
        title = f"{sdir.name}  {band}{seed} ({t.name})"
        print(f"[{k}/{len(jobs)}] {sdir.name} {band}{seed} ...", end="", flush=True)
        s = time.time()
        try:
            n, filled, fps = make_video(t, out, title, args.task_only, args.scale)
            ok += 1
            drop = f"，落ちたフレーム {filled} 枚を直前の画像で補完" if filled else ""
            print(f" {n} フレーム（{fps:g} fps，{n / fps:.1f} s{drop}），{time.time() - s:.0f} s")
            if filled:
                dropped.append(f"{sdir.name} {band}{seed}: {filled} 枚（{100 * filled / n:.1f}%）")
        except KeyboardInterrupt:
            print("\n[中断] Ctrl+C")
            break
        except Exception as e:                    # 1 本壊れていても残りは続ける
            failed.append(f"{sdir.name} {band}{seed}: {e}")
            print(f" [失敗] {e}")

    print(f"\n完了 {ok} 本 / 失敗 {len(failed)} 本（{(time.time() - t0) / 60:.1f} 分）")
    for f in failed:
        print(f"  {f}")
    if dropped:
        print(f"落ちたフレームを補完した動画 {len(dropped)} 本:")
        for d in dropped:
            print(f"  {d}")
    print(f"書き出し先: {OUT_DIR}")


if __name__ == "__main__":
    main()