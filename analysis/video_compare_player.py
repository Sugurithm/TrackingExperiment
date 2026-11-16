# -*- coding: utf-8 -*-
"""
tile_player.py
────────────────────────────────────────────────────────────────────
trial_videos.py で書き出した mp4 を，格子状に並べて同時に再生する（目視で見比べるため）．

  python experiment/tile_player.py recordings/videos/slow_trial1/dpi400
  python experiment/tile_player.py recordings/videos/slow_trial1/dpi400 recordings/videos/slow_trial1/dpi800
  python experiment/tile_player.py recordings/videos/slow_trial1/dpi400/kakeru.mp4 recordings/videos/slow_trial1/dpi800/kakeru.mp4
  python experiment/tile_player.py recordings/videos/slow_trial1/dpi400 --cols 3 --width 1800
  python experiment/tile_player.py recordings/videos/slow_trial1/dpi400 --save tiled.mp4   # 画面に出さず並べた動画を書く

引数: フォルダ（中の *.mp4 をすべて，名前順）または mp4 ファイル．いくつでも並べられる．
      フォルダを複数渡すと，各動画のラベルにフォルダ名も付ける（dpi400/kakeru など）．

【時刻の揃え方】
  どの動画も 1 フレーム目を揃えて再生する．課題の開始（t = 0）で揃えたいときは，
  trial_videos.py を --task-only で書き出した動画を使う．

【操作】
  スペース   一時停止 / 再開
  → / d      1 フレーム進む（一時停止中）      ← / a   1 フレーム戻る
  l / j      5 s 進む / 戻る                   r       最初から
  ] / [      再生速度を 2 倍 / 半分（0.125〜2 倍）
  s          今の画面を PNG で保存（最初の引数のフォルダに）
  q / Esc    終了
────────────────────────────────────────────────────────────────────
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import cv2

# waitKeyEx の矢印キー（Windows）
KEY_LEFT, KEY_RIGHT = 2424832, 2555904
LABEL_H = 28                       # 各マスの下のラベルの高さ [px]


class Clip:
    """1 本の動画．read(i) で i フレーム目を返す（終わったら最後のフレームを暗くして返す）．"""

    def __init__(self, path, label):
        self.path, self.label = path, label
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"開けません: {path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        self.pos = 0                    # 次に cap.read() で読まれるフレーム番号
        self.last = None

    def read(self, i):
        if i >= self.n and self.last is not None:
            return (self.last * 0.4).astype(np.uint8), True     # 終わった動画は暗くして止める
        if i != self.pos:               # 順番に読むとき以外だけシークする（シークは遅い）
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.pos = i
        ok, img = self.cap.read()
        if not ok:                      # フレーム数の申告が実際より多いことがある
            self.n = i
            return self.read(i) if self.last is not None else (None, True)
        self.pos += 1
        self.last = img
        return img, False

    def close(self):
        self.cap.release()


def collect(paths):
    """引数（フォルダ / mp4）から [(mp4 のパス, ラベル)] を作る．"""
    multi = sum(Path(p).is_dir() for p in paths) > 1
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            files = sorted(f for f in p.glob("*.mp4") if not f.name.endswith(".part.mp4"))
            if not files:
                print(f"[WARN] mp4 がありません: {p}")
            out += [(f, f"{p.name}/{f.stem}" if multi else f.stem) for f in files]
        elif p.is_file():
            out.append((p, f"{p.parent.name}/{p.stem}"))
        else:
            print(f"[WARN] 見つかりません: {p}")
    return out


def layout(n, cols, width, aspect):
    cols = cols or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cw = width // cols
    ch = int(round(cw / aspect))
    return rows, cols, cw, ch


def compose(clips, i, rows, cols, cw, ch, status):
    canvas = np.zeros((rows * (ch + LABEL_H), cols * cw, 3), np.uint8)
    for k, c in enumerate(clips):
        img, ended = c.read(i)
        r, q = divmod(k, cols)
        y, x = r * (ch + LABEL_H), q * cw
        if img is not None:
            canvas[y:y + ch, x:x + cw] = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
        label = c.label + ("  (end)" if ended else "")
        cv2.putText(canvas, label, (x + 6, y + ch + LABEL_H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x, y), (x + cw - 1, y + ch + LABEL_H - 1), (80, 80, 80), 1)
    if status:                          # 右下に再生位置と速度
        (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(canvas, status, (canvas.shape[1] - tw - 8, canvas.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_tiled(clips, out, geom, fps):
    n_max = max(c.n for c in clips)
    writer = None
    try:
        for i in range(n_max):
            img = compose(clips, i, *geom, f"{i / fps:6.2f} s")
            if writer is None:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (img.shape[1], img.shape[0]))
            writer.write(img)
            if i % int(fps) == 0:
                print(f"\r[SAVE] {i / fps:6.1f} / {n_max / fps:.1f} s", end="", flush=True)
    finally:
        if writer is not None:
            writer.release()
    print(f"\n[SAVE] {out}")


def play(clips, geom, fps, snap_dir):
    n_max = max(c.n for c in clips)
    win = "tile_player"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    i, paused, speed = 0, False, 1.0
    redraw = True
    next_due = time.perf_counter()
    while True:
        if redraw:
            status = f"{i / fps:6.2f} / {n_max / fps:.2f} s  x{speed:g}" + ("  PAUSE" if paused else "")
            frame = compose(clips, i, *geom, status)
            cv2.imshow(win, frame)
            redraw = False

        # 次のフレームまでの待ち時間（速度に合わせる．描画が遅いときは待たずに次へ）
        wait = 30 if paused else max(1, int((next_due - time.perf_counter()) * 1000))
        key = cv2.waitKeyEx(wait)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:   # × で閉じた
            break

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
            next_due = time.perf_counter()
            redraw = True
        elif key in (KEY_RIGHT, ord("d")):
            paused, i, redraw = True, min(i + 1, n_max - 1), True
        elif key in (KEY_LEFT, ord("a")):
            paused, i, redraw = True, max(i - 1, 0), True
        elif key == ord("l"):
            i, redraw = min(i + int(5 * fps), n_max - 1), True
        elif key == ord("j"):
            i, redraw = max(i - int(5 * fps), 0), True
        elif key == ord("r"):
            i, redraw = 0, True
        elif key == ord("]"):
            speed, redraw = min(speed * 2, 2.0), True
        elif key == ord("["):
            speed, redraw = max(speed / 2, 0.125), True
        elif key == ord("s"):
            path = snap_dir / f"tile_{i:06d}.png"
            cv2.imwrite(str(path), frame)
            print(f"[SNAP] {path}")
        elif not paused and time.perf_counter() >= next_due:
            if i + 1 >= n_max:          # 最後まで来たら止める
                paused, redraw = True, True
            else:
                i += 1
                redraw = True
            next_due += 1.0 / (fps * speed)
            if next_due < time.perf_counter() - 0.5:     # 大きく遅れたら追いつこうとしない
                next_due = time.perf_counter()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="mp4 を格子状に並べて同時に再生する")
    ap.add_argument("paths", nargs="+", help="フォルダ（中の *.mp4 すべて）または mp4 ファイル")
    ap.add_argument("--cols", type=int, default=0, help="列の数（既定は本数の平方根を切り上げ）")
    ap.add_argument("--width", type=int, default=1800, help="全体の横幅 [px]")
    ap.add_argument("--save", help="画面に出さず，並べた動画をこの mp4 に書く")
    args = ap.parse_args()

    items = collect(args.paths)
    if not items:
        raise SystemExit("[ERROR] 再生する mp4 がありません")
    clips = [Clip(p, lab) for p, lab in items]
    try:
        fps_set = sorted({round(c.fps, 2) for c in clips})
        if len(fps_set) > 1:
            print(f"[WARN] fps が揃っていません {fps_set}．最初の動画の fps で再生します")
        fps = clips[0].fps
        h, w = clips[0].read(0)[0].shape[:2]
        geom = layout(len(clips), args.cols, args.width, w / h)
        for c in clips:
            print(f"  {c.label:30s} {c.n / c.fps:6.1f} s  ({c.n} フレーム, {c.fps:g} fps)")

        if args.save:
            save_tiled(clips, Path(args.save), geom, fps)
        else:
            first = Path(args.paths[0])
            play(clips, geom, fps, first if first.is_dir() else first.parent)
    finally:
        for c in clips:
            c.close()


if __name__ == "__main__":
    main()