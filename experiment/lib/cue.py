"""
cue.py
────────────────────────────────────────────────────────────────────
マウス操作の目安になる合図音（メトロノーム＋音声の案内）を作って鳴らす．
1 拍ずつ鳴らしたり記録中に読み上げたりすると時間がずれるので，全体を 1 本の wav にしてから再生する．

  schedule()            … 区間の一覧 [(開始 s, 終了 s, 名前), ...]（config.CUE_* から作る）
  render_wav(path)      … 合図を wav に書く．返り値は全体の長さ [s]
  Player(path).start()  … 非同期で再生し，区間の名前を画面にも出す．返り値は再生開始の epoch [s]
  labels_at(cue, t)     … meta.json の "cue" と時刻 t（epoch s）→ 各時刻の区間名（区間外は ""）

【音】
  音声          : 「ボードを振ってください」「テンポ40」「さん，に，いち」「ストップ」など（config.CUE_VOICE）
  カッ（短い）  : 拍．ブロックの最初の拍だけ少し高い（= 動き始め）
  ブー（低い）  : ブロックの終わり（最後の片道を終える時刻）
  ピッピッ      : 音声が使えないときのボード・カウントダウンの代わり
【音声】
  Windows の読み上げ（System.Speech．追加のインストール不要）を PowerShell から呼んで wav にし，混ぜる．
  作った音声は一時フォルダに保存して次回から使い回す．使えなければ音だけになる（警告を出す）．
────────────────────────────────────────────────────────────────────
"""

import hashlib
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

import config

RATE = 22050
VOICE_DIR = Path(tempfile.gettempdir()) / "trial_cue_voice"


# ══════════════════════════════════════════════════
#  スケジュール
# ══════════════════════════════════════════════════
def block_s(bpm):
    """1 ブロックの長さ [s]．拍は 0, 60/bpm, … に打ち，最後の片道が終わる時刻で区切る．"""
    return config.CUE_BEATS * 60.0 / bpm


def schedule():
    """[(開始 s, 終了 s, 名前)]．名前: ready, board, away, still, count, bpmNN, rest, tail"""
    segs, t = [], 0.0

    def add(dur, name):
        nonlocal t
        segs.append((round(t, 3), round(t + dur, 3), name))
        t += dur

    add(config.CUE_READY_S, "ready")
    add(config.CUE_BOARD_S, "board")
    add(config.CUE_AWAY_S, "away")
    add(config.CUE_STILL_S, "still")
    for k, bpm in enumerate(config.CUE_BPMS):
        if k:
            add(config.CUE_REST_S, "rest")
        add(config.CUE_COUNT, "count")
        add(block_s(bpm), f"bpm{bpm}")
    add(config.CUE_REST_S, "rest")
    add(config.CUE_BOARD_S, "board")
    add(config.CUE_TAIL_S, "tail")
    return segs


def events(segs, voice):
    """音の一覧 [(時刻 s, 種類, 中身)]．種類 "tone" の中身は (周波数, 長さ)，"voice" は読み上げる文．"""
    V = config.CUE_VOICE
    out = []
    for k, (t0, t1, name) in enumerate(segs):
        nxt = segs[k + 1][2] if k + 1 < len(segs) else ""
        after = segs[k + 2][2] if k + 2 < len(segs) else ""
        if name == "board":
            out += [(t0, "voice", V["board"])] if voice else [(t0, "tone", (1320, 0.08)), (t0 + 0.15, "tone", (1320, 0.08))]
        elif name == "away":
            out += [(t0, "voice", V["away"])] if voice else [(t0, "tone", (1320, 0.08)), (t0 + 0.15, "tone", (1320, 0.08))]
        elif name == "still" and voice:
            out.append((t0, "voice", V["still"]))
        elif name == "tail" and voice:
            out.append((t0, "voice", V["end"]))
        elif name == "count":
            out += [(t0 + j, "voice", V["count"][j]) if voice and j < len(V["count"]) else (t0 + j, "tone", (660, 0.12))
                    for j in range(config.CUE_COUNT)]
        elif name.startswith("bpm"):
            beat = 60.0 / int(name[3:])
            out += [(t0 + j * beat, "tone", (1760 if j == 0 else 1320, 0.04)) for j in range(config.CUE_BEATS)]
            out.append((t1, "voice", V["stop"]) if voice else (t1, "tone", (330, 0.4)))
        # 次のテンポを，その前の静止（still / rest）の終わり 2.5 s 前に読み上げる
        if voice and name in ("still", "rest") and nxt == "count" and after.startswith("bpm"):
            out.append((t1 - 2.5, "voice", V["tempo"].format(bpm=after[3:])))
    return out


# ══════════════════════════════════════════════════
#  音声（Windows の読み上げ）
# ══════════════════════════════════════════════════
PS_SPEAK = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {{ $s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::NotSet,
       [System.Speech.Synthesis.VoiceAge]::NotSet, 0, [System.Globalization.CultureInfo]'ja-JP') }} catch {{}}
$s.Rate = 1
$s.SetOutputToWaveFile('{path}')
$s.Speak('{text}')
$s.Dispose()
"""


def read_wav(path):
    """wav → float の 1 チャンネル（RATE に直す）．"""
    with wave.open(str(path)) as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        x = np.frombuffer(w.readframes(n), {1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32)
    x = x.reshape(-1, ch).mean(1)
    x = (x - 128) / 128 if sw == 1 else x / float(2 ** (8 * sw - 1))
    if sr != RATE:
        x = np.interp(np.arange(0, len(x), sr / RATE), np.arange(len(x)), x).astype(np.float32)
    return x


def voice_clip(text):
    """読み上げた音声（float 配列）．一時フォルダに保存して使い回す．失敗したら None．"""
    VOICE_DIR.mkdir(exist_ok=True)
    path = VOICE_DIR / (hashlib.md5(text.encode()).hexdigest()[:12] + ".wav")
    if not path.is_file():
        if sys.platform != "win32":
            return None
        script = PS_SPEAK.format(path=str(path).replace("'", "''"), text=text.replace("'", "''"))
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, timeout=30,
                           capture_output=True)
        except (OSError, subprocess.SubprocessError):
            return None
        if not path.is_file():
            return None
    x = read_wav(path)
    nz = np.where(np.abs(x) > 0.01)[0]                 # 前後の無音を削って，時刻どおりに聞こえるようにする
    x = x[nz[0]:nz[-1] + 1] if nz.size else x
    return 0.8 * x / max(np.abs(x).max(), 1e-6)


# ══════════════════════════════════════════════════
#  wav の作成と再生
# ══════════════════════════════════════════════════
def render_wav(path):
    segs = schedule()
    voice = config.CUE_VOICE_ON
    if voice and voice_clip(config.CUE_VOICE["stop"]) is None:
        print("[WARN] 音声の読み上げが使えないので，音だけの合図にします")
        voice = False
    total = segs[-1][1]
    y = np.zeros(int((total + 3.0) * RATE), np.float32)
    for t, kind, what in events(segs, voice):
        if kind == "tone":
            f, d = what
            n = int(d * RATE)
            x = 0.6 * np.sin(2 * np.pi * f * np.arange(n) / RATE) * np.minimum(1, np.linspace(8, 0, n))
        else:
            x = voice_clip(what)
            if x is None:
                continue
        i = int(t * RATE)
        y[i:i + len(x)] += x[:len(y) - i]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return total


class Player:
    """start() で wav を非同期再生し，別スレッドで区間名を表示する．stop() で止める．"""

    def __init__(self, path):
        self.path = path
        self._stop = threading.Event()

    def start(self):
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(str(self.path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            print("[WARN] Windows 以外では合図音は鳴りません（区間の記録だけ行います）")
        self.start_epoch = time.time()
        threading.Thread(target=self._announce, daemon=True).start()
        return self.start_epoch

    def stop(self):
        self._stop.set()
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(None, 0)

    def _announce(self):
        t_mono = time.monotonic()
        for t0, t1, name in schedule():
            if self._stop.wait(max(0.0, t0 - (time.monotonic() - t_mono))):
                return
            print(f"\n[CUE {t0:5.1f} s] {name}（{t1 - t0:.1f} s）", flush=True)


# ══════════════════════════════════════════════════
#  解析側
# ══════════════════════════════════════════════════
def labels_at(cue, t):
    """meta.json の "cue"（start_epoch, segments）→ 時刻 t（epoch s）ごとの区間名．"""
    lab = np.full(len(t), "", dtype=object)
    if not cue:
        return lab
    for t0, t1, name in cue["segments"]:
        lab[(t >= cue["start_epoch"] + t0) & (t < cue["start_epoch"] + t1)] = name
    return lab
