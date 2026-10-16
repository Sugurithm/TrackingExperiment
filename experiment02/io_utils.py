"""
io_utils.py
────────────────────────────────────────────────────────────────────
フォルダの作成・選択と JSON の読み書き．
記録プログラムと解析プログラムの両方から使う．
────────────────────────────────────────────────────────────────────
"""

import datetime
import json
from pathlib import Path

import config


# ══════════════════════════════════════════════════
#  JSON
# ══════════════════════════════════════════════════
def save_json(path, obj):
    """常に UTF-8 で書く（Windows 既定の cp932 で書かれるのを防ぐ）．"""
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path):
    raw = Path(path).read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:          # 旧スクリプトが Windows で書いた JSON は cp932
        return json.loads(raw.decode("cp932"))


# ══════════════════════════════════════════════════
#  フォルダ
# ══════════════════════════════════════════════════
def now_str(fmt="%Y%m%d_%H%M%S"):
    return datetime.datetime.now().strftime(fmt)


def new_session_dir(prefix, fmt="%Y%m%d_%H%M%S"):
    """recordings/<prefix><日時>/ を新しく作る．同名があれば _2, _3 … を付ける．"""
    base = config.RECORDINGS_DIR / (prefix + now_str(fmt))
    path, k = base, 2
    while path.exists():                 # 同じ分に2回起動しても上書きしない
        path = base.with_name(f"{base.name}_{k}")
        k += 1
    path.mkdir(parents=True)
    return path


def resolve_session_dir(arg, prefix):
    """引数で指定されたフォルダ，なければ recordings/<prefix>* の最新を返す．"""
    if arg:
        path = Path(arg)
        if not path.is_dir():            # 指定ミスはここで止める（別の日のデータを処理しないため）
            raise SystemExit(f"[ERROR] フォルダがありません: {path}")
        return path
    cands = sorted(p for p in config.RECORDINGS_DIR.glob(prefix + "*") if p.is_dir())
    if not cands:
        raise SystemExit(f"[ERROR] {config.RECORDINGS_DIR}/{prefix}* が見つかりません")
    return cands[-1]                     # フォルダ名が日時なので名前順の最後 = 最新
