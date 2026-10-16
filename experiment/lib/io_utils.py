"""
io_utils.py
────────────────────────────────────────────────────────────────────
フォルダの作成・選択と JSON の読み書き．
記録プログラムと解析プログラムの両方から使う．

  new_session_dir()     … recordings/<prefix><日時>/ を新しく作る
  resolve_session_dir() … 引数のフォルダ，なければ最新の <prefix>* を返す
  find_one()            … フォルダ内のファイルを 1 つだけ探す（MoCap の CSV など）
  calib_for_trial()     … 試行に使う較正フォルダ（meta.json の "calib"，なければ試行より前の最新）
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


def find_one(folder, pattern):
    """folder 内で pattern に合うファイルがちょうど 1 つならそれを返す（0 個・複数個は取り違えの元なので止める）．"""
    cands = sorted(Path(folder).glob(pattern))
    if len(cands) != 1:
        raise SystemExit(f"[ERROR] {folder} に {pattern} を 1 つだけ置いてください（現在 {len(cands)} 個）")
    return cands[0]


def calib_for_trial(trial_dir, override=None):
    """試行の比較に使う較正フォルダ．引数 override（フォルダ，または recordings/ 内の名前）があればそれ，
    なければ meta.json の "calib"，それもなければ試行より前に作られた calib_* のうち T_MC.json があるものの最新．"""
    meta = load_json(Path(trial_dir) / config.META_NAME)
    if override:
        path = Path(override)
        if not path.is_dir():                    # 名前だけ（calib_20261015_1914）でも指定できる
            path = config.RECORDINGS_DIR / override
        if not path.is_dir():
            raise SystemExit(f"[ERROR] 較正フォルダがありません: {override}")
    elif meta.get("calib"):
        path = config.RECORDINGS_DIR / meta["calib"]
    else:
        stamp = Path(trial_dir).name[len(config.TRIAL_PREFIX):]       # YYYYMMDD_HHMMSS
        cands = [p for p in sorted(config.RECORDINGS_DIR.glob(config.CALIB_PREFIX + "*"))
                 if (p / "T_MC.json").is_file() and p.name[len(config.CALIB_PREFIX):] <= stamp]
        if not cands:
            raise SystemExit(f"[ERROR] {Path(trial_dir).name} より前の較正（T_MC.json のある calib_*）がありません")
        path = cands[-1]
    if not (path / "T_MC.json").is_file():
        raise SystemExit(f"[ERROR] {path / 'T_MC.json'} がありません．先に calib_solve.py を実行してください")
    return path