"""
stash_db3.py
各試行の session.db3 を、名前を付け替えて別フォルダに「移動」する（コピーしない）。
aim_* ごとにサブフォルダを分けるので、セッション単位で別々に圧縮・外付けHDDへ移せる。
--restore で元の場所に戻せる。

  python stash_db3.py SRC STASH --dry-run     # 何が動くかだけ表示
  python stash_db3.py SRC STASH               # 移動
  python stash_db3.py SRC STASH --restore     # STASH 内のファイルを元の試行フォルダへ戻す

  SRC   : aim_* フォルダが並ぶ親フォルダ（例: ../recordings）
  STASH : 集める先のフォルダ。SRC と同じドライブに置くこと（名前変更だけで移動する）

移動後の構成
  STASH/
    aim_xxx_YYYYMMDD_HHMMSS/
      aim_xxx_YYYYMMDD_HHMMSS__trial_00_static_pre__session.db3
      ...
      db3_manifest.csv        ← このフォルダ分の（元の場所・サイズ）。圧縮するときは一緒に入れる
    aim_yyy_.../
      ...

安全のため
  * コピーは一切しない。os.rename で移動するので、STASH が別ドライブだと動かずにエラーで止まる
  * ファイル名に セッション名__試行名 を入れるので、全部を 1 か所に展開しても名前が重複しない
  * 同名ファイルがあれば上書きせず止める
  * rs_joints.npz が無い試行の db3 は、既定では動かさず警告する（解析に必要なため）。
    動かしたいときだけ --include-unprocessed
"""
import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

MANIFEST = "db3_manifest.csv"
SEP = "__"


def mb(n):
    return f"{n / 1e6:,.1f} MB"


def find_targets(src, include_unprocessed):
    """[(db3, セッション名, 新ファイル名)], [スキップした db3]"""
    todo, skipped = [], []
    for sess in sorted(d for d in src.glob("aim_*") if d.is_dir()):
        for db3 in sorted(sess.glob("*/session.db3")):
            tr = db3.parent
            if not (tr / "joints" / "rs_joints.npz").is_file() and not include_unprocessed:
                skipped.append(db3)
                continue
            todo.append((db3, sess.name, f"{sess.name}{SEP}{tr.name}{SEP}session.db3"))
    return todo, skipped


def stash(a):
    src, stash_dir = Path(a.src).resolve(), Path(a.stash).resolve()
    if not src.is_dir():
        sys.exit(f"[ERROR] SRC がありません: {src}")
    if src in stash_dir.parents or stash_dir == src:
        sys.exit("[ERROR] STASH は SRC の外に作ってください")
    if src.drive.lower() != stash_dir.drive.lower():
        sys.exit(f"[ERROR] STASH は SRC と同じドライブ ({src.drive}) に置いてください（コピーになるため）")
    todo, skipped = find_targets(src, a.include_unprocessed)
    if not todo and not skipped:
        sys.exit("[INFO] session.db3 が見つかりません（移動済みかもしれません）")

    total = 0
    for db3, sess, new in todo:
        total += db3.stat().st_size
        print(f"  {db3.relative_to(src)}  →  {sess}/{new}  ({mb(db3.stat().st_size)})")
    print(f"\n移動 {len(todo)} 件 / {mb(total)} / {len({s for _, s, _ in todo})} セッション")
    if skipped:
        print(f"[WARN] rs_joints.npz が無いので動かさなかった db3: {len(skipped)} 件")
        for s in skipped:
            print(f"       {s.relative_to(src)}")

    names = [new for _, _, new in todo]
    if len(names) != len(set(names)):
        sys.exit("[ERROR] 移動後の名前が重複します")
    clash = [f"{sess}/{new}" for _, sess, new in todo if (stash_dir / sess / new).exists()]
    if clash:
        sys.exit(f"[ERROR] STASH に同名ファイルがあります（上書きしません）: {clash[:3]} ...")
    if a.dry_run:
        print("（--dry-run: 移動していません）")
        return

    by_sess = {}
    for db3, sess, new in todo:
        by_sess.setdefault(sess, []).append((db3, new))
    for sess, items in by_sess.items():
        out = stash_dir / sess
        out.mkdir(parents=True, exist_ok=True)
        man = out / MANIFEST
        new_file = not man.exists()
        with open(man, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["stashed_name", "original_relpath", "size_bytes"])
            for db3, new in items:
                size = db3.stat().st_size
                dest = out / new
                os.rename(db3, dest)                   # 名前変更だけ（コピーしない・上書きしない）
                if dest.stat().st_size != size:
                    sys.exit(f"[ERROR] サイズ不一致: {dest}")
                w.writerow([new, db3.relative_to(src).as_posix(), size])
                f.flush()
        print(f"  [OK] {sess}: {len(items)} 件")
    print(f"[DONE] {stash_dir}\n       各フォルダの {MANIFEST} も一緒に圧縮・外付けHDDへ持っていってください")


def restore(a):
    src, stash_dir = Path(a.src).resolve(), Path(a.stash).resolve()
    # 新形式（aim_*/db3_manifest.csv）と旧形式（STASH 直下の db3_manifest.csv）の両方を見る
    mans = sorted(stash_dir.glob(f"*/{MANIFEST}")) + sorted(stash_dir.glob(MANIFEST))
    if not mans:
        sys.exit(f"[ERROR] {stash_dir} に {MANIFEST} がありません")
    n = 0
    for man in mans:
        for r in csv.DictReader(open(man, encoding="utf-8")):
            f, dest = man.parent / r["stashed_name"], src / r["original_relpath"]
            if not f.is_file():
                continue                               # 一部だけ戻す場合
            if dest.exists():
                print(f"  [SKIP] 既にあります: {dest}")
                continue
            print(f"  {f.relative_to(stash_dir)}  →  {dest}")
            if a.dry_run:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest))
            n += 1
    print(f"[DONE] 戻した件数: {n}" if not a.dry_run else "（--dry-run: 戻していません）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("stash")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--include-unprocessed", action="store_true")
    a = ap.parse_args()
    (restore if a.restore else stash)(a)
