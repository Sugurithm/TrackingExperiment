#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
stroke_compare.py — 同じ条件で，人によって関節の使い方（リンクの寄与）がどう違うか

  python temp/stroke_compare.py --list                  # 並べたセッションと選べる試行を表示
  python temp/stroke_compare.py                         # compare_config.py の COMPARES を全部
  python temp/stroke_compare.py --compare slow_s1_all   # 1 つだけ
  python temp/stroke_compare.py --config temp/compare_config.py --out temp/out

設定は temp/compare_config.py（どのセッション・どの試行を比べるか，軸の数，大小の境）．

やること
  1. stroke_feasibility.py と同じ方法で，ターゲットの 1 ストローク（目標の速さの谷から谷）ごとに，
     手先（mcp3）の変位を 5 つのリンク（肩の点 S・上腕 U・前腕 F・モデルのずれ M・手 H）に足し算で分ける．
  2. ストロークを「方向の軸」×「大きさ（小 / 大）」に分ける．
        方向: ターゲットの変位の向き．対向を畳む．AXES=4 … 横 / 斜め(/) / 縦 / 斜め(\)，AXES=3 … 横 / 斜め / 縦
              「/」= マウスを右へ＋奥へ（または左へ＋手前へ）
        大きさ: 同じ条件（dpi，帯域）のターゲットのストロークの中央値より小さいか大きいか（SIZE_SPLIT で変えられる）
  3. セルごとに，リンクの寄与割合 = Σ s_k / Σ|D|（ストロークを足してから割る．小さい動きの割合が暴れにくい）を出す．
        近位 = S + U，前腕 = F，遠位 = M + H にもまとめる．
     試行ごとのセルと，人 × 条件のセル（試行をまとめたもの）の両方を出す．
  4. 同じ条件の中で，人の間のばらつき（between）と，同じ人の試行間のばらつき（within）を並べる．

使える範囲
  - 統計的な検定はしない．本数 (n) と，足りないセルの表示までが目的．
  - 同じ帯域・同じ seed のターゲットの軌道は誰がやっても同じ．その試行を選べば，全員が同じ動きを追った結果を比べられる．
  - 「検出可能」なストローク（手先の変位 > SNR × ノイズ）だけを使う．

出力: temp/out/<比較の名前>/（--out で変えられる）
  cells_trial.csv   試行 × 軸 × 大きさ
  cells_person.csv  人 × 条件 × 軸 × 大きさ（その比較で選んだ試行をまとめたもの）
  cells_group.csv   by="group" のとき．グループごとの人の平均
  report.txt        本数の表，リンクの寄与，between / within
  cond_<dpi>_<帯域>.png   条件ごとの図（人 × 小/大の積み上げ棒．近位 / 前腕 / 遠位）
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stroke_feasibility as SF  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

AXIS_NAMES = {4: ["horiz", "diag/", "vert", "diag\\"], 3: ["horiz", "diag", "vert"]}
SIZE_NAMES = ["small", "large"]
GROUPS = {"prox": ("S", "U"), "fore": ("F",), "hand": ("M", "H")}
GROUP_COLORS = {"prox": "#4c78a8", "fore": "#f58518", "hand": "#54a24b"}


# ---------------------------------------------------------------- 設定
def load_config(path):
    spec = importlib.util.spec_from_file_location("compare_config", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    d = dict(SESSIONS=[], COMPARES=[], AXES=4, SIZE_SPLIT="median", ATTEMPT="last", MIN_N_TRIAL=3,
             MIN_N_PERSON=8, OUT="out", FC=3.0, MIN_NOISE=2.0, SNR=2.0)
    for k in d:
        if hasattr(m, k):
            d[k] = getattr(m, k)
    return d


def resolve_dir(p):
    if os.path.isabs(p) and os.path.isdir(p):
        return os.path.normpath(p)
    for base in (os.getcwd(), os.path.dirname(HERE)):
        q = os.path.join(base, p)
        if os.path.isdir(q):
            return os.path.normpath(os.path.abspath(q))
    return None


def default_label(sdir):
    name = os.path.basename(os.path.normpath(sdir))
    m = re.match(r"^aim_(.+?)_\d{8}_\d{6}$", name)
    return m.group(1) if m else name


def resolve_sessions(cfg):
    out, seen = [], set()
    for e in cfg["SESSIONS"]:
        d = resolve_dir(e["dir"])
        if d is None:
            print(f"[WARN] フォルダがありません: {e['dir']}")
            continue
        person, dpi_true, _ = SF.session_info(d)
        s = dict(dir=d, person=e.get("person") or person, group=e.get("group") or "",
                 label=e.get("label") or default_label(d), dpi=e.get("dpi"))
        if s["label"] in seen:
            print(f"[WARN] label が重複: {s['label']}（label を付けて区別してください）")
        seen.add(s["label"])
        out.append(s)
    by = {}
    for s in out:
        _, dpi_true, _ = SF.session_info(s["dir"])
        by.setdefault((s["person"], s["dpi"] or dpi_true), []).append(s["label"])
    for (person, dpi), labels in by.items():
        if len(labels) > 1:
            print(f"[WARN] person '{person}' ({dpi:g} dpi) が {len(labels)} セッションに割り当てられています: {labels}"
                  " → 1 人として合算されます．person の書き間違い（コピペ）でないか確認してください")
    return out


def trial_tokens(band, seed, attempt, tdir):
    return {f"{band}_s{seed}", os.path.basename(tdir)}


def pick_attempts(found, policy):
    """同じ (セッション, 帯域, seed) の中でやり直し (_2, _3) をどう扱うか"""
    if policy == "all":
        return found
    keep = {}
    for f in found:
        key = (f[0], f[2], f[3])
        if key not in keep:
            keep[key] = f
        elif (policy == "last" and f[4] > keep[key][4]) or (policy == "first" and f[4] < keep[key][4]):
            keep[key] = f
    return [f for f in found if keep[(f[0], f[2], f[3])] is f]


def session_match(s, spec):
    if spec in ("all", None):
        return True
    spec = [spec] if isinstance(spec, str) else list(spec)
    return any(x in (s["label"], s["person"], s["group"]) for x in spec)


# ---------------------------------------------------------------- 計算
def axis_of(dx, dy, mode):
    th = np.degrees(np.arctan2(dy, dx)) % 180.0
    if mode == 3:
        phi = min(th, 180.0 - th)
        return 0 if phi < 30 else (1 if phi < 60 else 2)
    return int(((th + 22.5) % 180.0) // 45)


def size_thresholds(rows, split):
    """条件 (dpi, 帯域) ごとの大小の境 [mm]．ターゲットだけで決まる（誰がやっても同じ）ので全ストロークを使う"""
    if isinstance(split, (int, float)):
        return None
    uniq = {}
    for r in rows:
        uniq[(r["dpi"], r["band"], r["seed"], r["stroke"])] = r["A_mm"]
    by = {}
    for (dpi, band, _, _), A in uniq.items():
        by.setdefault((dpi, band), []).append(A)
    return {k: float(np.median(v)) for k, v in by.items()}


def aggregate(rows):
    D = sum(r["D_mm"] for r in rows)
    out = {"n": len(rows), "D_sum_mm": D}
    for l in SF.LINKS:
        out[f"share_{l}"] = sum(r[f"s_{l}"] for r in rows) / D if D > 0 else float("nan")
    for g, ls in GROUPS.items():
        out[f"share_{g}"] = sum(out[f"share_{l}"] for l in ls)
    return out


def cells_for(rows, mode, thr, split):
    """rows(使えるストロークだけ) → {(軸, 大小): 集計}"""
    cells = {}
    for r in rows:
        a = axis_of(r["dx_mm"], r["dy_mm"], mode)
        cut = float(split) if thr is None else thr[(r["dpi"], r["band"])]
        sz = 0 if r["A_mm"] < cut else 1
        cells.setdefault((a, sz), []).append(r)
    return {k: aggregate(v) for k, v in cells.items()}


def blank(n=0):
    d = {"n": n, "D_sum_mm": 0.0}
    for l in SF.LINKS:
        d[f"share_{l}"] = float("nan")
    for g in GROUPS:
        d[f"share_{g}"] = float("nan")
    return d


# ---------------------------------------------------------------- 出力
def fmt(x, w=5):
    return f"{100 * x:{w}.0f}" if np.isfinite(x) else " " * (w - 1) + "-"


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_cond(path, title, ents, axn, min_n):
    """ents: {名前: {(軸, 大小): 集計}}．軸ごとの小さい図に，名前 × (小, 大) の積み上げ棒"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(ents)
    na = len(axn)
    fig, axs = plt.subplots(1, na, figsize=(max(10, 1.2 * na * len(names) * 0.7 + 2), 4.2), sharey=True, squeeze=False)
    for ai, ax in enumerate(axs[0]):
        for pi, nm in enumerate(names):
            for si in range(2):
                x = pi * 3 + si
                c = ents[nm].get((ai, si))
                if c is None or c["n"] < min_n or not np.isfinite(c["share_prox"]):
                    ax.bar(x, 1.0, color="#eeeeee", edgecolor="#bbbbbb", hatch="//", linewidth=0.5)
                    ax.text(x, 0.5, f"n={0 if c is None else c['n']}", rotation=90, ha="center", va="center", fontsize=6)
                    continue
                b = 0.0
                for g in ("prox", "fore", "hand"):
                    v = c[f"share_{g}"]
                    ax.bar(x, v, bottom=b, color=GROUP_COLORS[g], width=0.85)
                    b += v
                ax.text(x, max(b, 1.0) + 0.02, str(c["n"]), ha="center", fontsize=6)
        ax.set_xticks([pi * 3 + 0.5 for pi in range(len(names))])
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_title(axn[ai], fontsize=9)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-0.3, 1.25)
    axs[0][0].set_ylabel("share of hand displacement\n(left bar = small, right bar = large)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=GROUP_COLORS[g], label=l) for g, l in
                               (("prox", "shoulder+upper arm"), ("fore", "forearm"), ("hand", "wrist gap+hand"))]
                      + [Patch(facecolor="#eeeeee", hatch="//", edgecolor="#bbbbbb", label=f"n < {min_n}")],
                      fontsize=7, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def seed_of(trial_label):
    m = re.search(r"_(slow|fast)_s(\d+)", trial_label)
    return (m.group(1), int(m.group(2))) if m else (trial_label, 0)


def between_within(person_cells, trial_cells, conds, mode, min_p, min_t):
    """条件 × 軸 × 大小 × グループで，人の間の SD と，同じ人の試行間の SD．

    試行ごとに seed（= ターゲットの動き）が違うので，試行間のばらつきには「その seed のストロークの構成」の
    違いが入る（これは全員に共通で，人の違いではない）．そこで within は 2 通り出す:
      within_raw … 試行ごとの値のばらつきそのまま
      within     … 各 seed で 全員の平均を引いてから（seed の効果を除いて）のばらつき．こちらで比を取る
    """
    L = []
    axn = AXIS_NAMES[mode]
    for cond in conds:
        for ai in range(len(axn)):
            for si in range(2):
                for g in ("prox", "fore", "hand"):
                    pv = {}
                    for (person, c), cells in person_cells.items():
                        if c != cond:
                            continue
                        x = cells.get((ai, si))
                        if x and x["n"] >= min_p and np.isfinite(x[f"share_{g}"]):
                            pv[person] = x[f"share_{g}"]
                    if len(pv) < 2:
                        continue
                    between = float(np.std(list(pv.values()), ddof=1))
                    # 試行ごとの値: {person: {seed: value}}
                    tv = {}
                    for (p2, c2, tl), cells in trial_cells.items():
                        x = cells.get((ai, si))
                        if p2 in pv and c2 == cond and x and x["n"] >= min_t and np.isfinite(x[f"share_{g}"]):
                            tv.setdefault(p2, {}).setdefault(seed_of(tl), []).append(x[f"share_{g}"])
                    tv = {p: {k: float(np.mean(v)) for k, v in d.items()} for p, d in tv.items()}
                    raw_vars = [float(np.var(list(d.values()), ddof=1)) for d in tv.values() if len(d) >= 2]
                    seeds = {}
                    for d in tv.values():
                        for k, v in d.items():
                            seeds.setdefault(k, []).append(v)
                    smean = {k: float(np.mean(v)) for k, v in seeds.items() if len(v) >= 3}
                    adj_vars = []
                    for d in tv.values():
                        res = [v - smean[k] for k, v in d.items() if k in smean]
                        if len(res) >= 2:
                            adj_vars.append(float(np.var(res, ddof=1)))
                    raw = float(np.sqrt(np.mean(raw_vars))) if raw_vars else float("nan")
                    adj = float(np.sqrt(np.mean(adj_vars))) if adj_vars else float("nan")
                    L.append(dict(dpi=cond[0], band=cond[1], axis=axn[ai], size=SIZE_NAMES[si], link=g,
                                  n_persons=len(pv), between_sd=between, within_raw_sd=raw, within_sd=adj,
                                  ratio=between / adj if np.isfinite(adj) and adj > 0 else float("nan"),
                                  ratio_raw=between / raw if np.isfinite(raw) and raw > 0 else float("nan"),
                                  n_persons_within=len(adj_vars)))
    return L


def run_compare(cmp, cfg, sessions, found_all, cache, args):
    name = cmp["name"]
    mode = int(cmp.get("axes", cfg["AXES"]))
    split = cmp.get("size_split", cfg["SIZE_SPLIT"])
    policy = cmp.get("attempt", cfg["ATTEMPT"])
    by = cmp.get("by", "person")
    tr_sel = cmp.get("trials", "all")
    tr_sel = None if tr_sel == "all" else ([tr_sel] if isinstance(tr_sel, str) else list(tr_sel))
    sess = {s["dir"]: s for s in sessions if session_match(s, cmp.get("sessions", "all"))}
    found = pick_attempts([f for f in found_all if os.path.normpath(os.path.abspath(f[0])) in sess], policy)
    found = [f for f in found if tr_sel is None or (trial_tokens(f[2], f[3], f[4], f[1]) & set(tr_sel))]
    if not found:
        print(f"[{name}] 対象の試行がありません（sessions / trials の指定を確認: --list）")
        return
    rows, tlabels = [], {}
    for sd, td, band, seed, att in found:
        key = os.path.normpath(os.path.abspath(td))
        if key not in cache:
            r, t = SF.process_trial(sd, td, band, seed, att, args)
            cache[key] = (r, t)
            print(f"  {os.path.basename(sd)}/{os.path.basename(td)}: " +
                  ("skip" if r is None else f"strokes {t['n_strokes']}, valid {t['n_valid']}, lag {t['lag_s']:.2f} s, gain-scale {t['cm_scale']}"))
        r, t = cache[key]
        if r is None:
            continue
        s = sess[os.path.normpath(os.path.abspath(sd))]
        for x in r:
            x = dict(x)
            x.update(person=s["person"], group=s["group"], label=s["label"])
            rows.append(x)
            tlabels[(x["person"], x["dpi"], x["band"], x["trial"], s["label"])] = 1
    if not rows:
        print(f"[{name}] 使える試行がありません")
        return
    thr = size_thresholds(rows, split)
    use = [r for r in rows if r.get("valid") == 1 and r.get("hand_detectable") == 1]
    dropped = len(rows) - len(use)

    # 試行ごと
    by_trial = {}
    for r in use:
        by_trial.setdefault((r["person"], (r["dpi"], r["band"]), r["label"] + "/" + r["trial"]), []).append(r)
    trial_cells = {k: cells_for(v, mode, thr, split) for k, v in by_trial.items()}
    # 人 × 条件
    by_person = {}
    for r in use:
        by_person.setdefault((r["person"], (r["dpi"], r["band"])), []).append(r)
    person_cells = {k: cells_for(v, mode, thr, split) for k, v in by_person.items()}
    person_group = {}
    for r in use:
        person_group[r["person"]] = r["group"]
    conds = sorted({k[1] for k in person_cells}, key=lambda c: (c[1], c[0]))
    axn = AXIS_NAMES[mode]
    minT, minP = int(cfg["MIN_N_TRIAL"]), int(cfg["MIN_N_PERSON"])

    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)
    tr_rows, pr_rows, gr_rows = [], [], []
    for (person, cond, tl), cells in sorted(trial_cells.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
        for (ai, si), c in sorted(cells.items()):
            tr_rows.append(dict(compare=name, person=person, group=person_group.get(person, ""), trial=tl, dpi=cond[0],
                                band=cond[1], axis=axn[ai], size=SIZE_NAMES[si], **c, ok=int(c["n"] >= minT)))
    for (person, cond), cells in sorted(person_cells.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        ntr = len([k for k in trial_cells if k[0] == person and k[1] == cond])
        for (ai, si), c in sorted(cells.items()):
            pr_rows.append(dict(compare=name, person=person, group=person_group.get(person, ""), dpi=cond[0], band=cond[1],
                                axis=axn[ai], size=SIZE_NAMES[si], n_trials=ntr, **c, ok=int(c["n"] >= minP)))
    write_csv(os.path.join(out, "cells_trial.csv"), tr_rows)
    write_csv(os.path.join(out, "cells_person.csv"), pr_rows)

    group_cells = {}
    if by == "group":
        for cond in conds:
            for g in sorted({v for v in person_group.values() if v}):
                ps = [p for p in person_group if person_group[p] == g and (p, cond) in person_cells]
                cells = {}
                for ai in range(len(axn)):
                    for si in range(2):
                        cs = [person_cells[(p, cond)].get((ai, si)) for p in ps]
                        cs = [c for c in cs if c and c["n"] >= minP and np.isfinite(c["share_prox"])]
                        if cs:
                            m = blank(sum(c["n"] for c in cs))
                            for k in m:
                                if k.startswith("share_"):
                                    m[k] = float(np.mean([c[k] for c in cs]))
                            m["n_persons"] = len(cs)
                            cells[(ai, si)] = m
                            gr_rows.append(dict(compare=name, group=g, dpi=cond[0], band=cond[1], axis=axn[ai],
                                                size=SIZE_NAMES[si], n_persons=len(cs), **{k: v for k, v in m.items() if k != "n_persons"}))
                group_cells[(g, cond)] = cells
        write_csv(os.path.join(out, "cells_group.csv"), gr_rows)

    # レポート
    L = [f"[{name}] sessions: {sorted({s['label'] for s in sess.values()})}",
         f"trials: {sorted({os.path.basename(f[1]) for f in found})}",
         f"strokes used: {len(use)} (dropped {dropped}: not valid / hand not measurable)",
         f"axes: {mode} {axn}, size split: {split if thr is None else 'median per (dpi, band)'}",
         f"cell is shown when n >= {minT} per trial, n >= {minP} per person x condition", ""]
    if thr:
        L.append("size boundary [mm] per condition: " + ", ".join(f"{k[0]:g}dpi {k[1]}: {v:.1f}" for k, v in sorted(thr.items(), key=lambda kv: (kv[0][1], kv[0][0]))))
        L.append("")
    for cond in conds:
        L.append(f"=== {cond[0]:g} dpi / {cond[1]} ===")
        L.append("axis   size  | person   n  trials |  prox  fore  hand  (%)   S    U    F    M    H")
        for ai in range(len(axn)):
            for si in range(2):
                for (person, c), cells in sorted(person_cells.items(), key=lambda kv: str(kv[0][0])):
                    if c != cond:
                        continue
                    x = cells.get((ai, si), blank())
                    ntr = len([k for k in trial_cells if k[0] == person and k[1] == cond])
                    ok = x["n"] >= minP
                    row = f"{axn[ai]:<6} {SIZE_NAMES[si]:<5} | {str(person):<8}{x['n']:>3} {ntr:>5}   | "
                    if ok and np.isfinite(x["share_prox"]):
                        row += f"{fmt(x['share_prox'])} {fmt(x['share_fore'])} {fmt(x['share_hand'])}      " + \
                               " ".join(fmt(x[f'share_{l}'], 4) for l in SF.LINKS)
                    else:
                        row += "  (n too small)"
                    L.append(row)
        L.append("")
    bw = between_within(person_cells, trial_cells, conds, mode, minP, minT)
    write_csv(os.path.join(out, "between_within.csv"), bw)
    if bw:
        L.append("=== between-person SD vs within-person (trial-to-trial) SD of the share [percentage points] ===")
        L.append("within = trial-to-trial SD after removing the seed (target movement) effect shared by everyone; within_raw = without that removal")
        L.append("ratio = between / within.  ratio >> 1 suggests the people differ more than one person varies across trials")
        L.append("(descriptive only: few persons / trials; no test)")
        L.append("cond            axis   size  link | persons  between  within  within_raw  ratio")
        nan = float("nan")
        for b in bw:
            L.append(f"{b['dpi']:>4g}dpi {b['band']:<5} {b['axis']:<6} {b['size']:<5} {b['link']:<4} | {b['n_persons']:>4}   "
                     f"{100 * b['between_sd']:>6.1f}  {100 * b['within_sd'] if np.isfinite(b['within_sd']) else nan:>6.1f}  "
                     f"{100 * b['within_raw_sd'] if np.isfinite(b['within_raw_sd']) else nan:>9.1f}  "
                     f"{b['ratio'] if np.isfinite(b['ratio']) else nan:>5.1f}")
    txt = "\n".join(L)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    for cond in conds:
        ents = {p: person_cells[(p, cond)] for (p, c) in sorted(person_cells, key=lambda k: str(k[0])) if c == cond}
        if by == "group":
            ents.update({f"[{g}]": cells for (g, c), cells in group_cells.items() if c == cond})
        plot_cond(os.path.join(out, f"cond_{cond[0]:g}dpi_{cond[1]}.png"), f"{name}: {cond[0]:g} dpi / {cond[1]}", ents, axn, minP)
    print("\n" + txt)
    print(f"\n-> {out}")


def main():
    ap = argparse.ArgumentParser(description="同じ条件での人の違い（軸 × 大きさ別のリンクの寄与）")
    ap.add_argument("--config", default=os.path.join(HERE, "compare_config.py"))
    ap.add_argument("--compare", nargs="*", help="実行する比較の名前（省略 = 全部）")
    ap.add_argument("--list", action="store_true", help="セッションと選べる試行を表示して終わる")
    ap.add_argument("--out", default=None, help="出力先（省略 = temp/out/）")
    ap.add_argument("--noise-split", type=float, default=5.0)
    ap.add_argument("--min-cov", type=float, default=0.8)
    ap.add_argument("--max-gap", type=float, default=0.5)
    ap.add_argument("--trust-json", action="store_true", help="dpi はフォルダ名でなく session.json を信じる")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.out is None:    # 設定の OUT は temp/（このファイルの場所）からの相対
        args.out = cfg["OUT"] if os.path.isabs(cfg["OUT"]) else os.path.join(HERE, cfg["OUT"])
    args.fc, args.min_noise, args.snr = cfg["FC"], cfg["MIN_NOISE"], cfg["SNR"]
    sessions = resolve_sessions(cfg)
    if not sessions:
        print("SESSIONS が空，またはフォルダが見つかりません（compare_config.py を編集）")
        return
    args.dpi_override = {s["dir"]: s["dpi"] for s in sessions if s["dpi"]}
    found_all = []
    for s in sessions:
        found_all += SF.discover([s["dir"]])
    if args.list:
        for s in sessions:
            f = pick_attempts([x for x in found_all if os.path.normpath(os.path.abspath(x[0])) == s["dir"]], "all")
            toks = sorted({f"{x[2]}_s{x[3]}" + (f"(x{x[4]})" if x[4] > 1 else "") for x in f})
            print(f"{s['label']:<14} person={s['person']:<8} group={s['group'] or '-':<8} trials: {' '.join(toks) or '(joints/task.csv が揃った試行なし)'}")
        return
    cmps = [c for c in cfg["COMPARES"] if not args.compare or c["name"] in args.compare]
    if not cmps:
        print("実行する比較がありません（--compare の名前を確認）")
        return
    cache = {}
    for c in cmps:
        print(f"\n##### {c['name']}")
        run_compare(c, cfg, sessions, found_all, cache, args)


if __name__ == "__main__":
    main()