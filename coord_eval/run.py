"""coord_eval — 上肢の関節協調の解析（入口）

  python -m coord_eval.run --plan                 # 対象のセッション・試行の確認だけ
  python -m coord_eval.run                        # 解析 → analysis_out/coord/（report.md・CSV・図）
  python -m coord_eval.run --out analysis_out/coord_20261023 --no-bias
  python -m coord_eval.run --recordings <フォルダ> --groups groups.json   # 合成データなど別の場所

手順
  1. セッション（参加者 × DPI）ごとに試行を読み、机の座標系をセッションで 1 回決める
  2. 静止試行からノイズ床を測る
  3. 試行ごとに H1a / H1b / H2 / 補助の指標（analyses/trial.py）
  4. 低い DPI の試行から擬似的な高 DPI の試行を作り、振幅が縮むことによる偏りを測る（BIAS_CORRECT）
  5. 人 × 条件にまとめ、DPI・帯域での不変性（同等性検定）、個人差、FPS / 非 FPS を検定（analyses/summary.py）
  6. 図と report.md
"""
import argparse, csv, json, os, sys, time
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "coord_eval"

from coord_eval import config as K                       # noqa: E402
from coord_eval.lib import data, noise, plot              # noqa: E402
from coord_eval.analyses import trial as TR, summary as SM, metrics as MT, report as RP   # noqa: E402


def write_csv(path, rows, first=()):
    if not rows:
        open(path, "w", encoding="utf-8-sig").close()
        return path
    keys = list(first) + [k for k in rows[0] if k not in first]
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and not np.isfinite(v) else
                            (round(v, 6) if isinstance(v, float) else v)) for k, v in r.items()})
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="対象の確認だけ")
    ap.add_argument("--out", default=None, help=f"出力先（既定 {os.path.relpath(K.OUT, K.ROOT)}）")
    ap.add_argument("--recordings", default=None, help="recordings の場所（既定: リポジトリの recordings/）")
    ap.add_argument("--groups", default=None, help="群の定義の JSON（{群: [名前, ...]}。合成データ用）")
    ap.add_argument("--no-bias", action="store_true", help="擬似データによる偏りの補正をしない（速い）")
    a = ap.parse_args()
    if a.recordings:
        K.RECORDINGS = os.path.abspath(a.recordings)
    if a.groups:
        K.GROUPS = json.load(open(a.groups, encoding="utf-8"))
        K.GROUP_LABEL = {g: g for g in K.GROUPS} | {k: v for k, v in K.GROUP_LABEL.items() if k in K.GROUPS}
    if a.no_bias:
        K.BIAS_CORRECT = False
    out = os.path.abspath(a.out) if a.out else K.OUT
    t0 = time.time()
    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(s)

    log(f"# coord_eval 実行ログ（{time.strftime('%Y-%m-%d %H:%M:%S')}）\n")
    log(f"recordings: {K.RECORDINGS}\n出力: {out}\n")
    sessions = data.find_sessions(log)
    log(f"\nセッション {len(sessions)} 個:")
    for s in sessions:
        n = len([1 for d, m in data.list_trials(s["dir"], kinds=("track",)) if m["task"].get("band") in K.BANDS])
        log(f"  {s['participant']:8s} {s['group'] or '-':7s} DPI {s['dpi']:5g}（{s['dpi_source']}）"
            f" 試行 {n:2d}  {s['session']}")
    if a.plan or not sessions:
        return
    os.makedirs(out, exist_ok=True)
    d_fig = os.path.join(out, "figures"); os.makedirs(d_fig, exist_ok=True)
    d_dat = os.path.join(out, "data"); os.makedirs(d_dat, exist_ok=True)

    rng = np.random.default_rng(K.SEED)
    trials, moves, frames, floors, bias_rows = [], [], [], [], []
    by_person = {}
    for s in sessions:
        tracks, static, frame = data.load_session(s, log)
        fl = noise.static_floor(static)
        frames.append({"participant": s["participant"], "dpi": s["dpi"], "session": s["session"],
                       **{k: v for k, v in frame.items() if k not in ("R", "origin")}})
        floors.append({"participant": s["participant"], "dpi": s["dpi"], "session": s["session"],
                       **{k: v for k, v in fl.items()}})
        if not fl:
            log(f"  ! {s['session']}: 静止試行が無いのでノイズ床は 1 フレームの差から粗く見積もります")
        for tr in tracks:
            row, mv = TR.compute(tr, fl)
            trials.append(row)
            moves += mv
        by_person.setdefault(s["participant"], []).append((s, tracks, static, fl))
        log(f"  {s['participant']:8s} DPI {s['dpi']:g}: {len(tracks)} 試行（机の座標: ゲイン x {frame['gain_x']:.2f} / "
            f"y {frame['gain_y']:.2f}、残差 {frame['resid_mm']:.1f} mm）")

    # ---- 振幅が縮むことによる偏り（低い DPI の試行を縮めて高い DPI の SNR にする）
    cols = [c for c in MT.COLS if any(c in r for r in trials)]
    if K.BIAS_CORRECT:
        log("\n擬似データで偏りを測っています（低 DPI の試行 × "
            f"{K.BIAS_REPS} 回）…")
        for p, ss in by_person.items():
            dpis = sorted(x[0]["dpi"] for x in ss)
            for lo_i in range(len(dpis)):
                for hi_i in range(lo_i + 1, len(dpis)):
                    s_lo = next(x for x in ss if x[0]["dpi"] == dpis[lo_i])
                    s_hi = next(x for x in ss if x[0]["dpi"] == dpis[hi_i])
                    nz = noise.noise_samples(s_hi[2] if s_hi[2] is not None else s_lo[2], s_lo[1])
                    scale = dpis[lo_i] / dpis[hi_i]
                    fl_hi = s_hi[3] or s_lo[3]
                    for tr in s_lo[1]:
                        base = next(r for r in trials if r["session"] == tr.session and r["trial"] == tr.name)
                        for rep in range(K.BIAS_REPS):
                            ps = noise.pseudo_trial(tr, nz, scale, rng)
                            prow, _ = TR.compute(ps, fl_hi, with_extra=False)
                            bias_rows.append({"participant": p, "band": tr.band, "trial": tr.name, "rep": rep,
                                              "dpi_from": dpis[lo_i], "dpi_to": dpis[hi_i], "noise": nz["_source"],
                                              **{c: prow.get(c, np.nan) - base.get(c, np.nan) for c in cols
                                                 if c in prow and c in base
                                                 and any(c.startswith(k) for k in K.BIAS_METRICS)}})
    # ---- まとめ
    cell = SM.cells(trials, cols)
    bias = SM.bias_table(bias_rows, cols) if bias_rows else {}
    inv = SM.invariance(cell, bias, cols)
    pch = SM.person_changes(trials, cell, bias, cols)
    ind = SM.individuality(trials, cell, cols)
    sig_cols = ["gshare_trunk", "gshare_upperarm", "eshare3_distal", "N_eff3", "C_glob3", "mshare_prox",
                "C_move3", "rslope_prox", "psis_sd_per100"]
    sig_cols = [c for c in sig_cols if c in cols]
    sigm = SM.signature_match(cell, sig_cols, rng)
    grp, pm = SM.group_tests(cell, cols, rng)
    prof = SM.amp_profile(moves)
    mamp = SM.matched_amp_dpi(prof)

    first = ("participant", "group", "dpi", "band", "seed", "trial")
    files = {
        "trials": write_csv(os.path.join(d_dat, "trials.csv"), trials, first),
        "movements": write_csv(os.path.join(d_dat, "movements.csv"), moves, first),
        "cells": write_csv(os.path.join(d_dat, "person_condition.csv"), cell, ("participant", "group", "dpi", "band")),
        "persons": write_csv(os.path.join(d_dat, "person_means.csv"), pm, ("participant", "group")),
        "invariance": write_csv(os.path.join(d_dat, "invariance.csv"), inv, ("metric", "label", "factor", "stratum")),
        "person_changes": write_csv(os.path.join(d_dat, "invariance_person.csv"), pch,
                                    ("participant", "group", "metric", "label", "factor", "stratum")),
        "individuality": write_csv(os.path.join(d_dat, "individuality.csv"), ind, ("metric", "label")),
        "signature": write_csv(os.path.join(d_dat, "signature_match.csv"), sigm),
        "group": write_csv(os.path.join(d_dat, "group_tests.csv"), grp, ("metric", "label", "stratum")),
        "amp_profile": write_csv(os.path.join(d_dat, "amp_profile.csv"), prof),
        "matched_amp": write_csv(os.path.join(d_dat, "matched_amp_dpi.csv"), mamp),
        "bias": write_csv(os.path.join(d_dat, "bias_pseudo.csv"), bias_rows),
        "frames": write_csv(os.path.join(d_dat, "desk_frames.csv"), frames),
        "floors": write_csv(os.path.join(d_dat, "noise_floor.csv"), floors),
    }
    lab = {c: MT.label(c).replace("【参考】", "*") for c in cols}
    main_cols = [c for c in MT.MAIN if c in cols]
    figs = {
        "global": plot.global_shares(cell, os.path.join(d_fig, "fig1_global_shares.png")),
        "move_prox": plot.movement_amp(prof, os.path.join(d_fig, "fig2_movement_prox_by_amp.png")),
        "move_n": plot.movement_amp(prof, os.path.join(d_fig, "fig3_movement_Nmove_by_amp.png"), "N_move",
                                    "1 動作で動いた関節の実効数*"),
        "orient": plot.orientation(cell, os.path.join(d_fig, "fig4_orientation.png")),
        "lines": plot.person_lines(cell, main_cols, lab, os.path.join(d_fig, "fig5_person_by_condition.png")),
        "forest": plot.forest(inv, os.path.join(d_fig, "fig6_invariance.png"),
                              "人ごとの差（後 − 前）の平均と 95 % CI。灰色 = 同等とみなす幅"),
        "group": plot.group_dots(pm, main_cols, lab, os.path.join(d_fig, "fig7_group.png")),
    }
    rp = RP.write(out, sessions, trials, moves, cell, inv, pch, ind, sigm, grp, pm, mamp, frames, floors, figs, files)
    log(f"\n→ {rp}（{time.time() - t0:.0f} s）")
    open(os.path.join(out, "run_log.md"), "w", encoding="utf-8").write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
