"""1 条件（DPI × 帯域）ぶんを独立に解析して results/<条件>/ に書き出す。"""
import csv, json, os, time
import numpy as np

from . import config as K
from . import tracking, coordination, initiation, stats, figures, report
from .io import C, find_sessions, trial_dirs, load_trial, group_of


def _write_csv(path, rows, first=()):
    if not rows:
        open(path, "w", encoding="utf-8-sig").close()
        return path
    keys = list(first) + [k for k in rows[0] if k not in first]
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:      # Excel で文字化けしないよう BOM 付き
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (v if not isinstance(v, float) or np.isfinite(v) else "") for k, v in r.items()})
    return path


def run_condition(cond, out_root=K.RESULTS, log=print):
    name, dpi, band = cond["name"], cond["dpi"], cond["band"]
    out = os.path.join(out_root, name)
    d_data, d_fig, d_trial = (os.path.join(out, s) for s in ("data", "figures", "trials"))
    for d in (d_data, d_fig, d_trial):
        os.makedirs(d, exist_ok=True)
    t0 = time.time()
    log(f"== {name}（DPI {dpi} / {band}）→ {out}")

    sessions = find_sessions(dpi)
    used, trials, ftrack, fjoint, events, missing, ievents = [], [], [], [], [], [], []
    examples = {}
    for g, members in K.GROUPS.items():
        for person in members:
            sess = sessions.get(person)
            if not sess:
                missing.append({"participant": person, "group": g, "reason": f"DPI {dpi} のセッションが無い"})
                log(f"  ! {person}: DPI {dpi} のセッションが見つかりません")
                continue
            tds = trial_dirs(sess, band)
            if not tds:
                missing.append({"participant": person, "group": g, "reason": f"{band} の試行が無い"})
                continue
            for td, meta in tds:
                tname = os.path.basename(td)
                tr = load_trial(td, d_trial)
                head = {"participant": person, "group": g, "trial": tname, "seed": tr.seed,
                        "dpi": tr.dpi, "band": tr.band, "session": os.path.basename(sess)}
                tres, frows, evs = tracking.compute(tr)
                fmax = tracking.max_input_freq(tr)
                cres, jrows = coordination.compute(tr, fmax)
                if cres is None:
                    log(f"  ! {person} {tname}: 関節データが無いので協調は欠測")
                    cres, jrows = {}, []
                ires, ievs = initiation.compute(tr) if tr.has_rs else (None, None)
                ires, ievs = ires or {}, ievs or []
                row = {**head, **tres, **cres, **ires}
                trials.append(row)
                ievents += [{**head, **e} for e in ievs]
                ftrack += [{**head, **r} for r in frows]
                fjoint += [{**head, **r} for r in jrows]
                events += [{**head, **e} for e in evs]
                used.append({**head, "trial_dir": os.path.relpath(td, K.EXP_ROOT),
                             "has_joints": bool(tr.has_rs), "fs_rs": tr.fs,
                             "mouse_gain_x": cres.get("mouse_gain_x", np.nan),
                             "mouse_gain_y": cres.get("mouse_gain_y", np.nan),
                             "valid_pct": cres.get("valid_pct", np.nan)})
                json.dump({"header": head, "tracking": tres, "coordination": cres}, open(
                    os.path.join(d_trial, f"{person}_{tname}.json"), "w", encoding="utf-8"),
                    ensure_ascii=False, indent=1, default=C._jsonable)
                # 例示の軌跡（各群 + Syou の seed 1）
                if tr.seed == 1:
                    t, tg, cu, on = tracking._uniform(tr)
                    em = np.linalg.norm(cu - tg, axis=1)
                    examples[person] = (g, t, tg, cu, evs, tres["rms_2d"],
                                        float(np.percentile(em[on], 99.5)) if on.any() else np.nan)
                log(f"  {person:7s} ({g}) {tname}: RMS {tres['rms_2d']:.2f} mm, on {tres['on_target_pct']:.1f} %, "
                    f"N_eff {cres.get('N_eff', np.nan):.2f}")
    if not trials:
        log("  対象の試行がありません"); return None

    people = stats.person_table(trials)
    tests = stats.group_tests(people, trials)
    fp_track, fg_track = stats.freq_group_table(ftrack, ["gain", "lag_ms", "rel_err"])
    jkeys = [k for k in (fjoint[0] if fjoint else {}) if k.startswith(("share_", "proj_", "lead_"))]
    fp_joint, fg_joint = stats.freq_group_table(fjoint, jkeys)

    # ---- CSV
    first = ("participant", "group", "trial", "seed")
    files = {
        "sessions_used.csv": _write_csv(os.path.join(d_data, "sessions_used.csv"), used, first),
        "missing.csv": _write_csv(os.path.join(d_data, "missing.csv"), missing),
        "trials.csv": _write_csv(os.path.join(d_data, "trials.csv"), trials, first),
        "persons.csv": _write_csv(os.path.join(d_data, "persons.csv"), people, ("participant", "group")),
        "group_tests.csv": _write_csv(os.path.join(d_data, "group_tests.csv"), tests, ("metric", "label", "family")),
        "freq_tracking_trials.csv": _write_csv(os.path.join(d_data, "freq_tracking_trials.csv"), ftrack, first),
        "freq_tracking_group.csv": _write_csv(os.path.join(d_data, "freq_tracking_group.csv"), fg_track),
        "freq_joint_trials.csv": _write_csv(os.path.join(d_data, "freq_joint_trials.csv"), fjoint, first),
        "freq_joint_person.csv": _write_csv(os.path.join(d_data, "freq_joint_person.csv"), fp_joint),
        "freq_joint_group.csv": _write_csv(os.path.join(d_data, "freq_joint_group.csv"), fg_joint),
        "correction_events.csv": _write_csv(os.path.join(d_data, "correction_events.csv"), events, first),
        "initiation_events.csv": _write_csv(os.path.join(d_data, "initiation_events.csv"), ievents, first),
    }

    # ---- 図
    fam = stats.FAMILY
    tkeys = [k for k, *_ in stats.METRICS if fam[k] in ("追従精度", "滑らかさ", "軌道修正")]
    ckeys = [k for k, *_ in stats.METRICS if fam[k] == "関節の寄与"] + ["C", "S", "fshift_upperarm", "fshift_forearm", "fshift_hand"]
    figs = {}
    figs["tracking_metrics"] = figures.metric_grid(people, tests, tkeys, f"① ターゲット追従特性（{name}）", d_fig, "tracking_metrics.png")
    figs["tracking_by_freq"] = figures.tracking_freq(fg_track, d_fig)
    radius = np.nanmedian([v[6] for v in examples.values()]) if examples else np.nan
    if np.isfinite(radius):
        figures.K_TARGET_R_MM = radius
    ex = []
    for g, members in K.GROUPS.items():
        cand = [(examples[m][5], m) for m in members if m in examples and m not in K.OUTLIER_CANDIDATES]
        if cand:
            cand.sort(); m = cand[len(cand) // 2][1]     # 群の中で RMS 誤差が中央の人
            v = examples[m]; ex.append((f"{K.GROUP_LABEL[g]}: {m}（seed 1）", g, *v[1:5]))
    for m in K.OUTLIER_CANDIDATES:
        if m in examples:
            v = examples[m]; ex.append((f"{m}（{v[0]} 群、seed 1）", v[0], *v[1:5]))
    if ex:
        figs["example_traces"] = figures.example_traces(ex, d_fig)
    figs["correction_amplitude"] = figures.correction_ecdf(events, d_fig)
    figs["coordination_metrics"] = figures.metric_grid(people, tests, ckeys, f"② 関節の寄与と分担（{name}）", d_fig, "coordination_metrics.png")
    figs["joint_shares"] = figures.joint_shares(people, d_fig)
    figs["share_by_freq"] = figures.share_by_freq(fp_joint, d_fig)
    figs["timing"] = figures.timing(people, tests, d_fig)
    figs["initiation"] = figures.initiation(people, tests, ievents, d_fig)
    figs["syou_sensitivity"] = figures.syou_sensitivity(tests, d_fig)

    # ---- レポート
    rp = report.write(out, cond, people, tests, used, missing, fg_track, fg_joint, figs, files, radius)
    json.dump({"condition": cond, "n_trials": len(trials), "n_people": len(people),
               "elapsed_s": time.time() - t0, "aim_eval_common": os.path.relpath(C.__file__, K.EXP_ROOT)},
              open(os.path.join(out, "run_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"  → {rp}（{time.time() - t0:.0f} s）")
    return out
