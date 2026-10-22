"""config.py に並べたセッションを、関節位置の推定から比較まで実行する。

  python aim_eval/run.py --plan              # 何をするかだけ表示（実行しない）
  python aim_eval/run.py --steps joints      # 1. 関節位置の推定（時間がかかる。確認動画も作る）
  python aim_eval/run.py --steps video       #    確認動画だけ作り直す（推定済みの試行から）
  python aim_eval/run.py --steps analyze     # 2. 解析（a0〜a7 → プレイヤー内 → プレイヤー間）
  python aim_eval/run.py                     # 1 と 2 を続けて
  python aim_eval/run.py --config myconf.py

段階
  plan     … フォルダ・試行数・DPI・.db3 の有無を確認する
  joints   … 未処理の試行に experiment/rs_joints.py を回し、確認動画（既定 20 fps）と一覧を出す
  video    … 推定済みの試行に確認動画だけ作る（推定はしない）
  trials   … 試行ごとに a0〜a7（1 本 3 秒ほど）
  within   … 参加者ごとにプレイヤー内の解析（センシの効果）
  between  … 参加者 2 人以上ならプレイヤー間の解析（ランダムか偏りか）
"analyze" は trials,within,between のまとめ書き。途中で止めても、再実行すれば続きから進む。
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common as C                                      # noqa: E402
import a0_quality, a1_indices, a2_contribution, a3_coupling, a4_joint_freq, a5_describing, a6_task, a7_angles, a8_dof  # noqa: E402
import step1_joints, w_within, b_between                # noqa: E402

MODULES = [a0_quality, a1_indices, a2_contribution, a3_coupling,
           a4_joint_freq, a5_describing, a6_task, a7_angles, a8_dof]
STEPS = ["joints", "video", "trials", "within", "between"]


class Args:
    def __init__(self, trial, **kw):
        self.trial = trial
        self.rs = self.tmc = self.axes = self.out = None
        for k, v in kw.items():
            setattr(self, k, v)


def load_config(path):
    spec = importlib.util.spec_from_file_location("aim_config", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def resolve(d):
    return os.path.normpath(d if os.path.isabs(d) else os.path.join(ROOT, d))


def log(f, s=""):
    print(s)
    f.write(s + "\n"); f.flush()


def pick_axes(trials, out_dir):
    """手の動きが一番大きい試行で机の座標系を決め、desk_frame.json に保存して使い回す。"""
    best, best_rng = None, -1
    for d in trials:
        if not os.path.exists(os.path.join(d, "joints", "rs_joints.npz")):
            continue
        try:
            tr = C.Trial(Args(d, out=os.path.join(d, "eval")))
        except Exception:
            continue
        if not tr.has_rs or tr.kind != "track":
            continue
        rng = float(np.ptp(tr.mouse[:, 0]))
        if rng > best_rng and np.isfinite(tr.frame.get("gain_x", np.nan)):
            best, best_rng = tr.frame, rng
    if best is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "desk_frame.json")
    json.dump(best, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=C._jsonable)
    return path


# ------------------------------------------------------------------ plan
def plan(cfg, lg):
    rows = []
    for e in cfg.SESSIONS:
        d = resolve(e["dir"])
        ok = os.path.exists(os.path.join(d, "session.json"))
        trials = C.list_trials(d) if ok else []
        if not getattr(cfg, "INCLUDE_STATIC", True):
            trials = [t for t in trials if t[1].get("task", {}).get("kind") == "track"]
        bands = getattr(cfg, "BANDS", None)
        if bands:                      # 静止は帯域を持たないので残す
            trials = [t for t in trials
                      if t[1].get("task", {}).get("kind") != "track"
                      or t[1].get("task", {}).get("band") in bands]
        sess = json.load(open(os.path.join(d, "session.json"), encoding="utf-8")) if ok else {}
        rows.append({"dir": d, "entry": e, "ok": ok, "trials": trials,
                     "db3": sum(1 for t, _ in trials if os.path.exists(os.path.join(t, "session.db3"))),
                     "done": sum(1 for t, _ in trials
                                 if os.path.exists(os.path.join(t, "joints", "rs_joints.npz"))),
                     "participant": e.get("participant") or sess.get("participant", "?"),
                     "dpi": e.get("dpi") or sess.get("config", {}).get("mouse_dpi"),
                     "dpi_rec": sess.get("config", {}).get("mouse_dpi")})
    lg("\n## plan 対象の確認\n")
    bands = getattr(cfg, "BANDS", None)
    if bands:
        lg(f"帯域を {', '.join(bands)} に絞っています（他の帯域は推定も解析もしません）。\n")
    if not getattr(cfg, "INCLUDE_STATIC", True):
        lg("> **静止試行を外しています**。ノイズ床が測れないので R（時間の交代）の信頼性を"
           "判定できません。静止は 10 秒なので、含めることを勧めます。\n")
    tab = []
    for r in rows:
        state = ("session.json なし" if not r["ok"] else
                 ("OK（推定済み）" if r["done"] == len(r["trials"]) and r["trials"] else
                  (f"推定待ち {len(r['trials']) - r['done']} 本" if r["db3"] else ".db3 なし")))
        kinds = [t[1].get("task", {}).get("kind") for t in r["trials"]]
        tab.append({"s": os.path.basename(r["dir"]), "p": r["participant"],
                    "dpi": f"{r['dpi']:g}" if r["dpi"] else "–",
                    "rec": f"{r['dpi_rec']:g}" if r["dpi_rec"] else "–",
                    "n": len(r["trials"]),
                    "what": "，".join(sorted({(t[1].get("task", {}).get("band") or "静止")
                                              for t in r["trials"]})),
                    "db3": r["db3"], "done": r["done"], "state": state})
    lg(C.md_table(tab, [("セッション", "s"), ("参加者", "p"), ("DPI（指定）", "dpi"),
                        ("DPI（記録時）", "rec"), ("試行", "n", "{:.0f}"), ("内訳", "what"),
                        (".db3", "db3", "{:.0f}"),
                        ("推定済み", "done", "{:.0f}"), ("状態", "state")]))
    for r in rows:
        if r["ok"] and r["dpi"] and r["dpi_rec"] and abs(r["dpi"] - r["dpi_rec"]) > 1e-6:
            lg(f"\n> **{os.path.basename(r['dir'])}**: 記録時 {r['dpi_rec']:g} に対し {r['dpi']:g} と指定。"
               f"counts → cm の換算を {r['dpi_rec'] / r['dpi']:.3f} 倍に直します"
               "（a0 の「手先 / マウス のゲイン」が 1 付近になるか確認してください）。")
    # 参加者名・DPI の上書きをセッションに書く
    for r in rows:
        if not r["ok"]:
            continue
        ov = {k: v for k, v in (("participant", r["entry"].get("participant")),
                                ("dpi", r["entry"].get("dpi")),
                                ("note", r["entry"].get("note"))) if v is not None}
        if ov:
            json.dump(ov, open(os.path.join(r["dir"], C.OVERRIDE_NAME), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
    return rows


# ------------------------------------------------------------------ trials
def step_trials(cfg, rows, out, lg):
    lg("\n## step2 試行ごとの解析（a0〜a7）\n")
    by_person = {}
    for r in rows:
        by_person.setdefault(str(r["participant"]), []).extend([t for t, _ in r["trials"]])
    axes_of = {n: pick_axes(ts, os.path.join(out, n)) for n, ts in by_person.items()}
    gate, done = [], []
    for name, trials in sorted(by_person.items()):
        for t in trials:
            if not os.path.exists(os.path.join(t, "joints", "rs_joints.npz")):
                continue
            fresh = os.path.exists(os.path.join(t, "eval", "a8_dof.json")) and not cfg.FORCE
            if not fresh:
                t0 = time.time()
                try:
                    tr = C.Trial(Args(t, axes=axes_of.get(name), out=os.path.join(t, "eval")))
                except Exception as e:
                    lg(f"- [SKIP] {os.path.basename(t)}: {e}"); continue
                parts = []
                for m in MODULES:
                    try:
                        parts.append(m.run(tr))
                    except Exception as e:
                        parts.append(f"## {m.__name__}\n\n（エラー: {e}）\n")
                        lg(f"- [ERR] {os.path.basename(t)} {m.__name__}: {e}")
                head = (f"# {tr.name}（{tr.tag()}，DPI {tr.dpi:g}，参加者 {tr.participant}）\n\n"
                        f"- 解析区間 0–{tr.T:.0f} s，RS {tr.fs:.1f} fps\n"
                        f"- 机の座標系: {tr.frame_source}\n\n")
                open(os.path.join(tr.out, "summary.md"), "w", encoding="utf-8").write(
                    head + "\n\n".join(parts))
                lg(f"- {os.path.basename(t):30s} {time.time() - t0:5.1f} s")
            done.append(t)
            gate.append(check_gate(cfg, t))
    lg("\n### 点検（a0 の結果）\n")
    gate = [g for g in gate if g]
    lg(C.md_table(gate, [("試行", "trial"), ("参加者", "participant"), ("DPI", "dpi", "{:g}"),
                         ("内容", "tag"), ("手先/マウス", "gain", "{:.2f}"), ("遅れ [ms]", "lag", "{:+.0f}"),
                         ("手首の有効率 [%]", "valid", "{:.0f}"), ("左右 [%]", "lr", "{:.1f}"),
                         ("SNR≥2 [%]", "snr_ok", "{:.0f}"), ("点検", "note")]))
    bad = [g for g in gate if g["note"] not in ("OK", "–（静止）")]
    lg(f"\n引っかかった試行: {len(bad)} 本"
       + ("（原因の列を確認してください。確認動画の目視もおすすめします）" if bad else ""))
    if bad and cfg.STRICT:
        sys.exit("STRICT: 点検に引っかかったので止めます")
    return done


def check_gate(cfg, t):
    f = os.path.join(t, "eval", "a0_quality.json")
    if not os.path.exists(f):
        return None
    q = json.load(open(f, encoding="utf-8"))
    h = q["header"]; mm = q.get("mouse_match", {})
    j = {x["joint"]: x for x in q["joints"]}
    snr = q.get("snr", [])
    row = {"trial": os.path.basename(t), "participant": h["participant"], "dpi": h["dpi"],
           "tag": f"{h['band']} s{h['seed']}" if h["kind"] == "track" else "static",
           "gain": mm.get("gain_x", np.nan), "lag": mm.get("lag_ms", np.nan),
           "valid": j.get("wrist", {}).get("valid_pct", np.nan), "lr": q.get("lr_swapped_pct", np.nan),
           "snr_ok": 100 * np.mean([s["snr"] > C.SNR_MIN for s in snr]) if snr else np.nan}
    if h["kind"] != "track":
        row["note"] = "–（静止）"; return row
    g = cfg.GATE; n = []
    if np.isfinite(row["gain"]) and not (g["mouse_gain"][0] <= row["gain"] <= g["mouse_gain"][1]):
        n.append("ゲイン")
    if np.isfinite(row["lag"]) and not (g["mouse_lag_ms"][0] <= row["lag"] <= g["mouse_lag_ms"][1]):
        n.append("遅れ")
    if np.isfinite(row["valid"]) and row["valid"] < g["valid_pct"]:
        n.append("有効率")
    if np.isfinite(row["lr"]) and row["lr"] > g["lr_swapped_pct"]:
        n.append("左右")
    if np.isfinite(row["snr_ok"]) and row["snr_ok"] < g["snr_ok_pct"]:
        n.append("SNR")
    row["note"] = "，".join(n) if n else "OK"
    return row


# ------------------------------------------------------------------ within / between
def step_within(cfg, rows, out, lg):
    lg("\n## step2 プレイヤー内（センシの効果）\n")
    by_person = {}
    for r in rows:
        by_person.setdefault(str(r["participant"]), []).extend([t for t, _ in r["trials"]])
    allrows = []
    for name, trials in sorted(by_person.items()):
        pout = os.path.join(out, name)
        rws = w_within.collect(trials)
        if not rws:
            lg(f"- {name}: 解析済みの試行がありません"); continue
        md = w_within.run(rws, pout, title=name)
        open(os.path.join(pout, "summary.md"), "w", encoding="utf-8").write(
            f"# {name}\n\n試行 {len(rws)} 本。\n\n" + md)
        lg(f"- {name}: 試行 {len(rws)} 本 → {os.path.relpath(os.path.join(pout, 'summary.md'), ROOT)}")
        allrows += rws
    return allrows


def step_between(cfg, allrows, out, lg):
    lg("\n## step2 プレイヤー間（ランダムか偏りか）\n")
    people = sorted({r["participant"] for r in allrows})
    if len(people) < 2:
        lg(f"参加者が {len(people)} 人なので省略（config に別の人のセッションを足してください）。"); return
    pout = os.path.join(out, "between")
    mdw = w_within.run(allrows, pout, title="全参加者")
    mdb = b_between.run(allrows, pout, min_D=cfg.MIN_D, n_perm=cfg.N_PERM)
    head = (f"# プレイヤー間の比較\n\n参加者 {len(people)} 人（{', '.join(map(str, people))}）、"
            f"試行 {len(allrows)} 本。\n\n"
            "読む順: ① 指標はランダムか偏りか → ② プロファイルと (N_eff, S) 平面 → "
            "③ センシへの反応の個人差 → ④ 成績との相関。\n\n")
    open(os.path.join(pout, "summary.md"), "w", encoding="utf-8").write(
        head + mdb + "\n\n---\n\n" + mdw)
    lg(f"→ {os.path.relpath(os.path.join(pout, 'summary.md'), ROOT)}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.py"))
    ap.add_argument("--plan", action="store_true", help="確認だけして実行しない")
    ap.add_argument("--steps", default="joints,analyze", help=f"{'/'.join(STEPS)} / analyze")
    ap.add_argument("--out", help="出力先（既定は config の OUT）")
    ap.add_argument("--jobs", type=int, help="推定の並列数（config の JOBS を上書き）")
    ap.add_argument("--stride", type=int, help="何フレームに 1 枚処理するか（config の STRIDE を上書き）")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = a.out or (cfg.OUT if os.path.isabs(cfg.OUT) else os.path.join(ROOT, cfg.OUT))
    os.makedirs(out, exist_ok=True)
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    if a.jobs:
        cfg.JOBS = a.jobs
    if a.stride:
        cfg.STRIDE = a.stride
    if "analyze" in steps:
        steps = [s for s in steps if s != "analyze"] + ["trials", "within", "between"]

    path = os.path.join(out, "run_log.md")
    with open(path, "w", encoding="utf-8") as f:
        lg = lambda s="": log(f, s)
        lg(f"# 実行ログ（{time.strftime('%Y-%m-%d %H:%M:%S')}）\n")
        lg(f"設定: {a.config}　出力: {out}　段階: {', '.join(steps)}")
        rows = plan(cfg, lg)
        if a.plan:
            lg("\n（--plan なのでここまで）"); print("→", path); return
        if "joints" in steps or "video" in steps:
            sub = (["estimate"] if "joints" in steps else []) + \
                  (["video"] if ("video" in steps or "joints" in steps) else []) + ["list"]
            step1_joints.run([r for r in rows if r["ok"]], lg, fps=cfg.VIDEO_FPS,
                             skip_done=cfg.SKIP_DONE, include_static=cfg.INCLUDE_STATIC,
                             video_mode=cfg.VIDEO, steps=sub,
                             force_video=getattr(cfg, "FORCE_VIDEO", False),
                             workers=getattr(cfg, "JOBS", 0),
                             threads_per_worker=getattr(cfg, "THREADS_PER_JOB", None),
                             video_workers=getattr(cfg, "VIDEO_JOBS", 0),
                             stride=getattr(cfg, "STRIDE", None))
        if "trials" in steps:
            step_trials(cfg, rows, out, lg)
        allrows = step_within(cfg, rows, out, lg) if "within" in steps else []
        if "between" in steps:
            allrows = allrows or [r for rr in rows for r in w_within.collect([t for t, _ in rr["trials"]])]
            step_between(cfg, allrows, out, lg)
        lg("\n完了。")
    print("→", path)


if __name__ == "__main__":
    main()