"""推定の設定（Pose モデル・フレームの間引き）を変えたときの差を 1 試行で測る

heavy → full、60 → 30 fps は速いが、**計測系そのものを変える**ので、
そのまま本番に使う前に「何がどれだけ変わるか」を必ず見る。
基準（heavy・stride 1）と比べて、関節位置の差・協調の指標の差・処理時間を表にする。

  python aim_eval/compare_settings.py recordings/aim_X_.../trial_01_slow_s1
  python aim_eval/compare_settings.py <trial> --variants heavy:1 heavy:2 full:1 full:2
  python aim_eval/compare_settings.py <trial> --keep        # 各設定の npz を残す

何を見るか
  - 位置の差の中央値・95 %: **静止ジッタ（1.9〜3.8 mm）より小さければ実害はない**
  - 有効率の差: 検出が落ちていないか（full は横向きや速い動きで弱くなることがある）
  - N_eff・S・C の差: 解析の結論が変わらないか。**ここが変わらないなら速い設定でよい**
  - 1 フレームあたりの処理時間
"""
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common as C                                      # noqa: E402


class Args:
    def __init__(self, t):
        self.trial = t; self.rs = self.tmc = self.axes = None
        self.out = os.path.join(t, "eval")


def run_variant(trial, variant, stride, python=None):
    """config.POSE_VARIANT を一時的に差し替えて rs_joints.py を 1 回走らせる。"""
    cfg = os.path.join(ROOT, "experiment", "config.py")
    bak = cfg + ".bench_bak"
    src = open(cfg, encoding="utf-8").read()
    if 'POSE_VARIANT' not in src:
        raise SystemExit("experiment/config.py に POSE_VARIANT がありません（パッチを当ててください）")
    shutil.copy2(cfg, bak)
    try:
        import re
        new = re.sub(r'POSE_VARIANT\s*=\s*"[^"]*"', f'POSE_VARIANT = "{variant}"', src, count=1)
        open(cfg, "w", encoding="utf-8").write(new)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        t0 = time.time()
        p = subprocess.run([python or sys.executable, os.path.join(ROOT, "experiment", "rs_joints.py"),
                            trial, "--stride", str(stride)], cwd=ROOT, capture_output=True,
                           text=True, errors="replace", env=env)
        dt = time.time() - t0
    finally:
        shutil.move(bak, cfg)
    npz = os.path.join(trial, "joints", "rs_joints.npz")
    if not os.path.exists(npz):
        print((p.stderr or "")[-500:])
        raise SystemExit(f"{variant}:{stride} の推定に失敗しました")
    return npz, dt, p.stdout or ""


def load(npz):
    r = np.load(npz, allow_pickle=False)
    return {"t": r["t"].astype(float), "names": [str(n) for n in r["names"]],
            "P": r["P_C"].astype(float),
            "stride": int(r["stride"]) if "stride" in r.files else 1,
            "model": str(r["pose_model"]) if "pose_model" in r.files else "?"}


def compare(base, other):
    """other を base の時刻に合わせて、関節ごとの位置の差を出す。"""
    rows = []
    for j, n in enumerate(base["names"]):
        A = base["P"][:, j]
        B = np.stack([np.interp(base["t"], other["t"], other["P"][:, j, k]) for k in range(3)], 1)
        ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
        d = np.linalg.norm(A[ok] - B[ok], axis=1) if ok.sum() else np.array([np.nan])
        rows.append({"joint": n, "label": C.LABEL.get(n, n),
                     "median_mm": float(np.median(d)), "p95_mm": float(np.percentile(d, 95)),
                     "valid_base": 100 * float(np.isfinite(A).all(1).mean()),
                     "valid_other": 100 * float(np.isfinite(other["P"][:, j]).all(1).mean()),
                     "n": int(ok.sum())})
    return rows


def indices_of(trial):
    try:
        tr = C.Trial(Args(trial))
        r = C.joint_indices(tr)
        return {} if r is None else {"N_eff": r["N_eff"], "S": r["S"], "C": r["C"],
                                     "dominant": r["dominant"]}
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial")
    ap.add_argument("--variants", nargs="+", default=["heavy:1", "heavy:2", "full:1", "full:2"],
                    help="<Pose モデル>:<stride> の並び。最初のものを基準にする")
    ap.add_argument("--keep", action="store_true", help="各設定の npz を joints/ に残す")
    a = ap.parse_args()
    trial = a.trial if os.path.isabs(a.trial) else os.path.join(ROOT, a.trial)
    npz = os.path.join(trial, "joints", "rs_joints.npz")
    saved = npz + ".before_compare" if os.path.exists(npz) else None
    if saved:
        shutil.copy2(npz, saved)

    results = []
    try:
        for spec in a.variants:
            variant, _, st = spec.partition(":")
            st = int(st or 1)
            print(f"\n=== {variant} / stride {st} ===")
            _, dt, out = run_variant(trial, variant, st)
            data = load(npz)
            idx = indices_of(trial)
            keep = npz.replace(".npz", f".{variant}_s{st}.npz")
            shutil.copy2(npz, keep)
            results.append({"spec": spec, "variant": variant, "stride": st, "sec": dt,
                            "frames": len(data["t"]), "data": data, "idx": idx, "file": keep})
            print(f"  {len(data['t'])} frames  {dt / 60:.1f} 分  "
                  f"{1000 * dt / max(len(data['t']), 1):.0f} ms/frame  "
                  + "  ".join(f"{k} {v:.2f}" for k, v in idx.items() if isinstance(v, float)))
    finally:
        if saved:
            shutil.move(saved, npz)
        elif os.path.exists(npz):
            os.remove(npz)

    base = results[0]
    print("\n" + "=" * 72)
    print(f"基準: {base['spec']}（{base['frames']} frames，{base['sec'] / 60:.1f} 分）\n")
    rows = []
    for r in results:
        row = {"spec": r["spec"], "frames": r["frames"], "min": r["sec"] / 60,
               "ms": 1000 * r["sec"] / max(r["frames"], 1),
               "speed": base["sec"] / r["sec"] if r["sec"] else np.nan}
        row.update({k: v for k, v in r["idx"].items() if isinstance(v, (int, float))})
        if r is not base:
            d = compare(base["data"], r["data"])
            row["dmed"] = float(np.nanmedian([x["median_mm"] for x in d]))
            row["dmax"] = float(np.nanmax([x["p95_mm"] for x in d]))
            row["dvalid"] = float(np.nanmin([x["valid_other"] for x in d]))
            r["joints"] = d
        else:
            row["dmed"] = row["dmax"] = 0.0
            row["dvalid"] = float(np.nanmin([100 * np.isfinite(base["data"]["P"][:, j]).all(1).mean()
                                             for j in range(len(base["data"]["names"]))]))
        rows.append(row)
    print(C.md_table(rows, [("設定", "spec"), ("フレーム", "frames", "{:.0f}"), ("時間 [分]", "min", "{:.1f}"),
                            ("ms/frame", "ms", "{:.0f}"), ("速さの比", "speed", "{:.2f}x"),
                            ("位置の差 中央値 [mm]", "dmed", "{:.1f}"), ("同 95% 最大", "dmax", "{:.1f}"),
                            ("最低の有効率 [%]", "dvalid", "{:.0f}"),
                            ("N_eff", "N_eff", "{:.2f}"), ("S", "S", "{:.2f}"), ("C", "C", "{:.2f}")]))
    print("判断のめやす:")
    print("  位置の差が静止ジッタ（手首 1.9 / 肘 3.1 / 肩 3.8 mm）より小さく、")
    print("  N_eff・S・C が基準とほぼ同じ（S は 0.03、N_eff は 0.2 以内）なら、速い設定に替えてよい。")
    print("  有効率が基準より 5 ポイント以上落ちる設定は使わない。")
    for r in results[1:]:
        print(f"\n  {r['spec']} の関節ごとの差:")
        for x in r["joints"]:
            print(f"    {x['label']:14s} 中央値 {x['median_mm']:5.1f} mm  95% {x['p95_mm']:6.1f} mm  "
                  f"有効率 {x['valid_base']:.0f}% → {x['valid_other']:.0f}%")
    if not a.keep:
        for r in results:
            if os.path.exists(r["file"]):
                os.remove(r["file"])
    else:
        print("\n各設定の npz を残しました（--keep）")
    json.dump([{k: v for k, v in r.items() if k not in ("data",)} for r in results],
              open(os.path.join(trial, "joints", "compare_settings.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=C._jsonable)


if __name__ == "__main__":
    main()
