"""関節推定を速くできるか、自分の PC で測る

1 試行を題材に「1 本だけ」と「N 本を並列」でスループットを比べ、JOBS をいくつにすべきかを決める。
MediaPipe の CPU 推論は 1 本の中でも複数スレッドを使うので、**並列が効くかは PC による**。
効かない PC で JOBS を上げてもメモリを食うだけなので、先にこれで測る。

  python aim_eval/bench_joints.py recordings/aim_X_.../trial_01_slow_s1
  python aim_eval/bench_joints.py <trial> --jobs 2 4      # 試す並列数
  python aim_eval/bench_joints.py <trial> --seconds 20    # 1 回あたりの計測を短くする（既定 30 s）

実際に rs_joints.py を走らせて途中で止めるので、**結果の .npz は書き換わらない**
（rs_joints.py は最後まで走ってから保存するため）。
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def frames_done(out):
    """rs_joints.py の進捗表示から、処理できたフレーム数を拾う。"""
    best = 0
    for line in out.splitlines():
        for tok in line.replace("/", " ").replace("(", " ").split():
            if tok.isdigit():
                best = max(best, int(tok))
    return best


def run_n(trial, n, seconds, python=None, env_extra=None):
    """n 本を同時に起動し、seconds 秒だけ走らせて、合計の処理フレーム数を返す。"""
    script = os.path.join(ROOT, "experiment", "rs_joints.py")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", **(env_extra or {})}
    ps = [subprocess.Popen([python or sys.executable, script, trial], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           text=True, errors="replace", env=env) for _ in range(n)]
    t0 = time.time()
    while time.time() - t0 < seconds and any(p.poll() is None for p in ps):
        time.sleep(0.5)
    el = time.time() - t0
    total = 0
    for p in ps:
        p.terminate()
        try:
            out, _ = p.communicate(timeout=20)
        except Exception:
            p.kill(); out = ""
        total += frames_done(out or "")
    return total, el


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial", help="題材にする試行フォルダ（.db3 がある本番の試行）")
    ap.add_argument("--jobs", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--seconds", type=float, default=30.0, help="1 回あたりの計測時間 [s]")
    ap.add_argument("--threads", type=int, help="1 本あたりのスレッド数を絞って試す")
    a = ap.parse_args()
    trial = a.trial if os.path.isabs(a.trial) else os.path.join(ROOT, a.trial)
    if not os.path.exists(os.path.join(trial, "session.db3")):
        sys.exit(f"{trial} に session.db3 がありません")
    env = {}
    if a.threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "XNNPACK_NUM_THREADS"):
            env[k] = str(a.threads)

    import multiprocessing
    print(f"CPU {multiprocessing.cpu_count()} コア　題材 {os.path.basename(trial)}　"
          f"1 回 {a.seconds:.0f} s" + (f"　スレッド上限 {a.threads}" if a.threads else ""))
    print(f"\n{'並列':>4} {'合計フレーム':>12} {'スループット [fps]':>18} {'1 本あたり [fps]':>16} {'1 並列との比':>12}")
    base = None
    for n in a.jobs:
        total, el = run_n(trial, n, a.seconds, env_extra=env)
        thr = total / el
        base = base or thr
        print(f"{n:>4} {total:>12} {thr:>18.1f} {thr / n:>16.1f} {thr / base:>12.2f}x")
    print("\n「1 並列との比」が 1.5 を超える並列数があれば、config.py の JOBS をその値にする。")
    print("どの並列数でも 1.2 倍に届かないなら JOBS = 1 のままにして、")
    print("別の手（Pose モデルを full にする / フレームを間引く）を検討する。")


if __name__ == "__main__":
    main()
