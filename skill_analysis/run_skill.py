"""練度別のターゲット追従・上肢協調の解析（条件ごとに独立して実行）

  python skill_analysis/run_skill.py                       # 既定: dpi400_slow
  python skill_analysis/run_skill.py --cond dpi800_fast    # 条件を指定（skill_eval/config.py の CONDITIONS）
  python skill_analysis/run_skill.py --all                 # 4 条件すべて（それぞれ results/<条件>/ に独立に出力）
  python skill_analysis/run_skill.py --out D:/tmp/res      # 出力先を変える（既定: skill_analysis/results）

入力は recordings/aim_*（task.csv・meta.json・joints/rs_joints.npz・session.json・aim_eval_override.json）を読むだけで、
記録フォルダにも既存の analysis/・analysis_out/ にも書き込まない。
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skill_eval import config as K          # noqa: E402
from skill_eval.pipeline import run_condition  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cond", default="dpi400_slow", help="条件名（" + ", ".join(c["name"] for c in K.CONDITIONS) + "）")
    ap.add_argument("--all", action="store_true", help="4 条件すべてを順に（条件ごとに独立に）解析する")
    ap.add_argument("--out", default=K.RESULTS, help="出力のルート")
    a = ap.parse_args()
    conds = K.CONDITIONS if a.all else [c for c in K.CONDITIONS if c["name"] == a.cond]
    if not conds:
        sys.exit(f"条件 {a.cond} は config.CONDITIONS にありません")
    for c in conds:
        run_condition(c, out_root=a.out)


if __name__ == "__main__":
    main()
