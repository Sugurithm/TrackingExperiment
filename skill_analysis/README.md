# skill_analysis — 練度別のターゲット追従・上肢協調の解析

練度グループ（A 高: Syou, Yuuya, Suguru / B 中: Kakeru, Xin / C 低: Kosuke, Yuuki, Suzuka）で、
① ターゲット追従特性（精度・滑らかさ・軌道修正）と ② 上肢関節（体幹・肩・肘・手首）の寄与と時間的な分担を比べる。
**条件（DPI × 帯域）ごとに独立に解析**し、条件をまたいだ統合はしない。

## 使い方

```
python skill_analysis/run_skill.py                     # 400 DPI / SLOW（既定）
python skill_analysis/run_skill.py --cond dpi800_fast  # 他の条件（dpi400_slow, dpi400_fast, dpi800_slow, dpi800_fast）
python skill_analysis/run_skill.py --all               # 4 条件それぞれを独立に
python skill_analysis/compare_conditions.py            # 解析済み条件の「条件 × 指標」比較表
```

`.venv` の Python（numpy・scipy・matplotlib）で動く。1 条件 40 秒ほど（ほとんどが図の書き出し）。
入力は `recordings/aim_*`（task.csv・meta.json・joints/rs_joints.npz・session.json・aim_eval_override.json）を
**読むだけ**で、記録フォルダ・`analysis/`・`analysis_out/`・`aim_eval/` には書き込まない。
関節の読み込み・机座標系・寄与の分解は既存の `aim_eval/common.py` をそのまま使う（書き換えていない）。

## 構成

| ファイル | 役割 |
|---|---|
| `run_skill.py` | 入口。条件を選んで `pipeline.run_condition` を呼ぶ |
| `compare_conditions.py` | 各条件の `group_tests.csv` を並べ、「特定の条件でだけ / どの条件でも 練度差あり」を一覧にする |
| `skill_eval/config.py` | **設定はここだけ**: グループ、外れ値候補（Syou）、名前の対応（Yuya → Yuuya など）、条件、閾値 |
| `skill_eval/io.py` | セッションの探索（`aim_eval_override.json` の DPI・参加者名）と試行の読み込み |
| `skill_eval/tracking.py` | ① 追従特性（250 Hz の課題ログ） |
| `skill_eval/coordination.py` | ② 関節の寄与・分担・先行・支え合い（3 次元関節位置） |
| `skill_eval/initiation.py` | ② 動き出しの順序（手先が向きを変える瞬間に、どの関節が先に向きを変えるか） |
| `skill_eval/stats.py` | 指標の一覧（`METRICS`）、参加者平均、練度との順位相関（全並べ替えの正確な p）、BH 補正、Syou の除外有無 |
| `skill_eval/figures.py` | 図 |
| `skill_eval/report.py` | 条件ごとの `report.md` |

指標を足すときは `tracking.py` / `coordination.py` で値を返し、`stats.METRICS` に 1 行足すだけで、
表・検定・図（指標グリッド・Syou の感度）・条件比較に自動で載る。

## 出力（results/）

```
results/
  dpi400_slow/
    report.md            条件のレポート（数値要約・群比較・Syou の影響・図）。再実行で作り直される
    discussion.md        手書きの考察（再実行で上書きされない）
    data/*.csv           trials / persons / group_tests / 周波数ごとの追従・分担 / 修正イベント / 使った試行
    figures/*.png        図
    trials/*.json        試行ごとの全指標
    run_info.json
  _compare/              compare_conditions.py の出力（条件 × 指標）
```

CSV は Excel で開けるよう UTF-8（BOM 付き）。

## 注意

- 1 群 2〜3 人なので、p は探索的な目安。q（BH）・効果量 g・判別比 D（参加者間 SD / seed 間 SD）を併せて読む。
- 手首（hand）・MCP3 を使う成分は MoCap で未照合のため【参考】（`aim_eval/README.md` の精度の節）。
- Keito はグループに含まれないので対象外。名前の対応は `config.ALIASES`。
