# run_log（2026-10-22 予備解析）

結果の読み方は [`report.md`](report.md)。ここは**何を・どの設定で・どの順に実行したか**の記録。

## 環境

| 項目 | 値 |
|---|---|
| リポジトリ | Sugurithm/TrackingExperiment（`C:\project\experiment`） |
| コミット | `83d82a81d998ccc2d760713f2fa3a5fe59f9556e`（2026-10-22 17:02 "add&modify file"）+ 未コミットの変更（下記） |
| Python | 3.11.9（`.venv`）、numpy 2.2.6、scipy 1.17.1、matplotlib 3.11.1、mediapipe 0.10.35 |
| OS | Windows 11 |
| `experiment/config.py` | `POSE_VARIANT = "full"`、`JOINT_STRIDE = 1`（関節推定はこの解析では行っていない。npz は既存のもの） |

作業開始時の未コミットの変更（この解析より前から存在。こちらでは変更していない）:
`.gitignore`、`aim_eval/config.py`（SESSIONS を 9 人 18 セッションに、BANDS = ["slow", "fast"]）、`analysis/run_log.md`、
`experiment/lib/angle.py` → `angles.py` へのリネーム。

## 0. 前提の確認

- **angle.py → angles.py のリネームは済んでいた**。`git show HEAD:experiment/lib/angle.py` と `experiment/lib/angles.py` は改行コードを除いて同一。
  import しているのは `experiment/compare_joints.py`、`experiment/mocap_joints.py`、`experiment/lib/mocap_io.py`、`aim_eval/a8_dof.py`（ファイルパス指定）で、すべて `angles` を参照している。修正は不要だった。
- **関節推定**: 18 セッション × 7 試行がすべて `joints/rs_joints.npz` を持つことを確認した。Yuuki400 は 22:59 にユーザー側の `run.py --steps joints` で推定済み（`analysis/run_log.md`）。この解析では推定をしていない。
- **DPI**: 400 の 9 セッションはすべて記録時の `mouse_dpi` が 800（`--dpi` の付け忘れ）。config の `dpi=400` で counts → cm を 2 倍に直している。手先 / マウスのゲインが 0.64〜0.90（中央値 0.80）に収まり、2 倍のずれ（0.4 または 1.6 付近）が無いことで換算は正しいと判断した。
- **run.py の挙動**（コードで確認）:
  - `FORCE = False` では `eval/a8_dof.json` がある試行を飛ばす。今回は再現性のため `FORCE = True` にして、全試行を現行コードで作り直した。
  - `SKIP_DONE` は step1（関節推定）にだけ効き、`rs_joints.npz` がある試行を飛ばす。
  - 各試行の `eval/` は**試行フォルダ内に上書き**される（`--out` の影響を受けない）。
  - `aim_eval_override.json` は各セッションに上書きされる（内容は config のまま）。
  - `static_floor.json` はセッションにキャッシュされたものを読み直す。
  - やり直し試行は attempt の大きいものだけを使う（`common.list_trials`）。

## 1. 設定

`analysis_out/20261022/config_used.py` = `aim_eval/config.py` のコピーで、次の 2 行だけを変えた。

```diff
-OUT = "analysis"            # 出力先（参加者ごとのフォルダと between/ がこの下にできる）
+OUT = "analysis_out/20261022"   # 今回の解析の出力先
-FORCE = False               # すでに eval/ がある試行もやり直す
+FORCE = True                # 既存の eval/ も現行コードで作り直す（再現性のため）
```

その他の主な値: BANDS = ["slow", "fast"]、INCLUDE_STATIC = True、STRIDE = 2、MIN_D = 1.5、N_PERM = 2000、
GATE = {mouse_gain: (0.7, 1.4), mouse_lag_ms: (−50, 120), valid_pct: 70, lr_swapped_pct: 10, snr_ok_pct: 50}。

## 2. 実行したコマンド（この順）

作業ディレクトリは `C:\project\experiment`。Windows では `PYTHONIOENCODING=utf-8` を付ける（付けないと日本語の出力で落ちることがある）。

```bash
# (1) 対象の確認
.venv/Scripts/python.exe aim_eval/run.py --config analysis_out/20261022/config_used.py --plan

# (2) 合成データで指標を検証 → verify.log（20 / 20 OK）
.venv/Scripts/python.exe aim_eval/verify.py > analysis_out/20261022/verify.log

# (3) 解析パイプライン（a0〜a8 → w_within → b_between）。約 11 分
.venv/Scripts/python.exe aim_eval/run.py --config analysis_out/20261022/config_used.py --steps analyze \
    > analysis_out/20261022/analyze_stdout.log
#     run.py は OUT/run_log.md を書くので、この記録と衝突しないよう pipeline_run_log.md に改名した
mv analysis_out/20261022/run_log.md analysis_out/20261022/pipeline_run_log.md

# (4) 手首の MoCap 照合（既存の mc_joints.npz と compare/ は先に退避）
#     退避先: analysis_out/20261022/mocap_check/backup_before_rerun/<trial>/
cd experiment
for t in trial_20261016_195332YOKO trial_20261016_200449TATE trial_20261019_015509 trial_20261019_032750; do
  ../.venv/Scripts/python.exe mocap_joints.py   ../recordings/$t > ../analysis_out/20261022/mocap_check/mocap_joints_$t.log
  ../.venv/Scripts/python.exe compare_joints.py ../recordings/$t > ../analysis_out/20261022/mocap_check/compare_$t.log
done
cd ..
#     10/16 の 2 本は compare が FileNotFoundError（recordings/calib_20261016_1850/T_MC.json が無い）で比較できず

# (5) 推論（新規の別プログラム。既存コードは変更していない）
.venv/Scripts/python.exe aim_eval/prelim_inference.py --config analysis_out/20261022/config_used.py \
    --out analysis_out/20261022 --gate relaxed     # 主解析   → prelim_relaxed/
.venv/Scripts/python.exe aim_eval/prelim_inference.py --config analysis_out/20261022/config_used.py \
    --out analysis_out/20261022 --gate strict      # 感度解析 → prelim_strict/
#     既定: --n-boot 5000 --n-perm 5000 --seed 0（乱数は seed 0 の 1 本の系列。指標の並びを変えると CI・p が小数点以下で動く）
```

`aim_eval/prelim_inference.py`（新規、sha256 先頭 `7bb7026c4fad6da8`）が行うこと:
- **除外**: `--gate strict` は config.GATE のまま。`--gate relaxed` はゲインを (0.5, 1.5)、遅れを (−100, 150) ms に広げる（DPI の指定ミスだけを捉える幅）。
- **再計算**: `common.py` の関数（`Trial`、`joint_grid`、`bin_power`、`freqs`）をそのまま使い、3 通りの分解で N_eff・S・C を作る。
  - v4 = a1 と同じ 4 成分。再計算が a1 と**最大差 0** で一致することを確認した。
  - v3 = 手首の点を使わない 3 成分。
  - v2 = 近位 / 遠位。
  
  これに加えて次を出す。
  - C_corr（ノイズパワーを引いた C）
  - 近位 / 遠位の位相差。両成分とも SNR > 2 のビンだけを使う
  - 周波数の低い半分 / 高い半分、および x / y 方向別のシェア
- **統計**:
  - 人物内（DPI の効果）: d、dz、Wilcoxon、符号の一致、帯域をまたいだ再現
  - 人物間: ICC(1,1)、並べ替え検定、人のブートストラップ、Spearman ρ
  - 練度: Kendall τ と並べ替え検定

## 3. 確認のために実行したもの（出力は report.md に反映）

```bash
# ゲインの分布（参加者 × DPI × 帯域）: 0.64〜0.90、中央値 0.80、二峰性なし
# → 主解析を relaxed にした根拠
# Syou の体幹成分: 肩の変位 RMS は 400 で 23〜35 mm・800 で 11〜19 mm（比べた Suguru は 9〜15 / 5〜9 mm）、周波数 SNR 2.5〜6.3
# 成分ごとの SNR（周波数・時間）を DPI × 帯域で比較: 手の成分の周波数 SNR は 800 でも 4.4（ノイズではない）
# 記録順（400 先: Keito, Kosuke, Suguru / 800 先: 残り 6 人）で DPI の効果を分けても、手のシェア・N_eff・C_corr・fast の誤差の向きは同じ
# a8 の自由度ごとの手先変位（参考値）: 肩・回旋 52〜63 mm、肩・方位 約 52 mm、肘・屈曲 41〜45 mm、手首 5〜7 mm
```

（いずれも `between/trials.csv`・`prelim_relaxed/trials_prelim.csv`・`prelim_relaxed/person_means.csv` を読む短い Python。**ファイルとしては保存していない**。）

## 4. 主解析（relaxed）と感度解析（strict）の比較

| 指標 | 主解析（108 本） | 感度解析（93 本） |
|---|---|---|
| 手のシェアの DPI 効果 slow: d / dz / 人数 | +0.102 / +1.77 / 9 of 9 | +0.101 / +1.52 / 7 of 7 |
| 手首なし N_eff の DPI 効果 slow: d / dz | −0.03 / −0.29 | −0.05 / −0.50 |
| C_corr の DPI 効果 slow: d / dz / 人数 | +0.074 / +1.49 / 9 of 9 | +0.082 / +1.51 / 7 of 7 |
| fast の追従誤差: d / dz | −1.25 / −2.91 | −1.30 / −2.55 |
| 手首なし N_eff の ICC slow / D | 0.78 / 1.92 | 0.75 / 1.80 |
| 遠位シェアの ICC slow / D | 0.81 / 2.10 | 0.81 / 2.03 |
| 手のシェアの ICC fast / D | 0.66 / 1.45 | 0.78 / 1.94 |

感度解析の DPI 効果は、両 DPI に 2 本以上残った人だけで計算している（slow 7 人、fast 6 人）。すべての行で向きは同じ。

## 5. 出力の一覧

| パス | 内容 |
|---|---|
| `report.md` | 結論・根拠・除外・限界・今後の解析 |
| `config_used.py` | 使った設定 |
| `pipeline_run_log.md`、`analyze_stdout.log` | run.py の実行ログ（点検表を含む） |
| `verify.log` | 合成データの検証（20 / 20 OK） |
| `<参加者>/summary.md`、`w_within.png`、`trials.csv`、`desk_frame.json` | プレイヤー内（pipeline） |
| `between/summary.md`、`b_between.png`、`players.csv`、`trials.csv` | プレイヤー間（pipeline） |
| `prelim_relaxed/`（主解析）、`prelim_strict/`（感度解析） | `prelim_log.md`（全表）、`trials_prelim.csv`、`person_means.csv`、`dpi_effect.csv`、`between_icc.csv`、`skill_corr.csv`、`excluded_trials.csv`、`prelim.json`、`fig1_shares.png`、`fig2_dpi_pairs.png`、`fig3_robustness.png`、`fig4_skill.png`、`fig5_direction.png` |
| `mocap_check/` | MoCap 照合のログと、上書き前の `mc_joints.npz`・`compare/` の退避 |

## 6. 解析フォルダの外に書いたもの（ご注意）

- `recordings/aim_*/trial_*/eval/`: 108 + 18 試行ぶんを現行コードで上書き（FORCE = True）。
- `recordings/aim_*/aim_eval_override.json`: run.py が config の内容で再書き込み（内容は同じ）。
- `recordings/trial_20261016_*/joints/mc_joints.npz`、`recordings/trial_20261019_*/joints/mc_joints.npz`: mocap_joints.py で作り直した（手の点 RHandOut と関節角が加わった）。
- `recordings/trial_20261019_*/compare/`: compare_joints.py で新規作成（手首角の比較を含む）。
- 元の `mc_joints.npz` と `compare/` は `mocap_check/backup_before_rerun/` にある。

## 7. 再現するには

上の §2 の (1)〜(5) を順に実行する。乱数は `--seed 0` で固定。**(3) の run.py は `OUT/run_log.md` を上書きする**ので、この記録を残したいときは先に退避するか、(3) のあとに改名すること。

## 8. 追補（2026-10-23）: エイムの質と協調の関係

結果は [`report_perf.md`](report_perf.md)。

```bash
# prelim_inference.py（§2 の (5)）のあとに実行。20 s ほど
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe aim_eval/perf_coordination.py     --config analysis_out/20261022/config_used.py --out analysis_out/20261022     > analysis_out/20261022/perf_stdout.log
#   既定: --prelim prelim_relaxed --n-boot 5000 --n-perm 5000 --seed 0
```

- `aim_eval/perf_coordination.py`（新規、sha256 先頭 `91978de302d1e6b4`）。既存コードは変更していない。
- 入力:
  - `between/trials.csv`（a5 の遅れ・remnant、a6 の RMS・on-target）
  - `prelim_relaxed/trials_prelim.csv`（協調の指標）
  - 各試行の `task.csv`（250 Hz のカーソル・ターゲット）
- 出力は `perf_coordination/`。
- 途中で決めたこと:
  - **SPARC を解析から外した**: 遅れて小さく追う試行ほど「滑らか」に出たため。代わりに「1.5 Hz 超の割合」を使う。
  - **追従帯域 fb を外した**: 全条件で約 20 Hz に張り付く外挿値だったため。
  - 置換検定とブートストラップを行列計算にした（結果の定義は同じ、乱数の消費の仕方が変わる）。
- 多重比較: 帯域ごとに 110 検定（協調 10 × 質 11）の BH-FDR。練度の τ は帯域ごとに 11 検定。

### 8.1 DPI の層に分けて再実行（2026-10-23）

`perf_coordination.py` を改め、B1・B2・練度の τ を **400 だけ / 800 だけ / 400+800** の 3 層で別々に計算するようにした（出力の `stratum` 列）。
- BH-FDR は (層 × 帯域) ごと。
- 400 と 800 の関係の差の検定も足した（`dpi_heterogeneity.csv`）。
- 実行コマンドは §8 と同じ（約 1 分）。
- 改める前の版は `C:\Users\IMURA-~1\AppData\Local\Temp\claude\…\scratchpad\perf_coordination_before_strata.py` に退避したが、作業用の一時領域なので残らない。差分は「§2 の層のループの追加」と「図を層ごとに並べる」だけ。
- 現在の `aim_eval/perf_coordination.py` の sha256 先頭: `188c049945423b7e`

## 9. 追補 2（2026-10-23）: 入力周波数ごとの分解

結果は [`report_freq.md`](report_freq.md)。

```bash
# prelim_inference.py（§2 の (5)）のあとに実行。約 40 s
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe aim_eval/freq_decomp.py     --config analysis_out/20261022/config_used.py --out analysis_out/20261022     > analysis_out/20261022/freq_stdout.log
#   既定: --prelim prelim_relaxed --n-boot 5000 --seed 0
```

- `aim_eval/freq_decomp.py`（新規、sha256 先頭 `f8327fc7ed08cc38`）。既存コードは変更していない。
- 使う関数: `common.py` の `Trial`、`task_grid`、`contributions`、`spectrum`、`at_bins`、`bin_noise`、`list_trials`。
- 静止試行は参加者の `desk_frame.json` を使って読み直し、課題外の帯域のノイズ床に使う。
- 途中で決めたこと: 課題外の帯域の下端を 1.2 Hz 固定から「最高の入力周波数 + 0.4 Hz」（fast は 1.5 Hz）に変えた。fast で入力帯域の端（1.25 Hz）がピークとして拾われ、Welch の漏れと判断したため。
- 周波数ごとの協調 ↔ 追従の p は周波数を独立とみなした近似。試行ごとに周波数を平均した確認は、短い Python で計算した（`report_freq.md` §8 の表、ファイルとしては保存していない）。

## 10. 追補 3（2026-10-23）: C-1 外れた瞬間 / C-4 2〜4 Hz の修正

結果は [`report_corrections.md`](report_corrections.md)。

```bash
# freq_decomp.py（§9）のあとに実行（freq_decomp/per_bin.csv を使う）。約 40 s
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe aim_eval/corrections.py     --config analysis_out/20261022/config_used.py --out analysis_out/20261022     > analysis_out/20261022/corrections_stdout.log
#   既定: --prelim prelim_relaxed --n-boot 5000 --seed 0
```

- `aim_eval/corrections.py`（新規、sha256 先頭 `1ead4cc8352bc62e`）。既存コードは変更していない。
- 入力:
  - `between/trials.csv`（カメラ − マウスの遅れ mouse_lag_ms）
  - `freq_decomp/per_bin.csv`（試行ごとの課題の帯域での分担とカメラ / マウスのゲイン）
  - `prelim_relaxed/trials_prelim.csv`（除外）
  - 各試行の task.csv・rs_joints.npz
- すべての集計を DPI の層（400 / 800 / まとめ）に分けて出力した（`stratum` 列）。
- 画面の軸とマウス（机）の軸の向きが一致していることを、事前に 2 試行で確かめた（速度の相関 +1.00）。
