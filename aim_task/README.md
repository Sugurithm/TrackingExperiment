# aim_task — 2D エイムトラッキング課題（+ RealSense の記録）

正弦波の和（sum-of-sines）で動くターゲットをマウスで 40 秒追い続ける課題．
課題だけで動かす `tracking_task.py` と，RealSense の記録を組み込んだ `run_experiment.py` がある．
カメラの設定と記録の処理は `experiment/config.py` と `experiment/lib/rs_device.py` をそのまま使う（較正と同じ設定になる）．

## 準備

```
pip install -r aim_task/requirements.txt      # pygame, matplotlib（RealSense まわりはリポジトリの requirements.txt）
```

- Windows の「ポインターの精度を高める」をオフにする．Logicool G HUB などで DPI を固定する．
- マウスの DPI を実測して `aim_task/task_config.py` の `mouse_dpi` に入れる:
  `python aim_task/tracking_task.py --measure-dpi`（定規に沿って 20 cm．ゆっくりと速くの 2 回で同じ値になれば加速は切れている）
- この日の較正（机にボードを平置き）を `experiment/rec_board.py` で撮っておき，フォルダ名を `--calib` に渡す．

## 実行（リポジトリの直下から）

```
python aim_task/run_experiment.py P01 --calib calib_20261020_1400       # 本番（課題 + RealSense）
python aim_task/run_experiment.py P01 --only fast:1 --no-practice       # 1 本だけ（確認用）
python aim_task/tracking_task.py P01                                    # 課題だけ（カメラなし）
python aim_task/tracking_task.py P99 --simulate --fast --window         # 自動プレイ（動作確認）
```

1 人あたり: 静止 10 秒 → 練習 2 本 → 本番 12 本（各 46 秒）．約 15〜20 分．

画面の操作: SPACE = 開始・次へ，R = 結果の画面でその試行をやり直す，ESC = 中断（終わった試行までは保存済み）．
カメラの記録が始まらないときは R（もう一度）/ S（カメラなしで続ける）．
記録が途中で止まった・fps が落ちた・課題の全体を覆っていないときは結果の画面に赤字で出るので，R でやり直す．

## 1 試行の流れ（run_experiment.py）

SPACE → カメラの記録開始（最初のフレームが届くまで待つ）→ カウントダウン 3 s → 立ち上がり 3 s →
解析区間 40 s → 0.5 s 余分に記録して停止 → 保存 → 結果の画面

- 1 試行 = 1 つの .db3（`experiment/rec_trial.py` と同じく，接続 → 記録 → 切断）．
- 記録は別プロセスで行う（課題の画面は 240 fps で回るので，同じプロセスだとどちらかが詰まる）．
- フレームの時刻（`global_time_enabled`）と task.csv の `epoch` は同じ PC の時計．そのまま対応づけられる．
- 練習はカメラを記録しない（`rs_record_practice`）．静止の計測は記録する．

## 保存されるもの

```
recordings/aim_P01_20261020_140512/
    session.json                  設定・画面・計画・試行の一覧（記録の成否，やり直し）
    summary.csv                   試行ごとの成績・fps・カメラのフレーム数
    trial_00_static_pre/          静止（マウスを握ったまま 10 s）
        task.csv  meta.json  session.db3
    trial_01_practice_fast_s901/  練習（task.csv と meta.json だけ）
    trial_03_fast_s1/             本番
    trial_05_fast_s3_2/           R でやり直した試行（前の試行も残る．session.json の redone で区別）
```

- `task.csv`（1 画面フレーム 1 行）: frame, t（解析区間の先頭 = 0 s），epoch [s]，target_x/y_cm，cursor_x/y_cm，
  mouse_dx/dy_counts（生の移動量．y は奥へ押すと正），n_events，on_target
- `meta.json`: `experiment/` の trial_* と同じ形（status, frames, *_epoch_ms, color_intrinsics, calib …）+ `"task"`（軌道のパラメータ・開始時刻など）

カメラの容量は 640×480・60 fps・圧縮なしの概算で 1 試行 4 GB 程度，1 人 50〜60 GB．
起動時に必要量と空き容量を表示するので，最初のパイロットで実際の .db3 の大きさを確認すること．

## 記録の後

```
python experiment/rs_joints.py recordings/aim_P01_xxx/trial_03_fast_s1   # 関節の 3 次元座標（試行ごと）
python aim_task/check_sync.py recordings/aim_P01_xxx                     # 時刻合わせの確認（推定済みの試行すべて）
python aim_task/analyze_tracking.py                                      # マウスの解析（recordings/aim_* すべて）
```

- `check_sync.py`: マウスの速度を手首（Hand）の速度の 1 次式で表したとき最もよく合う時間差．
  同じ PC の時計なので数十 ms 以内のはず．±60 ms を超える，または相関が 0.7 未満なら「要確認」と出る．
- `analyze_tracking.py`: 周波数ごとのゲイン・位相，RMS 誤差，ICC など（出力は `recordings/analysis_aim/`）．

## ファイル

| ファイル | 内容 |
| --- | --- |
| task_config.py | 課題の設定（DPI，軌道，予告線，静止の長さ，カメラの扱い，保存先） |
| targets.py | 軌道（正弦波の和）と試行の順番．`python aim_task/targets.py` で一覧と図 |
| task_core.py | 画面・試行・保存の本体 |
| tracking_task.py | 課題だけ（カメラなし）．DPI の実測 |
| run_experiment.py | 課題 + RealSense の記録 |
| rs_recorder.py | RealSense の記録プロセス（experiment/lib/rs_device.py を使う） |
| check_sync.py | 課題と RealSense の時刻合わせの確認 |
| analyze_tracking.py | マウスの解析 |
| sim_player.py | 自動プレイ（動作確認用） |
