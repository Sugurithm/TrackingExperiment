# task — 2D エイムトラッキング課題（+ RealSense の記録）

正弦波の和（sum-of-sines）で動くターゲットを，マウスで 40 秒間追い続ける課題．
本番は `run_experiment.py` を実行するだけでよい（課題と RealSense の記録を同時に行う）．
カメラの設定と記録の処理は `experiment/config.py` と `experiment/lib/rs_device.py` をそのまま使うので，較正と同じ設定になる．

## 準備

```
pip install -r requirements.txt          # リポジトリ直下（pygame, matplotlib, pandas を追加した）
```

- Windows の「ポインターの精度を高める」をオフにする．Logicool G HUB などで DPI を固定する．
- マウスの DPI を実測して，`task/task_config.py` の `mouse_dpi` に入れる．
  `python task/tracking_task.py --measure-dpi` を実行し，定規に沿って 20 cm 動かす．
  ゆっくり動かした場合と速く動かした場合で同じ値になれば，加速は切れている．
- この日の較正（机にボードを平置き）を `experiment/rec_board.py` で撮っておき，そのフォルダ名を `--calib` に渡す．

## 実行（リポジトリの直下から）

```
python task/targets.py                                       # 軌道を作って画像に出す（task/trajectories.png）
python task/targets.py --each --plan P01                     # 1 本ずつの図（task/trajectories/）と P01 の試行の順番

python task/tracking_task.py P01                             # 課題だけ（カメラなし．練習・動作確認）
python task/tracking_task.py P99 --simulate --fast --window  # 自動プレイ（画面を見ずに保存まで確認）

python task/run_experiment.py P01 --calib calib_20261020_1400   # 本番（課題 + RealSense）
python task/run_experiment.py P01 --only fast:1 --no-practice   # 1 本だけ（パイロット・確認用）
```

1 人あたりの流れは，静止 10 秒 → 練習 2 本 → 本番 12 本（各 46 秒）で，約 15〜20 分．

画面の操作:
- SPACE = 開始・次へ
- R = 結果の画面で，その試行をやり直す
- ESC = 中断（終わった試行までは保存済み）

カメラの記録が始まらないときは，R（もう一度）か S（カメラなしで続ける）を選ぶ．
記録が途中で止まった・fps が落ちた・課題の全体を覆っていないときは結果の画面に赤字で出るので，R でやり直す．

## 1 試行の流れ（run_experiment.py）

SPACE → カメラの記録を開始（最初のフレームが届くまで待つ）→ カウントダウン 3 s → 立ち上がり 3 s →
解析区間 40 s → 0.5 s 余分に記録して停止 → 保存 → 結果の画面

- 1 試行 = 1 つの .db3．`experiment/rec_trial.py` と同じく，試行ごとに接続 → 記録 → 切断する．
- 記録は別プロセスで行う．課題の画面は 240 fps で回るので，同じプロセスだとどちらかが詰まるため．
- 時刻: RealSense のフレーム時刻（`global_time_enabled`）と task.csv の `epoch` は同じ PC の時計なので，そのまま対応づけられる．
- 練習はカメラを記録しない（`task_config.py` の `rs_record_practice`）．静止の計測は記録する．

## 保存されるもの

```
recordings/aim_P01_20261020_140512/
    session.json                  設定・画面・計画・試行の一覧（記録の成否，やり直し）
    summary.csv                   試行ごとの成績・fps・カメラのフレーム数
    trial_00_static_pre/          静止（マウスを握ったまま 10 s）
        task.csv  meta.json  session.db3
    trial_01_practice_fast_s901/  練習（task.csv と meta.json だけ）
    trial_03_fast_s1/             本番
    trial_05_fast_s3_2/           R でやり直した試行（前の試行も残す．session.json の redone で区別）
```

- `task.csv`（1 画面フレーム 1 行）の列:
  - frame
  - t（解析区間の先頭 = 0 s．カウントダウンと立ち上がりは負）
  - epoch [s]
  - target_x/y_cm，cursor_x/y_cm（画面中央 = 原点，x は右，y はマウスを奥へ押す向き）
  - mouse_dx/dy_counts（そのフレームに届いたマウスの生の移動量．y は奥へ押すと正）
  - n_events，on_target
- `meta.json`: `experiment/` の trial_* と同じ形（status, frames, *_epoch_ms, color_intrinsics, calib …）に，
  `"task"`（軌道のパラメータ・開始時刻など）を加えたもの．

関節の推定は，既存のプログラムで試行のフォルダごとに行う:

```
python experiment/rs_joints.py recordings/aim_P01_xxx/trial_03_fast_s1
```

カメラの容量の概算: 640×480・60 fps・圧縮なしで，1 試行 4 GB 程度，1 人 50〜60 GB．
起動時に必要量と空き容量を表示する．最初のパイロットで，実際の .db3 の大きさと書き込みが間に合っているか（fps）を確認すること．

## ファイル

| ファイル | 内容 |
| --- | --- |
| task_config.py | 課題の設定（DPI，軌道，予告線，静止の長さ，カメラの扱い，保存先） |
| targets.py | 軌道（正弦波の和）と試行の順番．単体で実行すると軌道の画像と一覧を出す |
| tracking_task.py | 課題だけ（カメラなし）．DPI の実測，自動プレイ |
| run_experiment.py | 課題 + RealSense の記録（本番用） |
| task_core.py | 画面・試行・保存の本体（tracking_task.py と run_experiment.py が共通で使う） |
| rs_recorder.py | RealSense の記録プロセス（experiment/lib/rs_device.py を使う） |
| sim_player.py | 自動プレイ（動作確認用） |
