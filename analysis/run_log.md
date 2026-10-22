# 実行ログ（2026-10-22 22:59:01）

設定: C:\project\experiment\aim_eval\config.py　出力: C:\project\experiment\analysis　段階: joints

## plan 対象の確認

帯域を slow, fast に絞っています（他の帯域は推定も解析もしません）。

| セッション | 参加者 | DPI（指定） | DPI（記録時） | 試行 | 内訳 | .db3 | 推定済み | 状態 |
|---|---|---|---|---|---|---|---|---|
| aim_Yuuki400_20261022_154046 | Yuuki | 400 | 800 | 7 | fast，slow，静止 | 7 | 0 | 推定待ち 7 本 |
| aim_Yuuki800_20261022_153317 | Yuuki | 800 | 800 | 7 | fast，slow，静止 | 7 | 7 | OK（推定済み） |
| aim_Yuya400_20261022_134407 | Yuya | 400 | 800 | 7 | fast，slow，静止 | 7 | 7 | OK（推定済み） |
| aim_Yuya800_20261022_133738 | Yuya | 800 | 800 | 7 | fast，slow，静止 | 7 | 7 | OK（推定済み） |


> **aim_Yuuki400_20261022_154046**: 記録時 800 に対し 400 と指定。counts → cm の換算を 2.000 倍に直します（a0 の「手先 / マウス のゲイン」が 1 付近になるか確認してください）。

> **aim_Yuya400_20261022_134407**: 記録時 800 に対し 400 と指定。counts → cm の換算を 2.000 倍に直します（a0 の「手先 / マウス のゲイン」が 1 付近になるか確認してください）。

## step1 関節位置の推定と確認動画

### 推定する試行: **7 本**（CPU 16 コア，並列 4，stride 2。1 本 4.7 分なので目安 0 時間 8 分）

（4 本を並列に処理します）

- [1/7] trial_00_static_pre … OK（0.4 分，経過 0 分，残り目安 2 分）
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 240 mm  蝗帛��菴咲ｯ�蝗ｲ 2.5 mm  �ｼ�406 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 232 mm  蝗帛��菴咲ｯ�蝗ｲ 2.5 mm  �ｼ�406 frames�ｼ�
      [LENGTH] wrist窶杜cp3: 荳ｭ螟ｮ蛟､ 66 mm  蝗帛��菴咲ｯ�蝗ｲ 4.3 mm  �ｼ�406 frames�ｼ�
- [2/7] trial_03_slow_s3 … OK（1.0 分，経過 1 分，残り目安 2 分）
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 232 mm  蝗帛��菴咲ｯ�蝗ｲ 13.4 mm  �ｼ�1300 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 231 mm  蝗帛��菴咲ｯ�蝗ｲ 13.2 mm  �ｼ�1300 frames�ｼ�
      [LENGTH] wrist窶杜cp3: 荳ｭ螟ｮ蛟､ 68 mm  蝗帛��菴咲ｯ�蝗ｲ 5.2 mm  �ｼ�1300 frames�ｼ�
- [3/7] trial_02_slow_s2 … OK（1.0 分，経過 1 分，残り目安 1 分）
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 231 mm  蝗帛��菴咲ｯ�蝗ｲ 8.0 mm  �ｼ�1313 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 230 mm  蝗帛��菴咲ｯ�蝗ｲ 12.8 mm  �ｼ�1312 frames�ｼ�
      [LENGTH] wrist窶杜cp3: 荳ｭ螟ｮ蛟､ 69 mm  蝗帛��菴咲ｯ�蝗ｲ 5.2 mm  �ｼ�1312 frames�ｼ�
- [4/7] trial_01_slow_s1 … OK（1.0 分，経過 1 分，残り目安 1 分）
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 235 mm  蝗帛��菴咲ｯ�蝗ｲ 9.8 mm  �ｼ�1387 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 227 mm  蝗帛��菴咲ｯ�蝗ｲ 16.7 mm  �ｼ�1387 frames�ｼ�
      [LENGTH] wrist窶杜cp3: 荳ｭ螟ｮ蛟､ 70 mm  蝗帛��菴咲ｯ�蝗ｲ 4.6 mm  �ｼ�1387 frames�ｼ�
- [5/7] trial_04_fast_s1 … OK（1.0 分，経過 1 分，残り目安 1 分）
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 227 mm  蝗帛��菴咲ｯ�蝗ｲ 13.4 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 222 mm  蝗帛��菴咲ｯ�蝗ｲ 17.9 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] wrist窶杜cp3: 荳ｭ螟ｮ蛟､ 65 mm  蝗帛��菴咲ｯ�蝗ｲ 4.9 mm  �ｼ�1388 frames�ｼ�
- [6/7] trial_06_fast_s3 … OK（0.9 分，経過 2 分，残り目安 0 分）
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 223 mm  蝗帛��菴咲ｯ�蝗ｲ 15.0 mm  �ｼ�1387 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 220 mm  蝗帛��菴咲ｯ�蝗ｲ 14.4 mm  �ｼ�1387 frames�ｼ�
      [LENGTH] wrist窶杜cp3: 荳ｭ螟ｮ蛟､ 63 mm  蝗帛��菴咲ｯ�蝗ｲ 4.8 mm  �ｼ�1387 frames�ｼ�
- [7/7] trial_05_fast_s2 … OK（0.9 分，経過 2 分，残り目安 0 分）
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 225 mm  蝗帛��菴咲ｯ�蝗ｲ 15.1 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 220 mm  蝗帛��菴咲ｯ�蝗ｲ 15.1 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] wrist窶杜cp3: 荳ｭ螟ｮ蛟､ 64 mm  蝗帛��菴咲ｯ�蝗ｲ 4.7 mm  �ｼ�1388 frames�ｼ�

### 確認動画（20 fps）: 28 本

- [1/28] trial_00_static_pre
      確認動画: recordings\aim_Yuuki400_20261022_154046\trial_00_static_pre\joints\joints_check_20fps.mp4（20 fps，271 枚 / 811 フレーム）
- [2/28] trial_01_slow_s1
      確認動画: recordings\aim_Yuuki400_20261022_154046\trial_01_slow_s1\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2774 フレーム）
- [3/28] trial_02_slow_s2
      確認動画: recordings\aim_Yuuki400_20261022_154046\trial_02_slow_s2\joints\joints_check_20fps.mp4（20 fps，875 枚 / 2625 フレーム）
- [4/28] trial_03_slow_s3
      確認動画: recordings\aim_Yuuki400_20261022_154046\trial_03_slow_s3\joints\joints_check_20fps.mp4（20 fps，867 枚 / 2600 フレーム）
- [5/28] trial_04_fast_s1
      確認動画: recordings\aim_Yuuki400_20261022_154046\trial_04_fast_s1\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2775 フレーム）
- [6/28] trial_05_fast_s2
      確認動画: recordings\aim_Yuuki400_20261022_154046\trial_05_fast_s2\joints\joints_check_20fps.mp4（20 fps，926 枚 / 2776 フレーム）
- [7/28] trial_06_fast_s3
      確認動画: recordings\aim_Yuuki400_20261022_154046\trial_06_fast_s3\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2773 フレーム）
- [8/28] trial_00_static_pre
      確認動画: recordings\aim_Yuuki800_20261022_153317\trial_00_static_pre\joints\joints_check_20fps.mp4（すでにあります）
- [9/28] trial_01_slow_s1
      確認動画: recordings\aim_Yuuki800_20261022_153317\trial_01_slow_s1\joints\joints_check_20fps.mp4（すでにあります）
- [10/28] trial_02_slow_s2
      確認動画: recordings\aim_Yuuki800_20261022_153317\trial_02_slow_s2\joints\joints_check_20fps.mp4（すでにあります）
- [11/28] trial_03_slow_s3
      確認動画: recordings\aim_Yuuki800_20261022_153317\trial_03_slow_s3\joints\joints_check_20fps.mp4（すでにあります）
- [12/28] trial_04_fast_s1
      確認動画: recordings\aim_Yuuki800_20261022_153317\trial_04_fast_s1\joints\joints_check_20fps.mp4（すでにあります）
- [13/28] trial_05_fast_s2
      確認動画: recordings\aim_Yuuki800_20261022_153317\trial_05_fast_s2\joints\joints_check_20fps.mp4（すでにあります）
- [14/28] trial_06_fast_s3
      確認動画: recordings\aim_Yuuki800_20261022_153317\trial_06_fast_s3\joints\joints_check_20fps.mp4（すでにあります）
- [15/28] trial_00_static_pre
      確認動画: recordings\aim_Yuya400_20261022_134407\trial_00_static_pre\joints\joints_check_20fps.mp4（すでにあります）
- [16/28] trial_01_slow_s1
      確認動画: recordings\aim_Yuya400_20261022_134407\trial_01_slow_s1\joints\joints_check_20fps.mp4（すでにあります）
- [17/28] trial_02_slow_s2
      確認動画: recordings\aim_Yuya400_20261022_134407\trial_02_slow_s2\joints\joints_check_20fps.mp4（すでにあります）
- [18/28] trial_03_slow_s3
      確認動画: recordings\aim_Yuya400_20261022_134407\trial_03_slow_s3\joints\joints_check_20fps.mp4（すでにあります）
- [19/28] trial_04_fast_s1
      確認動画: recordings\aim_Yuya400_20261022_134407\trial_04_fast_s1\joints\joints_check_20fps.mp4（すでにあります）
- [20/28] trial_05_fast_s2
      確認動画: recordings\aim_Yuya400_20261022_134407\trial_05_fast_s2\joints\joints_check_20fps.mp4（すでにあります）
- [21/28] trial_06_fast_s3
      確認動画: recordings\aim_Yuya400_20261022_134407\trial_06_fast_s3\joints\joints_check_20fps.mp4（すでにあります）
- [22/28] trial_00_static_pre
      確認動画: recordings\aim_Yuya800_20261022_133738\trial_00_static_pre\joints\joints_check_20fps.mp4（すでにあります）
- [23/28] trial_01_slow_s1
      確認動画: recordings\aim_Yuya800_20261022_133738\trial_01_slow_s1\joints\joints_check_20fps.mp4（すでにあります）
- [24/28] trial_02_slow_s2
      確認動画: recordings\aim_Yuya800_20261022_133738\trial_02_slow_s2\joints\joints_check_20fps.mp4（すでにあります）
- [25/28] trial_03_slow_s3
      確認動画: recordings\aim_Yuya800_20261022_133738\trial_03_slow_s3\joints\joints_check_20fps.mp4（すでにあります）
- [26/28] trial_04_fast_s1
      確認動画: recordings\aim_Yuya800_20261022_133738\trial_04_fast_s1\joints\joints_check_20fps.mp4（すでにあります）
- [27/28] trial_05_fast_s2
      確認動画: recordings\aim_Yuya800_20261022_133738\trial_05_fast_s2\joints\joints_check_20fps.mp4（すでにあります）
- [28/28] trial_06_fast_s3
      確認動画: recordings\aim_Yuya800_20261022_133738\trial_06_fast_s3\joints\joints_check_20fps.mp4（すでにあります）

### 目視確認用の一覧

| セッション | 試行 | フレーム | 肩 有効率 [%] | 肘 有効率 [%] | 手首 有効率 [%] | MCP3 有効率 [%] | 肩 最長欠損 [s] | 肘 最長欠損 [s] | 手首 最長欠損 [s] | MCP3 最長欠損 [s] | 左右の取り違え [%] | NO_POSE | NO_HAND | REJECT | OUT | NO_DEPTH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aim_Yuuki400_20261022_154046 | trial_00_static_pre | 406 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki400_20261022_154046 | trial_01_slow_s1 | 1387 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki400_20261022_154046 | trial_02_slow_s2 | 1313 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.03 | 0.03 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki400_20261022_154046 | trial_03_slow_s3 | 1300 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki400_20261022_154046 | trial_04_fast_s1 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki400_20261022_154046 | trial_05_fast_s2 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki400_20261022_154046 | trial_06_fast_s3 | 1387 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki800_20261022_153317 | trial_00_static_pre | 405 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki800_20261022_153317 | trial_01_slow_s1 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki800_20261022_153317 | trial_02_slow_s2 | 1386 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki800_20261022_153317 | trial_03_slow_s3 | 1387 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki800_20261022_153317 | trial_04_fast_s1 | 1389 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki800_20261022_153317 | trial_05_fast_s2 | 1389 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki800_20261022_153317 | trial_06_fast_s3 | 1389 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuya400_20261022_134407 | trial_00_static_pre | 406 | 99 | 99 | 99 | 99 | 0.10 | 0.10 | 0.10 | 0.10 | 0.0 | 0.0 | 0.0 | 0.7 | 0.0 | 0.0 |
| aim_Yuya400_20261022_134407 | trial_01_slow_s1 | 1387 | 100 | 100 | 100 | 100 | 0.10 | 0.10 | 0.10 | 0.10 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0 |
| aim_Yuya400_20261022_134407 | trial_02_slow_s2 | 1387 | 100 | 100 | 100 | 100 | 0.03 | 0.03 | 0.03 | 0.03 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 |
| aim_Yuya400_20261022_134407 | trial_03_slow_s3 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.03 | 0.03 | 0.0 | 0.0 | 0.3 | 0.0 | 0.0 | 0.0 |
| aim_Yuya400_20261022_134407 | trial_04_fast_s1 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuya400_20261022_134407 | trial_05_fast_s2 | 1387 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuya400_20261022_134407 | trial_06_fast_s3 | 1387 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuya800_20261022_133738 | trial_00_static_pre | 405 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuya800_20261022_133738 | trial_01_slow_s1 | 1389 | 100 | 100 | 100 | 100 | 0.03 | 0.03 | 0.03 | 0.03 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 |
| aim_Yuya800_20261022_133738 | trial_02_slow_s2 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuya800_20261022_133738 | trial_03_slow_s3 | 1386 | 100 | 100 | 100 | 100 | 0.07 | 0.07 | 0.07 | 0.07 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 |
| aim_Yuya800_20261022_133738 | trial_04_fast_s1 | 1386 | 100 | 100 | 100 | 100 | 0.10 | 0.10 | 0.10 | 0.10 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0 |
| aim_Yuya800_20261022_133738 | trial_05_fast_s2 | 1386 | 100 | 100 | 100 | 100 | 0.13 | 0.13 | 0.13 | 0.13 | 0.0 | 0.0 | 0.0 | 0.3 | 0.0 | 0.0 |
| aim_Yuya800_20261022_133738 | trial_06_fast_s3 | 1387 | 100 | 100 | 100 | 100 | 0.13 | 0.13 | 0.13 | 0.13 | 0.0 | 0.0 | 0.0 | 0.3 | 0.0 | 0.0 |


完了。
