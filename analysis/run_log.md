# 実行ログ（2026-10-22 16:14:01）

設定: C:\project\experiment\aim_eval\config.py　出力: C:\project\experiment\analysis　段階: joints

## plan 対象の確認

帯域を slow に絞っています（他の帯域は推定も解析もしません）。

| セッション | 参加者 | DPI（指定） | DPI（記録時） | 試行 | 内訳 | .db3 | 推定済み | 状態 |
|---|---|---|---|---|---|---|---|---|
| aim_Yuuki400_20261022_1540461 | Yuuki | 400 | – | 0 |  | 0 | 0 | session.json なし |
| aim_Yuuki800_20261022_153317 | Yuuki | 800 | 800 | 4 | slow，静止 | 4 | 0 | 推定待ち 4 本 |
| aim_Suzuka400_20261022_155937 | Suzuka | 400 | 800 | 4 | slow，静止 | 4 | 0 | 推定待ち 4 本 |
| aim_Suzuka800_20261022_155314 | Suzuka | 800 | 800 | 4 | slow，静止 | 4 | 0 | 推定待ち 4 本 |


> **aim_Suzuka400_20261022_155937**: 記録時 800 に対し 400 と指定。counts → cm の換算を 2.000 倍に直します（a0 の「手先 / マウス のゲイン」が 1 付近になるか確認してください）。

## step1 関節位置の推定と確認動画

### 推定する試行: **12 本**（CPU 16 コア，並列 4，stride 2。1 本 4.7 分なので目安 0 時間 13 分）

（4 本を並列に処理します）

- [1/12] trial_00_static_pre … OK（0.3 分，経過 0 分，残り目安 4 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 226 mm  蝗帛��菴咲ｯ�蝗ｲ 1.2 mm  �ｼ�405 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 230 mm  蝗帛��菴咲ｯ�蝗ｲ 2.5 mm  �ｼ�405 frames�ｼ�
- [2/12] trial_00_static_pre … OK（0.4 分，経過 1 分，残り目安 3 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 228 mm  蝗帛��菴咲ｯ�蝗ｲ 1.5 mm  �ｼ�405 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 212 mm  蝗帛��菴咲ｯ�蝗ｲ 2.2 mm  �ｼ�405 frames�ｼ�
- [3/12] trial_03_slow_s3 … OK（1.0 分，経過 1 分，残り目安 3 分）
- [4/12] trial_01_slow_s1 … OK（1.0 分，経過 1 分，残り目安 2 分）
          mcp3: 0 縺区園
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 225 mm  蝗帛��菴咲ｯ�蝗ｲ 7.3 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 227 mm  蝗帛��菴咲ｯ�蝗ｲ 7.4 mm  �ｼ�1387 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 233 mm  蝗帛��菴咲ｯ�蝗ｲ 8.4 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 239 mm  蝗帛��菴咲ｯ�蝗ｲ 7.6 mm  �ｼ�1387 frames�ｼ�
- [5/12] trial_02_slow_s2 … OK（1.0 分，経過 1 分，残り目安 1 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 224 mm  蝗帛��菴咲ｯ�蝗ｲ 6.8 mm  �ｼ�1386 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 235 mm  蝗帛��菴咲ｯ�蝗ｲ 7.9 mm  �ｼ�1386 frames�ｼ�
- [6/12] trial_00_static_pre … OK（0.4 分，経過 1 分，残り目安 1 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 221 mm  蝗帛��菴咲ｯ�蝗ｲ 2.6 mm  �ｼ�403 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 198 mm  蝗帛��菴咲ｯ�蝗ｲ 1.8 mm  �ｼ�403 frames�ｼ�
- [7/12] trial_01_slow_s1 … OK（1.0 分，経過 2 分，残り目安 1 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 227 mm  蝗帛��菴咲ｯ�蝗ｲ 7.2 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 205 mm  蝗帛��菴咲ｯ�蝗ｲ 16.1 mm  �ｼ�1388 frames�ｼ�
- [8/12] trial_03_slow_s3 … OK（1.0 分，経過 2 分，残り目安 1 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 226 mm  蝗帛��菴咲ｯ�蝗ｲ 7.0 mm  �ｼ�1295 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 200 mm  蝗帛��菴咲ｯ�蝗ｲ 13.5 mm  �ｼ�1294 frames�ｼ�
- [9/12] trial_02_slow_s2 … OK（1.0 分，経過 2 分，残り目安 1 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 227 mm  蝗帛��菴咲ｯ�蝗ｲ 7.8 mm  �ｼ�1307 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 201 mm  蝗帛��菴咲ｯ�蝗ｲ 13.7 mm  �ｼ�1306 frames�ｼ�
- [10/12] trial_01_slow_s1 … OK（1.0 分，経過 2 分，残り目安 0 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 220 mm  蝗帛��菴咲ｯ�蝗ｲ 5.5 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 193 mm  蝗帛��菴咲ｯ�蝗ｲ 9.5 mm  �ｼ�1388 frames�ｼ�
- [11/12] trial_02_slow_s2 … OK（1.0 分，経過 3 分，残り目安 0 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 223 mm  蝗帛��菴咲ｯ�蝗ｲ 5.2 mm  �ｼ�1385 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 194 mm  蝗帛��菴咲ｯ�蝗ｲ 9.8 mm  �ｼ�1384 frames�ｼ�
- [12/12] trial_03_slow_s3 … OK（0.9 分，経過 3 分，残り目安 0 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 216 mm  蝗帛��菴咲ｯ�蝗ｲ 6.8 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 202 mm  蝗帛��菴咲ｯ�蝗ｲ 9.4 mm  �ｼ�1388 frames�ｼ�

### 確認動画（20 fps）: 12 本

- [1/12] trial_00_static_pre
      確認動画: recordings\aim_Yuuki800_20261022_153317\trial_00_static_pre\joints\joints_check_20fps.mp4（20 fps，270 枚 / 810 フレーム）
- [2/12] trial_01_slow_s1
      確認動画: recordings\aim_Yuuki800_20261022_153317\trial_01_slow_s1\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2775 フレーム）
- [3/12] trial_02_slow_s2
      確認動画: recordings\aim_Yuuki800_20261022_153317\trial_02_slow_s2\joints\joints_check_20fps.mp4（20 fps，924 枚 / 2771 フレーム）
- [4/12] trial_03_slow_s3
      確認動画: recordings\aim_Yuuki800_20261022_153317\trial_03_slow_s3\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2774 フレーム）
- [5/12] trial_00_static_pre
      確認動画: recordings\aim_Suzuka400_20261022_155937\trial_00_static_pre\joints\joints_check_20fps.mp4（20 fps，270 枚 / 809 フレーム）
- [6/12] trial_01_slow_s1
      確認動画: recordings\aim_Suzuka400_20261022_155937\trial_01_slow_s1\joints\joints_check_20fps.mp4（20 fps，926 枚 / 2776 フレーム）
- [7/12] trial_02_slow_s2
      確認動画: recordings\aim_Suzuka400_20261022_155937\trial_02_slow_s2\joints\joints_check_20fps.mp4（20 fps，872 枚 / 2614 フレーム）
- [8/12] trial_03_slow_s3
      確認動画: recordings\aim_Suzuka400_20261022_155937\trial_03_slow_s3\joints\joints_check_20fps.mp4（20 fps，863 枚 / 2589 フレーム）
- [9/12] trial_00_static_pre
      確認動画: recordings\aim_Suzuka800_20261022_155314\trial_00_static_pre\joints\joints_check_20fps.mp4（20 fps，269 枚 / 806 フレーム）
- [10/12] trial_01_slow_s1
      確認動画: recordings\aim_Suzuka800_20261022_155314\trial_01_slow_s1\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2775 フレーム）
- [11/12] trial_02_slow_s2
      確認動画: recordings\aim_Suzuka800_20261022_155314\trial_02_slow_s2\joints\joints_check_20fps.mp4（20 fps，926 枚 / 2777 フレーム）
- [12/12] trial_03_slow_s3
      確認動画: recordings\aim_Suzuka800_20261022_155314\trial_03_slow_s3\joints\joints_check_20fps.mp4（20 fps，926 枚 / 2776 フレーム）

### 目視確認用の一覧

| セッション | 試行 | フレーム | 肩 有効率 [%] | 肘 有効率 [%] | 手首 有効率 [%] | MCP3 有効率 [%] | 肩 最長欠損 [s] | 肘 最長欠損 [s] | 手首 最長欠損 [s] | MCP3 最長欠損 [s] | 左右の取り違え [%] | NO_POSE | NO_HAND | REJECT | OUT | NO_DEPTH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aim_Yuuki800_20261022_153317 | trial_00_static_pre | 405 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki800_20261022_153317 | trial_01_slow_s1 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki800_20261022_153317 | trial_02_slow_s2 | 1386 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuuki800_20261022_153317 | trial_03_slow_s3 | 1387 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Suzuka400_20261022_155937 | trial_00_static_pre | 405 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Suzuka400_20261022_155937 | trial_01_slow_s1 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Suzuka400_20261022_155937 | trial_02_slow_s2 | 1307 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.03 | 0.03 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 |
| aim_Suzuka400_20261022_155937 | trial_03_slow_s3 | 1295 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.03 | 0.03 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 |
| aim_Suzuka800_20261022_155314 | trial_00_static_pre | 403 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Suzuka800_20261022_155314 | trial_01_slow_s1 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Suzuka800_20261022_155314 | trial_02_slow_s2 | 1389 | 100 | 100 | 100 | 100 | 0.00 | 0.10 | 0.10 | 0.10 | 0.0 | 0.0 | 0.1 | 0.1 | 0.0 | 0.0 |
| aim_Suzuka800_20261022_155314 | trial_03_slow_s3 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |


完了。
