# 実行ログ（2026-10-22 14:30:17）

設定: C:\project\experiment\aim_eval\config.py　出力: C:\project\experiment\analysis　段階: joints

## plan 対象の確認

帯域を slow に絞っています（他の帯域は推定も解析もしません）。

| セッション | 参加者 | DPI（指定） | DPI（記録時） | 試行 | 内訳 | .db3 | 推定済み | 状態 |
|---|---|---|---|---|---|---|---|---|
| aim_kakeru400_20261022_123951 | Kakeru | 400 | 800 | 4 | slow，静止 | 4 | 0 | 推定待ち 4 本 |
| aim_kakeru800_20261022_123338 | Kakeru | 800 | 800 | 4 | slow，静止 | 4 | 0 | 推定待ち 4 本 |
| aim_xin400_20261022_125957 | xin | 400 | 800 | 4 | slow，静止 | 4 | 0 | 推定待ち 4 本 |
| aim_xin800_20261022_125327 | xin | 800 | 800 | 4 | slow，静止 | 4 | 0 | 推定待ち 4 本 |
| aim_Yuya400_20261022_134407 | Yuya | 400 | 800 | 4 | slow，静止 | 4 | 0 | 推定待ち 4 本 |
| aim_Yuya800_20261022_133738 | Yuya | 800 | 800 | 4 | slow，静止 | 4 | 0 | 推定待ち 4 本 |
| aim_Syou400_20261022_135724 | Syou | 400 | 800 | 4 | slow，静止 | 4 | 0 | 推定待ち 4 本 |
| aim_Syou800_20261022_135037 | Syou | 800 | 800 | 4 | slow，静止 | 4 | 0 | 推定待ち 4 本 |


> **aim_kakeru400_20261022_123951**: 記録時 800 に対し 400 と指定。counts → cm の換算を 2.000 倍に直します（a0 の「手先 / マウス のゲイン」が 1 付近になるか確認してください）。

> **aim_xin400_20261022_125957**: 記録時 800 に対し 400 と指定。counts → cm の換算を 2.000 倍に直します（a0 の「手先 / マウス のゲイン」が 1 付近になるか確認してください）。

> **aim_Yuya400_20261022_134407**: 記録時 800 に対し 400 と指定。counts → cm の換算を 2.000 倍に直します（a0 の「手先 / マウス のゲイン」が 1 付近になるか確認してください）。

> **aim_Syou400_20261022_135724**: 記録時 800 に対し 400 と指定。counts → cm の換算を 2.000 倍に直します（a0 の「手先 / マウス のゲイン」が 1 付近になるか確認してください）。

## step1 関節位置の推定と確認動画

### 推定する試行: **32 本**（CPU 16 コア，並列 4，stride 2。1 本 4.7 分なので目安 0 時間 37 分）

（4 本を並列に処理します）

- [1/32] trial_00_static_pre … OK（0.3 分，経過 0 分，残り目安 11 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 252 mm  蝗帛��菴咲ｯ�蝗ｲ 1.7 mm  �ｼ�405 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 239 mm  蝗帛��菴咲ｯ�蝗ｲ 2.5 mm  �ｼ�405 frames�ｼ�
- [2/32] trial_00_static_pre … OK（0.3 分，経過 1 分，残り目安 10 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 257 mm  蝗帛��菴咲ｯ�蝗ｲ 1.9 mm  �ｼ�405 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 236 mm  蝗帛��菴咲ｯ�蝗ｲ 2.6 mm  �ｼ�405 frames�ｼ�
- [3/32] trial_02_slow_s2 … OK（1.0 分，経過 1 分，残り目安 10 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 246 mm  蝗帛��菴咲ｯ�蝗ｲ 11.7 mm  �ｼ�1387 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 224 mm  蝗帛��菴咲ｯ�蝗ｲ 22.5 mm  �ｼ�1387 frames�ｼ�
- [4/32] trial_01_slow_s1 … OK（1.0 分，経過 1 分，残り目安 7 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 238 mm  蝗帛��菴咲ｯ�蝗ｲ 26.0 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 201 mm  蝗帛��菴咲ｯ�蝗ｲ 44.2 mm  �ｼ�1388 frames�ｼ�
- [5/32] trial_03_slow_s3 … OK（1.0 分，経過 1 分，残り目安 5 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 238 mm  蝗帛��菴咲ｯ�蝗ｲ 21.9 mm  �ｼ�1385 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 212 mm  蝗帛��菴咲ｯ�蝗ｲ 33.9 mm  �ｼ�1385 frames�ｼ�
- [6/32] trial_00_static_pre … OK（0.4 分，経過 1 分，残り目安 6 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 264 mm  蝗帛��菴咲ｯ�蝗ｲ 5.4 mm  �ｼ�402 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 291 mm  蝗帛��菴咲ｯ�蝗ｲ 4.3 mm  �ｼ�402 frames�ｼ�
- [7/32] trial_01_slow_s1 … OK（1.0 分，経過 2 分，残り目安 6 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 257 mm  蝗帛��菴咲ｯ�蝗ｲ 6.4 mm  �ｼ�1389 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 231 mm  蝗帛��菴咲ｯ�蝗ｲ 9.6 mm  �ｼ�1389 frames�ｼ�
- [8/32] trial_03_slow_s3 … OK（1.1 分，経過 2 分，残り目安 6 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 249 mm  蝗帛��菴咲ｯ�蝗ｲ 6.9 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 221 mm  蝗帛��菴咲ｯ�蝗ｲ 11.7 mm  �ｼ�1388 frames�ｼ�
- [9/32] trial_02_slow_s2 … OK（1.1 分，経過 2 分，残り目安 5 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 248 mm  蝗帛��菴咲ｯ�蝗ｲ 9.4 mm  �ｼ�1389 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 224 mm  蝗帛��菴咲ｯ�蝗ｲ 8.7 mm  �ｼ�1389 frames�ｼ�
- [10/32] trial_01_slow_s1 … OK（1.0 分，経過 2 分，残り目安 5 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 264 mm  蝗帛��菴咲ｯ�蝗ｲ 9.1 mm  �ｼ�1386 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 255 mm  蝗帛��菴咲ｯ�蝗ｲ 30.2 mm  �ｼ�1386 frames�ｼ�
- [11/32] trial_00_static_pre … OK（0.4 分，経過 2 分，残り目安 5 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 244 mm  蝗帛��菴咲ｯ�蝗ｲ 6.2 mm  �ｼ�399 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 278 mm  蝗帛��菴咲ｯ�蝗ｲ 4.9 mm  �ｼ�399 frames�ｼ�
- [12/32] trial_02_slow_s2 … OK（1.0 分，経過 3 分，残り目安 5 分）
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.1%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
              蜈･繧梧崛縺�   44.52窶�  44.55 s
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 262 mm  蝗帛��菴咲ｯ�蝗ｲ 8.1 mm  �ｼ�1381 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 258 mm  蝗帛��菴咲ｯ�蝗ｲ 17.9 mm  �ｼ�1380 frames�ｼ�
- [13/32] trial_03_slow_s3 … OK（1.0 分，経過 3 分，残り目安 5 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 260 mm  蝗帛��菴咲ｯ�蝗ｲ 9.4 mm  �ｼ�1385 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 256 mm  蝗帛��菴咲ｯ�蝗ｲ 29.3 mm  �ｼ�1385 frames�ｼ�
- [14/32] trial_01_slow_s1 … OK（1.0 分，経過 3 分，残り目安 4 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 258 mm  蝗帛��菴咲ｯ�蝗ｲ 8.2 mm  �ｼ�1382 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 272 mm  蝗帛��菴咲ｯ�蝗ｲ 16.4 mm  �ｼ�1382 frames�ｼ�
- [15/32] trial_02_slow_s2 … OK（1.0 分，経過 3 分，残り目安 4 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 262 mm  蝗帛��菴咲ｯ�蝗ｲ 6.4 mm  �ｼ�1382 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 280 mm  蝗帛��菴咲ｯ�蝗ｲ 11.8 mm  �ｼ�1382 frames�ｼ�
- [16/32] trial_00_static_pre … OK（0.4 分，経過 3 分，残り目安 3 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 236 mm  蝗帛��菴咲ｯ�蝗ｲ 1.7 mm  �ｼ�403 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 260 mm  蝗帛��菴咲ｯ�蝗ｲ 2.5 mm  �ｼ�403 frames�ｼ�
- [17/32] trial_03_slow_s3 … OK（1.0 分，経過 4 分，残り目安 3 分）
                 0.00窶�   0.74 s  ( 0.74 s)  荳ｻ縺ｪ逅�逕ｱ REJECT
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 265 mm  蝗帛��菴咲ｯ�蝗ｲ 6.7 mm  �ｼ�1364 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 276 mm  蝗帛��菴咲ｯ�蝗ｲ 17.1 mm  �ｼ�1364 frames�ｼ�
- [18/32] trial_00_static_pre … OK（0.4 分，経過 4 分，残り目安 3 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 244 mm  蝗帛��菴咲ｯ�蝗ｲ 2.8 mm  �ｼ�405 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 258 mm  蝗帛��菴咲ｯ�蝗ｲ 2.8 mm  �ｼ�405 frames�ｼ�
- [19/32] trial_01_slow_s1 … OK（1.1 分，経過 5 分，残り目安 3 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 238 mm  蝗帛��菴咲ｯ�蝗ｲ 10.6 mm  �ｼ�1384 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 241 mm  蝗帛��菴咲ｯ�蝗ｲ 23.4 mm  �ｼ�1384 frames�ｼ�
- [20/32] trial_03_slow_s3 … OK（1.0 分，経過 5 分，残り目安 3 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 242 mm  蝗帛��菴咲ｯ�蝗ｲ 12.6 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 251 mm  蝗帛��菴咲ｯ�蝗ｲ 17.5 mm  �ｼ�1384 frames�ｼ�
- [21/32] trial_02_slow_s2 … OK（1.1 分，経過 5 分，残り目安 2 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 241 mm  蝗帛��菴咲ｯ�蝗ｲ 8.1 mm  �ｼ�1386 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 247 mm  蝗帛��菴咲ｯ�蝗ｲ 17.1 mm  �ｼ�1386 frames�ｼ�
- [22/32] trial_00_static_pre … OK（0.4 分，経過 5 分，残り目安 2 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 214 mm  蝗帛��菴咲ｯ�蝗ｲ 2.9 mm  �ｼ�406 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 219 mm  蝗帛��菴咲ｯ�蝗ｲ 2.2 mm  �ｼ�406 frames�ｼ�
- [23/32] trial_01_slow_s1 … OK（1.0 分，経過 5 分，残り目安 2 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 244 mm  蝗帛��菴咲ｯ�蝗ｲ 4.8 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 249 mm  蝗帛��菴咲ｯ�蝗ｲ 8.1 mm  �ｼ�1388 frames�ｼ�
- [24/32] trial_02_slow_s2 … OK（1.0 分，経過 6 分，残り目安 2 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 242 mm  蝗帛��菴咲ｯ�蝗ｲ 4.3 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 248 mm  蝗帛��菴咲ｯ�蝗ｲ 7.9 mm  �ｼ�1388 frames�ｼ�
- [25/32] trial_03_slow_s3 … OK（1.0 分，経過 6 分，残り目安 2 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 238 mm  蝗帛��菴咲ｯ�蝗ｲ 6.3 mm  �ｼ�1384 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 245 mm  蝗帛��菴咲ｯ�蝗ｲ 6.8 mm  �ｼ�1384 frames�ｼ�
- [26/32] trial_00_static_pre … OK（0.4 分，経過 6 分，残り目安 1 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 227 mm  蝗帛��菴咲ｯ�蝗ｲ 11.0 mm  �ｼ�406 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 221 mm  蝗帛��菴咲ｯ�蝗ｲ 2.9 mm  �ｼ�406 frames�ｼ�
- [27/32] trial_01_slow_s1 … OK（1.1 分，経過 6 分，残り目安 1 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 233 mm  蝗帛��菴咲ｯ�蝗ｲ 10.9 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 229 mm  蝗帛��菴咲ｯ�蝗ｲ 12.8 mm  �ｼ�1388 frames�ｼ�
- [28/32] trial_02_slow_s2 … OK（1.0 分，経過 6 分，残り目安 1 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 240 mm  蝗帛��菴咲ｯ�蝗ｲ 9.1 mm  �ｼ�1387 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 227 mm  蝗帛��菴咲ｯ�蝗ｲ 9.2 mm  �ｼ�1388 frames�ｼ�
- [29/32] trial_03_slow_s3 … OK（1.1 分，経過 7 分，残り目安 1 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 240 mm  蝗帛��菴咲ｯ�蝗ｲ 10.8 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 231 mm  蝗帛��菴咲ｯ�蝗ｲ 8.9 mm  �ｼ�1388 frames�ｼ�
- [30/32] trial_01_slow_s1 … OK（1.0 分，経過 7 分，残り目安 0 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 229 mm  蝗帛��菴咲ｯ�蝗ｲ 7.4 mm  �ｼ�1387 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 225 mm  蝗帛��菴咲ｯ�蝗ｲ 8.5 mm  �ｼ�1387 frames�ｼ�
- [31/32] trial_02_slow_s2 … OK（1.0 分，経過 7 分，残り目安 0 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 234 mm  蝗帛��菴咲ｯ�蝗ｲ 5.8 mm  �ｼ�1388 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 226 mm  蝗帛��菴咲ｯ�蝗ｲ 5.3 mm  �ｼ�1388 frames�ｼ�
- [32/32] trial_03_slow_s3 … OK（1.0 分，経過 7 分，残り目安 0 分）
          mcp3: 0 縺区園
      [蟾ｦ蜿ｳ] Pose 縺ｮ縺ゅｋ繝輔Ξ繝ｼ繝縺ｮ縺�縺｡ 蜈･繧梧崛縺� 0.0%  �ｼ域ｱｺ繧∵婿: label 0%  shoulders 100%  continuity 0%�ｼ�
      [LENGTH] shoulder窶兎lbow: 荳ｭ螟ｮ蛟､ 234 mm  蝗帛��菴咲ｯ�蝗ｲ 9.5 mm  �ｼ�1386 frames�ｼ�
      [LENGTH] elbow窶努rist: 荳ｭ螟ｮ蛟､ 227 mm  蝗帛��菴咲ｯ�蝗ｲ 6.2 mm  �ｼ�1386 frames�ｼ�

### 確認動画（20 fps）: 32 本

- [1/32] trial_00_static_pre
      確認動画: recordings\aim_kakeru400_20261022_123951\trial_00_static_pre\joints\joints_check_20fps.mp4（20 fps，270 枚 / 810 フレーム）
- [2/32] trial_01_slow_s1
      確認動画: recordings\aim_kakeru400_20261022_123951\trial_01_slow_s1\joints\joints_check_20fps.mp4（20 fps，926 枚 / 2776 フレーム）
- [3/32] trial_02_slow_s2
      確認動画: recordings\aim_kakeru400_20261022_123951\trial_02_slow_s2\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2773 フレーム）
- [4/32] trial_03_slow_s3
      確認動画: recordings\aim_kakeru400_20261022_123951\trial_03_slow_s3\joints\joints_check_20fps.mp4（20 fps，924 枚 / 2772 フレーム）
- [5/32] trial_00_static_pre
      確認動画: recordings\aim_kakeru800_20261022_123338\trial_00_static_pre\joints\joints_check_20fps.mp4（20 fps，270 枚 / 810 フレーム）
- [6/32] trial_01_slow_s1
      確認動画: recordings\aim_kakeru800_20261022_123338\trial_01_slow_s1\joints\joints_check_20fps.mp4（20 fps，926 枚 / 2777 フレーム）
- [7/32] trial_02_slow_s2
      確認動画: recordings\aim_kakeru800_20261022_123338\trial_02_slow_s2\joints\joints_check_20fps.mp4（20 fps，926 枚 / 2777 フレーム）
- [8/32] trial_03_slow_s3
      確認動画: recordings\aim_kakeru800_20261022_123338\trial_03_slow_s3\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2775 フレーム）
- [9/32] trial_00_static_pre
      確認動画: recordings\aim_xin400_20261022_125957\trial_00_static_pre\joints\joints_check_20fps.mp4（20 fps，271 枚 / 812 フレーム）
- [10/32] trial_01_slow_s1
      確認動画: recordings\aim_xin400_20261022_125957\trial_01_slow_s1\joints\joints_check_20fps.mp4（20 fps，926 枚 / 2777 フレーム）
- [11/32] trial_02_slow_s2
      確認動画: recordings\aim_xin400_20261022_125957\trial_02_slow_s2\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2774 フレーム）
- [12/32] trial_03_slow_s3
      確認動画: recordings\aim_xin400_20261022_125957\trial_03_slow_s3\joints\joints_check_20fps.mp4（20 fps，926 枚 / 2777 フレーム）
- [13/32] trial_00_static_pre
      確認動画: recordings\aim_xin800_20261022_125327\trial_00_static_pre\joints\joints_check_20fps.mp4（20 fps，270 枚 / 810 フレーム）
- [14/32] trial_01_slow_s1
      確認動画: recordings\aim_xin800_20261022_125327\trial_01_slow_s1\joints\joints_check_20fps.mp4（20 fps，926 枚 / 2776 フレーム）
- [15/32] trial_02_slow_s2
      確認動画: recordings\aim_xin800_20261022_125327\trial_02_slow_s2\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2773 フレーム）
- [16/32] trial_03_slow_s3
      確認動画: recordings\aim_xin800_20261022_125327\trial_03_slow_s3\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2774 フレーム）
- [17/32] trial_00_static_pre
      確認動画: recordings\aim_Yuya400_20261022_134407\trial_00_static_pre\joints\joints_check_20fps.mp4（20 fps，271 枚 / 811 フレーム）
- [18/32] trial_01_slow_s1
      確認動画: recordings\aim_Yuya400_20261022_134407\trial_01_slow_s1\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2773 フレーム）
- [19/32] trial_02_slow_s2
      確認動画: recordings\aim_Yuya400_20261022_134407\trial_02_slow_s2\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2774 フレーム）
- [20/32] trial_03_slow_s3
      確認動画: recordings\aim_Yuya400_20261022_134407\trial_03_slow_s3\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2775 フレーム）
- [21/32] trial_00_static_pre
      確認動画: recordings\aim_Yuya800_20261022_133738\trial_00_static_pre\joints\joints_check_20fps.mp4（20 fps，270 枚 / 810 フレーム）
- [22/32] trial_01_slow_s1
      確認動画: recordings\aim_Yuya800_20261022_133738\trial_01_slow_s1\joints\joints_check_20fps.mp4（20 fps，926 枚 / 2777 フレーム）
- [23/32] trial_02_slow_s2
      確認動画: recordings\aim_Yuya800_20261022_133738\trial_02_slow_s2\joints\joints_check_20fps.mp4（20 fps，926 枚 / 2776 フレーム）
- [24/32] trial_03_slow_s3
      確認動画: recordings\aim_Yuya800_20261022_133738\trial_03_slow_s3\joints\joints_check_20fps.mp4（20 fps，924 枚 / 2772 フレーム）
- [25/32] trial_00_static_pre
      確認動画: recordings\aim_Syou400_20261022_135724\trial_00_static_pre\joints\joints_check_20fps.mp4（20 fps，271 枚 / 811 フレーム）
- [26/32] trial_01_slow_s1
      確認動画: recordings\aim_Syou400_20261022_135724\trial_01_slow_s1\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2775 フレーム）
- [27/32] trial_02_slow_s2
      確認動画: recordings\aim_Syou400_20261022_135724\trial_02_slow_s2\joints\joints_check_20fps.mp4（20 fps，926 枚 / 2776 フレーム）
- [28/32] trial_03_slow_s3
      確認動画: recordings\aim_Syou400_20261022_135724\trial_03_slow_s3\joints\joints_check_20fps.mp4（20 fps，926 枚 / 2776 フレーム）
- [29/32] trial_00_static_pre
      確認動画: recordings\aim_Syou800_20261022_135037\trial_00_static_pre\joints\joints_check_20fps.mp4（20 fps，271 枚 / 812 フレーム）
- [30/32] trial_01_slow_s1
      確認動画: recordings\aim_Syou800_20261022_135037\trial_01_slow_s1\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2774 フレーム）
- [31/32] trial_02_slow_s2
      確認動画: recordings\aim_Syou800_20261022_135037\trial_02_slow_s2\joints\joints_check_20fps.mp4（20 fps，925 枚 / 2775 フレーム）
- [32/32] trial_03_slow_s3
      確認動画: recordings\aim_Syou800_20261022_135037\trial_03_slow_s3\joints\joints_check_20fps.mp4（20 fps，924 枚 / 2772 フレーム）

### 目視確認用の一覧

| セッション | 試行 | フレーム | 肩 有効率 [%] | 肘 有効率 [%] | 手首 有効率 [%] | MCP3 有効率 [%] | 肩 最長欠損 [s] | 肘 最長欠損 [s] | 手首 最長欠損 [s] | MCP3 最長欠損 [s] | 左右の取り違え [%] | NO_POSE | NO_HAND | REJECT | OUT | NO_DEPTH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aim_kakeru400_20261022_123951 | trial_00_static_pre | 405 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_kakeru400_20261022_123951 | trial_01_slow_s1 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_kakeru400_20261022_123951 | trial_02_slow_s2 | 1387 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_kakeru400_20261022_123951 | trial_03_slow_s3 | 1386 | 100 | 100 | 100 | 100 | 0.03 | 0.03 | 0.03 | 0.03 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 |
| aim_kakeru800_20261022_123338 | trial_00_static_pre | 405 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_kakeru800_20261022_123338 | trial_01_slow_s1 | 1389 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_kakeru800_20261022_123338 | trial_02_slow_s2 | 1389 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_kakeru800_20261022_123338 | trial_03_slow_s3 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_xin400_20261022_125957 | trial_00_static_pre | 406 | 99 | 99 | 99 | 99 | 0.13 | 0.13 | 0.13 | 0.13 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| aim_xin400_20261022_125957 | trial_01_slow_s1 | 1389 | 100 | 100 | 100 | 100 | 0.07 | 0.07 | 0.07 | 0.07 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0 |
| aim_xin400_20261022_125957 | trial_02_slow_s2 | 1387 | 100 | 100 | 99 | 99 | 0.10 | 0.10 | 0.10 | 0.10 | 0.1 | 0.0 | 0.0 | 0.5 | 0.0 | 0.0 |
| aim_xin400_20261022_125957 | trial_03_slow_s3 | 1389 | 100 | 100 | 100 | 100 | 0.13 | 0.13 | 0.13 | 0.13 | 0.0 | 0.0 | 0.0 | 0.3 | 0.0 | 0.0 |
| aim_xin800_20261022_125327 | trial_00_static_pre | 405 | 99 | 99 | 99 | 99 | 0.20 | 0.20 | 0.20 | 0.20 | 0.0 | 0.0 | 0.0 | 1.5 | 0.0 | 0.0 |
| aim_xin800_20261022_125327 | trial_01_slow_s1 | 1388 | 100 | 100 | 100 | 100 | 0.20 | 0.20 | 0.20 | 0.20 | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 | 0.0 |
| aim_xin800_20261022_125327 | trial_02_slow_s2 | 1387 | 100 | 100 | 100 | 100 | 0.17 | 0.17 | 0.17 | 0.17 | 0.0 | 0.0 | 0.0 | 0.4 | 0.0 | 0.0 |
| aim_xin800_20261022_125327 | trial_03_slow_s3 | 1387 | 98 | 98 | 98 | 98 | 0.77 | 0.77 | 0.77 | 0.77 | 0.0 | 0.0 | 0.0 | 1.7 | 0.0 | 0.0 |
| aim_Yuya400_20261022_134407 | trial_00_static_pre | 406 | 99 | 99 | 99 | 99 | 0.10 | 0.10 | 0.10 | 0.10 | 0.0 | 0.0 | 0.0 | 0.7 | 0.0 | 0.0 |
| aim_Yuya400_20261022_134407 | trial_01_slow_s1 | 1387 | 100 | 100 | 100 | 100 | 0.10 | 0.10 | 0.10 | 0.10 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0 |
| aim_Yuya400_20261022_134407 | trial_02_slow_s2 | 1387 | 100 | 100 | 100 | 100 | 0.03 | 0.03 | 0.03 | 0.03 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 |
| aim_Yuya400_20261022_134407 | trial_03_slow_s3 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.03 | 0.03 | 0.0 | 0.0 | 0.3 | 0.0 | 0.0 | 0.0 |
| aim_Yuya800_20261022_133738 | trial_00_static_pre | 405 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuya800_20261022_133738 | trial_01_slow_s1 | 1389 | 100 | 100 | 100 | 100 | 0.03 | 0.03 | 0.03 | 0.03 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 |
| aim_Yuya800_20261022_133738 | trial_02_slow_s2 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Yuya800_20261022_133738 | trial_03_slow_s3 | 1386 | 100 | 100 | 100 | 100 | 0.07 | 0.07 | 0.07 | 0.07 | 0.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 |
| aim_Syou400_20261022_135724 | trial_00_static_pre | 406 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Syou400_20261022_135724 | trial_01_slow_s1 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Syou400_20261022_135724 | trial_02_slow_s2 | 1388 | 100 | 100 | 100 | 100 | 0.03 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Syou400_20261022_135724 | trial_03_slow_s3 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Syou800_20261022_135037 | trial_00_static_pre | 406 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Syou800_20261022_135037 | trial_01_slow_s1 | 1387 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Syou800_20261022_135037 | trial_02_slow_s2 | 1388 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aim_Syou800_20261022_135037 | trial_03_slow_s3 | 1386 | 100 | 100 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |


完了。
