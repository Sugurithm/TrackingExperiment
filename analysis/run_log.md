# 実行ログ（2026-10-22 07:47:47）

設定: C:\project\experiment\aim_eval\config.py　出力: C:\project\experiment\analysis　段階: joints

## plan 対象の確認

| セッション | 参加者 | DPI（指定） | DPI（記録時） | 試行 | .db3 | 推定済み | 状態 |
|---|---|---|---|---|---|---|---|
| aim_Keito400_20261021_170947 | Keito | 400 | 800 | 7 | 7 | 0 | 推定待ち 7 本 |
| aim_Keito800_20261021_171605 | Keito | 800 | 800 | 7 | 7 | 0 | 推定待ち 7 本 |
| aim_Keito1600_20261021_172214 | Keito | 1600 | 800 | 7 | 7 | 0 | 推定待ち 7 本 |
| aim_Kosuke400_20261021_172948 | Kosuke | 400 | 800 | 7 | 7 | 0 | 推定待ち 7 本 |
| aim_Kosuke800_20261021_173700 | Kosuke | 800 | 800 | 7 | 7 | 0 | 推定待ち 7 本 |
| aim_Kosuke1600_20261021_174337 | Kosuke | 1600 | 800 | 7 | 7 | 0 | 推定待ち 7 本 |
| aim_Suguru400_20261021_175327 | Suguru | 400 | 800 | 7 | 7 | 0 | 推定待ち 7 本 |
| aim_Suguru800_20261021_175931 | Suguru | 800 | 800 | 7 | 7 | 0 | 推定待ち 7 本 |
| aim_Suguru800_20261021_175931 | Suguru | 1600 | 800 | 7 | 7 | 0 | 推定待ち 7 本 |


> **aim_Keito400_20261021_170947**: 記録時 800 に対し 400 と指定。counts → cm の換算を 2.000 倍に直します（a0 の「手先 / マウス のゲイン」が 1 付近になるか確認してください）。

> **aim_Keito1600_20261021_172214**: 記録時 800 に対し 1600 と指定。counts → cm の換算を 0.500 倍に直します（a0 の「手先 / マウス のゲイン」が 1 付近になるか確認してください）。

> **aim_Kosuke400_20261021_172948**: 記録時 800 に対し 400 と指定。counts → cm の換算を 2.000 倍に直します（a0 の「手先 / マウス のゲイン」が 1 付近になるか確認してください）。

> **aim_Kosuke1600_20261021_174337**: 記録時 800 に対し 1600 と指定。counts → cm の換算を 0.500 倍に直します（a0 の「手先 / マウス のゲイン」が 1 付近になるか確認してください）。

> **aim_Suguru400_20261021_175327**: 記録時 800 に対し 400 と指定。counts → cm の換算を 2.000 倍に直します（a0 の「手先 / マウス のゲイン」が 1 付近になるか確認してください）。

> **aim_Suguru800_20261021_175931**: 記録時 800 に対し 1600 と指定。counts → cm の換算を 0.500 倍に直します（a0 の「手先 / マウス のゲイン」が 1 付近になるか確認してください）。

## step1 関節位置の推定

推定する試行: **63 本**（1 本 3〜7 分なので目安 4 時間 12 分）

- [1/63] trial_00_static_pre
      失敗（77 s）
        [TIME] 1 フレームあたり [ms]: 読み込み・位置合わせ   6.1  Pose  58.3  Hand  20.3  深度→3 次元   0.5  描画・表示・動画   0.1  （合計 1.2 分，11.1 fps）
        [VALID 全体]     shoulder: 100%  elbow: 100%  wrist: 100%  wrist_pose: 100%  mcp3: 100%
        [VALID 左 全体]  shoulder: 100%  elbow: 100%  wrist: 100%  wrist_pose: 100%  mcp3: 100%  （参考）
        [VALID 5 秒ごと] shoulder  elbow  wrist  wrist_pose  mcp3（%）
          File "C:\project\experiment\experiment\rs_joints.py", line 488, in print_summary
            print(f"    {s0:5.0f}\u2013{s0 + 5:3.0f} s  " + "  ".join(f"{100 * ok[m, k].mean():5.0f}" for k in range(len(NAMES))))
        UnicodeEncodeError: 'cp932' codec can't encode character '\u2013' in position 9: illegal multibyte sequence
- [2/63] trial_01_slow_s1
      失敗（559 s）
        [TIME] 1 フレームあたり [ms]: 読み込み・位置合わせ  10.6  Pose 137.5  Hand  49.5  深度→3 次元   1.0  描画・表示・動画   0.1  （合計 9.3 分，5.0 fps）
        [VALID 全体]     shoulder: 100%  elbow: 100%  wrist: 100%  wrist_pose: 100%  mcp3: 100%
        [VALID 左 全体]  shoulder: 100%  elbow: 100%  wrist: 100%  wrist_pose: 100%  mcp3: 100%  （参考）
        [VALID 5 秒ごと] shoulder  elbow  wrist  wrist_pose  mcp3（%）
          File "C:\project\experiment\experiment\rs_joints.py", line 488, in print_summary
            print(f"    {s0:5.0f}\u2013{s0 + 5:3.0f} s  " + "  ".join(f"{100 * ok[m, k].mean():5.0f}" for k in range(len(NAMES))))
        UnicodeEncodeError: 'cp932' codec can't encode character '\u2013' in position 9: illegal multibyte sequence
- [3/63] trial_02_slow_s2
      失敗（370 s）
        [TIME] 1 フレームあたり [ms]: 読み込み・位置合わせ   8.1  Pose  89.7  Hand  32.2  深度→3 次元   0.7  描画・表示・動画   0.0  （合計 6.1 分，7.5 fps）
        [VALID 全体]     shoulder: 100%  elbow: 100%  wrist: 100%  wrist_pose: 100%  mcp3: 100%
        [VALID 左 全体]  shoulder: 100%  elbow: 100%  wrist:  99%  wrist_pose: 100%  mcp3:  94%  （参考）
        [VALID 5 秒ごと] shoulder  elbow  wrist  wrist_pose  mcp3（%）
          File "C:\project\experiment\experiment\rs_joints.py", line 488, in print_summary
            print(f"    {s0:5.0f}\u2013{s0 + 5:3.0f} s  " + "  ".join(f"{100 * ok[m, k].mean():5.0f}" for k in range(len(NAMES))))
        UnicodeEncodeError: 'cp932' codec can't encode character '\u2013' in position 9: illegal multibyte sequence
- [4/63] trial_03_slow_s3
