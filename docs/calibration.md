# MoCap(Motive) ↔ RealSense キャリブレーション

## 出力
`recordings/calib/T_mocap_cam.json`
- `T_mocap_cam` (4x4): `p_mocap = T_mocap_cam @ [p_cam; 1]`，単位 m．
- `cam` = RealSense **カラー光学座標系**（X右, Y下, Z前方）．`rs_estimate.py` の骨格（depth を color に align → color 内部パラメータで deproject）と同じ座標系なので，そのまま掛けられる．
- 残差・LOO 誤差・各姿勢の診断値も同じファイルに保存．

## 使い方
1. `recordings/calib/` に姿勢ごとの Motive CSV と RealSense db3 を置く．
2. `scripts/calib_mocap_rs.py` 冒頭の `POSES` を実ファイル名に合わせる．
3. `python scripts/calib_mocap_rs.py`

## 手法
- Motive: 各フレームの3点を距離（155 < 230 < 277 mm）で S1/S2/S3 に識別 → 中央値．
  ラベルを使わないので Unlabeled マーカやラベル入れ替わりにも対応．
- RealSense: 各フレーム PnP（`board_calcAxis.py` と同じ手順）→ S1–S3 のボード座標をカメラ座標へ → 中央値．
- 3点 × 5姿勢 = 15 対応点から Kabsch で剛体変換（スケールなし）．
- Motive の生の計測点を直接対応させるので，MoCap 側でボード座標系を組み立てる必要はない（組み立てるとマーカ誤差が回転に増幅される）．

## 実行前に確認すること（最低限）
- **Motive CSV のエクスポート設定**: Markers を出力に含める．単位は `Length Units` 行から自動換算（m / cm / mm）．`Coordinate Space` は Global．
  Motive は既定で Y-up の右手系．Up 軸の違いは剛体変換に吸収されるが，左手系（鏡像）だと合わない → スクリプトが警告を出す．
- **静止**: 各姿勢の記録中にボードが動いていないこと（std が 2 mm 超で警告）．
- **マーカ識別**: 全フレームで識別失敗なら，CSV の Type 列名（`Marker`）か，マーカ間距離の公称値を確認．Motive が計測した距離は実寸確認にもなる（230.0 / 155.0 / 277.4 mm との差を表示）．
- **マーカ中心の高さ** `board_lib.MARKER_CENTER_Z`: 平らなステッカーなら 0．半球/球マーカに替えたら −半径（カメラ側が負）．
- **姿勢の多様性**: 5姿勢が位置・傾きともにばらけていること（同一平面・平行移動だけだと回転が不定に近い）．計測したい作業空間全体をカバーする．
- **判定の目安**: RMS(fit) と RMS(LOO) がほぼ同じで数 mm 程度なら良好．LOO だけ大きい姿勢は外れ（誤検出・動き）を疑う．

## 次に進めること
1. `rs_estimate.py` の `skeleton3d.npz` の各点に `T_mocap_cam` を掛け，MoCap 座標系に変換する．
2. **時刻同期**: Motive と RealSense の時刻を対応づける（静止キャリブでは不要だったが，動きの誤差検証では必須）．
   Motive の CSV は `Capture Start Time` + 相対時刻，RealSense は `global_time_enabled` による PC epoch ms．
   同一 PC でなければ時計差がある → 素早い動作（タップ等）の相互相関でオフセットを推定するのが簡単．
3. **誤差検証**: 同じ部位に MoCap マーカを貼り，対応点の距離（3D 誤差）と軸別誤差を時系列で評価．
   `board_lib.fit_rigid` / `transform` と JSON の `points_*` をそのまま流用できる．

## 考慮事項（メモのみ）
- MediaPipe の関節点は皮膚表面上の点，MoCap マーカは皮膚上に高さを持つ → 定常オフセットが乗る．
- depth バイアス: PnP（カラーのみ）で求めた外部パラメータは depth の誤差を含まない．骨格は depth 依存なので，表示される `dZ(depth-PnP)` が大きいと検証誤差に系統誤差として乗る．
- 推定精度は PnP の奥行き精度に依存（ボードが小さい・遠いと Z 方向が弱い）．
- RealSense の内部パラメータは工場値を使用．
- 外れ値に弱い最小二乗なので，必要なら姿勢を増やす / RANSAC．
- カメラや MoCap を動かしたらキャリブはやり直し．
