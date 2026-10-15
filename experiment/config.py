"""
config.py
────────────────────────────────────────────────────────────────────
experiment/ の全プログラムが共有する設定値．
値を変えるときはここだけを直す（各プログラムに数値を書かない）．
────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

# ── 保存先 ────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = ROOT_DIR / "recordings"
CALIB_PREFIX   = "calib_"   # 較正1回分の親フォルダ  recordings/calib_YYYYMMDD_HHMM/
TRIAL_PREFIX   = "trial_"   # マウス操作1試行のフォルダ recordings/trial_YYYYMMDD_HHMMSS/
DB3_NAME       = "session.db3"
META_NAME      = "meta.json"

# ── RealSense ストリーム ──────────────────────────
# 較正と本番で同じ設定にする（T_MC はこのカラー光学座標系に対する変換）
STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
FPS           = 60

# ── 記録時間 [s] ─────────────────────────────────
BOARD_DURATION = 5.0    # チェッカーボード 1 姿勢
TRIAL_DURATION = 60.0   # マウス操作 1 試行

# ── 接続/切断のエラー処理（scripts/rs_review.py と同じ考え方） ──
RESET_ON_START     = True   # プログラム起動時にハードウェアリセット（前回の異常終了を解消）
FRAME_TIMEOUT_MS   = 1000   # フレーム 1 回の待ち時間
MAX_TIMEOUTS       = 3      # 連続タイムアウトがこの回数に達したら記録失敗とする
RECONNECT_WAIT_SEC = 10     # リセット後，デバイス再認識を待つ最大時間

# ── チェッカーボード ─────────────────────────────
# 印刷が 0.9588 倍に縮小されている（MoCap のマーカー間距離から推定）．
# 定規で 7 マス分を実測したら PRINT_SCALE を置き換える．
PRINT_SCALE    = 0.9588
PATTERN        = (7, 4)                    # 内部コーナー数（X方向, Y方向）
SQUARE         = 25.0 * PRINT_SCALE        # 1 マス [mm]（≒ 23.97）
# 反射マーカー S1〜S3 のボード座標 [mm]（原点 = 内部コーナー#0，設計値 × 印刷倍率）
MARKERS_B = {name: (x * PRINT_SCALE, y * PRINT_SCALE, 0.0)
             for name, (x, y) in {"S1": (-40, -40), "S2": (190, -40), "S3": (-40, 115)}.items()}

# ── calib_rs.py（.db3 → T_CB） ────────────────────
RS_STRIDE     = 2      # n フレームに 1 枚処理
RS_MAX_ERR_PX = 0.5    # 再投影誤差がこれを超えるフレームは捨てる [px]
RS_TRIM_HEAD  = 1.0    # 先頭を捨てる [s]（接続直後は自動露出が安定しない）
RS_TRIM_TAIL  = 0.5    # 末尾を捨てる [s]
RS_MIN_FRAMES = 5      # これ未満しか残らなければその姿勢は失敗

# ── calib_mocap.py（Motive CSV → T_MB） ──────────
MC_TOL     = 3.0    # マーカー間距離の許容 [mm]（S1〜S3 の識別）
MC_WIN     = 0.5    # 静止判定の窓幅 [s]
MC_STILL   = 4.0    # 窓内の振れ幅がこれ未満なら静止 [mm]
MC_MIN_LEN = 5.0    # 静止区間の最短長 [s]
MC_MERGE   = 3.0    # この時間以内の途切れは同じ姿勢として結合 [s]
MC_TRIM    = 0.25   # 静止区間の最初と最後を捨てる [s]
MC_OUTLIER = 3.0    # 中央値からこれ以上離れたフレームを捨てる [mm]
