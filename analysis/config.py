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

MODELS_DIR     = ROOT_DIR / "models"      # MediaPipe のモデル（Git には載せない）

# ── RealSense ストリーム ──────────────────────────
# 較正と本番で同じ設定にする（T_MC はこのカラー光学座標系に対する変換）
STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
FPS           = 60
# カラーの露出．None = 自動．数値にすると手動で固定する（単位は RealSense Viewer の Exposure と同じ．
# Viewer でブレが少なく十分明るい値を確かめてから入れる）．同期・較正・本番で同じ値にすること
COLOR_EXPOSURE = None

# ── 記録時間 [s] ─────────────────────────────────
BOARD_DURATION = 5.0    # チェッカーボード 1 姿勢
TRIAL_DURATION = 60.0   # マウス操作 1 試行（rec_trial.py --cue のときは合図の長さで決まる）

# ── 合図音（rec_trial.py --cue）：記録開始と同時に再生し，区間を meta.json に残す ──
#   準備 → ボードを振る → ボードを置く → 静止 → [テンポの読み上げ → 3,2,1 → 拍に合わせて動作 → ストップ]
#   → ボードを振る → 終わり
#   1 拍 = マウスを片方の端からもう片方の端へ（往復で 2 拍）．振り幅は毎回同じにする
CUE_READY_S   = 8.0     # 記録開始から最初のボードまで
CUE_BOARD_S   = 5.0     # ボードを振る（開始・終了とも）
CUE_AWAY_S    = 5.0     # ボードを画角の外に置き，マウスを握る
CUE_STILL_S   = 10.0    # マウスを握って静止（MediaPipe の揺れだけを測る区間）
CUE_COUNT     = 3       # 動作前のカウントダウン（3, 2, 1）
CUE_BEATS     = 16      # 1 つのテンポで打つ拍の数（16 拍 = 8 往復）．ブロックの長さ = 拍数 × 60 / bpm
CUE_REST_S    = 4.0     # テンポの間の休み（静止）．この間に次のテンポを読み上げる
CUE_BPMS      = [20, 40, 60, 80, 100, 120]
CUE_TAIL_S    = 3.0     # 最後のボードの後，記録終了まで
# 音声の案内（Windows の読み上げで wav を作って合図音に混ぜる．False なら音だけ）
CUE_VOICE_ON  = True
CUE_VOICE = {
    "board": "ボードを振ってください",
    "away":  "ボードを置いて、マウスを握ってください",
    "still": "そのまま静止",
    "tempo": "テンポ{bpm}",                 # {bpm} にテンポが入る
    "count": ["さん", "に", "いち"],
    "stop":  "ストップ",
    "end":   "終了です",
}

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

# ── sync_check.py（ボードを振った区間 → 時刻ずれ δ） ──
SYNC_MAX_ERR_PX   = 1.0     # 再投影誤差の上限 [px]（動いているので較正より緩め）
SYNC_SEG_GAP      = 2.0     # ボードの検出がこれ以上途切れたら別区間 [s]
SYNC_SEARCH       = 30.0    # 粗い探索の範囲 ±[s]（2 台の PC の時計のずれ）

# ── rs_joints.py（.db3 → MediaPipe → 関節の 3 次元座標） ──
# Pose のモデル: "heavy"（精度優先．精度検証はこれで行った）/ "full"（約 2 倍速い）/ "lite"（最速）
#   full・lite を使うときは models/ に pose_landmarker_<変種>.task を置く．
#   変えたら experiment/ の比較で heavy との差を確かめること（aim_eval/compare_settings.py）
POSE_VARIANT = "full"
POSE_MODEL = MODELS_DIR / f"pose_landmarker_{POSE_VARIANT}.task"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"

# 何フレームに 1 枚だけ MediaPipe に渡すか（rs_joints.py --stride の既定）．
#   1 = 全フレーム（60 fps）．2 = 30 fps で処理し，時間がちょうど半分になる．
#   課題の最高周波数は 1.08 Hz，解析前に 5 Hz 低域通過をかけるので 30 fps でも情報は落ちない．
JOINT_STRIDE = 1
POSE_CONFIDENCE = 0.5       # 検出・追跡の信頼度（scripts/rs_estimate.py と同じ）
HAND_CONFIDENCE = 0.5
DEPTH_PATCH     = 2         # 関節の画素の周り ±n px の深度の中央値を使う（縁で背景を拾わないように）
DEPTH_RANGE_MM  = (200, 2500)   # これ以外の深度は無効

# RealSense 側の関節: 名前 → (MediaPipe のモデル, ランドマーク番号)
#   pose: 12 右肩, 14 右肘, 16 右手首    hand: 0 手首, 5 人差し指付け根, 9 中指付け根
RS_JOINTS = {
    "shoulder":   ("pose", 12),
    "elbow":      ("pose", 14),
    "wrist":      ("hand", 0),
    "wrist_pose": ("pose", 16),
    "mcp3":       ("hand", 9),
}

# ── mocap_joints.py（Motive CSV → 関節位置） ──────
# MoCap 側の関節: 名前 → CSV の列名（Name）のリスト．複数なら平均（例: 肘 = 内側・外側上顆の中点）．
# ラベル付きマーカー名でもスケルトンのボーン名でもよい（ボーンの位置 = 根元側の関節中心）．
# 実際の列名は  python experiment/mocap_joints.py --list  で確認して書き換える．
# 2026-10-16 の全身スケルトン（Skeleton 001）: ボーンの位置 = 根元側の関節中心
#   RUArm = 肩関節中心，RFArm = 肘関節中心，RHand = 手首関節中心
# "mcp3" は**手首より遠位の点**．これがあると手首の角度が出せて RealSense と比べられる．
#   ボーンマーカー（RHandOut など）でもよい．--list で実際の名前を確かめて書く．
#   見つからなければ MC_HAND_FALLBACKS を順に探し、それでも無ければ手首の角度は NaN になる．
MC_JOINTS = {
    "shoulder": ["RUArm"],
    "elbow":    ["RFArm"],
    "wrist":    ["RHand"],
    "mcp3":     ["RHandOut"],
}
# MC_JOINTS["mcp3"] が見つからないときに順に探す名前
MC_HAND_FALLBACKS = ["RHandOut", "RHand_end", "RHandIn", "RHandThumb", "RIndexFinger1"]

# ── 前腕の回内外（3 点では求まらない自由度） ──────
# ボーンの回転が書き出されていれば、前腕ボーンと手ボーンの相対回転から求められる．
# Motive の書き出しで Rotation にチェックが入っていないと出ない（--list で確認できる）．
MC_FOREARM_BONE   = "RFArm"
MC_HAND_BONE      = "RHand"
MC_BONE_LONG_AXIS = 1        # ボーンのローカル長軸（Motive のスケルトンは Y = 1）

# ── compare_joints.py（RealSense と MoCap の比較） ──
# 比較する組: (RealSense 側の名前, MoCap 側の名前)
COMPARE_PAIRS = [
    ("shoulder", "shoulder"),
    ("elbow", "elbow"),
    ("wrist", "wrist"),
    ("wrist_pose", "wrist"),
    ("mcp3", "mcp3"),          # MoCap 側に手の点があるときだけ比較される（無ければ自動で飛ばす）
]
SEGMENTS       = [("shoulder", "elbow"), ("elbow", "wrist"), ("wrist", "mcp3")]  # 長さが一定のはずの区間
# compare_joints.py で比べる関節角（mc_joints.npz の ang_names と同じ名前）
COMPARE_ANGLES = ["el_flex", "wr_flex", "wr_dev"]
CMP_MAX_GAP    = 0.05       # MoCap の欠損をこれ以上またいで補間しない [s]
CMP_SYNC_MARGIN = 1.0       # 同期でボードを振った区間の前後これだけは比較に使わない [s]
CMP_MAX_LAG    = 0.2        # MediaPipe の遅れを探す範囲 ±[s]
CMP_SPEED_BINS = [0, 100, 300, 600, 1e9]   # 誤差を速さ別に集計する区切り [mm/s]