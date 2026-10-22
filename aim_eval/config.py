"""解析の設定（ここだけを編集する）

SESSIONS に解析したいセッションのフォルダを並べる。**ここに含めたものだけが処理される**
（1600 DPI を外したいときはその行を消す / コメントアウトする）。

  dir         … recordings/ 以下のフォルダ（相対でもフルパスでも可）
  participant … 参加者 ID。**同じ人は同じ文字列にする**（省略すると session.json の値）
  dpi         … そのセッションの実測 DPI。記録時に --dpi を付け忘れた場合はここで指定する
                 （session.json の値と違えば counts → cm の換算も自動で直す）
  note        … 任意のメモ
"""

SESSIONS = [
    dict(dir=r"recordings/aim_Yuuki400_20261022_1540461",   participant="Yuuki",  dpi=400),
    dict(dir=r"recordings/aim_Yuuki800_20261022_153317",   participant="Yuuki",  dpi=800),
    dict(dir=r"recordings/aim_Suzuka400_20261022_155937",  participant="Suzuka", dpi=400),
    dict(dir=r"recordings/aim_Suzuka800_20261022_155314",  participant="Suzuka", dpi=800)
]

OUT = "analysis"            # 出力先（参加者ごとのフォルダと between/ がこの下にできる）

# ---- 解析する試行を絞る ----
# BANDS … 使う帯域（課題の速さ）。ここに書いた帯域だけを推定も解析もする。
#     ["slow"]          … slow だけ（既定。一番速く終わる）
#     ["fast"]          … fast だけ
#     ["slow", "fast"]  … 両方（帯域の効果も見られる。推定時間は 2 倍）
#     None              … セッションにある帯域すべて
#   静止試行（trial_00_static_pre）は帯域を持たないので、ここの指定に関わらず
#   INCLUDE_STATIC = True なら必ず処理される（ノイズ床の測定に要る）。
BANDS = ["slow"]

# ---- step1 関節位置の推定（時間がかかるのはここだけ。1 試行 3〜7 分）----
SKIP_DONE = True            # joints/rs_joints.npz がある試行は飛ばす
# 静止試行（trial_00_static_pre）を含めるか。
#   **True を強く推奨**。静止はノイズ床の測定に使い、これが無いと R（時間の交代）の信頼性を
#   判定できなくなる（動いていない成分のノイズが混ざり、R が「同時」側に偏る）。
#   10 秒しかないので stride 2 なら 1 本 30 秒ほどで終わる。
INCLUDE_STATIC = True
VIDEO = "all"               # 確認動画: "all"（全試行）/ "first"（各セッションの最初の本番試行）/ "none"
VIDEO_FPS = 20              # 確認動画の fps（はじめからこの fps で書き出す。ffmpeg は不要）
FORCE_VIDEO = False         # すでに動画がある試行も作り直す

# ---- 推定の並列実行（試行どうしは独立なので結果は変わらない）----
# 1 試行 2790 フレーム × 約 200 ms = 9 分ほどかかるので、並列にすると実時間が縮む。
# JOBS = 0 で自動（CPU コア数 − 1、最大 4）。メモリは 1 本あたり 1〜2 GB 見ておく。
# MediaPipe は 1 本の中でも複数スレッドを使うため、コア数ぶん並べても線形には速くならない。
# 並列にしてかえって遅いときは THREADS_PER_JOB を 2〜4 に絞る。
JOBS = 0
THREADS_PER_JOB = None      # 1 本あたりのスレッド数の上限（None = 制限しない）
VIDEO_JOBS = 0              # 動画の並列数（0 = 1 本ずつ）

# 何フレームに 1 枚だけ MediaPipe に渡すか（None = experiment/config.py の JOINT_STRIDE に従う）。
#   2 にすると 30 fps で処理し、時間がちょうど半分になる。
#   課題の最高周波数は 1.08 Hz、解析前に 5 Hz 低域通過をかけるので 30 fps でも情報は落ちない。
#   **Pose モデル（heavy / full）は experiment/config.py の POSE_VARIANT で変える。**
#   どちらも計測系を変えるので、本番の前に 1 試行で compare_settings.py を回して差を見ること。
STRIDE = 2

# ---- step2 解析（全試行で 1 分ほど。何度でも回せる）----
FORCE = False               # すでに eval/ がある試行もやり直す
MIN_D = 1.5                 # 「プレイヤーごとの偏り」と認める判別比の下限
N_PERM = 2000               # 並べ替え検定の回数

# ---- 点検（a0 の結果がこの範囲を外れたら印を付ける）----
GATE = {
    "mouse_gain": (0.7, 1.4),     # 手先（MCP3）/ マウス のゲイン。1 から外れたら DPI の指定ミスを疑う
    "mouse_lag_ms": (-50, 120),   # マウスと手先の遅れ [ms]
    "valid_pct": 70.0,            # 手首（hand）の有効率の下限 [%]
    "lr_swapped_pct": 10.0,       # 左右の取り違えの上限 [%]
    "snr_ok_pct": 50.0,           # SNR ≥ 2 の成分の割合の下限 [%]
}
STRICT = False              # True にすると点検に引っかかった時点で止める