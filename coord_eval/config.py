"""coord_eval の設定（ここだけを編集する）

aim_eval / skill_analysis / analysis_out の解析スクリプト群で重複していた
読み込み・前処理・寄与の分解・統計を 1 本にまとめたもの。
入力は tasks/run_experiment.py が作る recordings/aim_*/ （task.csv, meta.json, joints/rs_joints.npz,
session.json, aim_eval_override.json）を**読むだけ**で、記録フォルダには何も書かない。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                         # リポジトリ（= C:\project\experiment）
RECORDINGS = os.path.join(ROOT, "recordings")
OUT = os.path.join(ROOT, "analysis_out", "coord")    # 出力先（--out で上書き）

# ---- 対象のセッション ------------------------------------------------------------
# None なら recordings/aim_* を自動で探す。参加者名と DPI は次の順で決める:
#   1. ここに書いた値  2. aim_eval_override.json  3. フォルダ名（aim_<名前><DPI>_…）  4. session.json
# 明示したいときは aim_eval/config.py と同じ形で並べる:
#   SESSIONS = [dict(dir=r"recordings/aim_Keito400_20261021_170947", participant="Keito", dpi=400), ...]
SESSIONS = None
DPIS = [400, 800]           # 解析する DPI（1600 は関節推定が済んだら足す）
BANDS = ["slow", "fast"]    # 解析する帯域

# ---- 参加者 ------------------------------------------------------------------------
# 記録フォルダ・override の名前 → 表示名（大文字小文字は無視して照合）
ALIASES = {"kakeru": "Kakeru", "xin": "Xin", "yuya": "Yuya", "yuuya": "Yuya", "syou": "Syou",
           "suguru": "Suguru", "keito": "Keito", "kosuke": "Kosuke", "suzuka": "Suzuka", "yuuki": "Yuuki"}
GROUPS = {
    "FPS": ["Kakeru", "Suguru", "Syou", "Xin", "Yuya"],
    "nonFPS": ["Keito", "Kosuke", "Suzuka", "Yuuki"],
}
GROUP_LABEL = {"FPS": "FPS プレイヤー", "nonFPS": "非 FPS プレイヤー"}

# ---- 前処理（aim_eval / accuracy_eval と同じ値）---------------------------------------
FS_U = 60.0                 # 解析グリッド [Hz]（解析区間 [0, T) をちょうど 1 周期にする）
FC_LP = 5.0                 # 関節・マウスの低域通過 [Hz]（課題の最高周波数は 1.08 Hz）
HAMPEL_K = 4                # Hampel の窓の片側（9 フレーム）
HAMPEL_N = 3.0              # 3·MAD
GAP_MAX_S = 0.2             # これより長い欠損は補間しない
NOISE_HALF = 4              # 入力ビンのノイズを測る近傍ビンの数（片側）
SNR_MIN = 2.0               # 振幅 / ノイズ がこれ未満のビンは「測れていない」

# ---- 寄与の分解 ----------------------------------------------------------------------
# 手先（MCP3）= 肩 + (肘 − 肩) + (手首 − 肘) + (MCP3 − 手首)
PARTS = ["trunk", "upperarm", "forearm", "hand"]
# 手首の点（hand モデル）を使わない 3 成分（手首・MCP3 は MoCap で未照合のため頑健性の確認に使う）
PARTS3 = ["trunk", "upperarm", "distal"]
PROXIMAL = ["trunk", "upperarm"]
DISTAL = ["forearm", "hand"]
DESK_POINT = "palm"         # 机の座標系を決める点（palm = 手首と MCP3 の中点 ≒ マウスのセンサ）

# ---- 1 動作（ストローク）の切り出し ---------------------------------------------------
# マウス（真値。カメラに依らない）の速さの谷で区切る
MOVE_MIN_PROM = 0.25        # 谷の深さ（その試行の速さの中央値に対する比）
MOVE_DUR_S = (0.20, 4.0)    # 1 動作の長さの範囲 [s]
MOVE_MIN_MM = 8.0           # 正味の移動量（マウス）の下限 [mm]
MOVE_MIN_STRAIGHT = 0.70    # 直線性 = 正味の移動量 / 道のり の下限（往復を含む区間を除く）
MOVE_EDGE_N = 2             # 始点・終点の位置を ±n フレームの平均で取る（ノイズを減らす）
MOVE_ACTIVE_Z = 3.0         # 成分が「動いた」とみなす変位 / ノイズ床 の下限
AMP_BINS_MM = [8, 16, 32, 64, 128, 256]   # 振幅の区分（対数等間隔。DPI をまたいで同じ振幅どうしを比べる）
MIN_MOVES_PER_CELL = 5      # 区分ごとの平均を出す最小の動作数

# ---- マウスの向き ---------------------------------------------------------------------
YAW_WIN_S = 1.0             # センサの回転の推定窓 [s]
YAW_STEP_S = 0.25
YAW_MIN_PATH_MM = 15.0      # 窓の中のマウスの道のりの下限 [mm]（短いと角度が決まらない）

# ---- 不変性（DPI・帯域）の判定 ----------------------------------------------------------
# 「変わらない」と言うための同等性の幅（SESOI: これ以内の差は実質的に同じとみなす）。
# 指標の単位で書く。キーは指標名の接頭辞（最長一致）。
SESOI = {
    "gshare_": 0.05, "eshare_": 0.05, "vshare_": 0.05, "mshare_": 0.05, "share3_": 0.05,
    "N_eff": 0.25, "N_move": 0.25, "C_": 0.10,
    "K_orient": 0.15, "mK_orient": 0.15,
    "rslope_": 0.05, "rslope_Nmove": 0.15, "lag_": 30.0, "nondom": 0.05,
    "psi_": 3.0, "psis_": 3.0, "theta_": 3.0, "phi_": 3.0, "psi_pos_r2": 0.10, "psi_sdc_per100": 2.0, "theta_sdc_per100": 2.0, "psis_sd_per100": 2.0,
    "psis_vs_psi_r": 0.15, "psis_vs_psi_slope": 0.15,
    "dofshare_": 0.05, "dof_cancel": 0.10, "phase_": 10.0,
    "rms_err_mm": 1.0, "track_lag_ms": 30.0, "track_gain": 0.05,
}
BIAS_CORRECT = True         # 振幅が半分になることによる SNR の偏りを 400 の擬似 800 で測って引く
# 偏りの補正をかける指標（接頭辞）。大きさの単位を持つ指標（mm, deg）は擬似データで縮めた分が
# そのまま「偏り」に入ってしまうので、比・割合・単位のない指標だけにかける。
# センサの向き（psis_）は擬似データではマウスの座標系の回転を再現できないので除く。
BIAS_METRICS = ["gshare_", "share3_", "eshare", "vshare_", "N_eff", "C_glob", "mshare", "N_move", "C_move",
                "nondom", "n_active", "rslope_", "K_orient", "mK_orient", "psi_sdc_per100", "theta_sdc_per100",
                "psi_pos_r2", "phase_glob", "lag_", "dofshare_", "dof_cancel"]
BIAS_REPS = 2               # 擬似データの繰り返し回数（試行ごと）
N_PERM = 5000               # 並べ替え検定の回数（全組み合わせが少なければ全列挙）
N_BOOT = 2000
ALPHA = 0.05
SEED = 0

# ---- 点検（外れたら印を付ける。除外はしない）------------------------------------------
GATE = {"mouse_gain": (0.5, 1.5), "valid_pct": 70.0, "lr_swapped_pct": 10.0}
EXCLUDE = []                # 除外する試行（"Keito/aim_Keito400_…/trial_04_fast_s1" のように書く）
