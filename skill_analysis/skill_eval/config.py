"""練度別解析の設定（ここだけを編集する）

条件（DPI × 帯域）は CONDITIONS に並べる。run_skill.py は条件ごとに独立して解析し、
results/<条件名>/ に書き出す（条件をまたいだ統合はしない。比較は compare_conditions.py）。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)                        # skill_analysis/
EXP_ROOT = os.path.dirname(PKG_ROOT)                    # experiment/
RECORDINGS = os.path.join(EXP_ROOT, "recordings")
AIM_EVAL = os.path.join(EXP_ROOT, "aim_eval")           # 既存の共通処理（読むだけ。書き換えない）
RESULTS = os.path.join(PKG_ROOT, "results")

# ---- 練度グループ（表示名 → グループ）。順序は 高 → 低
GROUPS = {
    "A": ["Syou", "Yuuya", "Suguru"],     # 高練度
    "B": ["Kakeru", "Xin"],               # 中練度
    "C": ["Kosuke", "Yuuki", "Suzuka"],   # 低練度
}
GROUP_LABEL = {"A": "A 高練度", "B": "B 中練度", "C": "C 低練度"}
GROUP_LEVEL = {"A": 3, "B": 2, "C": 1}  # 順位相関で使う練度の順位（大きいほど高練度）

# 外れ値候補（個別表示・除外の有無で結果がどう変わるかを必ず出す）
OUTLIER_CANDIDATES = ["Syou"]

# 記録フォルダ / aim_eval_override.json の参加者名 → 表示名（大文字小文字は無視して照合）
ALIASES = {"yuya": "Yuuya", "yuuya": "Yuuya", "kakeru": "Kakeru", "xin": "Xin",
           "syou": "Syou", "suguru": "Suguru", "kosuke": "Kosuke", "yuuki": "Yuuki", "suzuka": "Suzuka"}

# ---- 条件（4 条件。dpi は aim_eval_override.json の実測 DPI、band は課題の帯域）
CONDITIONS = [
    dict(name="dpi400_slow", dpi=400, band="slow"),
    dict(name="dpi400_fast", dpi=400, band="fast"),
    dict(name="dpi800_slow", dpi=800, band="slow"),
    dict(name="dpi800_fast", dpi=800, band="fast"),
]

# ---- 解析パラメータ
FS_TASK = 250.0          # 課題ログを載せる一様グリッド [Hz]
FC_TASK = 10.0           # カーソル / ターゲットの低域通過 [Hz]（マウスの量子化を落とす。課題は ≤ 1.08 Hz）
HF_BAND = (2.0, 10.0)    # 課題に無い速い揺れ（微小修正）の帯域 [Hz]
NONIN_MARGIN = 0.5       # 課題外の帯域の下端 = 最高の入力周波数 + これ [Hz]
ONTARGET_MEDFILT_S = 0.02  # on-target の瞬間的な出入りを除く中央値フィルタ [s]
ERR_PEAK_PROM_MM = 1.0   # 修正イベント: 誤差の大きさ |e| の山の突出度の下限 [mm]（画面上）
ERR_PEAK_MIN_S = 0.10    # 修正イベントどうしの最小間隔 [s]
SPEED_PEAK_PROM = 0.10   # サブムーブメント: 速さの山の突出度の下限（ターゲットの RMS 速さに対する比）
REVERSAL_DEADBAND = 0.05 # 方向転換: 速度の符号が変わったとみなす閾値（軸ごとのターゲット RMS 速度に対する比）
JERK_WIN_S = 2.0         # LDLJ を計算する窓 [s]
XCORR_MAX_LAG_S = 0.5    # 関節成分の相互相関で探すラグの範囲 [s]
N_BOOT = 0               # （予約）
