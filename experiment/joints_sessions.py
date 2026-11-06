# -*- coding: utf-8 -*-
"""60fps_joints.py の設定（関節推定をするセッション）．ここだけを編集する．

  python experiment/60fps_joints.py --plan          # 対象と状態だけ表示
  python experiment/60fps_joints.py --config-only   # SESSIONS に並べたものだけを推定
  python experiment/60fps_joints.py                 # SESSIONS + recordings/aim_* の 400/800 dpi を自動追加

SESSIONS … 推定するセッションのフォルダ（リポジトリ直下からの相対，またはフルパス）．
    文字列をそのまま並べればよい．外したいものは行頭に # を付ける．
    person / group / label で絞り込みたい（--compare を使う）ときだけ dict で書く:
        dict(dir="recordings/aim_xxx", person="Suguru", group="FPS")
    中の試行（trial_*/session.db3 があるもの．静止試行・やり直し _2, _3 も含む）はすべて対象．

COMPARES … --compare NAME で使う絞り込み（省略可）．
    sessions  "all" か，label・person・group のリスト（label の既定はフォルダ名から aim_ と日時を除いたもの，例 Suguru400）
    trials    "all" か，帯域_seed のリスト（例 ["slow_s1"]）
"""

SESSIONS = [
    # ---- 400 dpi ----
    "recordings/aim_kakeru400_20261022_123951",
    "recordings/aim_Keito400_20261021_170947",
    "recordings/aim_Kosuke400_20261021_172948",
    "recordings/aim_Suguru400_20261021_175327",
    "recordings/aim_Suzuka400_20261022_155937",
    "recordings/aim_Syou400_20261022_135724",
    "recordings/aim_xin400_20261022_125957",
    "recordings/aim_Yuuki400_20261022_154046",
    "recordings/aim_Yuya400_20261022_134407",
    # ---- 800 dpi ----
    "recordings/aim_kakeru800_20261022_123338",
    "recordings/aim_Keito800_20261021_171605",
    "recordings/aim_Kosuke800_20261021_173700",
    "recordings/aim_Suguru800_20261021_175931",
    "recordings/aim_Suzuka800_20261022_155314",
    "recordings/aim_Syou800_20261022_135037",
    "recordings/aim_xin800_20261022_125327",
    "recordings/aim_Yuuki800_20261022_153317",
    "recordings/aim_Yuya800_20261022_133738",
]

COMPARES = [
    dict(name="slow_s1_all", sessions="all", trials=["slow_s1"]),
    dict(name="slow_all",    sessions="all", trials=["slow_s1", "slow_s2", "slow_s3"]),
    dict(name="dpi400",      sessions=["kakeru400", "Keito400", "Kosuke400", "Suguru400", "Suzuka400",
                                       "Syou400", "xin400", "Yuuki400", "Yuya400"], trials="all"),
]