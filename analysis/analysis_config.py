# -*- coding: utf-8 -*-
"""joint_lag_norm.py の設定（ここだけを編集する）

SESSIONS … 解析の対象にするセッション．ここに並べたものだけが読まれる．
    dir    recordings/ 以下のフォルダ（リポジトリ直下からの相対，またはフルパス）
    person 参加者 ID．同じ人は同じ文字列にする
    group  グループ名（"FPS" / "nonFPS"）．省略可
    label  セッションの呼び名．省略するとフォルダ名から作る
    dpi    本当の dpi．省略するとフォルダ名の数字
"""

SESSIONS = [
    dict(dir="recordings/aim_kakeru400_20261022_123951", person="A", group="FPS"),
    dict(dir="recordings/aim_Keito400_20261021_170947", person="B", group="nonFPS"),
    dict(dir="recordings/aim_Kosuke400_20261021_172948", person="C", group="nonFPS"),
    dict(dir="recordings/aim_Suguru400_20261021_175327", person="D", group="FPS"),
    dict(dir="recordings/aim_Syou400_20261022_135724", person="E", group="FPS"),
    dict(dir="recordings/aim_xin400_20261022_125957", person="F", group="FPS"),
    dict(dir="recordings/aim_Yuuki400_20261022_154046", person="G", group="nonFPS"),
    dict(dir="recordings/aim_Yuya400_20261022_134407", person="H", group="FPS"),
    dict(dir="recordings/aim_Suzuka400_20261022_155937", person="I", group="nonFPS"),
]

ATTEMPT = "last"   # やり直した試行 (_2, _3) が複数あるとき．"last" / "first" / "all"
FC = 3.0           # 関節位置の低域通過 [Hz]
OUT = "out"        # 出力先．temp/ からの相対（temp/out/joint_lag_norm/）