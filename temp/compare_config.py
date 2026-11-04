# -*- coding: utf-8 -*-
"""stroke_compare.py の設定（ここだけを編集する）

  python temp/stroke_compare.py --list                    # 並べたセッションと，選べる試行 (slow_s1 など) を表示
  python temp/stroke_compare.py                           # COMPARES を全部実行
  python temp/stroke_compare.py --compare s1_all          # 名前を指定して 1 つだけ

SESSIONS … 解析の対象にするセッション．**ここに並べたものだけ**が読まれる．
    dir    recordings/ 以下のフォルダ（リポジトリ直下からの相対，またはフルパス）
    person 参加者 ID．同じ人は同じ文字列にする（省略するとフォルダ名 aim_<名前><dpi>_… の名前）
    group  グループ名（例 "FPS" / "non-FPS"）．省略可
    label  このセッションの呼び名．省略するとフォルダ名の aim_ と日時を除いた部分（例 Suguru400）
    dpi    本当の dpi．省略するとフォルダ名の数字（記録時に --dpi を付け忘れた場合の補正）

COMPARES … 比べる組み合わせ．
    name      出力のフォルダ名（temp/out/<name>/）
    sessions  "all" か，label・person・group のリスト（どれに一致してもよい）
    trials    "all" か，帯域_seed のリスト．例 ["slow_s1"]．"trial_01_slow_s1" のフォルダ名でも指定できる
              同じ帯域・同じ seed はターゲットの軌道が同じ（だから「同じ条件」で人を比べられる）
    by        "person"（既定）…人どうしを比べる / "group" … グループの平均も出す
    （axes, size_split, attempt をここに書くと，この比較だけ上書きできる）
"""

SESSIONS = [
    dict(dir="recordings/aim_kakeru400_20261022_123951", person="Kakeru", group="FPS"),
    dict(dir="recordings/aim_Keito400_20261021_170947", person="Keito", group="nonFPS"),
    dict(dir="recordings/aim_Kosuke400_20261021_172948", person="Kosuke", group="nonFPS"),
    dict(dir="recordings/aim_Suguru400_20261021_175327", person="Suguru", group="FPS"),    
    dict(dir="recordings/aim_Syou400_20261022_135724", person="Syou", group="FPS"),
    dict(dir="recordings/aim_xin400_20261022_125957", person="Xin", group="FPS"),
    dict(dir="recordings/aim_Yuuki400_20261022_154046", person="Yuuki", group="nonFPS"),
    dict(dir="recordings/aim_Yuya400_20261022_134407", person="Yuya", group="FPS"),
    dict(dir="recordings/aim_Suzuka400_20261022_155937", person="Suzuka", group="nonFPS"),    
]

COMPARES = [
    dict(name="slow_s1_all", sessions="all", trials=["slow_s1"]),
    dict(name="slow_all",    sessions="all", trials=["slow_s1", "slow_s2", "slow_s3"]),
    dict(name="fps_vs_non",  sessions="all", trials="all", by="group"),
]

# ---- 解析の設定 ----
AXES = 4                # 4 = 横 / 斜め(/) / 縦 / 斜め(\)，3 = 横 / 斜め / 縦（斜めを 1 つにまとめる）
SIZE_SPLIT = "median"   # 小 / 大の境．"median" = 同じ条件(dpi, 帯域)のターゲットのストロークの中央値，数値 = その mm
ATTEMPT = "last"        # やり直した試行 (_2, _3) が複数あるとき．"last" / "first" / "all"
MIN_N_TRIAL = 3         # 1 試行のセルに必要なストローク数（足りなければ空欄）
MIN_N_PERSON = 8        # 人 × 条件のセル（試行をまとめたもの）に必要なストローク数

OUT = "out"             # 出力先．temp/ からの相対（temp/out/<比較の名前>/）

# ---- 関節の推定の扱い（stroke_feasibility.py と同じ意味）----
FC = 3.0                # 関節位置の低域通過 [Hz]
MIN_NOISE = 2.0         # 1 関節・1 軸の位置ノイズ(sd)の下限 [mm]
SNR = 2.0               # 手先の変位がノイズの何倍を超えたら「測れた」とするか