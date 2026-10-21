"""トラッキング課題の設定（全員共通にする値はここだけで変える）"""
from pathlib import Path

CFG = {
    # ---- マウス ----
    # 公称 DPI ではなく tracking_task.py --measure-dpi で実測した counts/inch を入れる
    "mouse_dpi": 800.0,

    # ---- 感度 ----
    # "ingame" = FPS と同じ．感度は PC 側のマウスの DPI で変える（プログラムは DPI を使って画面の動きを決めない）．
    #            eDPI = DPI × game_sens．画面上のターゲットの道筋は固定で，DPI を変えると
    #            それを追うのに必要なマウスの移動量が変わる（低 DPI = 大きく動く = 腕，高 DPI = 手首）
    # "fit"    = 動作確認用．マウスの移動量 amp_cm がちょうど画面に収まる倍率にする（DPI を変えても動きは同じ）
    # DPI を変えたら，--dpi に実測値（--measure-dpi）を渡す（cm への換算と eDPI の記録に使う）
    "sens_mode": "ingame",
    "game_sens": 0.4,             # ゲーム内感度（eDPI = DPI × 0.4: 400 → 160, 800 → 320, 1600 → 640）
    "game_yaw_deg": 0.07,         # ゲーム内感度 1・1 カウントあたりの回転 [deg]（VALORANT 0.07，CS2 / Apex 0.022）
    "game_fov_h_deg": 103.0,      # 水平の視野角 [deg]（VALORANT は 103 で固定）

    # ---- 軌道 ----
    "trial_s": 40.0,        # 解析区間の長さ T [s]。周波数はすべて 1/T の整数倍
    "ramp_s": 3.0,          # 解析区間の前の立ち上がり [s]（中央から振幅を 0→1 にする。解析しない）
    # 道筋の最大振幅（中央から片側）[cm]．sens_mode = "fit" のときの実際のマウスの移動量，
    # "ingame" のときは「画面にちょうど収まる大きさ」を決める基準（実際の移動量は DPI で変わる）
    "amp_cm": {"x": 12.0, "y": 6.0},
    # 成分の振幅 ∝ f^(-amp_exponent)。1.0 だと高い周波数が数 mm になり RealSense のノイズに埋もれるので 0.5
    "amp_exponent": 0.5,
    # 帯域ごとの周波数（1/T の何倍か）。x と y で重ならない素数にして、軸ごとに分離できるようにする
    #   T = 40 s のとき slow: 0.05–0.48 Hz, fast: 0.075–1.08 Hz
    "bands": {
        "slow": {"x": [2, 5, 11, 17], "y": [3, 7, 13, 19]},
        "fast": {"x": [3, 7, 13, 19, 29, 41], "y": [5, 11, 17, 23, 31, 43]},
    },
    "seeds_main": [1, 2, 3],         # 本番の軌道（帯域ごと）。slow 3 本 + fast 3 本 = 6 本
    "repeat_first": False,           # True にすると seeds_main[0] を帯域の最後にもう一度やる（再現性の確認．1, 2, 3, 1 の 4 本）
    "seeds_practice": [],            # 練習用（帯域ごと）。空 = 練習なし．[901] なら帯域ごとに 1 本
    # 帯域の順（練習・本番とも）。["slow", "fast"] = 全員が遅い帯域から（慣れてから速い帯域へ）
    #   "alternate" にすると参加者番号の偶奇で slow → fast / fast → slow を入れ替える
    "band_order": ["slow", "fast"],
    # 帯域内の seed の順。"fixed" = 全員同じ順（seeds_main の順に並べ，repeat_first なら最後に seeds_main[0] をもう一度）
    #   "shuffle" = seed 2–5 の順を参加者ごとに入れ替える
    "seed_order": "fixed",

    # ---- 画面 ----
    "margin_px": 60,          # 振幅いっぱいでも画面端からこれだけ空ける
    # ターゲットの半径（amp_cm と同じ「画面に収まる倍率」の cm．画面上の大きさは DPI に依らず固定）
    "target_radius_cm": 0.8,
    "cursor_radius_px": 6,
    # 予告線: これから preview_s 秒のターゲットの道筋を線で見せる（反応時間への依存を減らす）。0 で表示しない
    "preview_s": 0.5,
    "preview_points": 40,
    "preview_width_px": 3,
    "preview_fade": True,                # 先へ行くほど背景に溶かす
    "preview_color": (230, 140, 140),
    "fps_cap": 240,
    "countdown_s": 3.0,
    "bg": (24, 24, 28),
    "target_color": (230, 80, 80),
    "target_on_color": (90, 210, 120),   # カーソルがターゲット内のとき
    "cursor_color": (240, 240, 240),

    # ---- 静止の計測 ----
    # マウスを握ったまま静止する区間 [s]。RealSense の関節の揺れ（ノイズの大きさ）を人ごとに測る
    "static_s": 10.0,
    "static_at": ["start"],       # "start" = 課題の前、"end" = 課題の後（両方なら ["start", "end"]）

    # ---- RealSense（run_experiment.py のとき）----
    # カメラの設定（解像度・fps・露出）は experiment/config.py を使う（較正と同じ設定にするため）
    "rs_record_practice": True,   # 練習も記録するか（1 本あたり数 GB 増える．False で記録しない）
    "rs_tail_s": 0.5,             # 試行の終わりからこれだけ余分に記録してから止める [s]
    "rs_start_timeout_s": 15.0,   # 記録が始まらないときに失敗とみなすまでの時間 [s]

    # ---- 保存先 ----
    # None = リポジトリの recordings/（experiment/ と同じ場所。Git には載らない）
    "out_dir": None,
    "session_prefix": "aim_",     # recordings/aim_<参加者>_<日時>/
}


def out_root(cfg=CFG):
    """保存先．out_dir が None ならリポジトリの recordings/（experiment/ がなければ task/sessions/）"""
    if cfg.get("out_dir"):
        return Path(cfg["out_dir"])
    root = Path(__file__).resolve().parent.parent
    return root / "recordings" if (root / "experiment").is_dir() else Path(__file__).resolve().parent / "sessions"