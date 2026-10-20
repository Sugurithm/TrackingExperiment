"""トラッキング課題の設定（全員共通にする値はここだけで変える）"""
from pathlib import Path

CFG = {
    # ---- マウス ----
    # 公称 DPI ではなく tracking_task.py --measure-dpi で実測した counts/inch を入れる
    "mouse_dpi": 800.0,

    # ---- 軌道 ----
    "trial_s": 40.0,        # 解析区間の長さ T [s]。周波数はすべて 1/T の整数倍
    "ramp_s": 3.0,          # 解析区間の前の立ち上がり [s]（中央から振幅を 0→1 にする。解析しない）
    # マウスの移動量での最大振幅（中央から片側）[cm]
    "amp_cm": {"x": 12.0, "y": 6.0},
    # 成分の振幅 ∝ f^(-amp_exponent)。1.0 だと高い周波数が数 mm になり RealSense のノイズに埋もれるので 0.5
    "amp_exponent": 0.5,
    # 帯域ごとの周波数（1/T の何倍か）。x と y で重ならない素数にして、軸ごとに分離できるようにする
    #   T = 40 s のとき slow: 0.05–0.48 Hz, fast: 0.075–1.08 Hz
    "bands": {
        "slow": {"x": [2, 5, 11, 17], "y": [3, 7, 13, 19]},
        "fast": {"x": [3, 7, 13, 19, 29, 41], "y": [5, 11, 17, 23, 31, 43]},
    },
    "seeds_main": [1, 2, 3, 4, 5],   # 本番の軌道（帯域ごと）。seeds_main[0] は帯域の最初と最後に 2 回
    "seeds_practice": [901],         # 練習用（帯域ごと 1 本）

    # ---- 画面 ----
    "margin_px": 60,          # 振幅いっぱいでも画面端からこれだけ空ける
    # ターゲットの半径はマウスの移動量 [cm] で決める（画面が変わっても難しさが同じ）
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
    "rs_record_practice": False,  # 練習も記録するか（容量が大きいので既定は記録しない）
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