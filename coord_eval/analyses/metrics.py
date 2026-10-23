"""指標の一覧（表・検定・図・レポートはすべてここから作る）

(列名, 表示名, 仮説, 主要か, 参考か)
  仮説   H1a = 全体の寄与 / H1b = 1 動作の寄与 / H2 = マウスの向き / X = 補助
  主要   不変性・群の検定でまず読む指標（多重比較の補正はこの集合の中で行う）
  参考   手首・MCP3（hand モデル）を使う指標。MoCap で未照合なので【参考】と付ける
"""
from .. import config as K

P4 = K.PARTS
JA = {"trunk": "体幹", "upperarm": "上腕（肩関節）", "forearm": "前腕（肘関節）", "hand": "手（手首）",
      "distal": "肘→MCP3（前腕+手）", "prox": "近位（体幹+上腕）", "dist": "遠位（前腕+手）"}

M = []


def add(col, label, hyp, main=False, ref=False):
    M.append((col, label, hyp, main, ref))


# ---- H1a 全体の寄与
for p in P4:
    add(f"gshare_{p}", f"全体の運動量シェア: {JA[p]}", "H1a", True, p in ("forearm", "hand"))
add("gshare_prox", "全体の運動量シェア: 近位（体幹+上腕）", "H1a", False, False)
for p in K.PARTS3:
    add(f"share3_{p}", f"全体の運動量シェア（3 成分）: {JA[p]}", "H1a", False, False)
for p in P4:
    add(f"eshare_{p}", f"手先への寄与（マウス基準）: {JA[p]}", "H1a", False, p in ("forearm", "hand"))
for p in K.PARTS3:
    add(f"eshare3_{p}", f"手先への寄与（3 成分）: {JA[p]}", "H1a", False, False)
add("N_eff", "関与関節数 N_eff（4 成分）", "H1a", True, True)
add("N_eff3", "関与関節数 N_eff（3 成分）", "H1a", True, False)
add("C_glob", "打ち消し比 C（全体、4 成分）", "H1a", False, True)
add("C_glob3", "打ち消し比 C（全体、3 成分）", "H1a", True, False)

# ---- H1b 1 動作
for p in P4:
    add(f"mshare_{p}", f"1 動作の寄与: {JA[p]}", "H1b", p in ("trunk", "upperarm"), p in ("forearm", "hand"))
add("mshare_prox", "1 動作の寄与: 近位（体幹+上腕）", "H1b", True, False)
add("mshare3_distal", "1 動作の寄与: 肘→MCP3", "H1b", False, False)
add("N_move", "1 動作で動いた関節の実効数", "H1b", True, True)
add("C_move", "1 動作の冗長度（Σ|成分| / |手先|、4 成分）", "H1b", False, True)
add("C_move3", "1 動作の冗長度（3 成分）", "H1b", True, False)
add("nondom", "主役以外の関節が担った割合", "H1b", False, True)
add("rslope_prox", "近位の寄与の振幅依存（/ 振幅 2 倍）", "H1b", True, False)
add("rslope_hand", "手首の寄与の振幅依存（/ 振幅 2 倍）", "H1b", False, True)
add("rslope_Nmove", "動いた関節数の振幅依存（/ 振幅 2 倍）", "H1b", False, True)
add("lag_move_ms", "1 動作の近位→遠位の速度の山の時間差 [ms]", "H1b", False, False)
for ax in ("x", "y"):
    for p in P4:
        add(f"mshare_{ax}_{p}", f"1 動作の寄与（{'左右' if ax == 'x' else '前後'}の動作）: {JA[p]}", "H1b", False,
            p in ("forearm", "hand"))

# ---- H2 マウスの向き
add("K_orient", "向きの安定化 K（0 = 手首と一緒に回る、1 = 向きを保つ）", "H2", True, True)
add("mK_orient", "向きの安定化 K（1 動作ごと）", "H2", False, True)
add("psi_sdc_deg", "手の向き ψ の SD [deg]（ノイズ床を引く）", "H2", False, True)
add("psi_sdc_per100", "手の向き ψ の SD / マウス 100 mm [deg]", "H2", True, True)
add("theta_sdc_per100", "手首の水平回転 θ の SD / マウス 100 mm [deg]", "H2", False, True)
add("theta_sdc_deg", "手首の水平回転 θ の SD [deg]", "H2", False, True)
add("phi_sdc_deg", "前腕の向き φ の SD [deg]", "H2", False, False)
add("psi_per100mm_x", "手の向きの位置依存（左右 100 mm あたり）[deg]", "H2", False, True)
add("psi_pos_r2", "手の向きを位置で説明できる割合 R²", "H2", False, True)
add("psis_sd_deg", "マウスのセンサの向きの SD [deg]（ランドマーク不要）", "H2", False, False)
add("psis_sd_per100", "センサの向きの SD / マウス 100 mm [deg]", "H2", True, False)
add("psis_per100mm_x", "センサの向きの位置依存（左右 100 mm あたり）[deg]", "H2", False, False)
add("psis_vs_psi_r", "センサの向きと手の向きの相関 r", "H2", False, True)
add("psis_vs_psi_slope", "センサの向き / 手の向き（握りの固さの目安）", "H2", False, True)

# ---- 補助
for p in P4:
    add(f"vshare_{p}", f"速度で重み付けたシェア（旧 a1）: {JA[p]}", "X", False, p in ("forearm", "hand"))
for p in P4:
    add(f"gshare_lo_{p}", f"低い入力周波数でのシェア: {JA[p]}", "X", False, p in ("forearm", "hand"))
    add(f"gshare_hi_{p}", f"高い入力周波数でのシェア: {JA[p]}", "X", False, p in ("forearm", "hand"))
add("phase_glob_deg", "近位→遠位の位相差 [deg]（+ = 遠位が遅れる）", "X", False, False)
add("lag_glob_ms", "近位→遠位の時間差 [ms]（0.25 Hz 以上）", "X", False, False)
for d, lab in (("sh_elev", "肩・仰角"), ("sh_azim", "肩・方位"), ("hum_rot", "肩・回旋"),
               ("el_flex", "肘・屈曲"), ("wr_flex", "手首・屈曲"), ("wr_dev", "手首・偏位")):
    add(f"dofshare_{d}", f"自由度の寄与（関節固定法）: {lab}", "X", False, d.startswith("wr") or d in ("hum_rot", "el_flex"))
add("dof_cancel", "自由度の相殺指数", "X", False, True)
add("rms_err_mm", "追従誤差 RMS [mm]", "X", False, False)
add("track_lag_ms", "追従の遅れ [ms]", "X", False, False)
add("track_gain", "追従のゲイン", "X", False, False)
add("rms_mouse_mm", "マウスの変位 RMS [mm]", "X", False, False)

COLS = [m[0] for m in M]
LABEL = {m[0]: m[1] for m in M}
HYP = {m[0]: m[2] for m in M}
MAIN = [m[0] for m in M if m[3]]
REF = {m[0] for m in M if m[4]}
HYP_LABEL = {"H1a": "H1a 全体を通した関節の寄与", "H1b": "H1b 1 動作の中での関節の寄与・協調",
             "H2": "H2 マウスの向き", "X": "補助"}


def label(col, mark=True):
    s = LABEL.get(col, col)
    return s + ("【参考】" if mark and col in REF else "")


def sesoi(col):
    best, val = "", float("nan")
    for k, v in K.SESOI.items():
        if col.startswith(k) and len(k) > len(best):
            best, val = k, v
    return val
