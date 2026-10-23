"""運動学: 手先の分解（成分）、水平面内の向き（ヨー角）、6 自由度

成分（足し算で厳密。モデルもヤコビアンも使わない）
    手先 MCP3 = 肩 + (肘 − 肩) + (手首 − 肘) + (MCP3 − 手首)
              = trunk（肩の並進＝体幹）+ upperarm（上腕の回転＝肩関節）
                + forearm（前腕の回転＝肘関節 + 肩の回旋）+ hand（手の回転＝手首）
    3 成分版: distal = MCP3 − 肘（手首の点を使わない。手首は MoCap で未照合のため頑健性の確認用）

向き（机の水平面に投影したベクトルの方位。0° = 奥（+y）、+ = 右回り（+x 側））
    psi   手の向き  = 手首 → MCP3 の方位（握りが固ければ ≒ マウスの向き）
    phi   前腕の向き = 肘 → 手首 の方位
    theta 手首の水平面内の回転 = psi − phi（撓屈・尺屈と回内外が混ざった「見かけの」手首の回転）
"""
import importlib.util
import os
import numpy as np

from .. import config as K

_ANG = os.path.join(K.ROOT, "experiment", "lib", "angles.py")
AN = None
if os.path.exists(_ANG):                         # 角度の定義は MoCap 側と同じ 1 本を使う
    _spec = importlib.util.spec_from_file_location("coord_eval_angles", _ANG)
    AN = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(AN)

DOF = ["sh_elev", "sh_azim", "hum_rot", "el_flex", "wr_flex", "wr_dev"]
DOFLABEL = {"sh_elev": "肩・仰角", "sh_azim": "肩・方位", "hum_rot": "肩・回旋", "el_flex": "肘・屈曲",
            "wr_flex": "手首・屈曲", "wr_dev": "手首・偏位"}


def components(J, parts=K.PARTS):
    """机座標の関節 {名前: (N,3)} → {成分: (N,3)} と手先 (N,3)。欠損フレームは全成分 NaN。"""
    S, E, W, M = J["shoulder"], J["elbow"], J["wrist"], J["mcp3"]
    full = {"trunk": S, "upperarm": E - S, "forearm": W - E, "hand": M - W, "distal": M - E}
    out = {p: full[p].copy() for p in parts}
    ok = np.all([np.isfinite(X).all(1) for X in (S, E, W, M)], axis=0)
    for p in out:
        out[p][~ok] = np.nan
    end = M.copy()
    end[~ok] = np.nan
    return out, end, ok


def yaw(V):
    """水平面内の方位 [deg]（0 = +y、+x 側が正）。連続になるように展開する。"""
    a = np.degrees(np.arctan2(V[:, 0], V[:, 1]))
    out = a.copy()
    ok = np.isfinite(a)
    if ok.sum() > 1:
        idx = np.flatnonzero(ok)
        out[idx] = np.degrees(np.unwrap(np.radians(a[idx])))
    return out


def orientation(J):
    """{psi, phi, theta} [deg] の時系列。"""
    psi = yaw(J["mcp3"] - J["wrist"])
    phi = yaw(J["wrist"] - J["elbow"])
    th = psi - phi
    th = th - 360 * np.round(np.nanmedian(th) / 360)
    return {"psi": psi, "phi": phi, "theta": th}


def dof6(J):
    """6 自由度 (N,6) [deg] とセグメント長 (3,) [mm]。angles.py が無ければ None。"""
    if AN is None:
        return None, None
    S, E, W, M = (J[n] for n in ("shoulder", "elbow", "wrist", "mcp3"))
    L = np.array([np.nanmedian(np.linalg.norm(E - S, axis=1)), np.nanmedian(np.linalg.norm(W - E, axis=1)),
                  np.nanmedian(np.linalg.norm(M - W, axis=1))])
    return AN.dof6(S, E, W, M, up=np.array([0.0, 0.0, 1.0])), L


def from_dof(A, L):
    return AN.from_dof(A, L, up=np.array([0.0, 0.0, 1.0]))
