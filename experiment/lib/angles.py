"""
angles.py
────────────────────────────────────────────────────────────────────
関節角の定義（RealSense と MoCap で**同じ式**を使うための唯一の実装）

3 次元の点だけから決まる 6 自由度．セグメント座標系（骨に固定した座標系）は使わない．

    肩・仰角   sh_elev   上腕ベクトル（肩→肘）の鉛直からの角．0 = 真下
    肩・方位   sh_azim   上腕ベクトルの水平面内の方位
    肩・回旋   hum_rot   上腕軸まわりの前腕の回転（humeral rotation）
    肘・屈曲   el_flex   上腕と前腕のなす角．0 = まっすぐ
    手首・屈曲 wr_flex   腕平面（上腕 × 前腕）内での手ベクトルの角
    手首・偏位 wr_dev    腕平面から外れる向きの角（撓屈・尺屈に相当）

手ベクトルは「手首 → 手の上の遠位点」．RealSense では MCP3，MoCap では手のマーカー
（RHandOut など）を使う．**点の取り方が違うので絶対値は一致しない**が，
時間変化（SD・可動域・波形）は同じ定義なので比較できる．

前腕の回内外（軸まわりの回旋）は手の点が 1 つだけでは求まらない．
MoCap のボーンの回転が書き出されていれば pronation_from_quat() で求められる．

座標系は呼ぶ側が決める．up に「上向き」の単位ベクトルを渡すこと．
    RealSense の机座標 : up = (0, 0, 1)
    Motive（Y-up）     : up = (0, 1, 0)
────────────────────────────────────────────────────────────────────
"""

import numpy as np

DOF = ["sh_elev", "sh_azim", "hum_rot", "el_flex", "wr_flex", "wr_dev"]
DOF_JA = ["肩・仰角", "肩・方位", "肩・回旋", "肘・屈曲", "手首・屈曲", "手首・偏位"]
WRIST_DOF = ["wr_flex", "wr_dev"]


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def unwrap_deg(a):
    """±180 のとびを連続にする（NaN があるところは切って別々に処理する）."""
    a = np.asarray(a, float)
    out = a.copy()
    ok = np.isfinite(a)
    if ok.sum() < 2:
        return out
    idx = np.flatnonzero(ok)
    # 連続した区間ごとに unwrap（欠損をまたいで位相をつなげない）
    brk = np.flatnonzero(np.diff(idx) > 1)
    for s, e in zip(np.r_[0, brk + 1], np.r_[brk + 1, len(idx)]):
        seg = idx[s:e]
        if len(seg) > 1:
            out[seg] = np.degrees(np.unwrap(np.radians(a[seg])))
    return out


def arm_frame(shoulder, elbow, wrist):
    """上腕 e1・前腕 e2・腕平面の法線 n・平面内で e2 に直交する m を返す."""
    e1 = unit(elbow - shoulder)
    e2 = unit(wrist - elbow)
    n = unit(np.cross(e1, e2))
    m = unit(np.cross(n, e2))
    return e1, e2, n, m


def wrist_angles(shoulder, elbow, wrist, hand, degrees=True):
    """手首の屈曲・偏位 [deg]．hand = 手首より遠位の点（MCP3 / RHandOut など）.

    wr_flex : 腕平面内で，前腕の延長から手ベクトルが何度ずれているか
    wr_dev  : 腕平面からどれだけ外れているか（撓屈・尺屈）
    """
    e1, e2, n, m = arm_frame(shoulder, elbow, wrist)
    e3 = unit(hand - wrist)
    dev = np.arcsin(np.clip(np.sum(e3 * n, axis=-1), -1, 1))
    flex = np.arctan2(np.sum(e3 * m, axis=-1), np.sum(e3 * e2, axis=-1))
    if degrees:
        flex, dev = np.degrees(flex), np.degrees(dev)
        flex = unwrap_deg(flex)
    return flex, dev


def elbow_angle(shoulder, elbow, wrist, degrees=True):
    """肘の屈曲角．0 = まっすぐ伸びている."""
    e1 = unit(elbow - shoulder)
    e2 = unit(wrist - elbow)
    a = np.arccos(np.clip(np.sum(e1 * e2, axis=-1), -1, 1))
    return np.degrees(a) if degrees else a


def dof6(shoulder, elbow, wrist, hand, up=(0.0, 0.0, 1.0)):
    """6 自由度 (N,6) [deg]．hand が None なら手首の 2 列は NaN."""
    up = np.asarray(up, float)
    down = -up
    e1 = unit(elbow - shoulder)
    e2 = unit(wrist - elbow)
    n_f = len(np.atleast_2d(e1))

    sh_elev = np.degrees(np.arccos(np.clip(e1 @ down, -1, 1)))
    # 水平面内の 2 軸（up に直交する任意の正規直交基底）
    ref = np.array([1.0, 0.0, 0.0])
    if abs(ref @ up) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    ax1 = unit(ref - (ref @ up) * up)
    ax2 = np.cross(up, ax1)
    sh_azim = np.degrees(np.arctan2(e1 @ ax2, e1 @ ax1))

    r = unit(np.cross(np.broadcast_to(down, e1.shape), e1))
    s = np.cross(e1, r)
    perp = e2 - np.sum(e2 * e1, axis=-1)[..., None] * e1
    hum_rot = np.degrees(np.arctan2(np.sum(perp * s, axis=-1), np.sum(perp * r, axis=-1)))

    el_flex = elbow_angle(shoulder, elbow, wrist)

    if hand is None:
        wr_flex = np.full(n_f, np.nan)
        wr_dev = np.full(n_f, np.nan)
    else:
        wr_flex, wr_dev = wrist_angles(shoulder, elbow, wrist, hand)

    return np.column_stack([sh_elev, unwrap_deg(sh_azim), unwrap_deg(hum_rot),
                            el_flex, wr_flex, wr_dev])


def from_dof(A, L, up=(0.0, 0.0, 1.0)):
    """6 自由度 → 肩を原点とした手先位置 [mm]（順運動学．dof6 の厳密な逆）."""
    up = np.asarray(up, float)
    down = -up
    ref = np.array([1.0, 0.0, 0.0])
    if abs(ref @ up) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    ax1 = unit(ref - (ref @ up) * up)
    ax2 = np.cross(up, ax1)

    a = np.radians(A)
    se, sa, hr, ef, wf, wd = (a[:, i] for i in range(6))
    e1 = (np.sin(se)[:, None] * (np.cos(sa)[:, None] * ax1 + np.sin(sa)[:, None] * ax2)
          + np.cos(se)[:, None] * down)
    r = unit(np.cross(np.broadcast_to(down, e1.shape), e1))
    s = np.cross(e1, r)
    e2 = unit(np.cos(ef)[:, None] * e1
              + np.sin(ef)[:, None] * (np.cos(hr)[:, None] * r + np.sin(hr)[:, None] * s))
    n = unit(np.cross(e1, e2))
    m = unit(np.cross(n, e2))
    e3 = (np.cos(wd)[:, None] * (np.cos(wf)[:, None] * e2 + np.sin(wf)[:, None] * m)
          + np.sin(wd)[:, None] * n)
    L = np.asarray(L, float)
    if L.ndim == 1:
        L = np.broadcast_to(L, (len(A), 3))
    return L[:, [0]] * e1 + L[:, [1]] * e2 + L[:, [2]] * e3


# ------------------------------------------------------------------ 回内外
def quat_to_mat(q):
    """クォータニオン (N,4) [X, Y, Z, W] → 回転行列 (N,3,3)."""
    q = np.asarray(q, float)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    M = np.empty((len(q), 3, 3))
    M[:, 0, 0] = 1 - 2 * (y * y + z * z); M[:, 0, 1] = 2 * (x * y - z * w); M[:, 0, 2] = 2 * (x * z + y * w)
    M[:, 1, 0] = 2 * (x * y + z * w); M[:, 1, 1] = 1 - 2 * (x * x + z * z); M[:, 1, 2] = 2 * (y * z - x * w)
    M[:, 2, 0] = 2 * (x * z - y * w); M[:, 2, 1] = 2 * (y * z + x * w); M[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return M


def euler_to_mat(e, order="ZXY", degrees=True):
    """オイラー角 (N,3) → 回転行列 (N,3,3)．Motive の既定は ZXY（内因性）."""
    e = np.asarray(e, float)
    if degrees:
        e = np.radians(e)
    axes = {"X": 0, "Y": 1, "Z": 2}

    def R(axis, a):
        c, s = np.cos(a), np.sin(a)
        M = np.zeros((len(a), 3, 3))
        M[:, axis, axis] = 1
        i, j = [k for k in range(3) if k != axis]
        M[:, i, i] = c; M[:, j, j] = c
        M[:, i, j] = -s if (j - i) % 3 == 1 else s
        M[:, j, i] = s if (j - i) % 3 == 1 else -s
        return M

    # Motive の列は書き出し順（order）に並ぶ．内因性なので R = R1 @ R2 @ R3
    M = R(axes[order[0]], e[:, 0])
    for k in (1, 2):
        M = M @ R(axes[order[k]], e[:, k])
    return M


def pronation_from_mat(M_forearm, M_hand, axis=1):
    """前腕と手のボーン回転から回内外角 [deg]．

    相対回転 R = M_forearm^T @ M_hand の「前腕の長軸まわり」の成分を取り出す．
    axis は前腕ボーンのローカル長軸（Motive のスケルトンは Y が骨の向き = 1）．
    手首の屈曲・偏位と直交する成分なので、3 点では測れない自由度がここで得られる．
    """
    R = np.einsum("nji,njk->nik", M_forearm, M_hand)
    i = axis
    j, k = (i + 1) % 3, (i + 2) % 3      # 右手系の巡回順（ここを逆にすると符号が反転する）
    return np.degrees(np.arctan2(R[:, k, j] - R[:, j, k], R[:, j, j] + R[:, k, k]))