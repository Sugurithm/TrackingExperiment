"""
viz.py
────────────────────────────────────────────────────────────────────
推定した関節を RGB 画像に重ねて描く（rs_joints.py --show と joints_video.py で共用）．

  draw_joints(img, names, uv, has_3d, mc_uv=None, lines=())

  ● 塗りつぶし : 2 次元の検出 + 深度あり（3 次元座標あり）
  ○ 輪郭だけ   : 2 次元では見つかったが深度が取れなかった
  × （緑）     : MoCap の関節を画像に投影した位置（同期後のみ）
────────────────────────────────────────────────────────────────────
"""

import numpy as np
import cv2

RS_COLOR = (0, 200, 255)     # 橙
MC_COLOR = (80, 220, 80)     # 緑
BONES = [("shoulder", "elbow"), ("elbow", "wrist"), ("wrist", "mcp3")]


def _pt(p):
    return int(round(p[0])), int(round(p[1]))


def draw_joints(img, names, uv, has_3d, mc_uv=None, lines=()):
    """img に描いた複製を返す．mc_uv は {関節名: (u, v)}，lines は左上に出す文字列．"""
    vis = img.copy()
    pos = {n: p for n, p in zip(names, uv) if np.isfinite(p[0])}
    for a, b in BONES:
        if a in pos and b in pos:
            cv2.line(vis, _pt(pos[a]), _pt(pos[b]), RS_COLOR, 1, cv2.LINE_AA)
    for n, p, ok in zip(names, uv, has_3d):
        if np.isfinite(p[0]):
            cv2.circle(vis, _pt(p), 5, RS_COLOR, -1 if ok else 1, cv2.LINE_AA)
            cv2.putText(vis, n, (_pt(p)[0] + 7, _pt(p)[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, RS_COLOR, 1, cv2.LINE_AA)
    for n, p in (mc_uv or {}).items():
        if np.isfinite(p[0]):
            cv2.drawMarker(vis, _pt(p), MC_COLOR, cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
    for k, s in enumerate(x for x in lines if x):
        (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (4, 6 + 20 * k), (12 + w, 10 + 20 * k + h + 4), (0, 0, 0), -1)   # 読みやすいよう黒い帯
        cv2.putText(vis, s, (8, 10 + 20 * k + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis
