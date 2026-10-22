"""a8 自由度ごとの分解（肩の 3 自由度を分ける）

何をするか
  a1 / a2 は手先を「肩の並進 + 上腕の回転 + 前腕の回転 + 手の回転」の 4 成分に分けるが、
  **上腕の回転は肩関節の 3 自由度をひとまとめ**にしている。ここではその中身を開く。

  観測できる 6 自由度（机座標の 3 次元座標だけから決まる。セグメント座標系は使わない）:

    肩・仰角   上腕ベクトル（肩→肘）の鉛直からの角
    肩・方位   上腕ベクトルの机 xy 面内の方位
    肩・回旋   上腕軸まわりの前腕の回転（humeral rotation）
    肘・屈曲   上腕と前腕のなす角
    手首・屈曲 腕平面（上腕 × 前腕）内での手（手首→MCP3）の角
    手首・偏位 腕平面から外れる向きの角（撓屈・尺屈に相当）

  1. **関節固定法**: その自由度だけ実測どおり動かし、他をすべて平均姿勢に固定したときの
     手先変位 SD [mm]。順運動学で手先位置を組み直すので「その自由度がマウスを何 mm 動かすか」
     と直接読める。a2 の寄与（mm）と同じ単位なので突き合わせられる。
  2. **自由度どうしの相関**: 6×6 の相関行列。負 = 打ち消し合う（冗長性を使った補償）。
  3. **巡回シフト帰無**: 自由度を関節群ごとに独立にランダムな量だけ巡回シフトして
     帰無分布を作る。各自由度の動きの統計はそのままで**関節間の同期だけ**を壊すので、
     「協調に見えるのは各自由度が勝手に動いた副産物ではないか」を棄却できる。
  4. **相殺指数** = sqrt(自由度単独の手先分散の合計 / 実際の手先分散)。
     1 = 独立、>1 = 互いに打ち消しながら手先を作っている。a2 の rms_sum_ratio の分散版。

わかること
  a1 の N_eff が「いくつの関節が関わるか」なのに対し、ここは**どの自由度を使う癖があるか**。
  肩を外転で振るのか（仰角・方位）、上腕をひねるのか（回旋）は a1/a2 では区別できない。
  静止試行から自由度ごとのノイズ床を測るので、「使っていない」と「測れていない」も分かる。

注意
  前腕の回内外（軸まわりの回旋）は手に 2 点（手首・MCP3）しかないため求まらない。
  手の平面が取れる点（親指 or 小指の MCP）を足すと手首が 2 → 4 自由度になる。
"""
import json
import os
import numpy as np
import common as C

DOF = ["sh_elev", "sh_azim", "hum_rot", "el_flex", "wr_flex", "wr_dev"]
DOFLABEL = {"sh_elev": "肩・仰角", "sh_azim": "肩・方位", "hum_rot": "肩・回旋",
            "el_flex": "肘・屈曲", "wr_flex": "手首・屈曲", "wr_dev": "手首・偏位"}
DOFGROUP = {"sh_elev": "upperarm", "sh_azim": "upperarm", "hum_rot": "upperarm",
            "el_flex": "forearm", "wr_flex": "hand", "wr_dev": "hand"}
GROUPDOF = {"upperarm": [0, 1, 2], "forearm": [3], "hand": [4, 5]}
NEED = ["shoulder", "elbow", "wrist", "mcp3"]
DOWN = np.array([0.0, 0.0, -1.0])      # 机座標は z = 上なので、下向きは −z
N_SURR = 200                            # 巡回シフトの回数
FLOOR_CACHE = "dof_floor.json"


# ------------------------------------------------------------------ 幾何
def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def to_dof(P):
    """机座標の関節位置 {名前: (N,3)} → 関節角 (N,6)[deg] とセグメント長 (3,)[mm]"""
    sh, el, wr, mcp = (P[n] for n in NEED)
    e1 = _unit(el - sh)        # 上腕
    e2 = _unit(wr - el)        # 前腕
    e3 = _unit(mcp - wr)       # 手
    L = np.array([np.nanmedian(np.linalg.norm(el - sh, axis=1)),
                  np.nanmedian(np.linalg.norm(wr - el, axis=1)),
                  np.nanmedian(np.linalg.norm(mcp - wr, axis=1))])
    sh_elev = np.degrees(np.arccos(np.clip(e1 @ DOWN, -1, 1)))
    sh_azim = np.degrees(np.arctan2(e1[:, 1], e1[:, 0]))
    r = _unit(np.cross(np.broadcast_to(DOWN, e1.shape), e1))
    s = np.cross(e1, r)
    perp = e2 - np.sum(e2 * e1, axis=1)[:, None] * e1
    hum_rot = np.degrees(np.arctan2(np.sum(perp * s, axis=1), np.sum(perp * r, axis=1)))
    el_flex = np.degrees(np.arccos(np.clip(np.sum(e1 * e2, axis=1), -1, 1)))
    n = _unit(np.cross(e1, e2))        # 腕平面の法線
    m = _unit(np.cross(n, e2))
    wr_dev = np.degrees(np.arcsin(np.clip(np.sum(e3 * n, axis=1), -1, 1)))
    wr_flex = np.degrees(np.arctan2(np.sum(e3 * m, axis=1), np.sum(e3 * e2, axis=1)))
    A = np.column_stack([sh_elev, _unwrap(sh_azim), _unwrap(hum_rot),
                         el_flex, _unwrap(wr_flex), wr_dev])
    return A, L


def _unwrap(a):
    return np.degrees(np.unwrap(np.radians(a)))


def from_dof(A, L):
    """関節角 (N,6)[deg] → 肩を原点とした手先位置 (N,3)[mm]（順運動学）"""
    a = np.radians(A)
    se, sa, hr, ef, wf, wd = (a[:, i] for i in range(6))
    e1 = np.column_stack([np.sin(se) * np.cos(sa), np.sin(se) * np.sin(sa), -np.cos(se)])
    r = _unit(np.cross(np.broadcast_to(DOWN, e1.shape), e1))
    s = np.cross(e1, r)
    e2 = _unit(np.cos(ef)[:, None] * e1
               + np.sin(ef)[:, None] * (np.cos(hr)[:, None] * r + np.sin(hr)[:, None] * s))
    n = _unit(np.cross(e1, e2))
    m = _unit(np.cross(n, e2))
    e3 = (np.cos(wd)[:, None] * (np.cos(wf)[:, None] * e2 + np.sin(wf)[:, None] * m)
          + np.sin(wd)[:, None] * n)
    return L[0] * e1 + L[1] * e2 + L[2] * e3


# ------------------------------------------------------------------ 指標
def freeze(A, L):
    """関節固定法。その自由度（群）だけ動かしたときの手先変位 SD [mm]。"""
    Am = A.mean(0)
    out = {}
    for j, d in enumerate(DOF):
        B = np.broadcast_to(Am, A.shape).copy()
        B[:, j] = A[:, j]
        out[d] = float(np.sqrt(from_dof(B, L).var(0).sum()))
    grp = {}
    for g, idxs in GROUPDOF.items():
        B = np.broadcast_to(Am, A.shape).copy()
        B[:, idxs] = A[:, idxs]
        grp[g] = float(np.sqrt(from_dof(B, L).var(0).sum()))
    full = float(np.sqrt(from_dof(A, L).var(0).sum()))
    tot = sum(v ** 2 for v in grp.values())
    return out, grp, full, (float(np.sqrt(tot / full ** 2)) if full > 0 else np.nan)


def shift_surrogate(A, rng):
    """関節群ごとに独立な巡回シフト（各自由度の統計は保ち、群間の同期だけ壊す）。"""
    B = A.copy()
    n = len(A)
    lo, hi = max(1, n // 10), max(2, n - n // 10)
    for idxs in GROUPDOF.values():
        B[:, idxs] = np.roll(A[:, idxs], int(rng.integers(lo, hi)), axis=0)
    return B


def dof_floor(tr, cache=True):
    """同じセッションの静止試行から、自由度ごとのノイズ床 [deg] を測る。"""
    if not tr.sess_dir:
        return {}
    path = os.path.join(tr.sess_dir, FLOOR_CACHE)
    if cache and os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            pass
    cand = [d for d, m in C.list_trials(tr.sess_dir, kinds=("static",))
            if os.path.exists(os.path.join(d, "joints", "rs_joints.npz"))]
    if not cand:
        return {}
    import argparse as _ap
    ns = _ap.Namespace(trial=cand[0], rs=None, tmc=None, axes=None,
                       out=os.path.join(cand[0], "eval"))
    try:
        st = C.Trial(ns)
        A, L = _trial_dof(st)
    except Exception:
        return {}
    if A is None:
        return {}
    floor = {d: float(A[:, j].std()) for j, d in enumerate(DOF)}
    floor["_endpoint_mm"] = float(np.sqrt(from_dof(A, L).var(0).sum()))
    floor["_trial"] = os.path.basename(cand[0])
    if cache:
        try:
            json.dump(floor, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        except Exception:
            pass
    return floor


def fk_quality(tr, A, L):
    """剛体リンク近似（セグメント長を中央値で固定）がどれだけ手先を再現できるか。

    幾何そのものは厳密（フレームごとの実測長を使えば往復誤差 0）なので、
    ここに出る残差は「セグメント長の揺らぎ」= MediaPipe の関節推定誤差の大きさを表す。
    R² が 0.8 を下回る試行は、寄与の数値を鵜呑みにしないこと。
    """
    G = {n: tr.joint_grid(n, True) for n in NEED}
    ok = np.all([np.isfinite(G[n]).all(1) for n in NEED], axis=0)
    q = (G["mcp3"] - G["shoulder"])[ok]
    e = from_dof(A, L) - q
    e = e - e.mean(0)
    den = (q - q.mean(0)).var(0).sum()
    seg = {}
    for nm, (a, b) in [("upperarm", ("shoulder", "elbow")), ("forearm", ("elbow", "wrist")),
                       ("hand", ("wrist", "mcp3"))]:
        d = np.linalg.norm(G[b][ok] - G[a][ok], axis=1)
        seg[nm] = float(np.std(d))
    return {"fk_r2": float(1 - e.var(0).sum() / den) if den > 0 else np.nan,
            "fk_rms_mm": float(np.sqrt((e ** 2).sum(1).mean())),
            "seg_sd_mm": seg}


def _trial_dof(tr):
    """試行 → 関節角（欠損フレームを除いたもの）とセグメント長。"""
    if not tr.has_rs or any(n not in tr.rs_names for n in NEED):
        return None, None
    G = {n: tr.joint_grid(n, True) for n in NEED}
    ok = np.all([np.isfinite(G[n]).all(1) for n in NEED], axis=0)
    if ok.sum() < 200:
        return None, None
    G = {n: v[ok] for n, v in G.items()}
    return to_dof(G)


# ------------------------------------------------------------------ 本体
def run(tr):
    if tr.kind != "track":
        return ("## a8 自由度ごとの分解\n\n"
                "静止の試行なので省略（この試行は本番試行のノイズ床として使われる）。\n")
    A, L = _trial_dof(tr)
    if A is None:
        return "## a8 自由度ごとの分解\n\n必要な関節（肩・肘・手首・MCP3）が揃っていないため省略。\n"

    only, grp, full, cancel = freeze(A, L)
    fk = fk_quality(tr, A, L)
    Cm = np.corrcoef(A.T)
    floor = dof_floor(tr)
    rng = np.random.default_rng(0)

    # --- 巡回シフト帰無（相関行列と相殺指数）
    pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    null_c = np.zeros((N_SURR, len(pairs)))
    null_cancel = np.zeros(N_SURR)
    for s in range(N_SURR):
        B = shift_surrogate(A, rng)
        Cb = np.corrcoef(B.T)
        null_c[s] = [Cb[i, j] for i, j in pairs]
        null_cancel[s] = freeze(B, L)[3]
    z_cancel = float((cancel - null_cancel.mean()) / max(null_cancel.std(), 1e-12))

    pair_rows = []
    for k, (i, j) in enumerate(pairs):
        mu, sd = null_c[:, k].mean(), null_c[:, k].std()
        z = (Cm[i, j] - mu) / max(sd, 1e-12)
        p = (np.sum(np.abs(null_c[:, k] - mu) >= abs(Cm[i, j] - mu)) + 1) / (N_SURR + 1)
        pair_rows.append({"a": DOF[i], "b": DOF[j], "label": f"{DOFLABEL[DOF[i]]} × {DOFLABEL[DOF[j]]}",
                          "r": float(Cm[i, j]), "null": float(mu), "z": float(z), "p": float(p)})
    pair_rows.sort(key=lambda r: -abs(r["z"]))

    rows = []
    for j, d in enumerate(DOF):
        f = floor.get(d, np.nan)
        rows.append({"dof": d, "label": DOFLABEL[d], "group": DOFGROUP[d],
                     "sd_deg": float(A[:, j].std()),
                     "rom_deg": float(np.percentile(A[:, j], 95) - np.percentile(A[:, j], 5)),
                     "only_mm": only[d], "floor_deg": float(f) if np.isfinite(f) else np.nan,
                     "snr": float(A[:, j].std() / f) if (np.isfinite(f) and f > 0) else np.nan})

    res = {"dof": rows, "group_mm": grp, "endpoint_sd_mm": full,
           "cancel_index": cancel, "cancel_null": float(null_cancel.mean()),
           "cancel_z": z_cancel, "seg_mm": {"upperarm": L[0], "forearm": L[1], "hand": L[2]},
           "corr": Cm.tolist(), "pairs": pair_rows, "n_frames": int(len(A)),
           "floor_trial": floor.get("_trial"), "n_surrogate": N_SURR, **fk}
    tr.save_json("a8_dof.json", res)

    # ---------------------------------------------------------------- 図
    plt = C.setup_plot()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    # (1) 自由度ごとの寄与
    a = axes[0]
    xs = np.arange(6)
    a.bar(xs, [r["only_mm"] for r in rows],
          color=[C.PARTCOLOR[r["group"]] for r in rows], width=.66)
    for i, r in enumerate(rows):
        if np.isfinite(r["snr"]):
            a.text(i, r["only_mm"], f"SNR {r['snr']:.0f}", ha="center", va="bottom", fontsize=7)
    a.set_xticks(xs)
    a.set_xticklabels([r["label"] for r in rows], fontsize=8, rotation=20, ha="right")
    a.set_ylabel("その自由度だけが動いたときの手先変位 SD [mm]")
    a.set_title(f"自由度ごとの寄与（手先合計 {full:.0f} mm）")
    # (2) 相関行列
    a = axes[1]
    M = np.full((5, 5), np.nan)
    sub = Cm[1:, :-1]
    iu = np.tril_indices(5, 0)
    M[iu] = sub[iu]
    im = a.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(5):
        for j in range(5):
            if i >= j:
                a.text(j, i, f"{sub[i, j]:+.2f}", ha="center", va="center", fontsize=7.5,
                       color="white" if abs(sub[i, j]) > .55 else "#222")
    a.set_xticks(range(5)); a.set_yticks(range(5))
    a.set_xticklabels([DOFLABEL[d] for d in DOF[:-1]], fontsize=7.5, rotation=45, ha="right")
    a.set_yticklabels([DOFLABEL[d] for d in DOF[1:]], fontsize=7.5)
    a.set_title("自由度間の相関（青 = 打ち消し合う）")
    fig.colorbar(im, ax=a, fraction=.046)
    # (3) 最も強く結合している対
    a = axes[2]
    top = pair_rows[0]
    i, j = DOF.index(top["a"]), DOF.index(top["b"])
    x = A[:, i] - A[:, i].mean(); y = A[:, j] - A[:, j].mean()
    a.scatter(x, y, s=3, alpha=.25, color="#2a78d6", linewidths=0)
    xr = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 20)
    a.plot(xr, np.polyval(np.polyfit(x, y, 1), xr), color="#eb6834", lw=2)
    a.axhline(0, color="#ccc", lw=.8); a.axvline(0, color="#ccc", lw=.8)
    a.set_xlabel(f"{DOFLABEL[top['a']]}（平均からの差）[deg]")
    a.set_ylabel(f"{DOFLABEL[top['b']]}（平均からの差）[deg]")
    a.set_title(f"最も強い結合 r = {top['r']:+.2f}（帰無 {top['null']:+.2f}，z = {top['z']:+.1f}）"
                f"\n相殺指数 = {cancel:.2f}（帰無 {null_cancel.mean():.2f}，z = {z_cancel:+.1f}）",
                fontsize=9)
    fig.suptitle(f"{tr.name}（{tr.tag()}，DPI {tr.dpi:g}，参加者 {tr.participant}）自由度ごとの分解",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(f"{tr.out}/a8_dof.png", dpi=120); plt.close(fig)

    # ---------------------------------------------------------------- 文章
    md = "## a8 自由度ごとの分解\n\n"
    top_dof = max(rows, key=lambda r: r["only_mm"])
    md += (f"手先変位 SD {full:.1f} mm（{len(A)} フレーム）。"
           f"最も効いている自由度は **{top_dof['label']}（{top_dof['only_mm']:.1f} mm）**。"
           f"関節群では 肩 {grp['upperarm']:.1f} / 肘 {grp['forearm']:.1f} / "
           f"手首 {grp['hand']:.1f} mm。\n\n")
    md += C.md_table(rows, [("自由度", "label"), ("角度 SD [deg]", "sd_deg", "{:.2f}"),
                            ("可動域 [deg]", "rom_deg", "{:.1f}"),
                            ("単独の手先変位 [mm]", "only_mm", "{:.1f}"),
                            ("ノイズ床 [deg]", "floor_deg", "{:.2f}"),
                            ("SNR", "snr", "{:.1f}")])
    md += (f"\n**相殺指数 = {cancel:.2f}**（巡回シフト帰無 {null_cancel.mean():.2f}，z = {z_cancel:+.1f}）。"
           + ("1 を超えており、自由度どうしが**打ち消し合いながら**手先を作っている。\n"
              if cancel > 1.1 and z_cancel > 2 else
              "帰無と区別できず、自由度はほぼ独立に働いている。\n"))
    md += "\n自由度間の結合（巡回シフト帰無との比較。|z| > 2 が有意）\n\n"
    md += C.md_table(pair_rows[:6], [("自由度の対", "label"), ("r", "r", "{:+.2f}"),
                                     ("帰無", "null", "{:+.2f}"), ("z", "z", "{:+.1f}"),
                                     ("p", "p", "{:.3f}")])
    md += (f"\n剛体リンク近似の再現性: R² = {fk['fk_r2']:.3f}、残差 RMS {fk['fk_rms_mm']:.1f} mm"
           f"（セグメント長の SD: 上腕 {fk['seg_sd_mm']['upperarm']:.1f} / "
           f"前腕 {fk['seg_sd_mm']['forearm']:.1f} / 手 {fk['seg_sd_mm']['hand']:.1f} mm）。"
           + ("幾何は厳密なので、この残差はセグメント長の揺らぎ＝関節推定の誤差を表す。\n"
              if fk["fk_r2"] >= 0.80 else
              "**R² が 0.80 を下回るので、この試行の寄与の数値は鵜呑みにしないこと。**\n"))
    if not floor:
        md += ("\n静止試行が見つからないのでノイズ床は出していない。"
               "`INCLUDE_STATIC = True` で静止試行も推定すること。\n")
    else:
        low = [r["label"] for r in rows if np.isfinite(r["snr"]) and r["snr"] < 2]
        if low:
            md += (f"\nSNR が 2 未満の自由度（{'、'.join(low)}）は「使っていない」のではなく"
                   "「測れていない」可能性がある。DPI が高い条件で起きやすい。\n")
    md += ("\n注: 前腕の回内外は手に 2 点しかないため求まらない。"
           "手の平面が取れる点を足すと手首が 2 → 4 自由度になる。\n")
    return md


if __name__ == "__main__":
    tr = C.Trial(C.parser(__doc__).parse_args())
    print(run(tr))