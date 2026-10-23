"""report.md を作る（数値はすべて CSV と同じもの。解釈の文は機械的に書けるものだけ）"""
import os
import numpy as np

from .. import config as K
from ..lib import stats as S
from . import metrics as MT

G = list(K.GROUPS)
PS = {"trunk": "体幹", "upperarm": "上腕", "forearm": "前腕", "hand": "手"}


def fmt(v, f="{:.2f}"):
    if v is None:
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "○" if v else ""
    if isinstance(v, (int, np.integer)):
        return str(v)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f.format(v) if np.isfinite(v) else "–"


def table(rows, cols):
    """cols = [(見出し, キー or 関数, 書式)]"""
    h = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        cells = []
        for _, k, f in cols:
            v = k(r) if callable(k) else r.get(k)
            cells.append(fmt(v, f) if f else str(v))
        h += "| " + " | ".join(cells) + " |\n"
    return h


def rel(p, out):
    return os.path.relpath(p, out).replace("\\", "/")


def _cond_mean(cell, c, cond=None, group=None):
    v = [r.get(c) for r in cell if (cond is None or r["cond"] == cond) and (group is None or r["group"] == group)]
    v = S.finite(v)
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else np.nan, len(v)) if len(v) else (np.nan, np.nan, 0)


def write(out, sessions, trials, moves, cell, inv, pch, ind, sigm, grp, pm, mamp, frames, floors, figs, files):
    global G
    G = list(K.GROUPS)                       # --groups で差し替えられるので呼ぶたびに読む
    L = []
    w = L.append
    conds = [c for c in ("400_slow", "800_slow", "400_fast", "800_fast") if c in {r["cond"] for r in cell}]
    people = sorted({r["participant"] for r in cell})
    w("# 上肢の関節協調の解析（coord_eval）\n")
    w(f"- 参加者 {len(people)} 人（" + "、".join(
        f"{K.GROUP_LABEL.get(g, g)}: {', '.join(p for p in K.GROUPS[g] if p in people)}" for g in G) + "）")
    w(f"- 本番試行 {len(trials)} 本、1 動作 {sum(1 for m in moves if m.get('valid'))} 個（有効）/ {len(moves)} 個")
    w(f"- 条件: {', '.join(conds)}（DPI_帯域）。各条件 seed 3 本の平均を人の値とする")
    w(f"- 偏りの補正: {'あり（擬似データ ' + str(K.BIAS_REPS) + ' 回）' if K.BIAS_CORRECT else 'なし'}")
    w("- 【参考】= 手首・MCP3（MediaPipe Hand）を使う指標。MoCap でまだ照合できていないので主結論には使わない")
    w("- 方法の詳細と旧プログラムとの対応は `coord_eval/README.md`\n")

    # ------------------------------------------------------------ 0 品質
    w("## 0. データの点検\n")
    flagged = [r for r in trials if r.get("flags")]
    w(f"印の付いた試行 {len(flagged)} 本（除外はしていない。除外するときは config.EXCLUDE）:\n")
    if flagged:
        w(table(flagged, [("参加者", "participant", None), ("DPI", "dpi", "{:.0f}"), ("試行", "trial", None),
                          ("印", "flags", None), ("ゲイン", "mouse_gain", "{:.2f}"), ("手首の有効率 %", "valid_wrist_pct", "{:.0f}"),
                          ("左右の取り違え %", "lr_swapped_pct", "{:.1f}")]))
    w("机の座標系（セッションごと）とノイズ床:\n")
    fl = {(r["participant"], r["dpi"]): r for r in floors}
    w(table(frames, [("参加者", "participant", None), ("DPI", "dpi", "{:.0f}"), ("ゲイン x", "gain_x", "{:.2f}"),
                     ("ゲイン y", "gain_y", "{:.2f}"), ("軸のなす角", "axis_angle_deg", "{:.0f}"), ("残差 mm", "resid_mm", "{:.1f}"),
                     ("ノイズ 上腕 mm", lambda r: fl.get((r["participant"], r["dpi"]), {}).get("upperarm"), "{:.1f}"),
                     ("ノイズ 手先 mm", lambda r: fl.get((r["participant"], r["dpi"]), {}).get("end"), "{:.1f}"),
                     ("ノイズ ψ deg", lambda r: fl.get((r["participant"], r["dpi"]), {}).get("psi"), "{:.1f}")]))
    w("ゲイン = マウス 1 mm あたりカメラの palm が何 mm 動いたか（1 に近いほどよい。軸のなす角は 90° が理想）。\n")

    # ------------------------------------------------------------ H1a
    w("## 1. H1a 全体を通した関節の寄与\n")
    w(f"![]({rel(figs['global'], out)})\n")
    w("**運動量シェア** = 各成分（体幹・上腕・前腕・手）の変位のうち課題の入力周波数に乗ったパワーの割合"
      "（近傍の周波数のノイズを引く。和 = 1）。「全体の関節移動に対してその関節がどれだけ動いたか」。  \n"
      "**手先への寄与** = 各成分の変位をマウスの動きに射影した割合（和 = 1、負 = 手先と逆向きに動いて他の成分を打ち消す）。"
      "マウスはカメラと独立に測っているので、カメラのノイズは平均で消える。\n")
    rows = []
    for g in G + [None]:
        for c in conds:
            r = {"group": K.GROUP_LABEL.get(g, "全員") if g else "全員", "cond": c}
            for col in [f"gshare_{p}" for p in K.PARTS] + [f"eshare_{p}" for p in K.PARTS] + ["N_eff", "N_eff3", "C_glob3"]:
                r[col] = _cond_mean(cell, col, c, g)[0]
            rows.append(r)
    w(table(rows, [("群", "group", None), ("条件", "cond", None)] +
            [(f"運動量 {PS[p]}", f"gshare_{p}", "{:.2f}") for p in K.PARTS] +
            [(f"寄与 {PS[p]}", f"eshare_{p}", "{:.2f}") for p in K.PARTS] +
            [("N_eff", "N_eff", "{:.2f}"), ("N_eff(3)", "N_eff3", "{:.2f}"), ("C(3)", "C_glob3", "{:.2f}")]))
    w("前腕・手の列と N_eff（4 成分）は【参考】。N_eff(3) と C(3) は手首の点を使わない 3 成分（体幹・上腕・肘→MCP3）。\n")
    w("人ごと（全条件の平均）:\n")
    w(table(pm, [("参加者", "participant", None), ("群", "group", None)] +
            [(f"運動量 {PS[p]}", f"gshare_{p}", "{:.2f}") for p in K.PARTS] +
            [("寄与 体幹", "eshare3_trunk", "{:.2f}"), ("寄与 上腕", "eshare3_upperarm", "{:.2f}"),
             ("寄与 肘→MCP3", "eshare3_distal", "{:.2f}"), ("N_eff(3)", "N_eff3", "{:.2f}"), ("C(3)", "C_glob3", "{:.2f}")]))

    # ------------------------------------------------------------ H1b
    w("## 2. H1b 1 動作（A → B）の中での関節の寄与と協調\n")
    w("課題は連続追従なので、マウスの速さの谷で区切った「ほぼまっすぐな 1 回の移動」を 1 動作とした"
      f"（{K.MOVE_DUR_S[0]}–{K.MOVE_DUR_S[1]} s、正味 {K.MOVE_MIN_MM} mm 以上、直線性 {K.MOVE_MIN_STRAIGHT} 以上）。"
      "動作の始点と終点で各成分の変位を測る。\n")
    w("- **1 動作の寄与** = 成分の変位を動作の向きに射影した割合（和 = 1）\n"
      "- **動いた関節の実効数 N_move** = 成分の変位の大きさ（ノイズ床を引く）から 1/Σq²。"
      "1 = 1 つの関節だけで動かした、4 = 全成分が同じだけ動いた\n"
      "- **冗長度 C_move** = Σ|成分の変位| / |手先の変位|。1 = 全成分が同じ向きに足し合わさった（最小の動き）、"
      "> 1 = 手先に出ない動き（互いに打ち消す動き）も使った\n"
      "- **振幅依存 rslope** = 動作の振幅が 2 倍になると寄与が何ポイント変わるか。"
      "0 = どの大きさの動作も同じ組み合わせ、+ = 大きい動作ほど近位を足す（小さい動作は遠位だけで済ませる）\n")
    rows = []
    for g in G + [None]:
        for c in conds:
            r = {"group": K.GROUP_LABEL.get(g, "全員") if g else "全員", "cond": c}
            for col in ["n_moves_valid", "move_amp_med_mm", "mshare_prox", "mshare_hand", "N_move", "C_move3",
                        "rslope_prox", "rslope_Nmove", "lag_move_ms"]:
                r[col] = _cond_mean(cell, col, c, g)[0]
            rows.append(r)
    w(table(rows, [("群", "group", None), ("条件", "cond", None), ("動作数/試行", "n_moves_valid", "{:.0f}"),
                   ("振幅 中央値 mm", "move_amp_med_mm", "{:.0f}"), ("近位の寄与", "mshare_prox", "{:.2f}"),
                   ("手の寄与*", "mshare_hand", "{:.2f}"), ("N_move*", "N_move", "{:.2f}"), ("C_move(3)", "C_move3", "{:.2f}"),
                   ("rslope 近位", "rslope_prox", "{:+.3f}"), ("rslope N*", "rslope_Nmove", "{:+.3f}"),
                   ("近位→遠位 ms", "lag_move_ms", "{:+.0f}")]))
    w(f"![]({rel(figs['move_prox'], out)})\n")
    w(f"![]({rel(figs['move_n'], out)})\n")
    w("**同じ振幅どうしで比べた DPI の効果**（振幅の区分ごとに 800 − 400 を出して平均。"
      "DPI を上げると動作が小さくなるので、全体の平均の差には「振幅の効果」が混ざる。区分をそろえると DPI そのものの効果になる）:\n")
    w(table(mamp, [("指標", "label", None), ("帯域", "band", None), ("人数", "n", "{:.0f}"), ("差", "d", "{:+.3f}"),
                   ("95% CI", lambda r: f"[{fmt(r['ci95_lo'], '{:+.3f}')}, {fmt(r['ci95_hi'], '{:+.3f}')}]", None),
                   ("同等の幅", "sesoi", "{:.2f}"), ("p(差)", "p_diff", "{:.3f}"), ("p(同等)", "p_tost", "{:.3f}"),
                   ("判定", "verdict", None)]))

    # ------------------------------------------------------------ H2
    w("## 3. H2 マウスの向き（手首 → MCP3）\n")
    w("机の水平面で、手の向き ψ（手首 → MCP3）、前腕の向き φ（肘 → 手首）、手首の水平回転 θ = ψ − φ を求める。"
      "握りが固ければ ψ の変化 ≒ マウスの回転。\n\n"
      "- **向きの安定化 K** = 1 − dψ/dθ（入力周波数での回帰）。**K ≈ 0**: 手首を回すとマウスも同じだけ回る（向きが変わる）。"
      "**K ≈ 1**: 手首を回しても前腕が逆に回って打ち消し、マウスは同じ向きのまま平行に動く。K < 0: 前腕も同じ向きに回る\n"
      "- **センサの向き** = マウスのセンサが報告する移動の向きと、カメラで測った手の移動の向きのずれ（1 s の窓ごと）。"
      "手のランドマークの取り方に依らない「マウスが実際に何度回ったか」。ψ と一致すれば握りが固い\n"
      "- **位置依存** = 手を左右に 100 mm 動かしたとき向きが何度変わるか（弧を描く動きなら大きい）\n")
    w(f"![]({rel(figs['orient'], out)})\n")
    rows = []
    for r0 in pm:
        rows.append(r0)
    w(table(rows, [("参加者", "participant", None), ("群", "group", None), ("K*", "K_orient", "{:+.2f}"),
                   ("K 1 動作*", "mK_orient", "{:+.2f}"), ("ψ SD*", "psi_sdc_deg", "{:.1f}"), ("θ SD*", "theta_sdc_deg", "{:.1f}"),
                   ("φ SD", "phi_sdc_deg", "{:.1f}"), ("センサ SD", "psis_sd_deg", "{:.1f}"),
                   ("センサ/100mm", "psis_per100mm_x", "{:+.1f}"), ("r(センサ, ψ)*", "psis_vs_psi_r", "{:+.2f}"),
                   ("傾き(センサ/ψ)*", "psis_vs_psi_slope", "{:.2f}")]))
    w("角度は deg、全条件の平均。* = 手のランドマークを使う【参考】。センサ SD はセンサの向きのノイズ（カメラの速度のノイズ）も含むので、"
      "条件どうし・人どうしの比較に使い、絶対値はノイズ床（静止試行）と比べて読む。\n")

    # ------------------------------------------------------------ H3
    w("## 4. H3 DPI・帯域が変わっても同じ人の協調・寄与は変わらないか\n")
    w("人ごとに差 d（後 − 前、例: 800 DPI − 400 DPI）を出し、**同等性検定（TOST）**で「±幅 の範囲で同じ」と言えるかを調べる"
      "（差が有意でないことは「同じ」の証拠にならないため）。幅（SESOI）は config.SESOI。"
      + ("DPI の比較は、400 の試行を半分に縮めて静止試行のノイズを入れ直した擬似 800 で測った**SNR の偏り**を引いた値。" if K.BIAS_CORRECT else "") +
      "\n\n判定: **同等** = p(同等) < 0.05 / **変化あり** = 95 % CI が 0 を含まず |d| ≥ 幅 / **判定不能** = どちらも言えない（人数・ばらつきが足りない）。"
      "「MDC 超え」= 自分の seed 間のばらつきから見て変わったと言える人数。ρ = 人の順位の一致。\n")
    w(f"![]({rel(figs['forest'], out)})\n")
    rows = [r for r in inv if r["main"]]
    w(table(rows, [("指標", "label", None), ("比較", lambda r: f"{r['factor']} {r['contrast']}（{r['stratum']}）", None),
                   ("前", "mean_a", "{:.3f}"), ("後", "mean_b", "{:.3f}"), ("d", "d", "{:+.3f}"),
                   ("偏り", "bias", "{:+.3f}"),
                   ("95% CI", lambda r: f"[{fmt(r['ci95_lo'], '{:+.3f}')}, {fmt(r['ci95_hi'], '{:+.3f}')}]", None),
                   ("幅", "sesoi", "{:.2f}"), ("p(同等)", "p_tost", "{:.3f}"), ("q(差)", "q_diff", "{:.3f}"),
                   ("MDC 超え", lambda r: f"{r['n_changed_mdc']}/{r['n']}", None), ("ρ", "rho_rank", "{:+.2f}"),
                   ("判定", "verdict", None)]))
    w("偏り = 振幅が縮むことだけで生じる見かけの変化（擬似データ）。d は偏りを引いた後の値。\n")
    # 人ごとの判定（主要な指標 × 比較）
    w("**人ごとの判定**（その人の seed 間のばらつきから。= 同等、↑ / ↓ 変化（幅を超える）、"
      "+ / − 差はあるが幅の内、? 判定不能）:\n")
    mains = [c for c in MT.MAIN if any(r["metric"] == c for r in pch)]
    comps = list(dict.fromkeys((r["factor"], r["stratum"]) for r in pch))
    sym = lambda r: {"同等": "=", "判定不能": "?", "–": "–"}.get(r["verdict"], ("↑" if r["d"] > 0 else "↓")
                                                         if r["verdict"] == "変化" else ("+" if r["d"] > 0 else "−"))
    for fac, st in comps:
        w(f"*{fac}（{st}）*\n")
        hdr = "| 参加者 | 群 | " + " | ".join(MT.label(c, False)[:18] for c in mains) + " |\n|" + "---|" * (len(mains) + 2) + "\n"
        body = ""
        for p in people:
            cells_ = []
            for c in mains:
                r = next((x for x in pch if x["participant"] == p and x["metric"] == c and x["factor"] == fac
                          and x["stratum"] == st), None)
                cells_.append(sym(r) if r else "")
            g = next((x["group"] for x in pch if x["participant"] == p), "")
            body += f"| {p} | {K.GROUP_LABEL.get(g, g)} | " + " | ".join(cells_) + " |\n"
        n_eq = {c: sum(1 for x in pch if x["metric"] == c and x["factor"] == fac and x["stratum"] == st and x["verdict"] == "同等") for c in mains}
        body += "| 同等の人数 | | " + " | ".join(str(n_eq[c]) for c in mains) + " |\n"
        w(hdr + body)
    w(f"![]({rel(figs['lines'], out)})\n")
    w("**個人差の構造は条件をまたいで保たれるか**（分散成分）:\n")
    rows = [r for r in ind if r["main"]]
    w(table(rows, [("指標", "label", None), ("人 %", "pct_person", "{:.0f}"), ("条件 %", "pct_cond", "{:.0f}"),
                   ("人×条件 %", "pct_person_x_cond", "{:.0f}"), ("試行 %", "pct_resid", "{:.0f}"),
                   ("人×条件 / 試行", "pxc_over_resid", "{:.2f}"), ("ICC(3,1) 条件をまたぐ", "icc3_across_cond", "{:.2f}"),
                   ("ICC(1,1) 条件内", "icc1_within_cond", "{:.2f}")]))
    w("「人×条件 / 試行」が 1 前後 = 条件に対する反応の個人差は試行のばらつき以上には無い。"
      "ICC(3,1) が高い = 条件が変わっても人の順序と間隔が保たれる（DPI の主効果は除く）。\n")
    if sigm:
        w("**多変量の照合**（条件 A の各人の指標ベクトルに最も近い条件 B の人が本人か）:\n")
        w(table(sigm, [("条件 A", "cond_a", None), ("条件 B", "cond_b", None),
                       ("当たり", lambda r: f"{r['hits']}/{r['n']}", None), ("偶然の期待値", "chance", "{:.0f}"),
                       ("p（並べ替え）", "p_perm", "{:.4f}")]))
        w(f"使った指標: {sigm[0]['metrics']}\n")

    # ------------------------------------------------------------ 群
    w("## 5. FPS プレイヤー vs 非 FPS プレイヤー\n")
    w(f"人の平均（全条件）で比較。並べ替え検定（{'・'.join(f'{K.GROUP_LABEL[g]} {len([p for p in K.GROUPS[g] if p in people])} 人' for g in G)}、全組み合わせ）、"
      "Hedges g、主要指標の中で BH 補正した q。少人数なので **探索的**。\n")
    w(f"![]({rel(figs['group'], out)})\n")
    g1, g2 = G[0], G[1]
    rows = [r for r in grp if r["main"] and r["stratum"] == "全条件の平均"]
    w(table(rows, [("指標", "label", None), (K.GROUP_LABEL[g1], f"mean_{g1}", "{:.3f}"), (K.GROUP_LABEL[g2], f"mean_{g2}", "{:.3f}"),
                   ("差", "diff", "{:+.3f}"), ("g", "g", "{:+.2f}"), ("p", "p_perm", "{:.3f}"), ("q", "q_bh", "{:.3f}")]))
    rows = [r for r in grp if r["main"] and r["stratum"].startswith("DPI の変化")]
    w("**DPI を変えたときの変化量（800 − 400）は群で違うか**（不変性の度合いが群で違うか）:\n")
    w(table(rows, [("指標", "label", None), ("層", "stratum", None), (K.GROUP_LABEL[g1], f"mean_{g1}", "{:+.3f}"),
                   (K.GROUP_LABEL[g2], f"mean_{g2}", "{:+.3f}"), ("g", "g", "{:+.2f}"), ("p", "p_perm", "{:.3f}")]))

    # ------------------------------------------------------------ 補助
    w("## 6. 補助の解析（詳細は CSV）\n")
    w("旧プログラムとの連続性のための指標。すべて `data/person_condition.csv` に条件ごとの値、"
      "`data/invariance.csv` と `data/group_tests.csv` に全指標の検定がある。\n")
    rows = []
    for c in conds:
        r = {"cond": c}
        for col in ["vshare_forearm", "vshare_hand", "gshare_lo_upperarm", "gshare_hi_upperarm", "phase_glob_deg",
                    "dofshare_hum_rot", "dofshare_sh_azim", "dofshare_el_flex", "rms_err_mm", "track_lag_ms"]:
            r[col] = _cond_mean(cell, col, c)[0]
        rows.append(r)
    w(table(rows, [("条件", "cond", None), ("速度シェア 前腕*", "vshare_forearm", "{:.2f}"), ("速度シェア 手*", "vshare_hand", "{:.2f}"),
                   ("上腕 低周波", "gshare_lo_upperarm", "{:.2f}"), ("上腕 高周波", "gshare_hi_upperarm", "{:.2f}"),
                   ("近位→遠位 位相 deg", "phase_glob_deg", "{:+.0f}"), ("肩・回旋*", "dofshare_hum_rot", "{:.2f}"),
                   ("肩・方位", "dofshare_sh_azim", "{:.2f}"), ("肘・屈曲*", "dofshare_el_flex", "{:.2f}"),
                   ("追従誤差 mm", "rms_err_mm", "{:.1f}"), ("追従の遅れ ms", "track_lag_ms", "{:.0f}")]))
    w("\n## 7. 読むときの注意\n")
    w("- 手首・MCP3 は MediaPipe Hand の点で、MoCap での照合は肘・肩より弱い（手首の角度は r = 0.14〜0.54 だった。"
      "`analysis_out/20261022/report.md` §4）。手・前腕の成分、N_eff（4 成分）、手の向き ψ・θ は【参考】。"
      "手首を使わない 3 成分（体幹・上腕・肘→MCP3）と、マウスのセンサの向きを主に読む\n"
      "- 「前腕の回転」は肘→手首ベクトルの動きで、肘の屈伸だけでなく肩の回旋で前腕を振る動きも含む（6 自由度の内訳は補助の表）\n"
      "- 1 動作の切り出しは連続追従の中の区切りなので、ポイントからポイントへの到達運動そのものではない。"
      "折り返し（速さの谷）から次の折り返しまでの 1 ストローク\n"
      "- 同等性の幅（SESOI）は結果を見る前に決めておく値。幅を変えると判定が変わるので、幅は報告に必ず書く\n"
      "- 群の比較は 5 人 vs 4 人。p の下限は 2/126 ≈ 0.016。効果量 g と人ごとの点を併せて読む\n")
    w("## 出力ファイル\n")
    for k, v in files.items():
        w(f"- `{rel(v, out)}`")
    path = os.path.join(out, "report.md")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return path
