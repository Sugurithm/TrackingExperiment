"""b プレイヤー間の解析（指標の違いはランダムか、プレイヤーごとの偏りか）

何をするか
  複数の参加者の全試行を集めて、協調の指標（N_eff・R・C）が
  **「試行ごとのばらつき」で説明できるのか、「プレイヤーごとの偏り」なのか**を判定する。

  1. **ICC(1,1)** と判別比 D = SD(人の間) / SD(人の中)
     同じ人・同じ条件の繰り返し（seed 3 本）を「人の中」として使う。
     ICC ≈ 0 … 完全にランダム（人の違いとは言えない）
     ICC 大 … プレイヤーごとの偏りがある
  2. **並べ替え検定**: プレイヤーのラベルを条件内でシャッフルして「人の間の分散」の帰無分布を作り、
     実際の値がその何パーセンタイルに来るかを見る（p 値）。
     「完全にランダムなのか」に対する直接の答え
  3. 参加者ごとのプロファイル（指標・成分シェア・主役の関節）と **(N_eff, R) 平面への配置**
  4. 階層クラスタリング（Ward）… 人数が少ないうちは参考
  5. 指標と成績（RMS 誤差・追従の帯域）の相関。**両方の D が小さい行は読まない**
わかること
  「人によって関節の使い方が違う」と言えるかどうか。言えるとして、どの指標でどのくらい違うか。
"""
import csv, json, os
import numpy as np
import common as C

# 核の指標（これが仕様の問いに直接対応する）
CORE = [("N_eff", "N_eff（関与関節数）"), ("S", "S（同時性）"), ("C", "C（相殺比）"),
        ("dominant_share", "主役のシェア"), ("R", "R（時間の交代・補助）")]
# 補助（分担の内訳・周波数・自由度）
AUX = [("ishare_trunk", "体幹のシェア"), ("ishare_upperarm", "上腕のシェア"),
       ("ishare_forearm", "前腕のシェア"), ("ishare_hand", "手のシェア"),
       ("style_distal_share", "遠位の寄与率"), ("rms_sum_ratio", "打ち消し合いの比"),
       ("fb_shoulder_hz", "肩の帯域 [Hz]"), ("fb_elbow_hz", "肘の帯域 [Hz]"),
       ("fb_wrist_hz", "手首の帯域 [Hz]"), ("pc1", "PC1 の寄与率"),
       ("n_for_90pct", "90 % に必要な成分数"),
       ("ang_elbow_range", "肘の角度の範囲 [deg]"), ("ang_wrist_range", "手首の角度の範囲 [deg]"),
       ("azrange_upperarm", "上腕の方位角 [deg]"), ("azrange_forearm", "前腕の方位角 [deg]"),
       # a8 自由度ごと（肩の 3 自由度を分けたもの）
       ("dof_cancel", "相殺指数（自由度）"),
       ("dofmm_sh_elev", "肩・仰角の寄与 [mm]"), ("dofmm_sh_azim", "肩・方位の寄与 [mm]"),
       ("dofmm_hum_rot", "肩・回旋の寄与 [mm]"), ("dofmm_el_flex", "肘・屈曲の寄与 [mm]"),
       ("dofmm_wr_flex", "手首・屈曲の寄与 [mm]"), ("dofmm_wr_dev", "手首・偏位の寄与 [mm]"),
       ("dofsd_sh_azim", "肩・方位 SD [deg]"), ("dofsd_hum_rot", "肩・回旋 SD [deg]"),
       ("dofsd_el_flex", "肘・屈曲 SD [deg]"), ("dofsd_wr_flex", "手首・屈曲 SD [deg]"),
       ("dofsd_wr_dev", "手首・偏位 SD [deg]"),
       ("dofr_hum_rot_el_flex", "r(肩回旋, 肘屈曲)"),
       ("dofr_sh_elev_sh_azim", "r(肩仰角, 肩方位)")]
PERF = [("rms_2d", "RMS 誤差 [mm]"), ("on_target_pct", "on-target [%]"),
        ("fb_x_hz", "追従の帯域 fb [Hz]"), ("remnant_x", "remnant"), ("delay_x_ms", "実効遅れ [ms]")]
JOINTS = ["shoulder", "elbow", "wrist", "mcp3"]


# ------------------------------------------------------------------ 統計
def _groups(rows, field, cond):
    """条件 cond の中で {参加者: 値の配列}。"""
    g = {}
    for r in rows:
        if tuple(r.get(k) for k in ("dpi", "band")) != cond:
            continue
        v = r.get(field, np.nan)
        if isinstance(v, (int, float)) and np.isfinite(v):
            g.setdefault(r["participant"], []).append(float(v))
    return {p: np.array(v) for p, v in g.items() if len(v) >= 2}


def icc_and_D(rows, field):
    """条件ごとに ICC(1,1) と判別比 D を出し、条件で平均する。"""
    iccs, Ds, ns = [], [], []
    conds = sorted({tuple(r.get(k) for k in ("dpi", "band")) for r in rows},
                   key=lambda c: [str(x) for x in c])
    for c in conds:
        g = _groups(rows, field, c)
        if len(g) < 2:
            continue
        k = np.mean([len(v) for v in g.values()])
        msw = np.mean([v.var(ddof=1) for v in g.values()])
        means = np.array([v.mean() for v in g.values()])
        msb = k * means.var(ddof=1)
        den = msb + (k - 1) * msw
        iccs.append((msb - msw) / den if den > 0 else np.nan)
        Ds.append(means.std(ddof=1) / np.sqrt(msw) if msw > 0 else np.inf)
        ns.append(len(g))
    if not iccs:
        return {"icc": np.nan, "D": np.nan, "n_participants": 0, "n_conditions": 0, "p_perm": np.nan}
    return {"icc": float(np.nanmean(iccs)), "D": float(np.nanmean(Ds)),
            "n_participants": int(max(ns)), "n_conditions": len(iccs)}


def perm_test(rows, field, n_perm=2000, seed=0):
    """プレイヤーのラベルを条件内でシャッフルし、「人の間の分散」の帰無分布と比べる。

    返り値: p 値（帰無分布で実際の値以上になる割合）と、効果の大きさ（実際 / 帰無の中央値）。
    """
    rng = np.random.default_rng(seed)
    conds = sorted({tuple(r.get(k) for k in ("dpi", "band")) for r in rows},
                   key=lambda c: [str(x) for x in c])
    data = []                                  # [(条件, 参加者ラベル配列, 値配列)]
    for c in conds:
        vals, labs = [], []
        for r in rows:
            if tuple(r.get(k) for k in ("dpi", "band")) != c:
                continue
            v = r.get(field, np.nan)
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v)); labs.append(r["participant"])
        if len(set(labs)) >= 2 and len(vals) >= 4:
            data.append((np.array(labs), np.array(vals)))
    if not data:
        return np.nan, np.nan

    def stat(perm=False):
        """全条件で平均した「参加者平均の分散」。"""
        out = []
        for labs, vals in data:
            L = rng.permutation(labs) if perm else labs
            ms = [vals[L == p].mean() for p in set(L)]
            out.append(np.var(ms, ddof=1) if len(ms) > 1 else np.nan)
        return float(np.nanmean(out))

    obs = stat(False)
    null = np.array([stat(True) for _ in range(n_perm)])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    med = float(np.median(null))
    return p, (obs / med if med > 0 else np.nan)


def per_participant(rows, field):
    out = {}
    for r in rows:
        v = r.get(field, np.nan)
        if isinstance(v, (int, float)) and np.isfinite(v):
            out.setdefault(r["participant"], []).append(float(v))
    return {k: float(np.mean(v)) for k, v in out.items()}


def dpi_slope(rows, participant, field):
    x, y = [], []
    for r in rows:
        if r.get("participant") != participant:
            continue
        v = r.get(field, np.nan); d = r.get("dpi", np.nan)
        if isinstance(v, (int, float)) and np.isfinite(v) and v > 0 and np.isfinite(d):
            x.append(np.log(d)); y.append(np.log(v))
    return float(np.polyfit(x, y, 1)[0]) if len(set(x)) >= 2 else np.nan


# ------------------------------------------------------------------ 本体
def run(rows, out, min_D=1.5, n_perm=2000):
    os.makedirs(out, exist_ok=True)
    track = [r for r in rows if r.get("kind") == "track"]
    players = sorted({r["participant"] for r in track})
    if len(players) < 2:
        return ("## b プレイヤー間の解析\n\n"
                f"参加者が {len(players)} 人のため省略（2 人以上で実行してください）。\n")
    dpis = sorted({r["dpi"] for r in track})
    bands = sorted({r["band"] for r in track})
    md = "## b プレイヤー間の解析\n\n"
    md += (f"参加者 {len(players)} 人（{', '.join(map(str, players))}）、"
           f"DPI {', '.join(f'{d:g}' for d in dpis)}、帯域 {', '.join(bands)}、計 {len(track)} 試行。\n\n")

    # --- 1. ランダムか偏りか
    rel = []
    for f, lab in CORE + AUX + PERF:
        s = icc_and_D(track, f)
        if not s["n_conditions"]:
            continue
        p, eff = perm_test(track, f, n_perm=n_perm)
        s.update({"field": f, "label": lab, "p_perm": p, "eff": eff,
                  "core": (f, lab) in CORE,
                  "verdict": ("偏りあり" if (np.isfinite(p) and p < 0.05 and s["D"] >= min_D)
                              else ("判定保留" if np.isfinite(p) and p < 0.05 else "ランダム"))})
        rel.append(s)
    rel.sort(key=lambda r: (not r["core"], -(r["D"] if np.isfinite(r["D"]) else -1)))
    md += "### 1. 指標はランダムか、プレイヤーごとの偏りか\n\n"
    md += C.md_table(rel, [("指標", "label"), ("核", "core"), ("ICC(1,1)", "icc", "{:.2f}"),
                           ("判別比 D", "D", "{:.2f}"), ("並べ替え検定 p", "p_perm", "{:.3f}"),
                           ("人の間の分散 / 帰無", "eff", "{:.1f}"),
                           ("人数", "n_participants", "{:.0f}"), ("判定", "verdict")])
    md += (f"\n判定: **偏りあり** = p < 0.05 かつ D ≥ {min_D}（プレイヤーごとの違いと言える）、"
           "**判定保留** = p < 0.05 だが D が小さい（有意だが繰り返しのばらつきと同程度）、"
           "**ランダム** = 人の違いとして読めない。"
           f"並べ替えは条件内でプレイヤーのラベルを {n_perm} 回入れ替えたもの。\n\n")
    good = [r for r in rel if r["verdict"] == "偏りあり" and r["field"] in dict(CORE + AUX)]

    # --- 2. プロファイル
    prof = []
    for p in players:
        row = {"participant": p}
        for f, lab in CORE + AUX + PERF:
            row[f] = per_participant(track, f).get(p, np.nan)
        dom = [r.get("dominant") for r in track if r["participant"] == p and r.get("dominant")]
        row["dominant"] = max(set(dom), key=dom.count) if dom else "–"
        row["dominant_label"] = C.PARTLABEL.get(row["dominant"], row["dominant"])
        for pt in C.PARTS:
            row[f"slope_{pt}"] = dpi_slope(track, p, f"rms_{pt}")
        row["slope_Neff"] = dpi_slope(track, p, "N_eff")
        prof.append(row)
    md += "### 2. 参加者ごとのプロファイル（全試行の平均）\n\n"
    md += C.md_table(prof, [("参加者", "participant"), ("N_eff", "N_eff", "{:.2f}"),
                            ("S", "S", "{:.2f}"), ("C", "C", "{:.2f}"),
                            ("主役の成分", "dominant_label"), ("主役のシェア", "dominant_share", "{:.0%}"),
                            ("体幹", "ishare_trunk", "{:.0%}"), ("上腕", "ishare_upperarm", "{:.0%}"),
                            ("前腕", "ishare_forearm", "{:.0%}"), ("手", "ishare_hand", "{:.0%}"),
                            ("RMS 誤差 [mm]", "rms_2d", "{:.1f}")])
    md += "\n### 3. センシへの反応の個人差（log(寄与)/log(DPI)。−1 = 相似に縮める）\n\n"
    md += C.md_table(prof, [("参加者", "participant")] +
                     [(C.PARTLABEL[p], f"slope_{p}", "{:+.2f}") for p in C.PARTS] +
                     [("N_eff の傾き", "slope_Neff", "{:+.2f}")])

    # --- 4. スタイル空間
    space = {}
    fields = [r["field"] for r in good] or [f for f, _ in CORE[:3]]
    X = np.array([[p.get(f, np.nan) for f in fields] for p in prof], float)
    keep = np.isfinite(X).all(0) & (X.std(0) > 0)
    X, fields = X[:, keep], [f for f, k in zip(fields, keep) if k]
    if X.shape[1] >= 2 and len(players) >= 3:
        Z = (X - X.mean(0)) / X.std(0)
        U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        ev = S ** 2 / np.sum(S ** 2)
        space = {"fields": fields, "explained": ev[:3].tolist(),
                 "scores": {p: (U * S)[i, :2].tolist() for i, p in enumerate(players)}}
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            L = linkage(Z, method="ward")
            space["cluster"] = {p: int(c) for p, c in
                                zip(players, fcluster(L, 2 if len(players) < 6 else 3, criterion="maxclust"))}
            space["linkage"] = L.tolist()
        except Exception as e:
            space["cluster_error"] = str(e)
        md += "\n### 4. スタイル空間（偏りありの指標のみ）\n\n"
        md += (f"使った指標: {'，'.join(dict(CORE + AUX).get(f, f) for f in fields)}。"
               f"PC1 {ev[0]:.0%}，PC2 {ev[1]:.0%}。")
        if "cluster" in space:
            md += "　群: " + "，".join(f"{p} → {c}" for p, c in space["cluster"].items())
        md += "\n（人数が少ないうちは群より下の散布図を見ること）\n"

    # --- 5. 成績との相関
    cor = []
    Dof = {r["field"]: r["D"] for r in rel}
    for f, lab in CORE + AUX:
        x = np.array([p.get(f, np.nan) for p in prof], float)
        for pf, plab in PERF[:3]:
            y = np.array([p.get(pf, np.nan) for p in prof], float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 3 and x[ok].std() > 0 and y[ok].std() > 0:
                dx, dy = Dof.get(f, np.nan), Dof.get(pf, np.nan)
                cor.append({"style": lab, "perf": plab, "n": int(ok.sum()),
                            "r": float(np.corrcoef(x[ok], y[ok])[0, 1]), "D_style": dx, "D_perf": dy,
                            "ok": bool(np.isfinite(dx) and np.isfinite(dy)
                                       and dx >= min_D and dy >= min_D)})
    if cor:
        md += "\n### 5. 指標と成績の相関（参加者平均どうし）\n\n"
        md += C.md_table(sorted(cor, key=lambda c: -abs(c["r"]))[:12],
                         [("指標", "style"), ("成績", "perf"), ("n", "n", "{:.0f}"), ("r", "r", "{:+.2f}"),
                          ("指標の D", "D_style", "{:.1f}"), ("成績の D", "D_perf", "{:.1f}"),
                          ("読んでよいか", "ok")])
        md += ("\n**両方の D が大きい行だけ読む**こと。成績が人によって変わらない（D が小さい）のに "
               "r が大きく出るのは、人数が少ないときの見かけの相関です。\n")

    # --- 図
    plt = C.setup_plot()
    fig = plt.figure(figsize=(16, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    pc = {p: colors[i % 10] for i, p in enumerate(players)}
    # (N_eff, R) 平面
    a = fig.add_subplot(2, 3, 1)
    for r in track:
        a.scatter(r.get("N_eff", np.nan), r.get("S", np.nan), s=22, alpha=.35,
                  color=pc[r["participant"]])
    for p in prof:
        a.scatter(p["N_eff"], p["S"], s=160, color=pc[p["participant"]], edgecolor="k", zorder=3)
        a.annotate(str(p["participant"]), (p["N_eff"], p["S"]), fontsize=9,
                   xytext=(6, 4), textcoords="offset points")
    a.axhline(1, color="#888", lw=1); a.axvline(1.6, color="#888", ls="--", lw=1)
    a.set_xlabel("N_eff（1 = 一関節依存 → 全体）"); a.set_ylabel("S（<1 = 一つずつ → 1 = 同時）")
    a.set_title("協調の型（小点 = 各試行，大点 = 参加者平均）")
    # 指標ごとの参加者平均
    for k, (f, lab) in enumerate(CORE[:3]):
        a = fig.add_subplot(2, 3, 2 + k)
        for i, p in enumerate(players):
            xs = [r["dpi"] for r in track if r["participant"] == p and np.isfinite(r.get(f, np.nan))]
            ys = [r[f] for r in track if r["participant"] == p and np.isfinite(r.get(f, np.nan))]
            a.plot(xs, ys, "o", ms=5, alpha=.4, color=pc[p])
            m = {}
            for x, y in zip(xs, ys):
                m.setdefault(x, []).append(y)
            a.plot(sorted(m), [np.mean(m[x]) for x in sorted(m)], "-", lw=1.8, color=pc[p], label=str(p))
        a.set_xscale("log"); a.set_xticks(dpis); a.set_xticklabels([f"{d:g}" for d in dpis])
        a.set_xlabel("DPI"); a.set_ylabel(lab); a.set_title(f"{lab} の個人差")
        a.legend(fontsize=7)
    # シェアの積み上げ
    a = fig.add_subplot(2, 3, 5)
    xs = np.arange(len(players)); bottoms = np.zeros(len(players))
    for pt in C.PARTS:
        v = np.array([prof[i].get(f"ishare_{pt}", 0.0) or 0.0 for i in range(len(players))])
        v = np.nan_to_num(v)
        a.bar(xs, v, bottom=bottoms, color=C.PARTCOLOR[pt], label=C.PARTLABEL[pt], width=.6)
        bottoms += v
    a.set_xticks(xs); a.set_xticklabels([str(p) for p in players], fontsize=8)
    a.set_ylabel("寄与シェア"); a.set_title("成分ごとの寄与（参加者平均）"); a.legend(fontsize=7)
    # 樹形図
    a = fig.add_subplot(2, 3, 6)
    if space.get("linkage"):
        from scipy.cluster.hierarchy import dendrogram
        dendrogram(np.array(space["linkage"]), labels=[str(p) for p in players], ax=a,
                   color_threshold=0, above_threshold_color="#666")
        a.set_ylabel("Ward の距離")
    a.set_title("階層クラスタリング")
    fig.suptitle("プレイヤー間の比較", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out, "b_between.png"), dpi=120); plt.close(fig)

    with open(os.path.join(out, "players.csv"), "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for p in prof for k in p})
        w = csv.DictWriter(f, fieldnames=["participant"] + [k for k in keys if k != "participant"])
        w.writeheader(); w.writerows(prof)
    json.dump({"reliability": rel, "profiles": prof, "space": space, "correlations": cor},
              open(os.path.join(out, "b_between.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=C._jsonable)
    return md