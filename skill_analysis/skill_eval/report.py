"""条件ごとのレポート（results/<条件>/report.md）。数値はすべて data/ の CSV と同じもの。"""
import datetime, os
import numpy as np

from . import config as K
from .stats import METRICS, MLAB, FAMILY, BETTER
from .coordination import PARTS, PLAB

P_SHOW = 0.10      # 要約に載せる p の目安（探索的）


def _f(v, fmt="{:.2f}"):
    return fmt.format(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else "–"


def _fmt_for(k):
    if k.endswith("_pct") or k in ("delay_ms",) or k.endswith("_ms") or k in ("elbow_range", "wrist_range", "jerk_rms"):
        return "{:.1f}"
    if k.endswith("_mm") or k in ("rms_2d", "rms_x", "rms_y", "hf_jitter"):
        return "{:.2f}"
    return "{:.3f}"


def _dir(rho, k):
    if not np.isfinite(rho):
        return "–"
    s = "高練度ほど大きい" if rho > 0 else "高練度ほど小さい"
    b = BETTER.get(k, 0)
    if b:
        s += "（良い向き）" if np.sign(rho) == b else "（悪い向き）"
    return s


def _rel(p, f):
    return os.path.relpath(p, f).replace("\\", "/")


def _table(rows, cols):
    h = "| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        h += "| " + " | ".join(c[1](r) for c in cols) + " |\n"
    return h


def _group_table(people, keys):
    sets = [("A 高練度", lambda p: p["group"] == "A"),
            ("A（Syou 除外）", lambda p: p["group"] == "A" and p["participant"] not in K.OUTLIER_CANDIDATES),
            ("B 中練度", lambda p: p["group"] == "B"), ("C 低練度", lambda p: p["group"] == "C")]
    rows = []
    for k in keys:
        r = {"label": MLAB[k]}
        for lab, sel in sets:
            v = np.array([p[k] for p in people if sel(p)], float); v = v[np.isfinite(v)]
            fmt = _fmt_for(k)
            r[lab] = (fmt.format(v.mean()) + (" ± " + fmt.format(v.std(ddof=1)) if len(v) > 1 else "")) if len(v) else "–"
        for nm in K.OUTLIER_CANDIDATES:
            v = [p[k] for p in people if p["participant"] == nm]
            r[nm] = _fmt_for(k).format(v[0]) if v and np.isfinite(v[0]) else "–"
        rows.append(r)
    cols = [("指標", lambda r: r["label"])] + [(lab, (lambda l: lambda r: r[l])(lab)) for lab, _ in sets] + \
           [(f"{nm}（個人）", (lambda n: lambda r: r[n])(nm)) for nm in K.OUTLIER_CANDIDATES]
    return _table(rows, cols)


def _test_table(tests, fams):
    rows = [t for t in tests if t["family"] in fams]
    nm = K.OUTLIER_CANDIDATES[0]
    cols = [("指標", lambda t: t["label"]),
            ("ρ 全員", lambda t: _f(t["rho_all"], "{:+.2f}")), ("p", lambda t: _f(t["p_all"])),
            ("q(BH)", lambda t: _f(t.get("q_all", np.nan))),
            ("ρ Syou除外", lambda t: _f(t["rho_noSyou"], "{:+.2f}")), ("p", lambda t: _f(t["p_noSyou"])),
            ("g(A−C) 全員", lambda t: _f(t["g_AC_all"], "{:+.2f}")), ("g(A−C) 除外", lambda t: _f(t["g_AC_noSyou"], "{:+.2f}")),
            ("KW p", lambda t: _f(t["p_kw_all"])), ("判別比 D", lambda t: _f(t["D"], "{:.1f}")),
            (f"{nm} z", lambda t: _f(t.get(f"{nm}_z", np.nan), "{:+.1f}")),
            ("除外で変化", lambda t: "**あり**" if t["flip"] else "")]
    return _table(rows, cols)


def _person_table(people, keys):
    cols = [("参加者", lambda p: (f"**{p['participant']}**" if p["participant"] in K.OUTLIER_CANDIDATES else p["participant"])),
            ("群", lambda p: p["group"])]
    cols += [(MLAB[k], (lambda k_: lambda p: _f(p[k_], _fmt_for(k_)))(k)) for k in keys]
    return _table(people, cols)


def _highlights(tests, fams):
    out = []
    for t in tests:
        if t["family"] not in fams:
            continue
        sig_all = np.isfinite(t["p_all"]) and t["p_all"] < P_SHOW
        sig_ex = np.isfinite(t["p_noSyou"]) and t["p_noSyou"] < P_SHOW
        if not (sig_all or sig_ex):
            continue
        which = "全員でも Syou 除外でも" if sig_all and sig_ex else ("全員でのみ" if sig_all else "Syou 除外でのみ")
        out.append(f"- **{t['label']}**: {_dir(t['rho_all'] if sig_all else t['rho_noSyou'], t['metric'])}。"
                   f"ρ = {_f(t['rho_all'], '{:+.2f}')}（p = {_f(t['p_all'])}）/ Syou 除外 ρ = "
                   f"{_f(t['rho_noSyou'], '{:+.2f}')}（p = {_f(t['p_noSyou'])}）、{which} p < {P_SHOW}。"
                   f"判別比 D = {_f(t['D'], '{:.1f}')}")
    return "\n".join(out) if out else f"- p < {P_SHOW} の指標はありませんでした（{'・'.join(fams)}）。"


def write(out, cond, people, tests, used, missing, fg_track, fg_joint, figs, files, radius):
    nm = K.OUTLIER_CANDIDATES[0]
    n_tr = len(used)
    md = f"# 練度別の追従特性・上肢協調の解析 — {cond['dpi']} DPI / {cond['band'].upper()}\n\n"
    md += f"生成: {datetime.datetime.now():%Y-%m-%d %H:%M}　条件名: `{cond['name']}`　スクリプト: `skill_analysis/run_skill.py`\n\n"
    md += "この条件だけを独立に解析した結果です（他条件のデータは混ぜていません）。条件どうしの比較は `results/_compare/` を見てください。\n\n"
    if os.path.exists(os.path.join(out, "discussion.md")):
        md += "**考察（手書き）: [discussion.md](discussion.md)**（このレポートは再実行で作り直されますが、考察は上書きされません）\n\n"

    md += "## 0. 対象と読み方\n\n"
    for g, members in K.GROUPS.items():
        have = [p["participant"] for p in people if p["group"] == g]
        md += f"- **{K.GROUP_LABEL[g]}**: {', '.join(members)}（解析できた人: {len(have)}）\n"
    md += f"- 試行: 本番の {cond['band']} 試行 seed 1〜3（各参加者 3 本、計 {n_tr} 本）。参加者ごとに 3 本を平均してから群を比べる\n"
    if missing:
        md += "- 欠測: " + "、".join(f"{m['participant']}（{m['reason']}）" for m in missing) + "\n"
    md += ("- 統計: 練度の順位（A=3 / B=2 / C=1）と参加者平均の Spearman ρ。p はラベルの全並べ替えによる正確な p"
           "（8 人 560 通り / Syou 除外 7 人 210 通り）、q は族（追従精度・滑らかさ・軌道修正・関節の寄与・時間の役割分担）ごとの BH 補正。"
           "g(A−C) は Hedges の g。判別比 D = 参加者間 SD / seed 間 SD（1 付近なら人の違いは試行のばらつきと同程度）\n")
    md += ("- **人数が 3 / 2 / 3 人と少ないため、p は探索的な目安です。** 1 人の違いで順位が入れ替わるので、"
           "必ず「Syou 除外」の列と個人の点（図）を併せて読んでください\n")
    md += ("- 関節は RealSense + MediaPipe の推定値。既存の検証（aim_eval/README）より、手首（hand）・MCP3 を使う成分"
           "（手首のシェアなど）は MoCap で未照合のため【参考】。肘は振幅を 6〜7 % 小さく捉える癖がある\n\n")

    md += "## 1. 要約（p < 0.10 の指標を自動抽出）\n\n"
    md += "### ① ターゲット追従特性\n\n" + _highlights(tests, ("追従精度", "滑らかさ", "軌道修正")) + "\n\n"
    md += "### ② 上肢関節の協調・分担\n\n" + _highlights(tests, ("関節の寄与", "時間の役割分担")) + "\n\n"
    flips = [t for t in tests if t["flip"]]
    md += f"### {nm} の影響\n\n"
    md += (f"- {nm} を除くと ρ の符号が変わる、または |Δρ| ≥ 0.3 の指標: "
           + ("、".join(f"{t['label']}（{_f(t['rho_all'], '{:+.2f}')} → {_f(t['rho_noSyou'], '{:+.2f}')}）" for t in flips) or "なし") + "\n")
    zs = [t for t in tests if np.isfinite(t.get(f"{nm}_z", np.nan)) and abs(t[f"{nm}_z"]) >= 2]
    md += (f"- {nm} が他 7 人から外れている指標（|ロバスト z| ≥ 2）: "
           + ("、".join(f"{t['label']}（z = {t[f'{nm}_z']:+.1f}）" for t in zs) or "なし") + "\n\n")

    # ---- ①
    md += "## 2. ① ターゲット追従特性\n\n"
    md += ("指標の定義: 追従率 VAF = 1 − Σ|誤差|²/Σ|ターゲットの変動|²、線形追従率 = カーソルのパワーのうち入力周波数に乗った割合、"
           "躍度比 = log10(RMS 躍度 カーソル / ターゲット)（0 = ターゲットと同じ滑らかさ）、"
           "修正イベント = 誤差の大きさ |e| の山（ずれが最大になり戻り始めた点。突出度 ≥ "
           f"{K.ERR_PEAK_PROM_MM} mm）。すべて画面上の mm。\n\n")
    tk = [k for k, *_ in METRICS if FAMILY[k] in ("追従精度", "滑らかさ", "軌道修正")]
    md += "### 群ごとの値（参加者平均の平均 ± SD）\n\n" + _group_table(people, tk) + "\n"
    md += "### 練度との関係\n\n" + _test_table(tests, ("追従精度", "滑らかさ", "軌道修正")) + "\n"
    md += "### 参加者ごと（主な指標）\n\n" + _person_table(people, ["rms_2d", "on_target_pct", "vaf_pct", "delay_ms",
                                                         "log_jerk_ratio", "corr_rate", "corr_amp_mm", "exit_rate", "hf_jitter"]) + "\n"
    md += f"![追従特性]({_rel(figs['tracking_metrics'], out)})\n\n"
    md += f"![周波数ごとの追従]({_rel(figs['tracking_by_freq'], out)})\n\n"
    if "example_traces" in figs:
        md += f"![軌跡の例]({_rel(figs['example_traces'], out)})\n\n"
    md += f"![修正の分布]({_rel(figs['correction_amplitude'], out)})\n\n"

    # ---- ②
    md += "## 3. ② 上肢関節の協調・分担（肩 / 肘 / 手首）\n\n"
    md += ("手先（MCP3）の動き = 体幹（肩の並進）+ 肩関節（上腕の回転）+ 肘関節（前腕の回転）+ 手首（手の回転）に足し算で分け、"
           "入力周波数での速度パワーのシェアを寄与とする（近傍ビンのノイズを引く。aim_eval の a1 と同じ定義）。"
           "「体幹（肩の並進）」は肩の点そのものの移動で、体幹の傾き・回旋と肩甲帯（肩をすくめる・前に出す）の両方を含む。\n\n")
    md += "### 3.1 寄与・運動量の分配\n\n"
    ck = [k for k, *_ in METRICS if FAMILY[k] == "関節の寄与"]
    md += _group_table(people, ck) + "\n"
    md += f"![関節の寄与]({_rel(figs['joint_shares'], out)})\n\n"
    md += f"![協調の指標]({_rel(figs['coordination_metrics'], out)})\n\n"
    md += "### 3.2 時間軸の役割分担（周波数による分担・先行・支え合い）\n\n"
    md += ("- **周波数による分担**: 各軸の低い入力周波数 2 つと高い 2 つでのシェアの差（高 − 低）。+ = 速い成分ほどその関節が担う\n"
           "- **先行**: 入力周波数での手先に対する位相の進み [ms]（+ = 手先より先に動く）。成分の振幅で重み付け\n"
           "- **相互相関**: 課題の帯域に絞った成分の速度どうし（x・y の平均）。r < 0 = 逆向きに動いて互いを打ち消す＝支え合い（補償）、"
           "ラグ + = 前の成分が先行\n"
           "- **相殺比 C**: Σ|成分| / |手先|。1 = 素直な足し算、> 1 = 打ち消し合いながら手先を作る\n\n")
    tk2 = [k for k, *_ in METRICS if FAMILY[k] == "時間の役割分担"]
    md += _group_table(people, tk2) + "\n"
    md += f"![周波数ごとの分担]({_rel(figs['share_by_freq'], out)})\n\n"
    md += f"![時間の役割分担]({_rel(figs['timing'], out)})\n\n"
    md += "### 3.3 練度との関係\n\n" + _test_table(tests, ("関節の寄与", "時間の役割分担")) + "\n"
    md += "### 参加者ごと（主な指標）\n\n" + _person_table(people, ["share_trunk", "share_upperarm", "share_forearm", "share_hand",
                                                         "N_eff", "S", "C", "fshift_forearm", "lead_upperarm_ms", "xr_upperarm_forearm"]) + "\n"

    # ---- Syou
    md += f"## 4. {nm} の個別評価（含める / 除く）\n\n"
    md += (f"図: [{nm} を含める / 除くで変わる指標]({_rel(figs['syou_sensitivity'], out)})。"
           f"左 = 練度との ρ（● 全員、○ {nm} 除外）、右 = {nm} の他 7 人に対するロバスト z。\n\n")
    md += f"![Syou の影響]({_rel(figs['syou_sensitivity'], out)})\n\n"
    rows = sorted([t for t in tests if np.isfinite(t.get(f"{nm}_z", np.nan))], key=lambda t: -abs(t[f"{nm}_z"]))[:12]
    md += _table(rows, [("指標", lambda t: t["label"]), (f"{nm}", lambda t: _f(t[f"{nm}_value"], _fmt_for(t["metric"]))),
                        ("A 群平均（除外）", lambda t: _f(t["mean_A_noSyou"], _fmt_for(t["metric"]))),
                        ("B 群平均", lambda t: _f(t["mean_B_all"], _fmt_for(t["metric"]))),
                        ("C 群平均", lambda t: _f(t["mean_C_all"], _fmt_for(t["metric"]))),
                        ("z", lambda t: _f(t[f"{nm}_z"], "{:+.1f}")),
                        ("ρ 全員 → 除外", lambda t: f"{_f(t['rho_all'], '{:+.2f}')} → {_f(t['rho_noSyou'], '{:+.2f}')}")])
    md += "\n（|z| の大きい順に 12 指標）\n\n"

    # ---- 計測
    md += "## 5. 計測の確認（試行ごと）\n\n"
    md += ("手首の 3 次元変位 / マウスの変位 のゲイン（机の座標系を決める回帰の副産物）と、関節が 4 点そろったフレームの割合。"
           "マウスの動きの一部は手首の回転（手首より先）が作るので、手首のゲインは 1 より小さいのが普通"
           "（特に x。既存の a0 でも同程度）。参加者間・seed 間で極端に外れていないかを見る。\n\n")
    md += _table(used, [("参加者", lambda r: r["participant"]), ("群", lambda r: r["group"]), ("試行", lambda r: r["trial"]),
                        ("ゲイン x", lambda r: _f(r["mouse_gain_x"])), ("ゲイン y", lambda r: _f(r["mouse_gain_y"])),
                        ("有効率 [%]", lambda r: _f(r["valid_pct"], "{:.0f}")), ("RS fps", lambda r: _f(r["fs_rs"], "{:.1f}"))])
    md += "\n"

    md += "## 6. 出力ファイル\n\n"
    desc = {"trials.csv": "試行ごとの全指標", "persons.csv": "参加者ごとの平均（`__sd` = seed 間 SD）",
            "group_tests.csv": "指標ごとの群平均・ρ・p・q・g・D・Syou の z（全員 / Syou 除外）",
            "freq_tracking_trials.csv": "入力周波数ごとの追従（ゲイン・遅れ・相対誤差）", "freq_tracking_group.csv": "同・群平均",
            "freq_joint_trials.csv": "入力周波数ごとの関節のシェア・射影・位相", "freq_joint_person.csv": "同・参加者平均",
            "freq_joint_group.csv": "同・群平均", "correction_events.csv": "修正イベント 1 件ずつ",
            "sessions_used.csv": "使った試行フォルダと計測の確認値", "missing.csv": "欠測"}
    for k, p in files.items():
        md += f"- [`data/{k}`]({_rel(p, out)}) — {desc.get(k, '')}\n"
    md += "- `figures/*.png` — 本文の図、`trials/*.json` — 試行ごとの詳細\n\n"
    md += "## 7. 他条件への展開\n\n"
    md += ("```\npython skill_analysis/run_skill.py --cond dpi800_slow   # 1 条件\n"
           "python skill_analysis/run_skill.py --all                # 4 条件（それぞれ独立に results/<条件>/）\n"
           "python skill_analysis/compare_conditions.py             # 条件 × 指標の比較表（results/_compare/）\n```\n")
    p = os.path.join(out, "report.md")
    open(p, "w", encoding="utf-8").write(md)
    return p
