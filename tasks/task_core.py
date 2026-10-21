"""
task_core.py
────────────────────────────────────────────────────────────────────
2D エイムトラッキング課題の本体（画面・試行・保存）．
tracking_task.py（課題だけ）と run_experiment.py（課題 + RealSense の記録）の両方から使う．

1 セッション = 1 人 1 回．保存先（task_config の out_dir．既定はリポジトリの recordings/）:

  recordings/aim_<参加者>_<日時>/
      session.json                 設定・画面・計画・試行の一覧（どの試行が何か，記録の成否）
      summary.csv                  試行ごとの成績・fps・カメラのフレーム数
      trial_00_static_pre/         静止の計測（マウスを握ったまま 10 s）
          task.csv                 課題のログ（1 画面フレーム 1 行，下記）
          meta.json                experiment/ の trial_* と同じ形（rs_joints.py がそのまま読める）+ "task"
          session.db3              RealSense（run_experiment.py のときだけ）
      trial_01_practice_fast_s901/ 練習（既定ではカメラは記録しない）
      trial_03_fast_s1/            本番 …
      trial_05_fast_s3_2/          R でやり直した試行は _2, _3 … が付く（前の試行も消さない）

task.csv の列:
  frame, t（解析区間の先頭 = 0 s．カウントダウン・立ち上がりは負）, epoch（PC の時刻 [s]），
  target_x/y_cm, cursor_x/y_cm（画面中央 = 原点，x 右・y 上 = マウスを奥へ），
  mouse_dx/dy_counts（そのフレームに届いたマウスの生の移動量．y は上が正），n_events, on_target
epoch は time.time() と同じ時計．RealSense のフレーム時刻（global_time_enabled）とそのまま比べられる．
────────────────────────────────────────────────────────────────────
"""

import csv
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pygame

from task_config import CFG, out_root
from targets import build_plan, make_trajectory, target_position

TASK_COLS = ["frame", "t", "epoch", "target_x_cm", "target_y_cm", "cursor_x_cm", "cursor_y_cm",
             "mouse_dx_counts", "mouse_dy_counts", "n_events", "on_target"]

TEXT = {
    "ja": {
        "next": "試行 {i}/{n}（{kind}）   クリックで開始   ESC で中断",
        "practice": "練習", "main": "本番", "static": "静止",
        "howto": "赤い円の中にカーソルを入れ続けてください（入ると緑になります）",
        "howto_static": "マウスを握ったまま，腕も手も動かさずに静止してください",
        "static_now": "静止  あと {s} 秒",
        "result": "ターゲット上 {on:.0f}%   平均のずれ {rms:.1f} cm",
        "result_static": "静止中のマウスの動き {d:.2f} cm",
        "result_keys": "クリック: 次へ   R: この試行をやり直す",
        "done": "終了です。ありがとうございました（クリックで閉じる）",
        "cam_boot": "カメラを準備しています…",
        "cam_wait": "カメラの記録を開始しています…",
        "cam_fail": "カメラの記録を開始できません: {r}",
        "cam_fail_keys": "R: もう一度   S: カメラなしで続ける   ESC: 中断",
        "cam_bad": "カメラ: {msg}（R でやり直すことをおすすめします）",
        "saving": "保存しています…",
        "dpi1": "定規に沿ってマウスを {cm:.0f} cm 右へ動かします。クリックで開始",
        "dpi2": "動かして、止めたらクリック   （x counts = {c}）",
        "dpi3": "{c} counts / {cm:.0f} cm = {dpi:.1f} counts/inch   クリックで再測定、ESC で終了",
    },
    "en": {
        "next": "Trial {i}/{n} ({kind})   click to start   ESC to quit",
        "practice": "practice", "main": "main", "static": "still",
        "howto": "Keep the cursor inside the red circle (it turns green)",
        "howto_static": "Hold the mouse and keep your arm and hand still",
        "static_now": "Hold still  {s} s",
        "result": "on target {on:.0f}%   RMS error {rms:.1f} cm",
        "result_static": "mouse moved {d:.2f} cm while still",
        "result_keys": "click: next   R: redo this trial",
        "done": "Finished. Thank you (click to close)",
        "cam_boot": "Preparing the camera...",
        "cam_wait": "Starting the camera recording...",
        "cam_fail": "Camera recording failed to start: {r}",
        "cam_fail_keys": "R: retry   S: continue without camera   ESC: quit",
        "cam_bad": "Camera: {msg} (redo with R is recommended)",
        "saving": "Saving...",
        "dpi1": "Move the mouse {cm:.0f} cm to the right along a ruler. Click to start",
        "dpi2": "Move, then click   (x counts = {c})",
        "dpi3": "{c} counts / {cm:.0f} cm = {dpi:.1f} counts/inch   click again, ESC to quit",
    },
}


CLICK = "click"  # 左クリックを表す疑似キー


class Quit(Exception):
    pass


def unique_dir(base):
    path, k = base, 2
    while path.exists():
        path = base.with_name(f"{base.name}_{k}")
        k += 1
    path.mkdir(parents=True)
    return path


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def full_plan(participant, cfg=CFG, only=None, no_practice=False, no_static=False):
    """静止 → 練習 → 本番（→ 静止）の順の計画．各要素に kind / name / index を付ける．"""
    items = []
    if not no_static and "start" in cfg["static_at"]:
        items.append({"kind": "static", "name": "static_pre", "practice": False})
    track = build_plan(participant, cfg)
    if only:
        b, s = only.split(":")
        track = [{"band": b, "seed": int(s), "practice": False, "repeat": 0}]
    for p in track:
        if no_practice and p["practice"]:
            continue
        p = {k: v for k, v in p.items() if k != "index"}
        items.append({"kind": "track", **p,
                      "name": f"{'practice_' if p['practice'] else ''}{p['band']}_s{p['seed']}"})
    if not no_static and "end" in cfg["static_at"]:
        items.append({"kind": "static", "name": "static_post", "practice": False})
    for i, it in enumerate(items):
        it["index"] = i
    return items


class App:
    def __init__(self, args, recorder):
        self.args = args
        self.rec = recorder
        self.cfg = CFG
        if args.window:
            self.screen = pygame.display.set_mode((1280, 720))
        else:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("tracking task")
        self.W, self.H = self.screen.get_size()
        self.cx, self.cy = self.W / 2, self.H / 2
        jp = pygame.font.match_font("meiryo,yugothic,yugothicui,msgothic,notosanscjkjp,notosansjp,hiraginosans,ipagothic")
        self.lang = "ja" if jp else "en"
        self.font = pygame.font.Font(jp, 28) if jp else pygame.font.SysFont(None, 34)
        self.T = TEXT[self.lang]
        if getattr(args, "dpi", None):
            self.cfg["mouse_dpi"] = float(args.dpi)      # PC 側で設定した DPI（--dpi）
        self.cm_per_count = 2.54 / self.cfg["mouse_dpi"]
        self.preview_s = self.cfg["preview_s"] if getattr(args, "preview", None) is None else args.preview
        self.setup_scale()
        self.clock = pygame.time.Clock()
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)   # 相対モード（画面端で止まらない．Windows では Raw Input）
        self.sim = None

    # ---------- 感度 ----------
    def setup_scale(self):
        """マウスのカウント → 画面の px の倍率を決める．

        sens_mode = "ingame"（既定．感度は PC 側の DPI で変える）:
            FPS と同じ．1 カウントで game_yaw_deg × game_sens [deg] 回る．
            画面の中央での 1 deg は px_per_deg = (W/2) / tan(FOV/2) × π/180 [px]．
            px_per_count = game_yaw_deg × game_sens × px_per_deg   … DPI に依らない（プログラムは DPI を使わない）
            → PC の DPI を変えると，同じマウスの 1 cm で画面が動く量が変わる（eDPI = DPI × game_sens）．
            ターゲットの道筋は画面上で固定（amp_cm を画面に収めた大きさ）なので，
            必要なマウスの移動量 [cm] が DPI に反比例して変わる．
            task.csv の cm は，宣言した mouse_dpi（--dpi）で換算した「実際のマウスの移動量」．
        sens_mode = "fit"（動作確認用）:
            DPI に依らず，マウスの移動量 amp_cm がちょうど画面に収まる倍率にする（DPI を変えても動きは同じ）．
        """
        cfg = self.cfg
        m = cfg["margin_px"]
        fit = min((self.W / 2 - m) / cfg["amp_cm"]["x"], (self.H / 2 - m) / cfg["amp_cm"]["y"])
        px_per_deg = (self.W / 2) / math.tan(math.radians(cfg["game_fov_h_deg"] / 2)) * math.pi / 180
        yaw, sens = cfg["game_yaw_deg"], cfg["game_sens"]
        dpi = cfg["mouse_dpi"]
        if cfg["sens_mode"] == "ingame":
            self.px_per_cm = yaw * sens * px_per_deg * dpi / 2.54     # 実際のマウス 1 cm あたりの px
        elif cfg["sens_mode"] == "fit":
            self.px_per_cm = fit
        else:
            raise SystemExit(f"[ERROR] sens_mode は 'ingame' か 'fit': {cfg['sens_mode']}")
        # 画面上で固定した道筋（nominal = fit の倍率の cm）を，実際のマウスの cm に直す倍率
        self.traj_scale = fit / self.px_per_cm
        deg_per_cm = self.px_per_cm / px_per_deg
        self.sens = {"mode": cfg["sens_mode"], "mouse_dpi": dpi, "game_sens": sens,
                     "edpi": round(dpi * sens, 1), "px_per_cm": round(self.px_per_cm, 3),
                     "deg_per_cm": round(deg_per_cm, 4), "cm_per_360": round(360 / deg_per_cm, 2),
                     "game_yaw_deg": yaw, "game_fov_h_deg": cfg["game_fov_h_deg"],
                     "traj_scale": round(self.traj_scale, 4),
                     "amp_x_cm": round(cfg["amp_cm"]["x"] * self.traj_scale, 2),
                     "amp_y_cm": round(cfg["amp_cm"]["y"] * self.traj_scale, 2)}
        print(f"[SENS] {self.sens['mode']}: DPI {dpi:g} × 感度 {sens:g} = eDPI {self.sens['edpi']:g}"
              f"（{self.sens['cm_per_360']:.1f} cm/360，水平 FOV {cfg['game_fov_h_deg']:g}°）"
              f" → 道筋を追うマウスの移動量 x ±{self.sens['amp_x_cm']:g} cm, y ±{self.sens['amp_y_cm']:g} cm"
              f"（画面 {self.W}×{self.H}）")

    def target_cm(self, traj, t):
        """時刻 t のターゲット位置を，実際のマウスの cm で返す（道筋は画面上で固定）"""
        x, y = target_position(traj, t)
        return x * self.traj_scale, y * self.traj_scale

    # ---------- 共通 ----------
    def to_px(self, x_cm, y_cm):
        return self.cx + x_cm * self.px_per_cm, self.cy - y_cm * self.px_per_cm

    def text(self, s, y, color=(220, 220, 220)):
        img = self.font.render(s, True, color)
        self.screen.blit(img, img.get_rect(center=(self.cx, y)))

    def lines(self, lines, color=(220, 220, 220)):
        self.screen.fill(self.cfg["bg"])
        for i, s in enumerate(lines):
            c = color if isinstance(s, str) else s[1]
            s = s if isinstance(s, str) else s[0]
            self.text(s, self.cy - 40 * (len(lines) - 1) / 2 + 40 * i, c)
        pygame.display.flip()

    def pump(self):
        """イベントを読んで (dx, dy, n, keys) を返す．dy は上が正"""
        dx = dy = n = 0
        keys = []
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise Quit
            if e.type == pygame.MOUSEMOTION:
                dx += e.rel[0]
                dy -= e.rel[1]
                n += 1
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    keys.append(CLICK)  # 左クリック．待ち画面でだけ使う（試行中は呼び出し側が無視する）
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    raise Quit
                keys.append(e.key)
        return dx, dy, n, keys

    def wait_keys(self, lines, keys=(CLICK,), auto=CLICK):
        """lines を表示して keys のどれか（CLICK = 左クリック）が来るまで待つ．来たものを返す（自動プレイでは auto）．
        直前の試行中のクリックや連打で次の画面が飛ばないよう，入るときにイベントを捨て，0.4 s は入力を受け付けない．"""
        if self.args.simulate:
            return auto
        pygame.event.clear()
        t_in = time.perf_counter()
        while True:
            self.lines(lines)
            self.rec.poll()
            _, _, _, pressed = self.pump()
            if time.perf_counter() - t_in >= 0.4:
                for k in pressed:
                    if k in keys:
                        return k
            self.clock.tick(60)

    # ---------- カメラ ----------
    def boot_camera(self):
        if not self.rec.enabled:
            return
        ok = self.rec.wait_ready(tick=lambda: (self.lines([self.T["cam_boot"]]), self.pump(), self.clock.tick(30)))
        if not ok:
            raise SystemExit(f"[ERROR] RealSense の記録プロセスが起動しません: {self.rec.error or 'タイムアウト'}")
        print(f"[RS] 準備完了 {self.rec.info}")

    def start_camera(self, db3_path):
        """記録を始めて最初のフレームが届くまで待つ．"ok" / "skip"（カメラなしで続ける）を返す．"""
        while True:
            self.rec.request_start(db3_path)
            t0 = time.monotonic()
            while self.rec.poll() == "starting":
                if time.monotonic() - t0 > self.cfg["rs_start_timeout_s"]:
                    self.rec.request_stop()
                    break
                self.lines([self.T["cam_wait"]])
                self.pump()
                self.clock.tick(30)
            if self.rec.state == "recording":
                return "ok"
            # 失敗: 結果（failed）を受け取ってから聞く
            res = self.wait_camera_done(timeout=30)
            reason = (res or {}).get("reason") or self.rec.error or "タイムアウト"
            print(f"[RS][ERROR] 記録を開始できません: {reason}")
            if self.rec.state == "dead":
                raise SystemExit("[ERROR] RealSense の記録プロセスが止まりました")
            k = self.wait_keys([(self.T["cam_fail"].format(r=reason), (240, 140, 120)), self.T["cam_fail_keys"]],
                               keys=(pygame.K_r, pygame.K_s), auto=pygame.K_s)
            for f in Path(db3_path).parent.glob("*.db3"):   # 失敗した .db3 は残さない
                f.unlink(missing_ok=True)
            if k == pygame.K_s:
                return "skip"

    def wait_camera_done(self, timeout=60.0):
        """記録の終了（stopped）を待って結果の dict を返す．"""
        t0 = time.monotonic()
        while self.rec.poll() in ("starting", "recording", "stopping"):
            if time.monotonic() - t0 > timeout:
                return None
            self.lines([self.T["saving"]])
            self.pump()
            self.clock.tick(30)
        return self.rec.take_result() if self.rec.state == "done" else None

    # ---------- 描画 ----------
    def draw_frame(self, traj, t, tx, ty, cur, on):
        cfg = self.cfg
        R = traj["ramp_s"]
        self.screen.fill(cfg["bg"])
        if self.preview_s > 0:
            self.draw_preview(traj, t)
        pygame.draw.circle(self.screen, cfg["target_on_color"] if on else cfg["target_color"],
                           self.to_px(tx, ty), cfg["target_radius_cm"] * self.traj_scale * self.px_per_cm)
        if t < -R:
            self.text(str(int(math.ceil(-R - t))), self.cy - 120)
        cpx = self.to_px(*cur)
        cpx = (min(max(cpx[0], 0), self.W - 1), min(max(cpx[1], 0), self.H - 1))
        pygame.draw.circle(self.screen, cfg["cursor_color"], cpx, cfg["cursor_radius_px"])

    def draw_preview(self, traj, t):
        """t から t + preview_s までのターゲットの道筋を線で描く（先へ行くほど背景に溶ける）"""
        cfg = self.cfg
        n = cfg["preview_points"]
        tp = t + np.linspace(0, self.preview_s, n)
        xs, ys = self.target_cm(traj, tp)
        pts = [self.to_px(a, b) for a, b in zip(xs, ys)]
        bg, col = np.array(cfg["bg"], float), np.array(cfg["preview_color"], float)
        w = cfg["preview_width_px"]
        for i in range(n - 1):
            k = i / (n - 1)
            c = bg + (col - bg) * (1 - k) ** 1.5 if cfg["preview_fade"] else col
            pygame.draw.line(self.screen, c.astype(int).tolist(), pts[i], pts[i + 1], w)

    def draw_static(self, remain, counting):
        self.screen.fill(self.cfg["bg"])
        c = (200, 200, 200)
        pygame.draw.line(self.screen, c, (self.cx - 20, self.cy), (self.cx + 20, self.cy), 2)
        pygame.draw.line(self.screen, c, (self.cx, self.cy - 20), (self.cx, self.cy + 20), 2)
        if counting:
            self.text(str(int(math.ceil(remain))), self.cy - 120)
        else:
            self.text(self.T["static_now"].format(s=int(math.ceil(remain))), self.cy - 120)
        self.text(self.T["howto_static"], self.cy + 100, (170, 170, 170))

    # ---------- 1 試行 ----------
    def _clock(self):
        """(経過時間 [s], 次へ進める関数)．自動プレイの --fast では実時間を待たない"""
        if self.args.fast:
            st = {"el": 0.0}

            def step():
                st["el"] += 1 / 120
                return st["el"]
            return step
        t_start = time.perf_counter()
        return lambda: time.perf_counter() - t_start

    def run_track(self, traj):
        """トラッキング 1 試行．rows（task.csv の行）と開始 epoch を返す．"""
        cfg = self.cfg
        R, Tlen = traj["ramp_s"], traj["T"]
        cd = cfg["countdown_s"]
        rad_cm = cfg["target_radius_cm"] * self.traj_scale      # 画面上の大きさは固定
        cur = np.zeros(2)
        rows = []
        self.pump()  # 溜まった動きを捨てる
        if self.sim is not None:
            self.sim.reset()
        tick = self._clock()
        epoch0 = time.time()
        frame = 0
        # t = 経過時間 − cd − R : カウントダウン中は t < −R，立ち上がり −R ≤ t < 0，解析区間 0 ≤ t < T
        while True:
            el = tick()
            t = el - cd - R
            if t >= Tlen:
                break
            ep = epoch0 + el
            dx, dy, n, _ = self.pump()
            if frame % 8 == 0:
                self.rec.poll()
            if t < -R:
                tx = ty = 0.0
                cur[:] = 0                     # カウントダウン中はカーソルを中央に固定（生の動きは記録する）
            else:
                tx, ty = (float(v) for v in self.target_cm(traj, t))
                if self.sim is not None:
                    lx, ly = ((float(v) for v in self.target_cm(traj, t + self.sim.lead))
                              if self.sim.lead > 0 else (tx, ty))
                    dx, dy, n = self.sim.step(t, lx, ly, cur, 1 / 120 if self.args.fast
                                              else max(self.clock.get_time() / 1000, 1e-3), self.cm_per_count)
                cur += np.array([dx, dy]) * self.cm_per_count
            on = math.hypot(cur[0] - tx, cur[1] - ty) <= rad_cm
            rows.append((frame, round(t, 6), f"{ep:.6f}", round(tx, 5), round(ty, 5),
                         round(cur[0], 5), round(cur[1], 5), dx, dy, n, int(on)))
            frame += 1
            if not self.args.fast or frame % 4 == 0:
                self.draw_frame(traj, t, tx, ty, cur, on)
                pygame.display.flip()
            if not self.args.fast:
                self.clock.tick_busy_loop(cfg["fps_cap"])
        return rows, epoch0

    def run_static(self, duration):
        """静止の計測．カウントダウンの後 duration 秒．マウスの生の動きも記録する（動いていないかの確認）．"""
        cd = self.cfg["countdown_s"]
        cur = np.zeros(2)
        rows = []
        self.pump()
        tick = self._clock()
        epoch0 = time.time()
        frame = 0
        while True:
            el = tick()
            t = el - cd
            if t >= duration:
                break
            dx, dy, n, _ = self.pump()
            if frame % 8 == 0:
                self.rec.poll()
            if t >= 0:
                cur += np.array([dx, dy]) * self.cm_per_count
            rows.append((frame, round(t, 6), f"{epoch0 + el:.6f}", 0.0, 0.0,
                         round(cur[0], 5), round(cur[1], 5), dx, dy, n, 0))
            frame += 1
            if not self.args.fast or frame % 4 == 0:
                self.draw_static(-t if t < 0 else duration - t, t < 0)
                pygame.display.flip()
            if not self.args.fast:
                self.clock.tick_busy_loop(self.cfg["fps_cap"])
        return rows, epoch0

    # ---------- セッション ----------
    def run_session(self, participant, calib=None, note=""):
        cfg = self.cfg
        plan = full_plan(participant, cfg, self.args.only, self.args.no_practice, self.args.no_static)
        root = out_root(cfg)
        dpi_tag = f"_dpi{cfg['mouse_dpi']:g}" if getattr(self.args, "dpi", None) else ""
        sdir = unique_dir(root / f"{cfg['session_prefix']}{participant}{dpi_tag}_{time.strftime('%Y%m%d_%H%M%S')}")
        session = {"kind": "aim_session", "participant": participant, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "calib": calib, "note": note, "camera": bool(self.rec.enabled),
                   "camera_info": self.rec.info, "config": cfg, "screen_px": [self.W, self.H],
                   "px_per_cm": self.px_per_cm, "sensitivity": self.sens, "cm_per_count": self.cm_per_count, "preview_s": self.preview_s,
                   "simulate": bool(self.args.simulate), "plan": plan, "trials": []}
        summary = []
        print(f"[SESSION] {sdir}")
        self.check_disk(root, plan)
        if self.args.simulate:
            from sim_play import SimPlayer
            a = self.args
            self.sim = SimPlayer(traj_lead=min(a.sim_lead, self.preview_s), delay=a.sim_delay, tau=a.sim_tau,
                                 noise=a.sim_noise, seed=sum(map(ord, participant)) % 1000)
        save_json(sdir / "session.json", session)
        try:
            n_track = sum(1 for p in plan if p["kind"] == "track")
            k_track = 0
            for p in plan:
                if p["kind"] == "track":
                    k_track += 1
                attempt = 0
                while True:
                    attempt += 1
                    row, again = self.run_item(p, sdir, participant, calib, attempt, k_track, n_track, len(plan))
                    summary.append(row)
                    session["trials"].append(row["_trial"])
                    del row["_trial"]
                    save_json(sdir / "session.json", session)
                    self.write_summary(sdir, summary)
                    if not again:
                        break
            self.wait_keys([self.T["done"]])
        except Quit:
            print("[SESSION] 中断しました（終わった試行までは保存済み）")
        save_json(sdir / "session.json", session)
        self.write_summary(sdir, summary)
        print(f"[SESSION] 保存先 {sdir}")
        return sdir

    def run_item(self, p, sdir, participant, calib, attempt, k_track, n_track, n_all):
        """1 試行（静止またはトラッキング）．(summary の行, やり直すか) を返す．"""
        cfg = self.cfg
        static = p["kind"] == "static"
        if static:
            head = self.T["next"].format(i=p["index"] + 1, n=n_all, kind=self.T["static"])
            self.wait_keys([head, self.T["howto_static"]])
        else:
            kind = self.T["practice"] if p["practice"] else self.T["main"]
            head = self.T["next"].format(i=p["index"] + 1, n=n_all, kind=kind)
            self.wait_keys([head, self.T["howto"]])
        tdir = unique_dir(sdir / f"trial_{p['index']:02d}_{p['name']}")

        # カメラ
        want_cam = self.rec.enabled and (static or not p["practice"] or cfg["rs_record_practice"])
        cam = "off"
        if want_cam:
            cam = "on" if self.start_camera(tdir / "session.db3") == "ok" else "skipped"

        # 課題
        traj = None
        if static:
            rows, epoch0 = self.run_static(cfg["static_s"])
        else:
            traj = make_trajectory(p["band"], p["seed"], cfg)
            rows, epoch0 = self.run_track(traj)

        # カメラを止める（少し余分に記録してから）
        rs_res = None
        if cam == "on":
            t_end = time.monotonic() + (0 if self.args.fast else cfg["rs_tail_s"])
            while time.monotonic() < t_end:
                self.pump()
                self.rec.poll()
                self.clock.tick(120)
            self.rec.request_stop()
            rs_res = self.wait_camera_done()
            if rs_res is None:
                rs_res = {"status": "failed", "reason": "記録の終了を確認できない", "frames": 0}

        # 保存
        with open(tdir / "task.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(TASK_COLS)
            w.writerows(rows)
        a = np.array([[r[1], r[3], r[4], r[5], r[6], r[10]] for r in rows], float)
        m = a[:, 0] >= 0
        dt = np.diff(a[m, 0])
        stats = {"frames": int(m.sum()), "fps_median": round(float(1 / np.median(dt)), 1),
                 "dt_max_ms": round(float(1000 * dt.max()), 1)}
        if static:
            stats["static_move_cm"] = round(float(np.hypot(a[m, 3], a[m, 4]).max()), 3)
        else:
            err = np.hypot(a[m, 3] - a[m, 1], a[m, 4] - a[m, 2])
            stats["rms_err_cm"] = round(float(np.sqrt(np.mean(err ** 2))), 3)
            stats["on_target_pct"] = round(float(100 * a[m, 5].mean()), 1)
        cam_msg = self.camera_check(rs_res, rows) if cam == "on" else ("記録なし（スキップ）" if cam == "skipped" else "")
        task_info = {**{k: v for k, v in p.items()}, "attempt": attempt, "dir": tdir.name, "epoch_start": epoch0,
                     "epoch_end": float(rows[-1][2]), "preview_s": self.preview_s, "camera": cam,
                     "trajectory": traj, **stats}
        rs_part = dict(rs_res) if rs_res else {"status": "no_camera", "frames": 0}
        meta = {"kind": "trial", "session": tdir.name, "subject": participant, "trial": p["index"],
                "calib": calib, **rs_part, "task": task_info}
        save_json(tdir / "meta.json", meta)

        # 画面とログ
        label = "static" if static else f"{p['band']} s{p['seed']}{' (練習)' if p['practice'] else ''}"
        perf = (f"動き {stats['static_move_cm']:.2f} cm" if static
                else f"rms {stats['rms_err_cm']:.2f} cm, on {stats['on_target_pct']:.0f}%")
        rs_txt = f", camera {rs_part.get('status')} {rs_part.get('frames', 0)} frames" if cam == "on" else ""
        print(f"[{p['index']:02d}] {label} {perf}, {stats['fps_median']} fps{rs_txt}  -> {tdir.name}")
        res_line = (self.T["result_static"].format(d=stats["static_move_cm"]) if static
                    else self.T["result"].format(on=stats["on_target_pct"], rms=stats["rms_err_cm"]))
        shown = [res_line]
        if cam_msg and cam == "on":
            shown.append((self.T["cam_bad"].format(msg=cam_msg), (240, 140, 120)))
            print(f"[RS][WARN] {cam_msg}")
        shown.append(self.T["result_keys"])
        k = self.wait_keys(shown, keys=(CLICK, pygame.K_r))
        again = k == pygame.K_r

        row = {"index": p["index"], "attempt": attempt, "kind": p["kind"], "band": p.get("band", ""),
               "seed": p.get("seed", ""), "practice": p["practice"], "repeat": p.get("repeat", 0),
               "dir": tdir.name, "preview_s": self.preview_s, **stats,
               "camera": cam, "rs_status": rs_part.get("status"), "rs_frames": rs_part.get("frames", 0),
               "camera_note": cam_msg, "redone": again}
        trial_entry = {k2: v for k2, v in task_info.items() if k2 != "trajectory"}
        trial_entry.update(rs_status=rs_part.get("status"), rs_frames=rs_part.get("frames", 0),
                           camera_note=cam_msg, redone=again)
        row["_trial"] = trial_entry
        return row, again

    def camera_check(self, res, rows):
        """記録が課題の全体を覆っているか，fps が落ちていないかを確かめる．問題がなければ ""．"""
        if res.get("status") != "ok":
            return f"{res.get('status')}（{res.get('reason', '')}）"
        f0, f1 = res.get("first_frame_epoch_ms"), res.get("last_frame_epoch_ms")
        if not f0 or not f1:
            return "フレームの時刻がない"
        e0, e1 = float(rows[0][2]) * 1000, float(rows[-1][2]) * 1000
        if f0 > e0 or f1 < e1:
            return f"記録が課題の一部しか覆っていない（前 {max(0, f0 - e0):.0f} ms，後 {max(0, e1 - f1):.0f} ms 欠け）"
        fps = (res["frames"] - 1) / max((f1 - f0) / 1000, 1e-6)
        if fps < 0.9 * res.get("fps", fps):
            return f"fps が低い（{fps:.1f} / {res.get('fps')}）．コマ落ちの可能性"
        return ""

    def check_disk(self, root, plan):
        """カメラの記録に必要な容量の目安（圧縮なしの概算）と空き容量を比べる．"""
        if not self.rec.enabled or not self.rec.info:
            return
        i = self.rec.info
        per_s = i["width"] * i["height"] * (3 + 2) * i["fps"]             # カラー 3 B + 深度 2 B / 画素
        secs = 0.0
        for p in plan:
            if p["kind"] == "static":
                secs += self.cfg["static_s"] + self.cfg["countdown_s"] + 2
            elif not p["practice"] or self.cfg["rs_record_practice"]:
                secs += self.cfg["trial_s"] + self.cfg["ramp_s"] + self.cfg["countdown_s"] + 2
        need = per_s * secs
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free
        print(f"[DISK] カメラの記録 約 {need / 1e9:.0f} GB（圧縮なしの概算．実際の .db3 はこれより小さいことがある），"
              f"空き {free / 1e9:.0f} GB")
        if free < need:
            print("[DISK][WARN] 空き容量が足りない可能性があります")

    def write_summary(self, sdir, summary):
        if not summary:
            return
        keys = []
        for r in summary:
            keys += [k for k in r if k not in keys]
        with open(sdir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summary)

    # ---------- DPI の実測 ----------
    def measure_dpi(self, cm=20.0):
        while True:
            self.wait_keys([self.T["dpi1"].format(cm=cm)])
            total = 0
            while True:
                dx, _, _, keys = self.pump()
                total += dx
                self.lines([self.T["dpi2"].format(c=total)])
                if CLICK in keys:
                    break
                self.clock.tick(240)
            dpi = abs(total) / (cm / 2.54)
            print(f"{total} counts / {cm} cm = {dpi:.1f} counts/inch")
            self.wait_keys([self.T["dpi3"].format(c=total, cm=cm, dpi=dpi)])


def add_common_args(ap):
    ap.add_argument("participant", nargs="?", default=None, help="参加者 ID（例 P01）")
    ap.add_argument("--dpi", type=float, default=None,
                    help="PC 側で設定したマウスの DPI（実測値．task_config の mouse_dpi を上書きし，フォルダ名にも入る）")
    ap.add_argument("--only", help="band:seed の 1 本だけ（例 fast:1）")
    ap.add_argument("--no-practice", action="store_true", help="練習を飛ばす")
    ap.add_argument("--no-static", action="store_true", help="静止の計測を飛ばす")
    ap.add_argument("--window", action="store_true", help="全画面にしない")
    ap.add_argument("--preview", type=float, default=None,
                    help="予告線の長さ [s]（0 で表示しない．省略時は task_config の preview_s）")
    ap.add_argument("--note", default="", help="session.json に残すメモ")
    ap.add_argument("--simulate", action="store_true", help="自動プレイ（動作確認用）")
    ap.add_argument("--fast", action="store_true", help="--simulate で実時間を待たない")
    ap.add_argument("--sim-delay", type=float, default=0.15)
    ap.add_argument("--sim-tau", type=float, default=0.12)
    ap.add_argument("--sim-noise", type=float, default=0.3)
    ap.add_argument("--sim-lead", type=float, default=0.0, help="自動プレイが予告線を見て先回りする時間 [s]")