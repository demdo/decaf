# -*- coding: utf-8 -*-
"""
test_color_exposure.py

Checks the marker exposure control (tracking.hydramarker.color_exposure)
and its hooks with fakes - no camera, no Qt; the C++ engine only for
tracker.TrackerResult (e) and HydraTracker.ir_frame_timing (k).

(a) CommandedColorExposure: AE without a number; frames up to the call
    return old, the next LATENCY_FRAMES + 1 in transition (mean), then new;
    the frame register decides a manual -> manual step exactly
(b) every frame gets the right E_RGB: a scripted command sequence against a
    camera whose frames change 3 (or 4) frames after the call -> each
    frame's exposure is exact or a transition that contains the truth,
    never wrong; with the register always exact
(c) controller, closed loop on a plant (brightness ~ exposure x the
    measured gain response x light, tone curve, clip 255, 3-4 old frames,
    gain drift, a frame with a highlight): settles into the band; a static
    scene then makes ZERO steps at every light level (x0.3 .. x30; x100
    and x300 cannot be tracked: slow search); 7.8 ms at the table,
    <= 3.9 ms in bright light; lamp on / off -> tracked again within
    frames; out of view under the same light (table, and x30 without a
    lamp in view) -> no write at all, tracked again at once when back;
    out of view while the lamp goes on -> the ladder finds it; steps
    <= x2 (<= x1.41 moving); at the end of the range no pumping
(d) actuator: AE off once, exposure then gain, newest command wins,
    applied events with the frame index, restore exposure -> gain -> AE on
    and its read-back, nothing restored if never manual, a failing
    set_option -> a failed event with the read-back and the frames follow
    it, the thread goes on; restore: a failing write does not stop the AE
    write, mismatches reported; an AE-off call that raised but took
    effect is restored; stop joins
(l) policy: a marker that comes back into a lamp spot under unchanged
    frame light locks (the spot clips in the frame: darker probes), without
    a spot nothing is written; a short glare does not cap the brightness
    for good (ceiling expiry, back to 3.9 ms), a lamp spot does not make it
    cycle (backoff); 1.95 ms is left as soon as the gain at 3.9 ms stays >=
    its floor; the gain response; the frames between a write and its
    effect do not count as a light change; a cold start in bright light
    does not restart its search at the clipped first preset; a cold start
    in a pose that never decodes keeps the preset with detections as home
    of the slow cycle; one that detects only at a preset too dark to
    decode leaves it after 2 s and locks; a marker lost out of view (dark
    hull, frame unchanged) is not brightened, under a darker frame (lamp
    off) at once; at the table a short bright region while the drill is
    away costs one probe cycle, a single glint nothing; a label that goes
    dark alone under the lamp is found by the brightest preset after 3 s
(m) MarkerExposureControl with a failing camera: the first write fails ->
    failure (the caller stops it); one failed write later -> dropped, the
    control goes on; two in a row -> failure
(n) camera source: after the stop the restore's frames are in transition,
    then the state read at the start again; a restore stuck in set_option
    leaves the source busy (no new control) until it returns
(e) MarkerMeter: white / black level and clipped share of a synthetic
    marker, grey == BGRA, the last hull without a pose (and its expiry),
    the frame grid; cost on a 1920x1080 BGRA frame; the detected corners
    of tracker.TrackerResult (the type the Live page hands it)
(f) TrackingWorker: the metering travels with its output, set_meter(None)
    stops it, a raising meter costs only the metering
(g) DisplayBrightener: brightens a dark display in place, leaves a bright
    one alone; cost
(h) camera start after a crash in manual: the K1 writes put the AE, the
    exposure register and the gain back (fake rs sensor)
(i) ColorExposureState: exposure_ms None under the AE (stale register)
(j) CameraConfig.realsense_color_exposure_mode default "auto", validated
(k) HydraTracker.ir_frame_timing: a transition frame gets the mean offset

Run from src/:

    python -m overlay.tests.test_color_exposure
"""

from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

from overlay.tools.live_tracking import TrackingWorker
from tracking.hydramarker import color_exposure as ce
from tracking.hydramarker.camera_setup import (
    ColorExposureState,
    _configure_realsense_color_sensor,
)
from tracking.hydramarker.config import CameraConfig


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ============================================================
# (a) / (b) per-frame exposure state
# ============================================================

def test_commanded_state():
    c = ce.CommandedColorExposure(True, 15.6)
    check(c.for_frame(5) == {"auto": True, "exposure_ms": None, "gain": None}, "AE carries no number")
    c.applied(False, 7.8, 10)                      # AE -> 7.8 ms, call returned at frame 10
    check(c.for_frame(10) == {"auto": True, "exposure_ms": None, "gain": None}, "frame of the return: old")
    for k in range(11, 11 + ce.LATENCY_FRAMES + 1):
        s = c.for_frame(k, 156.0)
        check("previous" in s and s["previous"]["auto"] and s["exposure_ms"] == 7.8,
              f"AE -> manual frame {k} in transition: {s}")
    check(c.for_frame(11 + ce.LATENCY_FRAMES + 1)["exposure_ms"] == 7.8, "after: new")
    c.applied(False, 3.9, 30)
    check(c.for_frame(32, 78.0) == {"auto": False, "exposure_ms": 7.8, "gain": None}, "register says old")
    check(c.for_frame(33, 38.0) == {"auto": False, "exposure_ms": 3.9, "gain": None},
          "register says new (38 for 39)")
    s = c.for_frame(33, None)
    check(s["exposure_ms"] == 3.9 and s["previous"]["exposure_ms"] == 7.8, "no register: both")
    check(ce.register_exposure_ms(20.0) == 1.95 and ce.register_exposure_ms(19.0) == 1.95, "1.95 ms register")
    check(ce.register_exposure_ms(50.0) is None, "no ladder value near register 50")
    print("  (a) commanded state: AE without a number, old/transition/new by frame index, register decides")


def test_every_frame_e_rgb():
    rng = np.random.default_rng(3)
    for use_reg in (False, True):
        cmd = ce.CommandedColorExposure(True, None)
        truth = []           # (frame index, exposure ms or "AE") of every frame
        cur, n_old = "AE", 0
        pending = None
        seq = [7.8, 3.9, 7.8, 1.95, 3.9, 7.8, 7.8, 3.9] * 6
        k = 0
        n_exact = n_mean = 0
        for step in seq:
            # the call returns at frame k; 3 (sometimes 4) more frames show the old value
            old = cur
            cmd.applied(False, step, k)
            delay = 3 + int(rng.random() < 0.2)
            for j in range(1, 12):
                k += 1
                is_new = j > delay
                val = step if is_new else old
                reg = None
                if use_reg:
                    reg = (156.0 if val == "AE"
                           else ce.exposure_option_value(val) - (1.0 if val == 3.9 else 0.0))
                s = cmd.for_frame(k, reg)
                tv = None if val == "AE" else val
                options = [(s["auto"], s["exposure_ms"])]
                if "previous" in s:
                    options.append((s["previous"]["auto"], s["previous"]["exposure_ms"]))
                    n_mean += 1
                else:
                    n_exact += 1
                ok = any((a and val == "AE") or (not a and tv is not None and abs(e - tv) < 1e-6)
                         for a, e in options)
                check(ok, f"frame {k}: truth {val}, got {s} (register {'on' if use_reg else 'off'})")
            cur = step
        if use_reg:
            # only the AE -> manual start has no register decision
            check(n_mean <= ce.LATENCY_FRAMES + 1,
                  f"with the register at most the first step in transition ({n_mean})")
        print(f"  (b) every frame's E_RGB right, register {'used' if use_reg else 'not used'}: "
              f"{n_exact} exact, {n_mean} transition (mean), 0 wrong")


# ============================================================
# (c) controller on a plant
# ============================================================

class Plant:
    """brightness of the marker's white = light x setting_brightness (exposure x the measured gain response)
    in linear units through the tone curve; a setting written at frame k shows from frame k + 4 (3 old frames;
    4 with prob 0.2), gain steps drift +5.9 / +4 / +2 % over the first 3 new frames. The frame: a background at
    0.1-4 x 0.3 of the marker white (the top 1/4 a lit highlight); its clipped share is the frame clip."""

    def __init__(self, light, seed=0):
        self.light = light
        self.rng = np.random.default_rng(seed)
        self.E, self.G = 15.6, 64.0
        self.queue = []     # (frame from which it shows, E, G)
        self.drift = []

    def write(self, k, E, G):
        d = 4 + int(self.rng.random() < 0.2)
        self.queue.append((k + d, E, G))

    def frame(self, k):
        for item in list(self.queue):
            if item[0] <= k:
                if abs(item[2] - self.G) > 1e-6:
                    self.drift = [1.059, 1.04, 1.02]
                self.E, self.G = item[1], item[2]
                self.queue.remove(item)
        f = self.drift.pop(0) if self.drift else 1.0
        lin = 0.00027 * self.light(k) * ce.setting_brightness(self.E, self.G) * f
        white_lin = lin * np.linspace(0.85, 1.15, 200)
        dn = np.array([float(ce.dn_of_linear(x)) for x in np.minimum(white_lin, 1.0)])
        white = float(np.median(dn))
        # the frame: a background at 0.3 x the marker white, a highlight up to 4 x that
        self.frame_median = float(ce.dn_of_linear(min(1.0, 0.3 * lin)))
        bg = ce.dn_of_linear(np.minimum(0.3 * lin * np.linspace(0.1, 4.0, 200), 1.0))
        self.frame_clip = float(np.mean(bg >= 250))
        return white, float(np.mean(dn >= 250)), self.E, self.G


def run_loop(light, visible, n, speed=None, seed=0, start=None):
    plant = Plant(light, seed)
    ctl = ce.MarkerExposureController()
    act_queue = []
    t0 = 0.0
    cmd = ctl.start(t0)
    log = []
    plant.write(0, cmd.exposure_ms, cmd.gain)
    ctl.applied(cmd, 0)
    for k in range(1, n):
        white, clip, E, G = plant.frame(k)
        vis = visible(k)
        # a pose needs a marker in view, a white level >= 25 DN and < 90 % of the white clipped
        fresh = vis and white >= 25.0 and clip < 0.9
        m = ce.Metering(frame_index=k, t_frame=k / 30.0, fresh=fresh,
                        white_dn=white if (vis and fresh) else (white if vis else float("nan")),
                        clip_frac=clip if vis else float("nan"),
                        n_samples=2000 if vis else 0, hull_from_pose=fresh,
                        frame_clip_frac=plant.frame_clip, frame_median_dn=plant.frame_median,
                        speed_px=(speed(k) if speed else 0.0))
        c = ctl.update(m)
        if c is not None:
            plant.write(k, c.exposure_ms, c.gain)
            ctl.applied(c, k)
            act_queue.append((k, c.exposure_ms, c.gain, c.reason))
        log.append((k, fresh, white, clip, E, G, ctl.state))
    return ctl, log, act_queue


def test_controller_closed_loop():
    cfg = ce.MarkerExposureConfig()
    lo, hi = cfg.white_low_dn, cfg.white_high_dn
    # table light: settles, then a static scene produces no step
    ctl, log, steps = run_loop(lambda k: 1.0, lambda k: True, 900)
    settle = next(k for k, fr, w, c, E, G, s in log if fr and lo <= w <= hi and c <= cfg.clip_max_frac)
    late = [s for s in steps if s[0] > settle + 30]
    check(not late, f"static table scene: steps after settling {late}")
    print(f"  (c) table: in band from frame {settle}, {len(steps)} writes in all, 0 after settling "
          f"(final {log[-1][4]:g} ms / {log[-1][5]:.0f}, white {log[-1][2]:.0f})")
    # every light level in 0.3 .. 30 x: static -> zero steps after settling (or a limit state);
    # x100 / x300: clipped even at 1.95 ms / 16 (only x0.073 of 7.8 ms / 128 with the measured gain
    # response) -> no pose at all, the search cycles slowly
    for L in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
        ctl, log, steps = run_loop(lambda k, L=L: L, lambda k: True, 600, seed=int(L * 10))
        inb = [k for k, fr, w, c, E, G, s in log if fr and lo <= w <= hi and c <= cfg.clip_max_frac]
        last = steps[-1][0] if steps else 0
        tail = [s for s in steps if s[0] > 300]
        if L < 100:
            check(not tail, f"light x{L}: steps in the static second half {tail}")
        else:
            check(len(tail) <= 2 * 300 / 30 / cfg.search_slow_s + 2 and ctl.state == "search-slow",
                  f"x{L}: slow search expected, {len(tail)} writes in 10 s")
        fin = log[-1]
        print(f"      light x{L:<5g}: last write frame {last:3d}, final {fin[4]:g} ms / {fin[5]:.0f}, "
              f"white {fin[2]:.0f} DN, clip {100 * fin[3]:.0f} %, state {fin[6]}, in band from "
              f"{inb[0] if inb else '-'}")
        if 30.0 <= L < 100:
            check(fin[4] <= 3.9 + 1e-6, f"bright light x{L}: exposure {fin[4]} should be <= 3.9 ms")
        if L <= 1.0:
            # (x3 needs < 7.8 ms / 16: the gain darkens 7.8 ms only to x0.29)
            check(fin[4] == 7.8, f"table light x{L}: exposure {fin[4]} should stay 7.8 ms")
    # lamp switched on at frame 200 (x1 -> x40), off at 500
    light = lambda k: 40.0 if 200 <= k < 500 else 1.0
    ctl, log, steps = run_loop(light, lambda k: True, 800)
    lost = [k for k, fr, *_ in log if not fr]
    on_back = next(k for k, fr, w, c, *_ in log if k > 200 and fr and c <= cfg.clip_max_frac)
    off_back = next(k for k, fr, w, c, *_ in log if k > 500 and fr and lo <= w <= hi)
    late = [s for s in steps if 300 < s[0] < 500 or s[0] > 650]
    check(not late, f"lamp: steps in the static parts {late}")
    print(f"      lamp on at 200: tracked unclipped again at {on_back} ({on_back - 200} frames), off at 500: "
          f"{off_back} ({off_back - 500} frames), frames without a pose {len(lost)}")
    # out of view 100..400 under the same light, nothing in the frame clips (table, and x30 once settled at
    # 1.95 ms / 16 - no lamp in view there, see (l) for one): the setting is kept, nothing is written (x30:
    # the label above the band at the darkest setting, nothing darker left); back -> tracks at once
    for L in (1.0, 30.0):
        ctl, log, steps = run_loop(lambda k, L=L: L, lambda k: not (100 <= k < 400), 600)
        out_w = [s for s in steps if 100 <= s[0] < 400]
        check(not out_w, f"x{L:g}: writes out of view under unchanged light {out_w}")
        back = next(k for k, fr, *_ in log if k >= 400 and fr)
        check(back - 400 <= 6, f"x{L:g}: pose again {back - 400} frames after the marker came back")
        states = sorted(set(s for *_, s in log[100:400]))
        print(f"      out of view 100-400, same light x{L:g}: 0 writes ({states}), pose again "
              f"{back - 400} frames after the marker came back")
    # out of view 100..400 and the lamp goes on at 200 (x30): the ladder runs, the marker is found
    # under the lamp
    ctl, log, steps = run_loop(lambda k: 30.0 if k >= 200 else 1.0, lambda k: not (100 <= k < 400), 700)
    probe = [s for s in steps if 200 <= s[0] < 400]
    check(2 <= len(probe) <= 20, f"probes after the light changed: {len(probe)}")
    back = next(k for k, fr, w, c, *_ in log if k >= 400 and fr and c <= cfg.clip_max_frac)
    check(back - 400 <= 60, f"found under the lamp {back - 400} frames after it came back")
    print(f"      out of view 100-400, lamp x30 on at 200: {len(probe)} probe writes, pose unclipped again "
          f"{back - 400} frames after the marker came back")
    # step size: a dark marker at 7.8 ms / 40 -> x1.41 while moving, x2 while still
    def at(E, G):
        c = ce.MarkerExposureController()
        first = c.start(0.0)
        c.applied(ce.ExposureCommand(E, G, "x", first.seq), 0)
        return c

    for speed, want in ((12.0, 1.41), (1.0, 2.0)):
        c = at(7.8, 40.0)
        got = None
        for i in range(cfg.confirm_frames + 2):
            got = got or c.update(ce.Metering(frame_index=100 + i, t_frame=1.0, fresh=True, white_dn=20.0,
                                              clip_frac=0.0, n_samples=2000, hull_from_pose=True,
                                              speed_px=speed))
        check(got is not None and abs(ce.setting_brightness(got.exposure_ms, got.gain)
                                      / ce.setting_brightness(7.8, 40.0) - want) < 0.05,
              f"step at {speed} px: {got}")
    # at the end of the range: too dark at 7.8 / 128 -> "limit", no write, no pumping
    ctl4 = at(7.8, 128.0)
    writes = [ctl4.update(ce.Metering(frame_index=100 + i, t_frame=1.0 + i / 30, fresh=True, white_dn=30.0,
                                      clip_frac=0.0, n_samples=2000, hull_from_pose=True))
              for i in range(200)]
    check(all(w is None for w in writes) and ctl4.state == "limit",
          "too dark at the maximum: no write, limit")
    print("      steps: x1.41 moving, x2 still; too dark at 7.8 ms / 128: state limit, "
          "0 writes in 200 frames")


# ============================================================
# (d) actuator
# ============================================================

class FakeSensor:
    """fail_on: options whose set_option raises (applied_then_raise: the
    value is taken, then it raises - a time-out that took effect);
    fail_read: get_option raises too."""

    def __init__(self, fail_on=None, call_s=0.002, applied_then_raise=False):
        self.values = {"enable_auto_exposure": 1.0, "exposure": 156.0, "gain": 64.0}
        self.calls = []
        self.fail_on = set() if fail_on is None else {fail_on} if isinstance(fail_on, str) else set(fail_on)
        self.applied_then_raise = applied_then_raise
        self.fail_read = False
        self.call_s = call_s
        self.threads = set()

    def set_option(self, opt, value):
        self.threads.add(threading.current_thread().name)
        time.sleep(self.call_s)          # like the SDK call
        if opt in self.fail_on:
            if self.applied_then_raise:
                self.values[opt] = value
            raise RuntimeError("device gone")
        self.calls.append((opt, value))
        self.values[opt] = value

    def get_option(self, opt):
        if self.fail_read:
            raise RuntimeError("device gone")
        return self.values[opt]


OPTS = SimpleNamespace(enable_auto_exposure="enable_auto_exposure", exposure="exposure", gain="gain")


def test_actuator():
    sensor = FakeSensor()
    frame = [100]
    commanded = ce.CommandedColorExposure(True, None)
    act = ce.ColorExposureActuator(sensor, OPTS, lambda: frame[0], commanded,
                                   start_settings={"enable_auto_exposure": True, "exposure": 156.0,
                                                   "gain": 64.0})
    act.start()
    act.submit(ce.ExposureCommand(7.8, 128.0, "start", 1))
    t0 = time.time()
    while not act.calls or len(act.calls) < 3:
        time.sleep(0.001)
        check(time.time() - t0 < 2, "actuator did not write")
    ev = act.poll_applied()
    check(len(ev) == 1 and ev[0][1] == 100, f"applied event {ev}")
    check([c[0] for c in sensor.calls] == ["enable_auto_exposure", "exposure", "gain"],
          f"order {sensor.calls}")
    check(sensor.values == {"enable_auto_exposure": 0.0, "exposure": 78.0, "gain": 128.0}, f"{sensor.values}")
    check(commanded.for_frame(200) == {"auto": False, "exposure_ms": 7.8, "gain": 128.0}, "commanded follows")
    # a burst: only the newest command (or the one already running) is written
    for i in range(20):
        act.submit(ce.ExposureCommand(3.9, 16.0 + i, "burst", 10 + i))
    time.sleep(0.2)
    gains = [v for o, v in sensor.calls if o == "gain"]
    check(gains[-1] == 35.0 and len(gains) <= 4, f"newest wins: {gains}")
    check(sensor.threads == {"colour-exposure"}, f"set_option threads {sensor.threads}")
    t = time.perf_counter()
    ok = act.stop(restore=True)
    ms = (time.perf_counter() - t) * 1000
    check(ok and not act.alive, "stop joins")
    check([c[0] for c in sensor.calls[-3:]] == ["exposure", "gain", "enable_auto_exposure"], "restore order")
    check(sensor.values == {"enable_auto_exposure": 1.0, "exposure": 156.0, "gain": 64.0}, "restored")
    check(act.restored == {"enable_auto_exposure": 1.0, "exposure": 156.0, "gain": 64.0}, f"{act.restored}")
    check(commanded.for_frame(10 ** 6) == {"auto": True, "exposure_ms": None, "gain": None},
          "commanded AE again")
    # never manual: nothing written at the stop
    s2 = FakeSensor()
    a2 = ce.ColorExposureActuator(s2, OPTS, lambda: 0, None, start_settings={"exposure": 156.0, "gain": 64.0})
    a2.start()
    a2.stop(restore=True)
    check(not s2.calls and a2.restored.get("skipped"), "no write without a manual phase")
    # a failing write: recorded, a failed event with the read-back, the frames follow the device; the
    # thread goes on
    s3 = FakeSensor(fail_on="gain")
    c3 = ce.CommandedColorExposure(True, None)
    a3 = ce.ColorExposureActuator(s3, OPTS, lambda: 50, c3)
    a3.start()
    a3.submit(ce.ExposureCommand(7.8, 128.0, "x", 1))
    time.sleep(0.05)
    a3.submit(ce.ExposureCommand(3.9, 64.0, "x", 2))
    time.sleep(0.05)
    failed = a3.poll_failed()
    check(len(a3.errors) == 2 and a3.alive and not a3.poll_applied(), f"errors recorded, alive: {a3.errors}")
    check(len(failed) == 2 and failed[0][4] == (False, 7.8, 64.0) and failed[1][4] == (False, 3.9, 64.0),
          f"failed events with the read-back (exposure written, gain not): {[f[4] for f in failed]}")
    check(c3.for_frame(10 ** 6) == {"auto": False, "exposure_ms": 3.9, "gain": 64.0},
          f"frames carry the device's setting, not the AE: {c3.for_frame(10 ** 6)}")
    a3.stop(restore=False)
    # every write fails, the read-back too: nothing known -> the frames keep the old state
    s4 = FakeSensor(fail_on=("enable_auto_exposure", "exposure", "gain"))
    s4.fail_read = True
    c4 = ce.CommandedColorExposure(True, None)
    a4 = ce.ColorExposureActuator(s4, OPTS, lambda: 0, c4,
                                  start_settings={"enable_auto_exposure": True, "exposure": 156.0, "gain": 64.0})
    a4.start()
    a4.submit(ce.ExposureCommand(7.8, 128.0, "x", 1))
    time.sleep(0.05)
    f4 = a4.poll_failed()
    check(len(f4) == 1 and f4[0][4] is None and c4.for_frame(10 ** 6)["auto"], f"all fails: {f4}")
    a4.stop(restore=True)
    check(a4.restored.get("errors") and a4.restored.get("restore_mismatch"),
          f"a restore that cannot write or read reports it: {a4.restored}")
    # restore: the exposure write fails -> gain and the AE are still written, the mismatch reported
    s5 = FakeSensor()
    c5 = ce.CommandedColorExposure(True, None)
    a5 = ce.ColorExposureActuator(s5, OPTS, lambda: 7, c5,
                                  start_settings={"enable_auto_exposure": True, "exposure": 156.0, "gain": 64.0})
    a5.start()
    a5.submit(ce.ExposureCommand(3.9, 16.0, "x", 1))
    time.sleep(0.05)
    s5.fail_on = {"exposure"}
    a5.stop(restore=True)
    check(s5.values["enable_auto_exposure"] == 1.0 and s5.values["gain"] == 64.0,
          f"AE back on although the exposure write failed: {s5.values}")
    check(a5.restored.get("restore_mismatch") == ["exposure"] and len(a5.restored.get("errors", [])) == 1,
          f"restore mismatch reported: {a5.restored}")
    check(c5.for_frame(10 ** 6) == {"auto": True, "exposure_ms": None, "gain": None}, "frames: AE (read back)")
    # the AE-off call raised but took effect: the restore still runs
    s6 = FakeSensor(fail_on="enable_auto_exposure", applied_then_raise=True)
    a6 = ce.ColorExposureActuator(s6, OPTS, lambda: 0, None,
                                  start_settings={"enable_auto_exposure": True, "exposure": 156.0, "gain": 64.0})
    a6.start()
    a6.submit(ce.ExposureCommand(7.8, 128.0, "x", 1))
    time.sleep(0.05)
    check(s6.values["enable_auto_exposure"] == 0.0, "fake: AE off taken despite the error")
    s6.fail_on = set()
    a6.stop(restore=True)
    check(s6.values["enable_auto_exposure"] == 1.0 and "skipped" not in a6.restored,
          f"restored after an AE-off call that raised: {a6.restored}")
    print(f"  (d) actuator: AE off once, exposure, gain; newest wins ({len(gains)} gain writes for 21 "
          f"commands); applied event with the frame index; stop + restore in {ms:.1f} ms (fake calls 2 ms "
          f"each); no restore without manual; failing writes -> failed events with the read-back, frames "
          f"follow the device; restore: AE written after a failed exposure write, mismatch reported; "
          f"AE-off that raised but took effect restored")


# ============================================================
# (l) policy cases
# ============================================================

def _feed(ctl, pending, i, white, clip, fresh, frame_median, writes, frame_clip=0.0, n_detected=0):
    """one frame through the controller; a command is 'written' at once (the actuator's return =
    this frame) and shows from frame + 4"""
    m = ce.Metering(frame_index=i, t_frame=i / 30.0, fresh=fresh, white_dn=white, black_dn=20.0,
                    clip_frac=clip, n_samples=2000 if fresh or np.isfinite(white) else 0,
                    hull_from_pose=fresh, frame_clip_frac=frame_clip, frame_median_dn=frame_median,
                    speed_px=0.0, n_detected=n_detected)
    if pending is not None:
        writes.append((i, pending.exposure_ms, pending.gain, pending.reason))
        ctl.applied(pending, i)
    return ctl.update(m)


def spot_reentry(spot, n=600):
    """table setting 7.8 / 128, marker white 90 DN; lost at 100 (out of view), back at 160 inside a lamp
    spot `spot` x brighter, in view all the time beside the marker (frame median unchanged; the frame clip
    is 4 % while the spot clips at the setting). A pose iff 30 <= white < 250. Returns (frames from
    re-entry to the first pose or None, writes after the loss)."""
    dn = lambda lin: float(ce.dn_of_linear(lin))
    B = ce.setting_brightness
    ctl = ce.MarkerExposureController(gain_range=(0, 128))
    pending = ctl.start(0.0)
    writes, actual = [], (7.8, 128.0)
    L_w = ce.linear_dn(90.0) / B(7.8, 128)
    L_f = ce.linear_dn(110.0) / B(7.8, 128)
    lock = None
    for i in range(1, n):
        for w in writes:
            if i == w[0] + 4:
                actual = (w[1], w[2])
        P = B(*actual)
        present = i < 100 or i >= 160
        white = dn(L_w * P * (spot if i >= 160 else 1.0))
        fresh = present and 30.0 <= white < 250.0
        if fresh and i >= 160 and lock is None:
            lock = i - 160
        fclip = 0.04 if dn(L_f * P * spot) >= 250.0 else 0.0
        pending = _feed(ctl, pending, i, white if fresh else float("nan"), 0.0 if fresh else float("nan"),
                        fresh, dn(L_f * P), writes, frame_clip=fclip)
    return lock, len([w for w in writes if w[0] >= 100])


def glare(glare_frames, base_dn, persistent=False, n=600):
    """the controller settled at 3.9 / 16 with the marker at base_dn; at frame 100 a glare clips 80 % of
    the white for glare_frames (persistent: a lamp spot that clips the label at >= 3.9 / 16 for good).
    Returns (end exposure, gain, end white, below-band frames and writes after 2 s past the glare,
    clipped frames after that)."""
    dn = lambda lin: float(ce.dn_of_linear(lin))
    start = (3.9, 16.0)
    cfg = ce.MarkerExposureConfig(search_presets=(start,) + ce.MarkerExposureConfig().search_presets[1:])
    ctl = ce.MarkerExposureController(cfg, gain_range=(0, 128))
    pending = ctl.start(0.0)
    writes, actual = [], (7.8, 128.0)
    B = ce.setting_brightness
    L = ce.linear_dn(base_dn) / B(*start)
    L_f = ce.linear_dn(110.0) / B(*start)
    below = n_clip = 0
    t_after = 100 + glare_frames + 60
    for i in range(1, n):
        for w in writes:
            if i == w[0] + 4:
                actual = (w[1], w[2])
        P = B(*actual)
        wd = dn(L * P)
        g = (i >= 100 and P >= 0.99 * B(*start)) if persistent else (100 <= i < 100 + glare_frames)
        white, clip = (252.0, 0.8) if g else (wd, 1.0 if wd >= 250 else 0.0)
        if i > t_after:
            below += (not g) and wd < 60
            n_clip += bool(g)
        pending = _feed(ctl, pending, i, white, clip, True, dn(L_f * P), writes)
    P = B(*actual)
    return actual[0], actual[1], dn(L * P), below, len([w for w in writes if w[0] > t_after]), n_clip


def test_policy():
    cfg = ce.MarkerExposureConfig()
    # a marker back inside a lamp spot (x16 ~ the OR estimate 3.9 ms / 16 against 7.8 ms / 128)
    out = {}
    for spot in (1.0, 2.0, 16.0):
        lock, nw = spot_reentry(spot)
        out[spot] = (lock, nw)
        check(lock is not None, f"spot x{spot}: never locked after the re-entry")
    check(out[1.0][1] == 0 and out[2.0][1] == 0 and out[1.0][0] == 0,
          f"no spot: locks on the re-entry frame without a write ({out})")
    check(out[16.0][0] <= 6 * 30, f"spot x16: locked {out[16.0][0]} frames after the re-entry")
    # a short glare at 3.9 / 16: back to 3.9 ms, in the band
    res = {}
    for base in (90.0, 110.0):
        for gl in (10, 40):
            e, g, w, below, nw, _ = glare(gl, base)
            res[(base, gl)] = (e, g, w)
            check(e == 3.9 and cfg.white_low_dn <= w <= cfg.white_high_dn,
                  f"glare {gl} frames, marker {base:.0f} DN: end {e} ms / {g:.0f}, white {w:.0f}")
    # a lamp spot that clips the label at every try: backoff, not a cycle
    e, g, w, below, nw, n_clip = glare(0, 90.0, persistent=True)
    check(nw <= 12 and n_clip <= 40, f"persistent spot: {nw} writes, {n_clip} clipped frames in 15 s")
    # 1.95 ms is left when the gain at 3.9 ms stays >= the floor
    ctl = ce.MarkerExposureController(gain_range=(0, 128))
    ctl.exposure_ms, ctl.gain = 1.95, 32.0
    B = ce.setting_brightness
    for want, clipped in (((3.9, 32.0), False), ((3.9, 16.0), False), ((1.95, 16.0), False),
                          ((1.95, 16.0), True)):
        got = ctl._choose(B(*want), clipped=clipped)
        check(got[0] == want[0] and abs(got[1] - want[1]) < 0.01,
              f"1.95 ms only while 3.9 ms would need a gain below 16: {want} -> {got}")
    # the gain response: below 64 the gain hardly acts (08.10.)
    check(abs(B(3.9, 16) / B(7.8, 128) - 0.146) < 0.005 and abs(B(1.95, 16) / B(7.8, 128) - 0.073) < 0.005
          and abs(ce.gain_for_response(ce.gain_response(40.0)) - 40.0) < 1e-6, "gain response")
    # the frames between a write and its effect are not a light change: lost 1..4 frames after a step
    # under constant light -> no search write
    dn = lambda lin: float(ce.dn_of_linear(lin))
    for lag in (1, 2, 3, 4):
        ctl = ce.MarkerExposureController(gain_range=(0, 128))
        pending = ctl.start(0.0)
        writes, actual = [], (7.8, 128.0)
        L_w, L_f = ce.linear_dn(200.0) / B(7.8, 128), ce.linear_dn(110.0) / B(7.8, 128)
        step_at = None
        # 3 s, 2 s of them after the loss
        for i in range(1, 90):
            for w in writes:
                if i == w[0] + 4:
                    actual = (w[1], w[2])
            if step_at is None and len(writes) >= 2:
                step_at = writes[1][0]
            P = B(*actual)
            fresh = step_at is None or i < step_at + lag
            white = dn(L_w * P)
            pending = _feed(ctl, pending, i, white if fresh else float("nan"), 0.0 if fresh else float("nan"),
                            fresh, dn(L_f * P), writes)
        late = [w for w in writes[2:] if w[3].startswith("search")]
        check(not late, f"lost {lag} frames after a step: search writes {late}")
    print(f"  (l) policy: marker back in a lamp spot x16 locks {out[16.0][0]} frames after it is back "
          f"({out[16.0][1]} writes; x1/x2: at once, 0 writes); glare 10/40 frames at 3.9 / 16: back to "
          f"3.9 ms, white {res[(90.0, 10)][2]:.0f} / {res[(110.0, 10)][2]:.0f} DN; persistent spot: {nw} "
          f"writes, {n_clip} clipped frames in 15 s; 1.95 ms left at gain >= 16; step transition frames "
          f"are no light change (0 search writes)")
    # out of view at a dark setting: the last hull shows the dark background (white 5 DN), the frame stays
    # -> no dark-loss brightening, no write; the frame 3x darker (lamp off) -> brightened at once
    for frame_dn, want_write in ((60.0, False), (30.0, True)):
        ctl = ce.MarkerExposureController(gain_range=(0, 128))
        pending = ctl.start(0.0)
        writes = []
        pending = _feed(ctl, pending, 1, float("nan"), float("nan"), False, 60.0, writes)
        ctl.applied(ce.ExposureCommand(1.95, 16.0, "x", 99), 1)          # tracked at 1.95 / 16
        for i in range(20, 60):
            pending = _feed(ctl, pending, i, 120.0, 0.0, True, 60.0, writes)
        n0 = len(writes)
        for i in range(60, 120):
            pending = _feed(ctl, pending, i, 5.0, 0.0, False, frame_dn if i >= 61 else 60.0, writes)
        got = [w for w in writes[n0:] if w[0] < 120]
        check(bool(got) == want_write and (not got or got[0][3] == "dark-loss"),
              f"lost at 1.95 / 16, hull 5 DN, frame {frame_dn:.0f} DN: writes {got}")
    # a cold start under a lamp in a pose that does not decode: the checkerboard is detected only at
    # the darkest preset (as 08.10. at 1.95 ms / 16) -> the search keeps it 2 s more and makes it home of
    # the slow cycle: most frames there, one probe per 2 s at home
    ctl = ce.MarkerExposureController(gain_range=(0, 128))
    pending = ctl.start(0.0)
    writes, actual = [], (15.6, 64.0)
    L_w = ce.linear_dn(230.0) / B(1.95, 16)          # the label at ~230 DN at 1.95 / 16
    at_det = 0
    for i in range(1, 900):
        for w in writes:
            if i == w[0] + 4:
                actual = (w[1], w[2])
        white = dn(L_w * B(*actual))
        det = 60 if white < 250.0 else 0
        at_det += i >= 300 and det > 0
        pending = _feed(ctl, pending, i, float("nan"), float("nan"), False, dn(0.2 * L_w * B(*actual)),
                        writes, frame_clip=0.05 if white >= 250.0 else 0.004, n_detected=det)
    late = [w for w in writes if w[0] >= 300]
    check(at_det >= 0.7 * 600 and all(w[3] == "search-slow" for w in late)
          and len(late) <= 2 * 600 / 30 / cfg.search_stale_slow_s + 2,
          f"undecodable pose under the lamp: {at_det} of 600 frames at the detecting setting, writes {late}")
    print(f"      cold start, pose that never decodes, detected only at 1.95 / 16: {len(writes) - len(late)} "
          f"writes, then {at_det} of 600 frames there, {len(late)} probe writes "
          f"{sorted(set((w[1], w[2]) for w in late))}")
    # the spot only in the view, the drill under table light outside it (white 90 DN at 7.8 / 128): the
    # frame clips 4 % at 7.8 / 128 (1.2 % at 3.9 / 64), the checkerboard is detected at 3.9 / 64 (27 DN,
    # too dark to decode); a cold engine needs 15 frames in a row at 40-250 DN (the AE frames before the
    # first write count as none) -> the detecting preset is left after 2 s, the slow cycle comes back to
    # 7.8 / 128 and locks
    ctl = ce.MarkerExposureController(gain_range=(0, 128))
    pending = ctl.start(0.0)
    writes, actual = [], (15.6, 64.0)
    L_w = ce.linear_dn(90.0) / B(7.8, 128)
    L_f = ce.linear_dn(110.0) / B(7.8, 128)
    run, first, det_frames = 0, None, 0
    for i in range(1, 600):
        for w in writes:
            if i == w[0] + 4:
                actual = (w[1], w[2])
        P = B(*actual)
        white = dn(L_w * P)
        ok = 40.0 <= white < 250.0 and actual != (15.6, 64.0)
        run = run + 1 if ok else 0
        fresh = ok and (first is not None or run >= 15)
        if fresh and first is None:
            first = i
        det = 40 if 20.0 <= white < 40.0 else 0
        det_frames += det > 0
        fclip = 0.04 if P >= 0.99 * B(7.8, 128) else 0.012 if P >= 0.99 * B(3.9, 64) else 0.004
        pending = _feed(ctl, pending, i, white if fresh else float("nan"), 0.0 if fresh else float("nan"),
                        fresh, dn(L_f * P), writes, frame_clip=fclip, n_detected=det)
    check(first is not None and first <= 12 * 30,
          f"spot in view, drill outside it, detected at 3.9 / 64 only: first pose {first}, writes {writes}")
    print(f"      cold start beside a spot, detected only at 3.9 / 64 (too dark): {det_frames} frames there, "
          f"pose at frame {first} ({len([w for w in writes if first is None or w[0] < first])} writes before)")
    # at the table, the drill out of view: a white object clips 3 % of the frame for 10 frames, then the
    # frame is as before -> one probe cycle, back to 7.8 / 128 and nothing more; 8 % (a light-change search)
    # -> the same once the cycle probes 7.8 / 128 again; a single glint -> nothing
    for n_tr, frac, max_w in ((10, 0.03, 4), (10, 0.08, 6), (1, 0.03, 0)):
        ctl = ce.MarkerExposureController(gain_range=(0, 128))
        pending = ctl.start(0.0)
        writes, actual = [], (7.8, 128.0)
        for i in range(1, 126 + 10 + 600):
            for w in writes:
                if i == w[0] + 4:
                    actual = (w[1], w[2])
            P = B(*actual)
            fresh = i < 90
            white = dn(L_w * P)
            fclip = frac if 126 <= i < 126 + n_tr and P > 0.5 * B(7.8, 128) else 0.0
            pending = _feed(ctl, pending, i, white if fresh else float("nan"), 0.0 if fresh else float("nan"),
                            fresh, dn(L_f * P), writes, frame_clip=fclip)
        after = [w for w in writes if w[0] >= 126]
        tail = [w for w in writes if w[0] >= 126 + 10 + 300]
        check(len(after) <= max_w and not tail and ctl.state == "search-home" and actual == (7.8, 128.0),
              f"table, bright region {n_tr} frames at {frac:.0%} while away: writes {after}, state {ctl.state}, "
              f"{actual}")
        print(f"      table, drill away, {100 * frac:.0f} % of the frame clips for {n_tr} frame(s): {len(after)} "
              f"writes, then {ctl.state} at {actual[0]:g} ms / {actual[1]:.0f}, 0 writes in the last 10 s")
    # under the lamp at 1.95 / 16 the label alone goes dark (a hand shadow on the marker, turned away from
    # the lamp, the drill moved out of the spot): 20 DN at 1.95 / 16, the frame unchanged (0.4 % clip: the
    # lamp in view) -> held, after search_dim_after_s the brightest preset first -> a pose again (old:
    # never). The same at a setting below the brightest without a lamp in view (frame clip 0, the table
    # tracked at 3.9 / 64): nothing is probed
    res = {}
    for start, fclip in (((1.95, 16.0), 0.004), ((3.9, 64.0), 0.0)):
        ctl = ce.MarkerExposureController(gain_range=(0, 128))
        pending = ctl.start(0.0)
        writes = []
        pending = _feed(ctl, pending, 1, float("nan"), float("nan"), False, 40.0, writes)
        ctl.applied(ce.ExposureCommand(start[0], start[1], "x", 99), 1)       # tracked there
        actual = start
        back = None
        for i in range(20, 400):
            for w in writes:
                if i == w[0] + 4:
                    actual = (w[1], w[2])
            rel = B(*actual) / B(*start)
            white = dn(ce.linear_dn(150.0 if i < 100 else 20.0) * rel)
            fresh = 30.0 <= white < 250.0
            if fresh and i >= 100 and back is None:
                back = i - 100
            pending = _feed(ctl, pending, i, white, 0.0, fresh, dn(ce.linear_dn(40.0) * rel), writes,
                            frame_clip=fclip if rel <= 1.0 else 0.05)
        res[start] = (back, [w for w in writes if w[0] >= 100])
    back, writes = res[(1.95, 16.0)]
    dim = [w for w in writes if w[3] == "search-bracket"]
    check(back is not None and 30 * cfg.search_dim_after_s - 2 <= back <= 30 * (cfg.search_dim_after_s + 1)
          and dim and dim[0][1:3] == (7.8, 128.0),
          f"label dark under the lamp: pose again {back} frames after the loss, writes {writes}")
    check(res[(3.9, 64.0)][0] is None and not res[(3.9, 64.0)][1],
          f"label dark at 3.9 / 64 without a lamp in view: writes {res[(3.9, 64.0)][1]}")
    print(f"      label dark at 1.95 / 16 (20 DN), frame unchanged, lamp in view: held, first probe "
          f"{dim[0][1]:g} ms / {dim[0][2]:.0f}, pose again {back} frames after the loss; at 3.9 / 64 without "
          f"a lamp in view: 0 writes in 10 s")
    # 08.10. Lauf 5 (no lamp clip in the frame: below search_dim_frame_clip at the dark settings):
    # into the spot - lost at 3.9 / 29 after a pose at 236 DN (42 % clipped), the hull of the label
    # still at 215 DN (8 % clipped) -> darker while it reads above the band, a pose again (5b: held, then
    # nothing to probe); out of the spot in view - the last pose at 52 DN at 1.95 / 16, the ceiling of
    # the spot there, then 44 DN -> brighter after the loss (5b: held at 1.95 / 16). One pose frame
    # before the loss (no step while tracked), then a pose iff 45 <= white <= 200 (the decoder failed
    # above ~215 and below ~45 DN live)
    res = {}
    for name, start, w0 in (("into", (3.9, 29.0), 215.0), ("out of", (1.95, 16.0), 44.0)):
        ctl = ce.MarkerExposureController(gain_range=(0, 128))
        pending = ctl.start(0.0)
        writes = []
        pending = _feed(ctl, pending, 1, float("nan"), float("nan"), False, 20.0, writes)
        ctl.applied(ce.ExposureCommand(start[0], start[1], "x", 99), 1)
        actual, lost, back = start, None, None
        for i in range(20, 420):
            for w in writes:
                if i == w[0] + 4:
                    actual = (w[1], w[2])
            rel = B(*actual) / B(*start)
            if name == "into":
                at_start, clip0 = (150.0, 0.0) if i < 100 else ((236.0, 0.42) if i == 100 else (w0, 0.08))
            else:
                at_start = 205.0 if i < 100 else (52.0 if i == 100 else w0)
                clip0 = 0.27 if at_start >= 200.0 else 0.0
            white = dn(ce.linear_dn(at_start) * rel)
            clip = clip0 if rel >= 1.0 - 1e-6 else (1.0 if white >= 250.0 else 0.0)
            fresh = 45.0 <= white <= 200.0 or i <= 100
            if not fresh and lost is None and i >= 100:
                lost = i
            if fresh and lost is not None and back is None:
                back = i - lost
            pending = _feed(ctl, pending, i, white, clip, fresh, 2.0, writes, frame_clip=0.0)
        res[name] = (lost, back, [w for w in writes if w[0] >= 100])
        check(back is not None and back <= 30,
              f"{name} the spot: lost at {lost}, pose again {back} frames later, writes {res[name][2]}")
    print(f"      Lauf 5, no lamp clip in the frame: into the spot (lost at 3.9 / 29, hull 215 DN) pose again "
          f"{res['into'][1]} frames after the loss, writes {[(w[1], w[2], w[3]) for w in res['into'][2]]}; out "
          f"of the spot in view (44 DN at 1.95 / 16) {res['out of'][1]} frames, writes "
          f"{[(w[1], w[2], w[3]) for w in res['out of'][2]]}")
    # the loss step and the bracket go by the last pose only while its region still reads like it (08.10.
    # review): under the lamp at 1.95 / 16 (label 185 DN, the frame clips 0.5 % there and 5 % at 7.8 / 128)
    # the last 2 poses fade to 50 DN, then the drill is out of view for 2 s (the hull shows the background,
    # 8 DN) and back -> no dark-loss on the background, a pose again at once (5c before the fix: 3.9 / 77,
    # 7.8 / 128, a search, the pose 0.3 s after the return)
    ctl = ce.MarkerExposureController(gain_range=(0, 128))
    pending = ctl.start(0.0)
    writes = []
    pending = _feed(ctl, pending, 1, float("nan"), float("nan"), False, 3.0, writes)
    ctl.applied(ce.ExposureCommand(1.95, 16.0, "x", 99), 1)
    actual, back = (1.95, 16.0), None
    for i in range(20, 400):
        for w in writes:
            if i == w[0] + 4:
                actual = (w[1], w[2])
        rel = B(*actual) / B(1.95, 16.0)
        lab = 185.0 if i < 148 or i >= 210 else 50.0 if i < 150 else 8.0
        white = dn(ce.linear_dn(lab) * rel)
        fresh = (i < 150 or i >= 210) and 45.0 <= white <= 200.0
        if fresh and i >= 210 and back is None:
            back = i - 210
        fclip = float(0.005 * 10.0 ** (np.log(rel) / np.log(B(7.8, 128.0) / B(1.95, 16.0))))
        pending = _feed(ctl, pending, i, white, 0.0, fresh, dn(ce.linear_dn(3.0) * rel), writes, frame_clip=fclip)
    away = [w for w in writes if 150 <= w[0] < 210]
    check(not away and back is not None and back <= 3,
          f"faded last pose, then out of view under the lamp: writes while away {away}, pose again {back}")
    fade = (away, back)
    # at the table (7.8 / 128, no frame clip, light unchanged) a single last pose above the band (a glint,
    # a tilt), then the drill out of view for 20 s with a 70 DN background in the hull: nothing (5c before
    # the fix: a bracket after 3 s, 12 writes in 20 s)
    res = {}
    for last, lclip in ((180.0, 0.0), (150.0, 0.35)):
        ctl = ce.MarkerExposureController(gain_range=(0, 128))
        pending = ctl.start(0.0)
        writes = []
        for i in range(1, 700):
            fresh = i <= 100
            white = (last if i == 100 else 90.0) if fresh else (70.0 if i < 190 else float("nan"))
            pending = _feed(ctl, pending, i, white, lclip if i == 100 else 0.0, fresh, 60.0, writes)
        res[last] = [w for w in writes if w[0] > 100]
        check(not res[last], f"table, last pose {last:.0f} DN / {lclip:.0%} clipped, then away: writes {res[last]}")
    # tracked at 3.9 / 64 without a lamp, the label fades to 40 DN and does not decode any more, its region
    # keeps reading so (dark-loss brightens it, still no pose) -> the bracket on the label alone after
    # search_dim_after_s, one round of probes, then home and nothing more
    ctl = ce.MarkerExposureController(gain_range=(0, 128))
    pending = ctl.start(0.0)
    writes = []
    pending = _feed(ctl, pending, 1, float("nan"), float("nan"), False, 20.0, writes)
    ctl.applied(ce.ExposureCommand(3.9, 64.0, "x", 99), 1)
    actual = (3.9, 64.0)
    for i in range(20, 1200):
        for w in writes:
            if i == w[0] + 4:
                actual = (w[1], w[2])
        rel = B(*actual) / B(3.9, 64.0)
        fresh = i <= 100
        lab = 100.0 if i < 100 else 40.0
        white = dn(ce.linear_dn(lab) * rel) if i < 190 else float("nan")
        pending = _feed(ctl, pending, i, white, 0.0, fresh, dn(ce.linear_dn(20.0) * rel), writes)
    bracket = [w for w in writes if w[3] == "search-bracket"]
    late = [w for w in writes if w[0] >= 100 + 30 * 20]
    check(bracket and len(bracket) <= len(cfg.search_presets) * 2 and not late and ctl.state == "search-home"
          and actual == bracket[-1][1:3],
          f"bracket on the label alone: writes {[w for w in writes if w[0] > 100]}, state {ctl.state}")
    print(f"      faded last pose (50 DN) under the lamp, then out of view 2 s: writes while away {fade[0]}, pose "
          f"again {fade[1]} frames after the return; table, a single last pose at 180 DN / 150 DN 35 % clipped, "
          f"then away 20 s: {len(res[180.0])} / {len(res[150.0])} writes; label alone out of the band, never "
          f"decoded: {len(bracket)} bracket writes (last {bracket[-1][1]:g} ms / {bracket[-1][2]:.0f} at frame "
          f"{bracket[-1][0]}), then {ctl.state}")
    # a cold start in bright light: the search goes on darker, it does not start over at the first
    # preset (clipped frame) after the hold
    plant_light = 30.0
    ctl, log, steps = run_loop(lambda k: plant_light, lambda k: True, 300)
    first = next((k for k, fr, *_ in log if fr), None)
    again = [s for s in steps[1:] if s[1] == 7.8 and s[2] == 128.0]
    check(first is not None and not again, f"cold start x30: first pose {first}, back to 7.8 / 128: {again}")


# ============================================================
# (m) control with a failing camera
# ============================================================

class FakeCamera:
    """start_color_control / stop_color_control like RealSenseCameraSource, on a FakeSensor."""

    def __init__(self, sensor):
        self.sensor = sensor
        self.frame = 100
        self.commanded = None
        self.actuator = None

    def color_option_range(self, name):
        return {"min": 0.0, "max": 128.0, "step": 1.0, "default": 64.0}

    @property
    def color_control_busy(self):
        return self.actuator is not None and self.actuator.alive

    def start_color_control(self):
        self.commanded = ce.CommandedColorExposure(True, None)
        self.actuator = ce.ColorExposureActuator(
            self.sensor, OPTS, lambda: self.frame, self.commanded,
            start_settings={"enable_auto_exposure": True, "exposure": 156.0, "gain": 64.0})
        self.actuator.start()
        return self.actuator

    def stop_color_control(self, actuator, timeout_s=1.0):
        if not actuator.stop(restore=True, timeout_s=timeout_s):
            return {"late": True}
        self.actuator = None
        return actuator.restored


class FakeWorker:
    def set_meter(self, meter):
        self.meter = meter


def _drive(control, cam, n, until=None):
    """n meterings (fresh pose; the white alternates above and below the band every 15 frames, so
    commands keep coming), the camera's frame index moving; stops early when until() is true.
    Returns the number of meterings fed."""
    for i in range(n):
        cam.frame += 1
        white = 200.0 if (cam.frame // 15) % 2 == 0 else 30.0
        control.step(ce.Metering(frame_index=cam.frame, t_frame=cam.frame / 30.0, fresh=True, white_dn=white,
                                 clip_frac=0.0, n_samples=2000, hull_from_pose=True, frame_clip_frac=0.0,
                                 frame_median_dn=100.0, speed_px=0.0))
        time.sleep(0.003)
        if control.failure or (until is not None and until()):
            return i + 1
    return n


def test_control_failures():
    # the first write (AE off) fails: failure, the stop leaves the AE on
    cam = FakeCamera(FakeSensor(fail_on="enable_auto_exposure"))
    ctl = ce.MarkerExposureControl(cam, lambda r: True)
    check(ctl.start(FakeWorker(), 0.0), "start")
    _drive(ctl, cam, 60)
    ctl.stop()
    check("first write" in ctl.failure, f"first write fails -> failure ({ctl.failure})")
    check(cam.sensor.values["enable_auto_exposure"] == 1.0, "AE on after the stop")
    # one gain write fails later: dropped, the control goes on (not frozen)
    sensor = FakeSensor()
    cam = FakeCamera(sensor)
    ctl = ce.MarkerExposureControl(cam, lambda r: True)
    ctl.start(FakeWorker(), 0.0)
    _drive(ctl, cam, 40)
    sensor.fail_on = {"gain"}
    _drive(ctl, cam, 200, until=lambda: len(ctl.failures) >= 1)
    sensor.fail_on = set()
    n_before = ctl.n_commands
    _drive(ctl, cam, 90)
    one = (ctl.failure, len(ctl.failures), ctl.n_commands - n_before)
    ctl.stop()
    check(not one[0] and one[1] == 1 and one[2] >= 2,
          f"one failed write: control goes on (failure '{one[0]}', failed {one[1]}, commands after {one[2]})")
    # every gain write fails: failure after MAX_FAILED_ROW in a row
    sensor = FakeSensor()
    cam = FakeCamera(sensor)
    ctl = ce.MarkerExposureControl(cam, lambda r: True)
    ctl.start(FakeWorker(), 0.0)
    _drive(ctl, cam, 40)
    sensor.fail_on = {"gain"}
    _drive(ctl, cam, 300)
    at = ctl.failure
    sensor.fail_on = set()
    summ2 = ctl.stop()
    check("in a row" in at and summ2["failure"] and not summ2["restore_problems"],
          f"writes failing in a row -> failure ({ctl.failure}), restore ok {summ2['restored']}")
    # a restore that cannot switch the AE on is a restore problem
    sensor = FakeSensor()
    cam = FakeCamera(sensor)
    ctl = ce.MarkerExposureControl(cam, lambda r: True)
    ctl.start(FakeWorker(), 0.0)
    _drive(ctl, cam, 15)
    sensor.fail_on = {"enable_auto_exposure"}
    summ3 = ctl.stop()
    check("AE not back on" in summ3["restore_problems"] and "restore write failed" in summ3["restore_problems"],
          f"restore problems: {summ3['restore_problems']}")
    print(f"  (m) control: first write fails -> failure; 1 failed write -> dropped, {one[2]} commands after "
          f"it; failing in a row -> failure after {ce.MarkerExposureControl.MAX_FAILED_ROW}; AE not back "
          f"on -> restore problem {summ3['restore_problems']}")


# ============================================================
# (n) camera source after the stop
# ============================================================

class SlowSensor(FakeSensor):
    def set_option(self, opt, value):
        if getattr(self, "hang", None) is not None:
            self.hang.wait(5.0)
        super().set_option(opt, value)


def test_source_after_stop():
    from tracking.hydramarker.camera_setup import RealSenseCameraSource

    src = RealSenseCameraSource(CameraConfig())
    sensor = SlowSensor()
    md = SimpleNamespace(frame_timestamp="ft", sensor_timestamp="st")
    src._rs = SimpleNamespace(option=OPTS, frame_metadata_value=md)
    src._profile = object()
    src._color_sensor_cache = sensor
    src._color_exposure = ColorExposureState(True, None)
    src.calibration_info = {"realsense_enable_auto_exposure": True, "realsense_exposure": 156.0,
                            "realsense_gain": 64.0}
    frame = SimpleNamespace(get_timestamp=lambda: 1000.0, supports_frame_metadata=lambda k: False)

    def meta():
        src._frame_index += 1
        return src._frame_metadata(frame, None)["color_exposure"]

    act = src.start_color_control()
    act.submit(ce.ExposureCommand(7.8, 128.0, "start", 1))
    time.sleep(0.05)
    for _ in range(10):
        meta()
    check(meta()["exposure_ms"] == 7.8, "manual while the control runs")
    out = src.stop_color_control(act)
    check(out.get("enable_auto_exposure") == 1.0, f"restored {out}")
    trans = [meta() for _ in range(ce.LATENCY_FRAMES + 1)]
    check(all("previous" in t and t["previous"]["exposure_ms"] == 7.8 and t["auto"] for t in trans),
          f"the restore's frames in transition (mean): {trans}")
    after = meta()
    check(after == {"auto": True, "exposure_ms": None}, f"then the state read at the start: {after}")
    # a restore stuck in set_option: late, busy, no new control; the camera stop joins it
    act = src.start_color_control()
    act.submit(ce.ExposureCommand(3.9, 16.0, "start", 1))
    time.sleep(0.05)
    sensor.hang = threading.Event()
    out = src.stop_color_control(act, timeout_s=0.1)
    busy = src.color_control_busy
    second = src.start_color_control()
    check(out == {"late": True} and busy and second is None, f"late stop: {out}, busy {busy}, {second}")
    sensor.hang.set()
    act._thread.join(2.0)
    check(not src.color_control_busy and sensor.values["enable_auto_exposure"] == 1.0,
          "the late restore runs when the call returns")
    sensor.hang = None
    print("  (n) camera source: after the stop the restore's frames in transition (mean), then the AE "
          "state as at the start; a restore stuck in set_option: late, source busy, no new control; "
          "it restores when the call returns")


# ============================================================
# (e) meter
# ============================================================

def synthetic_marker(h=1080, w=1920, white=180, black=25, clip_rows=0):
    img = np.full((h, w), 90, np.uint8)
    y0, x0, cs = 400, 800, 30
    pts = []
    for r in range(9):
        for c in range(11):
            v = white if (r + c) % 2 == 0 else black
            img[y0 + r * cs:y0 + (r + 1) * cs, x0 + c * cs:x0 + (c + 1) * cs] = v
    if clip_rows:
        top = img[y0:y0 + clip_rows * cs, x0:x0 + 11 * cs]
        top[top == white] = 255
    for r in range(1, 9):
        for c in range(1, 11):
            pts.append(SimpleNamespace(uv=(x0 + c * cs, y0 + r * cs)))
    return img, pts


def test_meter():
    fresh = {"v": True}
    meter = ce.MarkerMeter(lambda r: fresh["v"])
    img, pts = synthetic_marker()
    res = SimpleNamespace(corners=pts)
    m = meter.measure(img, res, 1, 0.0)
    check(abs(m.white_dn - 180) <= 1 and abs(m.black_dn - 25) <= 1 and m.clip_frac == 0.0,
          f"levels {m.white_dn} {m.black_dn} {m.clip_frac}")
    check(m.hull_from_pose and m.n_samples > 1000, f"samples {m.n_samples}")
    bgra = np.dstack([img, img, img, np.full_like(img, 255)])
    m2 = meter.measure(bgra, res, 2, 0.03)
    check(abs(m2.white_dn - m.white_dn) <= 1 and abs(m2.black_dn - m.black_dn) <= 1, "BGRA == grey")
    imgc, _ = synthetic_marker(clip_rows=4)
    m3 = meter.measure(imgc, res, 3, 0.06)
    check(0.3 < m3.clip_frac < 0.6, f"clipped share {m3.clip_frac}")
    fresh["v"] = False
    m4 = meter.measure(imgc, SimpleNamespace(corners=[]), 4, 1.0)
    check(not m4.hull_from_pose and np.isfinite(m4.clip_frac), "last hull metered without a pose")
    m5 = meter.measure(imgc, SimpleNamespace(corners=[]), 5, 10.0)
    check(not np.isfinite(m5.white_dn), "last hull expires after keep_s")
    check(np.isfinite(m5.frame_median_dn) and m5.frame_clip_frac >= 0, "frame grid")
    # cost on BGRA with a pose
    fresh["v"] = True
    big, pts2 = synthetic_marker()
    big4 = np.dstack([big, big, big, np.full_like(big, 255)])
    ts = []
    for i in range(300):
        t = time.perf_counter()
        meter.measure(big4, SimpleNamespace(corners=pts2), 10 + i, 2.0 + i / 30)
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts[30:])
    check(np.median(ts) < 0.5, f"meter cost median {np.median(ts):.3f} ms")
    print(f"  (e) meter: white/black/clip right, BGRA == grey, last hull, expiry; cost on 1920x1080 BGRA "
          f"(hull 300 x 240 px) median {np.median(ts):.3f} ms, p95 {np.percentile(ts, 95):.3f} ms")
    return float(np.median(ts)), float(np.percentile(ts, 95))


# ============================================================
# (f) worker hook
# ============================================================

class FakeTracker:
    def __init__(self):
        self.n_reset = 0

    def process_frame(self, img, run_detection=True, ir_left=None, ir_right=None, t_frame=None, **kw):
        time.sleep(0.002)
        return SimpleNamespace(success=True, corners=[], k=int(img[0, 0, 0]))

    def reset(self):
        self.n_reset += 1


class CountingMeter:
    def __init__(self, fail=False):
        self.n = 0
        self.resets = 0
        self.fail = fail

    def measure(self, image, result, frame_index, t_frame):
        self.n += 1
        if self.fail:
            raise ValueError("meter bug")
        return ("m", frame_index, int(image[0, 0, 0]))

    def reset(self):
        self.resets += 1


def test_worker_meter():
    meter = CountingMeter()
    w = TrackingWorker(FakeTracker(), pool_size=5, meter=meter)
    w.start()
    outs = []
    for i in range(30):
        img = np.full((8, 8, 4), i % 250, np.uint8)
        w.submit(SimpleNamespace(image_bgr=img, ir_left=None, ir_right=None, frame_index=i, metadata={}),
                 i / 30.0)
        time.sleep(0.004)
        for o in w.take_outputs():
            outs.append(o)
            w.release(o.slot)
        if i == 15:
            w.set_meter(None)
        if i == 20:
            w.request_reset()
            w.set_meter(meter)
    time.sleep(0.05)
    for o in w.take_outputs():
        outs.append(o)
        w.release(o.slot)
    w.stop()
    with_m = [o for o in outs if o.metering is not None]
    check(all(o.metering[1] == o.frame_index and o.metering[2] == o.frame_index % 250 for o in with_m),
          "metering of the output's own frame")
    check(any(o.metering is None for o in outs), "set_meter(None) stops it")
    check(meter.resets >= 1, "meter reset with the tracker")
    w2 = TrackingWorker(FakeTracker(), pool_size=5, meter=CountingMeter(fail=True))
    w2.start()
    w2.submit(SimpleNamespace(image_bgr=np.zeros((8, 8, 4), np.uint8), ir_left=None, ir_right=None,
                              frame_index=1, metadata={}), 0.0)
    time.sleep(0.05)
    o = w2.take_outputs()
    w2.stop()
    check(len(o) == 1 and o[0].error is None and o[0].metering is None and w2.n_meter_errors == 1,
          "a raising meter costs only the metering")
    print(f"  (f) worker: metering of the output's own frame ({len(with_m)}/{len(outs)} outputs), off by "
          f"set_meter(None), reset with the tracker, a raising meter -> no metering, tracking goes on")


# ============================================================
# (g) display
# ============================================================

def test_display():
    br = ce.DisplayBrightener()
    dark = np.full((1080, 1920, 4), 40, np.uint8)
    for _ in range(40):
        br.update(dark)
    check(br.k > 2.0, f"dark display gain {br.k}")
    img = dark.copy()
    ptr = img.__array_interface__["data"][0]
    check(br.apply(img) and img.__array_interface__["data"][0] == ptr and img[0, 0, 1] > 80, "in place")
    br2 = ce.DisplayBrightener()
    bright = np.full((1080, 1920, 4), 150, np.uint8)
    for _ in range(10):
        br2.update(bright)
    check(br2.lut() is None, "bright display unchanged")
    ts = []
    for i in range(100):
        im = dark.copy()
        t = time.perf_counter()
        br.update(im)
        br.apply(im)
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts[10:])
    print(f"  (g) display LUT: dark x{br.k:.2f}, bright unchanged; update + LUT in place on 1920x1080 BGRA "
          f"median {np.median(ts):.2f} ms, p95 {np.percentile(ts, 95):.2f} ms")
    return float(np.median(ts)), float(np.percentile(ts, 95))


# ============================================================
# (h) / (i) / (j) camera start and state
# ============================================================

class FakeRsSensor:
    NAMES = ("enable_auto_exposure", "exposure", "gain", "enable_auto_white_balance", "white_balance",
             "auto_exposure_priority", "power_line_frequency", "brightness", "contrast", "saturation", "hue",
             "sharpness", "gamma", "backlight_compensation")

    def __init__(self, values):
        self.v = dict(values)
        self.writes = []

    def get_info(self, _):
        return "RGB Camera"

    def supports(self, opt):
        return opt in self.v

    def get_option(self, opt):
        return self.v[opt]

    def set_option(self, opt, value):
        self.writes.append((opt, value))
        self.v[opt] = float(value)

    def get_option_range(self, opt):
        return SimpleNamespace(min=0.0, max=10000.0, step=1.0, def_=0.0)


def test_start_after_crash():
    crashed = {"enable_auto_exposure": 0.0, "exposure": 39.0, "gain": 16.0, "enable_auto_white_balance": 1.0,
               "white_balance": 4600.0, "auto_exposure_priority": 0.0, "power_line_frequency": 1.0,
               "brightness": 0.0, "contrast": 50.0, "saturation": 64.0, "hue": 0.0, "sharpness": 50.0,
               "gamma": 300.0, "backlight_compensation": 0.0}
    sensor = FakeRsSensor(crashed)
    rs = SimpleNamespace(option=SimpleNamespace(**{n: n for n in FakeRsSensor.NAMES}),
                         camera_info=SimpleNamespace(name="name"),
                         stream=SimpleNamespace(color="color"),
                         roi_sensor=lambda s: (_ for _ in ()).throw(RuntimeError("no roi in the fake")))
    profile = SimpleNamespace(get_device=lambda: SimpleNamespace(query_sensors=lambda: [sensor]),
                              get_stream=lambda s: (_ for _ in ()).throw(RuntimeError("no stream")))
    cfg = CameraConfig(width=1920, height=1080, fps=30)
    details = _configure_realsense_color_sensor(profile, cfg, rs)
    check(sensor.v["enable_auto_exposure"] == 1.0 and sensor.v["exposure"] == 156.0
          and sensor.v["gain"] == 64.0,
          f"K1 start restores: {sensor.v}")
    check([w[0] for w in sensor.writes] == ["exposure", "gain", "enable_auto_exposure"],
          f"K1 writes {sensor.writes}")
    check("realsense_color_mismatch" not in details, f"mismatch {details.get('realsense_color_mismatch')}")
    st = ColorExposureState(True, 3.9)
    check(st.exposure_ms is None and st.snapshot() == {"auto": True, "exposure_ms": None}, "AE: no number")
    check(ColorExposureState(False, 7.8).exposure_ms == 7.8, "manual keeps its number")
    check(CameraConfig().realsense_color_exposure_mode == "auto", "default auto")
    try:
        CameraConfig(realsense_color_exposure_mode="bogus").validate_common()
        check(False, "invalid mode accepted")
    except ValueError:
        pass
    print("  (h) start after a crash in manual 3.9 ms / 16 (AE off): K1 writes 156, 64, AE on -> device as "
          "configured, no mismatch")
    print("  (i) ColorExposureState: None under the AE (stale register), the number in manual")
    print("  (j) CameraConfig.realsense_color_exposure_mode: default 'auto', 'bogus' rejected")


def test_meter_result_type():
    # the result the Live page and the evaluation script hand the meter (TrackingWorker ->
    # HydraTracker.process_frame) is the Python TrackerResult: it has the detected corners as a list,
    # no count - the meter must see them (08.10. review: n_detected was 0 live)
    from tracking.hydramarker.tracker import DetectedCorner, TrackerMode, TrackerResult

    img, _ = synthetic_marker()
    res = TrackerResult(success=False, mode=TrackerMode.DETECTING,
                        detection_corners=[DetectedCorner(0, i, (800.0 + 30 * i, 430.0)) for i in range(12)])
    m = ce.MarkerMeter(lambda r: False).measure(img, res, 1, 0.0)
    check(m.n_detected == 12 and not hasattr(res, "detection_corner_count"),
          f"TrackerResult with 12 detected corners: n_detected {m.n_detected}")
    raw = SimpleNamespace(detection_corner_count=34)
    check(ce.MarkerMeter.detected_count(raw) == 34 and ce.MarkerMeter.detected_count(SimpleNamespace()) == 0,
          "C++ result: detection_corner_count; nothing: 0")
    print("  (e) meter on tracker.TrackerResult: n_detected = len(detection_corners) (12); raw C++ result: "
          "detection_corner_count")


def test_tracker_timing():
    from tracking.hydramarker.config import TrackerConfig
    from tracking.hydramarker.tracker import HydraTracker, ir_time_offset_ms

    fake = SimpleNamespace(config=TrackerConfig())
    base = {"color_time_ms": 1000.0, "ir_exposure_us": 8000.0}
    _, ae = HydraTracker.ir_frame_timing(fake, dict(base, color_exposure={"auto": True, "exposure_ms": None}))
    _, m78 = HydraTracker.ir_frame_timing(
        fake, dict(base, color_exposure={"auto": False, "exposure_ms": 7.8}))
    _, tr = HydraTracker.ir_frame_timing(fake, dict(base, color_exposure={
        "auto": False, "exposure_ms": 7.8, "previous": {"auto": True, "exposure_ms": None}}))
    check(abs(ae - (8.0 - 24.0) / 2) < 1e-9 and abs(m78 - 0.1) < 1e-9, f"offsets {ae} {m78}")
    check(abs(tr - 0.5 * (ae + m78)) < 1e-9, f"transition mean {tr}")
    check(abs(ir_time_offset_ms(8.0, True, 3.9) - ae) < 1e-9, "AE ignores a stale number")
    print(f"  (k) ir_frame_timing: AE {ae:+.2f} ms, 7.8 ms {m78:+.2f} ms, transition {tr:+.2f} ms (mean)")


def main() -> int:
    test_commanded_state()
    test_every_frame_e_rgb()
    test_controller_closed_loop()
    test_policy()
    test_actuator()
    test_control_failures()
    test_source_after_stop()
    test_meter()
    test_worker_meter()
    test_display()
    test_start_after_crash()
    test_meter_result_type()
    test_tracker_timing()
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
