# -*- coding: utf-8 -*-
"""
test_display_exposure.py

Checks overlay.tools.display_exposure.AutoExposureLook (the Live page's
display LUT while the marker exposure control runs) on a plant: a fixed
scene (a radiance ramp in the AE field, a dark border) exposed at a
setting -> linear x setting_brightness, the camera's tone curve, clip 255
- no camera, no Qt.

(a) under the AE / without a known setting: no LUT (k 1)
(b) a static scene at 7.8 ms / 128 that the AE would bring to TARGET_DN:
    the shown AE field reads TARGET_DN; at the table (scene too dark for
    the AE) the ceiling holds it darker, as the AE does
(c) controller steps (x2 exposure, x0.48 gain) within K_MAX: the frame's
    own setting in its metadata -> the shown field does not move by more
    than 3 % on the step frame (a raw frame jumps by the step); a
    transition frame that carries both settings is told apart by its
    image, so is the switch frame that still has the old gain under the
    new setting's name (7.8 / 128 -> 3.9 / 64 read as 3.9 / 128); with
    the setting 8 frames late (no metadata per frame) it would jump; a
    search probe x0.15 beyond K_MAX darkens the view (by design, k is
    capped)
(d) a light change (lamp on x4): the shown level follows at the AE's pace
    (half-way within ~4-5 frames), not at once
(e) K_MAX: settings far darker than the AE are shown as bright as K_MAX
    allows, never brighter
(f) update + LUT pass (as the copy into a persistent buffer) on a
    1920x1080 BGRA frame: the LUT is the tone curve x k, the alpha byte
    stays 255 above and below k 1; cost
(g) the control's start and stop at the table (the AE at its ceiling):
    the 3 frames after the first write that still show the AE image and
    the 3 after the restore that still show the manual one keep the shown
    field within 10 % (they carry both settings, AE and manual)
(h) the control's start at the table when the first written frame still
    has the AE's gain under the new exposure (08.10.): only that frame is
    off; the manual frames after it are not taken for AE frames

Run from src/:

    python -m overlay.tests.test_display_exposure
"""

from __future__ import annotations

import sys
import time

import numpy as np

from overlay.tools.display_exposure import AutoExposureLook
from tracking.hydramarker.color_exposure import (
    _LIN,
    dn_of_linear,
    linear_dn,
    setting_brightness,
)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


H, W = 1080, 1920
B0 = setting_brightness(7.8, 128.0)


def _scene(level: float) -> np.ndarray:
    """Linear radiance per pixel (units: brightness at B0), a ramp around
    level in the AE field and a darker border."""
    x = np.linspace(0.5, 1.5, W)[None, :]
    y = np.linspace(0.8, 1.2, H)[:, None]
    lin = level * x * y
    lin[: H // 8] *= 0.3
    return lin


def _frame(lin: np.ndarray, brightness: float) -> np.ndarray:
    """The camera image of the scene at a setting (BGRA, grey, alpha 255)."""
    dn = np.clip(np.round(dn_of_linear(lin * brightness / B0)), 0, 255).astype(np.uint8)
    img = np.repeat(dn[:, :, None], 4, axis=2)
    img[..., 3] = 255
    return np.ascontiguousarray(img)


def _ex(ms, gain, prev=None):
    d = {"auto": False, "exposure_ms": ms, "gain": gain}
    if prev is not None:
        d["previous"] = {"auto": False, "exposure_ms": prev[0], "gain": prev[1]}
    return d


def _field(img: np.ndarray) -> float:
    return float(img[H // 4:3 * H // 4:4, W // 4:3 * W // 4:4, 1].mean())


def _shown(look: AutoExposureLook, img: np.ndarray, ex) -> float:
    look.update(img, ex)
    out = img.copy()
    look.apply(out)
    return _field(out)


def _level_for(target_dn: float, brightness: float) -> float:
    """Scene level whose AE field mean at brightness is ~target_dn."""
    lo, hi = 1e-4, 10.0
    for _ in range(50):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if _field(_frame(_scene(mid), brightness)) < target_dn else (lo, mid)
    return (lo * hi) ** 0.5


# ============================================================

def test_auto_passthrough():
    look = AutoExposureLook()
    img = _frame(_scene(0.05), B0)
    for ex in (None, {"auto": True, "exposure_ms": None, "gain": None},
               {"auto": False, "exposure_ms": 7.8, "gain": None}):
        check(look.update(img, ex) == 1.0 and look.lut() is None, f"no LUT for {ex}")
    print("  (a) AE / unknown setting: shown as it is")


def test_target_and_ceiling():
    look = AutoExposureLook()
    # a scene the AE brings to its target with 2x the brightness of 7.8/128
    lvl = _level_for(look.TARGET_DN, 2.0 * B0)
    img = _frame(_scene(lvl), B0)
    for _ in range(40):
        s = _shown(look, img, _ex(7.8, 128.0))
    check(abs(s - look.TARGET_DN) < 3.0, f"shown field {s:.1f} vs target {look.TARGET_DN}")
    check(abs(look.k - 2.0) < 0.15, f"k {look.k:.2f} for an AE at 2x")
    # the table: the AE would need 10x but stops at CEILING
    look2 = AutoExposureLook()
    lvl2 = _level_for(look.TARGET_DN, 10.0 * B0)
    img2 = _frame(_scene(lvl2), B0)
    for _ in range(60):
        s2 = _shown(look2, img2, _ex(7.8, 128.0))
    k_ceiling = look.CEILING / B0
    check(abs(look2.k - k_ceiling) < 0.05 * k_ceiling and s2 < look.TARGET_DN - 10,
          f"ceiling: k {look2.k:.2f} (want {k_ceiling:.2f}), field {s2:.1f}")
    print(f"  (b) AE target: field {s:.1f} DN (target {look.TARGET_DN:.0f}), k {look.k:.2f}; "
          f"table: k {look2.k:.2f} = ceiling, field {s2:.1f} DN (darker, as the AE)")


def test_steps():
    lvl = _level_for(AutoExposureLook.TARGET_DN, 2.0 * B0)
    lin = _scene(lvl)
    seq = ([(7.8, 128.0)] * 15 + [(3.9, 128.0)] * 12 + [(3.9, 64.0)] * 12 + [(7.8, 64.0)] * 12
           + [(7.8, 128.0)] * 12)
    for late in (False, True):
        look = AutoExposureLook()
        shown, raw, prev = [], [], seq[0]
        for i, st in enumerate(seq):
            img = _frame(lin, setting_brightness(*st))
            told = seq[max(0, i - 8)] if late else st         # late: the setting 8 frames before
            ex = _ex(*told)
            if not late and i > 0 and seq[i - 1] != st:
                ex = _ex(*st, prev=seq[i - 1])               # transition: both settings
            shown.append(_shown(look, img, ex))
            raw.append(_field(img))
        steps = [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]
        jump = max(abs(np.log(shown[i] / shown[i - 1])) for i in steps)
        jump_raw = max(abs(np.log(raw[i] / raw[i - 1])) for i in steps)
        if not late:
            check(jump < np.log(1.03), f"shown field jumps x{np.exp(jump):.3f} at a step")
            check(jump_raw > np.log(1.8), f"raw jumps only x{np.exp(jump_raw):.2f}")
            j_ok, j_raw = jump, jump_raw
        else:
            check(jump > np.log(1.3), f"8 frames late should jump: x{np.exp(jump):.2f}")
            j_late = jump
    # a transition frame told apart by its image: both orders
    look = AutoExposureLook()
    img_a = _frame(lin, setting_brightness(7.8, 128.0))
    img_b = _frame(lin, setting_brightness(3.9, 64.0))
    for _ in range(30):
        s_a = _shown(look, img_a, _ex(7.8, 128.0))
    s_old = _shown(look, img_a, _ex(3.9, 64.0, prev=(7.8, 128.0)))   # commanded new, image still old
    s_new = _shown(look, img_b, _ex(3.9, 64.0, prev=(7.8, 128.0)))
    check(abs(np.log(s_old / s_a)) < np.log(1.03) and abs(np.log(s_new / s_a)) < np.log(1.03),
          f"transition frames {s_old:.1f} / {s_new:.1f} vs {s_a:.1f}")
    look = AutoExposureLook()
    for _ in range(30):
        s_a = _shown(look, img_a, _ex(7.8, 128.0))
    s_sw = _shown(look, _frame(lin, setting_brightness(3.9, 128.0)), _ex(3.9, 64.0))   # no "previous"
    s_16 = _shown(look, _frame(lin, setting_brightness(3.9, 64.0)), _ex(3.9, 64.0))
    check(abs(np.log(s_sw / s_a)) < np.log(1.03) and abs(np.log(s_16 / s_a)) < np.log(1.03),
          f"switch frame {s_sw:.1f}, next {s_16:.1f} vs {s_a:.1f}")
    # a search probe 7.8/64 -> 1.95/16 (x0.15) needs k 27 > K_MAX: the view darkens for the probe
    look = AutoExposureLook()
    for _ in range(30):
        s_64 = _shown(look, _frame(lin, setting_brightness(7.8, 64.0)), _ex(7.8, 64.0))
    s_probe = _shown(look, _frame(lin, setting_brightness(1.95, 16.0)), _ex(1.95, 16.0, prev=(7.8, 64.0)))
    check(look.k == look.K_MAX and s_probe < s_64, f"probe: k {look.k}, {s_probe:.1f} vs {s_64:.1f}")
    print(f"  (c) steps x2 / x0.48: shown field moves at most x{np.exp(j_ok):.3f} (raw x{np.exp(j_raw):.2f}); "
          f"the setting 8 frames late: x{np.exp(j_late):.2f}; transition frames told apart by the image; "
          f"a x0.15 probe beyond K_MAX: {s_64:.0f} -> {s_probe:.0f} DN")


def test_pace():
    look = AutoExposureLook()
    lvl = _level_for(look.TARGET_DN, 0.5 * B0)
    for _ in range(40):
        s0 = _shown(look, _frame(_scene(lvl), setting_brightness(1.95, 128.0)), _ex(1.95, 128.0))
    lamp = _scene(lvl * 4.0)
    seq = [_shown(look, _frame(lamp, setting_brightness(1.95, 128.0)), _ex(1.95, 128.0)) for _ in range(40)]
    first = seq[0]
    half = next(i for i, s in enumerate(seq) if s - s0 < 0.5 * (first - s0))
    check(first > s0 + 40 and 2 <= half <= 8 and abs(seq[-1] - look.TARGET_DN) < 6,
          f"lamp on: {s0:.0f} -> {first:.0f} -> half at frame {half} -> {seq[-1]:.0f}")
    print(f"  (d) lamp on x4: shown {s0:.0f} -> {first:.0f} DN, back half-way in {half} frames, "
          f"{seq[-1]:.0f} DN after 40")


def test_k_max():
    look = AutoExposureLook()
    lvl = _level_for(look.TARGET_DN, 2.0 * B0)
    img = _frame(_scene(lvl), setting_brightness(0.98, 16.0))
    for _ in range(80):
        _shown(look, img, _ex(0.98, 16.0))
    check(abs(look.k - look.K_MAX) < 1e-9, f"k {look.k} at a setting far below the AE")
    print(f"  (e) setting far below the AE: k held at K_MAX {look.K_MAX:g}")


def test_cost():
    look = AutoExposureLook()
    src = _frame(_scene(0.03), B0)
    dst = np.empty_like(src)
    ts = []
    for i in range(220):
        t = time.perf_counter()
        look.update(src, _ex(7.8, 128.0))
        looked = look.apply(src, dst)
        ts.append((time.perf_counter() - t) * 1000.0)
    ts = np.array(ts[20:])
    check(looked and look.k > 1.0, "a LUT in this case")
    check(abs(float(dst[H // 2, W // 2, 1]) - float(np.clip(np.round(
        dn_of_linear(_LIN[src[H // 2, W // 2, 1]] * look.k)), 0, 255))) <= 1, "LUT = tone curve x k")
    check(bool((dst[..., 3] == 255).all()), f"alpha at k {look.k:.2f}")
    check(abs(linear_dn(255) - 1.0) < 1e-9, "tone curve")
    # a setting brighter than the AE (k < 1): the alpha byte still 255
    dark = AutoExposureLook()
    src2 = _frame(_scene(_level_for(dark.TARGET_DN, 0.5 * B0)), B0)
    for _ in range(40):
        dark.update(src2, _ex(7.8, 128.0))
    check(dark.apply(src2, dst) and dark.k < 0.6, f"k {dark.k:.2f} below 1")
    check(bool((dst[..., 3] == 255).all()), f"alpha at k {dark.k:.2f}")
    check(int(dst[H // 2, W // 2, 1]) == int(dark.lut()[src2[H // 2, W // 2, 1]]), "four-channel LUT = LUT")
    print(f"  (f) update + LUT pass into the frame buffer, 1920x1080 BGRA: median {np.median(ts):.2f} ms, "
          f"p95 {np.percentile(ts, 95):.2f} ms (n {len(ts)}); alpha 255 at k {look.k:.2f} and {dark.k:.2f}")


def test_start_stop():
    look = AutoExposureLook()
    lvl = _level_for(look.TARGET_DN, 10.0 * B0)
    lin = _scene(lvl)
    ae_img = _frame(lin, look.CEILING)                 # the AE at its ceiling
    man_img = _frame(lin, B0)                           # 7.8 ms / 128
    auto = {"auto": True, "exposure_ms": None, "gain": None}
    seq = ([(ae_img, auto)] * 5
           + [(ae_img, _ex(7.8, 128.0) | {"previous": auto})] * 3      # written, image still AE
           + [(man_img, _ex(7.8, 128.0) | {"previous": auto})] * 2
           + [(man_img, _ex(7.8, 128.0))] * 20
           + [(man_img, auto | {"previous": _ex(7.8, 128.0)})] * 3      # restored, image still manual
           + [(ae_img, auto | {"previous": _ex(7.8, 128.0)})] * 2
           + [(ae_img, auto)] * 5)
    shown = [_shown(look, img, ex) for img, ex in seq]
    ref = shown[0]
    worst = max(abs(np.log(s_ / ref)) for s_ in shown)
    check(worst < np.log(1.10), f"start / stop: shown {min(shown):.1f}-{max(shown):.1f} vs AE {ref:.1f}")
    print(f"  (g) control start / stop at the table: shown field {min(shown):.1f}-{max(shown):.1f} DN "
          f"(AE image {ref:.1f}, raw manual {_field(man_img):.1f})")


def test_start_switch_frame():
    look = AutoExposureLook()
    lvl = _level_for(look.TARGET_DN, 10.0 * B0)
    lin = _scene(lvl)
    ae_img = _frame(lin, look.CEILING)
    man_img = _frame(lin, B0)
    mixed = _frame(lin, setting_brightness(7.8, 64.0))    # new exposure, the AE's gain
    auto = {"auto": True, "exposure_ms": None, "gain": None}
    trans = _ex(7.8, 128.0) | {"previous": auto}
    seq = ([(ae_img, auto)] * 5 + [(mixed, trans)] + [(man_img, trans)] * 3
           + [(man_img, _ex(7.8, 128.0))] * 10)
    shown = [_shown(look, img, ex) for img, ex in seq]
    ref = shown[0]
    after = shown[6:]
    worst = max(abs(np.log(s_ / ref)) for s_ in after)
    check(worst < np.log(1.10), f"after the switch frame: shown {min(after):.1f}-{max(after):.1f} vs AE {ref:.1f}")
    print(f"  (h) control start at the table, first frame with the AE's gain: that frame {shown[5]:.1f} DN, "
          f"the manual frames after it {min(after):.1f}-{max(after):.1f} DN (AE image {ref:.1f}, raw manual "
          f"{_field(man_img):.1f})")


def main() -> int:
    test_auto_passthrough()
    test_target_and_ceiling()
    test_steps()
    test_pace()
    test_k_max()
    test_cost()
    test_start_stop()
    test_start_switch_frame()
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
