# overlay/tools/display_exposure.py
"""The displayed camera image as the colour auto exposure would show it.

While the marker exposure control runs (tracking.hydramarker.
color_exposure), the camera is exposed for the marker: at the table
7.8 ms / 128 is ~5x darker than the AE, under the OR lamp 1.95 ms / 16
~13-22x. The tracker needs exactly that frame; the person looking at the
screen wants the familiar AE picture. AutoExposureLook brightens (or
darkens) only the display's copy of the camera image, before the X-ray
blend and the drill drawing, with one 256-entry LUT:

  1. invert the camera's tone curve (color_exposure's CRF: grey DN ->
     linear), scale by k, re-apply the curve, clip at 255;
  2. k = (the setting the AE would run) / (the setting this frame was
     taken with). The frame's own setting comes from the camera source
     (frame metadata "color_exposure", CommandedColorExposure: the old
     setting until the camera switched, both in the transition), so a
     controller step changes the image and k in the same frame and the
     display does not flicker;
  3. the AE's setting is emulated: the one that brings the mean grey level
     of the AE field (AE_ROI 480,270,1440,810 = the centre half) to the
     AE's target, followed at the AE's pace and limited by the AE's
     ceiling.

Measured basis (recorder takes 05./06./08.10. and the evaluation runs of
08.10.; scratch analysis phase6/V/run5):
  * Tone curve + GAIN_RESPONSE: the mean frame of a static scene at one
    setting, through the LUT of the step, lands on the mean frame at the
    other to 0.6-1.9 DN median / 1.5-3.0 DN p90 per pixel (E02: 3.9 ->
    7.8 ms and gain 64 -> 128; OR lamp L01-L03: 1.95 -> 3.9 ms and gain
    1 -> 16; 49-240 frames per setting); per-pixel linear ratio 2.08 /
    2.05 / 2.03 / 1.245 for model 2.00 / 2.08 / 2.00 / 1.248.
  * AE target: mean grey of the AE field 108.7-114.1 DN (median per take)
    in the six AE takes / runs whose AE was not at its limit (E01, K01-K03,
    M01, OR lamp 08.10.; 360-1356 frames each): TARGET_DN 110.
  * AE ceiling: at the table the AE sits at its longest exposure (register
    16.6 ms) and stays darker (field median 87 DN, p10-p90 79-103); the
    same table at 7.8 ms / 128 is darker by 5.9x per pixel (morning run,
    median of 348 static grid pixels) and 4.6x (afternoon, other daylight,
    3528 px): CEILING = 5.5 x the brightness of 7.8 ms / 128.
  * AE pace: after the OR lamp went off (x0.21) the AE recovered half the
    step in 6 frames, 90 % in 12 (the one such step in the AE runs): RATE
    0.15 per frame (log brightness; 4.3 / 14.2 frames).
  * K_MAX 12: the darkest settings hold the background in 0-10 DN, and k
    turns each input DN into k output DN of banding and noise (13 DN per
    step at k 12, 17 at k 16, 26 at k 24; the dark gradients of the OR
    scene already show the bands at k 12). Under the OR lamp at 1.95 ms /
    16 the AE field reads 87 / 105 / 119 / 131 DN (one run) and 76 / 89 /
    99 / 107 DN (another) at k 8 / 12 / 16 / 20 against the real AE's 114.
    The marker control's search probes at the table (3.9 ms / 64 down to
    1.95 ms / 16, k 23-75 needed) are therefore shown darker for as long
    as they last.

Qt-free and engine-free: the Live page and the evaluation script use the
same class.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from tracking.hydramarker.color_exposure import (
    _LIN,
    _grey_samples,
    dn_of_linear,
    setting_brightness,
)


class AutoExposureLook:
    """A LUT that makes the display's camera copy look like the AE image.

    update(image, exposure) once per displayed camera image (before
    anything is drawn into it; exposure = the frame's metadata
    "color_exposure", None if unknown), then apply(image, buffer) as the
    copy into the display buffer. Frames taken under the AE (or without a
    known setting) are shown as they are. reset() when the control starts.
    """

    TARGET_DN = 110.0                   # AE field mean the AE regulates to
    AE_FIELD_K = 3.0                    # a frame this close to the AE look can be an AE frame
    CEILING = 5.5 * setting_brightness(7.8, 128.0)  # AE's brightest setting
    RATE = 0.15                         # AE pace, per frame (log brightness)
    K_MAX = 12.0                        # 8-bit banding above
    K_MIN = 0.25
    GRID_STEP = 8                       # AE field samples: 120 x 68 at 1080p
    SWITCH_FRAMES = 6                   # frames after a change that may carry the old gain

    # mean grey of LUT_k(field) is looked up on a log grid of k (one small
    # matrix product per frame instead of a search)
    _K_GRID = np.exp(np.linspace(math.log(1.0 / 64.0), math.log(64.0), 97))

    def __init__(self) -> None:
        luts = np.clip(np.round(dn_of_linear(np.outer(self._K_GRID, _LIN))), 0, 255)
        self._grid_means = luts.astype(np.float64)        # (97, 256): DN out per DN in
        self.reset()

    def reset(self) -> None:
        self.k = 1.0                     # display gain of the last update (linear units)
        self.ae_brightness: Optional[float] = None   # emulated AE setting
        # the AE setting the last frame asked for, before ceiling and pace
        # (a transition frame's setting is the one that asks for the same)
        self._demand: Optional[float] = None
        self._setting = None             # the frame's manual setting (ms, gain), last frame
        self._before = None              # the one before its last change
        self._since = 0                  # frames since that change
        self._k_lut = 1.0
        self._lut: Optional[np.ndarray] = None
        self._lut4: Optional[np.ndarray] = None

    @staticmethod
    def _brightness(exposure_ms, gain) -> Optional[float]:
        if exposure_ms is None or gain is None:
            return None
        return setting_brightness(float(exposure_ms), float(gain))

    def _candidates(self, exposure: Optional[dict]) -> tuple[list[float], bool]:
        """Brightness of the manual settings this frame can carry (the
        commanded one, and in a transition the previous one and the two
        mixes: exposure and gain are written one after the other), and
        whether it can be an AE frame (in the transition at the control's
        start or stop)."""
        if not exposure:
            return [], True
        settings, maybe_ae = [], False
        for st in (exposure, exposure.get("previous")):
            if not st:
                continue
            if st.get("auto", True):
                maybe_ae = True
            else:
                settings.append((st.get("exposure_ms"), st.get("gain")))
        if settings and not exposure.get("auto", True):
            # the frame in which the exposure switches can still carry the
            # old gain although its register names the new exposure (no
            # transition then; 08.10.: 7.8 / 128 -> 3.9 / 16 read as
            # 3.9 / 128): the setting before the change, for a few frames
            cur = settings[0]
            if cur != self._setting:
                self._before, self._setting, self._since = self._setting, cur, 0
            else:
                self._since += 1
            if (self._before is not None and self._since <= self.SWITCH_FRAMES
                    and self._before not in settings):
                settings.append(self._before)
        if len(settings) >= 2:
            (e1, g1) = settings[0]
            for e0, g0 in settings[1:]:
                settings += [(e0, g1), (e1, g0)]
        out = [self._brightness(*st) for st in settings]
        return [b for b in out if b is not None and b > 0.0], maybe_ae

    def _field_gain(self, image: np.ndarray) -> float:
        """k that brings the AE field's mean grey of this frame to
        TARGET_DN (log-interpolated on the grid)."""
        h, w = image.shape[:2]
        s = self.GRID_STEP
        grey = _grey_samples(image, h // 4, 3 * h // 4, w // 4, 3 * w // 4, s)
        hist = np.bincount(grey.ravel(), minlength=256)[:256].astype(np.float64)
        means = self._grid_means @ hist / max(1.0, hist.sum())
        j = int(np.searchsorted(means, self.TARGET_DN))
        if j <= 0:
            return float(self._K_GRID[0])
        if j >= len(means):
            return float(self._K_GRID[-1])
        m0, m1 = means[j - 1], means[j]
        f = 0.0 if m1 <= m0 else (self.TARGET_DN - m0) / (m1 - m0)
        lk0, lk1 = math.log(self._K_GRID[j - 1]), math.log(self._K_GRID[j])
        return math.exp(lk0 + f * (lk1 - lk0))

    def update(self, image: np.ndarray, exposure: Optional[dict]) -> float:
        cands, maybe_ae = self._candidates(exposure)
        if not cands:
            self.k = 1.0
            return self.k
        k_field = self._field_gain(image)
        ae, demand = self.ae_brightness, self._demand
        if maybe_ae:
            # AE <-> manual (the 4-5 frames after the control's first write
            # or its restore): an AE frame is shown as it is. It is one if it
            # looks like the AE (k <= AE_FIELD_K: the AE field reads 79-118
            # DN under the AE, the first manual frames at the table 10-20)
            # and no manual setting asks for the AE setting the last frame
            # asked for (before there is one, the look alone). Both must
            # agree: the first written frame can carry the new exposure with
            # the AE's gain (08.10. table: 10 DN, then 20 at 7.8 / 128); its
            # demand is then off by the gain, and by that alone the next
            # manual frame would pass as an AE frame (shown at 20 DN)
            is_ae = k_field <= self.AE_FIELD_K
            if is_ae and demand is not None:
                is_ae = min(abs(math.log(k_field * b / demand)) for b in cands) > math.log(1.5)
            if is_ae:
                self.k = 1.0
                return self.k
        if demand is None or len(cands) == 1:
            frame_b = cands[0]
        else:
            # transition: the setting under which this image asks for the
            # AE setting the last frame asked for (the scene does not
            # change within the 3-4 frames)
            frame_b = min(cands, key=lambda b: abs(math.log(k_field * b / demand)))
        self._demand = k_field * frame_b
        want = min(self._demand, self.CEILING)
        if ae is None:
            ae = want
        else:
            ae = math.exp(math.log(ae) + self.RATE * (math.log(want) - math.log(ae)))
        self.ae_brightness = ae
        self.k = float(np.clip(ae / frame_b, self.K_MIN, self.K_MAX))
        return self.k

    def lut(self) -> Optional[np.ndarray]:
        if abs(self.k - 1.0) < 0.02:
            return None
        if self._lut is None or abs(self.k / self._k_lut - 1.0) > 0.005:
            self._k_lut = self.k
            self._lut = np.clip(np.round(dn_of_linear(_LIN * self.k)), 0, 255).astype(np.uint8)
            self._lut4 = None
        return self._lut

    def apply(self, src: np.ndarray, dst: Optional[np.ndarray] = None) -> bool:
        """The LUT from src into dst (in place without dst); with dst the
        pass is the copy - a plain copy when there is no LUT. False if the
        image is shown as it is. A BGRA image keeps its alpha byte (Qt's
        RGB32 wants 0xff there): from k 1 up the one-channel LUT keeps 255
        at 255 anyway, below it a four-channel LUT passes the alpha through
        (only then: ~1.6x the time of the one-channel pass)."""
        lut = self.lut()
        if dst is None:
            dst = src
        if lut is None:
            if dst is not src:
                np.copyto(dst, src)
            return False
        if lut[255] != 255 and src.ndim == 3 and src.shape[2] == 4:
            if self._lut4 is None:
                self._lut4 = np.ascontiguousarray(np.stack(
                    [lut, lut, lut, np.arange(256, dtype=np.uint8)], axis=-1).reshape(256, 1, 4))
            lut = self._lut4
        cv2.LUT(src, lut, dst=dst)
        return True
