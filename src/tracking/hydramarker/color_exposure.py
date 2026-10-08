"""Marker-metered colour exposure for the RealSense colour sensor.

While the drill is tracked, the colour auto exposure (AE) is the wrong
meter: it regulates the image centre (the AE region), not the marker, it
keeps a long exposure (~18-22 ms measured, blur in fast motion) and under a
bright lamp it overexposes the marker until nothing is detected (06.10.:
0 of 359 poses under the OR lamp at AE-equivalent brightness). This module
sets exposure and gain by hand from the white level of the marker itself:

  MarkerMeter              tracking thread, per processed frame: white
                           level and clipped share of the marker (hull of
                           the pose's image corners) and of the frame
  MarkerExposureController GUI thread, per drained result: the policy -
                           a target band, steps of at most x2, a hold
                           without a pose, then a bounded search ladder
  ColorExposureActuator    its own thread: every set_option call, never in
                           the GUI thread or on the tracker's critical path;
                           restores the start settings (AE on) on stop
  CommandedColorExposure   camera source, per polled frame: which colour
                           exposure the frame was taken with, for the IR
                           time-offset correction (HydraTracker.
                           ir_frame_timing)
  DisplayBrightener        GUI thread: a LUT for the displayed image only,
                           so the darker frames look like the AE image

Measured basis (06.10. camera session E01-E04, Phase-2 t24):
  * set_option takes 3.5-4.0 ms (exposure) / 0.6-1.3 ms (gain) in the
    recorder (whether it releases the GIL is not measured: hence its own
    thread here); 3 frames arriving after the call still carry the old
    setting (once 4): LATENCY_FRAMES = 4.
  * Windows/MF sets the exposure on the x2 ladder 0.98 / 1.95 / 3.9 / 7.8 /
    15.6 ms (registers 10 / 20 / 39 / 78 / 156, 100 us units); the frame
    timestamps show the register per frame (2 x (frame - sensor timestamp)
    in us), switching in the same frame as the image.
  * Gain is linear in UVC units (64 -> 128 = x1.93-2.06); after a gain step
    the brightness drifts 1-3 more frames (auto white balance).
  * Fast motion: 15.6 ms breaks above ~15 px/frame, 3.9 ms holds; 7.8 and
    3.9 ms are within take scatter at the W-take speeds. The OR lamp
    flickers at 300.3 Hz: 3.9 ms damps the banding ~4x more than 1.95 ms.
  * Brightness steps of x2 in fast motion cost tracked frames for ~4 frames,
    x1.41 did not; in slow motion x0.25..x4 cost nothing.

Qt-free and engine-free: the Live page and the evaluation script use the
same classes; the unit test drives them with a fake sensor.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


# ============================================================
# Device facts
# ============================================================

# Exposures Windows/MF can set (ms) and the option value written for each
# (100 us units; the device reads 19 back for 20).
EXPOSURE_LADDER_MS = (0.98, 1.95, 3.9, 7.8, 15.6)
_EXPOSURE_OPTION = {0.98: 10.0, 1.95: 20.0, 3.9: 39.0, 7.8: 78.0, 15.6: 156.0}

# Frames that arrive after set_option returned and still show the old
# setting: 3 measured (19/20 steps), 4 once (call returned 1.8 ms before an
# arrival).
LATENCY_FRAMES = 4

MODE_AUTO = "auto"        # the colour AE (today's behaviour)
MODE_MARKER = "marker"    # this controller while the drill is tracked
EXPOSURE_MODES = (MODE_AUTO, MODE_MARKER)


def exposure_option_value(exposure_ms: float) -> float:
    """Option value (100 us units) of a ladder exposure."""
    for e, v in _EXPOSURE_OPTION.items():
        if abs(e - float(exposure_ms)) < 1e-6:
            return v
    return float(round(float(exposure_ms) * 10.0))


def register_exposure_ms(register: Optional[float]) -> Optional[float]:
    """The ladder exposure a per-frame register reading belongs to (the
    timestamp difference reads 38 for 39, 20 for 19), None if none is
    within 15 %."""
    if register is None or not np.isfinite(register) or register <= 0:
        return None
    ms = float(register) / 10.0
    best = min(EXPOSURE_LADDER_MS, key=lambda e: abs(math.log(ms / e)))
    return best if abs(math.log(ms / best)) < math.log(1.15) else None


# ============================================================
# Settings
# ============================================================

@dataclass
class MarkerExposureConfig:
    """Policy of MarkerExposureController (see the module docstring).

    The band is the white level of the marker's white cells (median of the
    hull samples above the black/white midpoint, grey DN as the tracker
    sees it). Inside the band nothing changes - a static scene produces no
    step. The bounds (07.10., offline, phase5 controller package):
      * low 60 DN: unclipped takes darkened in linear units with the
        measured noise of gain 128 added: D05 (x2), W02, M02 and M03 keep
        their tracked frames down to a white of ~40 DN, M02/M03 lose
        18-22 % at 32-33 DN, W03 (3.9 ms, waves) loses 22 % already at
        42 DN, at 20-27 DN tracking collapses; the static tip (D05) moves
        <= 0.24 mm at >= 29 DN -> 60 DN keeps x2 of light above the cliff
        for the static takes, only ~x1.3 for W03 in motion;
      * high 170 DN: Phase 2 brightened M02 to ~184 DN without clipping and
        lost 7 % of its tracked frames to the tone curve (black cells
        lifted), C02 at ~200 DN 2 %;
      * clip 30 %: up to ~50 % clipped white samples cost nothing when
        nothing else changed (Phase-2 clip-only test), 55-65 % did.
    The table setting 7.8 ms / 128 gives ~90 DN (D05, W02, M02 metered with
    this meter): inside the band, so the table runs without a step.
    """

    white_low_dn: float = 60.0
    white_high_dn: float = 170.0
    target_dn: float = 110.0          # where a step aims (log-middle of the band in linear units)
    clip_max_frac: float = 0.30       # white samples >= 250 DN
    min_samples: int = 400            # hull samples for a valid metering
    confirm_frames: int = 3           # consecutive out-of-band meterings before a step
    max_step: float = 2.0             # brightness factor per step
    max_step_moving: float = 1.41     # ... while the marker moves faster than moving_px
    moving_px: float = 5.0            # image speed (px/frame) of "moving"
    latency_frames: int = LATENCY_FRAMES
    gain_settle_frames: int = 3       # extra frames after a gain step (white balance drift)
    exposure_max_ms: float = 7.8      # never longer by default (motion blur)
    exposure_min_ms: float = 1.95     # ... only against clipping:
    exposure_min_unclipped_ms: float = 3.9   # shortest for a white merely above the band (300 Hz banding)
    gain_floor: float = 16.0          # lowest gain used (device minimum if higher)
    gain_ceiling: float = 128.0       # highest gain used (device maximum if lower)
    gain_longer_min: float = 40.0     # go to the next longer exposure only if its gain stays >= this
    min_gain_change: float = 0.03     # relative gain change below which nothing is written
    hold_s: float = 1.0               # no pose: keep the setting this long
    # search without a pose: presets (exposure ms, gain), each held for
    # LATENCY + settle frames and then search_attempts processed frames;
    # after a cycle without a lock every preset is held search_slow_s
    search_presets: tuple = ((7.8, 128.0), (3.9, 64.0), (3.9, 16.0), (1.95, 16.0))
    # Metered frames per preset. 07.10. live, a re-entry locked 0-0.4 s
    # after the corners were visible again, but 1.1-2.6 s after the engine
    # reset; from cold the engine needs 3-37 frames at the right setting
    # (closed loop, D05 at 7.8 ms / 128, 4 start points) - longer than 10.
    # Still 10: in the closed loop (OR light x30 / x55, 5 start points
    # each) 20-40 frames at the first unclipped preset locked no sooner,
    # 20-30 at every preset made the slowest start slower (up to 357
    # frames); the slow cycle comes back to every preset.
    search_attempts: int = 10
    search_slow_s: float = 1.0
    # lost under unchanged frame light: keep the setting ("search-home")
    # this long, then search anyway - a marker that comes back into a lamp
    # spot clips at the old setting while the frame light stays the same
    search_home_s: float = 3.0
    # ... and its slow cycle holds home this long between two probes (a
    # marker back under the old light that meets a dark probe waits for
    # it to end)
    search_stale_slow_s: float = 2.0
    # the clip ceiling (lowest exposure x gain at which the marker clipped)
    # expires after clip_forget_s without clipping (a short glare must not
    # cap the brightness for good); if the brighter step after an expiry
    # clips again, the next expiry waits twice as long, up to
    # clip_forget_max_s (a lamp spot: one try per period, not a cycle)
    clip_forget_s: float = 2.0
    clip_forget_max_s: float = 32.0
    # lost while the marker region clipped (lamp switched on, glare): darken
    # at once instead of holding
    clip_loss_frac: float = 0.30
    # frame clipped share above which a search preset counts as failed
    # after one attempt (bright light: no need to wait at a bright preset)
    search_frame_clip: float = 0.10
    # without a pose the presets are probed only if the frame says the
    # light changed since the last pose (frame median, per exposure x gain,
    # off by more than this factor, or more of the frame clips): a marker
    # that just left the view or turned away keeps its setting
    search_light_change: float = 1.5

    def snapshot(self) -> dict:
        out = {}
        for name in self.__dataclass_fields__:
            v = getattr(self, name)
            out[name] = [list(p) for p in v] if name == "search_presets" else v
        return out


# ============================================================
# Metering (tracking thread)
# ============================================================

@dataclass
class Metering:
    """One processed frame's numbers for the controller."""

    frame_index: int
    t_frame: float
    fresh: bool                      # a fresh pose in this frame
    white_dn: float = float("nan")   # marker white level (fresh pose, else the last hull)
    black_dn: float = float("nan")
    clip_frac: float = float("nan")  # share of the white samples >= 250
    n_samples: int = 0
    hull_from_pose: bool = False     # the hull is this frame's (False: the last pose's)
    frame_clip_frac: float = float("nan")   # share of a coarse frame grid >= 250
    frame_median_dn: float = float("nan")
    speed_px: float = float("nan")   # hull centre motion since the last metered pose
    meter_ms: float = 0.0


def _grey_samples(image: np.ndarray, y0: int, y1: int, x0: int, x1: int, step: int) -> np.ndarray:
    """Grey values (the tracker's BGR(A)->GRAY weights, integer) of a strided
    window - a view plus one small array, no frame copy."""
    win = image[y0:y1:step, x0:x1:step]
    if win.ndim == 2:
        return win.astype(np.uint16)
    b = win[..., 0].astype(np.uint16)
    g = win[..., 1].astype(np.uint16)
    r = win[..., 2].astype(np.uint16)
    return (b * 29 + g * 150 + r * 77 + 128) >> 8


class MarkerMeter:
    """White level of the marker from the convex hull of the pose's image
    corners, in the tracking thread right after the tracker.

    Without a fresh pose the hull of the last fresh pose is metered (for at
    most keep_s of frame time): that is what tells a marker lost to
    clipping (lamp switched on) from a marker that left the view; and a
    coarse grid of the whole frame gives its clipped share for the search.
    Cost: a strided view of the hull's bounding box (<= max_samples
    samples) and a frame grid (~8k samples), each into a 256-bin histogram
    (levels and shares from it, no sort); no frame copy.
    """

    def __init__(self, fresh: Callable[[Any], bool], *, max_samples: int = 4000,
                 frame_step: int = 16, keep_s: float = 3.0, clip_dn: int = 250) -> None:
        self._fresh = fresh               # tracker_log.has_fresh_pose
        self.max_samples = int(max_samples)
        self.frame_step = int(frame_step)
        self.keep_s = float(keep_s)
        self.clip_dn = int(clip_dn)
        self.reset()

    def reset(self) -> None:
        self._hull: Optional[np.ndarray] = None   # int32 (N, 2) image points
        self._t_hull = 0.0
        self._centre: Optional[np.ndarray] = None

    @staticmethod
    def corner_points(result) -> Optional[np.ndarray]:
        """Image points of the result's pose (corners, else the
        correspondence corners), None with fewer than 4."""
        for name in ("corners", "correspondence_corners"):
            pts = getattr(result, name, None) or ()
            if len(pts) >= 4:
                return np.asarray([c.uv for c in pts], dtype=np.float64).reshape(-1, 2)
        return None

    def measure(self, image: np.ndarray, result, frame_index: int, t_frame: float) -> Metering:
        t0 = time.perf_counter()
        fresh = bool(self._fresh(result))
        m = Metering(frame_index=int(frame_index), t_frame=float(t_frame), fresh=fresh)
        h, w = image.shape[:2]
        if fresh:
            pts = self.corner_points(result)
            if pts is not None:
                import cv2

                hull = cv2.convexHull(np.round(pts).astype(np.int32)).reshape(-1, 2)
                centre = pts.mean(axis=0)
                if self._centre is not None:
                    m.speed_px = float(np.linalg.norm(centre - self._centre))
                self._centre = centre
                self._hull = hull
                self._t_hull = float(t_frame)
                m.hull_from_pose = True
        elif self._hull is not None and t_frame - self._t_hull > self.keep_s:
            self._hull = None
            self._centre = None
        if self._hull is not None:
            self._meter_hull(image, m, w, h)
        fs = max(1, self.frame_step)
        hist = np.bincount(_grey_samples(image, fs // 2, h, fs // 2, w, fs).ravel(), minlength=256)
        n = int(hist.sum())
        if n:
            m.frame_clip_frac = float(hist[self.clip_dn:].sum()) / n
            m.frame_median_dn = float(np.searchsorted(np.cumsum(hist), 0.5 * n))
        m.meter_ms = (time.perf_counter() - t0) * 1000.0
        return m

    def _meter_hull(self, image: np.ndarray, m: Metering, w: int, h: int) -> None:
        import cv2

        hull = self._hull
        x0 = max(0, int(hull[:, 0].min()))
        x1 = min(w, int(hull[:, 0].max()) + 1)
        y0 = max(0, int(hull[:, 1].min()))
        y1 = min(h, int(hull[:, 1].max()) + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        area = (x1 - x0) * (y1 - y0)
        step = max(1, int(math.ceil(math.sqrt(area / float(self.max_samples)))))
        vals = _grey_samples(image, y0, y1, x0, x1, step)
        mh, mw = vals.shape
        mask = np.zeros((mh, mw), np.uint8)
        poly = np.round((hull - (x0, y0)) / float(step)).astype(np.int32)
        cv2.fillConvexPoly(mask, poly, 1)
        hist = np.bincount(vals[mask.view(bool)], minlength=256)
        cum = np.cumsum(hist)
        n = int(cum[-1])
        m.n_samples = n
        if n < 16:
            return
        p5 = int(np.searchsorted(cum, 0.05 * n))
        p95 = int(np.searchsorted(cum, 0.95 * n))
        # white = the samples at or above the black/white midpoint (all of
        # them when even the 5th percentile clips)
        mid = 0 if p5 >= self.clip_dn else int(math.ceil(0.5 * (p5 + p95)))
        n_black = int(cum[mid - 1]) if mid > 0 else 0
        n_white = n - n_black
        if n_white:
            m.white_dn = float(np.searchsorted(cum, n_black + 0.5 * n_white))
            m.clip_frac = float(hist[self.clip_dn:].sum()) / n_white
        if n_black:
            m.black_dn = float(np.searchsorted(cum, 0.5 * n_black))


# ============================================================
# Policy (GUI thread)
# ============================================================

@dataclass
class ExposureCommand:
    """A manual colour setting to write (exposure in ms on the ladder,
    gain in UVC units); reason for the log."""

    exposure_ms: float
    gain: float
    reason: str
    seq: int = 0
    write_ms: float = float("nan")    # set_option time of its writes (actuator)

    @property
    def product(self) -> float:
        return float(self.exposure_ms) * float(self.gain)


class MarkerExposureController:
    """Decides the colour setting from the marker's white level.

    update() takes the newest Metering (in processing order) and returns an
    ExposureCommand or None; the caller hands the command to the actuator.
    applied() is the actuator's report that a command was written: from
    the frame after frame_index + latency_frames (+ gain_settle_frames
    after a gain change) on, frames show the new setting and are metered
    again; until then nothing new is decided. failed() is its report that
    a write raised: the command is dropped, the setting is what the device
    reads back.

    States: "start" (first command not applied yet), "track" (fresh pose,
    in band or stepping), "limit" (out of band but at the end of the
    range: nothing to do), "split" (part of the marker clips while its
    median is dark, or brighter would go back to a setting that clipped:
    the light spans more than the camera takes, so nothing is changed),
    "hold" (no pose for less than hold_s),
    "clip-loss" (lost while the last hull clipped: darken at once),
    "dark-loss" (lost while the last hull is far below the band: brighten
    at once),
    "search", "search-slow", "search-home" (lost under the light of the
    last pose: the setting is kept, nothing is probed - for search_home_s,
    then the search runs anyway).

    The search starts at "home", the setting the pose was lost with (at
    the start: the first preset; after a light change: that setting
    scaled by the frame's change): home, then every other preset, each
    for search_attempts metered frames. A preset whose frame grid clips is
    left after one frame when a darker one follows, and the rest of the
    cycle skips the presets at least as bright. After that cycle without a
    lock it goes slow: home (if its frame clipped: the brightest preset
    whose frame did not) for search_slow_s, one probe, home again, the
    next probe - a marker that comes back under the old light finds the
    old setting most of the time, a changed light is found within a cycle,
    and the camera is written twice per search_slow_s + one probe. A
    search begun by search_home_s (frame light unchanged) probes only the
    presets darker than home, leaves home at once for them and holds home
    search_stale_slow_s in its slow cycle.
    """

    def __init__(self, config: Optional[MarkerExposureConfig] = None,
                 gain_range: Optional[tuple] = None) -> None:
        self.config = config or MarkerExposureConfig()
        c = self.config
        lo, hi = c.gain_floor, c.gain_ceiling
        if gain_range is not None:
            lo = max(lo, float(gain_range[0]))
            hi = min(hi, float(gain_range[1]))
        self.gain_min = lo
        self.gain_max = max(lo, hi)
        self.exposures = tuple(e for e in EXPOSURE_LADDER_MS
                               if c.exposure_min_ms - 1e-6 <= e <= c.exposure_max_ms + 1e-6)
        self.state = "idle"
        self.exposure_ms = float("nan")   # setting written last (applied)
        self.gain = float("nan")
        self._pending: Optional[ExposureCommand] = None
        self._effective_from = 0          # first frame index that shows the applied setting
        self._seq = 0
        self._out_count = 0
        self._out_dir = 0
        # lowest exposure x gain at which the marker clipped (cleared when
        # a search starts or the frame light changes: the light may have
        # changed; expires after _clip_forget_s without clipping)
        self._clip_product = float("inf")
        self._clip_ref = float("nan")             # frame light (per exposure x gain) when it was set
        self._t_clip = float("-inf")              # last metering with the marker clipped
        self._clip_forget_s = self.config.clip_forget_s
        self._t_ceiling_expired: Optional[float] = None
        self._search_clip_product = float("inf")  # lowest preset whose frame clipped (this cycle)
        self._stale = False                       # the search was begun by search_home_s
        self.n_failed = 0                         # failed writes in all
        self.n_failed_row = 0                     # ... since the last applied one
        self._t_last_pose: Optional[float] = None
        self._t_start = 0.0
        # frame median (linear, per exposure x gain) and clipped share at the
        # last pose
        self._ref_frame: Optional[tuple] = None
        self._home: tuple = (7.8, 128.0)
        self._plan: list = []             # [(exposure, gain, "attempts" | "slow")]
        self._plan_i = 0
        self._probe_i = 0
        self._search_frames = 0
        self._search_t0 = 0.0
        self.n_steps = 0
        self.n_search_steps = 0
        self.last_reason = ""
        self.last_metering: Optional[Metering] = None

    # ---- bookkeeping ----

    def _command(self, exposure_ms: float, gain: float, reason: str) -> Optional[ExposureCommand]:
        gain = float(np.clip(round(gain), self.gain_min, self.gain_max))
        if (np.isfinite(self.exposure_ms) and abs(exposure_ms - self.exposure_ms) < 1e-6
                and abs(gain - self.gain) <= self.config.min_gain_change * self.gain):
            return None
        self._seq += 1
        cmd = ExposureCommand(float(exposure_ms), gain, reason, self._seq)
        self._pending = cmd
        self.last_reason = reason
        return cmd

    def start(self, t_now: float) -> ExposureCommand:
        """The first setting (the first search preset), at tracking start."""
        self.state = "start"
        self._t_last_pose = None
        self._t_start = float(t_now)
        e, g = self.config.search_presets[0]
        return self._command(float(e), float(g), "start")

    def applied(self, cmd: ExposureCommand, frame_index: int) -> None:
        """cmd was written; frame_index = the newest frame polled when the
        call returned."""
        gain_changed = not np.isfinite(self.gain) or abs(cmd.gain - self.gain) > 1e-6
        self.exposure_ms = float(cmd.exposure_ms)
        self.gain = float(cmd.gain)
        settle = self.config.latency_frames + (self.config.gain_settle_frames if gain_changed else 0)
        self._effective_from = int(frame_index) + settle + 1
        if self._pending is not None and self._pending.seq == cmd.seq:
            self._pending = None
        self._out_count = 0
        self._search_frames = 0
        self.n_failed_row = 0

    def failed(self, cmd: ExposureCommand, frame_index: int,
               setting: Optional[tuple] = None) -> None:
        """A write of cmd raised. setting = (exposure ms, gain) the device
        reads back afterwards (manual), None if unknown or under the AE: the
        command is dropped and the old setting kept (frames from
        frame_index + latency_frames on are metered again)."""
        self.n_failed += 1
        self.n_failed_row += 1
        if self._pending is not None and self._pending.seq == cmd.seq:
            self._pending = None
        if setting is not None:
            self.exposure_ms, self.gain = float(setting[0]), float(setting[1])
        self._effective_from = int(frame_index) + self.config.latency_frames + 1
        self._out_count = 0
        self._search_frames = 0
        self.last_reason = "failed"

    def _choose(self, product: float, clipped: bool = False) -> tuple:
        """(exposure, gain) for an exposure x gain product: keep the
        current exposure while its gain stays in range, the next shorter
        one when the gain would fall below the floor, the next longer one
        (up to exposure_max_ms) when its gain would stay >= gain_longer_min.
        Below exposure_min_unclipped_ms (1.95 ms: 4x the 300 Hz banding of
        3.9 ms) only when the marker clips: unclipped, a setting below it
        goes back up as soon as the gain there stays >= the floor."""
        ex = self.exposures
        e = self.exposure_ms if self.exposure_ms in ex else ex[-1]
        i = ex.index(e)
        shortest = 0 if clipped else next(
            (k for k, x in enumerate(ex) if x >= self.config.exposure_min_unclipped_ms - 1e-6), 0)
        while True:
            g = product / ex[i]
            if g < self.gain_min and i > shortest:
                i -= 1
                continue
            if i + 1 < len(ex) and product / ex[i + 1] >= max(self.config.gain_longer_min, self.gain_min):
                i += 1
                continue
            break
        if i < shortest and product / ex[shortest] >= self.gain_min:
            i = shortest
        e = ex[i]
        return e, float(np.clip(product / e, self.gain_min, self.gain_max))

    # ---- per result ----

    def update(self, m: Metering) -> Optional[ExposureCommand]:
        self.last_metering = m
        c = self.config
        if m.fresh:
            self._t_last_pose = m.t_frame
            # the frame light at the pose, per exposure x gain - only from a
            # frame that shows the setting written last (frames before
            # _effective_from were taken with the old one)
            if (np.isfinite(m.frame_median_dn) and np.isfinite(self.exposure_ms)
                    and self._pending is None and m.frame_index >= self._effective_from):
                self._ref_frame = (linear_dn(m.frame_median_dn) / (self.exposure_ms * self.gain),
                                   m.frame_clip_frac)
            if self.state in ("search", "search-slow", "search-home", "hold", "clip-loss",
                              "dark-loss", "start"):
                self.state = "track"
                self._out_count = 0
        if self._pending is not None or m.frame_index < self._effective_from:
            return None
        if m.fresh:
            return self._track(m)
        return self._lost(m)

    def _track(self, m: Metering) -> Optional[ExposureCommand]:
        c = self.config
        if (m.n_samples < c.min_samples or not np.isfinite(m.white_dn)
                or not m.hull_from_pose):
            return None
        clip = m.clip_frac if np.isfinite(m.clip_frac) else 0.0
        product = self.exposure_ms * self.gain
        frame_lin = (linear_dn(m.frame_median_dn) / product
                     if np.isfinite(m.frame_median_dn) else float("nan"))
        if (np.isfinite(self._clip_product) and np.isfinite(frame_lin)
                and np.isfinite(self._clip_ref)
                and not (1.0 / c.search_light_change <= frame_lin / self._clip_ref
                         <= c.search_light_change)):
            self._forget_ceiling(reset_backoff=True)      # the light changed
        if clip > c.clip_max_frac:
            if (self._t_ceiling_expired is not None
                    and m.t_frame - self._t_ceiling_expired < self._clip_forget_s):
                # clipped again soon after the ceiling expired (a lamp spot,
                # not a glare): the next expiry waits twice as long
                self._clip_forget_s = min(2.0 * self._clip_forget_s, c.clip_forget_max_s)
            self._t_ceiling_expired = None
            self._t_clip = m.t_frame
        elif (np.isfinite(self._clip_product)
              and m.t_frame - self._t_clip >= self._clip_forget_s):
            # no clipping for _clip_forget_s (a glare is over): the ceiling
            # expires, brighter steps are allowed again
            self._forget_ceiling(reset_backoff=False)
            self._t_ceiling_expired = m.t_frame
        unclip = False
        if clip > c.clip_max_frac:
            # part of the marker clips (lamp spot, glare); remember where, so
            # no brightening goes back there. With the median already dark
            # the light spans more than the camera can take: leave it
            # ("split")
            if product < self._clip_product:
                self._clip_ref = frame_lin
            self._clip_product = min(self._clip_product, product)
            direction = -1 if m.white_dn >= c.white_low_dn else 0
            split = direction == 0
        elif m.white_dn < c.white_low_dn:
            # brighter, but not up to a setting that clipped
            direction = +1 if product * 1.05 < self._clip_product / c.max_step_moving else 0
            split = direction == 0
        elif m.white_dn > c.white_high_dn:
            direction, split = -1, False
        else:
            direction, split = 0, False
            # in the band below exposure_min_unclipped_ms (left there by a
            # clip step or a search) with no ceiling and nothing clipped:
            # back up to it - the same brightness if its gain allows, else
            # its gain floor when the white there stays at or below the
            # middle of target and upper bound
            emin = c.exposure_min_unclipped_ms
            if (self.exposure_ms < emin - 1e-6 and not np.isfinite(self._clip_product)
                    and clip == 0.0 and emin in self.exposures):
                up = emin * max(self.gain_min, product / emin) / product
                if (up <= 1.0 + 1e-6 or dn_of_linear(linear_dn(m.white_dn) * up)
                        <= 0.5 * (c.target_dn + c.white_high_dn)):
                    direction, unclip = +1, True
        if direction == 0:
            self._out_count = 0
            self.state = "split" if split else "track"
            return None
        if direction != self._out_dir:
            self._out_dir = direction
            self._out_count = 0
        self._out_count += 1
        if self._out_count < c.confirm_frames:
            return None
        moving = np.isfinite(m.speed_px) and m.speed_px > c.moving_px
        step = c.max_step_moving if moving else c.max_step
        if unclip:
            if up > step + 1e-6:
                return None             # up to x2: not while the marker moves fast
            f = up
        elif clip > c.clip_max_frac:
            f = 1.0 / step              # level unknown above the clip: one full step down
        else:
            f = linear_dn(c.target_dn) / max(linear_dn(m.white_dn), 1e-4)
            f = float(np.clip(f, 1.0 / step, step))
            if f > 1.0:
                f = min(f, self._clip_product / c.max_step_moving / product)
        e, g = self._choose(product * f, clipped=clip > c.clip_max_frac)
        cmd = self._command(e, g, "unclip" if unclip else "darker" if f < 1 else "brighter")
        if cmd is None:
            self.state = "limit"
            self._out_count = 0
            return None
        self.state = "track"
        self.n_steps += 1
        return cmd

    def _lost(self, m: Metering) -> Optional[ExposureCommand]:
        c = self.config
        t_pose = self._t_last_pose
        # after the start the first preset is held like a lost pose's
        # setting - unless the frame clips (bright light: search at once);
        # a search begun before the first pose goes on (a darker preset
        # whose frame does not clip is no reason to start over)
        since = m.t_frame - (self._t_start if t_pose is None else t_pose)
        bright_start = (t_pose is None and np.isfinite(m.frame_clip_frac)
                        and m.frame_clip_frac > c.search_frame_clip)
        searching = t_pose is None and self.state in ("search", "search-slow")
        if since < c.hold_s and not bright_start and not searching:
            # clipped marker region of the last pose: darken now
            if (np.isfinite(m.clip_frac) and m.clip_frac > c.clip_loss_frac
                    and m.n_samples >= c.min_samples):
                self.state = "clip-loss"
                # lost anyway: no motion limit; far over (> 60 % clipped) x4
                f = 0.25 if m.clip_frac > 0.6 else 1.0 / c.max_step
                e, g = self._choose(self.exposure_ms * self.gain * f, clipped=True)
                cmd = self._command(e, g, "clip-loss")
                if cmd is not None:
                    self.n_steps += 1
                return cmd
            # far below the band at the last pose's place (lamp switched
            # off): brighten now, up to x4, not back into a clipping setting
            if (np.isfinite(m.white_dn) and m.white_dn < 0.5 * c.white_low_dn
                    and m.n_samples >= c.min_samples):
                product = self.exposure_ms * self.gain
                f = min(4.0, linear_dn(c.target_dn) / max(linear_dn(m.white_dn), 1e-4),
                        self._clip_product / c.max_step_moving / product)
                if f > 1.05:
                    self.state = "dark-loss"
                    e, g = self._choose(product * f)
                    cmd = self._command(e, g, "dark-loss")
                    if cmd is not None:
                        self.n_steps += 1
                    return cmd
            self.state = "hold"
            return None
        return self._search(m)

    def _probes(self) -> list:
        h = self._home
        out = [(float(e), float(g)) for e, g in self.config.search_presets
               if not (abs(e - h[0]) < 1e-6 and abs(g - h[1]) < 1.0)]
        if self._stale:
            # under unchanged frame light the marker can only come back
            # brighter (a lamp spot): only the darker presets
            out = [q for q in out if q[0] * q[1] < h[0] * h[1] * (1.0 - 1e-6)]
        return out

    def _light_changed(self, m: Metering) -> bool:
        ref = self._ref_frame
        if ref is None or not np.isfinite(m.frame_median_dn):
            return True
        now = linear_dn(m.frame_median_dn) / (self.exposure_ms * self.gain)
        ratio = now / max(ref[0], 1e-9)
        lim = self.config.search_light_change
        return (not (1.0 / lim <= ratio <= lim)
                or m.frame_clip_frac > ref[1] + 0.5 * self.config.search_frame_clip)

    def _forget_ceiling(self, reset_backoff: bool) -> None:
        self._clip_product = float("inf")
        if reset_backoff:
            self._clip_forget_s = self.config.clip_forget_s
            self._t_ceiling_expired = None

    def _search(self, m: Metering) -> Optional[ExposureCommand]:
        c = self.config
        if self._t_last_pose is not None and self.state not in ("search", "search-slow"):
            # lost under unchanged light: keep the setting (no write) for
            # search_home_s - a marker that comes back into a lamp spot
            # clips at it while the frame light stays the same
            if (not self._light_changed(m)
                    and m.t_frame - self._t_last_pose < c.search_home_s):
                self.state = "search-home"
                return None
        if self.state not in ("search", "search-slow"):
            # after search_home_s under unchanged frame light the setting
            # was held long enough: straight to the probes, short each (the
            # frame tells nothing about the marker)
            stale = self._t_last_pose is not None and not self._light_changed(m)
            if stale and not any(e * g < self.exposure_ms * self.gain * (1.0 - 1e-6)
                                 for e, g in c.search_presets):
                self.state = "search-home"          # nothing darker to try
                return None
            self.state = "search"
            self._forget_ceiling(reset_backoff=True)
            self._search_clip_product = float("inf")
            self._stale = stale
            self._home = (self.exposure_ms, self.gain)
            ref = self._ref_frame
            if (not stale and self._t_last_pose is not None and ref is not None
                    and np.isfinite(m.frame_median_dn)):
                # the light changed while the marker was away: home = the
                # setting scaled by the frame's change (a clipping frame
                # understates it; the probes cover the rest)
                now = linear_dn(m.frame_median_dn) / (self.exposure_ms * self.gain)
                r = float(np.clip(now / max(ref[0], 1e-9), 1.0 / 64.0, 64.0))
                self._home = self._choose(self.exposure_ms * self.gain / r, clipped=r > 1.0)
            self._plan = ([(*self._home, "once" if stale else "attempts")]
                          + [(*p, "attempts") for p in self._probes()])
            self._plan_i = 0
            self._probe_i = 0
            self._search_frames = 0
        self._search_frames += 1
        e, g, kind = self._plan[self._plan_i]
        frame_clips = np.isfinite(m.frame_clip_frac) and m.frame_clip_frac > c.search_frame_clip
        if frame_clips:
            self._search_clip_product = min(self._search_clip_product, e * g)
        if self._search_frames == 1:
            self._search_t0 = m.t_frame
        if kind == "slow":
            done = m.t_frame - self._search_t0 >= (c.search_stale_slow_s if self._stale else c.search_slow_s)
        elif kind == "once":
            done = True
        else:
            nxt = self._plan[self._plan_i + 1] if self._plan_i + 1 < len(self._plan) else None
            bright = (self.state == "search" and frame_clips
                      and nxt is not None and nxt[0] * nxt[1] < e * g)
            done = self._search_frames >= c.search_attempts or bright
        if not done:
            return None
        self._plan_i += 1
        if self.state == "search":
            # not again at a preset as bright as one whose frame clipped
            while (self._plan_i < len(self._plan) and self._plan[self._plan_i][0]
                   * self._plan[self._plan_i][1] >= self._search_clip_product * (1.0 - 1e-6)):
                self._plan_i += 1
        if self._plan_i >= len(self._plan):
            # slow: home for search_slow_s, then the next probe; a home
            # whose frame clipped becomes the brightest preset whose frame
            # did not
            if self.state == "search" and (self._home[0] * self._home[1]
                                           >= self._search_clip_product * (1.0 - 1e-6)):
                darker = [(float(pe), float(pg)) for pe, pg in c.search_presets
                          if pe * pg < self._search_clip_product * (1.0 - 1e-6)]
                self._home = (max(darker, key=lambda p: p[0] * p[1]) if darker
                              else min(((float(pe), float(pg)) for pe, pg in c.search_presets),
                                       key=lambda p: p[0] * p[1]))
            self.state = "search-slow"
            probes = self._probes()
            p = probes[self._probe_i % len(probes)] if probes else self._home
            self._probe_i += 1
            self._plan = [(*self._home, "slow"), (*p, "attempts")]
            self._plan_i = 0
        e, g, _ = self._plan[self._plan_i]
        self._search_frames = 0
        cmd = self._command(e, g, self.state)
        if cmd is not None:
            self.n_search_steps += 1
        return cmd

    def status(self) -> dict:
        m = self.last_metering
        return {
            "state": self.state,
            "exposure_ms": self.exposure_ms,
            "gain": self.gain,
            "white_dn": None if m is None else m.white_dn,
            "clip_frac": None if m is None else m.clip_frac,
            "steps": self.n_steps,
            "search_steps": self.n_search_steps,
            "reason": self.last_reason,
        }


# ============================================================
# Tone curve (display and step size)
# ============================================================

# Inverse response of the D435i colour pipeline (grey DN -> linear), fitted
# on the 30.09. step takes (Phase-2 results/camera/crf_fit.json):
# lin(u) = u (1 + sum c_i u^i) / (1 + sum c_i), u = DN / 255.
_CRF = (-1.6203490929090678, 9.161667099455086, -13.83606807435354,
        12.44095292353315, -3.520692703298443)
_DN = np.arange(256, dtype=np.float64)


def _lin_of_dn(dn):
    u = np.clip(np.asarray(dn, np.float64), 0.0, 255.0) / 255.0
    acc = np.ones_like(u)
    p = np.ones_like(u)
    for c in _CRF:
        p = p * u
        acc = acc + c * p
    return u * acc / (1.0 + sum(_CRF))


_LIN = _lin_of_dn(_DN)


def linear_dn(dn: float) -> float:
    """Linear brightness (0..1) of a grey level."""
    return float(np.interp(float(dn), _DN, _LIN))


def dn_of_linear(lin) -> np.ndarray:
    return np.interp(np.minimum(np.asarray(lin, np.float64), 1.0), _LIN, _DN)


class DisplayBrightener:
    """Brightens the displayed image (never the tracking input or a saved
    frame) while the controller keeps the camera darker than the AE would.

    The gain k (linear units) brings the median of a coarse grid of the
    centre half of the camera image to the AE's centre level (target_dn,
    the AE's measured centre mean ~111 DN), only up, at most max_gain,
    smoothed; the LUT is rebuilt only when k moved by more than 2 %.
    update() and apply() get the display's copy of the camera image
    before anything is drawn or blended into it: k follows the camera,
    and the X-ray and the drill keep their own grey levels and colours.
    """

    def __init__(self, target_dn: float = 111.0, max_gain: float = 8.0,
                 smooth: float = 0.2, grid_step: int = 8) -> None:
        self.target_dn = float(target_dn)
        self.max_gain = float(max_gain)
        self.smooth = float(smooth)
        self.grid_step = int(grid_step)
        self.reset()

    def reset(self) -> None:
        self.k = 1.0
        self._k_lut = 1.0
        self._lut: Optional[np.ndarray] = None

    def update(self, image: np.ndarray) -> float:
        h, w = image.shape[:2]
        s = self.grid_step
        grid = _grey_samples(image, h // 4, 3 * h // 4, w // 4, 3 * w // 4, s)
        if grid.size == 0:
            return self.k
        med = float(np.median(grid))
        want = linear_dn(self.target_dn) / max(linear_dn(max(med, 1.0)), 1e-4)
        want = float(np.clip(want, 1.0, self.max_gain))
        self.k = (1.0 - self.smooth) * self.k + self.smooth * want
        return self.k

    def lut(self) -> Optional[np.ndarray]:
        if self.k < 1.02:
            return None
        if self._lut is None or abs(self.k / self._k_lut - 1.0) > 0.02:
            self._k_lut = self.k
            self._lut = np.clip(np.round(dn_of_linear(_LIN * self.k)), 0, 255).astype(np.uint8)
        return self._lut

    def apply(self, image: np.ndarray) -> bool:
        """Brighten image in place; False if nothing was done."""
        lut = self.lut()
        if lut is None:
            return False
        import cv2

        cv2.LUT(image, lut, dst=image)
        return True


# ============================================================
# Per-frame exposure state (camera source, poll thread)
# ============================================================

class CommandedColorExposure:
    """Which colour exposure a polled frame was taken with, while the
    controller writes the settings.

    The actuator reports every write (applied) with the newest frame index
    polled when the call returned. Frames up to that index carry the old
    setting; the next latency_frames + 1 are in transition (the measured 3,
    once 4, old frames after a call, and a frame polled late counts as
    after the call); later frames carry the new one. In transition a frame
    whose register (from its timestamps) names the old or the new exposure
    unambiguously takes that one; otherwise it gets both and the time
    correction uses the mean (error <= |delta tau| / 2). Under the AE there
    is no usable number (the register keeps the last manual value): auto
    True, exposure_ms None.

    for_frame() runs in the poll thread, applied() in the actuator thread:
    the state is one tuple, replaced whole.
    """

    def __init__(self, auto: bool, exposure_ms: Optional[float],
                 latency_frames: int = LATENCY_FRAMES) -> None:
        self.latency_frames = int(latency_frames)
        cur = (bool(auto), None if auto else exposure_ms, None)
        # (previous (auto, ms, gain), current (auto, ms, gain), frame index
        # at the call return); the gain only goes into the logs
        self._state = (cur, cur, -10 ** 9)

    @staticmethod
    def _d(st, **extra) -> dict:
        return {"auto": st[0], "exposure_ms": st[1], "gain": st[2], **extra}

    @property
    def current(self) -> dict:
        return self._d(self._state[1])

    def applied(self, auto: bool, exposure_ms: Optional[float], frame_index: int,
                gain: Optional[float] = None) -> None:
        prev = self._state[1]
        cur = (bool(auto), None if auto else float(exposure_ms), None if auto else gain)
        self._state = (prev, cur, int(frame_index))

    def settled(self, frame_index: int) -> bool:
        """frame_index is past the transition of the last write."""
        return frame_index > self._state[2] + self.latency_frames + 1

    def for_frame(self, frame_index: int, register: Optional[float] = None) -> dict:
        prev, cur, idx = self._state
        if frame_index <= idx:
            # polled before the call returned: the old setting (a frame
            # polled late belongs to the transition below)
            return self._d(prev)
        if frame_index > idx + self.latency_frames + 1 or prev == cur:
            return self._d(cur)
        reg_ms = register_exposure_ms(register)
        if reg_ms is not None and not prev[0] and not cur[0] and prev[1] != cur[1]:
            # manual -> manual exposure step: the register flips with the image
            if abs(math.log(reg_ms / cur[1])) < math.log(1.15):
                return self._d(cur)
            if abs(math.log(reg_ms / prev[1])) < math.log(1.15):
                return self._d(prev)
        return self._d(cur, previous={"auto": prev[0], "exposure_ms": prev[1], "gain": prev[2]})


# ============================================================
# Actuator thread
# ============================================================

class ColorExposureActuator:
    """Writes the colour settings in its own thread.

    set_option of the colour sensor takes 1-4 ms (whether it releases the
    GIL is not measured); here it never runs in the GUI thread or in the
    tracking worker. submit()
    queues a command (a newer one replaces one not started yet); the
    thread switches the AE off before its first write, writes exposure
    then gain, and reports each write (applied events, polled by the GUI
    thread; CommandedColorExposure for the frames). A write that raises is
    reported as a failed event with the setting the device reads back
    afterwards (the frames then carry that one). stop(restore=True) writes
    the start settings back (manual exposure and gain as read at the
    start, then the AE on, the order the camera start uses - each write
    on its own, the AE always tried), reads them back and joins.

    sensor: object with set_option(option, value) / get_option(option);
    options: object with the attributes enable_auto_exposure, exposure,
    gain (rs.option); frame_index: callable, the newest polled frame index.
    """

    def __init__(self, sensor, options, frame_index: Callable[[], int],
                 commanded: Optional[CommandedColorExposure] = None, *,
                 start_settings: Optional[dict] = None) -> None:
        self._sensor = sensor
        self._opt = options
        self._frame_index = frame_index
        self._commanded = commanded
        self.start_settings = dict(start_settings or {})
        self._q: deque = deque()
        self._cond = threading.Condition()
        self._stop = False
        self._restore = False
        self._manual = False
        self._thread: Optional[threading.Thread] = None
        self._events: deque = deque()
        self._failed: deque = deque()
        self.errors: list = []
        self.calls: list = []          # (option, value, call ms) of every write
        self.restored: Optional[dict] = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="colour-exposure", daemon=True)
            self._thread.start()

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, cmd: ExposureCommand) -> None:
        with self._cond:
            self._q.clear()               # only the newest command matters
            self._q.append(cmd)
            self._cond.notify()

    def poll_applied(self) -> list:
        """(command, frame index at return, t_return) of every write since
        the last call."""
        out = []
        while self._events:
            out.append(self._events.popleft())
        return out

    def poll_failed(self) -> list:
        """(command, frame index, t, error, setting read back) of every
        command whose writes raised since the last call; setting =
        (auto, exposure ms or None, gain or None) as the device reads back
        afterwards, None if the read-back failed too."""
        out = []
        while self._failed:
            out.append(self._failed.popleft())
        return out

    def stop(self, restore: bool = True, timeout_s: float = 1.0) -> bool:
        """Stop the thread (restore: write the start settings back first);
        False if it is still inside a set_option after timeout_s - it then
        restores on its own when the call returns."""
        with self._cond:
            self._stop = True
            self._restore = bool(restore)
            self._q.clear()
            self._cond.notify()
        t = self._thread
        if t is None:
            if restore and self.restored is None:
                self._do_restore()
            return True
        t.join(timeout_s)
        return not t.is_alive()

    def _set(self, option_name: str, value: float) -> None:
        opt = getattr(self._opt, option_name)
        t0 = time.perf_counter()
        self._sensor.set_option(opt, float(value))
        self.calls.append((option_name, float(value), (time.perf_counter() - t0) * 1000.0))

    def _read_back(self) -> dict:
        out = {}
        for name in ("enable_auto_exposure", "exposure", "gain"):
            try:
                out[name] = float(self._sensor.get_option(getattr(self._opt, name)))
            except Exception:
                pass
        return out

    @staticmethod
    def _setting_of(back: dict) -> Optional[tuple]:
        """(auto, exposure ms, gain) from a read-back, None if incomplete."""
        if "enable_auto_exposure" not in back:
            return None
        if back["enable_auto_exposure"] >= 0.5:
            return (True, None, None)
        if back.get("exposure", 0.0) <= 0.0:
            return None                   # no usable exposure: unknown
        ms = register_exposure_ms(back["exposure"])
        gain = back.get("gain")
        return (False, back["exposure"] / 10.0 if ms is None else ms,
                gain if gain is not None and gain > 0.0 else None)

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._q and not self._stop:
                    self._cond.wait()
                if self._stop:
                    break
                cmd = self._q.popleft()
            written = []
            t0 = time.perf_counter()
            try:
                if not self._manual:
                    # manual from the first attempt on: a call that raised
                    # may still have switched the AE off (the restore must
                    # then run)
                    self._manual = True
                    self._set("enable_auto_exposure", 0.0)
                    written.append("enable_auto_exposure")
                self._set("exposure", exposure_option_value(cmd.exposure_ms))
                written.append("exposure")
                self._set("gain", float(cmd.gain))
                cmd.write_ms = (time.perf_counter() - t0) * 1000.0
                idx = int(self._frame_index())
                if self._commanded is not None:
                    self._commanded.applied(False, cmd.exposure_ms, idx, cmd.gain)
                self._events.append((cmd, idx, time.perf_counter()))
            except Exception as exc:   # the write failed (device gone, call rejected)
                self.errors.append(repr(exc))
                idx = int(self._frame_index())
                setting = self._setting_of(self._read_back())
                if setting is None and "exposure" in written:
                    setting = (False, float(cmd.exposure_ms), None)   # exposure written, gain not
                # the frames carry what the device runs with now (unknown:
                # unchanged)
                if self._commanded is not None and setting is not None:
                    self._commanded.applied(setting[0], setting[1], idx, setting[2])
                self._failed.append((cmd, idx, time.perf_counter(), repr(exc), setting))
        if self._restore:
            self._do_restore()

    def _do_restore(self) -> None:
        """The start settings back, every write on its own (a failed one
        does not stop the others), the AE last; then the read-back:
        restored = what the device reads, plus "errors" and
        "restore_mismatch" (options that read back otherwise than written)
        when there are any. The frames carry the read-back setting."""
        if not self._manual and not self.start_settings.get("force", False):
            self.restored = {"skipped": "never switched to manual"}
            return
        s = self.start_settings
        errors = []
        want = {}
        # the camera start's order: manual values with the AE off, then the AE
        for name in ("exposure", "gain"):
            if s.get(name) is not None:
                want[name] = float(s[name])
                try:
                    self._set(name, want[name])
                except Exception as exc:
                    errors.append(f"{name}: {exc!r}")
        if s.get("enable_auto_exposure", True):
            want["enable_auto_exposure"] = 1.0
            try:
                self._set("enable_auto_exposure", 1.0)
            except Exception as exc:
                errors.append(f"enable_auto_exposure: {exc!r}")
        idx = int(self._frame_index())
        back = self._read_back()
        out: dict = dict(back)
        # the device reads 19 back for 20
        tol = {"exposure": 1.0}
        mismatch = [n for n, v in want.items() if n in back and abs(back[n] - v) > tol.get(n, 0.5)]
        mismatch += [n + " (no read-back)" for n in want if n not in back]
        if errors:
            out["errors"] = errors
            self.errors.extend("restore: " + e for e in errors)
        if mismatch:
            out["restore_mismatch"] = mismatch
        setting = self._setting_of(back)
        if setting is None:
            auto = bool(s.get("enable_auto_exposure", True)) and not errors
            setting = (True, None, None) if auto else None
        if self._commanded is not None and setting is not None:
            self._commanded.applied(setting[0], setting[1], idx, setting[2])
        self._manual = setting is None or not setting[0]
        self.restored = out


# ============================================================
# The control as the Live page and the evaluation script run it
# ============================================================

class MarkerExposureControl:
    """Controller + meter + the camera's actuator, wired as the Live page
    uses them (the evaluation script uses this class too, so both run the
    same control).

    start(worker): the camera hands over its colour sensor
    (RealSenseCameraSource.start_color_control, False without one or while
    a previous control is still restoring: unavailable says which), the
    worker gets the meter, the first setting is written. step(metering) in
    the GUI thread for every drained result: actuator reports in, the
    controller decides, a command goes to the actuator thread; returns the
    command or None. A failed write is dropped (the controller keeps what
    the device reads back); failure is set - and the caller must stop the
    control, which restores the AE - when the device reads the AE back
    after a failure, when the first command fails, or after
    MAX_FAILED_ROW failures in a row. stop(): meter off, the start
    settings written back (AE on), a summary for the logs.
    """

    MAX_FAILED_ROW = 2

    def __init__(self, camera, fresh: Callable[[Any], bool],
                 config: Optional[MarkerExposureConfig] = None) -> None:
        self._camera = camera
        self._fresh = fresh
        self._config = config or MarkerExposureConfig()
        self.controller: Optional[MarkerExposureController] = None
        self.actuator = None
        self.meter: Optional[MarkerMeter] = None
        self._worker = None
        self.gain_range: Optional[dict] = None
        self._errors_seen = 0
        self.n_commands = 0
        self.n_applied = 0
        self.unavailable = ""                  # why start() returned False
        self.failure = ""                      # set by step(): stop the control
        self.failures: list = []               # (seq, exposure, gain, error, setting) of failed writes
        self._first_write: Optional[ExposureCommand] = None

    @property
    def active(self) -> bool:
        return self.actuator is not None

    def start(self, worker, t_now: float) -> bool:
        cam = self._camera
        if getattr(cam, "color_control_busy", False):
            self.unavailable = "busy"           # the last control's restore still runs
            return False
        self.gain_range = cam.color_option_range("gain")
        rng = self.gain_range
        self.controller = MarkerExposureController(
            self._config, None if rng is None else (rng["min"], rng["max"]))
        act = cam.start_color_control()
        if act is None:
            self.controller = None
            self.unavailable = "busy" if getattr(cam, "color_control_busy", False) else "no colour sensor"
            return False
        self.actuator = act
        self.meter = MarkerMeter(self._fresh)
        self._worker = worker
        worker.set_meter(self.meter)
        act.submit(self.controller.start(t_now))
        self.n_commands = 1
        return True

    def step(self, metering) -> Optional[ExposureCommand]:
        ctl, act = self.controller, self.actuator
        if ctl is None or act is None or metering is None or self.failure:
            return None
        for cmd, index, _t in act.poll_applied():
            ctl.applied(cmd, index)
            self.n_applied += 1
            if self.n_applied == 1:
                self._first_write = cmd
        for cmd, index, _t, err, setting in act.poll_failed():
            manual = setting is not None and not setting[0] and setting[2] is not None
            ctl.failed(cmd, index, (setting[1], setting[2]) if manual else None)
            self.failures.append((cmd.seq, cmd.exposure_ms, cmd.gain, err, setting))
            if self.n_applied == 0:
                self.failure = "the first write failed"
            elif setting is not None and setting[0]:
                self.failure = "the camera reads the AE back after a failed write"
            elif ctl.n_failed_row >= self.MAX_FAILED_ROW:
                self.failure = f"{ctl.n_failed_row} writes in a row failed"
        if self.failure:
            return None
        cmd = ctl.update(metering)
        if cmd is not None:
            act.submit(cmd)
            self.n_commands += 1
        return cmd

    def take_first_write(self) -> Optional[ExposureCommand]:
        """The first command the camera applied, once (None before and
        after) - for one confirmation line in the logs."""
        cmd, self._first_write = self._first_write, None
        return cmd

    def new_errors(self) -> list:
        """actuator errors since the last call (a write that failed)."""
        act = self.actuator
        if act is None or len(act.errors) == self._errors_seen:
            return []
        out = act.errors[self._errors_seen:]
        self._errors_seen = len(act.errors)
        return out

    def snapshot(self) -> dict:
        """config, device gain range and start settings (session records)."""
        return {
            "config": self._config.snapshot(),
            "gain_range_device": self.gain_range,
            "gain_used": None if self.controller is None
            else [self.controller.gain_min, self.controller.gain_max],
            "start_settings": None if self.actuator is None else self.actuator.start_settings,
        }

    def text(self, metering=None) -> str:
        """One line for the profile log: state, setting, white level, clip."""
        ctl = self.controller
        if ctl is None:
            return ""
        m = metering if metering is not None else ctl.last_metering
        white = float("nan") if m is None else m.white_dn
        clip = float("nan") if m is None else m.clip_frac
        return (f"{ctl.state} {ctl.exposure_ms:g} ms/{ctl.gain:.0f} white "
                + ("-" if white != white else f"{white:.0f}")
                + " clip " + ("-" if clip != clip else f"{100 * clip:.0f}%"))

    def stop(self, timeout_s: float = 1.0) -> dict:
        """Summary: steps, commands, restored (the read-back; {"late":
        True} while the actuator is still inside a set_option), restore
        problems (restore_mismatch, restore errors), failed writes."""
        act, ctl = self.actuator, self.controller
        self.actuator = None
        if self._worker is not None:
            self._worker.set_meter(None)
            self._worker = None
        t = time.perf_counter()
        restored = None
        error = None
        if act is not None:
            try:
                restored = self._camera.stop_color_control(act, timeout_s)
            except Exception as exc:
                error = repr(exc)
        rd = restored if isinstance(restored, dict) else {}
        problems = []
        if error:
            problems.append("restore raised")
        if rd.get("late"):
            problems.append("restore late")
        if rd.get("errors"):
            problems.append("restore write failed")
        if "enable_auto_exposure" in " ".join(rd.get("restore_mismatch", [])):
            problems.append("AE not back on")
        return {
            "steps": 0 if ctl is None else ctl.n_steps,
            "search_steps": 0 if ctl is None else ctl.n_search_steps,
            "commands": self.n_commands,
            "applied": self.n_applied,
            "restored": restored,
            "restore_error": error,
            "restore_mismatch": rd.get("restore_mismatch"),
            # the camera may not run with the start settings: AE not back
            # on, a restore write failed, or the restore is still running
            "restore_problems": problems,
            "stop_ms": round((time.perf_counter() - t) * 1000.0, 2),
            "failure": self.failure,
            "failed_writes": [list(f[:4]) for f in self.failures],
            "actuator_errors": None if act is None else list(act.errors),
            "set_option_ms": None if act is None else [round(c[2], 2) for c in act.calls[-40:]],
        }
