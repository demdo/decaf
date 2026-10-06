# -*- coding: utf-8 -*-
"""
test_live_tracking.py

Checks the Live page's tracking thread (overlay.tools.live_tracking) with a
fake tracker - no camera, no C++ engine, no Qt.

The fake's process_frame sleeps a scripted time (time.sleep releases the
GIL like the real engine does) and checks that the slot still holds the
pattern it was submitted with, before and after the sleep: a slot reused
while the worker reads it fails the test.

(a) hand-off: newest frame wins, camera order, nothing processed twice,
    skipped counts add up, every slot back after stop()
(b) page mode: one image in the outbox, the pool of 5 never runs dry
(c) eval mode: every output keeps its image; a dry pool takes back the
    oldest untaken output's slot (counted, and not as an exhausted pool)
(d) epoch: a reset requested while a frame is in flight -> that output keeps
    the old epoch, reset() runs once, in the worker, before the next frame
(e) stop during a frame joins after it; a hung tracker -> stop() False
(f) start/stop toggles: no deadlock, no slot leak, no output of an old
    worker in a new one
(g) an exception -> one failed output with traceback, thread ended; a
    failure outside the tracker call -> ended, failure set, no output
(h) TrackingSession: texts, reset after 1.5 s of frame time however many
    outputs that is, hint at 3 s, old epochs ignored, re-entry flag,
    failure-log calls
(i) tracker_log classes for held detections, appearance rejects and holds
(j) ToolPoints: fused pose, offset carry within max_carry_s of frame time,
    RGB-only after it, offset dropped by the reprojection gate and reset();
    in the worker: points of every result, in order, reset with the tracker;
    the frame time goes to the tracker (t_frame)
(k) frame timing: frame_timing(frame.metadata) runs in submit() and its
    (frame_time_ms, ir_time_offset_ms) reach process_frame with that frame;
    without frame_timing process_frame gets neither; one that raises takes
    no slot
(l) ToolPoints: the empirical line-of-sight shift of both points, off on
    request (tool_points(los_offset_mm=0)), in the snapshot record

Run from src/:

    python -m overlay.tests.test_live_tracking [--long]

--long runs (b) for 60 s and (f) with 200 toggles at 0-200 ms.
"""

from __future__ import annotations

import random
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

from overlay.tools.live_tracking import (
    STATUS_LOST,
    STATUS_SEARCH_HINT,
    STATUS_SETTLING,
    STATUS_TRACKING,
    FrameSlot,
    TrackingSession,
    TrackingWorker,
    TrackOutput,
)
from overlay.tools.tracking_overlay import ToolPoints, TrackingOverlay


COLOR_SHAPE = (48, 64, 4)
IR_SHAPE = (36, 48)
FRAME_S = 1.0 / 30.0
ACCEPTED_KEY = "cpp_tracker_engine_current_pose_accepted_count"


# ============================================================
# Fakes
# ============================================================

def _pattern(frame_index: int) -> int:
    return frame_index % 251


def _frame(frame_index: int, with_ir: bool = True):
    v = _pattern(frame_index)
    return SimpleNamespace(
        image_bgr=np.full(COLOR_SHAPE, v, np.uint8),
        ir_left=np.full(IR_SHAPE, v, np.uint8) if with_ir else None,
        ir_right=np.full(IR_SHAPE, (v + 1) % 256, np.uint8) if with_ir else None,
        frame_index=frame_index,
    )


def _result(fresh: bool = True, *, source: str = "decode", converged: bool = True,
            message: str = "", timings: dict | None = None, tag=None):
    t = {ACCEPTED_KEY: 1.0 if fresh else 0.0}
    t.update(timings or {})
    return SimpleNamespace(
        success=fresh or source == "hold",
        pose_source=SimpleNamespace(value=source if (fresh or source == "hold") else "none"),
        pose_converged=converged,
        message=message,
        timings_ms=t,
        tag=tag,
    )


def _has_fresh_pose(result) -> bool:
    return bool(result.success) and result.pose_source.value not in ("none", "hold")


class FakeTracker:
    """process_frame sleeps sleep_s(frame_index) and checks its input."""

    def __init__(self, sleep_s=lambda i: 0.0, fail_at: int | None = None, tag=None):
        self.sleep_s = sleep_s
        self.fail_at = fail_at
        self.tag = tag
        self.calls: list[tuple[str, int, int]] = []    # (what, frame_index, thread id)
        self.errors: list[str] = []

    def _check(self, color, ir_left, ir_right, where: str) -> int:
        v = int(color[0, 0, 0])
        if not (color == v).all():
            self.errors.append(f"colour pattern broken {where}")
        if ir_left is not None and not ((ir_left == v).all() and (ir_right == (v + 1) % 256).all()):
            self.errors.append(f"IR pattern broken {where}")
        return v

    def process_frame(self, color, *, run_detection=True, ir_left=None, ir_right=None, t_frame=None):
        v = self._check(color, ir_left, ir_right, "before")
        self.calls.append(("process", v, threading.get_ident()))
        if self.fail_at is not None and v == _pattern(self.fail_at):
            raise RuntimeError("fake engine failure")
        time.sleep(self.sleep_s(v))
        if self._check(color, ir_left, ir_right, "after") != v:
            self.errors.append("slot changed during processing")
        return _result(True, tag=self.tag)

    def reset(self) -> None:
        self.calls.append(("reset", -1, threading.get_ident()))


def _wait_until(cond, timeout_s: float = 5.0) -> bool:
    t_end = time.perf_counter() + timeout_s
    while time.perf_counter() < t_end:
        if cond():
            return True
        time.sleep(0.002)
    return cond()


def _drain_all(worker: TrackingWorker) -> list[TrackOutput]:
    outs = worker.take_outputs()
    for out in outs:
        worker.release(out.slot)
    return outs


def _pool_balanced(worker: TrackingWorker) -> bool:
    pool = worker._pool
    return pool.n_free == pool.n_allocated


# ============================================================
# Worker
# ============================================================

def test_handoff() -> None:
    tracker = FakeTracker(lambda v: 0.035)
    worker = TrackingWorker(tracker, pool_size=5)
    worker.start()
    outs: list[TrackOutput] = []
    t0 = time.perf_counter()
    n = 90
    for i in range(n):
        while time.perf_counter() < t0 + i * FRAME_S:
            time.sleep(0.001)
        assert worker.submit(_frame(i), time.perf_counter())
        outs += _drain_all(worker)
    assert _wait_until(lambda: worker._pending is None and worker.n_processed + worker.n_skipped == n)
    time.sleep(0.05)
    outs += _drain_all(worker)
    assert worker.stop()
    outs += _drain_all(worker)

    idx = [o.frame_index for o in outs]
    assert idx == sorted(set(idx)), "processed out of order or twice"
    assert sum(o.skipped for o in outs) + len(outs) == worker.n_submitted == n
    assert worker.n_skipped == sum(o.skipped for o in outs)
    assert not tracker.errors, tracker.errors
    assert _pool_balanced(worker)
    assert worker.n_pool_exhausted == 0
    print(f"  (a) hand-off: {n} submitted, {len(outs)} processed, "
          f"{worker.n_skipped} skipped, pool {worker._pool.n_allocated}")


def _gui_loop(worker: TrackingWorker, duration_s: float, *, keep: bool, drain_every_s: float,
              n_hold: int = 2) -> dict:
    """Simulates the page: poll every 6.7 ms, a frame every 33.3 ms, drain,
    hold the newest output with an image (page: pose frame + shown frame)."""
    held: list = []
    stats = {"outs": 0, "max_images_per_drain": 0, "submit_false": 0, "dropped": 0}
    t0 = time.perf_counter()
    next_frame, next_drain, i = t0, t0, 0
    while time.perf_counter() - t0 < duration_s:
        now = time.perf_counter()
        if now >= next_frame:
            if not worker.submit(_frame(i), now):
                stats["submit_false"] += 1
            i += 1
            next_frame += FRAME_S
        if now >= next_drain:
            outs = worker.take_outputs()
            stats["outs"] += len(outs)
            with_image = [o for o in outs if o.slot is not None]
            stats["max_images_per_drain"] = max(stats["max_images_per_drain"], len(with_image))
            stats["dropped"] += sum(1 for o in outs if o.frame_dropped)
            if keep:
                for o in outs:
                    assert o.slot is not None or o.frame_dropped
            for o in outs[:-1]:
                worker.release(o.slot)
            if outs:
                last = outs[-1]
                if last.slot is not None:
                    held.append(last.slot)
                while len(held) > n_hold - 1:
                    worker.release(held.pop(0))
            next_drain += drain_every_s
        time.sleep(0.0067)
    stats["frames"] = i
    worker.stop()
    for s in held:
        worker.release(s)
    _drain_all(worker)
    return stats


def test_page_mode(duration_s: float) -> None:
    rng = random.Random(1)
    tracker = FakeTracker(lambda v: rng.uniform(0.005, 0.060))
    worker = TrackingWorker(tracker, pool_size=5)
    worker.start()
    st = _gui_loop(worker, duration_s, keep=False, drain_every_s=0.0067)
    assert not tracker.errors, tracker.errors
    assert st["max_images_per_drain"] <= 1
    assert st["submit_false"] == 0 and worker.n_pool_exhausted == 0, st
    assert _pool_balanced(worker)
    print(f"  (b) page mode {duration_s:.0f} s: {st['frames']} frames, {st['outs']} processed, "
          f"pool {worker._pool.n_allocated}, exhausted {worker.n_pool_exhausted}")


def test_eval_mode() -> None:
    tracker = FakeTracker(lambda v: 0.010)
    worker = TrackingWorker(tracker, pool_size=8, keep_every_frame=True)
    worker.start()
    # draining every 0.4 s lets ~12 outputs pile up: the pool of 8 runs dry
    st = _gui_loop(worker, 3.0, keep=True, drain_every_s=0.4)
    assert not tracker.errors, tracker.errors
    assert worker.n_frame_dropped > 0 and st["dropped"] == worker.n_frame_dropped, st
    assert st["submit_false"] == 0 and worker.n_pool_exhausted == 0, st
    assert _pool_balanced(worker)
    print(f"  (c) eval mode: {st['outs']} processed, {worker.n_frame_dropped} images taken back")


def test_epoch() -> None:
    tracker = FakeTracker(lambda v: 0.100)
    worker = TrackingWorker(tracker, pool_size=5)
    worker.start()
    worker.submit(_frame(1), time.perf_counter())
    assert _wait_until(lambda: len(tracker.calls) == 1)
    epoch = worker.request_reset()
    worker.submit(_frame(2), time.perf_counter())
    assert _wait_until(lambda: worker.n_processed == 2)
    outs = _drain_all(worker)
    assert worker.stop()
    assert [(o.frame_index, o.epoch, o.reset) for o in outs] == [(1, 0, False), (2, epoch, True)]
    whats = [c[0] for c in tracker.calls]
    assert whats == ["process", "reset", "process"], whats
    assert len({c[2] for c in tracker.calls}) == 1
    assert tracker.calls[0][2] != threading.get_ident()
    print("  (d) epoch: in-flight output keeps epoch 0, reset once in the worker before frame 2")


def test_stop() -> None:
    tracker = FakeTracker(lambda v: 0.120)
    worker = TrackingWorker(tracker, pool_size=5)
    worker.start()
    worker.submit(_frame(1), time.perf_counter())
    assert _wait_until(lambda: len(tracker.calls) == 1)
    t = time.perf_counter()
    assert worker.stop(timeout_s=2.0)
    dt = time.perf_counter() - t
    assert dt < 0.3, dt
    _drain_all(worker)
    assert _pool_balanced(worker)

    hung = FakeTracker(lambda v: 5.0)
    worker = TrackingWorker(hung, pool_size=5)
    worker.start()
    worker.submit(_frame(1), time.perf_counter())
    assert _wait_until(lambda: len(hung.calls) == 1)
    t = time.perf_counter()
    assert not worker.stop(timeout_s=2.0)
    dt2 = time.perf_counter() - t
    assert 1.9 < dt2 < 2.5, dt2
    assert worker.alive
    print(f"  (e) stop: returned {1000 * dt:.0f} ms after the call (120 ms frame in "
          f"progress); hung tracker -> False after {dt2:.1f} s")


def test_toggles(n: int, max_gap_s: float) -> None:
    rng = random.Random(2)
    stop_feed = threading.Event()
    current: dict = {"worker": None}
    lock = threading.Lock()

    def feed() -> None:
        i = 0
        while not stop_feed.is_set():
            with lock:
                w = current["worker"]
                if w is not None:
                    w.submit(_frame(i), time.perf_counter())
            i += 1
            time.sleep(FRAME_S)

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    leaks = 0
    for k in range(n):
        tracker = FakeTracker(lambda v: rng.uniform(0.005, 0.040), tag=k)
        w = TrackingWorker(tracker, pool_size=5)
        w.start()
        with lock:
            current["worker"] = w
        time.sleep(rng.uniform(0.0, max_gap_s))
        for o in _drain_all(w):
            assert o.result.tag == k
        with lock:
            current["worker"] = None
        assert w.stop(timeout_s=2.0)
        for o in _drain_all(w):
            assert o.result.tag == k
        assert not tracker.errors, tracker.errors
        leaks += 0 if _pool_balanced(w) else 1
    stop_feed.set()
    feeder.join()
    assert leaks == 0, leaks
    print(f"  (f) {n} start/stop toggles: no deadlock, no slot leak")


def test_exception() -> None:
    tracker = FakeTracker(lambda v: 0.005, fail_at=3)
    worker = TrackingWorker(tracker, pool_size=5)
    worker.start()
    outs = []
    for i in range(6):
        worker.submit(_frame(i), time.perf_counter())
        time.sleep(0.03)
        outs += _drain_all(worker)
    assert _wait_until(lambda: not worker.alive)
    outs += _drain_all(worker)
    failed = [o for o in outs if o.error is not None]
    assert len(failed) == 1 and failed[0].frame_index == 3
    assert "fake engine failure" in failed[0].traceback
    assert outs[-1] is failed[0]
    assert worker.ended and worker.failure is None
    # not taken because the worker has ended - not an exhausted pool
    assert not worker.submit(_frame(9), time.perf_counter())
    assert worker.n_pool_exhausted == 0
    assert worker.stop()
    assert _pool_balanced(worker)

    class PublishFails(TrackingWorker):
        def _publish(self, out):
            raise MemoryError("fake publish failure")

    worker = PublishFails(FakeTracker(lambda v: 0.0), pool_size=5)
    worker.start()
    worker.submit(_frame(1), time.perf_counter())
    assert _wait_until(lambda: worker.ended)
    assert isinstance(worker.failure, MemoryError)
    assert "fake publish failure" in worker.failure_traceback
    assert worker.take_outputs() == []
    assert worker.stop()
    print("  (g) exception: one failed output with traceback, thread ended, slots back; "
          "a failure outside the tracker call ends it with failure set, no output")


# ============================================================
# Session
# ============================================================

class FakeFailLog:
    def __init__(self):
        self.calls: list[tuple] = []

    def begin_episode(self, episode, frame_idx):
        self.calls.append(("begin", episode, frame_idx))

    def note(self, result, frame_idx, stale):
        self.calls.append(("note", frame_idx, stale))

    def note_appearance(self, result, frame_idx, reentry):
        self.calls.append(("appearance", frame_idx, reentry))

    def flush(self, reason, frame_idx, n_results, final=False):
        self.calls.append(("flush", reason, frame_idx, n_results, final))


def _out(frame_index: int, t_frame: float, epoch: int, result) -> TrackOutput:
    slot = FrameSlot(None, (1, 1, 4), None)
    slot.frame_index = frame_index
    slot.t_frame = t_frame
    return TrackOutput(slot, epoch, result=result)


def _session(log=None):
    epochs = {"n": 0}

    def request_reset() -> int:
        epochs["n"] += 1
        return epochs["n"]

    s = TrackingSession(_has_fresh_pose, request_reset, lost_reset_s=1.5,
                        search_hint_s=3.0, fail_log=log)
    s.start(0.0)
    return s


def test_session() -> None:
    # search hint at 3.0 s of frame time, once
    s = _session()
    steps = [s.update(_out(i, 0.1 * i, 0, _result(False))) for i in range(40)]
    hints = [i for i, st in enumerate(steps) if st.hint]
    assert hints == [30] and steps[30].status == STATUS_SEARCH_HINT, hints

    # lock (settling), converge, then a loss: the reset comes after 1.5 s of
    # frame time, whether that is 10, 45 or 90 outputs
    for n_out in (10, 45, 90):
        log = FakeFailLog()
        s = _session(log)
        st = s.update(_out(0, 0.0, 0, _result(True, converged=False)))
        assert st.locked_now and not st.relock and st.status == STATUS_SETTLING
        st = s.update(_out(1, 0.033, 0, _result(True, converged=True)))
        assert st.status == STATUS_TRACKING and not st.locked_now
        assert s.update(_out(2, 0.066, 0, _result(True))).status is None
        dt = 1.5 / n_out
        t_fresh = 0.066
        resets = []
        texts = []
        for k in range(1, n_out + 5):
            st = s.update(_out(2 + k, t_fresh + k * dt + 1e-9, s.epoch, _result(False)))
            texts.append(st.status)
            if st.reset:
                resets.append(k)
                break
        assert resets == [n_out], (n_out, resets)
        # the dropout text from the first output without a pose, once
        assert texts == [STATUS_LOST] + [None] * (n_out - 1), texts
        assert s.epoch == 1 and not s.locked
        flushes = [c for c in log.calls if c[0] == "flush"]
        # the episode's results: 3 fresh + n_out stale
        assert flushes == [("flush", "lost (reset)", 2 + n_out, 3 + n_out, False)], flushes
        assert sum(1 for c in log.calls if c[0] == "note") == n_out
        # outputs of the old epoch, still in flight, change nothing
        st = s.update(_out(100, 9.0, 0, _result(True)))
        assert not st.current and not s.locked and s.n_old_epoch == 1
        # re-lock in the new epoch
        st = s.update(_out(101, 9.1, 1, _result(True)))
        assert st.locked_now and st.relock and st.reentry and s.episode == 2
        s.finish("stopped")
        assert log.calls[-1] == ("flush", "stopped", 101, 1, True)
        assert [c for c in log.calls if c[0] == "begin"] == [("begin", 1, 0), ("begin", 2, 101)]

    # a dropout shorter than the reset window: lost text at once, the
    # tracking text back with the pose, same episode, no reset, no flush
    log = FakeFailLog()
    s = _session(log)
    s.update(_out(0, 0.0, 0, _result(True, converged=True)))
    st = s.update(_out(1, 0.033, 0, _result(False)))
    assert st.status == STATUS_LOST and not st.reset and s.locked
    assert s.update(_out(2, 0.066, 0, _result(False))).status is None
    st = s.update(_out(3, 0.1, 0, _result(True, converged=True)))
    assert st.recovered and st.status == STATUS_TRACKING and not st.locked_now
    assert s.update(_out(4, 0.133, 0, _result(True, converged=True))).status is None
    assert s.episode == 1 and s.epoch == 0 and s.resets == 0
    assert not [c for c in log.calls if c[0] == "flush"]
    # a convergence change on the recovering output sets the text as such
    s.update(_out(5, 0.166, 0, _result(False)))
    st = s.update(_out(6, 0.2, 0, _result(True, converged=False)))
    assert st.status == STATUS_SETTLING and not st.recovered

    # re-entry flag = previous current-epoch output had no accepted pose
    s = _session()
    flags = [s.update(_out(i, 0.033 * i, 0, _result(f))).reentry
             for i, f in enumerate((False, True, True, False, True))]
    assert flags == [True, True, False, False, True], flags

    # no failure-log notes before the first lock
    log = FakeFailLog()
    s = _session(log)
    for i in range(5):
        s.update(_out(i, 0.033 * i, 0, _result(False)))
    assert all(c[0] == "appearance" for c in log.calls)
    print("  (h) session: hint at 3 s, lost text at once, reset after 1.5 s for "
          "10/45/90 outputs, recovery before it, old epoch ignored, re-entry flag, "
          "log calls")


# ============================================================
# tracker_log classes
# ============================================================

def test_tracker_log_classes() -> None:
    from tracking.hydramarker import tracker_log as tl

    def res(success, message, source="none", **timings):
        return SimpleNamespace(success=success, message=message,
                               pose_source=SimpleNamespace(value=source),
                               timings_ms=dict(timings))

    cf = tl.classify_failure
    # existing keys unchanged
    assert cf(res(False, "No valid decoded patches.")) == ("PATCH_DECODER", "NO_VALID_DECODED_PATCHES")
    assert cf(res(False, "Correspondence build failed.")) == ("CORRESPONDENCE", "CORRESPONDENCE_REJECTED")
    assert cf(res(False, "Idle: checkerboard detection skipped.")) == ("IDLE", "DETECTION_SKIPPED")
    assert cf(res(False, "Rotation jump too large (31.0 deg).")) == ("MOTION_GATE", "ROTATION_JUMP_TOO_LARGE")
    assert cf(res(False, "something new")) == ("OTHER", "OTHER")
    assert cf(res(True, "")) == ("OK", "OK")
    # new keys
    assert cf(res(False, "No valid checkerboard detection.")) == ("DETECTION", "NO_VALID_DETECTION")
    assert cf(res(False, "No valid checkerboard detection.", held_detection_count=1.0)) == \
        ("DETECTION", "HELD_DETECTION")
    assert cf(res(False, "Appearance check failed (parity=0.12, dot=0.30, cells=40)",
                  appearance_reject_count=1.0)) == ("APPEARANCE", "APPEARANCE_CHECK_FAILED")

    ch = tl.classify_hold
    assert ch(res(True, "Pose held from last accepted pose without checkerboard detection.",
                  "hold")) == ("POSE_HELD", "HOLD:NO_VALID_DETECTION")
    assert ch(res(True, "Pose held from last accepted pose without checkerboard detection.",
                  "hold", held_detection_count=1.0)) == ("POSE_HELD", "HOLD:HELD_DETECTION")
    assert ch(res(True, "Emergency pose held from last accepted pose after: Appearance check "
                        "failed (parity=0.1, dot=0.2, cells=30) (age=2, blue_align=0).",
                  "hold")) == ("POSE_HELD", "EMERGENCY_HOLD:APPEARANCE_CHECK_FAILED")
    assert ch(res(True, "Emergency pose held from last accepted pose after: No valid "
                        "checkerboard detection (age=1, no_align).", "hold")) == \
        ("POSE_HELD", "EMERGENCY_HOLD:NO_VALID_DETECTION")
    assert ch(res(True, "Emergency pose held from last accepted pose after: Correspondence "
                        "build failed (age=1).", "hold")) == \
        ("POSE_HELD", "EMERGENCY_HOLD:CORRESPONDENCE_REJECTED")

    r = res(True, "", "fast", fast_appearance_reject_count=1.0,
            persistent_appearance_undecided_count=1.0, fast_appearance_parity=0.2,
            fast_appearance_dot=0.1, fast_appearance_cells=12.0)
    assert tl.appearance_rejects(r) == ("fast",)
    assert tl.appearance_undecided(r) == ("persistent",)
    assert tl.appearance_values(r, "fast") == {"parity": 0.2, "dot": 0.1, "cells": 12.0}
    assert tl.appearance_rejects(res(False, "x", appearance_reject_count=1.0)) == ("decode",)
    assert tl.appearance_rejects(res(False, "x")) == ()
    print("  (i) tracker_log: old keys unchanged, held / appearance / hold-cause keys")


# ============================================================
# Drawn points
# ============================================================

P_TIP = np.array([12.0, -179.0, -1.0])
P_BACK = np.array([12.0, -129.0, -1.0])


def _pose_result(t_rgb, *, applied: bool, d_fused=(0.0, 0.0, 0.0), gate: bool = False,
                 ir: bool = True):
    """Identity rotation; RGB-only translation t_rgb, the fusion adds d_fused."""
    t_rgb = np.asarray(t_rgb, np.float64)
    t_out = t_rgb + np.asarray(d_fused, np.float64) if applied else t_rgb
    timings = {}
    if ir:
        timings = {"ir_applied": 1.0 if applied else 0.0,
                   "ir_reproj_gate_reject": 1.0 if gate else 0.0}
    return SimpleNamespace(
        rvec=np.zeros(3), tvec=t_out,
        rvec_prefusion=np.zeros(3) if ir else None,
        tvec_prefusion=t_rgb if ir else None,
        timings_ms=timings,
    )


def test_tool_points() -> None:
    los, ToolPoints.LOS_OFFSET_MM = ToolPoints.LOS_OFFSET_MM, 0.0    # geometry below without it
    try:
        _check_tool_points()
    finally:
        ToolPoints.LOS_OFFSET_MM = los
    # the line-of-sight shift: both points toward the camera along the ray to
    # the marker origin (of the RGB-only pose under a carried offset)
    t0 = np.array([0.0, 0.0, 400.0])
    d = np.array([0.1, -0.2, 2.0])
    tp = ToolPoints(P_TIP, P_BACK, los_offset_mm=2.0)
    pts = tp.update(_pose_result(t0, applied=False, ir=False), 0.0)
    assert np.allclose(pts, [P_TIP + t0 - [0.0, 0.0, 2.0], P_BACK + t0 - [0.0, 0.0, 2.0]])
    pts = tp.update(_pose_result(t0, applied=True, d_fused=d), 0.1)
    assert tp.kind == "fused" and np.allclose(pts[0], P_TIP + t0 + d - [0.0, 0.0, 2.0])
    assert ToolPoints(P_TIP, P_BACK).los_offset_mm == ToolPoints.LOS_OFFSET_MM
    ov = TrackingOverlay(np.eye(3), None, P_TIP, [0.0, 2.0, 0.0])
    assert ov.tool_points().los_offset_mm == ToolPoints.LOS_OFFSET_MM
    assert ov.tool_points(los_offset_mm=0.0).los_offset_mm == 0.0
    ov.update(np.stack([P_TIP + t0, P_BACK + t0]))
    assert ov.snapshot_record()["los_offset_mm"] == ToolPoints.LOS_OFFSET_MM
    print("  (l) line-of-sight shift of the drawn points (LOS_OFFSET_MM)")


def _check_tool_points() -> None:
    tp = ToolPoints(P_TIP, P_BACK, max_carry_s=0.5)
    back = P_BACK
    t0 = np.array([0.0, 0.0, 400.0])
    d = np.array([0.1, -0.2, 2.0])

    pts = tp.update(_pose_result(t0, applied=True, d_fused=d), 0.0)
    assert tp.kind == "fused"
    assert np.allclose(pts, [P_TIP + t0 + d, back + t0 + d])
    # no fusion: RGB-only pose + the fused-minus-RGB offset, the drill moved 1 mm
    t1 = t0 + [1.0, 0.0, 0.0]
    pts = tp.update(_pose_result(t1, applied=False), 0.5)
    assert tp.kind == "carry" and np.allclose(pts, [P_TIP + t1 + d, back + t1 + d])
    # past max_carry_s of frame time after the last fused frame: RGB-only
    pts = tp.update(_pose_result(t1, applied=False), 0.51)
    assert tp.kind == "rgb" and np.allclose(pts, [P_TIP + t1, back + t1])
    # IR stage not run at all (no pair): carried like a rejected fusion
    tp.update(_pose_result(t0, applied=True, d_fused=d), 1.0)
    pts = tp.update(_pose_result(t1, applied=False, ir=False), 1.1)
    assert tp.kind == "carry" and np.allclose(pts[0], P_TIP + t1 + d)
    # the reprojection gate: RGB-only, and the offset is gone afterwards
    pts = tp.update(_pose_result(t1, applied=False, gate=True), 1.2)
    assert tp.kind == "gate" and np.allclose(pts[0], P_TIP + t1)
    tp.update(_pose_result(t1, applied=False), 1.3)
    assert tp.kind == "rgb"
    # reset drops the offset
    tp.update(_pose_result(t0, applied=True, d_fused=d), 2.0)
    tp.reset()
    tp.update(_pose_result(t1, applied=False), 2.1)
    assert tp.kind == "rgb"
    # no pose
    assert tp.update(SimpleNamespace(rvec=None, tvec=None, timings_ms={}), 3.0) is None
    assert tp.kind is None
    # the default carry age; the overlay hands over its own tip and back point
    assert ToolPoints(P_TIP, P_BACK).max_carry_s == ToolPoints.MAX_CARRY_S
    ov = TrackingOverlay(np.eye(3), None, P_TIP, [0.0, 2.0, 0.0])
    pts = ov.tool_points().update(_pose_result(t0, applied=False, ir=False), 0.0)
    assert np.allclose(pts, [P_TIP + t0, P_TIP + [0.0, TrackingOverlay.AXIS_BACK_MM, 0.0] + t0])

    # in the worker: every result gets its points, in order, with its frame
    # time; a requested reset resets ToolPoints with the tracker
    class PoseTracker(FakeTracker):
        def process_frame(self, color, *, run_detection=True, ir_left=None, ir_right=None, t_frame=None):
            v = int(color[0, 0, 0])
            self.calls.append(("process", v, threading.get_ident()))
            self.t_frames.append(t_frame)
            # frames 0, 1 fused; 2, 3 not
            return _pose_result(t0 + [v, 0.0, 0.0], applied=v < 2, d_fused=d)

    tracker = PoseTracker()
    tracker.t_frames = []
    worker = TrackingWorker(tracker, pool_size=5, tool_points=ToolPoints(P_TIP, P_BACK, max_carry_s=0.5))
    worker.start()
    outs = []
    for i in range(4):
        if i == 3:
            worker.request_reset()
        worker.submit(_frame(i), 0.1 * i)
        assert _wait_until(lambda: worker.n_processed == i + 1)
        outs += _drain_all(worker)
    assert worker.stop()
    assert [o.points_kind for o in outs] == ["fused", "fused", "carry", "rgb"], \
        [o.points_kind for o in outs]
    assert np.allclose(outs[2].points[0], P_TIP + t0 + [2.0, 0.0, 0.0] + d)
    assert np.allclose(outs[3].points[0], P_TIP + t0 + [3.0, 0.0, 0.0])
    # the tracker gets each frame's time (time base of its output filter)
    assert tracker.t_frames == [0.1 * i for i in range(4)], tracker.t_frames
    assert _pool_balanced(worker)
    print("  (j) tool points: fused, carry for 0.5 s of frame time, then RGB-only; "
          "gate and reset drop the offset; computed in the worker, reset with the tracker")


def test_frame_timing() -> None:
    class TimingTracker(FakeTracker):
        def __init__(self):
            super().__init__()
            self.timing = []

        def process_frame(self, color, *, run_detection=True, ir_left=None, ir_right=None,
                          t_frame=None, **timing):
            self.timing.append((int(color[0, 0, 0]), timing))
            return _result(True)

    def frame(i):
        f = _frame(i)
        f.metadata = {"color_time_ms": 1000.0 + 33.4 * i}
        return f

    tracker = TimingTracker()
    seen = []

    def timing(meta):
        seen.append(threading.get_ident())
        return meta["color_time_ms"], -8.0

    worker = TrackingWorker(tracker, pool_size=5, frame_timing=timing)
    worker.start()
    for i in range(3):
        worker.submit(frame(i), 0.1 * i)
        assert _wait_until(lambda: worker.n_processed == i + 1)
        _drain_all(worker)
    assert worker.stop()
    assert tracker.timing == [
        (i, {"frame_time_ms": 1000.0 + 33.4 * i, "ir_time_offset_ms": -8.0}) for i in range(3)
    ], tracker.timing
    assert seen == [threading.get_ident()] * 3     # in submit(), the caller's thread

    tracker = TimingTracker()
    worker = TrackingWorker(tracker, pool_size=5)
    worker.start()
    worker.submit(_frame(0), 0.0)
    assert _wait_until(lambda: worker.n_processed == 1)
    _drain_all(worker)
    assert worker.stop()
    assert tracker.timing == [(0, {})], tracker.timing
    assert _pool_balanced(worker)

    # a frame_timing that raises: the error reaches the caller, no slot is lost
    def broken(meta):
        raise ValueError("no camera time")

    worker = TrackingWorker(TimingTracker(), pool_size=5, frame_timing=broken)
    worker.start()
    try:
        worker.submit(frame(0), 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("frame_timing error swallowed")
    assert worker.stop()
    assert worker.n_submitted == 0 and _pool_balanced(worker)
    print("  (k) frame timing: computed in submit(), handed to process_frame with its "
          "frame; nothing without frame_timing; a raising one loses no slot")


def main() -> None:
    long = "--long" in sys.argv
    print("test_live_tracking")
    test_handoff()
    test_page_mode(60.0 if long else 5.0)
    test_eval_mode()
    test_epoch()
    test_stop()
    test_toggles(200 if long else 40, 0.2 if long else 0.05)
    test_exception()
    test_session()
    test_tracker_log_classes()
    test_tool_points()
    test_frame_timing()
    print("all checks passed")


if __name__ == "__main__":
    main()
