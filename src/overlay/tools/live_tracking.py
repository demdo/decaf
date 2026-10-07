# overlay/tools/live_tracking.py
"""Drill tracking of the Live page in its own thread.

The GUI thread keeps the camera: it polls, shows the newest image and
hands a copy of every polled frame to TrackingWorker. The worker runs the
tracker on the newest frame it has and returns every result, in
processing order, through an outbox the GUI thread drains on its timer.
TrackingSession turns those results into the page's tracking state
(searching, tracking, engine reset after a loss, status texts), also in
the GUI thread, so every widget, the overlay and the logs keep a single
writer. The drill's camera-frame points (tracking_overlay.ToolPoints) are
computed in the worker, right after the tracker, and travel with the
result.

Qt-free and engine-free on purpose: the evaluation script and the unit
test use exactly these classes, without a QApplication, a camera or the
C++ build. The tracker is whatever object has process_frame() (with the
frame time as t_frame) and
reset() (HydraTracker on the page).
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ============================================================
# Frame copies
# ============================================================

class FrameSlot:
    """
    One pre-allocated copy of a camera frameset: colour image + IR pair.

    The camera's poll() hands out views that die with the next poll, so the
    worker gets copies. A slot has exactly one owner at a time - the GUI
    thread filling it, the pending hand-off, the worker, the outbox, or the
    page holding it for the display - and changes owner only under the
    worker's lock; the owner reads and writes the arrays without a lock.
    """

    __slots__ = (
        "pool", "color", "ir_left", "ir_right", "has_ir",
        "frame_index", "t_frame", "meta", "timing",
    )

    def __init__(self, pool: "_FramePool", color_shape: tuple, ir_shape: tuple | None):
        self.pool = pool
        self.color = np.empty(color_shape, np.uint8)
        self.ir_left = None if ir_shape is None else np.empty(ir_shape, np.uint8)
        self.ir_right = None if ir_shape is None else np.empty(ir_shape, np.uint8)
        self.has_ir = False
        self.frame_index = -1
        self.t_frame = 0.0
        self.meta: dict | None = None
        self.timing: tuple | None = None

    def fits(self, color_shape: tuple, ir_shape: tuple | None) -> bool:
        if self.color.shape != tuple(color_shape):
            return False
        if ir_shape is None:
            return True
        return self.ir_left is not None and self.ir_left.shape == tuple(ir_shape)


class _FramePool:
    """
    At most `cap` slots, recycled instead of reallocated (~10 MB per
    1920x1080 BGRA frame + IR pair; no allocation per frame).
    """

    def __init__(self, cap: int):
        self.cap = max(1, int(cap))
        self._lock = threading.Lock()
        self._free: list[FrameSlot] = []
        self.n_allocated = 0

    def acquire(self, color_shape: tuple, ir_shape: tuple | None) -> FrameSlot | None:
        with self._lock:
            while self._free:
                slot = self._free.pop()
                if slot.fits(color_shape, ir_shape):
                    return slot
                # stream format changed: drop the old buffers
                self.n_allocated -= 1
            if self.n_allocated >= self.cap:
                return None
            self.n_allocated += 1
        return FrameSlot(self, color_shape, ir_shape)

    def release(self, slot: FrameSlot | None) -> None:
        if slot is None or slot.pool is not self:
            return
        slot.meta = None
        with self._lock:
            self._free.append(slot)

    @property
    def n_free(self) -> int:
        with self._lock:
            return len(self._free)


# ============================================================
# Results
# ============================================================

class TrackOutput:
    """
    One processed frame: the tracker result plus where and when it came from.

    frame_index / t_frame / meta are copied out of the slot because the slot
    can be given back before the output is read (only the newest output
    keeps its image on the page). t_frame is the perf_counter() of the poll
    that delivered the frame, t_start / t_done bracket the tracker call in
    the worker (reset included), skipped counts the newer frames that
    replaced a pending one since the previous processed frame. points /
    points_kind / points_marked: the worker's ToolPoints.update(), .kind and
    .marked for this result (None / None / False without tool_points or
    without a pose).
    """

    __slots__ = (
        "slot", "frame_index", "t_frame", "meta", "has_ir", "epoch",
        "result", "t_start", "t_done", "skipped", "reset",
        "error", "traceback", "frame_dropped", "points", "points_kind",
        "points_marked",
    )

    def __init__(self, slot: FrameSlot, epoch: int, *, result=None,
                 t_start: float = 0.0, t_done: float = 0.0, skipped: int = 0,
                 reset: bool = False, error: BaseException | None = None,
                 traceback_text: str | None = None, points=None,
                 points_kind: str | None = None, points_marked: bool = False) -> None:
        self.slot: FrameSlot | None = slot
        self.frame_index = slot.frame_index
        self.t_frame = slot.t_frame
        self.meta = slot.meta
        self.has_ir = slot.has_ir
        self.epoch = epoch
        self.result = result
        self.t_start = t_start
        self.t_done = t_done
        self.skipped = skipped
        self.reset = reset
        self.error = error
        self.traceback = traceback_text
        self.points = points
        self.points_kind = points_kind
        self.points_marked = points_marked
        # eval mode: the slot was taken back for a newer frame before the
        # output was drained (its image is gone, the result is not)
        self.frame_dropped = False

    @property
    def image(self) -> np.ndarray | None:
        return None if self.slot is None else self.slot.color


# ============================================================
# Worker thread
# ============================================================

class TrackingWorker:
    """
    Runs tracker.process_frame on the newest submitted frame, in its own
    thread, one frame at a time, in camera order.

    submit()        GUI thread, per polled frame: copy into a slot; a newer
                    submit replaces a pending frame (counted as skipped), so
                    the tracker always works on the newest frame - the same
                    policy the camera's queue of size 1 imposes.
    take_outputs()  GUI thread: every result since the last call, in
                    processing order. The caller owns the returned slots and
                    gives back each one it does not keep (release()).
    request_reset() GUI thread: the tracker is reset before the next frame,
                    in the worker; every output carries the epoch it was
                    computed in, so results of frames already in flight can
                    be told apart.
    stop()          GUI thread: ends the thread after the current frame.

    The thread ends by itself only on an error: a tracker exception comes
    as an output with error set; anything else (outside the tracker call)
    sets failure, and ended turns True without such an output.

    keep_every_frame=False (page): only the newest untaken output keeps its
    image, older ones give their slot back at once. True (evaluation
    script): every output keeps its image for the raw recording; when the
    pool runs dry a submit takes back the oldest untaken output's slot
    (frame_dropped) instead of skipping the frame, so the tracker input is
    the page's.

    tool_points: tracking_overlay.ToolPoints or None. Runs in this thread
    on every result, right after the tracker, with the frame time
    (TrackOutput.points / points_kind / points_marked), and is reset with
    the tracker.

    frame_timing: HydraTracker.ir_frame_timing or None. Called in submit()
    with the frame's metadata, before a slot is taken (if it raises, no
    slot is lost); its (frame_time_ms, ir_time_offset_ms) go
    with the frame to process_frame (IR time-offset correction). None:
    process_frame gets neither.

    The tracker belongs to this thread while it runs: nobody else may call
    it until stop() returned True.
    """

    OUTBOX_MAX = 64

    def __init__(self, tracker, *, pool_size: int, keep_every_frame: bool = False,
                 notify: Callable[[], None] | None = None, tool_points=None,
                 frame_timing: Callable[[dict], tuple] | None = None) -> None:
        self._tracker = tracker
        self._tool_points = tool_points
        self._frame_timing = frame_timing
        self._pool = _FramePool(pool_size)
        self._keep_every_frame = bool(keep_every_frame)
        self._notify = notify

        self._cond = threading.Condition()
        self._pending: FrameSlot | None = None
        self._outbox: deque[TrackOutput] = deque()
        self._stopping = False
        self._ended = False
        self._epoch = 0            # epoch of the worker's current frame
        self._reset_epoch = 0      # newest requested epoch
        self._skipped_since = 0
        self._thread: threading.Thread | None = None

        # set by the worker thread before it ends (see ended)
        self.failure: BaseException | None = None
        self.failure_traceback: str | None = None

        self.n_submitted = 0
        self.n_skipped = 0
        self.n_processed = 0
        self.n_outbox_dropped = 0
        self.n_frame_dropped = 0
        # A frame not taken for want of a free slot. Not part of the design
        # (the page's pool covers every owner at once); counted so a log
        # shows it. GUI thread only.
        self.n_pool_exhausted = 0

    # ---------------- GUI thread ----------------

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def ended(self) -> bool:
        """The thread has finished its loop: after stop(), or by itself on
        an error. Every output it produced is in the outbox by then."""
        with self._cond:
            return self._ended

    def start(self) -> None:
        if self._thread is not None:
            return
        # daemon: a worker stuck in the engine never blocks the interpreter
        # exit; normal shutdown goes through stop()
        self._thread = threading.Thread(
            target=self._run, name="drill-tracking", daemon=True
        )
        self._thread.start()

    def submit(self, frame, t_frame: float, meta: dict | None = None) -> bool:
        """Copy one polled frame (colour + IR pair of the same frameset) and
        hand it to the worker. False: not taken (no free slot - counted in
        n_pool_exhausted - or the worker is stopping or has ended)."""
        color = frame.image_bgr
        ir_left, ir_right = frame.ir_left, frame.ir_right
        has_ir = ir_left is not None and ir_right is not None
        ir_shape = ir_left.shape if has_ir else None
        timing = (None if self._frame_timing is None
                  else self._frame_timing(frame.metadata))

        slot = self._pool.acquire(color.shape, ir_shape)
        if slot is None and self._keep_every_frame:
            slot = self._take_back_outbox_slot(color.shape, ir_shape)
        if slot is None:
            self.n_pool_exhausted += 1
            return False

        # Outside the lock: numpy copies without the GIL, the worker keeps
        # running meanwhile.
        np.copyto(slot.color, color)
        if has_ir:
            np.copyto(slot.ir_left, ir_left)
            np.copyto(slot.ir_right, ir_right)
        slot.has_ir = has_ir
        slot.frame_index = int(frame.frame_index)
        slot.t_frame = float(t_frame)
        slot.meta = meta
        slot.timing = timing

        with self._cond:
            if self._stopping or self._ended:
                replaced, slot = slot, None
            else:
                replaced, self._pending = self._pending, slot
                self.n_submitted += 1
                if replaced is not None:
                    self._skipped_since += 1
                    self.n_skipped += 1
                self._cond.notify()
        self._pool.release(replaced)
        return slot is not None

    def request_reset(self) -> int:
        """Reset the tracker before its next frame; returns the new epoch."""
        with self._cond:
            self._reset_epoch += 1
            return self._reset_epoch

    def take_outputs(self) -> list[TrackOutput]:
        with self._cond:
            if not self._outbox:
                return []
            outs = list(self._outbox)
            self._outbox.clear()
        return outs

    def release(self, slot: FrameSlot | None) -> None:
        """Give a slot back (slots of another worker are ignored)."""
        self._pool.release(slot)

    def stop(self, timeout_s: float = 2.0) -> bool:
        """Stop after the current frame and join. False: the thread did not
        end within timeout_s (engine hung); it is abandoned, and its tracker
        must not be used again. Untaken outputs stay for take_outputs()."""
        with self._cond:
            self._stopping = True
            pending, self._pending = self._pending, None
            self._cond.notify_all()
        self._pool.release(pending)

        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout_s)
        return not thread.is_alive()

    def _take_back_outbox_slot(self, color_shape: tuple, ir_shape: tuple | None) -> FrameSlot | None:
        with self._cond:
            for out in self._outbox:
                if out.slot is not None and out.slot.fits(color_shape, ir_shape):
                    slot, out.slot = out.slot, None
                    out.frame_dropped = True
                    self.n_frame_dropped += 1
                    return slot
        return None

    # ---------------- worker thread ----------------

    def _run(self) -> None:
        slot = None                # the frame in hand, until an output owns it
        try:
            while True:
                with self._cond:
                    while self._pending is None and not self._stopping:
                        self._cond.wait()
                    if self._stopping:
                        return
                    slot, self._pending = self._pending, None
                    skipped, self._skipped_since = self._skipped_since, 0
                    reset = self._reset_epoch > self._epoch
                    self._epoch = self._reset_epoch
                    epoch = self._epoch

                t_start = time.perf_counter()
                points = points_kind = None
                points_marked = False
                try:
                    if reset:
                        self._tracker.reset()
                        if self._tool_points is not None:
                            self._tool_points.reset()
                    timing = {}
                    if slot.timing is not None:
                        timing = dict(zip(("frame_time_ms", "ir_time_offset_ms"),
                                          slot.timing))
                    result = self._tracker.process_frame(
                        slot.color,
                        run_detection=True,
                        ir_left=slot.ir_left if slot.has_ir else None,
                        ir_right=slot.ir_right if slot.has_ir else None,
                        t_frame=slot.t_frame,
                        **timing,
                    )
                    if self._tool_points is not None:
                        points = self._tool_points.update(result, slot.t_frame)
                        points_kind = self._tool_points.kind
                        points_marked = self._tool_points.marked
                except Exception as exc:
                    out = TrackOutput(
                        slot, epoch, t_start=t_start, t_done=time.perf_counter(),
                        skipped=skipped, reset=reset, error=exc,
                        traceback_text=traceback.format_exc(),
                    )
                    slot = None
                    self._publish(out)
                    return
                out = TrackOutput(
                    slot, epoch, result=result, t_start=t_start,
                    t_done=time.perf_counter(), skipped=skipped, reset=reset,
                    points=points, points_kind=points_kind,
                    points_marked=points_marked,
                )
                slot = None
                self._publish(out)
        except BaseException as exc:
            # Outside the tracker call (building or publishing an output,
            # e.g. a MemoryError): no output carries it. The thread ends
            # with failure set, which the GUI thread finds via ended.
            self.failure = exc
            self.failure_traceback = traceback.format_exc()
        finally:
            with self._cond:
                self._ended = True
                pending, self._pending = self._pending, None
            self._pool.release(pending)
            self._pool.release(slot)

    def _publish(self, out: TrackOutput) -> None:
        released = []
        with self._cond:
            if not self._keep_every_frame:
                # only the newest result's image can still be shown
                for old in self._outbox:
                    if old.slot is not None:
                        released.append(old.slot)
                        old.slot = None
            self._outbox.append(out)
            while len(self._outbox) > self.OUTBOX_MAX:
                dropped = self._outbox.popleft()
                released.append(dropped.slot)
                dropped.slot = None
                self.n_outbox_dropped += 1
            self.n_processed += 1
        for slot in released:
            self._pool.release(slot)
        if self._notify is not None:
            try:
                self._notify()
            except Exception:
                pass


# ============================================================
# Tracking state (GUI thread)
# ============================================================

STATUS_SEARCHING = "searching…"
STATUS_SEARCH_HINT = "searching… (no marker found yet)"
STATUS_LOST = "lost — searching…"
STATUS_TRACKING = "tracking"
STATUS_SETTLING = "tracking (settling…)"

# timings_ms key the HydraTracker wrapper sets from the engine's
# current_pose_accepted (1.0 / 0.0)
_ACCEPTED_KEY = "cpp_tracker_engine_current_pose_accepted_count"


@dataclass
class SessionStep:
    """What one output changed. current=False: output of an older epoch,
    ignored (everything else stays at its default)."""

    current: bool = False
    fresh: bool = False
    status: str | None = None      # new status text, None = unchanged
    locked_now: bool = False       # this output (re)locked
    relock: bool = False           # ... after an engine reset
    converged: bool = False
    reset: bool = False            # this output ended the episode -> new epoch
    recovered: bool = False        # pose back after a dropout, before the reset
                                   # (status back to the tracking text; a lock or
                                   # a convergence change sets status itself)
    hint: bool = False             # the search hint was shown with this output
    reentry: bool = False          # the engine treats this frame as a re-entry


class TrackingSession:
    """
    The Live page's tracking state machine, fed with every TrackOutput in
    processing order.

    Searching until the first fresh pose, then tracking. A result without a
    fresh pose while tracking clears the overlay and the status says
    STATUS_LOST at once; a fresh pose before the reset brings the tracking
    text back. After lost_reset_s of frame time without a fresh pose the
    episode ends: the engine is reset (new epoch, request_reset) and the
    search goes on - no stop, the user keeps the Start/Stop button. Without
    a lock for search_hint_s the status adds a hint once per search. All
    windows run on t_frame (the poll time of the frame), not on the number
    of processed frames or on the arrival time of the results.

    has_fresh_pose: tracker_log.has_fresh_pose; request_reset: the worker's;
    fail_log: the page's _TrackFailureLog or None (begin_episode, note,
    note_appearance, flush). Starts at construction; start() begins anew.
    """

    def __init__(self, has_fresh_pose: Callable, request_reset: Callable[[], int], *,
                 lost_reset_s: float, search_hint_s: float, fail_log=None) -> None:
        self._has_fresh_pose = has_fresh_pose
        self._request_reset = request_reset
        self.lost_reset_s = float(lost_reset_s)
        self.search_hint_s = float(search_hint_s)
        self._fail_log = fail_log
        self.start(time.perf_counter())

    def start(self, t_now: float) -> None:
        self.epoch = 0
        self.locked = False
        self.converged = False
        self.frames = 0            # current-epoch outputs since start
        self.stale = 0             # consecutive outputs without a fresh pose while locked
        self.episode = 0           # lock episodes so far
        self.episode_results = 0   # outputs of the current (or last) episode
        self.resets = 0
        self.n_old_epoch = 0       # outputs ignored (older epoch)
        self.t_start = float(t_now)
        self.t_search = float(t_now)   # search start (Start, or the last reset)
        self.t_fresh: float | None = None
        self.t_reset: float | None = None
        self.last_frame_index = -1
        self._hint_shown = False
        self._prev_accepted = False

    def update(self, out: TrackOutput) -> SessionStep:
        step = SessionStep()
        if out.epoch != self.epoch:
            self.n_old_epoch += 1
            return step
        step.current = True
        result = out.result
        self.frames += 1
        self.last_frame_index = out.frame_index

        # The engine's re-entry rule: the previous processed frame has no
        # accepted pose (the first frame of an epoch is one).
        step.reentry = not self._prev_accepted
        timings = getattr(result, "timings_ms", None) or {}
        self._prev_accepted = bool(timings.get(_ACCEPTED_KEY, 0.0))
        log = self._fail_log
        if log is not None:
            log.note_appearance(result, out.frame_index, step.reentry)

        if self._has_fresh_pose(result):
            step.fresh = True
            was_lost = self.stale > 0
            self.stale = 0
            self.t_fresh = out.t_frame
            converged = bool(getattr(result, "pose_converged", False))
            if not self.locked:
                self.locked = True
                self.episode += 1
                self.episode_results = 0
                step.locked_now = True
                step.relock = self.resets > 0
                if log is not None:
                    log.begin_episode(self.episode, out.frame_index)
            self.episode_results += 1
            if step.locked_now or converged != self.converged:
                self.converged = converged
                step.status = STATUS_TRACKING if converged else STATUS_SETTLING
            elif was_lost:
                step.recovered = True
                step.status = STATUS_TRACKING if converged else STATUS_SETTLING
            step.converged = self.converged
            return step

        if not self.locked:
            if not self._hint_shown and out.t_frame - self.t_search >= self.search_hint_s:
                self._hint_shown = True
                step.hint = True
                step.status = STATUS_SEARCH_HINT
            return step

        # Dropout after a lock: aggregate the cause; search failures are
        # deliberately not logged. The status says so at once, the drill is
        # no longer drawn.
        self.stale += 1
        self.episode_results += 1
        if self.stale == 1:
            step.status = STATUS_LOST
        if log is not None:
            log.note(result, out.frame_index, self.stale)
        if out.t_frame - self.t_fresh >= self.lost_reset_s:
            if log is not None:
                log.flush("lost (reset)", out.frame_index, self.episode_results)
            self.epoch = self._request_reset()
            self.resets += 1
            self.locked = False
            self.converged = False
            self.stale = 0
            self.t_search = out.t_frame
            self.t_reset = out.t_frame
            self._hint_shown = False
            self._prev_accepted = False
            step.reset = True
        return step

    def finish(self, reason: str) -> None:
        """End of the run (Stop, page change, plane, error): close the
        failure log's open episode."""
        if self._fail_log is not None:
            self._fail_log.flush(
                reason, self.last_frame_index,
                self.episode_results if self.locked else 0, final=True,
            )
