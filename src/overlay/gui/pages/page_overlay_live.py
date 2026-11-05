from __future__ import annotations

import json
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, QRect, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from overlay.calib.calib_xray_to_pointer import extract_depth
from overlay.gui.state import SessionState
from overlay.tools.display_exposure import AutoExposureLook
from overlay.tools.homography import estimate_plane_induced_homography
from overlay.tools.live_tracking import STATUS_SEARCHING, TrackingSession, TrackingWorker
from overlay.tools.pointer_depth import PointerDepthEstimator, depth_capture_arrays
from overlay.tools.tracking_overlay import TrackingOverlay
from overlay.tools.warp import prepare_live_overlay
from overlay.tools.xray_io import read_xray_gray_u8
from overlay.tools.xray_view import XrayField, XrayView

# The camera is owned by the tracking hardware boundary so the tuned
# RealSense options (auto_exposure_priority=0 for guaranteed fps, emitter
# off for the IR pair, ...) live in exactly one place. The tracker itself
# (HydraTracker, C++ engine) is imported lazily in _start_tracking so the
# GUI starts and the overlay runs even when the tracking build is absent;
# overlay.tools.live_tracking (its thread) imports neither Qt nor the engine.
from tracking.hydramarker.camera_setup import (
    RealSenseCameraSource,
    format_realsense_color,
    realsense_color_readback,
    realsense_color_summary,
)
from tracking.hydramarker.color_exposure import (
    MODE_MARKER,
    MarkerExposureControl,
)
from tracking.hydramarker.config import CameraConfig


# ============================================================
# Frame profiling
# ============================================================

PROFILE_LIVE = True
PROFILE_EVERY = 30          # frames between summary lines

# The profile log is only written in Evaluation mode, into the session
# folder (live_profile.log); in Normal mode nothing is written.
PROFILE_LOG_NAME = "live_profile.log"

# Presentation snapshots (Save snapshot button): one timestamped folder per
# press with pixel-aligned PNG layers of the current live frame. Normal
# mode: overlay/data/snapshots/; Evaluation mode: <label>_live/ in the
# session folder.
SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "data" / "snapshots"


# ============================================================
# Tracking failure log
# ============================================================

# Aggregated dropout causes per lock episode (lock -> engine reset after a
# loss, or Stop), written as one JSONL record when the episode ends. Costs
# next to nothing in the live loop: fresh results only look up the
# appearance-check keys, stale results only update an in-memory dict (and
# take a one-time detail snapshot per distinct cause); the file write
# happens only when an episode ends. Everything runs in the GUI thread, on
# the drained tracking results (TrackingSession calls it).
# On/off switch: TrackerConfig.failure_log_enabled (set in
# run_tracker.make_live_tracker_config, like every other tracking setting);
# the file itself only exists in Evaluation mode (session folder).
TRACK_FAILURE_LOG_NAME = "tracking_failure_log.jsonl"


class _TrackFailureLog:
    """Dropout causes of one tracking run, one record per lock episode,
    deduplicated by (stage, reason).

    note() is only called for stale results AFTER a lock (search failures
    are not logged). Repeats of the same cause bump a counter instead of
    producing new entries; the full diagnostic snapshot is taken once, at
    the first occurrence of each cause. Frame numbers (*_frame_idx) are the
    camera's frame_index of the processed frame.

    note_appearance() counts every appearance check that rejected a pose or
    could not decide, on fresh results and while searching too - the safety
    of every (re)lock rests on that check. A record carries the counts since
    the previous record (the search before its lock included), so the
    records of a run add up; what is left at the end of the run is written
    as "appearance_checks".

    log_version 2. Version 1 (no log_version field) wrote one record per
    run: frames_total = processed frames of the run,
    causes[].first_frame/last_frame = processed-frame counter.
    """

    LOG_VERSION = 2

    def __init__(self, path: str):
        self._path: Path | None = Path(path).resolve()
        self._tracker_log = None            # tracking.hydramarker.tracker_log
        self._causes: dict[tuple[str, str], dict] = {}
        self._t0 = 0.0
        self._episode = 0
        self._episode_frame0 = -1
        self._t_episode = 0.0
        self._appearance: dict = {}

    @staticmethod
    def _no_appearance() -> dict:
        return {"reentry": {}, "tracking": {}, "undecided": {}, "first": {}}

    def start(self, tracker_log_module) -> None:
        """Arm for a new tracking run (called from _start_tracking)."""
        self._tracker_log = tracker_log_module
        self._causes = {}
        self._t0 = time.perf_counter()
        self._episode = 0
        self._episode_frame0 = -1
        self._appearance = self._no_appearance()

    def begin_episode(self, episode: int, frame_idx: int) -> None:
        """A (re)lock starts a new episode."""
        self._episode = int(episode)
        self._episode_frame0 = int(frame_idx)
        self._t_episode = time.perf_counter()
        self._causes = {}

    def note(self, result, frame_idx: int, stale: int) -> None:
        """Record one stale result. Dict-only work; never raises."""
        try:
            tl = self._tracker_log
            if tl is None:
                return
            if getattr(result, "success", False):
                # Held pose: success without a fresh pose is its own cause,
                # classify_failure would report it as "OK"; the key names
                # what the engine held through.
                key = tl.classify_hold(result)
            else:
                key = tl.classify_failure(result)

            entry = self._causes.get(key)
            if entry is not None:
                entry["count"] += 1
                entry["last_frame_idx"] = frame_idx
                entry["last_t_s"] = round(time.perf_counter() - self._t0, 3)
                return

            # First occurrence: snapshot everything needed to diagnose it.
            self._causes[key] = {
                "stage": key[0],
                "reason": key[1],
                "count": 1,
                "first_frame_idx": frame_idx,
                "last_frame_idx": frame_idx,
                "first_t_s": round(time.perf_counter() - self._t0, 3),
                "last_t_s": round(time.perf_counter() - self._t0, 3),
                "first": {
                    "message": str(getattr(result, "message", "") or ""),
                    "mode": getattr(getattr(result, "mode", None), "value", None),
                    "pose_source": tl.pose_source_value(result),
                    "stale_frames": stale,
                    "num_points": getattr(result, "num_points", None),
                    "num_inliers": getattr(result, "num_inliers", None),
                    "mean_reproj_px": getattr(
                        result, "mean_reprojection_error_px", None
                    ),
                    "max_reproj_px": getattr(
                        result, "max_reprojection_error_px", None
                    ),
                    "n_corners": len(getattr(result, "corners", ()) or ()),
                    "n_detection_corners": len(
                        getattr(result, "detection_corners", ()) or ()
                    ),
                    "debug_counters": dict(
                        getattr(result, "debug_counters", {}) or {}
                    ),
                },
            }
        except Exception:
            pass

    def note_appearance(self, result, frame_idx: int, reentry: bool) -> None:
        """Count the appearance checks of one result that rejected a pose or
        could not decide (every result of the current epoch comes here).
        Dict-only work; never raises."""
        try:
            tl = self._tracker_log
            if tl is None:
                return
            rejects = tl.appearance_rejects(result)
            undecided = tl.appearance_undecided(result)
            if not rejects and not undecided:
                return
            app = self._appearance
            group = app["reentry" if reentry else "tracking"]
            for kind in rejects:
                group[kind] = group.get(kind, 0) + 1
            for kind in undecided:
                app["undecided"][kind] = app["undecided"].get(kind, 0) + 1
            for kind in rejects:
                if kind not in app["first"]:
                    app["first"][kind] = {
                        "frame_idx": frame_idx,
                        "reentry": bool(reentry),
                        "message": str(getattr(result, "message", "") or ""),
                        **tl.appearance_values(result, kind),
                    }
        except Exception:
            pass

    def flush(self, stop_reason: str, frame_idx: int, n_results: int,
              final: bool = False) -> None:
        """End the current episode (n_results: its processed results, the
        locking one included): write one JSONL record if any dropout was
        recorded. final (end of the run): without such a record, the
        appearance counts not written yet go out on their own. Never
        raises."""
        causes, self._causes = self._causes, {}
        episode, self._episode = self._episode, 0
        if self._path is None:
            return
        appearance = {k: v for k, v in self._appearance.items() if v}
        if causes and episode:
            record = {
                "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "event": "tracking_dropout",
                "log_version": self.LOG_VERSION,
                "stop_reason": stop_reason,
                "episode": episode,
                "frames_episode": int(n_results),
                "frames_stale": sum(c["count"] for c in causes.values()),
                "frame_idx_span": [self._episode_frame0, frame_idx],
                "episode_duration_s": round(time.perf_counter() - self._t_episode, 3),
                "run_duration_s": round(time.perf_counter() - self._t0, 3),
                "causes": sorted(
                    causes.values(), key=lambda c: c["first_frame_idx"]
                ),
                "appearance": appearance,
            }
        elif final and appearance:
            record = {
                "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "event": "appearance_checks",
                "log_version": self.LOG_VERSION,
                "stop_reason": stop_reason,
                "run_duration_s": round(time.perf_counter() - self._t0, 3),
                "appearance": appearance,
            }
        else:
            return
        # in this record now: the next one counts from zero
        self._appearance = self._no_appearance()
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            self._path = None


class _FrameProfile:
    """
    Per-stage timing for the live loop.

    Prints one summary line every PROFILE_EVERY shown frames with mean and
    max per stage in ms. 'interval' is the wall time between two shown
    frames - if that is well above 1000/FPS while the stages are cheap, the
    timer itself is being starved rather than the processing being slow.

    Written by the GUI thread only. The tracking thread's numbers arrive with
    its results: 'track' (tracker call in the worker), 'wait' (poll -> start
    of the call), 'lag' (poll -> result), 'pickup' (result -> drained by the
    GUI); 'age' is poll -> shown, per shown frame. 'camera lost' counts the
    colour frames the page never got (gaps in the RealSense frame numbers:
    the SDK keeps only the newest frame for poll(), or the camera dropped
    them).

    'xview' is the X-ray view's drawing per shown frame, 'xview_up' its
    geometry per fresh pose (TrackingOverlay.update, in the drain); both with
    three decimals, because they are below the 0.1 ms of the other stages.

    frame_sink (Evaluation mode, _EvalFrameLog.add_display): also gets one
    row per shown frame, the stages added between frame_start and frame_end
    as "<stage>_ms" (summed if a stage comes twice), plus interval_ms,
    total_ms, t_s (frame_start, perf_counter), t_wall_s (frame_end,
    time.time()) and what frame_end is given.
    """

    def __init__(self, fps: int, log_path: Path | None):
        self.budget_ms = 1000.0 / float(fps)
        self.log_path = None if log_path is None else Path(log_path).resolve()
        self._camera_number: int | None = None   # last polled frame number
        self._exposure = ""                       # note_exposure()
        self.frame_sink = None                    # callable(row) or None
        self._cur: dict | None = None             # row of the frame in progress
        self.reset()

    def emit(self, line: str) -> None:
        """Write one line to the log file. Never raises - the live loop wins."""
        if self.log_path is None:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            self.log_path = None

    def reset(self) -> None:
        self._t: dict[str, list[float]] = defaultdict(list)
        self._c: dict[str, int] = defaultdict(int)
        self._n_frames = 0
        self._n_polls = 0
        self._n_empty = 0
        self._t_last_start: float | None = None
        self._t_window0 = time.perf_counter()

    def note_poll(self, empty: bool) -> None:
        self._n_polls += 1
        if empty:
            self._n_empty += 1

    def count(self, name: str, n: int = 1) -> None:
        """Event counters of the window (tracking thread hand-off, display)."""
        self._c[name] += n

    def note_exposure(self, text: str) -> None:
        """The marker exposure control's state for the summary lines ("":
        the control is off)."""
        self._exposure = text

    def note_camera_frame(self, number: int | None) -> None:
        """RealSense frame number of a polled colour frame (None: unknown)."""
        if number is None:
            return
        last, self._camera_number = self._camera_number, number
        if last is not None and number > last + 1:
            self._c["camera_lost"] += number - last - 1

    def frame_start(self) -> float:
        now = time.perf_counter()
        interval = float("nan")
        if self._t_last_start is not None:
            interval = (now - self._t_last_start) * 1000.0
            self._t[" interval"].append(interval)
        self._t_last_start = now
        if self.frame_sink is not None:
            self._cur = {"t_s": now, "interval_ms": interval}
        return now

    def add(self, stage: str, seconds: float) -> None:
        ms = seconds * 1000.0
        self._t[stage].append(ms)
        cur = self._cur
        if cur is not None:
            key = stage.strip().rstrip("*") + "_ms"
            cur[key] = cur.get(key, 0.0) + ms

    def frame_end(self, t0: float, extra: dict | None = None) -> None:
        total = (time.perf_counter() - t0) * 1000.0
        self._t["total"].append(total)
        self._n_frames += 1
        cur, self._cur = self._cur, None
        if cur is not None and self.frame_sink is not None:
            cur["total_ms"] = total
            cur["t_wall_s"] = time.time()
            if extra:
                cur.update(extra)
            self.frame_sink(cur)
        if self._n_frames >= PROFILE_EVERY:
            self._report()
            self.reset()

    def _report(self) -> None:
        elapsed = time.perf_counter() - self._t_window0
        fps = self._n_frames / elapsed if elapsed > 0 else 0.0

        parts = []
        for stage in sorted(self._t):
            v = self._t[stage]
            if v:
                d = 3 if stage.startswith("xview") else 1
                parts.append(f"{stage.strip()} {np.mean(v):6.{d}f}/{np.max(v):6.{d}f}")

        empty_pct = 100.0 * self._n_empty / max(1, self._n_polls)

        # While tracking: processed frames per second, share of submitted
        # frames the worker skipped (a newer one replaced them), share of
        # shown frames that carried a pose.
        c = self._c
        track = ""
        if c["submitted"]:
            proc = c["processed"] / elapsed if elapsed > 0 else 0.0
            skipped_pct = 100.0 * c["skipped"] / c["submitted"]
            shown = c["shown_pose"] + c["shown_live"]
            pose_pct = 100.0 * c["shown_pose"] / max(1, shown)
            track = (
                f"proc {proc:4.1f}/s skipped {skipped_pct:3.0f}%  "
                f"pose {pose_pct:3.0f}%  "
            )
        # Pose frames that could not be shown any more (older than the
        # frame on screen - one at every lock - or than
        # TRACK_POSE_MAX_AGE_S), and what must not happen: no free frame
        # slot (also in a window where nothing could be submitted), results
        # lost in the outbox.
        for name in ("pose_dropped", "pool_exhausted", "outbox_dropped"):
            if c[name]:
                track += f"{name} {c[name]}  "

        camera = ""
        if self._camera_number is not None:
            camera = f"camera lost {c['camera_lost']}  "
        # marker exposure control: setting, marker white level, steps
        # (exposure_steps: written in this window)
        exposure = ""
        if self._exposure:
            exposure = f"exposure {self._exposure} steps {c['exposure_steps']}  "

        self.emit(
            f"[live] {time.strftime('%H:%M:%S')}  {fps:5.1f} fps "
            f"(budget {self.budget_ms:.1f} ms)  polls empty {empty_pct:4.0f}%  "
            + camera
            + track
            + exposure
            + "|  mean/max ms: " + "  ".join(parts)
        )


# ============================================================
# Per-frame log (Evaluation mode)
# ============================================================

# The profile lines are means per window; for the evaluation every frame is
# kept as well, from the same session: one row per processed tracking
# result (worker timing, the engine's own stage timings and counters, the
# path it took, pose and drawn tip) and one row per shown frame (the display
# stages of _FrameProfile, the frame's camera time). The rows stay in memory
# and go to the session folder as .npz when a tracking run ends (every
# _stop_tracking: Stop, reference plane, error, page change) and when the
# page is left: <label>_tracking.npz (that run's results) and
# <label>_display.npz (the frames shown since the last write). Arrays are
# built and written in a thread of their own; the live loop only appends.
# Version 2: the tracking rows carry the X-ray view (d_x_mm, tip_x_mm,
# tip_uv_x, back_uv_x, tip_uv_on_plane, off_plane_mm, inclination_deg,
# tip_in_field, xview_update_ms), the display rows xview_ms.
FRAME_LOG_VERSION = 2

_FRAME_LOG_CLOCKS = (
    "t_s, t_frame, t_start, t_done, t_drain: time.perf_counter() [s] "
    "(t_frame = poll of the frame); t_host_s: time.time() at the poll [s]; "
    "t_wall_s: time.time() at the end of the display [s]; color_ts_ms: the "
    "RealSense colour frame timestamp (frame.get_timestamp()) [ms]"
)


def _keys_values(d) -> tuple[tuple, np.ndarray | None]:
    """A dict of numbers as (keys, float64 values), for the timing and
    counter dicts of a result; non-numbers become NaN."""
    if not d:
        return (), None
    keys = tuple(d.keys())
    try:
        values = np.fromiter(d.values(), dtype=np.float64, count=len(keys))
    except (TypeError, ValueError):
        values = np.full(len(keys), np.nan)
        for i, v in enumerate(d.values()):
            try:
                values[i] = float(v)
            except (TypeError, ValueError):
                pass
    return keys, values


def _frame_rows_to_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    """Column arrays of per-frame rows: bool / str / number (NaN where a
    row lacks it) / vector (n, m); "_name" entries are (keys, values) pairs
    and become name_keys + name_values (union of keys, NaN = absent)."""
    out: dict[str, np.ndarray] = {}
    n = len(rows)
    names: dict[str, None] = {}
    for r in rows:
        for k in r:
            names.setdefault(k, None)
    for k in names:
        if k.startswith("_"):
            groups: dict[tuple, list[int]] = {}
            for i, r in enumerate(rows):
                kv = r.get(k)
                if kv is not None and kv[0]:
                    groups.setdefault(kv[0], []).append(i)
            keys = sorted(set().union(*groups)) if groups else []
            col = {key: j for j, key in enumerate(keys)}
            mat = np.full((n, len(keys)), np.nan)
            for ks, idx in groups.items():
                cols = np.array([col[key] for key in ks], dtype=np.int64)
                mat[np.ix_(idx, cols)] = np.vstack([rows[i][k][1] for i in idx])
            out[f"{k[1:]}_keys"] = np.array(keys, dtype=str)
            out[f"{k[1:]}_values"] = mat
            continue
        sample = next((r[k] for r in rows if r.get(k) is not None), None)
        if sample is None:
            continue
        if isinstance(sample, (bool, np.bool_)):
            out[k] = np.array([bool(r.get(k, False)) for r in rows], dtype=bool)
        elif isinstance(sample, str):
            out[k] = np.array([r.get(k) or "" for r in rows], dtype=str)
        elif np.ndim(sample) > 0:
            m = int(np.asarray(sample).size)
            a = np.full((n, m), np.nan)
            for i, r in enumerate(rows):
                v = r.get(k)
                if v is not None:
                    v = np.asarray(v, dtype=np.float64).reshape(-1)
                    if v.size == m:
                        a[i] = v
            out[k] = a
        else:
            out[k] = np.array(
                [np.nan if r.get(k) is None else float(r[k]) for r in rows],
                dtype=np.float64,
            )
    return out


class _EvalFrameLog:
    """Per-frame rows of the live loop for the evaluation (see above).
    Used by the GUI thread only; a writer thread gets lists the page no
    longer touches."""

    # without a tracking stop, the display rows are written every ~5 min
    DISPLAY_FLUSH_FRAMES = 9000

    def __init__(self, ev, fps: int):
        self._ev = ev
        self._fps = int(fps)
        self._track: list[dict] = []
        self._display: list[dict] = []
        # frame_index -> (colour timestamp ms, time.time() of the poll)
        self._frames: dict[int, tuple[float, float]] = {}
        self._run = 0

    def note_frame(self, frame) -> None:
        """Camera time of a polled frame, for the rows of its result and of
        its display (the last ~2 s of frames are kept)."""
        index = int(frame.frame_index)
        ts = (frame.metadata or {}).get("color_time_ms")
        self._frames[index] = (
            float("nan") if ts is None else float(ts), float(frame.timestamp_s)
        )
        if len(self._frames) > 4 * self._fps:
            self._frames = {
                i: v for i, v in self._frames.items() if i > index - 2 * self._fps
            }

    def add_display(self, row: dict) -> None:
        """_FrameProfile.frame_sink: one shown frame."""
        ts, t_host = self._frames.get(int(row.get("frame_idx", -1)), (np.nan, np.nan))
        row["color_ts_ms"] = ts
        row["t_host_s"] = t_host
        self._display.append(row)
        if len(self._display) >= self.DISPLAY_FLUSH_FRAMES:
            self.flush("display rows", tracking=False)

    def add_track(self, out, step, t_drain: float, overlay, session, tl) -> None:
        """One processed tracking result, including those of an older epoch
        the session ignored (current False). overlay: the TrackingOverlay
        after this result (None: nothing drawn from it)."""
        res = out.result
        ts, t_host = self._frames.get(int(out.frame_index), (np.nan, np.nan))
        row = {
            "frame_idx": int(out.frame_index),
            "epoch": int(out.epoch),
            "t_frame": out.t_frame,
            "t_start": out.t_start,
            "t_done": out.t_done,
            "t_drain": t_drain,
            "track_ms": (out.t_done - out.t_start) * 1000.0,
            "lag_ms": (out.t_done - out.t_frame) * 1000.0,
            "color_ts_ms": ts,
            "t_host_s": t_host,
            "skipped": int(out.skipped),
            "reset_before": bool(out.reset),
            "frame_dropped": bool(out.frame_dropped),
            "has_ir": bool(out.has_ir),
            "current": bool(step.current),
            "fresh": bool(step.fresh),
            "locked_now": bool(step.locked_now),
            "relock": bool(step.relock),
            "reentry": bool(step.reentry),
            "lost_reset": bool(step.reset),
            "locked": bool(getattr(session, "locked", False)),
            "points_kind": out.points_kind or "",
            "points_marked": bool(out.points_marked),
        }
        if res is not None:
            vis = res.timings_ms.get("visible_cell_fraction")
            row.update({
                "success": bool(res.success),
                "mode": str(getattr(res.mode, "value", res.mode)),
                "pose_source": str(tl.pose_source_value(res)),
                "num_points": int(res.num_points),
                "num_inliers": int(res.num_inliers),
                "mean_reproj_px": float(res.mean_reprojection_error_px),
                "visible_frac": None if vis is None else float(vis),
                "rvec": None if res.rvec is None else np.asarray(res.rvec, dtype=np.float64).reshape(-1),
                "tvec": None if res.tvec is None else np.asarray(res.tvec, dtype=np.float64).reshape(-1),
                "_timings": _keys_values(res.timings_ms),
                "_debug": _keys_values(res.debug_counters),
            })
        if out.points is not None:
            pts = np.asarray(out.points, dtype=np.float64).reshape(-1, 3)
            row["tip_cam_mm"] = pts[0]
            row["back_cam_mm"] = pts[1]
        drawn = None if overlay is None else overlay.snapshot_record()
        if drawn is not None:
            row["tip_uv"] = np.asarray(drawn["tip_uv"], dtype=np.float64)
            row["back_uv"] = np.asarray(drawn["back_uv"], dtype=np.float64)
            xv = drawn["xray_view"]
            if xv is not None:
                row["d_x_mm"] = xv["d_x_mm"]
                row["tip_x_mm"] = np.asarray(xv["tip_x_mm"], dtype=np.float64)
                row["tip_uv_x"] = np.asarray(xv["tip_uv_x"], dtype=np.float64)
                row["back_uv_x"] = np.asarray(xv["back_uv_x"], dtype=np.float64)
                row["tip_uv_on_plane"] = np.asarray(xv["tip_uv_on_plane"], dtype=np.float64)
                row["off_plane_mm"] = xv["off_plane_mm"]
                row["inclination_deg"] = xv["inclination_deg"]
                row["tip_in_field"] = xv["tip_in_field"]
                row["xview_update_ms"] = overlay.xray_update_s * 1000.0
        self._track.append(row)

    def flush(self, reason: str, tracking: bool) -> None:
        """Write the rows collected so far: the tracking run that just ended
        (tracking True) and the shown frames. Files are reserved here (the
        label in force now), written in a thread."""
        jobs = []
        if tracking and self._track:
            self._run += 1
            rows, self._track = self._track, []
            jobs.append(self._job("tracking", rows, reason, run=self._run))
        if self._display:
            rows, self._display = self._display, []
            jobs.append(self._job("display", rows, reason))
        jobs = [j for j in jobs if j is not None]
        if jobs:
            threading.Thread(
                target=self._write, args=(jobs,), name="frame-log-writer"
            ).start()

    def _job(self, kind: str, rows: list[dict], reason: str, run: int = 0):
        try:
            path = self._ev.path_for(kind, ".npz")
            path.touch()          # reserved: the next flush gets _2, _3, ...
            summary = {"frames": len(rows), "reason": reason}
            if run:
                summary["run"] = run
            self._ev.log("overlay_live", path.name, summary)
        except Exception as e:
            print(f"[WARN] Frame log: could not reserve the {kind} file: {e}")
            return None
        meta = {
            "log_version": FRAME_LOG_VERSION, "kind": kind, "reason": reason,
            "run": run, "clocks": _FRAME_LOG_CLOCKS,
        }
        return path, rows, meta

    @staticmethod
    def _write(jobs) -> None:
        for path, rows, meta in jobs:
            try:
                arrays = _frame_rows_to_arrays(rows)
                arrays["meta_json"] = np.array(json.dumps(meta))
                np.savez_compressed(path, **arrays)
            except Exception as e:
                print(f"[WARN] Frame log: writing {path.name} failed: {e}")


# ============================================================
# Helpers
# ============================================================

def _draw_point_inplace(
    img: np.ndarray,
    uv: np.ndarray,
    *,
    color=(0, 0, 255, 255),
    radius=8,
    cross_size=20,
    thickness=2,
) -> None:
    """
    Draw the tip directly into the frame buffer.

    The previous version copied the whole 1920x1080 frame first, which cost
    1.9 ms for a marker of a few hundred pixels.
    """
    if uv is None:
        return

    uv = np.asarray(uv, dtype=np.float64).reshape(2)
    if not np.isfinite(uv).all():
        return

    u, v = np.round(uv).astype(int)

    cv2.circle(img, (u, v), radius, color, thickness, cv2.LINE_AA)
    cv2.drawMarker(
        img,
        (u, v),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=cross_size,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )


def _bgra_to_qpixmap(img_bgra: np.ndarray) -> QPixmap:
    """
    Wrap a BGRA buffer without converting or copying anything.

    Format_RGB32 IS Qt's native 32-bit layout on little-endian - BGRA with the
    alpha ignored - so QPixmap.fromImage degrades to a reference and costs
    ~0.01 ms. The neighbouring choices are traps: Format_BGR888 forces a
    per-pixel expansion (1.75 ms) and Format_ARGB32 forces a premultiply
    (1.60 ms). Both are two hundred times more expensive for the same picture.

    The caller must keep img_bgra alive while the pixmap is in use.
    """
    if img_bgra is None:
        return QPixmap()

    h, w = img_bgra.shape[:2]
    qimg = QImage(img_bgra.data, w, h, 4 * w, QImage.Format_RGB32)
    return QPixmap.fromImage(qimg)


# ============================================================
# Professional sidebar — scaled 60%
# ============================================================

class _OverlayControlRail(QFrame):
    """
    Right-side live overlay controls.

    - Starts collapsed.
    - Can be opened/closed repeatedly.
    - Professional dark translucent sidebar.
    - Overall size scaled to 60%.
    """

    SCALE = 0.60

    HANDLE_W = int(round(34 * SCALE))
    PANEL_W = int(round(380 * SCALE))
    PANEL_H = int(round(486 * SCALE))
    # Extra panel height while the tool-tracking card is visible.
    TRACK_EXTRA_H = int(round(186 * SCALE))
    # Extra panel height while the reference-plane card is visible.
    PLANE_EXTRA_H = int(round(150 * SCALE))
    # Without an X-ray (set_tracking_only): the alpha and plane-depth cards
    # and the pointer-tip and X-ray-view boxes go, the Back card comes.
    XRAY_H = int(round(284 * SCALE))
    BACK_H = int(round(100 * SCALE))
    GAP = int(round(10 * SCALE))

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._expanded = False

        self.setObjectName("overlayControlRail")
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_StyledBackground, True)

        # ---------------- Handle ----------------
        self.handle = QToolButton(self)
        self.handle.setObjectName("overlayRailHandle")
        self.handle.setText("‹")
        self.handle.setCursor(Qt.PointingHandCursor)
        self.handle.setToolTip("Show overlay controls")
        self.handle.clicked.connect(self._toggle)

        # ---------------- Panel ----------------
        self.panel = QFrame(self)
        self.panel.setObjectName("overlayRailPanel")
        self.panel.setFrameShape(QFrame.NoFrame)
        self.panel.setAttribute(Qt.WA_StyledBackground, True)

        self.title_label = QLabel("Overlay Controls")
        self.title_label.setObjectName("overlayRailTitle")
        self.title_label.setFixedWidth(self._s(230))

        # ---------------- Alpha ----------------
        self.lbl_alpha = QLabel("Alpha")
        self.lbl_alpha.setObjectName("overlayRailLabel")

        self.lbl_alpha_value = QLabel("0.50")
        self.lbl_alpha_value.setObjectName("overlayRailValue")
        self.lbl_alpha_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        alpha_head = QHBoxLayout()
        alpha_head.setContentsMargins(0, 0, 0, 0)
        alpha_head.setSpacing(self._s(12))
        alpha_head.addWidget(self.lbl_alpha)
        alpha_head.addStretch(1)
        alpha_head.addWidget(self.lbl_alpha_value)

        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setObjectName("overlayRailSlider")
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(50)
        self.slider_alpha.setSingleStep(1)
        self.slider_alpha.setPageStep(10)
        self.slider_alpha.setTracking(True)
        self.slider_alpha.setMinimumHeight(self._s(28))

        # ---------------- d_x ----------------
        self.lbl_dx = QLabel("Plane depth")
        self.lbl_dx.setObjectName("overlayRailLabel")

        self.lbl_dx_value = QLabel("0.0")
        self.lbl_dx_value.setObjectName("overlayRailValue")
        self.lbl_dx_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        dx_head = QHBoxLayout()
        dx_head.setContentsMargins(0, 0, 0, 0)
        dx_head.setSpacing(self._s(12))
        dx_head.addWidget(self.lbl_dx)
        dx_head.addStretch(1)
        dx_head.addWidget(self.lbl_dx_value)

        self.slider_dx = QSlider(Qt.Horizontal)
        self.slider_dx.setObjectName("overlayRailSlider")
        self.slider_dx.setRange(-50, 50)
        self.slider_dx.setValue(0)
        self.slider_dx.setSingleStep(1)
        self.slider_dx.setPageStep(5)
        self.slider_dx.setTracking(True)
        self.slider_dx.setMinimumHeight(self._s(28))

        # ---------------- Tip / drill views ----------------
        self.chk_show_tip = QCheckBox("Show pointer tip")
        self.chk_show_tip.setObjectName("overlayRailCheck")
        self.chk_show_tip.setChecked(True)
        self.chk_show_tip.setMinimumHeight(self._s(26))

        # The two views of the drill (tracking_overlay.TrackingOverlay).
        # NoFocus: SPACE accepts the pointer pose and must not toggle them.
        self.chk_show_camera_view = QCheckBox("Show camera view")
        self.chk_show_camera_view.setObjectName("overlayRailCheck")
        self.chk_show_camera_view.setChecked(True)
        self.chk_show_camera_view.setMinimumHeight(self._s(26))
        self.chk_show_camera_view.setFocusPolicy(Qt.NoFocus)
        self.chk_show_camera_view.setToolTip("Green: the drill as the camera sees it")

        self.chk_show_xray_view = QCheckBox("Show X-ray view")
        self.chk_show_xray_view.setObjectName("overlayRailCheck")
        self.chk_show_xray_view.setChecked(True)
        self.chk_show_xray_view.setMinimumHeight(self._s(26))
        self.chk_show_xray_view.setFocusPolicy(Qt.NoFocus)
        self.chk_show_xray_view.setToolTip(
            "Magenta: the drill moved along the source rays onto the reference "
            "plane, where the X-ray shows its shadow; only inside the X-ray field"
        )

        # ---------------- Tool tracking ----------------
        # Only shown when a tool is loaded (set_tracking_visible). Tracking
        # is started and stopped via the button; in between it searches
        # until stopped: after a loss the engine is reset and the search
        # goes on.
        self.btn_track = QPushButton("Start tracking")
        self.btn_track.setObjectName("overlayRailTrackBtn")
        self.btn_track.setMinimumHeight(self._s(40))

        self.lbl_track_status = QLabel("idle")
        self.lbl_track_status.setObjectName("overlayRailValue")
        self.lbl_track_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Colour exposure while tracking: checked = the marker exposure
        # control (OverlayLivePage._start_exposure_control), unchecked = the
        # colour AE. Can be switched while tracking.
        self.chk_marker_exposure = QCheckBox("Marker exposure")
        self.chk_marker_exposure.setObjectName("overlayRailCheck")
        self.chk_marker_exposure.setMinimumHeight(self._s(26))
        self.chk_marker_exposure.setFocusPolicy(Qt.NoFocus)
        self.chk_marker_exposure.setToolTip(
            "Set the colour exposure from the drill marker while tracking "
            "(off: the camera's auto exposure)"
        )

        # ---------------- Cards ----------------
        alpha_card = QFrame()
        alpha_card.setObjectName("overlayRailGroup")
        alpha_card.setAttribute(Qt.WA_StyledBackground, True)
        alpha_layout = QVBoxLayout(alpha_card)
        alpha_layout.setContentsMargins(
            self._s(18),
            self._s(14),
            self._s(18),
            self._s(14),
        )
        alpha_layout.setSpacing(self._s(10))
        alpha_layout.addLayout(alpha_head)
        alpha_layout.addWidget(self.slider_alpha)

        dx_card = QFrame()
        dx_card.setObjectName("overlayRailGroup")
        dx_card.setAttribute(Qt.WA_StyledBackground, True)
        dx_layout = QVBoxLayout(dx_card)
        dx_layout.setContentsMargins(
            self._s(18),
            self._s(14),
            self._s(18),
            self._s(14),
        )
        dx_layout.setSpacing(self._s(10))
        dx_layout.addLayout(dx_head)
        dx_layout.addWidget(self.slider_dx)

        tip_card = QFrame()
        tip_card.setObjectName("overlayRailGroup")
        tip_card.setAttribute(Qt.WA_StyledBackground, True)
        tip_layout = QVBoxLayout(tip_card)
        tip_layout.setContentsMargins(
            self._s(18),
            self._s(13),
            self._s(18),
            self._s(13),
        )
        tip_layout.setSpacing(self._s(8))
        tip_layout.addWidget(self.chk_show_tip)
        tip_layout.addWidget(self.chk_show_camera_view)
        tip_layout.addWidget(self.chk_show_xray_view)

        # ---------------- Snapshot ----------------
        # Saves pixel-aligned presentation layers of the current frame
        # (see OverlayLivePage._on_save_snapshot).
        self.btn_save = QPushButton("Save snapshot")
        self.btn_save.setObjectName("overlayRailTrackBtn")
        self.btn_save.setMinimumHeight(self._s(40))

        save_card = QFrame()
        save_card.setObjectName("overlayRailGroup")
        save_card.setAttribute(Qt.WA_StyledBackground, True)
        save_layout = QVBoxLayout(save_card)
        save_layout.setContentsMargins(
            self._s(18),
            self._s(13),
            self._s(18),
            self._s(13),
        )
        save_layout.setSpacing(self._s(8))
        save_layout.addWidget(self.btn_save)

        # ---------------- Reference plane (Evaluation mode) ----------------
        # New X-ray + pointer measurement for the next plane without leaving
        # the live view (see OverlayLivePage._on_plane_button). NoFocus:
        # SPACE accepts the pointer pose and must not re-click the button.
        self.btn_plane = QPushButton("Set reference plane")
        self.btn_plane.setObjectName("overlayRailTrackBtn")
        self.btn_plane.setMinimumHeight(self._s(40))
        self.btn_plane.setFocusPolicy(Qt.NoFocus)

        self.lbl_plane_status = QLabel("")
        self.lbl_plane_status.setObjectName("overlayRailValue")
        self.lbl_plane_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_plane_status.setWordWrap(True)

        self.plane_card = QFrame()
        self.plane_card.setObjectName("overlayRailGroup")
        self.plane_card.setAttribute(Qt.WA_StyledBackground, True)
        plane_layout = QVBoxLayout(self.plane_card)
        plane_layout.setContentsMargins(
            self._s(18),
            self._s(13),
            self._s(18),
            self._s(13),
        )
        plane_layout.setSpacing(self._s(8))
        plane_layout.addWidget(self.btn_plane)
        plane_layout.addWidget(self.lbl_plane_status)
        self.plane_card.hide()
        self._plane_visible = False

        self.track_card = QFrame()
        self.track_card.setObjectName("overlayRailGroup")
        self.track_card.setAttribute(Qt.WA_StyledBackground, True)
        track_layout = QVBoxLayout(self.track_card)
        track_layout.setContentsMargins(
            self._s(18),
            self._s(13),
            self._s(18),
            self._s(13),
        )
        track_layout.setSpacing(self._s(8))
        track_layout.addWidget(self.btn_track)
        track_layout.addWidget(self.lbl_track_status)
        track_layout.addWidget(self.chk_marker_exposure)
        self.track_card.hide()
        self._tracking_visible = False

        # ---------------- Back (tracking only) ----------------
        # In the tracking test (no X-ray) the page has the rail as its only
        # control, so it carries the way back to Tool Tracking (the main
        # window wires it as the page's btn_back). With an X-ray it stays
        # hidden.
        self.btn_back = QPushButton("Back to Tool Tracking")
        self.btn_back.setObjectName("overlayRailTrackBtn")
        self.btn_back.setMinimumHeight(self._s(40))
        self.btn_back.setFocusPolicy(Qt.NoFocus)

        self.back_card = QFrame()
        self.back_card.setObjectName("overlayRailGroup")
        self.back_card.setAttribute(Qt.WA_StyledBackground, True)
        back_layout = QVBoxLayout(self.back_card)
        back_layout.setContentsMargins(
            self._s(18),
            self._s(13),
            self._s(18),
            self._s(13),
        )
        back_layout.addWidget(self.btn_back)
        self.back_card.hide()
        self._tracking_only = False

        # hidden without an X-ray (set_tracking_only)
        self.alpha_card = alpha_card
        self.dx_card = dx_card

        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(
            self._s(22),
            self._s(22),
            self._s(22),
            self._s(22),
        )
        panel_layout.setSpacing(self._s(16))
        panel_layout.addWidget(self.title_label, 0, Qt.AlignLeft)
        panel_layout.addWidget(alpha_card)
        panel_layout.addWidget(dx_card)
        panel_layout.addWidget(tip_card)
        panel_layout.addWidget(save_card)
        panel_layout.addWidget(self.plane_card)
        panel_layout.addWidget(self.track_card)
        panel_layout.addWidget(self.back_card)
        panel_layout.addStretch(1)

        self._apply_style()
        self._update_geometry()

    def _s(self, value: float) -> int:
        return max(1, int(round(float(value) * self.SCALE)))

    def set_tracking_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible == self._tracking_visible:
            return
        self._tracking_visible = visible
        self.track_card.setVisible(visible)
        self._update_geometry()

        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "_position_controls"):
            parent._position_controls()

    def set_plane_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible == self._plane_visible:
            return
        self._plane_visible = visible
        self.plane_card.setVisible(visible)
        self._update_geometry()

        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "_position_controls"):
            parent._position_controls()

    def set_tracking_only(self, on: bool) -> None:
        """Camera image + drill, no X-ray: the X-ray controls (alpha, plane
        depth, pointer tip, X-ray view) are hidden and the Back card is shown."""
        on = bool(on)
        if on == self._tracking_only:
            return
        self._tracking_only = on
        self.alpha_card.setVisible(not on)
        self.dx_card.setVisible(not on)
        self.chk_show_tip.setVisible(not on)
        self.chk_show_xray_view.setVisible(not on)
        self.back_card.setVisible(on)
        self.title_label.setText("Tracking Controls" if on else "Overlay Controls")
        self.handle.setToolTip(self._handle_tooltip())
        self._update_geometry()

        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "_position_controls"):
            parent._position_controls()

    def _panel_height(self) -> int:
        return (
            self.PANEL_H
            + (self.TRACK_EXTRA_H if self._tracking_visible else 0)
            + (self.PLANE_EXTRA_H if self._plane_visible else 0)
            + (self.BACK_H - self.XRAY_H if self._tracking_only else 0)
        )

    def _handle_tooltip(self) -> str:
        what = "tracking" if self._tracking_only else "overlay"
        return f"{'Hide' if self._expanded else 'Show'} {what} controls"

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self.handle.setText("›" if self._expanded else "‹")
        self.handle.setToolTip(self._handle_tooltip())
        self._update_geometry()

        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "_position_controls"):
            parent._position_controls()

    def _update_geometry(self) -> None:
        handle_h = self._s(86)
        handle_y = 0

        if self._expanded:
            panel_h = self._panel_height()
            total_w = self.PANEL_W + self.GAP + self.HANDLE_W
            total_h = panel_h
            self.setFixedSize(total_w, total_h)

            self.panel.show()
            self.panel.setGeometry(0, 0, self.PANEL_W, panel_h)
            self.handle.setGeometry(
                self.PANEL_W + self.GAP,
                handle_y,
                self.HANDLE_W,
                handle_h,
            )
        else:
            self.panel.hide()
            self.setFixedSize(self.HANDLE_W, handle_h)
            self.handle.setGeometry(0, 0, self.HANDLE_W, handle_h)

    def _apply_style(self) -> None:
        border_radius_panel = self._s(22)
        border_radius_handle = self._s(17)
        border_radius_group = self._s(16)
        checkbox_radius = self._s(5)

        font_title = self._s(22)
        font_label = self._s(14)
        font_handle = self._s(28)

        slider_groove_h = self._s(7)
        slider_handle = self._s(18)
        slider_handle_margin = -self._s(6)

        checkbox_size = self._s(18)

        self.setStyleSheet(
            f"""
            QFrame#overlayControlRail {{
                background: transparent;
                border: none;
            }}

            QFrame#overlayRailPanel {{
                background-color: rgba(18, 24, 38, 232);
                border: 1px solid rgba(255, 255, 255, 44);
                border-radius: {border_radius_panel}px;
            }}

            QToolButton#overlayRailHandle {{
                background-color: rgba(18, 24, 38, 226);
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: {border_radius_handle}px;
                color: #f8fafc;
                font-size: {font_handle}px;
                font-weight: 700;
                padding: 0px;
            }}

            QToolButton#overlayRailHandle:hover {{
                background-color: rgba(30, 41, 59, 242);
                border: 1px solid rgba(96, 165, 250, 150);
                color: #ffffff;
            }}

            QLabel#overlayRailTitle {{
                background: transparent;
                color: #f8fafc;
                font-size: {font_title}px;
                font-weight: 800;
                min-height: {self._s(28)}px;
            }}

            QFrame#overlayRailGroup {{
                background-color: rgba(255, 255, 255, 24);
                border: 1px solid rgba(255, 255, 255, 34);
                border-radius: {border_radius_group}px;
            }}

            QLabel#overlayRailLabel {{
                background: transparent;
                color: #e2e8f0;
                font-size: {font_label}px;
                font-weight: 700;
                min-height: {self._s(22)}px;
            }}

            QLabel#overlayRailValue {{
                background: transparent;
                color: #ffffff;
                font-size: {font_label}px;
                font-weight: 800;
                min-width: {self._s(86)}px;
                min-height: {self._s(22)}px;
            }}

            QCheckBox#overlayRailCheck {{
                background: transparent;
                color: #ffffff;
                font-size: {font_label}px;
                font-weight: 700;
                spacing: {self._s(10)}px;
            }}

            QCheckBox#overlayRailCheck::indicator {{
                width: {checkbox_size}px;
                height: {checkbox_size}px;
                border-radius: {checkbox_radius}px;
                border: 1px solid rgba(255, 255, 255, 95);
                background-color: rgba(15, 23, 42, 180);
            }}

            QCheckBox#overlayRailCheck::indicator:checked {{
                background-color: #3b82f6;
                border: 1px solid #93c5fd;
            }}

            QSlider#overlayRailSlider {{
                background: transparent;
                min-height: {self._s(28)}px;
            }}

            QSlider#overlayRailSlider::groove:horizontal {{
                height: {slider_groove_h}px;
                background-color: rgba(148, 163, 184, 80);
                border-radius: {max(1, slider_groove_h // 2)}px;
            }}

            QSlider#overlayRailSlider::sub-page:horizontal {{
                height: {slider_groove_h}px;
                background-color: #3b82f6;
                border-radius: {max(1, slider_groove_h // 2)}px;
            }}

            QSlider#overlayRailSlider::add-page:horizontal {{
                height: {slider_groove_h}px;
                background-color: rgba(148, 163, 184, 80);
                border-radius: {max(1, slider_groove_h // 2)}px;
            }}

            QSlider#overlayRailSlider::handle:horizontal {{
                width: {slider_handle}px;
                height: {slider_handle}px;
                margin: {slider_handle_margin}px 0px;
                border-radius: {max(1, slider_handle // 2)}px;
                background-color: #ffffff;
                border: {self._s(3)}px solid #3b82f6;
            }}

            QSlider#overlayRailSlider::handle:horizontal:hover {{
                border: {self._s(3)}px solid #60a5fa;
            }}

            QPushButton#overlayRailTrackBtn {{
                background-color: #3b82f6;
                color: #ffffff;
                border: none;
                border-radius: {self._s(10)}px;
                font-size: {font_label}px;
                font-weight: 700;
                padding: {self._s(8)}px;
            }}

            QPushButton#overlayRailTrackBtn:hover {{
                background-color: #2563eb;
            }}

            QPushButton#overlayRailTrackBtn:disabled {{
                background-color: rgba(148, 163, 184, 120);
                color: rgba(255, 255, 255, 160);
            }}
            """
        )


# ============================================================
# Page
# ============================================================

class OverlayLivePage(QWidget):
    FPS = 30
    POLL_OVERSAMPLE = 5          # timer ticks per camera frame
    COLOR_SIZE = (1920, 1080)
    ALPHA_SLIDER_MAX = 100
    DX_OFFSET_MIN_MM = -10
    DX_OFFSET_MAX_MM = 10

    # Tool tracking (button-driven: started and stopped via the rail button,
    # searching in between). The tracker runs in its own thread
    # (overlay.tools.live_tracking), so the windows count seconds of frame
    # time, not processed frames; 1.5 s is the validated 45 frames at 30 fps
    # (run_tracker TRACKING_STALE_TO_IDLE_FRAMES).
    TRACK_SEARCH_HINT_S = 3.0     # no lock for 3 s -> status hint, search goes on
    TRACK_LOST_RESET_S = 1.5      # 1.5 s without a fresh pose -> engine reset, search on
    # A pose frame is shown only while it is the newest result and younger
    # than this: bounds the freeze if the tracker stalls, and stays above
    # tracking time + one camera period + pickup, so a pose stream never
    # flickers to the live image.
    TRACK_POSE_MAX_AGE_S = 0.10
    # Frame copies in flight at a submit: the one being filled + pending +
    # processing + the newest undrained result + the one the page holds
    # (the result frame on screen; _choose_frame shows or drops every other
    # one within the tick that drained it).
    TRACK_FRAME_POOL = 5
    # Manual IR exposure while the IR pair streams (same default the
    # run_tracker IR path uses; short = less motion blur on the pair).
    TRACK_IR_EXPOSURE_US = 8000.0

    # Emitted by the tracking thread for every result, queued to the GUI
    # thread: the result's frame (with or without the drill) is shown when
    # the result is ready instead of at the next timer tick (up to one
    # Windows timer tick, 15.6 ms, later).
    _track_result_ready = Signal()
    # Emitted by a snapshot's writer thread when its files are written,
    # queued to the GUI thread: (folder, written layers, eval numbers,
    # error text or "").
    _snapshot_saved = Signal(object)

    def __init__(self, state: SessionState, on_complete_changed=None, parent=None):
        super().__init__(parent)

        self.state = state
        self.on_complete_changed = on_complete_changed

        self.camera: RealSenseCameraSource | None = None
        # colour options read back after the camera start
        # (camera_setup.realsense_color_readback)
        self._camera_color: dict = {}
        self.timer: QTimer | None = None

        # --- tool tracking runtime state ---
        self._tool = None                  # ToolAssets while on the page
        self._tracker = None               # HydraTracker while on the page
        self._tracking_active = False
        # Thread + state machine while tracking (overlay.tools.live_tracking);
        # the worker owns self._tracker until it is stopped.
        self._track_worker: TrackingWorker | None = None
        self._track_session: TrackingSession | None = None
        self._tool_overlay: TrackingOverlay | None = None
        self._overlay_out = None           # TrackOutput behind _tool_overlay's geometry
        self._pose_out = None              # newest result with a fresh pose (holds its frame)
        self._live_out = None              # newest result without one (holds its frame)
        self._shown_out = None             # result whose frame is on screen (None: a polled frame)
        self._drawn_out = None             # result whose drill is drawn on it (None: no drill)
        self._mark_below_visible_frac = 0.0  # marking threshold of the last tracking start
        self._shown_index = -1             # camera frame_index of the frame on screen
        self._shown_t_frame = 0.0          # its poll time
        self._outbox_dropped_seen = 0      # profiler: worker counter already reported
        # The newest submitted frame and the newest result, so the display
        # can wait for a pose on its way (_pose_on_its_way).
        self._track_submitted_index = -1   # camera frame_index of the newest submit
        self._track_result_index = -1      # ... of the newest result drained
        self._track_result_lag_s = 0.0     # its poll -> result time
        self._track_result_t_done = 0.0    # when it was ready
        # The newest polled frame the display held back (its view lives
        # until the next poll) and the poll time of the first one held back
        # since a frame went up: the wait ends TRACK_POSE_MAX_AGE_S after it.
        self._withheld = None              # (CameraFrame, t_frame) or None
        self._t_withheld: float | None = None
        self._tick_s = 0.0                 # timer interval (_start_timer)
        # A worker whose engine call had not returned when tracking stopped;
        # no new tracker starts while it still runs (_start_tracking).
        self._abandoned_worker: TrackingWorker | None = None
        self._abandoned_reason = ""        # the status of that stop
        self._engine_releases_gil: bool | None = None
        self._has_fresh_pose = None        # tracker_log.has_fresh_pose (lazy)
        self._tracker_log_module = None    # tracking.hydramarker.tracker_log (lazy)
        # Created in _make_tracker iff TrackerConfig.failure_log_enabled and
        # Evaluation mode (the file lives in the session folder).
        self._track_fail_log: _TrackFailureLog | None = None

        # The image on screen (before the X-ray layer): a result's frame
        # copy, the page's copy of a polled frame, or the view of the last
        # poll (_screen_view: valid until the next poll, and the display
        # holds no poll back while it is on screen).
        self._live_color: np.ndarray | None = None
        self._screen_view = False
        self._screen_copy: np.ndarray | None = None
        self._display_bgra: np.ndarray | None = None

        self._xray_gray_u8: np.ndarray | None = None
        self._xray_working_u8: np.ndarray | None = None
        # (XrayView, XrayField) of the current X-ray, _attach_xray_view
        self._xray_geometry: tuple[XrayView, XrayField] | None = None
        self._overlay_cache = None
        self._frame_buf: np.ndarray | None = None
        self._scaled_buf: np.ndarray | None = None

        self._alpha: float = 0.50
        self._show_tip: bool = True
        self._show_camera_view: bool = True
        self._show_xray_view: bool = True

        self._base_d_x_mm: float | None = None
        self._current_d_x_mm: float | None = None
        self._dx_offset_mm: int = 0

        self._H_xc_lookup: dict[int, np.ndarray] = {}

        # The Evaluation tracking test (no X-ray calibration), set by the
        # main window. Only then does the page run without an X-ray: camera
        # image + drill, set in on_enter (_is_tracking_only).
        self.tracking_test = False
        self._tracking_only = False

        # --- "Set reference plane" (Evaluation mode) ---
        # While acquiring, every frame runs the pointer solve of the
        # Pointer Tool Depth page; SPACE + confirm swaps in the new X-ray
        # and d_x. Nothing in state changes before the confirm.
        self._pointer: PointerDepthEstimator | None = None
        self._plane_acquiring = False
        self._plane_res = None             # filtered solve of the last frame
        self._plane_raw = None             # unfiltered solve of the same frame
        self._plane_d_x_mm: float | None = None
        self._plane_rgb: np.ndarray | None = None   # BGR copy of that frame
        self._plane_xray: np.ndarray | None = None  # pending X-ray (raw u8)
        self._plane_xray_path: str | None = None
        self._planes_set = 0               # accepted since entering the page

        # Profiling lines go to a file only in Evaluation mode (see
        # PROFILE_LOG_NAME); the session is attached later, so the profile
        # is created in on_enter.
        self._profile: _FrameProfile | None = None
        # Evaluation mode: every processed result and every shown frame
        # (_EvalFrameLog), also created in on_enter.
        self._frame_log: _EvalFrameLog | None = None

        # --- colour exposure while tracking (color_exposure.py) ---
        # "marker": the marker exposure control runs from the tracking
        # start to its stop (the checkbox switches it, also while
        # tracking); "auto": the colour AE as configured.
        self._exposure_mode = CameraConfig().realsense_color_exposure_mode
        self._exposure: MarkerExposureControl | None = None   # while it runs
        # the displayed image only: shown as the AE would expose it while the
        # control sets the camera for the marker (display_exposure)
        self._display_look = AutoExposureLook()
        # the colour setting of each recently polled frame (frame metadata
        # "color_exposure", by frame_index): the LUT needs the setting of the
        # frame on screen, which may be an older frame of a tracking result
        self._frame_exposure: dict[int, dict | None] = {}

        self._build_ui()
        self._connect_signals()
        self._load_xray_from_state()
        self._initialize_dx()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _build_ui(self) -> None:
        self.setObjectName("overlayLivePage")
        self.setFocusPolicy(Qt.StrongFocus)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: black;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.controls = _OverlayControlRail(self)
        # the main window wires a page's btn_back (shown in tracking only)
        self.btn_back = self.controls.btn_back

        self.setStyleSheet(
            """
            QWidget#overlayLivePage {
                background: black;
            }
            """
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        self.image_label.setGeometry(self.rect())
        self._position_controls()
        self._update_image_label()

    def _position_controls(self) -> None:
        margin_top = 18
        margin_right = 18
        size = self.controls.size()

        x = self.width() - size.width() - margin_right - 8
        y = margin_top

        self.controls.setGeometry(QRect(x, y, size.width(), size.height()))
        self.controls.raise_()

    def _connect_signals(self) -> None:
        self.controls.slider_alpha.valueChanged.connect(self._on_alpha_changed)
        self.controls.slider_dx.valueChanged.connect(self._on_dx_offset_changed)
        self.controls.chk_show_tip.toggled.connect(self._on_show_tip_toggled)
        self.controls.chk_show_camera_view.toggled.connect(self._on_show_camera_view_toggled)
        self.controls.chk_show_xray_view.toggled.connect(self._on_show_xray_view_toggled)
        self.controls.btn_track.clicked.connect(self._on_track_button)
        self.controls.chk_marker_exposure.setChecked(self._exposure_mode == MODE_MARKER)
        self.controls.chk_marker_exposure.toggled.connect(self._on_marker_exposure_toggled)
        self.controls.btn_save.clicked.connect(self._on_save_snapshot)
        self.controls.btn_plane.clicked.connect(self._on_plane_button)
        self._track_result_ready.connect(self._on_track_result_ready, Qt.QueuedConnection)
        self._snapshot_saved.connect(self._on_snapshot_saved, Qt.QueuedConnection)

    # --------------------------------------------------------
    # State / prerequisites
    # --------------------------------------------------------

    def _has_prerequisites(self) -> bool:
        return (
            self.state.K_rgb is not None
            and self.state.K_xray is not None
            and self.state.T_xc is not None
            and self.state.d_x is not None
            and self.state.tip_uv_c is not None
        )

    def _is_tracking_only(self) -> bool:
        """The Evaluation tracking test (tracking_test, set by the main
        window) with a tool and the camera intrinsics: the page shows the
        camera image with the drill only, no X-ray. Outside the test the
        page needs the whole X-ray chain as before."""
        return (
            self.tracking_test
            and self.state.tool_assets is not None
            and self.state.K_rgb is not None
        )

    def _clear_xray_layer(self) -> None:
        """Tracking only: no X-ray image, plane depth or homographies on the
        page (state is left as it is)."""
        self._xray_gray_u8 = None
        self._xray_working_u8 = None
        self._xray_geometry = None
        self._overlay_cache = None
        self._frame_buf = None
        self._reset_dx()

    def _reset_dx(self) -> None:
        """No plane depth on the page (no d_x in state, or tracking only)."""
        self._base_d_x_mm = None
        self._current_d_x_mm = None
        self._dx_offset_mm = 0
        self._H_xc_lookup = {}
        self.controls.lbl_dx_value.setText("-")

    def _load_xray_from_state(self) -> None:
        self._xray_geometry = None
        if self.state.xray_image_anatomy is None:
            self._xray_gray_u8 = None
            self._xray_working_u8 = None
            return

        img = np.asarray(self.state.xray_image_anatomy)

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Keep stored image RAW. It is converted to the X-ray working image space
        # only immediately before warping, so H_xc remains unchanged.
        self._xray_gray_u8 = img

        # Flip and detector undistortion are frame independent, so they happen
        # once here instead of per frame. Same treatment as the preview page:
        # the pose lives in the ideal X-ray frame, so the image must too.
        working = np.ascontiguousarray(cv2.flip(img, 1))
        field = getattr(self.state, "xray_dist_field", None)
        if field is not None:
            working = field.undistort_image_working(working)
        self._xray_working_u8 = working

    def _initialize_dx(self) -> None:
        if self.state.d_x is None:
            self._reset_dx()
            return

        self._base_d_x_mm = float(self.state.d_x)
        self._current_d_x_mm = float(self.state.d_x)
        self._dx_offset_mm = 0

        self.controls.slider_dx.blockSignals(True)
        self.controls.slider_dx.setRange(self.DX_OFFSET_MIN_MM, self.DX_OFFSET_MAX_MM)
        self.controls.slider_dx.setValue(0)
        self.controls.slider_dx.setSingleStep(1)
        self.controls.slider_dx.setPageStep(5)
        self.controls.slider_dx.blockSignals(False)

        self.controls.lbl_dx_value.setText(f"{self._current_d_x_mm:.1f} mm")
        self.controls.lbl_alpha_value.setText(f"{self._alpha:.2f}")

    # --------------------------------------------------------
    # Homography
    # --------------------------------------------------------

    def _precompute_H_xc_lookup(self) -> None:
        self._H_xc_lookup = {}

        if not self._has_prerequisites():
            return

        if self._base_d_x_mm is None:
            return

        T_xc = np.asarray(self.state.T_xc, dtype=np.float64)
        R_xc = T_xc[:3, :3]
        t_xc = T_xc[:3, 3]

        K_c = np.asarray(self.state.K_rgb, dtype=np.float64)
        K_x = np.asarray(self.state.K_xray, dtype=np.float64)

        for dx_offset_mm in range(self.DX_OFFSET_MIN_MM, self.DX_OFFSET_MAX_MM + 1):
            d_x_mm = float(self._base_d_x_mm + dx_offset_mm)

            H_xc = estimate_plane_induced_homography(
                K_c=K_c,
                R_xc=R_xc,
                t_xc=t_xc,
                K_x=K_x,
                d_x=d_x_mm,
            )
            self._H_xc_lookup[int(dx_offset_mm)] = H_xc

    def _get_current_H_xc(self) -> np.ndarray:
        if not self._H_xc_lookup:
            raise RuntimeError("H_xc lookup is empty.")

        if self._dx_offset_mm not in self._H_xc_lookup:
            raise RuntimeError(
                f"No precomputed H_xc for d_x offset {self._dx_offset_mm} mm."
            )

        return self._H_xc_lookup[self._dx_offset_mm]

    def _invalidate_overlay_cache(self) -> None:
        self._overlay_cache = None
        self._frame_buf = None
        self.state.H_xc = None

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    def _compose_frame(self, frame_bgra: np.ndarray, draw_tool: bool = False,
                       exposure: dict | None = None) -> np.ndarray:
        """
        Build the display frame in a persistent buffer.

        Nothing here allocates and nothing touches the full frame except the
        one pass that copies the camera image (a plain memcpy, or the display
        LUT below). draw_tool: frame_bgra is the frame the drill pose was
        computed on, so its tip and axis are drawn (never on any other
        frame). exposure: the colour setting that frame was taken with
        (frame metadata "color_exposure").
        """
        prof = self._profile

        if self._frame_buf is None or self._frame_buf.shape != frame_bgra.shape:
            self._frame_buf = np.empty_like(frame_bgra)

        # The marker exposure control exposes the camera for the marker, not
        # like the AE: a LUT shows the display's copy of the camera image as
        # the AE would (k = emulated AE setting / this frame's own setting,
        # so a controller step does not flicker), before the X-ray blend and
        # the drill drawing - never the tracking input, a saved frame, the
        # X-ray or the drill colours. The LUT pass is the copy. It follows
        # the frame's own setting, also for the frames right after a stop;
        # under the AE (or with no known setting) the frame is shown as is.
        t = time.perf_counter()
        self._display_look.update(frame_bgra, exposure)
        looked = self._display_look.apply(frame_bgra, self._frame_buf)
        if prof:
            prof.add("lut" if looked else "copy", time.perf_counter() - t)

        if self._xray_gray_u8 is not None:
            H_xc = self._get_current_H_xc()
            self.state.H_xc = H_xc

            if self._overlay_cache is None:
                # H_xc/K_xray are defined in XRAY_WORKING_FLIPPED_UV space, so
                # the RAW X-ray image goes into that space before warping.
                # H_xc itself is never modified.
                t = time.perf_counter()
                h, w = frame_bgra.shape[:2]
                # K_rgb/dist_rgb: the warp routes through the lens model so
                # the X-ray lands where the observed frame shows the anatomy,
                # consistent with the tracker's tip/axis projection.
                self._overlay_cache = prepare_live_overlay(
                    xray_gray_u8=self._xray_working_u8,
                    H_xc=H_xc,
                    camera_size_wh=(w, h),
                    K_camera=self.state.K_rgb,
                    dist_coeffs=self.state.dist_rgb,
                )
                if prof:
                    prof.add("prepare*", time.perf_counter() - t)

            if self._overlay_cache is not None:
                t = time.perf_counter()
                self._overlay_cache.blend_into(self._frame_buf, self._alpha)
                if prof:
                    prof.add("blend", time.perf_counter() - t)

        # the reference-plane point of the X-ray overlay (not the drill tip)
        if self._show_tip and not self._tracking_only:
            t = time.perf_counter()
            _draw_point_inplace(self._frame_buf, self.state.tip_uv_c)
            if prof:
                prof.add("tip", time.perf_counter() - t)

        if draw_tool and self._tool_overlay is not None:
            t = time.perf_counter()
            self._tool_overlay.draw_into(
                self._frame_buf, show_camera=self._show_camera_view, show_xray=False
            )
            if prof:
                prof.add("drill", time.perf_counter() - t)
            if self._show_xray_view and self._tool_overlay.has_xray_view:
                t = time.perf_counter()
                self._tool_overlay.draw_into(
                    self._frame_buf, show_camera=False, show_xray=True
                )
                if prof:
                    prof.add("xview", time.perf_counter() - t)

        return self._frame_buf

    # --------------------------------------------------------
    # RealSense
    # --------------------------------------------------------

    def _start_realsense(self) -> None:
        if self.camera is not None:
            return

        # bgra8, not bgr8: this is Qt's native 32-bit layout, so the frame
        # goes to the screen without a single per-pixel conversion. The IR
        # pair only streams when a tool is loaded; enabling it also switches
        # the emitter/laser off inside the camera source (speckle would ruin
        # the checkerboard corners) — which is irrelevant here because this
        # page never uses depth.
        with_ir = self._tool is not None
        cfg = CameraConfig(
            width=self.COLOR_SIZE[0],
            height=self.COLOR_SIZE[1],
            fps=self.FPS,
            realsense_color_format="bgra8",
            realsense_frames_queue_size=1,
            realsense_enable_ir=with_ir,
            realsense_ir_exposure_us=self.TRACK_IR_EXPOSURE_US if with_ir else None,
        )
        camera = RealSenseCameraSource(cfg)
        camera.start()
        self.camera = camera
        self._frame_exposure = {}     # frame numbers start over
        # The colour options are the CameraConfig defaults; the start sets
        # them and reads them back. Evaluation mode keeps the read-back:
        # session.json "realsense_color" (the last start) and a log.md row
        # per start.
        self._camera_color = realsense_color_readback(camera.calibration_info)
        ev = self.state.eval
        if ev is not None:
            # bookkeeping must not fail the camera start
            try:
                ev.set_meta("realsense_color", self._camera_color)
                ev.log("overlay_live", "-", realsense_color_summary(self._camera_color),
                       tagged=False)
            except Exception as e:
                print(f"[WARN] Could not record the colour options: {e}")

    def _stop_realsense(self) -> None:
        if self.camera is None:
            return

        self._stop_exposure_control("camera stop")
        try:
            self.camera.stop()
        except Exception:
            pass

        self.camera = None

    def _start_timer(self) -> None:
        if self.timer is None:
            self.timer = QTimer(self)
            self.timer.setTimerType(Qt.PreciseTimer)
            self.timer.timeout.connect(self._on_timer)

        # Poll several times per camera period instead of once.
        #
        # With one tick per frame the timer and the camera run at the same
        # nominal rate but drift against each other, so a missed tick costs a
        # whole period - measured 47 ms of interval for 2.9 ms of work, i.e.
        # 21 fps instead of 30. An empty poll costs 0.1 ms, so oversampling is
        # far cheaper than the beat it removes.
        interval_ms = max(1, int(round(1000.0 / (float(self.FPS) * self.POLL_OVERSAMPLE))))
        self._tick_s = interval_ms / 1000.0
        self.timer.start(interval_ms)

    def _stop_timer(self) -> None:
        if self.timer is not None:
            self.timer.stop()

    def _on_timer(self, poll: bool = True) -> None:
        if self.camera is None:
            return

        abandoned = self._abandoned_worker
        if abandoned is not None and not abandoned.alive:
            # The engine call it was stuck in has returned: a new tracker
            # can start, the status no longer says the engine is busy.
            self._abandoned_worker = None
            if not self._tracking_active:
                self.controls.lbl_track_status.setText(self._abandoned_reason)

        prof = self._profile

        try:
            t_poll = time.perf_counter()
            # Zero-copy views into the SDK buffers, valid until the next
            # poll() — everything that reads them runs within this tick: the
            # tracking thread gets copies (submit), and a live frame is
            # copied into the persistent frame buffer as the first step of
            # _compose_frame. poll=False: woken by a tracking result, which
            # only needs the drain and the display.
            frame = self.camera.poll() if poll else None
            if prof and poll:
                prof.note_poll(empty=frame is None)

            t_frame = 0.0
            if frame is not None:
                t_frame = time.perf_counter()
                if prof:
                    prof.add("poll", t_frame - t_poll)
                    prof.note_camera_frame(self._camera_frame_number())
                self._note_frame_exposure(frame)
                if self._frame_log is not None:
                    self._frame_log.note_frame(frame)
                worker = self._track_worker
                if worker is not None:
                    n_dry = worker.n_pool_exhausted
                    taken = worker.submit(frame, t_frame)
                    if taken:
                        self._track_submitted_index = int(frame.frame_index)
                    if prof:
                        prof.add("submit", time.perf_counter() - t_frame)
                        if taken:
                            prof.count("submitted")
                        # only a missing slot counts (a worker that has
                        # ended takes nothing either)
                        prof.count("pool_exhausted", worker.n_pool_exhausted - n_dry)

            if self._track_worker is not None:
                self._drain_tracking()

            if frame is not None and self._plane_acquiring:
                t = time.perf_counter()
                self._plane_step(frame.image_bgr)
                if prof:
                    prof.add("plane", time.perf_counter() - t)

            shown = self._choose_frame(frame, t_frame)
            if shown is None:
                return
            img, pose_out = shown   # BGRA (realsense_color_format="bgra8")

            t0 = prof.frame_start() if prof else 0.0
            self._live_color = img
            self._display_bgra = self._compose_frame(
                img, draw_tool=pose_out is not None,
                exposure=self._frame_exposure.get(self._shown_index),
            )
            # the result whose drill is now on screen (the snapshot redraws it)
            ov = self._tool_overlay
            self._drawn_out = (
                pose_out if pose_out is not None and ov is not None and ov.has_pose else None
            )
            if self._plane_acquiring:
                self._draw_plane_acquire(self._display_bgra)
            self._update_image_label()

            if prof:
                prof.add("age", time.perf_counter() - self._shown_t_frame)
                prof.count("shown_pose" if pose_out is not None else "shown_live")
                prof.frame_end(t0, {
                    "frame_idx": self._shown_index,
                    "pose": pose_out is not None,
                    "tracking": self._tracking_active,
                    "xray": self._overlay_cache is not None,
                })

        except Exception as e:
            self._end_plane_acquire("error")
            self._stop_tracking("error")
            self._stop_timer()
            self._stop_realsense()
            QMessageBox.critical(
                self,
                "Live Overlay",
                f"Live update failed.\n\n{e}",
            )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    def _note_frame_exposure(self, frame) -> None:
        """Keep the colour setting of a polled frame for the display LUT
        (only the last ~second of frames: a shown result is younger than
        TRACK_POSE_MAX_AGE_S)."""
        index = int(frame.frame_index)
        self._frame_exposure[index] = (frame.metadata or {}).get("color_exposure")
        if len(self._frame_exposure) > 2 * self.FPS:
            self._frame_exposure = {
                i: e for i, e in self._frame_exposure.items() if i > index - self.FPS
            }

    def _camera_frame_number(self) -> int | None:
        """RealSense frame number of the colour frame behind the last poll()
        (the source keeps that frameset until the next poll), for the
        profile's count of camera frames the page never got."""
        frames = getattr(self.camera, "_live_frames", None)
        if frames is None:
            return None
        try:
            return int(frames.get_color_frame().get_frame_number())
        except Exception:
            return None

    def _choose_frame(self, frame, t_frame: float):
        """
        The display rule: (image, pose result or None), or None when there
        is nothing new to show.

        The drill is only ever drawn on the frame its pose was computed on.
        If the newest tracking result has a fresh pose, its frame is not
        older than the one on screen and at most TRACK_POSE_MAX_AGE_S old,
        that frame is shown with tip and axis - later than the live image
        by the tracking time, exactly on the drill. Otherwise (searching, a
        dropout, tracking off, reference plane) the newest frame there is
        is shown at once, without the drill: the one polled in this tick,
        or else the frame of the newest result without a pose. Shown frames
        never go back in time: at a (re)lock the locking frame is usually
        older than the live frame already on screen, so the drill appears
        with the next result. While locked, a polled frame is held back as
        long as the pose of a newer frame is on its way (_pose_on_its_way):
        shown at once, it would make that pose older than the frame on
        screen, and with a tracker slower than the camera no pose would be
        drawn at all. The wait never freezes the image longer than
        TRACK_POSE_MAX_AGE_S after the first polled frame it held back:
        then the newest polled frame goes up without the drill.
        """
        now = time.perf_counter()
        if frame is not None:
            # this poll ended the view of the frame held back before
            self._withheld = None
        cand = self._pose_out
        if cand is not None and (
            cand.frame_index < self._shown_index
            or now - cand.t_frame > self.TRACK_POSE_MAX_AGE_S
        ):
            # can never be shown any more: give its frame copy back
            self._pose_out = None
            self._release_out(cand)
            cand = None
            if self._profile:
                self._profile.count("pose_dropped")
        held = self._withheld
        live, self._live_out = self._live_out, None
        if live is not None and (
            frame is not None
            or live.frame_index <= self._shown_index
            or (held is not None and held[0].frame_index > live.frame_index)
        ):
            # shown now or never: a newer frame is polled, held or on screen
            self._release_out(live)
            live = None

        pose = holder = image = None
        if cand is not None:
            if cand is not self._shown_out:
                holder = pose = cand
                image, index, t_shown = cand.image, cand.frame_index, cand.t_frame
            # else on screen; the frames polled meanwhile come as results
        elif frame is not None:
            # unless the frame on screen stays until a pose arrives
            if not self._pose_on_its_way(now):
                image, index, t_shown = frame.image_bgr, frame.frame_index, t_frame
        elif live is not None:
            # nothing polled in this tick: the result's frame is the newest
            holder = live
            image, index, t_shown = live.image, live.frame_index, live.t_frame
        elif held is not None and not self._pose_on_its_way(now):
            # the wait is over: the frame it held back goes up (its view is
            # still valid, nothing was polled since)
            image, index, t_shown = held[0].image_bgr, held[0].frame_index, held[1]

        if image is not None:
            previous, self._shown_out = self._shown_out, holder
            self._release_out(previous)
            self._shown_index = int(index)
            self._shown_t_frame = t_shown
            self._t_withheld = None
            if self._withheld is not None and self._withheld[0].frame_index <= index:
                self._withheld = None
            image = self._screen_image(image, holder, now)
        if frame is not None and (image is None or holder is not None):
            # polled in this tick and not shown: held back
            self._withheld = (frame, t_frame)
        if self._withheld is not None and self._t_withheld is None:
            self._t_withheld = t_frame if frame is not None else now
        return None if image is None else (image, pose)

    def _screen_image(self, image, holder, now: float):
        """The image that goes on screen. A result's frame copy stays valid
        while the page holds the result; a camera view only until the next
        poll, so it is copied when the wait is armed (the next poll may be
        held back while the view is still on screen)."""
        if holder is not None:
            self._screen_view = False
            return image
        if not self._wait_armed(now):
            self._screen_view = True
            return image
        if self._screen_copy is None or self._screen_copy.shape != image.shape:
            self._screen_copy = np.empty_like(image)
        np.copyto(self._screen_copy, image)
        self._screen_view = False
        return self._screen_copy

    def _wait_armed(self, now: float) -> bool:
        """
        Locked, and a pose of a newer frame can still be shown: results
        come back within TRACK_POSE_MAX_AGE_S (the last one took less, and
        it is not longer ago: a stalled tracker ends the wait, a tracker
        slower than that never starts it), and the first polled frame held
        back is younger than TRACK_POSE_MAX_AGE_S minus two timer ticks (so
        the tick that ends the wait comes in time, also when it is one
        interval late).
        """
        session = self._track_session
        if session is None or not session.locked or session.stale:
            return False
        max_age = self.TRACK_POSE_MAX_AGE_S
        if self._t_withheld is not None and now - self._t_withheld + 2 * self._tick_s >= max_age:
            return False
        return (
            self._track_result_lag_s < max_age
            and now - self._track_result_t_done < max_age
        )

    def _pose_on_its_way(self, now: float) -> bool:
        """
        A frame newer than the one on screen is still in the worker and the
        wait is armed (_wait_armed); never while a camera view is on screen
        (the next poll would end it).
        """
        if self._screen_view:
            return False
        newest = max(self._track_result_index, self._shown_index)
        if self._track_submitted_index <= newest:
            return False
        return self._wait_armed(now)

    def _update_image_label(self) -> None:
        if self._display_bgra is None:
            return

        prof = self._profile

        # Fit the finished full-resolution frame into the window.
        #
        # The composition above stays at 1920x1080 - only this last step
        # reduces to what the widget can actually show, exactly as
        # pix.scaled() did before. Doing it with cv2 on the BGRA buffer costs
        # 0.5 ms instead of 1.5 ms, because the result is then already in Qt's
        # native layout and QPixmap.fromImage has nothing left to convert.
        t = time.perf_counter()
        src_h, src_w = self._display_bgra.shape[:2]
        lw = max(1, self.image_label.width())
        lh = max(1, self.image_label.height())

        # KeepAspectRatioByExpanding: cover the label, let it clip the overflow
        s = max(lw / src_w, lh / src_h)
        tw, th = max(1, int(round(src_w * s))), max(1, int(round(src_h * s)))

        if self._scaled_buf is None or self._scaled_buf.shape[:2] != (th, tw):
            self._scaled_buf = np.empty((th, tw, 4), dtype=np.uint8)

        cv2.resize(self._display_bgra, (tw, th), dst=self._scaled_buf,
                   interpolation=cv2.INTER_LINEAR)
        if prof:
            prof.add("scale", time.perf_counter() - t)

        t = time.perf_counter()
        pix = _bgra_to_qpixmap(self._scaled_buf)
        if not pix.isNull():
            self.image_label.setPixmap(pix)
        if prof:
            prof.add("qpixmap", time.perf_counter() - t)

    # --------------------------------------------------------
    # Tool tracking (button-driven, own thread)
    # --------------------------------------------------------
    #
    # Started and stopped via the rail button. In between the tracker runs
    # in a TrackingWorker thread on copies of the polled frames, so the
    # live image never waits for it, and searches until stopped: after
    # TRACK_LOST_RESET_S without a fresh pose TrackingSession resets the
    # engine and the search goes on (re-locks pass the engine's appearance
    # check). The results are drained in the GUI tick, in order; the state,
    # the overlay, the failure log and the profile are only touched here.

    def _on_track_button(self) -> None:
        if self._plane_acquiring:
            return
        if self._tracking_active:
            self._stop_tracking("stopped")
        else:
            self._start_tracking()

    def _make_tracker(self):
        # Lazy imports: the C++ engine (.pyd) only loads when tracking is
        # actually started, so the overlay works without the tracking build.
        from tracking.hydramarker import tracker_log
        from tracking.hydramarker.run_tracker import (
            make_live_tracker_config,
            make_tracker,
        )

        self._has_fresh_pose = tracker_log.has_fresh_pose
        self._tracker_log_module = tracker_log

        # JSONL run logs (if ever enabled) must land in a fixed absolute
        # location, not in whatever cwd the GUI was started from: the
        # session folder in Evaluation mode, overlay/data/ otherwise.
        ev = self.state.eval
        log_root = (
            ev.dir if ev is not None
            else Path(__file__).resolve().parents[2] / "data"
        )
        tracker_log.set_run_log_dir(str(log_root / "hydramarker_tracker_runs"))

        if self._tracker is not None:
            self._tracker.reset()
            return self._tracker

        config = make_live_tracker_config()
        ir_path = getattr(self.state, "ir_calibration_path", None)
        config.ir_refine_enabled = bool(ir_path)
        # The drill is drawn from the unfiltered pose (tracking_overlay.
        # ToolPoints): the output filter pulls the pose off the image in
        # motion and does not smooth at rest. On again once it is repaired.
        config.pose_kf_enabled = False

        # The log file lives in the Evaluation session folder; Normal mode
        # keeps no failure log.
        if getattr(config, "failure_log_enabled", False) and ev is not None:
            self._track_fail_log = _TrackFailureLog(
                str(ev.dir / TRACK_FAILURE_LOG_NAME)
            )
        else:
            self._track_fail_log = None

        tracker = make_tracker(
            self._tool.field_path,
            self._tool.marker_json_path,
            np.asarray(self.state.K_rgb, dtype=np.float64),
            self.state.dist_rgb,
            config,
        )
        if ir_path:
            tracker.set_ir_calibration(ir_path)
        return tracker

    def _new_tool_overlay(self) -> TrackingOverlay:
        """The drill's overlay, with the X-ray view as the page shows it now."""
        overlay = TrackingOverlay(
            self.state.K_rgb,
            self.state.dist_rgb,
            self._tool.p_tip_mm,
            self._tool.axis_dir,
        )
        self._attach_xray_view(overlay)
        return overlay

    def _attach_xray_view(self, overlay: TrackingOverlay) -> None:
        """Give the overlay the current X-ray and d_x; in the tracking test,
        or without the X-ray chain, no X-ray view. The field's hull is computed
        once per X-ray (_load_xray_from_state drops it). A failure costs only
        the X-ray view: the camera view and the overlay keep running."""
        d_x = self._current_d_x_mm
        if (
            self._tracking_only
            or self._xray_working_u8 is None
            or d_x is None
            or self.state.T_xc is None
            or self.state.K_xray is None
        ):
            overlay.set_xray_view(None)
            return
        try:
            if self._xray_geometry is None:
                view = XrayView(self.state.T_xc, self.state.K_xray)
                h, w = self._xray_working_u8.shape[:2]
                self._xray_geometry = (view, XrayField.from_image(
                    self._xray_working_u8, overlay.field_inset_px(view, (w, h), d_x)
                ))
            overlay.set_xray_view(*self._xray_geometry, d_x)
        except Exception as e:
            print(f"[WARN] Live page: no X-ray view of the drill: {e}")
            overlay.set_xray_view(None)

    def _start_tracking(self) -> None:
        if self._tracking_active or self._tool is None or self.camera is None:
            return
        abandoned = self._abandoned_worker
        if abandoned is not None:
            if abandoned.alive:
                # Its engine call has still not returned: no second tracker
                # next to a hung one (the status of that stop stays).
                self.controls.lbl_track_status.setText(
                    f"{self._abandoned_reason} — engine still busy"
                )
                return
            self._abandoned_worker = None

        try:
            self._tracker = self._make_tracker()
            if self._tool_overlay is None:
                self._tool_overlay = self._new_tool_overlay()
            else:
                # a new reference plane may have come in since the last run
                self._attach_xray_view(self._tool_overlay)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Tool Tracking",
                f"Could not start the tracker.\n\n{e}",
            )
            return

        self._check_engine_gil()
        self._tool_overlay.clear()
        self._overlay_out = None
        if self._track_fail_log is not None:
            self._track_fail_log.start(self._tracker_log_module)
        # A pose kept under a partial occlusion is drawn marked; the threshold
        # is kept for the snapshot record (the tracker may be gone by then).
        self._mark_below_visible_frac = float(getattr(
            self._tracker.config, "drawn_pose_mark_below_visible_frac", 0.0
        ))
        # The worker owns the tracker from here until _stop_tracking joined it.
        self._track_worker = TrackingWorker(
            self._tracker,
            pool_size=self.TRACK_FRAME_POOL,
            notify=self._track_result_ready.emit,
            tool_points=self._tool_overlay.tool_points(
                mark_below_visible_frac=self._mark_below_visible_frac,
            ),
            frame_timing=self._tracker.ir_frame_timing,
        )
        self._track_session = TrackingSession(
            self._has_fresh_pose,
            self._track_worker.request_reset,
            lost_reset_s=self.TRACK_LOST_RESET_S,
            search_hint_s=self.TRACK_SEARCH_HINT_S,
            fail_log=self._track_fail_log,
        )
        self._outbox_dropped_seen = 0
        self._track_submitted_index = self._track_result_index = -1
        self._track_worker.start()
        self._tracking_active = True
        self.controls.btn_track.setText("Stop tracking")
        self.controls.lbl_track_status.setText(STATUS_SEARCHING)
        if self._exposure_mode == MODE_MARKER:
            self._start_exposure_control()

    def _check_engine_gil(self) -> None:
        """A build from before the tracking thread holds the GIL for the
        whole frame: tracking works, but the GUI thread waits for every
        frame as before. Said once on the console, and in the profile log
        at every start."""
        if self._engine_releases_gil is None:
            from tracking.hydramarker.tracker import hydramarker_cpp

            self._engine_releases_gil = bool(
                getattr(hydramarker_cpp, "PROCESS_FRAME_RELEASES_GIL", False)
            )
            if not self._engine_releases_gil:
                print(
                    "[WARN] Live page: this hydramarker_cpp build holds the GIL in "
                    "process_frame; the live image waits for the tracker until it "
                    "is rebuilt."
                )
        if self._profile is not None:
            self._profile.emit(
                f"[live] tracking start {time.strftime('%H:%M:%S')}: worker thread, "
                f"engine releases GIL: {'yes' if self._engine_releases_gil else 'NO'}"
            )

    def _stop_tracking(self, reason: str) -> None:
        self._stop_exposure_control(reason)
        self._tracking_active = False
        worker, self._track_worker = self._track_worker, None
        stopped = True
        error = None
        if worker is not None:
            # Returns after the frame in progress (at most one engine call).
            stopped = worker.stop(timeout_s=2.0)
            # results of that frame are not shown any more; an error no
            # tick has drained yet (a Stop click right after it) is still
            # reported
            for out in worker.take_outputs():
                if out.error is not None and error is None:
                    error = (out.error, out.traceback)
                worker.release(out.slot)
            if error is None and worker.failure is not None:
                error = (worker.failure, worker.failure_traceback or "")
        if error is not None and reason != "error":
            self._report_track_error(*error)
            reason = "error"
        # The frame on screen stays until the next one; the other results
        # the page kept give their frame back.
        for out in (self._pose_out, self._live_out):
            if out is not None and out is not self._shown_out and worker is not None:
                worker.release(out.slot)
                out.slot = None
        self._pose_out = None
        self._live_out = None
        self._track_submitted_index = self._track_result_index = -1
        if self._track_session is not None:
            self._track_session.finish(reason)
            self._track_session = None
        if self._tool_overlay is not None:
            self._tool_overlay.clear()
        self._overlay_out = None
        if self._tracker is not None:
            if stopped:
                try:
                    self._tracker.reset()
                except Exception:
                    pass
            else:
                # Still inside process_frame in the abandoned thread: this
                # engine is never touched again; the next start builds one
                # once that thread has ended.
                self._tracker = None
                self._abandoned_worker = worker
                self._abandoned_reason = reason
                reason += " — engine still busy"
        self.controls.btn_track.setText("Start tracking")
        self.controls.lbl_track_status.setText(reason)
        if self._frame_log is not None:
            # this run's results and the frames shown so far go to disk
            self._frame_log.flush(reason, tracking=True)

    def _on_track_result_ready(self) -> None:
        # a result between two ticks: drain and show it now, no poll
        if self._track_worker is not None:
            self._on_timer(poll=False)

    def _drain_tracking(self) -> None:
        """Every result of the tracking thread since the last tick, in
        processing order."""
        worker = self._track_worker
        prof = self._profile
        t = time.perf_counter()
        # read first: once the thread has ended, all its results are here
        ended = worker.ended
        outs = worker.take_outputs()
        for i, out in enumerate(outs):
            if out.error is not None:
                for rest in outs[i:]:
                    worker.release(rest.slot)
                self._on_track_error(out.error, out.traceback)
                return
            try:
                self._on_track_output(out, t)
            except Exception as e:
                # The GUI side of a result (session, overlay, failure log):
                # a tracking error as well - tracking stops, the camera and
                # the X-ray overlay keep running.
                for rest in outs[i:]:
                    self._release_out(rest)
                self._on_track_error(e, traceback.format_exc())
                return
        if ended:
            # ended without a tracker error: it failed outside the call
            self._on_track_error(
                worker.failure or RuntimeError("the tracking thread ended"),
                worker.failure_traceback or "",
            )
            return
        if not outs:
            return
        if prof:
            prof.add("drain", time.perf_counter() - t)
            dropped = worker.n_outbox_dropped
            if dropped != self._outbox_dropped_seen:
                prof.count("outbox_dropped", dropped - self._outbox_dropped_seen)
                self._outbox_dropped_seen = dropped

    def _on_track_output(self, out, t_drain: float) -> None:
        prof = self._profile
        if prof:
            prof.count("processed")
            prof.count("skipped", out.skipped)
            prof.add("track", out.t_done - out.t_start)
            prof.add("wait", out.t_start - out.t_frame)
            prof.add("lag", out.t_done - out.t_frame)
            prof.add("pickup", t_drain - out.t_done)
        self._track_result_index = max(self._track_result_index, out.frame_index)
        self._track_result_lag_s = out.t_done - out.t_frame
        self._track_result_t_done = out.t_done
        if self._exposure is not None and out.metering is not None:
            self._exposure_step(out.metering)

        step = self._track_session.update(out)
        log = self._frame_log
        if not step.current:
            # computed before the last engine reset: neither shown nor logged
            # (only the evaluation's frame log keeps it: the engine ran on it)
            if log is not None:
                log.add_track(out, step, t_drain, None, self._track_session,
                              self._tracker_log_module)
            self._release_out(out)
            return

        # the overlay first: if it raises, no result has changed hands yet
        if step.fresh:
            self._tool_overlay.update(out.points, marked=out.points_marked)
            if prof and self._tool_overlay.has_xray_view:
                prof.add("xview_up", self._tool_overlay.xray_update_s)
        else:
            self._tool_overlay.clear()
        if log is not None:
            log.add_track(out, step, t_drain,
                          self._tool_overlay if step.fresh else None,
                          self._track_session, self._tracker_log_module)
        previous = (self._pose_out, self._live_out)
        self._pose_out = self._live_out = None
        # only the newest result of a batch still has its frame
        if step.fresh:
            self._overlay_out = out
            if out.slot is not None:
                self._pose_out = out
        else:
            self._overlay_out = None
            if out.slot is not None:
                # the newest image, unless this tick polled a newer one
                self._live_out = out
        for old in previous:
            self._release_out(old)
        self._release_out(out)
        if step.status is not None:
            self.controls.lbl_track_status.setText(step.status)

    def _on_track_error(self, error: BaseException, traceback_text: str) -> None:
        """The tracking thread ended with an exception: stop tracking only;
        the camera and the X-ray overlay keep running."""
        self._report_track_error(error, traceback_text)
        self._stop_tracking("error")

    def _report_track_error(self, error: BaseException, traceback_text: str) -> None:
        print(f"[live] drill tracking failed: {error!r}\n{traceback_text}")
        # After this tick: the box runs its own event loop, and this tick
        # still uses the views of its poll.
        QTimer.singleShot(0, lambda e=error: QMessageBox.critical(
            self,
            "Tool Tracking",
            f"Tracking failed.\n\n{str(e) or type(e).__name__}",
        ))

    def _release_out(self, out) -> None:
        """Give a result's frame copy back to the worker's pool unless it is
        still needed (newest result with or without a pose, or on screen)."""
        if (out is None or out is self._pose_out or out is self._live_out
                or out is self._shown_out):
            return
        if out.slot is not None and self._track_worker is not None:
            self._track_worker.release(out.slot)
        out.slot = None

    # --------------------------------------------------------
    # Marker exposure control (color_exposure.py)
    # --------------------------------------------------------
    #
    # While tracking with the mode "marker": the worker meters the marker's
    # white level on every result (MarkerMeter), the controller decides in
    # this thread when a result is drained, and the camera's actuator thread
    # writes the colour settings (set_option never runs here). Every stop
    # path (Stop, error, plane, page change, camera stop) writes the start
    # settings back - AE on - before anything else.

    def _on_marker_exposure_toggled(self, checked: bool) -> None:
        self._exposure_mode = MODE_MARKER if checked else "auto"
        self.controls.chk_marker_exposure.setText("Marker exposure")
        if not self._tracking_active:
            return
        if checked:
            self._start_exposure_control()
        else:
            self._stop_exposure_control("switched off")

    def _start_exposure_control(self) -> None:
        if self._exposure is not None or self.camera is None or self._track_worker is None:
            return
        control = MarkerExposureControl(self.camera, self._has_fresh_pose)
        try:
            ok = control.start(self._track_worker, time.perf_counter())
        except Exception as e:
            ok = False
            print(f"[WARN] Marker exposure: could not start: {e}")
        if not ok:
            text = ("marker exposure busy — AE" if control.unavailable == "busy"
                    else "marker exposure unavailable — AE")
            self.controls.lbl_track_status.setText(text)
            print(f"[WARN] Marker exposure: {text} ({control.unavailable or 'start failed'})")
            if self._profile is not None:
                self._profile.emit(f"[live] marker exposure unavailable {time.strftime('%H:%M:%S')}: "
                                   f"{control.unavailable or 'start failed'}")
            return
        self._exposure = control
        self._display_look.reset()
        info = control.snapshot()
        cfg = info["config"]
        if self._profile is not None:
            self._profile.emit(
                f"[live] marker exposure on {time.strftime('%H:%M:%S')}: band "
                f"{cfg['white_low_dn']:.0f}-{cfg['white_high_dn']:.0f} DN, clip <= "
                f"{100 * cfg['clip_max_frac']:.0f} %, gain {info['gain_used'][0]:.0f}-"
                f"{info['gain_used'][1]:.0f}, exposure <= {cfg['exposure_max_ms']:g} ms"
            )
        ev = self.state.eval
        if ev is not None:
            try:
                ev.set_meta("color_exposure_control", info)
                ev.log("overlay_live", "-", {"marker_exposure": "on"}, tagged=False)
            except Exception as e:
                print(f"[WARN] Could not record the marker exposure: {e}")

    def _exposure_step(self, metering) -> None:
        control = self._exposure
        cmd = control.step(metering)
        prof = self._profile
        for err in control.new_errors():
            print(f"[WARN] Marker exposure: {err}")
            if prof:
                prof.emit(f"[live] marker exposure error {time.strftime('%H:%M:%S')}: {err}")
            self._log_exposure_event({"marker_exposure": "error", "error": err})
        first = control.take_first_write()
        if first is not None:
            # one line to check after the start: the camera took the first setting
            text = (f"first write {first.exposure_ms:g} ms/{first.gain:.0f} applied, "
                    f"set_option {first.write_ms:.1f} ms")
            print(f"[live] marker exposure: {text}")
            if prof:
                prof.emit(f"[live] marker exposure {time.strftime('%H:%M:%S')}: {text}")
        if control.failure:
            # writes fail: the camera goes back to its AE, the box unchecked
            # (a restore problem found by the stop overrides both texts)
            reason = control.failure
            print(f"[WARN] Marker exposure failed ({reason}): back to the camera's auto exposure")
            self._set_marker_exposure_check(False, "Marker exposure — failed, AE")
            self.controls.lbl_track_status.setText("marker exposure failed — AE")
            self._stop_exposure_control("failed: " + reason)
            return
        if prof:
            if cmd is not None:
                prof.count("exposure_steps")
            prof.note_exposure(
                f"{control.text(metering)} meter {metering.meter_ms:.2f} ms "
                f"lut x{self._display_look.k:.2f}"
            )

    def _stop_exposure_control(self, reason: str) -> None:
        # the display LUT stays with the frames: the 3-4 frames after the
        # restore still carry the control's last setting
        control, self._exposure = self._exposure, None
        if control is None:
            return
        summary = control.stop()
        problems = summary["restore_problems"]
        if self._profile is not None:
            self._profile.note_exposure("")
            self._profile.emit(
                f"[live] marker exposure off {time.strftime('%H:%M:%S')} ({reason}): "
                f"{summary['steps']} steps, {summary['search_steps']} search steps, "
                f"{summary['applied']}/{summary['commands']} commands applied, "
                f"{len(summary['failed_writes'])} failed, restored "
                f"{summary['restored']} in {summary['stop_ms']:.1f} ms"
                + (f" - RESTORE PROBLEM: {', '.join(problems)}" if problems else "")
            )
        if problems:
            print(f"[WARN] Marker exposure: camera not restored ({', '.join(problems)}): "
                  f"{summary['restored']} - restart the app")
            self.controls.lbl_track_status.setText("camera not restored — restart the app")
            self._set_marker_exposure_check(False, "Marker exposure — camera not restored")
        self._log_exposure_event({
            "marker_exposure": "off", "reason": reason, "steps": summary["steps"],
            "search_steps": summary["search_steps"], "restored": summary["restored"],
            "restore_problems": problems, "failure": summary["failure"],
            "failed_writes": summary["failed_writes"],
        })

    def _set_marker_exposure_check(self, checked: bool, text: str) -> None:
        """The checkbox's state and text without starting or stopping the
        control (a failed control unchecks it and says why)."""
        chk = self.controls.chk_marker_exposure
        chk.blockSignals(True)
        chk.setChecked(checked)
        chk.blockSignals(False)
        chk.setText(text)
        self._exposure_mode = MODE_MARKER if checked else "auto"

    def _log_exposure_event(self, data: dict) -> None:
        ev = self.state.eval
        if ev is None:
            return
        try:
            ev.log("overlay_live", "-", data, tagged=False)
        except Exception as e:
            print(f"[WARN] Could not record the marker exposure: {e}")

    # --------------------------------------------------------
    # Controls callbacks
    # --------------------------------------------------------

    def _on_alpha_changed(self, value: int) -> None:
        self._alpha = float(value) / float(self.ALPHA_SLIDER_MAX)
        self.controls.lbl_alpha_value.setText(f"{self._alpha:.2f}")

    def _on_dx_offset_changed(self, value: int) -> None:
        if self._base_d_x_mm is None:
            return

        self._dx_offset_mm = int(value)
        self._current_d_x_mm = float(self._base_d_x_mm + self._dx_offset_mm)

        self.controls.lbl_dx_value.setText(f"{self._current_d_x_mm:.1f} mm")

        self.state.d_x = float(self._current_d_x_mm)
        self.state.H_xc = self._H_xc_lookup.get(self._dx_offset_mm)

        self._overlay_cache = None
        if self._tool_overlay is not None:
            self._tool_overlay.set_plane_depth(self._current_d_x_mm)

    def _on_show_tip_toggled(self, checked: bool) -> None:
        self._show_tip = bool(checked)

    def _on_show_camera_view_toggled(self, checked: bool) -> None:
        self._show_camera_view = bool(checked)

    def _on_show_xray_view_toggled(self, checked: bool) -> None:
        self._show_xray_view = bool(checked)

    def _on_save_snapshot(self) -> None:
        """Save the current frame as stacked presentation layers.

        One timestamped folder with pixel-aligned, same-size PNGs that can be
        placed on top of each other in PowerPoint and revealed in order:

            00_xray.png               raw anatomical X-ray (reference)
            01_rgb.png                camera image only
            02_rgb_overlay.png        + X-ray overlay at the current alpha
            03_rgb_overlay_drill.png  + drill tip and axis, camera view (only
                                        while a tracked pose is on screen)
            04_rgb_overlay_drill_xrayview.png
                                      + the X-ray view of the drill (only while
                                        there is one)
            snapshot.json             the numbers behind the layers: alpha,
                                        d_x and its offset, H_xc, camera
                                        model, reference-plane point, and
                                        while tracking the projected tip/
                                        axis pixels with the pose they came
                                        from - so a thesis figure can redraw
                                        the marks from the bare camera frame
                                        instead of recovering them from the
                                        burnt-in pixels

        Without an X-ray (tracking only) 00, 02 and 04 are left out: 03 is
        the drill on the bare camera image, snapshot.json says tracking_only
        and its X-ray fields are null (show_tip false). Like 03, layer 04 does
        not follow the rail's switches; snapshot.json records them.

        The frame is the one on screen: the tracker's copy of the frame a
        result came from, the page's copy of a polled frame, or the view of
        the last poll - valid, because the click handler runs between two
        timer ticks and the display holds no poll back while a view is on
        screen. The drill layer is the one drawn on that frame, redrawn from
        its result.

        Here, in the click handler, only the layers are built (BGR copies:
        the frame on screen goes back to the camera or the tracker with the
        next tick) together with snapshot.json's numbers; encoding and
        writing the PNGs (~0.2 s, which stalled the live image) runs in a
        thread of its own, which reports back through _snapshot_saved.
        """
        t_click = time.perf_counter()
        if self._live_color is None:
            QMessageBox.information(
                self, "Save snapshot", "No live image yet — start the camera first."
            )
            return

        try:
            ev = self.state.eval
            if ev is not None:
                out_dir = ev.subdir("live")
            else:
                out_dir = SNAPSHOT_DIR / time.strftime("snapshot_%Y%m%d_%H%M%S")
                out_dir.mkdir(parents=True, exist_ok=True)
            layers = []

            if self._xray_gray_u8 is not None:
                # replaced, never changed in place: no copy needed
                layers.append(("00_xray", self._xray_gray_u8))

            base = self._live_color
            layers.append(("01_rgb", cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)))

            # Cumulative composite, built exactly like _compose_frame.
            comp = base.copy()
            if self._overlay_cache is not None:
                self._overlay_cache.blend_into(comp, self._alpha)
                if self._show_tip:
                    _draw_point_inplace(comp, self.state.tip_uv_c)
                layers.append(("02_rgb_overlay", cv2.cvtColor(comp, cv2.COLOR_BGRA2BGR)))

            # the drill drawn on this frame (the page's overlay may already
            # follow a newer result)
            drawn = self._drawn_out
            drill = None
            if drawn is not None:
                drill = self._new_tool_overlay()
                # as on the screen: a pose drawn marked stays marked
                drill.update(drawn.points, marked=bool(drawn.points_marked))
                drill.draw_into(comp, show_camera=True, show_xray=False)
                layers.append(("03_rgb_overlay_drill", cv2.cvtColor(comp, cv2.COLOR_BGRA2BGR)))
                if drill.has_xray_view:
                    drill.draw_into(comp, show_camera=False, show_xray=True)
                    layers.append(("04_rgb_overlay_drill_xrayview",
                                   cv2.cvtColor(comp, cv2.COLOR_BGRA2BGR)))

            record = self._snapshot_record(out_dir.name, t_click, drawn, drill)

            numbers = None
            if ev is not None:
                numbers = {"layers": 0}       # set when written
                if self._tracking_only:
                    numbers["tracking_only"] = True
                else:
                    numbers["d_x_mm"] = self._current_d_x_mm
                numbers["tracking"] = bool(self._tracking_active)
                numbers["shown"] = "pose" if drawn is not None else "live"

            # one snapshot at a time: the button comes back with the
            # writer's report (_on_snapshot_saved) - two quick clicks
            # would start two writers on the same folder in evaluation
            # mode (ev.subdir("live"))
            self.controls.btn_save.setEnabled(False)
            threading.Thread(
                target=self._write_snapshot,
                args=(out_dir, layers, record, numbers),
                name="snapshot-writer",
            ).start()
        except Exception as e:
            self.controls.btn_save.setEnabled(True)
            QMessageBox.critical(self, "Save snapshot", f"Saving failed.\n\n{e}")

    def _write_snapshot(self, out_dir: Path, layers: list, record: dict,
                        numbers: dict | None) -> None:
        """Writer thread of one snapshot: the PNG layers (cv2.imwrite runs
        without the GIL) and snapshot.json, then the outcome to the GUI
        thread. Touches nothing of the page but the signal."""
        written, error = [], ""
        try:
            for name, img in layers:
                cv2.imwrite(str(out_dir / f"{name}.png"), img)
                written.append(name)
            with open(out_dir / "snapshot.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            written.append("snapshot.json")
        except Exception as e:
            error = str(e)
        try:
            self._snapshot_saved.emit((out_dir, written, numbers, error))
        except RuntimeError:
            pass    # the page is gone; the files are written

    def _on_snapshot_saved(self, saved) -> None:
        """GUI thread: a snapshot's files are written (or failed)."""
        self.controls.btn_save.setEnabled(True)
        out_dir, written, numbers, error = saved
        try:
            if error:
                raise RuntimeError(error)
            ev = self.state.eval
            if ev is not None and numbers is not None:
                numbers["layers"] = len(written)
                ev.log("overlay_live", out_dir.name + "/", numbers)

            QMessageBox.information(
                self,
                "Save snapshot",
                f"Saved {len(written)} layer(s):\n{', '.join(written)}\n\n{out_dir}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Save snapshot", f"Saving failed.\n\n{e}")

    def _snapshot_record(self, name: str, t_click: float, drawn, drill) -> dict:
        """Everything a figure needs to rebuild the layers from 01_rgb.png
        (drawn: the result whose drill is on the frame, drill: its overlay)."""

        def arr(v):
            return None if v is None else np.asarray(v, dtype=np.float64).tolist()

        h, w = self._live_color.shape[:2]
        rec = {
            "snapshot": name,
            "frame_wh": [int(w), int(h)],
            "alpha": float(self._alpha),
            "d_x_mm": self._current_d_x_mm,
            "d_x_base_mm": self._base_d_x_mm,
            "d_x_offset_mm": int(self._dx_offset_mm),
            "H_xc": arr(self.state.H_xc),
            "K_rgb": arr(self.state.K_rgb),
            "dist_rgb": arr(self.state.dist_rgb),
            "K_xray": arr(self.state.K_xray),
            "T_xc": arr(self.state.T_xc),
            # the live warp runs through the colour lens (prepare_live_overlay
            # with K_rgb/dist_rgb); recorded so an offline re-render knows
            "warp_through_lens": True,
            "xray_image_path": self.state.xray_image_path,
            # the X-ray actually shown (xray_image_path is Cam2X's board shot)
            "xray_image_anatomy_path": self.state.xray_image_anatomy_path,
            "reference_planes_set_live": int(self._planes_set),
            "reference_plane_point_uv": arr(self.state.tip_uv_c),
            "reference_plane_point_xyz_c_mm": arr(self.state.tip_xyz_c),
            "show_tip": bool(self._show_tip),
            # the rail's switches; show_axis is the older name of the first
            "show_axis": bool(self._show_camera_view),
            "show_camera_view": bool(self._show_camera_view),
            "show_xray_view": bool(self._show_xray_view),
            "tracking_active": bool(self._tracking_active),
            # "pose": the frame a drill pose was computed on, with its tip and
            # axis; "live": the newest camera frame, no drill. The age is
            # poll -> the click (before the PNGs were written).
            "shown": "pose" if drawn is not None else "live",
            "frame_age_ms": round((t_click - self._shown_t_frame) * 1000.0, 1),
            # the colour exposure the camera ran with (the PNGs are the raw
            # frame; the display LUT of the marker exposure is not in them)
            "color_exposure": (
                {"mode": "auto"} if self._exposure is None
                else {"mode": MODE_MARKER, **self._exposure.controller.status(),
                      "display_gain": round(self._display_look.k, 3)}
            ),
            "tool": None,
        }
        if self._tracking_only:
            # camera image + drill: nothing of the X-ray overlay was drawn,
            # so none of its numbers (state may still hold some) go in
            rec["tracking_only"] = True
            for key in ("alpha", "H_xc", "K_xray", "T_xc", "warp_through_lens",
                        "xray_image_path", "xray_image_anatomy_path",
                        "reference_plane_point_uv", "reference_plane_point_xyz_c_mm"):
                rec[key] = None
            rec["show_tip"] = False
            rec["show_xray_view"] = False

        if drawn is not None:
            tool = drill.snapshot_record()
            # fused / carry / rgb / gate: tracking_overlay.ToolPoints.kind
            tool["points_kind"] = drawn.points_kind
            # drawn marked (tool["marked"]): visible_cell_fraction below the
            # threshold in force when the pose was drawn
            tool["mark_below_visible_frac"] = self._mark_below_visible_frac
            res = drawn.result
            vis_frac = float(getattr(res, "visible_cell_fraction", float("nan")))
            tool["pose"] = None if res is None else {
                "rvec": arr(res.rvec),
                "tvec": arr(res.tvec),
                "rvec_prefusion": arr(res.rvec_prefusion),
                "tvec_prefusion": arr(res.tvec_prefusion),
                "pose_source": getattr(res.pose_source, "name", str(res.pose_source)),
                "pose_converged": bool(getattr(res, "pose_converged", False)),
                "mean_reprojection_error_px": float(res.mean_reprojection_error_px),
                "num_points": int(res.num_points),
                "num_inliers": int(res.num_inliers),
                "visible_cell_fraction": vis_frac if vis_frac == vis_frac else None,
            }
            if self._tool is not None:
                tool["assets"] = {
                    "name": self._tool.name,
                    "field_path": str(self._tool.field_path),
                    "marker_json_path": str(self._tool.marker_json_path),
                    "pivot_npz_path": str(self._tool.pivot_npz_path),
                }
            rec["tool"] = tool

        return rec

    # --------------------------------------------------------
    # Reference plane (Evaluation mode)
    # --------------------------------------------------------
    #
    # One overlay-error plane = new X-ray + new pointer d_x + snapshot, with
    # the object unmoved in between. The button asks for the new X-ray
    # first, then solves the pointer on every live frame (same solve and
    # tip filter as the Pointer Tool Depth page); SPACE + confirm swaps both
    # into state, rebuilds the homographies, advances the evaluation label
    # (plane03 -> plane04) and saves <label>_depth.npz + <label>_xray.<ext>.
    # ESC or the button again cancels without touching state.

    def _on_plane_button(self) -> None:
        if self._plane_acquiring:
            self._end_plane_acquire("cancelled")
            return
        if self._tracking_only:
            # no X-ray overlay to set a plane for (the card is hidden)
            return

        if self.camera is None or not self._has_prerequisites():
            QMessageBox.information(
                self,
                "Set reference plane",
                "The live view is not running.",
            )
            return

        path = self._ask_xray_path()
        if not path:
            return
        try:
            xray = read_xray_gray_u8(path)
        except Exception as e:
            QMessageBox.critical(self, "Failed to load X-ray image", str(e))
            return

        # The drill tracker and the pointer solve do not share the frame
        # budget; tracking restarts with its button afterwards.
        if self._tracking_active:
            self._stop_tracking("paused for reference plane")

        if self._pointer is None:
            self._pointer = PointerDepthEstimator(self.FPS)
        self._pointer.reset()

        self._plane_xray = xray
        self._plane_xray_path = path
        self._plane_res = None
        self._plane_raw = None
        self._plane_d_x_mm = None
        self._plane_acquiring = True

        self.controls.btn_plane.setText("Cancel")
        self.controls.btn_track.setEnabled(False)
        self.controls.lbl_plane_status.setText(
            f"X-ray: {Path(path).name}\npointer: SPACE accept · ESC cancel"
        )
        self.setFocus()

    def _ask_xray_path(self) -> str | None:
        # Start where the last X-ray came from (the USB-stick copy folder).
        last = self.state.xray_image_anatomy_path or self.state.xray_image_path
        start_dir = str(Path(last).parent) if last else ""

        dlg = QFileDialog(self.window(), "Select X-ray image of the new plane", start_dir)
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter(
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.ima);;All Files (*)"
        )
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        if dlg.exec() != QFileDialog.Accepted:
            return None
        files = dlg.selectedFiles()
        return files[0] if files else None

    def _plane_step(self, img_bgra: np.ndarray) -> None:
        """Pointer solve on the current frame (BGRA goes straight into the
        ArUco detection). Only a pose of THIS frame can be accepted."""
        try:
            raw, res = self._pointer.solve(img_bgra, self.state.K_rgb, self.state.dist_rgb)
            d_x = float(extract_depth(T_xc=self.state.T_xc, T_tc=res.T_4x4).d_x_mm)
        except Exception:
            self._plane_res = None
            self._plane_raw = None
            self._plane_d_x_mm = None
            return

        self._plane_raw = raw
        self._plane_res = res
        self._plane_d_x_mm = d_x

        # The camera frame is a zero-copy view that dies with the next
        # poll; the accepted capture needs its own copy.
        h, w = img_bgra.shape[:2]
        if self._plane_rgb is None or self._plane_rgb.shape[:2] != (h, w):
            self._plane_rgb = np.empty((h, w, 3), dtype=np.uint8)
        cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR, dst=self._plane_rgb)

    def _draw_plane_acquire(self, frame_bgra: np.ndarray) -> None:
        res = self._plane_res
        if res is None:
            lines = ["REFERENCE PLANE: pointer NOT FOUND"]
            color = (0, 0, 255, 255)
        else:
            pts = getattr(res, "image_points_uv", None)
            if pts is not None:
                for u, v in np.asarray(pts, dtype=np.float64).reshape(-1, 2):
                    if np.isfinite(u) and np.isfinite(v):
                        cv2.circle(frame_bgra, (int(round(u)), int(round(v))), 3,
                                   (0, 255, 255, 255), -1, cv2.LINE_AA)
            _draw_point_inplace(frame_bgra, res.tip_uv, color=(0, 255, 0, 255))
            old = self._base_d_x_mm
            lines = [
                "REFERENCE PLANE: FOUND (SPACE accept, ESC cancel)",
                f"new d_x {self._plane_d_x_mm:.1f} mm"
                + ("" if old is None else f"   (current {old:.1f} mm)"),
            ]
            color = (0, 255, 0, 255)

        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        x, y = 24, 24
        line_h = 40
        width = max(cv2.getTextSize(s, font, scale, thick)[0][0] for s in lines)
        cv2.rectangle(frame_bgra, (x - 12, y - 8),
                      (x + width + 12, y + line_h * len(lines) + 4),
                      (0, 0, 0, 255), -1)
        for i, s in enumerate(lines):
            cv2.putText(frame_bgra, s, (x, y + 30 + i * line_h), font, scale,
                        color, thick, cv2.LINE_AA)

    def _confirm_plane(self) -> None:
        res, raw, d_new = self._plane_res, self._plane_raw, self._plane_d_x_mm
        if res is None or d_new is None or self._plane_rgb is None:
            return
        rgb = self._plane_rgb.copy()

        # Freeze the view on the frame the pose came from while asking.
        self._stop_timer()
        old = self._base_d_x_mm
        delta = "" if old is None else f"\nprevious: {old:.1f} mm  (Δ {d_new - old:+.1f} mm)"
        ans = QMessageBox.question(
            self,
            "Set reference plane",
            f"New reference plane\n\n"
            f"d_x: {d_new:.1f} mm{delta}\n"
            f"reprojection: {float(getattr(res, 'reproj_mean_px', np.nan)):.2f} px\n"
            f"X-ray: {Path(self._plane_xray_path).name}\n\n"
            "Use this plane?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ans == QMessageBox.Yes:
            try:
                self._apply_plane(res, raw, rgb, d_new)
            except Exception as e:
                QMessageBox.critical(self, "Set reference plane", f"Failed.\n\n{e}")
        else:
            # keep acquiring; the next found frame can be accepted
            self._plane_res = None
        self._start_timer()
        self.setFocus()

    def _apply_plane(self, res, raw, rgb_bgr: np.ndarray, d_x_mm: float) -> None:
        ev = self.state.eval
        if ev is not None:
            ev.set_label(ev.next_label(ev.label))

        self.state.d_x = float(d_x_mm)
        self.state.tip_uv_c = np.asarray(res.tip_uv, dtype=np.float64).copy()
        self.state.tip_xyz_c = np.asarray(res.tip_point_camera_mm, dtype=np.float64).copy()
        self.state.T_tc = np.asarray(res.T_4x4, dtype=np.float64).copy()
        self.state.xray_image_anatomy = self._plane_xray
        self.state.xray_image_anatomy_path = self._plane_xray_path

        self._load_xray_from_state()
        self._initialize_dx()
        self._precompute_H_xc_lookup()
        self._invalidate_overlay_cache()
        self._planes_set += 1

        if ev is not None:
            arrays, summary = depth_capture_arrays(
                rgb_image=rgb_bgr,
                K_rgb=self.state.K_rgb,
                dist_rgb=self.state.dist_rgb,
                T_xc=self.state.T_xc,
                result=res,
                raw_result=raw,
                d_x_mm=d_x_mm,
            )
            try:
                ev.save_npz("depth", arrays, page="overlay_live_plane", summary=summary)
            except Exception as e:
                print(f"[WARN] Could not save pointer depth capture: {e}")
            try:
                ev.copy_file("xray", self._plane_xray_path, page="overlay_live_plane")
            except Exception as e:
                print(f"[WARN] Could not copy X-ray image: {e}")

        label = "" if ev is None else f"{ev.sanitize_label(ev.label)}: "
        self._end_plane_acquire(f"{label}d_x {d_x_mm:.1f} mm — save snapshot")

    def _end_plane_acquire(self, status: str) -> None:
        self._plane_acquiring = False
        self._plane_res = None
        self._plane_raw = None
        self._plane_d_x_mm = None
        self._plane_xray = None
        self._plane_xray_path = None
        self.controls.btn_plane.setText("Set reference plane")
        self.controls.btn_track.setEnabled(True)
        self.controls.lbl_plane_status.setText(status)

    def keyPressEvent(self, event) -> None:
        if self._plane_acquiring:
            if event.key() == Qt.Key_Space:
                self._confirm_plane()
                event.accept()
                return
            if event.key() == Qt.Key_Escape:
                self._end_plane_acquire("cancelled")
                event.accept()
                return
        super().keyPressEvent(event)

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def on_enter(self) -> None:
        # Tool tracking is optional: without a loaded tool the page behaves
        # exactly as before (no IR streams, no tracker, no rail card).
        self._tool = self.state.tool_assets
        # The tracking test with a tool: camera image + drill only (no
        # X-ray blend, no homographies, no reference plane).
        self._tracking_only = self._is_tracking_only()
        self.controls.set_tracking_visible(self._tool is not None)
        self.controls.set_tracking_only(self._tracking_only)
        self.controls.btn_track.setText("Start tracking")
        self.controls.lbl_track_status.setText("idle")
        self.controls.set_plane_visible(self.state.eval is not None and not self._tracking_only)
        self._end_plane_acquire("")
        self._planes_set = 0
        # a new camera counts its frames from 0 again
        self._shown_out = None
        self._drawn_out = None
        self._shown_index = -1
        self._withheld = None
        self._t_withheld = None
        self._screen_view = False

        self._load_xray_from_state()
        self._initialize_dx()

        if not self._tracking_only and not self._has_prerequisites():
            QMessageBox.information(
                self,
                "Live Overlay",
                "Missing prerequisites from previous steps.",
            )
            return

        if not self._tracking_only and self._xray_gray_u8 is None:
            QMessageBox.information(
                self,
                "Live Overlay",
                "No anatomical X-ray image available in state.",
            )
            return

        try:
            if self._tracking_only:
                self._clear_xray_layer()
            else:
                self._precompute_H_xc_lookup()
                self._invalidate_overlay_cache()

            ev = self.state.eval
            self._profile = (
                _FrameProfile(self.FPS, None if ev is None else ev.dir / PROFILE_LOG_NAME)
                if (PROFILE_LIVE and ev is not None) else None
            )
            if self._profile is not None:
                # the session header follows the camera start (it names the
                # colour options read back); the first window counts from here
                self._profile.reset()
            self._frame_log = None if ev is None else _EvalFrameLog(ev, self.FPS)
            if self._profile is not None and self._frame_log is not None:
                self._profile.frame_sink = self._frame_log.add_display
            if self._tracking_only and ev is not None:
                # which drill the tracking test runs with
                ir_path = getattr(self.state, "ir_calibration_path", None)
                ev.log("overlay_live", "-", {
                    "tracking_only": True,
                    "tool": self._tool.name,
                    "field": Path(self._tool.field_path).name,
                    "markers": Path(self._tool.marker_json_path).name,
                    "pivot": Path(self._tool.pivot_npz_path).name,
                    "ir_calibration": Path(ir_path).name if ir_path else "off",
                }, tagged=False)

            self._start_realsense()
            if self._profile is not None:
                if self._tracking_only:
                    layer = "no x-ray (tracking only)"
                else:
                    h, w = self._xray_gray_u8.shape[:2]
                    layer = (
                        f"x-ray {w}x{h}, "
                        f"{len(self._H_xc_lookup)} precomputed homographies"
                    )
                self._profile.emit(
                    f"[live] === session start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"=== colour {self.COLOR_SIZE[0]}x{self.COLOR_SIZE[1]} @ "
                    f"{self.FPS} fps, {layer}, "
                    f"drill tracking in a worker thread; colour sensor "
                    f"{format_realsense_color(self._camera_color) or 'not read back'}"
                )
            self._start_timer()
            self.setFocus()

        except Exception as e:
            self._stop_timer()
            self._stop_realsense()
            QMessageBox.critical(
                self,
                "Camera",
                f"Could not open RealSense camera.\n\n{e}",
            )

    def on_leave(self) -> None:
        self._end_plane_acquire("")
        if self._tracking_active:
            # joins the tracking thread, so the tracker is free to be
            # dropped below
            self._stop_tracking("idle")
        self._tracker = None
        self._tool_overlay = None
        self._shown_out = None
        self._withheld = None              # its view ends with the camera
        self._stop_timer()
        self._stop_realsense()
        log, self._frame_log = self._frame_log, None
        if log is not None:
            # the frames shown since the last tracking stop
            if self._profile is not None:
                self._profile.frame_sink = None
            log.flush("page left", tracking=True)

    def closeEvent(self, event) -> None:
        self.on_leave()
        super().closeEvent(event)