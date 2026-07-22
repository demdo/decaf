"""Generic live runner for HydraMarker tracking.

The runner owns orchestration only: camera frames, tracker lifetime, app state,
keyboard controls, callbacks, and optional JSONL logging. It does not implement
detector, pose, or debug-overlay drawing logic.
"""

from __future__ import annotations

from dataclasses import replace
import gc
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np


def _ensure_src_on_path() -> None:
    src_root = Path(__file__).resolve().parents[2]
    src = str(src_root)
    if src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()

from tracking.hydramarker import tracker_log
from tracking.hydramarker.camera_setup import CameraSource, create_camera_source
from tracking.hydramarker.config import LiveTrackerConfig, LoggingConfig, TrackerConfig
from tracking.hydramarker.tracker import HydraTracker


APP_IDLE = "IDLE"
APP_ACQUIRE = "ACQUIRE"
APP_PROVISIONAL = "PROVISIONAL"
APP_TRACKING = "TRACKING"

ACQUIRE_TIMEOUT_FRAMES = 90
PROVISIONAL_MIN_CORNERS = 6
PROVISIONAL_STALE_TIMEOUT_FRAMES = 30
PROVISIONAL_TOTAL_TIMEOUT_FRAMES = 180
TRACKING_STALE_TO_IDLE_FRAMES = 45

# IDLE is no longer a dead state: every N-th frame runs one detection probe
# (~5 ms each 0.5 s).  Sees the probe a plausible marker fragment, the app
# re-arms acquisition automatically — no manual 's' needed after a timeout.
# (Fail-run 2026-07-13: 232 frames sat in IDLE with detection skipped while
# the tool was held perfectly; forced-detection replay locked after ~41
# frames and tracked 191/232.)
IDLE_REACQUIRE_PROBE_INTERVAL_FRAMES = 15

LOG_FRAME_DETAILS_ENV = "HYDRAMARKER_LOG_FRAME_DETAILS"
LOG_POSE_CANDIDATES_ENV = "HYDRAMARKER_LOG_POSE_CANDIDATES"

# preview cap (Hz->interval): display is decoupled from the 30 fps processing
# loop; imshow+waitKey on Windows cost 15-25 ms and previously throttled the
# whole pipeline to ~19 fps
RENDER_MIN_INTERVAL_S = 0.05


class IrExposureController:
    """Continuous marker-tuned IR exposure control (replaces the old one-shot
    freeze). Driven ONLY by tracking frames: ir_saturated_frac exists solely
    while a pose + IR reference are live, so idle/lost phases hold the last
    exposure. Two phases:

    TUNING (right after lock, aligned with the pose warmup): step DOWN fast
    (every 4 tracking frames) until the marker saturation is ~zero. No active
    brightening toward a saturation band -- the old target band [0.05, 0.18)
    tolerated up to 18% clipped corners and their sub-threshold halos distort
    the MAP warp measurement (21.07 run: frozen at the band edge from the
    startup pose, working pose then sat 0.3-0.4 all run, fit RMS ~1.4).

    HOLD (settled): windowed-median watchdog. Persistent saturation (a pose-
    dependent specular stripe entering the marker mid-run) steps the exposure
    down; long clean stretches recover it slowly toward the configured base,
    never above. Every applied change invalidates the IR reference library
    (templates are exposure-dependent) via the caller.
    """

    def __init__(self, base_exposure_us: float):
        self.base = float(base_exposure_us)
        self.exposure = float(base_exposure_us)
        self.state = "tuning"
        self.min_exposure = 500.0
        self.tune_interval = 6        # tracking frames between tuning steps
                                      # (exposure applies asynchronously ~2-3
                                      # frames later; 6 avoids double-stepping
                                      # on stale saturation samples)
        self.tune_target = 0.02       # marker saturation considered "clean"
        self.tune_settle_samples = 6  # consecutive clean samples to settle
        self.hold_window = 45         # sat samples for the runtime median
        self.hold_down_med = 0.03     # persistent saturation -> step down
        self.hold_down_cooldown = 30  # tracking frames between down steps
        self.hold_up_clean = 120      # clean tracking frames before recovery
        self.down_factor = 0.7
        self.up_factor = 1.3          # asymmetric: down fast, recover moderate
                                      # (a wrong up-step is corrected within
                                      # ~30 frames by the down watchdog)
        self._last_change = -10**9
        self._clean_streak = 0
        self._win: list[float] = []

    def update(self, frame_idx: int, sat: float) -> Optional[float]:
        """Feed one frame's ir_saturated_frac (< 0 = not measured). Returns a
        new exposure value when the camera should be re-programmed."""
        if sat is None or sat < 0.0:
            return None
        if self.state == "tuning":
            if sat <= self.tune_target:
                self._clean_streak += 1
                if self._clean_streak >= self.tune_settle_samples:
                    self.state = "hold"
                    self._win.clear()
                    self._clean_streak = 0
                    return self.exposure   # settle signal: re-enroll refs now
                return None
            self._clean_streak = 0
            if (frame_idx - self._last_change) < self.tune_interval:
                return None
            if self.exposure <= self.min_exposure:
                self.state = "hold"        # floor reached: watchdog takes over
                self._win.clear()
                return self.exposure
            self.exposure = max(self.min_exposure,
                                self.exposure * self.down_factor)
            self._last_change = frame_idx
            return self.exposure

        # HOLD: windowed watchdog on tracking frames only.
        self._win.append(sat)
        if len(self._win) > self.hold_window:
            self._win.pop(0)
        self._clean_streak = self._clean_streak + 1 if sat <= 0.0 else 0
        if (frame_idx - self._last_change) < self.hold_down_cooldown:
            return None
        if len(self._win) >= self.hold_window:
            med = sorted(self._win)[len(self._win) // 2]
            if med > self.hold_down_med and self.exposure > self.min_exposure:
                self.exposure = max(self.min_exposure,
                                    self.exposure * self.down_factor)
                self._last_change = frame_idx
                self._win.clear()
                return self.exposure
        if (self._clean_streak >= self.hold_up_clean
                and self.exposure < self.base):
            self.exposure = min(self.base, self.exposure * self.up_factor)
            self._last_change = frame_idx
            self._clean_streak = 0
            self._win.clear()
            return self.exposure
        return None


TrackerRenderCallback = Callable[
    [np.ndarray, Any, HydraTracker, int, bool],
    np.ndarray,
]
TrackerKeyCallback = Callable[[int, HydraTracker, Any, int], bool]


def choose_file_qt(title: str, file_filter: str) -> Path:
    from PySide6.QtWidgets import QApplication, QFileDialog

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(None, title, "", file_filter)
    if not path:
        raise RuntimeError(f"No file selected: {title}")

    return Path(path)


class NullTrackerLogger:
    """No-op live logger used when logging is disabled."""

    def is_active(self) -> bool:
        return False

    def current_path(self) -> Optional[Path]:
        return None

    def current_run_id(self) -> str:
        return ""

    def open(self, field_path: Path, marker_json_path: Path, tracker: HydraTracker) -> None:
        return None

    def log_frame(
        self,
        frame_idx: int,
        result,
        wall_ms: float,
        tracker: HydraTracker,
        draw_ms: float,
    ) -> None:
        return None

    def close(self) -> Optional[Path]:
        return None


class JsonlTrackerLogger:
    """JSONL logger adapter around ``tracker_log``."""

    def __init__(self, config: LoggingConfig) -> None:
        self.config = config
        tracker_log.set_run_log_dir(Path(config.output_dir))
        os.environ[LOG_FRAME_DETAILS_ENV] = "1" if config.frame_details else "0"
        os.environ[LOG_POSE_CANDIDATES_ENV] = "1" if config.pose_candidates else "0"

    def is_active(self) -> bool:
        return tracker_log.is_active()

    def current_path(self) -> Optional[Path]:
        return tracker_log.current_path()

    def current_run_id(self) -> str:
        return tracker_log.current_run_id()

    def open(self, field_path: Path, marker_json_path: Path, tracker: HydraTracker) -> None:
        tracker_log.log_open(field_path, marker_json_path, tracker)

    def log_frame(
        self,
        frame_idx: int,
        result,
        wall_ms: float,
        tracker: HydraTracker,
        draw_ms: float,
    ) -> None:
        tracker_log.log_frame(frame_idx, result, wall_ms, tracker, draw_ms)

    def close(self) -> Optional[Path]:
        path = tracker_log.current_path()
        tracker_log.log_close()
        return None if path is None else Path(path)


def create_tracker_logger(config: LoggingConfig):
    config.validate_common()
    if not config.active:
        return NullTrackerLogger()
    return JsonlTrackerLogger(config)


def make_live_tracker_config() -> TrackerConfig:
    """Return the tuned live config used by the legacy RealSense runner."""
    return TrackerConfig(
        min_points=6,
        min_inliers=5,
        max_mean_reprojection_error_px=4.0,
        max_max_reprojection_error_px=15.0,
        max_lost_frames=8,
        max_translation_jump_mm=40.0,
        max_rotation_jump_deg=45.0,
        rotation_gate_scale_per_lost_frame=8.0,
        rotation_gate_max_deg=90.0,
        decode_update_min_visual_corners=12,
        decode_update_min_distinct_rows=3,
        decode_update_min_distinct_cols=3,
        pnp_ransac_iterations=500,
        pnp_ransac_reprojection_px=3.0,
        pnp_ransac_confidence=0.99,
        use_pose_prior=True,
        pnp_direct_refine_method="vvs",
        # Pose output filter: ADAPTIVE process noise (PoseOutputFilter). While
        # the tool is still, Q collapses to a tiny floor so the filter averages
        # many frames -> static tip jitter 0.10 -> 0.034 mm (validated offline
        # on raw pose logs, calib/output/080726/filter_validation.png); it opens
        # up the instant motion is measured so the pose follows without lag
        # (rotation/translation stay lag-free, all cases < 1 mm tip error). The
        # anisotropic measurement noise sigma^2 (J^T J)^-1 still smooths the weak
        # observability mode. The fixed pose_kf_q_* values are the non-adaptive
        # fallback and unused while pose_kf_adaptive is True.
        pose_kf_adaptive=True,
        corr_min_votes=2,
        corr_discard_conflicts=True,
        corr_require_detection_stable=False,
        corr_enable_dominant_rotation_filter=True,
        corr_min_rotation_support=2,
        corr_min_rotation_support_ratio=0.55,
        checker_refresh_interval_frames=3,
        checker_tracking_recovery_stable_interval_frames=9,
        checker_max_undecodeable_tracking_frames=12,
        checker_max_low_fresh_correspondence_frames=12,
        enable_pose_propagation=False,
        enable_temporal_correspondence_persistence=True,
        persistence_use_pose_projection=True,
        persistence_projection_max_reproj_px=12.0,
        persistence_projection_max_pose_error_px=2.5,
        fast_persistent_dense_refine_enabled=True,
        fast_persistent_dense_min_points=24,
        fast_persistent_dense_match_max_px=3.0,
        fast_persistent_dense_min_second_best_margin_px=2.0,
        fast_persistent_dense_max_median_px=1.2,
        fast_persistent_dense_max_p90_px=2.5,
        fast_persistent_dense_min_image_coverage=0.35,
        fast_persistent_dense_min_object_span_mm=12.0,
        fast_persistent_dense_min_distinct_rows=2,
        fast_persistent_dense_min_distinct_cols=2,
        fast_persistent_dense_pose_solver="direct_prior",
        dot_use_temporal_smoothing=False,
        dot_commit_frames=1,
        dot_revoke_frames=5,
        persistence_max_frames=8,
    )


def make_tracker(field_path, marker_json_path, K, dist, config: TrackerConfig | None = None) -> HydraTracker:
    return HydraTracker(
        field_path=str(field_path),
        marker_json_path=str(marker_json_path),
        K=K,
        dist_coeffs=dist,
        config=config or make_live_tracker_config(),
    )


def log_console(frame_idx: int, result, tracker, *, console_prefix: str, force: bool = False) -> None:
    if not force and (not result.success or frame_idx % 30 != 0):
        return

    failure_stage, failure_reason = tracker_log.classify_failure(result)
    debug_counters = getattr(result, "debug_counters", {}) or {}
    print(
        console_prefix,
        f"frame={frame_idx}",
        f"mode={result.mode.value}",
        f"success={result.success}",
        f"src={getattr(getattr(result, 'pose_source', None), 'value', 'none')}",
        f"fast={int(float(debug_counters.get('fast_attempted', 0.0)) > 0.0)}/"
        f"{int(float(debug_counters.get('fast_success', 0.0)) > 0.0)}:"
        f"{int(float(debug_counters.get('fast_matches', 0.0)))}",
        f"stage={failure_stage}",
        f"reason={failure_reason}",
        f"msg={result.message}",
        f"det={len(getattr(result, 'detection_corners', []))}",
        f"pose={'Y' if result.rvec is not None and result.tvec is not None else 'N'}",
        f"vis={len(result.corners)}",
        f"pts={result.num_points}",
        f"inl={result.num_inliers}",
        f"mean={result.mean_reprojection_error_px:.3f}",
        f"pers={len(tracker._persistent_corners)}",
    )


def run_tracker(
    *,
    config: Optional[LiveTrackerConfig] = None,
    field_path: Optional[Path] = None,
    marker_json_path: Optional[Path] = None,
    calibration_path: Optional[Path] = None,
    window_name: str = "HydraTracker",
    console_prefix: str = "[run_tracker]",
    after_camera_ready: Optional[
        Callable[[CameraSource, np.ndarray, np.ndarray], bool]
    ] = None,
    after_tracker_created: Optional[Callable[[HydraTracker], None]] = None,
    on_key: Optional[TrackerKeyCallback] = None,
    on_space_key: Optional[Callable[[HydraTracker, Any, int], bool]] = None,
    on_log_open: Optional[Callable[[Path, Path, HydraTracker], None]] = None,
    on_log_close: Optional[Callable[[Path], None]] = None,
    on_frame: Optional[Callable[[int, np.ndarray, Any, bool], None]] = None,
    render_frame: Optional[TrackerRenderCallback] = None,
    stop_after_log_close: bool = False,
    quit_on_q: bool = True,
    final_cleanup: Optional[Callable[[], None]] = None,
) -> Optional[Path]:
    """Run the generic live HydraMarker tracker loop."""
    live_config = config or LiveTrackerConfig(tracker=make_live_tracker_config())
    live_config.validate_common()

    if field_path is None:
        field_path = choose_file_qt("Select HydraMarker .field file", "HydraMarker field (*.field)")
    if marker_json_path is None:
        marker_json_path = choose_file_qt("Select marker .json file", "Marker JSON (*.json)")
    if calibration_path is None:
        calibration_path = live_config.camera.calibration_path_obj()
    if calibration_path is None:
        calibration_path = choose_file_qt("Select camera calibration .npz", "NPZ files (*.npz)")

    camera_config = replace(
        live_config.camera,
        calibration_path=str(Path(calibration_path).expanduser()),
    )
    camera = create_camera_source(camera_config)
    logger = create_tracker_logger(live_config.logging)
    recorded_log_path: Optional[Path] = None

    try:
        camera.start()
        if camera.K is None or camera.dist_coeffs is None:
            raise RuntimeError("Live tracking requires a camera calibration NPZ.")

        K = np.asarray(camera.K, dtype=np.float64).reshape(3, 3)
        dist = np.asarray(camera.dist_coeffs, dtype=np.float64).reshape(-1, 1)
        tracker_log.set_camera_intrinsics_info(camera.metadata())

        if after_camera_ready is not None and not after_camera_ready(camera, K, dist):
            return None

        tracker = make_tracker(field_path, marker_json_path, K, dist, live_config.tracker)
        # arm the IR pose refinement stage (design B) when it is enabled in
        # the tracker config AND the rig calibration exists; the third gate
        # (the IR frame pair) comes from the camera per frame below.
        if getattr(live_config.tracker, "ir_refine_enabled", False):
            ir_calib = (
                Path(__file__).resolve().parent
                / "data" / "realsense" / "realsense_ir_calibration.npz"
            )
            if ir_calib.exists():
                tracker.set_ir_calibration(ir_calib)
                print(f"{console_prefix} IR pose refinement armed ({ir_calib.name})")
            else:
                print(f"{console_prefix} ir_refine_enabled but calibration "
                      f"missing: {ir_calib} - stage stays inert")
        if after_tracker_created is not None:
            after_tracker_created(tracker)

        if live_config.show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_idx = 0
        _last_render_t = [0.0]
        _fps_win: list[float] = []
        from collections import deque
        _fps_times: deque = deque(maxlen=30)   # rolling window for the overlay
        last_mode: Optional[str] = None
        last_success: Optional[bool] = None
        last_message: Optional[str] = None
        app_state = APP_ACQUIRE if not live_config.start_tracking_manually else APP_IDLE
        acquire_start_frame = 0
        provisional_start_frame = 0
        last_candidate_frame = 0
        stale_pose_frames = 0
        last_result = None

        def enter_idle(reason: str) -> None:
            nonlocal app_state, acquire_start_frame, provisional_start_frame
            nonlocal last_candidate_frame, stale_pose_frames
            nonlocal last_mode, last_success, last_message
            tracker.reset()
            app_state = APP_IDLE
            acquire_start_frame = 0
            provisional_start_frame = 0
            last_candidate_frame = 0
            stale_pose_frames = 0
            last_mode = None
            last_success = None
            last_message = None
            print(f"{console_prefix} idle ({reason})")

        def start_acquire(*, manual: bool = False) -> None:
            nonlocal app_state, acquire_start_frame, provisional_start_frame
            nonlocal last_candidate_frame, stale_pose_frames
            nonlocal last_mode, last_success, last_message
            tracker.reset()
            app_state = APP_ACQUIRE
            acquire_start_frame = frame_idx
            provisional_start_frame = 0
            last_candidate_frame = 0
            stale_pose_frames = 0
            last_mode = None
            last_success = None
            last_message = None
            reason = "manual" if manual else "requested"
            print(f"{console_prefix} acquire started ({reason})")

        if (
            isinstance(logger, JsonlTrackerLogger)
            and str(live_config.logging.start_mode).strip().lower() == "auto"
        ):
            logger.open(Path(field_path), Path(marker_json_path), tracker)
            if on_log_open is not None:
                on_log_open(Path(field_path), Path(marker_json_path), tracker)

        # Frame-time jitter: Python's automatic cyclic GC fires mid-frame at
        # unpredictable moments and adds multi-ms pauses (proven: the >27ms
        # wall spikes land on DIFFERENT frames across bit-identical replays of
        # the same session = scheduler/GC jitter, not tracker work). The
        # per-frame objects (corners, dicts, arrays) are non-cyclic and freed
        # immediately by reference counting, so disabling automatic collection
        # does NOT leak them; only genuine reference-cycle cleanup is deferred.
        # gc.freeze() moves all setup objects out of the scan set; a periodic
        # manual collect below bounds any cycle buildup at a controlled moment.
        # Restored in the finally block. Behaviour-neutral: GC never affects a
        # computed value, only WHEN memory is reclaimed.
        gc.collect()
        gc.freeze()
        gc.disable()
        _gc_collect_interval = 900   # ~30 s at 30 fps; one scheduled collect

        # ---- continuous marker-tuned IR exposure control ----
        # The IR fusion reports the fraction of MARKER corners clipped in the
        # IR pair (ir_saturated_frac, computed for free by the saturation
        # gate) -- a tracking-period signal by construction. The controller
        # tunes fast during warmup and keeps a slow watchdog afterwards, so a
        # pose-dependent specular reflex entering the marker MID-RUN re-tunes
        # instead of staying frozen (21.07: frozen at the startup pose, the
        # working pose then ran at sat 0.3-0.4 for the whole session). Every
        # applied change drops the IR reference library: the warp templates
        # are exposure-dependent.
        _ir_exp_on = (getattr(live_config.tracker, "ir_refine_enabled", False)
                      and hasattr(camera, "set_ir_exposure"))
        _ir_exp_ctrl = None
        # set_ir_exposure is a blocking UVC control transfer (~100 ms measured
        # on the first in-loop call, fb2 frame 3386: tracker 13.6 ms but wall
        # 133 ms). Loop-time changes therefore run on a worker thread — the
        # new exposure takes effect a frame later anyway. _ir_exp_pending
        # holds at most the newest requested value until the worker is free.
        _ir_exp_pending = None
        _ir_exp_worker = None
        if _ir_exp_on:
            _base_us = float(getattr(camera_config, "realsense_ir_exposure_us", None) or 8000.0)
            if camera.set_ir_exposure(_base_us):   # pre-loop: blocking is fine
                _ir_exp_ctrl = IrExposureController(_base_us)
                print(f"{console_prefix} IR exposure control: start {_base_us:.0f} us")

        while True:
            frame = camera.read()
            if frame is None:
                continue

            frame_idx = int(frame.frame_index)
            raw_frame = frame.image_bgr

            idle_probe = (
                app_state == APP_IDLE
                and getattr(live_config, "idle_auto_reacquire", False)
                and frame_idx % IDLE_REACQUIRE_PROBE_INTERVAL_FRAMES == 0
            )

            t0 = time.perf_counter()
            result = tracker.process_frame(
                raw_frame,
                run_detection=(app_state != APP_IDLE) or idle_probe,
                ir_left=getattr(frame, "ir_left", None),
                ir_right=getattr(frame, "ir_right", None),
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            last_result = result

            # continuous IR exposure control step (see setup above). Any
            # applied change re-programs the camera AND drops the exposure-
            # dependent IR reference templates so they re-enroll at the new
            # exposure through the normal gates.
            if _ir_exp_ctrl is not None:
                _sat = (result.timings_ms or {}).get("ir_saturated_frac", -1.0)
                _was_tuning = _ir_exp_ctrl.state == "tuning"
                _new_exp = _ir_exp_ctrl.update(frame_idx, _sat)
                if _new_exp is not None:
                    _ir_exp_pending = _new_exp   # newest value wins
                    # References are dropped ONLY during the startup tuning
                    # phase (incl. the settle transition): templates enrolled
                    # mid-ramp are worthless. Runtime HOLD steps must NOT
                    # reset the library - fb_rl2 21.07: a mid-run reset
                    # re-bootstrapped the references from the current
                    # (already biased) pose, the MAP then self-confirmed a
                    # +3.5 mm error at rms 0.11. ZNCC is intensity-normalised,
                    # so existing templates survive a moderate exposure step;
                    # fresh tiles replace them through the normal gates.
                    if _was_tuning and hasattr(tracker, "reset_ir_references"):
                        tracker.reset_ir_references()
                        print(f"{console_prefix} IR exposure tuning "
                              f"-> {_new_exp:.0f} us (marker sat "
                              f"{100*_sat:.0f}%), IR references re-enroll")
                    else:
                        print(f"{console_prefix} IR exposure hold "
                              f"-> {_new_exp:.0f} us (marker sat "
                              f"{100*_sat:.0f}%), references kept")
                if _ir_exp_pending is not None and (
                        _ir_exp_worker is None or not _ir_exp_worker.is_alive()):
                    _ir_exp_worker = threading.Thread(
                        target=camera.set_ir_exposure,
                        args=(_ir_exp_pending,),
                        daemon=True,
                    )
                    _ir_exp_pending = None
                    _ir_exp_worker.start()

            det_count = len(getattr(result, "detection_corners", []))
            if idle_probe and det_count >= PROVISIONAL_MIN_CORNERS:
                # marker sighted from IDLE: re-arm WITHOUT tracker.reset() so
                # the probe's corners immediately count toward acquisition
                app_state = APP_ACQUIRE
                acquire_start_frame = frame_idx
                provisional_start_frame = 0
                last_candidate_frame = 0
                stale_pose_frames = 0
                print(f"{console_prefix} auto re-acquire "
                      f"(idle probe det={det_count})")
            if app_state in (APP_ACQUIRE, APP_PROVISIONAL):
                active_frames = max(0, frame_idx - acquire_start_frame + 1)
                if tracker_log.has_fresh_pose(result):
                    app_state = APP_TRACKING
                    stale_pose_frames = 0
                    print(f"{console_prefix} tracking locked")
                elif det_count >= PROVISIONAL_MIN_CORNERS:
                    last_candidate_frame = frame_idx
                    if app_state != APP_PROVISIONAL:
                        app_state = APP_PROVISIONAL
                        provisional_start_frame = frame_idx
                        print(f"{console_prefix} provisional fragment det={det_count}")
                elif (
                    app_state == APP_ACQUIRE
                    and active_frames >= ACQUIRE_TIMEOUT_FRAMES
                ):
                    enter_idle("acquire timeout")
                elif app_state == APP_PROVISIONAL:
                    no_candidate_frames = (
                        frame_idx - last_candidate_frame
                        if last_candidate_frame > 0
                        else active_frames
                    )
                    provisional_frames = (
                        frame_idx - provisional_start_frame + 1
                        if provisional_start_frame > 0
                        else active_frames
                    )
                    if no_candidate_frames >= PROVISIONAL_STALE_TIMEOUT_FRAMES:
                        enter_idle("provisional stale")
                    elif provisional_frames >= PROVISIONAL_TOTAL_TIMEOUT_FRAMES:
                        enter_idle("provisional timeout")
            elif app_state == APP_TRACKING:
                if tracker_log.has_fresh_pose(result):
                    stale_pose_frames = 0
                else:
                    stale_pose_frames += 1
                    if stale_pose_frames >= TRACKING_STALE_TO_IDLE_FRAMES:
                        enter_idle("tracking lost")

            mode_changed = last_mode != result.mode.value
            success_changed = last_success != bool(result.success)
            message_changed = last_message != result.message
            force_log = (
                mode_changed
                or success_changed
                or message_changed
                or (not result.success and app_state != APP_IDLE)
            )
            log_console(frame_idx, result, tracker, console_prefix=console_prefix, force=force_log)

            last_mode = result.mode.value
            last_success = bool(result.success)
            last_message = result.message

            # Display is DECOUPLED from processing: rendering + imshow + waitKey
            # cost 15-25 ms on Windows and silently throttled the camera loop to
            # ~19 fps (frames dropped inside wait_for_frames). Every frame is
            # still TRACKED and LOGGED; only the on-screen preview is capped at
            # ~20 Hz, which keeps the loop under the 33 ms budget for true 30 fps.
            draw_t0 = time.perf_counter()
            now_render = time.perf_counter()
            do_render = (
                live_config.show_window
                and (now_render - _last_render_t[0]) >= RENDER_MIN_INTERVAL_S
            )
            _fps_times.append(now_render)
            if do_render:
                _last_render_t[0] = now_render
                if render_frame is None:
                    vis = raw_frame.copy()
                else:
                    vis = render_frame(
                        raw_frame.copy(),
                        result,
                        tracker,
                        frame_idx,
                        logger.is_active(),
                    )
                # live fps overlay (top right): green = full rate, yellow =
                # borderline, red = the camera or the loop is throttling
                if len(_fps_times) >= 2:
                    span = _fps_times[-1] - _fps_times[0]
                    fps_now = (len(_fps_times) - 1) / span if span > 0 else 0.0
                    color = ((0, 220, 0) if fps_now >= 27.0
                             else (0, 220, 220) if fps_now >= 22.0
                             else (0, 0, 255))
                    cv2.putText(
                        vis,
                        f"{fps_now:4.1f} fps | tracker {wall_ms:4.0f} ms",
                        (vis.shape[1] - 460, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                # app state (below fps): the dead-IDLE trap was invisible —
                # the user held the tool while no detection ran at all
                state_color = ((0, 220, 0) if app_state == APP_TRACKING
                               else (0, 0, 255) if app_state == APP_IDLE
                               else (0, 220, 220))
                cv2.putText(
                    vis,
                    app_state,
                    (vis.shape[1] - 460, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2, cv2.LINE_AA)
                cv2.imshow(window_name, vis)
            draw_ms = (time.perf_counter() - draw_t0) * 1000.0

            # Expose the loop-level costs to on_frame hooks: wall_ms is the full
            # process_frame time (C++ engine + numpy->Mat convert + marshal),
            # draw_ms is the throttled display cost. Lets diagnostic logs tell
            # tracker-bound from display-bound apart without extra plumbing.
            try:
                tmap = getattr(result, "timings_ms", None)
                if isinstance(tmap, dict):
                    tmap["wall_ms"] = wall_ms
                    tmap["draw_ms"] = draw_ms
            except Exception:
                pass

            logger.log_frame(frame_idx, result, wall_ms, tracker, draw_ms)

            if on_frame is not None:
                try:
                    on_frame(frame_idx, raw_frame, result, logger.is_active())
                except Exception as exc:
                    print(f"{console_prefix} on_frame hook failed: {exc}")

            # effective-fps heartbeat so a throttled loop is VISIBLE immediately
            _fps_win.append(time.perf_counter())
            if len(_fps_win) >= 60:
                span = _fps_win[-1] - _fps_win[0]
                if span > 0:
                    print(f"{console_prefix} effective {59.0 / span:.1f} fps "
                          f"(tracker {wall_ms:.0f} ms)")
                _fps_win.clear()

            # Scheduled cycle cleanup at a controlled moment (frame fully
            # processed + logged): keeps any reference-cycle buildup bounded
            # while turning the one unavoidable collection into a predictable,
            # rare event instead of random mid-frame pauses.
            if frame_idx % _gc_collect_interval == 0:
                gc.collect()

            key = cv2.waitKey(1) & 0xFF if do_render else 0xFF
            if key != 0xFF and on_key is not None:
                try:
                    if on_key(key, tracker, last_result, frame_idx):
                        continue
                except Exception as exc:
                    print(f"{console_prefix} on_key hook failed: {exc}")
            if key == 27 or (quit_on_q and key == ord("q")):
                if logger.is_active():
                    closed_log_path = logger.close()
                    if closed_log_path is not None:
                        recorded_log_path = closed_log_path
                        if on_log_close is not None:
                            on_log_close(closed_log_path)
                break
            if key == ord("r"):
                enter_idle("reset")
            if key == ord("s"):
                if app_state == APP_IDLE:
                    start_acquire(manual=True)
                    # Single conscious start: open the logger together with
                    # tracking so the operator arms both with one 's' press.
                    if (
                        live_config.start_logging_with_tracking
                        and isinstance(logger, JsonlTrackerLogger)
                        and not logger.is_active()
                    ):
                        logger.open(Path(field_path), Path(marker_json_path), tracker)
                        if on_log_open is not None:
                            on_log_open(Path(field_path), Path(marker_json_path), tracker)
                else:
                    enter_idle("manual stop")
                    if (
                        live_config.start_logging_with_tracking
                        and isinstance(logger, JsonlTrackerLogger)
                        and logger.is_active()
                    ):
                        closed_log_path = logger.close()
                        if closed_log_path is not None:
                            recorded_log_path = closed_log_path
                            if on_log_close is not None:
                                on_log_close(closed_log_path)
            if key == ord(" "):
                if on_space_key is not None and on_space_key(tracker, last_result, frame_idx):
                    continue
                if isinstance(logger, NullTrackerLogger):
                    continue
                if logger.is_active():
                    closed_log_path = logger.close()
                    if closed_log_path is not None:
                        recorded_log_path = closed_log_path
                        if on_log_close is not None:
                            on_log_close(closed_log_path)
                    if stop_after_log_close:
                        break
                else:
                    logger.open(Path(field_path), Path(marker_json_path), tracker)
                    if on_log_open is not None:
                        on_log_open(Path(field_path), Path(marker_json_path), tracker)

    finally:
        # Restore automatic GC and release the frozen set so a later
        # run_tracking() call in the same process starts from a clean state.
        gc.unfreeze()
        gc.enable()
        try:
            if final_cleanup is not None:
                final_cleanup()
        finally:
            camera.stop()
            if live_config.show_window:
                cv2.destroyAllWindows()
            closed_path = logger.close()
            if closed_path is not None:
                recorded_log_path = closed_path

    return recorded_log_path


def run_live_tracker(**kwargs) -> Optional[Path]:
    """Backward-compatible alias for the generic runner."""
    return run_tracker(**kwargs)


def main() -> None:
    run_tracker()


if __name__ == "__main__":
    main()


__all__ = [
    "APP_ACQUIRE",
    "APP_IDLE",
    "APP_PROVISIONAL",
    "APP_TRACKING",
    "JsonlTrackerLogger",
    "NullTrackerLogger",
    "create_tracker_logger",
    "make_live_tracker_config",
    "make_tracker",
    "run_live_tracker",
    "run_tracker",
    "tracker_log",
]
