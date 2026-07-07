"""Generic live runner for HydraMarker tracking.

The runner owns orchestration only: camera frames, tracker lifetime, app state,
keyboard controls, callbacks, and optional JSONL logging. It does not implement
detector, pose, or debug-overlay drawing logic.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys
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

LOG_FRAME_DETAILS_ENV = "HYDRAMARKER_LOG_FRAME_DETAILS"
LOG_POSE_CANDIDATES_ENV = "HYDRAMARKER_LOG_POSE_CANDIDATES"


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
        if after_tracker_created is not None:
            after_tracker_created(tracker)

        if live_config.show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_idx = 0
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

        while True:
            frame = camera.read()
            if frame is None:
                continue

            frame_idx = int(frame.frame_index)
            raw_frame = frame.image_bgr

            t0 = time.perf_counter()
            result = tracker.process_frame(
                raw_frame,
                run_detection=(app_state != APP_IDLE),
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0
            last_result = result

            det_count = len(getattr(result, "detection_corners", []))
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

            draw_t0 = time.perf_counter()
            if live_config.show_window:
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
                cv2.imshow(window_name, vis)
            draw_ms = (time.perf_counter() - draw_t0) * 1000.0

            logger.log_frame(frame_idx, result, wall_ms, tracker, draw_ms)

            if on_frame is not None:
                try:
                    on_frame(frame_idx, raw_frame, result, logger.is_active())
                except Exception as exc:
                    print(f"{console_prefix} on_frame hook failed: {exc}")

            key = cv2.waitKey(1) & 0xFF if live_config.show_window else 0xFF
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
                else:
                    enter_idle("manual stop")
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
