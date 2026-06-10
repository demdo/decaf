from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from tracking.hydramarker.backend import cpp_impl as hm
from tracking.hydramarker.identity_store import IdentityStore
from tracking.hydramarker.config import TrackerConfig
from tracking.hydramarker.tracker_types import (
    FastPathDebug,
    PersistentMatchStats,
    TrackerMode,
    TrackerResult,
)
from tracking.hydramarker.tracker_decode_helpers import DecodeHelperMixin
from tracking.hydramarker.tracker_decode_pipeline import DecodePipelineMixin
from tracking.hydramarker.tracker_dense_refine import DenseRefineMixin
from tracking.hydramarker.tracker_factories import TrackerFactoryMixin
from tracking.hydramarker.tracker_fallbacks import FallbackPoseMixin
from tracking.hydramarker.tracker_fast_path import FastPathMixin
from tracking.hydramarker.tracker_geometry import GeometryMixin
from tracking.hydramarker.tracker_persistence import PersistenceMixin
from tracking.hydramarker.tracker_pose_estimation import PoseEstimationMixin
from tracking.hydramarker.tracker_pose_propagation import PosePropagationMixin
from tracking.hydramarker.tracker_projection import ProjectionMixin


class HydraTracker(
    TrackerFactoryMixin,
    DecodePipelineMixin,
    PoseEstimationMixin,
    FastPathMixin,
    DenseRefineMixin,
    FallbackPoseMixin,
    ProjectionMixin,
    PosePropagationMixin,
    DecodeHelperMixin,
    PersistenceMixin,
    GeometryMixin,
):
    """Frame-level tracker orchestrator; implementation lives in tracker_* mixins."""

    def __init__(
        self,
        field_path: str,
        marker_json_path: str,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        config: Optional[TrackerConfig] = None,
    ) -> None:
        self.config = config or TrackerConfig()

        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = (
            np.zeros((0, 1), dtype=np.float64)
            if dist_coeffs is None
            else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        )

        self.field = hm.MarkerField.loadFromFile(field_path)
        self.geometry = hm.MarkerGeometry.load_from_json(marker_json_path)
        self._geometry_corner_cache = self._build_geometry_corner_cache()

        self.checkerboard_detector = self._create_checkerboard_detector()
        self.dot_detector = self._create_dot_detector()
        self.patch_extractor = self._create_patch_extractor()
        self.patch_decoder = self._create_patch_decoder()
        self.correspondence_builder = self._create_correspondence_builder()
        self.pose_tracker = self._create_pose_tracker(K, dist_coeffs)

        self.mode = TrackerMode.LOST
        self.frame_index = 0
        self.lost_frames = 0

        # Highest accepted point count seen so far for early drift detection.
        self._max_pts_seen: int = 0

        # Last accepted reprojection error used by pose propagation.
        self._last_good_reproj_px: float = -1.0

        # Last accepted pose used by motion gates and rotation-change checks.
        self._last_accepted_rvec: Optional[np.ndarray] = None
        self._last_accepted_tvec: Optional[np.ndarray] = None
        self._last_accepted_T_marker_camera: Optional[np.ndarray] = None
        self._last_accepted_pose_frame: int = -1

        self._identity_store = IdentityStore()
        self._persistent_frame_index: int = -1
        self._undecodeable_detection_frames: int = 0
        self._low_fresh_correspondence_frames: int = 0
        self._pose_propagation_block_until_frame: int = -1
        self._last_uncoded_bootstrap_reason: str = ""
        self._last_persistent_match_stats = PersistentMatchStats()
        self._last_fast_path_debug = FastPathDebug()

    @property
    def rvec(self) -> Optional[np.ndarray]:
        return self.pose_tracker.rvec

    @property
    def tvec(self) -> Optional[np.ndarray]:
        return self.pose_tracker.tvec

    @property
    def T_marker_camera(self) -> Optional[np.ndarray]:
        return self.pose_tracker.T_marker_camera

    def reset(self) -> None:
        self.mode = TrackerMode.LOST
        self.frame_index = 0
        self.lost_frames = 0
        self._max_pts_seen = 0
        self._last_good_reproj_px = -1.0
        self._last_accepted_rvec = None
        self._last_accepted_tvec = None
        self._last_accepted_T_marker_camera = None
        self._last_accepted_pose_frame = -1
        self._undecodeable_detection_frames = 0
        self._low_fresh_correspondence_frames = 0
        self._pose_propagation_block_until_frame = -1
        self._last_uncoded_bootstrap_reason = ""
        self._last_persistent_match_stats = PersistentMatchStats()
        self._last_fast_path_debug = FastPathDebug()

        self.pose_tracker.reset()
        self.checkerboard_detector.reset_tracking()

        # Full reset: recreate dot detector to clear all smoothed state.
        # reset_smoothing() is called on partial resets (_on_tracking_failure)
        # to preserve warmup state while clearing stale cell scores.
        self.dot_detector = self._create_dot_detector()

        self._clear_persistent_correspondences()

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        run_detection: bool = True,
    ) -> TrackerResult:
        frame_t0 = time.perf_counter()
        timings_ms: Dict[str, float] = {}

        def mark(name: str, start: float) -> None:
            timings_ms[name] = (time.perf_counter() - start) * 1000.0

        def finish(result: TrackerResult) -> TrackerResult:
            timings_ms["tracker_total_ms"] = (time.perf_counter() - frame_t0) * 1000.0
            merged_timings = dict(getattr(result, "timings_ms", {}) or {})
            merged_timings.update(timings_ms)
            result.timings_ms = merged_timings
            return result

        self.frame_index += 1

        if not run_detection:
            result = TrackerResult(
                success=False,
                mode=self.mode,
                message="Idle: checkerboard detection skipped.",
            )
            result.fast_path_debug = FastPathDebug(reason="idle_skipped")
            timings_ms["checkerboard_ms"] = 0.0
            timings_ms["idle_skip"] = 1.0
            self._last_fast_path_debug = result.fast_path_debug
            return finish(result)

        stage_t0 = time.perf_counter()
        detection = self.checkerboard_detector.detect(frame)
        mark("checkerboard_ms", stage_t0)
        if hasattr(self.checkerboard_detector, "last_timings_ms"):
            try:
                for key, value in self.checkerboard_detector.last_timings_ms().items():
                    timings_ms[f"checkerboard_{key}"] = float(value)
            except Exception:
                pass

        if detection is None or not detection.valid():
            self._last_fast_path_debug = FastPathDebug(reason="no_checkerboard")
            self._undecodeable_detection_frames = 0
            self._on_tracking_failure()
            stage_t0 = time.perf_counter()
            held = self._hold_last_pose_without_detection_result(detection)
            mark("hold_pose_ms", stage_t0)
            if held is not None:
                self._attach_fast_path_debug(held)
                return finish(held)
            stage_t0 = time.perf_counter()
            emergency = self._emergency_last_pose_result(
                detection,
                reason="No valid checkerboard detection",
            )
            mark("emergency_hold_ms", stage_t0)
            if emergency is not None:
                self._attach_fast_path_debug(emergency)
                return finish(emergency)

            result = TrackerResult(
                success=False,
                mode=self.mode,
                message="No valid checkerboard detection.",
                detection_valid=False,
                detection_tracking=False if detection is None else bool(detection.tracking),
                detection_stable=False if detection is None else bool(detection.stable),
                detection_corners=self._detected_corners_from_detection(detection),
            )
            self._attach_fast_path_debug(result)
            return finish(result)

        stage_t0 = time.perf_counter()
        fast_result = self._try_fast_pose_from_persistent_correspondences(detection)
        mark("fast_persistent_ms", stage_t0)
        if fast_result is not None:
            self._attach_detection_info(fast_result, detection)
            self._attach_fast_path_debug(fast_result)
            self.mode = TrackerMode.TRACKING
            self.lost_frames = 0
            fast_result.mode = self.mode
            return finish(fast_result)

        # If the previous pose is trustworthy, decode against projected marker
        # corners instead of the LK detection to reduce accumulated LK drift.
        h, w = frame.shape[:2]
        stage_t0 = time.perf_counter()
        propagated = self._build_pose_propagated_detection((h, w))
        mark("pose_propagation_ms", stage_t0)
        detection_for_dots = propagated if propagated is not None else detection

        stage_t0 = time.perf_counter()
        result = self._decode_and_estimate_pose(frame, detection_for_dots)
        mark("decode_pose_ms", stage_t0)
        self._attach_detection_info(result, detection)
        self._attach_fast_path_debug(result)

        if result.success:
            self.mode = TrackerMode.TRACKING
            self.lost_frames = 0
            result.mode = self.mode
            return finish(result)

        self._on_tracking_failure()
        stage_t0 = time.perf_counter()
        emergency = self._emergency_last_pose_result(
            detection,
            reason=result.message,
        )
        mark("emergency_hold_ms", stage_t0)
        if emergency is not None:
            self._attach_fast_path_debug(emergency)
            return finish(emergency)

        result.mode = self.mode
        result.corners = []
        self._attach_fast_path_debug(result)
        return finish(result)

    def _on_tracking_failure(self) -> None:
        self.lost_frames += 1

        if self.lost_frames > self.config.max_lost_frames:
            self.pose_tracker.reset()
            self._clear_persistent_correspondences()
            self.mode = TrackerMode.LOST

            # Full dot detector reset only on complete loss: recreate to
            # clear all state including stale cell scores from a different
            # marker position.
            self.dot_detector = self._create_dot_detector()

        elif self.pose_tracker.rvec is not None and self.pose_tracker.tvec is not None:
            self.mode = TrackerMode.RECOVERING

            # Partial reset: tell the dot detector to clear its smoothed
            # scores so it re-commits quickly in the next frame, but don't
            # destroy the detector object (keeps warmup done).
            if hasattr(self.dot_detector, "reset_smoothing"):
                self.dot_detector.reset_smoothing()

        else:
            self.mode = TrackerMode.DETECTING

            if hasattr(self.dot_detector, "reset_smoothing"):
                self.dot_detector.reset_smoothing()
