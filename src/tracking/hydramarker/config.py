"""Runtime configuration for HydraMarker tracking.

This module is intentionally the user-facing Python configuration surface for
live HydraMarker tracking. Camera and logging options are plain data consumed by
``camera_setup.py`` and ``run_tracker.py``; they do not import or open camera
SDKs. ``TrackerConfig`` is copied into the native tracker runtime in
``tracker.py`` and then consumed by the staged C++ pipeline.

Changing these values does not require rebuilding C++. A new ``HydraTracker``
instance copies the current Python tracker dataclass values into the native
runtime at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


MAX_CAMERA_FPS = 30
CAMERA_BACKENDS = ("realsense", "basler")
LOGGING_START_MODES = ("manual", "auto", "disabled")


@dataclass
class CameraConfig:
    """User-selected live camera settings.

    The config is deliberately inert: hardware-specific validation and SDK
    imports live in ``camera_setup.py`` so importing the tracker remains
    possible without any camera package installed.

    ----- Tested cameras ----------------------------------
    backend: str = "basler"
    width: int = 1920
    height: int = 1080
    fps: int = MAX_CAMERA_FPS
    pixel_format: Optional[str] = "auto"
    exposure_auto: Optional[str] = "Off"
    exposure_time_us: Optional[float] = 8000.0
    gain_auto: Optional[str] = "Off"
    gain_db: Optional[float] = None
    device_link_throughput_limit_mode: Optional[str] = "Off"
    device_link_throughput_limit: Optional[int] = None
    calibration_path: Optional[str] = "data/basler/basler_camera_calibration_standard5.npz"
    serial: Optional[str] = None

    backend: str = "realsense"
    width: int = 1920
    height: int = 1080
    fps: int = MAX_CAMERA_FPS
    calibration_path: Optional[str] = None
    serial: Optional[str] = None
    """

    backend: str = "basler"
    width: int = 1920
    height: int = 1080
    fps: int = MAX_CAMERA_FPS
    pixel_format: Optional[str] = "auto"
    exposure_auto: Optional[str] = "Off"
    exposure_time_us: Optional[float] = 8000.0
    gain_auto: Optional[str] = "Off"
    gain_db: Optional[float] = None
    device_link_throughput_limit_mode: Optional[str] = "Off"
    device_link_throughput_limit: Optional[int] = None
    calibration_path: Optional[str] = "data/basler/basler_camera_calibration_standard5.npz"
    serial: Optional[str] = None

    def validate_common(self) -> None:
        backend = str(self.backend).strip().lower()
        if backend not in CAMERA_BACKENDS:
            raise ValueError(
                "camera backend must be one of "
                f"{', '.join(CAMERA_BACKENDS)}; got {self.backend!r}."
            )
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError(
                f"camera width/height must be positive; got "
                f"{self.width}x{self.height}."
            )
        if int(self.fps) <= 0:
            raise ValueError(f"camera fps must be positive; got {self.fps}.")
        if int(self.fps) > MAX_CAMERA_FPS:
            raise ValueError(
                f"camera fps={self.fps} exceeds the HydraMarker limit of "
                f"{MAX_CAMERA_FPS} fps."
            )
        if self.pixel_format is not None and not str(self.pixel_format).strip():
            raise ValueError("camera pixel_format must not be empty.")
        if self.exposure_auto is not None and not str(self.exposure_auto).strip():
            raise ValueError("camera exposure_auto must not be empty.")
        if self.exposure_time_us is not None and float(self.exposure_time_us) <= 0.0:
            raise ValueError(
                f"camera exposure_time_us must be positive; got {self.exposure_time_us}."
            )
        if self.gain_auto is not None and not str(self.gain_auto).strip():
            raise ValueError("camera gain_auto must not be empty.")
        if self.device_link_throughput_limit_mode is not None and not str(
            self.device_link_throughput_limit_mode
        ).strip():
            raise ValueError("camera device_link_throughput_limit_mode must not be empty.")
        if (
            self.device_link_throughput_limit is not None
            and int(self.device_link_throughput_limit) <= 0
        ):
            raise ValueError(
                "camera device_link_throughput_limit must be positive when set; "
                f"got {self.device_link_throughput_limit}."
            )

    def calibration_path_obj(self) -> Optional[Path]:
        if self.calibration_path is None or str(self.calibration_path).strip() == "":
            return None
        path = Path(self.calibration_path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        return path


@dataclass
class LoggingConfig:
    """Live run logging behavior."""

    enabled: bool = True
    start_mode: str = "manual"
    output_dir: str = "hydramarker_tracker_runs"
    frame_details: bool = True
    pose_candidates: bool = True

    def validate_common(self) -> None:
        mode = str(self.start_mode).strip().lower()
        if mode not in LOGGING_START_MODES:
            raise ValueError(
                "logging start_mode must be one of "
                f"{', '.join(LOGGING_START_MODES)}; got {self.start_mode!r}."
            )

    @property
    def active(self) -> bool:
        return bool(self.enabled) and str(self.start_mode).strip().lower() != "disabled"


@dataclass
class LiveTrackerConfig:
    """Top-level live tracking options used by ``run_tracker.py``."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tracker: "TrackerConfig" = field(default_factory=lambda: TrackerConfig())
    show_window: bool = True
    start_tracking_manually: bool = True

    def validate_common(self) -> None:
        self.camera.validate_common()
        self.logging.validate_common()


@dataclass
class TrackerConfig:
    """Grouped runtime thresholds for the native tracking pipeline."""

    # Engine
    # Copied directly into cpp/include/tracker_config.hpp::TrackerConfig.
    # TrackerEngine uses these values for global mode selection, lost-frame
    # accounting, and reset behavior in cpp/src/tracker_engine.cpp.
    max_lost_frames: int = 8
    decode_only_mode: bool = False

    # Checkerboard
    # Copied by TrackerEngine::makeCheckerboardConfig into
    # CheckerboardDetectorConfig. These values control LK tracking, refresh
    # cadence, and recovery for the visible checkerboard lattice.
    checker_min_tracking_decode_cell_span: int = 3
    checker_refresh_interval_frames: int = 1
    checker_tracking_recovery_stable_interval_frames: int = 9
    checker_tracking_recovery_zero_gain_backoff_after: int = 3
    checker_tracking_recovery_zero_gain_backoff_max_factor: int = 16
    checker_local_completion_skip_enabled: bool = True
    checker_local_completion_probe_interval_frames: int = 6
    checker_local_completion_zero_gain_backoff_after: int = 3
    checker_local_completion_zero_gain_backoff_max_factor: int = 16
    # Caps old LK predictions after roughly three soft-probe intervals so
    # static rim corners stop triggering repeated local-completion work.
    # Set to 0 to keep the legacy dynamic predicted-corner lifetime.
    checker_local_completion_stale_predicted_frames: int = 18
    checker_max_undecodeable_tracking_frames: int = 12
    checker_min_fresh_correspondences_for_stable_tracking: int = 8
    checker_max_low_fresh_correspondence_frames: int = 12
    # Measurement operator for LK-tracked corners ("subpix" = cornerSubPix
    # snap, "model_warp" = forward-model template registration once
    # implemented; unknown values fall back to "subpix").
    checker_tracked_refine_method: str = "model_warp"

    # Pose-set stabilisation (2026-07-05). The cylinder pose is nearly blind
    # along one direction (~1 mm z per 0.1 px), so predicted corners (which
    # feed the previous pose back into PnP) and freshly appeared corners
    # (which jump the pose along the weak mode when they enter the set) are
    # kept out of the POSE input. Detection/decoding/persistence still use
    # every corner; the filter relaxes automatically if it would starve the
    # solver.
    pose_exclude_predicted_corners: bool = True
    pose_min_observed_frames: int = 5  # 0 disables the entry hysteresis
    # Pose warmup status: after (re)acquisition the pose wanders while the
    # corner set saturates (measured: z std 0.64 mm although static).
    # result.pose_converged latches once the set has been quiet; poses are
    # still produced during warmup, downstream decides how to treat them.
    pose_warmup_min_accepted_frames: int = 20  # 0 = always converged
    pose_warmup_stable_window: int = 15
    pose_warmup_max_young_corners: int = 2
    # Anisotropic pose Kalman filter (OUTPUT-only: rvec/tvec/T of the frame
    # result are filtered, the internal tracking chain keeps the raw pose so
    # no feedback loop forms). Constant-velocity model; the per-frame
    # measurement covariance sigma^2*(J^T J)^-1 smooths only the weak
    # observability mode, real motion follows the measurement directly.
    pose_kf_enabled: bool = True
    pose_kf_sigma_px: float = 0.12
    pose_kf_q_translation_mm: float = 0.15  # accel noise per frame
    pose_kf_q_rotation_deg: float = 0.05    # accel noise per frame
    pose_kf_gate_mahalanobis: float = 30.0  # spike deweighting gate
    pose_kf_reset_rotation_deg: float = 10.0
    # Model-warp reference re-enrollment: fresh reference after a full
    # tracking loss and when the viewing direction moved beyond the angle
    # threshold (stale-reference guard for reorientation — NOT a slope fix;
    # threshold far above normal in-run angle changes). (Re-)enrollment
    # only happens while the tool is quiet (sharp reference).
    # Threshold sized against real data: +-85mm step runs reach ~15 deg
    # viewing-angle offset, a deliberate fb<->rl reorientation is ~90 deg.
    model_warp_reenroll_on_loss: bool = True
    model_warp_reenroll_angle_deg: float = 20.0  # 0 disables
    model_warp_enroll_max_motion_mm: float = 1.0     # per frame
    model_warp_enroll_max_rotation_deg: float = 0.25  # per frame

    # Dot/Patch/Decode
    # Dot values are copied by TrackerEngine::makeDotDetectorConfig, decoder
    # values by makePatchDecoderConfig, and correspondence values by
    # makeCorrespondenceBuilderConfig. Together they form the fresh
    # patch-decoding stage before pose estimation.
    dot_canonical_size: int = 80
    dot_canonical_margin_px: float = 4.0
    dot_min_dot_contrast: float = 8.0
    dot_strong_dot_contrast: float = 35.0
    dot_commit_threshold: float = 0.45
    dot_revoke_threshold: float = 0.20
    dot_uncertainty_low: float = 0.20
    dot_uncertainty_high: float = 0.45
    dot_warmup_frames: int = 1
    dot_temporal_alpha: float = 0.35
    dot_commit_frames: int = 2
    dot_revoke_frames: int = 3
    dot_use_temporal_smoothing: bool = False
    dot_use_cell_value_cache: bool = True
    dot_cell_cache_max_age_frames: int = 12
    dot_cell_cache_max_corner_motion_px: float = 35.0

    decoder_require_geometry_valid: bool = True
    decoder_accept_ambiguous: bool = False

    corr_min_votes: int = 2
    corr_discard_conflicts: bool = True
    corr_require_detection_stable: bool = False
    corr_enable_dominant_rotation_filter: bool = True
    corr_min_rotation_support: int = 2
    corr_min_rotation_support_ratio: float = 0.55

    # Pose/PnP
    # Translated by TrackerEngine::makeMapPoseTrackerConfig into
    # MapPoseTrackerConfig. MapPoseTracker and the fast-path pose estimator use
    # these thresholds for RANSAC, direct-prior solving, and motion/reprojection
    # gates.
    min_points: int = 6
    min_inliers: int = 5
    max_mean_reprojection_error_px: float = 4.0
    max_max_reprojection_error_px: float = 15.0
    max_translation_jump_mm: float = 40.0
    max_rotation_jump_deg: float = 45.0
    rotation_gate_scale_per_lost_frame: float = 8.0
    rotation_gate_max_deg: float = 90.0
    pnp_ransac_iterations: int = 500
    pnp_ransac_reprojection_px: float = 3.0
    pnp_ransac_confidence: float = 0.99
    use_pose_prior: bool = True
    pnp_direct_prior_enabled: bool = True
    pnp_direct_refine_method: str = "lm"
    pnp_direct_max_mean_reprojection_error_px: float = 1.5
    pnp_direct_max_max_reprojection_error_px: float = 3.0

    # Persistence/Fast Path
    # Used by PersistentMatcher, TrackerGeometry, and TrackerEngine in
    # cpp/src/tracker_persistence.cpp, tracker_geometry.cpp, and
    # tracker_engine.cpp. Domain-level feature switches remain here; backend
    # selection switches are intentionally absent.
    enable_fast_persistent_path: bool = True
    fast_persistent_min_points: int = 10
    fast_persistent_refresh_mean_error_px: float = 1.5
    fast_persistent_dense_refine_enabled: bool = True
    fast_persistent_dense_min_points: int = 24
    fast_persistent_dense_match_max_px: float = 3.0
    fast_persistent_dense_min_second_best_margin_px: float = 2.0
    fast_persistent_dense_max_median_px: float = 1.2
    fast_persistent_dense_max_p90_px: float = 2.5
    fast_persistent_dense_rescue_enabled: bool = False
    fast_persistent_dense_rescue_min_green_ratio: float = 0.85
    fast_persistent_dense_rescue_min_seed_median_px: float = 1.5
    fast_persistent_dense_min_image_coverage: float = 0.35
    fast_persistent_dense_min_object_span_mm: float = 12.0
    fast_persistent_dense_min_distinct_rows: int = 2
    fast_persistent_dense_min_distinct_cols: int = 2
    fast_persistent_dense_pose_solver: str = "direct_prior"
    fast_persistent_dense_robust_refine_method: str = "auto"
    fast_persistent_dense_robust_trim_enabled: bool = True
    fast_persistent_dense_robust_trim_quantile: float = 0.85
    fast_persistent_dense_robust_min_keep_ratio: float = 0.75
    fast_persistent_dense_robust_max_mean_px: float = 1.2
    fast_persistent_dense_robust_max_max_px: float = 2.5
    fast_persistent_dense_adaptive_refine_enabled: bool = True
    fast_persistent_dense_adaptive_min_match_ratio: float = 0.85
    fast_persistent_dense_adaptive_motion_px: float = 8.0
    fast_persistent_dense_adaptive_max_seed_mean_px: float = 1.2
    fast_persistent_dense_adaptive_max_seed_max_px: float = 2.8

    enable_temporal_correspondence_persistence: bool = True
    persistence_max_frames: int = 8
    persistence_min_points: int = 6
    persistence_min_fresh_points_for_merge: int = 6
    persistence_min_points_after_decode_fail: int = 10
    persistence_refresh_mean_error_px: float = 1.5
    persistence_max_translation_jump_mm: float = 60.0
    persistence_max_rotation_jump_deg: float = 20.0
    persistence_use_pose_projection: bool = True
    persistence_projection_max_reproj_px: float = 9.0
    persistence_projection_adaptive_match_enabled: bool = True
    persistence_projection_adaptive_motion_start_px: float = 6.0
    persistence_projection_adaptive_motion_scale: float = 1.0
    persistence_projection_adaptive_max_reproj_px: float = 18.0
    persistence_projection_max_pose_error_px: float = 1.5
    persistence_match_min_second_best_margin_px: float = 3.0
    persistence_uv_match_dist_px: float = 25.0

    decode_update_min_visual_corners: int = 12
    decode_update_min_distinct_rows: int = 3
    decode_update_min_distinct_cols: int = 3

    # Recovery/Hold
    # Controls pose propagation, fallback pose generation, and hold paths in
    # tracker_engine.cpp when fresh decode results or correspondences are
    # missing.
    enable_pose_propagation: bool = True
    pose_propagation_max_reproj_px: float = 2.0
    pose_propagation_border_px: float = 8.0
    pose_hold_max_frames: int = 45
    pose_hold_min_detection_corners: int = 8
    emergency_pose_hold_enabled: bool = True
    emergency_pose_hold_max_frames: int = -1
    fallback_pose_min_detection_matches: int = 8
    fallback_pose_max_median_corner_error_px: float = 9.0
    fallback_pose_max_p90_corner_error_px: float = 18.0
    fallback_pose_max_mean_reprojection_error_px: float = 1.8
    fallback_pose_max_max_reprojection_error_px: float = 4.0

    # Logging Output
    # Controls the corner list that TrackerEngine returns to Python.
    # tracker_log.py records these values but does not make tracking decisions.
    visual_corner_max_reprojection_error_px: float = 3.0
    visual_corner_min_count: int = 6


__all__ = [
    "CAMERA_BACKENDS",
    "LOGGING_START_MODES",
    "MAX_CAMERA_FPS",
    "CameraConfig",
    "LiveTrackerConfig",
    "LoggingConfig",
    "TrackerConfig",
]
