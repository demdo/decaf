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

    # RealSense D435i color tuning. Leave values as None to keep the device
    # default/current setting. For low-noise tracking, use fixed exposure/gain
    # together with enough external light.
    realsense_enable_auto_exposure: Optional[bool] = None
    realsense_exposure: Optional[float] = None
    realsense_gain: Optional[float] = None
    realsense_enable_auto_white_balance: Optional[bool] = None
    realsense_white_balance: Optional[float] = None
    realsense_power_line_frequency: Optional[float] = None
    realsense_brightness: Optional[float] = None
    realsense_contrast: Optional[float] = None
    realsense_saturation: Optional[float] = None
    realsense_sharpness: Optional[float] = None
    realsense_gamma: Optional[float] = None
    # Global-shutter IR pair co-streaming for the tracker's IR pose
    # refinement (design B). RealSense-only: every other backend ignores it
    # and delivers ir_left/ir_right = None. Enabling this also disables the
    # IR emitter (speckle would ruin the checkerboard corners).
    realsense_enable_ir: bool = False
    realsense_ir_width: int = 1280
    realsense_ir_height: int = 720
    """

    backend: str = "realsense"
    width: int = 1920
    height: int = 1080
    fps: int = MAX_CAMERA_FPS
    calibration_path: Optional[str] = None
    serial: Optional[str] = None
    # Global-shutter IR pair co-streaming for the tracker's IR pose
    # refinement (design B). RealSense-only: every other backend ignores it
    # and delivers ir_left/ir_right = None. Enabling this also disables the
    # IR emitter (speckle would ruin the checkerboard corners).
    realsense_enable_ir: bool = False
    realsense_ir_width: int = 1280
    realsense_ir_height: int = 720
    # Constant frame rate: auto_exposure_priority=0 makes the colour AE keep
    # the configured fps ALWAYS - exposure is capped at the frame period and
    # brightness is compensated via gain instead of longer exposures. Without
    # this (None = camera default, priority ON) the camera silently drops to
    # ~17-19 fps in dim light (measured on the 12.07 evening runs). Trade-off:
    # in dim light the image gets noisier (gain); provide decent lighting.
    # With the IR refinement the geometry comes from IR anyway - RGB only
    # needs to identify corners, which tolerates gain noise well.
    realsense_auto_exposure_priority: Optional[float] = 0.0
    # Manual IR exposure (microseconds). None = auto. Short values (3000-8000)
    # reduce motion blur on the IR pair -> higher model_warp success during
    # fast motion; needs enough scene brightness (or realsense_ir_gain).
    realsense_ir_exposure_us: Optional[float] = None
    realsense_ir_gain: Optional[float] = None

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
        # Basler-specific fields are optional: the RealSense variant of this
        # dataclass omits them, so validate only the ones actually present.
        pixel_format = getattr(self, "pixel_format", None)
        if pixel_format is not None and not str(pixel_format).strip():
            raise ValueError("camera pixel_format must not be empty.")
        exposure_auto = getattr(self, "exposure_auto", None)
        if exposure_auto is not None and not str(exposure_auto).strip():
            raise ValueError("camera exposure_auto must not be empty.")
        exposure_time_us = getattr(self, "exposure_time_us", None)
        if exposure_time_us is not None and float(exposure_time_us) <= 0.0:
            raise ValueError(
                f"camera exposure_time_us must be positive; got {exposure_time_us}."
            )
        gain_auto = getattr(self, "gain_auto", None)
        if gain_auto is not None and not str(gain_auto).strip():
            raise ValueError("camera gain_auto must not be empty.")
        throughput_mode = getattr(self, "device_link_throughput_limit_mode", None)
        if throughput_mode is not None and not str(throughput_mode).strip():
            raise ValueError("camera device_link_throughput_limit_mode must not be empty.")
        throughput_limit = getattr(self, "device_link_throughput_limit", None)
        if throughput_limit is not None and int(throughput_limit) <= 0:
            raise ValueError(
                "camera device_link_throughput_limit must be positive when set; "
                f"got {throughput_limit}."
            )
        realsense_positive_fields = (
            "realsense_exposure",
            "realsense_gain",
            "realsense_white_balance",
            "realsense_gamma",
        )
        for name in realsense_positive_fields:
            value = getattr(self, name, None)
            if value is not None and float(value) <= 0.0:
                raise ValueError(f"camera {name} must be positive when set; got {value}.")

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
    # Couple the JSONL logger to the manual 's' tracking start.  When True the
    # logger opens exactly when tracking starts (single conscious start via 's')
    # and closes when the operator stops tracking, instead of the separate SPACE
    # toggle.  Opt-in so other tools keep the independent tracking/logging keys.
    start_logging_with_tracking: bool = False
    # Re-arm acquisition automatically from IDLE via a periodic detection
    # probe.  Default OFF: during tests the operator keeps explicit control
    # over the start via the 's' key.
    idle_auto_reacquire: bool = False

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
    checker_refresh_interval_frames: int = 3
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
    checker_tracked_refine_method: str = "quadratic_form"

    # quadratic_form corner measurement (reference-free curve intersection;
    # active when checker_tracked_refine_method == "quadratic_form").
    checker_qf_profile_half_px: float = 4.5
    checker_qf_min_contrast: float = 12.0
    checker_qf_max_profile_rms: float = 0.35
    checker_qf_junction_margin_frac: float = 0.18
    checker_qf_min_row_points: int = 6
    checker_qf_min_col_points: int = 3
    checker_qf_conic_gain: float = 0.95
    checker_qf_max_fit_rms_px: float = 0.6
    checker_qf_max_dev_px: float = 2.5

    # Pose-filter distrust for frames where the IR MAP fusion attempted and
    # REJECTED (references existed, gates failed: glare veil, adverse
    # geometry): the surviving RGB-only pose is known-degraded evidence, so
    # its measurement covariance is inflated by this factor (the filter
    # coasts on its motion model through the episode). C++ struct default
    # is 1.0 (off) so acceptance replays stay bit-identical.
    ir_reject_cov_inflate: float = 25.0

    # IR corner measurement operator: "model_warp" = registration against
    # enrolled reference photo pairs (established behaviour), or
    # "quadratic_form" = reference-free direct measurement of the projected
    # model grid curves in each IR view (Phase 2 — no reference library, no
    # enrollment, works at any orientation from frame 1). Stays model_warp
    # until the offline A/B on the full corpus validates the QF-IR path.
    ir_corner_method: str = "quadratic_form"
    ir_qf_max_dev_px: float = 4.0

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
    pose_kf_q_translation_mm: float = 0.15  # moving accel noise per frame
    pose_kf_q_rotation_deg: float = 0.5     # moving accel noise per frame
    pose_kf_gate_mahalanobis: float = 30.0  # spike deweighting gate
    pose_kf_reset_rotation_deg: float = 10.0
    # Adaptive process noise (default): Q collapses to the floor while the tool
    # is still (averages many frames -> static tip jitter 0.10 -> 0.034 mm) and
    # opens up the instant motion is measured (follows without lag). The fixed
    # pose_kf_q_* above are the non-adaptive fallback, unused while adaptive.
    pose_kf_adaptive: bool = True
    pose_kf_q_rotation_floor_deg: float = 0.002
    pose_kf_q_translation_floor_mm: float = 0.002
    pose_kf_adaptive_ema_alpha: float = 0.4
    # Self-calibrating noise level (default): estimate the quiet-baseline pose
    # delta online and derive both floors from it (adapts per session/orientation;
    # fixed floors cannot). Manual floors are fallback + lower bound.
    pose_kf_auto_noise: bool = True
    pose_kf_noise_window: int = 90
    pose_kf_motion_floor_rot_deg: float = 0.12
    pose_kf_motion_floor_trans_mm: float = 0.05
    pose_kf_meas_floor_rot_deg: float = 0.035
    pose_kf_meas_floor_trans_mm: float = 0.02
    # IR depth-only stereo fusion (output-only). Fuses absolute depth and the
    # two out-of-plane tilt DoF from the global-shutter IR stereo pair into
    # the reported pose; RGB keeps rotation and lateral position (full IR
    # pose replacement is WORSE at the tip: stereo rotation noise times the
    # tool lever). Multi-reference model_warp measurement (>= min_ref_rot_deg
    # away from each reference's own enrollment orientation, best reference
    # by summed ZNCC quality), per-corner saturation gate, robust
    # ZNCC-weighted tilt fit with clamped median-depth fallback; without
    # enough pairs the RGB pose stays. Validated offline on divot session
    # 20260718_181438 (tests/prove_ir_depth_fusion.py: |e| med 1.43->0.20mm).
    # TRIPLE-GATED: this flag AND tracker.set_ir_calibration(...) AND the IR
    # frame pair per process_frame call; without any of these the tracker
    # behaves exactly as before.
    ir_refine_enabled: bool = False
    ir_min_pairs: int = 6
    ir_min_ref_rot_deg: float = 6.0
    ir_max_ref_rot_deg: float = 180.0
    # Fixed-orientation fallback floor: only used when no reference reaches
    # ir_min_ref_rot_deg (pure translation). Lets IR still correct depth at a
    # single held orientation without touching the divot/tilt path.
    ir_fallback_min_ref_rot_deg: float = 0.05
    ir_sat_threshold: int = 250
    ir_sat_half_px: int = 8
    ir_epipolar_max_dv_px: float = 2.5
    ir_zncc_weight_floor: float = 0.05
    ir_dtz_clamp_mm: float = 2.5
    # Absolute stereo scale from the session's ChArUco validation shots
    # (triangulated square size vs known; 1.00197 on session 20260718).
    ir_depth_scale: float = 1.00197
    # MAP fusion (tightly-coupled RGB reproj + IR stereo 3D). Sigmas calibrated
    # from residuals -> ir_w3d = 1. The fit gate rejects an unreliable fusion.
    ir_sigma_px: float = 0.10
    # Rolling-shutter compensation: sigma_px_eff^2 = sigma_px^2 + (coeff*v_px)^2
    # (v_px = median per-frame corner motion). 0 disables.
    ir_sigma_px_motion_coeff: float = 0.20
    # Rotation share of the image velocity (px per deg/frame), subtracted from
    # vpx so only the TRANSLATION share drives the shear compensation.
    ir_sigma_px_motion_rot_px_per_deg: float = 8.0
    ir_sigma_ir_px: float = 0.05
    # IR/reproj balance: 1 = still-frame sigma calibration; 4 compensates the
    # correlated (non-IID) RGB corner errors during motion (weak-mode wobble).
    ir_w3d: float = 4.0
    # Gate calibration 2026-07-22 (fb1_22 glare window): monocular RGB z drifts
    # 3-4.5mm while stereo fit stays 1.5-1.65, so the old 1.5/3.0 gates rejected
    # the fusion exactly when it was needed most (z spikes to -4.4). 1.8/6.0
    # validated across fb1_22/fb4/fb5/fb1_21/fb_rl (p95|z| 10.8->2.4) with only
    # a small robust-std cost on fb_rl2 (0.87->1.15); divot ABS unchanged.
    ir_fit_gate_rms_mm: float = 1.8
    ir_fit_gate_max_trans_jump_mm: float = 6.0
    # Weak-evidence honesty: inflate the MAP covariance handed to the temporal
    # filter by the reduced chi-square (fit_rms / this)^2 so the filter damps
    # poor-fit frames instead of following them. 0 disables.
    ir_cov_inflate_ref_rms_mm: float = 0.10
    ir_mw_min_zncc: float = 0.35
    ir_mw_max_shift_px: float = 6.0
    ir_enroll_max_rot_deg: float = 3.0
    ir_enroll_max_trans_mm: float = 8.0
    ir_ref_tile_deg: float = 12.0
    # Position tile: also enroll when the pose moved this far from every
    # reference (a translation sweep at fixed orientation never leaves the
    # orientation tile, freezing the library at one home reference).
    # <= 0 disables the position dimension = previous behaviour.
    ir_ref_tile_trans_mm: float = 40.0
    ir_fallback_min_ref_trans_mm: float = 15.0
    # Bootstrap-bias guard: position-only references enroll only on a frame
    # with a demonstrably good fusion (never blocks new-orientation refs).
    ir_enroll_max_fit_rms_mm: float = 0.20
    ir_enroll_max_sat_frac: float = 0.35
    ir_max_references: int = 12
    # Model-warp reference re-enrollment: fresh reference after a full
    # tracking loss and when the viewing direction moved beyond the angle
    # threshold (stale-reference guard for reorientation — NOT a slope fix;
    # threshold far above normal in-run angle changes). (Re-)enrollment
    # only happens while the tool is quiet (sharp reference).
    # Threshold sized against real data: +-85mm step runs reach ~15 deg
    # viewing-angle offset, a deliberate fb<->rl reorientation is ~90 deg.
    model_warp_reenroll_on_loss: bool = True
    # Template refresh distance: 20 was never reached by the fb sweeps (max
    # ~11deg) -> whole push refined against a skewed template (z step +0.4 in
    # the 8-12deg band). 6 re-enrolls during natural pauses. 0 disables.
    model_warp_reenroll_angle_deg: float = 6.0
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
    recovery_grace_frames: int = 3
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
    fast_persistent_min_points: int = 8
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
