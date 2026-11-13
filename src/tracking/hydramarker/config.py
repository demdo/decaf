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

import hashlib
from dataclasses import dataclass, field, fields
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
    # RealSense colour sensor options (None = leave the device as it is).
    # The defaults are the state the D435i was in at every camera start of
    # the debug recordings (30.09., 05.10.) and the evaluation runs (28.09.,
    # 05.10.): AE on, exposure register 156, gain 64, 60 Hz anti-flicker
    # (50 Hz since, see realsense_power_line_frequency), the device defaults
    # otherwise - what the app ran with while these
    # fields stood in the class docstring and were never applied.
    # Setting them explicitly keeps that image even after another program
    # left the device elsewhere. camera_setup writes exposure/gain with the
    # AE off and white_balance with the auto white balance off, then the
    # auto flags, then the rest; an option already at its value is not
    # written. Windows rounds a manual exposure to 1 s / 2^n (register
    # 10000 / 2^n in 100 us): only 9/19/39/78/156/312 are reached exactly.
    realsense_enable_auto_exposure: Optional[bool] = True
    # exposure in 100 us; the register stays as set while the AE is on
    realsense_exposure: Optional[float] = 156.0
    realsense_gain: Optional[float] = 64.0
    realsense_enable_auto_white_balance: Optional[bool] = True
    realsense_white_balance: Optional[float] = 4600.0
    # 0 off, 1 = 50 Hz, 2 = 60 Hz, 3 = auto (device default 3). 50 Hz, the
    # grid here: off / 50 Hz / auto gave the same image and tracking in the
    # debug recordings of 05.10. (K01-K03), and 60 Hz in a 50 Hz grid is the
    # one setting that cannot be right.
    realsense_power_line_frequency: Optional[float] = 1.0
    realsense_brightness: Optional[float] = 0.0
    realsense_contrast: Optional[float] = 50.0
    realsense_saturation: Optional[float] = 64.0
    realsense_hue: Optional[float] = 0.0
    realsense_sharpness: Optional[float] = 50.0
    realsense_gamma: Optional[float] = 300.0
    realsense_backlight_compensation: Optional[float] = 0.0
    # Colour AE metering region as fractions of the colour image (min_x,
    # min_y, max_x, max_y), set while the AE is on. Default: the centre
    # region (middle half of each axis, a quarter of the image), the region
    # the device was found with (480, 270, 1440, 810 at 1920x1080) - it
    # meters the image centre, not the marker.
    realsense_color_ae_roi: Optional[tuple[float, float, float, float]] = (
        0.25, 0.25, 0.75, 0.75)
    # Manual IR exposure (microseconds). None = auto. Short values (3000-8000)
    # reduce motion blur on the IR pair -> higher corner-detection success during
    # fast motion; needs enough scene brightness (or realsense_ir_gain).
    realsense_ir_exposure_us: Optional[float] = None
    realsense_ir_gain: Optional[float] = None
    # Colour pixel format delivered by the SDK. "bgr8" is the tracking
    # default; "bgra8" is Qt's native 32-bit layout (QImage.Format_RGB32
    # wraps it without any per-pixel conversion) and is what the overlay
    # live view requests. The tracker core accepts both (4-channel input
    # is converted via COLOR_BGRA2GRAY in C++).
    realsense_color_format: str = "bgr8"
    # When set, applies rs.option.frames_queue_size to every sensor after
    # start (1 = lowest latency: the SDK keeps only the newest frame).
    # None keeps the SDK default — the behaviour of all existing callers.
    realsense_frames_queue_size: Optional[int] = None
    # Colour exposure while the drill is tracked (Live page, evaluation
    # script): "auto" = the colour AE with the options above (the start
    # writes and reads them back as always); "marker" = the marker exposure
    # control of color_exposure.py takes over at the tracking start (AE
    # off, exposure on the x2 ladder up to 7.8 ms and gain set from the
    # marker's white level) and writes the start settings back at the
    # stop. Default "auto" until the marker control has passed its live
    # test; the Live page can switch it per run.
    realsense_color_exposure_mode: str = "auto"

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
        ae_roi = getattr(self, "realsense_color_ae_roi", None)
        if ae_roi is not None:
            box = tuple(float(v) for v in ae_roi)
            if len(box) != 4 or not (
                0.0 <= box[0] < box[2] <= 1.0 and 0.0 <= box[1] < box[3] <= 1.0
            ):
                raise ValueError(
                    "camera realsense_color_ae_roi must be (min_x, min_y, max_x, "
                    f"max_y) as fractions of the image in 0..1; got {ae_roi!r}."
                )
        color_format = getattr(self, "realsense_color_format", "bgr8")
        if str(color_format).strip().lower() not in ("bgr8", "bgra8"):
            raise ValueError(
                f"camera realsense_color_format must be 'bgr8' or 'bgra8'; "
                f"got {color_format!r}."
            )
        queue_size = getattr(self, "realsense_frames_queue_size", None)
        if queue_size is not None and int(queue_size) <= 0:
            raise ValueError(
                "camera realsense_frames_queue_size must be positive when set; "
                f"got {queue_size}."
            )
        exposure_mode = getattr(self, "realsense_color_exposure_mode", "auto")
        if str(exposure_mode).strip().lower() not in ("auto", "marker"):
            raise ValueError(
                "camera realsense_color_exposure_mode must be 'auto' or 'marker'; "
                f"got {exposure_mode!r}."
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
    # Re-acquisition (phase 4): from a visible, sharp marker the grid stage
    # sometimes builds a wrong lattice - the 45-degree sub-lattice of one
    # corner class (cells = rhombi around corners) or a sheared one (one step
    # along a square diagonal). Nothing decodes, and LK tracking holds that
    # grid until the episode ends (B1 run 2: 36 frames). A re-entry frame (no
    # accepted pose in the previous frame) whose LK-tracked grid decodes no
    # patch and is wrong resets the detector's tracking, so the next frame
    # searches anew; a fresh full detection is kept for one frame (resetting it
    # too lost a whole board take, G07). Each reset costs a full detection, and
    # a search that keeps finding a wrong lattice would reset every few frames
    # (without a limit BT3 reset 31 times, BT2 22, rep H02 17 times): at most
    # checker_wrong_lattice_reset_max_per_gap resets until the next accepted
    # pose (<= 0: no limit). Replays of 108 takes on top of the occlusion gate
    # and the prior sub-grid, every frame processed: fresh frames 32717 ->
    # 33091, first lock 06.10. D04 1.90 -> 0.37 s, BN1 2.97 -> 1.10 s, rep H02
    # 4.97 -> 0.73 s, BT3 4.14 -> 1.40 s (one replay per take, started at its
    # first frame).
    checker_wrong_lattice_reset: bool = True
    checker_wrong_lattice_reset_max_per_gap: int = 1
    checker_min_fresh_correspondences_for_stable_tracking: int = 8
    checker_max_low_fresh_correspondence_frames: int = 12
    # Measurement operator for LK-tracked corners: "quadratic_form" =
    # reference-free curve-intersection refinement, "subpix" = cornerSubPix snap
    # (also the fallback for unknown values / corners QF cannot measure).
    #
    # "subpix" since 2026-07-29. The curve-intersection refinement was measured
    # against subpix on the RGB path and showed no accuracy gain: on the fb run
    # (board frame, common origin) subpix reached p95|x| 0.91 vs 1.07 mm and
    # still-plateau noise 0.055 vs 0.079 mm, i.e. equal or better, and on a
    # static drill frame QF moved corners ~0.5-1.6 px AWAY from the position
    # that subpix and the projected model anchor agree on (anchor sat 0.1-0.5 px
    # from subpix). Consistent with the earlier tip-error finding (subpix 1.05
    # vs model_warp 1.07 mm): the corner operator was never the dominant error.
    # The QF code stays in place (it is still selectable) but is no longer used.
    checker_tracked_refine_method: str = "subpix"

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

    # Quadrant tests of the refined corners (recovery, local completion,
    # guided refine): boxes on the bisectors of the measured edge lines around
    # the sub-pixel corner instead of axis-aligned boxes (which fail on a grid
    # rotated by ~45 deg). Only with appearance_check_enabled (the extra
    # candidates need the check); False = the axis-aligned tests. The
    # refiner's diagonal tolerance of the oriented test is relative to the
    # local range (0-255 scale); the detector keeps its absolute 60 (C++).
    checker_quadrant_oriented: bool = True
    checker_quadrant_refiner_max_diagonal_diff: float = 120.0

    # Pose-filter distrust for frames where the IR MAP fusion attempted and
    # REJECTED (references existed, gates failed: glare veil, adverse
    # geometry): the surviving RGB-only pose is known-degraded evidence, so
    # its measurement covariance is inflated by this factor (the filter
    # coasts on its motion model through the episode). C++ struct default
    # is 1.0 (off) so acceptance replays stay bit-identical.
    ir_reject_cov_inflate: float = 25.0

    # IR corner measurement operator: "quadratic_form" = reference-free direct
    # measurement of the projected model grid curves in each IR view (no
    # reference library, no enrollment, works at any orientation from frame 1);
    # "subpix" = plain cornerSubPix snap on the rig-transferred seeds. Any other
    # value disables the IR fusion (RGB pose passes through).
    #
    # "subpix" since 2026-07-29, for consistency with the RGB path. Requires the
    # re-calibrated ir_fit_gate_rms_mm / ir_cov_inflate_ref_rms_mm below (both
    # compare against fit_rms and are operator-bound) AND a marker model whose
    # IR post-pass was built with subpix (refine_model_ir.py --corner-method
    # subpix -> *_ir3d_subpix.json; the QF-built model made subpix look 1.1 mm
    # worse in z purely by self-consistency). HONEST LIMIT: with all of that in
    # place subpix matches quadratic_form on systematic error and lateral noise
    # and beats it laterally, but its z RANDOM noise stays ~2x higher (fb board
    # frame: p95|z| 1.05 vs 0.55 mm) -- z is the weakly observable direction on
    # the cylinder, where the finer measurement on the ~11 px IR cells pays off.
    #
    # Back to "quadratic_form" on 2026-09-28: subpix only works on large IR
    # cells. Its cornerSubPix window (9x9 px, hard-wired in
    # IrPoseRefiner::measureView) reaches into the neighbouring cells once a
    # 6 mm cell is only ~7-8.5 px in the IR image (drill beyond ~45 cm). The
    # resulting 1-3 px disparity outliers push fit_rms to 3.4-3.9 mm > gate,
    # so live the fusion applied on only 2-4 % of frames (probe sessions
    # tests/output/tool_tracking_eval/20260928_*). Replayed offline with
    # quadratic_form, gates and model unchanged: 96-100 % applied at
    # 315-535 mm, fit 1.3-2.1 mm (workflow wf_8db0eb66-6d6, scratchpad ir_debug).
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
    # Innovation gate (squared Mahalanobis distance): a frame beyond it does
    # not update the filter (its pose is reported as measured);
    # pose_kf_reinit_frames in a row are motion and restart the filter there.
    pose_kf_gate_mahalanobis: float = 30.0
    pose_kf_reinit_frames: int = 2
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
    # Time base (tracker.process_frame(t_frame=...)): one model step is
    # dt / pose_kf_frame_period_s, so camera frames skipped while the tracker
    # was busy count as elapsed time; a gap above pose_kf_max_gap_s restarts
    # the filter. Without frame times one step per processed frame.
    pose_kf_frame_period_s: float = 1.0 / 30.0
    pose_kf_max_gap_s: float = 0.5
    # Bumpless fused <-> RGB-only switching: where the IR fusion starts or
    # stops applying, the filter shifts its measurement by the fused-minus-
    # RGB-only offset (a few mm along the weak depth mode) so the drill does
    # not jump, and lets the shift decay with this time constant (s); 0 off.
    # A fusion rejected by ir_max_reproj_increase_px drops the shift.
    # Replays (20 switch takes, 15 board takes; 0 vs 0.1 s): fusion switches
    # with a tip jump > 2 mm 117 -> 49 of 877, board lateral RMS 1.229 ->
    # 1.255 mm, RMS distance of the drawn from the unfiltered tip in fast
    # motion (M01-M04, median) 1.38 -> 2.28 mm.
    pose_kf_switch_offset_time_s: float = 0.1
    # IR depth-only stereo fusion (output-only). Fuses absolute depth and the
    # two out-of-plane tilt DoF from the global-shutter IR stereo pair into
    # the reported pose; RGB keeps rotation and lateral position (full IR
    # pose replacement is WORSE at the tip: stereo rotation noise times the
    # tool lever). Multi-reference IR measurement (>= min_ref_rot_deg
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
    # Half width (px) of the window around each IR seed that must stay below
    # ir_sat_threshold: 2 = the corner's own junction. 8 (17x17 px, ~1.5 IR
    # cells) also took in the neighbouring white cells, which clip first in
    # bright IR, and blocked the fusion with the corner itself unclipped.
    ir_sat_half_px: int = 2
    # Half width (px) of the window behind ir_saturated_frac, the clipped
    # share of the marker corners that run_tracker's IR exposure controller
    # drives down: 8 as before the gate window was narrowed, so the clipped
    # halo of the white cells around a corner still counts there.
    ir_sat_frac_half_px: int = 8
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
    # IR -> RGB time-offset correction (engine): the IR pair is exposed at
    # another time than the rolling-shutter colour rows (row y after row 0 by
    # ir_time_offset_row_ms * y); under motion the engine moves the IR seeds
    # and 3D points to the colour time of each corner, with the frame-median
    # colour image velocity, above ir_time_offset_deadband_px per frame.
    # HydraTracker.ir_frame_timing gives the engine the frame time and the
    # IR offset of colour row 0, (E_ir - E_rgb) / 2 from the exposures: the
    # IR exposure of the frame, the colour exposure the camera was told to
    # use. The colour AE reports none; it runs as
    # ir_time_offset_ae_exposure_ms. Measured on the 30.09./05.10. debug
    # recordings: 27.8 us per row; 7.0 ms at row 540 with the AE and IR
    # 8 ms, so the AE counts as 24.0 ms. Dead band 3.0 px per frame:
    # below it the shift is at most a few tenths of a pixel and the
    # velocity noise costs more than the correction gains (board takes,
    # hand held in a seat: 1 px moved the drawn tip 0.041 mm further from
    # the seat, 3.0 px 0.014 mm; fast motion fused the same).
    ir_time_offset_enabled: bool = True
    ir_time_offset_row_ms: float = 0.0278
    ir_time_offset_deadband_px: float = 3.0
    ir_time_offset_ae_exposure_ms: float = 24.0
    ir_sigma_ir_px: float = 0.05
    # IR/reproj balance: 1 = still-frame sigma calibration; 4 compensates the
    # correlated (non-IID) RGB corner errors during motion (weak-mode wobble).
    ir_w3d: float = 4.0
    # Gate calibration 2026-07-22 (fb1_22 glare window): monocular RGB z drifts
    # 3-4.5mm while stereo fit stays 1.5-1.65, so the old 1.5/3.0 gates rejected
    # the fusion exactly when it was needed most (z spikes to -4.4). 1.8/6.0
    # validated across fb1_22/fb4/fb5/fb1_21/fb_rl (p95|z| 10.8->2.4) with only
    # a small robust-std cost on fb_rl2 (0.87->1.15); divot ABS unchanged.
    #
    # RE-CALIBRATED 2026-07-29 for the subpix corner measurement. Both values
    # below are SCALE-BOUND TO THE OPERATOR: they compare against fit_rms, whose
    # distribution differs per operator (fb run: quadratic_form median 0.80 /
    # max 1.41, subpix median 1.83 / max 2.50). The old 1.8 therefore never
    # fired for quadratic_form but rejected 52% of the subpix frames, and the
    # old 0.10 reference inflated the surviving subpix frames ~5x harder
    # ((1.83/0.10)^2 = 335 vs (0.80/0.10)^2 = 64), damping the fusion the
    # filter was supposed to use. Percentile-equivalent gate = 2.5, median-ratio
    # reference = 0.23; that pair restores 100% fusion coverage AND lands the
    # median cov_scale on 63.3, exactly the quadratic_form value. Validated on
    # the fb run in the board frame: return-to-origin dz +0.35 -> -0.04 mm and
    # lateral still-noise 0.048 mm (best of all arms). ==> IF THE IR OPERATOR
    # CHANGES, RE-DERIVE BOTH from its fit_rms distribution.
    ir_fit_gate_rms_mm: float = 2.5
    ir_fit_gate_max_trans_jump_mm: float = 6.0
    # Near-range safety net: reject a fusion that raises the RMS reprojection
    # of the pose's own RGB corners by more than this (px) over the RGB-only
    # pose; the engine then reports the RGB-only pose. Under motion the gate
    # is scaled by the shear compensation's inflation sigma_px_eff /
    # ir_sigma_px (at least 1): there the MAP is meant to leave the sheared
    # RGB image. timings_ms: ir_reproj_increase_px and ir_reproj_gate_px on
    # every fusion, ir_reproj_gate_reject.
    # Replays: at 221 mm marker distance (30.09. G02) it rejects all 44
    # fusions (fused tip 2.8 mm / 11.5 px from the RGB-only tip), at 237 mm
    # (A06) 14 of 358; on the fast-motion takes (05.10. M01-M04) 13 of 775
    # (256 without the motion scaling); elsewhere 0-0.12 %.
    # 0 disables.
    ir_max_reproj_increase_px: float = 2.0
    # Upper bound (px) of that gate. The motion scaling grows with the image
    # motion between two processed frames, i.e. with every frame the tracking
    # thread drops: live-paced replays reached 1181 px, where the
    # gate no longer judges anything. <= 0: no bound.
    ir_reproj_gate_max_px: float = 10.0
    # Weak-evidence honesty: inflate the MAP covariance handed to the temporal
    # filter by the reduced chi-square (fit_rms / this)^2 so the filter damps
    # poor-fit frames instead of following them. 0 disables.
    # Operator-bound (quadratic_form here). 0.36 makes the covariance
    # in the filter's chart consistent with the real scatter of the fused pose
    # of the fixed drill (30.09. G01 G04 G05 G09, 700 fused frames, fit
    # RMS median 0.72 mm: mean NEES 6.0; the former 0.23 gave 2.5).
    ir_cov_inflate_ref_rms_mm: float = 0.36
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
    pose_hold_min_detection_corners: int = 8
    emergency_pose_hold_enabled: bool = True
    emergency_pose_hold_max_frames: int = -1
    fallback_pose_min_detection_matches: int = 8
    fallback_pose_max_median_corner_error_px: float = 9.0
    fallback_pose_max_p90_corner_error_px: float = 18.0
    fallback_pose_max_mean_reprojection_error_px: float = 1.8
    fallback_pose_max_max_reprojection_error_px: float = 4.0

    # Appearance check
    # Photometric check of a pose, independent of the decode that produced it:
    # the projected marker cells must show the expected checker parity and dot
    # pattern (correlations, 1 = perfect). Applied to every (re)acquisition
    # pose of the decode path; a rejected pose counts as a failed frame.
    appearance_check_enabled: bool = True
    appearance_min_parity: float = 0.6
    appearance_min_dot: float = 0.6
    # Below appearance_min_cells sampled cells the check cannot decide. The
    # dot term also needs 8 cells of each base colour (C++), so a check
    # passes only with at least 16 cells (8 + 8).
    appearance_min_cells: int = 6
    # Also check every fast / persistent tracking pose (they carry the corner
    # identities over and never re-decode); a rejected fast pose falls back to
    # the decode path, a rejected persistent pose to the hold. A tracking pose
    # is only rejected on evidence (enough cells, a correlation below the
    # minimum); an acquisition pose has to pass, and so does every pose of a
    # frame that re-enters tracking (the previous frame has no accepted pose)
    # or whose fast pose failed the check. Those poses are checked even with
    # appearance_check_tracking = False.
    appearance_check_tracking: bool = True
    # Occlusion gate of the TRACKING checks (fast / persistent poses and the
    # prior sub-grid pose): a cell counts only where it shows checker
    # contrast at its projected place (|off-centre mean - median of its
    # sampled 4-neighbours| >= max(appearance_gate_min_dn,
    # appearance_gate_frac * p90 of those differences)). A finger over the
    # marker then drops out of both correlations instead of failing a
    # correct pose (05.10. live_check run_05 / 30.09. S05: the true pose
    # failed the ungated check on 183 of 1214 / 88 of 300 frames); identity
    # shifts still fail (0 of 16155 shifted poses passed). Not on the decode
    # (re)acquisition check (appearance_gate_acquisition): gated there, a
    # partial-grid decode pose 70 deg off passed (run_05 k534-536).
    appearance_gate_min_dn: float = 12.0
    appearance_gate_frac: float = 0.25
    appearance_gate_acquisition: bool = False
    # Prior-seeded sub-grid fallback: on a frame without a fresh pose, up to
    # prior_subgrid_max_age_frames after the last decode / fast / persistent
    # pose (a sub-grid pose seeds the next one but does not extend the window),
    # refine the model corners projected with the last accepted pose
    # (cornerSubPix, at most prior_subgrid_search_frac of a cell away, local
    # checker contrast >= prior_subgrid_min_contrast DN), solve PnP around it
    # (motion gate prior_subgrid_max_rot_deg / _max_trans_mm) and accept the
    # pose only if the gated appearance check PASSES. With it on, every decode
    # pose is checked, also in tracking mode. Replays with the Live page's
    # config and reset, every frame processed: 06.10. O02 (two fingers over
    # half of the label) 72 -> 292 of 301 frames with a pose; 108 takes 31960
    # -> 32717 fresh frames (891 gained, 134 lost), poses failing both
    # photometric checks 10 -> 3, none of the gained poses with a shifted
    # identity (label-shift test of the RGB-only pose). The motion gate was 8
    # deg / 15 mm: two sub-grid poses seeded by a decode pose with few corners
    # (7 and 10) passed it with the tip 29 mm from the next fresh pose (07.10.
    # live check k4049 right after a re-lock, 6.8 deg; 08.10. B8 table k2690,
    # 12.6 mm). The other sub-grid poses moved at most 4.2 deg / 8.7 mm (77
    # poses, 15 live runs of 07./08.10.) and 4.9 deg / 7.9 mm (175 poses, the
    # 108 replayed takes, which 6 deg / 10 mm therefore leaves unchanged).
    # Replays of the two runs with 6 deg / 10 mm: both wrong poses gone, 6780
    # -> 6775 fresh frames (3 held after k4049), every other pose within 0.8
    # mm of the 8 deg / 15 mm replay; two other runs (20 sub-grid poses)
    # identical.
    prior_subgrid_enabled: bool = True
    prior_subgrid_max_age_frames: int = 15
    prior_subgrid_min_points: int = 10
    prior_subgrid_search_frac: float = 0.35
    prior_subgrid_min_contrast: float = 20.0
    prior_subgrid_max_rot_deg: float = 6.0
    prior_subgrid_max_trans_mm: float = 10.0
    # Continuity guard of the gated tracking poses (mode 2 = every pose checked
    # with the gate, 1 = only those the ungated check rejects, 0 = off): such a
    # pose is rejected when it is more than occlusion_guard_max_rot_deg or
    # occlusion_guard_max_trans_mm (marker origin) from the last accepted pose.
    # Without it the gate let a persistent pose 42.5 deg / 31.6 mm off
    # through in fast motion (05.10. M04 k35). In the 108 replayed takes it
    # rejected 4 poses (17-43 deg), which removed 9 jump frames; the largest
    # deviation it accepted was 14.7 deg / 23.9 mm.
    occlusion_guard_mode: int = 2
    occlusion_guard_max_rot_deg: float = 15.0
    occlusion_guard_max_trans_mm: float = 25.0

    # Logging Output
    # Controls the corner list that TrackerEngine returns to Python.
    # tracker_log.py records these values but does not make tracking decisions.
    visual_corner_max_reprojection_error_px: float = 3.0
    visual_corner_min_count: int = 6

    # Python-side only (the engine ignores it): aggregated dropout logging in
    # the overlay live page. Records the cause of every tracking loss AFTER a
    # successful lock, deduplicated per cause with a counter, written once
    # when tracking stops. No effect on the frame loop.
    failure_log_enabled: bool = True

    # Python-side only (the engine ignores it): the Live page draws a pose
    # whose visible_cell_fraction is below this value MARKED (other colour): a
    # pose the engine kept with part of the marker covered. The tip error
    # rises step by step as the fraction falls, with no clear knee above 0.5:
    # label painted over in the image, the same frames painted vs unpainted
    # (6 takes): 0.70-0.75 x1.34 (+0.32 mm), 0.50-0.55 x1.63 (+0.63 mm),
    # 0.40-0.50 x1.75, below 0.40 x3.1 (+2.0 mm). A finger at 0.72 is
    # already less exact, so it is marked: 07.10. live check, drill held
    # still, jitter 0.59 mm (RMS per second) against 0.21 mm without the
    # finger, drawn tip 1.3 mm (depth) off where it sat once the finger was
    # lifted; 06.10. hand-held takes (median fraction / drawn tip error, board
    # ground truth): unoccluded D01-D05 1.00 / 1.12 mm, two fingers over half
    # (O02) 0.72 / 1.78 mm (marked), one finger over a third (O01) 0.78 /
    # 1.63 mm (drawn unmarked: not every less exact pose is flagged).
    # 0 = never marked.
    drawn_pose_mark_below_visible_frac: float = 0.75

    # Drill pose estimator (iterated EKF in the C++ engine, output stage;
    # cpp/include/tracker_config.hpp). Off by default: the engine computes
    # nothing and adds no result fields or timings keys. On: own result fields
    # drill_est_* in TrackerResult (rvec / tvec stay the engine pose), and the
    # engine forces the output pose filter off (pose_kf_enabled reads back
    # False). Not switched on in make_live_tracker_config(): the Live page sets
    # it and its additions itself, so the repository scripts keep the old
    # behaviour. Defaults = the validated frozen state (variant co_alt_eb);
    # gap 0.5 s, prior_max 0 and hide_sig 0 keep it. Same types as in C++
    # (int stays int: tracker_config_from_python drops a mismatch silently,
    # check_drill_est_config finds it). w3d / sigma_ir_px / row_ms are the
    # estimator's own copies of ir_w3d / ir_sigma_ir_px /
    # ir_time_offset_row_ms (s_rgb / s_ir were fitted with these values;
    # drill_est_fingerprint compares them with the engine fields).
    drill_est_enabled: bool = False
    drill_est_sig_t_mm: float = 0.02          # base noise per T0 = 1/30 s [mm]
    drill_est_sig_r_deg: float = 0.008        # base noise per T0 [deg]
    drill_est_adapt_px: float = 1.0           # dead band of nu1 [px/frame]
    drill_est_db3_px: float = 0.3             # dead band of nu3 [px/frame]
    drill_est_gain_t: float = 2.0             # gain_t * nu * Z / f in mm
    drill_est_gain_r_deg: float = 0.5         # [deg per px/frame]
    drill_est_vmin: int = 4                   # matched corners for nu
    drill_est_s_rgb: float = 7.57             # RGB sigma scale
    drill_est_s_ir: float = 11.08             # IR sigma scale
    drill_est_huber_k: float = 1.5            # Huber threshold
    drill_est_n_iter: int = 3                 # Gauss-Newton steps
    drill_est_gap_s: float = 0.5              # chain break after a gap [s]
    drill_est_rs_px: float = 3.0              # rolling-shutter dead band; < 0: off
    drill_est_ir_gated: float = 1.0           # sigma factor, IR of a not applied fusion
    drill_est_ir_fit_max_mm: float = 6.0      # IR switch-off above the fit RMS; 0: off
    drill_est_ir_fit_fac: float = 1000.0      # sigma factor of switched-off IR pairs
    drill_est_min_start: int = 12             # RGB corners for a chain start
    drill_est_warm: int = 5                   # start-up gate at most [frames]
    drill_est_sig_tip_max_mm: float = 2.0     # start-up gate ends at sigma_tip <=
    drill_est_w3d: float = 4.0                # copy of ir_w3d
    drill_est_sigma_ir_px: float = 0.05       # copy of ir_sigma_ir_px
    drill_est_row_ms: float = 0.0278          # copy of ir_time_offset_row_ms
    drill_est_prior_max_mm: float = 0.0       # chain restart above the prior sigma; 0: off
    drill_est_hide_sig_mm: float = 0.0        # hide frames with sigma_tip above; 0: off


# ---- Drill pose estimator: read-back and validation checks ----
# Used only while the estimator is switched on (Live page, tool eval), not in
# tracker_config_from_python: with an older module every script would warn.
DRILL_EST_FIELDS = tuple(
    f.name for f in fields(TrackerConfig) if f.name.startswith("drill_est_")
)
# Engine values the scales s_rgb / s_ir of the estimator were validated with
# (meta of the co_base dumps; sigma_px_eff = 0.100 px at rest) and the IR
# calibration file of that validation (realsense_ir_calibration.npz).
DRILL_EST_VALIDATION = {
    "ir_sigma_px": 0.10,
    "ir_sigma_px_motion_coeff": 0.20,
    "ir_sigma_px_motion_rot_px_per_deg": 8.0,
    "ir_depth_scale": 1.00197,
    "ir_min_pairs": 6,
    "ir_sigma_ir_px": 0.05,
    "ir_w3d": 4.0,
    "ir_time_offset_row_ms": 0.0278,
}
DRILL_EST_VALIDATION_IRCAL_MD5 = "e0c6420a7bf84be6ca1069695736be4d"
# the estimator's own copies of engine fields (copy, engine field)
DRILL_EST_COPIES = (
    ("drill_est_w3d", "ir_w3d"),
    ("drill_est_sigma_ir_px", "ir_sigma_ir_px"),
    ("drill_est_row_ms", "ir_time_offset_row_ms"),
)
DRILL_EST_KF_WARNING = "Ausgabe-KF aus, weil Schaetzer aktiv"


def _same_config_value(have, want) -> bool:
    """Equal as the engine holds it: bool and int fields with the same type
    (a float for an int field is not taken over), float fields by value."""
    if isinstance(have, bool) or isinstance(want, bool):
        return type(have) is type(want) and have == want
    if isinstance(have, int):
        return type(want) is int and have == want
    try:
        return float(have) == float(want)
    except (TypeError, ValueError):
        return False


def check_drill_est_config(engine_cfg, py_cfg) -> list:
    """Read-back check of the drill estimator configuration: every drill_est_*
    field of the engine's configuration (HydraTracker._engine.config, the
    copy the engine runs with) against the Python value. A field the loaded
    module lacks, or a value it did not take (wrong type), is a deviation.
    The output pose filter is compared too: while the estimator is on the
    engine forces it off, so a requested filter that reads back off is not a
    deviation (drill_est_kf_warning gives the warning line). Returns the
    deviations as text lines, empty = all taken over."""
    dev = []
    for name in DRILL_EST_FIELDS:
        want = getattr(py_cfg, name)
        if not hasattr(engine_cfg, name):
            dev.append(f"{name}: not in the loaded module")
            continue
        have = getattr(engine_cfg, name)
        if not _same_config_value(have, want):
            dev.append(f"{name}: engine {have!r} != python {want!r}")
    want_kf = getattr(py_cfg, "pose_kf_enabled", None)
    have_kf = getattr(engine_cfg, "pose_kf_enabled", None)
    if want_kf is not None and have_kf is not None and bool(have_kf) != bool(want_kf):
        forced = (
            bool(want_kf)
            and not bool(have_kf)
            and bool(getattr(engine_cfg, "drill_est_enabled", False))
        )
        if not forced:
            dev.append(f"pose_kf_enabled: engine {have_kf!r} != python {want_kf!r}")
    return dev


def drill_est_kf_warning(engine_cfg, py_cfg) -> Optional[str]:
    """Warning line when the output pose filter was requested but the engine
    runs it off because the drill estimator is on; None otherwise."""
    if (
        bool(getattr(py_cfg, "pose_kf_enabled", False))
        and bool(getattr(engine_cfg, "drill_est_enabled", False))
        and not bool(getattr(engine_cfg, "pose_kf_enabled", True))
    ):
        return DRILL_EST_KF_WARNING
    return None


def drill_est_fingerprint(engine_cfg, ircal_path=None) -> list:
    """Validation fingerprint of the drill estimator (only with it on): the
    engine values its scales depend on but it cannot copy, the IR calibration
    file (MD5) and the equality of its three copies with the engine fields.
    Returns the deviations from the validated state as text lines (a warning:
    the estimator runs anyway)."""
    dev = []
    for name, ref in DRILL_EST_VALIDATION.items():
        have = getattr(engine_cfg, name, None)
        if have is None or not _same_config_value(have, ref):
            dev.append(f"{name}: {have!r} (validated {ref!r})")
    for copy_name, engine_name in DRILL_EST_COPIES:
        a = getattr(engine_cfg, copy_name, None)
        b = getattr(engine_cfg, engine_name, None)
        if a is None or b is None or float(a) != float(b):
            dev.append(f"{copy_name} {a!r} != {engine_name} {b!r}")
    if ircal_path is None:
        dev.append("IR calibration: no file")
    else:
        try:
            digest = hashlib.md5(Path(ircal_path).read_bytes()).hexdigest()
        except OSError as exc:
            dev.append(f"IR calibration {ircal_path}: {exc}")
        else:
            if digest != DRILL_EST_VALIDATION_IRCAL_MD5:
                dev.append(
                    f"IR calibration {ircal_path}: MD5 {digest} "
                    f"(validated {DRILL_EST_VALIDATION_IRCAL_MD5})"
                )
    return dev


# ---- Drill pose estimator on the Live page and in the tool evaluation ----
# The switch: LIVE_DRILL_ESTIMATOR, read (drill_est_requested) whenever the
# page (or test_tracker_tool_eval) builds its tracker. True (default): the
# estimator with the Live additions (DRILL_EST_LIVE) and the start checks
# S1-S8; an engine without it (S1-S4 failed) tracks as before, with a red
# message. False: the tracker exactly as before the estimator (no start
# checks, no tool tip). make_live_tracker_config() and the TrackerConfig
# default (drill_est_enabled False) never set it.
LIVE_DRILL_ESTIMATOR = True
# The additions the Live page runs the estimator with (validated on TUNE):
# a chain break after 0.2 s instead of 0.5 s, the prior bound 20 mm and frames
# hidden above sigma_tip 5 mm. The TrackerConfig defaults keep the frozen state.
DRILL_EST_LIVE = (
    ("drill_est_gap_s", 0.2),
    ("drill_est_prior_max_mm", 20.0),
    ("drill_est_hide_sig_mm", 5.0),
)
# Start check S7: K_rgb of the GUI against K_rgb of the IR calibration file [px].
DRILL_EST_K_TOL_PX = 1e-6


def drill_est_requested() -> bool:
    """The drill estimator is asked for (Live page, test_tracker_tool_eval):
    LIVE_DRILL_ESTIMATOR, read at every call (a test may set it)."""
    return bool(LIVE_DRILL_ESTIMATOR)


def apply_drill_est_live(config):
    """Switch the estimator on in a TrackerConfig, with the Live additions
    (DRILL_EST_LIVE); returns the config."""
    config.drill_est_enabled = True
    for name, value in DRILL_EST_LIVE:
        setattr(config, name, value)
    return config


def drill_est_start_checks(module, engine, py_cfg, *, tip_set, ir_path=None,
                           K_rgb=None) -> dict:
    """Start checks of the drill estimator (only with it requested), against
    the loaded module (hydramarker_cpp), the engine (TrackerEngine) and the
    Python config it was built from:
      S1 the module has the estimator (TrackerConfig field, BUILD_INFO)
      S2 read-back: every drill_est_* of the engine equals the Python value
      S3 the engine's parameter check passed (drill_est_config_valid)
      S4 the tool tip is set (tip_set: set_tool_tip and has_tool_tip)
      S5 IR on (ir_refine_enabled and an IR calibration file)
      S6 validation fingerprint (drill_est_fingerprint)
      S7 K_rgb of the GUI = K_rgb of the IR calibration file
      S8 a release build (BUILD_INFO ndebug)
    S1-S4 failed: the caller builds the tracker with the estimator off; S5-S8
    failed: a warning, the estimator runs. Returns {"checks": [{id, ok (None:
    not checked), fatal, text}], "fatal": [...], "warnings": [...],
    "readback": [...], "fingerprint": [...]} (texts "S<n> ...")."""
    checks = []

    def add(cid: str, ok, fatal: bool, text: str) -> None:
        checks.append({"id": cid, "ok": ok, "fatal": fatal, "text": text})

    ir_path = ir_path or None
    engine_cfg = getattr(engine, "config", None)
    try:
        has_field = hasattr(module.TrackerConfig(), "drill_est_enabled")
    except Exception:
        has_field = False
    build = getattr(module, "BUILD_INFO", None)
    build = {} if build is None else dict(build)
    ok = has_field and bool(build.get("drill_est", False))
    add("S1", ok, True, "estimator in the loaded module" if ok else
        f"the loaded module has no drill estimator (TrackerConfig field "
        f"{'yes' if has_field else 'missing'}, BUILD_INFO drill_est {build.get('drill_est')!r})")

    if engine_cfg is None:
        readback = ["the engine has no configuration"]
    else:
        readback = check_drill_est_config(engine_cfg, py_cfg)
    more = f" (+{len(readback) - 3} more)" if len(readback) > 3 else ""
    add("S2", not readback, True, "read-back of drill_est_* complete" if not readback
        else "read-back: " + "; ".join(readback[:3]) + more)

    ok = bool(getattr(engine, "drill_est_config_valid", False))
    add("S3", ok, True, "parameters valid" if ok else
        "parameters rejected by the engine (or no parameter check in the module)")

    add("S4", bool(tip_set), True, "tool tip set" if tip_set else "tool tip not set")

    ok = bool(getattr(py_cfg, "ir_refine_enabled", False)) and bool(ir_path)
    add("S5", ok, False, "IR on" if ok else
        "IR off: the estimator runs on the colour corners only (weakest case)")

    if engine_cfg is None:
        fingerprint = ["the engine has no configuration"]
    else:
        fingerprint = drill_est_fingerprint(engine_cfg, ir_path)
    add("S6", not fingerprint, False, "validation fingerprint unchanged" if not fingerprint
        else "fingerprint: " + "; ".join(fingerprint))

    if not ir_path or K_rgb is None:
        add("S7", None, False, "K_rgb not compared (no IR calibration file)"
            if not ir_path else "K_rgb not compared (no GUI intrinsics)")
    else:
        try:
            import numpy as np

            with np.load(ir_path, allow_pickle=False) as z:
                k_ir = np.asarray(z["K_rgb"], dtype=np.float64).reshape(3, 3)
            k_gui = np.asarray(K_rgb, dtype=np.float64).reshape(3, 3)
            d = float(np.max(np.abs(k_gui - k_ir)))
        except Exception as exc:
            add("S7", False, False, f"K_rgb of the IR calibration not compared: {exc!r}")
        else:
            ok = d <= DRILL_EST_K_TOL_PX
            add("S7", ok, False, "K_rgb of the GUI = K_rgb of the IR calibration" if ok else
                f"K_rgb of the GUI differs from the IR calibration's by up to {d:.6g} px")

    ndebug = build.get("ndebug")
    add("S8", bool(ndebug), False, "release build" if ndebug else
        f"not a release build (BUILD_INFO ndebug {ndebug!r}): timings not usable")

    return {
        "checks": checks,
        "fatal": [f"{c['id']} {c['text']}" for c in checks if c["fatal"] and not c["ok"]],
        "warnings": [f"{c['id']} {c['text']}" for c in checks
                     if not c["fatal"] and c["ok"] is False],
        "readback": readback,
        "fingerprint": fingerprint,
    }


def drill_est_summary(requested: bool, active: bool, report: Optional[dict],
                      engine_cfg=None) -> str:
    """One line on the estimator at a tracking start: on (with the Live
    additions) or off and why, then the warnings of the start checks.
    engine_cfg: the configuration read back from the engine; the additions
    are then its values (None: the DRILL_EST_LIVE values)."""
    if not requested:
        return "drill estimator off (LIVE_DRILL_ESTIMATOR False)"
    fatal = (report or {}).get("fatal") or []
    if not active:
        why = "; ".join(fatal) if fatal else "not active in the engine"
        return f"drill estimator OFF although requested: {why}"
    extra = ", ".join(
        f"{name[len('drill_est_'):]} "
        f"{float(value if engine_cfg is None else getattr(engine_cfg, name, float('nan'))):g}"
        for name, value in DRILL_EST_LIVE
    )
    text = f"drill estimator on ({extra})"
    warnings = (report or {}).get("warnings") or []
    if warnings:
        text += "; warnings: " + "; ".join(warnings)
    return text


__all__ = [
    "CAMERA_BACKENDS",
    "LOGGING_START_MODES",
    "MAX_CAMERA_FPS",
    "CameraConfig",
    "LiveTrackerConfig",
    "LoggingConfig",
    "TrackerConfig",
    "DRILL_EST_FIELDS",
    "DRILL_EST_LIVE",
    "LIVE_DRILL_ESTIMATOR",
    "apply_drill_est_live",
    "check_drill_est_config",
    "drill_est_fingerprint",
    "drill_est_kf_warning",
    "drill_est_requested",
    "drill_est_start_checks",
    "drill_est_summary",
]
