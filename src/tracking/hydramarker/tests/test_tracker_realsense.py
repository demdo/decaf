from __future__ import annotations

import os
import time
from pathlib import Path
import sys
from typing import Any, Callable, Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication, QFileDialog


def _ensure_src_on_path() -> None:
    src_root = Path(__file__).resolve().parents[3]
    src = str(src_root)
    if src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()

from tracking.hydramarker import tracker_log
from tracking.hydramarker.backend import cpp_impl as hydramarker_cpp
from tracking.hydramarker.config import TrackerConfig
from tracking.hydramarker.tracker import HydraTracker


# ============================================================
# RealSense Live Runner / Diagnostics
# ============================================================

DISTORTION_MODE_ENV = "HYDRAMARKER_DISTORTION_MODE"
APP_IDLE = "IDLE"
APP_ACQUIRE = "ACQUIRE"
APP_PROVISIONAL = "PROVISIONAL"
APP_TRACKING = "TRACKING"

# 30 Hz camera: keep these budgets long enough for a bad-but-visible cold
# start, while still returning to a cheap video-only idle state automatically.
ACQUIRE_TIMEOUT_FRAMES = 90
PROVISIONAL_MIN_CORNERS = 6
PROVISIONAL_STALE_TIMEOUT_FRAMES = 30
PROVISIONAL_TOTAL_TIMEOUT_FRAMES = 180
TRACKING_STALE_TO_IDLE_FRAMES = 45
IDLE_PREVIEW_DET_WIDTH = 960
IDLE_PREVIEW_AUTO_ACQUIRE_CORNERS = PROVISIONAL_MIN_CORNERS



def _first_npz_array(npz: Any, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in npz:
            return np.asarray(npz[name], dtype=np.float64)
    return None


def _read_calibration_image_size(npz: Any) -> list[int] | None:
    if "image_size" in npz:
        values = np.asarray(npz["image_size"]).reshape(-1)
        if values.size >= 2:
            return [int(values[0]), int(values[1])]

    width_keys = ("width", "image_width", "rgb_width")
    height_keys = ("height", "image_height", "rgb_height")
    width = next((int(np.asarray(npz[k]).reshape(-1)[0]) for k in width_keys if k in npz), None)
    height = next((int(np.asarray(npz[k]).reshape(-1)[0]) for k in height_keys if k in npz), None)
    if width is not None and height is not None:
        return [width, height]

    return None


def load_opencv_camera_calibration_from_file(
    path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if path is None:
        path = choose_file_qt(
            "Select camera calibration .npz",
            "NPZ files (*.npz)",
        )
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Camera calibration file not found: {path}")

    with np.load(path, allow_pickle=True) as npz:
        K = _first_npz_array(
            npz,
            ("K", "K_rgb", "camera_matrix", "camera_intrinsics", "intrinsics"),
        )
        if K is None:
            raise KeyError(
                "Camera calibration NPZ must contain one of: "
                "K, K_rgb, camera_matrix, camera_intrinsics, intrinsics."
            )

        dist = _first_npz_array(
            npz,
            (
                "dist",
                "dist_rgb",
                "dist_coeffs",
                "distortion_coeffs",
                "opencv_dist_coeffs",
                "effective_opencv_dist_coeffs",
            ),
        )
        if dist is None:
            raise KeyError(
                "Camera calibration NPZ must contain OpenCV distortion coefficients: "
                "dist, dist_rgb, dist_coeffs, distortion_coeffs, opencv_dist_coeffs, "
                "or effective_opencv_dist_coeffs."
            )

        image_size = _read_calibration_image_size(npz)

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    if dist.size == 0:
        raise ValueError("Distortion coefficients must not be empty.")

    info: dict[str, Any] = {
        "camera_source": "opencv_calibration_npz",
        "camera_calibration_path": str(path),
        "distortion_model": "opencv_brown_conrady",
        "K": K.tolist(),
        "opencv_dist_coeffs": dist.reshape(-1).tolist(),
        "effective_opencv_dist_coeffs": dist.reshape(-1).tolist(),
    }
    if image_size is not None:
        info["calibration_image_size"] = image_size

    return K, dist, info


def normalize_distortion_mode(mode: Optional[str]) -> str:
    value = (mode or os.environ.get(DISTORTION_MODE_ENV) or "realsense").strip().lower()
    aliases = {
        "real": "realsense",
        "rs": "realsense",
        "camera": "realsense",
        "on": "realsense",
        "true": "realsense",
        "1": "realsense",
        "none": "zero",
        "off": "zero",
        "false": "zero",
        "0": "zero",
        "no": "zero",
        "undistorted": "zero",
    }
    value = aliases.get(value, value)
    if value not in ("realsense", "zero"):
        print(
            f"[camera_intrinsics] unknown {DISTORTION_MODE_ENV}={mode!r}; "
            "using realsense"
        )
        return "realsense"
    return value




# ============================================================
# Helpers
# ============================================================

def choose_file_qt(title: str, file_filter: str) -> Path:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(None, title, "", file_filter)
    if not path:
        raise RuntimeError(f"No file selected: {title}")

    return Path(path)


def put_text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 255),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def load_tracker_camera_calibration(
    profile,
    distortion_mode: Optional[str] = None,
    calibration_path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    rs_K = np.array(
        [[intr.fx, 0.0, intr.ppx],
         [0.0, intr.fy, intr.ppy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    raw_dist = np.asarray(intr.coeffs[:5], dtype=np.float64).reshape(-1, 1)
    coeffs = [float(c) for c in intr.coeffs]
    model = str(getattr(intr, "model", "unknown"))

    K, dist, calib_info = load_opencv_camera_calibration_from_file(calibration_path)
    stream_size = [int(getattr(intr, "width", 0)), int(getattr(intr, "height", 0))]
    calib_size = calib_info.get("calibration_image_size")
    if calib_size is not None and list(calib_size) != stream_size:
        raise RuntimeError(
            f"Selected calibration image_size={calib_size} does not match "
            f"the active RealSense color stream {stream_size}."
        )

    tracker_log.set_camera_intrinsics_info({
        "width": int(getattr(intr, "width", 0)),
        "height": int(getattr(intr, "height", 0)),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "ppx": float(K[0, 2]),
        "ppy": float(K[1, 2]),
        "model": "opencv_brown_conrady",
        "coeffs": dist.reshape(-1).tolist(),
        "distortion_mode": "opencv_calibration_npz",
        "raw_realsense_model": model,
        "raw_realsense_coeffs": coeffs,
        "raw_realsense_K": rs_K.tolist(),
        "raw_realsense_dist_coeffs": raw_dist.reshape(-1).tolist(),
        **calib_info,
    })
    print(
        "[camera_intrinsics] using selected OpenCV calibration NPZ="
        f"{calib_info['camera_calibration_path']} "
        f"dist={dist.reshape(-1).tolist()}"
    )
    return K, dist


def create_realsense_pipeline():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    return pipe, profile


def make_idle_preview_detector():
    cfg = hydramarker_cpp.CheckerboardDetectorConfig()
    cfg.det_width = IDLE_PREVIEW_DET_WIDTH
    cfg.refresh_interval_frames = 1
    if hasattr(cfg, "max_undecodeable_tracking_frames"):
        cfg.max_undecodeable_tracking_frames = 12
    if hasattr(cfg, "max_low_corner_frames"):
        cfg.max_low_corner_frames = 12
    if hasattr(cfg, "min_tracking_decode_cell_span"):
        cfg.min_tracking_decode_cell_span = 3
    return hydramarker_cpp.CheckerboardDetector(cfg)


def point_xy(p) -> tuple[float, float]:
    if hasattr(p, "x") and hasattr(p, "y"):
        return float(p.x), float(p.y)
    return float(p[0]), float(p[1])


def preview_corner_uvs(detection) -> list[tuple[float, float]]:
    if detection is None:
        return []
    corners = getattr(detection, "corners", []) or []
    pts: list[tuple[float, float]] = []
    for corner in corners:
        try:
            pts.append(point_xy(corner.uv))
        except Exception:
            continue
    return pts


def draw_preview_corners(vis: np.ndarray, corners: list[tuple[float, float]]) -> None:
    for u, v in corners:
        cv2.circle(
            vis,
            (int(round(u)), int(round(v))),
            3,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_detection_corners(vis: np.ndarray, result) -> None:
    for p in getattr(result, "detection_corners", []):
        u = int(round(p.uv[0]))
        v = int(round(p.uv[1]))
        cv2.circle(vis, (u, v), 3, (255, 180, 0), -1, cv2.LINE_AA)


def draw_pose_corners(vis: np.ndarray, result) -> None:
    if not result.success:
        return
    for p in result.corners:
        u = int(round(p.uv[0]))
        v = int(round(p.uv[1]))
        cv2.circle(vis, (u, v), 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(vis, f"{p.global_row},{p.global_col}",
                    (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 255, 0), 1, cv2.LINE_AA)


def draw_reprojection(vis: np.ndarray, result, K, dist) -> None:
    if not result.success or result.rvec is None or result.tvec is None:
        return
    if len(result.corners) == 0:
        return

    object_points = np.asarray([p.xyz_mm for p in result.corners], dtype=np.float64).reshape(-1, 3)
    measured = np.asarray([p.uv for p in result.corners], dtype=np.float64).reshape(-1, 2)

    projected, _ = cv2.projectPoints(
        object_points,
        result.rvec.reshape(3, 1),
        result.tvec.reshape(3, 1),
        K, dist,
    )
    projected = projected.reshape(-1, 2)

    for m, q in zip(measured, projected):
        cv2.circle(vis, (int(round(q[0])), int(round(q[1]))), 3, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.line(vis, (int(round(m[0])), int(round(m[1]))),
                      (int(round(q[0])), int(round(q[1]))), (255, 0, 255), 1, cv2.LINE_AA)


def draw_status(vis: np.ndarray, result, frame_idx: int, tracker: HydraTracker) -> None:
    detection_corners = getattr(result, "detection_corners", [])
    status_color = (0, 255, 0) if result.success else (0, 165, 255)
    failure_stage, failure_reason = tracker_log.classify_failure(result)
    fast = getattr(result, "fast_path_debug", None)

    line1 = (
        f"frame={frame_idx} | {result.mode.value} | ok={result.success} | "
        f"src={getattr(getattr(result, 'pose_source', None), 'value', 'none')} | "
        f"det={len(detection_corners)} | pose={'Y' if result.rvec is not None and result.tvec is not None else 'N'} | "
        f"vis={len(result.corners)} | "
        f"pts={result.num_points} | inl={result.num_inliers} | "
        f"pers={len(tracker._persistent_corners)}"
    )
    line2 = (
        f"mean={result.mean_reprojection_error_px:.3f}px | "
        f"max={result.max_reprojection_error_px:.3f}px | "
        f"conf={result.confidence:.2f} | "
        f"fast={int(bool(getattr(fast, 'attempted', False)))}/"
        f"{int(bool(getattr(fast, 'success', False)))}:"
        f"{int(getattr(fast, 'matches', 0))} | "
        f"stage={failure_stage}"
    )

    put_text(vis, line1, (25, 35), color=status_color, scale=0.55)
    put_text(vis, line2, (25, 65), color=status_color, scale=0.50)
    put_text(vis, f"reason: {failure_reason}", (25, 95), color=(0, 255, 255), scale=0.46)
    put_text(vis,
             "yellow=idle preview | blue=detector | green=global corr | magenta=reprojection | s=start/stop | r=reset | q=quit",
             (25, 125), color=(255, 180, 0), scale=0.46)
    timings = getattr(result, "timings_ms", {}) or {}
    put_text(vis,
             "ms: "
             f"track={timings.get('tracker_total_ms', 0.0):.1f} "
             f"cb={timings.get('checkerboard_ms', 0.0):.1f} "
             f"fast={timings.get('fast_persistent_ms', 0.0):.1f} "
             f"pnp={timings.get('pnp_ms', 0.0):.1f}",
             (25, 155), color=(0, 255, 255), scale=0.46)
    put_text(vis,
             f"SPACE=log {'STOP' if tracker_log.is_active() else 'START'}",
             (25, 185), color=(0, 255, 255), scale=0.46)


def draw_app_state(
    vis: np.ndarray,
    app_state: str,
    acquire_frames: int,
    stale_frames: int,
    *,
    tracking_armed: bool,
    preview_count: int,
    preview_ms: float,
    auto_blocked: bool,
) -> None:
    if app_state == APP_IDLE:
        color = (180, 220, 255)
        armed = "armed" if tracking_armed else "manual"
        blocked = ", waiting for reposition" if auto_blocked else ""
        detail = (
            f"preview {preview_count} corners ({preview_ms:.1f} ms), "
            f"{armed}{blocked}; s=start/stop"
        )
    elif app_state == APP_ACQUIRE:
        color = (0, 255, 255)
        detail = f"searching {acquire_frames}/{ACQUIRE_TIMEOUT_FRAMES}"
    elif app_state == APP_PROVISIONAL:
        color = (0, 255, 255)
        detail = f"fragment warmup {acquire_frames}/{PROVISIONAL_TOTAL_TIMEOUT_FRAMES}"
    else:
        color = (0, 255, 0)
        detail = f"tracking; stale {stale_frames}/{TRACKING_STALE_TO_IDLE_FRAMES}"

    put_text(vis, f"app={app_state} | {detail}", (25, 215), color=color, scale=0.50)


def draw_debug(vis, result, K, dist, frame_idx, tracker) -> np.ndarray:
    draw_detection_corners(vis, result)
    draw_pose_corners(vis, result)
    draw_reprojection(vis, result, K, dist)
    draw_status(vis, result, frame_idx, tracker)
    return vis


def log_console(frame_idx: int, result, tracker, *, force: bool = False) -> None:
    if not force and (not result.success or frame_idx % 30 != 0):
        return

    failure_stage, failure_reason = tracker_log.classify_failure(result)
    fast = getattr(result, "fast_path_debug", None)
    print(
        "[test_tracker]",
        f"frame={frame_idx}",
        f"mode={result.mode.value}",
        f"success={result.success}",
        f"src={getattr(getattr(result, 'pose_source', None), 'value', 'none')}",
        f"fast={int(bool(getattr(fast, 'attempted', False)))}/"
        f"{int(bool(getattr(fast, 'success', False)))}:"
        f"{int(getattr(fast, 'matches', 0))}",
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


def make_tracker(field_path, marker_json_path, K, dist) -> HydraTracker:
    return HydraTracker(
        field_path=str(field_path),
        marker_json_path=str(marker_json_path),
        K=K,
        dist_coeffs=dist,
        config=TrackerConfig(
            min_points=6,
            min_inliers=5,
            max_mean_reprojection_error_px=4.0,
            max_max_reprojection_error_px=15.0,
            max_lost_frames=8,
            max_translation_jump_mm=40.0,
            # Reject one-frame pose-branch jumps before they can seed history.
            max_rotation_jump_deg=45.0,
            rotation_gate_scale_per_lost_frame=8.0,
            rotation_gate_max_deg=90.0,
            decode_update_min_visual_corners=12,
            decode_update_min_distinct_rows=3,
            decode_update_min_distinct_cols=3,
            # On the drill/cylinder, low pts is often caused by visibility,
            # not LK drift. Do not reset dot state based on point count.
            dot_early_reset_pts_ratio=0.0,
            dot_early_reset_min_pts=6,
            pnp_ransac_iterations=500,
            pnp_ransac_reprojection_px=3.0,
            pnp_ransac_confidence=0.99,
            use_pose_prior=True,
            pnp_direct_refine_method="vvs",   # "lm" / "vvs"
            pose_depth_filter_enabled=True,
            pose_depth_filter_observation_std_mm=16.0,
            pose_depth_filter_process_std_mm=0.05,
            pose_depth_filter_initial_velocity_std_mm=0.1,
            pose_depth_filter_reprojection_guard_px=1.0,
            pose_depth_filter_min_points=6,
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
            # Use only real checkerboard detections for dot decoding.
            # Pose-projected cells are unsafe on a cylinder because projected
            # corners may be on the occluded side.
            enable_pose_propagation=False,

            # Safe decode-outage bridge: cached global IDs are matched by
            # last-pose reprojection, not stale UV proximity. This can bridge
            # short decoder dropouts without accepting newly ambiguous IDs.
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

            # Current-frame dot decisions: no EMA warmup-lock.
            dot_use_temporal_smoothing=False,

            dot_commit_frames=1,
            dot_revoke_frames=5,
            persistence_max_frames=8,
        ),
    )


def run_live_tracker(
    *,
    window_name: str = "HydraTracker RealSense Test",
    console_prefix: str = "[test_tracker]",
    after_camera_ready: Optional[
        Callable[[Any, np.ndarray, np.ndarray], bool]
    ] = None,
    after_tracker_created: Optional[Callable[[HydraTracker], None]] = None,
    on_space_key: Optional[Callable[[HydraTracker, Any, int], bool]] = None,
    on_log_open: Optional[Callable[[Path, Path, HydraTracker], None]] = None,
    on_log_close: Optional[Callable[[Path], None]] = None,
    draw_extra_overlay: Optional[Callable[[np.ndarray, bool], None]] = None,
    stop_after_log_close: bool = False,
    quit_on_q: bool = True,
    final_cleanup: Optional[Callable[[], None]] = None,
) -> Optional[Path]:
    """Run the shared RealSense tracker loop used by tests and debug tools."""
    field_path = choose_file_qt("Select HydraMarker .field file", "HydraMarker field (*.field)")
    marker_json_path = choose_file_qt("Select marker .json file", "Marker JSON (*.json)")
    calibration_path = choose_file_qt(
        "Select camera calibration .npz",
        "NPZ files (*.npz)",
    )

    pipe, profile = create_realsense_pipeline()
    recorded_log_path: Optional[Path] = None

    try:
        K_rgb, dist_rgb = load_tracker_camera_calibration(
            profile,
            calibration_path=calibration_path,
        )
        if after_camera_ready is not None and not after_camera_ready(pipe, K_rgb, dist_rgb):
            return None

        tracker = make_tracker(field_path, marker_json_path, K_rgb, dist_rgb)
        if after_tracker_created is not None:
            after_tracker_created(tracker)

        preview_detector = make_idle_preview_detector()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_idx = 0
        last_mode: Optional[str] = None
        last_success: Optional[bool] = None
        last_message: Optional[str] = None
        app_state = APP_IDLE
        acquire_start_frame = 0
        provisional_start_frame = 0
        last_candidate_frame = 0
        stale_pose_frames = 0
        tracking_armed = False
        auto_acquire_blocked = False

        def enter_idle(
            reason: str,
            *,
            keep_armed: bool = True,
            block_auto_acquire: bool = False,
        ) -> None:
            nonlocal app_state, acquire_start_frame, provisional_start_frame
            nonlocal last_candidate_frame, stale_pose_frames
            nonlocal tracking_armed, auto_acquire_blocked
            nonlocal last_mode, last_success, last_message
            tracker.reset()
            preview_detector.reset_tracking()
            app_state = APP_IDLE
            acquire_start_frame = 0
            provisional_start_frame = 0
            last_candidate_frame = 0
            stale_pose_frames = 0
            tracking_armed = bool(keep_armed)
            auto_acquire_blocked = bool(block_auto_acquire)
            last_mode = None
            last_success = None
            last_message = None
            print(f"{console_prefix} idle ({reason})")

        def start_acquire(*, manual: bool = False) -> None:
            nonlocal app_state, acquire_start_frame, provisional_start_frame
            nonlocal last_candidate_frame, stale_pose_frames
            nonlocal tracking_armed, auto_acquire_blocked
            nonlocal last_mode, last_success, last_message
            tracker.reset()
            preview_detector.reset_tracking()
            app_state = APP_ACQUIRE
            acquire_start_frame = frame_idx
            provisional_start_frame = 0
            last_candidate_frame = 0
            stale_pose_frames = 0
            tracking_armed = True
            auto_acquire_blocked = False
            last_mode = None
            last_success = None
            last_message = None
            reason = "manual" if manual else "auto"
            print(f"{console_prefix} acquire started ({reason})")

        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_idx += 1
            frame = np.asanyarray(color_frame.get_data())

            t0 = time.perf_counter()
            result = tracker.process_frame(
                frame,
                run_detection=(app_state != APP_IDLE),
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0

            preview_detection = None
            preview_corners: list[tuple[float, float]] = []
            preview_ms = 0.0
            if app_state == APP_IDLE:
                preview_t0 = time.perf_counter()
                preview_detection = preview_detector.detect(frame)
                preview_ms = (time.perf_counter() - preview_t0) * 1000.0
                preview_corners = preview_corner_uvs(preview_detection)
                if len(preview_corners) < PROVISIONAL_MIN_CORNERS:
                    auto_acquire_blocked = False
                if (
                    tracking_armed
                    and not auto_acquire_blocked
                    and len(preview_corners) >= IDLE_PREVIEW_AUTO_ACQUIRE_CORNERS
                ):
                    start_acquire(manual=False)

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
                    enter_idle("acquire timeout", block_auto_acquire=True)
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
                        enter_idle("provisional stale", block_auto_acquire=True)
                    elif provisional_frames >= PROVISIONAL_TOTAL_TIMEOUT_FRAMES:
                        enter_idle("provisional timeout", block_auto_acquire=True)
            elif app_state == APP_TRACKING:
                if tracker_log.has_fresh_pose(result):
                    stale_pose_frames = 0
                else:
                    stale_pose_frames += 1
                    if stale_pose_frames >= TRACKING_STALE_TO_IDLE_FRAMES:
                        enter_idle("tracking lost", keep_armed=True)

            # Console log — only on state changes or failures
            mode_changed = last_mode != result.mode.value
            success_changed = last_success != bool(result.success)
            message_changed = last_message != result.message
            force_log = (
                mode_changed
                or success_changed
                or message_changed
                or (not result.success and app_state != APP_IDLE)
            )
            log_console(frame_idx, result, tracker, force=force_log)

            last_mode = result.mode.value
            last_success = bool(result.success)
            last_message = result.message

            draw_t0 = time.perf_counter()
            vis = draw_debug(frame.copy(), result, K_rgb, dist_rgb, frame_idx, tracker)
            if app_state == APP_IDLE:
                draw_preview_corners(vis, preview_corners)
            active_frames_for_display = (
                max(0, frame_idx - acquire_start_frame + 1)
                if app_state in (APP_ACQUIRE, APP_PROVISIONAL)
                and acquire_start_frame > 0
                else 0
            )
            draw_app_state(
                vis,
                app_state,
                active_frames_for_display,
                stale_pose_frames,
                tracking_armed=tracking_armed,
                preview_count=len(preview_corners),
                preview_ms=preview_ms,
                auto_blocked=auto_acquire_blocked,
            )
            if draw_extra_overlay is not None:
                try:
                    draw_extra_overlay(vis, tracker_log.is_active(), result)
                except TypeError:
                    draw_extra_overlay(vis, tracker_log.is_active())
            draw_ms = (time.perf_counter() - draw_t0) * 1000.0

            # Run log - only while SPACE-started logging is active.
            tracker_log.log_frame(
                frame_idx,
                result,
                wall_ms,
                tracker,
                draw_ms,
                frame=frame,
            )

            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or (quit_on_q and key == ord("q")):
                if tracker_log.is_active() and tracker_log.current_path() is not None:
                    recorded_log_path = Path(tracker_log.current_path())
                    tracker_log.log_close()
                    if on_log_close is not None:
                        on_log_close(recorded_log_path)
                break
            if key == ord("r"):
                enter_idle("reset", keep_armed=False)
            if key == ord("s"):
                if app_state == APP_IDLE:
                    start_acquire(manual=True)
                else:
                    enter_idle("manual stop", keep_armed=False)
            if key == ord(" "):
                if on_space_key is not None and on_space_key(tracker, result, frame_idx):
                    continue
                if tracker_log.is_active():
                    log_path = tracker_log.current_path()
                    closed_log_path = None
                    if log_path is not None:
                        recorded_log_path = Path(log_path)
                        closed_log_path = recorded_log_path
                    tracker_log.log_close()
                    if closed_log_path is not None and on_log_close is not None:
                        on_log_close(closed_log_path)
                    if stop_after_log_close:
                        break
                else:
                    tracker_log.log_open(field_path, marker_json_path, tracker)
                    if on_log_open is not None:
                        on_log_open(field_path, marker_json_path, tracker)

    finally:
        try:
            if final_cleanup is not None:
                final_cleanup()
        finally:
            pipe.stop()
            cv2.destroyAllWindows()
            tracker_log.log_close()

    return recorded_log_path


# ============================================================
# Main
# ============================================================

def main() -> None:
    run_live_tracker()


if __name__ == "__main__":
    main()
