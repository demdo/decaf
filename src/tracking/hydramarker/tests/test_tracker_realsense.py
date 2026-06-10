from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.hydramarker.backend import cpp_impl as hydramarker_cpp
from tracking.hydramarker.config import TrackerConfig
from tracking.hydramarker.tracker import HydraTracker


# ============================================================
# Frame Logger / Diagnostics
# ============================================================

RUN_LOG_DIR = Path("hydramarker_tracker_runs")
CAMERA_CALIBRATION_ENV = "HYDRAMARKER_CAMERA_CALIB_NPZ"
DISTORTION_MODE_ENV = "HYDRAMARKER_DISTORTION_MODE"
LOG_FRAME_DETAILS_ENV = "HYDRAMARKER_LOG_FRAME_DETAILS"
LOG_POSE_CANDIDATES_ENV = "HYDRAMARKER_LOG_POSE_CANDIDATES"
POSE_CANDIDATE_WORST_COUNT = 8

FEW_GREEN_THRESHOLD = 5
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

COLUMNS = [
    # frame / timing
    "frame", "wall_ms",
    "tracker_total_ms", "checkerboard_ms", "fast_persistent_ms",
    "persistent_match_ms", "pnp_ms", "pnp_method", "pose_propagation_ms",
    "decode_pose_ms", "hold_pose_ms", "emergency_hold_ms", "draw_ms",
    "checkerboard_to_gray_ms", "checkerboard_track_total_ms",
    "checkerboard_lk_ms", "checkerboard_tracking_validate_ms",
    "checkerboard_build_visible_tracked_ms",
    "checkerboard_grid_build_tracking_ms",
    "checkerboard_update_tracking_state_ms",
    "checkerboard_refresh_recovery_call_ms",
    "checkerboard_recovery_total_ms",
    "checkerboard_recovery_corner_detect_ms",
    "checkerboard_recovery_refine_ms",
    "checkerboard_recovery_build_best_ms",
    "checkerboard_lattice_fit_ms",
    "checkerboard_grid_build_lattice_ms",
    "checkerboard_detail_timings",

    # tracker result
    "mode", "success", "pose_source", "pose_is_fresh", "pose_age_frames", "message",
    "failure_stage", "failure_reason",

    # detection / pose counts
    "det_valid", "det_tracking", "det_stable", "det_corners",
    "pose_corners", "num_points", "num_inliers",
    "mean_err", "max_err", "confidence",
    "persistent_count",

    # fast persistent path diagnostics
    "fast_attempted", "fast_success", "fast_matches", "fast_reason",
    "fast_identities", "fast_current_corners",
    "fast_dense_attempted", "fast_dense_success",
    "fast_dense_matches", "fast_dense_reason",
    "fast_dense_median_px", "fast_dense_p90_px",
    "fast_dense_projected", "fast_dense_detected",
    "fast_dense_rejected_no_projection", "fast_dense_rejected_far",
    "fast_dense_rejected_ambiguous", "fast_dense_rejected_non_mutual",
    "fast_dense_image_coverage",
    "fast_dense_image_span_u_px", "fast_dense_image_span_v_px",
    "fast_dense_object_span_mm",
    "fast_dense_distinct_rows", "fast_dense_distinct_cols",
    "fast_rejected_far", "fast_rejected_ambiguous",
    "fast_rejected_claimed", "fast_rejected_no_projection",

    # useful ratios
    "green_ratio", "inlier_ratio",

    # per-frame motion diagnostics from detector corners
    "median_corner_motion_px",
    "mean_corner_motion_px",
    "p95_corner_motion_px",
    "max_corner_motion_px",
    "matched_corner_motion_count",
    "det_count_change",
    "det_count_change_abs",

    # pose jump diagnostics from accepted poses
    "tvec_x_mm",
    "tvec_y_mm",
    "tvec_z_mm",
    "rvec_x_rad",
    "rvec_y_rad",
    "rvec_z_rad",
    "rvec_angle_deg",
    "camera_roll_deg",
    "camera_pitch_deg",
    "camera_yaw_deg",
    "pose_rotation_delta_deg",
    "pose_translation_delta_mm",

    # board-relative pose diagnostics, filled by debug_tracker_translation
    "board_pose_available",
    "board_tvec_x_mm",
    "board_tvec_y_mm",
    "board_tvec_z_mm",
    "board_delta_x_mm",
    "board_delta_y_mm",
    "board_delta_z_mm",
    "board_roll_deg",
    "board_pitch_deg",
    "board_yaw_deg",
    "board_rotation_delta_deg",
    "board_translation_delta_mm",
    "board_z_per_y_from_origin_mm_per_100mm",

    # residual and point-distribution diagnostics for the final pose
    "pose_reproj_mean_px",
    "pose_reproj_median_px",
    "pose_reproj_p95_px",
    "pose_reproj_max_px",
    "pose_reproj_mean_du_px",
    "pose_reproj_mean_dv_px",
    "pose_reproj_std_du_px",
    "pose_reproj_std_dv_px",
    "pose_image_u_min_px",
    "pose_image_u_max_px",
    "pose_image_v_min_px",
    "pose_image_v_max_px",
    "pose_image_span_u_px",
    "pose_image_span_v_px",
    "pose_image_centroid_u_px",
    "pose_image_centroid_v_px",
    "pose_global_row_min",
    "pose_global_row_max",
    "pose_global_col_min",
    "pose_global_col_max",
    "pose_distinct_rows",
    "pose_distinct_cols",
    "pose_object_x_span_mm",
    "pose_object_y_span_mm",
    "pose_object_z_span_mm",
    "pose_object_centroid_x_mm",
    "pose_object_centroid_y_mm",
    "pose_object_centroid_z_mm",
    "corr_distinct_rows",
    "corr_distinct_cols",
    "corr_object_x_span_mm",
    "corr_object_y_span_mm",
    "corr_object_z_span_mm",

    # debug-only alternative pose solver comparison
    "pose_candidate_count",
    "pose_candidate_best_method",
    "pose_candidate_best_all_mean_px",
    "pose_candidate_best_all_p95_px",
    "pose_candidate_best_board_delta_z_mm",
    "pose_candidate_logged_board_delta_z_mm",
    "pose_candidate_prior_lm_board_delta_z_mm",
    "pose_candidate_prior_vvs_board_delta_z_mm",
    "pose_candidate_sqpnp_board_delta_z_mm",
    "pose_candidate_ransac_board_delta_z_mm",
    "pose_candidate_trim_board_delta_z_mm",

    # session counters directly in CSV
    "pose_available_pct",
    "current_outage", "longest_outage",
    "pose_frames", "no_pose_frames",
    "blue_only_total", "zero_green_total", "few_green_total",

    # cause counters directly in CSV
    "decode_fail_total",
    "correspondence_fail_total",
    "rotation_gate_fail_total",
    "translation_gate_fail_total",
    "motion_gate_fail_total",
    "pnp_fail_total",
    "reprojection_fail_total",
    "other_fail_total",
]

_log_file = None
_log_path: Optional[Path] = None
_log_active = False
_log_run_id = ""
_camera_intrinsics_info: dict = {}

# Session statistics
_total_frames = 0
_pose_frames = 0
_no_pose_frames = 0

_current_outage = 0
_longest_outage = 0
_outage_lengths: list[int] = []

_blue_only_frames = 0
_zero_green_frames = 0
_few_green_frames = 0

_failure_counter: Counter[str] = Counter()

# Previous-frame state for motion diagnostics
_prev_detection_uv: Optional[np.ndarray] = None
_prev_det_count: Optional[int] = None
_prev_pose_rvec: Optional[np.ndarray] = None
_prev_pose_tvec: Optional[np.ndarray] = None
_debug_board_T_B_C: Optional[np.ndarray] = None
_debug_board_origin_tvec: Optional[np.ndarray] = None
_prev_board_R: Optional[np.ndarray] = None
_prev_board_tvec: Optional[np.ndarray] = None


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


def load_required_opencv_camera_calibration_from_env() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw_path = os.environ.get(CAMERA_CALIBRATION_ENV, "").strip()
    if not raw_path:
        raise RuntimeError(
            f"{CAMERA_CALIBRATION_ENV} must point to an OpenCV camera "
            "calibration .npz containing K and dist coefficients."
        )

    path = Path(raw_path).expanduser().resolve()
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
            ("dist", "dist_rgb", "dist_coeffs", "distortion_coeffs", "opencv_dist_coeffs"),
        )
        if dist is None:
            raise KeyError(
                "Camera calibration NPZ must contain OpenCV distortion coefficients: "
                "dist, dist_rgb, dist_coeffs, distortion_coeffs, or opencv_dist_coeffs."
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


def log_frame_details_enabled() -> bool:
    value = os.environ.get(LOG_FRAME_DETAILS_ENV, "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def log_pose_candidates_enabled() -> bool:
    value = os.environ.get(LOG_POSE_CANDIDATES_ENV, "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def _fmt_float(x: Optional[float], digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return ""
    return f"{float(x):.{digits}f}"


def set_debug_board_transform(T_B_C: Optional[np.ndarray]) -> None:
    global _debug_board_T_B_C
    global _debug_board_origin_tvec, _prev_board_R, _prev_board_tvec

    _debug_board_T_B_C = (
        None
        if T_B_C is None
        else np.asarray(T_B_C, dtype=np.float64).reshape(4, 4).copy()
    )
    _debug_board_origin_tvec = None
    _prev_board_R = None
    _prev_board_tvec = None


def _rotation_matrix_to_rpy_deg(R: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        sy = math.sqrt(float(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
        if sy > 1e-9:
            roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
            pitch = math.atan2(float(-R[2, 0]), sy)
            yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
        else:
            roll = math.atan2(float(-R[1, 2]), float(R[1, 1]))
            pitch = math.atan2(float(-R[2, 0]), sy)
            yaw = 0.0
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
    except Exception:
        return None, None, None


def _make_pose_matrix(rvec: Optional[np.ndarray], tvec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if rvec is None or tvec is None:
        return None
    try:
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    except Exception:
        return None


def _camera_rpy_deg(rvec: Optional[np.ndarray]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if rvec is None:
        return None, None, None
    try:
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        return _rotation_matrix_to_rpy_deg(R)
    except Exception:
        return None, None, None


def _rotation_delta_from_matrices_deg(
    R_prev: Optional[np.ndarray],
    R_curr: Optional[np.ndarray],
) -> Optional[float]:
    if R_prev is None or R_curr is None:
        return None
    try:
        R_delta = np.asarray(R_curr, dtype=np.float64).reshape(3, 3) @ np.asarray(
            R_prev, dtype=np.float64
        ).reshape(3, 3).T
        cos_angle = (np.trace(R_delta) - 1.0) / 2.0
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        return math.degrees(math.acos(cos_angle))
    except Exception:
        return None


def _board_pose_debug(
    rvec: Optional[np.ndarray],
    tvec: Optional[np.ndarray],
) -> dict[str, Optional[float] | int]:
    global _debug_board_origin_tvec, _prev_board_R, _prev_board_tvec

    empty = {
        "available": 0,
        "x": None,
        "y": None,
        "z": None,
        "dx": None,
        "dy": None,
        "dz": None,
        "roll": None,
        "pitch": None,
        "yaw": None,
        "rot_delta": None,
        "trans_delta": None,
        "z_per_y": None,
    }
    if _debug_board_T_B_C is None:
        return empty

    T_C_T = _make_pose_matrix(rvec, tvec)
    if T_C_T is None:
        return empty

    try:
        T_B_T = _debug_board_T_B_C @ T_C_T
        R_B_T = T_B_T[:3, :3]
        t_B_T = T_B_T[:3, 3].astype(np.float64)
        if _debug_board_origin_tvec is None:
            _debug_board_origin_tvec = t_B_T.copy()
        delta = t_B_T - _debug_board_origin_tvec

        rot_delta = _rotation_delta_from_matrices_deg(_prev_board_R, R_B_T)
        trans_delta = _translation_delta_mm(_prev_board_tvec, t_B_T)
        _prev_board_R = R_B_T.copy()
        _prev_board_tvec = t_B_T.copy()

        roll, pitch, yaw = _rotation_matrix_to_rpy_deg(R_B_T)
        z_per_y = None
        if abs(float(delta[1])) >= 1.0:
            z_per_y = 100.0 * float(delta[2]) / float(delta[1])

        return {
            "available": 1,
            "x": float(t_B_T[0]),
            "y": float(t_B_T[1]),
            "z": float(t_B_T[2]),
            "dx": float(delta[0]),
            "dy": float(delta[1]),
            "dz": float(delta[2]),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "rot_delta": rot_delta,
            "trans_delta": trans_delta,
            "z_per_y": z_per_y,
        }
    except Exception:
        return empty


def _corner_distribution_stats(corners) -> dict[str, Optional[float] | int]:
    if not corners:
        return {
            "row_min": None,
            "row_max": None,
            "col_min": None,
            "col_max": None,
            "distinct_rows": 0,
            "distinct_cols": 0,
            "x_span": None,
            "y_span": None,
            "z_span": None,
            "centroid_x": None,
            "centroid_y": None,
            "centroid_z": None,
            "u_min": None,
            "u_max": None,
            "v_min": None,
            "v_max": None,
            "u_span": None,
            "v_span": None,
            "centroid_u": None,
            "centroid_v": None,
        }

    rows, cols, xyz, uv = [], [], [], []
    for corner in corners:
        try:
            rows.append(int(getattr(corner, "global_row", getattr(corner, "local_row", 0))))
            cols.append(int(getattr(corner, "global_col", getattr(corner, "local_col", 0))))
            xyz.append([float(v) for v in corner.xyz_mm])
            uv.append([float(v) for v in corner.uv])
        except Exception:
            continue

    if not xyz or not uv:
        return _corner_distribution_stats([])

    rows_arr = np.asarray(rows, dtype=np.int32)
    cols_arr = np.asarray(cols, dtype=np.int32)
    xyz_arr = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    return {
        "row_min": int(np.min(rows_arr)),
        "row_max": int(np.max(rows_arr)),
        "col_min": int(np.min(cols_arr)),
        "col_max": int(np.max(cols_arr)),
        "distinct_rows": int(len(np.unique(rows_arr))),
        "distinct_cols": int(len(np.unique(cols_arr))),
        "x_span": float(np.ptp(xyz_arr[:, 0])),
        "y_span": float(np.ptp(xyz_arr[:, 1])),
        "z_span": float(np.ptp(xyz_arr[:, 2])),
        "centroid_x": float(np.mean(xyz_arr[:, 0])),
        "centroid_y": float(np.mean(xyz_arr[:, 1])),
        "centroid_z": float(np.mean(xyz_arr[:, 2])),
        "u_min": float(np.min(uv_arr[:, 0])),
        "u_max": float(np.max(uv_arr[:, 0])),
        "v_min": float(np.min(uv_arr[:, 1])),
        "v_max": float(np.max(uv_arr[:, 1])),
        "u_span": float(np.ptp(uv_arr[:, 0])),
        "v_span": float(np.ptp(uv_arr[:, 1])),
        "centroid_u": float(np.mean(uv_arr[:, 0])),
        "centroid_v": float(np.mean(uv_arr[:, 1])),
    }


def _pose_reprojection_debug(result, tracker: HydraTracker) -> dict[str, Optional[float]]:
    corners = list(getattr(result, "corners", []) or [])
    rvec = getattr(result, "rvec", None)
    tvec = getattr(result, "tvec", None)
    if not corners or rvec is None or tvec is None:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
            "mean_du": None,
            "mean_dv": None,
            "std_du": None,
            "std_dv": None,
        }

    object_points, image_points = [], []
    for corner in corners:
        try:
            object_points.append([float(v) for v in corner.xyz_mm])
            image_points.append([float(v) for v in corner.uv])
        except Exception:
            continue
    if not object_points:
        return _pose_reprojection_debug(None, tracker)

    try:
        projected, _ = cv2.projectPoints(
            np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tracker.K, dtype=np.float64).reshape(3, 3),
            np.asarray(tracker.dist_coeffs, dtype=np.float64).reshape(-1, 1),
        )
        projected = projected.reshape(-1, 2)
        image = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        residual = projected - image
        errors = np.linalg.norm(residual, axis=1)
        return {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95)),
            "max": float(np.max(errors)),
            "mean_du": float(np.mean(residual[:, 0])),
            "mean_dv": float(np.mean(residual[:, 1])),
            "std_du": float(np.std(residual[:, 0])),
            "std_dv": float(np.std(residual[:, 1])),
        }
    except Exception:
        return _pose_reprojection_debug(None, tracker)


def _extract_corner_pose_arrays(corners) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    object_points: list[list[float]] = []
    image_points: list[list[float]] = []
    labels: list[dict[str, Any]] = []
    for idx, corner in enumerate(list(corners or [])):
        try:
            xyz = [float(v) for v in corner.xyz_mm]
            uv = [float(v) for v in corner.uv]
        except Exception:
            continue

        object_points.append(xyz)
        image_points.append(uv)
        labels.append({
            "index": int(idx),
            "global_row": int(getattr(corner, "global_row", getattr(corner, "local_row", 0))),
            "global_col": int(getattr(corner, "global_col", getattr(corner, "local_col", 0))),
            "local_row": int(getattr(corner, "local_row", 0)),
            "local_col": int(getattr(corner, "local_col", 0)),
            "xyz_mm": xyz,
            "uv_px": uv,
        })

    if not object_points:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            [],
        )

    return (
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
        labels,
    )


def _projection_metrics(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> Optional[dict[str, Any]]:
    try:
        projected, _ = cv2.projectPoints(
            np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            np.asarray(K, dtype=np.float64).reshape(3, 3),
            np.asarray(dist, dtype=np.float64).reshape(-1, 1),
        )
    except Exception:
        return None

    projected = projected.reshape(-1, 2)
    measured = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if projected.shape != measured.shape or projected.size == 0:
        return None

    residual = projected - measured
    errors = np.linalg.norm(residual, axis=1)
    if not np.all(np.isfinite(errors)):
        return None

    return {
        "projected": projected,
        "residual": residual,
        "errors": errors,
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
        "mean_du": float(np.mean(residual[:, 0])),
        "mean_dv": float(np.mean(residual[:, 1])),
        "std_du": float(np.std(residual[:, 0])),
        "std_dv": float(np.std(residual[:, 1])),
    }


def _pose_score_from_metrics(metrics: dict[str, Any]) -> float:
    return (
        float(metrics["median"])
        + 0.35 * float(metrics["p90"])
        + 0.15 * float(metrics["mean"])
    )


def _board_pose_for_candidate(
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> dict[str, Optional[float] | int]:
    empty = {
        "available": 0,
        "x": None,
        "y": None,
        "z": None,
        "dx": None,
        "dy": None,
        "dz": None,
        "roll": None,
        "pitch": None,
        "yaw": None,
        "z_per_y": None,
    }
    if _debug_board_T_B_C is None:
        return empty

    T_C_T = _make_pose_matrix(rvec, tvec)
    if T_C_T is None:
        return empty

    try:
        T_B_T = _debug_board_T_B_C @ T_C_T
        R_B_T = T_B_T[:3, :3]
        t_B_T = T_B_T[:3, 3].astype(np.float64)
        origin = _debug_board_origin_tvec
        if origin is None:
            origin = t_B_T
        delta = t_B_T - np.asarray(origin, dtype=np.float64).reshape(3)
        roll, pitch, yaw = _rotation_matrix_to_rpy_deg(R_B_T)
        z_per_y = None
        if abs(float(delta[1])) >= 1.0:
            z_per_y = 100.0 * float(delta[2]) / float(delta[1])
        return {
            "available": 1,
            "x": float(t_B_T[0]),
            "y": float(t_B_T[1]),
            "z": float(t_B_T[2]),
            "dx": float(delta[0]),
            "dy": float(delta[1]),
            "dz": float(delta[2]),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "z_per_y": z_per_y,
        }
    except Exception:
        return empty


def _refined_pose_variants(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    prefix: str,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    base_rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    base_tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    variants = [(prefix, base_rvec.copy(), base_tvec.copy())]

    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            refined = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                K,
                dist,
                base_rvec.copy(),
                base_tvec.copy(),
            )
            if refined is not None:
                rvec_ref, tvec_ref = refined[:2]
                variants.append((
                    f"{prefix}_lm",
                    np.asarray(rvec_ref, dtype=np.float64).reshape(3, 1),
                    np.asarray(tvec_ref, dtype=np.float64).reshape(3, 1),
                ))
        except Exception:
            pass

    if hasattr(cv2, "solvePnPRefineVVS"):
        try:
            refined = cv2.solvePnPRefineVVS(
                object_points,
                image_points,
                K,
                dist,
                base_rvec.copy(),
                base_tvec.copy(),
            )
            if refined is not None:
                rvec_ref, tvec_ref = refined[:2]
                variants.append((
                    f"{prefix}_vvs",
                    np.asarray(rvec_ref, dtype=np.float64).reshape(3, 1),
                    np.asarray(tvec_ref, dtype=np.float64).reshape(3, 1),
                ))
        except Exception:
            pass

    return variants


def _make_pose_candidate(
    *,
    method: str,
    rvec: np.ndarray,
    tvec: np.ndarray,
    object_points: np.ndarray,
    image_points: np.ndarray,
    labels: list[dict[str, Any]],
    K: np.ndarray,
    dist: np.ndarray,
    inlier_indices: Optional[np.ndarray] = None,
) -> Optional[dict[str, Any]]:
    total_count = int(len(object_points))
    if total_count <= 0:
        return None

    all_metrics = _projection_metrics(object_points, image_points, rvec, tvec, K, dist)
    if all_metrics is None:
        return None

    if inlier_indices is None:
        inlier_indices = np.arange(total_count, dtype=np.int64)
    else:
        inlier_indices = np.asarray(inlier_indices, dtype=np.int64).reshape(-1)
        inlier_indices = inlier_indices[
            (0 <= inlier_indices) & (inlier_indices < total_count)
        ]
    if len(inlier_indices) == 0:
        return None

    fit_metrics = _projection_metrics(
        object_points[inlier_indices],
        image_points[inlier_indices],
        rvec,
        tvec,
        K,
        dist,
    )
    if fit_metrics is None:
        return None

    inlier_set = {int(i) for i in inlier_indices.reshape(-1)}
    rejected_indices = [idx for idx in range(total_count) if idx not in inlier_set]
    errors = np.asarray(all_metrics["errors"], dtype=np.float64).reshape(-1)
    residual = np.asarray(all_metrics["residual"], dtype=np.float64).reshape(-1, 2)
    worst_indices = np.argsort(errors)[::-1][:POSE_CANDIDATE_WORST_COUNT]
    worst = []
    for idx in worst_indices:
        label = dict(labels[int(idx)]) if int(idx) < len(labels) else {"index": int(idx)}
        label.update({
            "error_px": float(errors[int(idx)]),
            "residual_px": [
                float(residual[int(idx), 0]),
                float(residual[int(idx), 1]),
            ],
            "used_as_inlier": int(int(idx) in inlier_set),
        })
        worst.append(label)

    board = _board_pose_for_candidate(rvec, tvec)
    all_score = _pose_score_from_metrics(all_metrics)
    fit_score = _pose_score_from_metrics(fit_metrics)
    rvec_arr = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec_arr = np.asarray(tvec, dtype=np.float64).reshape(3)

    return {
        "method": str(method),
        "success": 1,
        "num_points": total_count,
        "num_inliers": int(len(inlier_indices)),
        "num_rejected": int(len(rejected_indices)),
        "inlier_indices": [int(i) for i in inlier_indices.reshape(-1)],
        "rejected_indices": [int(i) for i in rejected_indices],
        "all_score": float(all_score),
        "fit_score": float(fit_score),
        "all_reproj_mean_px": float(all_metrics["mean"]),
        "all_reproj_median_px": float(all_metrics["median"]),
        "all_reproj_p90_px": float(all_metrics["p90"]),
        "all_reproj_p95_px": float(all_metrics["p95"]),
        "all_reproj_max_px": float(all_metrics["max"]),
        "all_reproj_mean_du_px": float(all_metrics["mean_du"]),
        "all_reproj_mean_dv_px": float(all_metrics["mean_dv"]),
        "all_reproj_std_du_px": float(all_metrics["std_du"]),
        "all_reproj_std_dv_px": float(all_metrics["std_dv"]),
        "fit_reproj_mean_px": float(fit_metrics["mean"]),
        "fit_reproj_median_px": float(fit_metrics["median"]),
        "fit_reproj_p90_px": float(fit_metrics["p90"]),
        "fit_reproj_p95_px": float(fit_metrics["p95"]),
        "fit_reproj_max_px": float(fit_metrics["max"]),
        "rvec_rad": [float(v) for v in rvec_arr],
        "tvec_mm": [float(v) for v in tvec_arr],
        "board": board,
        "worst_corners": worst,
    }


def _trim_indices_for_candidate(errors: np.ndarray, tracker: HydraTracker) -> Optional[np.ndarray]:
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    if len(errors) < 12 or not np.all(np.isfinite(errors)):
        return None

    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    robust_sigma = 1.4826 * mad
    robust_threshold = max(0.75, median + 4.0 * robust_sigma)
    max_threshold = float(
        getattr(tracker.config, "fast_persistent_dense_robust_max_max_px", 2.5)
    )
    threshold = min(max_threshold, robust_threshold)
    quantile = float(
        getattr(tracker.config, "fast_persistent_dense_robust_trim_quantile", 0.85)
    )
    if 0.0 < quantile < 1.0:
        threshold = min(threshold, float(np.percentile(errors, quantile * 100.0)))

    keep = errors <= threshold
    min_keep = max(
        int(getattr(tracker.config, "min_inliers", 5)),
        int(np.ceil(
            float(
                getattr(
                    tracker.config,
                    "fast_persistent_dense_robust_min_keep_ratio",
                    0.75,
                )
            )
            * len(errors)
        )),
    )
    if int(np.count_nonzero(keep)) < min_keep or np.all(keep):
        return None
    return np.where(keep)[0].astype(np.int64)


def _pose_candidates_for_corners(
    point_set_name: str,
    corners,
    result,
    tracker: HydraTracker,
) -> Optional[dict[str, Any]]:
    object_points, image_points, labels = _extract_corner_pose_arrays(corners)
    count = int(len(object_points))
    if count < int(getattr(tracker.config, "min_inliers", 5)):
        return None

    K = np.asarray(tracker.K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(tracker.dist_coeffs, dtype=np.float64).reshape(-1, 1)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_variant(method: str, rvec, tvec, inlier_indices=None) -> None:
        method = str(method)
        if method in seen:
            return
        cand = _make_pose_candidate(
            method=method,
            rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            object_points=object_points,
            image_points=image_points,
            labels=labels,
            K=K,
            dist=dist,
            inlier_indices=inlier_indices,
        )
        if cand is not None:
            candidates.append(cand)
            seen.add(method)

    result_rvec = getattr(result, "rvec", None)
    result_tvec = getattr(result, "tvec", None)
    if result_rvec is not None and result_tvec is not None:
        for method, rvec, tvec in _refined_pose_variants(
            object_points,
            image_points,
            K,
            dist,
            result_rvec,
            result_tvec,
            "logged_result",
        ):
            add_variant(method, rvec, tvec)

        current_metrics = _projection_metrics(
            object_points,
            image_points,
            result_rvec,
            result_tvec,
            K,
            dist,
        )
        if current_metrics is not None:
            trim_idx = _trim_indices_for_candidate(current_metrics["errors"], tracker)
            if trim_idx is not None:
                for method, rvec, tvec in _refined_pose_variants(
                    object_points[trim_idx],
                    image_points[trim_idx],
                    K,
                    dist,
                    result_rvec,
                    result_tvec,
                    "logged_result_trim",
                ):
                    add_variant(method, rvec, tvec, trim_idx)

    if _prev_pose_rvec is not None and _prev_pose_tvec is not None:
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                K,
                dist,
                rvec=np.asarray(_prev_pose_rvec, dtype=np.float64).reshape(3, 1).copy(),
                tvec=np.asarray(_prev_pose_tvec, dtype=np.float64).reshape(3, 1).copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if success:
                for method, cand_rvec, cand_tvec in _refined_pose_variants(
                    object_points,
                    image_points,
                    K,
                    dist,
                    rvec,
                    tvec,
                    "prior_iterative",
                ):
                    add_variant(method, cand_rvec, cand_tvec)
        except Exception:
            pass

    solve_flags: list[tuple[int, str]] = []
    if hasattr(cv2, "SOLVEPNP_SQPNP"):
        solve_flags.append((int(cv2.SOLVEPNP_SQPNP), "sqpnp"))
    if hasattr(cv2, "SOLVEPNP_EPNP"):
        solve_flags.append((int(cv2.SOLVEPNP_EPNP), "epnp"))
    if count >= 6:
        solve_flags.append((int(cv2.SOLVEPNP_ITERATIVE), "iterative_no_guess"))

    for flag, name in solve_flags:
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                K,
                dist,
                flags=flag,
            )
        except Exception:
            continue
        if not success:
            continue
        for method, cand_rvec, cand_tvec in _refined_pose_variants(
            object_points,
            image_points,
            K,
            dist,
            rvec,
            tvec,
            name,
        ):
            add_variant(method, cand_rvec, cand_tvec)

    if count >= int(getattr(tracker.config, "min_inliers", 5)):
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                K,
                dist,
                iterationsCount=int(getattr(tracker.config, "pnp_ransac_iterations", 500)),
                reprojectionError=float(getattr(tracker.config, "pnp_ransac_reprojection_px", 3.0)),
                confidence=float(getattr(tracker.config, "pnp_ransac_confidence", 0.99)),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            success, rvec, tvec, inliers = False, None, None, None
        if success and rvec is not None and tvec is not None and inliers is not None:
            inlier_idx = np.asarray(inliers, dtype=np.int64).reshape(-1)
            if len(inlier_idx) >= int(getattr(tracker.config, "min_inliers", 5)):
                add_variant("ransac_iterative", rvec, tvec, inlier_idx)
                for method, cand_rvec, cand_tvec in _refined_pose_variants(
                    object_points[inlier_idx],
                    image_points[inlier_idx],
                    K,
                    dist,
                    rvec,
                    tvec,
                    "ransac_iterative",
                ):
                    add_variant(method, cand_rvec, cand_tvec, inlier_idx)

    if not candidates:
        return None

    candidates.sort(key=lambda c: (float(c["all_score"]), int(c["num_rejected"])))
    distribution = _corner_distribution_stats(corners)
    return {
        "point_set": point_set_name,
        "point_count": count,
        "distribution": distribution,
        "candidates": candidates,
        "best_method": candidates[0]["method"],
    }


def _pose_candidate_sets(result, tracker: HydraTracker) -> list[dict[str, Any]]:
    sets: list[dict[str, Any]] = []
    seen_signatures: set[tuple[tuple[int, int], ...]] = set()

    for name, corners in (
        ("correspondence", list(getattr(result, "correspondence_corners", []) or [])),
        ("visual", list(getattr(result, "corners", []) or [])),
    ):
        signature = tuple(
            sorted(
                (
                    int(getattr(corner, "global_row", getattr(corner, "local_row", 0))),
                    int(getattr(corner, "global_col", getattr(corner, "local_col", 0))),
                )
                for corner in corners
            )
        )
        if not corners or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        candidate_set = _pose_candidates_for_corners(name, corners, result, tracker)
        if candidate_set is not None:
            sets.append(candidate_set)
    return sets


def _find_candidate(
    candidates: list[dict[str, Any]],
    prefix: str,
    *,
    exact: bool = False,
) -> Optional[dict[str, Any]]:
    if exact:
        matching = [c for c in candidates if str(c.get("method", "")) == prefix]
    else:
        matching = [c for c in candidates if str(c.get("method", "")).startswith(prefix)]
    if not matching:
        return None
    return min(matching, key=lambda c: (float(c["all_score"]), int(c["num_rejected"])))


def _candidate_board_dz(candidate: Optional[dict[str, Any]]) -> Optional[float]:
    if not candidate:
        return None
    board = candidate.get("board") or {}
    value = board.get("dz")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _pose_candidate_row_summary(candidate_sets: list[dict[str, Any]]) -> dict[str, Optional[float] | str | int]:
    empty: dict[str, Optional[float] | str | int] = {
        "count": 0,
        "best_method": "",
        "best_all_mean_px": None,
        "best_all_p95_px": None,
        "best_board_delta_z_mm": None,
        "logged_board_delta_z_mm": None,
        "prior_lm_board_delta_z_mm": None,
        "prior_vvs_board_delta_z_mm": None,
        "sqpnp_board_delta_z_mm": None,
        "ransac_board_delta_z_mm": None,
        "trim_board_delta_z_mm": None,
    }
    if not candidate_sets:
        return empty

    primary = next(
        (s for s in candidate_sets if s.get("point_set") == "correspondence"),
        candidate_sets[0],
    )
    candidates = list(primary.get("candidates") or [])
    if not candidates:
        return empty

    best = min(candidates, key=lambda c: (float(c["all_score"]), int(c["num_rejected"])))
    logged = _find_candidate(candidates, "logged_result", exact=True)
    prior_lm = _find_candidate(candidates, "prior_iterative_lm")
    prior_vvs = _find_candidate(candidates, "prior_iterative_vvs")
    sqpnp = _find_candidate(candidates, "sqpnp")
    ransac = _find_candidate(candidates, "ransac_iterative")
    trim = _find_candidate(candidates, "logged_result_trim")

    return {
        "count": int(sum(len(s.get("candidates") or []) for s in candidate_sets)),
        "best_method": str(best.get("method", "")),
        "best_all_mean_px": float(best["all_reproj_mean_px"]),
        "best_all_p95_px": float(best["all_reproj_p95_px"]),
        "best_board_delta_z_mm": _candidate_board_dz(best),
        "logged_board_delta_z_mm": _candidate_board_dz(logged),
        "prior_lm_board_delta_z_mm": _candidate_board_dz(prior_lm),
        "prior_vvs_board_delta_z_mm": _candidate_board_dz(prior_vvs),
        "sqpnp_board_delta_z_mm": _candidate_board_dz(sqpnp),
        "ransac_board_delta_z_mm": _candidate_board_dz(ransac),
        "trim_board_delta_z_mm": _candidate_board_dz(trim),
    }


def _pose_candidate_record(
    frame_idx: int,
    result,
    candidate_sets: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "pose_candidates",
        "run_id": _log_run_id,
        "frame": int(frame_idx),
        "success": int(bool(getattr(result, "success", False))),
        "pose_source": getattr(getattr(result, "pose_source", None), "value", "none"),
        "pnp_method": getattr(result, "pnp_method", ""),
        "point_sets": candidate_sets,
    }


def _compact_corner_record(corner) -> dict:
    record = {}
    for attr in ("local_row", "local_col", "global_row", "global_col", "votes"):
        if hasattr(corner, attr):
            try:
                record[attr] = int(getattr(corner, attr))
            except Exception:
                pass
    if hasattr(corner, "xyz_mm"):
        try:
            record["xyz_mm"] = [float(v) for v in corner.xyz_mm]
        except Exception:
            pass
    if hasattr(corner, "uv"):
        try:
            record["uv_px"] = [float(v) for v in corner.uv]
        except Exception:
            pass
    return record


def _pose_corner_detail_records(result, tracker: HydraTracker) -> list[dict]:
    corners = list(getattr(result, "corners", []) or [])
    rvec = getattr(result, "rvec", None)
    tvec = getattr(result, "tvec", None)
    if not corners:
        return []

    records = [_compact_corner_record(corner) for corner in corners]
    if rvec is None or tvec is None:
        return records

    object_points, image_points, valid_indices = [], [], []
    for idx, corner in enumerate(corners):
        try:
            object_points.append([float(v) for v in corner.xyz_mm])
            image_points.append([float(v) for v in corner.uv])
            valid_indices.append(idx)
        except Exception:
            continue
    if not object_points:
        return records

    try:
        projected, _ = cv2.projectPoints(
            np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tracker.K, dtype=np.float64).reshape(3, 3),
            np.asarray(tracker.dist_coeffs, dtype=np.float64).reshape(-1, 1),
        )
        projected = projected.reshape(-1, 2)
        image = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        residual = projected - image
        errors = np.linalg.norm(residual, axis=1)
        for local_idx, record_idx in enumerate(valid_indices):
            records[record_idx]["projected_uv_px"] = [
                float(projected[local_idx, 0]),
                float(projected[local_idx, 1]),
            ]
            records[record_idx]["residual_px"] = [
                float(residual[local_idx, 0]),
                float(residual[local_idx, 1]),
            ]
            records[record_idx]["error_px"] = float(errors[local_idx])
    except Exception:
        pass

    return records


def _frame_detail_record(frame_idx: int, result, tracker: HydraTracker) -> dict:
    return {
        "type": "frame_detail",
        "run_id": _log_run_id,
        "frame": int(frame_idx),
        "success": int(bool(getattr(result, "success", False))),
        "pose_source": getattr(getattr(result, "pose_source", None), "value", "none"),
        "pnp_method": getattr(result, "pnp_method", ""),
        "detection_corners": [
            _compact_corner_record(corner)
            for corner in list(getattr(result, "detection_corners", []) or [])
        ],
        "pose_corners": _pose_corner_detail_records(result, tracker),
        "correspondence_corners": [
            _compact_corner_record(corner)
            for corner in list(getattr(result, "correspondence_corners", []) or [])
        ],
    }


def _extract_detection_uv(result) -> np.ndarray:
    pts = getattr(result, "detection_corners", [])
    uv = []
    for p in pts:
        try:
            uv.append([float(p.uv[0]), float(p.uv[1])])
        except Exception:
            pass
    if not uv:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(uv, dtype=np.float64).reshape(-1, 2)


def _nearest_neighbor_motion(prev_uv: Optional[np.ndarray], curr_uv: np.ndarray) -> dict[str, Optional[float] | int]:
    """
    Estimate image motion from blue detector corners.

    This does not require stable corner IDs. For every current detected corner,
    we find the nearest previous detected corner and use the nearest-neighbor
    distance as an approximate per-frame motion. This is intentionally simple
    and robust enough for debugging motion-related tracking failures.
    """
    if prev_uv is None or len(prev_uv) == 0 or len(curr_uv) == 0:
        return {
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
            "count": 0,
        }

    # Pairwise distances: curr x prev. Number of corners is small, so this is fine.
    diff = curr_uv[:, None, :] - prev_uv[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    nearest = np.min(dists, axis=1)

    # Reject implausibly large NN matches caused by detection topology changes.
    # Keep the threshold generous, because we want to see fast motion too.
    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) == 0:
        return {
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
            "count": 0,
        }

    return {
        "median": float(np.median(nearest)),
        "mean": float(np.mean(nearest)),
        "p95": float(np.percentile(nearest, 95)),
        "max": float(np.max(nearest)),
        "count": int(len(nearest)),
    }


def _rotation_delta_deg(rvec_prev: Optional[np.ndarray], rvec_curr: Optional[np.ndarray]) -> Optional[float]:
    if rvec_prev is None or rvec_curr is None:
        return None
    try:
        R_prev, _ = cv2.Rodrigues(np.asarray(rvec_prev, dtype=np.float64).reshape(3, 1))
        R_curr, _ = cv2.Rodrigues(np.asarray(rvec_curr, dtype=np.float64).reshape(3, 1))
        R_delta = R_curr @ R_prev.T
        cos_angle = (np.trace(R_delta) - 1.0) / 2.0
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        return math.degrees(math.acos(cos_angle))
    except Exception:
        return None


def _translation_delta_mm(tvec_prev: Optional[np.ndarray], tvec_curr: Optional[np.ndarray]) -> Optional[float]:
    if tvec_prev is None or tvec_curr is None:
        return None
    try:
        a = np.asarray(tvec_prev, dtype=np.float64).reshape(3)
        b = np.asarray(tvec_curr, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(b - a))
    except Exception:
        return None


def _tvec_components_mm(tvec: Optional[np.ndarray]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if tvec is None:
        return None, None, None
    try:
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        return float(t[0]), float(t[1]), float(t[2])
    except Exception:
        return None, None, None


def _rvec_components_rad(
    rvec: Optional[np.ndarray],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if rvec is None:
        return None, None, None, None
    try:
        r = np.asarray(rvec, dtype=np.float64).reshape(3)
        angle_deg = math.degrees(float(np.linalg.norm(r)))
        return float(r[0]), float(r[1]), float(r[2]), angle_deg
    except Exception:
        return None, None, None, None


def classify_failure(result) -> tuple[str, str]:
    """
    Classify why a frame has no accepted pose.

    This is intentionally based on result.message so the test script works
    without requiring changes inside HydraTracker. Later, if HydraTracker exposes
    structured debug fields, this can be replaced by those fields.
    """
    if result.success:
        return "OK", "OK"

    msg = str(getattr(result, "message", "") or "")
    msg_l = msg.lower()

    if "idle" in msg_l and "skipped" in msg_l:
        return "IDLE", "DETECTION_SKIPPED"

    if "no valid decoded patches" in msg_l:
        return "PATCH_DECODER", "NO_VALID_DECODED_PATCHES"

    if "correspondence build failed" in msg_l:
        return "CORRESPONDENCE", "CORRESPONDENCE_REJECTED"

    if "too few correspondences" in msg_l:
        return "CORRESPONDENCE", "CORRESPONDENCE_REJECTED"

    if "rotation jump too large" in msg_l:
        return "MOTION_GATE", "ROTATION_JUMP_TOO_LARGE"

    if "translation jump too large" in msg_l:
        return "MOTION_GATE", "TRANSLATION_JUMP_TOO_LARGE"

    if "motion gate rejected" in msg_l:
        return "MOTION_GATE", "MOTION_GATE_REJECTED"

    if "pnp" in msg_l and ("failed" in msg_l or "fail" in msg_l):
        return "PNP", "PNP_FAILED"

    if "too few inliers" in msg_l:
        return "PNP", "TOO_FEW_INLIERS"

    if "reprojection" in msg_l and ("too large" in msg_l or "rejected" in msg_l):
        return "REPROJECTION_GATE", "REPROJECTION_ERROR_TOO_LARGE"

    return "OTHER", "OTHER"


def pose_source_value(result) -> str:
    return getattr(getattr(result, "pose_source", None), "value", "none")


def has_fresh_pose(result) -> bool:
    return bool(getattr(result, "success", False)) and pose_source_value(result) not in (
        "none",
        "hold",
    )


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _reset_log_state() -> None:
    global _total_frames, _pose_frames, _no_pose_frames
    global _current_outage, _longest_outage, _outage_lengths
    global _blue_only_frames, _zero_green_frames, _few_green_frames
    global _failure_counter
    global _prev_detection_uv, _prev_det_count, _prev_pose_rvec, _prev_pose_tvec
    global _debug_board_origin_tvec, _prev_board_R, _prev_board_tvec

    _total_frames = 0
    _pose_frames = 0
    _no_pose_frames = 0
    _current_outage = 0
    _longest_outage = 0
    _outage_lengths = []
    _blue_only_frames = 0
    _zero_green_frames = 0
    _few_green_frames = 0
    _failure_counter = Counter()
    _prev_detection_uv = None
    _prev_det_count = None
    _prev_pose_rvec = None
    _prev_pose_tvec = None
    _debug_board_origin_tvec = None
    _prev_board_R = None
    _prev_board_tvec = None


def _write_json_record(record: dict) -> None:
    if _log_file is None:
        return
    _log_file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    _log_file.flush()


def log_open(field_path: Path, marker_json_path: Path, tracker: HydraTracker) -> None:
    global _log_file, _log_path, _log_active, _log_run_id

    if _log_active:
        return

    _reset_log_state()

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    _log_path = RUN_LOG_DIR / f"hydramarker_tracker_run_{_log_run_id}.jsonl"
    _log_file = open(_log_path, "w", encoding="utf-8")
    _log_active = True

    _write_json_record({
        "type": "run_start",
        "run_id": _log_run_id,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "field_path": str(Path(field_path).resolve()),
        "marker_json_path": str(Path(marker_json_path).resolve()),
        "camera_intrinsics": dict(_camera_intrinsics_info),
        "frame_detail_records": bool(log_frame_details_enabled()),
        "pose_candidate_records": bool(log_pose_candidates_enabled()),
        "columns": COLUMNS,
        "config": {
            name: getattr(tracker.config, name)
            for name in sorted(vars(tracker.config).keys())
        },
    })

    print(f"[run_log] started -> {_log_path.resolve()}")


def _write_event(kind: str, frame_idx: int, **fields) -> None:
    if not _log_active:
        return
    _write_json_record({
        "type": "event",
        "run_id": _log_run_id,
        "event": kind,
        "frame": int(frame_idx),
        **fields,
    })


def log_frame(frame_idx: int, result, wall_ms: float, tracker: HydraTracker, draw_ms: float) -> None:
    global _total_frames, _pose_frames, _no_pose_frames
    global _current_outage, _longest_outage, _outage_lengths
    global _blue_only_frames, _zero_green_frames, _few_green_frames
    global _prev_detection_uv, _prev_det_count, _prev_pose_rvec, _prev_pose_tvec

    if not _log_active or _log_file is None:
        return

    detection_uv = _extract_detection_uv(result)
    det_corners = int(len(detection_uv))
    pose_corners = int(_safe_len(getattr(result, "corners", [])))
    num_points = int(getattr(result, "num_points", 0) or 0)
    num_inliers = int(getattr(result, "num_inliers", 0) or 0)

    failure_stage, failure_reason = classify_failure(result)

    _total_frames += 1

    has_pose = bool(getattr(result, "success", False))
    was_in_outage = _current_outage > 0

    if has_pose:
        _pose_frames += 1
        if _current_outage > 0:
            _outage_lengths.append(_current_outage)
            _longest_outage = max(_longest_outage, _current_outage)
            pose_available_pct_now = 100.0 * _pose_frames / max(_total_frames, 1)
            _write_event(
                "OUTAGE_END",
                frame_idx,
                duration=_current_outage,
                pose_available_pct=round(pose_available_pct_now, 2),
            )
            _current_outage = 0
    else:
        _no_pose_frames += 1
        _current_outage += 1
        _longest_outage = max(_longest_outage, _current_outage)
        _failure_counter[failure_reason] += 1

        event_kind = "OUTAGE_START" if not was_in_outage else "OUTAGE_CONT"
        _write_event(
            event_kind,
            frame_idx,
            length=_current_outage,
            stage=failure_stage,
            reason=failure_reason,
            det_corners=det_corners,
            pose_corners=pose_corners,
            num_points=num_points,
            num_inliers=num_inliers,
            message=getattr(result, "message", ""),
        )

    if det_corners > 0 and pose_corners == 0:
        _blue_only_frames += 1

    if pose_corners == 0:
        _zero_green_frames += 1

    if pose_corners < FEW_GREEN_THRESHOLD:
        _few_green_frames += 1

    green_ratio = pose_corners / max(det_corners, 1)
    inlier_ratio = num_inliers / max(num_points, 1)
    pose_available_pct = 100.0 * _pose_frames / max(_total_frames, 1)

    det_count_change = 0 if _prev_det_count is None else det_corners - _prev_det_count
    det_count_change_abs = abs(det_count_change)

    motion = _nearest_neighbor_motion(_prev_detection_uv, detection_uv)

    tvec_x_mm, tvec_y_mm, tvec_z_mm = _tvec_components_mm(
        getattr(result, "tvec", None) if has_pose else None
    )
    rvec_x_rad, rvec_y_rad, rvec_z_rad, rvec_angle_deg = _rvec_components_rad(
        getattr(result, "rvec", None) if has_pose else None
    )
    camera_roll_deg, camera_pitch_deg, camera_yaw_deg = _camera_rpy_deg(
        getattr(result, "rvec", None) if has_pose else None
    )
    board_pose = _board_pose_debug(
        getattr(result, "rvec", None) if has_pose else None,
        getattr(result, "tvec", None) if has_pose else None,
    )
    pose_reproj = _pose_reprojection_debug(result, tracker) if has_pose else _pose_reprojection_debug(None, tracker)
    pose_distribution = _corner_distribution_stats(getattr(result, "corners", []) or [])
    corr_distribution = _corner_distribution_stats(getattr(result, "correspondence_corners", []) or [])
    candidate_sets = (
        _pose_candidate_sets(result, tracker)
        if has_pose and log_pose_candidates_enabled()
        else []
    )
    candidate_summary = _pose_candidate_row_summary(candidate_sets)

    # Pose delta is only meaningful between accepted poses.
    if has_pose and getattr(result, "rvec", None) is not None and getattr(result, "tvec", None) is not None:
        rot_delta = _rotation_delta_deg(_prev_pose_rvec, result.rvec)
        trans_delta = _translation_delta_mm(_prev_pose_tvec, result.tvec)
        _prev_pose_rvec = np.asarray(result.rvec, dtype=np.float64).reshape(3, 1).copy()
        _prev_pose_tvec = np.asarray(result.tvec, dtype=np.float64).reshape(3, 1).copy()
    else:
        rot_delta = None
        trans_delta = None

    fast = getattr(result, "fast_path_debug", None)
    timings = getattr(result, "timings_ms", {}) or {}

    dense_median = getattr(fast, "dense_refine_median_error_px", None)
    if dense_median is not None and float(dense_median) < 0.0:
        dense_median = None
    dense_p90 = getattr(fast, "dense_refine_p90_error_px", None)
    if dense_p90 is not None and float(dense_p90) < 0.0:
        dense_p90 = None
    dense_coverage = getattr(fast, "dense_refine_image_coverage", None)
    if dense_coverage is not None and float(dense_coverage) < 0.0:
        dense_coverage = None
    dense_span_u = getattr(fast, "dense_refine_image_span_u_px", None)
    if dense_span_u is not None and float(dense_span_u) < 0.0:
        dense_span_u = None
    dense_span_v = getattr(fast, "dense_refine_image_span_v_px", None)
    if dense_span_v is not None and float(dense_span_v) < 0.0:
        dense_span_v = None
    dense_object_span = getattr(fast, "dense_refine_object_span_mm", None)
    if dense_object_span is not None and float(dense_object_span) < 0.0:
        dense_object_span = None

    row = {
        "frame": frame_idx,
        "wall_ms": f"{wall_ms:.1f}",
        "tracker_total_ms": _fmt_float(timings.get("tracker_total_ms")),
        "checkerboard_ms": _fmt_float(timings.get("checkerboard_ms")),
        "fast_persistent_ms": _fmt_float(timings.get("fast_persistent_ms")),
        "persistent_match_ms": _fmt_float(timings.get("persistent_match_ms")),
        "pnp_ms": _fmt_float(timings.get("pnp_ms")),
        "pnp_method": getattr(result, "pnp_method", ""),
        "pose_propagation_ms": _fmt_float(timings.get("pose_propagation_ms")),
        "decode_pose_ms": _fmt_float(timings.get("decode_pose_ms")),
        "hold_pose_ms": _fmt_float(timings.get("hold_pose_ms")),
        "emergency_hold_ms": _fmt_float(timings.get("emergency_hold_ms")),
        "draw_ms": _fmt_float(draw_ms),
        "checkerboard_to_gray_ms": _fmt_float(timings.get("checkerboard_to_gray_ms")),
        "checkerboard_track_total_ms": _fmt_float(timings.get("checkerboard_track_total_ms")),
        "checkerboard_lk_ms": _fmt_float(timings.get("checkerboard_lk_ms")),
        "checkerboard_tracking_validate_ms": _fmt_float(timings.get("checkerboard_tracking_validate_ms")),
        "checkerboard_build_visible_tracked_ms": _fmt_float(timings.get("checkerboard_build_visible_tracked_ms")),
        "checkerboard_grid_build_tracking_ms": _fmt_float(timings.get("checkerboard_grid_build_tracking_ms")),
        "checkerboard_update_tracking_state_ms": _fmt_float(timings.get("checkerboard_update_tracking_state_ms")),
        "checkerboard_refresh_recovery_call_ms": _fmt_float(timings.get("checkerboard_refresh_recovery_call_ms")),
        "checkerboard_recovery_total_ms": _fmt_float(timings.get("checkerboard_recovery_total_ms")),
        "checkerboard_recovery_corner_detect_ms": _fmt_float(timings.get("checkerboard_recovery_corner_detect_ms")),
        "checkerboard_recovery_refine_ms": _fmt_float(timings.get("checkerboard_recovery_refine_ms")),
        "checkerboard_recovery_build_best_ms": _fmt_float(timings.get("checkerboard_recovery_build_best_ms")),
        "checkerboard_lattice_fit_ms": _fmt_float(timings.get("checkerboard_lattice_fit_ms")),
        "checkerboard_grid_build_lattice_ms": _fmt_float(timings.get("checkerboard_grid_build_lattice_ms")),
        "checkerboard_detail_timings": {
            key[len("checkerboard_"):]: round(float(value), 3)
            for key, value in sorted(timings.items())
            if key.startswith("checkerboard_")
        },

        "mode": result.mode.value,
        "success": int(has_pose),
        "pose_source": getattr(
            getattr(result, "pose_source", None),
            "value",
            "none",
        ),
        "pose_is_fresh": int(
            getattr(getattr(result, "pose_source", None), "value", "none")
            not in ("none", "hold")
        ),
        "pose_age_frames": (
            ""
            if not has_pose or getattr(tracker, "_last_accepted_pose_frame", -1) < 0
            else max(0, int(getattr(tracker, "frame_index", frame_idx)) - int(tracker._last_accepted_pose_frame))
        ),
        "message": getattr(result, "message", ""),
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,

        "det_valid": int(getattr(result, "detection_valid", False)),
        "det_tracking": int(getattr(result, "detection_tracking", False)),
        "det_stable": int(getattr(result, "detection_stable", False)),
        "det_corners": det_corners,
        "pose_corners": pose_corners,
        "num_points": num_points,
        "num_inliers": num_inliers,
        "mean_err": f"{float(getattr(result, 'mean_reprojection_error_px', 0.0)):.3f}",
        "max_err": f"{float(getattr(result, 'max_reprojection_error_px', 0.0)):.3f}",
        "confidence": f"{float(getattr(result, 'confidence', 0.0)):.3f}",
        "persistent_count": _safe_len(getattr(tracker, "_persistent_corners", [])),

        "fast_attempted": int(bool(getattr(fast, "attempted", False))),
        "fast_success": int(bool(getattr(fast, "success", False))),
        "fast_matches": int(getattr(fast, "matches", 0)),
        "fast_reason": getattr(fast, "reason", ""),
        "fast_identities": int(getattr(fast, "identities", 0)),
        "fast_current_corners": int(getattr(fast, "current_corners", 0)),
        "fast_dense_attempted": int(bool(getattr(fast, "dense_refine_attempted", False))),
        "fast_dense_success": int(bool(getattr(fast, "dense_refine_success", False))),
        "fast_dense_matches": int(getattr(fast, "dense_refine_matches", 0)),
        "fast_dense_reason": getattr(fast, "dense_refine_reason", ""),
        "fast_dense_median_px": _fmt_float(dense_median),
        "fast_dense_p90_px": _fmt_float(dense_p90),
        "fast_dense_projected": int(getattr(fast, "dense_refine_projected", 0)),
        "fast_dense_detected": int(getattr(fast, "dense_refine_detected", 0)),
        "fast_dense_rejected_no_projection": int(
            getattr(fast, "dense_refine_rejected_no_projection", 0)
        ),
        "fast_dense_rejected_far": int(getattr(fast, "dense_refine_rejected_far", 0)),
        "fast_dense_rejected_ambiguous": int(
            getattr(fast, "dense_refine_rejected_ambiguous", 0)
        ),
        "fast_dense_rejected_non_mutual": int(
            getattr(fast, "dense_refine_rejected_non_mutual", 0)
        ),
        "fast_dense_image_coverage": _fmt_float(dense_coverage),
        "fast_dense_image_span_u_px": _fmt_float(dense_span_u),
        "fast_dense_image_span_v_px": _fmt_float(dense_span_v),
        "fast_dense_object_span_mm": _fmt_float(dense_object_span),
        "fast_dense_distinct_rows": int(getattr(fast, "dense_refine_distinct_rows", 0)),
        "fast_dense_distinct_cols": int(getattr(fast, "dense_refine_distinct_cols", 0)),
        "fast_rejected_far": int(getattr(fast, "rejected_far", 0)),
        "fast_rejected_ambiguous": int(getattr(fast, "rejected_ambiguous", 0)),
        "fast_rejected_claimed": int(getattr(fast, "rejected_claimed", 0)),
        "fast_rejected_no_projection": int(
            getattr(fast, "rejected_no_projection", 0)
        ),

        "green_ratio": f"{green_ratio:.3f}",
        "inlier_ratio": f"{inlier_ratio:.3f}",

        "median_corner_motion_px": _fmt_float(motion["median"]),
        "mean_corner_motion_px": _fmt_float(motion["mean"]),
        "p95_corner_motion_px": _fmt_float(motion["p95"]),
        "max_corner_motion_px": _fmt_float(motion["max"]),
        "matched_corner_motion_count": motion["count"],
        "det_count_change": det_count_change,
        "det_count_change_abs": det_count_change_abs,

        "tvec_x_mm": _fmt_float(tvec_x_mm),
        "tvec_y_mm": _fmt_float(tvec_y_mm),
        "tvec_z_mm": _fmt_float(tvec_z_mm),
        "rvec_x_rad": _fmt_float(rvec_x_rad, digits=8),
        "rvec_y_rad": _fmt_float(rvec_y_rad, digits=8),
        "rvec_z_rad": _fmt_float(rvec_z_rad, digits=8),
        "rvec_angle_deg": _fmt_float(rvec_angle_deg),
        "camera_roll_deg": _fmt_float(camera_roll_deg),
        "camera_pitch_deg": _fmt_float(camera_pitch_deg),
        "camera_yaw_deg": _fmt_float(camera_yaw_deg),
        "pose_rotation_delta_deg": _fmt_float(rot_delta),
        "pose_translation_delta_mm": _fmt_float(trans_delta),

        "board_pose_available": int(board_pose["available"]),
        "board_tvec_x_mm": _fmt_float(board_pose["x"]),
        "board_tvec_y_mm": _fmt_float(board_pose["y"]),
        "board_tvec_z_mm": _fmt_float(board_pose["z"]),
        "board_delta_x_mm": _fmt_float(board_pose["dx"]),
        "board_delta_y_mm": _fmt_float(board_pose["dy"]),
        "board_delta_z_mm": _fmt_float(board_pose["dz"]),
        "board_roll_deg": _fmt_float(board_pose["roll"]),
        "board_pitch_deg": _fmt_float(board_pose["pitch"]),
        "board_yaw_deg": _fmt_float(board_pose["yaw"]),
        "board_rotation_delta_deg": _fmt_float(board_pose["rot_delta"]),
        "board_translation_delta_mm": _fmt_float(board_pose["trans_delta"]),
        "board_z_per_y_from_origin_mm_per_100mm": _fmt_float(board_pose["z_per_y"]),

        "pose_reproj_mean_px": _fmt_float(pose_reproj["mean"]),
        "pose_reproj_median_px": _fmt_float(pose_reproj["median"]),
        "pose_reproj_p95_px": _fmt_float(pose_reproj["p95"]),
        "pose_reproj_max_px": _fmt_float(pose_reproj["max"]),
        "pose_reproj_mean_du_px": _fmt_float(pose_reproj["mean_du"]),
        "pose_reproj_mean_dv_px": _fmt_float(pose_reproj["mean_dv"]),
        "pose_reproj_std_du_px": _fmt_float(pose_reproj["std_du"]),
        "pose_reproj_std_dv_px": _fmt_float(pose_reproj["std_dv"]),
        "pose_image_u_min_px": _fmt_float(pose_distribution["u_min"]),
        "pose_image_u_max_px": _fmt_float(pose_distribution["u_max"]),
        "pose_image_v_min_px": _fmt_float(pose_distribution["v_min"]),
        "pose_image_v_max_px": _fmt_float(pose_distribution["v_max"]),
        "pose_image_span_u_px": _fmt_float(pose_distribution["u_span"]),
        "pose_image_span_v_px": _fmt_float(pose_distribution["v_span"]),
        "pose_image_centroid_u_px": _fmt_float(pose_distribution["centroid_u"]),
        "pose_image_centroid_v_px": _fmt_float(pose_distribution["centroid_v"]),
        "pose_global_row_min": (
            "" if pose_distribution["row_min"] is None else int(pose_distribution["row_min"])
        ),
        "pose_global_row_max": (
            "" if pose_distribution["row_max"] is None else int(pose_distribution["row_max"])
        ),
        "pose_global_col_min": (
            "" if pose_distribution["col_min"] is None else int(pose_distribution["col_min"])
        ),
        "pose_global_col_max": (
            "" if pose_distribution["col_max"] is None else int(pose_distribution["col_max"])
        ),
        "pose_distinct_rows": int(pose_distribution["distinct_rows"]),
        "pose_distinct_cols": int(pose_distribution["distinct_cols"]),
        "pose_object_x_span_mm": _fmt_float(pose_distribution["x_span"]),
        "pose_object_y_span_mm": _fmt_float(pose_distribution["y_span"]),
        "pose_object_z_span_mm": _fmt_float(pose_distribution["z_span"]),
        "pose_object_centroid_x_mm": _fmt_float(pose_distribution["centroid_x"]),
        "pose_object_centroid_y_mm": _fmt_float(pose_distribution["centroid_y"]),
        "pose_object_centroid_z_mm": _fmt_float(pose_distribution["centroid_z"]),
        "corr_distinct_rows": int(corr_distribution["distinct_rows"]),
        "corr_distinct_cols": int(corr_distribution["distinct_cols"]),
        "corr_object_x_span_mm": _fmt_float(corr_distribution["x_span"]),
        "corr_object_y_span_mm": _fmt_float(corr_distribution["y_span"]),
        "corr_object_z_span_mm": _fmt_float(corr_distribution["z_span"]),

        "pose_candidate_count": int(candidate_summary["count"]),
        "pose_candidate_best_method": str(candidate_summary["best_method"]),
        "pose_candidate_best_all_mean_px": _fmt_float(candidate_summary["best_all_mean_px"]),
        "pose_candidate_best_all_p95_px": _fmt_float(candidate_summary["best_all_p95_px"]),
        "pose_candidate_best_board_delta_z_mm": _fmt_float(candidate_summary["best_board_delta_z_mm"]),
        "pose_candidate_logged_board_delta_z_mm": _fmt_float(candidate_summary["logged_board_delta_z_mm"]),
        "pose_candidate_prior_lm_board_delta_z_mm": _fmt_float(candidate_summary["prior_lm_board_delta_z_mm"]),
        "pose_candidate_prior_vvs_board_delta_z_mm": _fmt_float(candidate_summary["prior_vvs_board_delta_z_mm"]),
        "pose_candidate_sqpnp_board_delta_z_mm": _fmt_float(candidate_summary["sqpnp_board_delta_z_mm"]),
        "pose_candidate_ransac_board_delta_z_mm": _fmt_float(candidate_summary["ransac_board_delta_z_mm"]),
        "pose_candidate_trim_board_delta_z_mm": _fmt_float(candidate_summary["trim_board_delta_z_mm"]),

        "pose_available_pct": f"{pose_available_pct:.2f}",
        "current_outage": _current_outage,
        "longest_outage": _longest_outage,
        "pose_frames": _pose_frames,
        "no_pose_frames": _no_pose_frames,
        "blue_only_total": _blue_only_frames,
        "zero_green_total": _zero_green_frames,
        "few_green_total": _few_green_frames,

        "decode_fail_total": _failure_counter["NO_VALID_DECODED_PATCHES"],
        "correspondence_fail_total": _failure_counter["CORRESPONDENCE_REJECTED"],
        "rotation_gate_fail_total": _failure_counter["ROTATION_JUMP_TOO_LARGE"],
        "translation_gate_fail_total": _failure_counter["TRANSLATION_JUMP_TOO_LARGE"],
        "motion_gate_fail_total": (
            _failure_counter["MOTION_GATE_REJECTED"]
            + _failure_counter["ROTATION_JUMP_TOO_LARGE"]
            + _failure_counter["TRANSLATION_JUMP_TOO_LARGE"]
        ),
        "pnp_fail_total": _failure_counter["PNP_FAILED"] + _failure_counter["TOO_FEW_INLIERS"],
        "reprojection_fail_total": _failure_counter["REPROJECTION_ERROR_TOO_LARGE"],
        "other_fail_total": _failure_counter["OTHER"],
    }

    _write_json_record({
        "type": "frame",
        "run_id": _log_run_id,
        "data": row,
    })
    if candidate_sets:
        _write_json_record(_pose_candidate_record(frame_idx, result, candidate_sets))
    if log_frame_details_enabled():
        _write_json_record(_frame_detail_record(frame_idx, result, tracker))

    _prev_detection_uv = detection_uv.copy()
    _prev_det_count = det_corners


def log_close() -> None:
    global _current_outage, _longest_outage
    global _log_file, _log_path, _log_active, _log_run_id

    if not _log_active:
        return

    if _current_outage > 0:
        _outage_lengths.append(_current_outage)
        _longest_outage = max(_longest_outage, _current_outage)
        _current_outage = 0

    pose_availability = 100.0 * _pose_frames / max(_total_frames, 1)
    blue_only_ratio = 100.0 * _blue_only_frames / max(_total_frames, 1)
    zero_green_ratio = 100.0 * _zero_green_frames / max(_total_frames, 1)
    few_green_ratio = 100.0 * _few_green_frames / max(_total_frames, 1)
    mean_outage = sum(_outage_lengths) / len(_outage_lengths) if _outage_lengths else 0.0

    summary = {
        "total_frames": _total_frames,
        "pose_frames": _pose_frames,
        "no_pose_frames": _no_pose_frames,
        "pose_availability_pct": round(pose_availability, 2),
        "blue_only_frames": _blue_only_frames,
        "blue_only_ratio_pct": round(blue_only_ratio, 2),
        "zero_green_frames": _zero_green_frames,
        "zero_green_ratio_pct": round(zero_green_ratio, 2),
        "few_green_frames": _few_green_frames,
        "few_green_threshold": FEW_GREEN_THRESHOLD,
        "few_green_ratio_pct": round(few_green_ratio, 2),
        "pose_outage_count": len(_outage_lengths),
        "longest_pose_outage_frames": _longest_outage,
        "mean_pose_outage_frames": round(mean_outage, 2),
        "outage_lengths": list(_outage_lengths),
        "pose_losses_by_cause": dict(_failure_counter.most_common()),
    }

    _write_json_record({
        "type": "run_summary",
        "run_id": _log_run_id,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "summary": summary,
    })

    closed_path = _log_path
    if _log_file and not _log_file.closed:
        _log_file.close()

    _log_file = None
    _log_path = None
    _log_active = False
    _log_run_id = ""

    print(f"[run_log] closed -> {closed_path.resolve() if closed_path else ''}")


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
) -> tuple[np.ndarray, np.ndarray]:
    global _camera_intrinsics_info

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

    K, dist, calib_info = load_required_opencv_camera_calibration_from_env()
    stream_size = [int(getattr(intr, "width", 0)), int(getattr(intr, "height", 0))]
    calib_size = calib_info.get("calibration_image_size")
    if calib_size is not None and list(calib_size) != stream_size:
        raise RuntimeError(
            f"{CAMERA_CALIBRATION_ENV} image_size={calib_size} does not match "
            f"the active RealSense color stream {stream_size}."
        )

    _camera_intrinsics_info = {
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
    }
    print(
        f"[camera_intrinsics] using required {CAMERA_CALIBRATION_ENV}="
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
    failure_stage, failure_reason = classify_failure(result)
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
             f"SPACE=log {'STOP' if _log_active else 'START'}",
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

    failure_stage, failure_reason = classify_failure(result)
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
            max_translation_jump_mm=120.0,
            # Drill kann sich beliebig schnell drehen — Rotations-Gate deaktiviert.
            # Nur Translation wird geprüft (max_translation_jump_mm).
            max_rotation_jump_deg=360.0,
            rotation_gate_scale_per_lost_frame=0.0,
            rotation_gate_max_deg=360.0,
            # On the drill/cylinder, low pts is often caused by visibility,
            # not LK drift. Do not reset dot state based on point count.
            dot_early_reset_pts_ratio=0.0,
            dot_early_reset_min_pts=6,
            pnp_ransac_iterations=500,
            pnp_ransac_reprojection_px=3.0,
            pnp_ransac_confidence=0.99,
            use_pose_prior=True,
            pnp_direct_refine_method="vvs",   # "lm" / "vvs"
            corr_min_votes=2,
            corr_discard_conflicts=True,
            corr_require_detection_stable=False,
            corr_enable_dominant_rotation_filter=True,
            corr_min_rotation_support=2,
            corr_min_rotation_support_ratio=0.55,
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


# ============================================================
# Main
# ============================================================

def main() -> None:
    field_path = choose_file_qt("Select HydraMarker .field file", "HydraMarker field (*.field)")
    marker_json_path = choose_file_qt("Select marker .json file", "Marker JSON (*.json)")

    pipe, profile = create_realsense_pipeline()
    K_rgb, dist_rgb = load_tracker_camera_calibration(profile)

    tracker = make_tracker(field_path, marker_json_path, K_rgb, dist_rgb)
    preview_detector = make_idle_preview_detector()

    window_name = "HydraTracker RealSense Test"
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
        print(f"[test_tracker] idle ({reason})")

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
        print(f"[test_tracker] acquire started ({reason})")

    try:
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
                if has_fresh_pose(result):
                    app_state = APP_TRACKING
                    stale_pose_frames = 0
                    print("[test_tracker] tracking locked")
                elif det_count >= PROVISIONAL_MIN_CORNERS:
                    last_candidate_frame = frame_idx
                    if app_state != APP_PROVISIONAL:
                        app_state = APP_PROVISIONAL
                        provisional_start_frame = frame_idx
                        print(f"[test_tracker] provisional fragment det={det_count}")
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
                if has_fresh_pose(result):
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
            draw_ms = (time.perf_counter() - draw_t0) * 1000.0

            # Run log - only while SPACE-started logging is active.
            log_frame(frame_idx, result, wall_ms, tracker, draw_ms)

            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                enter_idle("reset", keep_armed=False)
            if key == ord("s"):
                if app_state == APP_IDLE:
                    start_acquire(manual=True)
                else:
                    enter_idle("manual stop", keep_armed=False)
            if key == ord(" "):
                if _log_active:
                    log_close()
                else:
                    log_open(field_path, marker_json_path, tracker)

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        log_close()


if __name__ == "__main__":
    main()
