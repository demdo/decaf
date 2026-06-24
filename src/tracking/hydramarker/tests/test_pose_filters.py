from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


def _ensure_src_on_path() -> None:
    src_root = Path(__file__).resolve().parents[3]
    src = str(src_root)
    if src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()

from tracking.pose_filters import PoseDepthKalmanFilter
from tracking.pose_prior import solve_plateau_pose_prior


def _project_points(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    import cv2

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    return np.asarray(projected, dtype=np.float64).reshape(-1, 2)


def test_depth_filter_limits_large_z_span_negative_delta() -> None:
    depth_filter = PoseDepthKalmanFilter(
        observation_std_mm=16.0,
        process_std_mm=0.05,
        initial_velocity_std_mm=0.1,
        reprojection_guard_px=0.0,
        K=np.eye(3, dtype=np.float64),
        dist_coeffs=np.zeros((0, 1), dtype=np.float64),
        innovation_guard_window=3,
        innovation_guard_bias_threshold_mm=0.25,
        innovation_guard_min_same_sign=3,
        innovation_cusum_threshold_mm=1000.0,
        negative_delta_guard_min_z_span_mm=10.0,
        negative_delta_guard_max_negative_delta_mm=0.1,
    )
    object_points = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 15.0],
            [1.0, -1.0, 15.0],
        ],
        dtype=np.float64,
    )
    image_points = np.zeros((6, 2), dtype=np.float64)

    last = None
    for z_mm in (100.0, 101.0, 102.0, 103.0, 104.0, 105.0):
        last = depth_filter.update(
            rvec=np.zeros((3, 1), dtype=np.float64),
            tvec=np.asarray([[0.0], [0.0], [z_mm]], dtype=np.float64),
            object_points=object_points,
            image_points=image_points,
        )

    assert last is not None
    assert last.negative_delta_guard_limited
    assert last.delta_z_mm >= -0.1


def test_depth_filter_holds_previous_z_when_negative_velocity_overshoots() -> None:
    depth_filter = PoseDepthKalmanFilter(
        observation_std_mm=16.0,
        process_std_mm=0.05,
        initial_velocity_std_mm=0.1,
        reprojection_guard_px=0.0,
        K=np.eye(3, dtype=np.float64),
        dist_coeffs=np.zeros((0, 1), dtype=np.float64),
        negative_delta_guard_min_z_span_mm=10.0,
        negative_delta_guard_max_negative_delta_mm=0.0,
        negative_delta_guard_hold_previous_z=True,
        negative_delta_guard_hold_requires_innovation_bias=False,
        negative_delta_guard_max_hold_correction_mm=0.5,
    )
    depth_filter._x = np.asarray([101.0, -2.0], dtype=np.float64)
    depth_filter._P = np.diag([0.01, 0.01]).astype(np.float64)

    object_points = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 15.0],
            [1.0, -1.0, 15.0],
        ],
        dtype=np.float64,
    )

    result = depth_filter.update(
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [100.0]], dtype=np.float64),
        object_points=object_points,
        image_points=np.zeros((6, 2), dtype=np.float64),
    )

    assert result.negative_delta_guard_limited
    assert result.filtered_z_mm == 100.5
    assert result.delta_z_mm == 0.5


def test_plateau_pose_prior_accepts_static_pose_with_small_reprojection_excess() -> None:
    object_points = np.asarray(
        [
            [-10.0, -10.0, 0.0],
            [10.0, -10.0, 0.0],
            [-10.0, 10.0, 0.0],
            [10.0, 10.0, 0.0],
            [-10.0, -10.0, 15.0],
            [10.0, -10.0, 15.0],
            [-10.0, 10.0, 15.0],
            [10.0, 10.0, 15.0],
        ],
        dtype=np.float64,
    )
    K = np.asarray(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((5, 1), dtype=np.float64)
    seed_rvec = np.zeros((3, 1), dtype=np.float64)
    seed_tvec = np.asarray([[0.0], [0.0], [300.0]], dtype=np.float64)
    raw_rvec = seed_rvec.copy()
    raw_tvec = np.asarray([[0.0], [0.0], [299.5]], dtype=np.float64)
    image_points = _project_points(object_points, seed_rvec, seed_tvec, K, dist)

    result = solve_plateau_pose_prior(
        object_points=object_points,
        image_points=image_points,
        K=K,
        dist_coeffs=dist,
        raw_rvec=raw_rvec,
        raw_tvec=raw_tvec,
        seed_rvec=seed_rvec,
        seed_tvec=seed_tvec,
        static_max_excess_px=0.12,
        min_positive_z_correction_mm=0.02,
        max_positive_z_correction_mm=1.0,
    )

    assert result.success
    assert result.method == "static"
    assert result.delta_z_mm == 0.5
