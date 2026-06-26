from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


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


def test_cpp_depth_filter_matches_python_reference_sequence() -> None:
    cpp = pytest.importorskip("tracking.hydramarker.backend.cpp_impl")
    from tracking.hydramarker.config import TrackerConfig

    K = np.asarray(
        [[615.0, 0.0, 320.0], [0.0, 612.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.asarray([[0.01], [-0.003], [0.0002], [-0.0001], [0.0]], dtype=np.float64)
    object_points = np.asarray(
        [
            [-12.0, -10.0, 0.0],
            [12.0, -10.0, 0.0],
            [-12.0, 10.0, 0.0],
            [12.0, 10.0, 0.0],
            [-12.0, -10.0, 15.0],
            [12.0, -10.0, 15.0],
            [-12.0, 10.0, 15.0],
            [12.0, 10.0, 15.0],
            [0.0, 0.0, 8.0],
            [6.0, -4.0, 11.0],
            [-6.0, 4.0, 3.0],
        ],
        dtype=np.float64,
    )
    rvec = np.asarray([[0.13], [-0.08], [0.025]], dtype=np.float64)
    cfg = TrackerConfig(
        pose_depth_filter_innovation_window=5,
        pose_depth_filter_innovation_bias_threshold_mm=0.25,
        pose_depth_filter_innovation_min_same_sign=4,
        pose_depth_filter_innovation_cusum_slack_mm=0.05,
        pose_depth_filter_innovation_cusum_threshold_mm=1.5,
        pose_depth_filter_negative_delta_guard_min_z_span_mm=10.0,
        pose_depth_filter_negative_delta_guard_max_negative_delta_mm=0.1,
    )
    py_filter = PoseDepthKalmanFilter(
        observation_std_mm=cfg.pose_depth_filter_observation_std_mm,
        process_std_mm=cfg.pose_depth_filter_process_std_mm,
        initial_velocity_std_mm=cfg.pose_depth_filter_initial_velocity_std_mm,
        reprojection_guard_px=cfg.pose_depth_filter_reprojection_guard_px,
        K=K,
        dist_coeffs=dist,
        innovation_guard_enabled=cfg.pose_depth_filter_innovation_guard_enabled,
        innovation_guard_window=cfg.pose_depth_filter_innovation_window,
        innovation_guard_bias_threshold_mm=(
            cfg.pose_depth_filter_innovation_bias_threshold_mm
        ),
        innovation_guard_min_same_sign=cfg.pose_depth_filter_innovation_min_same_sign,
        innovation_cusum_slack_mm=cfg.pose_depth_filter_innovation_cusum_slack_mm,
        innovation_cusum_threshold_mm=(
            cfg.pose_depth_filter_innovation_cusum_threshold_mm
        ),
        negative_delta_guard_enabled=cfg.pose_depth_filter_negative_delta_guard_enabled,
        negative_delta_guard_min_z_span_mm=(
            cfg.pose_depth_filter_negative_delta_guard_min_z_span_mm
        ),
        negative_delta_guard_max_negative_delta_mm=(
            cfg.pose_depth_filter_negative_delta_guard_max_negative_delta_mm
        ),
        negative_delta_guard_hold_previous_z=(
            cfg.pose_depth_filter_negative_delta_guard_hold_previous_z
        ),
        negative_delta_guard_hold_requires_innovation_bias=(
            cfg.pose_depth_filter_negative_delta_guard_hold_requires_innovation_bias
        ),
        negative_delta_guard_hold_min_negative_delta_mm=(
            cfg.pose_depth_filter_negative_delta_guard_hold_min_negative_delta_mm
        ),
        negative_delta_guard_max_hold_correction_mm=(
            cfg.pose_depth_filter_negative_delta_guard_max_hold_correction_mm
        ),
        negative_delta_guard_velocity_damping=(
            cfg.pose_depth_filter_negative_delta_guard_velocity_damping
        ),
    )
    cpp_filter = cpp.create_pose_depth_filter(K, dist, cfg)

    scalar_fields = (
        "raw_z_mm",
        "filtered_z_mm",
        "delta_z_mm",
        "raw_reprojection_rms_px",
        "filtered_reprojection_rms_px",
        "reprojection_excess_px",
        "guard_alpha",
        "applied",
        "innovation_z_mm",
        "innovation_mean_z_mm",
        "innovation_cusum_pos_mm",
        "innovation_cusum_neg_mm",
        "innovation_bias_detected",
        "innovation_bias_direction",
        "innovation_bias_limited",
        "object_z_span_mm",
        "negative_delta_guard_limited",
    )

    for idx, z_mm in enumerate(
        (300.0, 300.4, 301.0, 301.8, 301.1, 300.7, 299.8, 298.9)
    ):
        tvec = np.asarray(
            [[3.0 + 0.01 * idx], [-2.0 + 0.02 * idx], [z_mm]],
            dtype=np.float64,
        )
        reference_tvec = np.asarray(
            [[3.0], [-2.0], [z_mm + (0.2 if idx % 3 == 0 else -0.1)]],
            dtype=np.float64,
        )
        image_points = _project_points(object_points, rvec, reference_tvec, K, dist)
        py_result = py_filter.update(
            rvec=rvec,
            tvec=tvec,
            object_points=object_points,
            image_points=image_points,
        )
        cpp_result = cpp_filter.update(
            rvec=rvec,
            tvec=tvec,
            object_points=object_points,
            image_points=image_points,
        )

        for field in scalar_fields:
            assert getattr(cpp_result, field) == pytest.approx(
                getattr(py_result, field),
                abs=2.0e-7,
            )
        np.testing.assert_allclose(
            np.asarray(cpp_result.rvec, dtype=np.float64).reshape(3, 1),
            py_result.rvec,
            atol=2.0e-7,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(cpp_result.tvec, dtype=np.float64).reshape(3, 1),
            py_result.tvec,
            atol=2.0e-7,
            rtol=0.0,
        )

    py_snapshot = py_filter.snapshot()
    cpp_snapshot = cpp_filter.snapshot()
    py_filter.update(
        rvec=rvec,
        tvec=np.asarray([[0.0], [0.0], [280.0]], dtype=np.float64),
        object_points=object_points,
        image_points=_project_points(
            object_points,
            rvec,
            np.asarray([[0.0], [0.0], [280.0]], dtype=np.float64),
            K,
            dist,
        ),
    )
    cpp_filter.update(
        rvec=rvec,
        tvec=np.asarray([[0.0], [0.0], [280.0]], dtype=np.float64),
        object_points=object_points,
        image_points=_project_points(
            object_points,
            rvec,
            np.asarray([[0.0], [0.0], [280.0]], dtype=np.float64),
            K,
            dist,
        ),
    )
    py_filter.restore(py_snapshot)
    cpp_filter.restore(cpp_snapshot)

    image_points = _project_points(
        object_points,
        rvec,
        np.asarray([[1.0], [2.0], [302.6]], dtype=np.float64),
        K,
        dist,
    )
    py_result = py_filter.update(
        rvec=rvec,
        tvec=np.asarray([[1.0], [2.0], [302.6]], dtype=np.float64),
        object_points=object_points,
        image_points=image_points,
    )
    cpp_result = cpp_filter.update(
        rvec=rvec,
        tvec=np.asarray([[1.0], [2.0], [302.6]], dtype=np.float64),
        object_points=object_points,
        image_points=image_points,
    )
    assert cpp_result.filtered_z_mm == pytest.approx(
        py_result.filtered_z_mm,
        abs=2.0e-7,
    )


def test_cpp_fast_pose_transaction_can_apply_depth_filter() -> None:
    cpp = pytest.importorskip("tracking.hydramarker.backend.cpp_impl")
    from tracking.hydramarker.config import TrackerConfig

    geometry_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "drill"
        / "PRO6350M"
        / "drill_marker_geometry_sfm_10x8_3x3_6mm.json"
    )
    geometry = cpp.MarkerGeometry.load_from_json(str(geometry_path))
    K = np.asarray(
        [[650.0, 0.0, 320.0], [0.0, 650.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((5, 1), dtype=np.float64)
    rvec = np.asarray([[0.12], [-0.06], [0.03]], dtype=np.float64)
    previous_tvec = np.asarray([[4.0], [-3.0], [300.0]], dtype=np.float64)
    current_tvec = np.asarray([[4.1], [-2.9], [302.0]], dtype=np.float64)

    keys: list[tuple[int, int]] = []
    for row in range(int(geometry.corner_rows())):
        for col in range(int(geometry.corner_cols())):
            if geometry.has_corner(row, col):
                keys.append((int(row), int(col)))
            if len(keys) >= 24:
                break
        if len(keys) >= 24:
            break
    assert len(keys) >= 12

    object_points = np.asarray(
        [
            [
                geometry.corner_point(row, col).x,
                geometry.corner_point(row, col).y,
                geometry.corner_point(row, col).z,
            ]
            for row, col in keys
        ],
        dtype=np.float64,
    )
    previous_uv = _project_points(object_points, rvec, previous_tvec, K, dist)
    current_uv = _project_points(object_points, rvec, current_tvec, K, dist)

    identities = []
    detection_corners = []
    for idx, ((row, col), xyz, prev_uv, curr_uv) in enumerate(
        zip(keys, object_points, previous_uv, current_uv)
    ):
        identity = cpp.GlobalCornerIdentity()
        identity.global_row = int(row)
        identity.global_col = int(col)
        identity.xyz_mm = [float(v) for v in xyz]
        identity.uv = [float(prev_uv[0]), float(prev_uv[1])]
        identity.votes = 8
        identities.append(identity)

        point = cpp.Point2f()
        point.x = float(curr_uv[0])
        point.y = float(curr_uv[1])
        corner = cpp.GridCorner()
        corner.i = idx % 6
        corner.j = idx // 6
        corner.uv = point
        detection_corners.append(corner)

    detection = cpp.CheckerboardDetection()
    detection.corners = detection_corners
    detection.tracking = True

    cfg = TrackerConfig(
        fast_persistent_dense_refine_enabled=False,
        pose_depth_filter_reprojection_guard_px=0.0,
    )
    matcher = cpp.create_persistent_matcher(cfg)
    matcher.replace_identities(identities, 0)
    native_filter = cpp.create_pose_depth_filter(K, dist, cfg)
    manual_filter = cpp.create_pose_depth_filter(K, dist, cfg)

    native_filter.update(
        rvec=rvec,
        tvec=previous_tvec,
        object_points=object_points,
        image_points=previous_uv,
    )
    manual_filter.update(
        rvec=rvec,
        tvec=previous_tvec,
        object_points=object_points,
        image_points=previous_uv,
    )

    result = matcher.estimate_fast_pose(
        detection,
        geometry,
        1,
        K,
        dist,
        rvec,
        previous_tvec,
        0.0,
        rvec,
        previous_tvec,
        0,
        native_filter,
    )
    assert result.success, result.reason
    assert result.depth_filter_available
    assert result.accepted_state.evaluated
    assert result.accepted_state.reliable_pose
    assert result.accepted_state.accepted_pose_frame == 1
    assert result.accepted_state.max_pts_seen == result.depth_filtered_pose.num_inliers
    assert result.accepted_state.last_good_reproj_px == pytest.approx(
        result.depth_filtered_pose.reprojection_mean_px,
        abs=2.0e-7,
    )
    assert result.persistence_refresh_available
    assert result.persistence_refresh_frame == 1
    assert result.persistence_refresh_count == len(
        result.persistence_refresh_identities
    )
    assert result.persistence_refresh_count == len(result.visual_corners)
    assert {
        (int(identity.global_row), int(identity.global_col))
        for identity in result.persistence_refresh_identities
    } == {
        (int(corner.global_row), int(corner.global_col))
        for corner in result.visual_corners
    }

    filter_points = result.pose.points or result.points
    filter_object_points = np.asarray(
        [point.xyz_mm for point in filter_points],
        dtype=np.float64,
    ).reshape(-1, 3)
    filter_image_points = np.asarray(
        [point.uv for point in filter_points],
        dtype=np.float64,
    ).reshape(-1, 2)
    manual = manual_filter.update(
        rvec=np.asarray(result.pose.rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(result.pose.tvec, dtype=np.float64).reshape(3, 1),
        object_points=filter_object_points,
        image_points=filter_image_points,
    )

    assert result.depth_filter_result.filtered_z_mm == pytest.approx(
        manual.filtered_z_mm,
        abs=2.0e-7,
    )
    assert result.depth_filtered_pose.tvec[2] == pytest.approx(
        manual.filtered_z_mm,
        abs=2.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(result.accepted_state.tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(result.depth_filtered_pose.tvec, dtype=np.float64).reshape(3, 1),
        atol=2.0e-7,
        rtol=0.0,
    )


def test_cpp_tracker_engine_switch_routes_process_frame() -> None:
    pytest.importorskip("tracking.hydramarker.backend.cpp_impl")
    from tracking.hydramarker.config import TrackerConfig
    from tracking.hydramarker.tracker import HydraTracker

    root = Path(__file__).resolve().parents[1]
    field_path = root / "data" / "drill" / "PRO6350M" / "marker_10x8_p3_6mm.field"
    geometry_path = (
        root
        / "data"
        / "drill"
        / "PRO6350M"
        / "drill_marker_geometry_sfm_10x8_3x3_6mm.json"
    )
    K = np.eye(3, dtype=np.float64)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    python_tracker = HydraTracker(
        str(field_path),
        str(geometry_path),
        K,
        None,
        TrackerConfig(),
    )
    python_result = python_tracker.process_frame(frame, run_detection=False)
    assert "cpp_tracker_engine_count" not in python_result.timings_ms

    cpp_tracker = HydraTracker(
        str(field_path),
        str(geometry_path),
        K,
        None,
        TrackerConfig(cpp_tracker_engine_enabled=True),
    )
    cpp_result = cpp_tracker.process_frame(frame, run_detection=False)
    assert cpp_result.timings_ms["cpp_tracker_engine_count"] == pytest.approx(1.0)
    assert cpp_result.timings_ms[
        "cpp_tracker_engine_current_pose_accepted_count"
    ] == pytest.approx(0.0)
    assert cpp_result.timings_ms[
        "cpp_tracker_engine_has_accepted_pose_count"
    ] == pytest.approx(0.0)
    assert cpp_tracker.frame_index == 1
    assert cpp_tracker._last_accepted_pose_frame == -1
    assert cpp_tracker.pose_tracker.rvec is None

    cpp_tracker.reset()
    assert cpp_tracker.frame_index == 0


def test_cpp_tracker_engine_result_carries_corner_lists() -> None:
    cpp = pytest.importorskip("tracking.hydramarker.backend.cpp_impl")
    from tracking.hydramarker.config import TrackerConfig
    from tracking.hydramarker.tracker import HydraTracker

    root = Path(__file__).resolve().parents[1]
    field_path = root / "data" / "drill" / "PRO6350M" / "marker_10x8_p3_6mm.field"
    geometry_path = (
        root
        / "data"
        / "drill"
        / "PRO6350M"
        / "drill_marker_geometry_sfm_10x8_3x3_6mm.json"
    )
    tracker = HydraTracker(
        str(field_path),
        str(geometry_path),
        np.eye(3, dtype=np.float64),
        None,
        TrackerConfig(cpp_tracker_engine_enabled=True),
    )

    detected = cpp.DetectedCorner()
    detected.local_row = 3
    detected.local_col = 4
    detected.uv = (12.5, 34.5)

    corner = cpp.TrackerCorner()
    corner.local_row = 3
    corner.local_col = 4
    corner.global_row = 8
    corner.global_col = 9
    corner.xyz_mm = (1.0, 2.0, 3.0)
    corner.uv = (12.5, 34.5)
    corner.votes = 2

    cpp_result = cpp.TrackerFrameResult()
    cpp_result.success = True
    cpp_result.mode = cpp.TrackerMode.TRACKING
    cpp_result.pose_source = cpp.PoseSource.DECODE
    cpp_result.message = "ok"
    cpp_result.detection_valid = True
    cpp_result.detection_corners = [detected]
    cpp_result.corners = [corner]
    cpp_result.correspondence_corners = [corner]
    cpp_result.rvec = [0.0, 0.0, 0.0]
    cpp_result.tvec = [0.0, 0.0, 100.0]
    cpp_result.T_marker_camera = np.eye(4, dtype=np.float64).reshape(-1).tolist()
    cpp_result.pose_tracker_has_pose = True
    cpp_result.pose_tracker_rvec = cpp_result.rvec
    cpp_result.pose_tracker_tvec = cpp_result.tvec
    cpp_result.pose_tracker_T_marker_camera = cpp_result.T_marker_camera

    result = tracker._tracker_result_from_cpp_engine(cpp_result)

    assert len(result.detection_corners) == 1
    assert result.detection_corners[0].uv == pytest.approx((12.5, 34.5))
    assert len(result.corners) == 1
    assert result.corners[0].global_row == 8
    assert len(result.correspondence_corners) == 1
    assert result.correspondence_corners[0].votes == 2
    np.testing.assert_allclose(tracker.pose_tracker.tvec, [[0.0], [0.0], [100.0]])


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


def test_cpp_plateau_pose_prior_matches_python_static_case() -> None:
    pytest.importorskip("tracking.hydramarker.backend.cpp_impl")
    from tracking.hydramarker.backend import cpp_impl as cpp

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

    py_result = solve_plateau_pose_prior(
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

    cfg = cpp.PlateauPosePriorConfig()
    cfg.static_max_excess_px = 0.12
    cfg.min_positive_z_correction_mm = 0.02
    cfg.max_positive_z_correction_mm = 1.0
    cpp_result = cpp.solve_plateau_pose_prior(
        object_points,
        image_points,
        K,
        dist,
        raw_rvec,
        raw_tvec,
        seed_rvec,
        seed_tvec,
        cfg,
    )

    assert cpp_result.success
    assert cpp_result.method == py_result.method == "static"
    assert cpp_result.delta_z_mm == pytest.approx(py_result.delta_z_mm)
    assert cpp_result.reprojection_excess_px == pytest.approx(
        py_result.reprojection_excess_px,
        abs=1.0e-9,
    )
    np.testing.assert_allclose(
        np.asarray(cpp_result.tvec, dtype=np.float64).reshape(3, 1),
        py_result.tvec,
        atol=1.0e-12,
        rtol=0.0,
    )
