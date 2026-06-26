"""C++ tracker facade tests for pose filtering and result conversion.

The tests verify that Python code reaches the native depth-filter, fast-path
pose transaction, tracker-engine result conversion, and plateau-prior bindings
without depending on retired Python tracker implementations.
"""

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


def test_cpp_fast_pose_transaction_can_apply_depth_filter() -> None:
    cpp = pytest.importorskip("tracking.hydramarker.tracker")
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


def test_hydratracker_routes_process_frame_through_cpp_engine() -> None:
    pytest.importorskip("tracking.hydramarker.tracker")
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

    tracker = HydraTracker(
        str(field_path),
        str(geometry_path),
        K,
        None,
        TrackerConfig(),
    )

    cpp_result = tracker.process_frame(frame, run_detection=False)
    assert cpp_result.timings_ms["cpp_tracker_engine_count"] == pytest.approx(1.0)
    assert cpp_result.timings_ms[
        "cpp_tracker_engine_current_pose_accepted_count"
    ] == pytest.approx(0.0)
    assert cpp_result.timings_ms[
        "cpp_tracker_engine_has_accepted_pose_count"
    ] == pytest.approx(0.0)
    assert tracker.frame_index == 1
    assert tracker._last_accepted_pose_frame == -1
    assert tracker.pose_tracker.rvec is None

    tracker.reset()
    assert tracker.frame_index == 0


def test_cpp_tracker_engine_result_carries_corner_lists() -> None:
    cpp = pytest.importorskip("tracking.hydramarker.tracker")
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
        TrackerConfig(),
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


def test_cpp_plateau_pose_prior_accepts_static_case() -> None:
    cpp = pytest.importorskip("tracking.hydramarker.tracker")

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

    cfg = cpp.PlateauPosePriorConfig()
    cfg.static_max_excess_px = 0.12
    cfg.min_positive_z_correction_mm = 0.02
    cfg.max_positive_z_correction_mm = 1.0
    result = cpp.solve_plateau_pose_prior(
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

    assert result.success
    assert result.method == "static"
    assert result.delta_z_mm == pytest.approx(0.5)
    assert result.reprojection_excess_px <= 0.12
    np.testing.assert_allclose(
        np.asarray(result.tvec, dtype=np.float64).reshape(3, 1),
        seed_tvec,
        atol=1.0e-12,
        rtol=0.0,
    )
