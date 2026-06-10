from __future__ import annotations

import json
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("QT_API", "pyside6")


CHECKERBOARD_PATTERN = (9, 9)  # inner corners: 10x10 printed cells -> 9x9 corners
CHECKERBOARD_SQUARE_SIZE_MM = 10.0
BOARD_REFINE_TARGET_FRAMES = 25
BOARD_REFINE_MIN_FRAMES = 8
BOARD_REFINE_MAX_ATTEMPTS = 90
BOARD_AXIS_LENGTH_MM = 30.0

COMPONENTS = (
    ("x", "x", "#1f77b4"),
    ("y", "y", "#2ca02c"),
    ("z", "z", "#d62728"),
)


@dataclass
class BoardPoseCalibration:
    rvec_cb: np.ndarray
    tvec_cb_mm: np.ndarray
    T_C_B: np.ndarray
    T_B_C: np.ndarray
    corners_uv: np.ndarray
    reproj_mean_px: float
    reproj_median_px: float
    reproj_p95_px: float
    reproj_max_px: float
    inlier_count: int
    total_count: int
    collected_frames: int
    mean_corner_std_px: float
    max_corner_std_px: float


def _ensure_src_on_path() -> None:
    tracking_root = Path(__file__).resolve().parents[2]
    src_root = tracking_root.parent
    src = str(src_root)
    sys.path = [p for p in sys.path if str(p) != src]
    sys.path.insert(0, src)


def _forget_tracking_modules() -> None:
    for name in (
        "tracking.hydramarker.tests.test_tracker_realsense",
        "tracking.hydramarker.tracker",
        "tracking.hydramarker.map_pose_tracker",
    ):
        sys.modules.pop(name, None)


def _force_dense_refine_config(tracker) -> None:
    cfg = getattr(tracker, "config", None)
    if cfg is None:
        return

    dense_settings = {
        "fast_persistent_dense_refine_enabled": True,
        "fast_persistent_dense_min_points": 24,
        "fast_persistent_dense_match_max_px": 3.0,
        "fast_persistent_dense_min_second_best_margin_px": 2.0,
        "fast_persistent_dense_max_median_px": 1.2,
        "fast_persistent_dense_max_p90_px": 2.5,
        "fast_persistent_dense_min_image_coverage": 0.35,
        "fast_persistent_dense_min_object_span_mm": 12.0,
        "fast_persistent_dense_min_distinct_rows": 2,
        "fast_persistent_dense_min_distinct_cols": 2,
        "fast_persistent_dense_pose_solver": "direct_prior",
    }
    for name, value in dense_settings.items():
        setattr(cfg, name, value)

    print(
        "[debug_tracker_translation] dense refine enabled "
        f"(min_points={getattr(cfg, 'fast_persistent_dense_min_points', '?')}, "
        f"max_px={getattr(cfg, 'fast_persistent_dense_match_max_px', '?')}, "
        f"solver={getattr(cfg, 'fast_persistent_dense_pose_solver', '?')})"
    )


def _load_qt_widgets():
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    return QApplication, QFileDialog, QMessageBox


def _load_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _to_float(value) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _to_int(value, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _qt_app():
    QApplication, _, _ = _load_qt_widgets()
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def select_jsonl_with_qt() -> Path | None:
    _qt_app()
    _, QFileDialog, _ = _load_qt_widgets()

    script_path = Path(__file__).resolve()
    default_dir = script_path.parents[1] / "tests" / "hydramarker_tracker_runs"
    if not default_dir.exists():
        default_dir = Path.cwd()

    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select HydraTracker run log",
        str(default_dir),
        "HydraTracker JSONL (*.jsonl);;All Files (*)",
    )

    if not path:
        return None
    return Path(path)


def checkerboard_object_points_mm() -> np.ndarray:
    cols, rows = CHECKERBOARD_PATTERN
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj = np.zeros((rows * cols, 3), dtype=np.float64)
    obj[:, :2] = grid.astype(np.float64) * CHECKERBOARD_SQUARE_SIZE_MM
    return obj


def _make_transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    import cv2

    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _camera_tvec_to_board_mm(tvec_camera_mm: np.ndarray, T_B_C: np.ndarray) -> np.ndarray:
    p_c = np.ones(4, dtype=np.float64)
    p_c[:3] = np.asarray(tvec_camera_mm, dtype=np.float64).reshape(3)
    p_b = np.asarray(T_B_C, dtype=np.float64).reshape(4, 4) @ p_c
    return p_b[:3]


def _reprojection_errors_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    import cv2

    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        dist,
    )
    projected = projected.reshape(-1, 2)
    measured = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    return np.linalg.norm(projected - measured, axis=1)


def _pose_stats(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> dict[str, Any]:
    errors = _reprojection_errors_px(object_points, image_points, rvec, tvec, K, dist)
    return {
        "errors": errors,
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
    }


def _refined_pose_candidates(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    import cv2

    candidates = [
        (
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        )
    ]

    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            lm_rvec, lm_tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                K,
                dist,
                candidates[0][0].copy(),
                candidates[0][1].copy(),
            )
            candidates.append((lm_rvec.reshape(3, 1), lm_tvec.reshape(3, 1)))
        except cv2.error:
            pass

    if hasattr(cv2, "solvePnPRefineVVS"):
        try:
            vvs_rvec, vvs_tvec = cv2.solvePnPRefineVVS(
                object_points,
                image_points,
                K,
                dist,
                candidates[0][0].copy(),
                candidates[0][1].copy(),
            )
            candidates.append((vvs_rvec.reshape(3, 1), vvs_tvec.reshape(3, 1)))
        except cv2.error:
            pass

    return candidates


def _solve_best_board_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import cv2

    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)

    flags: list[int] = []
    if hasattr(cv2, "SOLVEPNP_SQPNP"):
        flags.append(int(cv2.SOLVEPNP_SQPNP))
    if hasattr(cv2, "SOLVEPNP_IPPE"):
        flags.append(int(cv2.SOLVEPNP_IPPE))
    flags.append(int(cv2.SOLVEPNP_ITERATIVE))

    best: tuple[float, np.ndarray, np.ndarray, dict[str, Any]] | None = None
    for flag in flags:
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                K,
                dist,
                flags=flag,
            )
        except cv2.error:
            continue
        if not success:
            continue

        for cand_rvec, cand_tvec in _refined_pose_candidates(
            object_points,
            image_points,
            K,
            dist,
            rvec,
            tvec,
        ):
            stats = _pose_stats(
                object_points,
                image_points,
                cand_rvec,
                cand_tvec,
                K,
                dist,
            )
            # Prefer physically plausible camera-facing candidates, but keep a
            # fallback so diagnostics remain visible if OpenCV returns a flipped
            # planar solution.
            penalty = 0.0 if float(cand_tvec.reshape(3)[2]) > 0.0 else 1000.0
            score = float(stats["mean"]) + penalty
            if best is None or score < best[0]:
                stats["pnp_flag"] = int(flag)
                best = (score, cand_rvec, cand_tvec, stats)

    if best is None:
        raise RuntimeError("Could not estimate checkerboard pose.")

    _, best_rvec, best_tvec, best_stats = best
    return best_rvec.reshape(3, 1), best_tvec.reshape(3, 1), best_stats


def estimate_board_pose_from_corners(
    corners_uv: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    collected_frames: int = 1,
    mean_corner_std_px: float = 0.0,
    max_corner_std_px: float = 0.0,
) -> BoardPoseCalibration:
    object_points = checkerboard_object_points_mm()
    image_points = np.asarray(corners_uv, dtype=np.float64).reshape(-1, 2)

    rvec, tvec, stats = _solve_best_board_pose(object_points, image_points, K, dist)
    errors = np.asarray(stats["errors"], dtype=np.float64)

    inlier_count = len(errors)
    total_count = len(errors)
    if total_count >= 20:
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median)))
        robust_sigma = 1.4826 * mad
        trim_threshold = max(0.75, median + 4.0 * robust_sigma)
        inlier_mask = errors <= trim_threshold
        if int(np.count_nonzero(inlier_mask)) >= max(12, int(0.75 * total_count)):
            inlier_count = int(np.count_nonzero(inlier_mask))
            if inlier_count < total_count:
                rvec_trim, tvec_trim, _ = _solve_best_board_pose(
                    object_points[inlier_mask],
                    image_points[inlier_mask],
                    K,
                    dist,
                )
                full_stats = _pose_stats(
                    object_points,
                    image_points,
                    rvec_trim,
                    tvec_trim,
                    K,
                    dist,
                )
                if float(full_stats["mean"]) <= float(stats["mean"]) * 1.25:
                    rvec, tvec, stats = rvec_trim, tvec_trim, full_stats
                    errors = np.asarray(stats["errors"], dtype=np.float64)

    T_C_B = _make_transform_from_rvec_tvec(rvec, tvec)
    T_B_C = _invert_transform(T_C_B)

    return BoardPoseCalibration(
        rvec_cb=rvec.reshape(3, 1),
        tvec_cb_mm=tvec.reshape(3, 1),
        T_C_B=T_C_B,
        T_B_C=T_B_C,
        corners_uv=image_points.reshape(-1, 2),
        reproj_mean_px=float(np.mean(errors)),
        reproj_median_px=float(np.median(errors)),
        reproj_p95_px=float(np.percentile(errors, 95)),
        reproj_max_px=float(np.max(errors)),
        inlier_count=int(inlier_count),
        total_count=int(total_count),
        collected_frames=int(collected_frames),
        mean_corner_std_px=float(mean_corner_std_px),
        max_corner_std_px=float(max_corner_std_px),
    )


def detect_checkerboard_corners(frame: np.ndarray) -> np.ndarray | None:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pattern = CHECKERBOARD_PATTERN

    corners = None
    ok = False
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
        try:
            ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags)
        except cv2.error:
            ok, corners = False, None

    if not ok:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if ok:
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                80,
                1e-4,
            )
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if not ok or corners is None:
        return None

    return np.asarray(corners, dtype=np.float64).reshape(-1, 2)


def put_text_cv(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 255),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    import cv2

    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_board_axes(
    vis: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> None:
    import cv2

    if hasattr(cv2, "drawFrameAxes"):
        try:
            cv2.drawFrameAxes(vis, K, dist, rvec, tvec, BOARD_AXIS_LENGTH_MM, 3)
            return
        except cv2.error:
            pass

    axis = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [BOARD_AXIS_LENGTH_MM, 0.0, 0.0],
            [0.0, BOARD_AXIS_LENGTH_MM, 0.0],
            [0.0, 0.0, BOARD_AXIS_LENGTH_MM],
        ],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    pts = np.round(projected.reshape(-1, 2)).astype(int)
    origin = tuple(pts[0])
    cv2.line(vis, origin, tuple(pts[1]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(vis, origin, tuple(pts[2]), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(vis, origin, tuple(pts[3]), (255, 0, 0), 3, cv2.LINE_AA)


def draw_checkerboard_overlay(
    frame: np.ndarray,
    corners_uv: np.ndarray | None,
    pose: BoardPoseCalibration | None,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    status: str,
    collecting: str = "",
) -> np.ndarray:
    import cv2

    vis = frame.copy()
    if corners_uv is not None:
        cv2.drawChessboardCorners(
            vis,
            CHECKERBOARD_PATTERN,
            np.asarray(corners_uv, dtype=np.float32).reshape(-1, 1, 2),
            True,
        )

    if pose is not None:
        draw_board_axes(vis, K, dist, pose.rvec_cb, pose.tvec_cb_mm)

    overlay = vis.copy()
    cv2.rectangle(overlay, (20, 20), (1080, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, vis, 0.42, 0, vis)

    put_text_cv(vis, "Checkerboard pose calibration", (40, 55), scale=0.68)
    put_text_cv(vis, status, (40, 88), scale=0.50)
    if pose is None:
        detail = "SPACE=accept current detection | q/ESC=quit"
    else:
        detail = (
            f"mean={pose.reproj_mean_px:.3f}px p95={pose.reproj_p95_px:.3f}px "
            f"max={pose.reproj_max_px:.3f}px frames={pose.collected_frames}"
        )
    put_text_cv(vis, detail, (40, 120), scale=0.48)
    if collecting:
        put_text_cv(vis, collecting, (40, 150), color=(0, 255, 0), scale=0.48)
    else:
        put_text_cv(
            vis,
            "Board: 10x10 cells, 1 cm cells, OpenCV pattern 9x9 inner corners",
            (40, 150),
            color=(180, 220, 255),
            scale=0.45,
        )
    return vis


def collect_refined_board_pose(
    pipe,
    K: np.ndarray,
    dist: np.ndarray,
    window_name: str,
    first_corners: np.ndarray,
) -> BoardPoseCalibration | None:
    import cv2

    collected: list[np.ndarray] = [np.asarray(first_corners, dtype=np.float64).reshape(-1, 2)]
    reference = collected[0]
    last_frame = None
    attempts = 0

    while len(collected) < BOARD_REFINE_TARGET_FRAMES and attempts < BOARD_REFINE_MAX_ATTEMPTS:
        attempts += 1
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        last_frame = frame
        corners = detect_checkerboard_corners(frame)
        if corners is not None:
            direct_err = float(np.mean(np.linalg.norm(corners - reference, axis=1)))
            reverse_err = float(np.mean(np.linalg.norm(corners[::-1] - reference, axis=1)))
            if reverse_err < direct_err:
                corners = corners[::-1]
            collected.append(corners)

        progress = (
            f"collecting stable detections {len(collected)}/{BOARD_REFINE_TARGET_FRAMES} "
            f"(attempt {attempts}/{BOARD_REFINE_MAX_ATTEMPTS})"
        )
        pose = None
        if corners is not None:
            try:
                pose = estimate_board_pose_from_corners(
                    corners,
                    K,
                    dist,
                    collected_frames=len(collected),
                )
            except RuntimeError:
                pose = None

        vis = draw_checkerboard_overlay(
            frame,
            corners,
            pose,
            K,
            dist,
            status="Keep camera and board still while the pose is refined.",
            collecting=progress,
        )
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return None

    if len(collected) < BOARD_REFINE_MIN_FRAMES:
        _qt_app()
        _, _, QMessageBox = _load_qt_widgets()
        QMessageBox.warning(
            None,
            "Checkerboard pose",
            (
                "Not enough stable checkerboard detections were collected.\n\n"
                f"Needed at least {BOARD_REFINE_MIN_FRAMES}, got {len(collected)}."
            ),
        )
        return None

    stack = np.stack(collected, axis=0)
    mean_corners = np.mean(stack, axis=0)
    corner_std = np.linalg.norm(np.std(stack, axis=0), axis=1)
    pose = estimate_board_pose_from_corners(
        mean_corners,
        K,
        dist,
        collected_frames=len(collected),
        mean_corner_std_px=float(np.mean(corner_std)),
        max_corner_std_px=float(np.max(corner_std)),
    )

    if last_frame is not None:
        vis = draw_checkerboard_overlay(
            last_frame,
            mean_corners,
            pose,
            K,
            dist,
            status="Refined checkerboard pose. Confirm the axes in the dialog.",
        )
        cv2.imshow(window_name, vis)
        cv2.waitKey(1)

    return pose


def confirm_board_pose(pose: BoardPoseCalibration) -> bool:
    _qt_app()
    _, _, QMessageBox = _load_qt_widgets()
    t = pose.tvec_cb_mm.reshape(3)
    reply = QMessageBox.question(
        None,
        "Checkerboard pose uebernehmen?",
        (
            "Checkerboard-Pose uebernehmen?\n\n"
            f"Reprojection mean: {pose.reproj_mean_px:.3f} px\n"
            f"Reprojection p95:  {pose.reproj_p95_px:.3f} px\n"
            f"Reprojection max:  {pose.reproj_max_px:.3f} px\n"
            f"Collected frames:  {pose.collected_frames}\n"
            f"Corner noise mean: {pose.mean_corner_std_px:.3f} px\n"
            f"Camera board tvec: x={t[0]:.1f} mm, y={t[1]:.1f} mm, z={t[2]:.1f} mm\n\n"
            "Yes: use this board pose and start the tracker view.\n"
            "No: repeat checkerboard detection."
        ),
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes,
    )
    return reply == QMessageBox.Yes


def calibrate_checkerboard_pose(
    pipe,
    K: np.ndarray,
    dist: np.ndarray,
) -> BoardPoseCalibration | None:
    import cv2

    window_name = "HydraTracker Translation Debug"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        corners = detect_checkerboard_corners(frame)
        pose = None
        status = "No full checkerboard detected."
        if corners is not None:
            try:
                pose = estimate_board_pose_from_corners(corners, K, dist)
                status = "Checkerboard detected. Check the axes, then press SPACE."
            except RuntimeError as exc:
                status = f"Checkerboard detected, pose failed: {exc}"

        vis = draw_checkerboard_overlay(frame, corners, pose, K, dist, status=status)
        cv2.imshow(window_name, vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return None
        if key == ord(" ") and corners is not None:
            refined_pose = collect_refined_board_pose(
                pipe,
                K,
                dist,
                window_name,
                corners,
            )
            if refined_pose is None:
                continue
            if confirm_board_pose(refined_pose):
                return refined_pose


def board_pose_record(board_pose: BoardPoseCalibration, run_id: str) -> dict[str, Any]:
    return {
        "type": "board_pose",
        "run_id": run_id,
        "coordinate_frame": "checkerboard",
        "pattern_inner_corners": list(CHECKERBOARD_PATTERN),
        "printed_cells": [10, 10],
        "square_size_mm": CHECKERBOARD_SQUARE_SIZE_MM,
        "rvec_cb": board_pose.rvec_cb.reshape(3).tolist(),
        "tvec_cb_mm": board_pose.tvec_cb_mm.reshape(3).tolist(),
        "T_C_B": board_pose.T_C_B.tolist(),
        "T_B_C": board_pose.T_B_C.tolist(),
        "reprojection": {
            "mean_px": board_pose.reproj_mean_px,
            "median_px": board_pose.reproj_median_px,
            "p95_px": board_pose.reproj_p95_px,
            "max_px": board_pose.reproj_max_px,
            "inlier_count": board_pose.inlier_count,
            "total_count": board_pose.total_count,
            "collected_frames": board_pose.collected_frames,
            "mean_corner_std_px": board_pose.mean_corner_std_px,
            "max_corner_std_px": board_pose.max_corner_std_px,
        },
    }


def write_board_pose_to_live_log(live, board_pose: BoardPoseCalibration) -> None:
    if not getattr(live, "_log_active", False):
        return
    live._write_json_record(
        board_pose_record(board_pose, str(getattr(live, "_log_run_id", "")))
    )


def load_live_tracker_module():
    _ensure_src_on_path()
    _forget_tracking_modules()

    module_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "test_tracker_realsense.py"
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "tracking.hydramarker.tests.test_tracker_realsense",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create import spec for {module_path}")
        live = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = live
        spec.loader.exec_module(live)
        print(f"[debug_tracker_translation] live module -> {module_path}")
    except Exception as exc:
        raise RuntimeError(
            "Could not import the RealSense tracker test module. "
            "Run this from the tracking project environment."
        ) from exc

    script_path = Path(__file__).resolve()
    live.RUN_LOG_DIR = script_path.parents[1] / "tests" / "hydramarker_tracker_runs"
    return live


def run_live_tracker_translation() -> Path | None:
    import cv2

    live = load_live_tracker_module()

    field_path = live.choose_file_qt("Select HydraMarker .field file", "HydraMarker field (*.field)")
    marker_json_path = live.choose_file_qt("Select marker .json file", "Marker JSON (*.json)")

    pipe, profile = live.create_realsense_pipeline()
    K_rgb, dist_rgb = live.load_tracker_camera_calibration(profile)
    if hasattr(live, "_camera_intrinsics_info"):
        live._camera_intrinsics_info.update(
            {
                "debug_rectification_mode": "disabled_raw_realsense",
                "debug_rectification_enabled": False,
                "debug_tracker_uses_loaded_camera_calibration": bool(
                    live._camera_intrinsics_info.get("camera_source") == "opencv_calibration_npz"
                ),
                "tracker_K": K_rgb.tolist(),
                "tracker_dist_coeffs": dist_rgb.reshape(-1).tolist(),
            }
        )

    recorded_log_path: Path | None = None
    window_name = "HydraTracker Translation Debug"

    try:
        board_pose = calibrate_checkerboard_pose(
            pipe,
            K_rgb,
            dist_rgb,
        )
        if board_pose is None:
            return None
        if hasattr(live, "set_debug_board_transform"):
            live.set_debug_board_transform(board_pose.T_B_C)

        tracker = live.make_tracker(field_path, marker_json_path, K_rgb, dist_rgb)
        _force_dense_refine_config(tracker)
        preview_detector = live.make_idle_preview_detector()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_idx = 0
        last_mode = None
        last_success = None
        last_message = None
        app_state = live.APP_IDLE
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
            app_state = live.APP_IDLE
            acquire_start_frame = 0
            provisional_start_frame = 0
            last_candidate_frame = 0
            stale_pose_frames = 0
            tracking_armed = bool(keep_armed)
            auto_acquire_blocked = bool(block_auto_acquire)
            last_mode = None
            last_success = None
            last_message = None
            print(f"[debug_tracker_translation] idle ({reason})")

        def start_acquire(*, manual: bool = False) -> None:
            nonlocal app_state, acquire_start_frame, provisional_start_frame
            nonlocal last_candidate_frame, stale_pose_frames
            nonlocal tracking_armed, auto_acquire_blocked
            nonlocal last_mode, last_success, last_message
            tracker.reset()
            preview_detector.reset_tracking()
            app_state = live.APP_ACQUIRE
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
            print(f"[debug_tracker_translation] acquire started ({reason})")

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
                run_detection=(app_state != live.APP_IDLE),
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0

            preview_corners: list[tuple[float, float]] = []
            preview_ms = 0.0
            if app_state == live.APP_IDLE:
                preview_t0 = time.perf_counter()
                preview_detection = preview_detector.detect(frame)
                preview_ms = (time.perf_counter() - preview_t0) * 1000.0
                preview_corners = live.preview_corner_uvs(preview_detection)
                if len(preview_corners) < live.PROVISIONAL_MIN_CORNERS:
                    auto_acquire_blocked = False
                if (
                    tracking_armed
                    and not auto_acquire_blocked
                    and len(preview_corners) >= live.IDLE_PREVIEW_AUTO_ACQUIRE_CORNERS
                ):
                    start_acquire(manual=False)

            det_count = len(getattr(result, "detection_corners", []))
            if app_state in (live.APP_ACQUIRE, live.APP_PROVISIONAL):
                active_frames = max(0, frame_idx - acquire_start_frame + 1)
                if live.has_fresh_pose(result):
                    app_state = live.APP_TRACKING
                    stale_pose_frames = 0
                    print("[debug_tracker_translation] tracking locked")
                elif det_count >= live.PROVISIONAL_MIN_CORNERS:
                    last_candidate_frame = frame_idx
                    if app_state != live.APP_PROVISIONAL:
                        app_state = live.APP_PROVISIONAL
                        provisional_start_frame = frame_idx
                        print(f"[debug_tracker_translation] provisional fragment det={det_count}")
                elif app_state == live.APP_ACQUIRE and active_frames >= live.ACQUIRE_TIMEOUT_FRAMES:
                    enter_idle("acquire timeout", block_auto_acquire=True)
                elif app_state == live.APP_PROVISIONAL:
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
                    if no_candidate_frames >= live.PROVISIONAL_STALE_TIMEOUT_FRAMES:
                        enter_idle("provisional stale", block_auto_acquire=True)
                    elif provisional_frames >= live.PROVISIONAL_TOTAL_TIMEOUT_FRAMES:
                        enter_idle("provisional timeout", block_auto_acquire=True)
            elif app_state == live.APP_TRACKING:
                if live.has_fresh_pose(result):
                    stale_pose_frames = 0
                else:
                    stale_pose_frames += 1
                    if stale_pose_frames >= live.TRACKING_STALE_TO_IDLE_FRAMES:
                        enter_idle("tracking lost", keep_armed=True)

            mode_changed = last_mode != result.mode.value
            success_changed = last_success != bool(result.success)
            message_changed = last_message != result.message
            force_log = (
                mode_changed
                or success_changed
                or message_changed
                or (not result.success and app_state != live.APP_IDLE)
            )
            live.log_console(frame_idx, result, tracker, force=force_log)

            last_mode = result.mode.value
            last_success = bool(result.success)
            last_message = result.message

            draw_t0 = time.perf_counter()
            vis = live.draw_debug(
                frame.copy(),
                result,
                K_rgb,
                dist_rgb,
                frame_idx,
                tracker,
            )
            if app_state == live.APP_IDLE:
                live.draw_preview_corners(vis, preview_corners)
            active_frames_for_display = (
                max(0, frame_idx - acquire_start_frame + 1)
                if app_state in (live.APP_ACQUIRE, live.APP_PROVISIONAL)
                and acquire_start_frame > 0
                else 0
            )
            live.draw_app_state(
                vis,
                app_state,
                active_frames_for_display,
                stale_pose_frames,
                tracking_armed=tracking_armed,
                preview_count=len(preview_corners),
                preview_ms=preview_ms,
                auto_blocked=auto_acquire_blocked,
            )

            state_line = (
                "Board pose fixed | camera=raw_realsense_distortion | s=start/stop tracking | "
                f"SPACE={'STOP recording and analyze' if live._log_active else 'START recording'} | q=quit"
            )
            put_text_cv(vis, state_line, (25, 245), color=(0, 255, 255), scale=0.48)
            draw_ms = (time.perf_counter() - draw_t0) * 1000.0

            live.log_frame(frame_idx, result, wall_ms, tracker, draw_ms)

            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                if live._log_active:
                    recorded_log_path = Path(live._log_path)
                    live.log_close()
                break
            if key == ord("r"):
                enter_idle("reset", keep_armed=False)
            if key == ord("s"):
                if app_state == live.APP_IDLE:
                    start_acquire(manual=True)
                else:
                    enter_idle("manual stop", keep_armed=False)
            if key == ord(" "):
                if live._log_active:
                    recorded_log_path = Path(live._log_path)
                    live.log_close()
                    break
                live.log_open(field_path, marker_json_path, tracker)
                write_board_pose_to_live_log(live, board_pose)

    finally:
        if getattr(live, "_log_active", False):
            if recorded_log_path is None and getattr(live, "_log_path", None) is not None:
                recorded_log_path = Path(live._log_path)
            live.log_close()
        if hasattr(live, "set_debug_board_transform"):
            live.set_debug_board_transform(None)
        pipe.stop()
        cv2.destroyAllWindows()

    return recorded_log_path


def _parse_board_pose_record(record: dict) -> dict[str, Any] | None:
    T_B_C = record.get("T_B_C")
    if T_B_C is None:
        return None
    try:
        return {
            "T_B_C": np.asarray(T_B_C, dtype=np.float64).reshape(4, 4),
            "T_C_B": np.asarray(record.get("T_C_B"), dtype=np.float64).reshape(4, 4)
            if record.get("T_C_B") is not None
            else None,
            "record": record,
        }
    except Exception:
        return None


def load_tracker_run(path: Path) -> dict:
    frames: list[int] = []
    success: list[int] = []
    pose_is_fresh: list[int] = []
    pose_source: list[str] = []
    tvec_camera: list[list[float]] = []
    logged_delta: list[float] = []

    run_id = path.stem
    run_timestamp = ""
    columns: list[str] = []
    summary: dict = {}
    board_pose: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at line {line_no}: {exc}") from exc

            record_type = record.get("type")

            if record_type == "run_start":
                run_id = str(record.get("run_id") or run_id)
                run_timestamp = str(record.get("timestamp") or "")
                columns = list(record.get("columns") or [])
                continue

            if record_type == "board_pose":
                parsed = _parse_board_pose_record(record)
                if parsed is not None:
                    board_pose = parsed
                continue

            if record_type == "run_summary":
                summary = dict(record.get("summary") or {})
                continue

            if record_type != "frame":
                continue

            data = record.get("data") or {}
            frames.append(_to_int(data.get("frame"), default=len(frames)))
            success.append(_to_int(data.get("success"), default=0))
            pose_is_fresh.append(_to_int(data.get("pose_is_fresh"), default=0))
            pose_source.append(str(data.get("pose_source") or ""))
            logged_delta.append(_to_float(data.get("pose_translation_delta_mm")))
            tvec_camera.append(
                [
                    _to_float(data.get("tvec_x_mm")),
                    _to_float(data.get("tvec_y_mm")),
                    _to_float(data.get("tvec_z_mm")),
                ]
            )

    if not frames:
        raise RuntimeError(f"No frame records found in:\n{path}")

    frame_arr = np.asarray(frames, dtype=np.int64)
    tvec_camera_arr = np.asarray(tvec_camera, dtype=np.float64).reshape(-1, 3)
    has_tvec = np.all(np.isfinite(tvec_camera_arr), axis=1)
    success_arr = np.asarray(success, dtype=np.int64)
    fresh_arr = np.asarray(pose_is_fresh, dtype=np.int64)

    if not np.any(has_tvec):
        available = ", ".join(columns) if columns else "unknown"
        raise RuntimeError(
            "This run log has no tvec_x_mm/tvec_y_mm/tvec_z_mm values.\n"
            "Record a new run with the updated tracker logger first.\n\n"
            f"Available columns:\n{available}"
        )

    coordinate_frame = "camera"
    absolute_label = "T_C_T"
    tvec_abs = tvec_camera_arr.copy()
    if board_pose is not None:
        T_B_C = np.asarray(board_pose["T_B_C"], dtype=np.float64).reshape(4, 4)
        tvec_abs = np.full_like(tvec_camera_arr, np.nan, dtype=np.float64)
        for idx, p_c in enumerate(tvec_camera_arr):
            if np.all(np.isfinite(p_c)):
                tvec_abs[idx] = _camera_tvec_to_board_mm(p_c, T_B_C)
        coordinate_frame = "checkerboard"
        absolute_label = "T_B_T"

    has_tvec = np.all(np.isfinite(tvec_abs), axis=1)
    origin_idx = int(np.where(has_tvec)[0][0])
    origin_frame = int(frame_arr[origin_idx])
    origin_tvec = tvec_abs[origin_idx].copy()
    relative_tvec = tvec_abs - origin_tvec
    z_vs_y_slope_mm_per_100mm = np.nan
    if coordinate_frame == "checkerboard":
        valid_yz = has_tvec & np.isfinite(relative_tvec[:, 1]) & np.isfinite(relative_tvec[:, 2])
        if int(np.count_nonzero(valid_yz)) >= 8 and np.ptp(relative_tvec[valid_yz, 1]) > 1e-6:
            A = np.c_[relative_tvec[valid_yz, 1], np.ones(int(np.count_nonzero(valid_yz)))]
            slope, _ = np.linalg.lstsq(A, relative_tvec[valid_yz, 2], rcond=None)[0]
            z_vs_y_slope_mm_per_100mm = float(100.0 * slope)

    computed_delta = np.full(len(frame_arr), np.nan, dtype=np.float64)
    for idx in range(1, len(frame_arr)):
        if has_tvec[idx] and has_tvec[idx - 1]:
            computed_delta[idx] = float(np.linalg.norm(relative_tvec[idx] - relative_tvec[idx - 1]))

    logged_delta_arr = np.asarray(logged_delta, dtype=np.float64)
    plot_delta = np.where(np.isfinite(computed_delta), computed_delta, logged_delta_arr)

    return {
        "path": path,
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "summary": summary,
        "frames": frame_arr,
        "success": success_arr,
        "pose_is_fresh": fresh_arr,
        "pose_source": np.asarray(pose_source, dtype=object),
        "tvec": relative_tvec,
        "tvec_abs": tvec_abs,
        "origin_frame": origin_frame,
        "origin_tvec": origin_tvec,
        "has_tvec": has_tvec,
        "delta_mm": plot_delta,
        "board_pose": board_pose,
        "coordinate_frame": coordinate_frame,
        "absolute_label": absolute_label,
        "z_vs_y_slope_mm_per_100mm": z_vs_y_slope_mm_per_100mm,
    }


def contiguous_frame_ranges(frames: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    selected = frames[mask]
    if len(selected) == 0:
        return []

    ranges: list[tuple[int, int]] = []
    start = int(selected[0])
    prev = int(selected[0])

    for frame in selected[1:]:
        frame = int(frame)
        if frame == prev + 1:
            prev = frame
            continue

        ranges.append((start, prev))
        start = prev = frame

    ranges.append((start, prev))
    return ranges


def robust_peak_frames(frames: np.ndarray, delta_mm: np.ndarray, limit: int = 8) -> tuple[np.ndarray, float]:
    valid = delta_mm[np.isfinite(delta_mm)]
    if len(valid) < 8:
        return np.asarray([], dtype=np.int64), np.nan

    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(
        median + 5.0 * robust_sigma,
        float(np.percentile(valid, 99)),
    )

    peak_indices = np.where(np.isfinite(delta_mm) & (delta_mm >= threshold))[0]
    if len(peak_indices) > limit:
        order = np.argsort(delta_mm[peak_indices])[::-1][:limit]
        peak_indices = np.sort(peak_indices[order])

    return frames[peak_indices], threshold


def setup_plot_style(plt) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfd",
            "axes.edgecolor": "#d0d4dc",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "grid.color": "#d9dee8",
            "grid.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d0d4dc",
        }
    )


def plot_translation(run: dict) -> Path:
    plt = _load_pyplot()
    setup_plot_style(plt)

    path: Path = run["path"]
    frames: np.ndarray = run["frames"]
    tvec: np.ndarray = run["tvec"]
    has_tvec: np.ndarray = run["has_tvec"]
    success: np.ndarray = run["success"]
    fresh: np.ndarray = run["pose_is_fresh"]
    delta_mm: np.ndarray = run["delta_mm"]
    absolute_label = str(run.get("absolute_label", "T_C_T"))
    coordinate_frame = str(run.get("coordinate_frame", "camera"))

    missing_mask = (success == 0) | ~has_tvec
    held_mask = (success == 1) & has_tvec & (fresh == 0)
    missing_ranges = contiguous_frame_ranges(frames, missing_mask)
    held_ranges = contiguous_frame_ranges(frames, held_mask)
    peak_frames, peak_threshold = robust_peak_frames(frames, delta_mm)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15.5, 9.0),
        sharex=True,
        constrained_layout=True,
    )

    run_label = run["run_id"]
    if run["run_timestamp"]:
        run_label += f"  |  {run['run_timestamp']}"

    pose_pct = 100.0 * float(np.count_nonzero(success == 1)) / max(len(success), 1)
    origin_frame = int(run["origin_frame"])
    origin_tvec = np.asarray(run["origin_tvec"], dtype=np.float64).reshape(3)

    title = (
        f"HydraTracker relative translation components ({absolute_label}, {coordinate_frame} frame)\n"
        f"{run_label}"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for comp_idx, (_, label, color) in enumerate(COMPONENTS):
        ax = axes[comp_idx]
        values = tvec[:, comp_idx].copy()
        values[~has_tvec] = np.nan

        ax.plot(
            frames,
            values,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=2.8,
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=f"{absolute_label} {label}",
        )

        if np.any(np.isfinite(values)):
            first_value = float(values[np.where(np.isfinite(values))[0][0]])
            ax.axhline(first_value, color=color, alpha=0.22, linewidth=1.2, linestyle="--")

            value_range = float(np.nanmax(values) - np.nanmin(values))
            ax.set_title(f"{label.upper()} relative component   range={value_range:.2f} mm", loc="left")
        else:
            ax.set_title(f"{label.upper()} relative component", loc="left")

        for start, end in missing_ranges:
            ax.axvspan(start - 0.5, end + 0.5, color="#e45756", alpha=0.14, lw=0)

        for start, end in held_ranges:
            ax.axvspan(start - 0.5, end + 0.5, color="#f2b701", alpha=0.14, lw=0)

        for peak_frame in peak_frames:
            ax.axvline(int(peak_frame), color="#7f3c8d", alpha=0.38, linewidth=1.1)

        ax.set_ylabel(f"delta {absolute_label} {label} [mm]")
        ax.grid(True, axis="both")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("frame")

    info = (
        f"relative to frame {origin_frame} "
        f"({origin_tvec[0]:.2f}, {origin_tvec[1]:.2f}, {origin_tvec[2]:.2f}) mm   "
        f"frames={len(frames)}   pose={pose_pct:.1f}%   "
        f"missing={int(np.count_nonzero(missing_mask))}   held={int(np.count_nonzero(held_mask))}"
    )
    if np.isfinite(peak_threshold) and len(peak_frames) > 0:
        info += f"   peak threshold={peak_threshold:.2f} mm   peaks={', '.join(str(int(f)) for f in peak_frames)}"

    board_pose = run.get("board_pose")
    if board_pose is not None:
        reproj = (board_pose.get("record") or {}).get("reprojection") or {}
        if reproj:
            info += (
                f"   board reproj mean={float(reproj.get('mean_px', np.nan)):.3f}px"
                f" p95={float(reproj.get('p95_px', np.nan)):.3f}px"
            )
        z_vs_y = float(run.get("z_vs_y_slope_mm_per_100mm", np.nan))
        if np.isfinite(z_vs_y):
            info += f"   z~y slope={z_vs_y:.2f} mm/100mm"

    axes[0].text(
        0.01,
        1.02,
        info,
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#333333",
    )

    if missing_ranges:
        axes[0].text(
            0.99,
            1.02,
            "red = missing pose   yellow = held/stale pose   purple = large translation jump",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="#555555",
        )
    elif held_ranges or len(peak_frames) > 0:
        axes[0].text(
            0.99,
            1.02,
            "yellow = held/stale pose   purple = large translation jump",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="#555555",
        )

    suffix = "translation_board_relative_plot" if coordinate_frame == "checkerboard" else "translation_relative_plot"
    out_path = path.with_name(f"{path.stem}_{suffix}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[translation_plot] saved -> {out_path.resolve()}")
    plt.show()
    return out_path


def plot_existing_run(path: Path | None = None) -> Path | None:
    if path is None:
        path = select_jsonl_with_qt()

    if path is None:
        print("No file selected.")
        return None

    run = load_tracker_run(path)
    return plot_translation(run)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] in ("--plot", "plot"):
        path = Path(args[1]) if len(args) > 1 else None
        plot_existing_run(path)
        return

    if args and args[0] in ("--select", "select"):
        plot_existing_run(None)
        return

    if args and Path(args[0]).suffix.lower() == ".jsonl":
        plot_existing_run(Path(args[0]))
        return

    recorded_path = run_live_tracker_translation()
    if recorded_path is None:
        print("No recording to analyze.")
        return

    plot_existing_run(recorded_path)


if __name__ == "__main__":
    main()
