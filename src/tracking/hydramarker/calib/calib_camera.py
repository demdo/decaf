"""Standalone ChArUco camera calibration for HydraMarker tracking.

This script mirrors the camera calibration logic from ``overlay.calib`` and
packages the result as the NPZ file expected by the HydraMarker tracker:

    K
    dist

plus a few alias keys (``camera_matrix``, ``dist_coeffs``,
``opencv_dist_coeffs``) and image-size metadata for safety checks.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# Same board/workflow defaults as overlay.gui.pages.page_camera_calibration.
SQUARES_X = 9
SQUARES_Y = 7
SQUARE_LEN_M = 25.40e-3
MARKER_LEN_M = 17.78e-3
DICT_ID = cv2.aruco.DICT_5X5_50
MAX_ARUCO = 31
MAX_CHARUCO_CORNERS = (SQUARES_X - 1) * (SQUARES_Y - 1)

N_VIEWS = 80
MIN_CHARUCO_LIVE_FOUND = 8
MIN_CHARUCO_CAPTURE = 6
MIN_CHARUCO_EDGE_CAPTURE = 4

AUTO_CAPTURE_INTERVAL_S = 0.08
MAX_CAPTURE_CANDIDATES = 3000
CHARUCO_INTRINSIC_REFINEMENT_PASSES = 2
SELECTION_GRID_COLS = 7
SELECTION_GRID_ROWS = 5
COVERAGE_EDGE_FRACTION = 0.12
MIN_COVERAGE_CELLS = 28
MIN_EDGE_CORNERS = 36
MIN_QUADRANT_CORNERS = 72
MIN_CORNER_RADIUS_NORM = 0.75
VIEW_CENTER_GRID_COLS = 5
VIEW_CENTER_GRID_ROWS = 5
MIN_VIEW_CENTER_CELLS = 9
MIN_CENTER_VIEWS = 30
MIN_VIEW_QUADRANT_VIEWS = 8
CENTER_VIEW_HALF_WIDTH_NORM = 0.25

REALSENSE_WIDTH = 1920
REALSENSE_HEIGHT = 1080
REALSENSE_FPS = 30

WINDOW_NAME = "HydraMarker Camera Calibration"


@dataclass
class CharucoDetection:
    charuco_corners: Optional[np.ndarray]
    charuco_ids: Optional[np.ndarray]
    aruco_corners: List[np.ndarray]
    aruco_ids: Optional[np.ndarray]
    num_charuco: int
    num_aruco: int


@dataclass
class CalibrationCandidate:
    image: np.ndarray
    frame_index: int
    capture_time_s: float
    det: CharucoDetection
    score: float
    metrics: Dict[str, float]


@dataclass(frozen=True)
class CalibrationModelSpec:
    name: str
    flags: int
    dist_coeff_count: int
    description: str


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _image_size(img: np.ndarray) -> Tuple[int, int]:
    gray = _ensure_gray(img)
    h, w = gray.shape[:2]
    return (w, h)


def _make_detector_params() -> Any:
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    raise RuntimeError("No compatible ArUco DetectorParameters API found.")


def _detect_aruco_markers(
    gray: np.ndarray,
    aruco_dict: Any,
    detector_params: Optional[Any],
) -> tuple[List[np.ndarray], Optional[np.ndarray]]:
    if detector_params is None:
        detector_params = _make_detector_params()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        aruco_corners, aruco_ids, _ = detector.detectMarkers(gray)
    else:
        aruco_corners, aruco_ids, _ = cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=detector_params,
        )

    return aruco_corners, aruco_ids


_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
_SUBPIX_WIN = (5, 5)
_SUBPIX_ZERO = (-1, -1)


def _refine_charuco_subpix(
    gray: np.ndarray,
    cc: np.ndarray,
) -> np.ndarray:
    if cc is None or cc.size == 0:
        return cc
    pts = cc.reshape(-1, 1, 2).astype(np.float32)
    refined = cv2.cornerSubPix(gray, pts, _SUBPIX_WIN, _SUBPIX_ZERO, _SUBPIX_CRITERIA)
    return refined.reshape(cc.shape).astype(cc.dtype)


def _interpolate_charuco_compat(
    *,
    gray: np.ndarray,
    board: Any,
    aruco_corners: List[np.ndarray],
    aruco_ids: np.ndarray,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
) -> tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    K = None
    dist = None
    if camera_matrix is not None and dist_coeffs is not None:
        K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    if hasattr(cv2.aruco, "interpolateCornersCharuco"):
        kwargs = {
            "markerCorners": aruco_corners,
            "markerIds": aruco_ids,
            "image": gray,
            "board": board,
        }
        if K is not None and dist is not None:
            kwargs["cameraMatrix"] = K
            kwargs["distCoeffs"] = dist
        try:
            ret, cc, ci = cv2.aruco.interpolateCornersCharuco(**kwargs)
        except (cv2.error, TypeError):
            if K is None or dist is None:
                raise
            ret, cc, ci = cv2.aruco.interpolateCornersCharuco(
                markerCorners=aruco_corners,
                markerIds=aruco_ids,
                image=gray,
                board=board,
            )

        n = 0 if ret is None else int(ret)
        if n <= 0 or cc is None or ci is None:
            return 0, None, None
        cc = _refine_charuco_subpix(gray, cc)
        return n, cc, ci

    if hasattr(cv2.aruco, "CharucoDetector"):
        if hasattr(cv2.aruco, "CharucoParameters"):
            charuco_params = cv2.aruco.CharucoParameters()
            charuco_params.cameraMatrix = K
            charuco_params.distCoeffs = dist
            detector = cv2.aruco.CharucoDetector(
                board,
                charucoParams=charuco_params,
            )
        else:
            detector = cv2.aruco.CharucoDetector(board)

        cc, ci, _, _ = detector.detectBoard(
            gray,
            markerCorners=aruco_corners,
            markerIds=aruco_ids,
        )

        if cc is None or ci is None or len(ci) == 0:
            return 0, None, None

        cc = _refine_charuco_subpix(gray, cc)
        return int(len(ci)), cc, ci

    raise RuntimeError("No compatible ChArUco interpolation API available in cv2.aruco.")


def detect_charuco(
    image: np.ndarray,
    board: Any,
    aruco_dict: Any,
    detector_params: Optional[Any] = None,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
) -> CharucoDetection:
    gray = _ensure_gray(image)

    aruco_corners, aruco_ids = _detect_aruco_markers(
        gray=gray,
        aruco_dict=aruco_dict,
        detector_params=detector_params,
    )

    charuco_corners = None
    charuco_ids = None

    if aruco_ids is not None and len(aruco_ids) > 0:
        ret, cc, ci = _interpolate_charuco_compat(
            gray=gray,
            board=board,
            aruco_corners=aruco_corners,
            aruco_ids=aruco_ids,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        if ret > 0:
            charuco_corners, charuco_ids = cc, ci

    return CharucoDetection(
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
        aruco_corners=aruco_corners,
        aruco_ids=aruco_ids,
        num_charuco=0 if charuco_ids is None else int(len(charuco_ids)),
        num_aruco=0 if aruco_ids is None else int(len(aruco_ids)),
    )


def _normalized_laplacian_var(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    scale = min(1.0, 640.0 / float(max(h, w)))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return lap_var / max(scale * scale, 1e-6)


def _charuco_id_set(det: CharucoDetection) -> set[int]:
    if det.charuco_ids is None:
        return set()
    return {int(x) for x in det.charuco_ids.reshape(-1)}


def _candidate_metrics(image: np.ndarray, det: CharucoDetection) -> Dict[str, float]:
    gray = _ensure_gray(image)
    h, w = gray.shape[:2]
    metrics: Dict[str, float] = {
        "num_charuco": float(det.num_charuco),
        "num_aruco": float(det.num_aruco),
        "corner_fraction": float(det.num_charuco) / float(MAX_CHARUCO_CORNERS),
        "sharpness": _normalized_laplacian_var(gray),
        "centroid_u": float("nan"),
        "centroid_v": float("nan"),
        "centroid_u_norm": float("nan"),
        "centroid_v_norm": float("nan"),
        "bbox_area_norm": 0.0,
        "bbox_diag_norm": 0.0,
        "edge_margin_px": 0.0,
        "edge_margin_norm": 0.0,
        "radius_norm": float("nan"),
        "corner_radius_norm_max": float("nan"),
    }

    if det.charuco_corners is None or det.num_charuco <= 0:
        return metrics

    pts = det.charuco_corners.reshape(-1, 2).astype(np.float64)
    min_uv = np.min(pts, axis=0)
    max_uv = np.max(pts, axis=0)
    centroid = np.mean(pts, axis=0)

    bbox_wh = np.maximum(max_uv - min_uv, 0.0)
    edge_margin = float(
        min(
            np.min(pts[:, 0]),
            np.min(pts[:, 1]),
            float(w - 1) - np.max(pts[:, 0]),
            float(h - 1) - np.max(pts[:, 1]),
        )
    )
    u_norm = float(centroid[0] / max(w - 1, 1))
    v_norm = float(centroid[1] / max(h - 1, 1))
    radius_norm = float(np.hypot(u_norm - 0.5, v_norm - 0.5) / np.hypot(0.5, 0.5))
    pts_norm = np.c_[
        pts[:, 0] / max(w - 1, 1),
        pts[:, 1] / max(h - 1, 1),
    ]
    corner_radius_norm_max = float(
        np.max(np.hypot(pts_norm[:, 0] - 0.5, pts_norm[:, 1] - 0.5))
        / np.hypot(0.5, 0.5)
    )

    metrics.update(
        {
            "centroid_u": float(centroid[0]),
            "centroid_v": float(centroid[1]),
            "centroid_u_norm": u_norm,
            "centroid_v_norm": v_norm,
            "bbox_area_norm": float((bbox_wh[0] * bbox_wh[1]) / max(w * h, 1)),
            "bbox_diag_norm": float(np.linalg.norm(bbox_wh) / np.linalg.norm([w, h])),
            "edge_margin_px": edge_margin,
            "edge_margin_norm": float(edge_margin / max(min(w, h), 1)),
            "radius_norm": radius_norm,
            "corner_radius_norm_max": corner_radius_norm_max,
        }
    )

    return metrics


def _candidate_score(metrics: Dict[str, float]) -> float:
    corner_score = float(np.clip(metrics.get("corner_fraction", 0.0), 0.0, 1.0))
    area_score = float(np.clip(metrics.get("bbox_area_norm", 0.0) / 0.14, 0.0, 1.0))
    sharpness_score = float(
        np.clip(np.log1p(max(metrics.get("sharpness", 0.0), 0.0)) / np.log1p(2500.0), 0.0, 1.0)
    )
    radius_score = float(np.clip(metrics.get("corner_radius_norm_max", 0.0), 0.0, 1.0))
    edge_margin_norm = float(max(metrics.get("edge_margin_norm", 1.0), 0.0))
    edge_score = 1.0 - float(
        np.clip(edge_margin_norm / max(COVERAGE_EDGE_FRACTION, 1e-6), 0.0, 1.0)
    )

    return (
        0.35 * corner_score
        + 0.18 * area_score
        + 0.22 * sharpness_score
        + 0.15 * radius_score
        + 0.10 * edge_score
    )


def _is_fov_edge_candidate(
    metrics: Dict[str, float],
    image_size: tuple[int, int],
    *,
    edge_fraction: float = COVERAGE_EDGE_FRACTION,
) -> bool:
    width, height = int(image_size[0]), int(image_size[1])
    u = float(np.nan_to_num(metrics.get("centroid_u_norm", 0.5), nan=0.5))
    v = float(np.nan_to_num(metrics.get("centroid_v_norm", 0.5), nan=0.5))
    edge_margin_px = float(np.nan_to_num(metrics.get("edge_margin_px", np.inf), nan=np.inf))
    edge_band_px = float(edge_fraction) * float(min(width, height))
    centroid_in_edge_band = (
        u <= edge_fraction
        or u >= 1.0 - edge_fraction
        or v <= edge_fraction
        or v >= 1.0 - edge_fraction
    )
    return bool(centroid_in_edge_band or edge_margin_px <= edge_band_px)


def make_calibration_candidate(
    image: np.ndarray,
    *,
    frame_index: int,
    capture_time_s: float,
    det: CharucoDetection,
) -> CalibrationCandidate:
    metrics = _candidate_metrics(image, det)
    return CalibrationCandidate(
        image=image.copy(),
        frame_index=int(frame_index),
        capture_time_s=float(capture_time_s),
        det=det,
        score=_candidate_score(metrics),
        metrics=metrics,
    )


def _candidate_grid_cell(metrics: Dict[str, float]) -> tuple[int, int]:
    u = float(np.nan_to_num(metrics.get("centroid_u_norm", 0.5), nan=0.5))
    v = float(np.nan_to_num(metrics.get("centroid_v_norm", 0.5), nan=0.5))
    col = int(np.clip(np.floor(u * SELECTION_GRID_COLS), 0, SELECTION_GRID_COLS - 1))
    row = int(np.clip(np.floor(v * SELECTION_GRID_ROWS), 0, SELECTION_GRID_ROWS - 1))
    return col, row


def _candidate_view_grid_cell(
    metrics: Dict[str, float],
    *,
    grid_cols: int,
    grid_rows: int,
) -> tuple[int, int]:
    u = float(np.nan_to_num(metrics.get("centroid_u_norm", 0.5), nan=0.5))
    v = float(np.nan_to_num(metrics.get("centroid_v_norm", 0.5), nan=0.5))
    col = int(np.clip(np.floor(u * grid_cols), 0, grid_cols - 1))
    row = int(np.clip(np.floor(v * grid_rows), 0, grid_rows - 1))
    return col, row


def _candidate_center_offsets(metrics: Dict[str, float]) -> tuple[float, float]:
    u = float(np.nan_to_num(metrics.get("centroid_u_norm", 0.5), nan=0.5))
    v = float(np.nan_to_num(metrics.get("centroid_v_norm", 0.5), nan=0.5))
    return 2.0 * (u - 0.5), 2.0 * (v - 0.5)


def _is_center_view(
    metrics: Dict[str, float],
    *,
    center_half_width_norm: float = CENTER_VIEW_HALF_WIDTH_NORM,
) -> bool:
    dx, dy = _candidate_center_offsets(metrics)
    return bool(abs(dx) <= center_half_width_norm and abs(dy) <= center_half_width_norm)


def _candidate_radius_bin(metrics: Dict[str, float]) -> int:
    radius = float(np.nan_to_num(metrics.get("corner_radius_norm_max", 0.0), nan=0.0))
    return int(np.clip(np.floor(radius * 4.0), 0, 3))


def _candidate_size_bin(metrics: Dict[str, float]) -> int:
    area = float(metrics.get("bbox_area_norm", 0.0))
    return int(np.searchsorted([0.025, 0.06, 0.12, 0.22], area, side="right"))


def _candidate_feature(metrics: Dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            float(np.nan_to_num(metrics.get("centroid_u_norm", 0.5), nan=0.5)),
            float(np.nan_to_num(metrics.get("centroid_v_norm", 0.5), nan=0.5)),
            float(np.clip(metrics.get("bbox_area_norm", 0.0) / 0.25, 0.0, 1.0)),
            float(np.clip(metrics.get("corner_fraction", 0.0), 0.0, 1.0)),
            float(np.clip(metrics.get("corner_radius_norm_max", 0.0), 0.0, 1.0)),
        ],
        dtype=np.float64,
    )


def _candidate_corner_points(candidate: CalibrationCandidate) -> np.ndarray:
    if candidate.det.charuco_corners is None:
        return np.empty((0, 2), dtype=np.float64)
    return candidate.det.charuco_corners.reshape(-1, 2).astype(np.float64)


def _corner_grid_cells(
    points: np.ndarray,
    image_size: tuple[int, int],
    *,
    grid_cols: int,
    grid_rows: int,
) -> set[tuple[int, int]]:
    if points.size == 0:
        return set()

    width, height = image_size
    cols = np.clip(np.floor(points[:, 0] / max(width, 1) * grid_cols), 0, grid_cols - 1)
    rows = np.clip(np.floor(points[:, 1] / max(height, 1) * grid_rows), 0, grid_rows - 1)
    return {(int(c), int(r)) for c, r in zip(cols, rows)}


def compute_corner_coverage(
    candidates: Sequence[CalibrationCandidate],
    image_size: tuple[int, int],
    *,
    grid_cols: int = SELECTION_GRID_COLS,
    grid_rows: int = SELECTION_GRID_ROWS,
    edge_fraction: float = COVERAGE_EDGE_FRACTION,
) -> Dict[str, Any]:
    width, height = int(image_size[0]), int(image_size[1])
    grid_counts = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    edge_counts = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    quadrant_counts = np.zeros((2, 2), dtype=np.int32)
    all_points: list[np.ndarray] = []

    for cand in candidates:
        pts = _candidate_corner_points(cand)
        if pts.size == 0:
            continue
        all_points.append(pts)

        cols = np.clip(np.floor(pts[:, 0] / max(width, 1) * grid_cols), 0, grid_cols - 1).astype(int)
        rows = np.clip(np.floor(pts[:, 1] / max(height, 1) * grid_rows), 0, grid_rows - 1).astype(int)
        for col, row in zip(cols, rows):
            grid_counts[row, col] += 1

        edge_counts["left"] += int(np.count_nonzero(pts[:, 0] <= edge_fraction * width))
        edge_counts["right"] += int(np.count_nonzero(pts[:, 0] >= (1.0 - edge_fraction) * width))
        edge_counts["top"] += int(np.count_nonzero(pts[:, 1] <= edge_fraction * height))
        edge_counts["bottom"] += int(np.count_nonzero(pts[:, 1] >= (1.0 - edge_fraction) * height))

        q_cols = (pts[:, 0] >= 0.5 * width).astype(int)
        q_rows = (pts[:, 1] >= 0.5 * height).astype(int)
        for q_col, q_row in zip(q_cols, q_rows):
            quadrant_counts[q_row, q_col] += 1

    if all_points:
        stacked = np.vstack(all_points)
        min_u, min_v = np.min(stacked, axis=0)
        max_u, max_v = np.max(stacked, axis=0)
        total_corners = int(stacked.shape[0])
        stacked_norm = np.c_[
            stacked[:, 0] / max(width - 1, 1),
            stacked[:, 1] / max(height - 1, 1),
        ]
        radius_norm = np.hypot(stacked_norm[:, 0] - 0.5, stacked_norm[:, 1] - 0.5) / np.hypot(0.5, 0.5)
        max_corner_radius_norm = float(np.max(radius_norm))
        p95_corner_radius_norm = float(np.percentile(radius_norm, 95))
    else:
        min_u = min_v = max_u = max_v = float("nan")
        total_corners = 0
        max_corner_radius_norm = float("nan")
        p95_corner_radius_norm = float("nan")

    covered_cells = int(np.count_nonzero(grid_counts > 0))
    total_cells = int(grid_cols * grid_rows)

    return {
        "grid_counts": grid_counts,
        "edge_counts": edge_counts,
        "quadrant_counts": quadrant_counts,
        "covered_cells": covered_cells,
        "total_cells": total_cells,
        "coverage_fraction": float(covered_cells / max(total_cells, 1)),
        "total_corners": total_corners,
        "min_u": float(min_u),
        "max_u": float(max_u),
        "min_v": float(min_v),
        "max_v": float(max_v),
        "max_corner_radius_norm": max_corner_radius_norm,
        "p95_corner_radius_norm": p95_corner_radius_norm,
        "grid_cols": int(grid_cols),
        "grid_rows": int(grid_rows),
        "edge_fraction": float(edge_fraction),
    }


def compute_view_center_coverage(
    candidates: Sequence[CalibrationCandidate],
    image_size: tuple[int, int],
    *,
    grid_cols: int = VIEW_CENTER_GRID_COLS,
    grid_rows: int = VIEW_CENTER_GRID_ROWS,
    center_half_width_norm: float = CENTER_VIEW_HALF_WIDTH_NORM,
    edge_fraction: float = COVERAGE_EDGE_FRACTION,
) -> Dict[str, Any]:
    width, height = int(image_size[0]), int(image_size[1])
    grid_counts = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    edge_counts = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    quadrant_counts = np.zeros((2, 2), dtype=np.int32)
    centers: list[tuple[float, float]] = []
    center_count = 0

    for cand in candidates:
        metrics = cand.metrics
        u = float(np.nan_to_num(metrics.get("centroid_u_norm", np.nan), nan=np.nan))
        v = float(np.nan_to_num(metrics.get("centroid_v_norm", np.nan), nan=np.nan))
        if not (np.isfinite(u) and np.isfinite(v)):
            continue

        u = float(np.clip(u, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))
        centers.append((u, v))

        col = int(np.clip(np.floor(u * grid_cols), 0, grid_cols - 1))
        row = int(np.clip(np.floor(v * grid_rows), 0, grid_rows - 1))
        grid_counts[row, col] += 1

        if u <= edge_fraction:
            edge_counts["left"] += 1
        if u >= 1.0 - edge_fraction:
            edge_counts["right"] += 1
        if v <= edge_fraction:
            edge_counts["top"] += 1
        if v >= 1.0 - edge_fraction:
            edge_counts["bottom"] += 1

        q_col = int(u >= 0.5)
        q_row = int(v >= 0.5)
        quadrant_counts[q_row, q_col] += 1

        dx = 2.0 * (u - 0.5)
        dy = 2.0 * (v - 0.5)
        if abs(dx) <= center_half_width_norm and abs(dy) <= center_half_width_norm:
            center_count += 1

    if centers:
        centers_arr = np.asarray(centers, dtype=np.float64)
        dx_norm = 2.0 * (centers_arr[:, 0] - 0.5)
        dy_norm = 2.0 * (centers_arr[:, 1] - 0.5)
        radius_norm = np.hypot(dx_norm, dy_norm) / np.sqrt(2.0)
        dx_p05, dx_p50, dx_p95 = np.percentile(dx_norm, [5, 50, 95])
        dy_p05, dy_p50, dy_p95 = np.percentile(dy_norm, [5, 50, 95])
        radius_p05, radius_p50, radius_p95 = np.percentile(radius_norm, [5, 50, 95])
        radius_max = float(np.max(radius_norm))
    else:
        centers_arr = np.empty((0, 2), dtype=np.float64)
        dx_p05 = dx_p50 = dx_p95 = float("nan")
        dy_p05 = dy_p50 = dy_p95 = float("nan")
        radius_p05 = radius_p50 = radius_p95 = radius_max = float("nan")

    covered_cells = int(np.count_nonzero(grid_counts > 0))
    total_cells = int(grid_cols * grid_rows)

    return {
        "grid_counts": grid_counts,
        "edge_counts": edge_counts,
        "quadrant_counts": quadrant_counts,
        "covered_cells": covered_cells,
        "total_cells": total_cells,
        "coverage_fraction": float(covered_cells / max(total_cells, 1)),
        "total_views": int(centers_arr.shape[0]),
        "center_count": int(center_count),
        "center_half_width_norm": float(center_half_width_norm),
        "centroids_uv_norm": centers_arr,
        "dx_norm_p05_p50_p95": [float(dx_p05), float(dx_p50), float(dx_p95)],
        "dy_norm_p05_p50_p95": [float(dy_p05), float(dy_p50), float(dy_p95)],
        "radius_norm_p05_p50_p95_max": [
            float(radius_p05),
            float(radius_p50),
            float(radius_p95),
            float(radius_max),
        ],
        "grid_cols": int(grid_cols),
        "grid_rows": int(grid_rows),
        "edge_fraction": float(edge_fraction),
        "image_width": width,
        "image_height": height,
    }


def coverage_failures(
    coverage: Dict[str, Any],
    *,
    min_coverage_cells: int = MIN_COVERAGE_CELLS,
    min_edge_corners: int = MIN_EDGE_CORNERS,
    min_quadrant_corners: int = MIN_QUADRANT_CORNERS,
    min_corner_radius_norm: float = MIN_CORNER_RADIUS_NORM,
) -> list[str]:
    failures: list[str] = []
    if int(coverage.get("covered_cells", 0)) < int(min_coverage_cells):
        failures.append(
            f"grid {coverage.get('covered_cells', 0)}/{coverage.get('total_cells', 0)} "
            f"(need {min_coverage_cells})"
        )

    edge_counts = coverage.get("edge_counts", {})
    for edge in ("left", "right", "top", "bottom"):
        count = int(edge_counts.get(edge, 0))
        if count < int(min_edge_corners):
            failures.append(f"{edge} edge {count}/{min_edge_corners}")

    quadrant_counts = np.asarray(coverage.get("quadrant_counts", np.zeros((2, 2))), dtype=int)
    for row, col, name in ((0, 0, "top-left"), (0, 1, "top-right"), (1, 0, "bottom-left"), (1, 1, "bottom-right")):
        count = int(quadrant_counts[row, col]) if quadrant_counts.shape == (2, 2) else 0
        if count < int(min_quadrant_corners):
            failures.append(f"{name} quadrant {count}/{min_quadrant_corners}")

    max_radius = float(coverage.get("max_corner_radius_norm", float("nan")))
    if not np.isfinite(max_radius) or max_radius < float(min_corner_radius_norm):
        failures.append(f"corner radius {max_radius:.2f}/{float(min_corner_radius_norm):.2f}")

    return failures


def view_center_coverage_failures(
    coverage: Dict[str, Any],
    *,
    min_view_center_cells: int = MIN_VIEW_CENTER_CELLS,
    min_center_views: int = MIN_CENTER_VIEWS,
    min_view_quadrant_views: int = MIN_VIEW_QUADRANT_VIEWS,
) -> list[str]:
    failures: list[str] = []
    if not coverage:
        return ["view-center coverage unavailable"]

    if int(coverage.get("covered_cells", 0)) < int(min_view_center_cells):
        failures.append(
            f"view grid {coverage.get('covered_cells', 0)}/{coverage.get('total_cells', 0)} "
            f"(need {min_view_center_cells})"
        )

    if int(coverage.get("center_count", 0)) < int(min_center_views):
        failures.append(
            f"center views {coverage.get('center_count', 0)} "
            f"(need {min_center_views})"
        )

    quadrant_counts = np.asarray(coverage.get("quadrant_counts", np.zeros((2, 2))), dtype=int)
    if quadrant_counts.size:
        min_quadrant = int(np.min(quadrant_counts))
        if min_quadrant < int(min_view_quadrant_views):
            failures.append(
                f"view quadrants min {min_quadrant} "
                f"(need {min_view_quadrant_views})"
            )

    return failures


def _coverage_short_text(coverage: Optional[Dict[str, Any]]) -> str:
    if not coverage:
        return "Coverage: 0/0"
    edges = coverage.get("edge_counts", {})
    return (
        f"Coverage: {coverage.get('covered_cells', 0)}/{coverage.get('total_cells', 0)} cells "
        f"L/R/T/B={edges.get('left', 0)}/{edges.get('right', 0)}/"
        f"{edges.get('top', 0)}/{edges.get('bottom', 0)} "
        f"Rmax={float(coverage.get('max_corner_radius_norm', float('nan'))):.2f}"
    )


def _view_coverage_short_text(coverage: Optional[Dict[str, Any]]) -> str:
    if not coverage:
        return "View centers: 0/0"
    radius_stats = coverage.get("radius_norm_p05_p50_p95_max") or [np.nan, np.nan, np.nan, np.nan]
    return (
        f"View centers: {coverage.get('covered_cells', 0)}/{coverage.get('total_cells', 0)} cells "
        f"center={coverage.get('center_count', 0)} "
        f"R50={float(radius_stats[1]):.2f}"
    )


def select_calibration_candidates(
    candidates: Sequence[CalibrationCandidate],
    *,
    target_views: int,
    min_charuco_corners: int,
    min_edge_charuco_corners: int = MIN_CHARUCO_EDGE_CAPTURE,
    image_size: tuple[int, int],
    grid_cols: int = SELECTION_GRID_COLS,
    grid_rows: int = SELECTION_GRID_ROWS,
    edge_fraction: float = COVERAGE_EDGE_FRACTION,
    min_edge_corners: int = MIN_EDGE_CORNERS,
    min_quadrant_corners: int = MIN_QUADRANT_CORNERS,
    view_grid_cols: int = VIEW_CENTER_GRID_COLS,
    view_grid_rows: int = VIEW_CENTER_GRID_ROWS,
    min_view_center_cells: int = MIN_VIEW_CENTER_CELLS,
    min_center_views: int = MIN_CENTER_VIEWS,
    min_view_quadrant_views: int = MIN_VIEW_QUADRANT_VIEWS,
    center_view_half_width_norm: float = CENTER_VIEW_HALF_WIDTH_NORM,
) -> list[CalibrationCandidate]:
    valid = [
        c
        for c in candidates
        if c.det.charuco_corners is not None
        and c.det.charuco_ids is not None
        and (
            c.det.num_charuco >= min_charuco_corners
            or (
                c.det.num_charuco >= min_edge_charuco_corners
                and _is_fov_edge_candidate(
                    c.metrics,
                    image_size,
                    edge_fraction=edge_fraction,
                )
            )
        )
    ]

    if len(valid) <= target_views:
        return sorted(valid, key=lambda c: c.frame_index)

    width, height = int(image_size[0]), int(image_size[1])
    selected: list[CalibrationCandidate] = []
    selected_features: list[np.ndarray] = []
    selected_cells: set[tuple[int, int]] = set()
    selected_view_cells: set[tuple[int, int]] = set()
    selected_center_views = 0
    selected_radius_bins: set[int] = set()
    selected_size_bins: set[int] = set()
    selected_edge_counts = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    selected_quadrant_counts = np.zeros((2, 2), dtype=np.int32)
    selected_view_quadrant_counts = np.zeros((2, 2), dtype=np.int32)
    covered_ids: set[int] = set()
    remaining = list(valid)

    while remaining and len(selected) < target_views:
        best_i = 0
        best_score = -float("inf")

        for i, cand in enumerate(remaining):
            metrics = cand.metrics
            feature = _candidate_feature(metrics)
            radius_bin = _candidate_radius_bin(metrics)
            size_bin = _candidate_size_bin(metrics)
            ids = _charuco_id_set(cand.det)
            pts = _candidate_corner_points(cand)
            view_cell = _candidate_view_grid_cell(
                metrics,
                grid_cols=view_grid_cols,
                grid_rows=view_grid_rows,
            )
            is_center_view = _is_center_view(
                metrics,
                center_half_width_norm=center_view_half_width_norm,
            )
            u = float(np.nan_to_num(metrics.get("centroid_u_norm", 0.5), nan=0.5))
            v = float(np.nan_to_num(metrics.get("centroid_v_norm", 0.5), nan=0.5))
            view_q_col = int(u >= 0.5)
            view_q_row = int(v >= 0.5)
            candidate_cells = _corner_grid_cells(
                pts,
                image_size,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
            )

            diversity_bonus = 0.0

            new_cells = candidate_cells - selected_cells
            diversity_bonus += 0.45 * min(1.0, len(new_cells) / 3.0)

            if len(selected_view_cells) < min_view_center_cells and view_cell not in selected_view_cells:
                diversity_bonus += 0.95

            if selected_center_views < min_center_views:
                if is_center_view:
                    diversity_bonus += 1.10
                else:
                    dx, dy = _candidate_center_offsets(metrics)
                    center_distance = float(np.hypot(dx, dy) / np.sqrt(2.0))
                    diversity_bonus += 0.35 * (1.0 - float(np.clip(center_distance, 0.0, 1.0)))

            view_quadrant_need = max(
                0,
                int(min_view_quadrant_views)
                - int(selected_view_quadrant_counts[view_q_row, view_q_col]),
            )
            if view_quadrant_need > 0:
                diversity_bonus += 0.45

            if pts.size:
                edge_candidate_counts = {
                    "left": int(np.count_nonzero(pts[:, 0] <= edge_fraction * width)),
                    "right": int(np.count_nonzero(pts[:, 0] >= (1.0 - edge_fraction) * width)),
                    "top": int(np.count_nonzero(pts[:, 1] <= edge_fraction * height)),
                    "bottom": int(np.count_nonzero(pts[:, 1] >= (1.0 - edge_fraction) * height)),
                }
                for edge, count in edge_candidate_counts.items():
                    need = max(0, int(min_edge_corners) - selected_edge_counts[edge])
                    if need > 0:
                        diversity_bonus += 0.35 * min(1.0, count / max(need, 1))

                q_cols = (pts[:, 0] >= 0.5 * width).astype(int)
                q_rows = (pts[:, 1] >= 0.5 * height).astype(int)
                candidate_quadrants = np.zeros((2, 2), dtype=np.int32)
                for q_col, q_row in zip(q_cols, q_rows):
                    candidate_quadrants[q_row, q_col] += 1
                for row in range(2):
                    for col in range(2):
                        need = max(
                            0,
                            int(min_quadrant_corners)
                            - int(selected_quadrant_counts[row, col]),
                        )
                        if need > 0:
                            diversity_bonus += 0.20 * min(1.0, int(candidate_quadrants[row, col]) / max(need, 1))

            if radius_bin not in selected_radius_bins:
                diversity_bonus += 0.18
            if size_bin not in selected_size_bins:
                diversity_bonus += 0.18

            new_ids = ids - covered_ids
            diversity_bonus += 0.20 * min(1.0, len(new_ids) / 10.0)

            if selected_features:
                min_dist = min(float(np.linalg.norm(feature - prev)) for prev in selected_features)
                diversity_bonus += 0.18 * float(np.clip(min_dist / 0.35, 0.0, 1.0))
            else:
                diversity_bonus += 0.30

            score = float(cand.score) + diversity_bonus
            if score > best_score:
                best_score = score
                best_i = i

        chosen = remaining.pop(best_i)
        selected.append(chosen)
        selected_features.append(_candidate_feature(chosen.metrics))
        selected_view_cells.add(
            _candidate_view_grid_cell(
                chosen.metrics,
                grid_cols=view_grid_cols,
                grid_rows=view_grid_rows,
            )
        )
        if _is_center_view(
            chosen.metrics,
            center_half_width_norm=center_view_half_width_norm,
        ):
            selected_center_views += 1
        chosen_u = float(np.nan_to_num(chosen.metrics.get("centroid_u_norm", 0.5), nan=0.5))
        chosen_v = float(np.nan_to_num(chosen.metrics.get("centroid_v_norm", 0.5), nan=0.5))
        selected_view_quadrant_counts[int(chosen_v >= 0.5), int(chosen_u >= 0.5)] += 1
        pts = _candidate_corner_points(chosen)
        selected_cells.update(
            _corner_grid_cells(
                pts,
                image_size,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
            )
        )
        if pts.size:
            selected_edge_counts["left"] += int(np.count_nonzero(pts[:, 0] <= edge_fraction * width))
            selected_edge_counts["right"] += int(np.count_nonzero(pts[:, 0] >= (1.0 - edge_fraction) * width))
            selected_edge_counts["top"] += int(np.count_nonzero(pts[:, 1] <= edge_fraction * height))
            selected_edge_counts["bottom"] += int(np.count_nonzero(pts[:, 1] >= (1.0 - edge_fraction) * height))
            q_cols = (pts[:, 0] >= 0.5 * width).astype(int)
            q_rows = (pts[:, 1] >= 0.5 * height).astype(int)
            for q_col, q_row in zip(q_cols, q_rows):
                selected_quadrant_counts[q_row, q_col] += 1
        selected_radius_bins.add(_candidate_radius_bin(chosen.metrics))
        selected_size_bins.add(_candidate_size_bin(chosen.metrics))
        covered_ids.update(_charuco_id_set(chosen.det))

    return sorted(selected, key=lambda c: c.frame_index)


def _charuco_object_points(board: Any, charuco_ids: np.ndarray) -> np.ndarray:
    all_obj = board.getChessboardCorners()
    ids = charuco_ids.reshape(-1).astype(int)
    return all_obj[ids, :].astype(np.float32)


def _calibrate_charuco_compat(
    *,
    all_charuco_corners: Sequence[np.ndarray],
    all_charuco_ids: Sequence[np.ndarray],
    board: Any,
    image_size: Tuple[int, int],
    K_init: np.ndarray,
    dist_init: np.ndarray,
    flags: int,
    criteria: Tuple[int, int, float],
) -> tuple[float, np.ndarray, np.ndarray]:
    if hasattr(cv2.aruco, "calibrateCameraCharuco"):
        rms, K, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=K_init,
            distCoeffs=dist_init,
            flags=flags,
            criteria=criteria,
        )
        return float(rms), np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []

    for corners, ids in zip(all_charuco_corners, all_charuco_ids):
        if corners is None or ids is None:
            continue

        obj = _charuco_object_points(board, ids).astype(np.float32)
        img = np.asarray(corners, dtype=np.float32).reshape(-1, 2)

        if obj.shape[0] != img.shape[0] or obj.shape[0] < 4:
            continue

        object_points.append(obj)
        image_points.append(img)

    if len(object_points) < 3:
        raise RuntimeError("Not enough valid ChArUco views for fallback calibration.")

    rms, K, dist, _, _ = cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=image_points,
        imageSize=image_size,
        cameraMatrix=K_init,
        distCoeffs=dist_init,
        flags=flags,
        criteria=criteria,
    )

    return float(rms), np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64)


def calibrate_charuco_intrinsics(
    calib_images: Sequence[np.ndarray],
    board: Any,
    aruco_dict: Any,
    detector_params: Optional[Any] = None,
    min_charuco_corners: int = MIN_CHARUCO_CAPTURE,
    flags: int = 0,
    dist_coeff_count: int = 5,
    criteria: Optional[Tuple[int, int, float]] = None,
    intrinsic_refinement_passes: int = CHARUCO_INTRINSIC_REFINEMENT_PASSES,
    K_seed: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    if len(calib_images) == 0:
        raise ValueError("No calibration images provided.")

    if criteria is None:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-6,
        )

    image_size = _image_size(calib_images[0])

    K: Optional[np.ndarray] = None
    dist: Optional[np.ndarray] = None
    rms = float("nan")
    all_charuco_corners: list[np.ndarray] = []
    all_charuco_ids: list[np.ndarray] = []
    used_idx: list[int] = []
    per_img_charuco: list[int] = []
    per_img_aruco: list[int] = []
    pass_stats: list[dict[str, Any]] = []
    requested_refinement_passes = max(0, int(intrinsic_refinement_passes))
    total_passes = 1 + requested_refinement_passes

    for pass_index in range(total_passes):
        use_intrinsics = pass_index > 0 and K is not None and dist is not None
        pass_corners: list[np.ndarray] = []
        pass_ids: list[np.ndarray] = []
        pass_used_idx: list[int] = []
        pass_per_img_charuco: list[int] = []
        pass_per_img_aruco: list[int] = []

        for i, img in enumerate(calib_images):
            if _image_size(img) != image_size:
                raise ValueError("All images must have same resolution.")

            det = detect_charuco(
                img,
                board,
                aruco_dict,
                detector_params,
                camera_matrix=K if use_intrinsics else None,
                dist_coeffs=dist if use_intrinsics else None,
            )
            pass_per_img_charuco.append(det.num_charuco)
            pass_per_img_aruco.append(det.num_aruco)

            if det.charuco_ids is None or det.charuco_corners is None:
                continue
            if det.num_charuco < min_charuco_corners:
                continue

            pass_corners.append(det.charuco_corners)
            pass_ids.append(det.charuco_ids)
            pass_used_idx.append(i)

        if len(pass_corners) < 3:
            raise RuntimeError(
                "Not enough valid views for calibration "
                f"in ChArUco refinement pass {pass_index}."
            )

        if use_intrinsics:
            K_init = np.asarray(K, dtype=np.float64).reshape(3, 3)
            dist_init = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
            effective_flags = int(flags) | int(getattr(cv2, "CALIB_USE_INTRINSIC_GUESS", 0))
        elif K_seed is not None:
            K_init = np.asarray(K_seed, dtype=np.float64).reshape(3, 3)
            dist_init = np.zeros((max(4, int(dist_coeff_count)), 1), dtype=np.float64)
            effective_flags = int(flags) | int(getattr(cv2, "CALIB_USE_INTRINSIC_GUESS", 0))
        else:
            K_init = np.eye(3, dtype=np.float64)
            dist_init = np.zeros((max(4, int(dist_coeff_count)), 1), dtype=np.float64)
            effective_flags = int(flags)

        rms, K, dist = _calibrate_charuco_compat(
            all_charuco_corners=pass_corners,
            all_charuco_ids=pass_ids,
            board=board,
            image_size=image_size,
            K_init=K_init,
            dist_init=dist_init,
            flags=effective_flags,
            criteria=criteria,
        )

        all_charuco_corners = pass_corners
        all_charuco_ids = pass_ids
        used_idx = pass_used_idx
        per_img_charuco = pass_per_img_charuco
        per_img_aruco = pass_per_img_aruco
        pass_stats.append(
            {
                "pass_index": int(pass_index),
                "used_intrinsics_for_charuco": bool(use_intrinsics),
                "calibration_flags": int(effective_flags),
                "num_images_used": int(len(pass_corners)),
                "mean_charuco_corners": float(np.mean(pass_per_img_charuco))
                if pass_per_img_charuco
                else float("nan"),
                "min_charuco_corners": int(min(pass_per_img_charuco))
                if pass_per_img_charuco
                else 0,
                "max_charuco_corners": int(max(pass_per_img_charuco))
                if pass_per_img_charuco
                else 0,
                "rms": float(rms),
                "fx": float(np.asarray(K, dtype=np.float64).reshape(3, 3)[0, 0]),
                "fy": float(np.asarray(K, dtype=np.float64).reshape(3, 3)[1, 1]),
                "cx": float(np.asarray(K, dtype=np.float64).reshape(3, 3)[0, 2]),
                "cy": float(np.asarray(K, dtype=np.float64).reshape(3, 3)[1, 2]),
                "dist_coeff_count": int(np.asarray(dist).reshape(-1).size),
            }
        )

    if K is None or dist is None:
        raise RuntimeError("ChArUco calibration did not produce intrinsics.")

    # Kannala & Brandt (2006): if polynomial is non-monotonic (radial_turns > 1),
    # retry with CALIB_FIX_K3 using the current K as seed.  Accept the monotonic
    # result when its RMS stays within 15 % of the original (Brown 1966: physical
    # lenses have monotone radial distortion).
    monotonic_fallback_applied = False
    fix_k3_flag = int(getattr(cv2, "CALIB_FIX_K3", 0))
    if fix_k3_flag and not (int(flags) & fix_k3_flag):
        r_vals = np.linspace(0.0, 1.0, 200)
        scale = (
            1.0
            + float(np.asarray(dist).reshape(-1)[0]) * r_vals ** 2
            + float(np.asarray(dist).reshape(-1)[1]) * r_vals ** 4
            + float(np.asarray(dist).reshape(-1)[4] if np.asarray(dist).reshape(-1).size > 4 else 0.0)
            * r_vals ** 6
        )
        diff = np.diff(scale)
        eps = 1e-7
        signs = np.sign(diff[np.abs(diff) > eps])
        turn_count = int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size >= 2 else 0
        if turn_count > 1:
            try:
                retry_flags = (
                    int(flags) | fix_k3_flag | int(getattr(cv2, "CALIB_USE_INTRINSIC_GUESS", 0))
                )
                rms_mono, K_mono, dist_mono = _calibrate_charuco_compat(
                    all_charuco_corners=all_charuco_corners,
                    all_charuco_ids=all_charuco_ids,
                    board=board,
                    image_size=image_size,
                    K_init=np.asarray(K, dtype=np.float64).reshape(3, 3),
                    dist_init=np.zeros((max(4, int(dist_coeff_count)), 1), dtype=np.float64),
                    flags=retry_flags,
                    criteria=criteria,
                )
                scale_m = (
                    1.0
                    + float(np.asarray(dist_mono).reshape(-1)[0]) * r_vals ** 2
                    + float(np.asarray(dist_mono).reshape(-1)[1]) * r_vals ** 4
                )
                diff_m = np.diff(scale_m)
                signs_m = np.sign(diff_m[np.abs(diff_m) > eps])
                turns_m = int(np.count_nonzero(signs_m[1:] != signs_m[:-1])) if signs_m.size >= 2 else 0
                if turns_m <= 1 and float(rms_mono) <= float(rms) * 1.20:
                    K, dist, rms = K_mono, dist_mono, rms_mono
                    monotonic_fallback_applied = True
            except Exception:
                pass

    stats = {
        "image_size": image_size,
        "num_images_total": len(calib_images),
        "num_images_used": len(all_charuco_corners),
        "used_indices": used_idx,
        "per_image_num_charuco": per_img_charuco,
        "per_image_num_aruco": per_img_aruco,
        "rms": float(rms),
        "calibration_flags": int(pass_stats[-1]["calibration_flags"]) if pass_stats else int(flags),
        "calibration_flags_requested": int(flags),
        "dist_coeff_count_requested": int(dist_coeff_count),
        "dist_coeff_count_returned": int(np.asarray(dist).reshape(-1).size),
        "charuco_intrinsic_refinement_enabled": bool(requested_refinement_passes > 0),
        "charuco_intrinsic_refinement_passes_requested": int(requested_refinement_passes),
        "charuco_intrinsic_refinement_passes_completed": int(max(0, len(pass_stats) - 1)),
        "charuco_refinement_pass_rms": [
            float(pass_stat["rms"]) for pass_stat in pass_stats
        ],
        "charuco_refinement_pass_num_images_used": [
            int(pass_stat["num_images_used"]) for pass_stat in pass_stats
        ],
        "charuco_refinement_pass_used_intrinsics": [
            bool(pass_stat["used_intrinsics_for_charuco"]) for pass_stat in pass_stats
        ],
        "charuco_refinement_pass_mean_corners": [
            float(pass_stat["mean_charuco_corners"]) for pass_stat in pass_stats
        ],
        "charuco_refinement_pass_fx_fy_cx_cy": [
            [
                float(pass_stat["fx"]),
                float(pass_stat["fy"]),
                float(pass_stat["cx"]),
                float(pass_stat["cy"]),
            ]
            for pass_stat in pass_stats
        ],
        "charuco_interpolation_api": (
            "interpolateCornersCharuco"
            if hasattr(cv2.aruco, "interpolateCornersCharuco")
            else "CharucoDetector"
        ),
        "charuco_interpolation_mode": (
            "intrinsics_refined"
            if requested_refinement_passes > 0
            else "homography_bootstrap_only"
        ),
        "charuco_calibration_api": (
            "calibrateCameraCharuco"
            if hasattr(cv2.aruco, "calibrateCameraCharuco")
            else "calibrateCamera_fallback"
        ),
        "monotonic_fallback_applied": bool(monotonic_fallback_applied),
    }

    return K, dist, float(rms), stats


def estimate_charuco_pose(
    image: np.ndarray,
    board: Any,
    aruco_dict: Any,
    K: np.ndarray,
    dist: np.ndarray,
    detector_params: Optional[Any] = None,
    min_charuco_corners: int = MIN_CHARUCO_LIVE_FOUND,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], CharucoDetection, bool]:
    det = detect_charuco(
        image,
        board,
        aruco_dict,
        detector_params,
        camera_matrix=K,
        dist_coeffs=dist,
    )

    if det.charuco_ids is None or det.charuco_corners is None:
        return None, None, det, False
    if det.num_charuco < min_charuco_corners:
        return None, None, det, False

    obj_pts = _charuco_object_points(board, det.charuco_ids)
    img_pts = det.charuco_corners.reshape(-1, 2).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    return rvec, tvec, det, bool(ok)


def reprojection_error_charuco(
    test_images: Sequence[np.ndarray],
    board: Any,
    aruco_dict: Any,
    K: np.ndarray,
    dist: np.ndarray,
    detector_params: Optional[Any] = None,
    min_charuco_corners: int = MIN_CHARUCO_LIVE_FOUND,
) -> Tuple[float, List[Optional[float]], Dict[str, Any]]:
    per_view = []
    all_err = []
    all_corner_err: list[float] = []
    edge_corner_err: list[float] = []
    center_corner_err: list[float] = []
    radial_residuals: list[float] = []
    tangential_residuals: list[float] = []

    per_img_charuco = []
    per_img_aruco = []
    used_idx = []

    for i, img in enumerate(test_images):
        rvec, tvec, det, ok = estimate_charuco_pose(
            img,
            board,
            aruco_dict,
            K,
            dist,
            detector_params,
            min_charuco_corners,
        )

        per_img_charuco.append(det.num_charuco)
        per_img_aruco.append(det.num_aruco)

        if not ok:
            per_view.append(None)
            continue

        obj_pts = _charuco_object_points(board, det.charuco_ids)
        obs = det.charuco_corners.reshape(-1, 2).astype(np.float32)

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)

        residual = obs - proj
        err = np.linalg.norm(residual, axis=1)
        mean_err = float(np.mean(err))

        per_view.append(mean_err)
        all_err.append(mean_err)
        all_corner_err.extend(float(x) for x in err.reshape(-1))
        used_idx.append(i)

        cx, cy = float(K[0, 2]), float(K[1, 2])
        width, height = _image_size(img)
        edge_band_u = COVERAGE_EDGE_FRACTION * float(width)
        edge_band_v = COVERAGE_EDGE_FRACTION * float(height)
        is_edge = (
            (obs[:, 0] <= edge_band_u)
            | (obs[:, 0] >= (1.0 - COVERAGE_EDGE_FRACTION) * float(width))
            | (obs[:, 1] <= edge_band_v)
            | (obs[:, 1] >= (1.0 - COVERAGE_EDGE_FRACTION) * float(height))
        )
        edge_corner_err.extend(float(x) for x in err[is_edge].reshape(-1))
        center_corner_err.extend(float(x) for x in err[~is_edge].reshape(-1))

        rel = obs.astype(np.float64) - np.asarray([cx, cy], dtype=np.float64)
        radius = np.linalg.norm(rel, axis=1)
        valid_radius = radius > 1e-9
        if np.any(valid_radius):
            radial_unit = rel[valid_radius] / radius[valid_radius, None]
            tangential_unit = np.c_[-radial_unit[:, 1], radial_unit[:, 0]]
            res_valid = residual[valid_radius].astype(np.float64)
            radial_residuals.extend(
                float(x) for x in np.sum(res_valid * radial_unit, axis=1)
            )
            tangential_residuals.extend(
                float(x) for x in np.sum(res_valid * tangential_unit, axis=1)
            )

    mean_px = float(np.mean(all_err)) if all_err else float("nan")
    corner_err_arr = np.asarray(all_corner_err, dtype=np.float64)
    edge_err_arr = np.asarray(edge_corner_err, dtype=np.float64)
    center_err_arr = np.asarray(center_corner_err, dtype=np.float64)
    radial_arr = np.asarray(radial_residuals, dtype=np.float64)
    tangential_arr = np.asarray(tangential_residuals, dtype=np.float64)

    stats = {
        "num_images_total": len(test_images),
        "num_images_valid": len(all_err),
        "used_indices": used_idx,
        "per_image_num_charuco": per_img_charuco,
        "per_image_num_aruco": per_img_aruco,
        "charuco_interpolation_api": (
            "interpolateCornersCharuco"
            if hasattr(cv2.aruco, "interpolateCornersCharuco")
            else "CharucoDetector"
        ),
        "corner_reprojection_mean_px": float(np.mean(corner_err_arr))
        if corner_err_arr.size
        else float("nan"),
        "corner_reprojection_p95_px": float(np.percentile(corner_err_arr, 95))
        if corner_err_arr.size
        else float("nan"),
        "edge_corner_reprojection_mean_px": float(np.mean(edge_err_arr))
        if edge_err_arr.size
        else float("nan"),
        "center_corner_reprojection_mean_px": float(np.mean(center_err_arr))
        if center_err_arr.size
        else float("nan"),
        "radial_residual_mean_px": float(np.mean(radial_arr))
        if radial_arr.size
        else float("nan"),
        "radial_residual_abs_mean_px": float(np.mean(np.abs(radial_arr)))
        if radial_arr.size
        else float("nan"),
        "tangential_residual_abs_mean_px": float(np.mean(np.abs(tangential_arr)))
        if tangential_arr.size
        else float("nan"),
        "edge_corner_count": int(edge_err_arr.size),
        "corner_count": int(corner_err_arr.size),
    }

    return mean_px, per_view, stats


def calibration_model_specs() -> list[CalibrationModelSpec]:
    return [
        CalibrationModelSpec(
            name="standard5",
            flags=0,
            dist_coeff_count=5,
            description="OpenCV k1,k2,p1,p2,k3",
        ),
        CalibrationModelSpec(
            name="no_k3",
            flags=int(getattr(cv2, "CALIB_FIX_K3", 0)),
            dist_coeff_count=5,
            description="OpenCV k1,k2,p1,p2 with k3 fixed to zero",
        ),
        CalibrationModelSpec(
            name="rational8",
            flags=int(getattr(cv2, "CALIB_RATIONAL_MODEL", 0)),
            dist_coeff_count=8,
            description="OpenCV rational model k1,k2,p1,p2,k3,k4,k5,k6",
        ),
    ]


def calibration_variant_path(base_path: Path, model_name: str) -> Path:
    base_path = Path(base_path).expanduser().resolve()
    return base_path.with_name(f"{base_path.stem}_{model_name}{base_path.suffix}")


def model_comparison_path(base_path: Path) -> Path:
    base_path = Path(base_path).expanduser().resolve()
    return base_path.with_name(f"{base_path.stem}_model_comparison.csv")


def _dist_coeff(dist: np.ndarray, idx: int) -> float:
    values = np.asarray(dist, dtype=np.float64).reshape(-1)
    return float(values[idx]) if idx < values.size else 0.0


def radial_scale_for_dist(r: np.ndarray, dist: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    r2 = r * r
    r4 = r2 * r2
    r6 = r4 * r2
    numerator = (
        1.0
        + _dist_coeff(dist, 0) * r2
        + _dist_coeff(dist, 1) * r4
        + _dist_coeff(dist, 4) * r6
    )
    values = np.asarray(dist, dtype=np.float64).reshape(-1)
    if values.size >= 8:
        denominator = (
            1.0
            + _dist_coeff(dist, 5) * r2
            + _dist_coeff(dist, 6) * r4
            + _dist_coeff(dist, 7) * r6
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return numerator / denominator
    return numerator


def max_normalized_image_radius(K: np.ndarray, image_size: tuple[int, int]) -> float:
    width, height = int(image_size[0]), int(image_size[1])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    corners = (
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (0.0, float(height - 1)),
        (float(width - 1), float(height - 1)),
    )
    return max(float(np.hypot((u - cx) / fx, (v - cy) / fy)) for u, v in corners)


def radial_plausibility_stats(K: np.ndarray, dist: np.ndarray, image_size: tuple[int, int]) -> dict[str, Any]:
    r_max = max_normalized_image_radius(K, image_size)
    r = np.linspace(0.0, r_max, 400)
    scale = radial_scale_for_dist(r, dist)
    finite = np.isfinite(scale)
    if not np.any(finite):
        return {
            "radial_r_max": float(r_max),
            "radial_scale_min": np.nan,
            "radial_scale_max": np.nan,
            "radial_monotonic": False,
            "radial_positive": False,
            "radial_turn_count": -1,
        }

    scale_f = scale[finite]
    diff = np.diff(scale_f)
    eps = 1e-7
    signs = np.sign(diff[np.abs(diff) > eps])
    turn_count = int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size >= 2 else 0
    monotonic = bool(turn_count == 0)
    positive = bool(float(np.nanmin(scale_f)) > 0.0)
    return {
        "radial_r_max": float(r_max),
        "radial_scale_min": float(np.nanmin(scale_f)),
        "radial_scale_max": float(np.nanmax(scale_f)),
        "radial_monotonic": monotonic,
        "radial_positive": positive,
        "radial_turn_count": turn_count,
    }


def calibration_model_quality_score(stats: Dict[str, Any]) -> float:
    def value(name: str, default: float = 0.0) -> float:
        try:
            out = float(stats.get(name, default))
        except (TypeError, ValueError):
            return float(default)
        return out if np.isfinite(out) else float(default)

    mean_px = value("corner_reprojection_mean_px", value("selected_reprojection_mean_px", 10.0))
    p95_px = value("corner_reprojection_p95_px", mean_px)
    edge_px = value("edge_corner_reprojection_mean_px", mean_px)
    radial_bias_px = abs(value("radial_residual_mean_px", 0.0))
    radial_abs_px = value("radial_residual_abs_mean_px", mean_px)
    tangential_abs_px = value("tangential_residual_abs_mean_px", 0.0)
    dist_abs_max = value("dist_coeff_abs_max", 0.0)
    radial_turns = int(value("radial_turn_count", 0.0))
    radial_positive = bool(stats.get("radial_positive", True))

    score = (
        mean_px
        + 0.35 * p95_px
        + 0.75 * edge_px
        + 1.20 * radial_bias_px
        + 0.35 * radial_abs_px
        + 0.10 * tangential_abs_px
        + 0.012 * dist_abs_max
    )
    if not radial_positive:
        score += 100.0
    if radial_turns > 0:
        score += 2.0 * float(radial_turns)
    return float(score)


def make_charuco_board() -> tuple[Any, Any, Any]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_ID)
    detector_params = _make_detector_params()

    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LEN_M,
            MARKER_LEN_M,
            aruco_dict,
        )
    elif hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            SQUARES_X,
            SQUARES_Y,
            SQUARE_LEN_M,
            MARKER_LEN_M,
            aruco_dict,
        )
    else:
        raise RuntimeError("No compatible ChArUco board API found.")

    return board, aruco_dict, detector_params


def draw_text_box(
    img_bgr: np.ndarray,
    lines: list[str],
    org: tuple[int, int] = (30, 55),
    color: tuple[int, int, int] = (255, 255, 255),
    line_gap: int = 35,
    font_scale: float = 1.0,
) -> np.ndarray:
    out = img_bgr.copy()
    x, y = org

    for i, text in enumerate(lines):
        yy = y + i * line_gap
        cv2.putText(
            out,
            text,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            text,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def draw_live_overlay(
    frame_bgr: np.ndarray,
    det: Optional[CharucoDetection],
    found: bool,
    num_candidates: int,
    target_views: int,
    have_calibration: bool,
    mean_reproj_px: Optional[float],
    recording: bool = False,
    num_selected: int = 0,
    coverage: Optional[Dict[str, Any]] = None,
    view_coverage: Optional[Dict[str, Any]] = None,
    coverage_ok: bool = False,
    view_coverage_ok: bool = False,
) -> np.ndarray:
    vis = frame_bgr.copy()

    if det is not None:
        if det.aruco_ids is not None and len(det.aruco_ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, det.aruco_corners, det.aruco_ids)
        if (
            det.charuco_corners is not None
            and det.charuco_ids is not None
            and det.num_charuco > 0
        ):
            try:
                cv2.aruco.drawDetectedCornersCharuco(
                    vis,
                    det.charuco_corners,
                    det.charuco_ids,
                    (255, 255, 0),
                )
            except Exception:
                pts = det.charuco_corners.reshape(-1, 2)
                for u, v in pts:
                    cv2.circle(vis, (int(round(u)), int(round(v))), 4, (255, 255, 0), 2)

    aruco = "-" if det is None else str(int(det.num_aruco))
    charuco = "-" if det is None else str(int(det.num_charuco))
    if recording:
        status = "RECORDING (SPACE stops and calibrates)"
        status_color = (0, 210, 255)
    else:
        status = "READY (SPACE starts auto capture)" if found else "NOT FOUND"
        status_color = (0, 255, 0) if found else (0, 0, 255)

    lines = [
        status,
        f"ArUco: {aruco}/{MAX_ARUCO}  ChArUco: {charuco}/{MAX_CHARUCO_CORNERS}",
        f"Candidates: {num_candidates}  Selected: {num_selected}/{target_views}",
        _coverage_short_text(coverage),
        _view_coverage_short_text(view_coverage),
        "Keys: SPACE start/stop | T accuracy | R redo | Q/ESC quit",
    ]

    if recording and coverage is not None and (not coverage_ok or not view_coverage_ok):
        missing = coverage_failures(coverage)
        if view_coverage is not None:
            missing += view_center_coverage_failures(view_coverage)
        if missing:
            lines.append("Need: " + "; ".join(missing[:2]))

    if have_calibration:
        lines.append("Calibration ready and saved.")
    if mean_reproj_px is not None:
        lines.append(f"Accuracy test mean reprojection: {mean_reproj_px:.3f} px")

    return draw_text_box(vis, lines, color=status_color)


def draw_accuracy_overlay(
    frame_bgr: np.ndarray,
    board: Any,
    aruco_dict: Any,
    detector_params: Any,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, Optional[float]]:
    vis = frame_bgr.copy()

    rvec, tvec, det, ok = estimate_charuco_pose(
        image=vis,
        board=board,
        aruco_dict=aruco_dict,
        K=K,
        dist=dist,
        detector_params=detector_params,
        min_charuco_corners=MIN_CHARUCO_LIVE_FOUND,
    )

    if (not ok) or det.charuco_corners is None or det.charuco_ids is None:
        return draw_text_box(
            vis,
            ["Accuracy test failed: pose not found"],
            color=(0, 0, 255),
        ), None

    if det.aruco_ids is not None and len(det.aruco_ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, det.aruco_corners, det.aruco_ids)

    measured = det.charuco_corners.reshape(-1, 2)
    for u, v in measured:
        cv2.circle(vis, (int(round(u)), int(round(v))), 5, (255, 255, 0), 2, cv2.LINE_AA)

    obj_pts = _charuco_object_points(board, det.charuco_ids)
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)

    for u, v in projected:
        cv2.drawMarker(
            vis,
            (int(round(u)), int(round(v))),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

    residuals = measured - projected
    residual_norm_px = np.linalg.norm(residuals, axis=1)
    mean_px = float(np.mean(residual_norm_px))

    return draw_text_box(
        vis,
        [
            "Accuracy test",
            f"ChArUco corners: {len(measured)}/{MAX_CHARUCO_CORNERS}",
            f"Mean reprojection error: {mean_px:.3f} px",
            "cyan = measured, red = projected",
        ],
        color=(255, 255, 0),
    ), mean_px


def default_output_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / f"hydramarker_camera_calibration_{ts}.npz"


def save_tracking_calibration_npz(
    path: Path,
    *,
    K: np.ndarray,
    dist: np.ndarray,
    image_size: tuple[int, int],
    rms: float,
    stats: Dict[str, Any],
    latest_accuracy_mean_px: Optional[float] = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    width, height = int(image_size[0]), int(image_size[1])

    payload = {
        "K": K,
        "dist": dist,
        "camera_matrix": K,
        "camera_intrinsics": K,
        "intrinsics": K,
        "K_rgb": K,
        "dist_coeffs": dist,
        "distortion_coeffs": dist,
        "opencv_dist_coeffs": dist,
        "effective_opencv_dist_coeffs": dist,
        "dist_rgb": dist,
        "image_size": np.asarray([width, height], dtype=np.int32),
        "width": np.asarray(width, dtype=np.int32),
        "height": np.asarray(height, dtype=np.int32),
        "image_width": np.asarray(width, dtype=np.int32),
        "image_height": np.asarray(height, dtype=np.int32),
        "rgb_width": np.asarray(width, dtype=np.int32),
        "rgb_height": np.asarray(height, dtype=np.int32),
        "rms": np.asarray(float(rms), dtype=np.float64),
        "calibration_rms": np.asarray(float(rms), dtype=np.float64),
        "calibration_model": np.asarray(str(stats.get("calibration_model", "standard5"))),
        "calibration_model_description": np.asarray(
            str(stats.get("calibration_model_description", ""))
        ),
        "calibration_flags": np.asarray(
            int(stats.get("calibration_flags", 0)),
            dtype=np.int32,
        ),
        "dist_coeff_count_requested": np.asarray(
            int(stats.get("dist_coeff_count_requested", dist.size)),
            dtype=np.int32,
        ),
        "dist_coeff_count_returned": np.asarray(
            int(stats.get("dist_coeff_count_returned", dist.size)),
            dtype=np.int32,
        ),
        "charuco_intrinsic_refinement_enabled": np.asarray(
            bool(stats.get("charuco_intrinsic_refinement_enabled", False)),
            dtype=np.bool_,
        ),
        "charuco_intrinsic_refinement_passes_requested": np.asarray(
            int(stats.get("charuco_intrinsic_refinement_passes_requested", 0)),
            dtype=np.int32,
        ),
        "charuco_intrinsic_refinement_passes_completed": np.asarray(
            int(stats.get("charuco_intrinsic_refinement_passes_completed", 0)),
            dtype=np.int32,
        ),
        "charuco_refinement_pass_rms": np.asarray(
            stats.get("charuco_refinement_pass_rms", []),
            dtype=np.float64,
        ),
        "charuco_refinement_pass_num_images_used": np.asarray(
            stats.get("charuco_refinement_pass_num_images_used", []),
            dtype=np.int32,
        ),
        "charuco_refinement_pass_used_intrinsics": np.asarray(
            stats.get("charuco_refinement_pass_used_intrinsics", []),
            dtype=np.bool_,
        ),
        "charuco_refinement_pass_mean_corners": np.asarray(
            stats.get("charuco_refinement_pass_mean_corners", []),
            dtype=np.float64,
        ),
        "charuco_refinement_pass_fx_fy_cx_cy": np.asarray(
            stats.get("charuco_refinement_pass_fx_fy_cx_cy", []),
            dtype=np.float64,
        ),
        "charuco_interpolation_mode": np.asarray(
            str(stats.get("charuco_interpolation_mode", "unknown"))
        ),
        "num_images_total": np.asarray(
            int(stats.get("num_images_total", 0)),
            dtype=np.int32,
        ),
        "num_images_used": np.asarray(
            int(stats.get("num_images_used", 0)),
            dtype=np.int32,
        ),
        "used_indices": np.asarray(stats.get("used_indices", []), dtype=np.int32),
        "per_image_num_charuco": np.asarray(
            stats.get("per_image_num_charuco", []),
            dtype=np.int32,
        ),
        "per_image_num_aruco": np.asarray(
            stats.get("per_image_num_aruco", []),
            dtype=np.int32,
        ),
        "num_candidates_total": np.asarray(
            int(stats.get("num_candidates_total", 0)),
            dtype=np.int32,
        ),
        "num_candidates_selected": np.asarray(
            int(stats.get("num_candidates_selected", 0)),
            dtype=np.int32,
        ),
        "selected_frame_indices": np.asarray(
            stats.get("selected_frame_indices", []),
            dtype=np.int32,
        ),
        "selected_candidate_scores": np.asarray(
            stats.get("selected_candidate_scores", []),
            dtype=np.float64,
        ),
        "selected_candidate_num_charuco": np.asarray(
            stats.get("selected_candidate_num_charuco", []),
            dtype=np.int32,
        ),
        "selected_candidate_centroids_uv": np.asarray(
            stats.get("selected_candidate_centroids_uv", []),
            dtype=np.float64,
        ),
        "selected_candidate_bbox_area_norm": np.asarray(
            stats.get("selected_candidate_bbox_area_norm", []),
            dtype=np.float64,
        ),
        "selected_reprojection_per_view_px": np.asarray(
            stats.get("selected_reprojection_per_view_px", []),
            dtype=np.float64,
        ),
        "selected_reprojection_mean_px": np.asarray(
            float(stats.get("selected_reprojection_mean_px", np.nan)),
            dtype=np.float64,
        ),
        "candidate_coverage_cells": np.asarray(
            int(stats.get("candidate_coverage_cells", 0)),
            dtype=np.int32,
        ),
        "candidate_coverage_total_cells": np.asarray(
            int(stats.get("candidate_coverage_total_cells", 0)),
            dtype=np.int32,
        ),
        "candidate_coverage_grid_counts": np.asarray(
            stats.get("candidate_coverage_grid_counts", []),
            dtype=np.int32,
        ),
        "candidate_coverage_edge_counts_lrtb": np.asarray(
            stats.get("candidate_coverage_edge_counts", []),
            dtype=np.int32,
        ),
        "selected_coverage_cells": np.asarray(
            int(stats.get("selected_coverage_cells", 0)),
            dtype=np.int32,
        ),
        "selected_coverage_total_cells": np.asarray(
            int(stats.get("selected_coverage_total_cells", 0)),
            dtype=np.int32,
        ),
        "selected_coverage_grid_counts": np.asarray(
            stats.get("selected_coverage_grid_counts", []),
            dtype=np.int32,
        ),
        "selected_coverage_edge_counts_lrtb": np.asarray(
            stats.get("selected_coverage_edge_counts", []),
            dtype=np.int32,
        ),
        "candidate_view_center_cells": np.asarray(
            int(stats.get("candidate_view_center_cells", 0)),
            dtype=np.int32,
        ),
        "candidate_view_center_total_cells": np.asarray(
            int(stats.get("candidate_view_center_total_cells", 0)),
            dtype=np.int32,
        ),
        "candidate_view_center_grid_counts": np.asarray(
            stats.get("candidate_view_center_grid_counts", []),
            dtype=np.int32,
        ),
        "candidate_view_center_quadrant_counts": np.asarray(
            stats.get("candidate_view_center_quadrant_counts", []),
            dtype=np.int32,
        ),
        "candidate_view_center_edge_counts_lrtb": np.asarray(
            stats.get("candidate_view_center_edge_counts", []),
            dtype=np.int32,
        ),
        "candidate_view_center_count": np.asarray(
            int(stats.get("candidate_view_center_count", 0)),
            dtype=np.int32,
        ),
        "candidate_view_center_dx_norm_p05_p50_p95": np.asarray(
            stats.get("candidate_view_center_dx_norm_p05_p50_p95", []),
            dtype=np.float64,
        ),
        "candidate_view_center_dy_norm_p05_p50_p95": np.asarray(
            stats.get("candidate_view_center_dy_norm_p05_p50_p95", []),
            dtype=np.float64,
        ),
        "candidate_view_center_radius_norm_p05_p50_p95_max": np.asarray(
            stats.get("candidate_view_center_radius_norm_p05_p50_p95_max", []),
            dtype=np.float64,
        ),
        "selected_view_center_cells": np.asarray(
            int(stats.get("selected_view_center_cells", 0)),
            dtype=np.int32,
        ),
        "selected_view_center_total_cells": np.asarray(
            int(stats.get("selected_view_center_total_cells", 0)),
            dtype=np.int32,
        ),
        "selected_view_center_grid_counts": np.asarray(
            stats.get("selected_view_center_grid_counts", []),
            dtype=np.int32,
        ),
        "selected_view_center_quadrant_counts": np.asarray(
            stats.get("selected_view_center_quadrant_counts", []),
            dtype=np.int32,
        ),
        "selected_view_center_edge_counts_lrtb": np.asarray(
            stats.get("selected_view_center_edge_counts", []),
            dtype=np.int32,
        ),
        "selected_view_center_count": np.asarray(
            int(stats.get("selected_view_center_count", 0)),
            dtype=np.int32,
        ),
        "selected_view_center_dx_norm_p05_p50_p95": np.asarray(
            stats.get("selected_view_center_dx_norm_p05_p50_p95", []),
            dtype=np.float64,
        ),
        "selected_view_center_dy_norm_p05_p50_p95": np.asarray(
            stats.get("selected_view_center_dy_norm_p05_p50_p95", []),
            dtype=np.float64,
        ),
        "selected_view_center_radius_norm_p05_p50_p95_max": np.asarray(
            stats.get("selected_view_center_radius_norm_p05_p50_p95_max", []),
            dtype=np.float64,
        ),
        "view_center_grid_cols": np.asarray(
            int(stats.get("view_center_grid_cols", VIEW_CENTER_GRID_COLS)),
            dtype=np.int32,
        ),
        "view_center_grid_rows": np.asarray(
            int(stats.get("view_center_grid_rows", VIEW_CENTER_GRID_ROWS)),
            dtype=np.int32,
        ),
        "view_center_min_cells": np.asarray(
            int(stats.get("view_center_min_cells", MIN_VIEW_CENTER_CELLS)),
            dtype=np.int32,
        ),
        "view_center_min_center_views": np.asarray(
            int(stats.get("view_center_min_center_views", MIN_CENTER_VIEWS)),
            dtype=np.int32,
        ),
        "view_center_min_quadrant_views": np.asarray(
            int(stats.get("view_center_min_quadrant_views", MIN_VIEW_QUADRANT_VIEWS)),
            dtype=np.int32,
        ),
        "view_center_half_width_norm": np.asarray(
            float(stats.get("view_center_half_width_norm", CENTER_VIEW_HALF_WIDTH_NORM)),
            dtype=np.float64,
        ),
        "coverage_min_cells": np.asarray(
            int(stats.get("coverage_min_cells", 0)),
            dtype=np.int32,
        ),
        "coverage_min_edge_corners": np.asarray(
            int(stats.get("coverage_min_edge_corners", 0)),
            dtype=np.int32,
        ),
        "coverage_min_quadrant_corners": np.asarray(
            int(stats.get("coverage_min_quadrant_corners", 0)),
            dtype=np.int32,
        ),
        "coverage_min_corner_radius_norm": np.asarray(
            float(stats.get("coverage_min_corner_radius_norm", MIN_CORNER_RADIUS_NORM)),
            dtype=np.float64,
        ),
        "candidate_coverage_max_corner_radius_norm": np.asarray(
            float(stats.get("candidate_coverage_max_corner_radius_norm", np.nan)),
            dtype=np.float64,
        ),
        "candidate_coverage_p95_corner_radius_norm": np.asarray(
            float(stats.get("candidate_coverage_p95_corner_radius_norm", np.nan)),
            dtype=np.float64,
        ),
        "selected_coverage_max_corner_radius_norm": np.asarray(
            float(stats.get("selected_coverage_max_corner_radius_norm", np.nan)),
            dtype=np.float64,
        ),
        "selected_coverage_p95_corner_radius_norm": np.asarray(
            float(stats.get("selected_coverage_p95_corner_radius_norm", np.nan)),
            dtype=np.float64,
        ),
        "min_capture_corners": np.asarray(
            int(stats.get("min_capture_corners", MIN_CHARUCO_CAPTURE)),
            dtype=np.int32,
        ),
        "min_edge_capture_corners": np.asarray(
            int(stats.get("min_edge_capture_corners", MIN_CHARUCO_EDGE_CAPTURE)),
            dtype=np.int32,
        ),
        "coverage_forced_save": np.asarray(
            bool(stats.get("coverage_forced_save", False)),
            dtype=np.bool_,
        ),
        "model_quality_score": np.asarray(
            float(stats.get("model_quality_score", np.nan)),
            dtype=np.float64,
        ),
        "dist_coeff_abs_max": np.asarray(
            float(stats.get("dist_coeff_abs_max", np.nan)),
            dtype=np.float64,
        ),
        "corner_reprojection_mean_px": np.asarray(
            float(stats.get("corner_reprojection_mean_px", np.nan)),
            dtype=np.float64,
        ),
        "corner_reprojection_p95_px": np.asarray(
            float(stats.get("corner_reprojection_p95_px", np.nan)),
            dtype=np.float64,
        ),
        "edge_corner_reprojection_mean_px": np.asarray(
            float(stats.get("edge_corner_reprojection_mean_px", np.nan)),
            dtype=np.float64,
        ),
        "center_corner_reprojection_mean_px": np.asarray(
            float(stats.get("center_corner_reprojection_mean_px", np.nan)),
            dtype=np.float64,
        ),
        "radial_residual_mean_px": np.asarray(
            float(stats.get("radial_residual_mean_px", np.nan)),
            dtype=np.float64,
        ),
        "radial_residual_abs_mean_px": np.asarray(
            float(stats.get("radial_residual_abs_mean_px", np.nan)),
            dtype=np.float64,
        ),
        "tangential_residual_abs_mean_px": np.asarray(
            float(stats.get("tangential_residual_abs_mean_px", np.nan)),
            dtype=np.float64,
        ),
        "edge_corner_count": np.asarray(
            int(stats.get("edge_corner_count", 0)),
            dtype=np.int32,
        ),
        "corner_count": np.asarray(
            int(stats.get("corner_count", 0)),
            dtype=np.int32,
        ),
        "radial_r_max": np.asarray(
            float(stats.get("radial_r_max", np.nan)),
            dtype=np.float64,
        ),
        "radial_scale_min": np.asarray(
            float(stats.get("radial_scale_min", np.nan)),
            dtype=np.float64,
        ),
        "radial_scale_max": np.asarray(
            float(stats.get("radial_scale_max", np.nan)),
            dtype=np.float64,
        ),
        "radial_monotonic": np.asarray(
            bool(stats.get("radial_monotonic", False)),
            dtype=np.bool_,
        ),
        "radial_positive": np.asarray(
            bool(stats.get("radial_positive", False)),
            dtype=np.bool_,
        ),
        "radial_turn_count": np.asarray(
            int(stats.get("radial_turn_count", -1)),
            dtype=np.int32,
        ),
        "squares_x": np.asarray(SQUARES_X, dtype=np.int32),
        "squares_y": np.asarray(SQUARES_Y, dtype=np.int32),
        "square_length_m": np.asarray(SQUARE_LEN_M, dtype=np.float64),
        "marker_length_m": np.asarray(MARKER_LEN_M, dtype=np.float64),
        "aruco_dictionary_id": np.asarray(DICT_ID, dtype=np.int32),
        "created_at": np.asarray(datetime.now().isoformat(timespec="seconds")),
        "camera_source": np.asarray("charuco_realsense_color"),
        "distortion_model": np.asarray(
            str(stats.get("distortion_model", "opencv_brown_conrady"))
        ),
        "recommended_primary_model": np.asarray(
            str(stats.get("recommended_primary_model", ""))
        ),
        "charuco_interpolation_api": np.asarray(
            str(stats.get("charuco_interpolation_api", "unknown"))
        ),
        "charuco_calibration_api": np.asarray(
            str(stats.get("charuco_calibration_api", "unknown"))
        ),
        "diagnostics_dir": np.asarray(str(stats.get("diagnostics_dir", ""))),
    }

    if latest_accuracy_mean_px is not None:
        payload["latest_accuracy_mean_reproj_px"] = np.asarray(
            float(latest_accuracy_mean_px),
            dtype=np.float64,
        )

    np.savez(path, **payload)
    return path


def start_realsense(
    *,
    width: int = REALSENSE_WIDTH,
    height: int = REALSENSE_HEIGHT,
    fps: int = REALSENSE_FPS,
):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is required for this calibration script."
        ) from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)

    for _ in range(15):
        pipeline.wait_for_frames()

    return pipeline, profile


def get_realsense_factory_intrinsics(
    profile: Any,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """Return D435i factory K matrix from pyrealsense2 profile, or None on failure."""
    try:
        import pyrealsense2 as rs
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        if intr.width != width or intr.height != height:
            return None
        return np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    except Exception:
        return None


def color_frame_to_bgr(color_frame) -> np.ndarray:
    return np.asanyarray(color_frame.get_data()).copy()


def get_color_frame_bgr(pipeline) -> Optional[np.ndarray]:
    frames = pipeline.poll_for_frames()
    if not frames:
        return None

    color_frame = frames.get_color_frame()
    if not color_frame:
        return None

    return color_frame_to_bgr(color_frame)


def reset_state() -> dict[str, Any]:
    return {
        "candidates": [],
        "selected_candidates": [],
        "recording": False,
        "recording_started_s": None,
        "last_candidate_time_s": -float("inf"),
        "coverage": None,
        "view_coverage": None,
        "K": None,
        "dist": None,
        "rms": None,
        "stats": None,
        "saved_path": None,
        "diagnostics_dir": None,
        "latest_accuracy_mean_px": None,
    }


def save_calibration_diagnostics(
    output_path: Path,
    *,
    candidates: Sequence[CalibrationCandidate],
    selected_candidates: Sequence[CalibrationCandidate],
    selected_reprojection_errors: Sequence[Optional[float]],
    save_selected_images: bool,
    grid_cols: int = SELECTION_GRID_COLS,
    grid_rows: int = SELECTION_GRID_ROWS,
    edge_fraction: float = COVERAGE_EDGE_FRACTION,
) -> Path:
    diagnostics_dir = output_path.with_suffix("")
    diagnostics_dir = diagnostics_dir.parent / f"{diagnostics_dir.name}_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    selected_by_frame = {
        cand.frame_index: i for i, cand in enumerate(selected_candidates)
    }
    reproj_by_frame = {
        cand.frame_index: selected_reprojection_errors[i]
        for i, cand in enumerate(selected_candidates)
        if i < len(selected_reprojection_errors)
    }

    fieldnames = [
        "candidate_rank",
        "frame_index",
        "capture_time_s",
        "used_for_calibration",
        "selected_rank",
        "score",
        "num_charuco",
        "num_aruco",
        "corner_fraction",
        "sharpness",
        "centroid_u",
        "centroid_v",
        "centroid_u_norm",
        "centroid_v_norm",
        "bbox_area_norm",
        "bbox_diag_norm",
        "edge_margin_px",
        "edge_margin_norm",
        "radius_norm",
        "corner_radius_norm_max",
        "grid_col",
        "grid_row",
        "corner_grid_cells",
        "edge_corners_lrtb",
        "radius_bin",
        "size_bin",
        "selected_reprojection_mean_px",
    ]
    image_size = _image_size(candidates[0].image) if candidates else (0, 0)

    csv_path = diagnostics_dir / "calibration_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for rank, cand in enumerate(candidates):
            selected_rank = selected_by_frame.get(cand.frame_index)
            grid_col, grid_row = _candidate_grid_cell(cand.metrics)
            pts = _candidate_corner_points(cand)
            corner_cells = sorted(
                _corner_grid_cells(
                    pts,
                    image_size,
                    grid_cols=grid_cols,
                    grid_rows=grid_rows,
                )
            )
            width, height = image_size
            edge_counts = [
                int(np.count_nonzero(pts[:, 0] <= edge_fraction * width)) if pts.size else 0,
                int(np.count_nonzero(pts[:, 0] >= (1.0 - edge_fraction) * width)) if pts.size else 0,
                int(np.count_nonzero(pts[:, 1] <= edge_fraction * height)) if pts.size else 0,
                int(np.count_nonzero(pts[:, 1] >= (1.0 - edge_fraction) * height)) if pts.size else 0,
            ]
            reproj = reproj_by_frame.get(cand.frame_index)
            row = {
                "candidate_rank": rank,
                "frame_index": cand.frame_index,
                "capture_time_s": f"{cand.capture_time_s:.6f}",
                "used_for_calibration": int(selected_rank is not None),
                "selected_rank": "" if selected_rank is None else selected_rank,
                "score": f"{cand.score:.6f}",
                "grid_col": grid_col,
                "grid_row": grid_row,
                "corner_grid_cells": " ".join(f"{col}:{row}" for col, row in corner_cells),
                "edge_corners_lrtb": "/".join(str(v) for v in edge_counts),
                "radius_bin": _candidate_radius_bin(cand.metrics),
                "size_bin": _candidate_size_bin(cand.metrics),
                "selected_reprojection_mean_px": (
                    "" if reproj is None else f"{float(reproj):.6f}"
                ),
            }
            for key in fieldnames:
                if key in row:
                    continue
                value = cand.metrics.get(key, "")
                if isinstance(value, float):
                    row[key] = f"{value:.6f}"
                else:
                    row[key] = value
            writer.writerow(row)

    if save_selected_images:
        views_dir = diagnostics_dir / "selected_views"
        views_dir.mkdir(parents=True, exist_ok=True)
        for rank, cand in enumerate(selected_candidates):
            image_path = views_dir / f"selected_{rank:03d}_frame_{cand.frame_index:06d}.png"
            cv2.imwrite(str(image_path), cand.image)

    return diagnostics_dir


def write_model_comparison_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return

    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect_image_paths(images_dir: Path) -> list[Path]:
    images_dir = Path(images_dir).expanduser().resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in suffixes)
    if not paths:
        raise RuntimeError(f"No calibration images found in {images_dir}")
    return paths


def load_images(paths: Sequence[Path]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        images.append(img)
    return images


def run_model_sweep_for_images(
    *,
    calib_images: Sequence[np.ndarray],
    output_path: Path,
    min_capture_corners: int,
    intrinsic_refinement_passes: int = CHARUCO_INTRINSIC_REFINEMENT_PASSES,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    board, aruco_dict, detector_params = make_charuco_board()
    image_size = _image_size(calib_images[0])

    model_rows: list[dict[str, Any]] = []
    first_success: Path | None = None
    best_success: tuple[np.ndarray, np.ndarray, float, dict[str, Any]] | None = None

    for spec in calibration_model_specs():
        print(f"[calib_camera] Model {spec.name}: {spec.description}")
        try:
            K, dist, rms, stats = calibrate_charuco_intrinsics(
                calib_images=calib_images,
                board=board,
                aruco_dict=aruco_dict,
                detector_params=detector_params,
                min_charuco_corners=min_capture_corners,
                flags=spec.flags,
                dist_coeff_count=spec.dist_coeff_count,
                intrinsic_refinement_passes=intrinsic_refinement_passes,
            )
            reproj_mean_px, selected_reproj, reproj_stats = reprojection_error_charuco(
                calib_images,
                board=board,
                aruco_dict=aruco_dict,
                K=K,
                dist=dist,
                detector_params=detector_params,
                min_charuco_corners=min_capture_corners,
            )
            selected_reproj_float = [
                float("nan") if value is None else float(value)
                for value in selected_reproj
            ]
            radial_stats = radial_plausibility_stats(K, dist, image_size)
            dist_flat = np.asarray(dist, dtype=np.float64).reshape(-1)
            stats.update(
                {
                    "calibration_model": spec.name,
                    "calibration_model_description": spec.description,
                    "distortion_model": (
                        "opencv_brown_conrady_rational"
                        if spec.name == "rational8"
                        else "opencv_brown_conrady"
                    ),
                    "selected_reprojection_mean_px": float(reproj_mean_px),
                    "selected_reprojection_per_view_px": selected_reproj_float,
                    "dist_coeff_abs_max": float(np.max(np.abs(dist_flat)))
                    if dist_flat.size
                    else 0.0,
                    **reproj_stats,
                    **radial_stats,
                }
            )
            stats["model_quality_score"] = calibration_model_quality_score(stats)

            save_paths: list[Path] = []
            if spec.name == "standard5":
                save_paths.append(output_path)
            variant_path = calibration_variant_path(output_path, spec.name)
            if variant_path not in save_paths:
                save_paths.append(variant_path)

            for save_path in save_paths:
                save_tracking_calibration_npz(
                    save_path,
                    K=K,
                    dist=np.asarray(dist, dtype=np.float64).reshape(-1, 1),
                    image_size=image_size,
                    rms=float(rms),
                    stats=stats,
                )

            if first_success is None:
                first_success = save_paths[0]
            if best_success is None or float(stats["model_quality_score"]) < float(
                best_success[3].get("model_quality_score", float("inf"))
            ):
                best_success = (K, dist, float(rms), stats)

            model_rows.append(
                {
                    "model": spec.name,
                    "description": spec.description,
                    "status": "ok",
                    "saved_paths": " | ".join(str(path) for path in save_paths),
                    "rms": float(rms),
                    "selected_reprojection_mean_px": float(reproj_mean_px),
                    "dist_coeff_count": int(dist_flat.size),
                    "fx": float(K[0, 0]),
                    "fy": float(K[1, 1]),
                    "cx": float(K[0, 2]),
                    "cy": float(K[1, 2]),
                    "dist": " ".join(f"{x:.12g}" for x in dist_flat),
                    "model_quality_score": float(stats["model_quality_score"]),
                    "corner_reprojection_p95_px": stats.get("corner_reprojection_p95_px"),
                    "edge_corner_reprojection_mean_px": stats.get(
                        "edge_corner_reprojection_mean_px"
                    ),
                    "radial_residual_mean_px": stats.get("radial_residual_mean_px"),
                    "radial_residual_abs_mean_px": stats.get(
                        "radial_residual_abs_mean_px"
                    ),
                    **radial_stats,
                }
            )
            print(
                "[calib_camera]   "
                f"rms={float(rms):.6f} reproj={float(reproj_mean_px):.6f}px "
                f"score={float(stats['model_quality_score']):.6f} "
                f"dist_n={dist_flat.size} radial_turns={radial_stats['radial_turn_count']} "
                f"positive={radial_stats['radial_positive']}"
            )
            print(
                "[calib_camera]   saved -> "
                + " | ".join(str(path) for path in save_paths)
            )
        except Exception as exc:
            model_rows.append(
                {
                    "model": spec.name,
                    "description": spec.description,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[calib_camera]   failed: {type(exc).__name__}: {exc}")

    comparison_csv = model_comparison_path(output_path)
    write_model_comparison_csv(comparison_csv, model_rows)
    print(f"[calib_camera] Saved model comparison -> {comparison_csv}")

    if first_success is None:
        raise RuntimeError("Calibration failed for all models.")

    if best_success is not None:
        K, dist, rms, stats = best_success
        recommended_stats = dict(stats)
        recommended_stats["recommended_primary_model"] = str(
            stats.get("calibration_model", "")
        )
        save_tracking_calibration_npz(
            output_path,
            K=K,
            dist=np.asarray(dist, dtype=np.float64).reshape(-1, 1),
            image_size=image_size,
            rms=float(rms),
            stats=recommended_stats,
        )
        print(
            "[calib_camera] Recommended primary model: "
            f"{recommended_stats.get('recommended_primary_model')} "
            f"(score={float(recommended_stats.get('model_quality_score', float('nan'))):.6f}) "
            f"-> {output_path}"
        )
        return output_path

    return first_success


def run_live_calibration(
    *,
    output_path: Path,
    width: int = REALSENSE_WIDTH,
    height: int = REALSENSE_HEIGHT,
    fps: int = REALSENSE_FPS,
    target_views: int = N_VIEWS,
    min_capture_corners: int = MIN_CHARUCO_CAPTURE,
    min_edge_capture_corners: int = MIN_CHARUCO_EDGE_CAPTURE,
    auto_capture_interval_s: float = AUTO_CAPTURE_INTERVAL_S,
    max_candidates: int = MAX_CAPTURE_CANDIDATES,
    save_selected_images: bool = True,
    coverage_grid_cols: int = SELECTION_GRID_COLS,
    coverage_grid_rows: int = SELECTION_GRID_ROWS,
    min_coverage_cells: int = MIN_COVERAGE_CELLS,
    min_edge_corners: int = MIN_EDGE_CORNERS,
    min_quadrant_corners: int = MIN_QUADRANT_CORNERS,
    min_corner_radius_norm: float = MIN_CORNER_RADIUS_NORM,
    view_grid_cols: int = VIEW_CENTER_GRID_COLS,
    view_grid_rows: int = VIEW_CENTER_GRID_ROWS,
    min_view_center_cells: int = MIN_VIEW_CENTER_CELLS,
    min_center_views: int = MIN_CENTER_VIEWS,
    min_view_quadrant_views: int = MIN_VIEW_QUADRANT_VIEWS,
    center_view_half_width_norm: float = CENTER_VIEW_HALF_WIDTH_NORM,
    force_save_insufficient_coverage: bool = False,
    intrinsic_refinement_passes: int = CHARUCO_INTRINSIC_REFINEMENT_PASSES,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    target_views = max(3, int(target_views))
    min_capture_corners = max(4, int(min_capture_corners))
    min_edge_capture_corners = max(4, min(int(min_edge_capture_corners), min_capture_corners))
    auto_capture_interval_s = max(0.0, float(auto_capture_interval_s))
    max_candidates = max(target_views, int(max_candidates))
    coverage_grid_cols = max(2, int(coverage_grid_cols))
    coverage_grid_rows = max(2, int(coverage_grid_rows))
    min_coverage_cells = max(1, min(int(min_coverage_cells), coverage_grid_cols * coverage_grid_rows))
    view_grid_cols = max(2, int(view_grid_cols))
    view_grid_rows = max(2, int(view_grid_rows))
    min_view_center_cells = max(1, min(int(min_view_center_cells), view_grid_cols * view_grid_rows))
    min_center_views = max(0, int(min_center_views))
    min_view_quadrant_views = max(0, int(min_view_quadrant_views))
    center_view_half_width_norm = float(np.clip(center_view_half_width_norm, 0.05, 1.0))
    min_edge_corners = max(1, int(min_edge_corners))
    min_quadrant_corners = max(1, int(min_quadrant_corners))
    min_corner_radius_norm = float(np.clip(float(min_corner_radius_norm), 0.0, 1.0))

    board, aruco_dict, detector_params = make_charuco_board()
    state = reset_state()

    pipeline = None
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        pipeline, rs_profile = start_realsense(width=width, height=height, fps=fps)
        factory_K = get_realsense_factory_intrinsics(rs_profile, width, height)
        if factory_K is not None:
            print(f"[calib_camera] Factory K: fx={factory_K[0,0]:.1f} fy={factory_K[1,1]:.1f} "
                  f"cx={factory_K[0,2]:.1f} cy={factory_K[1,2]:.1f}")
        else:
            print("[calib_camera] Factory intrinsics not available, using identity init.")
        print("[calib_camera] RealSense running.")
        print("[calib_camera] SPACE: start/stop automatic capture")
        print(
            "[calib_camera] Move the board through the full image: center, "
            "corners, edges, near/far, and tilted views."
        )
        print("[calib_camera] T: accuracy test after calibration")
        print("[calib_camera] R: redo")
        print("[calib_camera] Q/ESC: quit")

        last_frame: Optional[np.ndarray] = None
        frame_index = 0

        while True:
            frame = get_color_frame_bgr(pipeline)
            got_new_frame = frame is not None
            if got_new_frame:
                frame_index += 1
            else:
                frame = last_frame
            if frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            last_frame = frame
            last_det = detect_charuco(
                frame,
                board=board,
                aruco_dict=aruco_dict,
                detector_params=detector_params,
            )
            found = bool(last_det.num_charuco >= MIN_CHARUCO_LIVE_FOUND)

            now_s = time.monotonic()
            live_metrics = _candidate_metrics(frame, last_det)
            edge_capture_ready = bool(
                last_det.num_charuco >= min_edge_capture_corners
                and _is_fov_edge_candidate(
                    live_metrics,
                    _image_size(frame),
                    edge_fraction=COVERAGE_EDGE_FRACTION,
                )
            )
            capture_ready = bool(
                last_det.num_charuco >= min_capture_corners or edge_capture_ready
            )
            found = bool(found or capture_ready)
            can_store_candidate = (
                state["recording"]
                and got_new_frame
                and capture_ready
                and len(state["candidates"]) < max_candidates
                and now_s - float(state["last_candidate_time_s"]) >= auto_capture_interval_s
            )
            if can_store_candidate:
                start_s = state["recording_started_s"]
                if start_s is None:
                    start_s = now_s
                    state["recording_started_s"] = start_s
                candidate = make_calibration_candidate(
                    frame,
                    frame_index=frame_index,
                    capture_time_s=now_s - float(start_s),
                    det=last_det,
                )
                state["candidates"].append(candidate)
                state["last_candidate_time_s"] = now_s
                state["coverage"] = compute_corner_coverage(
                    state["candidates"],
                    _image_size(frame),
                    grid_cols=coverage_grid_cols,
                    grid_rows=coverage_grid_rows,
                )
                state["view_coverage"] = compute_view_center_coverage(
                    state["candidates"],
                    _image_size(frame),
                    grid_cols=view_grid_cols,
                    grid_rows=view_grid_rows,
                    center_half_width_norm=center_view_half_width_norm,
                )
                if len(state["candidates"]) % 25 == 0:
                    print(
                        "[calib_camera] Collected "
                        f"{len(state['candidates'])} candidates "
                        f"(latest {last_det.num_charuco} ChArUco corners). "
                        + _coverage_short_text(state["coverage"])
                        + " | "
                        + _view_coverage_short_text(state["view_coverage"])
                    )

            vis = draw_live_overlay(
                frame_bgr=frame,
                det=last_det,
                found=found,
                num_candidates=len(state["candidates"]),
                target_views=target_views,
                have_calibration=state["K"] is not None and state["dist"] is not None,
                mean_reproj_px=state["latest_accuracy_mean_px"],
                recording=bool(state["recording"]),
                num_selected=len(state["selected_candidates"]),
                coverage=state["coverage"],
                view_coverage=state["view_coverage"],
                coverage_ok=not coverage_failures(
                    state["coverage"] or {},
                    min_coverage_cells=min_coverage_cells,
                    min_edge_corners=min_edge_corners,
                    min_quadrant_corners=min_quadrant_corners,
                    min_corner_radius_norm=min_corner_radius_norm,
                ),
                view_coverage_ok=not view_center_coverage_failures(
                    state["view_coverage"] or {},
                    min_view_center_cells=min_view_center_cells,
                    min_center_views=min_center_views,
                    min_view_quadrant_views=min_view_quadrant_views,
                ),
            )

            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            if key == ord("r"):
                state = reset_state()
                print("[calib_camera] Reset calibration run.")
                continue

            if key == ord("t"):
                if state["K"] is None or state["dist"] is None:
                    print("[calib_camera] No calibration yet. Record a run first.")
                    continue

                test_vis, mean_px = draw_accuracy_overlay(
                    frame,
                    board=board,
                    aruco_dict=aruco_dict,
                    detector_params=detector_params,
                    K=state["K"],
                    dist=state["dist"],
                )
                state["latest_accuracy_mean_px"] = mean_px
                if mean_px is not None:
                    state["saved_path"] = save_tracking_calibration_npz(
                        output_path,
                        K=state["K"],
                        dist=state["dist"],
                        image_size=_image_size(frame),
                        rms=state["rms"],
                        stats=state["stats"],
                        latest_accuracy_mean_px=mean_px,
                    )
                    print(
                        "[calib_camera] Updated calibration NPZ with "
                        f"accuracy={mean_px:.3f}px -> {state['saved_path']}"
                    )
                cv2.imshow(WINDOW_NAME, test_vis)
                cv2.waitKey(0)
                continue

            if key != 32:
                continue

            if not state["recording"]:
                state = reset_state()
                state["recording"] = True
                state["recording_started_s"] = time.monotonic()
                state["last_candidate_time_s"] = -float("inf")
                print(
                    "[calib_camera] Automatic capture started. "
                    "Press SPACE again to stop and calibrate."
                )
                print(
                    "[calib_camera] Required coverage: "
                    f"{min_coverage_cells}/{coverage_grid_cols * coverage_grid_rows} grid cells, "
                    f"{min_edge_corners}+ corners near each edge, "
                    f"{min_quadrant_corners}+ corners in each quadrant, "
                    f"Rmax>={min_corner_radius_norm:.2f}."
                )
                print(
                    "[calib_camera] Required board-center views: "
                    f"{min_view_center_cells}/{view_grid_cols * view_grid_rows} center-grid cells, "
                    f"{min_center_views}+ views near image center, "
                    f"{min_view_quadrant_views}+ views per image quadrant."
                )
                print(
                    "[calib_camera] Capture thresholds: "
                    f"{min_capture_corners}+ ChArUco corners normally, "
                    f"{min_edge_capture_corners}+ at the image edge/corners."
                )
                continue

            state["recording"] = False
            print(
                "[calib_camera] Automatic capture stopped with "
                f"{len(state['candidates'])} candidates."
            )

            selected_candidates = select_calibration_candidates(
                state["candidates"],
                target_views=target_views,
                min_charuco_corners=min_capture_corners,
                min_edge_charuco_corners=min_edge_capture_corners,
                image_size=_image_size(frame),
                grid_cols=coverage_grid_cols,
                grid_rows=coverage_grid_rows,
                min_edge_corners=min_edge_corners,
                min_quadrant_corners=min_quadrant_corners,
                view_grid_cols=view_grid_cols,
                view_grid_rows=view_grid_rows,
                min_view_center_cells=min_view_center_cells,
                min_center_views=min_center_views,
                min_view_quadrant_views=min_view_quadrant_views,
                center_view_half_width_norm=center_view_half_width_norm,
            )
            state["selected_candidates"] = selected_candidates

            if len(selected_candidates) < 3:
                print(
                    "[calib_camera] Calibration skipped: need at least 3 usable "
                    f"candidates ({min_capture_corners}+ normally, "
                    f"{min_edge_capture_corners}+ at image edges)."
                )
                continue

            candidate_coverage = compute_corner_coverage(
                state["candidates"],
                _image_size(frame),
                grid_cols=coverage_grid_cols,
                grid_rows=coverage_grid_rows,
            )
            candidate_failures = coverage_failures(
                candidate_coverage,
                min_coverage_cells=min_coverage_cells,
                min_edge_corners=min_edge_corners,
                min_quadrant_corners=min_quadrant_corners,
                min_corner_radius_norm=min_corner_radius_norm,
            )
            selected_coverage = compute_corner_coverage(
                selected_candidates,
                _image_size(frame),
                grid_cols=coverage_grid_cols,
                grid_rows=coverage_grid_rows,
            )
            selected_failures = coverage_failures(
                selected_coverage,
                min_coverage_cells=min_coverage_cells,
                min_edge_corners=min_edge_corners,
                min_quadrant_corners=min_quadrant_corners,
                min_corner_radius_norm=min_corner_radius_norm,
            )
            candidate_view_coverage = compute_view_center_coverage(
                state["candidates"],
                _image_size(frame),
                grid_cols=view_grid_cols,
                grid_rows=view_grid_rows,
                center_half_width_norm=center_view_half_width_norm,
            )
            candidate_view_failures = view_center_coverage_failures(
                candidate_view_coverage,
                min_view_center_cells=min_view_center_cells,
                min_center_views=min_center_views,
                min_view_quadrant_views=min_view_quadrant_views,
            )
            selected_view_coverage = compute_view_center_coverage(
                selected_candidates,
                _image_size(frame),
                grid_cols=view_grid_cols,
                grid_rows=view_grid_rows,
                center_half_width_norm=center_view_half_width_norm,
            )
            selected_view_failures = view_center_coverage_failures(
                selected_view_coverage,
                min_view_center_cells=min_view_center_cells,
                min_center_views=min_center_views,
                min_view_quadrant_views=min_view_quadrant_views,
            )

            if (
                candidate_failures
                or selected_failures
                or candidate_view_failures
                or selected_view_failures
            ) and not force_save_insufficient_coverage:
                print("[calib_camera] Coverage is not good enough. Calibration not saved.")
                if candidate_failures:
                    print("[calib_camera] Candidate corner coverage missing: " + "; ".join(candidate_failures))
                if selected_failures:
                    print("[calib_camera] Selected corner coverage missing: " + "; ".join(selected_failures))
                if candidate_view_failures:
                    print("[calib_camera] Candidate view-center coverage missing: " + "; ".join(candidate_view_failures))
                if selected_view_failures:
                    print("[calib_camera] Selected view-center coverage missing: " + "; ".join(selected_view_failures))
                print(
                    "[calib_camera] Keep recording a new run and move the board into the missing "
                    "image regions, especially the camera center and all four image quadrants."
                )
                continue

            print(
                "[calib_camera] Selected "
                f"{len(selected_candidates)}/{target_views} views from "
                f"{len(state['candidates'])} candidates."
            )

            print("[calib_camera] Calibrating intrinsics model sweep...")

            selected_images = [cand.image for cand in selected_candidates]
            image_size = _image_size(frame)
            common_stats: dict[str, Any] = {
                "num_candidates_total": len(state["candidates"]),
                "num_candidates_selected": len(selected_candidates),
                "selected_frame_indices": [
                    cand.frame_index for cand in selected_candidates
                ],
                "selected_candidate_scores": [
                    float(cand.score) for cand in selected_candidates
                ],
                "selected_candidate_num_charuco": [
                    int(cand.det.num_charuco) for cand in selected_candidates
                ],
                "selected_candidate_centroids_uv": [
                    [
                        float(cand.metrics.get("centroid_u", np.nan)),
                        float(cand.metrics.get("centroid_v", np.nan)),
                    ]
                    for cand in selected_candidates
                ],
                "selected_candidate_bbox_area_norm": [
                    float(cand.metrics.get("bbox_area_norm", 0.0))
                    for cand in selected_candidates
                ],
                "candidate_coverage_cells": int(candidate_coverage["covered_cells"]),
                "candidate_coverage_total_cells": int(candidate_coverage["total_cells"]),
                "candidate_coverage_grid_counts": candidate_coverage["grid_counts"],
                "candidate_coverage_edge_counts": [
                    int(candidate_coverage["edge_counts"][edge])
                    for edge in ("left", "right", "top", "bottom")
                ],
                "candidate_coverage_max_corner_radius_norm": float(
                    candidate_coverage.get("max_corner_radius_norm", np.nan)
                ),
                "candidate_coverage_p95_corner_radius_norm": float(
                    candidate_coverage.get("p95_corner_radius_norm", np.nan)
                ),
                "selected_coverage_cells": int(selected_coverage["covered_cells"]),
                "selected_coverage_total_cells": int(selected_coverage["total_cells"]),
                "selected_coverage_grid_counts": selected_coverage["grid_counts"],
                "selected_coverage_edge_counts": [
                    int(selected_coverage["edge_counts"][edge])
                    for edge in ("left", "right", "top", "bottom")
                ],
                "selected_coverage_max_corner_radius_norm": float(
                    selected_coverage.get("max_corner_radius_norm", np.nan)
                ),
                "selected_coverage_p95_corner_radius_norm": float(
                    selected_coverage.get("p95_corner_radius_norm", np.nan)
                ),
                "candidate_view_center_cells": int(candidate_view_coverage["covered_cells"]),
                "candidate_view_center_total_cells": int(candidate_view_coverage["total_cells"]),
                "candidate_view_center_grid_counts": candidate_view_coverage["grid_counts"],
                "candidate_view_center_quadrant_counts": candidate_view_coverage["quadrant_counts"],
                "candidate_view_center_edge_counts": [
                    int(candidate_view_coverage["edge_counts"][edge])
                    for edge in ("left", "right", "top", "bottom")
                ],
                "candidate_view_center_count": int(candidate_view_coverage["center_count"]),
                "candidate_view_center_dx_norm_p05_p50_p95": candidate_view_coverage[
                    "dx_norm_p05_p50_p95"
                ],
                "candidate_view_center_dy_norm_p05_p50_p95": candidate_view_coverage[
                    "dy_norm_p05_p50_p95"
                ],
                "candidate_view_center_radius_norm_p05_p50_p95_max": candidate_view_coverage[
                    "radius_norm_p05_p50_p95_max"
                ],
                "selected_view_center_cells": int(selected_view_coverage["covered_cells"]),
                "selected_view_center_total_cells": int(selected_view_coverage["total_cells"]),
                "selected_view_center_grid_counts": selected_view_coverage["grid_counts"],
                "selected_view_center_quadrant_counts": selected_view_coverage["quadrant_counts"],
                "selected_view_center_edge_counts": [
                    int(selected_view_coverage["edge_counts"][edge])
                    for edge in ("left", "right", "top", "bottom")
                ],
                "selected_view_center_count": int(selected_view_coverage["center_count"]),
                "selected_view_center_dx_norm_p05_p50_p95": selected_view_coverage[
                    "dx_norm_p05_p50_p95"
                ],
                "selected_view_center_dy_norm_p05_p50_p95": selected_view_coverage[
                    "dy_norm_p05_p50_p95"
                ],
                "selected_view_center_radius_norm_p05_p50_p95_max": selected_view_coverage[
                    "radius_norm_p05_p50_p95_max"
                ],
                "view_center_grid_cols": int(view_grid_cols),
                "view_center_grid_rows": int(view_grid_rows),
                "view_center_min_cells": int(min_view_center_cells),
                "view_center_min_center_views": int(min_center_views),
                "view_center_min_quadrant_views": int(min_view_quadrant_views),
                "view_center_half_width_norm": float(center_view_half_width_norm),
                "coverage_min_cells": int(min_coverage_cells),
                "coverage_min_edge_corners": int(min_edge_corners),
                "coverage_min_quadrant_corners": int(min_quadrant_corners),
                "coverage_min_corner_radius_norm": float(min_corner_radius_norm),
                "min_capture_corners": int(min_capture_corners),
                "min_edge_capture_corners": int(min_edge_capture_corners),
                "coverage_forced_save": bool(force_save_insufficient_coverage),
            }

            diagnostics_dir: Path | None = None
            model_rows: list[dict[str, Any]] = []
            first_success: tuple[np.ndarray, np.ndarray, float, dict[str, Any], Path] | None = None
            best_success: tuple[np.ndarray, np.ndarray, float, dict[str, Any], Path] | None = None
            primary_success = False

            for spec in calibration_model_specs():
                print(f"[calib_camera] Model {spec.name}: {spec.description}")
                try:
                    K, dist, rms, stats = calibrate_charuco_intrinsics(
                        calib_images=selected_images,
                        board=board,
                        aruco_dict=aruco_dict,
                        detector_params=detector_params,
                        min_charuco_corners=min_edge_capture_corners,
                        flags=spec.flags,
                        dist_coeff_count=spec.dist_coeff_count,
                        intrinsic_refinement_passes=intrinsic_refinement_passes,
                        K_seed=factory_K,
                    )

                    reproj_mean_px, selected_reproj, reproj_stats = reprojection_error_charuco(
                        selected_images,
                        board=board,
                        aruco_dict=aruco_dict,
                        K=K,
                        dist=dist,
                        detector_params=detector_params,
                        min_charuco_corners=min_edge_capture_corners,
                    )
                    selected_reproj_float = [
                        float("nan") if value is None else float(value)
                        for value in selected_reproj
                    ]

                    if diagnostics_dir is None:
                        diagnostics_dir = save_calibration_diagnostics(
                            output_path,
                            candidates=state["candidates"],
                            selected_candidates=selected_candidates,
                            selected_reprojection_errors=selected_reproj,
                            save_selected_images=save_selected_images,
                            grid_cols=coverage_grid_cols,
                            grid_rows=coverage_grid_rows,
                        )

                    radial_stats = radial_plausibility_stats(K, dist, image_size)
                    dist_flat = np.asarray(dist, dtype=np.float64).reshape(-1)
                    stats.update(common_stats)
                    stats.update(
                        {
                            "calibration_model": spec.name,
                            "calibration_model_description": spec.description,
                            "distortion_model": (
                                "opencv_brown_conrady_rational"
                                if spec.name == "rational8"
                                else "opencv_brown_conrady"
                            ),
                            "selected_reprojection_mean_px": float(reproj_mean_px),
                            "selected_reprojection_per_view_px": selected_reproj_float,
                            "diagnostics_dir": "" if diagnostics_dir is None else str(diagnostics_dir),
                            "dist_coeff_abs_max": float(np.max(np.abs(dist_flat)))
                            if dist_flat.size
                            else 0.0,
                            **reproj_stats,
                            **radial_stats,
                        }
                    )
                    stats["model_quality_score"] = calibration_model_quality_score(stats)

                    save_paths: list[Path] = []
                    if spec.name == "standard5":
                        save_paths.append(output_path)
                    variant_path = calibration_variant_path(output_path, spec.name)
                    if variant_path not in save_paths:
                        save_paths.append(variant_path)

                    for save_path in save_paths:
                        save_tracking_calibration_npz(
                            save_path,
                            K=K,
                            dist=np.asarray(dist, dtype=np.float64).reshape(-1, 1),
                            image_size=image_size,
                            rms=float(rms),
                            stats=stats,
                        )

                    primary_path = save_paths[0]
                    success_tuple = (K, dist, float(rms), stats, primary_path)
                    if first_success is None:
                        first_success = success_tuple
                    if best_success is None or float(stats["model_quality_score"]) < float(
                        best_success[3].get("model_quality_score", float("inf"))
                    ):
                        best_success = success_tuple
                    if spec.name == "standard5":
                        primary_success = True
                        state["K"] = np.asarray(K, dtype=np.float64)
                        state["dist"] = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
                        state["rms"] = float(rms)
                        state["stats"] = stats
                        state["diagnostics_dir"] = diagnostics_dir
                        state["saved_path"] = primary_path

                    model_rows.append(
                        {
                            "model": spec.name,
                            "description": spec.description,
                            "status": "ok",
                            "saved_paths": " | ".join(str(path) for path in save_paths),
                            "rms": float(rms),
                            "selected_reprojection_mean_px": float(reproj_mean_px),
                            "dist_coeff_count": int(dist_flat.size),
                            "fx": float(K[0, 0]),
                            "fy": float(K[1, 1]),
                            "cx": float(K[0, 2]),
                            "cy": float(K[1, 2]),
                            "dist": " ".join(f"{x:.12g}" for x in dist_flat),
                            "model_quality_score": float(stats["model_quality_score"]),
                            "corner_reprojection_p95_px": stats.get(
                                "corner_reprojection_p95_px"
                            ),
                            "edge_corner_reprojection_mean_px": stats.get(
                                "edge_corner_reprojection_mean_px"
                            ),
                            "radial_residual_mean_px": stats.get("radial_residual_mean_px"),
                            "radial_residual_abs_mean_px": stats.get(
                                "radial_residual_abs_mean_px"
                            ),
                            **radial_stats,
                        }
                    )
                    print(
                        "[calib_camera]   "
                        f"rms={float(rms):.6f} reproj={float(reproj_mean_px):.6f}px "
                        f"score={float(stats['model_quality_score']):.6f} "
                        f"dist_n={dist_flat.size} radial_turns={radial_stats['radial_turn_count']} "
                        f"positive={radial_stats['radial_positive']}"
                    )
                    print(
                        "[calib_camera]   saved -> "
                        + " | ".join(str(path) for path in save_paths)
                    )
                except Exception as exc:
                    model_rows.append(
                        {
                            "model": spec.name,
                            "description": spec.description,
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(f"[calib_camera]   failed: {type(exc).__name__}: {exc}")

            comparison_csv = model_comparison_path(output_path)
            write_model_comparison_csv(comparison_csv, model_rows)
            print(f"[calib_camera] Saved model comparison -> {comparison_csv}")

            if best_success is not None:
                K, dist, rms, stats, saved_path = best_success
                recommended_stats = dict(stats)
                recommended_stats["recommended_primary_model"] = str(
                    stats.get("calibration_model", "")
                )
                saved_path = save_tracking_calibration_npz(
                    output_path,
                    K=K,
                    dist=np.asarray(dist, dtype=np.float64).reshape(-1, 1),
                    image_size=image_size,
                    rms=float(rms),
                    stats=recommended_stats,
                )
                state["K"] = np.asarray(K, dtype=np.float64)
                state["dist"] = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
                state["rms"] = float(rms)
                state["stats"] = recommended_stats
                state["diagnostics_dir"] = diagnostics_dir
                state["saved_path"] = saved_path
                print(
                    "[calib_camera] Recommended primary model: "
                    f"{recommended_stats.get('recommended_primary_model')} "
                    f"(score={float(recommended_stats.get('model_quality_score', float('nan'))):.6f})"
                )
            elif not primary_success and first_success is not None:
                K, dist, rms, stats, saved_path = first_success
                state["K"] = np.asarray(K, dtype=np.float64)
                state["dist"] = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
                state["rms"] = float(rms)
                state["stats"] = stats
                state["diagnostics_dir"] = diagnostics_dir
                state["saved_path"] = saved_path

            if state["saved_path"] is None:
                print("[calib_camera] Calibration failed for all models.")
                print("[calib_camera] Press SPACE to record a new run or R to reset.")
                continue

            print(f"[calib_camera] Primary RMS: {state['rms']:.6f}")
            print(f"[calib_camera] Primary K:\n{state['K']}")
            print(f"[calib_camera] Primary dist: {state['dist'].reshape(-1).tolist()}")
            print(f"[calib_camera] Primary calibration NPZ -> {state['saved_path']}")
            print(f"[calib_camera] Saved diagnostics -> {state['diagnostics_dir']}")
            print("[calib_camera] Optional: press T for the accuracy test on the primary model.")

    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()

    if state["saved_path"] is None:
        raise RuntimeError("No calibration was saved.")

    return Path(state["saved_path"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the RGB camera for HydraMarker tracking."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Output .npz path. Defaults to hydramarker/calib with a timestamp.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Run the model sweep offline on images from this directory instead of opening the camera.",
    )
    parser.add_argument("--width", type=int, default=REALSENSE_WIDTH)
    parser.add_argument("--height", type=int, default=REALSENSE_HEIGHT)
    parser.add_argument("--fps", type=int, default=REALSENSE_FPS)
    parser.add_argument(
        "--target-views",
        type=int,
        default=N_VIEWS,
        help="Number of diverse frames selected from the recorded candidates.",
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=MIN_CHARUCO_CAPTURE,
        help=(
            "Minimum ChArUco corners required for normal candidate frames. "
            "Extreme edge/corner frames use --min-edge-corners-visible."
        ),
    )
    parser.add_argument(
        "--min-edge-corners-visible",
        type=int,
        default=MIN_CHARUCO_EDGE_CAPTURE,
        help=(
            "Minimum ChArUco corners for edge/corner candidates. "
            "This lets the board cover extreme FOV regions where only a few corners are visible."
        ),
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=AUTO_CAPTURE_INTERVAL_S,
        help="Minimum seconds between automatically stored candidates.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=MAX_CAPTURE_CANDIDATES,
        help="Maximum number of candidates to keep during one recording.",
    )
    parser.add_argument(
        "--charuco-refinement-passes",
        type=int,
        default=CHARUCO_INTRINSIC_REFINEMENT_PASSES,
        help=(
            "Number of extra calibration passes that re-interpolate ChArUco corners "
            "with the current cameraMatrix/distCoeffs before recalibrating. "
            "Use 0 to reproduce the old homography-bootstrap-only behavior."
        ),
    )
    parser.add_argument(
        "--coverage-grid-cols",
        type=int,
        default=SELECTION_GRID_COLS,
        help="Number of image columns used for hard corner coverage checks.",
    )
    parser.add_argument(
        "--coverage-grid-rows",
        type=int,
        default=SELECTION_GRID_ROWS,
        help="Number of image rows used for hard corner coverage checks.",
    )
    parser.add_argument(
        "--min-coverage-cells",
        type=int,
        default=MIN_COVERAGE_CELLS,
        help="Minimum covered corner-grid cells required before saving.",
    )
    parser.add_argument(
        "--min-edge-corners",
        type=int,
        default=MIN_EDGE_CORNERS,
        help="Minimum ChArUco corner observations required near each image edge.",
    )
    parser.add_argument(
        "--min-quadrant-corners",
        type=int,
        default=MIN_QUADRANT_CORNERS,
        help="Minimum ChArUco corner observations required in each image quadrant.",
    )
    parser.add_argument(
        "--min-corner-radius-norm",
        type=float,
        default=MIN_CORNER_RADIUS_NORM,
        help=(
            "Required maximum normalized ChArUco-corner radius. "
            "Use this to force true image-corner/FOV coverage."
        ),
    )
    parser.add_argument(
        "--view-center-grid-cols",
        type=int,
        default=VIEW_CENTER_GRID_COLS,
        help="Number of image columns used for board-center coverage checks.",
    )
    parser.add_argument(
        "--view-center-grid-rows",
        type=int,
        default=VIEW_CENTER_GRID_ROWS,
        help="Number of image rows used for board-center coverage checks.",
    )
    parser.add_argument(
        "--min-view-center-cells",
        type=int,
        default=MIN_VIEW_CENTER_CELLS,
        help="Minimum occupied board-center grid cells required before saving.",
    )
    parser.add_argument(
        "--min-center-views",
        type=int,
        default=MIN_CENTER_VIEWS,
        help="Minimum selected views whose board center lies near the camera center.",
    )
    parser.add_argument(
        "--min-view-quadrant-views",
        type=int,
        default=MIN_VIEW_QUADRANT_VIEWS,
        help="Minimum selected board centers required in each image quadrant.",
    )
    parser.add_argument(
        "--center-view-half-width-norm",
        type=float,
        default=CENTER_VIEW_HALF_WIDTH_NORM,
        help=(
            "Half-width of the accepted central board-center box in normalized "
            "half-FOV coordinates. 0.25 means roughly the central 25% of half-width/height."
        ),
    )
    parser.add_argument(
        "--force-save",
        action="store_true",
        help="Save even when the hard FOV coverage check fails.",
    )
    parser.add_argument(
        "--no-save-selected-images",
        action="store_true",
        help="Do not save the selected calibration images next to the NPZ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.images_dir is not None:
        image_paths = collect_image_paths(args.images_dir)
        print(f"[calib_camera] Offline model sweep from {len(image_paths)} images in {args.images_dir}")
        saved_path = run_model_sweep_for_images(
            calib_images=load_images(image_paths),
            output_path=args.output,
            min_capture_corners=args.min_corners,
            intrinsic_refinement_passes=args.charuco_refinement_passes,
        )
        print(f"[calib_camera] Done: {saved_path}")
        return

    saved_path = run_live_calibration(
        output_path=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        target_views=args.target_views,
        min_capture_corners=args.min_corners,
        min_edge_capture_corners=args.min_edge_corners_visible,
        auto_capture_interval_s=args.capture_interval,
        max_candidates=args.max_candidates,
        save_selected_images=not args.no_save_selected_images,
        coverage_grid_cols=args.coverage_grid_cols,
        coverage_grid_rows=args.coverage_grid_rows,
        min_coverage_cells=args.min_coverage_cells,
        min_edge_corners=args.min_edge_corners,
        min_quadrant_corners=args.min_quadrant_corners,
        min_corner_radius_norm=args.min_corner_radius_norm,
        view_grid_cols=args.view_center_grid_cols,
        view_grid_rows=args.view_center_grid_rows,
        min_view_center_cells=args.min_view_center_cells,
        min_center_views=args.min_center_views,
        min_view_quadrant_views=args.min_view_quadrant_views,
        center_view_half_width_norm=args.center_view_half_width_norm,
        force_save_insufficient_coverage=args.force_save,
        intrinsic_refinement_passes=args.charuco_refinement_passes,
    )
    print(f"[calib_camera] Done: {saved_path}")


if __name__ == "__main__":
    main()
