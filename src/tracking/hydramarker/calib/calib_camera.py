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
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _ensure_tracking_package_on_path() -> None:
    src_root = Path(__file__).resolve().parents[3]
    src = str(src_root)
    if src not in sys.path:
        sys.path.insert(0, src)


_ensure_tracking_package_on_path()

from tracking.hydramarker.camera_setup import create_camera_source
from tracking.hydramarker.config import CameraConfig


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
# A cell counting as "covered" with a single corner allowed captures whose
# lower image half held almost no data (2026-07-05 session: bottom row
# [4,17,16,21,...] corners) — the distortion there is then extrapolated and
# the model choice becomes a per-session lottery. Require a real minimum of
# corners in every grid cell instead.
MIN_CELL_CORNERS = 40
MIN_EDGE_CORNERS = 36
MIN_QUADRANT_CORNERS = 72
MIN_CORNER_RADIUS_NORM = 0.75
VIEW_CENTER_GRID_COLS = 5
VIEW_CENTER_GRID_ROWS = 5
MIN_VIEW_CENTER_CELLS = 9
MIN_CENTER_VIEWS = 30
MIN_VIEW_QUADRANT_VIEWS = 8
CENTER_VIEW_HALF_WIDTH_NORM = 0.25
# Strongly tilted views decorrelate focal length, principal point and the
# radial/tangential coefficients — without them several distortion models fit
# the data equally well but disagree in pose space (measured: ~0.9 mm/100 mm
# z-slope spread between standard5 and no_k3 on the same views).
STRONG_TILT_DEG = 35.0  # Rojtberg & Kuijper (ISMAR'18): ~45 deg optimally
# constrains focal length (0 deg = focal undetermined, 90 deg = principal
# point undetermined); we stay slightly below 45 for reliable board detection.
MIN_STRONG_TILT_VIEWS = 12
# Rolling-shutter guard: the D435i colour sensor reads out line by line, so a
# board that moves during capture stretches vertically and biases fy (observed:
# fx-fy split of ~7.5px and doubled RMS in a continuously swept capture).
# Candidates are therefore only stored while the board is (nearly) still.
MAX_CAPTURE_MOTION_PX = 1.5

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
        params = cv2.aruco.DetectorParameters()
    elif hasattr(cv2.aruco, "DetectorParameters_create"):
        params = cv2.aruco.DetectorParameters_create()
    else:
        raise RuntimeError("No compatible ArUco DetectorParameters API found.")

    # Subpixel-refined marker corners give the ChArUco interpolation better seeds.
    refine_subpix = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", None)
    if refine_subpix is not None:
        try:
            params.cornerRefinementMethod = refine_subpix
        except AttributeError:
            pass

    return params


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
_SUBPIX_MIN_HALF_WIN = 5
_SUBPIX_MAX_HALF_WIN = 15
_SUBPIX_ZERO = (-1, -1)


def _subpix_half_window(points: np.ndarray) -> int:
    """Half window ~ a quarter of the median corner spacing, so the refinement
    sees a meaningful patch of the X-junction at any board distance while never
    reaching the neighbouring corners."""
    if points.shape[0] < 2:
        return _SUBPIX_MIN_HALF_WIN
    diff = points[None, :, :] - points[:, None, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    spacing = float(np.median(np.min(dist, axis=1)))
    if not np.isfinite(spacing):
        return _SUBPIX_MIN_HALF_WIN
    return int(np.clip(round(spacing * 0.25), _SUBPIX_MIN_HALF_WIN, _SUBPIX_MAX_HALF_WIN))


def _filter_corners_inside_image(
    gray: np.ndarray,
    cc: Optional[np.ndarray],
    ci: Optional[np.ndarray],
    margin_px: float = 2.0,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Drop interpolated ChArUco corners that lie outside the image.

    With CharucoParameters.minMarkers=1 the detector extrapolates corners of a
    board that partially leaves the FOV; those points are pure extrapolation
    (never measured) and crash cornerSubPix."""
    if cc is None or ci is None:
        return cc, ci

    h, w = gray.shape[:2]
    pts = cc.reshape(-1, 2)
    keep = (
        (pts[:, 0] >= margin_px)
        & (pts[:, 0] <= float(w - 1) - margin_px)
        & (pts[:, 1] >= margin_px)
        & (pts[:, 1] <= float(h - 1) - margin_px)
    )
    if np.all(keep):
        return cc, ci

    return cc.reshape(-1, 1, 2)[keep], ci.reshape(-1, 1)[keep]


def _refine_charuco_subpix(
    gray: np.ndarray,
    cc: np.ndarray,
) -> np.ndarray:
    if cc is None or cc.size == 0:
        return cc

    h, w = gray.shape[:2]
    pts_all = cc.reshape(-1, 2).astype(np.float64)
    half = _subpix_half_window(pts_all)

    # Only refine corners whose search window fits inside the image; the rest
    # (extreme edge corners) are kept at their interpolated position.
    margin = float(half + 2)
    safe = (
        (pts_all[:, 0] >= margin)
        & (pts_all[:, 0] <= float(w - 1) - margin)
        & (pts_all[:, 1] >= margin)
        & (pts_all[:, 1] <= float(h - 1) - margin)
    )
    if not np.any(safe):
        return cc

    pts = pts_all[safe].reshape(-1, 1, 2).astype(np.float32)
    refined = cv2.cornerSubPix(gray, pts, (half, half), _SUBPIX_ZERO, _SUBPIX_CRITERIA)

    out = pts_all.copy()
    out[safe] = refined.reshape(-1, 2).astype(np.float64)
    return out.reshape(cc.shape).astype(cc.dtype)


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
        cc, ci = _filter_corners_inside_image(gray, cc, ci)
        if cc is None or ci is None or len(ci) == 0:
            return 0, None, None
        cc = _refine_charuco_subpix(gray, cc)
        return int(len(ci)), cc, ci

    if hasattr(cv2.aruco, "CharucoDetector"):
        if hasattr(cv2.aruco, "CharucoParameters"):
            charuco_params = cv2.aruco.CharucoParameters()
            charuco_params.cameraMatrix = K
            charuco_params.distCoeffs = dist
            try:
                # Keep corners that only have a single decoded marker next to
                # them (board leaving the FOV) — those edge observations are
                # the most valuable ones for k1/k2.
                charuco_params.minMarkers = 1
            except AttributeError:
                pass
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

        cc, ci = _filter_corners_inside_image(gray, cc, ci)
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


def _charuco_corner_map(det: CharucoDetection) -> dict[int, np.ndarray]:
    if det.charuco_ids is None or det.charuco_corners is None:
        return {}
    ids = det.charuco_ids.reshape(-1).astype(int)
    pts = det.charuco_corners.reshape(-1, 2).astype(np.float64)
    return {int(i): p for i, p in zip(ids, pts)}


def _median_corner_motion_px(
    prev_map: dict[int, np.ndarray],
    cur_map: dict[int, np.ndarray],
) -> float:
    common = cur_map.keys() & prev_map.keys()
    if len(common) < 4:
        return float("nan")
    return float(np.median([np.linalg.norm(cur_map[i] - prev_map[i]) for i in common]))


def _charuco_tilt_deg(det: CharucoDetection) -> float:
    """Approximate board tilt vs. the image plane from the affine part of the
    object->image mapping (weak perspective: singular-value ratio = cos(tilt))."""
    if det.charuco_ids is None or det.charuco_corners is None or det.num_charuco < 4:
        return float("nan")

    ids = det.charuco_ids.reshape(-1).astype(int)
    cols = ids % (SQUARES_X - 1)
    rows = ids // (SQUARES_X - 1)
    obj = np.c_[cols, rows].astype(np.float64) * SQUARE_LEN_M
    img = det.charuco_corners.reshape(-1, 2).astype(np.float64)

    obj_c = obj - obj.mean(axis=0)
    img_c = img - img.mean(axis=0)
    if np.linalg.matrix_rank(obj_c) < 2:
        return float("nan")

    A, *_ = np.linalg.lstsq(obj_c, img_c, rcond=None)
    s = np.linalg.svd(A, compute_uv=False)
    if s[0] <= 1e-9:
        return float("nan")
    ratio = float(np.clip(s[1] / s[0], 0.0, 1.0))
    return float(np.degrees(np.arccos(ratio)))


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
        "tilt_deg": float("nan"),
    }

    if det.charuco_corners is None or det.num_charuco <= 0:
        return metrics

    metrics["tilt_deg"] = _charuco_tilt_deg(det)

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
    tilt_deg = float(np.nan_to_num(metrics.get("tilt_deg", 0.0), nan=0.0))
    tilt_score = float(np.clip(tilt_deg / 45.0, 0.0, 1.0))

    return (
        0.35 * corner_score
        + 0.18 * area_score
        + 0.22 * sharpness_score
        + 0.15 * radius_score
        + 0.10 * edge_score
        + 0.15 * tilt_score
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


def _candidate_tilt_bin(metrics: Dict[str, float]) -> int:
    tilt = float(np.nan_to_num(metrics.get("tilt_deg", 0.0), nan=0.0))
    return int(np.clip(tilt // 15.0, 0, 3))


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


def _corner_grid_cell_counts(
    points: np.ndarray,
    image_size: tuple[int, int],
    *,
    grid_cols: int,
    grid_rows: int,
) -> np.ndarray:
    counts = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    if points.size:
        width, height = image_size
        cols = np.clip(
            np.floor(points[:, 0] / max(width, 1) * grid_cols), 0, grid_cols - 1
        ).astype(int)
        rows = np.clip(
            np.floor(points[:, 1] / max(height, 1) * grid_rows), 0, grid_rows - 1
        ).astype(int)
        np.add.at(counts, (rows, cols), 1)
    return counts


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
    strong_tilt_count = 0

    for cand in candidates:
        metrics = cand.metrics
        tilt = float(np.nan_to_num(metrics.get("tilt_deg", 0.0), nan=0.0))
        if tilt >= STRONG_TILT_DEG:
            strong_tilt_count += 1
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
        "strong_tilt_views": int(strong_tilt_count),
        "strong_tilt_min_deg": float(STRONG_TILT_DEG),
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
    min_cell_corners: int = MIN_CELL_CORNERS,
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

    grid_counts = np.asarray(coverage.get("grid_counts", []))
    if grid_counts.ndim == 2 and grid_counts.size and int(min_cell_corners) > 0:
        thin = grid_counts < int(min_cell_corners)
        if np.any(thin):
            worst_row, worst_col = np.unravel_index(
                int(np.argmin(grid_counts)), grid_counts.shape
            )
            failures.append(
                f"{int(np.count_nonzero(thin))} cells < {int(min_cell_corners)} corners "
                f"(worst r{worst_row + 1}c{worst_col + 1}="
                f"{int(grid_counts[worst_row, worst_col])})"
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
    min_strong_tilt_views: int = MIN_STRONG_TILT_VIEWS,
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

    if int(coverage.get("strong_tilt_views", 0)) < int(min_strong_tilt_views):
        failures.append(
            f"tilted views (>= {STRONG_TILT_DEG:.0f} deg) "
            f"{coverage.get('strong_tilt_views', 0)}/{min_strong_tilt_views}"
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
    min_cell_corners: int = MIN_CELL_CORNERS,
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
    selected_cell_counts = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    selected_view_cells: set[tuple[int, int]] = set()
    selected_center_views = 0
    selected_radius_bins: set[int] = set()
    selected_size_bins: set[int] = set()
    selected_tilt_bins: set[int] = set()
    selected_strong_tilt_views = 0
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

            # feed cells still below the per-cell corner minimum, so the final
            # 80-view selection cannot starve a region the operator did cover
            if pts.size and int(min_cell_corners) > 0:
                cand_cell_counts = _corner_grid_cell_counts(
                    pts,
                    image_size,
                    grid_cols=grid_cols,
                    grid_rows=grid_rows,
                )
                hungry = selected_cell_counts < int(min_cell_corners)
                hungry_hits = int(cand_cell_counts[hungry].sum())
                if hungry_hits > 0:
                    diversity_bonus += 0.60 * min(1.0, hungry_hits / 30.0)

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

            tilt_bin = _candidate_tilt_bin(metrics)
            tilt_deg = float(np.nan_to_num(metrics.get("tilt_deg", 0.0), nan=0.0))
            if tilt_bin not in selected_tilt_bins:
                diversity_bonus += 0.18
            if selected_strong_tilt_views < MIN_STRONG_TILT_VIEWS and tilt_deg >= STRONG_TILT_DEG:
                diversity_bonus += 0.85

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
        selected_cell_counts += _corner_grid_cell_counts(
            pts,
            image_size,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
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
        selected_tilt_bins.add(_candidate_tilt_bin(chosen.metrics))
        if float(np.nan_to_num(chosen.metrics.get("tilt_deg", 0.0), nan=0.0)) >= STRONG_TILT_DEG:
            selected_strong_tilt_views += 1
        covered_ids.update(_charuco_id_set(chosen.det))

    _feed_hungry_cells(
        selected,
        remaining,
        selected_cell_counts,
        image_size=image_size,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        min_cell_corners=min_cell_corners,
        max_extra=target_views,
    )

    return sorted(selected, key=lambda c: c.frame_index)


def _feed_hungry_cells(
    selected: list[CalibrationCandidate],
    remaining: list[CalibrationCandidate],
    selected_cell_counts: np.ndarray,
    *,
    image_size: tuple[int, int],
    grid_cols: int,
    grid_rows: int,
    min_cell_corners: int,
    max_extra: int,
) -> None:
    """Guarantee phase for the per-cell corner minimum: any greedy selection
    optimizes several goals at once and may stop at target_views with
    individual cells still underfed although the candidate pool has plenty of
    corners there (observed live 2026-07-05: candidates 35/35 cells >= 40 but
    the selected 80 views left r1c1 at 13). Appends the candidates that feed
    the most still-missing corners into hungry cells; mutates all three
    leading arguments in place."""
    if int(min_cell_corners) <= 0:
        return
    extra = 0
    while remaining and extra < max_extra:
        hungry = selected_cell_counts < int(min_cell_corners)
        if not np.any(hungry):
            break
        best_i = -1
        best_hits = 0
        for i, cand in enumerate(remaining):
            pts = _candidate_corner_points(cand)
            if not pts.size:
                continue
            cnts = _corner_grid_cell_counts(
                pts,
                image_size,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
            )
            hits = int(np.minimum(cnts, np.maximum(
                int(min_cell_corners) - selected_cell_counts, 0
            ))[hungry].sum())
            if hits > best_hits:
                best_hits = hits
                best_i = i
        if best_i < 0 or best_hits == 0:
            break
        chosen = remaining.pop(best_i)
        selected.append(chosen)
        selected_cell_counts += _corner_grid_cell_counts(
            _candidate_corner_points(chosen),
            image_size,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
        )
        extra += 1


def _candidate_information_matrix(
    cand: CalibrationCandidate,
    board: Any,
    K: np.ndarray,
    dist: np.ndarray,
) -> Optional[np.ndarray]:
    """Intrinsics information contribution of one view: J_C^T J_C with the
    per-view pose marginalized out via the Schur complement. Uses the
    analytic Jacobian from cv2.projectPoints (columns: rvec 3, t 3, fx fy,
    cx cy, dist n)."""
    det = cand.det
    if det.charuco_ids is None or det.charuco_corners is None:
        return None
    if det.num_charuco < 6:
        return None
    obj = _charuco_object_points(board, det.charuco_ids).astype(np.float64)
    img = det.charuco_corners.reshape(-1, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj.reshape(-1, 1, 3),
        img.reshape(-1, 1, 2),
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    proj, J = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    if float(np.mean(err)) > 3.0:
        return None
    J = np.asarray(J, dtype=np.float64)
    n_intr = 4 + int(np.asarray(dist).reshape(-1).size)
    if J.shape[1] < 6 + n_intr:
        return None
    Jp = J[:, :6]
    Jc = J[:, 6:6 + n_intr]
    Mpp = Jp.T @ Jp
    Mpc = Jp.T @ Jc
    Mcc = Jc.T @ Jc
    return Mcc - Mpc.T @ np.linalg.pinv(Mpp) @ Mpc


def select_calibration_candidates_information(
    candidates: Sequence[CalibrationCandidate],
    *,
    target_views: int,
    min_charuco_corners: int,
    image_size: tuple[int, int],
    board: Any,
    min_edge_charuco_corners: int = MIN_CHARUCO_EDGE_CAPTURE,
    grid_cols: int = SELECTION_GRID_COLS,
    grid_rows: int = SELECTION_GRID_ROWS,
    edge_fraction: float = COVERAGE_EDGE_FRACTION,
    min_cell_corners: int = MIN_CELL_CORNERS,
    **heuristic_kwargs: Any,
) -> list[CalibrationCandidate]:
    """Information-driven (D-optimal) candidate selection.

    Following Rojtberg & Kuijper (ISMAR'18) / optimal experiment design: a
    view is worth selecting iff it grows the information volume
    logdet(M + dM) of the intrinsics. This subsumes the hand-tuned diversity
    bonuses — near-duplicate views contribute ~zero new information, tilted
    views are picked automatically because they decorrelate focal length and
    principal point, and edge corners weigh more because the distortion
    Jacobian is largest there. The per-cell coverage guarantee stays on top
    (it is model-agnostic; the Jacobian is conditional on the seed model).
    Falls back to the heuristic selection if seeding or Jacobians fail.
    """
    def _fallback() -> list[CalibrationCandidate]:
        return select_calibration_candidates(
            candidates,
            target_views=target_views,
            min_charuco_corners=min_charuco_corners,
            min_edge_charuco_corners=min_edge_charuco_corners,
            image_size=image_size,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            edge_fraction=edge_fraction,
            min_cell_corners=min_cell_corners,
            **heuristic_kwargs,
        )

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
                    c.metrics, image_size, edge_fraction=edge_fraction
                )
            )
        )
    ]
    if len(valid) <= target_views:
        return sorted(valid, key=lambda c: c.frame_index)

    # Seed intrinsics for the Jacobians: quick standard5 calibration on a
    # small heuristically diverse subset. The selection is conditional on
    # this rough model, which is fine — information geometry changes little
    # with the seed, and the final calibration refits everything.
    seed_views = select_calibration_candidates(
        valid,
        target_views=12,
        min_charuco_corners=min_charuco_corners,
        min_edge_charuco_corners=min_edge_charuco_corners,
        image_size=image_size,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        edge_fraction=edge_fraction,
        min_cell_corners=0,
        **heuristic_kwargs,
    )
    try:
        _, K0, dist0 = _calibrate_charuco_compat(
            all_charuco_corners=[c.det.charuco_corners for c in seed_views],
            all_charuco_ids=[c.det.charuco_ids for c in seed_views],
            board=board,
            image_size=image_size,
            K_init=np.eye(3, dtype=np.float64),
            dist_init=np.zeros((5, 1), dtype=np.float64),
            flags=0,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6),
        )
    except Exception:
        print("[calib_camera] Information selection: seed calibration failed, "
              "using heuristic selection.")
        return _fallback()

    infos: list[tuple[CalibrationCandidate, np.ndarray]] = []
    for cand in valid:
        dM = _candidate_information_matrix(cand, board, K0, dist0)
        if dM is not None and np.all(np.isfinite(dM)):
            infos.append((cand, dM))
    if len(infos) < target_views:
        print("[calib_camera] Information selection: too few usable "
              f"Jacobians ({len(infos)}), using heuristic selection.")
        return _fallback()

    dim = infos[0][1].shape[0]
    mean_diag = np.mean([np.diag(dM) for _, dM in infos], axis=0)
    M = np.diag(np.maximum(mean_diag, 1e-12)) * 1e-6

    selected: list[CalibrationCandidate] = []
    remaining_info = list(infos)
    for _ in range(min(target_views, len(remaining_info))):
        best_i = -1
        best_gain = -np.inf
        for i, (_, dM) in enumerate(remaining_info):
            sign, logdet = np.linalg.slogdet(M + dM)
            if sign > 0 and logdet > best_gain:
                best_gain = logdet
                best_i = i
        if best_i < 0:
            break
        cand, dM = remaining_info.pop(best_i)
        selected.append(cand)
        M = M + dM
    if len(selected) < min(target_views, 8):
        print("[calib_camera] Information selection: greedy selection "
              "degenerated, using heuristic selection.")
        return _fallback()

    selected_cell_counts = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    for cand in selected:
        selected_cell_counts += _corner_grid_cell_counts(
            _candidate_corner_points(cand),
            image_size,
            grid_cols=grid_cols,
            grid_rows=grid_rows,
        )
    selected_ids = {id(c) for c in selected}
    remaining = [c for c in valid if id(c) not in selected_ids]

    # The information greedy concentrates on informative poses and knows
    # nothing about the hard coverage gates that are later checked on the
    # SELECTED set (observed live: pool center=154 but only 17 center views
    # selected; 7 cells left hungry with a capped top-up). Top up every hard
    # gate explicitly, using the information gain as tie-breaker.
    _feed_hungry_cells(
        selected,
        remaining,
        selected_cell_counts,
        image_size=image_size,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        min_cell_corners=min_cell_corners,
        max_extra=target_views,
    )

    info_by_id = {id(c): dM for c, dM in infos}

    def info_gain(cand: CalibrationCandidate) -> float:
        dM = info_by_id.get(id(cand))
        if dM is None:
            return -np.inf
        sign, logdet = np.linalg.slogdet(M + dM)
        return logdet if sign > 0 else -np.inf

    def remove_by_identity(pool: list[CalibrationCandidate], item: CalibrationCandidate) -> None:
        # list.remove() compares via the dataclass __eq__, which chokes on the
        # numpy array fields ("truth value of an array is ambiguous")
        for i, c in enumerate(pool):
            if c is item:
                pool.pop(i)
                return

    def top_up(predicate, needed: int) -> None:
        nonlocal M
        added = 0
        while added < needed:
            pool = [c for c in remaining if predicate(c)]
            if not pool:
                break
            best = max(pool, key=info_gain)
            remove_by_identity(remaining, best)
            selected.append(best)
            dM = info_by_id.get(id(best))
            if dM is not None:
                M = M + dM
            added += 1

    view_grid_cols = int(heuristic_kwargs.get("view_grid_cols", VIEW_CENTER_GRID_COLS))
    view_grid_rows = int(heuristic_kwargs.get("view_grid_rows", VIEW_CENTER_GRID_ROWS))
    min_center_views = int(heuristic_kwargs.get("min_center_views", MIN_CENTER_VIEWS))
    min_view_quadrant_views = int(
        heuristic_kwargs.get("min_view_quadrant_views", MIN_VIEW_QUADRANT_VIEWS)
    )
    min_view_center_cells = int(
        heuristic_kwargs.get("min_view_center_cells", MIN_VIEW_CENTER_CELLS)
    )
    center_half = float(
        heuristic_kwargs.get("center_view_half_width_norm", CENTER_VIEW_HALF_WIDTH_NORM)
    )

    def is_center(c: CalibrationCandidate) -> bool:
        return _is_center_view(c.metrics, center_half_width_norm=center_half)

    n_center = sum(1 for c in selected if is_center(c))
    top_up(is_center, max(0, min_center_views - n_center))

    def is_strong_tilt(c: CalibrationCandidate) -> bool:
        return float(
            np.nan_to_num(c.metrics.get("tilt_deg", 0.0), nan=0.0)
        ) >= STRONG_TILT_DEG

    n_tilt = sum(1 for c in selected if is_strong_tilt(c))
    top_up(is_strong_tilt, max(0, MIN_STRONG_TILT_VIEWS - n_tilt))

    def view_quadrant(c: CalibrationCandidate) -> tuple[int, int]:
        u = float(np.nan_to_num(c.metrics.get("centroid_u_norm", 0.5), nan=0.5))
        v = float(np.nan_to_num(c.metrics.get("centroid_v_norm", 0.5), nan=0.5))
        return (int(v >= 0.5), int(u >= 0.5))

    for q_row in range(2):
        for q_col in range(2):
            n_q = sum(1 for c in selected if view_quadrant(c) == (q_row, q_col))
            top_up(
                lambda c, rc=(q_row, q_col): view_quadrant(c) == rc,
                max(0, min_view_quadrant_views - n_q),
            )

    def view_cell(c: CalibrationCandidate) -> tuple[int, int]:
        return _candidate_view_grid_cell(
            c.metrics, grid_cols=view_grid_cols, grid_rows=view_grid_rows
        )

    seen_cells = {view_cell(c) for c in selected}
    while len(seen_cells) < min_view_center_cells:
        pool = [c for c in remaining if view_cell(c) not in seen_cells]
        if not pool:
            break
        best = max(pool, key=info_gain)
        remove_by_identity(remaining, best)
        selected.append(best)
        seen_cells.add(view_cell(best))

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
    dist_seed: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """dist_seed (only used together with K_seed): warm-start distortion, e.g.
    the compact-model solution when fitting the rational model. Starting LM
    from k4..k6=0 near a good compact fit avoids the degenerate rational
    solutions with huge near-cancelling numerator/denominator coefficients."""
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
            n_coeffs = max(4, int(dist_coeff_count))
            if dist_seed is not None:
                seed = np.asarray(dist_seed, dtype=np.float64).reshape(-1)
                n_coeffs = max(n_coeffs, seed.size)
                dist_init = np.zeros((n_coeffs, 1), dtype=np.float64)
                dist_init[: seed.size, 0] = seed
            else:
                dist_init = np.zeros((n_coeffs, 1), dtype=np.float64)
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


def intrinsics_sanity_warnings(
    K: np.ndarray,
    factory_K: Optional[np.ndarray] = None,
) -> list[str]:
    """Plausibility checks that catch bad capture sessions early.

    |fx-fy| beyond ~2px is non-physical for square pixels and is the
    rolling-shutter/motion-blur fingerprint; large deviations from the
    per-unit factory intrinsics point the same way."""
    warnings: list[str] = []
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    if abs(fx - fy) > 2.0:
        warnings.append(
            f"fx-fy = {fx - fy:+.1f} px (expected ~0): motion blur / rolling "
            "shutter during capture suspected - redo with stop-and-hold poses"
        )

    if factory_K is not None:
        ffx, ffy = float(factory_K[0, 0]), float(factory_K[1, 1])
        fcx, fcy = float(factory_K[0, 2]), float(factory_K[1, 2])
        if abs(fx - ffx) > 10.0 or abs(fy - ffy) > 10.0:
            warnings.append(
                f"focal length deviates from factory by ({fx - ffx:+.1f}, "
                f"{fy - ffy:+.1f}) px (>10)"
            )
        if abs(cx - fcx) > 15.0 or abs(cy - fcy) > 15.0:
            warnings.append(
                f"principal point deviates from factory by ({cx - fcx:+.1f}, "
                f"{cy - fcy:+.1f}) px (>15)"
            )

    return warnings


def holdout_model_stats(
    calib_images: Sequence[np.ndarray],
    board: Any,
    aruco_dict: Any,
    detector_params: Optional[Any],
    *,
    min_charuco_corners: int,
    flags: int,
    dist_coeff_count: int,
    K_seed: Optional[np.ndarray] = None,
    dist_seed: Optional[np.ndarray] = None,
    n_folds: int = 3,
) -> Dict[str, Any]:
    """K-fold hold-out validation for one calibration model.

    The in-sample reprojection error rewards the most flexible model, which is
    exactly the failure mode we measured (rational8 fitting near-cancelling
    k2/k6). Calibrating on train folds and scoring corners on held-out images
    punishes that honestly; the fx/fy/cx/cy spread across folds exposes
    parameter instability.
    """
    n_folds = int(n_folds)
    images = list(calib_images)
    if n_folds < 2 or len(images) < 2 * n_folds:
        return {}

    # Contiguous blocks, not interleaved: capture frames are temporally ordered
    # and neighbouring frames are near-duplicates, so an i%n split leaks the
    # test views into training and lets overfitted models look like they
    # generalize (observed: rational8 scored the best interleaved hold-out).
    fold_bounds = np.linspace(0, len(images), n_folds + 1).astype(int)

    fold_err: list[float] = []
    fold_params: list[list[float]] = []
    for fold in range(n_folds):
        lo, hi = int(fold_bounds[fold]), int(fold_bounds[fold + 1])
        train = images[:lo] + images[hi:]
        test = images[lo:hi]
        if not test or len(train) < 3:
            continue
        try:
            K_f, dist_f, _, _ = calibrate_charuco_intrinsics(
                train,
                board,
                aruco_dict,
                detector_params,
                min_charuco_corners=min_charuco_corners,
                flags=flags,
                dist_coeff_count=dist_coeff_count,
                intrinsic_refinement_passes=0,
                K_seed=K_seed,
                dist_seed=dist_seed,
            )
        except Exception:
            continue

        _, _, reproj_stats = reprojection_error_charuco(
            test,
            board,
            aruco_dict,
            K_f,
            dist_f,
            detector_params,
            min_charuco_corners=min_charuco_corners,
        )
        err = float(reproj_stats.get("corner_reprojection_mean_px", float("nan")))
        if np.isfinite(err):
            fold_err.append(err)
        fold_params.append(
            [float(K_f[0, 0]), float(K_f[1, 1]), float(K_f[0, 2]), float(K_f[1, 2])]
        )

    if not fold_err or len(fold_params) < 2:
        return {}

    params_arr = np.asarray(fold_params, dtype=np.float64)
    spread_px = float(np.max(np.ptp(params_arr, axis=0)))
    return {
        "holdout_folds": int(len(fold_err)),
        "holdout_corner_reprojection_mean_px": float(np.mean(fold_err)),
        "holdout_corner_reprojection_worst_px": float(np.max(fold_err)),
        "holdout_intrinsics_spread_px": spread_px,
    }


def detect_charuco_all(
    calib_images: Sequence[np.ndarray],
    board: Any,
    aruco_dict: Any,
    detector_params: Optional[Any],
    *,
    min_charuco_corners: int,
) -> list[CharucoDetection]:
    """Detect once, reuse across the model sweep — detection dominates the
    spatial-holdout cost and is model-independent."""
    dets: list[CharucoDetection] = []
    for img in calib_images:
        det = detect_charuco(img, board, aruco_dict, detector_params)
        if (
            det.charuco_corners is None
            or det.charuco_ids is None
            or det.num_charuco < min_charuco_corners
        ):
            continue
        dets.append(det)
    return dets


def spatial_holdout_model_stats(
    calib_images: Sequence[np.ndarray],
    board: Any,
    aruco_dict: Any,
    detector_params: Optional[Any],
    *,
    min_charuco_corners: int,
    flags: int,
    dist_coeff_count: int,
    K_seed: Optional[np.ndarray] = None,
    dist_seed: Optional[np.ndarray] = None,
    region_cols: int = 3,
    region_rows: int = 3,
    min_train_corners_per_view: int = 8,
    min_test_corners: int = 40,
    detections: Optional[Sequence[CharucoDetection]] = None,
    max_views: int = 40,
) -> Dict[str, Any]:
    """Leave-REGION-out validation: refit without the corners of one image
    region, then score the reprojection error exactly there.

    The temporal k-fold hold-out shares the spatial coverage between train and
    test folds, so a flexible model that extrapolates wildly into thinly
    covered image regions still scores well (measured 2026-07-05: rational8
    had the best temporal hold-out AND a -1.8 mm/100mm pose slope caused in
    the starved lower image half). This metric measures extrapolation
    quality directly and is the honest overfitting detector for the model
    sweep.
    """
    if not calib_images:
        return {}
    image_size = _image_size(calib_images[0])
    width, height = int(image_size[0]), int(image_size[1])

    if detections is not None:
        dets = list(detections)
    else:
        dets = detect_charuco_all(
            calib_images,
            board,
            aruco_dict,
            detector_params,
            min_charuco_corners=min_charuco_corners,
        )
    if len(dets) < 3:
        return {}
    # evenly subsample: 9 region refits on all 80 views cost minutes per
    # model; ~40 views give the same regional error signal
    if int(max_views) > 0 and len(dets) > int(max_views):
        idx = np.linspace(0, len(dets) - 1, int(max_views)).astype(int)
        dets = [dets[i] for i in idx]

    if K_seed is not None:
        K_init = np.asarray(K_seed, dtype=np.float64).reshape(3, 3)
        n_coeffs = max(4, int(dist_coeff_count))
        if dist_seed is not None:
            seed = np.asarray(dist_seed, dtype=np.float64).reshape(-1)
            n_coeffs = max(n_coeffs, seed.size)
            dist_init = np.zeros((n_coeffs, 1), dtype=np.float64)
            dist_init[: seed.size, 0] = seed
        else:
            dist_init = np.zeros((n_coeffs, 1), dtype=np.float64)
        effective_flags = int(flags) | int(getattr(cv2, "CALIB_USE_INTRINSIC_GUESS", 0))
    else:
        K_init = np.eye(3, dtype=np.float64)
        dist_init = np.zeros((max(4, int(dist_coeff_count)), 1), dtype=np.float64)
        effective_flags = int(flags)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6)
    region_err = np.full((region_rows, region_cols), np.nan)

    for r_row in range(region_rows):
        for r_col in range(region_cols):
            x0 = r_col * width / region_cols
            x1 = (r_col + 1) * width / region_cols
            y0 = r_row * height / region_rows
            y1 = (r_row + 1) * height / region_rows

            train_corners: list[np.ndarray] = []
            train_ids: list[np.ndarray] = []
            test_views: list[tuple[int, np.ndarray, np.ndarray]] = []
            n_test = 0
            for det in dets:
                pts = det.charuco_corners.reshape(-1, 2)
                ids = det.charuco_ids.reshape(-1)
                inside = (
                    (pts[:, 0] >= x0)
                    & (pts[:, 0] < x1)
                    & (pts[:, 1] >= y0)
                    & (pts[:, 1] < y1)
                )
                keep = ~inside
                if int(keep.sum()) < int(min_train_corners_per_view):
                    continue
                view_index = len(train_corners)
                train_corners.append(
                    np.ascontiguousarray(pts[keep].reshape(-1, 1, 2), dtype=np.float32)
                )
                train_ids.append(
                    np.ascontiguousarray(ids[keep].reshape(-1, 1), dtype=np.int32)
                )
                if int(inside.sum()) > 0:
                    test_views.append((view_index, pts[inside], ids[inside]))
                    n_test += int(inside.sum())

            if len(train_corners) < 3 or n_test < int(min_test_corners):
                continue

            try:
                _, K_f, dist_f = _calibrate_charuco_compat(
                    all_charuco_corners=train_corners,
                    all_charuco_ids=train_ids,
                    board=board,
                    image_size=image_size,
                    K_init=K_init.copy(),
                    dist_init=dist_init.copy(),
                    flags=effective_flags,
                    criteria=criteria,
                )
            except Exception:
                continue

            errs: list[float] = []
            for view_index, test_pts, test_ids in test_views:
                obj_train = _charuco_object_points(board, train_ids[view_index])
                img_train = train_corners[view_index].reshape(-1, 1, 2)
                ok, rvec, tvec = cv2.solvePnP(
                    obj_train.reshape(-1, 1, 3),
                    img_train.astype(np.float64),
                    K_f,
                    dist_f,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not ok:
                    continue
                obj_test = _charuco_object_points(
                    board, test_ids.reshape(-1, 1).astype(np.int32)
                )
                proj, _ = cv2.projectPoints(
                    obj_test.reshape(-1, 1, 3), rvec, tvec, K_f, dist_f
                )
                errs.extend(
                    np.linalg.norm(proj.reshape(-1, 2) - test_pts, axis=1).tolist()
                )
            if errs:
                region_err[r_row, r_col] = float(np.mean(errs))

    finite = region_err[np.isfinite(region_err)]
    if finite.size == 0:
        return {}
    worst_idx = np.unravel_index(int(np.nanargmax(region_err)), region_err.shape)
    return {
        "spatial_holdout_regions": int(finite.size),
        "spatial_holdout_mean_px": float(np.mean(finite)),
        "spatial_holdout_worst_px": float(np.max(finite)),
        "spatial_holdout_worst_region": f"r{worst_idx[0] + 1}c{worst_idx[1] + 1}",
        "spatial_holdout_region_err_px": region_err,
    }


# Production model set: no_k3 and the no_tangent variants were diagnostic
# models from the 2026-07-02 distortion analysis and only slow the sweep down.
DEFAULT_SWEEP_MODELS = "standard5,rational6,rational8"


def calibration_model_specs(
    model_names: Optional[str] = None,
) -> list[CalibrationModelSpec]:
    specs = _all_calibration_model_specs()
    if not model_names:
        return specs
    wanted = [name.strip() for name in str(model_names).split(",") if name.strip()]
    unknown = [n for n in wanted if n not in {s.name for s in specs}]
    if unknown:
        raise ValueError(
            f"Unknown calibration model(s) {unknown}; "
            f"available: {[s.name for s in specs]}"
        )
    return [s for s in specs if s.name in wanted]


def _all_calibration_model_specs() -> list[CalibrationModelSpec]:
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
            name="standard5_no_tangent",
            flags=int(getattr(cv2, "CALIB_ZERO_TANGENT_DIST", 0)),
            dist_coeff_count=5,
            description="OpenCV k1,k2,k3 with tangential distortion zeroed",
        ),
        CalibrationModelSpec(
            name="no_k3_no_tangent",
            flags=(
                int(getattr(cv2, "CALIB_FIX_K3", 0))
                | int(getattr(cv2, "CALIB_ZERO_TANGENT_DIST", 0))
            ),
            dist_coeff_count=5,
            description="OpenCV k1,k2 only (k3 fixed, tangential zeroed)",
        ),
        CalibrationModelSpec(
            name="rational6",
            flags=(
                int(getattr(cv2, "CALIB_RATIONAL_MODEL", 0))
                | int(getattr(cv2, "CALIB_FIX_K5", 0))
                | int(getattr(cv2, "CALIB_FIX_K6", 0))
            ),
            dist_coeff_count=8,
            description="OpenCV rational model k1,k2,p1,p2,k3,k4 (k5,k6 fixed)",
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

    # Hold-out terms dominate the in-sample ones when available: generalization
    # to unseen views is what predicts pose stability, in-sample RMS does not.
    holdout_px = value("holdout_corner_reprojection_mean_px", float("nan"))
    if np.isfinite(holdout_px):
        score += 2.0 * holdout_px
        score += 0.05 * value("holdout_intrinsics_spread_px", 0.0)

    # Spatial (leave-region-out) hold-out weighs heaviest: it is the only
    # metric here that measures extrapolation into image regions, which is
    # where flexible models silently fail (pose-space z-slope) while every
    # in-sample and temporal-holdout pixel metric rewards them.
    spatial_mean_px = value("spatial_holdout_mean_px", float("nan"))
    if np.isfinite(spatial_mean_px):
        score += 2.0 * spatial_mean_px
        score += 1.0 * value("spatial_holdout_worst_px", 0.0)

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


def _draw_coverage_grid(
    vis: np.ndarray,
    coverage: Optional[Dict[str, Any]],
    min_cell_corners: int = MIN_CELL_CORNERS,
) -> np.ndarray:
    """Overlay the corner-coverage grid on the live view. Cells are graded by
    corner COUNT, not mere touch: empty cells are tinted red, cells below
    min_cell_corners get an orange border plus their current count so the
    operator keeps the board there until the cell is actually fed, and only
    cells at/above the minimum turn green."""
    if not coverage:
        return vis
    counts = np.asarray(coverage.get("grid_counts", []))
    if counts.ndim != 2 or counts.size == 0:
        return vis

    rows, cols = counts.shape
    h, w = vis.shape[:2]
    red_fill = vis.copy()
    for row in range(rows):
        for col in range(cols):
            x0 = int(round(col * w / cols))
            x1 = int(round((col + 1) * w / cols))
            y0 = int(round(row * h / rows))
            y1 = int(round((row + 1) * h / rows))
            count = int(counts[row, col])
            if count >= int(min_cell_corners):
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 170, 0), 1)
            elif count > 0:
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 150, 255), 2)
                cv2.putText(
                    vis,
                    f"{count}/{int(min_cell_corners)}",
                    (x0 + 6, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 150, 255),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.rectangle(red_fill, (x0, y0), (x1, y1), (0, 0, 220), -1)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 220), 2)

    return cv2.addWeighted(red_fill, 0.18, vis, 0.82, 0)


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
    sharpness: Optional[float] = None,
    motion_px: Optional[float] = None,
    min_cell_corners: int = MIN_CELL_CORNERS,
) -> np.ndarray:
    vis = frame_bgr.copy()

    if recording:
        vis = _draw_coverage_grid(vis, coverage, min_cell_corners=min_cell_corners)

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

    sharp_text = (
        "-" if sharpness is None or not np.isfinite(sharpness) else f"{sharpness:.0f}"
    )
    motion_text = (
        "-" if motion_px is None or not np.isfinite(motion_px) else f"{motion_px:.2f}"
    )
    moving = (
        motion_px is not None
        and np.isfinite(motion_px)
        and motion_px > MAX_CAPTURE_MOTION_PX
    )
    quality_line = (
        f"Sharpness: {sharp_text}  Motion: {motion_text} px/frame "
        f"(max {MAX_CAPTURE_MOTION_PX:.1f})"
    )
    if recording and moving:
        quality_line += "  -> HOLD STILL, not capturing"

    lines = [
        status,
        f"ArUco: {aruco}/{MAX_ARUCO}  ChArUco: {charuco}/{MAX_CHARUCO_CORNERS}",
        f"Candidates: {num_candidates}  Selected: {num_selected}/{target_views}",
        quality_line,
        _coverage_short_text(coverage),
        _view_coverage_short_text(view_coverage),
        "Keys: SPACE start/stop | T accuracy | R redo | Q/ESC quit",
    ]

    if recording and coverage is not None and (not coverage_ok or not view_coverage_ok):
        missing = coverage_failures(coverage, min_cell_corners=min_cell_corners)
        if view_coverage is not None:
            missing += view_center_coverage_failures(view_coverage)
        if missing:
            lines.append("Need: " + "; ".join(missing[:4]))

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
    width = next(
        (int(np.asarray(npz[key]).reshape(-1)[0]) for key in width_keys if key in npz),
        None,
    )
    height = next(
        (int(np.asarray(npz[key]).reshape(-1)[0]) for key in height_keys if key in npz),
        None,
    )
    if width is not None and height is not None:
        return [width, height]

    return None


def load_tracking_calibration_npz(
    path: Path | str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the OpenCV camera model used by HydraMarker tracking."""
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
        info: dict[str, Any] = {
            key: _npz_scalar_or_list(npz[key])
            for key in (
                "camera_source",
                "camera_backend",
                "camera_serial",
                "camera_model",
                "pixel_format",
                "distortion_model",
                "created_at",
                "calibration_model",
                "recommended_primary_model",
            )
            if key in npz
        }

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    if dist.size == 0:
        raise ValueError("Distortion coefficients must not be empty.")

    info.update(
        {
            "camera_source": str(info.get("camera_source", "opencv_calibration_npz")),
            "camera_calibration_path": str(path),
            "distortion_model": str(
                info.get("distortion_model", "opencv_brown_conrady")
            ),
            "K": K.tolist(),
            "opencv_dist_coeffs": dist.reshape(-1).tolist(),
            "effective_opencv_dist_coeffs": dist.reshape(-1).tolist(),
        }
    )
    if image_size is not None:
        info["calibration_image_size"] = image_size

    return K, dist, info


def validate_calibration_image_size(
    calibration_info: dict[str, Any],
    *,
    width: int,
    height: int,
) -> None:
    calibration_size = calibration_info.get("calibration_image_size")
    if calibration_size is None:
        return
    expected = [int(width), int(height)]
    if list(calibration_size) != expected:
        raise RuntimeError(
            f"Selected camera calibration image_size={calibration_size} does not "
            f"match the active camera stream {expected}."
        )


def _npz_scalar_or_list(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return arr.tolist()


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
        "coverage_min_cell_corners": np.asarray(
            int(stats.get("coverage_min_cell_corners", 0)),
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
        "spatial_holdout_regions": np.asarray(
            int(stats.get("spatial_holdout_regions", 0)),
            dtype=np.int32,
        ),
        "spatial_holdout_mean_px": np.asarray(
            float(stats.get("spatial_holdout_mean_px", np.nan)),
            dtype=np.float64,
        ),
        "spatial_holdout_worst_px": np.asarray(
            float(stats.get("spatial_holdout_worst_px", np.nan)),
            dtype=np.float64,
        ),
        "spatial_holdout_worst_region": np.asarray(
            str(stats.get("spatial_holdout_worst_region", ""))
        ),
        "spatial_holdout_region_err_px": np.asarray(
            stats.get("spatial_holdout_region_err_px", np.zeros((0, 0))),
            dtype=np.float64,
        ),
        "holdout_folds": np.asarray(
            int(stats.get("holdout_folds", 0)),
            dtype=np.int32,
        ),
        "holdout_corner_reprojection_mean_px": np.asarray(
            float(stats.get("holdout_corner_reprojection_mean_px", np.nan)),
            dtype=np.float64,
        ),
        "holdout_corner_reprojection_worst_px": np.asarray(
            float(stats.get("holdout_corner_reprojection_worst_px", np.nan)),
            dtype=np.float64,
        ),
        "holdout_intrinsics_spread_px": np.asarray(
            float(stats.get("holdout_intrinsics_spread_px", np.nan)),
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
        "camera_source": np.asarray(
            str(stats.get("camera_source", "charuco_camera"))
        ),
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

    for key in (
        "camera_backend",
        "camera_serial",
        "camera_model",
        "camera_width",
        "camera_height",
        "camera_fps",
        "pixel_format",
    ):
        if key in stats:
            payload[key] = np.asarray(stats[key])

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
    disable_realsense_ir_projector(profile, log_prefix="[calib_camera]")

    for _ in range(15):
        pipeline.wait_for_frames()

    return pipeline, profile


def disable_realsense_ir_projector(
    profile: Any,
    *,
    log_prefix: str = "[calib_camera]",
) -> dict[str, Any]:
    """Best-effort guard against IR speckle during visual calibration.

    The camera calibration uses only the colour stream, but explicitly forcing
    the depth-module emitter/laser off makes the capture state unambiguous and
    protects against persistent device settings from a previous depth session.
    """
    result: dict[str, Any] = {
        "attempted": False,
        "emitter_disabled": False,
        "laser_disabled": False,
        "errors": [],
    }
    try:
        import pyrealsense2 as rs
    except ImportError:
        result["errors"].append("pyrealsense2 unavailable")
        return result

    if profile is None:
        result["errors"].append("no active profile")
        return result

    try:
        device = profile.get_device()
        sensors = list(device.query_sensors())
    except Exception as exc:
        result["errors"].append(str(exc))
        return result

    option_specs = (
        ("emitter_enabled", "emitter_disabled"),
        ("laser_power", "laser_disabled"),
    )
    for sensor in sensors:
        for option_name, result_key in option_specs:
            option = getattr(rs.option, option_name, None)
            if option is None:
                continue
            try:
                if not sensor.supports(option):
                    continue
                result["attempted"] = True
                sensor.set_option(option, 0.0)
                result[result_key] = True
            except Exception as exc:
                result["errors"].append(f"{option_name}: {exc}")

    if result["attempted"]:
        print(
            f"{log_prefix} RealSense IR projector guard: "
            f"emitter_off={int(bool(result['emitter_disabled']))} "
            f"laser_off={int(bool(result['laser_disabled']))}"
        )
    elif result["errors"]:
        print(f"{log_prefix} RealSense IR projector guard skipped: {result['errors'][0]}")

    return result


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


def make_live_camera_config(
    *,
    backend: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
    serial: Optional[str] = None,
) -> CameraConfig:
    default_camera = CameraConfig()
    return CameraConfig(
        backend=default_camera.backend if backend is None else backend,
        width=default_camera.width if width is None else int(width),
        height=default_camera.height if height is None else int(height),
        fps=default_camera.fps if fps is None else int(fps),
        serial=default_camera.serial if serial is None else serial,
    )


def start_camera_source(camera_config):
    camera = create_camera_source(camera_config)
    camera.start()
    return camera


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
        "tilt_deg",
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


def select_images_dir_via_dialog(start_dir: Optional[Path] = None) -> Optional[Path]:
    """Open a Qt folder picker and return the chosen directory, or None on cancel."""
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog
    except ImportError as exc:
        try:
            from PyQt5.QtWidgets import QApplication, QFileDialog
        except ImportError:
            raise RuntimeError(
                "Offline mode needs PySide6 or PyQt5 for the folder dialog. "
                "Alternatively pass --images-dir explicitly."
            ) from exc

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    if start_dir is None:
        start_dir = Path(__file__).resolve().parent

    directory = QFileDialog.getExistingDirectory(
        None,
        "Ordner mit gespeicherten Kalibrierbildern (selected views) waehlen",
        str(start_dir),
        QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
    )

    return Path(directory) if directory else None


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
    holdout_folds: int = 3,
    model_names: Optional[str] = DEFAULT_SWEEP_MODELS,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    board, aruco_dict, detector_params = make_charuco_board()
    image_size = _image_size(calib_images[0])

    model_rows: list[dict[str, Any]] = []
    first_success: Path | None = None
    best_success: tuple[np.ndarray, np.ndarray, float, dict[str, Any]] | None = None
    compact_warm_start: Optional[tuple[np.ndarray, np.ndarray]] = None
    spatial_dets = detect_charuco_all(
        calib_images,
        board,
        aruco_dict,
        detector_params,
        min_charuco_corners=min_capture_corners,
    )

    for spec in calibration_model_specs(model_names):
        print(f"[calib_camera] Model {spec.name}: {spec.description}")
        is_rational = "rational" in spec.name
        seed_K = compact_warm_start[0] if (is_rational and compact_warm_start) else None
        seed_dist = compact_warm_start[1] if (is_rational and compact_warm_start) else None
        if seed_K is not None:
            print("[calib_camera]   warm start from compact-model solution")
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
                K_seed=seed_K,
                dist_seed=seed_dist,
            )
            if not is_rational and compact_warm_start is None:
                compact_warm_start = (
                    np.asarray(K, dtype=np.float64).reshape(3, 3).copy(),
                    np.asarray(dist, dtype=np.float64).reshape(-1).copy(),
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
            holdout_stats = holdout_model_stats(
                calib_images,
                board,
                aruco_dict,
                detector_params,
                min_charuco_corners=min_capture_corners,
                flags=spec.flags,
                dist_coeff_count=spec.dist_coeff_count,
                K_seed=seed_K,
                dist_seed=seed_dist,
                n_folds=holdout_folds,
            )
            stats.update(holdout_stats)
            stats.update(
                spatial_holdout_model_stats(
                    calib_images,
                    board,
                    aruco_dict,
                    detector_params,
                    min_charuco_corners=min_capture_corners,
                    flags=spec.flags,
                    dist_coeff_count=spec.dist_coeff_count,
                    K_seed=seed_K,
                    dist_seed=seed_dist,
                    detections=spatial_dets,
                )
            )
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
                    "holdout_folds": stats.get("holdout_folds"),
                    "holdout_corner_reprojection_mean_px": stats.get(
                        "holdout_corner_reprojection_mean_px"
                    ),
                    "holdout_corner_reprojection_worst_px": stats.get(
                        "holdout_corner_reprojection_worst_px"
                    ),
                    "holdout_intrinsics_spread_px": stats.get(
                        "holdout_intrinsics_spread_px"
                    ),
                    "spatial_holdout_mean_px": stats.get("spatial_holdout_mean_px"),
                    "spatial_holdout_worst_px": stats.get("spatial_holdout_worst_px"),
                    "spatial_holdout_worst_region": stats.get(
                        "spatial_holdout_worst_region"
                    ),
                    **radial_stats,
                }
            )
            print(
                "[calib_camera]   "
                f"rms={float(rms):.6f} reproj={float(reproj_mean_px):.6f}px "
                f"holdout={float(stats.get('holdout_corner_reprojection_mean_px', float('nan'))):.6f}px "
                f"spatial={float(stats.get('spatial_holdout_mean_px', float('nan'))):.3f}px"
                f"/worst {float(stats.get('spatial_holdout_worst_px', float('nan'))):.3f}px"
                f"@{stats.get('spatial_holdout_worst_region', '-')} "
                f"score={float(stats['model_quality_score']):.6f} "
                f"dist_n={dist_flat.size} radial_turns={radial_stats['radial_turn_count']} "
                f"positive={radial_stats['radial_positive']}"
            )
            print(
                "[calib_camera]   saved -> "
                + " | ".join(str(path) for path in save_paths)
            )
            for warning in intrinsics_sanity_warnings(K):
                print(f"[calib_camera]   WARNING: {warning}")
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
    camera_config: Any = None,
    camera_backend: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
    serial: Optional[str] = None,
    target_views: int = N_VIEWS,
    min_capture_corners: int = MIN_CHARUCO_CAPTURE,
    min_edge_capture_corners: int = MIN_CHARUCO_EDGE_CAPTURE,
    auto_capture_interval_s: float = AUTO_CAPTURE_INTERVAL_S,
    max_candidates: int = MAX_CAPTURE_CANDIDATES,
    save_selected_images: bool = True,
    coverage_grid_cols: int = SELECTION_GRID_COLS,
    coverage_grid_rows: int = SELECTION_GRID_ROWS,
    min_coverage_cells: int = MIN_COVERAGE_CELLS,
    min_cell_corners: int = MIN_CELL_CORNERS,
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
    holdout_folds: int = 3,
    selection_mode: str = "information",
    model_names: Optional[str] = DEFAULT_SWEEP_MODELS,
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
    min_cell_corners = max(0, int(min_cell_corners))
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

    camera = None
    camera_metadata: dict[str, Any] = {}
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        if camera_config is None:
            camera_config = make_live_camera_config(
                backend=camera_backend,
                width=width,
                height=height,
                fps=fps,
                serial=serial,
            )
        camera = start_camera_source(camera_config)
        camera_metadata = camera.metadata()
        profile = getattr(camera, "profile", None)
        factory_K = (
            None
            if profile is None
            else get_realsense_factory_intrinsics(
                profile,
                int(camera_metadata.get("width", width)),
                int(camera_metadata.get("height", height)),
            )
        )
        if factory_K is not None:
            print(f"[calib_camera] Factory K: fx={factory_K[0,0]:.1f} fy={factory_K[1,1]:.1f} "
                  f"cx={factory_K[0,2]:.1f} cy={factory_K[1,2]:.1f}")
        else:
            print("[calib_camera] Factory intrinsics not available, using identity init.")
        print(
            "[calib_camera] Camera running: "
            f"{camera_metadata.get('camera_backend', 'camera')} "
            f"{camera_metadata.get('width', '')}x{camera_metadata.get('height', '')} "
            f"@ {camera_metadata.get('fps', '')} fps"
        )
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
        prev_corner_map: dict[int, np.ndarray] = {}
        motion_px = float("nan")

        while True:
            camera_frame = camera.read()
            frame = None if camera_frame is None else camera_frame.image_bgr
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

            if got_new_frame:
                cur_corner_map = _charuco_corner_map(last_det)
                motion_px = _median_corner_motion_px(prev_corner_map, cur_corner_map)
                prev_corner_map = cur_corner_map
            board_still = bool(
                np.isfinite(motion_px) and motion_px <= MAX_CAPTURE_MOTION_PX
            )

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
                and board_still
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
                    min_cell_corners=min_cell_corners,
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
                sharpness=float(live_metrics.get("sharpness", float("nan"))),
                motion_px=motion_px,
                min_cell_corners=min_cell_corners,
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
                    f"{min_cell_corners}+ corners in EVERY grid cell, "
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

            select_fn = (
                select_calibration_candidates_information
                if selection_mode == "information"
                else select_calibration_candidates
            )
            select_extra = (
                {"board": board} if selection_mode == "information" else {}
            )
            selected_candidates = select_fn(
                state["candidates"],
                target_views=target_views,
                min_charuco_corners=min_capture_corners,
                min_edge_charuco_corners=min_edge_capture_corners,
                image_size=_image_size(frame),
                grid_cols=coverage_grid_cols,
                grid_rows=coverage_grid_rows,
                min_cell_corners=min_cell_corners,
                min_edge_corners=min_edge_corners,
                min_quadrant_corners=min_quadrant_corners,
                view_grid_cols=view_grid_cols,
                view_grid_rows=view_grid_rows,
                min_view_center_cells=min_view_center_cells,
                min_center_views=min_center_views,
                min_view_quadrant_views=min_view_quadrant_views,
                center_view_half_width_norm=center_view_half_width_norm,
                **select_extra,
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
                min_cell_corners=min_cell_corners,
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
                min_cell_corners=min_cell_corners,
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
            selected_sharpness = np.asarray(
                [
                    float(cand.metrics.get("sharpness", float("nan")))
                    for cand in selected_candidates
                ],
                dtype=np.float64,
            )
            if np.any(np.isfinite(selected_sharpness)):
                print(
                    "[calib_camera] Selected sharpness "
                    f"median={np.nanmedian(selected_sharpness):.0f} "
                    f"min={np.nanmin(selected_sharpness):.0f} "
                    "(known-good session: median ~6500)"
                )

            print("[calib_camera] Calibrating intrinsics model sweep...")

            selected_images = [cand.image for cand in selected_candidates]
            image_size = _image_size(frame)
            common_stats: dict[str, Any] = {
                "camera_source": f"charuco_{camera_metadata.get('camera_backend', 'camera')}",
                "camera_backend": str(camera_metadata.get("camera_backend", "")),
                "camera_serial": str(camera_metadata.get("camera_serial", "")),
                "camera_model": str(camera_metadata.get("camera_model", "")),
                "camera_width": int(camera_metadata.get("width", image_size[0])),
                "camera_height": int(camera_metadata.get("height", image_size[1])),
                "camera_fps": int(camera_metadata.get("fps", fps)),
                "pixel_format": str(camera_metadata.get("pixel_format", "")),
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
                "coverage_min_cell_corners": int(min_cell_corners),
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
            compact_warm_start: Optional[tuple[np.ndarray, np.ndarray]] = None
            # detections already exist on the candidates — reuse them for the
            # spatial holdout instead of re-detecting per model
            spatial_dets = [
                c.det
                for c in selected_candidates
                if c.det.charuco_corners is not None
                and c.det.charuco_ids is not None
                and c.det.num_charuco >= min_edge_capture_corners
            ]

            for spec in calibration_model_specs(model_names):
                print(f"[calib_camera] Model {spec.name}: {spec.description}")
                is_rational = "rational" in spec.name
                if is_rational and compact_warm_start is not None:
                    seed_K, seed_dist = compact_warm_start
                    print("[calib_camera]   warm start from compact-model solution")
                else:
                    seed_K, seed_dist = factory_K, None
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
                        K_seed=seed_K,
                        dist_seed=seed_dist,
                    )
                    if not is_rational and compact_warm_start is None:
                        compact_warm_start = (
                            np.asarray(K, dtype=np.float64).reshape(3, 3).copy(),
                            np.asarray(dist, dtype=np.float64).reshape(-1).copy(),
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

                    holdout_stats = holdout_model_stats(
                        selected_images,
                        board,
                        aruco_dict,
                        detector_params,
                        min_charuco_corners=min_edge_capture_corners,
                        flags=spec.flags,
                        dist_coeff_count=spec.dist_coeff_count,
                        K_seed=seed_K,
                        dist_seed=seed_dist,
                        n_folds=holdout_folds,
                    )
                    stats.update(holdout_stats)
                    stats.update(
                        spatial_holdout_model_stats(
                            selected_images,
                            board,
                            aruco_dict,
                            detector_params,
                            min_charuco_corners=min_edge_capture_corners,
                            flags=spec.flags,
                            dist_coeff_count=spec.dist_coeff_count,
                            K_seed=seed_K,
                            dist_seed=seed_dist,
                            detections=spatial_dets,
                        )
                    )

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
                            "holdout_folds": stats.get("holdout_folds"),
                            "holdout_corner_reprojection_mean_px": stats.get(
                                "holdout_corner_reprojection_mean_px"
                            ),
                            "holdout_corner_reprojection_worst_px": stats.get(
                                "holdout_corner_reprojection_worst_px"
                            ),
                            "holdout_intrinsics_spread_px": stats.get(
                                "holdout_intrinsics_spread_px"
                            ),
                            "spatial_holdout_mean_px": stats.get(
                                "spatial_holdout_mean_px"
                            ),
                            "spatial_holdout_worst_px": stats.get(
                                "spatial_holdout_worst_px"
                            ),
                            "spatial_holdout_worst_region": stats.get(
                                "spatial_holdout_worst_region"
                            ),
                            **radial_stats,
                        }
                    )
                    print(
                        "[calib_camera]   "
                        f"rms={float(rms):.6f} reproj={float(reproj_mean_px):.6f}px "
                        f"holdout={float(stats.get('holdout_corner_reprojection_mean_px', float('nan'))):.6f}px "
                        f"spatial={float(stats.get('spatial_holdout_mean_px', float('nan'))):.3f}px"
                        f"/worst {float(stats.get('spatial_holdout_worst_px', float('nan'))):.3f}px"
                        f"@{stats.get('spatial_holdout_worst_region', '-')} "
                        f"score={float(stats['model_quality_score']):.6f} "
                        f"dist_n={dist_flat.size} radial_turns={radial_stats['radial_turn_count']} "
                        f"positive={radial_stats['radial_positive']}"
                    )
                    print(
                        "[calib_camera]   saved -> "
                        + " | ".join(str(path) for path in save_paths)
                    )
                    for warning in intrinsics_sanity_warnings(K, factory_K):
                        print(f"[calib_camera]   WARNING: {warning}")
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
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()

    if state["saved_path"] is None:
        raise RuntimeError("No calibration was saved.")

    return Path(state["saved_path"])


def parse_args() -> argparse.Namespace:
    default_camera = CameraConfig()
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
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Offline mode: pick the folder with saved calibration images "
            "(e.g. a selected_views directory) via a Qt folder dialog, then run "
            "the model sweep on those images. Combine with --images-dir to skip the dialog."
        ),
    )
    parser.add_argument(
        "--camera",
        choices=("realsense", "basler"),
        default=default_camera.backend,
        help="Live camera backend. Defaults to CameraConfig in config.py.",
    )
    parser.add_argument(
        "--serial",
        default=default_camera.serial,
        help="Optional camera serial number. Defaults to CameraConfig in config.py.",
    )
    parser.add_argument("--width", type=int, default=default_camera.width)
    parser.add_argument("--height", type=int, default=default_camera.height)
    parser.add_argument("--fps", type=int, default=default_camera.fps)
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
        "--holdout-folds",
        type=int,
        default=3,
        help=(
            "Number of cross-validation folds for the model-selection score. "
            "Each model is recalibrated on train folds and scored on held-out "
            "views, so overfitted distortion models lose. Use 0 to disable."
        ),
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
        "--models",
        default=DEFAULT_SWEEP_MODELS,
        help=(
            "Comma-separated calibration models for the sweep "
            "(default: production set). Available: standard5, no_k3, "
            "standard5_no_tangent, no_k3_no_tangent, rational6, rational8. "
            "Empty string = all."
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=("information", "heuristic"),
        default="information",
        help=(
            "View selection strategy: 'information' = D-optimal greedy on the "
            "intrinsics information matrix (Rojtberg & Kuijper ISMAR'18; "
            "auto-diversity, tilt and edge weighting from the Jacobian), "
            "'heuristic' = legacy hand-tuned diversity bonuses."
        ),
    )
    parser.add_argument(
        "--min-cell-corners",
        type=int,
        default=MIN_CELL_CORNERS,
        help=(
            "Minimum ChArUco corner observations required in EVERY coverage "
            "grid cell before saving (0 disables). Prevents captures whose "
            "distortion is extrapolated in whole image regions."
        ),
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
            "half-FOV coordinates. 0.25 means roughly the central 25%% of half-width/height."
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
    images_dir = args.images_dir
    if args.offline and images_dir is None:
        images_dir = select_images_dir_via_dialog()
        if images_dir is None:
            print("[calib_camera] Offline mode cancelled: no folder selected.")
            return
    if images_dir is not None:
        image_paths = collect_image_paths(images_dir)
        print(f"[calib_camera] Offline model sweep from {len(image_paths)} images in {images_dir}")
        saved_path = run_model_sweep_for_images(
            calib_images=load_images(image_paths),
            output_path=args.output,
            min_capture_corners=args.min_corners,
            intrinsic_refinement_passes=args.charuco_refinement_passes,
            holdout_folds=args.holdout_folds,
            model_names=args.models,
        )
        print(f"[calib_camera] Done: {saved_path}")
        return

    camera_config = CameraConfig(
        backend=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial,
    )
    saved_path = run_live_calibration(
        output_path=args.output,
        camera_config=camera_config,
        target_views=args.target_views,
        min_capture_corners=args.min_corners,
        min_edge_capture_corners=args.min_edge_corners_visible,
        auto_capture_interval_s=args.capture_interval,
        max_candidates=args.max_candidates,
        save_selected_images=not args.no_save_selected_images,
        coverage_grid_cols=args.coverage_grid_cols,
        coverage_grid_rows=args.coverage_grid_rows,
        min_coverage_cells=args.min_coverage_cells,
        min_cell_corners=args.min_cell_corners,
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
        holdout_folds=args.holdout_folds,
        selection_mode=args.selection_mode,
        model_names=args.models,
    )
    print(f"[calib_camera] Done: {saved_path}")


if __name__ == "__main__":
    main()
