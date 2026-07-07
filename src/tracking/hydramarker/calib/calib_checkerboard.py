"""Checkerboard pose helper for HydraMarker.

This module does not load camera models, print results, or write files. Callers
provide OpenCV camera intrinsics/distortion and receive the estimated pose.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from tracking.hydramarker.calib import calib_camera
from tracking.pose_solvers import (
    _build_ippe_candidates,
    compute_reprojection_error_px,
)


CHECKERBOARD_PATTERN = (9, 9)  # inner corners: 10x10 printed cells -> 9x9 corners
CHECKERBOARD_SQUARE_SIZE_MM = 10.0
BOARD_AXIS_LENGTH_MM = 30.0

REALSENSE_WIDTH = 1920
REALSENSE_HEIGHT = 1080
REALSENSE_FPS = 30

DEFAULT_CAPTURE_INTERVAL_S = 0.05
DEFAULT_MIN_FRAMES = 12
DEFAULT_MAX_FRAMES = 300
DEFAULT_FRAME_OUTLIER_MAD_SCALE = 3.5
DEFAULT_AMBIGUITY_LIKELIHOOD_RATIO = 0.1
DEFAULT_AMBIGUITY_MIN_TRANSLATION_DELTA_MM = 0.1
DEFAULT_AMBIGUITY_MIN_ROTATION_DELTA_DEG = 0.03
DEFAULT_BURST_REFINE_CANDIDATE_MAX_RMS_GAP_PX = 0.25
DEFAULT_BURST_REFINE_CANDIDATE_MAX_RMS_RATIO = 2.0
CHARUCO_TABLE_MIN_FRAMES_PER_POSITION = 100
CHARUCO_TABLE_MAX_FRAMES_PER_POSITION = 300
CHARUCO_TABLE_MIN_POSITIONS = 3
CHARUCO_TABLE_TARGET_POSITIONS = 5
CHARUCO_TABLE_MIN_CORNERS = max(24, calib_camera.MIN_CHARUCO_CAPTURE)

WINDOW_NAME = "HydraMarker Checkerboard Pose Calibration"
CHARUCO_TABLE_WINDOW_NAME = "HydraMarker ChArUco Table Calibration"

StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class _BurstPoseCandidate:
    source_index: int
    rvec: np.ndarray
    tvec: np.ndarray
    mean_px: float
    median_px: float
    rms_px: float
    p95_px: float
    max_px: float
    sse_px2: float


@dataclass(frozen=True)
class CheckerboardPose:
    rvec_cb: np.ndarray
    tvec_cb_mm: np.ndarray
    T_C_B: np.ndarray
    T_B_C: np.ndarray
    median_corners_uv: np.ndarray
    frame_mask: np.ndarray
    all_errors_px: np.ndarray
    frame_rms_px: np.ndarray
    corner_std_px: np.ndarray
    pnp_flag: int
    pattern_inner_corners: tuple[int, int] = CHECKERBOARD_PATTERN
    square_size_mm: float = CHECKERBOARD_SQUARE_SIZE_MM
    solver_mode: str = "legacy"
    candidate_count: int = 0
    selected_candidate_index: int | None = None
    alternative_rms_px: float = float("nan")
    alternative_error_gap_px: float = float("nan")
    alternative_error_ratio: float = float("nan")
    alternative_likelihood_ratio: float = float("nan")
    alternative_translation_delta_mm: float = float("nan")
    alternative_rotation_delta_deg: float = float("nan")
    pose_ambiguous: bool = False

    @property
    def frames_total(self) -> int:
        return int(self.frame_mask.size)

    @property
    def frames_used(self) -> int:
        return int(np.count_nonzero(self.frame_mask))

    @property
    def reproj_mean_px(self) -> float:
        return float(np.mean(self.all_errors_px))

    @property
    def reproj_median_px(self) -> float:
        return float(np.median(self.all_errors_px))

    @property
    def reproj_p95_px(self) -> float:
        return float(np.percentile(self.all_errors_px, 95))

    @property
    def reproj_max_px(self) -> float:
        return float(np.max(self.all_errors_px))

    @property
    def collected_frames(self) -> int:
        return self.frames_total

    @property
    def mean_corner_std_px(self) -> float:
        return float(np.mean(self.corner_std_px))

    @property
    def max_corner_std_px(self) -> float:
        return float(np.max(self.corner_std_px))


@dataclass(frozen=True)
class CharucoTableFramePose:
    frame_index: int
    rvec_cb: np.ndarray
    tvec_cb_mm: np.ndarray
    R_C_B: np.ndarray
    normal_camera: np.ndarray
    x_axis_camera: np.ndarray
    origin_camera_mm: np.ndarray
    object_points_mm: np.ndarray
    image_points_uv: np.ndarray
    errors_px: np.ndarray
    rms_px: float
    mean_px: float
    p95_px: float
    max_px: float
    num_charuco: int
    num_aruco: int


@dataclass(frozen=True)
class CharucoTablePosition:
    position_index: int
    frames: tuple[CharucoTableFramePose, ...]
    frame_mask: np.ndarray

    @property
    def used_frames(self) -> tuple[CharucoTableFramePose, ...]:
        return tuple(frame for frame, keep in zip(self.frames, self.frame_mask) if bool(keep))

    @property
    def normal_camera(self) -> np.ndarray:
        return _mean_unit_vectors([frame.normal_camera for frame in self.used_frames])

    @property
    def x_axis_camera(self) -> np.ndarray:
        return _mean_unit_vectors([frame.x_axis_camera for frame in self.used_frames])

    @property
    def origin_camera_mm(self) -> np.ndarray:
        used = self.used_frames
        if not used:
            return np.zeros(3, dtype=np.float64)
        return np.mean([frame.origin_camera_mm for frame in used], axis=0)

    @property
    def median_rms_px(self) -> float:
        used = self.used_frames
        if not used:
            return float("nan")
        return float(np.median([frame.rms_px for frame in used]))


@dataclass(frozen=True)
class CharucoTableCalibration:
    rvec_cb: np.ndarray
    tvec_cb_mm: np.ndarray
    T_C_B: np.ndarray
    T_B_C: np.ndarray
    positions: tuple[CharucoTablePosition, ...]
    frame_mask: np.ndarray
    all_errors_px: np.ndarray
    frame_rms_px: np.ndarray
    normal_camera: np.ndarray
    x_axis_camera: np.ndarray
    y_axis_camera: np.ndarray
    source_position_index: int
    square_length_mm: float
    marker_length_mm: float
    aruco_dictionary_id: int
    solver_mode: str = "charuco_table_multi_position"
    pnp_flag: int = cv2.SOLVEPNP_ITERATIVE

    @property
    def frames_total(self) -> int:
        return int(self.frame_mask.size)

    @property
    def frames_used(self) -> int:
        return int(np.count_nonzero(self.frame_mask))

    @property
    def positions_used(self) -> int:
        return len(self.positions)

    @property
    def reproj_mean_px(self) -> float:
        return float(np.mean(self.all_errors_px)) if self.all_errors_px.size else float("nan")

    @property
    def reproj_median_px(self) -> float:
        return float(np.median(self.all_errors_px)) if self.all_errors_px.size else float("nan")

    @property
    def reproj_p95_px(self) -> float:
        return float(np.percentile(self.all_errors_px, 95)) if self.all_errors_px.size else float("nan")

    @property
    def reproj_max_px(self) -> float:
        return float(np.max(self.all_errors_px)) if self.all_errors_px.size else float("nan")

    @property
    def collected_frames(self) -> int:
        return self.frames_total

    @property
    def mean_corner_std_px(self) -> float:
        kept = self.frame_rms_px[self.frame_mask]
        return float(np.std(kept)) if kept.size else float("nan")

    @property
    def max_corner_std_px(self) -> float:
        kept = self.frame_rms_px[self.frame_mask]
        return float(np.max(kept) - np.min(kept)) if kept.size else float("nan")


def checkerboard_object_points_mm(
    *,
    pattern: tuple[int, int] = CHECKERBOARD_PATTERN,
    square_size_mm: float = CHECKERBOARD_SQUARE_SIZE_MM,
) -> np.ndarray:
    cols, rows = pattern
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj = np.zeros((rows * cols, 3), dtype=np.float64)
    obj[:, :2] = grid.astype(np.float64) * float(square_size_mm)
    return obj


def make_transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def reprojection_vectors_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        np.asarray(dist, dtype=np.float64).reshape(-1, 1),
    )
    projected = projected.reshape(-1, 2)
    measured = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    return projected - measured


def _robust_frame_mask(frame_rms: np.ndarray, mad_scale: float) -> np.ndarray:
    if frame_rms.size == 0:
        return np.zeros(0, dtype=bool)
    median = float(np.median(frame_rms))
    mad = float(np.median(np.abs(frame_rms - median)))
    sigma = 1.4826 * mad
    threshold = median + float(mad_scale) * max(sigma, 1e-9)
    return frame_rms <= threshold


def _rotation_delta_deg_from_rvecs(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    delta = (
        np.asarray(rvec_b, dtype=np.float64).reshape(3)
        - np.asarray(rvec_a, dtype=np.float64).reshape(3)
    )
    norm_rad = float(np.linalg.norm(delta))
    if not np.isfinite(norm_rad):
        return float("nan")
    return float(np.degrees(norm_rad))


def _stack_repeated_object_points(
    object_points: np.ndarray,
    corner_stack: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    corner_stack = np.asarray(corner_stack, dtype=np.float64)
    if corner_stack.ndim != 3 or corner_stack.shape[1:] != (len(object_points), 2):
        raise ValueError(
            "corner_stack must have shape (frames, corners, 2) matching object_points."
        )
    object_all = np.tile(object_points, (corner_stack.shape[0], 1))
    image_all = corner_stack.reshape(-1, 2)
    return object_all, image_all


def _score_burst_candidate(
    *,
    source_index: int,
    rvec: np.ndarray,
    tvec: np.ndarray,
    object_all: np.ndarray,
    image_all: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> _BurstPoseCandidate:
    _uv, mean_px, median_px, max_px, per_point = compute_reprojection_error_px(
        object_points_xyz=object_all,
        image_points_uv=image_all,
        rvec=rvec,
        tvec=tvec,
        K=K,
        dist=dist,
    )
    per_point = np.asarray(per_point, dtype=np.float64).reshape(-1)
    sse_px2 = float(np.sum(per_point * per_point))
    rms_px = float(np.sqrt(np.mean(per_point * per_point)))
    return _BurstPoseCandidate(
        source_index=int(source_index),
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        mean_px=float(mean_px),
        median_px=float(median_px),
        rms_px=rms_px,
        p95_px=float(np.percentile(per_point, 95)),
        max_px=float(max_px),
        sse_px2=sse_px2,
    )


def _refine_burst_pose(
    object_all: np.ndarray,
    image_all: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec_init: np.ndarray,
    tvec_init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_all = np.asarray(object_all, dtype=np.float64).reshape(-1, 3)
    image_all = np.asarray(image_all, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    rvec = np.asarray(rvec_init, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec_init, dtype=np.float64).reshape(3, 1)

    if hasattr(cv2, "solvePnPRefineVVS"):
        try:
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                30,
                1e-6,
            )
            rv, tv = cv2.solvePnPRefineVVS(
                object_all,
                image_all,
                K,
                dist,
                rvec.copy(),
                tvec.copy(),
                criteria,
            )
            return (
                np.asarray(rv, dtype=np.float64).reshape(3, 1),
                np.asarray(tv, dtype=np.float64).reshape(3, 1),
            )
        except cv2.error:
            pass

    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                30,
                1e-6,
            )
            rv, tv = cv2.solvePnPRefineLM(
                object_all,
                image_all,
                K,
                dist,
                rvec.copy(),
                tvec.copy(),
                criteria,
            )
            return (
                np.asarray(rv, dtype=np.float64).reshape(3, 1),
                np.asarray(tv, dtype=np.float64).reshape(3, 1),
            )
        except cv2.error:
            pass

    return rvec, tvec


def _solve_burst_checkerboard_pose(
    object_points: np.ndarray,
    corner_stack: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, float | int | bool | str | None]]:
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    corner_stack = np.asarray(corner_stack, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    median_corners = np.median(corner_stack, axis=0)
    object_all, image_all = _stack_repeated_object_points(object_points, corner_stack)

    refined_candidates: list[_BurstPoseCandidate] = []
    try:
        ippe_candidates = _build_ippe_candidates(
            object_points_xyz=object_points,
            image_points_uv=median_corners,
            K=K,
            dist=dist,
        )
    except Exception:
        ippe_candidates = []

    raw_candidates = [
        _score_burst_candidate(
            source_index=idx,
            rvec=np.asarray(candidate.rvec, dtype=np.float64).reshape(3, 1),
            tvec=np.asarray(candidate.tvec, dtype=np.float64).reshape(3, 1),
            object_all=object_all,
            image_all=image_all,
            K=K,
            dist=dist,
        )
        for idx, candidate in enumerate(ippe_candidates)
    ]
    raw_candidates.sort(key=lambda c: (c.sse_px2, c.rms_px, c.max_px))
    raw_best_rms = raw_candidates[0].rms_px if raw_candidates else float("inf")

    for raw in raw_candidates:
        should_refine = (
            raw is raw_candidates[0]
            or raw.rms_px <= raw_best_rms + DEFAULT_BURST_REFINE_CANDIDATE_MAX_RMS_GAP_PX
            or raw.rms_px <= raw_best_rms * DEFAULT_BURST_REFINE_CANDIDATE_MAX_RMS_RATIO
        )
        if not should_refine:
            refined_candidates.append(raw)
            continue

        rvec, tvec = _refine_burst_pose(
            object_all,
            image_all,
            K,
            dist,
            raw.rvec,
            raw.tvec,
        )
        refined_candidates.append(
            _score_burst_candidate(
                source_index=raw.source_index,
                rvec=rvec,
                tvec=tvec,
                object_all=object_all,
                image_all=image_all,
                K=K,
                dist=dist,
            )
        )

    if not refined_candidates:
        rvec, tvec, flag = solve_best_checkerboard_pose(
            object_points,
            median_corners,
            K,
            dist,
        )
        rvec, tvec = _refine_burst_pose(
            object_all,
            image_all,
            K,
            dist,
            rvec,
            tvec,
        )
        candidate = _score_burst_candidate(
            source_index=0,
            rvec=rvec,
            tvec=tvec,
            object_all=object_all,
            image_all=image_all,
            K=K,
            dist=dist,
        )
        diagnostics: dict[str, float | int | bool | str | None] = {
            "solver_mode": "legacy_burst_refine",
            "candidate_count": 1,
            "selected_candidate_index": None,
            "alternative_rms_px": float("nan"),
            "alternative_error_gap_px": float("nan"),
            "alternative_error_ratio": float("nan"),
            "alternative_likelihood_ratio": float("nan"),
            "alternative_translation_delta_mm": float("nan"),
            "alternative_rotation_delta_deg": float("nan"),
            "pose_ambiguous": False,
        }
        return candidate.rvec, candidate.tvec, int(flag), diagnostics

    refined_candidates.sort(key=lambda c: (c.sse_px2, c.rms_px, c.max_px))
    best = refined_candidates[0]
    alt = refined_candidates[1] if len(refined_candidates) > 1 else None

    diagnostics = {
        "solver_mode": "ippe_burst_refine",
        "candidate_count": len(refined_candidates),
        "selected_candidate_index": int(best.source_index),
        "alternative_rms_px": float("nan"),
        "alternative_error_gap_px": float("nan"),
        "alternative_error_ratio": float("nan"),
        "alternative_likelihood_ratio": float("nan"),
        "alternative_translation_delta_mm": float("nan"),
        "alternative_rotation_delta_deg": float("nan"),
        "pose_ambiguous": False,
    }

    if alt is not None:
        delta_sse = max(0.0, float(alt.sse_px2 - best.sse_px2))
        dof = max(1, 2 * int(image_all.shape[0]) - 6)
        sigma2 = max(float(best.sse_px2) / float(dof), 1e-12)
        likelihood_ratio = float(np.exp(-0.5 * delta_sse / sigma2))
        translation_delta = float(np.linalg.norm(alt.tvec.reshape(3) - best.tvec.reshape(3)))
        rotation_delta = _rotation_delta_deg_from_rvecs(best.rvec, alt.rvec)

        diagnostics.update(
            {
                "alternative_rms_px": float(alt.rms_px),
                "alternative_error_gap_px": float(alt.rms_px - best.rms_px),
                "alternative_error_ratio": float(alt.rms_px / max(best.rms_px, 1e-12)),
                "alternative_likelihood_ratio": likelihood_ratio,
                "alternative_translation_delta_mm": translation_delta,
                "alternative_rotation_delta_deg": rotation_delta,
                "pose_ambiguous": bool(
                    likelihood_ratio >= DEFAULT_AMBIGUITY_LIKELIHOOD_RATIO
                    and (
                        translation_delta >= DEFAULT_AMBIGUITY_MIN_TRANSLATION_DELTA_MM
                        or rotation_delta >= DEFAULT_AMBIGUITY_MIN_ROTATION_DELTA_DEG
                    )
                ),
            }
        )

    return best.rvec, best.tvec, int(cv2.SOLVEPNP_IPPE), diagnostics


def _solve_fast_planar_pose_for_mask(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    try:
        candidates = _build_ippe_candidates(
            object_points_xyz=object_points,
            image_points_uv=image_points,
            K=K,
            dist=dist,
        )
        best = min(candidates, key=lambda c: c.reproj_mean_px)
        return (
            np.asarray(best.rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(best.tvec, dtype=np.float64).reshape(3, 1),
        )
    except Exception:
        pass

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("Could not estimate checkerboard pose for frame masking.")
    return (
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
    )


def solve_best_checkerboard_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    flags: list[int] = []
    if hasattr(cv2, "SOLVEPNP_SQPNP"):
        flags.append(int(cv2.SOLVEPNP_SQPNP))
    if hasattr(cv2, "SOLVEPNP_IPPE"):
        flags.append(int(cv2.SOLVEPNP_IPPE))
    flags.append(int(cv2.SOLVEPNP_ITERATIVE))

    best: tuple[float, np.ndarray, np.ndarray, int] | None = None
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    for flag in flags:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                K,
                dist,
                flags=flag,
            )
        except cv2.error:
            continue
        if not ok:
            continue

        candidates = [(rvec.reshape(3, 1), tvec.reshape(3, 1))]
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

        for cand_rvec, cand_tvec in candidates:
            vec = reprojection_vectors_px(
                object_points,
                image_points,
                cand_rvec,
                cand_tvec,
                K,
                dist,
            )
            err = np.linalg.norm(vec, axis=1)
            penalty = 0.0 if float(cand_tvec.reshape(3)[2]) > 0.0 else 1000.0
            score = float(np.mean(err)) + penalty
            if best is None or score < best[0]:
                best = (score, cand_rvec, cand_tvec, flag)

    if best is None:
        raise RuntimeError("Could not estimate checkerboard pose.")
    _, rvec, tvec, flag = best
    return rvec.reshape(3, 1), tvec.reshape(3, 1), int(flag)


def estimate_pose_from_detections(
    detections: Sequence[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    *,
    pattern: tuple[int, int] = CHECKERBOARD_PATTERN,
    square_size_mm: float = CHECKERBOARD_SQUARE_SIZE_MM,
    mad_scale: float = DEFAULT_FRAME_OUTLIER_MAD_SCALE,
) -> CheckerboardPose:
    if not detections:
        raise RuntimeError("No checkerboard detections were collected.")

    stack = np.stack([np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in detections])
    object_points = checkerboard_object_points_mm(
        pattern=pattern,
        square_size_mm=square_size_mm,
    )

    median_corners = np.median(stack, axis=0)
    rvec_mask, tvec_mask = _solve_fast_planar_pose_for_mask(
        object_points,
        median_corners,
        K,
        dist,
    )

    frame_vecs = np.asarray(
        [
            reprojection_vectors_px(object_points, corners, rvec_mask, tvec_mask, K, dist)
            for corners in stack
        ],
        dtype=np.float64,
    )
    frame_errors = np.linalg.norm(frame_vecs, axis=2)
    frame_rms = np.sqrt(np.mean(frame_errors * frame_errors, axis=1))
    keep = _robust_frame_mask(frame_rms, mad_scale)
    if int(np.count_nonzero(keep)) >= max(3, min(len(detections), 8)):
        median_corners = np.median(stack[keep], axis=0)
    else:
        keep = np.ones(len(detections), dtype=bool)
        median_corners = np.median(stack, axis=0)

    rvec, tvec, flag, diagnostics = _solve_burst_checkerboard_pose(
        object_points,
        stack[keep],
        K,
        dist,
    )

    frame_vecs = np.asarray(
        [
            reprojection_vectors_px(object_points, corners, rvec, tvec, K, dist)
            for corners in stack
        ],
        dtype=np.float64,
    )
    frame_errors = np.linalg.norm(frame_vecs, axis=2)
    frame_rms = np.sqrt(np.mean(frame_errors * frame_errors, axis=1))
    corner_std = np.linalg.norm(np.std(stack[keep], axis=0), axis=1)

    T_C_B = make_transform_from_rvec_tvec(rvec, tvec)
    T_B_C = invert_transform(T_C_B)
    return CheckerboardPose(
        rvec_cb=rvec,
        tvec_cb_mm=tvec,
        T_C_B=T_C_B,
        T_B_C=T_B_C,
        median_corners_uv=median_corners,
        frame_mask=keep,
        all_errors_px=frame_errors.reshape(-1),
        frame_rms_px=frame_rms,
        corner_std_px=corner_std,
        pnp_flag=flag,
        pattern_inner_corners=pattern,
        square_size_mm=float(square_size_mm),
        solver_mode=str(diagnostics.get("solver_mode", "unknown")),
        candidate_count=int(diagnostics.get("candidate_count", 0) or 0),
        selected_candidate_index=(
            None
            if diagnostics.get("selected_candidate_index") is None
            else int(diagnostics["selected_candidate_index"])
        ),
        alternative_rms_px=float(diagnostics.get("alternative_rms_px", float("nan"))),
        alternative_error_gap_px=float(
            diagnostics.get("alternative_error_gap_px", float("nan"))
        ),
        alternative_error_ratio=float(
            diagnostics.get("alternative_error_ratio", float("nan"))
        ),
        alternative_likelihood_ratio=float(
            diagnostics.get("alternative_likelihood_ratio", float("nan"))
        ),
        alternative_translation_delta_mm=float(
            diagnostics.get("alternative_translation_delta_mm", float("nan"))
        ),
        alternative_rotation_delta_deg=float(
            diagnostics.get("alternative_rotation_delta_deg", float("nan"))
        ),
        pose_ambiguous=bool(diagnostics.get("pose_ambiguous", False)),
    )


def detect_checkerboard_corners(
    frame_bgr: np.ndarray,
    *,
    pattern: tuple[int, int] = CHECKERBOARD_PATTERN,
) -> np.ndarray | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
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
            # Adaptive window (~1/4 corner spacing) + border guard, shared with
            # the camera-calibration pipeline; the SB path above refines itself.
            corners = calib_camera._refine_charuco_subpix(gray, corners)

    if not ok or corners is None:
        return None
    return np.asarray(corners, dtype=np.float64).reshape(-1, 2)


def start_realsense_color_stream(*, width: int, height: int, fps: int):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required for live checkerboard capture.") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    calib_camera.disable_realsense_ir_projector(
        profile,
        log_prefix="[calib_checkerboard]",
    )
    for _ in range(15):
        pipeline.wait_for_frames()
    return pipeline, profile


def get_color_frame_bgr(pipeline) -> np.ndarray | None:
    frames = pipeline.poll_for_frames()
    if not frames:
        return None
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data()).copy()


def wait_color_frame_bgr(source) -> np.ndarray | None:
    if hasattr(source, "read") and not hasattr(source, "wait_for_frames"):
        frame = source.read()
        return None if frame is None else frame.image_bgr

    frames = source.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data()).copy()


def capture_charuco_table_calibration_from_camera_source(
    camera,
    K: np.ndarray,
    dist: np.ndarray,
    **kwargs,
) -> CharucoTableCalibration | None:
    return capture_charuco_table_calibration_from_pipeline(
        camera,
        K,
        dist,
        **kwargs,
    )


def capture_checkerboard_detections_from_camera_source(
    camera,
    K: np.ndarray,
    dist: np.ndarray,
    **kwargs,
) -> list[np.ndarray]:
    return capture_checkerboard_detections_from_pipeline(
        camera,
        K,
        dist,
        **kwargs,
    )


def capture_checkerboard_pose_from_camera_source(
    camera,
    K: np.ndarray,
    dist: np.ndarray,
    **kwargs,
) -> CheckerboardPose | None:
    return capture_checkerboard_pose_from_pipeline(
        camera,
        K,
        dist,
        **kwargs,
    )


def _draw_text_box(
    img_bgr: np.ndarray,
    lines: Sequence[str],
    *,
    color: tuple[int, int, int],
) -> np.ndarray:
    vis = img_bgr.copy()
    x, y = 28, 46
    line_gap = 30
    for i, text in enumerate(lines):
        yy = y + i * line_gap
        cv2.putText(
            vis,
            text,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            text,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            color,
            2,
            cv2.LINE_AA,
        )
    return vis


def _draw_axes(
    frame_bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> None:
    try:
        cv2.drawFrameAxes(
            frame_bgr,
            np.asarray(K, dtype=np.float64).reshape(3, 3),
            np.asarray(dist, dtype=np.float64).reshape(-1, 1),
            rvec,
            tvec,
            BOARD_AXIS_LENGTH_MM,
            3,
        )
    except cv2.error:
        pass


def draw_capture_overlay(
    frame_bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    corners: np.ndarray | None,
    detections: Sequence[np.ndarray],
    *,
    recording: bool,
    pattern: tuple[int, int] = CHECKERBOARD_PATTERN,
    square_size_mm: float = CHECKERBOARD_SQUARE_SIZE_MM,
) -> np.ndarray:
    vis = frame_bgr.copy()
    if corners is not None:
        cv2.drawChessboardCorners(
            vis,
            pattern,
            np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
            True,
        )
        try:
            obj = checkerboard_object_points_mm(
                pattern=pattern,
                square_size_mm=square_size_mm,
            )
            rvec, tvec, _ = solve_best_checkerboard_pose(obj, corners, K, dist)
            _draw_axes(vis, K, dist, rvec, tvec)
        except RuntimeError:
            pass

    color = (0, 210, 255) if recording else ((0, 255, 0) if corners is not None else (0, 0, 255))
    status = "RECORDING good checkerboard frames (SPACE stops)" if recording else "READY (SPACE starts)"
    lines = [
        status,
        f"valid detections in memory: {len(detections)}",
        "Keep checkerboard/camera rigid. Slight lighting/noise variation is fine.",
        "Keys: SPACE start/stop | Q/ESC quit",
    ]
    return _draw_text_box(vis, lines, color=color)


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12 or not np.isfinite(norm):
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return arr / norm


def _mean_unit_vectors(vectors: Sequence[np.ndarray]) -> np.ndarray:
    values = [_normalize_vector(v) for v in vectors]
    if not values:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    reference = values[0]
    aligned = []
    for value in values:
        aligned.append(-value if float(np.dot(value, reference)) < 0.0 else value)
    return _normalize_vector(np.mean(aligned, axis=0))


def _angular_error_deg(vectors: np.ndarray, reference: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    reference = _normalize_vector(reference)
    dots = np.clip(vectors @ reference, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _robust_scalar_mask(values: np.ndarray, mad_scale: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * mad
    threshold = median + float(mad_scale) * max(sigma, 1e-12)
    return values <= threshold


def _charuco_object_points_mm(board: Any, charuco_ids: np.ndarray) -> np.ndarray:
    obj_m = calib_camera._charuco_object_points(board, charuco_ids)
    return np.asarray(obj_m, dtype=np.float64).reshape(-1, 3) * 1000.0


def estimate_charuco_table_frame_pose(
    frame_bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    board: Any,
    aruco_dict: Any,
    detector_params: Any,
    frame_index: int,
    min_charuco_corners: int = CHARUCO_TABLE_MIN_CORNERS,
) -> tuple[CharucoTableFramePose | None, calib_camera.CharucoDetection]:
    det = calib_camera.detect_charuco(
        frame_bgr,
        board,
        aruco_dict,
        detector_params,
        camera_matrix=K,
        dist_coeffs=dist,
    )
    if det.charuco_ids is None or det.charuco_corners is None:
        return None, det
    if det.num_charuco < int(min_charuco_corners):
        return None, det

    object_points = _charuco_object_points_mm(board, det.charuco_ids)
    image_points = np.asarray(det.charuco_corners, dtype=np.float64).reshape(-1, 2)
    try:
        rvec, tvec, _flag = solve_best_checkerboard_pose(
            object_points,
            image_points,
            K,
            dist,
        )
    except RuntimeError:
        return None, det

    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    if hasattr(cv2, "solvePnPRefineVVS"):
        try:
            rvec, tvec = cv2.solvePnPRefineVVS(
                object_points,
                image_points,
                np.asarray(K, dtype=np.float64).reshape(3, 3),
                np.asarray(dist, dtype=np.float64).reshape(-1, 1),
                rvec,
                tvec,
            )
        except cv2.error:
            pass

    residual = reprojection_vectors_px(object_points, image_points, rvec, tvec, K, dist)
    errors = np.linalg.norm(residual, axis=1)
    R_C_B, _ = cv2.Rodrigues(rvec)
    R_C_B = np.asarray(R_C_B, dtype=np.float64).reshape(3, 3)

    pose = CharucoTableFramePose(
        frame_index=int(frame_index),
        rvec_cb=rvec.reshape(3, 1).copy(),
        tvec_cb_mm=tvec.reshape(3, 1).copy(),
        R_C_B=R_C_B.copy(),
        normal_camera=_normalize_vector(R_C_B[:, 2]),
        x_axis_camera=_normalize_vector(R_C_B[:, 0]),
        origin_camera_mm=tvec.reshape(3).copy(),
        object_points_mm=np.asarray(object_points, dtype=np.float64).reshape(-1, 3).copy(),
        image_points_uv=np.asarray(image_points, dtype=np.float64).reshape(-1, 2).copy(),
        errors_px=np.asarray(errors, dtype=np.float64).reshape(-1),
        rms_px=float(np.sqrt(np.mean(errors * errors))),
        mean_px=float(np.mean(errors)),
        p95_px=float(np.percentile(errors, 95)),
        max_px=float(np.max(errors)),
        num_charuco=int(det.num_charuco),
        num_aruco=int(det.num_aruco),
    )
    return pose, det


def _charuco_position_from_frames(
    frames: Sequence[CharucoTableFramePose],
    *,
    position_index: int,
    mad_scale: float,
) -> CharucoTablePosition:
    if not frames:
        raise RuntimeError("No ChArUco pose frames collected for this table position.")
    rms = np.asarray([frame.rms_px for frame in frames], dtype=np.float64)
    keep = _robust_scalar_mask(rms, mad_scale)
    if int(np.count_nonzero(keep)) < max(8, min(len(frames), 20)):
        keep = np.ones(len(frames), dtype=bool)

    normals = np.asarray([frame.normal_camera for frame in frames], dtype=np.float64)
    normal_ref = _mean_unit_vectors(normals[keep])
    angular = _angular_error_deg(normals, normal_ref)
    keep_angle = _robust_scalar_mask(angular, mad_scale)
    combined = keep & keep_angle
    if int(np.count_nonzero(combined)) >= max(8, min(len(frames), 20)):
        keep = combined

    return CharucoTablePosition(
        position_index=int(position_index),
        frames=tuple(frames),
        frame_mask=np.asarray(keep, dtype=bool).reshape(-1),
    )


def _build_charuco_table_calibration(
    positions: Sequence[CharucoTablePosition],
    *,
    mad_scale: float,
) -> CharucoTableCalibration:
    if len(positions) < CHARUCO_TABLE_MIN_POSITIONS:
        raise RuntimeError(
            f"Need at least {CHARUCO_TABLE_MIN_POSITIONS} ChArUco table positions."
        )

    all_kept_frames: list[CharucoTableFramePose] = []
    frame_mask_parts: list[np.ndarray] = []
    for position in positions:
        all_kept_frames.extend(position.used_frames)
        frame_mask_parts.append(position.frame_mask)
    if not all_kept_frames:
        raise RuntimeError("No robust ChArUco table frames left after outlier rejection.")

    normals = np.asarray([frame.normal_camera for frame in all_kept_frames], dtype=np.float64)
    normal_ref = _mean_unit_vectors(normals)
    angular = _angular_error_deg(normals, normal_ref)
    normal_keep = _robust_scalar_mask(angular, mad_scale)
    if int(np.count_nonzero(normal_keep)) >= max(20, len(all_kept_frames) // 2):
        normal = _mean_unit_vectors(normals[normal_keep])
    else:
        normal = normal_ref

    source_position = min(positions, key=lambda pos: pos.median_rms_px)
    source_x = source_position.x_axis_camera
    x_axis = source_x - float(np.dot(source_x, normal)) * normal
    if float(np.linalg.norm(x_axis)) <= 1e-9:
        source_y = _mean_unit_vectors([frame.R_C_B[:, 1] for frame in source_position.used_frames])
        x_axis = np.cross(source_y, normal)
    x_axis = _normalize_vector(x_axis)
    y_axis = _normalize_vector(np.cross(normal, x_axis))
    x_axis = _normalize_vector(np.cross(y_axis, normal))

    R_C_B = np.column_stack([x_axis, y_axis, normal]).astype(np.float64)
    origin_camera = source_position.origin_camera_mm.reshape(3)
    T_C_B = np.eye(4, dtype=np.float64)
    T_C_B[:3, :3] = R_C_B
    T_C_B[:3, 3] = origin_camera
    T_B_C = invert_transform(T_C_B)
    rvec, _ = cv2.Rodrigues(R_C_B)

    all_errors = np.concatenate([frame.errors_px for frame in all_kept_frames])
    all_rms = np.asarray(
        [frame.rms_px for position in positions for frame in position.frames],
        dtype=np.float64,
    )
    frame_mask = np.concatenate(frame_mask_parts).astype(bool)

    return CharucoTableCalibration(
        rvec_cb=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec_cb_mm=origin_camera.reshape(3, 1),
        T_C_B=T_C_B,
        T_B_C=T_B_C,
        positions=tuple(positions),
        frame_mask=frame_mask,
        all_errors_px=np.asarray(all_errors, dtype=np.float64).reshape(-1),
        frame_rms_px=all_rms.reshape(-1),
        normal_camera=normal.reshape(3),
        x_axis_camera=x_axis.reshape(3),
        y_axis_camera=y_axis.reshape(3),
        source_position_index=int(source_position.position_index),
        square_length_mm=float(calib_camera.SQUARE_LEN_M * 1000.0),
        marker_length_mm=float(calib_camera.MARKER_LEN_M * 1000.0),
        aruco_dictionary_id=int(calib_camera.DICT_ID),
    )


def draw_charuco_table_capture_overlay(
    frame_bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    pose: CharucoTableFramePose | None,
    det: calib_camera.CharucoDetection | None,
    positions: Sequence[CharucoTablePosition],
    current_frames: Sequence[CharucoTableFramePose],
    *,
    recording: bool,
    min_positions: int,
    target_positions: int,
    min_frames_per_position: int,
) -> np.ndarray:
    vis = frame_bgr.copy()
    if det is not None:
        if det.aruco_ids is not None and len(det.aruco_ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, det.aruco_corners, det.aruco_ids)
        if det.charuco_corners is not None and det.charuco_ids is not None:
            try:
                cv2.aruco.drawDetectedCornersCharuco(
                    vis,
                    det.charuco_corners,
                    det.charuco_ids,
                    (255, 255, 0),
                )
            except Exception:
                pts = np.asarray(det.charuco_corners, dtype=np.float64).reshape(-1, 2)
                for u, v in pts:
                    cv2.circle(vis, (int(round(u)), int(round(v))), 4, (255, 255, 0), 2)

    if pose is not None:
        _draw_axes(vis, K, dist, pose.rvec_cb, pose.tvec_cb_mm)

    color = (0, 210, 255) if recording else ((0, 255, 0) if pose is not None else (0, 0, 255))
    found = 0 if det is None else int(det.num_charuco)
    status = (
        "RECORDING table position (SPACE stores it)"
        if recording
        else "READY (SPACE records this table position)"
    )
    lines = [
        status,
        f"ChArUco corners: {found}/{calib_camera.MAX_CHARUCO_CORNERS}",
        (
            f"positions: {len(positions)}/{target_positions} "
            f"(finish ENTER after {min_positions}+)"
        ),
        f"current position frames: {len(current_frames)}/{min_frames_per_position}+",
        "Move board to another table spot between positions; keep it flat.",
        "Keys: SPACE start/store | ENTER finish | Q/ESC abort",
    ]
    if pose is not None:
        lines.append(
            f"live reproj mean={pose.mean_px:.3f}px p95={pose.p95_px:.3f}px"
        )
    if len(positions) >= min_positions and not recording:
        lines.append("Enough positions collected; ENTER accepts calibration.")
    return _draw_text_box(vis, lines, color=color)


def capture_charuco_table_calibration_from_pipeline(
    pipeline,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    min_positions: int = CHARUCO_TABLE_MIN_POSITIONS,
    target_positions: int = CHARUCO_TABLE_TARGET_POSITIONS,
    min_frames_per_position: int = CHARUCO_TABLE_MIN_FRAMES_PER_POSITION,
    max_frames_per_position: int = CHARUCO_TABLE_MAX_FRAMES_PER_POSITION,
    capture_interval_s: float = DEFAULT_CAPTURE_INTERVAL_S,
    window_name: str = CHARUCO_TABLE_WINDOW_NAME,
    min_charuco_corners: int = CHARUCO_TABLE_MIN_CORNERS,
    mad_scale: float = DEFAULT_FRAME_OUTLIER_MAD_SCALE,
    status_callback: StatusCallback | None = None,
) -> CharucoTableCalibration | None:
    board, aruco_dict, detector_params = calib_camera.make_charuco_board()
    positions: list[CharucoTablePosition] = []
    current_frames: list[CharucoTableFramePose] = []
    recording = False
    last_capture_s = -float("inf")
    last_frame: np.ndarray | None = None
    frame_index = 0

    def store_current_position() -> bool:
        nonlocal current_frames, recording
        if len(current_frames) < int(min_frames_per_position):
            return False
        position = _charuco_position_from_frames(
            current_frames,
            position_index=len(positions) + 1,
            mad_scale=mad_scale,
        )
        positions.append(position)
        if status_callback is not None:
            status_callback(
                f"charuco_table_position:{len(positions)}:"
                f"frames={len(position.used_frames)}:"
                f"rms={position.median_rms_px:.3f}"
            )
        current_frames = []
        recording = False
        return True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        if status_callback is not None:
            status_callback("charuco_table_capture_ready")

        while True:
            frame = wait_color_frame_bgr(pipeline)
            got_new_frame = frame is not None
            if frame is None:
                frame = last_frame
            if frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return None
                continue

            if got_new_frame:
                frame_index += 1
            last_frame = frame

            pose, det = estimate_charuco_table_frame_pose(
                frame,
                K,
                dist,
                board=board,
                aruco_dict=aruco_dict,
                detector_params=detector_params,
                frame_index=frame_index,
                min_charuco_corners=min_charuco_corners,
            )

            now_s = time.monotonic()
            if (
                recording
                and got_new_frame
                and pose is not None
                and len(current_frames) < int(max_frames_per_position)
                and now_s - last_capture_s >= float(capture_interval_s)
            ):
                current_frames.append(pose)
                last_capture_s = now_s
                if status_callback is not None:
                    status_callback(f"charuco_table_frames:{len(current_frames)}")

            if recording and len(current_frames) >= int(max_frames_per_position):
                store_current_position()

            vis = draw_charuco_table_capture_overlay(
                frame,
                K,
                dist,
                pose,
                det,
                positions,
                current_frames,
                recording=recording,
                min_positions=min_positions,
                target_positions=target_positions,
                min_frames_per_position=min_frames_per_position,
            )
            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                return None

            if key == 32:
                if not recording:
                    recording = True
                    current_frames = []
                    last_capture_s = -float("inf")
                    if status_callback is not None:
                        status_callback("charuco_table_position_started")
                elif not store_current_position():
                    recording = True
                    if status_callback is not None:
                        status_callback("charuco_table_position_needs_more_frames")

            if key in (10, 13) and not recording and len(positions) >= int(min_positions):
                if status_callback is not None:
                    status_callback("charuco_table_capture_finished")
                break

        return _build_charuco_table_calibration(
            positions,
            mad_scale=mad_scale,
        )
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass
        cv2.waitKey(1)


def capture_charuco_table_calibration(
    K: np.ndarray,
    dist: np.ndarray,
    *,
    camera_config: Any = None,
    camera_backend: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    serial: str | None = None,
    min_positions: int = CHARUCO_TABLE_MIN_POSITIONS,
    target_positions: int = CHARUCO_TABLE_TARGET_POSITIONS,
    min_frames_per_position: int = CHARUCO_TABLE_MIN_FRAMES_PER_POSITION,
    max_frames_per_position: int = CHARUCO_TABLE_MAX_FRAMES_PER_POSITION,
    capture_interval_s: float = DEFAULT_CAPTURE_INTERVAL_S,
    window_name: str = CHARUCO_TABLE_WINDOW_NAME,
    min_charuco_corners: int = CHARUCO_TABLE_MIN_CORNERS,
    mad_scale: float = DEFAULT_FRAME_OUTLIER_MAD_SCALE,
    status_callback: StatusCallback | None = None,
) -> CharucoTableCalibration | None:
    camera = None
    try:
        if camera_config is None:
            camera_config = calib_camera.make_live_camera_config(
                backend=camera_backend,
                width=width,
                height=height,
                fps=fps,
                serial=serial,
            )
        else:
            camera_config = calib_camera.without_tracking_calibration(camera_config)
        camera = calib_camera.start_camera_source(camera_config)
        return capture_charuco_table_calibration_from_camera_source(
            camera,
            K,
            dist,
            min_positions=min_positions,
            target_positions=target_positions,
            min_frames_per_position=min_frames_per_position,
            max_frames_per_position=max_frames_per_position,
            capture_interval_s=capture_interval_s,
            window_name=window_name,
            min_charuco_corners=min_charuco_corners,
            mad_scale=mad_scale,
            status_callback=status_callback,
        )
    finally:
        if camera is not None:
            camera.stop()


def align_detection_to_reference(corners: np.ndarray, reference: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1, 2)
    direct = float(np.mean(np.linalg.norm(corners - reference, axis=1)))
    reverse = float(np.mean(np.linalg.norm(corners[::-1] - reference, axis=1)))
    return corners[::-1] if reverse < direct else corners


def capture_checkerboard_detections_from_pipeline(
    pipeline,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
    capture_interval_s: float = DEFAULT_CAPTURE_INTERVAL_S,
    window_name: str = WINDOW_NAME,
    pattern: tuple[int, int] = CHECKERBOARD_PATTERN,
    square_size_mm: float = CHECKERBOARD_SQUARE_SIZE_MM,
    status_callback: StatusCallback | None = None,
) -> list[np.ndarray]:
    detections: list[np.ndarray] = []
    reference: np.ndarray | None = None
    recording = False
    last_capture_s = -float("inf")
    last_frame: np.ndarray | None = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        if status_callback is not None:
            status_callback("checkerboard_capture_ready")

        while True:
            frame = wait_color_frame_bgr(pipeline)
            got_new_frame = frame is not None
            if frame is None:
                frame = last_frame
            if frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    detections = []
                    break
                continue
            last_frame = frame

            corners = detect_checkerboard_corners(frame, pattern=pattern)
            if corners is not None and reference is not None:
                corners = align_detection_to_reference(corners, reference)

            now_s = time.monotonic()
            if (
                recording
                and got_new_frame
                and corners is not None
                and len(detections) < max_frames
                and now_s - last_capture_s >= capture_interval_s
            ):
                if reference is None:
                    reference = corners
                corners = align_detection_to_reference(corners, reference)
                detections.append(corners)
                last_capture_s = now_s
                if status_callback is not None:
                    status_callback(f"checkerboard_detections:{len(detections)}")

            vis = draw_capture_overlay(
                frame,
                K,
                dist,
                corners,
                detections,
                recording=recording,
                pattern=pattern,
                square_size_mm=square_size_mm,
            )
            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                detections = []
                break

            if key == 32:
                if not recording:
                    recording = True
                    detections = []
                    reference = None
                    last_capture_s = -float("inf")
                    if status_callback is not None:
                        status_callback("checkerboard_capture_started")
                else:
                    recording = False
                    if len(detections) >= min_frames:
                        if status_callback is not None:
                            status_callback("checkerboard_capture_finished")
                        break
                    recording = True
                    if status_callback is not None:
                        status_callback("checkerboard_capture_needs_more_frames")

            if len(detections) >= max_frames:
                if status_callback is not None:
                    status_callback("checkerboard_capture_reached_max_frames")
                break

    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass
        cv2.waitKey(1)

    return detections


def capture_checkerboard_detections(
    K: np.ndarray,
    dist: np.ndarray,
    *,
    camera_config: Any = None,
    camera_backend: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    serial: str | None = None,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
    capture_interval_s: float = DEFAULT_CAPTURE_INTERVAL_S,
    window_name: str = WINDOW_NAME,
    pattern: tuple[int, int] = CHECKERBOARD_PATTERN,
    square_size_mm: float = CHECKERBOARD_SQUARE_SIZE_MM,
    status_callback: StatusCallback | None = None,
) -> list[np.ndarray]:
    camera = None
    try:
        if camera_config is None:
            camera_config = calib_camera.make_live_camera_config(
                backend=camera_backend,
                width=width,
                height=height,
                fps=fps,
                serial=serial,
            )
        else:
            camera_config = calib_camera.without_tracking_calibration(camera_config)
        camera = calib_camera.start_camera_source(camera_config)
        return capture_checkerboard_detections_from_camera_source(
            camera,
            K,
            dist,
            min_frames=min_frames,
            max_frames=max_frames,
            capture_interval_s=capture_interval_s,
            window_name=window_name,
            pattern=pattern,
            square_size_mm=square_size_mm,
            status_callback=status_callback,
        )
    finally:
        if camera is not None:
            camera.stop()


def capture_checkerboard_pose_from_pipeline(
    pipeline,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
    capture_interval_s: float = DEFAULT_CAPTURE_INTERVAL_S,
    window_name: str = WINDOW_NAME,
    pattern: tuple[int, int] = CHECKERBOARD_PATTERN,
    square_size_mm: float = CHECKERBOARD_SQUARE_SIZE_MM,
    mad_scale: float = DEFAULT_FRAME_OUTLIER_MAD_SCALE,
    status_callback: StatusCallback | None = None,
) -> CheckerboardPose | None:
    detections = capture_checkerboard_detections_from_pipeline(
        pipeline,
        K,
        dist,
        min_frames=min_frames,
        max_frames=max_frames,
        capture_interval_s=capture_interval_s,
        window_name=window_name,
        pattern=pattern,
        square_size_mm=square_size_mm,
        status_callback=status_callback,
    )
    if len(detections) < min_frames:
        return None
    return estimate_pose_from_detections(
        detections,
        K,
        dist,
        pattern=pattern,
        square_size_mm=square_size_mm,
        mad_scale=mad_scale,
    )


def capture_checkerboard_pose(
    K: np.ndarray,
    dist: np.ndarray,
    *,
    camera_config: Any = None,
    camera_backend: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    serial: str | None = None,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
    capture_interval_s: float = DEFAULT_CAPTURE_INTERVAL_S,
    window_name: str = WINDOW_NAME,
    pattern: tuple[int, int] = CHECKERBOARD_PATTERN,
    square_size_mm: float = CHECKERBOARD_SQUARE_SIZE_MM,
    mad_scale: float = DEFAULT_FRAME_OUTLIER_MAD_SCALE,
    status_callback: StatusCallback | None = None,
) -> CheckerboardPose | None:
    detections = capture_checkerboard_detections(
        K,
        dist,
        camera_config=camera_config,
        camera_backend=camera_backend,
        width=width,
        height=height,
        fps=fps,
        serial=serial,
        min_frames=min_frames,
        max_frames=max_frames,
        capture_interval_s=capture_interval_s,
        window_name=window_name,
        pattern=pattern,
        square_size_mm=square_size_mm,
        status_callback=status_callback,
    )
    if len(detections) < min_frames:
        return None
    return estimate_pose_from_detections(
        detections,
        K,
        dist,
        pattern=pattern,
        square_size_mm=square_size_mm,
        mad_scale=mad_scale,
    )


def pose_quality_dict(pose: CheckerboardPose | CharucoTableCalibration) -> dict[str, float | int]:
    kept_errors = pose.frame_rms_px[pose.frame_mask]
    all_errors = pose.all_errors_px
    corner_std = np.asarray(getattr(pose, "corner_std_px", kept_errors), dtype=np.float64)
    return {
        "frames_total": int(pose.frame_mask.size),
        "frames_used": int(np.count_nonzero(pose.frame_mask)),
        "frame_rms_mean_px": float(np.mean(kept_errors)),
        "frame_rms_median_px": float(np.median(kept_errors)),
        "frame_rms_p95_px": float(np.percentile(kept_errors, 95)),
        "corner_error_mean_px": float(np.mean(all_errors)),
        "corner_error_median_px": float(np.median(all_errors)),
        "corner_error_p95_px": float(np.percentile(all_errors, 95)),
        "corner_error_max_px": float(np.max(all_errors)),
        "corner_std_mean_px": float(np.mean(corner_std)),
        "corner_std_max_px": float(np.max(corner_std)),
        "pnp_flag": int(getattr(pose, "pnp_flag", cv2.SOLVEPNP_ITERATIVE)),
        "solver_mode": str(getattr(pose, "solver_mode", "unknown")),
        "candidate_count": int(getattr(pose, "candidate_count", 0) or 0),
        "selected_candidate_index": getattr(pose, "selected_candidate_index", None),
        "alternative_rms_px": float(getattr(pose, "alternative_rms_px", float("nan"))),
        "alternative_error_gap_px": float(getattr(pose, "alternative_error_gap_px", float("nan"))),
        "alternative_error_ratio": float(getattr(pose, "alternative_error_ratio", float("nan"))),
        "alternative_likelihood_ratio": float(
            getattr(pose, "alternative_likelihood_ratio", float("nan"))
        ),
        "alternative_translation_delta_mm": float(
            getattr(pose, "alternative_translation_delta_mm", float("nan"))
        ),
        "alternative_rotation_delta_deg": float(
            getattr(pose, "alternative_rotation_delta_deg", float("nan"))
        ),
        "pose_ambiguous": bool(getattr(pose, "pose_ambiguous", False)),
    }
