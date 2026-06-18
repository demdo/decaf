"""Checkerboard pose helper for HydraMarker.

This module does not load camera models, print results, or write files. Callers
provide OpenCV camera intrinsics/distortion and receive the estimated pose.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Sequence

import cv2
import numpy as np

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

WINDOW_NAME = "HydraMarker Checkerboard Pose Calibration"

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
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                80,
                1e-4,
            )
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

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


def wait_color_frame_bgr(pipeline) -> np.ndarray | None:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data()).copy()


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
    width: int = REALSENSE_WIDTH,
    height: int = REALSENSE_HEIGHT,
    fps: int = REALSENSE_FPS,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
    capture_interval_s: float = DEFAULT_CAPTURE_INTERVAL_S,
    window_name: str = WINDOW_NAME,
    pattern: tuple[int, int] = CHECKERBOARD_PATTERN,
    square_size_mm: float = CHECKERBOARD_SQUARE_SIZE_MM,
    status_callback: StatusCallback | None = None,
) -> list[np.ndarray]:
    pipeline = None
    try:
        pipeline, _profile = start_realsense_color_stream(
            width=int(width),
            height=int(height),
            fps=int(fps),
        )
        return capture_checkerboard_detections_from_pipeline(
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
    finally:
        if pipeline is not None:
            pipeline.stop()


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
    width: int = REALSENSE_WIDTH,
    height: int = REALSENSE_HEIGHT,
    fps: int = REALSENSE_FPS,
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
        width=width,
        height=height,
        fps=fps,
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


def pose_quality_dict(pose: CheckerboardPose) -> dict[str, float | int]:
    kept_errors = pose.frame_rms_px[pose.frame_mask]
    all_errors = pose.all_errors_px
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
        "corner_std_mean_px": float(np.mean(pose.corner_std_px)),
        "corner_std_max_px": float(np.max(pose.corner_std_px)),
        "pnp_flag": int(pose.pnp_flag),
        "solver_mode": pose.solver_mode,
        "candidate_count": int(pose.candidate_count),
        "selected_candidate_index": pose.selected_candidate_index,
        "alternative_rms_px": float(pose.alternative_rms_px),
        "alternative_error_gap_px": float(pose.alternative_error_gap_px),
        "alternative_error_ratio": float(pose.alternative_error_ratio),
        "alternative_likelihood_ratio": float(pose.alternative_likelihood_ratio),
        "alternative_translation_delta_mm": float(pose.alternative_translation_delta_mm),
        "alternative_rotation_delta_deg": float(pose.alternative_rotation_delta_deg),
        "pose_ambiguous": bool(pose.pose_ambiguous),
    }
