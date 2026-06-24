from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from tracking.hydramarker.model.observations import FrameObservation


@dataclass(slots=True)
class CameraCalibration:
    K: np.ndarray
    dist_coeffs: np.ndarray

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1, 1)


@dataclass(slots=True)
class BootstrapPair:
    frame_a: FrameObservation
    frame_b: FrameObservation
    shared_ids: list[int]
    score: float


@dataclass(slots=True)
class BootstrapResult:
    success: bool
    message: str

    frame_a_id: int = -1
    frame_b_id: int = -1

    marker_ids: np.ndarray | None = None
    points_3d: np.ndarray | None = None

    R_ba: np.ndarray | None = None
    t_ba: np.ndarray | None = None

    pts_a_norm: np.ndarray | None = None
    pts_b_norm: np.ndarray | None = None

    reprojection_errors: np.ndarray | None = None
    depths_a: np.ndarray | None = None
    depths_b: np.ndarray | None = None

    num_valid_cols: int = 0
    num_valid_rows: int = 0
    valid_col_span: int = 0
    valid_row_span: int = 0
    median_triangulation_angle_deg: float = float("nan")


def _frame_points_for_ids(
    frame: FrameObservation,
    ids: list[int],
) -> np.ndarray:
    return np.asarray(
        [frame.observations[mid].uv for mid in ids],
        dtype=np.float64,
    ).reshape(-1, 2)


def _coverage_metrics(
    marker_ids: np.ndarray | list[int],
    *,
    id_num_cols: int | None,
) -> tuple[int, int, int, int]:
    if id_num_cols is None or int(id_num_cols) <= 0:
        return 0, 0, 0, 0

    ids = np.asarray(
        marker_ids,
        dtype=np.int64,
    ).reshape(-1)

    if ids.size == 0:
        return 0, 0, 0, 0

    rows = ids // int(id_num_cols)
    cols = ids % int(id_num_cols)

    unique_rows = np.unique(rows)
    unique_cols = np.unique(cols)

    row_span = int(np.max(unique_rows) - np.min(unique_rows) + 1)
    col_span = int(np.max(unique_cols) - np.min(unique_cols) + 1)

    return (
        int(unique_cols.size),
        int(unique_rows.size),
        col_span,
        row_span,
    )


def _shared_id_score(
    shared_ids: list[int],
    *,
    gap: int,
    id_num_cols: int | None,
) -> float:
    num_cols, num_rows, col_span, row_span = _coverage_metrics(
        shared_ids,
        id_num_cols=id_num_cols,
    )

    return (
        1000.0 * float(len(shared_ids))
        + 50.0 * float(num_cols)
        + 10.0 * float(col_span)
        + 5.0 * float(num_rows)
        + float(row_span)
        + 0.001 * float(gap)
    )


def triangulation_angles_deg(
    points_a: np.ndarray,
    R_ba: np.ndarray,
    t_ba: np.ndarray,
) -> np.ndarray:
    points_a = np.asarray(
        points_a,
        dtype=np.float64,
    ).reshape(-1, 3)

    if points_a.size == 0:
        return np.empty((0,), dtype=np.float64)

    R_ba = np.asarray(R_ba, dtype=np.float64).reshape(3, 3)
    t_ba = np.asarray(t_ba, dtype=np.float64).reshape(3)

    camera_a_center = np.zeros(3, dtype=np.float64)
    camera_b_center = -R_ba.T @ t_ba

    rays_a = points_a - camera_a_center.reshape(1, 3)
    rays_b = points_a - camera_b_center.reshape(1, 3)

    norm_a = np.linalg.norm(rays_a, axis=1)
    norm_b = np.linalg.norm(rays_b, axis=1)
    valid = (
        np.isfinite(rays_a).all(axis=1)
        & np.isfinite(rays_b).all(axis=1)
        & (norm_a > 1e-12)
        & (norm_b > 1e-12)
    )

    angles = np.full(
        points_a.shape[0],
        np.nan,
        dtype=np.float64,
    )

    if np.any(valid):
        dots = np.sum(
            rays_a[valid] * rays_b[valid],
            axis=1,
        ) / (
            norm_a[valid]
            * norm_b[valid]
        )
        dots = np.clip(dots, -1.0, 1.0)
        angles[valid] = np.degrees(np.arccos(dots))

    return angles


def undistort_points_normalized(
    uv: np.ndarray,
    calib: CameraCalibration,
) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)

    pts = cv2.undistortPoints(
        uv,
        calib.K,
        calib.dist_coeffs,
    )

    return pts.reshape(-1, 2)


def select_bootstrap_pair(
    frames: list[FrameObservation],
    *,
    min_shared_ids: int = 20,
    max_pairs: int | None = 2000,
    min_frame_gap: int = 5,
    id_num_cols: int | None = None,
) -> BootstrapPair:
    if len(frames) < 2:
        raise ValueError("Need at least two frames for SfM bootstrap.")

    best: Optional[BootstrapPair] = None
    tested = 0

    for i, frame_a in enumerate(frames):
        for j in range(i + 1, len(frames)):
            frame_b = frames[j]

            gap = abs(frame_b.frame_id - frame_a.frame_id)
            if gap < min_frame_gap:
                continue

            shared = frame_a.shared_ids(frame_b)
            n_shared = len(shared)

            if n_shared < min_shared_ids:
                continue

            score = _shared_id_score(
                shared,
                gap=gap,
                id_num_cols=id_num_cols,
            )

            if best is None or score > best.score:
                best = BootstrapPair(
                    frame_a=frame_a,
                    frame_b=frame_b,
                    shared_ids=shared,
                    score=score,
                )

            tested += 1
            if max_pairs is not None and max_pairs > 0 and tested >= max_pairs:
                break

        if max_pairs is not None and max_pairs > 0 and tested >= max_pairs:
            break

    if best is None:
        raise RuntimeError(
            f"Could not find bootstrap pair with at least {min_shared_ids} shared IDs."
        )

    return best


def select_bootstrap_pair_candidates(
    frames: list[FrameObservation],
    *,
    min_shared_ids: int = 20,
    max_pairs: int | None = 2000,
    min_frame_gap: int = 5,
    id_num_cols: int | None = None,
) -> list[BootstrapPair]:
    if len(frames) < 2:
        raise ValueError("Need at least two frames for SfM bootstrap.")

    candidates: list[BootstrapPair] = []
    tested = 0

    for i, frame_a in enumerate(frames):
        for j in range(i + 1, len(frames)):
            frame_b = frames[j]

            gap = abs(frame_b.frame_id - frame_a.frame_id)
            if gap < min_frame_gap:
                continue

            shared = frame_a.shared_ids(frame_b)
            n_shared = len(shared)

            if n_shared < min_shared_ids:
                continue

            candidates.append(
                BootstrapPair(
                    frame_a=frame_a,
                    frame_b=frame_b,
                    shared_ids=shared,
                    score=_shared_id_score(
                        shared,
                        gap=gap,
                        id_num_cols=id_num_cols,
                    ),
                )
            )

            tested += 1
            if max_pairs is not None and max_pairs > 0 and tested >= max_pairs:
                break

        if max_pairs is not None and max_pairs > 0 and tested >= max_pairs:
            break

    candidates.sort(key=lambda pair: pair.score, reverse=True)
    return candidates


def estimate_relative_pose(
    frame_a: FrameObservation,
    frame_b: FrameObservation,
    shared_ids: list[int],
    calib: CameraCalibration,
    *,
    ransac_threshold_norm: float = 1e-3,
    confidence: float = 0.999,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts_a_px = _frame_points_for_ids(frame_a, shared_ids)
    pts_b_px = _frame_points_for_ids(frame_b, shared_ids)

    pts_a_norm = undistort_points_normalized(pts_a_px, calib)
    pts_b_norm = undistort_points_normalized(pts_b_px, calib)

    E, mask = cv2.findEssentialMat(
        pts_a_norm,
        pts_b_norm,
        focal=1.0,
        pp=(0.0, 0.0),
        method=cv2.RANSAC,
        prob=confidence,
        threshold=ransac_threshold_norm,
    )

    if E is None or mask is None:
        raise RuntimeError("cv2.findEssentialMat failed.")

    _, R_ba, t_ba, pose_mask = cv2.recoverPose(
        E,
        pts_a_norm,
        pts_b_norm,
        focal=1.0,
        pp=(0.0, 0.0),
        mask=mask,
    )

    inlier_mask = pose_mask.reshape(-1).astype(bool)

    if int(np.count_nonzero(inlier_mask)) < 8:
        raise RuntimeError(
            f"Too few recoverPose inliers: {np.count_nonzero(inlier_mask)}."
        )

    marker_ids = np.asarray(shared_ids, dtype=np.int64)[inlier_mask]

    return (
        R_ba.astype(np.float64),
        t_ba.reshape(3).astype(np.float64),
        marker_ids,
        pts_a_norm[inlier_mask],
        pts_b_norm[inlier_mask],
    )


def triangulate_two_view(
    marker_ids: np.ndarray,
    pts_a_norm: np.ndarray,
    pts_b_norm: np.ndarray,
    R_ba: np.ndarray,
    t_ba: np.ndarray,
    *,
    max_reprojection_error_norm: float = 2e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    marker_ids = np.asarray(marker_ids, dtype=np.int64).reshape(-1)
    pts_a_norm = np.asarray(pts_a_norm, dtype=np.float64).reshape(-1, 2)
    pts_b_norm = np.asarray(pts_b_norm, dtype=np.float64).reshape(-1, 2)
    R_ba = np.asarray(R_ba, dtype=np.float64).reshape(3, 3)
    t_ba = np.asarray(t_ba, dtype=np.float64).reshape(3, 1)

    P_a = np.hstack(
        [
            np.eye(3, dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
        ]
    )

    P_b = np.hstack([R_ba, t_ba])

    homog = cv2.triangulatePoints(
        P_a,
        P_b,
        pts_a_norm.T,
        pts_b_norm.T,
    )

    points = cv2.convertPointsFromHomogeneous(homog.T).reshape(-1, 3)

    depths_a = points[:, 2]
    points_b = (R_ba @ points.T + t_ba).T
    depths_b = points_b[:, 2]

    proj_a = points[:, :2] / depths_a[:, None]
    proj_b = points_b[:, :2] / depths_b[:, None]

    err_a = np.linalg.norm(proj_a - pts_a_norm, axis=1)
    err_b = np.linalg.norm(proj_b - pts_b_norm, axis=1)
    reproj = np.maximum(err_a, err_b)

    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(reproj)
        & (depths_a > 0.0)
        & (depths_b > 0.0)
        & (reproj <= max_reprojection_error_norm)
    )

    return (
        marker_ids[valid],
        points[valid],
        depths_a[valid],
        depths_b[valid],
        reproj[valid],
    )


def run_bootstrap(
    frames: list[FrameObservation],
    calib: CameraCalibration,
    *,
    min_shared_ids: int = 20,
    min_frame_gap: int = 5,
    max_pairs: int | None = 2000,
    max_candidate_pairs_to_try: int = 50,
    id_num_cols: int | None = None,
    ransac_threshold_norm: float = 1e-3,
    max_reprojection_error_norm: float = 2e-3,
) -> BootstrapResult:
    try:
        candidates = select_bootstrap_pair_candidates(
            frames,
            min_shared_ids=min_shared_ids,
            min_frame_gap=min_frame_gap,
            max_pairs=max_pairs,
            id_num_cols=id_num_cols,
        )

        if not candidates:
            raise RuntimeError(
                f"Could not find bootstrap pair with at least {min_shared_ids} shared IDs."
            )

        best = None
        failures = []

        for pair in candidates[: max(1, int(max_candidate_pairs_to_try))]:
            try:
                R_ba, t_ba, marker_ids, pts_a_norm, pts_b_norm = estimate_relative_pose(
                    pair.frame_a,
                    pair.frame_b,
                    pair.shared_ids,
                    calib,
                    ransac_threshold_norm=ransac_threshold_norm,
                )

                ids_valid, points_valid, depths_a, depths_b, reproj = triangulate_two_view(
                    marker_ids,
                    pts_a_norm,
                    pts_b_norm,
                    R_ba,
                    t_ba,
                    max_reprojection_error_norm=max_reprojection_error_norm,
                )
            except Exception as exc:
                failures.append(
                    f"{pair.frame_a.frame_id}-{pair.frame_b.frame_id}: {exc}"
                )
                continue

            median_reproj = (
                float(np.median(reproj))
                if len(reproj) > 0
                else float("inf")
            )
            num_cols, num_rows, col_span, row_span = _coverage_metrics(
                ids_valid,
                id_num_cols=id_num_cols,
            )
            angles = triangulation_angles_deg(
                points_valid,
                R_ba,
                t_ba,
            )
            finite_angles = angles[np.isfinite(angles)]
            median_angle = (
                float(np.median(finite_angles))
                if finite_angles.size > 0
                else float("-inf")
            )
            candidate_score = (
                int(len(ids_valid)),
                int(num_cols),
                int(col_span),
                int(num_rows),
                int(row_span),
                float(median_angle),
                -median_reproj,
                int(len(pair.shared_ids)),
                float(pair.score),
            )

            if best is None or candidate_score > best[0]:
                best = (
                    candidate_score,
                    pair,
                    R_ba,
                    t_ba,
                    marker_ids,
                    pts_a_norm,
                    pts_b_norm,
                    ids_valid,
                    points_valid,
                    depths_a,
                    depths_b,
                    reproj,
                    num_cols,
                    num_rows,
                    col_span,
                    row_span,
                    median_angle,
                )

        if best is None:
            detail = "; ".join(failures[:5])
            return BootstrapResult(
                success=False,
                message=f"Bootstrap failed for all candidate pairs. {detail}",
            )

        (
            _,
            pair,
            R_ba,
            t_ba,
            marker_ids,
            pts_a_norm,
            pts_b_norm,
            ids_valid,
            points_valid,
            depths_a,
            depths_b,
            reproj,
            num_cols,
            num_rows,
            col_span,
            row_span,
            median_angle,
        ) = best

        if len(ids_valid) < 8:
            return BootstrapResult(
                success=False,
                message=f"Best bootstrap triangulated too few valid points: {len(ids_valid)}.",
                frame_a_id=pair.frame_a.frame_id,
                frame_b_id=pair.frame_b.frame_id,
            )

        return BootstrapResult(
            success=True,
            message="Bootstrap successful.",
            frame_a_id=pair.frame_a.frame_id,
            frame_b_id=pair.frame_b.frame_id,
            marker_ids=ids_valid,
            points_3d=points_valid,
            R_ba=R_ba,
            t_ba=t_ba,
            pts_a_norm=pts_a_norm,
            pts_b_norm=pts_b_norm,
            reprojection_errors=reproj,
            depths_a=depths_a,
            depths_b=depths_b,
            num_valid_cols=num_cols,
            num_valid_rows=num_rows,
            valid_col_span=col_span,
            valid_row_span=row_span,
            median_triangulation_angle_deg=median_angle,
        )

    except Exception as exc:
        return BootstrapResult(
            success=False,
            message=f"Bootstrap failed: {exc}",
        )
