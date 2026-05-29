from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from overlay.tracking.hydramarker.model.observations import FrameObservation


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


def _frame_points_for_ids(
    frame: FrameObservation,
    ids: list[int],
) -> np.ndarray:
    return np.asarray(
        [frame.observations[mid].uv for mid in ids],
        dtype=np.float64,
    ).reshape(-1, 2)


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
    max_pairs: int = 2000,
    min_frame_gap: int = 5,
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

            score = float(n_shared) + 0.01 * float(gap)

            if best is None or score > best.score:
                best = BootstrapPair(
                    frame_a=frame_a,
                    frame_b=frame_b,
                    shared_ids=shared,
                    score=score,
                )

            tested += 1
            if tested >= max_pairs:
                break

        if tested >= max_pairs:
            break

    if best is None:
        raise RuntimeError(
            f"Could not find bootstrap pair with at least {min_shared_ids} shared IDs."
        )

    return best


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
    ransac_threshold_norm: float = 1e-3,
    max_reprojection_error_norm: float = 2e-3,
) -> BootstrapResult:
    try:
        pair = select_bootstrap_pair(
            frames,
            min_shared_ids=min_shared_ids,
            min_frame_gap=min_frame_gap,
        )

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

        if len(ids_valid) < 8:
            return BootstrapResult(
                success=False,
                message=f"Bootstrap triangulated too few valid points: {len(ids_valid)}.",
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
        )

    except Exception as exc:
        return BootstrapResult(
            success=False,
            message=f"Bootstrap failed: {exc}",
        )
