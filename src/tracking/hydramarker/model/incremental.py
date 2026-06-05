from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np

from tracking.hydramarker.model.observations import FrameObservation
from tracking.hydramarker.model.state import CameraPose, SfMState


@dataclass(slots=True)
class FrameRegistrationResult:
    success: bool
    frame_id: int
    message: str

    num_matches: int = 0
    num_inliers: int = 0

    mean_reprojection_error_px: float = float("nan")
    median_reprojection_error_px: float = float("nan")
    max_reprojection_error_px: float = float("nan")

    marker_ids: Optional[np.ndarray] = None
    inlier_mask: Optional[np.ndarray] = None


def _as_rvec_tvec(pose: CameraPose) -> tuple[np.ndarray, np.ndarray]:
    rvec, _ = cv2.Rodrigues(pose.R)
    tvec = pose.t.reshape(3, 1)
    return rvec.astype(np.float64), tvec.astype(np.float64)


def _pose_from_rvec_tvec(
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> CameraPose:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)

    return CameraPose(R=R, t=t)


def compute_reprojection_errors_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    pose: CameraPose,
    state: SfMState,
) -> np.ndarray:
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

    rvec, tvec = _as_rvec_tvec(pose)

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        state.calibration.K,
        state.calibration.dist_coeffs,
    )

    projected = projected.reshape(-1, 2)

    return np.linalg.norm(projected - image_points, axis=1)


def register_frame_pnp(
    state: SfMState,
    frame: FrameObservation,
    *,
    min_points: int = 12,
    ransac_reprojection_error_px: float = 1.5,
    confidence: float = 0.999,
    iterations_count: int = 200,
    refine_lm: bool = True,
    max_mean_reprojection_error_px: Optional[float] = None,
) -> FrameRegistrationResult:
    marker_ids, object_points, image_points = state.known_observations_in_frame(frame)

    num_matches = int(len(marker_ids))

    if num_matches < min_points:
        return FrameRegistrationResult(
            success=False,
            frame_id=frame.frame_id,
            message=f"Too few known 2D-3D matches: {num_matches} < {min_points}.",
            num_matches=num_matches,
            marker_ids=marker_ids,
        )

    object_points = object_points.astype(np.float64).reshape(-1, 3)
    image_points = image_points.astype(np.float64).reshape(-1, 2)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=state.calibration.K,
        distCoeffs=state.calibration.dist_coeffs,
        iterationsCount=iterations_count,
        reprojectionError=ransac_reprojection_error_px,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not success or rvec is None or tvec is None or inliers is None:
        return FrameRegistrationResult(
            success=False,
            frame_id=frame.frame_id,
            message="solvePnPRansac failed.",
            num_matches=num_matches,
            marker_ids=marker_ids,
        )

    inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)
    num_inliers = int(len(inliers))

    if num_inliers < min_points:
        return FrameRegistrationResult(
            success=False,
            frame_id=frame.frame_id,
            message=f"Too few PnP inliers: {num_inliers} < {min_points}.",
            num_matches=num_matches,
            num_inliers=num_inliers,
            marker_ids=marker_ids,
        )

    inlier_object_points = object_points[inliers]
    inlier_image_points = image_points[inliers]

    if refine_lm:
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=inlier_object_points,
            imagePoints=inlier_image_points,
            cameraMatrix=state.calibration.K,
            distCoeffs=state.calibration.dist_coeffs,
            rvec=rvec,
            tvec=tvec,
        )

    pose = _pose_from_rvec_tvec(rvec, tvec)

    errors = compute_reprojection_errors_px(
        inlier_object_points,
        inlier_image_points,
        pose,
        state,
    )

    mean_error = float(np.mean(errors))
    median_error = float(np.median(errors))
    max_error = float(np.max(errors))

    if (
        max_mean_reprojection_error_px is not None
        and mean_error > max_mean_reprojection_error_px
    ):
        return FrameRegistrationResult(
            success=False,
            frame_id=frame.frame_id,
            message=(
                f"Mean reprojection error too high: "
                f"{mean_error:.3f}px > {max_mean_reprojection_error_px:.3f}px."
            ),
            num_matches=num_matches,
            num_inliers=num_inliers,
            mean_reprojection_error_px=mean_error,
            median_reprojection_error_px=median_error,
            max_reprojection_error_px=max_error,
            marker_ids=marker_ids,
        )

    state.add_pose(frame.frame_id, pose)

    inlier_mask = np.zeros(num_matches, dtype=bool)
    inlier_mask[inliers] = True

    return FrameRegistrationResult(
        success=True,
        frame_id=frame.frame_id,
        message="Frame registered.",
        num_matches=num_matches,
        num_inliers=num_inliers,
        mean_reprojection_error_px=mean_error,
        median_reprojection_error_px=median_error,
        max_reprojection_error_px=max_error,
        marker_ids=marker_ids,
        inlier_mask=inlier_mask,
    )


def register_remaining_frames(
    state: SfMState,
    *,
    min_points: int = 12,
    ransac_reprojection_error_px: float = 1.5,
    confidence: float = 0.999,
    iterations_count: int = 200,
    refine_lm: bool = True,
    max_mean_reprojection_error_px: Optional[float] = None,
) -> list[FrameRegistrationResult]:
    results: list[FrameRegistrationResult] = []

    for frame in state.unposed_frames():
        result = register_frame_pnp(
            state,
            frame,
            min_points=min_points,
            ransac_reprojection_error_px=ransac_reprojection_error_px,
            confidence=confidence,
            iterations_count=iterations_count,
            refine_lm=refine_lm,
            max_mean_reprojection_error_px=max_mean_reprojection_error_px,
        )

        results.append(result)

    return results


@dataclass(slots=True)
class MarkerTriangulationResult:
    marker_id: int
    success: bool
    message: str

    num_observations: int = 0
    num_inliers: int = 0

    mean_reprojection_error_px: float = float("nan")
    median_reprojection_error_px: float = float("nan")
    max_reprojection_error_px: float = float("nan")


def _projection_matrix_normalized(pose: CameraPose) -> np.ndarray:
    return np.hstack(
        [
            np.asarray(pose.R, dtype=np.float64).reshape(3, 3),
            np.asarray(pose.t, dtype=np.float64).reshape(3, 1),
        ]
    )


def _triangulate_multiview_linear(
    poses: list[CameraPose],
    normalized_points: np.ndarray,
) -> Optional[np.ndarray]:
    if len(poses) < 2:
        return None

    normalized_points = np.asarray(
        normalized_points,
        dtype=np.float64,
    ).reshape(-1, 2)

    rows = []

    for pose, (x, y) in zip(poses, normalized_points, strict=False):
        P = _projection_matrix_normalized(pose)
        rows.append(x * P[2, :] - P[0, :])
        rows.append(y * P[2, :] - P[1, :])

    A = np.asarray(rows, dtype=np.float64)

    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None

    X = vt[-1, :]

    if not np.isfinite(X).all() or abs(float(X[3])) < 1e-12:
        return None

    point = X[:3] / X[3]

    if not np.isfinite(point).all():
        return None

    return point.astype(np.float64)


def _reprojection_errors_for_point(
    point: np.ndarray,
    poses: list[CameraPose],
    image_points: np.ndarray,
    state: SfMState,
) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(point, dtype=np.float64).reshape(1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

    projected_all = []
    depths = []

    for pose in poses:
        rvec, tvec = _as_rvec_tvec(pose)
        projected, _ = cv2.projectPoints(
            point,
            rvec,
            tvec,
            state.calibration.K,
            state.calibration.dist_coeffs,
        )
        projected_all.append(projected.reshape(2))

        point_cam = pose.R @ point.reshape(3) + pose.t
        depths.append(float(point_cam[2]))

    projected_np = np.asarray(projected_all, dtype=np.float64).reshape(-1, 2)
    errors = np.linalg.norm(projected_np - image_points, axis=1)

    return errors, np.asarray(depths, dtype=np.float64)


def _normalized_reprojection_errors_for_point(
    point: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    normalized_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    rotations = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    translations = np.asarray(translations, dtype=np.float64).reshape(-1, 3)
    normalized_points = np.asarray(
        normalized_points,
        dtype=np.float64,
    ).reshape(-1, 2)

    points_cam = np.einsum("nij,j->ni", rotations, point) + translations
    depths = points_cam[:, 2]

    projected = np.full((len(points_cam), 2), np.nan, dtype=np.float64)
    valid_depth = depths > 1e-12
    projected[valid_depth] = (
        points_cam[valid_depth, :2] / depths[valid_depth, None]
    )

    errors = np.linalg.norm(projected - normalized_points, axis=1)

    return errors, depths


def _evenly_sample_observations(
    frame_observations: list[tuple[int, np.ndarray]],
    max_observations: int,
) -> list[tuple[int, np.ndarray]]:
    if max_observations <= 0 or len(frame_observations) <= max_observations:
        return frame_observations

    indices = np.linspace(
        0,
        len(frame_observations) - 1,
        int(max_observations),
    )
    indices = np.unique(np.rint(indices).astype(np.int64))

    return [frame_observations[int(i)] for i in indices]


def _candidate_pair_indices(
    n: int,
    max_pairs: int,
) -> list[tuple[int, int]]:
    if n < 2:
        return []

    pairs: list[tuple[int, int]] = []

    # Prefer large temporal baselines first. They triangulate boundary points
    # much better than adjacent frames and avoid an O(n^2) sweep.
    for gap in range(n - 1, 0, -1):
        for i in range(0, n - gap):
            pairs.append((i, i + gap))
            if len(pairs) >= max_pairs:
                return pairs

    return pairs


def _robust_triangulate_marker(
    state: SfMState,
    marker_id: int,
    frame_observations: list[tuple[int, np.ndarray]],
    *,
    max_reprojection_error_px: float,
    min_inliers: int,
    max_observations: int,
    max_pair_candidates: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    frame_observations = _evenly_sample_observations(
        frame_observations,
        max_observations=max_observations,
    )

    poses = []
    image_points = []

    for frame_id, uv in frame_observations:
        pose = state.poses.get(int(frame_id))
        if pose is None:
            continue

        poses.append(pose)
        image_points.append(uv)

    if len(poses) < min_inliers:
        return None, None, None

    image_points_np = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    normalized = cv2.undistortPoints(
        image_points_np.reshape(-1, 1, 2),
        state.calibration.K,
        state.calibration.dist_coeffs,
    ).reshape(-1, 2)

    rotations = np.asarray([pose.R for pose in poses], dtype=np.float64)
    translations = np.asarray([pose.t for pose in poses], dtype=np.float64)

    focal = float(
        max(
            1e-6,
            0.5
            * (
                float(state.calibration.K[0, 0])
                + float(state.calibration.K[1, 1])
            ),
        )
    )
    max_reprojection_error_norm = max_reprojection_error_px / focal

    best_point = None
    best_mask = None
    best_count = -1
    best_median = float("inf")

    n = len(poses)

    for i, j in _candidate_pair_indices(n, max_pair_candidates):
        candidate = _triangulate_multiview_linear(
            [poses[i], poses[j]],
            normalized[[i, j]],
        )

        if candidate is None:
            continue

        errors_norm, depths = _normalized_reprojection_errors_for_point(
            candidate,
            rotations,
            translations,
            normalized,
        )

        mask = (
            np.isfinite(errors_norm)
            & np.isfinite(depths)
            & (depths > 0.0)
            & (errors_norm <= max_reprojection_error_norm)
        )

        count = int(np.count_nonzero(mask))

        if count <= 0:
            continue

        median = float(np.median(errors_norm[mask]))

        if count > best_count or (count == best_count and median < best_median):
            best_point = candidate
            best_mask = mask
            best_count = count
            best_median = median

    if best_point is None or best_mask is None or best_count < min_inliers:
        return None, None, None

    refined = _triangulate_multiview_linear(
        [pose for pose, keep in zip(poses, best_mask, strict=False) if keep],
        normalized[best_mask],
    )

    if refined is None:
        return None, None, None

    errors_norm, depths = _normalized_reprojection_errors_for_point(
        refined,
        rotations,
        translations,
        normalized,
    )

    final_mask = (
        np.isfinite(errors_norm)
        & np.isfinite(depths)
        & (depths > 0.0)
        & (errors_norm <= max_reprojection_error_norm)
    )

    if int(np.count_nonzero(final_mask)) < min_inliers:
        return None, None, None

    errors, _ = _reprojection_errors_for_point(
        refined,
        poses,
        image_points_np,
        state,
    )

    return refined, errors, final_mask


def triangulate_missing_markers(
    state: SfMState,
    *,
    min_observations: int = 8,
    min_inliers: int = 6,
    max_reprojection_error_px: float = 2.5,
    max_observations_per_marker: int = 48,
    max_pair_candidates_per_marker: int = 220,
) -> list[MarkerTriangulationResult]:
    observations_by_marker: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)

    for frame in state.frames:
        if not state.has_pose(frame.frame_id):
            continue

        for marker_id, obs in frame.observations.items():
            if int(marker_id) in state.marker_positions:
                continue

            observations_by_marker[int(marker_id)].append(
                (
                    int(frame.frame_id),
                    np.asarray(obs.uv, dtype=np.float64).reshape(2),
                )
            )

    results: list[MarkerTriangulationResult] = []

    for marker_id in sorted(observations_by_marker):
        frame_observations = observations_by_marker[marker_id]
        num_observations = int(len(frame_observations))

        if num_observations < min_observations:
            results.append(
                MarkerTriangulationResult(
                    marker_id=int(marker_id),
                    success=False,
                    message=(
                        f"Too few posed observations: "
                        f"{num_observations} < {min_observations}."
                    ),
                    num_observations=num_observations,
                )
            )
            continue

        point, errors, inlier_mask = _robust_triangulate_marker(
            state,
            marker_id,
            frame_observations,
            max_reprojection_error_px=max_reprojection_error_px,
            min_inliers=min_inliers,
            max_observations=max_observations_per_marker,
            max_pair_candidates=max_pair_candidates_per_marker,
        )

        if point is None or errors is None or inlier_mask is None:
            results.append(
                MarkerTriangulationResult(
                    marker_id=int(marker_id),
                    success=False,
                    message="Could not triangulate with enough reprojection inliers.",
                    num_observations=num_observations,
                )
            )
            continue

        inlier_errors = errors[inlier_mask]
        num_inliers = int(np.count_nonzero(inlier_mask))

        state.add_marker_position(marker_id, point)

        results.append(
            MarkerTriangulationResult(
                marker_id=int(marker_id),
                success=True,
                message="Marker triangulated.",
                num_observations=num_observations,
                num_inliers=num_inliers,
                mean_reprojection_error_px=float(np.mean(inlier_errors)),
                median_reprojection_error_px=float(np.median(inlier_errors)),
                max_reprojection_error_px=float(np.max(inlier_errors)),
            )
        )

    return results
