from __future__ import annotations

from dataclasses import dataclass
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
