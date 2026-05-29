from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tracking.pose_solvers import (
    make_transform_from_rvec_tvec,
)


@dataclass
class PoseTrackPoint:
    global_row: int
    global_col: int

    xyz_mm: Tuple[float, float, float]
    uv: Tuple[float, float]

    votes: int = 0


@dataclass
class MapPoseResult:
    success: bool
    message: str

    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    T_marker_camera: Optional[np.ndarray] = None

    inlier_indices: Optional[np.ndarray] = None

    reprojection_mean_px: float = -1.0
    reprojection_max_px: float = -1.0

    num_points: int = 0
    num_inliers: int = 0

    points: Optional[List[PoseTrackPoint]] = None


@dataclass
class MapPoseTrackerConfig:
    min_points: int = 8
    min_inliers: int = 6

    ransac_reproj_px: float = 3.0
    ransac_confidence: float = 0.99
    ransac_iterations: int = 500

    max_mean_reproj_px: float = 4.0
    max_max_reproj_px: float = 15.0

    max_translation_jump_mm: float = 120.0
    max_rotation_jump_deg: float = 45.0

    use_pose_prior: bool = True


class MapPoseTracker:
    """
    Robust pose tracker using ONLY:
        global IDs
        + 2D observations
        + previous pose prior

    No persistent local checkerboard semantics.
    """

    def __init__(
        self,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        config: Optional[MapPoseTrackerConfig] = None,
    ) -> None:
        self.config = config or MapPoseTrackerConfig()

        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)

        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((0, 1), dtype=np.float64)
        else:
            self.dist_coeffs = (
                np.asarray(dist_coeffs, dtype=np.float64)
                .reshape(-1, 1)
            )

        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None
        self.T_marker_camera: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.rvec = None
        self.tvec = None
        self.T_marker_camera = None

    def estimate_pose(
        self,
        points: List[PoseTrackPoint],
    ) -> MapPoseResult:

        if len(points) < self.config.min_points:
            return MapPoseResult(
                success=False,
                message=(
                    f"Too few points: "
                    f"{len(points)} < {self.config.min_points}"
                ),
                num_points=len(points),
                points=[],
            )

        object_points = np.asarray(
            [p.xyz_mm for p in points],
            dtype=np.float64,
        ).reshape(-1, 3)

        image_points = np.asarray(
            [p.uv for p in points],
            dtype=np.float64,
        ).reshape(-1, 2)

        use_guess = (
            self.config.use_pose_prior
            and self.rvec is not None
            and self.tvec is not None
        )

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                self.K,
                self.dist_coeffs,
                rvec=(
                    self.rvec.copy()
                    if use_guess
                    else None
                ),
                tvec=(
                    self.tvec.copy()
                    if use_guess
                    else None
                ),
                useExtrinsicGuess=bool(use_guess),
                iterationsCount=int(
                    self.config.ransac_iterations
                ),
                reprojectionError=float(
                    self.config.ransac_reproj_px
                ),
                confidence=float(
                    self.config.ransac_confidence
                ),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        except Exception as e:
            return MapPoseResult(
                success=False,
                message=f"solvePnPRansac failed: {e}",
                num_points=len(points),
                points=[],
            )

        if (
            not success
            or inliers is None
            or len(inliers) < self.config.min_inliers
        ):
            return MapPoseResult(
                success=False,
                message=(
                    f"Too few inliers: "
                    f"{0 if inliers is None else len(inliers)}"
                ),
                num_points=len(points),
                num_inliers=(
                    0 if inliers is None else len(inliers)
                ),
                points=[],
            )

        inlier_idx = (
            np.asarray(inliers, dtype=np.int64)
            .reshape(-1)
        )

        object_inliers = object_points[inlier_idx]
        image_inliers = image_points[inlier_idx]

        projected, _ = cv2.projectPoints(
            object_inliers,
            rvec,
            tvec,
            self.K,
            self.dist_coeffs,
        )

        projected = projected.reshape(-1, 2)

        reproj_errors = np.linalg.norm(
            projected - image_inliers,
            axis=1,
        )

        mean_err = float(np.mean(reproj_errors))
        max_err = float(np.max(reproj_errors))

        if (
            mean_err > self.config.max_mean_reproj_px
            or max_err > self.config.max_max_reproj_px
        ):
            return MapPoseResult(
                success=False,
                message=(
                    f"Reprojection error too high "
                    f"(mean={mean_err:.3f}, "
                    f"max={max_err:.3f})"
                ),
                rvec=rvec,
                tvec=tvec,
                T_marker_camera=(
                    make_transform_from_rvec_tvec(
                        rvec,
                        tvec,
                    )
                ),
                reprojection_mean_px=mean_err,
                reprojection_max_px=max_err,
                num_points=len(points),
                num_inliers=len(inlier_idx),
                points=[],
            )

        if (
            self.rvec is not None
            and self.tvec is not None
        ):
            accepted, reason = self._check_motion_gate(
                rvec,
                tvec,
            )

            if not accepted:
                return MapPoseResult(
                    success=False,
                    message=(
                        f"Motion gate rejected pose: "
                        f"{reason}"
                    ),
                    rvec=rvec,
                    tvec=tvec,
                    T_marker_camera=(
                        make_transform_from_rvec_tvec(
                            rvec,
                            tvec,
                        )
                    ),
                    reprojection_mean_px=mean_err,
                    reprojection_max_px=max_err,
                    num_points=len(points),
                    num_inliers=len(inlier_idx),
                    points=[],
                )

        selected_points = [
            points[int(i)]
            for i in inlier_idx
        ]

        T = make_transform_from_rvec_tvec(
            rvec,
            tvec,
        )

        self.rvec = (
            np.asarray(rvec, dtype=np.float64)
            .reshape(3, 1)
        )

        self.tvec = (
            np.asarray(tvec, dtype=np.float64)
            .reshape(3, 1)
        )

        self.T_marker_camera = (
            np.asarray(T, dtype=np.float64)
            .reshape(4, 4)
        )

        return MapPoseResult(
            success=True,
            message="Pose estimation successful.",

            rvec=self.rvec.copy(),
            tvec=self.tvec.copy(),
            T_marker_camera=self.T_marker_camera.copy(),

            inlier_indices=inlier_idx.copy(),

            reprojection_mean_px=mean_err,
            reprojection_max_px=max_err,

            num_points=len(points),
            num_inliers=len(inlier_idx),

            points=selected_points,
        )

    def _check_motion_gate(
        self,
        candidate_rvec: np.ndarray,
        candidate_tvec: np.ndarray,
    ) -> Tuple[bool, str]:

        prev_R, _ = cv2.Rodrigues(self.rvec)
        cand_R, _ = cv2.Rodrigues(candidate_rvec)

        dR = cand_R @ prev_R.T

        trace = np.trace(dR)
        trace = np.clip(
            (trace - 1.0) * 0.5,
            -1.0,
            1.0,
        )

        angle_deg = float(
            np.degrees(np.arccos(trace))
        )

        translation_mm = float(
            np.linalg.norm(
                candidate_tvec.reshape(3)
                - self.tvec.reshape(3)
            )
        )

        if (
            angle_deg
            > self.config.max_rotation_jump_deg
        ):
            return (
                False,
                (
                    f"Rotation jump too large: "
                    f"{angle_deg:.2f} deg > "
                    f"{self.config.max_rotation_jump_deg:.2f} deg"
                ),
            )

        if (
            translation_mm
            > self.config.max_translation_jump_mm
        ):
            return (
                False,
                (
                    f"Translation jump too large: "
                    f"{translation_mm:.2f} mm > "
                    f"{self.config.max_translation_jump_mm:.2f} mm"
                ),
            )

        return True, ""