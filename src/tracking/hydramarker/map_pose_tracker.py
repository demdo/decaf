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
    method: str = ""


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

    # Adaptiver Motion Gate:
    # Threshold wächst um diesen Wert pro verlorenem Frame.
    # Beispiel: 8.0 -> nach 5 Frames: 45 + 40 = 85 deg
    rotation_gate_scale_per_lost_frame: float = 8.0

    # Absolutes Maximum, unabhaengig von lost_frames.
    rotation_gate_max_deg: float = 120.0

    use_pose_prior: bool = True
    refine_with_iterative: bool = True
    use_direct_prior_solver: bool = True
    direct_refine_method: str = "lm"
    direct_max_mean_reproj_px: float = 1.5
    direct_max_max_reproj_px: float = 3.0


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
        lost_frames: int = 0,
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

        if use_guess and self.config.use_direct_prior_solver:
            direct = self._estimate_pose_direct_prior(
                points=points,
                object_points=object_points,
                image_points=image_points,
                lost_frames=lost_frames,
            )
            if direct is not None and direct.success:
                return direct

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
                method="ransac_iterative",
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
                method="ransac_iterative",
            )

        inlier_idx = (
            np.asarray(inliers, dtype=np.int64)
            .reshape(-1)
        )

        object_inliers = object_points[inlier_idx]
        image_inliers = image_points[inlier_idx]

        if self.config.refine_with_iterative:
            try:
                refine_success, rvec_ref, tvec_ref = cv2.solvePnP(
                    object_inliers,
                    image_inliers,
                    self.K,
                    self.dist_coeffs,
                    rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                    tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if refine_success:
                    rvec = np.asarray(rvec_ref, dtype=np.float64).reshape(3, 1)
                    tvec = np.asarray(tvec_ref, dtype=np.float64).reshape(3, 1)
            except Exception:
                pass

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
            # Bei sehr hohem Fehler (>3x) ist die Pose strukturell falsch.
            # Prior löschen damit PnP im nächsten Frame neu startet.
            if mean_err > self.config.max_mean_reproj_px * 3.0:
                self.rvec = None
                self.tvec = None
                self.T_marker_camera = None
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
                method="ransac_iterative",
            )

        if (
            self.rvec is not None
            and self.tvec is not None
        ):
            accepted, reason = self._check_motion_gate(
                rvec,
                tvec,
                lost_frames=lost_frames,
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
                    method="ransac_iterative",
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
            method="ransac_iterative",
        )

    def _estimate_pose_direct_prior(
        self,
        *,
        points: List[PoseTrackPoint],
        object_points: np.ndarray,
        image_points: np.ndarray,
        lost_frames: int,
    ) -> Optional[MapPoseResult]:
        if self.rvec is None or self.tvec is None:
            return None

        if len(points) < self.config.min_inliers:
            return None

        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.K,
                self.dist_coeffs,
                rvec=np.asarray(self.rvec, dtype=np.float64).reshape(3, 1).copy(),
                tvec=np.asarray(self.tvec, dtype=np.float64).reshape(3, 1).copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            return None

        if not success:
            return None

        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

        rvec, tvec, method = self._refine_direct_prior_pose(
            object_points,
            image_points,
            rvec,
            tvec,
        )

        try:
            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                self.K,
                self.dist_coeffs,
            )
        except Exception:
            return None

        projected = projected.reshape(-1, 2)
        reproj_errors = np.linalg.norm(projected - image_points, axis=1)
        mean_err = float(np.mean(reproj_errors))
        max_err = float(np.max(reproj_errors))

        if (
            mean_err > self.config.direct_max_mean_reproj_px
            or max_err > self.config.direct_max_max_reproj_px
        ):
            return None

        accepted, _ = self._check_motion_gate(
            rvec,
            tvec,
            lost_frames=lost_frames,
        )
        if not accepted:
            return None

        T = make_transform_from_rvec_tvec(rvec, tvec)
        self.rvec = rvec.copy()
        self.tvec = tvec.copy()
        self.T_marker_camera = np.asarray(T, dtype=np.float64).reshape(4, 4)
        inlier_idx = np.arange(len(points), dtype=np.int64)

        return MapPoseResult(
            success=True,
            message="Direct prior pose estimation successful.",
            rvec=self.rvec.copy(),
            tvec=self.tvec.copy(),
            T_marker_camera=self.T_marker_camera.copy(),
            inlier_indices=inlier_idx.copy(),
            reprojection_mean_px=mean_err,
            reprojection_max_px=max_err,
            num_points=len(points),
            num_inliers=len(points),
            points=list(points),
            method=method,
        )

    def _refine_direct_prior_pose(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        configured = str(self.config.direct_refine_method or "none").lower()
        if configured in ("", "none", "off", "false"):
            return rvec, tvec, "direct_prior_unrefined"

        if configured == "auto":
            methods = ("lm", "vvs")
        elif configured in ("lm", "vvs"):
            methods = (configured,)
        else:
            methods = ("lm",)

        for method in methods:
            if method == "lm" and hasattr(cv2, "solvePnPRefineLM"):
                try:
                    refined = cv2.solvePnPRefineLM(
                        object_points,
                        image_points,
                        self.K,
                        self.dist_coeffs,
                        rvec.copy(),
                        tvec.copy(),
                    )
                    if refined is not None:
                        rvec_ref, tvec_ref = refined[:2]
                        return (
                            np.asarray(rvec_ref, dtype=np.float64).reshape(3, 1),
                            np.asarray(tvec_ref, dtype=np.float64).reshape(3, 1),
                            "direct_prior_lm",
                        )
                except Exception:
                    pass

            if method == "vvs" and hasattr(cv2, "solvePnPRefineVVS"):
                try:
                    refined = cv2.solvePnPRefineVVS(
                        object_points,
                        image_points,
                        self.K,
                        self.dist_coeffs,
                        rvec.copy(),
                        tvec.copy(),
                    )
                    if refined is not None:
                        rvec_ref, tvec_ref = refined[:2]
                        return (
                            np.asarray(rvec_ref, dtype=np.float64).reshape(3, 1),
                            np.asarray(tvec_ref, dtype=np.float64).reshape(3, 1),
                            "direct_prior_vvs",
                        )
                except Exception:
                    pass

        return rvec, tvec, "direct_prior_iterative"

    def _check_motion_gate(
        self,
        candidate_rvec: np.ndarray,
        candidate_tvec: np.ndarray,
        lost_frames: int = 0,
    ) -> Tuple[bool, str]:

        # Rotations-Threshold skaliert mit Outage-Laenge.
        # Nach 0 verlorenen Frames: max_rotation_jump_deg (z.B. 45 deg)
        # Nach 5 verlorenen Frames: 45 + 5*8 = 85 deg
        # Gedeckelt bei rotation_gate_max_deg (z.B. 120 deg)
        effective_rotation_limit = min(
            self.config.max_rotation_jump_deg
            + lost_frames * self.config.rotation_gate_scale_per_lost_frame,
            self.config.rotation_gate_max_deg,
        )

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

        if angle_deg > effective_rotation_limit:
            return (
                False,
                (
                    f"Rotation jump too large: "
                    f"{angle_deg:.2f} deg > "
                    f"{effective_rotation_limit:.2f} deg"
                    f" (lost_frames={lost_frames})"
                ),
            )

        if translation_mm > self.config.max_translation_jump_mm:
            return (
                False,
                (
                    f"Translation jump too large: "
                    f"{translation_mm:.2f} mm > "
                    f"{self.config.max_translation_jump_mm:.2f} mm"
                ),
            )

        return True, ""
