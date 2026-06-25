"""Pose solving, pose result packaging, and pose fallback strategies."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tracking.hydramarker.tracker_types import (
    FastPathDebug,
    PoseSource,
    TrackerCorner,
    TrackerMode,
    TrackerResult,
)
from tracking.pose_filters import PoseDepthFilterResult
from tracking.pose_prior import PlateauPosePriorResult, solve_plateau_pose_prior
from tracking.pose_solvers import make_transform_from_rvec_tvec


@dataclass
class PoseTrackPoint:
    """Single 3D/2D correspondence used by pose estimation."""

    global_row: int
    global_col: int

    xyz_mm: Tuple[float, float, float]
    uv: Tuple[float, float]

    votes: int = 0


@dataclass
class MapPoseResult:
    """Internal result returned by the map-based PnP pose tracker."""

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
    """PnP solver thresholds and motion gates for map-based pose tracking."""

    min_points: int = 8
    min_inliers: int = 6

    ransac_reproj_px: float = 3.0
    ransac_confidence: float = 0.99
    ransac_iterations: int = 500

    max_mean_reproj_px: float = 4.0
    max_max_reproj_px: float = 15.0

    max_translation_jump_mm: float = 40.0
    max_rotation_jump_deg: float = 45.0

    # Adaptiver Motion Gate:
    # Threshold wächst um diesen Wert pro verlorenem Frame.
    # Beispiel: 8.0 -> nach 5 Frames: 45 + 40 = 85 deg
    rotation_gate_scale_per_lost_frame: float = 8.0

    # Absolutes Maximum, unabhaengig von lost_frames.
    rotation_gate_max_deg: float = 90.0

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
        """Forget the current pose prior."""
        self.rvec = None
        self.tvec = None
        self.T_marker_camera = None

    def estimate_pose(
        self,
        points: List[PoseTrackPoint],
        lost_frames: int = 0,
    ) -> MapPoseResult:
        """Estimate marker pose from global correspondences and update the pose prior."""

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
        """Try a fast iterative solve seeded by the previous accepted pose."""
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
        """Apply the configured OpenCV pose refinement method to a prior solve."""
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
        """Reject pose jumps that exceed configured translation or rotation limits."""

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


class PoseEstimationMixin:
    """Package internal PnP results into public tracker results."""

    def _apply_depth_filter_to_pose(
        self,
        pose: MapPoseResult,
        fallback_points: List[PoseTrackPoint],
    ) -> PoseDepthFilterResult | None:
        """Apply the configured camera-Z stabilizer to an accepted pose."""
        if not bool(getattr(self.config, "pose_depth_filter_enabled", False)):
            return None
        if pose.rvec is None or pose.tvec is None:
            return None

        points = list(pose.points or fallback_points or [])
        min_points = max(1, int(getattr(self.config, "pose_depth_filter_min_points", 6)))
        if len(points) < min_points:
            return None

        object_points = np.asarray([p.xyz_mm for p in points], dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray([p.uv for p in points], dtype=np.float64).reshape(-1, 2)
        filtered = self.pose_depth_filter.update(
            rvec=pose.rvec,
            tvec=pose.tvec,
            object_points=object_points,
            image_points=image_points,
        )

        pose.rvec = filtered.rvec.copy()
        pose.tvec = filtered.tvec.copy()
        pose.T_marker_camera = filtered.T_marker_camera.copy()
        errors = self._reprojection_errors_for_pose(
            object_points,
            image_points,
            filtered.rvec,
            filtered.tvec,
        )
        if errors is not None and len(errors):
            pose.reprojection_mean_px = float(np.mean(errors))
            pose.reprojection_max_px = float(np.max(errors))

        self.pose_tracker.rvec = filtered.rvec.copy()
        self.pose_tracker.tvec = filtered.tvec.copy()
        self.pose_tracker.T_marker_camera = filtered.T_marker_camera.copy()
        return filtered

    @staticmethod
    def _depth_filter_kwargs(
        filtered: PoseDepthFilterResult | None,
    ) -> dict:
        if filtered is None:
            return {}
        return {
            "depth_filter_applied": bool(filtered.applied),
            "depth_filter_delta_z_mm": float(filtered.delta_z_mm),
            "depth_filter_raw_z_mm": float(filtered.raw_z_mm),
            "depth_filter_z_mm": float(filtered.filtered_z_mm),
            "depth_filter_reproj_excess_px": float(filtered.reprojection_excess_px),
            "depth_filter_guard_alpha": float(filtered.guard_alpha),
            "depth_filter_innovation_z_mm": float(filtered.innovation_z_mm),
            "depth_filter_innovation_mean_z_mm": float(filtered.innovation_mean_z_mm),
            "depth_filter_innovation_cusum_pos_mm": float(filtered.innovation_cusum_pos_mm),
            "depth_filter_innovation_cusum_neg_mm": float(filtered.innovation_cusum_neg_mm),
            "depth_filter_innovation_bias_detected": bool(filtered.innovation_bias_detected),
            "depth_filter_innovation_bias_direction": int(filtered.innovation_bias_direction),
            "depth_filter_innovation_bias_limited": bool(filtered.innovation_bias_limited),
            "depth_filter_object_z_span_mm": float(filtered.object_z_span_mm),
            "depth_filter_negative_delta_guard_limited": bool(
                filtered.negative_delta_guard_limited
            ),
        }

    @staticmethod
    def _plateau_prior_kwargs(
        prior: PlateauPosePriorResult | None,
        *,
        triggered: bool = False,
        attempted: bool = False,
        applied: bool = False,
    ) -> dict:
        if prior is None:
            if not triggered and not attempted and not applied:
                return {}
            return {
                "pose_plateau_prior_triggered": bool(triggered),
                "pose_plateau_prior_attempted": bool(attempted),
                "pose_plateau_prior_applied": bool(applied),
            }
        return {
            "pose_plateau_prior_triggered": bool(triggered),
            "pose_plateau_prior_attempted": bool(attempted),
            "pose_plateau_prior_applied": bool(applied and prior.success),
            "pose_plateau_prior_method": str(prior.method),
            "pose_plateau_prior_reason": str(prior.reason),
            "pose_plateau_prior_delta_z_mm": float(prior.delta_z_mm),
            "pose_plateau_prior_reproj_excess_px": float(prior.reprojection_excess_px),
            "pose_plateau_prior_max_reproj_excess_px": float(
                prior.max_reprojection_excess_px
            ),
            "pose_plateau_prior_iterations": int(prior.iterations),
        }

    @staticmethod
    def _clone_map_pose_result(pose: MapPoseResult) -> MapPoseResult:
        return MapPoseResult(
            success=bool(pose.success),
            message=str(pose.message),
            rvec=None if pose.rvec is None else np.asarray(pose.rvec, dtype=np.float64).reshape(3, 1).copy(),
            tvec=None if pose.tvec is None else np.asarray(pose.tvec, dtype=np.float64).reshape(3, 1).copy(),
            T_marker_camera=(
                None
                if pose.T_marker_camera is None
                else np.asarray(pose.T_marker_camera, dtype=np.float64).reshape(4, 4).copy()
            ),
            inlier_indices=(
                None
                if pose.inlier_indices is None
                else np.asarray(pose.inlier_indices, dtype=np.int64).reshape(-1).copy()
            ),
            reprojection_mean_px=float(pose.reprojection_mean_px),
            reprojection_max_px=float(pose.reprojection_max_px),
            num_points=int(pose.num_points),
            num_inliers=int(pose.num_inliers),
            points=None if pose.points is None else list(pose.points),
            method=str(pose.method),
        )

    def _pose_plateau_prior_triggered(
        self,
        filtered: PoseDepthFilterResult | None,
    ) -> bool:
        if filtered is None:
            return False
        if not bool(getattr(self.config, "pose_plateau_prior_enabled", False)):
            return False
        delta_limit = float(
            getattr(self.config, "pose_plateau_prior_trigger_negative_delta_mm", 0.0)
        )
        if float(filtered.delta_z_mm) >= delta_limit:
            return False
        min_z_span = float(
            getattr(self.config, "pose_plateau_prior_min_object_z_span_mm", 14.835)
        )
        if not np.isfinite(filtered.object_z_span_mm):
            return False
        return float(filtered.object_z_span_mm) >= min_z_span

    def _maybe_apply_plateau_pose_prior(
        self,
        raw_pose: MapPoseResult,
        fallback_points: List[PoseTrackPoint],
        filtered: PoseDepthFilterResult | None,
        prev_rvec: Optional[np.ndarray],
        prev_tvec: Optional[np.ndarray],
    ) -> tuple[MapPoseResult | None, PlateauPosePriorResult | None, bool, bool]:
        """Try a second-stage pose when the depth filter sees a Z plateau."""
        triggered = self._pose_plateau_prior_triggered(filtered)
        if not triggered:
            return None, None, False, False

        if prev_rvec is None or prev_tvec is None:
            prior = PlateauPosePriorResult(False, "none", "missing_previous_pose")
            return None, prior, True, False
        if raw_pose.rvec is None or raw_pose.tvec is None:
            prior = PlateauPosePriorResult(False, "none", "missing_raw_pose")
            return None, prior, True, False

        points = list(raw_pose.points or fallback_points or [])
        min_points = max(1, int(getattr(self.config, "pose_plateau_prior_min_points", 6)))
        if len(points) < min_points:
            prior = PlateauPosePriorResult(False, "none", "too_few_points")
            return None, prior, True, False

        object_points = np.asarray([p.xyz_mm for p in points], dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray([p.uv for p in points], dtype=np.float64).reshape(-1, 2)
        prior = solve_plateau_pose_prior(
            object_points=object_points,
            image_points=image_points,
            K=self.K,
            dist_coeffs=self.dist_coeffs,
            raw_rvec=raw_pose.rvec,
            raw_tvec=raw_pose.tvec,
            seed_rvec=prev_rvec,
            seed_tvec=prev_tvec,
            static_max_excess_px=float(
                getattr(self.config, "pose_plateau_prior_static_max_excess_px", 0.18)
            ),
            candidate_max_excess_px=float(
                getattr(self.config, "pose_plateau_prior_candidate_max_excess_px", 0.25)
            ),
            candidate_max_max_excess_px=float(
                getattr(self.config, "pose_plateau_prior_candidate_max_max_excess_px", 1.00)
            ),
            min_positive_z_correction_mm=float(
                getattr(self.config, "pose_plateau_prior_min_positive_z_correction_mm", 0.0)
            ),
            max_positive_z_correction_mm=float(
                getattr(self.config, "pose_plateau_prior_max_positive_z_correction_mm", 0.75)
            ),
            robust_c_px=float(getattr(self.config, "pose_plateau_prior_robust_c_px", 0.20)),
            max_iterations=int(getattr(self.config, "pose_plateau_prior_max_iterations", 6)),
            max_step_translation_mm=float(
                getattr(self.config, "pose_plateau_prior_max_step_translation_mm", 5.0)
            ),
            max_step_rotation_deg=float(
                getattr(self.config, "pose_plateau_prior_max_step_rotation_deg", 5.0)
            ),
            lm_damping=float(getattr(self.config, "pose_plateau_prior_lm_damping", 1.0e-5)),
        )
        if not prior.success or prior.rvec is None or prior.tvec is None:
            return None, prior, True, False

        pose = self._clone_map_pose_result(raw_pose)
        pose.rvec = prior.rvec.copy()
        pose.tvec = prior.tvec.copy()
        pose.T_marker_camera = (
            None
            if prior.T_marker_camera is None
            else prior.T_marker_camera.copy()
        )
        pose.reprojection_mean_px = float(prior.reprojection_mean_px)
        pose.reprojection_max_px = float(prior.reprojection_max_px)
        pose.method = f"{raw_pose.method}+plateau_prior_{prior.method}"
        pose.message = "Pose refined by triggered plateau prior."
        return pose, prior, True, True

    def _decode_update_rejection_reason(
        self,
        visual_corners: List[TrackerCorner],
    ) -> str:
        """Return why a decoded pose is too weak to refresh tracker state."""
        min_visual = max(0, int(self.config.decode_update_min_visual_corners))
        if len(visual_corners) < min_visual:
            return (
                "Decode pose rejected by low visual coverage "
                f"({len(visual_corners)}/{min_visual} visible corners)."
            )

        min_rows = max(0, int(self.config.decode_update_min_distinct_rows))
        min_cols = max(0, int(self.config.decode_update_min_distinct_cols))
        distinct_rows = len({int(c.global_row) for c in visual_corners})
        distinct_cols = len({int(c.global_col) for c in visual_corners})

        if distinct_rows < min_rows or distinct_cols < min_cols:
            return (
                "Decode pose rejected by narrow marker coverage "
                f"(rows={distinct_rows}/{min_rows}, cols={distinct_cols}/{min_cols})."
            )

        return ""

    def _estimate_and_package_pose(
        self,
        track_points: List[PoseTrackPoint],
        tracker_corners: List[TrackerCorner],
        success_message: str,
        update_persistence: bool,
        pose_source: PoseSource,
        detection=None,
        precomputed_pose: Optional[MapPoseResult] = None,
        precomputed_pnp_ms: Optional[float] = None,
        previous_pose_rvec: Optional[np.ndarray] = None,
        previous_pose_tvec: Optional[np.ndarray] = None,
        previous_pose_T: Optional[np.ndarray] = None,
        previous_depth_filter_state: Optional[tuple] = None,
        previous_last_rvec: Optional[np.ndarray] = None,
        previous_last_tvec: Optional[np.ndarray] = None,
    ) -> TrackerResult:
        """Run pose estimation, validate fallback poses, and update accepted state."""
        pose_timings: dict[str, float] = {}

        snapshot_t0 = time.perf_counter()
        if precomputed_pose is None:
            prev_pose_rvec = None if self.pose_tracker.rvec is None else self.pose_tracker.rvec.copy()
            prev_pose_tvec = None if self.pose_tracker.tvec is None else self.pose_tracker.tvec.copy()
            prev_pose_T = (
                None
                if self.pose_tracker.T_marker_camera is None
                else self.pose_tracker.T_marker_camera.copy()
            )
            prev_depth_filter_state = self.pose_depth_filter.snapshot()
            prev_last_rvec = (
                None
                if self._last_accepted_rvec is None
                else self._last_accepted_rvec.copy()
            )
            prev_last_tvec = (
                None
                if self._last_accepted_tvec is None
                else self._last_accepted_tvec.copy()
            )
        else:
            prev_pose_rvec = (
                None if previous_pose_rvec is None else previous_pose_rvec.copy()
            )
            prev_pose_tvec = (
                None if previous_pose_tvec is None else previous_pose_tvec.copy()
            )
            prev_pose_T = (
                None if previous_pose_T is None else previous_pose_T.copy()
            )
            prev_depth_filter_state = (
                self.pose_depth_filter.snapshot()
                if previous_depth_filter_state is None
                else previous_depth_filter_state
            )
            prev_last_rvec = (
                None if previous_last_rvec is None else previous_last_rvec.copy()
            )
            prev_last_tvec = (
                None if previous_last_tvec is None else previous_last_tvec.copy()
            )
        pose_timings["pose_state_snapshot_ms"] = (
            time.perf_counter() - snapshot_t0
        ) * 1000.0

        def mark_pose_timing(name: str, start: float) -> None:
            pose_timings[name] = (time.perf_counter() - start) * 1000.0

        if precomputed_pose is None:
            pnp_t0 = time.perf_counter()
            pose = self.pose_tracker.estimate_pose(
                track_points,
                lost_frames=self.lost_frames,
            )
            pnp_ms = (time.perf_counter() - pnp_t0) * 1000.0
        else:
            pose = precomputed_pose
            pnp_ms = 0.0 if precomputed_pnp_ms is None else float(precomputed_pnp_ms)
        pose_timings["pnp_ms"] = pnp_ms

        if not pose.success:
            return TrackerResult(
                success=False,
                mode=self.mode,
                message=pose.message,
                rvec=pose.rvec,
                tvec=pose.tvec,
                T_marker_camera=pose.T_marker_camera,
                mean_reprojection_error_px=pose.reprojection_mean_px,
                max_reprojection_error_px=pose.reprojection_max_px,
                num_points=pose.num_points,
                num_inliers=pose.num_inliers,
                pnp_method=str(getattr(pose, "method", "")),
                corners=[],
                correspondence_corners=tracker_corners,
                timings_ms=dict(pose_timings),
            )

        stage_t0 = time.perf_counter()
        inlier_corners = self._inlier_corners_from_pose(pose, tracker_corners)
        mark_pose_timing("pose_inlier_corners_ms", stage_t0)

        if (
            not update_persistence
            and not self._persistent_pose_motion_plausible(
                pose.rvec,
                pose.tvec,
                prev_last_rvec,
                prev_last_tvec,
            )
        ):
            self.pose_tracker.rvec = prev_pose_rvec
            self.pose_tracker.tvec = prev_pose_tvec
            self.pose_tracker.T_marker_camera = prev_pose_T
            return TrackerResult(
                success=False,
                mode=self.mode,
                message="Persistent pose rejected by motion gate.",
                rvec=pose.rvec,
                tvec=pose.tvec,
                T_marker_camera=pose.T_marker_camera,
                mean_reprojection_error_px=pose.reprojection_mean_px,
                max_reprojection_error_px=pose.reprojection_max_px,
                num_points=pose.num_points,
                num_inliers=pose.num_inliers,
                pnp_method=str(getattr(pose, "method", "")),
                corners=[],
                correspondence_corners=tracker_corners,
                timings_ms=dict(pose_timings),
            )

        if not update_persistence:
            reject_reason = self._fallback_pose_rejection_reason(
                detection,
                pose.rvec,
                pose.tvec,
                pose.reprojection_mean_px,
                pose.reprojection_max_px,
            )
            if reject_reason:
                self.pose_tracker.rvec = prev_pose_rvec
                self.pose_tracker.tvec = prev_pose_tvec
                self.pose_tracker.T_marker_camera = prev_pose_T
                return TrackerResult(
                    success=False,
                    mode=self.mode,
                    message=reject_reason,
                    rvec=pose.rvec,
                    tvec=pose.tvec,
                    T_marker_camera=pose.T_marker_camera,
                    mean_reprojection_error_px=pose.reprojection_mean_px,
                    max_reprojection_error_px=pose.reprojection_max_px,
                    num_points=pose.num_points,
                    num_inliers=pose.num_inliers,
                    pnp_method=str(getattr(pose, "method", "")),
                    corners=[],
                    correspondence_corners=tracker_corners,
                    timings_ms=dict(pose_timings),
                )

        stage_t0 = time.perf_counter()
        raw_pose = self._clone_map_pose_result(pose)
        mark_pose_timing("pose_clone_map_result_ms", stage_t0)
        stage_t0 = time.perf_counter()
        filtered_depth = self._apply_depth_filter_to_pose(pose, track_points)
        mark_pose_timing("pose_depth_filter_pre_ms", stage_t0)

        plateau_prior_t0 = time.perf_counter()
        prior_pose, plateau_prior, plateau_triggered, plateau_applied = (
            self._maybe_apply_plateau_pose_prior(
                raw_pose,
                track_points,
                filtered_depth,
                prev_pose_rvec,
                prev_pose_tvec,
            )
        )
        plateau_prior_ms = (time.perf_counter() - plateau_prior_t0) * 1000.0
        pose_timings["pose_plateau_prior_ms"] = plateau_prior_ms
        if prior_pose is not None:
            self.pose_depth_filter.restore(prev_depth_filter_state)
            pose = prior_pose
            if pose.rvec is not None:
                self.pose_tracker.rvec = pose.rvec.copy()
            if pose.tvec is not None:
                self.pose_tracker.tvec = pose.tvec.copy()
            self.pose_tracker.T_marker_camera = (
                None if pose.T_marker_camera is None else pose.T_marker_camera.copy()
            )
            stage_t0 = time.perf_counter()
            filtered_depth = self._apply_depth_filter_to_pose(pose, track_points)
            mark_pose_timing("pose_depth_filter_prior_ms", stage_t0)
            stage_t0 = time.perf_counter()
            inlier_corners = self._inlier_corners_from_pose(pose, tracker_corners)
            mark_pose_timing("pose_inlier_corners_prior_ms", stage_t0)

        stage_t0 = time.perf_counter()
        visual_corners = self._visual_corners_from_pose(
            inlier_corners,
            pose.rvec,
            pose.tvec,
        )
        mark_pose_timing("pose_visual_corners_ms", stage_t0)
        visual_note = ""
        if len(visual_corners) != len(inlier_corners):
            visual_note = (
                f" Visual corners filtered {len(visual_corners)}/"
                f"{len(inlier_corners)}."
            )
        if update_persistence:
            stage_t0 = time.perf_counter()
            reject_reason = self._decode_update_rejection_reason(visual_corners)
            mark_pose_timing("pose_decode_update_guard_ms", stage_t0)
            if reject_reason:
                self.pose_tracker.rvec = prev_pose_rvec
                self.pose_tracker.tvec = prev_pose_tvec
                self.pose_tracker.T_marker_camera = prev_pose_T
                self.pose_depth_filter.restore(prev_depth_filter_state)
                return TrackerResult(
                    success=False,
                    mode=self.mode,
                    message=reject_reason + visual_note,
                    rvec=pose.rvec,
                    tvec=pose.tvec,
                    T_marker_camera=pose.T_marker_camera,
                    mean_reprojection_error_px=pose.reprojection_mean_px,
                    max_reprojection_error_px=pose.reprojection_max_px,
                    num_points=pose.num_points,
                    num_inliers=pose.num_inliers,
                    pose_source=pose_source,
                    pnp_method=str(getattr(pose, "method", "")),
                    corners=[],
                    correspondence_corners=tracker_corners,
                    timings_ms=dict(pose_timings),
                    **self._depth_filter_kwargs(filtered_depth),
                    **self._plateau_prior_kwargs(
                        plateau_prior,
                        triggered=plateau_triggered,
                        attempted=plateau_triggered,
                        applied=plateau_applied,
                    ),
                )

            stage_t0 = time.perf_counter()
            self._store_persistent_correspondences(visual_corners)
            mark_pose_timing("pose_persistent_store_ms", stage_t0)

        if not update_persistence and len(visual_corners) < self.config.visual_corner_min_count:
            visual_corners = []
            visual_note += " Visual corners suppressed for fallback pose."

        reliable_pose = (
            update_persistence
            or len(visual_corners) >= self.config.visual_corner_min_count
        )

        stage_t0 = time.perf_counter()
        # Max-pts und Reprojektionsfehler nur fuer verlaessliche Posen aktualisieren.
        if reliable_pose:
            if pose.num_inliers > self._max_pts_seen:
                self._max_pts_seen = pose.num_inliers
            if pose.reprojection_mean_px >= 0.0:
                self._last_good_reproj_px = pose.reprojection_mean_px
            if pose.rvec is not None:
                self._last_accepted_rvec = np.asarray(pose.rvec, dtype=np.float64).reshape(3, 1)
            if pose.tvec is not None:
                self._last_accepted_tvec = np.asarray(pose.tvec, dtype=np.float64).reshape(3, 1)
            if pose.T_marker_camera is not None:
                self._last_accepted_T_marker_camera = np.asarray(
                    pose.T_marker_camera,
                    dtype=np.float64,
                ).copy()
            self._last_accepted_pose_frame = self.frame_index
        mark_pose_timing("pose_accept_state_update_ms", stage_t0)

        confidence = self._confidence(
            pose.num_inliers,
            pose.reprojection_mean_px,
        )

        stage_t0 = time.perf_counter()
        result = TrackerResult(
            success=True,
            mode=TrackerMode.TRACKING,
            message=success_message + visual_note,
            corners=visual_corners,
            correspondence_corners=tracker_corners,
            rvec=pose.rvec,
            tvec=pose.tvec,
            T_marker_camera=pose.T_marker_camera,
            mean_reprojection_error_px=pose.reprojection_mean_px,
            max_reprojection_error_px=pose.reprojection_max_px,
            num_points=pose.num_points,
            num_inliers=pose.num_inliers,
            confidence=confidence,
            pose_source=pose_source,
            pnp_method=str(getattr(pose, "method", "")),
            timings_ms=dict(pose_timings),
            **self._depth_filter_kwargs(filtered_depth),
            **self._plateau_prior_kwargs(
                plateau_prior,
                triggered=plateau_triggered,
                attempted=plateau_triggered,
                applied=plateau_applied,
            ),
        )
        mark_pose_timing("pose_result_build_ms", stage_t0)
        result.timings_ms = dict(pose_timings)
        return result

    def _reprojection_errors_for_pose(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return per-point reprojection errors for a candidate pose."""
        try:
            projected, _ = cv2.projectPoints(
                np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                self.K,
                self.dist_coeffs,
            )
        except Exception:
            return None

        projected = projected.reshape(-1, 2)
        return np.linalg.norm(projected - image_points, axis=1)

    def _refresh_persistent_correspondences_from_result(
        self,
        result: TrackerResult,
        max_mean_error_px: float,
    ) -> None:
        """Refresh persistent identities when a fallback or fast-path pose is accurate."""
        if (
            result.mean_reprojection_error_px >= 0.0
            and result.mean_reprojection_error_px <= float(max_mean_error_px)
            and len(result.corners) >= self.config.persistence_min_points
        ):
            self._store_persistent_correspondences(result.corners)

    def _set_fast_path_debug(
        self,
        *,
        attempted: bool,
        success: bool = False,
        reason: str = "",
        matches: int = 0,
    ) -> None:
        """Record fast-path diagnostics from the latest persistence match attempt."""
        stats = self._last_persistent_match_stats
        self._last_fast_path_debug = FastPathDebug(
            attempted=bool(attempted),
            success=bool(success),
            reason=str(reason),
            matches=int(matches),
            identities=int(stats.identities),
            current_corners=int(stats.current_corners),
            used_pose_projection=bool(stats.used_pose_projection),
            rejected_no_projection=int(stats.rejected_no_projection),
            rejected_far=int(stats.rejected_far),
            rejected_ambiguous=int(stats.rejected_ambiguous),
            rejected_claimed=int(stats.rejected_claimed),
        )

    def _attach_fast_path_debug(self, result: TrackerResult) -> None:
        result.fast_path_debug = self._last_fast_path_debug


class FallbackPoseMixin:
    """Recover pose from cached identities, uncoded grids, or held previous poses."""

    def _estimate_pose_from_persistent_correspondences(
        self,
        detection,
        reason: str,
    ) -> Optional[TrackerResult]:
        """Estimate pose from cached global identities after fresh decode failure."""
        if self.config.decode_only_mode:
            return None

        if not self.config.enable_temporal_correspondence_persistence:
            return None

        points, corners = self._persistent_correspondences_for_detection(detection)

        if len(points) < self.config.persistence_min_points:
            return None

        if (
            "No valid decoded patches" in reason
            and len(points) < self.config.persistence_min_points_after_decode_fail
        ):
            return None

        result = self._estimate_and_package_pose(
            points,
            corners,
            success_message=(
                f"Pose estimated from persistent correspondences after: {reason}."
            ),
            update_persistence=False,
            pose_source=PoseSource.PERSISTENT,
            detection=detection,
        )

        if result.success:
            result.confidence *= 0.85

            # If the persistent-fallback pose is good, refresh the persistent
            # state so the tracker doesn't run out of time budget
            # (persistence_max_frames) while the main decode is warming up.
            self._refresh_persistent_correspondences_from_result(
                result,
                max_mean_error_px=self.config.persistence_refresh_mean_error_px,
            )

            return result

        return None

    def _estimate_pose_from_uncoded_grid_bootstrap(
        self,
        detection,
        reason: str,
    ) -> Optional[TrackerResult]:
        """Bootstrap a first pose from checkerboard topology before patches decode."""
        self._last_uncoded_bootstrap_reason = ""
        if self.config.decode_only_mode:
            self._last_uncoded_bootstrap_reason = "decode_only_mode"
            return None

        if not self.config.enable_uncoded_grid_bootstrap:
            self._last_uncoded_bootstrap_reason = "disabled"
            return None

        current = self._detected_corners_from_detection(detection)
        if len(current) < self.config.uncoded_bootstrap_min_corners:
            self._last_uncoded_bootstrap_reason = f"too_few_corners:{len(current)}"
            return None

        if self._last_accepted_rvec is not None and self._last_accepted_tvec is not None:
            self._last_uncoded_bootstrap_reason = "pose_history_exists"
            return None

        local_rows = [int(c.local_row) for c in current]
        local_cols = [int(c.local_col) for c in current]
        rows = self.geometry.corner_rows()
        cols = self.geometry.corner_cols()

        min_row_off = -min(local_rows)
        max_row_off = rows - 1 - max(local_rows)
        min_col_off = -min(local_cols)
        max_col_off = cols - 1 - max(local_cols)

        candidates = []
        for row_off in range(min_row_off, max_row_off + 1):
            for col_off in range(min_col_off, max_col_off + 1):
                points: List[PoseTrackPoint] = []
                corners: List[TrackerCorner] = []

                for corner in current:
                    gr = int(corner.local_row) + row_off
                    gc = int(corner.local_col) + col_off
                    if not self.geometry.has_corner(gr, gc):
                        continue

                    pt = self.geometry.corner_point(gr, gc)
                    xyz = (float(pt.x), float(pt.y), float(pt.z))
                    uv = (float(corner.uv[0]), float(corner.uv[1]))
                    points.append(
                        PoseTrackPoint(
                            global_row=gr,
                            global_col=gc,
                            xyz_mm=xyz,
                            uv=uv,
                            votes=0,
                        )
                    )
                    corners.append(
                        TrackerCorner(
                            local_row=int(corner.local_row),
                            local_col=int(corner.local_col),
                            global_row=gr,
                            global_col=gc,
                            xyz_mm=xyz,
                            uv=uv,
                            votes=0,
                        )
                    )

                if len(points) < self.config.uncoded_bootstrap_min_corners:
                    continue

                candidate = self._solve_uncoded_bootstrap_candidate(points, corners)
                if candidate is not None:
                    candidates.append((candidate, row_off, col_off))

        if not candidates:
            self._last_uncoded_bootstrap_reason = "no_valid_candidates"
            return None

        candidates.sort(key=lambda x: (x[0].mean_reprojection_error_px, x[0].max_reprojection_error_px))
        best, row_off, col_off = candidates[0]
        second_mean = (
            candidates[1][0].mean_reprojection_error_px
            if len(candidates) > 1
            else float("inf")
        )

        if best.mean_reprojection_error_px > self.config.uncoded_bootstrap_max_mean_reprojection_error_px:
            self._last_uncoded_bootstrap_reason = (
                f"mean_error:{best.mean_reprojection_error_px:.3f}"
            )
            return None

        if best.max_reprojection_error_px > self.config.uncoded_bootstrap_max_max_reprojection_error_px:
            self._last_uncoded_bootstrap_reason = (
                f"max_error:{best.max_reprojection_error_px:.3f}"
            )
            return None

        if (
            np.isfinite(second_mean)
            and (second_mean - best.mean_reprojection_error_px)
            < self.config.uncoded_bootstrap_min_second_best_margin_px
        ):
            self._last_uncoded_bootstrap_reason = (
                f"ambiguous:best={best.mean_reprojection_error_px:.3f},"
                f"second={second_mean:.3f}"
            )
            return None

        best.message = (
            "Pose estimated from uncoded grid bootstrap after: "
            f"{reason} (offset={row_off},{col_off}, "
            f"second_mean={second_mean:.3f})."
        )
        best.confidence *= 0.55
        self.pose_tracker.rvec = best.rvec.copy()
        self.pose_tracker.tvec = best.tvec.copy()
        self.pose_tracker.T_marker_camera = (
            None
            if best.T_marker_camera is None
            else best.T_marker_camera.copy()
        )
        self._last_good_reproj_px = best.mean_reprojection_error_px
        self._last_accepted_rvec = best.rvec.copy()
        self._last_accepted_tvec = best.tvec.copy()
        self._last_accepted_T_marker_camera = (
            None
            if best.T_marker_camera is None
            else best.T_marker_camera.copy()
        )
        self._last_accepted_pose_frame = self.frame_index
        self._store_persistent_correspondences(best.corners)
        return best

    def _solve_uncoded_bootstrap_candidate(
        self,
        points: List[PoseTrackPoint],
        corners: List[TrackerCorner],
    ) -> Optional[TrackerResult]:
        """Evaluate one uncoded-grid offset candidate with PnP and visual gates."""
        object_points = np.asarray([p.xyz_mm for p in points], dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray([p.uv for p in points], dtype=np.float64).reshape(-1, 2)

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                self.K,
                self.dist_coeffs,
                iterationsCount=int(self.config.pnp_ransac_iterations),
                reprojectionError=float(self.config.pnp_ransac_reprojection_px),
                confidence=float(self.config.pnp_ransac_confidence),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            return None

        if not success or inliers is None or len(inliers) < self.config.min_inliers:
            return None

        inlier_idx = np.asarray(inliers, dtype=np.int64).reshape(-1)
        object_inliers = object_points[inlier_idx]
        image_inliers = image_points[inlier_idx]

        try:
            projected, _ = cv2.projectPoints(
                object_inliers,
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                self.K,
                self.dist_coeffs,
            )
        except Exception:
            return None

        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - image_inliers, axis=1)
        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))

        inlier_corners = [
            corners[int(i)]
            for i in inlier_idx
            if 0 <= int(i) < len(corners)
        ]
        visual_corners = self._visual_corners_from_pose(inlier_corners, rvec, tvec)
        if len(visual_corners) < self.config.visual_corner_min_count:
            return None

        T = self.pose_tracker.T_marker_camera
        try:
            T = make_transform_from_rvec_tvec(rvec, tvec)
        except Exception:
            T = None

        rvec_arr = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec_arr = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

        T_arr = None if T is None else np.asarray(T, dtype=np.float64).reshape(4, 4)

        confidence = self._confidence(len(visual_corners), mean_err) * 0.5
        return TrackerResult(
            success=True,
            mode=TrackerMode.TRACKING,
            message="Pose estimated from uncoded grid bootstrap.",
            corners=visual_corners,
            correspondence_corners=inlier_corners,
            rvec=rvec_arr,
            tvec=tvec_arr,
            T_marker_camera=T_arr,
            mean_reprojection_error_px=mean_err,
            max_reprojection_error_px=max_err,
            num_points=len(points),
            num_inliers=len(inlier_corners),
            confidence=confidence,
            pose_source=PoseSource.UNCODED_GRID,
        )

    def _persistent_pose_motion_plausible(
        self,
        rvec: Optional[np.ndarray],
        tvec: Optional[np.ndarray],
        prev_rvec: Optional[np.ndarray],
        prev_tvec: Optional[np.ndarray],
    ) -> bool:
        """Check whether a fallback pose is plausible relative to last accepted pose."""
        if rvec is None or tvec is None:
            return False

        if prev_rvec is None or prev_tvec is None:
            return True

        try:
            R_prev, _ = cv2.Rodrigues(
                np.asarray(prev_rvec, dtype=np.float64).reshape(3, 1)
            )
            R_curr, _ = cv2.Rodrigues(
                np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            )
            dR = R_curr @ R_prev.T
            cos_a = float(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
            rot_delta_deg = float(np.degrees(np.arccos(cos_a)))

            t_prev = np.asarray(prev_tvec, dtype=np.float64).reshape(3, 1)
            t_curr = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            trans_delta_mm = float(np.linalg.norm(t_curr - t_prev))
        except Exception:
            return False

        return (
            rot_delta_deg <= self.config.persistence_max_rotation_jump_deg
            and trans_delta_mm <= self.config.persistence_max_translation_jump_mm
        )

    def _detection_has_decodeable_cell_span(self, detection) -> bool:
        """Return whether a detection spans enough cells to support patch decoding."""
        cells = list(getattr(detection, "cells", []))
        if not cells:
            return False

        min_span = max(1, int(self.config.checker_min_tracking_decode_cell_span))
        rows = [int(getattr(c, "j", getattr(c, "row", 0))) for c in cells]
        cols = [int(getattr(c, "i", getattr(c, "col", 0))) for c in cells]
        row_span = max(rows) - min(rows) + 1 if rows else 0
        col_span = max(cols) - min(cols) + 1 if cols else 0
        return row_span >= min_span and col_span >= min_span

    def _force_local_recovery(self) -> None:
        """Clear local tracking state after repeated topology or correspondence failures."""
        self.checkerboard_detector.reset_tracking()
        self.dot_detector = self._create_dot_detector()
        self._clear_persistent_correspondences()
        self._undecodeable_detection_frames = 0
        self._pose_propagation_block_until_frame = max(
            self._pose_propagation_block_until_frame,
            self.frame_index + 5,
        )

    def _note_low_fresh_correspondence_failure(self, fresh_count: int) -> None:
        """Track repeated low-fresh-correspondence frames and trigger recovery."""
        if fresh_count >= self.config.checker_min_fresh_correspondences_for_stable_tracking:
            self._low_fresh_correspondence_frames = 0
            return

        self._low_fresh_correspondence_frames += 1
        if (
            self._low_fresh_correspondence_frames
            > self.config.checker_max_low_fresh_correspondence_frames
        ):
            self._force_local_recovery()

    def _hold_last_pose_result(
        self,
        detection,
        reason: str,
        correspondence_corners: List[TrackerCorner],
    ) -> Optional[TrackerResult]:
        """Publish the current pose prior when it still aligns with detected corners."""
        if self.config.decode_only_mode:
            return None

        if self.pose_tracker.rvec is None or self.pose_tracker.tvec is None:
            return None

        if (
            self._low_fresh_correspondence_frames > self.config.pose_hold_max_frames
            and self.config.pose_hold_max_frames >= 0
        ):
            return None

        if detection is None or not bool(detection.valid()):
            return None

        detected_count = len(self._detected_corners_from_detection(detection))
        if detected_count < self.config.pose_hold_min_detection_corners:
            return None

        rvec = np.asarray(self.pose_tracker.rvec, dtype=np.float64).reshape(3, 1).copy()
        tvec = np.asarray(self.pose_tracker.tvec, dtype=np.float64).reshape(3, 1).copy()
        T = (
            None
            if self.pose_tracker.T_marker_camera is None
            else self.pose_tracker.T_marker_camera.copy()
        )

        held_corners, match_count, median_err, p90_err = (
            self._projected_tracker_corners_for_detection_pose(
                detection,
                rvec,
                tvec,
                max_dist_px=self.config.visual_corner_max_reprojection_error_px,
            )
        )

        if (
            match_count < self.config.visual_corner_min_count
            or median_err > self.config.visual_corner_max_reprojection_error_px
            or p90_err > self.config.visual_corner_max_reprojection_error_px
        ):
            return None

        return TrackerResult(
            success=True,
            mode=TrackerMode.TRACKING,
            message=(
                f"Pose held from last accepted pose after: {reason} "
                f"(blue_align={match_count}, median={median_err:.2f}px, "
                f"p90={p90_err:.2f}px)."
            ),
            corners=held_corners,
            correspondence_corners=correspondence_corners,
            rvec=rvec,
            tvec=tvec,
            T_marker_camera=T,
            mean_reprojection_error_px=self._last_good_reproj_px,
            max_reprojection_error_px=-1.0,
            num_points=max(len(held_corners), 0),
            num_inliers=max(len(held_corners), 0),
            confidence=0.25,
            pose_source=PoseSource.HOLD,
        )

    def _hold_last_pose_without_detection_result(self, detection) -> Optional[TrackerResult]:
        """Publish the current pose prior when checkerboard detection is unavailable."""
        if self.config.decode_only_mode:
            return None

        if self.pose_tracker.rvec is None or self.pose_tracker.tvec is None:
            return None

        if (
            self._last_good_reproj_px < 0.0
            or self._last_good_reproj_px
            > self.config.fallback_pose_max_mean_reprojection_error_px
        ):
            return None

        rvec = np.asarray(self.pose_tracker.rvec, dtype=np.float64).reshape(3, 1).copy()
        tvec = np.asarray(self.pose_tracker.tvec, dtype=np.float64).reshape(3, 1).copy()
        T = (
            None
            if self.pose_tracker.T_marker_camera is None
            else self.pose_tracker.T_marker_camera.copy()
        )

        return TrackerResult(
            success=True,
            mode=self.mode,
            message=(
                "Pose held from last accepted pose without checkerboard detection."
            ),
            detection_valid=False,
            detection_tracking=False if detection is None else bool(detection.tracking),
            detection_stable=False if detection is None else bool(detection.stable),
            detection_corners=self._detected_corners_from_detection(detection),
            corners=[],
            correspondence_corners=[],
            rvec=rvec,
            tvec=tvec,
            T_marker_camera=T,
            mean_reprojection_error_px=self._last_good_reproj_px,
            max_reprojection_error_px=-1.0,
            num_points=0,
            num_inliers=0,
            confidence=0.10,
            pose_source=PoseSource.HOLD,
        )

    def _emergency_last_pose_result(
        self,
        detection,
        reason: str,
    ) -> Optional[TrackerResult]:
        """Publish the last accepted pose as a final low-confidence fallback."""
        if self.config.decode_only_mode:
            return None

        if not self.config.emergency_pose_hold_enabled:
            return None

        if self._last_accepted_rvec is None or self._last_accepted_tvec is None:
            return None

        age = self.frame_index - self._last_accepted_pose_frame
        if age < 0:
            return None

        max_age = int(self.config.emergency_pose_hold_max_frames)
        if max_age >= 0 and age > max_age:
            return None

        rvec = np.asarray(self._last_accepted_rvec, dtype=np.float64).reshape(3, 1).copy()
        tvec = np.asarray(self._last_accepted_tvec, dtype=np.float64).reshape(3, 1).copy()
        T = (
            None
            if self._last_accepted_T_marker_camera is None
            else self._last_accepted_T_marker_camera.copy()
        )

        self.pose_tracker.rvec = rvec.copy()
        self.pose_tracker.tvec = tvec.copy()
        self.pose_tracker.T_marker_camera = None if T is None else T.copy()

        held_corners: List[TrackerCorner] = []
        align_msg = "no_blue_alignment"
        if detection is not None and bool(detection.valid()):
            corners, match_count, median_err, p90_err = (
                self._projected_tracker_corners_for_detection_pose(
                    detection,
                    rvec,
                    tvec,
                    max_dist_px=self.config.visual_corner_max_reprojection_error_px,
                )
            )
            if (
                match_count >= self.config.visual_corner_min_count
                and median_err <= self.config.visual_corner_max_reprojection_error_px
                and p90_err <= self.config.visual_corner_max_reprojection_error_px
            ):
                held_corners = corners
                align_msg = (
                    f"blue_align={match_count}, median={median_err:.2f}px, "
                    f"p90={p90_err:.2f}px"
                )

        confidence = max(0.03, 0.20 * (0.96 ** max(age, 0)))

        return TrackerResult(
            success=True,
            mode=self.mode,
            message=(
                f"Emergency pose held from last accepted pose after: {reason} "
                f"(age={age}, {align_msg})."
            ),
            detection_valid=False if detection is None else bool(detection.valid()),
            detection_tracking=False if detection is None else bool(detection.tracking),
            detection_stable=False if detection is None else bool(detection.stable),
            detection_corners=self._detected_corners_from_detection(detection),
            corners=held_corners,
            correspondence_corners=[],
            rvec=rvec,
            tvec=tvec,
            T_marker_camera=T,
            mean_reprojection_error_px=self._last_good_reproj_px,
            max_reprojection_error_px=-1.0,
            num_points=len(held_corners),
            num_inliers=len(held_corners),
            confidence=confidence,
            pose_source=PoseSource.HOLD,
        )
