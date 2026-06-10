from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tracking.hydramarker.identity_store import GlobalCornerIdentity
from tracking.hydramarker.map_pose_tracker import PoseTrackPoint
from tracking.hydramarker.tracker_types import (
    DenseProjectionMatchStats,
    DetectedCorner,
    FastPathDebug,
    PersistentMatchStats,
    PoseSource,
    TrackerCorner,
    TrackerMode,
    TrackerResult,
)
from tracking.pose_solvers import make_transform_from_rvec_tvec


class FallbackPoseMixin:
    def _estimate_pose_from_persistent_correspondences(
        self,
        detection,
        reason: str,
    ) -> Optional[TrackerResult]:
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
            from tracking.hydramarker.map_pose_tracker import make_transform_from_rvec_tvec
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
        self.checkerboard_detector.reset_tracking()
        self.dot_detector = self._create_dot_detector()
        self._clear_persistent_correspondences()
        self._undecodeable_detection_frames = 0
        self._pose_propagation_block_until_frame = max(
            self._pose_propagation_block_until_frame,
            self.frame_index + 5,
        )

    def _note_low_fresh_correspondence_failure(self, fresh_count: int) -> None:
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

