from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

from tracking.hydramarker.map_pose_tracker import PoseTrackPoint
from tracking.hydramarker.tracker_types import (
    FastPathDebug,
    PoseSource,
    TrackerCorner,
    TrackerMode,
    TrackerResult,
)


class PoseEstimationMixin:
    def _estimate_and_package_pose(
        self,
        track_points: List[PoseTrackPoint],
        tracker_corners: List[TrackerCorner],
        success_message: str,
        update_persistence: bool,
        pose_source: PoseSource,
        detection=None,
    ) -> TrackerResult:
        prev_pose_rvec = None if self.pose_tracker.rvec is None else self.pose_tracker.rvec.copy()
        prev_pose_tvec = None if self.pose_tracker.tvec is None else self.pose_tracker.tvec.copy()
        prev_pose_T = (
            None
            if self.pose_tracker.T_marker_camera is None
            else self.pose_tracker.T_marker_camera.copy()
        )
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

        pnp_t0 = time.perf_counter()
        pose = self.pose_tracker.estimate_pose(
            track_points,
            lost_frames=self.lost_frames,
        )
        pnp_ms = (time.perf_counter() - pnp_t0) * 1000.0

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
                timings_ms={"pnp_ms": pnp_ms},
            )

        inlier_corners = self._inlier_corners_from_pose(pose, tracker_corners)

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
                timings_ms={"pnp_ms": pnp_ms},
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
                    timings_ms={"pnp_ms": pnp_ms},
                )

        if update_persistence:
            self._store_persistent_correspondences(inlier_corners)

        visual_corners = self._visual_corners_from_pose(
            inlier_corners,
            pose.rvec,
            pose.tvec,
        )
        visual_note = ""
        if len(visual_corners) != len(inlier_corners):
            visual_note = (
                f" Visual corners filtered {len(visual_corners)}/"
                f"{len(inlier_corners)}."
            )
        if not update_persistence and len(visual_corners) < self.config.visual_corner_min_count:
            visual_corners = []
            visual_note += " Visual corners suppressed for fallback pose."

        reliable_pose = (
            update_persistence
            or len(visual_corners) >= self.config.visual_corner_min_count
        )

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

        confidence = self._confidence(
            pose.num_inliers,
            pose.reprojection_mean_px,
        )

        return TrackerResult(
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
            timings_ms={"pnp_ms": pnp_ms},
        )

    def _reprojection_errors_for_pose(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> Optional[np.ndarray]:
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
