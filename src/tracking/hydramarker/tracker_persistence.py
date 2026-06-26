"""Persistent identity cache, fast-path tracking, consistency checks, and logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tracking.hydramarker.backend import cpp_impl as hm
from tracking.hydramarker.tracker_pose import MapPoseResult, PoseTrackPoint
from tracking.hydramarker.tracker_types import (
    GlobalCornerIdentity,
    GridKey,
    PersistentMatchStats,
    PoseSource,
    TrackerCorner,
    TrackerResult,
)


class PersistenceMixin:
    """Maintain and reuse global corner identities across short decode outages."""

    def _cpp_persistent_matcher_enabled(self) -> bool:
        return bool(
            getattr(
                self.config,
                "cpp_persistent_matcher_enabled",
                True,
            )
        )

    def _cpp_fast_persistent_seed_enabled(self) -> bool:
        return bool(
            getattr(
                self.config,
                "cpp_fast_persistent_seed_enabled",
                True,
            )
        )

    def _cpp_fast_pose_transaction_enabled(self) -> bool:
        return bool(
            getattr(
                self.config,
                "cpp_fast_pose_transaction_enabled",
                True,
            )
        )

    def _cpp_depth_filter_for_fast_pose(self):
        if not bool(getattr(self.config, "pose_depth_filter_enabled", False)):
            return None

        depth_filter = getattr(self, "pose_depth_filter", None)
        if type(depth_filter).__module__ != "hydramarker_cpp":
            return None
        return depth_filter

    def _cpp_persistent_matcher_config_signature(self) -> Tuple[object, ...]:
        names = (
            "min_points",
            "min_inliers",
            "persistence_min_points",
            "persistence_max_frames",
            "fast_persistent_refresh_mean_error_px",
            "persistence_use_pose_projection",
            "persistence_projection_max_reproj_px",
            "persistence_projection_adaptive_match_enabled",
            "persistence_projection_adaptive_motion_start_px",
            "persistence_projection_adaptive_motion_scale",
            "persistence_projection_adaptive_max_reproj_px",
            "persistence_projection_max_pose_error_px",
            "persistence_match_min_second_best_margin_px",
            "persistence_uv_match_dist_px",
            "fast_persistent_min_points",
            "pnp_ransac_reprojection_px",
            "pnp_ransac_confidence",
            "pnp_ransac_iterations",
            "max_mean_reprojection_error_px",
            "max_max_reprojection_error_px",
            "max_translation_jump_mm",
            "max_rotation_jump_deg",
            "rotation_gate_scale_per_lost_frame",
            "rotation_gate_max_deg",
            "use_pose_prior",
            "pnp_direct_prior_enabled",
            "pnp_direct_refine_method",
            "pnp_direct_max_mean_reprojection_error_px",
            "pnp_direct_max_max_reprojection_error_px",
            "persistence_max_translation_jump_mm",
            "persistence_max_rotation_jump_deg",
            "fallback_pose_min_detection_matches",
            "fallback_pose_max_median_corner_error_px",
            "fallback_pose_max_p90_corner_error_px",
            "fallback_pose_max_mean_reprojection_error_px",
            "fallback_pose_max_max_reprojection_error_px",
            "fast_persistent_dense_refine_enabled",
            "fast_persistent_dense_min_points",
            "fast_persistent_dense_match_max_px",
            "fast_persistent_dense_min_second_best_margin_px",
            "fast_persistent_dense_max_median_px",
            "fast_persistent_dense_max_p90_px",
            "fast_persistent_dense_rescue_enabled",
            "fast_persistent_dense_rescue_min_green_ratio",
            "fast_persistent_dense_rescue_min_seed_median_px",
            "fast_persistent_dense_min_image_coverage",
            "fast_persistent_dense_min_object_span_mm",
            "fast_persistent_dense_min_distinct_rows",
            "fast_persistent_dense_min_distinct_cols",
            "fast_persistent_dense_pose_solver",
            "fast_persistent_dense_robust_refine_method",
            "fast_persistent_dense_robust_trim_enabled",
            "fast_persistent_dense_robust_trim_quantile",
            "fast_persistent_dense_robust_min_keep_ratio",
            "fast_persistent_dense_robust_max_mean_px",
            "fast_persistent_dense_robust_max_max_px",
            "fast_persistent_dense_adaptive_refine_enabled",
            "fast_persistent_dense_adaptive_min_match_ratio",
            "fast_persistent_dense_adaptive_motion_px",
            "fast_persistent_dense_adaptive_max_seed_mean_px",
            "fast_persistent_dense_adaptive_max_seed_max_px",
            "visual_corner_min_count",
            "visual_corner_max_reprojection_error_px",
            "decode_only_mode",
            "enable_temporal_correspondence_persistence",
        )
        return tuple(getattr(self.config, name, None) for name in names)

    def _ensure_cpp_persistent_matcher(self):
        """Create the C++ persistent matcher on demand."""
        if not self._cpp_persistent_matcher_enabled():
            return None

        if bool(getattr(self, "_cpp_persistent_matcher_unavailable", False)):
            return None

        signature = self._cpp_persistent_matcher_config_signature()
        matcher = getattr(self, "_cpp_persistent_matcher", None)
        if (
            matcher is not None
            and getattr(
                self,
                "_cpp_persistent_matcher_config_state",
                None,
            ) == signature
        ):
            return matcher

        try:
            matcher = hm.create_persistent_matcher(self.config)
        except Exception:
            self._cpp_persistent_matcher_unavailable = True
            self._cpp_persistent_matcher = None
            return None

        self._cpp_persistent_matcher = matcher
        self._cpp_persistent_matcher_config_state = signature
        self._sync_cpp_persistent_matcher()
        return matcher

    def _sync_cpp_persistent_matcher(self) -> None:
        """Mirror Python persistent identities into the C++ matcher."""
        if not self._cpp_persistent_matcher_enabled():
            return

        matcher = getattr(self, "_cpp_persistent_matcher", None)
        if matcher is None:
            return

        identities = self._identity_store.all()
        if self._persistent_frame_index < 0 or not identities:
            try:
                matcher.clear_identities()
            except Exception:
                self._cpp_persistent_matcher_unavailable = True
            return

        try:
            matcher.replace_identities(
                hm.global_corner_identities_from_python(identities),
                int(self._persistent_frame_index),
            )
        except Exception:
            self._cpp_persistent_matcher_unavailable = True

    @staticmethod
    def _persistent_match_stats_from_cpp(stats) -> PersistentMatchStats:
        return PersistentMatchStats(
            age=int(getattr(stats, "age", 0)),
            identities=int(getattr(stats, "identities", 0)),
            current_corners=int(getattr(stats, "current_corners", 0)),
            accepted=int(getattr(stats, "accepted", 0)),
            used_pose_projection=bool(
                getattr(stats, "used_pose_projection", False)
            ),
            adaptive_motion_px=float(
                getattr(stats, "adaptive_motion_px", 0.0)
            ),
            adaptive_max_dist_px=float(
                getattr(stats, "adaptive_max_dist_px", 0.0)
            ),
            rejected_no_projection=int(
                getattr(stats, "rejected_no_projection", 0)
            ),
            rejected_far=int(getattr(stats, "rejected_far", 0)),
            rejected_ambiguous=int(getattr(stats, "rejected_ambiguous", 0)),
            rejected_claimed=int(getattr(stats, "rejected_claimed", 0)),
        )

    @staticmethod
    def _pose_track_points_from_cpp(points) -> List[PoseTrackPoint]:
        return [
            PoseTrackPoint(
                global_row=int(point.global_row),
                global_col=int(point.global_col),
                xyz_mm=tuple(float(v) for v in point.xyz_mm),
                uv=tuple(float(v) for v in point.uv),
                votes=int(getattr(point, "votes", 0)),
            )
            for point in points
        ]

    @staticmethod
    def _tracker_corners_from_cpp(corners) -> List[TrackerCorner]:
        return [
            TrackerCorner(
                local_row=int(corner.local_row),
                local_col=int(corner.local_col),
                global_row=int(corner.global_row),
                global_col=int(corner.global_col),
                xyz_mm=tuple(float(v) for v in corner.xyz_mm),
                uv=tuple(float(v) for v in corner.uv),
                votes=int(getattr(corner, "votes", 0)),
            )
            for corner in corners
        ]

    @staticmethod
    def _global_corner_identities_from_cpp(
        identities,
    ) -> List[GlobalCornerIdentity]:
        return [
            GlobalCornerIdentity(
                global_row=int(identity.global_row),
                global_col=int(identity.global_col),
                xyz_mm=tuple(float(v) for v in identity.xyz_mm),
                uv=tuple(float(v) for v in identity.uv),
                votes=int(getattr(identity, "votes", 0)),
            )
            for identity in identities
        ]

    def _commit_cpp_persistent_refresh(self, fast_result) -> bool:
        """Commit C++-computed fast-path persistence identities."""
        if not bool(
            getattr(fast_result, "persistence_refresh_available", False)
        ):
            return False

        cpp_identities = list(
            getattr(fast_result, "persistence_refresh_identities", [])
        )
        identities = self._global_corner_identities_from_cpp(cpp_identities)
        if len(identities) < int(self.config.persistence_min_points):
            return False

        default_frame_index = int(getattr(self, "frame_index", -1))
        frame_index = int(
            getattr(
                fast_result,
                "persistence_refresh_frame",
                default_frame_index,
            )
        )
        if frame_index < 0:
            return False

        matcher = getattr(self, "_cpp_persistent_matcher", None)
        if matcher is None:
            return False

        try:
            matcher.replace_identities(cpp_identities, frame_index)
        except Exception:
            self._cpp_persistent_matcher_unavailable = True
            return False

        self._identity_store.replace(identities)
        self._persistent_frame_index = frame_index
        return True

    @staticmethod
    def _cpp_pose_vector_to_array(
        values,
        shape: Tuple[int, ...],
    ) -> Optional[np.ndarray]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return None
        try:
            return arr.reshape(shape).copy()
        except ValueError:
            return None

    @staticmethod
    def _map_pose_result_from_cpp(pose) -> MapPoseResult:
        inliers = np.asarray(
            getattr(pose, "inlier_indices", []),
            dtype=np.int64,
        ).reshape(-1)
        return MapPoseResult(
            success=bool(getattr(pose, "success", False)),
            message=str(getattr(pose, "message", "")),
            rvec=PersistenceMixin._cpp_pose_vector_to_array(
                getattr(pose, "rvec", []),
                (3, 1),
            ),
            tvec=PersistenceMixin._cpp_pose_vector_to_array(
                getattr(pose, "tvec", []),
                (3, 1),
            ),
            T_marker_camera=PersistenceMixin._cpp_pose_vector_to_array(
                getattr(pose, "T_marker_camera", []),
                (4, 4),
            ),
            inlier_indices=inliers.copy() if inliers.size > 0 else None,
            reprojection_mean_px=float(
                getattr(pose, "reprojection_mean_px", -1.0)
            ),
            reprojection_max_px=float(
                getattr(pose, "reprojection_max_px", -1.0)
            ),
            num_points=int(getattr(pose, "num_points", 0)),
            num_inliers=int(getattr(pose, "num_inliers", 0)),
            points=PersistenceMixin._pose_track_points_from_cpp(
                getattr(pose, "points", [])
            ),
            method=str(getattr(pose, "method", "")),
        )

    def _persistent_correspondences_for_detection_cpp(
        self,
        detection,
    ) -> Optional[Tuple[List[PoseTrackPoint], List[TrackerCorner]]]:
        """Match cached identities through the C++ implementation when enabled."""
        matcher = self._ensure_cpp_persistent_matcher()
        if matcher is None:
            return None

        try:
            result = matcher.match(
                detection,
                int(self.frame_index),
                self.K,
                self.dist_coeffs,
                None if self.pose_tracker.rvec is None else self.pose_tracker.rvec,
                None if self.pose_tracker.tvec is None else self.pose_tracker.tvec,
                float(self._last_good_reproj_px),
            )
        except Exception:
            return None

        self._last_persistent_match_stats = self._persistent_match_stats_from_cpp(
            result.stats
        )
        self._last_persistent_match_backend = "cpp"
        return (
            self._pose_track_points_from_cpp(result.points),
            self._tracker_corners_from_cpp(result.corners),
        )

    def _fast_persistent_seed_pose_cpp(self, detection):
        """Run persistent matching and seed pose solving in one C++ call."""
        if not self._cpp_fast_persistent_seed_enabled():
            return None

        matcher = self._ensure_cpp_persistent_matcher()
        if matcher is None:
            return None

        try:
            result = matcher.estimate_pose(
                detection,
                int(self.frame_index),
                self.K,
                self.dist_coeffs,
                None if self.pose_tracker.rvec is None else self.pose_tracker.rvec,
                None if self.pose_tracker.tvec is None else self.pose_tracker.tvec,
                float(self._last_good_reproj_px),
                int(self.lost_frames),
            )
        except Exception:
            return None

        self._last_persistent_match_stats = self._persistent_match_stats_from_cpp(
            result.stats
        )
        self._last_persistent_match_backend = "cpp_seed"
        return (
            self._pose_track_points_from_cpp(result.points),
            self._tracker_corners_from_cpp(result.corners),
            self._map_pose_result_from_cpp(result.pose),
            float(getattr(result, "match_ms", 0.0)),
            float(getattr(result, "pose_ms", 0.0)),
            float(getattr(result, "total_ms", 0.0)),
        )

    def _fast_pose_transaction_cpp(self, detection):
        """Run the persistent fast-pose transaction in C++ when enabled."""
        if not self._cpp_fast_pose_transaction_enabled():
            return None

        matcher = self._ensure_cpp_persistent_matcher()
        if matcher is None:
            return None

        depth_filter = self._cpp_depth_filter_for_fast_pose()
        prev_depth_filter_state = (
            None if depth_filter is None else depth_filter.snapshot()
        )
        try:
            result = matcher.estimate_fast_pose(
                detection,
                self.geometry,
                int(self.frame_index),
                self.K,
                self.dist_coeffs,
                None if self.pose_tracker.rvec is None else self.pose_tracker.rvec,
                None if self.pose_tracker.tvec is None else self.pose_tracker.tvec,
                float(self._last_good_reproj_px),
                None if self._last_accepted_rvec is None else self._last_accepted_rvec,
                None if self._last_accepted_tvec is None else self._last_accepted_tvec,
                int(self.lost_frames),
                depth_filter,
                int(getattr(self, "_max_pts_seen", 0)),
            )
        except Exception:
            if prev_depth_filter_state is not None:
                depth_filter.restore(prev_depth_filter_state)
            return None

        self._last_persistent_match_stats = self._persistent_match_stats_from_cpp(
            result.stats
        )
        self._last_persistent_match_backend = "cpp_fast_pose"
        return result, prev_depth_filter_state

    def _persistent_detection_motion_px(self, current_uvs: np.ndarray) -> float:
        """Estimate robust image motion between consecutive checker detections."""
        current_uvs = np.asarray(current_uvs, dtype=np.float64).reshape(-1, 2)
        prev_uvs = getattr(self, "_persistent_match_prev_detection_uv", None)
        prev_frame = int(getattr(self, "_persistent_match_prev_detection_frame", -1))

        if prev_frame == int(self.frame_index):
            return float(getattr(self, "_persistent_match_last_motion_px", 0.0))

        motion_px = 0.0
        if (
            prev_uvs is not None
            and len(prev_uvs) > 0
            and len(current_uvs) > 0
        ):
            previous = np.asarray(prev_uvs, dtype=np.float64).reshape(-1, 2)
            diff = current_uvs[:, None, :] - previous[None, :, :]
            distances = np.linalg.norm(diff, axis=2)
            nearest = np.min(distances, axis=1)
            nearest = nearest[np.isfinite(nearest)]
            if len(nearest) > 0:
                motion_px = float(np.median(nearest))

        self._persistent_match_prev_detection_uv = current_uvs.copy()
        self._persistent_match_prev_detection_frame = int(self.frame_index)
        self._persistent_match_last_motion_px = float(motion_px)
        return float(motion_px)

    def _adaptive_persistence_projection_match_radius_px(
        self,
        base_radius_px: float,
        motion_px: float,
    ) -> float:
        """Return a conservative motion-adaptive radius for projected IDs."""
        if not bool(
            getattr(
                self.config,
                "persistence_projection_adaptive_match_enabled",
                True,
            )
        ):
            return float(base_radius_px)

        start_px = float(
            getattr(
                self.config,
                "persistence_projection_adaptive_motion_start_px",
                6.0,
            )
        )
        scale = float(
            getattr(
                self.config,
                "persistence_projection_adaptive_motion_scale",
                1.0,
            )
        )
        max_radius_px = float(
            getattr(
                self.config,
                "persistence_projection_adaptive_max_reproj_px",
                base_radius_px,
            )
        )
        max_radius_px = max(float(base_radius_px), max_radius_px)
        extra_px = max(0.0, float(motion_px) - max(0.0, start_px)) * max(0.0, scale)
        return min(max_radius_px, max(float(base_radius_px), float(base_radius_px) + extra_px))

    def _merge_with_persistent_correspondences(
        self,
        detection,
        fresh_points: List[PoseTrackPoint],
        fresh_corners: List[TrackerCorner],
    ) -> Tuple[List[PoseTrackPoint], List[TrackerCorner]]:
        """Merge fresh correspondences with non-conflicting persistent correspondences."""
        if self.config.decode_only_mode:
            return fresh_points, fresh_corners

        if not self.config.enable_temporal_correspondence_persistence:
            return fresh_points, fresh_corners

        if len(fresh_points) < self.config.persistence_min_fresh_points_for_merge:
            return fresh_points, fresh_corners

        persistent_points, persistent_corners = self._persistent_correspondences_for_detection(detection)

        if not persistent_points:
            return fresh_points, fresh_corners

        merged_points: List[PoseTrackPoint] = []
        merged_corners: List[TrackerCorner] = []
        used_globals: set[Tuple[int, int]] = set()

        for point, corner in zip(fresh_points, fresh_corners):
            key = (int(point.global_row), int(point.global_col))
            if key in used_globals:
                continue
            merged_points.append(point)
            merged_corners.append(corner)
            used_globals.add(key)

        for point, corner in zip(persistent_points, persistent_corners):
            key = (int(point.global_row), int(point.global_col))
            if key in used_globals:
                continue
            merged_points.append(point)
            merged_corners.append(corner)
            used_globals.add(key)

        return merged_points, merged_corners

    def _match_predicted_uv_to_detection_corner(
        self,
        predicted_uv: Tuple[float, float],
        current_uvs: np.ndarray,
        used_current_indices: set[int],
        max_dist_px: float,
    ) -> Tuple[Optional[int], float, float, str]:
        """Find the best available detection corner for a predicted persistent UV."""
        diff = current_uvs - np.asarray(predicted_uv, dtype=np.float64)
        dist_sq = (diff * diff).sum(axis=1)
        order = np.argsort(dist_sq)

        best_idx = int(order[0])
        best_dist = float(np.sqrt(dist_sq[best_idx]))
        second_dist = (
            float(np.sqrt(dist_sq[int(order[1])]))
            if len(order) > 1
            else float("inf")
        )

        if best_dist > float(max_dist_px):
            return None, best_dist, second_dist, "far"

        min_margin = float(
            self.config.persistence_match_min_second_best_margin_px
        )
        if (
            min_margin > 0.0
            and np.isfinite(second_dist)
            and (second_dist - best_dist) < min_margin
        ):
            return None, best_dist, second_dist, "ambiguous"

        if best_idx in used_current_indices:
            return None, best_dist, second_dist, "claimed"

        return best_idx, best_dist, second_dist, ""

    def _persistent_correspondences_for_detection(
        self,
        detection,
    ) -> Tuple[List[PoseTrackPoint], List[TrackerCorner]]:
        """Match cached global identities to the current checkerboard detection."""
        cpp_result = self._persistent_correspondences_for_detection_cpp(detection)
        if cpp_result is not None:
            return cpp_result

        self._last_persistent_match_backend = "python"
        identities = self._identity_store.all()
        stats = PersistentMatchStats(identities=len(identities))
        self._last_persistent_match_stats = stats

        if not identities:
            return [], []

        if self._persistent_frame_index < 0:
            return [], []

        age = self.frame_index - self._persistent_frame_index
        stats.age = int(age)
        if age < 0 or age > self.config.persistence_max_frames:
            return [], []

        # Build a list of all current detection corner UVs for proximity search.
        # We use UV-proximity matching instead of exact local (i,j) key matching
        # because the CheckerboardDetector can re-index its corners after a
        # tracking reset or lattice drift event, silently changing the local
        # coordinate system while the physical UV positions remain correct.
        # Local-key lookup would then find 0 matches even though 50+ corners
        # are visible -- this was the root cause of the 'frozen' failure mode.
        current_corners = self._detected_corners_from_detection(detection)
        stats.current_corners = len(current_corners)
        if not current_corners:
            return [], []

        current_uvs = np.array(
            [(float(c.uv[0]), float(c.uv[1])) for c in current_corners],
            dtype=np.float64,
        )  # shape (N, 2)
        motion_px = self._persistent_detection_motion_px(current_uvs)

        use_pose_projection = (
            self.config.persistence_use_pose_projection
            and self.pose_tracker.rvec is not None
            and self.pose_tracker.tvec is not None
            and self._last_good_reproj_px >= 0.0
            and self._last_good_reproj_px
            <= self.config.persistence_projection_max_pose_error_px
        )
        stats.used_pose_projection = bool(use_pose_projection)
        stats.adaptive_motion_px = float(motion_px)
        stats.adaptive_max_dist_px = float(self.config.persistence_uv_match_dist_px)
        projection_max_dist = float(self.config.persistence_projection_max_reproj_px)
        if use_pose_projection:
            projection_max_dist = self._adaptive_persistence_projection_match_radius_px(
                projection_max_dist,
                motion_px,
            )
            stats.adaptive_max_dist_px = float(projection_max_dist)

        points: List[PoseTrackPoint] = []
        corners: List[TrackerCorner] = []
        used_globals: set[Tuple[int, int]] = set()
        used_current_indices: set[int] = set()

        for cached in identities:
            global_key = (int(cached.global_row), int(cached.global_col))
            if global_key in used_globals:
                continue

            if use_pose_projection:
                projected_uv = self._project_point_uv(cached.xyz_mm)
                if projected_uv is None:
                    stats.rejected_no_projection += 1
                    continue
                max_dist = float(projection_max_dist)
                predicted_uv = projected_uv
            else:
                max_dist = float(self.config.persistence_uv_match_dist_px)
                predicted_uv = (
                    float(cached.uv[0]),
                    float(cached.uv[1]),
                )

            best_idx, _, _, reject_reason = (
                self._match_predicted_uv_to_detection_corner(
                    predicted_uv=predicted_uv,
                    current_uvs=current_uvs,
                    used_current_indices=used_current_indices,
                    max_dist_px=max_dist,
                )
            )

            if reject_reason == "far":
                stats.rejected_far += 1
                continue
            if reject_reason == "ambiguous":
                stats.rejected_ambiguous += 1
                continue
            if reject_reason == "claimed":
                stats.rejected_claimed += 1
                continue
            if best_idx is None:
                continue

            matched = current_corners[best_idx]
            uv = (float(matched.uv[0]), float(matched.uv[1]))
            xyz = self._point3(cached.xyz_mm)
            votes = max(0, int(cached.votes) - age)

            points.append(
                PoseTrackPoint(
                    global_row=global_key[0],
                    global_col=global_key[1],
                    xyz_mm=xyz,
                    uv=uv,
                    votes=votes,
                )
            )

            corners.append(
                TrackerCorner(
                    local_row=int(matched.local_row),
                    local_col=int(matched.local_col),
                    global_row=global_key[0],
                    global_col=global_key[1],
                    xyz_mm=xyz,
                    uv=uv,
                    votes=votes,
                )
            )

            used_globals.add(global_key)
            used_current_indices.add(best_idx)
            stats.accepted += 1

        return points, corners

    def _project_point_uv(
        self,
        xyz_mm,
    ) -> Optional[Tuple[float, float]]:
        """Project a marker-space point using the current pose prior."""
        if self.pose_tracker.rvec is None or self.pose_tracker.tvec is None:
            return None

        return self._project_point_uv_with_pose(
            xyz_mm,
            self.pose_tracker.rvec,
            self.pose_tracker.tvec,
        )

    def _project_point_uv_with_pose(
        self,
        xyz_mm,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        """Project a marker-space point with an explicit pose."""
        obj = np.asarray(
            [self._point3(xyz_mm)],
            dtype=np.float64,
        ).reshape(1, 3)

        try:
            projected, _ = cv2.projectPoints(
                obj,
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                self.K,
                self.dist_coeffs,
            )
        except Exception:
            return None

        uv = projected.reshape(-1, 2)[0]
        return float(uv[0]), float(uv[1])

    def _current_uv_by_local_corner(self, detection) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """Build a frame-local corner-to-UV lookup for the current detection."""
        uv_by_local: Dict[Tuple[int, int], Tuple[float, float]] = {}

        for corner in self._detected_corners_from_detection(detection):
            uv_by_local[(int(corner.local_row), int(corner.local_col))] = (
                float(corner.uv[0]),
                float(corner.uv[1]),
            )

        return uv_by_local

    def _store_persistent_correspondences(self, corners: List[TrackerCorner]) -> None:
        """Replace the persistent identity cache from accepted visual corners."""
        if self.config.decode_only_mode:
            return

        if not self.config.enable_temporal_correspondence_persistence:
            return

        identities: List[GlobalCornerIdentity] = []
        used_global: set[Tuple[int, int]] = set()

        for corner in corners:
            global_key = (int(corner.global_row), int(corner.global_col))

            if global_key in used_global:
                continue

            identities.append(
                GlobalCornerIdentity(
                    global_row=global_key[0],
                    global_col=global_key[1],
                    xyz_mm=self._point3(corner.xyz_mm),
                    uv=self._point2(corner.uv),
                    votes=int(corner.votes),
                )
            )

            used_global.add(global_key)

        if len(identities) >= self.config.persistence_min_points:
            self._identity_store.replace(identities)
            self._persistent_frame_index = self.frame_index
            self._sync_cpp_persistent_matcher()

    def _clear_persistent_correspondences(self) -> None:
        """Clear all cached persistent identities."""
        self._identity_store.clear()
        self._persistent_frame_index = -1
        matcher = getattr(self, "_cpp_persistent_matcher", None)
        if matcher is not None:
            try:
                matcher.clear_identities()
            except Exception:
                self._cpp_persistent_matcher_unavailable = True
    @property
    def _persistent_corners(self) -> List[TrackerCorner]:
        """
        Compatibility view for existing logs/debug scripts.

        The semantic persistence store is IdentityStore. Local indices here are
        intentionally unset because they are frame-local, not persistent IDs.
        """
        corners: List[TrackerCorner] = []
        for identity in self._identity_store.all():
            corners.append(
                TrackerCorner(
                    local_row=-1,
                    local_col=-1,
                    global_row=int(identity.global_row),
                    global_col=int(identity.global_col),
                    xyz_mm=self._point3(identity.xyz_mm),
                    uv=self._point2(identity.uv),
                    votes=int(identity.votes),
                )
            )
        return corners


class FastPathMixin:
    """Estimate pose directly from persistent identities before full decode."""

    def _fast_dense_refine_required_for_seed(
        self,
        *,
        seed_pose,
        match_count: int,
        stats: PersistentMatchStats,
    ) -> Tuple[bool, str, Dict[str, float]]:
        """Return whether a fast-path seed needs dense projection validation."""
        current_corners = max(0, int(getattr(stats, "current_corners", 0)))
        sparse_ratio = float(match_count) / float(max(current_corners, 1))
        motion_px = float(getattr(stats, "adaptive_motion_px", 0.0))
        ambiguous = max(0, int(getattr(stats, "rejected_ambiguous", 0)))
        seed_mean_px = float(getattr(seed_pose, "reprojection_mean_px", -1.0))
        seed_max_px = float(getattr(seed_pose, "reprojection_max_px", -1.0))

        metrics = {
            "match_ratio": sparse_ratio,
            "motion_px": motion_px,
            "ambiguous_count": float(ambiguous),
            "seed_mean_px": seed_mean_px,
            "seed_max_px": seed_max_px,
        }

        if not bool(
            getattr(
                self.config,
                "fast_persistent_dense_adaptive_refine_enabled",
                True,
            )
        ):
            return True, "adaptive_disabled", metrics

        reasons: List[str] = []
        min_ratio = float(
            getattr(
                self.config,
                "fast_persistent_dense_adaptive_min_match_ratio",
                0.85,
            )
        )
        if min_ratio > 0.0 and sparse_ratio < min_ratio:
            reasons.append("low_match_ratio")

        motion_threshold = float(
            getattr(
                self.config,
                "fast_persistent_dense_adaptive_motion_px",
                8.0,
            )
        )
        if motion_threshold > 0.0 and motion_px >= motion_threshold:
            reasons.append("motion")

        if ambiguous > 0:
            reasons.append("ambiguous")

        max_seed_mean = float(
            getattr(
                self.config,
                "fast_persistent_dense_adaptive_max_seed_mean_px",
                1.2,
            )
        )
        if (
            max_seed_mean > 0.0
            and np.isfinite(seed_mean_px)
            and seed_mean_px > max_seed_mean
        ):
            reasons.append("seed_mean")

        max_seed_max = float(
            getattr(
                self.config,
                "fast_persistent_dense_adaptive_max_seed_max_px",
                2.8,
            )
        )
        if (
            max_seed_max > 0.0
            and np.isfinite(seed_max_px)
            and seed_max_px > max_seed_max
        ):
            reasons.append("seed_max")

        if reasons:
            return True, "+".join(reasons), metrics

        return False, "clean_seed", metrics

    def _finish_fast_pose_transaction_cpp(
        self,
        detection,
        fast_transaction,
        fast_timings: Dict[str, float],
    ) -> Optional[TrackerResult]:
        """Package the C++ fast-pose transaction through the normal Python result path."""
        fast_result, cpp_depth_filter_prev_state = fast_transaction
        points = self._pose_track_points_from_cpp(
            getattr(fast_result, "points", [])
        )
        corners = self._tracker_corners_from_cpp(
            getattr(fast_result, "corners", [])
        )
        visual_corners = self._tracker_corners_from_cpp(
            getattr(fast_result, "visual_corners", [])
        )
        pose = self._map_pose_result_from_cpp(fast_result.pose)
        seed_pose = self._map_pose_result_from_cpp(fast_result.seed_pose)
        depth_filter_result = None
        depth_filtered_pose = None
        if bool(getattr(fast_result, "depth_filter_available", False)):
            depth_filter_result = getattr(fast_result, "depth_filter_result", None)
            depth_filtered_pose = self._map_pose_result_from_cpp(
                fast_result.depth_filtered_pose
            )
            if (
                depth_filtered_pose.rvec is None
                or depth_filtered_pose.tvec is None
                or depth_filtered_pose.T_marker_camera is None
            ):
                if cpp_depth_filter_prev_state is not None:
                    self.pose_depth_filter.restore(cpp_depth_filter_prev_state)
                depth_filter_result = None
                depth_filtered_pose = None
        stats = self._last_persistent_match_stats

        fast_timings["fast_persistent_seed_cpp_count"] = 1.0
        fast_timings["fast_persistent_transaction_cpp_count"] = 1.0
        fast_timings["fast_persistent_seed_cpp_total_ms"] = float(
            getattr(fast_result, "cpp_seed_total_ms", 0.0)
        )
        fast_timings["fast_persistent_seed_cpp_pose_ms"] = float(
            getattr(fast_result, "seed_pnp_ms", 0.0)
        )
        fast_timings["fast_persistent_seed_cpp_match_ms"] = float(
            getattr(fast_result, "persistent_match_ms", 0.0)
        )
        fast_timings["fast_persistent_transaction_cpp_total_ms"] = float(
            getattr(fast_result, "total_ms", 0.0)
        )
        fast_timings["fast_persistent_match_ms"] = float(
            getattr(fast_result, "persistent_match_ms", 0.0)
        )
        fast_timings["fast_persistent_seed_pnp_ms"] = float(
            getattr(fast_result, "seed_pnp_ms", 0.0)
        )
        if bool(getattr(fast_result, "depth_filter_available", False)):
            fast_timings["pose_depth_filter_cpp_ms"] = float(
                getattr(fast_result, "depth_filter_ms", 0.0)
            )
            fast_timings["pose_depth_filter_cpp_count"] = 1.0
        fast_timings["fast_persistent_points_count"] = float(len(points))
        fast_timings["fast_persistent_corners_count"] = float(len(corners))
        fast_timings["fast_persistent_match_cpp_count"] = 1.0
        fast_timings["fast_persistent_match_motion_px"] = float(
            getattr(stats, "adaptive_motion_px", 0.0)
        )
        fast_timings["fast_persistent_match_radius_px"] = float(
            getattr(stats, "adaptive_max_dist_px", 0.0)
        )
        fast_timings["fast_persistent_min_points_count"] = float(
            getattr(fast_result, "min_points", 0)
        )

        dense_metrics = getattr(fast_result, "dense_gate_metrics", None)
        if dense_metrics is not None:
            fast_timings["fast_dense_adaptive_required_count"] = (
                1.0 if bool(getattr(fast_result, "dense_required", False)) else 0.0
            )
            fast_timings["fast_dense_adaptive_skipped_count"] = (
                0.0 if bool(getattr(fast_result, "dense_required", False)) else 1.0
            )
            fast_timings["fast_dense_adaptive_match_ratio"] = float(
                getattr(dense_metrics, "match_ratio", 0.0)
            )
            fast_timings["fast_dense_adaptive_motion_px"] = float(
                getattr(dense_metrics, "motion_px", 0.0)
            )
            fast_timings["fast_dense_adaptive_ambiguous_count"] = float(
                getattr(dense_metrics, "ambiguous_count", 0.0)
            )
            fast_timings["fast_dense_adaptive_seed_mean_px"] = float(
                getattr(dense_metrics, "seed_mean_px", -1.0)
            )
            fast_timings["fast_dense_adaptive_seed_max_px"] = float(
                getattr(dense_metrics, "seed_max_px", -1.0)
            )

        dense_total_ms = float(getattr(fast_result, "dense_match_ms", 0.0)) + float(
            getattr(fast_result, "dense_pose_ms", 0.0)
        )
        fast_timings["fast_dense_total_ms"] = dense_total_ms
        if bool(getattr(fast_result, "route_decode", False)):
            fast_timings["fast_persistent_preflight_decode_route_count"] = 1.0
        if str(getattr(fast_result, "dense_reason", "")).startswith(
            "rescue_skipped_decode:"
        ):
            fast_timings["fast_dense_rescue_skipped_count"] = 1.0
            fast_timings["fast_dense_route_decode_count"] = 1.0

        self._set_fast_path_debug(
            attempted=True,
            success=bool(getattr(fast_result, "success", False)),
            reason=str(getattr(fast_result, "reason", "")),
            matches=len(points),
        )

        dense_reason = str(getattr(fast_result, "dense_reason", ""))
        dense_attempted = bool(getattr(fast_result, "dense_attempted", False))
        if dense_attempted or dense_reason:
            dense_stats = self._dense_projection_match_stats_from_cpp(
                fast_result.dense_stats
            )
            self._set_dense_refine_debug(
                attempted=dense_attempted,
                success=bool(getattr(fast_result, "dense_success", False)),
                reason=dense_reason,
                matches=int(getattr(fast_result, "dense_matches", 0)),
                median_error_px=float(dense_stats.median_error_px),
                p90_error_px=float(dense_stats.p90_error_px),
                stats=dense_stats,
            )

        if not bool(getattr(fast_result, "success", False)):
            return None

        if pose.rvec is None or pose.tvec is None or pose.T_marker_camera is None:
            if cpp_depth_filter_prev_state is not None:
                self.pose_depth_filter.restore(cpp_depth_filter_prev_state)
            self._set_fast_path_debug(
                attempted=True,
                reason="C++ fast pose missing pose vectors.",
                matches=len(points),
            )
            return None

        prev_pose_rvec = (
            None
            if self.pose_tracker.rvec is None
            else self.pose_tracker.rvec.copy()
        )
        prev_pose_tvec = (
            None
            if self.pose_tracker.tvec is None
            else self.pose_tracker.tvec.copy()
        )
        prev_pose_T = (
            None
            if self.pose_tracker.T_marker_camera is None
            else self.pose_tracker.T_marker_camera.copy()
        )
        prev_depth_filter_state = (
            self.pose_depth_filter.snapshot()
            if cpp_depth_filter_prev_state is None
            else cpp_depth_filter_prev_state
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

        self.pose_tracker.rvec = pose.rvec.copy()
        self.pose_tracker.tvec = pose.tvec.copy()
        self.pose_tracker.T_marker_camera = pose.T_marker_camera.copy()

        if bool(getattr(fast_result, "used_dense", False)):
            success_message = (
                "Fast pose estimated from dense projection correspondences "
                f"(seed_matches={seed_pose.num_inliers}, "
                f"dense_matches={len(points)}, "
                f"seed_median={float(getattr(fast_result.dense_stats, 'median_error_px', -1.0)):.3f}px, "
                f"seed_p90={float(getattr(fast_result.dense_stats, 'p90_error_px', -1.0)):.3f}px)."
            )
            precomputed_pnp_ms = float(getattr(fast_result, "dense_pose_ms", 0.0))
        else:
            success_message = (
                "Fast pose estimated from persistent correspondences "
                f"(matches={len(points)}, identities={stats.identities}, "
                f"far={stats.rejected_far}, ambiguous={stats.rejected_ambiguous}, "
                f"claimed={stats.rejected_claimed})."
            )
            precomputed_pnp_ms = float(getattr(fast_result, "seed_pnp_ms", 0.0))

        package_t0 = time.perf_counter()
        result = self._estimate_and_package_pose(
            points,
            corners,
            success_message=success_message,
            update_persistence=False,
            pose_source=PoseSource.FAST_PERSISTENT,
            detection=detection,
            precomputed_pose=pose,
            precomputed_pnp_ms=precomputed_pnp_ms,
            previous_pose_rvec=prev_pose_rvec,
            previous_pose_tvec=prev_pose_tvec,
            previous_pose_T=prev_pose_T,
            previous_depth_filter_state=prev_depth_filter_state,
            previous_last_rvec=prev_last_rvec,
            previous_last_tvec=prev_last_tvec,
            precomputed_visual_corners=visual_corners,
            precomputed_visual_is_final=bool(
                getattr(fast_result, "depth_filter_available", False)
            ),
            precomputed_depth_filter=depth_filter_result,
            precomputed_depth_pose=depth_filtered_pose,
            precomputed_accept_state=getattr(fast_result, "accepted_state", None),
        )
        fast_timings["fast_persistent_estimate_package_ms"] = (
            time.perf_counter() - package_t0
        ) * 1000.0
        result.timings_ms["persistent_match_ms"] = float(
            getattr(fast_result, "persistent_match_ms", 0.0)
        )
        result.timings_ms["fast_dense_total_ms"] = dense_total_ms
        result.timings_ms["fast_seed_pnp_ms"] = float(
            getattr(fast_result, "seed_pnp_ms", 0.0)
        )
        if bool(getattr(fast_result, "dense_attempted", False)):
            result.timings_ms["fast_dense_match_ms"] = float(
                getattr(fast_result, "dense_match_ms", 0.0)
            )
        if bool(getattr(fast_result, "depth_filter_available", False)):
            result.timings_ms["pose_depth_filter_cpp_ms"] = float(
                getattr(fast_result, "depth_filter_ms", 0.0)
            )

        if not result.success:
            if cpp_depth_filter_prev_state is not None:
                self.pose_depth_filter.restore(cpp_depth_filter_prev_state)
            self._set_fast_path_debug(
                attempted=True,
                reason=result.message,
                matches=len(points),
            )
            return None

        self._attach_fast_path_debug(result)
        result.confidence *= 0.95
        refresh_t0 = time.perf_counter()
        cpp_refresh_committed = (
            float(
                result.timings_ms.get(
                    "pose_visual_corners_cpp_precomputed_count",
                    0.0,
                )
            )
            > 0.5
            and not bool(getattr(result, "pose_plateau_prior_applied", False))
            and self._commit_cpp_persistent_refresh(fast_result)
        )
        if not cpp_refresh_committed:
            self._refresh_persistent_correspondences_from_result(
                result,
                max_mean_error_px=self.config.fast_persistent_refresh_mean_error_px,
            )
        result.timings_ms["fast_refresh_persistence_cpp_count"] = (
            1.0 if cpp_refresh_committed else 0.0
        )
        result.timings_ms["fast_refresh_persistence_cpp_ms"] = float(
            getattr(fast_result, "persistence_refresh_ms", 0.0)
        )
        result.timings_ms["fast_refresh_persistence_ms"] = (
            time.perf_counter() - refresh_t0
        ) * 1000.0
        return result

    def _try_fast_pose_from_persistent_correspondences(
        self,
        detection,
    ) -> Optional[TrackerResult]:
        """Attempt a fast pose solve from the persistence cache."""
        self._last_persistent_match_stats = PersistentMatchStats()
        fast_timings: Dict[str, float] = {}
        self._last_fast_path_timings = fast_timings

        def mark_fast_timing(name: str, start: float) -> None:
            fast_timings[name] = (time.perf_counter() - start) * 1000.0

        if self.config.decode_only_mode:
            self._set_fast_path_debug(
                attempted=False,
                reason="decode_only_mode",
            )
            return None

        if not self.config.enable_fast_persistent_path:
            self._set_fast_path_debug(
                attempted=False,
                reason="disabled",
            )
            return None

        if detection is None or not bool(detection.valid()):
            self._set_fast_path_debug(
                attempted=False,
                reason="invalid_detection",
            )
            return None

        fast_transaction = self._fast_pose_transaction_cpp(detection)
        if fast_transaction is not None:
            return self._finish_fast_pose_transaction_cpp(
                detection,
                fast_transaction,
                fast_timings,
            )

        seed_pose: Optional[MapPoseResult] = None
        seed_pnp_ms = 0.0
        cpp_seed_total_ms = 0.0
        seed_bundle = self._fast_persistent_seed_pose_cpp(detection)
        if seed_bundle is not None:
            (
                points,
                corners,
                seed_pose,
                persistent_match_ms,
                seed_pnp_ms,
                cpp_seed_total_ms,
            ) = seed_bundle
            fast_timings["fast_persistent_seed_cpp_count"] = 1.0
            fast_timings["fast_persistent_seed_cpp_total_ms"] = cpp_seed_total_ms
            fast_timings["fast_persistent_seed_cpp_pose_ms"] = seed_pnp_ms
            fast_timings["fast_persistent_seed_cpp_match_ms"] = persistent_match_ms
        else:
            match_t0 = time.perf_counter()
            points, corners = self._persistent_correspondences_for_detection(detection)
            persistent_match_ms = (time.perf_counter() - match_t0) * 1000.0
            fast_timings["fast_persistent_seed_cpp_count"] = 0.0
        fast_timings["fast_persistent_match_ms"] = persistent_match_ms
        fast_timings["fast_persistent_points_count"] = float(len(points))
        fast_timings["fast_persistent_corners_count"] = float(len(corners))
        stats = self._last_persistent_match_stats
        match_backend = str(
            getattr(self, "_last_persistent_match_backend", "unknown")
        )
        fast_timings["fast_persistent_match_cpp_count"] = (
            1.0 if match_backend in ("cpp", "cpp_seed") else 0.0
        )
        fast_timings["fast_persistent_match_motion_px"] = float(
            getattr(stats, "adaptive_motion_px", 0.0)
        )
        fast_timings["fast_persistent_match_radius_px"] = float(
            getattr(stats, "adaptive_max_dist_px", 0.0)
        )
        min_points = max(
            int(self.config.min_points),
            int(self.config.persistence_min_points),
            int(self.config.fast_persistent_min_points),
        )
        fast_timings["fast_persistent_min_points_count"] = float(min_points)
        if len(points) < min_points:
            self._set_fast_path_debug(
                attempted=True,
                reason=f"too_few_matches:{len(points)}<{min_points}",
                matches=len(points),
            )
            return None

        snapshot_t0 = time.perf_counter()
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
        prev_last_T = (
            None
            if self._last_accepted_T_marker_camera is None
            else self._last_accepted_T_marker_camera.copy()
        )
        prev_last_pose_frame = int(self._last_accepted_pose_frame)
        prev_last_good_reproj_px = float(self._last_good_reproj_px)
        prev_max_pts_seen = int(self._max_pts_seen)
        mark_fast_timing("fast_persistent_snapshot_ms", snapshot_t0)

        def restore_seed_state() -> None:
            restore_t0 = time.perf_counter()
            self.pose_tracker.rvec = prev_pose_rvec
            self.pose_tracker.tvec = prev_pose_tvec
            self.pose_tracker.T_marker_camera = prev_pose_T
            self.pose_depth_filter.restore(prev_depth_filter_state)
            self._last_accepted_rvec = (
                None if prev_last_rvec is None else prev_last_rvec.copy()
            )
            self._last_accepted_tvec = (
                None if prev_last_tvec is None else prev_last_tvec.copy()
            )
            self._last_accepted_T_marker_camera = (
                None if prev_last_T is None else prev_last_T.copy()
            )
            self._last_accepted_pose_frame = prev_last_pose_frame
            self._last_good_reproj_px = prev_last_good_reproj_px
            self._max_pts_seen = prev_max_pts_seen
            fast_timings["fast_persistent_restore_seed_ms"] = (
                fast_timings.get("fast_persistent_restore_seed_ms", 0.0)
                + (time.perf_counter() - restore_t0) * 1000.0
            )

        if seed_pose is None:
            seed_t0 = time.perf_counter()
            seed_pose = self.pose_tracker.estimate_pose(
                points,
                lost_frames=self.lost_frames,
            )
            seed_pnp_ms = (time.perf_counter() - seed_t0) * 1000.0
        elif (
            seed_pose.success
            and (
                seed_pose.rvec is None
                or seed_pose.tvec is None
                or seed_pose.T_marker_camera is None
            )
        ):
            seed_pose.success = False
            seed_pose.message = "C++ seed pose missing pose vectors."
        elif seed_pose.success:
            self.pose_tracker.rvec = seed_pose.rvec.copy()
            self.pose_tracker.tvec = seed_pose.tvec.copy()
            self.pose_tracker.T_marker_camera = seed_pose.T_marker_camera.copy()
        fast_timings["fast_persistent_seed_pnp_ms"] = seed_pnp_ms

        if not seed_pose.success:
            restore_seed_state()
            self._set_fast_path_debug(
                attempted=True,
                reason=seed_pose.message,
                matches=len(points),
            )
            return None

        if not self._persistent_pose_motion_plausible(
            seed_pose.rvec,
            seed_pose.tvec,
            prev_last_rvec,
            prev_last_tvec,
        ):
            restore_seed_state()
            self._set_fast_path_debug(
                attempted=True,
                reason="Persistent pose rejected by motion gate.",
                matches=len(points),
            )
            return None

        reject_reason = self._fallback_pose_rejection_reason(
            detection,
            seed_pose.rvec,
            seed_pose.tvec,
            seed_pose.reprojection_mean_px,
            seed_pose.reprojection_max_px,
        )
        if reject_reason:
            restore_seed_state()
            self._set_fast_path_debug(
                attempted=True,
                reason=reject_reason,
                matches=len(points),
            )
            return None

        debug_t0 = time.perf_counter()
        self._set_fast_path_debug(
            attempted=True,
            success=True,
            reason="ok",
            matches=len(points),
        )
        mark_fast_timing("fast_persistent_debug_success_ms", debug_t0)

        success_message = (
            "Fast pose estimated from persistent correspondences "
            f"(matches={len(points)}, identities={stats.identities}, "
            f"far={stats.rejected_far}, ambiguous={stats.rejected_ambiguous}, "
            f"claimed={stats.rejected_claimed})."
        )
        seed_result = TrackerResult(
            success=True,
            mode=self.mode,
            message=success_message,
            rvec=seed_pose.rvec,
            tvec=seed_pose.tvec,
            T_marker_camera=seed_pose.T_marker_camera,
            mean_reprojection_error_px=seed_pose.reprojection_mean_px,
            max_reprojection_error_px=seed_pose.reprojection_max_px,
            num_points=seed_pose.num_points,
            num_inliers=seed_pose.num_inliers,
            pose_source=PoseSource.FAST_PERSISTENT,
            pnp_method=str(getattr(seed_pose, "method", "")),
            corners=[],
            correspondence_corners=corners,
            timings_ms={"pnp_ms": seed_pnp_ms},
        )

        dense_required, dense_gate_reason, dense_gate_metrics = (
            self._fast_dense_refine_required_for_seed(
                seed_pose=seed_pose,
                match_count=len(points),
                stats=stats,
            )
        )
        fast_timings["fast_dense_adaptive_required_count"] = (
            1.0 if dense_required else 0.0
        )
        fast_timings["fast_dense_adaptive_skipped_count"] = (
            0.0 if dense_required else 1.0
        )
        fast_timings["fast_dense_adaptive_match_ratio"] = float(
            dense_gate_metrics["match_ratio"]
        )
        fast_timings["fast_dense_adaptive_motion_px"] = float(
            dense_gate_metrics["motion_px"]
        )
        fast_timings["fast_dense_adaptive_ambiguous_count"] = float(
            dense_gate_metrics["ambiguous_count"]
        )
        fast_timings["fast_dense_adaptive_seed_mean_px"] = float(
            dense_gate_metrics["seed_mean_px"]
        )
        fast_timings["fast_dense_adaptive_seed_max_px"] = float(
            dense_gate_metrics["seed_max_px"]
        )

        if dense_required:
            dense_t0 = time.perf_counter()
            dense_result = self._try_dense_projection_refine_from_fast_pose(
                detection,
                seed_result=seed_result,
            )
            dense_total_ms = (time.perf_counter() - dense_t0) * 1000.0
        else:
            dense_result = None
            dense_total_ms = 0.0
            self._set_dense_refine_debug(
                attempted=False,
                reason=f"adaptive_skip:{dense_gate_reason}",
            )
        fast_timings["fast_dense_total_ms"] = dense_total_ms
        if dense_result is not None:
            dense_result.timings_ms["persistent_match_ms"] = persistent_match_ms
            dense_result.timings_ms["fast_dense_total_ms"] = dense_total_ms
            dense_result.timings_ms["fast_seed_pnp_ms"] = seed_pnp_ms
            result = dense_result
        else:
            post_dense_t0 = time.perf_counter()
            debug = self._last_fast_path_debug
            dense_reason = str(getattr(debug, "dense_refine_reason", ""))
            current_corners = int(getattr(debug, "current_corners", 0))
            sparse_match_ratio = float(len(points)) / float(max(current_corners, 1))
            dense_validation_failed = (
                bool(getattr(debug, "dense_refine_attempted", False))
                and not bool(getattr(debug, "dense_refine_success", False))
                and current_corners
                >= int(self.config.fast_persistent_dense_min_points)
                and sparse_match_ratio
                < float(self.config.fast_persistent_dense_rescue_min_green_ratio)
                and dense_reason not in ("disabled", "missing_seed_pose")
            )
            mark_fast_timing(
                "fast_persistent_post_dense_gate_ms",
                post_dense_t0,
            )
            if (
                dense_reason.startswith("rescue_failed:")
                or dense_reason.startswith("rescue_skipped_decode:")
                or dense_validation_failed
            ):
                restore_seed_state()
                debug.success = False
                debug.reason = (
                    "dense_validation_rejected_seed:"
                    f"{dense_reason}; sparse_ratio={sparse_match_ratio:.3f}"
                )
                debug.matches = int(len(points))
                fast_timings["fast_persistent_preflight_decode_route_count"] = 1.0
                return None

            estimate_t0 = time.perf_counter()
            result = self._estimate_and_package_pose(
                points,
                corners,
                success_message=success_message,
                update_persistence=False,
                pose_source=PoseSource.FAST_PERSISTENT,
                detection=detection,
                precomputed_pose=seed_pose,
                precomputed_pnp_ms=seed_pnp_ms,
                previous_pose_rvec=prev_pose_rvec,
                previous_pose_tvec=prev_pose_tvec,
                previous_pose_T=prev_pose_T,
                previous_depth_filter_state=prev_depth_filter_state,
                previous_last_rvec=prev_last_rvec,
                previous_last_tvec=prev_last_tvec,
            )
            mark_fast_timing("fast_persistent_estimate_package_ms", estimate_t0)
            result.timings_ms["persistent_match_ms"] = persistent_match_ms
            result.timings_ms["fast_dense_total_ms"] = dense_total_ms
            result.timings_ms["fast_seed_pnp_ms"] = seed_pnp_ms

            if not result.success:
                restore_seed_state()
                self._set_fast_path_debug(
                    attempted=True,
                    reason=result.message,
                    matches=len(points),
                )
                return None

        self._attach_fast_path_debug(result)
        result.confidence *= 0.95
        refresh_t0 = time.perf_counter()
        self._refresh_persistent_correspondences_from_result(
            result,
            max_mean_error_px=self.config.fast_persistent_refresh_mean_error_px,
        )
        result.timings_ms["fast_refresh_persistence_ms"] = (
            time.perf_counter() - refresh_t0
        ) * 1000.0
        return result

    def _try_dense_projection_refine_from_fast_pose(
        self,
        detection,
        seed_result: TrackerResult,
    ) -> Optional[TrackerResult]:
        """Try to improve a successful fast-path pose with dense projection matches."""
        if not self.config.fast_persistent_dense_refine_enabled:
            self._set_dense_refine_debug(
                attempted=False,
                reason="disabled",
            )
            return None

        if (
            detection is None
            or seed_result.rvec is None
            or seed_result.tvec is None
        ):
            self._set_dense_refine_debug(
                attempted=False,
                reason="missing_seed_pose",
            )
            return None

        min_dense_points = max(
            int(self.config.fast_persistent_dense_min_points),
            int(seed_result.num_inliers) + 1,
        )
        detection_corners = getattr(detection, "corners", None)
        detected_corner_count = 0 if detection_corners is None else len(detection_corners)
        if detected_corner_count < min_dense_points:
            # Dense matching cannot produce more matches than detected corners.
            self._set_dense_refine_debug(
                attempted=True,
                reason=(
                    "too_few_detection_corners:"
                    f"{detected_corner_count}<{min_dense_points}"
                ),
                matches=detected_corner_count,
            )
            return None

        match_t0 = time.perf_counter()
        matched_corners, dense_stats = (
            self._strict_projected_tracker_corners_for_detection_pose(
                detection,
                seed_result.rvec,
                seed_result.tvec,
                max_dist_px=self.config.fast_persistent_dense_match_max_px,
                ambiguity_margin_px=(
                    self.config.fast_persistent_dense_min_second_best_margin_px
                ),
            )
        )
        match_ms = (time.perf_counter() - match_t0) * 1000.0
        match_count = len(matched_corners)
        median_err = float(dense_stats.median_error_px)
        p90_err = float(dense_stats.p90_error_px)

        if match_count < min_dense_points:
            self._set_dense_refine_debug(
                attempted=True,
                reason=f"too_few_matches:{match_count}<{min_dense_points}",
                matches=match_count,
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        if (
            dense_stats.distinct_rows
            < self.config.fast_persistent_dense_min_distinct_rows
            or dense_stats.distinct_cols
            < self.config.fast_persistent_dense_min_distinct_cols
        ):
            self._set_dense_refine_debug(
                attempted=True,
                reason=(
                    "poor_grid_spread:"
                    f"{dense_stats.distinct_rows}x{dense_stats.distinct_cols}"
                ),
                matches=match_count,
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        if (
            dense_stats.object_span_mm
            < self.config.fast_persistent_dense_min_object_span_mm
        ):
            self._set_dense_refine_debug(
                attempted=True,
                reason=f"poor_object_span:{dense_stats.object_span_mm:.1f}mm",
                matches=match_count,
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        min_coverage = float(self.config.fast_persistent_dense_min_image_coverage)
        if 0.0 <= dense_stats.image_coverage < min_coverage:
            self._set_dense_refine_debug(
                attempted=True,
                reason=f"poor_image_coverage:{dense_stats.image_coverage:.3f}",
                matches=match_count,
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        points, corners = self._points_from_correspondences(matched_corners)
        if len(points) < min_dense_points:
            self._set_dense_refine_debug(
                attempted=True,
                reason=f"too_few_unique_points:{len(points)}<{min_dense_points}",
                matches=len(points),
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        seed_error_reason = ""
        if median_err > self.config.fast_persistent_dense_max_median_px:
            seed_error_reason = f"median_error:{median_err:.3f}"
        elif p90_err > self.config.fast_persistent_dense_max_p90_px:
            seed_error_reason = f"p90_error:{p90_err:.3f}"

        seed_green_ratio = float(seed_result.num_inliers) / float(
            max(int(dense_stats.detected), 1)
        )
        rescue_required = bool(seed_error_reason) and (
            seed_error_reason.startswith("p90_error")
            or seed_green_ratio
            < float(self.config.fast_persistent_dense_rescue_min_green_ratio)
            or median_err
            >= float(self.config.fast_persistent_dense_rescue_min_seed_median_px)
        )
        if (
            rescue_required
            and not bool(self.config.fast_persistent_dense_rescue_enabled)
        ):
            fast_timings = getattr(self, "_last_fast_path_timings", None)
            if fast_timings is not None:
                fast_timings["fast_dense_rescue_skipped_count"] = 1.0
                fast_timings["fast_dense_route_decode_count"] = 1.0
            self._set_dense_refine_debug(
                attempted=True,
                reason=f"rescue_skipped_decode:{seed_error_reason}",
                matches=match_count,
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        if seed_error_reason and not rescue_required:
            self._set_dense_refine_debug(
                attempted=True,
                reason=seed_error_reason,
                matches=match_count,
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        seed_rvec = np.asarray(seed_result.rvec, dtype=np.float64).reshape(3, 1).copy()
        seed_tvec = np.asarray(seed_result.tvec, dtype=np.float64).reshape(3, 1).copy()
        seed_T = (
            None
            if seed_result.T_marker_camera is None
            else np.asarray(seed_result.T_marker_camera, dtype=np.float64).reshape(4, 4).copy()
        )
        seed_last_rvec = (
            None
            if self._last_accepted_rvec is None
            else np.asarray(self._last_accepted_rvec, dtype=np.float64).reshape(3, 1).copy()
        )
        seed_last_tvec = (
            None
            if self._last_accepted_tvec is None
            else np.asarray(self._last_accepted_tvec, dtype=np.float64).reshape(3, 1).copy()
        )
        seed_last_T = (
            None
            if self._last_accepted_T_marker_camera is None
            else np.asarray(self._last_accepted_T_marker_camera, dtype=np.float64).reshape(4, 4).copy()
        )
        seed_last_pose_frame = int(self._last_accepted_pose_frame)
        seed_last_good_reproj_px = float(self._last_good_reproj_px)
        seed_max_pts_seen = int(self._max_pts_seen)

        self.pose_tracker.rvec = seed_rvec.copy()
        self.pose_tracker.tvec = seed_tvec.copy()
        self.pose_tracker.T_marker_camera = None if seed_T is None else seed_T.copy()

        dense_message = (
            "Fast pose estimated from dense projection correspondences "
            f"(seed_matches={seed_result.num_inliers}, "
            f"dense_matches={len(points)}, "
            f"seed_median={median_err:.3f}px, "
            f"seed_p90={p90_err:.3f}px)."
        )
        dense_solver = str(
            self.config.fast_persistent_dense_pose_solver or "direct_prior"
        ).lower()
        use_robust_solver = rescue_required or dense_solver in (
            "sqpnp",
            "robust_sqpnp",
            "sqpnp_trim",
            "robust",
        )
        if use_robust_solver:
            dense_result = self._estimate_dense_pose_with_robust_solver(
                points,
                corners,
                success_message=dense_message,
                pose_source=PoseSource.FAST_PERSISTENT,
                detection=detection,
            )
        else:
            dense_result = self._estimate_dense_pose_with_direct_prior_cpp(
                points,
                corners,
                success_message=dense_message,
                pose_source=PoseSource.FAST_PERSISTENT,
                detection=detection,
            )
            if dense_result is None:
                dense_result = self._estimate_and_package_pose(
                    points,
                    corners,
                    success_message=dense_message,
                    update_persistence=False,
                    pose_source=PoseSource.FAST_PERSISTENT,
                    detection=detection,
                )
        dense_result.timings_ms["fast_dense_match_ms"] = match_ms

        if (
            not dense_result.success
            or dense_result.num_inliers < min_dense_points
        ):
            if not dense_result.success:
                reject_reason = dense_result.message
            else:
                reject_reason = (
                    f"too_few_inliers:{dense_result.num_inliers}<"
                    f"{min_dense_points}"
                )
            if seed_error_reason:
                reject_reason = f"rescue_failed:{seed_error_reason}; {reject_reason}"

            self.pose_tracker.rvec = seed_rvec.copy()
            self.pose_tracker.tvec = seed_tvec.copy()
            self.pose_tracker.T_marker_camera = None if seed_T is None else seed_T.copy()
            self._last_accepted_rvec = (
                None if seed_last_rvec is None else seed_last_rvec.copy()
            )
            self._last_accepted_tvec = (
                None if seed_last_tvec is None else seed_last_tvec.copy()
            )
            self._last_accepted_T_marker_camera = (
                None if seed_last_T is None else seed_last_T.copy()
            )
            self._last_accepted_pose_frame = seed_last_pose_frame
            self._last_good_reproj_px = seed_last_good_reproj_px
            self._max_pts_seen = seed_max_pts_seen
            self._set_dense_refine_debug(
                attempted=True,
                reason=reject_reason,
                matches=len(points),
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        self._set_dense_refine_debug(
            attempted=True,
            success=True,
            reason=f"rescue_ok:{seed_error_reason}" if seed_error_reason else "ok",
            matches=len(points),
            median_error_px=median_err,
            p90_error_px=p90_err,
            stats=dense_stats,
        )
        return dense_result


@dataclass
class TrackerConsistencyConfig:
    """Thresholds for standalone temporal and semantic consistency checks."""

    max_translation_jump_mm: float = 120.0
    max_rotation_jump_deg: float = 45.0

    min_identity_overlap_count: int = 6
    min_identity_overlap_ratio: float = 0.70
    max_identity_conflict_ratio: float = 0.30

    min_rotation_vote_count: int = 3
    min_rotation_vote_ratio: float = 0.70


@dataclass
class ConsistencyResult:
    """Pose-jump consistency result."""

    accepted: bool
    reason: str = ""

    translation_jump_mm: float = 0.0
    rotation_jump_deg: float = 0.0


@dataclass
class IdentityConsistencyResult:
    """Identity-overlap consistency result."""

    accepted: bool
    reason: str = ""

    checked_local: int = 0
    consistent_local: int = 0
    checked_global: int = 0
    consistent_global: int = 0

    local_ratio: float = 1.0
    global_ratio: float = 1.0
    conflict_count: int = 0
    conflict_ratio: float = 0.0

    conflicts: list[str] = field(default_factory=list)


@dataclass
class RotationConsistencyResult:
    """Patch-rotation consistency result."""

    accepted: bool
    reason: str = ""

    dominant_rotation_deg: Optional[int] = None
    dominant_count: int = 0
    total_count: int = 0
    dominant_ratio: float = 0.0


@dataclass
class CombinedConsistencyResult:
    """Combined consistency result for pose, identity, and rotation checks."""

    accepted: bool
    reason: str = ""

    pose: ConsistencyResult = field(
        default_factory=lambda: ConsistencyResult(True, "Pose check not evaluated.")
    )
    identity: IdentityConsistencyResult = field(
        default_factory=lambda: IdentityConsistencyResult(True, "Identity check not evaluated.")
    )
    rotation: RotationConsistencyResult = field(
        default_factory=lambda: RotationConsistencyResult(True, "Rotation check not evaluated.")
    )


class TrackerConsistency:
    """
    Temporal / semantic consistency layer for HydraMarker tracking.

    This class intentionally keeps the tracker.py orchestration clean.
    It validates:
        - pose jumps,
        - local/global identity consistency,
        - decoded patch rotation consistency,
        - combined tracking decisions.
    """

    def __init__(self, config: TrackerConsistencyConfig) -> None:
        self.config = config
        self.last_dominant_rotation_deg: Optional[int] = None

    def reset(self) -> None:
        self.last_dominant_rotation_deg = None

    def validate_pose_jump(
        self,
        previous_rvec: Optional[np.ndarray],
        previous_tvec: Optional[np.ndarray],
        candidate_rvec: Optional[np.ndarray],
        candidate_tvec: Optional[np.ndarray],
    ) -> ConsistencyResult:
        """Validate a candidate pose against the previous pose motion envelope."""
        if previous_rvec is None or previous_tvec is None:
            return ConsistencyResult(
                accepted=True,
                reason="No previous pose available.",
            )

        if candidate_rvec is None or candidate_tvec is None:
            return ConsistencyResult(
                accepted=False,
                reason="Candidate pose is incomplete.",
            )

        previous_t = np.asarray(previous_tvec, dtype=np.float64).reshape(3, 1)
        candidate_t = np.asarray(candidate_tvec, dtype=np.float64).reshape(3, 1)

        translation_jump_mm = float(np.linalg.norm(candidate_t - previous_t))

        R_prev, _ = cv2.Rodrigues(
            np.asarray(previous_rvec, dtype=np.float64).reshape(3, 1)
        )
        R_candidate, _ = cv2.Rodrigues(
            np.asarray(candidate_rvec, dtype=np.float64).reshape(3, 1)
        )

        R_delta = R_candidate @ R_prev.T

        cos_angle = np.clip(
            (np.trace(R_delta) - 1.0) * 0.5,
            -1.0,
            1.0,
        )

        rotation_jump_deg = float(np.degrees(np.arccos(cos_angle)))

        if translation_jump_mm > self.config.max_translation_jump_mm:
            return ConsistencyResult(
                accepted=False,
                reason=(
                    f"Translation jump too large: "
                    f"{translation_jump_mm:.2f} mm > "
                    f"{self.config.max_translation_jump_mm:.2f} mm."
                ),
                translation_jump_mm=translation_jump_mm,
                rotation_jump_deg=rotation_jump_deg,
            )

        if rotation_jump_deg > self.config.max_rotation_jump_deg:
            return ConsistencyResult(
                accepted=False,
                reason=(
                    f"Rotation jump too large: "
                    f"{rotation_jump_deg:.2f} deg > "
                    f"{self.config.max_rotation_jump_deg:.2f} deg."
                ),
                translation_jump_mm=translation_jump_mm,
                rotation_jump_deg=rotation_jump_deg,
            )

        return ConsistencyResult(
            accepted=True,
            reason="Pose jump plausible.",
            translation_jump_mm=translation_jump_mm,
            rotation_jump_deg=rotation_jump_deg,
        )

    def validate_identity_consistency(
        self,
        previous_identities,
        candidate_identities,
        *,
        check_global_to_local: bool = True,
    ) -> IdentityConsistencyResult:
        """Validate local/global identity continuity between two correspondence sets."""
        previous = list(previous_identities or [])
        candidate = list(candidate_identities or [])

        if not previous:
            return IdentityConsistencyResult(
                accepted=True,
                reason="No previous identities available.",
            )

        if not candidate:
            return IdentityConsistencyResult(
                accepted=False,
                reason="No candidate identities available.",
            )

        previous_by_local = {
            self._local_key(p): self._global_key(p)
            for p in previous
        }

        previous_by_global = {
            self._global_key(p): self._local_key(p)
            for p in previous
        }

        checked_local = 0
        consistent_local = 0

        checked_global = 0
        consistent_global = 0

        conflicts: list[str] = []

        for p in candidate:
            local_key = self._local_key(p)
            global_key = self._global_key(p)

            old_global = previous_by_local.get(local_key)
            if old_global is not None:
                checked_local += 1
                if old_global == global_key:
                    consistent_local += 1
                elif len(conflicts) < 8:
                    conflicts.append(
                        f"local {local_key}: old global {old_global}, "
                        f"new global {global_key}"
                    )

            if check_global_to_local:
                old_local = previous_by_global.get(global_key)
                if old_local is not None:
                    checked_global += 1
                    if old_local == local_key:
                        consistent_global += 1
                    elif len(conflicts) < 8:
                        conflicts.append(
                            f"global {global_key}: old local {old_local}, "
                            f"new local {local_key}"
                        )

        local_ratio = (
            float(consistent_local) / float(checked_local)
            if checked_local > 0
            else 1.0
        )

        global_ratio = (
            float(consistent_global) / float(checked_global)
            if checked_global > 0
            else 1.0
        )

        total_checked = checked_local + checked_global
        total_consistent = consistent_local + consistent_global
        conflict_count = total_checked - total_consistent

        conflict_ratio = (
            float(conflict_count) / float(total_checked)
            if total_checked > 0
            else 0.0
        )

        enough_local = checked_local >= self.config.min_identity_overlap_count
        enough_global = checked_global >= self.config.min_identity_overlap_count

        local_ok = (
            not enough_local
            or local_ratio >= self.config.min_identity_overlap_ratio
        )

        global_ok = (
            not enough_global
            or global_ratio >= self.config.min_identity_overlap_ratio
        )

        conflict_ok = conflict_ratio <= self.config.max_identity_conflict_ratio

        accepted = bool(local_ok and global_ok and conflict_ok)

        reason = (
            f"local {consistent_local}/{checked_local} "
            f"({local_ratio:.2f}), "
            f"global {consistent_global}/{checked_global} "
            f"({global_ratio:.2f}), "
            f"conflicts={conflict_count}/{total_checked} "
            f"({conflict_ratio:.2f})"
        )

        if conflicts:
            reason += "; " + " | ".join(conflicts)

        return IdentityConsistencyResult(
            accepted=accepted,
            reason=reason,
            checked_local=checked_local,
            consistent_local=consistent_local,
            checked_global=checked_global,
            consistent_global=consistent_global,
            local_ratio=local_ratio,
            global_ratio=global_ratio,
            conflict_count=conflict_count,
            conflict_ratio=conflict_ratio,
            conflicts=conflicts,
        )

    def validate_rotation_consistency(
        self,
        rotations_deg: Sequence[int],
        *,
        update_state_on_accept: bool = False,
    ) -> RotationConsistencyResult:
        """Validate that decoded patch rotations agree with the recent dominant rotation."""
        rotations = [
            self._normalize_rotation_deg(r)
            for r in rotations_deg
            if r is not None
        ]

        if not rotations:
            return RotationConsistencyResult(
                accepted=True,
                reason="No patch rotations available.",
            )

        counts: dict[int, int] = {}
        for r in rotations:
            counts[r] = counts.get(r, 0) + 1

        dominant_rotation = max(counts, key=counts.get)
        dominant_count = int(counts[dominant_rotation])
        total_count = int(len(rotations))
        dominant_ratio = float(dominant_count) / float(total_count)

        enough_votes = total_count >= self.config.min_rotation_vote_count
        enough_ratio = dominant_ratio >= self.config.min_rotation_vote_ratio

        same_as_previous = (
            self.last_dominant_rotation_deg is None
            or dominant_rotation == self.last_dominant_rotation_deg
        )

        accepted = bool((not enough_votes or enough_ratio) and same_as_previous)

        if not same_as_previous:
            reason = (
                f"Dominant patch rotation changed: "
                f"{self.last_dominant_rotation_deg} -> {dominant_rotation} deg."
            )
        elif enough_votes and not enough_ratio:
            reason = (
                f"No stable dominant patch rotation: "
                f"{dominant_count}/{total_count} "
                f"({dominant_ratio:.2f})."
            )
        else:
            reason = (
                f"Dominant patch rotation stable: "
                f"{dominant_rotation} deg, "
                f"{dominant_count}/{total_count} "
                f"({dominant_ratio:.2f})."
            )

        if accepted and update_state_on_accept:
            self.last_dominant_rotation_deg = dominant_rotation

        return RotationConsistencyResult(
            accepted=accepted,
            reason=reason,
            dominant_rotation_deg=dominant_rotation,
            dominant_count=dominant_count,
            total_count=total_count,
            dominant_ratio=dominant_ratio,
        )

    def validate_combined(
        self,
        *,
        previous_rvec: Optional[np.ndarray],
        previous_tvec: Optional[np.ndarray],
        candidate_rvec: Optional[np.ndarray],
        candidate_tvec: Optional[np.ndarray],
        previous_identities=None,
        candidate_identities=None,
        decoded_rotations_deg: Optional[Sequence[int]] = None,
        check_global_to_local: bool = True,
        update_rotation_state_on_accept: bool = False,
    ) -> CombinedConsistencyResult:
        """Evaluate pose, identity, and rotation consistency as one decision."""
        pose_result = self.validate_pose_jump(
            previous_rvec=previous_rvec,
            previous_tvec=previous_tvec,
            candidate_rvec=candidate_rvec,
            candidate_tvec=candidate_tvec,
        )

        identity_result = self.validate_identity_consistency(
            previous_identities=previous_identities,
            candidate_identities=candidate_identities,
            check_global_to_local=check_global_to_local,
        )

        rotation_result = self.validate_rotation_consistency(
            decoded_rotations_deg or [],
            update_state_on_accept=False,
        )

        accepted = bool(
            identity_result.accepted
            and rotation_result.accepted
            and (
                pose_result.accepted
                or identity_result.local_ratio >= 0.95
                or identity_result.global_ratio >= 0.95
            )
        )

        reasons = [
            f"pose: {pose_result.reason}",
            f"identity: {identity_result.reason}",
            f"rotation: {rotation_result.reason}",
        ]

        if accepted and update_rotation_state_on_accept:
            if rotation_result.dominant_rotation_deg is not None:
                self.last_dominant_rotation_deg = rotation_result.dominant_rotation_deg

        return CombinedConsistencyResult(
            accepted=accepted,
            reason=" | ".join(reasons),
            pose=pose_result,
            identity=identity_result,
            rotation=rotation_result,
        )

    @staticmethod
    def _local_key(p) -> GridKey:
        return int(p.local_row), int(p.local_col)

    @staticmethod
    def _global_key(p) -> GridKey:
        return int(p.global_row), int(p.global_col)

    @staticmethod
    def _normalize_rotation_deg(rotation_deg: int) -> int:
        r = int(round(float(rotation_deg))) % 360

        if r in (0, 90, 180, 270):
            return r

        allowed = np.array([0, 90, 180, 270], dtype=np.float64)
        idx = int(np.argmin(np.abs(allowed - float(r))))
        return int(allowed[idx])


@dataclass
class TrackerLogEvent:
    """Single tracker log line before formatting."""

    level: str
    stage: str
    frame_index: int
    message: str


class TrackerLogger:
    """Small text logger for throttled tracker-result diagnostics."""

    def __init__(
        self,
        log_path: str = "hydramarker_tracker.log",
        enable_console: bool = False,
    ) -> None:
        """Open a fresh tracker log at the configured path."""
        self.log_path = Path(log_path)
        self.enable_console = bool(enable_console)

        self._last_signature = None

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=== HydraMarker Tracker Log ===\n")

    def info(
        self,
        stage: str,
        frame_index: int,
        message: str,
    ) -> None:
        self._write(
            TrackerLogEvent(
                level="INFO",
                stage=stage,
                frame_index=frame_index,
                message=message,
            )
        )

    def warn(
        self,
        stage: str,
        frame_index: int,
        message: str,
    ) -> None:
        self._write(
            TrackerLogEvent(
                level="WARN",
                stage=stage,
                frame_index=frame_index,
                message=message,
            )
        )

    def error(
        self,
        stage: str,
        frame_index: int,
        message: str,
    ) -> None:
        self._write(
            TrackerLogEvent(
                level="ERROR",
                stage=stage,
                frame_index=frame_index,
                message=message,
            )
        )

    def log_tracker_result(
        self,
        stage: str,
        frame_index: int,
        result: TrackerResult,
        *,
        decode_only: bool,
        lost_frames: int,
        persisted_count: int,
    ) -> None:
        """Write a compact tracker-result summary if the throttling policy allows it."""
        if not self._should_log_tracker_result(frame_index, result):
            return

        fast = result.fast_path_debug
        policy = "decode_only" if decode_only else "tracking"
        message = (
            f"mode={result.mode.value} | "
            f"policy={policy} | "
            f"success={result.success} | "
            f"source={result.pose_source.value} | "
            f"pnp={result.pnp_method} | "
            f"fast={int(fast.attempted)}/{int(fast.success)}:"
            f"{fast.matches}:{fast.reason} | "
            f"msg={result.message} | "
            f"det_valid={result.detection_valid} | "
            f"det_tracking={result.detection_tracking} | "
            f"det_stable={result.detection_stable} | "
            f"det={len(result.detection_corners)} | "
            f"corr={len(result.correspondence_corners)} | "
            f"pose={len(result.corners)} | "
            f"points={result.num_points} | "
            f"inliers={result.num_inliers} | "
            f"mean_err={result.mean_reprojection_error_px:.3f} | "
            f"max_err={result.max_reprojection_error_px:.3f} | "
            f"lost_frames={lost_frames} | "
            f"persisted={persisted_count}"
        )

        if result.success:
            self.info(stage, frame_index, message)
        else:
            self.warn(stage, frame_index, message)

    @staticmethod
    def _should_log_tracker_result(
        frame_index: int,
        result: TrackerResult,
    ) -> bool:
        """Return whether a frame result should bypass log throttling."""
        return (
            not result.success
            or frame_index <= 5
            or frame_index % 30 == 0
            or "persistent" in result.message.lower()
        )

    def _write(self, event: TrackerLogEvent) -> None:
        """Format and append one deduplicated log event."""
        signature = (
            event.level,
            event.stage,
            event.message,
        )

        if signature == self._last_signature:
            return

        self._last_signature = signature

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        line = (
            f"[{timestamp}] "
            f"[{event.level}] "
            f"[frame={event.frame_index}] "
            f"[{event.stage}] "
            f"{event.message}"
        )

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if self.enable_console:
            print(line)
