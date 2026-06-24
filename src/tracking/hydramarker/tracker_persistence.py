"""Persistent identity cache, fast-path tracking, consistency checks, and logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tracking.hydramarker.tracker_pose import PoseTrackPoint
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

        use_pose_projection = (
            self.config.persistence_use_pose_projection
            and self.pose_tracker.rvec is not None
            and self.pose_tracker.tvec is not None
            and self._last_good_reproj_px >= 0.0
            and self._last_good_reproj_px
            <= self.config.persistence_projection_max_pose_error_px
        )
        stats.used_pose_projection = bool(use_pose_projection)

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
                max_dist = float(self.config.persistence_projection_max_reproj_px)
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

    def _clear_persistent_correspondences(self) -> None:
        """Clear all cached persistent identities."""
        self._identity_store.clear()
        self._persistent_frame_index = -1
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

    def _try_fast_pose_from_persistent_correspondences(
        self,
        detection,
    ) -> Optional[TrackerResult]:
        """Attempt a fast pose solve from the persistence cache."""
        self._last_persistent_match_stats = PersistentMatchStats()

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

        match_t0 = time.perf_counter()
        points, corners = self._persistent_correspondences_for_detection(detection)
        persistent_match_ms = (time.perf_counter() - match_t0) * 1000.0
        min_points = max(
            int(self.config.min_points),
            int(self.config.persistence_min_points),
            int(self.config.fast_persistent_min_points),
        )
        if len(points) < min_points:
            self._set_fast_path_debug(
                attempted=True,
                reason=f"too_few_matches:{len(points)}<{min_points}",
                matches=len(points),
            )
            return None

        stats = self._last_persistent_match_stats
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

        def restore_seed_state() -> None:
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

        result = self._estimate_and_package_pose(
            points,
            corners,
            success_message=(
                "Fast pose estimated from persistent correspondences "
                f"(matches={len(points)}, identities={stats.identities}, "
                f"far={stats.rejected_far}, ambiguous={stats.rejected_ambiguous}, "
                f"claimed={stats.rejected_claimed})."
            ),
            update_persistence=False,
            pose_source=PoseSource.FAST_PERSISTENT,
            detection=detection,
        )
        result.timings_ms["persistent_match_ms"] = persistent_match_ms

        if not result.success:
            restore_seed_state()
            self._set_fast_path_debug(
                attempted=True,
                reason=result.message,
                matches=len(points),
            )
            return None

        self._set_fast_path_debug(
            attempted=True,
            success=True,
            reason="ok",
            matches=len(points),
        )

        dense_t0 = time.perf_counter()
        dense_result = self._try_dense_projection_refine_from_fast_pose(
            detection,
            seed_result=result,
        )
        dense_total_ms = (time.perf_counter() - dense_t0) * 1000.0
        if dense_result is not None:
            dense_result.timings_ms["persistent_match_ms"] = persistent_match_ms
            dense_result.timings_ms["fast_dense_total_ms"] = dense_total_ms
            dense_result.timings_ms["fast_seed_pnp_ms"] = result.timings_ms.get(
                "pnp_ms",
                0.0,
            )
            result = dense_result
        else:
            result.timings_ms["fast_dense_total_ms"] = dense_total_ms
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
            if dense_reason.startswith("rescue_failed:") or dense_validation_failed:
                restore_seed_state()
                debug.success = False
                debug.reason = (
                    "dense_validation_rejected_seed:"
                    f"{dense_reason}; sparse_ratio={sparse_match_ratio:.3f}"
                )
                debug.matches = int(len(points))
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

        min_dense_points = max(
            int(self.config.fast_persistent_dense_min_points),
            int(seed_result.num_inliers) + 1,
        )
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
