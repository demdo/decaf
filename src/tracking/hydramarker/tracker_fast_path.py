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


class FastPathMixin:
    def _try_fast_pose_from_persistent_correspondences(
        self,
        detection,
    ) -> Optional[TrackerResult]:
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
            self.pose_tracker.rvec = prev_pose_rvec
            self.pose_tracker.tvec = prev_pose_tvec
            self.pose_tracker.T_marker_camera = prev_pose_T
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

        dense_result = self._try_dense_projection_refine_from_fast_pose(
            detection,
            seed_result=result,
        )
        if dense_result is not None:
            dense_result.timings_ms["persistent_match_ms"] = persistent_match_ms
            dense_result.timings_ms["fast_seed_pnp_ms"] = result.timings_ms.get(
                "pnp_ms",
                0.0,
            )
            result = dense_result

        self._attach_fast_path_debug(result)
        result.confidence *= 0.95
        self._refresh_persistent_correspondences_from_result(
            result,
            max_mean_error_px=self.config.fast_persistent_refresh_mean_error_px,
        )
        return result

    def _try_dense_projection_refine_from_fast_pose(
        self,
        detection,
        seed_result: TrackerResult,
    ) -> Optional[TrackerResult]:
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

        if median_err > self.config.fast_persistent_dense_max_median_px:
            self._set_dense_refine_debug(
                attempted=True,
                reason=f"median_error:{median_err:.3f}",
                matches=match_count,
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        if p90_err > self.config.fast_persistent_dense_max_p90_px:
            self._set_dense_refine_debug(
                attempted=True,
                reason=f"p90_error:{p90_err:.3f}",
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
        if dense_solver in ("sqpnp", "robust_sqpnp", "sqpnp_trim", "robust"):
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
                reason=(
                    dense_result.message
                    if not dense_result.success
                    else f"too_few_inliers:{dense_result.num_inliers}<{min_dense_points}"
                ),
                matches=len(points),
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        self._set_dense_refine_debug(
            attempted=True,
            success=True,
            reason="ok",
            matches=len(points),
            median_error_px=median_err,
            p90_error_px=p90_err,
            stats=dense_stats,
        )
        return dense_result

