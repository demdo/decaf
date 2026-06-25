"""Decode-side tracker stages from raw checkerboard detections to correspondences."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tracking.hydramarker.backend import cpp_impl as hm
from tracking.hydramarker.tracker_pose import MapPoseTracker, MapPoseTrackerConfig, PoseTrackPoint
from tracking.hydramarker.tracker_types import (
    DenseProjectionMatchStats,
    DetectedCorner,
    FastPathDebug,
    GlobalCornerIdentity,
    PersistentMatchStats,
    PoseSource,
    TrackerCorner,
    TrackerMode,
    TrackerResult,
)
from tracking.pose_solvers import make_transform_from_rvec_tvec


class TrackerFactoryMixin:
    """Create backend detectors, decoders, builders, and the pose tracker."""

    def _create_checkerboard_detector(self):
        """Build the C++ checkerboard detector from central tracker config."""
        cfg = hm.CheckerboardDetectorConfig()
        cfg.recovery_correction_weight = 0.5
        cfg.recovery_correction_max_dist_rel = 0.6
        if hasattr(cfg, "refresh_interval_frames"):
            cfg.refresh_interval_frames = int(
                self.config.checker_refresh_interval_frames
            )
        if hasattr(cfg, "tracking_recovery_stable_interval_frames"):
            cfg.tracking_recovery_stable_interval_frames = int(
                self.config.checker_tracking_recovery_stable_interval_frames
            )
        if hasattr(cfg, "tracking_local_completion_skip_enabled"):
            cfg.tracking_local_completion_skip_enabled = bool(
                self.config.checker_local_completion_skip_enabled
            )
        if hasattr(cfg, "tracking_local_completion_probe_interval_frames"):
            cfg.tracking_local_completion_probe_interval_frames = int(
                self.config.checker_local_completion_probe_interval_frames
            )
        if hasattr(cfg, "min_tracking_decode_cell_span"):
            cfg.min_tracking_decode_cell_span = (
                self.config.checker_min_tracking_decode_cell_span
            )
        if hasattr(cfg, "max_undecodeable_tracking_frames"):
            cfg.max_undecodeable_tracking_frames = (
                self.config.checker_max_undecodeable_tracking_frames
            )
        return hm.CheckerboardDetector(cfg)

    def _create_patch_extractor(self):
        return hm.PatchExtractor()

    def _create_pose_tracker(
        self,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray],
    ) -> MapPoseTracker:
        """Create the PnP pose tracker with thresholds copied from TrackerConfig."""
        return MapPoseTracker(
            K=K,
            dist_coeffs=dist_coeffs,
            config=MapPoseTrackerConfig(
                min_points=self.config.min_points,
                min_inliers=self.config.min_inliers,
                ransac_reproj_px=self.config.pnp_ransac_reprojection_px,
                ransac_confidence=self.config.pnp_ransac_confidence,
                ransac_iterations=self.config.pnp_ransac_iterations,
                max_mean_reproj_px=self.config.max_mean_reprojection_error_px,
                max_max_reproj_px=self.config.max_max_reprojection_error_px,
                max_translation_jump_mm=self.config.max_translation_jump_mm,
                max_rotation_jump_deg=self.config.max_rotation_jump_deg,
                rotation_gate_scale_per_lost_frame=self.config.rotation_gate_scale_per_lost_frame,
                rotation_gate_max_deg=self.config.rotation_gate_max_deg,
                use_pose_prior=self.config.use_pose_prior,
                refine_with_iterative=True,
                use_direct_prior_solver=self.config.pnp_direct_prior_enabled,
                direct_refine_method=self.config.pnp_direct_refine_method,
                direct_max_mean_reproj_px=(
                    self.config.pnp_direct_max_mean_reprojection_error_px
                ),
                direct_max_max_reproj_px=(
                    self.config.pnp_direct_max_max_reprojection_error_px
                ),
            ),
        )

    def _create_dot_detector(self):
        """Build the dot detector, including temporal and cell-cache settings."""
        cfg = hm.DotDetectorConfig()

        cfg.canonical_size = self.config.dot_canonical_size
        cfg.canonical_margin_px = self.config.dot_canonical_margin_px

        cfg.min_dot_contrast = self.config.dot_min_dot_contrast
        cfg.strong_dot_contrast = self.config.dot_strong_dot_contrast

        cfg.commit_threshold = self.config.dot_commit_threshold
        cfg.revoke_threshold = self.config.dot_revoke_threshold

        cfg.uncertainty_low = self.config.dot_uncertainty_low
        cfg.uncertainty_high = self.config.dot_uncertainty_high

        cfg.warmup_frames = self.config.dot_warmup_frames

        cfg.temporal_alpha = self.config.dot_temporal_alpha
        cfg.commit_frames = self.config.dot_commit_frames
        cfg.revoke_frames = self.config.dot_revoke_frames
        cfg.use_temporal_smoothing = self.config.dot_use_temporal_smoothing
        if hasattr(cfg, "use_cell_value_cache"):
            cfg.use_cell_value_cache = self.config.dot_use_cell_value_cache
        if hasattr(cfg, "cell_cache_max_age_frames"):
            cfg.cell_cache_max_age_frames = self.config.dot_cell_cache_max_age_frames
        if hasattr(cfg, "cell_cache_max_corner_motion_px"):
            cfg.cell_cache_max_corner_motion_px = self.config.dot_cell_cache_max_corner_motion_px

        return hm.DotDetector(cfg)

    def _create_patch_decoder(self):
        cfg = hm.PatchDecoderConfig()
        cfg.require_geometry_valid = self.config.decoder_require_geometry_valid
        cfg.accept_ambiguous = self.config.decoder_accept_ambiguous
        return hm.PatchDecoder(cfg)

    def _create_correspondence_builder(self):
        """Build the decoded-patch to geometry correspondence builder."""
        cfg = hm.CorrespondenceBuilderConfig()
        cfg.min_votes = self.config.corr_min_votes
        cfg.discard_conflicts = self.config.corr_discard_conflicts
        cfg.require_detection_stable = self.config.corr_require_detection_stable
        cfg.enable_dominant_rotation_filter = self.config.corr_enable_dominant_rotation_filter
        cfg.min_rotation_support = self.config.corr_min_rotation_support
        cfg.min_rotation_support_ratio = self.config.corr_min_rotation_support_ratio
        return hm.CorrespondenceBuilder(cfg)


class DecodePipelineMixin:
    """Run dot detection, patch decoding, correspondence building, and pose handoff."""

    def _decode_and_estimate_pose(self, frame: np.ndarray, detection) -> TrackerResult:
        """Decode visible patches and estimate pose, falling back when decode is weak."""
        # Reset the dot detector immediately after large drill rotations.
        # This must run before dot_detector.detect() so the reset affects the
        # same frame in which the rotation is detected.
        if (
            self.mode == TrackerMode.TRACKING
            and self._last_accepted_rvec is not None
            and self.pose_tracker.rvec is not None
        ):
            try:
                R_prev, _ = cv2.Rodrigues(
                    np.asarray(self._last_accepted_rvec, dtype=np.float64).reshape(3, 1)
                )
                R_curr, _ = cv2.Rodrigues(
                    np.asarray(self.pose_tracker.rvec, dtype=np.float64).reshape(3, 1)
                )
                dR = R_curr @ R_prev.T
                cos_a = float(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
                rot_delta_deg = float(np.degrees(np.arccos(cos_a)))
                if rot_delta_deg > 15.0:
                    self.dot_detector = self._create_dot_detector()
                    self._last_accepted_rvec = None
            except Exception:
                pass

        dots = self.dot_detector.detect(frame, detection)

        # Reset smoothing early when the point count drops far below the best
        # recent count. That gives the EMA state a chance to recover before a
        # complete decode failure.
        if (
            self.mode == TrackerMode.TRACKING
            and self.config.dot_early_reset_pts_ratio > 0.0
            and self._max_pts_seen >= self.config.dot_early_reset_min_pts
        ):
            # Estimate points from valid, non-ambiguous detector cells.
            current_pts = sum(
                1 for c in dots.cells
                if c.valid and not c.ambiguous
            )
            threshold = int(
                self._max_pts_seen * self.config.dot_early_reset_pts_ratio
            )
            if current_pts < threshold:
                if hasattr(self.dot_detector, "reset_smoothing"):
                    self.dot_detector.reset_smoothing()

        patches = self.patch_extractor.extract(
            dots,
            self.field.patchSize(),
        )

        decoded = self.patch_decoder.decode(
            patches,
            self.field,
        )

        decoded_valid = [
            p for p in decoded
            if p.valid and not p.ambiguous
        ]

        if not decoded_valid:
            decode_msg = self._decode_failure_message(dots, patches, decoded)
            self._note_decode_topology_failure(dots, patches)
            bootstrap = self._estimate_pose_from_uncoded_grid_bootstrap(
                detection,
                reason=decode_msg,
            )
            if bootstrap is not None:
                return bootstrap
            if self._last_uncoded_bootstrap_reason:
                decode_msg = (
                    f"{decode_msg}; uncoded_bootstrap="
                    f"{self._last_uncoded_bootstrap_reason}"
                )

            fallback = self._estimate_pose_from_persistent_correspondences(
                detection,
                reason=decode_msg,
            )
            if fallback is not None:
                return fallback

            held = self._hold_last_pose_result(
                detection,
                reason=decode_msg,
                correspondence_corners=[],
            )
            if held is not None:
                return held

            return TrackerResult(
                success=False,
                mode=self.mode,
                message=decode_msg + ".",
            )

        self._undecodeable_detection_frames = 0

        corr_result = self.correspondence_builder.build(
            detection,
            decoded_valid,
            self.geometry,
        )

        if not corr_result.valid():
            self._note_low_fresh_correspondence_failure(0)
            fallback = self._estimate_pose_from_persistent_correspondences(
                detection,
                reason="Correspondence build failed",
            )
            if fallback is not None:
                return fallback

            held = self._hold_last_pose_result(
                detection,
                reason="Correspondence build failed",
                correspondence_corners=[],
            )
            if held is not None:
                return held

            return TrackerResult(
                success=False,
                mode=self.mode,
                message="Correspondence build failed.",
            )

        track_points, tracker_corners = self._points_from_correspondences(
            corr_result.correspondences,
        )

        if len(track_points) < self.config.min_points:
            corr_msg = self._correspondence_failure_message(
                len(track_points),
                corr_result,
            )
            self._note_low_fresh_correspondence_failure(len(track_points))
            merged_points, merged_corners = self._merge_with_persistent_correspondences(
                detection,
                track_points,
                tracker_corners,
            )

            if len(merged_points) >= self.config.min_points:
                pose_result = self._estimate_and_package_pose(
                    merged_points,
                    merged_corners,
                    success_message=(
                        f"Pose estimated with merged fresh+persistent correspondences "
                        f"({len(track_points)} fresh, {len(merged_points)} total)."
                    ),
                    update_persistence=False,
                    pose_source=PoseSource.PERSISTENT,
                    detection=detection,
                )
                if pose_result.success:
                    return pose_result

            fallback = self._estimate_pose_from_persistent_correspondences(
                detection,
                reason=corr_msg,
            )
            if fallback is not None:
                return fallback

            held = self._hold_last_pose_result(
                detection,
                reason=corr_msg,
                correspondence_corners=tracker_corners,
            )
            if held is not None:
                return held

            return TrackerResult(
                success=False,
                mode=self.mode,
                message=corr_msg + ".",
                num_points=len(track_points),
                correspondence_corners=tracker_corners,
            )

        self._low_fresh_correspondence_frames = 0

        pose_result = self._estimate_and_package_pose(
            track_points,
            tracker_corners,
            success_message="Pose estimation successful.",
            update_persistence=True,
            pose_source=PoseSource.DECODE,
            detection=detection,
        )

        if pose_result.success:
            return pose_result

        fallback = self._estimate_pose_from_persistent_correspondences(
            detection,
            reason=pose_result.message,
        )
        if fallback is not None:
            return fallback

        return pose_result


class DecodeHelperMixin:
    """Provide diagnostics and topology bookkeeping for decode failures."""

    def _note_decode_topology_failure(self, dots, patches) -> None:
        """Track repeated non-decodeable detections and force local recovery if needed."""
        if len(patches) > 0:
            self._undecodeable_detection_frames = 0
            return

        dot_cells = list(getattr(dots, "cells", []))
        if not dot_cells:
            self._undecodeable_detection_frames = 0
            return

        min_span = max(1, int(self.config.checker_min_tracking_decode_cell_span))
        rows = [int(getattr(c, "row", 0)) for c in dot_cells]
        cols = [int(getattr(c, "col", 0)) for c in dot_cells]
        row_span = max(rows) - min(rows) + 1 if rows else 0
        col_span = max(cols) - min(cols) + 1 if cols else 0

        if row_span >= min_span and col_span >= min_span:
            self._undecodeable_detection_frames = 0
            return

        self._undecodeable_detection_frames += 1
        if (
            self._undecodeable_detection_frames
            > self.config.checker_max_undecodeable_tracking_frames
        ):
            self._force_local_recovery()

    @staticmethod
    def _correspondence_failure_message(num_points: int, corr_result) -> str:
        """Build a compact diagnostic string for correspondence-builder failures."""
        return (
            f"Too few correspondences: {num_points} "
            f"(patches_used={int(getattr(corr_result, 'decoded_patches_used', 0))}, "
            f"rot_rejected={int(getattr(corr_result, 'decoded_patches_rejected_by_rotation', 0))}, "
            f"assign_total={int(getattr(corr_result, 'assignments_total', 0))}, "
            f"assign_accepted={int(getattr(corr_result, 'assignments_accepted', 0))}, "
            f"conflicted={int(getattr(corr_result, 'assignments_conflicted', 0))}, "
            f"no_geom={int(getattr(corr_result, 'corners_without_geometry', 0))}, "
            f"single_boundary={int(getattr(corr_result, 'single_vote_boundary_corners_accepted', 0))}, "
            f"single_non_boundary_rej={int(getattr(corr_result, 'single_vote_non_boundary_corners_rejected', 0))}, "
            f"rot={int(getattr(corr_result, 'dominant_rotation_deg', -1))}/"
            f"{int(getattr(corr_result, 'dominant_rotation_count', 0))}/"
            f"{int(getattr(corr_result, 'rotation_vote_count', 0))})"
        )

    @staticmethod
    def _decode_failure_message(dots, patches, decoded) -> str:
        """Build a compact diagnostic string for dot, patch, and decoder failures."""
        dot_cells = list(getattr(dots, "cells", []))
        dot_cell_count = len(dot_cells)
        dot_valid_count = sum(
            1 for c in dot_cells
            if bool(getattr(c, "valid", False))
        )
        dot_ambiguous_count = sum(
            1 for c in dot_cells
            if bool(getattr(c, "ambiguous", False))
        )
        dot_cache_reused_count = sum(
            1 for c in dot_cells
            if bool(getattr(c, "cache_reused", False))
        )
        dot_rows = int(getattr(dots, "rows", 0))
        dot_cols = int(getattr(dots, "cols", 0))

        patch_count = len(patches)
        decoded_count = len(decoded)
        invalid_geometry = sum(
            1 for p in decoded
            if getattr(p, "local", None) is not None
            and getattr(p.local, "valid", False)
            and not getattr(p.local, "geometry_valid", False)
        )
        ambiguous = sum(
            1 for p in decoded
            if getattr(p, "ambiguous", False)
        )
        matched_but_rejected = sum(
            1 for p in decoded
            if int(getattr(p, "num_matches", 0)) > 0 and not getattr(p, "valid", False)
        )

        return (
            "No valid decoded patches "
            f"(cells={dot_cell_count}, valid_cells={dot_valid_count}, "
            f"ambig_cells={dot_ambiguous_count}, cached_cells={dot_cache_reused_count}, "
            f"grid={dot_rows}x{dot_cols}, patches={patch_count}, decoded={decoded_count}, "
            f"bad_geom={invalid_geometry}, ambiguous={ambiguous}, "
            f"matched_rejected={matched_but_rejected})"
        )


class PosePropagationMixin:
    """Project known geometry from the last pose to stabilize the next decode frame."""

    def _build_pose_propagated_detection(
        self,
        image_shape: Tuple[int, int],
    ):
        """Build a synthetic detection from last-pose projected geometry corners."""
        if self.config.decode_only_mode or not self.config.enable_pose_propagation:
            return None

        if self.frame_index <= self._pose_propagation_block_until_frame:
            return None

        rvec = self.pose_tracker.rvec
        tvec = self.pose_tracker.tvec

        if rvec is None or tvec is None:
            return None

        if (
            self._last_good_reproj_px < 0.0
            or self._last_good_reproj_px > self.config.pose_propagation_max_reproj_px
        ):
            return None

        rows = self.geometry.corner_rows()
        cols = self.geometry.corner_cols()
        border = self.config.pose_propagation_border_px
        h, w = image_shape[0], image_shape[1]

        # Collect all valid 3D geometry corners.
        obj_pts = []
        row_col_list = []

        for gr in range(rows):
            for gc in range(cols):
                if not self.geometry.has_corner(gr, gc):
                    continue
                pt = self.geometry.corner_point(gr, gc)
                obj_pts.append([pt.x, pt.y, pt.z])
                row_col_list.append((gr, gc))

        if len(obj_pts) < self.config.min_points:
            return None

        obj_pts_np = np.array(obj_pts, dtype=np.float64).reshape(-1, 3)

        projected, _ = cv2.projectPoints(
            obj_pts_np,
            rvec.reshape(3, 1),
            tvec.reshape(3, 1),
            self.K,
            self.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)

        # Build synthetic GridCorners for visible projections only.
        # global_row -> corner.j, global_col -> corner.i
        detection = hm.CheckerboardDetection()
        ij_to_uv: Dict[Tuple[int, int], Tuple[float, float]] = {}

        for idx, (gr, gc) in enumerate(row_col_list):
            u, v = float(projected[idx, 0]), float(projected[idx, 1])

            if u < border or v < border or u >= w - border or v >= h - border:
                continue

            corner = hm.GridCorner()
            corner.j = gr   # row = vertikal = j
            corner.i = gc   # col = horizontal = i
            corner.uv = hm.Point2f()
            corner.uv.x = u
            corner.uv.y = v
            corner.visibility_score = 1.0

            detection.corners.append(corner)
            ij_to_uv[(gc, gr)] = (u, v)  # key: (i,j)

        if len(detection.corners) < self.config.min_points:
            return None

        # Build synthetic cells from projected corners.
        # Cell (i,j) hat Corners: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
        for ci, cj in list(ij_to_uv.keys()):
            if (ci+1, cj) not in ij_to_uv:
                continue
            if (ci+1, cj+1) not in ij_to_uv:
                continue
            if (ci, cj+1) not in ij_to_uv:
                continue

            cell = hm.GridCell()
            cell.i = ci
            cell.j = cj

            p00 = ij_to_uv[(ci,   cj)]
            p10 = ij_to_uv[(ci+1, cj)]
            p11 = ij_to_uv[(ci+1, cj+1)]
            p01 = ij_to_uv[(ci,   cj+1)]

            def make_pt(xy):
                p = hm.Point2f()
                p.x = xy[0]
                p.y = xy[1]
                return p

            cell.corner_uv = [make_pt(p00), make_pt(p10), make_pt(p11), make_pt(p01)]
            cell.center_uv = make_pt((
                (p00[0]+p10[0]+p11[0]+p01[0]) * 0.25,
                (p00[1]+p10[1]+p11[1]+p01[1]) * 0.25,
            ))

            detection.cells.append(cell)

        if len(detection.cells) == 0:
            return None

        if not self._detection_has_decodeable_cell_span(detection):
            return None

        detection.tracking = True
        detection.stable = True

        return detection
