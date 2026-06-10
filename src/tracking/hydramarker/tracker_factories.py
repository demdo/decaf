from __future__ import annotations

from typing import Optional

import numpy as np

from tracking.hydramarker.backend import cpp_impl as hm
from tracking.hydramarker.map_pose_tracker import MapPoseTracker, MapPoseTrackerConfig


class TrackerFactoryMixin:
    def _create_checkerboard_detector(self):
        cfg = hm.CheckerboardDetectorConfig()
        cfg.recovery_correction_weight = 0.5
        cfg.recovery_correction_max_dist_rel = 0.6
        if hasattr(cfg, min_tracking_decode_cell_span):
            cfg.min_tracking_decode_cell_span = (
                self.config.checker_min_tracking_decode_cell_span
            )
        if hasattr(cfg, max_undecodeable_tracking_frames):
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
        cfg = hm.CorrespondenceBuilderConfig()
        cfg.min_votes = self.config.corr_min_votes
        cfg.discard_conflicts = self.config.corr_discard_conflicts
        cfg.require_detection_stable = self.config.corr_require_detection_stable
        cfg.enable_dominant_rotation_filter = self.config.corr_enable_dominant_rotation_filter
        cfg.min_rotation_support = self.config.corr_min_rotation_support
        cfg.min_rotation_support_ratio = self.config.corr_min_rotation_support_ratio
        return hm.CorrespondenceBuilder(cfg)
