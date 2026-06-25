"""Central configuration surface for the HydraMarker frame tracker."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackerConfig:
    """Runtime thresholds and feature toggles shared by all tracker stages."""

    min_points: int = 6
    min_inliers: int = 5

    max_mean_reprojection_error_px: float = 4.0
    max_max_reprojection_error_px: float = 15.0

    max_lost_frames: int = 8

    max_translation_jump_mm: float = 40.0
    max_rotation_jump_deg: float = 45.0
    # Adaptiver Motion Gate: Threshold waechst um diesen Wert pro verlorenem Frame.
    # Beispiel: 8.0 -> nach 5 Frames: 45 + 40 = 85 deg
    rotation_gate_scale_per_lost_frame: float = 8.0
    # Absolutes Maximum fuer den skalierten Rotation-Threshold.
    rotation_gate_max_deg: float = 90.0

    pnp_ransac_iterations: int = 500
    pnp_ransac_reprojection_px: float = 3.0
    pnp_ransac_confidence: float = 0.99
    use_pose_prior: bool = True
    pnp_direct_prior_enabled: bool = True
    pnp_direct_refine_method: str = "lm"
    pnp_direct_max_mean_reprojection_error_px: float = 1.5
    pnp_direct_max_max_reprojection_error_px: float = 3.0

    # Runtime camera-Z stabilizer. This is the live, zero-latency version of
    # the depth Kalman/guard replay experiments: x/y and rotation stay with the
    # normal frame solver; only tvec[2] is filtered.
    pose_depth_filter_enabled: bool = True
    pose_depth_filter_observation_std_mm: float = 16.0
    pose_depth_filter_process_std_mm: float = 0.05
    pose_depth_filter_initial_velocity_std_mm: float = 0.1
    pose_depth_filter_reprojection_guard_px: float = 1.0
    pose_depth_filter_min_points: int = 6
    pose_depth_filter_innovation_guard_enabled: bool = True
    pose_depth_filter_innovation_window: int = 10
    pose_depth_filter_innovation_bias_threshold_mm: float = 0.75
    pose_depth_filter_innovation_min_same_sign: int = 8
    pose_depth_filter_innovation_cusum_slack_mm: float = 0.2
    pose_depth_filter_innovation_cusum_threshold_mm: float = 8.0
    pose_depth_filter_negative_delta_guard_enabled: bool = True
    pose_depth_filter_negative_delta_guard_min_z_span_mm: float = 14.835
    pose_depth_filter_negative_delta_guard_max_negative_delta_mm: float = 0.0
    pose_depth_filter_negative_delta_guard_hold_previous_z: bool = False
    pose_depth_filter_negative_delta_guard_hold_requires_innovation_bias: bool = True
    pose_depth_filter_negative_delta_guard_hold_min_negative_delta_mm: float = 0.4
    pose_depth_filter_negative_delta_guard_max_hold_correction_mm: float = 0.75
    pose_depth_filter_negative_delta_guard_velocity_damping: float = 0.25

    # Triggered second-stage pose prior for reprojection-consistent negative-Z
    # plateaus. The normal fast PnP result remains the default; this stage only
    # runs when the depth filter observes a geometry-gated negative-Z pull.
    pose_plateau_prior_enabled: bool = True
    pose_plateau_prior_trigger_negative_delta_mm: float = 0.0
    pose_plateau_prior_min_object_z_span_mm: float = 14.835
    pose_plateau_prior_min_points: int = 6
    pose_plateau_prior_static_max_excess_px: float = 0.18
    pose_plateau_prior_candidate_max_excess_px: float = 0.25
    pose_plateau_prior_candidate_max_max_excess_px: float = 1.00
    pose_plateau_prior_min_positive_z_correction_mm: float = 0.0
    pose_plateau_prior_max_positive_z_correction_mm: float = 0.75
    pose_plateau_prior_robust_c_px: float = 0.20
    pose_plateau_prior_max_iterations: int = 6
    pose_plateau_prior_max_step_translation_mm: float = 5.0
    pose_plateau_prior_max_step_rotation_deg: float = 5.0
    pose_plateau_prior_lm_damping: float = 1.0e-5

    # Early smoothing reset when the current point count falls below this
    # fraction of the best recent count. This catches gradual LK drift before
    # a complete decode failure.
    # 0.0 = disabled, 0.4 = reset at 40 percent of the maximum count.
    dot_early_reset_pts_ratio: float = 0.4
    # Minimum point count before the ratio gate is evaluated.
    dot_early_reset_min_pts: int = 6

    dot_canonical_size: int = 80
    dot_canonical_margin_px: float = 4.0
    dot_min_dot_contrast: float = 8.0
    dot_strong_dot_contrast: float = 35.0
    dot_commit_threshold: float = 0.45
    dot_revoke_threshold: float = 0.20
    dot_uncertainty_low: float = 0.20
    dot_uncertainty_high: float = 0.45
    dot_warmup_frames: int = 1
    dot_temporal_alpha: float = 0.35
    dot_commit_frames: int = 2
    dot_revoke_frames: int = 3

    # Drill/cylinder default: stateless dot decisions. Temporal smoothing at
    # cell level is unsafe after fast rotations because local cells can refer
    # to a different physical surface region.
    dot_use_temporal_smoothing: bool = False
    dot_use_cell_value_cache: bool = True
    dot_cell_cache_max_age_frames: int = 12
    dot_cell_cache_max_corner_motion_px: float = 35.0

    checker_min_tracking_decode_cell_span: int = 3
    checker_refresh_interval_frames: int = 1
    checker_tracking_recovery_stable_interval_frames: int = 9
    checker_local_completion_skip_enabled: bool = True
    checker_local_completion_probe_interval_frames: int = 6
    checker_max_undecodeable_tracking_frames: int = 12
    checker_min_fresh_correspondences_for_stable_tracking: int = 8
    checker_max_low_fresh_correspondence_frames: int = 12

    decoder_require_geometry_valid: bool = True
    decoder_accept_ambiguous: bool = False

    corr_min_votes: int = 2
    corr_discard_conflicts: bool = True
    corr_require_detection_stable: bool = False

    corr_enable_dominant_rotation_filter: bool = True
    corr_min_rotation_support: int = 2
    corr_min_rotation_support_ratio: float = 0.55

    # SfM/observation recording mode. Default is off so live tracking keeps
    # using fast/persistent recovery. When enabled, pose output must come from
    # fresh Dot/Patch/Decode correspondences.
    decode_only_mode: bool = False

    enable_fast_persistent_path: bool = True
    # Use the C++ implementation for cached-identity to detection-corner
    # matching. Python remains the fallback path for development and parity
    # checks.
    cpp_persistent_matcher_enabled: bool = True
    # Solve the fast persistent seed pose in the same C++ call as matching.
    # Python still owns the gates, logging, dense refine decision, and final
    # result packaging.
    cpp_fast_persistent_seed_enabled: bool = True
    cpp_dense_projection_matcher_enabled: bool = True
    cpp_visual_corner_filter_enabled: bool = True
    cpp_dense_robust_solver_enabled: bool = True
    fast_persistent_min_points: int = 10
    fast_persistent_refresh_mean_error_px: float = 1.5
    fast_persistent_dense_refine_enabled: bool = True
    fast_persistent_dense_min_points: int = 24
    fast_persistent_dense_match_max_px: float = 3.0
    fast_persistent_dense_min_second_best_margin_px: float = 2.0
    fast_persistent_dense_max_median_px: float = 1.2
    fast_persistent_dense_max_p90_px: float = 2.5
    # Only suspicious fast-path seeds pay for robust dense re-solving. A frame
    # is suspicious when the seed projection disagrees with dense visible
    # corners, or when many detected corners were not carried by the sparse
    # persistent pose.
    fast_persistent_dense_rescue_enabled: bool = False
    fast_persistent_dense_rescue_min_green_ratio: float = 0.85
    fast_persistent_dense_rescue_min_seed_median_px: float = 1.5
    fast_persistent_dense_min_image_coverage: float = 0.35
    fast_persistent_dense_min_object_span_mm: float = 12.0
    fast_persistent_dense_min_distinct_rows: int = 2
    fast_persistent_dense_min_distinct_cols: int = 2
    fast_persistent_dense_pose_solver: str = "direct_prior"
    fast_persistent_dense_robust_refine_method: str = "auto"
    fast_persistent_dense_robust_trim_enabled: bool = True
    fast_persistent_dense_robust_trim_quantile: float = 0.85
    fast_persistent_dense_robust_min_keep_ratio: float = 0.75
    fast_persistent_dense_robust_max_mean_px: float = 1.2
    fast_persistent_dense_robust_max_max_px: float = 2.5
    # Avoid paying dense validation on clean fast-path seeds. Dense is still
    # used when the sparse persistent match looks risky.
    fast_persistent_dense_adaptive_refine_enabled: bool = True
    fast_persistent_dense_adaptive_min_match_ratio: float = 0.85
    fast_persistent_dense_adaptive_motion_px: float = 8.0
    fast_persistent_dense_adaptive_max_seed_mean_px: float = 1.2
    fast_persistent_dense_adaptive_max_seed_max_px: float = 2.8

    enable_temporal_correspondence_persistence: bool = True
    persistence_max_frames: int = 8
    # Lowered from 8: on a curved/partial marker we often get fewer inliers
    # per frame. With 8, persistence was never updated on frames with <8
    # inliers, causing the tracker to freeze on stale correspondences.
    persistence_min_points: int = 6
    persistence_min_fresh_points_for_merge: int = 6
    persistence_min_points_after_decode_fail: int = 10
    persistence_max_translation_jump_mm: float = 60.0
    persistence_max_rotation_jump_deg: float = 20.0

    # When a persistent-fallback pose is good (below this threshold), refresh
    # the persistent correspondences so the tracker doesn't run out of time.
    persistence_refresh_mean_error_px: float = 1.5

    # Match cached global IDs through last-pose reprojection instead of stale
    # image-space UVs. This keeps the fallback conservative on the cylinder:
    # IDs are only reused when the current checkerboard corner is still close
    # to where the last accepted pose predicts that exact 3D corner.
    persistence_use_pose_projection: bool = True
    persistence_projection_max_reproj_px: float = 9.0
    persistence_projection_adaptive_match_enabled: bool = True
    persistence_projection_adaptive_motion_start_px: float = 6.0
    persistence_projection_adaptive_motion_scale: float = 1.0
    persistence_projection_adaptive_max_reproj_px: float = 18.0
    persistence_projection_max_pose_error_px: float = 1.5
    persistence_match_min_second_best_margin_px: float = 3.0

    # Maximum UV distance (pixels) to match a persistent corner to a current
    # detection corner.  Used instead of exact local (i,j) key matching so
    # that persistent state survives CheckerboardDetector re-indexing events
    # (lattice drift, LK reset) which change the local coordinate system
    # without moving the physical corners in the image.
    persistence_uv_match_dist_px: float = 25.0

    # Pose propagation projects known marker corners from the last good pose
    # into the next frame. It replaces the LK-driven checkerboard detection
    # only while the last accepted reprojection error is below the threshold.
    enable_pose_propagation: bool = True
    pose_propagation_max_reproj_px: float = 2.0
    # Minimum image-border distance for projected corners in pixels.
    pose_propagation_border_px: float = 8.0
    pose_hold_max_frames: int = 45
    pose_hold_min_detection_corners: int = 8
    emergency_pose_hold_enabled: bool = True
    # -1 means: after the first valid pose, keep publishing the last pose
    # indefinitely. This is intentionally separate from visual corner output.
    emergency_pose_hold_max_frames: int = -1
    fallback_pose_min_detection_matches: int = 8
    fallback_pose_max_median_corner_error_px: float = 9.0
    fallback_pose_max_p90_corner_error_px: float = 18.0
    fallback_pose_max_mean_reprojection_error_px: float = 1.8
    fallback_pose_max_max_reprojection_error_px: float = 4.0
    visual_corner_max_reprojection_error_px: float = 3.0
    visual_corner_min_count: int = 6
    # A freshly decoded pose is allowed to refresh pose history/persistence only
    # when enough currently visible, pose-consistent marker corners remain.
    decode_update_min_visual_corners: int = 12
    decode_update_min_distinct_rows: int = 3
    decode_update_min_distinct_cols: int = 3
    enable_uncoded_grid_bootstrap: bool = True
    uncoded_bootstrap_min_corners: int = 8
    uncoded_bootstrap_max_mean_reprojection_error_px: float = 1.2
    uncoded_bootstrap_max_max_reprojection_error_px: float = 3.0
    uncoded_bootstrap_min_second_best_margin_px: float = 1.0


__all__ = ["TrackerConfig"]
