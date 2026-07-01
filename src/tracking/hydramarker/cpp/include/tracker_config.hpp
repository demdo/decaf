#pragma once

#include <string>

namespace hydramarker {

struct TrackerConfig {
    int min_points = 6;
    int min_inliers = 5;

    double max_mean_reprojection_error_px = 4.0;
    double max_max_reprojection_error_px = 15.0;

    int max_lost_frames = 8;

    double max_translation_jump_mm = 40.0;
    double max_rotation_jump_deg = 45.0;
    double rotation_gate_scale_per_lost_frame = 8.0;
    double rotation_gate_max_deg = 90.0;

    int pnp_ransac_iterations = 500;
    double pnp_ransac_reprojection_px = 3.0;
    double pnp_ransac_confidence = 0.99;
    bool use_pose_prior = true;
    bool pnp_direct_prior_enabled = true;
    std::string pnp_direct_refine_method = "lm";
    double pnp_direct_max_mean_reprojection_error_px = 1.5;
    double pnp_direct_max_max_reprojection_error_px = 3.0;

    int checker_min_tracking_decode_cell_span = 3;
    int checker_refresh_interval_frames = 1;
    int checker_tracking_recovery_stable_interval_frames = 9;
    int checker_tracking_recovery_zero_gain_backoff_after = 3;
    int checker_tracking_recovery_zero_gain_backoff_max_factor = 16;
    bool checker_local_completion_skip_enabled = true;
    int checker_local_completion_probe_interval_frames = 6;
    int checker_local_completion_zero_gain_backoff_after = 3;
    int checker_local_completion_zero_gain_backoff_max_factor = 16;
    int checker_local_completion_stale_predicted_frames = 18;
    int checker_max_undecodeable_tracking_frames = 12;
    int checker_min_fresh_correspondences_for_stable_tracking = 8;
    int checker_max_low_fresh_correspondence_frames = 12;

    int dot_canonical_size = 80;
    double dot_canonical_margin_px = 4.0;
    double dot_min_dot_contrast = 8.0;
    double dot_strong_dot_contrast = 35.0;
    double dot_commit_threshold = 0.45;
    double dot_revoke_threshold = 0.20;
    double dot_uncertainty_low = 0.20;
    double dot_uncertainty_high = 0.45;
    int dot_warmup_frames = 1;
    double dot_temporal_alpha = 0.35;
    int dot_commit_frames = 2;
    int dot_revoke_frames = 3;
    bool dot_use_temporal_smoothing = false;
    bool dot_use_cell_value_cache = true;
    int dot_cell_cache_max_age_frames = 12;
    double dot_cell_cache_max_corner_motion_px = 35.0;

    bool decoder_require_geometry_valid = true;
    bool decoder_accept_ambiguous = false;

    int corr_min_votes = 2;
    bool corr_discard_conflicts = true;
    bool corr_require_detection_stable = false;
    bool corr_enable_dominant_rotation_filter = true;
    int corr_min_rotation_support = 2;
    double corr_min_rotation_support_ratio = 0.55;

    bool decode_only_mode = false;
    bool enable_fast_persistent_path = true;
    int fast_persistent_min_points = 10;
    double fast_persistent_refresh_mean_error_px = 1.5;
    bool fast_persistent_dense_refine_enabled = true;
    int fast_persistent_dense_min_points = 24;
    double fast_persistent_dense_match_max_px = 3.0;
    double fast_persistent_dense_min_second_best_margin_px = 2.0;
    double fast_persistent_dense_max_median_px = 1.2;
    double fast_persistent_dense_max_p90_px = 2.5;
    bool fast_persistent_dense_rescue_enabled = false;
    double fast_persistent_dense_rescue_min_green_ratio = 0.85;
    double fast_persistent_dense_rescue_min_seed_median_px = 1.5;
    double fast_persistent_dense_min_image_coverage = 0.35;
    double fast_persistent_dense_min_object_span_mm = 12.0;
    int fast_persistent_dense_min_distinct_rows = 2;
    int fast_persistent_dense_min_distinct_cols = 2;
    std::string fast_persistent_dense_pose_solver = "direct_prior";
    std::string fast_persistent_dense_robust_refine_method = "auto";
    bool fast_persistent_dense_robust_trim_enabled = true;
    double fast_persistent_dense_robust_trim_quantile = 0.85;
    double fast_persistent_dense_robust_min_keep_ratio = 0.75;
    double fast_persistent_dense_robust_max_mean_px = 1.2;
    double fast_persistent_dense_robust_max_max_px = 2.5;
    bool fast_persistent_dense_adaptive_refine_enabled = true;
    double fast_persistent_dense_adaptive_min_match_ratio = 0.85;
    double fast_persistent_dense_adaptive_motion_px = 8.0;
    double fast_persistent_dense_adaptive_max_seed_mean_px = 1.2;
    double fast_persistent_dense_adaptive_max_seed_max_px = 2.8;

    bool enable_temporal_correspondence_persistence = true;
    int persistence_max_frames = 8;
    int persistence_min_points = 6;
    int persistence_min_fresh_points_for_merge = 6;
    int persistence_min_points_after_decode_fail = 10;
    double persistence_refresh_mean_error_px = 1.5;
    double persistence_max_translation_jump_mm = 60.0;
    double persistence_max_rotation_jump_deg = 20.0;
    bool persistence_use_pose_projection = true;
    double persistence_projection_max_reproj_px = 9.0;
    bool persistence_projection_adaptive_match_enabled = true;
    double persistence_projection_adaptive_motion_start_px = 6.0;
    double persistence_projection_adaptive_motion_scale = 1.0;
    double persistence_projection_adaptive_max_reproj_px = 18.0;
    double persistence_projection_max_pose_error_px = 1.5;
    double persistence_match_min_second_best_margin_px = 3.0;
    double persistence_uv_match_dist_px = 25.0;

    bool enable_pose_propagation = true;
    double pose_propagation_max_reproj_px = 2.0;
    double pose_propagation_border_px = 8.0;

    int pose_hold_max_frames = 45;
    int pose_hold_min_detection_corners = 8;
    bool emergency_pose_hold_enabled = true;
    int emergency_pose_hold_max_frames = -1;

    int fallback_pose_min_detection_matches = 8;
    double fallback_pose_max_median_corner_error_px = 9.0;
    double fallback_pose_max_p90_corner_error_px = 18.0;
    double fallback_pose_max_mean_reprojection_error_px = 1.8;
    double fallback_pose_max_max_reprojection_error_px = 4.0;

    double visual_corner_max_reprojection_error_px = 3.0;
    int visual_corner_min_count = 6;
    int decode_update_min_visual_corners = 12;
    int decode_update_min_distinct_rows = 3;
    int decode_update_min_distinct_cols = 3;
};

} // namespace hydramarker
