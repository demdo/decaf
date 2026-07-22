#pragma once

#include <string>

namespace hydramarker {

struct TrackerConfig {
    int min_points = 6;
    int min_inliers = 5;

    double max_mean_reprojection_error_px = 4.0;
    double max_max_reprojection_error_px = 15.0;

    int max_lost_frames = 8;
    // Hysteresis before escalating a tracking failure to the EXPENSIVE full
    // recovery: a single high-reprojection frame (e.g. the cylinder marker's
    // viewpoint-dependent residual spiking past the pose gate) should hold the
    // last pose and let the next frame retry the cheap tracking path, NOT nuke
    // the tracked corner set and re-detect on the full frame. Only escalate to
    // Recovering after this many consecutive failures (still bounded by
    // max_lost_frames for the full loss).
    int recovery_grace_frames = 3;

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

    // Measurement operator for LK-tracked corners in the checkerboard
    // detector ("subpix" = cv::cornerSubPix snap, "model_warp" =
    // forward-model template registration once implemented).
    std::string checker_tracked_refine_method = "subpix";

    // quadratic_form corner measurement (reference-free curve intersection;
    // active when checker_tracked_refine_method == "quadratic_form").
    double checker_qf_profile_half_px = 4.5;
    double checker_qf_min_contrast = 12.0;
    double checker_qf_max_profile_rms = 0.35;
    double checker_qf_junction_margin_frac = 0.18;
    int    checker_qf_min_row_points = 6;
    int    checker_qf_min_col_points = 3;
    double checker_qf_conic_gain = 0.95;
    double checker_qf_max_fit_rms_px = 0.6;
    double checker_qf_max_dev_px = 2.5;

    // Measurement-covariance inflation for frames where the IR MAP fusion
    // ATTEMPTED and REJECTED (references existed, all gates failed): the
    // surviving RGB-only pose rests on suspect corner evidence (glare
    // veil, adverse geometry), so the pose filter damps it instead of
    // following the drift at full confidence.  1.0 disables (default for
    // bit-identical acceptance replays; the live config enables it).
    double ir_reject_cov_inflate = 1.0;

    // IR corner measurement operator. "quadratic_form" = reference-free direct
    // measurement (the only measurement path); anything else passes the RGB
    // pose through unchanged (no IR fusion).
    std::string ir_corner_method = "quadratic_form";
    double ir_qf_max_dev_px = 4.0;

    // Pose-set stabilisation. On the cylinder marker the pose is nearly
    // blind along one direction (~1 mm z per 0.1 px), so two effects that
    // are harmless elsewhere dominate the z error: (a) predicted corners
    // feed the PREVIOUS pose back into PnP, (b) freshly appeared corners
    // entering the pose set cause weak-mode jumps. Both filters apply to
    // the POSE input only; detection/decoding/persistence keep every
    // corner. If filtering would leave fewer than the minimum usable
    // points, it relaxes automatically rather than losing the pose.
    bool pose_exclude_predicted_corners = true;
    int pose_min_observed_frames = 5;   // 0 disables the entry hysteresis

    // Pose warmup reporting: after (re)acquisition the pose wanders along
    // the weak mode while the corner set saturates (measured: z std 0.64 mm
    // static). pose_converged latches once the set has been quiet; it is
    // a STATUS for downstream consumers, poses are still produced.
    int pose_warmup_min_accepted_frames = 20;  // 0 disables (always converged)
    int pose_warmup_stable_window = 15;
    int pose_warmup_max_young_corners = 2;

    // Anisotropic pose Kalman filter (OUTPUT-only; the internal tracking
    // chain keeps the raw pose so no feedback loop forms). Constant-velocity
    // model; measurement covariance sigma^2*(J^T J)^-1 per frame, so
    // observable directions follow the measurement instantly while the weak
    // observability mode is smoothed. Mahalanobis-deweighted spikes.
    bool pose_kf_enabled = true;
    double pose_kf_sigma_px = 0.12;
    double pose_kf_q_translation_mm = 0.15;   // moving accel noise per frame
    double pose_kf_q_rotation_deg = 0.5;      // moving accel noise per frame
    double pose_kf_gate_mahalanobis = 30.0;
    double pose_kf_reset_rotation_deg = 10.0; // innovation beyond -> reset
    // Adaptive process noise (default). While the tool is still the process
    // noise collapses to the floor so the filter averages many frames (static
    // tip jitter 0.10 -> 0.034 mm); it opens up the instant motion is measured
    // so the pose follows without lag. Validated offline on raw pose logs.
    bool pose_kf_adaptive = true;
    double pose_kf_q_rotation_floor_deg = 0.002;
    double pose_kf_q_translation_floor_mm = 0.002;
    double pose_kf_adaptive_ema_alpha = 0.4;
    // Self-calibrating noise level (default): the filter estimates its quiet-
    // baseline frame-to-frame delta over a rolling window and derives both the
    // measurement-noise floor and the motion threshold from it, adapting to the
    // session/orientation (fixed floors cannot). The manual floors below are
    // the fallback (auto off) and the lower bound of the online estimate.
    bool pose_kf_auto_noise = true;
    int pose_kf_noise_window = 90;
    double pose_kf_motion_floor_rot_deg = 0.12;
    double pose_kf_motion_floor_trans_mm = 0.05;
    double pose_kf_meas_floor_rot_deg = 0.035;
    double pose_kf_meas_floor_trans_mm = 0.02;

    // ---- IR depth-only stereo fusion (opt-in, output-only) ----
    // Fuses the absolute depth / out-of-plane tilt from the global-shutter
    // IR stereo pair into the reported pose; RGB keeps rotation and lateral
    // position (a full IR pose replacement is WORSE at the tip: stereo
    // rotation noise times the tool lever). Multi-reference model_warp
    // measurement with self-enslavement guard, saturation gate, robust
    // ZNCC-weighted tilt fit; fallback keeps the RGB pose. DOUBLE-GATED:
    // needs this flag AND setIrCalibration() with the rig calibration AND
    // the IR frame pair per processFrame call; without any of these the
    // stage is inert and the tracker behaves exactly as before.
    bool ir_refine_enabled = false;
    int ir_min_pairs = 6;                 // stereo pairs to fuse at all
    double ir_min_ref_rot_deg = 6.0;      // self-enslavement guard
    double ir_max_ref_rot_deg = 180.0;    // optional upper selection window
    double ir_fallback_min_ref_rot_deg = 0.05;  // fixed-orientation fusion floor:
                                          // used only when no reference reaches
                                          // ir_min_ref_rot_deg (pure translation).
                                          // 0.5 was too high for a steady tool
                                          // (orientation drift < 0.5deg -> almost
                                          // no admissible ref -> IR barely fused,
                                          // z ~= RGB); 0.05 fires ~70-90% and
                                          // drops the translation tip-z to ~0.18.
    int ir_sat_threshold = 250;           // saturation gate (patch max)
    int ir_sat_half_px = 8;
    double ir_epipolar_max_dv_px = 2.5;   // |vL - vR| row consistency
    double ir_zncc_weight_floor = 0.05;
    double ir_dtz_clamp_mm = 2.5;         // robust-median depth shift clamp
    double ir_depth_scale = 1.0;          // stereo scale (ChArUco validation)
    // MAP fusion (tightly-coupled RGB reproj + IR stereo 3D). Sigmas calibrated
    // from residuals -> w3d = 1. The fit gate rejects an unreliable fusion (the
    // MAP 3D-residual RMS runs ~2.4x the pure-kabsch RMS; the trans-jump + pairs
    // gates catch divergence).
    double ir_sigma_px = 0.10;            // RGB reprojection noise (px), still
    // Rolling-shutter compensation: sigma_px_eff^2 = sigma_px^2 +
    // (coeff * v_px)^2 with v_px = median per-frame corner motion. The RGB
    // rolling shutter shears the marker coherently under motion (fake roll
    // proportional to velocity over the tool lever); the global-shutter IR is
    // immune, so the velocity-inflated sigma hands orientation authority to IR
    // exactly then. 0 disables (static sigma).
    double ir_sigma_px_motion_coeff = 0.20;
    // Rotation share of the image velocity (~px per deg/frame of pose
    // rotation), subtracted from vpx before the shear compensation: rotation-
    // driven velocity stresses the IR warp itself, so only the TRANSLATION
    // share may shift authority to IR (divot regressed otherwise).
    double ir_sigma_px_motion_rot_px_per_deg = 8.0;
    double ir_sigma_ir_px = 0.05;         // IR corner-localisation noise (px)
    // IR/reproj balance. 1 = the still-frame sigma calibration. During MOTION
    // the RGB corner errors are CORRELATED (blur/LK lag, the weak-mode wobble),
    // so the IID sigma_px overstates the RGB information exactly when the
    // lever-coupled orientation error strikes; 4 rebalances for that (fb5 tip-z
    // std 0.61->0.37, fb4 0.57->0.28 exc-bias +0.58->+0.17, divot replay z/x
    // better too - validated on all three 20.07 datasets).
    double ir_w3d = 4.0;
    double ir_fit_gate_rms_mm = 1.50;     // reject if MAP 3D-residual RMS exceeds
    double ir_fit_gate_max_trans_jump_mm = 3.0;  // reject if |t_map-t_rgb| exceeds
    // Weak-evidence honesty: the Huber-trimmed GN understates uncertainty when
    // the fit is poor (few pairs / bad matches survive). Inflate the covariance
    // handed to the temporal filter by the reduced chi-square vs this typical
    // fit RMS so the filter damps weak frames instead of following them.
    // 0 disables (raw MAP covariance).
    double ir_cov_inflate_ref_rms_mm = 0.10;
    double ir_mw_min_zncc = 0.35;         // soft 720p IR: lower ZNCC floor
    double ir_mw_max_shift_px = 6.0;
    // Reference enrollment: fill the library across the orientation sweep
    // during CONTINUOUS motion. Enroll on entering an uncovered tile as long
    // as the pose is converged and the per-frame motion is below the RELAXED
    // IR caps (global shutter -> a moving frame is still a sharp reference).
    double ir_enroll_max_rot_deg = 3.0;   // per-frame rotation cap (relaxed)
    double ir_enroll_max_trans_mm = 8.0;  // per-frame translation cap (relaxed)
    double ir_ref_tile_deg = 12.0;        // orientation tile per reference
    // Position tile: also enroll when the pose moved this far from every
    // reference (translation sweep at fixed orientation used to freeze the
    // library at one home reference -> warp stretch + saturation flip 80mm
    // away). <= 0 disables the position dimension = previous behaviour.
    double ir_ref_tile_trans_mm = 40.0;
    double ir_fallback_min_ref_trans_mm = 15.0;  // position-displaced same-
                                          // orientation refs are admissible
                                          // (not the identity-degenerate view)
    // Bootstrap-bias guard: a POSITION-only reference is enrolled only when
    // the fusion at that frame is trustworthy (the enrolled pose is baked into
    // the template; fb4 enrolled mid-sweep with a +2.1mm pose error and every
    // later measurement inherited it). Never blocks new-ORIENTATION refs.
    double ir_enroll_max_fit_rms_mm = 0.20;   // fusion residual must be under
    double ir_enroll_max_sat_frac = 0.35;     // IR saturation must be under
    int ir_max_references = 12;           // library capacity across the sweep

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
