#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

struct GridCorner {
    int i = -1;
    int j = -1;
    cv::Point2f uv;

    // Photometric visibility score in [0,1], updated each tracking frame.
    // 1.0 = strong checkerboard contrast visible (front-facing corner).
    // 0.0 = no checkerboard structure (back-side / occluded corner).
    // Only meaningful during tracking; set to 1.0 during recovery detection.
    float visibility_score = 1.0f;

    // Internal helper corners may be used while testing local cell geometry,
    // but must not leak out as final visible checkerboard corners.
    bool synthetic = false;

    // Number of successful observations in the persistent tracker.  Recovery
    // detections leave this at 0; tracked output uses it for stable edge
    // hysteresis.
    int observed_frames = 0;

    // True when the point was carried by the motion model for this frame
    // instead of directly confirmed by LK/recovery.
    bool predicted = false;
};

struct GridCell {
    int i = -1;
    int j = -1;

    // Order must stay consistent:
    // p00 -> p10 -> p11 -> p01
    std::array<int, 4> corner_indices = {-1, -1, -1, -1};
    std::array<cv::Point2f, 4> corner_uv;

    cv::Point2f center_uv;
};

struct CheckerboardDetection {
    std::vector<GridCorner> corners;
    std::vector<GridCell> cells;

    int cols = 0;
    int rows = 0;

    bool tracking = false;
    bool stable = false;

    bool valid() const {
        return !corners.empty() && !cells.empty();
    }
};

struct CheckerboardDetectorConfig {
    int min_corners = 6;
    int min_cells = 2;

    int min_tracking_corners = 6;
    int min_tracking_cells = 2;

    // A k x k HydraMarker code patch needs at least k contiguous cell rows and
    // k contiguous cell columns.  LK tracking can otherwise get stuck in a
    // visually plausible but undecodable narrow strip (for example 4x2 cells).
    int min_tracking_decode_cell_span = 3;
    int max_undecodeable_tracking_frames = 12;

    float min_tracking_corner_ratio = 0.45f;
    float max_tracking_homography_error_px = 12.0f;

    // How often (in frames) a background recovery is run to pick up new
    // corners that appeared since the last full detection.
    // 1 = every frame.
    int refresh_interval_frames = 1;

    // Fraction of previously tracked corners that must be lost before an
    // immediate mid-interval recovery refresh is triggered.
    float refresh_corner_loss_ratio = 0.75f;

    // If the tracked detection is geometrically degraded for this many
    // consecutive frames without recovery finding anything better, force reset.
    int max_degraded_frames_before_reset = 3;

    // Recovery gain threshold: adopt fresh recovery when it finds this many
    // more corners than the current tracker.
    int refresh_gain_threshold = 5;

    // Recovery position correction weight (Fix B+C).
    //
    // When a recovery corner matches an actively tracked persistent corner by
    // grid ID, the tracked position is blended toward the recovery position:
    //   new_pos = (1 - w) * lk_pos + w * recovery_pos
    //
    // Recovery runs a full saddle-point detection + cornerSubPix on the
    // current frame and is therefore more accurate than accumulated LK drift.
    // A weight of 0.5 corrects drift quickly without causing instability.
    // Set to 0.0 to disable position correction (inject-only behaviour).
    float recovery_correction_weight = 0.5f;

    // Maximum distance (as fraction of spacing) between a tracked corner and
    // a recovery corner for position correction to be applied.
    // Beyond this distance the recovery corner is likely a different physical
    // corner and correction is skipped.
    float recovery_correction_max_dist_rel = 0.4f;

    // How many consecutive frames a corner may be missed by LK before it is
    // removed from the persistent set.
    // Kept small (2-3) so that genuinely lost corners are evicted quickly.
    int max_missed_frames = 3;

    // Consecutive frames with fewer than (min_corners * 2) corners before a
    // forced reset.  Lower = faster escape from stuck states.
    int max_low_corner_frames = 12;

    int lk_win_size = 31;
    int lk_max_level = 4;
    int lk_max_iters = 30;
    double lk_epsilon = 0.01;
    float max_lk_error = 35.0f;

    // Forward-backward LK consistency threshold.
    float max_lk_forward_backward_error_px = 3.5f;

    float stable_motion_threshold_px = 2.0f;

    int det_width = 0;
    int max_recovery_corners = 150;

    // During active tracking, recovery can search only a generous full-res ROI
    // around the currently tracked board. This keeps per-frame correction fast
    // without losing precision through image downscaling. Full-frame recovery is
    // still used for startup, tracking loss, and as fallback when ROI recovery
    // cannot produce a valid board.
    bool use_tracking_roi_recovery = true;
    float tracking_recovery_roi_margin_cells = 3.0f;
    int tracking_recovery_roi_min_margin_px = 80;
    float tracking_recovery_roi_max_area_ratio = 0.85f;
    // 0 disables periodic full-frame retry while active tracking is still
    // usable. Full-frame recovery still runs at startup or after tracking loss.
    int tracking_recovery_align_fail_full_retry_frames = 0;
    float tracking_recovery_align_fail_roi_margin_multiplier = 2.0f;
    // >1 enables a larger second ROI attempt after a normal ROI failure.
    float tracking_recovery_roi_fail_retry_margin_multiplier = 1.0f;
    // 0 disables periodic full-frame fallback while active tracking is still
    // usable. Full-frame recovery is too expensive for the normal partial-board
    // path and is mainly useful as an explicit diagnostic escape hatch.
    int tracking_recovery_roi_fail_full_retry_frames = 0;
    // During stable active tracking, keep detecting/refining ROI candidates
    // every refresh, but only rebuild the full ROI lattice/grid every N frames.
    // Candidate-only refreshes are enough for same-frame saddle snapping and
    // local completion; a value of 1 keeps the old every-frame full build.
    int tracking_recovery_full_build_interval_frames = 4;

    float merge_radius_px = 5.0f;
    float duplicate_corner_dist_px = 4.0f;

    float min_neighbor_dist_rel = 0.55f;
    float max_neighbor_dist_rel = 1.55f;

    float max_lattice_residual_rel = 0.35f;
    float outlier_residual_rel = 0.45f;

    int max_axis_seed_points = 20;

    float checker_corner_half_px = 12.0f;

    // Saddle-point recovery parameters.
    bool use_saddle_recovery = true;
    int saddle_radius = 5;
    int saddle_iterations = 2;
    float saddle_sigma = 3.0f;
    float saddle_response_threshold = 0.06f;
    float saddle_max_angle_bias_deg = 20.0f;
    float saddle_correlation_drop = 0.2f;

    // Sub-pixel refinement via cv::cornerSubPix, applied after the
    // gradient-intersection step and before saddle scoring.
    // -1 = auto (max(3, saddle_radius-1)), 0 = disabled, >0 = explicit px.
    int    saddle_subpix_win_size  = -1;
    int    saddle_subpix_max_iters = 20;
    double saddle_subpix_epsilon   = 0.05;

    // Quadrant intensity symmetry filter — used ONLY during recovery
    // (detectRecovery) to distinguish true checkerboard corners from
    // dot centres and cell interiors in the initial detection.
    //
    // NOT applied during tracking: LK + forward-backward + spacing filter
    // already provide strong geometric consistency.  Applying the quadrant
    // test during tracking causes corners to flicker out under illumination
    // changes, blur, and partial occlusion — exactly the conditions where
    // tracking stability matters most.
    int   quadrant_half_r = 3;
    float quadrant_min_contrast = 12.0f;       // in [0,255]; internally scaled relative to local range
    float quadrant_max_diagonal_diff = 60.0f;  // in [0,255]; relaxed — relative scaling makes it robust

    // LK spacing consistency filter.
    float tracking_spacing_min_rel = 0.50f;
    float tracking_spacing_max_rel = 1.55f;

    // Photometric visibility eviction.
    //
    // After each LK update, every actively tracked corner (missed_frames==0)
    // is scored by sampling the local checkerboard contrast along the grid
    // axes derived from its persistent neighbours (Option B: neighbour-derived
    // axes, not axis-aligned).  This makes the test rotation- and
    // perspective-robust for any marker size.
    //
    // If the score drops below visibility_evict_threshold the corner is
    // immediately evicted regardless of missed_frames, so back-side corners
    // disappear in 1-2 frames instead of waiting for max_missed_frames.
    //
    // visibility_sample_rel  — half-offset of each quadrant centre as a
    //                          fraction of the local grid spacing.  0.35 means
    //                          the sample sits 35% of one cell width from the
    //                          corner along each grid axis.
    // visibility_box_rel     — half-side of the averaging box as a fraction
    //                          of spacing.  0.15 ≈ 15% of cell width.
    // visibility_evict_threshold — corners with score below this are evicted.
    //                          0.10 = "essentially no checkerboard contrast".
    // visibility_min_spacing — skip the test when spacing < this (px) to
    //                          avoid false evictions on very small markers.
    float visibility_sample_rel      = 0.35f;
    float visibility_box_rel         = 0.15f;
    float visibility_evict_threshold = 0.18f;
    float visibility_min_spacing     = 8.0f;

    // EMA smoothing factor for visibility score.
    // smoothed = alpha * raw + (1 - alpha) * prev_smoothed
    // 0.4 = ~3-frame effective window; prevents single-frame dips from
    // triggering eviction while still reacting to genuine fade-out.
    float visibility_smoothing_alpha = 0.40f;
};

} // namespace hydramarker
