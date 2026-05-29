#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

struct GridCorner {
    int i = -1;
    int j = -1;
    cv::Point2f uv;
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

    float min_tracking_corner_ratio = 0.45f;
    float max_tracking_homography_error_px = 8.0f;

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

    // How many consecutive frames a corner may be missed by LK before it is
    // removed from the persistent set.
    // Kept small (2-3) so that genuinely lost corners are evicted quickly.
    int max_missed_frames = 2;

    // Consecutive frames with fewer than (min_corners * 2) corners before a
    // forced reset.  Lower = faster escape from stuck states.
    int max_low_corner_frames = 6;

    int lk_win_size = 21;
    int lk_max_level = 3;
    int lk_max_iters = 30;
    double lk_epsilon = 0.01;
    float max_lk_error = 35.0f;

    // Forward-backward LK consistency threshold.
    float max_lk_forward_backward_error_px = 2.0f;

    float stable_motion_threshold_px = 2.0f;

    int det_width = 0;
    int max_recovery_corners = 150;

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
};

} // namespace hydramarker