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
    int min_corners = 8;
    int min_cells = 3;

    int min_tracking_corners = 6;
    int min_tracking_cells = 2;

    float min_tracking_corner_ratio = 0.45f;
    float max_tracking_homography_error_px = 8.0f;

    int refresh_interval_frames = 10;

    int lk_win_size = 21;
    int lk_max_level = 3;
    int lk_max_iters = 30;
    double lk_epsilon = 0.01;
    float max_lk_error = 35.0f;

    // Forward-backward LK consistency threshold.
    // Lower values are stricter. 1.5-2.5 px is a good first range.
    float max_lk_forward_backward_error_px = 2.0f;

    float stable_motion_threshold_px = 2.0f;

    int det_width = 960;
    int max_recovery_corners = 400;

    float merge_radius_px = 5.0f;
    float duplicate_corner_dist_px = 4.0f;

    float min_neighbor_dist_rel = 0.55f;
    float max_neighbor_dist_rel = 1.55f;

    float max_lattice_residual_rel = 0.35f;
    float outlier_residual_rel = 0.45f;

    int max_axis_seed_points = 20;

    float checker_corner_half_px = 12.0f;

    // Hybrid Samu/ReadMarker-style recovery.
    // Samus pipeline uses pre_filter -> pt_refine -> pt_struct,
    // where pre_filter is gradient-junction based and pt_refine validates
    // saddle/checkerboard-like local structure.
    bool use_saddle_recovery = true;
    int saddle_radius = 5;
    int saddle_iterations = 2;
    float saddle_sigma = 3.0f;
    float saddle_response_threshold = 0.1f;
    float saddle_max_angle_bias_deg = 20.0f;
    float saddle_correlation_drop = 0.2f;
};

} // namespace hydramarker