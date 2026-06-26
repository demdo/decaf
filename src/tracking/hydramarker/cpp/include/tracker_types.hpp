#pragma once

#include <array>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "tracker_persistence.hpp"

namespace hydramarker {

enum class TrackerMode {
    Lost,
    Detecting,
    Tracking,
    Recovering,
};

enum class PoseSource {
    None,
    Decode,
    Persistent,
    FastPersistent,
    UncodedGrid,
    Hold,
};

struct FrameDetectedCorner {
    int local_row = -1;
    int local_col = -1;
    std::array<double, 2> uv = {0.0, 0.0};
};

struct TrackerFrameResult {
    bool success = false;
    TrackerMode mode = TrackerMode::Lost;
    std::string message;

    bool detection_valid = false;
    bool detection_tracking = false;
    bool detection_stable = false;
    int detection_corner_count = 0;
    int detection_cell_count = 0;
    std::vector<FrameDetectedCorner> detection_corners;

    int frame_index = 0;
    int lost_frames = 0;

    bool pose_tracker_has_pose = false;
    std::vector<double> pose_tracker_rvec;
    std::vector<double> pose_tracker_tvec;
    std::vector<double> pose_tracker_T_marker_camera;

    bool current_pose_accepted = false;
    bool has_accepted_pose = false;
    int accepted_pose_frame = -1;
    int accepted_visual_corner_count = 0;
    int max_pts_seen = 0;
    double last_good_reproj_px = -1.0;
    std::vector<double> accepted_rvec;
    std::vector<double> accepted_tvec;
    std::vector<double> accepted_T_marker_camera;

    PoseSource pose_source = PoseSource::None;
    std::vector<double> rvec;
    std::vector<double> tvec;
    std::vector<double> T_marker_camera;
    int num_points = 0;
    int num_inliers = 0;
    double mean_reprojection_error_px = -1.0;
    double max_reprojection_error_px = -1.0;
    double confidence = 0.0;
    std::string pnp_method;
    int visual_corner_count = 0;
    std::vector<TrackerCorner> corners;
    std::vector<TrackerCorner> correspondence_corners;
    int persistent_count = 0;

    bool depth_filter_available = false;
    bool depth_filter_applied = false;
    double depth_filter_delta_z_mm = 0.0;
    double depth_filter_raw_z_mm = std::numeric_limits<double>::quiet_NaN();
    double depth_filter_z_mm = std::numeric_limits<double>::quiet_NaN();
    double depth_filter_reproj_excess_px =
        std::numeric_limits<double>::quiet_NaN();
    double depth_filter_guard_alpha = 1.0;
    double depth_filter_innovation_z_mm = 0.0;
    double depth_filter_innovation_mean_z_mm = 0.0;
    double depth_filter_innovation_cusum_pos_mm = 0.0;
    double depth_filter_innovation_cusum_neg_mm = 0.0;
    bool depth_filter_innovation_bias_detected = false;
    int depth_filter_innovation_bias_direction = 0;
    bool depth_filter_innovation_bias_limited = false;
    double depth_filter_object_z_span_mm =
        std::numeric_limits<double>::quiet_NaN();
    bool depth_filter_negative_delta_guard_limited = false;

    bool pose_plateau_prior_triggered = false;
    bool pose_plateau_prior_attempted = false;
    bool pose_plateau_prior_applied = false;
    std::string pose_plateau_prior_method;
    std::string pose_plateau_prior_reason;
    double pose_plateau_prior_delta_z_mm =
        std::numeric_limits<double>::quiet_NaN();
    double pose_plateau_prior_reproj_excess_px =
        std::numeric_limits<double>::quiet_NaN();
    double pose_plateau_prior_max_reproj_excess_px =
        std::numeric_limits<double>::quiet_NaN();
    int pose_plateau_prior_iterations = 0;

    bool fast_attempted = false;
    bool fast_success = false;
    bool fast_route_decode = false;
    int fast_matches = 0;
    std::string fast_reason;
    bool fast_dense_attempted = false;
    bool fast_dense_success = false;
    int fast_dense_matches = 0;
    std::string fast_dense_reason;

    int dot_cell_count = 0;
    int dot_valid_cell_count = 0;
    int patch_count = 0;
    int decoded_patch_count = 0;
    int decoded_valid_patch_count = 0;
    int correspondence_count = 0;

    std::unordered_map<std::string, double> timings_ms;
};

} // namespace hydramarker
