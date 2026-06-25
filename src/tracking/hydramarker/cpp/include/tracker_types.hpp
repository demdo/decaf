#pragma once

#include <string>
#include <unordered_map>
#include <vector>

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

struct TrackerFrameResult {
    bool success = false;
    TrackerMode mode = TrackerMode::Lost;
    std::string message;

    bool detection_valid = false;
    bool detection_tracking = false;
    bool detection_stable = false;
    int detection_corner_count = 0;
    int detection_cell_count = 0;

    int frame_index = 0;
    int lost_frames = 0;

    PoseSource pose_source = PoseSource::None;
    std::vector<double> rvec;
    std::vector<double> tvec;
    std::vector<double> T_marker_camera;
    int num_points = 0;
    int num_inliers = 0;
    double mean_reprojection_error_px = -1.0;
    double max_reprojection_error_px = -1.0;
    double confidence = 0.0;

    int dot_cell_count = 0;
    int dot_valid_cell_count = 0;
    int patch_count = 0;
    int decoded_patch_count = 0;
    int decoded_valid_patch_count = 0;
    int correspondence_count = 0;

    std::unordered_map<std::string, double> timings_ms;
};

} // namespace hydramarker
