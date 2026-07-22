#pragma once

#include <array>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "checkerboard_types.hpp"
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
    double visibility_score = 1.0;
    int observed_frames = 0;
    bool predicted = false;
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
    // Latched warmup status: false right after (re)acquisition while the
    // corner set is still saturating (pose wanders along the weak
    // observability mode there); true once the set has been stable for the
    // configured window. Downstream should treat poses with
    // pose_converged == false as "initializing".
    bool pose_converged = false;
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
    // Pre-fusion (RGB-only) pose, captured before the IR stereo fusion may
    // overwrite rvec/tvec. Empty when IR refinement did not run. Downstream tip
    // computation blends the two: image-plane (camera x,y) from this RGB pose
    // (rolling-shutter-sharp, monocular-precise laterally), depth (camera z)
    // from the fused rvec/tvec (the unique IR stereo contribution) -- keeps the
    // IR depth gain without the IR lateral tilt wandering the tip over the lever.
    std::vector<double> rvec_prefusion;
    std::vector<double> tvec_prefusion;
    // 6x6 covariance (rvec, tvec) of the output pose filter, row-major (36
    // entries), empty when the filter did not run this frame. Downstream can
    // propagate a rigidly attached point p (marker frame) to its camera-frame
    // uncertainty via J * P * J^T with J = d(R(rvec)*p + tvec)/d(rvec, tvec).
    std::vector<double> pose_covariance;
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

    // Per-corner operator comparison (LK / subpix baseline / final
    // measurement) of this frame's tracked refinement.  Only populated
    // while model_warp is the active refine method, so such runs also log
    // the full subpix measurement for offline method comparison.
    std::vector<TrackedRefineSample> tracked_refine_samples;

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
