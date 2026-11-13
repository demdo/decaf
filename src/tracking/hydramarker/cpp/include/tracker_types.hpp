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
    // overwrite rvec/tvec. Empty when IR refinement did not run. Used by the
    // drawn-tip offset carry of the Live page (fused minus RGB-only points of
    // the last fused frame, overlay.tools.tracking_overlay.ToolPoints) and by
    // tip_camera_blend() in the scripts (image plane from this pose, depth
    // from the fused one).
    std::vector<double> rvec_prefusion;
    std::vector<double> tvec_prefusion;
    // 6x6 covariance (rvec, tvec) of the output pose filter, row-major (36
    // entries), empty when the filter did not run this frame. Downstream can
    // propagate a rigidly attached point p (marker frame) to its camera-frame
    // uncertainty via J * P * J^T with J = d(R(rvec)*p + tvec)/d(rvec, tvec).
    std::vector<double> pose_covariance;
    // ---- Drill pose estimator (TrackerConfig::drill_est_enabled) ----
    // Written only with the estimator on and on a fresh frame (success and
    // a pose that is not held); never into rvec / tvec / T_marker_camera /
    // pose_covariance. The vectors are empty unless drill_est_valid (then
    // all of them are complete); drill_est_tip_mm is set also while the
    // frame is hidden (drill_est_shown false).
    bool drill_est_valid = false;     // posterior present (event 0 or 1)
    bool drill_est_shown = false;     // start-up gate over (and sigma_tip <= hide_sig)
    bool drill_est_chained = false;   // prior of the previous frame used
    int drill_est_since = 0;          // chained frames since the chain start
    std::vector<double> drill_est_rvec;              // 3 [rad], marker -> colour camera
    std::vector<double> drill_est_tvec;              // 3 [mm]
    std::vector<double> drill_est_T_marker_camera;   // 16, row-major like T_marker_camera,
                                                     // R as estimated, last row 0 0 0 1
    std::vector<double> drill_est_tip_mm;            // 3, camera frame, also if hidden
    double drill_est_sigma_tip_mm = std::numeric_limits<double>::quiet_NaN();
    // 6x6 covariance, row-major, tangent [omega (rad); tau (mm)], right
    // perturbation in the MARKER frame (not the [rvec; tvec] chart of
    // pose_covariance)
    std::vector<double> drill_est_cov;
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
    // while a model-based refine method is active, so such runs also log
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
