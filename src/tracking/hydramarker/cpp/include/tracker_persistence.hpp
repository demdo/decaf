#pragma once

#include <array>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "checkerboard_types.hpp"
#include "marker_geometry.hpp"
#include "tracker_config.hpp"
#include "tracker_depth_filter.hpp"
#include "tracker_pose.hpp"

namespace hydramarker {

struct GlobalCornerIdentity {
    int global_row = -1;
    int global_col = -1;
    std::array<double, 3> xyz_mm = {0.0, 0.0, 0.0};
    std::array<double, 2> uv = {0.0, 0.0};
    int votes = 0;
};

struct TrackerCorner {
    int local_row = -1;
    int local_col = -1;
    int global_row = -1;
    int global_col = -1;
    std::array<double, 3> xyz_mm = {0.0, 0.0, 0.0};
    std::array<double, 2> uv = {0.0, 0.0};
    int votes = 0;
};

struct PersistentMatchStats {
    int age = 0;
    int identities = 0;
    int current_corners = 0;
    int accepted = 0;
    bool used_pose_projection = false;
    double adaptive_motion_px = 0.0;
    double adaptive_max_dist_px = 0.0;
    int rejected_no_projection = 0;
    int rejected_far = 0;
    int rejected_ambiguous = 0;
    int rejected_claimed = 0;
};

struct PersistentMatchResult {
    std::vector<PoseTrackPoint> points;
    std::vector<TrackerCorner> corners;
    PersistentMatchStats stats;
    std::string message;

    bool valid() const {
        return !points.empty() && points.size() == corners.size();
    }
};

struct PersistentPoseSeedResult {
    std::vector<PoseTrackPoint> points;
    std::vector<TrackerCorner> corners;
    PersistentMatchStats stats;
    MapPoseResult pose;
    std::string message;

    double match_ms = 0.0;
    double pose_ms = 0.0;
    double total_ms = 0.0;

    bool validMatch() const {
        return !points.empty() && points.size() == corners.size();
    }
};

struct FastDenseGateMetrics {
    double match_ratio = 0.0;
    double motion_px = 0.0;
    double ambiguous_count = 0.0;
    double seed_mean_px = -1.0;
    double seed_max_px = -1.0;
};

struct FastDenseProjectionStats {
    int detected = 0;
    int projected = 0;
    int rejected_no_projection = 0;
    int rejected_far = 0;
    int rejected_ambiguous = 0;
    int rejected_non_mutual = 0;
    double median_error_px = std::numeric_limits<double>::infinity();
    double p90_error_px = std::numeric_limits<double>::infinity();
    double image_coverage = -1.0;
    double image_span_u_px = -1.0;
    double image_span_v_px = -1.0;
    double object_span_mm = -1.0;
    int distinct_rows = 0;
    int distinct_cols = 0;
};

struct FastAcceptedState {
    bool evaluated = false;
    bool reliable_pose = false;

    std::vector<double> rvec;
    std::vector<double> tvec;
    std::vector<double> T_marker_camera;

    double last_good_reproj_px = -1.0;
    int max_pts_seen = 0;
    int accepted_pose_frame = -1;
    int visual_corner_count = 0;
};

struct FastPoseResult {
    bool attempted = false;
    bool success = false;
    bool route_decode = false;

    bool used_dense = false;
    bool dense_required = false;
    bool dense_attempted = false;
    bool dense_success = false;
    bool rescue_required = false;

    std::string reason;
    std::string dense_reason;
    std::string dense_gate_reason;
    std::string seed_error_reason;

    std::vector<PoseTrackPoint> points;
    std::vector<TrackerCorner> corners;
    std::vector<TrackerCorner> visual_corners;
    PersistentMatchStats stats;
    FastDenseProjectionStats dense_stats;
    FastDenseGateMetrics dense_gate_metrics;

    MapPoseResult seed_pose;
    MapPoseResult pose;
    MapPoseResult depth_filtered_pose;
    PoseDepthFilterResult depth_filter_result;

    double persistent_match_ms = 0.0;
    double seed_pnp_ms = 0.0;
    double cpp_seed_total_ms = 0.0;
    double dense_match_ms = 0.0;
    double dense_pose_ms = 0.0;
    double depth_filter_ms = 0.0;
    double persistence_refresh_ms = 0.0;
    double total_ms = 0.0;

    int min_points = 0;
    int min_dense_points = 0;
    int dense_matches = 0;
    bool depth_filter_available = false;
    FastAcceptedState accepted_state;

    bool persistence_refresh_available = false;
    int persistence_refresh_frame = -1;
    int persistence_refresh_count = 0;
    std::vector<GlobalCornerIdentity> persistence_refresh_identities;
};

class PersistentMatcher {
public:
    explicit PersistentMatcher(
        const TrackerConfig& config = TrackerConfig()
    );

    void reset();
    void clearIdentities();
    void replaceIdentities(
        const std::vector<GlobalCornerIdentity>& identities,
        int frame_index
    );

    const std::vector<GlobalCornerIdentity>& identities() const;
    int persistentFrameIndex() const;
    const TrackerConfig& config() const;

    PersistentMatchResult match(
        const CheckerboardDetection& detection,
        int frame_index,
        const cv::Matx33d& K,
        const std::vector<double>& dist_coeffs = {},
        const std::vector<double>& rvec = {},
        const std::vector<double>& tvec = {},
        double last_good_reproj_px = -1.0
    );

    PersistentPoseSeedResult estimatePose(
        const CheckerboardDetection& detection,
        int frame_index,
        const cv::Matx33d& K,
        const std::vector<double>& dist_coeffs = {},
        const std::vector<double>& rvec = {},
        const std::vector<double>& tvec = {},
        double last_good_reproj_px = -1.0,
        int lost_frames = 0
    );

    FastPoseResult estimateFastPose(
        const CheckerboardDetection& detection,
        const MarkerGeometry& geometry,
        int frame_index,
        const cv::Matx33d& K,
        const std::vector<double>& dist_coeffs = {},
        const std::vector<double>& rvec = {},
        const std::vector<double>& tvec = {},
        double last_good_reproj_px = -1.0,
        const std::vector<double>& previous_rvec = {},
        const std::vector<double>& previous_tvec = {},
        int lost_frames = 0,
        PoseDepthKalmanFilter* depth_filter = nullptr,
        int max_pts_seen = 0
    );

private:
    struct DetectedCornerCpp {
        int local_row = -1;
        int local_col = -1;
        cv::Point2d uv;
    };

    struct CornerMatch {
        int index = -1;
        double best_dist = 0.0;
        double second_dist = 0.0;
        std::string reject_reason;
    };

    TrackerConfig config_;
    std::vector<GlobalCornerIdentity> identities_;
    int persistent_frame_index_ = -1;

    std::vector<cv::Point2d> prev_detection_uvs_;
    int prev_detection_frame_ = -1;
    double last_motion_px_ = 0.0;

    double detectionMotionPx(
        const std::vector<cv::Point2d>& current_uvs,
        int frame_index
    );

    double adaptiveProjectionMatchRadiusPx(
        double base_radius_px,
        double motion_px
    ) const;

    CornerMatch matchPredictedUvToDetectionCorner(
        const cv::Point2d& predicted_uv,
        const std::vector<DetectedCornerCpp>& current_corners,
        const std::vector<bool>& used_current_indices,
        double max_dist_px
    ) const;

    static std::vector<DetectedCornerCpp> detectedCornersFromDetection(
        const CheckerboardDetection& detection
    );

    static cv::Mat makeDistCoeffsMat(
        const std::vector<double>& dist_coeffs
    );

    static bool vectorToMat3x1(
        const std::vector<double>& values,
        cv::Mat& mat
    );

    static bool projectPoint(
        const std::array<double, 3>& xyz_mm,
        const cv::Matx33d& K,
        const cv::Mat& dist_coeffs,
        const cv::Mat& rvec,
        const cv::Mat& tvec,
        cv::Point2d& uv
    );

    static MapPoseTrackerConfig makeMapPoseTrackerConfig(
        const TrackerConfig& config
    );
};

} // namespace hydramarker
