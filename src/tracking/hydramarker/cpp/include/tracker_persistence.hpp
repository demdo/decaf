#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "checkerboard_types.hpp"
#include "tracker_config.hpp"
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
};

} // namespace hydramarker
