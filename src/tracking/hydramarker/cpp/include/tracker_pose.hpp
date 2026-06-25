#pragma once

#include <array>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

struct PoseTrackPoint {
    int global_row = -1;
    int global_col = -1;
    std::array<double, 3> xyz_mm = {0.0, 0.0, 0.0};
    std::array<double, 2> uv = {0.0, 0.0};
    int votes = 0;
};

struct MapPoseResult {
    bool success = false;
    std::string message;

    std::vector<double> rvec;
    std::vector<double> tvec;
    std::vector<double> T_marker_camera;

    std::vector<int> inlier_indices;

    double reprojection_mean_px = -1.0;
    double reprojection_max_px = -1.0;

    int num_points = 0;
    int num_inliers = 0;

    std::vector<PoseTrackPoint> points;
    std::string method;
};

struct MapPoseTrackerConfig {
    int min_points = 8;
    int min_inliers = 6;

    double ransac_reproj_px = 3.0;
    double ransac_confidence = 0.99;
    int ransac_iterations = 500;

    double max_mean_reproj_px = 4.0;
    double max_max_reproj_px = 15.0;

    double max_translation_jump_mm = 40.0;
    double max_rotation_jump_deg = 45.0;
    double rotation_gate_scale_per_lost_frame = 8.0;
    double rotation_gate_max_deg = 90.0;

    bool use_pose_prior = true;
    bool refine_with_iterative = true;
    bool use_direct_prior_solver = true;
    std::string direct_refine_method = "lm";
    double direct_max_mean_reproj_px = 1.5;
    double direct_max_max_reproj_px = 3.0;
};

class MapPoseTracker {
public:
    MapPoseTracker(
        const cv::Matx33d& K,
        const std::vector<double>& dist_coeffs = {},
        const MapPoseTrackerConfig& config = MapPoseTrackerConfig()
    );

    void reset();
    bool setPose(
        const std::vector<double>& rvec,
        const std::vector<double>& tvec
    );

    MapPoseResult estimatePose(
        const std::vector<PoseTrackPoint>& points,
        int lost_frames = 0
    );

    bool hasPose() const;
    std::vector<double> rvec() const;
    std::vector<double> tvec() const;
    std::vector<double> TMarkerCamera() const;
    const MapPoseTrackerConfig& config() const;

private:
    MapPoseTrackerConfig config_;
    cv::Matx33d K_;
    cv::Mat dist_coeffs_;

    bool has_pose_ = false;
    cv::Mat rvec_;
    cv::Mat tvec_;
    cv::Matx44d T_marker_camera_ = cv::Matx44d::eye();

    MapPoseResult estimatePoseDirectPrior(
        const std::vector<PoseTrackPoint>& points,
        const std::vector<cv::Point3d>& object_points,
        const std::vector<cv::Point2d>& image_points,
        int lost_frames
    );

    std::tuple<cv::Mat, cv::Mat, std::string> refineDirectPriorPose(
        const std::vector<cv::Point3d>& object_points,
        const std::vector<cv::Point2d>& image_points,
        const cv::Mat& rvec,
        const cv::Mat& tvec
    ) const;

    std::pair<bool, std::string> checkMotionGate(
        const cv::Mat& candidate_rvec,
        const cv::Mat& candidate_tvec,
        int lost_frames
    ) const;

    bool computeReprojectionStats(
        const std::vector<cv::Point3d>& object_points,
        const std::vector<cv::Point2d>& image_points,
        const cv::Mat& rvec,
        const cv::Mat& tvec,
        double& mean_error_px,
        double& max_error_px
    ) const;

    void acceptPose(const cv::Mat& rvec, const cv::Mat& tvec);

    static cv::Matx44d makeTransform(const cv::Mat& rvec, const cv::Mat& tvec);
    static std::vector<double> mat3x1ToVector(const cv::Mat& mat);
    static std::vector<double> transformToVector(const cv::Matx44d& T);
    static std::string formatDouble(double value, int precision);
};

} // namespace hydramarker
