#pragma once

#include <array>
#include <deque>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

struct PoseDepthFilterConfig {
    double observation_std_mm = 16.0;
    double process_std_mm = 0.05;
    double initial_velocity_std_mm = 0.1;
    double reprojection_guard_px = 1.0;
    bool innovation_guard_enabled = true;
    int innovation_guard_window = 10;
    double innovation_guard_bias_threshold_mm = 0.75;
    int innovation_guard_min_same_sign = 8;
    double innovation_cusum_slack_mm = 0.2;
    double innovation_cusum_threshold_mm = 8.0;
    bool negative_delta_guard_enabled = true;
    double negative_delta_guard_min_z_span_mm = 14.835;
    double negative_delta_guard_max_negative_delta_mm = 0.0;
    bool negative_delta_guard_hold_previous_z = false;
    bool negative_delta_guard_hold_requires_innovation_bias = true;
    double negative_delta_guard_hold_min_negative_delta_mm = 0.4;
    double negative_delta_guard_max_hold_correction_mm = 0.75;
    double negative_delta_guard_velocity_damping = 0.25;
};

struct PoseDepthFilterResult {
    std::vector<double> rvec;
    std::vector<double> tvec;
    std::vector<double> T_marker_camera;
    double raw_z_mm = 0.0;
    double filtered_z_mm = 0.0;
    double delta_z_mm = 0.0;
    double raw_reprojection_rms_px = 0.0;
    double filtered_reprojection_rms_px = 0.0;
    double reprojection_excess_px = 0.0;
    double guard_alpha = 1.0;
    bool applied = false;
    double innovation_z_mm = 0.0;
    double innovation_mean_z_mm = 0.0;
    double innovation_cusum_pos_mm = 0.0;
    double innovation_cusum_neg_mm = 0.0;
    bool innovation_bias_detected = false;
    int innovation_bias_direction = 0;
    bool innovation_bias_limited = false;
    double object_z_span_mm = 0.0;
    bool negative_delta_guard_limited = false;
};

struct PoseDepthFilterSnapshot {
    bool has_state = false;
    std::array<double, 2> x = {0.0, 0.0};
    std::array<double, 4> P = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> innovation_history;
    double innovation_cusum_pos = 0.0;
    double innovation_cusum_neg = 0.0;
};

class PoseDepthKalmanFilter {
public:
    PoseDepthKalmanFilter(
        const cv::Matx33d& K,
        const std::vector<double>& dist_coeffs = {},
        const PoseDepthFilterConfig& config = PoseDepthFilterConfig()
    );

    void reset();
    PoseDepthFilterSnapshot snapshot() const;
    void restore(const PoseDepthFilterSnapshot& state);

    PoseDepthFilterResult update(
        const std::vector<double>& rvec,
        const std::vector<double>& tvec,
        const std::vector<cv::Point3d>& object_points,
        const std::vector<cv::Point2d>& image_points
    );

private:
    struct InnovationDiagnostics {
        double innovation_z_mm = 0.0;
        double innovation_mean_z_mm = 0.0;
        double innovation_cusum_pos_mm = 0.0;
        double innovation_cusum_neg_mm = 0.0;
        bool innovation_bias_detected = false;
        int innovation_bias_direction = 0;
        bool innovation_bias_limited = false;
    };

    PoseDepthFilterConfig config_;
    cv::Matx33d K_;
    cv::Mat dist_coeffs_;

    bool has_state_ = false;
    cv::Vec2d x_ = cv::Vec2d(0.0, 0.0);
    cv::Matx22d P_ = cv::Matx22d::zeros();
    std::deque<double> innovation_history_;
    double innovation_cusum_pos_ = 0.0;
    double innovation_cusum_neg_ = 0.0;

    static cv::Mat makeDistCoeffsMat(const std::vector<double>& dist_coeffs);
    static bool vectorToMat3x1(const std::vector<double>& values, cv::Mat& mat);
    static std::vector<double> mat3x1ToVector(const cv::Mat& mat);
    static std::vector<double> transformToVector(const cv::Matx44d& T);
    static cv::Matx44d makeTransform(const cv::Mat& rvec, const cv::Mat& tvec);

    std::pair<double, InnovationDiagnostics> updateState(double measurement_z_mm);
    InnovationDiagnostics updateInnovationMonitor(double innovation_z_mm);
    std::optional<double> limitGeometryGatedNegativeDelta(
        double raw_z,
        double filtered_z,
        double object_z_span_mm,
        double previous_filtered_z,
        bool innovation_bias_detected
    );
    double reprojectionRms(
        const std::vector<cv::Point3d>& object_points,
        const std::vector<cv::Point2d>& image_points,
        const cv::Mat& rvec,
        const cv::Mat& tvec
    ) const;
};

} // namespace hydramarker
