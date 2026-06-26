#pragma once

#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

struct PlateauPosePriorConfig {
    double static_max_excess_px = 0.18;
    double candidate_max_excess_px = 0.25;
    double candidate_max_max_excess_px = 1.0;
    double min_positive_z_correction_mm = 0.0;
    double max_positive_z_correction_mm = 0.75;
    double robust_c_px = 0.20;
    int max_iterations = 6;
    double max_step_translation_mm = 5.0;
    double max_step_rotation_deg = 5.0;
    double lm_damping = 1.0e-5;
};

struct PlateauPosePriorResult {
    bool success = false;
    std::string method = "none";
    std::string reason;

    std::vector<double> rvec;
    std::vector<double> tvec;
    std::vector<double> T_marker_camera;

    double reprojection_mean_px = std::numeric_limits<double>::quiet_NaN();
    double reprojection_max_px = std::numeric_limits<double>::quiet_NaN();
    double reprojection_excess_px = std::numeric_limits<double>::quiet_NaN();
    double max_reprojection_excess_px =
        std::numeric_limits<double>::quiet_NaN();
    double delta_z_mm = std::numeric_limits<double>::quiet_NaN();
    int iterations = 0;
};

PlateauPosePriorResult solvePlateauPosePrior(
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points,
    const cv::Matx33d& K,
    const std::vector<double>& dist_coeffs,
    const std::vector<double>& raw_rvec,
    const std::vector<double>& raw_tvec,
    const std::vector<double>& seed_rvec,
    const std::vector<double>& seed_tvec,
    const PlateauPosePriorConfig& config = PlateauPosePriorConfig()
);

} // namespace hydramarker
