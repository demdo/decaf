#pragma once

#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "checkerboard_types.hpp"
#include "marker_geometry.hpp"
#include "tracker_config.hpp"
#include "tracker_persistence.hpp"

namespace hydramarker {

struct DenseProjectionMatchStats {
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

struct DenseProjectionMatchResult {
    std::vector<TrackerCorner> corners;
    DenseProjectionMatchStats stats;

    bool valid() const {
        return !corners.empty();
    }
};

class TrackerGeometry {
public:
    TrackerGeometry(
        const MarkerGeometry& geometry,
        const cv::Matx33d& K,
        const std::vector<double>& dist_coeffs = {},
        const TrackerConfig& config = TrackerConfig()
    );

    DenseProjectionMatchResult strictProjectedMatch(
        const CheckerboardDetection& detection,
        const std::vector<double>& rvec,
        const std::vector<double>& tvec,
        double max_dist_px,
        double ambiguity_margin_px
    ) const;

    DenseProjectionMatchResult greedyProjectedMatch(
        const CheckerboardDetection& detection,
        const std::vector<double>& rvec,
        const std::vector<double>& tvec,
        double max_dist_px
    ) const;

    std::vector<TrackerCorner> visualCornersFromPose(
        const std::vector<TrackerCorner>& corners,
        const std::vector<double>& rvec,
        const std::vector<double>& tvec,
        double max_error_px
    ) const;

    MapPoseResult estimateDenseRobustPose(
        const std::vector<PoseTrackPoint>& points,
        const CheckerboardDetection& detection,
        const std::vector<double>& seed_rvec = {},
        const std::vector<double>& seed_tvec = {},
        const std::vector<double>& previous_rvec = {},
        const std::vector<double>& previous_tvec = {}
    ) const;

private:
    struct DensePoseCandidate {
        cv::Mat rvec;
        cv::Mat tvec;
        std::string method;
    };

    struct DensePoseScore {
        double score = std::numeric_limits<double>::infinity();
        std::vector<double> errors;
    };

    struct ProjectedCorner {
        int global_row = -1;
        int global_col = -1;
        cv::Point3d xyz_mm;
        cv::Point2d uv;
    };

    MarkerGeometry geometry_;
    cv::Matx33d K_;
    cv::Mat dist_coeffs_;
    TrackerConfig config_;
    std::vector<ProjectedCorner> cached_corners_;

    static cv::Mat makeDistCoeffsMat(const std::vector<double>& dist_coeffs);
    static bool vectorToMat3x1(const std::vector<double>& values, cv::Mat& mat);
    static double percentile(std::vector<double> values, double q);
    static std::string formatDouble(double value, int precision);
    static std::string lowerAscii(std::string value);
    static std::vector<double> mat3x1ToVector(const cv::Mat& mat);
    static std::vector<double> transformToVector(const cv::Matx44d& T);
    static cv::Matx44d makeTransform(const cv::Mat& rvec, const cv::Mat& tvec);

    std::vector<ProjectedCorner> projectGeometry(
        const cv::Mat& rvec,
        const cv::Mat& tvec,
        int& rejected_no_projection
    ) const;

    std::vector<DensePoseCandidate> denseRefinePoseVariants(
        const std::vector<cv::Point3d>& object_points,
        const std::vector<cv::Point2d>& image_points,
        const cv::Mat& rvec,
        const cv::Mat& tvec,
        const std::string& method_prefix
    ) const;

    bool scoreDensePoseCandidate(
        const std::vector<cv::Point3d>& object_points,
        const std::vector<cv::Point2d>& image_points,
        const cv::Mat& rvec,
        const cv::Mat& tvec,
        DensePoseScore& score
    ) const;

    bool reprojectionErrors(
        const std::vector<cv::Point3d>& object_points,
        const std::vector<cv::Point2d>& image_points,
        const cv::Mat& rvec,
        const cv::Mat& tvec,
        std::vector<double>& errors
    ) const;

    bool persistentMotionPlausible(
        const cv::Mat& rvec,
        const cv::Mat& tvec,
        const cv::Mat& previous_rvec,
        const cv::Mat& previous_tvec
    ) const;

    std::string fallbackPoseRejectionReason(
        const CheckerboardDetection& detection,
        const cv::Mat& rvec,
        const cv::Mat& tvec,
        double mean_reproj_px,
        double max_reproj_px
    ) const;
};

} // namespace hydramarker
