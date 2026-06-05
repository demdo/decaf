#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "checkerboard_types.hpp"
#include "lk_tracker.hpp"

namespace hydramarker {

struct TrackingValidationResult {
    // Only directly confirmed, currently visible corners.
    // No homography-only projections and no old_uv fallback points are exported.
    std::vector<int> visible_indices;
    std::vector<cv::Point2f> visible_points;
    std::vector<bool> visible_predicted;

    int total = 0;
    int lk_basic_good = 0;
    int homography_inliers = 0;
    int directly_tracked = 0;
    int homography_projected = 0;
    int rejected = 0;

    float inlier_ratio = 0.0f;
    float direct_ratio = 0.0f;
    float projected_ratio = 0.0f;
    float mean_motion_px = 0.0f;
    float mean_lk_to_h_px = 0.0f;
    float max_lk_to_h_px = 0.0f;

    bool valid = false;
    bool stable = false;
};

class TrackingValidator {
public:
    TrackingValidator();

    TrackingValidationResult validate(
        const CheckerboardDetection& previous,
        const LKTrackingResult& lk,
        const cv::Size& image_size,
        const CheckerboardDetectorConfig& config
    ) const;

private:
    static bool isInsideImage(
        const cv::Point2f& p,
        const cv::Size& size
    );

    static bool isInsideImageSafe(
        const cv::Point2f& p,
        const cv::Size& size,
        float margin_px
    );

    static float pointDistance(
        const cv::Point2f& a,
        const cv::Point2f& b
    );

    static cv::Point2f projectPoint(
        const cv::Mat& H,
        const cv::Point2f& p
    );
};

} // namespace hydramarker
