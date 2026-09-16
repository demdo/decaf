#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "checkerboard_types.hpp"
#include "lk_tracker.hpp"

namespace hydramarker {

// Why a tracked corner did or did not survive the propagation filters of
// TrackingValidator::validate, in the order the tests are applied. Recorded
// only when the caller passes a verdict buffer; the default nullptr keeps the
// hot path free of the extra write and leaves behaviour unchanged. Used for
// documentation figures, not by the tracker itself.
enum class TrackVerdict : std::uint8_t {
    Accepted = 0,
    FlowFailed = 1,          // optical flow itself reported failure
    FlowResidual = 2,        // aligned-window residual above max_lk_error
    ForwardBackward = 3,     // backward track did not return to the origin
    Border = 4,              // in the border margin, flow window truncated
    NoNeighbourSupport = 5   // no grid neighbour kept a plausible edge length
};

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

    // `verdicts`, when given, receives one TrackVerdict per corner of
    // `previous`, in the same order. Purely diagnostic: passing nullptr (the
    // default) reproduces the previous behaviour exactly.
    TrackingValidationResult validate(
        const CheckerboardDetection& previous,
        const LKTrackingResult& lk,
        const cv::Size& image_size,
        const CheckerboardDetectorConfig& config,
        std::vector<std::uint8_t>* verdicts = nullptr
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
