#include "tracking_validator.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>

namespace hydramarker {

namespace {

constexpr float kBorderMarginPx = 6.0f;
constexpr float kMinHomographyInlierRatio = 0.45f;
constexpr float kStableMinDirectRatio = 0.85f;

} // namespace


TrackingValidator::TrackingValidator() = default;


// ------------------------------------------------------------

TrackingValidationResult TrackingValidator::validate(
    const CheckerboardDetection& previous,
    const LKTrackingResult& lk,
    const cv::Size& image_size,
    const CheckerboardDetectorConfig& config
) const {
    TrackingValidationResult result;

    if (!previous.valid()) {
        return result;
    }

    const int n = static_cast<int>(previous.corners.size());
    result.total = n;

    if (n < config.min_tracking_corners) {
        return result;
    }

    if (!lk.valid) {
        return result;
    }

    if (lk.prev_points.size() != previous.corners.size() ||
        lk.curr_points.size() != previous.corners.size() ||
        lk.status.size() != previous.corners.size() ||
        lk.error.size() != previous.corners.size() ||
        lk.fb_status.size() != previous.corners.size() ||
        lk.fb_error.size() != previous.corners.size()) {
        return result;
    }

    std::vector<cv::Point2f> prev_good;
    std::vector<cv::Point2f> curr_good;
    std::vector<int> good_indices;

    prev_good.reserve(previous.corners.size());
    curr_good.reserve(previous.corners.size());
    good_indices.reserve(previous.corners.size());

    for (int k = 0; k < n; ++k) {
        if (!lk.status[k]) {
            continue;
        }

        if (lk.error[k] > config.max_lk_error) {
            continue;
        }

        // Forward-backward consistency:
        // If p_old -> p_new cannot be tracked back close to p_old,
        // the forward LK result is probably a drift / wrong local match.
        if (!lk.fb_status[k]) {
            continue;
        }

        if (lk.fb_error[k] > config.max_lk_forward_backward_error_px) {
            continue;
        }

        const cv::Point2f& old_p = previous.corners[k].uv;
        const cv::Point2f& new_p = lk.curr_points[k];

        // LK patches close to the image border are incomplete and drift easily.
        // Such corners are treated as not currently visible.
        if (!isInsideImageSafe(old_p, image_size, kBorderMarginPx)) {
            continue;
        }

        if (!isInsideImageSafe(new_p, image_size, kBorderMarginPx)) {
            continue;
        }

        prev_good.push_back(old_p);
        curr_good.push_back(new_p);
        good_indices.push_back(k);
    }

    result.lk_basic_good = static_cast<int>(curr_good.size());

    if (result.lk_basic_good < config.min_tracking_corners) {
        return result;
    }

    cv::Mat H;
    std::vector<uchar> inlier_mask;

    if (prev_good.size() >= 4) {
        H = cv::findHomography(
            prev_good,
            curr_good,
            cv::RANSAC,
            config.max_tracking_homography_error_px,
            inlier_mask
        );
    }

    if (H.empty() || inlier_mask.size() != prev_good.size()) {
        return result;
    }

    int inliers = 0;
    for (uchar v : inlier_mask) {
        if (v) {
            ++inliers;
        }
    }

    result.homography_inliers = inliers;
    result.inlier_ratio =
        prev_good.empty()
            ? 0.0f
            : static_cast<float>(inliers) /
              static_cast<float>(prev_good.size());

    if (result.inlier_ratio < kMinHomographyInlierRatio) {
        return result;
    }

    const float max_h_residual =
        std::max(2.0f, config.max_tracking_homography_error_px);

    result.visible_indices.reserve(good_indices.size());
    result.visible_points.reserve(good_indices.size());

    float residual_sum = 0.0f;
    int residual_count = 0;

    float motion_sum = 0.0f;
    int motion_count = 0;

    for (size_t m = 0; m < good_indices.size(); ++m) {
        if (!inlier_mask[m]) {
            continue;
        }

        const int k = good_indices[m];
        const cv::Point2f old_uv = previous.corners[k].uv;
        const cv::Point2f measured = lk.curr_points[k];
        const cv::Point2f projected = projectPoint(H, old_uv);

        if (!isInsideImageSafe(projected, image_size, kBorderMarginPx)) {
            continue;
        }

        const float residual = pointDistance(measured, projected);

        residual_sum += residual;
        ++residual_count;

        result.max_lk_to_h_px =
            std::max(result.max_lk_to_h_px, residual);

        if (residual > max_h_residual) {
            continue;
        }

        // Key rule:
        // Only directly measured LK points are exported.
        // Homography projection is used only as a plausibility test.
        result.visible_indices.push_back(k);
        result.visible_points.push_back(measured);
        ++result.directly_tracked;

        motion_sum += pointDistance(old_uv, measured);
        ++motion_count;
    }

    result.mean_lk_to_h_px =
        residual_count > 0
            ? residual_sum / static_cast<float>(residual_count)
            : 0.0f;

    result.mean_motion_px =
        motion_count > 0
            ? motion_sum / static_cast<float>(motion_count)
            : 0.0f;

    result.rejected = n - result.directly_tracked;
    result.homography_projected = 0;
    result.projected_ratio = 0.0f;

    result.direct_ratio =
        n > 0
            ? static_cast<float>(result.directly_tracked) /
              static_cast<float>(n)
            : 0.0f;

    if (result.directly_tracked < config.min_tracking_corners) {
        return result;
    }

    // Degrade rule:
    // If too many previously visible corners are no longer directly confirmed,
    // tracking must stop and recovery must re-anchor the detector.
    if (result.direct_ratio < config.min_tracking_corner_ratio) {
        return result;
    }

    result.valid = true;

    result.stable =
        result.mean_motion_px <= config.stable_motion_threshold_px &&
        result.direct_ratio >= kStableMinDirectRatio;

    return result;
}


// ------------------------------------------------------------

bool TrackingValidator::isInsideImage(
    const cv::Point2f& p,
    const cv::Size& size
) {
    return
        p.x >= 0.0f &&
        p.y >= 0.0f &&
        p.x < static_cast<float>(size.width) &&
        p.y < static_cast<float>(size.height);
}


// ------------------------------------------------------------

bool TrackingValidator::isInsideImageSafe(
    const cv::Point2f& p,
    const cv::Size& size,
    float margin_px
) {
    return
        p.x >= margin_px &&
        p.y >= margin_px &&
        p.x < static_cast<float>(size.width) - margin_px &&
        p.y < static_cast<float>(size.height) - margin_px;
}


// ------------------------------------------------------------

float TrackingValidator::pointDistance(
    const cv::Point2f& a,
    const cv::Point2f& b
) {
    const cv::Point2f d = a - b;
    return std::sqrt(d.x * d.x + d.y * d.y);
}


// ------------------------------------------------------------

cv::Point2f TrackingValidator::projectPoint(
    const cv::Mat& H,
    const cv::Point2f& p
) {
    const double x =
        H.at<double>(0, 0) * p.x +
        H.at<double>(0, 1) * p.y +
        H.at<double>(0, 2);

    const double y =
        H.at<double>(1, 0) * p.x +
        H.at<double>(1, 1) * p.y +
        H.at<double>(1, 2);

    const double w =
        H.at<double>(2, 0) * p.x +
        H.at<double>(2, 1) * p.y +
        H.at<double>(2, 2);

    if (std::abs(w) < 1e-9) {
        return p;
    }

    return cv::Point2f(
        static_cast<float>(x / w),
        static_cast<float>(y / w)
    );
}

} // namespace hydramarker