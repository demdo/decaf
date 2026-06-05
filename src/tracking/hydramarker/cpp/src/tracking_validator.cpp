#include "tracking_validator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

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

    result.visible_indices.reserve(good_indices.size());
    result.visible_points.reserve(good_indices.size());
    result.visible_predicted.reserve(good_indices.size());

    std::vector<char> exported(n, 0);

    const float min_edge_ratio =
        std::max(0.30f, config.tracking_spacing_min_rel * 0.75f);
    const float max_edge_ratio =
        std::max(1.20f, config.tracking_spacing_max_rel * 1.35f);

    std::vector<int> local_support(good_indices.size(), 0);
    std::vector<float> local_error(
        good_indices.size(),
        std::numeric_limits<float>::max());

    for (size_t a = 0; a < good_indices.size(); ++a) {
        const int ka = good_indices[a];
        const auto& ca = previous.corners[ka];

        for (size_t b = a + 1; b < good_indices.size(); ++b) {
            const int kb = good_indices[b];
            const auto& cb = previous.corners[kb];

            const int di = std::abs(ca.i - cb.i);
            const int dj = std::abs(ca.j - cb.j);
            if (di + dj != 1) {
                continue;
            }

            const float old_d = pointDistance(ca.uv, cb.uv);
            const float new_d = pointDistance(curr_good[a], curr_good[b]);
            if (old_d < 2.0f || new_d < 2.0f) {
                continue;
            }

            const float ratio = new_d / old_d;
            if (ratio < min_edge_ratio || ratio > max_edge_ratio) {
                continue;
            }

            ++local_support[a];
            ++local_support[b];

            const float e = std::abs(std::log(ratio));
            local_error[a] = std::min(local_error[a], e);
            local_error[b] = std::min(local_error[b], e);
        }
    }

    int supported_tracks = 0;
    float local_error_sum = 0.0f;
    float local_error_max = 0.0f;

    float motion_sum = 0.0f;
    int motion_count = 0;

    for (size_t m = 0; m < good_indices.size(); ++m) {
        const int k = good_indices[m];
        const cv::Point2f old_uv = previous.corners[k].uv;
        const cv::Point2f measured = lk.curr_points[k];

        if (local_support[m] <= 0) {
            continue;
        }

        ++supported_tracks;
        if (std::isfinite(local_error[m])) {
            local_error_sum += local_error[m];
            local_error_max = std::max(local_error_max, local_error[m]);
        }

        result.visible_indices.push_back(k);
        result.visible_points.push_back(measured);
        result.visible_predicted.push_back(false);
        exported[k] = 1;
        ++result.directly_tracked;

        motion_sum += pointDistance(old_uv, measured);
        ++motion_count;
    }

    if (result.directly_tracked < config.min_tracking_corners) {
        // Fallback for very sparse but LK-consistent frames: keep the basic
        // forward-backward-checked points instead of dropping the pose. The
        // downstream grid builder still removes points that do not form a
        // plausible checkerboard patch.
        result.visible_indices.clear();
        result.visible_points.clear();
        result.visible_predicted.clear();
        std::fill(exported.begin(), exported.end(), 0);

        motion_sum = 0.0f;
        motion_count = 0;
        result.directly_tracked = 0;

        for (size_t m = 0; m < good_indices.size(); ++m) {
            const int k = good_indices[m];
            result.visible_indices.push_back(k);
            result.visible_points.push_back(curr_good[m]);
            result.visible_predicted.push_back(false);
            exported[k] = 1;
            ++result.directly_tracked;

            motion_sum += pointDistance(previous.corners[k].uv, curr_good[m]);
            ++motion_count;
        }
    }

    result.homography_inliers = supported_tracks;
    result.inlier_ratio =
        result.lk_basic_good > 0
            ? static_cast<float>(supported_tracks) /
              static_cast<float>(result.lk_basic_good)
            : 0.0f;

    result.mean_lk_to_h_px =
        supported_tracks > 0
            ? local_error_sum / static_cast<float>(supported_tracks)
            : 0.0f;
    result.max_lk_to_h_px = local_error_max;

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

    result.homography_projected = 0;
    result.projected_ratio = 0.0f;

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
