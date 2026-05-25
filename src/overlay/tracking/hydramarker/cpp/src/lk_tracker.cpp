#include "lk_tracker.hpp"

#include <cmath>

#include <opencv2/video/tracking.hpp>

namespace hydramarker {

namespace {

static float pointDistance(
    const cv::Point2f& a,
    const cv::Point2f& b
) {
    const cv::Point2f d = a - b;
    return std::sqrt(d.x * d.x + d.y * d.y);
}

} // namespace

LKTracker::LKTracker() = default;

LKTrackingResult LKTracker::track(
    const cv::Mat& prev_gray,
    const cv::Mat& curr_gray,
    const std::vector<cv::Point2f>& prev_points,
    int win_size,
    int max_level,
    int max_iters,
    double epsilon,
    float /*max_error*/
) const {
    LKTrackingResult result;

    if (prev_gray.empty() || curr_gray.empty() || prev_points.empty()) {
        return result;
    }

    cv::TermCriteria criteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
        max_iters,
        epsilon
    );

    std::vector<cv::Point2f> curr_points;
    std::vector<uchar> status;
    std::vector<float> error;

    // Forward LK: previous frame -> current frame.
    cv::calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_points,
        curr_points,
        status,
        error,
        cv::Size(win_size, win_size),
        max_level,
        criteria
    );

    if (
        curr_points.size() != prev_points.size() ||
        status.size() != prev_points.size() ||
        error.size() != prev_points.size()
    ) {
        return result;
    }

    std::vector<cv::Point2f> back_points;
    std::vector<uchar> back_status;
    std::vector<float> back_error;

    // Backward LK: current frame -> previous frame.
    // This catches tracks that look locally plausible in forward direction,
    // but do not return to the original corner.
    cv::calcOpticalFlowPyrLK(
        curr_gray,
        prev_gray,
        curr_points,
        back_points,
        back_status,
        back_error,
        cv::Size(win_size, win_size),
        max_level,
        criteria
    );

    if (
        back_points.size() != prev_points.size() ||
        back_status.size() != prev_points.size()
    ) {
        return result;
    }

    std::vector<uchar> fb_status(prev_points.size(), 0);
    std::vector<float> fb_error(prev_points.size(), 1.0e9f);

    for (size_t k = 0; k < prev_points.size(); ++k) {
        if (!status[k] || !back_status[k]) {
            continue;
        }

        fb_error[k] = pointDistance(prev_points[k], back_points[k]);
        fb_status[k] = 1;
    }

    result.prev_points = prev_points;
    result.curr_points = std::move(curr_points);
    result.status = std::move(status);
    result.error = std::move(error);
    result.fb_status = std::move(fb_status);
    result.fb_error = std::move(fb_error);
    result.valid = true;

    return result;
}

} // namespace hydramarker
