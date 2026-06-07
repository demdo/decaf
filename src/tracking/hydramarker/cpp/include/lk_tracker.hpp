#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

struct LKTrackingResult {
    std::vector<cv::Point2f> prev_points;
    std::vector<cv::Point2f> curr_points;

    std::vector<uchar> status;
    std::vector<float> error;

    // Forward-backward consistency check.
    // fb_error[k] = distance between original prev_points[k]
    // and the point obtained by tracking curr_points[k] back to the previous frame.
    std::vector<uchar> fb_status;
    std::vector<float> fb_error;

    bool valid = false;
};

class LKTracker {
public:
    LKTracker();

    LKTrackingResult track(
        const cv::Mat& prev_gray,
        const cv::Mat& curr_gray,
        const std::vector<cv::Point2f>& prev_points,
        int win_size,
        int max_level,
        int max_iters,
        double epsilon,
        float max_error,
        const std::vector<cv::Mat>* prev_pyramid_cache = nullptr,
        std::vector<cv::Mat>* curr_pyramid_out = nullptr
    ) const;
};

} // namespace hydramarker
