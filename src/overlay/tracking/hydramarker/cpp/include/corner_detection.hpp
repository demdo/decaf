#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

struct CornerDetectionResult {
    std::vector<cv::Point2f> points;

    // optional debug / reuse (für refinement wichtig!)
    cv::Mat response;
    cv::Mat grad_x;
    cv::Mat grad_y;
};

class CornerDetector {
public:
    CornerDetector();

    CornerDetectionResult detect(
        const cv::Mat& gray,
        int max_corners,
        float sigma,
        float response_threshold
    ) const;

private:
    // --- Samu pre_filter (Gradient Junction Response) ---
    CornerDetectionResult detectGradientJunctions(
        const cv::Mat& gray,
        int max_corners,
        float sigma,
        float response_threshold
    ) const;

    // --- dein schneller Ansatz (optional) ---
    std::vector<cv::Point2f> detectFastCandidates(
        const cv::Mat& gray
    ) const;

    // --- merging ---
    std::vector<cv::Point2f> mergeCandidates(
        const std::vector<cv::Point2f>& a,
        const std::vector<cv::Point2f>& b,
        float merge_radius,
        int max_points
    ) const;
};

} // namespace hydramarker