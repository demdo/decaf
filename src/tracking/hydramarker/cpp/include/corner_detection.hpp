#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

struct CornerDetectionResult {
    std::vector<cv::Point2f> points;

    // optional debug / reuse (für refinement wichtig!)
    cv::Mat response;
    cv::Mat grad_x;
    cv::Mat grad_y;
    std::unordered_map<std::string, double> timings_ms;
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
        float response_threshold,
        std::unordered_map<std::string, double>* timings = nullptr,
        const char* timing_prefix = nullptr
    ) const;

    // --- dein schneller Ansatz (optional) ---
    std::vector<cv::Point2f> detectFastCandidates(
        const cv::Mat& gray,
        std::unordered_map<std::string, double>* timings = nullptr,
        const char* timing_prefix = nullptr
    ) const;

    // --- merging ---
    std::vector<cv::Point2f> mergeCandidates(
        const std::vector<cv::Point2f>& a,
        const std::vector<cv::Point2f>& b,
        float merge_radius,
        int max_points,
        std::unordered_map<std::string, double>* timings = nullptr,
        const char* timing_name = nullptr
    ) const;
};

} // namespace hydramarker
