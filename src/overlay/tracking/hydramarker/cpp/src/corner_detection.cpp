#include "corner_detection.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace hydramarker {

// --------------------------------------------

CornerDetector::CornerDetector() = default;

// --------------------------------------------

CornerDetectionResult CornerDetector::detect(
    const cv::Mat& gray,
    int max_corners,
    float sigma,
    float response_threshold
) const {
    // --- Samu ---
    auto grad_result = detectGradientJunctions(
        gray,
        max_corners,
        sigma,
        response_threshold
    );

    // --- optional: unser schneller Ansatz ---
    auto fast = detectFastCandidates(gray);

    // --- merge ---
    auto merged = mergeCandidates(
        grad_result.points,
        fast,
        4.0f,
        max_corners
    );

    grad_result.points = std::move(merged);
    return grad_result;
}

// --------------------------------------------
// Samu: pre_filter
// --------------------------------------------

CornerDetectionResult CornerDetector::detectGradientJunctions(
    const cv::Mat& gray,
    int max_corners,
    float sigma,
    float response_threshold
) const {
    cv::Mat img;
    gray.convertTo(img, CV_32F);

    cv::Mat gx, gy;
    cv::Sobel(img, gx, CV_32F, 1, 0, 3);
    cv::Sobel(img, gy, CV_32F, 0, 1, 3);

    cv::Mat magnitude;
    cv::magnitude(gx, gy, magnitude);

    cv::Mat gpow, gx_blur, gy_blur, gsum;

    if (sigma > 0.0f) {
        cv::GaussianBlur(magnitude, gpow, cv::Size(), sigma);
        cv::GaussianBlur(gx, gx_blur, cv::Size(), sigma);
        cv::GaussianBlur(gy, gy_blur, cv::Size(), sigma);
        cv::magnitude(gx_blur, gy_blur, gsum);
    } else {
        gpow = magnitude;
        gsum = magnitude;
    }

    cv::Mat response = gpow - gsum;

    // local maxima
    cv::Mat dilated;
    cv::dilate(response, dilated, cv::Mat());

    response.setTo(0, response != dilated);
    cv::max(response, 0, response);

    // Fix B: response_threshold is treated as a relative fraction of the
    // per-image maximum, not as an absolute pixel value.
    //
    // The original code passed the threshold directly to cv::threshold(),
    // which interprets it as an absolute float value.  The junction response
    // (gpow - gsum) has a scale that depends on image contrast, resolution,
    // and Gaussian sigma — on a typical 8-bit image it reaches values in the
    // range [0, ~200].  The config default of 0.1 therefore suppressed almost
    // everything, causing zero candidates on most frames (Image 1 & 2:
    // "checker: no detection" despite a clearly visible board).
    //
    // Interpreting response_threshold as a fraction of the image maximum
    // makes the filter scale-invariant: 0.1 means "keep only the top 90 %
    // of the response range", which is a sensible, contrast-independent
    // behaviour regardless of lighting or resolution.
    if (response_threshold > 0.0f) {
        double max_val = 0.0;
        cv::minMaxLoc(response, nullptr, &max_val);

        if (max_val > 0.0) {
            const float abs_threshold =
                static_cast<float>(max_val) * response_threshold;

            cv::threshold(
                response,
                response,
                static_cast<double>(abs_threshold),
                0,
                cv::THRESH_TOZERO
            );
        }
    }

    // collect candidates
    std::vector<cv::Point2f> points;

    for (int y = 0; y < response.rows; ++y) {
        const float* row = response.ptr<float>(y);
        for (int x = 0; x < response.cols; ++x) {
            if (row[x] > 0.0f) {
                points.emplace_back(static_cast<float>(x),
                                    static_cast<float>(y));
            }
        }
    }

    // top-k selection
    if (static_cast<int>(points.size()) > max_corners) {
        std::partial_sort(
            points.begin(),
            points.begin() + max_corners,
            points.end(),
            [&](const cv::Point2f& a, const cv::Point2f& b) {
                return response.at<float>(static_cast<int>(a.y),
                                          static_cast<int>(a.x)) >
                       response.at<float>(static_cast<int>(b.y),
                                          static_cast<int>(b.x));
            }
        );
        points.resize(max_corners);
    }

    CornerDetectionResult result;
    result.points   = std::move(points);
    result.response = response;
    result.grad_x   = gx;
    result.grad_y   = gy;

    return result;
}

// --------------------------------------------
// FAST / fallback (simple version for now)
// --------------------------------------------

std::vector<cv::Point2f> CornerDetector::detectFastCandidates(
    const cv::Mat& gray
) const {
    std::vector<cv::Point2f> corners;

    cv::goodFeaturesToTrack(
        gray,
        corners,
        200,
        0.01,
        10
    );

    return corners;
}

// --------------------------------------------
// Merge
// --------------------------------------------

std::vector<cv::Point2f> CornerDetector::mergeCandidates(
    const std::vector<cv::Point2f>& a,
    const std::vector<cv::Point2f>& b,
    float merge_radius,
    int max_points
) const {
    std::vector<cv::Point2f> merged = a;

    const float r2 = merge_radius * merge_radius;

    for (const auto& p : b) {
        bool duplicate = false;

        for (const auto& q : merged) {
            float dx = p.x - q.x;
            float dy = p.y - q.y;
            if (dx * dx + dy * dy < r2) {
                duplicate = true;
                break;
            }
        }

        if (!duplicate) {
            merged.push_back(p);
        }
    }

    if (static_cast<int>(merged.size()) > max_points) {
        merged.resize(max_points);
    }

    return merged;
}

} // namespace hydramarker