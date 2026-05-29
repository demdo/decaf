#include "corner_detection.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace hydramarker {

namespace {

cv::Mat toGray8(const cv::Mat& gray) {
    cv::Mat gray8;

    if (gray.empty()) {
        return gray8;
    }

    if (gray.channels() == 1) {
        if (gray.depth() == CV_8U) {
            gray8 = gray.clone();
        } else {
            cv::normalize(gray, gray8, 0, 255, cv::NORM_MINMAX, CV_8U);
        }
    } else {
        cv::Mat tmp;
        cv::cvtColor(gray, tmp, cv::COLOR_BGR2GRAY);

        if (tmp.depth() == CV_8U) {
            gray8 = tmp;
        } else {
            cv::normalize(tmp, gray8, 0, 255, cv::NORM_MINMAX, CV_8U);
        }
    }

    return gray8;
}

cv::Mat makeFastImage(const cv::Mat& gray) {
    cv::Mat gray8 = toGray8(gray);

    if (gray8.empty()) {
        return gray8;
    }

    cv::Mat blurred;
    cv::GaussianBlur(gray8, blurred, cv::Size(3, 3), 0.45);

    return blurred;
}

cv::Mat localNormalizeCLAHE(const cv::Mat& gray8) {
    CV_Assert(gray8.type() == CV_8U);

    cv::Mat denoised;
    cv::GaussianBlur(gray8, denoised, cv::Size(3, 3), 0.6);

    cv::Mat clahe_img;

    static cv::Ptr<cv::CLAHE> clahe =
        cv::createCLAHE(2.5, cv::Size(8, 8));

    clahe->apply(denoised, clahe_img);

    return clahe_img;
}

cv::Mat gammaNormalize(const cv::Mat& gray8, double gamma) {
    CV_Assert(gray8.type() == CV_8U);

    cv::Mat lut(1, 256, CV_8U);
    uchar* l = lut.ptr<uchar>(0);

    const double inv_gamma = 1.0 / std::max(1e-6, gamma);

    for (int i = 0; i < 256; ++i) {
        const double x = static_cast<double>(i) / 255.0;
        const int y = static_cast<int>(
            std::round(std::pow(x, inv_gamma) * 255.0)
        );

        l[i] = static_cast<uchar>(std::clamp(y, 0, 255));
    }

    cv::Mat out;
    cv::LUT(gray8, lut, out);

    return out;
}

std::vector<cv::Mat> buildRecoveryVariants(const cv::Mat& gray) {
    std::vector<cv::Mat> variants;

    cv::Mat gray8 = toGray8(gray);

    if (gray8.empty()) {
        return variants;
    }

    variants.push_back(gray8);

    cv::Mat fast_img;
    cv::GaussianBlur(gray8, fast_img, cv::Size(3, 3), 0.45);
    variants.push_back(fast_img);

    cv::Mat clahe_img = localNormalizeCLAHE(gray8);
    variants.push_back(clahe_img);

    cv::Mat gamma_bright = gammaNormalize(clahe_img, 1.35);
    variants.push_back(gamma_bright);

    cv::Mat gamma_dark = gammaNormalize(clahe_img, 0.75);
    variants.push_back(gamma_dark);

    return variants;
}

} // namespace

// --------------------------------------------

CornerDetector::CornerDetector() = default;

// --------------------------------------------

CornerDetectionResult CornerDetector::detect(
    const cv::Mat& gray,
    int max_corners,
    float sigma,
    float response_threshold
) const {
    if (gray.empty()) {
        return CornerDetectionResult{};
    }

    const cv::Mat fast_img = makeFastImage(gray);

    if (fast_img.empty()) {
        return CornerDetectionResult{};
    }

    CornerDetectionResult fast_result = detectGradientJunctions(
        fast_img,
        std::max(max_corners, 300),
        sigma,
        response_threshold
    );

    std::vector<cv::Point2f> fast_points = fast_result.points;

    std::vector<cv::Point2f> fast_gftt = detectFastCandidates(fast_img);

    fast_points = mergeCandidates(
        fast_points,
        fast_gftt,
        4.0f,
        max_corners * 2
    );

    // Accept the fast path only when we have a solid majority of expected
    // corners.  0.45 was too permissive: with max_corners=150 it accepted 67
    // points and skipped recovery variants even when the upper rows were
    // missing.  0.65 forces recovery variants whenever a significant portion
    // of corners is absent, at the cost of slightly more CPU on full frames.
    const int fast_accept_min = std::max(
        28,
        static_cast<int>(std::round(static_cast<float>(max_corners) * 0.65f))
    );

    if (static_cast<int>(fast_points.size()) >= fast_accept_min) {
        if (static_cast<int>(fast_points.size()) > max_corners * 2) {
            fast_points.resize(max_corners * 2);
        }

        // Fast path: points and gradients both come from fast_img, consistent.
        fast_result.points = std::move(fast_points);
        return fast_result;
    }

    const std::vector<cv::Mat> variants = buildRecoveryVariants(gray);

    if (variants.empty()) {
        fast_result.points = std::move(fast_points);
        return fast_result;
    }

    std::vector<cv::Point2f> merged = fast_points;

    const int per_variant_max = std::max(max_corners, 300);

    for (const cv::Mat& v : variants) {
        CornerDetectionResult jr = detectGradientJunctions(
            v,
            per_variant_max,
            sigma,
            response_threshold * 0.65f
        );

        merged = mergeCandidates(
            merged,
            jr.points,
            3.0f,
            max_corners * 3
        );

        std::vector<cv::Point2f> gftt = detectFastCandidates(v);

        merged = mergeCandidates(
            merged,
            gftt,
            4.0f,
            max_corners * 3
        );
    }

    if (static_cast<int>(merged.size()) > max_corners * 2) {
        merged.resize(max_corners * 2);
    }

    // Gradient fix: in the recovery path, points come from a mix of all
    // variants (original, blurred, CLAHE, gamma-bright, gamma-dark).
    // The previous code returned gradients from variants[2] (CLAHE), which
    // are inconsistent with points sourced from the other variants.
    // CornerRefiner::refineGradientIntersections uses these gradients to solve
    // a local least-squares system per candidate — mismatched gradients cause
    // subtle position drift, especially on low-contrast corners.
    //
    // Fix: recompute gradients from variants[0] (plain gray8, no CLAHE/gamma).
    // This is the image closest to what the refined positions will actually
    // land on, and avoids CLAHE amplifying local texture artefacts into the
    // gradient field.
    CornerDetectionResult result = detectGradientJunctions(
        variants[0],
        1,             // max_corners=1: we only want grad_x/grad_y, not points
        sigma,
        0.0f           // threshold=0: suppress all point output, keep grads
    );

    result.points = std::move(merged);

    return result;
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

    cv::Mat gx;
    cv::Mat gy;

    cv::Sobel(img, gx, CV_32F, 1, 0, 3);
    cv::Sobel(img, gy, CV_32F, 0, 1, 3);

    cv::Mat magnitude;
    cv::magnitude(gx, gy, magnitude);

    cv::Mat gpow;
    cv::Mat gx_blur;
    cv::Mat gy_blur;
    cv::Mat gsum;

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

    cv::Mat dilated;
    cv::dilate(response, dilated, cv::Mat());

    response.setTo(0, response != dilated);
    cv::max(response, 0, response);

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

    std::vector<cv::Point2f> points;

    for (int y = 0; y < response.rows; ++y) {
        const float* row = response.ptr<float>(y);

        for (int x = 0; x < response.cols; ++x) {
            if (row[x] > 0.0f) {
                points.emplace_back(
                    static_cast<float>(x),
                    static_cast<float>(y)
                );
            }
        }
    }

    if (static_cast<int>(points.size()) > max_corners) {
        std::partial_sort(
            points.begin(),
            points.begin() + max_corners,
            points.end(),
            [&](const cv::Point2f& a, const cv::Point2f& b) {
                return response.at<float>(
                           static_cast<int>(a.y),
                           static_cast<int>(a.x)
                       ) >
                       response.at<float>(
                           static_cast<int>(b.y),
                           static_cast<int>(b.x)
                       );
            }
        );

        points.resize(max_corners);
    }

    CornerDetectionResult result;

    result.points = std::move(points);
    result.response = response;
    result.grad_x = gx;
    result.grad_y = gy;

    return result;
}

// --------------------------------------------
// FAST / fallback
// --------------------------------------------

std::vector<cv::Point2f> CornerDetector::detectFastCandidates(
    const cv::Mat& gray
) const {
    std::vector<cv::Point2f> corners;

    cv::Mat gray8 = toGray8(gray);

    if (gray8.empty()) {
        return corners;
    }

    cv::goodFeaturesToTrack(
        gray8,
        corners,
        400,
        0.004,
        7,
        cv::Mat(),
        5,
        false,
        0.04
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
            const float dx = p.x - q.x;
            const float dy = p.y - q.y;

            if (dx * dx + dy * dy < r2) {
                duplicate = true;
                break;
            }
        }

        if (!duplicate) {
            merged.push_back(p);
        }
    }

    if (max_points > 0 && static_cast<int>(merged.size()) > max_points) {
        merged.resize(max_points);
    }

    return merged;
}

} // namespace hydramarker