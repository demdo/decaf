#include "corner_detection.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <string>
#include <unordered_map>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace hydramarker {

namespace {

double elapsedMs(std::int64_t start_tick) {
    return 1000.0 *
           (static_cast<double>(cv::getTickCount() - start_tick) /
            cv::getTickFrequency());
}

std::string timingName(const char* prefix, const char* name) {
    return std::string(prefix ? prefix : "") + name;
}

void addTiming(
    std::unordered_map<std::string, double>* timings,
    const std::string& name,
    double elapsed_ms
) {
    if (!timings) return;
    (*timings)[name] += elapsed_ms;
}

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

std::vector<cv::Mat> buildRecoveryVariants(
    const cv::Mat& gray,
    std::unordered_map<std::string, double>* timings
) {
    std::vector<cv::Mat> variants;

    const auto to_gray_t0 = cv::getTickCount();
    cv::Mat gray8 = toGray8(gray);
    addTiming(
        timings,
        "corner_detect_variants_to_gray_ms",
        elapsedMs(to_gray_t0));

    if (gray8.empty()) {
        return variants;
    }

    variants.push_back(gray8);

    const auto fast_blur_t0 = cv::getTickCount();
    cv::Mat fast_img;
    cv::GaussianBlur(gray8, fast_img, cv::Size(3, 3), 0.45);
    addTiming(
        timings,
        "corner_detect_variants_fast_blur_ms",
        elapsedMs(fast_blur_t0));
    variants.push_back(fast_img);

    const auto clahe_t0 = cv::getTickCount();
    cv::Mat clahe_img = localNormalizeCLAHE(gray8);
    addTiming(
        timings,
        "corner_detect_variants_clahe_ms",
        elapsedMs(clahe_t0));
    variants.push_back(clahe_img);

    const auto gamma_bright_t0 = cv::getTickCount();
    cv::Mat gamma_bright = gammaNormalize(clahe_img, 1.35);
    addTiming(
        timings,
        "corner_detect_variants_gamma_bright_ms",
        elapsedMs(gamma_bright_t0));
    variants.push_back(gamma_bright);

    const auto gamma_dark_t0 = cv::getTickCount();
    cv::Mat gamma_dark = gammaNormalize(clahe_img, 0.75);
    addTiming(
        timings,
        "corner_detect_variants_gamma_dark_ms",
        elapsedMs(gamma_dark_t0));
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
    std::unordered_map<std::string, double> timings;
    const auto total_t0 = cv::getTickCount();

    auto finish = [&](CornerDetectionResult result) -> CornerDetectionResult {
        addTiming(
            &timings,
            "corner_detect_total_ms",
            elapsedMs(total_t0));
        result.timings_ms = std::move(timings);
        return result;
    };

    if (gray.empty()) {
        return finish(CornerDetectionResult{});
    }

    const auto make_fast_t0 = cv::getTickCount();
    const cv::Mat fast_img = makeFastImage(gray);
    addTiming(
        &timings,
        "corner_detect_make_fast_image_ms",
        elapsedMs(make_fast_t0));

    if (fast_img.empty()) {
        return finish(CornerDetectionResult{});
    }

    CornerDetectionResult fast_result = detectGradientJunctions(
        fast_img,
        std::max(max_corners, 300),
        sigma,
        response_threshold,
        &timings,
        "corner_detect_fast_gradient_"
    );

    std::vector<cv::Point2f> fast_points = fast_result.points;

    std::vector<cv::Point2f> fast_gftt = detectFastCandidates(
        fast_img,
        &timings,
        "corner_detect_fast_gftt_");

    fast_points = mergeCandidates(
        fast_points,
        fast_gftt,
        4.0f,
        max_corners * 2,
        &timings,
        "corner_detect_fast_merge_ms"
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
        return finish(std::move(fast_result));
    }

    const auto variants_t0 = cv::getTickCount();
    const std::vector<cv::Mat> variants = buildRecoveryVariants(gray, &timings);
    addTiming(
        &timings,
        "corner_detect_build_variants_ms",
        elapsedMs(variants_t0));

    if (variants.empty()) {
        fast_result.points = std::move(fast_points);
        return finish(std::move(fast_result));
    }

    std::vector<cv::Point2f> merged = fast_points;

    const int per_variant_max = std::max(max_corners, 300);

    for (const cv::Mat& v : variants) {
        CornerDetectionResult jr = detectGradientJunctions(
            v,
            per_variant_max,
            sigma,
            response_threshold * 0.65f,
            &timings,
            "corner_detect_variant_gradient_"
        );

        merged = mergeCandidates(
            merged,
            jr.points,
            3.0f,
            max_corners * 3,
            &timings,
            "corner_detect_variant_merge_gradient_ms"
        );

        std::vector<cv::Point2f> gftt = detectFastCandidates(
            v,
            &timings,
            "corner_detect_variant_gftt_");

        merged = mergeCandidates(
            merged,
            gftt,
            4.0f,
            max_corners * 3,
            &timings,
            "corner_detect_variant_merge_gftt_ms"
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
        0.0f,          // threshold=0: suppress all point output, keep grads
        &timings,
        "corner_detect_final_gradient_"
    );

    result.points = std::move(merged);

    return finish(std::move(result));
}

// --------------------------------------------
// Samu: pre_filter
// --------------------------------------------

CornerDetectionResult CornerDetector::detectGradientJunctions(
    const cv::Mat& gray,
    int max_corners,
    float sigma,
    float response_threshold,
    std::unordered_map<std::string, double>* timings,
    const char* timing_prefix
) const {
    const auto total_t0 = cv::getTickCount();

    const auto convert_t0 = cv::getTickCount();
    cv::Mat img;
    gray.convertTo(img, CV_32F);
    addTiming(
        timings,
        timingName(timing_prefix, "convert_ms"),
        elapsedMs(convert_t0));

    cv::Mat gx;
    cv::Mat gy;

    const auto sobel_t0 = cv::getTickCount();
    cv::Sobel(img, gx, CV_32F, 1, 0, 3);
    cv::Sobel(img, gy, CV_32F, 0, 1, 3);
    addTiming(
        timings,
        timingName(timing_prefix, "sobel_ms"),
        elapsedMs(sobel_t0));

    const auto magnitude_t0 = cv::getTickCount();
    cv::Mat magnitude;
    cv::magnitude(gx, gy, magnitude);
    addTiming(
        timings,
        timingName(timing_prefix, "magnitude_ms"),
        elapsedMs(magnitude_t0));

    cv::Mat gpow;
    cv::Mat gx_blur;
    cv::Mat gy_blur;
    cv::Mat gsum;

    const auto blur_t0 = cv::getTickCount();
    if (sigma > 0.0f) {
        cv::GaussianBlur(magnitude, gpow, cv::Size(), sigma);
        cv::GaussianBlur(gx, gx_blur, cv::Size(), sigma);
        cv::GaussianBlur(gy, gy_blur, cv::Size(), sigma);
        cv::magnitude(gx_blur, gy_blur, gsum);
    } else {
        gpow = magnitude;
        gsum = magnitude;
    }
    addTiming(
        timings,
        timingName(timing_prefix, "blur_ms"),
        elapsedMs(blur_t0));

    const auto response_t0 = cv::getTickCount();
    cv::Mat response = gpow - gsum;
    addTiming(
        timings,
        timingName(timing_prefix, "response_ms"),
        elapsedMs(response_t0));

    const auto nms_t0 = cv::getTickCount();
    cv::Mat dilated;
    cv::dilate(response, dilated, cv::Mat());

    response.setTo(0, response != dilated);
    cv::max(response, 0, response);
    addTiming(
        timings,
        timingName(timing_prefix, "nms_ms"),
        elapsedMs(nms_t0));

    const auto threshold_t0 = cv::getTickCount();
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
    addTiming(
        timings,
        timingName(timing_prefix, "threshold_ms"),
        elapsedMs(threshold_t0));

    const auto scan_points_t0 = cv::getTickCount();
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
    addTiming(
        timings,
        timingName(timing_prefix, "scan_points_ms"),
        elapsedMs(scan_points_t0));

    const auto partial_sort_t0 = cv::getTickCount();
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
    addTiming(
        timings,
        timingName(timing_prefix, "partial_sort_ms"),
        elapsedMs(partial_sort_t0));

    CornerDetectionResult result;

    result.points = std::move(points);
    result.response = response;
    result.grad_x = gx;
    result.grad_y = gy;
    addTiming(
        timings,
        timingName(timing_prefix, "total_ms"),
        elapsedMs(total_t0));

    return result;
}

// --------------------------------------------
// FAST / fallback
// --------------------------------------------

std::vector<cv::Point2f> CornerDetector::detectFastCandidates(
    const cv::Mat& gray,
    std::unordered_map<std::string, double>* timings,
    const char* timing_prefix
) const {
    const auto total_t0 = cv::getTickCount();
    std::vector<cv::Point2f> corners;

    const auto to_gray_t0 = cv::getTickCount();
    cv::Mat gray8 = toGray8(gray);
    addTiming(
        timings,
        timingName(timing_prefix, "to_gray_ms"),
        elapsedMs(to_gray_t0));

    if (gray8.empty()) {
        addTiming(
            timings,
            timingName(timing_prefix, "total_ms"),
            elapsedMs(total_t0));
        return corners;
    }

    const auto gftt_t0 = cv::getTickCount();
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
    addTiming(
        timings,
        timingName(timing_prefix, "good_features_ms"),
        elapsedMs(gftt_t0));
    addTiming(
        timings,
        timingName(timing_prefix, "total_ms"),
        elapsedMs(total_t0));

    return corners;
}

// --------------------------------------------
// Merge
// --------------------------------------------

std::vector<cv::Point2f> CornerDetector::mergeCandidates(
    const std::vector<cv::Point2f>& a,
    const std::vector<cv::Point2f>& b,
    float merge_radius,
    int max_points,
    std::unordered_map<std::string, double>* timings,
    const char* timing_name
) const {
    const auto total_t0 = cv::getTickCount();
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

    addTiming(
        timings,
        timing_name ? timing_name : "corner_detect_merge_ms",
        elapsedMs(total_t0));

    return merged;
}

} // namespace hydramarker
