#include "dot_detector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace hydramarker {

DotDetector::DotDetector()
    : config_()
{
}

DotDetector::DotDetector(DotDetectorConfig config)
    : config_(config)
{
}

DotDetectionResult DotDetector::detect(
    const cv::Mat& image,
    const CheckerboardDetection& checkerboard
)
{
    DotDetectionResult result;

    if (image.empty()) {
        return result;
    }

    cv::Mat gray = toGray8(image);

    cv::Mat gray_f32;
    gray.convertTo(gray_f32, CV_32F);

    cv::Scalar mean_scalar;
    cv::Scalar std_scalar;
    cv::meanStdDev(gray_f32, mean_scalar, std_scalar);

    const double frame_mean = mean_scalar[0];
    const double frame_std = std::max(std_scalar[0], 1.0);

    int max_row = -1;
    int max_col = -1;

    result.cells.reserve(checkerboard.cells.size());

    for (const GridCell& cell : checkerboard.cells) {
        DotCellObservation obs;

        /*
         * IMPORTANT:
         *
         * GridCell uses lattice coordinates:
         *   i = horizontal grid coordinate
         *   j = vertical grid coordinate
         *
         * Dot/Patch grid uses image-style coordinates:
         *   row = vertical coordinate
         *   col = horizontal coordinate
         *
         * Therefore:
         *   obs.row = cell.j
         *   obs.col = cell.i
         */
        obs.row = cell.j;
        obs.col = cell.i;

        obs.center_uv = cell.center_uv;
        obs.corners_uv = cell.corner_uv;

        const LocalScoreResult score = evaluateCell(
            gray_f32,
            cell,
            frame_mean,
            frame_std
        );

        obs.score = score.score;

        obs.center_mean = score.fg_mean;
        obs.ring_mean = score.bg_mean;

        obs.local_mean = score.local_mean;
        obs.local_std = score.local_std;

        obs.polarity = score.polarity;

        /*
         * The cell itself is valid if it came from the checkerboard detector
         * and could be evaluated photometrically.
         *
         * Empty cell != invalid cell.
         */
        obs.valid = true;

        obs.ambiguous =
            score.score >= config_.uncertainty_low &&
            score.score < config_.uncertainty_high;

        obs.has_dot = score.score >= config_.commit_threshold;

        result.cells.push_back(obs);

        max_row = std::max(max_row, obs.row);
        max_col = std::max(max_col, obs.col);
    }

    result.rows = max_row + 1;
    result.cols = max_col + 1;

    return result;
}

cv::Mat DotDetector::toGray8(const cv::Mat& image)
{
    if (image.empty()) {
        return {};
    }

    if (image.type() == CV_8UC1) {
        return image;
    }

    cv::Mat gray;

    if (image.type() == CV_8UC3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }

    if (image.type() == CV_8UC4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
        return gray;
    }

    throw std::runtime_error("DotDetector: unsupported image type.");
}

double DotDetector::sampleBilinearClamp(
    const cv::Mat& gray_f32,
    const cv::Point2f& p
)
{
    const int width = gray_f32.cols;
    const int height = gray_f32.rows;

    float x = std::clamp(p.x, 0.0f, static_cast<float>(width - 1));
    float y = std::clamp(p.y, 0.0f, static_cast<float>(height - 1));

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));

    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);

    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);

    const float v00 = gray_f32.at<float>(y0, x0);
    const float v10 = gray_f32.at<float>(y0, x1);
    const float v01 = gray_f32.at<float>(y1, x0);
    const float v11 = gray_f32.at<float>(y1, x1);

    const float v0 = v00 * (1.0f - dx) + v10 * dx;
    const float v1 = v01 * (1.0f - dx) + v11 * dx;

    return static_cast<double>(v0 * (1.0f - dy) + v1 * dy);
}

DotDetector::LocalScoreResult DotDetector::evaluateCell(
    const cv::Mat& gray_f32,
    const GridCell& cell,
    double frame_mean,
    double frame_std
) const
{
    LocalScoreResult result;

    const cv::Point2f center = cell.center_uv;

    double fg_sum = 0.0;
    double bg_sum = 0.0;

    constexpr int n = 4;

    for (int i = 0; i < n; ++i) {
        const cv::Point2f corner = cell.corner_uv[i];

        /*
         * Samu-style sampling:
         *
         * foreground samples:
         *   closer to the cell center
         *
         * background samples:
         *   closer to the cell corners
         */
        const cv::Point2f fg =
            corner * 0.20f + center * 0.80f;

        const cv::Point2f bg =
            corner * 0.80f + center * 0.20f;

        fg_sum += sampleBilinearClamp(gray_f32, fg);
        bg_sum += sampleBilinearClamp(gray_f32, bg);
    }

    result.fg_mean = fg_sum / static_cast<double>(n);
    result.bg_mean = bg_sum / static_cast<double>(n);

    result.signed_contrast = result.fg_mean - result.bg_mean;
    result.abs_contrast = std::abs(result.signed_contrast);

    result.local_mean = 0.5 * (result.fg_mean + result.bg_mean);
    result.local_std = frame_std;

    if (result.signed_contrast > 0.0) {
        result.polarity = 1;
    }
    else if (result.signed_contrast < 0.0) {
        result.polarity = -1;
    }
    else {
        result.polarity = 0;
    }

    /*
     * Polarity-independent score.
     *
     * 0.0 = no contrast
     * 1.0 = strong dot contrast
     */
    const double denom = std::max(config_.strong_dot_contrast, 1.0);
    result.score = std::clamp(result.abs_contrast / denom, 0.0, 1.0);

    /*
     * Optional hard floor:
     * very weak contrast is considered score 0 but still a valid empty cell.
     */
    if (result.abs_contrast < config_.min_dot_contrast) {
        result.score = 0.0;
    }

    return result;
}

} // namespace hydramarker