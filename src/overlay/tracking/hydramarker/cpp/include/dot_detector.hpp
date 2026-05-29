#pragma once

#include <array>
#include <map>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "checkerboard_types.hpp"

namespace hydramarker {

struct DotDetectorConfig {
    int canonical_size = 80;
    float canonical_margin_px = 6.0f;

    double min_dot_contrast = 8.0;
    double strong_dot_contrast = 35.0;

    double commit_threshold = 0.45;
    double revoke_threshold = 0.20;

    double uncertainty_low = 0.20;
    double uncertainty_high = 0.45;

    int warmup_frames = 1;

    double temporal_alpha = 0.35;
    int commit_frames = 2;
    int revoke_frames = 3;
};

struct DotCellObservation {
    int row = 0;
    int col = 0;

    bool valid = false;
    bool has_dot = false;
    bool ambiguous = false;

    double score = 0.0;
    double raw_score = 0.0;

    double center_mean = 0.0;
    double ring_mean = 0.0;

    double local_mean = 0.0;
    double local_std = 0.0;

    int polarity = 0;

    cv::Point2f center_uv;

    std::array<cv::Point2f, 4> corners_uv;
};

struct DotDetectionResult {
    int rows = 0;
    int cols = 0;

    std::vector<DotCellObservation> cells;
};

class DotDetector {
public:
    DotDetector();
    explicit DotDetector(DotDetectorConfig config);

    DotDetectionResult detect(
        const cv::Mat& image,
        const CheckerboardDetection& checkerboard
    );

    void reset();

private:
    struct LocalScoreResult {
        double score = 0.0;

        double fg_mean = 0.0;
        double bg_mean = 0.0;

        double local_mean = 0.0;
        double local_std = 0.0;

        double signed_contrast = 0.0;
        double abs_contrast = 0.0;

        int polarity = 0;
    };

    struct TemporalCellState {
        bool initialized = false;
        bool has_dot = false;

        double ema_score = 0.0;

        int seen_frames = 0;
        int commit_count = 0;
        int revoke_count = 0;
        int missed_frames = 0;
    };

    static cv::Mat toGray8(const cv::Mat& image);

    static double sampleBilinearClamp(
        const cv::Mat& gray_f32,
        const cv::Point2f& p
    );

    LocalScoreResult evaluateCell(
        const cv::Mat& gray_f32,
        const GridCell& cell,
        double frame_mean,
        double frame_std
    ) const;

    bool updateTemporalState(
        TemporalCellState& state,
        double raw_score
    ) const;

private:
    DotDetectorConfig config_;
    std::map<std::pair<int, int>, TemporalCellState> temporal_states_;
};

} // namespace hydramarker