#pragma once

#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

struct LatticePoint {
    cv::Point2f uv;
    cv::Point2f ij;

    float residual = 0.0f;
    bool valid = false;
};

struct LatticeResult {
    std::vector<LatticePoint> points;

    cv::Point2f axis_u;
    cv::Point2f axis_v;
    cv::Point2f origin;

    float spacing_u = 0.0f;
    float spacing_v = 0.0f;

    bool valid = false;
};

class LatticeModel {
public:
    LatticeModel();

    std::optional<LatticeResult> fit(
        const std::vector<cv::Point2f>& corners
    ) const;

private:
    bool estimateAxes(
        const std::vector<cv::Point2f>& pts,
        cv::Point2f& axis_u,
        cv::Point2f& axis_v,
        float& spacing_u,
        float& spacing_v
    ) const;

    std::vector<LatticePoint> growGrid(
        const std::vector<cv::Point2f>& pts,
        const cv::Point2f& axis_u,
        const cv::Point2f& axis_v,
        float spacing_u,
        float spacing_v
    ) const;

    cv::Point2f estimateOrigin(
        const std::vector<cv::Point2f>& pts
    ) const;
};

} // namespace hydramarker