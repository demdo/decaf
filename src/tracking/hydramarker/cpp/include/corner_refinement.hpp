#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

struct RefinedCorner {
    cv::Point2f uv;

    // Samu/ReadMarker-style local checkerboard structure features.
    // Angles are in degrees.
    cv::Vec2f ledge_angles_deg = cv::Vec2f(0.0f, 0.0f);

    float correlation = 0.0f;
    float angle_bias_deg = 0.0f;
    bool valid = false;
};

struct CornerRefinementConfig {
    int radius = 5;
    int iterations = 2;

    float max_angle_bias_deg = 20.0f;
    float correlation_drop = 0.2f;

    float merge_radius_px = 2.0f;

    // Quadrant intensity symmetry filter — see CheckerboardDetectorConfig for
    // full documentation. Set quadrant_half_r = 0 to disable.
    int   quadrant_half_r = 3;
    float quadrant_min_contrast = 12.0f;       // in [0,255]; internally scaled relative to local range
    float quadrant_max_diagonal_diff = 60.0f;  // in [0,255]; relaxed — relative scaling makes it robust

    // cv::cornerSubPix window half-size applied after gradient-intersection
    // refinement and before saddle-feature computation.
    //
    // This extra subpixel step is particularly beneficial under motion blur
    // and defocus: the gradient-intersection solver converges to a rough
    // position, then cornerSubPix iterates to the nearest local gradient
    // minimum with sub-pixel accuracy.
    //
    // -1 (default): automatically set to max(3, radius - 1), which gives a
    //   window large enough to see the gradient transition but small enough
    //   not to cross into adjacent cells on small markers.
    //  0: disabled.
    // >0: explicit half-size in pixels.
    int subpix_win_size = -1;

    // Maximum number of cornerSubPix iterations.
    int subpix_max_iters = 20;

    // cornerSubPix convergence epsilon (pixels).
    double subpix_epsilon = 0.05;
};

class CornerRefiner {
public:
    CornerRefiner();

    std::vector<RefinedCorner> refine(
        const cv::Mat& gray,
        const std::vector<cv::Point2f>& candidates,
        const cv::Mat& grad_x,
        const cv::Mat& grad_y,
        const CornerRefinementConfig& config
    ) const;

private:
    std::vector<cv::Point2f> refineGradientIntersections(
        const std::vector<cv::Point2f>& candidates,
        const cv::Mat& grad_x,
        const cv::Mat& grad_y,
        int radius,
        int iterations
    ) const;

    std::vector<RefinedCorner> computeSaddleFeatures(
        const cv::Mat& gray,
        const std::vector<cv::Point2f>& points,
        const CornerRefinementConfig& config
    ) const;

    std::vector<RefinedCorner> filterBySaddleScore(
        const std::vector<RefinedCorner>& corners,
        const CornerRefinementConfig& config
    ) const;

    std::vector<RefinedCorner> mergeCloseCorners(
        const std::vector<RefinedCorner>& corners,
        float merge_radius_px
    ) const;

    // Returns false if the candidate clearly lacks checkerboard quadrant
    // symmetry (e.g. dot centre, cell interior, edge crossing).
    // Operates directly on the original gray image so the test is independent
    // of the saddle model.
    static bool passesQuadrantSymmetry(
        const cv::Mat& gray_f,
        const cv::Point2f& uv,
        int half_r,
        float min_contrast,
        float max_diagonal_diff
    );
};

} // namespace hydramarker