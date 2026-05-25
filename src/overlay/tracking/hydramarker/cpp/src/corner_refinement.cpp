#include "corner_refinement.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

namespace hydramarker {

namespace {

constexpr float kPi = 3.14159265358979323846f;

float rad2deg(float x) {
    return x * 180.0f / kPi;
}

float deg2rad(float x) {
    return x * kPi / 180.0f;
}

float angleDiffRad(float a, float b) {
    float d = a - b;
    while (d > kPi) {
        d -= 2.0f * kPi;
    }
    while (d < -kPi) {
        d += 2.0f * kPi;
    }
    return d;
}

float corr1D(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0f;
    }

    float mean_a = 0.0f;
    float mean_b = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
        mean_a += a[i];
        mean_b += b[i];
    }

    mean_a /= static_cast<float>(a.size());
    mean_b /= static_cast<float>(b.size());

    float num = 0.0f;
    float den_a = 0.0f;
    float den_b = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
        const float da = a[i] - mean_a;
        const float db = b[i] - mean_b;

        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    const float den = std::sqrt(den_a * den_b);
    if (den < 1e-12f) {
        return 0.0f;
    }

    return num / den;
}

bool insideWithRadius(const cv::Mat& img, const cv::Point2f& p, int r) {
    return p.x >= r &&
           p.y >= r &&
           p.x < static_cast<float>(img.cols - r) &&
           p.y < static_cast<float>(img.rows - r);
}

} // namespace

// ------------------------------------------------------------

CornerRefiner::CornerRefiner() = default;

// ------------------------------------------------------------

std::vector<RefinedCorner> CornerRefiner::refine(
    const cv::Mat& gray,
    const std::vector<cv::Point2f>& candidates,
    const cv::Mat& grad_x,
    const cv::Mat& grad_y,
    const CornerRefinementConfig& config
) const {
    if (gray.empty() || candidates.empty()) {
        return {};
    }

    std::vector<cv::Point2f> refined_points = refineGradientIntersections(
        candidates,
        grad_x,
        grad_y,
        config.radius,
        config.iterations
    );

    if (refined_points.empty()) {
        return {};
    }

    std::vector<RefinedCorner> featured = computeSaddleFeatures(
        gray,
        refined_points,
        config
    );

    std::vector<RefinedCorner> filtered = filterBySaddleScore(
        featured,
        config
    );

    return mergeCloseCorners(filtered, config.merge_radius_px);
}

// ------------------------------------------------------------
// Samu/ReadMarker-style pt_refine core:
// local gradient-intersection refinement.
// ------------------------------------------------------------

std::vector<cv::Point2f> CornerRefiner::refineGradientIntersections(
    const std::vector<cv::Point2f>& candidates,
    const cv::Mat& grad_x,
    const cv::Mat& grad_y,
    int radius,
    int iterations
) const {
    std::vector<cv::Point2f> points = candidates;

    if (grad_x.empty() || grad_y.empty()) {
        return {};
    }

    CV_Assert(grad_x.size() == grad_y.size());
    CV_Assert(grad_x.type() == CV_32F || grad_x.type() == CV_64F);
    CV_Assert(grad_y.type() == CV_32F || grad_y.type() == CV_64F);

    const int width = grad_x.cols;
    const int height = grad_x.rows;

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<cv::Point2f> next;
        next.reserve(points.size());

        for (const auto& p : points) {
            const int cx = static_cast<int>(std::lround(p.x));
            const int cy = static_cast<int>(std::lround(p.y));

            if (cx < radius || cy < radius ||
                cx >= width - radius || cy >= height - radius) {
                continue;
            }

            double g11 = 0.0;
            double g22 = 0.0;
            double g12 = 0.0;
            double rhs1 = 0.0;
            double rhs2 = 0.0;

            for (int dy = -radius; dy <= radius; ++dy) {
                const int y = cy + dy;

                for (int dx = -radius; dx <= radius; ++dx) {
                    const int x = cx + dx;

                    const double gx =
                        grad_x.type() == CV_32F
                            ? static_cast<double>(grad_x.at<float>(y, x))
                            : grad_x.at<double>(y, x);

                    const double gy =
                        grad_y.type() == CV_32F
                            ? static_cast<double>(grad_y.at<float>(y, x))
                            : grad_y.at<double>(y, x);

                    // Samu notation:
                    // gm = gy, gn = gx
                    const double gm = gy;
                    const double gn = gx;

                    const double p_vec =
                        static_cast<double>(y) * gm +
                        static_cast<double>(x) * gn;

                    g11 += gm * gm;
                    g22 += gn * gn;
                    g12 += gm * gn;

                    rhs1 += gm * p_vec;
                    rhs2 += gn * p_vec;
                }
            }

            const double det = g11 * g22 - g12 * g12;
            if (std::abs(det) < 1e-9) {
                continue;
            }

            const double y_sol = (rhs1 * g22 - g12 * rhs2) / det;
            const double x_sol = (g11 * rhs2 - g12 * rhs1) / det;

            cv::Point2f q(
                static_cast<float>(x_sol),
                static_cast<float>(y_sol)
            );

            if (q.x < radius + 2 || q.y < radius + 2 ||
                q.x > width - radius - 3 ||
                q.y > height - radius - 3) {
                continue;
            }

            next.push_back(q);
        }

        points = std::move(next);

        if (points.empty()) {
            break;
        }
    }

    return points;
}

// ------------------------------------------------------------
// Samu/ReadMarker-style _poly_features:
// fit local quadratic saddle model and validate checker structure.
// ------------------------------------------------------------

std::vector<RefinedCorner> CornerRefiner::computeSaddleFeatures(
    const cv::Mat& gray,
    const std::vector<cv::Point2f>& points,
    const CornerRefinementConfig& config
) const {
    std::vector<RefinedCorner> output;
    output.reserve(points.size());

    cv::Mat gray_f;
    gray.convertTo(gray_f, CV_32F);

    const int r = config.radius;
    const int patch_size = 2 * r + 1;

    // Design matrix:
    // z = c0*u^2 + c1*u*v + c2*v^2 + c3
    cv::Mat A(patch_size * patch_size, 4, CV_32F);

    int row = 0;
    for (int v = -r; v <= r; ++v) {
        for (int u = -r; u <= r; ++u) {
            A.at<float>(row, 0) = static_cast<float>(u * u);
            A.at<float>(row, 1) = static_cast<float>(u * v);
            A.at<float>(row, 2) = static_cast<float>(v * v);
            A.at<float>(row, 3) = 1.0f;
            ++row;
        }
    }

    cv::Mat A_pinv;
    cv::invert(A, A_pinv, cv::DECOMP_SVD);

    for (const auto& p : points) {
        RefinedCorner corner;
        corner.uv = p;

        if (!insideWithRadius(gray_f, p, r)) {
            output.push_back(corner);
            continue;
        }

        cv::Mat patch;
        cv::getRectSubPix(
            gray_f,
            cv::Size(patch_size, patch_size),
            p,
            patch
        );

        if (patch.empty() ||
            patch.rows != patch_size ||
            patch.cols != patch_size) {
            output.push_back(corner);
            continue;
        }

        cv::Mat z(patch_size * patch_size, 1, CV_32F);

        int zi = 0;
        for (int yy = 0; yy < patch_size; ++yy) {
            for (int xx = 0; xx < patch_size; ++xx) {
                z.at<float>(zi++, 0) = patch.at<float>(yy, xx);
            }
        }

        cv::Mat coeff = A_pinv * z;

        const float c0 = coeff.at<float>(0, 0); // u^2
        const float c1 = coeff.at<float>(1, 0); // u*v
        const float c2 = coeff.at<float>(2, 0); // v^2

        // Samu root formulation:
        // a = c2, b = c1, c = c0
        const float a = c2;
        const float b = c1;
        const float c = c0;

        const float discriminant = b * b - 4.0f * a * c;

        if (std::abs(a) < 1e-12f || discriminant < 0.0f) {
            output.push_back(corner);
            continue;
        }

        const float sqrt_disc = std::sqrt(discriminant);
        const float root1 = (-b + sqrt_disc) / (2.0f * a);
        const float root2 = (-b - sqrt_disc) / (2.0f * a);

        const float theta0 = rad2deg(std::atan(root1));
        const float theta1 = rad2deg(std::atan(root2));

        const float angle_diff = rad2deg(
            std::abs(angleDiffRad(deg2rad(theta0), deg2rad(theta1)))
        );

        corner.angle_bias_deg = std::abs(angle_diff - 90.0f);

        // Build sign template from quadratic part.
        std::vector<float> templ;
        std::vector<float> samples;
        templ.reserve(patch_size * patch_size);
        samples.reserve(patch_size * patch_size);

        for (int v = -r; v <= r; ++v) {
            for (int u = -r; u <= r; ++u) {
                float val =
                    c0 * static_cast<float>(u * u) +
                    c1 * static_cast<float>(u * v) +
                    c2 * static_cast<float>(v * v);

                templ.push_back(val >= 0.0f ? 1.0f : -1.0f);

                const int px = u + r;
                const int py = v + r;
                samples.push_back(patch.at<float>(py, px));
            }
        }

        corner.correlation = corr1D(templ, samples);

        const float sign_k = std::copysign(
            1.0f,
            theta0 * theta1 * c
        );

        if (sign_k >= 0.0f) {
            corner.ledge_angles_deg = cv::Vec2f(
                std::max(theta0, theta1),
                std::min(theta0, theta1)
            );
        } else {
            corner.ledge_angles_deg = cv::Vec2f(
                std::min(theta0, theta1),
                std::max(theta0, theta1)
            );
        }

        corner.valid =
            std::isfinite(corner.correlation) &&
            std::isfinite(corner.angle_bias_deg) &&
            corner.angle_bias_deg <= config.max_angle_bias_deg;

        output.push_back(corner);
    }

    return output;
}

// ------------------------------------------------------------

std::vector<RefinedCorner> CornerRefiner::filterBySaddleScore(
    const std::vector<RefinedCorner>& corners,
    const CornerRefinementConfig& config
) const {
    if (corners.empty()) {
        return {};
    }

    float best_corr = -std::numeric_limits<float>::infinity();

    for (const auto& c : corners) {
        if (c.valid && std::isfinite(c.correlation)) {
            best_corr = std::max(best_corr, c.correlation);
        }
    }

    if (!std::isfinite(best_corr)) {
        return {};
    }

    const float min_corr = best_corr - config.correlation_drop;

    std::vector<RefinedCorner> filtered;
    filtered.reserve(corners.size());

    for (const auto& c : corners) {
        if (!c.valid) {
            continue;
        }

        if (c.correlation < min_corr) {
            continue;
        }

        filtered.push_back(c);
    }

    return filtered;
}

// ------------------------------------------------------------

std::vector<RefinedCorner> CornerRefiner::mergeCloseCorners(
    const std::vector<RefinedCorner>& corners,
    float merge_radius_px
) const {
    if (corners.empty()) {
        return {};
    }

    const float r2 = merge_radius_px * merge_radius_px;
    std::vector<char> used(corners.size(), 0);

    std::vector<RefinedCorner> merged;
    merged.reserve(corners.size());

    for (size_t i = 0; i < corners.size(); ++i) {
        if (used[i]) {
            continue;
        }

        cv::Point2f sum_uv(0.0f, 0.0f);
        cv::Vec2f sum_ledge(0.0f, 0.0f);
        float sum_corr = 0.0f;
        float sum_bias = 0.0f;
        int count = 0;

        for (size_t j = i; j < corners.size(); ++j) {
            if (used[j]) {
                continue;
            }

            const float dx = corners[i].uv.x - corners[j].uv.x;
            const float dy = corners[i].uv.y - corners[j].uv.y;

            if (dx * dx + dy * dy <= r2) {
                used[j] = 1;
                sum_uv += corners[j].uv;
                sum_ledge += corners[j].ledge_angles_deg;
                sum_corr += corners[j].correlation;
                sum_bias += corners[j].angle_bias_deg;
                ++count;
            }
        }

        if (count <= 0) {
            continue;
        }

        RefinedCorner c;
        c.uv = sum_uv * (1.0f / static_cast<float>(count));
        c.ledge_angles_deg = sum_ledge * (1.0f / static_cast<float>(count));
        c.correlation = sum_corr / static_cast<float>(count);
        c.angle_bias_deg = sum_bias / static_cast<float>(count);
        c.valid = true;

        merged.push_back(c);
    }

    return merged;
}

} // namespace hydramarker