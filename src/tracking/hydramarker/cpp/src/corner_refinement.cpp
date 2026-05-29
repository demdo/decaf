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

    std::vector<RefinedCorner> merged = mergeCloseCorners(
        filtered,
        config.merge_radius_px
    );

    // Quadrant intensity symmetry filter.
    // Applied last so it operates on the already-merged, saddle-validated set.
    // Skipped when disabled (quadrant_half_r == 0) or image unavailable.
    if (config.quadrant_half_r > 0 && !gray.empty()) {
        cv::Mat gray_f;
        gray.convertTo(gray_f, CV_32F);

        std::vector<RefinedCorner> quad_filtered;
        quad_filtered.reserve(merged.size());

        for (const auto& c : merged) {
            if (passesQuadrantSymmetry(
                    gray_f,
                    c.uv,
                    config.quadrant_half_r,
                    config.quadrant_min_contrast,
                    config.quadrant_max_diagonal_diff)) {
                quad_filtered.push_back(c);
            }
        }

        // Safety: if the quadrant filter rejects everything (e.g. heavily
        // blurred image at low resolution), fall back to the pre-filter set
        // rather than returning empty and losing the detection entirely.
        // The lattice / grid stage will still reject geometric outliers.
        if (!quad_filtered.empty()) {
            return quad_filtered;
        }
    }

    return merged;
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

    // Under low light / high gain noise the quadratic saddle fit can still
    // produce useful junction hypotheses, but the angle bias and correlation
    // become less clean.  The grid reconstruction stage is a stronger global
    // validator than this purely local filter, so this stage should avoid the
    // previous all-or-nothing behaviour.
    const float relaxed_angle_limit = std::max(
        config.max_angle_bias_deg,
        35.0f
    );

    float best_corr = -std::numeric_limits<float>::infinity();

    for (const auto& c : corners) {
        const bool usable =
            std::isfinite(c.correlation) &&
            std::isfinite(c.angle_bias_deg) &&
            c.angle_bias_deg <= relaxed_angle_limit;

        if (usable) {
            best_corr = std::max(best_corr, c.correlation);
        }
    }

    if (!std::isfinite(best_corr)) {
        return {};
    }

    const float adaptive_drop = std::max(config.correlation_drop, 0.35f);
    const float min_corr = best_corr - adaptive_drop;

    std::vector<RefinedCorner> filtered;
    filtered.reserve(corners.size());

    for (const auto& c : corners) {
        const bool usable =
            std::isfinite(c.correlation) &&
            std::isfinite(c.angle_bias_deg) &&
            c.angle_bias_deg <= relaxed_angle_limit;

        if (!usable) {
            continue;
        }

        if (c.correlation < min_corr) {
            continue;
        }

        RefinedCorner out = c;
        out.valid = true;
        filtered.push_back(out);
    }

    // Safety net: do not let a single overconfident local saddle suppress all
    // other candidates in difficult frames.  Keeping a moderate number of the
    // best relaxed candidates gives the downstream lattice/grid gates enough
    // evidence to recover partial checkerboard structure.
    constexpr size_t kMinKeep = 24;
    constexpr size_t kMaxKeep = 180;

    if (filtered.size() < kMinKeep) {
        std::vector<RefinedCorner> relaxed;
        relaxed.reserve(corners.size());

        for (const auto& c : corners) {
            const bool usable =
                std::isfinite(c.correlation) &&
                std::isfinite(c.angle_bias_deg) &&
                c.angle_bias_deg <= relaxed_angle_limit;

            if (!usable) {
                continue;
            }

            RefinedCorner out = c;
            out.valid = true;
            relaxed.push_back(out);
        }

        std::sort(
            relaxed.begin(),
            relaxed.end(),
            [](const RefinedCorner& a, const RefinedCorner& b) {
                const float sa = a.correlation - 0.01f * a.angle_bias_deg;
                const float sb = b.correlation - 0.01f * b.angle_bias_deg;
                return sa > sb;
            }
        );

        if (relaxed.size() > kMaxKeep) {
            relaxed.resize(kMaxKeep);
        }

        return relaxed;
    }

    if (filtered.size() > kMaxKeep) {
        std::sort(
            filtered.begin(),
            filtered.end(),
            [](const RefinedCorner& a, const RefinedCorner& b) {
                const float sa = a.correlation - 0.01f * a.angle_bias_deg;
                const float sb = b.correlation - 0.01f * b.angle_bias_deg;
                return sa > sb;
            }
        );
        filtered.resize(kMaxKeep);
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

// ------------------------------------------------------------
// Quadrant intensity symmetry filter.
//
// A real checkerboard corner has four quadrants with ABAB intensity order
// (where A = bright, B = dark, or vice versa). Concretely:
//
//   Q0 (top-left)     Q1 (top-right)
//   Q2 (bottom-left)  Q3 (bottom-right)
//
// Opposite quadrant pairs (Q0,Q3) and (Q1,Q2) should be similar to each
// other, and adjacent pairs (Q0,Q1), (Q0,Q2), etc. should differ.
//
// For a Dot centre: all four quadrants are similarly dark → min adjacent
//   difference is near zero → fails min_contrast check.
// For a cell interior: all four quadrants are similarly bright → same.
// For a dot edge (half-covered): one quadrant is an outlier and the
//   diagonal-pair difference is large → fails max_diagonal_diff check.
//
// The test uses a small axis-aligned box average in each quadrant to be
// robust against sub-pixel noise.  half_r is the box half-side in pixels.
// ------------------------------------------------------------

// static
bool CornerRefiner::passesQuadrantSymmetry(
    const cv::Mat& gray_f,
    const cv::Point2f& uv,
    int half_r,
    float min_contrast,
    float max_diagonal_diff
) {
    CV_Assert(gray_f.type() == CV_32F);

    if (half_r <= 0) {
        return true;
    }

    const int margin = half_r + 1;

    if (uv.x < static_cast<float>(margin) ||
        uv.y < static_cast<float>(margin) ||
        uv.x >= static_cast<float>(gray_f.cols - margin) ||
        uv.y >= static_cast<float>(gray_f.rows - margin)) {
        return true;  // near border — keep
    }

    const int cx = static_cast<int>(std::lround(uv.x));
    const int cy = static_cast<int>(std::lround(uv.y));

    const int offset = std::max(1, (half_r + 1) / 2);

    auto boxMean = [&](int qcx, int qcy) -> float {
        const int x0 = std::max(0, qcx - half_r / 2);
        const int y0 = std::max(0, qcy - half_r / 2);
        const int x1 = std::min(gray_f.cols - 1, qcx + half_r / 2);
        const int y1 = std::min(gray_f.rows - 1, qcy + half_r / 2);

        if (x1 < x0 || y1 < y0) return 0.0f;

        float sum = 0.0f;
        int   cnt = 0;

        for (int y = y0; y <= y1; ++y) {
            const float* row = gray_f.ptr<float>(y);
            for (int x = x0; x <= x1; ++x) { sum += row[x]; ++cnt; }
        }

        return cnt > 0 ? sum / static_cast<float>(cnt) : 0.0f;
    };

    const float q0 = boxMean(cx - offset, cy - offset); // top-left
    const float q1 = boxMean(cx + offset, cy - offset); // top-right
    const float q2 = boxMean(cx - offset, cy + offset); // bottom-left
    const float q3 = boxMean(cx + offset, cy + offset); // bottom-right

    // --- Local dynamic range ---
    // All thresholds are scaled relative to the local contrast range.
    // This makes the test invariant to global brightness and illumination
    // gradients: an overexposed region with only 30 grey levels of local
    // contrast is treated the same as a well-exposed region with 150 levels.
    const float local_min = std::min({q0, q1, q2, q3});
    const float local_max = std::max({q0, q1, q2, q3});
    const float local_range = local_max - local_min;

    // If the local region is essentially flat (no gradient at all), there is
    // no checkerboard structure here regardless of thresholds.
    // 4.0f is intentionally very low — only rejects truly featureless patches.
    if (local_range < 4.0f) {
        return false;
    }

    // Normalise all quadrant values to [0, 1] within the local range so that
    // subsequent threshold comparisons are scale-invariant.
    const float inv_range = 1.0f / local_range;
    const float n0 = (q0 - local_min) * inv_range;
    const float n1 = (q1 - local_min) * inv_range;
    const float n2 = (q2 - local_min) * inv_range;
    const float n3 = (q3 - local_min) * inv_range;

    // Scale the caller-supplied thresholds from the [0,255] domain to [0,1].
    // min_contrast = 12 → 12/255 ≈ 0.047 relative threshold.
    // max_diagonal_diff = 30 → 30/255 ≈ 0.118 relative threshold.
    const float rel_min_contrast   = min_contrast   / 255.0f;
    const float rel_max_diag_diff  = max_diagonal_diff / 255.0f;

    // --- Diagonal consistency (relative) ---
    if (std::abs(n0 - n3) > rel_max_diag_diff) return false;
    if (std::abs(n1 - n2) > rel_max_diag_diff) return false;

    // --- Checkerboard contrast (relative) ---
    // At least one adjacent axis-pair must show sufficient contrast.
    const float adj_top    = std::abs(n0 - n1);
    const float adj_bottom = std::abs(n2 - n3);
    const float adj_left   = std::abs(n0 - n2);
    const float adj_right  = std::abs(n1 - n3);

    const float max_adj = std::max({adj_top, adj_bottom, adj_left, adj_right});

    if (max_adj < rel_min_contrast) {
        return false;
    }

    return true;
}

} // namespace hydramarker