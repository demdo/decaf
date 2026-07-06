#include "corner_refinement.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace hydramarker {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// --- samplers for the model-warp measurement -------------------------------

// Catmull-Rom bicubic kernel (a = -0.5).
inline float cubicWeight(float x) {
    x = std::abs(x);
    if (x < 1.0f) {
        return ((1.5f * x - 2.5f) * x) * x + 1.0f;
    }
    if (x < 2.0f) {
        return (((-0.5f * x) + 2.5f) * x - 4.0f) * x + 2.0f;
    }
    return 0.0f;
}

inline float sampleBicubic(const cv::Mat& img32, float x, float y) {
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);

    float wx[4];
    float wy[4];
    for (int i = 0; i < 4; ++i) {
        wx[i] = cubicWeight(static_cast<float>(i - 1) - fx);
        wy[i] = cubicWeight(static_cast<float>(i - 1) - fy);
    }

    // Interior fast path: no per-tap clamping.
    if (x0 >= 1 && y0 >= 1 && x0 <= img32.cols - 3 && y0 <= img32.rows - 3) {
        const size_t step = img32.step1();
        const float* row = img32.ptr<float>(y0 - 1) + (x0 - 1);
        float acc = 0.0f;
        for (int j = 0; j < 4; ++j, row += step) {
            acc += wy[j] * (wx[0] * row[0] + wx[1] * row[1] +
                            wx[2] * row[2] + wx[3] * row[3]);
        }
        return acc;
    }

    float acc = 0.0f;
    for (int j = 0; j < 4; ++j) {
        const int yy = std::min(std::max(y0 - 1 + j, 0), img32.rows - 1);
        const float* row = img32.ptr<float>(yy);
        float row_acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const int xx = std::min(std::max(x0 - 1 + i, 0), img32.cols - 1);
            row_acc += wx[i] * row[xx];
        }
        acc += wy[j] * row_acc;
    }
    return acc;
}

// Bilinear lookup of the pixel->normalized-ray table (CV_32FC2).
inline cv::Point2d sampleRayLut(const cv::Mat& lut, double x, double y) {
    const int x0 = std::min(std::max(static_cast<int>(std::floor(x)), 0),
                            lut.cols - 2);
    const int y0 = std::min(std::max(static_cast<int>(std::floor(y)), 0),
                            lut.rows - 2);
    const double fx = std::min(std::max(x - x0, 0.0), 1.0);
    const double fy = std::min(std::max(y - y0, 0.0), 1.0);
    const cv::Vec2f* r0 = lut.ptr<cv::Vec2f>(y0);
    const cv::Vec2f* r1 = lut.ptr<cv::Vec2f>(y0 + 1);
    const cv::Vec2f v00 = r0[x0];
    const cv::Vec2f v01 = r0[x0 + 1];
    const cv::Vec2f v10 = r1[x0];
    const cv::Vec2f v11 = r1[x0 + 1];
    return cv::Point2d(
        (1.0 - fy) * ((1.0 - fx) * v00[0] + fx * v01[0]) +
        fy * ((1.0 - fx) * v10[0] + fx * v11[0]),
        (1.0 - fy) * ((1.0 - fx) * v00[1] + fx * v01[1]) +
        fy * ((1.0 - fx) * v10[1] + fx * v11[1]));
}

inline float sampleBilinear(const cv::Mat& img32, float x, float y) {
    const int x0 = std::min(std::max(static_cast<int>(std::floor(x)), 0),
                            img32.cols - 2);
    const int y0 = std::min(std::max(static_cast<int>(std::floor(y)), 0),
                            img32.rows - 2);
    const float fx = std::min(std::max(x - static_cast<float>(x0), 0.0f), 1.0f);
    const float fy = std::min(std::max(y - static_cast<float>(y0), 0.0f), 1.0f);

    const float* r0 = img32.ptr<float>(y0);
    const float* r1 = img32.ptr<float>(y0 + 1);
    const float v00 = r0[x0];
    const float v01 = r0[x0 + 1];
    const float v10 = r1[x0];
    const float v11 = r1[x0 + 1];
    return (1.0f - fy) * ((1.0f - fx) * v00 + fx * v01) +
           fy * ((1.0f - fx) * v10 + fx * v11);
}

// Nearest-rank percentile; identical to the detector-side helper so the
// tracking_subpix_* statistics stay bit-compatible after the move.
double percentileNearestRank(std::vector<float> values, double q) {
    if (values.empty()) return 0.0;
    q = std::max(0.0, std::min(1.0, q));
    std::sort(values.begin(), values.end());
    const size_t rank = std::max<size_t>(
        1,
        static_cast<size_t>(std::ceil(
            q * static_cast<double>(values.size()))));
    const size_t idx = std::min(
        values.size() - 1,
        rank - 1);
    return static_cast<double>(values[idx]);
}

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

// Gradient-weighted Pearson correlation.
//
// Weights each pixel by its local gradient magnitude so that low-contrast
// regions (e.g. the white background that bleeds into a border-corner patch)
// contribute almost nothing to the correlation score.  This makes the saddle
// model fit robust to partial patches — exactly the situation at the physical
// edge of the checkerboard marker where roughly half the sampling window falls
// outside the patterned area.
//
// Without weighting, a border corner that is a perfect saddle in its inner
// half gets a correlation of ~0.5 because the outer half is flat white and
// pulls the Pearson numerator toward zero.  With weighting the flat half is
// effectively masked and the correlation reflects only the informative pixels.
float weightedCorr1D(
    const std::vector<float>& templ,
    const std::vector<float>& signal,
    const std::vector<float>& weights
) {
    const size_t n = templ.size();
    if (n == 0 || signal.size() != n || weights.size() != n) {
        return 0.0f;
    }

    float w_sum  = 0.0f;
    float wt_sum = 0.0f;
    float ws_sum = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        w_sum  += weights[i];
        wt_sum += weights[i] * templ[i];
        ws_sum += weights[i] * signal[i];
    }

    if (w_sum < 1e-12f) {
        return 0.0f;
    }

    const float mean_t = wt_sum / w_sum;
    const float mean_s = ws_sum / w_sum;

    float num   = 0.0f;
    float den_t = 0.0f;
    float den_s = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        const float dt = templ[i]  - mean_t;
        const float ds = signal[i] - mean_s;
        num   += weights[i] * dt * ds;
        den_t += weights[i] * dt * dt;
        den_s += weights[i] * ds * ds;
    }

    const float den = std::sqrt(den_t * den_s);
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

    // Sub-pixel refinement via cv::cornerSubPix.
    //
    // Applied after gradient-intersection refinement and before saddle-feature
    // computation.  cornerSubPix iterates toward the nearest local gradient
    // minimum, which is more robust to blur than the single-pass least-squares
    // solve in refineGradientIntersections.  The combined approach:
    //   1. Gradient-intersection gives a coarse but topology-aware position.
    //   2. cornerSubPix refines to true sub-pixel accuracy under blur.
    //
    // Window size: auto = max(3, radius-1) so it is large enough to straddle
    // the gradient transition but small enough not to overlap adjacent cells
    // on small markers.  Dead zone = (-1,-1) lets OpenCV choose automatically.
    if (!refined_points.empty()) {
        const int win = config.subpix_win_size > 0
            ? config.subpix_win_size
            : (config.subpix_win_size == 0
                ? 0
                : std::max(3, config.radius - 1));

        if (win > 0) {
            cv::cornerSubPix(
                gray,
                refined_points,
                cv::Size(win, win),
                cv::Size(-1, -1),
                cv::TermCriteria(
                    cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                    config.subpix_max_iters,
                    config.subpix_epsilon
                )
            );
        }
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
// Tracked-corner measurement (tracking snap).
// Moved verbatim from the checkerboard-detector tracking path so that every
// corner measurement of the tracker lives in this component; the
// "model_warp" operator plugs in here without touching the detector again.
// ------------------------------------------------------------

TrackedRefineStats CornerRefiner::refineTrackedCorners(
    const cv::Mat& gray,
    std::vector<cv::Point2f>& points,
    const std::vector<bool>& predicted,
    const CornerRefinementConfig& config,
    const CornerModelContext* model_context
) const {
    TrackedRefineStats stats;

    const bool want_model_warp =
        config.tracked_refine_method == "model_warp";

    if (!want_model_warp && config.tracked_refine_method != "subpix") {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr
                << "[hydramarker] unknown tracked_refine_method '"
                << config.tracked_refine_method
                << "'; falling back to 'subpix'."
                << std::endl;
        }
    }

    const bool subpix_enabled =
        config.subpix_win_size != 0 &&
        config.subpix_max_iters > 0 &&
        config.subpix_epsilon > 0.0;

    stats.enabled = subpix_enabled;

    if (want_model_warp) {
        stats.input_uv.assign(points.begin(), points.end());
    }

    // The subpix snap always runs first: it is the per-point fallback for
    // every corner the model-warp operator cannot measure (missing anchor,
    // grazing angle, failed quality gate).
    if (subpix_enabled && !points.empty() && !gray.empty()) {
        runSubpixSnap(gray, points, predicted, config, stats);
    }

    if (want_model_warp) {
        stats.subpix_uv.assign(points.begin(), points.end());
    }

    if (want_model_warp && !points.empty() && !gray.empty() &&
        model_context != nullptr && model_context->usable(points.size())) {
        runModelWarp(gray, points, predicted, config, *model_context, stats);
    }

    return stats;
}

void CornerRefiner::runSubpixSnap(
    const cv::Mat& gray,
    std::vector<cv::Point2f>& points,
    const std::vector<bool>& predicted,
    const CornerRefinementConfig& config,
    TrackedRefineStats& stats
) const {
    const int configured_win =
        config.subpix_win_size > 0
            ? config.subpix_win_size
            : config.radius - 1;
    const int win = std::max(3, configured_win);
    const float border = static_cast<float>(win + 1);

    std::vector<int> refine_indices;
    std::vector<cv::Point2f> refined;
    refine_indices.reserve(points.size());
    refined.reserve(points.size());

    for (size_t k = 0; k < points.size(); ++k) {
        const bool is_predicted =
            k < predicted.size()
                ? predicted[k]
                : false;
        if (is_predicted) continue;

        const cv::Point2f& uv = points[k];
        if (!std::isfinite(uv.x) || !std::isfinite(uv.y)) continue;
        if (uv.x < border || uv.y < border ||
            uv.x >= static_cast<float>(gray.cols) - border ||
            uv.y >= static_cast<float>(gray.rows) - border) {
            continue;
        }

        refine_indices.push_back(static_cast<int>(k));
        refined.push_back(uv);
    }

    if (refined.empty()) {
        return;
    }

    const std::vector<cv::Point2f> before = refined;
    bool refined_ok = true;
    try {
        cv::cornerSubPix(
            gray,
            refined,
            cv::Size(win, win),
            cv::Size(-1, -1),
            cv::TermCriteria(
                cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                config.subpix_max_iters,
                config.subpix_epsilon));
    } catch (const cv::Exception&) {
        refined_ok = false;
    }

    if (!refined_ok) {
        return;
    }

    std::vector<float> shifts;
    shifts.reserve(refined.size());
    double shift_sum = 0.0;
    double shift_max = 0.0;

    for (size_t r = 0; r < refined.size(); ++r) {
        const cv::Point2f& uv = refined[r];
        if (!std::isfinite(uv.x) ||
            !std::isfinite(uv.y)) {
            continue;
        }
        if (uv.x < 0.0f || uv.y < 0.0f ||
            uv.x >= static_cast<float>(gray.cols) ||
            uv.y >= static_cast<float>(gray.rows)) {
            continue;
        }

        const cv::Point2f d = uv - before[r];
        const float shift = std::sqrt(d.x * d.x + d.y * d.y);
        shifts.push_back(shift);
        shift_sum += static_cast<double>(shift);
        shift_max = std::max(
            shift_max,
            static_cast<double>(shift));
        points[static_cast<size_t>(refine_indices[r])] = uv;
    }

    if (!shifts.empty()) {
        stats.refined_count = static_cast<int>(shifts.size());
        stats.mean_shift_px =
            shift_sum / static_cast<double>(shifts.size());
        stats.p95_shift_px = percentileNearestRank(shifts, 0.95);
        stats.max_shift_px = shift_max;
    }
}

void CornerRefiner::runModelWarp(
    const cv::Mat& gray,
    std::vector<cv::Point2f>& points,
    const std::vector<bool>& predicted,
    const CornerRefinementConfig& config,
    const CornerModelContext& ctx,
    TrackedRefineStats& stats
) const {
    const SurfaceModel& sm = ctx.surface;

    const int half = std::max(4, config.model_warp_half_window);
    const int win = 2 * half + 1;
    const int npx = win * win;
    const double min_valid =
        config.model_warp_min_valid_frac * static_cast<double>(npx);
    const double cos_min =
        std::cos(config.model_warp_max_incidence_deg * CV_PI / 180.0);
    const double max_dev = config.model_warp_max_shift_px;
    constexpr double kMaxSeedOffsetPx = 32.0;

    stats.model_warp_ok.assign(points.size(), 0);
    stats.model_warp_zncc.assign(points.size(), 0.0f);

    const cv::Mat K_mat(ctx.K);
    cv::Mat dist_mat;
    if (!ctx.dist.empty()) {
        dist_mat = cv::Mat(ctx.dist, true).reshape(1, 1);
    }

    cv::Mat rvec_cur;
    cv::Rodrigues(cv::Mat(ctx.R), rvec_cur);
    const cv::Mat tvec_cur = (cv::Mat_<double>(3, 1)
        << ctx.t[0], ctx.t[1], ctx.t[2]);

    // ---- pass 1: batched anchor projection, registration seeds, ROI ----
    // The registration is seeded with the LK/subpix estimate of each point,
    // which already follows the frame-to-frame motion.  The anchor+shift
    // construction stays exact: measurement = anchor projection + registered
    // offset; only the search start and the outlier gate are motion-aware.
    std::vector<int> cand;
    std::vector<cv::Point3d> anchor_obj;
    cand.reserve(points.size());
    anchor_obj.reserve(points.size());
    for (size_t k = 0; k < points.size(); ++k) {
        const bool is_predicted =
            k < predicted.size() ? predicted[k] : false;
        if (is_predicted || !ctx.anchor_valid[k]) continue;
        const cv::Vec3d& a = ctx.anchor_xyz_mm[k];
        cand.push_back(static_cast<int>(k));
        anchor_obj.emplace_back(a[0], a[1], a[2]);
    }
    if (cand.empty()) {
        return;
    }

    std::vector<cv::Point2d> anchor_uv;
    cv::projectPoints(anchor_obj, rvec_cur, tvec_cur, K_mat, dist_mat,
                      anchor_uv);

    struct WarpItem {
        int k;
        cv::Point2d c_uv;
        cv::Point2d seed;
    };
    std::vector<WarpItem> items;
    items.reserve(cand.size());

    double roi_x0 = 1e12, roi_y0 = 1e12;
    double roi_x1 = -1e12, roi_y1 = -1e12;
    for (size_t j = 0; j < cand.size(); ++j) {
        const int k = cand[j];
        const cv::Point2d c_uv = anchor_uv[j];
        if (!(c_uv.x > half + 2 && c_uv.y > half + 2 &&
              c_uv.x < gray.cols - half - 3 &&
              c_uv.y < gray.rows - half - 3)) {
            continue;
        }
        const cv::Point2f& lk = points[static_cast<size_t>(k)];
        if (!std::isfinite(lk.x) || !std::isfinite(lk.y)) continue;
        const cv::Point2d seed(lk.x - c_uv.x, lk.y - c_uv.y);
        if (std::max(std::abs(seed.x), std::abs(seed.y)) >
            kMaxSeedOffsetPx) {
            continue;
        }
        items.push_back({k, c_uv, seed});
        roi_x0 = std::min({roi_x0, c_uv.x, c_uv.x + seed.x});
        roi_y0 = std::min({roi_y0, c_uv.y, c_uv.y + seed.y});
        roi_x1 = std::max({roi_x1, c_uv.x, c_uv.x + seed.x});
        roi_y1 = std::max({roi_y1, c_uv.y, c_uv.y + seed.y});
    }
    if (items.empty()) {
        return;
    }

    // ---- ROI-bounded conversions and gradients (perf: the old full-frame
    // Sobel + float conversion dominated the frame budget) ----
    const int margin = half + static_cast<int>(std::ceil(max_dev)) + 8;
    cv::Rect roi;
    roi.x = std::max(0, static_cast<int>(std::floor(roi_x0)) - margin);
    roi.y = std::max(0, static_cast<int>(std::floor(roi_y0)) - margin);
    roi.width = std::min(
        gray.cols, static_cast<int>(std::ceil(roi_x1)) + margin) - roi.x;
    roi.height = std::min(
        gray.rows, static_cast<int>(std::ceil(roi_y1)) + margin) - roi.y;
    if (roi.width <= 2 || roi.height <= 2) {
        return;
    }

    cv::Mat gray32;
    gray(roi).convertTo(gray32, CV_32F);
    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Sobel(gray32, grad_x, CV_32F, 1, 0, 3, 0.125);
    cv::Sobel(gray32, grad_y, CV_32F, 0, 1, 3, 0.125);
    const float ox = static_cast<float>(roi.x);
    const float oy = static_cast<float>(roi.y);

    cv::Mat ref32;
    if (ctx.ref_gray.type() == CV_32F) {
        ref32 = ctx.ref_gray;
    } else {
        ctx.ref_gray.convertTo(ref32, CV_32F);
    }

    const bool has_lut =
        !ctx.ray_lut.empty() && ctx.ray_lut.type() == CV_32FC2 &&
        ctx.ray_lut.rows == gray.rows && ctx.ray_lut.cols == gray.cols;

    // Manual projection into the reference view (standard OpenCV rational
    // model, up to 8 coefficients); avoids one cv::projectPoints call per
    // corner.  Exotic coefficient layouts fall back to cv::projectPoints.
    double dc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const bool manual_proj = ctx.dist.size() <= 8;
    for (size_t i = 0; i < ctx.dist.size() && i < 8; ++i) {
        dc[i] = ctx.dist[i];
    }
    const double fx = ctx.K(0, 0);
    const double fy = ctx.K(1, 1);
    const double cx0 = ctx.K(0, 2);
    const double cy0 = ctx.K(1, 2);
    const auto project_ref = [&](const cv::Vec3d& X_ref_cam,
                                 double& u, double& v) -> bool {
        if (X_ref_cam[2] < 1e-6) return false;
        const double x = X_ref_cam[0] / X_ref_cam[2];
        const double y = X_ref_cam[1] / X_ref_cam[2];
        const double r2 = x * x + y * y;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double num = 1.0 + dc[0] * r2 + dc[1] * r4 + dc[4] * r6;
        const double den = 1.0 + dc[5] * r2 + dc[6] * r4 + dc[7] * r6;
        if (std::abs(den) < 1e-12) return false;
        const double s = num / den;
        const double xd = x * s + 2.0 * dc[2] * x * y +
                          dc[3] * (r2 + 2.0 * x * x);
        const double yd = y * s + dc[2] * (r2 + 2.0 * y * y) +
                          2.0 * dc[3] * x * y;
        u = fx * xd + cx0;
        v = fy * yd + cy0;
        return true;
    };

    cv::Mat rvec_ref;
    cv::Rodrigues(cv::Mat(ctx.R_ref), rvec_ref);
    const cv::Mat tvec_ref = (cv::Mat_<double>(3, 1)
        << ctx.t_ref[0], ctx.t_ref[1], ctx.t_ref[2]);

    const bool is_plane = sm.type == SurfaceModel::Type::Plane;

    // Surface entities in the current camera frame.
    const cv::Vec3d surf_point_c = ctx.R * sm.point + ctx.t;
    const cv::Vec3d surf_dir_c = ctx.R * sm.dir;

    // Plane normal sign: the printed face looks toward the camera, so pick
    // the sign with negative dot against the view ray to the marker centre.
    cv::Vec3d plane_normal_m = sm.dir;
    if (is_plane && surf_dir_c.dot(surf_point_c) > 0.0) {
        plane_normal_m = -sm.dir;
    }

    // Cylinder angular basis.
    const cv::Vec3d cyl_e2 = sm.dir.cross(sm.radial_ref);

    // Window offset grid, shared by all corners.
    std::vector<float> grid_x(npx);
    std::vector<float> grid_y(npx);
    {
        int idx = 0;
        for (int dy = -half; dy <= half; ++dy) {
            for (int dx = -half; dx <= half; ++dx, ++idx) {
                grid_x[idx] = static_cast<float>(dx);
                grid_y[idx] = static_cast<float>(dy);
            }
        }
    }

    // Corners are fully independent: disjoint writes into points[] and the
    // per-point stats arrays, everything else read-only -> deterministic
    // parallel execution.
    cv::parallel_for_(
        cv::Range(0, static_cast<int>(items.size())),
        [&](const cv::Range& range) {

    // Reusable per-window buffers (one set per worker range).
    std::vector<cv::Point2d> pix(npx);
    std::vector<cv::Vec3d> X_m(npx);
    std::vector<uint8_t> usable(npx);
    std::vector<float> templ(npx);
    std::vector<float> samples(npx);

    for (int item_i = range.start; item_i < range.end; ++item_i) {
        const WarpItem& item = items[static_cast<size_t>(item_i)];
        const size_t k = static_cast<size_t>(item.k);
        const cv::Point2d& c_uv = item.c_uv;

        // Pixel rays for the window.
        std::vector<cv::Point2d> und;
        if (has_lut) {
            und.resize(npx);
            for (int i = 0; i < npx; ++i) {
                und[i] = sampleRayLut(ctx.ray_lut,
                                      c_uv.x + grid_x[i],
                                      c_uv.y + grid_y[i]);
            }
        } else {
            for (int i = 0; i < npx; ++i) {
                pix[i] = cv::Point2d(c_uv.x + grid_x[i],
                                     c_uv.y + grid_y[i]);
            }
            cv::undistortPoints(pix, und, K_mat, dist_mat);
        }

        int n_usable = 0;

        for (int i = 0; i < npx; ++i) {
            usable[i] = 0;

            cv::Vec3d v(und[i].x, und[i].y, 1.0);
            v /= cv::norm(v);

            // Ray-surface intersection in the current camera frame.
            double s = -1.0;
            if (is_plane) {
                const cv::Vec3d n_c = ctx.R * plane_normal_m;
                const double denom = v.dot(n_c);
                if (std::abs(denom) < 1e-9) continue;
                s = n_c.dot(surf_point_c) / denom;
            } else {
                const cv::Vec3d& u = surf_dir_c;
                const cv::Vec3d w = v - v.dot(u) * u;
                const cv::Vec3d qa = -surf_point_c;
                const cv::Vec3d q = qa - qa.dot(u) * u;
                const double A = w.dot(w);
                const double B = 2.0 * w.dot(q);
                const double C = q.dot(q) - sm.radius_mm * sm.radius_mm;
                const double disc = B * B - 4.0 * A * C;
                if (disc <= 0.0 || A < 1e-12) continue;
                s = (-B - std::sqrt(disc)) / (2.0 * A);
            }
            if (s <= 1e-6) continue;

            const cv::Vec3d Xc = s * v;
            const cv::Vec3d Xm = ctx.R.t() * (Xc - ctx.t);

            // Band mask + local surface normal (marker frame).
            cv::Vec3d n_m;
            if (is_plane) {
                const cv::Vec3d w_m = Xm - sm.point;
                const double a_coord = w_m.dot(sm.basis_u);
                const double b_coord = w_m.dot(sm.basis_v);
                if (a_coord < sm.band_a_min || a_coord > sm.band_a_max ||
                    b_coord < sm.band_b_min || b_coord > sm.band_b_max) {
                    continue;
                }
                n_m = plane_normal_m;
            } else {
                const cv::Vec3d w_m = Xm - sm.point;
                const double along = w_m.dot(sm.dir);
                const cv::Vec3d radial = w_m - along * sm.dir;
                const double rn = cv::norm(radial);
                if (rn < 1e-9) continue;
                const double theta = std::atan2(radial.dot(cyl_e2),
                                                radial.dot(sm.radial_ref));
                if (along < sm.band_a_min || along > sm.band_a_max ||
                    theta < sm.band_b_min || theta > sm.band_b_max) {
                    continue;
                }
                n_m = radial / rn;
            }

            // Incidence gates in both views.
            const cv::Vec3d n_cur = ctx.R * n_m;
            const double cos_cur = -n_cur.dot(Xc / cv::norm(Xc));
            if (cos_cur < cos_min) continue;

            const cv::Vec3d X_rc = ctx.R_ref * Xm + ctx.t_ref;
            const cv::Vec3d n_rc = ctx.R_ref * n_m;
            const double cos_ref = -n_rc.dot(X_rc / cv::norm(X_rc));
            if (cos_ref < cos_min) continue;

            X_m[i] = Xm;
            usable[i] = 1;
            ++n_usable;
        }

        if (static_cast<double>(n_usable) < min_valid) continue;

        // Project the usable surface points into the reference view and
        // sample the template.
        int n_mask = 0;
        std::fill(templ.begin(), templ.end(), 0.0f);
        if (manual_proj) {
            for (int i = 0; i < npx; ++i) {
                if (!usable[i]) continue;
                usable[i] = 0;
                const cv::Vec3d X_rc = ctx.R_ref * X_m[i] + ctx.t_ref;
                double qu = 0.0;
                double qv = 0.0;
                if (!project_ref(X_rc, qu, qv)) continue;
                if (qu <= 2.0 || qv <= 2.0 ||
                    qu >= ref32.cols - 3.0 || qv >= ref32.rows - 3.0) {
                    continue;
                }
                templ[i] = sampleBicubic(ref32,
                                         static_cast<float>(qu),
                                         static_cast<float>(qv));
                usable[i] = 1;
                ++n_mask;
            }
        } else {
            std::vector<cv::Point3d> obj;
            std::vector<int> obj_idx;
            obj.reserve(n_usable);
            obj_idx.reserve(n_usable);
            for (int i = 0; i < npx; ++i) {
                if (!usable[i]) continue;
                obj.emplace_back(X_m[i][0], X_m[i][1], X_m[i][2]);
                obj_idx.push_back(i);
            }

            std::vector<cv::Point2d> uv_ref;
            cv::projectPoints(obj, rvec_ref, tvec_ref, K_mat, dist_mat,
                              uv_ref);

            std::fill(usable.begin(), usable.end(), 0);
            for (size_t j = 0; j < uv_ref.size(); ++j) {
                const cv::Point2d& q = uv_ref[j];
                if (q.x <= 2.0 || q.y <= 2.0 ||
                    q.x >= ref32.cols - 3.0 || q.y >= ref32.rows - 3.0) {
                    continue;
                }
                const int i = obj_idx[j];
                templ[i] = sampleBicubic(ref32,
                                         static_cast<float>(q.x),
                                         static_cast<float>(q.y));
                usable[i] = 1;
                ++n_mask;
            }
        }
        if (static_cast<double>(n_mask) < min_valid) continue;

        // Template statistics for the final ZNCC gate.
        double t_sum = 0.0;
        for (int i = 0; i < npx; ++i) {
            if (usable[i]) t_sum += templ[i];
        }
        const double t_mean = t_sum / static_cast<double>(n_mask);
        double t_var = 0.0;
        for (int i = 0; i < npx; ++i) {
            if (usable[i]) {
                const double d = templ[i] - t_mean;
                t_var += d * d;
            }
        }
        if (t_var < 1e-6) continue;

        // Translation-only LK with linear photometric compensation:
        // image(x + p) ~= a * template(x) + b.  Seeded with the LK/subpix
        // estimate so fast motion stays inside the convergence basin.
        //
        // Two-phase sampling (perf): the early iterations only need to walk
        // into the basin and use cheap bilinear image samples; the final
        // iterations and the ZNCC use bicubic samples, so the converged
        // fixed point keeps the low-bias interpolator.
        double px = item.seed.x;
        double py = item.seed.y;
        bool failed = false;
        bool bicubic_phase = false;

        for (int iter = 0; iter < config.model_warp_max_iters; ++iter) {
            cv::Matx44d ATA = cv::Matx44d::zeros();
            cv::Vec4d ATb(0.0, 0.0, 0.0, 0.0);

            for (int i = 0; i < npx; ++i) {
                if (!usable[i]) continue;
                const float sx =
                    static_cast<float>(c_uv.x + px) + grid_x[i] - ox;
                const float sy =
                    static_cast<float>(c_uv.y + py) + grid_y[i] - oy;
                const double gx = sampleBilinear(grad_x, sx, sy);
                const double gy = sampleBilinear(grad_y, sx, sy);
                const double ii = bicubic_phase
                    ? sampleBicubic(gray32, sx, sy)
                    : sampleBilinear(gray32, sx, sy);
                const double tv = templ[i];

                const double row[4] = {gx, gy, tv, 1.0};
                for (int r = 0; r < 4; ++r) {
                    for (int c = r; c < 4; ++c) {
                        ATA(r, c) += row[r] * row[c];
                    }
                    ATb[r] += row[r] * ii;
                }
            }
            for (int r = 1; r < 4; ++r) {
                for (int c = 0; c < r; ++c) {
                    ATA(r, c) = ATA(c, r);
                }
            }

            cv::Vec4d sol;
            if (!cv::solve(ATA, ATb, sol, cv::DECOMP_LU)) {
                failed = true;
                break;
            }
            const double dpx = sol[0];
            const double dpy = sol[1];
            const double a = sol[2];
            if (a < 0.05) {
                failed = true;
                break;
            }
            const double a_safe = std::max(a, 1e-6);
            px -= dpx / a_safe;
            py -= dpy / a_safe;
            if (std::abs(dpx / a_safe) < 1e-3 &&
                std::abs(dpy / a_safe) < 1e-3) {
                if (bicubic_phase) {
                    break;
                }
                bicubic_phase = true;   // refine the fixed point bicubically
                continue;
            }
            if (!bicubic_phase &&
                std::abs(dpx / a_safe) < 0.25 &&
                std::abs(dpy / a_safe) < 0.25) {
                bicubic_phase = true;   // close enough: switch interpolator
            }
            if (std::max(std::abs(px - item.seed.x),
                         std::abs(py - item.seed.y)) > max_dev + 2.0) {
                failed = true;
                break;
            }
        }
        if (failed) continue;

        // Final ZNCC between template and image at the solution.
        double i_sum = 0.0;
        for (int i = 0; i < npx; ++i) {
            if (!usable[i]) continue;
            const float sx =
                static_cast<float>(c_uv.x + px) + grid_x[i] - ox;
            const float sy =
                static_cast<float>(c_uv.y + py) + grid_y[i] - oy;
            samples[i] = sampleBicubic(gray32, sx, sy);
            i_sum += samples[i];
        }
        const double i_mean = i_sum / static_cast<double>(n_mask);
        double num = 0.0;
        double i_var = 0.0;
        for (int i = 0; i < npx; ++i) {
            if (!usable[i]) continue;
            const double dt = templ[i] - t_mean;
            const double di = samples[i] - i_mean;
            num += dt * di;
            i_var += di * di;
        }
        const double den = std::sqrt(t_var * i_var);
        const double zncc = den > 1e-9 ? num / den : 0.0;

        stats.model_warp_zncc[k] = static_cast<float>(zncc);

        if (zncc < config.model_warp_min_zncc) continue;
        if (std::max(std::abs(px - item.seed.x),
                     std::abs(py - item.seed.y)) > max_dev) {
            continue;
        }

        points[k] = cv::Point2f(static_cast<float>(c_uv.x + px),
                                static_cast<float>(c_uv.y + py));
        stats.model_warp_ok[k] = 1;
    }

        });  // cv::parallel_for_

    int ok_count = 0;
    double zncc_sum = 0.0;
    for (size_t k = 0; k < stats.model_warp_ok.size(); ++k) {
        if (stats.model_warp_ok[k]) {
            ++ok_count;
            zncc_sum += stats.model_warp_zncc[k];
        }
    }
    stats.model_warp_count = ok_count;
    stats.model_warp_mean_zncc =
        ok_count > 0 ? zncc_sum / static_cast<double>(ok_count) : 0.0;
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
        // Weights are local gradient magnitudes so that low-contrast regions
        // (e.g. the white background bleeding into a border-corner patch) do
        // not dilute the correlation score.  Border corners are perfect saddles
        // in their inner half — the weighted correlation reflects that, whereas
        // the plain Pearson correlation is pulled toward 0.5 by the flat outer
        // half and falls below the adaptive_drop threshold in filterBySaddleScore.
        std::vector<float> templ;
        std::vector<float> samples;
        std::vector<float> grad_weights;
        templ.reserve(patch_size * patch_size);
        samples.reserve(patch_size * patch_size);
        grad_weights.reserve(patch_size * patch_size);

        // We need the gradient magnitude at each patch pixel.
        // Approximate it with a simple 3x3 Sobel on the patch itself.
        // For pixels on the patch border we fall back to a simpler forward-
        // difference, but those are rare and the weight just keeps them mild.
        cv::Mat patch_gx, patch_gy;
        cv::Sobel(patch, patch_gx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REFLECT);
        cv::Sobel(patch, patch_gy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REFLECT);

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

                const float gx = patch_gx.at<float>(py, px);
                const float gy = patch_gy.at<float>(py, px);
                // Use sqrt of gradient magnitude as weight so that very
                // strong edges don't dominate everything; a sqrt-compression
                // keeps the weighting moderate and numerically stable.
                grad_weights.push_back(std::sqrt(std::sqrt(gx * gx + gy * gy) + 1.0f));
            }
        }

        corner.correlation = weightedCorr1D(templ, samples, grad_weights);

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
    const float min_corr = std::max(best_corr - adaptive_drop, -0.15f);

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