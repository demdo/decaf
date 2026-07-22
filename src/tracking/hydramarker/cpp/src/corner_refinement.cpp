#include "corner_refinement.hpp"
#include "parallel_util.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace hydramarker {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Chunked cv::cornerSubPix: OpenCV refines every point independently
// (plain per-point loop internally), so splitting the point set into
// disjoint chunks across threads yields bit-identical positions.
// Exceptions from any chunk propagate to the caller like the single call.
void cornerSubPixChunked(
    const cv::Mat& gray,
    std::vector<cv::Point2f>& points,
    const cv::Size& win,
    const cv::TermCriteria& criteria
) {
    parallelChunks(
        static_cast<int>(points.size()),
        16,
        [&](int begin, int end) {
            std::vector<cv::Point2f> chunk(points.begin() + begin,
                                           points.begin() + end);
            cv::cornerSubPix(gray, chunk, win, cv::Size(-1, -1), criteria);
            std::copy(chunk.begin(), chunk.end(), points.begin() + begin);
        });
}

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
            cornerSubPixChunked(
                gray,
                refined_points,
                cv::Size(win, win),
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

    const bool want_quadratic =
        config.tracked_refine_method == "quadratic_form";

    if (!want_quadratic &&
        config.tracked_refine_method != "subpix") {
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

    if (want_quadratic) {
        stats.input_uv.assign(points.begin(), points.end());
    }

    // The subpix snap always runs first: it is the per-point fallback for
    // every corner the model-based operators cannot measure (missing anchor,
    // grazing angle, failed quality gate).
    if (subpix_enabled && !points.empty() && !gray.empty()) {
        runSubpixSnap(gray, points, predicted, config, stats);
    }

    if (want_quadratic) {
        stats.subpix_uv.assign(points.begin(), points.end());
    }

    if (want_quadratic && !points.empty() && !gray.empty() &&
        model_context != nullptr &&
        model_context->usableQuadratic(points.size())) {
        runQuadraticForm(
            gray, points, predicted, config, *model_context, stats);
        // High-incidence corners: every 1D measurement (subpix and the qf
        // profiles) is geometrically blind under strong foreshortening —
        // those corners get the reference-free 2D saddle registration,
        // overriding the 1D result.
        runSaddleWarp(
            gray, points, predicted, config, *model_context, stats);
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
        cornerSubPixChunked(
            gray,
            refined,
            cv::Size(win, win),
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

// ------------------------------------------------------------
// Quadratic-form corner measurement (Wang et al., IEEE TIM 2022, adapted).
//
// Reference-free: each checkerboard grid line is measured directly in the
// current frame.  Circumferential rows of a cylindrical marker are circles
// in 3D and hence conics in the (undistorted) image; axial columns are
// straight lines.  Edge points are localized with sigmoid profile fits
// along the predicted grid-line normals, the full row/column curves are
// fitted with weighted LS across ALL cells they span, and each corner is
// the closed-form intersection of its row curve with its column line.
// Coded (uniform) cells simply yield no edge contrast and drop out via the
// per-profile contrast gate; the pose prediction only shapes the sampling
// geometry — endpoint anchoring on the measured corners cancels pose error
// to first order, so no reference view and no enrollment state exist.
// ------------------------------------------------------------

namespace {

// Bilinear intensity sample; CV_8U or CV_32F single channel.
inline bool qfSample(const cv::Mat& g, double x, double y, double& out)
{
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    if (x0 < 0 || y0 < 0 || x0 + 1 >= g.cols || y0 + 1 >= g.rows) {
        return false;
    }
    const double fx = x - x0;
    const double fy = y - y0;
    double v00, v01, v10, v11;
    if (g.type() == CV_8UC1) {
        const uchar* r0 = g.ptr<uchar>(y0);
        const uchar* r1 = g.ptr<uchar>(y0 + 1);
        v00 = r0[x0]; v01 = r0[x0 + 1];
        v10 = r1[x0]; v11 = r1[x0 + 1];
    } else if (g.type() == CV_32FC1) {
        const float* r0 = g.ptr<float>(y0);
        const float* r1 = g.ptr<float>(y0 + 1);
        v00 = r0[x0]; v01 = r0[x0 + 1];
        v10 = r1[x0]; v11 = r1[x0 + 1];
    } else {
        return false;
    }
    out = (1.0 - fy) * ((1.0 - fx) * v00 + fx * v01) +
          fy * ((1.0 - fx) * v10 + fx * v11);
    return true;
}

// Bilinear lookup in the pixel->normalized-ray LUT (CV_32FC2).
inline bool qfLutLookup(
    const cv::Mat& lut, double x, double y, cv::Point2d& out)
{
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    if (x0 < 0 || y0 < 0 || x0 + 1 >= lut.cols || y0 + 1 >= lut.rows) {
        return false;
    }
    const double fx = x - x0;
    const double fy = y - y0;
    const cv::Vec2f* r0 = lut.ptr<cv::Vec2f>(y0);
    const cv::Vec2f* r1 = lut.ptr<cv::Vec2f>(y0 + 1);
    const cv::Vec2f v =
        (1.0 - fy) * ((1.0 - fx) * r0[x0] + fx * r0[x0 + 1]) +
        fy * ((1.0 - fx) * r1[x0] + fx * r1[x0 + 1]);
    out = cv::Point2d(v[0], v[1]);
    return true;
}

// Sigmoid edge localization along one profile.  s[] are signed positions
// (px) along the normal, I[] the sampled intensities.  Returns the edge
// offset mu (px, along the normal) and a contrast/fit weight.
bool qfSigmoidFit(
    const double* s, const double* I, int n,
    double min_contrast, double max_rms_n,
    double& mu, double& weight, int* fail_reason = nullptr)
{
    if (n < 7) return false;
    const double lo = (I[0] + I[1] + I[2]) / 3.0;
    const double hi = (I[n - 3] + I[n - 2] + I[n - 1]) / 3.0;
    const double amp = hi - lo;
    if (std::abs(amp) < min_contrast) {
        if (fail_reason) *fail_reason = 1;  // no edge contrast
        return false;
    }

    // 2-parameter GN (mu, sc) with the plateaus FIXED from the endpoint
    // means.  A fitted photometric model (b0/amp via LS, optional glare
    // ramp) was tried against the fb1 glare window and REGRESSED the
    // whole run 0.160 -> 0.36-0.43: the extra freedom trades against the
    // sigmoid position and doubles the mu noise.  Glare episodes are
    // handled at the pose-trust level, not measured away here.
    mu = 0.0;
    double sc = 1.2;

    for (int it = 0; it < 6; ++it) {
        double h00 = 0.0, h01 = 0.0, h11 = 0.0, g0 = 0.0, g1 = 0.0;
        for (int j = 0; j < n; ++j) {
            const double z = (s[j] - mu) / sc;
            const double e = std::exp(-std::max(-30.0, std::min(30.0, z)));
            const double gsig = 1.0 / (1.0 + e);
            const double r = I[j] - (lo + amp * gsig);
            const double dgdz = gsig * (1.0 - gsig);
            const double dmu = -amp * dgdz / sc;
            const double dsc = -amp * dgdz * z / sc;
            h00 += dmu * dmu; h01 += dmu * dsc; h11 += dsc * dsc;
            g0 += dmu * r; g1 += dsc * r;
        }
        const double det = h00 * h11 - h01 * h01;
        if (std::abs(det) < 1e-12) break;
        mu += (h11 * g0 - h01 * g1) / det;
        sc += (-h01 * g0 + h00 * g1) / det;
        // Keep mu inside the sampled span (the window may be asymmetric).
        mu = std::max(s[0] + 0.75, std::min(s[n - 1] - 0.75, mu));
        sc = std::max(0.35, std::min(0.45 * (s[n - 1] - s[0]), sc));
    }

    double ss = 0.0;
    for (int j = 0; j < n; ++j) {
        const double z = (s[j] - mu) / sc;
        const double e = std::exp(-std::max(-30.0, std::min(30.0, z)));
        const double r = I[j] - (lo + amp / (1.0 + e));
        ss += r * r;
    }
    const double rms_n = std::sqrt(ss / n) / std::abs(amp);
    if (rms_n > max_rms_n) return false;
    weight = std::min(std::abs(amp), 80.0) * (1.0 - rms_n);
    return weight > 1e-6;
}

// Weighted total-least-squares line fit: a*x + b*y + c = 0 with a^2+b^2=1.
// Also returns the weighted centroid (chord frame origin for the parabola).
bool qfFitLineW(
    const std::vector<cv::Point2d>& pts, const std::vector<double>& ws,
    cv::Vec3d& line, double& rms, cv::Point2d* centroid = nullptr)
{
    double sw = 0.0, mx = 0.0, my = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        sw += ws[i]; mx += ws[i] * pts[i].x; my += ws[i] * pts[i].y;
    }
    if (sw < 1e-12) return false;
    mx /= sw; my /= sw;
    if (centroid) *centroid = cv::Point2d(mx, my);
    double sxx = 0.0, sxy = 0.0, syy = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        const double dx = pts[i].x - mx;
        const double dy = pts[i].y - my;
        sxx += ws[i] * dx * dx; sxy += ws[i] * dx * dy;
        syy += ws[i] * dy * dy;
    }
    // Normal = eigenvector of the smaller eigenvalue of the 2x2 scatter.
    const double tr = sxx + syy;
    const double dt = std::sqrt(
        std::max(0.0, (sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy));
    const double lmin = 0.5 * (tr - dt);
    double nx = sxy, ny = lmin - sxx;
    double nn = std::sqrt(nx * nx + ny * ny);
    if (nn < 1e-12) {  // axis-aligned scatter
        if (sxx >= syy) { nx = 0.0; ny = 1.0; }
        else { nx = 1.0; ny = 0.0; }
        nn = 1.0;
    }
    nx /= nn; ny /= nn;
    line = cv::Vec3d(nx, ny, -(nx * mx + ny * my));
    double ss = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        const double d = nx * pts[i].x + ny * pts[i].y + line[2];
        ss += ws[i] * d * d;
    }
    rms = std::sqrt(ss / sw);
    return true;
}

// Weighted Halir-Flusser direct ellipse fit; conic = (A,B,C,D,E,F) with
// A x^2 + B xy + C y^2 + D x + E y + F = 0.  rms is the weighted Sampson
// (gradient-normalized algebraic) distance.
bool qfFitConicW(
    const std::vector<cv::Point2d>& pts, const std::vector<double>& ws,
    std::array<double, 6>& conic, double& rms)
{
    if (pts.size() < 6) return false;
    cv::Matx33d S1 = cv::Matx33d::zeros();
    cv::Matx33d S2 = cv::Matx33d::zeros();
    cv::Matx33d S3 = cv::Matx33d::zeros();
    for (size_t i = 0; i < pts.size(); ++i) {
        const double w = ws[i];
        const double x = pts[i].x, y = pts[i].y;
        const cv::Vec3d d1(x * x, x * y, y * y);
        const cv::Vec3d d2(x, y, 1.0);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                S1(r, c) += w * d1[r] * d1[c];
                S2(r, c) += w * d1[r] * d2[c];
                S3(r, c) += w * d2[r] * d2[c];
            }
        }
    }
    cv::Matx33d S3i;
    if (cv::invert(S3, S3i, cv::DECOMP_SVD) == 0.0) return false;
    const cv::Matx33d T = -S3i * S2.t();
    const cv::Matx33d M = S1 + S2 * T;
    // Reduced problem C1^-1 * M with C1 the ellipse constraint matrix.
    cv::Matx33d Mc;
    for (int c = 0; c < 3; ++c) {
        Mc(0, c) = 0.5 * M(2, c);
        Mc(1, c) = -M(1, c);
        Mc(2, c) = 0.5 * M(0, c);
    }
    cv::Mat evals, evecs;
    cv::eigenNonSymmetric(cv::Mat(Mc), evals, evecs);
    cv::Vec3d a1;
    bool found = false;
    for (int r = 0; r < evecs.rows; ++r) {
        const double a = evecs.at<double>(r, 0);
        const double b = evecs.at<double>(r, 1);
        const double c = evecs.at<double>(r, 2);
        if (4.0 * a * c - b * b > 0.0) {
            a1 = cv::Vec3d(a, b, c);
            found = true;
            break;
        }
    }
    if (!found) return false;
    const cv::Vec3d a2 = T * a1;
    conic = {a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]};

    double sw = 0.0, ss = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        const double x = pts[i].x, y = pts[i].y;
        const double q = conic[0] * x * x + conic[1] * x * y +
                         conic[2] * y * y + conic[3] * x +
                         conic[4] * y + conic[5];
        const double gx = 2.0 * conic[0] * x + conic[1] * y + conic[3];
        const double gy = conic[1] * x + 2.0 * conic[2] * y + conic[4];
        const double gn = std::sqrt(gx * gx + gy * gy);
        if (gn < 1e-12) return false;
        const double d = q / gn;
        sw += ws[i]; ss += ws[i] * d * d;
    }
    if (sw < 1e-12) return false;
    rms = std::sqrt(ss / sw);
    return true;
}

// Weighted quadratic y(t) = a + b t + c t^2 in the chord frame (origin m,
// tangent d, normal n).  Numerically stable curved model for the shallow
// arcs where the direct ellipse fit is ill-conditioned.
struct QfParabola {
    cv::Point2d m, d, n;
    double a = 0.0, b = 0.0, c = 0.0;
};

bool qfFitParabolaW(
    const std::vector<cv::Point2d>& pts, const std::vector<double>& ws,
    const cv::Point2d& m, const cv::Vec3d& line,
    QfParabola& par, double& rms)
{
    const cv::Point2d nrm(line[0], line[1]);
    const cv::Point2d dir(-line[1], line[0]);
    // Normal equations for weighted LS on {1, t, t^2}.
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    double b0 = 0, b1 = 0, b2 = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        const double w = ws[i];
        const cv::Point2d r = pts[i] - m;
        const double t = r.x * dir.x + r.y * dir.y;
        const double y = r.x * nrm.x + r.y * nrm.y;
        const double t2 = t * t;
        s0 += w; s1 += w * t; s2 += w * t2;
        s3 += w * t2 * t; s4 += w * t2 * t2;
        b0 += w * y; b1 += w * t * y; b2 += w * t2 * y;
    }
    if (s0 < 1e-12) return false;
    const cv::Matx33d A(s0, s1, s2,
                        s1, s2, s3,
                        s2, s3, s4);
    cv::Vec3d coef;
    if (!cv::solve(A, cv::Vec3d(b0, b1, b2), coef, cv::DECOMP_SVD)) {
        return false;
    }
    par.m = m; par.d = dir; par.n = nrm;
    par.a = coef[0]; par.b = coef[1]; par.c = coef[2];
    double ss = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        const cv::Point2d r = pts[i] - m;
        const double t = r.x * dir.x + r.y * dir.y;
        const double y = r.x * nrm.x + r.y * nrm.y;
        const double e = y - (par.a + par.b * t + par.c * t * t);
        ss += ws[i] * e * e;
    }
    rms = std::sqrt(ss / s0);
    return true;
}

// Parabola x column line: substitute p(t) = m + t d + y(t) n into the line
// and pick the root nearest the seed's chord coordinate.
bool qfIntersectParabolaLine(
    const QfParabola& par, const cv::Vec3d& line,
    const cv::Point2d& seed, cv::Point2d& out)
{
    const double L0 = line[0] * par.m.x + line[1] * par.m.y + line[2];
    const double ld = line[0] * par.d.x + line[1] * par.d.y;
    const double ln = line[0] * par.n.x + line[1] * par.n.y;
    const double qa = par.c * ln;
    const double qb = ld + par.b * ln;
    const double qc = L0 + par.a * ln;
    const cv::Point2d rs = seed - par.m;
    const double t_seed = rs.x * par.d.x + rs.y * par.d.y;

    double t = 0.0;
    if (std::abs(qa) < 1e-14) {
        if (std::abs(qb) < 1e-14) return false;
        t = -qc / qb;
    } else {
        const double disc = qb * qb - 4.0 * qa * qc;
        if (disc < 0.0) return false;
        const double sq = std::sqrt(disc);
        const double t1 = (-qb + sq) / (2.0 * qa);
        const double t2 = (-qb - sq) / (2.0 * qa);
        t = (std::abs(t1 - t_seed) <= std::abs(t2 - t_seed)) ? t1 : t2;
    }
    const double y = par.a + par.b * t + par.c * t * t;
    out = cv::Point2d(par.m.x + t * par.d.x + y * par.n.x,
                      par.m.y + t * par.d.y + y * par.n.y);
    return true;
}

bool qfIntersectLines(
    const cv::Vec3d& l1, const cv::Vec3d& l2, cv::Point2d& out)
{
    const double det = l1[0] * l2[1] - l1[1] * l2[0];
    if (std::abs(det) < 1e-9) return false;  // near-parallel
    out.x = (l1[1] * l2[2] - l1[2] * l2[1]) / det;
    out.y = (l1[2] * l2[0] - l1[0] * l2[2]) / det;
    return true;
}

// Conic x line: substitute the line parametrization into the conic and
// pick the root closest to the seed.
bool qfIntersectConicLine(
    const std::array<double, 6>& q, const cv::Vec3d& line,
    const cv::Point2d& seed, cv::Point2d& out)
{
    // Point on the line closest to the seed + unit direction.
    const double dist = line[0] * seed.x + line[1] * seed.y + line[2];
    const cv::Point2d p(seed.x - dist * line[0], seed.y - dist * line[1]);
    const cv::Point2d d(-line[1], line[0]);

    const double qa = q[0] * d.x * d.x + q[1] * d.x * d.y +
                      q[2] * d.y * d.y;
    const double qb = 2.0 * q[0] * p.x * d.x +
                      q[1] * (p.x * d.y + p.y * d.x) +
                      2.0 * q[2] * p.y * d.y +
                      q[3] * d.x + q[4] * d.y;
    const double qc = q[0] * p.x * p.x + q[1] * p.x * p.y +
                      q[2] * p.y * p.y + q[3] * p.x + q[4] * p.y + q[5];

    double t = 0.0;
    if (std::abs(qa) < 1e-14) {
        if (std::abs(qb) < 1e-14) return false;
        t = -qc / qb;
    } else {
        const double disc = qb * qb - 4.0 * qa * qc;
        if (disc < 0.0) return false;
        const double sq = std::sqrt(disc);
        const double t1 = (-qb + sq) / (2.0 * qa);
        const double t2 = (-qb - sq) / (2.0 * qa);
        t = (std::abs(t1) <= std::abs(t2)) ? t1 : t2;
    }
    out = cv::Point2d(p.x + t * d.x, p.y + t * d.y);
    return true;
}

}  // namespace

void CornerRefiner::runQuadraticForm(
    const cv::Mat& gray,
    std::vector<cv::Point2f>& points,
    const std::vector<bool>& predicted,
    const CornerRefinementConfig& config,
    const CornerModelContext& ctx,
    TrackedRefineStats& stats
) const {
    const size_t n = points.size();
    if (stats.model_warp_ok.size() != n) {
        stats.model_warp_ok.assign(n, 0);
        stats.model_warp_zncc.assign(n, 0.0f);
    }

    const SurfaceModel& sm = ctx.surface;

    // Surface frame: axis direction a, orthonormal radial basis e1/e2
    // (cylinder), or the in-plane basis (plane).
    cv::Vec3d axis = sm.dir;
    const double an = cv::norm(axis);
    if (an < 1e-9) return;
    axis /= an;
    cv::Vec3d e1, e2;
    const bool cyl = sm.type == SurfaceModel::Type::Cylinder;
    if (cyl) {
        e1 = sm.radial_ref - axis * sm.radial_ref.dot(axis);
        const double e1n = cv::norm(e1);
        if (e1n < 1e-9 || sm.radius_mm < 1e-6) return;
        e1 /= e1n;
        e2 = axis.cross(e1);
    } else {
        e1 = sm.basis_u;
        e2 = sm.basis_v;
        const double n1 = cv::norm(e1), n2 = cv::norm(e2);
        if (n1 < 1e-9 || n2 < 1e-9) return;
        e1 /= n1; e2 /= n2;
    }

    // Anchored, measured (non-predicted) corners with surface coordinates:
    // u = axial (plane: basis_u), theta/s = circumferential arc (plane:
    // basis_v).  Rows = constant u (a circle -> image conic on the
    // cylinder), columns = constant s (a straight 3D line).
    struct QfCorner {
        size_t idx;          // index into points
        cv::Point2d seed;    // subpix baseline (current measurement)
        cv::Vec3d X;         // marker-frame anchor (mm)
        double u, theta, s, rmag;
        int row_g = -1, col_g = -1;
    };
    std::vector<QfCorner> cs;
    cs.reserve(n);
    double sin_sum = 0.0, cos_sum = 0.0;
    for (size_t k = 0; k < n; ++k) {
        if (predicted[k] || !ctx.anchor_valid[k]) continue;
        QfCorner c;
        c.idx = k;
        c.seed = cv::Point2d(points[k].x, points[k].y);
        c.X = ctx.anchor_xyz_mm[k];
        const cv::Vec3d rel = c.X - sm.point;
        c.u = rel.dot(axis);
        if (cyl) {
            const cv::Vec3d rv = rel - axis * c.u;
            c.rmag = cv::norm(rv);
            if (c.rmag < 1e-6) continue;
            c.theta = std::atan2(rv.dot(e2), rv.dot(e1));
            sin_sum += std::sin(c.theta);
            cos_sum += std::cos(c.theta);
        } else {
            c.rmag = 0.0;
            c.theta = 0.0;
            c.s = rel.dot(e2);
        }
        cs.push_back(c);
    }
    if (cs.size() < 8) return;

    if (cyl) {
        // Re-center the angular branch cut on the marker band and use the
        // arc length as the circumferential coordinate.
        const double mean_th = std::atan2(sin_sum, cos_sum);
        for (QfCorner& c : cs) {
            c.theta = std::remainder(c.theta - mean_th, 2.0 * CV_PI);
            c.s = sm.radius_mm * c.theta;
        }
    }

    // Grid pitch from the data: median nearest-neighbour anchor distance.
    std::vector<double> nn(cs.size(), 1e18);
    for (size_t i = 0; i < cs.size(); ++i) {
        for (size_t j = i + 1; j < cs.size(); ++j) {
            const double d = cv::norm(cs[i].X - cs[j].X);
            nn[i] = std::min(nn[i], d);
            nn[j] = std::min(nn[j], d);
        }
    }
    std::vector<double> nns = nn;
    std::nth_element(nns.begin(), nns.begin() + nns.size() / 2, nns.end());
    const double pitch = nns[nns.size() / 2];
    if (!(pitch > 0.5) || pitch > 1e17) return;
    const double cluster_tol = 0.45 * pitch;

    // 1D clustering along u (rows) and s (columns): sort, split at gaps.
    const auto cluster = [&](const bool by_u) {
        std::vector<size_t> order(cs.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return (by_u ? cs[a].u : cs[a].s) < (by_u ? cs[b].u : cs[b].s);
        });
        int g = 0;
        double last = by_u ? cs[order[0]].u : cs[order[0]].s;
        for (const size_t i : order) {
            const double v = by_u ? cs[i].u : cs[i].s;
            if (v - last > cluster_tol) ++g;
            if (by_u) cs[i].row_g = g; else cs[i].col_g = g;
            last = v;
        }
        return g + 1;
    };
    const int n_rows = cluster(true);
    const int n_cols = cluster(false);
    if (n_rows < 1 || n_cols < 1) return;

    // ---- Segment collection: adjacent corner pairs inside each group ----
    struct QfSegment {
        size_t ca, cb;       // indices into cs
        bool row;            // row (circumferential) or column (axial)
        int group;
        int pt_begin = 0, pt_count = 0;  // range in the projection batch
    };
    std::vector<QfSegment> segs;
    std::vector<cv::Point3d> proj_pts;   // marker-frame batch
    std::vector<double> proj_fracs;      // fraction per batch point

    const double max_gap = config.qf_max_gap_pitch_ratio * pitch;
    const double min_gap = 0.5 * pitch;

    const auto addSegments = [&](const bool rows) {
        // Bucket corners by group, sort along the segment direction.
        const int ng = rows ? n_rows : n_cols;
        std::vector<std::vector<size_t>> buckets(
            static_cast<size_t>(ng));
        for (size_t i = 0; i < cs.size(); ++i) {
            buckets[static_cast<size_t>(
                rows ? cs[i].row_g : cs[i].col_g)].push_back(i);
        }
        for (int g = 0; g < ng; ++g) {
            std::vector<size_t>& b = buckets[static_cast<size_t>(g)];
            if (b.size() < 2) continue;
            std::sort(b.begin(), b.end(), [&](size_t a, size_t bb) {
                return (rows ? cs[a].s : cs[a].u) <
                       (rows ? cs[bb].s : cs[bb].u);
            });
            for (size_t i = 0; i + 1 < b.size(); ++i) {
                const QfCorner& c1 = cs[b[i]];
                const QfCorner& c2 = cs[b[i + 1]];
                const double gap = rows ? (c2.s - c1.s) : (c2.u - c1.u);
                if (gap < min_gap || gap > max_gap) continue;
                QfSegment seg;
                seg.ca = b[i];
                seg.cb = b[i + 1];
                seg.row = rows;
                seg.group = g;
                segs.push_back(seg);
            }
        }
    };
    addSegments(true);
    addSegments(false);
    if (segs.empty()) return;

    // Marker-frame point on the grid line between the two segment corners.
    // Rows on the cylinder follow the surface arc (helix segment when the
    // print is slightly skewed); everything else is a straight 3D segment.
    const auto surfacePoint = [&](const QfSegment& seg, double f) {
        const QfCorner& c1 = cs[seg.ca];
        const QfCorner& c2 = cs[seg.cb];
        if (cyl && seg.row) {
            const double u = c1.u + f * (c2.u - c1.u);
            const double th = c1.theta + f * (c2.theta - c1.theta);
            const double rm = c1.rmag + f * (c2.rmag - c1.rmag);
            return cv::Point3d(
                sm.point + axis * u +
                rm * (std::cos(th) * e1 + std::sin(th) * e2));
        }
        const cv::Vec3d X = c1.X + f * (c2.X - c1.X);
        return cv::Point3d(X);
    };

    // Build the projection batch: per segment the two endpoints (f = 0, 1)
    // followed by interior samples (junction margin applied after the
    // segment length is known in pixels -> generous fixed fractions here,
    // trimmed below).  9 interior samples max keeps the batch small.
    constexpr int kMaxInterior = 9;
    // Neighbouring parallel grid lines, one pitch to either side of the
    // segment midpoint: their projected image distance is the TRANSVERSE
    // CLEARANCE the edge profiles must fit into.  Under strong
    // foreshortening (tool axis toward the camera) this clearance shrinks
    // to a few pixels and a fixed-length profile crosses the neighbour
    // line — the sigmoid then locks onto an arbitrary transition (fb1
    // steep-touch window: reproj 0.56, MAP gate rejects, RGB drifts -7mm;
    // model_warp survives via its 72-deg incidence cull).
    const auto neighborPoint = [&](const QfSegment& seg, double sign) {
        const cv::Point3d mid = surfacePoint(seg, 0.5);
        if (seg.row) {
            // rows: neighbour rows sit one pitch along the axis
            return cv::Point3d(cv::Vec3d(mid) + axis * (sign * pitch));
        }
        if (cyl) {
            // columns: neighbour columns one pitch along the arc
            const cv::Vec3d rel = cv::Vec3d(mid) - sm.point;
            const double u = rel.dot(axis);
            const cv::Vec3d rv = rel - axis * u;
            const double rm = cv::norm(rv);
            if (rm < 1e-9 || sm.radius_mm < 1e-6) return mid;
            const double th0 = std::atan2(rv.dot(e2), rv.dot(e1));
            const double th = th0 + sign * pitch / sm.radius_mm;
            return cv::Point3d(
                sm.point + axis * u +
                rm * (std::cos(th) * e1 + std::sin(th) * e2));
        }
        return cv::Point3d(cv::Vec3d(mid) + e2 * (sign * pitch));
    };

    for (QfSegment& seg : segs) {
        seg.pt_begin = static_cast<int>(proj_pts.size());
        proj_pts.push_back(surfacePoint(seg, 0.0));
        proj_fracs.push_back(0.0);
        proj_pts.push_back(surfacePoint(seg, 1.0));
        proj_fracs.push_back(1.0);
        for (int i = 0; i < kMaxInterior; ++i) {
            const double f =
                (static_cast<double>(i) + 0.5) / kMaxInterior;
            proj_pts.push_back(surfacePoint(seg, f));
            proj_fracs.push_back(f);
        }
        proj_pts.push_back(neighborPoint(seg, +1.0));
        proj_fracs.push_back(0.5);
        proj_pts.push_back(neighborPoint(seg, -1.0));
        proj_fracs.push_back(0.5);
        seg.pt_count = 2 + kMaxInterior + 2;
    }

    // One projectPoints call for the whole frame.
    cv::Mat rvec;
    cv::Rodrigues(cv::Mat(ctx.R), rvec);
    const cv::Vec3d& tv = ctx.t;
    cv::Mat dist_mat;
    if (!ctx.dist.empty()) {
        dist_mat = cv::Mat(ctx.dist, true).reshape(1, 1);
    }
    std::vector<cv::Point2d> proj_uv;
    cv::projectPoints(
        proj_pts, rvec, cv::Mat(tv), cv::Mat(ctx.K), dist_mat, proj_uv);

    // ---- Edge localization ----
    // Each accepted profile yields the SIGNED deviation mu of the measured
    // edge from the model-predicted (endpoint-anchored) grid line, along
    // the local curve normal.  The curve SHAPE comes entirely from the
    // surface model + pose (free-form conic fits hallucinate curvature
    // from the alternating-polarity edge noise on near-straight views and
    // bias toward the chord on curved views); the data measures ONE mean
    // offset per cell-edge SEGMENT.  Per-segment locality matters: the
    // residual seed bias varies along a grid line (incidence grows toward
    // the cylinder limb), which a global per-line fit cannot follow.  Each
    // corner is then solved from its adjacent segments — and because the
    // two row (or column) segments meeting at a corner have OPPOSITE edge
    // polarity, their equal-weight mean also cancels the polarity-
    // dependent localization bias (blooming shifts toward the dark side).
    std::vector<double> seg_sw(segs.size(), 0.0);   // sum of weights
    std::vector<double> seg_smu(segs.size(), 0.0);  // sum w*mu
    std::vector<double> seg_smu2(segs.size(), 0.0); // sum w*mu^2
    std::vector<double> seg_sf(segs.size(), 0.0);   // sum w*f
    std::vector<double> seg_sf2(segs.size(), 0.0);  // sum w*f^2
    std::vector<double> seg_sfmu(segs.size(), 0.0); // sum w*f*mu
    std::vector<int> seg_np(segs.size(), 0);        // accepted profiles
    int dbg_edges = 0;

    // Per-corner local curve tangents (image space, increasing-t direction)
    // for the final 2x2 corner solve.
    std::vector<cv::Point2d> tan_row(cs.size(), cv::Point2d(0.0, 0.0));
    std::vector<cv::Point2d> tan_col(cs.size(), cv::Point2d(0.0, 0.0));
    std::vector<int> segs_per_row(static_cast<size_t>(n_rows), 0);
    std::vector<int> segs_per_col(static_cast<size_t>(n_cols), 0);

    // Optional diagnostics (env HYDRA_QF_DEBUG): where do profiles die?
    static const bool qf_debug = std::getenv("HYDRA_QF_DEBUG") != nullptr;
    int dbg_seg_corr = 0, dbg_seg_short = 0, dbg_prof_tried = 0;
    int dbg_prof_border = 0, dbg_prof_contrast = 0, dbg_prof_fit = 0;

    // Profile length adapts UP to qf_profile_half_max_px when the local
    // clearance allows: close to the camera the RGB is DEFOCUSED (fixed
    // focus) and a sigma~3-4 px transition never reaches its plateaus
    // inside a 4.5 px window (the fb1 near-camera window: subpix AND the
    // capped profiles drift mm-scale).  Cells are large there, so the
    // longer window is geometrically free.  Sample density stays <=0.9 px.
    constexpr double kProfileHalfMax = 9.0;
    constexpr int kMaxProf = 25;
    std::vector<double> prof_s(static_cast<size_t>(kMaxProf));
    std::vector<double> prof_i(static_cast<size_t>(kMaxProf));

    for (size_t si = 0; si < segs.size(); ++si) {
        const QfSegment& seg = segs[si];
        const cv::Point2d& uvm0 = proj_uv[static_cast<size_t>(seg.pt_begin)];
        const cv::Point2d& uvm1 =
            proj_uv[static_cast<size_t>(seg.pt_begin) + 1];
        // Endpoint anchoring: shift the predicted curve onto the measured
        // corner positions (linear blend cancels pose error to 1st order).
        const cv::Point2d d0 = cs[seg.ca].seed - uvm0;
        const cv::Point2d d1 = cs[seg.cb].seed - uvm1;
        // (A differential |d1-d0| gate was tried (v10) and culled healthy
        // segments during legitimate rotation lag — fb_rl2 0.603 -> 0.955;
        // with the per-frame correspondence anchors mis-anchors are rare
        // and the corner-level consistency gate already covers them.)
        if (cv::norm(d0) > config.qf_max_endpoint_corr_px ||
            cv::norm(d1) > config.qf_max_endpoint_corr_px) {
            ++dbg_seg_corr;
            continue;
        }

        // Corrected interior samples + segment length in pixels.
        std::array<cv::Point2d, kMaxInterior> suv;
        std::array<double, kMaxInterior> sfrac;
        for (int i = 0; i < kMaxInterior; ++i) {
            const size_t bi =
                static_cast<size_t>(seg.pt_begin) + 2 +
                static_cast<size_t>(i);
            const double f = proj_fracs[bi];
            suv[static_cast<size_t>(i)] =
                proj_uv[bi] + (1.0 - f) * d0 + f * d1;
            sfrac[static_cast<size_t>(i)] = f;
        }
        const double seg_len =
            cv::norm(cs[seg.cb].seed - cs[seg.ca].seed);
        if (seg_len < 6.0) { ++dbg_seg_short; continue; }

        const double margin = std::max(
            config.qf_junction_margin_min_px,
            config.qf_junction_margin_frac * seg_len);
        const double f_lo = margin / seg_len;
        const double f_hi = 1.0 - f_lo;
        if (f_hi - f_lo < 0.15) { ++dbg_seg_short; continue; }

        // Transverse clearance to the neighbouring parallel grid lines
        // (raw projections; the anchor shift cancels in the difference to
        // first order and clearance is a scale, not a position).  A
        // per-side ASYMMETRIC window was tried (v9) and regressed
        // divot/fb_rl2 — the shortened plateaus raise noise more than the
        // asymmetry saves bias; symmetric min-clearance stays.
        const cv::Point2d& uv_mid_raw =
            proj_uv[static_cast<size_t>(seg.pt_begin) + 2 + 4];
        const double clear_px = std::min(
            cv::norm(proj_uv[static_cast<size_t>(seg.pt_begin) + 2 +
                             kMaxInterior] - uv_mid_raw),
            cv::norm(proj_uv[static_cast<size_t>(seg.pt_begin) + 2 +
                             kMaxInterior + 1] - uv_mid_raw));

        const double half = std::min(
            {kProfileHalfMax, 0.4 * seg_len, 0.45 * clear_px});
        if (half < 1.5) { ++dbg_seg_short; continue; }
        int n_prof = 1 + 2 * static_cast<int>(std::ceil(half / 0.75));
        n_prof = std::max(7, std::min(kMaxProf, n_prof));
        const double step =
            2.0 * half / static_cast<double>(n_prof - 1);
        for (int j = 0; j < n_prof; ++j) {
            prof_s[static_cast<size_t>(j)] = -half + j * step;
        }

        // Local curve tangents at the two segment corners (increasing-t
        // direction; used by the corner solve) + segment bookkeeping.
        {
            const cv::Point2d ta = suv[0] - cs[seg.ca].seed;
            const cv::Point2d tb =
                cs[seg.cb].seed - suv[kMaxInterior - 1];
            const double na = cv::norm(ta), nb = cv::norm(tb);
            if (na > 1e-9) {
                (seg.row ? tan_row : tan_col)[seg.ca] += ta * (1.0 / na);
            }
            if (nb > 1e-9) {
                (seg.row ? tan_row : tan_col)[seg.cb] += tb * (1.0 / nb);
            }
            if (seg.row) {
                ++segs_per_row[static_cast<size_t>(seg.group)];
            } else {
                ++segs_per_col[static_cast<size_t>(seg.group)];
            }
        }


        for (int i = 0; i < kMaxInterior; ++i) {
            const double f = sfrac[static_cast<size_t>(i)];
            if (f < f_lo || f > f_hi) continue;
            const cv::Point2d& c = suv[static_cast<size_t>(i)];
            // Tangent from the corrected neighbours (one-sided at ends).
            const int ip = std::min(i + 1, kMaxInterior - 1);
            const int im = std::max(i - 1, 0);
            cv::Point2d tan =
                suv[static_cast<size_t>(ip)] - suv[static_cast<size_t>(im)];
            const double tn = cv::norm(tan);
            if (tn < 1e-9) continue;
            tan *= 1.0 / tn;
            const cv::Point2d nrm(-tan.y, tan.x);

            ++dbg_prof_tried;
            bool ok = true;
            int sat_mid = 0;
            const int j_mid_lo = n_prof / 2 - 2;
            const int j_mid_hi = n_prof / 2 + 2;
            for (int j = 0; j < n_prof; ++j) {
                const double sj = prof_s[static_cast<size_t>(j)];
                double& iv = prof_i[static_cast<size_t>(j)];
                if (!qfSample(gray, c.x + sj * nrm.x, c.y + sj * nrm.y,
                              iv)) {
                    ok = false;
                    break;
                }
                // Specular highlight ACROSS the transition region: the
                // sigmoid then locks onto the highlight boundary instead
                // of the printed edge.  Only the central samples count —
                // clipped bright cells at the profile ENDS are legitimate
                // (overexposed white paper) and must not kill the edge
                // (v6 regression: corner counts collapsed 45->23).
                if (j >= j_mid_lo && j <= j_mid_hi && iv >= 250.0) {
                    ++sat_mid;
                }
            }
            if (!ok) { ++dbg_prof_border; continue; }
            if (sat_mid >= 4) { ++dbg_prof_contrast; continue; }
            double mu = 0.0, w = 0.0;
            int fail = 0;
            if (!qfSigmoidFit(prof_s.data(), prof_i.data(), n_prof,
                              config.qf_min_contrast,
                              config.qf_max_profile_rms, mu, w, &fail)) {
                if (fail == 1) ++dbg_prof_contrast; else ++dbg_prof_fit;
                continue;
            }
            // The anchored prediction is sub-pixel accurate; a transition
            // further out is some OTHER edge (specular boundary, dirt).
            if (std::abs(mu) > 2.0) { ++dbg_prof_fit; continue; }
            seg_sw[si] += w;
            seg_smu[si] += w * mu;
            seg_smu2[si] += w * mu * mu;
            seg_sf[si] += w * f;
            seg_sf2[si] += w * f * f;
            seg_sfmu[si] += w * f * mu;
            ++seg_np[si];
            ++dbg_edges;
        }
    }
    if (dbg_edges == 0) return;

    // ---- Per-segment validity + per-corner accumulation ----
    // A segment is valid when enough profiles survived and their scatter
    // about the segment mean is small.  Each valid segment contributes its
    // MEAN offset to both endpoint corners; row/col contributions at a
    // corner are averaged with EQUAL weight (polarity cancellation).
    const int min_seg_profiles =
        std::max(2, config.qf_min_col_points);
    constexpr double kMaxDirSpreadPx = 1.5;
    std::vector<double> corner_dr(cs.size(), 0.0);
    std::vector<double> corner_dc(cs.size(), 0.0);
    std::vector<int> corner_nr(cs.size(), 0);
    std::vector<int> corner_nc(cs.size(), 0);
    std::vector<double> corner_rms(cs.size(), 0.0);
    std::vector<double> corner_dr_lo(cs.size(), 1e9);
    std::vector<double> corner_dr_hi(cs.size(), -1e9);
    std::vector<double> corner_dc_lo(cs.size(), 1e9);
    std::vector<double> corner_dc_hi(cs.size(), -1e9);
    int valid_segs = 0;
    double prof_sum = 0.0;

    for (size_t si = 0; si < segs.size(); ++si) {
        if (seg_np[si] < min_seg_profiles || seg_sw[si] < 1e-9) continue;
        const double s0 = seg_sw[si];
        const double mean = seg_smu[si] / s0;

        // Per-segment offset+tilt in the fraction coordinate f, evaluated
        // at the segment ENDS: the correction varies along a grid line
        // (incidence grows toward the cylinder limb), so applying the
        // mid-segment mean at the corner leaves a first-order residual.
        // With too few profiles or degenerate f spread fall back to the
        // mean at both ends.
        double d_a = mean, d_b = mean, rms;
        {
            const double det = s0 * seg_sf2[si] - seg_sf[si] * seg_sf[si];
            double alpha = mean, beta = 0.0;
            if (seg_np[si] >= 4 && det > 1e-6 * s0) {
                alpha = (seg_sf2[si] * seg_smu[si] -
                         seg_sf[si] * seg_sfmu[si]) / det;
                beta = (s0 * seg_sfmu[si] -
                        seg_sf[si] * seg_smu[si]) / det;
                d_a = alpha;
                d_b = alpha + beta;
            }
            const double ss = std::max(
                0.0,
                seg_smu2[si] - 2.0 * alpha * seg_smu[si] -
                    2.0 * beta * seg_sfmu[si] + alpha * alpha * s0 +
                    2.0 * alpha * beta * seg_sf[si] +
                    beta * beta * seg_sf2[si]);
            rms = std::sqrt(ss / s0);
        }
        if (rms > config.qf_max_fit_rms_px) continue;
        ++valid_segs;
        prof_sum += seg_np[si];
        const QfSegment& seg = segs[si];
        for (int end = 0; end < 2; ++end) {
            const size_t ci = end == 0 ? seg.ca : seg.cb;
            const double d = end == 0 ? d_a : d_b;
            if (seg.row) {
                corner_dr[ci] += d;
                ++corner_nr[ci];
                corner_dr_lo[ci] = std::min(corner_dr_lo[ci], d);
                corner_dr_hi[ci] = std::max(corner_dr_hi[ci], d);
            } else {
                corner_dc[ci] += d;
                ++corner_nc[ci];
                corner_dc_lo[ci] = std::min(corner_dc_lo[ci], d);
                corner_dc_hi[ci] = std::max(corner_dc_hi[ci], d);
            }
            corner_rms[ci] += rms;
        }
    }

    if (qf_debug) {
        std::cerr << "[qf] corners " << cs.size() << " segs " << segs.size()
                  << " valid " << valid_segs
                  << " (corr " << dbg_seg_corr << " short " << dbg_seg_short
                  << ") prof " << dbg_prof_tried
                  << " border " << dbg_prof_border
                  << " contrast " << dbg_prof_contrast
                  << " fitrej " << dbg_prof_fit
                  << " edges " << dbg_edges << std::endl;
    }

    // ---- Corner solve: row correction x column correction ----
    // The measured row curve passes the corner displaced by d_r along the
    // local row normal; the column line by d_c along the column normal.
    // The corner is the point satisfying both constraints: a 2x2 solve in
    // the (n_r, n_c) basis, applied to the subpix seed.
    int accepted = 0;
    double dev_sum = 0.0;
    for (size_t i = 0; i < cs.size(); ++i) {
        const QfCorner& c = cs[i];
        if (corner_nr[i] < 1 || corner_nc[i] < 1) continue;
        // Consistency: the segments meeting at this corner measure the
        // SAME physical grid lines — a large disagreement means one of
        // them locked onto a foreign transition (specular boundary).
        if (corner_dr_hi[i] - corner_dr_lo[i] > kMaxDirSpreadPx ||
            corner_dc_hi[i] - corner_dc_lo[i] > kMaxDirSpreadPx) {
            continue;
        }
        cv::Point2d tr = tan_row[i];
        cv::Point2d tc = tan_col[i];
        const double nr = cv::norm(tr);
        const double ncn = cv::norm(tc);
        if (nr < 1e-9 || ncn < 1e-9) continue;
        tr *= 1.0 / nr;
        tc *= 1.0 / ncn;
        const cv::Point2d n_r(-tr.y, tr.x);
        const cv::Point2d n_c(-tc.y, tc.x);
        const double d_r = corner_dr[i] / corner_nr[i];
        const double d_c = corner_dc[i] / corner_nc[i];
        const double det = n_r.x * n_c.y - n_r.y * n_c.x;
        if (std::abs(det) < 0.30) continue;  // grazing grid geometry
        const double dx = (n_c.y * d_r - n_r.y * d_c) / det;
        const double dy = (-n_c.x * d_r + n_r.x * d_c) / det;
        const double dev = std::hypot(dx, dy);
        if (!(dev <= config.qf_max_dev_px)) continue;
        points[c.idx] = cv::Point2f(
            static_cast<float>(c.seed.x + dx),
            static_cast<float>(c.seed.y + dy));
        const double rms_avg =
            corner_rms[i] / (corner_nr[i] + corner_nc[i]);
        stats.model_warp_ok[c.idx] = 1;
        stats.model_warp_zncc[c.idx] =
            static_cast<float>(1.0 / (1.0 + rms_avg));
        ++accepted;
        dev_sum += dev;
    }

    stats.qf_count = accepted;
    stats.qf_mean_dev_px =
        accepted > 0 ? dev_sum / static_cast<double>(accepted) : 0.0;
    stats.qf_row_curves = valid_segs;
    stats.qf_row_conics = valid_segs;
    stats.qf_col_lines = valid_segs;
    stats.qf_mean_edge_pts = valid_segs > 0
        ? prof_sum / static_cast<double>(valid_segs) : 0.0;
}

void CornerRefiner::runSaddleWarp(
    const cv::Mat& gray,
    std::vector<cv::Point2f>& points,
    const std::vector<bool>& predicted,
    const CornerRefinementConfig& config,
    const CornerModelContext& ctx,
    TrackedRefineStats& stats
) const {
    if (config.qf_saddle_min_incidence_deg <= 0.0) return;
    const SurfaceModel& sm = ctx.surface;
    const bool is_plane = sm.type == SurfaceModel::Type::Plane;
    const double cos_engage =
        std::cos(config.qf_saddle_min_incidence_deg * CV_PI / 180.0);

    if (stats.model_warp_ok.size() != points.size()) {
        stats.model_warp_ok.assign(points.size(), 0);
        stats.model_warp_zncc.assign(points.size(), 0.0f);
    }

    const cv::Vec3d cyl_e2 = sm.dir.cross(sm.radial_ref);

    // Grid pitch (mm) from the anchors — bounds the synthetic template to
    // the four cells adjacent to the corner.
    std::vector<size_t> anchored;
    for (size_t k = 0; k < points.size(); ++k) {
        const bool is_pred = k < predicted.size() ? predicted[k] : false;
        if (!is_pred && ctx.anchor_valid[k]) anchored.push_back(k);
    }
    if (anchored.size() < 2) return;
    std::vector<double> nn(anchored.size(), 1e18);
    for (size_t a = 0; a < anchored.size(); ++a) {
        for (size_t b = a + 1; b < anchored.size(); ++b) {
            const double d = cv::norm(ctx.anchor_xyz_mm[anchored[a]] -
                                      ctx.anchor_xyz_mm[anchored[b]]);
            nn[a] = std::min(nn[a], d);
            nn[b] = std::min(nn[b], d);
        }
    }
    std::nth_element(nn.begin(), nn.begin() + nn.size() / 2, nn.end());
    const double pitch = nn[nn.size() / 2];
    if (!(pitch > 0.5) || pitch > 1e17) return;
    const double cell_lim = 0.95 * pitch;
    const double dot_r = 0.35 * pitch;

    // Candidates: anchored corners seen beyond the engage incidence — the
    // regime where every 1D edge measurement (subpix AND qf profiles) goes
    // geometrically blind and only a 2D template holds.
    struct SwItem {
        size_t k;
        cv::Vec3d X0;
        double u0, th0;   // surface coords of the corner
        double s0;        // absolute arc coordinate (mm; plane: basis_v)
        cv::Vec3d n_m;    // local surface normal (marker frame)
    };
    std::vector<SwItem> cands;
    for (const size_t k : anchored) {
        const cv::Vec3d& X0 = ctx.anchor_xyz_mm[k];
        SwItem it;
        it.k = k;
        it.X0 = X0;
        const cv::Vec3d w = X0 - sm.point;
        if (is_plane) {
            it.u0 = w.dot(sm.basis_u);
            it.th0 = w.dot(sm.basis_v);
            it.s0 = it.th0;
            it.n_m = sm.dir;
        } else {
            it.u0 = w.dot(sm.dir);
            const cv::Vec3d radial = w - it.u0 * sm.dir;
            const double rn = cv::norm(radial);
            if (rn < 1e-9) continue;
            it.th0 = std::atan2(radial.dot(cyl_e2),
                                radial.dot(sm.radial_ref));
            it.s0 = it.th0 * sm.radius_mm;
            it.n_m = radial / rn;
        }
        const cv::Vec3d Xc = ctx.R * X0 + ctx.t;
        const double xn = cv::norm(Xc);
        if (xn < 1e-6) continue;
        cv::Vec3d n_c = ctx.R * it.n_m;
        double cos_inc = -n_c.dot(Xc / xn);
        if (is_plane) cos_inc = std::abs(cos_inc);
        if (cos_inc >= cos_engage) continue;  // benign view: qf handles it
        if (cos_inc < 0.08) continue;         // fully grazing: hopeless
        cands.push_back(it);
    }
    if (cands.empty()) return;

    constexpr int kHalf = 12;
    constexpr int kWin = 2 * kHalf + 1;
    constexpr int kNpx = kWin * kWin;
    const double max_dev = config.model_warp_max_shift_px;
    constexpr double kMaxSeedOffsetPx = 32.0;
    const double min_valid = 0.35 * static_cast<double>(kNpx);

    const cv::Mat K_mat(ctx.K);
    cv::Mat dist_mat;
    if (!ctx.dist.empty()) {
        dist_mat = cv::Mat(ctx.dist, true).reshape(1, 1);
    }
    cv::Mat rvec_cur;
    cv::Rodrigues(cv::Mat(ctx.R), rvec_cur);
    const cv::Mat tvec_cur = (cv::Mat_<double>(3, 1)
        << ctx.t[0], ctx.t[1], ctx.t[2]);

    std::vector<cv::Point3d> obj;
    obj.reserve(cands.size());
    for (const SwItem& it : cands) {
        obj.emplace_back(it.X0[0], it.X0[1], it.X0[2]);
    }
    std::vector<cv::Point2d> proj0;
    cv::projectPoints(obj, rvec_cur, tvec_cur, K_mat, dist_mat, proj0);

    struct SwWork {
        SwItem it;
        cv::Point2d c_uv;
        cv::Point2d seed;
    };
    std::vector<SwWork> work;
    double roi_x0 = 1e12, roi_y0 = 1e12, roi_x1 = -1e12, roi_y1 = -1e12;
    for (size_t j = 0; j < cands.size(); ++j) {
        const cv::Point2d& c_uv = proj0[j];
        if (!(c_uv.x > kHalf + 2 && c_uv.y > kHalf + 2 &&
              c_uv.x < gray.cols - kHalf - 3 &&
              c_uv.y < gray.rows - kHalf - 3)) {
            continue;
        }
        const cv::Point2f& lk = points[cands[j].k];
        if (!std::isfinite(lk.x) || !std::isfinite(lk.y)) continue;
        const cv::Point2d seed(lk.x - c_uv.x, lk.y - c_uv.y);
        if (std::max(std::abs(seed.x), std::abs(seed.y)) >
            kMaxSeedOffsetPx) {
            continue;
        }
        work.push_back({cands[j], c_uv, seed});
        roi_x0 = std::min({roi_x0, c_uv.x, c_uv.x + seed.x});
        roi_y0 = std::min({roi_y0, c_uv.y, c_uv.y + seed.y});
        roi_x1 = std::max({roi_x1, c_uv.x, c_uv.x + seed.x});
        roi_y1 = std::max({roi_y1, c_uv.y, c_uv.y + seed.y});
    }
    if (work.empty()) return;

    const int margin = kHalf + static_cast<int>(std::ceil(max_dev)) + 8;
    cv::Rect roi;
    roi.x = std::max(0, static_cast<int>(std::floor(roi_x0)) - margin);
    roi.y = std::max(0, static_cast<int>(std::floor(roi_y0)) - margin);
    roi.width = std::min(
        gray.cols, static_cast<int>(std::ceil(roi_x1)) + margin) - roi.x;
    roi.height = std::min(
        gray.rows, static_cast<int>(std::ceil(roi_y1)) + margin) - roi.y;
    if (roi.width <= 2 || roi.height <= 2) return;

    cv::Mat gray32;
    gray(roi).convertTo(gray32, CV_32F);
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray32, grad_x, CV_32F, 1, 0, 3, 0.125);
    cv::Sobel(gray32, grad_y, CV_32F, 0, 1, 3, 0.125);
    const float ox = static_cast<float>(roi.x);
    const float oy = static_cast<float>(roi.y);

    const bool has_lut =
        !ctx.ray_lut.empty() && ctx.ray_lut.type() == CV_32FC2 &&
        ctx.ray_lut.rows == gray.rows && ctx.ray_lut.cols == gray.cols;

    const cv::Vec3d surf_point_c = ctx.R * sm.point + ctx.t;
    const cv::Vec3d surf_dir_c = ctx.R * sm.dir;
    cv::Vec3d plane_normal_m = sm.dir;
    if (is_plane && surf_dir_c.dot(surf_point_c) > 0.0) {
        plane_normal_m = -sm.dir;
    }

    std::vector<float> grid_x(kNpx), grid_y(kNpx);
    {
        int idx = 0;
        for (int dy = -kHalf; dy <= kHalf; ++dy) {
            for (int dx = -kHalf; dx <= kHalf; ++dx, ++idx) {
                grid_x[idx] = static_cast<float>(dx);
                grid_y[idx] = static_cast<float>(dy);
            }
        }
    }

    int saddle_count = 0;

    cv::parallel_for_(
        cv::Range(0, static_cast<int>(work.size())),
        [&](const cv::Range& range) {

    std::vector<cv::Point2d> pix(kNpx);
    std::vector<uint8_t> usable(kNpx);
    std::vector<float> templ(kNpx);
    std::vector<float> samples(kNpx);

    for (int wi = range.start; wi < range.end; ++wi) {
        const SwWork& w = work[static_cast<size_t>(wi)];
        const size_t k = w.it.k;
        const cv::Point2d& c_uv = w.c_uv;

        std::vector<cv::Point2d> und;
        if (has_lut) {
            und.resize(kNpx);
            for (int i = 0; i < kNpx; ++i) {
                und[i] = sampleRayLut(ctx.ray_lut,
                                      c_uv.x + grid_x[i],
                                      c_uv.y + grid_y[i]);
            }
        } else {
            for (int i = 0; i < kNpx; ++i) {
                pix[i] = cv::Point2d(c_uv.x + grid_x[i],
                                     c_uv.y + grid_y[i]);
            }
            cv::undistortPoints(pix, und, K_mat, dist_mat);
        }

        // Synthetic quadrant template from the surface intersection.
        int n_mask = 0;
        for (int i = 0; i < kNpx; ++i) {
            usable[i] = 0;
            templ[i] = 0.0f;

            cv::Vec3d v(und[i].x, und[i].y, 1.0);
            v /= cv::norm(v);
            double s = -1.0;
            if (is_plane) {
                const cv::Vec3d n_c = ctx.R * plane_normal_m;
                const double denom = v.dot(n_c);
                if (std::abs(denom) < 1e-9) continue;
                s = n_c.dot(surf_point_c) / denom;
            } else {
                const cv::Vec3d& u = surf_dir_c;
                const cv::Vec3d wv = v - v.dot(u) * u;
                const cv::Vec3d qa = -surf_point_c;
                const cv::Vec3d q = qa - qa.dot(u) * u;
                const double A = wv.dot(wv);
                const double B = 2.0 * wv.dot(q);
                const double C = q.dot(q) - sm.radius_mm * sm.radius_mm;
                const double disc = B * B - 4.0 * A * C;
                if (disc <= 0.0 || A < 1e-12) continue;
                s = (-B - std::sqrt(disc)) / (2.0 * A);
            }
            if (s <= 1e-6) continue;
            const cv::Vec3d Xc = s * v;
            const cv::Vec3d Xm = ctx.R.t() * (Xc - ctx.t);
            const cv::Vec3d wm = Xm - sm.point;

            double du, ds;
            if (is_plane) {
                du = wm.dot(sm.basis_u) - w.it.u0;
                ds = wm.dot(sm.basis_v) - w.it.th0;
            } else {
                const double along = wm.dot(sm.dir);
                const cv::Vec3d radial = wm - along * sm.dir;
                const double rn = cv::norm(radial);
                if (rn < 1e-9) continue;
                const double theta = std::atan2(
                    radial.dot(cyl_e2), radial.dot(sm.radial_ref));
                du = along - w.it.u0;
                ds = std::remainder(theta - w.it.th0, 2.0 * CV_PI) *
                     sm.radius_mm;
            }
            if (ctx.pattern != nullptr && ctx.pattern->valid()) {
                // Full marker pattern (parity + coded dots): the template
                // is the REAL print, so the window may span any number of
                // cells and the dots contribute signal instead of being
                // masked. u/s are the ABSOLUTE surface coordinates.
                const int col = ctx.pattern->colorAt(
                    du + w.it.u0, ds + w.it.s0);
                if (col < 0) continue;
                templ[i] = col > 0 ? 1.0f : -1.0f;
                usable[i] = 1;
                ++n_mask;
                continue;
            }

            // Quadrant fallback: stay inside the four adjacent cells (the
            // model is exact there) and outside the coded dot regions at
            // the four cell centres.
            if (std::abs(du) > cell_lim || std::abs(ds) > cell_lim) {
                continue;
            }
            const double dcu = std::abs(du) - 0.5 * pitch;
            const double dcs = std::abs(ds) - 0.5 * pitch;
            if (std::sqrt(dcu * dcu + dcs * dcs) < dot_r) continue;

            templ[i] = ((du > 0.0) != (ds > 0.0)) ? -1.0f : 1.0f;
            usable[i] = 1;
            ++n_mask;
        }
        if (static_cast<double>(n_mask) < min_valid) continue;

        double t_sum = 0.0;
        for (int i = 0; i < kNpx; ++i) {
            if (usable[i]) t_sum += templ[i];
        }
        const double t_mean = t_sum / static_cast<double>(n_mask);
        double t_var = 0.0;
        for (int i = 0; i < kNpx; ++i) {
            if (usable[i]) {
                const double d = templ[i] - t_mean;
                t_var += d * d;
            }
        }
        if (t_var < 1e-6) continue;

        // Translation-only LK with photometric gain/bias.  The parity of
        // the synthetic saddle is unknown, so a NEGATIVE gain is a valid
        // solution (flipped quadrants), unlike the reference-photo warp.
        double px = w.seed.x;
        double py = w.seed.y;
        bool failed = false;
        bool bicubic_phase = false;
        for (int iter = 0; iter < config.model_warp_max_iters; ++iter) {
            cv::Matx44d ATA = cv::Matx44d::zeros();
            cv::Vec4d ATb(0.0, 0.0, 0.0, 0.0);
            for (int i = 0; i < kNpx; ++i) {
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
            if (std::abs(a) < 2.0) {  // gray levels per template unit
                failed = true;
                break;
            }
            px -= dpx / a;
            py -= dpy / a;
            if (std::abs(dpx / a) < 1e-3 && std::abs(dpy / a) < 1e-3) {
                if (bicubic_phase) break;
                bicubic_phase = true;
                continue;
            }
            if (!bicubic_phase &&
                std::abs(dpx / a) < 0.25 && std::abs(dpy / a) < 0.25) {
                bicubic_phase = true;
            }
            if (std::max(std::abs(px - w.seed.x),
                         std::abs(py - w.seed.y)) > max_dev + 2.0) {
                failed = true;
                break;
            }
        }
        if (failed) continue;

        double i_sum = 0.0;
        for (int i = 0; i < kNpx; ++i) {
            if (!usable[i]) continue;
            const float sx =
                static_cast<float>(c_uv.x + px) + grid_x[i] - ox;
            const float sy =
                static_cast<float>(c_uv.y + py) + grid_y[i] - oy;
            samples[i] = sampleBicubic(gray32, sx, sy);
            i_sum += samples[i];
        }
        const double i_mean = i_sum / static_cast<double>(n_mask);
        double num = 0.0, i_var = 0.0;
        for (int i = 0; i < kNpx; ++i) {
            if (!usable[i]) continue;
            const double dt = templ[i] - t_mean;
            const double di = samples[i] - i_mean;
            num += dt * di;
            i_var += di * di;
        }
        const double den = std::sqrt(t_var * i_var);
        const double zncc = den > 1e-9 ? std::abs(num) / den : 0.0;

        if (zncc < config.model_warp_min_zncc) continue;
        if (std::max(std::abs(px - w.seed.x),
                     std::abs(py - w.seed.y)) > max_dev) {
            continue;
        }

        points[k] = cv::Point2f(static_cast<float>(c_uv.x + px),
                                static_cast<float>(c_uv.y + py));
        stats.model_warp_ok[k] = 1;
        stats.model_warp_zncc[k] = static_cast<float>(zncc);
    }

        });  // cv::parallel_for_

    for (const SwWork& w : work) {
        if (stats.model_warp_ok[w.it.k]) ++saddle_count;
    }
    stats.qf_saddle_count = saddle_count;
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

    // Per-point work is independent; the surviving points are assembled
    // serially in index order afterwards, so both the values and the order
    // match the serial loop bit for bit.
    std::vector<cv::Point2f> result_pts;
    std::vector<uint8_t> result_ok;

    for (int iter = 0; iter < iterations; ++iter) {
        const int n = static_cast<int>(points.size());
        result_pts.assign(static_cast<size_t>(n), cv::Point2f());
        result_ok.assign(static_cast<size_t>(n), 0);

        parallelChunks(n, 32, [&](int begin, int end) {
        for (int pi = begin; pi < end; ++pi) {
            const cv::Point2f& p = points[static_cast<size_t>(pi)];
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

            result_pts[static_cast<size_t>(pi)] = q;
            result_ok[static_cast<size_t>(pi)] = 1;
        }
        });  // parallelChunks

        std::vector<cv::Point2f> next;
        next.reserve(points.size());
        for (int pi = 0; pi < n; ++pi) {
            if (result_ok[static_cast<size_t>(pi)]) {
                next.push_back(result_pts[static_cast<size_t>(pi)]);
            }
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

    // One output slot per input point (every code path below fills exactly
    // one), so the loop parallelizes with disjoint writes -> bit-identical.
    output.assign(points.size(), RefinedCorner{});

    parallelChunks(
        static_cast<int>(points.size()),
        16,
        [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
        const cv::Point2f& p = points[static_cast<size_t>(i)];
        RefinedCorner& corner = output[static_cast<size_t>(i)];
        corner.uv = p;

        if (!insideWithRadius(gray_f, p, r)) {
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
    }
    });  // parallelChunks

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