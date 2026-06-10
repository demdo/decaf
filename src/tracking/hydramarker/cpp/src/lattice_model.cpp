#include "lattice_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

namespace hydramarker {

namespace {

constexpr float kEps = 1e-6f;

static float sqr(float x) {
    return x * x;
}

static float dist2(const cv::Point2f& a, const cv::Point2f& b) {
    return sqr(a.x - b.x) + sqr(a.y - b.y);
}

static float norm(const cv::Point2f& v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

static float dot(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x * b.x + a.y * b.y;
}

static float cross(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x * b.y - a.y * b.x;
}

static cv::Point2f normalizeSafe(const cv::Point2f& v) {
    const float n = norm(v);

    if (n < kEps) {
        return cv::Point2f(1.0f, 0.0f);
    }

    return cv::Point2f(v.x / n, v.y / n);
}

static float angleModPi(const cv::Point2f& v) {
    float a = std::atan2(v.y, v.x);

    if (a < 0.0f) {
        a += static_cast<float>(CV_PI);
    }

    if (a >= static_cast<float>(CV_PI)) {
        a -= static_cast<float>(CV_PI);
    }

    return a;
}

static float angleDiffModPi(float a, float b) {
    const float d = std::abs(a - b);
    return std::min(d, static_cast<float>(CV_PI) - d);
}

static float medianValue(std::vector<float> values) {
    if (values.empty()) {
        return 0.0f;
    }

    const size_t mid = values.size() / 2;

    std::nth_element(
        values.begin(),
        values.begin() + static_cast<std::ptrdiff_t>(mid),
        values.end()
    );

    return values[mid];
}

static float estimateNearestSpacing(
    const std::vector<cv::Point2f>& pts
) {
    if (pts.size() < 2) {
        return 1.0f;
    }

    std::vector<float> nn;
    nn.reserve(pts.size());

    for (size_t i = 0; i < pts.size(); ++i) {
        float best_d2 = std::numeric_limits<float>::max();

        for (size_t j = 0; j < pts.size(); ++j) {
            if (i == j) {
                continue;
            }

            best_d2 = std::min(best_d2, dist2(pts[i], pts[j]));
        }

        if (std::isfinite(best_d2)) {
            nn.push_back(std::sqrt(best_d2));
        }
    }

    return std::max(1.0f, medianValue(nn));
}

static std::int64_t gridKey(int i, int j) {
    return (static_cast<std::int64_t>(i) << 32) ^
           static_cast<std::uint32_t>(j);
}

static int findCentralSeed(
    const std::vector<cv::Point2f>& pts
) {
    cv::Point2f center(0.0f, 0.0f);

    for (const auto& p : pts) {
        center += p;
    }

    center *= 1.0f / static_cast<float>(pts.size());

    int best_idx = -1;
    float best_d = std::numeric_limits<float>::max();

    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        const float d = dist2(pts[i], center);

        if (d < best_d) {
            best_d = d;
            best_idx = i;
        }
    }

    return best_idx;
}

static bool directionalGeometryOk(
    const cv::Point2f& src,
    const cv::Point2f& dst,
    const cv::Point2f& dir,
    float spacing
) {
    const cv::Point2f v = dst - src;

    const float forward = dot(v, dir);
    const float lateral = std::abs(cross(dir, v));

    const float min_forward = 0.60f * spacing;
    const float max_forward = 1.35f * spacing;
    const float max_lateral = 0.28f * spacing;

    if (forward < min_forward || forward > max_forward) {
        return false;
    }

    if (lateral > max_lateral) {
        return false;
    }

    return true;
}

static int findDirectionalNeighbor(
    int src_idx,
    const std::vector<cv::Point2f>& pts,
    const std::vector<bool>& assigned,
    const cv::Point2f& dir,
    float spacing
) {
    const cv::Point2f src = pts[src_idx];

    int best_idx = -1;
    float best_score = std::numeric_limits<float>::max();

    for (int k = 0; k < static_cast<int>(pts.size()); ++k) {
        if (k == src_idx) {
            continue;
        }

        if (assigned[k]) {
            continue;
        }

        if (!directionalGeometryOk(src, pts[k], dir, spacing)) {
            continue;
        }

        // Bidirectional consistency:
        // If k is the +dir neighbor of src, then src must also be the -dir
        // neighbor candidate of k geometrically.
        if (!directionalGeometryOk(pts[k], src, -dir, spacing)) {
            continue;
        }

        const cv::Point2f v = pts[k] - src;

        const float forward = dot(v, dir);
        const float lateral = std::abs(cross(dir, v));

        const float score =
            std::abs(forward - spacing) +
            2.5f * lateral;

        if (score < best_score) {
            best_score = score;
            best_idx = k;
        }
    }

    return best_idx;
}

static float localNeighborResidual(
    const std::vector<cv::Point2f>& pts,
    int src_idx,
    int nb_idx,
    const cv::Point2f& dir,
    float spacing
) {
    const cv::Point2f v = pts[nb_idx] - pts[src_idx];

    const float forward = dot(v, dir);
    const float lateral = std::abs(cross(dir, v));

    return std::abs(forward - spacing) + lateral;
}

static bool validateLatticeStructure(
    const std::vector<LatticePoint>& points,
    float spacing_u,
    float spacing_v
) {
    if (points.size() < 6) {
        return false;
    }

    std::unordered_map<std::int64_t, int> by_grid;
    by_grid.reserve(points.size());

    int min_i = std::numeric_limits<int>::max();
    int max_i = std::numeric_limits<int>::lowest();
    int min_j = std::numeric_limits<int>::max();
    int max_j = std::numeric_limits<int>::lowest();

    for (int idx = 0; idx < static_cast<int>(points.size()); ++idx) {
        const int i = static_cast<int>(std::round(points[idx].ij.x));
        const int j = static_cast<int>(std::round(points[idx].ij.y));

        by_grid[gridKey(i, j)] = idx;

        min_i = std::min(min_i, i);
        max_i = std::max(max_i, i);
        min_j = std::min(min_j, j);
        max_j = std::max(max_j, j);
    }

    const int cols = max_i - min_i + 1;
    const int rows = max_j - min_j + 1;

    if (cols <= 0 || rows <= 0) {
        return false;
    }

    const int bbox_count = cols * rows;
    const float density =
        static_cast<float>(points.size()) /
        static_cast<float>(std::max(1, bbox_count));

    // Important:
    // Do not require a full board. Partial views are allowed.
    // But very sparse huge boxes are usually caused by greedy drift.
    if (bbox_count >= 16 && density < 0.35f) {
        return false;
    }

    std::vector<float> u_edge_d2;
    std::vector<float> v_edge_d2;
    std::vector<float> cell_crosses;

    u_edge_d2.reserve(points.size());
    v_edge_d2.reserve(points.size());
    cell_crosses.reserve(points.size());

    for (const auto& p : points) {
        const int i = static_cast<int>(std::round(p.ij.x));
        const int j = static_cast<int>(std::round(p.ij.y));

        const auto it_u = by_grid.find(gridKey(i + 1, j));
        if (it_u != by_grid.end()) {
            u_edge_d2.push_back(dist2(p.uv, points[it_u->second].uv));
        }

        const auto it_v = by_grid.find(gridKey(i, j + 1));
        if (it_v != by_grid.end()) {
            v_edge_d2.push_back(dist2(p.uv, points[it_v->second].uv));
        }

        const auto it_10 = by_grid.find(gridKey(i + 1, j));
        const auto it_01 = by_grid.find(gridKey(i, j + 1));
        const auto it_11 = by_grid.find(gridKey(i + 1, j + 1));

        if (it_10 != by_grid.end() &&
            it_01 != by_grid.end() &&
            it_11 != by_grid.end()) {
            const cv::Point2f p00 = p.uv;
            const cv::Point2f p10 = points[it_10->second].uv;
            const cv::Point2f p01 = points[it_01->second].uv;

            cell_crosses.push_back(cross(p10 - p00, p01 - p00));
        }
    }

    const int edge_count =
        static_cast<int>(u_edge_d2.size() + v_edge_d2.size());

    if (edge_count < 4) {
        return false;
    }

    const float med_u =
        u_edge_d2.empty()
            ? spacing_u
            : std::sqrt(medianValue(u_edge_d2));

    const float med_v =
        v_edge_d2.empty()
            ? spacing_v
            : std::sqrt(medianValue(v_edge_d2));

    if (med_u < kEps || med_v < kEps) {
        return false;
    }

    int good_u = 0;
    const float min_u_d2 = sqr(0.60f * med_u);
    const float max_u_d2 = sqr(1.40f * med_u);
    for (const float e_d2 : u_edge_d2) {
        if (e_d2 >= min_u_d2 && e_d2 <= max_u_d2) {
            good_u++;
        }
    }

    int good_v = 0;
    const float min_v_d2 = sqr(0.60f * med_v);
    const float max_v_d2 = sqr(1.40f * med_v);
    for (const float e_d2 : v_edge_d2) {
        if (e_d2 >= min_v_d2 && e_d2 <= max_v_d2) {
            good_v++;
        }
    }

    const int total_u = static_cast<int>(u_edge_d2.size());
    const int total_v = static_cast<int>(v_edge_d2.size());

    if (total_u > 0 &&
        static_cast<float>(good_u) / static_cast<float>(total_u) < 0.75f) {
        return false;
    }

    if (total_v > 0 &&
        static_cast<float>(good_v) / static_cast<float>(total_v) < 0.75f) {
        return false;
    }

    if (!cell_crosses.empty()) {
        int pos = 0;
        int neg = 0;

        std::vector<float> abs_crosses;
        abs_crosses.reserve(cell_crosses.size());

        for (const float c : cell_crosses) {
            if (c > 0.0f) {
                pos++;
            } else if (c < 0.0f) {
                neg++;
            }

            abs_crosses.push_back(std::abs(c));
        }

        const int dominant = std::max(pos, neg);
        const int total = pos + neg;

        if (total > 0 &&
            static_cast<float>(dominant) / static_cast<float>(total) < 0.90f) {
            return false;
        }

        const float med_area = medianValue(abs_crosses);

        if (med_area < 0.20f * med_u * med_v) {
            return false;
        }
    }

    return true;
}

} // namespace


// ============================================================

LatticeModel::LatticeModel() = default;


// ============================================================

std::optional<LatticeResult> LatticeModel::fit(
    const std::vector<cv::Point2f>& corners
) const {
    if (corners.size() < 6) {
        return std::nullopt;
    }

    cv::Point2f axis_u;
    cv::Point2f axis_v;

    float spacing_u = 0.0f;
    float spacing_v = 0.0f;

    if (!estimateAxes(
            corners,
            axis_u,
            axis_v,
            spacing_u,
            spacing_v
        )) {
        return std::nullopt;
    }

    if (spacing_u <= kEps || spacing_v <= kEps) {
        return std::nullopt;
    }

    auto lattice_points = growGrid(
        corners,
        axis_u,
        axis_v,
        spacing_u,
        spacing_v
    );

    if (lattice_points.size() < 6) {
        return std::nullopt;
    }

    if (!validateLatticeStructure(
            lattice_points,
            spacing_u,
            spacing_v
        )) {
        return std::nullopt;
    }

    LatticeResult result;
    result.points    = std::move(lattice_points);
    result.axis_u    = axis_u;
    result.axis_v    = axis_v;
    result.origin    = estimateOrigin(corners);
    result.spacing_u = spacing_u;
    result.spacing_v = spacing_v;
    result.valid     = true;

    return result;
}


// ============================================================

bool LatticeModel::estimateAxes(
    const std::vector<cv::Point2f>& pts,
    cv::Point2f& axis_u,
    cv::Point2f& axis_v,
    float& spacing_u,
    float& spacing_v
) const {
    if (pts.size() < 4) {
        return false;
    }

    const float spacing = estimateNearestSpacing(pts);

    // Fix A: the diagonal distance between grid corners is sqrt(2) * spacing
    // ≈ 1.41 * spacing.  The original d_max = 1.55 * spacing admitted
    // diagonal pairs into the direction histogram, causing the dominant bin
    // to land on the 45° diagonal instead of on the true grid axis whenever
    // the board is rotated or perspective-distorted.
    //
    // Tightening d_max to 1.25 * spacing safely excludes all diagonal pairs
    // (1.41 > 1.25) while still tolerating up to ±25 % perspective stretch
    // on genuine axis-aligned neighbours.
    const float d_min = 0.55f * spacing;
    const float d_max = 1.25f * spacing;
    const float d_min2 = d_min * d_min;
    const float d_max2 = d_max * d_max;

    std::vector<cv::Point2f> dirs;
    std::vector<cv::Point2f> diffs;

    dirs.reserve(pts.size() * 8);
    diffs.reserve(pts.size() * 8);

    for (size_t a = 0; a < pts.size(); ++a) {
        for (size_t b = a + 1; b < pts.size(); ++b) {
            const cv::Point2f v = pts[b] - pts[a];
            const float d2 = v.x * v.x + v.y * v.y;

            if (d2 < d_min2 || d2 > d_max2) {
                continue;
            }

            const float inv_d = 1.0f / std::sqrt(d2);
            dirs.emplace_back(v.x * inv_d, v.y * inv_d);
            diffs.push_back(v);
        }
    }

    if (dirs.size() < 6) {
        return false;
    }

    constexpr int bins = 72;
    std::vector<int> hist(bins, 0);

    for (const auto& d : dirs) {
        int bin = static_cast<int>(
            std::floor(
                angleModPi(d) /
                static_cast<float>(CV_PI) *
                static_cast<float>(bins)
            )
        );

        bin = std::clamp(bin, 0, bins - 1);
        hist[bin]++;
    }

    const int b0 = static_cast<int>(
        std::max_element(hist.begin(), hist.end()) - hist.begin()
    );

    const float a0 =
        (static_cast<float>(b0) + 0.5f) *
        static_cast<float>(CV_PI) /
        static_cast<float>(bins);

    int b1   = -1;
    int best = -1;

    for (int b = 0; b < bins; ++b) {
        const float a =
            (static_cast<float>(b) + 0.5f) *
            static_cast<float>(CV_PI) /
            static_cast<float>(bins);

        const float sep = angleDiffModPi(a0, a);

        if (sep < 45.0f * static_cast<float>(CV_PI) / 180.0f ||
            sep > 135.0f * static_cast<float>(CV_PI) / 180.0f) {
            continue;
        }

        if (hist[b] > best) {
            best = hist[b];
            b1   = b;
        }
    }

    if (b1 < 0) {
        return false;
    }

    const float a1 =
        (static_cast<float>(b1) + 0.5f) *
        static_cast<float>(CV_PI) /
        static_cast<float>(bins);

    // Fix A (cont.): explicit orthogonality guard.
    // Even after tightening d_max there are degenerate point configurations
    // (very sparse, heavily foreshortened) where the two histogram peaks end
    // up closer than ~55° or further apart than ~125°.  Such axis pairs
    // cannot describe a valid rectangular lattice and must be rejected early
    // rather than propagating into growGrid where they produce diagonal cells.
    {
        const float sep_deg =
            angleDiffModPi(a0, a1) * 180.0f / static_cast<float>(CV_PI);

        if (sep_deg < 55.0f || sep_deg > 125.0f) {
            return false;
        }
    }

    axis_u = normalizeSafe(cv::Point2f(std::cos(a0), std::sin(a0)));
    axis_v = normalizeSafe(cv::Point2f(std::cos(a1), std::sin(a1)));

    if (cross(axis_u, axis_v) < 0.0f) {
        axis_v = -axis_v;
    }

    // Spacing estimation: collect projections onto the confirmed axes.
    // Relax the outer window slightly vs. d_max so minor per-pair perspective
    // variation does not thin out the sample too aggressively.
    std::vector<float> du_values;
    std::vector<float> dv_values;

    for (const auto& d : diffs) {
        const float du = std::abs(dot(d, axis_u));
        const float dv = std::abs(dot(d, axis_v));

        if (du > dv &&
            du > 0.45f * spacing &&
            du < 1.35f * spacing) {
            du_values.push_back(du);
        } else if (
            dv > 0.45f * spacing &&
            dv < 1.35f * spacing
        ) {
            dv_values.push_back(dv);
        }
    }

    if (du_values.empty() || dv_values.empty()) {
        return false;
    }

    spacing_u = medianValue(du_values);
    spacing_v = medianValue(dv_values);

    return spacing_u > kEps && spacing_v > kEps;
}


// ============================================================

std::vector<LatticePoint> LatticeModel::growGrid(
    const std::vector<cv::Point2f>& pts,
    const cv::Point2f& axis_u,
    const cv::Point2f& axis_v,
    float spacing_u,
    float spacing_v
) const {
    std::vector<LatticePoint> result;

    if (pts.empty()) {
        return result;
    }

    const int seed = findCentralSeed(pts);

    if (seed < 0) {
        return result;
    }

    std::vector<bool>      assigned(pts.size(), false);
    std::vector<cv::Point2i> ij(pts.size(), cv::Point2i(0, 0));
    std::vector<float>     residuals(pts.size(), 0.0f);

    std::unordered_map<std::int64_t, int> index_by_grid;
    index_by_grid.reserve(pts.size());

    std::queue<int> q;

    assigned[seed]              = true;
    ij[seed]                    = cv::Point2i(0, 0);
    residuals[seed]             = 0.0f;
    index_by_grid[gridKey(0, 0)] = seed;
    q.push(seed);

    const std::array<cv::Point2f, 4> dirs = {
        axis_u,
        -axis_u,
        axis_v,
        -axis_v
    };

    const std::array<cv::Point2i, 4> steps = {
        cv::Point2i(1,  0),
        cv::Point2i(-1, 0),
        cv::Point2i(0,  1),
        cv::Point2i(0, -1)
    };

    const std::array<float, 4> spacings = {
        spacing_u,
        spacing_u,
        spacing_v,
        spacing_v
    };

    while (!q.empty()) {
        const int current = q.front();
        q.pop();

        for (int d = 0; d < 4; ++d) {
            const int nb = findDirectionalNeighbor(
                current,
                pts,
                assigned,
                dirs[d],
                spacings[d]
            );

            if (nb < 0) {
                continue;
            }

            const cv::Point2i next_ij = ij[current] + steps[d];
            const auto key = gridKey(next_ij.x, next_ij.y);

            if (index_by_grid.find(key) != index_by_grid.end()) {
                continue;
            }

            assigned[nb] = true;
            ij[nb]       = next_ij;
            residuals[nb] = localNeighborResidual(
                pts,
                current,
                nb,
                dirs[d],
                spacings[d]
            );

            index_by_grid[key] = nb;
            q.push(nb);
        }
    }

    result.reserve(pts.size());

    for (int k = 0; k < static_cast<int>(pts.size()); ++k) {
        if (!assigned[k]) {
            continue;
        }

        LatticePoint lp;
        lp.uv      = pts[k];
        lp.ij      = cv::Point2f(
            static_cast<float>(ij[k].x),
            static_cast<float>(ij[k].y)
        );
        lp.residual = residuals[k];
        lp.valid    = true;

        result.push_back(lp);
    }

    return result;
}


// ============================================================

cv::Point2f LatticeModel::estimateOrigin(
    const std::vector<cv::Point2f>& pts
) const {
    cv::Point2f mean(0.0f, 0.0f);

    for (const auto& p : pts) {
        mean += p;
    }

    mean *= 1.0f / static_cast<float>(pts.size());

    return mean;
}

} // namespace hydramarker
