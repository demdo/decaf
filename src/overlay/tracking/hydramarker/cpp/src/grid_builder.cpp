#include "grid_builder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <vector>

#include "geometry_utils.hpp"

namespace hydramarker {

namespace {

constexpr float kMinCellAreaPx2 = 15.0f;
constexpr float kMinCellEdgePx = 3.0f;
constexpr float kMaxCellEdgeRatio = 2.4f;
constexpr float kMaxCellDiagonalRatio = 2.0f;
constexpr float kMaxDominantOrientationDiffDeg = 18.0f;

bool basicCellPlausible(const GridCell& cell) {
    const auto& q = cell.corner_uv;

    for (const auto& p : q) {
        if (!geom::isFinite(p)) {
            return false;
        }
    }

    if (std::abs(geom::polygonSignedArea(q)) < kMinCellAreaPx2) {
        return false;
    }

    if (!geom::isConvexQuad(q)) {
        return false;
    }

    const float e0 = geom::dist(q[0], q[1]);
    const float e1 = geom::dist(q[1], q[2]);
    const float e2 = geom::dist(q[2], q[3]);
    const float e3 = geom::dist(q[3], q[0]);

    const float min_e = std::min(std::min(e0, e1), std::min(e2, e3));
    const float max_e = std::max(std::max(e0, e1), std::max(e2, e3));

    if (min_e < kMinCellEdgePx) {
        return false;
    }

    if (max_e / std::max(min_e, 1e-6f) > kMaxCellEdgeRatio) {
        return false;
    }

    const float d0 = geom::dist(q[0], q[2]);
    const float d1 = geom::dist(q[1], q[3]);

    const float min_d = std::min(d0, d1);
    const float max_d = std::max(d0, d1);

    if (min_d < kMinCellEdgePx) {
        return false;
    }

    if (max_d / std::max(min_d, 1e-6f) > kMaxCellDiagonalRatio) {
        return false;
    }

    return true;
}

int sharedCornerCount(const GridCell& a, const GridCell& b) {
    int count = 0;

    for (int ia : a.corner_indices) {
        for (int ib : b.corner_indices) {
            if (ia == ib) {
                ++count;
            }
        }
    }

    return count;
}

bool shareFullGridEdge(const GridCell& a, const GridCell& b) {
    return sharedCornerCount(a, b) >= 2;
}

bool sameWindingSign(const GridCell& a, const GridCell& b) {
    const float aa = geom::polygonSignedArea(a.corner_uv);
    const float ab = geom::polygonSignedArea(b.corner_uv);

    if (std::abs(aa) < 1e-6f || std::abs(ab) < 1e-6f) {
        return false;
    }

    return aa * ab > 0.0f;
}

float canonicalCellAngleDeg(const GridCell& cell) {
    const auto& q = cell.corner_uv;

    const cv::Point2f e0 = q[1] - q[0];
    const cv::Point2f e1 = q[3] - q[0];

    cv::Point2f v =
        (geom::dist2(q[0], q[1]) >= geom::dist2(q[0], q[3]))
        ? e0
        : e1;

    float angle =
        std::atan2(v.y, v.x) * 180.0f / static_cast<float>(CV_PI);

    while (angle < 0.0f) {
        angle += 90.0f;
    }

    while (angle >= 90.0f) {
        angle -= 90.0f;
    }

    return angle;
}

float circularAngleDiffDeg(float a, float b) {
    float d = std::abs(a - b);

    while (d >= 90.0f) {
        d -= 90.0f;
    }

    d = std::abs(d);

    return std::min(d, 90.0f - d);
}

std::vector<GridCell> filterByDominantOrientation(
    const std::vector<GridCell>& cells
) {
    if (cells.size() < 4) {
        return cells;
    }

    constexpr int bins = 36;

    std::array<int, bins> hist{};
    hist.fill(0);

    std::vector<float> angles;
    angles.reserve(cells.size());

    for (const auto& c : cells) {
        if (!basicCellPlausible(c)) {
            angles.push_back(0.0f);
            continue;
        }

        const float a = canonicalCellAngleDeg(c);

        angles.push_back(a);

        int bin = static_cast<int>(
            std::floor(a / 90.0f * static_cast<float>(bins))
        );

        bin = std::clamp(bin, 0, bins - 1);

        hist[bin]++;
    }

    int best_bin = 0;

    for (int i = 1; i < bins; ++i) {
        if (hist[i] > hist[best_bin]) {
            best_bin = i;
        }
    }

    const float dominant_angle =
        (static_cast<float>(best_bin) + 0.5f)
        * 90.0f
        / static_cast<float>(bins);

    std::vector<GridCell> filtered;
    filtered.reserve(cells.size());

    for (size_t i = 0; i < cells.size(); ++i) {
        if (!basicCellPlausible(cells[i])) {
            continue;
        }

        const float diff =
            circularAngleDiffDeg(angles[i], dominant_angle);

        if (diff > kMaxDominantOrientationDiffDeg) {
            continue;
        }

        filtered.push_back(cells[i]);
    }

    if (filtered.size() < std::max<size_t>(4, cells.size() / 4)) {
        return cells;
    }

    return filtered;
}

std::vector<GridCell> keepLargestEdgeConnectedComponent(
    const std::vector<GridCell>& raw_cells
) {
    if (raw_cells.size() <= 2) {
        std::vector<GridCell> plausible;
        plausible.reserve(raw_cells.size());

        for (const auto& c : raw_cells) {
            if (basicCellPlausible(c)) {
                plausible.push_back(c);
            }
        }

        return plausible;
    }

    const int n = static_cast<int>(raw_cells.size());

    std::vector<std::vector<int>> adjacency(n);

    for (int i = 0; i < n; ++i) {
        if (!basicCellPlausible(raw_cells[i])) {
            continue;
        }

        for (int j = i + 1; j < n; ++j) {
            if (!basicCellPlausible(raw_cells[j])) {
                continue;
            }

            if (!shareFullGridEdge(raw_cells[i], raw_cells[j])) {
                continue;
            }

            if (!sameWindingSign(raw_cells[i], raw_cells[j])) {
                continue;
            }

            adjacency[i].push_back(j);
            adjacency[j].push_back(i);
        }
    }

    std::vector<int> best_component;
    std::vector<char> visited(n, 0);

    for (int start = 0; start < n; ++start) {
        if (visited[start]) {
            continue;
        }

        if (!basicCellPlausible(raw_cells[start])) {
            visited[start] = 1;
            continue;
        }

        std::vector<int> component;
        std::queue<int> q;

        visited[start] = 1;
        q.push(start);

        while (!q.empty()) {
            const int u = q.front();
            q.pop();

            component.push_back(u);

            for (int v : adjacency[u]) {
                if (visited[v]) {
                    continue;
                }

                visited[v] = 1;
                q.push(v);
            }
        }

        if (component.size() > best_component.size()) {
            best_component = std::move(component);
        }
    }

    if (best_component.empty()) {
        return {};
    }

    if (best_component.size() < 2 && raw_cells.size() >= 4) {
        std::vector<GridCell> plausible;

        for (const auto& c : raw_cells) {
            if (basicCellPlausible(c)) {
                plausible.push_back(c);
            }
        }

        return plausible;
    }

    std::sort(best_component.begin(), best_component.end());

    std::vector<GridCell> filtered;
    filtered.reserve(best_component.size());

    for (int idx : best_component) {
        filtered.push_back(raw_cells[idx]);
    }

    return filtered;
}

void computeGridExtent(
    const std::vector<GridCorner>& corners,
    int& min_i,
    int& max_i,
    int& min_j,
    int& max_j
) {
    min_i = std::numeric_limits<int>::max();
    min_j = std::numeric_limits<int>::max();
    max_i = std::numeric_limits<int>::min();
    max_j = std::numeric_limits<int>::min();

    for (const auto& c : corners) {
        min_i = std::min(min_i, c.i);
        min_j = std::min(min_j, c.j);
        max_i = std::max(max_i, c.i);
        max_j = std::max(max_j, c.j);
    }
}

} // namespace

GridBuilder::GridBuilder() = default;

std::optional<CheckerboardDetection> GridBuilder::build(
    const LatticeResult& lattice,
    float duplicate_dist_px,
    int min_corners,
    int min_cells
) const {
    if (!lattice.valid) {
        return std::nullopt;
    }

    std::vector<GridCorner> corners = makeGridCorners(
        lattice,
        duplicate_dist_px
    );

    return buildFromCorners(
        corners,
        duplicate_dist_px,
        min_corners,
        min_cells,
        false,
        false
    );
}

std::optional<CheckerboardDetection> GridBuilder::buildFromCorners(
    const std::vector<GridCorner>& input_corners,
    float duplicate_dist_px,
    int min_corners,
    int min_cells,
    bool tracking,
    bool stable
) const {
    std::vector<GridCorner> corners = sanitizeGridCorners(
        input_corners,
        duplicate_dist_px
    );

    if (static_cast<int>(corners.size()) < min_corners) {
        return std::nullopt;
    }

    std::vector<GridCell> cells = makeGridCells(corners);

    if (static_cast<int>(cells.size()) < min_cells) {
        return std::nullopt;
    }

    int min_i = 0;
    int max_i = 0;
    int min_j = 0;
    int max_j = 0;

    computeGridExtent(corners, min_i, max_i, min_j, max_j);

    CheckerboardDetection detection;
    detection.corners = std::move(corners);
    detection.cells = std::move(cells);
    detection.cols = max_i - min_i + 1;
    detection.rows = max_j - min_j + 1;
    detection.tracking = tracking;
    detection.stable = stable;

    return detection;
}

std::vector<GridCorner> GridBuilder::makeGridCorners(
    const LatticeResult& lattice,
    float duplicate_dist_px
) const {
    std::vector<GridCorner> corners;
    corners.reserve(lattice.points.size());

    for (const auto& p : lattice.points) {
        if (!p.valid) {
            continue;
        }

        const int i = static_cast<int>(std::lround(p.ij.x));
        const int j = static_cast<int>(std::lround(p.ij.y));

        if (isDuplicate(corners, p.uv, duplicate_dist_px)) {
            continue;
        }

        GridCorner corner;
        corner.i = i;
        corner.j = j;
        corner.uv = p.uv;

        corners.push_back(corner);
    }

    std::sort(
        corners.begin(),
        corners.end(),
        [](const GridCorner& a, const GridCorner& b) {
            if (a.j != b.j) {
                return a.j < b.j;
            }
            return a.i < b.i;
        }
    );

    return corners;
}

std::vector<GridCorner> GridBuilder::sanitizeGridCorners(
    const std::vector<GridCorner>& input_corners,
    float duplicate_dist_px
) const {
    std::vector<GridCorner> corners;
    corners.reserve(input_corners.size());

    for (const auto& c : input_corners) {
        if (!geom::isFinite(c.uv)) {
            continue;
        }

        if (c.i == std::numeric_limits<int>::min() ||
            c.j == std::numeric_limits<int>::min()) {
            continue;
        }

        if (isDuplicate(corners, c.uv, duplicate_dist_px)) {
            continue;
        }

        const int existing_idx = findCornerIndex(corners, c.i, c.j);

        if (existing_idx >= 0) {
            const cv::Point2f d = corners[existing_idx].uv - c.uv;
            const float d2 = d.dot(d);

            if (d2 < 1.0f) {
                continue;
            }

            continue;
        }

        corners.push_back(c);
    }

    std::sort(
        corners.begin(),
        corners.end(),
        [](const GridCorner& a, const GridCorner& b) {
            if (a.j != b.j) {
                return a.j < b.j;
            }
            return a.i < b.i;
        }
    );

    return corners;
}

std::vector<GridCell> GridBuilder::makeGridCells(
    const std::vector<GridCorner>& corners
) const {
    std::vector<GridCell> raw_cells;

    for (const auto& c : corners) {
        const int i = c.i;
        const int j = c.j;

        const int idx00 = findCornerIndex(corners, i,     j);
        const int idx10 = findCornerIndex(corners, i + 1, j);
        const int idx11 = findCornerIndex(corners, i + 1, j + 1);
        const int idx01 = findCornerIndex(corners, i,     j + 1);

        if (idx00 < 0 || idx10 < 0 || idx11 < 0 || idx01 < 0) {
            continue;
        }

        GridCell cell;
        cell.i = i;
        cell.j = j;

        cell.corner_indices = {idx00, idx10, idx11, idx01};

        cell.corner_uv = {
            corners[idx00].uv,
            corners[idx10].uv,
            corners[idx11].uv,
            corners[idx01].uv
        };

        cell.center_uv =
            0.25f * (
                cell.corner_uv[0] +
                cell.corner_uv[1] +
                cell.corner_uv[2] +
                cell.corner_uv[3]
            );

        if (basicCellPlausible(cell)) {
            raw_cells.push_back(cell);
        }
    }

    raw_cells = filterByDominantOrientation(raw_cells);

    return keepLargestEdgeConnectedComponent(raw_cells);
}

int GridBuilder::findCornerIndex(
    const std::vector<GridCorner>& corners,
    int i,
    int j
) const {
    for (int idx = 0; idx < static_cast<int>(corners.size()); ++idx) {
        if (corners[idx].i == i && corners[idx].j == j) {
            return idx;
        }
    }

    return -1;
}

bool GridBuilder::isDuplicate(
    const std::vector<GridCorner>& corners,
    const cv::Point2f& uv,
    float duplicate_dist_px
) const {
    const float r2 = duplicate_dist_px * duplicate_dist_px;

    for (const auto& c : corners) {
        if (geom::dist2(c.uv, uv) <= r2) {
            return true;
        }
    }

    return false;
}

} // namespace hydramarker
