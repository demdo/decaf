#include "grid_builder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "geometry_utils.hpp"

namespace hydramarker {

GridBuilder::GridBuilder() = default;

// ------------------------------------------------------------

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

    if (static_cast<int>(corners.size()) < min_corners) {
        return std::nullopt;
    }

    std::vector<GridCell> cells = makeGridCells(corners);

    if (static_cast<int>(cells.size()) < min_cells) {
        return std::nullopt;
    }

    int min_i = std::numeric_limits<int>::max();
    int min_j = std::numeric_limits<int>::max();
    int max_i = std::numeric_limits<int>::min();
    int max_j = std::numeric_limits<int>::min();

    for (const auto& c : corners) {
        min_i = std::min(min_i, c.i);
        min_j = std::min(min_j, c.j);
        max_i = std::max(max_i, c.i);
        max_j = std::max(max_j, c.j);
    }

    CheckerboardDetection detection;
    detection.corners = std::move(corners);
    detection.cells = std::move(cells);
    detection.cols = max_i - min_i + 1;
    detection.rows = max_j - min_j + 1;
    detection.tracking = false;
    detection.stable = false;

    return detection;
}

// ------------------------------------------------------------

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

// ------------------------------------------------------------

std::vector<GridCell> GridBuilder::makeGridCells(
    const std::vector<GridCorner>& corners
) const {
    std::vector<GridCell> cells;

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

        // Important order:
        // p00 -> p10 -> p11 -> p01
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

        cells.push_back(cell);
    }

    return cells;
}

// ------------------------------------------------------------

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

// ------------------------------------------------------------

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