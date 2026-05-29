#pragma once

#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "checkerboard_types.hpp"
#include "lattice_model.hpp"

namespace hydramarker {

class GridBuilder {
public:
    GridBuilder();

    std::optional<CheckerboardDetection> build(
        const LatticeResult& lattice,
        float duplicate_dist_px,
        int min_corners,
        int min_cells
    ) const;

    std::optional<CheckerboardDetection> buildFromCorners(
        const std::vector<GridCorner>& input_corners,
        float duplicate_dist_px,
        int min_corners,
        int min_cells,
        bool tracking,
        bool stable
    ) const;

private:
    std::vector<GridCorner> makeGridCorners(
        const LatticeResult& lattice,
        float duplicate_dist_px
    ) const;

    std::vector<GridCorner> sanitizeGridCorners(
        const std::vector<GridCorner>& input_corners,
        float duplicate_dist_px
    ) const;

    std::vector<GridCell> makeGridCells(
        const std::vector<GridCorner>& corners
    ) const;

    int findCornerIndex(
        const std::vector<GridCorner>& corners,
        int i,
        int j
    ) const;

    bool isDuplicate(
        const std::vector<GridCorner>& corners,
        const cv::Point2f& uv,
        float duplicate_dist_px
    ) const;
};

} // namespace hydramarker
