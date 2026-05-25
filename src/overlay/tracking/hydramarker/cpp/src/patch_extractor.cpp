#include "patch_extractor.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace hydramarker {

namespace {

using CellKey = std::pair<int, int>;

CellKey makeKey(int row, int col)
{
    return CellKey(row, col);
}

} // namespace

std::vector<LocalPatch> PatchExtractor::extract(
    const DotDetectionResult& dots,
    int patch_size
) const
{
    if (patch_size <= 0) {
        throw std::runtime_error("PatchExtractor: patch_size must be positive.");
    }

    std::vector<LocalPatch> patches;

    if (dots.cells.empty()) {
        return patches;
    }

    std::map<CellKey, const DotCellObservation*> cell_lookup;

    int min_row = std::numeric_limits<int>::max();
    int max_row = std::numeric_limits<int>::lowest();
    int min_col = std::numeric_limits<int>::max();
    int max_col = std::numeric_limits<int>::lowest();

    for (const DotCellObservation& cell : dots.cells) {
        cell_lookup[makeKey(cell.row, cell.col)] = &cell;

        min_row = std::min(min_row, cell.row);
        max_row = std::max(max_row, cell.row);
        min_col = std::min(min_col, cell.col);
        max_col = std::max(max_col, cell.col);
    }

    if (min_row > max_row || min_col > max_col) {
        return patches;
    }

    const int observed_rows = max_row - min_row + 1;
    const int observed_cols = max_col - min_col + 1;

    if (observed_rows < patch_size || observed_cols < patch_size) {
        return patches;
    }

    for (int row0 = min_row; row0 <= max_row - patch_size + 1; ++row0) {
        for (int col0 = min_col; col0 <= max_col - patch_size + 1; ++col0) {

            LocalPatch patch;
            patch.row = row0;
            patch.col = col0;
            patch.k = patch_size;
            patch.valid = true;

            patch.bits.reserve(patch_size * patch_size);
            patch.scores.reserve(patch_size * patch_size);

            std::vector<std::array<cv::Point2f, 4>> patch_cell_quads;
            patch_cell_quads.reserve(patch_size * patch_size);

            double score_sum = 0.0;
            int score_count = 0;

            for (int r = 0; r < patch_size; ++r) {
                for (int c = 0; c < patch_size; ++c) {
                    const int cell_row = row0 + r;
                    const int cell_col = col0 + c;

                    const auto it = cell_lookup.find(makeKey(cell_row, cell_col));

                    if (it == cell_lookup.end()) {
                        patch.valid = false;
                        break;
                    }

                    const DotCellObservation* cell = it->second;

                    if (cell == nullptr ||
                        !cell->valid ||
                        cell->ambiguous)
                    {
                        patch.valid = false;
                        break;
                    }

                    patch.bits.push_back(cell->has_dot ? 1u : 0u);
                    patch.scores.push_back(cell->score);
                    patch_cell_quads.push_back(cell->corners_uv);

                    score_sum += cell->score;
                    ++score_count;
                }

                if (!patch.valid) {
                    break;
                }
            }

            if (!patch.valid) {
                continue;
            }

            if (score_count != patch_size * patch_size) {
                continue;
            }

            patch.mean_score = score_sum / static_cast<double>(score_count);

            patch.geometry =
                geom::validatePatchGeometry(patch_cell_quads);

            patch.geometry_valid = patch.geometry.valid;
            patch.geometry_quality =
                static_cast<double>(patch.geometry.quality);

            patches.push_back(std::move(patch));
        }
    }

    return patches;
}

const DotCellObservation* PatchExtractor::findCell(
    const DotDetectionResult& dots,
    int row,
    int col
)
{
    for (const auto& cell : dots.cells) {
        if (cell.row == row && cell.col == col) {
            return &cell;
        }
    }

    return nullptr;
}

} // namespace hydramarker