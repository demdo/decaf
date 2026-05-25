#include "correspondence_builder.hpp"

#include <map>
#include <stdexcept>
#include <utility>

namespace hydramarker {

namespace {

struct AssignmentAccumulator {
    bool has_assignment = false;
    bool conflict = false;

    int global_row = -1;
    int global_col = -1;

    int votes = 0;
};

using LocalKey = std::pair<int, int>;

} // namespace

CorrespondenceBuilder::CorrespondenceBuilder(CorrespondenceBuilderConfig config)
    : config_(config)
{
}

const GridCorner* CorrespondenceBuilder::findLocalCorner(
    const CheckerboardDetection& detection,
    int row,
    int col
) {
    for (const GridCorner& corner : detection.corners) {
        /*
         * Patch/Dot coordinates:
         *   row = vertical coordinate
         *   col = horizontal coordinate
         *
         * Checkerboard/Grid coordinates:
         *   i = horizontal coordinate
         *   j = vertical coordinate
         *
         * Therefore:
         *   local row == corner.j
         *   local col == corner.i
         */
        if (corner.j == row && corner.i == col) {
            return &corner;
        }
    }

    return nullptr;
}

bool CorrespondenceBuilder::rotatedCornerOffset(
    int local_drow,
    int local_dcol,
    int k,
    int rotation_deg,
    int& global_drow,
    int& global_dcol
) {
    switch (rotation_deg) {
    case 0:
        global_drow = local_drow;
        global_dcol = local_dcol;
        return true;

    case 90:
        global_drow = local_dcol;
        global_dcol = k - local_drow;
        return true;

    case 180:
        global_drow = k - local_drow;
        global_dcol = k - local_dcol;
        return true;

    case 270:
        global_drow = k - local_dcol;
        global_dcol = local_drow;
        return true;

    default:
        return false;
    }
}

CorrespondenceBuildResult CorrespondenceBuilder::build(
    const CheckerboardDetection& detection,
    const std::vector<DecodedPatch>& decoded_patches,
    const MarkerGeometry& geometry
) const {
    CorrespondenceBuildResult result;

    if (!detection.valid()) {
        return result;
    }

    if (config_.require_detection_stable && !detection.stable) {
        return result;
    }

    if (geometry.empty()) {
        throw std::runtime_error("CorrespondenceBuilder: MarkerGeometry is empty.");
    }

    std::map<LocalKey, AssignmentAccumulator> assignments;

    for (const DecodedPatch& patch : decoded_patches) {
        if (!patch.valid || patch.ambiguous) {
            continue;
        }

        if (!patch.local.valid) {
            continue;
        }

        const int k = patch.local.k;

        if (k <= 0) {
            continue;
        }

        result.decoded_patches_used += 1;

        // A k x k cell patch contains (k+1) x (k+1) grid corners.
        for (int drow = 0; drow <= k; ++drow) {
            for (int dcol = 0; dcol <= k; ++dcol) {
                int global_drow = 0;
                int global_dcol = 0;

                if (!rotatedCornerOffset(
                        drow,
                        dcol,
                        k,
                        patch.rotation_deg,
                        global_drow,
                        global_dcol)) {
                    continue;
                }

                const int local_row = patch.local.row + drow;
                const int local_col = patch.local.col + dcol;

                const int global_row = patch.global_row + global_drow;
                const int global_col = patch.global_col + global_dcol;

                result.assignments_total += 1;

                if (!geometry.hasCorner(global_row, global_col)) {
                    result.corners_without_geometry += 1;
                    continue;
                }

                const LocalKey key(local_row, local_col);
                AssignmentAccumulator& acc = assignments[key];

                if (!acc.has_assignment) {
                    acc.has_assignment = true;
                    acc.global_row = global_row;
                    acc.global_col = global_col;
                    acc.votes = 1;
                    result.assignments_accepted += 1;
                    continue;
                }

                if (acc.global_row == global_row && acc.global_col == global_col) {
                    acc.votes += 1;
                    result.assignments_accepted += 1;
                } else {
                    acc.conflict = true;
                    result.assignments_conflicted += 1;
                }
            }
        }
    }

    for (const auto& item : assignments) {
        const LocalKey& key = item.first;
        const AssignmentAccumulator& acc = item.second;

        if (!acc.has_assignment) {
            continue;
        }

        if (acc.conflict && config_.discard_conflicts) {
            continue;
        }

        if (acc.votes < config_.min_votes) {
            continue;
        }

        const int local_row = key.first;
        const int local_col = key.second;

        const GridCorner* local_corner = findLocalCorner(
            detection,
            local_row,
            local_col
        );

        if (local_corner == nullptr) {
            continue;
        }

        if (!geometry.hasCorner(acc.global_row, acc.global_col)) {
            continue;
        }

        Correspondence2D3D corr;
        corr.uv = local_corner->uv;
        corr.xyz_mm = geometry.cornerPoint(acc.global_row, acc.global_col);

        corr.local_row = local_row;
        corr.local_col = local_col;

        corr.global_row = acc.global_row;
        corr.global_col = acc.global_col;

        corr.votes = acc.votes;

        result.correspondences.push_back(corr);
    }

    return result;
}

} // namespace hydramarker