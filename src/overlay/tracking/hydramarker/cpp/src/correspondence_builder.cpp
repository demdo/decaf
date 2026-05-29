#include "correspondence_builder.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace hydramarker {

namespace {

using LocalKey = std::pair<int, int>;
using GlobalKey = std::pair<int, int>;

struct AssignmentAccumulator {
    std::map<GlobalKey, int> votes;
    int total_votes = 0;

    void add(int global_row, int global_col) {
        const GlobalKey key(global_row, global_col);
        votes[key] += 1;
        total_votes += 1;
    }

    bool empty() const {
        return votes.empty();
    }

    int bestVotes() const {
        int best = 0;
        for (const auto& item : votes) {
            best = std::max(best, item.second);
        }
        return best;
    }

    int secondBestVotes() const {
        int best = 0;
        int second = 0;

        for (const auto& item : votes) {
            const int v = item.second;

            if (v > best) {
                second = best;
                best = v;
            } else if (v > second) {
                second = v;
            }
        }

        return second;
    }

    GlobalKey bestKey() const {
        GlobalKey best_key(-1, -1);
        int best = -1;

        for (const auto& item : votes) {
            if (item.second > best) {
                best = item.second;
                best_key = item.first;
            }
        }

        return best_key;
    }
};

bool isUsablePatch(const DecodedPatch& patch) {
    return patch.valid && !patch.ambiguous && patch.local.valid && patch.local.k > 0;
}

} // namespace

CorrespondenceBuilder::CorrespondenceBuilder(CorrespondenceBuilderConfig config)
    : config_(config)
{
}

int CorrespondenceBuilder::normalizeRotationDeg(int rotation_deg) {
    int r = rotation_deg % 360;
    if (r < 0) {
        r += 360;
    }

    if (r == 0 || r == 90 || r == 180 || r == 270) {
        return r;
    }

    const int allowed[4] = {0, 90, 180, 270};

    int best = allowed[0];
    int best_dist = std::abs(r - allowed[0]);

    for (int i = 1; i < 4; ++i) {
        const int d = std::abs(r - allowed[i]);
        if (d < best_dist) {
            best_dist = d;
            best = allowed[i];
        }
    }

    return best;
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

bool CorrespondenceBuilder::isBoundaryLocalCorner(
    int local_row,
    int local_col,
    int min_row,
    int max_row,
    int min_col,
    int max_col,
    int margin
) {
    const int m = std::max(0, margin);

    return (
        local_row <= min_row + m ||
        local_row >= max_row - m ||
        local_col <= min_col + m ||
        local_col >= max_col - m
    );
}

bool CorrespondenceBuilder::rotatedCornerOffset(
    int local_drow,
    int local_dcol,
    int k,
    int rotation_deg,
    int& global_drow,
    int& global_dcol
) {
    const int r = normalizeRotationDeg(rotation_deg);

    switch (r) {
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

    if (detection.corners.empty()) {
        return result;
    }

    int min_local_row = detection.corners.front().j;
    int max_local_row = detection.corners.front().j;
    int min_local_col = detection.corners.front().i;
    int max_local_col = detection.corners.front().i;

    for (const GridCorner& corner : detection.corners) {
        min_local_row = std::min(min_local_row, corner.j);
        max_local_row = std::max(max_local_row, corner.j);
        min_local_col = std::min(min_local_col, corner.i);
        max_local_col = std::max(max_local_col, corner.i);
    }

    std::map<int, int> rotation_votes;

    for (const DecodedPatch& patch : decoded_patches) {
        if (!isUsablePatch(patch)) {
            continue;
        }

        const int r = normalizeRotationDeg(patch.rotation_deg);
        rotation_votes[r] += 1;
    }

    int dominant_rotation = -1;
    int dominant_count = 0;
    int rotation_total = 0;

    for (const auto& item : rotation_votes) {
        rotation_total += item.second;

        if (item.second > dominant_count) {
            dominant_count = item.second;
            dominant_rotation = item.first;
        }
    }

    result.dominant_rotation_deg = dominant_rotation;
    result.dominant_rotation_count = dominant_count;
    result.rotation_vote_count = rotation_total;

    bool use_rotation_filter = false;

    if (
        config_.enable_dominant_rotation_filter &&
        rotation_total >= config_.min_rotation_support &&
        dominant_rotation >= 0
    ) {
        const double ratio =
            static_cast<double>(dominant_count) /
            std::max(1.0, static_cast<double>(rotation_total));

        use_rotation_filter = ratio >= config_.min_rotation_support_ratio;
    }

    std::map<LocalKey, AssignmentAccumulator> assignments;

    for (const DecodedPatch& patch : decoded_patches) {
        if (!isUsablePatch(patch)) {
            continue;
        }

        const int patch_rotation = normalizeRotationDeg(patch.rotation_deg);

        if (use_rotation_filter && patch_rotation != dominant_rotation) {
            result.decoded_patches_rejected_by_rotation += 1;
            continue;
        }

        const int k = patch.local.k;

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
                        patch_rotation,
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

                const LocalKey local_key(local_row, local_col);
                assignments[local_key].add(global_row, global_col);
            }
        }
    }

    for (const auto& item : assignments) {
        const LocalKey& local_key = item.first;
        const AssignmentAccumulator& acc = item.second;

        if (acc.empty()) {
            continue;
        }

        const int best_votes = acc.bestVotes();
        const int second_votes = acc.secondBestVotes();
        const int conflict_votes = acc.total_votes - best_votes;

        if (conflict_votes > 0) {
            result.assignments_conflicted += conflict_votes;
        }

        const int local_row = local_key.first;
        const int local_col = local_key.second;

        const bool is_boundary = isBoundaryLocalCorner(
            local_row,
            local_col,
            min_local_row,
            max_local_row,
            min_local_col,
            max_local_col,
            config_.boundary_margin_cells
        );

        const bool has_conflict = second_votes > 0;

        bool accept_single_vote_boundary = false;

        if (
            config_.allow_single_vote_boundary_corners &&
            is_boundary &&
            !has_conflict &&
            best_votes == 1
        ) {
            accept_single_vote_boundary = true;
        }

        if (best_votes < config_.min_votes && !accept_single_vote_boundary) {
            if (best_votes == 1 && !is_boundary) {
                result.single_vote_non_boundary_corners_rejected += 1;
            }
            continue;
        }

        if (accept_single_vote_boundary) {
            result.single_vote_boundary_corners_accepted += 1;
        }

        if (config_.discard_conflicts && second_votes >= best_votes) {
            continue;
        }

        const GlobalKey global_key = acc.bestKey();

        const int global_row = global_key.first;
        const int global_col = global_key.second;

        const GridCorner* local_corner = findLocalCorner(
            detection,
            local_row,
            local_col
        );

        if (local_corner == nullptr) {
            continue;
        }

        if (!geometry.hasCorner(global_row, global_col)) {
            continue;
        }

        Correspondence2D3D corr;
        corr.uv = local_corner->uv;
        corr.xyz_mm = geometry.cornerPoint(global_row, global_col);

        corr.local_row = local_row;
        corr.local_col = local_col;

        corr.global_row = global_row;
        corr.global_col = global_col;

        corr.votes = best_votes;

        result.correspondences.push_back(corr);
        result.assignments_accepted += best_votes;
    }

    return result;
}

} // namespace hydramarker