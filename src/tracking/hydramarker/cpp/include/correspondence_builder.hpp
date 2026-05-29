#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "checkerboard_types.hpp"
#include "marker_geometry.hpp"
#include "patch_decoder.hpp"

namespace hydramarker {

struct Correspondence2D3D {
    cv::Point2f uv;
    cv::Point3f xyz_mm;

    int local_row = -1;
    int local_col = -1;

    int global_row = -1;
    int global_col = -1;

    int votes = 0;
};

struct CorrespondenceBuilderConfig {
    int min_votes = 2;

    bool discard_conflicts = true;
    bool require_detection_stable = false;

    bool enable_dominant_rotation_filter = true;
    int min_rotation_support = 2;
    double min_rotation_support_ratio = 0.55;

    // Allow edge/boundary corners with only one vote if there is no conflict.
    // This keeps the global robustness of min_votes=2 while recovering the
    // outermost marker corners, which are naturally supported by fewer patches.
    bool allow_single_vote_boundary_corners = true;
    int boundary_margin_cells = 0;
};

struct CorrespondenceBuildResult {
    std::vector<Correspondence2D3D> correspondences;

    int decoded_patches_used = 0;
    int decoded_patches_rejected_by_rotation = 0;

    int assignments_total = 0;
    int assignments_accepted = 0;
    int assignments_conflicted = 0;
    int corners_without_geometry = 0;

    int single_vote_boundary_corners_accepted = 0;
    int single_vote_non_boundary_corners_rejected = 0;

    int dominant_rotation_deg = -1;
    int dominant_rotation_count = 0;
    int rotation_vote_count = 0;

    bool valid() const {
        return !correspondences.empty();
    }
};

class CorrespondenceBuilder {
public:
    CorrespondenceBuilder() = default;
    explicit CorrespondenceBuilder(CorrespondenceBuilderConfig config);

    CorrespondenceBuildResult build(
        const CheckerboardDetection& detection,
        const std::vector<DecodedPatch>& decoded_patches,
        const MarkerGeometry& geometry
    ) const;

private:
    CorrespondenceBuilderConfig config_;

    static int normalizeRotationDeg(int rotation_deg);

    static bool rotatedCornerOffset(
        int local_drow,
        int local_dcol,
        int k,
        int rotation_deg,
        int& global_drow,
        int& global_dcol
    );

    static const GridCorner* findLocalCorner(
        const CheckerboardDetection& detection,
        int row,
        int col
    );

    static bool isBoundaryLocalCorner(
        int local_row,
        int local_col,
        int min_row,
        int max_row,
        int min_col,
        int max_col,
        int margin
    );
};

} // namespace hydramarker