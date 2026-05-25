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
    int min_votes = 1;

    // Strict mode:
    // If one local corner receives two different global assignments,
    // discard this local corner completely.
    bool discard_conflicts = true;

    bool require_detection_stable = false;
};

struct CorrespondenceBuildResult {
    std::vector<Correspondence2D3D> correspondences;

    int decoded_patches_used = 0;
    int assignments_total = 0;
    int assignments_accepted = 0;
    int assignments_conflicted = 0;
    int corners_without_geometry = 0;

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
};

} // namespace hydramarker