#pragma once

#include <cstdint>
#include <vector>

#include "dot_detector.hpp"
#include "geometry_utils.hpp"

namespace hydramarker {

struct LocalPatch {
    int row = 0;
    int col = 0;
    int k = 0;

    std::vector<uint8_t> bits;      // k*k, row-major
    std::vector<double> scores;     // k*k, row-major

    double mean_score = 0.0;

    bool geometry_valid = false;
    double geometry_quality = 0.0;

    geom::PatchGeometryValidation geometry;

    bool valid = false;
};

class PatchExtractor {
public:
    PatchExtractor() = default;

    std::vector<LocalPatch> extract(
        const DotDetectionResult& dots,
        int patch_size
    ) const;

private:
    static const DotCellObservation* findCell(
        const DotDetectionResult& dots,
        int row,
        int col
    );
};

} // namespace hydramarker