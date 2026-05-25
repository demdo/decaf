#pragma once

#include <vector>

#include "marker_field.hpp"
#include "patch_extractor.hpp"

namespace hydramarker {

struct PatchDecoderConfig {
    bool require_geometry_valid = true;
    bool accept_ambiguous = false;
};

struct DecodedPatch {
    LocalPatch local;

    bool valid = false;
    bool ambiguous = false;

    int global_row = -1;
    int global_col = -1;

    int rotation_deg = 0;

    int num_matches = 0;

    double confidence = 0.0;
};

class PatchDecoder {
public:
    PatchDecoder() = default;
    explicit PatchDecoder(PatchDecoderConfig config);

    DecodedPatch decodeOne(
        const LocalPatch& patch,
        const MarkerField& field
    ) const;

    std::vector<DecodedPatch> decode(
        const std::vector<LocalPatch>& patches,
        const MarkerField& field
    ) const;

private:
    PatchDecoderConfig config_;
};

} // namespace hydramarker