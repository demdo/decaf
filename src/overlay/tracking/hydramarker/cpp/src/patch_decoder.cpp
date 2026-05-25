#include "patch_decoder.hpp"

#include <stdexcept>

namespace hydramarker {

PatchDecoder::PatchDecoder(PatchDecoderConfig config)
    : config_(config)
{
}

DecodedPatch PatchDecoder::decodeOne(
    const LocalPatch& patch,
    const MarkerField& field
) const
{
    DecodedPatch decoded;
    decoded.local = patch;

    decoded.valid = false;
    decoded.ambiguous = false;
    decoded.global_row = -1;
    decoded.global_col = -1;
    decoded.rotation_deg = 0;
    decoded.num_matches = 0;
    decoded.confidence = 0.0;

    if (field.empty()) {
        return decoded;
    }

    if (!patch.valid) {
        return decoded;
    }

    if (config_.require_geometry_valid && !patch.geometry_valid) {
        return decoded;
    }

    if (patch.k != field.patchSize()) {
        throw std::runtime_error(
            "PatchDecoder: local patch size does not match MarkerField patch size."
        );
    }

    if (patch.bits.size() != static_cast<size_t>(patch.k * patch.k)) {
        throw std::runtime_error(
            "PatchDecoder: local patch bits have wrong size."
        );
    }

    const std::vector<PatchMatch> matches = field.findPatch(patch.bits);

    decoded.num_matches = static_cast<int>(matches.size());

    if (matches.empty()) {
        return decoded;
    }

    if (matches.size() > 1) {
        decoded.ambiguous = true;

        if (!config_.accept_ambiguous) {
            return decoded;
        }
    }

    const PatchMatch& match = matches.front();

    decoded.valid = true;
    decoded.ambiguous = matches.size() > 1;

    decoded.global_col = match.x;
    decoded.global_row = match.y;
    decoded.rotation_deg = match.rotation_deg;

    decoded.confidence = decoded.ambiguous ? 0.5 : 1.0;

    return decoded;
}

std::vector<DecodedPatch> PatchDecoder::decode(
    const std::vector<LocalPatch>& patches,
    const MarkerField& field
) const
{
    std::vector<DecodedPatch> decoded;
    decoded.reserve(patches.size());

    for (const LocalPatch& patch : patches) {
        decoded.push_back(decodeOne(patch, field));
    }

    return decoded;
}

} // namespace hydramarker