#include "patch_decoder.hpp"

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace hydramarker {

namespace {

int normalizeRotationDeg(int rotation_deg) {
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

std::vector<PatchMatch> normalizeAndDeduplicateMatches(
    const std::vector<PatchMatch>& matches
) {
    std::vector<PatchMatch> cleaned;
    cleaned.reserve(matches.size());

    for (PatchMatch m : matches) {
        m.rotation_deg = normalizeRotationDeg(m.rotation_deg);
        cleaned.push_back(m);
    }

    std::sort(
        cleaned.begin(),
        cleaned.end(),
        [](const PatchMatch& a, const PatchMatch& b) {
            return std::tie(a.y, a.x, a.rotation_deg)
                 < std::tie(b.y, b.x, b.rotation_deg);
        }
    );

    cleaned.erase(
        std::unique(
            cleaned.begin(),
            cleaned.end(),
            [](const PatchMatch& a, const PatchMatch& b) {
                return a.x == b.x
                    && a.y == b.y
                    && normalizeRotationDeg(a.rotation_deg)
                        == normalizeRotationDeg(b.rotation_deg);
            }
        ),
        cleaned.end()
    );

    return cleaned;
}

} // namespace

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

    if (patch.k <= 0) {
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

    const std::vector<PatchMatch> raw_matches = field.findPatch(patch.bits);
    const std::vector<PatchMatch> matches =
        normalizeAndDeduplicateMatches(raw_matches);

    decoded.num_matches = static_cast<int>(matches.size());

    if (matches.empty()) {
        return decoded;
    }

    decoded.ambiguous = matches.size() > 1;

    if (decoded.ambiguous && !config_.accept_ambiguous) {
        return decoded;
    }

    const PatchMatch& match = matches.front();

    decoded.valid = true;
    decoded.global_col = match.x;
    decoded.global_row = match.y;
    decoded.rotation_deg = normalizeRotationDeg(match.rotation_deg);

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