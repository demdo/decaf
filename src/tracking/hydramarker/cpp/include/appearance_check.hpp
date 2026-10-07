#pragma once

#include <limits>
#include <vector>

#include <opencv2/core.hpp>

#include "marker_field.hpp"
#include "marker_geometry.hpp"

namespace hydramarker {

// Photometric check of a marker pose, independent of the decode that produced
// it. The model cells facing the camera are projected with the pose and two
// correlations are measured over them:
//   parity: expected base colour ((row + col) % 2) vs. the image intensity at
//           four off-centre points per cell (outside the dot)
//   dot:    expected centre colour (base XOR field bit) relative to the base
//           vs. the centre-minus-off-centre intensity, correlated inside each
//           base colour (needs both colours with enough cells)
// A correct pose gives both close to +1; a one-cell identity shift flips the
// parity sign, a two-cell shift keeps the parity but drops the dot term to
// about 0, a gross pose error gives about 0 for both.
struct AppearanceCheck {
    double parity = std::numeric_limits<double>::quiet_NaN();
    double dot = std::numeric_limits<double>::quiet_NaN();
    int cells = 0;          // sampled cells (used cells when the gate is on)
    int cells_all = 0;      // sampled cells before the occlusion gate
    // cells that pass the occlusion gate, also when it is not applied
    // (gate_apply false); = cells_all without a configured gate
    int cells_gated = 0;
    bool passed = false;    // both correlations finite and above the minimum
    bool failed = false;    // enough cells and a finite correlation below it
};

// frame: grey, BGR or BGRA. rvec/tvec: marker -> camera (mm). Fewer than
// min_cells visible cells never pass and never fail (undecided), nor does
// a correlation without variance. The dot term needs at least 8 cells of
// each base colour, so a check passes only with 16 cells or more.
AppearanceCheck checkMarkerAppearance(
    const cv::Mat& frame,
    const MarkerGeometry& geometry,
    const MarkerField& field,
    const cv::Matx33d& K,
    const std::vector<double>& dist_coeffs,
    const std::vector<double>& rvec,
    const std::vector<double>& tvec,
    double min_parity,
    double min_dot,
    int min_cells,
    // Occlusion gate (off when both are <= 0): a cell is used only when its
    // off-centre mean differs from the median off-centre mean of its sampled
    // 4-neighbours (at least 2) by max(gate_min_dn, gate_frac * p90 of those
    // differences) -- i.e. it shows checker contrast at the projected place.
    // A finger or glove over the marker has no such contrast and drops out
    // instead of pulling both correlations towards zero; a whole-cell
    // identity shift keeps the contrast and still fails.
    double gate_min_dn = 0.0,
    double gate_frac = 0.0,
    // false: only count the cells that pass the gate (cells_gated), the
    // correlations use every sampled cell
    bool gate_apply = true
);

} // namespace hydramarker
