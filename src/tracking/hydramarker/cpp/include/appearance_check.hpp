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
//           vs. the centre-minus-off-centre intensity
// A correct pose gives both close to +1; a one-cell identity shift flips the
// parity sign, a two-cell shift keeps the parity but lowers the dot term, a
// gross pose error gives about 0 for both.
struct AppearanceCheck {
    double parity = std::numeric_limits<double>::quiet_NaN();
    double dot = std::numeric_limits<double>::quiet_NaN();
    int cells = 0;          // sampled cells
    bool passed = false;    // both correlations finite and above the minimum
};

// frame: grey, BGR or BGRA. rvec/tvec: marker -> camera (mm). Fewer than
// min_cells visible cells never pass.
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
    int min_cells
);

} // namespace hydramarker
