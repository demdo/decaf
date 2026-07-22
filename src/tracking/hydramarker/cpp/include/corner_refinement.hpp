#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "marker_geometry.hpp"

namespace hydramarker {

// Full marker pattern in SURFACE coordinates for synthetic-template
// rendering: checkerboard parity + coded dots (HydraMarker field bits).
// Conventions validated against a real frame (corr +0.73, counter-
// hypothesis -0.73): white = (cell_row + cell_col) % 2 == 0 is BLACK,
// i.e. color = (r + c) % 2; a set field bit draws a dot of the INVERTED
// cell color (radius dot_radius_rel of the cell); the column index runs
// AGAINST the arc direction (s grid descending -> stored negated).
struct MarkerPatternLut {
    std::vector<double> u_nodes;      // ascending axial corner coords (mm)
    std::vector<double> s_neg_nodes;  // s_sign * arc coords, ascending, in
                                      // column-index order
    std::vector<uint8_t> bits;        // row-major (cell_rows x cell_cols)
    int cell_rows = 0;
    int cell_cols = 0;
    double dot_radius_rel = 0.22;
    double s_sign = -1.0;             // sign that makes the arc grid
                                      // ascending in column-index order

    bool valid() const {
        return cell_rows > 0 && cell_cols > 0 &&
               static_cast<int>(u_nodes.size()) == cell_rows + 1 &&
               static_cast<int>(s_neg_nodes.size()) == cell_cols + 1 &&
               static_cast<int>(bits.size()) == cell_rows * cell_cols;
    }

    // Marker color at the surface coordinate (u = axial mm, s = arc mm):
    // 1 = white, 0 = black, -1 = outside the printed grid.
    int colorAt(double u, double s) const {
        const auto iu = std::upper_bound(u_nodes.begin(), u_nodes.end(), u);
        const int r = static_cast<int>(iu - u_nodes.begin()) - 1;
        if (r < 0 || r >= cell_rows) return -1;
        const double sn = s_sign * s;
        const auto is =
            std::upper_bound(s_neg_nodes.begin(), s_neg_nodes.end(), sn);
        const int c = static_cast<int>(is - s_neg_nodes.begin()) - 1;
        if (c < 0 || c >= cell_cols) return -1;
        int col = (r + c) % 2;   // validated parity: (r+c)%2 = white
        const double fu = (u - u_nodes[static_cast<size_t>(r)]) /
                          (u_nodes[static_cast<size_t>(r) + 1] -
                           u_nodes[static_cast<size_t>(r)]);
        const double fs = (sn - s_neg_nodes[static_cast<size_t>(c)]) /
                          (s_neg_nodes[static_cast<size_t>(c) + 1] -
                           s_neg_nodes[static_cast<size_t>(c)]);
        if (bits[static_cast<size_t>(r * cell_cols + c)] != 0) {
            const double du = fu - 0.5;
            const double ds = fs - 0.5;
            if (du * du + ds * ds < dot_radius_rel * dot_radius_rel) {
                col = 1 - col;
            }
        }
        return col;
    }
};

// Per-frame model context for the quadratic_form corner measurement.  Filled
// by the tracker engine; the subpix operator ignores it.  All poses map
// marker-frame points X (mm) to camera-frame points R*X + t.
struct CornerModelContext {
    // Camera intrinsics shared by the current and reference view.
    cv::Matx33d K = cv::Matx33d::eye();
    std::vector<double> dist;

    // Predicted pose of the current frame (previous accepted pose is fine:
    // translation errors cancel in the anchor+shift construction and only
    // second-order warp-shape errors remain).
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d t{0.0, 0.0, 0.0};

    // Surface the marker sits on (marker frame, mm).
    SurfaceModel surface;

    // Per-tracked-point 3D anchors: the marker corner coordinate (mm) for
    // each entry of the point vector handed to refineTrackedCorners.
    // Points without a known identity get anchor_valid = 0 and fall back to
    // the subpix measurement.
    std::vector<cv::Vec3d> anchor_xyz_mm;
    std::vector<uint8_t> anchor_valid;

    // Optional pixel->normalized-ray lookup table (CV_32FC2, full frame,
    // undistorted normalized coordinates per pixel centre).  When present it
    // replaces the iterative cv::undistortPoints call per corner window.
    cv::Mat ray_lut;

    // Optional full marker pattern: the saddle-warp then renders the REAL
    // pattern (parity + coded dots, unlimited support) instead of the
    // 4-cell quadrant approximation.
    const MarkerPatternLut* pattern = nullptr;

    // The quadratic-form operator is reference-free: it only needs the
    // surface model, the pose prediction and the per-point anchors.
    bool usableQuadratic(size_t n_points) const {
        return surface.valid() &&
               anchor_xyz_mm.size() == n_points &&
               anchor_valid.size() == n_points;
    }
};

// Engine-side per-frame input for the corner measurement.  The tracker
// engine fills this before the detector runs; the detector resolves the
// per-point anchors (matching its tracked points against the previous
// accepted frame's pose corners by their previous-frame uv) and builds the
// CornerModelContext from it.  enabled = false disables the operator for
// the frame (subpix runs as usual).
struct CornerModelFrameInput {
    bool enabled = false;

    cv::Matx33d K = cv::Matx33d::eye();
    std::vector<double> dist;

    // Predicted pose of the current frame (= last accepted pose; pose
    // translation errors cancel in the anchor+shift construction).
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d t{0.0, 0.0, 0.0};

    SurfaceModel surface;

    // Optional pixel->ray lookup table (see CornerModelContext::ray_lut).
    cv::Mat ray_lut;

    // Pose corners of the last accepted frame: uv in that frame plus the
    // marker-frame corner coordinate.  Used to give tracked points their
    // 3D anchor identity.
    std::vector<cv::Point2f> prev_uv;
    std::vector<cv::Vec3d> prev_xyz_mm;
};

// Result statistics of refineTrackedCorners, mirrored by the detector into
// its timing/diagnostic map under the established tracking_subpix_* keys.
// The corner fields stay empty/zero when the subpix operator ran.
struct TrackedRefineStats {
    bool enabled = false;
    int refined_count = 0;
    double mean_shift_px = 0.0;
    double p95_shift_px = 0.0;
    double max_shift_px = 0.0;

    std::vector<uint8_t> corner_ok;   // per input point, empty if unused
    std::vector<float> corner_zncc;   // per input point, empty if unused

    // Quadratic-form operator statistics.  corner_ok/corner_zncc
    // double as the generic per-point ok/quality channels (ok = the operator
    // replaced the subpix baseline, quality in [0,1]).
    int qf_count = 0;              // corners replaced by curve intersection
    int qf_row_curves = 0;         // fitted row curves (line or conic)
    int qf_row_conics = 0;         // of those, true conic fits
    int qf_col_lines = 0;          // fitted column lines
    double qf_mean_dev_px = 0.0;   // mean |qf - subpix| over replaced points
    double qf_mean_edge_pts = 0.0; // mean accepted edge points per curve
    int qf_saddle_count = 0;       // high-incidence corners measured by the
                                   // synthetic saddle registration

    // Per-point measurement snapshots, filled when quadratic_form or
    // quadratic_form is the active operator: the incoming LK position and
    // the subpix baseline the measurement overwrote.  Lets a run log both
    // operators on the SAME frames for offline comparison.
    std::vector<cv::Point2f> input_uv;    // before any refinement (LK)
    std::vector<cv::Point2f> subpix_uv;   // after the subpix snap
};

struct RefinedCorner {
    cv::Point2f uv;

    // Samu/ReadMarker-style local checkerboard structure features.
    // Angles are in degrees.
    cv::Vec2f ledge_angles_deg = cv::Vec2f(0.0f, 0.0f);

    float correlation = 0.0f;
    float angle_bias_deg = 0.0f;
    bool valid = false;
};

struct CornerRefinementConfig {
    int radius = 5;
    int iterations = 2;

    float max_angle_bias_deg = 20.0f;
    float correlation_drop = 0.2f;

    float merge_radius_px = 2.0f;

    // Quadrant intensity symmetry filter — see CheckerboardDetectorConfig for
    // full documentation. Set quadrant_half_r = 0 to disable.
    int   quadrant_half_r = 3;
    float quadrant_min_contrast = 12.0f;       // in [0,255]; internally scaled relative to local range
    float quadrant_max_diagonal_diff = 60.0f;  // in [0,255]; relaxed — relative scaling makes it robust

    // cv::cornerSubPix window half-size applied after gradient-intersection
    // refinement and before saddle-feature computation.
    //
    // This extra subpixel step is particularly beneficial under motion blur
    // and defocus: the gradient-intersection solver converges to a rough
    // position, then cornerSubPix iterates to the nearest local gradient
    // minimum with sub-pixel accuracy.
    //
    // -1 (default): automatically set to max(3, radius - 1), which gives a
    //   window large enough to see the gradient transition but small enough
    //   not to cross into adjacent cells on small markers.
    //  0: disabled.
    // >0: explicit half-size in pixels.
    int subpix_win_size = -1;

    // Maximum number of cornerSubPix iterations.
    int subpix_max_iters = 20;

    // cornerSubPix convergence epsilon (pixels).
    double subpix_epsilon = 0.05;

    // Measurement operator used by refineTrackedCorners on LK-tracked points:
    //   "subpix"     - cv::cornerSubPix snap (default, established behaviour)
    //   "quadratic_form" - reference-free curve-intersection measurement
    //                  (Wang et al., IEEE TIM 2022): sigmoid edge-profile
    //                  fits along the grid lines, weighted conic/line fits
    //                  in undistorted normalized coordinates, corner = row
    //                  curve x column line.  Per-point fallback to subpix.
    std::string tracked_refine_method = "subpix";

    // ---- synthetic-saddle template registration (runSaddleWarp) ----
    // Reused by the high-incidence 2D saddle snap that overrides the 1D
    // quadratic-form result on strongly foreshortened corners.
    int saddle_max_iters = 12;            // LK iterations
    double saddle_min_zncc = 0.6;         // template/image match gate
    // Maximum deviation of the registered position from the LK/subpix
    // estimate the registration was seeded with.  The seed already follows
    // the frame-to-frame motion, so this gate stays tight without rejecting
    // fast motion (the old absolute-shift gate killed whole frames as soon
    // as the marker moved faster than the gate).
    double saddle_max_shift_px = 4.0;

    // ---- quadratic_form parameters ----
    // Edge profiles: samples along the edge normal, sigmoid fit per profile.
    double qf_profile_half_px = 4.5;   // max half-length of a profile
    int    qf_profile_samples = 13;    // samples per profile (odd)
    double qf_min_contrast = 12.0;     // gray levels; below = no edge (coded
                                       // cells) -> profile skipped
    double qf_max_profile_rms = 0.35;  // contrast-normalized sigmoid fit rms
    // Along-edge sampling: keep away from the X-junctions at the corners.
    double qf_junction_margin_frac = 0.18;
    double qf_junction_margin_min_px = 2.0;
    // Segment admission: predicted-vs-measured endpoint mismatch cap and
    // the allowed corner gap relative to the median cell pitch.
    double qf_max_endpoint_corr_px = 60.0;
    double qf_max_gap_pitch_ratio = 1.6;
    // Curve fits: minimum accepted edge points, conic preference and
    // residual gates (px equivalent).
    int    qf_min_row_points = 6;
    int    qf_min_col_points = 3;
    int    qf_min_conic_points = 8;
    double qf_conic_gain = 0.95;       // curved must beat line rms by this
    double qf_max_fit_rms_px = 0.6;
    // Final acceptance: deviation of the intersection from the subpix seed.
    double qf_max_dev_px = 2.5;
    // Synthetic saddle registration for HIGH-INCIDENCE corners (deg): 1D
    // edge profiles fail under strong foreshortening (fb1 near-camera
    // window: subpix AND qf drift mm-scale while the 2D reference warp
    // holds).  Corners steeper than this get a perspective-rendered
    // quadrant template (marker code free, reference free) registered
    // <= 0 disables.
    double qf_saddle_min_incidence_deg = 55.0;
};

class CornerRefiner {
public:
    CornerRefiner();

    std::vector<RefinedCorner> refine(
        const cv::Mat& gray,
        const std::vector<cv::Point2f>& candidates,
        const cv::Mat& grad_x,
        const cv::Mat& grad_y,
        const CornerRefinementConfig& config
    ) const;

    // Measurement step for LK-tracked corners: snaps each non-predicted
    // point to the current-frame corner position before the tracked
    // detection is rebuilt and handed to PnP.  Points are updated in place;
    // entries flagged as predicted (or too close to the image border, or
    // rejected by the operator) keep their incoming position.
    //
    // The operator is selected by config.tracked_refine_method.  The
    // model_context is only used by the quadratic_form operator and may be
    // null for "subpix".
    TrackedRefineStats refineTrackedCorners(
        const cv::Mat& gray,
        std::vector<cv::Point2f>& points,
        const std::vector<bool>& predicted,
        const CornerRefinementConfig& config,
        const CornerModelContext* model_context = nullptr
    ) const;

private:
    // cornerSubPix snap on the filtered tracked points (baseline operator).
    void runSubpixSnap(
        const cv::Mat& gray,
        std::vector<cv::Point2f>& points,
        const std::vector<bool>& predicted,
        const CornerRefinementConfig& config,
        TrackedRefineStats& stats
    ) const;

    // Reference-free quadratic-form measurement pass: edge points from
    // sigmoid profile fits along the surface grid lines, weighted conic /
    // line fits in undistorted normalized coordinates, corner = row curve
    // intersected with column line.  Successful measurements overwrite the
    // subpix baseline in-place; gated points keep the baseline.
    void runQuadraticForm(
        const cv::Mat& gray,
        std::vector<cv::Point2f>& points,
        const std::vector<bool>& predicted,
        const CornerRefinementConfig& config,
        const CornerModelContext& ctx,
        TrackedRefineStats& stats
    ) const;

    // Synthetic-template registration for high-incidence corners: the
    // local checkerboard saddle is RENDERED through the surface model
    // (quadrant sign in surface coordinates, dot regions masked) and
    // registered with the same translation-only LK + gain/bias as
    // Reference-free 2D measurement where 1D edge profiles
    // are geometrically blind.
    void runSaddleWarp(
        const cv::Mat& gray,
        std::vector<cv::Point2f>& points,
        const std::vector<bool>& predicted,
        const CornerRefinementConfig& config,
        const CornerModelContext& ctx,
        TrackedRefineStats& stats
    ) const;

    std::vector<cv::Point2f> refineGradientIntersections(
        const std::vector<cv::Point2f>& candidates,
        const cv::Mat& grad_x,
        const cv::Mat& grad_y,
        int radius,
        int iterations
    ) const;

    std::vector<RefinedCorner> computeSaddleFeatures(
        const cv::Mat& gray,
        const std::vector<cv::Point2f>& points,
        const CornerRefinementConfig& config
    ) const;

    std::vector<RefinedCorner> filterBySaddleScore(
        const std::vector<RefinedCorner>& corners,
        const CornerRefinementConfig& config
    ) const;

    std::vector<RefinedCorner> mergeCloseCorners(
        const std::vector<RefinedCorner>& corners,
        float merge_radius_px
    ) const;

    // Returns false if the candidate clearly lacks checkerboard quadrant
    // symmetry (e.g. dot centre, cell interior, edge crossing).
    // Operates directly on the original gray image so the test is independent
    // of the saddle model.
    static bool passesQuadrantSymmetry(
        const cv::Mat& gray_f,
        const cv::Point2f& uv,
        int half_r,
        float min_contrast,
        float max_diagonal_diff
    );
};

} // namespace hydramarker