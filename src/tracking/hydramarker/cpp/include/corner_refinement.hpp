#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "marker_geometry.hpp"

namespace hydramarker {

// Per-frame model context for forward-model ("model_warp") corner
// measurement.  Filled by the tracker engine; the subpix operator ignores it.
//
// All poses map marker-frame points X (mm) to camera-frame points R*X + t.
// The reference template source (A1 "ref-warp") is a full gray frame taken
// at enrollment together with its pose; template pixels are produced by
// intersecting current-frame pixel rays with the surface model and
// projecting the surface points into the reference view.
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

    // Reference view (A1 template source).  CV_32F preferred (avoids a
    // per-frame conversion); CV_8U is converted on the fly.
    cv::Mat ref_gray;
    cv::Matx33d R_ref = cv::Matx33d::eye();
    cv::Vec3d t_ref{0.0, 0.0, 0.0};

    // Optional pixel->normalized-ray lookup table (CV_32FC2, full frame,
    // undistorted normalized coordinates per pixel centre).  When present it
    // replaces the iterative cv::undistortPoints call per corner window.
    cv::Mat ray_lut;

    bool usable(size_t n_points) const {
        return surface.valid() &&
               !ref_gray.empty() &&
               anchor_xyz_mm.size() == n_points &&
               anchor_valid.size() == n_points;
    }
};

// Engine-side per-frame input for the model-warp measurement.  The tracker
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

    // Reference view captured at enrollment (CV_32F preferred).
    cv::Mat ref_gray;
    cv::Matx33d R_ref = cv::Matx33d::eye();
    cv::Vec3d t_ref{0.0, 0.0, 0.0};

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
// The model-warp fields stay empty/zero when the subpix operator ran.
struct TrackedRefineStats {
    bool enabled = false;
    int refined_count = 0;
    double mean_shift_px = 0.0;
    double p95_shift_px = 0.0;
    double max_shift_px = 0.0;

    int model_warp_count = 0;          // points measured by model-warp
    double model_warp_mean_zncc = 0.0;
    std::vector<uint8_t> model_warp_ok;   // per input point, empty if unused
    std::vector<float> model_warp_zncc;   // per input point, empty if unused

    // Per-point measurement snapshots, only filled when model_warp is the
    // active operator: the incoming LK position and the subpix baseline the
    // warp measurement overwrote.  Lets a run log both operators on the
    // SAME frames for offline comparison.
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
    //   "model_warp" - forward-model template registration against the
    //                  reference view through the marker surface model;
    //                  per-point fallback to the subpix measurement when a
    //                  quality gate fails or no model context is available.
    std::string tracked_refine_method = "subpix";

    // ---- model_warp parameters (defaults = validated A1 prototype) ----
    int model_warp_half_window = 12;          // patch half size -> 25x25
    int model_warp_max_iters = 12;            // LK iterations
    double model_warp_min_zncc = 0.6;         // template/image match gate
    // Maximum deviation of the registered position from the LK/subpix
    // estimate the registration was seeded with.  The seed already follows
    // the frame-to-frame motion, so this gate stays tight without rejecting
    // fast motion (the old absolute-shift gate killed whole frames as soon
    // as the marker moved faster than the gate).
    double model_warp_max_shift_px = 4.0;
    double model_warp_max_incidence_deg = 72.0;  // grazing-angle cull
    double model_warp_min_valid_frac = 0.55;  // usable template pixels
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
    // model_context is only used by the "model_warp" operator and may be
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

    // Forward-model measurement pass: for every tracked point with a valid
    // anchor, predict its local appearance by warping the reference view
    // through the surface model into the current frame and register the
    // prediction with translation-only LK (+ gain/bias).  Successful
    // measurements overwrite the subpix baseline in-place; gated points
    // keep the baseline.
    void runModelWarp(
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