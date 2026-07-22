#pragma once

#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "corner_refinement.hpp"
#include "marker_geometry.hpp"

namespace hydramarker {

// Rigid three-camera calibration of the RGB + IR-left + IR-right rig.
// All rotations/translations map a point INTO the named target frame:
//   X_irL = R_rgb_left  * X_rgb + t_rgb_left_mm
//   X_irR = R_left_right * X_irL + t_left_right_mm
// (millimetres throughout; this is the realsense_ir_calibration.npz content).
// K_rgb/dist_rgb are the RGB intrinsics OF THE SAME npz: the fusion seeds are
// transferred RGB -> IR through these (the tracker's own K may differ).
struct IrCameraCalibration {
    cv::Matx33d K_rgb = cv::Matx33d::eye();
    std::vector<double> dist_rgb;
    cv::Matx33d K_left = cv::Matx33d::eye();
    cv::Matx33d K_right = cv::Matx33d::eye();
    std::vector<double> dist_left;
    std::vector<double> dist_right;
    cv::Matx33d R_rgb_left = cv::Matx33d::eye();
    cv::Vec3d t_rgb_left_mm{0.0, 0.0, 0.0};
    cv::Matx33d R_left_right = cv::Matx33d::eye();
    cv::Vec3d t_left_right_mm{0.0, 0.0, 0.0};
    bool valid = false;
};

// Result of the tightly-coupled MAP pose fusion (RGB reprojection + IR stereo
// 3D), robust Gauss-Newton on SE(3). cov = H^-1 at convergence (6x6, order
// [t(3), rot(3)]) -> the measurement covariance for the temporal filter.
struct MapFuseResult {
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d t{0.0, 0.0, 0.0};
    cv::Matx<double, 6, 6> cov = cv::Matx<double, 6, 6>::zeros();
    int n_used = 0;
    bool ok = false;
};

// Tightly-coupled MAP pose fusion. Mirrors the validated Python prototype
// map_core.fuse_map_vec operation-for-operation (normalised-undistorted RGB
// reprojection residuals with weight (f/sig_px)^2, anisotropic IR-stereo 3D
// residuals with per-corner information SigY_inv*w3d, Huber robustification,
// plain Gauss-Newton with a 1e-9 diagonal, an initial MAD trim). Frames are:
//   X_i  : model corner positions (marker frame, mm)
//   u_i  : normalised UNDISTORTED RGB observations (x/z, y/z)
//   Y_i  : IR-triangulated 3D corner positions (RGB camera frame, mm)
//   SigY_inv_i : per-corner 3x3 information (already anisotropic), pre-w3d
// Seeded from (R0, t0) = the RGB pose. Returns the fused pose + covariance.
MapFuseResult mapPoseFuse(const std::vector<cv::Vec3d>& X,
                          const std::vector<cv::Point2d>& u_norm,
                          const std::vector<cv::Vec3d>& Y,
                          const std::vector<cv::Matx33d>& SigY_inv,
                          const cv::Matx33d& R0, const cv::Vec3d& t0,
                          double f, double sig_px, double w3d = 1.0,
                          int iters = 8, double huber = 1.5, bool trim = true);

// Configuration of the depth-only IR fusion. Defaults are the final
// prototype configuration validated on the divot ground-truth session
// 20260718_181438 (tests/prove_ir_depth_fusion.py: |e| med 1.43 -> 0.20 mm).
struct IrPoseRefinerConfig {
    bool enabled = false;

    // IR corner measurement operator:
    //   "model_warp"     - registration against enrolled reference photo
    //                      pairs (the reference library below; established
    //                      behaviour, acceptance replays pin it)
    //   "quadratic_form" - reference-free: the model grid curves are
    //                      projected into each IR view and the corners are
    //                      measured directly (same QF machinery as the RGB
    //                      tracking path, incl. the saddle-warp layer).
    //                      The ENTIRE reference library (enrollment, tiles,
    //                      selection, starvation escape) is bypassed; the
    //                      saturation measurement for the exposure
    //                      controller stays.
    std::string corner_method = "model_warp";
    // QF acceptance gate vs the seed for the IR views: the seeds are
    // rig-transferred RGB pixels (1-3 px transfer error), so the gate is
    // looser than the RGB-path 2.5 px.
    double qf_max_dev_px = 4.0;

    // --- per-frame fusion ---
    int min_pairs = 6;                 // stereo pairs needed to fuse at all
    double min_ref_rot_deg = 6.0;      // NEVER use a reference closer than
                                       // this to the current orientation: at
                                       // ~0 deg the warp is the identity and
                                       // the measurement is slaved to the
                                       // (biased) enrollment pose.
    double max_ref_rot_deg = 180.0;    // optional upper selection window
    double fallback_min_ref_rot_deg = 0.05;  // fixed-orientation fallback: when
                                       // NO reference reaches min_ref_rot_deg
                                       // (a pure-translation sweep never rotates
                                       // far enough), use the nearest reference
                                       // above this tiny floor so depth still
                                       // gets a correction. Excludes the exact-
                                       // identity view. Only ADDS fusion on
                                       // frames that had no >= min_ref_rot_deg
                                       // reference; it never replaces one, so
                                       // frames that already fused are
                                       // bit-identical (validated: divot z_perp
                                       // unchanged, translation 0% -> 54%).
    int sat_threshold = 250;           // any pixel >= this in the seed patch
    int sat_half_px = 8;               //   (+-half, either view) drops the
                                       //   corner: saturation destroys the
                                       //   information in the sensor.
    double epipolar_max_dv_px = 2.5;   // |vL - vR| row-consistency gate
    double zncc_weight_floor = 0.05;   // w = clip(min(znccL,znccR), floor,1)^2
    double dtz_clamp_mm = 2.5;         // robust-median depth shift is clamped
                                       //   to this (guards against a broken
                                       //   stereo depth on a bad frame)
    double depth_scale = 1.0;          // absolute stereo scale correction
                                       // from the ChArUco validation shots
                                       // (K_SCALE, session-level)

    // --- MAP fusion (tightly-coupled RGB reprojection + IR stereo 3D) ---
    // Replaces the depth-only dtz median with a robust SE(3) Gauss-Newton over
    // both sensors. sigmas are calibrated from residuals so w3d = 1.
    double sigma_px = 0.10;            // RGB reprojection noise (px)
    double sigma_ir_px = 0.05;         // IR corner-localisation noise (px) ->
                                       // physical anisotropic stereo covariance
    double w3d = 1.0;                  // IR/reproj balance. Library default 1
                                       // = the still-frame sigma calibration
                                       // (bit-identical to the prototype and
                                       // the acceptance replays). The TRACKER
                                       // runs 4 via TrackerConfig::ir_w3d to
                                       // compensate correlated RGB corner
                                       // errors during motion.
    double fit_gate_rms_mm = 1.50;     // reject fusion if the MAP 3D-residual RMS
                                       // exceeds (the reproj-constrained MAP
                                       // residual runs ~2.4x the pure-kabsch RMS
                                       // it was first tuned on; the trans-jump +
                                       // pairs gates catch the divergent tail)
    double fit_gate_max_trans_jump_mm = 3.0;  // reject if |t_map - t_rgb| exceeds

    // --- model_warp tuning for the soft 720p IR views ---
    int mw_half_window = 8;
    double mw_min_zncc = 0.35;
    double mw_max_shift_px = 6.0;
    double mw_max_incidence_deg = 80.0;
    double mw_min_valid_frac = 0.4;

    // --- reference library enrollment ---
    // Enroll a reference the moment the tool enters an ORIENTATION TILE the
    // library does not cover yet (no quiet streak: the IR pair is global
    // shutter, so a moderately-moving frame still yields a sharp reference,
    // and a blurred one is rejected by the fusion-time ZNCC/saturation gates
    // = RGB fallback, never worse). The engine gates the call on pose
    // convergence + a RELAXED motion cap; here we only enforce the tile gap.
    double ref_tile_deg = 12.0;        // orientation tile size: enroll only
                                       // when no reference is closer than this
    // Translation tile: references also cover a POSITION neighbourhood. A pure
    // translation sweep (fixed orientation) leaves the orientation tile covered
    // forever, so the library stayed at ONE home reference and 80 mm away the
    // warp stretched, the saturation pattern flipped (0.15 -> 0.51) and the
    // pairs halved (fb4). Enroll whenever the pose leaves the (orientation AND
    // position) neighbourhood of every reference. <= 0 disables the position
    // dimension everywhere (tile, selection, fallback) = previous behaviour.
    double ref_tile_trans_mm = 40.0;
    // A reference displaced in POSITION at the same orientation is NOT the
    // degenerate identity view (the warp genuinely re-localises), so the
    // fallback also admits references at least this far away in translation
    // even when their rotation distance is under fallback_min_ref_rot_deg.
    double fallback_min_ref_trans_mm = 15.0;
    // Saturation ceiling for ENROLLMENT, checked on the candidate frames
    // themselves (marker corners projected into both IR views): the engine's
    // last-frame quality gate has a one-frame race (fb_rl2: a specular blob
    // arrived in the very enrollment frame and the contaminated template
    // biased z by +1.7 mm). 0.50, NOT the fusion-gate 0.35: a template with
    // 38% clipped corners still carries 62% usable ones (the per-corner sat
    // gates handle the rest at measurement time) - 0.35 STARVED a run that
    // started in a specular zone (sat 0.38 -> no reference for 500 frames,
    // silently RGB-only). <= 0 disables.
    double enroll_max_sat_frac = 0.50;
    // Starvation escape: with NO references at all, after this many refused
    // bootstrap attempts enroll the candidate anyway - a partially saturated
    // template is strictly better than no IR fusion. <= 0 disables.
    int enroll_starvation_frames = 60;
    int max_references = 12;           // library capacity across the sweep
};

enum class IrFusionMode {
    Rgb = 0,     // no usable reference/pairs -> RGB pose kept
    Depth = 1,   // absolute-depth shift applied (the only correction we make)
};

struct IrPoseRefinerResult {
    bool applied = false;          // fusion modified the pose
    IrFusionMode mode = IrFusionMode::Rgb;
    int ref_count = 0;             // references currently enrolled
    int refs_measured = 0;         // references actually measured this frame
    int pairs = 0;                 // stereo pairs surviving all gates
    double saturated_frac = -1.0;  // fraction of visible marker corners clipped
                                   // in either IR view (-1 = not measured this
                                   // frame). The one-shot IR-exposure calibrator
                                   // drives this toward a low value at startup.
    double best_ref_angle_deg = 0.0;  // rotation distance to the chosen ref
    double best_ref_trans_mm = 0.0;   // translation distance to the chosen ref
    double quality = 0.0;          // sum of min(znccL,znccR)^2 of the used ref
    double tilt_deg = 0.0;         // fitted out-of-plane tilt (DIAGNOSTIC ONLY,
                                   // never applied - unobservable per frame,
                                   // amplifies over the tool lever)
    double dtz_mm = 0.0;           // (legacy diagnostic; MAP reports fit_rms_mm)
    double fit_rms_mm = 0.0;       // MAP 3D-residual RMS at convergence (gate)
    std::array<double, 3> rvec{{0.0, 0.0, 0.0}};  // fused, marker -> RGB
    std::array<double, 3> tvec{{0.0, 0.0, 0.0}};
    // 6x6 pose covariance H^-1 at convergence, order [t(3), rot(3)]; the
    // measurement covariance handed to the temporal filter (0 when not applied).
    std::array<double, 36> cov{};
};

// Depth-only IR stereo fusion of the reported pose ("fuse, don't replace").
//
// WHY strictly depth (no rotation): the stereo cloud's ROTATION is far noisier
// than the RGB rotation. The out-of-plane tilt from one frame's stereo is
// ~1 mm point noise over a ~50 mm marker patch = ~1.1 deg, and over the
// ~180 mm tool lever that is ~3.5 mm of tip motion - so the tilt is UNDER the
// noise floor per frame and unobservable. Applying it (as a pose rotation, or
// equivalently as a lever-extrapolated depth) only injects lateral tip jitter
// (measured on the live divot run: RGB tip x-std 0.39 mm -> tilt-mode
// 2.59 mm, while z was unchanged). The RGB pose already observes rotation and
// lateral position to 0.1 px. Therefore the fusion contributes ONLY the one
// thing RGB cannot observe well: the marker's absolute depth, applied as a
// pure translation (no rotation, no lever amplification). The depth GRADIENT
// (tilt) can only be recovered by AVERAGING over frames - that is the job of
// the reference library / the offline pivot two-pass, not a single frame.
//
// Per frame: model corners visible under the RGB pose are seeded into both IR
// views, measured by model_warp against the ANGULARLY NEAREST admissible
// enrolled reference pair (only warp-converged corners in BOTH views survive;
// saturated patches are gated out), triangulated over the stereo baseline;
// the robust (MAD-trimmed, clamped) median of the depth residuals
// z_stereo - z_pose is applied as a pure depth translation. Without enough
// pairs the RGB pose stays (never worse). The tilt is still fitted and
// reported for diagnostics, but NEVER applied.
//
// Reference selection is nearest-first for BOTH accuracy and speed: the
// closest admissible orientation has the least warp stretch (best ZNCC), so
// measuring it first and stopping at the first reference that yields enough
// pairs bounds the cost to ~one measurement regardless of library size
// (measuring every reference cost ~3 ms each and scaled with the library).
//
// Reference library: multiple references over the orientation range
// (~tile_deg apart), enrolled from quiet converged phases WITH THE FUSED pose
// (bootstrap), each used only >= min_ref_rot_deg away from its own enrollment
// orientation (self-enslavement guard).
//
// Like PoseOutputFilter this is an output-only stage on the REPORTED pose;
// the internal tracking chain keeps the raw RGB pose so no feedback loop
// forms. Double-gated: config flag AND calibration data.
class IrPoseRefiner {
public:
    IrPoseRefiner() = default;

    // corner_cloud_mm: ALL marker corners (marker frame, mm) - the fusion
    // computes its own visibility set from the surface axis per frame.
    void configure(const IrPoseRefinerConfig& config,
                   const SurfaceModel& surface,
                   const std::vector<cv::Vec3d>& corner_cloud_mm);
    void setCalibration(const IrCameraCalibration& calib);
    // Full marker pattern for the synthetic-template registration in the
    // QF-IR path (invalid LUT = quadrant fallback).
    void setPattern(const MarkerPatternLut& pattern) { pattern_ = pattern; }

    // Stereo self-calibration correction (QF-IR path only): 7 polynomial
    // coefficients applied to the MEASURED right-view x before the epipolar
    // gate / triangulation:
    //   xR += c0 + c1*xn + c2*yn + c3*xn^2 + c4*xn*yn + c5*yn^2
    //         + c6*(xL-xR)/100
    // (xn/yn = normalized right-view coords). Fitted offline from marker
    // runs (rigid-model constraint); empty disables.
    void setSelfCal(const std::vector<double>& coef) {
        selfcal_valid_ = coef.size() == 7;
        if (selfcal_valid_) {
            std::copy(coef.begin(), coef.end(), selfcal_.begin());
        }
    }

    // Per-corner disparity correction (QF-IR): the stable print/model
    // discrepancy per PHYSICAL corner (cross-session corr +0.73), applied
    // as xR -= dxr for the matching marker corner. flat_xyz = 3N marker
    // coords (mm), dxr = N offsets (px).
    void setSelfCalCorners(const std::vector<double>& flat_xyz,
                           const std::vector<double>& dxr) {
        corner_dxr_.clear();
        if (dxr.empty() || flat_xyz.size() != dxr.size() * 3) return;
        for (size_t i = 0; i < dxr.size(); ++i) {
            corner_dxr_[cornerKey(cv::Vec3d(flat_xyz[3 * i],
                                            flat_xyz[3 * i + 1],
                                            flat_xyz[3 * i + 2]))] = dxr[i];
        }
    }

    void reset();

    // Last frame's measured stereo pairs of the QF path (survivors of all
    // gates), for the offline stereo SELF-CALIBRATION: marker xyz (mm) and
    // the measured pixel in each IR view. Cleared when the frame produced
    // no QF measurement.
    struct IrPairDump {
        std::vector<cv::Vec3d> xyz;
        std::vector<cv::Point2f> uvL;
        std::vector<cv::Point2f> uvR;
    };
    const IrPairDump& lastPairDump() const { return pair_dump_; }

    // Reference-free quadratic-form measurement in ONE IR view under an
    // EXPLICIT pose. Public for the offline model-refinement post-pass
    // (refine_model_ir): observation stills have no tracking context, the
    // pose comes from solvePnP on the stored correspondences.
    void measureViewQf(const cv::Mat& gray,
                       const cv::Matx33d& K,
                       const std::vector<double>& dist,
                       const cv::Matx33d& R_view,
                       const cv::Vec3d& t_view,
                       const std::vector<cv::Vec3d>& xyz,
                       const std::vector<cv::Point2f>& seeds,
                       std::vector<cv::Point2f>& uv_out,
                       std::vector<uint8_t>& ok_out,
                       std::vector<float>& q_out) const;

    const IrCameraCalibration& calibration() const { return calib_; }

    bool active() const {
        return config_.enabled && calib_.valid && surface_.valid() &&
               !corners_.empty();
    }
    // One MAP fusion step. rvec/tvec: accepted RAW RGB pose, marker -> RGB
    // camera (mm). rgb_xyz/rgb_uv: the tracked RGB corners the pose was solved
    // on - model position (marker frame, mm) and DETECTED pixel (distorted, RGB
    // image). The reprojection residual uses them; the IR stereo triangulation
    // of the SAME corners gives the 3D residual. Returns the fused pose (== input
    // when mode == Rgb, e.g. no reference / too few pairs / gate reject).
    // sigma_px_override > 0 replaces config.sigma_px for THIS frame. The engine
    // passes a velocity-inflated sigma: the RGB camera is ROLLING SHUTTER, so
    // fast in-image motion shears the marker coherently (roll wobble measured
    // proportional to push velocity, corr +0.6..0.8) - an error the IID
    // still-frame sigma cannot represent. The global-shutter IR is immune, so
    // widening the RGB sigma with velocity hands orientation authority to IR
    // exactly when the shear strikes. <= 0 keeps the configured sigma
    // (bindings/acceptance replays stay bit-identical).
    IrPoseRefinerResult fuse(const cv::Mat& ir_left,
                             const cv::Mat& ir_right,
                             const std::array<double, 3>& rvec,
                             const std::array<double, 3>& tvec,
                             const std::vector<cv::Vec3d>& rgb_xyz,
                             const std::vector<cv::Point2d>& rgb_uv,
                             double sigma_px_override = -1.0);

private:
    IrPoseRefinerConfig config_;
    SurfaceModel surface_;       // band opened (gates only the angular
                                 // coordinate, which the IR views may exceed)
    std::vector<cv::Vec3d> corners_;       // marker frame, mm
    std::vector<cv::Vec3d> corner_normals_;  // outward radial unit normals
    IrCameraCalibration calib_;
    CornerRefiner refiner_;

    MarkerPatternLut pattern_;
    IrPairDump pair_dump_;
    std::array<double, 7> selfcal_{};
    bool selfcal_valid_ = false;
    // Quantized marker-coordinate key (0.1 mm) -> per-corner disparity fix.
    static long long cornerKey(const cv::Vec3d& x) {
        const long long a = static_cast<long long>(std::llround(x[0] * 10.0));
        const long long b = static_cast<long long>(std::llround(x[1] * 10.0));
        const long long c = static_cast<long long>(std::llround(x[2] * 10.0));
        return (a + 500000) * 1000000000000LL + (b + 500000) * 1000000LL +
               (c + 500000);
    }
    std::unordered_map<long long, double> corner_dxr_;

    // One MAP solve against the measured pair set of one reference. Returns
    // true (fills out.rvec/tvec/cov, applied) when the fit gate accepts;
    // false so fuse() can retry with the next admissible reference.
    // q_min (optional, indexed like use_good over the FULL corner set):
    // per-corner measurement quality in (0,1]; scales the per-corner IR
    // sigma as sigma_ir_px / q (clamped) so poorly-measured corners carry
    // less depth authority. nullptr = uniform sigma (mw path, bit-identical).
    bool fuseAttempt(IrPoseRefinerResult& out,
                     const std::vector<uint8_t>& use_good,
                     const std::vector<cv::Point2f>& use_ptsL,
                     const std::vector<cv::Point2f>& use_ptsR,
                     const std::vector<cv::Vec3d>& xyz,
                     const std::vector<cv::Point2d>& uv_det,
                     const cv::Matx33d& R_rgb,
                     const cv::Vec3d& t_rgb,
                     int use_pairs,
                     size_t n,
                     double sigma_px,
                     const std::vector<float>* q_min = nullptr) const;
};

}  // namespace hydramarker
