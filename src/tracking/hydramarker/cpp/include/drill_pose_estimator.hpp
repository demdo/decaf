#pragma once

// Drill pose estimator (iterated EKF, variant "co_alt_eb" of the offline
// prototype smoother3.py): a random-walk prior on the marker pose T_cm
// (marker -> colour camera, mm), opened by the coherent image motion of the
// pose corners, and a robust (Huber) Gauss-Newton MAP per frame over the RGB
// corners and the IR stereo pairs, started at the engine pose.
//
// Self-contained unit: no engine dependency, no image access, no OpenCV calls
// beyond the small fixed-size types. Only member state (no static or
// thread_local data), fixed-size cv::Matx/std::array arithmetic and member
// scratch vectors that are reused frame to frame, so two estimators may run
// in parallel without the GIL and no cv::Mat of the process-wide cache is
// touched. Every guard is an explicit check (no assert, which /DNDEBUG
// removes in Release).
//
// Conventions (do not mix with the engine's mapPoseFuse / pose_covariance):
//  * perturbation on the right, in the MARKER frame, tangent [omega; tau]
//    with the rotation (rad) first and the translation (mm) second:
//    boxplus(R, t, d) = (R Exp(d[0:3]), t + R d[3:6]);
//  * P is the 6x6 covariance of that tangent, row-major;
//  * times: t_s and gap_s in s, fdt/toff/row_ms in ms;
//  * angles of the configuration (sig_r_deg, gain_r_deg) in degrees.
//
// Computation order follows iekf_bench.cpp (the C++ prototype that matches
// smoother3.py to 6.5e-13 mm); the affine motion statistic vaff follows the
// least-squares semantics of numpy.linalg.lstsq (rcond = None) instead of the
// prototype's normal equations. Branches marked "new" in the comments of the
// .cpp never occur on the validation data and are tested synthetically.

#include <array>
#include <limits>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

// Parameters. Defaults = frozen_config.json "estimator" (co_alt_eb); the
// three additions prior_max_mm, hide_sig_mm (0 = off) and gap_s (0.5) keep
// the frozen state by default.
struct DrillEstimatorConfig {
    double sig_t_mm = 0.02;        // base translation noise per T0 = 1/30 s [mm]
    double sig_r_deg = 0.008;      // base rotation noise per T0 [deg]
    double adapt_px = 1.0;         // dead band of nu1 [px/frame]; <= 0: no opening
    double db3_px = 0.3;           // dead band of nu3 [px/frame]; <= 0: nu3 off
    double gain_t = 2.0;           // [-]; term gain_t * nu * Z / f in mm
    double gain_r_deg = 0.5;       // [deg per px/frame]
    int vmin = 4;                  // matched corners for nu (at least 3)
    double s_rgb = 7.57;           // RGB sigma scale
    double s_ir = 11.08;           // IR sigma scale
    double huber_k = 1.5;          // Huber threshold (normalised residual norm)
    int n_iter = 3;                // Gauss-Newton steps (stop also at |step| < 1e-7)
    double gap_s = 0.5;            // chain break after a gap >= gap_s [s]
    double rs_px = 3.0;            // rolling-shutter dead band [px/frame]; < 0: off
    double ir_gated = 1.0;         // sigma factor of IR pairs of gate-rejected fusions
    double ir_fit_max_mm = 6.0;    // IR switch-off above this engine fit RMS; 0: off
    double ir_fit_fac = 1000.0;    // sigma factor of switched-off IR pairs
    int min_start = 12;            // RGB corners needed for a chain start; 0: off
    int warm = 5;                  // start-up gate at most [frames]
    double sig_tip_max_mm = 2.0;   // start-up gate ends at sigma_tip <= [mm]; <= 0: off
    double w3d = 4.0;              // IR weight divisor (copy of ir_w3d)
    double sigma_ir_px = 0.05;     // IR pixel sigma of S_j (copy of ir_sigma_ir_px; adapter)
    double row_ms = 0.0278;        // rolling-shutter row time [ms] (copy of ir_time_offset_row_ms)
    double prior_max_mm = 0.0;     // chain restart above this prior sigma [mm]; 0: off
    double hide_sig_mm = 0.0;      // hide frames with sigma_tip above [mm]; 0: off
};

// One fresh engine pose with its measurements.
struct DrillEstimatorFrame {
    double t_s = std::numeric_limits<double>::quiet_NaN();   // host frame time [s], valid if finite and >= 0
    bool reset_before = false;                               // test hook only; the engine calls reset()
    std::array<double, 3> rvec{};                            // engine pose after the IR stage [rad]
    std::array<double, 3> tvec{};                            // [mm]
    std::vector<cv::Vec3d> X;                                // model corners, not predicted [mm, marker frame]
    std::vector<cv::Vec2d> u;                                // undistorted normalised (K_, dist_coeffs_)
    std::vector<double> row_px;                              // raw (distorted) pixel row [px]
    std::vector<int> id;                                     // global_row * 1024 + global_col
    double sig_px = std::numeric_limits<double>::quiet_NaN();     // ir_sigma_px_eff [px] or NaN
    cv::Vec2d vmed = cv::Vec2d(std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::quiet_NaN());   // [px] or NaN
    double fdt_ms = std::numeric_limits<double>::quiet_NaN();
    double toff_ms = std::numeric_limits<double>::quiet_NaN();
    double ir_fit_mm = std::numeric_limits<double>::quiet_NaN();
    bool ir_applied = false;
    std::vector<cv::Vec3d> Xi, Yi;                           // IR pairs: model [mm], RGB frame without time offset [mm]
    std::vector<cv::Matx33d> Si;                             // [mm^2], without QF scale, not inverted
    double f = std::numeric_limits<double>::quiet_NaN();     // K_(0,0) [px]
    double fy = std::numeric_limits<double>::quiet_NaN();    // K_(1,1) [px]
};

struct DrillEstimatorOutput {
    bool valid = false;            // posterior present (event 0 or 1)
    bool shown = false;            // start-up gate over (and sigma_tip <= hide_sig if set)
    bool chained = false;          // prior of the previous frame used
    bool rs_on = false;            // rolling-shutter model applied
    int since = 0;                 // chained frames since the chain start
    int iters = 0;                 // Gauss-Newton steps
    int n_rgb = 0, n_ir = 0;       // corners / pairs used
    int n_drop_rgb = 0, n_drop_ir = 0, n_dup_id = 0;   // dropped by the input rules
    // Result: 0 chained, 1 chain start, 2 chain-start gate, 3 inversion of P
    // failed, 4 non-finite result, 5 no tool tip, 6 no frame time,
    // 7 exception (adapter; the core reports it too if it ever catches one),
    // 10 configuration invalid.
    int event = 0;
    // Reason of a chain start (event 1/2): 0 none, 1 no previous state, 2 gap,
    // 3 reset_before, 4 prior limit, 5 pose jump, 6 time jump,
    // 7 prior inversion failed.
    int restart = 0;
    // Posterior T_cm (marker -> colour camera) [mm]; NaN unless valid.
    cv::Matx33d R = cv::Matx33d::all(std::numeric_limits<double>::quiet_NaN());
    cv::Vec3d t = cv::Vec3d::all(std::numeric_limits<double>::quiet_NaN());
    // [omega (rad); tau (mm)], right perturbation, marker frame, row-major; NaN unless valid.
    std::array<double, 36> P = filledP();
    cv::Vec3d tip = cv::Vec3d::all(std::numeric_limits<double>::quiet_NaN());   // R p_tip + t, also if hidden
    double sigma_tip_mm = std::numeric_limits<double>::quiet_NaN();
    double nu1 = std::numeric_limits<double>::quiet_NaN();
    double nu3 = std::numeric_limits<double>::quiet_NaN();
    double nis = std::numeric_limits<double>::quiet_NaN();
    double sig_px_used = std::numeric_limits<double>::quiet_NaN();
    bool ir_off = false;           // IR switch-off (fit > ir_fit_max) active

private:
    static std::array<double, 36> filledP()
    {
        std::array<double, 36> p;
        p.fill(std::numeric_limits<double>::quiet_NaN());
        return p;
    }
};

// Snapshot of the internal state (test hook only).
struct DrillEstimatorState {
    bool have = false;
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d t = cv::Vec3d(0.0, 0.0, 0.0);
    std::array<double, 36> P{};
    double prev_t = 0.0;
    int since = 0;
    bool starting = false;
    int hist_n = 0;
};

class DrillPoseEstimator {
public:
    explicit DrillPoseEstimator(const DrillEstimatorConfig& cfg = DrillEstimatorConfig());

    const DrillEstimatorConfig& config() const { return cfg_; }
    bool configValid() const { return cfg_ok_; }

    // Clears the chain (have, prev_t). The motion history stays (as in the
    // prototype, where a reset never cleared it); the tool tip stays.
    void reset();

    // Tool tip in the marker frame [mm]. false (tip unchanged) if not finite.
    bool setToolTip(const cv::Vec3d& p_tip_mm);
    bool hasToolTip() const { return tip_set_; }

    // One fresh frame. Never throws.
    DrillEstimatorOutput update(const DrillEstimatorFrame& in);

    // Test hooks (bound to Python only, never used by the engine).
    DrillEstimatorState debugState() const;
    // Sets (R, t, P), have = true; prev_t only if finite. since/starting and
    // the motion history are not touched.
    void debugSetState(const cv::Matx33d& R, const cv::Vec3d& t,
                       const std::array<double, 36>& P, double prev_t);

private:
    typedef double M3[3][3];

    void updateImpl(const DrillEstimatorFrame& in, DrillEstimatorOutput& out);
    double vaff(int b);
    void measTerms(const M3 R, const double t[3], double f, double sig,
                   double H[36], double g[6]) const;

    DrillEstimatorConfig cfg_;
    bool cfg_ok_ = false;

    // chain state
    bool have_ = false;
    M3 R_ = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    double t_[3] = {0.0, 0.0, 0.0};
    double P_[36] = {0.0};
    double prev_t_ = 0.0;
    int since_ = 0;
    bool starting_ = false;

    // motion history: f*u of the last 4 frames, each sorted by id (binary
    // search); ring buffer, slot hist_head_ is written next
    struct HistFrame {
        std::vector<std::pair<int, cv::Vec2d>> uv;
    };
    std::array<HistFrame, 4> hist_;
    int hist_n_ = 0;
    int hist_head_ = 0;

    cv::Vec3d p_tip_ = cv::Vec3d(0.0, 0.0, 0.0);
    bool tip_set_ = false;

    // scratch (reused, capacity reserved once)
    std::vector<std::pair<int, int>> sort_;     // (id, position) of valid corners
    std::vector<char> keep_;                     // per input corner: used
    std::vector<int> cur_id_;                    // used corners in list order
    std::vector<cv::Vec2d> cur_uv_;              // f * u, list order
    std::vector<cv::Vec3d> Xr_;
    std::vector<cv::Vec2d> ur_;                  // u after the rolling-shutter shift
    std::vector<double> rows_;
    std::vector<double> rows_tmp_;
    std::vector<cv::Vec3d> Xi_, Yi_;
    std::vector<cv::Matx33d> Si_;
    std::vector<cv::Matx33d> Wi_;
    std::vector<cv::Vec2d> vU_, vD_;            // vaff: matched U0 and displacement
    std::vector<double> vq1_, vq2_;
};

// Image motion vmed of the prototype (co_prep.py): per-axis median (mean of
// the two middle values for an even count) of the raw-pixel displacement of
// the corners against the previous fresh frame, valid only if that frame is
// the directly preceding camera frame (frame_index + 1) and >= 8 corners match.
class DrillVmed {
public:
    DrillVmed();
    void reset();
    // id / uv_raw of the not-predicted corners of a fresh frame (list order;
    // a duplicate id: the last one counts). Returns NaN if not valid. Always
    // stores this frame as the new predecessor.
    cv::Vec2d update(const std::vector<int>& id, const std::vector<cv::Vec2d>& uv_raw,
                     int frame_index);

private:
    std::vector<std::pair<int, cv::Vec2d>> prev_;   // sorted by id, unique
    std::vector<std::pair<int, cv::Vec2d>> cur_;
    std::vector<std::pair<int, int>> sort_;
    std::vector<double> dx_, dy_;
    bool have_prev_ = false;
    int prev_frame_ = -1;
};

}  // namespace hydramarker
