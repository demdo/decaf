#pragma once

#include <array>
#include <deque>
#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

// Configuration for the anisotropic output pose filter. The fixed-Q fields
// mirror the former TrackerConfig::pose_kf_* values; the adaptive-Q fields are
// the addition validated offline (static tip jitter 0.10 mm -> 0.034 mm at
// zero added lag, see calib/output/080726/filter_validation.png).
struct PoseOutputFilterConfig {
    bool enabled = true;
    double sigma_px = 0.12;

    // Constant-velocity acceleration process noise while MOVING. Used as-is in
    // fixed mode (adaptive == false), and as the upper end of the adaptive
    // ramp. 0.5 deg tracks a normal hand rotation with zero lag and no
    // overshoot (validated on raw logs); 0.15 mm likewise for translation.
    double q_translation_mm = 0.15;
    double q_rotation_deg = 0.5;

    // Adaptive process noise: a bounded two-regime ramp. While the marker is
    // still (motion below the floor) Q collapses to q_*_floor so the filter
    // averages many frames -> heavy jitter suppression; as motion rises from
    // the floor to 2x the floor Q ramps up to the moving value q_*_deg/mm and
    // saturates there, so fast motion follows without lag AND without the
    // unbounded velocity overshoot an open-ended gain would cause. Motion is
    // judged causally from the EMA of the frame-to-frame pose-change NORM
    // (axis-invariant), so there is no output latency. Absolute floors are
    // used (not derived from sigma_px^2 (J^T J)^-1, which underestimates the
    // real per-frame pose noise and made a covariance-derived floor misfire on
    // jitter and AMPLIFY it).
    bool adaptive = true;
    double q_rotation_floor_deg = 0.002;
    double q_translation_floor_mm = 0.002;
    double adaptive_ema_alpha = 0.4;      // EMA weight on the new delta norm
    // Self-calibrating noise level (default): the filter estimates its own
    // quiet-baseline frame-to-frame delta over a rolling window and derives
    // both the measurement-noise floor and the motion threshold from it. This
    // is why it adapts across sessions/orientations where the pose noise
    // varies several-fold; fixed floors cannot. When off, the manual
    // motion_floor_*/meas_floor_* below are used verbatim.
    bool auto_noise = true;
    int noise_window = 90;         // frames in the rolling estimate (~3 s)

    // Motion floor (fixed coarse still/motion separator, on the frame-to-frame
    // pose-change norm) and measurement-noise floor (added to the
    // sigma^2 (J^T J)^-1 diagonal). Under auto_noise the measurement floor is
    // the LOWER BOUND of the self-calibrated estimate; the motion floor is
    // always used verbatim. Under manual mode both are used verbatim.
    double motion_floor_rot_deg = 0.12;
    double motion_floor_trans_mm = 0.05;
    double meas_floor_rot_deg = 0.035;
    double meas_floor_trans_mm = 0.02;

    // Gating / reset (same semantics as the former inline filter).
    double gate_mahalanobis = 30.0;
    double reset_rotation_deg = 10.0;
    double max_translation_jump_mm = 40.0;
    int min_points = 6;
};

// Outcome of one filter step.
struct PoseOutputFilterResult {
    bool applied = false;   // true => rvec/tvec/covariance below are valid and
                            // the caller should write them to its output pose
    bool initialized_this_frame = false;
    bool gated = false;
    double mahalanobis = 0.0;
    std::array<double, 3> rvec{{0.0, 0.0, 0.0}};
    std::array<double, 3> tvec{{0.0, 0.0, 0.0}};
    // 6x6 pose covariance (rvec, tvec block of the state covariance),
    // row-major. Downstream can propagate any rigidly attached point
    // p (marker frame) to its camera-frame uncertainty via J * P * J^T,
    // where J = d(R(rvec)*p + tvec)/d(rvec, tvec).
    std::array<double, 36> covariance{};
};

// Anisotropic constant-velocity output pose filter over (rvec, tvec).
//
// It is meant to run on the REPORTED pose only, as the last output stage; the
// caller keeps the raw pose for its internal tracking chain so no feedback
// loop forms. Deliberately tip-agnostic: it smooths the generic 6-DoF pose and
// exposes the pose covariance, so any consumer can derive a rigidly attached
// point (e.g. a tool tip = R*p + t) and its uncertainty downstream. That keeps
// the filter reusable outside this particular tool-tip application.
class PoseOutputFilter {
public:
    PoseOutputFilter() = default;

    // K / distortion are needed to build the per-frame measurement covariance
    // sigma_px^2 (J^T J)^-1 from the pose-set corner geometry.
    void configure(const PoseOutputFilterConfig& config,
                   const cv::Matx33d& K,
                   const std::vector<double>& dist_coeffs);

    void reset();
    bool initialized() const { return initialized_; }

    // No fresh measurement this frame (hold / rejection): coast the state so
    // the velocities stay time-consistent. The caller's output is untouched.
    void coast();

    // Fresh measurement. object_points are the pose-set corner coordinates
    // (marker/object frame) used to build the anisotropic measurement
    // covariance sigma_px^2 (J^T J)^-1 at the measured pose.
    //
    // ext_meas_cov (optional): an EXTERNAL 6x6 measurement covariance in
    // [rvec(3), tvec(3)] order (row-major), e.g. the IR MAP fusion's H^-1 which
    // is IR-informed (tighter in the depth mode the RGB-only Sigma leaves loose).
    // When provided it REPLACES the internally computed Sigma; the noise floors
    // are still applied. Pass nullptr for the pure-RGB behaviour (unchanged).
    // meas_var_scale (optional): scalar inflation of the measurement
    // covariance (internal or external) BEFORE the noise floors.  Used to
    // distrust frames whose corner evidence is known-degraded (the IR MAP
    // fusion ATTEMPTED and REJECTED: glare veil, adverse geometry) so the
    // filter leans on its motion model instead of following a drifting
    // RGB-only pose at full confidence.
    PoseOutputFilterResult update(
        const std::array<double, 3>& rvec,
        const std::array<double, 3>& tvec,
        const std::vector<cv::Point3d>& object_points,
        const std::array<double, 36>* ext_meas_cov = nullptr,
        double meas_var_scale = 1.0);

private:
    PoseOutputFilterConfig config_;
    cv::Matx33d K_ = cv::Matx33d::eye();
    cv::Mat dist_coeffs_;

    bool initialized_ = false;
    cv::Mat x_;       // 12x1: rvec, tvec, and their per-frame velocities
    cv::Mat P_;       // 12x12 covariance
    cv::Mat prev_z_;  // 6x1 previous measurement (adaptive motion detector)
    // EMA of the frame-to-frame pose-change norm: [rot (rad), trans (mm)].
    cv::Vec2d motion_ema_{0.0, 0.0};
    // Rolling window of STILL-frame pose-change norms and the persisted quiet
    // baseline derived from them (rot in rad, trans in mm), for the online
    // noise estimate.
    std::deque<double> drot_hist_;
    std::deque<double> dtrans_hist_;
    double est_r_ = 0.0;
    double est_t_ = 0.0;
};

}  // namespace hydramarker
