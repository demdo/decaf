#include "pose_output_filter.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace hydramarker {

namespace {

// Robust low percentile of a window of frame-to-frame deltas: the quiet
// baseline (still-frame noise), insensitive to the high deltas of motion
// frames mixed into the window.
double lowPercentile(const std::deque<double>& values, double frac)
{
    if (values.empty()) {
        return 0.0;
    }
    std::vector<double> sorted(values.begin(), values.end());
    const size_t k = static_cast<size_t>(
        frac * static_cast<double>(sorted.size() - 1));
    std::nth_element(sorted.begin(), sorted.begin() + k, sorted.end());
    return sorted[k];
}

}  // namespace

void PoseOutputFilter::configure(
    const PoseOutputFilterConfig& config,
    const cv::Matx33d& K,
    const std::vector<double>& dist_coeffs)
{
    config_ = config;
    K_ = K;
    dist_coeffs_ = dist_coeffs.empty()
        ? cv::Mat()
        : cv::Mat(dist_coeffs, true).reshape(1, 1).clone();
    reset();
}

void PoseOutputFilter::reset()
{
    initialized_ = false;
    x_ = cv::Mat();
    P_ = cv::Mat();
    prev_z_ = cv::Mat();
    motion_ema_ = cv::Vec2d(0.0, 0.0);
    drot_hist_.clear();
    dtrans_hist_.clear();
    est_r_ = 0.0;
    est_t_ = 0.0;
}

void PoseOutputFilter::coast()
{
    if (!initialized_) {
        return;
    }
    // Constant-velocity coast: advance position by velocity, leave the
    // output to the caller (there is no fresh measurement to report).
    for (int i = 0; i < 6; ++i) {
        x_.at<double>(i) += x_.at<double>(i + 6);
    }
}

PoseOutputFilterResult PoseOutputFilter::update(
    const std::array<double, 3>& rvec,
    const std::array<double, 3>& tvec,
    const std::vector<cv::Point3d>& object_points,
    const std::array<double, 36>* ext_meas_cov,
    double meas_var_scale)
{
    PoseOutputFilterResult result;
    if (!config_.enabled) {
        return result;
    }
    if (static_cast<int>(object_points.size()) < config_.min_points) {
        return result;
    }

    cv::Mat z(6, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        z.at<double>(i) = rvec[static_cast<size_t>(i)];
        z.at<double>(i + 3) = tvec[static_cast<size_t>(i)];
    }

    // Large real jumps (accepted by the caller's own gates, e.g. after
    // recovery) restart the filter instead of being dragged in slowly.
    if (initialized_) {
        double dr = 0.0;
        double dt = 0.0;
        for (int i = 0; i < 3; ++i) {
            const double er = z.at<double>(i) - x_.at<double>(i);
            const double et = z.at<double>(i + 3) - x_.at<double>(i + 3);
            dr += er * er;
            dt += et * et;
        }
        const double reset_rot_rad = config_.reset_rotation_deg * CV_PI / 180.0;
        if (std::sqrt(dr) > reset_rot_rad ||
            std::sqrt(dt) > config_.max_translation_jump_mm) {
            initialized_ = false;
        }
    }

    // Measurement covariance sigma^2 (J^T J)^-1 from the pose Jacobian: huge
    // along the weak observability mode, tiny along observable directions.
    // Computed from the measurement (independent of the filter state), so it
    // also seeds the adaptive motion floor below.
    cv::Mat rvec_m = z.rowRange(0, 3).clone();
    cv::Mat tvec_m = z.rowRange(3, 6).clone();
    std::vector<cv::Point2d> projected;
    cv::Mat J;
    cv::projectPoints(object_points, rvec_m, tvec_m, K_, dist_coeffs_,
                      projected, J);
    const cv::Mat Jp = J.colRange(0, 6);  // columns: drvec(3), dtvec(3)
    const cv::Mat info = Jp.t() * Jp;
    const double sigma2 = std::pow(config_.sigma_px, 2.0);
    cv::Mat Sigma;
    cv::invert(info, Sigma, cv::DECOMP_SVD);
    Sigma *= sigma2;

    // IR MAP fusion supplied a measurement covariance (H^-1, IR-informed):
    // replace the RGB-only sigma^2 (J^T J)^-1 with it. Same [rvec, tvec] order
    // and units (the MAP reprojection Hessian is info/sigma_px^2); the IR term
    // only tightens the weak depth mode. The noise floors below still apply.
    if (ext_meas_cov != nullptr) {
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                Sigma.at<double>(r, c) =
                    (*ext_meas_cov)[static_cast<size_t>(6 * r + c)];
            }
        }
    }

    // Known-degraded evidence (fusion attempted and rejected): distrust
    // the whole measurement BEFORE the floors, so the filter coasts on its
    // motion model through glare/adverse episodes instead of following a
    // drifting full-confidence RGB pose.
    if (meas_var_scale > 1.0) {
        Sigma *= meas_var_scale;
    }

    if (!initialized_) {
        x_ = cv::Mat::zeros(12, 1, CV_64F);
        z.copyTo(x_.rowRange(0, 6));
        P_ = cv::Mat::zeros(12, 12, CV_64F);
        const double r2 = std::pow(2.0 * CV_PI / 180.0, 2.0);
        const double vr2 = std::pow(0.5 * CV_PI / 180.0, 2.0);
        for (int i = 0; i < 3; ++i) {
            P_.at<double>(i, i) = r2;
            P_.at<double>(i + 3, i + 3) = 25.0;       // (5 mm)^2
            P_.at<double>(i + 6, i + 6) = vr2;
            P_.at<double>(i + 9, i + 9) = 4.0;        // (2 mm/f)^2
        }
        prev_z_ = z.clone();
        motion_ema_ = cv::Vec2d(0.0, 0.0);
        drot_hist_.clear();
        dtrans_hist_.clear();
        initialized_ = true;
        // First frame: output = measurement.
        for (int i = 0; i < 3; ++i) {
            result.rvec[static_cast<size_t>(i)] = z.at<double>(i);
            result.tvec[static_cast<size_t>(i)] = z.at<double>(i + 3);
        }
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) {
                result.covariance[static_cast<size_t>(r * 6 + c)] =
                    P_.at<double>(r, c);
            }
        }
        result.applied = true;
        result.initialized_this_frame = true;
        return result;
    }

    // Frame-to-frame pose-change norm (axis-invariant): drives both the noise
    // estimate and the motion detector.
    double drot = 0.0;
    double dtrans = 0.0;
    for (int i = 0; i < 3; ++i) {
        drot += std::pow(z.at<double>(i) - prev_z_.at<double>(i), 2.0);
        dtrans += std::pow(z.at<double>(i + 3) - prev_z_.at<double>(i + 3), 2.0);
    }
    drot = std::sqrt(drot);
    dtrans = std::sqrt(dtrans);

    // Noise level: either self-calibrated online or the manual floors. The
    // online estimate uses the PERSISTED quiet baseline (updated only on still
    // frames below, so sustained motion can never inflate it -> motion is not
    // misclassified as still). The manual floor is the lower bound so a
    // not-yet-seeded estimate can never under-smooth / amplify.
    double meas_floor_std_r;
    double meas_floor_std_t;
    double motion_floor_r;
    double motion_floor_t;
    // The motion threshold is ALWAYS the fixed floor: it is a coarse still-vs-
    // motion separator (static ~0.02 deg/frame vs slow rotation ~0.2 -> huge
    // margin, robust across sessions). Only the measurement-noise floor (R) is
    // self-calibrated, because that -- not the threshold -- was what
    // underestimated the real noise live and caused the amplification. Making
    // the threshold depend on the estimate created a positive-feedback runaway
    // (higher est -> higher threshold -> admits motion frames -> higher est).
    motion_floor_r = config_.motion_floor_rot_deg * CV_PI / 180.0;
    motion_floor_t = config_.motion_floor_trans_mm;
    if (config_.auto_noise) {
        meas_floor_std_r = std::max(est_r_, config_.meas_floor_rot_deg * CV_PI / 180.0);
        meas_floor_std_t = std::max(est_t_, config_.meas_floor_trans_mm);
    } else {
        meas_floor_std_r = config_.meas_floor_rot_deg * CV_PI / 180.0;
        meas_floor_std_t = config_.meas_floor_trans_mm;
    }

    // Measurement-noise floor on the Sigma diagonal so it never underestimates
    // the real per-frame pose noise (which over-trusts the measurement and lets
    // the velocity state overshoot -> amplified static jitter). Sigma still
    // carries the anisotropy / weak-observability mode off-diagonal.
    for (int i = 0; i < 3; ++i) {
        Sigma.at<double>(i, i) += meas_floor_std_r * meas_floor_std_r;
        Sigma.at<double>(i + 3, i + 3) += meas_floor_std_t * meas_floor_std_t;
    }

    // --- process noise: fixed or adaptive ---
    double q_r;
    double q_t;
    if (config_.adaptive) {
        // Below the motion floor the EMA reads as "still" and Q collapses to
        // the floor (heavy averaging); from the floor to 2x the floor Q ramps
        // linearly up to the moving value and saturates there, so fast motion
        // follows lag-free without unbounded velocity overshoot.
        const double a = config_.adaptive_ema_alpha;
        motion_ema_[0] = (1.0 - a) * motion_ema_[0] + a * drot;
        motion_ema_[1] = (1.0 - a) * motion_ema_[1] + a * dtrans;
        const double floor_r = config_.q_rotation_floor_deg * CV_PI / 180.0;
        const double move_r = config_.q_rotation_deg * CV_PI / 180.0;
        const double s_r = std::min(1.0, std::max(0.0,
            (motion_ema_[0] - motion_floor_r) / std::max(motion_floor_r, 1e-12)));
        const double s_t = std::min(1.0, std::max(0.0,
            (motion_ema_[1] - motion_floor_t) / std::max(motion_floor_t, 1e-12)));
        q_r = std::pow(floor_r + s_r * (move_r - floor_r), 2.0);
        q_t = std::pow(config_.q_translation_floor_mm +
                       s_t * (config_.q_translation_mm - config_.q_translation_floor_mm), 2.0);

        // Online noise estimate: feed the window ONLY on still frames, gated on
        // the INSTANTANEOUS delta (not the EMA). The EMA lags at motion onset,
        // so the first motion frames would slip in below the floor and inflate
        // the estimate, which then reclassifies the whole motion as "still"
        // (observed as slow-rotation lag). The instantaneous delta rejects the
        // onset frame immediately. Recompute the persisted quiet baseline
        // (median of the still deltas) for the next frame's floors.
        if (config_.auto_noise) {
            if (drot < motion_floor_r) {
                drot_hist_.push_back(drot);
                while (static_cast<int>(drot_hist_.size()) > config_.noise_window) {
                    drot_hist_.pop_front();
                }
                est_r_ = lowPercentile(drot_hist_, 0.5);
            }
            if (dtrans < motion_floor_t) {
                dtrans_hist_.push_back(dtrans);
                while (static_cast<int>(dtrans_hist_.size()) > config_.noise_window) {
                    dtrans_hist_.pop_front();
                }
                est_t_ = lowPercentile(dtrans_hist_, 0.5);
            }
        }
    } else {
        q_r = std::pow(config_.q_rotation_deg * CV_PI / 180.0, 2.0);
        q_t = std::pow(config_.q_translation_mm, 2.0);
    }
    prev_z_ = z.clone();

    // Constant-velocity predict; process noise on the acceleration (DWNA).
    cv::Mat F = cv::Mat::eye(12, 12, CV_64F);
    for (int i = 0; i < 6; ++i) {
        F.at<double>(i, i + 6) = 1.0;
    }
    cv::Mat Q = cv::Mat::zeros(12, 12, CV_64F);
    for (int i = 0; i < 6; ++i) {
        const double q = (i < 3) ? q_r : q_t;
        Q.at<double>(i, i) = 0.25 * q;
        Q.at<double>(i, i + 6) = 0.5 * q;
        Q.at<double>(i + 6, i) = 0.5 * q;
        Q.at<double>(i + 6, i + 6) = q;
    }
    x_ = F * x_;
    P_ = F * P_ * F.t() + Q;

    cv::Mat Hm = cv::Mat::zeros(6, 12, CV_64F);  // H = [I6 | 0]
    for (int i = 0; i < 6; ++i) {
        Hm.at<double>(i, i) = 1.0;
    }

    cv::Mat innov = z - Hm * x_;
    cv::Mat S = Hm * P_ * Hm.t() + Sigma;
    cv::Mat S_inv;
    cv::invert(S, S_inv, cv::DECOMP_SVD);
    const double m2 = cv::Mat(innov.t() * S_inv * innov).at<double>(0);
    if (m2 > config_.gate_mahalanobis && config_.gate_mahalanobis > 0.0) {
        // Spike: deweight the measurement instead of dropping it, so a
        // genuine fast motion still pulls the state over a few frames.
        Sigma *= m2 / config_.gate_mahalanobis;
        S = Hm * P_ * Hm.t() + Sigma;
        cv::invert(S, S_inv, cv::DECOMP_SVD);
        result.gated = true;
    }
    const cv::Mat Kg = P_ * Hm.t() * S_inv;
    x_ = x_ + Kg * innov;
    P_ = (cv::Mat::eye(12, 12, CV_64F) - Kg * Hm) * P_;

    for (int i = 0; i < 3; ++i) {
        result.rvec[static_cast<size_t>(i)] = x_.at<double>(i);
        result.tvec[static_cast<size_t>(i)] = x_.at<double>(i + 3);
    }
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            result.covariance[static_cast<size_t>(r * 6 + c)] =
                P_.at<double>(r, c);
        }
    }
    result.applied = true;
    result.mahalanobis = m2;
    return result;
}

}  // namespace hydramarker
