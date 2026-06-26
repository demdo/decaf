#include "tracker_depth_filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <opencv2/calib3d.hpp>

namespace hydramarker {

namespace {

double clampMin(double value, double minimum)
{
    return std::max(static_cast<double>(value), static_cast<double>(minimum));
}

double quietNaN()
{
    return std::numeric_limits<double>::quiet_NaN();
}

double pointZSpan(const std::vector<cv::Point3d>& points)
{
    if (points.empty()) {
        return quietNaN();
    }

    double min_z = points.front().z;
    double max_z = points.front().z;
    for (const cv::Point3d& point : points) {
        min_z = std::min(min_z, point.z);
        max_z = std::max(max_z, point.z);
    }
    return max_z - min_z;
}

} // namespace

PoseDepthKalmanFilter::PoseDepthKalmanFilter(
    const cv::Matx33d& K,
    const std::vector<double>& dist_coeffs,
    const PoseDepthFilterConfig& config
)
    : config_(config),
      K_(K),
      dist_coeffs_(makeDistCoeffsMat(dist_coeffs))
{
    config_.observation_std_mm = clampMin(config_.observation_std_mm, 1.0e-12);
    config_.process_std_mm = clampMin(config_.process_std_mm, 1.0e-12);
    config_.initial_velocity_std_mm =
        clampMin(config_.initial_velocity_std_mm, 1.0e-12);
    config_.innovation_guard_window =
        std::max(1, config_.innovation_guard_window);
    config_.innovation_guard_bias_threshold_mm =
        clampMin(config_.innovation_guard_bias_threshold_mm, 0.0);
    config_.innovation_guard_min_same_sign =
        std::max(1, config_.innovation_guard_min_same_sign);
    config_.innovation_cusum_slack_mm =
        clampMin(config_.innovation_cusum_slack_mm, 0.0);
    config_.innovation_cusum_threshold_mm =
        clampMin(config_.innovation_cusum_threshold_mm, 0.0);
    config_.negative_delta_guard_min_z_span_mm =
        clampMin(config_.negative_delta_guard_min_z_span_mm, 0.0);
    config_.negative_delta_guard_max_negative_delta_mm =
        clampMin(config_.negative_delta_guard_max_negative_delta_mm, 0.0);
    config_.negative_delta_guard_hold_min_negative_delta_mm =
        clampMin(config_.negative_delta_guard_hold_min_negative_delta_mm, 0.0);
    config_.negative_delta_guard_max_hold_correction_mm =
        clampMin(config_.negative_delta_guard_max_hold_correction_mm, 0.0);
    config_.negative_delta_guard_velocity_damping = std::clamp(
        config_.negative_delta_guard_velocity_damping,
        0.0,
        1.0
    );
}

void PoseDepthKalmanFilter::reset()
{
    has_state_ = false;
    x_ = cv::Vec2d(0.0, 0.0);
    P_ = cv::Matx22d::zeros();
    innovation_history_.clear();
    innovation_cusum_pos_ = 0.0;
    innovation_cusum_neg_ = 0.0;
}

PoseDepthFilterSnapshot PoseDepthKalmanFilter::snapshot() const
{
    PoseDepthFilterSnapshot state;
    state.has_state = has_state_;
    if (has_state_) {
        state.x = {x_[0], x_[1]};
        state.P = {P_(0, 0), P_(0, 1), P_(1, 0), P_(1, 1)};
    }
    state.innovation_history.assign(
        innovation_history_.begin(),
        innovation_history_.end()
    );
    state.innovation_cusum_pos = innovation_cusum_pos_;
    state.innovation_cusum_neg = innovation_cusum_neg_;
    return state;
}

void PoseDepthKalmanFilter::restore(const PoseDepthFilterSnapshot& state)
{
    has_state_ = state.has_state;
    if (has_state_) {
        x_ = cv::Vec2d(state.x[0], state.x[1]);
        P_ = cv::Matx22d(
            state.P[0],
            state.P[1],
            state.P[2],
            state.P[3]
        );
    } else {
        x_ = cv::Vec2d(0.0, 0.0);
        P_ = cv::Matx22d::zeros();
    }

    innovation_history_.clear();
    for (double value : state.innovation_history) {
        innovation_history_.push_back(value);
    }
    while (static_cast<int>(innovation_history_.size()) >
           config_.innovation_guard_window) {
        innovation_history_.pop_front();
    }
    innovation_cusum_pos_ = state.innovation_cusum_pos;
    innovation_cusum_neg_ = state.innovation_cusum_neg;
}

PoseDepthFilterResult PoseDepthKalmanFilter::update(
    const std::vector<double>& rvec,
    const std::vector<double>& tvec,
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points
)
{
    cv::Mat rvec_mat;
    cv::Mat raw_tvec;
    if (!vectorToMat3x1(rvec, rvec_mat) || !vectorToMat3x1(tvec, raw_tvec)) {
        throw std::runtime_error("rvec and tvec must contain exactly 3 values");
    }

    const double raw_z = raw_tvec.at<double>(2, 0);
    const double previous_filtered_z = has_state_ ? x_[0] : quietNaN();
    auto [filtered_z_initial, innovation_diag] = updateState(raw_z);

    double filtered_z = filtered_z_initial;
    cv::Mat out_tvec = raw_tvec.clone();
    out_tvec.at<double>(2, 0) = filtered_z;

    const double object_z_span = pointZSpan(object_points);
    const double raw_rms =
        reprojectionRms(object_points, image_points, rvec_mat, raw_tvec);
    double filtered_rms =
        reprojectionRms(object_points, image_points, rvec_mat, out_tvec);
    double guard_alpha = 1.0;

    if (
        config_.reprojection_guard_px > 0.0 &&
        std::isfinite(raw_rms) &&
        std::isfinite(filtered_rms) &&
        filtered_rms - raw_rms > config_.reprojection_guard_px
    ) {
        double accepted_z = raw_z;
        double accepted_rms = raw_rms;
        double lo = 0.0;
        double hi = 1.0;

        for (int iter = 0; iter < 20; ++iter) {
            const double alpha = 0.5 * (lo + hi);
            const double candidate_z =
                raw_z + alpha * (filtered_z - raw_z);
            cv::Mat candidate_tvec = raw_tvec.clone();
            candidate_tvec.at<double>(2, 0) = candidate_z;
            const double candidate_rms = reprojectionRms(
                object_points,
                image_points,
                rvec_mat,
                candidate_tvec
            );
            if (
                std::isfinite(candidate_rms) &&
                candidate_rms - raw_rms <= config_.reprojection_guard_px
            ) {
                accepted_z = candidate_z;
                accepted_rms = candidate_rms;
                guard_alpha = alpha;
                lo = alpha;
            } else {
                hi = alpha;
            }
        }

        filtered_z = accepted_z;
        filtered_rms = accepted_rms;
        out_tvec = raw_tvec.clone();
        out_tvec.at<double>(2, 0) = filtered_z;
        if (has_state_) {
            x_[0] = filtered_z;
        }
    }

    const std::optional<double> limited_z =
        limitGeometryGatedNegativeDelta(
            raw_z,
            filtered_z,
            object_z_span,
            previous_filtered_z,
            innovation_diag.innovation_bias_detected
        );
    const bool negative_delta_limited = limited_z.has_value();
    if (negative_delta_limited) {
        filtered_z = *limited_z;
        out_tvec = raw_tvec.clone();
        out_tvec.at<double>(2, 0) = filtered_z;
        filtered_rms = reprojectionRms(
            object_points,
            image_points,
            rvec_mat,
            out_tvec
        );
    }

    const cv::Matx44d T = makeTransform(rvec_mat, out_tvec);

    PoseDepthFilterResult result;
    result.rvec = mat3x1ToVector(rvec_mat);
    result.tvec = mat3x1ToVector(out_tvec);
    result.T_marker_camera = transformToVector(T);
    result.raw_z_mm = raw_z;
    result.filtered_z_mm = filtered_z;
    result.delta_z_mm = filtered_z - raw_z;
    result.raw_reprojection_rms_px = raw_rms;
    result.filtered_reprojection_rms_px = filtered_rms;
    result.reprojection_excess_px = filtered_rms - raw_rms;
    result.guard_alpha = guard_alpha;
    result.applied = std::abs(filtered_z - raw_z) > 1.0e-12;
    result.innovation_z_mm = innovation_diag.innovation_z_mm;
    result.innovation_mean_z_mm = innovation_diag.innovation_mean_z_mm;
    result.innovation_cusum_pos_mm = innovation_diag.innovation_cusum_pos_mm;
    result.innovation_cusum_neg_mm = innovation_diag.innovation_cusum_neg_mm;
    result.innovation_bias_detected =
        innovation_diag.innovation_bias_detected;
    result.innovation_bias_direction =
        innovation_diag.innovation_bias_direction;
    result.innovation_bias_limited =
        innovation_diag.innovation_bias_limited;
    result.object_z_span_mm = object_z_span;
    result.negative_delta_guard_limited = negative_delta_limited;
    return result;
}

std::pair<double, PoseDepthKalmanFilter::InnovationDiagnostics>
PoseDepthKalmanFilter::updateState(double measurement_z_mm)
{
    InnovationDiagnostics zero_diag;
    const double q = config_.process_std_mm;
    const double r = config_.observation_std_mm;
    const double r2 = r * r;

    if (!has_state_) {
        has_state_ = true;
        x_ = cv::Vec2d(measurement_z_mm, 0.0);
        P_ = cv::Matx22d(
            r2,
            0.0,
            0.0,
            config_.initial_velocity_std_mm *
                config_.initial_velocity_std_mm
        );
        innovation_history_.clear();
        innovation_cusum_pos_ = 0.0;
        innovation_cusum_neg_ = 0.0;
        return {x_[0], zero_diag};
    }

    const cv::Matx22d F(1.0, 1.0, 0.0, 1.0);
    const cv::Vec2d x_pred(x_[0] + x_[1], x_[1]);
    const cv::Matx22d Q(
        0.25 * q * q,
        0.5 * q * q,
        0.5 * q * q,
        q * q
    );
    const cv::Matx22d P_pred = F * P_ * F.t() + Q;

    const double innovation = measurement_z_mm - x_pred[0];
    InnovationDiagnostics diag = updateInnovationMonitor(innovation);
    const double S = P_pred(0, 0) + r2;
    const double K0 = P_pred(0, 0) / S;
    const double K1 = P_pred(1, 0) / S;

    x_[0] = x_pred[0] + K0 * innovation;
    x_[1] = x_pred[1] + K1 * innovation;

    P_ = cv::Matx22d(
        (1.0 - K0) * P_pred(0, 0),
        (1.0 - K0) * P_pred(0, 1),
        P_pred(1, 0) - K1 * P_pred(0, 0),
        P_pred(1, 1) - K1 * P_pred(0, 1)
    );

    return {x_[0], diag};
}

PoseDepthKalmanFilter::InnovationDiagnostics
PoseDepthKalmanFilter::updateInnovationMonitor(double innovation_z_mm)
{
    InnovationDiagnostics diag;
    diag.innovation_z_mm = innovation_z_mm;

    if (!config_.innovation_guard_enabled) {
        return diag;
    }

    innovation_history_.push_back(innovation_z_mm);
    while (static_cast<int>(innovation_history_.size()) >
           config_.innovation_guard_window) {
        innovation_history_.pop_front();
    }

    const double sum = std::accumulate(
        innovation_history_.begin(),
        innovation_history_.end(),
        0.0
    );
    const double mean = sum / static_cast<double>(innovation_history_.size());
    int direction = mean > 0.0 ? 1 : mean < 0.0 ? -1 : 0;
    int same_sign = 0;
    if (direction > 0) {
        for (double value : innovation_history_) {
            if (value >= 0.0) ++same_sign;
        }
    } else if (direction < 0) {
        for (double value : innovation_history_) {
            if (value <= 0.0) ++same_sign;
        }
    }

    const double slack = config_.innovation_cusum_slack_mm;
    innovation_cusum_pos_ =
        std::max(0.0, innovation_cusum_pos_ + innovation_z_mm - slack);
    innovation_cusum_neg_ =
        std::max(0.0, innovation_cusum_neg_ - innovation_z_mm - slack);

    const bool enough_history =
        static_cast<int>(innovation_history_.size()) >=
        config_.innovation_guard_window;
    const bool enough_same_sign =
        same_sign >= std::min(
            config_.innovation_guard_min_same_sign,
            config_.innovation_guard_window
        );
    const bool running_bias =
        enough_history &&
        enough_same_sign &&
        std::abs(mean) >= config_.innovation_guard_bias_threshold_mm;

    bool cusum_bias = false;
    if (config_.innovation_cusum_threshold_mm > 0.0 && enough_history) {
        if (mean > 0.0) {
            cusum_bias =
                innovation_cusum_pos_ >=
                config_.innovation_cusum_threshold_mm;
        } else if (mean < 0.0) {
            cusum_bias =
                innovation_cusum_neg_ >=
                config_.innovation_cusum_threshold_mm;
        }
    }

    const bool bias_detected = running_bias || cusum_bias;
    if (running_bias) {
        direction = mean > 0.0 ? 1 : mean < 0.0 ? -1 : 0;
    } else if (innovation_cusum_pos_ > innovation_cusum_neg_) {
        direction = 1;
    } else if (innovation_cusum_neg_ > innovation_cusum_pos_) {
        direction = -1;
    } else {
        direction = 0;
    }

    diag.innovation_mean_z_mm = mean;
    diag.innovation_cusum_pos_mm = innovation_cusum_pos_;
    diag.innovation_cusum_neg_mm = innovation_cusum_neg_;
    diag.innovation_bias_detected = bias_detected;
    diag.innovation_bias_direction = direction;
    diag.innovation_bias_limited = false;
    return diag;
}

std::optional<double> PoseDepthKalmanFilter::limitGeometryGatedNegativeDelta(
    double raw_z,
    double filtered_z,
    double object_z_span_mm,
    double previous_filtered_z,
    bool innovation_bias_detected
)
{
    if (!config_.negative_delta_guard_enabled) {
        return std::nullopt;
    }
    if (!std::isfinite(object_z_span_mm)) {
        return std::nullopt;
    }
    if (object_z_span_mm < config_.negative_delta_guard_min_z_span_mm) {
        return std::nullopt;
    }

    double min_allowed_z =
        raw_z - config_.negative_delta_guard_max_negative_delta_mm;
    bool use_hold =
        config_.negative_delta_guard_hold_previous_z &&
        (
            !config_.negative_delta_guard_hold_requires_innovation_bias ||
            innovation_bias_detected
        );
    use_hold =
        use_hold &&
        raw_z - filtered_z >=
            config_.negative_delta_guard_hold_min_negative_delta_mm;

    if (use_hold && std::isfinite(previous_filtered_z)) {
        double held_z = previous_filtered_z;
        held_z = std::min(
            held_z,
            raw_z + config_.negative_delta_guard_max_hold_correction_mm
        );
        min_allowed_z = std::max(min_allowed_z, held_z);
    }

    if (filtered_z >= min_allowed_z) {
        return std::nullopt;
    }

    if (has_state_) {
        x_[0] = min_allowed_z;
        x_[1] *= config_.negative_delta_guard_velocity_damping;
        P_(0, 1) = 0.0;
        P_(1, 0) = 0.0;
    }
    return min_allowed_z;
}

double PoseDepthKalmanFilter::reprojectionRms(
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points,
    const cv::Mat& rvec,
    const cv::Mat& tvec
) const
{
    if (object_points.empty()) {
        return quietNaN();
    }
    if (object_points.size() != image_points.size()) {
        return quietNaN();
    }

    std::vector<cv::Point2d> projected;
    try {
        cv::projectPoints(
            object_points,
            rvec,
            tvec,
            K_,
            dist_coeffs_,
            projected
        );
    } catch (...) {
        return quietNaN();
    }

    double err2_sum = 0.0;
    for (size_t idx = 0; idx < projected.size(); ++idx) {
        const double du = projected[idx].x - image_points[idx].x;
        const double dv = projected[idx].y - image_points[idx].y;
        err2_sum += du * du + dv * dv;
    }
    return std::sqrt(err2_sum / static_cast<double>(projected.size()));
}

cv::Mat PoseDepthKalmanFilter::makeDistCoeffsMat(
    const std::vector<double>& dist_coeffs
)
{
    if (dist_coeffs.empty()) {
        return cv::Mat();
    }

    cv::Mat mat(static_cast<int>(dist_coeffs.size()), 1, CV_64F);
    for (int i = 0; i < static_cast<int>(dist_coeffs.size()); ++i) {
        mat.at<double>(i, 0) = dist_coeffs[static_cast<size_t>(i)];
    }
    return mat;
}

bool PoseDepthKalmanFilter::vectorToMat3x1(
    const std::vector<double>& values,
    cv::Mat& mat
)
{
    if (values.size() != 3) {
        return false;
    }

    mat = cv::Mat(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        mat.at<double>(i, 0) = values[static_cast<size_t>(i)];
    }
    return true;
}

std::vector<double> PoseDepthKalmanFilter::mat3x1ToVector(const cv::Mat& mat)
{
    std::vector<double> out(3, 0.0);
    for (int i = 0; i < 3; ++i) {
        out[static_cast<size_t>(i)] = mat.at<double>(i, 0);
    }
    return out;
}

std::vector<double> PoseDepthKalmanFilter::transformToVector(
    const cv::Matx44d& T
)
{
    std::vector<double> out;
    out.reserve(16);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out.push_back(T(r, c));
        }
    }
    return out;
}

cv::Matx44d PoseDepthKalmanFilter::makeTransform(
    const cv::Mat& rvec,
    const cv::Mat& tvec
)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    cv::Matx44d T = cv::Matx44d::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            T(r, c) = R.at<double>(r, c);
        }
        T(r, 3) = tvec.at<double>(r, 0);
    }
    return T;
}

} // namespace hydramarker
