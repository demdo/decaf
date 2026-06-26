#include "tracker_pose_prior.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/calib3d.hpp>

namespace hydramarker {

namespace {

double quietNaN()
{
    return std::numeric_limits<double>::quiet_NaN();
}

cv::Mat makeDistCoeffsMat(const std::vector<double>& dist_coeffs)
{
    if (dist_coeffs.empty()) {
        return cv::Mat();
    }
    cv::Mat dist(static_cast<int>(dist_coeffs.size()), 1, CV_64F);
    for (int i = 0; i < static_cast<int>(dist_coeffs.size()); ++i) {
        dist.at<double>(i, 0) = dist_coeffs[static_cast<size_t>(i)];
    }
    return dist;
}

bool vectorToMat3x1(const std::vector<double>& values, cv::Mat& mat)
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

std::vector<double> mat3x1ToVector(const cv::Mat& mat)
{
    std::vector<double> values(3, 0.0);
    for (int i = 0; i < 3; ++i) {
        values[static_cast<size_t>(i)] = mat.at<double>(i, 0);
    }
    return values;
}

std::vector<double> transformToVector(const cv::Matx44d& T)
{
    std::vector<double> values;
    values.reserve(16);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            values.push_back(T(r, c));
        }
    }
    return values;
}

cv::Matx33d skew(const cv::Vec3d& v)
{
    return cv::Matx33d(
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0
    );
}

cv::Matx33d expSo3(const cv::Vec3d& w)
{
    const double theta = cv::norm(w);
    const cv::Matx33d W = skew(w);
    const cv::Matx33d I = cv::Matx33d::eye();
    if (theta < 1.0e-12) {
        return I + W + 0.5 * (W * W);
    }

    const double a = std::sin(theta) / theta;
    const double b = (1.0 - std::cos(theta)) / (theta * theta);
    return I + a * W + b * (W * W);
}

cv::Matx33d leftJacobianSo3(const cv::Vec3d& w)
{
    const double theta = cv::norm(w);
    const cv::Matx33d W = skew(w);
    const cv::Matx33d I = cv::Matx33d::eye();
    if (theta < 1.0e-12) {
        return I + 0.5 * W + (1.0 / 6.0) * (W * W);
    }

    const double b = (1.0 - std::cos(theta)) / (theta * theta);
    const double c = (theta - std::sin(theta)) /
        (theta * theta * theta);
    return I + b * W + c * (W * W);
}

cv::Matx44d expSe3(const cv::Vec<double, 6>& delta)
{
    const cv::Vec3d v(delta[0], delta[1], delta[2]);
    const cv::Vec3d w(delta[3], delta[4], delta[5]);
    const cv::Matx33d R = expSo3(w);
    const cv::Matx33d V = leftJacobianSo3(w);
    const cv::Vec3d t = V * v;

    cv::Matx44d T = cv::Matx44d::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            T(r, c) = R(r, c);
        }
        T(r, 3) = t[r];
    }
    return T;
}

cv::Matx44d makeTransform(const cv::Mat& rvec, const cv::Mat& tvec)
{
    cv::Mat R_mat;
    cv::Rodrigues(rvec, R_mat);
    cv::Matx44d T = cv::Matx44d::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            T(r, c) = R_mat.at<double>(r, c);
        }
        T(r, 3) = tvec.at<double>(r, 0);
    }
    return T;
}

void transformToPose(const cv::Matx44d& T, cv::Mat& rvec, cv::Mat& tvec)
{
    cv::Mat R(3, 3, CV_64F);
    tvec = cv::Mat(3, 1, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R.at<double>(r, c) = T(r, c);
        }
        tvec.at<double>(r, 0) = T(r, 3);
    }
    cv::Rodrigues(R, rvec);
}

bool projectPose(
    const std::vector<cv::Point3d>& object_points,
    const cv::Matx33d& K,
    const cv::Mat& dist,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    std::vector<cv::Point2d>& projected
)
{
    try {
        cv::projectPoints(object_points, rvec, tvec, K, dist, projected);
    } catch (const cv::Exception&) {
        return false;
    }
    return projected.size() == object_points.size();
}

bool projectTransform(
    const std::vector<cv::Point3d>& object_points,
    const cv::Matx44d& T,
    const cv::Matx33d& K,
    const cv::Mat& dist,
    std::vector<cv::Point2d>& projected
)
{
    cv::Mat rvec;
    cv::Mat tvec;
    transformToPose(T, rvec, tvec);
    return projectPose(object_points, K, dist, rvec, tvec, projected);
}

bool poseStats(
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points,
    const cv::Matx33d& K,
    const cv::Mat& dist,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    double& mean_px,
    double& max_px
)
{
    std::vector<cv::Point2d> projected;
    if (!projectPose(object_points, K, dist, rvec, tvec, projected)) {
        return false;
    }
    if (projected.empty() || projected.size() != image_points.size()) {
        return false;
    }

    double sum = 0.0;
    max_px = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
        const double error = cv::norm(projected[i] - image_points[i]);
        sum += error;
        max_px = std::max(max_px, error);
    }
    mean_px = sum / static_cast<double>(projected.size());
    return true;
}

PlateauPosePriorResult candidateFromPose(
    const std::string& method,
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points,
    const cv::Matx33d& K,
    const cv::Mat& dist,
    const cv::Mat& raw_tvec,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    double raw_mean_px,
    double raw_max_px,
    int iterations
)
{
    double mean_px = quietNaN();
    double max_px = quietNaN();
    if (
        !poseStats(
            object_points,
            image_points,
            K,
            dist,
            rvec,
            tvec,
            mean_px,
            max_px
        )
    ) {
        PlateauPosePriorResult failed;
        failed.method = method;
        failed.reason = "projection_failed";
        return failed;
    }

    const cv::Matx44d T = makeTransform(rvec, tvec);
    PlateauPosePriorResult result;
    result.success = true;
    result.method = method;
    result.rvec = mat3x1ToVector(rvec);
    result.tvec = mat3x1ToVector(tvec);
    result.T_marker_camera = transformToVector(T);
    result.reprojection_mean_px = mean_px;
    result.reprojection_max_px = max_px;
    result.reprojection_excess_px = mean_px - raw_mean_px;
    result.max_reprojection_excess_px = max_px - raw_max_px;
    result.delta_z_mm =
        tvec.at<double>(2, 0) - raw_tvec.at<double>(2, 0);
    result.iterations = iterations;
    return result;
}

bool candidateAllowed(
    const PlateauPosePriorResult& candidate,
    const PlateauPosePriorConfig& config
)
{
    if (!candidate.success) {
        return false;
    }
    if (!std::isfinite(candidate.reprojection_excess_px)) {
        return false;
    }
    if (candidate.method == "static") {
        if (candidate.reprojection_excess_px > config.static_max_excess_px) {
            return false;
        }
    } else if (
        candidate.reprojection_excess_px > config.candidate_max_excess_px
    ) {
        return false;
    }
    if (
        std::isfinite(candidate.max_reprojection_excess_px) &&
        candidate.max_reprojection_excess_px >
            config.candidate_max_max_excess_px
    ) {
        return false;
    }
    if (candidate.delta_z_mm < config.min_positive_z_correction_mm) {
        return false;
    }
    if (candidate.delta_z_mm > config.max_positive_z_correction_mm) {
        return false;
    }
    return true;
}

bool numericMotionJacobian(
    const std::vector<cv::Point3d>& object_points,
    const cv::Matx44d& T,
    const cv::Matx33d& K,
    const cv::Mat& dist,
    cv::Mat& J
)
{
    std::vector<cv::Point2d> base;
    if (!projectTransform(object_points, T, K, dist, base)) {
        return false;
    }

    const int rows = static_cast<int>(base.size()) * 2;
    J = cv::Mat::zeros(rows, 6, CV_64F);
    for (int col = 0; col < 6; ++col) {
        const double eps = col < 3 ? 1.0e-3 : 1.0e-6;
        cv::Vec<double, 6> delta(0, 0, 0, 0, 0, 0);
        delta[col] = eps;
        const cv::Matx44d shifted_T = expSe3(delta) * T;
        std::vector<cv::Point2d> shifted;
        if (!projectTransform(object_points, shifted_T, K, dist, shifted)) {
            return false;
        }
        for (size_t i = 0; i < base.size(); ++i) {
            J.at<double>(static_cast<int>(i) * 2, col) =
                (shifted[i].x - base[i].x) / eps;
            J.at<double>(static_cast<int>(i) * 2 + 1, col) =
                (shifted[i].y - base[i].y) / eps;
        }
    }
    return true;
}

PlateauPosePriorResult solveRobustIrls(
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points,
    const cv::Matx33d& K,
    const cv::Mat& dist,
    const cv::Mat& raw_tvec,
    const cv::Mat& seed_rvec,
    const cv::Mat& seed_tvec,
    double raw_mean_px,
    double raw_max_px,
    const PlateauPosePriorConfig& config
)
{
    cv::Matx44d T;
    try {
        T = makeTransform(seed_rvec, seed_tvec);
    } catch (const cv::Exception&) {
        PlateauPosePriorResult failed;
        failed.method = "robust_irls";
        failed.reason = "seed_transform_failed";
        return failed;
    }

    int last_iteration = 0;
    const int max_iterations = std::max(1, config.max_iterations);
    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        last_iteration = iteration;
        std::vector<cv::Point2d> projected;
        if (!projectTransform(object_points, T, K, dist, projected)) {
            PlateauPosePriorResult failed;
            failed.method = "robust_irls";
            failed.reason = "projection_failed";
            return failed;
        }

        cv::Mat residual(static_cast<int>(projected.size()) * 2, 1, CV_64F);
        std::vector<double> weights(projected.size(), 0.0);
        double max_weight = 0.0;
        const double robust_c = std::max(config.robust_c_px, 1.0e-9);
        for (size_t i = 0; i < projected.size(); ++i) {
            const cv::Point2d diff = projected[i] - image_points[i];
            residual.at<double>(static_cast<int>(i) * 2, 0) = diff.x;
            residual.at<double>(static_cast<int>(i) * 2 + 1, 0) = diff.y;
            const double error = cv::norm(diff);
            const double weight = 1.0 / (robust_c + error);
            if (std::isfinite(weight) && weight > 0.0) {
                weights[i] = weight;
                max_weight = std::max(max_weight, weight);
            }
        }
        if (max_weight > 0.0) {
            for (double& weight : weights) {
                weight /= max_weight;
            }
        }

        cv::Mat J;
        if (!numericMotionJacobian(object_points, T, K, dist, J)) {
            PlateauPosePriorResult failed;
            failed.method = "robust_irls";
            failed.reason = "jacobian_failed";
            return failed;
        }

        cv::Mat weighted_J = J.clone();
        cv::Mat weighted_residual = residual.clone();
        for (size_t i = 0; i < weights.size(); ++i) {
            const double weight = std::isfinite(weights[i]) ? weights[i] : 0.0;
            weighted_J.row(static_cast<int>(i) * 2) *= weight;
            weighted_J.row(static_cast<int>(i) * 2 + 1) *= weight;
            weighted_residual.at<double>(static_cast<int>(i) * 2, 0) *= weight;
            weighted_residual.at<double>(
                static_cast<int>(i) * 2 + 1,
                0
            ) *= weight;
        }

        cv::Mat C = J.t() * weighted_J;
        cv::Mat g = J.t() * weighted_residual;
        cv::Mat normal = C.clone();
        for (int i = 0; i < 6; ++i) {
            const double damping =
                config.lm_damping * std::max(C.at<double>(i, i), 1.0e-9);
            normal.at<double>(i, i) += damping;
        }

        cv::Mat delta_mat;
        if (!cv::solve(normal, -g, delta_mat, cv::DECOMP_LU)) {
            cv::solve(normal, -g, delta_mat, cv::DECOMP_SVD);
        }

        cv::Vec<double, 6> delta;
        for (int i = 0; i < 6; ++i) {
            delta[i] = delta_mat.at<double>(i, 0);
        }

        const double t_norm = std::sqrt(
            delta[0] * delta[0] +
            delta[1] * delta[1] +
            delta[2] * delta[2]
        );
        const double r_norm = std::sqrt(
            delta[3] * delta[3] +
            delta[4] * delta[4] +
            delta[5] * delta[5]
        );
        const double max_t =
            std::max(config.max_step_translation_mm, 1.0e-9);
        const double max_r =
            CV_PI / 180.0 * std::max(config.max_step_rotation_deg, 1.0e-9);
        double scale = 1.0;
        if (t_norm > max_t) {
            scale = std::min(scale, max_t / std::max(t_norm, 1.0e-12));
        }
        if (r_norm > max_r) {
            scale = std::min(scale, max_r / std::max(r_norm, 1.0e-12));
        }
        for (int i = 0; i < 6; ++i) {
            delta[i] *= scale;
        }

        T = expSe3(delta) * T;
        if (t_norm < 1.0e-5 && r_norm < 1.0e-8) {
            break;
        }
    }

    cv::Mat rvec;
    cv::Mat tvec;
    transformToPose(T, rvec, tvec);
    return candidateFromPose(
        "robust_irls",
        object_points,
        image_points,
        K,
        dist,
        raw_tvec,
        rvec,
        tvec,
        raw_mean_px,
        raw_max_px,
        last_iteration
    );
}

} // namespace

PlateauPosePriorResult solvePlateauPosePrior(
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points,
    const cv::Matx33d& K,
    const std::vector<double>& dist_coeffs,
    const std::vector<double>& raw_rvec,
    const std::vector<double>& raw_tvec,
    const std::vector<double>& seed_rvec,
    const std::vector<double>& seed_tvec,
    const PlateauPosePriorConfig& config
)
{
    if (object_points.size() < 6 || object_points.size() != image_points.size()) {
        PlateauPosePriorResult failed;
        failed.reason = "invalid_points";
        return failed;
    }

    cv::Mat raw_r;
    cv::Mat raw_t;
    cv::Mat seed_r;
    cv::Mat seed_t;
    if (
        !vectorToMat3x1(raw_rvec, raw_r) ||
        !vectorToMat3x1(raw_tvec, raw_t) ||
        !vectorToMat3x1(seed_rvec, seed_r) ||
        !vectorToMat3x1(seed_tvec, seed_t)
    ) {
        PlateauPosePriorResult failed;
        failed.reason = "invalid_pose_vectors";
        return failed;
    }

    const cv::Mat dist = makeDistCoeffsMat(dist_coeffs);
    double raw_mean_px = quietNaN();
    double raw_max_px = quietNaN();
    if (
        !poseStats(
            object_points,
            image_points,
            K,
            dist,
            raw_r,
            raw_t,
            raw_mean_px,
            raw_max_px
        )
    ) {
        PlateauPosePriorResult failed;
        failed.reason = "raw_projection_failed";
        return failed;
    }

    PlateauPosePriorResult stat = candidateFromPose(
        "static",
        object_points,
        image_points,
        K,
        dist,
        raw_t,
        seed_r,
        seed_t,
        raw_mean_px,
        raw_max_px,
        0
    );
    if (candidateAllowed(stat, config)) {
        return stat;
    }

    PlateauPosePriorResult robust = solveRobustIrls(
        object_points,
        image_points,
        K,
        dist,
        raw_t,
        seed_r,
        seed_t,
        raw_mean_px,
        raw_max_px,
        config
    );
    if (candidateAllowed(robust, config)) {
        return robust;
    }

    PlateauPosePriorResult failed;
    failed.method = "none";
    failed.reason = robust.reason.empty()
        ? "no_candidate_within_reprojection_budget"
        : robust.reason;
    failed.reprojection_mean_px = robust.reprojection_mean_px;
    failed.reprojection_max_px = robust.reprojection_max_px;
    failed.reprojection_excess_px = robust.reprojection_excess_px;
    failed.max_reprojection_excess_px = robust.max_reprojection_excess_px;
    failed.delta_z_mm = robust.delta_z_mm;
    failed.iterations = robust.iterations;
    return failed;
}

} // namespace hydramarker
