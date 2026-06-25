#include "tracker_pose.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <tuple>

#include <opencv2/calib3d.hpp>

namespace hydramarker {

namespace {

cv::Mat makeDistCoeffsMat(const std::vector<double>& dist_coeffs)
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

std::string lowerAscii(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
    );
    return value;
}

} // namespace

MapPoseTracker::MapPoseTracker(
    const cv::Matx33d& K,
    const std::vector<double>& dist_coeffs,
    const MapPoseTrackerConfig& config
)
    : config_(config),
      K_(K),
      dist_coeffs_(makeDistCoeffsMat(dist_coeffs))
{
}

void MapPoseTracker::reset()
{
    has_pose_ = false;
    rvec_.release();
    tvec_.release();
    T_marker_camera_ = cv::Matx44d::eye();
}

MapPoseResult MapPoseTracker::estimatePose(
    const std::vector<PoseTrackPoint>& points,
    int lost_frames
)
{
    if (static_cast<int>(points.size()) < config_.min_points) {
        MapPoseResult result;
        result.success = false;
        result.message =
            "Too few points: " + std::to_string(points.size()) +
            " < " + std::to_string(config_.min_points);
        result.num_points = static_cast<int>(points.size());
        return result;
    }

    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2d> image_points;
    object_points.reserve(points.size());
    image_points.reserve(points.size());

    for (const PoseTrackPoint& point : points) {
        object_points.emplace_back(
            point.xyz_mm[0],
            point.xyz_mm[1],
            point.xyz_mm[2]
        );
        image_points.emplace_back(point.uv[0], point.uv[1]);
    }

    const bool use_guess =
        config_.use_pose_prior && has_pose_ &&
        !rvec_.empty() && !tvec_.empty();

    if (use_guess && config_.use_direct_prior_solver) {
        MapPoseResult direct = estimatePoseDirectPrior(
            points,
            object_points,
            image_points,
            lost_frames
        );
        if (direct.success) {
            return direct;
        }
    }

    cv::Mat rvec = use_guess ? rvec_.clone() : cv::Mat();
    cv::Mat tvec = use_guess ? tvec_.clone() : cv::Mat();
    cv::Mat inliers;
    bool success = false;

    try {
        success = cv::solvePnPRansac(
            object_points,
            image_points,
            K_,
            dist_coeffs_,
            rvec,
            tvec,
            use_guess,
            config_.ransac_iterations,
            static_cast<float>(config_.ransac_reproj_px),
            config_.ransac_confidence,
            inliers,
            cv::SOLVEPNP_ITERATIVE
        );
    } catch (const cv::Exception& e) {
        MapPoseResult result;
        result.success = false;
        result.message = std::string("solvePnPRansac failed: ") + e.what();
        result.num_points = static_cast<int>(points.size());
        result.method = "ransac_iterative";
        return result;
    } catch (const std::exception& e) {
        MapPoseResult result;
        result.success = false;
        result.message = std::string("solvePnPRansac failed: ") + e.what();
        result.num_points = static_cast<int>(points.size());
        result.method = "ransac_iterative";
        return result;
    }

    const int inlier_count = success ? static_cast<int>(inliers.total()) : 0;
    if (!success || inlier_count < config_.min_inliers) {
        MapPoseResult result;
        result.success = false;
        result.message = "Too few inliers: " + std::to_string(inlier_count);
        result.num_points = static_cast<int>(points.size());
        result.num_inliers = inlier_count;
        result.method = "ransac_iterative";
        return result;
    }

    std::vector<int> inlier_indices;
    std::vector<cv::Point3d> object_inliers;
    std::vector<cv::Point2d> image_inliers;
    inlier_indices.reserve(static_cast<size_t>(inlier_count));
    object_inliers.reserve(static_cast<size_t>(inlier_count));
    image_inliers.reserve(static_cast<size_t>(inlier_count));

    for (int row = 0; row < inliers.rows; ++row) {
        const int idx = inliers.at<int>(row, 0);
        if (idx < 0 || idx >= static_cast<int>(points.size())) {
            continue;
        }
        inlier_indices.push_back(idx);
        object_inliers.push_back(object_points[static_cast<size_t>(idx)]);
        image_inliers.push_back(image_points[static_cast<size_t>(idx)]);
    }

    if (config_.refine_with_iterative) {
        try {
            cv::Mat rvec_ref = rvec.clone();
            cv::Mat tvec_ref = tvec.clone();
            const bool refine_success = cv::solvePnP(
                object_inliers,
                image_inliers,
                K_,
                dist_coeffs_,
                rvec_ref,
                tvec_ref,
                true,
                cv::SOLVEPNP_ITERATIVE
            );
            if (refine_success) {
                rvec = rvec_ref.reshape(1, 3).clone();
                tvec = tvec_ref.reshape(1, 3).clone();
            }
        } catch (...) {
        }
    }

    double mean_error_px = -1.0;
    double max_error_px = -1.0;
    if (!computeReprojectionStats(
            object_inliers,
            image_inliers,
            rvec,
            tvec,
            mean_error_px,
            max_error_px
        )) {
        MapPoseResult result;
        result.success = false;
        result.message = "projectPoints failed.";
        result.rvec = mat3x1ToVector(rvec);
        result.tvec = mat3x1ToVector(tvec);
        result.T_marker_camera = transformToVector(makeTransform(rvec, tvec));
        result.num_points = static_cast<int>(points.size());
        result.num_inliers = static_cast<int>(inlier_indices.size());
        result.method = "ransac_iterative";
        return result;
    }

    if (
        mean_error_px > config_.max_mean_reproj_px ||
        max_error_px > config_.max_max_reproj_px
    ) {
        if (mean_error_px > config_.max_mean_reproj_px * 3.0) {
            reset();
        }

        MapPoseResult result;
        result.success = false;
        result.message =
            "Reprojection error too high (mean=" +
            formatDouble(mean_error_px, 3) +
            ", max=" + formatDouble(max_error_px, 3) + ")";
        result.rvec = mat3x1ToVector(rvec);
        result.tvec = mat3x1ToVector(tvec);
        result.T_marker_camera = transformToVector(makeTransform(rvec, tvec));
        result.reprojection_mean_px = mean_error_px;
        result.reprojection_max_px = max_error_px;
        result.num_points = static_cast<int>(points.size());
        result.num_inliers = static_cast<int>(inlier_indices.size());
        result.method = "ransac_iterative";
        return result;
    }

    if (has_pose_) {
        auto [accepted, reason] = checkMotionGate(rvec, tvec, lost_frames);
        if (!accepted) {
            MapPoseResult result;
            result.success = false;
            result.message = "Motion gate rejected pose: " + reason;
            result.rvec = mat3x1ToVector(rvec);
            result.tvec = mat3x1ToVector(tvec);
            result.T_marker_camera = transformToVector(makeTransform(rvec, tvec));
            result.reprojection_mean_px = mean_error_px;
            result.reprojection_max_px = max_error_px;
            result.num_points = static_cast<int>(points.size());
            result.num_inliers = static_cast<int>(inlier_indices.size());
            result.method = "ransac_iterative";
            return result;
        }
    }

    acceptPose(rvec, tvec);

    MapPoseResult result;
    result.success = true;
    result.message = "Pose estimation successful.";
    result.rvec = mat3x1ToVector(rvec_);
    result.tvec = mat3x1ToVector(tvec_);
    result.T_marker_camera = transformToVector(T_marker_camera_);
    result.inlier_indices = inlier_indices;
    result.reprojection_mean_px = mean_error_px;
    result.reprojection_max_px = max_error_px;
    result.num_points = static_cast<int>(points.size());
    result.num_inliers = static_cast<int>(inlier_indices.size());
    result.points.reserve(inlier_indices.size());
    for (int idx : inlier_indices) {
        result.points.push_back(points[static_cast<size_t>(idx)]);
    }
    result.method = "ransac_iterative";
    return result;
}

MapPoseResult MapPoseTracker::estimatePoseDirectPrior(
    const std::vector<PoseTrackPoint>& points,
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points,
    int lost_frames
)
{
    MapPoseResult empty;
    if (!has_pose_ || static_cast<int>(points.size()) < config_.min_inliers) {
        return empty;
    }

    cv::Mat rvec = rvec_.clone();
    cv::Mat tvec = tvec_.clone();

    try {
        const bool success = cv::solvePnP(
            object_points,
            image_points,
            K_,
            dist_coeffs_,
            rvec,
            tvec,
            true,
            cv::SOLVEPNP_ITERATIVE
        );
        if (!success) {
            return empty;
        }
    } catch (...) {
        return empty;
    }

    rvec = rvec.reshape(1, 3).clone();
    tvec = tvec.reshape(1, 3).clone();

    auto refined = refineDirectPriorPose(
        object_points,
        image_points,
        rvec,
        tvec
    );
    rvec = std::get<0>(refined);
    tvec = std::get<1>(refined);
    const std::string method = std::get<2>(refined);

    double mean_error_px = -1.0;
    double max_error_px = -1.0;
    if (!computeReprojectionStats(
            object_points,
            image_points,
            rvec,
            tvec,
            mean_error_px,
            max_error_px
        )) {
        return empty;
    }

    if (
        mean_error_px > config_.direct_max_mean_reproj_px ||
        max_error_px > config_.direct_max_max_reproj_px
    ) {
        return empty;
    }

    auto [accepted, reason] = checkMotionGate(rvec, tvec, lost_frames);
    if (!accepted) {
        return empty;
    }

    acceptPose(rvec, tvec);

    MapPoseResult result;
    result.success = true;
    result.message = "Direct prior pose estimation successful.";
    result.rvec = mat3x1ToVector(rvec_);
    result.tvec = mat3x1ToVector(tvec_);
    result.T_marker_camera = transformToVector(T_marker_camera_);
    result.reprojection_mean_px = mean_error_px;
    result.reprojection_max_px = max_error_px;
    result.num_points = static_cast<int>(points.size());
    result.num_inliers = static_cast<int>(points.size());
    result.points = points;
    result.method = method;
    result.inlier_indices.reserve(points.size());
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        result.inlier_indices.push_back(i);
    }
    return result;
}

std::tuple<cv::Mat, cv::Mat, std::string>
MapPoseTracker::refineDirectPriorPose(
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points,
    const cv::Mat& rvec,
    const cv::Mat& tvec
) const
{
    cv::Mat current_rvec = rvec.reshape(1, 3).clone();
    cv::Mat current_tvec = tvec.reshape(1, 3).clone();

    const std::string configured = lowerAscii(config_.direct_refine_method);
    if (
        configured.empty() ||
        configured == "none" ||
        configured == "off" ||
        configured == "false"
    ) {
        return {current_rvec, current_tvec, "direct_prior_unrefined"};
    }

    std::vector<std::string> methods;
    if (configured == "auto") {
        methods = {"lm", "vvs"};
    } else if (configured == "lm" || configured == "vvs") {
        methods = {configured};
    } else {
        methods = {"lm"};
    }

    for (const std::string& method : methods) {
        if (method == "lm") {
            try {
                cv::Mat rvec_ref = current_rvec.clone();
                cv::Mat tvec_ref = current_tvec.clone();
                cv::solvePnPRefineLM(
                    object_points,
                    image_points,
                    K_,
                    dist_coeffs_,
                    rvec_ref,
                    tvec_ref
                );
                return {
                    rvec_ref.reshape(1, 3).clone(),
                    tvec_ref.reshape(1, 3).clone(),
                    "direct_prior_lm"
                };
            } catch (...) {
            }
        }

        if (method == "vvs") {
            try {
                cv::Mat rvec_ref = current_rvec.clone();
                cv::Mat tvec_ref = current_tvec.clone();
                cv::solvePnPRefineVVS(
                    object_points,
                    image_points,
                    K_,
                    dist_coeffs_,
                    rvec_ref,
                    tvec_ref
                );
                return {
                    rvec_ref.reshape(1, 3).clone(),
                    tvec_ref.reshape(1, 3).clone(),
                    "direct_prior_vvs"
                };
            } catch (...) {
            }
        }
    }

    return {current_rvec, current_tvec, "direct_prior_iterative"};
}

std::pair<bool, std::string> MapPoseTracker::checkMotionGate(
    const cv::Mat& candidate_rvec,
    const cv::Mat& candidate_tvec,
    int lost_frames
) const
{
    if (!has_pose_) {
        return {true, ""};
    }

    const double effective_rotation_limit = std::min(
        config_.max_rotation_jump_deg +
            static_cast<double>(lost_frames) *
                config_.rotation_gate_scale_per_lost_frame,
        config_.rotation_gate_max_deg
    );

    cv::Mat prev_R_mat;
    cv::Mat cand_R_mat;
    cv::Rodrigues(rvec_, prev_R_mat);
    cv::Rodrigues(candidate_rvec, cand_R_mat);

    cv::Mat dR = cand_R_mat * prev_R_mat.t();
    const double trace = dR.at<double>(0, 0) +
                         dR.at<double>(1, 1) +
                         dR.at<double>(2, 2);
    const double cos_angle = std::clamp((trace - 1.0) * 0.5, -1.0, 1.0);
    const double angle_deg = std::acos(cos_angle) * 180.0 / CV_PI;

    const cv::Vec3d candidate_t(
        candidate_tvec.at<double>(0, 0),
        candidate_tvec.at<double>(1, 0),
        candidate_tvec.at<double>(2, 0)
    );
    const cv::Vec3d previous_t(
        tvec_.at<double>(0, 0),
        tvec_.at<double>(1, 0),
        tvec_.at<double>(2, 0)
    );
    const double translation_mm = cv::norm(candidate_t - previous_t);

    if (angle_deg > effective_rotation_limit) {
        return {
            false,
            "Rotation jump too large: " +
                formatDouble(angle_deg, 2) + " deg > " +
                formatDouble(effective_rotation_limit, 2) +
                " deg (lost_frames=" + std::to_string(lost_frames) + ")"
        };
    }

    if (translation_mm > config_.max_translation_jump_mm) {
        return {
            false,
            "Translation jump too large: " +
                formatDouble(translation_mm, 2) + " mm > " +
                formatDouble(config_.max_translation_jump_mm, 2) + " mm"
        };
    }

    return {true, ""};
}

bool MapPoseTracker::computeReprojectionStats(
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    double& mean_error_px,
    double& max_error_px
) const
{
    if (object_points.empty() || image_points.empty()) {
        return false;
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
        return false;
    }

    if (projected.size() != image_points.size()) {
        return false;
    }

    double sum_error = 0.0;
    max_error_px = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
        const double err = cv::norm(projected[i] - image_points[i]);
        sum_error += err;
        max_error_px = std::max(max_error_px, err);
    }

    mean_error_px = sum_error / static_cast<double>(projected.size());
    return true;
}

void MapPoseTracker::acceptPose(const cv::Mat& rvec, const cv::Mat& tvec)
{
    rvec_ = rvec.reshape(1, 3).clone();
    tvec_ = tvec.reshape(1, 3).clone();
    T_marker_camera_ = makeTransform(rvec_, tvec_);
    has_pose_ = true;
}

bool MapPoseTracker::hasPose() const
{
    return has_pose_;
}

std::vector<double> MapPoseTracker::rvec() const
{
    return has_pose_ ? mat3x1ToVector(rvec_) : std::vector<double>();
}

std::vector<double> MapPoseTracker::tvec() const
{
    return has_pose_ ? mat3x1ToVector(tvec_) : std::vector<double>();
}

std::vector<double> MapPoseTracker::TMarkerCamera() const
{
    return has_pose_ ? transformToVector(T_marker_camera_) : std::vector<double>();
}

const MapPoseTrackerConfig& MapPoseTracker::config() const
{
    return config_;
}

cv::Matx44d MapPoseTracker::makeTransform(
    const cv::Mat& rvec,
    const cv::Mat& tvec
)
{
    cv::Mat R_mat;
    cv::Rodrigues(rvec, R_mat);

    cv::Matx44d T = cv::Matx44d::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            T(r, c) = R_mat.at<double>(r, c);
        }
    }
    T(0, 3) = tvec.at<double>(0, 0);
    T(1, 3) = tvec.at<double>(1, 0);
    T(2, 3) = tvec.at<double>(2, 0);
    return T;
}

std::vector<double> MapPoseTracker::mat3x1ToVector(const cv::Mat& mat)
{
    cv::Mat reshaped = mat.reshape(1, 3);
    return {
        reshaped.at<double>(0, 0),
        reshaped.at<double>(1, 0),
        reshaped.at<double>(2, 0)
    };
}

std::vector<double> MapPoseTracker::transformToVector(const cv::Matx44d& T)
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

std::string MapPoseTracker::formatDouble(double value, int precision)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

} // namespace hydramarker
