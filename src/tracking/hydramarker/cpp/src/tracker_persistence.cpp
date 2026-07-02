#include "tracker_persistence.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>

#include <opencv2/calib3d.hpp>

#include "tracker_geometry.hpp"

namespace hydramarker {

namespace {

using GlobalKey = std::pair<int, int>;

double pointDistance(const cv::Point2d& a, const cv::Point2d& b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

double medianInPlace(std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }

    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    const double upper = values[mid];

    if ((values.size() % 2) != 0) {
        return upper;
    }

    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    return 0.5 * (values[mid - 1] + upper);
}

double elapsedMs(std::int64_t start_tick)
{
    return (
        static_cast<double>(cv::getTickCount() - start_tick) /
        cv::getTickFrequency()
    ) * 1000.0;
}

std::string formatDouble(double value, int precision)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
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

FastDenseProjectionStats copyDenseStats(
    const DenseProjectionMatchStats& stats
)
{
    FastDenseProjectionStats out;
    out.detected = stats.detected;
    out.projected = stats.projected;
    out.rejected_no_projection = stats.rejected_no_projection;
    out.rejected_far = stats.rejected_far;
    out.rejected_ambiguous = stats.rejected_ambiguous;
    out.rejected_non_mutual = stats.rejected_non_mutual;
    out.median_error_px = stats.median_error_px;
    out.p90_error_px = stats.p90_error_px;
    out.image_coverage = stats.image_coverage;
    out.image_span_u_px = stats.image_span_u_px;
    out.image_span_v_px = stats.image_span_v_px;
    out.object_span_mm = stats.object_span_mm;
    out.distinct_rows = stats.distinct_rows;
    out.distinct_cols = stats.distinct_cols;
    return out;
}

std::vector<PoseTrackPoint> pointsFromTrackerCorners(
    const std::vector<TrackerCorner>& corners
)
{
    std::vector<PoseTrackPoint> points;
    points.reserve(corners.size());
    std::set<GlobalKey> used_globals;

    for (const TrackerCorner& corner : corners) {
        const GlobalKey key{corner.global_row, corner.global_col};
        if (used_globals.find(key) != used_globals.end()) {
            continue;
        }
        used_globals.insert(key);

        PoseTrackPoint point;
        point.global_row = corner.global_row;
        point.global_col = corner.global_col;
        point.xyz_mm = corner.xyz_mm;
        point.uv = corner.uv;
        point.votes = corner.votes;
        points.push_back(point);
    }

    return points;
}

std::vector<GlobalCornerIdentity> identitiesFromTrackerCorners(
    const std::vector<TrackerCorner>& corners
)
{
    std::vector<GlobalCornerIdentity> identities;
    identities.reserve(corners.size());
    std::set<GlobalKey> used_globals;

    for (const TrackerCorner& corner : corners) {
        const GlobalKey key{corner.global_row, corner.global_col};
        if (used_globals.find(key) != used_globals.end()) {
            continue;
        }
        used_globals.insert(key);

        GlobalCornerIdentity identity;
        identity.global_row = corner.global_row;
        identity.global_col = corner.global_col;
        identity.xyz_mm = corner.xyz_mm;
        identity.uv = corner.uv;
        identity.votes = corner.votes;
        identities.push_back(identity);
    }

    return identities;
}

void pointsToCv(
    const std::vector<PoseTrackPoint>& points,
    std::vector<cv::Point3d>& object_points,
    std::vector<cv::Point2d>& image_points
)
{
    object_points.clear();
    image_points.clear();
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
}

bool vectorToMat3x1Local(const std::vector<double>& values, cv::Mat& mat)
{
    if (values.size() != 3) {
        mat.release();
        return false;
    }

    mat = cv::Mat(3, 1, CV_64F);
    mat.at<double>(0, 0) = values[0];
    mat.at<double>(1, 0) = values[1];
    mat.at<double>(2, 0) = values[2];
    return true;
}

bool reprojectionMeanMax(
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points,
    const cv::Matx33d& K,
    const std::vector<double>& dist_coeffs,
    const std::vector<double>& rvec,
    const std::vector<double>& tvec,
    double& mean_error_px,
    double& max_error_px
)
{
    mean_error_px = -1.0;
    max_error_px = -1.0;
    if (object_points.empty() || object_points.size() != image_points.size()) {
        return false;
    }

    cv::Mat rvec_mat;
    cv::Mat tvec_mat;
    if (!vectorToMat3x1Local(rvec, rvec_mat) ||
        !vectorToMat3x1Local(tvec, tvec_mat)) {
        return false;
    }

    cv::Mat dist_mat;
    if (!dist_coeffs.empty()) {
        dist_mat = cv::Mat(static_cast<int>(dist_coeffs.size()), 1, CV_64F);
        for (int i = 0; i < static_cast<int>(dist_coeffs.size()); ++i) {
            dist_mat.at<double>(i, 0) = dist_coeffs[static_cast<size_t>(i)];
        }
    }

    std::vector<cv::Point2d> projected;
    try {
        cv::projectPoints(
            object_points,
            rvec_mat,
            tvec_mat,
            K,
            dist_mat,
            projected
        );
    } catch (...) {
        return false;
    }

    if (projected.size() != image_points.size()) {
        return false;
    }

    double sum = 0.0;
    double max_value = 0.0;
    for (size_t idx = 0; idx < projected.size(); ++idx) {
        const double err = pointDistance(projected[idx], image_points[idx]);
        sum += err;
        max_value = std::max(max_value, err);
    }

    mean_error_px = sum / static_cast<double>(projected.size());
    max_error_px = max_value;
    return true;
}

std::vector<TrackerCorner> inlierCornersFromPose(
    const MapPoseResult& pose,
    const std::vector<TrackerCorner>& corners
)
{
    std::vector<TrackerCorner> inlier_corners;
    inlier_corners.reserve(pose.inlier_indices.size());

    for (int idx : pose.inlier_indices) {
        if (idx < 0 || idx >= static_cast<int>(corners.size())) {
            continue;
        }
        inlier_corners.push_back(corners[static_cast<size_t>(idx)]);
    }

    return inlier_corners;
}

bool poseMotionPlausible(
    const TrackerConfig& config,
    const std::vector<double>& rvec,
    const std::vector<double>& tvec,
    const std::vector<double>& previous_rvec,
    const std::vector<double>& previous_tvec
)
{
    if (rvec.size() != 3 || tvec.size() != 3) {
        return false;
    }
    if (previous_rvec.size() != 3 || previous_tvec.size() != 3) {
        return true;
    }

    try {
        cv::Mat rvec_mat(3, 1, CV_64F);
        cv::Mat tvec_mat(3, 1, CV_64F);
        cv::Mat prev_rvec_mat(3, 1, CV_64F);
        cv::Mat prev_tvec_mat(3, 1, CV_64F);
        for (int i = 0; i < 3; ++i) {
            rvec_mat.at<double>(i, 0) = rvec[static_cast<size_t>(i)];
            tvec_mat.at<double>(i, 0) = tvec[static_cast<size_t>(i)];
            prev_rvec_mat.at<double>(i, 0) =
                previous_rvec[static_cast<size_t>(i)];
            prev_tvec_mat.at<double>(i, 0) =
                previous_tvec[static_cast<size_t>(i)];
        }

        cv::Mat R;
        cv::Mat prev_R;
        cv::Rodrigues(rvec_mat, R);
        cv::Rodrigues(prev_rvec_mat, prev_R);
        const cv::Mat dR = R * prev_R.t();
        double cos_angle = (cv::trace(dR)[0] - 1.0) * 0.5;
        cos_angle = std::clamp(cos_angle, -1.0, 1.0);
        const double rot_delta_deg =
            std::acos(cos_angle) * 180.0 / CV_PI;
        const double trans_delta_mm = cv::norm(tvec_mat - prev_tvec_mat);

        return (
            rot_delta_deg <= config.persistence_max_rotation_jump_deg &&
            trans_delta_mm <= config.persistence_max_translation_jump_mm
        );
    } catch (...) {
        return false;
    }
}

std::string fallbackPoseRejectionReason(
    const TrackerConfig& config,
    const TrackerGeometry& geometry,
    const CheckerboardDetection& detection,
    const MapPoseResult& pose
)
{
    if (pose.reprojection_mean_px >
        config.fallback_pose_max_mean_reprojection_error_px) {
        return (
            "Fallback pose rejected by mean reprojection gate (" +
            formatDouble(pose.reprojection_mean_px, 2) +
            "px)."
        );
    }

    if (pose.reprojection_max_px >
        config.fallback_pose_max_max_reprojection_error_px) {
        return (
            "Fallback pose rejected by max reprojection gate (" +
            formatDouble(pose.reprojection_max_px, 2) +
            "px)."
        );
    }

    DenseProjectionMatchResult visual = geometry.greedyProjectedMatch(
        detection,
        pose.rvec,
        pose.tvec,
        config.fallback_pose_max_p90_corner_error_px
    );

    const int match_count = static_cast<int>(visual.corners.size());
    if (match_count < config.fallback_pose_min_detection_matches) {
        return (
            "Fallback pose rejected by blue-corner alignment (" +
            std::to_string(match_count) +
            " matches)."
        );
    }

    if (visual.stats.median_error_px >
        config.fallback_pose_max_median_corner_error_px) {
        return (
            "Fallback pose rejected by median blue-corner error (" +
            formatDouble(visual.stats.median_error_px, 2) +
            "px)."
        );
    }

    if (visual.stats.p90_error_px >
        config.fallback_pose_max_p90_corner_error_px) {
        return (
            "Fallback pose rejected by p90 blue-corner error (" +
            formatDouble(visual.stats.p90_error_px, 2) +
            "px)."
        );
    }

    return "";
}

std::pair<bool, std::string> fastDenseRefineRequiredForSeed(
    const TrackerConfig& config,
    const MapPoseResult& seed_pose,
    int match_count,
    const PersistentMatchStats& stats,
    FastDenseGateMetrics& metrics
)
{
    const int current_corners = std::max(0, stats.current_corners);
    metrics.match_ratio =
        static_cast<double>(match_count) /
        static_cast<double>(std::max(current_corners, 1));
    metrics.motion_px = stats.adaptive_motion_px;
    metrics.ambiguous_count =
        static_cast<double>(std::max(0, stats.rejected_ambiguous));
    metrics.seed_mean_px = seed_pose.reprojection_mean_px;
    metrics.seed_max_px = seed_pose.reprojection_max_px;

    if (!config.fast_persistent_dense_adaptive_refine_enabled) {
        return {true, "adaptive_disabled"};
    }

    std::vector<std::string> reasons;
    if (
        config.fast_persistent_dense_adaptive_min_match_ratio > 0.0 &&
        metrics.match_ratio <
            config.fast_persistent_dense_adaptive_min_match_ratio
    ) {
        reasons.push_back("low_match_ratio");
    }

    if (
        config.fast_persistent_dense_adaptive_motion_px > 0.0 &&
        metrics.motion_px >= config.fast_persistent_dense_adaptive_motion_px
    ) {
        reasons.push_back("motion");
    }

    if (stats.rejected_ambiguous > 0) {
        reasons.push_back("ambiguous");
    }

    if (
        config.fast_persistent_dense_adaptive_max_seed_mean_px > 0.0 &&
        std::isfinite(metrics.seed_mean_px) &&
        metrics.seed_mean_px >
            config.fast_persistent_dense_adaptive_max_seed_mean_px
    ) {
        reasons.push_back("seed_mean");
    }

    if (
        config.fast_persistent_dense_adaptive_max_seed_max_px > 0.0 &&
        std::isfinite(metrics.seed_max_px) &&
        metrics.seed_max_px >
            config.fast_persistent_dense_adaptive_max_seed_max_px
    ) {
        reasons.push_back("seed_max");
    }

    if (reasons.empty()) {
        return {false, "clean_seed"};
    }

    std::string joined = reasons.front();
    for (size_t i = 1; i < reasons.size(); ++i) {
        joined += "+" + reasons[i];
    }
    return {true, joined};
}

} // namespace

PersistentMatcher::PersistentMatcher(const TrackerConfig& config)
    : config_(config)
{
}

void PersistentMatcher::reset()
{
    clearIdentities();
    prev_detection_uvs_.clear();
    prev_detection_frame_ = -1;
    last_motion_px_ = 0.0;
}

void PersistentMatcher::clearIdentities()
{
    identities_.clear();
    persistent_frame_index_ = -1;
}

void PersistentMatcher::replaceIdentities(
    const std::vector<GlobalCornerIdentity>& identities,
    int frame_index
)
{
    identities_.clear();

    std::map<GlobalKey, size_t> index_by_global;
    for (const GlobalCornerIdentity& identity : identities) {
        const GlobalKey key{
            static_cast<int>(identity.global_row),
            static_cast<int>(identity.global_col)
        };

        const auto found = index_by_global.find(key);
        if (found == index_by_global.end()) {
            index_by_global.emplace(key, identities_.size());
            identities_.push_back(identity);
            continue;
        }

        GlobalCornerIdentity& existing = identities_[found->second];
        if (static_cast<int>(identity.votes) >= static_cast<int>(existing.votes)) {
            existing = identity;
        }
    }

    persistent_frame_index_ = identities_.empty() ? -1 : frame_index;
}

const std::vector<GlobalCornerIdentity>& PersistentMatcher::identities() const
{
    return identities_;
}

int PersistentMatcher::persistentFrameIndex() const
{
    return persistent_frame_index_;
}

const TrackerConfig& PersistentMatcher::config() const
{
    return config_;
}

PersistentMatchResult PersistentMatcher::match(
    const CheckerboardDetection& detection,
    int frame_index,
    const cv::Matx33d& K,
    const std::vector<double>& dist_coeffs,
    const std::vector<double>& rvec,
    const std::vector<double>& tvec,
    double last_good_reproj_px
)
{
    PersistentMatchResult result;
    PersistentMatchStats& stats = result.stats;
    stats.identities = static_cast<int>(identities_.size());
    stats.adaptive_max_dist_px = config_.persistence_uv_match_dist_px;

    if (identities_.empty()) {
        result.message = "No persistent identities.";
        return result;
    }

    if (persistent_frame_index_ < 0) {
        result.message = "No persistent frame.";
        return result;
    }

    const int age = frame_index - persistent_frame_index_;
    stats.age = age;
    if (age < 0 || age > config_.persistence_max_frames) {
        result.message = "Persistent identities too old.";
        return result;
    }

    std::vector<DetectedCornerCpp> current_corners =
        detectedCornersFromDetection(detection);
    stats.current_corners = static_cast<int>(current_corners.size());
    if (current_corners.empty()) {
        result.message = "No current detection corners.";
        return result;
    }

    std::vector<cv::Point2d> current_uvs;
    current_uvs.reserve(current_corners.size());
    for (const DetectedCornerCpp& corner : current_corners) {
        current_uvs.push_back(corner.uv);
    }

    const double motion_px = detectionMotionPx(current_uvs, frame_index);
    stats.adaptive_motion_px = motion_px;

    cv::Mat rvec_mat;
    cv::Mat tvec_mat;
    const bool has_pose =
        vectorToMat3x1(rvec, rvec_mat) && vectorToMat3x1(tvec, tvec_mat);

    const bool use_pose_projection =
        config_.persistence_use_pose_projection &&
        has_pose &&
        last_good_reproj_px >= 0.0 &&
        last_good_reproj_px <=
            config_.persistence_projection_max_pose_error_px;

    stats.used_pose_projection = use_pose_projection;

    double projection_max_dist = config_.persistence_projection_max_reproj_px;
    if (use_pose_projection) {
        projection_max_dist =
            adaptiveProjectionMatchRadiusPx(projection_max_dist, motion_px);
        stats.adaptive_max_dist_px = projection_max_dist;
    }

    cv::Mat dist_coeffs_mat = makeDistCoeffsMat(dist_coeffs);

    result.points.reserve(identities_.size());
    result.corners.reserve(identities_.size());

    std::set<GlobalKey> used_globals;
    std::vector<bool> used_current_indices(current_corners.size(), false);

    for (const GlobalCornerIdentity& cached : identities_) {
        const GlobalKey global_key{
            static_cast<int>(cached.global_row),
            static_cast<int>(cached.global_col)
        };
        if (used_globals.find(global_key) != used_globals.end()) {
            continue;
        }

        cv::Point2d predicted_uv(cached.uv[0], cached.uv[1]);
        double max_dist = config_.persistence_uv_match_dist_px;

        if (use_pose_projection) {
            if (!projectPoint(
                    cached.xyz_mm,
                    K,
                    dist_coeffs_mat,
                    rvec_mat,
                    tvec_mat,
                    predicted_uv
                )) {
                ++stats.rejected_no_projection;
                continue;
            }
            max_dist = projection_max_dist;
        }

        const CornerMatch match = matchPredictedUvToDetectionCorner(
            predicted_uv,
            current_corners,
            used_current_indices,
            max_dist
        );

        if (match.reject_reason == "far") {
            ++stats.rejected_far;
            continue;
        }
        if (match.reject_reason == "ambiguous") {
            ++stats.rejected_ambiguous;
            continue;
        }
        if (match.reject_reason == "claimed") {
            ++stats.rejected_claimed;
            continue;
        }
        if (match.index < 0) {
            continue;
        }

        const DetectedCornerCpp& matched =
            current_corners[static_cast<size_t>(match.index)];
        const std::array<double, 2> uv = {matched.uv.x, matched.uv.y};
        const int votes = std::max(0, static_cast<int>(cached.votes) - age);

        PoseTrackPoint point;
        point.global_row = global_key.first;
        point.global_col = global_key.second;
        point.xyz_mm = cached.xyz_mm;
        point.uv = uv;
        point.votes = votes;
        result.points.push_back(point);

        TrackerCorner corner;
        corner.local_row = matched.local_row;
        corner.local_col = matched.local_col;
        corner.global_row = global_key.first;
        corner.global_col = global_key.second;
        corner.xyz_mm = cached.xyz_mm;
        corner.uv = uv;
        corner.votes = votes;
        corner.visibility_score = matched.visibility_score;
        corner.observed_frames = matched.observed_frames;
        corner.predicted = matched.predicted;
        result.corners.push_back(corner);

        used_globals.insert(global_key);
        used_current_indices[static_cast<size_t>(match.index)] = true;
        ++stats.accepted;
    }

    result.message = result.valid()
        ? "Persistent identities matched."
        : "No persistent identities matched.";
    return result;
}

PersistentPoseSeedResult PersistentMatcher::estimatePose(
    const CheckerboardDetection& detection,
    int frame_index,
    const cv::Matx33d& K,
    const std::vector<double>& dist_coeffs,
    const std::vector<double>& rvec,
    const std::vector<double>& tvec,
    double last_good_reproj_px,
    int lost_frames
)
{
    const std::int64_t total_t0 = cv::getTickCount();
    PersistentPoseSeedResult result;

    const std::int64_t match_t0 = cv::getTickCount();
    PersistentMatchResult match_result = match(
        detection,
        frame_index,
        K,
        dist_coeffs,
        rvec,
        tvec,
        last_good_reproj_px
    );
    result.match_ms = (
        static_cast<double>(cv::getTickCount() - match_t0) /
        cv::getTickFrequency()
    ) * 1000.0;

    result.points = std::move(match_result.points);
    result.corners = std::move(match_result.corners);
    result.stats = match_result.stats;

    const int min_points = std::max({
        config_.min_points,
        config_.persistence_min_points,
        config_.fast_persistent_min_points
    });
    if (static_cast<int>(result.points.size()) < min_points) {
        result.pose.success = false;
        result.pose.message =
            "Too few matches: " + std::to_string(result.points.size()) +
            " < " + std::to_string(min_points);
        result.pose.num_points = static_cast<int>(result.points.size());
        result.message = result.pose.message;
        result.total_ms = (
            static_cast<double>(cv::getTickCount() - total_t0) /
            cv::getTickFrequency()
        ) * 1000.0;
        return result;
    }

    MapPoseTracker pose_tracker(
        K,
        dist_coeffs,
        makeMapPoseTrackerConfig(config_)
    );
    pose_tracker.setPose(rvec, tvec);

    const std::int64_t pose_t0 = cv::getTickCount();
    result.pose = pose_tracker.estimatePose(result.points, lost_frames);
    result.pose_ms = (
        static_cast<double>(cv::getTickCount() - pose_t0) /
        cv::getTickFrequency()
    ) * 1000.0;

    result.message = result.pose.message;
    result.total_ms = (
        static_cast<double>(cv::getTickCount() - total_t0) /
        cv::getTickFrequency()
    ) * 1000.0;
    return result;
}

FastPoseResult PersistentMatcher::estimateFastPose(
    const CheckerboardDetection& detection,
    const MarkerGeometry& geometry,
    int frame_index,
    const cv::Matx33d& K,
    const std::vector<double>& dist_coeffs,
    const std::vector<double>& rvec,
    const std::vector<double>& tvec,
    double last_good_reproj_px,
    const std::vector<double>& previous_rvec,
    const std::vector<double>& previous_tvec,
    int lost_frames,
    int max_pts_seen
)
{
    const std::int64_t total_t0 = cv::getTickCount();

    FastPoseResult result;
    result.attempted = true;

    TrackerGeometry geometry_helper(geometry, K, dist_coeffs, config_);

    PersistentPoseSeedResult seed = estimatePose(
        detection,
        frame_index,
        K,
        dist_coeffs,
        rvec,
        tvec,
        last_good_reproj_px,
        lost_frames
    );

    result.points = seed.points;
    result.corners = seed.corners;
    result.stats = seed.stats;
    result.seed_pose = seed.pose;
    result.persistent_match_ms = seed.match_ms;
    result.seed_pnp_ms = seed.pose_ms;
    result.cpp_seed_total_ms = seed.total_ms;

    result.min_points = std::max({
        config_.min_points,
        config_.persistence_min_points,
        config_.fast_persistent_min_points
    });
    if (static_cast<int>(result.points.size()) < result.min_points) {
        result.success = false;
        result.reason =
            "too_few_matches:" +
            std::to_string(result.points.size()) +
            "<" + std::to_string(result.min_points);
        result.total_ms = elapsedMs(total_t0);
        return result;
    }

    if (!result.seed_pose.success) {
        result.success = false;
        result.reason = result.seed_pose.message;
        result.total_ms = elapsedMs(total_t0);
        return result;
    }

    if (
        result.seed_pose.rvec.size() != 3 ||
        result.seed_pose.tvec.size() != 3 ||
        result.seed_pose.T_marker_camera.size() != 16
    ) {
        result.success = false;
        result.reason = "C++ seed pose missing pose vectors.";
        result.total_ms = elapsedMs(total_t0);
        return result;
    }

    if (!poseMotionPlausible(
            config_,
            result.seed_pose.rvec,
            result.seed_pose.tvec,
            previous_rvec,
            previous_tvec
        )) {
        result.success = false;
        result.reason = "Persistent pose rejected by motion gate.";
        result.total_ms = elapsedMs(total_t0);
        return result;
    }

    const std::string seed_reject_reason = fallbackPoseRejectionReason(
        config_,
        geometry_helper,
        detection,
        result.seed_pose
    );
    if (!seed_reject_reason.empty()) {
        result.success = false;
        result.reason = seed_reject_reason;
        result.total_ms = elapsedMs(total_t0);
        return result;
    }

    const auto dense_decision = fastDenseRefineRequiredForSeed(
        config_,
        result.seed_pose,
        static_cast<int>(result.points.size()),
        result.stats,
        result.dense_gate_metrics
    );
    result.dense_required = dense_decision.first;
    result.dense_gate_reason = dense_decision.second;

    auto finalPoseForPackaging = [&]() -> const MapPoseResult& {
        return result.pose;
    };

    auto fillVisualCorners = [&]() {
        const MapPoseResult& final_pose = finalPoseForPackaging();
        const std::vector<TrackerCorner> inlier_corners =
            inlierCornersFromPose(final_pose, result.corners);
        result.visual_corners = geometry_helper.visualCornersFromPose(
            inlier_corners,
            final_pose.rvec,
            final_pose.tvec,
            config_.visual_corner_max_reprojection_error_px
        );
    };

    auto fillAcceptedState = [&]() {
        const MapPoseResult& final_pose = finalPoseForPackaging();
        FastAcceptedState state;
        state.evaluated = true;
        state.visual_corner_count =
            static_cast<int>(result.visual_corners.size());
        state.reliable_pose =
            state.visual_corner_count >= config_.visual_corner_min_count;
        state.max_pts_seen = std::max(0, max_pts_seen);
        state.last_good_reproj_px = last_good_reproj_px;
        state.accepted_pose_frame = -1;

        if (state.reliable_pose) {
            state.max_pts_seen =
                std::max(state.max_pts_seen, final_pose.num_inliers);
            if (final_pose.reprojection_mean_px >= 0.0) {
                state.last_good_reproj_px = final_pose.reprojection_mean_px;
            }
            state.rvec = final_pose.rvec;
            state.tvec = final_pose.tvec;
            state.T_marker_camera = final_pose.T_marker_camera;
            state.accepted_pose_frame = frame_index;
        }

        result.accepted_state = std::move(state);
    };

    auto fillPersistenceRefresh = [&]() {
        const std::int64_t refresh_t0 = cv::getTickCount();
        result.persistence_refresh_available = false;
        result.persistence_refresh_frame = -1;
        result.persistence_refresh_count = 0;
        result.persistence_refresh_identities.clear();
        result.persistence_refresh_ms = 0.0;

        const MapPoseResult& final_pose = finalPoseForPackaging();
        if (
            config_.decode_only_mode ||
            !config_.enable_temporal_correspondence_persistence ||
            !result.accepted_state.reliable_pose ||
            final_pose.reprojection_mean_px < 0.0 ||
            final_pose.reprojection_mean_px >
                config_.fast_persistent_refresh_mean_error_px
        ) {
            result.persistence_refresh_ms = elapsedMs(refresh_t0);
            return;
        }

        std::vector<GlobalCornerIdentity> identities =
            identitiesFromTrackerCorners(result.visual_corners);
        if (static_cast<int>(identities.size()) < config_.persistence_min_points) {
            result.persistence_refresh_ms = elapsedMs(refresh_t0);
            return;
        }

        result.persistence_refresh_available = true;
        result.persistence_refresh_frame = frame_index;
        result.persistence_refresh_count = static_cast<int>(identities.size());
        result.persistence_refresh_identities = std::move(identities);
        result.persistence_refresh_ms = elapsedMs(refresh_t0);
    };

    auto acceptSeed = [&]() {
        result.success = true;
        result.used_dense = false;
        result.reason = "ok";
        result.pose = result.seed_pose;
        fillVisualCorners();
        fillAcceptedState();
        fillPersistenceRefresh();
        result.total_ms = elapsedMs(total_t0);
    };

    auto failDenseOrAcceptSeed = [&](const std::string& dense_reason) {
        result.dense_success = false;
        result.dense_reason = dense_reason;

        const double sparse_ratio =
            static_cast<double>(result.points.size()) /
            static_cast<double>(std::max(result.stats.current_corners, 1));
        const bool dense_validation_failed =
            result.dense_attempted &&
            !result.dense_success &&
            result.stats.current_corners >= result.min_dense_points &&
            sparse_ratio <
                config_.fast_persistent_dense_rescue_min_green_ratio &&
            dense_reason != "disabled" &&
            dense_reason != "missing_seed_pose";

        if (
            dense_reason.rfind("rescue_failed:", 0) == 0 ||
            dense_reason.rfind("rescue_skipped_decode:", 0) == 0 ||
            dense_validation_failed
        ) {
            result.success = false;
            result.route_decode = true;
            result.reason =
                "dense_validation_rejected_seed:" +
                dense_reason +
                "; sparse_ratio=" + formatDouble(sparse_ratio, 3);
            result.total_ms = elapsedMs(total_t0);
            return;
        }

        acceptSeed();
    };

    if (!result.dense_required) {
        result.dense_attempted = false;
        result.dense_reason = "adaptive_skip:" + result.dense_gate_reason;
        acceptSeed();
        return result;
    }

    if (!config_.fast_persistent_dense_refine_enabled) {
        result.dense_attempted = false;
        result.dense_reason = "disabled";
        acceptSeed();
        return result;
    }

    result.dense_attempted = true;
    result.min_dense_points = std::max(
        config_.fast_persistent_dense_min_points,
        result.seed_pose.num_inliers + 1
    );

    const int detected_corner_count =
        static_cast<int>(detection.corners.size());
    result.dense_stats.detected = detected_corner_count;
    if (detected_corner_count < result.min_dense_points) {
        failDenseOrAcceptSeed(
            "too_few_detection_corners:" +
            std::to_string(detected_corner_count) +
            "<" + std::to_string(result.min_dense_points)
        );
        return result;
    }

    const std::int64_t dense_match_t0 = cv::getTickCount();
    DenseProjectionMatchResult dense_match =
        geometry_helper.strictProjectedMatch(
            detection,
            result.seed_pose.rvec,
            result.seed_pose.tvec,
            config_.fast_persistent_dense_match_max_px,
            config_.fast_persistent_dense_min_second_best_margin_px
        );
    result.dense_match_ms = elapsedMs(dense_match_t0);
    result.dense_stats = copyDenseStats(dense_match.stats);
    result.dense_matches = static_cast<int>(dense_match.corners.size());

    const double median_err = result.dense_stats.median_error_px;
    const double p90_err = result.dense_stats.p90_error_px;

    if (result.dense_matches < result.min_dense_points) {
        failDenseOrAcceptSeed(
            "too_few_matches:" +
            std::to_string(result.dense_matches) +
            "<" + std::to_string(result.min_dense_points)
        );
        return result;
    }

    if (
        result.dense_stats.distinct_rows <
            config_.fast_persistent_dense_min_distinct_rows ||
        result.dense_stats.distinct_cols <
            config_.fast_persistent_dense_min_distinct_cols
    ) {
        failDenseOrAcceptSeed(
            "poor_grid_spread:" +
            std::to_string(result.dense_stats.distinct_rows) +
            "x" + std::to_string(result.dense_stats.distinct_cols)
        );
        return result;
    }

    if (
        result.dense_stats.object_span_mm <
        config_.fast_persistent_dense_min_object_span_mm
    ) {
        failDenseOrAcceptSeed(
            "poor_object_span:" +
            formatDouble(result.dense_stats.object_span_mm, 1) +
            "mm"
        );
        return result;
    }

    const double min_coverage =
        config_.fast_persistent_dense_min_image_coverage;
    if (
        result.dense_stats.image_coverage >= 0.0 &&
        result.dense_stats.image_coverage < min_coverage
    ) {
        failDenseOrAcceptSeed(
            "poor_image_coverage:" +
            formatDouble(result.dense_stats.image_coverage, 3)
        );
        return result;
    }

    std::vector<PoseTrackPoint> dense_points =
        pointsFromTrackerCorners(dense_match.corners);
    if (static_cast<int>(dense_points.size()) < result.min_dense_points) {
        failDenseOrAcceptSeed(
            "too_few_unique_points:" +
            std::to_string(dense_points.size()) +
            "<" + std::to_string(result.min_dense_points)
        );
        return result;
    }

    if (median_err > config_.fast_persistent_dense_max_median_px) {
        result.seed_error_reason =
            "median_error:" + formatDouble(median_err, 3);
    } else if (p90_err > config_.fast_persistent_dense_max_p90_px) {
        result.seed_error_reason =
            "p90_error:" + formatDouble(p90_err, 3);
    }

    const double seed_green_ratio =
        static_cast<double>(result.seed_pose.num_inliers) /
        static_cast<double>(std::max(result.dense_stats.detected, 1));
    result.rescue_required =
        !result.seed_error_reason.empty() &&
        (
            result.seed_error_reason.rfind("p90_error", 0) == 0 ||
            seed_green_ratio <
                config_.fast_persistent_dense_rescue_min_green_ratio ||
            median_err >=
                config_.fast_persistent_dense_rescue_min_seed_median_px
        );

    if (
        result.rescue_required &&
        !config_.fast_persistent_dense_rescue_enabled
    ) {
        failDenseOrAcceptSeed(
            "rescue_skipped_decode:" + result.seed_error_reason
        );
        return result;
    }

    if (!result.seed_error_reason.empty() && !result.rescue_required) {
        failDenseOrAcceptSeed(result.seed_error_reason);
        return result;
    }

    const std::string dense_solver = lowerAscii(
        config_.fast_persistent_dense_pose_solver
    );
    const bool use_robust_solver =
        result.rescue_required ||
        dense_solver == "sqpnp" ||
        dense_solver == "robust_sqpnp" ||
        dense_solver == "sqpnp_trim" ||
        dense_solver == "robust";

    const std::int64_t dense_pose_t0 = cv::getTickCount();
    MapPoseResult dense_pose;
    if (use_robust_solver) {
        dense_pose = geometry_helper.estimateDenseRobustPose(
            dense_points,
            detection,
            result.seed_pose.rvec,
            result.seed_pose.tvec,
            previous_rvec,
            previous_tvec
        );
    } else {
        dense_pose = geometry_helper.estimateDenseDirectPose(
            dense_points,
            result.seed_pose.rvec,
            result.seed_pose.tvec,
            lost_frames
        );
    }
    result.dense_pose_ms = elapsedMs(dense_pose_t0);

    if (
        !dense_pose.success ||
        dense_pose.num_inliers < result.min_dense_points
    ) {
        std::string reject_reason;
        if (!dense_pose.success) {
            reject_reason = dense_pose.message;
        } else {
            reject_reason =
                "too_few_inliers:" +
                std::to_string(dense_pose.num_inliers) +
                "<" + std::to_string(result.min_dense_points);
        }
        if (!result.seed_error_reason.empty()) {
            reject_reason =
                "rescue_failed:" +
                result.seed_error_reason +
                "; " + reject_reason;
        }
        failDenseOrAcceptSeed(reject_reason);
        return result;
    }

    result.success = true;
    result.used_dense = true;
    result.dense_success = true;
    result.dense_reason = result.seed_error_reason.empty()
        ? "ok"
        : "rescue_ok:" + result.seed_error_reason;
    result.reason = "ok";
    result.pose = dense_pose;
    result.points = std::move(dense_points);
    result.corners = std::move(dense_match.corners);
    fillVisualCorners();
    fillAcceptedState();
    fillPersistenceRefresh();
    result.total_ms = elapsedMs(total_t0);
    return result;
}

double PersistentMatcher::detectionMotionPx(
    const std::vector<cv::Point2d>& current_uvs,
    int frame_index
)
{
    if (prev_detection_frame_ == frame_index) {
        return last_motion_px_;
    }

    double motion_px = 0.0;
    if (!prev_detection_uvs_.empty() && !current_uvs.empty()) {
        std::vector<double> nearest_distances;
        nearest_distances.reserve(current_uvs.size());

        for (const cv::Point2d& current : current_uvs) {
            double best = std::numeric_limits<double>::infinity();
            for (const cv::Point2d& previous : prev_detection_uvs_) {
                best = std::min(best, pointDistance(current, previous));
            }
            if (std::isfinite(best)) {
                nearest_distances.push_back(best);
            }
        }

        if (!nearest_distances.empty()) {
            motion_px = medianInPlace(nearest_distances);
        }
    }

    prev_detection_uvs_ = current_uvs;
    prev_detection_frame_ = frame_index;
    last_motion_px_ = motion_px;
    return motion_px;
}

double PersistentMatcher::adaptiveProjectionMatchRadiusPx(
    double base_radius_px,
    double motion_px
) const
{
    if (!config_.persistence_projection_adaptive_match_enabled) {
        return base_radius_px;
    }

    const double start_px = config_.persistence_projection_adaptive_motion_start_px;
    const double scale = config_.persistence_projection_adaptive_motion_scale;
    const double max_radius_px = std::max(
        base_radius_px,
        config_.persistence_projection_adaptive_max_reproj_px
    );
    const double extra_px =
        std::max(0.0, motion_px - std::max(0.0, start_px)) *
        std::max(0.0, scale);
    return std::min(
        max_radius_px,
        std::max(base_radius_px, base_radius_px + extra_px)
    );
}

PersistentMatcher::CornerMatch
PersistentMatcher::matchPredictedUvToDetectionCorner(
    const cv::Point2d& predicted_uv,
    const std::vector<DetectedCornerCpp>& current_corners,
    const std::vector<bool>& used_current_indices,
    double max_dist_px
) const
{
    CornerMatch match;
    if (current_corners.empty()) {
        match.reject_reason = "far";
        match.best_dist = std::numeric_limits<double>::infinity();
        match.second_dist = std::numeric_limits<double>::infinity();
        return match;
    }

    int best_idx = -1;
    double best_dist_sq = std::numeric_limits<double>::infinity();
    double second_dist_sq = std::numeric_limits<double>::infinity();

    for (int idx = 0; idx < static_cast<int>(current_corners.size()); ++idx) {
        const cv::Point2d diff = current_corners[static_cast<size_t>(idx)].uv -
                                 predicted_uv;
        const double dist_sq = diff.x * diff.x + diff.y * diff.y;
        if (dist_sq < best_dist_sq) {
            second_dist_sq = best_dist_sq;
            best_dist_sq = dist_sq;
            best_idx = idx;
        } else if (dist_sq < second_dist_sq) {
            second_dist_sq = dist_sq;
        }
    }

    match.index = best_idx;
    match.best_dist = std::sqrt(best_dist_sq);
    match.second_dist = std::sqrt(second_dist_sq);

    if (match.best_dist > max_dist_px) {
        match.index = -1;
        match.reject_reason = "far";
        return match;
    }

    const double min_margin =
        config_.persistence_match_min_second_best_margin_px;
    if (
        min_margin > 0.0 &&
        std::isfinite(match.second_dist) &&
        (match.second_dist - match.best_dist) < min_margin
    ) {
        match.index = -1;
        match.reject_reason = "ambiguous";
        return match;
    }

    if (
        best_idx >= 0 &&
        best_idx < static_cast<int>(used_current_indices.size()) &&
        used_current_indices[static_cast<size_t>(best_idx)]
    ) {
        match.index = -1;
        match.reject_reason = "claimed";
        return match;
    }

    return match;
}

std::vector<PersistentMatcher::DetectedCornerCpp>
PersistentMatcher::detectedCornersFromDetection(
    const CheckerboardDetection& detection
)
{
    std::vector<DetectedCornerCpp> corners;
    corners.reserve(detection.corners.size());

    for (const GridCorner& corner : detection.corners) {
        DetectedCornerCpp out;
        out.local_row = corner.j;
        out.local_col = corner.i;
        out.uv = cv::Point2d(
            static_cast<double>(corner.uv.x),
            static_cast<double>(corner.uv.y)
        );
        out.visibility_score = static_cast<double>(corner.visibility_score);
        out.observed_frames = corner.observed_frames;
        out.predicted = corner.predicted;
        corners.push_back(out);
    }

    return corners;
}

cv::Mat PersistentMatcher::makeDistCoeffsMat(
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

bool PersistentMatcher::vectorToMat3x1(
    const std::vector<double>& values,
    cv::Mat& mat
)
{
    if (values.size() != 3) {
        mat.release();
        return false;
    }

    mat = cv::Mat(3, 1, CV_64F);
    mat.at<double>(0, 0) = values[0];
    mat.at<double>(1, 0) = values[1];
    mat.at<double>(2, 0) = values[2];
    return true;
}

bool PersistentMatcher::projectPoint(
    const std::array<double, 3>& xyz_mm,
    const cv::Matx33d& K,
    const cv::Mat& dist_coeffs,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    cv::Point2d& uv
)
{
    std::vector<cv::Point3d> object_points;
    object_points.emplace_back(xyz_mm[0], xyz_mm[1], xyz_mm[2]);

    std::vector<cv::Point2d> projected;
    try {
        cv::projectPoints(
            object_points,
            rvec,
            tvec,
            K,
            dist_coeffs,
            projected
        );
    } catch (...) {
        return false;
    }

    if (projected.empty()) {
        return false;
    }

    uv = projected[0];
    return std::isfinite(uv.x) && std::isfinite(uv.y);
}

MapPoseTrackerConfig PersistentMatcher::makeMapPoseTrackerConfig(
    const TrackerConfig& config
)
{
    MapPoseTrackerConfig pose_config;
    pose_config.min_points = std::max({
        config.min_points,
        config.persistence_min_points,
        config.fast_persistent_min_points
    });
    pose_config.min_inliers = config.min_inliers;
    pose_config.ransac_reproj_px = config.pnp_ransac_reprojection_px;
    pose_config.ransac_confidence = config.pnp_ransac_confidence;
    pose_config.ransac_iterations = config.pnp_ransac_iterations;
    pose_config.max_mean_reproj_px = config.max_mean_reprojection_error_px;
    pose_config.max_max_reproj_px = config.max_max_reprojection_error_px;
    pose_config.max_translation_jump_mm = config.max_translation_jump_mm;
    pose_config.max_rotation_jump_deg = config.max_rotation_jump_deg;
    pose_config.rotation_gate_scale_per_lost_frame =
        config.rotation_gate_scale_per_lost_frame;
    pose_config.rotation_gate_max_deg = config.rotation_gate_max_deg;
    pose_config.use_pose_prior = config.use_pose_prior;
    pose_config.refine_with_iterative = true;
    pose_config.use_direct_prior_solver = config.pnp_direct_prior_enabled;
    pose_config.direct_refine_method = config.pnp_direct_refine_method;
    pose_config.direct_max_mean_reproj_px =
        config.pnp_direct_max_mean_reprojection_error_px;
    pose_config.direct_max_max_reproj_px =
        config.pnp_direct_max_max_reprojection_error_px;
    return pose_config;
}

} // namespace hydramarker
