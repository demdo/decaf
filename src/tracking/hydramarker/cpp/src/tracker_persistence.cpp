#include "tracker_persistence.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>

#include <opencv2/calib3d.hpp>

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

} // namespace hydramarker
