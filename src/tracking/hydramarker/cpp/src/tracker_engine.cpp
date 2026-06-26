#include "tracker_engine.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include <opencv2/calib3d.hpp>

namespace hydramarker {

namespace {

double elapsedMs(std::int64_t start_tick)
{
    return (static_cast<double>(cv::getTickCount() - start_tick)
            / cv::getTickFrequency()) * 1000.0;
}

std::string formatDouble(double value, int precision)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

} // namespace

TrackerEngine::TrackerEngine(
    const std::string& field_path,
    const std::string& marker_json_path,
    const cv::Matx33d& K,
    const std::vector<double>& dist_coeffs,
    const TrackerConfig& config
)
    : config_(config),
      K_(K),
      dist_coeffs_(dist_coeffs),
      field_(MarkerField::loadFromFile(field_path)),
      geometry_(MarkerGeometry::loadFromJson(marker_json_path)),
      checkerboard_detector_(makeCheckerboardConfig(config)),
      dot_detector_(makeDotDetectorConfig(config)),
      patch_decoder_(makePatchDecoderConfig(config)),
      correspondence_builder_(makeCorrespondenceBuilderConfig(config)),
      pose_tracker_(K, dist_coeffs, makeMapPoseTrackerConfig(config)),
      pose_depth_filter_(K, dist_coeffs, makePoseDepthFilterConfig(config)),
      tracker_geometry_(geometry_, K, dist_coeffs, config),
      persistent_matcher_(config)
{
    if (field_.empty()) {
        throw std::runtime_error("TrackerEngine could not load marker field");
    }
    if (geometry_.empty()) {
        throw std::runtime_error("TrackerEngine could not load marker geometry");
    }
}

void TrackerEngine::reset()
{
    mode_ = TrackerMode::Lost;
    frame_index_ = 0;
    lost_frames_ = 0;
    low_fresh_correspondence_frames_ = 0;
    pose_propagation_block_until_frame_ = -1;
    max_pts_seen_ = 0;
    last_good_reproj_px_ = -1.0;
    last_accepted_pose_frame_ = -1;
    last_accepted_visual_corner_count_ = 0;
    last_accepted_rvec_.clear();
    last_accepted_tvec_.clear();
    last_accepted_T_marker_camera_.clear();
    checkerboard_detector_.resetTracking();
    dot_detector_.reset();
    pose_tracker_.reset();
    pose_depth_filter_.reset();
    persistent_matcher_.reset();
}

TrackerFrameResult TrackerEngine::processFrame(
    const cv::Mat& frame,
    bool run_detection
)
{
    const std::int64_t frame_t0 = cv::getTickCount();
    TrackerFrameResult result;
    result.mode = mode_;

    ++frame_index_;
    result.frame_index = frame_index_;

    if (!run_detection) {
        result.message = "Idle: checkerboard detection skipped.";
        result.timings_ms["checkerboard_ms"] = 0.0;
        result.timings_ms["idle_skip"] = 1.0;
        finalizeFrameResult(result, frame_t0);
        return result;
    }

    const std::int64_t detect_t0 = cv::getTickCount();
    std::optional<CheckerboardDetection> detection =
        checkerboard_detector_.detect(frame);
    result.timings_ms["checkerboard_ms"] = elapsedMs(detect_t0);

    for (const auto& item : checkerboard_detector_.lastTimingsMs()) {
        result.timings_ms["checkerboard_" + item.first] = item.second;
    }

    if (!detection.has_value() || !detection->valid()) {
        if (detection.has_value()) {
            attachDetectionResult(result, *detection);
        }
        onTrackingFailure();
        const std::int64_t hold_t0 = cv::getTickCount();
        if (tryHoldLastPoseWithoutDetection(result, frame_t0)) {
            result.timings_ms["hold_pose_ms"] = elapsedMs(hold_t0);
            return result;
        }
        result.timings_ms["hold_pose_ms"] = elapsedMs(hold_t0);

        const std::int64_t emergency_t0 = cv::getTickCount();
        if (tryEmergencyLastPose(
                nullptr,
                "No valid checkerboard detection",
                result,
                frame_t0
            )) {
            result.timings_ms["emergency_hold_ms"] =
                elapsedMs(emergency_t0);
            return result;
        }
        result.timings_ms["emergency_hold_ms"] = elapsedMs(emergency_t0);

        result.message = "No valid checkerboard detection.";
        finalizeFrameResult(result, frame_t0);
        return result;
    }

    attachDetectionResult(result, *detection);
    attachRuntimeState(result);

    if (tryFastPersistentPose(*detection, result, frame_t0)) {
        return result;
    }

    const CheckerboardDetection* detection_for_decode = &(*detection);
    std::optional<CheckerboardDetection> propagated_detection;
    const std::int64_t propagation_t0 = cv::getTickCount();
    propagated_detection =
        buildPosePropagatedDetection(frame.cols, frame.rows);
    result.timings_ms["pose_propagation_ms"] = elapsedMs(propagation_t0);
    if (propagated_detection.has_value()) {
        detection_for_decode = &(*propagated_detection);
        result.timings_ms["pose_propagation_used_count"] = 1.0;
        result.timings_ms["pose_propagation_corner_count"] =
            static_cast<double>(propagated_detection->corners.size());
        result.timings_ms["pose_propagation_cell_count"] =
            static_cast<double>(propagated_detection->cells.size());
    } else {
        result.timings_ms["pose_propagation_used_count"] = 0.0;
    }

    const std::int64_t dot_t0 = cv::getTickCount();
    DotDetectionResult dots = dot_detector_.detect(
        frame,
        *detection_for_decode
    );
    result.timings_ms["dot_detect_ms"] = elapsedMs(dot_t0);
    result.dot_cell_count = static_cast<int>(dots.cells.size());
    for (const DotCellObservation& cell : dots.cells) {
        if (cell.valid && !cell.ambiguous) {
            ++result.dot_valid_cell_count;
        }
    }

    const std::int64_t patch_t0 = cv::getTickCount();
    std::vector<LocalPatch> patches = patch_extractor_.extract(
        dots,
        field_.patchSize()
    );
    result.timings_ms["patch_extract_ms"] = elapsedMs(patch_t0);
    result.patch_count = static_cast<int>(patches.size());

    const std::int64_t decode_t0 = cv::getTickCount();
    std::vector<DecodedPatch> decoded = patch_decoder_.decode(patches, field_);
    result.timings_ms["patch_decode_ms"] = elapsedMs(decode_t0);
    result.decoded_patch_count = static_cast<int>(decoded.size());

    std::vector<DecodedPatch> decoded_valid;
    decoded_valid.reserve(decoded.size());
    for (const DecodedPatch& patch : decoded) {
        if (patch.valid && !patch.ambiguous) {
            decoded_valid.push_back(patch);
        }
    }
    result.decoded_valid_patch_count = static_cast<int>(decoded_valid.size());

    if (decoded_valid.empty()) {
        const std::string reason = "No valid decoded patches";
        const std::int64_t fallback_t0 = cv::getTickCount();
        if (tryPersistentFallbackPose(
                *detection_for_decode,
                reason,
                true,
                result,
                frame_t0
            )) {
            result.timings_ms["persistent_fallback_ms"] =
                elapsedMs(fallback_t0);
            return result;
        }
        result.timings_ms["persistent_fallback_ms"] =
            elapsedMs(fallback_t0);

        const std::int64_t hold_t0 = cv::getTickCount();
        if (tryHoldLastPose(
                *detection_for_decode,
                reason,
                std::vector<TrackerCorner>(),
                result,
                frame_t0
            )) {
            result.timings_ms["hold_pose_ms"] = elapsedMs(hold_t0);
            return result;
        }
        result.timings_ms["hold_pose_ms"] = elapsedMs(hold_t0);

        onTrackingFailure();
        const std::int64_t emergency_t0 = cv::getTickCount();
        if (tryEmergencyLastPose(
                &(*detection),
                reason,
                result,
                frame_t0
            )) {
            result.timings_ms["emergency_hold_ms"] =
                elapsedMs(emergency_t0);
            return result;
        }
        result.timings_ms["emergency_hold_ms"] = elapsedMs(emergency_t0);

        result.message = "No valid decoded patches.";
        finalizeFrameResult(result, frame_t0);
        return result;
    }

    const std::int64_t corr_t0 = cv::getTickCount();
    CorrespondenceBuildResult correspondences = correspondence_builder_.build(
        *detection_for_decode,
        decoded_valid,
        geometry_
    );
    result.timings_ms["correspondence_build_ms"] = elapsedMs(corr_t0);
    result.correspondence_count = static_cast<int>(
        correspondences.correspondences.size()
    );

    if (!correspondences.valid()) {
        noteLowFreshCorrespondenceFailure(0);
        const std::string reason = "Correspondence build failed";
        const std::int64_t fallback_t0 = cv::getTickCount();
        if (tryPersistentFallbackPose(
                *detection_for_decode,
                reason,
                false,
                result,
                frame_t0
            )) {
            result.timings_ms["persistent_fallback_ms"] =
                elapsedMs(fallback_t0);
            return result;
        }
        result.timings_ms["persistent_fallback_ms"] =
            elapsedMs(fallback_t0);

        const std::int64_t hold_t0 = cv::getTickCount();
        if (tryHoldLastPose(
                *detection_for_decode,
                reason,
                std::vector<TrackerCorner>(),
                result,
                frame_t0
            )) {
            result.timings_ms["hold_pose_ms"] = elapsedMs(hold_t0);
            return result;
        }
        result.timings_ms["hold_pose_ms"] = elapsedMs(hold_t0);

        onTrackingFailure();
        const std::int64_t emergency_t0 = cv::getTickCount();
        if (tryEmergencyLastPose(
                &(*detection),
                reason,
                result,
                frame_t0
            )) {
            result.timings_ms["emergency_hold_ms"] =
                elapsedMs(emergency_t0);
            return result;
        }
        result.timings_ms["emergency_hold_ms"] = elapsedMs(emergency_t0);

        result.message = "Correspondence build failed.";
        finalizeFrameResult(result, frame_t0);
        return result;
    }

    std::vector<PoseTrackPoint> pose_points;
    std::vector<TrackerCorner> correspondence_corners;
    pose_points.reserve(correspondences.correspondences.size());
    correspondence_corners.reserve(correspondences.correspondences.size());
    for (const Correspondence2D3D& corr : correspondences.correspondences) {
        pose_points.push_back(posePointFromCorrespondence(corr));
        correspondence_corners.push_back(trackerCornerFromCorrespondence(corr));
    }

    if (static_cast<int>(pose_points.size()) < config_.min_points) {
        noteLowFreshCorrespondenceFailure(static_cast<int>(pose_points.size()));
        const std::string reason =
            "Too few correspondences: " +
            std::to_string(pose_points.size()) +
            " < " + std::to_string(config_.min_points);

        const std::int64_t fallback_t0 = cv::getTickCount();
        if (tryPersistentFallbackPose(
                *detection_for_decode,
                reason,
                false,
                result,
                frame_t0
            )) {
            result.timings_ms["persistent_fallback_ms"] =
                elapsedMs(fallback_t0);
            return result;
        }
        result.timings_ms["persistent_fallback_ms"] =
            elapsedMs(fallback_t0);

        const std::int64_t hold_t0 = cv::getTickCount();
        if (tryHoldLastPose(
                *detection_for_decode,
                reason,
                correspondence_corners,
                result,
                frame_t0
            )) {
            result.timings_ms["hold_pose_ms"] = elapsedMs(hold_t0);
            return result;
        }
        result.timings_ms["hold_pose_ms"] = elapsedMs(hold_t0);

        onTrackingFailure();
        const std::int64_t emergency_t0 = cv::getTickCount();
        if (tryEmergencyLastPose(
                &(*detection),
                reason,
                result,
                frame_t0
            )) {
            result.timings_ms["emergency_hold_ms"] =
                elapsedMs(emergency_t0);
            return result;
        }
        result.timings_ms["emergency_hold_ms"] = elapsedMs(emergency_t0);

        result.message = reason + ".";
        result.num_points = static_cast<int>(pose_points.size());
        result.correspondence_count = static_cast<int>(pose_points.size());
        result.correspondence_corners = correspondence_corners;
        finalizeFrameResult(result, frame_t0);
        return result;
    }

    low_fresh_correspondence_frames_ = 0;

    const std::vector<double> previous_rvec = pose_tracker_.rvec();
    const std::vector<double> previous_tvec = pose_tracker_.tvec();
    const PoseDepthFilterSnapshot depth_snapshot =
        pose_depth_filter_.snapshot();

    const std::int64_t pose_t0 = cv::getTickCount();
    MapPoseResult pose = pose_tracker_.estimatePose(pose_points, lost_frames_);
    result.timings_ms["pnp_ms"] = elapsedMs(pose_t0);

    std::optional<PoseDepthFilterResult> filtered_depth;
    PlateauPosePriorResult plateau_prior;
    bool plateau_triggered = false;
    bool plateau_applied = false;
    std::vector<TrackerCorner> visual_corners;
    if (pose.success) {
        MapPoseResult raw_pose = pose;
        const std::int64_t depth_t0 = cv::getTickCount();
        filtered_depth = applyDepthFilterToPose(pose, pose_points);
        result.timings_ms["pose_depth_filter_pre_ms"] = elapsedMs(depth_t0);

        const std::int64_t plateau_t0 = cv::getTickCount();
        std::optional<MapPoseResult> prior_pose = maybeApplyPlateauPrior(
            raw_pose,
            pose,
            pose_points,
            filtered_depth,
            previous_rvec,
            previous_tvec,
            plateau_prior,
            plateau_triggered,
            plateau_applied
        );
        result.timings_ms["pose_plateau_prior_ms"] = elapsedMs(plateau_t0);
        if (prior_pose.has_value()) {
            pose_depth_filter_.restore(depth_snapshot);
            pose = *prior_pose;
            pose_tracker_.setPose(pose.rvec, pose.tvec);
            const std::int64_t prior_depth_t0 = cv::getTickCount();
            filtered_depth = applyDepthFilterToPose(pose, pose_points);
            result.timings_ms["pose_depth_filter_prior_ms"] =
                elapsedMs(prior_depth_t0);
        }
        visual_corners = visualCornersForPose(pose);
    } else {
        result.timings_ms["pose_depth_filter_pre_ms"] = 0.0;
        result.timings_ms["pose_plateau_prior_ms"] = 0.0;
    }
    packagePoseResult(
        result,
        pose,
        pose.success ? PoseSource::Decode : PoseSource::None,
        pose.message,
        visual_corners,
        correspondence_corners
    );
    attachDepthFilterResult(result, filtered_depth);
    attachPlateauPriorResult(
        result,
        plateau_prior,
        plateau_triggered,
        plateau_triggered,
        plateau_applied
    );

    if (pose.success) {
        result.current_pose_accepted =
            acceptPoseState(pose, static_cast<int>(visual_corners.size()));
        std::string decode_update_reject_reason;
        if (!decodeUpdateRejectionReason(visual_corners, decode_update_reject_reason)) {
            refreshPersistentIdentities(visual_corners, frame_index_);
            result.timings_ms["pose_decode_update_accepted_count"] = 1.0;
        } else {
            result.timings_ms["pose_decode_update_accepted_count"] = 0.0;
        }
        lost_frames_ = 0;
        low_fresh_correspondence_frames_ = 0;
    } else {
        onTrackingFailure();
        const std::int64_t emergency_t0 = cv::getTickCount();
        if (tryEmergencyLastPose(
                &(*detection),
                pose.message,
                result,
                frame_t0
            )) {
            result.timings_ms["emergency_hold_ms"] =
                elapsedMs(emergency_t0);
            return result;
        }
        result.timings_ms["emergency_hold_ms"] = elapsedMs(emergency_t0);
    }

    if (pose.success) {
        mode_ = TrackerMode::Tracking;
    }
    finalizeFrameResult(result, frame_t0);
    return result;
}

PoseTrackPoint TrackerEngine::posePointFromCorrespondence(
    const Correspondence2D3D& corr
)
{
    PoseTrackPoint point;
    point.global_row = corr.global_row;
    point.global_col = corr.global_col;
    point.xyz_mm = {
        static_cast<double>(corr.xyz_mm.x),
        static_cast<double>(corr.xyz_mm.y),
        static_cast<double>(corr.xyz_mm.z)
    };
    point.uv = {
        static_cast<double>(corr.uv.x),
        static_cast<double>(corr.uv.y)
    };
    point.votes = corr.votes;
    return point;
}

TrackerCorner TrackerEngine::trackerCornerFromCorrespondence(
    const Correspondence2D3D& corr
)
{
    TrackerCorner corner;
    corner.local_row = corr.local_row;
    corner.local_col = corr.local_col;
    corner.global_row = corr.global_row;
    corner.global_col = corr.global_col;
    corner.xyz_mm = {
        static_cast<double>(corr.xyz_mm.x),
        static_cast<double>(corr.xyz_mm.y),
        static_cast<double>(corr.xyz_mm.z)
    };
    corner.uv = {
        static_cast<double>(corr.uv.x),
        static_cast<double>(corr.uv.y)
    };
    corner.votes = corr.votes;
    return corner;
}

TrackerCorner TrackerEngine::trackerCornerFromPosePoint(
    const PoseTrackPoint& point
)
{
    TrackerCorner corner;
    corner.local_row = -1;
    corner.local_col = -1;
    corner.global_row = point.global_row;
    corner.global_col = point.global_col;
    corner.xyz_mm = point.xyz_mm;
    corner.uv = point.uv;
    corner.votes = point.votes;
    return corner;
}

GlobalCornerIdentity TrackerEngine::identityFromTrackerCorner(
    const TrackerCorner& corner
)
{
    GlobalCornerIdentity identity;
    identity.global_row = corner.global_row;
    identity.global_col = corner.global_col;
    identity.xyz_mm = corner.xyz_mm;
    identity.uv = corner.uv;
    identity.votes = corner.votes;
    return identity;
}

FrameDetectedCorner TrackerEngine::detectedCornerFromGridCorner(
    const GridCorner& corner
)
{
    FrameDetectedCorner out;
    out.local_row = corner.j;
    out.local_col = corner.i;
    out.uv = {
        static_cast<double>(corner.uv.x),
        static_cast<double>(corner.uv.y)
    };
    return out;
}

void TrackerEngine::attachDetectionResult(
    TrackerFrameResult& result,
    const CheckerboardDetection& detection
) const
{
    result.detection_valid = detection.valid();
    result.detection_tracking = detection.tracking;
    result.detection_stable = detection.stable;
    result.detection_corner_count =
        static_cast<int>(detection.corners.size());
    result.detection_cell_count =
        static_cast<int>(detection.cells.size());
    result.detection_corners.clear();
    result.detection_corners.reserve(detection.corners.size());
    for (const GridCorner& corner : detection.corners) {
        result.detection_corners.push_back(detectedCornerFromGridCorner(corner));
    }
}

void TrackerEngine::packagePoseResult(
    TrackerFrameResult& result,
    const MapPoseResult& pose,
    PoseSource source,
    const std::string& message,
    const std::vector<TrackerCorner>& visual_corners,
    const std::vector<TrackerCorner>& correspondence_corners
) const
{
    result.success = pose.success;
    result.message = message;
    result.pose_source = source;
    result.rvec = pose.rvec;
    result.tvec = pose.tvec;
    result.T_marker_camera = pose.T_marker_camera;
    result.num_points = pose.num_points;
    result.num_inliers = pose.num_inliers;
    result.mean_reprojection_error_px = pose.reprojection_mean_px;
    result.max_reprojection_error_px = pose.reprojection_max_px;
    result.confidence = confidence(
        pose.num_inliers,
        pose.reprojection_mean_px,
        config_
    );
    result.pnp_method = pose.method;
    result.visual_corner_count = static_cast<int>(visual_corners.size());
    result.corners = visual_corners;
    result.correspondence_corners = correspondence_corners;
}

std::vector<TrackerCorner> TrackerEngine::visualCornersForPose(
    const MapPoseResult& pose
) const
{
    std::vector<TrackerCorner> pose_corners;
    pose_corners.reserve(pose.points.size());
    for (const PoseTrackPoint& point : pose.points) {
        pose_corners.push_back(trackerCornerFromPosePoint(point));
    }

    return tracker_geometry_.visualCornersFromPose(
        pose_corners,
        pose.rvec,
        pose.tvec,
        config_.visual_corner_max_reprojection_error_px
    );
}

void TrackerEngine::pointsToCv(
    const std::vector<PoseTrackPoint>& points,
    std::vector<cv::Point3d>& object_points,
    std::vector<cv::Point2d>& image_points
) const
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

bool TrackerEngine::reprojectionMeanMax(
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points,
    const std::vector<double>& rvec,
    const std::vector<double>& tvec,
    double& mean_error_px,
    double& max_error_px
) const
{
    if (
        object_points.empty() ||
        object_points.size() != image_points.size() ||
        rvec.size() != 3 ||
        tvec.size() != 3
    ) {
        return false;
    }

    cv::Mat rvec_mat(3, 1, CV_64F);
    cv::Mat tvec_mat(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        rvec_mat.at<double>(i, 0) = rvec[static_cast<size_t>(i)];
        tvec_mat.at<double>(i, 0) = tvec[static_cast<size_t>(i)];
    }

    cv::Mat dist_mat;
    if (!dist_coeffs_.empty()) {
        dist_mat = cv::Mat(dist_coeffs_).clone();
    }

    std::vector<cv::Point2d> projected;
    try {
        cv::projectPoints(
            object_points,
            rvec_mat,
            tvec_mat,
            K_,
            dist_mat,
            projected
        );
    } catch (const cv::Exception&) {
        return false;
    }
    if (projected.size() != image_points.size()) {
        return false;
    }

    double sum = 0.0;
    max_error_px = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
        const double error = cv::norm(projected[i] - image_points[i]);
        sum += error;
        max_error_px = std::max(max_error_px, error);
    }
    mean_error_px = sum / static_cast<double>(projected.size());
    return true;
}

std::optional<PoseDepthFilterResult> TrackerEngine::applyDepthFilterToPose(
    MapPoseResult& pose,
    const std::vector<PoseTrackPoint>& fallback_points
)
{
    if (!config_.pose_depth_filter_enabled) {
        return std::nullopt;
    }
    if (!pose.success || pose.rvec.size() != 3 || pose.tvec.size() != 3) {
        return std::nullopt;
    }

    const std::vector<PoseTrackPoint>& filter_points =
        pose.points.empty() ? fallback_points : pose.points;
    if (
        static_cast<int>(filter_points.size()) <
        std::max(1, config_.pose_depth_filter_min_points)
    ) {
        return std::nullopt;
    }

    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2d> image_points;
    pointsToCv(filter_points, object_points, image_points);

    PoseDepthFilterResult filtered = pose_depth_filter_.update(
        pose.rvec,
        pose.tvec,
        object_points,
        image_points
    );

    pose.rvec = filtered.rvec;
    pose.tvec = filtered.tvec;
    pose.T_marker_camera = filtered.T_marker_camera;

    double mean_error_px = -1.0;
    double max_error_px = -1.0;
    if (
        reprojectionMeanMax(
            object_points,
            image_points,
            pose.rvec,
            pose.tvec,
            mean_error_px,
            max_error_px
        )
    ) {
        pose.reprojection_mean_px = mean_error_px;
        pose.reprojection_max_px = max_error_px;
    }

    pose_tracker_.setPose(pose.rvec, pose.tvec);
    return filtered;
}

bool TrackerEngine::plateauPriorTriggered(
    const std::optional<PoseDepthFilterResult>& filtered
) const
{
    if (!filtered.has_value()) {
        return false;
    }
    if (!config_.pose_plateau_prior_enabled) {
        return false;
    }
    if (
        filtered->delta_z_mm >=
        config_.pose_plateau_prior_trigger_negative_delta_mm
    ) {
        return false;
    }
    if (!std::isfinite(filtered->object_z_span_mm)) {
        return false;
    }
    return filtered->object_z_span_mm >=
        config_.pose_plateau_prior_min_object_z_span_mm;
}

std::optional<MapPoseResult> TrackerEngine::maybeApplyPlateauPrior(
    const MapPoseResult& raw_pose,
    MapPoseResult& pose,
    const std::vector<PoseTrackPoint>& fallback_points,
    const std::optional<PoseDepthFilterResult>& filtered,
    const std::vector<double>& previous_rvec,
    const std::vector<double>& previous_tvec,
    PlateauPosePriorResult& prior,
    bool& triggered,
    bool& applied
)
{
    triggered = plateauPriorTriggered(filtered);
    applied = false;
    prior = PlateauPosePriorResult();

    if (!triggered) {
        return std::nullopt;
    }

    if (previous_rvec.size() != 3 || previous_tvec.size() != 3) {
        prior.method = "none";
        prior.reason = "missing_previous_pose";
        return std::nullopt;
    }
    if (
        !raw_pose.success ||
        raw_pose.rvec.size() != 3 ||
        raw_pose.tvec.size() != 3
    ) {
        prior.method = "none";
        prior.reason = "missing_raw_pose";
        return std::nullopt;
    }

    const std::vector<PoseTrackPoint>& points =
        raw_pose.points.empty() ? fallback_points : raw_pose.points;
    if (
        static_cast<int>(points.size()) <
        std::max(1, config_.pose_plateau_prior_min_points)
    ) {
        prior.method = "none";
        prior.reason = "too_few_points";
        return std::nullopt;
    }

    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2d> image_points;
    pointsToCv(points, object_points, image_points);

    prior = solvePlateauPosePrior(
        object_points,
        image_points,
        K_,
        dist_coeffs_,
        raw_pose.rvec,
        raw_pose.tvec,
        previous_rvec,
        previous_tvec,
        makePlateauPosePriorConfig(config_)
    );

    if (
        !prior.success ||
        prior.rvec.size() != 3 ||
        prior.tvec.size() != 3
    ) {
        return std::nullopt;
    }

    MapPoseResult prior_pose = raw_pose;
    prior_pose.rvec = prior.rvec;
    prior_pose.tvec = prior.tvec;
    prior_pose.T_marker_camera = prior.T_marker_camera;
    prior_pose.reprojection_mean_px = prior.reprojection_mean_px;
    prior_pose.reprojection_max_px = prior.reprojection_max_px;
    prior_pose.method =
        raw_pose.method + "+plateau_prior_" + prior.method;
    prior_pose.message = "Pose refined by triggered plateau prior.";
    pose = prior_pose;
    applied = true;
    return prior_pose;
}

void TrackerEngine::attachDepthFilterResult(
    TrackerFrameResult& result,
    const std::optional<PoseDepthFilterResult>& filtered
) const
{
    result.depth_filter_available = filtered.has_value();
    if (!filtered.has_value()) {
        return;
    }

    result.depth_filter_applied = filtered->applied;
    result.depth_filter_delta_z_mm = filtered->delta_z_mm;
    result.depth_filter_raw_z_mm = filtered->raw_z_mm;
    result.depth_filter_z_mm = filtered->filtered_z_mm;
    result.depth_filter_reproj_excess_px = filtered->reprojection_excess_px;
    result.depth_filter_guard_alpha = filtered->guard_alpha;
    result.depth_filter_innovation_z_mm = filtered->innovation_z_mm;
    result.depth_filter_innovation_mean_z_mm =
        filtered->innovation_mean_z_mm;
    result.depth_filter_innovation_cusum_pos_mm =
        filtered->innovation_cusum_pos_mm;
    result.depth_filter_innovation_cusum_neg_mm =
        filtered->innovation_cusum_neg_mm;
    result.depth_filter_innovation_bias_detected =
        filtered->innovation_bias_detected;
    result.depth_filter_innovation_bias_direction =
        filtered->innovation_bias_direction;
    result.depth_filter_innovation_bias_limited =
        filtered->innovation_bias_limited;
    result.depth_filter_object_z_span_mm = filtered->object_z_span_mm;
    result.depth_filter_negative_delta_guard_limited =
        filtered->negative_delta_guard_limited;
}

void TrackerEngine::attachPlateauPriorResult(
    TrackerFrameResult& result,
    const PlateauPosePriorResult& prior,
    bool triggered,
    bool attempted,
    bool applied
) const
{
    result.pose_plateau_prior_triggered = triggered;
    result.pose_plateau_prior_attempted = attempted;
    result.pose_plateau_prior_applied = applied && prior.success;
    result.pose_plateau_prior_method = prior.method;
    result.pose_plateau_prior_reason = prior.reason;
    result.pose_plateau_prior_delta_z_mm = prior.delta_z_mm;
    result.pose_plateau_prior_reproj_excess_px =
        prior.reprojection_excess_px;
    result.pose_plateau_prior_max_reproj_excess_px =
        prior.max_reprojection_excess_px;
    result.pose_plateau_prior_iterations = prior.iterations;
}

bool TrackerEngine::acceptPoseState(
    const MapPoseResult& pose,
    int visual_corner_count
)
{
    if (!pose.success || visual_corner_count < config_.visual_corner_min_count) {
        return false;
    }

    max_pts_seen_ = std::max(max_pts_seen_, pose.num_inliers);
    if (pose.reprojection_mean_px >= 0.0) {
        last_good_reproj_px_ = pose.reprojection_mean_px;
    }
    last_accepted_rvec_ = pose.rvec;
    last_accepted_tvec_ = pose.tvec;
    last_accepted_T_marker_camera_ = pose.T_marker_camera;
    last_accepted_pose_frame_ = frame_index_;
    last_accepted_visual_corner_count_ = visual_corner_count;
    return true;
}

void TrackerEngine::clearAcceptedState()
{
    max_pts_seen_ = 0;
    last_good_reproj_px_ = -1.0;
    last_accepted_pose_frame_ = -1;
    last_accepted_visual_corner_count_ = 0;
    last_accepted_rvec_.clear();
    last_accepted_tvec_.clear();
    last_accepted_T_marker_camera_.clear();
}

void TrackerEngine::attachRuntimeState(TrackerFrameResult& result) const
{
    result.mode = mode_;
    result.lost_frames = lost_frames_;
    result.persistent_count =
        static_cast<int>(persistent_matcher_.identities().size());

    result.pose_tracker_has_pose = pose_tracker_.hasPose();
    if (result.pose_tracker_has_pose) {
        result.pose_tracker_rvec = pose_tracker_.rvec();
        result.pose_tracker_tvec = pose_tracker_.tvec();
        result.pose_tracker_T_marker_camera = pose_tracker_.TMarkerCamera();
    } else {
        result.pose_tracker_rvec.clear();
        result.pose_tracker_tvec.clear();
        result.pose_tracker_T_marker_camera.clear();
    }

    result.max_pts_seen = max_pts_seen_;
    result.last_good_reproj_px = last_good_reproj_px_;
    result.has_accepted_pose =
        last_accepted_rvec_.size() == 3 &&
        last_accepted_tvec_.size() == 3 &&
        last_accepted_T_marker_camera_.size() == 16 &&
        last_accepted_pose_frame_ >= 0;
    result.accepted_pose_frame = result.has_accepted_pose
        ? last_accepted_pose_frame_
        : -1;
    result.accepted_visual_corner_count = result.has_accepted_pose
        ? last_accepted_visual_corner_count_
        : 0;
    result.accepted_rvec = result.has_accepted_pose
        ? last_accepted_rvec_
        : std::vector<double>();
    result.accepted_tvec = result.has_accepted_pose
        ? last_accepted_tvec_
        : std::vector<double>();
    result.accepted_T_marker_camera = result.has_accepted_pose
        ? last_accepted_T_marker_camera_
        : std::vector<double>();
}

void TrackerEngine::finalizeFrameResult(
    TrackerFrameResult& result,
    std::int64_t frame_t0
) const
{
    attachRuntimeState(result);
    result.timings_ms["tracker_total_ms"] = elapsedMs(frame_t0);
}

void TrackerEngine::noteLowFreshCorrespondenceFailure(int fresh_count)
{
    if (
        fresh_count >=
        config_.checker_min_fresh_correspondences_for_stable_tracking
    ) {
        low_fresh_correspondence_frames_ = 0;
        return;
    }

    ++low_fresh_correspondence_frames_;
    if (
        low_fresh_correspondence_frames_ >
        config_.checker_max_low_fresh_correspondence_frames
    ) {
        forceLocalRecovery();
    }
}

void TrackerEngine::forceLocalRecovery()
{
    checkerboard_detector_.resetTracking();
    dot_detector_.reset();
    persistent_matcher_.clearIdentities();
    low_fresh_correspondence_frames_ = 0;
    pose_propagation_block_until_frame_ = std::max(
        pose_propagation_block_until_frame_,
        frame_index_ + 5
    );
}

bool TrackerEngine::detectionHasDecodeableCellSpan(
    const CheckerboardDetection& detection
) const
{
    if (detection.cells.empty()) {
        return false;
    }

    const int min_span =
        std::max(1, config_.checker_min_tracking_decode_cell_span);
    int min_row = detection.cells.front().j;
    int max_row = detection.cells.front().j;
    int min_col = detection.cells.front().i;
    int max_col = detection.cells.front().i;

    for (const GridCell& cell : detection.cells) {
        min_row = std::min(min_row, cell.j);
        max_row = std::max(max_row, cell.j);
        min_col = std::min(min_col, cell.i);
        max_col = std::max(max_col, cell.i);
    }

    return (max_row - min_row + 1) >= min_span &&
           (max_col - min_col + 1) >= min_span;
}

std::optional<CheckerboardDetection>
TrackerEngine::buildPosePropagatedDetection(
    int image_width,
    int image_height
) const
{
    if (config_.decode_only_mode || !config_.enable_pose_propagation) {
        return std::nullopt;
    }

    if (frame_index_ <= pose_propagation_block_until_frame_) {
        return std::nullopt;
    }

    if (!pose_tracker_.hasPose()) {
        return std::nullopt;
    }

    const std::vector<double> rvec = pose_tracker_.rvec();
    const std::vector<double> tvec = pose_tracker_.tvec();
    if (rvec.size() != 3 || tvec.size() != 3) {
        return std::nullopt;
    }

    if (
        last_good_reproj_px_ < 0.0 ||
        last_good_reproj_px_ > config_.pose_propagation_max_reproj_px
    ) {
        return std::nullopt;
    }

    std::vector<cv::Point3d> object_points;
    std::vector<std::pair<int, int>> row_col_list;
    const int rows = geometry_.cornerRows();
    const int cols = geometry_.cornerCols();
    object_points.reserve(static_cast<size_t>(rows * cols));
    row_col_list.reserve(static_cast<size_t>(rows * cols));

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (!geometry_.hasCorner(row, col)) {
                continue;
            }
            const cv::Point3f point = geometry_.cornerPoint(row, col);
            object_points.emplace_back(point.x, point.y, point.z);
            row_col_list.emplace_back(row, col);
        }
    }

    if (static_cast<int>(object_points.size()) < config_.min_points) {
        return std::nullopt;
    }

    cv::Mat rvec_mat(3, 1, CV_64F);
    cv::Mat tvec_mat(3, 1, CV_64F);
    for (int idx = 0; idx < 3; ++idx) {
        rvec_mat.at<double>(idx, 0) = rvec[static_cast<size_t>(idx)];
        tvec_mat.at<double>(idx, 0) = tvec[static_cast<size_t>(idx)];
    }

    cv::Mat dist_mat;
    if (!dist_coeffs_.empty()) {
        dist_mat = cv::Mat(dist_coeffs_).clone();
    }

    std::vector<cv::Point2d> projected;
    try {
        cv::projectPoints(
            object_points,
            rvec_mat,
            tvec_mat,
            K_,
            dist_mat,
            projected
        );
    } catch (const cv::Exception&) {
        return std::nullopt;
    }

    if (projected.size() != object_points.size()) {
        return std::nullopt;
    }

    CheckerboardDetection detection;
    detection.rows = rows;
    detection.cols = cols;

    std::map<std::pair<int, int>, cv::Point2f> ij_to_uv;
    const double border = config_.pose_propagation_border_px;
    for (size_t idx = 0; idx < projected.size(); ++idx) {
        const double u = projected[idx].x;
        const double v = projected[idx].y;
        if (
            u < border ||
            v < border ||
            u >= static_cast<double>(image_width) - border ||
            v >= static_cast<double>(image_height) - border
        ) {
            continue;
        }

        const int row = row_col_list[idx].first;
        const int col = row_col_list[idx].second;

        GridCorner corner;
        corner.j = row;
        corner.i = col;
        corner.uv = cv::Point2f(static_cast<float>(u), static_cast<float>(v));
        corner.visibility_score = 1.0f;
        detection.corners.push_back(corner);
        ij_to_uv[{col, row}] = corner.uv;
    }

    if (static_cast<int>(detection.corners.size()) < config_.min_points) {
        return std::nullopt;
    }

    for (const auto& item : ij_to_uv) {
        const int col = item.first.first;
        const int row = item.first.second;

        const auto p00_it = ij_to_uv.find({col, row});
        const auto p10_it = ij_to_uv.find({col + 1, row});
        const auto p11_it = ij_to_uv.find({col + 1, row + 1});
        const auto p01_it = ij_to_uv.find({col, row + 1});
        if (
            p00_it == ij_to_uv.end() ||
            p10_it == ij_to_uv.end() ||
            p11_it == ij_to_uv.end() ||
            p01_it == ij_to_uv.end()
        ) {
            continue;
        }

        GridCell cell;
        cell.i = col;
        cell.j = row;
        const cv::Point2f p00 = p00_it->second;
        const cv::Point2f p10 = p10_it->second;
        const cv::Point2f p11 = p11_it->second;
        const cv::Point2f p01 = p01_it->second;
        cell.corner_uv = {p00, p10, p11, p01};
        cell.center_uv = cv::Point2f(
            (p00.x + p10.x + p11.x + p01.x) * 0.25f,
            (p00.y + p10.y + p11.y + p01.y) * 0.25f
        );
        detection.cells.push_back(cell);
    }

    if (detection.cells.empty()) {
        return std::nullopt;
    }

    if (!detectionHasDecodeableCellSpan(detection)) {
        return std::nullopt;
    }

    detection.tracking = true;
    detection.stable = true;
    return detection;
}

void TrackerEngine::restorePoseTrackerState(
    bool had_pose,
    const std::vector<double>& rvec,
    const std::vector<double>& tvec
)
{
    pose_tracker_.reset();
    if (had_pose && rvec.size() == 3 && tvec.size() == 3) {
        pose_tracker_.setPose(rvec, tvec);
    }
}

bool TrackerEngine::fallbackPoseRejectionReason(
    const CheckerboardDetection& detection,
    const MapPoseResult& pose,
    std::string& reason
) const
{
    reason.clear();
    if (
        pose.reprojection_mean_px >
        config_.fallback_pose_max_mean_reprojection_error_px
    ) {
        reason =
            "Fallback pose rejected by mean reprojection gate (" +
            formatDouble(pose.reprojection_mean_px, 2) + "px).";
        return true;
    }

    if (
        pose.reprojection_max_px >
        config_.fallback_pose_max_max_reprojection_error_px
    ) {
        reason =
            "Fallback pose rejected by max reprojection gate (" +
            formatDouble(pose.reprojection_max_px, 2) + "px).";
        return true;
    }

    DenseProjectionMatchResult visual = tracker_geometry_.greedyProjectedMatch(
        detection,
        pose.rvec,
        pose.tvec,
        config_.fallback_pose_max_p90_corner_error_px
    );

    const int match_count = static_cast<int>(visual.corners.size());
    if (match_count < config_.fallback_pose_min_detection_matches) {
        reason =
            "Fallback pose rejected by blue-corner alignment (" +
            std::to_string(match_count) + " matches).";
        return true;
    }

    if (
        visual.stats.median_error_px >
        config_.fallback_pose_max_median_corner_error_px
    ) {
        reason =
            "Fallback pose rejected by median blue-corner error (" +
            formatDouble(visual.stats.median_error_px, 2) + "px).";
        return true;
    }

    if (
        visual.stats.p90_error_px >
        config_.fallback_pose_max_p90_corner_error_px
    ) {
        reason =
            "Fallback pose rejected by p90 blue-corner error (" +
            formatDouble(visual.stats.p90_error_px, 2) + "px).";
        return true;
    }

    return false;
}

bool TrackerEngine::decodeUpdateRejectionReason(
    const std::vector<TrackerCorner>& visual_corners,
    std::string& reason
) const
{
    reason.clear();

    const int min_visual =
        std::max(0, config_.decode_update_min_visual_corners);
    if (static_cast<int>(visual_corners.size()) < min_visual) {
        reason =
            "Decode pose rejected by low visual coverage (" +
            std::to_string(visual_corners.size()) + "/" +
            std::to_string(min_visual) + " visible corners).";
        return true;
    }

    const int min_rows = std::max(0, config_.decode_update_min_distinct_rows);
    const int min_cols = std::max(0, config_.decode_update_min_distinct_cols);
    std::set<int> rows;
    std::set<int> cols;
    for (const TrackerCorner& corner : visual_corners) {
        rows.insert(corner.global_row);
        cols.insert(corner.global_col);
    }

    if (
        static_cast<int>(rows.size()) < min_rows ||
        static_cast<int>(cols.size()) < min_cols
    ) {
        reason =
            "Decode pose rejected by narrow marker coverage (rows=" +
            std::to_string(rows.size()) + "/" + std::to_string(min_rows) +
            ", cols=" + std::to_string(cols.size()) + "/" +
            std::to_string(min_cols) + ").";
        return true;
    }

    return false;
}

bool TrackerEngine::tryPersistentFallbackPose(
    const CheckerboardDetection& detection,
    const std::string& reason,
    bool after_decode_fail,
    TrackerFrameResult& result,
    std::int64_t frame_t0
)
{
    if (
        config_.decode_only_mode ||
        !config_.enable_temporal_correspondence_persistence
    ) {
        return false;
    }

    const bool had_pose = pose_tracker_.hasPose();
    const std::vector<double> previous_rvec = pose_tracker_.rvec();
    const std::vector<double> previous_tvec = pose_tracker_.tvec();
    const PoseDepthFilterSnapshot depth_snapshot =
        pose_depth_filter_.snapshot();

    PersistentPoseSeedResult seed = persistent_matcher_.estimatePose(
        detection,
        frame_index_,
        K_,
        dist_coeffs_,
        previous_rvec,
        previous_tvec,
        last_good_reproj_px_,
        lost_frames_
    );

    result.timings_ms["persistent_fallback_match_ms"] = seed.match_ms;
    result.timings_ms["persistent_fallback_pnp_ms"] = seed.pose_ms;
    result.timings_ms["persistent_fallback_total_cpp_ms"] = seed.total_ms;

    const int min_points = after_decode_fail
        ? config_.persistence_min_points_after_decode_fail
        : config_.persistence_min_points;
    if (static_cast<int>(seed.points.size()) < min_points) {
        restorePoseTrackerState(had_pose, previous_rvec, previous_tvec);
        return false;
    }

    if (!seed.pose.success) {
        restorePoseTrackerState(had_pose, previous_rvec, previous_tvec);
        return false;
    }

    std::string reject_reason;
    if (fallbackPoseRejectionReason(detection, seed.pose, reject_reason)) {
        restorePoseTrackerState(had_pose, previous_rvec, previous_tvec);
        return false;
    }

    MapPoseResult pose = seed.pose;
    MapPoseResult raw_pose = seed.pose;
    const std::int64_t depth_t0 = cv::getTickCount();
    std::optional<PoseDepthFilterResult> filtered_depth =
        applyDepthFilterToPose(pose, seed.points);
    result.timings_ms["pose_depth_filter_pre_ms"] = elapsedMs(depth_t0);

    PlateauPosePriorResult plateau_prior;
    bool plateau_triggered = false;
    bool plateau_applied = false;
    const std::int64_t plateau_t0 = cv::getTickCount();
    std::optional<MapPoseResult> prior_pose = maybeApplyPlateauPrior(
        raw_pose,
        pose,
        seed.points,
        filtered_depth,
        previous_rvec,
        previous_tvec,
        plateau_prior,
        plateau_triggered,
        plateau_applied
    );
    result.timings_ms["pose_plateau_prior_ms"] = elapsedMs(plateau_t0);
    if (prior_pose.has_value()) {
        pose_depth_filter_.restore(depth_snapshot);
        pose = *prior_pose;
        pose_tracker_.setPose(pose.rvec, pose.tvec);
        const std::int64_t prior_depth_t0 = cv::getTickCount();
        filtered_depth = applyDepthFilterToPose(pose, seed.points);
        result.timings_ms["pose_depth_filter_prior_ms"] =
            elapsedMs(prior_depth_t0);
    }

    std::vector<TrackerCorner> visual_corners = visualCornersForPose(pose);
    packagePoseResult(
        result,
        pose,
        PoseSource::Persistent,
        "Pose estimated from persistent correspondences after: " + reason + ".",
        visual_corners,
        seed.corners
    );
    attachDepthFilterResult(result, filtered_depth);
    attachPlateauPriorResult(
        result,
        plateau_prior,
        plateau_triggered,
        plateau_triggered,
        plateau_applied
    );
    result.confidence *= 0.85;
    result.current_pose_accepted =
        acceptPoseState(pose, static_cast<int>(visual_corners.size()));
    result.num_points = pose.num_points;
    result.num_inliers = pose.num_inliers;
    result.correspondence_count = static_cast<int>(seed.points.size());

    if (
        pose.reprojection_mean_px >= 0.0 &&
        pose.reprojection_mean_px <= config_.persistence_refresh_mean_error_px
    ) {
        refreshPersistentIdentities(visual_corners, frame_index_);
    }

    mode_ = TrackerMode::Tracking;
    lost_frames_ = 0;
    finalizeFrameResult(result, frame_t0);
    return true;
}

bool TrackerEngine::tryHoldLastPose(
    const CheckerboardDetection& detection,
    const std::string& reason,
    const std::vector<TrackerCorner>& correspondence_corners,
    TrackerFrameResult& result,
    std::int64_t frame_t0
)
{
    if (config_.decode_only_mode || !pose_tracker_.hasPose()) {
        return false;
    }

    if (
        config_.pose_hold_max_frames >= 0 &&
        low_fresh_correspondence_frames_ > config_.pose_hold_max_frames
    ) {
        return false;
    }

    const int detected_count = static_cast<int>(detection.corners.size());
    if (detected_count < config_.pose_hold_min_detection_corners) {
        return false;
    }

    DenseProjectionMatchResult held = tracker_geometry_.greedyProjectedMatch(
        detection,
        pose_tracker_.rvec(),
        pose_tracker_.tvec(),
        config_.visual_corner_max_reprojection_error_px
    );

    const int match_count = static_cast<int>(held.corners.size());
    if (
        match_count < config_.visual_corner_min_count ||
        held.stats.median_error_px >
            config_.visual_corner_max_reprojection_error_px ||
        held.stats.p90_error_px >
            config_.visual_corner_max_reprojection_error_px
    ) {
        return false;
    }

    result.success = true;
    result.pose_source = PoseSource::Hold;
    result.message =
        "Pose held from last accepted pose after: " + reason +
        " (blue_align=" + std::to_string(match_count) +
        ", median=" + formatDouble(held.stats.median_error_px, 2) +
        "px, p90=" + formatDouble(held.stats.p90_error_px, 2) + "px).";
    result.rvec = pose_tracker_.rvec();
    result.tvec = pose_tracker_.tvec();
    result.T_marker_camera = pose_tracker_.TMarkerCamera();
    result.mean_reprojection_error_px = last_good_reproj_px_;
    result.max_reprojection_error_px = -1.0;
    result.num_points = match_count;
    result.num_inliers = match_count;
    result.confidence = 0.25;
    result.visual_corner_count = match_count;
    result.corners = held.corners;
    result.correspondence_corners = correspondence_corners;
    result.correspondence_count =
        static_cast<int>(correspondence_corners.size());

    mode_ = TrackerMode::Tracking;
    lost_frames_ = 0;
    finalizeFrameResult(result, frame_t0);
    return true;
}

bool TrackerEngine::tryHoldLastPoseWithoutDetection(
    TrackerFrameResult& result,
    std::int64_t frame_t0
)
{
    if (config_.decode_only_mode || !pose_tracker_.hasPose()) {
        return false;
    }

    if (
        last_good_reproj_px_ < 0.0 ||
        last_good_reproj_px_ >
            config_.fallback_pose_max_mean_reprojection_error_px
    ) {
        return false;
    }

    result.success = true;
    result.pose_source = PoseSource::Hold;
    result.message =
        "Pose held from last accepted pose without checkerboard detection.";
    result.detection_valid = false;
    result.rvec = pose_tracker_.rvec();
    result.tvec = pose_tracker_.tvec();
    result.T_marker_camera = pose_tracker_.TMarkerCamera();
    result.mean_reprojection_error_px = last_good_reproj_px_;
    result.max_reprojection_error_px = -1.0;
    result.num_points = 0;
    result.num_inliers = 0;
    result.confidence = 0.10;
    result.visual_corner_count = 0;

    finalizeFrameResult(result, frame_t0);
    return true;
}

bool TrackerEngine::tryEmergencyLastPose(
    const CheckerboardDetection* detection,
    const std::string& reason,
    TrackerFrameResult& result,
    std::int64_t frame_t0
)
{
    if (config_.decode_only_mode || !config_.emergency_pose_hold_enabled) {
        return false;
    }

    if (
        last_accepted_rvec_.size() != 3 ||
        last_accepted_tvec_.size() != 3 ||
        last_accepted_T_marker_camera_.size() != 16 ||
        last_accepted_pose_frame_ < 0
    ) {
        return false;
    }

    const int age = frame_index_ - last_accepted_pose_frame_;
    if (age < 0) {
        return false;
    }

    const int max_age = config_.emergency_pose_hold_max_frames;
    if (max_age >= 0 && age > max_age) {
        return false;
    }

    pose_tracker_.setPose(last_accepted_rvec_, last_accepted_tvec_);

    int match_count = 0;
    std::string align_msg = "no_blue_alignment";
    std::vector<TrackerCorner> held_corners;
    if (detection != nullptr && detection->valid()) {
        DenseProjectionMatchResult held =
            tracker_geometry_.greedyProjectedMatch(
                *detection,
                last_accepted_rvec_,
                last_accepted_tvec_,
                config_.visual_corner_max_reprojection_error_px
            );
        const int held_count = static_cast<int>(held.corners.size());
        if (
            held_count >= config_.visual_corner_min_count &&
            held.stats.median_error_px <=
                config_.visual_corner_max_reprojection_error_px &&
            held.stats.p90_error_px <=
                config_.visual_corner_max_reprojection_error_px
        ) {
            match_count = held_count;
            held_corners = held.corners;
            align_msg =
                "blue_align=" + std::to_string(match_count) +
                ", median=" + formatDouble(held.stats.median_error_px, 2) +
                "px, p90=" + formatDouble(held.stats.p90_error_px, 2) +
                "px";
        }
    }

    result.success = true;
    result.pose_source = PoseSource::Hold;
    result.message =
        "Emergency pose held from last accepted pose after: " + reason +
        " (age=" + std::to_string(age) + ", " + align_msg + ").";
    result.rvec = last_accepted_rvec_;
    result.tvec = last_accepted_tvec_;
    result.T_marker_camera = last_accepted_T_marker_camera_;
    result.mean_reprojection_error_px = last_good_reproj_px_;
    result.max_reprojection_error_px = -1.0;
    result.num_points = match_count;
    result.num_inliers = match_count;
    result.confidence = std::max(0.03, 0.20 * std::pow(0.96, std::max(age, 0)));
    result.visual_corner_count = match_count;
    result.corners = held_corners;
    if (detection != nullptr) {
        attachDetectionResult(result, *detection);
    }

    finalizeFrameResult(result, frame_t0);
    return true;
}

void TrackerEngine::onTrackingFailure()
{
    ++lost_frames_;

    if (lost_frames_ > config_.max_lost_frames) {
        mode_ = TrackerMode::Lost;
        pose_tracker_.reset();
        pose_depth_filter_.reset();
        dot_detector_.reset();
        persistent_matcher_.clearIdentities();
        return;
    }

    if (pose_tracker_.hasPose()) {
        mode_ = TrackerMode::Recovering;
        dot_detector_.reset_smoothing();
        return;
    }

    mode_ = TrackerMode::Detecting;
    dot_detector_.reset_smoothing();
}

void TrackerEngine::refreshPersistentIdentities(
    const std::vector<TrackerCorner>& visual_corners,
    int frame_index
)
{
    if (
        config_.decode_only_mode ||
        !config_.enable_temporal_correspondence_persistence ||
        static_cast<int>(visual_corners.size()) < config_.persistence_min_points
    ) {
        return;
    }

    std::vector<GlobalCornerIdentity> identities;
    identities.reserve(visual_corners.size());
    for (const TrackerCorner& corner : visual_corners) {
        identities.push_back(identityFromTrackerCorner(corner));
    }

    persistent_matcher_.replaceIdentities(identities, frame_index);
}

bool TrackerEngine::tryFastPersistentPose(
    const CheckerboardDetection& detection,
    TrackerFrameResult& result,
    std::int64_t frame_t0
)
{
    if (!config_.enable_fast_persistent_path || config_.decode_only_mode) {
        return false;
    }

    const std::vector<double> previous_rvec = pose_tracker_.rvec();
    const std::vector<double> previous_tvec = pose_tracker_.tvec();
    const PoseDepthFilterSnapshot depth_snapshot =
        pose_depth_filter_.snapshot();
    PoseDepthKalmanFilter* depth_filter_ptr =
        config_.pose_depth_filter_enabled ? &pose_depth_filter_ : nullptr;

    const std::int64_t fast_t0 = cv::getTickCount();
    FastPoseResult fast = persistent_matcher_.estimateFastPose(
        detection,
        geometry_,
        frame_index_,
        K_,
        dist_coeffs_,
        pose_tracker_.rvec(),
        pose_tracker_.tvec(),
        last_good_reproj_px_,
        last_accepted_rvec_,
        last_accepted_tvec_,
        lost_frames_,
        depth_filter_ptr,
        max_pts_seen_
    );
    result.timings_ms["fast_persistent_ms"] = elapsedMs(fast_t0);
    result.timings_ms["persistent_match_ms"] = fast.persistent_match_ms;
    result.timings_ms["fast_seed_pnp_ms"] = fast.seed_pnp_ms;
    result.timings_ms["fast_dense_total_ms"] =
        fast.dense_match_ms + fast.dense_pose_ms;
    result.timings_ms["fast_refresh_persistence_cpp_ms"] =
        fast.persistence_refresh_ms;
    result.fast_attempted = fast.attempted;
    result.fast_success = fast.success;
    result.fast_route_decode = fast.route_decode;
    result.fast_matches = static_cast<int>(fast.points.size());
    result.fast_reason = fast.reason;
    result.fast_dense_attempted = fast.dense_attempted;
    result.fast_dense_success = fast.dense_success;
    result.fast_dense_matches = fast.dense_matches;
    result.fast_dense_reason = fast.dense_reason;

    if (!fast.success) {
        return false;
    }

    MapPoseResult raw_pose = fast.pose;
    MapPoseResult pose = fast.depth_filter_available
        ? fast.depth_filtered_pose
        : fast.pose;
    if (
        !pose.success ||
        pose.rvec.size() != 3 ||
        pose.tvec.size() != 3
    ) {
        return false;
    }

    std::optional<PoseDepthFilterResult> filtered_depth;
    if (fast.depth_filter_available) {
        filtered_depth = fast.depth_filter_result;
        result.timings_ms["pose_depth_filter_pre_ms"] = fast.depth_filter_ms;
        result.timings_ms["pose_depth_filter_cpp_precomputed_count"] = 1.0;
    } else {
        result.timings_ms["pose_depth_filter_pre_ms"] = 0.0;
        result.timings_ms["pose_depth_filter_cpp_precomputed_count"] = 0.0;
    }

    PlateauPosePriorResult plateau_prior;
    bool plateau_triggered = false;
    bool plateau_applied = false;
    std::vector<TrackerCorner> visual_corners = fast.visual_corners;
    const std::int64_t plateau_t0 = cv::getTickCount();
    std::optional<MapPoseResult> prior_pose = maybeApplyPlateauPrior(
        raw_pose,
        pose,
        fast.points,
        filtered_depth,
        previous_rvec,
        previous_tvec,
        plateau_prior,
        plateau_triggered,
        plateau_applied
    );
    result.timings_ms["pose_plateau_prior_ms"] = elapsedMs(plateau_t0);
    if (prior_pose.has_value()) {
        pose_depth_filter_.restore(depth_snapshot);
        pose = *prior_pose;
        pose_tracker_.setPose(pose.rvec, pose.tvec);
        const std::int64_t prior_depth_t0 = cv::getTickCount();
        filtered_depth = applyDepthFilterToPose(pose, fast.points);
        result.timings_ms["pose_depth_filter_prior_ms"] =
            elapsedMs(prior_depth_t0);
        visual_corners = visualCornersForPose(pose);
    }

    pose_tracker_.setPose(pose.rvec, pose.tvec);
    packagePoseResult(
        result,
        pose,
        PoseSource::FastPersistent,
        "Fast pose estimated from persistent correspondences.",
        visual_corners,
        fast.corners
    );
    attachDepthFilterResult(result, filtered_depth);
    attachPlateauPriorResult(
        result,
        plateau_prior,
        plateau_triggered,
        plateau_triggered,
        plateau_applied
    );

    result.current_pose_accepted =
        acceptPoseState(pose, static_cast<int>(visual_corners.size()));
    if (prior_pose.has_value()) {
        if (
            pose.reprojection_mean_px >= 0.0 &&
            pose.reprojection_mean_px <=
                config_.fast_persistent_refresh_mean_error_px
        ) {
            refreshPersistentIdentities(visual_corners, frame_index_);
            result.timings_ms["fast_refresh_persistence_cpp_count"] = 1.0;
        } else {
            result.timings_ms["fast_refresh_persistence_cpp_count"] = 0.0;
        }
    } else if (fast.persistence_refresh_available) {
        persistent_matcher_.replaceIdentities(
            fast.persistence_refresh_identities,
            fast.persistence_refresh_frame
        );
        result.timings_ms["fast_refresh_persistence_cpp_count"] = 1.0;
    } else {
        result.timings_ms["fast_refresh_persistence_cpp_count"] = 0.0;
    }

    mode_ = TrackerMode::Tracking;
    lost_frames_ = 0;
    finalizeFrameResult(result, frame_t0);
    return true;
}

int TrackerEngine::frameIndex() const
{
    return frame_index_;
}

TrackerMode TrackerEngine::mode() const
{
    return mode_;
}

bool TrackerEngine::markerAssetsLoaded() const
{
    return !field_.empty() && !geometry_.empty();
}

const TrackerConfig& TrackerEngine::config() const
{
    return config_;
}

CheckerboardDetectorConfig TrackerEngine::makeCheckerboardConfig(
    const TrackerConfig& config
)
{
    CheckerboardDetectorConfig checker_config;
    checker_config.recovery_correction_weight = 0.5f;
    checker_config.recovery_correction_max_dist_rel = 0.6f;
    checker_config.refresh_interval_frames =
        config.checker_refresh_interval_frames;
    checker_config.tracking_recovery_stable_interval_frames =
        config.checker_tracking_recovery_stable_interval_frames;
    checker_config.tracking_recovery_zero_gain_backoff_after =
        config.checker_tracking_recovery_zero_gain_backoff_after;
    checker_config.tracking_recovery_zero_gain_backoff_max_factor =
        config.checker_tracking_recovery_zero_gain_backoff_max_factor;
    checker_config.tracking_local_completion_skip_enabled =
        config.checker_local_completion_skip_enabled;
    checker_config.tracking_local_completion_probe_interval_frames =
        config.checker_local_completion_probe_interval_frames;
    checker_config.tracking_local_completion_zero_gain_backoff_after =
        config.checker_local_completion_zero_gain_backoff_after;
    checker_config.tracking_local_completion_zero_gain_backoff_max_factor =
        config.checker_local_completion_zero_gain_backoff_max_factor;
    checker_config.tracking_local_completion_stale_predicted_frames =
        config.checker_local_completion_stale_predicted_frames;
    checker_config.min_tracking_decode_cell_span =
        config.checker_min_tracking_decode_cell_span;
    checker_config.max_undecodeable_tracking_frames =
        config.checker_max_undecodeable_tracking_frames;
    return checker_config;
}

DotDetectorConfig TrackerEngine::makeDotDetectorConfig(
    const TrackerConfig& config
)
{
    DotDetectorConfig dot_config;
    dot_config.canonical_size = config.dot_canonical_size;
    dot_config.canonical_margin_px =
        static_cast<float>(config.dot_canonical_margin_px);
    dot_config.min_dot_contrast = config.dot_min_dot_contrast;
    dot_config.strong_dot_contrast = config.dot_strong_dot_contrast;
    dot_config.commit_threshold = config.dot_commit_threshold;
    dot_config.revoke_threshold = config.dot_revoke_threshold;
    dot_config.uncertainty_low = config.dot_uncertainty_low;
    dot_config.uncertainty_high = config.dot_uncertainty_high;
    dot_config.warmup_frames = config.dot_warmup_frames;
    dot_config.temporal_alpha = config.dot_temporal_alpha;
    dot_config.commit_frames = config.dot_commit_frames;
    dot_config.revoke_frames = config.dot_revoke_frames;
    dot_config.use_temporal_smoothing = config.dot_use_temporal_smoothing;
    dot_config.use_cell_value_cache = config.dot_use_cell_value_cache;
    dot_config.cell_cache_max_age_frames = config.dot_cell_cache_max_age_frames;
    dot_config.cell_cache_max_corner_motion_px =
        static_cast<float>(config.dot_cell_cache_max_corner_motion_px);
    return dot_config;
}

PatchDecoderConfig TrackerEngine::makePatchDecoderConfig(
    const TrackerConfig& config
)
{
    PatchDecoderConfig decoder_config;
    decoder_config.require_geometry_valid =
        config.decoder_require_geometry_valid;
    decoder_config.accept_ambiguous = config.decoder_accept_ambiguous;
    return decoder_config;
}

CorrespondenceBuilderConfig TrackerEngine::makeCorrespondenceBuilderConfig(
    const TrackerConfig& config
)
{
    CorrespondenceBuilderConfig corr_config;
    corr_config.min_votes = config.corr_min_votes;
    corr_config.discard_conflicts = config.corr_discard_conflicts;
    corr_config.require_detection_stable = config.corr_require_detection_stable;
    corr_config.enable_dominant_rotation_filter =
        config.corr_enable_dominant_rotation_filter;
    corr_config.min_rotation_support = config.corr_min_rotation_support;
    corr_config.min_rotation_support_ratio =
        config.corr_min_rotation_support_ratio;
    return corr_config;
}

MapPoseTrackerConfig TrackerEngine::makeMapPoseTrackerConfig(
    const TrackerConfig& config
)
{
    MapPoseTrackerConfig pose_config;
    pose_config.min_points = config.min_points;
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

PoseDepthFilterConfig TrackerEngine::makePoseDepthFilterConfig(
    const TrackerConfig& config
)
{
    PoseDepthFilterConfig depth_config;
    depth_config.observation_std_mm =
        config.pose_depth_filter_observation_std_mm;
    depth_config.process_std_mm =
        config.pose_depth_filter_process_std_mm;
    depth_config.initial_velocity_std_mm =
        config.pose_depth_filter_initial_velocity_std_mm;
    depth_config.reprojection_guard_px =
        config.pose_depth_filter_reprojection_guard_px;
    depth_config.innovation_guard_enabled =
        config.pose_depth_filter_innovation_guard_enabled;
    depth_config.innovation_guard_window =
        config.pose_depth_filter_innovation_window;
    depth_config.innovation_guard_bias_threshold_mm =
        config.pose_depth_filter_innovation_bias_threshold_mm;
    depth_config.innovation_guard_min_same_sign =
        config.pose_depth_filter_innovation_min_same_sign;
    depth_config.innovation_cusum_slack_mm =
        config.pose_depth_filter_innovation_cusum_slack_mm;
    depth_config.innovation_cusum_threshold_mm =
        config.pose_depth_filter_innovation_cusum_threshold_mm;
    depth_config.negative_delta_guard_enabled =
        config.pose_depth_filter_negative_delta_guard_enabled;
    depth_config.negative_delta_guard_min_z_span_mm =
        config.pose_depth_filter_negative_delta_guard_min_z_span_mm;
    depth_config.negative_delta_guard_max_negative_delta_mm =
        config.pose_depth_filter_negative_delta_guard_max_negative_delta_mm;
    depth_config.negative_delta_guard_hold_previous_z =
        config.pose_depth_filter_negative_delta_guard_hold_previous_z;
    depth_config.negative_delta_guard_hold_requires_innovation_bias =
        config.pose_depth_filter_negative_delta_guard_hold_requires_innovation_bias;
    depth_config.negative_delta_guard_hold_min_negative_delta_mm =
        config.pose_depth_filter_negative_delta_guard_hold_min_negative_delta_mm;
    depth_config.negative_delta_guard_max_hold_correction_mm =
        config.pose_depth_filter_negative_delta_guard_max_hold_correction_mm;
    depth_config.negative_delta_guard_velocity_damping =
        config.pose_depth_filter_negative_delta_guard_velocity_damping;
    return depth_config;
}

PlateauPosePriorConfig TrackerEngine::makePlateauPosePriorConfig(
    const TrackerConfig& config
)
{
    PlateauPosePriorConfig prior_config;
    prior_config.static_max_excess_px =
        config.pose_plateau_prior_static_max_excess_px;
    prior_config.candidate_max_excess_px =
        config.pose_plateau_prior_candidate_max_excess_px;
    prior_config.candidate_max_max_excess_px =
        config.pose_plateau_prior_candidate_max_max_excess_px;
    prior_config.min_positive_z_correction_mm =
        config.pose_plateau_prior_min_positive_z_correction_mm;
    prior_config.max_positive_z_correction_mm =
        config.pose_plateau_prior_max_positive_z_correction_mm;
    prior_config.robust_c_px = config.pose_plateau_prior_robust_c_px;
    prior_config.max_iterations = config.pose_plateau_prior_max_iterations;
    prior_config.max_step_translation_mm =
        config.pose_plateau_prior_max_step_translation_mm;
    prior_config.max_step_rotation_deg =
        config.pose_plateau_prior_max_step_rotation_deg;
    prior_config.lm_damping = config.pose_plateau_prior_lm_damping;
    return prior_config;
}

double TrackerEngine::confidence(
    int num_inliers,
    double mean_error_px,
    const TrackerConfig& config
)
{
    const double point_score = std::min(1.0, static_cast<double>(num_inliers) / 30.0);
    double error_score = 0.0;
    if (mean_error_px >= 0.0) {
        error_score = 1.0 - std::min(
            1.0,
            mean_error_px / std::max(1.0e-6, config.max_mean_reprojection_error_px)
        );
    }
    return 0.6 * point_score + 0.4 * error_score;
}

} // namespace hydramarker
