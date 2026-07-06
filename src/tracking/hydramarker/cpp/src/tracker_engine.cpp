#include "tracker_engine.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

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
    model_ref_enrolled_ = false;
    model_ref_gray_ = cv::Mat();
    model_prev_uv_.clear();
    model_prev_xyz_.clear();
    model_curr_rvec_.clear();
    model_curr_tvec_.clear();
    model_prev_rvec_.clear();
    model_prev_tvec_.clear();
    model_enroll_count_ = 0;
    model_ref_view_angle_deg_ = 0.0;
    resetPoseWarmup();
    resetPoseKalman();
    checkerboard_detector_.resetTracking();
    dot_detector_.reset();
    pose_tracker_.reset();
    persistent_matcher_.reset();
}

TrackerFrameResult TrackerEngine::processFrame(
    const cv::Mat& frame,
    bool run_detection
)
{
    // Model-warp wiring wraps the actual frame processing: the per-frame
    // context must be on the detector before detect() runs, and the anchor
    // source / reference enrollment update after the pose was accepted.
    current_frame_ = frame;
    updateCornerModelInput();

    TrackerFrameResult result = processFrameInternal(frame, run_detection);

    result.tracked_refine_samples =
        checkerboard_detector_.lastTrackedRefineSamples();
    updatePoseWarmupState(result);
    updateModelWarpStateAfterFrame(result);
    // Output filter LAST: the model-warp anchors above must see the raw
    // pose, otherwise the filtered pose feeds back into tracking.
    applyPoseKalman(result);
    if (config_.checker_tracked_refine_method == "model_warp") {
        result.timings_ms["modelwarp_surface_valid"] =
            geometry_.surfaceModel().valid() ? 1.0 : 0.0;
        result.timings_ms["modelwarp_ref_enrolled"] =
            model_ref_enrolled_ ? 1.0 : 0.0;
        result.timings_ms["modelwarp_prev_corner_count"] =
            static_cast<double>(model_prev_uv_.size());
        result.timings_ms["modelwarp_enroll_count"] =
            static_cast<double>(model_enroll_count_);
        result.timings_ms["modelwarp_ref_view_angle_deg"] =
            model_ref_view_angle_deg_;
    }
    current_frame_ = cv::Mat();
    return result;
}

TrackerFrameResult TrackerEngine::processFrameInternal(
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
    // Pose-set stabilisation: predicted corners feed the previous pose back
    // into PnP and freshly appeared corners jump the pose along the weak
    // mode, so both are kept out of the POSE input (they still decode and
    // track). Relax automatically if filtering would starve the solver.
    std::vector<PoseTrackPoint> filtered_points;
    filtered_points.reserve(correspondences.correspondences.size());
    for (const Correspondence2D3D& corr : correspondences.correspondences) {
        pose_points.push_back(posePointFromCorrespondence(corr));
        correspondence_corners.push_back(trackerCornerFromCorrespondence(corr));
        if (posePointUsable(corr.predicted, corr.observed_frames)) {
            filtered_points.push_back(pose_points.back());
        }
    }
    if (static_cast<int>(filtered_points.size()) >= config_.min_points) {
        pose_points = std::move(filtered_points);
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

    const std::int64_t pose_t0 = cv::getTickCount();
    MapPoseResult pose = pose_tracker_.estimatePose(pose_points, lost_frames_);
    result.timings_ms["pnp_ms"] = elapsedMs(pose_t0);

    std::vector<TrackerCorner> visual_corners;
    if (pose.success) {
        pose_tracker_.setPose(pose.rvec, pose.tvec);
        visual_corners = visualCornersForPose(pose);
        // Corners withheld from the pose solve (predicted / young) must not
        // vanish from the visual set, or the persistence refresh would evict
        // them and the fast path could never match them again. Validate them
        // against the pose like every other visual corner and merge.
        std::set<std::pair<int, int>> in_visual;
        for (const TrackerCorner& c : visual_corners) {
            in_visual.insert({c.global_row, c.global_col});
        }
        std::vector<TrackerCorner> withheld;
        for (const TrackerCorner& c : correspondence_corners) {
            if (in_visual.find({c.global_row, c.global_col}) ==
                in_visual.end()) {
                withheld.push_back(c);
            }
        }
        const std::vector<TrackerCorner> validated =
            tracker_geometry_.visualCornersFromPose(
                withheld,
                pose.rvec,
                pose.tvec,
                config_.visual_corner_max_reprojection_error_px
            );
        visual_corners.insert(
            visual_corners.end(), validated.begin(), validated.end());
    }
    packagePoseResult(
        result,
        pose,
        pose.success ? PoseSource::Decode : PoseSource::None,
        pose.message,
        visual_corners,
        correspondence_corners
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
    corner.visibility_score = static_cast<double>(corr.visibility_score);
    corner.observed_frames = corr.observed_frames;
    corner.predicted = corr.predicted;
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
    out.visibility_score = static_cast<double>(corner.visibility_score);
    out.observed_frames = corner.observed_frames;
    out.predicted = corner.predicted;
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

    std::vector<TrackerCorner> packaged_visual_corners = visual_corners;
    for (TrackerCorner& visual_corner : packaged_visual_corners) {
        for (const TrackerCorner& corr_corner : correspondence_corners) {
            if (visual_corner.global_row != corr_corner.global_row ||
                visual_corner.global_col != corr_corner.global_col) {
                continue;
            }
            visual_corner.visibility_score = corr_corner.visibility_score;
            visual_corner.observed_frames = corr_corner.observed_frames;
            visual_corner.predicted = corr_corner.predicted;
            break;
        }
    }

    result.corners = packaged_visual_corners;
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

void TrackerEngine::updateCornerModelInput()
{
    if (config_.checker_tracked_refine_method != "model_warp") {
        return;
    }

    CornerModelFrameInput input;
    const SurfaceModel& surface = geometry_.surfaceModel();

    if (surface.valid() &&
        model_ref_enrolled_ &&
        !model_prev_uv_.empty() &&
        model_curr_rvec_.size() == 3 &&
        model_curr_tvec_.size() == 3) {
        // One-time pixel->ray lookup table (needs the frame size, hence
        // lazy construction on the first enabled frame).
        if (model_ray_lut_.empty() && !current_frame_.empty()) {
            const int w = current_frame_.cols;
            const int h = current_frame_.rows;
            std::vector<cv::Point2d> pix;
            pix.reserve(static_cast<size_t>(w) * h);
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    pix.emplace_back(x, y);
                }
            }
            std::vector<cv::Point2d> und;
            cv::Mat dist_mat;
            if (!dist_coeffs_.empty()) {
                dist_mat = cv::Mat(dist_coeffs_, true).reshape(1, 1);
            }
            cv::undistortPoints(pix, und, cv::Mat(K_), dist_mat);
            model_ray_lut_.create(h, w, CV_32FC2);
            size_t idx = 0;
            for (int y = 0; y < h; ++y) {
                cv::Vec2f* row = model_ray_lut_.ptr<cv::Vec2f>(y);
                for (int x = 0; x < w; ++x, ++idx) {
                    row[x] = cv::Vec2f(
                        static_cast<float>(und[idx].x),
                        static_cast<float>(und[idx].y));
                }
            }
        }

        input.enabled = true;
        input.K = K_;
        input.dist = dist_coeffs_;
        input.ray_lut = model_ray_lut_;

        // Constant-velocity pose prediction from the last two accepted
        // poses; reduces the anchor lag during fast motion.  Falls back to
        // the last pose when no history exists or the step looks abnormal.
        cv::Mat R_last;
        cv::Rodrigues(cv::Mat(model_curr_rvec_, true), R_last);
        cv::Vec3d t_last(model_curr_tvec_[0], model_curr_tvec_[1],
                         model_curr_tvec_[2]);
        cv::Matx33d R_pred(R_last.ptr<double>());
        cv::Vec3d t_pred = t_last;

        if (model_prev_rvec_.size() == 3 && model_prev_tvec_.size() == 3) {
            const cv::Vec3d t_prev(model_prev_tvec_[0], model_prev_tvec_[1],
                                   model_prev_tvec_[2]);
            const cv::Vec3d dt = t_last - t_prev;
            cv::Mat R_prev;
            cv::Rodrigues(cv::Mat(model_prev_rvec_, true), R_prev);
            const cv::Mat R_delta = R_last * R_prev.t();
            cv::Mat rvec_delta;
            cv::Rodrigues(R_delta, rvec_delta);
            const double rot_step = cv::norm(rvec_delta);

            if (cv::norm(dt) < 40.0 && rot_step < 10.0 * CV_PI / 180.0) {
                t_pred = t_last + dt;
                const cv::Mat R_p = R_delta * R_last;
                R_pred = cv::Matx33d(R_p.ptr<double>());
            }
        }

        input.R = R_pred;
        input.t = t_pred;
        input.surface = surface;
        input.ref_gray = model_ref_gray_;
        input.R_ref = model_ref_R_;
        input.t_ref = model_ref_t_;
        input.prev_uv = model_prev_uv_;
        input.prev_xyz_mm = model_prev_xyz_;
    }

    checkerboard_detector_.setCornerModelInput(input);
}

bool TrackerEngine::posePointUsable(bool predicted, int observed_frames) const
{
    if (config_.pose_exclude_predicted_corners && predicted) {
        return false;
    }
    if (config_.pose_min_observed_frames > 0 &&
        observed_frames < config_.pose_min_observed_frames) {
        return false;
    }
    return true;
}

void TrackerEngine::resetPoseWarmup()
{
    warmup_accepted_frames_ = 0;
    warmup_quiet_streak_ = 0;
    pose_converged_ = false;
}

void TrackerEngine::resetPoseKalman()
{
    pose_kf_initialized_ = false;
    pose_kf_x_ = cv::Mat();
    pose_kf_P_ = cv::Mat();
}

void TrackerEngine::applyPoseKalman(TrackerFrameResult& result)
{
    if (!config_.pose_kf_enabled) {
        return;
    }
    if (!result.current_pose_accepted ||
        result.rvec.size() != 3 ||
        result.tvec.size() != 3) {
        // No fresh measurement this frame (hold / rejection): coast the
        // state so velocities stay time-consistent, output stays untouched.
        if (pose_kf_initialized_) {
            for (int i = 0; i < 6; ++i) {
                pose_kf_x_.at<double>(i) += pose_kf_x_.at<double>(i + 6);
            }
        }
        return;
    }

    // Measured pose-set corners; their Jacobian at the measured pose gives
    // the per-frame information matrix (huge along observable directions,
    // near-singular along the weak mode).
    std::vector<cv::Point3d> object_points;
    object_points.reserve(result.corners.size());
    for (const TrackerCorner& c : result.corners) {
        if (posePointUsable(c.predicted, c.observed_frames)) {
            object_points.emplace_back(c.xyz_mm[0], c.xyz_mm[1], c.xyz_mm[2]);
        }
    }
    if (static_cast<int>(object_points.size()) < config_.min_points) {
        return;
    }

    cv::Mat z(6, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        z.at<double>(i) = result.rvec[static_cast<size_t>(i)];
        z.at<double>(i + 3) = result.tvec[static_cast<size_t>(i)];
    }

    // Large real jumps (accepted by the engine's own gates, e.g. after
    // recovery) restart the filter instead of being dragged in slowly.
    if (pose_kf_initialized_) {
        double dr = 0.0;
        double dt = 0.0;
        for (int i = 0; i < 3; ++i) {
            const double er = z.at<double>(i) - pose_kf_x_.at<double>(i);
            const double et = z.at<double>(i + 3) - pose_kf_x_.at<double>(i + 3);
            dr += er * er;
            dt += et * et;
        }
        const double reset_rot_rad =
            config_.pose_kf_reset_rotation_deg * CV_PI / 180.0;
        if (std::sqrt(dr) > reset_rot_rad ||
            std::sqrt(dt) > config_.max_translation_jump_mm) {
            pose_kf_initialized_ = false;
        }
    }

    if (!pose_kf_initialized_) {
        pose_kf_x_ = cv::Mat::zeros(12, 1, CV_64F);
        z.copyTo(pose_kf_x_.rowRange(0, 6));
        pose_kf_P_ = cv::Mat::zeros(12, 12, CV_64F);
        const double r2 = std::pow(2.0 * CV_PI / 180.0, 2.0);
        const double vr2 = std::pow(0.5 * CV_PI / 180.0, 2.0);
        for (int i = 0; i < 3; ++i) {
            pose_kf_P_.at<double>(i, i) = r2;
            pose_kf_P_.at<double>(i + 3, i + 3) = 25.0;       // (5 mm)^2
            pose_kf_P_.at<double>(i + 6, i + 6) = vr2;
            pose_kf_P_.at<double>(i + 9, i + 9) = 4.0;        // (2 mm/f)^2
        }
        pose_kf_initialized_ = true;
        result.timings_ms["pose_kf_applied"] = 0.0;
        return;  // first frame: output = measurement
    }

    // Constant-velocity predict; process noise on the acceleration.
    cv::Mat F = cv::Mat::eye(12, 12, CV_64F);
    for (int i = 0; i < 6; ++i) {
        F.at<double>(i, i + 6) = 1.0;
    }
    const double q_r =
        std::pow(config_.pose_kf_q_rotation_deg * CV_PI / 180.0, 2.0);
    const double q_t = std::pow(config_.pose_kf_q_translation_mm, 2.0);
    cv::Mat Q = cv::Mat::zeros(12, 12, CV_64F);
    for (int i = 0; i < 6; ++i) {
        const double q = (i < 3) ? q_r : q_t;
        Q.at<double>(i, i) = 0.25 * q;
        Q.at<double>(i, i + 6) = 0.5 * q;
        Q.at<double>(i + 6, i) = 0.5 * q;
        Q.at<double>(i + 6, i + 6) = q;
    }
    pose_kf_x_ = F * pose_kf_x_;
    pose_kf_P_ = F * pose_kf_P_ * F.t() + Q;

    // Measurement covariance sigma^2 (J^T J)^-1 from the pose Jacobian.
    cv::Mat rvec_m = z.rowRange(0, 3).clone();
    cv::Mat tvec_m = z.rowRange(3, 6).clone();
    cv::Mat dist_mat;
    if (!dist_coeffs_.empty()) {
        dist_mat = cv::Mat(dist_coeffs_, true);
    }
    std::vector<cv::Point2d> projected;
    cv::Mat J;
    cv::projectPoints(object_points, rvec_m, tvec_m, K_, dist_mat,
                      projected, J);
    const cv::Mat Jp = J.colRange(0, 6);  // columns: drvec(3), dtvec(3)
    const cv::Mat info = Jp.t() * Jp;
    const double sigma2 = std::pow(config_.pose_kf_sigma_px, 2.0);
    cv::Mat Sigma;
    cv::invert(info, Sigma, cv::DECOMP_SVD);
    Sigma *= sigma2;

    cv::Mat Hm = cv::Mat::zeros(6, 12, CV_64F);  // H = [I6 | 0]
    for (int i = 0; i < 6; ++i) {
        Hm.at<double>(i, i) = 1.0;
    }

    cv::Mat innov = z - Hm * pose_kf_x_;
    cv::Mat S = Hm * pose_kf_P_ * Hm.t() + Sigma;
    cv::Mat S_inv;
    cv::invert(S, S_inv, cv::DECOMP_SVD);
    const double m2 = cv::Mat(innov.t() * S_inv * innov).at<double>(0);
    if (m2 > config_.pose_kf_gate_mahalanobis &&
        config_.pose_kf_gate_mahalanobis > 0.0) {
        // Spike: deweight the measurement instead of dropping it, so a
        // genuine fast motion still pulls the state over a few frames.
        Sigma *= m2 / config_.pose_kf_gate_mahalanobis;
        S = Hm * pose_kf_P_ * Hm.t() + Sigma;
        cv::invert(S, S_inv, cv::DECOMP_SVD);
        result.timings_ms["pose_kf_gated"] = 1.0;
    }
    const cv::Mat Kg = pose_kf_P_ * Hm.t() * S_inv;
    pose_kf_x_ = pose_kf_x_ + Kg * innov;
    pose_kf_P_ = (cv::Mat::eye(12, 12, CV_64F) - Kg * Hm) * pose_kf_P_;

    // Write the filtered pose into the OUTPUT fields only.
    for (int i = 0; i < 3; ++i) {
        result.rvec[static_cast<size_t>(i)] = pose_kf_x_.at<double>(i);
        result.tvec[static_cast<size_t>(i)] = pose_kf_x_.at<double>(i + 3);
    }
    cv::Mat R;
    cv::Rodrigues(pose_kf_x_.rowRange(0, 3), R);
    if (result.T_marker_camera.size() == 16) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                result.T_marker_camera[static_cast<size_t>(r * 4 + c)] =
                    R.at<double>(r, c);
            }
            result.T_marker_camera[static_cast<size_t>(r * 4 + 3)] =
                pose_kf_x_.at<double>(r + 3);
        }
    }
    result.timings_ms["pose_kf_applied"] = 1.0;
    result.timings_ms["pose_kf_mahalanobis"] = m2;
}

void TrackerEngine::updatePoseWarmupState(TrackerFrameResult& result)
{
    if (config_.pose_warmup_min_accepted_frames <= 0) {
        pose_converged_ = true;
    } else if (!pose_converged_ && result.current_pose_accepted) {
        ++warmup_accepted_frames_;

        // "Young" corners are measured corners that only recently appeared;
        // while they keep showing up, the pose set is still saturating and
        // the pose can wander along the weak observability mode.
        const int young_threshold = std::max(3, config_.pose_min_observed_frames);
        int young = 0;
        for (const TrackerCorner& c : result.correspondence_corners) {
            if (!c.predicted && c.observed_frames < young_threshold) {
                ++young;
            }
        }
        if (young <= config_.pose_warmup_max_young_corners) {
            ++warmup_quiet_streak_;
        } else {
            warmup_quiet_streak_ = 0;
        }

        if (warmup_accepted_frames_ >= config_.pose_warmup_min_accepted_frames &&
            warmup_quiet_streak_ >= config_.pose_warmup_stable_window) {
            pose_converged_ = true;  // latched until tracking is lost
        }
    }

    result.pose_converged = pose_converged_;
    result.timings_ms["pose_converged"] = pose_converged_ ? 1.0 : 0.0;
}

void TrackerEngine::updateModelWarpStateAfterFrame(
    const TrackerFrameResult& result
)
{
    if (config_.checker_tracked_refine_method != "model_warp") {
        return;
    }

    if (!result.current_pose_accepted ||
        result.rvec.size() != 3 ||
        result.tvec.size() != 3) {
        return;
    }

    // Anchor source for the next frame: pose corners of this (now last
    // accepted) frame — their uv in this frame plus the marker coordinate.
    model_prev_uv_.clear();
    model_prev_xyz_.clear();
    model_prev_uv_.reserve(result.corners.size());
    model_prev_xyz_.reserve(result.corners.size());
    for (const TrackerCorner& c : result.corners) {
        if (c.predicted) {
            continue;
        }
        model_prev_uv_.emplace_back(
            static_cast<float>(c.uv[0]),
            static_cast<float>(c.uv[1]));
        model_prev_xyz_.emplace_back(
            c.xyz_mm[0], c.xyz_mm[1], c.xyz_mm[2]);
    }

    // Pose history for the constant-velocity prediction.
    model_prev_rvec_ = model_curr_rvec_;
    model_prev_tvec_ = model_curr_tvec_;
    model_curr_rvec_ = result.rvec;
    model_curr_tvec_ = result.tvec;

    // Re-enrollment trigger: drop the reference once the viewing direction
    // (camera centre seen from the marker) moved beyond the threshold —
    // the warp bias grows with the angle offset to the enrollment view.
    // The enrollment block below then re-enrolls at the next quiet,
    // high-quality frame; corners fall back to subpix in between.
    model_ref_view_angle_deg_ = 0.0;
    if (model_ref_enrolled_) {
        cv::Mat R_now_m;
        cv::Rodrigues(cv::Mat(result.rvec, true), R_now_m);
        const cv::Matx33d R_now(R_now_m.ptr<double>());
        const cv::Vec3d t_now(result.tvec[0], result.tvec[1],
                              result.tvec[2]);
        const cv::Vec3d c_now = -(R_now.t() * t_now);
        const cv::Vec3d c_ref = -(model_ref_R_.t() * model_ref_t_);
        const double n_now = cv::norm(c_now);
        const double n_ref = cv::norm(c_ref);
        if (n_now > 1e-6 && n_ref > 1e-6) {
            double cosang = c_now.dot(c_ref) / (n_now * n_ref);
            cosang = std::max(-1.0, std::min(1.0, cosang));
            model_ref_view_angle_deg_ =
                std::acos(cosang) * 180.0 / CV_PI;
        }
        if (config_.model_warp_reenroll_angle_deg > 0.0 &&
            model_ref_view_angle_deg_ >
                config_.model_warp_reenroll_angle_deg) {
            model_ref_enrolled_ = false;
        }
    }

    // Reference enrollment on a stable accepted pose AFTER pose warmup
    // (poses during set saturation wander along the weak mode, and a
    // reference enrolled from such a pose bakes that error in) and while
    // the tool is QUIET (a blurred reference degrades every later
    // measurement). Runs again whenever the re-enrollment policy dropped
    // the reference; short tracking losses keep it, a full loss drops it.
    if (!model_ref_enrolled_ &&
        pose_converged_ &&
        poseMotionQuiet() &&
        geometry_.surfaceModel().valid() &&
        !current_frame_.empty() &&
        result.num_inliers >= 15 &&
        result.mean_reprojection_error_px >= 0.0 &&
        result.mean_reprojection_error_px <= 1.5) {
        cv::Mat gray;
        if (current_frame_.channels() == 3) {
            cv::cvtColor(current_frame_, gray, cv::COLOR_BGR2GRAY);
        } else if (current_frame_.channels() == 4) {
            cv::cvtColor(current_frame_, gray, cv::COLOR_BGRA2GRAY);
        } else {
            gray = current_frame_;
        }
        // Stored as CV_32F so the refiner never converts the reference
        // again (it is sampled every frame).
        gray.convertTo(model_ref_gray_, CV_32F);

        cv::Mat R;
        cv::Rodrigues(cv::Mat(result.rvec, true), R);
        model_ref_R_ = cv::Matx33d(R.ptr<double>());
        model_ref_t_ = cv::Vec3d(
            result.tvec[0], result.tvec[1], result.tvec[2]);
        model_ref_enrolled_ = true;
        ++model_enroll_count_;
    }
}

bool TrackerEngine::poseMotionQuiet() const
{
    if (model_prev_rvec_.size() != 3 || model_prev_tvec_.size() != 3 ||
        model_curr_rvec_.size() != 3 || model_curr_tvec_.size() != 3) {
        return true;  // no pose history yet — nothing to compare against
    }
    double dt2 = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double d = model_curr_tvec_[static_cast<size_t>(i)] -
                         model_prev_tvec_[static_cast<size_t>(i)];
        dt2 += d * d;
    }
    if (std::sqrt(dt2) > config_.model_warp_enroll_max_motion_mm) {
        return false;
    }
    cv::Mat R_prev, R_curr, r_delta;
    cv::Rodrigues(cv::Mat(model_prev_rvec_, true), R_prev);
    cv::Rodrigues(cv::Mat(model_curr_rvec_, true), R_curr);
    cv::Rodrigues(R_curr * R_prev.t(), r_delta);
    const double rot_deg = cv::norm(r_delta) * 180.0 / CV_PI;
    return rot_deg <= config_.model_warp_enroll_max_rotation_deg;
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
    pose_tracker_.setPose(pose.rvec, pose.tvec);
    std::vector<TrackerCorner> visual_corners = visualCornersForPose(pose);
    packagePoseResult(
        result,
        pose,
        PoseSource::Persistent,
        "Pose estimated from persistent correspondences after: " + reason + ".",
        visual_corners,
        seed.corners
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
        dot_detector_.reset();
        persistent_matcher_.clearIdentities();
        // Full loss: the corner set will be rebuilt from scratch, so the
        // pose has to warm up again before it counts as converged.
        resetPoseWarmup();
        resetPoseKalman();
        // Re-acquisition may happen in a different tool orientation; a
        // reference from before the loss is then stale. Drop it — a fresh
        // one is enrolled after the pose has re-converged.
        if (config_.model_warp_reenroll_on_loss) {
            model_ref_enrolled_ = false;
        }
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

    MapPoseResult pose = fast.pose;
    if (
        !pose.success ||
        pose.rvec.size() != 3 ||
        pose.tvec.size() != 3
    ) {
        return false;
    }

    std::vector<TrackerCorner> visual_corners = fast.visual_corners;

    pose_tracker_.setPose(pose.rvec, pose.tvec);
    packagePoseResult(
        result,
        pose,
        PoseSource::FastPersistent,
        "Fast pose estimated from persistent correspondences.",
        visual_corners,
        fast.corners
    );

    result.current_pose_accepted =
        acceptPoseState(pose, static_cast<int>(visual_corners.size()));
    if (fast.persistence_refresh_available) {
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
    checker_config.tracked_refine_method =
        config.checker_tracked_refine_method;
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
