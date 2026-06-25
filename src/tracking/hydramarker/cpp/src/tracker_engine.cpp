#include "tracker_engine.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hydramarker {

namespace {

double elapsedMs(std::int64_t start_tick)
{
    return (static_cast<double>(cv::getTickCount() - start_tick)
            / cv::getTickFrequency()) * 1000.0;
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
      pose_tracker_(K, dist_coeffs, makeMapPoseTrackerConfig(config))
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
    checkerboard_detector_.resetTracking();
    dot_detector_.reset();
    pose_tracker_.reset();
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
        result.timings_ms["tracker_total_ms"] = elapsedMs(frame_t0);
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
        ++lost_frames_;
        if (lost_frames_ > config_.max_lost_frames) {
            mode_ = TrackerMode::Lost;
            pose_tracker_.reset();
            dot_detector_.reset();
        }
        result.lost_frames = lost_frames_;
        result.mode = mode_;
        result.message = "No valid checkerboard detection.";
        result.timings_ms["tracker_total_ms"] = elapsedMs(frame_t0);
        return result;
    }

    lost_frames_ = 0;

    result.detection_valid = true;
    result.detection_tracking = detection->tracking;
    result.detection_stable = detection->stable;
    result.detection_corner_count = static_cast<int>(detection->corners.size());
    result.detection_cell_count = static_cast<int>(detection->cells.size());

    const std::int64_t dot_t0 = cv::getTickCount();
    DotDetectionResult dots = dot_detector_.detect(frame, *detection);
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
        mode_ = TrackerMode::Detecting;
        result.mode = mode_;
        result.message = "No valid decoded patches.";
        result.timings_ms["tracker_total_ms"] = elapsedMs(frame_t0);
        return result;
    }

    const std::int64_t corr_t0 = cv::getTickCount();
    CorrespondenceBuildResult correspondences = correspondence_builder_.build(
        *detection,
        decoded_valid,
        geometry_
    );
    result.timings_ms["correspondence_build_ms"] = elapsedMs(corr_t0);
    result.correspondence_count = static_cast<int>(
        correspondences.correspondences.size()
    );

    if (!correspondences.valid()) {
        mode_ = TrackerMode::Detecting;
        result.mode = mode_;
        result.message = "Correspondence build failed.";
        result.timings_ms["tracker_total_ms"] = elapsedMs(frame_t0);
        return result;
    }

    std::vector<PoseTrackPoint> pose_points;
    pose_points.reserve(correspondences.correspondences.size());
    for (const Correspondence2D3D& corr : correspondences.correspondences) {
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
        pose_points.push_back(point);
    }

    const std::int64_t pose_t0 = cv::getTickCount();
    MapPoseResult pose = pose_tracker_.estimatePose(pose_points, lost_frames_);
    result.timings_ms["pnp_ms"] = elapsedMs(pose_t0);

    result.success = pose.success;
    result.message = pose.message;
    result.pose_source = pose.success ? PoseSource::Decode : PoseSource::None;
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

    mode_ = pose.success ? TrackerMode::Tracking : TrackerMode::Detecting;
    result.mode = mode_;
    result.timings_ms["tracker_total_ms"] = elapsedMs(frame_t0);
    return result;
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
    checker_config.tracking_local_completion_skip_enabled =
        config.checker_local_completion_skip_enabled;
    checker_config.tracking_local_completion_probe_interval_frames =
        config.checker_local_completion_probe_interval_frames;
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
