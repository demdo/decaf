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

PoseOutputFilterConfig makePoseOutputFilterConfig(const TrackerConfig& config)
{
    PoseOutputFilterConfig cfg;
    cfg.enabled = config.pose_kf_enabled;
    cfg.sigma_px = config.pose_kf_sigma_px;
    cfg.q_translation_mm = config.pose_kf_q_translation_mm;
    cfg.q_rotation_deg = config.pose_kf_q_rotation_deg;
    cfg.adaptive = config.pose_kf_adaptive;
    cfg.q_rotation_floor_deg = config.pose_kf_q_rotation_floor_deg;
    cfg.q_translation_floor_mm = config.pose_kf_q_translation_floor_mm;
    cfg.adaptive_ema_alpha = config.pose_kf_adaptive_ema_alpha;
    cfg.auto_noise = config.pose_kf_auto_noise;
    cfg.noise_window = config.pose_kf_noise_window;
    cfg.motion_floor_rot_deg = config.pose_kf_motion_floor_rot_deg;
    cfg.motion_floor_trans_mm = config.pose_kf_motion_floor_trans_mm;
    cfg.meas_floor_rot_deg = config.pose_kf_meas_floor_rot_deg;
    cfg.meas_floor_trans_mm = config.pose_kf_meas_floor_trans_mm;
    cfg.gate_mahalanobis = config.pose_kf_gate_mahalanobis;
    cfg.reset_rotation_deg = config.pose_kf_reset_rotation_deg;
    cfg.max_translation_jump_mm = config.max_translation_jump_mm;
    cfg.min_points = config.min_points;
    return cfg;
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
    pose_output_filter_.configure(
        makePoseOutputFilterConfig(config_), K_, dist_coeffs_);
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
    model_prev_uv_.clear();
    model_prev_xyz_.clear();
    model_curr_rvec_.clear();
    model_curr_tvec_.clear();
    model_prev_rvec_.clear();
    model_prev_tvec_.clear();
    resetPoseWarmup();
    resetPoseFilter();
    ir_pose_refiner_.reset();
    checkerboard_detector_.resetTracking();
    dot_detector_.reset();
    pose_tracker_.reset();
    persistent_matcher_.reset();
}

TrackerFrameResult TrackerEngine::processFrame(
    const cv::Mat& frame,
    const cv::Mat& ir_left,
    const cv::Mat& ir_right,
    bool run_detection
)
{
    current_ir_left_ = ir_left;
    current_ir_right_ = ir_right;
    TrackerFrameResult result = processFrame(frame, run_detection);
    current_ir_left_ = cv::Mat();
    current_ir_right_ = cv::Mat();
    return result;
}

void TrackerEngine::setIrCalibration(
    const cv::Matx33d& K_rgb,
    const std::vector<double>& dist_rgb,
    const cv::Matx33d& K_left,
    const std::vector<double>& dist_left,
    const cv::Matx33d& K_right,
    const std::vector<double>& dist_right,
    const cv::Matx33d& R_rgb_left,
    const cv::Vec3d& t_rgb_left_mm,
    const cv::Matx33d& R_left_right,
    const cv::Vec3d& t_left_right_mm
)
{
    IrCameraCalibration calib;
    calib.K_rgb = K_rgb;
    calib.dist_rgb = dist_rgb;
    calib.K_left = K_left;
    calib.dist_left = dist_left;
    calib.K_right = K_right;
    calib.dist_right = dist_right;
    calib.R_rgb_left = R_rgb_left;
    calib.t_rgb_left_mm = t_rgb_left_mm;
    calib.R_left_right = R_left_right;
    calib.t_left_right_mm = t_left_right_mm;
    calib.valid = true;

    IrPoseRefinerConfig ir_cfg;
    ir_cfg.enabled = config_.ir_refine_enabled;
    ir_cfg.min_pairs = config_.ir_min_pairs;
    ir_cfg.min_ref_rot_deg = config_.ir_min_ref_rot_deg;
    ir_cfg.max_ref_rot_deg = config_.ir_max_ref_rot_deg;
    ir_cfg.fallback_min_ref_rot_deg = config_.ir_fallback_min_ref_rot_deg;
    ir_cfg.sat_threshold = config_.ir_sat_threshold;
    ir_cfg.sat_half_px = config_.ir_sat_half_px;
    ir_cfg.epipolar_max_dv_px = config_.ir_epipolar_max_dv_px;
    ir_cfg.zncc_weight_floor = config_.ir_zncc_weight_floor;
    ir_cfg.dtz_clamp_mm = config_.ir_dtz_clamp_mm;
    ir_cfg.depth_scale = config_.ir_depth_scale;
    ir_cfg.sigma_px = config_.ir_sigma_px;
    ir_cfg.sigma_ir_px = config_.ir_sigma_ir_px;
    ir_cfg.w3d = config_.ir_w3d;
    ir_cfg.fit_gate_rms_mm = config_.ir_fit_gate_rms_mm;
    ir_cfg.fit_gate_max_trans_jump_mm = config_.ir_fit_gate_max_trans_jump_mm;
    ir_cfg.mw_min_zncc = config_.ir_mw_min_zncc;
    ir_cfg.mw_max_shift_px = config_.ir_mw_max_shift_px;
    ir_cfg.ref_tile_deg = config_.ir_ref_tile_deg;
    ir_cfg.ref_tile_trans_mm = config_.ir_ref_tile_trans_mm;
    ir_cfg.fallback_min_ref_trans_mm = config_.ir_fallback_min_ref_trans_mm;
    // NOTE: ir_cfg.enroll_max_sat_frac deliberately stays at its own default
    // (0.50, template-usability ceiling) and is NOT tied to
    // ir_enroll_max_sat_frac (0.35, the engine's measurement-quality gate for
    // position tiles): coupling them starved the IR path when a run started
    // inside a mildly specular zone (sat 0.38 -> no reference for 500 frames).
    ir_cfg.max_references = config_.ir_max_references;
    ir_cfg.corner_method = config_.ir_corner_method;
    ir_cfg.qf_max_dev_px = config_.ir_qf_max_dev_px;

    // Full corner cloud: the fusion computes its own visibility set.
    std::vector<cv::Vec3d> cloud;
    for (int row = 0; row < geometry_.cornerRows(); ++row) {
        for (int col = 0; col < geometry_.cornerCols(); ++col) {
            if (geometry_.hasCorner(row, col)) {
                const cv::Point3f p = geometry_.cornerPoint(row, col);
                cloud.emplace_back(p.x, p.y, p.z);
            }
        }
    }
    ir_pose_refiner_.configure(ir_cfg, geometry_.surfaceModel(), cloud);
    ir_pose_refiner_.setCalibration(calib);

    // Full marker-pattern LUT for the synthetic-template registration:
    // corner-node grids in surface coordinates (same decomposition the
    // saddle warp uses: u = w.dir, theta = atan2(w.(dir x radial_ref),
    // w.radial_ref)) + the HydraMarker field bits.  Conventions validated
    // against a real frame (corr +0.73 / counter-hypothesis -0.73).
    {
        MarkerPatternLut lut;
        const SurfaceModel& sm2 = geometry_.surfaceModel();
        const int R = geometry_.cornerRows();
        const int C = geometry_.cornerCols();
        if (sm2.valid() && !field_.empty() &&
            field_.height() == R - 1 && field_.width() == C - 1) {
            const bool cyl2 = sm2.type == SurfaceModel::Type::Cylinder;
            const cv::Vec3d e2c = sm2.dir.cross(sm2.radial_ref);
            std::vector<double> u_nodes(static_cast<size_t>(R));
            std::vector<double> s_nodes(static_cast<size_t>(C));
            std::vector<double> tmp;
            bool nodes_ok = true;
            for (int r = 0; r < R && nodes_ok; ++r) {
                tmp.clear();
                for (int c = 0; c < C; ++c) {
                    if (!geometry_.hasCorner(r, c)) continue;
                    const cv::Point3f p = geometry_.cornerPoint(r, c);
                    const cv::Vec3d w =
                        cv::Vec3d(p.x, p.y, p.z) - sm2.point;
                    tmp.push_back(cyl2 ? w.dot(sm2.dir)
                                       : w.dot(sm2.basis_u));
                }
                if (tmp.empty()) { nodes_ok = false; break; }
                std::nth_element(tmp.begin(),
                                 tmp.begin() + tmp.size() / 2, tmp.end());
                u_nodes[static_cast<size_t>(r)] = tmp[tmp.size() / 2];
            }
            for (int c = 0; c < C && nodes_ok; ++c) {
                tmp.clear();
                for (int r = 0; r < R; ++r) {
                    if (!geometry_.hasCorner(r, c)) continue;
                    const cv::Point3f p = geometry_.cornerPoint(r, c);
                    const cv::Vec3d w =
                        cv::Vec3d(p.x, p.y, p.z) - sm2.point;
                    if (cyl2) {
                        const double u = w.dot(sm2.dir);
                        const cv::Vec3d rv = w - u * sm2.dir;
                        tmp.push_back(
                            std::atan2(rv.dot(e2c), rv.dot(sm2.radial_ref)) *
                            sm2.radius_mm);
                    } else {
                        tmp.push_back(w.dot(sm2.basis_v));
                    }
                }
                if (tmp.empty()) { nodes_ok = false; break; }
                std::nth_element(tmp.begin(),
                                 tmp.begin() + tmp.size() / 2, tmp.end());
                s_nodes[static_cast<size_t>(c)] = tmp[tmp.size() / 2];
            }
            const bool u_asc =
                nodes_ok && std::is_sorted(u_nodes.begin(), u_nodes.end());
            if (nodes_ok && u_asc) {
                const double s_sign =
                    s_nodes.back() > s_nodes.front() ? 1.0 : -1.0;
                std::vector<double> sn(s_nodes.size());
                for (size_t i = 0; i < s_nodes.size(); ++i) {
                    sn[i] = s_sign * s_nodes[i];
                }
                if (std::is_sorted(sn.begin(), sn.end())) {
                    lut.u_nodes = u_nodes;
                    lut.s_neg_nodes = sn;
                    lut.s_sign = s_sign;
                    lut.cell_rows = R - 1;
                    lut.cell_cols = C - 1;
                    lut.bits.resize(
                        static_cast<size_t>((R - 1) * (C - 1)));
                    for (int r = 0; r < R - 1; ++r) {
                        for (int c = 0; c < C - 1; ++c) {
                            lut.bits[static_cast<size_t>(
                                r * (C - 1) + c)] = field_.at(c, r);
                        }
                    }
                }
            }
        }
        ir_pose_refiner_.setPattern(lut);
    }
}

void TrackerEngine::resetIrReferences()
{
    ir_pose_refiner_.reset();
    ir_meas_cov_valid_ = false;
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
    // Last accepted frame-to-frame pose rotation for the detector's lean
    // refresh cadence (translation-like motion skips the aggressive weak-state
    // maintenance; rotation keeps it). Conservative large default while no
    // pose pair exists yet.
    {
        double drot_deg = 1e9;
        if (model_prev_rvec_.size() == 3 && model_curr_rvec_.size() == 3) {
            cv::Mat R_prev, R_curr, r_delta;
            cv::Rodrigues(cv::Mat(model_prev_rvec_, true), R_prev);
            cv::Rodrigues(cv::Mat(model_curr_rvec_, true), R_curr);
            cv::Rodrigues(R_curr * R_prev.t(), r_delta);
            drot_deg = cv::norm(r_delta) * 180.0 / CV_PI;
        }
        checkerboard_detector_.setInterFrameRotationDeg(drot_deg);
    }

    TrackerFrameResult result = processFrameInternal(frame, run_detection);

    result.tracked_refine_samples =
        checkerboard_detector_.lastTrackedRefineSamples();
    updatePoseWarmupState(result);
    updateModelWarpStateAfterFrame(result);
    // IR refinement FIRST in the output chain: it replaces the reported pose
    // with the global-shutter measurement; the anchor/filter stages then act
    // on that pose. The internal chain keeps the raw RGB pose.
    applyIrRefinement(result);
    // Output filter LAST: the model-warp anchors above must see the raw
    // pose, otherwise the filtered pose feeds back into tracking.
    applyPoseFilter(result);
    if (config_.checker_tracked_refine_method == "quadratic_form") {
        result.timings_ms["qf_surface_valid"] =
            geometry_.surfaceModel().valid() ? 1.0 : 0.0;
        result.timings_ms["qf_prev_corner_count"] =
            static_cast<double>(model_prev_uv_.size());
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

    // quadratic_form anchor source: fresh per-frame correspondences (uv +
    // marker xyz via grid identity), captured BEFORE the pose decision so
    // the anchors stay current through pose-rejected streaks.  Frames
    // without a correspondence build (holds) keep the last capture — the
    // anchor ages by one frame per hold, exactly the accepted-only
    // degradation, never worse.
    if (config_.checker_tracked_refine_method == "quadratic_form") {
        qf_frame_anchor_uv_.clear();
        qf_frame_anchor_xyz_.clear();
        for (const Correspondence2D3D& c :
             correspondences.correspondences) {
            if (c.predicted) {
                continue;
            }
            qf_frame_anchor_uv_.push_back(c.uv);
            qf_frame_anchor_xyz_.emplace_back(
                c.xyz_mm.x, c.xyz_mm.y, c.xyz_mm.z);
        }
    }

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
    const bool is_qf =
        config_.checker_tracked_refine_method == "quadratic_form";
    if (!is_qf) {
        return;
    }

    CornerModelFrameInput input;
    const SurfaceModel& surface = geometry_.surfaceModel();

    // quadratic_form is reference-free: it needs no enrolled template, only
    // the surface model, a pose prediction and the previous-frame anchors.
    if (surface.valid() &&
        is_qf &&
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

void TrackerEngine::resetPoseFilter()
{
    pose_output_filter_.reset();
}

void TrackerEngine::applyPoseFilter(TrackerFrameResult& result)
{
    if (!config_.pose_kf_enabled) {
        return;
    }
    if (!result.current_pose_accepted ||
        result.rvec.size() != 3 ||
        result.tvec.size() != 3) {
        // No fresh measurement this frame (hold / rejection): coast the
        // state so velocities stay time-consistent, output stays untouched.
        pose_output_filter_.coast();
        return;
    }

    // Measured pose-set corners; their Jacobian at the measured pose gives
    // the per-frame information matrix (huge along observable directions,
    // near-singular along the weak mode). Tip-agnostic: the filter only ever
    // sees the generic pose and this corner geometry.
    std::vector<cv::Point3d> object_points;
    object_points.reserve(result.corners.size());
    for (const TrackerCorner& c : result.corners) {
        if (posePointUsable(c.predicted, c.observed_frames)) {
            object_points.emplace_back(c.xyz_mm[0], c.xyz_mm[1], c.xyz_mm[2]);
        }
    }

    const std::array<double, 3> rvec_in = {
        result.rvec[0], result.rvec[1], result.rvec[2]};
    const std::array<double, 3> tvec_in = {
        result.tvec[0], result.tvec[1], result.tvec[2]};
    // When the IR MAP fusion applied this frame, smooth its IR-informed pose
    // with the MAP measurement covariance instead of the RGB-only Sigma.
    // When the fusion ATTEMPTED and REJECTED (references existed, gates
    // failed: glare veil, adverse geometry), the surviving RGB-only pose is
    // known-degraded evidence — inflate its covariance so the filter damps
    // the episode instead of following the drift at full confidence.
    const double meas_var_scale =
        (ir_last_fusion_rejected_ && config_.ir_reject_cov_inflate > 1.0)
            ? config_.ir_reject_cov_inflate
            : 1.0;
    result.timings_ms["pose_kf_meas_var_scale"] = meas_var_scale;
    const PoseOutputFilterResult filtered = pose_output_filter_.update(
        rvec_in, tvec_in, object_points,
        ir_meas_cov_valid_ ? &ir_meas_cov_ : nullptr,
        meas_var_scale);
    if (!filtered.applied) {
        return;  // too few points: leave the raw pose as reported
    }

    // Write the filtered pose into the OUTPUT fields only.
    for (int i = 0; i < 3; ++i) {
        result.rvec[static_cast<size_t>(i)] = filtered.rvec[static_cast<size_t>(i)];
        result.tvec[static_cast<size_t>(i)] = filtered.tvec[static_cast<size_t>(i)];
    }
    cv::Mat rvec_mat(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        rvec_mat.at<double>(i) = filtered.rvec[static_cast<size_t>(i)];
    }
    cv::Mat R;
    cv::Rodrigues(rvec_mat, R);
    if (result.T_marker_camera.size() == 16) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                result.T_marker_camera[static_cast<size_t>(r * 4 + c)] =
                    R.at<double>(r, c);
            }
            result.T_marker_camera[static_cast<size_t>(r * 4 + 3)] =
                filtered.tvec[static_cast<size_t>(r)];
        }
    }
    result.pose_covariance.assign(
        filtered.covariance.begin(), filtered.covariance.end());
    result.timings_ms["pose_kf_applied"] =
        filtered.initialized_this_frame ? 0.0 : 1.0;
    result.timings_ms["pose_kf_mahalanobis"] = filtered.mahalanobis;
    if (filtered.gated) {
        result.timings_ms["pose_kf_gated"] = 1.0;
    }
}

void TrackerEngine::applyIrRefinement(TrackerFrameResult& result)
{
    ir_meas_cov_valid_ = false;   // reset each frame; set below iff MAP applied
    ir_last_fusion_rejected_ = false;
    if (!ir_pose_refiner_.active()) {
        return;
    }
    if (current_ir_left_.empty() || current_ir_right_.empty()) {
        return;
    }
    if (!result.current_pose_accepted ||
        result.rvec.size() != 3 || result.tvec.size() != 3) {
        return;
    }

    const std::array<double, 3> rvec_in = {
        result.rvec[0], result.rvec[1], result.rvec[2]};
    const std::array<double, 3> tvec_in = {
        result.tvec[0], result.tvec[1], result.tvec[2]};
    // The MAP fusion reprojects the SAME tracked corners the RGB pose was solved
    // on (model position + detected pixel) and fuses their IR triangulation.
    std::vector<cv::Vec3d> rgb_xyz;
    std::vector<cv::Point2d> rgb_uv;
    rgb_xyz.reserve(result.corners.size());
    rgb_uv.reserve(result.corners.size());
    for (const TrackerCorner& c : result.corners) {
        rgb_xyz.emplace_back(c.xyz_mm[0], c.xyz_mm[1], c.xyz_mm[2]);
        rgb_uv.emplace_back(c.uv[0], c.uv[1]);
    }
    // Rolling-shutter-aware RGB sigma. The RGB camera is rolling shutter, the
    // IR pair is global shutter: fast in-image motion shears the marker
    // COHERENTLY, which PnP absorbs as a fake roll about the view axis (live
    // fb runs: roll delta proportional to push velocity, corr +0.6..+0.8) -
    // over the ~180 mm tool lever that is millimetres of fake tip motion. The
    // shear is invisible to the IID still-frame sigma (the reprojection stays
    // consistent), so inflate sigma_px with the measured per-frame corner
    // velocity: at rest the calibrated weighting, under motion the orientation
    // authority moves to the shear-free IR measurement.
    double vpx = ir_last_vpx_;
    {
        std::vector<double> dmov;
        std::unordered_map<int, cv::Point2d> cur;
        cur.reserve(result.corners.size());
        for (const TrackerCorner& c : result.corners) {
            if (c.predicted) {
                continue;
            }
            const int key = c.global_row * 1024 + c.global_col;
            const cv::Point2d uv(c.uv[0], c.uv[1]);
            cur.emplace(key, uv);
            const auto it = ir_prev_corner_uv_.find(key);
            if (it != ir_prev_corner_uv_.end()) {
                dmov.push_back(cv::norm(uv - it->second));
            }
        }
        if (dmov.size() >= 8) {
            std::nth_element(dmov.begin(),
                             dmov.begin() +
                                 static_cast<std::ptrdiff_t>(dmov.size() / 2),
                             dmov.end());
            vpx = dmov[dmov.size() / 2];
            ir_last_vpx_ = vpx;    // few matches -> keep the last estimate
        }
        ir_prev_corner_uv_ = std::move(cur);
    }
    // Rotation gate on the shear compensation: image velocity from
    // TRANSLATION is the rolling-shutter case (RGB corrupt, global-shutter IR
    // clean -> shift authority). Image velocity from fast ROTATION (divot
    // sweep) stresses the IR warp itself - shifting authority there made the
    // divot z_perp WORSE (0.48 -> 0.72). Fade the inflation out with the
    // per-frame rotation delta so it only acts on translation-like motion.
    double drot_deg = 0.0;
    if (ir_prev_rvec_valid_) {
        const cv::Mat rv_prev =
            (cv::Mat_<double>(3, 1) << ir_prev_rvec_[0], ir_prev_rvec_[1],
             ir_prev_rvec_[2]);
        const cv::Mat rv_curr =
            (cv::Mat_<double>(3, 1) << result.rvec[0], result.rvec[1],
             result.rvec[2]);
        cv::Mat R_prev, R_curr, r_delta;
        cv::Rodrigues(rv_prev, R_prev);
        cv::Rodrigues(rv_curr, R_curr);
        cv::Rodrigues(R_curr * R_prev.t(), r_delta);
        drot_deg = cv::norm(r_delta) * 180.0 / CV_PI;
    }
    ir_prev_rvec_ = {result.rvec[0], result.rvec[1], result.rvec[2]};
    ir_prev_rvec_valid_ = true;
    double sigma_px_eff = config_.ir_sigma_px;
    if (config_.ir_sigma_px_motion_coeff > 0.0) {
        // Only the TRANSLATION share of the image velocity drives the shear
        // compensation: subtract the rotation-induced share (~rot_px_per_deg
        // px per deg/frame of pose rotation). A pure push keeps nearly the
        // full vpx; an orientation sweep (divot) cancels it, because there
        // the IR warp is stressed too and must keep its normal weight.
        const double v_trans = std::max(
            0.0, vpx - config_.ir_sigma_px_motion_rot_px_per_deg * drot_deg);
        const double sm = config_.ir_sigma_px_motion_coeff * v_trans;
        sigma_px_eff = std::sqrt(sigma_px_eff * sigma_px_eff + sm * sm);
    }
    result.timings_ms["ir_vpx"] = vpx;
    result.timings_ms["ir_drot_deg"] = drot_deg;
    result.timings_ms["ir_sigma_px_eff"] = sigma_px_eff;
    const std::int64_t ir_t0 = cv::getTickCount();
    const IrPoseRefinerResult fused = ir_pose_refiner_.fuse(
        current_ir_left_, current_ir_right_, rvec_in, tvec_in, rgb_xyz, rgb_uv,
        sigma_px_eff);
    result.timings_ms["ir_refine_ms"] = elapsedMs(ir_t0);
    result.timings_ms["ir_fit_rms_mm"] = fused.fit_rms_mm;
    result.timings_ms["ir_applied"] = fused.applied ? 1.0 : 0.0;
    result.timings_ms["ir_mode"] = static_cast<double>(fused.mode);
    result.timings_ms["ir_ref_count"] =
        static_cast<double>(fused.ref_count);
    result.timings_ms["ir_refs_measured"] =
        static_cast<double>(fused.refs_measured);
    result.timings_ms["ir_ref_enrolled"] = fused.ref_count > 0 ? 1.0 : 0.0;
    result.timings_ms["ir_pairs"] = static_cast<double>(fused.pairs);
    result.timings_ms["ir_saturated_frac"] = fused.saturated_frac;
    result.timings_ms["ir_best_ref_angle_deg"] = fused.best_ref_angle_deg;
    result.timings_ms["ir_best_ref_trans_mm"] = fused.best_ref_trans_mm;
    result.timings_ms["ir_quality"] = fused.quality;
    result.timings_ms["ir_dtz_mm"] = fused.dtz_mm;   // legacy: depth part of shift
    result.timings_ms["ir_delta_trans_mm"] = 0.0;
    // The MAP corrects the FULL pose; the depth-only path never touched rotation,
    // so ir_delta_rot_deg (how far the fusion rotated the pose) is a new signal.
    result.timings_ms["ir_delta_rot_deg"] = 0.0;

    // Remember the RGB-only pose before the fusion overwrites rvec/tvec. The tip
    // blend downstream keeps this pose's image-plane and takes only the fused
    // depth (see TrackerFrameResult::rvec_prefusion).
    result.rvec_prefusion = result.rvec;
    result.tvec_prefusion = result.tvec;

    if (fused.applied) {
        // MAP fusion: the tightly-coupled RGB-reprojection + IR-stereo-3D solve
        // replaces the reported pose (rotation + translation), covariance-
        // weighted so IR only moves the DOF it observes well.
        const cv::Vec3d dt(fused.tvec[0] - result.tvec[0],
                           fused.tvec[1] - result.tvec[1],
                           fused.tvec[2] - result.tvec[2]);
        result.timings_ms["ir_delta_trans_mm"] = cv::norm(dt);
        const cv::Mat rvec_rgb_m =
            (cv::Mat_<double>(3, 1) << rvec_in[0], rvec_in[1], rvec_in[2]);
        const cv::Mat rvec_fus_m =
            (cv::Mat_<double>(3, 1) << fused.rvec[0], fused.rvec[1], fused.rvec[2]);
        cv::Mat R_rgb_m, R_fus_m, r_delta;
        cv::Rodrigues(rvec_rgb_m, R_rgb_m);
        cv::Rodrigues(rvec_fus_m, R_fus_m);
        cv::Rodrigues(R_fus_m * R_rgb_m.t(), r_delta);
        result.timings_ms["ir_delta_rot_deg"] = cv::norm(r_delta) * 180.0 / CV_PI;
        // Weak-evidence honesty: the Huber-trimmed GN understates uncertainty
        // when the fit is poor (fb3 frame 4057: 23 pairs, fit_rms 0.42 vs
        // typical 0.09, yet near-nominal covariance -> +3.25mm tip spike passed
        // the filter). Scale by the reduced chi-square so the temporal filter
        // damps such frames instead of following them.
        double cov_scale = 1.0;
        if (config_.ir_cov_inflate_ref_rms_mm > 0.0 && fused.fit_rms_mm > 0.0) {
            const double r =
                fused.fit_rms_mm / config_.ir_cov_inflate_ref_rms_mm;
            cov_scale = std::max(1.0, r * r);
        }
        result.timings_ms["ir_cov_scale"] = cov_scale;
        // Hand the MAP measurement covariance to the temporal filter, reordered
        // from the MAP's [t, rot] layout to the filter's [rvec(rot), tvec(t)].
        for (int a = 0; a < 6; ++a) {
            const int ma = a < 3 ? a + 3 : a - 3;
            for (int b = 0; b < 6; ++b) {
                const int mb = b < 3 ? b + 3 : b - 3;
                ir_meas_cov_[static_cast<size_t>(6 * a + b)] =
                    cov_scale * fused.cov[static_cast<size_t>(6 * ma + mb)];
            }
        }
        ir_meas_cov_valid_ = true;
        for (int i = 0; i < 3; ++i) {
            result.rvec[static_cast<size_t>(i)] =
                fused.rvec[static_cast<size_t>(i)];
            result.tvec[static_cast<size_t>(i)] =
                fused.tvec[static_cast<size_t>(i)];
        }
        cv::Mat rv_out(3, 1, CV_64F);
        for (int i = 0; i < 3; ++i) {
            rv_out.at<double>(i) = fused.rvec[static_cast<size_t>(i)];
        }
        cv::Mat R_out;
        cv::Rodrigues(rv_out, R_out);
        if (result.T_marker_camera.size() == 16) {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    result.T_marker_camera[static_cast<size_t>(r * 4 + c)] =
                        R_out.at<double>(r, c);
                }
                result.T_marker_camera[static_cast<size_t>(r * 4 + 3)] =
                    fused.tvec[static_cast<size_t>(r)];
            }
        }
    }

    // Corner evidence existed but no attempt survived the gates: this
    // frame is suspect (glare, adverse geometry, RGB drift beyond the
    // trans-jump gate) — the pose filter inflates the measurement
    // covariance for this frame. Pair count is the reference-free
    // (quadratic_form) signal.
    ir_last_fusion_rejected_ =
        !fused.applied && (fused.ref_count > 0 || fused.pairs > 0);
    result.timings_ms["ir_fusion_rejected"] =
        ir_last_fusion_rejected_ ? 1.0 : 0.0;
    // Reference-free QF fusion has no IR reference library to enroll, so the
    // RGB model_ref enrollment gate below sees a clean quality signal every
    // frame.
    ir_last_fusion_quality_ok_ = true;
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
    const bool is_qf =
        config_.checker_tracked_refine_method == "quadratic_form";
    if (!is_qf) {
        return;
    }

    // Anchor source for the next frame: corners of this frame — their uv
    // plus the marker coordinate (id-derived, pose-independent).
    const auto refreshAnchors = [&]() {
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
    };

    if (!result.current_pose_accepted ||
        result.rvec.size() != 3 ||
        result.tvec.size() != 3) {
        // quadratic_form: the anchors track every FRESHLY MEASURED frame —
        // the corner identity and its marker coordinate do not depend on
        // the pose, and the detector matches anchors against its PREVIOUS
        // frame. Refreshing only on accepted poses lets the positional
        // match (1.5 px) tear off during rejected streaks under motion —
        // exactly the rotation phases where coverage matters most.  The
        // source is the per-frame correspondence capture (fresh uv by
        // construction) — result.corners on hold frames carry STALE
        // held positions and must never feed the anchors (v6 regression:
        // divot corners 45->30).  model_warp keeps the accepted-only
        // behaviour (bit-identity; its reference warp also bakes the
        // pose, so stale anchors are safer there than wrong ones).
        if (is_qf && qf_frame_anchor_uv_.size() >= 8) {
            model_prev_uv_ = qf_frame_anchor_uv_;
            model_prev_xyz_ = qf_frame_anchor_xyz_;
        }
        return;
    }

    refreshAnchors();

    // Pose history for the constant-velocity prediction.
    model_prev_rvec_ = model_curr_rvec_;
    model_prev_tvec_ = model_curr_tvec_;
    model_curr_rvec_ = result.rvec;
    model_curr_tvec_ = result.tvec;
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
        resetPoseFilter();
        return;
    }

    if (pose_tracker_.hasPose()) {
        // Hysteresis: don't escalate to the expensive full recovery on a
        // single transient failure (a high-reproj frame on the cylinder). Hold
        // the last pose in Tracking mode so the NEXT frame retries the cheap
        // tracking path from the still-tracked corners; escalate to Recovering
        // only once the failure persists.
        if (lost_frames_ >= config_.recovery_grace_frames) {
            mode_ = TrackerMode::Recovering;
        }
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
    checker_config.qf_profile_half_px = config.checker_qf_profile_half_px;
    checker_config.qf_min_contrast = config.checker_qf_min_contrast;
    checker_config.qf_max_profile_rms = config.checker_qf_max_profile_rms;
    checker_config.qf_junction_margin_frac =
        config.checker_qf_junction_margin_frac;
    checker_config.qf_min_row_points = config.checker_qf_min_row_points;
    checker_config.qf_min_col_points = config.checker_qf_min_col_points;
    checker_config.qf_conic_gain = config.checker_qf_conic_gain;
    checker_config.qf_max_fit_rms_px = config.checker_qf_max_fit_rms_px;
    checker_config.qf_max_dev_px = config.checker_qf_max_dev_px;
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
