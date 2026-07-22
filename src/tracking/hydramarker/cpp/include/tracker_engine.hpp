#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "checkerboard_detector.hpp"
#include "correspondence_builder.hpp"
#include "dot_detector.hpp"
#include "ir_pose_refiner.hpp"
#include "marker_field.hpp"
#include "marker_geometry.hpp"
#include "patch_decoder.hpp"
#include "patch_extractor.hpp"
#include "pose_output_filter.hpp"
#include "tracker_config.hpp"
#include "tracker_geometry.hpp"
#include "tracker_persistence.hpp"
#include "tracker_pose.hpp"
#include "tracker_types.hpp"

namespace hydramarker {

class TrackerEngine {
public:
    TrackerEngine(
        const std::string& field_path,
        const std::string& marker_json_path,
        const cv::Matx33d& K,
        const std::vector<double>& dist_coeffs,
        const TrackerConfig& config = TrackerConfig()
    );

    void reset();

    TrackerFrameResult processFrame(
        const cv::Mat& frame,
        bool run_detection = true
    );

    // Design "B": same as processFrame but with the synchronized IR stereo
    // pair. The RGB pipeline runs unchanged; when the IR refinement stage is
    // enabled (config flag) AND calibrated (setIrCalibration) the REPORTED
    // pose is re-measured on the global-shutter IR pair. Without both the
    // call is byte-identical to processFrame(frame, run_detection).
    TrackerFrameResult processFrame(
        const cv::Mat& frame,
        const cv::Mat& ir_left,
        const cv::Mat& ir_right,
        bool run_detection = true
    );

    // Data gate of the IR refinement stage (the rig calibration from
    // realsense_ir_calibration.npz). Rotations/translations map INTO the
    // named frame; translations in millimetres.
    void setIrCalibration(
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
    );

    // Drop the IR reference library (e.g. after a runtime IR-exposure change:
    // the warp templates are exposure-dependent, a stale template measured
    // fit RMS ~1.4 all run). References re-enroll through the normal gates.
    void resetIrReferences();

    // Last frame's measured QF-IR stereo pairs (self-calibration input).
    const IrPoseRefiner::IrPairDump& lastIrPairDump() const {
        return ir_pose_refiner_.lastPairDump();
    }

    // Stereo self-calibration correction for the QF-IR path (7 polynomial
    // coefficients; empty disables).
    void setIrSelfCal(const std::vector<double>& coef) {
        ir_pose_refiner_.setSelfCal(coef);
    }
    void setIrSelfCalCorners(const std::vector<double>& flat_xyz,
                             const std::vector<double>& dxr) {
        ir_pose_refiner_.setSelfCalCorners(flat_xyz, dxr);
    }

    // Offline model refinement: reference-free QF corner measurement in one
    // IR view (left/right of the configured rig) under an explicit
    // marker->IR-view pose.
    const IrPoseRefiner& irPoseRefiner() const { return ir_pose_refiner_; }

    int frameIndex() const;
    TrackerMode mode() const;
    bool markerAssetsLoaded() const;
    const TrackerConfig& config() const;

private:
    TrackerConfig config_;
    cv::Matx33d K_;
    std::vector<double> dist_coeffs_;

    MarkerField field_;
    MarkerGeometry geometry_;
    CheckerboardDetector checkerboard_detector_;
    DotDetector dot_detector_;
    PatchExtractor patch_extractor_;
    PatchDecoder patch_decoder_;
    CorrespondenceBuilder correspondence_builder_;
    MapPoseTracker pose_tracker_;
    TrackerGeometry tracker_geometry_;
    PersistentMatcher persistent_matcher_;

    TrackerMode mode_ = TrackerMode::Lost;
    int frame_index_ = 0;
    int lost_frames_ = 0;
    int low_fresh_correspondence_frames_ = 0;
    int pose_propagation_block_until_frame_ = -1;
    int max_pts_seen_ = 0;
    double last_good_reproj_px_ = -1.0;
    int last_accepted_pose_frame_ = -1;
    int last_accepted_visual_corner_count_ = 0;
    std::vector<double> last_accepted_rvec_;
    std::vector<double> last_accepted_tvec_;
    std::vector<double> last_accepted_T_marker_camera_;

    // ---- quadratic_form corner measurement (live wiring) ----
    // Anchors from the last accepted frame's corners (uv + pose-independent
    // marker coordinate) seed the next frame's measurement.  The input frame
    // of the current processFrame call is kept as a cheap header copy.
    std::vector<cv::Point2f> model_prev_uv_;
    std::vector<cv::Vec3d> model_prev_xyz_;
    // quadratic_form: per-frame fresh correspondence capture (uv + marker
    // xyz), the pose-independent anchor source for rejected/hold streaks.
    std::vector<cv::Point2f> qf_frame_anchor_uv_;
    std::vector<cv::Vec3d> qf_frame_anchor_xyz_;
    cv::Mat current_frame_;
    // Pixel->ray lookup table, built once on first use (perf).
    cv::Mat model_ray_lut_;
    // Last two accepted poses for constant-velocity pose prediction.
    std::vector<double> model_curr_rvec_;
    std::vector<double> model_curr_tvec_;
    std::vector<double> model_prev_rvec_;
    std::vector<double> model_prev_tvec_;

    // ---- pose warmup status ----
    // Right after (re)acquisition the corner set is still saturating and the
    // pose wanders along the weak observability mode; pose_converged_
    // latches once the set has been quiet for the configured window.
    int warmup_accepted_frames_ = 0;
    int warmup_quiet_streak_ = 0;
    bool pose_converged_ = false;

    // ---- anisotropic pose output filter (output-only) ----
    // Adaptive constant-velocity model over (rvec, tvec) with per-frame
    // measurement covariance sigma_px^2 (J^T J)^-1. Filters ONLY the reported
    // pose; the internal chain (anchors, persistence, prediction) keeps the
    // raw pose so no feedback loop forms. See PoseOutputFilter.
    PoseOutputFilter pose_output_filter_;

    // ---- IR global-shutter pose refinement (output-only, opt-in) ----
    // Re-measures the reported pose on the IR stereo pair; runs FIRST in the
    // output chain (measurement before bias correction before smoothing).
    // The internal chain keeps the raw RGB pose. Inert unless enabled AND
    // calibrated AND the frame pair was supplied.
    IrPoseRefiner ir_pose_refiner_;
    cv::Mat current_ir_left_;
    cv::Mat current_ir_right_;
    // When the IR MAP fusion applied this frame: its 6x6 measurement covariance
    // (H^-1) reordered to the filter's [rvec, tvec] layout, fed to the temporal
    // filter so it smooths the IR-informed pose with the right anisotropy.
    std::array<double, 36> ir_meas_cov_{};
    bool ir_meas_cov_valid_ = false;
    // Rolling-shutter velocity estimate: previous frame's measured pose-corner
    // pixels (key = global_row*1024+global_col) and the last valid median
    // per-frame corner motion, used to inflate the RGB sigma under motion.
    std::unordered_map<int, cv::Point2d> ir_prev_corner_uv_;
    double ir_last_vpx_ = 0.0;
    // Fusion attempted with references but rejected by the gates this
    // frame: the RGB corner evidence is suspect -> pose filter distrust.
    bool ir_last_fusion_rejected_ = false;
    std::array<double, 3> ir_prev_rvec_{};  // rotation gate on the shear comp.
    bool ir_prev_rvec_valid_ = false;
    // Last frame's IR fusion quality verdict: gates model_warp (re-)enrollment
    // (template pose is baked in; never enroll during an untrusted episode).
    // True while IR is off / has no references (RGB-only unaffected).
    bool ir_last_fusion_quality_ok_ = true;

    TrackerFrameResult processFrameInternal(
        const cv::Mat& frame,
        bool run_detection
    );
    void updateCornerModelInput();
    void updateModelWarpStateAfterFrame(const TrackerFrameResult& result);
    void updatePoseWarmupState(TrackerFrameResult& result);
    void resetPoseWarmup();
    bool posePointUsable(bool predicted, int observed_frames) const;
    void applyPoseFilter(TrackerFrameResult& result);
    void resetPoseFilter();
    void applyIrRefinement(TrackerFrameResult& result);

    static CheckerboardDetectorConfig makeCheckerboardConfig(
        const TrackerConfig& config
    );
    static DotDetectorConfig makeDotDetectorConfig(const TrackerConfig& config);
    static PatchDecoderConfig makePatchDecoderConfig(const TrackerConfig& config);
    static CorrespondenceBuilderConfig makeCorrespondenceBuilderConfig(
        const TrackerConfig& config
    );
    static MapPoseTrackerConfig makeMapPoseTrackerConfig(
        const TrackerConfig& config
    );
    static double confidence(int num_inliers, double mean_error_px, const TrackerConfig& config);
    static PoseTrackPoint posePointFromCorrespondence(
        const Correspondence2D3D& corr
    );
    static TrackerCorner trackerCornerFromCorrespondence(
        const Correspondence2D3D& corr
    );
    static TrackerCorner trackerCornerFromPosePoint(
        const PoseTrackPoint& point
    );
    static GlobalCornerIdentity identityFromTrackerCorner(
        const TrackerCorner& corner
    );
    static FrameDetectedCorner detectedCornerFromGridCorner(
        const GridCorner& corner
    );

    void attachDetectionResult(
        TrackerFrameResult& result,
        const CheckerboardDetection& detection
    ) const;

    void packagePoseResult(
        TrackerFrameResult& result,
        const MapPoseResult& pose,
        PoseSource source,
        const std::string& message,
        const std::vector<TrackerCorner>& visual_corners,
        const std::vector<TrackerCorner>& correspondence_corners
    ) const;

    std::vector<TrackerCorner> visualCornersForPose(
        const MapPoseResult& pose
    ) const;

    void pointsToCv(
        const std::vector<PoseTrackPoint>& points,
        std::vector<cv::Point3d>& object_points,
        std::vector<cv::Point2d>& image_points
    ) const;

    bool reprojectionMeanMax(
        const std::vector<cv::Point3d>& object_points,
        const std::vector<cv::Point2d>& image_points,
        const std::vector<double>& rvec,
        const std::vector<double>& tvec,
        double& mean_error_px,
        double& max_error_px
    ) const;

    bool acceptPoseState(
        const MapPoseResult& pose,
        int visual_corner_count
    );
    void clearAcceptedState();
    void onTrackingFailure();
    void attachRuntimeState(TrackerFrameResult& result) const;
    void finalizeFrameResult(
        TrackerFrameResult& result,
        std::int64_t frame_t0
    ) const;
    void noteLowFreshCorrespondenceFailure(int fresh_count);
    void forceLocalRecovery();
    void restorePoseTrackerState(
        bool had_pose,
        const std::vector<double>& rvec,
        const std::vector<double>& tvec
    );

    bool detectionHasDecodeableCellSpan(
        const CheckerboardDetection& detection
    ) const;

    std::optional<CheckerboardDetection> buildPosePropagatedDetection(
        int image_width,
        int image_height
    ) const;

    bool tryPersistentFallbackPose(
        const CheckerboardDetection& detection,
        const std::string& reason,
        bool after_decode_fail,
        TrackerFrameResult& result,
        std::int64_t frame_t0
    );

    bool tryHoldLastPose(
        const CheckerboardDetection& detection,
        const std::string& reason,
        const std::vector<TrackerCorner>& correspondence_corners,
        TrackerFrameResult& result,
        std::int64_t frame_t0
    );

    bool tryHoldLastPoseWithoutDetection(
        TrackerFrameResult& result,
        std::int64_t frame_t0
    );

    bool tryEmergencyLastPose(
        const CheckerboardDetection* detection,
        const std::string& reason,
        TrackerFrameResult& result,
        std::int64_t frame_t0
    );

    bool fallbackPoseRejectionReason(
        const CheckerboardDetection& detection,
        const MapPoseResult& pose,
        std::string& reason
    ) const;
    bool decodeUpdateRejectionReason(
        const std::vector<TrackerCorner>& visual_corners,
        std::string& reason
    ) const;

    void refreshPersistentIdentities(
        const std::vector<TrackerCorner>& visual_corners,
        int frame_index
    );

    bool tryFastPersistentPose(
        const CheckerboardDetection& detection,
        TrackerFrameResult& result,
        std::int64_t frame_t0
    );
};

} // namespace hydramarker
