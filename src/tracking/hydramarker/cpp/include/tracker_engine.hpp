#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "checkerboard_detector.hpp"
#include "correspondence_builder.hpp"
#include "dot_detector.hpp"
#include "marker_field.hpp"
#include "marker_geometry.hpp"
#include "patch_decoder.hpp"
#include "patch_extractor.hpp"
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

    // ---- model-warp corner measurement (live wiring) ----
    // Reference view enrolled on the first stable accepted pose plus the
    // pose corners of the last accepted frame (anchor source).  The input
    // frame of the current processFrame call is kept as a cheap header copy
    // so enrollment inside finalizeFrameResult can access it.
    bool model_ref_enrolled_ = false;
    cv::Mat model_ref_gray_;            // CV_32F at enrollment
    cv::Matx33d model_ref_R_ = cv::Matx33d::eye();
    cv::Vec3d model_ref_t_{0.0, 0.0, 0.0};
    std::vector<cv::Point2f> model_prev_uv_;
    std::vector<cv::Vec3d> model_prev_xyz_;
    cv::Mat current_frame_;
    // Pixel->ray lookup table, built once on first use (perf).
    cv::Mat model_ray_lut_;
    // Last two accepted poses for constant-velocity pose prediction.
    std::vector<double> model_curr_rvec_;
    std::vector<double> model_curr_tvec_;
    std::vector<double> model_prev_rvec_;
    std::vector<double> model_prev_tvec_;
    // Re-enrollment policy state: total enrollments this tracking session
    // and the current viewing-angle offset to the reference view.
    int model_enroll_count_ = 0;
    double model_ref_view_angle_deg_ = 0.0;

    // ---- pose warmup status ----
    // Right after (re)acquisition the corner set is still saturating and the
    // pose wanders along the weak observability mode; pose_converged_
    // latches once the set has been quiet for the configured window.
    int warmup_accepted_frames_ = 0;
    int warmup_quiet_streak_ = 0;
    bool pose_converged_ = false;

    // ---- anisotropic pose Kalman filter (output-only) ----
    // Constant-velocity model over (rvec, tvec); per-frame measurement
    // covariance sigma_px^2 (J^T J)^-1 keeps observable directions
    // measurement-dominated while the weak observability mode is smoothed.
    // Filters ONLY the reported pose; the internal chain (anchors,
    // persistence, prediction) keeps the raw pose so no feedback loop forms.
    bool pose_kf_initialized_ = false;
    cv::Mat pose_kf_x_;   // 12x1: rvec, tvec, and their per-frame velocities
    cv::Mat pose_kf_P_;   // 12x12 covariance

    TrackerFrameResult processFrameInternal(
        const cv::Mat& frame,
        bool run_detection
    );
    void updateCornerModelInput();
    void updateModelWarpStateAfterFrame(const TrackerFrameResult& result);
    void updatePoseWarmupState(TrackerFrameResult& result);
    void resetPoseWarmup();
    bool posePointUsable(bool predicted, int observed_frames) const;
    void applyPoseKalman(TrackerFrameResult& result);
    void resetPoseKalman();
    bool poseMotionQuiet() const;

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
