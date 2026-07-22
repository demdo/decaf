#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "checkerboard_types.hpp"
#include "corner_detection.hpp"
#include "corner_refinement.hpp"
#include "lattice_model.hpp"
#include "grid_builder.hpp"
#include "lk_tracker.hpp"
#include "tracking_validator.hpp"

namespace hydramarker {

struct CheckerboardRecoveryDebug {
    std::vector<cv::Point2f> raw_candidates;
    std::vector<RefinedCorner> refined_corners;
    std::vector<cv::Point2f> valid_refined_points;

    LatticeResult lattice;
    CheckerboardDetection detection;

    bool has_lattice = false;
    bool has_detection = false;

    float scale = 1.0f;

    // Intermediate image maps for step-by-step documentation. All at the
    // WORK resolution (post ROI/det_width scaling); multiply point overlays
    // that use these by `scale` accordingly. Populated by debugRecoveryStages.
    cv::Mat gray;        // full-res grayscale (toGray8), before any scaling
    cv::Mat work;        // actual detection input (ROI / det_width scaled)
    cv::Mat fast_image;  // blurred image the gradient detector runs on
    cv::Mat grad_x;      // Sobel x on fast_image (CV_32F)
    cv::Mat grad_y;      // Sobel y (CV_32F)
    cv::Mat response;    // saddle response after NMS+threshold (CV_32F)
};

// Simplified persistent corner — no quadrant scoring.
// The quadrant test is only used at recovery time to filter raw candidates.
// During tracking, LK + forward-backward + spacing filter are sufficient
// and more reliable under illumination changes and motion blur.
struct PersistentTrackedCorner {
    GridCorner corner;

    // Number of consecutive frames this corner was not found by LK.
    // Reset to 0 on every successful LK observation.
    // Evicted when missed_frames > config_.max_missed_frames.
    int missed_frames = 0;

    // True if this corner was successfully LK-tracked in the current frame.
    bool tracked = false;

    int observed_frames = 0;
    int predicted_frames = 0;

    // Photometric visibility score in [0, 1].
    // Computed from local checkerboard contrast along the grid axes
    // (axis_u = direction to i+1 neighbour, axis_v = direction to j+1
    // neighbour) sampled from the current frame.
    // A real front-facing corner has score close to 1.0; a corner that has
    // rotated to the back of the cylinder has score near 0.0 because the
    // four quadrants lose their alternating bright/dark structure.
    // Evicted immediately when smoothed_visibility < config_.visibility_evict_threshold.
    float visibility_score          = 1.0f;
    float smoothed_visibility_score = 1.0f;  // EMA over visibility_smoothing_alpha frames
    int low_visibility_frames = 0;
};

struct PendingCompletionCorner {
    int i = 0;
    int j = 0;
    cv::Point2f uv;
    int hits = 0;
    int measured_hits = 0;
    int missed = 0;
    int support = 0;
    float error = std::numeric_limits<float>::max();
};

class CheckerboardDetector {
public:
    CheckerboardDetector();
    explicit CheckerboardDetector(CheckerboardDetectorConfig config);

    std::optional<CheckerboardDetection> detect(const cv::Mat& image);

    // Per-frame model context for the "model_warp" tracked-corner
    // measurement.  Set by the tracker engine before detect(); an input
    // with enabled = false switches the operator off for the frame.
    void setCornerModelInput(const CornerModelFrameInput& input);

    CheckerboardRecoveryDebug debugRecoveryStages(const cv::Mat& image) const;

    void resetTracking();
    bool isTracking() const;
    std::unordered_map<std::string, double> lastTimingsMs() const;
    void addTimingMs(const std::string& name, double elapsed_ms) const;
    static double elapsedMs(std::int64_t start_tick);

    // Per-corner operator comparison of the last detect() call (empty
    // unless model_warp was active; see TrackedRefineSample).
    const std::vector<TrackedRefineSample>& lastTrackedRefineSamples() const {
        return last_tracked_refine_samples_;
    }

private:
    CheckerboardDetectorConfig config_;
    std::vector<TrackedRefineSample> last_tracked_refine_samples_;

    int frame_index_ = 0;
    int degraded_frames_count_ = 0;
    int low_corner_frames_ = 0;
    int undecodeable_tracking_frames_ = 0;
    int held_output_frames_ = 0;
    int roi_align_fail_frames_ = 0;
    int roi_recovery_fail_frames_ = 0;
    // Slow-decaying ceiling of the tracked corner count: reference for the
    // moving-and-eroding refresh trigger (health relative to the visible
    // marker region, not an absolute count).
    double refresh_corner_ceiling_ = 0.0;
    // Frame-to-frame pose rotation (deg) fed by the engine each frame; the
    // conservative default (large) keeps the old always-weak-while-moving
    // behaviour for callers that never provide it.
    double inter_frame_rotation_deg_ = 1e9;

public:
    void setInterFrameRotationDeg(double deg) {
        inter_frame_rotation_deg_ = deg;
    }

private:
    int stable_refresh_zero_gain_count_ = 0;
    int local_completion_soft_zero_gain_count_ = 0;

    cv::Mat last_gray_;
    CheckerboardDetection last_detection_;
    bool tracking_active_ = false;

    std::vector<PersistentTrackedCorner> persistent_corners_;
    std::vector<PendingCompletionCorner> pending_completion_corners_;
    std::unordered_map<std::int64_t, cv::Point2f> lk_corner_displacements_;

    struct CornerKalmanAxisState {
        cv::Vec3d x = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Matx33d P = cv::Matx33d::eye() * 100.0;
    };

    struct CornerKalmanState {
        CornerKalmanAxisState u;
        CornerKalmanAxisState v;
        int last_frame = -1;
        bool initialized = false;
    };

    std::unordered_map<std::int64_t, CornerKalmanState> lk_corner_kalman_;

    CornerModelFrameInput corner_model_input_;

    CornerDetector corner_detector_;
    CornerRefiner corner_refiner_;
    LatticeModel lattice_model_;
    GridBuilder grid_builder_;
    LKTracker lk_tracker_;
    TrackingValidator tracking_validator_;

    struct RecoveryRegionCache {
        int frame_index = -1;
        cv::Rect roi;
        CornerDetectionResult raw;
        std::vector<RefinedCorner> refined;
    };

    mutable std::optional<RecoveryRegionCache> recovery_region_cache_;

    std::vector<cv::Mat> last_gray_pyramid_;
    std::vector<cv::Mat> pending_current_gray_pyramid_;
    int last_gray_pyramid_frame_index_ = -1;
    int pending_current_gray_pyramid_frame_index_ = -1;

private:
    static cv::Mat toGray8(const cv::Mat& image);

    std::optional<CheckerboardDetection>
    detectRecovery(
        const cv::Mat& gray,
        const CheckerboardDetection* roi_hint = nullptr,
        bool allow_full_frame_fallback = true,
        bool roi_candidates_only = false
    ) const;

    std::optional<CheckerboardDetection>
    detectRecoveryInRegion(
        const cv::Mat& gray,
        const cv::Rect& roi,
        const char* timing_prefix = nullptr,
        bool candidates_only = false
    ) const;

    std::optional<CheckerboardDetection>
    buildDetectionFromCorners(
        const std::vector<cv::Point2f>& corners
    ) const;

    // Thread-safe variant for the parallel subset search: timings go into
    // the out-params instead of the shared (mutable) timings map.
    std::optional<CheckerboardDetection>
    buildDetectionFromCornersTimed(
        const std::vector<cv::Point2f>& corners,
        double& lattice_fit_ms,
        double& grid_build_ms
    ) const;

    std::optional<CheckerboardDetection>
    buildBestDetectionFromCornerClusters(
        const std::vector<cv::Point2f>& corners
    ) const;

    std::optional<CheckerboardDetection>
    buildVisibleTrackedDetection(
        const CheckerboardDetection& previous,
        const TrackingValidationResult& validation
    ) const;

    std::optional<CheckerboardDetection>
    trackFromPreviousFrame(const cv::Mat& gray);

    static void initializeCornerKalman(
        CornerKalmanState& state,
        const cv::Point2f& uv,
        int frame_index);
    static void predictCornerKalman(
        CornerKalmanState& state,
        double dt,
        double process_noise_px);
    static cv::Point2f cornerKalmanPosition(const CornerKalmanState& state);
    static cv::Point2f updateCornerKalman(
        CornerKalmanState& state,
        const cv::Point2f& measurement,
        double measurement_noise_px);
    static void predictCornerKalmanAxis(
        CornerKalmanAxisState& axis,
        double dt,
        double process_noise_px);
    static double updateCornerKalmanAxis(
        CornerKalmanAxisState& axis,
        double measurement,
        double measurement_noise_px);

    // recovery_detection: if provided and tracking is active, new corners
    // from recovery that are not yet in persistent_corners_ are injected
    // directly into the persistent set (no lattice refit needed).
    void updateTrackingState(
        const cv::Mat& gray,
        const CheckerboardDetection& measured_detection,
        const CheckerboardDetection* recovery_detection = nullptr
    );

    // Injects corners from recovery_detection into persistent_corners_
    // that are not already represented (by grid ID or proximity).
    // Called only during active tracking, after the LK update.
    void injectRecoveryCorners(
        const CheckerboardDetection& recovery_detection,
        float spacing
    );

    struct LocalCompletionResult {
        int added = 0;
        int durable_added = 0;
        int transient_added = 0;
    };

    // Searches for missing grid corners by interpolating expected positions
    // from known neighbours and looking for raw candidates nearby.
    LocalCompletionResult tryCompleteMissingCorners(
        const cv::Mat& gray,
        bool tracking
    );

    CheckerboardDetection buildDetectionFromPersistent(
        bool tracking,
        bool stable
    ) const;

    std::optional<CheckerboardDetection> mergeMeasuredDetections(
        const CheckerboardDetection& primary,
        const CheckerboardDetection& secondary,
        float duplicate_dist_px
    ) const;

    std::optional<CheckerboardDetection> alignDetectionGridToReference(
        const CheckerboardDetection& detection,
        const CheckerboardDetection& reference
    ) const;

    static int findPersistentCornerByGrid(
        const std::vector<PersistentTrackedCorner>& corners,
        int i,
        int j
    );

    static int findPersistentCornerByNearestUv(
        const std::vector<PersistentTrackedCorner>& corners,
        const cv::Point2f& uv,
        float max_dist_px
    );

    static bool hasNearbyPoint(
        const std::vector<cv::Point2f>& points,
        const cv::Point2f& uv,
        float radius_px
    );

    // Computes a photometric checkerboard-contrast score in [0,1] for a
    // tracked corner.  Quadrant sampling axes are derived from the corner's
    // grid neighbours in persistent_corners_ (Option B), so the test is
    // rotation- and perspective-robust for any marker size.
    // Returns 0 if axes cannot be estimated (isolated corner — handled by
    // the existing fast-eviction rule instead).
    float computeCornerVisibilityScore(
        const cv::Mat& gray,          // current frame, CV_8U
        const PersistentTrackedCorner& pc,
        float spacing                 // estimated grid spacing in pixels
    ) const;

    mutable std::unordered_map<std::string, double> last_timings_ms_;

    int stableRefreshBackoffFactor() const;
    void resetStableRefreshBackoff();
    void recordStableRefreshOutcome(
        bool stable_maintenance_refresh,
        bool durable_gain,
        bool transient_gain
    );
    int localCompletionSoftBackoffFactor() const;
    void resetLocalCompletionSoftBackoff();
    void recordLocalCompletionSoftOutcome(
        bool soft_probe,
        int durable_completed,
        int transient_completed
    );
    void clearTimings() const;
};

} // namespace hydramarker
