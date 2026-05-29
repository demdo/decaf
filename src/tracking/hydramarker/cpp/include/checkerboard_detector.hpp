#pragma once

#include <optional>
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
};

class CheckerboardDetector {
public:
    CheckerboardDetector();
    explicit CheckerboardDetector(CheckerboardDetectorConfig config);

    std::optional<CheckerboardDetection> detect(const cv::Mat& image);

    CheckerboardRecoveryDebug debugRecoveryStages(const cv::Mat& image) const;

    void resetTracking();
    bool isTracking() const;

private:
    CheckerboardDetectorConfig config_;

    int frame_index_ = 0;
    int degraded_frames_count_ = 0;
    int low_corner_frames_ = 0;

    cv::Mat last_gray_;
    CheckerboardDetection last_detection_;
    bool tracking_active_ = false;

    std::vector<PersistentTrackedCorner> persistent_corners_;

    CornerDetector corner_detector_;
    CornerRefiner corner_refiner_;
    LatticeModel lattice_model_;
    GridBuilder grid_builder_;
    LKTracker lk_tracker_;
    TrackingValidator tracking_validator_;

private:
    static cv::Mat toGray8(const cv::Mat& image);

    std::optional<CheckerboardDetection>
    detectRecovery(const cv::Mat& gray) const;

    std::optional<CheckerboardDetection>
    buildDetectionFromCorners(
        const std::vector<cv::Point2f>& corners
    ) const;

    std::optional<CheckerboardDetection>
    buildVisibleTrackedDetection(
        const CheckerboardDetection& previous,
        const TrackingValidationResult& validation
    ) const;

    std::optional<CheckerboardDetection>
    trackFromPreviousFrame(const cv::Mat& gray);

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

    // Searches for missing grid corners by interpolating expected positions
    // from known neighbours and looking for raw candidates nearby.
    void tryCompleteMissingCorners(
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
};

} // namespace hydramarker