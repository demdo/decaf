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

struct PersistentTrackedCorner {
    GridCorner corner;
    int missed_frames = 0;
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

    cv::Mat last_gray_;
    CheckerboardDetection last_detection_;
    bool tracking_active_ = false;

    // Persistent corner hypotheses only.
    // Important: cells/topology are never persisted. They are rebuilt from the
    // current merged point cloud every frame using LatticeModel + GridBuilder.
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

    // Builds a tracked detection from the validated LK result.
    //
    // Key contract (Fix 2):
    //   Corner UVs are taken directly from the LK measurement.
    //   Cell topology is NOT copied from the previous frame.
    //   Instead, LatticeModel + GridBuilder refit from scratch on the
    //   currently visible corners.  This guarantees that only topologically
    //   correct, geometrically valid cells are emitted regardless of how many
    //   corners are currently visible.
    std::optional<CheckerboardDetection>
    buildVisibleTrackedDetection(
        const CheckerboardDetection& previous,
        const TrackingValidationResult& validation
    ) const;

    std::optional<CheckerboardDetection>
    trackFromPreviousFrame(const cv::Mat& gray);

    void updateTrackingState(
        const cv::Mat& gray,
        const CheckerboardDetection& measured_detection
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