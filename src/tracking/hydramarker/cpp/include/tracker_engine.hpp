#pragma once

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

    TrackerMode mode_ = TrackerMode::Lost;
    int frame_index_ = 0;
    int lost_frames_ = 0;

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
};

} // namespace hydramarker
