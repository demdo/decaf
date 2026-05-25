#include "checkerboard_detector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

namespace hydramarker {

namespace {

constexpr int kMaxMissedFrames = 2;

// updateCellGeometryFromCorners is kept for the rare case where external
// code (e.g. debug visualisers) mutates corner UVs in-place and needs to
// resync the cell UV arrays without a full refit.
static void updateCellGeometryFromCorners(CheckerboardDetection& det) {
    for (auto& cell : det.cells) {
        bool ok = true;

        for (int k = 0; k < 4; ++k) {
            const int idx = cell.corner_indices[k];

            if (idx < 0 || idx >= static_cast<int>(det.corners.size())) {
                ok = false;
                break;
            }

            cell.corner_uv[k] = det.corners[idx].uv;
        }

        if (!ok) {
            continue;
        }

        cell.center_uv =
            0.25f * (
                cell.corner_uv[0] +
                cell.corner_uv[1] +
                cell.corner_uv[2] +
                cell.corner_uv[3]
            );
    }
}


static float dist2(const cv::Point2f& a, const cv::Point2f& b) {
    const cv::Point2f d = a - b;
    return d.x * d.x + d.y * d.y;
}


static bool isBetterThanTracked(
    const CheckerboardDetection& candidate,
    const CheckerboardDetection& tracked
) {
    if (!candidate.valid()) {
        return false;
    }

    if (!tracked.valid()) {
        return true;
    }

    const int cand_corners = static_cast<int>(candidate.corners.size());
    const int cand_cells   = static_cast<int>(candidate.cells.size());

    const int trk_corners = static_cast<int>(tracked.corners.size());
    const int trk_cells   = static_cast<int>(tracked.cells.size());

    if (cand_cells >= trk_cells + 8) {
        return true;
    }

    if (cand_corners >= trk_corners + 12 &&
        cand_cells   >= trk_cells   -  3) {
        return true;
    }

    if (!tracked.stable) {
        if (cand_cells   >= trk_cells   &&
            cand_corners >= trk_corners - 8) {
            return true;
        }

        if (cand_corners >= trk_corners &&
            cand_cells   >= trk_cells   - 5) {
            return true;
        }
    }

    return false;
}

} // namespace


// ------------------------------------------------------------

CheckerboardDetector::CheckerboardDetector()
    : CheckerboardDetector(CheckerboardDetectorConfig{}) {}

CheckerboardDetector::CheckerboardDetector(
    CheckerboardDetectorConfig config
)
    : config_(config) {}


// ------------------------------------------------------------

std::optional<CheckerboardDetection> CheckerboardDetector::detect(
    const cv::Mat& image
) {
    const cv::Mat gray = toGray8(image);

    if (gray.empty()) {
        resetTracking();
        return std::nullopt;
    }

    ++frame_index_;

    if (tracking_active_ &&
        !last_gray_.empty() &&
        !last_detection_.corners.empty()) {

        auto tracked = trackFromPreviousFrame(gray);

        if (tracked && tracked->valid()) {
            tracked->tracking = true;

            const bool periodic_refresh =
                config_.refresh_interval_frames > 0 &&
                frame_index_ % config_.refresh_interval_frames == 0;

            const bool do_refresh = periodic_refresh || !tracked->stable;

            if (do_refresh) {
                auto recovered = detectRecovery(gray);

                if (recovered && recovered->valid()) {
                    recovered->tracking = false;
                    recovered->stable   = false;

                    if (!tracked->stable ||
                        isBetterThanTracked(*recovered, *tracked)) {
                        updateTrackingState(gray, *recovered);
                        return last_detection_;
                    }

                    auto merged = mergeMeasuredDetections(
                        *tracked,
                        *recovered,
                        config_.duplicate_corner_dist_px
                    );

                    if (merged && merged->valid()) {
                        merged->tracking = true;
                        merged->stable   = tracked->stable;
                        updateTrackingState(gray, *merged);
                        return last_detection_;
                    }

                    updateTrackingState(gray, *tracked);
                    return last_detection_;
                }
            }

            updateTrackingState(gray, *tracked);
            return last_detection_;
        }

        tracking_active_ = false;
    }

    auto recovered = detectRecovery(gray);

    if (recovered && recovered->valid()) {
        recovered->tracking = false;
        recovered->stable   = false;

        updateTrackingState(gray, *recovered);

        if (config_.refresh_interval_frames > 0) {
            frame_index_ = config_.refresh_interval_frames - 1;
        }

        return last_detection_;
    }

    resetTracking();
    return std::nullopt;
}


// ------------------------------------------------------------

CheckerboardRecoveryDebug CheckerboardDetector::debugRecoveryStages(
    const cv::Mat& image
) const {
    CheckerboardRecoveryDebug dbg;

    const cv::Mat gray = toGray8(image);

    if (gray.empty()) {
        return dbg;
    }

    cv::Mat work = gray;
    float scale = 1.0f;

    if (config_.det_width > 0 && gray.cols > config_.det_width) {
        scale =
            static_cast<float>(config_.det_width) /
            static_cast<float>(gray.cols);

        const int new_w = config_.det_width;
        const int new_h = std::max(
            1,
            static_cast<int>(std::round(gray.rows * scale))
        );

        cv::resize(
            gray,
            work,
            cv::Size(new_w, new_h),
            0.0,
            0.0,
            cv::INTER_AREA
        );
    }

    dbg.scale = scale;

    CornerDetectionResult raw = corner_detector_.detect(
        work,
        config_.max_recovery_corners,
        config_.saddle_sigma,
        config_.saddle_response_threshold
    );

    const float inv_scale = 1.0f / scale;

    dbg.raw_candidates.reserve(raw.points.size());

    for (const auto& p : raw.points) {
        dbg.raw_candidates.emplace_back(
            p.x * inv_scale,
            p.y * inv_scale
        );
    }

    if (raw.points.empty()) {
        return dbg;
    }

    CornerRefinementConfig refine_config;
    refine_config.radius           = config_.saddle_radius;
    refine_config.iterations       = config_.saddle_iterations;
    refine_config.max_angle_bias_deg = config_.saddle_max_angle_bias_deg;
    refine_config.correlation_drop = config_.saddle_correlation_drop;
    refine_config.merge_radius_px  =
        std::max(1.0f, config_.duplicate_corner_dist_px * scale);

    std::vector<RefinedCorner> refined = corner_refiner_.refine(
        work,
        raw.points,
        raw.grad_x,
        raw.grad_y,
        refine_config
    );

    dbg.refined_corners.reserve(refined.size());
    dbg.valid_refined_points.reserve(refined.size());

    for (auto c : refined) {
        c.uv.x *= inv_scale;
        c.uv.y *= inv_scale;

        dbg.refined_corners.push_back(c);

        if (c.valid) {
            dbg.valid_refined_points.push_back(c.uv);
        }
    }

    if (static_cast<int>(dbg.valid_refined_points.size()) < config_.min_corners) {
        return dbg;
    }

    auto lattice = lattice_model_.fit(dbg.valid_refined_points);

    if (!lattice || !lattice->valid) {
        return dbg;
    }

    dbg.lattice     = *lattice;
    dbg.has_lattice = true;

    auto detection = grid_builder_.build(
        *lattice,
        config_.duplicate_corner_dist_px,
        config_.min_corners,
        config_.min_cells
    );

    if (!detection || !detection->valid()) {
        return dbg;
    }

    dbg.detection     = *detection;
    dbg.has_detection = true;

    return dbg;
}


// ------------------------------------------------------------

void CheckerboardDetector::resetTracking() {
    last_gray_.release();
    last_detection_ = CheckerboardDetection{};
    persistent_corners_.clear();
    tracking_active_ = false;
    frame_index_     = 0;
}

bool CheckerboardDetector::isTracking() const {
    return tracking_active_;
}


// ------------------------------------------------------------

cv::Mat CheckerboardDetector::toGray8(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }

    cv::Mat gray;

    if (image.channels() == 1) {
        gray = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        return {};
    }

    if (gray.type() == CV_8U) {
        return gray.clone();
    }

    cv::Mat gray8;

    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(gray, &min_val, &max_val);

    if (max_val <= min_val) {
        gray.convertTo(gray8, CV_8U);
    } else {
        gray.convertTo(
            gray8,
            CV_8U,
            255.0 / (max_val - min_val),
            -255.0 * min_val / (max_val - min_val)
        );
    }

    return gray8;
}


// ------------------------------------------------------------

std::optional<CheckerboardDetection> CheckerboardDetector::detectRecovery(
    const cv::Mat& gray
) const {
    if (gray.empty()) {
        return std::nullopt;
    }

    cv::Mat work = gray;
    float scale  = 1.0f;

    if (config_.det_width > 0 && gray.cols > config_.det_width) {
        scale =
            static_cast<float>(config_.det_width) /
            static_cast<float>(gray.cols);

        const int new_w = config_.det_width;
        const int new_h = std::max(
            1,
            static_cast<int>(std::round(gray.rows * scale))
        );

        cv::resize(
            gray,
            work,
            cv::Size(new_w, new_h),
            0.0,
            0.0,
            cv::INTER_AREA
        );
    }

    CornerDetectionResult raw = corner_detector_.detect(
        work,
        config_.max_recovery_corners,
        config_.saddle_sigma,
        config_.saddle_response_threshold
    );

    if (raw.points.empty()) {
        return std::nullopt;
    }

    CornerRefinementConfig refine_config;
    refine_config.radius             = config_.saddle_radius;
    refine_config.iterations         = config_.saddle_iterations;
    refine_config.max_angle_bias_deg = config_.saddle_max_angle_bias_deg;
    refine_config.correlation_drop   = config_.saddle_correlation_drop;
    refine_config.merge_radius_px    =
        std::max(1.0f, config_.duplicate_corner_dist_px * scale);

    std::vector<RefinedCorner> refined = corner_refiner_.refine(
        work,
        raw.points,
        raw.grad_x,
        raw.grad_y,
        refine_config
    );

    if (static_cast<int>(refined.size()) < config_.min_corners) {
        return std::nullopt;
    }

    const float inv_scale = 1.0f / scale;

    std::vector<cv::Point2f> corners;
    corners.reserve(refined.size());

    for (const auto& c : refined) {
        if (!c.valid) {
            continue;
        }

        corners.emplace_back(
            c.uv.x * inv_scale,
            c.uv.y * inv_scale
        );
    }

    if (static_cast<int>(corners.size()) < config_.min_corners) {
        return std::nullopt;
    }

    auto detection = buildDetectionFromCorners(corners);

    if (!detection || !detection->valid()) {
        return std::nullopt;
    }

    detection->tracking = false;
    detection->stable   = false;

    return detection;
}


// ------------------------------------------------------------

std::optional<CheckerboardDetection>
CheckerboardDetector::buildDetectionFromCorners(
    const std::vector<cv::Point2f>& corners
) const {
    if (static_cast<int>(corners.size()) < config_.min_corners) {
        return std::nullopt;
    }

    auto lattice = lattice_model_.fit(corners);

    if (!lattice || !lattice->valid) {
        return std::nullopt;
    }

    auto detection = grid_builder_.build(
        *lattice,
        config_.duplicate_corner_dist_px,
        config_.min_corners,
        config_.min_cells
    );

    if (!detection || !detection->valid()) {
        return std::nullopt;
    }

    return detection;
}


// ------------------------------------------------------------
// Fix 2: Topology is never copied from the previous frame.
//
// The old implementation did:
//   1. Assemble det.corners from LK-measured positions.
//   2. For each previous cell, re-index its four corners via old_to_new[].
//
// Problems with that approach:
//   - old_to_new is only defined for corners that survived LK + homography
//     validation.  Any cell with one missing corner was silently dropped,
//     so partial occlusion could wipe out most of the cell set.
//   - The (i,j) grid indices that the cells carry come from the *previous*
//     frame's lattice assignment.  After drift or a merge step those indices
//     may no longer be consistent with each other, producing cells that
//     connect non-adjacent corners or span the wrong grid distance.
//   - No geometry validation was applied to the re-mapped cells.
//
// New approach:
//   1. Assemble det.corners from LK-measured positions (unchanged).
//   2. Extract the plain UV coordinates and run LatticeModel::fit +
//      GridBuilder::build exactly as recovery does.
//   3. The lattice refit produces fresh, consistent (i,j) assignments and
//      the grid builder applies all geometry guards.
//   4. The tracking / stable flags are preserved from the validation result.
// ------------------------------------------------------------

std::optional<CheckerboardDetection>
CheckerboardDetector::buildVisibleTrackedDetection(
    const CheckerboardDetection& previous,
    const TrackingValidationResult& validation
) const {
    // Basic size guard.
    if (validation.visible_indices.size() != validation.visible_points.size()) {
        return std::nullopt;
    }

    if (static_cast<int>(validation.visible_points.size()) <
        config_.min_tracking_corners) {
        return std::nullopt;
    }

    // --- Step 1: assemble corners with updated UVs. ---
    //
    // We keep the GridCorner objects from the previous frame so that any
    // semantic data they carry (e.g. identity) is preserved.  Only the UV
    // coordinate is overwritten with the fresh LK measurement.

    std::vector<GridCorner> tracked_corners;
    tracked_corners.reserve(validation.visible_points.size());

    for (size_t m = 0; m < validation.visible_indices.size(); ++m) {
        const int old_idx = validation.visible_indices[m];

        if (old_idx < 0 ||
            old_idx >= static_cast<int>(previous.corners.size())) {
            continue;
        }

        GridCorner c = previous.corners[old_idx];
        c.uv = validation.visible_points[m];

        tracked_corners.push_back(c);
    }

    if (static_cast<int>(tracked_corners.size()) <
        config_.min_tracking_corners) {
        return std::nullopt;
    }

    // --- Step 2: extract plain UV coordinates for the lattice fit. ---

    std::vector<cv::Point2f> pts;
    pts.reserve(tracked_corners.size());

    for (const auto& c : tracked_corners) {
        pts.push_back(c.uv);
    }

    // --- Step 3: refit lattice and rebuild cell topology from scratch. ---
    //
    // Using min_tracking_corners / min_tracking_cells here (not the stricter
    // recovery thresholds) so that tracking can survive partial occlusion.

    auto lattice = lattice_model_.fit(pts);

    if (!lattice || !lattice->valid) {
        return std::nullopt;
    }

    auto rebuilt = grid_builder_.build(
        *lattice,
        config_.duplicate_corner_dist_px,
        config_.min_tracking_corners,
        config_.min_tracking_cells
    );

    if (!rebuilt || !rebuilt->valid()) {
        return std::nullopt;
    }

    // --- Step 4: stamp tracking metadata and return. ---

    rebuilt->tracking = true;
    rebuilt->stable   = validation.stable;

    return rebuilt;
}


// ------------------------------------------------------------

std::optional<CheckerboardDetection>
CheckerboardDetector::trackFromPreviousFrame(const cv::Mat& gray) {
    if (!tracking_active_ ||
        last_gray_.empty() ||
        last_detection_.corners.empty()) {
        return std::nullopt;
    }

    std::vector<cv::Point2f> prev_points;
    prev_points.reserve(last_detection_.corners.size());

    for (const auto& c : last_detection_.corners) {
        prev_points.push_back(c.uv);
    }

    LKTrackingResult lk = lk_tracker_.track(
        last_gray_,
        gray,
        prev_points,
        config_.lk_win_size,
        config_.lk_max_level,
        config_.lk_max_iters,
        config_.lk_epsilon,
        config_.max_lk_error
    );

    TrackingValidationResult validation =
        tracking_validator_.validate(
            last_detection_,
            lk,
            gray.size(),
            config_
        );

    if (!validation.valid) {
        return std::nullopt;
    }

    // buildVisibleTrackedDetection is now a member function and receives
    // only the previous detection + validation result.  The config_ is
    // accessed directly through this->config_.
    auto detection = buildVisibleTrackedDetection(
        last_detection_,
        validation
    );

    if (!detection || !detection->valid()) {
        return std::nullopt;
    }

    return detection;
}


// ------------------------------------------------------------

void CheckerboardDetector::updateTrackingState(
    const cv::Mat& gray,
    const CheckerboardDetection& measured_detection
) {
    last_gray_ = gray.clone();

    if (!measured_detection.valid()) {
        last_detection_ = CheckerboardDetection{};
        persistent_corners_.clear();
        tracking_active_ = false;
        return;
    }

    const float match_radius =
        std::max(3.0f, 1.5f * config_.duplicate_corner_dist_px);

    if (!tracking_active_ ||
        persistent_corners_.empty() ||
        !measured_detection.tracking) {
        persistent_corners_.clear();
        persistent_corners_.reserve(measured_detection.corners.size());

        for (const auto& c : measured_detection.corners) {
            PersistentTrackedCorner pc;
            pc.corner        = c;
            pc.missed_frames = 0;
            persistent_corners_.push_back(pc);
        }
    } else {
        for (auto& pc : persistent_corners_) {
            pc.missed_frames += 1;
        }

        for (const auto& c : measured_detection.corners) {
            int idx = findPersistentCornerByNearestUv(
                persistent_corners_,
                c.uv,
                match_radius
            );

            if (idx < 0) {
                const int grid_idx = findPersistentCornerByGrid(
                    persistent_corners_,
                    c.i,
                    c.j
                );

                if (grid_idx >= 0) {
                    const float d2 =
                        dist2(persistent_corners_[grid_idx].corner.uv, c.uv);

                    if (d2 <= (2.5f * match_radius) * (2.5f * match_radius)) {
                        idx = grid_idx;
                    }
                }
            }

            if (idx >= 0) {
                persistent_corners_[idx].corner        = c;
                persistent_corners_[idx].missed_frames = 0;
            } else {
                PersistentTrackedCorner pc;
                pc.corner        = c;
                pc.missed_frames = 0;
                persistent_corners_.push_back(pc);
            }
        }

        persistent_corners_.erase(
            std::remove_if(
                persistent_corners_.begin(),
                persistent_corners_.end(),
                [](const PersistentTrackedCorner& pc) {
                    return pc.missed_frames > kMaxMissedFrames;
                }
            ),
            persistent_corners_.end()
        );
    }

    last_detection_ = buildDetectionFromPersistent(
        measured_detection.tracking,
        measured_detection.stable
    );

    if (!last_detection_.valid()) {
        last_detection_ = measured_detection;
    }

    tracking_active_ = last_detection_.valid();
}


// ------------------------------------------------------------

CheckerboardDetection CheckerboardDetector::buildDetectionFromPersistent(
    bool tracking,
    bool stable
) const {
    std::vector<cv::Point2f> points;
    points.reserve(persistent_corners_.size());

    for (const auto& pc : persistent_corners_) {
        points.push_back(pc.corner.uv);
    }

    auto rebuilt = buildDetectionFromCorners(points);

    if (!rebuilt || !rebuilt->valid()) {
        CheckerboardDetection empty;
        empty.tracking = tracking;
        empty.stable   = stable;
        return empty;
    }

    rebuilt->tracking = tracking;
    rebuilt->stable   = stable;
    return *rebuilt;
}


// ------------------------------------------------------------

std::optional<CheckerboardDetection> CheckerboardDetector::mergeMeasuredDetections(
    const CheckerboardDetection& primary,
    const CheckerboardDetection& secondary,
    float duplicate_dist_px
) const {
    std::vector<cv::Point2f> merged_points;
    merged_points.reserve(primary.corners.size() + secondary.corners.size());

    const float r = std::max(2.0f, duplicate_dist_px);

    for (const auto& c : primary.corners) {
        if (!hasNearbyPoint(merged_points, c.uv, r)) {
            merged_points.push_back(c.uv);
        }
    }

    for (const auto& c : secondary.corners) {
        if (!hasNearbyPoint(merged_points, c.uv, r)) {
            merged_points.push_back(c.uv);
        }
    }

    auto rebuilt = buildDetectionFromCorners(merged_points);

    if (!rebuilt || !rebuilt->valid()) {
        return std::nullopt;
    }

    rebuilt->tracking = primary.tracking;
    rebuilt->stable   = primary.stable;
    return rebuilt;
}


// ------------------------------------------------------------

int CheckerboardDetector::findPersistentCornerByGrid(
    const std::vector<PersistentTrackedCorner>& corners,
    int i,
    int j
) {
    for (int idx = 0; idx < static_cast<int>(corners.size()); ++idx) {
        if (corners[idx].corner.i == i && corners[idx].corner.j == j) {
            return idx;
        }
    }

    return -1;
}


// ------------------------------------------------------------

int CheckerboardDetector::findPersistentCornerByNearestUv(
    const std::vector<PersistentTrackedCorner>& corners,
    const cv::Point2f& uv,
    float max_dist_px
) {
    const float max_d2 = max_dist_px * max_dist_px;

    int   best_idx = -1;
    float best_d2  = max_d2;

    for (int idx = 0; idx < static_cast<int>(corners.size()); ++idx) {
        const float d = dist2(corners[idx].corner.uv, uv);
        if (d <= best_d2) {
            best_d2  = d;
            best_idx = idx;
        }
    }

    return best_idx;
}


// ------------------------------------------------------------

bool CheckerboardDetector::hasNearbyPoint(
    const std::vector<cv::Point2f>& points,
    const cv::Point2f& uv,
    float radius_px
) {
    const float r2 = radius_px * radius_px;

    for (const auto& p : points) {
        if (dist2(p, uv) <= r2) {
            return true;
        }
    }

    return false;
}

} // namespace hydramarker