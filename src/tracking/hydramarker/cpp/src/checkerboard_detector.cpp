#include "checkerboard_detector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

namespace hydramarker {

namespace {

// ============================================================
// Geometry helpers
// ============================================================

static float dist2(const cv::Point2f& a, const cv::Point2f& b) {
    const cv::Point2f d = a - b;
    return d.x * d.x + d.y * d.y;
}

static float distf(const cv::Point2f& a, const cv::Point2f& b) {
    return std::sqrt(dist2(a, b));
}

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
        if (!ok) continue;
        cell.center_uv =
            0.25f * (cell.corner_uv[0] + cell.corner_uv[1] +
                     cell.corner_uv[2] + cell.corner_uv[3]);
    }
}


// ============================================================
// Spacing estimation (upper quartile — robust to perspective)
// ============================================================

static float estimateMedianSpacing(const CheckerboardDetection& det) {
    if (det.corners.size() < 2) return 0.0f;

    std::vector<std::pair<std::pair<int,int>, cv::Point2f>> by_ij;
    by_ij.reserve(det.corners.size());
    for (const auto& c : det.corners)
        by_ij.push_back({{c.i, c.j}, c.uv});

    auto findUv = [&](int i, int j) -> const cv::Point2f* {
        for (const auto& p : by_ij)
            if (p.first.first == i && p.first.second == j)
                return &p.second;
        return nullptr;
    };

    std::vector<float> dists;
    dists.reserve(det.corners.size() * 2);
    for (const auto& c : det.corners) {
        const cv::Point2f* nb_i = findUv(c.i + 1, c.j);
        if (nb_i) dists.push_back(distf(c.uv, *nb_i));
        const cv::Point2f* nb_j = findUv(c.i, c.j + 1);
        if (nb_j) dists.push_back(distf(c.uv, *nb_j));
    }

    if (dists.empty()) return 0.0f;

    // Upper quartile: dominated by frontal uncompressed corners.
    const size_t q3_idx = static_cast<size_t>(dists.size() * 3 / 4);
    std::nth_element(dists.begin(),
                     dists.begin() + static_cast<std::ptrdiff_t>(q3_idx),
                     dists.end());
    return dists[q3_idx];
}


static float detectionMinSpacingRatio(const CheckerboardDetection& det) {
    if (det.corners.size() < 2) return 1.0f;

    std::vector<std::pair<std::pair<int,int>, cv::Point2f>> by_ij;
    by_ij.reserve(det.corners.size());
    for (const auto& c : det.corners)
        by_ij.push_back({{c.i, c.j}, c.uv});

    auto findUv = [&](int i, int j) -> const cv::Point2f* {
        for (const auto& p : by_ij)
            if (p.first.first == i && p.first.second == j)
                return &p.second;
        return nullptr;
    };

    std::vector<float> dists;
    dists.reserve(det.corners.size() * 2);
    for (const auto& c : det.corners) {
        const cv::Point2f* nb_i = findUv(c.i + 1, c.j);
        if (nb_i) dists.push_back(distf(c.uv, *nb_i));
        const cv::Point2f* nb_j = findUv(c.i, c.j + 1);
        if (nb_j) dists.push_back(distf(c.uv, *nb_j));
    }

    if (dists.size() < 2) return 1.0f;

    std::nth_element(dists.begin(),
                     dists.begin() + static_cast<std::ptrdiff_t>(dists.size() / 2),
                     dists.end());
    const float median = dists[dists.size() / 2];
    if (median < 1.0f) return 1.0f;

    const float min_d = *std::min_element(dists.begin(), dists.end());
    return min_d / median;
}


// ============================================================
// Spacing consistency filter (used in tracking path)
// ============================================================

static std::vector<GridCorner> filterBySpacingConsistency(
    const std::vector<GridCorner>& corners,
    float median_spacing,
    float min_rel,
    float max_rel,
    int min_keep
) {
    if (min_rel <= 0.0f || median_spacing < 1.0f) return corners;

    const float min_d       = median_spacing * min_rel;
    const float max_d       = median_spacing * max_rel;
    const float duplicate_d = median_spacing * 0.15f;

    std::vector<GridCorner> current = corners;

    for (int pass = 0; pass < 8; ++pass) {
        const int n = static_cast<int>(current.size());
        if (n <= min_keep) break;

        std::vector<bool> keep(n, true);
        for (int i = 0; i < n; ++i) {
            bool is_duplicate          = false;
            bool has_spacing_neighbour = false;

            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                const float d = distf(current[i].uv, current[j].uv);
                if (d < duplicate_d) { is_duplicate = true; break; }
                if (d >= min_d && d <= max_d) has_spacing_neighbour = true;
            }

            if (is_duplicate || !has_spacing_neighbour) keep[i] = false;
        }

        int survive = 0;
        for (int i = 0; i < n; ++i) if (keep[i]) ++survive;
        if (survive < min_keep) break;

        bool changed = false;
        for (int i = 0; i < n; ++i)
            if (!keep[i]) { changed = true; break; }
        if (!changed) break;

        std::vector<GridCorner> next;
        next.reserve(survive);
        for (int i = 0; i < n; ++i)
            if (keep[i]) next.push_back(current[i]);
        current = std::move(next);
    }

    return current;
}


static std::vector<GridCorner> removeOutlierCorners(
    const std::vector<GridCorner>& corners,
    float q3_spacing,
    int min_keep
) {
    if (q3_spacing < 1.0f ||
        static_cast<int>(corners.size()) <= min_keep)
        return corners;

    const float outlier_max_d = q3_spacing * 0.30f;

    int   worst_idx = -1;
    float worst_min_d = std::numeric_limits<float>::max();

    for (int i = 0; i < static_cast<int>(corners.size()); ++i) {
        float min_d = std::numeric_limits<float>::max();
        for (int j = 0; j < static_cast<int>(corners.size()); ++j) {
            if (i == j) continue;
            min_d = std::min(min_d, distf(corners[i].uv, corners[j].uv));
        }
        if (min_d < outlier_max_d && min_d < worst_min_d) {
            worst_min_d = min_d;
            worst_idx   = i;
        }
    }

    if (worst_idx < 0) return corners;
    if (static_cast<int>(corners.size()) - 1 < min_keep) return corners;

    std::vector<GridCorner> result;
    result.reserve(corners.size() - 1);
    for (int i = 0; i < static_cast<int>(corners.size()); ++i)
        if (i != worst_idx) result.push_back(corners[i]);
    return result;
}


// ============================================================
// Quadrant intensity test — RECOVERY ONLY
//
// Used to filter raw saddle-point candidates during detectRecovery().
// NOT called during tracking: LK forward-backward + spacing filter
// are sufficient and far more robust to illumination changes.
// ============================================================

static bool passesQuadrantTest(
    const cv::Mat& gray_f,   // CV_32F
    const cv::Point2f& uv,
    int half_r,
    float min_contrast,
    float max_diagonal_diff
) {
    if (half_r <= 0 || gray_f.empty()) return true;

    const int margin = half_r + 1;
    if (uv.x < static_cast<float>(margin) ||
        uv.y < static_cast<float>(margin) ||
        uv.x >= static_cast<float>(gray_f.cols - margin) ||
        uv.y >= static_cast<float>(gray_f.rows - margin))
        return true;  // near border — keep

    const int cx = static_cast<int>(std::lround(uv.x));
    const int cy = static_cast<int>(std::lround(uv.y));

    const int box_r  = std::max(1, half_r / 2);
    const int offset = std::max(box_r + 1, half_r * 3 / 4);

    auto boxMean = [&](int qcx, int qcy) -> float {
        const int x0 = std::max(0, qcx - box_r);
        const int y0 = std::max(0, qcy - box_r);
        const int x1 = std::min(gray_f.cols - 1, qcx + box_r);
        const int y1 = std::min(gray_f.rows - 1, qcy + box_r);
        if (x1 < x0 || y1 < y0) return 0.0f;
        float sum = 0.0f; int cnt = 0;
        for (int y = y0; y <= y1; ++y) {
            const float* row = gray_f.ptr<float>(y);
            for (int x = x0; x <= x1; ++x) { sum += row[x]; ++cnt; }
        }
        return cnt > 0 ? sum / static_cast<float>(cnt) : 0.0f;
    };

    const float q0 = boxMean(cx - offset, cy - offset);
    const float q1 = boxMean(cx + offset, cy - offset);
    const float q2 = boxMean(cx - offset, cy + offset);
    const float q3 = boxMean(cx + offset, cy + offset);

    if (std::abs(q0 - q3) > max_diagonal_diff) return false;
    if (std::abs(q1 - q2) > max_diagonal_diff) return false;

    const float h_top    = std::abs(q0 - q1);
    const float h_bottom = std::abs(q2 - q3);
    const float v_left   = std::abs(q0 - q2);
    const float v_right  = std::abs(q1 - q3);

    const bool h_axis = (h_top >= min_contrast && h_bottom >= min_contrast);
    const bool v_axis = (v_left >= min_contrast && v_right >= min_contrast);

    return h_axis || v_axis;
}


// ============================================================
// isBetterThanTracked
// ============================================================

static bool isBetterThanTracked(
    const CheckerboardDetection& candidate,
    const CheckerboardDetection& tracked
) {
    if (!candidate.valid()) return false;
    if (!tracked.valid())   return true;

    const int cand_corners = static_cast<int>(candidate.corners.size());
    const int cand_cells   = static_cast<int>(candidate.cells.size());
    const int trk_corners  = static_cast<int>(tracked.corners.size());
    const int trk_cells    = static_cast<int>(tracked.cells.size());

    const float trk_quality  = detectionMinSpacingRatio(tracked);
    const bool  trk_degraded = trk_quality < 0.35f;

    if (trk_degraded) {
        const float cand_quality = detectionMinSpacingRatio(candidate);
        const bool  cand_healthy = cand_quality >= 0.55f;
        if (cand_healthy &&
            cand_corners >= trk_corners - 4 &&
            cand_cells   >= trk_cells   - 2)
            return true;
        if (cand_cells >= trk_cells + 3)
            return true;
    }

    if (cand_cells >= trk_cells + 8) return true;

    if (cand_corners > trk_corners && cand_cells >= trk_cells + 2)
        return true;

    if (cand_corners >= trk_corners + 12 && cand_cells >= trk_cells - 3)
        return true;

    if (!tracked.stable) {
        if (cand_cells   >= trk_cells   + 3 &&
            cand_corners >= trk_corners - 4)
            return true;
    }

    return false;
}

} // namespace


// ============================================================
// Construction / reset
// ============================================================

CheckerboardDetector::CheckerboardDetector()
    : CheckerboardDetector(CheckerboardDetectorConfig{}) {}

CheckerboardDetector::CheckerboardDetector(CheckerboardDetectorConfig config)
    : config_(config) {}


void CheckerboardDetector::resetTracking() {
    last_gray_.release();
    last_detection_ = CheckerboardDetection{};
    persistent_corners_.clear();
    tracking_active_       = false;
    frame_index_           = 0;
    degraded_frames_count_ = 0;
    low_corner_frames_     = 0;
}

bool CheckerboardDetector::isTracking() const {
    return tracking_active_;
}


// ============================================================
// Main detect() entry point
// ============================================================

std::optional<CheckerboardDetection> CheckerboardDetector::detect(
    const cv::Mat& image
) {
    const cv::Mat gray = toGray8(image);
    if (gray.empty()) { resetTracking(); return std::nullopt; }

    ++frame_index_;

    if (tracking_active_ &&
        !last_gray_.empty() &&
        !last_detection_.corners.empty()) {

        auto tracked = trackFromPreviousFrame(gray);

        if (tracked && tracked->valid()) {
            tracked->tracking = true;

            const bool refresh_due =
                config_.refresh_interval_frames > 0 &&
                (frame_index_ % config_.refresh_interval_frames == 0);

            const bool corner_loss =
                config_.refresh_corner_loss_ratio > 0.0f &&
                static_cast<int>(tracked->corners.size()) <
                static_cast<int>(last_detection_.corners.size()) *
                    config_.refresh_corner_loss_ratio;

            const bool geometry_degraded =
                detectionMinSpacingRatio(*tracked) < 0.35f;

            const bool do_refresh = refresh_due || corner_loss || geometry_degraded;

            if (do_refresh) {
                auto recovered = detectRecovery(gray);

                if (recovered && recovered->valid()) {
                    recovered->tracking = false;
                    recovered->stable   = false;

                    const bool gain_trigger =
                        config_.refresh_gain_threshold > 0 &&
                        static_cast<int>(recovered->corners.size()) >=
                        static_cast<int>(tracked->corners.size()) +
                            config_.refresh_gain_threshold;

                    if (gain_trigger || isBetterThanTracked(*recovered, *tracked)) {
                        // Recovery is clearly better — full reset to fresh detection.
                        updateTrackingState(gray, *recovered);
                        return last_detection_;
                    }

                    // Inject new corners from recovery directly into persistent
                    // state — no lattice refit, no Grid-ID loss.
                    updateTrackingState(gray, *tracked, &(*recovered));
                    return last_detection_;
                }

                // Recovery found nothing.
                if (geometry_degraded) {
                    ++degraded_frames_count_;
                    if (degraded_frames_count_ >=
                        config_.max_degraded_frames_before_reset) {
                        degraded_frames_count_ = 0;
                        resetTracking();
                        return std::nullopt;
                    }
                } else {
                    degraded_frames_count_ = 0;
                }
            }

            updateTrackingState(gray, *tracked);

            // Stuck-state detection: corners stuck below minimum for too long → reset.
            // Threshold is min_tracking_corners (not 2x) so partial visibility
            // does not trigger a reset loop.  Only reset when we have fewer
            // corners than the absolute minimum needed for tracking.
            if (static_cast<int>(last_detection_.corners.size()) <
                config_.min_tracking_corners) {
                ++low_corner_frames_;
            } else {
                low_corner_frames_ = 0;
            }

            if (low_corner_frames_ > config_.max_low_corner_frames) {
                low_corner_frames_ = 0;
                resetTracking();
                return std::nullopt;
            }

            return last_detection_;
        }

        tracking_active_ = false;
    }

    // No active tracking — run full recovery.
    auto recovered = detectRecovery(gray);

    if (recovered && recovered->valid()) {
        recovered->tracking = false;
        recovered->stable   = false;
        updateTrackingState(gray, *recovered);

        if (config_.refresh_interval_frames > 0)
            frame_index_ = config_.refresh_interval_frames - 1;

        return last_detection_;
    }

    resetTracking();
    return std::nullopt;
}


// ============================================================
// debugRecoveryStages
// ============================================================

CheckerboardRecoveryDebug CheckerboardDetector::debugRecoveryStages(
    const cv::Mat& image
) const {
    CheckerboardRecoveryDebug dbg;
    const cv::Mat gray = toGray8(image);
    if (gray.empty()) return dbg;

    cv::Mat work = gray;
    float scale  = 1.0f;

    if (config_.det_width > 0 && gray.cols > config_.det_width) {
        scale = static_cast<float>(config_.det_width) /
                static_cast<float>(gray.cols);
        const int new_w = config_.det_width;
        const int new_h = std::max(1, static_cast<int>(std::round(gray.rows * scale)));
        cv::resize(gray, work, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_AREA);
    }

    dbg.scale = scale;

    CornerDetectionResult raw = corner_detector_.detect(
        work, config_.max_recovery_corners,
        config_.saddle_sigma, config_.saddle_response_threshold);

    const float inv_scale = 1.0f / scale;
    dbg.raw_candidates.reserve(raw.points.size());
    for (const auto& p : raw.points)
        dbg.raw_candidates.emplace_back(p.x * inv_scale, p.y * inv_scale);

    if (raw.points.empty()) return dbg;

    CornerRefinementConfig refine_config;
    refine_config.radius             = config_.saddle_radius;
    refine_config.iterations         = config_.saddle_iterations;
    refine_config.max_angle_bias_deg = config_.saddle_max_angle_bias_deg;
    refine_config.correlation_drop   = config_.saddle_correlation_drop;
    refine_config.merge_radius_px    = std::max(1.0f, config_.duplicate_corner_dist_px * scale);
    refine_config.quadrant_half_r            = config_.quadrant_half_r;
    refine_config.quadrant_min_contrast      = config_.quadrant_min_contrast;
    refine_config.quadrant_max_diagonal_diff = config_.quadrant_max_diagonal_diff;
    refine_config.subpix_win_size  = config_.saddle_subpix_win_size;
    refine_config.subpix_max_iters = config_.saddle_subpix_max_iters;
    refine_config.subpix_epsilon   = config_.saddle_subpix_epsilon;

    std::vector<RefinedCorner> refined = corner_refiner_.refine(
        work, raw.points, raw.grad_x, raw.grad_y, refine_config);

    dbg.refined_corners.reserve(refined.size());
    dbg.valid_refined_points.reserve(refined.size());
    for (auto c : refined) {
        c.uv.x *= inv_scale;
        c.uv.y *= inv_scale;
        dbg.refined_corners.push_back(c);
        if (c.valid) dbg.valid_refined_points.push_back(c.uv);
    }

    if (static_cast<int>(dbg.valid_refined_points.size()) < config_.min_corners)
        return dbg;

    auto lattice = lattice_model_.fit(dbg.valid_refined_points);
    if (!lattice || !lattice->valid) return dbg;

    dbg.lattice     = *lattice;
    dbg.has_lattice = true;

    auto detection = grid_builder_.build(
        *lattice, config_.duplicate_corner_dist_px,
        config_.min_corners, config_.min_cells);

    if (!detection || !detection->valid()) return dbg;

    dbg.detection     = *detection;
    dbg.has_detection = true;

    return dbg;
}


// ============================================================
// resetTracking / toGray8
// ============================================================

cv::Mat CheckerboardDetector::toGray8(const cv::Mat& image) {
    if (image.empty()) return {};

    cv::Mat gray;
    if      (image.channels() == 1) gray = image;
    else if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if (image.channels() == 4) cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else return {};

    if (gray.type() == CV_8U) return gray.clone();

    cv::Mat gray8;
    double min_val = 0.0, max_val = 0.0;
    cv::minMaxLoc(gray, &min_val, &max_val);

    if (max_val <= min_val) {
        gray.convertTo(gray8, CV_8U);
    } else {
        gray.convertTo(gray8, CV_8U,
                       255.0 / (max_val - min_val),
                       -255.0 * min_val / (max_val - min_val));
    }
    return gray8;
}


// ============================================================
// detectRecovery
// Quadrant test applied HERE — only on fresh candidate corners,
// not on already-tracked corners.
// ============================================================

std::optional<CheckerboardDetection> CheckerboardDetector::detectRecovery(
    const cv::Mat& gray
) const {
    if (gray.empty()) return std::nullopt;

    cv::Mat work = gray;
    float scale  = 1.0f;

    if (config_.det_width > 0 && gray.cols > config_.det_width) {
        scale = static_cast<float>(config_.det_width) /
                static_cast<float>(gray.cols);
        const int new_w = config_.det_width;
        const int new_h = std::max(1, static_cast<int>(std::round(gray.rows * scale)));
        cv::resize(gray, work, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_AREA);
    }

    CornerDetectionResult raw = corner_detector_.detect(
        work, config_.max_recovery_corners,
        config_.saddle_sigma, config_.saddle_response_threshold);

    if (raw.points.empty()) return std::nullopt;

    CornerRefinementConfig refine_config;
    refine_config.radius             = config_.saddle_radius;
    refine_config.iterations         = config_.saddle_iterations;
    refine_config.max_angle_bias_deg = config_.saddle_max_angle_bias_deg;
    refine_config.correlation_drop   = config_.saddle_correlation_drop;
    refine_config.merge_radius_px    = std::max(1.0f, config_.duplicate_corner_dist_px * scale);
    refine_config.quadrant_half_r            = config_.quadrant_half_r;
    refine_config.quadrant_min_contrast      = config_.quadrant_min_contrast;
    refine_config.quadrant_max_diagonal_diff = config_.quadrant_max_diagonal_diff;
    refine_config.subpix_win_size  = config_.saddle_subpix_win_size;
    refine_config.subpix_max_iters = config_.saddle_subpix_max_iters;
    refine_config.subpix_epsilon   = config_.saddle_subpix_epsilon;

    std::vector<RefinedCorner> refined = corner_refiner_.refine(
        work, raw.points, raw.grad_x, raw.grad_y, refine_config);

    if (static_cast<int>(refined.size()) < config_.min_corners)
        return std::nullopt;

    // Apply quadrant test on the work image (recovery only).
    // This is the single place where passesQuadrantTest() is called.
    cv::Mat work_f;
    work.convertTo(work_f, CV_32F);

    const float inv_scale = 1.0f / scale;
    std::vector<cv::Point2f> corners;
    corners.reserve(refined.size());

    for (const auto& c : refined) {
        if (!c.valid) continue;

        // Quadrant test in work-image coordinates.
        if (config_.quadrant_half_r > 0 &&
            !passesQuadrantTest(work_f, c.uv,
                                config_.quadrant_half_r,
                                config_.quadrant_min_contrast,
                                config_.quadrant_max_diagonal_diff))
            continue;

        corners.emplace_back(c.uv.x * inv_scale, c.uv.y * inv_scale);
    }

    if (static_cast<int>(corners.size()) < config_.min_corners)
        return std::nullopt;

    auto detection = buildDetectionFromCorners(corners);
    if (!detection || !detection->valid()) return std::nullopt;

    // Lattice-guided completion: fill weak corners inside AND on the boundary
    // of the detected grid using raw.points candidates (already computed, free).
    //
    // The loop extends one step BEYOND the detected bbox (min_i-1..max_i+1,
    // min_j-1..max_j+1) so that physical border corners — which are real saddle
    // points but have lower saddle scores because their patch overlaps the
    // featureless white background — are picked up here instead of relying on
    // the saddle filter.
    //
    // Interior missing corners use interpolation (two opposing neighbours).
    // Border missing corners use extrapolation (one neighbour + lattice step).
    // Corners that extend beyond the image are skipped silently.
    {
        const float spacing = estimateMedianSpacing(*detection);

        if (spacing > 2.0f) {
            int min_i = std::numeric_limits<int>::max();
            int max_i = std::numeric_limits<int>::min();
            int min_j = std::numeric_limits<int>::max();
            int max_j = std::numeric_limits<int>::min();

            for (const auto& c : detection->corners) {
                min_i = std::min(min_i, c.i); max_i = std::max(max_i, c.i);
                min_j = std::min(min_j, c.j); max_j = std::max(max_j, c.j);
            }

            std::vector<std::pair<std::pair<int,int>, cv::Point2f>> by_ij;
            by_ij.reserve(detection->corners.size());
            for (const auto& c : detection->corners)
                by_ij.push_back({{c.i, c.j}, c.uv});

            auto findUv = [&](int i, int j) -> const cv::Point2f* {
                for (const auto& p : by_ij)
                    if (p.first.first == i && p.first.second == j)
                        return &p.second;
                return nullptr;
            };

            std::vector<cv::Point2f> detected_uvs;
            detected_uvs.reserve(corners.size());
            for (const auto& uv : corners) detected_uvs.push_back(uv);

            auto hasNearDetected = [&](const cv::Point2f& pt) -> bool {
                const float r = spacing * 0.4f;
                for (const auto& p : detected_uvs)
                    if (distf(p, pt) < r) return true;
                return false;
            };

            const float search_r_work = spacing * scale * 0.55f;
            std::vector<cv::Point2f> new_corners = corners;

            for (int gi = min_i; gi <= max_i; ++gi) {
                for (int gj = min_j; gj <= max_j; ++gj) {
                    if (findUv(gi, gj)) continue;

                    const cv::Point2f* p_im1 = findUv(gi - 1, gj);
                    const cv::Point2f* p_ip1 = findUv(gi + 1, gj);
                    const cv::Point2f* p_jm1 = findUv(gi, gj - 1);
                    const cv::Point2f* p_jp1 = findUv(gi, gj + 1);

                    cv::Point2f expected(0.0f, 0.0f);
                    int count = 0;

                    if (p_im1 && p_ip1) { expected += 0.5f * (*p_im1 + *p_ip1); ++count; }
                    if (p_jm1 && p_jp1) { expected += 0.5f * (*p_jm1 + *p_jp1); ++count; }
                    if (count == 0) continue;

                    expected *= 1.0f / static_cast<float>(count);
                    const cv::Point2f expected_work = expected * scale;

                    if (expected_work.x < 4.0f || expected_work.y < 4.0f) continue;
                    if (expected_work.x >= static_cast<float>(work.cols) - 4.0f) continue;
                    if (expected_work.y >= static_cast<float>(work.rows) - 4.0f) continue;
                    if (hasNearDetected(expected)) continue;

                    float best_d = search_r_work;
                    cv::Point2f best_pt(-1.0f, -1.0f);

                    for (const auto& cpt : raw.points) {
                        const float d = distf(cpt, expected_work);
                        if (d < best_d) { best_d = d; best_pt = cpt * inv_scale; }
                    }

                    if (best_pt.x >= 0.0f && !hasNearDetected(best_pt)) {
                        new_corners.push_back(best_pt);
                        detected_uvs.push_back(best_pt);
                    }
                }
            }

            if (new_corners.size() > corners.size()) {
                auto completed = buildDetectionFromCorners(new_corners);
                if (completed && completed->valid())
                    detection = std::move(completed);
            }
        }
    }

    detection->tracking = false;
    detection->stable   = false;
    return detection;
}


// ============================================================
// buildDetectionFromCorners
// ============================================================

std::optional<CheckerboardDetection>
CheckerboardDetector::buildDetectionFromCorners(
    const std::vector<cv::Point2f>& corners
) const {
    if (static_cast<int>(corners.size()) < config_.min_corners)
        return std::nullopt;

    auto lattice = lattice_model_.fit(corners);
    if (!lattice || !lattice->valid) return std::nullopt;

    auto detection = grid_builder_.build(
        *lattice, config_.duplicate_corner_dist_px,
        config_.min_corners, config_.min_cells);

    if (!detection || !detection->valid()) return std::nullopt;
    return detection;
}


// ============================================================
// buildVisibleTrackedDetection
// No quadrant test — LK + spacing filter is sufficient.
// ============================================================

std::optional<CheckerboardDetection>
CheckerboardDetector::buildVisibleTrackedDetection(
    const CheckerboardDetection& previous,
    const TrackingValidationResult& validation
) const {
    if (validation.visible_indices.size() != validation.visible_points.size())
        return std::nullopt;

    if (static_cast<int>(validation.visible_points.size()) <
        config_.min_tracking_corners)
        return std::nullopt;

    std::vector<GridCorner> tracked_corners;
    tracked_corners.reserve(validation.visible_points.size());

    for (size_t m = 0; m < validation.visible_indices.size(); ++m) {
        const int old_idx = validation.visible_indices[m];
        if (old_idx < 0 || old_idx >= static_cast<int>(previous.corners.size()))
            continue;

        const cv::Point2f& uv = validation.visible_points[m];
        constexpr float kBorderMargin = 4.0f;
        if (uv.x < kBorderMargin || uv.y < kBorderMargin) continue;

        GridCorner c = previous.corners[old_idx];
        c.uv = uv;
        tracked_corners.push_back(c);
    }

    if (static_cast<int>(tracked_corners.size()) < config_.min_tracking_corners)
        return std::nullopt;

    // Spacing consistency filter (geometric — not photometric).
    if (config_.tracking_spacing_min_rel > 0.0f) {
        const float q3_spacing = estimateMedianSpacing(previous);
        if (q3_spacing > 1.0f) {
            tracked_corners = removeOutlierCorners(
                tracked_corners, q3_spacing, config_.min_tracking_corners);
            tracked_corners = filterBySpacingConsistency(
                tracked_corners, q3_spacing,
                config_.tracking_spacing_min_rel,
                config_.tracking_spacing_max_rel,
                config_.min_tracking_corners);
        }
    }

    if (static_cast<int>(tracked_corners.size()) < config_.min_tracking_corners)
        return std::nullopt;

    auto rebuilt = grid_builder_.buildFromCorners(
        tracked_corners,
        config_.duplicate_corner_dist_px,
        config_.min_tracking_corners,
        config_.min_tracking_cells,
        true,
        validation.stable);

    if (!rebuilt || !rebuilt->valid()) return std::nullopt;

    rebuilt->tracking = true;
    rebuilt->stable   = validation.stable;
    return rebuilt;
}


// ============================================================
// trackFromPreviousFrame
// ============================================================

std::optional<CheckerboardDetection>
CheckerboardDetector::trackFromPreviousFrame(const cv::Mat& gray) {
    if (!tracking_active_ ||
        last_gray_.empty() ||
        last_detection_.corners.empty())
        return std::nullopt;

    std::vector<cv::Point2f> prev_points;
    prev_points.reserve(last_detection_.corners.size());
    for (const auto& c : last_detection_.corners)
        prev_points.push_back(c.uv);

    LKTrackingResult lk = lk_tracker_.track(
        last_gray_, gray, prev_points,
        config_.lk_win_size, config_.lk_max_level,
        config_.lk_max_iters, config_.lk_epsilon,
        config_.max_lk_error);

    TrackingValidationResult validation =
        tracking_validator_.validate(
            last_detection_, lk, gray.size(), config_);

    if (!validation.valid) return std::nullopt;

    // Cull right/bottom boundary.
    {
        const float max_x = static_cast<float>(gray.cols) - 4.0f;
        const float max_y = static_cast<float>(gray.rows) - 4.0f;

        std::vector<int>         culled_indices;
        std::vector<cv::Point2f> culled_points;
        culled_indices.reserve(validation.visible_indices.size());
        culled_points .reserve(validation.visible_points.size());

        for (size_t k = 0; k < validation.visible_points.size(); ++k) {
            const cv::Point2f& uv = validation.visible_points[k];
            if (uv.x >= max_x || uv.y >= max_y) continue;
            culled_indices.push_back(validation.visible_indices[k]);
            culled_points .push_back(uv);
        }

        validation.visible_indices = std::move(culled_indices);
        validation.visible_points  = std::move(culled_points);
    }

    auto detection = buildVisibleTrackedDetection(last_detection_, validation);
    if (!detection || !detection->valid()) return std::nullopt;
    return detection;
}


// ============================================================
// updateTrackingState
// Simplified: no quadrant scoring, no sharp-frame reference.
// Persistent corners are purely LK-based bookkeeping.
// ============================================================

void CheckerboardDetector::updateTrackingState(
    const cv::Mat& gray,
    const CheckerboardDetection& measured_detection,
    const CheckerboardDetection* recovery_detection
) {
    last_gray_ = gray.clone();

    if (!measured_detection.valid()) {
        last_detection_ = CheckerboardDetection{};
        persistent_corners_.clear();
        tracking_active_ = false;
        return;
    }

    // First detection or full reset (non-tracking): initialise persistent set.
    if (!tracking_active_ || persistent_corners_.empty() ||
        !measured_detection.tracking) {
        persistent_corners_.clear();
        persistent_corners_.reserve(measured_detection.corners.size());

        for (const auto& c : measured_detection.corners) {
            PersistentTrackedCorner pc;
            pc.corner        = c;
            pc.missed_frames = 0;
            pc.tracked       = true;
            persistent_corners_.push_back(pc);
        }

        last_detection_  = measured_detection;
        tracking_active_ = last_detection_.valid();
        return;
    }

    // Tracking update: mark all as missed, then update matched ones.
    for (auto& pc : persistent_corners_) {
        pc.missed_frames += 1;
        pc.tracked = false;
    }

    for (const auto& c : measured_detection.corners) {
        const int idx = findPersistentCornerByGrid(persistent_corners_, c.i, c.j);
        if (idx >= 0) {
            persistent_corners_[idx].corner        = c;
            persistent_corners_[idx].missed_frames = 0;
            persistent_corners_[idx].tracked       = true;
        } else {
            // New corner appeared (e.g. after rotation reveals hidden area).
            PersistentTrackedCorner pc;
            pc.corner        = c;
            pc.missed_frames = 0;
            pc.tracked       = true;
            persistent_corners_.push_back(pc);
        }
    }

    // Fast eviction for geometrically inconsistent corners.
    //
    // Problem: when a neighbour corner is correctly evicted, the corner next
    // to it loses its interpolation check and falls into the single-neighbour
    // spacing band [0.50, 1.30] * Q75.  Under strong perspective rotation this
    // band is too loose and the stale corner survives for max_missed_frames
    // more frames before finally being evicted.
    //
    // Fix: any persistent corner with missed_frames > 0 (LK lost it or spacing
    // filter rejected it this frame) that also has NO grid-adjacent neighbour
    // currently tracked (missed_frames == 0) gets immediately bumped to
    // max_missed_frames so it is evicted in the erase step below.
    //
    // "Grid-adjacent" means |Δi| + |Δj| == 1 (the four axis-aligned
    // neighbours), matching the connectivity used by the lattice model.
    if (config_.tracking_spacing_min_rel > 0.0f) {
        for (auto& pc : persistent_corners_) {
            if (pc.missed_frames == 0) continue;  // actively tracked — fine

            bool has_tracked_neighbour = false;
            for (const auto& nb : persistent_corners_) {
                if (nb.missed_frames != 0) continue;
                const int di = std::abs(nb.corner.i - pc.corner.i);
                const int dj = std::abs(nb.corner.j - pc.corner.j);
                if (di + dj == 1) {
                    has_tracked_neighbour = true;
                    break;
                }
            }

            if (!has_tracked_neighbour) {
                pc.missed_frames = config_.max_missed_frames + 1;
            }
        }
    }

    // Photometric visibility eviction.
    //
    // For every corner that LK is still actively tracking (missed_frames==0),
    // compute a checkerboard-contrast score along the local grid axes derived
    // from active neighbours.  If the score falls below the threshold the
    // corner has rotated to the back of the cylinder and is no longer
    // photometrically a checkerboard crossing — evict immediately.
    //
    // This runs AFTER the geometric fast-eviction above so that isolated
    // corners (no neighbours) are already handled and the axis estimation
    // here can rely on a clean neighbour set.
    if (config_.visibility_evict_threshold > 0.0f) {
        const float spacing = estimateMedianSpacing(measured_detection);
        if (spacing >= config_.visibility_min_spacing) {
            const float alpha = config_.visibility_smoothing_alpha;
            for (auto& pc : persistent_corners_) {
                if (pc.missed_frames != 0) continue;

                pc.visibility_score = computeCornerVisibilityScore(gray, pc, spacing);

                // EMA smoothing: damps single-frame dips from triggering
                // eviction while still reacting to genuine fade-out.
                pc.smoothed_visibility_score =
                    alpha * pc.visibility_score +
                    (1.0f - alpha) * pc.smoothed_visibility_score;

                if (pc.smoothed_visibility_score < config_.visibility_evict_threshold) {
                    pc.missed_frames = config_.max_missed_frames + 1;
                }
            }
        }
    }

    // Evict corners that have been missed too long.
    persistent_corners_.erase(
        std::remove_if(
            persistent_corners_.begin(),
            persistent_corners_.end(),
            [this](const PersistentTrackedCorner& pc) {
                return pc.missed_frames > config_.max_missed_frames;
            }
        ),
        persistent_corners_.end()
    );

    // Inject new corners from recovery into persistent state.
    // This is the main mechanism for picking up corners that became newly
    // visible after rotation or were missed by the initial detection.
    // We inject AFTER the LK update so that positions come from recovery
    // (which ran on the current frame) not from stale predictions.
    if (recovery_detection && recovery_detection->valid()) {
        const float spacing = estimateMedianSpacing(measured_detection);
        if (spacing > 1.0f) {
            injectRecoveryCorners(*recovery_detection, spacing);
        }
    }

    last_detection_ = buildDetectionFromPersistent(
        measured_detection.tracking, measured_detection.stable);

    if (!last_detection_.valid())
        last_detection_ = measured_detection;

    tracking_active_ = last_detection_.valid();
}


// ============================================================
// injectRecoveryCorners
//
// For each corner in recovery_detection, check whether it is already
// represented in persistent_corners_ (by grid ID or by proximity).
// If not, add it with missed_frames=0 so it appears in the next
// buildDetectionFromPersistent() output.
//
// The grid ID from recovery is trusted because recovery runs a full
// lattice fit on the current frame. Proximity guard (0.6 * spacing)
// prevents double-entries when a tracked corner and a recovery corner
// land on the same physical corner.
// ============================================================

// ============================================================
// injectRecoveryCorners
//
// Two behaviours depending on whether the recovery corner matches
// an existing persistent corner:
//
// A) Grid-ID match with active corner (missed_frames==0):
//    Blend the LK position toward the recovery position using
//    recovery_correction_weight.  Recovery ran cornerSubPix on the
//    current frame and is more accurate than accumulated LK drift.
//    Only applied when the distance is within
//    recovery_correction_max_dist_rel * spacing (sanity guard).
//
// B) Grid-ID match with stale corner (missed_frames>0):
//    Skip — the stale proximity guard already blocks re-injection;
//    the eviction path will handle it.
//
// C) No match by grid ID and not too close to any active corner:
//    Inject as a new persistent corner (original behaviour).
// ============================================================

void CheckerboardDetector::injectRecoveryCorners(
    const CheckerboardDetection& recovery_detection,
    float spacing
) {
    const float min_dist     = spacing * 0.6f;
    const float max_corr_d   = spacing * config_.recovery_correction_max_dist_rel;
    const float w            = config_.recovery_correction_weight;

    for (const auto& rc : recovery_detection.corners) {

        // --- Fix B: position correction for actively tracked corners ---
        const int existing_idx =
            findPersistentCornerByGrid(persistent_corners_, rc.i, rc.j);

        if (existing_idx >= 0) {
            auto& pc = persistent_corners_[existing_idx];

            // Only correct active corners — stale ones are handled by eviction.
            if (pc.missed_frames == 0 && w > 0.0f) {
                const float d = distf(pc.corner.uv, rc.uv);
                if (d < max_corr_d) {
                    // Blend LK position toward recovery position.
                    pc.corner.uv = (1.0f - w) * pc.corner.uv + w * rc.uv;
                }
            }
            continue;
        }

        // --- Fix C: inject new corners not yet in persistent set ---
        // Only check against active persistent corners to avoid stale
        // corners blocking newly visible ones.
        bool too_close = false;
        for (const auto& pc : persistent_corners_) {
            if (pc.missed_frames > 0) continue;  // ignore stale
            if (distf(pc.corner.uv, rc.uv) < min_dist) {
                too_close = true;
                break;
            }
        }
        if (too_close) continue;

        PersistentTrackedCorner pc;
        pc.corner        = rc;
        pc.missed_frames = 0;
        pc.tracked       = true;
        persistent_corners_.push_back(pc);
    }
}




// ============================================================
// buildDetectionFromPersistent
// Only uses corners that were tracked in the current frame
// (missed_frames == 0). Missed corners are held in the set for
// max_missed_frames but never emitted to the outside world.
// ============================================================

CheckerboardDetection CheckerboardDetector::buildDetectionFromPersistent(
    bool tracking,
    bool stable
) const {
    std::vector<GridCorner> visible_corners;
    visible_corners.reserve(persistent_corners_.size());

    for (const auto& pc : persistent_corners_) {
        if (pc.missed_frames != 0) continue;
        GridCorner gc = pc.corner;
        gc.visibility_score = pc.smoothed_visibility_score;
        visible_corners.push_back(gc);
    }

    auto rebuilt = grid_builder_.buildFromCorners(
        visible_corners,
        config_.duplicate_corner_dist_px,
        tracking ? config_.min_tracking_corners : config_.min_corners,
        tracking ? config_.min_tracking_cells   : config_.min_cells,
        tracking,
        stable);

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


// ============================================================
// mergeMeasuredDetections
// ============================================================

std::optional<CheckerboardDetection> CheckerboardDetector::mergeMeasuredDetections(
    const CheckerboardDetection& primary,
    const CheckerboardDetection& secondary,
    float duplicate_dist_px
) const {
    const float r = std::max(2.0f, duplicate_dist_px);

    if (primary.tracking) {
        std::vector<cv::Point2f> tracked_uvs;
        tracked_uvs.reserve(primary.corners.size());
        for (const auto& c : primary.corners)
            tracked_uvs.push_back(c.uv);

        std::vector<cv::Point2f> new_uvs;
        for (const auto& c : secondary.corners)
            if (!hasNearbyPoint(tracked_uvs, c.uv, r))
                new_uvs.push_back(c.uv);

        if (new_uvs.empty()) return primary;

        std::vector<cv::Point2f> all_uvs = tracked_uvs;
        for (const auto& uv : new_uvs) all_uvs.push_back(uv);

        auto rebuilt = buildDetectionFromCorners(all_uvs);
        if (!rebuilt || !rebuilt->valid()) return primary;

        const int min_keep =
            static_cast<int>(primary.corners.size()) -
            std::max(2, static_cast<int>(primary.corners.size()) / 6);

        if (static_cast<int>(rebuilt->corners.size()) < min_keep)
            return primary;

        rebuilt->tracking = true;
        rebuilt->stable   = primary.stable;
        return rebuilt;
    }

    std::vector<cv::Point2f> merged_points;
    merged_points.reserve(primary.corners.size() + secondary.corners.size());

    for (const auto& c : primary.corners)
        if (!hasNearbyPoint(merged_points, c.uv, r))
            merged_points.push_back(c.uv);

    for (const auto& c : secondary.corners)
        if (!hasNearbyPoint(merged_points, c.uv, r))
            merged_points.push_back(c.uv);

    auto rebuilt = buildDetectionFromCorners(merged_points);
    if (!rebuilt || !rebuilt->valid()) return std::nullopt;

    rebuilt->tracking = primary.tracking;
    rebuilt->stable   = primary.stable;
    return rebuilt;
}


// ============================================================
// Static helpers
// ============================================================

int CheckerboardDetector::findPersistentCornerByGrid(
    const std::vector<PersistentTrackedCorner>& corners, int i, int j
) {
    for (int idx = 0; idx < static_cast<int>(corners.size()); ++idx)
        if (corners[idx].corner.i == i && corners[idx].corner.j == j)
            return idx;
    return -1;
}

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
        if (d <= best_d2) { best_d2 = d; best_idx = idx; }
    }
    return best_idx;
}

bool CheckerboardDetector::hasNearbyPoint(
    const std::vector<cv::Point2f>& points,
    const cv::Point2f& uv,
    float radius_px
) {
    const float r2 = radius_px * radius_px;
    for (const auto& p : points)
        if (dist2(p, uv) <= r2) return true;
    return false;
}


// ============================================================
// computeCornerVisibilityScore
//
// Computes a photometric checkerboard-contrast score in [0,1].
//
// Approach (Option B — neighbour-derived axes):
//   1. Find up to two active grid neighbours of pc in persistent_corners_:
//      the (i+1,j) neighbour gives axis_u, the (i,j+1) neighbour gives axis_v.
//      If only one axis is available we use the perpendicular as the other.
//      If no neighbour is available we return 0 (isolated corner — handled
//      by the existing fast-eviction rule).
//   2. Sample four quadrant mean intensities displaced by
//      ±visibility_sample_rel * spacing along axis_u and axis_v.
//   3. Score = max adjacent-pair contrast / local_range, normalised to [0,1].
//      A perfect checkerboard corner scores ~1.0; a featureless or
//      back-side corner scores near 0.
// ============================================================

float CheckerboardDetector::computeCornerVisibilityScore(
    const cv::Mat& gray,
    const PersistentTrackedCorner& pc,
    float spacing
) const {
    // Minimum spacing guard — avoids noisy scores on tiny markers.
    if (spacing < config_.visibility_min_spacing) return 1.0f;

    const cv::Point2f& uv = pc.corner.uv;

    // ---- Step 1: derive grid axes from active neighbours ----
    cv::Point2f axis_u(0.0f, 0.0f);
    cv::Point2f axis_v(0.0f, 0.0f);
    bool has_u = false;
    bool has_v = false;

    for (const auto& nb : persistent_corners_) {
        if (nb.missed_frames != 0) continue;  // only active neighbours

        const int di = nb.corner.i - pc.corner.i;
        const int dj = nb.corner.j - pc.corner.j;

        if (!has_u && std::abs(di) == 1 && dj == 0) {
            cv::Point2f v = nb.corner.uv - uv;
            const float n = std::sqrt(v.x * v.x + v.y * v.y);
            if (n > 1.0f) {
                axis_u = v * (1.0f / n);
                // Normalise direction: always point toward increasing i.
                if (di < 0) axis_u = -axis_u;
                has_u = true;
            }
        }

        if (!has_v && di == 0 && std::abs(dj) == 1) {
            cv::Point2f v = nb.corner.uv - uv;
            const float n = std::sqrt(v.x * v.x + v.y * v.y);
            if (n > 1.0f) {
                axis_v = v * (1.0f / n);
                if (dj < 0) axis_v = -axis_v;
                has_v = true;
            }
        }

        if (has_u && has_v) break;
    }

    // If we have one axis, derive the other as its perpendicular.
    if (has_u && !has_v) {
        axis_v = cv::Point2f(-axis_u.y, axis_u.x);
        has_v  = true;
    } else if (has_v && !has_u) {
        axis_u = cv::Point2f(axis_v.y, -axis_v.x);
        has_u  = true;
    }

    // No neighbours at all — isolated corner, fast-eviction handles it.
    if (!has_u || !has_v) return 0.0f;

    // ---- Step 2: sample four quadrant means ----
    const float offset  = spacing * config_.visibility_sample_rel;
    const float box_r   = spacing * config_.visibility_box_rel;
    const int   box_r_i = std::max(1, static_cast<int>(std::round(box_r)));

    // Quadrant centres (in image coordinates):
    //   Q0: +axis_u  +axis_v  (top-right in grid space)
    //   Q1: -axis_u  +axis_v  (top-left)
    //   Q2: +axis_u  -axis_v  (bottom-right)
    //   Q3: -axis_u  -axis_v  (bottom-left)
    // Opposite pairs (Q0,Q3) and (Q1,Q2) should have similar intensity;
    // adjacent pairs should differ — the classic checkerboard pattern.
    const cv::Point2f centres[4] = {
        uv + offset * axis_u + offset * axis_v,
        uv - offset * axis_u + offset * axis_v,
        uv + offset * axis_u - offset * axis_v,
        uv - offset * axis_u - offset * axis_v
    };

    // Box-mean sampler on CV_8U gray image.
    auto boxMean = [&](const cv::Point2f& c) -> float {
        const int cx = static_cast<int>(std::lround(c.x));
        const int cy = static_cast<int>(std::lround(c.y));
        const int x0 = std::max(0, cx - box_r_i);
        const int y0 = std::max(0, cy - box_r_i);
        const int x1 = std::min(gray.cols - 1, cx + box_r_i);
        const int y1 = std::min(gray.rows - 1, cy + box_r_i);
        if (x1 < x0 || y1 < y0) return -1.0f;  // out of image
        float sum = 0.0f; int cnt = 0;
        for (int y = y0; y <= y1; ++y) {
            const uchar* row = gray.ptr<uchar>(y);
            for (int x = x0; x <= x1; ++x) { sum += row[x]; ++cnt; }
        }
        return cnt > 0 ? sum / static_cast<float>(cnt) : -1.0f;
    };

    float q[4];
    for (int k = 0; k < 4; ++k) {
        q[k] = boxMean(centres[k]);
        if (q[k] < 0.0f) return 1.0f;  // sample out of image — keep corner
    }

    // ---- Step 3: compute score ----
    const float local_min = std::min({q[0], q[1], q[2], q[3]});
    const float local_max = std::max({q[0], q[1], q[2], q[3]});
    const float local_range = local_max - local_min;

    // Completely flat region — no checkerboard structure.
    if (local_range < 2.0f) return 0.0f;

    // Normalise to [0,1].
    const float inv = 1.0f / local_range;
    const float n0 = (q[0] - local_min) * inv;
    const float n1 = (q[1] - local_min) * inv;
    const float n2 = (q[2] - local_min) * inv;
    const float n3 = (q[3] - local_min) * inv;

    // Adjacent contrast: max over all four axis-crossing pairs.
    // For a perfect checkerboard this equals 1.0 (one pair is 0 vs 1).
    // For a back-side corner with uniform dark/light it approaches 0.
    const float adj_score = std::max({
        std::abs(n0 - n1),   // same v, different u
        std::abs(n2 - n3),
        std::abs(n0 - n2),   // same u, different v
        std::abs(n1 - n3)
    });

    // Diagonal consistency penalty: opposite quadrants should be similar.
    // If they differ strongly, the corner is not a clean checkerboard crossing.
    const float diag_penalty = std::max(
        std::abs(n0 - n3),   // Q0 vs Q3 (opposite)
        std::abs(n1 - n2)    // Q1 vs Q2 (opposite)
    );

    // Final score: high adjacent contrast AND low diagonal difference → 1.0.
    // Back-side corner: adj_score small → score near 0.
    // Dot centre (all dark): local_range small → already caught above.
    const float score = std::max(0.0f, adj_score - diag_penalty);
    return std::min(1.0f, score);
}
} // namespace hydramarker