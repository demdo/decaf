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

    // Lattice-guided completion: fill weak corners inside the detected bbox
    // using raw.points candidates (already computed, free).
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

    // ----------------------------------------------------------------
    // Grid-consistency filter — the authoritative per-corner validity check.
    //
    // Strategy: use only corners that have both axis-aligned neighbours
    // (interpolation possible) as "anchor" corners. For each anchor, compute
    // the interpolated expected position and accept only if the LK position
    // is within pred_threshold of it.
    //
    // For non-anchor corners (boundary, only one neighbour per axis):
    // check that the distance to each existing neighbour is within a tight
    // window relative to the Q75 spacing of the anchor set. This rejects
    // corners that have been pulled too close (perspective artifact) or too
    // far (drifted off marker).
    //
    // Reference: Q75 of anchor-set spacings — dominated by well-visible
    // uncompressed corners in the marker interior.
    // ----------------------------------------------------------------

    // Build fast UV lookup for the current tracked_corners set.
    auto makeUvMap = [](const std::vector<GridCorner>& cs) {
        std::vector<std::pair<std::pair<int,int>, cv::Point2f>> m;
        m.reserve(cs.size());
        for (const auto& c : cs)
            m.push_back({{c.i, c.j}, c.uv});
        return m;
    };

    auto findInMap = [](
        const std::vector<std::pair<std::pair<int,int>, cv::Point2f>>& m,
        int i, int j) -> const cv::Point2f*
    {
        for (const auto& p : m)
            if (p.first.first == i && p.first.second == j)
                return &p.second;
        return nullptr;
    };

    // Compute Q75 spacing from the full tracked set.
    float q75_spacing = 0.0f;
    {
        auto uv_map = makeUvMap(tracked_corners);
        std::vector<float> dists;
        dists.reserve(tracked_corners.size() * 2);
        for (const auto& c : tracked_corners) {
            const cv::Point2f* nb;
            nb = findInMap(uv_map, c.i + 1, c.j);
            if (nb) dists.push_back(distf(c.uv, *nb));
            nb = findInMap(uv_map, c.i, c.j + 1);
            if (nb) dists.push_back(distf(c.uv, *nb));
        }
        if (dists.size() >= 4) {
            std::sort(dists.begin(), dists.end());
            q75_spacing = dists[dists.size() * 3 / 4];
        } else if (!dists.empty()) {
            std::sort(dists.begin(), dists.end());
            q75_spacing = dists[dists.size() / 2];
        }
    }

    if (q75_spacing > 1.0f) {
        // Strict thresholds relative to Q75 (best-visible corners):
        // pred: max deviation from interpolated grid position.
        // nb_lo/hi: allowed distance to a single axis-aligned neighbour.
        const float pred   = q75_spacing * 0.22f;
        const float nb_lo  = q75_spacing * 0.48f;
        const float nb_hi  = q75_spacing * 1.28f;

        std::vector<GridCorner> accepted;
        accepted.reserve(tracked_corners.size());

        auto uv_map = makeUvMap(tracked_corners);

        for (const auto& c : tracked_corners) {
            const cv::Point2f* p_im1 = findInMap(uv_map, c.i - 1, c.j);
            const cv::Point2f* p_ip1 = findInMap(uv_map, c.i + 1, c.j);
            const cv::Point2f* p_jm1 = findInMap(uv_map, c.i, c.j - 1);
            const cv::Point2f* p_jp1 = findInMap(uv_map, c.i, c.j + 1);

            // --- Interpolation check (both neighbours on same axis) ---
            cv::Point2f predicted(0.0f, 0.0f);
            int n_pred = 0;
            if (p_im1 && p_ip1) { predicted += 0.5f * (*p_im1 + *p_ip1); ++n_pred; }
            if (p_jm1 && p_jp1) { predicted += 0.5f * (*p_jm1 + *p_jp1); ++n_pred; }

            if (n_pred > 0) {
                predicted *= 1.0f / static_cast<float>(n_pred);
                if (distf(c.uv, predicted) <= pred) {
                    accepted.push_back(c);
                }
                // If interpolation possible but fails → reject (no fallback).
                continue;
            }

            // --- Single-neighbour distance check (boundary corners) ---
            // All available neighbours must be within [nb_lo, nb_hi].
            const cv::Point2f* neighbours[4] = {p_im1, p_ip1, p_jm1, p_jp1};
            float min_d = std::numeric_limits<float>::max();
            int   n_nb  = 0;
            bool  nb_fail = false;

            for (const auto* nb : neighbours) {
                if (!nb) continue;
                const float d = distf(c.uv, *nb);
                min_d = std::min(min_d, d);
                ++n_nb;
                if (d < nb_lo || d > nb_hi) { nb_fail = true; }
            }

            if (n_nb == 0) {
                // Truly isolated — keep only if Q75 not yet established
                // (early frames). Otherwise reject: no evidence it belongs
                // to the grid.
                continue;
            }

            if (!nb_fail) {
                accepted.push_back(c);
            }
        }

        if (static_cast<int>(accepted.size()) >= config_.min_tracking_corners) {
            tracked_corners = std::move(accepted);
        }
        // If filter leaves too few, keep original set (safety fallback).
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

    // Sub-pixel refinement for all active corners.
    // LK optical flow gives integer-accurate positions; cornerSubPix
    // refines to sub-pixel accuracy every frame.  This prevents gradual
    // position drift and ensures corners snap to the correct saddle point
    // rather than sticking to a slightly wrong position.
    if (!gray.empty()) {
        // Collect active corner positions.
        std::vector<cv::Point2f> pts;
        std::vector<int>         active_idx;
        pts.reserve(persistent_corners_.size());
        active_idx.reserve(persistent_corners_.size());

        for (int k = 0; k < static_cast<int>(persistent_corners_.size()); ++k) {
            if (persistent_corners_[k].missed_frames != 0) continue;
            pts.push_back(persistent_corners_[k].corner.uv);
            active_idx.push_back(k);
        }

        if (!pts.empty()) {
            // Window size: ~1/4 of expected spacing but at least 3px.
            // We don't know spacing yet at this point so use a fixed
            // reasonable default; tryCompleteMissingCorners uses spacing-
            // adaptive sizing.
            const cv::Size win(5, 5);
            const cv::TermCriteria crit(
                cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 15, 0.05);

            std::vector<cv::Point2f> refined = pts;
            cv::cornerSubPix(gray, refined, win, cv::Size(-1,-1), crit);

            // Accept refined position only if it didn't move more than
            // a small amount (prevents convergence to wrong feature).
            // 4px is generous enough for LK drift but tight enough to
            // reject jumps to neighbouring corners or dot centres.
            constexpr float kMaxMove = 4.0f;
            for (int m = 0; m < static_cast<int>(active_idx.size()); ++m) {
                if (distf(refined[m], pts[m]) < kMaxMove) {
                    persistent_corners_[active_idx[m]].corner.uv = refined[m];
                }
            }
        }
    }

    // Eviction: remove corners that are out-of-image.
    // Grid-consistency filtering is now done in buildVisibleTrackedDetection
    // before corners reach the persistent state, so only a simple boundary
    // check is needed here.
    if (!gray.empty()) {
        const float max_x = static_cast<float>(gray.cols) - 1.0f;
        const float max_y = static_cast<float>(gray.rows) - 1.0f;
        for (auto& pc : persistent_corners_) {
            if (pc.missed_frames != 0) continue;
            const cv::Point2f& uv = pc.corner.uv;
            if (uv.x < 0.0f || uv.y < 0.0f || uv.x > max_x || uv.y > max_y)
                pc.missed_frames = config_.max_missed_frames + 1;
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

    // Lattice-guided corner completion from the current gray frame.
    // After LK tracking + recovery injection, some grid slots may still be
    // empty (corners that just became visible or were missed by both LK and
    // recovery).  We search for them by interpolating their expected position
    // from known neighbours and looking for a gradient-junction candidate
    // nearby in the current frame.
    if (!gray.empty()) {
        tryCompleteMissingCorners(gray, measured_detection.tracking);
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

void CheckerboardDetector::injectRecoveryCorners(
    const CheckerboardDetection& recovery_detection,
    float spacing
) {
    // Proximity threshold: only reject if a persistent corner is very close.
    // 0.35 * spacing (not 0.6) so that corners adjacent on the grid are not
    // falsely suppressed — adjacent corners are ~1.0 * spacing apart, so
    // 0.35 leaves a safe margin while still blocking true duplicates.
    const float min_dist = spacing * 0.35f;

    for (const auto& rc : recovery_detection.corners) {
        // Skip if already tracked by grid ID.
        if (findPersistentCornerByGrid(persistent_corners_, rc.i, rc.j) >= 0)
            continue;

        // Skip if a persistent corner occupies the same grid slot (different
        // UV but same logical position — can happen after a partial reset).
        // Also skip if physically too close to any persistent corner.
        bool too_close = false;
        for (const auto& pc : persistent_corners_) {
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
// tryCompleteMissingCorners
//
// Finds missing grid corners every frame directly from the image,
// without relying on a full Recovery detection.
//
// For each empty grid slot within (and just outside) the bounding box
// of currently visible corners:
//   1. Interpolate the expected UV from grid neighbours.
//      Interpolation (two neighbours on same axis) is preferred;
//      extrapolation (one-sided, needs two points on same side) is
//      used at the marker boundary.
//   2. Run cornerSubPix at the expected position — this is a local
//      sub-pixel refinement that works independently of the global
//      saddle-response threshold and finds a corner immediately if
//      one exists near the predicted location.
//   3. Validate the result: refined position must be within
//      search_r of the expected position and not duplicate an
//      existing persistent corner.
//   4. Inject into persistent_corners_ with missed_frames=0.
//
// Using cornerSubPix instead of searching raw.points means the corner
// is found in frame N+0 (not after waiting for the saddle detector to
// fire) and is insensitive to global threshold tuning.
// ============================================================

void CheckerboardDetector::tryCompleteMissingCorners(
    const cv::Mat& gray,
    bool /*tracking*/
) {
    // Build lookup of currently visible corners.
    std::vector<std::pair<std::pair<int,int>, cv::Point2f>> by_ij;
    by_ij.reserve(persistent_corners_.size());

    for (const auto& pc : persistent_corners_) {
        if (pc.missed_frames != 0) continue;
        by_ij.push_back({{pc.corner.i, pc.corner.j}, pc.corner.uv});
    }

    if (static_cast<int>(by_ij.size()) < config_.min_corners) return;

    auto findUv = [&](int i, int j) -> const cv::Point2f* {
        for (const auto& p : by_ij)
            if (p.first.first == i && p.first.second == j)
                return &p.second;
        return nullptr;
    };

    // Estimate median spacing from visible corners.
    float spacing = 0.0f;
    {
        std::vector<float> dists;
        dists.reserve(by_ij.size() * 2);
        for (const auto& p : by_ij) {
            const cv::Point2f* nb = findUv(p.first.first + 1, p.first.second);
            if (nb) dists.push_back(distf(p.second, *nb));
            nb = findUv(p.first.first, p.first.second + 1);
            if (nb) dists.push_back(distf(p.second, *nb));
        }
        if (dists.empty()) return;
        std::sort(dists.begin(), dists.end());
        spacing = dists[dists.size() / 2];
    }

    if (spacing < 4.0f) return;

    // Bounding box of known grid indices — NO expansion.
    // Only search within the range already established by tracked corners.
    // Boundary corners (just outside current bbox) are picked up by
    // injectRecoveryCorners which has a full lattice fit for context.
    // Expanding here caused unbounded growth every frame.
    int min_i = std::numeric_limits<int>::max();
    int max_i = std::numeric_limits<int>::min();
    int min_j = std::numeric_limits<int>::max();
    int max_j = std::numeric_limits<int>::min();

    for (const auto& p : by_ij) {
        min_i = std::min(min_i, p.first.first);
        max_i = std::max(max_i, p.first.first);
        min_j = std::min(min_j, p.first.second);
        max_j = std::max(max_j, p.first.second);
    }

    // cornerSubPix parameters: search window ~ spacing/4, tight criterion.
    const int subpix_half = std::max(3, static_cast<int>(spacing * 0.25f));
    const cv::Size subpix_win(subpix_half, subpix_half);
    const cv::Size subpix_dead(-1, -1);
    const cv::TermCriteria subpix_crit(
        cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 0.05);

    // Max allowed displacement of refined position from expected.
    const float search_r  = spacing * 0.45f;
    // Min distance from existing persistent corners (duplicate guard).
    const float min_dist  = spacing * 0.35f;

    for (int gi = min_i; gi <= max_i; ++gi) {
        for (int gj = min_j; gj <= max_j; ++gj) {
            if (findUv(gi, gj)) continue;
            if (findPersistentCornerByGrid(persistent_corners_, gi, gj) >= 0) continue;

            // --- Compute expected position ---
            const cv::Point2f* p_im1 = findUv(gi - 1, gj);
            const cv::Point2f* p_ip1 = findUv(gi + 1, gj);
            const cv::Point2f* p_jm1 = findUv(gi, gj - 1);
            const cv::Point2f* p_jp1 = findUv(gi, gj + 1);

            cv::Point2f expected(0.0f, 0.0f);
            int count = 0;

            // Interpolation (preferred — both neighbours on same axis).
            if (p_im1 && p_ip1) { expected += 0.5f * (*p_im1 + *p_ip1); ++count; }
            if (p_jm1 && p_jp1) { expected += 0.5f * (*p_jm1 + *p_jp1); ++count; }

            // Extrapolation (boundary — one side only, needs 2 points).
            if (count == 0) {
                if (!p_ip1 && p_im1) {
                    const cv::Point2f* p_im2 = findUv(gi - 2, gj);
                    if (p_im2) { expected = *p_im1 + (*p_im1 - *p_im2); ++count; }
                }
                if (!p_im1 && p_ip1) {
                    const cv::Point2f* p_ip2 = findUv(gi + 2, gj);
                    if (p_ip2) { expected = *p_ip1 + (*p_ip1 - *p_ip2); ++count; }
                }
                if (count == 0 && !p_jp1 && p_jm1) {
                    const cv::Point2f* p_jm2 = findUv(gi, gj - 2);
                    if (p_jm2) { expected = *p_jm1 + (*p_jm1 - *p_jm2); ++count; }
                }
                if (count == 0 && !p_jm1 && p_jp1) {
                    const cv::Point2f* p_jp2 = findUv(gi, gj + 2);
                    if (p_jp2) { expected = *p_jp1 + (*p_jp1 - *p_jp2); ++count; }
                }
            }

            if (count == 0) continue;
            if (count > 1) expected *= 1.0f / static_cast<float>(count);

            // Boundary check.
            const float margin = static_cast<float>(subpix_half + 2);
            if (expected.x < margin || expected.y < margin) continue;
            if (expected.x >= static_cast<float>(gray.cols) - margin) continue;
            if (expected.y >= static_cast<float>(gray.rows) - margin) continue;

            // Duplicate guard on expected position.
            bool too_close = false;
            for (const auto& pc : persistent_corners_) {
                if (distf(pc.corner.uv, expected) < min_dist) {
                    too_close = true; break;
                }
            }
            if (too_close) continue;

            // --- cornerSubPix refinement at expected position ---
            // cornerSubPix refines locally without any global threshold —
            // it finds the corner immediately if the image structure exists.
            std::vector<cv::Point2f> pts = { expected };
            cv::cornerSubPix(gray, pts, subpix_win, subpix_dead, subpix_crit);
            const cv::Point2f refined = pts[0];

            // Reject if refinement moved too far (converged to wrong feature).
            if (distf(refined, expected) > search_r) continue;

            // Boundary check on refined position.
            if (refined.x < 2.0f || refined.y < 2.0f) continue;
            if (refined.x >= static_cast<float>(gray.cols) - 2.0f) continue;
            if (refined.y >= static_cast<float>(gray.rows) - 2.0f) continue;

            // Final duplicate guard on refined position.
            bool pt_too_close = false;
            for (const auto& pc : persistent_corners_) {
                if (distf(pc.corner.uv, refined) < min_dist) {
                    pt_too_close = true; break;
                }
            }
            if (pt_too_close) continue;

            // Quadrant symmetry check: verify the refined position actually
            // has a checkerboard corner pattern (alternating bright/dark).
            // This rejects cornerSubPix results that converged to table edges,
            // bottle edges, or other non-checkerboard features.
            if (config_.quadrant_half_r > 0) {
                cv::Mat gray_f;
                gray.convertTo(gray_f, CV_32F);
                // Use a half_r scaled to spacing for reliable sampling.
                const int half_r = std::max(
                    config_.quadrant_half_r,
                    static_cast<int>(spacing * 0.15f));
                if (!passesQuadrantTest(gray_f, refined, half_r,
                                        config_.quadrant_min_contrast,
                                        config_.quadrant_max_diagonal_diff))
                    continue;
            }

            // Inject.
            GridCorner gc;
            gc.i  = gi;
            gc.j  = gj;
            gc.uv = refined;

            PersistentTrackedCorner pc;
            pc.corner        = gc;
            pc.missed_frames = 0;
            pc.tracked       = true;
            persistent_corners_.push_back(pc);

            // Make available to subsequent slots in this iteration.
            by_ij.push_back({{gi, gj}, refined});
        }
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
        visible_corners.push_back(pc.corner);
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

} // namespace hydramarker