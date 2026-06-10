#include "checkerboard_detector.hpp"

#include <array>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_set>

#include <opencv2/calib3d.hpp>
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

static float cross2(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x * b.y - a.y * b.x;
}

static float pointSegmentDistance(
    const cv::Point2f& p,
    const cv::Point2f& a,
    const cv::Point2f& b
) {
    const cv::Point2f ab = b - a;
    const float denom = ab.x * ab.x + ab.y * ab.y;
    if (denom <= 1e-6f) return distf(p, a);

    const cv::Point2f ap = p - a;
    const float t =
        std::max(0.0f,
                 std::min(1.0f,
                          (ap.x * ab.x + ap.y * ab.y) / denom));
    return distf(p, a + t * ab);
}

static float quadArea(const std::array<cv::Point2f, 4>& q) {
    float twice_area = 0.0f;
    for (int k = 0; k < 4; ++k) {
        const auto& a = q[k];
        const auto& b = q[(k + 1) % 4];
        twice_area += a.x * b.y - b.x * a.y;
    }
    return 0.5f * std::abs(twice_area);
}

static float pointInsideQuadDistance(
    const std::array<cv::Point2f, 4>& q,
    const cv::Point2f& p
) {
    bool has_pos = false;
    bool has_neg = false;
    for (int k = 0; k < 4; ++k) {
        const cv::Point2f edge = q[(k + 1) % 4] - q[k];
        const cv::Point2f rel = p - q[k];
        const float c = cross2(edge, rel);
        if (c > 1e-3f) has_pos = true;
        if (c < -1e-3f) has_neg = true;
        if (has_pos && has_neg) return -1.0f;
    }

    float min_dist = std::numeric_limits<float>::max();
    for (int k = 0; k < 4; ++k) {
        min_dist = std::min(
            min_dist,
            pointSegmentDistance(p, q[k], q[(k + 1) % 4]));
    }
    return min_dist;
}

static float pointToQuadDistance(
    const std::array<cv::Point2f, 4>& q,
    const cv::Point2f& p
) {
    if (pointInsideQuadDistance(q, p) >= 0.0f) {
        return 0.0f;
    }

    float min_dist = std::numeric_limits<float>::max();
    for (int k = 0; k < 4; ++k) {
        min_dist = std::min(
            min_dist,
            pointSegmentDistance(p, q[k], q[(k + 1) % 4]));
    }
    return min_dist;
}

static bool projectHomographyPoint(
    const cv::Mat& H,
    const cv::Point2f& src,
    cv::Point2f& dst
) {
    const double x = src.x;
    const double y = src.y;
    const double w = H.at<double>(2, 0) * x + H.at<double>(2, 1) * y + H.at<double>(2, 2);
    if (std::abs(w) < 1e-6) return false;
    dst.x = static_cast<float>((H.at<double>(0, 0) * x + H.at<double>(0, 1) * y + H.at<double>(0, 2)) / w);
    dst.y = static_cast<float>((H.at<double>(1, 0) * x + H.at<double>(1, 1) * y + H.at<double>(1, 2)) / w);
    return std::isfinite(dst.x) && std::isfinite(dst.y);
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

static std::int64_t gridKey64(int i, int j) {
    return (static_cast<std::int64_t>(static_cast<std::uint32_t>(i)) << 32) |
           static_cast<std::uint32_t>(j);
}

static int maxContiguousCellSquare(const CheckerboardDetection& det) {
    if (det.cells.empty()) return 0;

    std::unordered_set<std::int64_t> cells;
    cells.reserve(det.cells.size() * 2);
    for (const auto& cell : det.cells) {
        cells.insert(gridKey64(cell.i, cell.j));
    }

    int best = 0;
    for (const auto& origin : det.cells) {
        for (int size = 1;; ++size) {
            bool ok = true;
            for (int di = 0; di < size && ok; ++di) {
                for (int dj = 0; dj < size; ++dj) {
                    if (cells.find(gridKey64(origin.i + di,
                                             origin.j + dj)) ==
                        cells.end()) {
                        ok = false;
                        break;
                    }
                }
            }
            if (!ok) break;
            best = std::max(best, size);
        }
    }
    return best;
}

static bool hasDecodeableCellSpan(
    const CheckerboardDetection& det,
    int min_span
) {
    if (min_span <= 1) return true;
    return maxContiguousCellSquare(det) >= min_span;
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

static cv::Rect trackingRecoveryRoi(
    const CheckerboardDetection& hint,
    const cv::Size& image_size,
    const CheckerboardDetectorConfig& config,
    float margin_multiplier = 1.0f
) {
    if (!hint.valid() || hint.corners.empty() ||
        image_size.width <= 0 || image_size.height <= 0) {
        return {};
    }

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    for (const auto& c : hint.corners) {
        if (!std::isfinite(c.uv.x) || !std::isfinite(c.uv.y)) continue;
        min_x = std::min(min_x, c.uv.x);
        min_y = std::min(min_y, c.uv.y);
        max_x = std::max(max_x, c.uv.x);
        max_y = std::max(max_y, c.uv.y);
    }

    if (min_x > max_x || min_y > max_y) return {};

    const float spacing = estimateMedianSpacing(hint);
    const float spacing_margin =
        spacing > 1.0f
            ? spacing * std::max(0.0f, config.tracking_recovery_roi_margin_cells)
            : 0.0f;
    const float base_margin = std::max(
        static_cast<float>(std::max(0, config.tracking_recovery_roi_min_margin_px)),
        spacing_margin);
    const float margin =
        base_margin * std::max(1.0f, margin_multiplier);

    const int x0 = std::max(0, static_cast<int>(std::floor(min_x - margin)));
    const int y0 = std::max(0, static_cast<int>(std::floor(min_y - margin)));
    const int x1 = std::min(
        image_size.width,
        static_cast<int>(std::ceil(max_x + margin)) + 1);
    const int y1 = std::min(
        image_size.height,
        static_cast<int>(std::ceil(max_y + margin)) + 1);

    if (x1 <= x0 || y1 <= y0) return {};

    const cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
    const double image_area =
        static_cast<double>(image_size.width) *
        static_cast<double>(image_size.height);
    const double roi_area =
        static_cast<double>(roi.width) *
        static_cast<double>(roi.height);

    if (image_area <= 0.0) return {};

    const double max_ratio =
        static_cast<double>(config.tracking_recovery_roi_max_area_ratio);
    if (max_ratio > 0.0 && max_ratio < 1.0 &&
        roi_area / image_area > max_ratio) {
        return {};
    }

    return roi;
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
            bool has_grid_neighbour    = false;

            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                const float d = distf(current[i].uv, current[j].uv);
                if (d < duplicate_d) { is_duplicate = true; break; }
                if (d >= min_d && d <= max_d) has_spacing_neighbour = true;

                const int di = std::abs(current[i].i - current[j].i);
                const int dj = std::abs(current[i].j - current[j].j);
                if (di + dj == 1 &&
                    d >= median_spacing * 0.25f &&
                    d <= median_spacing * 1.85f) {
                    has_grid_neighbour = true;
                }
            }

            const bool established_grid_corner =
                current[i].observed_frames >= 8 && has_grid_neighbour;

            if (is_duplicate ||
                (!has_spacing_neighbour && !established_grid_corner)) {
                keep[i] = false;
            }
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

double CheckerboardDetector::elapsedMs(std::int64_t start_tick) {
    return 1000.0 *
           (static_cast<double>(cv::getTickCount() - start_tick) /
            cv::getTickFrequency());
}

void CheckerboardDetector::clearTimings() const {
    last_timings_ms_.clear();
}

void CheckerboardDetector::addTimingMs(
    const std::string& name,
    double elapsed_ms
) const {
    last_timings_ms_[name] += elapsed_ms;
}

std::unordered_map<std::string, double>
CheckerboardDetector::lastTimingsMs() const {
    return last_timings_ms_;
}

void CheckerboardDetector::resetTracking() {
    last_gray_.release();
    last_gray_pyramid_.clear();
    pending_current_gray_pyramid_.clear();
    last_gray_pyramid_frame_index_ = -1;
    pending_current_gray_pyramid_frame_index_ = -1;
    recovery_region_cache_.reset();
    last_detection_ = CheckerboardDetection{};
    persistent_corners_.clear();
    pending_completion_corners_.clear();
    tracking_active_       = false;
    frame_index_           = 0;
    degraded_frames_count_ = 0;
    low_corner_frames_     = 0;
    undecodeable_tracking_frames_ = 0;
    held_output_frames_ = 0;
    roi_align_fail_frames_ = 0;
    roi_recovery_fail_frames_ = 0;
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
    clearTimings();
    recovery_region_cache_.reset();
    const auto to_gray_t0 = cv::getTickCount();
    const cv::Mat gray = toGray8(image);
    addTimingMs("to_gray_ms", elapsedMs(to_gray_t0));
    if (gray.empty()) { resetTracking(); return std::nullopt; }

    ++frame_index_;

    const int max_held_output_frames =
        std::max(1, config_.max_low_corner_frames);

    if (tracking_active_ &&
        !last_gray_.empty() &&
        !last_detection_.corners.empty()) {

        const auto track_t0 = cv::getTickCount();
        auto tracked = trackFromPreviousFrame(gray);
        addTimingMs("track_total_ms", elapsedMs(track_t0));

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
                const int roi_fail_retry_after_frames =
                    config_.tracking_recovery_roi_fail_full_retry_frames;
                const bool allow_roi_fail_count_retry =
                    roi_fail_retry_after_frames > 0 &&
                    tracked->stable &&
                    roi_recovery_fail_frames_ >= roi_fail_retry_after_frames;
                const bool allow_roi_fail_full_retry =
                    allow_roi_fail_count_retry;

                int persistent_missed_count = 0;
                int persistent_predicted_count = 0;
                for (const auto& pc : persistent_corners_) {
                    if (pc.missed_frames > 0) {
                        ++persistent_missed_count;
                    }
                    if (pc.corner.predicted || pc.predicted_frames > 0) {
                        ++persistent_predicted_count;
                    }
                }

                const int full_build_interval =
                    config_.tracking_recovery_full_build_interval_frames;
                const bool periodic_full_build_due =
                    full_build_interval <= 1 ||
                    (frame_index_ % full_build_interval == 0);
                const bool tracked_decodeable =
                    hasDecodeableCellSpan(
                        *tracked,
                        config_.min_tracking_decode_cell_span);
                const int tracked_corner_count =
                    static_cast<int>(tracked->corners.size());
                const int tracked_cell_count =
                    static_cast<int>(tracked->cells.size());
                const bool tracked_dense_enough =
                    tracked_cell_count >=
                    std::max(12, config_.min_tracking_cells * 6);
                const bool persistent_state_not_critical =
                    persistent_missed_count <
                        std::max(24, tracked_corner_count / 2) &&
                    persistent_predicted_count <
                        std::max(24, tracked_corner_count / 2);
                const bool roi_candidates_only =
                    config_.use_tracking_roi_recovery &&
                    full_build_interval > 1 &&
                    !periodic_full_build_due &&
                    !allow_roi_fail_full_retry &&
                    roi_align_fail_frames_ == 0 &&
                    roi_recovery_fail_frames_ == 0 &&
                    tracked->stable &&
                    tracked_decodeable &&
                    tracked_dense_enough &&
                    persistent_state_not_critical &&
                    !corner_loss &&
                    !geometry_degraded;
                if (roi_candidates_only) {
                    addTimingMs(
                        "refresh_roi_candidate_only_requested_count",
                        1.0);
                }

                const auto recovery_t0 = cv::getTickCount();
                auto recovered = detectRecovery(
                    gray,
                    &(*tracked),
                    allow_roi_fail_full_retry,
                    roi_candidates_only);
                addTimingMs("refresh_recovery_call_ms", elapsedMs(recovery_t0));
                const auto full_fallback_it =
                    last_timings_ms_.find("recovery_full_frame_fallback_ms");
                const bool full_frame_fallback_used =
                    full_fallback_it != last_timings_ms_.end() &&
                    full_fallback_it->second > 0.0;
                if (full_frame_fallback_used) {
                    addTimingMs("refresh_roi_fail_full_retry_count", 1.0);
                }

                if (recovered && recovered->valid()) {
                    roi_recovery_fail_frames_ = 0;
                    const auto align_t0 = cv::getTickCount();
                    auto aligned = alignDetectionGridToReference(*recovered, *tracked);
                    addTimingMs("align_recovery_ms", elapsedMs(align_t0));
                    if (!aligned) {
                        ++roi_align_fail_frames_;
                        addTimingMs("refresh_roi_align_fail_count", 1.0);

                        auto countPredictedCorners =
                            [](const CheckerboardDetection& det) {
                                int count = 0;
                                for (const auto& c : det.corners) {
                                    if (c.predicted) ++count;
                                }
                                return count;
                            };

                        const int recovered_corner_count =
                            static_cast<int>(recovered->corners.size());
                        const int recovered_cell_count =
                            static_cast<int>(recovered->cells.size());
                        const int tracked_corner_count =
                            static_cast<int>(tracked->corners.size());
                        const int tracked_cell_count =
                            static_cast<int>(tracked->cells.size());
                        const int tracked_predicted_count =
                            countPredictedCorners(*tracked);

                        const bool recovered_quality_ok =
                            recovered_corner_count >=
                                std::max(24, config_.min_corners * 4) &&
                            recovered_cell_count >=
                                std::max(8, config_.min_cells * 4) &&
                            hasDecodeableCellSpan(
                                *recovered,
                                config_.min_tracking_decode_cell_span);

                        const bool tracked_decode_weak =
                            tracked_cell_count <
                                std::max(8, config_.min_tracking_cells * 4) ||
                            !hasDecodeableCellSpan(
                                *tracked,
                                config_.min_tracking_decode_cell_span);

                        const bool tracked_predicted_heavy =
                            tracked_predicted_count >=
                            std::max(6, tracked_corner_count / 3);

                        const bool recovery_competitive =
                            recovered_cell_count >= tracked_cell_count - 2 ||
                            recovered_corner_count >= tracked_corner_count - 4 ||
                            tracked_decode_weak ||
                            tracked_predicted_heavy;

                        const bool roi_reacquire_quality_ok =
                            !full_frame_fallback_used &&
                            recovered_quality_ok &&
                            recovery_competitive &&
                            (tracked_decode_weak ||
                             tracked_predicted_heavy ||
                             roi_align_fail_frames_ >= 2);
                        if (roi_reacquire_quality_ok) {
                            addTimingMs(
                                "refresh_roi_unaligned_reset_count",
                                1.0);
                            recovered->tracking = false;
                            recovered->stable = false;
                            const auto update_roi_t0 = cv::getTickCount();
                            updateTrackingState(gray, *recovered);
                            addTimingMs(
                                "update_tracking_state_ms",
                                elapsedMs(update_roi_t0));
                            undecodeable_tracking_frames_ = 0;
                            degraded_frames_count_ = 0;
                            roi_align_fail_frames_ = 0;
                            roi_recovery_fail_frames_ = 0;
                            return last_detection_;
                        }

                        const bool full_reacquire_quality_ok =
                            full_frame_fallback_used &&
                            tracked->stable &&
                            static_cast<int>(recovered->corners.size()) >=
                                std::max(24, config_.min_corners * 4) &&
                            static_cast<int>(recovered->cells.size()) >=
                                std::max(8, config_.min_cells * 4);
                        if (full_reacquire_quality_ok) {
                            addTimingMs(
                                "refresh_full_recovery_unaligned_reset_count",
                                1.0);
                            recovered->tracking = false;
                            recovered->stable = false;
                            const auto update_full_t0 = cv::getTickCount();
                            updateTrackingState(gray, *recovered);
                            addTimingMs(
                                "update_tracking_state_ms",
                                elapsedMs(update_full_t0));
                            undecodeable_tracking_frames_ = 0;
                            degraded_frames_count_ = 0;
                            roi_align_fail_frames_ = 0;
                            roi_recovery_fail_frames_ = 0;
                            return last_detection_;
                        }

                        const int full_retry_after_frames =
                            config_.tracking_recovery_align_fail_full_retry_frames;
                        const bool allow_align_fail_count_retry =
                            full_retry_after_frames > 0 &&
                            roi_align_fail_frames_ >= full_retry_after_frames;
                        const bool allow_expensive_retry =
                            allow_align_fail_count_retry;

                        if (allow_expensive_retry) {
                            bool retry_aligned = false;

                            if (config_.use_tracking_roi_recovery &&
                                config_.tracking_recovery_align_fail_roi_margin_multiplier > 1.0f) {
                                const auto expanded_t0 = cv::getTickCount();
                                const cv::Rect expanded_roi = trackingRecoveryRoi(
                                    *tracked,
                                    gray.size(),
                                    config_,
                                    config_.tracking_recovery_align_fail_roi_margin_multiplier);
                                auto expanded_recovered =
                                    expanded_roi.empty()
                                        ? std::optional<CheckerboardDetection>{}
                                        : detectRecoveryInRegion(
                                              gray,
                                              expanded_roi,
                                              "recovery_expanded_roi_");
                                addTimingMs(
                                    "refresh_expanded_roi_after_align_fail_ms",
                                    elapsedMs(expanded_t0));

                                if (expanded_recovered &&
                                    expanded_recovered->valid()) {
                                    const auto expanded_align_t0 =
                                        cv::getTickCount();
                                    aligned = alignDetectionGridToReference(
                                        *expanded_recovered,
                                        *tracked);
                                    addTimingMs(
                                        "align_recovery_ms",
                                        elapsedMs(expanded_align_t0));
                                    if (aligned) {
                                        recovered = std::move(aligned);
                                        retry_aligned = true;
                                        roi_align_fail_frames_ = 0;
                                        addTimingMs(
                                            "refresh_expanded_roi_align_success_count",
                                            1.0);
                                    } else {
                                        addTimingMs(
                                            "refresh_expanded_roi_align_fail_count",
                                            1.0);
                                    }
                                }
                            }

                            if (!retry_aligned) {
                                const auto full_retry_t0 = cv::getTickCount();
                                auto full_recovered = detectRecovery(gray, nullptr);
                                addTimingMs(
                                    "refresh_full_recovery_after_roi_align_fail_ms",
                                    elapsedMs(full_retry_t0));

                                if (full_recovered && full_recovered->valid()) {
                                    const auto full_align_t0 = cv::getTickCount();
                                    aligned = alignDetectionGridToReference(
                                        *full_recovered,
                                        *tracked);
                                    addTimingMs(
                                        "align_recovery_ms",
                                        elapsedMs(full_align_t0));
                                    if (aligned) {
                                        recovered = std::move(aligned);
                                        retry_aligned = true;
                                        addTimingMs(
                                            "refresh_full_recovery_align_success_count",
                                            1.0);
                                    } else {
                                        recovered = std::nullopt;
                                    }
                                } else {
                                    recovered = std::nullopt;
                                }

                                roi_align_fail_frames_ = 0;
                            }

                            if (retry_aligned) {
                                roi_align_fail_frames_ = 0;
                            } else {
                                recovered = std::nullopt;
                            }
                        } else {
                            recovered = std::nullopt;
                            addTimingMs(
                                "refresh_full_recovery_after_roi_align_fail_deferred_count",
                                1.0);
                        }
                    } else {
                        recovered = std::move(aligned);
                        roi_align_fail_frames_ = 0;
                    }
                } else {
                    roi_align_fail_frames_ = 0;
                    if (roi_candidates_only) {
                        addTimingMs("refresh_roi_candidate_only_count", 1.0);
                    } else {
                        if (full_frame_fallback_used) {
                            roi_recovery_fail_frames_ = 0;
                        } else {
                            ++roi_recovery_fail_frames_;
                        }
                        addTimingMs("refresh_roi_recovery_fail_count", 1.0);
                    }
                }

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
                        const auto update_t0 = cv::getTickCount();
                        updateTrackingState(gray, *recovered);
                        addTimingMs("update_tracking_state_ms", elapsedMs(update_t0));
                        undecodeable_tracking_frames_ = 0;
                        return last_detection_;
                    }

                    // Inject new corners from recovery directly into persistent
                    // state — no lattice refit, no Grid-ID loss.
                    const auto update_t0 = cv::getTickCount();
                    updateTrackingState(gray, *tracked, &(*recovered));
                    addTimingMs("update_tracking_state_ms", elapsedMs(update_t0));
                    if (!last_detection_.valid()) {
                        const auto update_recovered_t0 = cv::getTickCount();
                        updateTrackingState(gray, *recovered);
                        addTimingMs("update_tracking_state_ms", elapsedMs(update_recovered_t0));
                        undecodeable_tracking_frames_ = 0;
                        return last_detection_;
                    }
                    if (hasDecodeableCellSpan(
                            last_detection_,
                            config_.min_tracking_decode_cell_span)) {
                        undecodeable_tracking_frames_ = 0;
                    } else {
                        ++undecodeable_tracking_frames_;
                        if (undecodeable_tracking_frames_ >
                            config_.max_undecodeable_tracking_frames) {
                            resetTracking();
                            return std::nullopt;
                        }
                    }
                    return last_detection_;
                }

                // Recovery found nothing.
                if (geometry_degraded) {
                    ++degraded_frames_count_;
                    if (degraded_frames_count_ >=
                        config_.max_degraded_frames_before_reset) {
                        degraded_frames_count_ = 0;
                        if (last_detection_.valid() &&
                            held_output_frames_ < max_held_output_frames) {
                            ++held_output_frames_;
                            CheckerboardDetection held = last_detection_;
                            held.tracking = true;
                            held.stable = false;
                            return held;
                        }
                        resetTracking();
                        return std::nullopt;
                    }
                } else {
                    degraded_frames_count_ = 0;
                }
            }

            const auto update_t0 = cv::getTickCount();
            updateTrackingState(gray, *tracked);
            addTimingMs("update_tracking_state_ms", elapsedMs(update_t0));
            const int locally_completed =
                tryCompleteMissingCorners(gray, true);
            if (locally_completed > 0) {
                addTimingMs(
                    "tracking_local_completion_added_count",
                    static_cast<double>(locally_completed));
            }
            if (!last_detection_.valid()) {
                const auto recovery_t0 = cv::getTickCount();
                auto recovered = detectRecovery(gray, &(*tracked), false);
                addTimingMs("fallback_recovery_call_ms", elapsedMs(recovery_t0));
                if (recovered && recovered->valid()) {
                    const auto align_t0 = cv::getTickCount();
                    auto aligned = alignDetectionGridToReference(*recovered, *tracked);
                    addTimingMs("align_recovery_ms", elapsedMs(align_t0));
                    if (aligned) {
                        recovered = std::move(aligned);
                    } else {
                        addTimingMs(
                            "fallback_full_recovery_after_roi_align_fail_deferred_count",
                            1.0);
                        recovered = std::nullopt;
                    }
                }
                if (recovered && recovered->valid()) {
                    recovered->tracking = false;
                    recovered->stable   = false;
                    const auto update_recovered_t0 = cv::getTickCount();
                    updateTrackingState(gray, *recovered);
                    addTimingMs("update_tracking_state_ms", elapsedMs(update_recovered_t0));
                    undecodeable_tracking_frames_ = 0;
                    return last_detection_;
                }
            }

            if (!hasDecodeableCellSpan(
                    last_detection_,
                    config_.min_tracking_decode_cell_span)) {
                ++undecodeable_tracking_frames_;
                if (undecodeable_tracking_frames_ >
                    config_.max_undecodeable_tracking_frames) {
                    if (last_detection_.valid() &&
                        held_output_frames_ < max_held_output_frames) {
                        ++held_output_frames_;
                        CheckerboardDetection held = last_detection_;
                        held.tracking = true;
                        held.stable = false;
                        return held;
                    }
                    resetTracking();
                    return std::nullopt;
                }
            } else {
                undecodeable_tracking_frames_ = 0;
            }

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
                if (last_detection_.valid() &&
                    held_output_frames_ < max_held_output_frames) {
                    ++held_output_frames_;
                    CheckerboardDetection held = last_detection_;
                    held.tracking = true;
                    held.stable = false;
                    return held;
                }
                resetTracking();
                return std::nullopt;
            }

            held_output_frames_ = 0;
            return last_detection_;
        }

        tracking_active_ = false;
    }

    // No active tracking — run full recovery.
    const auto recovery_t0 = cv::getTickCount();
    auto recovered = detectRecovery(gray);
    addTimingMs("full_recovery_call_ms", elapsedMs(recovery_t0));

    if (recovered && recovered->valid()) {
        recovered->tracking = false;
        recovered->stable   = false;
        const auto update_t0 = cv::getTickCount();
        updateTrackingState(gray, *recovered);
        addTimingMs("update_tracking_state_ms", elapsedMs(update_t0));
        undecodeable_tracking_frames_ = 0;
        held_output_frames_ = 0;

        if (config_.refresh_interval_frames > 0)
            frame_index_ = config_.refresh_interval_frames - 1;

        return last_detection_;
    }

    if (last_detection_.valid() &&
        held_output_frames_ < max_held_output_frames) {
        ++held_output_frames_;
        CheckerboardDetection held = last_detection_;
        held.tracking = true;
        held.stable = false;
        return held;
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
    if (!lattice || !lattice->valid) {
        auto clustered = buildBestDetectionFromCornerClusters(dbg.valid_refined_points);
        if (clustered && clustered->valid()) {
            dbg.detection     = *clustered;
            dbg.has_detection = true;
        }
        return dbg;
    }

    dbg.lattice     = *lattice;
    dbg.has_lattice = true;

    auto detection = grid_builder_.build(
        *lattice, config_.duplicate_corner_dist_px,
        config_.min_corners, config_.min_cells);

    if (!detection || !detection->valid()) {
        auto clustered = buildBestDetectionFromCornerClusters(dbg.valid_refined_points);
        if (clustered && clustered->valid()) {
            dbg.detection     = *clustered;
            dbg.has_detection = true;
        }
        return dbg;
    }

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
    const cv::Mat& gray,
    const CheckerboardDetection* roi_hint,
    bool allow_full_frame_fallback,
    bool roi_candidates_only
) const {
    struct ScopedRecoveryTimer {
        const CheckerboardDetector* self;
        std::int64_t start;
        ~ScopedRecoveryTimer() {
            self->addTimingMs(
                "recovery_total_ms",
                CheckerboardDetector::elapsedMs(start));
        }
    } recovery_timer{this, cv::getTickCount()};

    if (gray.empty()) return std::nullopt;

    cv::Rect roi;
    const bool can_try_roi =
        config_.use_tracking_roi_recovery &&
        roi_hint != nullptr &&
        roi_hint->valid();

    if (can_try_roi) {
        const auto roi_select_t0 = cv::getTickCount();
        roi = trackingRecoveryRoi(*roi_hint, gray.size(), config_);
        addTimingMs("recovery_roi_select_ms", elapsedMs(roi_select_t0));

        if (!roi.empty()) {
            const double image_area =
                static_cast<double>(gray.cols) *
                static_cast<double>(gray.rows);
            const double roi_area =
                static_cast<double>(roi.width) *
                static_cast<double>(roi.height);

            addTimingMs("recovery_roi_width_px", static_cast<double>(roi.width));
            addTimingMs("recovery_roi_height_px", static_cast<double>(roi.height));
            if (image_area > 0.0) {
                addTimingMs("recovery_roi_area_ratio", roi_area / image_area);
            }

            const auto roi_attempt_t0 = cv::getTickCount();
            auto recovered =
                detectRecoveryInRegion(
                    gray,
                    roi,
                    "recovery_roi_",
                    roi_candidates_only);
            addTimingMs(
                "recovery_roi_attempt_ms",
                elapsedMs(roi_attempt_t0));

            if (roi_candidates_only) {
                return std::nullopt;
            }

            if (recovered && recovered->valid()) {
                addTimingMs("recovery_roi_success_count", 1.0);
                return recovered;
            }

            addTimingMs("recovery_roi_fallback_count", 1.0);
            const float retry_margin_multiplier =
                config_.tracking_recovery_roi_fail_retry_margin_multiplier;
            if (retry_margin_multiplier > 1.0f) {
                const auto retry_select_t0 = cv::getTickCount();
                const cv::Rect retry_roi = trackingRecoveryRoi(
                    *roi_hint,
                    gray.size(),
                    config_,
                    retry_margin_multiplier);
                addTimingMs(
                    "recovery_roi_retry_select_ms",
                    elapsedMs(retry_select_t0));

                const bool retry_roi_changed =
                    !retry_roi.empty() &&
                    (retry_roi.x != roi.x ||
                     retry_roi.y != roi.y ||
                     retry_roi.width != roi.width ||
                     retry_roi.height != roi.height);

                if (retry_roi_changed) {
                    const double retry_roi_area =
                        static_cast<double>(retry_roi.width) *
                        static_cast<double>(retry_roi.height);

                    addTimingMs(
                        "recovery_roi_retry_width_px",
                        static_cast<double>(retry_roi.width));
                    addTimingMs(
                        "recovery_roi_retry_height_px",
                        static_cast<double>(retry_roi.height));
                    if (image_area > 0.0) {
                        addTimingMs(
                            "recovery_roi_retry_area_ratio",
                            retry_roi_area / image_area);
                    }

                    const auto retry_attempt_t0 = cv::getTickCount();
                    auto retry_recovered = detectRecoveryInRegion(
                        gray,
                        retry_roi,
                        "recovery_roi_retry_");
                    addTimingMs(
                        "recovery_roi_retry_attempt_ms",
                        elapsedMs(retry_attempt_t0));

                    if (retry_recovered && retry_recovered->valid()) {
                        addTimingMs(
                            "recovery_roi_retry_success_count",
                            1.0);
                        return retry_recovered;
                    }

                    addTimingMs("recovery_roi_retry_fallback_count", 1.0);
                } else {
                    addTimingMs("recovery_roi_retry_skipped_count", 1.0);
                }
            }

            if (!allow_full_frame_fallback) {
                addTimingMs("recovery_full_frame_fallback_deferred_count", 1.0);
                return std::nullopt;
            }
        } else {
            addTimingMs("recovery_roi_skipped_count", 1.0);
            if (!allow_full_frame_fallback) {
                addTimingMs("recovery_full_frame_fallback_deferred_count", 1.0);
                return std::nullopt;
            }
        }
    }

    const auto full_t0 = cv::getTickCount();
    auto recovered = detectRecoveryInRegion(
        gray,
        cv::Rect{},
        !roi.empty() ? "recovery_full_fallback_" : nullptr);
    if (can_try_roi) {
        addTimingMs("recovery_full_frame_fallback_ms", elapsedMs(full_t0));
    }
    return recovered;
}

std::optional<CheckerboardDetection>
CheckerboardDetector::detectRecoveryInRegion(
    const cv::Mat& gray,
    const cv::Rect& roi,
    const char* timing_prefix,
    bool candidates_only
) const {
    if (gray.empty()) return std::nullopt;

    auto addRecoveryTiming = [&](const std::string& suffix, double ms) {
        addTimingMs(std::string("recovery_") + suffix, ms);
        if (timing_prefix && timing_prefix[0] != '\0') {
            addTimingMs(std::string(timing_prefix) + suffix, ms);
        }
    };

    const auto crop_t0 = cv::getTickCount();
    cv::Mat source = gray;
    cv::Point2f offset(0.0f, 0.0f);
    const cv::Rect image_rect(0, 0, gray.cols, gray.rows);
    const cv::Rect active_roi = roi & image_rect;

    if (active_roi.width > 0 && active_roi.height > 0 &&
        (active_roi.width < gray.cols || active_roi.height < gray.rows)) {
        source = gray(active_roi);
        offset = cv::Point2f(
            static_cast<float>(active_roi.x),
            static_cast<float>(active_roi.y));
    }
    addRecoveryTiming("crop_ms", elapsedMs(crop_t0));

    const auto resize_t0 = cv::getTickCount();
    cv::Mat work = source;
    float scale  = 1.0f;

    if (config_.det_width > 0 && source.cols > config_.det_width) {
        scale = static_cast<float>(config_.det_width) /
                static_cast<float>(source.cols);
        const int new_w = config_.det_width;
        const int new_h = std::max(1, static_cast<int>(std::round(source.rows * scale)));
        cv::resize(source, work, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_AREA);
    }
    addRecoveryTiming("resize_ms", elapsedMs(resize_t0));

    auto rememberRecoveryRegion = [&](
        const CornerDetectionResult& raw_result,
        const std::vector<RefinedCorner>& refined_result
    ) {
        if (scale != 1.0f ||
            active_roi.width <= 0 ||
            active_roi.height <= 0) {
            return;
        }

        RecoveryRegionCache cache;
        cache.frame_index = frame_index_;
        cache.roi = active_roi;
        cache.raw = raw_result;
        cache.refined = refined_result;
        recovery_region_cache_ = std::move(cache);
    };

    const auto corner_detect_t0 = cv::getTickCount();
    CornerDetectionResult raw = corner_detector_.detect(
        work, config_.max_recovery_corners,
        config_.saddle_sigma, config_.saddle_response_threshold);
    addRecoveryTiming("corner_detect_ms", elapsedMs(corner_detect_t0));
    addRecoveryTiming(
        "raw_count",
        static_cast<double>(raw.points.size()));
    for (const auto& timing : raw.timings_ms) {
        addTimingMs(timing.first, timing.second);
        if (timing_prefix && timing_prefix[0] != '\0') {
            addTimingMs(std::string(timing_prefix) + timing.first,
                        timing.second);
        }
    }

    if (raw.points.empty()) {
        rememberRecoveryRegion(raw, {});
        return std::nullopt;
    }

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

    const auto refine_t0 = cv::getTickCount();
    std::vector<RefinedCorner> refined = corner_refiner_.refine(
        work, raw.points, raw.grad_x, raw.grad_y, refine_config);
    addRecoveryTiming("refine_ms", elapsedMs(refine_t0));
    rememberRecoveryRegion(raw, refined);

    if (candidates_only) {
        addRecoveryTiming("candidate_only_count", 1.0);
        return std::nullopt;
    }

    if (static_cast<int>(refined.size()) < config_.min_corners)
        return std::nullopt;

    // Apply quadrant test on the work image (recovery only).
    // This is the single place where passesQuadrantTest() is called.
    cv::Mat work_f;
    work.convertTo(work_f, CV_32F);

    const float inv_scale = 1.0f / scale;
    std::vector<cv::Point2f> refined_corners;
    refined_corners.reserve(refined.size());

    std::vector<cv::Point2f> quadrant_corners;
    quadrant_corners.reserve(refined.size());

    const auto quadrant_t0 = cv::getTickCount();
    for (const auto& c : refined) {
        if (!c.valid) continue;

        const cv::Point2f full_uv(
            c.uv.x * inv_scale + offset.x,
            c.uv.y * inv_scale + offset.y);
        refined_corners.push_back(full_uv);

        // Quadrant test in work-image coordinates.
        if (config_.quadrant_half_r > 0 &&
            !passesQuadrantTest(work_f, c.uv,
                                config_.quadrant_half_r,
                                config_.quadrant_min_contrast,
                                config_.quadrant_max_diagonal_diff))
            continue;

        quadrant_corners.push_back(full_uv);
    }
    addRecoveryTiming("quadrant_filter_ms", elapsedMs(quadrant_t0));
    addRecoveryTiming(
        "refined_count",
        static_cast<double>(refined_corners.size()));
    addRecoveryTiming(
        "quadrant_count",
        static_cast<double>(quadrant_corners.size()));

    if (static_cast<int>(refined_corners.size()) < config_.min_corners)
        return std::nullopt;

    // CornerRefiner already applies a quadrant symmetry check with a safe
    // fallback.  This second recovery-only check is useful for suppressing dot
    // centres in crisp front-facing frames, but under motion blur or steep
    // viewing angles it can reject too much and leave the tracker with no
    // green corners at all.  Use it only when enough corners survive; otherwise
    // let the lattice/grid geometry reject outliers.
    const std::vector<cv::Point2f>& corners =
        static_cast<int>(quadrant_corners.size()) >= config_.min_corners
            ? quadrant_corners
            : refined_corners;

    const auto build_best_t0 = cv::getTickCount();
    auto detection = buildBestDetectionFromCornerClusters(corners);
    addRecoveryTiming("build_best_ms", elapsedMs(build_best_t0));
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
        const auto completion_t0 = cv::getTickCount();
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
                    const cv::Point2f expected_work =
                        (expected - offset) * scale;

                    if (expected_work.x < 4.0f || expected_work.y < 4.0f) continue;
                    if (expected_work.x >= static_cast<float>(work.cols) - 4.0f) continue;
                    if (expected_work.y >= static_cast<float>(work.rows) - 4.0f) continue;
                    if (hasNearDetected(expected)) continue;

                    float best_d = search_r_work;
                    cv::Point2f best_pt(-1.0f, -1.0f);

                    for (const auto& cpt : raw.points) {
                        const float d = distf(cpt, expected_work);
                        if (d < best_d) {
                            best_d = d;
                            best_pt = cpt * inv_scale + offset;
                        }
                    }

                    if (best_pt.x >= 0.0f && !hasNearDetected(best_pt)) {
                        new_corners.push_back(best_pt);
                        detected_uvs.push_back(best_pt);
                    }
                }
            }

            if (new_corners.size() > corners.size()) {
                const auto completion_build_t0 = cv::getTickCount();
                auto completed = buildDetectionFromCorners(new_corners);
                addRecoveryTiming(
                    "completion_build_ms",
                    elapsedMs(completion_build_t0));
                if (completed && completed->valid())
                    detection = std::move(completed);
            }
        }
        addRecoveryTiming("completion_ms", elapsedMs(completion_t0));
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

    const auto lattice_t0 = cv::getTickCount();
    auto lattice = lattice_model_.fit(corners);
    addTimingMs("lattice_fit_ms", elapsedMs(lattice_t0));
    if (!lattice || !lattice->valid) return std::nullopt;

    const auto grid_build_t0 = cv::getTickCount();
    auto detection = grid_builder_.build(
        *lattice, config_.duplicate_corner_dist_px,
        config_.min_corners, config_.min_cells);
    addTimingMs("grid_build_lattice_ms", elapsedMs(grid_build_t0));

    if (!detection || !detection->valid()) return std::nullopt;
    return detection;
}

std::optional<CheckerboardDetection>
CheckerboardDetector::buildBestDetectionFromCornerClusters(
    const std::vector<cv::Point2f>& corners
) const {
    struct ScopedBuildBestTimer {
        const CheckerboardDetector* self;
        std::int64_t start;
        ~ScopedBuildBestTimer() {
            self->addTimingMs(
                "build_best_total_ms",
                CheckerboardDetector::elapsedMs(start));
        }
    } build_best_timer{this, cv::getTickCount()};

    if (static_cast<int>(corners.size()) < config_.min_corners)
        return std::nullopt;

    auto best = buildDetectionFromCorners(corners);

    auto detectionScore = [](const CheckerboardDetection& det) {
        return static_cast<int>(det.cells.size()) * 4 +
               static_cast<int>(det.corners.size());
    };

    auto better = [](const CheckerboardDetection& a,
                     const CheckerboardDetection& b) {
        const int score_a =
            static_cast<int>(a.cells.size()) * 4 +
            static_cast<int>(a.corners.size());
        const int score_b =
            static_cast<int>(b.cells.size()) * 4 +
            static_cast<int>(b.corners.size());
        if (score_a != score_b) return score_a > score_b;
        if (a.cells.size() != b.cells.size())
            return a.cells.size() > b.cells.size();
        return a.corners.size() > b.corners.size();
    };

    const int n = static_cast<int>(corners.size());
    const int min_subset = std::max(config_.min_corners, 8);
    const int max_subset = std::min(n, 44);

    if (n <= min_subset) {
        return best;
    }

    int best_score = best && best->valid()
        ? detectionScore(*best)
        : std::numeric_limits<int>::min();

    const int max_possible_subset_score = max_subset * 5;

    if (best && best->valid()) {
        // A subset detection can only use points from that subset. A valid grid
        // built from K corners cannot have more than K unit cells, so its score
        // is bounded by K corners + 4*K cells = 5*K.  If the all-corner
        // candidate already exceeds that strict upper bound, no subset below
        // can win under the existing scoring/tie-break rules.
        if (best_score > max_possible_subset_score) {
            addTimingMs("build_best_subset_pruned_count", 1.0);
            return best;
        }
    }

    std::vector<int> seed_indices;
    seed_indices.reserve(n);
    for (int i = 0; i < n; ++i) {
        seed_indices.push_back(i);
    }

    // Keep this recovery path fast: evaluate all seeds for ordinary point
    // counts, and a cheap spatially spread subset for dense noisy frames.
    if (n > 60) {
        std::sort(
            seed_indices.begin(),
            seed_indices.end(),
            [&](int a, int b) {
                if (corners[a].y != corners[b].y)
                    return corners[a].y < corners[b].y;
                return corners[a].x < corners[b].x;
            });
        std::vector<int> reduced;
        reduced.reserve(60);
        const int stride = std::max(1, n / 60);
        for (int k = 0; k < n; k += stride) {
            reduced.push_back(seed_indices[k]);
        }
        seed_indices = std::move(reduced);
    }

    std::vector<std::pair<float, int>> by_dist;
    by_dist.reserve(n);
    std::vector<cv::Point2f> subset;
    subset.reserve(max_subset);

    for (int seed : seed_indices) {
        if (best_score > max_possible_subset_score) {
            addTimingMs("build_best_subset_pruned_count", 1.0);
            return best;
        }

        by_dist.clear();
        for (int k = 0; k < n; ++k) {
            const float d2 = dist2(corners[seed], corners[k]);
            by_dist.push_back({d2, k});
        }

        const int take = max_subset;
        std::nth_element(
            by_dist.begin(),
            by_dist.begin() + static_cast<std::ptrdiff_t>(take - 1),
            by_dist.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
        by_dist.resize(take);
        std::sort(
            by_dist.begin(),
            by_dist.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first;
            });

        for (int subset_size : {12, 18, 28, 44}) {
            if (subset_size < min_subset || subset_size > take) continue;
            if (best_score > subset_size * 5) {
                addTimingMs("build_best_subset_pruned_count", 1.0);
                continue;
            }

            subset.clear();
            for (int m = 0; m < subset_size; ++m) {
                subset.push_back(corners[by_dist[m].second]);
            }

            auto candidate = buildDetectionFromCorners(subset);
            if (!candidate || !candidate->valid()) continue;

            if (!best || better(*candidate, *best)) {
                best = std::move(candidate);
                best_score = detectionScore(*best);
                if (best_score > max_possible_subset_score) {
                    addTimingMs("build_best_subset_pruned_count", 1.0);
                    return best;
                }
            }
        }
    }

    return best;
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
    struct ScopedBuildVisibleTimer {
        const CheckerboardDetector* self;
        std::int64_t start;
        ~ScopedBuildVisibleTimer() {
            self->addTimingMs(
                "build_visible_tracked_total_ms",
                CheckerboardDetector::elapsedMs(start));
        }
    } build_visible_timer{this, cv::getTickCount()};

    if (validation.visible_indices.size() != validation.visible_points.size())
        return std::nullopt;
    if (validation.visible_predicted.size() != validation.visible_points.size())
        return std::nullopt;

    if (static_cast<int>(validation.visible_points.size()) <
        config_.min_tracking_corners)
        return std::nullopt;

    std::vector<GridCorner> tracked_corners;
    tracked_corners.reserve(validation.visible_points.size());

    {
        const auto collect_t0 = cv::getTickCount();
        for (size_t m = 0; m < validation.visible_indices.size(); ++m) {
            const int old_idx = validation.visible_indices[m];
            if (old_idx < 0 || old_idx >= static_cast<int>(previous.corners.size()))
                continue;

            const cv::Point2f& uv = validation.visible_points[m];
            constexpr float kBorderMargin = 4.0f;
            if (uv.x < kBorderMargin || uv.y < kBorderMargin) continue;

            GridCorner c = previous.corners[old_idx];
            c.uv = uv;
            c.predicted = validation.visible_predicted[m];
            tracked_corners.push_back(c);
        }
        addTimingMs("build_visible_collect_ms", elapsedMs(collect_t0));
    }

    if (static_cast<int>(tracked_corners.size()) < config_.min_tracking_corners)
        return std::nullopt;

    // Spacing cleanup (geometric — not photometric).
    //
    // During axial rotation the projected spacing near the cylindrical edges
    // changes locally. LK plus TrackingValidator already checks frame-to-frame
    // grid-neighbour consistency, so the old global spacing pass is only safe
    // as a hard gate when the frame is stable. In moving frames it was dropping
    // long-lived real edge corners and causing visible block flicker.
    if (config_.tracking_spacing_min_rel > 0.0f) {
        const auto spacing_t0 = cv::getTickCount();
        const float q3_spacing = estimateMedianSpacing(previous);
        if (q3_spacing > 1.0f) {
            tracked_corners = removeOutlierCorners(
                tracked_corners, q3_spacing, config_.min_tracking_corners);
            if (validation.stable) {
                tracked_corners = filterBySpacingConsistency(
                    tracked_corners, q3_spacing,
                    config_.tracking_spacing_min_rel,
                    config_.tracking_spacing_max_rel,
                    config_.min_tracking_corners);
            }
        }
        addTimingMs("build_visible_spacing_cleanup_ms", elapsedMs(spacing_t0));
    }

    if (static_cast<int>(tracked_corners.size()) < config_.min_tracking_corners)
        return std::nullopt;

    const auto grid_t0 = cv::getTickCount();
    auto rebuilt = grid_builder_.buildFromCorners(
        tracked_corners,
        config_.duplicate_corner_dist_px,
        config_.min_tracking_corners,
        config_.min_tracking_cells,
        true,
        validation.stable);
    addTimingMs("grid_build_tracking_ms", elapsedMs(grid_t0));

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

    const auto prepare_t0 = cv::getTickCount();
    std::vector<cv::Point2f> prev_points;
    prev_points.reserve(last_detection_.corners.size());
    for (const auto& c : last_detection_.corners)
        prev_points.push_back(c.uv);
    addTimingMs("track_prepare_points_ms", elapsedMs(prepare_t0));

    pending_current_gray_pyramid_.clear();
    pending_current_gray_pyramid_frame_index_ = -1;
    const bool reuse_prev_pyramid = !last_gray_pyramid_.empty();

    const auto lk_t0 = cv::getTickCount();
    LKTrackingResult lk = lk_tracker_.track(
        last_gray_, gray, prev_points,
        config_.lk_win_size, config_.lk_max_level,
        config_.lk_max_iters, config_.lk_epsilon,
        config_.max_lk_error,
        reuse_prev_pyramid ? &last_gray_pyramid_ : nullptr,
        &pending_current_gray_pyramid_);
    addTimingMs("lk_ms", elapsedMs(lk_t0));
    if (reuse_prev_pyramid) {
        addTimingMs("lk_prev_pyramid_reused_count", 1.0);
    }
    if (!pending_current_gray_pyramid_.empty()) {
        pending_current_gray_pyramid_frame_index_ = frame_index_;
    }

    const auto validate_t0 = cv::getTickCount();
    TrackingValidationResult validation =
        tracking_validator_.validate(
            last_detection_, lk, gray.size(), config_);
    addTimingMs("tracking_validate_ms", elapsedMs(validate_t0));

    if (!validation.valid) return std::nullopt;

    // Cull right/bottom boundary.
    {
        const auto cull_t0 = cv::getTickCount();
        const float max_x = static_cast<float>(gray.cols) - 4.0f;
        const float max_y = static_cast<float>(gray.rows) - 4.0f;

        std::vector<int>         culled_indices;
        std::vector<cv::Point2f> culled_points;
        std::vector<bool>        culled_predicted;
        culled_indices.reserve(validation.visible_indices.size());
        culled_points .reserve(validation.visible_points.size());
        culled_predicted.reserve(validation.visible_predicted.size());

        for (size_t k = 0; k < validation.visible_points.size(); ++k) {
            const cv::Point2f& uv = validation.visible_points[k];
            if (uv.x >= max_x || uv.y >= max_y) continue;
            culled_indices.push_back(validation.visible_indices[k]);
            culled_points .push_back(uv);
            culled_predicted.push_back(
                k < validation.visible_predicted.size()
                    ? validation.visible_predicted[k]
                    : false);
        }

        validation.visible_indices   = std::move(culled_indices);
        validation.visible_points    = std::move(culled_points);
        validation.visible_predicted = std::move(culled_predicted);
        addTimingMs("tracking_cull_ms", elapsedMs(cull_t0));
    }

    const auto build_t0 = cv::getTickCount();
    auto detection = buildVisibleTrackedDetection(last_detection_, validation);
    addTimingMs("build_visible_tracked_ms", elapsedMs(build_t0));
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
    if (pending_current_gray_pyramid_frame_index_ == frame_index_ &&
        !pending_current_gray_pyramid_.empty()) {
        last_gray_pyramid_ = std::move(pending_current_gray_pyramid_);
        last_gray_pyramid_frame_index_ = frame_index_;
        pending_current_gray_pyramid_frame_index_ = -1;
    } else if (last_gray_pyramid_frame_index_ != frame_index_) {
        last_gray_pyramid_.clear();
        last_gray_pyramid_frame_index_ = -1;
    }

    if (!measured_detection.valid()) {
        last_detection_ = CheckerboardDetection{};
        persistent_corners_.clear();
        pending_completion_corners_.clear();
        last_gray_pyramid_.clear();
        pending_current_gray_pyramid_.clear();
        last_gray_pyramid_frame_index_ = -1;
        pending_current_gray_pyramid_frame_index_ = -1;
        tracking_active_ = false;
        return;
    }

    // First detection or full reset (non-tracking): initialise persistent set.
    if (!tracking_active_ || persistent_corners_.empty() ||
        !measured_detection.tracking) {
        persistent_corners_.clear();
        pending_completion_corners_.clear();
        persistent_corners_.reserve(measured_detection.corners.size());

        for (const auto& c : measured_detection.corners) {
            PersistentTrackedCorner pc;
            pc.corner        = c;
            pc.missed_frames = 0;
            pc.tracked       = true;
            pc.observed_frames = 1;
            pc.predicted_frames = c.predicted ? 1 : 0;
            pc.low_visibility_frames = 0;
            persistent_corners_.push_back(pc);
        }

        last_detection_  = measured_detection;
        tracking_active_ = last_detection_.valid();
        return;
    }

    std::vector<cv::Point2f> old_persistent_uvs;
    old_persistent_uvs.reserve(persistent_corners_.size());
    for (const auto& pc : persistent_corners_)
        old_persistent_uvs.push_back(pc.corner.uv);
    const bool unstable_tracking_update =
        measured_detection.tracking && !measured_detection.stable;

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
            if (!c.predicted) {
                persistent_corners_[idx].observed_frames += 1;
                persistent_corners_[idx].predicted_frames = 0;
            } else {
                persistent_corners_[idx].predicted_frames += 1;
            }
        } else {
            // New corner appeared (e.g. after rotation reveals hidden area).
            PersistentTrackedCorner pc;
            pc.corner        = c;
            pc.missed_frames = 0;
            pc.tracked       = true;
            pc.observed_frames = c.predicted ? 0 : 1;
            pc.predicted_frames = c.predicted ? 1 : 0;
            pc.low_visibility_frames = 0;
            persistent_corners_.push_back(pc);
        }
    }

    // Local grid-neighbour prediction for established corners that vanished
    // from LK/validation as a block. A single global homography is a poor
    // model for axial rotation of the cylindrical marker; adjacent grid
    // neighbours give a much better short-term motion estimate at the rim.
    {
        const int old_count = std::min(
            static_cast<int>(old_persistent_uvs.size()),
            static_cast<int>(persistent_corners_.size()));
        const float margin = 4.0f;
        const float min_sep =
            std::max(4.0f, config_.duplicate_corner_dist_px);
        const float min_sep2 = min_sep * min_sep;

        for (int pass = 0; pass < 4; ++pass) {
            bool changed = false;

            for (int k = 0; k < old_count; ++k) {
                auto& pc = persistent_corners_[k];
                if (pc.missed_frames == 0) continue;
                const bool long_established = pc.observed_frames >= 8;
                const bool locally_confirmed = pc.observed_frames >= 3;
                if (!locally_confirmed) continue;
                const bool very_long_established = pc.observed_frames >= 20;
                const int prediction_limit = [&]() {
                    if (unstable_tracking_update) {
                        return long_established ? 2 : 1;
                    }
                    if (very_long_established) {
                        return std::max(config_.max_missed_frames, 30);
                    }
                    if (long_established) {
                        return std::max(config_.max_missed_frames, 8);
                    }
                    return std::max(config_.max_missed_frames, 4);
                }();
                if (pc.predicted_frames >= prediction_limit)
                    continue;

                cv::Point2f displacement(0.0f, 0.0f);
                float weight_sum = 0.0f;
                int support = 0;

                for (int n = 0; n < old_count; ++n) {
                    if (n == k) continue;
                    const auto& nb = persistent_corners_[n];
                    if (nb.missed_frames != 0) continue;

                    const int di = std::abs(nb.corner.i - pc.corner.i);
                    const int dj = std::abs(nb.corner.j - pc.corner.j);
                    if (di + dj != 1) continue;

                    const float old_d = distf(old_persistent_uvs[k],
                                              old_persistent_uvs[n]);
                    const float new_d = distf(old_persistent_uvs[k],
                                              nb.corner.uv);
                    if (old_d < 2.0f || new_d < 2.0f) continue;

                    const float w = nb.corner.predicted ? 0.55f : 1.0f;
                    displacement += w * (nb.corner.uv - old_persistent_uvs[n]);
                    weight_sum += w;
                    ++support;
                }

                const int min_support =
                    unstable_tracking_update ? 2 : (long_established ? 1 : 2);
                if (support < min_support || weight_sum <= 0.0f) continue;

                const cv::Point2f projected =
                    old_persistent_uvs[k] + displacement * (1.0f / weight_sum);
                if (projected.x < margin || projected.y < margin ||
                    projected.x > static_cast<float>(gray.cols) - margin ||
                    projected.y > static_cast<float>(gray.rows) - margin)
                    continue;

                bool too_close = false;
                bool plausible_spacing = false;
                for (int n = 0; n < old_count; ++n) {
                    if (n == k) continue;
                    const auto& nb = persistent_corners_[n];
                    if (nb.missed_frames != 0) continue;

                    const float d2 = dist2(projected, nb.corner.uv);
                    if (d2 < min_sep2) {
                        too_close = true;
                        break;
                    }

                    const int di = std::abs(nb.corner.i - pc.corner.i);
                    const int dj = std::abs(nb.corner.j - pc.corner.j);
                    if (di + dj != 1) continue;

                    const float old_d = distf(old_persistent_uvs[k],
                                              old_persistent_uvs[n]);
                    const float new_d = distf(projected, nb.corner.uv);
                    if (old_d >= 2.0f) {
                        const float ratio = new_d / old_d;
                        if (ratio >= 0.25f && ratio <= 1.90f) {
                            plausible_spacing = true;
                        }
                    }
                }

                if (too_close || !plausible_spacing) continue;

                pc.corner.uv = projected;
                pc.corner.predicted = true;
                pc.missed_frames = 0;
                pc.tracked = true;
                pc.predicted_frames += 1;
                changed = true;
            }

            if (!changed) break;
        }
    }

    // Short-term motion-model fill for stable corners that vanished for a
    // single frame.  The previous validator can only project corners that were
    // still present in last_detection_; block losses have already disappeared
    // there.  This uses the persistent grid memory instead, so real corners can
    // survive brief LK/validation dropouts during rotation.
    if (!unstable_tracking_update &&
        old_persistent_uvs.size() >= 4 &&
        persistent_corners_.size() >= 4) {
        std::vector<cv::Point2f> h_src;
        std::vector<cv::Point2f> h_dst;
        h_src.reserve(persistent_corners_.size());
        h_dst.reserve(persistent_corners_.size());

        const int old_count = static_cast<int>(old_persistent_uvs.size());
        for (int k = 0; k < old_count &&
                        k < static_cast<int>(persistent_corners_.size()); ++k) {
            const auto& pc = persistent_corners_[k];
            if (pc.missed_frames != 0) continue;
            if (pc.corner.predicted) continue;
            h_src.push_back(old_persistent_uvs[k]);
            h_dst.push_back(pc.corner.uv);
        }

        if (h_src.size() >= 4) {
            cv::Mat inlier_mask;
            cv::Mat H = cv::findHomography(
                h_src, h_dst, cv::RANSAC,
                config_.max_tracking_homography_error_px,
                inlier_mask);
            if (!H.empty()) {
                if (H.depth() != CV_64F) {
                    cv::Mat H64;
                    H.convertTo(H64, CV_64F);
                    H = H64;
                }

                const float margin = 4.0f;
                const float min_sep2 =
                    std::max(4.0f, config_.duplicate_corner_dist_px) *
                    std::max(4.0f, config_.duplicate_corner_dist_px);

                for (int k = 0; k < old_count &&
                                k < static_cast<int>(persistent_corners_.size()); ++k) {
                    auto& pc = persistent_corners_[k];
                    if (pc.missed_frames == 0) continue;
                    if (pc.observed_frames < 3) continue;
                    const int prediction_limit =
                        measured_detection.stable && pc.observed_frames >= 20
                            ? std::max(config_.max_missed_frames, 30)
                            : config_.max_missed_frames;
                    if (pc.predicted_frames >= prediction_limit) continue;

                    cv::Point2f projected;
                    if (!projectHomographyPoint(H, old_persistent_uvs[k], projected))
                        continue;
                    if (projected.x < margin || projected.y < margin ||
                        projected.x > static_cast<float>(gray.cols) - margin ||
                        projected.y > static_cast<float>(gray.rows) - margin)
                        continue;

                    bool too_close = false;
                    bool has_grid_support = false;
                    for (const auto& nb : persistent_corners_) {
                        if (nb.missed_frames != 0) continue;
                        if (&nb == &pc) continue;

                        if (dist2(projected, nb.corner.uv) < min_sep2) {
                            too_close = true;
                            break;
                        }

                        const int di = std::abs(nb.corner.i - pc.corner.i);
                        const int dj = std::abs(nb.corner.j - pc.corner.j);
                        if (di + dj == 1)
                            has_grid_support = true;
                    }
                    if (too_close || !has_grid_support) continue;

                    pc.corner.uv = projected;
                    pc.corner.predicted = true;
                    pc.missed_frames = 0;
                    pc.tracked = true;
                    pc.predicted_frames += 1;
                }
            }
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
            if (pc.observed_frames >= 3) {
                continue;
            }

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
                if (pc.corner.predicted) continue;

                pc.visibility_score = computeCornerVisibilityScore(gray, pc, spacing);

                // EMA smoothing: damps single-frame dips from triggering
                // eviction while still reacting to genuine fade-out.
                pc.smoothed_visibility_score =
                    alpha * pc.visibility_score +
                    (1.0f - alpha) * pc.smoothed_visibility_score;

                const bool raw_invisible =
                    pc.visibility_score <
                    std::max(0.04f, config_.visibility_evict_threshold * 0.45f);

                const float low_visibility_threshold =
                    std::max(0.06f, config_.visibility_evict_threshold * 0.75f);
                if (pc.visibility_score < low_visibility_threshold) {
                    pc.low_visibility_frames += 1;
                } else if (pc.visibility_score >
                           config_.visibility_evict_threshold * 1.25f) {
                    pc.low_visibility_frames = 0;
                }

                const bool sustained_raw_invisible =
                    raw_invisible && pc.low_visibility_frames >= 3;
                const bool sustained_low_visibility =
                    pc.smoothed_visibility_score <
                        config_.visibility_evict_threshold * 0.60f &&
                    pc.low_visibility_frames >= 4;

                bool has_active_grid_neighbour = false;
                for (const auto& nb : persistent_corners_) {
                    if (&nb == &pc) continue;
                    if (nb.missed_frames != 0) continue;
                    const int di = std::abs(nb.corner.i - pc.corner.i);
                    const int dj = std::abs(nb.corner.j - pc.corner.j);
                    if (di + dj == 1) {
                        has_active_grid_neighbour = true;
                        break;
                    }
                }

                const bool confirmed_with_geometry =
                    pc.observed_frames >= 3 && has_active_grid_neighbour;
                const bool sustained_severe_invisible =
                    raw_invisible &&
                    pc.smoothed_visibility_score < 0.05f &&
                    pc.low_visibility_frames >= 8;

                const bool evict_unstable =
                    !measured_detection.stable &&
                    (confirmed_with_geometry
                         ? sustained_severe_invisible
                         : (sustained_raw_invisible ||
                            sustained_low_visibility));
                const bool evict_stable_severe =
                    measured_detection.stable &&
                    sustained_severe_invisible &&
                    pc.low_visibility_frames >= 12;

                if (evict_unstable || evict_stable_severe) {
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
            [this, &measured_detection](const PersistentTrackedCorner& pc) {
                const bool established = pc.observed_frames >= 8;
                const bool very_established = pc.observed_frames >= 20;
                const bool confirmed = pc.observed_frames >= 3;
                const int missed_limit = [&]() {
                    if (very_established) {
                        return std::max(
                            config_.max_missed_frames,
                            measured_detection.stable ? 90 : 24);
                    }
                    if (established) {
                        return std::max(
                            config_.max_missed_frames,
                            measured_detection.stable ? 30 : 20);
                    }
                    if (confirmed) {
                        return std::max(config_.max_missed_frames, 12);
                    }
                    return config_.max_missed_frames;
                }();
                const int predicted_limit = [&]() {
                    if (very_established) {
                        return std::max(
                            config_.max_missed_frames,
                            measured_detection.stable ? 45 : 10);
                    }
                    if (established) {
                        return std::max(
                            config_.max_missed_frames,
                            measured_detection.stable ? 12 : 8);
                    }
                    if (confirmed) {
                        return std::max(config_.max_missed_frames, 5);
                    }
                    return config_.max_missed_frames;
                }();
                return pc.missed_frames > missed_limit ||
                       pc.predicted_frames > predicted_limit;
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

    // Recovery-authoritative collision pruning.
    //
    // LK can keep tracking an old textured location even after that grid
    // corner rotated out of view.  If a newly visible recovery corner lands
    // close to it, the old long-established track used to block the new one
    // and then stayed visible through output hysteresis.  Resolve only true
    // conflicts here: pixel-near duplicates or strongly compressed adjacent
    // grid edges where recovery sees one side but not the other.
    if (recovery_detection && recovery_detection->valid()) {
        const float spacing = estimateMedianSpacing(measured_detection);
        if (spacing > 1.0f && !persistent_corners_.empty()) {
            auto recoveryHasGrid = [&](int i, int j) {
                for (const auto& c : recovery_detection->corners) {
                    if (c.i == i && c.j == j) return true;
                }
                return false;
            };

            auto activeSupport = [&](int idx) {
                int support = 0;
                const auto& c = persistent_corners_[idx].corner;
                for (int k = 0;
                     k < static_cast<int>(persistent_corners_.size());
                     ++k) {
                    if (k == idx) continue;
                    const auto& nb = persistent_corners_[k];
                    if (nb.missed_frames != 0) continue;
                    const int di = std::abs(nb.corner.i - c.i);
                    const int dj = std::abs(nb.corner.j - c.j);
                    if (di + dj == 1) ++support;
                }
                return support;
            };

            auto removalScore = [&](int idx) {
                const auto& pc = persistent_corners_[idx];
                float score = 0.0f;
                if (!recoveryHasGrid(pc.corner.i, pc.corner.j)) score += 6.0f;
                if (pc.corner.predicted) score += 4.0f;
                if (pc.smoothed_visibility_score < 0.18f) {
                    score += 3.0f;
                } else if (pc.smoothed_visibility_score < 0.35f) {
                    score += 1.0f;
                }
                if (pc.observed_frames < 3) score += 1.0f;
                score -= 0.75f * static_cast<float>(
                    std::min(activeSupport(idx), 4));
                return score;
            };

            auto evictIndex = [&](int idx) {
                auto& pc = persistent_corners_[idx];
                pc.missed_frames =
                    std::max(pc.missed_frames,
                             std::max(config_.max_missed_frames + 1, 16));
                pc.predicted_frames =
                    std::max(pc.predicted_frames,
                             std::max(config_.max_missed_frames + 1, 16));
                pc.tracked = false;
                pc.corner.predicted = true;
            };

            auto cellUsesGrid = [](const GridCell& cell, int i, int j) {
                return (i == cell.i && j == cell.j) ||
                       (i == cell.i + 1 && j == cell.j) ||
                       (i == cell.i + 1 && j == cell.j + 1) ||
                       (i == cell.i && j == cell.j + 1);
            };

            const float duplicate_d =
                std::max(config_.duplicate_corner_dist_px * 1.6f,
                         spacing * 0.42f);
            const float compressed_d = spacing * 0.58f;

            std::vector<char> evict(persistent_corners_.size(), 0);
            for (int a = 0;
                 a < static_cast<int>(persistent_corners_.size());
                 ++a) {
                if (persistent_corners_[a].missed_frames != 0 || evict[a])
                    continue;
                for (int b = a + 1;
                     b < static_cast<int>(persistent_corners_.size());
                     ++b) {
                    if (persistent_corners_[b].missed_frames != 0 ||
                        evict[b]) {
                        continue;
                    }

                    const auto& ca = persistent_corners_[a].corner;
                    const auto& cb = persistent_corners_[b].corner;
                    const int di = std::abs(ca.i - cb.i);
                    const int dj = std::abs(ca.j - cb.j);
                    if (di == 0 && dj == 0) continue;

                    const float d = distf(ca.uv, cb.uv);
                    const bool duplicate_collision = d < duplicate_d;
                    const bool compressed_adjacent =
                        (di + dj == 1) && d < compressed_d;
                    if (!duplicate_collision && !compressed_adjacent)
                        continue;

                    const bool a_recovered = recoveryHasGrid(ca.i, ca.j);
                    const bool b_recovered = recoveryHasGrid(cb.i, cb.j);

                    int remove_idx = -1;
                    if (a_recovered != b_recovered) {
                        remove_idx = a_recovered ? b : a;
                    } else if (duplicate_collision) {
                        remove_idx =
                            removalScore(a) >= removalScore(b) ? a : b;
                    } else {
                        // Adjacent compression can be a real perspective
                        // effect.  Only resolve it without recovery evidence
                        // when one side is already weakly supported.
                        const int a_support = activeSupport(a);
                        const int b_support = activeSupport(b);
                        if (a_support <= 1 || b_support <= 1) {
                            remove_idx =
                                removalScore(a) >= removalScore(b) ? a : b;
                        }
                    }

                    if (remove_idx >= 0) {
                        evict[remove_idx] = 1;
                    }
                }
            }

            const float min_cell_area = spacing * spacing * 0.20f;
            const float interior_margin =
                std::max(1.0f, spacing * 0.04f);
            const float outside_cell_margin =
                std::max(2.0f, spacing * 0.50f);
            for (int k = 0;
                 k < static_cast<int>(persistent_corners_.size());
                 ++k) {
                if (evict[k]) continue;
                auto& pc = persistent_corners_[k];
                if (pc.missed_frames != 0) continue;
                if (recoveryHasGrid(pc.corner.i, pc.corner.j)) continue;

                for (const auto& cell : recovery_detection->cells) {
                    if (cellUsesGrid(cell, pc.corner.i, pc.corner.j))
                        continue;
                    if (quadArea(cell.corner_uv) < min_cell_area)
                        continue;

                    const float inside_d =
                        pointInsideQuadDistance(cell.corner_uv,
                                                pc.corner.uv);
                    if (inside_d > interior_margin) {
                        evict[k] = 1;
                        break;
                    }
                }
            }

            for (int k = 0;
                 k < static_cast<int>(persistent_corners_.size());
                 ++k) {
                if (evict[k]) continue;
                auto& pc = persistent_corners_[k];
                if (pc.missed_frames != 0) continue;

                bool used_by_cell = false;
                float min_cell_dist = std::numeric_limits<float>::max();
                for (const auto& cell : recovery_detection->cells) {
                    if (quadArea(cell.corner_uv) < min_cell_area)
                        continue;
                    if (cellUsesGrid(cell, pc.corner.i, pc.corner.j)) {
                        used_by_cell = true;
                        break;
                    }
                    min_cell_dist = std::min(
                        min_cell_dist,
                        pointToQuadDistance(cell.corner_uv,
                                            pc.corner.uv));
                }
                if (used_by_cell) continue;
                if (min_cell_dist <= outside_cell_margin) continue;
                if (activeSupport(k) >= 2) continue;

                evict[k] = 1;
            }

            for (int k = 0;
                 k < static_cast<int>(persistent_corners_.size());
                 ++k) {
                if (evict[k]) evictIndex(k);
            }
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

    auto recoveryNeighbourCount = [&](const GridCorner& c) {
        int count = 0;
        for (const auto& nb : recovery_detection.corners) {
            if (nb.i == c.i && nb.j == c.j) continue;
            const int di = std::abs(nb.i - c.i);
            const int dj = std::abs(nb.j - c.j);
            if (di + dj == 1) {
                ++count;
            }
        }
        return count;
    };

    auto recoveryHasGrid = [&](int i, int j) {
        for (const auto& c : recovery_detection.corners) {
            if (c.i == i && c.j == j) return true;
        }
        return false;
    };

    auto recoveryCellUseCount = [&](const GridCorner& c) {
        int count = 0;
        for (const auto& cell : recovery_detection.cells) {
            const bool used =
                (c.i == cell.i && c.j == cell.j) ||
                (c.i == cell.i + 1 && c.j == cell.j) ||
                (c.i == cell.i + 1 && c.j == cell.j + 1) ||
                (c.i == cell.i && c.j == cell.j + 1);
            if (used) ++count;
        }
        return count;
    };

    auto activePersistentAt = [&](int i, int j, int skip_idx)
        -> const PersistentTrackedCorner* {
        for (int idx = 0;
             idx < static_cast<int>(persistent_corners_.size());
             ++idx) {
            if (idx == skip_idx) continue;
            const auto& pc = persistent_corners_[idx];
            if (pc.missed_frames != 0) continue;
            if (pc.corner.predicted) continue;
            if (pc.smoothed_visibility_score < 0.05f) continue;
            if (pc.corner.i == i && pc.corner.j == j) return &pc;
        }
        return nullptr;
    };

    auto persistentLocalError = [&](
        int i,
        int j,
        const cv::Point2f& uv,
        int skip_idx
    ) {
        float best_error = std::numeric_limits<float>::max();
        int count = 0;
        auto addExpected = [&](const cv::Point2f& expected) {
            best_error = std::min(best_error, distf(uv, expected));
            ++count;
        };

        const auto* im1 = activePersistentAt(i - 1, j, skip_idx);
        const auto* ip1 = activePersistentAt(i + 1, j, skip_idx);
        const auto* jm1 = activePersistentAt(i, j - 1, skip_idx);
        const auto* jp1 = activePersistentAt(i, j + 1, skip_idx);
        const auto* im2 = activePersistentAt(i - 2, j, skip_idx);
        const auto* ip2 = activePersistentAt(i + 2, j, skip_idx);
        const auto* jm2 = activePersistentAt(i, j - 2, skip_idx);
        const auto* jp2 = activePersistentAt(i, j + 2, skip_idx);

        if (im1 && ip1) {
            addExpected(0.5f * (im1->corner.uv + ip1->corner.uv));
        }
        if (jm1 && jp1) {
            addExpected(0.5f * (jm1->corner.uv + jp1->corner.uv));
        }
        if (im1 && im2) {
            addExpected(im1->corner.uv +
                        (im1->corner.uv - im2->corner.uv));
        }
        if (ip1 && ip2) {
            addExpected(ip1->corner.uv +
                        (ip1->corner.uv - ip2->corner.uv));
        }
        if (jm1 && jm2) {
            addExpected(jm1->corner.uv +
                        (jm1->corner.uv - jm2->corner.uv));
        }
        if (jp1 && jp2) {
            addExpected(jp1->corner.uv +
                        (jp1->corner.uv - jp2->corner.uv));
        }

        const int dirs[2] = {-1, 1};
        for (int di : dirs) {
            for (int dj : dirs) {
                const auto* edge_i =
                    activePersistentAt(i - di, j, skip_idx);
                const auto* edge_j =
                    activePersistentAt(i, j - dj, skip_idx);
                const auto* diag =
                    activePersistentAt(i - di, j - dj, skip_idx);
                if (!edge_i || !edge_j || !diag) continue;
                addExpected(
                    edge_i->corner.uv +
                    edge_j->corner.uv -
                    diag->corner.uv);
            }
        }

        return std::pair<int, float>{count, best_error};
    };

    auto evictActiveBlocker = [&](PersistentTrackedCorner& pc) {
        pc.missed_frames =
            std::max(pc.missed_frames,
                     std::max(config_.max_missed_frames + 1, 16));
        pc.predicted_frames =
            std::max(pc.predicted_frames,
                     std::max(config_.max_missed_frames + 1, 16));
        pc.tracked = false;
        pc.corner.predicted = true;
    };

    int recovery_position_corrections = 0;
    double recovery_position_correction_error_sum = 0.0;

    for (const auto& rc : recovery_detection.corners) {
        const int recovery_neighbours = recoveryNeighbourCount(rc);
        const bool strongly_supported_recovery = recovery_neighbours >= 2;
        const int recovery_cell_uses = recoveryCellUseCount(rc);
        const bool authoritative_recovery =
            (strongly_supported_recovery && recovery_cell_uses > 0) ||
            recovery_neighbours >= 3;

        // --- Fix B: position correction for actively tracked corners ---
        const int existing_idx =
            findPersistentCornerByGrid(persistent_corners_, rc.i, rc.j);

        if (existing_idx >= 0) {
            auto& pc = persistent_corners_[existing_idx];

            // Only correct active corners — stale ones are handled by eviction.
            const float d = distf(pc.corner.uv, rc.uv);
            if (pc.missed_frames == 0) {
                const float max_confirm_d =
                    spacing *
                    (authoritative_recovery
                         ? (pc.observed_frames >= 8 ? 2.70f : 2.10f)
                     : strongly_supported_recovery
                          ? (pc.observed_frames >= 8 ? 2.30f : 1.45f)
                          : (pc.observed_frames >= 8 ? 1.5f : 0.9f));
                if (pc.corner.predicted && d < max_confirm_d) {
                    const auto [new_error_count, new_error] =
                        persistentLocalError(
                            rc.i, rc.j, rc.uv, existing_idx);
                    if (new_error_count >= 2 &&
                        new_error >
                            spacing *
                                (authoritative_recovery ? 0.95f : 0.70f)) {
                        continue;
                    }
                    pc.corner = rc;
                    pc.missed_frames = 0;
                    pc.tracked = true;
                    pc.observed_frames += 1;
                    pc.predicted_frames = 0;
                    pc.low_visibility_frames = 0;
                    pc.visibility_score = 1.0f;
                    pc.smoothed_visibility_score =
                        std::max(pc.smoothed_visibility_score, 0.75f);
                    ++recovery_position_corrections;
                    recovery_position_correction_error_sum += d;
                } else if (!pc.corner.predicted && w > 0.0f) {
                    // Blend LK position toward recovery position.
                    const float supported_corr_d =
                        authoritative_recovery
                            ? spacing *
                                  (pc.observed_frames >= 8 ? 2.40f : 1.80f)
                        : strongly_supported_recovery
                            ? spacing *
                                  (pc.observed_frames >= 8 ? 1.85f : 1.30f)
                            : max_corr_d;
                    if (d < supported_corr_d && d > spacing * 0.025f) {
                        const auto [old_error_count, old_error] =
                            persistentLocalError(
                                pc.corner.i, pc.corner.j,
                                pc.corner.uv, existing_idx);
                        const auto [new_error_count, new_error] =
                            persistentLocalError(
                                rc.i, rc.j, rc.uv, existing_idx);
                        if (old_error_count >= 2 &&
                            new_error_count >= 2 &&
                            new_error >
                                old_error + spacing * 0.20f &&
                            pc.smoothed_visibility_score >= 0.35f &&
                            (!authoritative_recovery ||
                             new_error > spacing * 0.95f)) {
                            continue;
                        }
                        const float corr_w =
                            authoritative_recovery
                                ? std::max(
                                      w,
                                      d > max_corr_d ? 0.98f : 0.90f)
                        : strongly_supported_recovery
                                ? std::max(w, d > max_corr_d ? 0.95f : 0.82f)
                                : w;
                        pc.corner.uv =
                            (1.0f - corr_w) * pc.corner.uv +
                            corr_w * rc.uv;
                        pc.corner.predicted = false;
                        if (strongly_supported_recovery) {
                            pc.low_visibility_frames = 0;
                            pc.visibility_score =
                                std::max(pc.visibility_score, 0.75f);
                            pc.smoothed_visibility_score =
                                std::max(pc.smoothed_visibility_score, 0.60f);
                        }
                        ++recovery_position_corrections;
                        recovery_position_correction_error_sum += d;
                    }
                }
            } else if (pc.missed_frames > 0) {
                const float max_reactivate_d =
                    spacing * (pc.observed_frames >= 8 ? 1.5f : 0.9f);
                const bool trusted_grid_reactivation =
                    pc.observed_frames >= 3 &&
                    recovery_neighbours >= 1 &&
                    d < spacing * 6.0f;
                if (d < max_reactivate_d || trusted_grid_reactivation) {
                    bool too_close_to_active = false;
                    for (int k = 0; k < static_cast<int>(persistent_corners_.size()); ++k) {
                        if (k == existing_idx) continue;
                        auto& other = persistent_corners_[k];
                        if (other.missed_frames > 0) continue;
                        if (distf(other.corner.uv, rc.uv) < min_dist) {
                            const bool weak_active_blocker =
                                recovery_neighbours >= 1 &&
                                (other.corner.predicted ||
                                 other.smoothed_visibility_score < 0.18f ||
                                 (strongly_supported_recovery &&
                                  !recoveryHasGrid(other.corner.i,
                                                   other.corner.j)));
                            if (weak_active_blocker) {
                                evictActiveBlocker(other);
                                continue;
                            }
                            too_close_to_active = true;
                            break;
                        }
                    }
                    if (!too_close_to_active) {
                        pc.corner = rc;
                        pc.missed_frames = 0;
                        pc.tracked = true;
                        pc.observed_frames += 1;
                        pc.predicted_frames = 0;
                        pc.low_visibility_frames = 0;
                        pc.visibility_score = 1.0f;
                        pc.smoothed_visibility_score =
                            std::max(pc.smoothed_visibility_score, 0.75f);
                    }
                }
            }
            continue;
        }

        // --- Fix C: inject new corners not yet in persistent set ---
        // Only check against active persistent corners to avoid stale
        // corners blocking newly visible ones.
        bool too_close = false;
        for (auto& pc : persistent_corners_) {
            if (pc.missed_frames > 0) continue;  // ignore stale
            if (distf(pc.corner.uv, rc.uv) < min_dist) {
                const bool weak_active_blocker =
                    recovery_neighbours >= 1 &&
                    (pc.corner.predicted ||
                     pc.smoothed_visibility_score < 0.18f ||
                     (strongly_supported_recovery &&
                      !recoveryHasGrid(pc.corner.i, pc.corner.j)));
                if (weak_active_blocker) {
                    evictActiveBlocker(pc);
                    continue;
                }
                too_close = true;
                break;
            }
        }
        if (too_close) continue;

        PersistentTrackedCorner pc;
        pc.corner        = rc;
        pc.missed_frames = 0;
        pc.tracked       = true;
        pc.observed_frames = 1;
        pc.predicted_frames = 0;
        pc.low_visibility_frames = 0;
        persistent_corners_.push_back(pc);
    }

    if (recovery_position_corrections > 0) {
        addTimingMs(
            "recovery_position_correction_count",
            static_cast<double>(recovery_position_corrections));
        addTimingMs(
            "recovery_position_correction_error_px",
            recovery_position_correction_error_sum /
                static_cast<double>(recovery_position_corrections));
    }
}

int CheckerboardDetector::tryCompleteMissingCorners(
    const cv::Mat& gray,
    bool tracking
) {
    const auto attempt_t0 = cv::getTickCount();
    struct ScopedLocalCompletionTimer {
        const CheckerboardDetector* self;
        std::int64_t start;
        ~ScopedLocalCompletionTimer() {
            self->addTimingMs(
                "tracking_local_completion_attempt_ms",
                CheckerboardDetector::elapsedMs(start));
        }
    } attempt_timer{this, attempt_t0};

    if (!tracking || gray.empty() || !tracking_active_ ||
        !last_detection_.valid()) {
        return 0;
    }

    const float spacing = estimateMedianSpacing(last_detection_);
    if (spacing < 4.0f) return 0;

    const cv::Rect roi = trackingRecoveryRoi(
        last_detection_, gray.size(), config_);
    if (roi.empty()) return 0;

    const cv::Rect image_rect(0, 0, gray.cols, gray.rows);
    const cv::Rect active_roi = roi & image_rect;
    if (active_roi.width <= 0 || active_roi.height <= 0) return 0;

    const cv::Mat source = gray(active_roi);
    const cv::Point2f offset(
        static_cast<float>(active_roi.x),
        static_cast<float>(active_roi.y));

    auto sameRect = [](const cv::Rect& a, const cv::Rect& b) {
        return a.x == b.x &&
               a.y == b.y &&
               a.width == b.width &&
               a.height == b.height;
    };

    CornerDetectionResult raw;
    std::vector<RefinedCorner> refined;
    bool reused_recovery_region = false;

    if (recovery_region_cache_ &&
        recovery_region_cache_->frame_index == frame_index_ &&
        sameRect(recovery_region_cache_->roi, active_roi)) {
        raw = recovery_region_cache_->raw;
        refined = recovery_region_cache_->refined;
        reused_recovery_region = true;
        addTimingMs("tracking_local_completion_reused_recovery_count", 1.0);
    } else {
        const auto corner_t0 = cv::getTickCount();
        raw = corner_detector_.detect(
            source,
            config_.max_recovery_corners,
            config_.saddle_sigma,
            config_.saddle_response_threshold);
        addTimingMs(
            "tracking_local_completion_corner_detect_ms",
            elapsedMs(corner_t0));
        for (const auto& timing : raw.timings_ms) {
            addTimingMs(
                std::string("tracking_local_completion_") + timing.first,
                timing.second);
        }
    }

    CornerRefinementConfig refine_config;
    refine_config.radius             = config_.saddle_radius;
    refine_config.iterations         = config_.saddle_iterations;
    refine_config.max_angle_bias_deg = config_.saddle_max_angle_bias_deg;
    refine_config.correlation_drop   = config_.saddle_correlation_drop;
    refine_config.merge_radius_px =
        std::max(1.0f, config_.duplicate_corner_dist_px);
    refine_config.quadrant_half_r            = config_.quadrant_half_r;
    refine_config.quadrant_min_contrast      = config_.quadrant_min_contrast;
    refine_config.quadrant_max_diagonal_diff =
        config_.quadrant_max_diagonal_diff;
    refine_config.subpix_win_size  = config_.saddle_subpix_win_size;
    refine_config.subpix_max_iters = config_.saddle_subpix_max_iters;
    refine_config.subpix_epsilon   = config_.saddle_subpix_epsilon;

    if (!reused_recovery_region &&
        !raw.points.empty() &&
        !raw.grad_x.empty() &&
        !raw.grad_y.empty()) {
        const auto refine_t0 = cv::getTickCount();
        refined = corner_refiner_.refine(
            source, raw.points, raw.grad_x, raw.grad_y, refine_config);
        addTimingMs(
            "tracking_local_completion_refine_ms",
            elapsedMs(refine_t0));
    }

    std::vector<cv::Point2f> candidates;
    candidates.reserve(refined.size());
    for (const auto& c : refined) {
        if (!c.valid) continue;
        candidates.push_back(c.uv + offset);
    }
    addTimingMs(
        "tracking_local_completion_detected_candidate_count",
        static_cast<double>(candidates.size()));
    std::vector<char> guided_candidate_flags(candidates.size(), 0);

    struct ActiveCorner {
        int i = 0;
        int j = 0;
        cv::Point2f uv;
        int persistent_idx = -1;
    };

    std::vector<ActiveCorner> active;
    active.reserve(persistent_corners_.size());
    for (int idx = 0; idx < static_cast<int>(persistent_corners_.size());
         ++idx) {
        const auto& pc = persistent_corners_[idx];
        if (pc.missed_frames != 0) continue;
        if (pc.corner.predicted) continue;
        if (pc.smoothed_visibility_score < 0.05f) continue;
        active.push_back({pc.corner.i, pc.corner.j, pc.corner.uv, idx});
    }
    if (active.size() < 2) return 0;

    auto hasActiveGrid = [&](int i, int j) {
        for (const auto& c : active) {
            if (c.i == i && c.j == j) return true;
        }
        return false;
    };

    auto nearActiveCorner = [&](const cv::Point2f& uv, float radius) {
        const float r2 = radius * radius;
        for (const auto& c : active) {
            if (dist2(c.uv, uv) < r2) return true;
        }
        return false;
    };

    auto nearOtherActiveCorner = [&](
        const cv::Point2f& uv,
        int skip_i,
        int skip_j,
        float radius
    ) {
        const float r2 = radius * radius;
        for (const auto& c : active) {
            if (c.i == skip_i && c.j == skip_j) continue;
            if (dist2(c.uv, uv) < r2) return true;
        }
        return false;
    };

    auto findActive = [&](int i, int j) -> const ActiveCorner* {
        for (const auto& c : active) {
            if (c.i == i && c.j == j) return &c;
        }
        return nullptr;
    };

    auto bestCandidateNear = [&](
        const cv::Point2f& expected,
        float radius,
        const std::vector<char>* used_candidates = nullptr
    ) {
        int best_idx = -1;
        float best_d = radius;
        for (int idx = 0; idx < static_cast<int>(candidates.size()); ++idx) {
            if (used_candidates && (*used_candidates)[idx]) continue;
            const float d = distf(candidates[idx], expected);
            if (d < best_d) {
                best_d = d;
                best_idx = idx;
            }
        }
        return std::pair<int, float>{best_idx, best_d};
    };

    auto updateActiveCopy = [&](int i, int j, const cv::Point2f& uv) {
        for (auto& c : active) {
            if (c.i == i && c.j == j) {
                c.uv = uv;
                return;
            }
        }
    };

    auto upsertActiveCopy = [&](
        int i,
        int j,
        const cv::Point2f& uv,
        int persistent_idx
    ) {
        for (auto& c : active) {
            if (c.i == i && c.j == j) {
                c.uv = uv;
                c.persistent_idx = persistent_idx;
                return;
            }
        }
        active.push_back({i, j, uv, persistent_idx});
    };

    struct LocalExpectation {
        int count = 0;
        cv::Point2f expected{0.0f, 0.0f};
        float spread = 0.0f;
    };

    auto localExpectation = [&](int i, int j) {
        LocalExpectation result;
        std::vector<cv::Point2f> expected_positions;
        expected_positions.reserve(8);

        const ActiveCorner* im1 = findActive(i - 1, j);
        const ActiveCorner* ip1 = findActive(i + 1, j);
        const ActiveCorner* jm1 = findActive(i, j - 1);
        const ActiveCorner* jp1 = findActive(i, j + 1);
        const ActiveCorner* im2 = findActive(i - 2, j);
        const ActiveCorner* ip2 = findActive(i + 2, j);
        const ActiveCorner* jm2 = findActive(i, j - 2);
        const ActiveCorner* jp2 = findActive(i, j + 2);

        if (im1 && ip1) {
            expected_positions.push_back(0.5f * (im1->uv + ip1->uv));
        }
        if (jm1 && jp1) {
            expected_positions.push_back(0.5f * (jm1->uv + jp1->uv));
        }
        if (im1 && im2) {
            expected_positions.push_back(im1->uv + (im1->uv - im2->uv));
        }
        if (ip1 && ip2) {
            expected_positions.push_back(ip1->uv + (ip1->uv - ip2->uv));
        }
        if (jm1 && jm2) {
            expected_positions.push_back(jm1->uv + (jm1->uv - jm2->uv));
        }
        if (jp1 && jp2) {
            expected_positions.push_back(jp1->uv + (jp1->uv - jp2->uv));
        }

        const int dirs[2] = {-1, 1};
        for (int di : dirs) {
            for (int dj : dirs) {
                const ActiveCorner* edge_i = findActive(i - di, j);
                const ActiveCorner* edge_j = findActive(i, j - dj);
                const ActiveCorner* diag = findActive(i - di, j - dj);
                if (!edge_i || !edge_j || !diag) continue;
                expected_positions.push_back(edge_i->uv + edge_j->uv -
                                             diag->uv);
            }
        }

        if (expected_positions.empty()) return result;

        for (const auto& p : expected_positions) result.expected += p;
        result.expected *=
            1.0f / static_cast<float>(expected_positions.size());
        result.count = static_cast<int>(expected_positions.size());

        for (const auto& p : expected_positions) {
            result.spread = std::max(result.spread,
                                     distf(p, result.expected));
        }
        return result;
    };

    int corrected = 0;
    double correction_error_sum = 0.0;
    std::vector<char> corrected_candidate_used(candidates.size(), 0);

    int saddle_snap_count = 0;
    int saddle_snap_predicted_count = 0;
    double saddle_snap_error_sum = 0.0;

    for (int idx = 0; idx < static_cast<int>(persistent_corners_.size());
         ++idx) {
        auto& pc = persistent_corners_[idx];
        if (pc.missed_frames != 0) continue;
        if (pc.smoothed_visibility_score < 0.05f) continue;

        const bool predicted_corner =
            pc.corner.predicted || pc.predicted_frames > 0;
        const bool weak_track =
            pc.smoothed_visibility_score < 0.45f ||
            pc.low_visibility_frames > 0;
        const LocalExpectation expectation =
            localExpectation(pc.corner.i, pc.corner.j);
        const bool use_expectation_anchor =
            expectation.count >= 2 &&
            expectation.spread <= spacing * 0.70f &&
            (predicted_corner || weak_track);
        const float snap_radius =
            std::max(
                4.0f,
                spacing *
                    (use_expectation_anchor
                         ? (predicted_corner ? 0.55f : 0.38f) :
                     predicted_corner ? 0.55f :
                     weak_track ? 0.40f : 0.28f));
        const cv::Point2f snap_anchor =
            use_expectation_anchor ? expectation.expected : pc.corner.uv;

        const auto [best_idx, candidate_d] =
            bestCandidateNear(
                snap_anchor,
                snap_radius,
                &corrected_candidate_used);
        if (best_idx < 0) continue;
        if (!use_expectation_anchor &&
            !predicted_corner &&
            candidate_d < 0.75f) {
            continue;
        }

        const cv::Point2f candidate_uv = candidates[best_idx];
        if (nearOtherActiveCorner(
                candidate_uv,
                pc.corner.i,
                pc.corner.j,
                spacing * 0.35f)) {
            continue;
        }

        bool geometry_ok = true;
        if (expectation.count >= 2) {
            const float current_error =
                distf(pc.corner.uv, expectation.expected);
            const float candidate_error =
                distf(candidate_uv, expectation.expected);
            const float max_abs_error =
                spacing * (predicted_corner ? 0.90f : 0.70f);
            const float max_worse_error =
                current_error +
                spacing * (predicted_corner ? 0.30f : 0.12f);
            geometry_ok =
                expectation.spread <= spacing * 0.70f &&
                candidate_error <= max_abs_error &&
                candidate_error <= max_worse_error;
        } else {
            geometry_ok =
                candidate_d <=
                spacing * (predicted_corner ? 0.45f : 0.22f);
        }
        if (!geometry_ok) continue;

        pc.corner.uv = candidate_uv;
        pc.corner.predicted = false;
        pc.missed_frames = 0;
        pc.tracked = true;
        if (predicted_corner) {
            pc.observed_frames += 1;
        }
        pc.predicted_frames = 0;
        pc.low_visibility_frames = 0;
        pc.visibility_score = std::max(pc.visibility_score, 0.80f);
        pc.smoothed_visibility_score =
            std::max(pc.smoothed_visibility_score, 0.65f);

        upsertActiveCopy(pc.corner.i, pc.corner.j, pc.corner.uv, idx);
        corrected_candidate_used[best_idx] = 1;
        saddle_snap_error_sum += candidate_d;
        ++saddle_snap_count;
        if (predicted_corner) ++saddle_snap_predicted_count;
    }

    if (saddle_snap_count > 0) {
        addTimingMs(
            "tracking_saddle_snap_count",
            static_cast<double>(saddle_snap_count));
        addTimingMs(
            "tracking_saddle_snap_error_px",
            saddle_snap_error_sum /
                static_cast<double>(saddle_snap_count));
        if (saddle_snap_predicted_count > 0) {
            addTimingMs(
                "tracking_saddle_snap_predicted_count",
                static_cast<double>(saddle_snap_predicted_count));
        }
    }

    for (const auto& c : active) {
        if (c.persistent_idx < 0 ||
            c.persistent_idx >=
                static_cast<int>(persistent_corners_.size())) {
            continue;
        }

        std::vector<cv::Point2f> expected_positions;
        expected_positions.reserve(4);

        const ActiveCorner* im1 = findActive(c.i - 1, c.j);
        const ActiveCorner* ip1 = findActive(c.i + 1, c.j);
        const ActiveCorner* jm1 = findActive(c.i, c.j - 1);
        const ActiveCorner* jp1 = findActive(c.i, c.j + 1);
        const ActiveCorner* im2 = findActive(c.i - 2, c.j);
        const ActiveCorner* ip2 = findActive(c.i + 2, c.j);
        const ActiveCorner* jm2 = findActive(c.i, c.j - 2);
        const ActiveCorner* jp2 = findActive(c.i, c.j + 2);

        if (im1 && ip1) expected_positions.push_back(0.5f * (im1->uv + ip1->uv));
        if (jm1 && jp1) expected_positions.push_back(0.5f * (jm1->uv + jp1->uv));
        if (im1 && im2) expected_positions.push_back(im1->uv + (im1->uv - im2->uv));
        if (ip1 && ip2) expected_positions.push_back(ip1->uv + (ip1->uv - ip2->uv));
        if (jm1 && jm2) expected_positions.push_back(jm1->uv + (jm1->uv - jm2->uv));
        if (jp1 && jp2) expected_positions.push_back(jp1->uv + (jp1->uv - jp2->uv));

        if (expected_positions.size() < 2) continue;

        cv::Point2f expected(0.0f, 0.0f);
        for (const auto& p : expected_positions) expected += p;
        expected *= 1.0f / static_cast<float>(expected_positions.size());

        float spread = 0.0f;
        for (const auto& p : expected_positions) {
            spread = std::max(spread, distf(p, expected));
        }
        if (spread > spacing * 0.45f) continue;

        auto& pc = persistent_corners_[c.persistent_idx];
        const float current_error = distf(pc.corner.uv, expected);
        const bool suspicious_current =
            current_error > spacing * 0.30f ||
            pc.smoothed_visibility_score < 0.22f ||
            pc.low_visibility_frames >= 2;
        if (!suspicious_current) continue;

        const auto [best_idx, candidate_error] =
            bestCandidateNear(
                expected,
                spacing * 0.30f,
                &corrected_candidate_used);
        if (best_idx < 0) continue;

        const cv::Point2f candidate_uv = candidates[best_idx];
        if (nearOtherActiveCorner(
                candidate_uv, pc.corner.i, pc.corner.j,
                spacing * 0.38f)) {
            continue;
        }
        if (candidate_error >= current_error * 0.70f &&
            pc.smoothed_visibility_score >= 0.18f) {
            continue;
        }

        const float snap_w =
            current_error > spacing * 0.55f ? 0.90f : 0.70f;
        pc.corner.uv =
            (1.0f - snap_w) * pc.corner.uv + snap_w * candidate_uv;
        pc.corner.predicted = false;
        pc.missed_frames = 0;
        pc.tracked = true;
        pc.predicted_frames = 0;
        pc.low_visibility_frames = 0;
        pc.visibility_score = std::max(pc.visibility_score, 0.70f);
        pc.smoothed_visibility_score =
            std::max(pc.smoothed_visibility_score, 0.55f);

        updateActiveCopy(pc.corner.i, pc.corner.j, pc.corner.uv);
        corrected_candidate_used[best_idx] = 1;
        correction_error_sum += current_error;
        ++corrected;
    }

    if (corrected > 0) {
        addTimingMs(
            "tracking_local_correction_count",
            static_cast<double>(corrected));
        addTimingMs(
            "tracking_local_correction_error_px",
            correction_error_sum / static_cast<double>(corrected));
    }

    struct Proposal {
        int i = 0;
        int j = 0;
        cv::Point2f expected;
        int support = 0;
    };

    std::vector<Proposal> proposals;
    proposals.reserve(active.size() * 2);
    int cell_proposal_count = 0;
    int edge_proposal_count = 0;
    int line_proposal_count = 0;

    auto addProposal = [&](
        int i,
        int j,
        const cv::Point2f& expected,
        int support_weight
    ) -> bool {
        if (hasActiveGrid(i, j)) return false;
        if (expected.x < 4.0f || expected.y < 4.0f) return false;
        if (expected.x >= static_cast<float>(gray.cols) - 4.0f)
            return false;
        if (expected.y >= static_cast<float>(gray.rows) - 4.0f)
            return false;
        const float active_guard =
            spacing * (support_weight >= 3 ? 0.25f : 0.35f);
        if (nearActiveCorner(expected, active_guard)) return false;

        for (auto& p : proposals) {
            if (p.i != i || p.j != j) continue;
            if (distf(p.expected, expected) > spacing * 0.75f) continue;
            p.expected =
                (p.expected * static_cast<float>(p.support) +
                expected * static_cast<float>(support_weight)) *
                (1.0f / static_cast<float>(p.support + support_weight));
            p.support += support_weight;
            return true;
        }

        proposals.push_back({i, j, expected, support_weight});
        return true;
    };

    auto addCellCompletionProposal = [&](
        const ActiveCorner& a,
        const ActiveCorner& b,
        const ActiveCorner& c,
        int target_i,
        int target_j
    ) {
        const cv::Point2f ab = b.uv - a.uv;
        const cv::Point2f ac = c.uv - a.uv;
        const float dab = distf(a.uv, b.uv);
        const float dac = distf(a.uv, c.uv);
        if (dab < spacing * 0.30f || dac < spacing * 0.30f) return;
        if (dab > spacing * 2.35f || dac > spacing * 2.35f) return;
        const float area = std::abs(ab.x * ac.y - ab.y * ac.x);
        if (area < dab * dac * 0.18f) return;
        if (addProposal(target_i, target_j, b.uv + c.uv - a.uv, 3)) {
            ++cell_proposal_count;
        }
    };

    for (const auto& a : active) {
        const int dirs[2] = {-1, 1};
        for (int di : dirs) {
            for (int dj : dirs) {
                const ActiveCorner* b = findActive(a.i + di, a.j);
                const ActiveCorner* c = findActive(a.i, a.j + dj);
                if (!b || !c) continue;
                addCellCompletionProposal(
                    a, *b, *c, a.i + di, a.j + dj);
            }
        }
    }

    for (size_t a = 0; a < active.size(); ++a) {
        for (size_t b = a + 1; b < active.size(); ++b) {
            const auto& ca = active[a];
            const auto& cb = active[b];
            const int di = cb.i - ca.i;
            const int dj = cb.j - ca.j;
            const int adi = std::abs(di);
            const int adj = std::abs(dj);
            const cv::Point2f delta = cb.uv - ca.uv;
            const float d = distf(ca.uv, cb.uv);

            if (adi + adj == 1) {
                if (d < spacing * 0.35f || d > spacing * 2.20f)
                    continue;
                if (addProposal(cb.i + di, cb.j + dj, cb.uv + delta, 1)) {
                    ++line_proposal_count;
                }
                if (addProposal(ca.i - di, ca.j - dj, ca.uv - delta, 1)) {
                    ++line_proposal_count;
                }
            } else if ((adi == 2 && adj == 0) ||
                       (adi == 0 && adj == 2)) {
                if (d < spacing * 0.70f || d > spacing * 4.00f)
                    continue;
                if (addProposal(
                    (ca.i + cb.i) / 2,
                    (ca.j + cb.j) / 2,
                    0.5f * (ca.uv + cb.uv),
                    2)) {
                    ++line_proposal_count;
                }
            }
        }
    }

    addTimingMs(
        "tracking_local_completion_cell_proposal_count",
        static_cast<double>(cell_proposal_count));
    addTimingMs(
        "tracking_local_completion_edge_proposal_count",
        static_cast<double>(edge_proposal_count));
    addTimingMs(
        "tracking_local_completion_line_proposal_count",
        static_cast<double>(line_proposal_count));
    addTimingMs(
        "tracking_local_completion_proposal_count",
        static_cast<double>(proposals.size()));
    if (proposals.empty()) return 0;

    std::sort(
        proposals.begin(), proposals.end(),
        [](const Proposal& a, const Proposal& b) {
            if (a.support != b.support) return a.support > b.support;
            return a.expected.x + a.expected.y < b.expected.x + b.expected.y;
        });

    auto nearCandidate = [&](const cv::Point2f& uv, float radius) {
        const float r2 = radius * radius;
        for (const auto& c : candidates) {
            if (dist2(c, uv) < r2) return true;
        }
        return false;
    };

    int guided_seed_count = 0;
    int guided_valid_count = 0;
    if (!raw.grad_x.empty() && !raw.grad_y.empty()) {
        std::vector<cv::Point2f> guided_seeds;
        guided_seeds.reserve(32);

        const float margin =
            static_cast<float>(std::max(5, refine_config.radius + 2));
        for (const auto& p : proposals) {
            if (guided_seed_count >= 32) break;
            if (p.support < 2) continue;
            if (nearCandidate(p.expected, spacing * 0.18f)) continue;
            if (nearActiveCorner(p.expected, spacing * 0.22f)) continue;

            const cv::Point2f local = p.expected - offset;
            if (local.x < margin || local.y < margin) continue;
            if (local.x >= static_cast<float>(source.cols) - margin)
                continue;
            if (local.y >= static_cast<float>(source.rows) - margin)
                continue;

            guided_seeds.push_back(local);
            ++guided_seed_count;
        }

        if (!guided_seeds.empty()) {
            CornerRefinementConfig guided_config = refine_config;
            guided_config.merge_radius_px =
                std::max(1.0f, config_.duplicate_corner_dist_px * 0.75f);

            const auto guided_t0 = cv::getTickCount();
            const auto guided_refined = corner_refiner_.refine(
                source,
                guided_seeds,
                raw.grad_x,
                raw.grad_y,
                guided_config);
            addTimingMs(
                "tracking_local_completion_guided_refine_ms",
                elapsedMs(guided_t0));

            for (const auto& c : guided_refined) {
                if (!c.valid) continue;
                const cv::Point2f full_uv = c.uv + offset;
                if (nearActiveCorner(full_uv, spacing * 0.24f)) continue;
                if (nearCandidate(full_uv, spacing * 0.18f)) continue;
                candidates.push_back(full_uv);
                corrected_candidate_used.push_back(0);
                guided_candidate_flags.push_back(1);
                ++guided_valid_count;
            }
        }
    }
    addTimingMs(
        "tracking_local_completion_guided_seed_count",
        static_cast<double>(guided_seed_count));
    addTimingMs(
        "tracking_local_completion_guided_candidate_count",
        static_cast<double>(guided_valid_count));
    addTimingMs(
        "tracking_local_completion_candidate_count",
        static_cast<double>(candidates.size()));
    if (candidates.empty()) return 0;

    std::vector<char> used = corrected_candidate_used;
    const float search_r = std::max(5.0f, spacing * 0.55f);
    const float duplicate_r = std::max(
        config_.duplicate_corner_dist_px * 1.5f,
        spacing * 0.38f);
    auto completionDuplicateRadius = [&](int support) {
        if (support >= 3) {
            return std::max(
                config_.duplicate_corner_dist_px * 1.1f,
                spacing * 0.28f);
        }
        if (support >= 2) {
            return std::max(
                config_.duplicate_corner_dist_px * 1.3f,
                spacing * 0.33f);
        }
        return duplicate_r;
    };

    auto completionLocalError = [&](
        int i,
        int j,
        const cv::Point2f& uv
    ) {
        float best_error = std::numeric_limits<float>::max();
        int count = 0;
        auto addExpected = [&](const cv::Point2f& expected) {
            best_error = std::min(best_error, distf(uv, expected));
            ++count;
        };

        const ActiveCorner* im1 = findActive(i - 1, j);
        const ActiveCorner* ip1 = findActive(i + 1, j);
        const ActiveCorner* jm1 = findActive(i, j - 1);
        const ActiveCorner* jp1 = findActive(i, j + 1);
        const ActiveCorner* im2 = findActive(i - 2, j);
        const ActiveCorner* ip2 = findActive(i + 2, j);
        const ActiveCorner* jm2 = findActive(i, j - 2);
        const ActiveCorner* jp2 = findActive(i, j + 2);

        if (im1 && ip1) addExpected(0.5f * (im1->uv + ip1->uv));
        if (jm1 && jp1) addExpected(0.5f * (jm1->uv + jp1->uv));
        if (im1 && im2) addExpected(im1->uv + (im1->uv - im2->uv));
        if (ip1 && ip2) addExpected(ip1->uv + (ip1->uv - ip2->uv));
        if (jm1 && jm2) addExpected(jm1->uv + (jm1->uv - jm2->uv));
        if (jp1 && jp2) addExpected(jp1->uv + (jp1->uv - jp2->uv));

        const int dirs[2] = {-1, 1};
        for (int di : dirs) {
            for (int dj : dirs) {
                const ActiveCorner* edge_i = findActive(i - di, j);
                const ActiveCorner* edge_j = findActive(i, j - dj);
                const ActiveCorner* diag = findActive(i - di, j - dj);
                if (!edge_i || !edge_j || !diag) continue;
                addExpected(edge_i->uv + edge_j->uv - diag->uv);
            }
        }

        return std::pair<int, float>{count, best_error};
    };

    struct LocalCompletionMatch {
        int i = 0;
        int j = 0;
        cv::Point2f uv;
        int support = 0;
        float error = 0.0f;
        bool guided = false;
    };

    std::vector<LocalCompletionMatch> matches;
    matches.reserve(proposals.size());
    int guided_match_count = 0;
    int measured_match_count = 0;
    int edge_pair_match_count = 0;

    for (const auto& p : proposals) {
        int best_idx = -1;
        float best_d = std::min(
            search_r,
            spacing *
                (p.support >= 3 ? 0.58f :
                 p.support >= 2 ? 0.48f : 0.30f));
        for (int idx = 0; idx < static_cast<int>(candidates.size()); ++idx) {
            if (used[idx]) continue;
            const float d = distf(candidates[idx], p.expected);
            if (d < best_d) {
                best_d = d;
                best_idx = idx;
            }
        }
        if (best_idx < 0) continue;

        const cv::Point2f uv = candidates[best_idx];
        if (nearActiveCorner(uv, completionDuplicateRadius(p.support)))
            continue;

        const int existing_idx =
            findPersistentCornerByGrid(persistent_corners_, p.i, p.j);
        if (existing_idx >= 0 &&
            persistent_corners_[existing_idx].missed_frames == 0) {
            continue;
        }

        used[best_idx] = 1;
        const bool guided_match =
            best_idx < static_cast<int>(guided_candidate_flags.size()) &&
            guided_candidate_flags[static_cast<size_t>(best_idx)];
        if (guided_match) {
            ++guided_match_count;
        } else {
            ++measured_match_count;
        }
        matches.push_back({p.i, p.j, uv, p.support, best_d, guided_match});
    }

    auto addMeasuredEdgePair = [&](
        const ActiveCorner& a,
        const ActiveCorner& b,
        const ActiveCorner& inner_a,
        const ActiveCorner& inner_b,
        int target_ai,
        int target_aj,
        int target_bi,
        int target_bj
    ) {
        if (hasActiveGrid(target_ai, target_aj) ||
            hasActiveGrid(target_bi, target_bj)) {
            return;
        }
        for (const auto& m : matches) {
            if ((m.i == target_ai && m.j == target_aj) ||
                (m.i == target_bi && m.j == target_bj)) {
                return;
            }
        }

        const int existing_a =
            findPersistentCornerByGrid(
                persistent_corners_, target_ai, target_aj);
        const int existing_b =
            findPersistentCornerByGrid(
                persistent_corners_, target_bi, target_bj);
        if ((existing_a >= 0 &&
             persistent_corners_[existing_a].missed_frames == 0) ||
            (existing_b >= 0 &&
             persistent_corners_[existing_b].missed_frames == 0)) {
            return;
        }

        const cv::Point2f edge = b.uv - a.uv;
        const cv::Point2f inward_a = inner_a.uv - a.uv;
        const cv::Point2f inward_b = inner_b.uv - b.uv;
        const float edge_d = distf(a.uv, b.uv);
        const float inward_a_d = distf(a.uv, inner_a.uv);
        const float inward_b_d = distf(b.uv, inner_b.uv);
        if (edge_d < spacing * 0.35f || edge_d > spacing * 2.20f)
            return;
        if (inward_a_d < spacing * 0.35f ||
            inward_b_d < spacing * 0.35f ||
            inward_a_d > spacing * 2.35f ||
            inward_b_d > spacing * 2.35f) {
            return;
        }

        const float area =
            std::abs(edge.x * inward_a.y - edge.y * inward_a.x);
        if (area < edge_d * inward_a_d * 0.15f) return;
        if (distf(inward_a, inward_b) > spacing * 0.50f) return;

        const cv::Point2f expected_a = a.uv - inward_a;
        const cv::Point2f expected_b = b.uv - inward_b;
        const auto [idx_a, err_a] =
            bestCandidateNear(
                expected_a,
                std::max(4.0f, spacing * 0.20f),
                &used);
        if (idx_a < 0) return;
        std::vector<char> used_for_b = used;
        used_for_b[static_cast<size_t>(idx_a)] = 1;
        const auto [idx_b, err_b] =
            bestCandidateNear(
                expected_b,
                std::max(4.0f, spacing * 0.20f),
                &used_for_b);
        if (idx_b < 0) return;

        const cv::Point2f uv_a = candidates[idx_a];
        const cv::Point2f uv_b = candidates[idx_b];
        if (nearActiveCorner(uv_a, spacing * 0.24f) ||
            nearActiveCorner(uv_b, spacing * 0.24f)) {
            return;
        }

        const float candidate_edge_d = distf(uv_a, uv_b);
        const float edge_ratio = candidate_edge_d / std::max(edge_d, 1.0f);
        if (edge_ratio < 0.70f || edge_ratio > 1.35f) return;
        if (distf(uv_b - uv_a, edge) > spacing * 0.38f) return;

        used[idx_a] = 1;
        used[idx_b] = 1;
        matches.push_back(
            {target_ai, target_aj, uv_a, 4, err_a, false});
        matches.push_back(
            {target_bi, target_bj, uv_b, 4, err_b, false});
        measured_match_count += 2;
        edge_pair_match_count += 2;
    };

    for (const auto& a : active) {
        const int inward_dirs[2] = {-1, 1};
        const ActiveCorner* b_i = findActive(a.i + 1, a.j);
        if (b_i) {
            for (int inward_j : inward_dirs) {
                const ActiveCorner* inner_a =
                    findActive(a.i, a.j + inward_j);
                const ActiveCorner* inner_b =
                    findActive(a.i + 1, a.j + inward_j);
                if (!inner_a || !inner_b) continue;
                addMeasuredEdgePair(
                    a,
                    *b_i,
                    *inner_a,
                    *inner_b,
                    a.i,
                    a.j - inward_j,
                    a.i + 1,
                    a.j - inward_j);
            }
        }

        const ActiveCorner* b_j = findActive(a.i, a.j + 1);
        if (b_j) {
            for (int inward_i : inward_dirs) {
                const ActiveCorner* inner_a =
                    findActive(a.i + inward_i, a.j);
                const ActiveCorner* inner_b =
                    findActive(a.i + inward_i, a.j + 1);
                if (!inner_a || !inner_b) continue;
                addMeasuredEdgePair(
                    a,
                    *b_j,
                    *inner_a,
                    *inner_b,
                    a.i - inward_i,
                    a.j,
                    a.i - inward_i,
                    a.j + 1);
            }
        }
    }

    if (edge_pair_match_count > 0) {
        addTimingMs(
            "tracking_local_completion_edge_pair_match_count",
            static_cast<double>(edge_pair_match_count));
    }

    for (auto& pending : pending_completion_corners_) {
        pending.missed += 1;
    }

    auto findPending = [&](const LocalCompletionMatch& m) -> int {
        const float match_r = spacing * 0.80f;
        for (int idx = 0;
             idx < static_cast<int>(pending_completion_corners_.size());
             ++idx) {
            const auto& p = pending_completion_corners_[idx];
            if (p.i != m.i || p.j != m.j) continue;
            if (distf(p.uv, m.uv) > match_r) continue;
            return idx;
        }
        return -1;
    };

    for (const auto& m : matches) {
        const int pending_idx = findPending(m);
        if (pending_idx >= 0) {
            auto& p = pending_completion_corners_[pending_idx];
            p.uv = 0.45f * p.uv + 0.55f * m.uv;
            p.hits += 1;
            if (!m.guided) ++p.measured_hits;
            p.missed = 0;
            p.support = std::max(p.support, m.support);
            p.error = std::min(p.error, m.error);
        } else {
            PendingCompletionCorner p;
            p.i = m.i;
            p.j = m.j;
            p.uv = m.uv;
            p.hits = 1;
            p.measured_hits = m.guided ? 0 : 1;
            p.missed = 0;
            p.support = m.support;
            p.error = m.error;
            pending_completion_corners_.push_back(p);
        }
    }

    pending_completion_corners_.erase(
        std::remove_if(
            pending_completion_corners_.begin(),
            pending_completion_corners_.end(),
            [](const PendingCompletionCorner& p) {
                return p.missed > 2;
            }),
        pending_completion_corners_.end());

    addTimingMs(
        "tracking_local_completion_match_count",
        static_cast<double>(matches.size()));
    addTimingMs(
        "tracking_local_completion_guided_match_count",
        static_cast<double>(guided_match_count));
    addTimingMs(
        "tracking_local_completion_measured_match_count",
        static_cast<double>(measured_match_count));
    addTimingMs(
        "tracking_local_completion_pending_count",
        static_cast<double>(pending_completion_corners_.size()));

    std::sort(
        pending_completion_corners_.begin(),
        pending_completion_corners_.end(),
        [](const PendingCompletionCorner& a,
           const PendingCompletionCorner& b) {
            if (a.hits != b.hits) return a.hits > b.hits;
            if (a.support != b.support) return a.support > b.support;
            return a.error < b.error;
        });

    int added = 0;
    int deferred = 0;
    int geometry_rejected = 0;
    int fast_accepted = 0;
    constexpr int kMaxAddedPerFrame = 6;

    for (auto it = pending_completion_corners_.begin();
         it != pending_completion_corners_.end();) {
        if (added >= kMaxAddedPerFrame) break;
        if (it->missed != 0) {
            ++it;
            continue;
        }

        const bool has_measured_hit = it->measured_hits > 0;
        const bool precise_measured_support3 =
            has_measured_hit &&
            it->support == 3 &&
            it->hits >= 1 &&
            it->error <= spacing * 0.24f;
        const bool very_precise_multi_source =
            has_measured_hit &&
            it->support >= 4 &&
            it->hits >= 1 &&
            it->error <= spacing * 0.22f;
        const bool strong_measured_single_frame =
            precise_measured_support3 ||
            very_precise_multi_source;
        const bool confirmed =
            strong_measured_single_frame ||
            (it->support >= 3 &&
             it->hits >= 2 &&
             it->error <= spacing * 0.50f) ||
            (it->support >= 2 &&
             it->hits >= 2 &&
             it->error <= spacing * 0.42f) ||
            (it->support >= 1 &&
             it->hits >= 3 &&
             it->error <= spacing * 0.35f);
        if (!confirmed) {
            ++deferred;
            ++it;
            continue;
        }

        const auto [local_error_count, local_error] =
            completionLocalError(it->i, it->j, it->uv);
        const bool geometry_ok =
            local_error_count == 0 ||
            local_error <=
                spacing * (it->support >= 3 ? 0.55f : 0.45f);
        if (!geometry_ok) {
            ++geometry_rejected;
            it = pending_completion_corners_.erase(it);
            continue;
        }

        if (hasActiveGrid(it->i, it->j) ||
            nearActiveCorner(
                it->uv, completionDuplicateRadius(it->support))) {
            it = pending_completion_corners_.erase(it);
            continue;
        }

        const int existing_idx =
            findPersistentCornerByGrid(
                persistent_corners_, it->i, it->j);
        if (existing_idx >= 0 &&
            persistent_corners_[existing_idx].missed_frames == 0) {
            it = pending_completion_corners_.erase(it);
            continue;
        }

        GridCorner gc;
        gc.i = it->i;
        gc.j = it->j;
        gc.uv = it->uv;
        gc.visibility_score = 0.75f;
        gc.synthetic = false;
        gc.observed_frames = 1;
        gc.predicted = false;

        if (existing_idx >= 0) {
            auto& pc = persistent_corners_[existing_idx];
            pc.corner = gc;
            pc.missed_frames = 0;
            pc.tracked = true;
            pc.observed_frames += 1;
            pc.predicted_frames = 0;
            pc.low_visibility_frames = 0;
            pc.visibility_score = 0.75f;
            pc.smoothed_visibility_score =
                std::max(pc.smoothed_visibility_score, 0.60f);
        } else {
            PersistentTrackedCorner pc;
            pc.corner = gc;
            pc.missed_frames = 0;
            pc.tracked = true;
            pc.observed_frames = 1;
            pc.predicted_frames = 0;
            pc.low_visibility_frames = 0;
            pc.visibility_score = 0.75f;
            pc.smoothed_visibility_score = 0.75f;
            persistent_corners_.push_back(pc);
        }

        if (strong_measured_single_frame && it->hits == 1) {
            ++fast_accepted;
        }
        ++added;
        it = pending_completion_corners_.erase(it);
    }

    if (deferred > 0) {
        addTimingMs(
            "tracking_local_completion_deferred_count",
            static_cast<double>(deferred));
    }
    if (geometry_rejected > 0) {
        addTimingMs(
            "tracking_local_completion_geometry_reject_count",
            static_cast<double>(geometry_rejected));
    }
    if (fast_accepted > 0) {
        addTimingMs(
            "tracking_local_completion_fast_accept_count",
            static_cast<double>(fast_accepted));
    }

    if (added > 0 || corrected > 0 || saddle_snap_count > 0) {
        auto completed = buildDetectionFromPersistent(tracking, false);
        if (completed.valid()) {
            last_detection_ = std::move(completed);
            tracking_active_ = true;
        }
    }

    return added;
}




// ============================================================
// buildDetectionFromPersistent
// Emits directly tracked corners plus short-lived, locally supported memory
// hypotheses for long-observed corners. This keeps partial checkerboard
// blocks alive through brief LK/recovery dropouts without accepting isolated
// new points as truth.
// ============================================================

CheckerboardDetection CheckerboardDetector::buildDetectionFromPersistent(
    bool tracking,
    bool stable
) const {
    std::vector<GridCorner> visible_corners;
    visible_corners.reserve(persistent_corners_.size());

    int persistent_missed_count = 0;
    int persistent_predicted_count = 0;
    for (const auto& pc : persistent_corners_) {
        if (pc.missed_frames > 0) ++persistent_missed_count;
        if (pc.corner.predicted || pc.predicted_frames > 0)
            ++persistent_predicted_count;
    }
    last_timings_ms_["tracking_persistent_count"] =
        static_cast<double>(persistent_corners_.size());
    last_timings_ms_["tracking_persistent_missed_count"] =
        static_cast<double>(persistent_missed_count);
    last_timings_ms_["tracking_persistent_predicted_count"] =
        static_cast<double>(persistent_predicted_count);

    auto trackedNeighbourCount = [&](const PersistentTrackedCorner& pc) {
        int count = 0;
        for (const auto& nb : persistent_corners_) {
            if (nb.missed_frames != 0) continue;
            const int di = std::abs(nb.corner.i - pc.corner.i);
            const int dj = std::abs(nb.corner.j - pc.corner.j);
            if (di + dj == 1) {
                ++count;
            }
        }
        return count;
    };

    for (const auto& pc : persistent_corners_) {
        const bool measured_this_frame = pc.missed_frames == 0;
        const int tracked_neighbours = trackedNeighbourCount(pc);

        if (tracking &&
            measured_this_frame &&
            !pc.corner.predicted &&
            pc.smoothed_visibility_score < 0.05f) {
            continue;
        }

        const bool established_corner = pc.observed_frames >= 3;
        const bool long_established_corner = pc.observed_frames >= 8;
        const bool very_long_established_corner = pc.observed_frames >= 20;
        if (measured_this_frame && pc.corner.predicted) {
            const bool reliable_one_frame_bridge =
                pc.predicted_frames <= 1 &&
                pc.smoothed_visibility_score >= 0.35f &&
                tracked_neighbours >= 2;
            const bool stable_memory_bridge =
                tracking &&
                stable &&
                very_long_established_corner &&
                pc.predicted_frames <= std::max(config_.max_missed_frames, 30) &&
                pc.smoothed_visibility_score >= 0.05f &&
                tracked_neighbours >= 1;
            if (!reliable_one_frame_bridge && !stable_memory_bridge) {
                continue;
            }
        }

        const int max_hold_frames = [&]() {
            if (!tracking) {
                return stable || established_corner
                    ? config_.max_missed_frames
                    : 1;
            }
            if (very_long_established_corner && stable) {
                return pc.smoothed_visibility_score < 0.05f
                    ? 0
                    : std::max(config_.max_missed_frames, 30);
            }
            if (long_established_corner) {
                return pc.smoothed_visibility_score <
                    (stable ? 0.35f : 0.18f) ? 0 : 1;
            }
            return established_corner ? 1 : 0;
        }();
        const float min_hold_visibility = [&]() {
            if (!tracking) {
                return long_established_corner
                    ? 0.0f
                    : (stable ? 0.05f
                              : (established_corner ? 0.10f : 0.55f));
            }
            if (very_long_established_corner && stable) return 0.05f;
            if (long_established_corner) return stable ? 0.35f : 0.18f;
            return stable ? 0.18f : 0.30f;
        }();
        const int min_hold_neighbours =
            (tracking && !(very_long_established_corner && stable)) ? 2 : 1;
        const bool hold_visible_frame =
            pc.missed_frames > 0 &&
            pc.missed_frames <= max_hold_frames &&
            pc.smoothed_visibility_score >= min_hold_visibility &&
            tracked_neighbours >= min_hold_neighbours;

        if (!measured_this_frame && !hold_visible_frame) continue;

        GridCorner gc = pc.corner;
        gc.visibility_score = pc.smoothed_visibility_score;
        gc.synthetic = false;
        gc.observed_frames = pc.observed_frames;
        gc.predicted = pc.corner.predicted || !measured_this_frame;
        visible_corners.push_back(gc);
    }

    if (tracking &&
        static_cast<int>(visible_corners.size()) >
            config_.min_tracking_corners) {
        auto compressedOutputFilter = [&]() {
            std::vector<float> good_edges;
            good_edges.reserve(visible_corners.size() * 2);

            auto strongVisible = [](const GridCorner& c) {
                return !c.synthetic &&
                       !c.predicted &&
                       c.visibility_score >= 0.30f;
            };

            for (size_t a = 0; a < visible_corners.size(); ++a) {
                const auto& ca = visible_corners[a];
                if (!strongVisible(ca)) continue;

                for (size_t b = a + 1; b < visible_corners.size(); ++b) {
                    const auto& cb = visible_corners[b];
                    if (!strongVisible(cb)) continue;

                    const int di = std::abs(ca.i - cb.i);
                    const int dj = std::abs(ca.j - cb.j);
                    if (di + dj != 1) continue;

                    const float d = distf(ca.uv, cb.uv);
                    if (d >= 3.0f) {
                        good_edges.push_back(d);
                    }
                }
            }

            if (good_edges.size() < 6) {
                for (size_t a = 0; a < visible_corners.size(); ++a) {
                    const auto& ca = visible_corners[a];
                    if (ca.synthetic || ca.predicted ||
                        ca.visibility_score < 0.18f) {
                        continue;
                    }

                    for (size_t b = a + 1; b < visible_corners.size(); ++b) {
                        const auto& cb = visible_corners[b];
                        if (cb.synthetic || cb.predicted ||
                            cb.visibility_score < 0.18f) {
                            continue;
                        }

                        const int di = std::abs(ca.i - cb.i);
                        const int dj = std::abs(ca.j - cb.j);
                        if (di + dj != 1) continue;

                        const float d = distf(ca.uv, cb.uv);
                        if (d >= 3.0f) {
                            good_edges.push_back(d);
                        }
                    }
                }
            }

            if (good_edges.size() < 6) return;

            std::nth_element(
                good_edges.begin(),
                good_edges.begin() +
                    static_cast<std::ptrdiff_t>(good_edges.size() / 2),
                good_edges.end());
            const float spacing = good_edges[good_edges.size() / 2];
            if (spacing < 6.0f) return;

            const float compressed_d =
                std::max(config_.duplicate_corner_dist_px * 1.6f,
                         spacing * 0.65f);
            const float duplicate_d =
                std::max(config_.duplicate_corner_dist_px * 1.6f,
                         spacing * 0.42f);

            std::vector<char> remove(visible_corners.size(), 0);

            auto weakOrPredicted = [](const GridCorner& c) {
                return c.predicted || c.visibility_score < 0.28f;
            };

            auto removalScore = [](const GridCorner& c) {
                float score = 0.0f;
                if (c.predicted) score += 4.0f;
                if (c.visibility_score < 0.05f) {
                    score += 4.0f;
                } else if (c.visibility_score < 0.12f) {
                    score += 2.5f;
                } else if (c.visibility_score < 0.22f) {
                    score += 1.0f;
                } else if (c.visibility_score < 0.28f) {
                    score += 0.5f;
                }
                if (c.observed_frames < 3) score += 1.0f;
                return score;
            };

            for (size_t a = 0; a < visible_corners.size(); ++a) {
                if (remove[a]) continue;
                const auto& ca = visible_corners[a];
                if (ca.synthetic) continue;

                for (size_t b = a + 1; b < visible_corners.size(); ++b) {
                    if (remove[b]) continue;
                    const auto& cb = visible_corners[b];
                    if (cb.synthetic) continue;

                    const int di = std::abs(ca.i - cb.i);
                    const int dj = std::abs(ca.j - cb.j);
                    if (di == 0 && dj == 0) continue;

                    const float d = distf(ca.uv, cb.uv);
                    const bool duplicate_collision = d < duplicate_d;
                    const bool compressed_adjacent =
                        (di + dj == 1) && d < compressed_d;
                    if (!duplicate_collision && !compressed_adjacent)
                        continue;

                    if (!duplicate_collision &&
                        !weakOrPredicted(ca) && !weakOrPredicted(cb)) {
                        continue;
                    }

                    const bool a_strong = strongVisible(ca);
                    const bool b_strong = strongVisible(cb);
                    if (a_strong && !b_strong) {
                        remove[b] = 1;
                    } else if (b_strong && !a_strong) {
                        remove[a] = 1;
                        break;
                    } else if (removalScore(ca) >= removalScore(cb)) {
                        remove[a] = 1;
                        break;
                    } else {
                        remove[b] = 1;
                    }
                }
            }

            int survivors = 0;
            for (size_t k = 0; k < visible_corners.size(); ++k) {
                if (!remove[k]) ++survivors;
            }
            if (survivors < config_.min_tracking_corners) return;

            std::vector<GridCorner> filtered;
            filtered.reserve(static_cast<size_t>(survivors));
            for (size_t k = 0; k < visible_corners.size(); ++k) {
                if (!remove[k]) filtered.push_back(visible_corners[k]);
            }
            visible_corners = std::move(filtered);
        };

        compressedOutputFilter();
    }

    auto hasVisibleGrid = [&](int i, int j) -> bool {
        for (const auto& c : visible_corners) {
            if (c.i == i && c.j == j) return true;
        }
        return false;
    };

    auto addIfMissing = [&](int i, int j, const cv::Point2f& uv) {
        if (hasVisibleGrid(i, j)) return;
        for (const auto& c : visible_corners) {
            if (distf(c.uv, uv) < config_.duplicate_corner_dist_px) return;
        }
        GridCorner gc;
        gc.i = i;
        gc.j = j;
        gc.uv = uv;
        gc.visibility_score = 0.0f;
        gc.synthetic = true;
        visible_corners.push_back(gc);
    };

    // Fill single-cell holes only for fresh non-tracking detections. During
    // tracking, synthetic gap-fill corners can leak into the visible output
    // and look like floating or flickering real corners at the cylinder rim.
    if (!tracking) {
        const std::vector<GridCorner> before_gap_fill = visible_corners;
        for (const auto& left : before_gap_fill) {
            if (left.predicted || left.visibility_score < 0.18f) continue;
            const int mi = left.i + 1;
            const int mj = left.j;
            if (hasVisibleGrid(mi, mj)) continue;
            for (const auto& right : before_gap_fill) {
                if (right.predicted || right.visibility_score < 0.18f)
                    continue;
                if (right.i == left.i + 2 && right.j == left.j) {
                    addIfMissing(mi, mj, 0.5f * (left.uv + right.uv));
                    break;
                }
            }
        }
        for (const auto& top : before_gap_fill) {
            if (top.predicted || top.visibility_score < 0.18f) continue;
            const int mi = top.i;
            const int mj = top.j + 1;
            if (hasVisibleGrid(mi, mj)) continue;
            for (const auto& bottom : before_gap_fill) {
                if (bottom.predicted || bottom.visibility_score < 0.18f)
                    continue;
                if (bottom.i == top.i && bottom.j == top.j + 2) {
                    addIfMissing(mi, mj, 0.5f * (top.uv + bottom.uv));
                    break;
                }
            }
        }
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

    if (tracking) {
        const float spacing = std::max(1.0f, estimateMedianSpacing(*rebuilt));
        const float duplicate_d =
            std::max(2.0f, config_.duplicate_corner_dist_px);

        auto hasOutputGrid = [&](int i, int j) -> bool {
            for (const auto& c : rebuilt->corners) {
                if (c.i == i && c.j == j) return true;
            }
            return false;
        };

        auto outputDuplicate = [&](const cv::Point2f& uv) -> bool {
            for (const auto& c : rebuilt->corners) {
                if (distf(c.uv, uv) < duplicate_d) return true;
            }
            return false;
        };

        auto inputNeighbourCount = [&](const GridCorner& candidate) -> int {
            int count = 0;
            for (const auto& c : visible_corners) {
                if (c.synthetic) continue;
                if (c.i == candidate.i && c.j == candidate.j) continue;
                const int di = std::abs(c.i - candidate.i);
                const int dj = std::abs(c.j - candidate.j);
                if (di + dj == 1) ++count;
            }
            return count;
        };

        auto outputSupportOk = [&](const GridCorner& candidate) -> bool {
            for (const auto& c : rebuilt->corners) {
                const int di = std::abs(c.i - candidate.i);
                const int dj = std::abs(c.j - candidate.j);
                if (di + dj != 1) continue;

                const float d = distf(c.uv, candidate.uv);
                if (d >= spacing * 0.25f &&
                    d <= spacing * 2.10f) {
                    return true;
                }
            }
            return false;
        };

        auto predictedFramesForGrid = [&](const GridCorner& candidate) -> int {
            for (const auto& pc : persistent_corners_) {
                if (pc.corner.i == candidate.i &&
                    pc.corner.j == candidate.j) {
                    return pc.predicted_frames;
                }
            }
            return 0;
        };

        // The cell compaction stage is intentionally strict because it protects
        // pose/decode quality. For tracking, however, the green landmark output
        // should be temporally stable: an established rim corner may be valid
        // even when it does not close a full cell this frame. Re-attach those
        // locally supported corners after the cell graph has been built.
        for (int pass = 0; pass < 4; ++pass) {
            bool changed = false;
            for (const auto& c : visible_corners) {
                if (c.synthetic) continue;
                if (hasOutputGrid(c.i, c.j)) continue;
                if (outputDuplicate(c.uv)) continue;
                if (!stable &&
                    (c.predicted || c.visibility_score < 0.18f)) {
                    continue;
                }
                const int input_neighbours = inputNeighbourCount(c);
                const bool long_established = c.observed_frames >= 8;
                const bool locally_confirmed =
                    c.observed_frames >= 3 && input_neighbours >= 2;
                const bool strong_one_frame_observation =
                    !c.predicted &&
                    c.observed_frames >= 1 &&
                    c.visibility_score >= 0.55f &&
                    input_neighbours >= 2;
                const bool confirmed_new_observation =
                    !c.predicted &&
                    c.observed_frames >= 2 &&
                    c.visibility_score >=
                        (input_neighbours >= 2 ? 0.30f : 0.45f) &&
                    input_neighbours >= 1;
                const bool strong_new_observation =
                    strong_one_frame_observation ||
                    confirmed_new_observation;
                if (!long_established &&
                    !locally_confirmed &&
                    !strong_new_observation) {
                    continue;
                }
                if (c.predicted &&
                    c.visibility_score < 0.05f &&
                    predictedFramesForGrid(c) > 2) {
                    continue;
                }
                const float min_output_visibility =
                    long_established ? (stable ? 0.05f : 0.18f)
                                     : (stable ? 0.05f : 0.10f);
                if (c.visibility_score < min_output_visibility)
                    continue;
                if (input_neighbours < 1) continue;
                if (!outputSupportOk(c)) continue;

                rebuilt->corners.push_back(c);
                changed = true;
            }
            if (!changed) break;
        }

        auto finalCompressedOutputFilter = [&]() {
            auto& corners = rebuilt->corners;
            if (static_cast<int>(corners.size()) <=
                config_.min_tracking_corners) {
                return;
            }

            auto strongVisible = [](const GridCorner& c) {
                return !c.synthetic &&
                       !c.predicted &&
                       c.visibility_score >= 0.30f;
            };
            auto weakOrPredicted = [](const GridCorner& c) {
                return c.predicted || c.visibility_score < 0.28f;
            };
            auto removalScore = [](const GridCorner& c) {
                float score = 0.0f;
                if (c.predicted) score += 4.0f;
                if (c.visibility_score < 0.05f) {
                    score += 4.0f;
                } else if (c.visibility_score < 0.12f) {
                    score += 2.5f;
                } else if (c.visibility_score < 0.22f) {
                    score += 1.0f;
                } else if (c.visibility_score < 0.28f) {
                    score += 0.5f;
                }
                if (c.observed_frames < 3) score += 1.0f;
                return score;
            };

            std::vector<float> good_edges;
            good_edges.reserve(corners.size() * 2);
            for (size_t a = 0; a < corners.size(); ++a) {
                if (!strongVisible(corners[a])) continue;
                for (size_t b = a + 1; b < corners.size(); ++b) {
                    if (!strongVisible(corners[b])) continue;
                    const int di = std::abs(corners[a].i - corners[b].i);
                    const int dj = std::abs(corners[a].j - corners[b].j);
                    if (di + dj != 1) continue;
                    const float d = distf(corners[a].uv, corners[b].uv);
                    if (d >= 3.0f) good_edges.push_back(d);
                }
            }
            if (good_edges.size() < 6) return;

            std::nth_element(
                good_edges.begin(),
                good_edges.begin() +
                    static_cast<std::ptrdiff_t>(good_edges.size() / 2),
                good_edges.end());
            const float spacing = good_edges[good_edges.size() / 2];
            if (spacing < 6.0f) return;

            const float compressed_d =
                std::max(config_.duplicate_corner_dist_px * 1.6f,
                         spacing * 0.65f);
            const float duplicate_d =
                std::max(config_.duplicate_corner_dist_px * 1.6f,
                         spacing * 0.42f);

            std::vector<char> remove(corners.size(), 0);
            for (size_t a = 0; a < corners.size(); ++a) {
                if (remove[a] || corners[a].synthetic) continue;
                for (size_t b = a + 1; b < corners.size(); ++b) {
                    if (remove[b] || corners[b].synthetic) continue;
                    const int di = std::abs(corners[a].i - corners[b].i);
                    const int dj = std::abs(corners[a].j - corners[b].j);
                    if (di == 0 && dj == 0) continue;

                    const float d = distf(corners[a].uv, corners[b].uv);
                    const bool duplicate_collision = d < duplicate_d;
                    const bool compressed_adjacent =
                        (di + dj == 1) && d < compressed_d;
                    if (!duplicate_collision && !compressed_adjacent)
                        continue;
                    if (!duplicate_collision &&
                        !weakOrPredicted(corners[a]) &&
                        !weakOrPredicted(corners[b])) {
                        continue;
                    }

                    const bool a_strong = strongVisible(corners[a]);
                    const bool b_strong = strongVisible(corners[b]);
                    if (a_strong && !b_strong) {
                        remove[b] = 1;
                    } else if (b_strong && !a_strong) {
                        remove[a] = 1;
                        break;
                    } else if (removalScore(corners[a]) >=
                               removalScore(corners[b])) {
                        remove[a] = 1;
                        break;
                    } else {
                        remove[b] = 1;
                    }
                }
            }

            auto cellUsesGrid = [](const GridCell& cell,
                                   const GridCorner& c) {
                return (c.i == cell.i && c.j == cell.j) ||
                       (c.i == cell.i + 1 && c.j == cell.j) ||
                       (c.i == cell.i + 1 && c.j == cell.j + 1) ||
                       (c.i == cell.i && c.j == cell.j + 1);
            };

            const float min_cell_area = spacing * spacing * 0.20f;
            const float interior_margin =
                std::max(1.0f, spacing * 0.04f);
            const float outside_cell_margin =
                std::max(2.0f, spacing * 0.50f);
            for (int pass = 0; pass < 3; ++pass) {
                bool changed = false;

                for (size_t k = 0; k < corners.size(); ++k) {
                    if (remove[k] || corners[k].synthetic) continue;

                    for (const auto& cell : rebuilt->cells) {
                        if (cellUsesGrid(cell, corners[k])) continue;
                        if (quadArea(cell.corner_uv) < min_cell_area)
                            continue;

                        bool cell_already_removed = false;
                        for (int idx : cell.corner_indices) {
                            if (idx < 0 ||
                                idx >= static_cast<int>(remove.size()) ||
                                remove[static_cast<size_t>(idx)]) {
                                cell_already_removed = true;
                                break;
                            }
                        }
                        if (cell_already_removed) continue;

                        const float inside_d =
                            pointInsideQuadDistance(cell.corner_uv,
                                                    corners[k].uv);
                        if (inside_d <= interior_margin) continue;

                        int weakest_cell_idx = -1;
                        float weakest_cell_score =
                            std::numeric_limits<float>::lowest();
                        for (int idx : cell.corner_indices) {
                            if (idx < 0 ||
                                idx >= static_cast<int>(corners.size())) {
                                continue;
                            }
                            const float score =
                                removalScore(corners[
                                    static_cast<size_t>(idx)]);
                            if (score > weakest_cell_score) {
                                weakest_cell_score = score;
                                weakest_cell_idx = idx;
                            }
                        }
                        if (weakest_cell_idx < 0) continue;

                        const float candidate_score =
                            removalScore(corners[k]);
                        if (candidate_score >=
                            weakest_cell_score - 0.25f) {
                            remove[k] = 1;
                        } else {
                            remove[static_cast<size_t>(
                                weakest_cell_idx)] = 1;
                        }
                        changed = true;
                        break;
                    }
                }

                if (!changed) break;
            }

            auto outputGridSupport = [&](size_t idx) {
                int support = 0;
                const auto& c = corners[idx];
                for (size_t k = 0; k < corners.size(); ++k) {
                    if (k == idx || remove[k]) continue;
                    const int di = std::abs(corners[k].i - c.i);
                    const int dj = std::abs(corners[k].j - c.j);
                    if (di + dj == 1) ++support;
                }
                return support;
            };

            auto findOutputCorner = [&](int i, int j) -> int {
                for (int idx = 0; idx < static_cast<int>(corners.size());
                     ++idx) {
                    if (remove[static_cast<size_t>(idx)]) continue;
                    if (corners[static_cast<size_t>(idx)].synthetic)
                        continue;
                    if (corners[static_cast<size_t>(idx)].i == i &&
                        corners[static_cast<size_t>(idx)].j == j) {
                        return idx;
                    }
                }
                return -1;
            };

            auto cornerLocalError = [&](size_t idx) {
                const auto& c = corners[idx];
                float best_error = std::numeric_limits<float>::max();
                int count = 0;
                auto addExpected = [&](const cv::Point2f& expected) {
                    best_error = std::min(best_error,
                                          distf(c.uv, expected));
                    ++count;
                };
                auto pointAt = [&](int i, int j)
                    -> const cv::Point2f* {
                    const int found = findOutputCorner(i, j);
                    if (found < 0) return nullptr;
                    return &corners[static_cast<size_t>(found)].uv;
                };

                const cv::Point2f* im1 = pointAt(c.i - 1, c.j);
                const cv::Point2f* ip1 = pointAt(c.i + 1, c.j);
                const cv::Point2f* jm1 = pointAt(c.i, c.j - 1);
                const cv::Point2f* jp1 = pointAt(c.i, c.j + 1);
                const cv::Point2f* im2 = pointAt(c.i - 2, c.j);
                const cv::Point2f* ip2 = pointAt(c.i + 2, c.j);
                const cv::Point2f* jm2 = pointAt(c.i, c.j - 2);
                const cv::Point2f* jp2 = pointAt(c.i, c.j + 2);

                if (im1 && ip1) addExpected(0.5f * (*im1 + *ip1));
                if (jm1 && jp1) addExpected(0.5f * (*jm1 + *jp1));
                if (im1 && im2) addExpected(*im1 + (*im1 - *im2));
                if (ip1 && ip2) addExpected(*ip1 + (*ip1 - *ip2));
                if (jm1 && jm2) addExpected(*jm1 + (*jm1 - *jm2));
                if (jp1 && jp2) addExpected(*jp1 + (*jp1 - *jp2));

                const int dirs[2] = {-1, 1};
                for (int di : dirs) {
                    for (int dj : dirs) {
                        const cv::Point2f* edge_i =
                            pointAt(c.i - di, c.j);
                        const cv::Point2f* edge_j =
                            pointAt(c.i, c.j - dj);
                        const cv::Point2f* diag =
                            pointAt(c.i - di, c.j - dj);
                        if (!edge_i || !edge_j || !diag) continue;
                        addExpected(*edge_i + *edge_j - *diag);
                    }
                }

                return std::pair<int, float>{count, best_error};
            };

            int geometry_reject_count = 0;
            for (size_t k = 0; k < corners.size(); ++k) {
                if (remove[k] || corners[k].synthetic) continue;

                const auto [local_count, local_error] =
                    cornerLocalError(k);
                if (local_count < 2) continue;

                const int support = outputGridSupport(k);
                const bool low_quality =
                    corners[k].predicted ||
                    corners[k].visibility_score < 0.65f ||
                    support < 2;
                const bool severe_quality =
                    corners[k].visibility_score < 0.85f || support < 2;

                const bool suspicious =
                    local_error > spacing * 0.58f && low_quality;
                const bool severe =
                    local_error > spacing * 0.82f && severe_quality;
                if (!suspicious && !severe) continue;

                remove[k] = 1;
                ++geometry_reject_count;
            }
            if (geometry_reject_count > 0) {
                addTimingMs(
                    "tracking_output_geometry_reject_count",
                    static_cast<double>(geometry_reject_count));
            }

            auto findPreviousOutputCorner = [&](int i, int j)
                -> const GridCorner* {
                for (const auto& c : last_detection_.corners) {
                    if (c.synthetic) continue;
                    if (c.i == i && c.j == j) return &c;
                }
                return nullptr;
            };

            std::vector<cv::Point2f> prev_motion_pts;
            std::vector<cv::Point2f> curr_motion_pts;
            prev_motion_pts.reserve(corners.size());
            curr_motion_pts.reserve(corners.size());
            for (size_t k = 0; k < corners.size(); ++k) {
                if (remove[k] || corners[k].synthetic) continue;
                if (corners[k].predicted) continue;
                if (corners[k].visibility_score < 0.35f) continue;
                const GridCorner* prev =
                    findPreviousOutputCorner(corners[k].i, corners[k].j);
                if (!prev || prev->predicted) continue;
                prev_motion_pts.push_back(prev->uv);
                curr_motion_pts.push_back(corners[k].uv);
            }

            int temporal_reject_count = 0;
            if (prev_motion_pts.size() >= 8) {
                cv::Mat inlier_mask;
                cv::Mat affine = cv::estimateAffine2D(
                    prev_motion_pts,
                    curr_motion_pts,
                    inlier_mask,
                    cv::RANSAC,
                    std::max(3.0, static_cast<double>(spacing * 0.35f)),
                    200,
                    0.99,
                    10);
                if (!affine.empty()) {
                    if (affine.depth() != CV_64F) {
                        cv::Mat affine64;
                        affine.convertTo(affine64, CV_64F);
                        affine = affine64;
                    }

                    const double* a = affine.ptr<double>(0);
                    const double* b = affine.ptr<double>(1);
                    const float temporal_threshold =
                        std::max(8.0f, spacing * 0.70f);

                    for (size_t k = 0; k < corners.size(); ++k) {
                        if (remove[k] || corners[k].synthetic) continue;
                        const GridCorner* prev =
                            findPreviousOutputCorner(corners[k].i,
                                                     corners[k].j);
                        if (!prev) continue;

                        const cv::Point2f projected(
                            static_cast<float>(
                                a[0] * prev->uv.x +
                                a[1] * prev->uv.y +
                                a[2]),
                            static_cast<float>(
                                b[0] * prev->uv.x +
                                b[1] * prev->uv.y +
                                b[2]));
                        const float residual =
                            distf(corners[k].uv, projected);
                        if (residual <= temporal_threshold) continue;

                        const int support = outputGridSupport(k);
                        const bool weak_temporal_candidate =
                            corners[k].predicted ||
                            prev->predicted ||
                            corners[k].visibility_score < 0.85f ||
                            support < 3;
                        if (!weak_temporal_candidate) continue;

                        remove[k] = 1;
                        ++temporal_reject_count;
                    }
                }
            }
            if (temporal_reject_count > 0) {
                addTimingMs(
                    "tracking_output_temporal_reject_count",
                    static_cast<double>(temporal_reject_count));
            }

            int support_reject_count = 0;
            int single_neighbour_hold_count = 0;
            for (size_t k = 0; k < corners.size(); ++k) {
                if (remove[k] || corners[k].synthetic) continue;

                bool used_by_cell = false;
                float min_cell_dist = std::numeric_limits<float>::max();
                for (const auto& cell : rebuilt->cells) {
                    if (quadArea(cell.corner_uv) < min_cell_area)
                        continue;
                    if (cellUsesGrid(cell, corners[k])) {
                        used_by_cell = true;
                        break;
                    }
                    min_cell_dist = std::min(
                        min_cell_dist,
                        pointToQuadDistance(cell.corner_uv,
                                            corners[k].uv));
                }
                if (used_by_cell) continue;

                const int support = outputGridSupport(k);
                float only_neighbour_edge = std::numeric_limits<float>::max();
                if (support == 1) {
                    for (size_t n = 0; n < corners.size(); ++n) {
                        if (n == k || remove[n] || corners[n].synthetic)
                            continue;
                        const int di =
                            std::abs(corners[n].i - corners[k].i);
                        const int dj =
                            std::abs(corners[n].j - corners[k].j);
                        if (di + dj != 1) continue;
                        only_neighbour_edge =
                            distf(corners[n].uv, corners[k].uv);
                        break;
                    }
                }

                const GridCorner* previous_output =
                    findPreviousOutputCorner(
                        corners[k].i,
                        corners[k].j);
                const bool previous_measured_output =
                    previous_output && !previous_output->predicted;
                const bool plausible_single_edge =
                    support == 1 &&
                    only_neighbour_edge >= spacing * 0.45f &&
                    only_neighbour_edge <= spacing * 1.85f;
                const bool temporal_single_neighbour_hold =
                    plausible_single_edge &&
                    previous_measured_output &&
                    !corners[k].predicted &&
                    corners[k].observed_frames >= 3 &&
                    corners[k].visibility_score >= 0.55f;
                const bool weak_single_neighbour =
                    support == 1 &&
                    !temporal_single_neighbour_hold &&
                    (corners[k].predicted ||
                     corners[k].visibility_score < 0.70f ||
                     corners[k].observed_frames < 8);
                const bool implausible_single_edge =
                    support == 1 &&
                    (only_neighbour_edge < spacing * 0.45f ||
                     only_neighbour_edge > spacing * 1.85f);
                const bool very_weak_non_cell =
                    support <= 2 &&
                    corners[k].visibility_score < 0.18f;
                const bool unsupported_outside_cell =
                    min_cell_dist > outside_cell_margin &&
                    support < 2 &&
                    !temporal_single_neighbour_hold;

                if (temporal_single_neighbour_hold) {
                    ++single_neighbour_hold_count;
                }

                if (support == 0 ||
                    weak_single_neighbour ||
                    implausible_single_edge ||
                    very_weak_non_cell ||
                    unsupported_outside_cell) {
                    remove[k] = 1;
                    ++support_reject_count;
                }
            }
            if (support_reject_count > 0) {
                addTimingMs(
                    "tracking_output_support_reject_count",
                    static_cast<double>(support_reject_count));
            }
            if (single_neighbour_hold_count > 0) {
                addTimingMs(
                    "tracking_output_single_neighbour_hold_count",
                    static_cast<double>(single_neighbour_hold_count));
            }

            int survivors = 0;
            for (char r : remove) {
                if (!r) ++survivors;
            }
            if (survivors < config_.min_tracking_corners) return;

            std::vector<int> old_to_new(corners.size(), -1);
            std::vector<GridCorner> filtered_corners;
            filtered_corners.reserve(static_cast<size_t>(survivors));
            for (size_t k = 0; k < corners.size(); ++k) {
                if (remove[k]) continue;
                old_to_new[k] = static_cast<int>(filtered_corners.size());
                filtered_corners.push_back(corners[k]);
            }

            std::vector<GridCell> filtered_cells;
            filtered_cells.reserve(rebuilt->cells.size());
            for (auto cell : rebuilt->cells) {
                bool ok = true;
                for (int k = 0; k < 4; ++k) {
                    const int old_idx = cell.corner_indices[k];
                    if (old_idx < 0 ||
                        old_idx >= static_cast<int>(old_to_new.size()) ||
                        old_to_new[old_idx] < 0) {
                        ok = false;
                        break;
                    }
                    cell.corner_indices[k] = old_to_new[old_idx];
                    cell.corner_uv[k] =
                        filtered_corners[cell.corner_indices[k]].uv;
                }
                if (!ok) continue;
                cell.center_uv =
                    0.25f * (
                        cell.corner_uv[0] +
                        cell.corner_uv[1] +
                        cell.corner_uv[2] +
                        cell.corner_uv[3]);
                filtered_cells.push_back(cell);
            }

            if (static_cast<int>(filtered_cells.size()) <
                config_.min_tracking_cells) {
                return;
            }

            CheckerboardDetection filtered_detection;
            filtered_detection.corners = filtered_corners;
            filtered_detection.cells = filtered_cells;
            filtered_detection.cols = rebuilt->cols;
            filtered_detection.rows = rebuilt->rows;
            filtered_detection.tracking = rebuilt->tracking;
            filtered_detection.stable = rebuilt->stable;
            if (!hasDecodeableCellSpan(
                    filtered_detection,
                    config_.min_tracking_decode_cell_span) &&
                hasDecodeableCellSpan(
                    *rebuilt,
                    config_.min_tracking_decode_cell_span)) {
                addTimingMs("tracking_output_decode_span_guard_count", 1.0);
                return;
            }

            rebuilt->corners = std::move(filtered_corners);
            rebuilt->cells = std::move(filtered_cells);
        };

        finalCompressedOutputFilter();

        if (!rebuilt->corners.empty()) {
            int min_i = std::numeric_limits<int>::max();
            int max_i = std::numeric_limits<int>::min();
            int min_j = std::numeric_limits<int>::max();
            int max_j = std::numeric_limits<int>::min();
            for (const auto& c : rebuilt->corners) {
                min_i = std::min(min_i, c.i);
                max_i = std::max(max_i, c.i);
                min_j = std::min(min_j, c.j);
                max_j = std::max(max_j, c.j);
            }
            rebuilt->cols = max_i - min_i + 1;
            rebuilt->rows = max_j - min_j + 1;
        }
    }

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

std::optional<CheckerboardDetection>
CheckerboardDetector::alignDetectionGridToReference(
    const CheckerboardDetection& detection,
    const CheckerboardDetection& reference
) const {
    if (!detection.valid() || !reference.valid())
        return std::nullopt;

    struct GridTransform {
        int a, b, c, d;
    };

    constexpr std::array<GridTransform, 8> transforms = {{
        { 1,  0,  0,  1},
        {-1,  0,  0,  1},
        { 1,  0,  0, -1},
        {-1,  0,  0, -1},
        { 0,  1,  1,  0},
        { 0, -1,  1,  0},
        { 0,  1, -1,  0},
        { 0, -1, -1,  0}
    }};

    const float ref_spacing = estimateMedianSpacing(reference);
    const float det_spacing = estimateMedianSpacing(detection);
    const float spacing =
        std::max(1.0f, ref_spacing > 1.0f ? ref_spacing : det_spacing);
    const float seed_match_px = std::max(8.0f, spacing * 0.75f);
    const float accept_px = std::max(6.0f, spacing * 0.45f);

    auto findReferenceByGrid = [&](int i, int j) -> const GridCorner* {
        for (const auto& rc : reference.corners) {
            if (rc.i == i && rc.j == j) return &rc;
        }
        return nullptr;
    };

    struct Candidate {
        GridTransform tr;
        int ti = 0;
        int tj = 0;
        int matches = 0;
        float mean_error = std::numeric_limits<float>::max();
    };

    std::optional<Candidate> best;

    for (const auto& dc : detection.corners) {
        for (const auto& rc : reference.corners) {
            if (distf(dc.uv, rc.uv) > seed_match_px) continue;

            for (const auto& tr : transforms) {
                const int mi = tr.a * dc.i + tr.b * dc.j;
                const int mj = tr.c * dc.i + tr.d * dc.j;
                const int ti = rc.i - mi;
                const int tj = rc.j - mj;

                int matches = 0;
                float err_sum = 0.0f;

                for (const auto& c : detection.corners) {
                    const int ai = tr.a * c.i + tr.b * c.j + ti;
                    const int aj = tr.c * c.i + tr.d * c.j + tj;
                    const GridCorner* ref = findReferenceByGrid(ai, aj);
                    if (!ref) continue;

                    const float e = distf(c.uv, ref->uv);
                    if (e > accept_px) continue;

                    ++matches;
                    err_sum += e;
                }

                if (matches < config_.min_tracking_corners)
                    continue;

                const float mean_error =
                    err_sum / static_cast<float>(matches);

                if (mean_error > accept_px * 0.75f)
                    continue;

                if (!best ||
                    matches > best->matches ||
                    (matches == best->matches &&
                     mean_error < best->mean_error)) {
                    best = Candidate{tr, ti, tj, matches, mean_error};
                }
            }
        }
    }

    if (!best)
        return std::nullopt;

    std::vector<GridCorner> aligned_corners;
    aligned_corners.reserve(detection.corners.size());

    for (auto c : detection.corners) {
        const int ai = best->tr.a * c.i + best->tr.b * c.j + best->ti;
        const int aj = best->tr.c * c.i + best->tr.d * c.j + best->tj;
        c.i = ai;
        c.j = aj;
        aligned_corners.push_back(c);
    }

    auto rebuilt = grid_builder_.buildFromCorners(
        aligned_corners,
        config_.duplicate_corner_dist_px,
        config_.min_corners,
        config_.min_cells,
        detection.tracking,
        detection.stable);

    if (!rebuilt || !rebuilt->valid())
        return std::nullopt;

    rebuilt->tracking = detection.tracking;
    rebuilt->stable   = detection.stable;
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
