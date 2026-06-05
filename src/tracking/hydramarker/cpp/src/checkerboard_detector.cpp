#include "checkerboard_detector.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>

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

static bool hasDecodeableCellSpan(
    const CheckerboardDetection& det,
    int min_span
) {
    if (min_span <= 1) return true;
    if (det.cells.empty()) return false;

    int min_i = std::numeric_limits<int>::max();
    int max_i = std::numeric_limits<int>::min();
    int min_j = std::numeric_limits<int>::max();
    int max_j = std::numeric_limits<int>::min();

    for (const auto& cell : det.cells) {
        min_i = std::min(min_i, cell.i);
        max_i = std::max(max_i, cell.i);
        min_j = std::min(min_j, cell.j);
        max_j = std::max(max_j, cell.j);
    }

    if (min_i > max_i || min_j > max_j) return false;

    const int span_i = max_i - min_i + 1;
    const int span_j = max_j - min_j + 1;
    return span_i >= min_span && span_j >= min_span;
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


void CheckerboardDetector::resetTracking() {
    last_gray_.release();
    last_detection_ = CheckerboardDetection{};
    persistent_corners_.clear();
    tracking_active_       = false;
    frame_index_           = 0;
    degraded_frames_count_ = 0;
    low_corner_frames_     = 0;
    undecodeable_tracking_frames_ = 0;
    held_output_frames_ = 0;
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

    const int max_held_output_frames =
        std::max(1, config_.max_low_corner_frames);

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
                    auto aligned = alignDetectionGridToReference(*recovered, *tracked);
                    if (!aligned) {
                        recovered = std::nullopt;
                    } else {
                        recovered = std::move(aligned);
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
                        updateTrackingState(gray, *recovered);
                        undecodeable_tracking_frames_ = 0;
                        return last_detection_;
                    }

                    // Inject new corners from recovery directly into persistent
                    // state — no lattice refit, no Grid-ID loss.
                    updateTrackingState(gray, *tracked, &(*recovered));
                    if (!last_detection_.valid()) {
                        updateTrackingState(gray, *recovered);
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

            updateTrackingState(gray, *tracked);
            if (!last_detection_.valid()) {
                auto recovered = detectRecovery(gray);
                if (recovered && recovered->valid()) {
                    auto aligned = alignDetectionGridToReference(*recovered, *tracked);
                    if (aligned) {
                        recovered = std::move(aligned);
                    } else {
                        recovered = std::nullopt;
                    }
                }
                if (recovered && recovered->valid()) {
                    recovered->tracking = false;
                    recovered->stable   = false;
                    updateTrackingState(gray, *recovered);
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
    auto recovered = detectRecovery(gray);

    if (recovered && recovered->valid()) {
        recovered->tracking = false;
        recovered->stable   = false;
        updateTrackingState(gray, *recovered);
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
    std::vector<cv::Point2f> refined_corners;
    refined_corners.reserve(refined.size());

    std::vector<cv::Point2f> quadrant_corners;
    quadrant_corners.reserve(refined.size());

    for (const auto& c : refined) {
        if (!c.valid) continue;

        const cv::Point2f full_uv(c.uv.x * inv_scale, c.uv.y * inv_scale);
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

    auto detection = buildBestDetectionFromCornerClusters(corners);
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

std::optional<CheckerboardDetection>
CheckerboardDetector::buildBestDetectionFromCornerClusters(
    const std::vector<cv::Point2f>& corners
) const {
    if (static_cast<int>(corners.size()) < config_.min_corners)
        return std::nullopt;

    auto best = buildDetectionFromCorners(corners);

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

    for (int seed : seed_indices) {
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

            std::vector<cv::Point2f> subset;
            subset.reserve(subset_size);
            for (int m = 0; m < subset_size; ++m) {
                subset.push_back(corners[by_dist[m].second]);
            }

            auto candidate = buildDetectionFromCorners(subset);
            if (!candidate || !candidate->valid()) continue;

            if (!best || better(*candidate, *best)) {
                best = std::move(candidate);
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
    if (validation.visible_indices.size() != validation.visible_points.size())
        return std::nullopt;
    if (validation.visible_predicted.size() != validation.visible_points.size())
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
        c.predicted = validation.visible_predicted[m];
        tracked_corners.push_back(c);
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
                if (pc.predicted_frames >=
                    (long_established
                         ? std::max(config_.max_missed_frames, 8)
                         : std::max(config_.max_missed_frames, 4)))
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

                const int min_support = long_established ? 1 : 2;
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
    if (old_persistent_uvs.size() >= 4 && persistent_corners_.size() >= 4) {
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
                    if (pc.predicted_frames >= config_.max_missed_frames) continue;

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

                if (!measured_detection.stable &&
                    (confirmed_with_geometry
                         ? sustained_severe_invisible
                         : (sustained_raw_invisible ||
                            sustained_low_visibility))) {
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
                const bool confirmed = pc.observed_frames >= 3;
                const int missed_limit =
                    established
                        ? std::max(config_.max_missed_frames,
                                   measured_detection.stable ? 30 : 20)
                        : (confirmed
                               ? std::max(config_.max_missed_frames, 12)
                               : config_.max_missed_frames);
                const int predicted_limit =
                    established
                        ? std::max(config_.max_missed_frames,
                                   measured_detection.stable ? 12 : 8)
                        : (confirmed
                               ? std::max(config_.max_missed_frames, 5)
                               : config_.max_missed_frames);
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

    for (const auto& rc : recovery_detection.corners) {
        const int recovery_neighbours = recoveryNeighbourCount(rc);
        const bool strongly_supported_recovery = recovery_neighbours >= 2;

        // --- Fix B: position correction for actively tracked corners ---
        const int existing_idx =
            findPersistentCornerByGrid(persistent_corners_, rc.i, rc.j);

        if (existing_idx >= 0) {
            auto& pc = persistent_corners_[existing_idx];

            // Only correct active corners — stale ones are handled by eviction.
            const float d = distf(pc.corner.uv, rc.uv);
            if (pc.missed_frames == 0) {
                const float max_confirm_d =
                    spacing * (pc.observed_frames >= 8 ? 1.5f : 0.9f);
                if (pc.corner.predicted && d < max_confirm_d) {
                    pc.corner = rc;
                    pc.missed_frames = 0;
                    pc.tracked = true;
                    pc.observed_frames += 1;
                    pc.predicted_frames = 0;
                    pc.low_visibility_frames = 0;
                    pc.visibility_score = 1.0f;
                    pc.smoothed_visibility_score =
                        std::max(pc.smoothed_visibility_score, 0.75f);
                } else if (!pc.corner.predicted && w > 0.0f) {
                    // Blend LK position toward recovery position.
                    const float supported_corr_d =
                        strongly_supported_recovery
                            ? spacing * 1.15f
                            : max_corr_d;
                    if (d < supported_corr_d) {
                        const float corr_w =
                            strongly_supported_recovery
                                ? std::max(w, d > max_corr_d ? 0.85f : 0.65f)
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

        const bool established_corner = pc.observed_frames >= 3;
        const bool long_established_corner = pc.observed_frames >= 8;
        if (measured_this_frame && pc.corner.predicted) {
            const int min_predicted_neighbours =
                long_established_corner ? 1 : 2;
            if (tracked_neighbours < min_predicted_neighbours)
                continue;

            const bool severely_low_visibility =
                pc.smoothed_visibility_score < 0.05f;
            const int max_predicted_output_frames =
                severely_low_visibility
                    ? 2
                    : (long_established_corner ? 6 : 2);
            if (pc.predicted_frames > max_predicted_output_frames)
                continue;
        }

        const int max_hold_frames =
            long_established_corner
                ? (pc.smoothed_visibility_score < 0.05f
                       ? 2
                       : std::max(config_.max_missed_frames, 6))
                : (stable || established_corner ? config_.max_missed_frames : 1);
        const float min_hold_visibility =
            long_established_corner
                ? 0.0f
                : (stable ? 0.05f : (established_corner ? 0.10f : 0.55f));
        const int min_hold_neighbours =
            long_established_corner ? 1 : 2;
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

    // Fill single-cell holes between two currently visible opposite
    // neighbours. This is deliberately conservative: no extrapolation, no
    // isolated points, just the missing middle between a plausible local row
    // or column segment.
    const std::vector<GridCorner> before_gap_fill = visible_corners;
    for (const auto& left : before_gap_fill) {
        const int mi = left.i + 1;
        const int mj = left.j;
        if (hasVisibleGrid(mi, mj)) continue;
        for (const auto& right : before_gap_fill) {
            if (right.i == left.i + 2 && right.j == left.j) {
                addIfMissing(mi, mj, 0.5f * (left.uv + right.uv));
                break;
            }
        }
    }
    for (const auto& top : before_gap_fill) {
        const int mi = top.i;
        const int mj = top.j + 1;
        if (hasVisibleGrid(mi, mj)) continue;
        for (const auto& bottom : before_gap_fill) {
            if (bottom.i == top.i && bottom.j == top.j + 2) {
                addIfMissing(mi, mj, 0.5f * (top.uv + bottom.uv));
                break;
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
                const int input_neighbours = inputNeighbourCount(c);
                const bool long_established = c.observed_frames >= 8;
                const bool locally_confirmed =
                    c.observed_frames >= 3 && input_neighbours >= 2;
                const bool strong_new_observation =
                    !c.predicted &&
                    c.observed_frames >= 1 &&
                    c.visibility_score >=
                        (input_neighbours >= 2 ? 0.30f : 0.45f) &&
                    input_neighbours >= 1;
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
                    long_established ? 0.0f : (stable ? 0.05f : 0.10f);
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
                if (min_cell_dist <= outside_cell_margin) continue;
                if (outputGridSupport(k) >= 2) continue;

                remove[k] = 1;
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
