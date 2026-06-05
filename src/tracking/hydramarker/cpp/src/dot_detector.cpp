#include "dot_detector.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace hydramarker {

DotDetector::DotDetector()
    : config_()
{
}

DotDetector::DotDetector(DotDetectorConfig config)
    : config_(config)
{
}

void DotDetector::reset()
{
    temporal_states_.clear();
    cell_value_cache_.clear();
    frame_index_ = 0;
}

void DotDetector::reset_smoothing()
{
    for (auto& kv : temporal_states_) {
        TemporalCellState& state = kv.second;

        // EMA-Score auf den aktuellen has_dot-Zustand zuruecksetzen:
        // hat die Zelle einen Dot, setzen wir den Score auf commit_threshold,
        // sonst auf revoke_threshold. Damit startet der EMA neu ohne
        // dass der naechste raw_score gegen einen weit entfernten
        // eingefroren Wert ankämpfen muss.
        if (state.has_dot) {
            state.ema_score = config_.commit_threshold;
        } else {
            state.ema_score = config_.revoke_threshold;
        }

        // Commit/Revoke-Counter zuruecksetzen damit sofort
        // re-committed werden kann.
        state.commit_count = 0;
        state.revoke_count = 0;

        // initialized und has_dot bleiben erhalten —
        // der Warmup-State geht nicht verloren.
    }
}

DotDetectionResult DotDetector::detect(
    const cv::Mat& image,
    const CheckerboardDetection& checkerboard
)
{
    DotDetectionResult result;
    frame_index_ += 1;

    if (image.empty()) {
        return result;
    }

    cv::Mat gray = toGray8(image);

    cv::Mat gray_f32;
    gray.convertTo(gray_f32, CV_32F);

    cv::Scalar mean_scalar;
    cv::Scalar std_scalar;
    cv::meanStdDev(gray_f32, mean_scalar, std_scalar);

    const double frame_mean = mean_scalar[0];
    const double frame_std = std::max(std_scalar[0], 1.0);

    int max_row = -1;
    int max_col = -1;

    std::set<std::pair<int, int>> visible_keys;

    result.cells.reserve(checkerboard.cells.size());

    for (const GridCell& cell : checkerboard.cells) {
        DotCellObservation obs;

        obs.row = cell.j;
        obs.col = cell.i;

        obs.center_uv = cell.center_uv;
        obs.corners_uv = cell.corner_uv;

        const LocalScoreResult score = evaluateCell(
            gray_f32,
            cell,
            frame_mean,
            frame_std
        );

        obs.raw_score = score.score;

        obs.center_mean = score.fg_mean;
        obs.ring_mean = score.bg_mean;

        obs.local_mean = score.local_mean;
        obs.local_std = score.local_std;

        obs.polarity = score.polarity;
        obs.valid = true;

        const std::pair<int, int> key(obs.row, obs.col);
        visible_keys.insert(key);

        if (config_.use_temporal_smoothing) {
            TemporalCellState& state = temporal_states_[key];

            obs.has_dot = updateTemporalState(state, score.score);
            obs.score = state.ema_score;
        }
        else {
            // Stateless mode: use the current-frame score directly.
            // This avoids EMA warmup-lock and prevents temporal state from
            // leaking across different visible sides of a cylindrical marker.
            obs.score = score.score;
            obs.has_dot = score.score >= config_.commit_threshold;
        }

        obs.ambiguous =
            obs.score >= config_.uncertainty_low &&
            obs.score < config_.uncertainty_high;

        if (config_.use_cell_value_cache && obs.ambiguous) {
            const auto cached_it = cell_value_cache_.find(key);
            if (
                cached_it != cell_value_cache_.end() &&
                isCellCacheMatch(cached_it->second, obs)
            ) {
                obs.has_dot = cached_it->second.has_dot;
                obs.score = cached_it->second.score;
                obs.ambiguous = false;
                obs.cache_reused = true;
            }
        }

        if (config_.use_cell_value_cache && !obs.ambiguous) {
            CachedCellValue cached;
            cached.has_dot = obs.has_dot;
            cached.score = obs.score;
            cached.corners_uv = obs.corners_uv;
            cached.last_seen_frame = frame_index_;
            cell_value_cache_[key] = cached;
        }

        result.cells.push_back(obs);

        max_row = std::max(max_row, obs.row);
        max_col = std::max(max_col, obs.col);
    }

    if (config_.use_temporal_smoothing) {
        for (auto it = temporal_states_.begin(); it != temporal_states_.end();) {
            if (visible_keys.find(it->first) == visible_keys.end()) {
                it->second.missed_frames += 1;

                if (it->second.missed_frames > 15) {
                    it = temporal_states_.erase(it);
                    continue;
                }
            }
            else {
                it->second.missed_frames = 0;
            }

            ++it;
        }
    }

    result.rows = max_row + 1;
    result.cols = max_col + 1;

    return result;
}

bool DotDetector::isCellCacheMatch(
    const CachedCellValue& cached,
    const DotCellObservation& obs
) const
{
    const int max_age = std::max(config_.cell_cache_max_age_frames, 0);
    if (cached.last_seen_frame < 0 || frame_index_ - cached.last_seen_frame > max_age) {
        return false;
    }

    const float max_motion = std::max(config_.cell_cache_max_corner_motion_px, 0.0f);
    for (int i = 0; i < 4; ++i) {
        const cv::Point2f d = obs.corners_uv[i] - cached.corners_uv[i];
        if (std::sqrt(d.x * d.x + d.y * d.y) > max_motion) {
            return false;
        }
    }

    return true;
}

bool DotDetector::updateTemporalState(
    TemporalCellState& state,
    double raw_score
) const
{
    const double alpha = std::clamp(config_.temporal_alpha, 0.01, 1.0);
    const int warmup_frames = std::max(config_.warmup_frames, 1);
    const int commit_frames = std::max(config_.commit_frames, 1);
    const int revoke_frames = std::max(config_.revoke_frames, 1);

    if (!state.initialized) {
        state.initialized = true;
        state.ema_score = raw_score;
        state.has_dot = raw_score >= config_.commit_threshold;
        state.seen_frames = 1;
        state.commit_count = state.has_dot ? commit_frames : 0;
        state.revoke_count = state.has_dot ? 0 : revoke_frames;
        return state.has_dot;
    }

    state.seen_frames += 1;
    state.ema_score = alpha * raw_score + (1.0 - alpha) * state.ema_score;

    if (state.seen_frames <= warmup_frames) {
        state.has_dot = state.ema_score >= config_.commit_threshold;
        return state.has_dot;
    }

    if (state.ema_score >= config_.commit_threshold) {
        state.commit_count += 1;
        state.revoke_count = 0;
    }
    else if (state.ema_score <= config_.revoke_threshold) {
        state.revoke_count += 1;
        state.commit_count = 0;
    }
    else {
        state.commit_count = 0;
        state.revoke_count = 0;
    }

    if (!state.has_dot && state.commit_count >= commit_frames) {
        state.has_dot = true;
    }
    else if (state.has_dot && state.revoke_count >= revoke_frames) {
        state.has_dot = false;
    }

    return state.has_dot;
}

cv::Mat DotDetector::toGray8(const cv::Mat& image)
{
    if (image.empty()) {
        return {};
    }

    if (image.type() == CV_8UC1) {
        return image;
    }

    cv::Mat gray;

    if (image.type() == CV_8UC3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }

    if (image.type() == CV_8UC4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
        return gray;
    }

    throw std::runtime_error("DotDetector: unsupported image type.");
}

double DotDetector::sampleBilinearClamp(
    const cv::Mat& gray_f32,
    const cv::Point2f& p
)
{
    const int width = gray_f32.cols;
    const int height = gray_f32.rows;

    float x = std::clamp(p.x, 0.0f, static_cast<float>(width - 1));
    float y = std::clamp(p.y, 0.0f, static_cast<float>(height - 1));

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));

    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);

    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);

    const float v00 = gray_f32.at<float>(y0, x0);
    const float v10 = gray_f32.at<float>(y0, x1);
    const float v01 = gray_f32.at<float>(y1, x0);
    const float v11 = gray_f32.at<float>(y1, x1);

    const float v0 = v00 * (1.0f - dx) + v10 * dx;
    const float v1 = v01 * (1.0f - dx) + v11 * dx;

    return static_cast<double>(v0 * (1.0f - dy) + v1 * dy);
}

DotDetector::LocalScoreResult DotDetector::evaluateCell(
    const cv::Mat& gray_f32,
    const GridCell& cell,
    double frame_mean,
    double frame_std
) const
{
    (void)frame_mean;
    (void)frame_std;

    LocalScoreResult result;

    const int canonical_size = std::max(config_.canonical_size, 48);
    const float margin = std::max(config_.canonical_margin_px, 3.0f);

    std::vector<cv::Point2f> src = {
        cell.corner_uv[0],
        cell.corner_uv[1],
        cell.corner_uv[2],
        cell.corner_uv[3],
    };

    std::vector<cv::Point2f> dst = {
        cv::Point2f(margin, margin),
        cv::Point2f(canonical_size - 1.0f - margin, margin),
        cv::Point2f(canonical_size - 1.0f - margin, canonical_size - 1.0f - margin),
        cv::Point2f(margin, canonical_size - 1.0f - margin),
    };

    const cv::Mat H = cv::getPerspectiveTransform(src, dst);

    cv::Mat warped;
    cv::warpPerspective(
        gray_f32,
        warped,
        H,
        cv::Size(canonical_size, canonical_size),
        cv::INTER_LINEAR,
        cv::BORDER_REPLICATE
    );

    cv::GaussianBlur(warped, warped, cv::Size(3, 3), 0.0);

    const float cx = 0.5f * static_cast<float>(canonical_size - 1);
    const float cy = 0.5f * static_cast<float>(canonical_size - 1);
    const float half = 0.5f * static_cast<float>(canonical_size - 1) - margin;

    std::vector<double> center_values;
    std::vector<double> ring_values;
    std::vector<double> local_values;

    center_values.reserve(canonical_size * canonical_size / 16);
    ring_values.reserve(canonical_size * canonical_size / 8);
    local_values.reserve(canonical_size * canonical_size / 2);

    for (int y = 0; y < canonical_size; ++y) {
        const float* row_ptr = warped.ptr<float>(y);

        for (int x = 0; x < canonical_size; ++x) {
            const float dx = (static_cast<float>(x) - cx) / half;
            const float dy = (static_cast<float>(y) - cy) / half;
            const float r = std::sqrt(dx * dx + dy * dy);

            if (r <= 0.55f) {
                local_values.push_back(static_cast<double>(row_ptr[x]));
            }

            if (r <= 0.20f) {
                center_values.push_back(static_cast<double>(row_ptr[x]));
            }
            else if (r >= 0.32f && r <= 0.50f) {
                ring_values.push_back(static_cast<double>(row_ptr[x]));
            }
        }
    }

    if (center_values.empty() || ring_values.empty() || local_values.empty()) {
        result.score = 0.0;
        result.fg_mean = 0.0;
        result.bg_mean = 0.0;
        result.local_mean = 0.0;
        result.local_std = 1.0;
        result.signed_contrast = 0.0;
        result.abs_contrast = 0.0;
        result.polarity = 0;
        return result;
    }

    auto mean_of = [](const std::vector<double>& values) -> double {
        double s = 0.0;
        for (double v : values) {
            s += v;
        }
        return s / static_cast<double>(values.size());
    };

    auto std_of = [](const std::vector<double>& values, double mean) -> double {
        double s = 0.0;
        for (double v : values) {
            const double d = v - mean;
            s += d * d;
        }
        return std::sqrt(std::max(s / static_cast<double>(values.size()), 1.0));
    };

    const double center_mean = mean_of(center_values);
    const double ring_mean = mean_of(ring_values);
    const double local_mean = mean_of(local_values);

    const double center_std = std_of(center_values, center_mean);
    const double ring_std = std_of(ring_values, ring_mean);
    const double local_std = std_of(local_values, local_mean);

    result.fg_mean = center_mean;
    result.bg_mean = ring_mean;

    result.signed_contrast = center_mean - ring_mean;
    result.abs_contrast = std::abs(result.signed_contrast);

    result.local_mean = local_mean;
    result.local_std = std::max(local_std, 1.0);

    if (result.signed_contrast > 0.0) {
        result.polarity = 1;
    }
    else if (result.signed_contrast < 0.0) {
        result.polarity = -1;
    }
    else {
        result.polarity = 0;
    }

    if (result.abs_contrast < config_.min_dot_contrast) {
        result.score = 0.0;
        return result;
    }

    const double adaptive_delta = std::max(
        0.30 * result.local_std,
        0.45 * config_.min_dot_contrast
    );

    auto is_opposite_to_ring = [&](double v) -> bool {
        if (result.polarity > 0) {
            return v > ring_mean + adaptive_delta;
        }
        if (result.polarity < 0) {
            return v < ring_mean - adaptive_delta;
        }
        return false;
    };

    int inner_opposite = 0;
    int inner_total = 0;
    int mid_opposite = 0;
    int mid_total = 0;
    int outer_opposite = 0;
    int outer_total = 0;

    double radial_edge_sum = 0.0;
    double radial_edge_weighted = 0.0;
    double all_inner_edge_sum = 0.0;

    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Sobel(warped, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(warped, grad_y, CV_32F, 0, 1, 3);

    for (int y = 1; y < canonical_size - 1; ++y) {
        const float* row_ptr = warped.ptr<float>(y);
        const float* gx_ptr = grad_x.ptr<float>(y);
        const float* gy_ptr = grad_y.ptr<float>(y);

        for (int x = 1; x < canonical_size - 1; ++x) {
            const float nx = (static_cast<float>(x) - cx) / half;
            const float ny = (static_cast<float>(y) - cy) / half;
            const float r = std::sqrt(nx * nx + ny * ny);

            if (r > 0.58f) {
                continue;
            }

            const double v = static_cast<double>(row_ptr[x]);
            const bool opposite = is_opposite_to_ring(v);

            if (r <= 0.20f) {
                inner_total += 1;
                if (opposite) {
                    inner_opposite += 1;
                }
            }
            else if (r > 0.20f && r <= 0.34f) {
                mid_total += 1;
                if (opposite) {
                    mid_opposite += 1;
                }
            }
            else if (r > 0.34f && r <= 0.52f) {
                outer_total += 1;
                if (opposite) {
                    outer_opposite += 1;
                }
            }

            const double gx = static_cast<double>(gx_ptr[x]);
            const double gy = static_cast<double>(gy_ptr[x]);
            const double mag = std::sqrt(gx * gx + gy * gy);

            if (r >= 0.10f && r <= 0.50f) {
                all_inner_edge_sum += mag;
            }

            if (r >= 0.18f && r <= 0.36f && mag > 1e-6) {
                const double inv_r = 1.0 / std::max(static_cast<double>(r), 1e-6);
                const double ux = static_cast<double>(nx) * inv_r;
                const double uy = static_cast<double>(ny) * inv_r;
                const double radial_alignment = std::abs(gx * ux + gy * uy) / mag;

                radial_edge_sum += mag;
                radial_edge_weighted += mag * radial_alignment;
            }
        }
    }

    const double inner_ratio =
        inner_total > 0 ? static_cast<double>(inner_opposite) / static_cast<double>(inner_total) : 0.0;
    const double mid_ratio =
        mid_total > 0 ? static_cast<double>(mid_opposite) / static_cast<double>(mid_total) : 0.0;
    const double outer_ratio =
        outer_total > 0 ? static_cast<double>(outer_opposite) / static_cast<double>(outer_total) : 0.0;

    const double contrast_score = std::clamp(
        result.abs_contrast / std::max(config_.strong_dot_contrast, 1.0),
        0.0,
        1.0
    );

    const double area_score = std::clamp(
        inner_ratio / 0.45,
        0.0,
        1.0
    );

    const double compactness_score = std::clamp(
        (inner_ratio + 0.5 * mid_ratio - 0.8 * outer_ratio) / 0.65,
        0.0,
        1.0
    );

    const double radial_alignment_score =
        radial_edge_sum > 1e-6
            ? std::clamp(radial_edge_weighted / radial_edge_sum, 0.0, 1.0)
            : 0.0;

    const double edge_strength_score = std::clamp(
        radial_edge_sum / std::max(18.0 * static_cast<double>(canonical_size), 1.0),
        0.0,
        1.0
    );

    const double edge_localization_score =
        all_inner_edge_sum > 1e-6
            ? std::clamp(radial_edge_sum / all_inner_edge_sum, 0.0, 1.0)
            : 0.0;

    const double edge_ring_score = std::clamp(
        0.45 * radial_alignment_score +
        0.35 * edge_strength_score +
        0.20 * edge_localization_score,
        0.0,
        1.0
    );

    const double raw_score =
        0.25 * contrast_score +
        0.25 * area_score +
        0.25 * compactness_score +
        0.25 * edge_ring_score;

    result.score = std::clamp(raw_score, 0.0, 1.0);

    if (contrast_score < 0.12 && edge_ring_score < 0.30) {
        result.score = 0.0;
    }

    return result;
}

} // namespace hydramarker
