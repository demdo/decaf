#include "tracker_geometry.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

#include <opencv2/calib3d.hpp>

namespace hydramarker {

namespace {

double sqr(double value) {
    return value * value;
}

double pointDistance(const cv::Point2d& a, const cv::Point2d& b) {
    return std::sqrt(sqr(a.x - b.x) + sqr(a.y - b.y));
}

MapPoseTrackerConfig makeDenseMapPoseTrackerConfig(
    const TrackerConfig& config
) {
    MapPoseTrackerConfig pose_config;
    pose_config.min_points = config.min_points;
    pose_config.min_inliers = config.min_inliers;
    pose_config.ransac_reproj_px = config.pnp_ransac_reprojection_px;
    pose_config.ransac_confidence = config.pnp_ransac_confidence;
    pose_config.ransac_iterations = config.pnp_ransac_iterations;
    pose_config.max_mean_reproj_px = config.max_mean_reprojection_error_px;
    pose_config.max_max_reproj_px = config.max_max_reprojection_error_px;
    pose_config.max_translation_jump_mm = config.max_translation_jump_mm;
    pose_config.max_rotation_jump_deg = config.max_rotation_jump_deg;
    pose_config.rotation_gate_scale_per_lost_frame =
        config.rotation_gate_scale_per_lost_frame;
    pose_config.rotation_gate_max_deg = config.rotation_gate_max_deg;
    pose_config.use_pose_prior = config.use_pose_prior;
    pose_config.refine_with_iterative = true;
    pose_config.use_direct_prior_solver = config.pnp_direct_prior_enabled;
    pose_config.direct_refine_method = config.pnp_direct_refine_method;
    pose_config.direct_max_mean_reproj_px =
        config.pnp_direct_max_mean_reprojection_error_px;
    pose_config.direct_max_max_reproj_px =
        config.pnp_direct_max_max_reprojection_error_px;
    return pose_config;
}

} // namespace

TrackerGeometry::TrackerGeometry(
    const MarkerGeometry& geometry,
    const cv::Matx33d& K,
    const std::vector<double>& dist_coeffs,
    const TrackerConfig& config
)
    : geometry_(geometry),
      K_(K),
      dist_coeffs_(makeDistCoeffsMat(dist_coeffs)),
      dist_coeff_values_(dist_coeffs),
      config_(config)
{
    cached_corners_.clear();
    for (int row = 0; row < geometry_.cornerRows(); ++row) {
        for (int col = 0; col < geometry_.cornerCols(); ++col) {
            if (!geometry_.hasCorner(row, col)) {
                continue;
            }

            const cv::Point3f pt = geometry_.cornerPoint(row, col);
            ProjectedCorner corner;
            corner.global_row = row;
            corner.global_col = col;
            corner.xyz_mm = cv::Point3d(
                static_cast<double>(pt.x),
                static_cast<double>(pt.y),
                static_cast<double>(pt.z)
            );
            cached_corners_.push_back(corner);
        }
    }
}

DenseProjectionMatchResult TrackerGeometry::strictProjectedMatch(
    const CheckerboardDetection& detection,
    const std::vector<double>& rvec,
    const std::vector<double>& tvec,
    double max_dist_px,
    double ambiguity_margin_px
) const
{
    DenseProjectionMatchResult result;
    result.stats.detected = static_cast<int>(detection.corners.size());

    cv::Mat rvec_mat;
    cv::Mat tvec_mat;
    if (
        detection.corners.empty() ||
        cached_corners_.empty() ||
        !vectorToMat3x1(rvec, rvec_mat) ||
        !vectorToMat3x1(tvec, tvec_mat)
    ) {
        return result;
    }

    int rejected_no_projection = 0;
    const std::vector<ProjectedCorner> projected =
        projectGeometry(rvec_mat, tvec_mat, rejected_no_projection);
    result.stats.rejected_no_projection = rejected_no_projection;
    result.stats.projected = static_cast<int>(projected.size());
    if (projected.empty()) {
        return result;
    }

    const int n_projected = static_cast<int>(projected.size());
    const int n_detected = static_cast<int>(detection.corners.size());
    std::vector<int> best_detected_for_projected(
        static_cast<size_t>(n_projected),
        -1
    );
    std::vector<double> best_distances(
        static_cast<size_t>(n_projected),
        std::numeric_limits<double>::infinity()
    );
    std::vector<double> second_distances(
        static_cast<size_t>(n_projected),
        std::numeric_limits<double>::infinity()
    );
    std::vector<int> best_projected_for_detected(
        static_cast<size_t>(n_detected),
        -1
    );
    std::vector<double> best_projected_distances(
        static_cast<size_t>(n_detected),
        std::numeric_limits<double>::infinity()
    );

    for (int pi = 0; pi < n_projected; ++pi) {
        const cv::Point2d projected_uv = projected[static_cast<size_t>(pi)].uv;
        for (int di = 0; di < n_detected; ++di) {
            const GridCorner& detected = detection.corners[static_cast<size_t>(di)];
            const cv::Point2d detected_uv(
                static_cast<double>(detected.uv.x),
                static_cast<double>(detected.uv.y)
            );
            const double dist = pointDistance(projected_uv, detected_uv);

            if (dist < best_distances[static_cast<size_t>(pi)]) {
                second_distances[static_cast<size_t>(pi)] =
                    best_distances[static_cast<size_t>(pi)];
                best_distances[static_cast<size_t>(pi)] = dist;
                best_detected_for_projected[static_cast<size_t>(pi)] = di;
            } else if (dist < second_distances[static_cast<size_t>(pi)]) {
                second_distances[static_cast<size_t>(pi)] = dist;
            }

            if (dist < best_projected_distances[static_cast<size_t>(di)]) {
                best_projected_distances[static_cast<size_t>(di)] = dist;
                best_projected_for_detected[static_cast<size_t>(di)] = pi;
            }
        }
    }

    const double max_dist = static_cast<double>(max_dist_px);
    const double min_margin = static_cast<double>(ambiguity_margin_px);
    std::vector<double> accepted_distances;
    std::vector<cv::Point2d> accepted_uvs;
    std::vector<cv::Point3d> accepted_xyz;
    std::set<int> accepted_rows;
    std::set<int> accepted_cols;

    for (int pi = 0; pi < n_projected; ++pi) {
        const int di = best_detected_for_projected[static_cast<size_t>(pi)];
        if (di < 0) {
            continue;
        }

        const double best_dist = best_distances[static_cast<size_t>(pi)];
        const double second_dist = second_distances[static_cast<size_t>(pi)];

        if (best_dist > max_dist) {
            ++result.stats.rejected_far;
            continue;
        }

        if (
            std::isfinite(second_dist) &&
            (second_dist - best_dist) < min_margin
        ) {
            ++result.stats.rejected_ambiguous;
            continue;
        }

        if (best_projected_for_detected[static_cast<size_t>(di)] != pi) {
            ++result.stats.rejected_non_mutual;
            continue;
        }

        const GridCorner& detected = detection.corners[static_cast<size_t>(di)];
        const ProjectedCorner& projected_corner = projected[static_cast<size_t>(pi)];
        TrackerCorner corner;
        corner.local_row = detected.j;
        corner.local_col = detected.i;
        corner.global_row = projected_corner.global_row;
        corner.global_col = projected_corner.global_col;
        corner.xyz_mm = {
            projected_corner.xyz_mm.x,
            projected_corner.xyz_mm.y,
            projected_corner.xyz_mm.z
        };
        corner.uv = {
            static_cast<double>(detected.uv.x),
            static_cast<double>(detected.uv.y)
        };
        corner.votes = 0;
        result.corners.push_back(corner);

        accepted_distances.push_back(best_dist);
        accepted_uvs.emplace_back(corner.uv[0], corner.uv[1]);
        accepted_xyz.push_back(projected_corner.xyz_mm);
        accepted_rows.insert(projected_corner.global_row);
        accepted_cols.insert(projected_corner.global_col);
    }

    if (accepted_distances.empty()) {
        return result;
    }

    result.stats.median_error_px = percentile(accepted_distances, 50.0);
    result.stats.p90_error_px = percentile(accepted_distances, 90.0);

    double det_min_u = std::numeric_limits<double>::infinity();
    double det_min_v = std::numeric_limits<double>::infinity();
    double det_max_u = -std::numeric_limits<double>::infinity();
    double det_max_v = -std::numeric_limits<double>::infinity();
    for (const GridCorner& detected : detection.corners) {
        det_min_u = std::min(det_min_u, static_cast<double>(detected.uv.x));
        det_min_v = std::min(det_min_v, static_cast<double>(detected.uv.y));
        det_max_u = std::max(det_max_u, static_cast<double>(detected.uv.x));
        det_max_v = std::max(det_max_v, static_cast<double>(detected.uv.y));
    }

    double match_min_u = std::numeric_limits<double>::infinity();
    double match_min_v = std::numeric_limits<double>::infinity();
    double match_max_u = -std::numeric_limits<double>::infinity();
    double match_max_v = -std::numeric_limits<double>::infinity();
    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();

    for (const cv::Point2d& uv : accepted_uvs) {
        match_min_u = std::min(match_min_u, uv.x);
        match_min_v = std::min(match_min_v, uv.y);
        match_max_u = std::max(match_max_u, uv.x);
        match_max_v = std::max(match_max_v, uv.y);
    }

    for (const cv::Point3d& xyz : accepted_xyz) {
        min_x = std::min(min_x, xyz.x);
        min_y = std::min(min_y, xyz.y);
        min_z = std::min(min_z, xyz.z);
        max_x = std::max(max_x, xyz.x);
        max_y = std::max(max_y, xyz.y);
        max_z = std::max(max_z, xyz.z);
    }

    const double detected_span_u = det_max_u - det_min_u;
    const double detected_span_v = det_max_v - det_min_v;
    const double matched_span_u = match_max_u - match_min_u;
    const double matched_span_v = match_max_v - match_min_v;
    result.stats.image_span_u_px = matched_span_u;
    result.stats.image_span_v_px = matched_span_v;

    const double detected_area = detected_span_u * detected_span_v;
    if (detected_area > 1.0) {
        const double matched_area = matched_span_u * matched_span_v;
        result.stats.image_coverage = std::clamp(
            matched_area / detected_area,
            0.0,
            1.0
        );
    }

    result.stats.object_span_mm = std::sqrt(
        sqr(max_x - min_x) +
        sqr(max_y - min_y) +
        sqr(max_z - min_z)
    );
    result.stats.distinct_rows = static_cast<int>(accepted_rows.size());
    result.stats.distinct_cols = static_cast<int>(accepted_cols.size());

    return result;
}

DenseProjectionMatchResult TrackerGeometry::greedyProjectedMatch(
    const CheckerboardDetection& detection,
    const std::vector<double>& rvec,
    const std::vector<double>& tvec,
    double max_dist_px
) const
{
    DenseProjectionMatchResult result;
    result.stats.detected = static_cast<int>(detection.corners.size());

    cv::Mat rvec_mat;
    cv::Mat tvec_mat;
    if (
        detection.corners.empty() ||
        cached_corners_.empty() ||
        !vectorToMat3x1(rvec, rvec_mat) ||
        !vectorToMat3x1(tvec, tvec_mat)
    ) {
        return result;
    }

    int rejected_no_projection = 0;
    const std::vector<ProjectedCorner> projected =
        projectGeometry(rvec_mat, tvec_mat, rejected_no_projection);
    result.stats.rejected_no_projection = rejected_no_projection;
    result.stats.projected = static_cast<int>(projected.size());
    if (projected.empty()) {
        return result;
    }

    const double max_dist_sq = max_dist_px * max_dist_px;
    std::vector<bool> used_projected(projected.size(), false);
    std::vector<double> distances;

    for (const GridCorner& detected : detection.corners) {
        const cv::Point2d detected_uv(
            static_cast<double>(detected.uv.x),
            static_cast<double>(detected.uv.y)
        );
        int best_idx = -1;
        double best_dist_sq = std::numeric_limits<double>::infinity();

        for (int pi = 0; pi < static_cast<int>(projected.size()); ++pi) {
            if (used_projected[static_cast<size_t>(pi)]) {
                continue;
            }
            const cv::Point2d& uv = projected[static_cast<size_t>(pi)].uv;
            const double dist_sq = sqr(uv.x - detected_uv.x) +
                                   sqr(uv.y - detected_uv.y);
            if (dist_sq < best_dist_sq) {
                best_dist_sq = dist_sq;
                best_idx = pi;
            }
        }

        if (best_idx < 0 || best_dist_sq > max_dist_sq) {
            continue;
        }

        used_projected[static_cast<size_t>(best_idx)] = true;
        const ProjectedCorner& projected_corner =
            projected[static_cast<size_t>(best_idx)];

        TrackerCorner corner;
        corner.local_row = detected.j;
        corner.local_col = detected.i;
        corner.global_row = projected_corner.global_row;
        corner.global_col = projected_corner.global_col;
        corner.xyz_mm = {
            projected_corner.xyz_mm.x,
            projected_corner.xyz_mm.y,
            projected_corner.xyz_mm.z
        };
        corner.uv = {
            static_cast<double>(detected.uv.x),
            static_cast<double>(detected.uv.y)
        };
        corner.votes = 0;
        result.corners.push_back(corner);
        distances.push_back(std::sqrt(best_dist_sq));
    }

    if (!distances.empty()) {
        result.stats.median_error_px = percentile(distances, 50.0);
        result.stats.p90_error_px = percentile(distances, 90.0);
    }

    return result;
}

std::vector<TrackerCorner> TrackerGeometry::visualCornersFromPose(
    const std::vector<TrackerCorner>& corners,
    const std::vector<double>& rvec,
    const std::vector<double>& tvec,
    double max_error_px
) const
{
    std::vector<TrackerCorner> accepted;

    cv::Mat rvec_mat;
    cv::Mat tvec_mat;
    if (
        corners.empty() ||
        !vectorToMat3x1(rvec, rvec_mat) ||
        !vectorToMat3x1(tvec, tvec_mat)
    ) {
        return accepted;
    }

    std::vector<cv::Point3d> object_points;
    object_points.reserve(corners.size());
    for (const TrackerCorner& corner : corners) {
        object_points.emplace_back(
            corner.xyz_mm[0],
            corner.xyz_mm[1],
            corner.xyz_mm[2]
        );
    }

    std::vector<cv::Point2d> projected;
    try {
        cv::projectPoints(
            object_points,
            rvec_mat,
            tvec_mat,
            K_,
            dist_coeffs_,
            projected
        );
    } catch (...) {
        return accepted;
    }

    const double max_error = static_cast<double>(max_error_px);
    for (int idx = 0; idx < static_cast<int>(corners.size()); ++idx) {
        const cv::Point2d& uv = projected[static_cast<size_t>(idx)];
        if (!std::isfinite(uv.x) || !std::isfinite(uv.y)) {
            continue;
        }

        const TrackerCorner& corner = corners[static_cast<size_t>(idx)];
        const double error = std::sqrt(
            sqr(uv.x - corner.uv[0]) +
            sqr(uv.y - corner.uv[1])
        );
        if (error > max_error) {
            continue;
        }

        accepted.push_back(corner);
    }

    return accepted;
}

MapPoseResult TrackerGeometry::estimateDenseRobustPose(
    const std::vector<PoseTrackPoint>& points,
    const CheckerboardDetection& detection,
    const std::vector<double>& seed_rvec,
    const std::vector<double>& seed_tvec,
    const std::vector<double>& previous_rvec,
    const std::vector<double>& previous_tvec
) const
{
    MapPoseResult result;
    result.num_points = static_cast<int>(points.size());

    if (points.empty()) {
        result.success = false;
        result.message = "Dense robust solver failed: no candidate.";
        result.method = "dense_robust_failed";
        return result;
    }

    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2d> image_points;
    object_points.reserve(points.size());
    image_points.reserve(points.size());
    for (const PoseTrackPoint& point : points) {
        object_points.emplace_back(
            point.xyz_mm[0],
            point.xyz_mm[1],
            point.xyz_mm[2]
        );
        image_points.emplace_back(point.uv[0], point.uv[1]);
    }

    cv::Mat seed_rvec_mat;
    cv::Mat seed_tvec_mat;
    const bool has_seed =
        vectorToMat3x1(seed_rvec, seed_rvec_mat) &&
        vectorToMat3x1(seed_tvec, seed_tvec_mat);

    std::vector<DensePoseCandidate> candidates;
    if (has_seed) {
        std::vector<DensePoseCandidate> seed_candidates =
            denseRefinePoseVariants(
                object_points,
                image_points,
                seed_rvec_mat,
                seed_tvec_mat,
                "dense_seed"
            );
        candidates.insert(
            candidates.end(),
            seed_candidates.begin(),
            seed_candidates.end()
        );
    }

    const std::vector<std::pair<int, std::string>> solve_flags = {
        {cv::SOLVEPNP_SQPNP, "dense_sqpnp"},
        {cv::SOLVEPNP_EPNP, "dense_epnp"}
    };

    for (const auto& [flag, name] : solve_flags) {
        try {
            cv::Mat rvec;
            cv::Mat tvec;
            const bool success = cv::solvePnP(
                object_points,
                image_points,
                K_,
                dist_coeffs_,
                rvec,
                tvec,
                false,
                flag
            );
            if (!success) {
                continue;
            }

            std::vector<DensePoseCandidate> variants =
                denseRefinePoseVariants(
                    object_points,
                    image_points,
                    rvec,
                    tvec,
                    name
                );
            candidates.insert(
                candidates.end(),
                variants.begin(),
                variants.end()
            );
        } catch (...) {
        }
    }

    if (has_seed) {
        try {
            cv::Mat rvec = seed_rvec_mat.clone();
            cv::Mat tvec = seed_tvec_mat.clone();
            const bool success = cv::solvePnP(
                object_points,
                image_points,
                K_,
                dist_coeffs_,
                rvec,
                tvec,
                true,
                cv::SOLVEPNP_ITERATIVE
            );
            if (success) {
                std::vector<DensePoseCandidate> variants =
                    denseRefinePoseVariants(
                        object_points,
                        image_points,
                        rvec,
                        tvec,
                        "dense_iterative_guess"
                    );
                candidates.insert(
                    candidates.end(),
                    variants.begin(),
                    variants.end()
                );
            }
        } catch (...) {
        }
    }

    bool has_best = false;
    double best_score = std::numeric_limits<double>::infinity();
    cv::Mat best_rvec;
    cv::Mat best_tvec;
    std::string best_method;
    std::vector<double> best_errors;

    for (const DensePoseCandidate& candidate : candidates) {
        DensePoseScore scored;
        if (!scoreDensePoseCandidate(
                object_points,
                image_points,
                candidate.rvec,
                candidate.tvec,
                scored
            )) {
            continue;
        }

        if (!has_best || scored.score < best_score) {
            has_best = true;
            best_score = scored.score;
            best_rvec = candidate.rvec.reshape(1, 3).clone();
            best_tvec = candidate.tvec.reshape(1, 3).clone();
            best_method = candidate.method;
            best_errors = scored.errors;
        }
    }

    if (!has_best) {
        result.success = false;
        result.message = "Dense robust solver failed: no candidate.";
        result.num_inliers = 0;
        result.method = "dense_robust_failed";
        return result;
    }

    std::vector<int> inlier_indices(points.size());
    std::iota(inlier_indices.begin(), inlier_indices.end(), 0);

    if (
        config_.fast_persistent_dense_robust_trim_enabled &&
        best_errors.size() >= 12
    ) {
        const double median = percentile(best_errors, 50.0);
        std::vector<double> abs_deviation;
        abs_deviation.reserve(best_errors.size());
        for (double error : best_errors) {
            abs_deviation.push_back(std::abs(error - median));
        }

        const double mad = percentile(abs_deviation, 50.0);
        const double robust_sigma = 1.4826 * mad;
        const double robust_threshold = std::max(
            0.75,
            median + 4.0 * robust_sigma
        );
        const double max_threshold =
            config_.fast_persistent_dense_robust_max_max_px;
        double threshold = std::min(max_threshold, robust_threshold);

        const double quantile =
            config_.fast_persistent_dense_robust_trim_quantile;
        if (quantile > 0.0 && quantile < 1.0) {
            threshold = std::min(
                threshold,
                percentile(best_errors, quantile * 100.0)
            );
        }

        std::vector<int> trim_indices;
        trim_indices.reserve(best_errors.size());
        for (int idx = 0; idx < static_cast<int>(best_errors.size()); ++idx) {
            if (best_errors[static_cast<size_t>(idx)] <= threshold) {
                trim_indices.push_back(idx);
            }
        }

        const int min_keep = std::max(
            config_.min_inliers,
            static_cast<int>(std::ceil(
                config_.fast_persistent_dense_robust_min_keep_ratio *
                static_cast<double>(best_errors.size())
            ))
        );

        if (
            static_cast<int>(trim_indices.size()) >= min_keep &&
            trim_indices.size() < best_errors.size()
        ) {
            std::vector<cv::Point3d> object_trim;
            std::vector<cv::Point2d> image_trim;
            object_trim.reserve(trim_indices.size());
            image_trim.reserve(trim_indices.size());
            for (int idx : trim_indices) {
                object_trim.push_back(object_points[static_cast<size_t>(idx)]);
                image_trim.push_back(image_points[static_cast<size_t>(idx)]);
            }

            std::vector<DensePoseCandidate> trim_candidates =
                denseRefinePoseVariants(
                    object_trim,
                    image_trim,
                    best_rvec,
                    best_tvec,
                    best_method + "_trim" +
                        std::to_string(trim_indices.size())
                );

            bool has_trim_best = false;
            double trim_best_score = std::numeric_limits<double>::infinity();
            cv::Mat trim_best_rvec;
            cv::Mat trim_best_tvec;
            std::string trim_best_method;
            std::vector<double> trim_best_errors;

            for (const DensePoseCandidate& candidate : trim_candidates) {
                DensePoseScore scored;
                if (!scoreDensePoseCandidate(
                        object_trim,
                        image_trim,
                        candidate.rvec,
                        candidate.tvec,
                        scored
                    )) {
                    continue;
                }
                if (!has_trim_best || scored.score < trim_best_score) {
                    has_trim_best = true;
                    trim_best_score = scored.score;
                    trim_best_rvec = candidate.rvec.reshape(1, 3).clone();
                    trim_best_tvec = candidate.tvec.reshape(1, 3).clone();
                    trim_best_method = candidate.method;
                    trim_best_errors = scored.errors;
                }
            }

            if (has_trim_best) {
                best_rvec = trim_best_rvec;
                best_tvec = trim_best_tvec;
                best_method = trim_best_method;
                best_errors = trim_best_errors;
                inlier_indices = trim_indices;
            }
        }
    }

    const double mean_error = std::accumulate(
        best_errors.begin(),
        best_errors.end(),
        0.0
    ) / static_cast<double>(best_errors.size());
    const double max_error = *std::max_element(
        best_errors.begin(),
        best_errors.end()
    );

    auto fill_pose = [&]() {
        result.rvec = mat3x1ToVector(best_rvec);
        result.tvec = mat3x1ToVector(best_tvec);
        result.T_marker_camera = transformToVector(
            makeTransform(best_rvec, best_tvec)
        );
        result.reprojection_mean_px = mean_error;
        result.reprojection_max_px = max_error;
        result.num_points = static_cast<int>(points.size());
        result.num_inliers = static_cast<int>(inlier_indices.size());
        result.inlier_indices = inlier_indices;
        result.method = best_method;
    };

    if (
        mean_error > config_.fast_persistent_dense_robust_max_mean_px ||
        max_error > config_.fast_persistent_dense_robust_max_max_px
    ) {
        result.success = false;
        result.message =
            "Dense robust pose rejected by reprojection gate (mean=" +
            formatDouble(mean_error, 3) +
            ", max=" + formatDouble(max_error, 3) + ").";
        fill_pose();
        return result;
    }

    cv::Mat previous_rvec_mat;
    cv::Mat previous_tvec_mat;
    const bool has_previous =
        vectorToMat3x1(previous_rvec, previous_rvec_mat) &&
        vectorToMat3x1(previous_tvec, previous_tvec_mat);
    if (
        has_previous &&
        !persistentMotionPlausible(
            best_rvec,
            best_tvec,
            previous_rvec_mat,
            previous_tvec_mat
        )
    ) {
        result.success = false;
        result.message = "Dense robust pose rejected by motion gate.";
        fill_pose();
        return result;
    }

    const std::string fallback_reason = fallbackPoseRejectionReason(
        detection,
        best_rvec,
        best_tvec,
        mean_error,
        max_error
    );
    if (!fallback_reason.empty()) {
        result.success = false;
        result.message = fallback_reason;
        fill_pose();
        return result;
    }

    result.success = true;
    result.message = "Dense robust pose accepted.";
    fill_pose();
    result.points.reserve(inlier_indices.size());
    for (int idx : inlier_indices) {
        if (idx >= 0 && idx < static_cast<int>(points.size())) {
            result.points.push_back(points[static_cast<size_t>(idx)]);
        }
    }
    return result;
}

MapPoseResult TrackerGeometry::estimateDenseDirectPose(
    const std::vector<PoseTrackPoint>& points,
    const std::vector<double>& seed_rvec,
    const std::vector<double>& seed_tvec,
    int lost_frames
) const
{
    MapPoseTracker pose_tracker(
        K_,
        dist_coeff_values_,
        makeDenseMapPoseTrackerConfig(config_)
    );

    if (!seed_rvec.empty() && !seed_tvec.empty()) {
        pose_tracker.setPose(seed_rvec, seed_tvec);
    }

    return pose_tracker.estimatePose(points, lost_frames);
}

cv::Mat TrackerGeometry::makeDistCoeffsMat(
    const std::vector<double>& dist_coeffs
)
{
    if (dist_coeffs.empty()) {
        return cv::Mat();
    }

    cv::Mat mat(static_cast<int>(dist_coeffs.size()), 1, CV_64F);
    for (int i = 0; i < static_cast<int>(dist_coeffs.size()); ++i) {
        mat.at<double>(i, 0) = dist_coeffs[static_cast<size_t>(i)];
    }
    return mat;
}

bool TrackerGeometry::vectorToMat3x1(
    const std::vector<double>& values,
    cv::Mat& mat
)
{
    if (values.size() != 3) {
        mat.release();
        return false;
    }

    mat = cv::Mat(3, 1, CV_64F);
    mat.at<double>(0, 0) = values[0];
    mat.at<double>(1, 0) = values[1];
    mat.at<double>(2, 0) = values[2];
    return true;
}

double TrackerGeometry::percentile(std::vector<double> values, double q)
{
    if (values.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    std::sort(values.begin(), values.end());
    if (values.size() == 1) {
        return values.front();
    }

    const double clamped = std::clamp(q, 0.0, 100.0);
    const double pos = (clamped / 100.0) *
                       static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) {
        return values[lo];
    }

    const double alpha = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - alpha) + values[hi] * alpha;
}

std::string TrackerGeometry::formatDouble(double value, int precision)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string TrackerGeometry::lowerAscii(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
    );
    return value;
}

std::vector<double> TrackerGeometry::mat3x1ToVector(const cv::Mat& mat)
{
    cv::Mat reshaped = mat.reshape(1, 3);
    return {
        reshaped.at<double>(0, 0),
        reshaped.at<double>(1, 0),
        reshaped.at<double>(2, 0)
    };
}

std::vector<double> TrackerGeometry::transformToVector(const cv::Matx44d& T)
{
    std::vector<double> values;
    values.reserve(16);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            values.push_back(T(r, c));
        }
    }
    return values;
}

cv::Matx44d TrackerGeometry::makeTransform(
    const cv::Mat& rvec,
    const cv::Mat& tvec
)
{
    cv::Mat R_mat;
    cv::Rodrigues(rvec, R_mat);

    cv::Matx44d T = cv::Matx44d::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            T(r, c) = R_mat.at<double>(r, c);
        }
    }
    T(0, 3) = tvec.at<double>(0, 0);
    T(1, 3) = tvec.at<double>(1, 0);
    T(2, 3) = tvec.at<double>(2, 0);
    return T;
}

std::vector<TrackerGeometry::DensePoseCandidate>
TrackerGeometry::denseRefinePoseVariants(
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const std::string& method_prefix
) const
{
    std::vector<DensePoseCandidate> variants;
    DensePoseCandidate base;
    base.rvec = rvec.reshape(1, 3).clone();
    base.tvec = tvec.reshape(1, 3).clone();
    base.method = method_prefix;
    variants.push_back(base);

    const std::string configured = lowerAscii(
        config_.fast_persistent_dense_robust_refine_method
    );

    std::vector<std::string> methods;
    if (configured == "auto") {
        methods = {"lm", "vvs"};
    } else if (configured == "lm" || configured == "vvs") {
        methods = {configured};
    }

    for (const std::string& method : methods) {
        if (method == "lm") {
            try {
                cv::Mat rvec_ref = base.rvec.clone();
                cv::Mat tvec_ref = base.tvec.clone();
                cv::solvePnPRefineLM(
                    object_points,
                    image_points,
                    K_,
                    dist_coeffs_,
                    rvec_ref,
                    tvec_ref
                );
                DensePoseCandidate refined;
                refined.rvec = rvec_ref.reshape(1, 3).clone();
                refined.tvec = tvec_ref.reshape(1, 3).clone();
                refined.method = method_prefix + "_lm";
                variants.push_back(refined);
            } catch (...) {
            }
        }

        if (method == "vvs") {
            try {
                cv::Mat rvec_ref = base.rvec.clone();
                cv::Mat tvec_ref = base.tvec.clone();
                cv::solvePnPRefineVVS(
                    object_points,
                    image_points,
                    K_,
                    dist_coeffs_,
                    rvec_ref,
                    tvec_ref
                );
                DensePoseCandidate refined;
                refined.rvec = rvec_ref.reshape(1, 3).clone();
                refined.tvec = tvec_ref.reshape(1, 3).clone();
                refined.method = method_prefix + "_vvs";
                variants.push_back(refined);
            } catch (...) {
            }
        }
    }

    return variants;
}

bool TrackerGeometry::scoreDensePoseCandidate(
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    DensePoseScore& score
) const
{
    std::vector<double> errors;
    if (!reprojectionErrors(object_points, image_points, rvec, tvec, errors)) {
        return false;
    }

    for (double error : errors) {
        if (!std::isfinite(error)) {
            return false;
        }
    }

    const double median = percentile(errors, 50.0);
    const double p90 = percentile(errors, 90.0);
    const double mean = std::accumulate(errors.begin(), errors.end(), 0.0) /
                        static_cast<double>(errors.size());
    score.score = median + 0.35 * p90 + 0.15 * mean;
    score.errors = std::move(errors);
    return true;
}

bool TrackerGeometry::reprojectionErrors(
    const std::vector<cv::Point3d>& object_points,
    const std::vector<cv::Point2d>& image_points,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    std::vector<double>& errors
) const
{
    if (object_points.empty() || object_points.size() != image_points.size()) {
        return false;
    }

    std::vector<cv::Point2d> projected;
    try {
        cv::projectPoints(
            object_points,
            rvec,
            tvec,
            K_,
            dist_coeffs_,
            projected
        );
    } catch (...) {
        return false;
    }

    if (projected.size() != image_points.size()) {
        return false;
    }

    errors.clear();
    errors.reserve(projected.size());
    for (size_t idx = 0; idx < projected.size(); ++idx) {
        errors.push_back(pointDistance(projected[idx], image_points[idx]));
    }
    return true;
}

bool TrackerGeometry::persistentMotionPlausible(
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const cv::Mat& previous_rvec,
    const cv::Mat& previous_tvec
) const
{
    if (rvec.empty() || tvec.empty()) {
        return false;
    }
    if (previous_rvec.empty() || previous_tvec.empty()) {
        return true;
    }

    try {
        cv::Mat prev_R;
        cv::Mat curr_R;
        cv::Rodrigues(previous_rvec, prev_R);
        cv::Rodrigues(rvec, curr_R);
        const cv::Mat dR = curr_R * prev_R.t();
        double trace = cv::trace(dR)[0];
        double cos_a = (trace - 1.0) * 0.5;
        cos_a = std::clamp(cos_a, -1.0, 1.0);
        const double rot_delta_deg =
            std::acos(cos_a) * 180.0 / CV_PI;
        const double trans_delta_mm = cv::norm(tvec - previous_tvec);
        return (
            rot_delta_deg <= config_.persistence_max_rotation_jump_deg &&
            trans_delta_mm <= config_.persistence_max_translation_jump_mm
        );
    } catch (...) {
        return false;
    }
}

std::string TrackerGeometry::fallbackPoseRejectionReason(
    const CheckerboardDetection& detection,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    double mean_reproj_px,
    double max_reproj_px
) const
{
    if (mean_reproj_px > config_.fallback_pose_max_mean_reprojection_error_px) {
        return (
            "Fallback pose rejected by mean reprojection gate (" +
            formatDouble(mean_reproj_px, 2) +
            "px)."
        );
    }

    if (max_reproj_px > config_.fallback_pose_max_max_reprojection_error_px) {
        return (
            "Fallback pose rejected by max reprojection gate (" +
            formatDouble(max_reproj_px, 2) +
            "px)."
        );
    }

    DenseProjectionMatchResult visual = greedyProjectedMatch(
        detection,
        mat3x1ToVector(rvec),
        mat3x1ToVector(tvec),
        config_.fallback_pose_max_p90_corner_error_px
    );
    const int match_count = static_cast<int>(visual.corners.size());
    const double median_err = visual.stats.median_error_px;
    const double p90_err = visual.stats.p90_error_px;

    if (match_count < config_.fallback_pose_min_detection_matches) {
        return (
            "Fallback pose rejected by blue-corner alignment (" +
            std::to_string(match_count) +
            " matches)."
        );
    }

    if (median_err > config_.fallback_pose_max_median_corner_error_px) {
        return (
            "Fallback pose rejected by median blue-corner error (" +
            formatDouble(median_err, 2) +
            "px)."
        );
    }

    if (p90_err > config_.fallback_pose_max_p90_corner_error_px) {
        return (
            "Fallback pose rejected by p90 blue-corner error (" +
            formatDouble(p90_err, 2) +
            "px)."
        );
    }

    return "";
}

std::vector<TrackerGeometry::ProjectedCorner> TrackerGeometry::projectGeometry(
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    int& rejected_no_projection
) const
{
    rejected_no_projection = 0;
    std::vector<ProjectedCorner> projected_corners;
    if (cached_corners_.empty()) {
        return projected_corners;
    }

    std::vector<cv::Point3d> object_points;
    object_points.reserve(cached_corners_.size());
    for (const ProjectedCorner& corner : cached_corners_) {
        object_points.push_back(corner.xyz_mm);
    }

    std::vector<cv::Point2d> projected_uvs;
    cv::Mat R;
    try {
        cv::projectPoints(
            object_points,
            rvec,
            tvec,
            K_,
            dist_coeffs_,
            projected_uvs
        );
        cv::Rodrigues(rvec, R);
    } catch (...) {
        rejected_no_projection = static_cast<int>(cached_corners_.size());
        return {};
    }

    projected_corners.reserve(cached_corners_.size());
    for (int idx = 0; idx < static_cast<int>(cached_corners_.size()); ++idx) {
        const ProjectedCorner& corner = cached_corners_[static_cast<size_t>(idx)];
        const cv::Point2d& uv = projected_uvs[static_cast<size_t>(idx)];
        const cv::Mat X = (
            R * (cv::Mat_<double>(3, 1) <<
                corner.xyz_mm.x,
                corner.xyz_mm.y,
                corner.xyz_mm.z
            )
        ) + tvec;

        const bool valid =
            std::isfinite(uv.x) &&
            std::isfinite(uv.y) &&
            std::isfinite(X.at<double>(0, 0)) &&
            std::isfinite(X.at<double>(1, 0)) &&
            std::isfinite(X.at<double>(2, 0)) &&
            X.at<double>(2, 0) > 1e-6;

        if (!valid) {
            ++rejected_no_projection;
            continue;
        }

        ProjectedCorner out = corner;
        out.uv = uv;
        projected_corners.push_back(out);
    }

    return projected_corners;
}

} // namespace hydramarker
