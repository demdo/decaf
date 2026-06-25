#include "tracker_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
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

} // namespace

TrackerGeometry::TrackerGeometry(
    const MarkerGeometry& geometry,
    const cv::Matx33d& K,
    const std::vector<double>& dist_coeffs
)
    : geometry_(geometry),
      K_(K),
      dist_coeffs_(makeDistCoeffsMat(dist_coeffs))
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
