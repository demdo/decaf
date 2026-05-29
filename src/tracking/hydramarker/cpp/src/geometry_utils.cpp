#include "geometry_utils.hpp"

#include <limits>
#include <numeric>

#include <opencv2/imgproc.hpp>

namespace hydramarker::geom {

namespace {

float meanValue(const std::vector<float>& values)
{
    if (values.empty()) {
        return 0.0f;
    }

    const float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    return sum / static_cast<float>(values.size());
}

float relStdValue(const std::vector<float>& values, float mean)
{
    if (values.empty() || mean <= 1e-6f) {
        return std::numeric_limits<float>::infinity();
    }

    float accum = 0.0f;

    for (float v : values) {
        const float d = v - mean;
        accum += d * d;
    }

    const float variance = accum / static_cast<float>(values.size());
    return std::sqrt(variance) / mean;
}

float clamp01(float x)
{
    return std::max(0.0f, std::min(1.0f, x));
}

CellGeometryValidation validateQuadGeometry(
    const std::array<cv::Point2f, 4>& p,
    const CellGeometryValidationConfig& config
)
{
    CellGeometryValidation out;

    out.indices_valid = true;

    out.finite =
        isFinite(p[0]) &&
        isFinite(p[1]) &&
        isFinite(p[2]) &&
        isFinite(p[3]);

    if (!out.finite) {
        return out;
    }

    out.edge_0 = dist(p[0], p[1]);
    out.edge_1 = dist(p[1], p[2]);
    out.edge_2 = dist(p[2], p[3]);
    out.edge_3 = dist(p[3], p[0]);

    out.diagonal_0 = dist(p[0], p[2]);
    out.diagonal_1 = dist(p[1], p[3]);

    out.signed_area = polygonSignedArea(p);
    out.area = std::abs(out.signed_area);
    out.area_valid = out.area >= config.min_area_px2;

    out.convex = isConvexQuad(p);

    const cv::Point2f center = quadCenter(p);
    out.center_inside = pointInConvexQuad(p, center);

    out.opposite_edge_ratio_u = safeRatio(out.edge_0, out.edge_2);
    out.opposite_edge_ratio_v = safeRatio(out.edge_1, out.edge_3);

    out.opposite_edges_valid =
        out.opposite_edge_ratio_u <= config.max_opposite_edge_ratio &&
        out.opposite_edge_ratio_v <= config.max_opposite_edge_ratio;

    out.diagonal_ratio = safeRatio(out.diagonal_0, out.diagonal_1);
    out.diagonals_valid =
        out.diagonal_ratio <= config.max_diagonal_ratio;

    const float a0 = interiorAngleDeg(p[3], p[0], p[1]);
    const float a1 = interiorAngleDeg(p[0], p[1], p[2]);
    const float a2 = interiorAngleDeg(p[1], p[2], p[3]);
    const float a3 = interiorAngleDeg(p[2], p[3], p[0]);

    out.min_angle_deg = std::min(std::min(a0, a1), std::min(a2, a3));
    out.max_angle_deg = std::max(std::max(a0, a1), std::max(a2, a3));

    out.angles_valid =
        out.min_angle_deg >= config.min_angle_deg &&
        out.max_angle_deg <= config.max_angle_deg;

    const float edge_angle_0 = angle(p[1] - p[0]);
    const float edge_angle_1 = angle(p[2] - p[1]);
    const float edge_angle_2 = angle(p[3] - p[2]);
    const float edge_angle_3 = angle(p[0] - p[3]);

    out.opposite_edge_angle_diff_u_deg =
        rad2deg(angleDiffModPi(edge_angle_0, edge_angle_2));

    out.opposite_edge_angle_diff_v_deg =
        rad2deg(angleDiffModPi(edge_angle_1, edge_angle_3));

    out.opposite_edge_angles_valid =
        out.opposite_edge_angle_diff_u_deg <= config.max_opposite_edge_angle_diff_deg &&
        out.opposite_edge_angle_diff_v_deg <= config.max_opposite_edge_angle_diff_deg;

    out.valid =
        out.finite &&
        out.area_valid &&
        out.convex &&
        out.center_inside &&
        out.opposite_edges_valid &&
        out.diagonals_valid &&
        out.angles_valid &&
        out.opposite_edge_angles_valid;

    return out;
}

} // namespace

cv::Point2f quadCenter(const std::array<cv::Point2f, 4>& quad) {
    return 0.25f * (quad[0] + quad[1] + quad[2] + quad[3]);
}

cv::Point2f cellCenter(const GridCell& cell) {
    return quadCenter(cell.corner_uv);
}

std::array<cv::Point2f, 4> canonicalQuad(
    int size,
    float margin_px
) {
    const float lo = margin_px;
    const float hi = static_cast<float>(size - 1) - margin_px;

    return {
        cv::Point2f(lo, lo),
        cv::Point2f(hi, lo),
        cv::Point2f(hi, hi),
        cv::Point2f(lo, hi)
    };
}

cv::Mat cellHomographyToCanonical(
    const GridCell& cell,
    int canonical_size,
    float margin_px
) {
    const auto dst = canonicalQuad(canonical_size, margin_px);

    std::vector<cv::Point2f> src_pts;
    std::vector<cv::Point2f> dst_pts;

    src_pts.reserve(4);
    dst_pts.reserve(4);

    for (int k = 0; k < 4; ++k) {
        src_pts.push_back(cell.corner_uv[k]);
        dst_pts.push_back(dst[k]);
    }

    return cv::getPerspectiveTransform(src_pts, dst_pts);
}

bool warpCellToCanonical(
    const cv::Mat& image,
    const GridCell& cell,
    cv::Mat& canonical,
    int canonical_size,
    float margin_px,
    int interpolation
) {
    canonical.release();

    if (image.empty()) {
        return false;
    }

    if (canonical_size <= 4) {
        return false;
    }

    const auto validation = validateGridCellGeometry(cell);
    if (!validation.valid) {
        return false;
    }

    cv::Mat gray;

    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    const cv::Mat H = cellHomographyToCanonical(
        cell,
        canonical_size,
        margin_px
    );

    if (H.empty()) {
        return false;
    }

    cv::warpPerspective(
        gray,
        canonical,
        H,
        cv::Size(canonical_size, canonical_size),
        interpolation,
        cv::BORDER_REPLICATE
    );

    return !canonical.empty();
}

float polygonSignedArea(const std::array<cv::Point2f, 4>& p) {
    float a = 0.0f;

    for (int k = 0; k < 4; ++k) {
        const cv::Point2f& p0 = p[k];
        const cv::Point2f& p1 = p[(k + 1) % 4];

        a += p0.x * p1.y - p1.x * p0.y;
    }

    return 0.5f * a;
}

bool isConvexQuad(const std::array<cv::Point2f, 4>& p) {
    constexpr float eps = 1e-5f;

    float sign = 0.0f;

    for (int k = 0; k < 4; ++k) {
        const cv::Point2f& a = p[k];
        const cv::Point2f& b = p[(k + 1) % 4];
        const cv::Point2f& c = p[(k + 2) % 4];

        const float z = cross(b - a, c - b);

        if (std::abs(z) < eps) {
            return false;
        }

        if (sign == 0.0f) {
            sign = z;
        } else if (sign * z < 0.0f) {
            return false;
        }
    }

    return true;
}

bool pointInConvexQuad(
    const std::array<cv::Point2f, 4>& quad,
    const cv::Point2f& point
) {
    constexpr float eps = 1e-4f;

    float sign = 0.0f;

    for (int k = 0; k < 4; ++k) {
        const cv::Point2f& a = quad[k];
        const cv::Point2f& b = quad[(k + 1) % 4];

        const float z = cross(b - a, point - a);

        if (std::abs(z) < eps) {
            continue;
        }

        if (sign == 0.0f) {
            sign = z;
        } else if (sign * z < 0.0f) {
            return false;
        }
    }

    return true;
}

float interiorAngleDeg(
    const cv::Point2f& prev,
    const cv::Point2f& curr,
    const cv::Point2f& next
) {
    const cv::Point2f a = prev - curr;
    const cv::Point2f b = next - curr;

    const float na = norm(a);
    const float nb = norm(b);

    if (na < 1e-6f || nb < 1e-6f) {
        return 0.0f;
    }

    float c = dot(a, b) / (na * nb);
    c = std::max(-1.0f, std::min(1.0f, c));

    return rad2deg(std::acos(c));
}

CellGeometryValidation validateGridCellGeometry(
    const GridCell& cell,
    const CellGeometryValidationConfig& config
) {
    CellGeometryValidation out = validateQuadGeometry(cell.corner_uv, config);

    out.indices_valid =
        cell.corner_indices[0] >= 0 &&
        cell.corner_indices[1] >= 0 &&
        cell.corner_indices[2] >= 0 &&
        cell.corner_indices[3] >= 0;

    out.valid = out.valid && out.indices_valid;

    return out;
}

PatchGeometryValidation validatePatchGeometry(
    const std::vector<std::array<cv::Point2f, 4>>& cell_quads,
    const PatchGeometryValidationConfig& config
) {
    PatchGeometryValidation out;

    out.num_cells = static_cast<int>(cell_quads.size());

    if (cell_quads.empty()) {
        return out;
    }

    std::vector<float> areas;
    std::vector<float> edges;

    areas.reserve(cell_quads.size());
    edges.reserve(cell_quads.size() * 4);

    out.min_cell_angle_deg = std::numeric_limits<float>::infinity();
    out.max_cell_angle_deg = 0.0f;

    bool all_cells_valid = true;

    for (const auto& quad : cell_quads) {
        const CellGeometryValidation v =
            validateQuadGeometry(quad, config.cell_config);

        if (v.valid) {
            ++out.num_valid_cells;
        } else {
            all_cells_valid = false;
        }

        areas.push_back(v.area);

        edges.push_back(v.edge_0);
        edges.push_back(v.edge_1);
        edges.push_back(v.edge_2);
        edges.push_back(v.edge_3);

        out.min_cell_angle_deg =
            std::min(out.min_cell_angle_deg, v.min_angle_deg);

        out.max_cell_angle_deg =
            std::max(out.max_cell_angle_deg, v.max_angle_deg);

        out.max_opposite_edge_ratio =
            std::max(
                out.max_opposite_edge_ratio,
                std::max(v.opposite_edge_ratio_u, v.opposite_edge_ratio_v)
            );

        out.max_diagonal_ratio =
            std::max(out.max_diagonal_ratio, v.diagonal_ratio);

        out.max_opposite_edge_angle_diff_deg =
            std::max(
                out.max_opposite_edge_angle_diff_deg,
                std::max(
                    v.opposite_edge_angle_diff_u_deg,
                    v.opposite_edge_angle_diff_v_deg
                )
            );
    }

    out.mean_cell_area = meanValue(areas);
    out.rel_area_std = relStdValue(areas, out.mean_cell_area);

    out.mean_edge_length = meanValue(edges);
    out.rel_edge_std = relStdValue(edges, out.mean_edge_length);

    if (!std::isfinite(out.min_cell_angle_deg)) {
        out.min_cell_angle_deg = 0.0f;
    }

    const float q_area =
        1.0f - clamp01(out.rel_area_std / config.max_rel_area_std);

    const float q_edge =
        1.0f - clamp01(out.rel_edge_std / config.max_rel_edge_std);

    const float q_cells =
        static_cast<float>(out.num_valid_cells) /
        static_cast<float>(out.num_cells);

    out.quality =
        clamp01(0.40f * q_cells + 0.30f * q_area + 0.30f * q_edge);

    out.valid =
        all_cells_valid &&
        out.rel_area_std <= config.max_rel_area_std &&
        out.rel_edge_std <= config.max_rel_edge_std &&
        out.quality >= config.min_quality;

    return out;
}

} // namespace hydramarker::geom