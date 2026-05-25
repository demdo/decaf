#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/core.hpp>

#include "checkerboard_types.hpp"

namespace hydramarker::geom {

constexpr float PI_F = static_cast<float>(CV_PI);

// ---------------------------
// Basic vector ops
// ---------------------------

inline float dist2(const cv::Point2f& a, const cv::Point2f& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

inline float dist(const cv::Point2f& a, const cv::Point2f& b) {
    return std::sqrt(dist2(a, b));
}

inline float norm(const cv::Point2f& v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

inline cv::Point2f normalize(const cv::Point2f& v) {
    const float n = norm(v);
    if (n < 1e-6f) {
        return cv::Point2f(0.0f, 0.0f);
    }
    return v * (1.0f / n);
}

inline float dot(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x * b.x + a.y * b.y;
}

inline float cross(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x * b.y - a.y * b.x;
}

// ---------------------------
// Angles
// ---------------------------

inline float angle(const cv::Point2f& v) {
    return std::atan2(v.y, v.x);
}

inline float angleDiff(float a, float b) {
    float d = a - b;
    while (d > PI_F) d -= 2.0f * PI_F;
    while (d < -PI_F) d += 2.0f * PI_F;
    return d;
}

inline float angleDiffModPi(float a, float b) {
    const float d = std::abs(angleDiff(a, b));
    return std::min(d, std::abs(d - PI_F));
}

inline float rad2deg(float rad) {
    return rad * 180.0f / PI_F;
}

// ---------------------------
// Midpoint / interpolation
// ---------------------------

inline cv::Point2f midpoint(const cv::Point2f& a, const cv::Point2f& b) {
    return 0.5f * (a + b);
}

// ---------------------------
// Projection
// ---------------------------

inline float projectScalar(const cv::Point2f& v, const cv::Point2f& axis) {
    return dot(v, axis);
}

// ---------------------------
// Utility
// ---------------------------

inline bool isFinite(const cv::Point2f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

inline float safeRatio(float a, float b, float eps = 1e-6f) {
    const float aa = std::abs(a);
    const float bb = std::abs(b);

    if (aa < eps || bb < eps) {
        return std::numeric_limits<float>::infinity();
    }

    return std::max(aa, bb) / std::min(aa, bb);
}

// ---------------------------
// Cell / quad geometry
// ---------------------------

cv::Point2f quadCenter(const std::array<cv::Point2f, 4>& quad);

cv::Point2f cellCenter(const GridCell& cell);

std::array<cv::Point2f, 4> canonicalQuad(
    int size,
    float margin_px = 0.0f
);

cv::Mat cellHomographyToCanonical(
    const GridCell& cell,
    int canonical_size,
    float margin_px = 0.0f
);

bool warpCellToCanonical(
    const cv::Mat& image,
    const GridCell& cell,
    cv::Mat& canonical,
    int canonical_size = 40,
    float margin_px = 0.0f,
    int interpolation = 1
);

// ---------------------------
// Cell geometry validation
// ---------------------------

struct CellGeometryValidationConfig {
    float min_area_px2 = 25.0f;

    float max_opposite_edge_ratio = 1.6f;
    float max_diagonal_ratio = 1.7f;

    float min_angle_deg = 35.0f;
    float max_angle_deg = 145.0f;

    float max_opposite_edge_angle_diff_deg = 35.0f;
};

struct CellGeometryValidation {
    bool valid = false;

    bool finite = false;
    bool indices_valid = false;
    bool area_valid = false;
    bool convex = false;
    bool center_inside = false;
    bool opposite_edges_valid = false;
    bool diagonals_valid = false;
    bool angles_valid = false;
    bool opposite_edge_angles_valid = false;

    float signed_area = 0.0f;
    float area = 0.0f;

    float edge_0 = 0.0f;
    float edge_1 = 0.0f;
    float edge_2 = 0.0f;
    float edge_3 = 0.0f;

    float opposite_edge_ratio_u = 0.0f;
    float opposite_edge_ratio_v = 0.0f;

    float diagonal_0 = 0.0f;
    float diagonal_1 = 0.0f;
    float diagonal_ratio = 0.0f;

    float min_angle_deg = 0.0f;
    float max_angle_deg = 0.0f;

    float opposite_edge_angle_diff_u_deg = 0.0f;
    float opposite_edge_angle_diff_v_deg = 0.0f;
};

float polygonSignedArea(const std::array<cv::Point2f, 4>& p);

bool isConvexQuad(const std::array<cv::Point2f, 4>& p);

bool pointInConvexQuad(
    const std::array<cv::Point2f, 4>& quad,
    const cv::Point2f& point
);

float interiorAngleDeg(
    const cv::Point2f& prev,
    const cv::Point2f& curr,
    const cv::Point2f& next
);

CellGeometryValidation validateGridCellGeometry(
    const GridCell& cell,
    const CellGeometryValidationConfig& config = CellGeometryValidationConfig{}
);

// ---------------------------
// Patch geometry validation
// ---------------------------

struct PatchGeometryValidationConfig {
    CellGeometryValidationConfig cell_config;

    float max_rel_area_std = 0.40f;
    float max_rel_edge_std = 0.35f;

    float min_quality = 0.50f;
};

struct PatchGeometryValidation {
    bool valid = false;

    int num_cells = 0;
    int num_valid_cells = 0;

    float mean_cell_area = 0.0f;
    float rel_area_std = 0.0f;

    float mean_edge_length = 0.0f;
    float rel_edge_std = 0.0f;

    float min_cell_angle_deg = 0.0f;
    float max_cell_angle_deg = 0.0f;

    float max_opposite_edge_ratio = 0.0f;
    float max_diagonal_ratio = 0.0f;
    float max_opposite_edge_angle_diff_deg = 0.0f;

    float quality = 0.0f;
};

PatchGeometryValidation validatePatchGeometry(
    const std::vector<std::array<cv::Point2f, 4>>& cell_quads,
    const PatchGeometryValidationConfig& config = PatchGeometryValidationConfig{}
);

} // namespace hydramarker::geom