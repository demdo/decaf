#include "marker_geometry.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

namespace {

int readIntOrDefault(
    const cv::FileNode& node,
    const std::string& key,
    int default_value
) {
    cv::FileNode value = node[key];

    if (value.empty()) {
        return default_value;
    }

    int out = default_value;
    value >> out;
    return out;
}

double readDoubleOrDefault(
    const cv::FileNode& node,
    const std::string& key,
    double default_value
) {
    cv::FileNode value = node[key];

    if (value.empty()) {
        return default_value;
    }

    double out = default_value;
    value >> out;
    return out;
}

// Margins added around the exported marker band so that template windows of
// border corners are not masked away (matches the validated A1 prototype).
constexpr double kBandAxialMarginMm   = 4.0;
constexpr double kBandAngularMarginRad = 0.35;

} // namespace


int MarkerGeometry::index(int row, int col) const {
    return row * corner_cols_ + col;
}


bool MarkerGeometry::empty() const {
    return corner_rows_ <= 0 || corner_cols_ <= 0 || corner_xyz_mm_.empty();
}


int MarkerGeometry::cornerRows() const {
    return corner_rows_;
}


int MarkerGeometry::cornerCols() const {
    return corner_cols_;
}


int MarkerGeometry::detectableOriginRow() const {
    return detectable_origin_row_;
}


int MarkerGeometry::detectableOriginCol() const {
    return detectable_origin_col_;
}


bool MarkerGeometry::hasCorner(int row, int col) const {
    if (row < 0 || col < 0) {
        return false;
    }

    if (row >= corner_rows_ || col >= corner_cols_) {
        return false;
    }

    return corner_valid_[index(row, col)] != 0;
}


cv::Point3f MarkerGeometry::cornerPoint(int row, int col) const {
    if (!hasCorner(row, col)) {
        throw std::out_of_range("MarkerGeometry: requested invalid corner.");
    }

    return corner_xyz_mm_[index(row, col)];
}


MarkerGeometry MarkerGeometry::loadFromJson(const std::string& path) {
    cv::FileStorage fs(
        path,
        cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON
    );

    if (!fs.isOpened()) {
        throw std::runtime_error(
            "MarkerGeometry: could not open JSON file: " + path
        );
    }

    MarkerGeometry geometry;

    // Surface-model loading shared by both exit paths (explicit SfM corners
    // and planar fallback).  Must run after the corner coordinates are
    // filled because the planar band is derived from the corner bbox.
    const auto load_surface_model = [&fs, &geometry]() {
        SurfaceModel sm;

        const cv::FileNode surface_node = fs["surface_model"];
        std::string surface_type;
        if (!surface_node.empty()) {
            surface_node["type"] >> surface_type;
        }

        if (surface_type == "cylinder") {
            const cv::FileNode fitted = surface_node["fitted"];
            if (!fitted.empty()) {
                std::vector<double> p0, d, e1, ax, th;
                double r = 0.0;
                double rms = 0.0;
                fitted["axis_point_mm"] >> p0;
                fitted["axis_dir"] >> d;
                fitted["radial_ref_dir"] >> e1;
                fitted["radius_mm"] >> r;
                fitted["fit_rms_mm"] >> rms;
                fitted["axial_range_mm"] >> ax;
                fitted["angular_range_rad"] >> th;

                if (p0.size() == 3 && d.size() == 3 && e1.size() == 3 &&
                    ax.size() == 2 && th.size() == 2 && r > 0.0 &&
                    cv::norm(cv::Vec3d(d[0], d[1], d[2])) > 1e-9) {
                    sm.type = SurfaceModel::Type::Cylinder;
                    sm.point = cv::Vec3d(p0[0], p0[1], p0[2]);
                    const cv::Vec3d dd(d[0], d[1], d[2]);
                    sm.dir = dd / cv::norm(dd);
                    const cv::Vec3d ee(e1[0], e1[1], e1[2]);
                    sm.radial_ref = ee / cv::norm(ee);
                    sm.radius_mm = r;
                    sm.fit_rms_mm = rms;
                    sm.band_a_min = ax[0] - kBandAxialMarginMm;
                    sm.band_a_max = ax[1] + kBandAxialMarginMm;
                    sm.band_b_min = th[0] - kBandAngularMarginRad;
                    sm.band_b_max = th[1] + kBandAngularMarginRad;
                }
            }
        }

        // Plane fallback based on the ACTUAL tracked geometry: when every
        // valid corner lies in the z = 0 marker plane, the surface is that
        // plane — regardless of a declared (future) curved mounting that
        // has no fitted numbers yet.  marker_type is ignored on purpose:
        // a stale "surface_model: cylinder" without "fitted" must not
        // disable the measurement for a marker whose geometry is planar.
        if (sm.type == SurfaceModel::Type::None) {
            constexpr double kPlanarTolMm = 0.2;
            double x_min = 0.0, x_max = 0.0, y_min = 0.0, y_max = 0.0;
            bool first = true;
            bool planar_ok = true;
            for (size_t i = 0; i < geometry.corner_xyz_mm_.size(); ++i) {
                if (!geometry.corner_valid_[i]) continue;
                const cv::Point3f& p = geometry.corner_xyz_mm_[i];
                if (std::abs(static_cast<double>(p.z)) > kPlanarTolMm) {
                    planar_ok = false;
                    break;
                }
                if (first) {
                    x_min = x_max = p.x;
                    y_min = y_max = p.y;
                    first = false;
                } else {
                    x_min = std::min(x_min, static_cast<double>(p.x));
                    x_max = std::max(x_max, static_cast<double>(p.x));
                    y_min = std::min(y_min, static_cast<double>(p.y));
                    y_max = std::max(y_max, static_cast<double>(p.y));
                }
            }
            if (!first && planar_ok) {
                sm.type = SurfaceModel::Type::Plane;
                sm.point = cv::Vec3d(0.0, 0.0, 0.0);
                sm.dir = cv::Vec3d(0.0, 0.0, 1.0);
                sm.basis_u = cv::Vec3d(1.0, 0.0, 0.0);
                sm.basis_v = cv::Vec3d(0.0, 1.0, 0.0);
                sm.band_a_min = x_min - kBandAxialMarginMm;
                sm.band_a_max = x_max + kBandAxialMarginMm;
                sm.band_b_min = y_min - kBandAxialMarginMm;
                sm.band_b_max = y_max + kBandAxialMarginMm;
            }
        }

        geometry.surface_model_ = sm;
    };

    fs["corner_rows"] >> geometry.corner_rows_;
    fs["corner_cols"] >> geometry.corner_cols_;

    if (geometry.corner_rows_ <= 0 || geometry.corner_cols_ <= 0) {
        throw std::runtime_error(
            "MarkerGeometry: JSON must contain positive corner_rows and corner_cols."
        );
    }

    const int n = geometry.corner_rows_ * geometry.corner_cols_;

    geometry.corner_xyz_mm_.assign(
        n,
        cv::Point3f(0.0f, 0.0f, 0.0f)
    );

    geometry.corner_valid_.assign(
        n,
        0
    );

    /*
     * New convention:
     *
     * global row/col stay absolute checkerboard coordinates.
     *
     * Example for a 12 x 12 cell marker:
     *   full corner grid      : 13 x 13
     *   detectable origin     : row=1, col=1
     *
     * Then:
     *   corner(1,1) -> (0, 0, 0)
     *   corner(1,2) -> (+square_size_mm, 0, 0)
     *   corner(2,1) -> (0, +square_size_mm, 0)
     *
     * This keeps the tracker labels as 1,1 / 1,2 / 2,1,
     * but makes the first detectable corner the metric marker origin.
     */
    cv::FileNode id_encoding = fs["id_encoding"];

    if (!id_encoding.empty()) {
        geometry.detectable_origin_row_ = readIntOrDefault(
            id_encoding,
            "origin_row",
            0
        );

        geometry.detectable_origin_col_ = readIntOrDefault(
            id_encoding,
            "origin_col",
            0
        );
    } else {
        geometry.detectable_origin_row_ = readIntOrDefault(
            fs.root(),
            "detectable_origin_row",
            0
        );

        geometry.detectable_origin_col_ = readIntOrDefault(
            fs.root(),
            "detectable_origin_col",
            0
        );
    }

    if (
        geometry.detectable_origin_row_ < 0 ||
        geometry.detectable_origin_col_ < 0 ||
        geometry.detectable_origin_row_ >= geometry.corner_rows_ ||
        geometry.detectable_origin_col_ >= geometry.corner_cols_
    ) {
        throw std::runtime_error(
            "MarkerGeometry: detectable origin is outside the full corner grid."
        );
    }

    /*
     * Future non-planar / SfM marker format:
     *
     * "corners": [
     *   {"row": 1, "col": 1, "xyz_mm": [0.0, 0.0, 0.0]},
     *   {"row": 1, "col": 2, "xyz_mm": [14.3, 0.0, 0.0]},
     *   ...
     * ]
     *
     * For explicit corners, we trust the coordinates from JSON directly.
     */
    cv::FileNode corners_node = fs["corners"];

    if (!corners_node.empty()) {
        for (const cv::FileNode& node : corners_node) {
            int row = -1;
            int col = -1;
            std::vector<float> xyz;

            node["row"] >> row;
            node["col"] >> col;
            node["xyz_mm"] >> xyz;

            if (
                row < 0 ||
                col < 0 ||
                row >= geometry.corner_rows_ ||
                col >= geometry.corner_cols_
            ) {
                continue;
            }

            if (xyz.size() != 3) {
                continue;
            }

            const int idx = geometry.index(row, col);

            geometry.corner_xyz_mm_[idx] = cv::Point3f(
                xyz[0],
                xyz[1],
                xyz[2]
            );

            geometry.corner_valid_[idx] = 1;
        }

        load_surface_model();
        return geometry;
    }

    /*
     * Planar marker fallback:
     *
     * If no explicit 3D corner list exists, build planar coordinates from
     * square_size_mm and shift them so that the detectable origin is (0,0,0).
     */
    double square_size_mm = 0.0;

    fs["square_size_mm"] >> square_size_mm;

    if (square_size_mm <= 0.0) {
        const double square_size_cm = readDoubleOrDefault(
            fs.root(),
            "square_size_cm",
            0.0
        );

        if (square_size_cm > 0.0) {
            square_size_mm = 10.0 * square_size_cm;
        }
    }

    if (square_size_mm <= 0.0) {
        throw std::runtime_error(
            "MarkerGeometry: JSON has no explicit corners and no valid square_size_mm."
        );
    }

    for (int row = 0; row < geometry.corner_rows_; ++row) {
        for (int col = 0; col < geometry.corner_cols_; ++col) {
            const int idx = geometry.index(row, col);

            const float x_mm = static_cast<float>(
                (col - geometry.detectable_origin_col_) * square_size_mm
            );

            const float y_mm = static_cast<float>(
                (row - geometry.detectable_origin_row_) * square_size_mm
            );

            geometry.corner_xyz_mm_[idx] = cv::Point3f(
                x_mm,
                y_mm,
                0.0f
            );

            geometry.corner_valid_[idx] = 1;
        }
    }

    load_surface_model();
    return geometry;
}

} // namespace hydramarker