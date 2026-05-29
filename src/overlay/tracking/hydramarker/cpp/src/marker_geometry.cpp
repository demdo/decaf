#include "marker_geometry.hpp"

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

    return geometry;
}

} // namespace hydramarker