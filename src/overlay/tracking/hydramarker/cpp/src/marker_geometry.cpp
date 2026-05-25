#include "marker_geometry.hpp"

#include <stdexcept>

#include <opencv2/core.hpp>

namespace hydramarker {

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
    cv::FileStorage fs(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

    if (!fs.isOpened()) {
        throw std::runtime_error("MarkerGeometry: could not open JSON file: " + path);
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
    geometry.corner_xyz_mm_.assign(n, cv::Point3f(0.0f, 0.0f, 0.0f));
    geometry.corner_valid_.assign(n, 0);

    // Future non-planar / SfM marker format:
    //
    // "corners": [
    //   {"row": 0, "col": 0, "xyz_mm": [0.0, 0.0, 0.0]},
    //   ...
    // ]
    cv::FileNode corners_node = fs["corners"];

    if (!corners_node.empty()) {
        for (const cv::FileNode& node : corners_node) {
            int row = -1;
            int col = -1;
            std::vector<float> xyz;

            node["row"] >> row;
            node["col"] >> col;
            node["xyz_mm"] >> xyz;

            if (row < 0 || col < 0 ||
                row >= geometry.corner_rows_ ||
                col >= geometry.corner_cols_) {
                continue;
            }

            if (xyz.size() != 3) {
                continue;
            }

            const int idx = geometry.index(row, col);
            geometry.corner_xyz_mm_[idx] = cv::Point3f(xyz[0], xyz[1], xyz[2]);
            geometry.corner_valid_[idx] = 1;
        }

        return geometry;
    }

    // Current planar marker fallback:
    // X = col * square_size_mm
    // Y = row * square_size_mm
    // Z = 0
    double square_size_mm = 0.0;
    fs["square_size_mm"] >> square_size_mm;

    if (square_size_mm <= 0.0) {
        throw std::runtime_error(
            "MarkerGeometry: JSON has no explicit corners and no valid square_size_mm."
        );
    }

    for (int row = 0; row < geometry.corner_rows_; ++row) {
        for (int col = 0; col < geometry.corner_cols_; ++col) {
            const int idx = geometry.index(row, col);

            geometry.corner_xyz_mm_[idx] = cv::Point3f(
                static_cast<float>(col * square_size_mm),
                static_cast<float>(row * square_size_mm),
                0.0f
            );

            geometry.corner_valid_[idx] = 1;
        }
    }

    return geometry;
}

} // namespace hydramarker