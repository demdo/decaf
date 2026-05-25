#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

class MarkerGeometry {
public:
    static MarkerGeometry loadFromJson(const std::string& path);

    bool empty() const;
    bool hasCorner(int row, int col) const;

    cv::Point3f cornerPoint(int row, int col) const;

    int cornerRows() const;
    int cornerCols() const;

private:
    int corner_rows_ = 0;
    int corner_cols_ = 0;

    std::vector<cv::Point3f> corner_xyz_mm_;
    std::vector<uint8_t> corner_valid_;

    int index(int row, int col) const;
};

} // namespace hydramarker