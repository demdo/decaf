#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace hydramarker {

// Continuous surface the marker is mounted on, in the marker coordinate
// frame (mm).  Loaded from the geometry JSON:
//   - planar markers ("marker_type": "planar") get an analytic plane;
//   - curved markers need the surface_model.fitted block written by the SfM
//     export (export_marker_map.py); without it the type stays None and
//     model-based corner measurement is unavailable (subpix fallback).
struct SurfaceModel {
    enum class Type { None, Plane, Cylinder };

    Type type = Type::None;

    // Plane: point on plane + unit normal.
    // Cylinder: point on axis + unit axis direction.
    cv::Vec3d point{0.0, 0.0, 0.0};
    cv::Vec3d dir{0.0, 0.0, 1.0};

    // Cylinder only: radius and the angular reference direction that centres
    // the marker band at theta = 0 (radial_ref_dir from the export).
    double radius_mm = 0.0;
    cv::Vec3d radial_ref{1.0, 0.0, 0.0};

    // Marker band on the surface, used to mask template pixels that leave
    // the printed area.  Cylinder: along-axis range [mm] + theta range [rad].
    // Plane: in-plane bounding box along basis_u/basis_v [mm].
    double band_a_min = 0.0;
    double band_a_max = 0.0;
    double band_b_min = 0.0;
    double band_b_max = 0.0;
    cv::Vec3d basis_u{1.0, 0.0, 0.0};
    cv::Vec3d basis_v{0.0, 1.0, 0.0};

    double fit_rms_mm = 0.0;

    bool valid() const { return type != Type::None; }
};

class MarkerGeometry {
public:
    static MarkerGeometry loadFromJson(const std::string& path);

    bool empty() const;
    bool hasCorner(int row, int col) const;

    cv::Point3f cornerPoint(int row, int col) const;

    int cornerRows() const;
    int cornerCols() const;

    int detectableOriginRow() const;
    int detectableOriginCol() const;

    const SurfaceModel& surfaceModel() const { return surface_model_; }

private:
    SurfaceModel surface_model_;

    int corner_rows_ = 0;
    int corner_cols_ = 0;

    int detectable_origin_row_ = 0;
    int detectable_origin_col_ = 0;

    std::vector<cv::Point3f> corner_xyz_mm_;
    std::vector<uint8_t> corner_valid_;

    int index(int row, int col) const;
};

} // namespace hydramarker