// bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <optional>
#include <stdexcept>

#include <opencv2/core.hpp>

#include "marker_field.hpp"
#include "marker_geometry.hpp"
#include "correspondence_builder.hpp"
#include "generator_HydraMarker.h"

#include "checkerboard_detector.hpp"
#include "checkerboard_types.hpp"
#include "dot_detector.hpp"
#include "geometry_utils.hpp"

#include "corner_detection.hpp"
#include "corner_refinement.hpp"
#include "grid_builder.hpp"
#include "lattice_model.hpp"
#include "lk_tracker.hpp"
#include "tracking_validator.hpp"

#include "patch_extractor.hpp"
#include "patch_decoder.hpp"

namespace py = pybind11;

namespace hydramarker {

namespace {

cv::Mat numpyToMat(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img
) {
    py::buffer_info info = img.request();

    if (info.ndim != 2 && info.ndim != 3) {
        throw std::runtime_error("Image must have 2 or 3 dimensions");
    }

    if (info.ndim == 2) {
        return cv::Mat(
            static_cast<int>(info.shape[0]),
            static_cast<int>(info.shape[1]),
            CV_8UC1,
            info.ptr
        );
    }

    const int channels = static_cast<int>(info.shape[2]);

    if (channels == 3) {
        return cv::Mat(
            static_cast<int>(info.shape[0]),
            static_cast<int>(info.shape[1]),
            CV_8UC3,
            info.ptr
        );
    }

    if (channels == 4) {
        return cv::Mat(
            static_cast<int>(info.shape[0]),
            static_cast<int>(info.shape[1]),
            CV_8UC4,
            info.ptr
        );
    }

    throw std::runtime_error("Unsupported channel count");
}

py::array_t<uint8_t> mat1bToNumpy(const cv::Mat1b& mat)
{
    py::array_t<uint8_t> arr({mat.rows, mat.cols});
    py::buffer_info info = arr.request();

    uint8_t* dst = static_cast<uint8_t*>(info.ptr);

    for (int r = 0; r < mat.rows; ++r) {
        for (int c = 0; c < mat.cols; ++c) {
            dst[r * mat.cols + c] = mat(r, c);
        }
    }

    return arr;
}

} // namespace

PYBIND11_MODULE(hydramarker_cpp, m) {
    m.doc() = "HydraMarker C++ bindings";

    m.def(
        "generate_planar_field",
        [](int rows,
           int cols,
           int patch_size,
           double max_ms,
           int max_trial,
           bool is_print) -> py::array_t<uint8_t>
        {
            cv::Mat1b field = generator_HydraMarker::generate_planar_field(
                rows,
                cols,
                patch_size,
                max_ms,
                max_trial,
                is_print
            );

            return mat1bToNumpy(field);
        },
        py::arg("rows"),
        py::arg("cols"),
        py::arg("patch_size"),
        py::arg("max_ms") = 60000.0,
        py::arg("max_trial") = 100000,
        py::arg("is_print") = false
    );

    py::class_<cv::Point2f>(m, "Point2f")
        .def(py::init<>())
        .def_readwrite("x", &cv::Point2f::x)
        .def_readwrite("y", &cv::Point2f::y);

    py::class_<cv::Point3f>(m, "Point3f")
        .def(py::init<>())
        .def_readwrite("x", &cv::Point3f::x)
        .def_readwrite("y", &cv::Point3f::y)
        .def_readwrite("z", &cv::Point3f::z);

    py::class_<PatchMatch>(m, "PatchMatch")
        .def(py::init<>())
        .def_readwrite("x", &PatchMatch::x)
        .def_readwrite("y", &PatchMatch::y)
        .def_readwrite("rotation_deg", &PatchMatch::rotation_deg);

    py::class_<MarkerField>(m, "MarkerField")
        .def(py::init<>())
        .def_static("loadFromFile", &MarkerField::loadFromFile)
        .def("width", &MarkerField::width)
        .def("height", &MarkerField::height)
        .def("patchSize", &MarkerField::patchSize)
        .def("empty", &MarkerField::empty)
        .def("at", &MarkerField::at)
        .def("getPatch", &MarkerField::getPatch)
        .def("findPatch", &MarkerField::findPatch);

    py::class_<MarkerGeometry>(m, "MarkerGeometry")
        .def(py::init<>())
        .def_static("load_from_json", &MarkerGeometry::loadFromJson)
        .def("empty", &MarkerGeometry::empty)
        .def("has_corner", &MarkerGeometry::hasCorner)
        .def("corner_point", &MarkerGeometry::cornerPoint)
        .def("corner_rows", &MarkerGeometry::cornerRows)
        .def("corner_cols", &MarkerGeometry::cornerCols)
        .def("detectable_origin_row", &MarkerGeometry::detectableOriginRow)
        .def("detectable_origin_col", &MarkerGeometry::detectableOriginCol);

    py::class_<GridCorner>(m, "GridCorner")
        .def(py::init<>())
        .def_readwrite("i", &GridCorner::i)
        .def_readwrite("j", &GridCorner::j)
        .def_readwrite("uv", &GridCorner::uv)
        .def_readwrite("visibility_score", &GridCorner::visibility_score)
        .def_readwrite("observed_frames", &GridCorner::observed_frames)
        .def_readwrite("predicted", &GridCorner::predicted);

    py::class_<GridCell>(m, "GridCell")
        .def(py::init<>())
        .def_readwrite("i", &GridCell::i)
        .def_readwrite("j", &GridCell::j)
        .def_readwrite("corner_indices", &GridCell::corner_indices)
        .def_property(
            "corner_uv",
            [](const GridCell& self) -> py::list {
                py::list lst;
                for (const auto& pt : self.corner_uv) {
                    lst.append(pt);
                }
                return lst;
            },
            [](GridCell& self, py::sequence seq) {
                if (py::len(seq) != 4) {
                    throw std::runtime_error("corner_uv requires exactly 4 Point2f elements");
                }
                for (size_t k = 0; k < 4; ++k) {
                    self.corner_uv[k] = seq[k].cast<cv::Point2f>();
                }
            }
        )
        .def_readwrite("center_uv", &GridCell::center_uv);

    py::class_<CheckerboardDetection>(m, "CheckerboardDetection")
        .def(py::init<>())
        .def_readwrite("corners", &CheckerboardDetection::corners)
        .def_readwrite("cells", &CheckerboardDetection::cells)
        .def_readwrite("cols", &CheckerboardDetection::cols)
        .def_readwrite("rows", &CheckerboardDetection::rows)
        .def_readwrite("tracking", &CheckerboardDetection::tracking)
        .def_readwrite("stable", &CheckerboardDetection::stable)
        .def("valid", &CheckerboardDetection::valid);

    py::class_<CheckerboardDetectorConfig>(m, "CheckerboardDetectorConfig")
        .def(py::init<>())
        .def_readwrite("min_corners", &CheckerboardDetectorConfig::min_corners)
        .def_readwrite("min_cells", &CheckerboardDetectorConfig::min_cells)
        .def_readwrite("min_tracking_corners", &CheckerboardDetectorConfig::min_tracking_corners)
        .def_readwrite("min_tracking_cells", &CheckerboardDetectorConfig::min_tracking_cells)
        .def_readwrite("min_tracking_decode_cell_span", &CheckerboardDetectorConfig::min_tracking_decode_cell_span)
        .def_readwrite("max_undecodeable_tracking_frames", &CheckerboardDetectorConfig::max_undecodeable_tracking_frames)
        .def_readwrite("min_tracking_corner_ratio", &CheckerboardDetectorConfig::min_tracking_corner_ratio)
        .def_readwrite("max_tracking_homography_error_px", &CheckerboardDetectorConfig::max_tracking_homography_error_px)
        .def_readwrite("refresh_interval_frames", &CheckerboardDetectorConfig::refresh_interval_frames)
        .def_readwrite("lk_win_size", &CheckerboardDetectorConfig::lk_win_size)
        .def_readwrite("lk_max_level", &CheckerboardDetectorConfig::lk_max_level)
        .def_readwrite("lk_max_iters", &CheckerboardDetectorConfig::lk_max_iters)
        .def_readwrite("lk_epsilon", &CheckerboardDetectorConfig::lk_epsilon)
        .def_readwrite("max_lk_error", &CheckerboardDetectorConfig::max_lk_error)
        .def_readwrite("max_lk_forward_backward_error_px", &CheckerboardDetectorConfig::max_lk_forward_backward_error_px)
        .def_readwrite("stable_motion_threshold_px", &CheckerboardDetectorConfig::stable_motion_threshold_px)
        .def_readwrite("det_width", &CheckerboardDetectorConfig::det_width)
        .def_readwrite("max_recovery_corners", &CheckerboardDetectorConfig::max_recovery_corners)
        .def_readwrite("use_tracking_roi_recovery", &CheckerboardDetectorConfig::use_tracking_roi_recovery)
        .def_readwrite("tracking_recovery_roi_margin_cells", &CheckerboardDetectorConfig::tracking_recovery_roi_margin_cells)
        .def_readwrite("tracking_recovery_roi_min_margin_px", &CheckerboardDetectorConfig::tracking_recovery_roi_min_margin_px)
        .def_readwrite("tracking_recovery_roi_max_area_ratio", &CheckerboardDetectorConfig::tracking_recovery_roi_max_area_ratio)
        .def_readwrite("tracking_recovery_align_fail_full_retry_frames", &CheckerboardDetectorConfig::tracking_recovery_align_fail_full_retry_frames)
        .def_readwrite("tracking_recovery_align_fail_roi_margin_multiplier", &CheckerboardDetectorConfig::tracking_recovery_align_fail_roi_margin_multiplier)
        .def_readwrite("tracking_recovery_roi_fail_retry_margin_multiplier", &CheckerboardDetectorConfig::tracking_recovery_roi_fail_retry_margin_multiplier)
        .def_readwrite("tracking_recovery_roi_fail_full_retry_frames", &CheckerboardDetectorConfig::tracking_recovery_roi_fail_full_retry_frames)
        .def_readwrite("tracking_recovery_full_build_interval_frames", &CheckerboardDetectorConfig::tracking_recovery_full_build_interval_frames)
        .def_readwrite("merge_radius_px", &CheckerboardDetectorConfig::merge_radius_px)
        .def_readwrite("duplicate_corner_dist_px", &CheckerboardDetectorConfig::duplicate_corner_dist_px)
        .def_readwrite("min_neighbor_dist_rel", &CheckerboardDetectorConfig::min_neighbor_dist_rel)
        .def_readwrite("max_neighbor_dist_rel", &CheckerboardDetectorConfig::max_neighbor_dist_rel)
        .def_readwrite("max_lattice_residual_rel", &CheckerboardDetectorConfig::max_lattice_residual_rel)
        .def_readwrite("outlier_residual_rel", &CheckerboardDetectorConfig::outlier_residual_rel)
        .def_readwrite("max_axis_seed_points", &CheckerboardDetectorConfig::max_axis_seed_points)
        .def_readwrite("checker_corner_half_px", &CheckerboardDetectorConfig::checker_corner_half_px)
        .def_readwrite("use_saddle_recovery", &CheckerboardDetectorConfig::use_saddle_recovery)
        .def_readwrite("saddle_radius", &CheckerboardDetectorConfig::saddle_radius)
        .def_readwrite("saddle_iterations", &CheckerboardDetectorConfig::saddle_iterations)
        .def_readwrite("saddle_sigma", &CheckerboardDetectorConfig::saddle_sigma)
        .def_readwrite("saddle_response_threshold", &CheckerboardDetectorConfig::saddle_response_threshold)
        .def_readwrite("saddle_max_angle_bias_deg", &CheckerboardDetectorConfig::saddle_max_angle_bias_deg)
        .def_readwrite("saddle_correlation_drop", &CheckerboardDetectorConfig::saddle_correlation_drop)
        .def_readwrite("quadrant_half_r", &CheckerboardDetectorConfig::quadrant_half_r)
        .def_readwrite("quadrant_min_contrast", &CheckerboardDetectorConfig::quadrant_min_contrast)
        .def_readwrite("quadrant_max_diagonal_diff", &CheckerboardDetectorConfig::quadrant_max_diagonal_diff)
        .def_readwrite("refresh_corner_loss_ratio", &CheckerboardDetectorConfig::refresh_corner_loss_ratio)
        .def_readwrite("refresh_gain_threshold", &CheckerboardDetectorConfig::refresh_gain_threshold)
        .def_readwrite("tracking_spacing_min_rel", &CheckerboardDetectorConfig::tracking_spacing_min_rel)
        .def_readwrite("tracking_spacing_max_rel", &CheckerboardDetectorConfig::tracking_spacing_max_rel)
        .def_readwrite("max_degraded_frames_before_reset", &CheckerboardDetectorConfig::max_degraded_frames_before_reset)
        .def_readwrite("max_missed_frames", &CheckerboardDetectorConfig::max_missed_frames)
        .def_readwrite("max_low_corner_frames", &CheckerboardDetectorConfig::max_low_corner_frames)
        .def_readwrite("visibility_sample_rel", &CheckerboardDetectorConfig::visibility_sample_rel)
        .def_readwrite("visibility_box_rel", &CheckerboardDetectorConfig::visibility_box_rel)
        .def_readwrite("visibility_evict_threshold", &CheckerboardDetectorConfig::visibility_evict_threshold)
        .def_readwrite("visibility_min_spacing", &CheckerboardDetectorConfig::visibility_min_spacing)
        .def_readwrite("visibility_smoothing_alpha", &CheckerboardDetectorConfig::visibility_smoothing_alpha)
        .def_readwrite("saddle_subpix_win_size", &CheckerboardDetectorConfig::saddle_subpix_win_size)
        .def_readwrite("saddle_subpix_max_iters", &CheckerboardDetectorConfig::saddle_subpix_max_iters)
        .def_readwrite("saddle_subpix_epsilon", &CheckerboardDetectorConfig::saddle_subpix_epsilon)
        .def_readwrite("recovery_correction_weight", &CheckerboardDetectorConfig::recovery_correction_weight)
        .def_readwrite("recovery_correction_max_dist_rel", &CheckerboardDetectorConfig::recovery_correction_max_dist_rel);

    py::class_<geom::CellGeometryValidationConfig>(m, "CellGeometryValidationConfig")
        .def(py::init<>())
        .def_readwrite("min_area_px2", &geom::CellGeometryValidationConfig::min_area_px2)
        .def_readwrite("max_opposite_edge_ratio", &geom::CellGeometryValidationConfig::max_opposite_edge_ratio)
        .def_readwrite("max_diagonal_ratio", &geom::CellGeometryValidationConfig::max_diagonal_ratio)
        .def_readwrite("min_angle_deg", &geom::CellGeometryValidationConfig::min_angle_deg)
        .def_readwrite("max_angle_deg", &geom::CellGeometryValidationConfig::max_angle_deg)
        .def_readwrite("max_opposite_edge_angle_diff_deg", &geom::CellGeometryValidationConfig::max_opposite_edge_angle_diff_deg);

    py::class_<geom::CellGeometryValidation>(m, "CellGeometryValidation")
        .def(py::init<>())
        .def_readwrite("valid", &geom::CellGeometryValidation::valid)
        .def_readwrite("finite", &geom::CellGeometryValidation::finite)
        .def_readwrite("indices_valid", &geom::CellGeometryValidation::indices_valid)
        .def_readwrite("area_valid", &geom::CellGeometryValidation::area_valid)
        .def_readwrite("convex", &geom::CellGeometryValidation::convex)
        .def_readwrite("center_inside", &geom::CellGeometryValidation::center_inside)
        .def_readwrite("opposite_edges_valid", &geom::CellGeometryValidation::opposite_edges_valid)
        .def_readwrite("diagonals_valid", &geom::CellGeometryValidation::diagonals_valid)
        .def_readwrite("angles_valid", &geom::CellGeometryValidation::angles_valid)
        .def_readwrite("opposite_edge_angles_valid", &geom::CellGeometryValidation::opposite_edge_angles_valid)
        .def_readwrite("signed_area", &geom::CellGeometryValidation::signed_area)
        .def_readwrite("area", &geom::CellGeometryValidation::area)
        .def_readwrite("edge_0", &geom::CellGeometryValidation::edge_0)
        .def_readwrite("edge_1", &geom::CellGeometryValidation::edge_1)
        .def_readwrite("edge_2", &geom::CellGeometryValidation::edge_2)
        .def_readwrite("edge_3", &geom::CellGeometryValidation::edge_3)
        .def_readwrite("opposite_edge_ratio_u", &geom::CellGeometryValidation::opposite_edge_ratio_u)
        .def_readwrite("opposite_edge_ratio_v", &geom::CellGeometryValidation::opposite_edge_ratio_v)
        .def_readwrite("diagonal_0", &geom::CellGeometryValidation::diagonal_0)
        .def_readwrite("diagonal_1", &geom::CellGeometryValidation::diagonal_1)
        .def_readwrite("diagonal_ratio", &geom::CellGeometryValidation::diagonal_ratio)
        .def_readwrite("min_angle_deg", &geom::CellGeometryValidation::min_angle_deg)
        .def_readwrite("max_angle_deg", &geom::CellGeometryValidation::max_angle_deg)
        .def_readwrite("opposite_edge_angle_diff_u_deg", &geom::CellGeometryValidation::opposite_edge_angle_diff_u_deg)
        .def_readwrite("opposite_edge_angle_diff_v_deg", &geom::CellGeometryValidation::opposite_edge_angle_diff_v_deg);

    py::class_<geom::PatchGeometryValidationConfig>(m, "PatchGeometryValidationConfig")
        .def(py::init<>())
        .def_readwrite("cell_config", &geom::PatchGeometryValidationConfig::cell_config)
        .def_readwrite("max_rel_area_std", &geom::PatchGeometryValidationConfig::max_rel_area_std)
        .def_readwrite("max_rel_edge_std", &geom::PatchGeometryValidationConfig::max_rel_edge_std)
        .def_readwrite("min_quality", &geom::PatchGeometryValidationConfig::min_quality);

    py::class_<geom::PatchGeometryValidation>(m, "PatchGeometryValidation")
        .def(py::init<>())
        .def_readwrite("valid", &geom::PatchGeometryValidation::valid)
        .def_readwrite("num_cells", &geom::PatchGeometryValidation::num_cells)
        .def_readwrite("num_valid_cells", &geom::PatchGeometryValidation::num_valid_cells)
        .def_readwrite("mean_cell_area", &geom::PatchGeometryValidation::mean_cell_area)
        .def_readwrite("rel_area_std", &geom::PatchGeometryValidation::rel_area_std)
        .def_readwrite("mean_edge_length", &geom::PatchGeometryValidation::mean_edge_length)
        .def_readwrite("rel_edge_std", &geom::PatchGeometryValidation::rel_edge_std)
        .def_readwrite("min_cell_angle_deg", &geom::PatchGeometryValidation::min_cell_angle_deg)
        .def_readwrite("max_cell_angle_deg", &geom::PatchGeometryValidation::max_cell_angle_deg)
        .def_readwrite("max_opposite_edge_ratio", &geom::PatchGeometryValidation::max_opposite_edge_ratio)
        .def_readwrite("max_diagonal_ratio", &geom::PatchGeometryValidation::max_diagonal_ratio)
        .def_readwrite("max_opposite_edge_angle_diff_deg", &geom::PatchGeometryValidation::max_opposite_edge_angle_diff_deg)
        .def_readwrite("quality", &geom::PatchGeometryValidation::quality);

    py::class_<DotCellObservation>(m, "DotCellObservation")
        .def(py::init<>())
        .def_readwrite("row", &DotCellObservation::row)
        .def_readwrite("col", &DotCellObservation::col)
        .def_readwrite("valid", &DotCellObservation::valid)
        .def_readwrite("has_dot", &DotCellObservation::has_dot)
        .def_readwrite("ambiguous", &DotCellObservation::ambiguous)
        .def_readwrite("score", &DotCellObservation::score)
        .def_readwrite("raw_score", &DotCellObservation::raw_score)
        .def_readwrite("center_mean", &DotCellObservation::center_mean)
        .def_readwrite("ring_mean", &DotCellObservation::ring_mean)
        .def_readwrite("local_mean", &DotCellObservation::local_mean)
        .def_readwrite("local_std", &DotCellObservation::local_std)
        .def_readwrite("polarity", &DotCellObservation::polarity)
        .def_readwrite("cache_reused", &DotCellObservation::cache_reused)
        .def_readwrite("center_uv", &DotCellObservation::center_uv)
        .def_readwrite("corners_uv", &DotCellObservation::corners_uv);

    py::class_<DotDetectionResult>(m, "DotDetectionResult")
        .def(py::init<>())
        .def_readwrite("rows", &DotDetectionResult::rows)
        .def_readwrite("cols", &DotDetectionResult::cols)
        .def_readwrite("cells", &DotDetectionResult::cells);

    py::class_<DotDetectorConfig>(m, "DotDetectorConfig")
        .def(py::init<>())
        .def_readwrite("canonical_size", &DotDetectorConfig::canonical_size)
        .def_readwrite("canonical_margin_px", &DotDetectorConfig::canonical_margin_px)
        .def_readwrite("min_dot_contrast", &DotDetectorConfig::min_dot_contrast)
        .def_readwrite("strong_dot_contrast", &DotDetectorConfig::strong_dot_contrast)
        .def_readwrite("commit_threshold", &DotDetectorConfig::commit_threshold)
        .def_readwrite("revoke_threshold", &DotDetectorConfig::revoke_threshold)
        .def_readwrite("uncertainty_low", &DotDetectorConfig::uncertainty_low)
        .def_readwrite("uncertainty_high", &DotDetectorConfig::uncertainty_high)
        .def_readwrite("warmup_frames", &DotDetectorConfig::warmup_frames)
        .def_readwrite("temporal_alpha", &DotDetectorConfig::temporal_alpha)
        .def_readwrite("commit_frames", &DotDetectorConfig::commit_frames)
        .def_readwrite("revoke_frames", &DotDetectorConfig::revoke_frames)
        .def_readwrite("use_temporal_smoothing", &DotDetectorConfig::use_temporal_smoothing)
        .def_readwrite("use_cell_value_cache", &DotDetectorConfig::use_cell_value_cache)
        .def_readwrite("cell_cache_max_age_frames", &DotDetectorConfig::cell_cache_max_age_frames)
        .def_readwrite("cell_cache_max_corner_motion_px", &DotDetectorConfig::cell_cache_max_corner_motion_px);

    py::class_<RefinedCorner>(m, "RefinedCorner")
        .def(py::init<>())
        .def_readwrite("uv", &RefinedCorner::uv)
        .def_readwrite("ledge_angles_deg", &RefinedCorner::ledge_angles_deg)
        .def_readwrite("correlation", &RefinedCorner::correlation)
        .def_readwrite("angle_bias_deg", &RefinedCorner::angle_bias_deg)
        .def_readwrite("valid", &RefinedCorner::valid);

    py::class_<LatticePoint>(m, "LatticePoint")
        .def(py::init<>())
        .def_readwrite("uv", &LatticePoint::uv)
        .def_readwrite("ij", &LatticePoint::ij)
        .def_readwrite("residual", &LatticePoint::residual)
        .def_readwrite("valid", &LatticePoint::valid);

    py::class_<LatticeResult>(m, "LatticeResult")
        .def(py::init<>())
        .def_readwrite("points", &LatticeResult::points)
        .def_readwrite("axis_u", &LatticeResult::axis_u)
        .def_readwrite("axis_v", &LatticeResult::axis_v)
        .def_readwrite("origin", &LatticeResult::origin)
        .def_readwrite("spacing_u", &LatticeResult::spacing_u)
        .def_readwrite("spacing_v", &LatticeResult::spacing_v)
        .def_readwrite("valid", &LatticeResult::valid);

    py::class_<CheckerboardRecoveryDebug>(m, "CheckerboardRecoveryDebug")
        .def(py::init<>())
        .def_readwrite("raw_candidates", &CheckerboardRecoveryDebug::raw_candidates)
        .def_readwrite("refined_corners", &CheckerboardRecoveryDebug::refined_corners)
        .def_readwrite("valid_refined_points", &CheckerboardRecoveryDebug::valid_refined_points)
        .def_readwrite("lattice", &CheckerboardRecoveryDebug::lattice)
        .def_readwrite("detection", &CheckerboardRecoveryDebug::detection)
        .def_readwrite("has_lattice", &CheckerboardRecoveryDebug::has_lattice)
        .def_readwrite("has_detection", &CheckerboardRecoveryDebug::has_detection)
        .def_readwrite("scale", &CheckerboardRecoveryDebug::scale);

    py::class_<CheckerboardDetector>(m, "CheckerboardDetector")
        .def(py::init<>())
        .def(py::init<CheckerboardDetectorConfig>())
        .def(
            "detect",
            [](CheckerboardDetector& self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img)
                -> std::optional<CheckerboardDetection>
            {
                cv::Mat mat = numpyToMat(img);
                return self.detect(mat);
            }
        )
        .def(
            "debug_recovery_stages",
            [](const CheckerboardDetector& self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img)
                -> CheckerboardRecoveryDebug
            {
                cv::Mat mat = numpyToMat(img);
                return self.debugRecoveryStages(mat);
            }
        )
        .def("last_timings_ms", &CheckerboardDetector::lastTimingsMs)
        .def("reset_tracking", &CheckerboardDetector::resetTracking)
        .def("is_tracking", &CheckerboardDetector::isTracking);

    py::class_<DotDetector>(m, "DotDetector")
        .def(py::init<>())
        .def(py::init<DotDetectorConfig>())
        .def(
            "detect",
            [](DotDetector& self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img,
               const CheckerboardDetection& checkerboard)
                -> DotDetectionResult
            {
                cv::Mat mat = numpyToMat(img);
                return self.detect(mat, checkerboard);
            }
        )
        .def("reset", &DotDetector::reset)
        .def("reset_smoothing", &DotDetector::reset_smoothing);

    py::class_<LocalPatch>(m, "LocalPatch")
        .def_readonly("row", &LocalPatch::row)
        .def_readonly("col", &LocalPatch::col)
        .def_readonly("k", &LocalPatch::k)
        .def_readonly("bits", &LocalPatch::bits)
        .def_readonly("scores", &LocalPatch::scores)
        .def_readonly("mean_score", &LocalPatch::mean_score)
        .def_readonly("geometry_valid", &LocalPatch::geometry_valid)
        .def_readonly("geometry_quality", &LocalPatch::geometry_quality)
        .def_readonly("geometry", &LocalPatch::geometry)
        .def_readonly("valid", &LocalPatch::valid);

    py::class_<PatchExtractor>(m, "PatchExtractor")
        .def(py::init<>())
        .def("extract", &PatchExtractor::extract);

    py::class_<PatchDecoderConfig>(m, "PatchDecoderConfig")
        .def(py::init<>())
        .def_readwrite("require_geometry_valid", &PatchDecoderConfig::require_geometry_valid)
        .def_readwrite("accept_ambiguous", &PatchDecoderConfig::accept_ambiguous);

    py::class_<DecodedPatch>(m, "DecodedPatch")
        .def(py::init<>())
        .def_readonly("local", &DecodedPatch::local)
        .def_readonly("valid", &DecodedPatch::valid)
        .def_readonly("ambiguous", &DecodedPatch::ambiguous)
        .def_readonly("global_row", &DecodedPatch::global_row)
        .def_readonly("global_col", &DecodedPatch::global_col)
        .def_readonly("rotation_deg", &DecodedPatch::rotation_deg)
        .def_readonly("num_matches", &DecodedPatch::num_matches)
        .def_readonly("confidence", &DecodedPatch::confidence);

    py::class_<PatchDecoder>(m, "PatchDecoder")
        .def(py::init<>())
        .def(py::init<PatchDecoderConfig>())
        .def("decode_one", &PatchDecoder::decodeOne)
        .def("decode", &PatchDecoder::decode);

    py::class_<Correspondence2D3D>(m, "Correspondence2D3D")
        .def(py::init<>())
        .def_readonly("uv", &Correspondence2D3D::uv)
        .def_readonly("xyz_mm", &Correspondence2D3D::xyz_mm)
        .def_readonly("local_row", &Correspondence2D3D::local_row)
        .def_readonly("local_col", &Correspondence2D3D::local_col)
        .def_readonly("global_row", &Correspondence2D3D::global_row)
        .def_readonly("global_col", &Correspondence2D3D::global_col)
        .def_readonly("votes", &Correspondence2D3D::votes);

    py::class_<CorrespondenceBuilderConfig>(m, "CorrespondenceBuilderConfig")
        .def(py::init<>())
        .def_readwrite("min_votes", &CorrespondenceBuilderConfig::min_votes)
        .def_readwrite("discard_conflicts", &CorrespondenceBuilderConfig::discard_conflicts)
        .def_readwrite("require_detection_stable", &CorrespondenceBuilderConfig::require_detection_stable)
        .def_readwrite("enable_dominant_rotation_filter", &CorrespondenceBuilderConfig::enable_dominant_rotation_filter)
        .def_readwrite("min_rotation_support", &CorrespondenceBuilderConfig::min_rotation_support)
        .def_readwrite("min_rotation_support_ratio", &CorrespondenceBuilderConfig::min_rotation_support_ratio)
        .def_readwrite("allow_single_vote_boundary_corners", &CorrespondenceBuilderConfig::allow_single_vote_boundary_corners)
        .def_readwrite("boundary_margin_cells", &CorrespondenceBuilderConfig::boundary_margin_cells);

    py::class_<CorrespondenceBuildResult>(m, "CorrespondenceBuildResult")
        .def(py::init<>())
        .def_readonly("correspondences", &CorrespondenceBuildResult::correspondences)
        .def_readonly("decoded_patches_used", &CorrespondenceBuildResult::decoded_patches_used)
        .def_readonly("decoded_patches_rejected_by_rotation", &CorrespondenceBuildResult::decoded_patches_rejected_by_rotation)
        .def_readonly("assignments_total", &CorrespondenceBuildResult::assignments_total)
        .def_readonly("assignments_accepted", &CorrespondenceBuildResult::assignments_accepted)
        .def_readonly("assignments_conflicted", &CorrespondenceBuildResult::assignments_conflicted)
        .def_readonly("corners_without_geometry", &CorrespondenceBuildResult::corners_without_geometry)
        .def_readonly("single_vote_boundary_corners_accepted", &CorrespondenceBuildResult::single_vote_boundary_corners_accepted)
        .def_readonly("single_vote_non_boundary_corners_rejected", &CorrespondenceBuildResult::single_vote_non_boundary_corners_rejected)
        .def_readonly("dominant_rotation_deg", &CorrespondenceBuildResult::dominant_rotation_deg)
        .def_readonly("dominant_rotation_count", &CorrespondenceBuildResult::dominant_rotation_count)
        .def_readonly("rotation_vote_count", &CorrespondenceBuildResult::rotation_vote_count)
        .def("valid", &CorrespondenceBuildResult::valid);

    py::class_<CorrespondenceBuilder>(m, "CorrespondenceBuilder")
        .def(py::init<>())
        .def(py::init<CorrespondenceBuilderConfig>())
        .def("build", &CorrespondenceBuilder::build);
}

} // namespace hydramarker
