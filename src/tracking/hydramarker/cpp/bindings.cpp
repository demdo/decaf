// bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <optional>
#include <memory>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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
#include "tracker_config.hpp"
#include "tracker_engine.hpp"
#include "tracker_geometry.hpp"
#include "tracker_persistence.hpp"
#include "tracker_pose.hpp"
#include "tracker_types.hpp"

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

cv::Matx33d numpyToMatx33d(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr
) {
    py::buffer_info info = arr.request();

    if (info.size != 9) {
        throw std::runtime_error("Camera matrix K must contain exactly 9 values");
    }

    const double* data = static_cast<const double*>(info.ptr);
    return cv::Matx33d(
        data[0], data[1], data[2],
        data[3], data[4], data[5],
        data[6], data[7], data[8]
    );
}

std::vector<double> optionalNumpyToVectorDouble(py::object obj)
{
    if (obj.is_none()) {
        return {};
    }

    auto arr = py::cast<
        py::array_t<double, py::array::c_style | py::array::forcecast>
    >(obj);
    py::buffer_info info = arr.request();

    const double* data = static_cast<const double*>(info.ptr);
    return std::vector<double>(data, data + info.size);
}

std::vector<cv::Point3d> numpyToPoint3dVector(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr
) {
    py::buffer_info info = arr.request();
    if (info.size % 3 != 0) {
        throw std::runtime_error("object_points size must be divisible by 3");
    }

    const double* data = static_cast<const double*>(info.ptr);
    std::vector<cv::Point3d> points;
    const size_t size = static_cast<size_t>(info.size);
    points.reserve(size / 3);
    for (size_t i = 0; i < size; i += 3) {
        points.emplace_back(data[i], data[i + 1], data[i + 2]);
    }
    return points;
}

std::vector<cv::Point2d> numpyToPoint2dVector(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr
) {
    py::buffer_info info = arr.request();
    if (info.size % 2 != 0) {
        throw std::runtime_error("image_points size must be divisible by 2");
    }

    const double* data = static_cast<const double*>(info.ptr);
    std::vector<cv::Point2d> points;
    const size_t size = static_cast<size_t>(info.size);
    points.reserve(size / 2);
    for (size_t i = 0; i < size; i += 2) {
        points.emplace_back(data[i], data[i + 1]);
    }
    return points;
}

std::vector<cv::Point2f> numpyToPoint2fVector(
    py::array_t<double, py::array::c_style | py::array::forcecast> arr
) {
    py::buffer_info info = arr.request();
    if (info.size % 2 != 0) {
        throw std::runtime_error("points size must be divisible by 2");
    }

    const double* data = static_cast<const double*>(info.ptr);
    std::vector<cv::Point2f> points;
    const size_t size = static_cast<size_t>(info.size);
    points.reserve(size / 2);
    for (size_t i = 0; i < size; i += 2) {
        points.emplace_back(
            static_cast<float>(data[i]),
            static_cast<float>(data[i + 1])
        );
    }
    return points;
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

    m.def(
        "refine_saddle_points",
        [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img,
           py::array_t<double, py::array::c_style | py::array::forcecast> points,
           int radius,
           int iterations,
           float max_angle_bias_deg,
           float correlation_drop,
           float merge_radius_px,
           int quadrant_half_r,
           float quadrant_min_contrast,
           float quadrant_max_diagonal_diff,
           int subpix_win_size,
           int subpix_max_iters,
           double subpix_epsilon) -> py::array_t<double>
        {
            cv::Mat mat = numpyToMat(img);
            cv::Mat gray;

            if (mat.channels() == 1) {
                gray = mat;
            } else if (mat.channels() == 3) {
                cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
            } else if (mat.channels() == 4) {
                cv::cvtColor(mat, gray, cv::COLOR_BGRA2GRAY);
            } else {
                throw std::runtime_error("Image must have 1, 3, or 4 channels");
            }

            std::vector<cv::Point2f> candidates = numpyToPoint2fVector(points);

            cv::Mat gray_f;
            gray.convertTo(gray_f, CV_32F);

            cv::Mat grad_x;
            cv::Mat grad_y;
            cv::Sobel(gray_f, grad_x, CV_32F, 1, 0, 3);
            cv::Sobel(gray_f, grad_y, CV_32F, 0, 1, 3);

            CornerRefinementConfig config;
            config.radius = radius;
            config.iterations = iterations;
            config.max_angle_bias_deg = max_angle_bias_deg;
            config.correlation_drop = correlation_drop;
            config.merge_radius_px = merge_radius_px;
            config.quadrant_half_r = quadrant_half_r;
            config.quadrant_min_contrast = quadrant_min_contrast;
            config.quadrant_max_diagonal_diff = quadrant_max_diagonal_diff;
            config.subpix_win_size = subpix_win_size;
            config.subpix_max_iters = subpix_max_iters;
            config.subpix_epsilon = subpix_epsilon;

            CornerRefiner refiner;
            std::vector<RefinedCorner> refined = refiner.refine(
                gray,
                candidates,
                grad_x,
                grad_y,
                config
            );

            py::array_t<double> out({
                static_cast<py::ssize_t>(refined.size()),
                static_cast<py::ssize_t>(5)
            });
            py::buffer_info info = out.request();
            double* data = static_cast<double*>(info.ptr);

            for (size_t i = 0; i < refined.size(); ++i) {
                const size_t j = i * 5;
                data[j + 0] = static_cast<double>(refined[i].uv.x);
                data[j + 1] = static_cast<double>(refined[i].uv.y);
                data[j + 2] = static_cast<double>(refined[i].correlation);
                data[j + 3] = static_cast<double>(refined[i].angle_bias_deg);
                data[j + 4] = refined[i].valid ? 1.0 : 0.0;
            }

            return out;
        },
        py::arg("img"),
        py::arg("points"),
        py::arg("radius") = 5,
        py::arg("iterations") = 2,
        py::arg("max_angle_bias_deg") = 20.0f,
        py::arg("correlation_drop") = 0.2f,
        py::arg("merge_radius_px") = 2.0f,
        py::arg("quadrant_half_r") = 3,
        py::arg("quadrant_min_contrast") = 12.0f,
        py::arg("quadrant_max_diagonal_diff") = 60.0f,
        py::arg("subpix_win_size") = -1,
        py::arg("subpix_max_iters") = 20,
        py::arg("subpix_epsilon") = 0.05
    );

    py::enum_<TrackerMode>(m, "TrackerMode")
        .value("LOST", TrackerMode::Lost)
        .value("DETECTING", TrackerMode::Detecting)
        .value("TRACKING", TrackerMode::Tracking)
        .value("RECOVERING", TrackerMode::Recovering);

    py::enum_<PoseSource>(m, "PoseSource")
        .value("NONE", PoseSource::None)
        .value("DECODE", PoseSource::Decode)
        .value("PERSISTENT", PoseSource::Persistent)
        .value("FAST_PERSISTENT", PoseSource::FastPersistent)
        .value("UNCODED_GRID", PoseSource::UncodedGrid)
        .value("HOLD", PoseSource::Hold);

    py::class_<PoseTrackPoint>(m, "PoseTrackPoint")
        .def(py::init<>())
        .def_readwrite("global_row", &PoseTrackPoint::global_row)
        .def_readwrite("global_col", &PoseTrackPoint::global_col)
        .def_readwrite("xyz_mm", &PoseTrackPoint::xyz_mm)
        .def_readwrite("uv", &PoseTrackPoint::uv)
        .def_readwrite("votes", &PoseTrackPoint::votes);

    py::class_<MapPoseResult>(m, "MapPoseResult")
        .def(py::init<>())
        .def_readwrite("success", &MapPoseResult::success)
        .def_readwrite("message", &MapPoseResult::message)
        .def_readwrite("rvec", &MapPoseResult::rvec)
        .def_readwrite("tvec", &MapPoseResult::tvec)
        .def_readwrite("T_marker_camera", &MapPoseResult::T_marker_camera)
        .def_readwrite("inlier_indices", &MapPoseResult::inlier_indices)
        .def_readwrite("reprojection_mean_px", &MapPoseResult::reprojection_mean_px)
        .def_readwrite("reprojection_max_px", &MapPoseResult::reprojection_max_px)
        .def_readwrite("num_points", &MapPoseResult::num_points)
        .def_readwrite("num_inliers", &MapPoseResult::num_inliers)
        .def_readwrite("points", &MapPoseResult::points)
        .def_readwrite("method", &MapPoseResult::method);

    py::class_<MapPoseTrackerConfig>(m, "MapPoseTrackerConfig")
        .def(py::init<>())
        .def_readwrite("min_points", &MapPoseTrackerConfig::min_points)
        .def_readwrite("min_inliers", &MapPoseTrackerConfig::min_inliers)
        .def_readwrite("ransac_reproj_px", &MapPoseTrackerConfig::ransac_reproj_px)
        .def_readwrite("ransac_confidence", &MapPoseTrackerConfig::ransac_confidence)
        .def_readwrite("ransac_iterations", &MapPoseTrackerConfig::ransac_iterations)
        .def_readwrite("max_mean_reproj_px", &MapPoseTrackerConfig::max_mean_reproj_px)
        .def_readwrite("max_max_reproj_px", &MapPoseTrackerConfig::max_max_reproj_px)
        .def_readwrite("max_translation_jump_mm", &MapPoseTrackerConfig::max_translation_jump_mm)
        .def_readwrite("max_rotation_jump_deg", &MapPoseTrackerConfig::max_rotation_jump_deg)
        .def_readwrite("rotation_gate_scale_per_lost_frame", &MapPoseTrackerConfig::rotation_gate_scale_per_lost_frame)
        .def_readwrite("rotation_gate_max_deg", &MapPoseTrackerConfig::rotation_gate_max_deg)
        .def_readwrite("use_pose_prior", &MapPoseTrackerConfig::use_pose_prior)
        .def_readwrite("refine_with_iterative", &MapPoseTrackerConfig::refine_with_iterative)
        .def_readwrite("use_direct_prior_solver", &MapPoseTrackerConfig::use_direct_prior_solver)
        .def_readwrite("direct_refine_method", &MapPoseTrackerConfig::direct_refine_method)
        .def_readwrite("direct_max_mean_reproj_px", &MapPoseTrackerConfig::direct_max_mean_reproj_px)
        .def_readwrite("direct_max_max_reproj_px", &MapPoseTrackerConfig::direct_max_max_reproj_px);

    py::class_<MapPoseTracker>(m, "MapPoseTracker")
        .def(
            py::init([](
                py::array_t<double, py::array::c_style | py::array::forcecast> K,
                py::object dist_coeffs,
                const MapPoseTrackerConfig& config
            ) {
                return std::make_unique<MapPoseTracker>(
                    numpyToMatx33d(K),
                    optionalNumpyToVectorDouble(dist_coeffs),
                    config
                );
            }),
            py::arg("K"),
            py::arg("dist_coeffs") = py::none(),
            py::arg("config") = MapPoseTrackerConfig()
        )
        .def(
            "estimate_pose",
            &MapPoseTracker::estimatePose,
            py::arg("points"),
            py::arg("lost_frames") = 0
        )
        .def("reset", &MapPoseTracker::reset)
        .def(
            "set_pose",
            &MapPoseTracker::setPose,
            py::arg("rvec"),
            py::arg("tvec")
        )
        .def("has_pose", &MapPoseTracker::hasPose)
        .def_property_readonly("rvec", &MapPoseTracker::rvec)
        .def_property_readonly("tvec", &MapPoseTracker::tvec)
        .def_property_readonly("T_marker_camera", &MapPoseTracker::TMarkerCamera)
        .def_property_readonly(
            "config",
            [](const MapPoseTracker& self) { return self.config(); },
            py::return_value_policy::copy
        );

    py::class_<GlobalCornerIdentity>(m, "GlobalCornerIdentity")
        .def(py::init<>())
        .def_readwrite("global_row", &GlobalCornerIdentity::global_row)
        .def_readwrite("global_col", &GlobalCornerIdentity::global_col)
        .def_readwrite("xyz_mm", &GlobalCornerIdentity::xyz_mm)
        .def_readwrite("uv", &GlobalCornerIdentity::uv)
        .def_readwrite("votes", &GlobalCornerIdentity::votes);

    py::class_<TrackerCorner>(m, "TrackerCorner")
        .def(py::init<>())
        .def_readwrite("local_row", &TrackerCorner::local_row)
        .def_readwrite("local_col", &TrackerCorner::local_col)
        .def_readwrite("global_row", &TrackerCorner::global_row)
        .def_readwrite("global_col", &TrackerCorner::global_col)
        .def_readwrite("xyz_mm", &TrackerCorner::xyz_mm)
        .def_readwrite("uv", &TrackerCorner::uv)
        .def_readwrite("votes", &TrackerCorner::votes)
        .def_readwrite("visibility_score", &TrackerCorner::visibility_score)
        .def_readwrite("observed_frames", &TrackerCorner::observed_frames)
        .def_readwrite("predicted", &TrackerCorner::predicted);

    py::class_<FrameDetectedCorner>(m, "DetectedCorner")
        .def(py::init<>())
        .def_readwrite("local_row", &FrameDetectedCorner::local_row)
        .def_readwrite("local_col", &FrameDetectedCorner::local_col)
        .def_readwrite("uv", &FrameDetectedCorner::uv)
        .def_readwrite("visibility_score", &FrameDetectedCorner::visibility_score)
        .def_readwrite("observed_frames", &FrameDetectedCorner::observed_frames)
        .def_readwrite("predicted", &FrameDetectedCorner::predicted);

    py::class_<PersistentMatchStats>(m, "PersistentMatchStats")
        .def(py::init<>())
        .def_readwrite("age", &PersistentMatchStats::age)
        .def_readwrite("identities", &PersistentMatchStats::identities)
        .def_readwrite("current_corners", &PersistentMatchStats::current_corners)
        .def_readwrite("accepted", &PersistentMatchStats::accepted)
        .def_readwrite("used_pose_projection", &PersistentMatchStats::used_pose_projection)
        .def_readwrite("adaptive_motion_px", &PersistentMatchStats::adaptive_motion_px)
        .def_readwrite("adaptive_max_dist_px", &PersistentMatchStats::adaptive_max_dist_px)
        .def_readwrite("rejected_no_projection", &PersistentMatchStats::rejected_no_projection)
        .def_readwrite("rejected_far", &PersistentMatchStats::rejected_far)
        .def_readwrite("rejected_ambiguous", &PersistentMatchStats::rejected_ambiguous)
        .def_readwrite("rejected_claimed", &PersistentMatchStats::rejected_claimed);

    py::class_<PersistentMatchResult>(m, "PersistentMatchResult")
        .def(py::init<>())
        .def_readwrite("points", &PersistentMatchResult::points)
        .def_readwrite("corners", &PersistentMatchResult::corners)
        .def_readwrite("stats", &PersistentMatchResult::stats)
        .def_readwrite("message", &PersistentMatchResult::message)
        .def("valid", &PersistentMatchResult::valid);

    py::class_<PersistentPoseSeedResult>(m, "PersistentPoseSeedResult")
        .def(py::init<>())
        .def_readwrite("points", &PersistentPoseSeedResult::points)
        .def_readwrite("corners", &PersistentPoseSeedResult::corners)
        .def_readwrite("stats", &PersistentPoseSeedResult::stats)
        .def_readwrite("pose", &PersistentPoseSeedResult::pose)
        .def_readwrite("message", &PersistentPoseSeedResult::message)
        .def_readwrite("match_ms", &PersistentPoseSeedResult::match_ms)
        .def_readwrite("pose_ms", &PersistentPoseSeedResult::pose_ms)
        .def_readwrite("total_ms", &PersistentPoseSeedResult::total_ms)
        .def("valid_match", &PersistentPoseSeedResult::validMatch);

    py::class_<FastDenseGateMetrics>(m, "FastDenseGateMetrics")
        .def(py::init<>())
        .def_readwrite("match_ratio", &FastDenseGateMetrics::match_ratio)
        .def_readwrite("motion_px", &FastDenseGateMetrics::motion_px)
        .def_readwrite("ambiguous_count", &FastDenseGateMetrics::ambiguous_count)
        .def_readwrite("seed_mean_px", &FastDenseGateMetrics::seed_mean_px)
        .def_readwrite("seed_max_px", &FastDenseGateMetrics::seed_max_px);

    py::class_<FastDenseProjectionStats>(m, "FastDenseProjectionStats")
        .def(py::init<>())
        .def_readwrite("detected", &FastDenseProjectionStats::detected)
        .def_readwrite("projected", &FastDenseProjectionStats::projected)
        .def_readwrite("rejected_no_projection", &FastDenseProjectionStats::rejected_no_projection)
        .def_readwrite("rejected_far", &FastDenseProjectionStats::rejected_far)
        .def_readwrite("rejected_ambiguous", &FastDenseProjectionStats::rejected_ambiguous)
        .def_readwrite("rejected_non_mutual", &FastDenseProjectionStats::rejected_non_mutual)
        .def_readwrite("median_error_px", &FastDenseProjectionStats::median_error_px)
        .def_readwrite("p90_error_px", &FastDenseProjectionStats::p90_error_px)
        .def_readwrite("image_coverage", &FastDenseProjectionStats::image_coverage)
        .def_readwrite("image_span_u_px", &FastDenseProjectionStats::image_span_u_px)
        .def_readwrite("image_span_v_px", &FastDenseProjectionStats::image_span_v_px)
        .def_readwrite("object_span_mm", &FastDenseProjectionStats::object_span_mm)
        .def_readwrite("distinct_rows", &FastDenseProjectionStats::distinct_rows)
        .def_readwrite("distinct_cols", &FastDenseProjectionStats::distinct_cols);

    py::class_<FastAcceptedState>(m, "FastAcceptedState")
        .def(py::init<>())
        .def_readwrite("evaluated", &FastAcceptedState::evaluated)
        .def_readwrite("reliable_pose", &FastAcceptedState::reliable_pose)
        .def_readwrite("rvec", &FastAcceptedState::rvec)
        .def_readwrite("tvec", &FastAcceptedState::tvec)
        .def_readwrite("T_marker_camera", &FastAcceptedState::T_marker_camera)
        .def_readwrite("last_good_reproj_px", &FastAcceptedState::last_good_reproj_px)
        .def_readwrite("max_pts_seen", &FastAcceptedState::max_pts_seen)
        .def_readwrite("accepted_pose_frame", &FastAcceptedState::accepted_pose_frame)
        .def_readwrite("visual_corner_count", &FastAcceptedState::visual_corner_count);

    py::class_<FastPoseResult>(m, "FastPoseResult")
        .def(py::init<>())
        .def_readwrite("attempted", &FastPoseResult::attempted)
        .def_readwrite("success", &FastPoseResult::success)
        .def_readwrite("route_decode", &FastPoseResult::route_decode)
        .def_readwrite("used_dense", &FastPoseResult::used_dense)
        .def_readwrite("dense_required", &FastPoseResult::dense_required)
        .def_readwrite("dense_attempted", &FastPoseResult::dense_attempted)
        .def_readwrite("dense_success", &FastPoseResult::dense_success)
        .def_readwrite("rescue_required", &FastPoseResult::rescue_required)
        .def_readwrite("reason", &FastPoseResult::reason)
        .def_readwrite("dense_reason", &FastPoseResult::dense_reason)
        .def_readwrite("dense_gate_reason", &FastPoseResult::dense_gate_reason)
        .def_readwrite("seed_error_reason", &FastPoseResult::seed_error_reason)
        .def_readwrite("points", &FastPoseResult::points)
        .def_readwrite("corners", &FastPoseResult::corners)
        .def_readwrite("visual_corners", &FastPoseResult::visual_corners)
        .def_readwrite("stats", &FastPoseResult::stats)
        .def_readwrite("dense_stats", &FastPoseResult::dense_stats)
        .def_readwrite("dense_gate_metrics", &FastPoseResult::dense_gate_metrics)
        .def_readwrite("seed_pose", &FastPoseResult::seed_pose)
        .def_readwrite("pose", &FastPoseResult::pose)
        .def_readwrite("persistent_match_ms", &FastPoseResult::persistent_match_ms)
        .def_readwrite("seed_pnp_ms", &FastPoseResult::seed_pnp_ms)
        .def_readwrite("cpp_seed_total_ms", &FastPoseResult::cpp_seed_total_ms)
        .def_readwrite("dense_match_ms", &FastPoseResult::dense_match_ms)
        .def_readwrite("dense_pose_ms", &FastPoseResult::dense_pose_ms)
        .def_readwrite("persistence_refresh_ms", &FastPoseResult::persistence_refresh_ms)
        .def_readwrite("total_ms", &FastPoseResult::total_ms)
        .def_readwrite("min_points", &FastPoseResult::min_points)
        .def_readwrite("min_dense_points", &FastPoseResult::min_dense_points)
        .def_readwrite("dense_matches", &FastPoseResult::dense_matches)
        .def_readwrite("accepted_state", &FastPoseResult::accepted_state)
        .def_readwrite("persistence_refresh_available", &FastPoseResult::persistence_refresh_available)
        .def_readwrite("persistence_refresh_frame", &FastPoseResult::persistence_refresh_frame)
        .def_readwrite("persistence_refresh_count", &FastPoseResult::persistence_refresh_count)
        .def_readwrite("persistence_refresh_identities", &FastPoseResult::persistence_refresh_identities);

    py::class_<DenseProjectionMatchStats>(m, "DenseProjectionMatchStats")
        .def(py::init<>())
        .def_readwrite("detected", &DenseProjectionMatchStats::detected)
        .def_readwrite("projected", &DenseProjectionMatchStats::projected)
        .def_readwrite("rejected_no_projection", &DenseProjectionMatchStats::rejected_no_projection)
        .def_readwrite("rejected_far", &DenseProjectionMatchStats::rejected_far)
        .def_readwrite("rejected_ambiguous", &DenseProjectionMatchStats::rejected_ambiguous)
        .def_readwrite("rejected_non_mutual", &DenseProjectionMatchStats::rejected_non_mutual)
        .def_readwrite("median_error_px", &DenseProjectionMatchStats::median_error_px)
        .def_readwrite("p90_error_px", &DenseProjectionMatchStats::p90_error_px)
        .def_readwrite("image_coverage", &DenseProjectionMatchStats::image_coverage)
        .def_readwrite("image_span_u_px", &DenseProjectionMatchStats::image_span_u_px)
        .def_readwrite("image_span_v_px", &DenseProjectionMatchStats::image_span_v_px)
        .def_readwrite("object_span_mm", &DenseProjectionMatchStats::object_span_mm)
        .def_readwrite("distinct_rows", &DenseProjectionMatchStats::distinct_rows)
        .def_readwrite("distinct_cols", &DenseProjectionMatchStats::distinct_cols);

    py::class_<DenseProjectionMatchResult>(m, "DenseProjectionMatchResult")
        .def(py::init<>())
        .def_readwrite("corners", &DenseProjectionMatchResult::corners)
        .def_readwrite("stats", &DenseProjectionMatchResult::stats)
        .def("valid", &DenseProjectionMatchResult::valid);

    py::class_<TrackerGeometry>(m, "TrackerGeometry")
        .def(
            py::init([](
                const MarkerGeometry& geometry,
                py::array_t<double, py::array::c_style | py::array::forcecast> K,
                py::object dist_coeffs,
                py::object config
            ) {
                TrackerConfig cpp_config;
                if (!config.is_none()) {
                    cpp_config = config.cast<TrackerConfig>();
                }
                return std::make_unique<TrackerGeometry>(
                    geometry,
                    numpyToMatx33d(K),
                    optionalNumpyToVectorDouble(dist_coeffs),
                    cpp_config
                );
            }),
            py::arg("geometry"),
            py::arg("K"),
            py::arg("dist_coeffs") = py::none(),
            py::arg("config") = py::none()
        )
        .def(
            "strict_projected_match",
            [](
                const TrackerGeometry& self,
                const CheckerboardDetection& detection,
                py::object rvec,
                py::object tvec,
                double max_dist_px,
                double ambiguity_margin_px
            ) {
                return self.strictProjectedMatch(
                    detection,
                    optionalNumpyToVectorDouble(rvec),
                    optionalNumpyToVectorDouble(tvec),
                    max_dist_px,
                    ambiguity_margin_px
                );
            },
            py::arg("detection"),
            py::arg("rvec"),
            py::arg("tvec"),
            py::arg("max_dist_px"),
            py::arg("ambiguity_margin_px")
        )
        .def(
            "greedy_projected_match",
            [](
                const TrackerGeometry& self,
                const CheckerboardDetection& detection,
                py::object rvec,
                py::object tvec,
                double max_dist_px
            ) {
                return self.greedyProjectedMatch(
                    detection,
                    optionalNumpyToVectorDouble(rvec),
                    optionalNumpyToVectorDouble(tvec),
                    max_dist_px
                );
            },
            py::arg("detection"),
            py::arg("rvec"),
            py::arg("tvec"),
            py::arg("max_dist_px")
        )
        .def(
            "visual_corners_from_pose",
            [](
                const TrackerGeometry& self,
                const std::vector<TrackerCorner>& corners,
                py::object rvec,
                py::object tvec,
                double max_error_px
            ) {
                return self.visualCornersFromPose(
                    corners,
                    optionalNumpyToVectorDouble(rvec),
                    optionalNumpyToVectorDouble(tvec),
                    max_error_px
                );
            },
            py::arg("corners"),
            py::arg("rvec"),
            py::arg("tvec"),
            py::arg("max_error_px")
        )
        .def(
            "estimate_dense_robust_pose",
            [](
                const TrackerGeometry& self,
                const std::vector<PoseTrackPoint>& points,
                const CheckerboardDetection& detection,
                py::object seed_rvec,
                py::object seed_tvec,
                py::object previous_rvec,
                py::object previous_tvec
            ) {
                return self.estimateDenseRobustPose(
                    points,
                    detection,
                    optionalNumpyToVectorDouble(seed_rvec),
                    optionalNumpyToVectorDouble(seed_tvec),
                    optionalNumpyToVectorDouble(previous_rvec),
                    optionalNumpyToVectorDouble(previous_tvec)
                );
            },
            py::arg("points"),
            py::arg("detection"),
            py::arg("seed_rvec") = py::none(),
            py::arg("seed_tvec") = py::none(),
            py::arg("previous_rvec") = py::none(),
            py::arg("previous_tvec") = py::none()
        )
        .def(
            "estimate_dense_direct_pose",
            [](
                const TrackerGeometry& self,
                const std::vector<PoseTrackPoint>& points,
                py::object seed_rvec,
                py::object seed_tvec,
                int lost_frames
            ) {
                return self.estimateDenseDirectPose(
                    points,
                    optionalNumpyToVectorDouble(seed_rvec),
                    optionalNumpyToVectorDouble(seed_tvec),
                    lost_frames
                );
            },
            py::arg("points"),
            py::arg("seed_rvec") = py::none(),
            py::arg("seed_tvec") = py::none(),
            py::arg("lost_frames") = 0
        );

    py::class_<TrackerConfig>(m, "TrackerConfig")
        .def(py::init<>())
        .def_readwrite("min_points", &TrackerConfig::min_points)
        .def_readwrite("min_inliers", &TrackerConfig::min_inliers)
        .def_readwrite("max_mean_reprojection_error_px", &TrackerConfig::max_mean_reprojection_error_px)
        .def_readwrite("max_max_reprojection_error_px", &TrackerConfig::max_max_reprojection_error_px)
        .def_readwrite("max_lost_frames", &TrackerConfig::max_lost_frames)
        .def_readwrite("max_translation_jump_mm", &TrackerConfig::max_translation_jump_mm)
        .def_readwrite("max_rotation_jump_deg", &TrackerConfig::max_rotation_jump_deg)
        .def_readwrite("rotation_gate_scale_per_lost_frame", &TrackerConfig::rotation_gate_scale_per_lost_frame)
        .def_readwrite("rotation_gate_max_deg", &TrackerConfig::rotation_gate_max_deg)
        .def_readwrite("pnp_ransac_iterations", &TrackerConfig::pnp_ransac_iterations)
        .def_readwrite("pnp_ransac_reprojection_px", &TrackerConfig::pnp_ransac_reprojection_px)
        .def_readwrite("pnp_ransac_confidence", &TrackerConfig::pnp_ransac_confidence)
        .def_readwrite("use_pose_prior", &TrackerConfig::use_pose_prior)
        .def_readwrite("pnp_direct_prior_enabled", &TrackerConfig::pnp_direct_prior_enabled)
        .def_readwrite("pnp_direct_refine_method", &TrackerConfig::pnp_direct_refine_method)
        .def_readwrite("pnp_direct_max_mean_reprojection_error_px", &TrackerConfig::pnp_direct_max_mean_reprojection_error_px)
        .def_readwrite("pnp_direct_max_max_reprojection_error_px", &TrackerConfig::pnp_direct_max_max_reprojection_error_px)
        .def_readwrite("checker_min_tracking_decode_cell_span", &TrackerConfig::checker_min_tracking_decode_cell_span)
        .def_readwrite("checker_refresh_interval_frames", &TrackerConfig::checker_refresh_interval_frames)
        .def_readwrite("checker_tracking_recovery_stable_interval_frames", &TrackerConfig::checker_tracking_recovery_stable_interval_frames)
        .def_readwrite("checker_tracking_recovery_zero_gain_backoff_after", &TrackerConfig::checker_tracking_recovery_zero_gain_backoff_after)
        .def_readwrite("checker_tracking_recovery_zero_gain_backoff_max_factor", &TrackerConfig::checker_tracking_recovery_zero_gain_backoff_max_factor)
        .def_readwrite("checker_local_completion_skip_enabled", &TrackerConfig::checker_local_completion_skip_enabled)
        .def_readwrite("checker_local_completion_probe_interval_frames", &TrackerConfig::checker_local_completion_probe_interval_frames)
        .def_readwrite("checker_local_completion_zero_gain_backoff_after", &TrackerConfig::checker_local_completion_zero_gain_backoff_after)
        .def_readwrite("checker_local_completion_zero_gain_backoff_max_factor", &TrackerConfig::checker_local_completion_zero_gain_backoff_max_factor)
        .def_readwrite("checker_local_completion_stale_predicted_frames", &TrackerConfig::checker_local_completion_stale_predicted_frames)
        .def_readwrite("checker_max_undecodeable_tracking_frames", &TrackerConfig::checker_max_undecodeable_tracking_frames)
        .def_readwrite("checker_min_fresh_correspondences_for_stable_tracking", &TrackerConfig::checker_min_fresh_correspondences_for_stable_tracking)
        .def_readwrite("checker_max_low_fresh_correspondence_frames", &TrackerConfig::checker_max_low_fresh_correspondence_frames)
        .def_readwrite("dot_canonical_size", &TrackerConfig::dot_canonical_size)
        .def_readwrite("dot_canonical_margin_px", &TrackerConfig::dot_canonical_margin_px)
        .def_readwrite("dot_min_dot_contrast", &TrackerConfig::dot_min_dot_contrast)
        .def_readwrite("dot_strong_dot_contrast", &TrackerConfig::dot_strong_dot_contrast)
        .def_readwrite("dot_commit_threshold", &TrackerConfig::dot_commit_threshold)
        .def_readwrite("dot_revoke_threshold", &TrackerConfig::dot_revoke_threshold)
        .def_readwrite("dot_uncertainty_low", &TrackerConfig::dot_uncertainty_low)
        .def_readwrite("dot_uncertainty_high", &TrackerConfig::dot_uncertainty_high)
        .def_readwrite("dot_warmup_frames", &TrackerConfig::dot_warmup_frames)
        .def_readwrite("dot_temporal_alpha", &TrackerConfig::dot_temporal_alpha)
        .def_readwrite("dot_commit_frames", &TrackerConfig::dot_commit_frames)
        .def_readwrite("dot_revoke_frames", &TrackerConfig::dot_revoke_frames)
        .def_readwrite("dot_use_temporal_smoothing", &TrackerConfig::dot_use_temporal_smoothing)
        .def_readwrite("dot_use_cell_value_cache", &TrackerConfig::dot_use_cell_value_cache)
        .def_readwrite("dot_cell_cache_max_age_frames", &TrackerConfig::dot_cell_cache_max_age_frames)
        .def_readwrite("dot_cell_cache_max_corner_motion_px", &TrackerConfig::dot_cell_cache_max_corner_motion_px)
        .def_readwrite("decoder_require_geometry_valid", &TrackerConfig::decoder_require_geometry_valid)
        .def_readwrite("decoder_accept_ambiguous", &TrackerConfig::decoder_accept_ambiguous)
        .def_readwrite("corr_min_votes", &TrackerConfig::corr_min_votes)
        .def_readwrite("corr_discard_conflicts", &TrackerConfig::corr_discard_conflicts)
        .def_readwrite("corr_require_detection_stable", &TrackerConfig::corr_require_detection_stable)
        .def_readwrite("corr_enable_dominant_rotation_filter", &TrackerConfig::corr_enable_dominant_rotation_filter)
        .def_readwrite("corr_min_rotation_support", &TrackerConfig::corr_min_rotation_support)
        .def_readwrite("corr_min_rotation_support_ratio", &TrackerConfig::corr_min_rotation_support_ratio)
        .def_readwrite("decode_only_mode", &TrackerConfig::decode_only_mode)
        .def_readwrite("enable_fast_persistent_path", &TrackerConfig::enable_fast_persistent_path)
        .def_readwrite("fast_persistent_min_points", &TrackerConfig::fast_persistent_min_points)
        .def_readwrite("fast_persistent_refresh_mean_error_px", &TrackerConfig::fast_persistent_refresh_mean_error_px)
        .def_readwrite("fast_persistent_dense_refine_enabled", &TrackerConfig::fast_persistent_dense_refine_enabled)
        .def_readwrite("fast_persistent_dense_min_points", &TrackerConfig::fast_persistent_dense_min_points)
        .def_readwrite("fast_persistent_dense_match_max_px", &TrackerConfig::fast_persistent_dense_match_max_px)
        .def_readwrite("fast_persistent_dense_min_second_best_margin_px", &TrackerConfig::fast_persistent_dense_min_second_best_margin_px)
        .def_readwrite("fast_persistent_dense_max_median_px", &TrackerConfig::fast_persistent_dense_max_median_px)
        .def_readwrite("fast_persistent_dense_max_p90_px", &TrackerConfig::fast_persistent_dense_max_p90_px)
        .def_readwrite("fast_persistent_dense_rescue_enabled", &TrackerConfig::fast_persistent_dense_rescue_enabled)
        .def_readwrite("fast_persistent_dense_rescue_min_green_ratio", &TrackerConfig::fast_persistent_dense_rescue_min_green_ratio)
        .def_readwrite("fast_persistent_dense_rescue_min_seed_median_px", &TrackerConfig::fast_persistent_dense_rescue_min_seed_median_px)
        .def_readwrite("fast_persistent_dense_min_image_coverage", &TrackerConfig::fast_persistent_dense_min_image_coverage)
        .def_readwrite("fast_persistent_dense_min_object_span_mm", &TrackerConfig::fast_persistent_dense_min_object_span_mm)
        .def_readwrite("fast_persistent_dense_min_distinct_rows", &TrackerConfig::fast_persistent_dense_min_distinct_rows)
        .def_readwrite("fast_persistent_dense_min_distinct_cols", &TrackerConfig::fast_persistent_dense_min_distinct_cols)
        .def_readwrite("fast_persistent_dense_pose_solver", &TrackerConfig::fast_persistent_dense_pose_solver)
        .def_readwrite("fast_persistent_dense_robust_refine_method", &TrackerConfig::fast_persistent_dense_robust_refine_method)
        .def_readwrite("fast_persistent_dense_robust_trim_enabled", &TrackerConfig::fast_persistent_dense_robust_trim_enabled)
        .def_readwrite("fast_persistent_dense_robust_trim_quantile", &TrackerConfig::fast_persistent_dense_robust_trim_quantile)
        .def_readwrite("fast_persistent_dense_robust_min_keep_ratio", &TrackerConfig::fast_persistent_dense_robust_min_keep_ratio)
        .def_readwrite("fast_persistent_dense_robust_max_mean_px", &TrackerConfig::fast_persistent_dense_robust_max_mean_px)
        .def_readwrite("fast_persistent_dense_robust_max_max_px", &TrackerConfig::fast_persistent_dense_robust_max_max_px)
        .def_readwrite("fast_persistent_dense_adaptive_refine_enabled", &TrackerConfig::fast_persistent_dense_adaptive_refine_enabled)
        .def_readwrite("fast_persistent_dense_adaptive_min_match_ratio", &TrackerConfig::fast_persistent_dense_adaptive_min_match_ratio)
        .def_readwrite("fast_persistent_dense_adaptive_motion_px", &TrackerConfig::fast_persistent_dense_adaptive_motion_px)
        .def_readwrite("fast_persistent_dense_adaptive_max_seed_mean_px", &TrackerConfig::fast_persistent_dense_adaptive_max_seed_mean_px)
        .def_readwrite("fast_persistent_dense_adaptive_max_seed_max_px", &TrackerConfig::fast_persistent_dense_adaptive_max_seed_max_px)
        .def_readwrite("enable_temporal_correspondence_persistence", &TrackerConfig::enable_temporal_correspondence_persistence)
        .def_readwrite("persistence_max_frames", &TrackerConfig::persistence_max_frames)
        .def_readwrite("persistence_min_points", &TrackerConfig::persistence_min_points)
        .def_readwrite("persistence_min_fresh_points_for_merge", &TrackerConfig::persistence_min_fresh_points_for_merge)
        .def_readwrite("persistence_min_points_after_decode_fail", &TrackerConfig::persistence_min_points_after_decode_fail)
        .def_readwrite("persistence_refresh_mean_error_px", &TrackerConfig::persistence_refresh_mean_error_px)
        .def_readwrite("persistence_max_translation_jump_mm", &TrackerConfig::persistence_max_translation_jump_mm)
        .def_readwrite("persistence_max_rotation_jump_deg", &TrackerConfig::persistence_max_rotation_jump_deg)
        .def_readwrite("persistence_use_pose_projection", &TrackerConfig::persistence_use_pose_projection)
        .def_readwrite("persistence_projection_max_reproj_px", &TrackerConfig::persistence_projection_max_reproj_px)
        .def_readwrite("persistence_projection_adaptive_match_enabled", &TrackerConfig::persistence_projection_adaptive_match_enabled)
        .def_readwrite("persistence_projection_adaptive_motion_start_px", &TrackerConfig::persistence_projection_adaptive_motion_start_px)
        .def_readwrite("persistence_projection_adaptive_motion_scale", &TrackerConfig::persistence_projection_adaptive_motion_scale)
        .def_readwrite("persistence_projection_adaptive_max_reproj_px", &TrackerConfig::persistence_projection_adaptive_max_reproj_px)
        .def_readwrite("persistence_projection_max_pose_error_px", &TrackerConfig::persistence_projection_max_pose_error_px)
        .def_readwrite("persistence_match_min_second_best_margin_px", &TrackerConfig::persistence_match_min_second_best_margin_px)
        .def_readwrite("persistence_uv_match_dist_px", &TrackerConfig::persistence_uv_match_dist_px)
        .def_readwrite("enable_pose_propagation", &TrackerConfig::enable_pose_propagation)
        .def_readwrite("pose_propagation_max_reproj_px", &TrackerConfig::pose_propagation_max_reproj_px)
        .def_readwrite("pose_propagation_border_px", &TrackerConfig::pose_propagation_border_px)
        .def_readwrite("pose_hold_max_frames", &TrackerConfig::pose_hold_max_frames)
        .def_readwrite("pose_hold_min_detection_corners", &TrackerConfig::pose_hold_min_detection_corners)
        .def_readwrite("emergency_pose_hold_enabled", &TrackerConfig::emergency_pose_hold_enabled)
        .def_readwrite("emergency_pose_hold_max_frames", &TrackerConfig::emergency_pose_hold_max_frames)
        .def_readwrite("fallback_pose_min_detection_matches", &TrackerConfig::fallback_pose_min_detection_matches)
        .def_readwrite("fallback_pose_max_median_corner_error_px", &TrackerConfig::fallback_pose_max_median_corner_error_px)
        .def_readwrite("fallback_pose_max_p90_corner_error_px", &TrackerConfig::fallback_pose_max_p90_corner_error_px)
        .def_readwrite("fallback_pose_max_mean_reprojection_error_px", &TrackerConfig::fallback_pose_max_mean_reprojection_error_px)
        .def_readwrite("fallback_pose_max_max_reprojection_error_px", &TrackerConfig::fallback_pose_max_max_reprojection_error_px)
        .def_readwrite("visual_corner_max_reprojection_error_px", &TrackerConfig::visual_corner_max_reprojection_error_px)
        .def_readwrite("visual_corner_min_count", &TrackerConfig::visual_corner_min_count)
        .def_readwrite("decode_update_min_visual_corners", &TrackerConfig::decode_update_min_visual_corners)
        .def_readwrite("decode_update_min_distinct_rows", &TrackerConfig::decode_update_min_distinct_rows)
        .def_readwrite("decode_update_min_distinct_cols", &TrackerConfig::decode_update_min_distinct_cols);

    py::class_<PersistentMatcher>(m, "PersistentMatcher")
        .def(py::init<const TrackerConfig&>(), py::arg("config") = TrackerConfig())
        .def("reset", &PersistentMatcher::reset)
        .def("clear_identities", &PersistentMatcher::clearIdentities)
        .def(
            "replace_identities",
            &PersistentMatcher::replaceIdentities,
            py::arg("identities"),
            py::arg("frame_index")
        )
        .def(
            "match",
            [](
                PersistentMatcher& self,
                const CheckerboardDetection& detection,
                int frame_index,
                py::array_t<double, py::array::c_style | py::array::forcecast> K,
                py::object dist_coeffs,
                py::object rvec,
                py::object tvec,
                double last_good_reproj_px
            ) {
                return self.match(
                    detection,
                    frame_index,
                    numpyToMatx33d(K),
                    optionalNumpyToVectorDouble(dist_coeffs),
                    optionalNumpyToVectorDouble(rvec),
                    optionalNumpyToVectorDouble(tvec),
                    last_good_reproj_px
                );
            },
            py::arg("detection"),
            py::arg("frame_index"),
            py::arg("K"),
            py::arg("dist_coeffs") = py::none(),
            py::arg("rvec") = py::none(),
            py::arg("tvec") = py::none(),
            py::arg("last_good_reproj_px") = -1.0
        )
        .def(
            "estimate_pose",
            [](
                PersistentMatcher& self,
                const CheckerboardDetection& detection,
                int frame_index,
                py::array_t<double, py::array::c_style | py::array::forcecast> K,
                py::object dist_coeffs,
                py::object rvec,
                py::object tvec,
                double last_good_reproj_px,
                int lost_frames
            ) {
                return self.estimatePose(
                    detection,
                    frame_index,
                    numpyToMatx33d(K),
                    optionalNumpyToVectorDouble(dist_coeffs),
                    optionalNumpyToVectorDouble(rvec),
                    optionalNumpyToVectorDouble(tvec),
                    last_good_reproj_px,
                    lost_frames
                );
            },
            py::arg("detection"),
            py::arg("frame_index"),
            py::arg("K"),
            py::arg("dist_coeffs") = py::none(),
            py::arg("rvec") = py::none(),
            py::arg("tvec") = py::none(),
            py::arg("last_good_reproj_px") = -1.0,
            py::arg("lost_frames") = 0
        )
        .def(
            "estimate_fast_pose",
            [](
                PersistentMatcher& self,
                const CheckerboardDetection& detection,
                const MarkerGeometry& geometry,
                int frame_index,
                py::array_t<double, py::array::c_style | py::array::forcecast> K,
                py::object dist_coeffs,
                py::object rvec,
                py::object tvec,
                double last_good_reproj_px,
                py::object previous_rvec,
                py::object previous_tvec,
                int lost_frames,
                int max_pts_seen
            ) {
                return self.estimateFastPose(
                    detection,
                    geometry,
                    frame_index,
                    numpyToMatx33d(K),
                    optionalNumpyToVectorDouble(dist_coeffs),
                    optionalNumpyToVectorDouble(rvec),
                    optionalNumpyToVectorDouble(tvec),
                    last_good_reproj_px,
                    optionalNumpyToVectorDouble(previous_rvec),
                    optionalNumpyToVectorDouble(previous_tvec),
                    lost_frames,
                    max_pts_seen
                );
            },
            py::arg("detection"),
            py::arg("geometry"),
            py::arg("frame_index"),
            py::arg("K"),
            py::arg("dist_coeffs") = py::none(),
            py::arg("rvec") = py::none(),
            py::arg("tvec") = py::none(),
            py::arg("last_good_reproj_px") = -1.0,
            py::arg("previous_rvec") = py::none(),
            py::arg("previous_tvec") = py::none(),
            py::arg("lost_frames") = 0,
            py::arg("max_pts_seen") = 0
        )
        .def_property_readonly(
            "identities",
            [](const PersistentMatcher& self) { return self.identities(); },
            py::return_value_policy::copy
        )
        .def_property_readonly(
            "persistent_frame_index",
            &PersistentMatcher::persistentFrameIndex
        )
        .def_property_readonly(
            "config",
            [](const PersistentMatcher& self) { return self.config(); },
            py::return_value_policy::copy
        );

    py::class_<TrackerFrameResult>(m, "TrackerFrameResult")
        .def(py::init<>())
        .def_readwrite("success", &TrackerFrameResult::success)
        .def_readwrite("mode", &TrackerFrameResult::mode)
        .def_readwrite("message", &TrackerFrameResult::message)
        .def_readwrite("detection_valid", &TrackerFrameResult::detection_valid)
        .def_readwrite("detection_tracking", &TrackerFrameResult::detection_tracking)
        .def_readwrite("detection_stable", &TrackerFrameResult::detection_stable)
        .def_readwrite("detection_corner_count", &TrackerFrameResult::detection_corner_count)
        .def_readwrite("detection_cell_count", &TrackerFrameResult::detection_cell_count)
        .def_readwrite("detection_corners", &TrackerFrameResult::detection_corners)
        .def_readwrite("frame_index", &TrackerFrameResult::frame_index)
        .def_readwrite("lost_frames", &TrackerFrameResult::lost_frames)
        .def_readwrite("pose_tracker_has_pose", &TrackerFrameResult::pose_tracker_has_pose)
        .def_readwrite("pose_tracker_rvec", &TrackerFrameResult::pose_tracker_rvec)
        .def_readwrite("pose_tracker_tvec", &TrackerFrameResult::pose_tracker_tvec)
        .def_readwrite("pose_tracker_T_marker_camera", &TrackerFrameResult::pose_tracker_T_marker_camera)
        .def_readwrite("current_pose_accepted", &TrackerFrameResult::current_pose_accepted)
        .def_readwrite("has_accepted_pose", &TrackerFrameResult::has_accepted_pose)
        .def_readwrite("accepted_pose_frame", &TrackerFrameResult::accepted_pose_frame)
        .def_readwrite("accepted_visual_corner_count", &TrackerFrameResult::accepted_visual_corner_count)
        .def_readwrite("max_pts_seen", &TrackerFrameResult::max_pts_seen)
        .def_readwrite("last_good_reproj_px", &TrackerFrameResult::last_good_reproj_px)
        .def_readwrite("accepted_rvec", &TrackerFrameResult::accepted_rvec)
        .def_readwrite("accepted_tvec", &TrackerFrameResult::accepted_tvec)
        .def_readwrite("accepted_T_marker_camera", &TrackerFrameResult::accepted_T_marker_camera)
        .def_readwrite("pose_source", &TrackerFrameResult::pose_source)
        .def_readwrite("rvec", &TrackerFrameResult::rvec)
        .def_readwrite("tvec", &TrackerFrameResult::tvec)
        .def_readwrite("T_marker_camera", &TrackerFrameResult::T_marker_camera)
        .def_readwrite("num_points", &TrackerFrameResult::num_points)
        .def_readwrite("num_inliers", &TrackerFrameResult::num_inliers)
        .def_readwrite("mean_reprojection_error_px", &TrackerFrameResult::mean_reprojection_error_px)
        .def_readwrite("max_reprojection_error_px", &TrackerFrameResult::max_reprojection_error_px)
        .def_readwrite("confidence", &TrackerFrameResult::confidence)
        .def_readwrite("pnp_method", &TrackerFrameResult::pnp_method)
        .def_readwrite("visual_corner_count", &TrackerFrameResult::visual_corner_count)
        .def_readwrite("corners", &TrackerFrameResult::corners)
        .def_readwrite("correspondence_corners", &TrackerFrameResult::correspondence_corners)
        .def_readwrite("persistent_count", &TrackerFrameResult::persistent_count)
        .def_readwrite("fast_attempted", &TrackerFrameResult::fast_attempted)
        .def_readwrite("fast_success", &TrackerFrameResult::fast_success)
        .def_readwrite("fast_route_decode", &TrackerFrameResult::fast_route_decode)
        .def_readwrite("fast_matches", &TrackerFrameResult::fast_matches)
        .def_readwrite("fast_reason", &TrackerFrameResult::fast_reason)
        .def_readwrite("fast_dense_attempted", &TrackerFrameResult::fast_dense_attempted)
        .def_readwrite("fast_dense_success", &TrackerFrameResult::fast_dense_success)
        .def_readwrite("fast_dense_matches", &TrackerFrameResult::fast_dense_matches)
        .def_readwrite("fast_dense_reason", &TrackerFrameResult::fast_dense_reason)
        .def_readwrite("dot_cell_count", &TrackerFrameResult::dot_cell_count)
        .def_readwrite("dot_valid_cell_count", &TrackerFrameResult::dot_valid_cell_count)
        .def_readwrite("patch_count", &TrackerFrameResult::patch_count)
        .def_readwrite("decoded_patch_count", &TrackerFrameResult::decoded_patch_count)
        .def_readwrite("decoded_valid_patch_count", &TrackerFrameResult::decoded_valid_patch_count)
        .def_readwrite("correspondence_count", &TrackerFrameResult::correspondence_count)
        .def_readwrite("timings_ms", &TrackerFrameResult::timings_ms);

    py::class_<TrackerEngine>(m, "TrackerEngine")
        .def(
            py::init([](
                const std::string& field_path,
                const std::string& marker_json_path,
                py::array_t<double, py::array::c_style | py::array::forcecast> K,
                py::object dist_coeffs,
                const TrackerConfig& config
            ) {
                return std::make_unique<TrackerEngine>(
                    field_path,
                    marker_json_path,
                    numpyToMatx33d(K),
                    optionalNumpyToVectorDouble(dist_coeffs),
                    config
                );
            }),
            py::arg("field_path"),
            py::arg("marker_json_path"),
            py::arg("K"),
            py::arg("dist_coeffs") = py::none(),
            py::arg("config") = TrackerConfig()
        )
        .def(
            "process_frame",
            [](TrackerEngine& self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img,
               bool run_detection) -> TrackerFrameResult
            {
                cv::Mat mat = numpyToMat(img);
                return self.processFrame(mat, run_detection);
            },
            py::arg("frame"),
            py::arg("run_detection") = true
        )
        .def("reset", &TrackerEngine::reset)
        .def("frame_index", &TrackerEngine::frameIndex)
        .def("mode", &TrackerEngine::mode)
        .def("marker_assets_loaded", &TrackerEngine::markerAssetsLoaded)
        .def_property_readonly(
            "config",
            [](const TrackerEngine& self) { return self.config(); },
            py::return_value_policy::copy
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
        .def_readwrite("tracking_recovery_stable_interval_frames", &CheckerboardDetectorConfig::tracking_recovery_stable_interval_frames)
        .def_readwrite("tracking_recovery_zero_gain_backoff_after", &CheckerboardDetectorConfig::tracking_recovery_zero_gain_backoff_after)
        .def_readwrite("tracking_recovery_zero_gain_backoff_max_factor", &CheckerboardDetectorConfig::tracking_recovery_zero_gain_backoff_max_factor)
        .def_readwrite("tracking_local_completion_skip_enabled", &CheckerboardDetectorConfig::tracking_local_completion_skip_enabled)
        .def_readwrite("tracking_local_completion_probe_interval_frames", &CheckerboardDetectorConfig::tracking_local_completion_probe_interval_frames)
        .def_readwrite("tracking_local_completion_zero_gain_backoff_after", &CheckerboardDetectorConfig::tracking_local_completion_zero_gain_backoff_after)
        .def_readwrite("tracking_local_completion_zero_gain_backoff_max_factor", &CheckerboardDetectorConfig::tracking_local_completion_zero_gain_backoff_max_factor)
        .def_readwrite("tracking_local_completion_stale_predicted_frames", &CheckerboardDetectorConfig::tracking_local_completion_stale_predicted_frames)
        .def_readwrite("lk_win_size", &CheckerboardDetectorConfig::lk_win_size)
        .def_readwrite("lk_max_level", &CheckerboardDetectorConfig::lk_max_level)
        .def_readwrite("lk_max_iters", &CheckerboardDetectorConfig::lk_max_iters)
        .def_readwrite("lk_epsilon", &CheckerboardDetectorConfig::lk_epsilon)
        .def_readwrite("max_lk_error", &CheckerboardDetectorConfig::max_lk_error)
        .def_readwrite("lk_use_initial_flow_prediction", &CheckerboardDetectorConfig::lk_use_initial_flow_prediction)
        .def_readwrite("lk_initial_flow_max_prediction_px", &CheckerboardDetectorConfig::lk_initial_flow_max_prediction_px)
        .def_readwrite("lk_corner_kalman_enabled", &CheckerboardDetectorConfig::lk_corner_kalman_enabled)
        .def_readwrite("lk_corner_kalman_process_noise_px", &CheckerboardDetectorConfig::lk_corner_kalman_process_noise_px)
        .def_readwrite("lk_corner_kalman_measurement_noise_px", &CheckerboardDetectorConfig::lk_corner_kalman_measurement_noise_px)
        .def_readwrite("lk_corner_kalman_max_update_innovation_px", &CheckerboardDetectorConfig::lk_corner_kalman_max_update_innovation_px)
        .def_readwrite("lk_corner_kalman_max_gap_frames", &CheckerboardDetectorConfig::lk_corner_kalman_max_gap_frames)
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
        .def_readonly("votes", &Correspondence2D3D::votes)
        .def_readonly("visibility_score", &Correspondence2D3D::visibility_score)
        .def_readonly("observed_frames", &Correspondence2D3D::observed_frames)
        .def_readonly("predicted", &Correspondence2D3D::predicted);

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
