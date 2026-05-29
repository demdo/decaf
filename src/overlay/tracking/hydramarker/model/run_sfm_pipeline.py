from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import io
import sys

import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tracking.hydramarker.model.bootstrap import (
    CameraCalibration,
    run_bootstrap,
)
from overlay.tracking.hydramarker.model.bundle_adjustment import (
    PyCeresOptions,
    compute_adaptive_observation_weights,
    compute_observation_reprojection_errors,
    print_bundle_adjustment_summary,
    run_bundle_adjustment,
    select_good_frames_for_bundle_adjustment,
    select_observation_outliers,
)
from overlay.tracking.hydramarker.model.incremental import (
    register_remaining_frames,
)
from overlay.tracking.hydramarker.model.observations import (
    FrameObservation,
    load_observations_npz,
)
from overlay.tracking.hydramarker.model.state import (
    SfMState,
    create_state_from_bootstrap,
)
from overlay.tracking.hydramarker.model.visualization import (
    plot_sfm_state,
    visualize_aligned_state,
)
from overlay.tracking.hydramarker.model.alignment import (
    align_state_to_marker_frame_inplace,
)
from overlay.tracking.hydramarker.model.export_marker_map import (
    export_marker_geometry_json,
)


TOPOLOGY_REGULARIZATION_WEIGHT = 0.5
CELL_SHAPE_REGULARIZATION_WEIGHT = 0.1

BA_MIN_OBSERVATIONS = 15
BA_MAX_MEDIAN_ERROR_PX = 1.5

OUTLIER_ABSOLUTE_THRESHOLD_PX = 2.0
OUTLIER_MAD_SIGMA = 3.5
OUTLIER_MAX_FRACTION = 0.03
OUTLIER_MIN_ERROR_PX = 1.0

SHOW_PLOTS = True
EXPORT_FILENAME = "marker_geometry_sfm.json"


def choose_file_qt(title: str, file_filter: str) -> Path:
    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        file_filter,
    )

    if not path:
        raise RuntimeError(f"No file selected: {title}")

    return Path(path)


def load_camera_npz(path: Path) -> CameraCalibration:
    path = Path(path)

    with np.load(path, allow_pickle=True) as npz:
        if "K" in npz:
            K = npz["K"]
        elif "camera_matrix" in npz:
            K = npz["camera_matrix"]
        else:
            raise KeyError("Camera NPZ must contain 'K' or 'camera_matrix'.")

        if "dist" in npz:
            dist = npz["dist"]
        elif "dist_coeffs" in npz:
            dist = npz["dist_coeffs"]
        elif "distortion_coeffs" in npz:
            dist = npz["distortion_coeffs"]
        else:
            dist = np.zeros((0, 1), dtype=np.float64)

    return CameraCalibration(
        K=K,
        dist_coeffs=dist,
    )


def load_pipeline_inputs() -> tuple[list[FrameObservation], CameraCalibration, Path]:
    observations_path = choose_file_qt(
        "Select HydraMarker observations .npz",
        "NPZ files (*.npz)",
    )

    camera_path = choose_file_qt(
        "Select camera intrinsics .npz",
        "NPZ files (*.npz)",
    )

    marker_json_path = choose_file_qt(
        "Select HydraMarker marker .json",
        "JSON files (*.json)",
    )

    frames = load_observations_npz(observations_path)
    calibration = load_camera_npz(camera_path)

    return frames, calibration, marker_json_path


def final_json_path() -> Path:
    return Path(__file__).resolve().parent / EXPORT_FILENAME


def call_silent(func, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def export_final_marker_geometry(
    state: SfMState,
    marker_json_path: Path,
) -> Path:
    output_json_path = final_json_path()

    export_marker_geometry_json(
        state=state,
        source_marker_json_path=marker_json_path,
        output_json_path=output_json_path,
        xyz_decimals=6,
        include_camera_poses=False,
        overwrite_marker_type="sfm_map",
    )

    return output_json_path


def run_first_bundle_adjustment(
    state: SfMState,
    ba_frame_ids: list[int],
    marker_json_path: Path,
):
    ba_result = run_bundle_adjustment(
        state,
        frame_ids=ba_frame_ids,
        options=PyCeresOptions(
            loss="huber",
            loss_scale=1.0,
            max_iterations=100,
            report_full=True,
        ),
        update_state=True,
        marker_json_path=marker_json_path,
        topology_regularization_weight=TOPOLOGY_REGULARIZATION_WEIGHT,
        cell_shape_regularization_weight=CELL_SHAPE_REGULARIZATION_WEIGHT,
        ignored_observations=None,
    )

    print_bundle_adjustment_summary(ba_result)

    if not ba_result.success:
        raise RuntimeError(ba_result.message)

    return ba_result


def run_second_bundle_adjustment(
    state: SfMState,
    ba_frame_ids: list[int],
    marker_json_path: Path,
    ignored_observations: set[tuple[int, int]],
    adaptive_observation_weights: dict[tuple[int, int], float],
):
    ba_result = run_bundle_adjustment(
        state,
        frame_ids=ba_frame_ids,
        options=PyCeresOptions(
            loss="huber",
            loss_scale=1.0,
            max_iterations=100,
            report_full=True,
        ),
        update_state=True,
        marker_json_path=marker_json_path,
        topology_regularization_weight=TOPOLOGY_REGULARIZATION_WEIGHT,
        cell_shape_regularization_weight=CELL_SHAPE_REGULARIZATION_WEIGHT,
        ignored_observations=ignored_observations,
        observation_weights=adaptive_observation_weights,
    )

    print_bundle_adjustment_summary(ba_result)

    if not ba_result.success:
        raise RuntimeError(ba_result.message)

    return ba_result


def run_sfm_pipeline(
    frames: list[FrameObservation],
    calibration: CameraCalibration,
    marker_json_path: Path,
) -> SfMState:
    bootstrap = run_bootstrap(
        frames,
        calibration,
        min_shared_ids=20,
        min_frame_gap=5,
        ransac_threshold_norm=1e-3,
        max_reprojection_error_norm=2e-3,
    )

    if not bootstrap.success:
        raise RuntimeError(bootstrap.message)

    state = create_state_from_bootstrap(
        frames=frames,
        calibration=calibration,
        bootstrap=bootstrap,
    )

    register_remaining_frames(
        state,
        min_points=12,
        ransac_reprojection_error_px=1.5,
        confidence=0.999,
        iterations_count=200,
        refine_lm=True,
        max_mean_reprojection_error_px=None,
    )

    if SHOW_PLOTS:
        plot_sfm_state(
            state,
            show_marker_ids=True,
            show_camera_labels=False,
            show_camera_axes=False,
            title="HydraMarker SfM State Before BA",
            axis_unit="SfM scale",
        )

    ba_frame_ids = call_silent(
        select_good_frames_for_bundle_adjustment,
        state,
        min_observations=BA_MIN_OBSERVATIONS,
        max_median_error_px=BA_MAX_MEDIAN_ERROR_PX,
    )

    run_first_bundle_adjustment(
        state=state,
        ba_frame_ids=ba_frame_ids,
        marker_json_path=marker_json_path,
    )

    observation_errors = compute_observation_reprojection_errors(
        state,
        frame_ids=ba_frame_ids,
    )

    adaptive_observation_weights = compute_adaptive_observation_weights(
        observation_errors,
        marker_good_px=0.35,
        marker_bad_px=0.80,
        frame_good_px=0.35,
        frame_bad_px=0.90,
        min_marker_weight=0.35,
        min_frame_weight=0.25,
        min_observation_weight=0.10,
        print_summary=False,
    )

    ignored_observations = call_silent(
        select_observation_outliers,
        observation_errors,
        absolute_threshold_px=OUTLIER_ABSOLUTE_THRESHOLD_PX,
        mad_sigma=OUTLIER_MAD_SIGMA,
        max_fraction=OUTLIER_MAX_FRACTION,
        min_error_px=OUTLIER_MIN_ERROR_PX,
    )

    run_second_bundle_adjustment(
        state=state,
        ba_frame_ids=ba_frame_ids,
        marker_json_path=marker_json_path,
        ignored_observations=ignored_observations,
        adaptive_observation_weights=adaptive_observation_weights,
    )

    if SHOW_PLOTS:
        plot_sfm_state(
            state,
            show_marker_ids=True,
            show_camera_labels=False,
            show_camera_axes=False,
            title="HydraMarker SfM State After BA Before Alignment",
            axis_unit="SfM scale",
        )

    alignment_result = align_state_to_marker_frame_inplace(
        state,
        marker_json_path=marker_json_path,
        scale_metric=True,
        scale_mode="topology",
        alignment_mode="topology",
    )

    if SHOW_PLOTS:
        call_silent(
            visualize_aligned_state,
            state=state,
            marker_json_path=marker_json_path,
            alignment_result=alignment_result,
            show_full_sfm_plot=True,
            show_reference_plot=True,
        )

    export_final_marker_geometry(
        state=state,
        marker_json_path=marker_json_path,
    )

    return state


def main() -> None:
    frames, calibration, marker_json_path = load_pipeline_inputs()

    run_sfm_pipeline(
        frames=frames,
        calibration=calibration,
        marker_json_path=marker_json_path,
    )


if __name__ == "__main__":
    main()
