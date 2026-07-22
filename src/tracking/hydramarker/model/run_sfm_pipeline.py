"""Command-line orchestration for the HydraMarker SfM model pipeline.

The pipeline loads recorded observations, bootstraps a model, runs incremental
and bundle-adjustment stages, and writes diagnostics and tracker-ready exports.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import io
import json
import sys

import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.hydramarker.model.bootstrap import (
    CameraCalibration,
    run_bootstrap,
)
from tracking.hydramarker.model.bundle_adjustment import (
    PyCeresOptions,
    compute_adaptive_observation_weights,
    compute_observation_reprojection_errors,
    print_bundle_adjustment_summary,
    run_bundle_adjustment,
    select_good_frames_for_bundle_adjustment,
    select_observation_outliers,
)
from tracking.hydramarker.model.incremental import (
    register_remaining_frames,
    triangulate_missing_markers,
)
from tracking.hydramarker.model.observations import (
    FrameObservation,
    load_observations_npz,
)
from tracking.hydramarker.model.state import (
    SfMState,
    create_state_from_bootstrap,
)
from tracking.hydramarker.model.visualization import (
    plot_sfm_state,
    visualize_aligned_state,
)
from tracking.hydramarker.model.alignment import (
    align_state_to_marker_frame_inplace,
)
from tracking.hydramarker.model.export_marker_map import (
    export_marker_geometry_json,
)
from tracking.hydramarker.model.diagnostics import (
    write_sfm_geometry_diagnostics,
)


# NOTE (2026-07-15): cylinder surface/grid regularization was REMOVED from
# this pipeline. It assumed ideal conditions (perfect cylinder, rows exactly
# parallel to the axis, exact grid spacing) and erased real shape modes (label
# skew ~0.4 deg, ~0.2 mm glue/print relief) that barely change reprojection
# but bias the estimated poses -- measured as +2 mm tool-tip lever error at
# 180 mm. The exporter still fits cylinder parameters post-hoc from the
# exported corners (surface_model.fitted) for the runtime pose refinement.
TOPOLOGY_REGULARIZATION_WEIGHT = 0.5
CELL_SHAPE_REGULARIZATION_WEIGHT = 0.1
BA_MAX_ITERATIONS = 100
BA_SHOW_PROGRESS = True

BOOTSTRAP_MAX_PAIRS = None
BOOTSTRAP_MAX_CANDIDATE_PAIRS_TO_TRY = 5000

BA_MIN_OBSERVATIONS = 15
BA_MAX_MEDIAN_ERROR_PX = 1.5

OUTLIER_ABSOLUTE_THRESHOLD_PX = 2.0
OUTLIER_MAD_SIGMA = 3.5
OUTLIER_MAX_FRACTION = 0.03
OUTLIER_MIN_ERROR_PX = 1.0
OUTLIER_MAX_PER_MARKER_FRACTION = 0.15
OUTLIER_MAX_PER_MARKER_COUNT = 24

TRIANGULATE_MISSING_MIN_OBSERVATIONS = 8
TRIANGULATE_MISSING_MIN_INLIERS = 6
TRIANGULATE_MISSING_MAX_REPROJ_PX = 2.5
TRIANGULATE_MISSING_MAX_OBSERVATIONS_PER_MARKER = 0
TRIANGULATE_MISSING_MAX_PAIR_CANDIDATES_PER_MARKER = 5000

SHOW_PLOTS = True
EXPORT_FILENAME = "marker_geometry_sfm.json"
DIAGNOSTICS_FILENAME = "marker_geometry_sfm_diagnostics.txt"


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


def final_diagnostics_path() -> Path:
    return Path(__file__).resolve().parent / DIAGNOSTICS_FILENAME


def read_id_num_cols(marker_json_path: Path) -> int | None:
    marker_json_path = Path(marker_json_path)

    with marker_json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    id_encoding = meta.get("id_encoding", {})

    if isinstance(id_encoding, dict) and "num_cols" in id_encoding:
        return int(id_encoding["num_cols"])

    if "id_num_cols" in meta:
        return int(meta["id_num_cols"])

    if "corner_cols" in meta:
        return int(meta["corner_cols"])

    return None


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
            max_iterations=BA_MAX_ITERATIONS,
            progress_to_stdout=BA_SHOW_PROGRESS,
            report_full=False,
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
            max_iterations=BA_MAX_ITERATIONS,
            progress_to_stdout=BA_SHOW_PROGRESS,
            report_full=False,
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


def triangulate_missing_markers_for_stage(
    state: SfMState,
    *,
    label: str,
) -> list:
    results = triangulate_missing_markers(
        state,
        min_observations=TRIANGULATE_MISSING_MIN_OBSERVATIONS,
        min_inliers=TRIANGULATE_MISSING_MIN_INLIERS,
        max_reprojection_error_px=TRIANGULATE_MISSING_MAX_REPROJ_PX,
        max_observations_per_marker=(
            TRIANGULATE_MISSING_MAX_OBSERVATIONS_PER_MARKER
        ),
        max_pair_candidates_per_marker=(
            TRIANGULATE_MISSING_MAX_PAIR_CANDIDATES_PER_MARKER
        ),
    )

    triangulated_count = sum(1 for r in results if r.success)
    failed_count = len(results) - triangulated_count

    print(
        f"[SfM] missing-marker triangulation {label}: "
        f"added={triangulated_count}, failed={failed_count}, "
        f"total_markers={len(state.marker_positions)}"
    )

    return results


def register_remaining_frames_for_stage(
    state: SfMState,
    *,
    min_points: int,
    label: str,
) -> list:
    results = register_remaining_frames(
        state,
        min_points=min_points,
        ransac_reprojection_error_px=1.5,
        confidence=0.999,
        iterations_count=200,
        refine_lm=True,
        max_mean_reprojection_error_px=None,
    )

    success_count = sum(1 for r in results if r.success)
    failed_count = len(results) - success_count

    print(
        f"[SfM] frame registration {label}: "
        f"added={success_count}, failed={failed_count}, "
        f"total_poses={len(state.poses)}"
    )

    return results


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
        max_pairs=BOOTSTRAP_MAX_PAIRS,
        max_candidate_pairs_to_try=BOOTSTRAP_MAX_CANDIDATE_PAIRS_TO_TRY,
        id_num_cols=read_id_num_cols(marker_json_path),
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

    registration_min_points = min(
        12,
        max(6, len(state.marker_positions)),
    )

    print(
        "[SfM] frame registration: "
        f"min_points={registration_min_points}, "
        f"bootstrap_markers={len(state.marker_positions)}"
    )

    register_remaining_frames_for_stage(
        state,
        min_points=registration_min_points,
        label="after bootstrap",
    )

    triangulation_results = triangulate_missing_markers_for_stage(
        state,
        label="before first BA",
    )

    register_remaining_frames_for_stage(
        state,
        min_points=12,
        label="after pre-BA triangulation",
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

    post_ba_triangulation_results = triangulate_missing_markers_for_stage(
        state,
        label="after first BA",
    )

    triangulation_results.extend(post_ba_triangulation_results)

    ba_frame_ids = call_silent(
        select_good_frames_for_bundle_adjustment,
        state,
        min_observations=BA_MIN_OBSERVATIONS,
        max_median_error_px=BA_MAX_MEDIAN_ERROR_PX,
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
        max_per_marker_fraction=OUTLIER_MAX_PER_MARKER_FRACTION,
        max_per_marker_count=OUTLIER_MAX_PER_MARKER_COUNT,
    )

    run_second_bundle_adjustment(
        state=state,
        ba_frame_ids=ba_frame_ids,
        marker_json_path=marker_json_path,
        ignored_observations=ignored_observations,
        adaptive_observation_weights=adaptive_observation_weights,
    )

    final_observation_errors = compute_observation_reprojection_errors(
        state,
        frame_ids=ba_frame_ids,
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

    diagnostics_path = write_sfm_geometry_diagnostics(
        output_path=final_diagnostics_path(),
        state=state,
        marker_json_path=marker_json_path,
        observation_errors=final_observation_errors,
        triangulation_results=triangulation_results,
        column_regularization_stats={},
    )

    print(f"[SfM] geometry diagnostics written: {diagnostics_path}")

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
