from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("QT_API", "pyside6")


COMPONENTS = (
    ("x", "x", "#1f77b4"),
    ("y", "y", "#2ca02c"),
    ("z", "z", "#d62728"),
)


def _ensure_src_on_path() -> None:
    tracking_root = Path(__file__).resolve().parents[2]
    src_root = tracking_root.parent
    src = str(src_root)
    sys.path = [p for p in sys.path if str(p) != src]
    sys.path.insert(0, src)


_ensure_src_on_path()

from tracking.hydramarker.calib import calib_checkerboard


BoardPoseCalibration = calib_checkerboard.CheckerboardPose
CHECKERBOARD_PATTERN = calib_checkerboard.CHECKERBOARD_PATTERN
CHECKERBOARD_SQUARE_SIZE_MM = calib_checkerboard.CHECKERBOARD_SQUARE_SIZE_MM
BOARD_REFINE_TARGET_FRAMES = calib_checkerboard.DEFAULT_MAX_FRAMES
BOARD_REFINE_MIN_FRAMES = calib_checkerboard.DEFAULT_MIN_FRAMES
BOARD_AXIS_LENGTH_MM = calib_checkerboard.BOARD_AXIS_LENGTH_MM


def _tracker_log_module():
    from tracking.hydramarker import tracker_log

    return tracker_log


def _force_dense_refine_config(tracker) -> None:
    cfg = getattr(tracker, "config", None)
    if cfg is None:
        return

    dense_settings = {
        "fast_persistent_dense_refine_enabled": True,
        "fast_persistent_dense_min_points": 24,
        "fast_persistent_dense_match_max_px": 3.0,
        "fast_persistent_dense_min_second_best_margin_px": 2.0,
        "fast_persistent_dense_max_median_px": 1.2,
        "fast_persistent_dense_max_p90_px": 2.5,
        "fast_persistent_dense_min_image_coverage": 0.35,
        "fast_persistent_dense_min_object_span_mm": 12.0,
        "fast_persistent_dense_min_distinct_rows": 2,
        "fast_persistent_dense_min_distinct_cols": 2,
        "fast_persistent_dense_pose_solver": "direct_prior",
    }
    for name, value in dense_settings.items():
        setattr(cfg, name, value)

    print(
        "[debug_tracker_translation] dense refine enabled "
        f"(min_points={getattr(cfg, 'fast_persistent_dense_min_points', '?')}, "
        f"max_px={getattr(cfg, 'fast_persistent_dense_match_max_px', '?')}, "
        f"solver={getattr(cfg, 'fast_persistent_dense_pose_solver', '?')})"
    )


def _load_qt_widgets():
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    return QApplication, QFileDialog, QMessageBox


def _load_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _to_float(value) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _to_int(value, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _qt_app():
    QApplication, _, _ = _load_qt_widgets()
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def select_jsonl_with_qt() -> Path | None:
    _qt_app()
    _, QFileDialog, _ = _load_qt_widgets()

    script_path = Path(__file__).resolve()
    default_dir = script_path.parents[1] / "tests" / "hydramarker_tracker_runs"
    if not default_dir.exists():
        default_dir = Path.cwd()

    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select HydraTracker run log",
        str(default_dir),
        "HydraTracker JSONL (*.jsonl);;All Files (*)",
    )

    if not path:
        return None
    return Path(path)


def _camera_tvec_to_board_mm(tvec_camera_mm: np.ndarray, T_B_C: np.ndarray) -> np.ndarray:
    p_c = np.ones(4, dtype=np.float64)
    p_c[:3] = np.asarray(tvec_camera_mm, dtype=np.float64).reshape(3)
    p_b = np.asarray(T_B_C, dtype=np.float64).reshape(4, 4) @ p_c
    return p_b[:3]


def confirm_board_pose(pose: BoardPoseCalibration) -> bool:
    _qt_app()
    _, _, QMessageBox = _load_qt_widgets()
    t = pose.tvec_cb_mm.reshape(3)
    quality = calib_checkerboard.pose_quality_dict(pose)
    reply = QMessageBox.question(
        None,
        "Checkerboard pose uebernehmen?",
        (
            "Checkerboard-Pose uebernehmen?\n\n"
            f"Reprojection mean: {pose.reproj_mean_px:.3f} px\n"
            f"Reprojection p95:  {pose.reproj_p95_px:.3f} px\n"
            f"Reprojection max:  {pose.reproj_max_px:.3f} px\n"
            f"Collected frames:  {pose.collected_frames}\n"
            f"Used frames:       {quality['frames_used']}\n"
            f"Corner noise mean: {pose.mean_corner_std_px:.3f} px\n"
            f"Solver:            {pose.solver_mode} "
            f"(candidate={pose.selected_candidate_index}, "
            f"ambiguous={pose.pose_ambiguous})\n"
            f"IPPE alt gap:      {pose.alternative_error_gap_px:.4f} px\n"
            f"Camera board tvec: x={t[0]:.1f} mm, y={t[1]:.1f} mm, z={t[2]:.1f} mm\n\n"
            "Yes: use this board pose and start the tracker view.\n"
            "No: repeat checkerboard detection."
        ),
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes,
    )
    return reply == QMessageBox.Yes


def calibrate_checkerboard_pose(
    pipe,
    K: np.ndarray,
    dist: np.ndarray,
) -> BoardPoseCalibration | None:
    while True:
        pose = calib_checkerboard.capture_checkerboard_pose_from_pipeline(
            pipe,
            K,
            dist,
            min_frames=BOARD_REFINE_MIN_FRAMES,
            max_frames=BOARD_REFINE_TARGET_FRAMES,
            window_name="HydraTracker Translation Debug",
        )
        if pose is None:
            return None
        if confirm_board_pose(pose):
            return pose


def board_pose_record(board_pose: BoardPoseCalibration, run_id: str) -> dict[str, Any]:
    quality = calib_checkerboard.pose_quality_dict(board_pose)
    return {
        "type": "board_pose",
        "run_id": run_id,
        "coordinate_frame": "checkerboard",
        "pattern_inner_corners": list(CHECKERBOARD_PATTERN),
        "printed_cells": [10, 10],
        "square_size_mm": CHECKERBOARD_SQUARE_SIZE_MM,
        "rvec_cb": board_pose.rvec_cb.reshape(3).tolist(),
        "tvec_cb_mm": board_pose.tvec_cb_mm.reshape(3).tolist(),
        "T_C_B": board_pose.T_C_B.tolist(),
        "T_B_C": board_pose.T_B_C.tolist(),
        "reprojection": {
            "mean_px": board_pose.reproj_mean_px,
            "median_px": board_pose.reproj_median_px,
            "p95_px": board_pose.reproj_p95_px,
            "max_px": board_pose.reproj_max_px,
            "frames_used": quality["frames_used"],
            "frames_total": quality["frames_total"],
            "collected_frames": board_pose.collected_frames,
            "frame_rms_mean_px": quality["frame_rms_mean_px"],
            "frame_rms_median_px": quality["frame_rms_median_px"],
            "frame_rms_p95_px": quality["frame_rms_p95_px"],
            "mean_corner_std_px": board_pose.mean_corner_std_px,
            "max_corner_std_px": board_pose.max_corner_std_px,
            "pnp_flag": quality["pnp_flag"],
        },
        "solver": {
            "mode": board_pose.solver_mode,
            "candidate_count": board_pose.candidate_count,
            "selected_candidate_index": board_pose.selected_candidate_index,
            "alternative_rms_px": board_pose.alternative_rms_px,
            "alternative_error_gap_px": board_pose.alternative_error_gap_px,
            "alternative_error_ratio": board_pose.alternative_error_ratio,
            "alternative_likelihood_ratio": board_pose.alternative_likelihood_ratio,
            "alternative_translation_delta_mm": board_pose.alternative_translation_delta_mm,
            "alternative_rotation_delta_deg": board_pose.alternative_rotation_delta_deg,
            "pose_ambiguous": board_pose.pose_ambiguous,
        },
    }


def write_board_pose_to_live_log(
    board_pose: BoardPoseCalibration,
    tracker_log=None,
) -> None:
    if tracker_log is None:
        tracker_log = _tracker_log_module()
    if not tracker_log.is_active():
        raise RuntimeError("Cannot write board pose: tracker log is not active.")
    tracker_log.write_record(
        board_pose_record(board_pose, tracker_log.current_run_id())
    )
    print("[debug_tracker_translation] wrote board_pose record to active run log")


def load_live_tracker_module():
    _ensure_src_on_path()
    try:
        from tracking.hydramarker.tests import test_tracker_realsense as live
    except Exception as exc:
        raise RuntimeError(
            "Could not import the RealSense tracker test module. "
            "Run this from the tracking project environment."
        ) from exc

    script_path = Path(__file__).resolve()
    live.tracker_log.set_run_log_dir(
        script_path.parents[1] / "tests" / "hydramarker_tracker_runs"
    )
    print(f"[debug_tracker_translation] live module -> {Path(live.__file__).resolve()}")
    return live


def run_live_tracker_translation() -> Path | None:
    live = load_live_tracker_module()
    tracker_log = live.tracker_log
    board_pose_ref: dict[str, BoardPoseCalibration | None] = {"pose": None}

    def after_camera_ready(pipe, K_rgb: np.ndarray, dist_rgb: np.ndarray) -> bool:
        camera_info = tracker_log.camera_intrinsics_info()
        tracker_log.update_camera_intrinsics_info(
            {
                "debug_rectification_mode": "disabled_raw_realsense",
                "debug_rectification_enabled": False,
                "debug_tracker_uses_loaded_camera_calibration": bool(
                    camera_info.get("camera_source") == "opencv_calibration_npz"
                ),
                "tracker_K": K_rgb.tolist(),
                "tracker_dist_coeffs": dist_rgb.reshape(-1).tolist(),
            }
        )

        board_pose = calibrate_checkerboard_pose(
            pipe,
            K_rgb,
            dist_rgb,
        )
        if board_pose is None:
            print("[debug_tracker_translation] checkerboard board pose was not confirmed")
            return False

        board_pose_ref["pose"] = board_pose
        tracker_log.set_debug_board_transform(board_pose.T_B_C)
        tracker_log.update_camera_intrinsics_info(
            {
                "debug_translation_reference_frame": "checkerboard",
                "debug_board_pose_available": True,
                "debug_board_T_B_C": board_pose.T_B_C.tolist(),
                "debug_board_reprojection_mean_px": board_pose.reproj_mean_px,
                "debug_board_reprojection_p95_px": board_pose.reproj_p95_px,
                "debug_board_solver_mode": board_pose.solver_mode,
                "debug_board_selected_candidate_index": board_pose.selected_candidate_index,
                "debug_board_pose_ambiguous": board_pose.pose_ambiguous,
                "debug_board_alternative_error_gap_px": board_pose.alternative_error_gap_px,
                "debug_board_alternative_likelihood_ratio": board_pose.alternative_likelihood_ratio,
            }
        )
        print(
            "[debug_tracker_translation] board pose fixed; "
            "subsequent frame diagnostics use checkerboard coordinates"
        )
        return True

    def after_tracker_created(tracker) -> None:
        _force_dense_refine_config(tracker)

    def on_log_open(_field_path, _marker_json_path, _tracker) -> None:
        board_pose = board_pose_ref["pose"]
        if board_pose is None:
            raise RuntimeError(
                "Translation debug recording started without a checkerboard board pose."
            )
        if not tracker_log.is_active():
            tracker_log.log_open(_field_path, _marker_json_path, _tracker)
        write_board_pose_to_live_log(board_pose, tracker_log)

    def draw_extra_overlay(vis: np.ndarray, log_active: bool) -> None:
        state_line = (
            "Board pose fixed | camera=selected_opencv_calibration | s=start/stop tracking | "
            f"SPACE={'STOP recording and analyze' if log_active else 'START recording'} | q=quit"
        )
        live.put_text(vis, state_line, (25, 245), color=(0, 255, 255), scale=0.48)

    def final_cleanup() -> None:
        tracker_log.set_debug_board_transform(None)

    return live.run_live_tracker(
        window_name="HydraTracker Translation Debug",
        console_prefix="[debug_tracker_translation]",
        after_camera_ready=after_camera_ready,
        after_tracker_created=after_tracker_created,
        on_log_open=on_log_open,
        draw_extra_overlay=draw_extra_overlay,
        stop_after_log_close=True,
        final_cleanup=final_cleanup,
    )


def _parse_board_pose_record(record: dict) -> dict[str, Any] | None:
    T_B_C = record.get("T_B_C")
    if T_B_C is None:
        return None
    try:
        return {
            "T_B_C": np.asarray(T_B_C, dtype=np.float64).reshape(4, 4),
            "T_C_B": np.asarray(record.get("T_C_B"), dtype=np.float64).reshape(4, 4)
            if record.get("T_C_B") is not None
            else None,
            "record": record,
        }
    except Exception:
        return None


def load_tracker_run(path: Path) -> dict:
    frames: list[int] = []
    success: list[int] = []
    pose_is_fresh: list[int] = []
    pose_source: list[str] = []
    tvec_camera: list[list[float]] = []
    logged_delta: list[float] = []

    run_id = path.stem
    run_timestamp = ""
    columns: list[str] = []
    summary: dict = {}
    board_pose: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at line {line_no}: {exc}") from exc

            record_type = record.get("type")

            if record_type == "run_start":
                run_id = str(record.get("run_id") or run_id)
                run_timestamp = str(record.get("timestamp") or "")
                columns = list(record.get("columns") or [])
                continue

            if record_type == "board_pose":
                parsed = _parse_board_pose_record(record)
                if parsed is not None:
                    board_pose = parsed
                continue

            if record_type == "run_summary":
                summary = dict(record.get("summary") or {})
                continue

            if record_type != "frame":
                continue

            data = record.get("data") or {}
            frames.append(_to_int(data.get("frame"), default=len(frames)))
            success.append(_to_int(data.get("success"), default=0))
            pose_is_fresh.append(_to_int(data.get("pose_is_fresh"), default=0))
            pose_source.append(str(data.get("pose_source") or ""))
            logged_delta.append(_to_float(data.get("pose_translation_delta_mm")))
            tvec_camera.append(
                [
                    _to_float(data.get("tvec_x_mm")),
                    _to_float(data.get("tvec_y_mm")),
                    _to_float(data.get("tvec_z_mm")),
                ]
            )

    if not frames:
        raise RuntimeError(f"No frame records found in:\n{path}")

    frame_arr = np.asarray(frames, dtype=np.int64)
    tvec_camera_arr = np.asarray(tvec_camera, dtype=np.float64).reshape(-1, 3)
    has_tvec = np.all(np.isfinite(tvec_camera_arr), axis=1)
    success_arr = np.asarray(success, dtype=np.int64)
    fresh_arr = np.asarray(pose_is_fresh, dtype=np.int64)

    if not np.any(has_tvec):
        available = ", ".join(columns) if columns else "unknown"
        raise RuntimeError(
            "This run log has no tvec_x_mm/tvec_y_mm/tvec_z_mm values.\n"
            "Record a new run with the updated tracker logger first.\n\n"
            f"Available columns:\n{available}"
        )

    coordinate_frame = "camera"
    absolute_label = "T_C_T"
    tvec_abs = tvec_camera_arr.copy()
    if board_pose is not None:
        T_B_C = np.asarray(board_pose["T_B_C"], dtype=np.float64).reshape(4, 4)
        tvec_abs = np.full_like(tvec_camera_arr, np.nan, dtype=np.float64)
        for idx, p_c in enumerate(tvec_camera_arr):
            if np.all(np.isfinite(p_c)):
                tvec_abs[idx] = _camera_tvec_to_board_mm(p_c, T_B_C)
        coordinate_frame = "checkerboard"
        absolute_label = "T_B_T"

    has_tvec = np.all(np.isfinite(tvec_abs), axis=1)
    origin_idx = int(np.where(has_tvec)[0][0])
    origin_frame = int(frame_arr[origin_idx])
    origin_tvec = tvec_abs[origin_idx].copy()
    relative_tvec = tvec_abs - origin_tvec
    z_vs_y_slope_mm_per_100mm = np.nan
    if coordinate_frame == "checkerboard":
        valid_yz = has_tvec & np.isfinite(relative_tvec[:, 1]) & np.isfinite(relative_tvec[:, 2])
        if int(np.count_nonzero(valid_yz)) >= 8 and np.ptp(relative_tvec[valid_yz, 1]) > 1e-6:
            A = np.c_[relative_tvec[valid_yz, 1], np.ones(int(np.count_nonzero(valid_yz)))]
            slope, _ = np.linalg.lstsq(A, relative_tvec[valid_yz, 2], rcond=None)[0]
            z_vs_y_slope_mm_per_100mm = float(100.0 * slope)

    computed_delta = np.full(len(frame_arr), np.nan, dtype=np.float64)
    for idx in range(1, len(frame_arr)):
        if has_tvec[idx] and has_tvec[idx - 1]:
            computed_delta[idx] = float(np.linalg.norm(relative_tvec[idx] - relative_tvec[idx - 1]))

    logged_delta_arr = np.asarray(logged_delta, dtype=np.float64)
    plot_delta = np.where(np.isfinite(computed_delta), computed_delta, logged_delta_arr)

    return {
        "path": path,
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "summary": summary,
        "frames": frame_arr,
        "success": success_arr,
        "pose_is_fresh": fresh_arr,
        "pose_source": np.asarray(pose_source, dtype=object),
        "tvec": relative_tvec,
        "tvec_abs": tvec_abs,
        "origin_frame": origin_frame,
        "origin_tvec": origin_tvec,
        "has_tvec": has_tvec,
        "delta_mm": plot_delta,
        "board_pose": board_pose,
        "coordinate_frame": coordinate_frame,
        "absolute_label": absolute_label,
        "z_vs_y_slope_mm_per_100mm": z_vs_y_slope_mm_per_100mm,
    }


def contiguous_frame_ranges(frames: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    selected = frames[mask]
    if len(selected) == 0:
        return []

    ranges: list[tuple[int, int]] = []
    start = int(selected[0])
    prev = int(selected[0])

    for frame in selected[1:]:
        frame = int(frame)
        if frame == prev + 1:
            prev = frame
            continue

        ranges.append((start, prev))
        start = prev = frame

    ranges.append((start, prev))
    return ranges


def robust_peak_frames(frames: np.ndarray, delta_mm: np.ndarray, limit: int = 8) -> tuple[np.ndarray, float]:
    valid = delta_mm[np.isfinite(delta_mm)]
    if len(valid) < 8:
        return np.asarray([], dtype=np.int64), np.nan

    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(
        median + 5.0 * robust_sigma,
        float(np.percentile(valid, 99)),
    )

    peak_indices = np.where(np.isfinite(delta_mm) & (delta_mm >= threshold))[0]
    if len(peak_indices) > limit:
        order = np.argsort(delta_mm[peak_indices])[::-1][:limit]
        peak_indices = np.sort(peak_indices[order])

    return frames[peak_indices], threshold


def setup_plot_style(plt) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfd",
            "axes.edgecolor": "#d0d4dc",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "grid.color": "#d9dee8",
            "grid.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d0d4dc",
        }
    )


def plot_translation(run: dict) -> Path:
    plt = _load_pyplot()
    setup_plot_style(plt)

    path: Path = run["path"]
    frames: np.ndarray = run["frames"]
    tvec: np.ndarray = run["tvec"]
    has_tvec: np.ndarray = run["has_tvec"]
    success: np.ndarray = run["success"]
    fresh: np.ndarray = run["pose_is_fresh"]
    delta_mm: np.ndarray = run["delta_mm"]
    absolute_label = str(run.get("absolute_label", "T_C_T"))
    coordinate_frame = str(run.get("coordinate_frame", "camera"))

    missing_mask = (success == 0) | ~has_tvec
    held_mask = (success == 1) & has_tvec & (fresh == 0)
    missing_ranges = contiguous_frame_ranges(frames, missing_mask)
    held_ranges = contiguous_frame_ranges(frames, held_mask)
    peak_frames, peak_threshold = robust_peak_frames(frames, delta_mm)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15.5, 9.0),
        sharex=True,
        constrained_layout=True,
    )

    run_label = run["run_id"]
    if run["run_timestamp"]:
        run_label += f"  |  {run['run_timestamp']}"

    pose_pct = 100.0 * float(np.count_nonzero(success == 1)) / max(len(success), 1)
    origin_frame = int(run["origin_frame"])
    origin_tvec = np.asarray(run["origin_tvec"], dtype=np.float64).reshape(3)

    title = (
        f"HydraTracker relative translation components ({absolute_label}, {coordinate_frame} frame)\n"
        f"{run_label}"
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for comp_idx, (_, label, color) in enumerate(COMPONENTS):
        ax = axes[comp_idx]
        values = tvec[:, comp_idx].copy()
        values[~has_tvec] = np.nan

        ax.plot(
            frames,
            values,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=2.8,
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=f"{absolute_label} {label}",
        )

        if np.any(np.isfinite(values)):
            first_value = float(values[np.where(np.isfinite(values))[0][0]])
            ax.axhline(first_value, color=color, alpha=0.22, linewidth=1.2, linestyle="--")

            value_range = float(np.nanmax(values) - np.nanmin(values))
            ax.set_title(f"{label.upper()} relative component   range={value_range:.2f} mm", loc="left")
        else:
            ax.set_title(f"{label.upper()} relative component", loc="left")

        for start, end in missing_ranges:
            ax.axvspan(start - 0.5, end + 0.5, color="#e45756", alpha=0.14, lw=0)

        for start, end in held_ranges:
            ax.axvspan(start - 0.5, end + 0.5, color="#f2b701", alpha=0.14, lw=0)

        for peak_frame in peak_frames:
            ax.axvline(int(peak_frame), color="#7f3c8d", alpha=0.38, linewidth=1.1)

        ax.set_ylabel(f"delta {absolute_label} {label} [mm]")
        ax.grid(True, axis="both")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("frame")

    info = (
        f"relative to frame {origin_frame} "
        f"({origin_tvec[0]:.2f}, {origin_tvec[1]:.2f}, {origin_tvec[2]:.2f}) mm   "
        f"frames={len(frames)}   pose={pose_pct:.1f}%   "
        f"missing={int(np.count_nonzero(missing_mask))}   held={int(np.count_nonzero(held_mask))}"
    )
    if np.isfinite(peak_threshold) and len(peak_frames) > 0:
        info += f"   peak threshold={peak_threshold:.2f} mm   peaks={', '.join(str(int(f)) for f in peak_frames)}"

    board_pose = run.get("board_pose")
    if board_pose is not None:
        reproj = (board_pose.get("record") or {}).get("reprojection") or {}
        if reproj:
            info += (
                f"   board reproj mean={float(reproj.get('mean_px', np.nan)):.3f}px"
                f" p95={float(reproj.get('p95_px', np.nan)):.3f}px"
            )
        z_vs_y = float(run.get("z_vs_y_slope_mm_per_100mm", np.nan))
        if np.isfinite(z_vs_y):
            info += f"   z~y slope={z_vs_y:.2f} mm/100mm"

    axes[0].text(
        0.01,
        1.02,
        info,
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#333333",
    )

    if missing_ranges:
        axes[0].text(
            0.99,
            1.02,
            "red = missing pose   yellow = held/stale pose   purple = large translation jump",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="#555555",
        )
    elif held_ranges or len(peak_frames) > 0:
        axes[0].text(
            0.99,
            1.02,
            "yellow = held/stale pose   purple = large translation jump",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="#555555",
        )

    suffix = "translation_board_relative_plot" if coordinate_frame == "checkerboard" else "translation_relative_plot"
    out_path = path.with_name(f"{path.stem}_{suffix}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[translation_plot] saved -> {out_path.resolve()}")
    plt.show()
    return out_path


def plot_existing_run(
    path: Path | None = None,
    *,
    require_board_pose: bool = True,
) -> Path | None:
    if path is None:
        path = select_jsonl_with_qt()

    if path is None:
        print("No file selected.")
        return None

    run = load_tracker_run(path)
    if require_board_pose and run.get("board_pose") is None:
        raise RuntimeError(
            "This run has no board_pose record, so it cannot be plotted as the "
            "translation debug checkerboard-frame run.\n"
            "Run hydramarker/debug/debug_tracker_translation.py live and confirm "
            "the checkerboard pose before recording."
        )
    return plot_translation(run)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] in ("--plot", "plot"):
        path = Path(args[1]) if len(args) > 1 else None
        plot_existing_run(path)
        return

    if args and args[0] in ("--plot-camera", "plot-camera"):
        path = Path(args[1]) if len(args) > 1 else None
        plot_existing_run(path, require_board_pose=False)
        return

    if args and args[0] in ("--select", "select"):
        plot_existing_run(None)
        return

    if args and Path(args[0]).suffix.lower() == ".jsonl":
        plot_existing_run(Path(args[0]))
        return

    recorded_path = run_live_tracker_translation()
    if recorded_path is None:
        print("No recording to analyze.")
        return

    plot_existing_run(recorded_path)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[debug_tracker_translation] ERROR: {exc}")
        sys.exit(1)
