from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
import numpy as np

os.environ.setdefault("QT_API", "pyside6")


COMPONENTS = (
    ("x", "x", "#1f77b4"),
    ("y", "y", "#2ca02c"),
    ("z", "z", "#d62728"),
)

DEPTH_FILTER_COLUMNS = (
    "depth_filter_applied",
    "depth_filter_delta_z_mm",
    "depth_filter_raw_z_mm",
    "depth_filter_z_mm",
    "depth_filter_reproj_excess_px",
    "depth_filter_guard_alpha",
    "depth_filter_innovation_z_mm",
    "depth_filter_innovation_mean_z_mm",
    "depth_filter_innovation_cusum_pos_mm",
    "depth_filter_innovation_cusum_neg_mm",
    "depth_filter_innovation_bias_detected",
    "depth_filter_innovation_bias_direction",
    "depth_filter_innovation_bias_limited",
    "depth_filter_object_z_span_mm",
    "depth_filter_negative_delta_guard_limited",
)


def _ensure_src_on_path() -> None:
    tracking_root = Path(__file__).resolve().parents[2]
    src_root = tracking_root.parent
    src = str(src_root)
    sys.path = [p for p in sys.path if str(p) != src]
    sys.path.insert(0, src)


_ensure_src_on_path()

from tracking.hydramarker.calib import calib_camera, calib_checkerboard


BoardPoseCalibration = calib_checkerboard.CharucoTableCalibration
BOARD_AXIS_LENGTH_MM = 80.0


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
        "fast_persistent_dense_adaptive_refine_enabled": True,
        "fast_persistent_dense_adaptive_min_match_ratio": 0.85,
        "fast_persistent_dense_adaptive_motion_px": 8.0,
        "fast_persistent_dense_adaptive_max_seed_mean_px": 1.2,
        "fast_persistent_dense_adaptive_max_seed_max_px": 2.8,
    }
    peak_guard_settings = {
        "max_translation_jump_mm": 40.0,
        "max_rotation_jump_deg": 45.0,
        "rotation_gate_scale_per_lost_frame": 8.0,
        "rotation_gate_max_deg": 90.0,
        "decode_update_min_visual_corners": 12,
        "decode_update_min_distinct_rows": 3,
        "decode_update_min_distinct_cols": 3,
    }
    pose_cfg = getattr(getattr(tracker, "pose_tracker", None), "config", None)
    for name, value in {**dense_settings, **peak_guard_settings}.items():
        setattr(cfg, name, value)
        if pose_cfg is not None and hasattr(pose_cfg, name):
            setattr(pose_cfg, name, value)

    print(
        "[debug_tracker_translation] dense refine enabled "
        f"(min_points={getattr(cfg, 'fast_persistent_dense_min_points', '?')}, "
        f"max_px={getattr(cfg, 'fast_persistent_dense_match_max_px', '?')}, "
        f"solver={getattr(cfg, 'fast_persistent_dense_pose_solver', '?')}, "
        f"adaptive={getattr(cfg, 'fast_persistent_dense_adaptive_refine_enabled', '?')}, "
        f"checker_refresh={getattr(cfg, 'checker_refresh_interval_frames', '?')}/"
        f"{getattr(cfg, 'checker_tracking_recovery_stable_interval_frames', '?')}, "
        f"max_jump={getattr(cfg, 'max_translation_jump_mm', '?')}mm, "
        f"max_rot={getattr(cfg, 'rotation_gate_max_deg', '?')}deg)"
    )


def _load_qt_widgets():
    from PySide6.QtWidgets import QApplication, QFileDialog

    return QApplication, QFileDialog


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


def _finite_float(value) -> float | None:
    parsed = _to_float(value)
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def _timing_profile_from_frame_data(data: dict) -> dict[str, float]:
    timings: dict[str, float] = {}

    profile = data.get("timing_profile_ms")
    if isinstance(profile, dict):
        for key, value in profile.items():
            parsed = _finite_float(value)
            if parsed is not None:
                timings[str(key)] = parsed

    for key, value in data.items():
        if key == "timing_profile_ms":
            continue
        if not (str(key).endswith("_ms") or key == "wall_ms"):
            continue
        parsed = _finite_float(value)
        if parsed is not None:
            timings[str(key)] = parsed

    return timings


def _component_spread_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "var": np.nan,
            "rms": np.nan,
            "p95_abs": np.nan,
            "p95_low": np.nan,
            "p95_high": np.nan,
        }

    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "var": float(np.var(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "p95_abs": float(np.percentile(np.abs(finite), 95.0)),
        "p95_low": float(np.percentile(finite, 2.5)),
        "p95_high": float(np.percentile(finite, 97.5)),
    }


def timing_profile_summary(run: dict) -> list[dict[str, float | int | str]]:
    series = dict(run.get("timing_series") or {})
    rows: list[dict[str, float | int | str]] = []
    for key in sorted(series):
        values = np.asarray(series[key], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        rows.append(
            {
                "timing": key,
                "count": int(len(finite)),
                "mean_ms": float(np.mean(finite)),
                "median_ms": float(np.median(finite)),
                "p95_ms": float(np.percentile(finite, 95.0)),
                "max_ms": float(np.max(finite)),
                "sum_ms": float(np.sum(finite)),
            }
        )
    rows.sort(key=lambda row: float(row["sum_ms"]), reverse=True)
    return rows


def write_timing_profile_summary(run: dict) -> Path | None:
    rows = timing_profile_summary(run)
    if not rows:
        return None

    path: Path = run["path"]
    out_path = path.with_name(f"{path.stem}_timing_profile_summary.csv")
    fieldnames = ["timing", "count", "mean_ms", "median_ms", "p95_ms", "max_ms", "sum_ms"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timing": row["timing"],
                    "count": row["count"],
                    "mean_ms": f"{float(row['mean_ms']):.4f}",
                    "median_ms": f"{float(row['median_ms']):.4f}",
                    "p95_ms": f"{float(row['p95_ms']):.4f}",
                    "max_ms": f"{float(row['max_ms']):.4f}",
                    "sum_ms": f"{float(row['sum_ms']):.4f}",
                }
            )
    print(f"[timing_profile] saved -> {out_path.resolve()}")
    return out_path


def print_timing_profile_summary(run: dict, limit: int = 12) -> None:
    rows = timing_profile_summary(run)
    if not rows:
        return
    print("[timing_profile] slowest accumulated timings:")
    for row in rows[:limit]:
        print(
            "  "
            f"{row['timing']}: "
            f"mean={float(row['mean_ms']):.3f} ms, "
            f"p95={float(row['p95_ms']):.3f} ms, "
            f"max={float(row['max_ms']):.3f} ms, "
            f"sum={float(row['sum_ms']):.1f} ms, "
            f"n={int(row['count'])}"
        )


def _qt_app():
    QApplication, _ = _load_qt_widgets()
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def select_jsonl_with_qt() -> Path | None:
    _qt_app()
    _, QFileDialog = _load_qt_widgets()

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


def _camera_from_run_start(record: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    info = dict(record.get("camera_intrinsics") or {})
    K_value = (
        info.get("tracker_K")
        or info.get("K")
        or info.get("camera_matrix")
        or info.get("camera_intrinsics")
    )
    if K_value is None:
        fx = _to_float(info.get("fx"))
        fy = _to_float(info.get("fy"))
        cx = _to_float(info.get("ppx", info.get("cx")))
        cy = _to_float(info.get("ppy", info.get("cy")))
        if np.all(np.isfinite([fx, fy, cx, cy])):
            K_value = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    dist_value = (
        info.get("tracker_dist_coeffs")
        or info.get("effective_opencv_dist_coeffs")
        or info.get("opencv_dist_coeffs")
        or info.get("dist_coeffs")
        or info.get("coeffs")
    )

    K = None if K_value is None else np.asarray(K_value, dtype=np.float64).reshape(3, 3)
    dist = (
        np.zeros((5, 1), dtype=np.float64)
        if dist_value is None
        else np.asarray(dist_value, dtype=np.float64).reshape(-1, 1)
    )
    return K, dist


def _pose_from_frame_data(data: dict) -> tuple[np.ndarray, np.ndarray] | None:
    rvec = np.asarray(
        [
            _to_float(data.get("rvec_x_rad")),
            _to_float(data.get("rvec_y_rad")),
            _to_float(data.get("rvec_z_rad")),
        ],
        dtype=np.float64,
    ).reshape(3, 1)
    tvec = np.asarray(
        [
            _to_float(data.get("tvec_x_mm")),
            _to_float(data.get("tvec_y_mm")),
            _to_float(data.get("tvec_z_mm")),
        ],
        dtype=np.float64,
    ).reshape(3, 1)
    if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        return None
    return rvec, tvec


def _points_from_frame_detail(detail: dict) -> tuple[np.ndarray, np.ndarray]:
    corners = list(detail.get("pose_corners") or [])
    if not corners:
        corners = list(detail.get("correspondence_corners") or [])

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for corner in corners:
        if not isinstance(corner, dict):
            continue
        xyz = corner.get("xyz_mm")
        uv = corner.get("uv_px")
        if not isinstance(xyz, (list, tuple)) or not isinstance(uv, (list, tuple)):
            continue
        if len(xyz) < 3 or len(uv) < 2:
            continue
        xyz_arr = np.asarray([_to_float(v) for v in xyz[:3]], dtype=np.float64)
        uv_arr = np.asarray([_to_float(v) for v in uv[:2]], dtype=np.float64)
        if np.all(np.isfinite(xyz_arr)) and np.all(np.isfinite(uv_arr)):
            object_points.append(xyz_arr.reshape(3))
            image_points.append(uv_arr.reshape(2))

    if not object_points:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )
    return (
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
    )


def _fmt_debug_float(value: float, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _reprojection_stats(
    *,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    object_points: np.ndarray,
    image_points: np.ndarray,
) -> dict[str, str]:
    if len(object_points) == 0:
        return {}
    try:
        import cv2

        projected, _ = cv2.projectPoints(
            object_points.reshape(-1, 3),
            rvec.reshape(3, 1),
            tvec.reshape(3, 1),
            K,
            dist.reshape(-1, 1),
        )
    except Exception:
        return {}

    residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    err = np.linalg.norm(residual, axis=1)
    return {
        "mean_err": _fmt_debug_float(float(np.mean(err))),
        "max_err": _fmt_debug_float(float(np.max(err))),
        "pose_reproj_mean_px": _fmt_debug_float(float(np.mean(err))),
        "pose_reproj_median_px": _fmt_debug_float(float(np.median(err))),
        "pose_reproj_p95_px": _fmt_debug_float(float(np.percentile(err, 95))),
        "pose_reproj_max_px": _fmt_debug_float(float(np.max(err))),
        "pose_reproj_mean_du_px": _fmt_debug_float(float(np.mean(residual[:, 0]))),
        "pose_reproj_mean_dv_px": _fmt_debug_float(float(np.mean(residual[:, 1]))),
        "pose_reproj_std_du_px": _fmt_debug_float(float(np.std(residual[:, 0]))),
        "pose_reproj_std_dv_px": _fmt_debug_float(float(np.std(residual[:, 1]))),
    }


def _ensure_columns(columns: list[str], extra_columns: tuple[str, ...]) -> list[str]:
    out = list(columns)
    for column in extra_columns:
        if column not in out:
            out.append(column)
    return out


def replay_depth_filter_on_run(path: Path, output_path: Path | None = None) -> Path:
    _ensure_src_on_path()
    from tracking.hydramarker.config import TrackerConfig
    from tracking.pose_filters import PoseDepthKalmanFilter

    path = Path(path)
    if output_path is None:
        output_path = path.with_name(f"{path.stem}_depth_filter_replay.jsonl")

    records: list[dict] = []
    details_by_frame: dict[int, dict] = {}
    K: np.ndarray | None = None
    dist: np.ndarray | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at line {line_no}: {exc}") from exc

            if record.get("type") == "run_start":
                K, dist = _camera_from_run_start(record)
            elif record.get("type") == "frame_detail":
                frame = _to_int(record.get("frame"), default=-1)
                if frame >= 0:
                    details_by_frame[frame] = record
            records.append(record)

    if K is None or dist is None:
        raise RuntimeError("No camera intrinsics found in run_start record.")

    cfg = TrackerConfig()
    depth_filter = PoseDepthKalmanFilter(
        observation_std_mm=float(cfg.pose_depth_filter_observation_std_mm),
        process_std_mm=float(cfg.pose_depth_filter_process_std_mm),
        initial_velocity_std_mm=float(cfg.pose_depth_filter_initial_velocity_std_mm),
        reprojection_guard_px=float(cfg.pose_depth_filter_reprojection_guard_px),
        K=K,
        dist_coeffs=dist,
        innovation_guard_enabled=bool(cfg.pose_depth_filter_innovation_guard_enabled),
        innovation_guard_window=int(cfg.pose_depth_filter_innovation_window),
        innovation_guard_bias_threshold_mm=float(
            cfg.pose_depth_filter_innovation_bias_threshold_mm
        ),
        innovation_guard_min_same_sign=int(cfg.pose_depth_filter_innovation_min_same_sign),
        innovation_cusum_slack_mm=float(cfg.pose_depth_filter_innovation_cusum_slack_mm),
        innovation_cusum_threshold_mm=float(cfg.pose_depth_filter_innovation_cusum_threshold_mm),
        negative_delta_guard_enabled=bool(
            cfg.pose_depth_filter_negative_delta_guard_enabled
        ),
        negative_delta_guard_min_z_span_mm=float(
            cfg.pose_depth_filter_negative_delta_guard_min_z_span_mm
        ),
        negative_delta_guard_max_negative_delta_mm=float(
            cfg.pose_depth_filter_negative_delta_guard_max_negative_delta_mm
        ),
        negative_delta_guard_hold_previous_z=bool(
            cfg.pose_depth_filter_negative_delta_guard_hold_previous_z
        ),
        negative_delta_guard_hold_requires_innovation_bias=bool(
            cfg.pose_depth_filter_negative_delta_guard_hold_requires_innovation_bias
        ),
        negative_delta_guard_hold_min_negative_delta_mm=float(
            cfg.pose_depth_filter_negative_delta_guard_hold_min_negative_delta_mm
        ),
        negative_delta_guard_max_hold_correction_mm=float(
            cfg.pose_depth_filter_negative_delta_guard_max_hold_correction_mm
        ),
        negative_delta_guard_velocity_damping=float(
            cfg.pose_depth_filter_negative_delta_guard_velocity_damping
        ),
    )
    min_points = max(1, int(cfg.pose_depth_filter_min_points))

    applied_count = 0
    filtered_count = 0
    skipped_count = 0
    previous_filtered_tvec: np.ndarray | None = None
    lost_frames = 0
    max_lost_frames = int(getattr(cfg, "max_lost_frames", 8))

    for record in records:
        record_type = record.get("type")
        if record_type == "run_start":
            columns = list(record.get("columns") or [])
            record["columns"] = _ensure_columns(columns, DEPTH_FILTER_COLUMNS)
            config = dict(record.get("config") or {})
            config.update(
                {
                    "pose_depth_filter_enabled": True,
                    "pose_depth_filter_observation_std_mm": float(
                        cfg.pose_depth_filter_observation_std_mm
                    ),
                    "pose_depth_filter_process_std_mm": float(
                        cfg.pose_depth_filter_process_std_mm
                    ),
                    "pose_depth_filter_initial_velocity_std_mm": float(
                        cfg.pose_depth_filter_initial_velocity_std_mm
                    ),
                    "pose_depth_filter_reprojection_guard_px": float(
                        cfg.pose_depth_filter_reprojection_guard_px
                    ),
                    "pose_depth_filter_min_points": int(cfg.pose_depth_filter_min_points),
                    "pose_depth_filter_innovation_guard_enabled": bool(
                        cfg.pose_depth_filter_innovation_guard_enabled
                    ),
                    "pose_depth_filter_innovation_window": int(
                        cfg.pose_depth_filter_innovation_window
                    ),
                    "pose_depth_filter_innovation_bias_threshold_mm": float(
                        cfg.pose_depth_filter_innovation_bias_threshold_mm
                    ),
                    "pose_depth_filter_innovation_min_same_sign": int(
                        cfg.pose_depth_filter_innovation_min_same_sign
                    ),
                    "pose_depth_filter_innovation_cusum_slack_mm": float(
                        cfg.pose_depth_filter_innovation_cusum_slack_mm
                    ),
                    "pose_depth_filter_innovation_cusum_threshold_mm": float(
                        cfg.pose_depth_filter_innovation_cusum_threshold_mm
                    ),
                    "pose_depth_filter_negative_delta_guard_enabled": bool(
                        cfg.pose_depth_filter_negative_delta_guard_enabled
                    ),
                    "pose_depth_filter_negative_delta_guard_min_z_span_mm": float(
                        cfg.pose_depth_filter_negative_delta_guard_min_z_span_mm
                    ),
                    "pose_depth_filter_negative_delta_guard_max_negative_delta_mm": float(
                        cfg.pose_depth_filter_negative_delta_guard_max_negative_delta_mm
                    ),
                    "pose_depth_filter_negative_delta_guard_hold_previous_z": bool(
                        cfg.pose_depth_filter_negative_delta_guard_hold_previous_z
                    ),
                    "pose_depth_filter_negative_delta_guard_hold_requires_innovation_bias": bool(
                        cfg.pose_depth_filter_negative_delta_guard_hold_requires_innovation_bias
                    ),
                    "pose_depth_filter_negative_delta_guard_hold_min_negative_delta_mm": float(
                        cfg.pose_depth_filter_negative_delta_guard_hold_min_negative_delta_mm
                    ),
                    "pose_depth_filter_negative_delta_guard_max_hold_correction_mm": float(
                        cfg.pose_depth_filter_negative_delta_guard_max_hold_correction_mm
                    ),
                    "pose_depth_filter_negative_delta_guard_velocity_damping": float(
                        cfg.pose_depth_filter_negative_delta_guard_velocity_damping
                    ),
                    "offline_depth_filter_replay": True,
                }
            )
            record["config"] = config
            continue

        if record_type != "frame":
            continue

        data = record.get("data") or {}
        if _to_int(data.get("success"), default=0) == 0:
            lost_frames += 1
            previous_filtered_tvec = None
            if lost_frames > max_lost_frames:
                depth_filter.reset()
            continue
        lost_frames = 0

        frame = _to_int(data.get("frame"), default=-1)
        pose = _pose_from_frame_data(data)
        detail = details_by_frame.get(frame, {})
        object_points, image_points = _points_from_frame_detail(detail)
        if pose is None or len(object_points) < min_points:
            skipped_count += 1
            continue

        rvec, tvec = pose
        filtered = depth_filter.update(
            rvec=rvec,
            tvec=tvec,
            object_points=object_points,
            image_points=image_points,
        )
        filtered_count += 1
        applied_count += int(bool(filtered.applied))

        out_tvec = filtered.tvec.reshape(3, 1)
        data["tvec_z_mm"] = _fmt_debug_float(float(out_tvec[2, 0]))
        data["depth_filter_applied"] = int(bool(filtered.applied))
        data["depth_filter_delta_z_mm"] = _fmt_debug_float(float(filtered.delta_z_mm))
        data["depth_filter_raw_z_mm"] = _fmt_debug_float(float(filtered.raw_z_mm))
        data["depth_filter_z_mm"] = _fmt_debug_float(float(filtered.filtered_z_mm))
        data["depth_filter_reproj_excess_px"] = _fmt_debug_float(
            float(filtered.reprojection_excess_px)
        )
        data["depth_filter_guard_alpha"] = _fmt_debug_float(float(filtered.guard_alpha), digits=6)
        data["depth_filter_innovation_z_mm"] = _fmt_debug_float(float(filtered.innovation_z_mm))
        data["depth_filter_innovation_mean_z_mm"] = _fmt_debug_float(
            float(filtered.innovation_mean_z_mm)
        )
        data["depth_filter_innovation_cusum_pos_mm"] = _fmt_debug_float(
            float(filtered.innovation_cusum_pos_mm)
        )
        data["depth_filter_innovation_cusum_neg_mm"] = _fmt_debug_float(
            float(filtered.innovation_cusum_neg_mm)
        )
        data["depth_filter_innovation_bias_detected"] = int(
            bool(filtered.innovation_bias_detected)
        )
        data["depth_filter_innovation_bias_direction"] = int(
            filtered.innovation_bias_direction
        )
        data["depth_filter_innovation_bias_limited"] = int(
            bool(filtered.innovation_bias_limited)
        )
        data["depth_filter_object_z_span_mm"] = _fmt_debug_float(
            float(filtered.object_z_span_mm)
        )
        data["depth_filter_negative_delta_guard_limited"] = int(
            bool(filtered.negative_delta_guard_limited)
        )
        data.update(
            _reprojection_stats(
                K=K,
                dist=dist,
                rvec=filtered.rvec,
                tvec=out_tvec,
                object_points=object_points,
                image_points=image_points,
            )
        )

        if previous_filtered_tvec is None:
            data["pose_translation_delta_mm"] = ""
        else:
            delta = float(np.linalg.norm(out_tvec.reshape(3) - previous_filtered_tvec.reshape(3)))
            data["pose_translation_delta_mm"] = _fmt_debug_float(delta)
        previous_filtered_tvec = out_tvec.copy()

    for record in records:
        if record.get("type") == "run_summary":
            summary = dict(record.get("summary") or {})
            summary.update(
                {
                    "offline_depth_filter_replay": True,
                    "depth_filter_replayed_frames": filtered_count,
                    "depth_filter_applied_frames": applied_count,
                    "depth_filter_skipped_frames": skipped_count,
                }
            )
            record["summary"] = summary

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        "[depth_filter_replay] "
        f"frames={filtered_count}, applied={applied_count}, skipped={skipped_count}"
    )
    print(f"[depth_filter_replay] wrote -> {output_path.resolve()}")
    return output_path


def _make_pose_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    import cv2

    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return out


def _camera_tvec_to_board_mm(tvec_camera_mm: np.ndarray, T_B_C: np.ndarray) -> np.ndarray:
    p_c = np.ones(4, dtype=np.float64)
    p_c[:3] = np.asarray(tvec_camera_mm, dtype=np.float64).reshape(3)
    p_b = np.asarray(T_B_C, dtype=np.float64).reshape(4, 4) @ p_c
    return p_b[:3]


def _board_tvec_from_result(result, T_B_C: np.ndarray) -> np.ndarray | None:
    rvec = getattr(result, "rvec", None)
    tvec = getattr(result, "tvec", None)
    if rvec is None or tvec is None:
        return None
    try:
        T_B_T = np.asarray(T_B_C, dtype=np.float64).reshape(4, 4) @ _make_pose_matrix(
            rvec,
            tvec,
        )
        return np.asarray(T_B_T[:3, 3], dtype=np.float64).reshape(3)
    except Exception:
        return None


def table_calibration_record(table_pose: BoardPoseCalibration, run_id: str) -> dict:
    quality = calib_checkerboard.pose_quality_dict(table_pose)
    normal = np.asarray(table_pose.normal_camera, dtype=np.float64).reshape(3)
    return {
        "type": "table_calibration",
        "run_id": run_id,
        "coordinate_frame": "charuco_table",
        "board": {
            "kind": "charuco",
            "squares_x": int(calib_camera.SQUARES_X),
            "squares_y": int(calib_camera.SQUARES_Y),
            "square_length_mm": float(table_pose.square_length_mm),
            "marker_length_mm": float(table_pose.marker_length_mm),
            "aruco_dictionary_id": int(table_pose.aruco_dictionary_id),
        },
        "T_C_B": np.asarray(table_pose.T_C_B, dtype=np.float64).tolist(),
        "T_B_C": np.asarray(table_pose.T_B_C, dtype=np.float64).tolist(),
        "normal_camera": normal.tolist(),
        "x_axis_camera": np.asarray(table_pose.x_axis_camera, dtype=np.float64).tolist(),
        "y_axis_camera": np.asarray(table_pose.y_axis_camera, dtype=np.float64).tolist(),
        "source_position_index": int(table_pose.source_position_index),
        "positions_used": int(table_pose.positions_used),
        "raw_observations_available": True,
        "raw_observations_record_type": "table_calibration_observations",
        "quality": quality,
    }


def table_calibration_observation_records(
    table_pose: BoardPoseCalibration,
    run_id: str,
) -> list[dict]:
    records: list[dict] = []
    for position in table_pose.positions:
        frame_mask = np.asarray(position.frame_mask, dtype=bool).reshape(-1)
        frames: list[dict] = []
        for idx, frame in enumerate(position.frames):
            used = bool(frame_mask[idx]) if idx < frame_mask.size else False
            frames.append(
                {
                    "frame_index": int(frame.frame_index),
                    "used": used,
                    "num_charuco": int(frame.num_charuco),
                    "num_aruco": int(frame.num_aruco),
                    "rvec_cb": np.asarray(frame.rvec_cb, dtype=np.float64)
                    .reshape(3)
                    .tolist(),
                    "tvec_cb_mm": np.asarray(frame.tvec_cb_mm, dtype=np.float64)
                    .reshape(3)
                    .tolist(),
                    "normal_camera": np.asarray(frame.normal_camera, dtype=np.float64)
                    .reshape(3)
                    .tolist(),
                    "x_axis_camera": np.asarray(frame.x_axis_camera, dtype=np.float64)
                    .reshape(3)
                    .tolist(),
                    "origin_camera_mm": np.asarray(frame.origin_camera_mm, dtype=np.float64)
                    .reshape(3)
                    .tolist(),
                    "object_points_mm": np.asarray(
                        frame.object_points_mm,
                        dtype=np.float64,
                    )
                    .reshape(-1, 3)
                    .tolist(),
                    "image_points_uv": np.asarray(
                        frame.image_points_uv,
                        dtype=np.float64,
                    )
                    .reshape(-1, 2)
                    .tolist(),
                    "errors_px": np.asarray(frame.errors_px, dtype=np.float64)
                    .reshape(-1)
                    .tolist(),
                    "rms_px": float(frame.rms_px),
                    "mean_px": float(frame.mean_px),
                    "p95_px": float(frame.p95_px),
                    "max_px": float(frame.max_px),
                }
            )

        records.append(
            {
                "type": "table_calibration_observations",
                "run_id": run_id,
                "coordinate_frame": "charuco_table",
                "position_index": int(position.position_index),
                "frame_mask": frame_mask.tolist(),
                "frames_total": int(len(position.frames)),
                "frames_used": int(np.count_nonzero(frame_mask)),
                "median_rms_px": float(position.median_rms_px),
                "normal_camera": np.asarray(position.normal_camera, dtype=np.float64)
                .reshape(3)
                .tolist(),
                "x_axis_camera": np.asarray(position.x_axis_camera, dtype=np.float64)
                .reshape(3)
                .tolist(),
                "origin_camera_mm": np.asarray(position.origin_camera_mm, dtype=np.float64)
                .reshape(3)
                .tolist(),
                "frames": frames,
            }
        )
    return records


def table_origin_record(origin_tvec_board_mm: np.ndarray, run_id: str, frame_idx: int) -> dict:
    return {
        "type": "table_origin",
        "run_id": run_id,
        "frame": int(frame_idx),
        "origin_tvec_board_mm": np.asarray(
            origin_tvec_board_mm,
            dtype=np.float64,
        ).reshape(3).tolist(),
    }


def _draw_table_axes_near_result(
    vis: np.ndarray,
    result,
    *,
    K: np.ndarray,
    dist: np.ndarray,
    T_C_B: np.ndarray,
    T_B_C: np.ndarray,
    axis_length_mm: float = BOARD_AXIS_LENGTH_MM,
) -> None:
    import cv2

    anchor_board = None
    if getattr(result, "success", False):
        anchor_board = _board_tvec_from_result(result, T_B_C)
    if anchor_board is None:
        anchor_board = np.zeros(3, dtype=np.float64)
    anchor_board = np.asarray(anchor_board, dtype=np.float64).reshape(3)
    anchor_board[2] = 0.0

    points_board = np.asarray(
        [
            anchor_board,
            anchor_board + np.asarray([axis_length_mm, 0.0, 0.0], dtype=np.float64),
            anchor_board + np.asarray([0.0, axis_length_mm, 0.0], dtype=np.float64),
        ],
        dtype=np.float64,
    )
    points_h = np.c_[points_board, np.ones(3, dtype=np.float64)]
    points_camera = (np.asarray(T_C_B, dtype=np.float64).reshape(4, 4) @ points_h.T).T[:, :3]
    if np.any(points_camera[:, 2] <= 1e-6):
        return

    projected, _ = cv2.projectPoints(
        points_camera.reshape(-1, 3),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        np.asarray(dist, dtype=np.float64).reshape(-1, 1),
    )
    pts = projected.reshape(-1, 2)
    h, w = vis.shape[:2]

    def pt(idx: int) -> tuple[int, int]:
        x = int(round(float(np.clip(pts[idx, 0], -2000, w + 2000))))
        y = int(round(float(np.clip(pts[idx, 1], -2000, h + 2000))))
        return x, y

    origin = pt(0)
    x_end = pt(1)
    y_end = pt(2)
    cv2.arrowedLine(vis, origin, x_end, (0, 80, 255), 4, cv2.LINE_AA, tipLength=0.18)
    cv2.arrowedLine(vis, origin, y_end, (0, 255, 0), 4, cv2.LINE_AA, tipLength=0.18)
    cv2.putText(vis, "+X", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, "+Y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3, cv2.LINE_AA)


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


def create_color_only_realsense_pipeline(live):
    rs = live.rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    print("[debug_tracker_translation] RealSense running (color=1920x1080@30, no IMU)")
    return pipe, profile


def _apply_live_debug_options(
    tracker,
    *,
    no_fast_persistent: bool = False,
    no_temporal_persistence: bool = False,
    decode_only: bool = False,
) -> None:
    cfg = getattr(tracker, "config", None)
    if cfg is None:
        return

    if decode_only:
        cfg.decode_only_mode = True
        cfg.enable_fast_persistent_path = False
        cfg.fast_persistent_dense_refine_enabled = False
        cfg.enable_temporal_correspondence_persistence = False
        print("[debug_tracker_translation] decode-only mode enabled")
        return

    if no_fast_persistent:
        cfg.enable_fast_persistent_path = False
        cfg.fast_persistent_dense_refine_enabled = False
        print("[debug_tracker_translation] fast persistent path disabled")

    if no_temporal_persistence:
        cfg.enable_temporal_correspondence_persistence = False
        print("[debug_tracker_translation] temporal correspondence persistence disabled")


def run_live_tracker_translation(
    *,
    no_fast_persistent: bool = False,
    no_temporal_persistence: bool = False,
    decode_only: bool = False,
) -> list[Path]:
    live = load_live_tracker_module()
    live.create_realsense_pipeline = lambda: create_color_only_realsense_pipeline(live)
    tracker_log = live.tracker_log
    table_ref: dict[str, BoardPoseCalibration | None] = {"pose": None}
    origin_ref: dict[str, np.ndarray | None] = {"tvec": None}
    origin_frame_ref: dict[str, int] = {"frame": -1}
    camera_ref: dict[str, np.ndarray | None] = {"K": None, "dist": None}
    recorded_paths: list[Path] = []

    def after_camera_ready(pipe, K_rgb: np.ndarray, dist_rgb: np.ndarray) -> bool:
        camera_info = tracker_log.camera_intrinsics_info()
        camera_ref["K"] = np.asarray(K_rgb, dtype=np.float64).reshape(3, 3).copy()
        camera_ref["dist"] = np.asarray(dist_rgb, dtype=np.float64).reshape(-1, 1).copy()
        print("[debug_tracker_translation] starting ChArUco table calibration")
        table_pose = calib_checkerboard.capture_charuco_table_calibration_from_pipeline(
            pipe,
            K_rgb,
            dist_rgb,
            min_positions=calib_checkerboard.CHARUCO_TABLE_MIN_POSITIONS,
            target_positions=calib_checkerboard.CHARUCO_TABLE_TARGET_POSITIONS,
            min_frames_per_position=calib_checkerboard.CHARUCO_TABLE_MIN_FRAMES_PER_POSITION,
            max_frames_per_position=calib_checkerboard.CHARUCO_TABLE_MAX_FRAMES_PER_POSITION,
            window_name="HydraTracker ChArUco Table Calibration",
        )
        if table_pose is None:
            print("[debug_tracker_translation] ChArUco table calibration cancelled")
            return False

        table_ref["pose"] = table_pose
        tracker_log.set_debug_board_transform(table_pose.T_B_C)
        quality = calib_checkerboard.pose_quality_dict(table_pose)
        print(
            "[debug_tracker_translation] ChArUco table ready "
            f"(positions={table_pose.positions_used}, "
            f"frames={quality['frames_used']}/{quality['frames_total']}, "
            f"mean={quality['corner_error_mean_px']:.3f}px, "
            f"p95={quality['corner_error_p95_px']:.3f}px)"
        )

        tracker_log.update_camera_intrinsics_info(
            {
                "debug_rectification_mode": "disabled_raw_realsense",
                "debug_rectification_enabled": False,
                "debug_tracker_uses_loaded_camera_calibration": bool(
                    camera_info.get("camera_source") == "opencv_calibration_npz"
                ),
                "tracker_K": K_rgb.tolist(),
                "tracker_dist_coeffs": dist_rgb.reshape(-1).tolist(),
                "debug_translation_reference_frame": "charuco_table",
                "debug_table_pose_available": True,
                "debug_table_T_B_C": table_pose.T_B_C.tolist(),
                "debug_table_T_C_B": table_pose.T_C_B.tolist(),
                "debug_table_normal_camera": table_pose.normal_camera.tolist(),
                "debug_table_x_axis_camera": table_pose.x_axis_camera.tolist(),
                "debug_table_y_axis_camera": table_pose.y_axis_camera.tolist(),
                "debug_table_positions_used": int(table_pose.positions_used),
                "debug_table_frames_used": int(quality["frames_used"]),
                "debug_table_reprojection_mean_px": float(quality["corner_error_mean_px"]),
                "debug_table_reprojection_p95_px": float(quality["corner_error_p95_px"]),
                "debug_corner_refinement": "gradient_saddle_cornerSubPix",
            }
        )
        print("[debug_tracker_translation] table-frame translation debug ready")
        return True

    def after_tracker_created(tracker) -> None:
        _force_dense_refine_config(tracker)
        _apply_live_debug_options(
            tracker,
            no_fast_persistent=no_fast_persistent,
            no_temporal_persistence=no_temporal_persistence,
            decode_only=decode_only,
        )

    def on_space_key(_tracker, result, frame_idx: int) -> bool:
        table_pose = table_ref["pose"]
        if table_pose is None or origin_ref["tvec"] is not None:
            return False

        origin_tvec = _board_tvec_from_result(result, table_pose.T_B_C)
        if origin_tvec is None:
            print("[debug_tracker_translation] origin not set: no valid current tracker pose")
            return True

        origin_ref["tvec"] = origin_tvec.copy()
        origin_frame_ref["frame"] = int(frame_idx)
        tracker_log.set_debug_board_origin_tvec(origin_tvec)
        print(
            "[debug_tracker_translation] table origin set at frame "
            f"{frame_idx}: x={origin_tvec[0]:.3f} y={origin_tvec[1]:.3f} "
            f"z={origin_tvec[2]:.3f} mm"
        )
        return True

    def on_log_open(_field_path, _marker_json_path, _tracker) -> None:
        table_pose = table_ref["pose"]
        if table_pose is None:
            raise RuntimeError("Translation debug recording started without table calibration.")
        tracker_log.set_debug_board_transform(table_pose.T_B_C)
        if origin_ref["tvec"] is not None:
            tracker_log.set_debug_board_origin_tvec(origin_ref["tvec"])
        run_id = tracker_log.current_run_id()
        tracker_log.write_record(table_calibration_record(table_pose, run_id))
        for record in table_calibration_observation_records(table_pose, run_id):
            tracker_log.write_record(record)
        if origin_ref["tvec"] is not None:
            tracker_log.write_record(
                table_origin_record(
                    origin_ref["tvec"],
                    run_id,
                    frame_idx=origin_frame_ref["frame"],
                )
            )

    def on_log_close(path: Path) -> None:
        path = Path(path)
        if path not in recorded_paths:
            recorded_paths.append(path)
        print(
            "[debug_tracker_translation] recording saved; "
            "SPACE starts the next run, ESC finishes and analyzes all runs"
        )

    def draw_extra_overlay(vis: np.ndarray, log_active: bool, result=None) -> None:
        import cv2

        table_pose = table_ref["pose"]
        if table_pose is not None and camera_ref["K"] is not None and camera_ref["dist"] is not None:
            _draw_table_axes_near_result(
                vis,
                result,
                K=camera_ref["K"],
                dist=camera_ref["dist"],
                T_C_B=table_pose.T_C_B,
                T_B_C=table_pose.T_B_C,
            )

        if origin_ref["tvec"] is None:
            action = "SPACE=set ORIGIN here"
            color = (0, 255, 255)
        else:
            action = "SPACE=STOP recording" if log_active else "SPACE=START new recording"
            color = (0, 255, 0)

        cv2.rectangle(vis, (18, 168), (1500, 260), (0, 0, 0), thickness=-1)
        current_board = None if table_pose is None else _board_tvec_from_result(result, table_pose.T_B_C)
        if current_board is not None:
            live.put_text(
                vis,
                (
                    f"current T_B_T: x={current_board[0]:.1f} "
                    f"y={current_board[1]:.1f} z={current_board[2]:.1f} mm"
                ),
                (25, 185),
                color=(255, 255, 255),
                scale=0.48,
            )

        state_line = (
            "ChArUco table frame | orange=+X, green=+Y | "
            "move along +Y for FB/RL | "
            f"{action} | s=start/stop tracking | ESC=finish/analyze"
        )
        live.put_text(vis, state_line, (25, 215), color=color, scale=0.48)

    def final_cleanup() -> None:
        tracker_log.set_debug_board_transform(None)

    last_recorded_path = live.run_live_tracker(
        window_name="HydraTracker Translation Debug",
        console_prefix="[debug_tracker_translation]",
        after_camera_ready=after_camera_ready,
        after_tracker_created=after_tracker_created,
        on_space_key=on_space_key,
        on_log_open=on_log_open,
        on_log_close=on_log_close,
        draw_extra_overlay=draw_extra_overlay,
        stop_after_log_close=False,
        quit_on_q=False,
        final_cleanup=final_cleanup,
    )
    if last_recorded_path is not None and last_recorded_path not in recorded_paths:
        recorded_paths.append(last_recorded_path)
    return recorded_paths


def load_tracker_run(path: Path) -> dict:
    frames: list[int] = []
    success: list[int] = []
    pose_is_fresh: list[int] = []
    pose_source: list[str] = []
    tvec_camera: list[list[float]] = []
    tvec_board: list[list[float]] = []
    board_available: list[int] = []
    logged_delta: list[float] = []
    num_points: list[float] = []
    global_row_min: list[float] = []
    global_row_max: list[float] = []
    global_col_min: list[float] = []
    global_col_max: list[float] = []
    reproj_mean_px: list[float] = []
    roll_deg: list[float] = []
    pitch_deg: list[float] = []
    yaw_deg: list[float] = []
    best_method: list[str] = []
    timing_rows: list[dict[str, float]] = []

    run_id = path.stem
    run_timestamp = ""
    columns: list[str] = []
    summary: dict = {}
    table_origin_tvec: np.ndarray | None = None
    table_origin_frame: int | None = None

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

            if record_type == "run_summary":
                summary = dict(record.get("summary") or {})
                continue

            if record_type == "table_origin":
                value = record.get("origin_tvec_board_mm")
                if isinstance(value, (list, tuple)) and len(value) >= 3:
                    table_origin_tvec = np.asarray(
                        [_to_float(v) for v in value[:3]],
                        dtype=np.float64,
                    ).reshape(3)
                    if not np.all(np.isfinite(table_origin_tvec)):
                        table_origin_tvec = None
                table_origin_frame = _to_int(record.get("frame"), default=-1)
                continue

            if record_type != "frame":
                continue

            data = record.get("data") or {}
            timing_rows.append(_timing_profile_from_frame_data(data))
            frames.append(_to_int(data.get("frame"), default=len(frames)))
            success.append(_to_int(data.get("success"), default=0))
            pose_is_fresh.append(_to_int(data.get("pose_is_fresh"), default=0))
            pose_source.append(str(data.get("pose_source") or ""))
            logged_delta.append(_to_float(data.get("pose_translation_delta_mm")))
            board_available.append(_to_int(data.get("board_pose_available"), default=0))
            num_points.append(_to_float(data.get("num_points")))
            global_row_min.append(_to_float(data.get("pose_global_row_min")))
            global_row_max.append(_to_float(data.get("pose_global_row_max")))
            global_col_min.append(_to_float(data.get("pose_global_col_min")))
            global_col_max.append(_to_float(data.get("pose_global_col_max")))
            reproj_mean_px.append(_to_float(data.get("pose_reproj_mean_px")))
            roll_deg.append(_to_float(data.get("camera_roll_deg")))
            pitch_deg.append(_to_float(data.get("camera_pitch_deg")))
            yaw_deg.append(_to_float(data.get("camera_yaw_deg")))
            best_method.append(str(data.get("pose_candidate_best_method") or ""))
            tvec_camera.append(
                [
                    _to_float(data.get("tvec_x_mm")),
                    _to_float(data.get("tvec_y_mm")),
                    _to_float(data.get("tvec_z_mm")),
                ]
            )
            tvec_board.append(
                [
                    _to_float(data.get("board_tvec_x_mm")),
                    _to_float(data.get("board_tvec_y_mm")),
                    _to_float(data.get("board_tvec_z_mm")),
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
    tvec_board_arr = np.asarray(tvec_board, dtype=np.float64).reshape(-1, 3)
    board_available_arr = np.asarray(board_available, dtype=np.int64)
    has_board_tvec = (board_available_arr == 1) & np.all(np.isfinite(tvec_board_arr), axis=1)
    if np.any(has_board_tvec):
        tvec_abs = tvec_board_arr.copy()
        coordinate_frame = "charuco_table"
        absolute_label = "T_B_T"

    has_tvec = np.all(np.isfinite(tvec_abs), axis=1)
    origin_idx = int(np.where(has_tvec)[0][0])
    origin_frame = int(frame_arr[origin_idx])
    origin_tvec = tvec_abs[origin_idx].copy()
    relative_tvec = tvec_abs - origin_tvec

    computed_delta = np.full(len(frame_arr), np.nan, dtype=np.float64)
    for idx in range(1, len(frame_arr)):
        if has_tvec[idx] and has_tvec[idx - 1]:
            computed_delta[idx] = float(np.linalg.norm(relative_tvec[idx] - relative_tvec[idx - 1]))

    logged_delta_arr = np.asarray(logged_delta, dtype=np.float64)
    plot_delta = np.where(np.isfinite(computed_delta), computed_delta, logged_delta_arr)
    timing_keys = sorted({key for row in timing_rows for key in row})
    timing_series = {
        key: np.asarray([row.get(key, np.nan) for row in timing_rows], dtype=np.float64)
        for key in timing_keys
    }

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
        "coordinate_frame": coordinate_frame,
        "absolute_label": absolute_label,
        "num_points": np.asarray(num_points, dtype=np.float64),
        "global_row_min": np.asarray(global_row_min, dtype=np.float64),
        "global_row_max": np.asarray(global_row_max, dtype=np.float64),
        "global_col_min": np.asarray(global_col_min, dtype=np.float64),
        "global_col_max": np.asarray(global_col_max, dtype=np.float64),
        "reproj_mean_px": np.asarray(reproj_mean_px, dtype=np.float64),
        "roll_deg": np.asarray(roll_deg, dtype=np.float64),
        "pitch_deg": np.asarray(pitch_deg, dtype=np.float64),
        "yaw_deg": np.asarray(yaw_deg, dtype=np.float64),
        "best_method": np.asarray(best_method, dtype=object),
        "timing_series": timing_series,
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
    num_points: np.ndarray = run["num_points"]
    global_row_min: np.ndarray = run["global_row_min"]
    global_row_max: np.ndarray = run["global_row_max"]
    roll_deg: np.ndarray = run["roll_deg"]
    pitch_deg: np.ndarray = run["pitch_deg"]
    yaw_deg: np.ndarray = run["yaw_deg"]

    missing_mask = (success == 0) | ~has_tvec
    held_mask = (success == 1) & has_tvec & (fresh == 0)
    missing_ranges = contiguous_frame_ranges(frames, missing_mask)
    held_ranges = contiguous_frame_ranges(frames, held_mask)
    peak_frames, peak_threshold = robust_peak_frames(frames, delta_mm)

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15.5, 12.0),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(top=0.86, hspace=0.34)

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
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.985)

    def mark_ranges(ax) -> None:
        for start, end in missing_ranges:
            ax.axvspan(start - 0.5, end + 0.5, color="#e45756", alpha=0.14, lw=0)

        for start, end in held_ranges:
            ax.axvspan(start - 0.5, end + 0.5, color="#f2b701", alpha=0.14, lw=0)

        for peak_frame in peak_frames:
            ax.axvline(int(peak_frame), color="#7f3c8d", alpha=0.38, linewidth=1.1)

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
            stats = _component_spread_stats(values)
            if np.isfinite(stats["p95_low"]) and np.isfinite(stats["p95_high"]):
                ax.axhspan(
                    stats["p95_low"],
                    stats["p95_high"],
                    color=color,
                    alpha=0.08,
                    lw=0,
                    label="95% band",
                )
            ax.axhline(0.0, color="#555555", alpha=0.32, linewidth=1.1, linestyle="--")
            ax.set_title(
                (
                    f"{label.upper()} relative component   "
                    f"p95|0|={stats['p95_abs']:.2f} mm   "
                    f"95%=[{stats['p95_low']:+.2f},{stats['p95_high']:+.2f}] mm   "
                    f"std={stats['std']:.2f} mm   var={stats['var']:.3f} mm^2"
                ),
                loc="left",
            )
        else:
            ax.set_title(f"{label.upper()} relative component", loc="left")

        mark_ranges(ax)
        ax.set_ylabel(f"delta {absolute_label} {label} [mm]")
        ax.grid(True, axis="both")
        ax.legend(loc="upper right")

    point_ax = axes[3]
    points = num_points.copy()
    points[~np.isfinite(points)] = np.nan
    point_ax.plot(
        frames,
        points,
        color="#4c78a8",
        linewidth=1.8,
        marker="o",
        markersize=2.4,
        markerfacecolor="white",
        markeredgewidth=0.7,
        label="pose points",
    )
    point_ax.set_ylabel("points")
    point_ax.set_title("Pose diagnostics   point count and global row range", loc="left")
    point_ax.grid(True, axis="both")
    mark_ranges(point_ax)

    row_ax = point_ax.twinx()
    row_ax.plot(frames, global_row_min, color="#f58518", linewidth=1.2, label="row min")
    row_ax.plot(frames, global_row_max, color="#b279a2", linewidth=1.2, label="row max")
    row_ax.set_ylabel("global row")

    point_lines, point_labels = point_ax.get_legend_handles_labels()
    row_lines, row_labels = row_ax.get_legend_handles_labels()
    point_ax.legend(point_lines + row_lines, point_labels + row_labels, loc="upper right")

    orient_ax = axes[4]
    orientation = (
        (roll_deg, "roll", "#4c78a8"),
        (pitch_deg, "pitch", "#54a24b"),
        (yaw_deg, "yaw", "#e45756"),
    )
    for values, label, color in orientation:
        rel_values = values.astype(np.float64).copy()
        finite = np.isfinite(rel_values)
        if np.any(finite):
            rel_values = rel_values - rel_values[np.where(finite)[0][0]]
            rel_values[~finite] = np.nan
        orient_ax.plot(frames, rel_values, color=color, linewidth=1.6, label=f"{label} delta")

    orient_ax.axhline(0.0, color="#888888", alpha=0.3, linewidth=1.0, linestyle="--")
    orient_ax.set_ylabel("rotation delta [deg]")
    orient_ax.set_title("Orientation diagnostics   camera-frame Euler deltas", loc="left")
    orient_ax.grid(True, axis="both")
    orient_ax.legend(loc="upper right")
    mark_ranges(orient_ax)

    axes[-1].set_xlabel("frame")

    info = (
        f"relative to frame {origin_frame} "
        f"({origin_tvec[0]:.2f}, {origin_tvec[1]:.2f}, {origin_tvec[2]:.2f}) mm   "
        f"frames={len(frames)}   pose={pose_pct:.1f}%   "
        f"missing={int(np.count_nonzero(missing_mask))}   held={int(np.count_nonzero(held_mask))}"
    )
    if np.isfinite(peak_threshold) and len(peak_frames) > 0:
        info += f"   peak threshold={peak_threshold:.2f} mm   peaks={', '.join(str(int(f)) for f in peak_frames)}"

    fig.text(
        0.01,
        0.925,
        info,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#333333",
    )

    status_text = ""
    if missing_ranges:
        status_text = "red = missing pose   yellow = held/stale pose   purple = large translation jump"
    elif held_ranges or len(peak_frames) > 0:
        status_text = "yellow = held/stale pose   purple = large translation jump"

    if status_text:
        fig.text(
            0.99,
            0.925,
            status_text,
            ha="right",
            va="top",
            fontsize=9.5,
            color="#555555",
        )

    out_path = path.with_name(f"{path.stem}_translation_relative_plot.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[translation_plot] saved -> {out_path.resolve()}")
    plt.show()
    return out_path


def plot_existing_run(path: Path | None = None) -> Path | None:
    if path is None:
        path = select_jsonl_with_qt()

    if path is None:
        print("No file selected.")
        return None

    run = load_tracker_run(path)
    print_timing_profile_summary(run)
    write_timing_profile_summary(run)
    return plot_translation(run)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] in (
        "--offline-depth-filter",
        "--depth-filter-replay",
        "depth-filter-replay",
    ):
        if len(args) < 2:
            raise RuntimeError(
                "Missing JSONL path.\n"
                "Usage: debug_tracker_translation.py --offline-depth-filter <run.jsonl> [output.jsonl]"
            )
        input_path = Path(args[1])
        output_path = Path(args[2]) if len(args) > 2 else None
        replay_path = replay_depth_filter_on_run(input_path, output_path)
        plot_existing_run(replay_path)
        return

    if args and args[0] in ("--plot", "plot"):
        path = Path(args[1]) if len(args) > 1 else None
        plot_existing_run(path)
        return

    if args and args[0] in ("--select", "select"):
        plot_existing_run(None)
        return

    if args and Path(args[0]).suffix.lower() == ".jsonl":
        plot_existing_run(Path(args[0]))
        return

    allowed_live_flags = {
        "--no-fast-persistent",
        "--no-temporal-persistence",
        "--decode-only",
    }
    unknown = [arg for arg in args if arg not in allowed_live_flags]
    if unknown:
        raise RuntimeError(
            "Unknown live option(s): "
            + ", ".join(unknown)
            + "\nKnown live options: "
            + ", ".join(sorted(allowed_live_flags))
        )
    recorded_paths = run_live_tracker_translation(
        no_fast_persistent="--no-fast-persistent" in args,
        no_temporal_persistence="--no-temporal-persistence" in args,
        decode_only="--decode-only" in args,
    )
    if not recorded_paths:
        print("No recording to analyze.")
        return

    for recorded_path in recorded_paths:
        plot_existing_run(recorded_path)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[debug_tracker_translation] ERROR: {exc}")
        sys.exit(1)
