"""Offline z-bias map and feature-family diagnostic for HydraTracker JSONL runs.

Typical use:
    python hydramarker/debug/res_z_bias_map.py --latest
    python hydramarker/debug/res_z_bias_map.py path/to/run.jsonl --target xy-residual

The script writes frame-level features, empirical image bins, ridge-model summaries,
coefficients, and an optional plot next to the selected run.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


COMPONENTS = ("x", "y", "z")


def _suppress_windows_error_dialogs() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        sem_failcriticalerrors = 0x0001
        sem_nogpfaultbox = 0x0002
        sem_noopenfileerrorbox = 0x8000
        ctypes.windll.kernel32.SetErrorMode(
            sem_failcriticalerrors | sem_nogpfaultbox | sem_noopenfileerrorbox
        )
    except Exception:
        pass


_suppress_windows_error_dialogs()


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _finite(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _mean(values: list[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if len(arr) else math.nan


def _median(values: list[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(np.median(arr)) if len(arr) else math.nan


def _percentile(values: list[float] | np.ndarray, q: float) -> float:
    arr = _finite(values)
    return float(np.percentile(arr, q)) if len(arr) else math.nan


def _rms(values: list[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(math.sqrt(float(np.mean(arr * arr)))) if len(arr) else math.nan


def _corr(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(mask)) < 3:
        return math.nan
    aa = aa[mask]
    bb = bb[mask]
    if float(np.std(aa)) <= 0.0 or float(np.std(bb)) <= 0.0:
        return math.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def _safe_log10(value: Any) -> float:
    x = _to_float(value)
    return float(math.log10(x)) if np.isfinite(x) and x > 0.0 else math.nan


def _camera_info_from_run_start(record: dict[str, Any]) -> dict[str, float]:
    info = dict(record.get("camera_intrinsics") or {})
    width = _to_float(info.get("width"))
    height = _to_float(info.get("height"))
    K = (
        info.get("tracker_K")
        or info.get("K")
        or info.get("camera_matrix")
        or info.get("raw_realsense_K")
    )
    if K is not None:
        try:
            arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
            return {
                "width": width,
                "height": height,
                "cx": float(arr[0, 2]),
                "cy": float(arr[1, 2]),
                "fx": float(arr[0, 0]),
                "fy": float(arr[1, 1]),
            }
        except Exception:
            pass
    return {
        "width": width,
        "height": height,
        "cx": _to_float(info.get("ppx", info.get("cx"))),
        "cy": _to_float(info.get("ppy", info.get("cy"))),
        "fx": _to_float(info.get("fx")),
        "fy": _to_float(info.get("fy")),
    }


def _read_csv_by_frame(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        frame = _to_int(row.get("frame"), default=-1)
        if frame >= 0:
            out[frame] = row
    return out


def load_run(path: Path) -> dict[str, Any]:
    run_id = path.stem
    timestamp = ""
    camera_info = {
        "width": math.nan,
        "height": math.nan,
        "cx": math.nan,
        "cy": math.nan,
        "fx": math.nan,
        "fy": math.nan,
    }
    frames: dict[int, dict[str, Any]] = {}
    details: dict[int, dict[str, Any]] = {}
    summary: dict[str, Any] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
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
                timestamp = str(record.get("timestamp") or "")
                camera_info = _camera_info_from_run_start(record)
            elif record_type == "frame":
                data = dict(record.get("data") or {})
                frame = _to_int(data.get("frame"), default=len(frames))
                frames[frame] = data
            elif record_type == "frame_detail":
                frame = _to_int(record.get("frame"), default=-1)
                if frame >= 0:
                    details[frame] = dict(record)
            elif record_type == "run_summary":
                summary = dict(record.get("summary") or {})

    if not frames:
        raise RuntimeError(f"No frame records found in:\n{path}")

    return {
        "path": path,
        "run_id": run_id,
        "timestamp": timestamp,
        "camera": camera_info,
        "frames": frames,
        "details": details,
        "summary": summary,
    }


def _corner_id(corner: dict[str, Any], index: int) -> str:
    row = _to_int(corner.get("global_row"), default=-999)
    col = _to_int(corner.get("global_col"), default=-999)
    if row != -999 and col != -999:
        return f"r{row}_c{col}"
    local_row = _to_int(corner.get("local_row"), default=-999)
    local_col = _to_int(corner.get("local_col"), default=-999)
    if local_row != -999 and local_col != -999:
        return f"lr{local_row}_lc{local_col}"
    return f"corner_{index}"


def _corner_uv(corner: dict[str, Any]) -> tuple[float, float]:
    uv = corner.get("uv_px")
    if isinstance(uv, (list, tuple)) and len(uv) >= 2:
        return _to_float(uv[0]), _to_float(uv[1])
    return math.nan, math.nan


def _corner_xyz(corner: dict[str, Any]) -> tuple[float, float, float]:
    xyz = corner.get("xyz_mm")
    if isinstance(xyz, (list, tuple)) and len(xyz) >= 3:
        return _to_float(xyz[0]), _to_float(xyz[1]), _to_float(xyz[2])
    return math.nan, math.nan, math.nan


def _corner_residual(corner: dict[str, Any]) -> tuple[float, float, float]:
    residual = corner.get("residual_px")
    if isinstance(residual, (list, tuple)) and len(residual) >= 2:
        du = _to_float(residual[0])
        dv = _to_float(residual[1])
    else:
        uv = corner.get("uv_px")
        projected = corner.get("projected_uv_px")
        if (
            isinstance(uv, (list, tuple))
            and isinstance(projected, (list, tuple))
            and len(uv) >= 2
            and len(projected) >= 2
        ):
            du = _to_float(projected[0]) - _to_float(uv[0])
            dv = _to_float(projected[1]) - _to_float(uv[1])
        else:
            du = math.nan
            dv = math.nan
    err = _to_float(corner.get("error_px"))
    if not np.isfinite(err) and np.isfinite(du) and np.isfinite(dv):
        err = float(math.hypot(du, dv))
    return du, dv, err


def _radial_tangential(
    u: float,
    v: float,
    du: float,
    dv: float,
    cx: float,
    cy: float,
) -> tuple[float, float]:
    if not all(np.isfinite([u, v, du, dv, cx, cy])):
        return math.nan, math.nan
    rx = u - cx
    ry = v - cy
    norm = float(math.hypot(rx, ry))
    if norm <= 1e-12:
        return math.nan, math.nan
    radial = (du * rx + dv * ry) / norm
    tangential = (-du * ry + dv * rx) / norm
    return float(radial), float(tangential)


def _derive_corner_stats(
    corners: list[dict[str, Any]],
    *,
    frame: int,
    camera: dict[str, float],
    previous_uv_by_id: dict[str, tuple[int, np.ndarray]],
    max_gap_frames: int,
) -> dict[str, float]:
    cx = _to_float(camera.get("cx"))
    cy = _to_float(camera.get("cy"))
    fx = _to_float(camera.get("fx"))
    fy = _to_float(camera.get("fy"))

    rows: list[int] = []
    cols: list[int] = []
    us: list[float] = []
    vs: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    dus: list[float] = []
    dvs: list[float] = []
    errors: list[float] = []
    radial_residuals: list[float] = []
    tangential_residuals: list[float] = []
    votes: list[float] = []
    speeds: list[float] = []
    vu_values: list[float] = []
    vv_values: list[float] = []
    residual_along_motion: list[float] = []

    for idx, corner in enumerate(corners):
        u, v = _corner_uv(corner)
        x, y, z = _corner_xyz(corner)
        du, dv, error = _corner_residual(corner)
        row = _to_int(corner.get("global_row"), default=_to_int(corner.get("local_row"), -999))
        col = _to_int(corner.get("global_col"), default=_to_int(corner.get("local_col"), -999))
        radial, tangential = _radial_tangential(u, v, du, dv, cx, cy)
        vote = _to_float(corner.get("votes"))

        if row != -999:
            rows.append(row)
        if col != -999:
            cols.append(col)
        us.append(u)
        vs.append(v)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        dus.append(du)
        dvs.append(dv)
        errors.append(error)
        radial_residuals.append(radial)
        tangential_residuals.append(tangential)
        votes.append(vote)

        if np.isfinite(u) and np.isfinite(v):
            cid = _corner_id(corner, idx)
            uv = np.asarray([u, v], dtype=np.float64)
            prev = previous_uv_by_id.get(cid)
            if prev is not None:
                prev_frame, prev_uv = prev
                gap = frame - prev_frame
                if 0 < gap <= max_gap_frames and np.all(np.isfinite(prev_uv)):
                    velocity = (uv - prev_uv) / float(gap)
                    speed = float(math.hypot(float(velocity[0]), float(velocity[1])))
                    speeds.append(speed)
                    vu_values.append(float(velocity[0]))
                    vv_values.append(float(velocity[1]))
                    if speed > 1e-12 and np.isfinite(du) and np.isfinite(dv):
                        residual_along_motion.append(float((du * velocity[0] + dv * velocity[1]) / speed))
            previous_uv_by_id[cid] = (frame, uv)

    row_values = _finite(np.asarray(rows, dtype=np.float64))
    col_values = _finite(np.asarray(cols, dtype=np.float64))
    u_values = _finite(us)
    v_values = _finite(vs)
    x_values = _finite(xs)
    y_values = _finite(ys)
    z_values = _finite(zs)
    err_values = _finite(errors)
    image_u_span = float(np.ptp(u_values)) if len(u_values) else math.nan
    image_v_span = float(np.ptp(v_values)) if len(v_values) else math.nan
    image_area = image_u_span * image_v_span if np.isfinite(image_u_span) and np.isfinite(image_v_span) else math.nan
    image_aspect = (
        image_u_span / image_v_span
        if np.isfinite(image_u_span) and np.isfinite(image_v_span) and abs(image_v_span) > 1e-12
        else math.nan
    )
    image_u_centroid = _mean(us)
    image_v_centroid = _mean(vs)
    image_radius_px = (
        math.hypot(image_u_centroid - cx, image_v_centroid - cy)
        if all(np.isfinite([image_u_centroid, image_v_centroid, cx, cy]))
        else math.nan
    )
    image_radius_norm = (
        math.hypot((image_u_centroid - cx) / fx, (image_v_centroid - cy) / fy)
        if all(np.isfinite([image_u_centroid, image_v_centroid, cx, cy, fx, fy]))
        and abs(fx) > 1e-12
        and abs(fy) > 1e-12
        else math.nan
    )

    return {
        "point_count_used": float(len(corners)),
        "row_min": float(np.min(row_values)) if len(row_values) else math.nan,
        "row_max": float(np.max(row_values)) if len(row_values) else math.nan,
        "row_span": float(np.ptp(row_values)) if len(row_values) else math.nan,
        "col_min": float(np.min(col_values)) if len(col_values) else math.nan,
        "col_max": float(np.max(col_values)) if len(col_values) else math.nan,
        "col_span": float(np.ptp(col_values)) if len(col_values) else math.nan,
        "distinct_rows": float(len(set(rows))) if rows else math.nan,
        "distinct_cols": float(len(set(cols))) if cols else math.nan,
        "image_u_centroid_px": image_u_centroid,
        "image_v_centroid_px": image_v_centroid,
        "image_radius_px": image_radius_px,
        "image_radius_norm": image_radius_norm,
        "image_u_span_px": image_u_span,
        "image_v_span_px": image_v_span,
        "image_area_kpx2": image_area / 1000.0 if np.isfinite(image_area) else math.nan,
        "image_aspect": image_aspect,
        "object_x_centroid_mm": _mean(xs),
        "object_y_centroid_mm": _mean(ys),
        "object_z_centroid_mm": _mean(zs),
        "object_x_span_mm": float(np.ptp(x_values)) if len(x_values) else math.nan,
        "object_y_span_mm": float(np.ptp(y_values)) if len(y_values) else math.nan,
        "object_z_span_mm": float(np.ptp(z_values)) if len(z_values) else math.nan,
        "residual_mean_du_px": _mean(dus),
        "residual_mean_dv_px": _mean(dvs),
        "residual_abs_mean_du_px": _mean(np.abs(np.asarray(dus, dtype=np.float64))),
        "residual_abs_mean_dv_px": _mean(np.abs(np.asarray(dvs, dtype=np.float64))),
        "residual_rms_px": _rms(errors),
        "residual_median_px": _median(errors),
        "residual_p95_px": _percentile(errors, 95),
        "residual_max_px": float(np.max(err_values)) if len(err_values) else math.nan,
        "residual_radial_mean_px": _mean(radial_residuals),
        "residual_tangential_mean_px": _mean(tangential_residuals),
        "votes_mean": _mean(votes),
        "votes_min": float(np.min(_finite(votes))) if len(_finite(votes)) else math.nan,
        "corner_speed_median_px_per_frame": _median(speeds),
        "corner_speed_p95_px_per_frame": _percentile(speeds, 95),
        "corner_vu_median_px_per_frame": _median(vu_values),
        "corner_vv_median_px_per_frame": _median(vv_values),
        "residual_along_motion_median_px": _median(residual_along_motion),
        "motion_corner_count": float(len(speeds)),
    }


def build_frame_rows(
    run: dict[str, Any],
    *,
    point_set: str,
    uncertainty_by_frame: dict[int, dict[str, str]],
    max_gap_frames: int,
    max_frames: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = run["frames"]
    details: dict[int, dict[str, Any]] = run["details"]
    sorted_frames = sorted(frames)
    if max_frames is not None and max_frames > 0:
        sorted_frames = sorted_frames[: int(max_frames)]

    tvecs: list[np.ndarray] = []
    for frame in sorted_frames:
        data = frames[frame]
        tvecs.append(
            np.asarray(
                [
                    _to_float(data.get("tvec_x_mm")),
                    _to_float(data.get("tvec_y_mm")),
                    _to_float(data.get("tvec_z_mm")),
                ],
                dtype=np.float64,
            )
        )
    tvec_arr = np.asarray(tvecs, dtype=np.float64).reshape(-1, 3)
    valid_tvec = np.all(np.isfinite(tvec_arr), axis=1)
    if not np.any(valid_tvec):
        raise RuntimeError("No finite tvec_x_mm/tvec_y_mm/tvec_z_mm values found.")

    origin_index = int(np.where(valid_tvec)[0][0])
    origin_frame = sorted_frames[origin_index]
    origin_tvec = tvec_arr[origin_index].copy()
    rel_tvec = tvec_arr - origin_tvec
    ranges = np.nanmax(rel_tvec[valid_tvec], axis=0) - np.nanmin(rel_tvec[valid_tvec], axis=0)
    movement_axis_idx = int(np.nanargmax(ranges))
    movement_axis = COMPONENTS[movement_axis_idx]
    movement_values = rel_tvec[:, movement_axis_idx]
    turn_idx = int(np.nanargmax(np.abs(movement_values)))
    turn_frame = sorted_frames[turn_idx]

    previous_uv_by_id: dict[str, tuple[int, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for idx, frame in enumerate(sorted_frames):
        data = frames[frame]
        detail = details.get(frame, {})
        tvec = tvec_arr[idx]
        rel = rel_tvec[idx]
        success = _to_int(data.get("success"), default=0)
        if success == 0 or not np.all(np.isfinite(rel)):
            continue

        corners_key = "pose_corners" if point_set == "pose" else "correspondence_corners"
        corners = list(detail.get(corners_key) or [])
        if not corners and point_set == "pose":
            corners = list(detail.get("correspondence_corners") or [])

        branch = "out" if frame <= turn_frame else "return"
        detail_timings = dict(data.get("checkerboard_detail_timings") or {})
        corner_stats = _derive_corner_stats(
            corners,
            frame=frame,
            camera=run["camera"],
            previous_uv_by_id=previous_uv_by_id,
            max_gap_frames=max_gap_frames,
        )
        uncertainty = uncertainty_by_frame.get(frame, {})
        row: dict[str, Any] = {
            "run_id": run["run_id"],
            "frame": int(frame),
            "branch": branch,
            "branch_return": 1.0 if branch == "return" else 0.0,
            "movement_axis": movement_axis,
            "movement_axis_value_mm": float(rel[movement_axis_idx]),
            "turn_frame": int(turn_frame),
            "success": success,
            "pose_source": str(data.get("pose_source") or ""),
            "pnp_method": str(data.get("pnp_method") or ""),
            "tvec_x_mm": float(tvec[0]),
            "tvec_y_mm": float(tvec[1]),
            "tvec_z_mm": float(tvec[2]),
            "rel_x_mm": float(rel[0]),
            "rel_y_mm": float(rel[1]),
            "rel_z_mm": float(rel[2]),
            "rvec_x_rad": _to_float(data.get("rvec_x_rad")),
            "rvec_y_rad": _to_float(data.get("rvec_y_rad")),
            "rvec_z_rad": _to_float(data.get("rvec_z_rad")),
            "camera_roll_deg": _to_float(data.get("camera_roll_deg")),
            "camera_pitch_deg": _to_float(data.get("camera_pitch_deg")),
            "camera_yaw_deg": _to_float(data.get("camera_yaw_deg")),
            "num_points_logged": _to_float(data.get("num_points")),
            "det_corners": _to_float(data.get("det_corners")),
            "persistent_count": _to_float(data.get("persistent_count")),
            "fast_dense_image_coverage": _to_float(data.get("fast_dense_image_coverage")),
            "fast_dense_median_px": _to_float(data.get("fast_dense_median_px")),
            "fast_dense_p90_px": _to_float(data.get("fast_dense_p90_px")),
            "mean_corner_motion_px_logged": _to_float(data.get("mean_corner_motion_px")),
            "median_corner_motion_px_logged": _to_float(data.get("median_corner_motion_px")),
            "p95_corner_motion_px_logged": _to_float(data.get("p95_corner_motion_px")),
            "pose_reproj_mean_px_logged": _to_float(data.get("pose_reproj_mean_px")),
            "pose_reproj_p95_px_logged": _to_float(data.get("pose_reproj_p95_px")),
            "pose_reproj_max_px_logged": _to_float(data.get("pose_reproj_max_px")),
            "photometric_shadow_usable_signal": _to_float(detail_timings.get("photometric_shadow_usable_signal")),
            "photometric_shadow_consensus_fraction": _to_float(detail_timings.get("photometric_shadow_consensus_fraction")),
            "photometric_shadow_shift_median_px": _to_float(detail_timings.get("photometric_shadow_shift_median_px")),
            "photometric_shadow_residual_p95_px": _to_float(detail_timings.get("photometric_shadow_residual_p95_px")),
        }
        row.update(corner_stats)
        for key in (
            "sigma_z_mm",
            "sigma_z_fit_mm",
            "jacobian_condition",
            "normal_condition",
            "sigma_z_over_xy",
            "max_abs_z_rotation_corr",
            "cv_innovation_z_mm",
            "cv_innovation_z_over_sigma",
        ):
            if key in uncertainty:
                row[key] = _to_float(uncertainty.get(key))
        rows.append(row)

    if not rows:
        raise RuntimeError("No successful frames with finite pose data found.")

    meta = {
        "origin_frame": int(origin_frame),
        "origin_tvec_x_mm": float(origin_tvec[0]),
        "origin_tvec_y_mm": float(origin_tvec[1]),
        "origin_tvec_z_mm": float(origin_tvec[2]),
        "movement_axis": movement_axis,
        "movement_axis_idx": movement_axis_idx,
        "turn_frame": int(turn_frame),
        "x_range_mm": float(ranges[0]),
        "y_range_mm": float(ranges[1]),
        "z_range_mm": float(ranges[2]),
    }
    return rows, meta


def _fit_unregularized(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, X @ beta


def add_derived_features(rows: list[dict[str, Any]], *, target: str) -> None:
    rel_x = np.asarray([_to_float(row.get("rel_x_mm")) for row in rows], dtype=np.float64)
    rel_y = np.asarray([_to_float(row.get("rel_y_mm")) for row in rows], dtype=np.float64)
    rel_z = np.asarray([_to_float(row.get("rel_z_mm")) for row in rows], dtype=np.float64)
    movement = np.asarray([_to_float(row.get("movement_axis_value_mm")) for row in rows], dtype=np.float64)
    u = np.asarray([_to_float(row.get("image_u_centroid_px")) for row in rows], dtype=np.float64)
    v = np.asarray([_to_float(row.get("image_v_centroid_px")) for row in rows], dtype=np.float64)
    width = _median([_to_float(row.get("camera_width_px")) for row in rows])
    height = _median([_to_float(row.get("camera_height_px")) for row in rows])
    if not np.isfinite(width) or width <= 0.0:
        width = float(np.nanmax(u) - np.nanmin(u)) if np.any(np.isfinite(u)) else 1.0
    if not np.isfinite(height) or height <= 0.0:
        height = float(np.nanmax(v) - np.nanmin(v)) if np.any(np.isfinite(v)) else 1.0
    u_center = _median(u)
    v_center = _median(v)
    u_norm = (u - u_center) / max(width, 1.0)
    v_norm = (v - v_center) / max(height, 1.0)

    xy_design = np.column_stack(
        [
            np.ones_like(rel_z),
            rel_x,
            rel_y,
            rel_x * rel_x,
            rel_y * rel_y,
            rel_x * rel_y,
        ]
    )
    mask = np.isfinite(rel_z) & np.all(np.isfinite(xy_design), axis=1)
    xy_fit = np.full(len(rows), np.nan, dtype=np.float64)
    if int(np.sum(mask)) >= 8:
        _beta, pred = _fit_unregularized(xy_design[mask], rel_z[mask])
        xy_fit[mask] = pred

    target_values: np.ndarray
    if target == "rel-z":
        target_values = rel_z.copy()
    elif target == "demeaned-z":
        target_values = rel_z - _median(rel_z)
    elif target == "xy-residual":
        target_values = rel_z - xy_fit
    else:
        raise RuntimeError("--target must be one of: rel-z, demeaned-z, xy-residual")

    for idx, row in enumerate(rows):
        row["rel_x2_mm2"] = float(rel_x[idx] * rel_x[idx])
        row["rel_y2_mm2"] = float(rel_y[idx] * rel_y[idx])
        row["rel_xy_mm2"] = float(rel_x[idx] * rel_y[idx])
        row["movement_axis_value2_mm2"] = float(movement[idx] * movement[idx])
        row["image_u_norm"] = float(u_norm[idx])
        row["image_v_norm"] = float(v_norm[idx])
        row["image_u_norm2"] = float(u_norm[idx] * u_norm[idx])
        row["image_v_norm2"] = float(v_norm[idx] * v_norm[idx])
        row["image_uv_norm"] = float(u_norm[idx] * v_norm[idx])
        row["image_radius_norm2"] = _to_float(row.get("image_radius_norm")) ** 2
        row["residual_mean_vector_px"] = math.hypot(
            _to_float(row.get("residual_mean_du_px")),
            _to_float(row.get("residual_mean_dv_px")),
        )
        row["residual_abs_mean_vector_px"] = math.hypot(
            _to_float(row.get("residual_abs_mean_du_px")),
            _to_float(row.get("residual_abs_mean_dv_px")),
        )
        row["log_jacobian_condition"] = _safe_log10(row.get("jacobian_condition"))
        row["log_normal_condition"] = _safe_log10(row.get("normal_condition"))
        row["z_xy_quadratic_fit_mm"] = float(xy_fit[idx])
        row["z_xy_quadratic_resid_mm"] = float(rel_z[idx] - xy_fit[idx])
        row["target_z_bias_mm"] = float(target_values[idx])


MOVEMENT_KEYS = [
    "movement_axis_value_mm",
    "movement_axis_value2_mm2",
]
CAMERA_XY_KEYS = [
    "rel_x_mm",
    "rel_y_mm",
    "rel_x2_mm2",
    "rel_y2_mm2",
    "rel_xy_mm2",
]
IMAGE_CENTROID_KEYS = [
    "image_u_norm",
    "image_v_norm",
    "image_u_norm2",
    "image_v_norm2",
    "image_uv_norm",
    "image_radius_norm",
    "image_radius_norm2",
]
IMAGE_EXTENT_KEYS = [
    "image_u_span_px",
    "image_v_span_px",
    "image_area_kpx2",
    "image_aspect",
    "fast_dense_image_coverage",
]
COVERAGE_KEYS = [
    "point_count_used",
    "row_min",
    "row_max",
    "row_span",
    "col_min",
    "col_max",
    "col_span",
    "distinct_rows",
    "distinct_cols",
    "object_x_centroid_mm",
    "object_y_centroid_mm",
    "object_z_centroid_mm",
    "object_x_span_mm",
    "object_y_span_mm",
    "object_z_span_mm",
]
RESIDUAL_KEYS = [
    "residual_mean_du_px",
    "residual_mean_dv_px",
    "residual_mean_vector_px",
    "residual_abs_mean_du_px",
    "residual_abs_mean_dv_px",
    "residual_abs_mean_vector_px",
    "residual_rms_px",
    "residual_median_px",
    "residual_p95_px",
    "residual_max_px",
    "residual_radial_mean_px",
    "residual_tangential_mean_px",
    "fast_dense_median_px",
    "fast_dense_p90_px",
]
MOTION_KEYS = [
    "corner_speed_median_px_per_frame",
    "corner_speed_p95_px_per_frame",
    "corner_vu_median_px_per_frame",
    "corner_vv_median_px_per_frame",
    "residual_along_motion_median_px",
    "motion_corner_count",
    "mean_corner_motion_px_logged",
    "median_corner_motion_px_logged",
    "p95_corner_motion_px_logged",
]
UNCERTAINTY_KEYS = [
    "sigma_z_mm",
    "sigma_z_fit_mm",
    "sigma_z_over_xy",
    "log_jacobian_condition",
    "log_normal_condition",
    "max_abs_z_rotation_corr",
    "cv_innovation_z_mm",
    "cv_innovation_z_over_sigma",
]
SHADOW_KEYS = [
    "photometric_shadow_usable_signal",
    "photometric_shadow_consensus_fraction",
    "photometric_shadow_shift_median_px",
    "photometric_shadow_residual_p95_px",
]


def feature_sets() -> list[tuple[str, list[str]]]:
    image_quality = IMAGE_CENTROID_KEYS + IMAGE_EXTENT_KEYS + COVERAGE_KEYS + RESIDUAL_KEYS
    all_no_branch = (
        MOVEMENT_KEYS
        + CAMERA_XY_KEYS
        + IMAGE_CENTROID_KEYS
        + IMAGE_EXTENT_KEYS
        + COVERAGE_KEYS
        + RESIDUAL_KEYS
        + MOTION_KEYS
        + UNCERTAINTY_KEYS
        + SHADOW_KEYS
    )
    return [
        ("constant", []),
        ("movement_poly", MOVEMENT_KEYS),
        ("camera_xy_poly", CAMERA_XY_KEYS),
        ("image_centroid_poly", IMAGE_CENTROID_KEYS),
        ("image_extent", IMAGE_EXTENT_KEYS),
        ("coverage", COVERAGE_KEYS),
        ("residual_summary", RESIDUAL_KEYS),
        ("motion_summary", MOTION_KEYS),
        ("uncertainty", UNCERTAINTY_KEYS),
        ("image_plus_coverage", IMAGE_CENTROID_KEYS + IMAGE_EXTENT_KEYS + COVERAGE_KEYS),
        ("image_plus_residual", IMAGE_CENTROID_KEYS + RESIDUAL_KEYS),
        ("image_quality", image_quality),
        ("all_no_branch", all_no_branch),
        ("all_with_branch", all_no_branch + ["branch_return"]),
    ]


@dataclass
class FittedModel:
    name: str
    feature_keys: list[str]
    beta: np.ndarray
    fill_values: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    train_rows: int


def _feature_matrix(rows: list[dict[str, Any]], feature_keys: list[str]) -> np.ndarray:
    if not feature_keys:
        return np.empty((len(rows), 0), dtype=np.float64)
    return np.asarray(
        [[_to_float(row.get(key)) for key in feature_keys] for row in rows],
        dtype=np.float64,
    )


def _target_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([_to_float(row.get("target_z_bias_mm")) for row in rows], dtype=np.float64)


def _fit_ridge(
    rows: list[dict[str, Any]],
    feature_keys: list[str],
    train_idx: np.ndarray,
    *,
    alpha: float,
    model_name: str,
) -> FittedModel:
    X_all = _feature_matrix(rows, feature_keys)
    y_all = _target_array(rows)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if X_all.shape[1] == 0:
        X_train = np.empty((len(train_idx), 0), dtype=np.float64)
    else:
        X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    valid = np.isfinite(y_train)
    if X_train.shape[1] > 0:
        finite_feature_count = np.sum(np.isfinite(X_train), axis=1)
        valid &= finite_feature_count >= max(1, min(3, X_train.shape[1]))
    X_train = X_train[valid]
    y_train = y_train[valid]
    if len(y_train) == 0:
        raise RuntimeError(f"Model {model_name} has no finite training rows.")

    if X_train.shape[1] == 0:
        fill_values = np.empty(0, dtype=np.float64)
        means = np.empty(0, dtype=np.float64)
        scales = np.empty(0, dtype=np.float64)
        beta = np.asarray([float(np.mean(y_train))], dtype=np.float64)
        return FittedModel(model_name, list(feature_keys), beta, fill_values, means, scales, len(y_train))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fill_values = np.nanmedian(X_train, axis=0)
    fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0)
    X_filled = np.where(np.isfinite(X_train), X_train, fill_values.reshape(1, -1))
    means = np.mean(X_filled, axis=0)
    scales = np.std(X_filled, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
    X_scaled = (X_filled - means.reshape(1, -1)) / scales.reshape(1, -1)
    design = np.column_stack([np.ones(len(X_scaled), dtype=np.float64), X_scaled])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(design.T @ design + penalty, design.T @ y_train, rcond=None)
    return FittedModel(model_name, list(feature_keys), beta, fill_values, means, scales, len(y_train))


def _predict(model: FittedModel, rows: list[dict[str, Any]]) -> np.ndarray:
    if not model.feature_keys:
        return np.full(len(rows), float(model.beta[0]), dtype=np.float64)
    X = _feature_matrix(rows, model.feature_keys)
    X_filled = np.where(np.isfinite(X), X, model.fill_values.reshape(1, -1))
    X_scaled = (X_filled - model.means.reshape(1, -1)) / model.scales.reshape(1, -1)
    design = np.column_stack([np.ones(len(X_scaled), dtype=np.float64), X_scaled])
    return design @ model.beta


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(np.sum(mask)) == 0:
        return {"n": 0.0, "rmse_mm": math.nan, "mae_mm": math.nan, "r2": math.nan}
    yy = y_true[mask]
    pp = y_pred[mask]
    resid = yy - pp
    rmse = float(math.sqrt(float(np.mean(resid * resid))))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan
    return {"n": float(len(yy)), "rmse_mm": rmse, "mae_mm": mae, "r2": float(r2)}


def _blocked_folds(n: int, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = max(2, min(int(folds), n))
    indices = np.arange(n, dtype=np.int64)
    chunks = np.array_split(indices, folds)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for test_idx in chunks:
        if len(test_idx) == 0:
            continue
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
        if len(train_idx) == 0:
            continue
        out.append((train_idx, test_idx))
    return out


def evaluate_models(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    folds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, FittedModel]]:
    all_idx = np.arange(len(rows), dtype=np.int64)
    y = _target_array(rows)
    model_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    models: dict[str, FittedModel] = {}

    for name, keys in feature_sets():
        try:
            model = _fit_ridge(rows, keys, all_idx, alpha=alpha, model_name=name)
        except RuntimeError:
            continue
        models[name] = model
        pred = _predict(model, rows)
        fit_metrics = _metrics(y, pred)

        cv_pred = np.full(len(rows), math.nan, dtype=np.float64)
        cv_fold_count = 0
        for train_idx, test_idx in _blocked_folds(len(rows), folds):
            try:
                fold_model = _fit_ridge(
                    rows,
                    keys,
                    train_idx,
                    alpha=alpha,
                    model_name=name,
                )
            except RuntimeError:
                continue
            cv_pred[test_idx] = _predict(fold_model, [rows[int(i)] for i in test_idx])
            cv_fold_count += 1
        cv_metrics = _metrics(y, cv_pred)

        model_rows.append(
            {
                "model": name,
                "feature_count": len(keys),
                "train_rows": model.train_rows,
                "ridge_alpha": alpha,
                "fit_n": int(fit_metrics["n"]),
                "fit_r2": fit_metrics["r2"],
                "fit_rmse_mm": fit_metrics["rmse_mm"],
                "fit_mae_mm": fit_metrics["mae_mm"],
                "blocked_folds": cv_fold_count,
                "cv_n": int(cv_metrics["n"]),
                "cv_r2": cv_metrics["r2"],
                "cv_rmse_mm": cv_metrics["rmse_mm"],
                "cv_mae_mm": cv_metrics["mae_mm"],
            }
        )

        for idx, key in enumerate(["intercept"] + keys):
            coefficient_rows.append(
                {
                    "model": name,
                    "feature": key,
                    "standardized_coefficient_mm": float(model.beta[idx]),
                    "fill_value": "" if idx == 0 else float(model.fill_values[idx - 1]),
                    "mean": "" if idx == 0 else float(model.means[idx - 1]),
                    "scale": "" if idx == 0 else float(model.scales[idx - 1]),
                }
            )

    return model_rows, coefficient_rows, models


def choose_model(model_rows: list[dict[str, Any]], requested: str | None) -> str:
    available = {str(row.get("model")) for row in model_rows}
    if requested:
        if requested not in available:
            raise RuntimeError(f"--map-model {requested!r} is not available.")
        return requested
    valid_rows = [row for row in model_rows if np.isfinite(_to_float(row.get("cv_rmse_mm")))]
    if not valid_rows:
        raise RuntimeError("No fitted model with finite validation metrics.")
    return str(min(valid_rows, key=lambda row: _to_float(row.get("cv_rmse_mm"))).get("model"))


def attach_predictions(rows: list[dict[str, Any]], model: FittedModel) -> None:
    y = _target_array(rows)
    pred = _predict(model, rows)
    for idx, row in enumerate(rows):
        row["z_bias_model"] = model.name
        row["predicted_z_bias_mm"] = float(pred[idx])
        row["corrected_target_z_mm"] = float(y[idx] - pred[idx])
        row["corrected_rel_z_mm"] = float(_to_float(row.get("rel_z_mm")) - pred[idx])


def _parse_bins(text: str) -> tuple[int, int]:
    raw = str(text).lower().replace(",", "x").split("x")
    if len(raw) != 2:
        raise RuntimeError("--image-bins must look like 8x6")
    u_bins = max(1, int(raw[0]))
    v_bins = max(1, int(raw[1]))
    return u_bins, v_bins


def build_image_bin_rows(
    rows: list[dict[str, Any]],
    *,
    u_bins: int,
    v_bins: int,
) -> list[dict[str, Any]]:
    u = np.asarray([_to_float(row.get("image_u_centroid_px")) for row in rows], dtype=np.float64)
    v = np.asarray([_to_float(row.get("image_v_centroid_px")) for row in rows], dtype=np.float64)
    target = _target_array(rows)
    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(target)
    if int(np.sum(valid)) == 0:
        return []

    u_edges = np.linspace(float(np.min(u[valid])), float(np.max(u[valid])), u_bins + 1)
    v_edges = np.linspace(float(np.min(v[valid])), float(np.max(v[valid])), v_bins + 1)
    out: list[dict[str, Any]] = []
    branch_arr = np.asarray([str(row.get("branch")) for row in rows], dtype=object)
    for vi in range(v_bins):
        for ui in range(u_bins):
            u_lo, u_hi = float(u_edges[ui]), float(u_edges[ui + 1])
            v_lo, v_hi = float(v_edges[vi]), float(v_edges[vi + 1])
            in_bin = (
                valid
                & (u >= u_lo)
                & (u <= u_hi if ui == u_bins - 1 else u < u_hi)
                & (v >= v_lo)
                & (v <= v_hi if vi == v_bins - 1 else v < v_hi)
            )
            values = target[in_bin]
            out_values = target[in_bin & (branch_arr == "out")]
            return_values = target[in_bin & (branch_arr == "return")]
            out.append(
                {
                    "u_bin": ui,
                    "v_bin": vi,
                    "u_lo_px": u_lo,
                    "u_hi_px": u_hi,
                    "v_lo_px": v_lo,
                    "v_hi_px": v_hi,
                    "u_center_px": float((u_lo + u_hi) * 0.5),
                    "v_center_px": float((v_lo + v_hi) * 0.5),
                    "n": int(np.sum(in_bin)),
                    "target_z_median_mm": _median(values),
                    "target_z_mean_mm": _mean(values),
                    "target_z_p25_mm": _percentile(values, 25),
                    "target_z_p75_mm": _percentile(values, 75),
                    "out_n": int(np.sum(in_bin & (branch_arr == "out"))),
                    "return_n": int(np.sum(in_bin & (branch_arr == "return"))),
                    "return_minus_out_target_z_median_mm": _median(return_values) - _median(out_values),
                }
            )
    return out


def build_model_grid(
    rows: list[dict[str, Any]],
    model: FittedModel,
    *,
    u_bins: int,
    v_bins: int,
) -> list[dict[str, Any]]:
    u = np.asarray([_to_float(row.get("image_u_centroid_px")) for row in rows], dtype=np.float64)
    v = np.asarray([_to_float(row.get("image_v_centroid_px")) for row in rows], dtype=np.float64)
    valid = np.isfinite(u) & np.isfinite(v)
    if int(np.sum(valid)) == 0:
        return []

    template: dict[str, Any] = {}
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    for key in sorted(all_keys):
        values = [_to_float(row.get(key)) for row in rows]
        med = _median(values)
        template[key] = med if np.isfinite(med) else ""

    u_values = np.linspace(float(np.min(u[valid])), float(np.max(u[valid])), u_bins)
    v_values = np.linspace(float(np.min(v[valid])), float(np.max(v[valid])), v_bins)
    grid_rows_for_pred: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for vi, vv in enumerate(v_values):
        for ui, uu in enumerate(u_values):
            row = dict(template)
            row["image_u_centroid_px"] = float(uu)
            row["image_v_centroid_px"] = float(vv)
            grid_rows_for_pred.append(row)
            output_rows.append(
                {
                    "u_index": ui,
                    "v_index": vi,
                    "image_u_centroid_px": float(uu),
                    "image_v_centroid_px": float(vv),
                    "model": model.name,
                }
            )

    # Recompute virtual image polynomial features for the grid while keeping all other
    # features fixed at the data medians.
    grid_u = np.asarray([_to_float(row.get("image_u_centroid_px")) for row in grid_rows_for_pred])
    grid_v = np.asarray([_to_float(row.get("image_v_centroid_px")) for row in grid_rows_for_pred])
    data_u_center = _median(u)
    data_v_center = _median(v)
    width = _median([_to_float(row.get("camera_width_px")) for row in rows])
    height = _median([_to_float(row.get("camera_height_px")) for row in rows])
    if not np.isfinite(width) or width <= 0.0:
        width = max(float(np.nanmax(u[valid]) - np.nanmin(u[valid])), 1.0)
    if not np.isfinite(height) or height <= 0.0:
        height = max(float(np.nanmax(v[valid]) - np.nanmin(v[valid])), 1.0)
    grid_u_norm = (grid_u - data_u_center) / width
    grid_v_norm = (grid_v - data_v_center) / height
    cx = _median([_to_float(row.get("camera_cx_px")) for row in rows])
    cy = _median([_to_float(row.get("camera_cy_px")) for row in rows])
    fx = _median([_to_float(row.get("camera_fx_px")) for row in rows])
    fy = _median([_to_float(row.get("camera_fy_px")) for row in rows])
    for idx, row in enumerate(grid_rows_for_pred):
        row["image_u_norm"] = float(grid_u_norm[idx])
        row["image_v_norm"] = float(grid_v_norm[idx])
        row["image_u_norm2"] = float(grid_u_norm[idx] * grid_u_norm[idx])
        row["image_v_norm2"] = float(grid_v_norm[idx] * grid_v_norm[idx])
        row["image_uv_norm"] = float(grid_u_norm[idx] * grid_v_norm[idx])
        if all(np.isfinite([grid_u[idx], grid_v[idx], cx, cy, fx, fy])) and fx > 0.0 and fy > 0.0:
            radius = math.hypot((grid_u[idx] - cx) / fx, (grid_v[idx] - cy) / fy)
        else:
            radius = _to_float(row.get("image_radius_norm"))
        row["image_radius_norm"] = float(radius)
        row["image_radius_norm2"] = radius * radius if np.isfinite(radius) else math.nan

    pred = _predict(model, grid_rows_for_pred)
    for idx, row in enumerate(output_rows):
        row["predicted_z_bias_mm"] = float(pred[idx])
    return output_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _setup_plot_style(plt) -> None:
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


def plot_results(
    run: dict[str, Any],
    rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
    image_bin_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    *,
    chosen_model: str,
    target: str,
    show: bool,
) -> Path:
    import matplotlib

    matplotlib.use("Agg" if not show else "QtAgg")
    import matplotlib.pyplot as plt

    _setup_plot_style(plt)

    path: Path = run["path"]
    target_suffix = "" if target == "rel-z" else f"_{target.replace('-', '_')}"
    out_path = path.with_name(f"{path.stem}_z_bias_map{target_suffix}_plot.png")
    fig, axes = plt.subplots(3, 2, figsize=(18, 15), sharex=False)
    fig.suptitle("HydraTracker z bias map and feature model", fontsize=16, fontweight="bold")
    fig.text(
        0.01,
        0.965,
        (
            f"{run['run_id']} -- {run.get('timestamp', '')}   "
            f"target={target}   model={chosen_model}   movement={meta['movement_axis']}"
        ),
        fontsize=9,
        ha="left",
        va="top",
    )

    frames = np.asarray([_to_float(row.get("frame")) for row in rows], dtype=np.float64)
    movement = np.asarray([_to_float(row.get("movement_axis_value_mm")) for row in rows], dtype=np.float64)
    target_values = _target_array(rows)
    pred = np.asarray([_to_float(row.get("predicted_z_bias_mm")) for row in rows], dtype=np.float64)
    corrected = np.asarray([_to_float(row.get("corrected_target_z_mm")) for row in rows], dtype=np.float64)
    branches = np.asarray([str(row.get("branch")) for row in rows], dtype=object)
    out_mask = branches == "out"
    return_mask = branches == "return"

    ax = axes[0, 0]
    ax.plot(frames, target_values, color="#d62728", marker="o", markersize=2.5, linewidth=1.2, label="target z")
    ax.plot(frames, pred, color="#1f77b4", linewidth=1.4, label="model prediction")
    ax.plot(frames, corrected, color="#2ca02c", linewidth=1.2, label="target - prediction")
    ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(meta["turn_frame"], color="#c7c7c7", linewidth=1.0)
    ax.set_title("Observed z bias and fitted correction")
    ax.set_xlabel("frame")
    ax.set_ylabel("mm")
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    ax.scatter(movement[out_mask], target_values[out_mask], s=15, color="#1f77b4", alpha=0.65, label="out raw")
    ax.scatter(movement[return_mask], target_values[return_mask], s=15, color="#d62728", alpha=0.65, label="return raw")
    ax.scatter(movement[out_mask], corrected[out_mask], s=14, color="#8ecae6", alpha=0.75, label="out corrected")
    ax.scatter(movement[return_mask], corrected[return_mask], s=14, color="#ffb4a2", alpha=0.75, label="return corrected")
    ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("Raw and corrected z versus movement coordinate")
    ax.set_xlabel(f"delta {meta['movement_axis']} [mm]")
    ax.set_ylabel("target z [mm]")
    ax.legend(loc="best", ncols=2, fontsize=8)

    ax = axes[1, 0]
    if grid_rows:
        u_grid = sorted({_to_float(row.get("image_u_centroid_px")) for row in grid_rows})
        v_grid = sorted({_to_float(row.get("image_v_centroid_px")) for row in grid_rows})
        matrix = np.full((len(v_grid), len(u_grid)), np.nan, dtype=np.float64)
        u_idx = {float(value): idx for idx, value in enumerate(u_grid)}
        v_idx = {float(value): idx for idx, value in enumerate(v_grid)}
        for row in grid_rows:
            uu = _to_float(row.get("image_u_centroid_px"))
            vv = _to_float(row.get("image_v_centroid_px"))
            matrix[v_idx[float(vv)], u_idx[float(uu)]] = _to_float(row.get("predicted_z_bias_mm"))
        im = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            extent=[min(u_grid), max(u_grid), min(v_grid), max(v_grid)],
            cmap="coolwarm",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="predicted z [mm]")
    scatter = ax.scatter(
        [_to_float(row.get("image_u_centroid_px")) for row in rows],
        [_to_float(row.get("image_v_centroid_px")) for row in rows],
        c=target_values,
        s=14,
        cmap="coolwarm",
        edgecolors="none",
        alpha=0.8,
    )
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="observed target [mm]")
    ax.set_title("Image-centroid z bias map")
    ax.set_xlabel("centroid u [px]")
    ax.set_ylabel("centroid v [px]")

    ax = axes[1, 1]
    if image_bin_rows:
        u_bins = sorted({_to_int(row.get("u_bin")) for row in image_bin_rows})
        v_bins = sorted({_to_int(row.get("v_bin")) for row in image_bin_rows})
        matrix = np.full((len(v_bins), len(u_bins)), np.nan, dtype=np.float64)
        for row in image_bin_rows:
            n = _to_int(row.get("n"), 0)
            if n > 0:
                matrix[_to_int(row.get("v_bin")), _to_int(row.get("u_bin"))] = _to_float(row.get("target_z_median_mm"))
        im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="coolwarm")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="median target [mm]")
    ax.set_title("Empirical image-bin median z")
    ax.set_xlabel("u bin")
    ax.set_ylabel("v bin")

    ax = axes[2, 0]
    sorted_models = sorted(
        [row for row in model_rows if np.isfinite(_to_float(row.get("cv_rmse_mm")))],
        key=lambda row: _to_float(row.get("cv_rmse_mm")),
    )
    labels = [str(row.get("model")) for row in sorted_models]
    values = [_to_float(row.get("cv_rmse_mm")) for row in sorted_models]
    colors = ["#1f77b4" if label == chosen_model else "#8fa6c6" for label in labels]
    y_pos = np.arange(len(labels), dtype=np.float64)
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Blocked-validation RMSE by feature family")
    ax.set_xlabel("RMSE [mm]")

    ax = axes[2, 1]
    coeffs = [
        row
        for row in coefficient_rows
        if str(row.get("model")) == chosen_model and str(row.get("feature")) != "intercept"
    ]
    coeffs = sorted(
        coeffs,
        key=lambda row: abs(_to_float(row.get("standardized_coefficient_mm"))),
        reverse=True,
    )[:12]
    coeff_labels = [str(row.get("feature")) for row in coeffs]
    coeff_values = [_to_float(row.get("standardized_coefficient_mm")) for row in coeffs]
    coeff_pos = np.arange(len(coeff_labels), dtype=np.float64)
    coeff_colors = ["#d62728" if value >= 0.0 else "#1f77b4" for value in coeff_values]
    ax.barh(coeff_pos, coeff_values, color=coeff_colors)
    ax.set_yticks(coeff_pos, coeff_labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.0, color="#999999", linewidth=1.0)
    ax.set_title(f"Top standardized coefficients: {chosen_model}")
    ax.set_xlabel("mm per standardized feature")

    for ax in axes.flat:
        ax.grid(True, alpha=0.85)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _latest_run_path() -> Path:
    default_dir = Path(__file__).resolve().parents[1] / "tests" / "hydramarker_tracker_runs"
    files = sorted(default_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError(f"No JSONL runs found in {default_dir}")
    return files[-1]


def _default_runs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "tests" / "hydramarker_tracker_runs"


def _select_run_path_qt(initial_dir: Path | None = None) -> Path:
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog
    except Exception as exc:
        raise RuntimeError(
            "Qt file dialog is unavailable. Install/use PySide6 or pass a .jsonl path directly."
        ) from exc

    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv[:1])

    file_name, _selected_filter = QFileDialog.getOpenFileName(
        None,
        "Select HydraTracker JSONL run",
        str((initial_dir or _default_runs_dir()).resolve()),
        "HydraTracker runs (*.jsonl);;All files (*)",
    )
    if owns_app:
        app.quit()

    if not file_name:
        raise RuntimeError("No JSONL run selected.")
    return Path(file_name)


def _parse_args(argv: list[str]) -> dict[str, Any]:
    args: dict[str, Any] = {
        "path": None,
        "use_latest": False,
        "show": False,
        "make_plot": True,
        "point_set": "pose",
        "target": "rel-z",
        "uncertainty_csv": "auto",
        "max_gap_frames": 2,
        "max_frames": None,
        "folds": 5,
        "ridge_alpha": 1.0,
        "map_model": None,
        "image_bins": (8, 6),
        "grid_bins": (40, 30),
    }
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--latest":
            args["use_latest"] = True
        elif arg == "--show":
            args["show"] = True
        elif arg == "--no-plot":
            args["make_plot"] = False
        elif arg == "--point-set":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--point-set needs 'pose' or 'correspondence'")
            point_set = str(argv[idx]).strip().lower()
            if point_set not in ("pose", "correspondence"):
                raise RuntimeError("--point-set must be 'pose' or 'correspondence'")
            args["point_set"] = point_set
        elif arg == "--target":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--target needs rel-z, demeaned-z, or xy-residual")
            args["target"] = str(argv[idx]).strip().lower()
        elif arg == "--uncertainty-csv":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--uncertainty-csv needs PATH, auto, or none")
            args["uncertainty_csv"] = str(argv[idx])
        elif arg == "--max-gap-frames":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--max-gap-frames needs an integer")
            args["max_gap_frames"] = int(argv[idx])
        elif arg == "--max-frames":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--max-frames needs an integer")
            args["max_frames"] = int(argv[idx])
        elif arg == "--folds":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--folds needs an integer")
            args["folds"] = int(argv[idx])
        elif arg == "--ridge-alpha":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--ridge-alpha needs a numeric value")
            args["ridge_alpha"] = float(argv[idx])
        elif arg == "--map-model":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--map-model needs a model name")
            args["map_model"] = str(argv[idx]).strip()
        elif arg == "--image-bins":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--image-bins needs a value like 8x6")
            args["image_bins"] = _parse_bins(str(argv[idx]))
        elif arg == "--grid-bins":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--grid-bins needs a value like 40x30")
            args["grid_bins"] = _parse_bins(str(argv[idx]))
        elif arg.endswith(".jsonl"):
            args["path"] = Path(arg)
        else:
            raise RuntimeError(f"Unknown option: {arg}")
        idx += 1
    return args


def _print_summary(
    *,
    path: Path,
    target: str,
    chosen_model: str,
    model_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    paths: dict[str, Path],
) -> None:
    for label in ("frame", "model", "coefficient", "image_bin", "grid", "plot"):
        if label in paths:
            print(f"[res_z_bias_map] saved {label:11s} -> {paths[label].resolve()}")

    target_values = _target_array(rows)
    corrected = np.asarray([_to_float(row.get("corrected_target_z_mm")) for row in rows], dtype=np.float64)
    pred = np.asarray([_to_float(row.get("predicted_z_bias_mm")) for row in rows], dtype=np.float64)
    chosen = next((row for row in model_rows if str(row.get("model")) == chosen_model), {})
    baseline = next((row for row in model_rows if str(row.get("model")) == "constant"), {})
    branches = np.asarray([str(row.get("branch")) for row in rows], dtype=object)
    out_target = target_values[branches == "out"]
    return_target = target_values[branches == "return"]
    out_corr = corrected[branches == "out"]
    return_corr = corrected[branches == "return"]

    print(
        "[res_z_bias_map] target "
        f"{target}: range={float(np.nanmax(target_values) - np.nanmin(target_values)):.3f} mm, "
        f"pred_corr={_corr(target_values, pred):+.3f}"
    )
    if baseline:
        print(
            "[res_z_bias_map] baseline blocked CV "
            f"R2={_to_float(baseline.get('cv_r2')):+.3f}, "
            f"RMSE={_to_float(baseline.get('cv_rmse_mm')):.3f} mm"
        )
    if chosen:
        print(
            "[res_z_bias_map] chosen model "
            f"{chosen_model}: blocked CV R2={_to_float(chosen.get('cv_r2')):+.3f}, "
            f"RMSE={_to_float(chosen.get('cv_rmse_mm')):.3f} mm, "
            f"fit R2={_to_float(chosen.get('fit_r2')):+.3f}"
        )
    if len(out_target) and len(return_target):
        print(
            "[res_z_bias_map] branch median before/after correction "
            f"raw={_median(return_target) - _median(out_target):+.3f} mm, "
            f"corrected={_median(return_corr) - _median(out_corr):+.3f} mm"
        )

    ranked = sorted(
        [row for row in model_rows if np.isfinite(_to_float(row.get("cv_rmse_mm")))],
        key=lambda row: _to_float(row.get("cv_rmse_mm")),
    )[:5]
    if ranked:
        print("[res_z_bias_map] top validation models:")
        for row in ranked:
            print(
                "  "
                f"{row['model']}: "
                f"R2={_to_float(row.get('cv_r2')):+.3f}, "
                f"RMSE={_to_float(row.get('cv_rmse_mm')):.3f} mm"
            )
    print(f"[res_z_bias_map] input -> {path.resolve()}")


def main() -> None:
    args = _parse_args(sys.argv[1:])
    path = args["path"]
    if path is None:
        path = _latest_run_path() if args["use_latest"] else _select_run_path_qt()
    path = Path(path).resolve()
    run = load_run(path)

    uncertainty_arg = str(args["uncertainty_csv"])
    if uncertainty_arg == "none":
        uncertainty_by_frame: dict[int, dict[str, str]] = {}
    else:
        uncertainty_path = (
            path.with_name(f"{path.stem}_pose_uncertainty.csv")
            if uncertainty_arg in ("auto", "")
            else Path(uncertainty_arg).resolve()
        )
        uncertainty_by_frame = _read_csv_by_frame(uncertainty_path)

    rows, meta = build_frame_rows(
        run,
        point_set=str(args["point_set"]),
        uncertainty_by_frame=uncertainty_by_frame,
        max_gap_frames=int(args["max_gap_frames"]),
        max_frames=args["max_frames"],
    )
    for row in rows:
        row["camera_width_px"] = _to_float(run["camera"].get("width"))
        row["camera_height_px"] = _to_float(run["camera"].get("height"))
        row["camera_cx_px"] = _to_float(run["camera"].get("cx"))
        row["camera_cy_px"] = _to_float(run["camera"].get("cy"))
        row["camera_fx_px"] = _to_float(run["camera"].get("fx"))
        row["camera_fy_px"] = _to_float(run["camera"].get("fy"))

    target = str(args["target"])
    add_derived_features(rows, target=target)
    model_rows, coefficient_rows, models = evaluate_models(
        rows,
        alpha=float(args["ridge_alpha"]),
        folds=int(args["folds"]),
    )
    chosen_model = choose_model(model_rows, args["map_model"])
    attach_predictions(rows, models[chosen_model])

    image_u_bins, image_v_bins = args["image_bins"]
    grid_u_bins, grid_v_bins = args["grid_bins"]
    image_bin_rows = build_image_bin_rows(rows, u_bins=image_u_bins, v_bins=image_v_bins)
    grid_rows = build_model_grid(rows, models[chosen_model], u_bins=grid_u_bins, v_bins=grid_v_bins)

    target_suffix = "" if target == "rel-z" else f"_{target.replace('-', '_')}"
    frame_csv = path.with_name(f"{path.stem}_z_bias_map{target_suffix}_frames.csv")
    model_csv = path.with_name(f"{path.stem}_z_bias_map{target_suffix}_models.csv")
    coefficient_csv = path.with_name(f"{path.stem}_z_bias_map{target_suffix}_coefficients.csv")
    image_bin_csv = path.with_name(f"{path.stem}_z_bias_map{target_suffix}_image_bins.csv")
    grid_csv = path.with_name(f"{path.stem}_z_bias_map{target_suffix}_grid.csv")
    _write_csv(frame_csv, rows)
    _write_csv(model_csv, model_rows)
    _write_csv(coefficient_csv, coefficient_rows)
    _write_csv(image_bin_csv, image_bin_rows)
    _write_csv(grid_csv, grid_rows)

    paths = {
        "frame": frame_csv,
        "model": model_csv,
        "coefficient": coefficient_csv,
        "image_bin": image_bin_csv,
        "grid": grid_csv,
    }
    if bool(args["make_plot"]):
        try:
            paths["plot"] = plot_results(
                run,
                rows,
                model_rows,
                coefficient_rows,
                image_bin_rows,
                grid_rows,
                meta,
                chosen_model=chosen_model,
                target=target,
                show=bool(args["show"]),
            )
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print("[res_z_bias_map] WARNING: matplotlib is not installed; skipped plot.")

    _print_summary(
        path=path,
        target=target,
        chosen_model=chosen_model,
        model_rows=model_rows,
        rows=rows,
        paths=paths,
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[res_z_bias_map] ERROR: {exc}")
        sys.exit(1)
