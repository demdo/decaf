"""Analyze pose bias as a function of visible HydraMarker corners.

This offline diagnostic groups logged poses by corner coverage and studies how
specific feature subsets correlate with translation and rotation bias.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
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


def _camera_center_from_run_start(record: dict[str, Any]) -> tuple[float, float]:
    info = dict(record.get("camera_intrinsics") or {})
    K = (
        info.get("tracker_K")
        or info.get("K")
        or info.get("camera_matrix")
        or info.get("raw_realsense_K")
    )
    if K is not None:
        try:
            arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
            return float(arr[0, 2]), float(arr[1, 2])
        except Exception:
            pass
    return (
        _to_float(info.get("ppx", info.get("cx"))),
        _to_float(info.get("ppy", info.get("cy"))),
    )


def load_run(path: Path) -> dict[str, Any]:
    run_id = path.stem
    timestamp = ""
    frames: dict[int, dict[str, Any]] = {}
    details: dict[int, dict[str, Any]] = {}
    summary: dict[str, Any] = {}
    camera_center = (math.nan, math.nan)

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
                camera_center = _camera_center_from_run_start(record)
                continue

            if record_type == "frame":
                data = dict(record.get("data") or {})
                frame = _to_int(data.get("frame"), default=len(frames))
                frames[frame] = data
                continue

            if record_type == "frame_detail":
                frame = _to_int(record.get("frame"), default=-1)
                if frame >= 0:
                    details[frame] = dict(record)
                continue

            if record_type == "run_summary":
                summary = dict(record.get("summary") or {})

    if not frames:
        raise RuntimeError(f"No frame records found in:\n{path}")

    return {
        "path": path,
        "run_id": run_id,
        "timestamp": timestamp,
        "camera_center": camera_center,
        "frames": frames,
        "details": details,
        "summary": summary,
    }


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


def _radial_tangential_residual(
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


def _model_residuals(
    frame_rows: list[dict[str, Any]],
    *,
    include_branch: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    y = np.asarray([_to_float(row.get("rel_z_mm")) for row in frame_rows], dtype=np.float64)
    movement = np.asarray(
        [_to_float(row.get("movement_axis_value_mm")) for row in frame_rows],
        dtype=np.float64,
    )
    branch_return = np.asarray(
        [1.0 if str(row.get("branch")) == "return" else 0.0 for row in frame_rows],
        dtype=np.float64,
    )
    mask = np.isfinite(y) & np.isfinite(movement)
    residuals = np.full(len(frame_rows), np.nan, dtype=np.float64)
    if int(np.sum(mask)) < 5:
        return residuals, {"r2": math.nan, "rmse_mm": math.nan, "branch_coeff_mm": math.nan}

    x = movement[mask]
    scale = float(np.std(x))
    if scale <= 0.0:
        scale = 1.0
    x_norm = (x - float(np.mean(x))) / scale
    cols = [np.ones_like(x_norm), x_norm, x_norm * x_norm]
    branch_coeff_index = -1
    if include_branch:
        branch_coeff_index = len(cols)
        cols.append(branch_return[mask])
    design = np.column_stack(cols)
    yy = y[mask]
    beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
    pred = design @ beta
    residuals[mask] = yy - pred
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan
    rmse = float(math.sqrt(float(np.mean((yy - pred) ** 2))))
    branch_coeff = float(beta[branch_coeff_index]) if include_branch else math.nan
    return residuals, {"r2": r2, "rmse_mm": rmse, "branch_coeff_mm": branch_coeff}


def analyze_run(
    run: dict[str, Any],
    *,
    point_set: str,
    bin_width_mm: float,
    uncertainty_by_frame: dict[int, dict[str, str]] | None = None,
    max_frames: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = run["frames"]
    details: dict[int, dict[str, Any]] = run["details"]
    cx, cy = run["camera_center"]
    uncertainty_by_frame = uncertainty_by_frame or {}

    sorted_frames = sorted(frames)
    if max_frames is not None and max_frames > 0:
        sorted_frames = sorted_frames[:max_frames]

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

    origin = tvec_arr[int(np.where(valid_tvec)[0][0])].copy()
    rel_tvec = tvec_arr - origin
    ranges = np.nanmax(rel_tvec[valid_tvec], axis=0) - np.nanmin(rel_tvec[valid_tvec], axis=0)
    movement_axis_idx = int(np.nanargmax(ranges))
    movement_axis = COMPONENTS[movement_axis_idx]
    movement_values = rel_tvec[:, movement_axis_idx]
    turn_idx = int(np.nanargmax(np.abs(movement_values)))
    turn_frame = sorted_frames[turn_idx]

    frame_rows: list[dict[str, Any]] = []
    corner_rows: list[dict[str, Any]] = []

    for idx, frame in enumerate(sorted_frames):
        data = frames[frame]
        detail = details.get(frame, {})
        tvec = tvec_arr[idx]
        rel = rel_tvec[idx]
        branch = "out" if frame <= turn_frame else "return"

        corners_key = "pose_corners" if point_set == "pose" else "correspondence_corners"
        corners = list(detail.get(corners_key) or [])
        if not corners and point_set == "pose":
            corners = list(detail.get("correspondence_corners") or [])

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

        for corner_idx, corner in enumerate(corners):
            row = _to_int(corner.get("global_row"), default=_to_int(corner.get("local_row"), -999))
            col = _to_int(corner.get("global_col"), default=_to_int(corner.get("local_col"), -999))
            u, v = _corner_uv(corner)
            x, y_obj, z_obj = _corner_xyz(corner)
            du, dv, error = _corner_residual(corner)
            radial, tangential = _radial_tangential_residual(u, v, du, dv, cx, cy)
            vote = _to_float(corner.get("votes"))

            if row != -999:
                rows.append(row)
            if col != -999:
                cols.append(col)
            us.append(u)
            vs.append(v)
            xs.append(x)
            ys.append(y_obj)
            zs.append(z_obj)
            dus.append(du)
            dvs.append(dv)
            errors.append(error)
            radial_residuals.append(radial)
            tangential_residuals.append(tangential)
            votes.append(vote)

            corner_rows.append(
                {
                    "frame": frame,
                    "branch": branch,
                    "movement_axis": movement_axis,
                    "movement_axis_value_mm": float(rel[movement_axis_idx]),
                    "rel_x_mm": float(rel[0]),
                    "rel_y_mm": float(rel[1]),
                    "rel_z_mm": float(rel[2]),
                    "corner_index": corner_idx,
                    "global_row": row if row != -999 else "",
                    "global_col": col if col != -999 else "",
                    "corner_id": f"r{row}_c{col}" if row != -999 and col != -999 else "",
                    "u_px": u,
                    "v_px": v,
                    "x_mm": x,
                    "y_mm": y_obj,
                    "z_mm": z_obj,
                    "du_px": du,
                    "dv_px": dv,
                    "error_px": error,
                    "radial_residual_px": radial,
                    "tangential_residual_px": tangential,
                    "votes": vote,
                }
            )

        row_values = _finite(np.asarray(rows, dtype=np.float64))
        col_values = _finite(np.asarray(cols, dtype=np.float64))
        uncertainty = uncertainty_by_frame.get(frame, {})
        frame_row: dict[str, Any] = {
            "frame": frame,
            "branch": branch,
            "movement_axis": movement_axis,
            "movement_axis_value_mm": float(rel[movement_axis_idx]),
            "turn_frame": int(turn_frame),
            "success": _to_int(data.get("success"), default=0),
            "pose_source": str(data.get("pose_source") or ""),
            "pnp_method": str(data.get("pnp_method") or ""),
            "tvec_x_mm": float(tvec[0]),
            "tvec_y_mm": float(tvec[1]),
            "tvec_z_mm": float(tvec[2]),
            "rel_x_mm": float(rel[0]),
            "rel_y_mm": float(rel[1]),
            "rel_z_mm": float(rel[2]),
            "camera_roll_deg": _to_float(data.get("camera_roll_deg")),
            "camera_pitch_deg": _to_float(data.get("camera_pitch_deg")),
            "camera_yaw_deg": _to_float(data.get("camera_yaw_deg")),
            "num_points_logged": _to_float(data.get("num_points")),
            "point_count_used": float(len(corners)),
            "row_min": float(np.min(row_values)) if len(row_values) else math.nan,
            "row_max": float(np.max(row_values)) if len(row_values) else math.nan,
            "row_span": float(np.ptp(row_values)) if len(row_values) else math.nan,
            "col_min": float(np.min(col_values)) if len(col_values) else math.nan,
            "col_max": float(np.max(col_values)) if len(col_values) else math.nan,
            "col_span": float(np.ptp(col_values)) if len(col_values) else math.nan,
            "distinct_rows": float(len(set(rows))) if rows else math.nan,
            "distinct_cols": float(len(set(cols))) if cols else math.nan,
            "image_u_centroid_px": _mean(us),
            "image_v_centroid_px": _mean(vs),
            "image_u_span_px": float(np.ptp(_finite(us))) if len(_finite(us)) else math.nan,
            "image_v_span_px": float(np.ptp(_finite(vs))) if len(_finite(vs)) else math.nan,
            "object_x_span_mm": float(np.ptp(_finite(xs))) if len(_finite(xs)) else math.nan,
            "object_y_span_mm": float(np.ptp(_finite(ys))) if len(_finite(ys)) else math.nan,
            "object_z_span_mm": float(np.ptp(_finite(zs))) if len(_finite(zs)) else math.nan,
            "residual_mean_du_px": _mean(dus),
            "residual_mean_dv_px": _mean(dvs),
            "residual_abs_mean_du_px": _mean(np.abs(np.asarray(dus, dtype=np.float64))),
            "residual_abs_mean_dv_px": _mean(np.abs(np.asarray(dvs, dtype=np.float64))),
            "residual_rms_px": _rms(errors),
            "residual_median_px": _median(errors),
            "residual_p95_px": _percentile(errors, 95),
            "residual_max_px": float(np.max(_finite(errors))) if len(_finite(errors)) else math.nan,
            "residual_radial_mean_px": _mean(radial_residuals),
            "residual_tangential_mean_px": _mean(tangential_residuals),
            "votes_mean": _mean(votes),
            "votes_min": float(np.min(_finite(votes))) if len(_finite(votes)) else math.nan,
        }

        for key in ("sigma_z_mm", "jacobian_condition", "reproj_rms_px", "reproj_p95_px"):
            if key in uncertainty:
                frame_row[key] = _to_float(uncertainty.get(key))

        frame_rows.append(frame_row)

    residual_y, model_y = _model_residuals(frame_rows, include_branch=False)
    residual_y_branch, model_y_branch = _model_residuals(frame_rows, include_branch=True)
    for idx, row in enumerate(frame_rows):
        row["z_resid_y_quad_mm"] = float(residual_y[idx])
        row["z_resid_y_quad_branch_mm"] = float(residual_y_branch[idx])

    ybin_rows = _build_ybin_rows(frame_rows, bin_width_mm)
    group_rows = []
    for kind in ("global_row", "global_col", "cell"):
        group_rows.extend(_build_group_stats(frame_rows, corner_rows, kind))

    meta = {
        "movement_axis": movement_axis,
        "turn_frame": int(turn_frame),
        "model_y_r2": model_y["r2"],
        "model_y_rmse_mm": model_y["rmse_mm"],
        "model_y_branch_r2": model_y_branch["r2"],
        "model_y_branch_rmse_mm": model_y_branch["rmse_mm"],
        "model_branch_coeff_mm": model_y_branch["branch_coeff_mm"],
    }
    return frame_rows, corner_rows, ybin_rows, group_rows, meta


def _build_ybin_rows(frame_rows: list[dict[str, Any]], bin_width_mm: float) -> list[dict[str, Any]]:
    movement = _finite([_to_float(row.get("movement_axis_value_mm")) for row in frame_rows])
    if len(movement) == 0:
        return []
    width = max(float(bin_width_mm), 1e-6)
    low = math.floor(float(np.min(movement)) / width) * width
    high = math.ceil(float(np.max(movement)) / width) * width
    bins = np.arange(low, high + width * 1.5, width, dtype=np.float64)

    out: list[dict[str, Any]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        center = float((lo + hi) * 0.5)
        row: dict[str, Any] = {"movement_bin_lo_mm": float(lo), "movement_bin_hi_mm": float(hi), "movement_bin_center_mm": center}
        branch_values: dict[tuple[str, str], float] = {}
        for branch in ("out", "return"):
            subset = [
                r
                for r in frame_rows
                if str(r.get("branch")) == branch
                and lo <= _to_float(r.get("movement_axis_value_mm")) < hi
            ]
            row[f"{branch}_n"] = len(subset)
            for key in (
                "rel_z_mm",
                "point_count_used",
                "residual_rms_px",
                "residual_mean_du_px",
                "residual_mean_dv_px",
                "sigma_z_mm",
                "jacobian_condition",
                "camera_roll_deg",
                "camera_pitch_deg",
                "camera_yaw_deg",
            ):
                value = _median([_to_float(r.get(key)) for r in subset])
                row[f"{branch}_{key}_median"] = value
                branch_values[(branch, key)] = value

        row["return_minus_out_z_median_mm"] = (
            branch_values.get(("return", "rel_z_mm"), math.nan)
            - branch_values.get(("out", "rel_z_mm"), math.nan)
        )
        row["return_minus_out_residual_rms_median_px"] = (
            branch_values.get(("return", "residual_rms_px"), math.nan)
            - branch_values.get(("out", "residual_rms_px"), math.nan)
        )
        if row["out_n"] > 0 and row["return_n"] > 0:
            out.append(row)
    return out


def _build_group_stats(
    frame_rows: list[dict[str, Any]],
    corner_rows: list[dict[str, Any]],
    kind: str,
) -> list[dict[str, Any]]:
    frame_by_id = {_to_int(row.get("frame"), -1): row for row in frame_rows}
    all_frames = sorted(frame_by_id)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for corner in corner_rows:
        row = corner.get("global_row")
        col = corner.get("global_col")
        if row == "" or col == "":
            continue
        if kind == "global_row":
            key = f"row_{row}"
        elif kind == "global_col":
            key = f"col_{col}"
        else:
            key = f"r{row}_c{col}"
        groups[key].append(corner)

    out: list[dict[str, Any]] = []
    for key, observations in sorted(groups.items()):
        visible_frames = sorted({_to_int(obs.get("frame"), -1) for obs in observations})
        visible_frames = [frame for frame in visible_frames if frame in frame_by_id]
        present_set = set(visible_frames)
        present_rows = [frame_by_id[frame] for frame in visible_frames]
        absent_rows = [frame_by_id[frame] for frame in all_frames if frame not in present_set]

        stats: dict[str, Any] = {
            "group_kind": kind,
            "group": key,
            "observations": len(observations),
            "frames_visible": len(visible_frames),
            "presence_fraction": len(visible_frames) / max(1, len(all_frames)),
            "z_resid_y_present_median_mm": _median([_to_float(r.get("z_resid_y_quad_mm")) for r in present_rows]),
            "z_resid_y_absent_median_mm": _median([_to_float(r.get("z_resid_y_quad_mm")) for r in absent_rows]),
            "z_resid_y_branch_present_median_mm": _median(
                [_to_float(r.get("z_resid_y_quad_branch_mm")) for r in present_rows]
            ),
            "z_resid_y_branch_absent_median_mm": _median(
                [_to_float(r.get("z_resid_y_quad_branch_mm")) for r in absent_rows]
            ),
        }
        stats["z_resid_y_present_minus_absent_mm"] = (
            stats["z_resid_y_present_median_mm"] - stats["z_resid_y_absent_median_mm"]
        )
        stats["z_resid_y_branch_present_minus_absent_mm"] = (
            stats["z_resid_y_branch_present_median_mm"]
            - stats["z_resid_y_branch_absent_median_mm"]
        )

        for branch in ("out", "return"):
            obs_b = [obs for obs in observations if str(obs.get("branch")) == branch]
            frames_b = sorted({_to_int(obs.get("frame"), -1) for obs in obs_b})
            frames_b = [frame for frame in frames_b if frame in frame_by_id]
            row_b = [frame_by_id[frame] for frame in frames_b]
            stats[f"{branch}_observations"] = len(obs_b)
            stats[f"{branch}_frames_visible"] = len(frames_b)
            stats[f"{branch}_z_visible_median_mm"] = _median([_to_float(r.get("rel_z_mm")) for r in row_b])
            stats[f"{branch}_movement_visible_median_mm"] = _median(
                [_to_float(r.get("movement_axis_value_mm")) for r in row_b]
            )
            stats[f"{branch}_mean_du_px"] = _mean([_to_float(obs.get("du_px")) for obs in obs_b])
            stats[f"{branch}_mean_dv_px"] = _mean([_to_float(obs.get("dv_px")) for obs in obs_b])
            stats[f"{branch}_mean_error_px"] = _mean([_to_float(obs.get("error_px")) for obs in obs_b])
            stats[f"{branch}_mean_votes"] = _mean([_to_float(obs.get("votes")) for obs in obs_b])

        stats["return_minus_out_z_visible_median_mm"] = (
            stats["return_z_visible_median_mm"] - stats["out_z_visible_median_mm"]
        )
        stats["return_minus_out_mean_du_px"] = stats["return_mean_du_px"] - stats["out_mean_du_px"]
        stats["return_minus_out_mean_dv_px"] = stats["return_mean_dv_px"] - stats["out_mean_dv_px"]
        stats["return_minus_out_error_px"] = stats["return_mean_error_px"] - stats["out_mean_error_px"]
        stats["return_minus_out_residual_vector_px"] = math.hypot(
            stats["return_minus_out_mean_du_px"],
            stats["return_minus_out_mean_dv_px"],
        )
        out.append(stats)
    return out


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


def _frame_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_to_float(row.get(key)) for row in rows], dtype=np.float64)


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


def plot_bias(
    run: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    ybin_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    *,
    show: bool,
) -> Path:
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_plot_style(plt)

    path: Path = run["path"]
    out_path = path.with_name(f"{path.stem}_pose_bias_by_corners_plot.png")

    frames = _frame_array(frame_rows, "frame")
    movement = _frame_array(frame_rows, "movement_axis_value_mm")
    rel_z = _frame_array(frame_rows, "rel_z_mm")
    point_count = _frame_array(frame_rows, "point_count_used")
    row_min = _frame_array(frame_rows, "row_min")
    row_max = _frame_array(frame_rows, "row_max")
    rms = _frame_array(frame_rows, "residual_rms_px")
    mean_du = _frame_array(frame_rows, "residual_mean_du_px")
    mean_dv = _frame_array(frame_rows, "residual_mean_dv_px")
    branches = np.asarray([str(row.get("branch")) for row in frame_rows], dtype=object)

    fig, axes = plt.subplots(3, 2, figsize=(18, 13))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.06, top=0.9, hspace=0.42, wspace=0.18)

    run_label = str(run["run_id"])
    if run.get("timestamp"):
        run_label += f"  |  {run['timestamp']}"
    fig.suptitle("HydraTracker pose bias by corners", fontsize=16, fontweight="bold", y=0.985)
    fig.text(0.5, 0.955, run_label, ha="center", va="top", fontsize=11)
    fig.text(
        0.01,
        0.925,
        (
            f"movement_axis={meta['movement_axis']}   turn_frame={meta['turn_frame']}   "
            f"R2 y-only={meta['model_y_r2']:.3f}   "
            f"R2 y+branch={meta['model_y_branch_r2']:.3f}   "
            f"branch_coeff={meta['model_branch_coeff_mm']:+.3f} mm"
        ),
        ha="left",
        va="top",
        fontsize=10,
    )

    ax = axes[0, 0]
    for branch, color, label in (("out", "#1f77b4", "out"), ("return", "#d62728", "return")):
        mask = branches == branch
        ax.plot(
            movement[mask],
            rel_z[mask],
            color=color,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=label,
        )
    ax.axhline(0.0, color="#d0d4dc", linewidth=1.0)
    ax.set_title("Z hysteresis at same movement coordinate")
    ax.set_xlabel(f"delta {meta['movement_axis']} [mm]")
    ax.set_ylabel("delta z [mm]")
    ax.legend(loc="best")

    ax = axes[0, 1]
    bin_centers = _frame_array(ybin_rows, "movement_bin_center_mm")
    bin_diff = _frame_array(ybin_rows, "return_minus_out_z_median_mm")
    ax.bar(bin_centers, bin_diff, width=max(1.0, 0.8 * abs(_median(np.diff(bin_centers))) if len(bin_centers) > 1 else 8.0), color="#8c564b")
    ax.axhline(0.0, color="#d0d4dc", linewidth=1.0)
    ax.set_title("Return minus out z median by matched movement bins")
    ax.set_xlabel(f"delta {meta['movement_axis']} bin center [mm]")
    ax.set_ylabel("return - out z [mm]")

    ax = axes[1, 0]
    ax.plot(frames, point_count, color="#4c78a8", linewidth=1.4, label="point count")
    ax.set_ylabel("points")
    ax2 = ax.twinx()
    ax2.plot(frames, row_min, color="#f58518", linewidth=1.0, label="row min")
    ax2.plot(frames, row_max, color="#b279a2", linewidth=1.0, label="row max")
    ax2.set_ylabel("global row")
    ax.set_title("Visible point count and row range")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    ax = axes[1, 1]
    ax.plot(frames, rms, color="#f58518", linewidth=1.3, label="RMS error")
    ax.plot(frames, mean_du, color="#1f77b4", linewidth=1.0, label="mean du")
    ax.plot(frames, mean_dv, color="#2ca02c", linewidth=1.0, label="mean dv")
    ax.axhline(0.0, color="#d0d4dc", linewidth=1.0)
    ax.set_title("Frame reprojection residual summary")
    ax.set_xlabel("frame")
    ax.set_ylabel("px")
    ax.legend(loc="best")

    cell_rows = [row for row in group_rows if row.get("group_kind") == "cell"]
    heat_ax = axes[2, 0]
    quiver_ax = axes[2, 1]
    if cell_rows:
        parsed: list[tuple[int, int, dict[str, Any]]] = []
        for row in cell_rows:
            group = str(row.get("group", ""))
            try:
                row_s, col_s = group[1:].split("_c", 1)
                parsed.append((int(row_s), int(col_s), row))
            except Exception:
                continue
        if parsed:
            row_values = sorted({r for r, _c, _row in parsed})
            col_values = sorted({c for _r, c, _row in parsed})
            r_index = {r: i for i, r in enumerate(row_values)}
            c_index = {c: i for i, c in enumerate(col_values)}
            error_matrix = np.full((len(row_values), len(col_values)), np.nan, dtype=np.float64)
            vector_matrix = np.full((len(row_values), len(col_values)), np.nan, dtype=np.float64)
            xs: list[float] = []
            ys: list[float] = []
            us: list[float] = []
            vs: list[float] = []
            for r, c, row in parsed:
                i = r_index[r]
                j = c_index[c]
                out_err = _to_float(row.get("out_mean_error_px"))
                ret_err = _to_float(row.get("return_mean_error_px"))
                error_matrix[i, j] = _mean([out_err, ret_err])
                du = _to_float(row.get("return_minus_out_mean_du_px"))
                dv = _to_float(row.get("return_minus_out_mean_dv_px"))
                vector_matrix[i, j] = _to_float(row.get("return_minus_out_residual_vector_px"))
                if np.isfinite(du) and np.isfinite(dv):
                    xs.append(c)
                    ys.append(r)
                    us.append(du)
                    vs.append(dv)

            im = heat_ax.imshow(error_matrix, origin="lower", aspect="auto", cmap="magma")
            heat_ax.set_xticks(range(len(col_values)), col_values)
            heat_ax.set_yticks(range(len(row_values)), row_values)
            heat_ax.set_xlabel("global col")
            heat_ax.set_ylabel("global row")
            heat_ax.set_title("Mean reprojection error by corner cell")
            fig.colorbar(im, ax=heat_ax, fraction=0.046, pad=0.04, label="px")

            if xs:
                quiver_ax.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=1.0, color="#7f3c8d")
            quiver_ax.set_xticks(col_values)
            quiver_ax.set_yticks(row_values)
            quiver_ax.set_xlabel("global col")
            quiver_ax.set_ylabel("global row")
            quiver_ax.set_title("Return - out mean residual vector by cell [px]")
            quiver_ax.grid(True)
    else:
        heat_ax.set_title("Mean reprojection error by corner cell")
        heat_ax.text(0.5, 0.5, "no cell stats", ha="center", va="center", transform=heat_ax.transAxes)
        quiver_ax.set_title("Return - out residual vector by cell")
        quiver_ax.text(0.5, 0.5, "no cell stats", ha="center", va="center", transform=quiver_ax.transAxes)

    for ax in axes.flat:
        ax.grid(True, alpha=0.85)

    fig.savefig(out_path, dpi=160)
    print(f"[res_pose_bias_by_corners] saved plot -> {out_path.resolve()}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _print_summary(
    frame_rows: list[dict[str, Any]],
    ybin_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    rel_z = _frame_array(frame_rows, "rel_z_mm")
    branches = np.asarray([str(row.get("branch")) for row in frame_rows], dtype=object)
    out_z = rel_z[branches == "out"]
    return_z = rel_z[branches == "return"]
    ybin_diff = _frame_array(ybin_rows, "return_minus_out_z_median_mm")
    ybin_diff_finite = ybin_diff[np.isfinite(ybin_diff)]
    out_z_median = _median(out_z)
    return_z_median = _median(return_z)
    branch_diff = (
        return_z_median - out_z_median
        if np.isfinite(return_z_median) and np.isfinite(out_z_median)
        else math.nan
    )

    print(f"[res_pose_bias_by_corners] saved frame csv -> {paths['frame'].resolve()}")
    print(f"[res_pose_bias_by_corners] saved corner csv -> {paths['corner'].resolve()}")
    print(f"[res_pose_bias_by_corners] saved y-bin csv -> {paths['ybin'].resolve()}")
    print(f"[res_pose_bias_by_corners] saved group csv -> {paths['group'].resolve()}")
    if "plot" in paths:
        print(f"[res_pose_bias_by_corners] saved png -> {paths['plot'].resolve()}")
    print(
        "[res_pose_bias_by_corners] branch z median "
        f"out={out_z_median:+.3f} mm, "
        f"return={return_z_median:+.3f} mm, "
        f"diff={branch_diff:+.3f} mm"
    )
    if len(ybin_diff_finite):
        print(
            "[res_pose_bias_by_corners] matched-bin return-out z "
            f"median={_median(ybin_diff_finite):+.3f} mm, "
            f"range={float(np.nanmin(ybin_diff_finite)):+.3f}.."
            f"{float(np.nanmax(ybin_diff_finite)):+.3f} mm"
        )
    else:
        print("[res_pose_bias_by_corners] matched-bin return-out z unavailable")
    print(
        "[res_pose_bias_by_corners] z model "
        f"R2(y-only)={meta['model_y_r2']:.3f}, "
        f"R2(y+branch)={meta['model_y_branch_r2']:.3f}, "
        f"branch_coeff={meta['model_branch_coeff_mm']:+.3f} mm"
    )

    cells = [row for row in group_rows if row.get("group_kind") == "cell"]
    cells = [
        row
        for row in cells
        if _to_int(row.get("out_frames_visible"), 0) >= 5
        and _to_int(row.get("return_frames_visible"), 0) >= 5
    ]
    top = sorted(
        cells,
        key=lambda row: abs(_to_float(row.get("return_minus_out_residual_vector_px"))),
        reverse=True,
    )[:5]
    if top:
        print("[res_pose_bias_by_corners] strongest return-out residual vector cells:")
        for row in top:
            print(
                "  "
                f"{row['group']}: "
                f"dres={_to_float(row.get('return_minus_out_residual_vector_px')):.3f}px, "
                f"ddu={_to_float(row.get('return_minus_out_mean_du_px')):+.3f}px, "
                f"ddv={_to_float(row.get('return_minus_out_mean_dv_px')):+.3f}px, "
                f"dz_visible={_to_float(row.get('return_minus_out_z_visible_median_mm')):+.3f}mm"
            )


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

    start_dir = str((initial_dir or _default_runs_dir()).resolve())
    file_name, _selected_filter = QFileDialog.getOpenFileName(
        None,
        "Select HydraTracker JSONL run",
        start_dir,
        "HydraTracker runs (*.jsonl);;All files (*)",
    )

    if owns_app:
        app.quit()

    if not file_name:
        raise RuntimeError("No JSONL run selected.")
    return Path(file_name)


def main() -> None:
    args = list(sys.argv[1:])
    path: Path | None = None
    use_latest = False
    show = False
    make_plot = True
    point_set = "pose"
    bin_width_mm = 10.0
    max_frames: int | None = None
    uncertainty_csv: str | None = "auto"

    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "--show":
            show = True
        elif arg == "--latest":
            use_latest = True
        elif arg == "--no-plot":
            make_plot = False
        elif arg == "--point-set":
            idx += 1
            if idx >= len(args):
                raise RuntimeError("--point-set needs 'pose' or 'correspondence'")
            point_set = str(args[idx]).strip().lower()
            if point_set not in ("pose", "correspondence"):
                raise RuntimeError("--point-set must be 'pose' or 'correspondence'")
        elif arg == "--bin-mm":
            idx += 1
            if idx >= len(args):
                raise RuntimeError("--bin-mm needs a numeric value")
            bin_width_mm = float(args[idx])
        elif arg == "--max-frames":
            idx += 1
            if idx >= len(args):
                raise RuntimeError("--max-frames needs an integer value")
            max_frames = int(args[idx])
        elif arg == "--uncertainty-csv":
            idx += 1
            if idx >= len(args):
                raise RuntimeError("--uncertainty-csv needs PATH, 'auto', or 'none'")
            uncertainty_csv = str(args[idx])
        elif arg.endswith(".jsonl"):
            path = Path(arg)
        else:
            raise RuntimeError(f"Unknown option: {arg}")
        idx += 1

    if path is None:
        path = _latest_run_path() if use_latest else _select_run_path_qt()
    path = path.resolve()
    run = load_run(path)

    if uncertainty_csv == "none":
        uncertainty_by_frame = {}
    else:
        uncertainty_path = (
            path.with_name(f"{path.stem}_pose_uncertainty.csv")
            if uncertainty_csv in (None, "auto")
            else Path(str(uncertainty_csv)).resolve()
        )
        uncertainty_by_frame = _read_csv_by_frame(uncertainty_path)

    frame_rows, corner_rows, ybin_rows, group_rows, meta = analyze_run(
        run,
        point_set=point_set,
        bin_width_mm=bin_width_mm,
        uncertainty_by_frame=uncertainty_by_frame,
        max_frames=max_frames,
    )

    frame_csv = path.with_name(f"{path.stem}_pose_bias_by_corners_frames.csv")
    corner_csv = path.with_name(f"{path.stem}_pose_bias_by_corners_corners.csv")
    ybin_csv = path.with_name(f"{path.stem}_pose_bias_by_corners_ybins.csv")
    group_csv = path.with_name(f"{path.stem}_pose_bias_by_corners_groups.csv")
    _write_csv(frame_csv, frame_rows)
    _write_csv(corner_csv, corner_rows)
    _write_csv(ybin_csv, ybin_rows)
    _write_csv(group_csv, group_rows)

    paths = {"frame": frame_csv, "corner": corner_csv, "ybin": ybin_csv, "group": group_csv}
    if make_plot:
        paths["plot"] = plot_bias(run, frame_rows, ybin_rows, group_rows, meta, show=show)

    _print_summary(frame_rows, ybin_rows, group_rows, meta, paths)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[res_pose_bias_by_corners] ERROR: {exc}")
        sys.exit(1)
