"""Replay HydraMarker logs against an ideal cylindrical geometry model.

This analysis tool replaces or compares observed geometry with an idealized
cylinder to separate model-shape error from detector and pose-estimation error.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


COMPONENTS = (("x", "X", "#d62728"), ("y", "Y", "#2ca02c"), ("z", "Z", "#1f77b4"))


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


def _rvec_to_euler_deg(rvec: np.ndarray) -> tuple[float, float, float]:
    try:
        import cv2

        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    except Exception:
        return math.nan, math.nan, math.nan

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(float(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        pitch = math.atan2(float(-R[2, 0]), sy)
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _camera_from_run_start(record: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
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
        if all(np.isfinite([fx, fy, cx, cy])):
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


@dataclass(frozen=True)
class Observation:
    frame: int
    object_points_sfm: np.ndarray
    image_points: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    original_rvec: np.ndarray
    original_tvec: np.ndarray


@dataclass(frozen=True)
class CylinderModel:
    points_by_row_col: dict[tuple[int, int], np.ndarray]
    radius_mm: float
    center_x_mm: float
    center_z_mm: float
    theta0_rad: float
    theta_step_rad: float
    spacing_mm: float
    source_marker_json: Path
    current_edge_mean_mm: float
    current_edge_std_mm: float
    current_edge_p95_abs_dev_mm: float


def load_run(path: Path) -> dict[str, Any]:
    K: np.ndarray | None = None
    dist: np.ndarray | None = None
    marker_json_path: Path | None = None
    run_id = path.stem
    timestamp = ""
    frames: dict[int, dict[str, Any]] = {}
    details: dict[int, dict[str, Any]] = {}

    with Path(path).open("r", encoding="utf-8") as handle:
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
                marker_path_value = record.get("marker_json_path")
                if marker_path_value:
                    marker_json_path = Path(str(marker_path_value))
                K, dist = _camera_from_run_start(record)
            elif record_type == "frame":
                data = dict(record.get("data") or {})
                frame = _to_int(data.get("frame"), default=len(frames))
                frames[frame] = data
            elif record_type == "frame_detail":
                frame = _to_int(record.get("frame"), default=-1)
                if frame >= 0:
                    details[frame] = dict(record)

    if K is None or dist is None:
        raise RuntimeError("No camera intrinsics found in run_start record.")
    if marker_json_path is None:
        raise RuntimeError("No marker_json_path found in run_start record.")
    if not frames:
        raise RuntimeError(f"No frame records found in:\n{path}")
    return {
        "path": Path(path),
        "run_id": run_id,
        "timestamp": timestamp,
        "K": K,
        "dist": dist,
        "marker_json_path": marker_json_path,
        "frames": frames,
        "details": details,
    }


def _corner_arrays(corners: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    object_points: list[list[float]] = []
    image_points: list[list[float]] = []
    rows: list[int] = []
    cols: list[int] = []

    for corner in corners:
        xyz = corner.get("xyz_mm")
        uv = corner.get("uv_px")
        if not isinstance(xyz, (list, tuple)) or not isinstance(uv, (list, tuple)):
            continue
        if len(xyz) < 3 or len(uv) < 2:
            continue
        xyz_arr = np.asarray([_to_float(v) for v in xyz[:3]], dtype=np.float64)
        uv_arr = np.asarray([_to_float(v) for v in uv[:2]], dtype=np.float64)
        if not np.all(np.isfinite(xyz_arr)) or not np.all(np.isfinite(uv_arr)):
            continue
        row = _to_int(corner.get("global_row"), default=_to_int(corner.get("row"), -1))
        col = _to_int(corner.get("global_col"), default=_to_int(corner.get("col"), -1))
        if row < 0 or col < 0:
            continue
        object_points.append(xyz_arr.tolist())
        image_points.append(uv_arr.tolist())
        rows.append(row)
        cols.append(col)

    return (
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
        np.asarray(rows, dtype=np.int64).reshape(-1),
        np.asarray(cols, dtype=np.int64).reshape(-1),
    )


def build_observations(
    run: dict[str, Any],
    *,
    point_set: str,
    max_frames: int | None,
) -> list[Observation]:
    frames: dict[int, dict[str, Any]] = run["frames"]
    details: dict[int, dict[str, Any]] = run["details"]
    sorted_frames = sorted(frames)
    if max_frames is not None and max_frames > 0:
        sorted_frames = sorted_frames[: int(max_frames)]

    corners_key = "pose_corners" if point_set == "pose" else "correspondence_corners"
    observations: list[Observation] = []
    for frame in sorted_frames:
        data = frames[frame]
        if _to_int(data.get("success"), default=0) == 0:
            continue
        detail = details.get(frame, {})
        corners = list(detail.get(corners_key) or [])
        if not corners and point_set == "pose":
            corners = list(detail.get("correspondence_corners") or [])
        object_points, image_points, rows, cols = _corner_arrays(corners)
        rvec = np.asarray(
            [
                _to_float(data.get("rvec_x_rad")),
                _to_float(data.get("rvec_y_rad")),
                _to_float(data.get("rvec_z_rad")),
            ],
            dtype=np.float64,
        )
        tvec = np.asarray(
            [
                _to_float(data.get("tvec_x_mm")),
                _to_float(data.get("tvec_y_mm")),
                _to_float(data.get("tvec_z_mm")),
            ],
            dtype=np.float64,
        )
        if len(object_points) < 6 or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue
        observations.append(
            Observation(
                frame=int(frame),
                object_points_sfm=object_points,
                image_points=image_points,
                rows=rows,
                cols=cols,
                original_rvec=rvec,
                original_tvec=tvec,
            )
        )

    if not observations:
        raise RuntimeError("No usable observations found.")
    return observations


def _fit_circle_xz(xz: np.ndarray) -> tuple[float, float, float]:
    xz = np.asarray(xz, dtype=np.float64).reshape(-1, 2)
    x = xz[:, 0]
    z = xz[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * z, np.ones_like(x)])
    b = x * x + z * z
    cx, cz, c = np.linalg.lstsq(A, b, rcond=None)[0]
    radius_sq = float(c + cx * cx + cz * cz)
    if radius_sq <= 0.0:
        raise RuntimeError("Could not fit a valid cylinder circle.")
    return float(cx), float(cz), math.sqrt(radius_sq)


def _load_sfm_points(marker_json_path: Path) -> tuple[dict[tuple[int, int], np.ndarray], float]:
    with Path(marker_json_path).open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    spacing = _to_float(meta.get("square_size_mm"))
    if not np.isfinite(spacing) or spacing <= 0.0:
        spacing = 10.0 * _to_float(meta.get("square_size_cm"))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise RuntimeError("Marker JSON has no valid square_size_mm/square_size_cm.")

    points: dict[tuple[int, int], np.ndarray] = {}
    for corner in list(meta.get("corners") or []):
        if not isinstance(corner, dict):
            continue
        xyz = corner.get("xyz_mm")
        if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
            continue
        point = np.asarray([_to_float(v) for v in xyz[:3]], dtype=np.float64)
        if not np.all(np.isfinite(point)):
            continue
        row = _to_int(corner.get("row"), default=-1)
        col = _to_int(corner.get("col"), default=-1)
        if row < 0 or col < 0:
            continue
        points[(int(row), int(col))] = point.reshape(3)

    if len(points) < 6:
        raise RuntimeError("Marker JSON has too few SfM corners.")
    return points, float(spacing)


def _horizontal_edge_stats(points: dict[tuple[int, int], np.ndarray], spacing: float) -> tuple[float, float, float]:
    distances: list[float] = []
    for (row, col), point in points.items():
        other = points.get((row, col + 1))
        if other is None:
            continue
        distances.append(float(np.linalg.norm(other - point)))
    arr = _finite(distances)
    if len(arr) == 0:
        return math.nan, math.nan, math.nan
    return (
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.percentile(np.abs(arr - float(spacing)), 95)),
    )


def build_ideal_cylinder_model(
    marker_json_path: Path,
    *,
    radius_mm: float | None = None,
) -> CylinderModel:
    sfm_points, spacing = _load_sfm_points(marker_json_path)

    cols = sorted({col for _row, col in sfm_points})
    col_xz: list[list[float]] = []
    for col in cols:
        pts = np.asarray(
            [point for (row_i, col_i), point in sfm_points.items() if col_i == col],
            dtype=np.float64,
        )
        col_xz.append([float(np.median(pts[:, 0])), float(np.median(pts[:, 2]))])
    col_xz_arr = np.asarray(col_xz, dtype=np.float64).reshape(-1, 2)
    cx, cz, fitted_radius = _fit_circle_xz(col_xz_arr)
    radius = float(radius_mm) if radius_mm is not None and radius_mm > 0.0 else float(fitted_radius)

    theta_obs = np.unwrap(np.arctan2(col_xz_arr[:, 1] - cz, col_xz_arr[:, 0] - cx))
    col_arr = np.asarray(cols, dtype=np.float64)
    if len(theta_obs) >= 2:
        direction = float(np.sign(np.median(np.diff(theta_obs)))) or 1.0
    else:
        direction = 1.0
    theta_step = direction * float(spacing) / max(radius, 1e-9)
    theta0 = float(np.median(theta_obs - col_arr * theta_step))

    origin_key = (0, 0)
    origin = sfm_points.get(origin_key)
    if origin is not None:
        origin_theta = theta0
        origin_xz = np.asarray(
            [cx + radius * math.cos(origin_theta), cz + radius * math.sin(origin_theta)],
            dtype=np.float64,
        )
        xz_shift = np.asarray([origin[0], origin[2]], dtype=np.float64) - origin_xz
        y0 = float(origin[1])
    else:
        xz_shift = np.zeros(2, dtype=np.float64)
        y0 = float(np.median([point[1] - row * spacing for (row, _col), point in sfm_points.items()]))

    ideal: dict[tuple[int, int], np.ndarray] = {}
    for row, col in sorted(sfm_points):
        theta = theta0 + float(col) * theta_step
        x = cx + radius * math.cos(theta) + float(xz_shift[0])
        y = y0 + float(row) * spacing
        z = cz + radius * math.sin(theta) + float(xz_shift[1])
        ideal[(int(row), int(col))] = np.asarray([x, y, z], dtype=np.float64)

    edge_mean, edge_std, edge_p95 = _horizontal_edge_stats(sfm_points, spacing)
    return CylinderModel(
        points_by_row_col=ideal,
        radius_mm=float(radius),
        center_x_mm=float(cx + xz_shift[0]),
        center_z_mm=float(cz + xz_shift[1]),
        theta0_rad=float(theta0),
        theta_step_rad=float(theta_step),
        spacing_mm=float(spacing),
        source_marker_json=Path(marker_json_path),
        current_edge_mean_mm=edge_mean,
        current_edge_std_mm=edge_std,
        current_edge_p95_abs_dev_mm=edge_p95,
    )


def object_points_for_observation(obs: Observation, model: CylinderModel) -> np.ndarray | None:
    points: list[np.ndarray] = []
    for row, col in zip(obs.rows, obs.cols, strict=False):
        point = model.points_by_row_col.get((int(row), int(col)))
        if point is None:
            return None
        points.append(point)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _project_errors(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    import cv2

    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        K.reshape(3, 3),
        dist.reshape(-1, 1),
    )
    residual = projected.reshape(-1, 2) - np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    return np.sqrt(np.sum(residual * residual, axis=1))


def solve_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    seed_rvec: np.ndarray,
    seed_tvec: np.ndarray,
    *,
    refine: str,
) -> tuple[bool, np.ndarray, np.ndarray, dict[str, float]]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV/cv2 is required for ideal-cylinder replay.") from exc

    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    rvec = np.asarray(seed_rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(seed_tvec, dtype=np.float64).reshape(3, 1)

    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K.reshape(3, 3),
            dist.reshape(-1, 1),
            rvec=rvec.copy(),
            tvec=tvec.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except Exception:
        ok = False

    if not ok:
        try:
            flags = cv2.SOLVEPNP_SQPNP if hasattr(cv2, "SOLVEPNP_SQPNP") else cv2.SOLVEPNP_EPNP
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                K.reshape(3, 3),
                dist.reshape(-1, 1),
                flags=flags,
            )
        except Exception:
            return False, np.asarray(seed_rvec, dtype=np.float64).reshape(3), np.asarray(seed_tvec, dtype=np.float64).reshape(3), {}

    if not ok:
        return False, np.asarray(seed_rvec, dtype=np.float64).reshape(3), np.asarray(seed_tvec, dtype=np.float64).reshape(3), {}

    refine = str(refine).strip().lower()
    try:
        if refine == "vvs" and hasattr(cv2, "solvePnPRefineVVS"):
            rvec, tvec = cv2.solvePnPRefineVVS(
                object_points,
                image_points,
                K.reshape(3, 3),
                dist.reshape(-1, 1),
                rvec,
                tvec,
            )
        elif refine == "lm" and hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                K.reshape(3, 3),
                dist.reshape(-1, 1),
                rvec,
                tvec,
            )
    except Exception:
        pass

    rvec_out = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec_out = np.asarray(tvec, dtype=np.float64).reshape(3)
    errors = _project_errors(object_points, image_points, rvec_out, tvec_out, K, dist)
    stats = {
        "reproj_mean_px": _mean(errors),
        "reproj_median_px": _median(errors),
        "reproj_rms_px": _rms(errors),
        "reproj_p95_px": _percentile(errors, 95),
        "reproj_max_px": float(np.max(_finite(errors))) if len(_finite(errors)) else math.nan,
    }
    return True, rvec_out, tvec_out, stats


def replay_methods(
    observations: list[Observation],
    run: dict[str, Any],
    cylinder: CylinderModel,
    *,
    refine: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = ("logged_original", "current_sfm_refit", "ideal_cylinder")
    method_tvecs: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    method_rvecs: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    method_solved: dict[str, list[bool]] = {method: [] for method in methods}
    method_stats: dict[str, list[dict[str, float]]] = {method: [] for method in methods}

    for obs in observations:
        ideal_points = object_points_for_observation(obs, cylinder)
        method_inputs = {
            "logged_original": None,
            "current_sfm_refit": obs.object_points_sfm,
            "ideal_cylinder": ideal_points,
        }
        for method in methods:
            object_points = method_inputs[method]
            if method == "logged_original":
                rvec = obs.original_rvec.copy()
                tvec = obs.original_tvec.copy()
                solved = True
                try:
                    errors = _project_errors(
                        obs.object_points_sfm,
                        obs.image_points,
                        rvec,
                        tvec,
                        run["K"],
                        run["dist"],
                    )
                    stats = {
                        "reproj_mean_px": _mean(errors),
                        "reproj_median_px": _median(errors),
                        "reproj_rms_px": _rms(errors),
                        "reproj_p95_px": _percentile(errors, 95),
                        "reproj_max_px": float(np.max(_finite(errors))) if len(_finite(errors)) else math.nan,
                    }
                except Exception:
                    stats = {}
            elif object_points is None:
                rvec = obs.original_rvec.copy()
                tvec = obs.original_tvec.copy()
                solved = False
                stats = {}
            else:
                solved, rvec, tvec, stats = solve_pose(
                    object_points,
                    obs.image_points,
                    run["K"],
                    run["dist"],
                    obs.original_rvec,
                    obs.original_tvec,
                    refine=refine,
                )
                if not solved:
                    rvec = obs.original_rvec.copy()
                    tvec = obs.original_tvec.copy()
            method_rvecs[method].append(np.asarray(rvec, dtype=np.float64).reshape(3))
            method_tvecs[method].append(np.asarray(tvec, dtype=np.float64).reshape(3))
            method_solved[method].append(bool(solved))
            method_stats[method].append(stats)

    rel_tvecs: dict[str, np.ndarray] = {}
    for method in methods:
        arr = np.asarray(method_tvecs[method], dtype=np.float64).reshape(-1, 3)
        finite = np.all(np.isfinite(arr), axis=1)
        if np.any(finite):
            origin = arr[int(np.where(finite)[0][0])].copy()
            rel_tvecs[method] = arr - origin
        else:
            rel_tvecs[method] = np.full_like(arr, np.nan)

    for obs_idx, obs in enumerate(observations):
        for method in methods:
            tvec = np.asarray(method_tvecs[method][obs_idx], dtype=np.float64).reshape(3)
            rvec = np.asarray(method_rvecs[method][obs_idx], dtype=np.float64).reshape(3)
            rel = rel_tvecs[method][obs_idx]
            stats = method_stats[method][obs_idx]
            roll_deg, pitch_deg, yaw_deg = _rvec_to_euler_deg(rvec)
            row = {
                "method": method,
                "frame": int(obs.frame),
                "solved": int(method_solved[method][obs_idx]),
                "point_count": int(len(obs.image_points)),
                "row_min": int(np.min(obs.rows)) if len(obs.rows) else -1,
                "row_max": int(np.max(obs.rows)) if len(obs.rows) else -1,
                "col_min": int(np.min(obs.cols)) if len(obs.cols) else -1,
                "col_max": int(np.max(obs.cols)) if len(obs.cols) else -1,
                "distinct_rows": int(len(set(int(v) for v in obs.rows.tolist()))),
                "distinct_cols": int(len(set(int(v) for v in obs.cols.tolist()))),
                "tvec_x_mm": float(tvec[0]),
                "tvec_y_mm": float(tvec[1]),
                "tvec_z_mm": float(tvec[2]),
                "rvec_x_rad": float(rvec[0]),
                "rvec_y_rad": float(rvec[1]),
                "rvec_z_rad": float(rvec[2]),
                "roll_deg": float(roll_deg),
                "pitch_deg": float(pitch_deg),
                "yaw_deg": float(yaw_deg),
                "rel_x_mm": float(rel[0]),
                "rel_y_mm": float(rel[1]),
                "rel_z_mm": float(rel[2]),
                "delta_from_logged_x_mm": float(tvec[0] - obs.original_tvec[0]),
                "delta_from_logged_y_mm": float(tvec[1] - obs.original_tvec[1]),
                "delta_from_logged_z_mm": float(tvec[2] - obs.original_tvec[2]),
                "translation_delta_from_logged_mm": float(np.linalg.norm(tvec - obs.original_tvec)),
            }
            row.update(stats)
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], cylinder: CylinderModel) -> list[dict[str, Any]]:
    methods: list[str] = []
    for row in rows:
        method = str(row.get("method") or "")
        if method and method not in methods:
            methods.append(method)

    summary: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in rows if str(row.get("method")) == method]
        rel = {
            key: np.asarray([_to_float(row.get(key)) for row in method_rows], dtype=np.float64)
            for key in ("rel_x_mm", "rel_y_mm", "rel_z_mm")
        }
        summary.append(
            {
                "method": method,
                "frames": len(method_rows),
                "solve_failures": int(sum(1 for row in method_rows if _to_int(row.get("solved"), 0) == 0)),
                "x_range_mm": float(np.nanmax(rel["rel_x_mm"]) - np.nanmin(rel["rel_x_mm"])),
                "y_range_mm": float(np.nanmax(rel["rel_y_mm"]) - np.nanmin(rel["rel_y_mm"])),
                "z_range_mm": float(np.nanmax(rel["rel_z_mm"]) - np.nanmin(rel["rel_z_mm"])),
                "x_closure_mm": float(rel["rel_x_mm"][-1] - rel["rel_x_mm"][0]),
                "y_closure_mm": float(rel["rel_y_mm"][-1] - rel["rel_y_mm"][0]),
                "z_closure_mm": float(rel["rel_z_mm"][-1] - rel["rel_z_mm"][0]),
                "reproj_rms_median_px": _median([_to_float(row.get("reproj_rms_px")) for row in method_rows]),
                "reproj_p95_median_px": _median([_to_float(row.get("reproj_p95_px")) for row in method_rows]),
                "reproj_max_median_px": _median([_to_float(row.get("reproj_max_px")) for row in method_rows]),
                "translation_delta_from_logged_median_mm": _median(
                    [_to_float(row.get("translation_delta_from_logged_mm")) for row in method_rows]
                ),
                "translation_delta_from_logged_p95_mm": _percentile(
                    [_to_float(row.get("translation_delta_from_logged_mm")) for row in method_rows],
                    95,
                ),
                "cylinder_radius_mm": cylinder.radius_mm,
                "cylinder_theta_step_deg": math.degrees(cylinder.theta_step_rad),
                "cylinder_spacing_mm": cylinder.spacing_mm,
                "current_horizontal_edge_mean_mm": cylinder.current_edge_mean_mm,
                "current_horizontal_edge_std_mm": cylinder.current_edge_std_mm,
                "current_horizontal_edge_p95_abs_dev_mm": cylinder.current_edge_p95_abs_dev_mm,
            }
        )
    return summary


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
            "grid.color": "#d9dee8",
            "grid.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d0d4dc",
        }
    )


def plot_translation(
    run: dict[str, Any],
    rows: list[dict[str, Any]],
    cylinder: CylinderModel,
    *,
    show: bool,
) -> Path:
    import matplotlib

    matplotlib.use("Agg" if not show else "QtAgg")
    import matplotlib.pyplot as plt

    _setup_plot_style(plt)

    path: Path = run["path"]
    out_path = path.with_name(f"{path.stem}_ideal_cylinder_replay_plot.png")
    methods = ["logged_original", "current_sfm_refit", "ideal_cylinder"]
    colors = {
        "logged_original": "#d62728",
        "current_sfm_refit": "#4c78a8",
        "ideal_cylinder": "#9467bd",
    }
    labels = {
        "logged_original": "logged original",
        "current_sfm_refit": "current SfM refit",
        "ideal_cylinder": "ideal cylinder",
    }
    rows_by_method = {method: [row for row in rows if str(row.get("method")) == method] for method in methods}

    fig, axes = plt.subplots(5, 1, figsize=(15.5, 12.0), sharex=True, constrained_layout=False)
    fig.subplots_adjust(top=0.86, hspace=0.34)

    run_label = run["run_id"]
    if run.get("timestamp"):
        run_label += f"  |  {run['timestamp']}"
    fig.suptitle(
        "HydraTracker relative translation components (ideal cylinder offline replay, camera frame)\n"
        f"{run_label}",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    for comp_idx, (key_suffix, label, _base_color) in enumerate(COMPONENTS):
        key = f"rel_{key_suffix}_mm"
        ax = axes[comp_idx]
        title_parts: list[str] = []
        for method in methods:
            method_rows = rows_by_method[method]
            frames = np.asarray([_to_float(row.get("frame")) for row in method_rows], dtype=np.float64)
            values = np.asarray([_to_float(row.get(key)) for row in method_rows], dtype=np.float64)
            ax.plot(
                frames,
                values,
                color=colors[method],
                linewidth=1.8 if method in ("logged_original", "ideal_cylinder") else 1.25,
                marker="o",
                markersize=2.5,
                markerfacecolor="white",
                markeredgewidth=0.7,
                label=labels[method],
            )
            finite = values[np.isfinite(values)]
            if len(finite):
                title_parts.append(f"{labels[method]}={float(np.max(finite) - np.min(finite)):.2f}")
        ax.axhline(0.0, color="#888888", alpha=0.28, linewidth=1.0, linestyle="--")
        ax.set_title(f"{label} relative component   range [mm]: {', '.join(title_parts)}", loc="left")
        ax.set_ylabel(f"delta T_C_T {key_suffix} [mm]")
        ax.grid(True, axis="both")
        ax.legend(loc="upper right", fontsize=8)

    point_ax = axes[3]
    for method in methods:
        method_rows = rows_by_method[method]
        frames = np.asarray([_to_float(row.get("frame")) for row in method_rows], dtype=np.float64)
        points = np.asarray([_to_float(row.get("point_count")) for row in method_rows], dtype=np.float64)
        point_ax.plot(
            frames,
            points,
            color=colors[method],
            linewidth=1.4,
            marker="o",
            markersize=2.2,
            markerfacecolor="white",
            markeredgewidth=0.6,
            label=f"{labels[method]} points",
        )
    original_rows = rows_by_method["logged_original"]
    frames = np.asarray([_to_float(row.get("frame")) for row in original_rows], dtype=np.float64)
    row_min = np.asarray([_to_float(row.get("row_min")) for row in original_rows], dtype=np.float64)
    row_max = np.asarray([_to_float(row.get("row_max")) for row in original_rows], dtype=np.float64)
    point_ax.set_ylabel("points")
    point_ax.set_title("Pose diagnostics   point count and global row range", loc="left")
    point_ax.grid(True, axis="both")
    row_ax = point_ax.twinx()
    row_ax.plot(frames, row_min, color="#f58518", linewidth=1.1, label="row min")
    row_ax.plot(frames, row_max, color="#b279a2", linewidth=1.1, label="row max")
    row_ax.set_ylabel("global row")
    point_lines, point_labels = point_ax.get_legend_handles_labels()
    row_lines, row_labels = row_ax.get_legend_handles_labels()
    point_ax.legend(point_lines + row_lines, point_labels + row_labels, loc="upper right", fontsize=8)

    orient_ax = axes[4]
    orientation_specs = (
        ("roll_deg", "roll", "#4c78a8"),
        ("pitch_deg", "pitch", "#54a24b"),
        ("yaw_deg", "yaw", "#e45756"),
    )
    linestyles = {"logged_original": "-", "current_sfm_refit": ":", "ideal_cylinder": "--"}
    for method in methods:
        method_rows = rows_by_method[method]
        frames = np.asarray([_to_float(row.get("frame")) for row in method_rows], dtype=np.float64)
        for key, name, color in orientation_specs:
            values = np.asarray([_to_float(row.get(key)) for row in method_rows], dtype=np.float64)
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            rel = values - values[np.where(finite)[0][0]]
            rel[~finite] = np.nan
            orient_ax.plot(
                frames,
                rel,
                color=color,
                linestyle=linestyles[method],
                linewidth=1.45,
                label=f"{labels[method]} {name}",
            )
    orient_ax.axhline(0.0, color="#888888", alpha=0.3, linewidth=1.0, linestyle="--")
    orient_ax.set_ylabel("rotation delta [deg]")
    orient_ax.set_title("Orientation diagnostics   camera-frame Euler deltas", loc="left")
    orient_ax.grid(True, axis="both")
    orient_ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frame")

    first = original_rows[0]
    info = (
        f"relative to frame {int(_to_float(first.get('frame')))} "
        f"({_to_float(first.get('tvec_x_mm')):.2f}, {_to_float(first.get('tvec_y_mm')):.2f}, "
        f"{_to_float(first.get('tvec_z_mm')):.2f}) mm   "
        f"frames={len(original_rows)}   radius={cylinder.radius_mm:.3f} mm   "
        f"theta_step={math.degrees(cylinder.theta_step_rad):.2f} deg"
    )
    fig.text(0.01, 0.925, info, ha="left", va="top", fontsize=9.5, color="#333333")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
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


def _parse_args(argv: list[str]) -> dict[str, Any]:
    args: dict[str, Any] = {
        "path": None,
        "latest": False,
        "point_set": "correspondence",
        "max_frames": None,
        "refine": "vvs",
        "radius_mm": None,
        "make_plot": True,
        "show": False,
    }
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--latest":
            args["latest"] = True
        elif arg == "--point-set":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--point-set needs pose or correspondence")
            point_set = str(argv[idx]).strip().lower()
            if point_set not in ("pose", "correspondence"):
                raise RuntimeError("--point-set must be pose or correspondence")
            args["point_set"] = point_set
        elif arg == "--max-frames":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--max-frames needs an integer")
            args["max_frames"] = int(argv[idx])
        elif arg == "--refine":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--refine needs none, lm, or vvs")
            refine = str(argv[idx]).strip().lower()
            if refine not in ("none", "lm", "vvs"):
                raise RuntimeError("--refine must be none, lm, or vvs")
            args["refine"] = refine
        elif arg == "--radius-mm":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--radius-mm needs a numeric value")
            args["radius_mm"] = float(argv[idx])
        elif arg == "--no-plot":
            args["make_plot"] = False
        elif arg == "--show":
            args["show"] = True
        elif arg.endswith(".jsonl"):
            args["path"] = Path(arg)
        else:
            raise RuntimeError(f"Unknown option: {arg}")
        idx += 1
    return args


def _print_summary(paths: dict[str, Path], summary_rows: list[dict[str, Any]], cylinder: CylinderModel) -> None:
    print(f"[ideal_cylinder_replay] radius={cylinder.radius_mm:.3f} mm")
    print(f"[ideal_cylinder_replay] theta_step={math.degrees(cylinder.theta_step_rad):.3f} deg")
    print(
        "[ideal_cylinder_replay] current horizontal edge "
        f"std={cylinder.current_edge_std_mm:.3f} mm "
        f"p95_abs_dev={cylinder.current_edge_p95_abs_dev_mm:.3f} mm"
    )
    print(f"[ideal_cylinder_replay] saved frame csv   -> {paths['frame'].resolve()}")
    print(f"[ideal_cylinder_replay] saved summary csv -> {paths['summary'].resolve()}")
    if "plot" in paths:
        print(f"[ideal_cylinder_replay] saved plot        -> {paths['plot'].resolve()}")
    print("[ideal_cylinder_replay] method summary:")
    for row in summary_rows:
        print(
            "  "
            f"{row['method']}: "
            f"z_range={_to_float(row.get('z_range_mm')):.3f} mm, "
            f"z_closure={_to_float(row.get('z_closure_mm')):+.3f} mm, "
            f"rms={_to_float(row.get('reproj_rms_median_px')):.3f} px, "
            f"p95_shift={_to_float(row.get('translation_delta_from_logged_p95_mm')):.3f} mm"
        )


def main() -> None:
    args = _parse_args(sys.argv[1:])
    path = args["path"]
    if path is None:
        path = _latest_run_path() if args["latest"] else None
    if path is None:
        raise RuntimeError("Pass a HydraTracker JSONL path or --latest.")
    path = Path(path).resolve()
    run = load_run(path)
    cylinder = build_ideal_cylinder_model(
        run["marker_json_path"],
        radius_mm=args["radius_mm"],
    )
    observations = build_observations(
        run,
        point_set=str(args["point_set"]),
        max_frames=args["max_frames"],
    )
    frame_rows = replay_methods(
        observations,
        run,
        cylinder,
        refine=str(args["refine"]),
    )
    summary_rows = summarize(frame_rows, cylinder)

    frame_csv = path.with_name(f"{path.stem}_ideal_cylinder_replay_frames.csv")
    summary_csv = path.with_name(f"{path.stem}_ideal_cylinder_replay_summary.csv")
    _write_csv(frame_csv, frame_rows)
    _write_csv(summary_csv, summary_rows)
    paths = {"frame": frame_csv, "summary": summary_csv}
    if bool(args["make_plot"]):
        paths["plot"] = plot_translation(run, frame_rows, cylinder, show=bool(args["show"]))
    _print_summary(paths, summary_rows, cylinder)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[ideal_cylinder_replay] ERROR: {exc}")
        sys.exit(1)
