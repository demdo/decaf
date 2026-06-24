from __future__ import annotations

import csv
import json
import math
import sys
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


def _load_marker_geometry_prior(
    marker_json_path: Path | None,
    *,
    min_weight: float = 0.05,
    deviation_scale_mm: float = 0.35,
) -> dict[tuple[int, int], float]:
    """Build a global per-corner structural prior from the SfM marker model.

    GNC-Pose uses 3D structural consistency as a prior before GNC inlier
    selection. For sparse marker corners, voxel density is mostly uninformative,
    so we use the known row/col topology and penalize corners whose SfM
    neighbor edges deviate from the declared marker spacing.
    """
    if marker_json_path is None:
        return {}

    path = Path(marker_json_path)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except Exception:
        return {}

    expected = _to_float(meta.get("square_size_mm"))
    if not np.isfinite(expected) or expected <= 0.0:
        expected = 10.0 * _to_float(meta.get("square_size_cm"))
    if not np.isfinite(expected) or expected <= 0.0:
        return {}

    row_col_to_point: dict[tuple[int, int], np.ndarray] = {}
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
        row_col_to_point[(int(row), int(col))] = point.reshape(3)

    if len(row_col_to_point) < 4:
        return {}

    dev_sum: dict[tuple[int, int], float] = {key: 0.0 for key in row_col_to_point}
    support: dict[tuple[int, int], int] = {key: 0 for key in row_col_to_point}

    for key, point in row_col_to_point.items():
        row, col = key
        for neighbor_key in ((row, col + 1), (row + 1, col)):
            neighbor = row_col_to_point.get(neighbor_key)
            if neighbor is None:
                continue
            distance = float(np.linalg.norm(neighbor - point))
            if not np.isfinite(distance) or distance <= 0.0:
                continue
            deviation = abs(distance - expected)
            dev_sum[key] += deviation
            dev_sum[neighbor_key] += deviation
            support[key] += 1
            support[neighbor_key] += 1

    scale = max(float(deviation_scale_mm), 1e-6)
    weights: dict[tuple[int, int], float] = {}
    for key in row_col_to_point:
        if support[key] <= 0:
            weights[key] = float(min_weight)
            continue
        mean_dev = dev_sum[key] / float(support[key])
        quality = math.exp(-((mean_dev / scale) ** 2))
        weight = float(min_weight) + (1.0 - float(min_weight)) * quality
        weights[key] = float(np.clip(weight, min_weight, 1.0))

    return weights


def load_run(path: Path) -> dict[str, Any]:
    K: np.ndarray | None = None
    dist: np.ndarray | None = None
    marker_json_path: Path | None = None
    run_id = path.stem
    timestamp = ""
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
            elif record_type == "run_summary":
                summary = dict(record.get("summary") or {})

    if K is None or dist is None:
        raise RuntimeError("No camera intrinsics found in run_start record.")
    if not frames:
        raise RuntimeError(f"No frame records found in:\n{path}")

    return {
        "path": path,
        "run_id": run_id,
        "timestamp": timestamp,
        "K": K,
        "dist": dist,
        "marker_json_path": marker_json_path,
        "geometry_prior": _load_marker_geometry_prior(marker_json_path),
        "frames": frames,
        "details": details,
        "summary": summary,
    }


@dataclass(frozen=True)
class Observation:
    frame: int
    branch: str
    object_points: np.ndarray
    image_points: np.ndarray
    original_rvec: np.ndarray
    original_tvec: np.ndarray
    original_rel_tvec: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    point_count: int


@dataclass(frozen=True)
class ReplayMethod:
    name: str
    loss: str
    threshold_px: float


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
        object_points.append(xyz_arr.tolist())
        image_points.append(uv_arr.tolist())
        rows.append(_to_int(corner.get("global_row"), default=_to_int(corner.get("local_row"), -999)))
        cols.append(_to_int(corner.get("global_col"), default=_to_int(corner.get("local_col"), -999)))

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
) -> tuple[list[Observation], dict[str, Any]]:
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
    origin_tvec = tvec_arr[origin_index].copy()
    rel_tvec = tvec_arr - origin_tvec
    ranges = np.nanmax(rel_tvec[valid_tvec], axis=0) - np.nanmin(rel_tvec[valid_tvec], axis=0)
    movement_axis_idx = int(np.nanargmax(ranges))
    movement_axis = COMPONENTS[movement_axis_idx]
    movement_values = rel_tvec[:, movement_axis_idx]
    turn_idx = int(np.nanargmax(np.abs(movement_values)))
    turn_frame = sorted_frames[turn_idx]

    corners_key = "pose_corners" if point_set == "pose" else "correspondence_corners"
    observations: list[Observation] = []
    for idx, frame in enumerate(sorted_frames):
        data = frames[frame]
        detail = details.get(frame, {})
        success = _to_int(data.get("success"), default=0)
        if success == 0:
            continue

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
        tvec = tvec_arr[idx].copy()
        if len(object_points) < 6 or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue

        observations.append(
            Observation(
                frame=int(frame),
                branch="out" if frame <= turn_frame else "return",
                object_points=object_points,
                image_points=image_points,
                original_rvec=rvec,
                original_tvec=tvec,
                original_rel_tvec=rel_tvec[idx].copy(),
                rows=rows,
                cols=cols,
                point_count=int(len(object_points)),
            )
        )

    if not observations:
        raise RuntimeError("No frames with usable pose corners found.")

    meta = {
        "origin_frame": int(sorted_frames[origin_index]),
        "origin_tvec": origin_tvec,
        "movement_axis": movement_axis,
        "movement_axis_idx": movement_axis_idx,
        "turn_frame": int(turn_frame),
        "x_range_mm": float(ranges[0]),
        "y_range_mm": float(ranges[1]),
        "z_range_mm": float(ranges[2]),
    }
    return observations, meta


def _project_with_jacobian(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    projected, jacobian = cv2.projectPoints(
        object_points.reshape(-1, 3),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        K.reshape(3, 3),
        dist.reshape(-1, 1),
    )
    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    residual = projected - image_points.reshape(-1, 2)
    jacobian = np.asarray(jacobian, dtype=np.float64)[:, :6]
    return projected, residual, jacobian


def _weights_from_errors(errors: np.ndarray, loss: str, threshold_px: float) -> np.ndarray:
    loss = str(loss).strip().lower()
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    c = max(float(threshold_px), 1e-9)
    e = np.maximum(errors, 1e-12)
    if loss in ("none", "linear", "opencv"):
        return np.ones_like(e)
    if loss == "huber":
        return np.minimum(1.0, c / e)
    if loss == "cauchy":
        return 1.0 / (1.0 + (e / c) ** 2)
    if loss == "tukey":
        u = e / c
        weights = (1.0 - u * u) ** 2
        weights[u >= 1.0] = 0.0
        return weights
    raise RuntimeError(f"Unknown robust loss: {loss}")


def _geometry_aware_weights(
    obs: Observation,
    *,
    model_prior: dict[tuple[int, int], float] | None = None,
    min_weight: float = 0.05,
) -> np.ndarray:
    """Topology-support prior inspired by GNC-Pose's geometry-aware weighting."""
    n = int(obs.point_count)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)

    if model_prior:
        weights = np.asarray(
            [
                float(model_prior.get((int(row), int(col)), min_weight))
                for row, col in zip(obs.rows, obs.cols, strict=False)
            ],
            dtype=np.float64,
        ).reshape(-1)
        if len(weights) == n and np.any(np.isfinite(weights)):
            weights[~np.isfinite(weights)] = float(min_weight)
            return np.clip(weights, min_weight, 1.0)

    weights = np.ones(n, dtype=np.float64)
    row_col_to_idx: dict[tuple[int, int], int] = {}
    for idx, (row, col) in enumerate(zip(obs.rows, obs.cols, strict=False)):
        row_i = int(row)
        col_i = int(col)
        if row_i < 0 or col_i < 0:
            continue
        row_col_to_idx.setdefault((row_i, col_i), int(idx))

    if len(row_col_to_idx) < 4:
        return weights

    edges: list[tuple[int, int, float]] = []
    points = np.asarray(obs.object_points, dtype=np.float64).reshape(-1, 3)
    for (row, col), idx in row_col_to_idx.items():
        for dr, dc in ((0, 1), (1, 0)):
            other = row_col_to_idx.get((row + dr, col + dc))
            if other is None:
                continue
            distance = float(np.linalg.norm(points[int(other)] - points[int(idx)]))
            if np.isfinite(distance) and distance > 1e-9:
                edges.append((int(idx), int(other), distance))

    if len(edges) < 3:
        return weights

    distances = np.asarray([edge[2] for edge in edges], dtype=np.float64)
    expected = float(np.median(distances))
    mad = float(np.median(np.abs(distances - expected)))
    robust_sigma = 1.4826 * mad
    scale = max(0.25, 2.5 * robust_sigma)

    dev_sum = np.zeros(n, dtype=np.float64)
    support = np.zeros(n, dtype=np.float64)
    for i, j, distance in edges:
        deviation = abs(float(distance) - expected)
        dev_sum[i] += deviation
        dev_sum[j] += deviation
        support[i] += 1.0
        support[j] += 1.0

    has_support = support > 0.0
    if not np.any(has_support):
        return weights

    local_deviation = np.zeros(n, dtype=np.float64)
    local_deviation[has_support] = dev_sum[has_support] / support[has_support]
    if np.any(has_support):
        fallback_dev = float(np.max(local_deviation[has_support]))
    else:
        fallback_dev = scale
    local_deviation[~has_support] = fallback_dev + scale

    consistency = np.exp(-((local_deviation / scale) ** 2))
    support_weight = np.minimum(1.0, support / 2.0)
    raw = 0.75 * consistency + 0.25 * support_weight
    weights = float(min_weight) + (1.0 - float(min_weight)) * np.clip(raw, 0.0, 1.0)
    weights[~np.isfinite(weights)] = float(min_weight)
    return weights.reshape(-1)


def _reprojection_stats(errors: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    weights_arr = None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights_arr is None:
        downweighted = math.nan
        weight_min = math.nan
        weight_median = math.nan
    else:
        finite_weights = weights_arr[np.isfinite(weights_arr)]
        downweighted = float(np.mean(finite_weights < 0.99)) if len(finite_weights) else math.nan
        weight_min = float(np.min(finite_weights)) if len(finite_weights) else math.nan
        weight_median = float(np.median(finite_weights)) if len(finite_weights) else math.nan

    return {
        "reproj_mean_px": _mean(errors),
        "reproj_median_px": _median(errors),
        "reproj_rms_px": _rms(errors),
        "reproj_p95_px": _percentile(errors, 95),
        "reproj_max_px": float(np.max(_finite(errors))) if len(_finite(errors)) else math.nan,
        "downweighted_fraction": downweighted,
        "weight_min": weight_min,
        "weight_median": weight_median,
    }


def solve_opencv_replay(
    obs: Observation,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    refine: str,
) -> tuple[bool, np.ndarray, np.ndarray, dict[str, float]]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV/cv2 is required for PnP replay.") from exc

    rvec = obs.original_rvec.reshape(3, 1).astype(np.float64)
    tvec = obs.original_tvec.reshape(3, 1).astype(np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            obs.object_points.reshape(-1, 3),
            obs.image_points.reshape(-1, 2),
            K.reshape(3, 3),
            dist.reshape(-1, 1),
            rvec=rvec.copy(),
            tvec=tvec.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return False, obs.original_rvec.copy(), obs.original_tvec.copy(), {}

        refine = str(refine).strip().lower()
        if refine == "vvs" and hasattr(cv2, "solvePnPRefineVVS"):
            rvec, tvec = cv2.solvePnPRefineVVS(
                obs.object_points.reshape(-1, 3),
                obs.image_points.reshape(-1, 2),
                K.reshape(3, 3),
                dist.reshape(-1, 1),
                rvec,
                tvec,
            )
        elif refine == "lm" and hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                obs.object_points.reshape(-1, 3),
                obs.image_points.reshape(-1, 2),
                K.reshape(3, 3),
                dist.reshape(-1, 1),
                rvec,
                tvec,
            )

        _projected, residual, _J = _project_with_jacobian(
            obs.object_points,
            obs.image_points,
            rvec.reshape(3),
            tvec.reshape(3),
            K,
            dist,
        )
        errors = np.sqrt(np.sum(residual * residual, axis=1))
        return True, rvec.reshape(3), tvec.reshape(3), _reprojection_stats(errors)
    except Exception:
        return False, obs.original_rvec.copy(), obs.original_tvec.copy(), {}


def solve_robust_replay(
    obs: Observation,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    loss: str,
    threshold_px: float,
    iterations: int,
    damping: float,
) -> tuple[bool, np.ndarray, np.ndarray, dict[str, float]]:
    rvec = obs.original_rvec.copy().astype(np.float64).reshape(3)
    tvec = obs.original_tvec.copy().astype(np.float64).reshape(3)
    final_weights = np.ones(obs.point_count, dtype=np.float64)

    for _iteration in range(max(1, int(iterations))):
        try:
            _projected, residual, J = _project_with_jacobian(
                obs.object_points,
                obs.image_points,
                rvec,
                tvec,
                K,
                dist,
            )
        except Exception:
            return False, obs.original_rvec.copy(), obs.original_tvec.copy(), {}

        errors = np.sqrt(np.sum(residual * residual, axis=1))
        weights = _weights_from_errors(errors, loss, threshold_px)
        if not np.all(np.isfinite(weights)) or float(np.max(weights)) <= 0.0:
            return False, obs.original_rvec.copy(), obs.original_tvec.copy(), {}
        final_weights = weights

        sqrt_w = np.repeat(np.sqrt(np.maximum(weights, 1e-12)), 2)
        Jw = J * sqrt_w[:, None]
        rw = residual.reshape(-1) * sqrt_w
        normal = Jw.T @ Jw
        rhs = -(Jw.T @ rw)
        diag_scale = float(np.mean(np.diag(normal))) if np.all(np.isfinite(normal)) else math.nan
        if np.isfinite(diag_scale) and diag_scale > 0.0:
            normal = normal + float(damping) * diag_scale * np.eye(6)

        try:
            delta = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            try:
                delta, *_ = np.linalg.lstsq(normal, rhs, rcond=None)
            except np.linalg.LinAlgError:
                return False, obs.original_rvec.copy(), obs.original_tvec.copy(), {}

        if not np.all(np.isfinite(delta)):
            return False, obs.original_rvec.copy(), obs.original_tvec.copy(), {}

        rot_step = float(np.linalg.norm(delta[:3]))
        trans_step = float(np.linalg.norm(delta[3:]))
        if rot_step > 0.05:
            delta[:3] *= 0.05 / max(rot_step, 1e-12)
        if trans_step > 5.0:
            delta[3:] *= 5.0 / max(trans_step, 1e-12)

        rvec = rvec + delta[:3]
        tvec = tvec + delta[3:]
        if float(np.linalg.norm(delta)) < 1e-9:
            break

    try:
        _projected, residual, _J = _project_with_jacobian(
            obs.object_points,
            obs.image_points,
            rvec,
            tvec,
            K,
            dist,
        )
    except Exception:
        return False, obs.original_rvec.copy(), obs.original_tvec.copy(), {}

    errors = np.sqrt(np.sum(residual * residual, axis=1))
    stats = _reprojection_stats(errors, final_weights)
    return True, rvec, tvec, stats


def solve_gnc_replay(
    obs: Observation,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    threshold_px: float,
    iterations: int,
    refine: str,
    geometry_prior: bool,
    model_prior: dict[tuple[int, int], float] | None,
    kappa: float = 5.0,
    gamma: float = 0.5,
    tau_gnc: float = 0.5,
    tau_geom: float = 0.9,
    min_inliers: int = 6,
) -> tuple[bool, np.ndarray, np.ndarray, dict[str, float]]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV/cv2 is required for GNC-PnP replay.") from exc

    rvec = obs.original_rvec.copy().astype(np.float64).reshape(3)
    tvec = obs.original_tvec.copy().astype(np.float64).reshape(3)
    mu_final = max(float(threshold_px), 1e-6)
    geom_weights = (
        _geometry_aware_weights(obs, model_prior=model_prior)
        if bool(geometry_prior)
        else np.ones(obs.point_count, dtype=np.float64)
    )
    if len(geom_weights) != obs.point_count:
        geom_weights = np.ones(obs.point_count, dtype=np.float64)

    try:
        _projected, residual, _J = _project_with_jacobian(
            obs.object_points,
            obs.image_points,
            rvec,
            tvec,
            K,
            dist,
        )
    except Exception:
        return False, obs.original_rvec.copy(), obs.original_tvec.copy(), {}

    errors = np.sqrt(np.sum(residual * residual, axis=1))
    median_error = _median(errors)
    if not np.isfinite(median_error):
        return False, obs.original_rvec.copy(), obs.original_tvec.copy(), {}
    mu = max(float(kappa) * median_error + 1e-9, mu_final)
    active = np.ones(obs.point_count, dtype=bool)
    combined_weights = np.ones(obs.point_count, dtype=np.float64)

    for _iteration in range(max(1, int(iterations))):
        try:
            _projected, residual, _J = _project_with_jacobian(
                obs.object_points,
                obs.image_points,
                rvec,
                tvec,
                K,
                dist,
            )
        except Exception:
            return False, obs.original_rvec.copy(), obs.original_tvec.copy(), {}

        errors = np.sqrt(np.sum(residual * residual, axis=1))
        gnc_weights = (mu * mu) / (errors * errors + mu * mu)
        combined_weights = gnc_weights * geom_weights
        active = (gnc_weights > float(tau_gnc)) & (geom_weights > float(tau_geom))

        if int(np.count_nonzero(active)) < int(min_inliers):
            order = np.argsort(combined_weights)[::-1]
            active = np.zeros(obs.point_count, dtype=bool)
            active[order[: int(min_inliers)]] = True

        if int(np.count_nonzero(active)) < int(min_inliers):
            return False, obs.original_rvec.copy(), obs.original_tvec.copy(), {}

        try:
            ok, rvec_new, tvec_new = cv2.solvePnP(
                obs.object_points[active].reshape(-1, 3),
                obs.image_points[active].reshape(-1, 2),
                K.reshape(3, 3),
                dist.reshape(-1, 1),
                rvec=rvec.reshape(3, 1).copy(),
                tvec=tvec.reshape(3, 1).copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            return False, obs.original_rvec.copy(), obs.original_tvec.copy(), {}

        if not ok:
            return False, obs.original_rvec.copy(), obs.original_tvec.copy(), {}
        rvec = np.asarray(rvec_new, dtype=np.float64).reshape(3)
        tvec = np.asarray(tvec_new, dtype=np.float64).reshape(3)
        next_mu = max(float(gamma) * mu, mu_final)
        if abs(next_mu - mu) < 1e-12 and mu <= mu_final + 1e-12:
            break
        mu = next_mu

    refine = str(refine).strip().lower()
    try:
        if int(np.count_nonzero(active)) >= int(min_inliers):
            if refine == "vvs" and hasattr(cv2, "solvePnPRefineVVS"):
                rvec_ref, tvec_ref = cv2.solvePnPRefineVVS(
                    obs.object_points[active].reshape(-1, 3),
                    obs.image_points[active].reshape(-1, 2),
                    K.reshape(3, 3),
                    dist.reshape(-1, 1),
                    rvec.reshape(3, 1),
                    tvec.reshape(3, 1),
                )
                rvec = np.asarray(rvec_ref, dtype=np.float64).reshape(3)
                tvec = np.asarray(tvec_ref, dtype=np.float64).reshape(3)
            elif refine == "lm" and hasattr(cv2, "solvePnPRefineLM"):
                rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
                    obs.object_points[active].reshape(-1, 3),
                    obs.image_points[active].reshape(-1, 2),
                    K.reshape(3, 3),
                    dist.reshape(-1, 1),
                    rvec.reshape(3, 1),
                    tvec.reshape(3, 1),
                )
                rvec = np.asarray(rvec_ref, dtype=np.float64).reshape(3)
                tvec = np.asarray(tvec_ref, dtype=np.float64).reshape(3)
    except Exception:
        pass

    try:
        _projected, residual, _J = _project_with_jacobian(
            obs.object_points,
            obs.image_points,
            rvec,
            tvec,
            K,
            dist,
        )
    except Exception:
        return False, obs.original_rvec.copy(), obs.original_tvec.copy(), {}

    errors = np.sqrt(np.sum(residual * residual, axis=1))
    gnc_weights = (mu_final * mu_final) / (errors * errors + mu_final * mu_final)
    combined_weights = gnc_weights * geom_weights
    active = (gnc_weights > float(tau_gnc)) & (geom_weights > float(tau_geom))
    stats = _reprojection_stats(errors, combined_weights)
    selected_errors = errors[active]
    stats.update(
        {
            "gnc_inliers": int(np.count_nonzero(active)),
            "gnc_inlier_fraction": float(np.mean(active)) if len(active) else math.nan,
            "gnc_mu_final_px": float(mu_final),
            "geom_prior_enabled": int(bool(geometry_prior)),
            "geom_weight_min": float(np.min(geom_weights)) if len(geom_weights) else math.nan,
            "geom_weight_median": float(np.median(geom_weights)) if len(geom_weights) else math.nan,
            "selected_reproj_mean_px": _mean(selected_errors),
            "selected_reproj_rms_px": _rms(selected_errors),
            "selected_reproj_p95_px": _percentile(selected_errors, 95),
        }
    )
    return True, rvec, tvec, stats


def _fit_branch_model(
    rows: list[dict[str, Any]],
    *,
    geometry: str,
    include_branch: bool,
) -> dict[str, Any]:
    x = np.asarray([_to_float(row.get("rel_x_mm")) for row in rows], dtype=np.float64)
    y = np.asarray([_to_float(row.get("rel_y_mm")) for row in rows], dtype=np.float64)
    z = np.asarray([_to_float(row.get("rel_z_mm")) for row in rows], dtype=np.float64)
    branch = np.asarray([1.0 if str(row.get("branch")) == "return" else 0.0 for row in rows], dtype=np.float64)

    columns: list[np.ndarray] = [np.ones_like(z)]
    names = ["intercept"]
    if geometry == "none":
        pass
    elif geometry == "y":
        columns.append(y)
        names.append("y")
    elif geometry == "xy":
        columns.extend([x, y])
        names.extend(["x", "y"])
    elif geometry == "xy_quadratic":
        columns.extend([x, y, x * x, y * y, x * y])
        names.extend(["x", "y", "x2", "y2", "xy"])
    else:
        raise RuntimeError(f"Unknown geometry model: {geometry}")

    if include_branch:
        columns.append(branch)
        names.append("branch_return")

    X = np.column_stack(columns)
    mask = np.isfinite(z) & np.all(np.isfinite(X), axis=1)
    if int(np.sum(mask)) < max(4, len(names) + 2):
        return {"r2": math.nan, "rmse_mm": math.nan, "branch_coeff_mm": math.nan}

    try:
        beta, *_ = np.linalg.lstsq(X[mask], z[mask], rcond=None)
    except np.linalg.LinAlgError:
        return {"r2": math.nan, "rmse_mm": math.nan, "branch_coeff_mm": math.nan}

    pred = X[mask] @ beta
    zz = z[mask]
    ss_res = float(np.sum((zz - pred) ** 2))
    ss_tot = float(np.sum((zz - float(np.mean(zz))) ** 2))
    branch_coeff = math.nan
    if include_branch and "branch_return" in names:
        branch_coeff = float(beta[names.index("branch_return")])
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan,
        "rmse_mm": float(math.sqrt(float(np.mean((zz - pred) ** 2)))),
        "branch_coeff_mm": branch_coeff,
    }


def _nearest_pair_metric(rows: list[dict[str, Any]], threshold_mm: float) -> tuple[int, float, float]:
    out_rows = [row for row in rows if str(row.get("branch")) == "out"]
    return_rows = [row for row in rows if str(row.get("branch")) == "return"]
    if not out_rows or not return_rows:
        return 0, math.nan, math.nan

    out_xy = np.asarray([[_to_float(row.get("rel_x_mm")), _to_float(row.get("rel_y_mm"))] for row in out_rows], dtype=np.float64)
    out_z = np.asarray([_to_float(row.get("rel_z_mm")) for row in out_rows], dtype=np.float64)
    deltas: list[float] = []
    distances: list[float] = []
    for return_row in return_rows:
        return_xy = np.asarray([_to_float(return_row.get("rel_x_mm")), _to_float(return_row.get("rel_y_mm"))], dtype=np.float64)
        return_z = _to_float(return_row.get("rel_z_mm"))
        if not np.all(np.isfinite(return_xy)) or not np.isfinite(return_z):
            continue
        distance = np.sqrt(np.sum((out_xy - return_xy.reshape(1, 2)) ** 2, axis=1))
        distance[~np.isfinite(distance) | ~np.isfinite(out_z)] = math.inf
        if not np.any(np.isfinite(distance)):
            continue
        idx = int(np.argmin(distance))
        if float(distance[idx]) <= float(threshold_mm):
            deltas.append(float(return_z - out_z[idx]))
            distances.append(float(distance[idx]))
    return len(deltas), _median(deltas), _median(distances)


def summarize_method(
    rows: list[dict[str, Any]],
    method: ReplayMethod,
    thresholds_mm: list[float],
) -> dict[str, Any]:
    rel_x = np.asarray([_to_float(row.get("rel_x_mm")) for row in rows], dtype=np.float64)
    rel_y = np.asarray([_to_float(row.get("rel_y_mm")) for row in rows], dtype=np.float64)
    rel_z = np.asarray([_to_float(row.get("rel_z_mm")) for row in rows], dtype=np.float64)
    branches = np.asarray([str(row.get("branch")) for row in rows], dtype=object)
    out_z = rel_z[branches == "out"]
    return_z = rel_z[branches == "return"]

    xy = _fit_branch_model(rows, geometry="xy", include_branch=True)
    xy_quad = _fit_branch_model(rows, geometry="xy_quadratic", include_branch=True)
    y_model = _fit_branch_model(rows, geometry="y", include_branch=True)

    row: dict[str, Any] = {
        "method": method.name,
        "loss": method.loss,
        "threshold_px": method.threshold_px,
        "frames": len(rows),
        "solve_failures": int(sum(1 for r in rows if _to_int(r.get("solved"), 0) == 0)),
        "x_closure_mm": float(rel_x[-1] - rel_x[0]) if len(rel_x) else math.nan,
        "y_closure_mm": float(rel_y[-1] - rel_y[0]) if len(rel_y) else math.nan,
        "z_closure_mm": float(rel_z[-1] - rel_z[0]) if len(rel_z) else math.nan,
        "z_range_mm": float(np.nanmax(rel_z) - np.nanmin(rel_z)) if len(_finite(rel_z)) else math.nan,
        "out_z_median_mm": _median(out_z),
        "return_z_median_mm": _median(return_z),
        "return_minus_out_z_median_mm": _median(return_z) - _median(out_z),
        "branch_coeff_y_mm": y_model["branch_coeff_mm"],
        "branch_coeff_xy_mm": xy["branch_coeff_mm"],
        "branch_coeff_xy_quadratic_mm": xy_quad["branch_coeff_mm"],
        "r2_xy_quadratic_branch": xy_quad["r2"],
        "rmse_xy_quadratic_branch_mm": xy_quad["rmse_mm"],
        "reproj_rms_median_px": _median([_to_float(r.get("reproj_rms_px")) for r in rows]),
        "reproj_p95_median_px": _median([_to_float(r.get("reproj_p95_px")) for r in rows]),
        "reproj_max_median_px": _median([_to_float(r.get("reproj_max_px")) for r in rows]),
        "selected_reproj_rms_median_px": _median(
            [_to_float(r.get("selected_reproj_rms_px")) for r in rows]
        ),
        "gnc_inlier_fraction_median": _median(
            [_to_float(r.get("gnc_inlier_fraction")) for r in rows]
        ),
        "geom_weight_min_median": _median(
            [_to_float(r.get("geom_weight_min")) for r in rows]
        ),
        "geom_weight_median_median": _median(
            [_to_float(r.get("geom_weight_median")) for r in rows]
        ),
        "downweighted_fraction_median": _median([_to_float(r.get("downweighted_fraction")) for r in rows]),
        "translation_delta_from_original_median_mm": _median(
            [_to_float(r.get("translation_delta_from_original_mm")) for r in rows]
        ),
        "translation_delta_from_original_p95_mm": _percentile(
            [_to_float(r.get("translation_delta_from_original_mm")) for r in rows],
            95,
        ),
    }

    for threshold in thresholds_mm:
        n, dz, distance = _nearest_pair_metric(rows, threshold)
        key = str(threshold).replace(".", "p")
        row[f"nearest_xy_{key}_n"] = n
        row[f"nearest_xy_{key}_return_minus_out_z_median_mm"] = dz
        row[f"nearest_xy_{key}_distance_median_mm"] = distance
    return row


def replay_method(
    observations: list[Observation],
    run: dict[str, Any],
    method: ReplayMethod,
    *,
    iterations: int,
    damping: float,
    refine: str,
    gnc_kappa: float,
    gnc_gamma: float,
    gnc_tau_gnc: float,
    gnc_tau_geom: float,
    gnc_min_inliers: int,
) -> list[dict[str, Any]]:
    if not observations:
        return []

    rows: list[dict[str, Any]] = []
    tvecs: list[np.ndarray] = []
    rvecs: list[np.ndarray] = []
    solved_flags: list[bool] = []
    stats_by_frame: list[dict[str, float]] = []

    for obs in observations:
        if method.loss == "original":
            rvec = obs.original_rvec.copy()
            tvec = obs.original_tvec.copy()
            try:
                _projected, residual, _J = _project_with_jacobian(
                    obs.object_points,
                    obs.image_points,
                    rvec,
                    tvec,
                    run["K"],
                    run["dist"],
                )
                errors = np.sqrt(np.sum(residual * residual, axis=1))
                stats = _reprojection_stats(errors)
                solved = True
            except Exception:
                stats = {}
                solved = False
        elif method.loss == "opencv":
            solved, rvec, tvec, stats = solve_opencv_replay(obs, run["K"], run["dist"], refine=refine)
        elif method.loss in ("gnc", "gnc_geom"):
            solved, rvec, tvec, stats = solve_gnc_replay(
                obs,
                run["K"],
                run["dist"],
                threshold_px=method.threshold_px,
                iterations=iterations,
                refine=refine,
                geometry_prior=(method.loss == "gnc_geom"),
                model_prior=run.get("geometry_prior") if isinstance(run.get("geometry_prior"), dict) else None,
                kappa=gnc_kappa,
                gamma=gnc_gamma,
                tau_gnc=gnc_tau_gnc,
                tau_geom=gnc_tau_geom,
                min_inliers=gnc_min_inliers,
            )
        else:
            solved, rvec, tvec, stats = solve_robust_replay(
                obs,
                run["K"],
                run["dist"],
                loss=method.loss,
                threshold_px=method.threshold_px,
                iterations=iterations,
                damping=damping,
            )
        if not solved:
            tvec = obs.original_tvec.copy()
            rvec = obs.original_rvec.copy()
        rvecs.append(np.asarray(rvec, dtype=np.float64).reshape(3))
        tvecs.append(np.asarray(tvec, dtype=np.float64).reshape(3))
        solved_flags.append(bool(solved))
        stats_by_frame.append(stats)

    tvec_arr = np.asarray(tvecs, dtype=np.float64).reshape(-1, 3)
    rvec_arr = np.asarray(rvecs, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(tvec_arr), axis=1)
    if np.any(finite):
        origin = tvec_arr[int(np.where(finite)[0][0])].copy()
        rel = tvec_arr - origin
    else:
        rel = np.full_like(tvec_arr, np.nan)

    for idx, obs in enumerate(observations):
        stats = stats_by_frame[idx]
        trans_delta = float(np.linalg.norm(tvec_arr[idx] - obs.original_tvec))
        rvec_row = rvec_arr[idx]
        roll_deg, pitch_deg, yaw_deg = _rvec_to_euler_deg(rvec_row)
        row = {
            "method": method.name,
            "loss": method.loss,
            "threshold_px": method.threshold_px,
            "frame": int(obs.frame),
            "branch": obs.branch,
            "solved": int(solved_flags[idx]),
            "point_count": int(obs.point_count),
            "row_min": int(np.min(obs.rows)) if len(obs.rows) else -1,
            "row_max": int(np.max(obs.rows)) if len(obs.rows) else -1,
            "col_min": int(np.min(obs.cols)) if len(obs.cols) else -1,
            "col_max": int(np.max(obs.cols)) if len(obs.cols) else -1,
            "distinct_rows": int(len(set(int(v) for v in obs.rows.tolist()))),
            "distinct_cols": int(len(set(int(v) for v in obs.cols.tolist()))),
            "tvec_x_mm": float(tvec_arr[idx, 0]),
            "tvec_y_mm": float(tvec_arr[idx, 1]),
            "tvec_z_mm": float(tvec_arr[idx, 2]),
            "rvec_x_rad": float(rvec_row[0]),
            "rvec_y_rad": float(rvec_row[1]),
            "rvec_z_rad": float(rvec_row[2]),
            "roll_deg": float(roll_deg),
            "pitch_deg": float(pitch_deg),
            "yaw_deg": float(yaw_deg),
            "rel_x_mm": float(rel[idx, 0]),
            "rel_y_mm": float(rel[idx, 1]),
            "rel_z_mm": float(rel[idx, 2]),
            "orig_rel_x_mm": float(obs.original_rel_tvec[0]),
            "orig_rel_y_mm": float(obs.original_rel_tvec[1]),
            "orig_rel_z_mm": float(obs.original_rel_tvec[2]),
            "delta_from_original_x_mm": float(tvec_arr[idx, 0] - obs.original_tvec[0]),
            "delta_from_original_y_mm": float(tvec_arr[idx, 1] - obs.original_tvec[1]),
            "delta_from_original_z_mm": float(tvec_arr[idx, 2] - obs.original_tvec[2]),
            "translation_delta_from_original_mm": trans_delta,
        }
        row.update(stats)
        rows.append(row)

    return rows


def _default_methods() -> list[ReplayMethod]:
    return [
        ReplayMethod("original", "original", math.nan),
        ReplayMethod("opencv_iterative", "opencv", math.nan),
        ReplayMethod("huber_0p50", "huber", 0.50),
        ReplayMethod("huber_0p75", "huber", 0.75),
        ReplayMethod("huber_1p00", "huber", 1.00),
        ReplayMethod("cauchy_0p75", "cauchy", 0.75),
        ReplayMethod("tukey_1p00", "tukey", 1.00),
        ReplayMethod("gnc_0p50", "gnc", 0.50),
        ReplayMethod("gnc_geom_0p50", "gnc_geom", 0.50),
    ]


def _parse_methods(text: str | None) -> list[ReplayMethod]:
    if text is None or not str(text).strip():
        return _default_methods()
    methods: list[ReplayMethod] = []
    for raw in str(text).split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item == "original":
            methods.append(ReplayMethod("original", "original", math.nan))
        elif item in ("opencv", "opencv_iterative"):
            methods.append(ReplayMethod("opencv_iterative", "opencv", math.nan))
        else:
            parts = item.split(":")
            if len(parts) != 2:
                raise RuntimeError(
                    f"Bad method spec '{raw}'. Use loss:threshold, original, or opencv. "
                    "GNC examples: gnc:0.5, gnc_geom:0.5"
                )
            loss = parts[0].strip()
            threshold = float(parts[1])
            label = f"{loss}_{str(threshold).replace('.', 'p')}"
            methods.append(ReplayMethod(label, loss, threshold))
    return methods


def _parse_thresholds(text: str | None) -> list[float]:
    if text is None or not str(text).strip():
        return [0.5, 1.0, 2.0, 3.0, 5.0]
    out: list[float] = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out or [0.5, 1.0, 2.0, 3.0, 5.0]


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


def _best_summary_row(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    non_original = [row for row in summary_rows if str(row.get("method")) != "original"]
    rows = non_original or summary_rows

    def objective(row: dict[str, Any]) -> float:
        branch = abs(_to_float(row.get("branch_coeff_xy_quadratic_mm")))
        nearest = abs(_to_float(row.get("nearest_xy_0p5_return_minus_out_z_median_mm")))
        reproj = _to_float(row.get("reproj_rms_median_px"))
        delta = _to_float(row.get("translation_delta_from_original_p95_mm"))
        score = 0.0
        score += branch if np.isfinite(branch) else 3.0
        score += 0.5 * nearest if np.isfinite(nearest) else 0.0
        score += 0.05 * max(0.0, reproj - 0.5) if np.isfinite(reproj) else 0.0
        score += 0.05 * max(0.0, delta - 1.0) if np.isfinite(delta) else 0.0
        return float(score)

    return min(rows, key=objective)


def plot_results(
    run: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    show: bool,
) -> Path:
    import matplotlib

    matplotlib.use("Agg" if not show else "QtAgg")
    import matplotlib.pyplot as plt

    _setup_plot_style(plt)

    path: Path = run["path"]
    out_path = path.with_name(f"{path.stem}_robust_pnp_replay_plot.png")
    original_rows = [row for row in frame_rows if str(row.get("method")) == "original"]
    best = _best_summary_row(summary_rows)
    best_name = str(best.get("method"))
    best_rows = [row for row in frame_rows if str(row.get("method")) == best_name]

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=False)
    fig.suptitle("HydraTracker robust/weighted PnP offline replay", fontsize=16, fontweight="bold")
    fig.text(
        0.01,
        0.965,
        f"{run['run_id']} -- {run.get('timestamp', '')}   selected plot method={best_name}",
        fontsize=9,
        ha="left",
        va="top",
    )

    for rows, label, color in (
        (original_rows, "original", "#d62728"),
        (best_rows, best_name, "#1f77b4"),
    ):
        frames = np.asarray([_to_float(row.get("frame")) for row in rows], dtype=np.float64)
        z = np.asarray([_to_float(row.get("rel_z_mm")) for row in rows], dtype=np.float64)
        axes[0].plot(frames, z, marker="o", markersize=2.5, linewidth=1.3, color=color, label=label)
    axes[0].axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].set_title("Relative z replay")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("delta z [mm]")
    axes[0].legend(loc="best", fontsize=8)

    for rows, label, color in (
        (original_rows, "original", "#d62728"),
        (best_rows, best_name, "#1f77b4"),
    ):
        y = np.asarray([_to_float(row.get("rel_y_mm")) for row in rows], dtype=np.float64)
        z = np.asarray([_to_float(row.get("rel_z_mm")) for row in rows], dtype=np.float64)
        branches = np.asarray([str(row.get("branch")) for row in rows], dtype=object)
        axes[1].scatter(y[branches == "out"], z[branches == "out"], s=14, alpha=0.65, color=color, marker="o", label=f"{label} out")
        axes[1].scatter(y[branches == "return"], z[branches == "return"], s=14, alpha=0.65, color=color, marker="x", label=f"{label} return")
    axes[1].set_title("z versus y, out/return")
    axes[1].set_xlabel("delta y [mm]")
    axes[1].set_ylabel("delta z [mm]")
    axes[1].legend(loc="best", fontsize=8, ncols=2)

    labels = [str(row.get("method")) for row in summary_rows]
    x_idx = np.arange(len(labels), dtype=np.float64)
    branch = np.asarray([_to_float(row.get("branch_coeff_xy_quadratic_mm")) for row in summary_rows], dtype=np.float64)
    nearest = np.asarray(
        [_to_float(row.get("nearest_xy_0p5_return_minus_out_z_median_mm")) for row in summary_rows],
        dtype=np.float64,
    )
    reproj = np.asarray([_to_float(row.get("reproj_rms_median_px")) for row in summary_rows], dtype=np.float64)
    axes[2].plot(x_idx, branch, marker="o", linewidth=1.4, color="#9467bd", label="xy quadratic branch")
    axes[2].plot(x_idx, nearest, marker="o", linewidth=1.4, color="#8c564b", label="nearest xy <=0.5")
    axes[2].axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[2].set_title("Hysteresis metrics by replay method")
    axes[2].set_ylabel("mm")
    axes[2].set_xticks(x_idx)
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    axes[2].legend(loc="best", fontsize=8)

    ax_reproj = axes[3]
    ax_reproj.plot(x_idx, reproj, marker="o", linewidth=1.5, color="#f58518", label="median reprojection RMS")
    ax_reproj.set_title("Reprojection cost and pose movement by method")
    ax_reproj.set_ylabel("px")
    ax_reproj.set_xticks(x_idx)
    ax_reproj.set_xticklabels(labels, rotation=30, ha="right")
    ax_delta = ax_reproj.twinx()
    pose_delta = np.asarray(
        [_to_float(row.get("translation_delta_from_original_p95_mm")) for row in summary_rows],
        dtype=np.float64,
    )
    ax_delta.plot(x_idx, pose_delta, marker="o", linewidth=1.2, color="#4c78a8", label="p95 pose shift")
    ax_delta.set_ylabel("mm")
    handles1, labels1 = ax_reproj.get_legend_handles_labels()
    handles2, labels2 = ax_delta.get_legend_handles_labels()
    ax_reproj.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.85)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _preferred_translation_methods(frame_rows: list[dict[str, Any]]) -> list[str]:
    available: list[str] = []
    for row in frame_rows:
        method = str(row.get("method") or "")
        if method and method not in available:
            available.append(method)

    preferred = [
        "original",
        "opencv_iterative",
        "gnc_0p50",
        "gnc_geom_0p50",
    ]
    selected = [method for method in preferred if method in available]
    for method in available:
        if method not in selected and len(selected) < 5:
            selected.append(method)
    return selected


def plot_translation_style_results(
    run: dict[str, Any],
    frame_rows: list[dict[str, Any]],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_plot_style(plt)

    path: Path = run["path"]
    out_path = path.with_name(f"{path.stem}_robust_pnp_replay_translation_plot.png")
    methods = _preferred_translation_methods(frame_rows)
    rows_by_method = {
        method: [row for row in frame_rows if str(row.get("method")) == method]
        for method in methods
    }
    colors = {
        "original": "#d62728",
        "opencv_iterative": "#4c78a8",
        "gnc_0p50": "#54a24b",
        "gnc_geom_0p50": "#9467bd",
    }
    fallback_colors = ["#f58518", "#72b7b2", "#e45756", "#b279a2", "#ff9da6"]

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15.5, 12.0),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(top=0.86, hspace=0.34)

    run_label = run["run_id"]
    if run.get("timestamp"):
        run_label += f"  |  {run['timestamp']}"
    fig.suptitle(
        "HydraTracker relative translation components (offline robust PnP replay, camera frame)\n"
        f"{run_label}",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    def method_color(method: str, idx: int) -> str:
        return colors.get(method, fallback_colors[idx % len(fallback_colors)])

    comp_specs = (
        ("rel_x_mm", "x", "X"),
        ("rel_y_mm", "y", "Y"),
        ("rel_z_mm", "z", "Z"),
    )
    for comp_idx, (key, label, title_label) in enumerate(comp_specs):
        ax = axes[comp_idx]
        title_parts: list[str] = []
        for idx, method in enumerate(methods):
            rows = rows_by_method.get(method, [])
            frames = np.asarray([_to_float(row.get("frame")) for row in rows], dtype=np.float64)
            values = np.asarray([_to_float(row.get(key)) for row in rows], dtype=np.float64)
            color = method_color(method, idx)
            ax.plot(
                frames,
                values,
                color=color,
                linewidth=1.7 if method in ("original", "gnc_geom_0p50") else 1.25,
                marker="o",
                markersize=2.6,
                markerfacecolor="white",
                markeredgewidth=0.7,
                alpha=0.95,
                label=method,
            )
            finite = values[np.isfinite(values)]
            if len(finite):
                title_parts.append(f"{method}={float(np.max(finite) - np.min(finite)):.2f}")

        ax.axhline(0.0, color="#888888", alpha=0.28, linewidth=1.0, linestyle="--")
        suffix = ", ".join(title_parts[:4])
        ax.set_title(
            f"{title_label} relative component   range [mm]: {suffix}",
            loc="left",
        )
        ax.set_ylabel(f"delta T_C_T {label} [mm]")
        ax.grid(True, axis="both")
        ax.legend(loc="upper right", fontsize=8)

    point_ax = axes[3]
    for idx, method in enumerate(methods):
        rows = rows_by_method.get(method, [])
        frames = np.asarray([_to_float(row.get("frame")) for row in rows], dtype=np.float64)
        if str(method).startswith("gnc"):
            values = np.asarray([_to_float(row.get("gnc_inliers")) for row in rows], dtype=np.float64)
            label = f"{method} inliers"
            linestyle = "-"
        else:
            values = np.asarray([_to_float(row.get("point_count")) for row in rows], dtype=np.float64)
            label = f"{method} points"
            linestyle = "--" if method != "original" else "-"
        point_ax.plot(
            frames,
            values,
            color=method_color(method, idx),
            linewidth=1.4,
            linestyle=linestyle,
            marker="o",
            markersize=2.2,
            markerfacecolor="white",
            markeredgewidth=0.6,
            label=label,
        )

    original_rows = rows_by_method.get("original") or next(iter(rows_by_method.values()), [])
    frames = np.asarray([_to_float(row.get("frame")) for row in original_rows], dtype=np.float64)
    row_min = np.asarray([_to_float(row.get("row_min")) for row in original_rows], dtype=np.float64)
    row_max = np.asarray([_to_float(row.get("row_max")) for row in original_rows], dtype=np.float64)
    point_ax.set_ylabel("points / inliers")
    point_ax.set_title("Pose diagnostics   point count, GNC inliers, and global row range", loc="left")
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
    orientation_methods = [
        method for method in ("original", "gnc_geom_0p50", "gnc_0p50")
        if method in rows_by_method
    ]
    if not orientation_methods:
        orientation_methods = methods[:2]
    linestyles = {
        "original": "-",
        "gnc_geom_0p50": "--",
        "gnc_0p50": ":",
    }
    for method in orientation_methods:
        rows = rows_by_method.get(method, [])
        frames = np.asarray([_to_float(row.get("frame")) for row in rows], dtype=np.float64)
        for key, label, color in orientation_specs:
            values = np.asarray([_to_float(row.get(key)) for row in rows], dtype=np.float64)
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            rel = values - values[np.where(finite)[0][0]]
            rel[~finite] = np.nan
            orient_ax.plot(
                frames,
                rel,
                color=color,
                linestyle=linestyles.get(method, "-."),
                linewidth=1.45,
                label=f"{method} {label}",
            )

    orient_ax.axhline(0.0, color="#888888", alpha=0.3, linewidth=1.0, linestyle="--")
    orient_ax.set_ylabel("rotation delta [deg]")
    orient_ax.set_title("Orientation diagnostics   camera-frame Euler deltas", loc="left")
    orient_ax.grid(True, axis="both")
    orient_ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frame")

    original = rows_by_method.get("original", [])
    origin_frame = int(_to_float(original[0].get("frame"))) if original else -1
    origin = (
        _to_float(original[0].get("tvec_x_mm")) if original else math.nan,
        _to_float(original[0].get("tvec_y_mm")) if original else math.nan,
        _to_float(original[0].get("tvec_z_mm")) if original else math.nan,
    )
    method_text = ", ".join(methods)
    fig.text(
        0.01,
        0.925,
        f"relative to frame {origin_frame} "
        f"({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}) mm   "
        f"frames={len(original_rows)}   methods={method_text}",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#333333",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
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
        "max_frames": None,
        "iterations": 8,
        "damping": 1e-6,
        "refine": "vvs",
        "methods": None,
        "thresholds_mm": None,
        "gnc_kappa": 5.0,
        "gnc_gamma": 0.5,
        "gnc_tau_gnc": 0.5,
        "gnc_tau_geom": 0.9,
        "gnc_min_inliers": 6,
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
        elif arg == "--max-frames":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--max-frames needs an integer")
            args["max_frames"] = int(argv[idx])
        elif arg == "--iterations":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--iterations needs an integer")
            args["iterations"] = int(argv[idx])
        elif arg == "--damping":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--damping needs a numeric value")
            args["damping"] = float(argv[idx])
        elif arg == "--refine":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--refine needs one of: none, lm, vvs")
            refine = str(argv[idx]).strip().lower()
            if refine not in ("none", "lm", "vvs"):
                raise RuntimeError("--refine must be one of: none, lm, vvs")
            args["refine"] = refine
        elif arg == "--methods":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--methods needs a comma-separated list")
            args["methods"] = str(argv[idx])
        elif arg == "--gnc-kappa":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--gnc-kappa needs a numeric value")
            args["gnc_kappa"] = float(argv[idx])
        elif arg == "--gnc-gamma":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--gnc-gamma needs a numeric value")
            args["gnc_gamma"] = float(argv[idx])
        elif arg == "--gnc-tau-gnc":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--gnc-tau-gnc needs a numeric value")
            args["gnc_tau_gnc"] = float(argv[idx])
        elif arg == "--gnc-tau-geom":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--gnc-tau-geom needs a numeric value")
            args["gnc_tau_geom"] = float(argv[idx])
        elif arg == "--gnc-min-inliers":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--gnc-min-inliers needs an integer")
            args["gnc_min_inliers"] = int(argv[idx])
        elif arg == "--xy-thresholds":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--xy-thresholds needs comma-separated mm values")
            args["thresholds_mm"] = str(argv[idx])
        elif arg.endswith(".jsonl"):
            args["path"] = Path(arg)
        else:
            raise RuntimeError(f"Unknown option: {arg}")
        idx += 1
    return args


def _print_summary(paths: dict[str, Path], summary_rows: list[dict[str, Any]]) -> None:
    print(f"[res_robust_pnp_replay] saved frame csv   -> {paths['frame'].resolve()}")
    print(f"[res_robust_pnp_replay] saved summary csv -> {paths['summary'].resolve()}")
    if "plot" in paths:
        print(f"[res_robust_pnp_replay] saved plot        -> {paths['plot'].resolve()}")
    if "translation_plot" in paths:
        print(
            "[res_robust_pnp_replay] saved trans plot  -> "
            f"{paths['translation_plot'].resolve()}"
        )

    print("[res_robust_pnp_replay] method summary:")
    for row in summary_rows:
        gnc_note = ""
        if str(row.get("loss")) in ("gnc", "gnc_geom"):
            gnc_note = (
                f", gnc_inliers={_to_float(row.get('gnc_inlier_fraction_median')):.3f}"
                f", geom_min={_to_float(row.get('geom_weight_min_median')):.3f}"
                f", downweighted={_to_float(row.get('downweighted_fraction_median')):.3f}"
            )
        print(
            "  "
            f"{row['method']}: "
            f"z_closure={_to_float(row.get('z_closure_mm')):+.3f} mm, "
            f"xy_quad_branch={_to_float(row.get('branch_coeff_xy_quadratic_mm')):+.3f} mm, "
            f"near0.5={_to_float(row.get('nearest_xy_0p5_return_minus_out_z_median_mm')):+.3f} mm, "
            f"rms={_to_float(row.get('reproj_rms_median_px')):.3f} px, "
            f"p95_pose_shift={_to_float(row.get('translation_delta_from_original_p95_mm')):.3f} mm"
            f"{gnc_note}"
        )


def main() -> None:
    args = _parse_args(sys.argv[1:])
    path = args["path"]
    if path is None:
        path = _latest_run_path() if args["use_latest"] else _select_run_path_qt()
    path = Path(path).resolve()
    run = load_run(path)

    methods = _parse_methods(args["methods"])
    thresholds_mm = _parse_thresholds(args["thresholds_mm"])
    observations, _meta = build_observations(
        run,
        point_set=str(args["point_set"]),
        max_frames=args["max_frames"],
    )

    all_frame_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        frame_rows = replay_method(
            observations,
            run,
            method,
            iterations=int(args["iterations"]),
            damping=float(args["damping"]),
            refine=str(args["refine"]),
            gnc_kappa=float(args["gnc_kappa"]),
            gnc_gamma=float(args["gnc_gamma"]),
            gnc_tau_gnc=float(args["gnc_tau_gnc"]),
            gnc_tau_geom=float(args["gnc_tau_geom"]),
            gnc_min_inliers=int(args["gnc_min_inliers"]),
        )
        all_frame_rows.extend(frame_rows)
        summary_rows.append(summarize_method(frame_rows, method, thresholds_mm))

    frame_csv = path.with_name(f"{path.stem}_robust_pnp_replay_frames.csv")
    summary_csv = path.with_name(f"{path.stem}_robust_pnp_replay_summary.csv")
    _write_csv(frame_csv, all_frame_rows)
    _write_csv(summary_csv, summary_rows)
    paths = {"frame": frame_csv, "summary": summary_csv}
    if bool(args["make_plot"]):
        paths["plot"] = plot_results(run, all_frame_rows, summary_rows, show=bool(args["show"]))
        paths["translation_plot"] = plot_translation_style_results(run, all_frame_rows)

    _print_summary(paths, summary_rows)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[res_robust_pnp_replay] ERROR: {exc}")
        sys.exit(1)
