"""Static UV-freeze replay experiment for HydraMarker tracker logs.

The module freezes selected image observations across frames to study how much
pose drift is driven by UV jitter versus geometry or solver behavior.
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


COMPONENTS = (("x", "X"), ("y", "Y"), ("z", "Z"))


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


@dataclass(frozen=True)
class Observation:
    frame: int
    object_by_key: dict[tuple[int, int], np.ndarray]
    uv_by_key: dict[tuple[int, int], np.ndarray]
    original_rvec: np.ndarray
    original_tvec: np.ndarray


def load_run(path: Path, *, point_set: str) -> dict[str, Any]:
    K: np.ndarray | None = None
    dist: np.ndarray | None = None
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
    if not frames:
        raise RuntimeError(f"No frame records found in:\n{path}")

    corners_key = "pose_corners" if point_set == "pose" else "correspondence_corners"
    observations: list[Observation] = []
    for frame in sorted(frames):
        data = frames[frame]
        if _to_int(data.get("success"), default=0) == 0:
            continue
        detail = details.get(frame, {})
        corners = list(detail.get(corners_key) or [])
        if not corners and point_set == "pose":
            corners = list(detail.get("correspondence_corners") or [])

        object_by_key: dict[tuple[int, int], np.ndarray] = {}
        uv_by_key: dict[tuple[int, int], np.ndarray] = {}
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
            if not np.all(np.isfinite(xyz_arr)) or not np.all(np.isfinite(uv_arr)):
                continue
            row = _to_int(corner.get("global_row"), default=_to_int(corner.get("row"), -1))
            col = _to_int(corner.get("global_col"), default=_to_int(corner.get("col"), -1))
            if row < 0 or col < 0:
                continue
            key = (int(row), int(col))
            object_by_key[key] = xyz_arr.reshape(3)
            uv_by_key[key] = uv_arr.reshape(2)

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
        if len(uv_by_key) < 6 or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue
        observations.append(
            Observation(
                frame=int(frame),
                object_by_key=object_by_key,
                uv_by_key=uv_by_key,
                original_rvec=rvec,
                original_tvec=tvec,
            )
        )

    if not observations:
        raise RuntimeError("No usable frame_detail observations found.")
    return {
        "path": Path(path),
        "run_id": run_id,
        "timestamp": timestamp,
        "K": K,
        "dist": dist,
        "observations": observations,
    }


def build_static_uv_model(observations: list[Observation]) -> dict[str, Any]:
    all_keys = sorted({key for obs in observations for key in obs.uv_by_key})
    uv_values: dict[tuple[int, int], list[np.ndarray]] = {key: [] for key in all_keys}
    object_values: dict[tuple[int, int], list[np.ndarray]] = {key: [] for key in all_keys}
    for obs in observations:
        for key, uv in obs.uv_by_key.items():
            uv_values[key].append(np.asarray(uv, dtype=np.float64).reshape(2))
            object_values[key].append(np.asarray(obs.object_by_key[key], dtype=np.float64).reshape(3))

    uv_median: dict[tuple[int, int], np.ndarray] = {}
    object_median: dict[tuple[int, int], np.ndarray] = {}
    point_rows: list[dict[str, Any]] = []
    for key in all_keys:
        uv_arr = np.asarray(uv_values[key], dtype=np.float64).reshape(-1, 2)
        obj_arr = np.asarray(object_values[key], dtype=np.float64).reshape(-1, 3)
        med_uv = np.median(uv_arr, axis=0)
        uv_median[key] = med_uv
        object_median[key] = np.median(obj_arr, axis=0)
        delta = uv_arr - med_uv.reshape(1, 2)
        motion = np.sqrt(np.sum(delta * delta, axis=1))
        point_rows.append(
            {
                "global_row": key[0],
                "global_col": key[1],
                "count": int(len(uv_arr)),
                "present_fraction": float(len(uv_arr) / max(len(observations), 1)),
                "u_median_px": float(med_uv[0]),
                "v_median_px": float(med_uv[1]),
                "u_std_px": float(np.std(uv_arr[:, 0])),
                "v_std_px": float(np.std(uv_arr[:, 1])),
                "uv_motion_rms_px": _rms(motion),
                "uv_motion_p95_px": _percentile(motion, 95),
                "uv_motion_max_px": float(np.max(motion)) if len(motion) else math.nan,
            }
        )

    common_keys = sorted(set.intersection(*(set(obs.uv_by_key) for obs in observations)))
    return {
        "uv_median": uv_median,
        "object_median": object_median,
        "common_keys": common_keys,
        "point_rows": point_rows,
    }


def _arrays_for_keys(
    obs: Observation,
    keys: list[tuple[int, int]],
    *,
    uv_mode: str,
    static_model: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_keys: list[tuple[int, int]] = []
    uv_median: dict[tuple[int, int], np.ndarray] = static_model["uv_median"]
    object_median: dict[tuple[int, int], np.ndarray] = static_model["object_median"]

    for key in keys:
        if key not in obs.uv_by_key:
            continue
        object_points.append(object_median.get(key, obs.object_by_key[key]))
        if uv_mode == "frozen":
            image_points.append(uv_median[key])
        else:
            image_points.append(obs.uv_by_key[key])
        used_keys.append(key)

    return (
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
        used_keys,
    )


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
        raise RuntimeError("OpenCV/cv2 is required for UV freeze replay.") from exc

    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    rvec = np.asarray(seed_rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(seed_tvec, dtype=np.float64).reshape(3, 1)
    if len(object_points) < 6:
        return False, rvec.reshape(3), tvec.reshape(3), {}

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
        return False, rvec.reshape(3), tvec.reshape(3), {}
    if not ok:
        return False, rvec.reshape(3), tvec.reshape(3), {}

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
    projected, _ = cv2.projectPoints(
        object_points,
        rvec_out.reshape(3, 1),
        tvec_out.reshape(3, 1),
        K.reshape(3, 3),
        dist.reshape(-1, 1),
    )
    residual = projected.reshape(-1, 2) - image_points
    errors = np.sqrt(np.sum(residual * residual, axis=1))
    stats = {
        "reproj_mean_px": _mean(errors),
        "reproj_median_px": _median(errors),
        "reproj_rms_px": _rms(errors),
        "reproj_p95_px": _percentile(errors, 95),
        "reproj_max_px": float(np.max(_finite(errors))) if len(_finite(errors)) else math.nan,
    }
    return True, rvec_out, tvec_out, stats


def replay(run: dict[str, Any], static_model: dict[str, Any], *, refine: str) -> list[dict[str, Any]]:
    observations: list[Observation] = run["observations"]
    reference = observations[0]
    ref_rvec = reference.original_rvec.copy()
    ref_tvec = reference.original_tvec.copy()
    common_keys: list[tuple[int, int]] = static_model["common_keys"]

    methods = (
        ("logged_original", "all", "current", False),
        ("current_uv_all", "all", "current", True),
        ("frozen_uv_all", "all", "frozen", True),
        ("common_current_uv", "common", "current", True),
        ("common_frozen_uv", "common", "frozen", True),
    )

    raw_rows: list[dict[str, Any]] = []
    tvecs_by_method: dict[str, list[np.ndarray]] = {method[0]: [] for method in methods}

    for obs in observations:
        all_keys = sorted(obs.uv_by_key)
        for method, key_mode, uv_mode, do_solve in methods:
            keys = common_keys if key_mode == "common" else all_keys
            if method == "logged_original":
                rvec = obs.original_rvec.copy()
                tvec = obs.original_tvec.copy()
                used_keys = all_keys
                solved = True
                stats: dict[str, float] = {}
            else:
                object_points, image_points, used_keys = _arrays_for_keys(
                    obs,
                    keys,
                    uv_mode=uv_mode,
                    static_model=static_model,
                )
                solved, rvec, tvec, stats = solve_pose(
                    object_points,
                    image_points,
                    run["K"],
                    run["dist"],
                    ref_rvec,
                    ref_tvec,
                    refine=refine,
                )
                if not solved:
                    rvec = ref_rvec.copy()
                    tvec = ref_tvec.copy()

            uv_motion = []
            for key in used_keys:
                if key in obs.uv_by_key:
                    uv_motion.append(
                        float(np.linalg.norm(obs.uv_by_key[key] - static_model["uv_median"][key]))
                    )
            roll_deg, pitch_deg, yaw_deg = _rvec_to_euler_deg(rvec)
            row = {
                "method": method,
                "frame": int(obs.frame),
                "solved": int(solved),
                "point_count": int(len(used_keys)),
                "row_min": int(min((key[0] for key in used_keys), default=-1)),
                "row_max": int(max((key[0] for key in used_keys), default=-1)),
                "col_min": int(min((key[1] for key in used_keys), default=-1)),
                "col_max": int(max((key[1] for key in used_keys), default=-1)),
                "distinct_rows": int(len(set(key[0] for key in used_keys))),
                "distinct_cols": int(len(set(key[1] for key in used_keys))),
                "median_uv_motion_px": _median(uv_motion),
                "p95_uv_motion_px": _percentile(uv_motion, 95),
                "tvec_x_mm": float(tvec[0]),
                "tvec_y_mm": float(tvec[1]),
                "tvec_z_mm": float(tvec[2]),
                "rvec_x_rad": float(rvec[0]),
                "rvec_y_rad": float(rvec[1]),
                "rvec_z_rad": float(rvec[2]),
                "roll_deg": float(roll_deg),
                "pitch_deg": float(pitch_deg),
                "yaw_deg": float(yaw_deg),
            }
            row.update(stats)
            raw_rows.append(row)
            tvecs_by_method[method].append(np.asarray(tvec, dtype=np.float64).reshape(3))

    rel_by_method: dict[str, np.ndarray] = {}
    for method, *_rest in methods:
        arr = np.asarray(tvecs_by_method[method], dtype=np.float64).reshape(-1, 3)
        origin = arr[0].copy()
        rel_by_method[method] = arr - origin

    method_indices = {method[0]: 0 for method in methods}
    for row in raw_rows:
        method = str(row["method"])
        idx = method_indices[method]
        rel = rel_by_method[method][idx]
        row["rel_x_mm"] = float(rel[0])
        row["rel_y_mm"] = float(rel[1])
        row["rel_z_mm"] = float(rel[2])
        method_indices[method] += 1
    return raw_rows


def summarize(frame_rows: list[dict[str, Any]], point_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods: list[str] = []
    for row in frame_rows:
        method = str(row.get("method") or "")
        if method and method not in methods:
            methods.append(method)

    summary: list[dict[str, Any]] = []
    uv_rms = [_to_float(row.get("uv_motion_rms_px")) for row in point_rows]
    uv_p95 = [_to_float(row.get("uv_motion_p95_px")) for row in point_rows]
    for method in methods:
        rows = [row for row in frame_rows if str(row.get("method")) == method]
        rel = {
            key: np.asarray([_to_float(row.get(key)) for row in rows], dtype=np.float64)
            for key in ("rel_x_mm", "rel_y_mm", "rel_z_mm")
        }
        summary.append(
            {
                "method": method,
                "frames": len(rows),
                "solve_failures": int(sum(1 for row in rows if _to_int(row.get("solved"), 0) == 0)),
                "point_count_median": _median([_to_float(row.get("point_count")) for row in rows]),
                "x_range_mm": float(np.nanmax(rel["rel_x_mm"]) - np.nanmin(rel["rel_x_mm"])),
                "y_range_mm": float(np.nanmax(rel["rel_y_mm"]) - np.nanmin(rel["rel_y_mm"])),
                "z_range_mm": float(np.nanmax(rel["rel_z_mm"]) - np.nanmin(rel["rel_z_mm"])),
                "x_closure_mm": float(rel["rel_x_mm"][-1] - rel["rel_x_mm"][0]),
                "y_closure_mm": float(rel["rel_y_mm"][-1] - rel["rel_y_mm"][0]),
                "z_closure_mm": float(rel["rel_z_mm"][-1] - rel["rel_z_mm"][0]),
                "reproj_rms_median_px": _median([_to_float(row.get("reproj_rms_px")) for row in rows]),
                "reproj_p95_median_px": _median([_to_float(row.get("reproj_p95_px")) for row in rows]),
                "median_frame_uv_motion_px": _median([_to_float(row.get("median_uv_motion_px")) for row in rows]),
                "p95_frame_uv_motion_px": _median([_to_float(row.get("p95_uv_motion_px")) for row in rows]),
                "point_uv_rms_median_px": _median(uv_rms),
                "point_uv_rms_p95_px": _percentile(uv_rms, 95),
                "point_uv_p95_median_px": _median(uv_p95),
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


def plot_results(run: dict[str, Any], frame_rows: list[dict[str, Any]], *, show: bool) -> Path:
    import matplotlib

    matplotlib.use("Agg" if not show else "QtAgg")
    import matplotlib.pyplot as plt

    _setup_plot_style(plt)
    path: Path = run["path"]
    out_path = path.with_name(f"{path.stem}_static_uv_freeze_replay_plot.png")
    methods = [
        "logged_original",
        "current_uv_all",
        "frozen_uv_all",
        "common_current_uv",
        "common_frozen_uv",
    ]
    labels = {
        "logged_original": "logged",
        "current_uv_all": "current UV all",
        "frozen_uv_all": "frozen UV all",
        "common_current_uv": "common current UV",
        "common_frozen_uv": "common frozen UV",
    }
    colors = {
        "logged_original": "#d62728",
        "current_uv_all": "#4c78a8",
        "frozen_uv_all": "#9467bd",
        "common_current_uv": "#54a24b",
        "common_frozen_uv": "#f58518",
    }
    rows_by_method = {
        method: [row for row in frame_rows if str(row.get("method")) == method]
        for method in methods
    }

    fig, axes = plt.subplots(5, 1, figsize=(15.5, 12.0), sharex=True, constrained_layout=False)
    fig.subplots_adjust(top=0.86, hspace=0.34)
    run_label = run["run_id"]
    if run.get("timestamp"):
        run_label += f"  |  {run['timestamp']}"
    fig.suptitle(
        "HydraTracker static UV freeze replay (camera frame)\n"
        f"{run_label}",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    for comp_idx, (suffix, label) in enumerate(COMPONENTS):
        key = f"rel_{suffix}_mm"
        ax = axes[comp_idx]
        title_parts: list[str] = []
        for method in methods:
            rows = rows_by_method[method]
            frames = np.asarray([_to_float(row.get("frame")) for row in rows], dtype=np.float64)
            values = np.asarray([_to_float(row.get(key)) for row in rows], dtype=np.float64)
            ax.plot(
                frames,
                values,
                color=colors[method],
                linewidth=1.8 if method in ("logged_original", "frozen_uv_all") else 1.25,
                marker="o",
                markersize=2.4,
                markerfacecolor="white",
                markeredgewidth=0.65,
                label=labels[method],
            )
            finite = values[np.isfinite(values)]
            if len(finite):
                title_parts.append(f"{labels[method]}={float(np.max(finite) - np.min(finite)):.2f}")
        ax.axhline(0.0, color="#888888", alpha=0.28, linewidth=1.0, linestyle="--")
        ax.set_title(f"{label} relative component   range [mm]: {', '.join(title_parts[:5])}", loc="left")
        ax.set_ylabel(f"delta T_C_T {suffix} [mm]")
        ax.grid(True, axis="both")
        ax.legend(loc="upper right", fontsize=8)

    point_ax = axes[3]
    rows = rows_by_method["current_uv_all"]
    frames = np.asarray([_to_float(row.get("frame")) for row in rows], dtype=np.float64)
    points = np.asarray([_to_float(row.get("point_count")) for row in rows], dtype=np.float64)
    uv_motion = np.asarray([_to_float(row.get("median_uv_motion_px")) for row in rows], dtype=np.float64)
    point_ax.plot(frames, points, color="#4c78a8", linewidth=1.5, marker="o", markersize=2.4, label="points")
    point_ax.set_ylabel("points")
    point_ax.set_title("Pose diagnostics   point count and median per-frame UV motion", loc="left")
    point_ax.grid(True, axis="both")
    uv_ax = point_ax.twinx()
    uv_ax.plot(frames, uv_motion, color="#e45756", linewidth=1.4, label="median UV motion")
    uv_ax.set_ylabel("px")
    point_lines, point_labels = point_ax.get_legend_handles_labels()
    uv_lines, uv_labels = uv_ax.get_legend_handles_labels()
    point_ax.legend(point_lines + uv_lines, point_labels + uv_labels, loc="upper right", fontsize=8)

    orient_ax = axes[4]
    orient_methods = ("logged_original", "current_uv_all", "frozen_uv_all", "common_current_uv")
    linestyles = {
        "logged_original": "-",
        "current_uv_all": "--",
        "frozen_uv_all": ":",
        "common_current_uv": "-.",
    }
    orient_specs = (("roll_deg", "roll", "#4c78a8"), ("pitch_deg", "pitch", "#54a24b"), ("yaw_deg", "yaw", "#e45756"))
    for method in orient_methods:
        rows = rows_by_method[method]
        frames = np.asarray([_to_float(row.get("frame")) for row in rows], dtype=np.float64)
        for key, name, color in orient_specs:
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
                linestyle=linestyles[method],
                linewidth=1.35,
                label=f"{labels[method]} {name}",
            )
    orient_ax.axhline(0.0, color="#888888", alpha=0.3, linewidth=1.0, linestyle="--")
    orient_ax.set_ylabel("rotation delta [deg]")
    orient_ax.set_title("Orientation diagnostics   camera-frame Euler deltas", loc="left")
    orient_ax.grid(True, axis="both")
    orient_ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frame")

    first = rows_by_method["logged_original"][0]
    info = (
        f"relative to frame {int(_to_float(first.get('frame')))} "
        f"({_to_float(first.get('tvec_x_mm')):.2f}, {_to_float(first.get('tvec_y_mm')):.2f}, "
        f"{_to_float(first.get('tvec_z_mm')):.2f}) mm   "
        f"frames={len(rows_by_method['logged_original'])}"
    )
    fig.text(0.01, 0.925, info, ha="left", va="top", fontsize=9.5, color="#333333")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _parse_args(argv: list[str]) -> dict[str, Any]:
    args: dict[str, Any] = {
        "path": None,
        "point_set": "correspondence",
        "refine": "vvs",
        "show": False,
        "make_plot": True,
    }
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--point-set":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--point-set needs pose or correspondence")
            point_set = str(argv[idx]).strip().lower()
            if point_set not in ("pose", "correspondence"):
                raise RuntimeError("--point-set must be pose or correspondence")
            args["point_set"] = point_set
        elif arg == "--refine":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--refine needs none, lm, or vvs")
            refine = str(argv[idx]).strip().lower()
            if refine not in ("none", "lm", "vvs"):
                raise RuntimeError("--refine must be none, lm, or vvs")
            args["refine"] = refine
        elif arg == "--show":
            args["show"] = True
        elif arg == "--no-plot":
            args["make_plot"] = False
        elif arg.endswith(".jsonl"):
            args["path"] = Path(arg)
        else:
            raise RuntimeError(f"Unknown option: {arg}")
        idx += 1
    return args


def _print_summary(paths: dict[str, Path], summary_rows: list[dict[str, Any]]) -> None:
    print(f"[static_uv_freeze] saved frame csv   -> {paths['frame'].resolve()}")
    print(f"[static_uv_freeze] saved summary csv -> {paths['summary'].resolve()}")
    print(f"[static_uv_freeze] saved point csv   -> {paths['points'].resolve()}")
    if "plot" in paths:
        print(f"[static_uv_freeze] saved plot        -> {paths['plot'].resolve()}")
    print("[static_uv_freeze] method summary:")
    for row in summary_rows:
        print(
            "  "
            f"{row['method']}: "
            f"z_range={_to_float(row.get('z_range_mm')):.3f} mm, "
            f"z_closure={_to_float(row.get('z_closure_mm')):+.3f} mm, "
            f"rms={_to_float(row.get('reproj_rms_median_px')):.3f} px, "
            f"points={_to_float(row.get('point_count_median')):.1f}, "
            f"uv_med={_to_float(row.get('median_frame_uv_motion_px')):.3f} px"
        )


def main() -> None:
    args = _parse_args(sys.argv[1:])
    path = args["path"]
    if path is None:
        raise RuntimeError("Pass a HydraTracker JSONL path.")
    path = Path(path).resolve()
    run = load_run(path, point_set=str(args["point_set"]))
    static_model = build_static_uv_model(run["observations"])
    frame_rows = replay(run, static_model, refine=str(args["refine"]))
    point_rows = static_model["point_rows"]
    summary_rows = summarize(frame_rows, point_rows)

    frame_csv = path.with_name(f"{path.stem}_static_uv_freeze_replay_frames.csv")
    summary_csv = path.with_name(f"{path.stem}_static_uv_freeze_replay_summary.csv")
    point_csv = path.with_name(f"{path.stem}_static_uv_freeze_replay_points.csv")
    _write_csv(frame_csv, frame_rows)
    _write_csv(summary_csv, summary_rows)
    _write_csv(point_csv, point_rows)
    paths = {"frame": frame_csv, "summary": summary_csv, "points": point_csv}
    if bool(args["make_plot"]):
        paths["plot"] = plot_results(run, frame_rows, show=bool(args["show"]))
    _print_summary(paths, summary_rows)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[static_uv_freeze] ERROR: {exc}")
        sys.exit(1)
