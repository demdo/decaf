"""Fit and visualize motion-dependent pose bias models from tracker logs.

The module reads logged HydraMarker runs, extracts motion and residual features,
and estimates how pose bias changes with movement direction and speed.
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


def load_run(path: Path) -> dict[str, Any]:
    K: np.ndarray | None = None
    dist: np.ndarray | None = None
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
        "frames": frames,
        "details": details,
        "summary": summary,
    }


@dataclass(frozen=True)
class FrameObservation:
    frame: int
    branch: str
    movement_value_mm: float
    original_rvec: np.ndarray
    original_tvec: np.ndarray
    rel_original_tvec: np.ndarray
    object_points: np.ndarray
    image_points: np.ndarray
    velocities_px_per_frame: np.ndarray
    velocity_valid: np.ndarray
    corner_ids: list[str]
    rows: np.ndarray
    cols: np.ndarray
    residuals_px: np.ndarray
    point_count: int


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


def _corner_residual(corner: dict[str, Any]) -> tuple[float, float]:
    residual = corner.get("residual_px")
    if isinstance(residual, (list, tuple)) and len(residual) >= 2:
        return _to_float(residual[0]), _to_float(residual[1])

    uv = corner.get("uv_px")
    projected = corner.get("projected_uv_px")
    if (
        isinstance(uv, (list, tuple))
        and isinstance(projected, (list, tuple))
        and len(uv) >= 2
        and len(projected) >= 2
    ):
        return _to_float(projected[0]) - _to_float(uv[0]), _to_float(projected[1]) - _to_float(uv[1])
    return math.nan, math.nan


def _corners_to_arrays(
    corners: list[dict[str, Any]],
    *,
    frame: int,
    previous_uv_by_id: dict[str, tuple[int, np.ndarray]],
    max_gap_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    obj: list[list[float]] = []
    img: list[list[float]] = []
    vel: list[list[float]] = []
    valid_vel: list[bool] = []
    ids: list[str] = []
    rows: list[int] = []
    cols: list[int] = []
    residuals: list[list[float]] = []

    for idx, corner in enumerate(corners):
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

        cid = _corner_id(corner, idx)
        frame_gap = math.inf
        velocity = np.asarray([math.nan, math.nan], dtype=np.float64)
        prev = previous_uv_by_id.get(cid)
        if prev is not None:
            prev_frame, prev_uv = prev
            frame_gap = frame - prev_frame
            if 0 < frame_gap <= max_gap_frames and np.all(np.isfinite(prev_uv)):
                velocity = (uv_arr - prev_uv) / float(frame_gap)

        obj.append(xyz_arr.tolist())
        img.append(uv_arr.tolist())
        vel.append(velocity.tolist())
        valid_vel.append(bool(np.all(np.isfinite(velocity))))
        ids.append(cid)
        rows.append(_to_int(corner.get("global_row"), default=_to_int(corner.get("local_row"), -999)))
        cols.append(_to_int(corner.get("global_col"), default=_to_int(corner.get("local_col"), -999)))
        residuals.append(list(_corner_residual(corner)))
        previous_uv_by_id[cid] = (frame, uv_arr)

    return (
        np.asarray(obj, dtype=np.float64).reshape(-1, 3),
        np.asarray(img, dtype=np.float64).reshape(-1, 2),
        np.asarray(vel, dtype=np.float64).reshape(-1, 2),
        np.asarray(valid_vel, dtype=bool).reshape(-1),
        ids,
        np.asarray(rows, dtype=np.int64).reshape(-1),
        np.asarray(cols, dtype=np.int64).reshape(-1),
        np.asarray(residuals, dtype=np.float64).reshape(-1, 2),
    )


def build_observations(
    run: dict[str, Any],
    *,
    point_set: str,
    max_gap_frames: int,
    max_frames: int | None,
) -> tuple[list[FrameObservation], dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = run["frames"]
    details: dict[int, dict[str, Any]] = run["details"]
    sorted_frames = sorted(frames)
    if max_frames is not None and max_frames > 0:
        sorted_frames = sorted_frames[: int(max_frames)]

    original_tvecs: list[np.ndarray] = []
    for frame in sorted_frames:
        data = frames[frame]
        original_tvecs.append(
            np.asarray(
                [
                    _to_float(data.get("tvec_x_mm")),
                    _to_float(data.get("tvec_y_mm")),
                    _to_float(data.get("tvec_z_mm")),
                ],
                dtype=np.float64,
            )
        )
    original_tvecs_arr = np.asarray(original_tvecs, dtype=np.float64).reshape(-1, 3)
    valid_tvec = np.all(np.isfinite(original_tvecs_arr), axis=1)
    if not np.any(valid_tvec):
        raise RuntimeError("No finite tvec_x_mm/tvec_y_mm/tvec_z_mm values found.")

    origin_index = int(np.where(valid_tvec)[0][0])
    origin_tvec = original_tvecs_arr[origin_index].copy()
    relative_tvecs = original_tvecs_arr - origin_tvec
    ranges = np.nanmax(relative_tvecs[valid_tvec], axis=0) - np.nanmin(
        relative_tvecs[valid_tvec], axis=0
    )
    movement_axis_idx = int(np.nanargmax(ranges))
    movement_axis = COMPONENTS[movement_axis_idx]
    movement_values = relative_tvecs[:, movement_axis_idx]
    turn_idx = int(np.nanargmax(np.abs(movement_values)))
    turn_frame = sorted_frames[turn_idx]

    previous_uv_by_id: dict[str, tuple[int, np.ndarray]] = {}
    observations: list[FrameObservation] = []
    corners_key = "pose_corners" if point_set == "pose" else "correspondence_corners"

    for idx, frame in enumerate(sorted_frames):
        data = frames[frame]
        detail = details.get(frame, {})
        success = _to_int(data.get("success"), default=0)
        if success == 0:
            continue

        corners = list(detail.get(corners_key) or [])
        if not corners and point_set == "pose":
            corners = list(detail.get("correspondence_corners") or [])

        (
            obj,
            img,
            vel,
            valid_vel_arr,
            ids,
            rows,
            cols,
            residuals,
        ) = _corners_to_arrays(
            corners,
            frame=frame,
            previous_uv_by_id=previous_uv_by_id,
            max_gap_frames=max_gap_frames,
        )

        rvec = np.asarray(
            [
                _to_float(data.get("rvec_x_rad")),
                _to_float(data.get("rvec_y_rad")),
                _to_float(data.get("rvec_z_rad")),
            ],
            dtype=np.float64,
        )
        tvec = original_tvecs_arr[idx].copy()
        if len(obj) < 6 or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue

        observations.append(
            FrameObservation(
                frame=frame,
                branch="out" if frame <= turn_frame else "return",
                movement_value_mm=float(relative_tvecs[idx, movement_axis_idx]),
                original_rvec=rvec,
                original_tvec=tvec,
                rel_original_tvec=relative_tvecs[idx].copy(),
                object_points=obj,
                image_points=img,
                velocities_px_per_frame=vel,
                velocity_valid=valid_vel_arr,
                corner_ids=ids,
                rows=rows,
                cols=cols,
                residuals_px=residuals,
                point_count=int(len(obj)),
            )
        )

    if not observations:
        raise RuntimeError("No frames with usable pose corners found.")

    meta = {
        "origin_frame": sorted_frames[origin_index],
        "origin_tvec": origin_tvec,
        "movement_axis": movement_axis,
        "movement_axis_idx": movement_axis_idx,
        "turn_frame": int(turn_frame),
    }
    return observations, meta


def _solve_pose(
    observation: FrameObservation,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    beta_u: float,
    beta_v: float,
    refine: str,
) -> tuple[bool, np.ndarray, np.ndarray, float, float]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV/cv2 is required for offline PnP replay.") from exc

    image = observation.image_points.copy()
    mask = observation.velocity_valid & np.all(np.isfinite(observation.velocities_px_per_frame), axis=1)
    if np.any(mask):
        image[mask, 0] -= float(beta_u) * observation.velocities_px_per_frame[mask, 0]
        image[mask, 1] -= float(beta_v) * observation.velocities_px_per_frame[mask, 1]

    rvec = observation.original_rvec.reshape(3, 1).astype(np.float64)
    tvec = observation.original_tvec.reshape(3, 1).astype(np.float64)

    try:
        ok, solved_rvec, solved_tvec = cv2.solvePnP(
            observation.object_points.reshape(-1, 3),
            image.reshape(-1, 2),
            K.reshape(3, 3),
            dist.reshape(-1, 1),
            rvec=rvec.copy(),
            tvec=tvec.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except Exception:
        return False, rvec.reshape(3), tvec.reshape(3), math.nan, math.nan

    if not ok:
        return False, rvec.reshape(3), tvec.reshape(3), math.nan, math.nan

    solved_rvec = np.asarray(solved_rvec, dtype=np.float64).reshape(3, 1)
    solved_tvec = np.asarray(solved_tvec, dtype=np.float64).reshape(3, 1)

    refine = str(refine).strip().lower()
    try:
        if refine == "vvs" and hasattr(cv2, "solvePnPRefineVVS"):
            solved_rvec, solved_tvec = cv2.solvePnPRefineVVS(
                observation.object_points.reshape(-1, 3),
                image.reshape(-1, 2),
                K.reshape(3, 3),
                dist.reshape(-1, 1),
                solved_rvec,
                solved_tvec,
            )
        elif refine == "lm" and hasattr(cv2, "solvePnPRefineLM"):
            solved_rvec, solved_tvec = cv2.solvePnPRefineLM(
                observation.object_points.reshape(-1, 3),
                image.reshape(-1, 2),
                K.reshape(3, 3),
                dist.reshape(-1, 1),
                solved_rvec,
                solved_tvec,
            )
    except Exception:
        pass

    try:
        projected, _ = cv2.projectPoints(
            observation.object_points.reshape(-1, 3),
            solved_rvec.reshape(3, 1),
            solved_tvec.reshape(3, 1),
            K.reshape(3, 3),
            dist.reshape(-1, 1),
        )
        projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        errors = np.sqrt(np.sum((projected - image.reshape(-1, 2)) ** 2, axis=1))
        rms = float(math.sqrt(float(np.mean(errors * errors))))
        mean = float(np.mean(errors))
    except Exception:
        rms = math.nan
        mean = math.nan

    return True, solved_rvec.reshape(3), solved_tvec.reshape(3), rms, mean


def _model_branch_coeff(
    movement: np.ndarray,
    z: np.ndarray,
    branches: list[str],
) -> tuple[float, float, float]:
    movement = np.asarray(movement, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    branch_return = np.asarray([1.0 if branch == "return" else 0.0 for branch in branches], dtype=np.float64)
    mask = np.isfinite(movement) & np.isfinite(z)
    if int(np.sum(mask)) < 8 or len(set(np.asarray(branches, dtype=object)[mask])) < 2:
        return math.nan, math.nan, math.nan

    x = movement[mask]
    yy = z[mask]
    scale = float(np.std(x))
    x_norm = (x - float(np.mean(x))) / scale if scale > 0.0 else x * 0.0
    design = np.column_stack(
        [
            np.ones_like(x_norm),
            x_norm,
            x_norm * x_norm,
            branch_return[mask],
        ]
    )
    try:
        beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
    except np.linalg.LinAlgError:
        return math.nan, math.nan, math.nan
    pred = design @ beta
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan
    rmse = float(math.sqrt(float(np.mean((yy - pred) ** 2))))
    return float(beta[3]), r2, rmse


def _matched_bin_delta(
    movement: np.ndarray,
    z: np.ndarray,
    branches: list[str],
    *,
    bin_width_mm: float,
) -> tuple[float, list[dict[str, Any]]]:
    movement = np.asarray(movement, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    valid = np.isfinite(movement) & np.isfinite(z)
    if int(np.sum(valid)) == 0:
        return math.nan, []

    finite_movement = movement[valid]
    width = max(float(bin_width_mm), 1e-6)
    lo_all = math.floor(float(np.min(finite_movement)) / width) * width
    hi_all = math.ceil(float(np.max(finite_movement)) / width) * width
    bins = np.arange(lo_all, hi_all + width * 1.5, width, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    branch_arr = np.asarray(branches, dtype=object)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = valid & (movement >= lo) & (movement < hi)
        out_z = z[in_bin & (branch_arr == "out")]
        return_z = z[in_bin & (branch_arr == "return")]
        out_med = _median(out_z)
        return_med = _median(return_z)
        delta = return_med - out_med
        row = {
            "movement_bin_lo_mm": float(lo),
            "movement_bin_hi_mm": float(hi),
            "movement_bin_center_mm": float((lo + hi) * 0.5),
            "out_n": int(np.sum(in_bin & (branch_arr == "out"))),
            "return_n": int(np.sum(in_bin & (branch_arr == "return"))),
            "out_z_median_mm": out_med,
            "return_z_median_mm": return_med,
            "return_minus_out_z_median_mm": delta,
        }
        if row["out_n"] > 0 and row["return_n"] > 0 and np.isfinite(delta):
            deltas.append(delta)
            rows.append(row)
    return _median(deltas), rows


def _metric_objective(row: dict[str, Any]) -> float:
    branch = abs(_to_float(row.get("branch_coeff_mm")))
    ybin = abs(_to_float(row.get("matched_bin_return_minus_out_z_median_mm")))
    reproj = _to_float(row.get("reproj_rms_median_px"))
    if np.isfinite(branch) and np.isfinite(ybin):
        objective = branch + 0.35 * ybin
    elif np.isfinite(branch):
        objective = branch
    elif np.isfinite(ybin):
        objective = ybin
    else:
        objective = abs(_to_float(row.get("z_closure_mm")))
    if np.isfinite(reproj):
        objective += 0.03 * max(0.0, reproj - 0.45)
    return float(objective) if np.isfinite(objective) else math.inf


def evaluate_beta(
    observations: list[FrameObservation],
    K: np.ndarray,
    dist: np.ndarray,
    *,
    beta_u: float,
    beta_v: float,
    refine: str,
    bin_width_mm: float,
    keep_frames: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    solved_tvecs: list[np.ndarray] = []
    solved_flags: list[bool] = []
    rms_values: list[float] = []
    mean_values: list[float] = []
    frame_rows: list[dict[str, Any]] = []

    for obs in observations:
        ok, rvec, tvec, reproj_rms, reproj_mean = _solve_pose(
            obs,
            K,
            dist,
            beta_u=beta_u,
            beta_v=beta_v,
            refine=refine,
        )
        solved_tvecs.append(tvec)
        solved_flags.append(ok)
        rms_values.append(reproj_rms)
        mean_values.append(reproj_mean)

    tvec_arr = np.asarray(solved_tvecs, dtype=np.float64).reshape(-1, 3)
    solved = np.asarray(solved_flags, dtype=bool)
    finite_tvec = solved & np.all(np.isfinite(tvec_arr), axis=1)
    if np.any(finite_tvec):
        origin = tvec_arr[int(np.where(finite_tvec)[0][0])].copy()
        rel = tvec_arr - origin
    else:
        rel = np.full_like(tvec_arr, np.nan)

    movement = np.asarray([obs.movement_value_mm for obs in observations], dtype=np.float64)
    branches = [obs.branch for obs in observations]
    z = rel[:, 2]
    branch_coeff, branch_r2, branch_rmse = _model_branch_coeff(movement, z, branches)
    ybin_delta, ybin_rows = _matched_bin_delta(movement, z, branches, bin_width_mm=bin_width_mm)

    original_z = np.asarray([obs.rel_original_tvec[2] for obs in observations], dtype=np.float64)
    corrected_minus_original = z - original_z
    row = {
        "beta_u": float(beta_u),
        "beta_v": float(beta_v),
        "solved_frames": int(np.sum(finite_tvec)),
        "total_frames": int(len(observations)),
        "solved_pct": 100.0 * float(np.sum(finite_tvec)) / max(1, len(observations)),
        "branch_coeff_mm": branch_coeff,
        "branch_r2": branch_r2,
        "branch_rmse_mm": branch_rmse,
        "matched_bin_return_minus_out_z_median_mm": ybin_delta,
        "z_range_mm": float(np.nanmax(z[finite_tvec]) - np.nanmin(z[finite_tvec])) if np.any(finite_tvec) else math.nan,
        "z_closure_mm": float(z[np.where(finite_tvec)[0][-1]] - z[np.where(finite_tvec)[0][0]]) if np.any(finite_tvec) else math.nan,
        "reproj_rms_median_px": _median(rms_values),
        "reproj_mean_median_px": _median(mean_values),
        "z_correction_median_mm": _median(corrected_minus_original),
        "z_correction_p95_abs_mm": float(np.percentile(np.abs(_finite(corrected_minus_original)), 95))
        if len(_finite(corrected_minus_original))
        else math.nan,
    }
    row["objective"] = _metric_objective(row)

    if keep_frames:
        for idx, obs in enumerate(observations):
            speed = np.sqrt(np.sum(obs.velocities_px_per_frame * obs.velocities_px_per_frame, axis=1))
            speed_valid = speed[obs.velocity_valid & np.isfinite(speed)]
            frame_rows.append(
                {
                    "frame": int(obs.frame),
                    "branch": obs.branch,
                    "movement_axis_value_mm": float(obs.movement_value_mm),
                    "point_count": int(obs.point_count),
                    "velocity_valid_points": int(np.sum(obs.velocity_valid)),
                    "corner_speed_median_px_per_frame": _median(speed_valid),
                    "corner_speed_p95_px_per_frame": float(np.percentile(speed_valid, 95))
                    if len(speed_valid)
                    else math.nan,
                    "orig_rel_x_mm": float(obs.rel_original_tvec[0]),
                    "orig_rel_y_mm": float(obs.rel_original_tvec[1]),
                    "orig_rel_z_mm": float(obs.rel_original_tvec[2]),
                    "corr_rel_x_mm": float(rel[idx, 0]),
                    "corr_rel_y_mm": float(rel[idx, 1]),
                    "corr_rel_z_mm": float(rel[idx, 2]),
                    "corr_minus_orig_z_mm": float(rel[idx, 2] - obs.rel_original_tvec[2]),
                    "reproj_rms_px": rms_values[idx],
                    "reproj_mean_px": mean_values[idx],
                    "solved": int(bool(finite_tvec[idx])),
                }
            )

    return row, frame_rows, ybin_rows


def residual_fit_beta(observations: list[FrameObservation]) -> dict[str, float]:
    residual_components: list[np.ndarray] = []
    velocity_components: list[np.ndarray] = []
    residual_along_velocity: list[float] = []
    speed_values: list[float] = []

    for obs in observations:
        mask = (
            obs.velocity_valid
            & np.all(np.isfinite(obs.velocities_px_per_frame), axis=1)
            & np.all(np.isfinite(obs.residuals_px), axis=1)
        )
        if not np.any(mask):
            continue
        residual = obs.residuals_px[mask]
        velocity = obs.velocities_px_per_frame[mask]
        residual_components.append(residual)
        velocity_components.append(velocity)
        speed = np.sqrt(np.sum(velocity * velocity, axis=1))
        valid_speed = speed > 1e-9
        if np.any(valid_speed):
            res_along = np.sum(residual[valid_speed] * velocity[valid_speed], axis=1) / speed[valid_speed]
            residual_along_velocity.extend(res_along.tolist())
            speed_values.extend(speed[valid_speed].tolist())

    if not residual_components:
        return {
            "beta_scalar": math.nan,
            "beta_u": math.nan,
            "beta_v": math.nan,
            "residual_velocity_corr": math.nan,
            "corner_velocity_samples": 0.0,
        }

    residual = np.vstack(residual_components)
    velocity = np.vstack(velocity_components)
    denom_scalar = float(np.sum(velocity * velocity))
    beta_scalar = -float(np.sum(residual * velocity)) / denom_scalar if denom_scalar > 1e-12 else math.nan
    denom_u = float(np.sum(velocity[:, 0] * velocity[:, 0]))
    denom_v = float(np.sum(velocity[:, 1] * velocity[:, 1]))
    beta_u = -float(np.sum(residual[:, 0] * velocity[:, 0])) / denom_u if denom_u > 1e-12 else math.nan
    beta_v = -float(np.sum(residual[:, 1] * velocity[:, 1])) / denom_v if denom_v > 1e-12 else math.nan

    return {
        "beta_scalar": beta_scalar,
        "beta_u": beta_u,
        "beta_v": beta_v,
        "residual_velocity_corr": _corr(residual_along_velocity, speed_values),
        "corner_velocity_samples": float(len(speed_values)),
    }


def build_corner_rows(observations: list[FrameObservation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obs in observations:
        speed = np.sqrt(np.sum(obs.velocities_px_per_frame * obs.velocities_px_per_frame, axis=1))
        for idx, cid in enumerate(obs.corner_ids):
            if idx >= len(speed):
                continue
            residual = obs.residuals_px[idx]
            velocity = obs.velocities_px_per_frame[idx]
            valid_velocity = bool(obs.velocity_valid[idx] and np.all(np.isfinite(velocity)))
            speed_value = float(speed[idx]) if np.isfinite(speed[idx]) else math.nan
            residual_along = math.nan
            if valid_velocity and speed_value > 1e-9 and np.all(np.isfinite(residual)):
                residual_along = float(np.dot(residual, velocity) / speed_value)
            rows.append(
                {
                    "frame": int(obs.frame),
                    "branch": obs.branch,
                    "movement_axis_value_mm": float(obs.movement_value_mm),
                    "corner_id": cid,
                    "global_row": int(obs.rows[idx]) if int(obs.rows[idx]) != -999 else "",
                    "global_col": int(obs.cols[idx]) if int(obs.cols[idx]) != -999 else "",
                    "u_px": float(obs.image_points[idx, 0]),
                    "v_px": float(obs.image_points[idx, 1]),
                    "velocity_valid": int(valid_velocity),
                    "vu_px_per_frame": float(velocity[0]),
                    "vv_px_per_frame": float(velocity[1]),
                    "speed_px_per_frame": speed_value,
                    "residual_du_px": float(residual[0]),
                    "residual_dv_px": float(residual[1]),
                    "residual_along_velocity_px": residual_along,
                }
            )
    return rows


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
    observations: list[FrameObservation],
    frame_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
    ybin_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    *,
    beta_u: float,
    beta_v: float,
    show: bool,
) -> Path:
    import matplotlib

    matplotlib.use("Agg" if not show else "QtAgg")
    import matplotlib.pyplot as plt

    _setup_plot_style(plt)

    path: Path = run["path"]
    out_path = path.with_name(f"{path.stem}_motion_bias_model_plot.png")
    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=False)
    fig.suptitle("HydraTracker motion-dependent corner bias model", fontsize=16, fontweight="bold")
    fig.text(
        0.01,
        0.965,
        (
            f"{run['run_id']} -- {run.get('timestamp', '')}   "
            f"movement={meta['movement_axis']}   turn_frame={meta['turn_frame']}   "
            f"best beta=({beta_u:+.3f}, {beta_v:+.3f}) frame"
        ),
        fontsize=9,
        ha="left",
        va="top",
    )

    frames = np.asarray([row["frame"] for row in frame_rows], dtype=np.float64)
    orig_z = np.asarray([_to_float(row.get("orig_rel_z_mm")) for row in frame_rows], dtype=np.float64)
    corr_z = np.asarray([_to_float(row.get("corr_rel_z_mm")) for row in frame_rows], dtype=np.float64)
    movement = np.asarray([_to_float(row.get("movement_axis_value_mm")) for row in frame_rows], dtype=np.float64)
    speed = np.asarray([_to_float(row.get("corner_speed_median_px_per_frame")) for row in frame_rows], dtype=np.float64)
    point_count = np.asarray([_to_float(row.get("point_count")) for row in frame_rows], dtype=np.float64)

    axes[0].plot(frames, orig_z, color="#d62728", marker="o", markersize=2.5, linewidth=1.5, label="original z")
    axes[0].plot(frames, corr_z, color="#1f77b4", marker="o", markersize=2.5, linewidth=1.5, label="corrected z")
    axes[0].axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[0].set_title("Original versus corrected camera-frame z")
    axes[0].set_ylabel("delta z [mm]")
    axes[0].legend(loc="best")

    branches = np.asarray([obs.branch for obs in observations], dtype=object)
    out_mask = branches == "out"
    ret_mask = branches == "return"
    axes[1].scatter(movement[out_mask], orig_z[out_mask], s=16, color="#d62728", alpha=0.55, label="orig out")
    axes[1].scatter(movement[ret_mask], orig_z[ret_mask], s=16, color="#ff9896", alpha=0.55, label="orig return")
    axes[1].scatter(movement[out_mask], corr_z[out_mask], s=18, color="#1f77b4", alpha=0.75, label="corr out")
    axes[1].scatter(movement[ret_mask], corr_z[ret_mask], s=18, color="#9ecae1", alpha=0.75, label="corr return")
    axes[1].set_title("Hysteresis at same movement coordinate")
    axes[1].set_xlabel(f"delta {meta['movement_axis']} [mm]")
    axes[1].set_ylabel("delta z [mm]")
    axes[1].legend(loc="best", ncols=2, fontsize=8)

    betas = np.asarray([_to_float(row.get("beta_u")) for row in sweep_rows], dtype=np.float64)
    branch_coeffs = np.asarray([_to_float(row.get("branch_coeff_mm")) for row in sweep_rows], dtype=np.float64)
    ybin = np.asarray(
        [_to_float(row.get("matched_bin_return_minus_out_z_median_mm")) for row in sweep_rows],
        dtype=np.float64,
    )
    objective = np.asarray([_to_float(row.get("objective")) for row in sweep_rows], dtype=np.float64)
    axes[2].plot(betas, branch_coeffs, color="#9467bd", linewidth=1.8, label="branch coefficient")
    axes[2].plot(betas, ybin, color="#8c564b", linewidth=1.6, label="matched-bin return-out")
    axes[2].plot(betas, objective, color="#2ca02c", linewidth=1.2, label="objective")
    axes[2].axvline(beta_u, color="#1f77b4", linestyle="--", linewidth=1.0, label="chosen beta")
    axes[2].axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[2].set_title("Global beta sweep")
    axes[2].set_xlabel("beta [frames]")
    axes[2].set_ylabel("mm / score")
    axes[2].legend(loc="best", fontsize=8)

    if ybin_rows:
        centers = np.asarray([_to_float(row.get("movement_bin_center_mm")) for row in ybin_rows], dtype=np.float64)
        deltas = np.asarray([_to_float(row.get("return_minus_out_z_median_mm")) for row in ybin_rows], dtype=np.float64)
        width = float(_median(np.diff(centers))) * 0.75 if len(centers) > 1 else 7.5
        axes[3].bar(centers, deltas, width=width, color="#8c564b", alpha=0.85)
    axes[3].axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[3].set_title("Corrected return minus out z by matched movement bins")
    axes[3].set_xlabel(f"delta {meta['movement_axis']} bin center [mm]")
    axes[3].set_ylabel("return - out z [mm]")

    axes[4].plot(frames, point_count, color="#4c78a8", linewidth=1.5, label="pose points")
    ax_speed = axes[4].twinx()
    ax_speed.plot(frames, speed, color="#f58518", linewidth=1.4, label="median corner speed")
    axes[4].set_title("Point count and image motion used by the model")
    axes[4].set_xlabel("frame")
    axes[4].set_ylabel("points")
    ax_speed.set_ylabel("px/frame")
    handles1, labels1 = axes[4].get_legend_handles_labels()
    handles2, labels2 = ax_speed.get_legend_handles_labels()
    axes[4].legend(handles1 + handles2, labels1 + labels2, loc="best")

    for ax in axes:
        ax.grid(True, alpha=0.85)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
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
        "max_gap_frames": 2,
        "max_frames": None,
        "bin_width_mm": 10.0,
        "beta_min": -2.0,
        "beta_max": 2.0,
        "beta_steps": 41,
        "refine": "vvs",
        "corner_csv": False,
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
        elif arg == "--corner-csv":
            args["corner_csv"] = True
        elif arg == "--point-set":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--point-set needs 'pose' or 'correspondence'")
            point_set = str(argv[idx]).strip().lower()
            if point_set not in ("pose", "correspondence"):
                raise RuntimeError("--point-set must be 'pose' or 'correspondence'")
            args["point_set"] = point_set
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
        elif arg == "--bin-mm":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--bin-mm needs a numeric value")
            args["bin_width_mm"] = float(argv[idx])
        elif arg == "--beta-min":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--beta-min needs a numeric value")
            args["beta_min"] = float(argv[idx])
        elif arg == "--beta-max":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--beta-max needs a numeric value")
            args["beta_max"] = float(argv[idx])
        elif arg == "--beta-steps":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--beta-steps needs an integer")
            args["beta_steps"] = int(argv[idx])
        elif arg == "--refine":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--refine needs one of: none, lm, vvs")
            refine = str(argv[idx]).strip().lower()
            if refine not in ("none", "lm", "vvs"):
                raise RuntimeError("--refine must be one of: none, lm, vvs")
            args["refine"] = refine
        elif arg.endswith(".jsonl"):
            args["path"] = Path(arg)
        else:
            raise RuntimeError(f"Unknown option: {arg}")
        idx += 1
    return args


def _print_summary(
    *,
    path: Path,
    residual_fit: dict[str, float],
    zero_row: dict[str, Any],
    best_row: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    print(f"[res_motion_bias_model] saved beta sweep csv -> {paths['sweep'].resolve()}")
    print(f"[res_motion_bias_model] saved frame csv      -> {paths['frame'].resolve()}")
    if "corner" in paths:
        print(f"[res_motion_bias_model] saved corner csv     -> {paths['corner'].resolve()}")
    if "ybin" in paths:
        print(f"[res_motion_bias_model] saved y-bin csv      -> {paths['ybin'].resolve()}")
    if "plot" in paths:
        print(f"[res_motion_bias_model] saved plot           -> {paths['plot'].resolve()}")

    print(
        "[res_motion_bias_model] residual-fit beta "
        f"scalar={residual_fit['beta_scalar']:+.3f}, "
        f"u={residual_fit['beta_u']:+.3f}, "
        f"v={residual_fit['beta_v']:+.3f}, "
        f"residual-speed corr={residual_fit['residual_velocity_corr']:+.3f}, "
        f"samples={int(residual_fit['corner_velocity_samples'])}"
    )
    print(
        "[res_motion_bias_model] beta=0 replay "
        f"branch_coeff={_to_float(zero_row.get('branch_coeff_mm')):+.3f} mm, "
        f"matched_bin={_to_float(zero_row.get('matched_bin_return_minus_out_z_median_mm')):+.3f} mm, "
        f"z_range={_to_float(zero_row.get('z_range_mm')):.3f} mm, "
        f"rms={_to_float(zero_row.get('reproj_rms_median_px')):.3f} px"
    )
    print(
        "[res_motion_bias_model] best global beta "
        f"u={_to_float(best_row.get('beta_u')):+.3f}, "
        f"v={_to_float(best_row.get('beta_v')):+.3f}, "
        f"branch_coeff={_to_float(best_row.get('branch_coeff_mm')):+.3f} mm, "
        f"matched_bin={_to_float(best_row.get('matched_bin_return_minus_out_z_median_mm')):+.3f} mm, "
        f"z_range={_to_float(best_row.get('z_range_mm')):.3f} mm, "
        f"rms={_to_float(best_row.get('reproj_rms_median_px')):.3f} px"
    )
    print(f"[res_motion_bias_model] input -> {path.resolve()}")


def main() -> None:
    args = _parse_args(sys.argv[1:])
    path = args["path"]
    if path is None:
        path = _latest_run_path() if args["use_latest"] else _select_run_path_qt()
    path = Path(path).resolve()
    run = load_run(path)

    observations, meta = build_observations(
        run,
        point_set=str(args["point_set"]),
        max_gap_frames=int(args["max_gap_frames"]),
        max_frames=args["max_frames"],
    )
    residual_fit = residual_fit_beta(observations)

    beta_min = float(args["beta_min"])
    beta_max = float(args["beta_max"])
    beta_steps = max(3, int(args["beta_steps"]))
    sweep_betas = np.linspace(beta_min, beta_max, beta_steps, dtype=np.float64)
    extra = [0.0, residual_fit["beta_scalar"]]
    sweep_betas = np.asarray(
        sorted({round(float(v), 10) for v in list(sweep_betas) + extra if np.isfinite(v)}),
        dtype=np.float64,
    )

    sweep_rows: list[dict[str, Any]] = []
    zero_row: dict[str, Any] | None = None
    for beta in sweep_betas:
        row, _frame_rows, _ybin_rows = evaluate_beta(
            observations,
            run["K"],
            run["dist"],
            beta_u=float(beta),
            beta_v=float(beta),
            refine=str(args["refine"]),
            bin_width_mm=float(args["bin_width_mm"]),
            keep_frames=False,
        )
        sweep_rows.append(row)
        if abs(float(beta)) < 1e-12:
            zero_row = row

    if zero_row is None:
        zero_row, _frame_rows, _ybin_rows = evaluate_beta(
            observations,
            run["K"],
            run["dist"],
            beta_u=0.0,
            beta_v=0.0,
            refine=str(args["refine"]),
            bin_width_mm=float(args["bin_width_mm"]),
            keep_frames=False,
        )
        sweep_rows.append(zero_row)

    best_row = min(sweep_rows, key=_metric_objective)
    best_beta = float(best_row["beta_u"])
    best_row, frame_rows, ybin_rows = evaluate_beta(
        observations,
        run["K"],
        run["dist"],
        beta_u=best_beta,
        beta_v=best_beta,
        refine=str(args["refine"]),
        bin_width_mm=float(args["bin_width_mm"]),
        keep_frames=True,
    )

    sweep_csv = path.with_name(f"{path.stem}_motion_bias_model_beta_sweep.csv")
    frame_csv = path.with_name(f"{path.stem}_motion_bias_model_frames.csv")
    ybin_csv = path.with_name(f"{path.stem}_motion_bias_model_ybins.csv")
    _write_csv(sweep_csv, sweep_rows)
    _write_csv(frame_csv, frame_rows)
    _write_csv(ybin_csv, ybin_rows)

    paths = {"sweep": sweep_csv, "frame": frame_csv, "ybin": ybin_csv}
    if bool(args["corner_csv"]):
        corner_csv = path.with_name(f"{path.stem}_motion_bias_model_corners.csv")
        _write_csv(corner_csv, build_corner_rows(observations))
        paths["corner"] = corner_csv

    if bool(args["make_plot"]):
        paths["plot"] = plot_results(
            run,
            observations,
            frame_rows,
            sweep_rows,
            ybin_rows,
            meta,
            beta_u=best_beta,
            beta_v=best_beta,
            show=bool(args["show"]),
        )

    _print_summary(
        path=path,
        residual_fit=residual_fit,
        zero_row=zero_row,
        best_row=best_row,
        paths=paths,
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[res_motion_bias_model] ERROR: {exc}")
        sys.exit(1)
