from __future__ import annotations

import csv
import json
import math
import sys
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


def _ensure_src_on_path() -> None:
    tracking_root = Path(__file__).resolve().parents[2]
    src_root = tracking_root.parent
    src = str(src_root)
    sys.path = [p for p in sys.path if str(p) != src]
    sys.path.insert(0, src)


_ensure_src_on_path()


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_corr(cov: np.ndarray, i: int, j: int) -> float:
    try:
        denom = float(math.sqrt(max(cov[i, i], 0.0) * max(cov[j, j], 0.0)))
        if denom <= 0.0:
            return math.nan
        return float(cov[i, j] / denom)
    except Exception:
        return math.nan


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


def _corner_arrays(corners: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    object_points: list[list[float]] = []
    image_points: list[list[float]] = []
    rows: list[int] = []
    cols: list[int] = []

    for corner in corners:
        xyz = corner.get("xyz_mm")
        uv = corner.get("uv_px")
        if xyz is None or uv is None:
            continue
        try:
            xyz_f = [float(v) for v in xyz]
            uv_f = [float(v) for v in uv]
        except (TypeError, ValueError):
            continue
        if len(xyz_f) != 3 or len(uv_f) != 2:
            continue
        if not np.all(np.isfinite(xyz_f + uv_f)):
            continue

        object_points.append(xyz_f)
        image_points.append(uv_f)
        rows.append(_to_int(corner.get("global_row"), default=_to_int(corner.get("local_row"))))
        cols.append(_to_int(corner.get("global_col"), default=_to_int(corner.get("local_col"))))

    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(obj) == 0:
        return obj, img, {}

    stats = {
        "point_count": float(len(obj)),
        "row_min": float(min(rows)) if rows else math.nan,
        "row_max": float(max(rows)) if rows else math.nan,
        "col_min": float(min(cols)) if cols else math.nan,
        "col_max": float(max(cols)) if cols else math.nan,
        "distinct_rows": float(len(set(rows))) if rows else math.nan,
        "distinct_cols": float(len(set(cols))) if cols else math.nan,
        "object_x_span_mm": float(np.ptp(obj[:, 0])),
        "object_y_span_mm": float(np.ptp(obj[:, 1])),
        "object_z_span_mm": float(np.ptp(obj[:, 2])),
        "image_u_span_px": float(np.ptp(img[:, 0])),
        "image_v_span_px": float(np.ptp(img[:, 1])),
        "image_u_centroid_px": float(np.mean(img[:, 0])),
        "image_v_centroid_px": float(np.mean(img[:, 1])),
    }
    stats["row_span"] = stats["row_max"] - stats["row_min"]
    stats["col_span"] = stats["col_max"] - stats["col_min"]
    return obj, img, stats


def _empty_uncertainty(reason: str) -> dict[str, float | int | str]:
    return {
        "cov_valid": 0,
        "uncertainty_reason": reason,
        "jacobian_rank": 0,
        "jacobian_condition": math.nan,
        "normal_condition": math.nan,
        "pixel_sigma_fit_px": math.nan,
        "reproj_rms_px": math.nan,
        "reproj_mean_px": math.nan,
        "reproj_median_px": math.nan,
        "reproj_p95_px": math.nan,
        "reproj_max_px": math.nan,
        "reproj_mean_du_px": math.nan,
        "reproj_mean_dv_px": math.nan,
        "sigma_rx_deg": math.nan,
        "sigma_ry_deg": math.nan,
        "sigma_rz_deg": math.nan,
        "sigma_x_mm": math.nan,
        "sigma_y_mm": math.nan,
        "sigma_z_mm": math.nan,
        "sigma_x_fit_mm": math.nan,
        "sigma_y_fit_mm": math.nan,
        "sigma_z_fit_mm": math.nan,
        "sigma_z_over_xy": math.nan,
        "corr_z_rx": math.nan,
        "corr_z_ry": math.nan,
        "corr_z_rz": math.nan,
        "corr_z_x": math.nan,
        "corr_z_y": math.nan,
        "max_abs_z_rotation_corr": math.nan,
    }


def _pose_uncertainty(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    assumed_sigma_px: float,
) -> dict[str, float | int | str]:
    if len(object_points) < 6:
        return _empty_uncertainty("need_at_least_6_points")

    try:
        import cv2

        projected, jacobian = cv2.projectPoints(
            object_points.reshape(-1, 3),
            rvec.reshape(3, 1),
            tvec.reshape(3, 1),
            K.reshape(3, 3),
            dist.reshape(-1, 1),
        )
    except Exception as exc:
        return _empty_uncertainty(f"project_points_failed:{exc}")

    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    residual = projected - image_points.reshape(-1, 2)
    errors = np.sqrt(np.sum(residual * residual, axis=1))
    if not np.all(np.isfinite(errors)):
        return _empty_uncertainty("nonfinite_reprojection_error")

    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.shape[0] != 2 * len(object_points) or jacobian.shape[1] < 6:
        return _empty_uncertainty("bad_jacobian_shape")

    J_pose = jacobian[:, :6]
    if not np.all(np.isfinite(J_pose)):
        return _empty_uncertainty("nonfinite_jacobian")

    try:
        singular_values = np.linalg.svd(J_pose, compute_uv=False)
        normal = J_pose.T @ J_pose
    except np.linalg.LinAlgError:
        return _empty_uncertainty("jacobian_svd_failed")

    if not np.all(np.isfinite(singular_values)):
        return _empty_uncertainty("nonfinite_jacobian_svd")
    if not np.all(np.isfinite(normal)):
        return _empty_uncertainty("nonfinite_normal_matrix")

    max_s = float(np.max(singular_values)) if len(singular_values) else math.nan
    tol_s = (
        max(J_pose.shape) * np.finfo(float).eps * max_s
        if np.isfinite(max_s)
        else math.nan
    )
    positive_s = singular_values[singular_values > tol_s] if np.isfinite(tol_s) else np.empty(0)
    rank = int(len(positive_s))
    min_s = float(np.min(positive_s)) if len(positive_s) else math.nan
    jac_cond = (
        float(max_s / min_s)
        if np.isfinite(max_s) and np.isfinite(min_s) and min_s > 0.0
        else math.inf
    )
    normal_condition = jac_cond * jac_cond if np.isfinite(jac_cond) else math.inf

    dof = max(1, int(2 * len(object_points) - 6))
    pixel_sigma_fit = float(math.sqrt(float(np.sum(residual * residual)) / dof))

    try:
        normal_inv = np.linalg.pinv(
            normal,
            rcond=max(J_pose.shape) * np.finfo(float).eps,
            hermitian=True,
        )
    except np.linalg.LinAlgError:
        return _empty_uncertainty("normal_pinv_failed")

    cov_valid = int(rank == 6)
    reason = "ok" if cov_valid else "rank_deficient_pinv"

    cov_assumed = normal_inv * float(assumed_sigma_px) ** 2
    cov_fit = normal_inv * pixel_sigma_fit ** 2

    diag_assumed = np.diag(cov_assumed).astype(np.float64)
    diag_fit = np.diag(cov_fit).astype(np.float64)
    diag_assumed = np.where(diag_assumed >= 0.0, diag_assumed, np.nan)
    diag_fit = np.where(diag_fit >= 0.0, diag_fit, np.nan)

    sigma_assumed = np.sqrt(diag_assumed)
    sigma_fit = np.sqrt(diag_fit)
    sigma_xy = float(np.nanmean(sigma_assumed[3:5]))
    sigma_z_over_xy = (
        float(sigma_assumed[5] / sigma_xy)
        if np.isfinite(sigma_xy) and sigma_xy > 0.0
        else math.nan
    )

    corr_z_rx = _safe_corr(cov_assumed, 5, 0)
    corr_z_ry = _safe_corr(cov_assumed, 5, 1)
    corr_z_rz = _safe_corr(cov_assumed, 5, 2)
    rotation_corrs = np.asarray([corr_z_rx, corr_z_ry, corr_z_rz], dtype=np.float64)
    max_abs_z_rotation_corr = (
        float(np.nanmax(np.abs(rotation_corrs)))
        if np.any(np.isfinite(rotation_corrs))
        else math.nan
    )

    return {
        "cov_valid": cov_valid,
        "uncertainty_reason": reason,
        "jacobian_rank": rank,
        "jacobian_condition": jac_cond,
        "normal_condition": normal_condition,
        "pixel_sigma_fit_px": pixel_sigma_fit,
        "reproj_rms_px": pixel_sigma_fit,
        "reproj_mean_px": float(np.mean(errors)),
        "reproj_median_px": float(np.median(errors)),
        "reproj_p95_px": float(np.percentile(errors, 95)),
        "reproj_max_px": float(np.max(errors)),
        "reproj_mean_du_px": float(np.mean(residual[:, 0])),
        "reproj_mean_dv_px": float(np.mean(residual[:, 1])),
        "sigma_rx_deg": float(np.degrees(sigma_assumed[0])),
        "sigma_ry_deg": float(np.degrees(sigma_assumed[1])),
        "sigma_rz_deg": float(np.degrees(sigma_assumed[2])),
        "sigma_x_mm": float(sigma_assumed[3]),
        "sigma_y_mm": float(sigma_assumed[4]),
        "sigma_z_mm": float(sigma_assumed[5]),
        "sigma_x_fit_mm": float(sigma_fit[3]),
        "sigma_y_fit_mm": float(sigma_fit[4]),
        "sigma_z_fit_mm": float(sigma_fit[5]),
        "sigma_z_over_xy": sigma_z_over_xy,
        "corr_z_rx": corr_z_rx,
        "corr_z_ry": corr_z_ry,
        "corr_z_rz": corr_z_rz,
        "corr_z_x": _safe_corr(cov_assumed, 5, 3),
        "corr_z_y": _safe_corr(cov_assumed, 5, 4),
        "max_abs_z_rotation_corr": max_abs_z_rotation_corr,
    }


def load_run(path: Path) -> dict[str, Any]:
    K: np.ndarray | None = None
    dist: np.ndarray | None = None
    run_id = path.stem
    timestamp = ""
    frames: dict[int, dict[str, Any]] = {}
    details: dict[int, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}

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
                continue

            if record_type == "run_summary":
                summaries = dict(record.get("summary") or {})
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
        "summary": summaries,
    }


def analyze_run(
    run: dict[str, Any],
    *,
    assumed_sigma_px: float = 0.35,
    point_set: str = "pose",
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = run["frames"]
    details: dict[int, dict[str, Any]] = run["details"]
    K: np.ndarray = run["K"]
    dist: np.ndarray = run["dist"]

    rows: list[dict[str, Any]] = []
    sorted_frames = sorted(frames)
    if max_frames is not None and max_frames > 0:
        sorted_frames = sorted_frames[: int(max_frames)]

    absolute_tvecs = []
    for frame in sorted_frames:
        data = frames[frame]
        tvec = np.asarray(
            [
                _to_float(data.get("tvec_x_mm")),
                _to_float(data.get("tvec_y_mm")),
                _to_float(data.get("tvec_z_mm")),
            ],
            dtype=np.float64,
        )
        absolute_tvecs.append(tvec)
    absolute_tvecs_arr = np.asarray(absolute_tvecs, dtype=np.float64).reshape(-1, 3)
    valid_tvec = np.all(np.isfinite(absolute_tvecs_arr), axis=1)
    if not np.any(valid_tvec):
        raise RuntimeError("No finite tvec_x_mm/tvec_y_mm/tvec_z_mm values found.")
    origin_index = int(np.where(valid_tvec)[0][0])
    origin_tvec = absolute_tvecs_arr[origin_index].copy()
    relative_tvecs = absolute_tvecs_arr - origin_tvec

    ranges = np.nanmax(relative_tvecs[valid_tvec], axis=0) - np.nanmin(
        relative_tvecs[valid_tvec], axis=0
    )
    movement_axis_idx = int(np.nanargmax(ranges))
    movement_axis = COMPONENTS[movement_axis_idx]
    movement_values = relative_tvecs[:, movement_axis_idx]
    finite_movement = np.isfinite(movement_values)
    if np.any(finite_movement):
        farthest_idx = int(np.nanargmax(np.abs(movement_values)))
        turn_frame = sorted_frames[farthest_idx]
    else:
        turn_frame = sorted_frames[-1]

    prev_tvec = None
    prev_prev_tvec = None
    prev_sigma_z = math.nan
    prev_frame = None

    for idx, frame in enumerate(sorted_frames):
        data = frames[frame]
        detail = details.get(frame, {})
        success = _to_int(data.get("success"), default=0)
        pose_source = str(data.get("pose_source") or "")
        pnp_method = str(data.get("pnp_method") or "")
        best_method = str(data.get("pose_candidate_best_method") or "")

        rvec = np.asarray(
            [
                _to_float(data.get("rvec_x_rad")),
                _to_float(data.get("rvec_y_rad")),
                _to_float(data.get("rvec_z_rad")),
            ],
            dtype=np.float64,
        )
        tvec = absolute_tvecs_arr[idx].astype(np.float64)
        rel = relative_tvecs[idx].astype(np.float64)

        corners_key = "pose_corners" if point_set == "pose" else "correspondence_corners"
        corners = list(detail.get(corners_key) or [])
        if not corners and point_set == "pose":
            corners = list(detail.get("correspondence_corners") or [])
        obj, img, corner_stats = _corner_arrays(corners)

        if success and np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec)):
            uncertainty = _pose_uncertainty(
                obj,
                img,
                rvec,
                tvec,
                K,
                dist,
                assumed_sigma_px,
            )
        else:
            uncertainty = _empty_uncertainty("no_accepted_pose")

        sigma_z = _to_float(uncertainty.get("sigma_z_mm"))
        sigma_z_fit = _to_float(uncertainty.get("sigma_z_fit_mm"))
        z_step = math.nan
        z_step_norm = math.nan
        cv_innovation_z = math.nan
        cv_innovation_norm = math.nan
        cv_innovation_z_norm = math.nan

        if prev_tvec is not None and np.all(np.isfinite(tvec)) and np.all(np.isfinite(prev_tvec)):
            z_step = float(tvec[2] - prev_tvec[2])
            if np.isfinite(sigma_z) and np.isfinite(prev_sigma_z):
                denom = math.sqrt(sigma_z * sigma_z + prev_sigma_z * prev_sigma_z)
                if denom > 0.0:
                    z_step_norm = abs(z_step) / denom

        if (
            prev_tvec is not None
            and prev_prev_tvec is not None
            and np.all(np.isfinite(tvec))
            and np.all(np.isfinite(prev_tvec))
            and np.all(np.isfinite(prev_prev_tvec))
            and prev_frame is not None
        ):
            pred = prev_tvec + (prev_tvec - prev_prev_tvec)
            innovation = tvec - pred
            cv_innovation_z = float(innovation[2])
            cv_innovation_norm = float(math.sqrt(float(np.sum(innovation * innovation))))
            if np.isfinite(sigma_z):
                denom = max(float(sigma_z), 1e-12)
                cv_innovation_z_norm = abs(cv_innovation_z) / denom

        branch = "out" if frame <= turn_frame else "return"
        row: dict[str, Any] = {
            "frame": frame,
            "branch": branch,
            "movement_axis": movement_axis,
            "movement_axis_value_mm": float(rel[movement_axis_idx]),
            "turn_frame": int(turn_frame),
            "success": success,
            "pose_source": pose_source,
            "pnp_method": pnp_method,
            "pose_candidate_best_method": best_method,
            "tvec_x_mm": float(tvec[0]),
            "tvec_y_mm": float(tvec[1]),
            "tvec_z_mm": float(tvec[2]),
            "rel_x_mm": float(rel[0]),
            "rel_y_mm": float(rel[1]),
            "rel_z_mm": float(rel[2]),
            "rvec_x_rad": float(rvec[0]),
            "rvec_y_rad": float(rvec[1]),
            "rvec_z_rad": float(rvec[2]),
            "camera_roll_deg": _to_float(data.get("camera_roll_deg")),
            "camera_pitch_deg": _to_float(data.get("camera_pitch_deg")),
            "camera_yaw_deg": _to_float(data.get("camera_yaw_deg")),
            "num_points_logged": _to_float(data.get("num_points")),
            "point_count_used": float(len(obj)),
            "assumed_sigma_px": float(assumed_sigma_px),
            "z_step_mm": z_step,
            "z_step_over_combined_sigma": z_step_norm,
            "cv_innovation_z_mm": cv_innovation_z,
            "cv_innovation_norm_mm": cv_innovation_norm,
            "cv_innovation_z_over_sigma": cv_innovation_z_norm,
        }
        row.update(corner_stats)
        row.update(uncertainty)
        rows.append(row)

        if np.all(np.isfinite(tvec)):
            prev_prev_tvec = None if prev_tvec is None else prev_tvec.copy()
            prev_tvec = tvec.copy()
            prev_frame = frame
            prev_sigma_z = sigma_z

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
        for row in rows:
            writer.writerow(row)


def _finite_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
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


def plot_uncertainty(run: dict[str, Any], rows: list[dict[str, Any]], show: bool = False) -> Path:
    path: Path = run["path"]
    frames = _finite_array(rows, "frame")
    rel_z = _finite_array(rows, "rel_z_mm")
    sigma_x = _finite_array(rows, "sigma_x_mm")
    sigma_y = _finite_array(rows, "sigma_y_mm")
    sigma_z = _finite_array(rows, "sigma_z_mm")
    sigma_z_fit = _finite_array(rows, "sigma_z_fit_mm")
    cond = _finite_array(rows, "jacobian_condition")
    reproj_rms = _finite_array(rows, "reproj_rms_px")
    reproj_p95 = _finite_array(rows, "reproj_p95_px")
    z_step_norm = _finite_array(rows, "z_step_over_combined_sigma")
    innovation_z_norm = _finite_array(rows, "cv_innovation_z_over_sigma")
    z_rot_corr = _finite_array(rows, "max_abs_z_rotation_corr")

    movement_axis = str(rows[0].get("movement_axis", "?")) if rows else "?"
    turn_frame = _to_int(rows[0].get("turn_frame"), default=-1) if rows else -1

    run_label = str(run["run_id"])
    if run.get("timestamp"):
        run_label += f"  |  {run['timestamp']}"

    out_path = path.with_name(f"{path.stem}_pose_uncertainty_plot.png")
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_plot_style(plt)

    cond_log = np.full_like(cond, np.nan, dtype=np.float64)
    cond_mask = np.isfinite(cond) & (cond > 0.0)
    cond_log[cond_mask] = np.log10(cond[cond_mask])

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(18, 13),
        sharex=True,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.055, top=0.88, hspace=0.58)
    fig.suptitle(
        "HydraTracker pose uncertainty diagnostics",
        fontsize=16,
        fontweight="bold",
        y=0.988,
    )
    fig.text(0.5, 0.958, run_label, ha="center", va="top", fontsize=11)

    axes[0].plot(frames, rel_z, color="#d62728", marker="o", markersize=2.5, linewidth=1.4, label="relative z")
    axes[0].axhline(0.0, color="#e6b8b8", linewidth=1.0, linestyle="--")
    axes[0].set_title(f"Observed z drift and movement axis ({movement_axis})")
    axes[0].set_ylabel("delta z [mm]")
    axes[0].legend(loc="upper right")

    axes[1].plot(frames, sigma_x, color="#1f77b4", linewidth=1.2, label="sigma x")
    axes[1].plot(frames, sigma_y, color="#2ca02c", linewidth=1.2, label="sigma y")
    axes[1].plot(frames, sigma_z, color="#d62728", linewidth=1.5, label="sigma z")
    axes[1].plot(frames, sigma_z_fit, color="#ff9896", linewidth=1.1, label="sigma z fit")
    axes[1].set_title("Pose covariance from reprojection Jacobian")
    axes[1].set_ylabel("sigma [mm]")
    axes[1].legend(loc="upper right")

    axes[2].plot(frames, cond_log, color="#7f3c8d", linewidth=1.5, label="log10 condition")
    axes[2].set_title("PnP conditioning")
    axes[2].set_ylabel("log10(cond)")
    axes[2].legend(loc="upper right")

    axes[3].plot(frames, reproj_rms, color="#f58518", linewidth=1.4, label="RMS")
    axes[3].plot(frames, reproj_p95, color="#e45756", linewidth=1.1, label="p95")
    axes[3].set_title("Reprojection residuals")
    axes[3].set_ylabel("px")
    axes[3].legend(loc="upper right")

    axes[4].plot(frames, z_step_norm, color="#d62728", linewidth=1.1, label="|dz|/sigma")
    axes[4].plot(frames, innovation_z_norm, color="#7f3c8d", linewidth=1.1, label="|CV innov z|/sigma")
    axes[4].plot(frames, z_rot_corr, color="#4c78a8", linewidth=1.1, label="max |corr(z,rot)|")
    axes[4].axhline(1.0, color="#b8cde6", linewidth=1.0, linestyle="--")
    axes[4].axhline(3.0, color="#e6b8b8", linewidth=1.0, linestyle="--")
    axes[4].set_title("Z jumps versus estimated uncertainty")
    axes[4].set_ylabel("normalized")
    axes[4].set_xlabel("frame")
    axes[4].legend(loc="upper right")

    for ax in axes:
        if turn_frame >= 0:
            ax.axvline(turn_frame, color="#cfcfcf", linewidth=1.0)
        ax.grid(True, alpha=0.8)

    valid_rows = [row for row in rows if _to_int(row.get("success")) == 1]
    if valid_rows:
        last = valid_rows[-1]
        point_counts = _finite_array(rows, "point_count_used")
        finite_point_counts = point_counts[np.isfinite(point_counts)]
        point_count_label = (
            f"{float(np.nanmin(finite_point_counts)):.0f}-{float(np.nanmax(finite_point_counts)):.0f}"
            if len(finite_point_counts)
            else "n/a"
        )
        closure = (
            _to_float(last.get("rel_x_mm")),
            _to_float(last.get("rel_y_mm")),
            _to_float(last.get("rel_z_mm")),
        )
        info = (
            f"frames={len(rows)}   "
            f"point_count={point_count_label}   "
            f"closure=({closure[0]:+.2f}, {closure[1]:+.2f}, {closure[2]:+.2f}) mm   "
            f"assumed_sigma_px={_to_float(rows[0].get('assumed_sigma_px')):.3f}"
        )
        fig.text(0.01, 0.925, info, ha="left", va="top", fontsize=10)

    fig.savefig(out_path, dpi=160)
    print(f"[res_pose_uncertainty] saved plot -> {out_path.resolve()}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _print_summary(rows: list[dict[str, Any]], csv_path: Path, plot_path: Path | None) -> None:
    valid = [row for row in rows if _to_int(row.get("success")) == 1 and np.isfinite(_to_float(row.get("rel_z_mm")))]
    if not valid:
        print("[res_pose_uncertainty] no valid pose rows")
        return

    first = valid[0]
    last = valid[-1]
    rel_z = _finite_array(valid, "rel_z_mm")
    sigma_z = _finite_array(valid, "sigma_z_mm")
    sigma_z_fit = _finite_array(valid, "sigma_z_fit_mm")
    cond = _finite_array(valid, "jacobian_condition")
    z_step_norm = _finite_array(valid, "z_step_over_combined_sigma")
    branch = np.asarray([str(row.get("branch", "")) for row in valid], dtype=object)
    out_z = rel_z[branch == "out"]
    return_z = rel_z[branch == "return"]

    print(f"[res_pose_uncertainty] saved csv  -> {csv_path.resolve()}")
    if plot_path is not None:
        print(f"[res_pose_uncertainty] saved png  -> {plot_path.resolve()}")
    print(
        "[res_pose_uncertainty] z closure "
        f"{_to_float(last.get('rel_z_mm')) - _to_float(first.get('rel_z_mm')):+.3f} mm, "
        f"z range {float(np.nanmax(rel_z) - np.nanmin(rel_z)):.3f} mm"
    )
    print(
        "[res_pose_uncertainty] sigma_z assumed median "
        f"{float(np.nanmedian(sigma_z)):.4f} mm, "
        f"fit median {float(np.nanmedian(sigma_z_fit)):.4f} mm, "
        f"Jacobian condition median {float(np.nanmedian(cond)):.1f}"
    )
    if len(out_z) > 0 and len(return_z) > 0:
        print(
            "[res_pose_uncertainty] branch z median "
            f"out={float(np.nanmedian(out_z)):+.3f} mm, "
            f"return={float(np.nanmedian(return_z)):+.3f} mm, "
            f"diff={float(np.nanmedian(return_z) - np.nanmedian(out_z)):+.3f} mm"
        )
    if np.any(np.isfinite(z_step_norm)):
        worst_idx = int(np.nanargmax(z_step_norm))
        worst = valid[worst_idx]
        print(
            "[res_pose_uncertainty] largest normalized z step "
            f"frame={_to_int(worst.get('frame'))} "
            f"value={_to_float(worst.get('z_step_over_combined_sigma')):.2f}"
        )


def _latest_run_path() -> Path:
    default_dir = Path(__file__).resolve().parents[1] / "tests" / "hydramarker_tracker_runs"
    files = sorted(default_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError(f"No JSONL runs found in {default_dir}")
    return files[-1]


def main() -> None:
    args = list(sys.argv[1:])
    show = False
    make_plot = True
    assumed_sigma_px = 0.35
    point_set = "pose"
    max_frames: int | None = None
    path: Path | None = None

    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "--show":
            show = True
        elif arg == "--no-plot":
            make_plot = False
        elif arg == "--sigma-px":
            idx += 1
            if idx >= len(args):
                raise RuntimeError("--sigma-px needs a numeric value")
            assumed_sigma_px = float(args[idx])
        elif arg == "--point-set":
            idx += 1
            if idx >= len(args):
                raise RuntimeError("--point-set needs 'pose' or 'correspondence'")
            point_set = str(args[idx]).strip().lower()
            if point_set not in ("pose", "correspondence"):
                raise RuntimeError("--point-set must be 'pose' or 'correspondence'")
        elif arg == "--max-frames":
            idx += 1
            if idx >= len(args):
                raise RuntimeError("--max-frames needs an integer value")
            max_frames = int(args[idx])
        elif arg.endswith(".jsonl"):
            path = Path(arg)
        else:
            raise RuntimeError(f"Unknown option: {arg}")
        idx += 1

    if path is None:
        path = _latest_run_path()

    path = path.resolve()
    run = load_run(path)
    rows = analyze_run(
        run,
        assumed_sigma_px=assumed_sigma_px,
        point_set=point_set,
        max_frames=max_frames,
    )

    csv_path = path.with_name(f"{path.stem}_pose_uncertainty.csv")
    _write_csv(csv_path, rows)
    plot_path = plot_uncertainty(run, rows, show=show) if make_plot else None
    _print_summary(rows, csv_path, plot_path)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[res_pose_uncertainty] ERROR: {exc}")
        sys.exit(1)
