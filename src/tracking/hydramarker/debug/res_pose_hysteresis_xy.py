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


def _median(values: list[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(np.median(arr)) if len(arr) else math.nan


def _mean(values: list[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(np.mean(arr)) if len(arr) else math.nan


def _percentile(values: list[float] | np.ndarray, q: float) -> float:
    arr = _finite(values)
    return float(np.percentile(arr, q)) if len(arr) else math.nan


def load_run(path: Path) -> dict[str, Any]:
    run_id = path.stem
    timestamp = ""
    frames: dict[int, dict[str, Any]] = {}
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
            elif record_type == "frame":
                data = dict(record.get("data") or {})
                frame = _to_int(data.get("frame"), default=len(frames))
                frames[frame] = data
            elif record_type == "run_summary":
                summary = dict(record.get("summary") or {})

    if not frames:
        raise RuntimeError(f"No frame records found in:\n{path}")

    return {
        "path": path,
        "run_id": run_id,
        "timestamp": timestamp,
        "frames": frames,
        "summary": summary,
    }


def build_frame_rows(
    run: dict[str, Any],
    *,
    movement_axis_override: str | None,
    max_frames: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = run["frames"]
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
    if movement_axis_override:
        movement_axis = movement_axis_override.lower()
        if movement_axis not in COMPONENTS:
            raise RuntimeError("--movement-axis must be x, y, or z")
        movement_axis_idx = COMPONENTS.index(movement_axis)
    else:
        movement_axis_idx = int(np.nanargmax(ranges))
        movement_axis = COMPONENTS[movement_axis_idx]

    movement_values = rel_tvec[:, movement_axis_idx]
    turn_idx = int(np.nanargmax(np.abs(movement_values)))
    turn_frame = sorted_frames[turn_idx]

    rows: list[dict[str, Any]] = []
    for idx, frame in enumerate(sorted_frames):
        data = frames[frame]
        rel = rel_tvec[idx]
        tvec = tvec_arr[idx]
        if not np.all(np.isfinite(rel)):
            continue
        branch = "out" if frame <= turn_frame else "return"
        rows.append(
            {
                "frame": int(frame),
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
                "num_points": _to_float(data.get("num_points")),
                "pose_reproj_mean_px": _to_float(data.get("pose_reproj_mean_px")),
                "pose_reproj_p95_px": _to_float(data.get("pose_reproj_p95_px")),
                "pose_reproj_max_px": _to_float(data.get("pose_reproj_max_px")),
                "pose_image_centroid_u_px": _to_float(data.get("pose_image_centroid_u_px")),
                "pose_image_centroid_v_px": _to_float(data.get("pose_image_centroid_v_px")),
            }
        )

    meta = {
        "origin_frame": int(origin_frame),
        "origin_tvec": origin_tvec,
        "movement_axis": movement_axis,
        "movement_axis_idx": movement_axis_idx,
        "turn_frame": int(turn_frame),
        "x_range_mm": float(ranges[0]),
        "y_range_mm": float(ranges[1]),
        "z_range_mm": float(ranges[2]),
    }
    return rows, meta


def _array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_to_float(row.get(key)) for row in rows], dtype=np.float64)


def _model_matrix(
    rows: list[dict[str, Any]],
    *,
    geometry: str,
    include_branch: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    x = _array(rows, "rel_x_mm")
    y = _array(rows, "rel_y_mm")
    z = _array(rows, "rel_z_mm")
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
    return X[mask], z[mask], names, mask


def fit_model(
    rows: list[dict[str, Any]],
    *,
    geometry: str,
    include_branch: bool,
) -> dict[str, Any]:
    X, z, names, mask = _model_matrix(rows, geometry=geometry, include_branch=include_branch)
    if len(z) < max(4, len(names) + 2):
        return {
            "model": f"{geometry}{'+branch' if include_branch else ''}",
            "n": int(len(z)),
            "r2": math.nan,
            "rmse_mm": math.nan,
            "branch_coeff_mm": math.nan,
        }

    try:
        beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "model": f"{geometry}{'+branch' if include_branch else ''}",
            "n": int(len(z)),
            "r2": math.nan,
            "rmse_mm": math.nan,
            "branch_coeff_mm": math.nan,
        }

    pred = X @ beta
    ss_res = float(np.sum((z - pred) ** 2))
    ss_tot = float(np.sum((z - float(np.mean(z))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan
    rmse = float(math.sqrt(float(np.mean((z - pred) ** 2))))
    branch_coeff = math.nan
    if include_branch and "branch_return" in names:
        branch_coeff = float(beta[names.index("branch_return")])

    residuals = np.full(len(rows), np.nan, dtype=np.float64)
    residuals[mask] = z - pred
    return {
        "model": f"{geometry}{'+branch' if include_branch else ''}",
        "geometry": geometry,
        "include_branch": int(include_branch),
        "n": int(len(z)),
        "r2": float(r2),
        "rmse_mm": rmse,
        "branch_coeff_mm": branch_coeff,
        "coef_names": ";".join(names),
        "coef_values": ";".join(f"{float(v):.12g}" for v in beta),
        "residuals": residuals,
    }


def build_nearest_pairs(
    frame_rows: list[dict[str, Any]],
    *,
    thresholds_mm: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_rows = [row for row in frame_rows if str(row.get("branch")) == "out"]
    return_rows = [row for row in frame_rows if str(row.get("branch")) == "return"]
    if not out_rows or not return_rows:
        return [], []

    out_xy = np.asarray([[_to_float(row.get("rel_x_mm")), _to_float(row.get("rel_y_mm"))] for row in out_rows], dtype=np.float64)
    out_z = np.asarray([_to_float(row.get("rel_z_mm")) for row in out_rows], dtype=np.float64)
    pair_rows: list[dict[str, Any]] = []

    for return_row in return_rows:
        return_xy = np.asarray([_to_float(return_row.get("rel_x_mm")), _to_float(return_row.get("rel_y_mm"))], dtype=np.float64)
        return_z = _to_float(return_row.get("rel_z_mm"))
        if not np.all(np.isfinite(return_xy)) or not np.isfinite(return_z):
            continue
        delta = out_xy - return_xy.reshape(1, 2)
        distance = np.sqrt(np.sum(delta * delta, axis=1))
        distance[~np.isfinite(distance) | ~np.isfinite(out_z)] = math.inf
        if not np.any(np.isfinite(distance)):
            continue
        out_idx = int(np.argmin(distance))
        out_row = out_rows[out_idx]
        pair_rows.append(
            {
                "return_frame": int(return_row["frame"]),
                "out_frame": int(out_row["frame"]),
                "xy_distance_mm": float(distance[out_idx]),
                "dx_return_minus_out_mm": float(return_xy[0] - out_xy[out_idx, 0]),
                "dy_return_minus_out_mm": float(return_xy[1] - out_xy[out_idx, 1]),
                "return_x_mm": float(return_xy[0]),
                "return_y_mm": float(return_xy[1]),
                "out_x_mm": float(out_xy[out_idx, 0]),
                "out_y_mm": float(out_xy[out_idx, 1]),
                "return_z_mm": float(return_z),
                "out_z_mm": float(out_z[out_idx]),
                "return_minus_out_z_mm": float(return_z - out_z[out_idx]),
                "return_points": _to_float(return_row.get("num_points")),
                "out_points": _to_float(out_row.get("num_points")),
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for threshold in sorted({float(v) for v in thresholds_mm if float(v) > 0.0}):
        subset = [row for row in pair_rows if _to_float(row.get("xy_distance_mm")) <= threshold]
        dz = [_to_float(row.get("return_minus_out_z_mm")) for row in subset]
        distances = [_to_float(row.get("xy_distance_mm")) for row in subset]
        dx = [_to_float(row.get("dx_return_minus_out_mm")) for row in subset]
        dy = [_to_float(row.get("dy_return_minus_out_mm")) for row in subset]
        summary_rows.append(
            {
                "metric": "nearest_xy_pairs",
                "threshold_mm": threshold,
                "n": len(subset),
                "return_minus_out_z_median_mm": _median(dz),
                "return_minus_out_z_mean_mm": _mean(dz),
                "return_minus_out_z_p25_mm": _percentile(dz, 25),
                "return_minus_out_z_p75_mm": _percentile(dz, 75),
                "xy_distance_median_mm": _median(distances),
                "dx_median_mm": _median(dx),
                "dy_median_mm": _median(dy),
            }
        )
    return pair_rows, summary_rows


def build_summary_rows(
    frame_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    pair_summary_rows: list[dict[str, Any]],
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    branches = np.asarray([str(row.get("branch")) for row in frame_rows], dtype=object)
    z = _array(frame_rows, "rel_z_mm")
    x = _array(frame_rows, "rel_x_mm")
    y = _array(frame_rows, "rel_y_mm")
    out_z = z[branches == "out"]
    return_z = z[branches == "return"]
    out_x = x[branches == "out"]
    return_x = x[branches == "return"]
    out_y = y[branches == "out"]
    return_y = y[branches == "return"]

    rows: list[dict[str, Any]] = [
        {
            "metric": "run",
            "threshold_mm": "",
            "n": len(frame_rows),
            "movement_axis": meta["movement_axis"],
            "turn_frame": meta["turn_frame"],
            "x_range_mm": meta["x_range_mm"],
            "y_range_mm": meta["y_range_mm"],
            "z_range_mm": meta["z_range_mm"],
            "x_closure_mm": float(x[-1] - x[0]) if len(x) else math.nan,
            "y_closure_mm": float(y[-1] - y[0]) if len(y) else math.nan,
            "z_closure_mm": float(z[-1] - z[0]) if len(z) else math.nan,
            "out_z_median_mm": _median(out_z),
            "return_z_median_mm": _median(return_z),
            "return_minus_out_z_median_mm": _median(return_z) - _median(out_z),
            "out_x_median_mm": _median(out_x),
            "return_x_median_mm": _median(return_x),
            "return_minus_out_x_median_mm": _median(return_x) - _median(out_x),
            "out_y_median_mm": _median(out_y),
            "return_y_median_mm": _median(return_y),
            "return_minus_out_y_median_mm": _median(return_y) - _median(out_y),
        }
    ]

    for row in model_rows:
        clean = {key: value for key, value in row.items() if key != "residuals"}
        clean["metric"] = f"model_{row['model']}"
        rows.append(clean)
    rows.extend(pair_summary_rows)
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
    frame_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    pair_summary_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    *,
    show: bool,
) -> Path:
    import matplotlib

    matplotlib.use("Agg" if not show else "QtAgg")
    import matplotlib.pyplot as plt

    _setup_plot_style(plt)

    path: Path = run["path"]
    out_path = path.with_name(f"{path.stem}_xy_hysteresis_plot.png")
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=False)
    fig.suptitle("HydraTracker XY-controlled z hysteresis", fontsize=16, fontweight="bold")
    fig.text(
        0.01,
        0.965,
        (
            f"{run['run_id']} -- {run.get('timestamp', '')}   "
            f"movement={meta['movement_axis']}   turn_frame={meta['turn_frame']}"
        ),
        fontsize=9,
        ha="left",
        va="top",
    )

    x = _array(frame_rows, "rel_x_mm")
    y = _array(frame_rows, "rel_y_mm")
    z = _array(frame_rows, "rel_z_mm")
    frames = _array(frame_rows, "frame")
    branches = np.asarray([str(row.get("branch")) for row in frame_rows], dtype=object)
    out_mask = branches == "out"
    return_mask = branches == "return"

    axes[0].plot(x[out_mask], y[out_mask], color="#1f77b4", marker="o", markersize=3, linewidth=1.3, label="out")
    axes[0].plot(x[return_mask], y[return_mask], color="#d62728", marker="o", markersize=3, linewidth=1.3, label="return")
    axes[0].scatter([x[0]], [y[0]], color="#2ca02c", s=55, zorder=3, label="start")
    axes[0].scatter([x[-1]], [y[-1]], color="#111111", s=55, zorder=3, label="end")
    axes[0].set_title("Measured XY path in camera frame")
    axes[0].set_xlabel("delta x [mm]")
    axes[0].set_ylabel("delta y [mm]")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].scatter(y[out_mask], z[out_mask], color="#1f77b4", s=18, alpha=0.8, label="out")
    axes[1].scatter(y[return_mask], z[return_mask], color="#d62728", s=18, alpha=0.8, label="return")
    axes[1].set_title("Raw z versus movement coordinate")
    axes[1].set_xlabel("delta y [mm]")
    axes[1].set_ylabel("delta z [mm]")
    axes[1].legend(loc="best", fontsize=8)

    quad = next((row for row in model_rows if row.get("model") == "xy_quadratic"), None)
    branch_quad = next((row for row in model_rows if row.get("model") == "xy_quadratic+branch"), None)
    residual = np.asarray(quad.get("residuals"), dtype=np.float64) if quad else np.full(len(frame_rows), np.nan)
    axes[2].plot(frames, residual, color="#9467bd", marker="o", markersize=3, linewidth=1.4, label="z residual after f(x,y)")
    if branch_quad:
        branch_coeff = _to_float(branch_quad.get("branch_coeff_mm"))
        axes[2].axhline(branch_coeff, color="#d62728", linestyle="--", linewidth=1.0, label=f"branch coeff {branch_coeff:+.3f} mm")
    axes[2].axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[2].set_title("Residual z after XY geometry model")
    axes[2].set_xlabel("frame")
    axes[2].set_ylabel("residual z [mm]")
    axes[2].legend(loc="best", fontsize=8)

    thresholds = np.asarray([_to_float(row.get("threshold_mm")) for row in pair_summary_rows], dtype=np.float64)
    pair_dz = np.asarray([_to_float(row.get("return_minus_out_z_median_mm")) for row in pair_summary_rows], dtype=np.float64)
    pair_n = np.asarray([_to_float(row.get("n")) for row in pair_summary_rows], dtype=np.float64)
    valid = np.isfinite(thresholds)
    axes[3].plot(thresholds[valid], pair_dz[valid], color="#8c564b", marker="o", linewidth=1.8, label="median return-out z")
    ax_n = axes[3].twinx()
    ax_n.bar(thresholds[valid], pair_n[valid], width=0.18, color="#4c78a8", alpha=0.25, label="pair count")
    axes[3].axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[3].set_title("Nearest out/return pairs with XY distance threshold")
    axes[3].set_xlabel("max XY distance [mm]")
    axes[3].set_ylabel("return - out z [mm]")
    ax_n.set_ylabel("pairs")
    handles1, labels1 = axes[3].get_legend_handles_labels()
    handles2, labels2 = ax_n.get_legend_handles_labels()
    axes[3].legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=8)

    for ax in axes:
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


def _parse_thresholds(text: str) -> list[float]:
    out: list[float] = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out or [0.5, 1.0, 2.0, 3.0, 5.0]


def _parse_args(argv: list[str]) -> dict[str, Any]:
    args: dict[str, Any] = {
        "path": None,
        "use_latest": False,
        "show": False,
        "make_plot": True,
        "movement_axis": None,
        "max_frames": None,
        "thresholds_mm": [0.5, 1.0, 2.0, 3.0, 5.0],
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
        elif arg == "--movement-axis":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--movement-axis needs x, y, or z")
            args["movement_axis"] = str(argv[idx]).strip().lower()
        elif arg == "--max-frames":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--max-frames needs an integer")
            args["max_frames"] = int(argv[idx])
        elif arg == "--xy-thresholds":
            idx += 1
            if idx >= len(argv):
                raise RuntimeError("--xy-thresholds needs comma-separated mm values")
            args["thresholds_mm"] = _parse_thresholds(str(argv[idx]))
        elif arg.endswith(".jsonl"):
            args["path"] = Path(arg)
        else:
            raise RuntimeError(f"Unknown option: {arg}")
        idx += 1
    return args


def _print_summary(
    summary_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    paths: dict[str, Path],
) -> None:
    print(f"[res_pose_hysteresis_xy] saved frame csv   -> {paths['frame'].resolve()}")
    print(f"[res_pose_hysteresis_xy] saved pair csv    -> {paths['pair'].resolve()}")
    print(f"[res_pose_hysteresis_xy] saved summary csv -> {paths['summary'].resolve()}")
    if "plot" in paths:
        print(f"[res_pose_hysteresis_xy] saved plot        -> {paths['plot'].resolve()}")

    run_row = next((row for row in summary_rows if row.get("metric") == "run"), {})
    print(
        "[res_pose_hysteresis_xy] run closure "
        f"x={_to_float(run_row.get('x_closure_mm')):+.3f} mm, "
        f"y={_to_float(run_row.get('y_closure_mm')):+.3f} mm, "
        f"z={_to_float(run_row.get('z_closure_mm')):+.3f} mm"
    )
    print(
        "[res_pose_hysteresis_xy] raw branch medians "
        f"out_z={_to_float(run_row.get('out_z_median_mm')):+.3f} mm, "
        f"return_z={_to_float(run_row.get('return_z_median_mm')):+.3f} mm, "
        f"diff={_to_float(run_row.get('return_minus_out_z_median_mm')):+.3f} mm"
    )

    print("[res_pose_hysteresis_xy] branch coefficient after geometry control:")
    for row in model_rows:
        if not row.get("include_branch"):
            continue
        print(
            "  "
            f"{row['geometry']}: "
            f"branch={_to_float(row.get('branch_coeff_mm')):+.3f} mm, "
            f"R2={_to_float(row.get('r2')):.3f}, "
            f"RMSE={_to_float(row.get('rmse_mm')):.3f} mm"
        )

    nearest = [
        row
        for row in summary_rows
        if str(row.get("metric")) == "nearest_xy_pairs" and _to_int(row.get("n"), 0) > 0
    ]
    if nearest:
        print("[res_pose_hysteresis_xy] nearest XY pair medians:")
        for row in nearest:
            print(
                "  "
                f"d<={_to_float(row.get('threshold_mm')):.2f} mm: "
                f"n={_to_int(row.get('n'))}, "
                f"return-out z={_to_float(row.get('return_minus_out_z_median_mm')):+.3f} mm, "
                f"xy_med={_to_float(row.get('xy_distance_median_mm')):.3f} mm"
            )


def main() -> None:
    args = _parse_args(sys.argv[1:])
    path = args["path"]
    if path is None:
        path = _latest_run_path() if args["use_latest"] else _select_run_path_qt()
    path = Path(path).resolve()
    run = load_run(path)

    frame_rows, meta = build_frame_rows(
        run,
        movement_axis_override=args["movement_axis"],
        max_frames=args["max_frames"],
    )

    model_rows = [
        fit_model(frame_rows, geometry="none", include_branch=True),
        fit_model(frame_rows, geometry="y", include_branch=False),
        fit_model(frame_rows, geometry="y", include_branch=True),
        fit_model(frame_rows, geometry="xy", include_branch=False),
        fit_model(frame_rows, geometry="xy", include_branch=True),
        fit_model(frame_rows, geometry="xy_quadratic", include_branch=False),
        fit_model(frame_rows, geometry="xy_quadratic", include_branch=True),
    ]

    xy_residual = next((row.get("residuals") for row in model_rows if row.get("model") == "xy_quadratic"), None)
    xy_branch_residual = next(
        (row.get("residuals") for row in model_rows if row.get("model") == "xy_quadratic+branch"),
        None,
    )
    if xy_residual is not None:
        for idx, row in enumerate(frame_rows):
            row["z_resid_xy_quadratic_mm"] = float(np.asarray(xy_residual, dtype=np.float64)[idx])
    if xy_branch_residual is not None:
        for idx, row in enumerate(frame_rows):
            row["z_resid_xy_quadratic_branch_mm"] = float(np.asarray(xy_branch_residual, dtype=np.float64)[idx])

    pair_rows, pair_summary_rows = build_nearest_pairs(
        frame_rows,
        thresholds_mm=list(args["thresholds_mm"]),
    )
    summary_rows = build_summary_rows(frame_rows, model_rows, pair_summary_rows, meta)

    frame_csv = path.with_name(f"{path.stem}_xy_hysteresis_frames.csv")
    pair_csv = path.with_name(f"{path.stem}_xy_hysteresis_pairs.csv")
    summary_csv = path.with_name(f"{path.stem}_xy_hysteresis_summary.csv")
    _write_csv(frame_csv, frame_rows)
    _write_csv(pair_csv, pair_rows)
    _write_csv(summary_csv, summary_rows)

    paths = {"frame": frame_csv, "pair": pair_csv, "summary": summary_csv}
    if bool(args["make_plot"]):
        paths["plot"] = plot_results(
            run,
            frame_rows,
            pair_rows,
            pair_summary_rows,
            model_rows,
            meta,
            show=bool(args["show"]),
        )

    _print_summary(summary_rows, model_rows, paths)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[res_pose_hysteresis_xy] ERROR: {exc}")
        sys.exit(1)
