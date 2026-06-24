from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import res_fixed_lag_map_tracker as depth_replay  # noqa: E402


DEFAULT_RUNS = (
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_fb.jsonl",
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_rl.jsonl",
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_static.jsonl",
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_bf_rot.jsonl",
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_static_do.jsonl",
)


def _finite(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _median(values: list[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(np.median(arr)) if len(arr) else math.nan


def _percentile(values: list[float] | np.ndarray, q: float) -> float:
    arr = _finite(values)
    return float(np.percentile(arr, float(q))) if len(arr) else math.nan


def _corr(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 6:
        return math.nan
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 1.0e-12 else math.nan


def _load_residuals(path: Path, corner_set: str) -> tuple[dict[int, dict[str, Any]], dict[tuple[int, int], list[dict[str, Any]]]]:
    frame_rows: dict[int, dict[str, Any]] = {}
    per_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") != "frame_detail":
                continue
            frame = int(record.get("frame"))
            corners = list(record.get(corner_set) or [])
            errors: list[float] = []
            for corner in corners:
                if not isinstance(corner, dict) or "residual_px" not in corner:
                    continue
                try:
                    key = (int(corner.get("global_row")), int(corner.get("global_col")))
                    residual = np.asarray(corner.get("residual_px"), dtype=np.float64).reshape(2)
                except Exception:
                    continue
                if key[0] < 0 or key[1] < 0 or not np.all(np.isfinite(residual)):
                    continue
                error = float(np.linalg.norm(residual))
                uv = corner.get("uv_px")
                uv_arr = np.asarray(uv, dtype=np.float64).reshape(2) if isinstance(uv, (list, tuple)) and len(uv) >= 2 else np.full(2, math.nan)
                row = {
                    "frame": int(frame),
                    "global_row": int(key[0]),
                    "global_col": int(key[1]),
                    "residual_u_px": float(residual[0]),
                    "residual_v_px": float(residual[1]),
                    "error_px": float(error),
                    "uv_u_px": float(uv_arr[0]),
                    "uv_v_px": float(uv_arr[1]),
                }
                per_key[key].append(row)
                errors.append(error)
            frame_rows[frame] = {
                "frame": int(frame),
                "corner_count": int(len(errors)),
                "residual_median_px": _median(errors),
                "residual_p95_px": _percentile(errors, 95),
                "residual_mean_px": float(np.mean(errors)) if errors else math.nan,
            }
    return frame_rows, per_key


def _point_rows(per_key: dict[tuple[int, int], list[dict[str, Any]]], min_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, samples in sorted(per_key.items()):
        if len(samples) < int(min_count):
            continue
        residuals = np.asarray(
            [[sample["residual_u_px"], sample["residual_v_px"]] for sample in samples],
            dtype=np.float64,
        ).reshape(-1, 2)
        errors = np.asarray([sample["error_px"] for sample in samples], dtype=np.float64)
        median_residual = np.median(residuals, axis=0)
        jitter = np.linalg.norm(residuals - median_residual.reshape(1, 2), axis=1)
        rows.append(
            {
                "global_row": int(key[0]),
                "global_col": int(key[1]),
                "count": int(len(samples)),
                "residual_u_median_px": float(median_residual[0]),
                "residual_v_median_px": float(median_residual[1]),
                "residual_error_mean_px": float(np.mean(errors)) if len(errors) else math.nan,
                "residual_error_p95_px": _percentile(errors, 95),
                "residual_jitter_rms_px": float(math.sqrt(float(np.mean(jitter * jitter)))) if len(jitter) else math.nan,
                "residual_jitter_p95_px": _percentile(jitter, 95),
                "residual_jitter_max_px": float(np.max(jitter)) if len(jitter) else math.nan,
            }
        )
    rows.sort(key=lambda row: float(row.get("residual_jitter_p95_px", math.nan)), reverse=True)
    return rows


def _depth_rows(path: Path) -> dict[int, dict[str, Any]]:
    run = depth_replay.load_run(path, point_set="correspondence")
    cfg = depth_replay.FixedLagMapConfig(
        depth_filter_lag_frames=20,
        depth_observation_std_mm=16.0,
        depth_process_std_mm=0.05,
        depth_reprojection_guard_px=1.0,
    )
    rows, _summary = depth_replay.run_depth_kalman(run, cfg)
    return {int(row["frame"]): row for row in rows}


def evaluate(path: Path, *, corner_set: str, min_count: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    frame_rows_by_frame, per_key = _load_residuals(path, corner_set)
    point_rows = _point_rows(per_key, min_count=min_count)
    depth_by_frame = _depth_rows(path)

    frame_rows: list[dict[str, Any]] = []
    for frame in sorted(frame_rows_by_frame):
        row = dict(frame_rows_by_frame[frame])
        depth = depth_by_frame.get(frame)
        if depth is not None:
            row["kalman_abs_delta_z_mm"] = abs(float(depth.get("delta_raw_z_mm", math.nan)))
            row["kalman_reproj_excess_px"] = float(depth.get("reproj_excess_px", math.nan))
            row["tvec_z_mm"] = float(depth.get("raw_tvec_z_mm", math.nan))
        else:
            row["kalman_abs_delta_z_mm"] = math.nan
            row["kalman_reproj_excess_px"] = math.nan
            row["tvec_z_mm"] = math.nan
        frame_rows.append(row)

    median_residuals = [float(row["residual_median_px"]) for row in frame_rows]
    p95_residuals = [float(row["residual_p95_px"]) for row in frame_rows]
    corner_counts = [float(row["corner_count"]) for row in frame_rows]
    abs_dz = [float(row["kalman_abs_delta_z_mm"]) for row in frame_rows]
    summary = {
        "run_label": path.stem,
        "corner_set": str(corner_set),
        "frames_with_details": int(len(frame_rows)),
        "point_keys": int(len(per_key)),
        "point_keys_kept": int(len(point_rows)),
        "frame_residual_median_px": _median(median_residuals),
        "frame_residual_p95_median_px": _median(p95_residuals),
        "frame_residual_p95_p95_px": _percentile(p95_residuals, 95),
        "corner_count_median": _median(corner_counts),
        "corner_jitter_p95_median_px": _median([float(row["residual_jitter_p95_px"]) for row in point_rows]),
        "corner_jitter_p95_p95_px": _percentile([float(row["residual_jitter_p95_px"]) for row in point_rows], 95),
        "corr_abs_kalman_dz_vs_frame_residual_median": _corr(abs_dz, median_residuals),
        "corr_abs_kalman_dz_vs_frame_residual_p95": _corr(abs_dz, p95_residuals),
        "corr_abs_kalman_dz_vs_corner_count": _corr(abs_dz, corner_counts),
    }
    return summary, frame_rows, point_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-corner UV residual jitter diagnostics for HydraMarker logs.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--corner-set", choices=("pose_corners",), default="pose_corners")
    parser.add_argument("--min-count", type=int, default=8)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = [Path(p) for p in args.paths] if args.paths else [Path(p) for p in DEFAULT_RUNS]
    summary_rows: list[dict[str, Any]] = []
    for path in paths:
        summary, frame_rows, point_rows = evaluate(path.resolve(), corner_set=str(args.corner_set), min_count=int(args.min_count))
        summary_rows.append(summary)
        print(
            "[corner_uv_jitter] "
            f"{path.name}: frames={summary['frames_with_details']} keys={summary['point_keys']} "
            f"frame_med={summary['frame_residual_median_px']:.3f}px "
            f"frame_p95med={summary['frame_residual_p95_median_px']:.3f}px "
            f"corner_jitter_p95_p95={summary['corner_jitter_p95_p95_px']:.3f}px "
            f"corr_dz_med/p95={summary['corr_abs_kalman_dz_vs_frame_residual_median']:.3f}/"
            f"{summary['corr_abs_kalman_dz_vs_frame_residual_p95']:.3f}"
        )
        for row in point_rows[: max(int(args.top), 0)]:
            print(
                "  "
                f"key=({row['global_row']},{row['global_col']}) "
                f"jitter95={row['residual_jitter_p95_px']:.3f}px "
                f"jitterRMS={row['residual_jitter_rms_px']:.3f}px "
                f"err95={row['residual_error_p95_px']:.3f}px "
                f"count={row['count']}"
            )
        if not args.no_csv:
            stem = path.with_suffix("")
            _write_csv(stem.with_name(f"{stem.name}_corner_uv_jitter_frames.csv"), frame_rows)
            _write_csv(stem.with_name(f"{stem.name}_corner_uv_jitter_points.csv"), point_rows)
    if not args.no_csv and summary_rows:
        out_dir = paths[0].resolve().parent
        _write_csv(out_dir / "hydramarker_corner_uv_jitter_summary.csv", summary_rows)


if __name__ == "__main__":
    main()
