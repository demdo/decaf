"""Replay residual jitter across HydraMarker tracker observations.

This script reconstructs residual series from JSONL detail records to inspect
frame-to-frame reprojection noise and its relationship to pose stability.
"""

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

from res_static_irls_replay import (  # noqa: E402
    IrlsConfig,
    _T_to_pose,
    _median,
    _percentile,
    _pose_row,
    _pose_to_T,
    _rms,
    _to_float,
    _to_int,
    build_age_maps,
    load_run,
    solve_irls_lie_pose,
    summarize,
)


DEFAULT_RUNS = (
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_fb.jsonl",
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_rl.jsonl",
)


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


def _residual_by_frame(
    path: Path,
    *,
    corner_set: str,
) -> dict[int, dict[tuple[int, int], np.ndarray]]:
    residuals: dict[int, dict[tuple[int, int], np.ndarray]] = defaultdict(dict)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            if record.get("type") != "frame_detail":
                continue
            frame = _to_int(record.get("frame"), default=-1)
            if frame < 0:
                continue
            for corner in list(record.get(corner_set) or []):
                if not isinstance(corner, dict):
                    continue
                residual = corner.get("residual_px")
                if not isinstance(residual, (list, tuple)) or len(residual) < 2:
                    continue
                key = (
                    _to_int(corner.get("global_row"), default=_to_int(corner.get("row"), -1)),
                    _to_int(corner.get("global_col"), default=_to_int(corner.get("col"), -1)),
                )
                if key[0] < 0 or key[1] < 0:
                    continue
                vec = np.asarray([_to_float(v) for v in residual[:2]], dtype=np.float64)
                if np.all(np.isfinite(vec)):
                    residuals[int(frame)][(int(key[0]), int(key[1]))] = vec.reshape(2)
    return {int(frame): dict(values) for frame, values in residuals.items()}


def _jitter_metric(samples: list[np.ndarray], metric: str) -> float:
    if not samples:
        return math.nan
    residuals = np.asarray(samples, dtype=np.float64).reshape(-1, 2)
    residuals = residuals[np.all(np.isfinite(residuals), axis=1)]
    if len(residuals) == 0:
        return math.nan
    median = np.median(residuals, axis=0)
    jitter = np.linalg.norm(residuals - median.reshape(1, 2), axis=1)
    if metric == "rms":
        return _rms(jitter)
    if metric == "p95":
        return _percentile(jitter, 95)
    raise RuntimeError(f"Unknown jitter metric: {metric}")


def _weight_from_jitter(
    jitter_px: float,
    *,
    scale_px: float,
    power: float,
    min_weight: float,
    default_weight: float,
) -> float:
    if not np.isfinite(jitter_px):
        return float(default_weight)
    scale = max(float(scale_px), 1.0e-9)
    exponent = max(float(power), 1.0e-9)
    stability = 1.0 / (1.0 + (max(float(jitter_px), 0.0) / scale) ** exponent)
    return float(np.clip(stability, float(min_weight), 1.0))


def _point_summary_rows(
    per_key: dict[tuple[int, int], list[np.ndarray]],
    *,
    metric: str,
    scale_px: float,
    power: float,
    min_weight: float,
    default_weight: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, samples in sorted(per_key.items()):
        residuals = np.asarray(samples, dtype=np.float64).reshape(-1, 2)
        errors = np.linalg.norm(residuals, axis=1) if len(residuals) else np.asarray([], dtype=np.float64)
        jitter = _jitter_metric(samples, metric)
        rows.append(
            {
                "global_row": int(key[0]),
                "global_col": int(key[1]),
                "count": int(len(samples)),
                "residual_error_median_px": _median(errors),
                "residual_error_p95_px": _percentile(errors, 95),
                "residual_jitter_px": float(jitter),
                "base_weight": _weight_from_jitter(
                    jitter,
                    scale_px=scale_px,
                    power=power,
                    min_weight=min_weight,
                    default_weight=default_weight,
                ),
            }
        )
    rows.sort(key=lambda row: _to_float(row.get("residual_jitter_px")), reverse=True)
    return rows


def build_weight_maps(
    observations: list[Any],
    residuals_by_frame: dict[int, dict[tuple[int, int], np.ndarray]],
    *,
    mode: str,
    metric: str,
    scale_px: float,
    power: float,
    min_history: int,
    min_weight: float,
    default_weight: float,
) -> tuple[list[dict[tuple[int, int], float]], list[dict[str, Any]]]:
    observations = list(observations)
    min_history = max(int(min_history), 1)
    per_key: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for frame_values in residuals_by_frame.values():
        for key, residual in frame_values.items():
            per_key[key].append(np.asarray(residual, dtype=np.float64).reshape(2))

    if mode == "offline":
        key_weight: dict[tuple[int, int], float] = {}
        for key, samples in per_key.items():
            if len(samples) < min_history:
                continue
            jitter = _jitter_metric(samples, metric)
            key_weight[key] = _weight_from_jitter(
                jitter,
                scale_px=scale_px,
                power=power,
                min_weight=min_weight,
                default_weight=default_weight,
            )
        maps = [
            {key: float(key_weight.get(key, default_weight)) for key in obs.uv_by_key}
            for obs in observations
        ]
        return maps, _point_summary_rows(
            per_key,
            metric=metric,
            scale_px=scale_px,
            power=power,
            min_weight=min_weight,
            default_weight=default_weight,
        )

    if mode != "causal":
        raise RuntimeError("--jitter-mode must be causal or offline")

    history: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    maps: list[dict[tuple[int, int], float]] = []
    for obs in observations:
        weight_map: dict[tuple[int, int], float] = {}
        for key in obs.uv_by_key:
            samples = history.get(key, [])
            if len(samples) < min_history:
                weight_map[key] = float(default_weight)
            else:
                jitter = _jitter_metric(samples, metric)
                weight_map[key] = _weight_from_jitter(
                    jitter,
                    scale_px=scale_px,
                    power=power,
                    min_weight=min_weight,
                    default_weight=default_weight,
                )
        maps.append(weight_map)
        for key, residual in residuals_by_frame.get(int(obs.frame), {}).items():
            history[key].append(np.asarray(residual, dtype=np.float64).reshape(2))

    return maps, _point_summary_rows(
        history,
        metric=metric,
        scale_px=scale_px,
        power=power,
        min_weight=min_weight,
        default_weight=default_weight,
    )


def _arrays_for_observation(
    obs: Any,
    *,
    weight_map: dict[tuple[int, int], float],
    age_weight_by_key: dict[tuple[int, int], float],
    default_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    weights: list[float] = []
    used_keys: list[tuple[int, int]] = []
    for key in sorted(obs.uv_by_key):
        xyz = np.asarray(obs.object_by_key[key], dtype=np.float64).reshape(3)
        uv = np.asarray(obs.uv_by_key[key], dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(uv)):
            continue
        age_weight = float(age_weight_by_key.get(key, 1.0))
        object_points.append(xyz)
        image_points.append(uv)
        weights.append(float(weight_map.get(key, default_weight)) * age_weight)
        used_keys.append(key)
    return (
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
        np.asarray(weights, dtype=np.float64).reshape(-1),
        used_keys,
    )


def replay(
    run: dict[str, Any],
    *,
    config: IrlsConfig,
    jitter_weight_maps: list[dict[tuple[int, int], float]],
    default_weight: float,
    seed_mode: str,
) -> list[dict[str, Any]]:
    observations = list(run["observations"])
    reference = observations[0]
    ref_T = _pose_to_T(reference.original_rvec.copy(), reference.original_tvec.copy())
    age_maps = build_age_maps(observations, age_ramp_frames=config.age_ramp_frames)

    frame_rows: list[dict[str, Any]] = []
    pose_by_method: dict[str, list[np.ndarray]] = {
        "logged_original": [],
        "irls_uniform": [],
        "irls_residual_jitter": [],
    }
    current_T_by_method = {
        "irls_uniform": ref_T.copy(),
        "irls_residual_jitter": ref_T.copy(),
    }

    uniform_maps = [{key: 1.0 for key in obs.uv_by_key} for obs in observations]
    maps_by_method = {
        "irls_uniform": uniform_maps,
        "irls_residual_jitter": jitter_weight_maps,
    }

    for obs_idx, obs in enumerate(observations):
        all_keys = sorted(obs.uv_by_key)
        frame_rows.append(
            _pose_row(
                method="logged_original",
                obs=obs,
                solved=True,
                rvec=obs.original_rvec,
                tvec=obs.original_tvec,
                point_count=len(all_keys),
                distinct_key_count=len(set(all_keys)),
                used_frame_count=1,
                stats={},
            )
        )
        pose_by_method["logged_original"].append(obs.original_tvec.copy())

        for method in ("irls_uniform", "irls_residual_jitter"):
            if seed_mode == "raw":
                seed_T = _pose_to_T(obs.original_rvec.copy(), obs.original_tvec.copy())
            elif seed_mode == "ref":
                seed_T = ref_T.copy()
            elif seed_mode == "previous":
                seed_T = current_T_by_method[method]
            else:
                raise RuntimeError("--seed-mode must be previous, raw, or ref")
            obj, uv, weights, used_keys = _arrays_for_observation(
                obs,
                weight_map=maps_by_method[method][obs_idx],
                age_weight_by_key=age_maps[obs_idx],
                default_weight=default_weight,
            )
            result = solve_irls_lie_pose(
                obj,
                uv,
                weights,
                run["K"],
                run["dist"],
                seed_T,
                config,
            )
            current_T_by_method[method] = seed_T.copy()
            if result.success:
                current_T_by_method[method] = result.T.copy()
            rvec, tvec = _T_to_pose(current_T_by_method[method])
            frame_rows.append(
                _pose_row(
                    method=method,
                    obs=obs,
                    solved=result.success,
                    rvec=rvec,
                    tvec=tvec,
                    point_count=result.point_count,
                    distinct_key_count=len(set(used_keys)),
                    used_frame_count=1,
                    stats=result.stats,
                    iterations=result.iterations,
                    mean_weight=result.mean_weight,
                    condition_number=result.condition_number,
                    min_eigenvalue=result.min_eigenvalue,
                    weak_z_alignment=result.weak_z_alignment,
                    last_step_norm=result.last_step_norm,
                )
            )
            pose_by_method[method].append(tvec.copy())

    rel_by_method: dict[str, np.ndarray] = {}
    for method, tvecs in pose_by_method.items():
        arr = np.asarray(tvecs, dtype=np.float64).reshape(-1, 3)
        rel_by_method[method] = arr - arr[0].reshape(1, 3)

    method_offsets = {method: 0 for method in pose_by_method}
    for row in frame_rows:
        method = str(row.get("method") or "")
        idx = method_offsets[method]
        rel = rel_by_method[method][idx]
        row["rel_x_mm"] = float(rel[0])
        row["rel_y_mm"] = float(rel[1])
        row["rel_z_mm"] = float(rel[2])
        method_offsets[method] += 1

    return frame_rows


def _print_summary(path: Path, summary_rows: list[dict[str, Any]], point_rows: list[dict[str, Any]]) -> None:
    print(f"[residual_jitter_replay] {path.name}")
    for row in summary_rows:
        print(
            "  "
            f"{row['method']}: "
            f"x={_to_float(row.get('x_range_mm')):.3f} "
            f"y={_to_float(row.get('y_range_mm')):.3f} "
            f"z={_to_float(row.get('z_range_mm')):.3f} mm, "
            f"rms={_to_float(row.get('reproj_rms_median_px')):.3f}px, "
            f"wrms={_to_float(row.get('reproj_weighted_rms_median_px')):.3f}px, "
            f"mean_w={_to_float(row.get('mean_weight_median')):.3f}, "
            f"points={_to_float(row.get('point_count_median')):.1f}"
        )
    if point_rows:
        print("  highest residual-jitter corners:")
        for row in point_rows[:3]:
            print(
                "    "
                f"key=({row['global_row']},{row['global_col']}) "
                f"jitter={_to_float(row.get('residual_jitter_px')):.3f}px "
                f"weight={_to_float(row.get('base_weight')):.3f} "
                f"count={_to_int(row.get('count'))}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay HydraMarker poses with per-corner residual-jitter weighting."
    )
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--point-set", choices=("pose", "correspondence"), default="correspondence")
    parser.add_argument("--corner-set", choices=("pose_corners",), default="pose_corners")
    parser.add_argument("--jitter-mode", choices=("causal", "offline"), default="causal")
    parser.add_argument("--jitter-metric", choices=("p95", "rms"), default="p95")
    parser.add_argument("--jitter-scale-px", type=float, default=0.50)
    parser.add_argument("--jitter-power", type=float, default=2.0)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument("--min-base-weight", type=float, default=0.05)
    parser.add_argument("--default-weight", type=float, default=1.0)
    parser.add_argument("--robust-c-px", type=float, default=0.20)
    parser.add_argument("--condition-boost", type=float, default=0.0)
    parser.add_argument("--age-ramp-frames", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--seed-mode", choices=("previous", "raw", "ref"), default="previous")
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = [Path(path) for path in args.paths] if args.paths else [Path(path) for path in DEFAULT_RUNS]
    all_summary_rows: list[dict[str, Any]] = []

    for path in paths:
        path = path.resolve()
        run = load_run(path, point_set=str(args.point_set))
        residuals_by_frame = _residual_by_frame(path, corner_set=str(args.corner_set))
        weight_maps, point_rows = build_weight_maps(
            run["observations"],
            residuals_by_frame,
            mode=str(args.jitter_mode),
            metric=str(args.jitter_metric),
            scale_px=float(args.jitter_scale_px),
            power=float(args.jitter_power),
            min_history=int(args.min_history),
            min_weight=float(args.min_base_weight),
            default_weight=float(args.default_weight),
        )
        config = IrlsConfig(
            max_iterations=int(args.max_iterations),
            robust_c_px=float(args.robust_c_px),
            uv_stability_scale_px=1.0,
            min_base_weight=float(args.min_base_weight),
            age_ramp_frames=int(args.age_ramp_frames),
            condition_boost=float(args.condition_boost),
        )
        frame_rows = replay(
            run,
            config=config,
            jitter_weight_maps=weight_maps,
            default_weight=float(args.default_weight),
            seed_mode=str(args.seed_mode),
        )
        summary_rows = summarize(frame_rows)
        for row in summary_rows:
            row["run"] = path.name
            row["jitter_mode"] = str(args.jitter_mode)
            row["jitter_metric"] = str(args.jitter_metric)
            row["jitter_scale_px"] = float(args.jitter_scale_px)
            row["min_history"] = int(args.min_history)
            row["seed_mode"] = str(args.seed_mode)
        all_summary_rows.extend(summary_rows)
        _print_summary(path, summary_rows, point_rows)

        if not bool(args.no_csv):
            suffix = (
                f"resjit_{args.jitter_mode}_{args.jitter_metric}"
                f"_s{str(float(args.jitter_scale_px)).replace('.', 'p')}"
                f"_h{int(args.min_history)}"
            )
            _write_csv(path.with_name(f"{path.stem}_{suffix}_frames.csv"), frame_rows)
            _write_csv(path.with_name(f"{path.stem}_{suffix}_summary.csv"), summary_rows)
            _write_csv(path.with_name(f"{path.stem}_{suffix}_points.csv"), point_rows)

    if not bool(args.no_csv) and all_summary_rows:
        out_dir = paths[0].resolve().parent
        _write_csv(out_dir / "hydramarker_residual_jitter_replay_summary.csv", all_summary_rows)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[residual_jitter_replay] ERROR: {exc}")
        sys.exit(1)
