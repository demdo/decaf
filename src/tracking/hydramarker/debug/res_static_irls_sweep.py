"""Parameter sweep for static IRLS pose replay experiments.

The script runs multiple static robust-refinement settings over logged
HydraMarker observations and writes aggregate comparison tables.
"""

from __future__ import annotations

import csv
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from res_static_irls_replay import (  # noqa: E402
    IrlsConfig,
    _T_to_pose,
    _median,
    _pose_to_T,
    _to_float,
    arrays_for_window,
    build_age_maps,
    build_point_priors,
    build_static_uv_model,
    load_run,
    solve_irls_lie_pose,
)


DEFAULT_RUNS = (
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_static.jsonl",
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_static_do.jsonl",
)


def _parse_list(value: str, cast) -> list[Any]:
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(cast(part))
    if not out:
        raise RuntimeError(f"Empty value list: {value!r}")
    return out


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _range(values: Iterable[float]) -> float:
    arr = _finite(values)
    if len(arr) == 0:
        return math.nan
    return float(np.max(arr) - np.min(arr))


def _score(run_rows: list[dict[str, Any]]) -> dict[str, float]:
    z_ranges = [_to_float(row.get("z_range_mm")) for row in run_rows]
    z_closures = [abs(_to_float(row.get("z_closure_mm"))) for row in run_rows]
    reproj = [_to_float(row.get("reproj_weighted_rms_median_px")) for row in run_rows]
    max_z = float(np.nanmax(z_ranges))
    sum_z = float(np.nansum(z_ranges))
    max_closure = float(np.nanmax(z_closures))
    max_reproj = float(np.nanmax(reproj))
    score = max_z + 0.35 * sum_z + 0.20 * max_closure + 0.05 * max_reproj
    return {
        "score": float(score),
        "max_z_range_mm": max_z,
        "sum_z_range_mm": sum_z,
        "max_abs_z_closure_mm": max_closure,
        "max_weighted_reproj_rms_px": max_reproj,
    }


def _eval_indices(num_frames: int, stride: int) -> list[int]:
    stride = max(int(stride), 1)
    indices = list(range(0, int(num_frames), stride))
    if not indices or indices[-1] != num_frames - 1:
        indices.append(num_frames - 1)
    return sorted(set(indices))


def evaluate_run(
    run: dict[str, Any],
    static_model: dict[str, Any],
    *,
    window_frames: int,
    window_decay: float,
    robust_c_px: float,
    uv_stability_scale_px: float,
    condition_boost: float,
    age_ramp_frames: int,
    max_iterations: int,
    frame_stride: int,
) -> dict[str, Any]:
    observations = list(run["observations"])
    reference = observations[0]
    T = _pose_to_T(reference.original_rvec.copy(), reference.original_tvec.copy())

    config = IrlsConfig(
        max_iterations=int(max_iterations),
        robust_c_px=float(robust_c_px),
        uv_stability_scale_px=float(uv_stability_scale_px),
        condition_boost=float(condition_boost),
        age_ramp_frames=int(age_ramp_frames),
    )
    priors = build_point_priors(observations, static_model, config)
    age_maps = build_age_maps(observations, age_ramp_frames=config.age_ramp_frames)

    rows: list[dict[str, Any]] = []
    for obs_idx in _eval_indices(len(observations), frame_stride):
        obj, uv, weights, keys, used_frame_count = arrays_for_window(
            observations,
            obs_idx,
            priors=priors,
            age_maps=age_maps,
            window_frames=int(window_frames),
            window_decay=float(window_decay),
        )
        result = solve_irls_lie_pose(
            obj,
            uv,
            weights,
            run["K"],
            run["dist"],
            T,
            config,
        )
        if result.success:
            T = result.T.copy()
        _rvec, tvec = _T_to_pose(T)
        rows.append(
            {
                "frame": int(observations[obs_idx].frame),
                "tvec_x_mm": float(tvec[0]),
                "tvec_y_mm": float(tvec[1]),
                "tvec_z_mm": float(tvec[2]),
                "point_count": int(result.point_count),
                "distinct_key_count": int(len(set(keys))),
                "used_frame_count": int(used_frame_count),
                "solved": int(bool(result.success)),
                "iterations": int(result.iterations),
                "mean_weight": float(result.mean_weight),
                "condition_number": float(result.condition_number),
                "weak_z_alignment": float(result.weak_z_alignment),
                "reproj_weighted_rms_px": _to_float(result.stats.get("reproj_weighted_rms_px")),
                "reproj_rms_px": _to_float(result.stats.get("reproj_rms_px")),
            }
        )

    tvecs = np.asarray(
        [[row["tvec_x_mm"], row["tvec_y_mm"], row["tvec_z_mm"]] for row in rows],
        dtype=np.float64,
    ).reshape(-1, 3)
    rel = tvecs - tvecs[0].reshape(1, 3)
    return {
        "run_id": str(run["run_id"]),
        "frames_evaluated": int(len(rows)),
        "window_frames": int(window_frames),
        "window_decay": float(window_decay),
        "robust_c_px": float(robust_c_px),
        "uv_stability_scale_px": float(uv_stability_scale_px),
        "condition_boost": float(condition_boost),
        "age_ramp_frames": int(age_ramp_frames),
        "max_iterations": int(max_iterations),
        "frame_stride": int(frame_stride),
        "solve_failures": int(sum(1 for row in rows if int(row.get("solved", 0)) == 0)),
        "x_range_mm": _range(rel[:, 0]),
        "y_range_mm": _range(rel[:, 1]),
        "z_range_mm": _range(rel[:, 2]),
        "x_closure_mm": float(rel[-1, 0] - rel[0, 0]),
        "y_closure_mm": float(rel[-1, 1] - rel[0, 1]),
        "z_closure_mm": float(rel[-1, 2] - rel[0, 2]),
        "point_count_median": _median([_to_float(row.get("point_count")) for row in rows]),
        "distinct_key_count_median": _median(
            [_to_float(row.get("distinct_key_count")) for row in rows]
        ),
        "used_frame_count_median": _median([_to_float(row.get("used_frame_count")) for row in rows]),
        "condition_number_median": _median([_to_float(row.get("condition_number")) for row in rows]),
        "weak_z_alignment_median": _median([_to_float(row.get("weak_z_alignment")) for row in rows]),
        "mean_weight_median": _median([_to_float(row.get("mean_weight")) for row in rows]),
        "reproj_weighted_rms_median_px": _median(
            [_to_float(row.get("reproj_weighted_rms_px")) for row in rows]
        ),
        "reproj_rms_median_px": _median([_to_float(row.get("reproj_rms_px")) for row in rows]),
    }


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


def _load_runs(paths: list[Path], point_set: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    loaded = []
    for path in paths:
        run = load_run(path.resolve(), point_set=point_set)
        loaded.append((run, build_static_uv_model(run["observations"])))
    return loaded


def _parse_args(argv: list[str]) -> dict[str, Any]:
    args: dict[str, Any] = {
        "paths": [],
        "point_set": "correspondence",
        "window_frames": "10,15,20,30,45,60",
        "window_decay": "0,6,12,20,40",
        "robust_c_px": "0.1,0.2,0.4",
        "uv_stability_scale_px": "0.05,0.08,0.12",
        "condition_boost": "0,0.5,1",
        "age_ramp_frames": "1,4",
        "max_iterations": 6,
        "frame_stride": 3,
        "tag": "coarse",
    }
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--point-set":
            idx += 1
            args["point_set"] = argv[idx]
        elif arg == "--window-frames":
            idx += 1
            args["window_frames"] = argv[idx]
        elif arg == "--window-decay":
            idx += 1
            args["window_decay"] = argv[idx]
        elif arg == "--robust-c-px":
            idx += 1
            args["robust_c_px"] = argv[idx]
        elif arg == "--uv-stability-scale-px":
            idx += 1
            args["uv_stability_scale_px"] = argv[idx]
        elif arg == "--condition-boost":
            idx += 1
            args["condition_boost"] = argv[idx]
        elif arg == "--age-ramp-frames":
            idx += 1
            args["age_ramp_frames"] = argv[idx]
        elif arg == "--max-iterations":
            idx += 1
            args["max_iterations"] = int(argv[idx])
        elif arg == "--frame-stride":
            idx += 1
            args["frame_stride"] = int(argv[idx])
        elif arg == "--tag":
            idx += 1
            args["tag"] = argv[idx]
        elif arg.endswith(".jsonl"):
            args["paths"].append(Path(arg))
        else:
            raise RuntimeError(f"Unknown option: {arg}")
        idx += 1

    if not args["paths"]:
        args["paths"] = [Path(p) for p in DEFAULT_RUNS]
    return args


def main() -> None:
    args = _parse_args(sys.argv[1:])
    paths = [Path(p) for p in args["paths"]]
    loaded_runs = _load_runs(paths, str(args["point_set"]).strip().lower())

    windows = _parse_list(args["window_frames"], int)
    decays = _parse_list(args["window_decay"], float)
    robust_cs = _parse_list(args["robust_c_px"], float)
    uv_scales = _parse_list(args["uv_stability_scale_px"], float)
    boosts = _parse_list(args["condition_boost"], float)
    age_ramps = _parse_list(args["age_ramp_frames"], int)

    combos = list(itertools.product(windows, decays, robust_cs, uv_scales, boosts, age_ramps))
    print(f"[irls_sweep] runs={len(loaded_runs)} combos={len(combos)} tag={args['tag']}")

    run_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for combo_idx, (window, decay, robust_c, uv_scale, boost, age_ramp) in enumerate(combos, start=1):
        combo_run_rows: list[dict[str, Any]] = []
        for run, static_model in loaded_runs:
            row = evaluate_run(
                run,
                static_model,
                window_frames=int(window),
                window_decay=float(decay),
                robust_c_px=float(robust_c),
                uv_stability_scale_px=float(uv_scale),
                condition_boost=float(boost),
                age_ramp_frames=int(age_ramp),
                max_iterations=int(args["max_iterations"]),
                frame_stride=int(args["frame_stride"]),
            )
            combo_run_rows.append(row)
            run_rows.append(row)

        scored = _score(combo_run_rows)
        combined_rows.append(
            {
                "rank": 0,
                "window_frames": int(window),
                "window_decay": float(decay),
                "robust_c_px": float(robust_c),
                "uv_stability_scale_px": float(uv_scale),
                "condition_boost": float(boost),
                "age_ramp_frames": int(age_ramp),
                "max_iterations": int(args["max_iterations"]),
                "frame_stride": int(args["frame_stride"]),
                **scored,
            }
        )

        if combo_idx == 1 or combo_idx % 10 == 0 or combo_idx == len(combos):
            elapsed = time.perf_counter() - t0
            rate = elapsed / float(combo_idx)
            remaining = rate * float(len(combos) - combo_idx)
            best = min(combined_rows, key=lambda r: _to_float(r.get("score")))
            print(
                "[irls_sweep] "
                f"{combo_idx}/{len(combos)} elapsed={elapsed:.1f}s remaining={remaining:.1f}s "
                f"best_score={_to_float(best.get('score')):.4f} "
                f"best_w={best['window_frames']} d={best['window_decay']} "
                f"c={best['robust_c_px']} uv={best['uv_stability_scale_px']} "
                f"cb={best['condition_boost']} age={best['age_ramp_frames']}"
            )

    combined_rows.sort(key=lambda row: (_to_float(row.get("score")), _to_float(row.get("max_z_range_mm"))))
    for rank, row in enumerate(combined_rows, start=1):
        row["rank"] = int(rank)

    out_dir = paths[0].resolve().parent
    tag = str(args["tag"]).strip() or "sweep"
    combined_csv = out_dir / f"hydramarker_static_irls_sweep_{tag}_combined.csv"
    runs_csv = out_dir / f"hydramarker_static_irls_sweep_{tag}_runs.csv"
    _write_csv(combined_csv, combined_rows)
    _write_csv(runs_csv, run_rows)

    print(f"[irls_sweep] saved combined -> {combined_csv.resolve()}")
    print(f"[irls_sweep] saved runs     -> {runs_csv.resolve()}")
    print("[irls_sweep] top 10:")
    for row in combined_rows[:10]:
        print(
            "  "
            f"#{int(row['rank'])}: score={_to_float(row.get('score')):.4f}, "
            f"max_z={_to_float(row.get('max_z_range_mm')):.3f}, "
            f"sum_z={_to_float(row.get('sum_z_range_mm')):.3f}, "
            f"closure={_to_float(row.get('max_abs_z_closure_mm')):.3f}, "
            f"w={row['window_frames']}, d={row['window_decay']}, "
            f"c={row['robust_c_px']}, uv={row['uv_stability_scale_px']}, "
            f"cb={row['condition_boost']}, age={row['age_ramp_frames']}"
        )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[irls_sweep] ERROR: {exc}")
        sys.exit(1)
