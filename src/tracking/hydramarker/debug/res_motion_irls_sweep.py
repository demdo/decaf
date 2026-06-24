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
    _percentile,
    _pose_to_T,
    _project_points,
    _to_float,
    arrays_for_observation,
    arrays_for_window,
    build_age_maps,
    build_point_priors,
    build_static_uv_model,
    load_run,
    solve_irls_lie_pose,
)


DEFAULT_RUNS = (
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_fb.jsonl",
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_rl.jsonl",
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_bf_rot.jsonl",
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_static.jsonl",
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_static_do.jsonl",
)

COMPONENTS = ("x", "y", "z")


def _parse_list(value: str, cast) -> list[Any]:
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
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


def _rms(values: Iterable[float]) -> float:
    arr = _finite(values)
    if len(arr) == 0:
        return math.nan
    return float(math.sqrt(float(np.mean(arr * arr))))


def _weighted_rms_residual(
    object_points: np.ndarray,
    image_points: np.ndarray,
    weights: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    if len(object_points) == 0:
        return math.nan
    projected = _project_points(object_points, T, K, dist)
    residual = projected.reshape(-1, 2) - np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    err2 = np.sum(residual * residual, axis=1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        return float(math.sqrt(float(np.mean(err2))))
    return float(math.sqrt(float(np.sum(weights * err2) / np.sum(weights))))


def _best_lag_metrics(reference: np.ndarray, candidate: np.ndarray, *, max_lag: int) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    valid = np.isfinite(ref) & np.isfinite(cand)
    ref = ref[valid]
    cand = cand[valid]
    if len(ref) < 8:
        return {"best_lag_steps": math.nan, "best_lag_corr": math.nan, "best_lag_rmse_mm": math.nan}

    best: tuple[float, int, float] | None = None
    max_lag = max(int(max_lag), 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            r = ref[-lag:]
            c = cand[:lag]
        elif lag > 0:
            r = ref[:-lag]
            c = cand[lag:]
        else:
            r = ref
            c = cand
        if len(r) < 8:
            continue
        r0 = r - float(np.mean(r))
        c0 = c - float(np.mean(c))
        denom = float(np.linalg.norm(r0) * np.linalg.norm(c0))
        corr = float(np.dot(r0, c0) / denom) if denom > 1e-12 else math.nan
        rmse = _rms(c - r)
        key = -corr if np.isfinite(corr) else math.inf
        if best is None or key < best[0]:
            best = (key, lag, rmse)
    if best is None:
        return {"best_lag_steps": math.nan, "best_lag_corr": math.nan, "best_lag_rmse_mm": math.nan}
    lag = int(best[1])
    if lag < 0:
        r = ref[-lag:]
        c = cand[:lag]
    elif lag > 0:
        r = ref[:-lag]
        c = cand[lag:]
    else:
        r = ref
        c = cand
    r0 = r - float(np.mean(r))
    c0 = c - float(np.mean(c))
    denom = float(np.linalg.norm(r0) * np.linalg.norm(c0))
    corr = float(np.dot(r0, c0) / denom) if denom > 1e-12 else math.nan
    return {
        "best_lag_steps": float(lag),
        "best_lag_corr": float(corr),
        "best_lag_rmse_mm": float(best[2]),
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
    max_step_translation_mm: float,
    max_step_rotation_deg: float,
    frame_stride: int,
    max_lag_frames: int,
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
        max_step_translation_mm=float(max_step_translation_mm),
        max_step_rotation_deg=float(max_step_rotation_deg),
    )
    priors = build_point_priors(observations, static_model, config)
    age_maps = build_age_maps(observations, age_ramp_frames=config.age_ramp_frames)

    rows: list[dict[str, Any]] = []
    eval_indices = _eval_indices(len(observations), frame_stride)
    for obs_idx in eval_indices:
        obs = observations[obs_idx]
        win_obj, win_uv, win_weights, win_keys, used_frame_count = arrays_for_window(
            observations,
            obs_idx,
            priors=priors,
            age_maps=age_maps,
            window_frames=int(window_frames),
            window_decay=float(window_decay),
        )
        result = solve_irls_lie_pose(
            win_obj,
            win_uv,
            win_weights,
            run["K"],
            run["dist"],
            T,
            config,
        )
        if result.success:
            T = result.T.copy()
        _rvec, tvec = _T_to_pose(T)

        cur_obj, cur_uv, cur_weights, cur_keys = arrays_for_observation(
            obs,
            priors=priors,
            age_weight_by_key=age_maps[obs_idx],
        )
        cand_current_wrms = _weighted_rms_residual(
            cur_obj,
            cur_uv,
            cur_weights,
            T,
            run["K"],
            run["dist"],
        )
        logged_T = _pose_to_T(obs.original_rvec.copy(), obs.original_tvec.copy())
        logged_current_wrms = _weighted_rms_residual(
            cur_obj,
            cur_uv,
            cur_weights,
            logged_T,
            run["K"],
            run["dist"],
        )
        delta_logged = np.asarray(tvec, dtype=np.float64).reshape(3) - obs.original_tvec.reshape(3)
        rows.append(
            {
                "frame": int(obs.frame),
                "tvec_x_mm": float(tvec[0]),
                "tvec_y_mm": float(tvec[1]),
                "tvec_z_mm": float(tvec[2]),
                "logged_x_mm": float(obs.original_tvec[0]),
                "logged_y_mm": float(obs.original_tvec[1]),
                "logged_z_mm": float(obs.original_tvec[2]),
                "delta_logged_mm": float(np.linalg.norm(delta_logged)),
                "point_count": int(result.point_count),
                "current_point_count": int(len(cur_obj)),
                "distinct_key_count": int(len(set(win_keys))),
                "used_frame_count": int(used_frame_count),
                "solved": int(bool(result.success)),
                "iterations": int(result.iterations),
                "window_reproj_weighted_rms_px": _to_float(result.stats.get("reproj_weighted_rms_px")),
                "current_reproj_weighted_rms_px": float(cand_current_wrms),
                "logged_current_reproj_weighted_rms_px": float(logged_current_wrms),
            }
        )

    cand_t = np.asarray(
        [[row["tvec_x_mm"], row["tvec_y_mm"], row["tvec_z_mm"]] for row in rows],
        dtype=np.float64,
    ).reshape(-1, 3)
    logged_t = np.asarray(
        [[row["logged_x_mm"], row["logged_y_mm"], row["logged_z_mm"]] for row in rows],
        dtype=np.float64,
    ).reshape(-1, 3)
    cand_rel = cand_t - cand_t[0].reshape(1, 3)
    logged_rel = logged_t - logged_t[0].reshape(1, 3)
    logged_ranges = np.nanmax(logged_rel, axis=0) - np.nanmin(logged_rel, axis=0)
    cand_ranges = np.nanmax(cand_rel, axis=0) - np.nanmin(cand_rel, axis=0)
    movement_axis_idx = int(np.nanargmax(logged_ranges))
    movement_axis = COMPONENTS[movement_axis_idx]
    logged_axis_range = float(logged_ranges[movement_axis_idx])
    cand_axis_range = float(cand_ranges[movement_axis_idx])
    dynamic = bool(logged_axis_range >= 20.0)

    lag = _best_lag_metrics(
        logged_rel[:, movement_axis_idx],
        cand_rel[:, movement_axis_idx],
        max_lag=max(1, int(math.ceil(float(max_lag_frames) / max(float(frame_stride), 1.0)))),
    )

    current_wrms = [_to_float(row.get("current_reproj_weighted_rms_px")) for row in rows]
    logged_wrms = [_to_float(row.get("logged_current_reproj_weighted_rms_px")) for row in rows]
    delta_logged = [_to_float(row.get("delta_logged_mm")) for row in rows]
    reproj_excess = [
        max(0.0, _to_float(a) - _to_float(b))
        for a, b in zip(current_wrms, logged_wrms)
        if np.isfinite(_to_float(a)) and np.isfinite(_to_float(b))
    ]
    amplitude_ratio = cand_axis_range / logged_axis_range if logged_axis_range > 1e-9 else math.nan

    return {
        "run_id": str(run["run_id"]),
        "run_label": str(Path(run["path"]).stem),
        "is_dynamic": int(dynamic),
        "frames_evaluated": int(len(rows)),
        "window_frames": int(window_frames),
        "window_decay": float(window_decay),
        "robust_c_px": float(robust_c_px),
        "uv_stability_scale_px": float(uv_stability_scale_px),
        "condition_boost": float(condition_boost),
        "age_ramp_frames": int(age_ramp_frames),
        "max_iterations": int(max_iterations),
        "max_step_translation_mm": float(max_step_translation_mm),
        "max_step_rotation_deg": float(max_step_rotation_deg),
        "frame_stride": int(frame_stride),
        "solve_failures": int(sum(1 for row in rows if int(row.get("solved", 0)) == 0)),
        "movement_axis": movement_axis,
        "logged_axis_range_mm": float(logged_axis_range),
        "candidate_axis_range_mm": float(cand_axis_range),
        "amplitude_ratio": float(amplitude_ratio),
        "amplitude_error": abs(float(amplitude_ratio) - 1.0) if np.isfinite(amplitude_ratio) else math.nan,
        "best_lag_eval_steps": _to_float(lag.get("best_lag_steps")),
        "best_lag_frames": _to_float(lag.get("best_lag_steps")) * float(frame_stride),
        "best_lag_corr": _to_float(lag.get("best_lag_corr")),
        "best_lag_rmse_mm": _to_float(lag.get("best_lag_rmse_mm")),
        "x_range_mm": _range(cand_rel[:, 0]),
        "y_range_mm": _range(cand_rel[:, 1]),
        "z_range_mm": _range(cand_rel[:, 2]),
        "z_closure_mm": float(cand_rel[-1, 2] - cand_rel[0, 2]),
        "delta_logged_median_mm": _median(delta_logged),
        "delta_logged_p95_mm": _percentile(delta_logged, 95),
        "current_wrms_median_px": _median(current_wrms),
        "logged_wrms_median_px": _median(logged_wrms),
        "reproj_excess_median_px": _median(reproj_excess),
        "reproj_excess_p95_px": _percentile(reproj_excess, 95),
        "point_count_median": _median([_to_float(row.get("point_count")) for row in rows]),
        "used_frame_count_median": _median([_to_float(row.get("used_frame_count")) for row in rows]),
    }


def _score(run_rows: list[dict[str, Any]]) -> dict[str, float]:
    dynamic = [row for row in run_rows if int(_to_float(row.get("is_dynamic"))) == 1]
    static = [row for row in run_rows if int(_to_float(row.get("is_dynamic"))) == 0]

    dyn_amp = [_to_float(row.get("amplitude_error")) for row in dynamic]
    dyn_lag = [abs(_to_float(row.get("best_lag_frames"))) for row in dynamic]
    dyn_delta = [_to_float(row.get("delta_logged_p95_mm")) for row in dynamic]
    dyn_reproj = [_to_float(row.get("reproj_excess_p95_px")) for row in dynamic]
    dyn_rmse = [_to_float(row.get("best_lag_rmse_mm")) for row in dynamic]

    static_z = [_to_float(row.get("z_range_mm")) for row in static]
    static_closure = [abs(_to_float(row.get("z_closure_mm"))) for row in static]
    static_reproj = [_to_float(row.get("reproj_excess_p95_px")) for row in static]

    max_dyn_amp = float(np.nanmax(dyn_amp)) if dyn_amp else 0.0
    max_dyn_lag = float(np.nanmax(dyn_lag)) if dyn_lag else 0.0
    max_dyn_delta = float(np.nanmax(dyn_delta)) if dyn_delta else 0.0
    max_dyn_reproj = float(np.nanmax(dyn_reproj)) if dyn_reproj else 0.0
    max_dyn_rmse = float(np.nanmax(dyn_rmse)) if dyn_rmse else 0.0
    max_static_z = float(np.nanmax(static_z)) if static_z else 0.0
    sum_static_z = float(np.nansum(static_z)) if static_z else 0.0
    max_static_closure = float(np.nanmax(static_closure)) if static_closure else 0.0
    max_static_reproj = float(np.nanmax(static_reproj)) if static_reproj else 0.0

    score = (
        1.8 * max_dyn_amp
        + 0.06 * max_dyn_lag
        + 0.035 * max_dyn_delta
        + 0.60 * max_dyn_reproj
        + 0.015 * max_dyn_rmse
        + 0.85 * max_static_z
        + 0.20 * sum_static_z
        + 0.10 * max_static_closure
        + 0.35 * max_static_reproj
    )
    return {
        "score": float(score),
        "max_dynamic_amplitude_error": max_dyn_amp,
        "max_dynamic_abs_lag_frames": max_dyn_lag,
        "max_dynamic_delta_logged_p95_mm": max_dyn_delta,
        "max_dynamic_reproj_excess_p95_px": max_dyn_reproj,
        "max_dynamic_best_lag_rmse_mm": max_dyn_rmse,
        "max_static_z_range_mm": max_static_z,
        "sum_static_z_range_mm": sum_static_z,
        "max_static_abs_z_closure_mm": max_static_closure,
        "max_static_reproj_excess_p95_px": max_static_reproj,
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
        "window_frames": "5,10,15,20,30,45",
        "window_decay": "0,6,12,30",
        "robust_c_px": "0.1,0.2,0.4",
        "uv_stability_scale_px": "0.05,0.08",
        "condition_boost": "0",
        "age_ramp_frames": "1,4",
        "max_iterations": 6,
        "max_step_translation_mm": "5,15,40",
        "max_step_rotation_deg": "3,8",
        "frame_stride": 4,
        "max_lag_frames": 12,
        "tag": "motion_coarse",
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
        elif arg == "--max-step-translation-mm":
            idx += 1
            args["max_step_translation_mm"] = argv[idx]
        elif arg == "--max-step-rotation-deg":
            idx += 1
            args["max_step_rotation_deg"] = argv[idx]
        elif arg == "--frame-stride":
            idx += 1
            args["frame_stride"] = int(argv[idx])
        elif arg == "--max-lag-frames":
            idx += 1
            args["max_lag_frames"] = int(argv[idx])
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
    step_translations = _parse_list(args["max_step_translation_mm"], float)
    step_rotations = _parse_list(args["max_step_rotation_deg"], float)
    combos = list(
        itertools.product(
            windows,
            decays,
            robust_cs,
            uv_scales,
            boosts,
            age_ramps,
            step_translations,
            step_rotations,
        )
    )

    print(f"[motion_irls_sweep] runs={len(loaded_runs)} combos={len(combos)} tag={args['tag']}")
    run_rows: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for combo_idx, (
        window,
        decay,
        robust_c,
        uv_scale,
        boost,
        age_ramp,
        step_translation,
        step_rotation,
    ) in enumerate(combos, start=1):
        combo_rows = []
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
                max_step_translation_mm=float(step_translation),
                max_step_rotation_deg=float(step_rotation),
                frame_stride=int(args["frame_stride"]),
                max_lag_frames=int(args["max_lag_frames"]),
            )
            combo_rows.append(row)
            run_rows.append(row)

        scored = _score(combo_rows)
        combined_rows.append(
            {
                "rank": 0,
                "window_frames": int(window),
                "window_decay": float(decay),
                "robust_c_px": float(robust_c),
                "uv_stability_scale_px": float(uv_scale),
                "condition_boost": float(boost),
                "age_ramp_frames": int(age_ramp),
                "max_step_translation_mm": float(step_translation),
                "max_step_rotation_deg": float(step_rotation),
                "max_iterations": int(args["max_iterations"]),
                "frame_stride": int(args["frame_stride"]),
                **scored,
            }
        )

        if combo_idx == 1 or combo_idx % 10 == 0 or combo_idx == len(combos):
            elapsed = time.perf_counter() - t0
            remaining = (elapsed / float(combo_idx)) * float(len(combos) - combo_idx)
            best = min(combined_rows, key=lambda r: _to_float(r.get("score")))
            print(
                "[motion_irls_sweep] "
                f"{combo_idx}/{len(combos)} elapsed={elapsed:.1f}s remaining={remaining:.1f}s "
                f"best_score={_to_float(best.get('score')):.4f} "
                f"w={best['window_frames']} d={best['window_decay']} "
                f"c={best['robust_c_px']} uv={best['uv_stability_scale_px']} "
                f"age={best['age_ramp_frames']} "
                f"step={best['max_step_translation_mm']}/{best['max_step_rotation_deg']}"
            )

    combined_rows.sort(key=lambda row: (_to_float(row.get("score")), _to_float(row.get("max_static_z_range_mm"))))
    for rank, row in enumerate(combined_rows, start=1):
        row["rank"] = int(rank)

    out_dir = paths[0].resolve().parent
    tag = str(args["tag"]).strip() or "motion"
    combined_csv = out_dir / f"hydramarker_motion_irls_sweep_{tag}_combined.csv"
    runs_csv = out_dir / f"hydramarker_motion_irls_sweep_{tag}_runs.csv"
    _write_csv(combined_csv, combined_rows)
    _write_csv(runs_csv, run_rows)

    print(f"[motion_irls_sweep] saved combined -> {combined_csv.resolve()}")
    print(f"[motion_irls_sweep] saved runs     -> {runs_csv.resolve()}")
    print("[motion_irls_sweep] top 10:")
    for row in combined_rows[:10]:
        print(
            "  "
            f"#{int(row['rank'])}: score={_to_float(row.get('score')):.4f}, "
            f"dyn_amp={_to_float(row.get('max_dynamic_amplitude_error')):.3f}, "
            f"dyn_lag={_to_float(row.get('max_dynamic_abs_lag_frames')):.1f}, "
            f"dyn_delta={_to_float(row.get('max_dynamic_delta_logged_p95_mm')):.3f}, "
            f"dyn_reproj={_to_float(row.get('max_dynamic_reproj_excess_p95_px')):.3f}, "
            f"static_z={_to_float(row.get('max_static_z_range_mm')):.3f}, "
            f"w={row['window_frames']}, d={row['window_decay']}, "
            f"c={row['robust_c_px']}, uv={row['uv_stability_scale_px']}, "
            f"age={row['age_ramp_frames']}, "
            f"step={row['max_step_translation_mm']}/{row['max_step_rotation_deg']}"
        )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[motion_irls_sweep] ERROR: {exc}")
        sys.exit(1)
