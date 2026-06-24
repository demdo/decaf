from __future__ import annotations

import csv
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from res_motion_irls_adaptive import _plot_adaptive_translation  # noqa: E402
from res_motion_irls_sweep import (  # noqa: E402
    _best_lag_metrics,
    _eval_indices,
    _range,
    _weighted_rms_residual,
)
from res_static_irls_replay import (  # noqa: E402
    IrlsConfig,
    _T_to_pose,
    _median,
    _percentile,
    _pose_to_T,
    _to_float,
    arrays_for_observation,
    build_age_maps,
    build_point_priors,
    build_static_uv_model,
    load_run,
    solve_irls_lie_pose,
)


DEFAULT_RUN = Path("hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_fb.jsonl")

# Paper-style Lie algebra motion subspaces. Order is:
# tx, ty, tz, rx, ry, rz.
DOF_SETS: dict[str, tuple[int, ...] | None] = {
    "free6": None,
    "ty": (1,),
    "ty_rz": (1, 5),
    "ty_rot": (1, 3, 4, 5),
    "xy": (0, 1),
    "xy_rz": (0, 1, 5),
    "xy_rot": (0, 1, 3, 4, 5),
    "xyz_no_rot": (0, 1, 2),
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _score_fb(row: dict[str, Any]) -> float:
    z_range = _to_float(row.get("z_range_mm"))
    x_range = _to_float(row.get("x_range_mm"))
    amp_error = _to_float(row.get("amplitude_error"))
    lag = abs(_to_float(row.get("best_lag_frames")))
    reproj = _to_float(row.get("reproj_excess_p95_px"))
    delta = _to_float(row.get("delta_logged_p95_mm"))
    y_range = _to_float(row.get("candidate_axis_range_mm"))
    raw_y_range = _to_float(row.get("logged_axis_range_mm"))

    values = [z_range, x_range, amp_error, lag, reproj, delta, y_range, raw_y_range]
    if not all(np.isfinite(v) for v in values):
        return 1.0e12

    penalty = 0.0
    if y_range < 0.98 * raw_y_range:
        penalty += 1.0e6 * (0.98 * raw_y_range - y_range)
    if lag > 2.0:
        penalty += 1.0e6 * (lag - 2.0)
    if reproj > 0.75:
        penalty += 1.0e5 * (reproj - 0.75)

    score = z_range
    score += 0.20 * x_range
    score += 180.0 * max(0.0, amp_error - 0.001)
    score += 1.50 * lag
    score += 5.0 * max(0.0, reproj - 0.03)
    score += 0.10 * max(0.0, delta - 0.50)
    score += penalty
    return float(score)


def evaluate_run(
    run: dict[str, Any],
    *,
    dof_name: str,
    robust_c_px: float,
    uv_stability_scale_px: float,
    condition_boost: float,
    max_step_translation_mm: float,
    max_step_rotation_deg: float,
    max_iterations: int,
    frame_stride: int,
    include_rows: bool = False,
) -> dict[str, Any]:
    observations = list(run["observations"])
    reference = observations[0]
    T = _pose_to_T(reference.original_rvec.copy(), reference.original_tvec.copy())
    config = IrlsConfig(
        max_iterations=int(max_iterations),
        robust_c_px=float(robust_c_px),
        uv_stability_scale_px=float(uv_stability_scale_px),
        condition_boost=float(condition_boost),
        age_ramp_frames=1,
        max_step_translation_mm=float(max_step_translation_mm),
        max_step_rotation_deg=float(max_step_rotation_deg),
        active_dofs=DOF_SETS[dof_name],
    )
    static_model = build_static_uv_model(observations)
    priors = build_point_priors(observations, static_model, config)
    age_maps = build_age_maps(observations, age_ramp_frames=1)

    rows: list[dict[str, Any]] = []
    for obs_idx in _eval_indices(len(observations), frame_stride):
        obs = observations[obs_idx]
        obj, uv, weights, keys = arrays_for_observation(
            obs,
            priors=priors,
            age_weight_by_key=age_maps[obs_idx],
        )
        result = solve_irls_lie_pose(obj, uv, weights, run["K"], run["dist"], T, config)
        if result.success:
            T = result.T.copy()
        _rvec, tvec = _T_to_pose(T)

        cur_obj, cur_uv, cur_weights, _cur_keys = arrays_for_observation(
            obs,
            priors=priors,
            age_weight_by_key=age_maps[obs_idx],
        )
        current_wrms = _weighted_rms_residual(cur_obj, cur_uv, cur_weights, T, run["K"], run["dist"])
        logged_T = _pose_to_T(obs.original_rvec.copy(), obs.original_tvec.copy())
        logged_wrms = _weighted_rms_residual(cur_obj, cur_uv, cur_weights, logged_T, run["K"], run["dist"])
        delta_logged = np.asarray(tvec, dtype=np.float64).reshape(3) - obs.original_tvec.reshape(3)
        rows.append(
            {
                "frame": int(obs.frame),
                "static_candidate": 0,
                "use_static": 0,
                "guard_reject": 0,
                "tvec_x_mm": float(tvec[0]),
                "tvec_y_mm": float(tvec[1]),
                "tvec_z_mm": float(tvec[2]),
                "logged_x_mm": float(obs.original_tvec[0]),
                "logged_y_mm": float(obs.original_tvec[1]),
                "logged_z_mm": float(obs.original_tvec[2]),
                "delta_logged_mm": float(np.linalg.norm(delta_logged)),
                "point_count": int(result.point_count),
                "used_frame_count": 1,
                "solved": int(bool(result.success)),
                "iterations": int(result.iterations),
                "current_reproj_weighted_rms_px": float(current_wrms),
                "logged_current_reproj_weighted_rms_px": float(logged_wrms),
            }
        )

    cand_t = np.asarray([[r["tvec_x_mm"], r["tvec_y_mm"], r["tvec_z_mm"]] for r in rows], dtype=np.float64)
    logged_t = np.asarray([[r["logged_x_mm"], r["logged_y_mm"], r["logged_z_mm"]] for r in rows], dtype=np.float64)
    cand_rel = cand_t - cand_t[0].reshape(1, 3)
    logged_rel = logged_t - logged_t[0].reshape(1, 3)
    logged_ranges = np.nanmax(logged_rel, axis=0) - np.nanmin(logged_rel, axis=0)
    cand_ranges = np.nanmax(cand_rel, axis=0) - np.nanmin(cand_rel, axis=0)
    movement_axis_idx = int(np.nanargmax(logged_ranges))
    lag = _best_lag_metrics(logged_rel[:, movement_axis_idx], cand_rel[:, movement_axis_idx], max_lag=12)
    current_wrms = [_to_float(r.get("current_reproj_weighted_rms_px")) for r in rows]
    logged_wrms = [_to_float(r.get("logged_current_reproj_weighted_rms_px")) for r in rows]
    reproj_excess = [
        max(0.0, _to_float(a) - _to_float(b))
        for a, b in zip(current_wrms, logged_wrms)
        if np.isfinite(_to_float(a)) and np.isfinite(_to_float(b))
    ]
    delta_logged = [_to_float(r.get("delta_logged_mm")) for r in rows]
    logged_axis_range = float(logged_ranges[movement_axis_idx])
    candidate_axis_range = float(cand_ranges[movement_axis_idx])
    amplitude_ratio = candidate_axis_range / logged_axis_range if logged_axis_range > 1e-9 else math.nan

    summary: dict[str, Any] = {
        "run_label": str(Path(run["path"]).stem),
        "dof_name": dof_name,
        "active_dofs": "free6" if DOF_SETS[dof_name] is None else ",".join(str(v) for v in DOF_SETS[dof_name]),
        "robust_c_px": float(robust_c_px),
        "uv_stability_scale_px": float(uv_stability_scale_px),
        "condition_boost": float(condition_boost),
        "max_step_translation_mm": float(max_step_translation_mm),
        "max_step_rotation_deg": float(max_step_rotation_deg),
        "max_iterations": int(max_iterations),
        "frame_stride": int(frame_stride),
        "movement_axis": "xyz"[movement_axis_idx],
        "logged_axis_range_mm": float(logged_axis_range),
        "candidate_axis_range_mm": float(candidate_axis_range),
        "amplitude_ratio": float(amplitude_ratio),
        "amplitude_error": abs(float(amplitude_ratio) - 1.0) if np.isfinite(amplitude_ratio) else math.nan,
        "best_lag_frames": _to_float(lag.get("best_lag_steps")) * float(frame_stride),
        "best_lag_corr": _to_float(lag.get("best_lag_corr")),
        "best_lag_rmse_mm": _to_float(lag.get("best_lag_rmse_mm")),
        "x_range_mm": _range(cand_rel[:, 0]),
        "y_range_mm": _range(cand_rel[:, 1]),
        "z_range_mm": _range(cand_rel[:, 2]),
        "raw_x_range_mm": float(logged_ranges[0]),
        "raw_y_range_mm": float(logged_ranges[1]),
        "raw_z_range_mm": float(logged_ranges[2]),
        "z_closure_mm": float(cand_rel[-1, 2] - cand_rel[0, 2]),
        "delta_logged_median_mm": _median(delta_logged),
        "delta_logged_p95_mm": _percentile(delta_logged, 95),
        "delta_logged_max_mm": float(np.nanmax(delta_logged)) if len(delta_logged) else math.nan,
        "current_wrms_median_px": _median(current_wrms),
        "logged_wrms_median_px": _median(logged_wrms),
        "reproj_excess_median_px": _median(reproj_excess),
        "reproj_excess_p95_px": _percentile(reproj_excess, 95),
        "reproj_excess_max_px": float(np.nanmax(reproj_excess)) if len(reproj_excess) else math.nan,
        "point_count_median": _median([_to_float(r.get("point_count")) for r in rows]),
    }
    summary["fb_score"] = _score_fb(summary)
    if include_rows:
        summary["_frame_rows"] = rows
    return summary


def _candidate_grid() -> list[tuple[Any, ...]]:
    dofs = list(DOF_SETS)
    robust = [0.05, 0.10, 0.20, 0.40]
    uv_scales = [0.05, 0.20, 999.0]
    boosts = [0.0, 1.0]
    step_t = [2.0, 5.0, 8.0, 12.0]
    step_r = [1.0, 3.0, 5.0]
    iterations = [1, 3, 6]
    return list(itertools.product(dofs, robust, uv_scales, boosts, step_t, step_r, iterations))


def main() -> None:
    args = sys.argv[1:]
    path = Path(args[0]) if args and args[0].endswith(".jsonl") else DEFAULT_RUN
    run = load_run(path.resolve(), point_set="correspondence")
    out_dir = path.resolve().parent

    coarse_stride = 5
    full_top_n = 120
    grid = _candidate_grid()
    print(f"[fb_constraint_search] run={path} coarse_combos={len(grid)} stride={coarse_stride}")
    t0 = time.perf_counter()
    coarse_rows: list[dict[str, Any]] = []
    for idx, (dof, c, uv, boost, step_t, step_r, iters) in enumerate(grid, start=1):
        row = evaluate_run(
            run,
            dof_name=str(dof),
            robust_c_px=float(c),
            uv_stability_scale_px=float(uv),
            condition_boost=float(boost),
            max_step_translation_mm=float(step_t),
            max_step_rotation_deg=float(step_r),
            max_iterations=int(iters),
            frame_stride=coarse_stride,
        )
        coarse_rows.append(row)
        if idx == 1 or idx % 100 == 0 or idx == len(grid):
            best = min(coarse_rows, key=lambda r: _to_float(r.get("fb_score")))
            elapsed = time.perf_counter() - t0
            remaining = elapsed / float(idx) * float(len(grid) - idx)
            print(
                f"[fb_constraint_search] {idx}/{len(grid)} elapsed={elapsed:.1f}s "
                f"remaining={remaining:.1f}s best_score={_to_float(best['fb_score']):.3f} "
                f"dof={best['dof_name']} z={_to_float(best['z_range_mm']):.3f} "
                f"y={_to_float(best['y_range_mm']):.3f} "
                f"reproj95={_to_float(best['reproj_excess_p95_px']):.3f}"
            )

    coarse_rows.sort(key=lambda r: (_to_float(r.get("fb_score")), _to_float(r.get("z_range_mm"))))
    for rank, row in enumerate(coarse_rows, start=1):
        row["coarse_rank"] = rank
    coarse_csv = out_dir / "hydramarker_fb_motion_constraint_search_coarse.csv"
    _write_csv(coarse_csv, coarse_rows)
    print(f"[fb_constraint_search] saved coarse -> {coarse_csv}")

    full_candidates = coarse_rows[:full_top_n]
    full_rows: list[dict[str, Any]] = []
    t1 = time.perf_counter()
    print(f"[fb_constraint_search] full_eval={len(full_candidates)} stride=1")
    for idx, cand in enumerate(full_candidates, start=1):
        row = evaluate_run(
            run,
            dof_name=str(cand["dof_name"]),
            robust_c_px=float(cand["robust_c_px"]),
            uv_stability_scale_px=float(cand["uv_stability_scale_px"]),
            condition_boost=float(cand["condition_boost"]),
            max_step_translation_mm=float(cand["max_step_translation_mm"]),
            max_step_rotation_deg=float(cand["max_step_rotation_deg"]),
            max_iterations=int(cand["max_iterations"]),
            frame_stride=1,
            include_rows=False,
        )
        row["source_coarse_rank"] = int(cand["coarse_rank"])
        full_rows.append(row)
        if idx == 1 or idx % 10 == 0 or idx == len(full_candidates):
            best = min(full_rows, key=lambda r: _to_float(r.get("fb_score")))
            elapsed = time.perf_counter() - t1
            remaining = elapsed / float(idx) * float(len(full_candidates) - idx)
            print(
                f"[fb_constraint_search] full {idx}/{len(full_candidates)} elapsed={elapsed:.1f}s "
                f"remaining={remaining:.1f}s best_score={_to_float(best['fb_score']):.3f} "
                f"dof={best['dof_name']} z={_to_float(best['z_range_mm']):.3f} "
                f"y={_to_float(best['y_range_mm']):.3f} "
                f"reproj95={_to_float(best['reproj_excess_p95_px']):.3f}"
            )

    full_rows.sort(key=lambda r: (_to_float(r.get("fb_score")), _to_float(r.get("z_range_mm"))))
    for rank, row in enumerate(full_rows, start=1):
        row["full_rank"] = rank
    full_csv = out_dir / "hydramarker_fb_motion_constraint_search_full.csv"
    _write_csv(full_csv, full_rows)
    print(f"[fb_constraint_search] saved full -> {full_csv}")

    best = full_rows[0]
    best_with_rows = evaluate_run(
        run,
        dof_name=str(best["dof_name"]),
        robust_c_px=float(best["robust_c_px"]),
        uv_stability_scale_px=float(best["uv_stability_scale_px"]),
        condition_boost=float(best["condition_boost"]),
        max_step_translation_mm=float(best["max_step_translation_mm"]),
        max_step_rotation_deg=float(best["max_step_rotation_deg"]),
        max_iterations=int(best["max_iterations"]),
        frame_stride=1,
        include_rows=True,
    )
    frame_rows = best_with_rows.pop("_frame_rows")
    frame_csv = out_dir / f"{path.stem}_fb_constraint_best_frames.csv"
    _write_csv(frame_csv, frame_rows)
    plot_path = _plot_adaptive_translation(run, best_with_rows, frame_rows, "fb_constraint_best")
    best_with_rows["frame_csv"] = str(frame_csv.resolve())
    best_with_rows["plot_path"] = str(plot_path.resolve())
    best_csv = out_dir / "hydramarker_fb_motion_constraint_search_best.csv"
    _write_csv(best_csv, [best_with_rows])
    print(f"[fb_constraint_search] saved best frames -> {frame_csv}")
    print(f"[fb_constraint_search] saved best summary -> {best_csv}")
    print("[fb_constraint_search] best:")
    for key in [
        "dof_name",
        "active_dofs",
        "robust_c_px",
        "uv_stability_scale_px",
        "condition_boost",
        "max_step_translation_mm",
        "max_step_rotation_deg",
        "max_iterations",
        "raw_z_range_mm",
        "z_range_mm",
        "raw_y_range_mm",
        "y_range_mm",
        "amplitude_error",
        "best_lag_frames",
        "reproj_excess_p95_px",
        "delta_logged_p95_mm",
        "fb_score",
    ]:
        print(f"  {key}: {best_with_rows.get(key)}")


if __name__ == "__main__":
    main()
