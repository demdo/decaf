"""Frame-wise weak Tikhonov replay experiment for HydraMarker poses.

The script re-solves logged observations with weak regularization terms and
compares the resulting trajectory against the tracker output.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from res_motion_irls_sweep import _best_lag_metrics  # noqa: E402
from res_static_irls_replay import (  # noqa: E402
    IrlsConfig,
    _T_to_pose,
    _exp_se3_paper_order,
    _finite_condition_number,
    _median,
    _numeric_motion_jacobian,
    _percentile,
    _pose_to_T,
    _project_points,
    _to_float,
    arrays_for_observation,
    build_age_maps,
    build_point_priors,
    build_static_uv_model,
    load_run,
)


DEFAULT_RUNS = (
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_fb.jsonl",
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_rl.jsonl",
)

DOF_LABELS = ("tx", "ty", "tz", "rx", "ry", "rz")


@dataclass
class WeakTikhonovConfig:
    max_iterations: int = 6
    robust_c_px: float = 0.20
    uv_stability_scale_px: float = 0.05
    age_ramp_frames: int = 1
    min_base_weight: float = 0.05
    lm_damping: float = 1.0e-5
    max_step_translation_mm: float = 5.0
    max_step_rotation_deg: float = 5.0
    weak_eig_ratio: float = 0.015
    weak_strength: float = 1.0
    weak_power: float = 1.0
    weak_cap_ratio: float = 0.25
    weak_min_tz_alignment: float = 0.0
    reproj_guard_px: float = 1.0


@dataclass
class WeakTikhonovResult:
    success: bool
    T: np.ndarray
    iterations: int
    point_count: int
    condition_number: float
    min_eigenvalue: float
    weak_dim: int
    weak_tz_alignment: float
    weak_penalty_trace: float
    last_step_norm: float


def _solve_normal(normal: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(normal, rhs, rcond=None)[0]


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


def _range(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr) - np.min(arr)) if len(arr) else math.nan


def _weak_tikhonov_matrix(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    config: WeakTikhonovConfig,
) -> tuple[np.ndarray, int, float, float]:
    eigvals = np.asarray(eigvals, dtype=np.float64).reshape(6)
    eigvecs = np.asarray(eigvecs, dtype=np.float64).reshape(6, 6)
    finite_positive = eigvals[np.isfinite(eigvals) & (eigvals > 1.0e-12)]
    if len(finite_positive) == 0:
        return np.zeros((6, 6), dtype=np.float64), 0, math.nan, 0.0

    max_eig = float(np.max(finite_positive))
    ratio = max(float(config.weak_eig_ratio), 1.0e-12)
    threshold = ratio * max(max_eig, 1.0e-12)
    weak_mask = np.isfinite(eigvals) & (eigvals > 0.0) & (eigvals < threshold)
    min_tz = max(float(config.weak_min_tz_alignment), 0.0)
    if min_tz > 0.0:
        align = np.abs(eigvecs[2, :]) / np.maximum(np.linalg.norm(eigvecs, axis=0), 1.0e-12)
        weak_mask = weak_mask & (align >= min_tz)
    if not bool(np.any(weak_mask)):
        weakest_vec = eigvecs[:, int(np.nanargmin(eigvals))]
        weak_tz_alignment = float(abs(weakest_vec[2]) / max(float(np.linalg.norm(weakest_vec)), 1.0e-12))
        return np.zeros((6, 6), dtype=np.float64), 0, weak_tz_alignment, 0.0

    V = eigvecs[:, weak_mask]
    weak_vals = eigvals[weak_mask]
    raw = np.maximum(threshold - weak_vals, 0.0)
    if float(config.weak_power) != 1.0:
        scale = np.maximum(raw / max(threshold, 1.0e-12), 0.0)
        raw = threshold * np.power(scale, max(float(config.weak_power), 0.0))
    cap = max(float(config.weak_cap_ratio), 0.0) * max(max_eig, 1.0e-12)
    lambdas = np.minimum(float(config.weak_strength) * raw, cap)
    lambdas = np.where(np.isfinite(lambdas) & (lambdas > 0.0), lambdas, 0.0)
    L = V @ np.diag(lambdas) @ V.T

    weakest_vec = eigvecs[:, int(np.nanargmin(eigvals))]
    weak_tz_alignment = float(abs(weakest_vec[2]) / max(float(np.linalg.norm(weakest_vec)), 1.0e-12))
    return L, int(np.sum(weak_mask)), weak_tz_alignment, float(np.trace(L))


def solve_frame_weak_tikhonov_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    base_weights: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    seed_T: np.ndarray,
    config: WeakTikhonovConfig,
) -> WeakTikhonovResult:
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    base_weights = np.asarray(base_weights, dtype=np.float64).reshape(-1)
    T = np.asarray(seed_T, dtype=np.float64).reshape(4, 4).copy()
    if len(object_points) < 6:
        return WeakTikhonovResult(
            success=False,
            T=T,
            iterations=0,
            point_count=int(len(object_points)),
            condition_number=math.nan,
            min_eigenvalue=math.nan,
            weak_dim=0,
            weak_tz_alignment=math.nan,
            weak_penalty_trace=0.0,
            last_step_norm=math.nan,
        )

    condition_number = math.nan
    min_eig = math.nan
    weak_dim = 0
    weak_tz_alignment = math.nan
    weak_penalty_trace = 0.0
    last_step_norm = math.nan
    weights = np.clip(base_weights, float(config.min_base_weight), None)
    delta_total = np.zeros(6, dtype=np.float64)

    for iteration in range(1, int(config.max_iterations) + 1):
        projected = _project_points(object_points, T, K, dist)
        residual = projected - image_points
        errors = np.sqrt(np.sum(residual * residual, axis=1))

        robust_c = max(float(config.robust_c_px), 1.0e-9)
        robust = 1.0 / (robust_c + errors)
        finite_robust = robust[np.isfinite(robust) & (robust > 0.0)]
        if len(finite_robust):
            robust = robust / float(np.max(finite_robust))
        robust = np.where(np.isfinite(robust), robust, 0.0)

        weights = np.clip(base_weights, float(config.min_base_weight), None) * robust
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)

        J = _numeric_motion_jacobian(object_points, T, K, dist)
        residual_vec = residual.reshape(-1)
        W2 = np.repeat(weights, 2)
        H = J.T @ (W2[:, None] * J)
        g = J.T @ (W2 * residual_vec)
        H_sym = 0.5 * (H + H.T)
        eigvals, eigvecs = np.linalg.eigh(H_sym)
        min_eig = float(np.min(eigvals)) if len(eigvals) else math.nan
        condition_number = _finite_condition_number(eigvals)
        weak_L, weak_dim, weak_tz_alignment, weak_penalty_trace = _weak_tikhonov_matrix(
            eigvals,
            eigvecs,
            config,
        )

        lm_diag = float(config.lm_damping) * np.maximum(np.diag(H), 1.0e-9)
        normal = H + np.diag(lm_diag) + weak_L
        delta = _solve_normal(normal, -g - weak_L @ delta_total)
        delta = np.asarray(delta, dtype=np.float64).reshape(6)

        translation_norm = float(np.linalg.norm(delta[:3]))
        rotation_norm = float(np.linalg.norm(delta[3:]))
        max_translation = max(float(config.max_step_translation_mm), 1.0e-9)
        max_rotation = math.radians(max(float(config.max_step_rotation_deg), 1.0e-9))
        scale = 1.0
        if translation_norm > max_translation:
            scale = min(scale, max_translation / translation_norm)
        if rotation_norm > max_rotation:
            scale = min(scale, max_rotation / rotation_norm)
        delta *= scale

        last_step_norm = float(np.linalg.norm(delta))
        delta_total += delta
        T = _exp_se3_paper_order(delta) @ T
        if translation_norm < 1.0e-5 and rotation_norm < 1.0e-8:
            break

    return WeakTikhonovResult(
        success=True,
        T=T,
        iterations=int(iteration),
        point_count=int(len(object_points)),
        condition_number=float(condition_number),
        min_eigenvalue=float(min_eig),
        weak_dim=int(weak_dim),
        weak_tz_alignment=float(weak_tz_alignment),
        weak_penalty_trace=float(weak_penalty_trace),
        last_step_norm=float(last_step_norm),
    )


def evaluate_run(
    run: dict[str, Any],
    config: WeakTikhonovConfig,
    *,
    frame_stride: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observations = list(run["observations"])
    reference = observations[0]
    T = _pose_to_T(reference.original_rvec.copy(), reference.original_tvec.copy())
    irls_config = IrlsConfig(
        max_iterations=int(config.max_iterations),
        robust_c_px=float(config.robust_c_px),
        uv_stability_scale_px=float(config.uv_stability_scale_px),
        min_base_weight=float(config.min_base_weight),
        condition_boost=0.0,
        age_ramp_frames=int(config.age_ramp_frames),
        lm_damping=float(config.lm_damping),
        max_step_translation_mm=float(config.max_step_translation_mm),
        max_step_rotation_deg=float(config.max_step_rotation_deg),
    )
    static_model = build_static_uv_model(observations)
    priors = build_point_priors(observations, static_model, irls_config)
    age_maps = build_age_maps(observations, age_ramp_frames=int(config.age_ramp_frames))

    rows: list[dict[str, Any]] = []
    stride = max(int(frame_stride), 1)
    eval_indices = list(range(0, len(observations), stride))
    if not eval_indices or eval_indices[-1] != len(observations) - 1:
        eval_indices.append(len(observations) - 1)

    for obs_idx in eval_indices:
        obs = observations[obs_idx]
        obj, uv, weights, _keys = arrays_for_observation(
            obs,
            priors=priors,
            age_weight_by_key=age_maps[obs_idx],
        )
        seed_T = T.copy()
        result = solve_frame_weak_tikhonov_pose(obj, uv, weights, run["K"], run["dist"], seed_T, config)
        candidate_T = result.T.copy() if result.success else seed_T.copy()

        logged_T = _pose_to_T(obs.original_rvec.copy(), obs.original_tvec.copy())
        candidate_wrms = _weighted_rms_residual(obj, uv, weights, candidate_T, run["K"], run["dist"])
        logged_wrms = _weighted_rms_residual(obj, uv, weights, logged_T, run["K"], run["dist"])
        reproj_excess = max(0.0, float(candidate_wrms) - float(logged_wrms))
        guard_reject = bool(
            result.success
            and np.isfinite(float(config.reproj_guard_px))
            and float(config.reproj_guard_px) >= 0.0
            and reproj_excess > float(config.reproj_guard_px)
        )
        if result.success and not guard_reject:
            T = candidate_T
            current_wrms = candidate_wrms
        elif guard_reject:
            T = logged_T.copy()
            current_wrms = logged_wrms
            reproj_excess = 0.0
        else:
            current_wrms = candidate_wrms

        rvec, tvec = _T_to_pose(T)
        delta_logged = np.asarray(tvec, dtype=np.float64).reshape(3) - obs.original_tvec.reshape(3)
        rows.append(
            {
                "frame": int(obs.frame),
                "solved": int(bool(result.success)),
                "guard_reject": int(guard_reject),
                "tvec_x_mm": float(tvec[0]),
                "tvec_y_mm": float(tvec[1]),
                "tvec_z_mm": float(tvec[2]),
                "rvec_x_rad": float(rvec[0]),
                "rvec_y_rad": float(rvec[1]),
                "rvec_z_rad": float(rvec[2]),
                "logged_x_mm": float(obs.original_tvec[0]),
                "logged_y_mm": float(obs.original_tvec[1]),
                "logged_z_mm": float(obs.original_tvec[2]),
                "delta_logged_mm": float(np.linalg.norm(delta_logged)),
                "point_count": int(result.point_count),
                "iterations": int(result.iterations),
                "condition_number": float(result.condition_number),
                "min_eigenvalue": float(result.min_eigenvalue),
                "weak_dim": int(result.weak_dim),
                "weak_tz_alignment": float(result.weak_tz_alignment),
                "weak_penalty_trace": float(result.weak_penalty_trace),
                "last_step_norm": float(result.last_step_norm),
                "current_reproj_weighted_rms_px": float(current_wrms),
                "logged_current_reproj_weighted_rms_px": float(logged_wrms),
                "reproj_excess_px": float(reproj_excess),
            }
        )

    out_t = np.asarray([[r["tvec_x_mm"], r["tvec_y_mm"], r["tvec_z_mm"]] for r in rows], dtype=np.float64)
    raw_t = np.asarray([[r["logged_x_mm"], r["logged_y_mm"], r["logged_z_mm"]] for r in rows], dtype=np.float64)
    out_rel = out_t - out_t[0].reshape(1, 3)
    raw_rel = raw_t - raw_t[0].reshape(1, 3)
    out_ranges = np.nanmax(out_rel, axis=0) - np.nanmin(out_rel, axis=0)
    raw_ranges = np.nanmax(raw_rel, axis=0) - np.nanmin(raw_rel, axis=0)
    main_axis_idx = int(np.nanargmax(raw_ranges))
    lag = _best_lag_metrics(raw_rel[:, main_axis_idx], out_rel[:, main_axis_idx], max_lag=8)
    excess = [_to_float(row.get("reproj_excess_px")) for row in rows]
    delta_logged = [_to_float(row.get("delta_logged_mm")) for row in rows]
    weak_dims = [_to_float(row.get("weak_dim")) for row in rows]
    weak_tz = [_to_float(row.get("weak_tz_alignment")) for row in rows]
    penalty = [_to_float(row.get("weak_penalty_trace")) for row in rows]
    guard = [_to_float(row.get("guard_reject")) for row in rows]
    summary = {
        "run_id": str(run.get("run_id", "")),
        "run_label": str(Path(run["path"]).stem),
        "frames": int(len(rows)),
        "raw_x_range_mm": float(raw_ranges[0]),
        "raw_y_range_mm": float(raw_ranges[1]),
        "raw_z_range_mm": float(raw_ranges[2]),
        "weak_x_range_mm": float(out_ranges[0]),
        "weak_y_range_mm": float(out_ranges[1]),
        "weak_z_range_mm": float(out_ranges[2]),
        "main_axis": ("x", "y", "z")[main_axis_idx],
        "main_axis_raw_range_mm": float(raw_ranges[main_axis_idx]),
        "main_axis_weak_range_mm": float(out_ranges[main_axis_idx]),
        "main_axis_ratio": float(out_ranges[main_axis_idx] / raw_ranges[main_axis_idx])
        if raw_ranges[main_axis_idx] > 1.0e-12
        else math.nan,
        "best_lag_steps": _to_float(lag.get("best_lag_steps")),
        "best_lag_corr": _to_float(lag.get("best_lag_corr")),
        "best_lag_rmse_mm": _to_float(lag.get("best_lag_rmse_mm")),
        "reproj_excess_median_px": _median(excess),
        "reproj_excess_p95_px": _percentile(excess, 95),
        "delta_logged_p95_mm": _percentile(delta_logged, 95),
        "weak_dim_median": _median(weak_dims),
        "weak_tz_alignment_median": _median(weak_tz),
        "weak_penalty_trace_median": _median(penalty),
        "guard_reject_fraction": float(np.nanmean(guard)) if len(guard) else math.nan,
        "weak_eig_ratio": float(config.weak_eig_ratio),
        "weak_strength": float(config.weak_strength),
        "weak_power": float(config.weak_power),
        "weak_cap_ratio": float(config.weak_cap_ratio),
        "weak_min_tz_alignment": float(config.weak_min_tz_alignment),
        "reproj_guard_px": float(config.reproj_guard_px),
    }
    return summary, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_summary(summary: dict[str, Any]) -> str:
    return (
        f"{summary.get('run_id', '')}: "
        f"raw xyz=({summary['raw_x_range_mm']:.3f},"
        f"{summary['raw_y_range_mm']:.3f},{summary['raw_z_range_mm']:.3f}) mm | "
        f"weak xyz=({summary['weak_x_range_mm']:.3f},"
        f"{summary['weak_y_range_mm']:.3f},{summary['weak_z_range_mm']:.3f}) mm | "
        f"axis={summary['main_axis']} ratio={summary['main_axis_ratio']:.3f} "
        f"lag={summary['best_lag_steps']:.0f} "
        f"excess95={summary['reproj_excess_p95_px']:.3f}px "
        f"weak_dim_med={summary['weak_dim_median']:.1f} "
        f"weak_tz_med={summary['weak_tz_alignment_median']:.3f} "
        f"guard={summary['guard_reject_fraction']:.2%}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frame-local weak-eigenspace Tikhonov replay for HydraMarker logs.",
    )
    parser.add_argument("paths", nargs="*", type=Path, help="HydraMarker JSONL run logs.")
    parser.add_argument("--point-set", choices=("correspondence", "pose"), default="correspondence")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--tag", default="frame_weak_tikhonov")
    parser.add_argument("--no-write-csv", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=WeakTikhonovConfig.max_iterations)
    parser.add_argument("--robust-c-px", type=float, default=WeakTikhonovConfig.robust_c_px)
    parser.add_argument("--uv-stability-scale-px", type=float, default=WeakTikhonovConfig.uv_stability_scale_px)
    parser.add_argument("--age-ramp-frames", type=int, default=WeakTikhonovConfig.age_ramp_frames)
    parser.add_argument("--lm-damping", type=float, default=WeakTikhonovConfig.lm_damping)
    parser.add_argument("--max-step-translation-mm", type=float, default=WeakTikhonovConfig.max_step_translation_mm)
    parser.add_argument("--max-step-rotation-deg", type=float, default=WeakTikhonovConfig.max_step_rotation_deg)
    parser.add_argument("--weak-eig-ratio", type=float, default=WeakTikhonovConfig.weak_eig_ratio)
    parser.add_argument("--weak-strength", type=float, default=WeakTikhonovConfig.weak_strength)
    parser.add_argument("--weak-power", type=float, default=WeakTikhonovConfig.weak_power)
    parser.add_argument("--weak-cap-ratio", type=float, default=WeakTikhonovConfig.weak_cap_ratio)
    parser.add_argument("--weak-min-tz-alignment", type=float, default=WeakTikhonovConfig.weak_min_tz_alignment)
    parser.add_argument("--reproj-guard-px", type=float, default=WeakTikhonovConfig.reproj_guard_px)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> WeakTikhonovConfig:
    return WeakTikhonovConfig(
        max_iterations=int(args.max_iterations),
        robust_c_px=float(args.robust_c_px),
        uv_stability_scale_px=float(args.uv_stability_scale_px),
        age_ramp_frames=int(args.age_ramp_frames),
        lm_damping=float(args.lm_damping),
        max_step_translation_mm=float(args.max_step_translation_mm),
        max_step_rotation_deg=float(args.max_step_rotation_deg),
        weak_eig_ratio=float(args.weak_eig_ratio),
        weak_strength=float(args.weak_strength),
        weak_power=float(args.weak_power),
        weak_cap_ratio=float(args.weak_cap_ratio),
        weak_min_tz_alignment=float(args.weak_min_tz_alignment),
        reproj_guard_px=float(args.reproj_guard_px),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = [Path(p) for p in args.paths] if args.paths else [Path(p) for p in DEFAULT_RUNS]
    config = config_from_args(args)
    for path in paths:
        t0 = time.perf_counter()
        run = load_run(path.resolve(), point_set=str(args.point_set))
        summary, rows = evaluate_run(run, config, frame_stride=int(args.frame_stride))
        elapsed = time.perf_counter() - t0
        print(f"[frame_weak_tikhonov] {elapsed:.2f}s {_format_summary(summary)}")
        if args.summary_only or args.no_write_csv:
            continue
        stem = path.with_suffix("")
        tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(args.tag)) or "frame_weak"
        _write_csv(stem.with_name(f"{stem.name}_{tag}_frames.csv"), rows)
        _write_csv(stem.with_name(f"{stem.name}_{tag}_summary.csv"), [summary])


if __name__ == "__main__":
    main()
