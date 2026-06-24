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

from res_motion_irls_sweep import (  # noqa: E402
    COMPONENTS,
    _best_lag_metrics,
    _weighted_rms_residual,
)
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
    solve_irls_lie_pose,
)


DOF_LABELS = ("tx", "ty", "tz", "rx", "ry", "rz")


@dataclass
class AdaptivePriorConfig:
    max_iterations: int = 6
    robust_c_px: float = 0.20
    uv_stability_scale_px: float = 0.05
    age_ramp_frames: int = 1
    lm_damping: float = 1e-5
    max_step_translation_mm: float = 5.0
    max_step_rotation_deg: float = 5.0
    min_base_weight: float = 0.05

    tz_base_lambda: float = 4.0
    tilt_base_lambda: float = 0.35
    weak_z_gain: float = 5.0
    tilt_weak_gain: float = 0.35
    pointset_switch_gain: float = 3.0
    motion_relief: float = 0.70
    free_motion_relief: float = 0.20
    motion_scale_translation_mm: float = 1.0
    motion_scale_rotation_deg: float = 1.0
    weak_z_low: float = 0.25
    weak_z_high: float = 0.85
    cond_log_low: float = 5.0
    cond_log_high: float = 9.0

    velocity_alpha: float = 0.35
    velocity_decay_on_failure: float = 0.50

    stable_accum_rate: float = 0.25
    stable_cap_multiplier: float = 5.0
    stable_motion_decay: float = 0.0
    stable_translation_gate_mm: float = 0.08
    stable_rotation_gate_deg: float = 0.05
    stable_reproj_excess_gate_px: float = 0.04
    reproj_guard_px: float = 0.50
    candidate_tolerance_px: float = 0.20
    enable_camera_axis_candidates: bool = False
    enable_soft_prior_candidate: bool = False
    enable_motion_subspace_candidate: bool = False
    subspace_history_frames: int = 14
    subspace_min_translation_mm: float = 0.20
    subspace_min_rotation_deg: float = 0.20
    subspace_eigen_ratio: float = 0.10
    subspace_static_budget_px: float = 0.12
    subspace_reproj_budget_px: float = 0.50
    subspace_info_eigen_ratio: float = 0.02
    depth_release_alpha: float = 0.25
    depth_release_excess_px: float = 0.06
    depth_release_ratio: float = 1.20
    depth_release_min_z_step_mm: float = 0.06
    depth_release_on_score: float = 1.20
    depth_release_off_score: float = 0.80
    depth_release_max_switch_score: float = 0.35
    depth_release_planar_gate_mm: float = 1.0
    depth_release_min_cumulative_ratio: float = 0.25
    planar_lock_reproj_budget_px: float = -1.0


@dataclass
class AdaptiveSolveResult:
    success: bool
    T: np.ndarray
    iterations: int
    point_count: int
    stats: dict[str, float]
    mean_weight: float
    condition_number: float
    min_eigenvalue: float
    weak_z_alignment: float
    weak_z_score: float
    cond_score: float
    pointset_switch_score: float
    last_step_norm: float
    delta_total: np.ndarray
    free_delta: np.ndarray
    lambda_diag: np.ndarray
    lambda_multiplier: np.ndarray
    camera_z_lambda: float
    camera_z_row: np.ndarray
    stable_lambda_diag: np.ndarray
    C_diag: np.ndarray
    cov_diag: np.ndarray
    C_eigvals: np.ndarray
    C_eigvecs: np.ndarray


def _finite(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _range(values: list[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(np.max(arr) - np.min(arr)) if len(arr) else math.nan


def _rms(values: list[float] | np.ndarray) -> float:
    arr = _finite(values)
    return float(math.sqrt(float(np.mean(arr * arr)))) if len(arr) else math.nan


def _softstep(value: float, lo: float, hi: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if hi <= lo:
        return 1.0 if value >= hi else 0.0
    x = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
    return float(x * x * (3.0 - 2.0 * x))


def _solve_normal(normal: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(normal, rhs, rcond=None)[0]


def _weighted_reprojection_stats(residual: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    residual = np.asarray(residual, dtype=np.float64).reshape(-1, 2)
    errors = np.sqrt(np.sum(residual * residual, axis=1))
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(np.sum(weights)) > 0.0:
        weighted_rms = math.sqrt(float(np.sum(weights * errors * errors) / np.sum(weights)))
    else:
        weighted_rms = math.nan
    return {
        "reproj_mean_px": float(np.mean(errors)) if len(errors) else math.nan,
        "reproj_median_px": _median(errors),
        "reproj_rms_px": _rms(errors),
        "reproj_weighted_rms_px": float(weighted_rms),
        "reproj_p95_px": _percentile(errors, 95),
        "reproj_max_px": float(np.max(errors)) if len(errors) else math.nan,
    }


def _weighted_sse(residual_vec: np.ndarray, weights: np.ndarray) -> float:
    residual_vec = np.asarray(residual_vec, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    W2 = np.repeat(weights, 2)
    if len(W2) != len(residual_vec):
        return math.inf
    return float(np.sum(W2 * residual_vec * residual_vec))


def _exp_se3_paper_order(delta: np.ndarray) -> np.ndarray:
    from res_static_irls_replay import _exp_se3_paper_order as exp_se3

    return exp_se3(delta)


def _adaptive_lambda(
    C: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    free_delta: np.ndarray,
    motion_prior_delta: np.ndarray,
    stable_lambda_diag: np.ndarray,
    pointset_switch_score: float,
    T: np.ndarray,
    config: AdaptivePriorConfig,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float, float, np.ndarray, np.ndarray]:
    C = np.asarray(C, dtype=np.float64).reshape(6, 6)
    C_diag = np.maximum(np.diag(C), 1e-12)

    condition_number = _finite_condition_number(eigvals)
    cond_log = math.log10(condition_number) if np.isfinite(condition_number) and condition_number > 0.0 else math.nan
    cond_score = _softstep(cond_log, config.cond_log_low, config.cond_log_high)

    weak_z_alignment = math.nan
    if eigvecs.shape == (6, 6) and len(eigvals):
        weak_vec = eigvecs[:, int(np.argmin(eigvals))]
        weak_z_alignment = float(abs(weak_vec[2]) / max(float(np.linalg.norm(weak_vec)), 1e-12))
    weak_z_score = _softstep(weak_z_alignment, config.weak_z_low, config.weak_z_high)
    z_uncertainty = max(float(weak_z_score), float(cond_score) * float(weak_z_alignment if np.isfinite(weak_z_alignment) else 0.0))

    lambda_multiplier = np.zeros(6, dtype=np.float64)
    lambda_multiplier[2] = float(config.tz_base_lambda)
    lambda_multiplier[3] = float(config.tilt_base_lambda)
    lambda_multiplier[4] = float(config.tilt_base_lambda)

    switch_score = float(np.clip(pointset_switch_score, 0.0, 1.0))
    lambda_multiplier[2] += float(config.weak_z_gain) * z_uncertainty
    lambda_multiplier[3] += float(config.tilt_weak_gain) * z_uncertainty
    lambda_multiplier[4] += float(config.tilt_weak_gain) * z_uncertainty
    lambda_multiplier[2] += float(config.pointset_switch_gain) * switch_score
    lambda_multiplier[3] += 0.35 * float(config.pointset_switch_gain) * switch_score
    lambda_multiplier[4] += 0.35 * float(config.pointset_switch_gain) * switch_score

    motion_scales = np.asarray(
        [
            config.motion_scale_translation_mm,
            config.motion_scale_translation_mm,
            config.motion_scale_translation_mm,
            math.radians(config.motion_scale_rotation_deg),
            math.radians(config.motion_scale_rotation_deg),
            math.radians(config.motion_scale_rotation_deg),
        ],
        dtype=np.float64,
    )
    motion_scales = np.maximum(motion_scales, 1e-12)
    prior_delta = np.asarray(motion_prior_delta, dtype=np.float64).reshape(6)
    free_delta = np.asarray(free_delta, dtype=np.float64).reshape(6)
    prior_evidence = np.abs(prior_delta) / motion_scales
    free_evidence = np.abs(free_delta) / motion_scales
    t = np.asarray(T, dtype=np.float64).reshape(4, 4)[:3, 3]
    camera_z_row = np.asarray([0.0, 0.0, 1.0, float(t[1]), -float(t[0]), 0.0], dtype=np.float64)
    prior_camera_z = float(camera_z_row @ prior_delta)
    free_camera_z = float(camera_z_row @ free_delta)
    z_scale = max(float(config.motion_scale_translation_mm), 1e-12)
    prior_evidence[2] = abs(prior_camera_z) / z_scale
    free_evidence[2] = abs(free_camera_z) / z_scale
    evidence = np.maximum(prior_evidence, float(config.free_motion_relief) * free_evidence)
    evidence = np.clip(evidence, 0.0, 1.0)
    relief = 1.0 - float(np.clip(config.motion_relief, 0.0, 0.95)) * evidence
    lambda_multiplier *= relief
    lambda_multiplier = np.maximum(lambda_multiplier, 0.0)

    lambda_diag = lambda_multiplier * C_diag
    camera_z_lambda = float(lambda_diag[2])
    lambda_diag[2] = 0.0
    stable_lambda_diag = np.asarray(stable_lambda_diag, dtype=np.float64).reshape(6)
    return (
        lambda_diag,
        lambda_multiplier,
        camera_z_lambda,
        camera_z_row,
        float(weak_z_alignment),
        float(weak_z_score),
        float(cond_score),
        C_diag,
        stable_lambda_diag,
    )


def solve_adaptive_prior_lie_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    base_weights: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    seed_T: np.ndarray,
    config: AdaptivePriorConfig,
    *,
    motion_prior_delta: np.ndarray,
    stable_lambda_diag: np.ndarray,
    pointset_switch_score: float,
) -> AdaptiveSolveResult:
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    base_weights = np.asarray(base_weights, dtype=np.float64).reshape(-1)
    seed_T = np.asarray(seed_T, dtype=np.float64).reshape(4, 4)
    zero6 = np.zeros(6, dtype=np.float64)
    nan6 = np.full(6, math.nan, dtype=np.float64)

    if len(object_points) < 6:
        return AdaptiveSolveResult(
            success=False,
            T=seed_T.copy(),
            iterations=0,
            point_count=int(len(object_points)),
            stats={},
            mean_weight=math.nan,
            condition_number=math.nan,
            min_eigenvalue=math.nan,
            weak_z_alignment=math.nan,
            weak_z_score=math.nan,
            cond_score=math.nan,
            pointset_switch_score=float(pointset_switch_score),
            last_step_norm=math.nan,
            delta_total=zero6.copy(),
            free_delta=nan6.copy(),
            lambda_diag=nan6.copy(),
            lambda_multiplier=nan6.copy(),
            camera_z_lambda=math.nan,
            camera_z_row=nan6.copy(),
            stable_lambda_diag=nan6.copy(),
            C_diag=nan6.copy(),
            cov_diag=nan6.copy(),
            C_eigvals=nan6.copy(),
            C_eigvecs=np.full((6, 6), math.nan, dtype=np.float64),
        )

    T = seed_T.copy()
    weights = np.clip(base_weights, config.min_base_weight, None)
    delta_total = np.zeros(6, dtype=np.float64)
    last_step_norm = math.nan
    condition_number = math.nan
    min_eig = math.nan
    weak_z_alignment = math.nan
    weak_z_score = math.nan
    cond_score = math.nan
    free_delta = np.zeros(6, dtype=np.float64)
    lambda_diag = np.zeros(6, dtype=np.float64)
    lambda_multiplier = np.zeros(6, dtype=np.float64)
    camera_z_lambda = 0.0
    camera_z_row = np.zeros(6, dtype=np.float64)
    C_diag = np.zeros(6, dtype=np.float64)
    cov_diag = np.full(6, math.nan, dtype=np.float64)
    eigvals = np.full(6, math.nan, dtype=np.float64)
    eigvecs = np.full((6, 6), math.nan, dtype=np.float64)

    for iteration in range(1, int(config.max_iterations) + 1):
        projected = _project_points(object_points, T, K, dist)
        residual = projected - image_points
        errors = np.sqrt(np.sum(residual * residual, axis=1))

        robust_c = max(float(config.robust_c_px), 1e-9)
        robust = 1.0 / (robust_c + errors)
        finite_robust = robust[np.isfinite(robust) & (robust > 0.0)]
        if len(finite_robust):
            robust = robust / float(np.max(finite_robust))
        robust = np.where(np.isfinite(robust), robust, 0.0)

        weights = np.clip(base_weights, config.min_base_weight, None) * robust
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)

        J = _numeric_motion_jacobian(object_points, T, K, dist)
        residual_vec = residual.reshape(-1)
        W2 = np.repeat(weights, 2)

        C = J.T @ (W2[:, None] * J)
        g = J.T @ (W2 * residual_vec)
        C_sym = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C_sym)
        min_eig = float(np.min(eigvals)) if len(eigvals) else math.nan
        condition_number = _finite_condition_number(eigvals)
        lm_diag = float(config.lm_damping) * np.maximum(np.diag(C), 1e-9)

        free_delta = _solve_normal(C + np.diag(lm_diag), -g)
        (
            lambda_diag,
            lambda_multiplier,
            camera_z_lambda,
            camera_z_row,
            weak_z_alignment,
            weak_z_score,
            cond_score,
            C_diag,
            stable_lambda_diag,
        ) = _adaptive_lambda(
            C,
            eigvals,
            eigvecs,
            free_delta,
            motion_prior_delta,
            stable_lambda_diag,
            pointset_switch_score,
            T,
            config,
        )

        prior_delta = np.asarray(motion_prior_delta, dtype=np.float64).reshape(6)
        prior_camera_z = float(camera_z_row @ prior_delta)
        normal = (
            C
            + np.diag(lm_diag + lambda_diag + stable_lambda_diag)
            + float(camera_z_lambda) * np.outer(camera_z_row, camera_z_row)
        )
        rhs = (
            -g
            + lambda_diag * prior_delta
            + float(camera_z_lambda) * camera_z_row * prior_camera_z
        )
        delta = _solve_normal(normal, rhs)
        delta = np.asarray(delta, dtype=np.float64).reshape(6)

        translation_norm = float(np.linalg.norm(delta[:3]))
        rotation_norm = float(np.linalg.norm(delta[3:]))
        max_translation = max(float(config.max_step_translation_mm), 1e-9)
        max_rotation = math.radians(max(float(config.max_step_rotation_deg), 1e-9))
        scale = 1.0
        if translation_norm > max_translation:
            scale = min(scale, max_translation / translation_norm)
        if rotation_norm > max_rotation:
            scale = min(scale, max_rotation / rotation_norm)
        delta *= scale

        base_objective = (
            _weighted_sse(residual_vec, weights)
            + float(np.sum(lambda_diag * prior_delta * prior_delta))
            + float(camera_z_lambda * prior_camera_z * prior_camera_z)
        )
        accepted_delta = None
        accepted_T = None
        accepted_objective = math.inf
        line_scale = 1.0
        for _line_idx in range(12):
            trial_delta = delta * line_scale
            trial_T = _exp_se3_paper_order(trial_delta) @ T
            trial_residual = (_project_points(object_points, trial_T, K, dist) - image_points).reshape(-1)
            trial_objective = (
                _weighted_sse(trial_residual, weights)
                + float(np.sum(lambda_diag * (trial_delta - prior_delta) * (trial_delta - prior_delta)))
                + float(camera_z_lambda * (float(camera_z_row @ trial_delta) - prior_camera_z) ** 2)
                + float(np.sum(stable_lambda_diag * trial_delta * trial_delta))
            )
            if np.isfinite(trial_objective) and trial_objective <= base_objective:
                accepted_delta = trial_delta
                accepted_T = trial_T
                accepted_objective = float(trial_objective)
                break
            if np.isfinite(trial_objective) and trial_objective < accepted_objective:
                accepted_delta = trial_delta
                accepted_T = trial_T
                accepted_objective = float(trial_objective)
            line_scale *= 0.5

        if accepted_delta is None or accepted_T is None or accepted_objective > base_objective * (1.0 + 1e-9):
            last_step_norm = 0.0
            break

        delta_total += accepted_delta
        last_step_norm = float(np.linalg.norm(accepted_delta))
        T = accepted_T
        translation_norm = float(np.linalg.norm(accepted_delta[:3]))
        rotation_norm = float(np.linalg.norm(accepted_delta[3:]))
        if translation_norm < 1e-5 and rotation_norm < 1e-8:
            break

    try:
        cov = np.linalg.pinv(
            C
            + np.diag(lambda_diag + stable_lambda_diag)
            + float(camera_z_lambda) * np.outer(camera_z_row, camera_z_row),
            rcond=1e-10,
        )
        cov_diag = np.diag(cov).astype(np.float64)
    except Exception:
        cov_diag = np.full(6, math.nan, dtype=np.float64)

    projected = _project_points(object_points, T, K, dist)
    residual = projected - image_points
    stats = _weighted_reprojection_stats(residual, weights)
    return AdaptiveSolveResult(
        success=True,
        T=T,
        iterations=int(iteration),
        point_count=int(len(object_points)),
        stats=stats,
        mean_weight=float(np.mean(weights)) if len(weights) else math.nan,
        condition_number=float(condition_number),
        min_eigenvalue=float(min_eig),
        weak_z_alignment=float(weak_z_alignment),
        weak_z_score=float(weak_z_score),
        cond_score=float(cond_score),
        pointset_switch_score=float(pointset_switch_score),
        last_step_norm=float(last_step_norm),
        delta_total=delta_total.copy(),
        free_delta=np.asarray(free_delta, dtype=np.float64).reshape(6).copy(),
        lambda_diag=np.asarray(lambda_diag, dtype=np.float64).reshape(6).copy(),
        lambda_multiplier=np.asarray(lambda_multiplier, dtype=np.float64).reshape(6).copy(),
        camera_z_lambda=float(camera_z_lambda),
        camera_z_row=np.asarray(camera_z_row, dtype=np.float64).reshape(6).copy(),
        stable_lambda_diag=np.asarray(stable_lambda_diag, dtype=np.float64).reshape(6).copy(),
        C_diag=np.asarray(C_diag, dtype=np.float64).reshape(6).copy(),
        cov_diag=np.asarray(cov_diag, dtype=np.float64).reshape(6).copy(),
        C_eigvals=np.asarray(eigvals, dtype=np.float64).reshape(6).copy(),
        C_eigvecs=np.asarray(eigvecs, dtype=np.float64).reshape(6, 6).copy(),
    )


def _pointset_metrics(
    current_keys: set[tuple[int, int]],
    previous_keys: set[tuple[int, int]] | None,
) -> tuple[float, float, float]:
    if previous_keys is None:
        return 1.0, 0.0, 0.0
    union = current_keys | previous_keys
    intersection = current_keys & previous_keys
    jaccard = float(len(intersection) / len(union)) if union else 1.0
    new_fraction = float(len(current_keys - previous_keys) / max(len(current_keys), 1))
    lost_fraction = float(len(previous_keys - current_keys) / max(len(previous_keys), 1))
    switch_score = float(np.clip((1.0 - jaccard) + 0.5 * (new_fraction + lost_fraction), 0.0, 1.0))
    return jaccard, new_fraction, switch_score


def _append_vector_fields(row: dict[str, Any], prefix: str, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=np.float64).reshape(6)
    for label, value in zip(DOF_LABELS, arr, strict=False):
        row[f"{prefix}_{label}"] = float(value)


def _candidate_wrms(
    obj: np.ndarray,
    uv: np.ndarray,
    weights: np.ndarray,
    T: np.ndarray,
    run: dict[str, Any],
) -> float:
    return _weighted_rms_residual(obj, uv, weights, T, run["K"], run["dist"])


def _pose_delta_approx(T: np.ndarray, seed_T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    seed_T = np.asarray(seed_T, dtype=np.float64).reshape(4, 4)
    delta = np.zeros(6, dtype=np.float64)
    delta[:3] = T[:3, 3] - seed_T[:3, 3]
    try:
        cv2 = _load_cv2()
        R_delta = T[:3, :3] @ seed_T[:3, :3].T
        rvec, _ = cv2.Rodrigues(R_delta)
        delta[3:] = np.asarray(rvec, dtype=np.float64).reshape(3)
    except Exception:
        delta[3:] = 0.0
    return delta


def _make_hard_config(config: AdaptivePriorConfig, active_dofs: tuple[int, ...] | None) -> IrlsConfig:
    return IrlsConfig(
        max_iterations=int(config.max_iterations),
        robust_c_px=float(config.robust_c_px),
        uv_stability_scale_px=float(config.uv_stability_scale_px),
        min_base_weight=float(config.min_base_weight),
        condition_boost=0.0,
        age_ramp_frames=int(config.age_ramp_frames),
        lm_damping=float(config.lm_damping),
        max_step_translation_mm=float(config.max_step_translation_mm),
        max_step_rotation_deg=float(config.max_step_rotation_deg),
        active_dofs=active_dofs,
    )


def _solve_hard_candidate(
    mode: str,
    active_dofs: tuple[int, ...] | None,
    obj: np.ndarray,
    uv: np.ndarray,
    weights: np.ndarray,
    seed_T: np.ndarray,
    run: dict[str, Any],
    config: AdaptivePriorConfig,
    logged_wrms: float,
) -> dict[str, Any]:
    result = solve_irls_lie_pose(
        obj,
        uv,
        weights,
        run["K"],
        run["dist"],
        seed_T,
        _make_hard_config(config, active_dofs),
    )
    T = result.T.copy() if result.success else seed_T.copy()
    wrms = _candidate_wrms(obj, uv, weights, T, run)
    return {
        "mode": mode,
        "success": bool(result.success),
        "T": T,
        "wrms": float(wrms),
        "excess": max(0.0, float(wrms) - float(logged_wrms)),
        "result": result,
        "delta": _pose_delta_approx(T, seed_T),
    }


def _static_candidate(
    obj: np.ndarray,
    uv: np.ndarray,
    weights: np.ndarray,
    seed_T: np.ndarray,
    run: dict[str, Any],
    logged_wrms: float,
) -> dict[str, Any]:
    wrms = _candidate_wrms(obj, uv, weights, seed_T, run)
    return {
        "mode": "static",
        "success": True,
        "T": seed_T.copy(),
        "wrms": float(wrms),
        "excess": max(0.0, float(wrms) - float(logged_wrms)),
        "result": None,
        "delta": np.zeros(6, dtype=np.float64),
    }


def _principal_basis(vectors: np.ndarray, *, min_norm: float, eigen_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    norms = np.linalg.norm(vectors, axis=1)
    useful = vectors[np.isfinite(norms) & (norms >= float(min_norm))]
    if len(useful) < 2:
        return np.zeros((3, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)
    cov = useful.T @ useful / float(len(useful))
    eigvals, eigvecs = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    if eigvals[0] <= 1e-12:
        return np.zeros((3, 0), dtype=np.float64), eigvals
    keep = eigvals >= max(float(eigen_ratio), 0.0) * eigvals[0]
    return eigvecs[:, keep], eigvals


def _build_motion_subspace_basis(
    motion_history: list[np.ndarray],
    config: AdaptivePriorConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    if not motion_history:
        return np.zeros((6, 0), dtype=np.float64), {
            "translation_dims": 0.0,
            "rotation_dims": 0.0,
            "translation_eigen_ratio": math.nan,
            "rotation_eigen_ratio": math.nan,
        }

    window = np.asarray(motion_history[-max(int(config.subspace_history_frames), 1) :], dtype=np.float64).reshape(-1, 6)
    t_basis, t_eig = _principal_basis(
        window[:, :3],
        min_norm=float(config.subspace_min_translation_mm),
        eigen_ratio=float(config.subspace_eigen_ratio),
    )
    r_basis, r_eig = _principal_basis(
        window[:, 3:],
        min_norm=math.radians(float(config.subspace_min_rotation_deg)),
        eigen_ratio=float(config.subspace_eigen_ratio),
    )

    columns: list[np.ndarray] = []
    for idx in range(t_basis.shape[1]):
        col = np.zeros(6, dtype=np.float64)
        col[:3] = t_basis[:, idx]
        columns.append(col)
    for idx in range(r_basis.shape[1]):
        col = np.zeros(6, dtype=np.float64)
        col[3:] = r_basis[:, idx]
        columns.append(col)

    basis = np.stack(columns, axis=1) if columns else np.zeros((6, 0), dtype=np.float64)
    t_ratio = float(t_eig[1] / t_eig[0]) if len(t_eig) > 1 and t_eig[0] > 1e-12 else math.nan
    r_ratio = float(r_eig[1] / r_eig[0]) if len(r_eig) > 1 and r_eig[0] > 1e-12 else math.nan
    return basis, {
        "translation_dims": float(t_basis.shape[1]),
        "rotation_dims": float(r_basis.shape[1]),
        "translation_eigen_ratio": t_ratio,
        "rotation_eigen_ratio": r_ratio,
    }


def _filter_delta_by_information(
    delta: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    config: AdaptivePriorConfig,
) -> tuple[np.ndarray, float]:
    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    eigvals = np.asarray(eigvals, dtype=np.float64).reshape(6)
    eigvecs = np.asarray(eigvecs, dtype=np.float64).reshape(6, 6)
    if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
        return delta.copy(), math.nan
    max_eig = float(np.max(eigvals))
    if not np.isfinite(max_eig) or max_eig <= 1e-12:
        return np.zeros(6, dtype=np.float64), 0.0
    keep = eigvals >= float(config.subspace_info_eigen_ratio) * max_eig
    if not np.any(keep):
        return np.zeros(6, dtype=np.float64), 0.0
    basis = eigvecs[:, keep]
    filtered = basis @ (basis.T @ delta)
    kept_energy = float(np.linalg.norm(filtered) / max(float(np.linalg.norm(delta)), 1e-12))
    return np.asarray(filtered, dtype=np.float64).reshape(6), kept_energy


def _solve_basis_candidate(
    mode: str,
    basis: np.ndarray,
    obj: np.ndarray,
    uv: np.ndarray,
    weights: np.ndarray,
    seed_T: np.ndarray,
    run: dict[str, Any],
    config: AdaptivePriorConfig,
    logged_wrms: float,
) -> dict[str, Any]:
    basis = np.asarray(basis, dtype=np.float64).reshape(6, -1)
    if basis.shape[1] == 0:
        return _static_candidate(obj, uv, weights, seed_T, run, logged_wrms) | {"mode": mode}

    T = np.asarray(seed_T, dtype=np.float64).reshape(4, 4).copy()
    last_delta = np.zeros(6, dtype=np.float64)
    success = len(obj) >= 6
    active_weights = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), config.min_base_weight, None)
    iterations = 0
    for iteration in range(1, int(config.max_iterations) + 1):
        iterations = int(iteration)
        projected = _project_points(obj, T, run["K"], run["dist"])
        residual = projected - uv
        errors = np.sqrt(np.sum(residual * residual, axis=1))
        robust_c = max(float(config.robust_c_px), 1e-9)
        robust = 1.0 / (robust_c + errors)
        finite = robust[np.isfinite(robust) & (robust > 0.0)]
        if len(finite):
            robust = robust / float(np.max(finite))
        robust = np.where(np.isfinite(robust), robust, 0.0)
        active_weights = np.clip(weights, config.min_base_weight, None) * robust
        active_weights = np.where(np.isfinite(active_weights) & (active_weights > 0.0), active_weights, 0.0)

        J = _numeric_motion_jacobian(obj, T, run["K"], run["dist"])
        residual_vec = residual.reshape(-1)
        W2 = np.repeat(active_weights, 2)
        C = J.T @ (W2[:, None] * J)
        g = J.T @ (W2 * residual_vec)
        Cb = basis.T @ C @ basis
        gb = basis.T @ g
        damping = float(config.lm_damping) * np.maximum(np.diag(Cb), 1e-9)
        try:
            coeff = np.linalg.solve(Cb + np.diag(damping), -gb)
        except np.linalg.LinAlgError:
            coeff = np.linalg.lstsq(Cb + np.diag(damping), -gb, rcond=None)[0]
        delta = np.asarray(basis @ coeff, dtype=np.float64).reshape(6)

        translation_norm = float(np.linalg.norm(delta[:3]))
        rotation_norm = float(np.linalg.norm(delta[3:]))
        max_translation = max(float(config.max_step_translation_mm), 1e-9)
        max_rotation = math.radians(max(float(config.max_step_rotation_deg), 1e-9))
        scale = 1.0
        if translation_norm > max_translation:
            scale = min(scale, max_translation / translation_norm)
        if rotation_norm > max_rotation:
            scale = min(scale, max_rotation / rotation_norm)
        delta *= scale
        last_delta = delta
        T = _exp_se3_paper_order(delta) @ T
        if translation_norm < 1e-5 and rotation_norm < 1e-8:
            break

    wrms = _candidate_wrms(obj, uv, active_weights, T, run) if success else math.inf
    return {
        "mode": mode,
        "success": bool(success),
        "T": T if success else seed_T.copy(),
        "wrms": float(wrms),
        "excess": max(0.0, float(wrms) - float(logged_wrms)),
        "result": None,
        "delta": _pose_delta_approx(T, seed_T) if success else np.zeros(6, dtype=np.float64),
        "last_delta": last_delta,
        "basis_dim": int(basis.shape[1]),
    }


def _select_candidate(
    candidates: list[dict[str, Any]],
    tolerance_px: float,
    *,
    depth_release_active: bool = False,
    planar_lock_active: bool = False,
    planar_lock_reproj_budget_px: float = 0.50,
    enable_camera_axis_candidates: bool = False,
    enable_soft_prior_candidate: bool = False,
    enable_motion_subspace_candidate: bool = False,
    subspace_static_budget_px: float = 0.12,
    subspace_reproj_budget_px: float = 0.50,
) -> dict[str, Any]:
    valid = [c for c in candidates if bool(c.get("success")) and np.isfinite(_to_float(c.get("wrms")))]
    if not enable_camera_axis_candidates:
        valid = [c for c in valid if str(c.get("mode")) not in {"xy_rz", "xy_rot", "xyz_rz"}]
    if not enable_soft_prior_candidate:
        valid = [c for c in valid if str(c.get("mode")) != "soft_prior"]
    if not enable_motion_subspace_candidate:
        valid = [c for c in valid if str(c.get("mode")) != "motion_subspace"]
    valid = [
        c for c in valid if str(c.get("mode")) != "static" or float(c.get("excess", math.inf)) <= float(subspace_static_budget_px)
    ]
    if not valid:
        return candidates[0]
    for candidate in valid:
        if str(candidate.get("mode")) == "static" and float(candidate.get("excess", math.inf)) <= float(subspace_static_budget_px):
            return candidate
    for candidate in valid:
        if (
            str(candidate.get("mode")) == "motion_subspace"
            and int(candidate.get("basis_dim", 0)) > 0
            and float(candidate.get("excess", math.inf)) <= float(subspace_reproj_budget_px)
        ):
            return candidate
    if planar_lock_active and not depth_release_active:
        if float(planar_lock_reproj_budget_px) >= 0.0:
            for candidate in valid:
                if str(candidate.get("mode")) == "xy_rz" and float(candidate["excess"]) <= float(planar_lock_reproj_budget_px):
                    return candidate
    best_excess = min(float(c["excess"]) for c in valid)
    tolerance = max(float(tolerance_px), 0.0)
    eligible = [c for c in valid if float(c["excess"]) <= best_excess + tolerance]
    z_capable = {"xyz_rz", "free6"}
    if depth_release_active:
        z_eligible = [c for c in valid if str(c["mode"]) in z_capable]
        if z_eligible:
            z_best = min(float(c["excess"]) for c in z_eligible)
            eligible = [c for c in z_eligible if float(c["excess"]) <= z_best + min(tolerance, 0.05)]
    preference = {
        "static": 0,
        "motion_subspace": 1,
        "soft_prior": 2,
        "free6": 3,
        "xy_rot": 4,
        "xy_rz": 5,
        "xyz_rz": 6,
    }
    if depth_release_active:
        preference = {
            "xyz_rz": 0,
            "free6": 1,
            "xy_rot": 2,
            "xy_rz": 3,
            "soft_prior": 4,
        }
    eligible.sort(key=lambda c: (preference.get(str(c["mode"]), 99), float(c["excess"])))
    return eligible[0]


def evaluate_run(
    run: dict[str, Any],
    config: AdaptivePriorConfig,
    *,
    frame_stride: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observations = list(run["observations"])
    reference = observations[0]
    T = _pose_to_T(reference.original_rvec.copy(), reference.original_tvec.copy())
    reference_T = T.copy()

    prior_config = IrlsConfig(
        max_iterations=int(config.max_iterations),
        robust_c_px=float(config.robust_c_px),
        uv_stability_scale_px=float(config.uv_stability_scale_px),
        min_base_weight=float(config.min_base_weight),
        condition_boost=0.0,
        age_ramp_frames=int(config.age_ramp_frames),
        max_step_translation_mm=float(config.max_step_translation_mm),
        max_step_rotation_deg=float(config.max_step_rotation_deg),
    )
    static_model = build_static_uv_model(observations)
    priors = build_point_priors(observations, static_model, prior_config)
    age_maps = build_age_maps(observations, age_ramp_frames=config.age_ramp_frames)

    motion_prior_delta = np.zeros(6, dtype=np.float64)
    stable_lambda_diag = np.zeros(6, dtype=np.float64)
    depth_gap_ema = 0.0
    depth_z_ema = 0.0
    depth_xy_ema = 0.0
    depth_release_active = False
    motion_history: list[np.ndarray] = []
    previous_keys: set[tuple[int, int]] | None = None
    rows: list[dict[str, Any]] = []
    stride = max(int(frame_stride), 1)
    eval_indices = list(range(0, len(observations), stride))
    if eval_indices[-1] != len(observations) - 1:
        eval_indices.append(len(observations) - 1)

    for obs_idx in eval_indices:
        obs = observations[obs_idx]
        cur_obj, cur_uv, cur_weights, cur_keys = arrays_for_observation(
            obs,
            priors=priors,
            age_weight_by_key=age_maps[obs_idx],
        )
        current_keys = set(cur_keys)
        pointset_jaccard, new_key_fraction, switch_score = _pointset_metrics(current_keys, previous_keys)
        previous_keys = current_keys

        seed_T = T.copy()
        result = solve_adaptive_prior_lie_pose(
            cur_obj,
            cur_uv,
            cur_weights,
            run["K"],
            run["dist"],
            seed_T,
            config,
            motion_prior_delta=motion_prior_delta,
            stable_lambda_diag=stable_lambda_diag,
            pointset_switch_score=switch_score,
        )
        logged_T = _pose_to_T(obs.original_rvec.copy(), obs.original_tvec.copy())
        logged_wrms = _weighted_rms_residual(cur_obj, cur_uv, cur_weights, logged_T, run["K"], run["dist"])
        soft_T = result.T.copy() if result.success else T.copy()
        soft_wrms = _weighted_rms_residual(cur_obj, cur_uv, cur_weights, soft_T, run["K"], run["dist"])
        static_candidate = _static_candidate(cur_obj, cur_uv, cur_weights, seed_T, run, logged_wrms)
        subspace_basis, subspace_stats = _build_motion_subspace_basis(motion_history, config)
        subspace_candidate = _solve_basis_candidate(
            "motion_subspace",
            subspace_basis,
            cur_obj,
            cur_uv,
            cur_weights,
            seed_T,
            run,
            config,
            logged_wrms,
        )
        xy_rz_candidate = _solve_hard_candidate(
            "xy_rz", (0, 1, 5), cur_obj, cur_uv, cur_weights, seed_T, run, config, logged_wrms
        )
        xy_rot_candidate = _solve_hard_candidate(
            "xy_rot", (0, 1, 3, 4, 5), cur_obj, cur_uv, cur_weights, seed_T, run, config, logged_wrms
        )
        xyz_rz_candidate = _solve_hard_candidate(
            "xyz_rz", (0, 1, 2, 5), cur_obj, cur_uv, cur_weights, seed_T, run, config, logged_wrms
        )
        free6_candidate = _solve_hard_candidate(
            "free6", None, cur_obj, cur_uv, cur_weights, seed_T, run, config, logged_wrms
        )
        candidates = [
            static_candidate,
            subspace_candidate,
            xy_rz_candidate,
            xy_rot_candidate,
            xyz_rz_candidate,
            {
                "mode": "soft_prior",
                "success": bool(result.success),
                "T": soft_T,
                "wrms": float(soft_wrms),
                "excess": max(0.0, float(soft_wrms) - float(logged_wrms)),
                "result": result,
                "delta": _pose_delta_approx(soft_T, seed_T),
            },
            free6_candidate,
        ]
        candidate_by_mode = {str(candidate.get("mode")): candidate for candidate in candidates}

        free_delta = np.asarray(free6_candidate.get("delta"), dtype=np.float64).reshape(6)
        free_z_step = abs(float(free_delta[2]))
        free_xy_step = float(np.linalg.norm(free_delta[:2]))
        free_rel_t = np.asarray(free6_candidate.get("T"), dtype=np.float64).reshape(4, 4)[:3, 3] - reference_T[:3, 3]
        depth_cumulative_xy_mm = float(np.linalg.norm(free_rel_t[:2]))
        depth_cumulative_z_mm = abs(float(free_rel_t[2]))
        depth_cumulative_ratio = depth_cumulative_z_mm / max(depth_cumulative_xy_mm, 1e-9)
        depth_gap = max(0.0, float(xy_rot_candidate["excess"]) - float(free6_candidate["excess"]))
        alpha = float(np.clip(config.depth_release_alpha, 0.0, 1.0))
        depth_gap_ema = (1.0 - alpha) * depth_gap_ema + alpha * depth_gap
        depth_z_ema = (1.0 - alpha) * depth_z_ema + alpha * free_z_step
        depth_xy_ema = (1.0 - alpha) * depth_xy_ema + alpha * free_xy_step
        depth_ratio = depth_z_ema / max(depth_xy_ema, 1e-9)
        gap_score = depth_gap_ema / max(float(config.depth_release_excess_px), 1e-9)
        z_score = depth_z_ema / max(float(config.depth_release_min_z_step_mm), 1e-9)
        ratio_score = depth_ratio / max(float(config.depth_release_ratio), 1e-9)
        depth_release_score = float(min(gap_score, z_score, ratio_score))
        depth_switch_blocked = bool(switch_score > float(config.depth_release_max_switch_score))
        depth_cumulative_blocked = bool(
            depth_cumulative_xy_mm > float(config.depth_release_planar_gate_mm)
            and depth_cumulative_ratio < float(config.depth_release_min_cumulative_ratio)
        )
        planar_lock_active = bool(
            depth_cumulative_blocked
            and not depth_switch_blocked
            and float(config.planar_lock_reproj_budget_px) >= 0.0
            and float(xy_rz_candidate["excess"]) <= float(config.planar_lock_reproj_budget_px)
        )
        if depth_switch_blocked or depth_cumulative_blocked:
            depth_release_score = 0.0
            depth_release_active = False
            depth_gap_ema *= 0.25
            depth_z_ema *= 0.25
            depth_xy_ema *= 0.25
        if depth_release_active:
            depth_release_active = depth_release_score >= float(config.depth_release_off_score)
        else:
            depth_release_active = depth_release_score >= float(config.depth_release_on_score)

        selected = _select_candidate(
            candidates,
            config.candidate_tolerance_px,
            depth_release_active=depth_release_active,
            planar_lock_active=planar_lock_active,
            planar_lock_reproj_budget_px=config.planar_lock_reproj_budget_px,
            enable_camera_axis_candidates=bool(config.enable_camera_axis_candidates),
            enable_soft_prior_candidate=bool(config.enable_soft_prior_candidate),
            enable_motion_subspace_candidate=bool(config.enable_motion_subspace_candidate),
            subspace_static_budget_px=float(config.subspace_static_budget_px),
            subspace_reproj_budget_px=float(config.subspace_reproj_budget_px),
        )
        selected_mode = str(selected.get("mode", "unknown"))
        candidate_T = np.asarray(selected.get("T"), dtype=np.float64).reshape(4, 4)
        candidate_wrms = float(selected.get("wrms", math.nan))
        reproj_excess = max(0.0, float(candidate_wrms) - float(logged_wrms))
        guard_reject = bool(
            bool(selected.get("success"))
            and np.isfinite(float(config.reproj_guard_px))
            and float(config.reproj_guard_px) >= 0.0
            and reproj_excess > float(config.reproj_guard_px)
        )
        history_delta = np.zeros(6, dtype=np.float64)
        info_kept_energy = math.nan
        if bool(selected.get("success")) and not guard_reject:
            T = candidate_T
            selected_delta = _pose_delta_approx(T, seed_T)
            history_delta, info_kept_energy = _filter_delta_by_information(
                selected_delta,
                result.C_eigvals,
                result.C_eigvecs,
                config,
            )
            if selected_mode == "soft_prior":
                motion_prior_delta = (
                    (1.0 - float(config.velocity_alpha)) * motion_prior_delta
                    + float(config.velocity_alpha) * result.delta_total
                )
            else:
                motion_prior_delta *= float(config.velocity_decay_on_failure)
            if float(np.linalg.norm(history_delta[:3])) >= float(config.subspace_min_translation_mm) or math.degrees(
                float(np.linalg.norm(history_delta[3:]))
            ) >= float(config.subspace_min_rotation_deg):
                motion_history.append(history_delta.copy())
                max_history = max(4 * int(config.subspace_history_frames), int(config.subspace_history_frames), 1)
                motion_history = motion_history[-max_history:]
            current_wrms = candidate_wrms
        elif guard_reject:
            T = logged_T.copy()
            motion_prior_delta *= float(config.velocity_decay_on_failure)
            stable_lambda_diag *= float(np.clip(config.stable_motion_decay, 0.0, 1.0))
            current_wrms = logged_wrms
            reproj_excess = 0.0
        else:
            motion_prior_delta *= float(config.velocity_decay_on_failure)
            current_wrms = candidate_wrms
            reproj_excess = max(0.0, float(current_wrms) - float(logged_wrms))

        rvec, tvec = _T_to_pose(T)

        trans_step = float(np.linalg.norm(result.delta_total[:3])) if result.success else math.nan
        rot_step_deg = math.degrees(float(np.linalg.norm(result.delta_total[3:]))) if result.success else math.nan
        stable_frame = bool(
            result.success
            and trans_step <= float(config.stable_translation_gate_mm)
            and rot_step_deg <= float(config.stable_rotation_gate_deg)
            and reproj_excess <= float(config.stable_reproj_excess_gate_px)
        )
        if result.success and stable_frame and config.stable_accum_rate > 0.0:
            cap = float(config.stable_cap_multiplier) * np.maximum(result.C_diag, 1e-12)
            stable_lambda_diag = np.minimum(
                stable_lambda_diag + float(config.stable_accum_rate) * np.maximum(result.C_diag, 0.0),
                cap,
            )
        elif result.success:
            stable_lambda_diag *= float(np.clip(config.stable_motion_decay, 0.0, 1.0))

        delta_logged = np.asarray(tvec, dtype=np.float64).reshape(3) - obs.original_tvec.reshape(3)
        row: dict[str, Any] = {
            "frame": int(obs.frame),
            "solved": int(bool(selected.get("success"))),
            "selected_mode": selected_mode,
            "guard_reject": int(guard_reject),
            "stable_frame": int(stable_frame),
            "point_count": int(result.point_count),
            "distinct_key_count": int(len(current_keys)),
            "pointset_jaccard": float(pointset_jaccard),
            "new_key_fraction": float(new_key_fraction),
            "pointset_switch_score": float(switch_score),
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
            "current_reproj_weighted_rms_px": float(current_wrms),
            "logged_current_reproj_weighted_rms_px": float(logged_wrms),
            "reproj_excess_px": float(reproj_excess),
            "static_excess_px": float(candidate_by_mode["static"]["excess"]),
            "motion_subspace_excess_px": float(candidate_by_mode["motion_subspace"]["excess"]),
            "xy_rz_excess_px": float(candidate_by_mode["xy_rz"]["excess"]),
            "xy_rot_excess_px": float(candidate_by_mode["xy_rot"]["excess"]),
            "xyz_rz_excess_px": float(candidate_by_mode["xyz_rz"]["excess"]),
            "soft_prior_excess_px": float(candidate_by_mode["soft_prior"]["excess"]),
            "free6_excess_px": float(candidate_by_mode["free6"]["excess"]),
            "candidate_tolerance_px": float(config.candidate_tolerance_px),
            "enable_camera_axis_candidates": int(bool(config.enable_camera_axis_candidates)),
            "subspace_basis_dim": int(subspace_basis.shape[1]),
            "subspace_translation_dims": int(subspace_stats["translation_dims"]),
            "subspace_rotation_dims": int(subspace_stats["rotation_dims"]),
            "subspace_translation_eigen_ratio": float(subspace_stats["translation_eigen_ratio"]),
            "subspace_rotation_eigen_ratio": float(subspace_stats["rotation_eigen_ratio"]),
            "subspace_info_kept_energy": float(info_kept_energy),
            "history_delta_norm": float(np.linalg.norm(history_delta)),
            "depth_release_active": int(depth_release_active),
            "depth_release_score": float(depth_release_score),
            "depth_gap_ema_px": float(depth_gap_ema),
            "depth_z_step_ema_mm": float(depth_z_ema),
            "depth_xy_step_ema_mm": float(depth_xy_ema),
            "depth_step_ratio": float(depth_ratio),
            "depth_switch_blocked": int(depth_switch_blocked),
            "depth_cumulative_blocked": int(depth_cumulative_blocked),
            "depth_cumulative_xy_mm": float(depth_cumulative_xy_mm),
            "depth_cumulative_z_mm": float(depth_cumulative_z_mm),
            "depth_cumulative_ratio": float(depth_cumulative_ratio),
            "planar_lock_active": int(planar_lock_active),
            "planar_lock_reproj_budget_px": float(config.planar_lock_reproj_budget_px),
            "free6_step_x_mm": float(free_delta[0]),
            "free6_step_y_mm": float(free_delta[1]),
            "free6_step_z_mm": float(free_delta[2]),
            "iterations": int(result.iterations),
            "mean_weight": float(result.mean_weight),
            "condition_number": float(result.condition_number),
            "min_eigenvalue": float(result.min_eigenvalue),
            "weak_z_alignment": float(result.weak_z_alignment),
            "weak_z_score": float(result.weak_z_score),
            "cond_score": float(result.cond_score),
            "camera_z_lambda": float(result.camera_z_lambda),
            "camera_z_delta_mm": float(result.camera_z_row @ result.delta_total),
            "camera_z_free_delta_mm": float(result.camera_z_row @ result.free_delta),
            "camera_z_prior_delta_mm": float(result.camera_z_row @ motion_prior_delta),
            "last_step_norm": float(result.last_step_norm),
            "step_translation_norm_mm": float(trans_step),
            "step_rotation_norm_deg": float(rot_step_deg),
            "stable_lambda_tz": float(stable_lambda_diag[2]),
        }
        _append_vector_fields(row, "delta", result.delta_total)
        _append_vector_fields(row, "free_delta", result.free_delta)
        _append_vector_fields(row, "motion_prior", motion_prior_delta)
        _append_vector_fields(row, "lambda", result.lambda_diag)
        _append_vector_fields(row, "lambda_mult", result.lambda_multiplier)
        _append_vector_fields(row, "C_diag", result.C_diag)
        _append_vector_fields(row, "cov_diag", result.cov_diag)
        rows.append(row)

    adaptive_t = np.asarray([[r["tvec_x_mm"], r["tvec_y_mm"], r["tvec_z_mm"]] for r in rows], dtype=np.float64)
    logged_t = np.asarray([[r["logged_x_mm"], r["logged_y_mm"], r["logged_z_mm"]] for r in rows], dtype=np.float64)
    adaptive_rel = adaptive_t - adaptive_t[0].reshape(1, 3)
    logged_rel = logged_t - logged_t[0].reshape(1, 3)
    adaptive_ranges = np.nanmax(adaptive_rel, axis=0) - np.nanmin(adaptive_rel, axis=0)
    logged_ranges = np.nanmax(logged_rel, axis=0) - np.nanmin(logged_rel, axis=0)
    movement_axis_idx = int(np.nanargmax(logged_ranges))
    movement_axis = COMPONENTS[movement_axis_idx]
    logged_axis_range = float(logged_ranges[movement_axis_idx])
    adaptive_axis_range = float(adaptive_ranges[movement_axis_idx])
    lag = _best_lag_metrics(
        logged_rel[:, movement_axis_idx],
        adaptive_rel[:, movement_axis_idx],
        max_lag=max(1, int(math.ceil(12.0 / float(stride)))),
    )
    current_wrms = [_to_float(r.get("current_reproj_weighted_rms_px")) for r in rows]
    logged_wrms = [_to_float(r.get("logged_current_reproj_weighted_rms_px")) for r in rows]
    reproj_excess = [_to_float(r.get("reproj_excess_px")) for r in rows]
    delta_logged = [_to_float(r.get("delta_logged_mm")) for r in rows]
    amplitude_ratio = adaptive_axis_range / logged_axis_range if logged_axis_range > 1e-9 else math.nan

    summary = {
        "run_id": str(run["run_id"]),
        "run_label": str(Path(run["path"]).stem),
        "frames_evaluated": int(len(rows)),
        "frame_stride": int(stride),
        "robust_c_px": float(config.robust_c_px),
        "uv_stability_scale_px": float(config.uv_stability_scale_px),
        "age_ramp_frames": int(config.age_ramp_frames),
        "max_iterations": int(config.max_iterations),
        "max_step_translation_mm": float(config.max_step_translation_mm),
        "max_step_rotation_deg": float(config.max_step_rotation_deg),
        "tz_base_lambda": float(config.tz_base_lambda),
        "tilt_base_lambda": float(config.tilt_base_lambda),
        "weak_z_gain": float(config.weak_z_gain),
        "pointset_switch_gain": float(config.pointset_switch_gain),
        "motion_relief": float(config.motion_relief),
        "stable_accum_rate": float(config.stable_accum_rate),
        "stable_cap_multiplier": float(config.stable_cap_multiplier),
        "reproj_guard_px": float(config.reproj_guard_px),
        "enable_camera_axis_candidates": int(bool(config.enable_camera_axis_candidates)),
        "enable_soft_prior_candidate": int(bool(config.enable_soft_prior_candidate)),
        "enable_motion_subspace_candidate": int(bool(config.enable_motion_subspace_candidate)),
        "subspace_history_frames": int(config.subspace_history_frames),
        "subspace_min_translation_mm": float(config.subspace_min_translation_mm),
        "subspace_min_rotation_deg": float(config.subspace_min_rotation_deg),
        "subspace_eigen_ratio": float(config.subspace_eigen_ratio),
        "subspace_static_budget_px": float(config.subspace_static_budget_px),
        "subspace_reproj_budget_px": float(config.subspace_reproj_budget_px),
        "subspace_info_eigen_ratio": float(config.subspace_info_eigen_ratio),
        "depth_release_excess_px": float(config.depth_release_excess_px),
        "depth_release_ratio": float(config.depth_release_ratio),
        "depth_release_min_z_step_mm": float(config.depth_release_min_z_step_mm),
        "depth_release_max_switch_score": float(config.depth_release_max_switch_score),
        "depth_release_planar_gate_mm": float(config.depth_release_planar_gate_mm),
        "depth_release_min_cumulative_ratio": float(config.depth_release_min_cumulative_ratio),
        "planar_lock_reproj_budget_px": float(config.planar_lock_reproj_budget_px),
        "movement_axis": movement_axis,
        "logged_axis_range_mm": float(logged_axis_range),
        "adaptive_axis_range_mm": float(adaptive_axis_range),
        "amplitude_ratio": float(amplitude_ratio),
        "amplitude_error": abs(float(amplitude_ratio) - 1.0) if np.isfinite(amplitude_ratio) else math.nan,
        "best_lag_frames": _to_float(lag.get("best_lag_steps")) * float(stride),
        "best_lag_corr": _to_float(lag.get("best_lag_corr")),
        "best_lag_rmse_mm": _to_float(lag.get("best_lag_rmse_mm")),
        "raw_x_range_mm": float(logged_ranges[0]),
        "raw_y_range_mm": float(logged_ranges[1]),
        "raw_z_range_mm": float(logged_ranges[2]),
        "x_range_mm": float(adaptive_ranges[0]),
        "y_range_mm": float(adaptive_ranges[1]),
        "z_range_mm": float(adaptive_ranges[2]),
        "delta_logged_median_mm": _median(delta_logged),
        "delta_logged_p95_mm": _percentile(delta_logged, 95),
        "current_wrms_median_px": _median(current_wrms),
        "logged_wrms_median_px": _median(logged_wrms),
        "reproj_excess_median_px": _median(reproj_excess),
        "reproj_excess_p95_px": _percentile(reproj_excess, 95),
        "point_count_median": _median([_to_float(r.get("point_count")) for r in rows]),
        "pointset_jaccard_median": _median([_to_float(r.get("pointset_jaccard")) for r in rows]),
        "weak_z_alignment_median": _median([_to_float(r.get("weak_z_alignment")) for r in rows]),
        "lambda_mult_tz_median": _median([_to_float(r.get("lambda_mult_tz")) for r in rows]),
        "stable_frame_fraction": float(np.mean([_to_float(r.get("stable_frame")) for r in rows])),
        "depth_release_fraction": float(np.mean([_to_float(r.get("depth_release_active")) for r in rows])),
        "depth_switch_blocked_fraction": float(np.mean([_to_float(r.get("depth_switch_blocked")) for r in rows])),
        "depth_cumulative_blocked_fraction": float(
            np.mean([_to_float(r.get("depth_cumulative_blocked")) for r in rows])
        ),
        "planar_lock_fraction": float(np.mean([_to_float(r.get("planar_lock_active")) for r in rows])),
        "guard_reject_fraction": float(np.mean([_to_float(r.get("guard_reject")) for r in rows])),
        "mode_static_fraction": float(np.mean([1.0 if str(r.get("selected_mode")) == "static" else 0.0 for r in rows])),
        "mode_motion_subspace_fraction": float(
            np.mean([1.0 if str(r.get("selected_mode")) == "motion_subspace" else 0.0 for r in rows])
        ),
        "mode_xy_rz_fraction": float(np.mean([1.0 if str(r.get("selected_mode")) == "xy_rz" else 0.0 for r in rows])),
        "mode_xy_rot_fraction": float(np.mean([1.0 if str(r.get("selected_mode")) == "xy_rot" else 0.0 for r in rows])),
        "mode_xyz_rz_fraction": float(np.mean([1.0 if str(r.get("selected_mode")) == "xyz_rz" else 0.0 for r in rows])),
        "mode_soft_prior_fraction": float(np.mean([1.0 if str(r.get("selected_mode")) == "soft_prior" else 0.0 for r in rows])),
        "mode_free6_fraction": float(np.mean([1.0 if str(r.get("selected_mode")) == "free6" else 0.0 for r in rows])),
        "solve_failures": int(sum(1 for r in rows if int(r.get("solved", 0)) == 0)),
    }
    return summary, rows


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


def _safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip()) or "adaptive_prior"


def plot_translation(run: dict[str, Any], summary: dict[str, Any], rows: list[dict[str, Any]], tag: str) -> Path:
    from debug_tracker_translation import setup_plot_style
    import matplotlib.pyplot as plt

    setup_plot_style(plt)
    path = Path(run["path"]).resolve()
    frames = np.asarray([_to_float(row.get("frame")) for row in rows], dtype=np.float64)
    adaptive = np.asarray(
        [[_to_float(row.get("tvec_x_mm")), _to_float(row.get("tvec_y_mm")), _to_float(row.get("tvec_z_mm"))] for row in rows],
        dtype=np.float64,
    )
    logged = np.asarray(
        [[_to_float(row.get("logged_x_mm")), _to_float(row.get("logged_y_mm")), _to_float(row.get("logged_z_mm"))] for row in rows],
        dtype=np.float64,
    )
    adaptive_rel = adaptive - adaptive[0].reshape(1, 3)
    logged_rel = logged - logged[0].reshape(1, 3)
    delta = adaptive - logged
    current_wrms = np.asarray([_to_float(row.get("current_reproj_weighted_rms_px")) for row in rows], dtype=np.float64)
    logged_wrms = np.asarray([_to_float(row.get("logged_current_reproj_weighted_rms_px")) for row in rows], dtype=np.float64)
    excess = np.asarray([_to_float(row.get("reproj_excess_px")) for row in rows], dtype=np.float64)
    weak_z = np.asarray([_to_float(row.get("weak_z_alignment")) for row in rows], dtype=np.float64)
    lambda_tz = np.asarray([_to_float(row.get("lambda_mult_tz")) for row in rows], dtype=np.float64)
    switch_score = np.asarray([_to_float(row.get("pointset_switch_score")) for row in rows], dtype=np.float64)
    stable = np.asarray([_to_float(row.get("stable_frame")) for row in rows], dtype=np.float64)
    depth_release = np.asarray([_to_float(row.get("depth_release_active")) for row in rows], dtype=np.float64)
    depth_switch_blocked = np.asarray([_to_float(row.get("depth_switch_blocked")) for row in rows], dtype=np.float64)
    depth_cumulative_blocked = np.asarray([_to_float(row.get("depth_cumulative_blocked")) for row in rows], dtype=np.float64)
    planar_lock = np.asarray([_to_float(row.get("planar_lock_active")) for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(14.5, 14.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.72, 0.85, 0.72]},
    )
    fig.subplots_adjust(top=0.88, hspace=0.30)
    raw_ranges = np.nanmax(logged_rel, axis=0) - np.nanmin(logged_rel, axis=0)
    adaptive_ranges = np.nanmax(adaptive_rel, axis=0) - np.nanmin(adaptive_rel, axis=0)
    fig.suptitle(
        "Adaptive DOF-prior Lie-IRLS translation replay\n"
        f"{path.name} | tz_lambda={summary.get('tz_base_lambda')} weak_gain={summary.get('weak_z_gain')} "
        f"switch_gain={summary.get('pointset_switch_gain')} motion_relief={summary.get('motion_relief')}\n"
        f"raw range xyz=({raw_ranges[0]:.3f}, {raw_ranges[1]:.3f}, {raw_ranges[2]:.3f}) mm | "
        f"adaptive range xyz=({adaptive_ranges[0]:.3f}, {adaptive_ranges[1]:.3f}, {adaptive_ranges[2]:.3f}) mm",
        fontsize=13,
    )

    colors = {"x": "#4c78a8", "y": "#54a24b", "z": "#e45756"}
    for axis_idx, label in enumerate(("x", "y", "z")):
        ax = axes[axis_idx]
        ax.plot(frames, logged_rel[:, axis_idx], color="#8a8f98", linewidth=1.15, linestyle="--", label="logged/raw PnP")
        ax.plot(frames, adaptive_rel[:, axis_idx], color=colors[label], linewidth=1.55, label="adaptive DOF-prior")
        ax.set_ylabel(f"{label} rel. [mm]")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="upper right", fontsize=8)

    delta_ax = axes[3]
    for axis_idx, label in enumerate(("x", "y", "z")):
        delta_ax.plot(frames, delta[:, axis_idx], color=colors[label], linewidth=1.15, label=f"{label} adaptive - raw")
    delta_ax.axhline(0.0, color="#8a8f98", linewidth=0.8, linestyle="--")
    delta_ax.set_ylabel("delta [mm]")
    delta_ax.grid(True, alpha=0.28)
    delta_ax.legend(loc="upper right", fontsize=8, ncol=3)

    reproj_ax = axes[4]
    reproj_ax.plot(frames, logged_wrms, color="#8a8f98", linewidth=1.1, linestyle="--", label="raw current WRMS")
    reproj_ax.plot(frames, current_wrms, color="#f58518", linewidth=1.25, label="adaptive current WRMS")
    reproj_ax.plot(frames, excess, color="#b279a2", linewidth=1.15, label="excess")
    reproj_ax.set_ylabel("reproj. [px]")
    reproj_ax.grid(True, alpha=0.28)
    reproj_ax.legend(loc="upper right", fontsize=8)

    diag_ax = axes[5]
    diag_ax.plot(frames, lambda_tz, color="#e45756", linewidth=1.2, label="lambda mult tz")
    diag_ax.plot(frames, weak_z, color="#4c78a8", linewidth=1.1, label="weak-z alignment")
    diag_ax.plot(frames, switch_score, color="#54a24b", linewidth=1.1, label="pointset switch")
    diag_ax.fill_between(frames, 0.0, 1.0, where=stable >= 0.5, color="#54a24b", alpha=0.13, label="stable accum")
    diag_ax.fill_between(frames, 0.0, 1.0, where=depth_release >= 0.5, color="#f58518", alpha=0.18, label="depth release")
    diag_ax.fill_between(frames, 0.0, 1.0, where=planar_lock >= 0.5, color="#72b7b2", alpha=0.18, label="planar lock")
    diag_ax.fill_between(
        frames,
        0.0,
        1.0,
        where=(depth_switch_blocked >= 0.5) | (depth_cumulative_blocked >= 0.5),
        color="#b279a2",
        alpha=0.10,
        label="depth blocked",
    )
    diag_ax.set_ylabel("diagnostics")
    diag_ax.set_xlabel("frame")
    diag_ax.grid(True, alpha=0.28)
    diag_ax.legend(loc="upper right", fontsize=8, ncol=4)

    out_path = path.with_name(f"{path.stem}_{_safe_tag(tag)}_adaptive_prior_translation_plot.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _select_jsonl_with_qt() -> Path:
    from debug_tracker_translation import select_jsonl_with_qt

    selected = select_jsonl_with_qt()
    if selected is None:
        raise RuntimeError("No JSONL run selected")
    return Path(selected)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a HydraTracker JSONL with adaptive per-DOF Lie-IRLS priors.",
    )
    parser.add_argument("path", nargs="?", type=Path, help="HydraTracker JSONL run. If omitted, a Qt file dialog opens.")
    parser.add_argument("--select", action="store_true", help="Open a Qt file dialog even if no path is supplied.")
    parser.add_argument("--point-set", choices=("correspondence", "pose"), default="correspondence")
    parser.add_argument("--tag", default="adaptive_dof_prior")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-write-csv", action="store_true")

    parser.add_argument("--robust-c-px", type=float, default=0.20)
    parser.add_argument("--uv-stability-scale-px", type=float, default=0.05)
    parser.add_argument("--age-ramp-frames", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--max-step-translation-mm", type=float, default=5.0)
    parser.add_argument("--max-step-rotation-deg", type=float, default=5.0)

    parser.add_argument("--tz-base-lambda", type=float, default=4.0)
    parser.add_argument("--tilt-base-lambda", type=float, default=0.35)
    parser.add_argument("--weak-z-gain", type=float, default=5.0)
    parser.add_argument("--tilt-weak-gain", type=float, default=0.35)
    parser.add_argument("--pointset-switch-gain", type=float, default=3.0)
    parser.add_argument("--motion-relief", type=float, default=0.70)
    parser.add_argument("--free-motion-relief", type=float, default=0.20)
    parser.add_argument("--motion-scale-translation-mm", type=float, default=1.0)
    parser.add_argument("--motion-scale-rotation-deg", type=float, default=1.0)

    parser.add_argument("--stable-accum-rate", type=float, default=0.25)
    parser.add_argument("--stable-cap-multiplier", type=float, default=5.0)
    parser.add_argument("--stable-motion-decay", type=float, default=0.0)
    parser.add_argument("--stable-translation-gate-mm", type=float, default=0.08)
    parser.add_argument("--stable-rotation-gate-deg", type=float, default=0.05)
    parser.add_argument("--stable-reproj-excess-gate-px", type=float, default=0.04)
    parser.add_argument("--reproj-guard-px", type=float, default=0.50)
    parser.add_argument("--candidate-tolerance-px", type=float, default=0.20)
    parser.add_argument("--enable-camera-axis-candidates", action="store_true")
    parser.add_argument("--enable-soft-prior-candidate", action="store_true")
    parser.add_argument("--enable-motion-subspace-candidate", action="store_true")
    parser.add_argument("--subspace-history-frames", type=int, default=14)
    parser.add_argument("--subspace-min-translation-mm", type=float, default=0.20)
    parser.add_argument("--subspace-min-rotation-deg", type=float, default=0.20)
    parser.add_argument("--subspace-eigen-ratio", type=float, default=0.10)
    parser.add_argument("--subspace-static-budget-px", type=float, default=0.12)
    parser.add_argument("--subspace-reproj-budget-px", type=float, default=0.50)
    parser.add_argument("--subspace-info-eigen-ratio", type=float, default=0.02)
    parser.add_argument("--depth-release-alpha", type=float, default=0.25)
    parser.add_argument("--depth-release-excess-px", type=float, default=0.06)
    parser.add_argument("--depth-release-ratio", type=float, default=1.20)
    parser.add_argument("--depth-release-min-z-step-mm", type=float, default=0.06)
    parser.add_argument("--depth-release-on-score", type=float, default=1.20)
    parser.add_argument("--depth-release-off-score", type=float, default=0.80)
    parser.add_argument("--depth-release-max-switch-score", type=float, default=0.35)
    parser.add_argument("--depth-release-planar-gate-mm", type=float, default=1.0)
    parser.add_argument("--depth-release-min-cumulative-ratio", type=float, default=0.25)
    parser.add_argument(
        "--planar-lock-reproj-budget-px",
        type=float,
        default=-1.0,
        help="Optional diagnostic camera-XY lock budget. Negative disables it.",
    )
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> AdaptivePriorConfig:
    return AdaptivePriorConfig(
        max_iterations=int(args.max_iterations),
        robust_c_px=float(args.robust_c_px),
        uv_stability_scale_px=float(args.uv_stability_scale_px),
        age_ramp_frames=int(args.age_ramp_frames),
        max_step_translation_mm=float(args.max_step_translation_mm),
        max_step_rotation_deg=float(args.max_step_rotation_deg),
        tz_base_lambda=float(args.tz_base_lambda),
        tilt_base_lambda=float(args.tilt_base_lambda),
        weak_z_gain=float(args.weak_z_gain),
        tilt_weak_gain=float(args.tilt_weak_gain),
        pointset_switch_gain=float(args.pointset_switch_gain),
        motion_relief=float(args.motion_relief),
        free_motion_relief=float(args.free_motion_relief),
        motion_scale_translation_mm=float(args.motion_scale_translation_mm),
        motion_scale_rotation_deg=float(args.motion_scale_rotation_deg),
        stable_accum_rate=float(args.stable_accum_rate),
        stable_cap_multiplier=float(args.stable_cap_multiplier),
        stable_motion_decay=float(args.stable_motion_decay),
        stable_translation_gate_mm=float(args.stable_translation_gate_mm),
        stable_rotation_gate_deg=float(args.stable_rotation_gate_deg),
        stable_reproj_excess_gate_px=float(args.stable_reproj_excess_gate_px),
        reproj_guard_px=float(args.reproj_guard_px),
        candidate_tolerance_px=float(args.candidate_tolerance_px),
        enable_camera_axis_candidates=bool(args.enable_camera_axis_candidates),
        enable_soft_prior_candidate=bool(args.enable_soft_prior_candidate),
        enable_motion_subspace_candidate=bool(args.enable_motion_subspace_candidate),
        subspace_history_frames=int(args.subspace_history_frames),
        subspace_min_translation_mm=float(args.subspace_min_translation_mm),
        subspace_min_rotation_deg=float(args.subspace_min_rotation_deg),
        subspace_eigen_ratio=float(args.subspace_eigen_ratio),
        subspace_static_budget_px=float(args.subspace_static_budget_px),
        subspace_reproj_budget_px=float(args.subspace_reproj_budget_px),
        subspace_info_eigen_ratio=float(args.subspace_info_eigen_ratio),
        depth_release_alpha=float(args.depth_release_alpha),
        depth_release_excess_px=float(args.depth_release_excess_px),
        depth_release_ratio=float(args.depth_release_ratio),
        depth_release_min_z_step_mm=float(args.depth_release_min_z_step_mm),
        depth_release_on_score=float(args.depth_release_on_score),
        depth_release_off_score=float(args.depth_release_off_score),
        depth_release_max_switch_score=float(args.depth_release_max_switch_score),
        depth_release_planar_gate_mm=float(args.depth_release_planar_gate_mm),
        depth_release_min_cumulative_ratio=float(args.depth_release_min_cumulative_ratio),
        planar_lock_reproj_budget_px=float(args.planar_lock_reproj_budget_px),
    )


def main() -> None:
    args = _parse_args(sys.argv[1:])
    path = Path(args.path) if args.path is not None else _select_jsonl_with_qt()
    if args.select:
        path = _select_jsonl_with_qt()
    path = path.resolve()
    config = _config_from_args(args)

    t0 = time.perf_counter()
    run = load_run(path, point_set=str(args.point_set))
    summary, rows = evaluate_run(run, config, frame_stride=int(args.frame_stride))
    elapsed = time.perf_counter() - t0

    print(
        "[adaptive_prior] "
        f"frames={summary['frames_evaluated']} elapsed={elapsed:.2f}s "
        f"raw_xyz=({summary['raw_x_range_mm']:.3f}, {summary['raw_y_range_mm']:.3f}, {summary['raw_z_range_mm']:.3f})mm "
        f"adaptive_xyz=({summary['x_range_mm']:.3f}, {summary['y_range_mm']:.3f}, {summary['z_range_mm']:.3f})mm "
        f"amp_ratio={summary['amplitude_ratio']:.4f} lag={summary['best_lag_frames']:.1f} "
        f"reproj95={summary['reproj_excess_p95_px']:.3f}px "
        f"guard={summary['guard_reject_fraction']:.2%} "
        f"depth_release={summary['depth_release_fraction']:.2%} "
        f"planar_lock={summary['planar_lock_fraction']:.2%} "
        f"modes(static/subspace/soft/free/xyrz/xyrot/xyzrz)="
        f"{summary['mode_static_fraction']:.2%}/"
        f"{summary['mode_motion_subspace_fraction']:.2%}/"
        f"{summary['mode_soft_prior_fraction']:.2%}/"
        f"{summary['mode_free6_fraction']:.2%}/"
        f"{summary['mode_xy_rz_fraction']:.2%}/"
        f"{summary['mode_xy_rot_fraction']:.2%}/"
        f"{summary['mode_xyz_rz_fraction']:.2%} "
        f"lambda_tz_med={summary['lambda_mult_tz_median']:.3f} "
        f"weak_z_med={summary['weak_z_alignment_median']:.3f}"
    )

    out_dir = path.parent
    tag = _safe_tag(str(args.tag))
    if not args.no_write_csv:
        frame_csv = out_dir / f"{path.stem}_{tag}_adaptive_prior_frames.csv"
        summary_csv = out_dir / f"{path.stem}_{tag}_adaptive_prior_summary.csv"
        _write_csv(frame_csv, rows)
        _write_csv(summary_csv, [summary])
        print(f"[adaptive_prior] saved frames  -> {frame_csv.resolve()}")
        print(f"[adaptive_prior] saved summary -> {summary_csv.resolve()}")

    if not args.no_plot:
        plot_path = plot_translation(run, summary, rows, tag)
        print(f"[adaptive_prior] saved plot    -> {plot_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[adaptive_prior] ERROR: {exc}")
        sys.exit(1)
