"""Fixed-lag map-tracking replay for HydraMarker observations.

The module replays logged corner observations through a fixed-lag optimization
experiment to evaluate whether local temporal smoothing improves pose
consistency.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from res_static_irls_replay import (  # noqa: E402
    IrlsConfig,
    _T_to_pose,
    _exp_se3_paper_order,
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
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_bf_rot.jsonl",
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_static.jsonl",
    "hydramarker/tests/hydramarker_tracker_runs/hydramarker_tracker_run_static_do.jsonl",
)

TWIST_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz")


def _load_scipy_least_squares():
    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        raise RuntimeError(
            "scipy is required. Run this with the decaf environment, e.g. "
            "C:\\Users\\domin\\anaconda3\\envs\\decaf\\python.exe"
        ) from exc
    return least_squares


def _load_cv2():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "OpenCV/cv2 is required. Run this with the decaf environment, e.g. "
            "C:\\Users\\domin\\anaconda3\\envs\\decaf\\python.exe"
        ) from exc
    return cv2


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


def _safe_inv(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ t)
    return out


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _left_jacobian_so3_inv(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64) - 0.5 * W + (1.0 / 12.0) * (W @ W)
    half_theta = 0.5 * theta
    denom = max(1.0 - math.cos(theta), 1e-15)
    a = (1.0 / (theta * theta)) * (1.0 - (half_theta * math.sin(theta) / denom))
    return np.eye(3, dtype=np.float64) - 0.5 * W + a * (W @ W)


def _log_se3_paper_order(T: np.ndarray) -> np.ndarray:
    """SE(3) logarithm in [tx, ty, tz, rx, ry, rz] order."""

    cv2 = _load_cv2()
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    w = np.asarray(rvec, dtype=np.float64).reshape(3)
    v = _left_jacobian_so3_inv(w) @ T[:3, 3].reshape(3)
    return np.concatenate([v, w]).astype(np.float64)


def _relative_twist(prev_T: np.ndarray, cur_T: np.ndarray) -> np.ndarray:
    return _log_se3_paper_order(np.asarray(cur_T, dtype=np.float64) @ _safe_inv(prev_T))


def _predict_constant_velocity(prev_T: np.ndarray, cur_T: np.ndarray) -> np.ndarray:
    motion = np.asarray(cur_T, dtype=np.float64).reshape(4, 4) @ _safe_inv(prev_T)
    return motion @ np.asarray(cur_T, dtype=np.float64).reshape(4, 4)


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


@dataclass
class FixedLagMapConfig:
    algorithm: str = "depth-kalman"
    window_size: int = 7
    robust_c_px: float = 0.30
    uv_stability_scale_px: float = 0.08
    min_base_weight: float = 0.05
    age_ramp_frames: int = 4
    max_points_per_frame: int = 0
    measurement_weight: float = 1.0
    condition_boost: float = 0.0
    condition_boost_cap: float = 8.0
    switch_reproj_downweight: float = 0.75
    switch_temporal_boost: float = 2.0
    anchor_weight: float = 250.0
    seed_mode: str = "raw"
    output_mode: str = "online"
    velocity_weight: float = 0.0
    acceleration_weight: float = 0.60
    depth_velocity_weight: float = 0.0
    depth_acceleration_weight: float = 0.0
    camera_z_velocity_weight: float = 0.0
    camera_z_acceleration_weight: float = 0.0
    weak_velocity_weight: float = 0.12
    weak_acceleration_weight: float = 1.20
    weak_eig_ratio: float = 0.015
    weak_power: float = 1.0
    weak_cap: float = 25.0
    step_translation_mm: float = 4.0
    step_rotation_deg: float = 4.0
    var_translation_bound_mm: float = 20.0
    var_rotation_bound_deg: float = 15.0
    max_nfev: int = 18
    solver: str = "scipy"
    lm_damping: float = 1e-5
    loss: str = "linear"
    residual_clip: float = 1.0e5
    depth_filter_mode: str = "fixed-lag"
    depth_filter_lag_frames: int = 20
    depth_observation_std_mm: float = 16.0
    depth_process_std_mm: float = 0.05
    depth_initial_velocity_std_mm: float = 10.0
    depth_switch_observation_gain: float = 0.0
    depth_switch_observation_power: float = 2.0
    depth_weak_tz_observation_gain: float = 0.0
    depth_weak_tz_observation_power: float = 2.0
    depth_observation_scale_cap: float = 0.0
    depth_reprojection_guard_px: float = 1.0
    depth_guard_weak_tz_low: float = 0.0
    depth_guard_weak_tz_high: float = 1.0
    depth_guard_low_weak_tz_px: float = -1.0
    depth_guard_high_weak_tz_px: float = -1.0
    verbose: bool = False


@dataclass
class FramePacket:
    obs_index: int
    frame: int
    raw_T: np.ndarray
    T: np.ndarray
    object_points: np.ndarray
    image_points: np.ndarray
    weights: np.ndarray
    keys: list[tuple[int, int]]
    switch_score: float
    eigvals: np.ndarray
    eigvecs: np.ndarray
    weakness: np.ndarray
    condition_number: float
    weak_tz_alignment: float


def _condition_number(eigvals: np.ndarray) -> float:
    eig = np.asarray(eigvals, dtype=np.float64).reshape(-1)
    eig = eig[np.isfinite(eig) & (eig > 1e-12)]
    if len(eig) < 2:
        return math.nan
    return float(np.max(eig) / np.min(eig))


def _pointset_switch_score(prev_keys: list[tuple[int, int]] | None, keys: list[tuple[int, int]]) -> float:
    if not prev_keys:
        return 0.0
    prev = set(prev_keys)
    cur = set(keys)
    if not prev and not cur:
        return 0.0
    union = len(prev | cur)
    if union <= 0:
        return 0.0
    return float(1.0 - (len(prev & cur) / float(union)))


def _select_points(
    object_points: np.ndarray,
    image_points: np.ndarray,
    weights: np.ndarray,
    keys: list[tuple[int, int]],
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    max_points = int(max_points)
    if max_points <= 0 or len(object_points) <= max_points:
        return object_points, image_points, weights, keys
    order = np.argsort(-np.asarray(weights, dtype=np.float64).reshape(-1), kind="mergesort")
    keep = np.sort(order[:max_points])
    return (
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3)[keep],
        np.asarray(image_points, dtype=np.float64).reshape(-1, 2)[keep],
        np.asarray(weights, dtype=np.float64).reshape(-1)[keep],
        [keys[int(i)] for i in keep],
    )


def _information_profile(
    object_points: np.ndarray,
    weights: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    config: FixedLagMapConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if len(object_points) < 6:
        return (
            np.full(6, math.nan, dtype=np.float64),
            np.eye(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            math.nan,
            math.nan,
        )
    try:
        J = _numeric_motion_jacobian(object_points, T, K, dist)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        W2 = np.repeat(weights, 2)
        H = J.T @ (W2[:, None] * J)
        eigvals, eigvecs = np.linalg.eigh(0.5 * (H + H.T))
    except Exception:
        return (
            np.full(6, math.nan, dtype=np.float64),
            np.eye(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            math.nan,
            math.nan,
        )
    eigvals = np.asarray(eigvals, dtype=np.float64).reshape(6)
    eigvecs = np.asarray(eigvecs, dtype=np.float64).reshape(6, 6)
    finite_positive = eigvals[np.isfinite(eigvals) & (eigvals > 1e-12)]
    if len(finite_positive) == 0:
        weakness = np.zeros(6, dtype=np.float64)
    else:
        max_eig = float(np.max(finite_positive))
        rel = np.asarray(eigvals, dtype=np.float64) / max(max_eig, 1e-12)
        ratio = max(float(config.weak_eig_ratio), 1e-12)
        weak_raw = ratio / np.maximum(rel, 1e-12)
        weak_raw = np.where(rel < ratio, weak_raw, 0.0)
        weakness = np.power(np.maximum(weak_raw, 0.0), max(float(config.weak_power), 0.0))
        weakness = np.clip(weakness, 0.0, max(float(config.weak_cap), 0.0))
        weakness = np.where(np.isfinite(weakness), weakness, 0.0)
    condition = _condition_number(eigvals)
    if eigvecs.shape == (6, 6) and len(eigvals):
        weak_vec = eigvecs[:, int(np.nanargmin(eigvals))]
        weak_tz_alignment = float(abs(weak_vec[2]) / max(float(np.linalg.norm(weak_vec)), 1e-12))
    else:
        weak_tz_alignment = math.nan
    return eigvals, eigvecs, weakness, condition, weak_tz_alignment


def _condition_saliency_weights(
    object_points: np.ndarray,
    weights: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    config: FixedLagMapConfig,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1).copy()
    if float(config.condition_boost) <= 0.0 or len(object_points) < 6:
        return weights
    try:
        J = _numeric_motion_jacobian(object_points, T, K, dist)
        W2 = np.repeat(np.where(weights > 0.0, weights, 0.0), 2)
        H = J.T @ (W2[:, None] * J)
        H_inv = np.linalg.pinv(H, rcond=1e-10)
        leverage = np.empty(len(object_points), dtype=np.float64)
        for idx in range(len(object_points)):
            Ji = J[(2 * idx) : (2 * idx + 2), :]
            leverage[idx] = float(np.trace(Ji @ H_inv @ Ji.T))
        positive = leverage[np.isfinite(leverage) & (leverage > 1e-12)]
        if len(positive) == 0:
            return weights
        geom = math.exp(float(np.mean(np.log(positive))))
        saliency = np.sqrt(np.maximum(leverage, 0.0) / max(geom, 1e-12))
        saliency = np.where(np.isfinite(saliency), saliency, 0.0)
        saliency = np.clip(saliency, 0.0, max(float(config.condition_boost_cap), 1.0))
        return weights * (1.0 + float(config.condition_boost) * saliency)
    except Exception:
        return weights


def _build_packets(run: dict[str, Any], config: FixedLagMapConfig) -> list[FramePacket]:
    observations = list(run["observations"])
    static_model = build_static_uv_model(observations)
    irls_cfg = IrlsConfig(
        uv_stability_scale_px=float(config.uv_stability_scale_px),
        min_base_weight=float(config.min_base_weight),
        age_ramp_frames=int(config.age_ramp_frames),
    )
    priors = build_point_priors(observations, static_model, irls_cfg)
    age_maps = build_age_maps(observations, age_ramp_frames=int(config.age_ramp_frames))

    packets: list[FramePacket] = []
    prev_keys: list[tuple[int, int]] | None = None
    for obs_index, obs in enumerate(observations):
        object_points, image_points, weights, keys = arrays_for_observation(
            obs,
            priors=priors,
            age_weight_by_key=age_maps[obs_index],
        )
        object_points, image_points, weights, keys = _select_points(
            object_points,
            image_points,
            weights,
            keys,
            max_points=int(config.max_points_per_frame),
        )
        if len(object_points) < 6:
            continue
        raw_T = _pose_to_T(obs.original_rvec.copy(), obs.original_tvec.copy())
        weights = _condition_saliency_weights(object_points, weights, raw_T, run["K"], run["dist"], config)
        switch_score = _pointset_switch_score(prev_keys, keys)
        eigvals, eigvecs, weakness, condition, weak_tz_alignment = _information_profile(
            object_points,
            weights,
            raw_T,
            run["K"],
            run["dist"],
            config,
        )
        packets.append(
            FramePacket(
                obs_index=int(obs_index),
                frame=int(obs.frame),
                raw_T=raw_T,
                T=raw_T.copy(),
                object_points=object_points,
                image_points=image_points,
                weights=weights,
                keys=keys,
                switch_score=float(switch_score),
                eigvals=eigvals,
                eigvecs=eigvecs,
                weakness=weakness,
                condition_number=float(condition),
                weak_tz_alignment=float(weak_tz_alignment),
            )
        )
        prev_keys = keys
    return packets


def _twist_scales(config: FixedLagMapConfig) -> np.ndarray:
    step_r = math.radians(max(float(config.step_rotation_deg), 1e-9))
    step_t = max(float(config.step_translation_mm), 1e-9)
    return np.asarray([step_t, step_t, step_t, step_r, step_r, step_r], dtype=np.float64)


def _window_residual(
    x: np.ndarray,
    *,
    seed_Ts: list[np.ndarray],
    window: list[FramePacket],
    K: np.ndarray,
    dist: np.ndarray,
    config: FixedLagMapConfig,
) -> np.ndarray:
    deltas = np.asarray(x, dtype=np.float64).reshape(len(window), 6)
    Ts = [_exp_se3_paper_order(deltas[i]) @ seed_Ts[i] for i in range(len(window))]
    scale = _twist_scales(config)
    residuals: list[np.ndarray] = []

    if len(Ts) > 0 and config.anchor_weight > 0.0:
        anchor = _relative_twist(seed_Ts[0], Ts[0])
        residuals.append(math.sqrt(float(config.anchor_weight)) * (anchor / scale))

    robust_c = max(float(config.robust_c_px), 1e-9)
    measurement_weight = max(float(config.measurement_weight), 0.0)
    for T, packet in zip(Ts, window):
        projected = _project_points(packet.object_points, T, K, dist)
        reproj = projected.reshape(-1, 2) - packet.image_points.reshape(-1, 2)
        reproj = np.nan_to_num(reproj, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)
        trust = 1.0 / (
            1.0
            + max(float(config.switch_reproj_downweight), 0.0)
            * max(float(packet.switch_score), 0.0)
        )
        w = np.asarray(packet.weights, dtype=np.float64).reshape(-1)
        w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
        weight = np.sqrt(measurement_weight * trust * w)
        residuals.append((reproj * weight[:, None]).reshape(-1) / robust_c)

    velocities: list[np.ndarray] = []
    for i in range(1, len(Ts)):
        v = _relative_twist(Ts[i - 1], Ts[i])
        velocities.append(v)
        packet = window[i]
        boost = 1.0 + max(float(config.switch_temporal_boost), 0.0) * max(
            float(packet.switch_score),
            0.0,
        )
        if config.velocity_weight > 0.0:
            residuals.append(
                math.sqrt(float(config.velocity_weight) * boost) * (v / scale)
            )
        if config.depth_velocity_weight > 0.0:
            residuals.append(
                np.asarray(
                    [math.sqrt(float(config.depth_velocity_weight) * boost) * v[2] / scale[2]],
                    dtype=np.float64,
                )
            )
        if config.camera_z_velocity_weight > 0.0:
            dz = float(Ts[i][2, 3] - Ts[i - 1][2, 3])
            residuals.append(
                np.asarray(
                    [math.sqrt(float(config.camera_z_velocity_weight) * boost) * dz / scale[2]],
                    dtype=np.float64,
                )
            )
        if config.weak_velocity_weight > 0.0:
            weak = np.asarray(packet.weakness, dtype=np.float64).reshape(6)
            if float(np.sum(weak)) > 0.0:
                proj = packet.eigvecs.T @ v
                residuals.append(
                    np.sqrt(float(config.weak_velocity_weight) * boost * weak)
                    * (proj / scale)
                )

    for i in range(2, len(Ts)):
        packet = window[i]
        boost = 1.0 + max(float(config.switch_temporal_boost), 0.0) * max(
            float(packet.switch_score),
            0.0,
        )
        acc = velocities[i - 1] - velocities[i - 2]
        if config.acceleration_weight > 0.0:
            residuals.append(
                math.sqrt(float(config.acceleration_weight) * boost) * (acc / scale)
            )
        if config.depth_acceleration_weight > 0.0:
            residuals.append(
                np.asarray(
                    [
                        math.sqrt(float(config.depth_acceleration_weight) * boost)
                        * acc[2]
                        / scale[2]
                    ],
                    dtype=np.float64,
                )
            )
        if config.camera_z_acceleration_weight > 0.0:
            dz0 = float(Ts[i - 1][2, 3] - Ts[i - 2][2, 3])
            dz1 = float(Ts[i][2, 3] - Ts[i - 1][2, 3])
            ddz = dz1 - dz0
            residuals.append(
                np.asarray(
                    [
                        math.sqrt(float(config.camera_z_acceleration_weight) * boost)
                        * ddz
                        / scale[2]
                    ],
                    dtype=np.float64,
                )
            )
        if config.weak_acceleration_weight > 0.0:
            weak = np.asarray(packet.weakness, dtype=np.float64).reshape(6)
            if float(np.sum(weak)) > 0.0:
                proj = packet.eigvecs.T @ acc
                residuals.append(
                    np.sqrt(float(config.weak_acceleration_weight) * boost * weak)
                    * (proj / scale)
                )

    if not residuals:
        return np.empty((0,), dtype=np.float64)
    out = np.concatenate([np.asarray(part, dtype=np.float64).reshape(-1) for part in residuals])
    clip = max(float(config.residual_clip), 1.0)
    out = np.nan_to_num(out, nan=clip, posinf=clip, neginf=-clip)
    return np.clip(out, -clip, clip)


def _optimize_window(
    window: list[FramePacket],
    K: np.ndarray,
    dist: np.ndarray,
    config: FixedLagMapConfig,
) -> dict[str, float]:
    if not window:
        return {"success": 0.0, "cost": math.nan, "nfev": 0.0}
    if config.solver == "linearized":
        return _optimize_window_linearized(window, K, dist, config)
    least_squares = _load_scipy_least_squares()
    seed_Ts = [packet.T.copy() for packet in window]
    x0 = np.zeros((len(window), 6), dtype=np.float64).reshape(-1)
    t_bound = max(float(config.var_translation_bound_mm), 1e-6)
    r_bound = math.radians(max(float(config.var_rotation_bound_deg), 1e-6))
    one_bounds = np.asarray([t_bound, t_bound, t_bound, r_bound, r_bound, r_bound])
    bounds = np.tile(one_bounds, len(window))
    x_scale = np.tile(
        np.asarray([1.0, 1.0, 1.0, math.radians(1.0), math.radians(1.0), math.radians(1.0)]),
        len(window),
    )
    result = least_squares(
        _window_residual,
        x0,
        bounds=(-bounds, bounds),
        x_scale=x_scale,
        loss=str(config.loss),
        f_scale=1.0,
        max_nfev=max(int(config.max_nfev), 1),
        args=(),
        kwargs={
            "seed_Ts": seed_Ts,
            "window": window,
            "K": K,
            "dist": dist,
            "config": config,
        },
    )
    deltas = np.asarray(result.x, dtype=np.float64).reshape(len(window), 6)
    for i, packet in enumerate(window):
        packet.T = _exp_se3_paper_order(deltas[i]) @ seed_Ts[i]
    return {
        "success": 1.0 if bool(result.success) else 0.0,
        "cost": float(result.cost),
        "nfev": float(result.nfev),
        "optimality": float(getattr(result, "optimality", math.nan)),
    }


def _add_prior_block(
    H: np.ndarray,
    g: np.ndarray,
    *,
    residual: np.ndarray,
    coeffs: list[tuple[int, np.ndarray]],
    weight: np.ndarray | float,
    scale: np.ndarray,
    eigvecs: np.ndarray | None = None,
) -> float:
    residual = np.asarray(residual, dtype=np.float64).reshape(6)
    scale = np.asarray(scale, dtype=np.float64).reshape(6)
    if np.isscalar(weight):
        w = np.full(6, float(weight), dtype=np.float64)
    else:
        w = np.asarray(weight, dtype=np.float64).reshape(6)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    if eigvecs is None:
        transform = np.eye(6, dtype=np.float64)
    else:
        transform = np.asarray(eigvecs, dtype=np.float64).reshape(6, 6).T
    r = transform @ residual
    B_parts: list[tuple[int, np.ndarray]] = []
    for idx, coeff in coeffs:
        B_parts.append((idx, transform @ np.asarray(coeff, dtype=np.float64).reshape(6, 6)))
    weighted_r = np.sqrt(w) * (r / scale)
    cost = float(np.dot(weighted_r, weighted_r))
    for idx_a, Ba_raw in B_parts:
        Ba = (np.sqrt(w)[:, None] / scale[:, None]) * Ba_raw
        sl_a = slice(6 * idx_a, 6 * idx_a + 6)
        g[sl_a] += Ba.T @ weighted_r
        for idx_b, Bb_raw in B_parts:
            Bb = (np.sqrt(w)[:, None] / scale[:, None]) * Bb_raw
            sl_b = slice(6 * idx_b, 6 * idx_b + 6)
            H[sl_a, sl_b] += Ba.T @ Bb
    return cost


def _optimize_window_linearized(
    window: list[FramePacket],
    K: np.ndarray,
    dist: np.ndarray,
    config: FixedLagMapConfig,
) -> dict[str, float]:
    n = len(window)
    if n <= 0:
        return {"success": 0.0, "cost": math.nan, "nfev": 0.0}
    anchor_T = window[0].T.copy()
    scale = _twist_scales(config)
    robust_c = max(float(config.robust_c_px), 1e-9)
    total_cost = math.nan
    last_step_norm = math.nan
    max_step_t = max(float(config.step_translation_mm), 1e-9)
    max_step_r = math.radians(max(float(config.step_rotation_deg), 1e-9))

    for iteration in range(1, max(int(config.max_nfev), 1) + 1):
        H = np.zeros((6 * n, 6 * n), dtype=np.float64)
        g = np.zeros(6 * n, dtype=np.float64)
        total_cost = 0.0

        for i, packet in enumerate(window):
            projected = _project_points(packet.object_points, packet.T, K, dist)
            residual = projected.reshape(-1, 2) - packet.image_points.reshape(-1, 2)
            residual = np.nan_to_num(residual, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)
            errors = np.sqrt(np.sum(residual * residual, axis=1))
            robust = 1.0 / (1.0 + (errors / robust_c) ** 2)
            trust = 1.0 / (
                1.0
                + max(float(config.switch_reproj_downweight), 0.0)
                * max(float(packet.switch_score), 0.0)
            )
            weights = np.asarray(packet.weights, dtype=np.float64).reshape(-1)
            weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
            w2 = (
                max(float(config.measurement_weight), 0.0)
                * trust
                * weights
                * robust
                / (robust_c * robust_c)
            )
            J = _numeric_motion_jacobian(packet.object_points, packet.T, K, dist)
            r = residual.reshape(-1)
            W2 = np.repeat(w2, 2)
            sl = slice(6 * i, 6 * i + 6)
            H[sl, sl] += J.T @ (W2[:, None] * J)
            g[sl] += J.T @ (W2 * r)
            total_cost += float(np.sum(W2 * r * r))

        eye = np.eye(6, dtype=np.float64)
        if config.anchor_weight > 0.0:
            total_cost += _add_prior_block(
                H,
                g,
                residual=_relative_twist(anchor_T, window[0].T),
                coeffs=[(0, eye)],
                weight=float(config.anchor_weight),
                scale=scale,
            )

        velocities: list[np.ndarray] = []
        for i in range(1, n):
            v = _relative_twist(window[i - 1].T, window[i].T)
            velocities.append(v)
            packet = window[i]
            boost = 1.0 + max(float(config.switch_temporal_boost), 0.0) * max(
                float(packet.switch_score),
                0.0,
            )
            if config.velocity_weight > 0.0:
                total_cost += _add_prior_block(
                    H,
                    g,
                    residual=v,
                    coeffs=[(i - 1, -eye), (i, eye)],
                    weight=float(config.velocity_weight) * boost,
                    scale=scale,
                )
            if config.depth_velocity_weight > 0.0:
                depth_weight = np.asarray([0.0, 0.0, float(config.depth_velocity_weight) * boost, 0.0, 0.0, 0.0])
                total_cost += _add_prior_block(
                    H,
                    g,
                    residual=v,
                    coeffs=[(i - 1, -eye), (i, eye)],
                    weight=depth_weight,
                    scale=scale,
                )
            if config.weak_velocity_weight > 0.0:
                weak = np.asarray(packet.weakness, dtype=np.float64).reshape(6)
                if float(np.sum(weak)) > 0.0:
                    total_cost += _add_prior_block(
                        H,
                        g,
                        residual=v,
                        coeffs=[(i - 1, -eye), (i, eye)],
                        weight=float(config.weak_velocity_weight) * boost * weak,
                        scale=scale,
                        eigvecs=packet.eigvecs,
                    )

        for i in range(2, n):
            acc = velocities[i - 1] - velocities[i - 2]
            packet = window[i]
            boost = 1.0 + max(float(config.switch_temporal_boost), 0.0) * max(
                float(packet.switch_score),
                0.0,
            )
            if config.acceleration_weight > 0.0:
                total_cost += _add_prior_block(
                    H,
                    g,
                    residual=acc,
                    coeffs=[(i - 2, eye), (i - 1, -2.0 * eye), (i, eye)],
                    weight=float(config.acceleration_weight) * boost,
                    scale=scale,
                )
            if config.depth_acceleration_weight > 0.0:
                depth_weight = np.asarray([0.0, 0.0, float(config.depth_acceleration_weight) * boost, 0.0, 0.0, 0.0])
                total_cost += _add_prior_block(
                    H,
                    g,
                    residual=acc,
                    coeffs=[(i - 2, eye), (i - 1, -2.0 * eye), (i, eye)],
                    weight=depth_weight,
                    scale=scale,
                )
            if config.weak_acceleration_weight > 0.0:
                weak = np.asarray(packet.weakness, dtype=np.float64).reshape(6)
                if float(np.sum(weak)) > 0.0:
                    total_cost += _add_prior_block(
                        H,
                        g,
                        residual=acc,
                        coeffs=[(i - 2, eye), (i - 1, -2.0 * eye), (i, eye)],
                        weight=float(config.weak_acceleration_weight) * boost * weak,
                        scale=scale,
                        eigvecs=packet.eigvecs,
                    )

        diag = np.maximum(np.diag(H), 1.0)
        normal = H + np.diag(max(float(config.lm_damping), 0.0) * diag)
        try:
            step = -np.linalg.solve(normal, g)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(normal, g, rcond=1e-10)[0]
        step = np.asarray(step, dtype=np.float64).reshape(n, 6)
        step = np.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)

        for i in range(n):
            t_norm = float(np.linalg.norm(step[i, :3]))
            r_norm = float(np.linalg.norm(step[i, 3:]))
            if t_norm > max_step_t:
                step[i, :3] *= max_step_t / max(t_norm, 1e-12)
            if r_norm > max_step_r:
                step[i, 3:] *= max_step_r / max(r_norm, 1e-12)

        last_step_norm = float(np.max(np.linalg.norm(step, axis=1)))
        for i, packet in enumerate(window):
            packet.T = _exp_se3_paper_order(step[i]) @ packet.T
        if last_step_norm < 1e-7:
            break

    return {
        "success": 1.0,
        "cost": float(total_cost),
        "nfev": float(iteration),
        "optimality": float(last_step_norm),
    }


def run_fixed_lag_map(run: dict[str, Any], config: FixedLagMapConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packets = _build_packets(run, config)
    if not packets:
        raise RuntimeError("No usable packets found.")

    def emit_row(cur: FramePacket, opt: dict[str, float] | None = None) -> dict[str, Any]:
        opt = opt or {}
        _, tvec = _T_to_pose(cur.T)
        raw_rvec, raw_tvec = _T_to_pose(cur.raw_T)
        current_wrms = _weighted_rms_residual(
            cur.object_points,
            cur.image_points,
            cur.weights,
            cur.T,
            run["K"],
            run["dist"],
        )
        raw_wrms = _weighted_rms_residual(
            cur.object_points,
            cur.image_points,
            cur.weights,
            cur.raw_T,
            run["K"],
            run["dist"],
        )
        delta = np.asarray(tvec, dtype=np.float64).reshape(3) - np.asarray(
            raw_tvec,
            dtype=np.float64,
        ).reshape(3)
        return {
            "frame": int(cur.frame),
            "tvec_x_mm": float(tvec[0]),
            "tvec_y_mm": float(tvec[1]),
            "tvec_z_mm": float(tvec[2]),
            "raw_tvec_x_mm": float(raw_tvec[0]),
            "raw_tvec_y_mm": float(raw_tvec[1]),
            "raw_tvec_z_mm": float(raw_tvec[2]),
            "raw_rvec_x_rad": float(raw_rvec[0]),
            "raw_rvec_y_rad": float(raw_rvec[1]),
            "raw_rvec_z_rad": float(raw_rvec[2]),
            "delta_raw_x_mm": float(delta[0]),
            "delta_raw_y_mm": float(delta[1]),
            "delta_raw_z_mm": float(delta[2]),
            "delta_raw_norm_mm": float(np.linalg.norm(delta)),
            "point_count": int(len(cur.object_points)),
            "switch_score": float(cur.switch_score),
            "condition_number": float(cur.condition_number),
            "weak_tz_alignment": float(cur.weak_tz_alignment),
            "weakness_sum": float(np.sum(cur.weakness)),
            "current_reproj_weighted_rms_px": float(current_wrms),
            "raw_reproj_weighted_rms_px": float(raw_wrms),
            "reproj_excess_px": float(current_wrms - raw_wrms),
            "opt_cost": float(opt.get("cost", math.nan)),
            "opt_nfev": float(opt.get("nfev", math.nan)),
        }

    rows: list[dict[str, Any]] = []
    window: list[FramePacket] = []
    opt_costs: list[float] = []
    opt_nfev: list[float] = []
    start_time = time.perf_counter()

    for packet in packets:
        if config.output_mode == "delayed" and len(window) >= max(int(config.window_size), 1):
            rows.append(emit_row(window.pop(0)))

        if config.seed_mode == "predict" and len(window) >= 2:
            packet.T = _predict_constant_velocity(window[-2].T, window[-1].T)
        elif config.seed_mode == "predict" and len(window) == 1:
            packet.T = window[-1].T.copy()
        else:
            packet.T = packet.raw_T.copy()
        window.append(packet)
        if len(window) > max(int(config.window_size), 1):
            window.pop(0)

        opt = _optimize_window(window, run["K"], run["dist"], config)
        opt_costs.append(float(opt.get("cost", math.nan)))
        opt_nfev.append(float(opt.get("nfev", math.nan)))

        if config.output_mode == "online":
            rows.append(emit_row(window[-1], opt))

    if config.output_mode == "delayed":
        rows.extend(emit_row(packet) for packet in window)

    elapsed_s = time.perf_counter() - start_time
    summary = summarize_rows(rows)
    summary.update(
        {
            "run_id": str(run.get("run_id", "")),
            "frames": int(len(rows)),
            "elapsed_s": float(elapsed_s),
            "mean_nfev": float(np.nanmean(opt_nfev)) if len(opt_nfev) else math.nan,
            "mean_cost": float(np.nanmean(opt_costs)) if len(opt_costs) else math.nan,
        }
    )
    return rows, summary


def _forward_depth_kalman(
    measurements: np.ndarray,
    config: FixedLagMapConfig,
    observation_scales: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    z = np.asarray(measurements, dtype=np.float64).reshape(-1)
    if observation_scales is None:
        scales = np.ones_like(z, dtype=np.float64)
    else:
        scales = np.asarray(observation_scales, dtype=np.float64).reshape(-1)
        if len(scales) != len(z):
            raise ValueError("observation_scales must match measurements length.")
        scales = np.where(np.isfinite(scales) & (scales > 0.0), scales, 1.0)
    F = np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    H = np.asarray([[1.0, 0.0]], dtype=np.float64)
    G = np.asarray([[0.5], [1.0]], dtype=np.float64)
    q = max(float(config.depth_process_std_mm), 1e-12)
    r = max(float(config.depth_observation_std_mm), 1e-12)
    Q = G @ G.T * (q * q)
    R_base = r * r
    init_v = max(float(config.depth_initial_velocity_std_mm), 1e-12)

    x = np.asarray([float(z[0]), 0.0], dtype=np.float64)
    P = np.diag([R_base * float(scales[0]), init_v * init_v]).astype(np.float64)
    filtered: list[np.ndarray] = []
    filtered_cov: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    predicted_cov: list[np.ndarray] = []
    for idx, measurement in enumerate(z):
        if idx == 0:
            x_pred = x.copy()
            P_pred = P.copy()
        else:
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
        innovation = np.asarray([float(measurement)], dtype=np.float64) - H @ x_pred
        R = np.asarray([[R_base * float(scales[idx])]], dtype=np.float64)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + (K @ innovation).reshape(2)
        P = (np.eye(2, dtype=np.float64) - K @ H) @ P_pred
        predicted.append(x_pred.copy())
        predicted_cov.append(P_pred.copy())
        filtered.append(x.copy())
        filtered_cov.append(P.copy())
    return F, filtered, filtered_cov, predicted, predicted_cov


def _depth_fixed_lag_smooth(
    measurements: np.ndarray,
    config: FixedLagMapConfig,
    observation_scales: np.ndarray | None = None,
) -> np.ndarray:
    F, xs, Ps, x_preds, P_preds = _forward_depth_kalman(measurements, config, observation_scales)
    n = len(xs)
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    mode = str(config.depth_filter_mode)
    if mode == "causal":
        return np.asarray([x[0] for x in xs], dtype=np.float64)
    if mode == "rts":
        lag = n - 1
    else:
        lag = max(int(config.depth_filter_lag_frames), 0)

    out: list[float] = []
    for start in range(n):
        end = min(n - 1, start + lag)
        x = xs[end].copy()
        P = Ps[end].copy()
        for idx in range(end - 1, start - 1, -1):
            C = Ps[idx] @ F.T @ np.linalg.inv(P_preds[idx + 1])
            x = xs[idx] + C @ (x - x_preds[idx + 1])
            P = Ps[idx] + C @ (P - P_preds[idx + 1]) @ C.T
        out.append(float(x[0]))
    return np.asarray(out, dtype=np.float64)


def _depth_observation_scales(packets: list[FramePacket], config: FixedLagMapConfig) -> np.ndarray:
    if not packets:
        return np.empty((0,), dtype=np.float64)
    switch_gain = max(float(config.depth_switch_observation_gain), 0.0)
    weak_gain = max(float(config.depth_weak_tz_observation_gain), 0.0)
    switch_power = max(float(config.depth_switch_observation_power), 0.0)
    weak_power = max(float(config.depth_weak_tz_observation_power), 0.0)

    scales = np.ones((len(packets),), dtype=np.float64)
    if switch_gain > 0.0:
        switch = np.asarray([max(float(packet.switch_score), 0.0) for packet in packets], dtype=np.float64)
        switch = np.where(np.isfinite(switch), switch, 0.0)
        scales += switch_gain * np.power(switch, switch_power)
    if weak_gain > 0.0:
        weak = np.asarray([max(float(packet.weak_tz_alignment), 0.0) for packet in packets], dtype=np.float64)
        weak = np.where(np.isfinite(weak), np.clip(weak, 0.0, 1.0), 0.0)
        scales += weak_gain * np.power(weak, weak_power)
    cap = float(config.depth_observation_scale_cap)
    if np.isfinite(cap) and cap > 1.0:
        scales = np.minimum(scales, cap)
    return np.where(np.isfinite(scales) & (scales > 0.0), scales, 1.0)


def _depth_guard_px(packet: FramePacket, config: FixedLagMapConfig) -> float:
    base_guard = float(config.depth_reprojection_guard_px)
    low_guard = float(config.depth_guard_low_weak_tz_px)
    high_guard = float(config.depth_guard_high_weak_tz_px)
    if not (np.isfinite(low_guard) and low_guard >= 0.0 and np.isfinite(high_guard) and high_guard >= 0.0):
        return base_guard

    lo = float(config.depth_guard_weak_tz_low)
    hi = float(config.depth_guard_weak_tz_high)
    weak = float(packet.weak_tz_alignment)
    if not np.isfinite(weak):
        return base_guard
    if hi <= lo:
        t = 1.0 if weak >= hi else 0.0
    else:
        t = float(np.clip((weak - lo) / (hi - lo), 0.0, 1.0))
        t = t * t * (3.0 - 2.0 * t)
    return float((1.0 - t) * low_guard + t * high_guard)


def run_depth_kalman(run: dict[str, Any], config: FixedLagMapConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    packets = _build_packets(run, config)
    if not packets:
        raise RuntimeError("No usable packets found.")
    start_time = time.perf_counter()
    raw_z = np.asarray([packet.raw_T[2, 3] for packet in packets], dtype=np.float64)
    observation_scales = _depth_observation_scales(packets, config)
    filtered_z = _depth_fixed_lag_smooth(raw_z, config, observation_scales)
    rows: list[dict[str, Any]] = []
    base_observation_std = max(float(config.depth_observation_std_mm), 1.0e-12)
    for packet, z_value, observation_scale in zip(packets, filtered_z, observation_scales, strict=False):
        raw_z_value = float(packet.raw_T[2, 3])
        T = packet.raw_T.copy()
        T[2, 3] = float(z_value)
        raw_rvec, _raw_tvec = _T_to_pose(packet.raw_T)
        current_wrms = _weighted_rms_residual(
            packet.object_points,
            packet.image_points,
            packet.weights,
            T,
            run["K"],
            run["dist"],
        )
        raw_wrms = _weighted_rms_residual(
            packet.object_points,
            packet.image_points,
            packet.weights,
            packet.raw_T,
            run["K"],
            run["dist"],
        )
        guard = _depth_guard_px(packet, config)
        if guard > 0.0 and np.isfinite(current_wrms) and np.isfinite(raw_wrms):
            if current_wrms - raw_wrms > guard:
                lo = 0.0
                hi = 1.0
                best_z = raw_z_value
                best_wrms = raw_wrms
                for _ in range(20):
                    alpha = 0.5 * (lo + hi)
                    candidate_z = raw_z_value + alpha * (float(z_value) - raw_z_value)
                    candidate_T = packet.raw_T.copy()
                    candidate_T[2, 3] = candidate_z
                    candidate_wrms = _weighted_rms_residual(
                        packet.object_points,
                        packet.image_points,
                        packet.weights,
                        candidate_T,
                        run["K"],
                        run["dist"],
                    )
                    if np.isfinite(candidate_wrms) and candidate_wrms - raw_wrms <= guard:
                        best_z = float(candidate_z)
                        best_wrms = float(candidate_wrms)
                        lo = alpha
                    else:
                        hi = alpha
                z_value = best_z
                T = packet.raw_T.copy()
                T[2, 3] = float(z_value)
                current_wrms = best_wrms
        rows.append(
            {
                "frame": int(packet.frame),
                "tvec_x_mm": float(packet.raw_T[0, 3]),
                "tvec_y_mm": float(packet.raw_T[1, 3]),
                "tvec_z_mm": float(z_value),
                "raw_tvec_x_mm": float(packet.raw_T[0, 3]),
                "raw_tvec_y_mm": float(packet.raw_T[1, 3]),
                "raw_tvec_z_mm": float(packet.raw_T[2, 3]),
                "raw_rvec_x_rad": float(raw_rvec[0]),
                "raw_rvec_y_rad": float(raw_rvec[1]),
                "raw_rvec_z_rad": float(raw_rvec[2]),
                "delta_raw_x_mm": 0.0,
                "delta_raw_y_mm": 0.0,
                "delta_raw_z_mm": float(z_value - raw_z_value),
                "delta_raw_norm_mm": float(abs(z_value - raw_z_value)),
                "point_count": int(len(packet.object_points)),
                "switch_score": float(packet.switch_score),
                "condition_number": float(packet.condition_number),
                "weak_tz_alignment": float(packet.weak_tz_alignment),
                "weakness_sum": float(np.sum(packet.weakness)),
                "depth_observation_scale": float(observation_scale),
                "depth_observation_std_eff_mm": float(base_observation_std * math.sqrt(float(observation_scale))),
                "depth_guard_px": float(guard),
                "current_reproj_weighted_rms_px": float(current_wrms),
                "raw_reproj_weighted_rms_px": float(raw_wrms),
                "reproj_excess_px": float(current_wrms - raw_wrms),
                "opt_cost": math.nan,
                "opt_nfev": 0.0,
            }
        )
    elapsed_s = time.perf_counter() - start_time
    summary = summarize_rows(rows)
    summary.update(
        {
            "run_id": str(run.get("run_id", "")),
            "frames": int(len(rows)),
            "elapsed_s": float(elapsed_s),
            "mean_nfev": 0.0,
            "mean_cost": math.nan,
        }
    )
    return rows, summary


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    raw = np.asarray(
        [[row["raw_tvec_x_mm"], row["raw_tvec_y_mm"], row["raw_tvec_z_mm"]] for row in rows],
        dtype=np.float64,
    )
    out = np.asarray(
        [[row["tvec_x_mm"], row["tvec_y_mm"], row["tvec_z_mm"]] for row in rows],
        dtype=np.float64,
    )
    raw_ranges = np.ptp(raw, axis=0)
    out_ranges = np.ptp(out, axis=0)
    axis = int(np.nanargmax(raw_ranges))
    lag = _best_lag_metrics(raw[:, axis] - raw[0, axis], out[:, axis] - out[0, axis], max_lag=8)
    excess = [float(row["reproj_excess_px"]) for row in rows]
    current_wrms = [float(row["current_reproj_weighted_rms_px"]) for row in rows]
    raw_wrms = [float(row["raw_reproj_weighted_rms_px"]) for row in rows]
    switch_scores = [float(row["switch_score"]) for row in rows]
    conditions = [float(row["condition_number"]) for row in rows]
    weak_tz = [float(row["weak_tz_alignment"]) for row in rows]
    observation_scales = [float(row.get("depth_observation_scale", 1.0)) for row in rows]
    guard_values = [float(row.get("depth_guard_px", math.nan)) for row in rows]
    return {
        "raw_x_range_mm": float(raw_ranges[0]),
        "raw_y_range_mm": float(raw_ranges[1]),
        "raw_z_range_mm": float(raw_ranges[2]),
        "map_x_range_mm": float(out_ranges[0]),
        "map_y_range_mm": float(out_ranges[1]),
        "map_z_range_mm": float(out_ranges[2]),
        "main_axis": TWIST_NAMES[axis],
        "main_axis_raw_range_mm": float(raw_ranges[axis]),
        "main_axis_map_range_mm": float(out_ranges[axis]),
        "main_axis_ratio": float(out_ranges[axis] / raw_ranges[axis])
        if raw_ranges[axis] > 1e-12
        else math.nan,
        "best_lag_steps": lag["best_lag_steps"],
        "best_lag_corr": lag["best_lag_corr"],
        "best_lag_rmse_mm": lag["best_lag_rmse_mm"],
        "reproj_excess_median_px": float(np.nanmedian(excess)),
        "reproj_excess_p95_px": _percentile(excess, 95),
        "current_wrms_median_px": float(np.nanmedian(current_wrms)),
        "raw_wrms_median_px": float(np.nanmedian(raw_wrms)),
        "switch_median": float(np.nanmedian(switch_scores)),
        "switch_p95": _percentile(switch_scores, 95),
        "condition_median": float(np.nanmedian(conditions)),
        "weak_tz_alignment_median": float(np.nanmedian(weak_tz)),
        "depth_observation_scale_median": float(np.nanmedian(observation_scales)),
        "depth_observation_scale_p95": _percentile(observation_scales, 95),
        "depth_guard_median_px": float(np.nanmedian(guard_values)),
        "depth_guard_p95_px": _percentile(guard_values, 95),
    }


def write_debug_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_translation(rows: list[dict[str, Any]], path: Path, *, title: str) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray([row["frame"] for row in rows], dtype=np.float64)
    raw = np.asarray(
        [[row["raw_tvec_x_mm"], row["raw_tvec_y_mm"], row["raw_tvec_z_mm"]] for row in rows],
        dtype=np.float64,
    )
    out = np.asarray(
        [[row["tvec_x_mm"], row["tvec_y_mm"], row["tvec_z_mm"]] for row in rows],
        dtype=np.float64,
    )
    raw_rel = raw - raw[0]
    out_rel = out - out[0]

    colors = ("#1f77b4", "#2ca02c", "#d62728")
    labels = ("x", "y", "z")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title)
    for idx, ax in enumerate(axes[:3]):
        ax.plot(frames, raw_rel[:, idx], "--", color="#8a94a3", label="logged/raw PnP")
        ax.plot(frames, out_rel[:, idx], color=colors[idx], label="fixed-lag MAP")
        ax.set_ylabel(f"{labels[idx]} rel. [mm]")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper right")
    excess = np.asarray([row["reproj_excess_px"] for row in rows], dtype=np.float64)
    raw_wrms = np.asarray([row["raw_reproj_weighted_rms_px"] for row in rows], dtype=np.float64)
    cur_wrms = np.asarray([row["current_reproj_weighted_rms_px"] for row in rows], dtype=np.float64)
    axes[3].plot(frames, raw_wrms, "--", color="#8a94a3", label="raw current WRMS")
    axes[3].plot(frames, cur_wrms, color="#ff7f0e", label="MAP current WRMS")
    axes[3].plot(frames, excess, color="#9467bd", label="excess")
    axes[3].set_ylabel("reproj. [px]")
    axes[3].set_xlabel("frame")
    axes[3].grid(True, alpha=0.22)
    axes[3].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _select_jsonl_with_qt() -> Path | None:
    try:
        from debug_tracker_translation import select_jsonl_with_qt
    except Exception as exc:
        raise RuntimeError("Could not load Qt file picker.") from exc
    return select_jsonl_with_qt()


def _format_summary(summary: dict[str, Any]) -> str:
    return (
        f"{summary.get('run_id', '')}: "
        f"raw xyz=({summary['raw_x_range_mm']:.3f},"
        f"{summary['raw_y_range_mm']:.3f},{summary['raw_z_range_mm']:.3f}) mm | "
        f"map xyz=({summary['map_x_range_mm']:.3f},"
        f"{summary['map_y_range_mm']:.3f},{summary['map_z_range_mm']:.3f}) mm | "
        f"axis={summary['main_axis']} ratio={summary['main_axis_ratio']:.3f} "
        f"lag={summary['best_lag_steps']:.0f} "
        f"excess95={summary['reproj_excess_p95_px']:.3f}px "
        f"frames={summary.get('frames', 0)} time={summary.get('elapsed_s', math.nan):.2f}s"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-lag SE(3) MAP replay for HydraMarker tracking logs."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="HydraMarker JSONL run logs.")
    parser.add_argument("--all-default-runs", action="store_true", help="Evaluate the built-in debug logs.")
    parser.add_argument("--point-set", choices=("correspondence", "pose"), default="correspondence")
    parser.add_argument("--algorithm", choices=("map", "depth-kalman"), default=FixedLagMapConfig.algorithm)
    parser.add_argument("--window-size", type=int, default=FixedLagMapConfig.window_size)
    parser.add_argument("--robust-c-px", type=float, default=FixedLagMapConfig.robust_c_px)
    parser.add_argument("--uv-stability-scale-px", type=float, default=FixedLagMapConfig.uv_stability_scale_px)
    parser.add_argument("--age-ramp-frames", type=int, default=FixedLagMapConfig.age_ramp_frames)
    parser.add_argument("--max-points-per-frame", type=int, default=FixedLagMapConfig.max_points_per_frame)
    parser.add_argument("--measurement-weight", type=float, default=FixedLagMapConfig.measurement_weight)
    parser.add_argument("--condition-boost", type=float, default=FixedLagMapConfig.condition_boost)
    parser.add_argument("--condition-boost-cap", type=float, default=FixedLagMapConfig.condition_boost_cap)
    parser.add_argument("--switch-reproj-downweight", type=float, default=FixedLagMapConfig.switch_reproj_downweight)
    parser.add_argument("--switch-temporal-boost", type=float, default=FixedLagMapConfig.switch_temporal_boost)
    parser.add_argument("--anchor-weight", type=float, default=FixedLagMapConfig.anchor_weight)
    parser.add_argument("--seed-mode", choices=("raw", "predict"), default=FixedLagMapConfig.seed_mode)
    parser.add_argument("--output-mode", choices=("online", "delayed"), default=FixedLagMapConfig.output_mode)
    parser.add_argument("--velocity-weight", type=float, default=FixedLagMapConfig.velocity_weight)
    parser.add_argument("--acceleration-weight", type=float, default=FixedLagMapConfig.acceleration_weight)
    parser.add_argument("--depth-velocity-weight", type=float, default=FixedLagMapConfig.depth_velocity_weight)
    parser.add_argument("--depth-acceleration-weight", type=float, default=FixedLagMapConfig.depth_acceleration_weight)
    parser.add_argument("--camera-z-velocity-weight", type=float, default=FixedLagMapConfig.camera_z_velocity_weight)
    parser.add_argument("--camera-z-acceleration-weight", type=float, default=FixedLagMapConfig.camera_z_acceleration_weight)
    parser.add_argument("--weak-velocity-weight", type=float, default=FixedLagMapConfig.weak_velocity_weight)
    parser.add_argument("--weak-acceleration-weight", type=float, default=FixedLagMapConfig.weak_acceleration_weight)
    parser.add_argument("--weak-eig-ratio", type=float, default=FixedLagMapConfig.weak_eig_ratio)
    parser.add_argument("--weak-power", type=float, default=FixedLagMapConfig.weak_power)
    parser.add_argument("--weak-cap", type=float, default=FixedLagMapConfig.weak_cap)
    parser.add_argument("--step-translation-mm", type=float, default=FixedLagMapConfig.step_translation_mm)
    parser.add_argument("--step-rotation-deg", type=float, default=FixedLagMapConfig.step_rotation_deg)
    parser.add_argument("--var-translation-bound-mm", type=float, default=FixedLagMapConfig.var_translation_bound_mm)
    parser.add_argument("--var-rotation-bound-deg", type=float, default=FixedLagMapConfig.var_rotation_bound_deg)
    parser.add_argument("--max-nfev", type=int, default=FixedLagMapConfig.max_nfev)
    parser.add_argument("--solver", choices=("linearized", "scipy"), default=FixedLagMapConfig.solver)
    parser.add_argument("--lm-damping", type=float, default=FixedLagMapConfig.lm_damping)
    parser.add_argument("--loss", choices=("linear", "soft_l1", "huber", "cauchy", "arctan"), default=FixedLagMapConfig.loss)
    parser.add_argument("--residual-clip", type=float, default=FixedLagMapConfig.residual_clip)
    parser.add_argument("--depth-filter-mode", choices=("causal", "fixed-lag", "rts"), default=FixedLagMapConfig.depth_filter_mode)
    parser.add_argument("--depth-filter-lag-frames", type=int, default=FixedLagMapConfig.depth_filter_lag_frames)
    parser.add_argument("--depth-observation-std-mm", type=float, default=FixedLagMapConfig.depth_observation_std_mm)
    parser.add_argument("--depth-process-std-mm", type=float, default=FixedLagMapConfig.depth_process_std_mm)
    parser.add_argument("--depth-initial-velocity-std-mm", type=float, default=FixedLagMapConfig.depth_initial_velocity_std_mm)
    parser.add_argument("--depth-switch-observation-gain", type=float, default=FixedLagMapConfig.depth_switch_observation_gain)
    parser.add_argument("--depth-switch-observation-power", type=float, default=FixedLagMapConfig.depth_switch_observation_power)
    parser.add_argument("--depth-weak-tz-observation-gain", type=float, default=FixedLagMapConfig.depth_weak_tz_observation_gain)
    parser.add_argument("--depth-weak-tz-observation-power", type=float, default=FixedLagMapConfig.depth_weak_tz_observation_power)
    parser.add_argument("--depth-observation-scale-cap", type=float, default=FixedLagMapConfig.depth_observation_scale_cap)
    parser.add_argument("--depth-reprojection-guard-px", type=float, default=FixedLagMapConfig.depth_reprojection_guard_px)
    parser.add_argument("--depth-guard-weak-tz-low", type=float, default=FixedLagMapConfig.depth_guard_weak_tz_low)
    parser.add_argument("--depth-guard-weak-tz-high", type=float, default=FixedLagMapConfig.depth_guard_weak_tz_high)
    parser.add_argument("--depth-guard-low-weak-tz-px", type=float, default=FixedLagMapConfig.depth_guard_low_weak_tz_px)
    parser.add_argument("--depth-guard-high-weak-tz-px", type=float, default=FixedLagMapConfig.depth_guard_high_weak_tz_px)
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> FixedLagMapConfig:
    return FixedLagMapConfig(
        algorithm=str(args.algorithm),
        window_size=int(args.window_size),
        robust_c_px=float(args.robust_c_px),
        uv_stability_scale_px=float(args.uv_stability_scale_px),
        age_ramp_frames=int(args.age_ramp_frames),
        max_points_per_frame=int(args.max_points_per_frame),
        measurement_weight=float(args.measurement_weight),
        condition_boost=float(args.condition_boost),
        condition_boost_cap=float(args.condition_boost_cap),
        switch_reproj_downweight=float(args.switch_reproj_downweight),
        switch_temporal_boost=float(args.switch_temporal_boost),
        anchor_weight=float(args.anchor_weight),
        seed_mode=str(args.seed_mode),
        output_mode=str(args.output_mode),
        velocity_weight=float(args.velocity_weight),
        acceleration_weight=float(args.acceleration_weight),
        depth_velocity_weight=float(args.depth_velocity_weight),
        depth_acceleration_weight=float(args.depth_acceleration_weight),
        camera_z_velocity_weight=float(args.camera_z_velocity_weight),
        camera_z_acceleration_weight=float(args.camera_z_acceleration_weight),
        weak_velocity_weight=float(args.weak_velocity_weight),
        weak_acceleration_weight=float(args.weak_acceleration_weight),
        weak_eig_ratio=float(args.weak_eig_ratio),
        weak_power=float(args.weak_power),
        weak_cap=float(args.weak_cap),
        step_translation_mm=float(args.step_translation_mm),
        step_rotation_deg=float(args.step_rotation_deg),
        var_translation_bound_mm=float(args.var_translation_bound_mm),
        var_rotation_bound_deg=float(args.var_rotation_bound_deg),
        max_nfev=int(args.max_nfev),
        solver=str(args.solver),
        lm_damping=float(args.lm_damping),
        loss=str(args.loss),
        residual_clip=float(args.residual_clip),
        depth_filter_mode=str(args.depth_filter_mode),
        depth_filter_lag_frames=int(args.depth_filter_lag_frames),
        depth_observation_std_mm=float(args.depth_observation_std_mm),
        depth_process_std_mm=float(args.depth_process_std_mm),
        depth_initial_velocity_std_mm=float(args.depth_initial_velocity_std_mm),
        depth_switch_observation_gain=float(args.depth_switch_observation_gain),
        depth_switch_observation_power=float(args.depth_switch_observation_power),
        depth_weak_tz_observation_gain=float(args.depth_weak_tz_observation_gain),
        depth_weak_tz_observation_power=float(args.depth_weak_tz_observation_power),
        depth_observation_scale_cap=float(args.depth_observation_scale_cap),
        depth_reprojection_guard_px=float(args.depth_reprojection_guard_px),
        depth_guard_weak_tz_low=float(args.depth_guard_weak_tz_low),
        depth_guard_weak_tz_high=float(args.depth_guard_weak_tz_high),
        depth_guard_low_weak_tz_px=float(args.depth_guard_low_weak_tz_px),
        depth_guard_high_weak_tz_px=float(args.depth_guard_high_weak_tz_px),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.all_default_runs:
        paths = [Path(p) for p in DEFAULT_RUNS]
    elif args.paths:
        paths = [Path(p) for p in args.paths]
    else:
        selected = _select_jsonl_with_qt()
        if selected is None:
            return
        paths = [selected]

    config = config_from_args(args)
    for path in paths:
        run = load_run(path, point_set=str(args.point_set))
        if config.algorithm == "depth-kalman":
            rows, summary = run_depth_kalman(run, config)
        else:
            rows, summary = run_fixed_lag_map(run, config)
        print(_format_summary(summary))
        if args.summary_only:
            continue
        stem = path.with_suffix("")
        if not args.no_csv:
            write_debug_csv(rows, stem.with_name(stem.name + "_fixed_lag_map_debug.csv"))
        if not args.no_plot:
            title = (
                f"{config.algorithm} translation replay\n"
                f"{path.name} | W={config.window_size} c={config.robust_c_px}px "
                f"depth={config.depth_filter_mode}/{config.depth_filter_lag_frames} "
                f"Rz={config.depth_observation_std_mm:g} qz={config.depth_process_std_mm:g}"
            )
            plot_translation(rows, stem.with_name(stem.name + "_fixed_lag_map_translation_plot.png"), title=title)


if __name__ == "__main__":
    main()
