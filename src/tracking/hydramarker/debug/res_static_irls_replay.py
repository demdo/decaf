from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from res_static_uv_freeze_replay import (  # noqa: E402
    Observation,
    _mean,
    _median,
    _percentile,
    _rms,
    _rvec_to_euler_deg,
    _to_float,
    _to_int,
    build_static_uv_model,
    load_run,
    solve_pose as solve_pose_opencv,
)


TWIST_COLUMNS = (
    "dtx_mm",
    "dty_mm",
    "dtz_mm",
    "drx_rad",
    "dry_rad",
    "drz_rad",
)


def _load_cv2():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "OpenCV/cv2 is required. Run this with the decaf environment, e.g. "
            "C:\\Users\\domin\\anaconda3\\envs\\decaf\\python.exe"
        ) from exc
    return cv2


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _exp_so3(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + W + 0.5 * (W @ W)
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    return np.eye(3, dtype=np.float64) + a * W + b * (W @ W)


def _left_jacobian_so3(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + 0.5 * W + (1.0 / 6.0) * (W @ W)
    b = (1.0 - math.cos(theta)) / (theta * theta)
    c = (theta - math.sin(theta)) / (theta * theta * theta)
    return np.eye(3, dtype=np.float64) + b * W + c * (W @ W)


def _exp_se3_paper_order(delta: np.ndarray) -> np.ndarray:
    """SE(3) exponential using Drummond/Cipolla's generator order.

    The paper lists the three translations first and the three rotations
    second.  We keep that order throughout the script:
    [tx_mm, ty_mm, tz_mm, rx_rad, ry_rad, rz_rad].
    """

    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    v = delta[:3]
    w = delta[3:]
    R = _exp_so3(w)
    V = _left_jacobian_so3(w)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def _pose_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    cv2 = _load_cv2()
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def _T_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _load_cv2()
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return (
        np.asarray(rvec, dtype=np.float64).reshape(3),
        np.asarray(T[:3, 3], dtype=np.float64).reshape(3),
    )


def _project_points(
    object_points: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    cv2 = _load_cv2()
    rvec, tvec = _T_to_pose(T)
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        np.asarray(dist, dtype=np.float64).reshape(-1, 1),
    )
    return np.asarray(projected, dtype=np.float64).reshape(-1, 2)


def _numeric_motion_jacobian(
    object_points: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    eps_translation_mm: float = 1e-3,
    eps_rotation_rad: float = 1e-6,
) -> np.ndarray:
    base = _project_points(object_points, T, K, dist).reshape(-1)
    J = np.zeros((base.size, 6), dtype=np.float64)
    for col in range(6):
        eps = eps_translation_mm if col < 3 else eps_rotation_rad
        delta = np.zeros(6, dtype=np.float64)
        delta[col] = eps
        plus_T = _exp_se3_paper_order(delta) @ T
        plus = _project_points(object_points, plus_T, K, dist).reshape(-1)
        J[:, col] = (plus - base) / eps
    return J


def _finite_condition_number(values: np.ndarray) -> float:
    eig = np.asarray(values, dtype=np.float64).reshape(-1)
    eig = eig[np.isfinite(eig) & (eig > 1e-12)]
    if len(eig) < 2:
        return math.nan
    return float(np.max(eig) / np.min(eig))


def _weighted_reprojection_stats(
    residual: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    residual = np.asarray(residual, dtype=np.float64).reshape(-1, 2)
    errors = np.sqrt(np.sum(residual * residual, axis=1))
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(np.sum(weights)) > 0.0:
        weighted_rms = math.sqrt(float(np.sum(weights * errors * errors) / np.sum(weights)))
    else:
        weighted_rms = math.nan
    return {
        "reproj_mean_px": _mean(errors),
        "reproj_median_px": _median(errors),
        "reproj_rms_px": _rms(errors),
        "reproj_weighted_rms_px": float(weighted_rms),
        "reproj_p95_px": _percentile(errors, 95),
        "reproj_max_px": float(np.max(errors)) if len(errors) else math.nan,
    }


@dataclass
class PointPrior:
    presence_fraction: float
    uv_rms_px: float
    uv_p95_px: float
    base_weight: float


@dataclass
class IrlsConfig:
    max_iterations: int = 8
    robust_c_px: float = 0.20
    uv_stability_scale_px: float = 0.08
    min_base_weight: float = 0.05
    presence_power: float = 1.0
    age_ramp_frames: int = 4
    condition_boost: float = 1.0
    lm_damping: float = 1e-5
    max_step_translation_mm: float = 2.0
    max_step_rotation_deg: float = 1.0
    active_dofs: tuple[int, ...] | None = None


@dataclass
class IrlsResult:
    success: bool
    T: np.ndarray
    iterations: int
    point_count: int
    stats: dict[str, float]
    mean_weight: float
    condition_number: float
    min_eigenvalue: float
    weak_z_alignment: float
    last_step_norm: float


def build_point_priors(
    observations: list[Observation],
    static_model: dict[str, Any],
    config: IrlsConfig,
) -> dict[tuple[int, int], PointPrior]:
    point_rows = list(static_model["point_rows"])
    priors: dict[tuple[int, int], PointPrior] = {}
    for row in point_rows:
        key = (_to_int(row.get("global_row"), -1), _to_int(row.get("global_col"), -1))
        if key[0] < 0 or key[1] < 0:
            continue
        presence = _to_float(row.get("present_fraction"))
        uv_rms = _to_float(row.get("uv_motion_rms_px"))
        uv_p95 = _to_float(row.get("uv_motion_p95_px"))
        if not np.isfinite(presence):
            presence = 0.0
        if not np.isfinite(uv_rms):
            uv_rms = 999.0
        scale = max(float(config.uv_stability_scale_px), 1e-9)
        stability = 1.0 / (1.0 + (float(uv_rms) / scale) ** 2)
        support = max(float(presence), 0.0) ** max(float(config.presence_power), 0.0)
        base = support * stability
        base = float(np.clip(base, config.min_base_weight, 1.0))
        priors[key] = PointPrior(
            presence_fraction=float(presence),
            uv_rms_px=float(uv_rms),
            uv_p95_px=float(uv_p95),
            base_weight=base,
        )
    return priors


def build_age_maps(
    observations: list[Observation],
    *,
    age_ramp_frames: int,
) -> list[dict[tuple[int, int], float]]:
    streaks: dict[tuple[int, int], int] = {}
    age_maps: list[dict[tuple[int, int], float]] = []
    ramp = max(int(age_ramp_frames), 1)
    for obs in observations:
        current = set(obs.uv_by_key)
        next_streaks: dict[tuple[int, int], int] = {}
        for key in current:
            next_streaks[key] = int(streaks.get(key, 0)) + 1
        streaks = next_streaks
        age_maps.append({key: min(1.0, float(age) / float(ramp)) for key, age in streaks.items()})
    return age_maps


def arrays_for_observation(
    obs: Observation,
    *,
    priors: dict[tuple[int, int], PointPrior],
    age_weight_by_key: dict[tuple[int, int], float],
    frame_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    weights: list[float] = []
    used_keys: list[tuple[int, int]] = []
    for key in sorted(obs.uv_by_key):
        prior = priors.get(key)
        if prior is None:
            continue
        xyz = np.asarray(obs.object_by_key[key], dtype=np.float64).reshape(3)
        uv = np.asarray(obs.uv_by_key[key], dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(uv)):
            continue
        age_weight = float(age_weight_by_key.get(key, 1.0))
        object_points.append(xyz)
        image_points.append(uv)
        weights.append(float(frame_weight) * prior.base_weight * age_weight)
        used_keys.append(key)
    return (
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
        np.asarray(weights, dtype=np.float64).reshape(-1),
        used_keys,
    )


def arrays_for_window(
    observations: list[Observation],
    obs_index: int,
    *,
    priors: dict[tuple[int, int], PointPrior],
    age_maps: list[dict[tuple[int, int], float]],
    window_frames: int,
    window_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], int]:
    start = max(0, int(obs_index) - max(int(window_frames), 1) + 1)
    object_chunks: list[np.ndarray] = []
    image_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    used_keys: list[tuple[int, int]] = []
    used_frames = 0
    for j in range(start, int(obs_index) + 1):
        age = int(obs_index) - j
        if window_decay > 0.0:
            frame_weight = math.exp(-float(age) / float(window_decay))
        else:
            frame_weight = 1.0
        obj, uv, weights, keys = arrays_for_observation(
            observations[j],
            priors=priors,
            age_weight_by_key=age_maps[j],
            frame_weight=frame_weight,
        )
        if len(obj) == 0:
            continue
        object_chunks.append(obj)
        image_chunks.append(uv)
        weight_chunks.append(weights)
        used_keys.extend(keys)
        used_frames += 1
    if not object_chunks:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            [],
            0,
        )
    return (
        np.vstack(object_chunks).reshape(-1, 3),
        np.vstack(image_chunks).reshape(-1, 2),
        np.concatenate(weight_chunks).reshape(-1),
        used_keys,
        used_frames,
    )


def solve_irls_lie_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    base_weights: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    seed_T: np.ndarray,
    config: IrlsConfig,
) -> IrlsResult:
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    base_weights = np.asarray(base_weights, dtype=np.float64).reshape(-1)
    if len(object_points) < 6:
        return IrlsResult(
            success=False,
            T=np.asarray(seed_T, dtype=np.float64).reshape(4, 4),
            iterations=0,
            point_count=int(len(object_points)),
            stats={},
            mean_weight=math.nan,
            condition_number=math.nan,
            min_eigenvalue=math.nan,
            weak_z_alignment=math.nan,
            last_step_norm=math.nan,
        )

    T = np.asarray(seed_T, dtype=np.float64).reshape(4, 4).copy()
    last_step_norm = math.nan
    condition_number = math.nan
    min_eig = math.nan
    weak_z_alignment = math.nan
    weights = np.clip(base_weights, config.min_base_weight, None)

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

        eigvals, eigvecs = np.linalg.eigh(0.5 * (C + C.T))
        min_eig = float(np.min(eigvals)) if len(eigvals) else math.nan
        condition_number = _finite_condition_number(eigvals)
        if eigvecs.shape == (6, 6):
            weak_vec = eigvecs[:, int(np.argmin(eigvals))]
            weak_z_alignment = float(abs(weak_vec[2]) / max(float(np.linalg.norm(weak_vec)), 1e-12))

        if config.condition_boost > 0.0:
            try:
                C_inv = np.linalg.pinv(C, rcond=1e-10)
                leverage = np.empty(len(object_points), dtype=np.float64)
                for idx in range(len(object_points)):
                    Ji = J[(2 * idx) : (2 * idx + 2), :]
                    leverage[idx] = float(np.trace(Ji @ C_inv @ Ji.T))
                positive = leverage[np.isfinite(leverage) & (leverage > 1e-12)]
                if len(positive):
                    geom_mean = math.exp(float(np.mean(np.log(positive))))
                    salient = leverage > geom_mean
                    saliency = np.ones_like(weights)
                    saliency[salient] += float(config.condition_boost)
                    weights = weights * saliency
                    W2 = np.repeat(weights, 2)
                    C = J.T @ (W2[:, None] * J)
                    g = J.T @ (W2 * residual_vec)
                    eigvals, eigvecs = np.linalg.eigh(0.5 * (C + C.T))
                    min_eig = float(np.min(eigvals)) if len(eigvals) else math.nan
                    condition_number = _finite_condition_number(eigvals)
                    if eigvecs.shape == (6, 6):
                        weak_vec = eigvecs[:, int(np.argmin(eigvals))]
                        weak_z_alignment = float(
                            abs(weak_vec[2]) / max(float(np.linalg.norm(weak_vec)), 1e-12)
                        )
            except Exception:
                pass

        active_dofs = config.active_dofs
        if active_dofs is not None:
            active = np.asarray([int(idx) for idx in active_dofs], dtype=np.int64)
            active = active[(active >= 0) & (active < 6)]
            if len(active) == 0:
                delta = np.zeros(6, dtype=np.float64)
            else:
                C_sub = C[np.ix_(active, active)]
                g_sub = g[active]
                damping = float(config.lm_damping) * np.maximum(np.diag(C_sub), 1e-9)
                normal = C_sub + np.diag(damping)
                rhs = -g_sub
                try:
                    delta_sub = np.linalg.solve(normal, rhs)
                except np.linalg.LinAlgError:
                    delta_sub = np.linalg.lstsq(normal, rhs, rcond=None)[0]
                delta = np.zeros(6, dtype=np.float64)
                delta[active] = np.asarray(delta_sub, dtype=np.float64).reshape(-1)
        else:
            damping = float(config.lm_damping) * np.maximum(np.diag(C), 1e-9)
            normal = C + np.diag(damping)
            rhs = -g
            try:
                delta = np.linalg.solve(normal, rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(normal, rhs, rcond=None)[0]

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

        last_step_norm = float(np.linalg.norm(delta))
        T = _exp_se3_paper_order(delta) @ T
        if translation_norm < 1e-5 and rotation_norm < 1e-8:
            break

    projected = _project_points(object_points, T, K, dist)
    residual = projected - image_points
    stats = _weighted_reprojection_stats(residual, weights)
    return IrlsResult(
        success=True,
        T=T,
        iterations=int(iteration),
        point_count=int(len(object_points)),
        stats=stats,
        mean_weight=float(np.mean(weights)) if len(weights) else math.nan,
        condition_number=float(condition_number),
        min_eigenvalue=float(min_eig),
        weak_z_alignment=float(weak_z_alignment),
        last_step_norm=float(last_step_norm),
    )


def _pose_row(
    *,
    method: str,
    obs: Observation,
    solved: bool,
    rvec: np.ndarray,
    tvec: np.ndarray,
    point_count: int,
    distinct_key_count: int,
    used_frame_count: int,
    stats: dict[str, float],
    iterations: int = 0,
    mean_weight: float = math.nan,
    condition_number: float = math.nan,
    min_eigenvalue: float = math.nan,
    weak_z_alignment: float = math.nan,
    last_step_norm: float = math.nan,
) -> dict[str, Any]:
    roll_deg, pitch_deg, yaw_deg = _rvec_to_euler_deg(np.asarray(rvec, dtype=np.float64).reshape(3))
    row: dict[str, Any] = {
        "method": method,
        "frame": int(obs.frame),
        "solved": int(bool(solved)),
        "point_count": int(point_count),
        "distinct_key_count": int(distinct_key_count),
        "used_frame_count": int(used_frame_count),
        "iterations": int(iterations),
        "mean_weight": float(mean_weight),
        "condition_number": float(condition_number),
        "min_eigenvalue": float(min_eigenvalue),
        "weak_z_alignment": float(weak_z_alignment),
        "last_step_norm": float(last_step_norm),
        "tvec_x_mm": float(tvec[0]),
        "tvec_y_mm": float(tvec[1]),
        "tvec_z_mm": float(tvec[2]),
        "rvec_x_rad": float(rvec[0]),
        "rvec_y_rad": float(rvec[1]),
        "rvec_z_rad": float(rvec[2]),
        "roll_deg": float(roll_deg),
        "pitch_deg": float(pitch_deg),
        "yaw_deg": float(yaw_deg),
    }
    row.update(stats)
    return row


def replay(
    run: dict[str, Any],
    static_model: dict[str, Any],
    *,
    config: IrlsConfig,
    window_frames: int,
    window_decay: float,
    refine: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[Observation] = run["observations"]
    reference = observations[0]
    ref_rvec = reference.original_rvec.copy()
    ref_tvec = reference.original_tvec.copy()
    ref_T = _pose_to_T(ref_rvec, ref_tvec)

    priors = build_point_priors(observations, static_model, config)
    age_maps = build_age_maps(observations, age_ramp_frames=config.age_ramp_frames)

    frame_rows: list[dict[str, Any]] = []
    pose_by_method: dict[str, list[np.ndarray]] = {
        "logged_original": [],
        "pnp_current_ref": [],
        "irls_current": [],
        f"irls_window_{int(window_frames)}": [],
    }
    current_T = ref_T.copy()
    window_T = ref_T.copy()

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

        obj, uv, base_weights, used_keys = arrays_for_observation(
            obs,
            priors=priors,
            age_weight_by_key=age_maps[obs_idx],
        )

        solved, rvec, tvec, stats = solve_pose_opencv(
            obj,
            uv,
            run["K"],
            run["dist"],
            ref_rvec,
            ref_tvec,
            refine=refine,
        )
        if not solved:
            rvec = ref_rvec.copy()
            tvec = ref_tvec.copy()
            stats = {}
        frame_rows.append(
            _pose_row(
                method="pnp_current_ref",
                obs=obs,
                solved=solved,
                rvec=rvec,
                tvec=tvec,
                point_count=len(obj),
                distinct_key_count=len(set(used_keys)),
                used_frame_count=1,
                stats=stats,
            )
        )
        pose_by_method["pnp_current_ref"].append(np.asarray(tvec, dtype=np.float64).reshape(3))

        current_result = solve_irls_lie_pose(
            obj,
            uv,
            base_weights,
            run["K"],
            run["dist"],
            current_T,
            config,
        )
        if current_result.success:
            current_T = current_result.T.copy()
        current_rvec, current_tvec = _T_to_pose(current_T)
        frame_rows.append(
            _pose_row(
                method="irls_current",
                obs=obs,
                solved=current_result.success,
                rvec=current_rvec,
                tvec=current_tvec,
                point_count=current_result.point_count,
                distinct_key_count=len(set(used_keys)),
                used_frame_count=1,
                stats=current_result.stats,
                iterations=current_result.iterations,
                mean_weight=current_result.mean_weight,
                condition_number=current_result.condition_number,
                min_eigenvalue=current_result.min_eigenvalue,
                weak_z_alignment=current_result.weak_z_alignment,
                last_step_norm=current_result.last_step_norm,
            )
        )
        pose_by_method["irls_current"].append(current_tvec.copy())

        win_obj, win_uv, win_weights, win_keys, used_frame_count = arrays_for_window(
            observations,
            obs_idx,
            priors=priors,
            age_maps=age_maps,
            window_frames=window_frames,
            window_decay=window_decay,
        )
        window_result = solve_irls_lie_pose(
            win_obj,
            win_uv,
            win_weights,
            run["K"],
            run["dist"],
            window_T,
            config,
        )
        if window_result.success:
            window_T = window_result.T.copy()
        window_rvec, window_tvec = _T_to_pose(window_T)
        window_method = f"irls_window_{int(window_frames)}"
        frame_rows.append(
            _pose_row(
                method=window_method,
                obs=obs,
                solved=window_result.success,
                rvec=window_rvec,
                tvec=window_tvec,
                point_count=window_result.point_count,
                distinct_key_count=len(set(win_keys)),
                used_frame_count=used_frame_count,
                stats=window_result.stats,
                iterations=window_result.iterations,
                mean_weight=window_result.mean_weight,
                condition_number=window_result.condition_number,
                min_eigenvalue=window_result.min_eigenvalue,
                weak_z_alignment=window_result.weak_z_alignment,
                last_step_norm=window_result.last_step_norm,
            )
        )
        pose_by_method[window_method].append(window_tvec.copy())

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

    prior_rows: list[dict[str, Any]] = []
    for key, prior in sorted(priors.items()):
        prior_rows.append(
            {
                "global_row": int(key[0]),
                "global_col": int(key[1]),
                "presence_fraction": float(prior.presence_fraction),
                "uv_rms_px": float(prior.uv_rms_px),
                "uv_p95_px": float(prior.uv_p95_px),
                "base_weight": float(prior.base_weight),
            }
        )
    return frame_rows, prior_rows


def summarize(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods: list[str] = []
    for row in frame_rows:
        method = str(row.get("method") or "")
        if method and method not in methods:
            methods.append(method)

    summary: list[dict[str, Any]] = []
    for method in methods:
        rows = [row for row in frame_rows if str(row.get("method")) == method]
        rel = {
            comp: np.asarray([_to_float(row.get(f"rel_{comp}_mm")) for row in rows], dtype=np.float64)
            for comp in ("x", "y", "z")
        }
        summary.append(
            {
                "method": method,
                "frames": len(rows),
                "solve_failures": int(sum(1 for row in rows if _to_int(row.get("solved"), 0) == 0)),
                "point_count_median": _median([_to_float(row.get("point_count")) for row in rows]),
                "distinct_key_count_median": _median(
                    [_to_float(row.get("distinct_key_count")) for row in rows]
                ),
                "used_frame_count_median": _median(
                    [_to_float(row.get("used_frame_count")) for row in rows]
                ),
                "x_range_mm": float(np.nanmax(rel["x"]) - np.nanmin(rel["x"])),
                "y_range_mm": float(np.nanmax(rel["y"]) - np.nanmin(rel["y"])),
                "z_range_mm": float(np.nanmax(rel["z"]) - np.nanmin(rel["z"])),
                "x_closure_mm": float(rel["x"][-1] - rel["x"][0]),
                "y_closure_mm": float(rel["y"][-1] - rel["y"][0]),
                "z_closure_mm": float(rel["z"][-1] - rel["z"][0]),
                "reproj_rms_median_px": _median([_to_float(row.get("reproj_rms_px")) for row in rows]),
                "reproj_weighted_rms_median_px": _median(
                    [_to_float(row.get("reproj_weighted_rms_px")) for row in rows]
                ),
                "condition_number_median": _median(
                    [_to_float(row.get("condition_number")) for row in rows]
                ),
                "weak_z_alignment_median": _median(
                    [_to_float(row.get("weak_z_alignment")) for row in rows]
                ),
                "mean_weight_median": _median([_to_float(row.get("mean_weight")) for row in rows]),
            }
        )
    return summary


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


def plot_results(
    run: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    *,
    output_suffix: str,
    show: bool,
) -> Path:
    import matplotlib

    matplotlib.use("Agg" if not show else "QtAgg")
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    path: Path = run["path"]
    out_path = path.with_name(f"{path.stem}_{output_suffix}_plot.png")
    methods = []
    for row in frame_rows:
        method = str(row.get("method") or "")
        if method and method not in methods:
            methods.append(method)

    colors = {
        "logged_original": "#d62728",
        "pnp_current_ref": "#4c78a8",
        "irls_current": "#54a24b",
    }
    fallback_colors = ("#f58518", "#9467bd", "#72b7b2", "#e45756")
    for idx, method in enumerate(methods):
        colors.setdefault(method, fallback_colors[idx % len(fallback_colors)])

    rows_by_method = {
        method: [row for row in frame_rows if str(row.get("method")) == method]
        for method in methods
    }

    fig, axes = plt.subplots(4, 1, figsize=(15.5, 10.5), sharex=True)
    fig.suptitle(
        "HydraTracker static Lie-IRLS replay (Drummond/Cipolla style)\n"
        f"{run['run_id']}",
        fontsize=15,
        fontweight="bold",
    )

    for comp, ax in zip(("x", "y", "z"), axes[:3]):
        key = f"rel_{comp}_mm"
        title_parts = []
        for method in methods:
            rows = rows_by_method[method]
            frames = np.asarray([_to_float(row.get("frame")) for row in rows], dtype=np.float64)
            values = np.asarray([_to_float(row.get(key)) for row in rows], dtype=np.float64)
            ax.plot(
                frames,
                values,
                linewidth=1.9 if method in ("logged_original", "irls_current") else 1.35,
                marker="o",
                markersize=2.2,
                color=colors[method],
                label=method,
            )
            finite = values[np.isfinite(values)]
            if len(finite):
                title_parts.append(f"{method}={float(np.max(finite) - np.min(finite)):.3f}")
        ax.axhline(0.0, color="#888888", alpha=0.35, linewidth=1.0, linestyle="--")
        ax.set_ylabel(f"delta {comp} [mm]")
        ax.set_title(f"{comp.upper()} range [mm]: {', '.join(title_parts)}", loc="left")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)

    diag_ax = axes[3]
    for method in methods:
        rows = rows_by_method[method]
        if not method.startswith("irls"):
            continue
        frames = np.asarray([_to_float(row.get("frame")) for row in rows], dtype=np.float64)
        cond = np.asarray([_to_float(row.get("condition_number")) for row in rows], dtype=np.float64)
        diag_ax.plot(
            frames,
            cond,
            linewidth=1.4,
            color=colors[method],
            label=f"{method} condition",
        )
    diag_ax.set_yscale("log")
    diag_ax.set_ylabel("cond(JTWJ)")
    diag_ax.set_xlabel("frame")
    diag_ax.set_title("IRLS conditioning diagnostic", loc="left")
    diag_ax.grid(True)
    diag_ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _parse_args(argv: list[str]) -> dict[str, Any]:
    args: dict[str, Any] = {
        "path": None,
        "point_set": "correspondence",
        "refine": "vvs",
        "window_frames": 30,
        "window_decay": 12.0,
        "robust_c_px": 0.20,
        "uv_stability_scale_px": 0.08,
        "condition_boost": 1.0,
        "age_ramp_frames": 4,
        "max_iterations": 8,
        "show": False,
        "make_plot": True,
    }
    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--point-set":
            idx += 1
            args["point_set"] = argv[idx]
        elif arg == "--refine":
            idx += 1
            args["refine"] = argv[idx]
        elif arg == "--window-frames":
            idx += 1
            args["window_frames"] = int(argv[idx])
        elif arg == "--window-decay":
            idx += 1
            args["window_decay"] = float(argv[idx])
        elif arg == "--robust-c-px":
            idx += 1
            args["robust_c_px"] = float(argv[idx])
        elif arg == "--uv-stability-scale-px":
            idx += 1
            args["uv_stability_scale_px"] = float(argv[idx])
        elif arg == "--condition-boost":
            idx += 1
            args["condition_boost"] = float(argv[idx])
        elif arg == "--age-ramp-frames":
            idx += 1
            args["age_ramp_frames"] = int(argv[idx])
        elif arg == "--max-iterations":
            idx += 1
            args["max_iterations"] = int(argv[idx])
        elif arg == "--show":
            args["show"] = True
        elif arg == "--no-plot":
            args["make_plot"] = False
        elif arg.endswith(".jsonl"):
            args["path"] = Path(arg)
        else:
            raise RuntimeError(f"Unknown option: {arg}")
        idx += 1

    point_set = str(args["point_set"]).strip().lower()
    if point_set not in ("pose", "correspondence"):
        raise RuntimeError("--point-set must be pose or correspondence")
    args["point_set"] = point_set

    refine = str(args["refine"]).strip().lower()
    if refine not in ("none", "lm", "vvs"):
        raise RuntimeError("--refine must be none, lm, or vvs")
    args["refine"] = refine
    return args


def _tag_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _print_summary(paths: dict[str, Path], summary_rows: list[dict[str, Any]]) -> None:
    print(f"[irls_replay] saved frame csv   -> {paths['frame'].resolve()}")
    print(f"[irls_replay] saved summary csv -> {paths['summary'].resolve()}")
    print(f"[irls_replay] saved prior csv   -> {paths['priors'].resolve()}")
    if "plot" in paths:
        print(f"[irls_replay] saved plot        -> {paths['plot'].resolve()}")
    print("[irls_replay] method summary:")
    for row in summary_rows:
        print(
            "  "
            f"{row['method']}: "
            f"z_range={_to_float(row.get('z_range_mm')):.3f} mm, "
            f"z_closure={_to_float(row.get('z_closure_mm')):+.3f} mm, "
            f"rms={_to_float(row.get('reproj_rms_median_px')):.3f} px, "
            f"weighted_rms={_to_float(row.get('reproj_weighted_rms_median_px')):.3f} px, "
            f"cond={_to_float(row.get('condition_number_median')):.2e}, "
            f"weak_z={_to_float(row.get('weak_z_alignment_median')):.3f}, "
            f"points={_to_float(row.get('point_count_median')):.1f}"
        )


def main() -> None:
    args = _parse_args(sys.argv[1:])
    path = args["path"]
    if path is None:
        raise RuntimeError("Pass a HydraTracker JSONL path.")
    path = Path(path).resolve()

    config = IrlsConfig(
        max_iterations=int(args["max_iterations"]),
        robust_c_px=float(args["robust_c_px"]),
        uv_stability_scale_px=float(args["uv_stability_scale_px"]),
        condition_boost=float(args["condition_boost"]),
        age_ramp_frames=int(args["age_ramp_frames"]),
    )

    run = load_run(path, point_set=str(args["point_set"]))
    static_model = build_static_uv_model(run["observations"])
    frame_rows, prior_rows = replay(
        run,
        static_model,
        config=config,
        window_frames=int(args["window_frames"]),
        window_decay=float(args["window_decay"]),
        refine=str(args["refine"]),
    )
    summary_rows = summarize(frame_rows)

    output_suffix = (
        f"irls_w{int(args['window_frames'])}"
        f"_d{_tag_float(float(args['window_decay']))}"
        f"_c{_tag_float(float(args['robust_c_px']))}"
        f"_uv{_tag_float(float(args['uv_stability_scale_px']))}"
        f"_cb{_tag_float(float(args['condition_boost']))}"
        f"_age{int(args['age_ramp_frames'])}"
    )
    frame_csv = path.with_name(f"{path.stem}_{output_suffix}_frames.csv")
    summary_csv = path.with_name(f"{path.stem}_{output_suffix}_summary.csv")
    prior_csv = path.with_name(f"{path.stem}_{output_suffix}_point_priors.csv")
    _write_csv(frame_csv, frame_rows)
    _write_csv(summary_csv, summary_rows)
    _write_csv(prior_csv, prior_rows)
    paths = {"frame": frame_csv, "summary": summary_csv, "priors": prior_csv}
    if bool(args["make_plot"]):
        paths["plot"] = plot_results(
            run,
            frame_rows,
            output_suffix=output_suffix,
            show=bool(args["show"]),
        )
    _print_summary(paths, summary_rows)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"[irls_replay] ERROR: {exc}")
        sys.exit(1)
