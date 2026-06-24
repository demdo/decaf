"""Triggered pose-prior refinement for reprojection-consistent Z plateaus."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from tracking.pose_solvers import make_transform_from_rvec_tvec


@dataclass
class PlateauPosePriorResult:
    """One accepted or rejected triggered pose-prior candidate."""

    success: bool
    method: str
    reason: str = ""

    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    T_marker_camera: np.ndarray | None = None

    reprojection_mean_px: float = math.nan
    reprojection_max_px: float = math.nan
    reprojection_excess_px: float = math.nan
    max_reprojection_excess_px: float = math.nan
    delta_z_mm: float = math.nan
    iterations: int = 0


def solve_plateau_pose_prior(
    *,
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    raw_rvec: np.ndarray,
    raw_tvec: np.ndarray,
    seed_rvec: np.ndarray,
    seed_tvec: np.ndarray,
    static_max_excess_px: float = 0.18,
    candidate_max_excess_px: float = 0.25,
    candidate_max_max_excess_px: float = 1.00,
    min_positive_z_correction_mm: float = 0.0,
    max_positive_z_correction_mm: float = 0.75,
    robust_c_px: float = 0.20,
    max_iterations: int = 6,
    max_step_translation_mm: float = 5.0,
    max_step_rotation_deg: float = 5.0,
    lm_damping: float = 1.0e-5,
) -> PlateauPosePriorResult:
    """Return a pose-prior candidate for a detected negative-Z plateau.

    The normal PnP result remains the reference. This helper only accepts a
    candidate if the previous pose, or a short robust IRLS refinement from it,
    explains the current 2D observations with a very small reprojection cost
    increase and moves camera-Z in the direction needed to counter the plateau.
    """

    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(obj) < 6 or len(obj) != len(img):
        return PlateauPosePriorResult(False, "none", "invalid_points")

    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    raw_r = np.asarray(raw_rvec, dtype=np.float64).reshape(3, 1)
    raw_t = np.asarray(raw_tvec, dtype=np.float64).reshape(3, 1)
    seed_r = np.asarray(seed_rvec, dtype=np.float64).reshape(3, 1)
    seed_t = np.asarray(seed_tvec, dtype=np.float64).reshape(3, 1)

    raw_stats = _pose_stats(obj, img, K_arr, dist, raw_r, raw_t)
    if raw_stats is None:
        return PlateauPosePriorResult(False, "none", "raw_projection_failed")

    static = _candidate_from_pose(
        "static",
        obj,
        img,
        K_arr,
        dist,
        raw_t,
        seed_r,
        seed_t,
        raw_stats,
        iterations=0,
    )
    if _candidate_allowed(
        static,
        static_max_excess_px=static_max_excess_px,
        candidate_max_excess_px=static_max_excess_px,
        candidate_max_max_excess_px=candidate_max_max_excess_px,
        min_positive_z_correction_mm=min_positive_z_correction_mm,
        max_positive_z_correction_mm=max_positive_z_correction_mm,
    ):
        return static

    robust = _solve_robust_irls(
        obj,
        img,
        K_arr,
        dist,
        raw_t,
        seed_r,
        seed_t,
        raw_stats,
        robust_c_px=robust_c_px,
        max_iterations=max_iterations,
        max_step_translation_mm=max_step_translation_mm,
        max_step_rotation_deg=max_step_rotation_deg,
        lm_damping=lm_damping,
    )
    if _candidate_allowed(
        robust,
        static_max_excess_px=static_max_excess_px,
        candidate_max_excess_px=candidate_max_excess_px,
        candidate_max_max_excess_px=candidate_max_max_excess_px,
        min_positive_z_correction_mm=min_positive_z_correction_mm,
        max_positive_z_correction_mm=max_positive_z_correction_mm,
    ):
        return robust

    reason = (
        robust.reason
        if robust.reason
        else "no_candidate_within_reprojection_budget"
    )
    return PlateauPosePriorResult(
        False,
        "none",
        reason=reason,
        reprojection_mean_px=robust.reprojection_mean_px,
        reprojection_max_px=robust.reprojection_max_px,
        reprojection_excess_px=robust.reprojection_excess_px,
        max_reprojection_excess_px=robust.max_reprojection_excess_px,
        delta_z_mm=robust.delta_z_mm,
        iterations=robust.iterations,
    )


def _candidate_allowed(
    candidate: PlateauPosePriorResult,
    *,
    static_max_excess_px: float,
    candidate_max_excess_px: float,
    candidate_max_max_excess_px: float,
    min_positive_z_correction_mm: float,
    max_positive_z_correction_mm: float,
) -> bool:
    if not candidate.success:
        return False
    if not math.isfinite(candidate.reprojection_excess_px):
        return False
    if candidate.method == "static":
        if candidate.reprojection_excess_px > float(static_max_excess_px):
            return False
    elif candidate.reprojection_excess_px > float(candidate_max_excess_px):
        return False
    if (
        math.isfinite(candidate.max_reprojection_excess_px)
        and candidate.max_reprojection_excess_px > float(candidate_max_max_excess_px)
    ):
        return False
    if candidate.delta_z_mm < float(min_positive_z_correction_mm):
        return False
    if candidate.delta_z_mm > float(max_positive_z_correction_mm):
        return False
    return True


def _solve_robust_irls(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    raw_tvec: np.ndarray,
    seed_rvec: np.ndarray,
    seed_tvec: np.ndarray,
    raw_stats: tuple[float, float],
    *,
    robust_c_px: float,
    max_iterations: int,
    max_step_translation_mm: float,
    max_step_rotation_deg: float,
    lm_damping: float,
) -> PlateauPosePriorResult:
    try:
        T = make_transform_from_rvec_tvec(seed_rvec, seed_tvec)
    except Exception:
        return PlateauPosePriorResult(False, "robust_irls", "seed_transform_failed")

    last_iteration = 0
    for iteration in range(1, max(1, int(max_iterations)) + 1):
        last_iteration = iteration
        projected = _project_points(object_points, T, K, dist)
        if projected is None:
            return PlateauPosePriorResult(False, "robust_irls", "projection_failed")
        residual = (projected - image_points).reshape(-1)
        errors = np.linalg.norm(projected - image_points, axis=1)
        robust = 1.0 / (max(float(robust_c_px), 1.0e-9) + errors)
        finite = robust[np.isfinite(robust) & (robust > 0.0)]
        if len(finite):
            robust = robust / float(np.max(finite))
        robust = np.where(np.isfinite(robust), robust, 0.0)
        weights = np.repeat(robust, 2)
        J = _numeric_motion_jacobian(object_points, T, K, dist)
        if J is None:
            return PlateauPosePriorResult(False, "robust_irls", "jacobian_failed")

        C = J.T @ (weights[:, None] * J)
        g = J.T @ (weights * residual)
        damping = float(lm_damping) * np.maximum(np.diag(C), 1.0e-9)
        normal = C + np.diag(damping)
        try:
            delta = np.linalg.solve(normal, -g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(normal, -g, rcond=None)[0]

        delta = np.asarray(delta, dtype=np.float64).reshape(6)
        t_norm = float(np.linalg.norm(delta[:3]))
        r_norm = float(np.linalg.norm(delta[3:]))
        max_t = max(float(max_step_translation_mm), 1.0e-9)
        max_r = math.radians(max(float(max_step_rotation_deg), 1.0e-9))
        scale = 1.0
        if t_norm > max_t:
            scale = min(scale, max_t / max(t_norm, 1.0e-12))
        if r_norm > max_r:
            scale = min(scale, max_r / max(r_norm, 1.0e-12))
        delta *= scale

        T = _exp_se3(delta) @ T
        if t_norm < 1.0e-5 and r_norm < 1.0e-8:
            break

    rvec, tvec = _T_to_pose(T)
    return _candidate_from_pose(
        "robust_irls",
        object_points,
        image_points,
        K,
        dist,
        raw_tvec,
        rvec,
        tvec,
        raw_stats,
        iterations=last_iteration,
    )


def _candidate_from_pose(
    method: str,
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    raw_tvec: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    raw_stats: tuple[float, float],
    *,
    iterations: int,
) -> PlateauPosePriorResult:
    stats = _pose_stats(object_points, image_points, K, dist, rvec, tvec)
    if stats is None:
        return PlateauPosePriorResult(False, method, "projection_failed")
    mean_px, max_px = stats
    raw_mean, raw_max = raw_stats
    try:
        T = make_transform_from_rvec_tvec(rvec, tvec)
    except Exception:
        T = None
    return PlateauPosePriorResult(
        success=True,
        method=method,
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        T_marker_camera=None if T is None else np.asarray(T, dtype=np.float64).reshape(4, 4),
        reprojection_mean_px=float(mean_px),
        reprojection_max_px=float(max_px),
        reprojection_excess_px=float(mean_px - raw_mean),
        max_reprojection_excess_px=float(max_px - raw_max),
        delta_z_mm=float(np.asarray(tvec, dtype=np.float64).reshape(3)[2] - np.asarray(raw_tvec, dtype=np.float64).reshape(3)[2]),
        iterations=int(iterations),
    )


def _pose_stats(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[float, float] | None:
    try:
        projected, _ = cv2.projectPoints(
            np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            np.asarray(K, dtype=np.float64).reshape(3, 3),
            np.asarray(dist, dtype=np.float64).reshape(-1, 1),
        )
    except Exception:
        return None
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    if len(errors) == 0:
        return None
    return float(np.mean(errors)), float(np.max(errors))


def _project_points(
    object_points: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray | None:
    try:
        rvec, tvec = _T_to_pose(T)
        projected, _ = cv2.projectPoints(
            np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
            rvec,
            tvec,
            K,
            dist,
        )
    except Exception:
        return None
    return np.asarray(projected, dtype=np.float64).reshape(-1, 2)


def _numeric_motion_jacobian(
    object_points: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray | None:
    base = _project_points(object_points, T, K, dist)
    if base is None:
        return None
    base_vec = base.reshape(-1)
    J = np.zeros((base_vec.size, 6), dtype=np.float64)
    for col in range(6):
        eps = 1.0e-3 if col < 3 else 1.0e-6
        delta = np.zeros(6, dtype=np.float64)
        delta[col] = eps
        shifted = _project_points(object_points, _exp_se3(delta) @ T, K, dist)
        if shifted is None:
            return None
        J[:, col] = (shifted.reshape(-1) - base_vec) / eps
    return J


def _exp_se3(delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64).reshape(6)
    v = delta[:3]
    w = delta[3:]
    R = _exp_so3(w)
    V = _left_jacobian_so3(w)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def _exp_so3(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    if theta < 1.0e-12:
        return np.eye(3, dtype=np.float64) + W + 0.5 * (W @ W)
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    return np.eye(3, dtype=np.float64) + a * W + b * (W @ W)


def _left_jacobian_so3(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    if theta < 1.0e-12:
        return np.eye(3, dtype=np.float64) + 0.5 * W + (1.0 / 6.0) * (W @ W)
    b = (1.0 - math.cos(theta)) / (theta * theta)
    c = (theta - math.sin(theta)) / (theta * theta * theta)
    return np.eye(3, dtype=np.float64) + b * W + c * (W @ W)


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _T_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return (
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(T[:3, 3], dtype=np.float64).reshape(3, 1),
    )
