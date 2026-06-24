# overlay/tracking/pose_filters.py

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


def _rotation_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.degrees(theta))


def _compute_motion_score(
    *,
    tip_step_mm: float,
    rot_step_deg: float,
    tip_ref_mm: float = 0.8,
    rot_ref_deg: float = 0.4,
    w_tip: float = 0.7,
    w_rot: float = 0.3,
) -> float:
    motion_tip = np.clip(float(tip_step_mm) / float(tip_ref_mm), 0.0, 1.0)
    motion_rot = np.clip(float(rot_step_deg) / float(rot_ref_deg), 0.0, 1.0)
    motion_score = float(w_tip) * motion_tip + float(w_rot) * motion_rot
    return float(np.clip(motion_score, 0.0, 1.0))


@dataclass
class PoseDepthFilterResult:
    """Result of one camera-Z filter update."""

    rvec: np.ndarray
    tvec: np.ndarray
    T_marker_camera: np.ndarray

    raw_z_mm: float
    filtered_z_mm: float
    delta_z_mm: float

    raw_reprojection_rms_px: float = math.nan
    filtered_reprojection_rms_px: float = math.nan
    reprojection_excess_px: float = math.nan
    guard_alpha: float = 1.0
    applied: bool = False
    innovation_z_mm: float = 0.0
    innovation_mean_z_mm: float = 0.0
    innovation_cusum_pos_mm: float = 0.0
    innovation_cusum_neg_mm: float = 0.0
    innovation_bias_detected: bool = False
    innovation_bias_direction: int = 0
    innovation_bias_limited: bool = False
    object_z_span_mm: float = math.nan
    negative_delta_guard_limited: bool = False


class PoseDepthKalmanFilter:
    """Constant-velocity Kalman filter for the camera-Z translation channel."""

    def __init__(
        self,
        *,
        observation_std_mm: float,
        process_std_mm: float,
        initial_velocity_std_mm: float,
        reprojection_guard_px: float,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        innovation_guard_enabled: bool = True,
        innovation_guard_window: int = 10,
        innovation_guard_bias_threshold_mm: float = 0.75,
        innovation_guard_min_same_sign: int = 8,
        innovation_cusum_slack_mm: float = 0.2,
        innovation_cusum_threshold_mm: float = 8.0,
        negative_delta_guard_enabled: bool = True,
        negative_delta_guard_min_z_span_mm: float = 14.835,
        negative_delta_guard_max_negative_delta_mm: float = 0.0,
        negative_delta_guard_hold_previous_z: bool = False,
        negative_delta_guard_hold_requires_innovation_bias: bool = True,
        negative_delta_guard_hold_min_negative_delta_mm: float = 0.4,
        negative_delta_guard_max_hold_correction_mm: float = 0.75,
        negative_delta_guard_velocity_damping: float = 0.25,
    ) -> None:
        self.observation_std_mm = max(float(observation_std_mm), 1.0e-12)
        self.process_std_mm = max(float(process_std_mm), 1.0e-12)
        self.initial_velocity_std_mm = max(float(initial_velocity_std_mm), 1.0e-12)
        self.reprojection_guard_px = float(reprojection_guard_px)
        self.innovation_guard_enabled = bool(innovation_guard_enabled)
        self.innovation_guard_window = max(1, int(innovation_guard_window))
        self.innovation_guard_bias_threshold_mm = max(
            float(innovation_guard_bias_threshold_mm),
            0.0,
        )
        self.innovation_guard_min_same_sign = max(1, int(innovation_guard_min_same_sign))
        self.innovation_cusum_slack_mm = max(float(innovation_cusum_slack_mm), 0.0)
        self.innovation_cusum_threshold_mm = max(float(innovation_cusum_threshold_mm), 0.0)
        self.negative_delta_guard_enabled = bool(negative_delta_guard_enabled)
        self.negative_delta_guard_min_z_span_mm = max(
            float(negative_delta_guard_min_z_span_mm),
            0.0,
        )
        self.negative_delta_guard_max_negative_delta_mm = max(
            float(negative_delta_guard_max_negative_delta_mm),
            0.0,
        )
        self.negative_delta_guard_hold_previous_z = bool(
            negative_delta_guard_hold_previous_z
        )
        self.negative_delta_guard_hold_requires_innovation_bias = bool(
            negative_delta_guard_hold_requires_innovation_bias
        )
        self.negative_delta_guard_hold_min_negative_delta_mm = max(
            float(negative_delta_guard_hold_min_negative_delta_mm),
            0.0,
        )
        self.negative_delta_guard_max_hold_correction_mm = max(
            float(negative_delta_guard_max_hold_correction_mm),
            0.0,
        )
        self.negative_delta_guard_velocity_damping = float(
            np.clip(float(negative_delta_guard_velocity_damping), 0.0, 1.0)
        )
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._innovation_history: deque[float] = deque(maxlen=self.innovation_guard_window)
        self._innovation_cusum_pos = 0.0
        self._innovation_cusum_neg = 0.0

    def reset(self) -> None:
        """Forget the current depth state."""
        self._x = None
        self._P = None
        self._innovation_history.clear()
        self._innovation_cusum_pos = 0.0
        self._innovation_cusum_neg = 0.0

    def snapshot(self) -> tuple:
        """Return a restorable copy of the filter state."""
        return (
            None if self._x is None else self._x.copy(),
            None if self._P is None else self._P.copy(),
            tuple(self._innovation_history),
            float(self._innovation_cusum_pos),
            float(self._innovation_cusum_neg),
        )

    def restore(self, state: tuple) -> None:
        """Restore a state previously returned by snapshot()."""
        x, P = state[:2]
        self._x = None if x is None else np.asarray(x, dtype=np.float64).reshape(2).copy()
        self._P = None if P is None else np.asarray(P, dtype=np.float64).reshape(2, 2).copy()
        self._innovation_history.clear()
        if len(state) >= 5:
            self._innovation_history.extend(float(v) for v in state[2])
            self._innovation_cusum_pos = float(state[3])
            self._innovation_cusum_neg = float(state[4])
        else:
            self._innovation_cusum_pos = 0.0
            self._innovation_cusum_neg = 0.0

    def update(
        self,
        *,
        rvec: np.ndarray,
        tvec: np.ndarray,
        object_points: Sequence[Sequence[float]] | np.ndarray,
        image_points: Sequence[Sequence[float]] | np.ndarray,
    ) -> PoseDepthFilterResult:
        """Filter one accepted pose and enforce the reprojection guard."""
        from tracking.pose_solvers import make_transform_from_rvec_tvec

        rvec_arr = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        raw_tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        raw_z = float(raw_tvec[2, 0])
        previous_filtered_z = (
            math.nan if self._x is None else float(np.asarray(self._x).reshape(2)[0])
        )
        filtered_z, innovation_diag = self._update_state(raw_z)

        out_tvec = raw_tvec.copy()
        out_tvec[2, 0] = filtered_z

        object_arr = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        image_arr = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        object_z_span = (
            float(np.ptp(object_arr[:, 2])) if len(object_arr) else math.nan
        )
        raw_rms = self._reprojection_rms(object_arr, image_arr, rvec_arr, raw_tvec)
        filtered_rms = self._reprojection_rms(object_arr, image_arr, rvec_arr, out_tvec)
        guard_alpha = 1.0

        if (
            self.reprojection_guard_px > 0.0
            and np.isfinite(raw_rms)
            and np.isfinite(filtered_rms)
            and filtered_rms - raw_rms > self.reprojection_guard_px
        ):
            accepted_z = raw_z
            accepted_rms = raw_rms
            lo = 0.0
            hi = 1.0
            for _ in range(20):
                alpha = 0.5 * (lo + hi)
                candidate_z = raw_z + alpha * (filtered_z - raw_z)
                candidate_tvec = raw_tvec.copy()
                candidate_tvec[2, 0] = candidate_z
                candidate_rms = self._reprojection_rms(
                    object_arr,
                    image_arr,
                    rvec_arr,
                    candidate_tvec,
                )
                if (
                    np.isfinite(candidate_rms)
                    and candidate_rms - raw_rms <= self.reprojection_guard_px
                ):
                    accepted_z = float(candidate_z)
                    accepted_rms = float(candidate_rms)
                    guard_alpha = float(alpha)
                    lo = alpha
                else:
                    hi = alpha

            filtered_z = accepted_z
            filtered_rms = accepted_rms
            out_tvec = raw_tvec.copy()
            out_tvec[2, 0] = filtered_z
            if self._x is not None:
                self._x[0] = filtered_z

        limited_z = self._limit_geometry_gated_negative_delta(
            raw_z=raw_z,
            filtered_z=filtered_z,
            object_z_span_mm=object_z_span,
            previous_filtered_z=previous_filtered_z,
            innovation_bias_detected=bool(innovation_diag["innovation_bias_detected"]),
        )
        negative_delta_limited = limited_z is not None
        if negative_delta_limited:
            filtered_z = float(limited_z)
            out_tvec = raw_tvec.copy()
            out_tvec[2, 0] = filtered_z
            filtered_rms = self._reprojection_rms(object_arr, image_arr, rvec_arr, out_tvec)

        T = make_transform_from_rvec_tvec(rvec_arr, out_tvec)
        return PoseDepthFilterResult(
            rvec=rvec_arr.copy(),
            tvec=out_tvec.copy(),
            T_marker_camera=np.asarray(T, dtype=np.float64).reshape(4, 4),
            raw_z_mm=raw_z,
            filtered_z_mm=float(filtered_z),
            delta_z_mm=float(filtered_z - raw_z),
            raw_reprojection_rms_px=float(raw_rms),
            filtered_reprojection_rms_px=float(filtered_rms),
            reprojection_excess_px=float(filtered_rms - raw_rms),
            guard_alpha=float(guard_alpha),
            applied=bool(abs(filtered_z - raw_z) > 1.0e-12),
            innovation_z_mm=float(innovation_diag["innovation_z_mm"]),
            innovation_mean_z_mm=float(innovation_diag["innovation_mean_z_mm"]),
            innovation_cusum_pos_mm=float(innovation_diag["innovation_cusum_pos_mm"]),
            innovation_cusum_neg_mm=float(innovation_diag["innovation_cusum_neg_mm"]),
            innovation_bias_detected=bool(innovation_diag["innovation_bias_detected"]),
            innovation_bias_direction=int(innovation_diag["innovation_bias_direction"]),
            innovation_bias_limited=bool(innovation_diag["innovation_bias_limited"]),
            object_z_span_mm=float(object_z_span),
            negative_delta_guard_limited=bool(negative_delta_limited),
        )

    def _update_state(self, measurement_z_mm: float) -> tuple[float, dict[str, float | bool | int]]:
        F = np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        H = np.asarray([[1.0, 0.0]], dtype=np.float64)
        G = np.asarray([[0.5], [1.0]], dtype=np.float64)
        q = self.process_std_mm
        r = self.observation_std_mm
        Q = G @ G.T * (q * q)
        R = np.asarray([[r * r]], dtype=np.float64)
        zero_diag: dict[str, float | bool | int] = {
            "innovation_z_mm": 0.0,
            "innovation_mean_z_mm": 0.0,
            "innovation_cusum_pos_mm": 0.0,
            "innovation_cusum_neg_mm": 0.0,
            "innovation_bias_detected": False,
            "innovation_bias_direction": 0,
            "innovation_bias_limited": False,
        }

        if self._x is None or self._P is None:
            self._x = np.asarray([float(measurement_z_mm), 0.0], dtype=np.float64)
            self._P = np.diag([r * r, self.initial_velocity_std_mm**2]).astype(np.float64)
            self._innovation_history.clear()
            self._innovation_cusum_pos = 0.0
            self._innovation_cusum_neg = 0.0
            return float(self._x[0]), zero_diag

        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + Q
        innovation = np.asarray([float(measurement_z_mm)], dtype=np.float64) - H @ x_pred
        innovation_value = float(innovation[0])
        innovation_diag = self._update_innovation_monitor(innovation_value)
        S = H @ P_pred @ H.T + R
        K_gain = P_pred @ H.T @ np.linalg.inv(S)
        self._x = x_pred + (K_gain @ innovation).reshape(2)
        self._P = (np.eye(2, dtype=np.float64) - K_gain @ H) @ P_pred
        return float(self._x[0]), innovation_diag

    def _update_innovation_monitor(self, innovation_z_mm: float) -> dict[str, float | bool | int]:
        if not self.innovation_guard_enabled:
            return {
                "innovation_z_mm": float(innovation_z_mm),
                "innovation_mean_z_mm": 0.0,
                "innovation_cusum_pos_mm": 0.0,
                "innovation_cusum_neg_mm": 0.0,
                "innovation_bias_detected": False,
                "innovation_bias_direction": 0,
                "innovation_bias_limited": False,
            }

        self._innovation_history.append(float(innovation_z_mm))
        mean = float(sum(self._innovation_history) / len(self._innovation_history))
        direction = 1 if mean > 0.0 else -1 if mean < 0.0 else 0
        if direction > 0:
            same_sign = sum(1 for value in self._innovation_history if value >= 0.0)
        elif direction < 0:
            same_sign = sum(1 for value in self._innovation_history if value <= 0.0)
        else:
            same_sign = 0

        slack = self.innovation_cusum_slack_mm
        self._innovation_cusum_pos = max(
            0.0,
            self._innovation_cusum_pos + float(innovation_z_mm) - slack,
        )
        self._innovation_cusum_neg = max(
            0.0,
            self._innovation_cusum_neg - float(innovation_z_mm) - slack,
        )

        enough_history = len(self._innovation_history) >= self.innovation_guard_window
        enough_same_sign = same_sign >= min(
            self.innovation_guard_min_same_sign,
            self.innovation_guard_window,
        )
        running_bias = (
            enough_history
            and enough_same_sign
            and abs(mean) >= self.innovation_guard_bias_threshold_mm
        )
        cusum_bias = False
        if self.innovation_cusum_threshold_mm > 0.0 and enough_history:
            if mean > 0.0:
                cusum_bias = self._innovation_cusum_pos >= self.innovation_cusum_threshold_mm
            elif mean < 0.0:
                cusum_bias = self._innovation_cusum_neg >= self.innovation_cusum_threshold_mm
        bias_detected = bool(running_bias or cusum_bias)
        if running_bias:
            direction = 1 if mean > 0.0 else -1 if mean < 0.0 else 0
        elif self._innovation_cusum_pos > self._innovation_cusum_neg:
            direction = 1
        elif self._innovation_cusum_neg > self._innovation_cusum_pos:
            direction = -1
        else:
            direction = 0

        return {
            "innovation_z_mm": float(innovation_z_mm),
            "innovation_mean_z_mm": mean,
            "innovation_cusum_pos_mm": float(self._innovation_cusum_pos),
            "innovation_cusum_neg_mm": float(self._innovation_cusum_neg),
            "innovation_bias_detected": bias_detected,
            "innovation_bias_direction": int(direction),
            "innovation_bias_limited": False,
        }

    def _limit_geometry_gated_negative_delta(
        self,
        *,
        raw_z: float,
        filtered_z: float,
        object_z_span_mm: float,
        previous_filtered_z: float,
        innovation_bias_detected: bool,
    ) -> float | None:
        if not self.negative_delta_guard_enabled:
            return None
        if not math.isfinite(object_z_span_mm):
            return None
        if object_z_span_mm < self.negative_delta_guard_min_z_span_mm:
            return None

        min_allowed_z = float(raw_z) - self.negative_delta_guard_max_negative_delta_mm
        use_hold = self.negative_delta_guard_hold_previous_z and (
            not self.negative_delta_guard_hold_requires_innovation_bias
            or bool(innovation_bias_detected)
        )
        use_hold = use_hold and (
            float(raw_z) - float(filtered_z)
            >= self.negative_delta_guard_hold_min_negative_delta_mm
        )
        if use_hold and math.isfinite(previous_filtered_z):
            held_z = float(previous_filtered_z)
            held_z = min(
                held_z,
                float(raw_z) + self.negative_delta_guard_max_hold_correction_mm,
            )
            min_allowed_z = max(min_allowed_z, held_z)
        if float(filtered_z) >= min_allowed_z:
            return None

        if self._x is not None:
            self._x[0] = min_allowed_z
            self._x[1] *= self.negative_delta_guard_velocity_damping
            if self._P is not None:
                self._P[0, 1] = 0.0
                self._P[1, 0] = 0.0
        return float(min_allowed_z)

    def _reprojection_rms(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> float:
        if len(object_points) == 0:
            return math.nan
        try:
            import cv2

            projected, _ = cv2.projectPoints(
                np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                self.K,
                self.dist_coeffs,
            )
        except Exception:
            return math.nan
        residual = projected.reshape(-1, 2) - np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        err2 = np.sum(residual * residual, axis=1)
        return float(math.sqrt(float(np.mean(err2)))) if len(err2) else math.nan


class AdaptiveKalmanFilterCV3D:
    def __init__(
        self,
        dt: float,
        *,
        q_pos_still: float = 1e-4,
        q_vel_still: float = 1e-2,
        r_still: float = 8e-2,
        q_pos_move: float = 5e-3,
        q_vel_move: float = 3e-1,
        r_move: float = 2e-2,
        tip_ref_mm: float = 0.8,
        rot_ref_deg: float = 0.4,
        w_tip: float = 0.7,
        w_rot: float = 0.3,
    ) -> None:
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("dt must be > 0.")

        self.q_pos_still = float(q_pos_still)
        self.q_vel_still = float(q_vel_still)
        self.r_still = float(r_still)

        self.q_pos_move = float(q_pos_move)
        self.q_vel_move = float(q_vel_move)
        self.r_move = float(r_move)

        self.tip_ref_mm = float(tip_ref_mm)
        self.rot_ref_deg = float(rot_ref_deg)
        self.w_tip = float(w_tip)
        self.w_rot = float(w_rot)

        self.x = np.zeros((6, 1), dtype=np.float64)

        self.F = np.array(
            [
                [1.0, 0.0, 0.0, self.dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, self.dt, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, self.dt],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        self.Q = np.eye(6, dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1e3
        self.I = np.eye(6, dtype=np.float64)

        self.initialized = False

        self._prev_raw_position_mm: np.ndarray | None = None
        self._prev_raw_rotation: np.ndarray | None = None

        self.last_motion_score: float | None = None
        self.last_tip_step_mm: float | None = None
        self.last_rot_step_deg: float | None = None

    def reset(self) -> None:
        self.x[:] = 0.0
        self.P[:] = np.eye(6, dtype=np.float64) * 1e3
        self.initialized = False

        self._prev_raw_position_mm = None
        self._prev_raw_rotation = None

        self.last_motion_score = None
        self.last_tip_step_mm = None
        self.last_rot_step_deg = None

    def initialize(self, pos_xyz: np.ndarray) -> np.ndarray:
        pos_xyz = np.asarray(pos_xyz, dtype=np.float64).reshape(3)

        self.x[:3, 0] = pos_xyz
        self.x[3:, 0] = 0.0

        self.P = np.eye(6, dtype=np.float64)
        self.P[:3, :3] *= 1.0
        self.P[3:, 3:] *= 10.0

        self.initialized = True
        return self.x[:3, 0].copy()

    def _set_adaptive_noise(self, motion_score: float) -> None:
        m = float(np.clip(motion_score, 0.0, 1.0))

        q_pos = (1.0 - m) * self.q_pos_still + m * self.q_pos_move
        q_vel = (1.0 - m) * self.q_vel_still + m * self.q_vel_move
        r_meas = (1.0 - m) * self.r_still + m * self.r_move

        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float64)
        self.R = np.diag([r_meas, r_meas, r_meas]).astype(np.float64)

    def _compute_internal_motion_score(
        self,
        measurement_mm: np.ndarray,
        rotation_camera: np.ndarray | None,
    ) -> float:
        measurement_mm = np.asarray(measurement_mm, dtype=np.float64).reshape(3)

        if self._prev_raw_position_mm is None:
            tip_step_mm = 0.0
        else:
            tip_step_mm = float(np.linalg.norm(measurement_mm - self._prev_raw_position_mm))

        if self._prev_raw_rotation is None or rotation_camera is None:
            rot_step_deg = 0.0
        else:
            R_curr = np.asarray(rotation_camera, dtype=np.float64).reshape(3, 3)
            R_rel = self._prev_raw_rotation.T @ R_curr
            rot_step_deg = _rotation_angle_deg(R_rel)

        motion_score = _compute_motion_score(
            tip_step_mm=tip_step_mm,
            rot_step_deg=rot_step_deg,
            tip_ref_mm=self.tip_ref_mm,
            rot_ref_deg=self.rot_ref_deg,
            w_tip=self.w_tip,
            w_rot=self.w_rot,
        )

        self.last_tip_step_mm = tip_step_mm
        self.last_rot_step_deg = rot_step_deg
        self.last_motion_score = motion_score

        return motion_score

    def predict(self, motion_score: float) -> np.ndarray | None:
        if not self.initialized:
            return None

        self._set_adaptive_noise(motion_score)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3, 0].copy()

    def update(self, meas_xyz: np.ndarray, motion_score: float) -> np.ndarray:
        z = np.asarray(meas_xyz, dtype=np.float64).reshape(3, 1)

        if not self.initialized:
            pos = self.initialize(z.reshape(3))
            self._set_adaptive_noise(motion_score)
            return pos

        self._set_adaptive_noise(motion_score)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

        return self.x[:3, 0].copy()

    def filter(
        self,
        measurement_mm: np.ndarray,
        rotation_camera: np.ndarray | None = None,
    ) -> np.ndarray:
        measurement_mm = np.asarray(measurement_mm, dtype=np.float64).reshape(3)

        motion_score = self._compute_internal_motion_score(
            measurement_mm=measurement_mm,
            rotation_camera=rotation_camera,
        )

        self.predict(motion_score=motion_score)
        pos_filt = self.update(measurement_mm, motion_score=motion_score)

        self._prev_raw_position_mm = measurement_mm.copy()

        if rotation_camera is None:
            self._prev_raw_rotation = None
        else:
            self._prev_raw_rotation = np.asarray(
                rotation_camera,
                dtype=np.float64,
            ).reshape(3, 3).copy()

        return pos_filt


class PlaneKalmanFilter:
    def __init__(
        self,
        *,
        process_noise: float = 1e-7,
        measurement_noise: float = 1e-4,
        outlier_angle_deg: float = 1.5,
    ) -> None:
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.outlier_angle_deg = float(outlier_angle_deg)

        self._state: np.ndarray | None = None
        self._P = np.eye(4, dtype=np.float64)

        self._Q = np.eye(4, dtype=np.float64) * self.process_noise
        self._R = np.eye(4, dtype=np.float64) * self.measurement_noise

    def reset(self) -> None:
        self._state = None
        self._P = np.eye(4, dtype=np.float64)

    @property
    def is_initialized(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> np.ndarray | None:
        return self._state.copy() if self._state is not None else None

    def update(self, plane: np.ndarray) -> np.ndarray:
        plane = self._normalise(np.asarray(plane, dtype=np.float64))
        plane = self._enforce_sign(plane)

        if self._state is None:
            self._state = plane.copy()
            self._P = np.eye(4, dtype=np.float64)
            return self._state.copy()

        angle = float(np.degrees(np.arccos(
            np.clip(np.dot(plane[:3], self._state[:3]), -1.0, 1.0)
        )))

        if angle > self.outlier_angle_deg:
            print(f"[PlaneKF] Outlier rejected: {angle:.3f}° > {self.outlier_angle_deg}°")
            self._P = self._P + self._Q
            return self._state.copy()

        P_pred = self._P + self._Q

        y = plane - self._state
        S = P_pred + self._R
        K = P_pred @ np.linalg.inv(S)

        self._state = self._state + K @ y
        self._P = (np.eye(4, dtype=np.float64) - K) @ P_pred

        self._state = self._normalise(self._state)

        return self._state.copy()

    @staticmethod
    def _normalise(plane: np.ndarray) -> np.ndarray:
        n_norm = float(np.linalg.norm(plane[:3]))
        if n_norm < 1e-9:
            raise ValueError("Plane normal is near-zero — invalid plane.")
        return plane / n_norm

    def _enforce_sign(self, plane: np.ndarray) -> np.ndarray:
        if self._state is not None:
            if np.dot(plane[:3], self._state[:3]) < 0.0:
                return -plane
        else:
            if plane[2] > 0.0:
                return -plane
        return plane


class CornerKalmanFilterCA2D:
    def __init__(
        self,
        dt: float,
        *,
        process_noise: float = 1.0,
        measurement_noise: float = 0.03,
        initial_position_uncertainty: float = 1e-6,
        initial_velocity_uncertainty: float = 100.0,
        initial_acceleration_uncertainty: float = 1000.0,
    ) -> None:
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0.")

        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)

        self.initial_position_uncertainty = float(initial_position_uncertainty)
        self.initial_velocity_uncertainty = float(initial_velocity_uncertainty)
        self.initial_acceleration_uncertainty = float(initial_acceleration_uncertainty)

        dt2 = self.dt * self.dt

        self.A = np.array(
            [
                [1.0, self.dt, 0.5 * dt2, 0.0, 0.0, 0.0],
                [0.0, 1.0, self.dt,      0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0,          0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0,          1.0, self.dt, 0.5 * dt2],
                [0.0, 0.0, 0.0,          0.0, 1.0, self.dt],
                [0.0, 0.0, 0.0,          0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        self.Q = np.eye(6, dtype=np.float64) * self.process_noise
        self.R = np.eye(2, dtype=np.float64) * self.measurement_noise

        self.I = np.eye(6, dtype=np.float64)
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.P = self._initial_covariance()

        self.initialized = False
        self.missed_frames = 0

    def _initial_covariance(self) -> np.ndarray:
        return np.diag(
            [
                self.initial_position_uncertainty,
                self.initial_velocity_uncertainty,
                self.initial_acceleration_uncertainty,
                self.initial_position_uncertainty,
                self.initial_velocity_uncertainty,
                self.initial_acceleration_uncertainty,
            ]
        ).astype(np.float64)

    def reset(self) -> None:
        self.x[:] = 0.0
        self.P = self._initial_covariance()
        self.initialized = False
        self.missed_frames = 0

    def initialize(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(2)

        self.x[:] = 0.0
        self.x[0, 0] = uv[0]
        self.x[3, 0] = uv[1]

        self.P = self._initial_covariance()
        self.initialized = True
        self.missed_frames = 0

        return self.filtered_uv()

    def predict(self) -> np.ndarray | None:
        if not self.initialized:
            return None

        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        self.missed_frames += 1

        return self.filtered_uv()

    def update(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(2, 1)

        if not self.initialized:
            return self.initialize(uv.reshape(2))

        y = uv - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

        self.missed_frames = 0

        return self.filtered_uv()

    def filter(self, uv: np.ndarray) -> np.ndarray:
        if not self.initialized:
            return self.initialize(uv)

        self.predict()
        return self.update(uv)

    def filtered_uv(self) -> np.ndarray:
        return np.array(
            [self.x[0, 0], self.x[3, 0]],
            dtype=np.float64,
        )


class CornerKalmanBankCA2D:
    def __init__(
        self,
        dt: float,
        *,
        process_noise: float = 1.0,
        measurement_noise: float = 0.03,
        max_missed_frames: int = 2,
    ) -> None:
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0.")

        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.max_missed_frames = int(max_missed_frames)

        self.filters: dict[tuple[int, int], CornerKalmanFilterCA2D] = {}

    def reset(self) -> None:
        self.filters.clear()

    def _get_filter(self, key: tuple[int, int]) -> CornerKalmanFilterCA2D:
        if key not in self.filters:
            self.filters[key] = CornerKalmanFilterCA2D(
                dt=self.dt,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
            )
        return self.filters[key]

    def filter_uv(
        self,
        global_row: int,
        global_col: int,
        uv: np.ndarray,
    ) -> np.ndarray:
        key = (int(global_row), int(global_col))
        kf = self._get_filter(key)
        return kf.filter(uv)

    def filter_corners(self, corners) -> dict[tuple[int, int], np.ndarray]:
        visible_keys: set[tuple[int, int]] = set()
        filtered: dict[tuple[int, int], np.ndarray] = {}

        for p in corners:
            key = (int(p.global_row), int(p.global_col))
            visible_keys.add(key)

            uv_filt = self.filter_uv(
                global_row=p.global_row,
                global_col=p.global_col,
                uv=np.asarray(p.uv, dtype=np.float64),
            )

            filtered[key] = uv_filt

        self._prune_missing(visible_keys)

        return filtered

    def _prune_missing(
        self,
        visible_keys: set[tuple[int, int]],
    ) -> None:
        keys_to_delete: list[tuple[int, int]] = []

        for key, kf in self.filters.items():
            if key in visible_keys:
                continue

            kf.missed_frames += 1

            if kf.missed_frames > self.max_missed_frames:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self.filters[key]
