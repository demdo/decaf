from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class TrackerConsistencyConfig:
    max_translation_jump_mm: float = 120.0
    max_rotation_jump_deg: float = 45.0


@dataclass
class ConsistencyResult:
    accepted: bool
    reason: str = ""

    translation_jump_mm: float = 0.0
    rotation_jump_deg: float = 0.0


class TrackerConsistency:
    """
    Temporal / semantic consistency layer for HydraMarker tracking.

    First version:
        - only pose jump gating

    Later extensions:
        - global identity overlap
        - reprojection drift
        - multi-frame decode confirmation
        - prediction / Kalman support
    """

    def __init__(self, config: TrackerConsistencyConfig) -> None:
        self.config = config

    def validate_pose_jump(
        self,
        previous_rvec: Optional[np.ndarray],
        previous_tvec: Optional[np.ndarray],
        candidate_rvec: Optional[np.ndarray],
        candidate_tvec: Optional[np.ndarray],
    ) -> ConsistencyResult:
        if previous_rvec is None or previous_tvec is None:
            return ConsistencyResult(
                accepted=True,
                reason="No previous pose available.",
            )

        if candidate_rvec is None or candidate_tvec is None:
            return ConsistencyResult(
                accepted=False,
                reason="Candidate pose is incomplete.",
            )

        previous_t = np.asarray(previous_tvec, dtype=np.float64).reshape(3, 1)
        candidate_t = np.asarray(candidate_tvec, dtype=np.float64).reshape(3, 1)

        translation_jump_mm = float(np.linalg.norm(candidate_t - previous_t))

        R_prev, _ = cv2.Rodrigues(
            np.asarray(previous_rvec, dtype=np.float64).reshape(3, 1)
        )
        R_candidate, _ = cv2.Rodrigues(
            np.asarray(candidate_rvec, dtype=np.float64).reshape(3, 1)
        )

        R_delta = R_candidate @ R_prev.T

        cos_angle = np.clip(
            (np.trace(R_delta) - 1.0) * 0.5,
            -1.0,
            1.0,
        )

        rotation_jump_deg = float(np.degrees(np.arccos(cos_angle)))

        if translation_jump_mm > self.config.max_translation_jump_mm:
            return ConsistencyResult(
                accepted=False,
                reason=(
                    f"Translation jump too large: "
                    f"{translation_jump_mm:.2f} mm > "
                    f"{self.config.max_translation_jump_mm:.2f} mm."
                ),
                translation_jump_mm=translation_jump_mm,
                rotation_jump_deg=rotation_jump_deg,
            )

        if rotation_jump_deg > self.config.max_rotation_jump_deg:
            return ConsistencyResult(
                accepted=False,
                reason=(
                    f"Rotation jump too large: "
                    f"{rotation_jump_deg:.2f} deg > "
                    f"{self.config.max_rotation_jump_deg:.2f} deg."
                ),
                translation_jump_mm=translation_jump_mm,
                rotation_jump_deg=rotation_jump_deg,
            )

        return ConsistencyResult(
            accepted=True,
            reason="Pose jump plausible.",
            translation_jump_mm=translation_jump_mm,
            rotation_jump_deg=rotation_jump_deg,
        )