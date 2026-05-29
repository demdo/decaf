from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


GridKey = Tuple[int, int]


@dataclass
class TrackerConsistencyConfig:
    max_translation_jump_mm: float = 120.0
    max_rotation_jump_deg: float = 45.0

    min_identity_overlap_count: int = 6
    min_identity_overlap_ratio: float = 0.70
    max_identity_conflict_ratio: float = 0.30

    min_rotation_vote_count: int = 3
    min_rotation_vote_ratio: float = 0.70


@dataclass
class ConsistencyResult:
    accepted: bool
    reason: str = ""

    translation_jump_mm: float = 0.0
    rotation_jump_deg: float = 0.0


@dataclass
class IdentityConsistencyResult:
    accepted: bool
    reason: str = ""

    checked_local: int = 0
    consistent_local: int = 0
    checked_global: int = 0
    consistent_global: int = 0

    local_ratio: float = 1.0
    global_ratio: float = 1.0
    conflict_count: int = 0
    conflict_ratio: float = 0.0

    conflicts: list[str] = field(default_factory=list)


@dataclass
class RotationConsistencyResult:
    accepted: bool
    reason: str = ""

    dominant_rotation_deg: Optional[int] = None
    dominant_count: int = 0
    total_count: int = 0
    dominant_ratio: float = 0.0


@dataclass
class CombinedConsistencyResult:
    accepted: bool
    reason: str = ""

    pose: ConsistencyResult = field(
        default_factory=lambda: ConsistencyResult(True, "Pose check not evaluated.")
    )
    identity: IdentityConsistencyResult = field(
        default_factory=lambda: IdentityConsistencyResult(True, "Identity check not evaluated.")
    )
    rotation: RotationConsistencyResult = field(
        default_factory=lambda: RotationConsistencyResult(True, "Rotation check not evaluated.")
    )


class TrackerConsistency:
    """
    Temporal / semantic consistency layer for HydraMarker tracking.

    This class intentionally keeps the tracker.py orchestration clean.
    It validates:
        - pose jumps,
        - local/global identity consistency,
        - decoded patch rotation consistency,
        - combined tracking decisions.
    """

    def __init__(self, config: TrackerConsistencyConfig) -> None:
        self.config = config
        self.last_dominant_rotation_deg: Optional[int] = None

    def reset(self) -> None:
        self.last_dominant_rotation_deg = None

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

    def validate_identity_consistency(
        self,
        previous_identities,
        candidate_identities,
        *,
        check_global_to_local: bool = True,
    ) -> IdentityConsistencyResult:
        previous = list(previous_identities or [])
        candidate = list(candidate_identities or [])

        if not previous:
            return IdentityConsistencyResult(
                accepted=True,
                reason="No previous identities available.",
            )

        if not candidate:
            return IdentityConsistencyResult(
                accepted=False,
                reason="No candidate identities available.",
            )

        previous_by_local = {
            self._local_key(p): self._global_key(p)
            for p in previous
        }

        previous_by_global = {
            self._global_key(p): self._local_key(p)
            for p in previous
        }

        checked_local = 0
        consistent_local = 0

        checked_global = 0
        consistent_global = 0

        conflicts: list[str] = []

        for p in candidate:
            local_key = self._local_key(p)
            global_key = self._global_key(p)

            old_global = previous_by_local.get(local_key)
            if old_global is not None:
                checked_local += 1
                if old_global == global_key:
                    consistent_local += 1
                elif len(conflicts) < 8:
                    conflicts.append(
                        f"local {local_key}: old global {old_global}, "
                        f"new global {global_key}"
                    )

            if check_global_to_local:
                old_local = previous_by_global.get(global_key)
                if old_local is not None:
                    checked_global += 1
                    if old_local == local_key:
                        consistent_global += 1
                    elif len(conflicts) < 8:
                        conflicts.append(
                            f"global {global_key}: old local {old_local}, "
                            f"new local {local_key}"
                        )

        local_ratio = (
            float(consistent_local) / float(checked_local)
            if checked_local > 0
            else 1.0
        )

        global_ratio = (
            float(consistent_global) / float(checked_global)
            if checked_global > 0
            else 1.0
        )

        total_checked = checked_local + checked_global
        total_consistent = consistent_local + consistent_global
        conflict_count = total_checked - total_consistent

        conflict_ratio = (
            float(conflict_count) / float(total_checked)
            if total_checked > 0
            else 0.0
        )

        enough_local = checked_local >= self.config.min_identity_overlap_count
        enough_global = checked_global >= self.config.min_identity_overlap_count

        local_ok = (
            not enough_local
            or local_ratio >= self.config.min_identity_overlap_ratio
        )

        global_ok = (
            not enough_global
            or global_ratio >= self.config.min_identity_overlap_ratio
        )

        conflict_ok = conflict_ratio <= self.config.max_identity_conflict_ratio

        accepted = bool(local_ok and global_ok and conflict_ok)

        reason = (
            f"local {consistent_local}/{checked_local} "
            f"({local_ratio:.2f}), "
            f"global {consistent_global}/{checked_global} "
            f"({global_ratio:.2f}), "
            f"conflicts={conflict_count}/{total_checked} "
            f"({conflict_ratio:.2f})"
        )

        if conflicts:
            reason += "; " + " | ".join(conflicts)

        return IdentityConsistencyResult(
            accepted=accepted,
            reason=reason,
            checked_local=checked_local,
            consistent_local=consistent_local,
            checked_global=checked_global,
            consistent_global=consistent_global,
            local_ratio=local_ratio,
            global_ratio=global_ratio,
            conflict_count=conflict_count,
            conflict_ratio=conflict_ratio,
            conflicts=conflicts,
        )

    def validate_rotation_consistency(
        self,
        rotations_deg: Sequence[int],
        *,
        update_state_on_accept: bool = False,
    ) -> RotationConsistencyResult:
        rotations = [
            self._normalize_rotation_deg(r)
            for r in rotations_deg
            if r is not None
        ]

        if not rotations:
            return RotationConsistencyResult(
                accepted=True,
                reason="No patch rotations available.",
            )

        counts: dict[int, int] = {}
        for r in rotations:
            counts[r] = counts.get(r, 0) + 1

        dominant_rotation = max(counts, key=counts.get)
        dominant_count = int(counts[dominant_rotation])
        total_count = int(len(rotations))
        dominant_ratio = float(dominant_count) / float(total_count)

        enough_votes = total_count >= self.config.min_rotation_vote_count
        enough_ratio = dominant_ratio >= self.config.min_rotation_vote_ratio

        same_as_previous = (
            self.last_dominant_rotation_deg is None
            or dominant_rotation == self.last_dominant_rotation_deg
        )

        accepted = bool((not enough_votes or enough_ratio) and same_as_previous)

        if not same_as_previous:
            reason = (
                f"Dominant patch rotation changed: "
                f"{self.last_dominant_rotation_deg} -> {dominant_rotation} deg."
            )
        elif enough_votes and not enough_ratio:
            reason = (
                f"No stable dominant patch rotation: "
                f"{dominant_count}/{total_count} "
                f"({dominant_ratio:.2f})."
            )
        else:
            reason = (
                f"Dominant patch rotation stable: "
                f"{dominant_rotation} deg, "
                f"{dominant_count}/{total_count} "
                f"({dominant_ratio:.2f})."
            )

        if accepted and update_state_on_accept:
            self.last_dominant_rotation_deg = dominant_rotation

        return RotationConsistencyResult(
            accepted=accepted,
            reason=reason,
            dominant_rotation_deg=dominant_rotation,
            dominant_count=dominant_count,
            total_count=total_count,
            dominant_ratio=dominant_ratio,
        )

    def validate_combined(
        self,
        *,
        previous_rvec: Optional[np.ndarray],
        previous_tvec: Optional[np.ndarray],
        candidate_rvec: Optional[np.ndarray],
        candidate_tvec: Optional[np.ndarray],
        previous_identities=None,
        candidate_identities=None,
        decoded_rotations_deg: Optional[Sequence[int]] = None,
        check_global_to_local: bool = True,
        update_rotation_state_on_accept: bool = False,
    ) -> CombinedConsistencyResult:
        pose_result = self.validate_pose_jump(
            previous_rvec=previous_rvec,
            previous_tvec=previous_tvec,
            candidate_rvec=candidate_rvec,
            candidate_tvec=candidate_tvec,
        )

        identity_result = self.validate_identity_consistency(
            previous_identities=previous_identities,
            candidate_identities=candidate_identities,
            check_global_to_local=check_global_to_local,
        )

        rotation_result = self.validate_rotation_consistency(
            decoded_rotations_deg or [],
            update_state_on_accept=False,
        )

        accepted = bool(
            identity_result.accepted
            and rotation_result.accepted
            and (
                pose_result.accepted
                or identity_result.local_ratio >= 0.95
                or identity_result.global_ratio >= 0.95
            )
        )

        reasons = [
            f"pose: {pose_result.reason}",
            f"identity: {identity_result.reason}",
            f"rotation: {rotation_result.reason}",
        ]

        if accepted and update_rotation_state_on_accept:
            if rotation_result.dominant_rotation_deg is not None:
                self.last_dominant_rotation_deg = rotation_result.dominant_rotation_deg

        return CombinedConsistencyResult(
            accepted=accepted,
            reason=" | ".join(reasons),
            pose=pose_result,
            identity=identity_result,
            rotation=rotation_result,
        )

    @staticmethod
    def _local_key(p) -> GridKey:
        return int(p.local_row), int(p.local_col)

    @staticmethod
    def _global_key(p) -> GridKey:
        return int(p.global_row), int(p.global_col)

    @staticmethod
    def _normalize_rotation_deg(rotation_deg: int) -> int:
        r = int(round(float(rotation_deg))) % 360

        if r in (0, 90, 180, 270):
            return r

        allowed = np.array([0, 90, 180, 270], dtype=np.float64)
        idx = int(np.argmin(np.abs(allowed - float(r))))
        return int(allowed[idx])