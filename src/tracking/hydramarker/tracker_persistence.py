from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tracking.hydramarker.identity_store import GlobalCornerIdentity
from tracking.hydramarker.map_pose_tracker import PoseTrackPoint
from tracking.hydramarker.tracker_types import (
    DenseProjectionMatchStats,
    DetectedCorner,
    FastPathDebug,
    PersistentMatchStats,
    PoseSource,
    TrackerCorner,
    TrackerMode,
    TrackerResult,
)
from tracking.pose_solvers import make_transform_from_rvec_tvec


class PersistenceMixin:
    def _merge_with_persistent_correspondences(
        self,
        detection,
        fresh_points: List[PoseTrackPoint],
        fresh_corners: List[TrackerCorner],
    ) -> Tuple[List[PoseTrackPoint], List[TrackerCorner]]:
        if self.config.decode_only_mode:
            return fresh_points, fresh_corners

        if not self.config.enable_temporal_correspondence_persistence:
            return fresh_points, fresh_corners

        if len(fresh_points) < self.config.persistence_min_fresh_points_for_merge:
            return fresh_points, fresh_corners

        persistent_points, persistent_corners = self._persistent_correspondences_for_detection(detection)

        if not persistent_points:
            return fresh_points, fresh_corners

        merged_points: List[PoseTrackPoint] = []
        merged_corners: List[TrackerCorner] = []
        used_globals: set[Tuple[int, int]] = set()

        for point, corner in zip(fresh_points, fresh_corners):
            key = (int(point.global_row), int(point.global_col))
            if key in used_globals:
                continue
            merged_points.append(point)
            merged_corners.append(corner)
            used_globals.add(key)

        for point, corner in zip(persistent_points, persistent_corners):
            key = (int(point.global_row), int(point.global_col))
            if key in used_globals:
                continue
            merged_points.append(point)
            merged_corners.append(corner)
            used_globals.add(key)

        return merged_points, merged_corners

    def _match_predicted_uv_to_detection_corner(
        self,
        predicted_uv: Tuple[float, float],
        current_uvs: np.ndarray,
        used_current_indices: set[int],
        max_dist_px: float,
    ) -> Tuple[Optional[int], float, float, str]:
        diff = current_uvs - np.asarray(predicted_uv, dtype=np.float64)
        dist_sq = (diff * diff).sum(axis=1)
        order = np.argsort(dist_sq)

        best_idx = int(order[0])
        best_dist = float(np.sqrt(dist_sq[best_idx]))
        second_dist = (
            float(np.sqrt(dist_sq[int(order[1])]))
            if len(order) > 1
            else float("inf")
        )

        if best_dist > float(max_dist_px):
            return None, best_dist, second_dist, "far"

        min_margin = float(
            self.config.persistence_match_min_second_best_margin_px
        )
        if (
            min_margin > 0.0
            and np.isfinite(second_dist)
            and (second_dist - best_dist) < min_margin
        ):
            return None, best_dist, second_dist, "ambiguous"

        if best_idx in used_current_indices:
            return None, best_dist, second_dist, "claimed"

        return best_idx, best_dist, second_dist, ""

    def _persistent_correspondences_for_detection(
        self,
        detection,
    ) -> Tuple[List[PoseTrackPoint], List[TrackerCorner]]:
        identities = self._identity_store.all()
        stats = PersistentMatchStats(identities=len(identities))
        self._last_persistent_match_stats = stats

        if not identities:
            return [], []

        if self._persistent_frame_index < 0:
            return [], []

        age = self.frame_index - self._persistent_frame_index
        stats.age = int(age)
        if age < 0 or age > self.config.persistence_max_frames:
            return [], []

        # Build a list of all current detection corner UVs for proximity search.
        # We use UV-proximity matching instead of exact local (i,j) key matching
        # because the CheckerboardDetector can re-index its corners after a
        # tracking reset or lattice drift event, silently changing the local
        # coordinate system while the physical UV positions remain correct.
        # Local-key lookup would then find 0 matches even though 50+ corners
        # are visible -- this was the root cause of the 'frozen' failure mode.
        current_corners = self._detected_corners_from_detection(detection)
        stats.current_corners = len(current_corners)
        if not current_corners:
            return [], []

        current_uvs = np.array(
            [(float(c.uv[0]), float(c.uv[1])) for c in current_corners],
            dtype=np.float64,
        )  # shape (N, 2)

        use_pose_projection = (
            self.config.persistence_use_pose_projection
            and self.pose_tracker.rvec is not None
            and self.pose_tracker.tvec is not None
            and self._last_good_reproj_px >= 0.0
            and self._last_good_reproj_px
            <= self.config.persistence_projection_max_pose_error_px
        )
        stats.used_pose_projection = bool(use_pose_projection)

        points: List[PoseTrackPoint] = []
        corners: List[TrackerCorner] = []
        used_globals: set[Tuple[int, int]] = set()
        used_current_indices: set[int] = set()

        for cached in identities:
            global_key = (int(cached.global_row), int(cached.global_col))
            if global_key in used_globals:
                continue

            if use_pose_projection:
                projected_uv = self._project_point_uv(cached.xyz_mm)
                if projected_uv is None:
                    stats.rejected_no_projection += 1
                    continue
                max_dist = float(self.config.persistence_projection_max_reproj_px)
                predicted_uv = projected_uv
            else:
                max_dist = float(self.config.persistence_uv_match_dist_px)
                predicted_uv = (
                    float(cached.uv[0]),
                    float(cached.uv[1]),
                )

            best_idx, _, _, reject_reason = (
                self._match_predicted_uv_to_detection_corner(
                    predicted_uv=predicted_uv,
                    current_uvs=current_uvs,
                    used_current_indices=used_current_indices,
                    max_dist_px=max_dist,
                )
            )

            if reject_reason == "far":
                stats.rejected_far += 1
                continue
            if reject_reason == "ambiguous":
                stats.rejected_ambiguous += 1
                continue
            if reject_reason == "claimed":
                stats.rejected_claimed += 1
                continue
            if best_idx is None:
                continue

            matched = current_corners[best_idx]
            uv = (float(matched.uv[0]), float(matched.uv[1]))
            xyz = self._point3(cached.xyz_mm)
            votes = max(0, int(cached.votes) - age)

            points.append(
                PoseTrackPoint(
                    global_row=global_key[0],
                    global_col=global_key[1],
                    xyz_mm=xyz,
                    uv=uv,
                    votes=votes,
                )
            )

            corners.append(
                TrackerCorner(
                    local_row=int(matched.local_row),
                    local_col=int(matched.local_col),
                    global_row=global_key[0],
                    global_col=global_key[1],
                    xyz_mm=xyz,
                    uv=uv,
                    votes=votes,
                )
            )

            used_globals.add(global_key)
            used_current_indices.add(best_idx)
            stats.accepted += 1

        return points, corners

    def _project_point_uv(
        self,
        xyz_mm,
    ) -> Optional[Tuple[float, float]]:
        if self.pose_tracker.rvec is None or self.pose_tracker.tvec is None:
            return None

        return self._project_point_uv_with_pose(
            xyz_mm,
            self.pose_tracker.rvec,
            self.pose_tracker.tvec,
        )

    def _project_point_uv_with_pose(
        self,
        xyz_mm,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        obj = np.asarray(
            [self._point3(xyz_mm)],
            dtype=np.float64,
        ).reshape(1, 3)

        try:
            projected, _ = cv2.projectPoints(
                obj,
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                self.K,
                self.dist_coeffs,
            )
        except Exception:
            return None

        uv = projected.reshape(-1, 2)[0]
        return float(uv[0]), float(uv[1])

    def _current_uv_by_local_corner(self, detection) -> Dict[Tuple[int, int], Tuple[float, float]]:
        uv_by_local: Dict[Tuple[int, int], Tuple[float, float]] = {}

        for corner in self._detected_corners_from_detection(detection):
            uv_by_local[(int(corner.local_row), int(corner.local_col))] = (
                float(corner.uv[0]),
                float(corner.uv[1]),
            )

        return uv_by_local

    def _store_persistent_correspondences(self, corners: List[TrackerCorner]) -> None:
        if self.config.decode_only_mode:
            return

        if not self.config.enable_temporal_correspondence_persistence:
            return

        identities: List[GlobalCornerIdentity] = []
        used_global: set[Tuple[int, int]] = set()

        for corner in corners:
            global_key = (int(corner.global_row), int(corner.global_col))

            if global_key in used_global:
                continue

            identities.append(
                GlobalCornerIdentity(
                    global_row=global_key[0],
                    global_col=global_key[1],
                    xyz_mm=self._point3(corner.xyz_mm),
                    uv=self._point2(corner.uv),
                    votes=int(corner.votes),
                )
            )

            used_global.add(global_key)

        if len(identities) >= self.config.persistence_min_points:
            self._identity_store.replace(identities)
            self._persistent_frame_index = self.frame_index

    def _clear_persistent_correspondences(self) -> None:
        self._identity_store.clear()
        self._persistent_frame_index = -1
    @property
    def _persistent_corners(self) -> List[TrackerCorner]:
        """
        Compatibility view for existing logs/debug scripts.

        The semantic persistence store is IdentityStore. Local indices here are
        intentionally unset because they are frame-local, not persistent IDs.
        """
        corners: List[TrackerCorner] = []
        for identity in self._identity_store.all():
            corners.append(
                TrackerCorner(
                    local_row=-1,
                    local_col=-1,
                    global_row=int(identity.global_row),
                    global_col=int(identity.global_col),
                    xyz_mm=self._point3(identity.xyz_mm),
                    uv=self._point2(identity.uv),
                    votes=int(identity.votes),
                )
            )
        return corners

