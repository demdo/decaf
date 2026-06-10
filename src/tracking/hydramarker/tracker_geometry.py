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
    GeometryCornerCache,
    PersistentMatchStats,
    PoseSource,
    TrackerCorner,
    TrackerMode,
    TrackerResult,
)
from tracking.pose_solvers import make_transform_from_rvec_tvec


class GeometryMixin:
    def _inlier_corners_from_pose(self, pose, tracker_corners: List[TrackerCorner]) -> List[TrackerCorner]:
        inlier_corners: List[TrackerCorner] = []

        if pose.inlier_indices is None:
            return inlier_corners

        for idx in pose.inlier_indices.reshape(-1):
            i = int(idx)
            if 0 <= i < len(tracker_corners):
                inlier_corners.append(tracker_corners[i])

        return inlier_corners

    def _points_from_correspondences(self, correspondences) -> Tuple[List[PoseTrackPoint], List[TrackerCorner]]:
        points: List[PoseTrackPoint] = []
        corners: List[TrackerCorner] = []
        used_globals: set[Tuple[int, int]] = set()

        for c in correspondences:
            global_key = (int(c.global_row), int(c.global_col))
            if global_key in used_globals:
                continue

            xyz = self._point3(c.xyz_mm)
            uv = self._point2(c.uv)
            votes = int(getattr(c, "votes", 0))

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
                    local_row=int(c.local_row),
                    local_col=int(c.local_col),
                    global_row=global_key[0],
                    global_col=global_key[1],
                    xyz_mm=xyz,
                    uv=uv,
                    votes=votes,
                )
            )

            used_globals.add(global_key)

        return points, corners

    def _attach_detection_info(self, result: TrackerResult, detection) -> None:
        result.detection_valid = False if detection is None else bool(detection.valid())
        result.detection_tracking = False if detection is None else bool(detection.tracking)
        result.detection_stable = False if detection is None else bool(detection.stable)
        result.detection_corners = self._detected_corners_from_detection(detection)

    def _detected_corners_from_detection(self, detection) -> List[DetectedCorner]:
        if detection is None:
            return []

        detection_corners = getattr(detection, "corners", None)
        if detection_corners is None:
            return []

        corners: List[DetectedCorner] = []

        for corner in detection_corners:
            parsed = self._local_key_and_uv_from_detection_corner(corner)
            if parsed is None:
                continue

            (local_row, local_col), uv = parsed
            corners.append(
                DetectedCorner(
                    local_row=int(local_row),
                    local_col=int(local_col),
                    uv=(float(uv[0]), float(uv[1])),
                )
            )

        return corners

    def _local_key_and_uv_from_detection_corner(
        self,
        corner,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[float, float]]]:
        local_row = self._first_existing_attr(
            corner,
            ("local_row", "row", "r", "j"),
        )
        local_col = self._first_existing_attr(
            corner,
            ("local_col", "col", "c", "i"),
        )

        if local_row is None or local_col is None:
            return None

        uv_source = self._first_existing_attr(
            corner,
            ("uv", "pt", "point", "xy"),
        )

        if uv_source is None:
            if hasattr(corner, "x") and hasattr(corner, "y"):
                uv = (float(corner.x), float(corner.y))
            else:
                return None
        else:
            uv = self._point2(uv_source)

        return (int(local_row), int(local_col)), uv

    def _confidence(self, num_inliers: int, mean_error_px: float) -> float:
        point_score = min(1.0, float(num_inliers) / 30.0)

        if mean_error_px < 0.0:
            error_score = 0.0
        else:
            error_score = 1.0 - min(
                1.0,
                mean_error_px / max(1e-6, self.config.max_mean_reprojection_error_px),
            )

        return float(0.6 * point_score + 0.4 * error_score)

    @staticmethod
    def _first_existing_attr(obj, names: Tuple[str, ...]):
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return None

    @staticmethod
    def _point2(p) -> Tuple[float, float]:
        if hasattr(p, "x") and hasattr(p, "y"):
            return float(p.x), float(p.y)

        arr = np.asarray(p, dtype=np.float64).reshape(-1)
        return float(arr[0]), float(arr[1])

    @staticmethod
    def _point3(p) -> Tuple[float, float, float]:
        if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
            return float(p.x), float(p.y), float(p.z)

        arr = np.asarray(p, dtype=np.float64).reshape(-1)
        return float(arr[0]), float(arr[1]), float(arr[2])
    def _build_geometry_corner_cache(self) -> GeometryCornerCache:
        rows = []
        cols = []
        xyz = []

        for gr in range(int(self.geometry.corner_rows())):
            for gc in range(int(self.geometry.corner_cols())):
                if not self.geometry.has_corner(gr, gc):
                    continue

                pt = self.geometry.corner_point(gr, gc)
                rows.append(int(gr))
                cols.append(int(gc))
                xyz.append([float(pt.x), float(pt.y), float(pt.z)])

        if not xyz:
            return GeometryCornerCache()

        return GeometryCornerCache(
            rows=np.asarray(rows, dtype=np.int32),
            cols=np.asarray(cols, dtype=np.int32),
            xyz_mm=np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        )

