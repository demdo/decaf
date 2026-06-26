"""Geometry, projection, and dense spatial refinement helpers."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tracking.hydramarker.backend import cpp_impl as hm
from tracking.hydramarker.tracker_pose import MapPoseResult, PoseTrackPoint
from tracking.hydramarker.tracker_types import (
    DenseProjectionMatchStats,
    DetectedCorner,
    GeometryCornerCache,
    PoseSource,
    TrackerCorner,
    TrackerMode,
    TrackerResult,
)
from tracking.pose_solvers import make_transform_from_rvec_tvec


class GeometryMixin:
    """Convert backend observations into typed tracker geometry structures."""

    def _inlier_corners_from_pose(self, pose, tracker_corners: List[TrackerCorner]) -> List[TrackerCorner]:
        """Select tracker corners that correspond to accepted PnP inliers."""
        inlier_corners: List[TrackerCorner] = []

        if pose.inlier_indices is None:
            return inlier_corners

        for idx in pose.inlier_indices.reshape(-1):
            i = int(idx)
            if 0 <= i < len(tracker_corners):
                inlier_corners.append(tracker_corners[i])

        return inlier_corners

    def _points_from_correspondences(self, correspondences) -> Tuple[List[PoseTrackPoint], List[TrackerCorner]]:
        """Convert backend correspondences into unique pose points and public corners."""
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
        """Copy current checkerboard-detection metadata into a tracker result."""
        result.detection_valid = False if detection is None else bool(detection.valid())
        result.detection_tracking = False if detection is None else bool(detection.tracking)
        result.detection_stable = False if detection is None else bool(detection.stable)
        result.detection_corners = self._detected_corners_from_detection(detection)

    def _detected_corners_from_detection(self, detection) -> List[DetectedCorner]:
        """Normalize backend detection corners into frame-local detected corners."""
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
        """Extract a local grid key and UV coordinate from backend corner variants."""
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
        """Compute a compact confidence score from inlier count and reprojection error."""
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
        """Precompute all valid marker geometry corners for vectorized projection."""
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

    def _cpp_tracker_geometry_enabled(self) -> bool:
        return bool(
            getattr(self.config, "cpp_dense_projection_matcher_enabled", True)
            or getattr(self.config, "cpp_visual_corner_filter_enabled", True)
            or getattr(self.config, "cpp_dense_direct_solver_enabled", True)
            or getattr(self.config, "cpp_dense_robust_solver_enabled", True)
        )

    def _ensure_cpp_tracker_geometry(self):
        """Create the C++ geometry helper on demand."""
        if not self._cpp_tracker_geometry_enabled():
            return None

        if bool(getattr(self, "_cpp_tracker_geometry_unavailable", False)):
            return None

        helper = getattr(self, "_cpp_tracker_geometry", None)
        if helper is not None:
            return helper

        try:
            helper = hm.create_tracker_geometry(
                self.geometry,
                self.K,
                self.dist_coeffs,
                self.config,
            )
        except Exception:
            self._cpp_tracker_geometry_unavailable = True
            self._cpp_tracker_geometry = None
            return None

        self._cpp_tracker_geometry = helper
        return helper

    @staticmethod
    def _dense_projection_match_stats_from_cpp(stats) -> DenseProjectionMatchStats:
        return DenseProjectionMatchStats(
            detected=int(getattr(stats, "detected", 0)),
            projected=int(getattr(stats, "projected", 0)),
            rejected_no_projection=int(
                getattr(stats, "rejected_no_projection", 0)
            ),
            rejected_far=int(getattr(stats, "rejected_far", 0)),
            rejected_ambiguous=int(
                getattr(stats, "rejected_ambiguous", 0)
            ),
            rejected_non_mutual=int(
                getattr(stats, "rejected_non_mutual", 0)
            ),
            median_error_px=float(
                getattr(stats, "median_error_px", float("inf"))
            ),
            p90_error_px=float(
                getattr(stats, "p90_error_px", float("inf"))
            ),
            image_coverage=float(
                getattr(stats, "image_coverage", -1.0)
            ),
            image_span_u_px=float(
                getattr(stats, "image_span_u_px", -1.0)
            ),
            image_span_v_px=float(
                getattr(stats, "image_span_v_px", -1.0)
            ),
            object_span_mm=float(
                getattr(stats, "object_span_mm", -1.0)
            ),
            distinct_rows=int(getattr(stats, "distinct_rows", 0)),
            distinct_cols=int(getattr(stats, "distinct_cols", 0)),
        )

    @staticmethod
    def _geometry_tracker_corners_from_cpp(corners) -> List[TrackerCorner]:
        return [
            TrackerCorner(
                local_row=int(corner.local_row),
                local_col=int(corner.local_col),
                global_row=int(corner.global_row),
                global_col=int(corner.global_col),
                xyz_mm=tuple(float(v) for v in corner.xyz_mm),
                uv=tuple(float(v) for v in corner.uv),
                votes=int(getattr(corner, "votes", 0)),
            )
            for corner in corners
        ]

    @staticmethod
    def _geometry_tracker_corners_to_cpp(corners: List[TrackerCorner]):
        converted = []
        for corner in corners:
            out = hm.TrackerCorner()
            out.local_row = int(corner.local_row)
            out.local_col = int(corner.local_col)
            out.global_row = int(corner.global_row)
            out.global_col = int(corner.global_col)
            out.xyz_mm = tuple(float(v) for v in corner.xyz_mm)
            out.uv = tuple(float(v) for v in corner.uv)
            out.votes = int(getattr(corner, "votes", 0))
            converted.append(out)
        return converted


class ProjectionMixin:
    """Project marker geometry and match projected corners against detections."""

    def _projected_tracker_corners_from_current_pose(self) -> List[TrackerCorner]:
        """Return all visible geometry corners projected by the current pose prior."""
        rows = self.geometry.corner_rows()
        cols = self.geometry.corner_cols()
        corners: List[TrackerCorner] = []

        for gr in range(rows):
            for gc in range(cols):
                if not self.geometry.has_corner(gr, gc):
                    continue

                pt = self.geometry.corner_point(gr, gc)
                xyz = (float(pt.x), float(pt.y), float(pt.z))
                uv = self._project_point_uv(xyz)
                if uv is None:
                    continue

                corners.append(
                    TrackerCorner(
                        local_row=int(gr),
                        local_col=int(gc),
                        global_row=int(gr),
                        global_col=int(gc),
                        xyz_mm=xyz,
                        uv=uv,
                        votes=0,
                    )
                )

        return corners

    def _fallback_pose_rejection_reason(
        self,
        detection,
        rvec: Optional[np.ndarray],
        tvec: Optional[np.ndarray],
        mean_reproj_px: float,
        max_reproj_px: float,
    ) -> str:
        """Return a rejection reason when a fallback pose fails visual or error gates."""
        if mean_reproj_px > self.config.fallback_pose_max_mean_reprojection_error_px:
            return (
                "Fallback pose rejected by mean reprojection gate "
                f"({mean_reproj_px:.2f}px)."
            )

        if max_reproj_px > self.config.fallback_pose_max_max_reprojection_error_px:
            return (
                "Fallback pose rejected by max reprojection gate "
                f"({max_reproj_px:.2f}px)."
            )

        _, match_count, median_err, p90_err = (
            self._projected_tracker_corners_for_detection_pose(
                detection,
                rvec,
                tvec,
                max_dist_px=self.config.fallback_pose_max_p90_corner_error_px,
            )
        )

        if match_count < self.config.fallback_pose_min_detection_matches:
            return (
                "Fallback pose rejected by blue-corner alignment "
                f"({match_count} matches)."
            )

        if median_err > self.config.fallback_pose_max_median_corner_error_px:
            return (
                "Fallback pose rejected by median blue-corner error "
                f"({median_err:.2f}px)."
            )

        if p90_err > self.config.fallback_pose_max_p90_corner_error_px:
            return (
                "Fallback pose rejected by p90 blue-corner error "
                f"({p90_err:.2f}px)."
            )

        return ""

    def _visual_corners_from_pose(
        self,
        corners: List[TrackerCorner],
        rvec: Optional[np.ndarray],
        tvec: Optional[np.ndarray],
    ) -> List[TrackerCorner]:
        """Keep only corners whose measured UV agrees with the supplied pose."""
        if rvec is None or tvec is None:
            return []

        if bool(getattr(self.config, "cpp_visual_corner_filter_enabled", True)):
            helper = self._ensure_cpp_tracker_geometry()
            if helper is not None:
                try:
                    return self._geometry_tracker_corners_from_cpp(
                        helper.visual_corners_from_pose(
                            self._geometry_tracker_corners_to_cpp(corners),
                            rvec,
                            tvec,
                            float(self.config.visual_corner_max_reprojection_error_px),
                        )
                    )
                except Exception:
                    self._cpp_tracker_geometry_unavailable = True

        max_err = float(self.config.visual_corner_max_reprojection_error_px)
        accepted: List[TrackerCorner] = []

        for corner in corners:
            projected_uv = self._project_point_uv_with_pose(
                corner.xyz_mm,
                rvec,
                tvec,
            )
            if projected_uv is None:
                continue

            du = float(projected_uv[0]) - float(corner.uv[0])
            dv = float(projected_uv[1]) - float(corner.uv[1])
            if float(np.hypot(du, dv)) > max_err:
                continue

            accepted.append(corner)

        return accepted

    def _strict_projected_tracker_corners_for_detection_pose(
        self,
        detection,
        rvec: Optional[np.ndarray],
        tvec: Optional[np.ndarray],
        max_dist_px: float,
        ambiguity_margin_px: float,
    ) -> Tuple[List[TrackerCorner], DenseProjectionMatchStats]:
        """Match projected global corners to detected corners with mutual-nearest gates."""
        stats = DenseProjectionMatchStats()

        if detection is None or rvec is None or tvec is None:
            return [], stats

        if bool(getattr(self.config, "cpp_dense_projection_matcher_enabled", True)):
            helper = self._ensure_cpp_tracker_geometry()
            if helper is not None:
                try:
                    result = helper.strict_projected_match(
                        detection,
                        rvec,
                        tvec,
                        float(max_dist_px),
                        float(ambiguity_margin_px),
                    )
                    return (
                        self._geometry_tracker_corners_from_cpp(result.corners),
                        self._dense_projection_match_stats_from_cpp(result.stats),
                    )
                except Exception:
                    self._cpp_tracker_geometry_unavailable = True

        detected = self._detected_corners_from_detection(detection)
        stats.detected = len(detected)
        if not detected:
            return [], stats

        cache = self._geometry_corner_cache
        if len(cache.xyz_mm) == 0:
            return [], stats

        object_points = np.asarray(cache.xyz_mm, dtype=np.float64).reshape(-1, 3)
        try:
            rvec_arr = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec_arr = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            projected, _ = cv2.projectPoints(
                object_points,
                rvec_arr,
                tvec_arr,
                self.K,
                self.dist_coeffs,
            )
            R, _ = cv2.Rodrigues(rvec_arr)
        except Exception:
            stats.rejected_no_projection = int(len(object_points))
            return [], stats

        projected_uvs_all = projected.reshape(-1, 2)
        camera_xyz = object_points @ R.T + tvec_arr.reshape(1, 3)
        valid_projection = (
            np.isfinite(projected_uvs_all).all(axis=1)
            & np.isfinite(camera_xyz).all(axis=1)
            & (camera_xyz[:, 2] > 1e-6)
        )
        stats.rejected_no_projection = int(len(object_points) - np.count_nonzero(valid_projection))
        if not np.any(valid_projection):
            return [], stats

        projected_uvs = projected_uvs_all[valid_projection]
        projected_rows = np.asarray(cache.rows, dtype=np.int32)[valid_projection]
        projected_cols = np.asarray(cache.cols, dtype=np.int32)[valid_projection]
        projected_xyz = object_points[valid_projection]
        stats.projected = int(len(projected_uvs))

        detected_uvs = np.asarray(
            [(float(c.uv[0]), float(c.uv[1])) for c in detected],
            dtype=np.float64,
        ).reshape(-1, 2)

        deltas = projected_uvs[:, None, :] - detected_uvs[None, :, :]
        distances_px = np.linalg.norm(deltas, axis=2)
        best_detected_for_projected = np.argmin(distances_px, axis=1)
        best_distances = distances_px[
            np.arange(len(projected_uvs)),
            best_detected_for_projected,
        ]

        if len(detected_uvs) > 1:
            second_distances = np.partition(distances_px, 1, axis=1)[:, 1]
        else:
            second_distances = np.full(len(projected_uvs), float("inf"), dtype=np.float64)

        best_projected_for_detected = np.argmin(distances_px, axis=0)
        max_dist = float(max_dist_px)
        min_margin = float(ambiguity_margin_px)

        matched_corners: List[TrackerCorner] = []
        accepted_distances: List[float] = []
        accepted_uvs: List[Tuple[float, float]] = []
        accepted_xyz: List[Tuple[float, float, float]] = []
        accepted_rows: List[int] = []
        accepted_cols: List[int] = []

        for projected_idx in range(len(projected_uvs)):
            detected_idx = int(best_detected_for_projected[projected_idx])
            best_dist = float(best_distances[projected_idx])
            second_dist = float(second_distances[projected_idx])

            if best_dist > max_dist:
                stats.rejected_far += 1
                continue

            if np.isfinite(second_dist) and (second_dist - best_dist) < min_margin:
                stats.rejected_ambiguous += 1
                continue

            if int(best_projected_for_detected[detected_idx]) != projected_idx:
                stats.rejected_non_mutual += 1
                continue

            det = detected[detected_idx]
            gr = int(projected_rows[projected_idx])
            gc = int(projected_cols[projected_idx])
            xyz = tuple(float(v) for v in projected_xyz[projected_idx])
            uv = (float(det.uv[0]), float(det.uv[1]))

            matched_corners.append(
                TrackerCorner(
                    local_row=int(det.local_row),
                    local_col=int(det.local_col),
                    global_row=gr,
                    global_col=gc,
                    xyz_mm=xyz,
                    uv=uv,
                    votes=0,
                )
            )
            accepted_distances.append(best_dist)
            accepted_uvs.append(uv)
            accepted_xyz.append(xyz)
            accepted_rows.append(gr)
            accepted_cols.append(gc)

        if not accepted_distances:
            return [], stats

        distances_arr = np.asarray(accepted_distances, dtype=np.float64)
        stats.median_error_px = float(np.median(distances_arr))
        stats.p90_error_px = float(np.percentile(distances_arr, 90))

        matched_uvs = np.asarray(accepted_uvs, dtype=np.float64).reshape(-1, 2)
        detected_span = np.ptp(detected_uvs, axis=0)
        matched_span = np.ptp(matched_uvs, axis=0)
        stats.image_span_u_px = float(matched_span[0])
        stats.image_span_v_px = float(matched_span[1])
        detected_area = float(detected_span[0] * detected_span[1])
        if detected_area > 1.0:
            matched_area = float(matched_span[0] * matched_span[1])
            stats.image_coverage = float(np.clip(matched_area / detected_area, 0.0, 1.0))

        matched_xyz = np.asarray(accepted_xyz, dtype=np.float64).reshape(-1, 3)
        object_span = np.ptp(matched_xyz, axis=0)
        stats.object_span_mm = float(np.linalg.norm(object_span))
        stats.distinct_rows = int(len(set(accepted_rows)))
        stats.distinct_cols = int(len(set(accepted_cols)))

        return matched_corners, stats

    def _projected_tracker_corners_for_detection_pose(
        self,
        detection,
        rvec: Optional[np.ndarray],
        tvec: Optional[np.ndarray],
        max_dist_px: float,
    ) -> Tuple[List[TrackerCorner], int, float, float]:
        """Greedily align detected corners to projected geometry for visual validation."""
        if detection is None or rvec is None or tvec is None:
            return [], 0, float("inf"), float("inf")

        if bool(getattr(self.config, "cpp_dense_projection_matcher_enabled", True)):
            helper = self._ensure_cpp_tracker_geometry()
            if helper is not None:
                try:
                    result = helper.greedy_projected_match(
                        detection,
                        rvec,
                        tvec,
                        float(max_dist_px),
                    )
                    stats = self._dense_projection_match_stats_from_cpp(
                        result.stats
                    )
                    return (
                        self._geometry_tracker_corners_from_cpp(result.corners),
                        len(result.corners),
                        float(stats.median_error_px),
                        float(stats.p90_error_px),
                    )
                except Exception:
                    self._cpp_tracker_geometry_unavailable = True

        detected = self._detected_corners_from_detection(detection)
        if not detected:
            return [], 0, float("inf"), float("inf")

        projected: List[Tuple[int, int, Tuple[float, float, float], Tuple[float, float]]] = []
        rows = self.geometry.corner_rows()
        cols = self.geometry.corner_cols()
        for gr in range(rows):
            for gc in range(cols):
                if not self.geometry.has_corner(gr, gc):
                    continue

                pt = self.geometry.corner_point(gr, gc)
                xyz = (float(pt.x), float(pt.y), float(pt.z))
                uv = self._project_point_uv_with_pose(xyz, rvec, tvec)
                if uv is None:
                    continue
                projected.append((int(gr), int(gc), xyz, uv))

        if not projected:
            return [], 0, float("inf"), float("inf")

        projected_uvs = np.asarray([p[3] for p in projected], dtype=np.float64)
        max_dist_sq = float(max_dist_px) * float(max_dist_px)
        used_projected: set[int] = set()
        matched_corners: List[TrackerCorner] = []
        distances: List[float] = []

        for det in detected:
            duv = np.asarray([float(det.uv[0]), float(det.uv[1])], dtype=np.float64)
            dist_sq = ((projected_uvs - duv) ** 2).sum(axis=1)
            order = np.argsort(dist_sq)

            best_idx = -1
            for idx in order:
                i = int(idx)
                if i not in used_projected:
                    best_idx = i
                    break

            if best_idx < 0 or float(dist_sq[best_idx]) > max_dist_sq:
                continue

            used_projected.add(best_idx)
            gr, gc, xyz, _ = projected[best_idx]
            distances.append(float(np.sqrt(dist_sq[best_idx])))
            matched_corners.append(
                TrackerCorner(
                    local_row=int(det.local_row),
                    local_col=int(det.local_col),
                    global_row=gr,
                    global_col=gc,
                    xyz_mm=xyz,
                    uv=(float(det.uv[0]), float(det.uv[1])),
                    votes=0,
                )
            )

        if not distances:
            return [], 0, float("inf"), float("inf")

        return (
            matched_corners,
            len(distances),
            float(np.median(distances)),
            float(np.percentile(distances, 90)),
        )


class DenseRefineMixin:
    """Refine fast-path poses using dense projected-corner correspondences."""

    def _dense_refine_pose_variants(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        method_prefix: str,
    ) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """Generate configured LM/VVS refinements for a candidate dense pose."""
        variants = [
            (
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                method_prefix,
            )
        ]

        configured = str(
            self.config.fast_persistent_dense_robust_refine_method or "auto"
        ).lower()
        methods: Tuple[str, ...]
        if configured == "auto":
            methods = ("lm", "vvs")
        elif configured in ("lm", "vvs"):
            methods = (configured,)
        else:
            methods = tuple()

        for method in methods:
            if method == "lm" and hasattr(cv2, "solvePnPRefineLM"):
                try:
                    refined = cv2.solvePnPRefineLM(
                        object_points,
                        image_points,
                        self.K,
                        self.dist_coeffs,
                        np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy(),
                        np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy(),
                    )
                    if refined is not None:
                        rvec_ref, tvec_ref = refined[:2]
                        variants.append(
                            (
                                np.asarray(rvec_ref, dtype=np.float64).reshape(3, 1),
                                np.asarray(tvec_ref, dtype=np.float64).reshape(3, 1),
                                f"{method_prefix}_lm",
                            )
                        )
                except Exception:
                    pass

            if method == "vvs" and hasattr(cv2, "solvePnPRefineVVS"):
                try:
                    refined = cv2.solvePnPRefineVVS(
                        object_points,
                        image_points,
                        self.K,
                        self.dist_coeffs,
                        np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy(),
                        np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy(),
                    )
                    if refined is not None:
                        rvec_ref, tvec_ref = refined[:2]
                        variants.append(
                            (
                                np.asarray(rvec_ref, dtype=np.float64).reshape(3, 1),
                                np.asarray(tvec_ref, dtype=np.float64).reshape(3, 1),
                                f"{method_prefix}_vvs",
                            )
                        )
                except Exception:
                    pass

        return variants

    def _score_dense_pose_candidate(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> Optional[Tuple[float, np.ndarray]]:
        """Score a dense pose candidate from robust reprojection-error statistics."""
        errors = self._reprojection_errors_for_pose(
            object_points,
            image_points,
            rvec,
            tvec,
        )
        if errors is None or len(errors) == 0 or not np.all(np.isfinite(errors)):
            return None

        median = float(np.median(errors))
        p90 = float(np.percentile(errors, 90))
        mean = float(np.mean(errors))
        score = median + 0.35 * p90 + 0.15 * mean
        return score, errors

    def _estimate_dense_pose_with_robust_solver(
        self,
        track_points: List[PoseTrackPoint],
        tracker_corners: List[TrackerCorner],
        success_message: str,
        pose_source: PoseSource,
        detection=None,
    ) -> TrackerResult:
        """Estimate a dense fallback pose by trying and robustly trimming candidates."""
        if bool(getattr(self.config, "cpp_dense_robust_solver_enabled", True)):
            result = self._estimate_dense_pose_with_robust_solver_cpp(
                track_points,
                tracker_corners,
                success_message,
                pose_source,
                detection=detection,
            )
            if result is not None:
                return result

        pnp_t0 = time.perf_counter()

        object_points = np.asarray(
            [p.xyz_mm for p in track_points],
            dtype=np.float64,
        ).reshape(-1, 3)
        image_points = np.asarray(
            [p.uv for p in track_points],
            dtype=np.float64,
        ).reshape(-1, 2)

        candidates: List[Tuple[np.ndarray, np.ndarray, str]] = []

        if self.pose_tracker.rvec is not None and self.pose_tracker.tvec is not None:
            candidates.extend(
                self._dense_refine_pose_variants(
                    object_points,
                    image_points,
                    self.pose_tracker.rvec,
                    self.pose_tracker.tvec,
                    "dense_seed",
                )
            )

        solve_flags: List[Tuple[int, str]] = []
        if hasattr(cv2, "SOLVEPNP_SQPNP"):
            solve_flags.append((int(cv2.SOLVEPNP_SQPNP), "dense_sqpnp"))
        if hasattr(cv2, "SOLVEPNP_EPNP"):
            solve_flags.append((int(cv2.SOLVEPNP_EPNP), "dense_epnp"))

        for flag, name in solve_flags:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    self.K,
                    self.dist_coeffs,
                    flags=flag,
                )
            except Exception:
                continue
            if not success:
                continue

            candidates.extend(
                self._dense_refine_pose_variants(
                    object_points,
                    image_points,
                    rvec,
                    tvec,
                    name,
                )
            )

        if self.pose_tracker.rvec is not None and self.pose_tracker.tvec is not None:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    self.K,
                    self.dist_coeffs,
                    rvec=np.asarray(self.pose_tracker.rvec, dtype=np.float64).reshape(3, 1).copy(),
                    tvec=np.asarray(self.pose_tracker.tvec, dtype=np.float64).reshape(3, 1).copy(),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if success:
                    candidates.extend(
                        self._dense_refine_pose_variants(
                            object_points,
                            image_points,
                            rvec,
                            tvec,
                            "dense_iterative_guess",
                        )
                    )
            except Exception:
                pass

        best: Optional[Tuple[float, np.ndarray, np.ndarray, str, np.ndarray]] = None
        for cand_rvec, cand_tvec, method in candidates:
            scored = self._score_dense_pose_candidate(
                object_points,
                image_points,
                cand_rvec,
                cand_tvec,
            )
            if scored is None:
                continue
            score, errors = scored
            if best is None or score < best[0]:
                best = (
                    float(score),
                    np.asarray(cand_rvec, dtype=np.float64).reshape(3, 1),
                    np.asarray(cand_tvec, dtype=np.float64).reshape(3, 1),
                    method,
                    errors,
                )

        if best is None:
            return TrackerResult(
                success=False,
                mode=self.mode,
                message="Dense robust solver failed: no candidate.",
                num_points=len(track_points),
                num_inliers=0,
                pnp_method="dense_robust_failed",
                corners=[],
                correspondence_corners=tracker_corners,
                timings_ms={"pnp_ms": (time.perf_counter() - pnp_t0) * 1000.0},
            )

        _, rvec, tvec, method, errors = best
        inlier_idx = np.arange(len(track_points), dtype=np.int64)

        if self.config.fast_persistent_dense_robust_trim_enabled and len(errors) >= 12:
            median = float(np.median(errors))
            mad = float(np.median(np.abs(errors - median)))
            robust_sigma = 1.4826 * mad
            robust_threshold = max(0.75, median + 4.0 * robust_sigma)
            max_threshold = float(self.config.fast_persistent_dense_robust_max_max_px)
            threshold = min(max_threshold, robust_threshold)

            quantile = float(self.config.fast_persistent_dense_robust_trim_quantile)
            if 0.0 < quantile < 1.0:
                threshold = min(threshold, float(np.percentile(errors, quantile * 100.0)))

            keep_mask = errors <= threshold
            min_keep = max(
                int(self.config.min_inliers),
                int(np.ceil(float(self.config.fast_persistent_dense_robust_min_keep_ratio) * len(errors))),
            )
            if int(np.count_nonzero(keep_mask)) >= min_keep and not np.all(keep_mask):
                trim_idx = np.where(keep_mask)[0].astype(np.int64)
                object_trim = object_points[trim_idx]
                image_trim = image_points[trim_idx]
                trim_candidates = self._dense_refine_pose_variants(
                    object_trim,
                    image_trim,
                    rvec,
                    tvec,
                    f"{method}_trim{len(trim_idx)}",
                )
                trim_best: Optional[Tuple[float, np.ndarray, np.ndarray, str, np.ndarray]] = None
                for cand_rvec, cand_tvec, trim_method in trim_candidates:
                    scored = self._score_dense_pose_candidate(
                        object_trim,
                        image_trim,
                        cand_rvec,
                        cand_tvec,
                    )
                    if scored is None:
                        continue
                    score, trim_errors = scored
                    if trim_best is None or score < trim_best[0]:
                        trim_best = (
                            float(score),
                            np.asarray(cand_rvec, dtype=np.float64).reshape(3, 1),
                            np.asarray(cand_tvec, dtype=np.float64).reshape(3, 1),
                            trim_method,
                            trim_errors,
                        )

                if trim_best is not None:
                    _, rvec, tvec, method, errors = trim_best
                    inlier_idx = trim_idx

        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))

        if (
            mean_err > self.config.fast_persistent_dense_robust_max_mean_px
            or max_err > self.config.fast_persistent_dense_robust_max_max_px
        ):
            return TrackerResult(
                success=False,
                mode=self.mode,
                message=(
                    "Dense robust pose rejected by reprojection gate "
                    f"(mean={mean_err:.3f}, max={max_err:.3f})."
                ),
                rvec=rvec,
                tvec=tvec,
                T_marker_camera=make_transform_from_rvec_tvec(rvec, tvec),
                mean_reprojection_error_px=mean_err,
                max_reprojection_error_px=max_err,
                num_points=len(track_points),
                num_inliers=len(inlier_idx),
                pnp_method=method,
                corners=[],
                correspondence_corners=tracker_corners,
                timings_ms={"pnp_ms": (time.perf_counter() - pnp_t0) * 1000.0},
            )

        if not self._persistent_pose_motion_plausible(
            rvec,
            tvec,
            self._last_accepted_rvec,
            self._last_accepted_tvec,
        ):
            return TrackerResult(
                success=False,
                mode=self.mode,
                message="Dense robust pose rejected by motion gate.",
                rvec=rvec,
                tvec=tvec,
                T_marker_camera=make_transform_from_rvec_tvec(rvec, tvec),
                mean_reprojection_error_px=mean_err,
                max_reprojection_error_px=max_err,
                num_points=len(track_points),
                num_inliers=len(inlier_idx),
                pnp_method=method,
                corners=[],
                correspondence_corners=tracker_corners,
                timings_ms={"pnp_ms": (time.perf_counter() - pnp_t0) * 1000.0},
            )

        reject_reason = self._fallback_pose_rejection_reason(
            detection,
            rvec,
            tvec,
            mean_err,
            max_err,
        )
        if reject_reason:
            return TrackerResult(
                success=False,
                mode=self.mode,
                message=reject_reason,
                rvec=rvec,
                tvec=tvec,
                T_marker_camera=make_transform_from_rvec_tvec(rvec, tvec),
                mean_reprojection_error_px=mean_err,
                max_reprojection_error_px=max_err,
                num_points=len(track_points),
                num_inliers=len(inlier_idx),
                pnp_method=method,
                corners=[],
                correspondence_corners=tracker_corners,
                timings_ms={"pnp_ms": (time.perf_counter() - pnp_t0) * 1000.0},
            )

        inlier_corners = [
            tracker_corners[int(i)]
            for i in inlier_idx
            if 0 <= int(i) < len(tracker_corners)
        ]
        inlier_points = [
            track_points[int(i)]
            for i in inlier_idx
            if 0 <= int(i) < len(track_points)
        ]
        pose_for_filter = MapPoseResult(
            success=True,
            message="Dense robust pose accepted.",
            rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            T_marker_camera=make_transform_from_rvec_tvec(rvec, tvec),
            inlier_indices=np.asarray(inlier_idx, dtype=np.int64).reshape(-1),
            reprojection_mean_px=mean_err,
            reprojection_max_px=max_err,
            num_points=len(track_points),
            num_inliers=len(inlier_idx),
            points=inlier_points,
            method=method,
        )
        filtered_depth = self._apply_depth_filter_to_pose(pose_for_filter, track_points)
        rvec = np.asarray(pose_for_filter.rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(pose_for_filter.tvec, dtype=np.float64).reshape(3, 1)
        mean_err = float(pose_for_filter.reprojection_mean_px)
        max_err = float(pose_for_filter.reprojection_max_px)
        visual_corners = self._visual_corners_from_pose(
            inlier_corners,
            rvec,
            tvec,
        )
        visual_note = ""
        if len(visual_corners) != len(inlier_corners):
            visual_note = (
                f" Visual corners filtered {len(visual_corners)}/"
                f"{len(inlier_corners)}."
            )
        if len(visual_corners) < self.config.visual_corner_min_count:
            visual_corners = []
            visual_note += " Visual corners suppressed for dense robust pose."

        T = pose_for_filter.T_marker_camera
        self.pose_tracker.rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy()
        self.pose_tracker.tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy()
        self.pose_tracker.T_marker_camera = np.asarray(T, dtype=np.float64).reshape(4, 4)

        if len(visual_corners) >= self.config.visual_corner_min_count:
            if len(inlier_idx) > self._max_pts_seen:
                self._max_pts_seen = len(inlier_idx)
            self._last_good_reproj_px = mean_err
            self._last_accepted_rvec = self.pose_tracker.rvec.copy()
            self._last_accepted_tvec = self.pose_tracker.tvec.copy()
            self._last_accepted_T_marker_camera = self.pose_tracker.T_marker_camera.copy()
            self._last_accepted_pose_frame = self.frame_index

        confidence = self._confidence(len(inlier_idx), mean_err)
        return TrackerResult(
            success=True,
            mode=TrackerMode.TRACKING,
            message=success_message + visual_note,
            corners=visual_corners,
            correspondence_corners=tracker_corners,
            rvec=self.pose_tracker.rvec.copy(),
            tvec=self.pose_tracker.tvec.copy(),
            T_marker_camera=self.pose_tracker.T_marker_camera.copy(),
            mean_reprojection_error_px=mean_err,
            max_reprojection_error_px=max_err,
            num_points=len(track_points),
            num_inliers=len(inlier_idx),
            confidence=confidence,
            pose_source=pose_source,
            pnp_method=method,
            timings_ms={"pnp_ms": (time.perf_counter() - pnp_t0) * 1000.0},
            **self._depth_filter_kwargs(filtered_depth),
        )

    def _estimate_dense_pose_with_direct_prior_cpp(
        self,
        track_points: List[PoseTrackPoint],
        tracker_corners: List[TrackerCorner],
        success_message: str,
        pose_source: PoseSource,
        detection=None,
    ) -> Optional[TrackerResult]:
        """Use C++ for dense direct-prior pose solve, then Python packaging."""
        if not bool(getattr(self.config, "cpp_dense_direct_solver_enabled", True)):
            return None

        helper = self._ensure_cpp_tracker_geometry()
        if helper is None:
            return None

        prev_pose_rvec = (
            None
            if self.pose_tracker.rvec is None
            else self.pose_tracker.rvec.copy()
        )
        prev_pose_tvec = (
            None
            if self.pose_tracker.tvec is None
            else self.pose_tracker.tvec.copy()
        )
        prev_pose_T = (
            None
            if self.pose_tracker.T_marker_camera is None
            else self.pose_tracker.T_marker_camera.copy()
        )
        prev_depth_filter_state = self.pose_depth_filter.snapshot()
        prev_last_rvec = (
            None
            if self._last_accepted_rvec is None
            else self._last_accepted_rvec.copy()
        )
        prev_last_tvec = (
            None
            if self._last_accepted_tvec is None
            else self._last_accepted_tvec.copy()
        )

        pnp_t0 = time.perf_counter()
        try:
            pose_cpp = helper.estimate_dense_direct_pose(
                hm.pose_track_points_from_python(track_points),
                prev_pose_rvec,
                prev_pose_tvec,
                int(self.lost_frames),
            )
        except Exception:
            self._cpp_tracker_geometry_unavailable = True
            return None

        pnp_ms = (time.perf_counter() - pnp_t0) * 1000.0
        pose = self._map_pose_result_from_cpp(pose_cpp)
        if pose.success:
            if pose.rvec is None or pose.tvec is None or pose.T_marker_camera is None:
                return None
            self.pose_tracker.rvec = pose.rvec.copy()
            self.pose_tracker.tvec = pose.tvec.copy()
            self.pose_tracker.T_marker_camera = pose.T_marker_camera.copy()

        return self._estimate_and_package_pose(
            track_points,
            tracker_corners,
            success_message=success_message,
            update_persistence=False,
            pose_source=pose_source,
            detection=detection,
            precomputed_pose=pose,
            precomputed_pnp_ms=pnp_ms,
            previous_pose_rvec=prev_pose_rvec,
            previous_pose_tvec=prev_pose_tvec,
            previous_pose_T=prev_pose_T,
            previous_depth_filter_state=prev_depth_filter_state,
            previous_last_rvec=prev_last_rvec,
            previous_last_tvec=prev_last_tvec,
        )

    def _estimate_dense_pose_with_robust_solver_cpp(
        self,
        track_points: List[PoseTrackPoint],
        tracker_corners: List[TrackerCorner],
        success_message: str,
        pose_source: PoseSource,
        detection=None,
    ) -> Optional[TrackerResult]:
        """Use the C++ dense robust solver while Python keeps packaging/state."""
        helper = self._ensure_cpp_tracker_geometry()
        if helper is None:
            return None

        pnp_t0 = time.perf_counter()
        try:
            pose_cpp = helper.estimate_dense_robust_pose(
                hm.pose_track_points_from_python(track_points),
                detection,
                self.pose_tracker.rvec,
                self.pose_tracker.tvec,
                self._last_accepted_rvec,
                self._last_accepted_tvec,
            )
        except Exception:
            self._cpp_tracker_geometry_unavailable = True
            return None

        pnp_ms = (time.perf_counter() - pnp_t0) * 1000.0
        pose = self._map_pose_result_from_cpp(pose_cpp)
        if pose.inlier_indices is None:
            inlier_idx = np.asarray([], dtype=np.int64)
        else:
            inlier_idx = np.asarray(pose.inlier_indices, dtype=np.int64).reshape(-1)

        if not pose.success:
            return TrackerResult(
                success=False,
                mode=self.mode,
                message=pose.message,
                rvec=pose.rvec,
                tvec=pose.tvec,
                T_marker_camera=pose.T_marker_camera,
                mean_reprojection_error_px=float(pose.reprojection_mean_px),
                max_reprojection_error_px=float(pose.reprojection_max_px),
                num_points=len(track_points),
                num_inliers=int(len(inlier_idx)),
                pnp_method=pose.method,
                corners=[],
                correspondence_corners=tracker_corners,
                timings_ms={"pnp_ms": pnp_ms},
            )

        if pose.rvec is None or pose.tvec is None or pose.T_marker_camera is None:
            return None

        inlier_corners = [
            tracker_corners[int(i)]
            for i in inlier_idx
            if 0 <= int(i) < len(tracker_corners)
        ]
        pose_for_filter = pose
        pose_for_filter.points = [
            track_points[int(i)]
            for i in inlier_idx
            if 0 <= int(i) < len(track_points)
        ]
        filtered_depth = self._apply_depth_filter_to_pose(
            pose_for_filter,
            track_points,
        )

        rvec = np.asarray(pose_for_filter.rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(pose_for_filter.tvec, dtype=np.float64).reshape(3, 1)
        mean_err = float(pose_for_filter.reprojection_mean_px)
        max_err = float(pose_for_filter.reprojection_max_px)

        visual_corners = self._visual_corners_from_pose(
            inlier_corners,
            rvec,
            tvec,
        )
        visual_note = ""
        if len(visual_corners) != len(inlier_corners):
            visual_note = (
                f" Visual corners filtered {len(visual_corners)}/"
                f"{len(inlier_corners)}."
            )
        if len(visual_corners) < self.config.visual_corner_min_count:
            visual_corners = []
            visual_note += " Visual corners suppressed for dense robust pose."

        T = pose_for_filter.T_marker_camera
        self.pose_tracker.rvec = rvec.copy()
        self.pose_tracker.tvec = tvec.copy()
        self.pose_tracker.T_marker_camera = np.asarray(
            T,
            dtype=np.float64,
        ).reshape(4, 4)

        if len(visual_corners) >= self.config.visual_corner_min_count:
            if len(inlier_idx) > self._max_pts_seen:
                self._max_pts_seen = len(inlier_idx)
            self._last_good_reproj_px = mean_err
            self._last_accepted_rvec = self.pose_tracker.rvec.copy()
            self._last_accepted_tvec = self.pose_tracker.tvec.copy()
            self._last_accepted_T_marker_camera = (
                self.pose_tracker.T_marker_camera.copy()
            )
            self._last_accepted_pose_frame = self.frame_index

        confidence = self._confidence(len(inlier_idx), mean_err)
        return TrackerResult(
            success=True,
            mode=TrackerMode.TRACKING,
            message=success_message + visual_note,
            corners=visual_corners,
            correspondence_corners=tracker_corners,
            rvec=self.pose_tracker.rvec.copy(),
            tvec=self.pose_tracker.tvec.copy(),
            T_marker_camera=self.pose_tracker.T_marker_camera.copy(),
            mean_reprojection_error_px=mean_err,
            max_reprojection_error_px=max_err,
            num_points=len(track_points),
            num_inliers=len(inlier_idx),
            confidence=confidence,
            pose_source=pose_source,
            pnp_method=pose.method,
            timings_ms={"pnp_ms": pnp_ms},
            **self._depth_filter_kwargs(filtered_depth),
        )

    def _set_dense_refine_debug(
        self,
        *,
        attempted: bool,
        success: bool = False,
        reason: str = "",
        matches: int = 0,
        median_error_px: float = -1.0,
        p90_error_px: float = -1.0,
        stats: Optional[DenseProjectionMatchStats] = None,
    ) -> None:
        """Attach dense-refine diagnostics to the current fast-path debug record."""
        debug = self._last_fast_path_debug
        debug.dense_refine_attempted = bool(attempted)
        debug.dense_refine_success = bool(success)
        debug.dense_refine_reason = str(reason)
        debug.dense_refine_matches = int(matches)
        debug.dense_refine_median_error_px = float(median_error_px)
        debug.dense_refine_p90_error_px = float(p90_error_px)
        if stats is None:
            return

        debug.dense_refine_projected = int(stats.projected)
        debug.dense_refine_detected = int(stats.detected)
        debug.dense_refine_rejected_no_projection = int(stats.rejected_no_projection)
        debug.dense_refine_rejected_far = int(stats.rejected_far)
        debug.dense_refine_rejected_ambiguous = int(stats.rejected_ambiguous)
        debug.dense_refine_rejected_non_mutual = int(stats.rejected_non_mutual)
        debug.dense_refine_image_coverage = float(stats.image_coverage)
        debug.dense_refine_image_span_u_px = float(stats.image_span_u_px)
        debug.dense_refine_image_span_v_px = float(stats.image_span_v_px)
        debug.dense_refine_object_span_mm = float(stats.object_span_mm)
        debug.dense_refine_distinct_rows = int(stats.distinct_rows)
        debug.dense_refine_distinct_cols = int(stats.distinct_cols)
