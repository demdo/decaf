from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from tracking.hydramarker.tracker_types import (
    DenseProjectionMatchStats,
    TrackerCorner,
)


class ProjectionMixin:
    def _projected_tracker_corners_from_current_pose(self) -> List[TrackerCorner]:
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
        if rvec is None or tvec is None:
            return []

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
        stats = DenseProjectionMatchStats()

        if detection is None or rvec is None or tvec is None:
            return [], stats

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
        if detection is None or rvec is None or tvec is None:
            return [], 0, float("inf"), float("inf")

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
