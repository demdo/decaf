from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tracking.hydramarker.map_pose_tracker import PoseTrackPoint
from tracking.hydramarker.tracker_types import (
    DenseProjectionMatchStats,
    PoseSource,
    TrackerCorner,
    TrackerMode,
    TrackerResult,
)
from tracking.pose_solvers import make_transform_from_rvec_tvec


class DenseRefineMixin:
    def _dense_refine_pose_variants(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        method_prefix: str,
    ) -> List[Tuple[np.ndarray, np.ndarray, str]]:
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

        T = make_transform_from_rvec_tvec(rvec, tvec)
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

