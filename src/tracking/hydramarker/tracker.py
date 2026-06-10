from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tracking.hydramarker.backend import cpp_impl as hm
from tracking.hydramarker.identity_store import (
    GlobalCornerIdentity,
    IdentityStore,
)
from tracking.hydramarker.map_pose_tracker import (
    MapPoseTracker,
    MapPoseTrackerConfig,
    PoseTrackPoint,
)
from tracking.hydramarker.tracker_logger import TrackerLogger
from tracking.pose_solvers import make_transform_from_rvec_tvec


from tracking.hydramarker.config import TrackerConfig
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


class HydraTracker:
    def __init__(
        self,
        field_path: str,
        marker_json_path: str,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        config: Optional[TrackerConfig] = None,
    ) -> None:
        self.config = config or TrackerConfig()

        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = (
            np.zeros((0, 1), dtype=np.float64)
            if dist_coeffs is None
            else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        )

        self.field = hm.MarkerField.loadFromFile(field_path)
        self.geometry = hm.MarkerGeometry.load_from_json(marker_json_path)
        self._geometry_corner_cache = self._build_geometry_corner_cache()

        # CheckerboardDetector mit expliziter Config:
        # recovery_correction_max_dist_rel erhöht damit LK-Drift-Korrektur
        # auch bei größerem Offset zwischen LK- und Recovery-Corner greift.
        _cbd_cfg = hm.CheckerboardDetectorConfig()
        _cbd_cfg.recovery_correction_weight = 0.5
        _cbd_cfg.recovery_correction_max_dist_rel = 0.6
        if hasattr(_cbd_cfg, "min_tracking_decode_cell_span"):
            _cbd_cfg.min_tracking_decode_cell_span = (
                self.config.checker_min_tracking_decode_cell_span
            )
        if hasattr(_cbd_cfg, "max_undecodeable_tracking_frames"):
            _cbd_cfg.max_undecodeable_tracking_frames = (
                self.config.checker_max_undecodeable_tracking_frames
            )
        self.checkerboard_detector = hm.CheckerboardDetector(_cbd_cfg)
        self.dot_detector = self._create_dot_detector()
        self.patch_extractor = hm.PatchExtractor()
        self.patch_decoder = self._create_patch_decoder()
        self.correspondence_builder = self._create_correspondence_builder()

        self.pose_tracker = MapPoseTracker(
            K=K,
            dist_coeffs=dist_coeffs,
            config=MapPoseTrackerConfig(
                min_points=self.config.min_points,
                min_inliers=self.config.min_inliers,
                ransac_reproj_px=self.config.pnp_ransac_reprojection_px,
                ransac_confidence=self.config.pnp_ransac_confidence,
                ransac_iterations=self.config.pnp_ransac_iterations,
                max_mean_reproj_px=self.config.max_mean_reprojection_error_px,
                max_max_reproj_px=self.config.max_max_reprojection_error_px,
                max_translation_jump_mm=self.config.max_translation_jump_mm,
                max_rotation_jump_deg=self.config.max_rotation_jump_deg,
                rotation_gate_scale_per_lost_frame=self.config.rotation_gate_scale_per_lost_frame,
                rotation_gate_max_deg=self.config.rotation_gate_max_deg,
                use_pose_prior=self.config.use_pose_prior,
                refine_with_iterative=True,
                use_direct_prior_solver=self.config.pnp_direct_prior_enabled,
                direct_refine_method=self.config.pnp_direct_refine_method,
                direct_max_mean_reproj_px=(
                    self.config.pnp_direct_max_mean_reprojection_error_px
                ),
                direct_max_max_reproj_px=(
                    self.config.pnp_direct_max_max_reprojection_error_px
                ),
            ),
        )

        self.logger = (
            TrackerLogger(
                log_path=self.config.log_path,
                enable_console=self.config.log_to_console,
            )
            if self.config.enable_debug_prints
            else None
        )

        self.mode = TrackerMode.LOST
        self.frame_index = 0
        self.lost_frames = 0

        # Höchste bisher gesehene pts-Anzahl — für Frühwarnung LK-Drift.
        self._max_pts_seen: int = 0

        # Letzter akzeptierter Reprojektionsfehler — für Pose-Propagation.
        self._last_good_reproj_px: float = -1.0

        # Letzte akzeptierte rvec — für präventiven Rotations-Delta-Check.
        self._last_accepted_rvec: Optional[np.ndarray] = None
        self._last_accepted_tvec: Optional[np.ndarray] = None
        self._last_accepted_T_marker_camera: Optional[np.ndarray] = None
        self._last_accepted_pose_frame: int = -1

        self._identity_store = IdentityStore()
        self._persistent_frame_index: int = -1
        self._undecodeable_detection_frames: int = 0
        self._low_fresh_correspondence_frames: int = 0
        self._pose_propagation_block_until_frame: int = -1
        self._last_uncoded_bootstrap_reason: str = ""
        self._last_persistent_match_stats = PersistentMatchStats()
        self._last_fast_path_debug = FastPathDebug()

    @property
    def rvec(self) -> Optional[np.ndarray]:
        return self.pose_tracker.rvec

    @property
    def tvec(self) -> Optional[np.ndarray]:
        return self.pose_tracker.tvec

    @property
    def T_marker_camera(self) -> Optional[np.ndarray]:
        return self.pose_tracker.T_marker_camera

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

    def _create_dot_detector(self):
        cfg = hm.DotDetectorConfig()

        cfg.canonical_size = self.config.dot_canonical_size
        cfg.canonical_margin_px = self.config.dot_canonical_margin_px

        cfg.min_dot_contrast = self.config.dot_min_dot_contrast
        cfg.strong_dot_contrast = self.config.dot_strong_dot_contrast

        cfg.commit_threshold = self.config.dot_commit_threshold
        cfg.revoke_threshold = self.config.dot_revoke_threshold

        cfg.uncertainty_low = self.config.dot_uncertainty_low
        cfg.uncertainty_high = self.config.dot_uncertainty_high

        cfg.warmup_frames = self.config.dot_warmup_frames

        cfg.temporal_alpha = self.config.dot_temporal_alpha
        cfg.commit_frames = self.config.dot_commit_frames
        cfg.revoke_frames = self.config.dot_revoke_frames
        cfg.use_temporal_smoothing = self.config.dot_use_temporal_smoothing
        if hasattr(cfg, "use_cell_value_cache"):
            cfg.use_cell_value_cache = self.config.dot_use_cell_value_cache
        if hasattr(cfg, "cell_cache_max_age_frames"):
            cfg.cell_cache_max_age_frames = self.config.dot_cell_cache_max_age_frames
        if hasattr(cfg, "cell_cache_max_corner_motion_px"):
            cfg.cell_cache_max_corner_motion_px = self.config.dot_cell_cache_max_corner_motion_px

        return hm.DotDetector(cfg)

    def _create_patch_decoder(self):
        cfg = hm.PatchDecoderConfig()
        cfg.require_geometry_valid = self.config.decoder_require_geometry_valid
        cfg.accept_ambiguous = self.config.decoder_accept_ambiguous
        return hm.PatchDecoder(cfg)

    def _create_correspondence_builder(self):
        cfg = hm.CorrespondenceBuilderConfig()
        cfg.min_votes = self.config.corr_min_votes
        cfg.discard_conflicts = self.config.corr_discard_conflicts
        cfg.require_detection_stable = self.config.corr_require_detection_stable
        cfg.enable_dominant_rotation_filter = self.config.corr_enable_dominant_rotation_filter
        cfg.min_rotation_support = self.config.corr_min_rotation_support
        cfg.min_rotation_support_ratio = self.config.corr_min_rotation_support_ratio
        return hm.CorrespondenceBuilder(cfg)

    def reset(self) -> None:
        self.mode = TrackerMode.LOST
        self.frame_index = 0
        self.lost_frames = 0
        self._max_pts_seen = 0
        self._last_good_reproj_px = -1.0
        self._last_accepted_rvec = None
        self._last_accepted_tvec = None
        self._last_accepted_T_marker_camera = None
        self._last_accepted_pose_frame = -1
        self._undecodeable_detection_frames = 0
        self._low_fresh_correspondence_frames = 0
        self._pose_propagation_block_until_frame = -1
        self._last_uncoded_bootstrap_reason = ""
        self._last_persistent_match_stats = PersistentMatchStats()
        self._last_fast_path_debug = FastPathDebug()

        self.pose_tracker.reset()
        self.checkerboard_detector.reset_tracking()

        # Full reset: recreate dot detector to clear all smoothed state.
        # reset_smoothing() is called on partial resets (_on_tracking_failure)
        # to preserve warmup state while clearing stale cell scores.
        self.dot_detector = self._create_dot_detector()

        self._clear_persistent_correspondences()

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        run_detection: bool = True,
    ) -> TrackerResult:
        frame_t0 = time.perf_counter()
        timings_ms: Dict[str, float] = {}

        def mark(name: str, start: float) -> None:
            timings_ms[name] = (time.perf_counter() - start) * 1000.0

        def finish(result: TrackerResult) -> TrackerResult:
            timings_ms["tracker_total_ms"] = (time.perf_counter() - frame_t0) * 1000.0
            merged_timings = dict(getattr(result, "timings_ms", {}) or {})
            merged_timings.update(timings_ms)
            result.timings_ms = merged_timings
            return result

        self.frame_index += 1

        if not run_detection:
            result = TrackerResult(
                success=False,
                mode=self.mode,
                message="Idle: checkerboard detection skipped.",
            )
            result.fast_path_debug = FastPathDebug(reason="idle_skipped")
            timings_ms["checkerboard_ms"] = 0.0
            timings_ms["idle_skip"] = 1.0
            self._last_fast_path_debug = result.fast_path_debug
            return finish(result)

        stage_t0 = time.perf_counter()
        detection = self.checkerboard_detector.detect(frame)
        mark("checkerboard_ms", stage_t0)
        if hasattr(self.checkerboard_detector, "last_timings_ms"):
            try:
                for key, value in self.checkerboard_detector.last_timings_ms().items():
                    timings_ms[f"checkerboard_{key}"] = float(value)
            except Exception:
                pass

        if detection is None or not detection.valid():
            self._last_fast_path_debug = FastPathDebug(reason="no_checkerboard")
            self._undecodeable_detection_frames = 0
            self._on_tracking_failure()
            stage_t0 = time.perf_counter()
            held = self._hold_last_pose_without_detection_result(detection)
            mark("hold_pose_ms", stage_t0)
            if held is not None:
                self._attach_fast_path_debug(held)
                self._log_result("POSE_HELD_NO_DETECTION", held)
                return finish(held)
            stage_t0 = time.perf_counter()
            emergency = self._emergency_last_pose_result(
                detection,
                reason="No valid checkerboard detection",
            )
            mark("emergency_hold_ms", stage_t0)
            if emergency is not None:
                self._attach_fast_path_debug(emergency)
                self._log_result("POSE_HELD_EMERGENCY", emergency)
                return finish(emergency)

            result = TrackerResult(
                success=False,
                mode=self.mode,
                message="No valid checkerboard detection.",
                detection_valid=False,
                detection_tracking=False if detection is None else bool(detection.tracking),
                detection_stable=False if detection is None else bool(detection.stable),
                detection_corners=self._detected_corners_from_detection(detection),
            )
            self._attach_fast_path_debug(result)
            self._log_result("NO_DETECTION", result)
            return finish(result)

        stage_t0 = time.perf_counter()
        fast_result = self._try_fast_pose_from_persistent_correspondences(detection)
        mark("fast_persistent_ms", stage_t0)
        if fast_result is not None:
            self._attach_detection_info(fast_result, detection)
            self._attach_fast_path_debug(fast_result)
            self.mode = TrackerMode.TRACKING
            self.lost_frames = 0
            fast_result.mode = self.mode
            self._log_result("FAST_TRACK_OK", fast_result)
            return finish(fast_result)

        # Pose-Propagation: wenn eine gute Pose bekannt ist, ersetze die
        # LK-Detection durch projizierte Marker-Corners. Das eliminiert
        # LK-Drift-Akkumulation als Fehlerquelle für den Dot-Decoder.
        # Fallback auf normale Detection wenn Pose nicht gut genug.
        h, w = frame.shape[:2]
        stage_t0 = time.perf_counter()
        propagated = self._build_pose_propagated_detection((h, w))
        mark("pose_propagation_ms", stage_t0)
        detection_for_dots = propagated if propagated is not None else detection

        stage_t0 = time.perf_counter()
        result = self._decode_and_estimate_pose(frame, detection_for_dots)
        mark("decode_pose_ms", stage_t0)
        self._attach_detection_info(result, detection)
        self._attach_fast_path_debug(result)

        if result.success:
            self.mode = TrackerMode.TRACKING
            self.lost_frames = 0
            result.mode = self.mode
            self._log_result("TRACK_OK", result)
            return finish(result)

        self._on_tracking_failure()
        stage_t0 = time.perf_counter()
        emergency = self._emergency_last_pose_result(
            detection,
            reason=result.message,
        )
        mark("emergency_hold_ms", stage_t0)
        if emergency is not None:
            self._attach_fast_path_debug(emergency)
            self._log_result("POSE_HELD_EMERGENCY", emergency)
            return finish(emergency)

        result.mode = self.mode
        result.corners = []
        self._attach_fast_path_debug(result)
        self._log_result("TRACK_FAIL", result)
        return finish(result)

    def _on_tracking_failure(self) -> None:
        self.lost_frames += 1

        if self.lost_frames > self.config.max_lost_frames:
            self.pose_tracker.reset()
            self._clear_persistent_correspondences()
            self.mode = TrackerMode.LOST

            # Full dot detector reset only on complete loss — recreate to
            # clear all state including stale cell scores from a different
            # marker position.
            self.dot_detector = self._create_dot_detector()

        elif self.pose_tracker.rvec is not None and self.pose_tracker.tvec is not None:
            self.mode = TrackerMode.RECOVERING

            # Partial reset: tell the dot detector to clear its smoothed
            # scores so it re-commits quickly in the next frame, but don't
            # destroy the detector object (keeps warmup done).
            if hasattr(self.dot_detector, "reset_smoothing"):
                self.dot_detector.reset_smoothing()

        else:
            self.mode = TrackerMode.DETECTING

            if hasattr(self.dot_detector, "reset_smoothing"):
                self.dot_detector.reset_smoothing()


    def _build_pose_propagated_detection(
        self,
        image_shape: Tuple[int, int],
    ):
        """
        Baut eine synthetische CheckerboardDetection aus der letzten bekannten
        Pose durch Projektion aller MarkerGeometry-Corners in den aktuellen Frame.

        Ersetzt die LK-basierte Detection wenn die letzte Pose gut genug war.
        Vermeidet LK-Drift-Akkumulation bei längerem Tracking.

        Koordinaten-Mapping (aus correspondence_builder.cpp):
            global_row = vertikal = corner.j
            global_col = horizontal = corner.i

        Gibt None zurück wenn:
            - Keine gültige Pose vorhanden
            - Letzter Reprojektionsfehler zu hoch
            - Zu wenige Corners im Bild sichtbar
        """
        if self.config.decode_only_mode or not self.config.enable_pose_propagation:
            return None

        if self.frame_index <= self._pose_propagation_block_until_frame:
            return None

        rvec = self.pose_tracker.rvec
        tvec = self.pose_tracker.tvec

        if rvec is None or tvec is None:
            return None

        if (
            self._last_good_reproj_px < 0.0
            or self._last_good_reproj_px > self.config.pose_propagation_max_reproj_px
        ):
            return None

        rows = self.geometry.corner_rows()
        cols = self.geometry.corner_cols()
        border = self.config.pose_propagation_border_px
        h, w = image_shape[0], image_shape[1]

        # Alle gültigen 3D-Corners sammeln
        obj_pts = []
        row_col_list = []

        for gr in range(rows):
            for gc in range(cols):
                if not self.geometry.has_corner(gr, gc):
                    continue
                pt = self.geometry.corner_point(gr, gc)
                obj_pts.append([pt.x, pt.y, pt.z])
                row_col_list.append((gr, gc))

        if len(obj_pts) < self.config.min_points:
            return None

        obj_pts_np = np.array(obj_pts, dtype=np.float64).reshape(-1, 3)

        projected, _ = cv2.projectPoints(
            obj_pts_np,
            rvec.reshape(3, 1),
            tvec.reshape(3, 1),
            self.K,
            self.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)

        # Synthetische GridCorners bauen — nur sichtbare
        # global_row -> corner.j, global_col -> corner.i
        detection = hm.CheckerboardDetection()
        ij_to_uv: Dict[Tuple[int, int], Tuple[float, float]] = {}

        for idx, (gr, gc) in enumerate(row_col_list):
            u, v = float(projected[idx, 0]), float(projected[idx, 1])

            if u < border or v < border or u >= w - border or v >= h - border:
                continue

            corner = hm.GridCorner()
            corner.j = gr   # row = vertikal = j
            corner.i = gc   # col = horizontal = i
            corner.uv = hm.Point2f()
            corner.uv.x = u
            corner.uv.y = v
            corner.visibility_score = 1.0

            detection.corners.append(corner)
            ij_to_uv[(gc, gr)] = (u, v)  # key: (i,j)

        if len(detection.corners) < self.config.min_points:
            return None

        # Synthetische Cells aus projizierten Corners bauen
        # Cell (i,j) hat Corners: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
        for ci, cj in list(ij_to_uv.keys()):
            if (ci+1, cj) not in ij_to_uv:
                continue
            if (ci+1, cj+1) not in ij_to_uv:
                continue
            if (ci, cj+1) not in ij_to_uv:
                continue

            cell = hm.GridCell()
            cell.i = ci
            cell.j = cj

            p00 = ij_to_uv[(ci,   cj)]
            p10 = ij_to_uv[(ci+1, cj)]
            p11 = ij_to_uv[(ci+1, cj+1)]
            p01 = ij_to_uv[(ci,   cj+1)]

            def make_pt(xy):
                p = hm.Point2f()
                p.x = xy[0]
                p.y = xy[1]
                return p

            cell.corner_uv = [make_pt(p00), make_pt(p10), make_pt(p11), make_pt(p01)]
            cell.center_uv = make_pt((
                (p00[0]+p10[0]+p11[0]+p01[0]) * 0.25,
                (p00[1]+p10[1]+p11[1]+p01[1]) * 0.25,
            ))

            detection.cells.append(cell)

        if len(detection.cells) == 0:
            return None

        if not self._detection_has_decodeable_cell_span(detection):
            return None

        detection.tracking = True
        detection.stable = True

        return detection

    def _decode_and_estimate_pose(self, frame: np.ndarray, detection) -> TrackerResult:
        # Präventiver Dot-Detector-Reset bei starker Rotation des Drills.
        # MUSS vor dot_detector.detect() stehen damit der Reset im selben
        # Frame wirkt in dem die Rotation erkannt wird.
        # Zylindersymmetrie: bei rot_delta > 15° ändert sich welche Dots
        # sichtbar sind. Kompletter Reset damit EMA-Warmup nicht 10+ Frames
        # dauert.
        if (
            self.mode == TrackerMode.TRACKING
            and self._last_accepted_rvec is not None
            and self.pose_tracker.rvec is not None
        ):
            try:
                R_prev, _ = cv2.Rodrigues(
                    np.asarray(self._last_accepted_rvec, dtype=np.float64).reshape(3, 1)
                )
                R_curr, _ = cv2.Rodrigues(
                    np.asarray(self.pose_tracker.rvec, dtype=np.float64).reshape(3, 1)
                )
                dR = R_curr @ R_prev.T
                cos_a = float(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
                rot_delta_deg = float(np.degrees(np.arccos(cos_a)))
                if rot_delta_deg > 15.0:
                    self.dot_detector = self._create_dot_detector()
                    self._last_accepted_rvec = None  # einmalig triggern
            except Exception:
                pass

        dots = self.dot_detector.detect(frame, detection)

        # Frühzeitiger Smoothing-Reset bei graduell sinkendem pts.
        # Wenn die Anzahl der validen Correspondences stark unter den
        # bisher gesehenen Maximalwert fällt, ist das ein Zeichen für
        # LK-Drift. reset_smoothing() gibt dem EMA-Smoother Zeit sich
        # neu zu kalibrieren bevor der Totalausfall eintritt.
        # Wird nur im TRACKING-Modus geprüft — nicht beim ersten Warmup.
        if (
            self.mode == TrackerMode.TRACKING
            and self.config.dot_early_reset_pts_ratio > 0.0
            and self._max_pts_seen >= self.config.dot_early_reset_min_pts
        ):
            # pts schätzen: Anzahl der gültigen (non-ambiguous) Cells
            # aus dem aktuellen Dot-Detector-Ergebnis.
            current_pts = sum(
                1 for c in dots.cells
                if c.valid and not c.ambiguous
            )
            threshold = int(
                self._max_pts_seen * self.config.dot_early_reset_pts_ratio
            )
            if current_pts < threshold:
                if hasattr(self.dot_detector, "reset_smoothing"):
                    self.dot_detector.reset_smoothing()

        patches = self.patch_extractor.extract(
            dots,
            self.field.patchSize(),
        )

        decoded = self.patch_decoder.decode(
            patches,
            self.field,
        )

        decoded_valid = [
            p for p in decoded
            if p.valid and not p.ambiguous
        ]

        if not decoded_valid:
            decode_msg = self._decode_failure_message(dots, patches, decoded)
            self._note_decode_topology_failure(dots, patches)
            bootstrap = self._estimate_pose_from_uncoded_grid_bootstrap(
                detection,
                reason=decode_msg,
            )
            if bootstrap is not None:
                return bootstrap
            if self._last_uncoded_bootstrap_reason:
                decode_msg = (
                    f"{decode_msg}; uncoded_bootstrap="
                    f"{self._last_uncoded_bootstrap_reason}"
                )

            fallback = self._estimate_pose_from_persistent_correspondences(
                detection,
                reason=decode_msg,
            )
            if fallback is not None:
                return fallback

            held = self._hold_last_pose_result(
                detection,
                reason=decode_msg,
                correspondence_corners=[],
            )
            if held is not None:
                return held

            return TrackerResult(
                success=False,
                mode=self.mode,
                message=decode_msg + ".",
            )

        self._undecodeable_detection_frames = 0

        corr_result = self.correspondence_builder.build(
            detection,
            decoded_valid,
            self.geometry,
        )

        if not corr_result.valid():
            self._note_low_fresh_correspondence_failure(0)
            fallback = self._estimate_pose_from_persistent_correspondences(
                detection,
                reason="Correspondence build failed",
            )
            if fallback is not None:
                return fallback

            held = self._hold_last_pose_result(
                detection,
                reason="Correspondence build failed",
                correspondence_corners=[],
            )
            if held is not None:
                return held

            return TrackerResult(
                success=False,
                mode=self.mode,
                message="Correspondence build failed.",
            )

        track_points, tracker_corners = self._points_from_correspondences(
            corr_result.correspondences,
        )

        if len(track_points) < self.config.min_points:
            corr_msg = self._correspondence_failure_message(
                len(track_points),
                corr_result,
            )
            self._note_low_fresh_correspondence_failure(len(track_points))
            merged_points, merged_corners = self._merge_with_persistent_correspondences(
                detection,
                track_points,
                tracker_corners,
            )

            if len(merged_points) >= self.config.min_points:
                pose_result = self._estimate_and_package_pose(
                    merged_points,
                    merged_corners,
                    success_message=(
                        f"Pose estimated with merged fresh+persistent correspondences "
                        f"({len(track_points)} fresh, {len(merged_points)} total)."
                    ),
                    update_persistence=False,
                    pose_source=PoseSource.PERSISTENT,
                    detection=detection,
                )
                if pose_result.success:
                    return pose_result

            fallback = self._estimate_pose_from_persistent_correspondences(
                detection,
                reason=corr_msg,
            )
            if fallback is not None:
                return fallback

            held = self._hold_last_pose_result(
                detection,
                reason=corr_msg,
                correspondence_corners=tracker_corners,
            )
            if held is not None:
                return held

            return TrackerResult(
                success=False,
                mode=self.mode,
                message=corr_msg + ".",
                num_points=len(track_points),
                correspondence_corners=tracker_corners,
            )

        self._low_fresh_correspondence_frames = 0

        pose_result = self._estimate_and_package_pose(
            track_points,
            tracker_corners,
            success_message="Pose estimation successful.",
            update_persistence=True,
            pose_source=PoseSource.DECODE,
            detection=detection,
        )

        if pose_result.success:
            return pose_result

        fallback = self._estimate_pose_from_persistent_correspondences(
            detection,
            reason=pose_result.message,
        )
        if fallback is not None:
            return fallback

        return pose_result

    def _estimate_and_package_pose(
        self,
        track_points: List[PoseTrackPoint],
        tracker_corners: List[TrackerCorner],
        success_message: str,
        update_persistence: bool,
        pose_source: PoseSource,
        detection=None,
    ) -> TrackerResult:
        prev_pose_rvec = None if self.pose_tracker.rvec is None else self.pose_tracker.rvec.copy()
        prev_pose_tvec = None if self.pose_tracker.tvec is None else self.pose_tracker.tvec.copy()
        prev_pose_T = (
            None
            if self.pose_tracker.T_marker_camera is None
            else self.pose_tracker.T_marker_camera.copy()
        )
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
        pose = self.pose_tracker.estimate_pose(
            track_points,
            lost_frames=self.lost_frames,
        )
        pnp_ms = (time.perf_counter() - pnp_t0) * 1000.0

        if not pose.success:
            return TrackerResult(
                success=False,
                mode=self.mode,
                message=pose.message,
                rvec=pose.rvec,
                tvec=pose.tvec,
                T_marker_camera=pose.T_marker_camera,
                mean_reprojection_error_px=pose.reprojection_mean_px,
                max_reprojection_error_px=pose.reprojection_max_px,
                num_points=pose.num_points,
                num_inliers=pose.num_inliers,
                pnp_method=str(getattr(pose, "method", "")),
                corners=[],
                correspondence_corners=tracker_corners,
                timings_ms={"pnp_ms": pnp_ms},
            )

        inlier_corners = self._inlier_corners_from_pose(pose, tracker_corners)

        if (
            not update_persistence
            and not self._persistent_pose_motion_plausible(
                pose.rvec,
                pose.tvec,
                prev_last_rvec,
                prev_last_tvec,
            )
        ):
            self.pose_tracker.rvec = prev_pose_rvec
            self.pose_tracker.tvec = prev_pose_tvec
            self.pose_tracker.T_marker_camera = prev_pose_T
            return TrackerResult(
                success=False,
                mode=self.mode,
                message="Persistent pose rejected by motion gate.",
                rvec=pose.rvec,
                tvec=pose.tvec,
                T_marker_camera=pose.T_marker_camera,
                mean_reprojection_error_px=pose.reprojection_mean_px,
                max_reprojection_error_px=pose.reprojection_max_px,
                num_points=pose.num_points,
                num_inliers=pose.num_inliers,
                pnp_method=str(getattr(pose, "method", "")),
                corners=[],
                correspondence_corners=tracker_corners,
                timings_ms={"pnp_ms": pnp_ms},
            )

        if not update_persistence:
            reject_reason = self._fallback_pose_rejection_reason(
                detection,
                pose.rvec,
                pose.tvec,
                pose.reprojection_mean_px,
                pose.reprojection_max_px,
            )
            if reject_reason:
                self.pose_tracker.rvec = prev_pose_rvec
                self.pose_tracker.tvec = prev_pose_tvec
                self.pose_tracker.T_marker_camera = prev_pose_T
                return TrackerResult(
                    success=False,
                    mode=self.mode,
                    message=reject_reason,
                    rvec=pose.rvec,
                    tvec=pose.tvec,
                    T_marker_camera=pose.T_marker_camera,
                    mean_reprojection_error_px=pose.reprojection_mean_px,
                    max_reprojection_error_px=pose.reprojection_max_px,
                    num_points=pose.num_points,
                    num_inliers=pose.num_inliers,
                    pnp_method=str(getattr(pose, "method", "")),
                    corners=[],
                    correspondence_corners=tracker_corners,
                    timings_ms={"pnp_ms": pnp_ms},
                )

        if update_persistence:
            self._store_persistent_correspondences(inlier_corners)

        visual_corners = self._visual_corners_from_pose(
            inlier_corners,
            pose.rvec,
            pose.tvec,
        )
        visual_note = ""
        if len(visual_corners) != len(inlier_corners):
            visual_note = (
                f" Visual corners filtered {len(visual_corners)}/"
                f"{len(inlier_corners)}."
            )
        if not update_persistence and len(visual_corners) < self.config.visual_corner_min_count:
            visual_corners = []
            visual_note += " Visual corners suppressed for fallback pose."

        reliable_pose = (
            update_persistence
            or len(visual_corners) >= self.config.visual_corner_min_count
        )

        # Max-pts und Reprojektionsfehler nur fuer verlaessliche Posen aktualisieren.
        if reliable_pose:
            if pose.num_inliers > self._max_pts_seen:
                self._max_pts_seen = pose.num_inliers
            if pose.reprojection_mean_px >= 0.0:
                self._last_good_reproj_px = pose.reprojection_mean_px
            if pose.rvec is not None:
                self._last_accepted_rvec = np.asarray(pose.rvec, dtype=np.float64).reshape(3, 1)
            if pose.tvec is not None:
                self._last_accepted_tvec = np.asarray(pose.tvec, dtype=np.float64).reshape(3, 1)
            if pose.T_marker_camera is not None:
                self._last_accepted_T_marker_camera = np.asarray(
                    pose.T_marker_camera,
                    dtype=np.float64,
                ).copy()
            self._last_accepted_pose_frame = self.frame_index

        confidence = self._confidence(
            pose.num_inliers,
            pose.reprojection_mean_px,
        )

        return TrackerResult(
            success=True,
            mode=TrackerMode.TRACKING,
            message=success_message + visual_note,
            corners=visual_corners,
            correspondence_corners=tracker_corners,
            rvec=pose.rvec,
            tvec=pose.tvec,
            T_marker_camera=pose.T_marker_camera,
            mean_reprojection_error_px=pose.reprojection_mean_px,
            max_reprojection_error_px=pose.reprojection_max_px,
            num_points=pose.num_points,
            num_inliers=pose.num_inliers,
            confidence=confidence,
            pose_source=pose_source,
            pnp_method=str(getattr(pose, "method", "")),
            timings_ms={"pnp_ms": pnp_ms},
        )

    def _reprojection_errors_for_pose(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> Optional[np.ndarray]:
        try:
            projected, _ = cv2.projectPoints(
                np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                self.K,
                self.dist_coeffs,
            )
        except Exception:
            return None

        projected = projected.reshape(-1, 2)
        return np.linalg.norm(projected - image_points, axis=1)

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

    def _refresh_persistent_correspondences_from_result(
        self,
        result: TrackerResult,
        max_mean_error_px: float,
    ) -> None:
        if (
            result.mean_reprojection_error_px >= 0.0
            and result.mean_reprojection_error_px <= float(max_mean_error_px)
            and len(result.corners) >= self.config.persistence_min_points
        ):
            self._store_persistent_correspondences(result.corners)

    def _set_fast_path_debug(
        self,
        *,
        attempted: bool,
        success: bool = False,
        reason: str = "",
        matches: int = 0,
    ) -> None:
        stats = self._last_persistent_match_stats
        self._last_fast_path_debug = FastPathDebug(
            attempted=bool(attempted),
            success=bool(success),
            reason=str(reason),
            matches=int(matches),
            identities=int(stats.identities),
            current_corners=int(stats.current_corners),
            used_pose_projection=bool(stats.used_pose_projection),
            rejected_no_projection=int(stats.rejected_no_projection),
            rejected_far=int(stats.rejected_far),
            rejected_ambiguous=int(stats.rejected_ambiguous),
            rejected_claimed=int(stats.rejected_claimed),
        )

    def _attach_fast_path_debug(self, result: TrackerResult) -> None:
        result.fast_path_debug = self._last_fast_path_debug

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

    def _try_fast_pose_from_persistent_correspondences(
        self,
        detection,
    ) -> Optional[TrackerResult]:
        self._last_persistent_match_stats = PersistentMatchStats()

        if self.config.decode_only_mode:
            self._set_fast_path_debug(
                attempted=False,
                reason="decode_only_mode",
            )
            return None

        if not self.config.enable_fast_persistent_path:
            self._set_fast_path_debug(
                attempted=False,
                reason="disabled",
            )
            return None

        if detection is None or not bool(detection.valid()):
            self._set_fast_path_debug(
                attempted=False,
                reason="invalid_detection",
            )
            return None

        match_t0 = time.perf_counter()
        points, corners = self._persistent_correspondences_for_detection(detection)
        persistent_match_ms = (time.perf_counter() - match_t0) * 1000.0
        min_points = max(
            int(self.config.min_points),
            int(self.config.persistence_min_points),
            int(self.config.fast_persistent_min_points),
        )
        if len(points) < min_points:
            self._set_fast_path_debug(
                attempted=True,
                reason=f"too_few_matches:{len(points)}<{min_points}",
                matches=len(points),
            )
            return None

        stats = self._last_persistent_match_stats
        prev_pose_rvec = None if self.pose_tracker.rvec is None else self.pose_tracker.rvec.copy()
        prev_pose_tvec = None if self.pose_tracker.tvec is None else self.pose_tracker.tvec.copy()
        prev_pose_T = (
            None
            if self.pose_tracker.T_marker_camera is None
            else self.pose_tracker.T_marker_camera.copy()
        )

        result = self._estimate_and_package_pose(
            points,
            corners,
            success_message=(
                "Fast pose estimated from persistent correspondences "
                f"(matches={len(points)}, identities={stats.identities}, "
                f"far={stats.rejected_far}, ambiguous={stats.rejected_ambiguous}, "
                f"claimed={stats.rejected_claimed})."
            ),
            update_persistence=False,
            pose_source=PoseSource.FAST_PERSISTENT,
            detection=detection,
        )
        result.timings_ms["persistent_match_ms"] = persistent_match_ms

        if not result.success:
            self.pose_tracker.rvec = prev_pose_rvec
            self.pose_tracker.tvec = prev_pose_tvec
            self.pose_tracker.T_marker_camera = prev_pose_T
            self._set_fast_path_debug(
                attempted=True,
                reason=result.message,
                matches=len(points),
            )
            return None

        self._set_fast_path_debug(
            attempted=True,
            success=True,
            reason="ok",
            matches=len(points),
        )

        dense_result = self._try_dense_projection_refine_from_fast_pose(
            detection,
            seed_result=result,
        )
        if dense_result is not None:
            dense_result.timings_ms["persistent_match_ms"] = persistent_match_ms
            dense_result.timings_ms["fast_seed_pnp_ms"] = result.timings_ms.get(
                "pnp_ms",
                0.0,
            )
            result = dense_result

        self._attach_fast_path_debug(result)
        result.confidence *= 0.95
        self._refresh_persistent_correspondences_from_result(
            result,
            max_mean_error_px=self.config.fast_persistent_refresh_mean_error_px,
        )
        return result

    def _try_dense_projection_refine_from_fast_pose(
        self,
        detection,
        seed_result: TrackerResult,
    ) -> Optional[TrackerResult]:
        if not self.config.fast_persistent_dense_refine_enabled:
            self._set_dense_refine_debug(
                attempted=False,
                reason="disabled",
            )
            return None

        if (
            detection is None
            or seed_result.rvec is None
            or seed_result.tvec is None
        ):
            self._set_dense_refine_debug(
                attempted=False,
                reason="missing_seed_pose",
            )
            return None

        match_t0 = time.perf_counter()
        matched_corners, dense_stats = (
            self._strict_projected_tracker_corners_for_detection_pose(
                detection,
                seed_result.rvec,
                seed_result.tvec,
                max_dist_px=self.config.fast_persistent_dense_match_max_px,
                ambiguity_margin_px=(
                    self.config.fast_persistent_dense_min_second_best_margin_px
                ),
            )
        )
        match_ms = (time.perf_counter() - match_t0) * 1000.0
        match_count = len(matched_corners)
        median_err = float(dense_stats.median_error_px)
        p90_err = float(dense_stats.p90_error_px)

        min_dense_points = max(
            int(self.config.fast_persistent_dense_min_points),
            int(seed_result.num_inliers) + 1,
        )
        if match_count < min_dense_points:
            self._set_dense_refine_debug(
                attempted=True,
                reason=f"too_few_matches:{match_count}<{min_dense_points}",
                matches=match_count,
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        if (
            dense_stats.distinct_rows
            < self.config.fast_persistent_dense_min_distinct_rows
            or dense_stats.distinct_cols
            < self.config.fast_persistent_dense_min_distinct_cols
        ):
            self._set_dense_refine_debug(
                attempted=True,
                reason=(
                    "poor_grid_spread:"
                    f"{dense_stats.distinct_rows}x{dense_stats.distinct_cols}"
                ),
                matches=match_count,
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        if (
            dense_stats.object_span_mm
            < self.config.fast_persistent_dense_min_object_span_mm
        ):
            self._set_dense_refine_debug(
                attempted=True,
                reason=f"poor_object_span:{dense_stats.object_span_mm:.1f}mm",
                matches=match_count,
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        min_coverage = float(self.config.fast_persistent_dense_min_image_coverage)
        if 0.0 <= dense_stats.image_coverage < min_coverage:
            self._set_dense_refine_debug(
                attempted=True,
                reason=f"poor_image_coverage:{dense_stats.image_coverage:.3f}",
                matches=match_count,
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        if median_err > self.config.fast_persistent_dense_max_median_px:
            self._set_dense_refine_debug(
                attempted=True,
                reason=f"median_error:{median_err:.3f}",
                matches=match_count,
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        if p90_err > self.config.fast_persistent_dense_max_p90_px:
            self._set_dense_refine_debug(
                attempted=True,
                reason=f"p90_error:{p90_err:.3f}",
                matches=match_count,
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        points, corners = self._points_from_correspondences(matched_corners)
        if len(points) < min_dense_points:
            self._set_dense_refine_debug(
                attempted=True,
                reason=f"too_few_unique_points:{len(points)}<{min_dense_points}",
                matches=len(points),
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        seed_rvec = np.asarray(seed_result.rvec, dtype=np.float64).reshape(3, 1).copy()
        seed_tvec = np.asarray(seed_result.tvec, dtype=np.float64).reshape(3, 1).copy()
        seed_T = (
            None
            if seed_result.T_marker_camera is None
            else np.asarray(seed_result.T_marker_camera, dtype=np.float64).reshape(4, 4).copy()
        )
        seed_last_rvec = (
            None
            if self._last_accepted_rvec is None
            else np.asarray(self._last_accepted_rvec, dtype=np.float64).reshape(3, 1).copy()
        )
        seed_last_tvec = (
            None
            if self._last_accepted_tvec is None
            else np.asarray(self._last_accepted_tvec, dtype=np.float64).reshape(3, 1).copy()
        )
        seed_last_T = (
            None
            if self._last_accepted_T_marker_camera is None
            else np.asarray(self._last_accepted_T_marker_camera, dtype=np.float64).reshape(4, 4).copy()
        )
        seed_last_pose_frame = int(self._last_accepted_pose_frame)
        seed_last_good_reproj_px = float(self._last_good_reproj_px)
        seed_max_pts_seen = int(self._max_pts_seen)

        self.pose_tracker.rvec = seed_rvec.copy()
        self.pose_tracker.tvec = seed_tvec.copy()
        self.pose_tracker.T_marker_camera = None if seed_T is None else seed_T.copy()

        dense_message = (
            "Fast pose estimated from dense projection correspondences "
            f"(seed_matches={seed_result.num_inliers}, "
            f"dense_matches={len(points)}, "
            f"seed_median={median_err:.3f}px, "
            f"seed_p90={p90_err:.3f}px)."
        )
        dense_solver = str(
            self.config.fast_persistent_dense_pose_solver or "direct_prior"
        ).lower()
        if dense_solver in ("sqpnp", "robust_sqpnp", "sqpnp_trim", "robust"):
            dense_result = self._estimate_dense_pose_with_robust_solver(
                points,
                corners,
                success_message=dense_message,
                pose_source=PoseSource.FAST_PERSISTENT,
                detection=detection,
            )
        else:
            dense_result = self._estimate_and_package_pose(
                points,
                corners,
                success_message=dense_message,
                update_persistence=False,
                pose_source=PoseSource.FAST_PERSISTENT,
                detection=detection,
            )
        dense_result.timings_ms["fast_dense_match_ms"] = match_ms

        if (
            not dense_result.success
            or dense_result.num_inliers < min_dense_points
        ):
            self.pose_tracker.rvec = seed_rvec.copy()
            self.pose_tracker.tvec = seed_tvec.copy()
            self.pose_tracker.T_marker_camera = None if seed_T is None else seed_T.copy()
            self._last_accepted_rvec = (
                None if seed_last_rvec is None else seed_last_rvec.copy()
            )
            self._last_accepted_tvec = (
                None if seed_last_tvec is None else seed_last_tvec.copy()
            )
            self._last_accepted_T_marker_camera = (
                None if seed_last_T is None else seed_last_T.copy()
            )
            self._last_accepted_pose_frame = seed_last_pose_frame
            self._last_good_reproj_px = seed_last_good_reproj_px
            self._max_pts_seen = seed_max_pts_seen
            self._set_dense_refine_debug(
                attempted=True,
                reason=(
                    dense_result.message
                    if not dense_result.success
                    else f"too_few_inliers:{dense_result.num_inliers}<{min_dense_points}"
                ),
                matches=len(points),
                median_error_px=median_err,
                p90_error_px=p90_err,
                stats=dense_stats,
            )
            return None

        self._set_dense_refine_debug(
            attempted=True,
            success=True,
            reason="ok",
            matches=len(points),
            median_error_px=median_err,
            p90_error_px=p90_err,
            stats=dense_stats,
        )
        return dense_result

    def _estimate_pose_from_persistent_correspondences(
        self,
        detection,
        reason: str,
    ) -> Optional[TrackerResult]:
        if self.config.decode_only_mode:
            return None

        if not self.config.enable_temporal_correspondence_persistence:
            return None

        points, corners = self._persistent_correspondences_for_detection(detection)

        if len(points) < self.config.persistence_min_points:
            return None

        if (
            "No valid decoded patches" in reason
            and len(points) < self.config.persistence_min_points_after_decode_fail
        ):
            return None

        result = self._estimate_and_package_pose(
            points,
            corners,
            success_message=(
                f"Pose estimated from persistent correspondences after: {reason}."
            ),
            update_persistence=False,
            pose_source=PoseSource.PERSISTENT,
            detection=detection,
        )

        if result.success:
            result.confidence *= 0.85

            # If the persistent-fallback pose is good, refresh the persistent
            # state so the tracker doesn't run out of time budget
            # (persistence_max_frames) while the main decode is warming up.
            self._refresh_persistent_correspondences_from_result(
                result,
                max_mean_error_px=self.config.persistence_refresh_mean_error_px,
            )

            return result

        return None

    def _estimate_pose_from_uncoded_grid_bootstrap(
        self,
        detection,
        reason: str,
    ) -> Optional[TrackerResult]:
        self._last_uncoded_bootstrap_reason = ""
        if self.config.decode_only_mode:
            self._last_uncoded_bootstrap_reason = "decode_only_mode"
            return None

        if not self.config.enable_uncoded_grid_bootstrap:
            self._last_uncoded_bootstrap_reason = "disabled"
            return None

        current = self._detected_corners_from_detection(detection)
        if len(current) < self.config.uncoded_bootstrap_min_corners:
            self._last_uncoded_bootstrap_reason = f"too_few_corners:{len(current)}"
            return None

        if self._last_accepted_rvec is not None and self._last_accepted_tvec is not None:
            self._last_uncoded_bootstrap_reason = "pose_history_exists"
            return None

        local_rows = [int(c.local_row) for c in current]
        local_cols = [int(c.local_col) for c in current]
        rows = self.geometry.corner_rows()
        cols = self.geometry.corner_cols()

        min_row_off = -min(local_rows)
        max_row_off = rows - 1 - max(local_rows)
        min_col_off = -min(local_cols)
        max_col_off = cols - 1 - max(local_cols)

        candidates = []
        for row_off in range(min_row_off, max_row_off + 1):
            for col_off in range(min_col_off, max_col_off + 1):
                points: List[PoseTrackPoint] = []
                corners: List[TrackerCorner] = []

                for corner in current:
                    gr = int(corner.local_row) + row_off
                    gc = int(corner.local_col) + col_off
                    if not self.geometry.has_corner(gr, gc):
                        continue

                    pt = self.geometry.corner_point(gr, gc)
                    xyz = (float(pt.x), float(pt.y), float(pt.z))
                    uv = (float(corner.uv[0]), float(corner.uv[1]))
                    points.append(
                        PoseTrackPoint(
                            global_row=gr,
                            global_col=gc,
                            xyz_mm=xyz,
                            uv=uv,
                            votes=0,
                        )
                    )
                    corners.append(
                        TrackerCorner(
                            local_row=int(corner.local_row),
                            local_col=int(corner.local_col),
                            global_row=gr,
                            global_col=gc,
                            xyz_mm=xyz,
                            uv=uv,
                            votes=0,
                        )
                    )

                if len(points) < self.config.uncoded_bootstrap_min_corners:
                    continue

                candidate = self._solve_uncoded_bootstrap_candidate(points, corners)
                if candidate is not None:
                    candidates.append((candidate, row_off, col_off))

        if not candidates:
            self._last_uncoded_bootstrap_reason = "no_valid_candidates"
            return None

        candidates.sort(key=lambda x: (x[0].mean_reprojection_error_px, x[0].max_reprojection_error_px))
        best, row_off, col_off = candidates[0]
        second_mean = (
            candidates[1][0].mean_reprojection_error_px
            if len(candidates) > 1
            else float("inf")
        )

        if best.mean_reprojection_error_px > self.config.uncoded_bootstrap_max_mean_reprojection_error_px:
            self._last_uncoded_bootstrap_reason = (
                f"mean_error:{best.mean_reprojection_error_px:.3f}"
            )
            return None

        if best.max_reprojection_error_px > self.config.uncoded_bootstrap_max_max_reprojection_error_px:
            self._last_uncoded_bootstrap_reason = (
                f"max_error:{best.max_reprojection_error_px:.3f}"
            )
            return None

        if (
            np.isfinite(second_mean)
            and (second_mean - best.mean_reprojection_error_px)
            < self.config.uncoded_bootstrap_min_second_best_margin_px
        ):
            self._last_uncoded_bootstrap_reason = (
                f"ambiguous:best={best.mean_reprojection_error_px:.3f},"
                f"second={second_mean:.3f}"
            )
            return None

        best.message = (
            "Pose estimated from uncoded grid bootstrap after: "
            f"{reason} (offset={row_off},{col_off}, "
            f"second_mean={second_mean:.3f})."
        )
        best.confidence *= 0.55
        self.pose_tracker.rvec = best.rvec.copy()
        self.pose_tracker.tvec = best.tvec.copy()
        self.pose_tracker.T_marker_camera = (
            None
            if best.T_marker_camera is None
            else best.T_marker_camera.copy()
        )
        self._last_good_reproj_px = best.mean_reprojection_error_px
        self._last_accepted_rvec = best.rvec.copy()
        self._last_accepted_tvec = best.tvec.copy()
        self._last_accepted_T_marker_camera = (
            None
            if best.T_marker_camera is None
            else best.T_marker_camera.copy()
        )
        self._last_accepted_pose_frame = self.frame_index
        self._store_persistent_correspondences(best.corners)
        return best

    def _solve_uncoded_bootstrap_candidate(
        self,
        points: List[PoseTrackPoint],
        corners: List[TrackerCorner],
    ) -> Optional[TrackerResult]:
        object_points = np.asarray([p.xyz_mm for p in points], dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray([p.uv for p in points], dtype=np.float64).reshape(-1, 2)

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                self.K,
                self.dist_coeffs,
                iterationsCount=int(self.config.pnp_ransac_iterations),
                reprojectionError=float(self.config.pnp_ransac_reprojection_px),
                confidence=float(self.config.pnp_ransac_confidence),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            return None

        if not success or inliers is None or len(inliers) < self.config.min_inliers:
            return None

        inlier_idx = np.asarray(inliers, dtype=np.int64).reshape(-1)
        object_inliers = object_points[inlier_idx]
        image_inliers = image_points[inlier_idx]

        try:
            projected, _ = cv2.projectPoints(
                object_inliers,
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                self.K,
                self.dist_coeffs,
            )
        except Exception:
            return None

        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - image_inliers, axis=1)
        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))

        inlier_corners = [
            corners[int(i)]
            for i in inlier_idx
            if 0 <= int(i) < len(corners)
        ]
        visual_corners = self._visual_corners_from_pose(inlier_corners, rvec, tvec)
        if len(visual_corners) < self.config.visual_corner_min_count:
            return None

        T = self.pose_tracker.T_marker_camera
        try:
            from tracking.hydramarker.map_pose_tracker import make_transform_from_rvec_tvec
            T = make_transform_from_rvec_tvec(rvec, tvec)
        except Exception:
            T = None

        rvec_arr = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec_arr = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

        T_arr = None if T is None else np.asarray(T, dtype=np.float64).reshape(4, 4)

        confidence = self._confidence(len(visual_corners), mean_err) * 0.5
        return TrackerResult(
            success=True,
            mode=TrackerMode.TRACKING,
            message="Pose estimated from uncoded grid bootstrap.",
            corners=visual_corners,
            correspondence_corners=inlier_corners,
            rvec=rvec_arr,
            tvec=tvec_arr,
            T_marker_camera=T_arr,
            mean_reprojection_error_px=mean_err,
            max_reprojection_error_px=max_err,
            num_points=len(points),
            num_inliers=len(inlier_corners),
            confidence=confidence,
            pose_source=PoseSource.UNCODED_GRID,
        )

    def _persistent_pose_motion_plausible(
        self,
        rvec: Optional[np.ndarray],
        tvec: Optional[np.ndarray],
        prev_rvec: Optional[np.ndarray],
        prev_tvec: Optional[np.ndarray],
    ) -> bool:
        if rvec is None or tvec is None:
            return False

        if prev_rvec is None or prev_tvec is None:
            return True

        try:
            R_prev, _ = cv2.Rodrigues(
                np.asarray(prev_rvec, dtype=np.float64).reshape(3, 1)
            )
            R_curr, _ = cv2.Rodrigues(
                np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            )
            dR = R_curr @ R_prev.T
            cos_a = float(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
            rot_delta_deg = float(np.degrees(np.arccos(cos_a)))

            t_prev = np.asarray(prev_tvec, dtype=np.float64).reshape(3, 1)
            t_curr = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            trans_delta_mm = float(np.linalg.norm(t_curr - t_prev))
        except Exception:
            return False

        return (
            rot_delta_deg <= self.config.persistence_max_rotation_jump_deg
            and trans_delta_mm <= self.config.persistence_max_translation_jump_mm
        )

    def _detection_has_decodeable_cell_span(self, detection) -> bool:
        cells = list(getattr(detection, "cells", []))
        if not cells:
            return False

        min_span = max(1, int(self.config.checker_min_tracking_decode_cell_span))
        rows = [int(getattr(c, "j", getattr(c, "row", 0))) for c in cells]
        cols = [int(getattr(c, "i", getattr(c, "col", 0))) for c in cells]
        row_span = max(rows) - min(rows) + 1 if rows else 0
        col_span = max(cols) - min(cols) + 1 if cols else 0
        return row_span >= min_span and col_span >= min_span

    def _force_local_recovery(self) -> None:
        self.checkerboard_detector.reset_tracking()
        self.dot_detector = self._create_dot_detector()
        self._clear_persistent_correspondences()
        self._undecodeable_detection_frames = 0
        self._pose_propagation_block_until_frame = max(
            self._pose_propagation_block_until_frame,
            self.frame_index + 5,
        )

    def _note_low_fresh_correspondence_failure(self, fresh_count: int) -> None:
        if fresh_count >= self.config.checker_min_fresh_correspondences_for_stable_tracking:
            self._low_fresh_correspondence_frames = 0
            return

        self._low_fresh_correspondence_frames += 1
        if (
            self._low_fresh_correspondence_frames
            > self.config.checker_max_low_fresh_correspondence_frames
        ):
            self._force_local_recovery()

    def _hold_last_pose_result(
        self,
        detection,
        reason: str,
        correspondence_corners: List[TrackerCorner],
    ) -> Optional[TrackerResult]:
        if self.config.decode_only_mode:
            return None

        if self.pose_tracker.rvec is None or self.pose_tracker.tvec is None:
            return None

        if (
            self._low_fresh_correspondence_frames > self.config.pose_hold_max_frames
            and self.config.pose_hold_max_frames >= 0
        ):
            return None

        if detection is None or not bool(detection.valid()):
            return None

        detected_count = len(self._detected_corners_from_detection(detection))
        if detected_count < self.config.pose_hold_min_detection_corners:
            return None

        rvec = np.asarray(self.pose_tracker.rvec, dtype=np.float64).reshape(3, 1).copy()
        tvec = np.asarray(self.pose_tracker.tvec, dtype=np.float64).reshape(3, 1).copy()
        T = (
            None
            if self.pose_tracker.T_marker_camera is None
            else self.pose_tracker.T_marker_camera.copy()
        )

        held_corners, match_count, median_err, p90_err = (
            self._projected_tracker_corners_for_detection_pose(
                detection,
                rvec,
                tvec,
                max_dist_px=self.config.visual_corner_max_reprojection_error_px,
            )
        )

        if (
            match_count < self.config.visual_corner_min_count
            or median_err > self.config.visual_corner_max_reprojection_error_px
            or p90_err > self.config.visual_corner_max_reprojection_error_px
        ):
            return None

        return TrackerResult(
            success=True,
            mode=TrackerMode.TRACKING,
            message=(
                f"Pose held from last accepted pose after: {reason} "
                f"(blue_align={match_count}, median={median_err:.2f}px, "
                f"p90={p90_err:.2f}px)."
            ),
            corners=held_corners,
            correspondence_corners=correspondence_corners,
            rvec=rvec,
            tvec=tvec,
            T_marker_camera=T,
            mean_reprojection_error_px=self._last_good_reproj_px,
            max_reprojection_error_px=-1.0,
            num_points=max(len(held_corners), 0),
            num_inliers=max(len(held_corners), 0),
            confidence=0.25,
            pose_source=PoseSource.HOLD,
        )

    def _hold_last_pose_without_detection_result(self, detection) -> Optional[TrackerResult]:
        if self.config.decode_only_mode:
            return None

        if self.pose_tracker.rvec is None or self.pose_tracker.tvec is None:
            return None

        if (
            self._last_good_reproj_px < 0.0
            or self._last_good_reproj_px
            > self.config.fallback_pose_max_mean_reprojection_error_px
        ):
            return None

        rvec = np.asarray(self.pose_tracker.rvec, dtype=np.float64).reshape(3, 1).copy()
        tvec = np.asarray(self.pose_tracker.tvec, dtype=np.float64).reshape(3, 1).copy()
        T = (
            None
            if self.pose_tracker.T_marker_camera is None
            else self.pose_tracker.T_marker_camera.copy()
        )

        return TrackerResult(
            success=True,
            mode=self.mode,
            message=(
                "Pose held from last accepted pose without checkerboard detection."
            ),
            detection_valid=False,
            detection_tracking=False if detection is None else bool(detection.tracking),
            detection_stable=False if detection is None else bool(detection.stable),
            detection_corners=self._detected_corners_from_detection(detection),
            corners=[],
            correspondence_corners=[],
            rvec=rvec,
            tvec=tvec,
            T_marker_camera=T,
            mean_reprojection_error_px=self._last_good_reproj_px,
            max_reprojection_error_px=-1.0,
            num_points=0,
            num_inliers=0,
            confidence=0.10,
            pose_source=PoseSource.HOLD,
        )

    def _emergency_last_pose_result(
        self,
        detection,
        reason: str,
    ) -> Optional[TrackerResult]:
        if self.config.decode_only_mode:
            return None

        if not self.config.emergency_pose_hold_enabled:
            return None

        if self._last_accepted_rvec is None or self._last_accepted_tvec is None:
            return None

        age = self.frame_index - self._last_accepted_pose_frame
        if age < 0:
            return None

        max_age = int(self.config.emergency_pose_hold_max_frames)
        if max_age >= 0 and age > max_age:
            return None

        rvec = np.asarray(self._last_accepted_rvec, dtype=np.float64).reshape(3, 1).copy()
        tvec = np.asarray(self._last_accepted_tvec, dtype=np.float64).reshape(3, 1).copy()
        T = (
            None
            if self._last_accepted_T_marker_camera is None
            else self._last_accepted_T_marker_camera.copy()
        )

        self.pose_tracker.rvec = rvec.copy()
        self.pose_tracker.tvec = tvec.copy()
        self.pose_tracker.T_marker_camera = None if T is None else T.copy()

        held_corners: List[TrackerCorner] = []
        align_msg = "no_blue_alignment"
        if detection is not None and bool(detection.valid()):
            corners, match_count, median_err, p90_err = (
                self._projected_tracker_corners_for_detection_pose(
                    detection,
                    rvec,
                    tvec,
                    max_dist_px=self.config.visual_corner_max_reprojection_error_px,
                )
            )
            if (
                match_count >= self.config.visual_corner_min_count
                and median_err <= self.config.visual_corner_max_reprojection_error_px
                and p90_err <= self.config.visual_corner_max_reprojection_error_px
            ):
                held_corners = corners
                align_msg = (
                    f"blue_align={match_count}, median={median_err:.2f}px, "
                    f"p90={p90_err:.2f}px"
                )

        confidence = max(0.03, 0.20 * (0.96 ** max(age, 0)))

        return TrackerResult(
            success=True,
            mode=self.mode,
            message=(
                f"Emergency pose held from last accepted pose after: {reason} "
                f"(age={age}, {align_msg})."
            ),
            detection_valid=False if detection is None else bool(detection.valid()),
            detection_tracking=False if detection is None else bool(detection.tracking),
            detection_stable=False if detection is None else bool(detection.stable),
            detection_corners=self._detected_corners_from_detection(detection),
            corners=held_corners,
            correspondence_corners=[],
            rvec=rvec,
            tvec=tvec,
            T_marker_camera=T,
            mean_reprojection_error_px=self._last_good_reproj_px,
            max_reprojection_error_px=-1.0,
            num_points=len(held_corners),
            num_inliers=len(held_corners),
            confidence=confidence,
            pose_source=PoseSource.HOLD,
        )

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

    def _note_decode_topology_failure(self, dots, patches) -> None:
        if len(patches) > 0:
            self._undecodeable_detection_frames = 0
            return

        dot_cells = list(getattr(dots, "cells", []))
        if not dot_cells:
            self._undecodeable_detection_frames = 0
            return

        min_span = max(1, int(self.config.checker_min_tracking_decode_cell_span))
        rows = [int(getattr(c, "row", 0)) for c in dot_cells]
        cols = [int(getattr(c, "col", 0)) for c in dot_cells]
        row_span = max(rows) - min(rows) + 1 if rows else 0
        col_span = max(cols) - min(cols) + 1 if cols else 0

        if row_span >= min_span and col_span >= min_span:
            self._undecodeable_detection_frames = 0
            return

        self._undecodeable_detection_frames += 1
        if (
            self._undecodeable_detection_frames
            > self.config.checker_max_undecodeable_tracking_frames
        ):
            self._force_local_recovery()

    @staticmethod
    def _correspondence_failure_message(num_points: int, corr_result) -> str:
        return (
            f"Too few correspondences: {num_points} "
            f"(patches_used={int(getattr(corr_result, 'decoded_patches_used', 0))}, "
            f"rot_rejected={int(getattr(corr_result, 'decoded_patches_rejected_by_rotation', 0))}, "
            f"assign_total={int(getattr(corr_result, 'assignments_total', 0))}, "
            f"assign_accepted={int(getattr(corr_result, 'assignments_accepted', 0))}, "
            f"conflicted={int(getattr(corr_result, 'assignments_conflicted', 0))}, "
            f"no_geom={int(getattr(corr_result, 'corners_without_geometry', 0))}, "
            f"single_boundary={int(getattr(corr_result, 'single_vote_boundary_corners_accepted', 0))}, "
            f"single_non_boundary_rej={int(getattr(corr_result, 'single_vote_non_boundary_corners_rejected', 0))}, "
            f"rot={int(getattr(corr_result, 'dominant_rotation_deg', -1))}/"
            f"{int(getattr(corr_result, 'dominant_rotation_count', 0))}/"
            f"{int(getattr(corr_result, 'rotation_vote_count', 0))})"
        )

    @staticmethod
    def _decode_failure_message(dots, patches, decoded) -> str:
        dot_cells = list(getattr(dots, "cells", []))
        dot_cell_count = len(dot_cells)
        dot_valid_count = sum(
            1 for c in dot_cells
            if bool(getattr(c, "valid", False))
        )
        dot_ambiguous_count = sum(
            1 for c in dot_cells
            if bool(getattr(c, "ambiguous", False))
        )
        dot_cache_reused_count = sum(
            1 for c in dot_cells
            if bool(getattr(c, "cache_reused", False))
        )
        dot_rows = int(getattr(dots, "rows", 0))
        dot_cols = int(getattr(dots, "cols", 0))

        patch_count = len(patches)
        decoded_count = len(decoded)
        invalid_geometry = sum(
            1 for p in decoded
            if getattr(p, "local", None) is not None
            and getattr(p.local, "valid", False)
            and not getattr(p.local, "geometry_valid", False)
        )
        ambiguous = sum(
            1 for p in decoded
            if getattr(p, "ambiguous", False)
        )
        matched_but_rejected = sum(
            1 for p in decoded
            if int(getattr(p, "num_matches", 0)) > 0 and not getattr(p, "valid", False)
        )

        return (
            "No valid decoded patches "
            f"(cells={dot_cell_count}, valid_cells={dot_valid_count}, "
            f"ambig_cells={dot_ambiguous_count}, cached_cells={dot_cache_reused_count}, "
            f"grid={dot_rows}x{dot_cols}, patches={patch_count}, decoded={decoded_count}, "
            f"bad_geom={invalid_geometry}, ambiguous={ambiguous}, "
            f"matched_rejected={matched_but_rejected})"
        )

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

    def _log_result(self, stage: str, result: TrackerResult) -> None:
        if self.logger is None:
            return

        self.logger.log_tracker_result(
            stage,
            self.frame_index,
            result,
            decode_only=self.config.decode_only_mode,
            lost_frames=self.lost_frames,
            persisted_count=len(self._persistent_corners),
        )

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
