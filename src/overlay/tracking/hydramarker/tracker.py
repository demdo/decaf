from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import hydramarker_cpp as hm

from overlay.tracking.pose_solvers import (
    make_transform_from_rvec_tvec,
    solve_pose,
)
from overlay.tracking.hydramarker.tracker_consistency import (
    TrackerConsistency,
    TrackerConsistencyConfig,
)


class TrackerMode(str, Enum):
    LOST = "LOST"
    DETECTING = "DETECTING"
    TRACKING = "TRACKING"
    RECOVERING = "RECOVERING"


@dataclass
class GlobalCornerIdentity:
    local_row: int
    local_col: int

    global_row: int
    global_col: int

    xyz_mm: Tuple[float, float, float]
    uv: Tuple[float, float]

    votes: int = 0


@dataclass
class TrackerResult:
    success: bool
    mode: TrackerMode
    message: str = ""

    detection_valid: bool = False
    detection_tracking: bool = False
    detection_stable: bool = False

    corners: List[GlobalCornerIdentity] = field(default_factory=list)

    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    T_marker_camera: Optional[np.ndarray] = None

    mean_reprojection_error_px: float = -1.0
    max_reprojection_error_px: float = -1.0

    num_points: int = 0
    num_inliers: int = 0
    confidence: float = 0.0


@dataclass
class TrackerConfig:
    # Pose requirements
    min_points: int = 8
    min_inliers: int = 6

    max_mean_reprojection_error_px: float = 4.0
    max_max_reprojection_error_px: float = 15.0

    # State / fallback
    max_lost_frames: int = 8
    max_identity_match_px: float = 18.0
    max_translation_jump_mm: float = 120.0
    max_rotation_jump_deg: float = 45.0

    # OpenCV PnP
    pnp_ransac_iterations: int = 500
    pnp_ransac_reprojection_px: float = 3.0
    pnp_ransac_confidence: float = 0.99

    # DotDetector config from working tests
    dot_canonical_size: int = 80
    dot_canonical_margin_px: float = 4.0
    dot_min_dot_contrast: float = 8.0
    dot_strong_dot_contrast: float = 35.0
    dot_commit_threshold: float = 0.45
    dot_revoke_threshold: float = 0.20
    dot_uncertainty_low: float = 0.40
    dot_uncertainty_high: float = 0.55
    dot_warmup_frames: int = 1

    # PatchDecoder config from working tests
    decoder_require_geometry_valid: bool = True
    decoder_accept_ambiguous: bool = False

    # CorrespondenceBuilder config from working tests
    corr_min_votes: int = 1
    corr_discard_conflicts: bool = True
    corr_require_detection_stable: bool = False

    # Drawing
    draw_axes_length_mm: float = 30.0


class HydraTracker:
    """
    HydraMarker tracker.

    C++ backend:
        CheckerboardDetector
        DotDetector
        PatchExtractor
        PatchDecoder
        CorrespondenceBuilder

    Python backend:
        pose_solvers.solve_pose
        tracker_consistency.TrackerConsistency

    The tracker localizes the global marker origin through decoded local patches.
    """

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

        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((0, 1), dtype=np.float64)
        else:
            self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

        self.field = hm.MarkerField.loadFromFile(field_path)
        self.geometry = hm.MarkerGeometry.load_from_json(marker_json_path)

        self.checkerboard_detector = hm.CheckerboardDetector()
        self.dot_detector = self._create_dot_detector()
        self.patch_extractor = hm.PatchExtractor()
        self.patch_decoder = self._create_patch_decoder()
        self.correspondence_builder = self._create_correspondence_builder()

        self.consistency = TrackerConsistency(
            TrackerConsistencyConfig(
                max_translation_jump_mm=self.config.max_translation_jump_mm,
                max_rotation_jump_deg=self.config.max_rotation_jump_deg,
            )
        )

        self.mode = TrackerMode.LOST

        self.identities: List[GlobalCornerIdentity] = []
        self.identity_by_local: Dict[Tuple[int, int], GlobalCornerIdentity] = {}

        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None
        self.T_marker_camera: Optional[np.ndarray] = None

        self.frame_index = 0
        self.lost_frames = 0

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

        return hm.CorrespondenceBuilder(cfg)

    def reset(self) -> None:
        self.mode = TrackerMode.LOST

        self.identities.clear()
        self.identity_by_local.clear()

        self.rvec = None
        self.tvec = None
        self.T_marker_camera = None

        self.frame_index = 0
        self.lost_frames = 0

        self.checkerboard_detector.reset_tracking()
        self.dot_detector = self._create_dot_detector()

    def process_frame(self, frame: np.ndarray) -> TrackerResult:
        self.frame_index += 1

        detection = self.checkerboard_detector.detect(frame)

        if detection is None or not detection.valid():
            return self._lost_result(
                message="No valid checkerboard detection.",
                detection=None,
            )

        result = self._full_decode_pose(frame, detection)

        if result.success:
            self.mode = TrackerMode.TRACKING
            self.lost_frames = 0
            return result

        if self.identities:
            fallback = self._fallback_pose_from_known_identities(detection)

            if fallback.success:
                self.mode = TrackerMode.TRACKING
                self.lost_frames = 0
                return fallback

        self.lost_frames += 1

        if self.lost_frames > self.config.max_lost_frames:
            self._clear_pose_and_identities()
            self.mode = TrackerMode.LOST
        else:
            self.mode = TrackerMode.RECOVERING if self.identities else TrackerMode.DETECTING

        result.mode = self.mode
        return result

    def _lost_result(self, message: str, detection) -> TrackerResult:
        self.lost_frames += 1

        if self.lost_frames > self.config.max_lost_frames:
            self._clear_pose_and_identities()
            self.mode = TrackerMode.LOST
        else:
            self.mode = TrackerMode.RECOVERING if self.identities else TrackerMode.LOST

        return TrackerResult(
            success=False,
            mode=self.mode,
            message=message,
            detection_valid=False,
            detection_tracking=False if detection is None else bool(detection.tracking),
            detection_stable=False if detection is None else bool(detection.stable),
            corners=list(self.identities),
            rvec=None if self.rvec is None else self.rvec.copy(),
            tvec=None if self.tvec is None else self.tvec.copy(),
            T_marker_camera=None if self.T_marker_camera is None else self.T_marker_camera.copy(),
        )

    def _clear_pose_and_identities(self) -> None:
        self.identities.clear()
        self.identity_by_local.clear()

        self.rvec = None
        self.tvec = None
        self.T_marker_camera = None

    def _full_decode_pose(self, frame: np.ndarray, detection) -> TrackerResult:
        dots = self.dot_detector.detect(frame, detection)

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
            return TrackerResult(
                success=False,
                mode=TrackerMode.RECOVERING if self.identities else TrackerMode.DETECTING,
                message="No valid decoded patches.",
                detection_valid=True,
                detection_tracking=bool(detection.tracking),
                detection_stable=bool(detection.stable),
                corners=list(self.identities),
                num_points=0,
                num_inliers=0,
            )

        corr_result = self.correspondence_builder.build(
            detection,
            decoded_valid,
            self.geometry,
        )

        if not corr_result.valid():
            return TrackerResult(
                success=False,
                mode=TrackerMode.RECOVERING if self.identities else TrackerMode.DETECTING,
                message="Correspondence build failed.",
                detection_valid=True,
                detection_tracking=bool(detection.tracking),
                detection_stable=bool(detection.stable),
                corners=list(self.identities),
                num_points=0,
                num_inliers=0,
            )

        correspondences = list(corr_result.correspondences)

        identities = self._identities_from_correspondences(
            correspondences,
            [],
        )

        if len(identities) < self.config.min_points:
            return TrackerResult(
                success=False,
                mode=TrackerMode.RECOVERING if self.identities else TrackerMode.DETECTING,
                message=f"Too few correspondences: {len(identities)}.",
                detection_valid=True,
                detection_tracking=bool(detection.tracking),
                detection_stable=bool(detection.stable),
                corners=identities,
                num_points=len(identities),
                num_inliers=0,
            )

        pose = self._estimate_pose_from_identities(
            identities,
            use_previous_guess=False,
        )

        pose.detection_valid = True
        pose.detection_tracking = bool(detection.tracking)
        pose.detection_stable = bool(detection.stable)

        if not pose.success:
            pose.mode = TrackerMode.RECOVERING if self.identities else TrackerMode.DETECTING
            return pose

        consistency = self.consistency.validate_pose_jump(
            previous_rvec=self.rvec,
            previous_tvec=self.tvec,
            candidate_rvec=pose.rvec,
            candidate_tvec=pose.tvec,
        )

        if not consistency.accepted:
            pose.success = False
            pose.mode = TrackerMode.RECOVERING
            pose.message = f"Full decode rejected by consistency gate: {consistency.reason}"
            return pose

        self._store_pose(
            pose.rvec,
            pose.tvec,
            pose.T_marker_camera,
        )

        self._replace_identities(pose.corners)

        pose.mode = TrackerMode.TRACKING
        pose.message = "Full decode + pose solver successful."
        pose.rvec = self.rvec.copy()
        pose.tvec = self.tvec.copy()
        pose.T_marker_camera = self.T_marker_camera.copy()

        return pose

    def _fallback_pose_from_known_identities(self, detection) -> TrackerResult:
        current = self._known_identities_for_detection(detection)

        if len(current) < self.config.min_points:
            return TrackerResult(
                success=False,
                mode=TrackerMode.RECOVERING,
                message="Too few known identities for fallback pose.",
                detection_valid=True,
                detection_tracking=bool(detection.tracking),
                detection_stable=bool(detection.stable),
                corners=current,
                num_points=len(current),
                num_inliers=0,
            )

        pose = self._estimate_pose_from_identities(
            current,
            use_previous_guess=True,
        )

        pose.detection_valid = True
        pose.detection_tracking = bool(detection.tracking)
        pose.detection_stable = bool(detection.stable)

        if not pose.success:
            pose.mode = TrackerMode.RECOVERING
            return pose

        consistency = self.consistency.validate_pose_jump(
            previous_rvec=self.rvec,
            previous_tvec=self.tvec,
            candidate_rvec=pose.rvec,
            candidate_tvec=pose.tvec,
        )

        if not consistency.accepted:
            pose.success = False
            pose.mode = TrackerMode.RECOVERING
            pose.message = f"Fallback rejected by consistency gate: {consistency.reason}"
            return pose

        self._store_pose(
            pose.rvec,
            pose.tvec,
            pose.T_marker_camera,
        )

        self._replace_identities(pose.corners)

        pose.mode = TrackerMode.TRACKING
        pose.message = "Fallback pose from known global identities."
        pose.rvec = self.rvec.copy()
        pose.tvec = self.tvec.copy()
        pose.T_marker_camera = self.T_marker_camera.copy()

        return pose

    def _estimate_pose_from_identities(
        self,
        identities: List[GlobalCornerIdentity],
        use_previous_guess: bool,
    ) -> TrackerResult:
        object_points = np.asarray(
            [p.xyz_mm for p in identities],
            dtype=np.float64,
        ).reshape(-1, 3)

        image_points = np.asarray(
            [p.uv for p in identities],
            dtype=np.float64,
        ).reshape(-1, 2)

        if len(identities) < self.config.min_points:
            return TrackerResult(
                success=False,
                mode=TrackerMode.RECOVERING,
                message="Too few points for PnP.",
                corners=identities,
                num_points=len(identities),
                num_inliers=0,
            )

        try:
            if use_previous_guess and self.rvec is not None and self.tvec is not None:
                raw = solve_pose(
                    object_points_xyz=object_points,
                    image_points_uv=image_points,
                    K=self.K,
                    dist_coeffs=self.dist_coeffs,
                    pose_method="iterative",
                    rvec_init=self.rvec,
                    tvec_init=self.tvec,
                    use_extrinsic_guess=True,
                    refine_with_iterative=True,
                )
            else:
                raw = solve_pose(
                    object_points_xyz=object_points,
                    image_points_uv=image_points,
                    K=self.K,
                    dist_coeffs=self.dist_coeffs,
                    pose_method="iterative_ransac",
                    refine_with_iterative=True,
                    ransac_reprojection_error_px=self.config.pnp_ransac_reprojection_px,
                    ransac_confidence=self.config.pnp_ransac_confidence,
                    ransac_iterations_count=self.config.pnp_ransac_iterations,
                )

        except Exception as e:
            return TrackerResult(
                success=False,
                mode=TrackerMode.RECOVERING,
                message=f"Pose solver exception: {e}",
                corners=identities,
                num_points=len(identities),
                num_inliers=0,
            )

        inlier_indices = np.asarray(raw.inlier_idx, dtype=np.int64).reshape(-1)

        if len(inlier_indices) < self.config.min_inliers:
            return TrackerResult(
                success=False,
                mode=TrackerMode.RECOVERING,
                message="Too few pose inliers.",
                corners=identities,
                rvec=np.asarray(raw.rvec, dtype=np.float64).reshape(3, 1),
                tvec=np.asarray(raw.tvec, dtype=np.float64).reshape(3, 1),
                T_marker_camera=make_transform_from_rvec_tvec(raw.rvec, raw.tvec),
                num_points=len(identities),
                num_inliers=int(len(inlier_indices)),
                mean_reprojection_error_px=float(raw.reproj_mean_px),
                max_reprojection_error_px=float(raw.reproj_max_px),
            )

        selected_identities = [
            identities[int(i)]
            for i in inlier_indices
            if 0 <= int(i) < len(identities)
        ]

        mean_err = float(raw.reproj_mean_px)
        max_err = float(raw.reproj_max_px)

        success = (
            mean_err <= self.config.max_mean_reprojection_error_px
            and max_err <= self.config.max_max_reprojection_error_px
        )

        if not success:
            msg = (
                "Mean reprojection error too high."
                if mean_err > self.config.max_mean_reprojection_error_px
                else "Max reprojection error too high."
            )
        else:
            msg = "Pose solver successful."

        rvec = np.asarray(raw.rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(raw.tvec, dtype=np.float64).reshape(3, 1)
        T = make_transform_from_rvec_tvec(rvec, tvec)

        return TrackerResult(
            success=success,
            mode=TrackerMode.TRACKING if success else TrackerMode.RECOVERING,
            message=msg,
            corners=selected_identities,
            rvec=rvec,
            tvec=tvec,
            T_marker_camera=T,
            mean_reprojection_error_px=mean_err,
            max_reprojection_error_px=max_err,
            num_points=len(identities),
            num_inliers=int(len(inlier_indices)),
            confidence=self._confidence(len(inlier_indices), mean_err),
        )

    def _known_identities_for_detection(self, detection) -> List[GlobalCornerIdentity]:
        result: List[GlobalCornerIdentity] = []
        used_globals: set[Tuple[int, int]] = set()
        used_corner_indices: set[int] = set()

        for idx, c in enumerate(detection.corners):
            key = (int(c.i), int(c.j))
            old = self.identity_by_local.get(key)

            if old is None:
                continue

            gkey = (old.global_row, old.global_col)

            if gkey in used_globals:
                continue

            result.append(
                GlobalCornerIdentity(
                    local_row=int(c.i),
                    local_col=int(c.j),
                    global_row=old.global_row,
                    global_col=old.global_col,
                    xyz_mm=old.xyz_mm,
                    uv=self._point2(c.uv),
                    votes=old.votes,
                )
            )

            used_globals.add(gkey)
            used_corner_indices.add(idx)

        previous = list(self.identities)

        for idx, c in enumerate(detection.corners):
            if idx in used_corner_indices:
                continue

            uv = np.asarray(self._point2(c.uv), dtype=np.float64)

            best_old = None
            best_d = float("inf")

            for old in previous:
                gkey = (old.global_row, old.global_col)

                if gkey in used_globals:
                    continue

                d = float(
                    np.linalg.norm(
                        uv - np.asarray(old.uv, dtype=np.float64)
                    )
                )

                if d < best_d:
                    best_d = d
                    best_old = old

            if best_old is None:
                continue

            if best_d > self.config.max_identity_match_px:
                continue

            result.append(
                GlobalCornerIdentity(
                    local_row=int(c.i),
                    local_col=int(c.j),
                    global_row=best_old.global_row,
                    global_col=best_old.global_col,
                    xyz_mm=best_old.xyz_mm,
                    uv=(float(uv[0]), float(uv[1])),
                    votes=best_old.votes,
                )
            )

            used_globals.add((best_old.global_row, best_old.global_col))
            used_corner_indices.add(idx)

        return result

    def _identities_from_correspondences(
        self,
        correspondences,
        inlier_indices,
    ) -> List[GlobalCornerIdentity]:
        if len(inlier_indices) > 0:
            selected = [
                correspondences[int(i)]
                for i in inlier_indices
                if 0 <= int(i) < len(correspondences)
            ]
        else:
            selected = list(correspondences)

        identities: List[GlobalCornerIdentity] = []
        used_globals: set[Tuple[int, int]] = set()

        for c in selected:
            gkey = (int(c.global_row), int(c.global_col))

            if gkey in used_globals:
                continue

            identities.append(
                GlobalCornerIdentity(
                    local_row=int(c.local_row),
                    local_col=int(c.local_col),
                    global_row=int(c.global_row),
                    global_col=int(c.global_col),
                    xyz_mm=self._point3(c.xyz_mm),
                    uv=self._point2(c.uv),
                    votes=int(c.votes),
                )
            )

            used_globals.add(gkey)

        return identities

    def _replace_identities(self, identities: List[GlobalCornerIdentity]) -> None:
        self.identities = list(identities)

        self.identity_by_local = {
            (p.local_row, p.local_col): p
            for p in self.identities
        }

    def _store_pose(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        T_marker_camera: np.ndarray,
    ) -> None:
        self.rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        self.tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        self.T_marker_camera = np.asarray(
            T_marker_camera,
            dtype=np.float64,
        ).reshape(4, 4)

    def _confidence(
        self,
        num_inliers: int,
        mean_error_px: float,
    ) -> float:
        point_score = min(1.0, float(num_inliers) / 30.0)

        if mean_error_px < 0.0:
            error_score = 0.0
        else:
            error_score = 1.0 - min(
                1.0,
                mean_error_px / max(
                    1e-6,
                    self.config.max_mean_reprojection_error_px,
                ),
            )

        return float(0.6 * point_score + 0.4 * error_score)

    def draw_debug(
        self,
        frame: np.ndarray,
        result: TrackerResult,
        draw_ids: bool = True,
        draw_axes: bool = True,
    ) -> np.ndarray:
        vis = frame.copy()

        color = (0, 255, 0) if result.success else (0, 0, 255)

        for p in result.corners:
            u = int(round(p.uv[0]))
            v = int(round(p.uv[1]))

            cv2.circle(
                vis,
                (u, v),
                5,
                color,
                -1,
                cv2.LINE_AA,
            )

            if draw_ids:
                cv2.putText(
                    vis,
                    f"{p.global_row},{p.global_col}",
                    (u + 5, v - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        text = (
            f"{result.mode.value} | "
            f"det_track={int(result.detection_tracking)} | "
            f"stable={int(result.detection_stable)} | "
            f"pts={result.num_points} | "
            f"inl={result.num_inliers} | "
            f"err={result.mean_reprojection_error_px:.2f}px | "
            f"conf={result.confidence:.2f} | "
            f"{result.message}"
        )

        cv2.putText(
            vis,
            text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )

        cv2.putText(
            vis,
            text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if (
            draw_axes
            and result.success
            and result.rvec is not None
            and result.tvec is not None
        ):
            try:
                cv2.drawFrameAxes(
                    vis,
                    self.K,
                    self.dist_coeffs,
                    result.rvec.reshape(3, 1),
                    result.tvec.reshape(3, 1),
                    self.config.draw_axes_length_mm,
                )
            except cv2.error:
                pass

        return vis

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