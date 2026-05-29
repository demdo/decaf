from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from tracking.hydramarker.backend import cpp_impl as hm
from tracking.hydramarker.map_pose_tracker import (
    MapPoseTracker,
    MapPoseTrackerConfig,
    PoseTrackPoint,
)
from tracking.hydramarker.tracker_logger import TrackerLogger


class TrackerMode(str, Enum):
    LOST = "LOST"
    DETECTING = "DETECTING"
    TRACKING = "TRACKING"
    RECOVERING = "RECOVERING"


@dataclass
class DetectedCorner:
    local_row: int
    local_col: int
    uv: Tuple[float, float]


@dataclass
class TrackerCorner:
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
    detection_corners: List[DetectedCorner] = field(default_factory=list)

    corners: List[TrackerCorner] = field(default_factory=list)
    correspondence_corners: List[TrackerCorner] = field(default_factory=list)

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
    min_points: int = 6
    min_inliers: int = 5

    max_mean_reprojection_error_px: float = 4.0
    max_max_reprojection_error_px: float = 15.0

    max_lost_frames: int = 8

    max_translation_jump_mm: float = 120.0
    max_rotation_jump_deg: float = 45.0

    pnp_ransac_iterations: int = 500
    pnp_ransac_reprojection_px: float = 3.0
    pnp_ransac_confidence: float = 0.99
    use_pose_prior: bool = True

    dot_canonical_size: int = 80
    dot_canonical_margin_px: float = 4.0
    dot_min_dot_contrast: float = 8.0
    dot_strong_dot_contrast: float = 35.0
    dot_commit_threshold: float = 0.45
    dot_revoke_threshold: float = 0.20
    dot_uncertainty_low: float = 0.20
    dot_uncertainty_high: float = 0.45
    dot_warmup_frames: int = 1
    dot_temporal_alpha: float = 0.35
    dot_commit_frames: int = 2
    dot_revoke_frames: int = 3

    decoder_require_geometry_valid: bool = True
    decoder_accept_ambiguous: bool = False

    corr_min_votes: int = 2
    corr_discard_conflicts: bool = True
    corr_require_detection_stable: bool = False

    corr_enable_dominant_rotation_filter: bool = True
    corr_min_rotation_support: int = 2
    corr_min_rotation_support_ratio: float = 0.55

    enable_temporal_correspondence_persistence: bool = True
    persistence_max_frames: int = 8
    persistence_min_points: int = 8
    persistence_min_fresh_points_for_merge: int = 3

    enable_debug_prints: bool = True
    log_path: str = "hydramarker_tracker.log"
    log_to_console: bool = False


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

        self.field = hm.MarkerField.loadFromFile(field_path)
        self.geometry = hm.MarkerGeometry.load_from_json(marker_json_path)

        self.checkerboard_detector = hm.CheckerboardDetector()
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
                use_pose_prior=self.config.use_pose_prior,
            ),
        )

        self.logger = TrackerLogger(
            log_path=self.config.log_path,
            enable_console=self.config.log_to_console,
        )

        self.mode = TrackerMode.LOST
        self.frame_index = 0
        self.lost_frames = 0

        self._persistent_corners: List[TrackerCorner] = []
        self._persistent_frame_index: int = -1

    @property
    def rvec(self) -> Optional[np.ndarray]:
        return self.pose_tracker.rvec

    @property
    def tvec(self) -> Optional[np.ndarray]:
        return self.pose_tracker.tvec

    @property
    def T_marker_camera(self) -> Optional[np.ndarray]:
        return self.pose_tracker.T_marker_camera

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

        self.pose_tracker.reset()
        self.checkerboard_detector.reset_tracking()
        self.dot_detector = self._create_dot_detector()
        self._clear_persistent_correspondences()

    def process_frame(self, frame: np.ndarray) -> TrackerResult:
        self.frame_index += 1

        detection = self.checkerboard_detector.detect(frame)

        if detection is None or not detection.valid():
            self._on_tracking_failure()
            result = TrackerResult(
                success=False,
                mode=self.mode,
                message="No valid checkerboard detection.",
                detection_valid=False,
                detection_tracking=False if detection is None else bool(detection.tracking),
                detection_stable=False if detection is None else bool(detection.stable),
                detection_corners=self._detected_corners_from_detection(detection),
            )
            self._log_result("NO_DETECTION", result)
            return result

        result = self._decode_and_estimate_pose(frame, detection)
        self._attach_detection_info(result, detection)

        if result.success:
            self.mode = TrackerMode.TRACKING
            self.lost_frames = 0
            result.mode = self.mode
            self._log_result("TRACK_OK", result)
            return result

        self._on_tracking_failure()
        result.mode = self.mode
        result.corners = []
        self._log_result("TRACK_FAIL", result)
        return result

    def _on_tracking_failure(self) -> None:
        self.lost_frames += 1

        if self.lost_frames > self.config.max_lost_frames:
            self.pose_tracker.reset()
            self._clear_persistent_correspondences()
            self.mode = TrackerMode.LOST
        elif self.pose_tracker.rvec is not None and self.pose_tracker.tvec is not None:
            self.mode = TrackerMode.RECOVERING
        else:
            self.mode = TrackerMode.DETECTING

    def _decode_and_estimate_pose(self, frame: np.ndarray, detection) -> TrackerResult:
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
            fallback = self._estimate_pose_from_persistent_correspondences(
                detection,
                reason="No valid decoded patches",
            )
            if fallback is not None:
                return fallback

            return TrackerResult(
                success=False,
                mode=self.mode,
                message="No valid decoded patches.",
            )

        corr_result = self.correspondence_builder.build(
            detection,
            decoded_valid,
            self.geometry,
        )

        if not corr_result.valid():
            fallback = self._estimate_pose_from_persistent_correspondences(
                detection,
                reason="Correspondence build failed",
            )
            if fallback is not None:
                return fallback

            return TrackerResult(
                success=False,
                mode=self.mode,
                message="Correspondence build failed.",
            )

        track_points, tracker_corners = self._points_from_correspondences(
            corr_result.correspondences,
        )

        if len(track_points) < self.config.min_points:
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
                )
                if pose_result.success:
                    return pose_result

            fallback = self._estimate_pose_from_persistent_correspondences(
                detection,
                reason=f"Too few correspondences: {len(track_points)}",
            )
            if fallback is not None:
                return fallback

            return TrackerResult(
                success=False,
                mode=self.mode,
                message=f"Too few correspondences: {len(track_points)}.",
                num_points=len(track_points),
                correspondence_corners=tracker_corners,
            )

        pose_result = self._estimate_and_package_pose(
            track_points,
            tracker_corners,
            success_message="Pose estimation successful.",
            update_persistence=True,
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
    ) -> TrackerResult:
        pose = self.pose_tracker.estimate_pose(track_points)

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
                corners=[],
                correspondence_corners=tracker_corners,
            )

        inlier_corners = self._inlier_corners_from_pose(pose, tracker_corners)

        if update_persistence:
            self._store_persistent_correspondences(inlier_corners)

        confidence = self._confidence(
            pose.num_inliers,
            pose.reprojection_mean_px,
        )

        return TrackerResult(
            success=True,
            mode=TrackerMode.TRACKING,
            message=success_message,
            corners=inlier_corners,
            correspondence_corners=tracker_corners,
            rvec=pose.rvec,
            tvec=pose.tvec,
            T_marker_camera=pose.T_marker_camera,
            mean_reprojection_error_px=pose.reprojection_mean_px,
            max_reprojection_error_px=pose.reprojection_max_px,
            num_points=pose.num_points,
            num_inliers=pose.num_inliers,
            confidence=confidence,
        )

    def _estimate_pose_from_persistent_correspondences(
        self,
        detection,
        reason: str,
    ) -> Optional[TrackerResult]:
        if not self.config.enable_temporal_correspondence_persistence:
            return None

        points, corners = self._persistent_correspondences_for_detection(detection)

        if len(points) < self.config.persistence_min_points:
            return None

        result = self._estimate_and_package_pose(
            points,
            corners,
            success_message=(
                f"Pose estimated from persistent correspondences after: {reason}."
            ),
            update_persistence=False,
        )

        if result.success:
            result.confidence *= 0.85
            return result

        return None

    def _merge_with_persistent_correspondences(
        self,
        detection,
        fresh_points: List[PoseTrackPoint],
        fresh_corners: List[TrackerCorner],
    ) -> Tuple[List[PoseTrackPoint], List[TrackerCorner]]:
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

    def _persistent_correspondences_for_detection(
        self,
        detection,
    ) -> Tuple[List[PoseTrackPoint], List[TrackerCorner]]:
        if not self._persistent_corners:
            return [], []

        if self._persistent_frame_index < 0:
            return [], []

        age = self.frame_index - self._persistent_frame_index
        if age < 0 or age > self.config.persistence_max_frames:
            return [], []

        current_uv_by_local = self._current_uv_by_local_corner(detection)
        if not current_uv_by_local:
            return [], []

        points: List[PoseTrackPoint] = []
        corners: List[TrackerCorner] = []
        used_globals: set[Tuple[int, int]] = set()

        for cached in self._persistent_corners:
            local_key = (int(cached.local_row), int(cached.local_col))
            if local_key not in current_uv_by_local:
                continue

            global_key = (int(cached.global_row), int(cached.global_col))
            if global_key in used_globals:
                continue

            uv = current_uv_by_local[local_key]
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
                    local_row=local_key[0],
                    local_col=local_key[1],
                    global_row=global_key[0],
                    global_col=global_key[1],
                    xyz_mm=xyz,
                    uv=uv,
                    votes=votes,
                )
            )

            used_globals.add(global_key)

        return points, corners

    def _current_uv_by_local_corner(self, detection) -> Dict[Tuple[int, int], Tuple[float, float]]:
        uv_by_local: Dict[Tuple[int, int], Tuple[float, float]] = {}

        for corner in self._detected_corners_from_detection(detection):
            uv_by_local[(int(corner.local_row), int(corner.local_col))] = (
                float(corner.uv[0]),
                float(corner.uv[1]),
            )

        return uv_by_local

    def _store_persistent_correspondences(self, corners: List[TrackerCorner]) -> None:
        if not self.config.enable_temporal_correspondence_persistence:
            return

        clean: List[TrackerCorner] = []
        used_local: set[Tuple[int, int]] = set()
        used_global: set[Tuple[int, int]] = set()

        for corner in corners:
            local_key = (int(corner.local_row), int(corner.local_col))
            global_key = (int(corner.global_row), int(corner.global_col))

            if local_key in used_local or global_key in used_global:
                continue

            clean.append(
                TrackerCorner(
                    local_row=local_key[0],
                    local_col=local_key[1],
                    global_row=global_key[0],
                    global_col=global_key[1],
                    xyz_mm=self._point3(corner.xyz_mm),
                    uv=self._point2(corner.uv),
                    votes=int(corner.votes),
                )
            )

            used_local.add(local_key)
            used_global.add(global_key)

        if len(clean) >= self.config.persistence_min_points:
            self._persistent_corners = clean
            self._persistent_frame_index = self.frame_index

    def _clear_persistent_correspondences(self) -> None:
        self._persistent_corners = []
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
        if not self.config.enable_debug_prints:
            return

        should_log = (
            not result.success
            or self.frame_index <= 5
            or self.frame_index % 30 == 0
            or "persistent" in result.message.lower()
        )

        if not should_log:
            return

        message = (
            f"mode={result.mode.value} | "
            f"success={result.success} | "
            f"msg={result.message} | "
            f"det_valid={result.detection_valid} | "
            f"det_tracking={result.detection_tracking} | "
            f"det_stable={result.detection_stable} | "
            f"det={len(result.detection_corners)} | "
            f"corr={len(result.correspondence_corners)} | "
            f"pose={len(result.corners)} | "
            f"points={result.num_points} | "
            f"inliers={result.num_inliers} | "
            f"mean_err={result.mean_reprojection_error_px:.3f} | "
            f"max_err={result.max_reprojection_error_px:.3f} | "
            f"lost_frames={self.lost_frames} | "
            f"persisted={len(self._persistent_corners)}"
        )

        if result.success:
            self.logger.info(stage, self.frame_index, message)
        else:
            self.logger.warn(stage, self.frame_index, message)

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