from __future__ import annotations

import cv2
import numpy as np

from tracking.hydramarker.tracker_types import PoseSource, TrackerMode, TrackerResult


class DecodePipelineMixin:
    def _decode_and_estimate_pose(self, frame: np.ndarray, detection) -> TrackerResult:
        # Reset the dot detector immediately after large drill rotations.
        # This must run before dot_detector.detect() so the reset affects the
        # same frame in which the rotation is detected.
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
                    self._last_accepted_rvec = None
            except Exception:
                pass

        dots = self.dot_detector.detect(frame, detection)

        # Reset smoothing early when the point count drops far below the best
        # recent count. That gives the EMA state a chance to recover before a
        # complete decode failure.
        if (
            self.mode == TrackerMode.TRACKING
            and self.config.dot_early_reset_pts_ratio > 0.0
            and self._max_pts_seen >= self.config.dot_early_reset_min_pts
        ):
            # Estimate points from valid, non-ambiguous detector cells.
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
