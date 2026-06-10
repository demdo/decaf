from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from tracking.hydramarker.tracker_types import TrackerResult


@dataclass
class TrackerLogEvent:
    level: str
    stage: str
    frame_index: int
    message: str


class TrackerLogger:
    def __init__(
        self,
        log_path: str = "hydramarker_tracker.log",
        enable_console: bool = False,
    ) -> None:
        self.log_path = Path(log_path)
        self.enable_console = bool(enable_console)

        self._last_signature = None

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=== HydraMarker Tracker Log ===\n")

    def info(
        self,
        stage: str,
        frame_index: int,
        message: str,
    ) -> None:
        self._write(
            TrackerLogEvent(
                level="INFO",
                stage=stage,
                frame_index=frame_index,
                message=message,
            )
        )

    def warn(
        self,
        stage: str,
        frame_index: int,
        message: str,
    ) -> None:
        self._write(
            TrackerLogEvent(
                level="WARN",
                stage=stage,
                frame_index=frame_index,
                message=message,
            )
        )

    def error(
        self,
        stage: str,
        frame_index: int,
        message: str,
    ) -> None:
        self._write(
            TrackerLogEvent(
                level="ERROR",
                stage=stage,
                frame_index=frame_index,
                message=message,
            )
        )

    def log_tracker_result(
        self,
        stage: str,
        frame_index: int,
        result: TrackerResult,
        *,
        decode_only: bool,
        lost_frames: int,
        persisted_count: int,
    ) -> None:
        if not self._should_log_tracker_result(frame_index, result):
            return

        fast = result.fast_path_debug
        policy = "decode_only" if decode_only else "tracking"
        message = (
            f"mode={result.mode.value} | "
            f"policy={policy} | "
            f"success={result.success} | "
            f"source={result.pose_source.value} | "
            f"pnp={result.pnp_method} | "
            f"fast={int(fast.attempted)}/{int(fast.success)}:"
            f"{fast.matches}:{fast.reason} | "
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
            f"lost_frames={lost_frames} | "
            f"persisted={persisted_count}"
        )

        if result.success:
            self.info(stage, frame_index, message)
        else:
            self.warn(stage, frame_index, message)

    @staticmethod
    def _should_log_tracker_result(
        frame_index: int,
        result: TrackerResult,
    ) -> bool:
        return (
            not result.success
            or frame_index <= 5
            or frame_index % 30 == 0
            or "persistent" in result.message.lower()
        )

    def _write(self, event: TrackerLogEvent) -> None:
        signature = (
            event.level,
            event.stage,
            event.message,
        )

        if signature == self._last_signature:
            return

        self._last_signature = signature

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        line = (
            f"[{timestamp}] "
            f"[{event.level}] "
            f"[frame={event.frame_index}] "
            f"[{event.stage}] "
            f"{event.message}"
        )

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if self.enable_console:
            print(line)
