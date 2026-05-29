from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


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