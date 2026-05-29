from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(slots=True)
class MarkerObservation:
    marker_id: int
    uv: Tuple[float, float]
    confidence: float = 1.0


@dataclass(slots=True)
class FrameObservation:
    frame_id: int
    observations: Dict[int, MarkerObservation] = field(default_factory=dict)
    timestamp: Optional[float] = None

    def shared_ids(self, other: "FrameObservation") -> List[int]:
        return sorted(set(self.observations) & set(other.observations))

    def add(self, obs: MarkerObservation) -> None:
        self.observations[int(obs.marker_id)] = obs

    def to_array(self) -> np.ndarray:
        rows = [
            [obs.marker_id, obs.uv[0], obs.uv[1], obs.confidence]
            for obs in self.observations.values()
        ]

        if not rows:
            return np.empty((0, 4), dtype=np.float64)

        return np.asarray(rows, dtype=np.float64)


def corner_id(
    global_row: int,
    global_col: int,
    num_cols: int,
) -> int:
    return int(global_row) * int(num_cols) + int(global_col)


def frame_from_tracker_result(
    frame_id: int,
    result,
    *,
    num_cols: int,
    timestamp: Optional[float] = None,
    only_success: bool = False,
) -> FrameObservation:
    frame = FrameObservation(
        frame_id=int(frame_id),
        timestamp=timestamp,
    )

    if only_success and not result.success:
        return frame

    for c in result.corners:
        mid = corner_id(
            c.global_row,
            c.global_col,
            num_cols,
        )

        frame.add(
            MarkerObservation(
                marker_id=mid,
                uv=(float(c.uv[0]), float(c.uv[1])),
                confidence=float(getattr(result, "confidence", 1.0)),
            )
        )

    return frame


def save_observations_npz(
    path: Path,
    frames: list[FrameObservation],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = []

    for frame in frames:
        payload.append(
            {
                "frame_id": int(frame.frame_id),
                "timestamp": frame.timestamp,
                "detections": frame.to_array(),
            }
        )

    np.savez_compressed(
        path,
        frames=np.asarray(payload, dtype=object),
    )


def load_observations_npz(path: Path) -> list[FrameObservation]:
    path = Path(path)

    with np.load(path, allow_pickle=True) as npz:
        payload = npz["frames"]

    frames: list[FrameObservation] = []

    for item in payload:
        entry = item.item() if hasattr(item, "item") else item

        frame = FrameObservation(
            frame_id=int(entry["frame_id"]),
            timestamp=entry.get("timestamp"),
        )

        detections = np.asarray(
            entry["detections"],
            dtype=np.float64,
        )

        for row in detections:
            marker_id = int(row[0])
            u = float(row[1])
            v = float(row[2])
            conf = float(row[3]) if row.shape[0] >= 4 else 1.0

            frame.add(
                MarkerObservation(
                    marker_id=marker_id,
                    uv=(u, v),
                    confidence=conf,
                )
            )

        frames.append(frame)

    frames.sort(key=lambda f: f.frame_id)
    return frames