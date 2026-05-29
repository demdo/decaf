from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from overlay.tracking.hydramarker.model.bootstrap import (
    BootstrapResult,
    CameraCalibration,
)
from overlay.tracking.hydramarker.model.observations import (
    FrameObservation,
)


@dataclass(slots=True)
class CameraPose:
    """
    Pose convention:
        X_cam = R * X_world + t
    """

    R: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )

    t: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def camera_center_world(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass(slots=True)
class SfMState:
    calibration: CameraCalibration

    frames: list[FrameObservation]

    poses: Dict[int, CameraPose] = field(default_factory=dict)

    marker_positions: Dict[int, np.ndarray] = field(
        default_factory=dict
    )

    def has_pose(self, frame_id: int) -> bool:
        return int(frame_id) in self.poses

    def add_pose(
        self,
        frame_id: int,
        pose: CameraPose,
    ) -> None:
        self.poses[int(frame_id)] = pose

    def add_marker_position(
        self,
        marker_id: int,
        point: np.ndarray,
    ) -> None:
        self.marker_positions[int(marker_id)] = np.asarray(
            point,
            dtype=np.float64,
        ).reshape(3)

    def posed_frame_ids(self) -> list[int]:
        return sorted(self.poses.keys())

    def unposed_frames(self) -> list[FrameObservation]:
        return [
            frame
            for frame in self.frames
            if frame.frame_id not in self.poses
        ]

    def get_frame(
        self,
        frame_id: int,
    ) -> FrameObservation:
        for frame in self.frames:
            if frame.frame_id == frame_id:
                return frame

        raise KeyError(f"Frame {frame_id} not found.")

    def known_observations_in_frame(
        self,
        frame: FrameObservation,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        marker_ids = []
        object_points = []
        image_points = []

        for marker_id, obs in frame.observations.items():
            if marker_id not in self.marker_positions:
                continue

            marker_ids.append(int(marker_id))
            object_points.append(self.marker_positions[marker_id])
            image_points.append(obs.uv)

        if not marker_ids:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
            )

        return (
            np.asarray(marker_ids, dtype=np.int64),
            np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
            np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
        )


def create_state_from_bootstrap(
    frames: list[FrameObservation],
    calibration: CameraCalibration,
    bootstrap: BootstrapResult,
) -> SfMState:
    if not bootstrap.success:
        raise ValueError(
            "Cannot create SfMState from unsuccessful bootstrap result."
        )

    if bootstrap.marker_ids is None or bootstrap.points_3d is None:
        raise ValueError(
            "Bootstrap result missing marker_ids/points_3d."
        )

    if bootstrap.R_ba is None or bootstrap.t_ba is None:
        raise ValueError(
            "Bootstrap result missing R_ba/t_ba."
        )

    state = SfMState(
        calibration=calibration,
        frames=list(frames),
    )

    state.add_pose(
        bootstrap.frame_a_id,
        CameraPose(
            R=np.eye(3, dtype=np.float64),
            t=np.zeros(3, dtype=np.float64),
        ),
    )

    state.add_pose(
        bootstrap.frame_b_id,
        CameraPose(
            R=bootstrap.R_ba,
            t=bootstrap.t_ba,
        ),
    )

    marker_ids = np.asarray(
        bootstrap.marker_ids,
        dtype=np.int64,
    ).reshape(-1)

    points = np.asarray(
        bootstrap.points_3d,
        dtype=np.float64,
    ).reshape(-1, 3)

    for marker_id, point in zip(marker_ids, points, strict=False):
        state.add_marker_position(
            int(marker_id),
            point,
        )

    return state