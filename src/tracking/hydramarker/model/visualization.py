from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tracking.hydramarker.model.alignment import (
    AlignmentResult,
)
from tracking.hydramarker.model.state import CameraPose, SfMState


def set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    radius = 0.5 * max(x_range, y_range, z_range)

    ax.set_xlim3d([x_middle - radius, x_middle + radius])
    ax.set_ylim3d([y_middle - radius, y_middle + radius])
    ax.set_zlim3d([z_middle - radius, z_middle + radius])


def camera_axes_world(
    pose: CameraPose,
    axis_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = pose.camera_center_world()

    #
    # Pose convention:
    #   X_cam = R * X_world + t
    #
    # Therefore camera-to-world rotation is R.T.
    #
    R_cw = pose.R.T

    x_axis = R_cw[:, 0] * axis_length
    y_axis = R_cw[:, 1] * axis_length
    z_axis = R_cw[:, 2] * axis_length

    return center, x_axis, y_axis, z_axis


def draw_camera(
    ax,
    pose: CameraPose,
    *,
    label: str | None = None,
    axis_length: float = 0.15,
    draw_axes: bool = True,
) -> None:
    center, x_axis, y_axis, z_axis = camera_axes_world(
        pose,
        axis_length=axis_length,
    )

    ax.scatter(
        center[0],
        center[1],
        center[2],
        s=20,
        marker="^",
    )

    if label is not None:
        ax.text(
            center[0],
            center[1],
            center[2],
            label,
            fontsize=7,
        )

    if draw_axes:
        ax.quiver(
            center[0],
            center[1],
            center[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            length=1.0,
            normalize=False,
        )

        ax.quiver(
            center[0],
            center[1],
            center[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            length=1.0,
            normalize=False,
        )

        ax.quiver(
            center[0],
            center[1],
            center[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            length=1.0,
            normalize=False,
        )


def marker_arrays_from_state(
    state: SfMState,
) -> tuple[np.ndarray, np.ndarray]:
    marker_ids = np.asarray(
        sorted(state.marker_positions.keys()),
        dtype=np.int64,
    )

    if marker_ids.size == 0:
        return (
            marker_ids,
            np.empty((0, 3), dtype=np.float64),
        )

    points = np.asarray(
        [state.marker_positions[int(mid)] for mid in marker_ids],
        dtype=np.float64,
    ).reshape(-1, 3)

    return marker_ids, points


def camera_centers_from_state(
    state: SfMState,
) -> tuple[np.ndarray, np.ndarray]:
    frame_ids = np.asarray(
        state.posed_frame_ids(),
        dtype=np.int64,
    )

    if frame_ids.size == 0:
        return (
            frame_ids,
            np.empty((0, 3), dtype=np.float64),
        )

    centers = np.asarray(
        [
            state.poses[int(frame_id)].camera_center_world()
            for frame_id in frame_ids
        ],
        dtype=np.float64,
    ).reshape(-1, 3)

    return frame_ids, centers


def plot_sfm_state(
    state: SfMState,
    *,
    show_marker_ids: bool = True,
    show_camera_labels: bool = False,
    show_camera_axes: bool = False,
    max_labeled_cameras: int = 30,
    camera_axis_length: float | None = None,
    title: str = "HydraMarker SfM State",
    marker_label: str = "3D marker points",
    axis_unit: str = "SfM scale",
) -> None:
    marker_ids, points = marker_arrays_from_state(state)
    frame_ids, centers = camera_centers_from_state(state)

    if camera_axis_length is None:
        if len(points) > 1:
            marker_extent = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
            camera_axis_length = 0.04 * marker_extent
        else:
            camera_axis_length = 0.1

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    if len(points) > 0:
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=marker_ids,
            s=20,
            label=marker_label,
        )

        if show_marker_ids and len(points) <= 200:
            for mid, p in zip(marker_ids, points, strict=False):
                ax.text(
                    p[0],
                    p[1],
                    p[2],
                    str(int(mid)),
                    fontsize=6,
                )

        fig.colorbar(
            scatter,
            ax=ax,
            shrink=0.65,
            label="marker/corner ID",
        )

    if len(centers) > 0:
        ax.plot(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            marker=".",
            linewidth=1,
            markersize=3,
            label="camera trajectory",
        )

        for idx, frame_id in enumerate(frame_ids):
            pose = state.poses[int(frame_id)]

            label = None
            if show_camera_labels and idx < max_labeled_cameras:
                label = str(int(frame_id))

            draw_camera(
                ax,
                pose,
                label=label,
                axis_length=float(camera_axis_length),
                draw_axes=show_camera_axes,
            )

    ax.set_xlabel(f"X [{axis_unit}]")
    ax.set_ylabel(f"Y [{axis_unit}]")
    ax.set_zlabel(f"Z [{axis_unit}]")
    ax.set_title(title)

    ax.legend()
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()


def get_marker_point(
    state: SfMState,
    marker_id: int,
) -> np.ndarray:
    marker_id = int(marker_id)

    if marker_id not in state.marker_positions:
        raise KeyError(f"Marker ID {marker_id} missing in state.")

    return np.asarray(
        state.marker_positions[marker_id],
        dtype=np.float64,
    ).reshape(3)


def plot_alignment_reference_axes(
    state: SfMState,
    alignment_result: AlignmentResult,
    *,
    axis_length_mm: float = 50.0,
    show_marker_ids: bool = True,
) -> None:
    marker_ids, points = marker_arrays_from_state(state)

    if len(points) == 0:
        return

    p0 = get_marker_point(state, alignment_result.origin_id)
    px = get_marker_point(state, alignment_result.x_axis_id)
    py = get_marker_point(state, alignment_result.y_axis_id)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=marker_ids,
        s=24,
        label="aligned marker points",
    )

    ax.scatter(
        [p0[0]],
        [p0[1]],
        [p0[2]],
        s=140,
        marker="o",
        label=f"origin ID {alignment_result.origin_id}",
    )

    ax.scatter(
        [px[0]],
        [px[1]],
        [px[2]],
        s=90,
        marker="^",
        label=f"+X ref ID {alignment_result.x_axis_id}",
    )

    ax.scatter(
        [py[0]],
        [py[1]],
        [py[2]],
        s=90,
        marker="s",
        label=f"+Y ref ID {alignment_result.y_axis_id}",
    )

    ax.quiver(
        p0[0],
        p0[1],
        p0[2],
        axis_length_mm,
        0.0,
        0.0,
        length=1.0,
        normalize=False,
    )

    ax.quiver(
        p0[0],
        p0[1],
        p0[2],
        0.0,
        axis_length_mm,
        0.0,
        length=1.0,
        normalize=False,
    )

    ax.quiver(
        p0[0],
        p0[1],
        p0[2],
        0.0,
        0.0,
        axis_length_mm,
        length=1.0,
        normalize=False,
    )

    ax.text(
        p0[0] + axis_length_mm,
        p0[1],
        p0[2],
        "+X",
        fontsize=12,
        weight="bold",
    )

    ax.text(
        p0[0],
        p0[1] + axis_length_mm,
        p0[2],
        "+Y",
        fontsize=12,
        weight="bold",
    )

    ax.text(
        p0[0],
        p0[1],
        p0[2] + axis_length_mm,
        "+Z",
        fontsize=12,
        weight="bold",
    )

    if show_marker_ids and len(marker_ids) <= 200:
        for mid, p in zip(marker_ids, points, strict=False):
            ax.text(
                p[0],
                p[1],
                p[2],
                str(int(mid)),
                fontsize=6,
            )

    fig.colorbar(
        scatter,
        ax=ax,
        shrink=0.65,
        label="marker/corner ID",
    )

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("HydraMarker Alignment Reference Axes")
    ax.legend()

    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()


def visualize_aligned_state(
    state: SfMState,
    marker_json_path: str | Path,
    alignment_result: AlignmentResult,
    *,
    show_full_sfm_plot: bool = True,
    show_reference_plot: bool = True,
) -> None:
    if show_full_sfm_plot:
        plot_sfm_state(
            state,
            show_marker_ids=True,
            show_camera_labels=False,
            show_camera_axes=True,
            title="HydraMarker SfM State After Marker Alignment",
            axis_unit="mm",
        )

    if show_reference_plot:
        plot_alignment_reference_axes(
            state=state,
            alignment_result=alignment_result,
        )