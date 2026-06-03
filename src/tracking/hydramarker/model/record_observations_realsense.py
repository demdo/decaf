from __future__ import annotations

import json
from pathlib import Path
import sys
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.hydramarker.tracker import HydraTracker
from tracking.hydramarker.model.observations import (
    frame_from_tracker_result,
    save_observations_npz,
)


def choose_file_qt(title: str, file_filter: str) -> Path:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(None, title, "", file_filter)
    if not path:
        raise RuntimeError(f"No file selected: {title}")

    return Path(path)


def read_num_corner_cols(marker_json_path: Path) -> int:
    """
    Read the corner ID encoding stride directly from the marker JSON.

    The correct value is id_encoding.num_cols, which equals the number of
    corner columns in the detectable grid (cell_cols + 1).

    Do NOT compute this from the .field header — the field format does not
    reliably encode the corner count, and off-by-one errors there produce
    a wrong ID stride that silently corrupts all saved observations.
    """
    marker_json_path = Path(marker_json_path)

    with marker_json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    try:
        num_cols = int(meta["id_encoding"]["num_cols"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot read id_encoding.num_cols from {marker_json_path}: {exc}"
        ) from exc

    if num_cols < 2:
        raise ValueError(
            f"id_encoding.num_cols={num_cols} in {marker_json_path} is invalid "
            "(must be >= 2)."
        )

    return num_cols


def realsense_intrinsics_to_cv(profile) -> tuple[np.ndarray, np.ndarray]:
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    K = np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    dist = np.asarray(intr.coeffs[:5], dtype=np.float64).reshape(-1, 1)
    return K, dist


def create_realsense_pipeline():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    return pipe, profile


def put_text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 255),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_detection_corners(vis: np.ndarray, result) -> None:
    for c in result.detection_corners:
        u = int(round(c.uv[0]))
        v = int(round(c.uv[1]))

        cv2.circle(
            vis,
            (u, v),
            3,
            (255, 0, 0),
            -1,
            cv2.LINE_AA,
        )


def draw_correspondence_corners(vis: np.ndarray, result) -> None:
    for c in result.correspondence_corners:
        u = int(round(c.uv[0]))
        v = int(round(c.uv[1]))

        cv2.circle(
            vis,
            (u, v),
            4,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            vis,
            f"{c.global_row},{c.global_col}",
            (u + 5, v - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_pose_inlier_corners(vis: np.ndarray, result) -> None:
    color = (0, 255, 0) if result.success else (0, 0, 255)

    for c in result.corners:
        u = int(round(c.uv[0]))
        v = int(round(c.uv[1]))

        cv2.circle(
            vis,
            (u, v),
            6,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_observations(vis: np.ndarray, result) -> None:
    draw_detection_corners(vis, result)
    draw_correspondence_corners(vis, result)
    draw_pose_inlier_corners(vis, result)


def count_unique_marker_ids(frames) -> int:
    ids = set()
    for frame in frames:
        ids.update(frame.observations.keys())
    return len(ids)


def save_camera_intrinsics(path: Path, K: np.ndarray, dist: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        K=np.asarray(K, dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64),
    )


def make_output_paths() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    observations_path = script_dir / f"hydramarker_observations_{timestamp}.npz"
    camera_path = script_dir / f"camera_intrinsics_{timestamp}.npz"

    return observations_path, camera_path


def frame_from_correspondence_corners(
    frame_id: int,
    result,
    num_cols: int,
):
    old_corners = result.corners
    result.corners = result.correspondence_corners

    try:
        return frame_from_tracker_result(
            frame_id=frame_id,
            result=result,
            num_cols=num_cols,
            timestamp=None,
            only_success=False,
        )
    finally:
        result.corners = old_corners


def main() -> None:
    field_path = choose_file_qt(
        "Select HydraMarker .field file",
        "HydraMarker field (*.field)",
    )

    marker_json_path = choose_file_qt(
        "Select marker .json file",
        "Marker JSON (*.json)",
    )

    # Read the ID stride directly from the marker JSON (id_encoding.num_cols).
    # This is always correct regardless of .field header format or cell count.
    num_corner_cols = read_num_corner_cols(marker_json_path)
    print(f"[recorder] num_corner_cols = {num_corner_cols} (from marker JSON)")

    observations_path, camera_path = make_output_paths()

    pipe, profile = create_realsense_pipeline()
    K, dist = realsense_intrinsics_to_cv(profile)
    save_camera_intrinsics(camera_path, K, dist)

    tracker = HydraTracker(
        field_path=str(field_path),
        marker_json_path=str(marker_json_path),
        K=K,
        dist_coeffs=dist,
    )

    window_name = "HydraMarker SfM Observation Recorder"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    recording = False
    observations = []
    frame_id = 0

    try:
        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            result = tracker.process_frame(frame)

            vis = frame.copy()
            draw_observations(vis, result)

            if recording:
                obs = frame_from_correspondence_corners(
                    frame_id=frame_id,
                    result=result,
                    num_cols=num_corner_cols,
                )

                if len(obs.observations) > 0:
                    observations.append(obs)

                frame_id += 1

            status_1 = (
                f"{'REC' if recording else 'IDLE'} | "
                f"saved_frames={len(observations)} | "
                f"unique_ids={count_unique_marker_ids(observations)} | "
                f"num_corner_cols={num_corner_cols}"
            )

            status_2 = (
                f"mode={result.mode.value} | "
                f"success={result.success} | "
                f"det={len(result.detection_corners)} | "
                f"corr={len(result.correspondence_corners)} | "
                f"inliers={len(result.corners)}"
            )

            status_3 = (
                f"err={result.mean_reprojection_error_px:.2f}px | "
                "blue=detection yellow=sfm-corr green=pnp-inlier"
            )

            put_text(
                vis,
                status_1,
                (25, 35),
                color=(0, 180, 255) if recording else (0, 255, 255),
            )
            put_text(
                vis,
                status_2,
                (25, 65),
                color=(0, 255, 0) if result.success else (0, 0, 255),
            )
            put_text(
                vis,
                status_3,
                (25, 95),
                color=(255, 255, 255),
            )

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            if key == ord(" "):
                recording = not recording

            elif key == ord("s"):
                save_observations_npz(observations_path, observations)
                print(f"Saved observations: {observations_path}")

            elif key == ord("r"):
                tracker.reset()
                observations.clear()
                frame_id = 0
                recording = False

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

        if observations:
            save_observations_npz(observations_path, observations)
            print(f"Saved observations: {observations_path}")
            print(f"Saved intrinsics: {camera_path}")


if __name__ == "__main__":
    main()