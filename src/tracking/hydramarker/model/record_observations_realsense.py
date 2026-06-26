from __future__ import annotations

import json
from pathlib import Path
import sys
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.hydramarker.config import TrackerConfig
from tracking.hydramarker.tracker import HydraTracker, PoseSource
from tracking.hydramarker.model.observations import (
    FrameObservation,
    frame_from_tracker_result,
    save_observations_npz,
)


CAMERA_CALIBRATION_FILTER = "OpenCV camera calibration (*.npz);;All files (*)"
MIN_SAVE_CORRESPONDENCES = 20
MAX_SAVE_MEAN_REPROJECTION_ERROR_PX = 3.0
MIN_SAVE_MOTION_PX = 1.0
MIN_SHARED_IDS_FOR_MOTION = 8


def _first_npz_array(npz, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name in npz:
            return np.asarray(npz[name], dtype=np.float64)
    return None


def _read_calibration_image_size(npz) -> list[int] | None:
    if "image_size" in npz:
        values = np.asarray(npz["image_size"]).reshape(-1)
        if values.size >= 2:
            return [int(values[0]), int(values[1])]

    width_keys = ("width", "image_width", "rgb_width")
    height_keys = ("height", "image_height", "rgb_height")
    width = next((int(np.asarray(npz[k]).reshape(-1)[0]) for k in width_keys if k in npz), None)
    height = next((int(np.asarray(npz[k]).reshape(-1)[0]) for k in height_keys if k in npz), None)
    if width is not None and height is not None:
        return [width, height]

    return None


def load_opencv_camera_calibration_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Camera calibration file not found: {path}")

    with np.load(path, allow_pickle=True) as npz:
        K = _first_npz_array(
            npz,
            ("K", "K_rgb", "camera_matrix", "camera_intrinsics", "intrinsics"),
        )
        if K is None:
            raise KeyError(
                "Camera calibration NPZ must contain one of: "
                "K, K_rgb, camera_matrix, camera_intrinsics, intrinsics."
            )

        dist = _first_npz_array(
            npz,
            ("dist", "dist_rgb", "dist_coeffs", "distortion_coeffs", "opencv_dist_coeffs"),
        )
        if dist is None:
            raise KeyError(
                "Camera calibration NPZ must contain OpenCV distortion coefficients: "
                "dist, dist_rgb, dist_coeffs, distortion_coeffs, or opencv_dist_coeffs."
            )

        image_size = _read_calibration_image_size(npz)

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    if dist.size == 0:
        raise ValueError("Distortion coefficients must not be empty.")

    info: dict[str, object] = {
        "camera_source": "opencv_calibration_npz",
        "camera_calibration_path": str(path),
        "distortion_model": "opencv_brown_conrady",
        "K": K.tolist(),
        "opencv_dist_coeffs": dist.reshape(-1).tolist(),
        "effective_opencv_dist_coeffs": dist.reshape(-1).tolist(),
    }
    if image_size is not None:
        info["calibration_image_size"] = image_size

    return K, dist, info


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


def load_tracker_camera_calibration(profile) -> tuple[np.ndarray, np.ndarray]:
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    calib_path = choose_file_qt(
        "Select OpenCV camera calibration NPZ",
        CAMERA_CALIBRATION_FILTER,
    )
    K, dist, calib_info = load_opencv_camera_calibration_npz(calib_path)
    stream_size = [int(getattr(intr, "width", 0)), int(getattr(intr, "height", 0))]
    calib_size = calib_info.get("calibration_image_size")
    if calib_size is not None and list(calib_size) != stream_size:
        raise RuntimeError(
            f"Selected camera calibration image_size={calib_size} does not match "
            f"the active RealSense color stream {stream_size}."
        )

    print(
        "[record_observations] using selected camera calibration="
        f"{calib_info['camera_calibration_path']} "
        f"dist={dist.reshape(-1).tolist()}"
    )
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


def median_shared_displacement_px(
    a: FrameObservation,
    b: FrameObservation,
    *,
    min_shared_ids: int = MIN_SHARED_IDS_FOR_MOTION,
) -> float | None:
    shared_ids = sorted(set(a.observations) & set(b.observations))
    if len(shared_ids) < min_shared_ids:
        return None

    displacements = []
    for marker_id in shared_ids:
        uv_a = np.asarray(a.observations[marker_id].uv, dtype=np.float64)
        uv_b = np.asarray(b.observations[marker_id].uv, dtype=np.float64)
        displacements.append(float(np.linalg.norm(uv_b - uv_a)))

    return float(np.median(displacements))


def should_save_frame(result, obs: FrameObservation) -> tuple[bool, str]:
    if not is_decode_source(result):
        return False, "not decode"

    if not result.success:
        return False, "pose failed"

    if len(obs.observations) < MIN_SAVE_CORRESPONDENCES:
        return False, f"corr < {MIN_SAVE_CORRESPONDENCES}"

    mean_error = float(getattr(result, "mean_reprojection_error_px", float("inf")))
    if not np.isfinite(mean_error):
        return False, "invalid reproj"

    if mean_error > MAX_SAVE_MEAN_REPROJECTION_ERROR_PX:
        return False, f"reproj > {MAX_SAVE_MEAN_REPROJECTION_ERROR_PX:.1f}px"

    return True, "saved"


def make_output_path() -> Path:
    script_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return script_dir / f"hydramarker_observations_{timestamp}.npz"


def frame_from_correspondence_corners(
    frame_id: int,
    result,
    num_cols: int,
    *,
    require_decode_source: bool = False,
):
    if require_decode_source and not is_decode_source(result):
        return FrameObservation(frame_id=int(frame_id), timestamp=None)

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


def pose_source_value(result) -> str:
    source = getattr(result, "pose_source", PoseSource.NONE)
    return str(getattr(source, "value", source))


def is_decode_source(result) -> bool:
    return pose_source_value(result) == PoseSource.DECODE.value


def assert_sfm_recorder_config(tracker: HydraTracker) -> None:
    config = tracker.config

    if config.enable_fast_persistent_path:
        raise RuntimeError("SfM recorder must run with Fast Path disabled.")

    if not config.decode_only_mode:
        raise RuntimeError("SfM recorder must run in tracker decode-only mode.")

    if config.enable_pose_propagation:
        raise RuntimeError("SfM recorder must run with pose propagation disabled.")

    if config.enable_temporal_correspondence_persistence:
        raise RuntimeError(
            "SfM recorder must not save persistent fallback correspondences."
        )


def make_sfm_tracker(
    field_path: Path,
    marker_json_path: Path,
    K,
    dist,
) -> HydraTracker:
    return HydraTracker(
        field_path=str(field_path),
        marker_json_path=str(marker_json_path),
        K=K,
        dist_coeffs=dist,
        config=TrackerConfig(
            min_points=6,
            min_inliers=5,
            max_mean_reprojection_error_px=4.0,
            max_max_reprojection_error_px=15.0,
            max_lost_frames=8,
            max_translation_jump_mm=120.0,
            max_rotation_jump_deg=360.0,
            rotation_gate_scale_per_lost_frame=0.0,
            rotation_gate_max_deg=360.0,
            pnp_ransac_iterations=500,
            pnp_ransac_reprojection_px=3.0,
            pnp_ransac_confidence=0.99,
            use_pose_prior=True,
            corr_min_votes=2,
            corr_discard_conflicts=True,
            corr_require_detection_stable=False,
            corr_enable_dominant_rotation_filter=True,
            corr_min_rotation_support=2,
            corr_min_rotation_support_ratio=0.55,
            decode_only_mode=True,
            enable_fast_persistent_path=False,
            enable_pose_propagation=False,
            enable_temporal_correspondence_persistence=False,
            persistence_use_pose_projection=True,
            persistence_projection_max_reproj_px=12.0,
            persistence_projection_max_pose_error_px=2.5,
            dot_use_temporal_smoothing=False,
            dot_commit_frames=1,
            dot_revoke_frames=5,
            persistence_max_frames=8,
            pose_hold_max_frames=0,
            emergency_pose_hold_enabled=False,
        ),
    )


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

    observations_path = make_output_path()

    pipe, profile = create_realsense_pipeline()
    K, dist = load_tracker_camera_calibration(profile)

    tracker = make_sfm_tracker(
        field_path,
        marker_json_path,
        K,
        dist,
    )
    assert_sfm_recorder_config(tracker)
    print("[recorder] SfM mode: Fast Path disabled, Decode-only save enabled")

    window_name = "HydraMarker SfM Observation Recorder"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    recording = False
    observations: list[FrameObservation] = []
    frame_id = 0
    saved_frames = 0
    skipped_frames = 0
    last_decision = "idle"
    last_saved_motion_px: float | None = None

    def start_recording() -> None:
        nonlocal recording, last_decision

        recording = True
        last_decision = "recording"
        print("[recorder] recording started: move/rotate the drill slowly")

    def stop_recording() -> None:
        nonlocal recording

        recording = False
        print(f"[recorder] recording stopped: saved_frames={len(observations)}")

    try:
        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            result = tracker.process_frame(frame)
            source_value = pose_source_value(result)

            vis = frame.copy()
            draw_observations(vis, result)

            if recording:
                obs = frame_from_correspondence_corners(
                    frame_id=frame_id,
                    result=result,
                    num_cols=num_corner_cols,
                    require_decode_source=True,
                )

                should_save, reason = should_save_frame(result, obs)
                motion_px = None
                if should_save and observations:
                    motion_px = median_shared_displacement_px(observations[-1], obs)
                    if motion_px is not None and motion_px < MIN_SAVE_MOTION_PX:
                        should_save = False
                        reason = f"motion < {MIN_SAVE_MOTION_PX:.1f}px"

                if should_save:
                    observations.append(obs)
                    frame_id += 1
                    saved_frames += 1
                    last_saved_motion_px = motion_px
                    if motion_px is None:
                        last_decision = f"saved | corr={len(obs.observations)}"
                    else:
                        last_decision = (
                            f"saved | corr={len(obs.observations)} | "
                            f"motion={motion_px:.2f}px"
                        )
                else:
                    skipped_frames += 1
                    last_decision = f"skip: {reason}"

            status_1 = (
                f"{'REC' if recording else 'IDLE'} | "
                f"saved_frames={saved_frames} | "
                f"skipped={skipped_frames} | "
                f"unique_ids={count_unique_marker_ids(observations)} | "
                f"num_corner_cols={num_corner_cols}"
            )

            status_2 = (
                f"mode={result.mode.value} | "
                f"src={source_value} | "
                f"success={result.success} | "
                f"det={len(result.detection_corners)} | "
                f"corr={len(result.correspondence_corners)} | "
                f"inliers={len(result.corners)} | "
                f"err={result.mean_reprojection_error_px:.2f}px"
            )

            if last_saved_motion_px is None:
                motion_status = "last_motion=n/a"
            else:
                motion_status = f"last_motion={last_saved_motion_px:.2f}px"

            status_3 = (
                f"{last_decision} | "
                f"min_corr={MIN_SAVE_CORRESPONDENCES} | "
                f"max_err={MAX_SAVE_MEAN_REPROJECTION_ERROR_PX:.1f}px | "
                f"min_motion={MIN_SAVE_MOTION_PX:.1f}px | "
                f"{motion_status}"
            )

            status_4 = (
                "SfM decode-only, fast=OFF | "
                "blue=detection yellow=sfm-corr green=pnp-inlier | "
                "Space=start/stop, s=save, r=reset, q=quit"
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
            put_text(
                vis,
                status_4,
                (25, 125),
                color=(255, 255, 255),
            )

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                if recording:
                    stop_recording()
                break

            if key == ord(" "):
                if recording:
                    stop_recording()
                else:
                    start_recording()

            elif key == ord("s"):
                save_observations_npz(observations_path, observations)
                print(f"Saved observations: {observations_path}")

            elif key == ord("r"):
                tracker.reset()
                observations.clear()
                frame_id = 0
                saved_frames = 0
                skipped_frames = 0
                recording = False
                last_decision = "reset"
                last_saved_motion_px = None

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

        if observations:
            save_observations_npz(observations_path, observations)
            print(f"Saved observations: {observations_path}")


if __name__ == "__main__":
    main()
