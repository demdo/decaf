from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.hydramarker.tracker import (
    HydraTracker,
    TrackerConfig,
    TrackerResult,
)


INPUT_MODE = "realsense"  # "realsense" or "image"

WINDOW_NAME = "HydraMarker frame renderer"

OUTPUT_PREFIX = "hydramarker_bottle_frame"

DRAW_ALL_PROJECTED_POINTS = True
DRAW_VISIBLE_INLIER_POINTS = True
DRAW_GLOBAL_IDS = True
DRAW_FRAME = True
DRAW_STATUS_TEXT = True

AXIS_LENGTH_MM = 30.0

POINT_RADIUS_ALL = 4
POINT_RADIUS_VISIBLE = 5

REALSENSE_WIDTH = 1920
REALSENSE_HEIGHT = 1080
REALSENSE_FPS = 30


def choose_file_qt(title: str, file_filter: str) -> Path:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        file_filter,
    )

    if not path:
        raise RuntimeError(f"No file selected: {title}")

    return Path(path)


def load_camera_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        if "K" in npz:
            K = npz["K"]
        elif "camera_matrix" in npz:
            K = npz["camera_matrix"]
        else:
            raise KeyError("Camera NPZ must contain 'K' or 'camera_matrix'.")

        if "dist" in npz:
            dist = npz["dist"]
        elif "dist_coeffs" in npz:
            dist = npz["dist_coeffs"]
        elif "distortion_coeffs" in npz:
            dist = npz["distortion_coeffs"]
        else:
            dist = np.zeros((5, 1), dtype=np.float64)

    return np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64).reshape(-1, 1)


def load_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not load image: {path}")
    return image


def realsense_intrinsics_to_opencv(intr) -> tuple[np.ndarray, np.ndarray]:
    K = np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    dist = np.asarray(intr.coeffs, dtype=np.float64).reshape(-1, 1)
    return K, dist


def start_realsense():
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        REALSENSE_WIDTH,
        REALSENSE_HEIGHT,
        rs.format.bgr8,
        REALSENSE_FPS,
    )

    profile = pipeline.start(config)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    K, dist = realsense_intrinsics_to_opencv(intr)

    for _ in range(15):
        pipeline.wait_for_frames()

    return pipeline, K, dist


def get_realsense_frame_bgr(pipeline) -> Optional[np.ndarray]:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        return None

    return np.asanyarray(color_frame.get_data()).copy()


def point2_from_cpp(p) -> tuple[float, float]:
    if hasattr(p, "x") and hasattr(p, "y"):
        return float(p.x), float(p.y)

    arr = np.asarray(p, dtype=np.float64).reshape(-1)
    return float(arr[0]), float(arr[1])


def point3_from_cpp(p) -> tuple[float, float, float]:
    if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
        return float(p.x), float(p.y), float(p.z)

    arr = np.asarray(p, dtype=np.float64).reshape(-1)
    return float(arr[0]), float(arr[1]), float(arr[2])


def collect_all_geometry_points(tracker: HydraTracker) -> list[tuple[int, int, np.ndarray]]:
    geometry = tracker.geometry

    rows = list(geometry.corner_rows())
    cols = list(geometry.corner_cols())

    points: list[tuple[int, int, np.ndarray]] = []

    for row in rows:
        for col in cols:
            row_i = int(row)
            col_i = int(col)

            if not geometry.has_corner(row_i, col_i):
                continue

            xyz = np.array(
                point3_from_cpp(geometry.corner_point(row_i, col_i)),
                dtype=np.float64,
            )

            points.append((row_i, col_i, xyz))

    if len(points) < 4:
        raise RuntimeError("Marker geometry contains fewer than 4 points.")

    return points


def project_points(
    points_xyz: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    uv, _ = cv2.projectPoints(
        np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        np.asarray(dist, dtype=np.float64).reshape(-1, 1),
    )
    return uv.reshape(-1, 2)


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise RuntimeError("Cannot normalize near-zero vector.")
    return v / n


def find_geometry_point(
    points: list[tuple[int, int, np.ndarray]],
    row: int,
    col: int,
) -> Optional[np.ndarray]:
    for r, c, xyz in points:
        if int(r) == int(row) and int(c) == int(col):
            return xyz
    return None


def define_marker_frame_points(
    tracker: HydraTracker,
    all_points: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    geometry = tracker.geometry

    origin_row = int(geometry.detectable_origin_row())
    origin_col = int(geometry.detectable_origin_col())

    origin = find_geometry_point(all_points, origin_row, origin_col)

    if origin is not None:
        x_ref = find_geometry_point(all_points, origin_row, origin_col + 1)
        y_ref = find_geometry_point(all_points, origin_row + 1, origin_col)

        if x_ref is not None and y_ref is not None:
            x_axis = normalize(x_ref - origin)
            y_axis_raw = normalize(y_ref - origin)
            z_axis = normalize(np.cross(x_axis, y_axis_raw))
            y_axis = normalize(np.cross(z_axis, x_axis))

            return np.array(
                [
                    origin,
                    origin + AXIS_LENGTH_MM * x_axis,
                    origin + AXIS_LENGTH_MM * y_axis,
                    origin + AXIS_LENGTH_MM * z_axis,
                ],
                dtype=np.float64,
            )

    xyz_all = np.array([xyz for _, _, xyz in all_points], dtype=np.float64)
    origin = xyz_all.mean(axis=0)

    centered = xyz_all - origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    x_axis = normalize(vh[0])
    y_axis_raw = normalize(vh[1])
    z_axis = normalize(np.cross(x_axis, y_axis_raw))
    y_axis = normalize(np.cross(z_axis, x_axis))

    return np.array(
        [
            origin,
            origin + AXIS_LENGTH_MM * x_axis,
            origin + AXIS_LENGTH_MM * y_axis,
            origin + AXIS_LENGTH_MM * z_axis,
        ],
        dtype=np.float64,
    )


def draw_all_projected_geometry_points(
    image: np.ndarray,
    tracker: HydraTracker,
    result: TrackerResult,
    K: np.ndarray,
    dist: np.ndarray,
) -> None:
    all_points = collect_all_geometry_points(tracker)
    xyz = np.array([p[2] for p in all_points], dtype=np.float64)
    uv = project_points(xyz, result.rvec, result.tvec, K, dist)

    for (row, col, _), point_uv in zip(all_points, uv):
        u = int(round(point_uv[0]))
        v = int(round(point_uv[1]))

        if u < -50 or v < -50 or u > image.shape[1] + 50 or v > image.shape[0] + 50:
            continue

        cv2.circle(
            image,
            (u, v),
            POINT_RADIUS_ALL,
            (255, 0, 255),
            -1,
            lineType=cv2.LINE_AA,
        )

        if DRAW_GLOBAL_IDS:
            cv2.putText(
                image,
                f"{row},{col}",
                (u + 5, v - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def draw_visible_inlier_points(image: np.ndarray, result: TrackerResult) -> None:
    for corner in result.corners:
        u = int(round(corner.uv[0]))
        v = int(round(corner.uv[1]))

        cv2.circle(
            image,
            (u, v),
            POINT_RADIUS_VISIBLE,
            (0, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )


def draw_marker_frame(
    image: np.ndarray,
    tracker: HydraTracker,
    result: TrackerResult,
    K: np.ndarray,
    dist: np.ndarray,
) -> None:
    all_points = collect_all_geometry_points(tracker)

    frame_xyz = define_marker_frame_points(
        tracker=tracker,
        all_points=all_points,
    )

    frame_uv = project_points(
        frame_xyz,
        result.rvec,
        result.tvec,
        K,
        dist,
    )

    origin = tuple(np.round(frame_uv[0]).astype(int))
    x_tip = tuple(np.round(frame_uv[1]).astype(int))
    y_tip = tuple(np.round(frame_uv[2]).astype(int))
    z_tip = tuple(np.round(frame_uv[3]).astype(int))

    cv2.circle(image, origin, 7, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    cv2.line(image, origin, x_tip, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    cv2.line(image, origin, y_tip, (0, 255, 0), 3, lineType=cv2.LINE_AA)
    cv2.line(image, origin, z_tip, (255, 0, 0), 3, lineType=cv2.LINE_AA)

    cv2.putText(image, "x", x_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(image, "y", y_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, "z", z_tip, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


def draw_status_text(image: np.ndarray, result: TrackerResult) -> None:
    lines = [
        f"success={result.success} mode={result.mode.value}",
        f"points={result.num_points} inliers={result.num_inliers}",
        f"mean={result.mean_reprojection_error_px:.3f}px max={result.max_reprojection_error_px:.3f}px",
        result.message,
    ]

    x = 20
    y = 30

    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 26


def render_result(
    image_bgr: np.ndarray,
    tracker: HydraTracker,
    result: TrackerResult,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    output = image_bgr.copy()

    if not result.success or result.rvec is None or result.tvec is None:
        cv2.putText(
            output,
            f"Tracking failed: {result.message}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    if DRAW_ALL_PROJECTED_POINTS:
        draw_all_projected_geometry_points(
            output,
            tracker,
            result,
            K,
            dist,
        )

    if DRAW_VISIBLE_INLIER_POINTS:
        draw_visible_inlier_points(output, result)

    if DRAW_FRAME:
        draw_marker_frame(
            output,
            tracker,
            result,
            K,
            dist,
        )

    if DRAW_STATUS_TEXT:
        draw_status_text(output, result)

    return output


def make_output_path(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{OUTPUT_PREFIX}_{timestamp}.png"


def create_tracker(
    field_path: Path,
    marker_geometry_path: Path,
    K: np.ndarray,
    dist: np.ndarray,
) -> HydraTracker:
    config = TrackerConfig(
        enable_debug_prints=True,
        log_to_console=True,
        log_path="hydramarker_frame_renderer.log",
        min_points=8,
        min_inliers=6,
        max_mean_reprojection_error_px=4.0,
        max_max_reprojection_error_px=15.0,
        use_pose_prior=False,
    )

    return HydraTracker(
        field_path=str(field_path),
        marker_json_path=str(marker_geometry_path),
        K=K,
        dist_coeffs=dist,
        config=config,
    )


def run_image_mode(
    field_path: Path,
    marker_geometry_path: Path,
) -> None:
    image_path = choose_file_qt(
        "Select bottle image",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
    )

    camera_path = choose_file_qt(
        "Select camera intrinsics .npz",
        "NPZ files (*.npz)",
    )

    image = load_image_bgr(image_path)
    K, dist = load_camera_npz(camera_path)

    tracker = create_tracker(
        field_path=field_path,
        marker_geometry_path=marker_geometry_path,
        K=K,
        dist=dist,
    )

    result = tracker.process_frame(image)

    rendered = render_result(
        image_bgr=image,
        tracker=tracker,
        result=result,
        K=K,
        dist=dist,
    )

    output_path = make_output_path(image_path.parent)
    cv2.imwrite(str(output_path), rendered)

    cv2.imshow(WINDOW_NAME, rendered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Saved: {output_path}")


def run_realsense_mode(
    field_path: Path,
    marker_geometry_path: Path,
) -> None:
    pipeline = None

    try:
        pipeline, K, dist = start_realsense()

        tracker = create_tracker(
            field_path=field_path,
            marker_geometry_path=marker_geometry_path,
            K=K,
            dist=dist,
        )

        print("RealSense running.")
        print("SPACE: capture + render")
        print("ESC/q: quit")

        while True:
            frame = get_realsense_frame_bgr(pipeline)
            if frame is None:
                continue

            preview = frame.copy()

            cv2.putText(
                preview,
                "SPACE: capture/render | ESC/q: quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            if key == 32:
                tracker.reset()

                result = tracker.process_frame(frame)

                rendered = render_result(
                    image_bgr=frame,
                    tracker=tracker,
                    result=result,
                    K=K,
                    dist=dist,
                )

                output_path = make_output_path(Path.cwd())
                cv2.imwrite(str(output_path), rendered)

                print(f"Saved: {output_path}")
                print(f"success: {result.success}")
                print(f"message: {result.message}")
                print(f"points: {result.num_points}")
                print(f"inliers: {result.num_inliers}")
                print(f"mean error px: {result.mean_reprojection_error_px:.3f}")
                print(f"max error px: {result.max_reprojection_error_px:.3f}")

                cv2.imshow(WINDOW_NAME, rendered)
                cv2.waitKey(0)

    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()


def main() -> None:
    field_path = choose_file_qt(
        "Select HydraMarker .field",
        "HydraMarker field (*.field);;All files (*.*)",
    )

    marker_geometry_path = choose_file_qt(
        "Select marker_geometry_sfm.json",
        "JSON files (*.json)",
    )

    if INPUT_MODE == "image":
        run_image_mode(
            field_path=field_path,
            marker_geometry_path=marker_geometry_path,
        )
    elif INPUT_MODE == "realsense":
        run_realsense_mode(
            field_path=field_path,
            marker_geometry_path=marker_geometry_path,
        )
    else:
        raise ValueError(f"Unknown INPUT_MODE: {INPUT_MODE}")


if __name__ == "__main__":
    main()