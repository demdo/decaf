from pathlib import Path
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
)

from overlay.tracking.hydramarker.tracker import (
    HydraTracker,
)


def choose_file_qt(title, file_filter):
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
        raise RuntimeError(
            f"No file selected: {title}"
        )

    return Path(path)


def put_text(
    img,
    text,
    pos,
    color=(0, 255, 255),
    scale=0.55,
    thickness=1,
):
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


def realsense_intrinsics_to_cv(profile):
    color_stream = profile.get_stream(
        rs.stream.color
    ).as_video_stream_profile()

    intr = color_stream.get_intrinsics()

    K = np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    dist = np.asarray(
        intr.coeffs[:5],
        dtype=np.float64,
    ).reshape(-1, 1)

    return K, dist, intr


def draw_debug(
    vis,
    result,
    K,
    dist,
):
    color = (
        (0, 255, 0)
        if result.success
        else (0, 0, 255)
    )

    for p in result.corners:
        u = int(round(p.uv[0]))
        v = int(round(p.uv[1]))

        cv2.circle(
            vis,
            (u, v),
            5,
            color,
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            vis,
            f"{p.global_row},{p.global_col}",
            (u + 5, v - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    if (
        result.success
        and result.rvec is not None
        and result.tvec is not None
    ):
        cv2.drawFrameAxes(
            vis,
            K,
            dist,
            result.rvec.reshape(3, 1),
            result.tvec.reshape(3, 1),
            40.0,
        )

    text = (
        f"{result.mode.value} | "
        f"pts={result.num_points} | "
        f"inl={result.num_inliers} | "
        f"err={result.mean_reprojection_error_px:.2f}px | "
        f"{result.message}"
    )

    put_text(
        vis,
        text,
        (25, 35),
    )

    return vis


def create_realsense_pipeline():
    pipe = rs.pipeline()

    cfg = rs.config()

    cfg.enable_stream(
        rs.stream.color,
        1920,
        1080,
        rs.format.bgr8,
        30,
    )

    profile = pipe.start(cfg)

    return pipe, profile


def main():

    field_path = choose_file_qt(
        "Select HydraMarker .field file",
        "HydraMarker field (*.field)",
    )

    marker_json_path = choose_file_qt(
        "Select planar marker .json file",
        "Marker JSON (*.json)",
    )

    pipe, profile = (
        create_realsense_pipeline()
    )

    K_rgb, dist_rgb, intr = (
        realsense_intrinsics_to_cv(profile)
    )

    tracker = HydraTracker(
        field_path=str(field_path),
        marker_json_path=str(marker_json_path),
        K=K_rgb,
        dist_coeffs=dist_rgb,
    )

    window_name = (
        "HydraTracker RealSense Test"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    try:
        while True:

            frames = pipe.wait_for_frames()

            color_frame = (
                frames.get_color_frame()
            )

            if not color_frame:
                continue

            frame = np.asanyarray(
                color_frame.get_data()
            )

            result = tracker.process_frame(
                frame
            )

            vis = frame.copy()

            vis = draw_debug(
                vis,
                result,
                K_rgb,
                dist_rgb,
            )

            cv2.imshow(
                window_name,
                vis,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            elif key == ord("r"):
                tracker.reset()

    finally:
        pipe.stop()
        cv2.destroyAllWindows()