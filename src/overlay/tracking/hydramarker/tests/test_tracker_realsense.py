from pathlib import Path
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

from PySide6.QtWidgets import QApplication, QFileDialog

from hydramarker.tracker import HydraTracker


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
        raise RuntimeError(f"No file selected: {title}")

    return Path(path)


def put_text(img, text, pos, color=(0, 255, 255), scale=0.55, thickness=1):
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

    if intr.coeffs is not None and len(intr.coeffs) >= 5:
        dist = np.asarray(
            intr.coeffs[:5],
            dtype=np.float64,
        ).reshape(-1, 1)
    else:
        dist = np.zeros((5, 1), dtype=np.float64)

    return K, dist, intr


def draw_origin_and_global_axes(
    vis,
    result,
    K,
    dist,
    axis_len_mm=50.0,
):
    if not result.success:
        return vis

    if result.rvec is None or result.tvec is None:
        return vis

    pts_3d = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_len_mm, 0.0, 0.0],
            [0.0, axis_len_mm, 0.0],
        ],
        dtype=np.float32,
    )

    pts_2d, _ = cv2.projectPoints(
        pts_3d,
        result.rvec.reshape(3, 1),
        result.tvec.reshape(3, 1),
        K,
        dist,
    )

    pts_2d = pts_2d.reshape(-1, 2)

    o = tuple(np.round(pts_2d[0]).astype(int))
    x = tuple(np.round(pts_2d[1]).astype(int))
    y = tuple(np.round(pts_2d[2]).astype(int))

    cv2.circle(
        vis,
        o,
        9,
        (0, 0, 255),
        -1,
        cv2.LINE_AA,
    )

    put_text(
        vis,
        "ORIGIN",
        (o[0] + 10, o[1] - 10),
        (0, 0, 255),
        0.65,
        2,
    )

    cv2.arrowedLine(
        vis,
        o,
        x,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
        tipLength=0.15,
    )

    cv2.arrowedLine(
        vis,
        o,
        y,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
        tipLength=0.15,
    )

    put_text(
        vis,
        "X",
        x,
        (0, 0, 255),
        0.8,
        2,
    )

    put_text(
        vis,
        "Y",
        y,
        (0, 255, 0),
        0.8,
        2,
    )

    return vis


def draw_status_panel(vis, result):
    lines = [
        f"mode: {result.mode.value}",
        f"success: {int(result.success)}",
        f"detection_valid: {int(result.detection_valid)}",
        f"detection_tracking: {int(result.detection_tracking)}",
        f"detection_stable: {int(result.detection_stable)}",
        f"points: {result.num_points}",
        f"inliers: {result.num_inliers}",
        f"mean reproj: {result.mean_reprojection_error_px:.2f}px",
        f"max reproj: {result.max_reprojection_error_px:.2f}px",
        f"confidence: {result.confidence:.2f}",
        f"message: {result.message}",
    ]

    x = 25
    y = 35
    dy = 24

    for i, line in enumerate(lines):
        put_text(
            vis,
            line,
            (x, y + i * dy),
            (0, 255, 255),
            0.55,
            1,
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
        "HydraMarker field (*.field);;All files (*.*)",
    )

    marker_json_path = choose_file_qt(
        "Select planar marker .json file",
        "Marker JSON (*.json);;All files (*.*)",
    )

    pipe, profile = create_realsense_pipeline()

    K_rgb, dist_rgb, intr = realsense_intrinsics_to_cv(profile)

    print("Selected field:")
    print(field_path)
    print()
    print("Selected marker JSON:")
    print(marker_json_path)
    print()
    print("RealSense color intrinsics:")
    print(f"fx  = {intr.fx}")
    print(f"fy  = {intr.fy}")
    print(f"ppx = {intr.ppx}")
    print(f"ppy = {intr.ppy}")
    print()
    print("K_rgb:")
    print(K_rgb)
    print()
    print("dist_rgb:")
    print(dist_rgb.ravel())
    print()

    tracker = HydraTracker(
        field_path=str(field_path),
        marker_json_path=str(marker_json_path),
        K=K_rgb,
        dist_coeffs=dist_rgb,
    )

    window_name = "HydraTracker RealSense Test"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    output_dir = (
        Path(__file__).resolve().parent
        / "tracker_realsense_snapshots"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paused = False
    last_vis = None
    screenshot_idx = 0

    print("Controls:")
    print("  ESC / q   quit")
    print("  r         reset tracker")
    print("  p         pause")
    print("  SPACE     save screenshot")
    print()

    try:
        while True:
            if not paused:
                frames = pipe.wait_for_frames()

                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                frame = np.asanyarray(
                    color_frame.get_data()
                )

                result = tracker.process_frame(
                    frame
                )

                vis = tracker.draw_debug(
                    frame,
                    result,
                    draw_ids=True,
                    draw_axes=True,
                )

                vis = draw_origin_and_global_axes(
                    vis,
                    result,
                    K_rgb,
                    dist_rgb,
                    axis_len_mm=50.0,
                )

                vis = draw_status_panel(
                    vis,
                    result,
                )

                put_text(
                    vis,
                    "ORIGIN should stay fixed in GLOBAL marker coordinates, even when not visible",
                    (25, vis.shape[0] - 55),
                    (255, 255, 255),
                    0.65,
                    2,
                )

                put_text(
                    vis,
                    "SPACE save | p pause | r reset | ESC/q quit",
                    (25, vis.shape[0] - 25),
                    (255, 255, 255),
                    0.65,
                    2,
                )

                last_vis = vis

            if last_vis is not None:
                cv2.imshow(
                    window_name,
                    last_vis,
                )

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            if key == ord("r"):
                tracker.reset()
                print("Tracker reset.")

            elif key == ord("p"):
                paused = not paused
                print("Paused:", paused)

            elif key == 32 and last_vis is not None:
                out_path = output_dir / f"tracker_realsense_{screenshot_idx:04d}.png"

                cv2.imwrite(
                    str(out_path),
                    last_vis,
                )

                print("Saved screenshot:", out_path)

                screenshot_idx += 1

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()