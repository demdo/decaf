from __future__ import annotations

import cv2
import numpy as np
import pyrealsense2 as rs

from dataclasses import replace

from overlay.calib.calib_camera_to_pointer import (
    calibrate_camera_to_pointer,
    get_default_pointer_tool_model,
)
from tracking.pose_filters import AdaptiveKalmanFilterCV3D


# ============================================================
# USER SETTINGS
# ============================================================

K_RGB = np.array(
    [
        [1.34866392e+03, 0.00000000e+00, 9.76874392e+02],
        [0.00000000e+00, 1.35308666e+03, 5.58629717e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
    ],
    dtype=np.float64,
)

DIST_COEFFS = None
WINDOW_NAME = "debug_pointer_tip"


# ============================================================
# Drawing
# ============================================================

def draw_point(img, uv, color):
    uv = np.asarray(uv).reshape(2)
    if not np.isfinite(uv).all():
        return

    u, v = np.round(uv).astype(int)

    cv2.circle(img, (u, v), 8, color, 2, cv2.LINE_AA)
    cv2.drawMarker(
        img,
        (u, v),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2,
    )


# ============================================================
# Filter
# ============================================================

def filter_pointer_result(raw_result, tip_kf, K_rgb):
    tip_xyz_f_mm = tip_kf.filter(
        measurement_mm=raw_result.tip_point_camera_mm,
        rotation_camera=raw_result.rotation,
    ).reshape(3)

    tvec_f = tip_xyz_f_mm.reshape(3, 1)

    dist = np.zeros((5, 1), dtype=np.float64)

    tip_uv_f, _ = cv2.projectPoints(
        np.zeros((1, 3), dtype=np.float64),
        raw_result.rvec,
        tvec_f,
        K_rgb,
        dist,
    )

    return replace(
        raw_result,
        tvec=tvec_f,
        translation=tvec_f.copy(),
        tip_point_camera_mm=tip_xyz_f_mm,
        tip_uv=tip_uv_f.reshape(2),
    )


# ============================================================
# RealSense
# ============================================================

def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


def get_frame(pipeline):
    frames = pipeline.poll_for_frames()
    if not frames:
        return None
    color = frames.get_color_frame()
    if not color:
        return None
    return np.asanyarray(color.get_data())


# ============================================================
# Main
# ============================================================

def main():
    pointer_model = get_default_pointer_tool_model()

    tip_kf = AdaptiveKalmanFilterCV3D(dt=1/30)

    saved_points = []  # max 2, reset after second

    pipeline = start_realsense()

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            img = get_frame(pipeline)
            if img is None:
                continue

            vis = img.copy()

            current_result = None

            try:
                raw = calibrate_camera_to_pointer(
                    image_bgr=img,
                    camera_intrinsics=K_RGB,
                    dist_coeffs=DIST_COEFFS,
                    pointer_model=pointer_model,
                    pose_method="iterative_ransac",  # wichtig!
                    refine_with_iterative=True,
                )

                current_result = filter_pointer_result(raw, tip_kf, K_RGB)

            except Exception:
                current_result = None

            # gespeicherte Punkte zeichnen
            colors = [(0, 255, 255), (255, 255, 0)]

            for i, p in enumerate(saved_points):
                draw_point(vis, p, colors[i])

            cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            elif key == 32:  # SPACE
                if current_result is not None:
                    uv = current_result.tip_uv.copy()

                    if len(saved_points) < 2:
                        saved_points.append(uv)
                    else:
                        # reset → neuer Zyklus
                        saved_points = [uv]

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()