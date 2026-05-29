from __future__ import annotations

import csv
import time
from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.hydramarker.tracker import HydraTracker, TrackerConfig


# ============================================================
# Frame Logger
# ============================================================

LOG_PATH = Path("hydramarker_frame_log.csv")

COLUMNS = [
    "frame", "wall_ms", "mode", "success", "message",
    "det_valid", "det_tracking", "det_stable", "det_corners",
    "pose_corners", "num_points", "num_inliers",
    "mean_err", "max_err", "confidence",
    "persistent_count",
]

_log_file = None
_log_writer = None


def log_open():
    global _log_file, _log_writer
    _log_file = open(LOG_PATH, "w", newline="", encoding="utf-8")
    _log_writer = csv.DictWriter(_log_file, fieldnames=COLUMNS)
    _log_writer.writeheader()
    _log_file.flush()
    print(f"[frame_log] {LOG_PATH.resolve()}")


def log_frame(frame_idx: int, result, wall_ms: float, tracker: HydraTracker):
    if _log_writer is None:
        return

    _log_writer.writerow({
        "frame":            frame_idx,
        "wall_ms":          f"{wall_ms:.1f}",
        "mode":             result.mode.value,
        "success":          int(result.success),
        "message":          result.message,
        "det_valid":        int(result.detection_valid),
        "det_tracking":     int(result.detection_tracking),
        "det_stable":       int(result.detection_stable),
        "det_corners":      len(getattr(result, "detection_corners", [])),
        "pose_corners":     len(result.corners),
        "num_points":       result.num_points,
        "num_inliers":      result.num_inliers,
        "mean_err":         f"{result.mean_reprojection_error_px:.3f}",
        "max_err":          f"{result.max_reprojection_error_px:.3f}",
        "confidence":       f"{result.confidence:.3f}",
        "persistent_count": len(tracker._persistent_corners),
    })
    _log_file.flush()


def log_close():
    if _log_file and not _log_file.closed:
        _log_file.close()
        print(f"[frame_log] closed → {LOG_PATH.resolve()}")


# ============================================================
# Helpers
# ============================================================

def choose_file_qt(title: str, file_filter: str) -> Path:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(None, title, "", file_filter)
    if not path:
        raise RuntimeError(f"No file selected: {title}")

    return Path(path)


def put_text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int] = (0, 255, 255),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def realsense_intrinsics_to_cv(profile) -> tuple[np.ndarray, np.ndarray]:
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    K = np.array(
        [[intr.fx, 0.0, intr.ppx],
         [0.0, intr.fy, intr.ppy],
         [0.0, 0.0, 1.0]],
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


def draw_detection_corners(vis: np.ndarray, result) -> None:
    for p in getattr(result, "detection_corners", []):
        u = int(round(p.uv[0]))
        v = int(round(p.uv[1]))
        cv2.circle(vis, (u, v), 3, (255, 180, 0), -1, cv2.LINE_AA)


def draw_pose_corners(vis: np.ndarray, result) -> None:
    if not result.success:
        return
    for p in result.corners:
        u = int(round(p.uv[0]))
        v = int(round(p.uv[1]))
        cv2.circle(vis, (u, v), 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(vis, f"{p.global_row},{p.global_col}",
                    (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 255, 0), 1, cv2.LINE_AA)


def draw_reprojection(vis: np.ndarray, result, K, dist) -> None:
    if not result.success or result.rvec is None or result.tvec is None:
        return
    if len(result.corners) == 0:
        return

    object_points = np.asarray([p.xyz_mm for p in result.corners], dtype=np.float64).reshape(-1, 3)
    measured = np.asarray([p.uv for p in result.corners], dtype=np.float64).reshape(-1, 2)

    projected, _ = cv2.projectPoints(
        object_points,
        result.rvec.reshape(3, 1),
        result.tvec.reshape(3, 1),
        K, dist,
    )
    projected = projected.reshape(-1, 2)

    for m, q in zip(measured, projected):
        cv2.circle(vis, (int(round(q[0])), int(round(q[1]))), 3, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.line(vis, (int(round(m[0])), int(round(m[1]))),
                      (int(round(q[0])), int(round(q[1]))), (255, 0, 255), 1, cv2.LINE_AA)


def draw_pose_axes(vis, result, K, dist, axis_length_mm=30.0) -> None:
    if not result.success or result.rvec is None or result.tvec is None:
        return
    try:
        cv2.drawFrameAxes(vis, K, dist,
                          result.rvec.reshape(3, 1),
                          result.tvec.reshape(3, 1),
                          float(axis_length_mm))
    except cv2.error:
        return


def draw_status(vis: np.ndarray, result, frame_idx: int, tracker: HydraTracker) -> None:
    detection_corners = getattr(result, "detection_corners", [])
    status_color = (0, 255, 0) if result.success else (0, 165, 255)

    line1 = (
        f"frame={frame_idx} | {result.mode.value} | ok={result.success} | "
        f"det={len(detection_corners)} | pose={len(result.corners)} | "
        f"pts={result.num_points} | inl={result.num_inliers} | "
        f"pers={len(tracker._persistent_corners)}"
    )
    line2 = (
        f"mean={result.mean_reprojection_error_px:.3f}px | "
        f"max={result.max_reprojection_error_px:.3f}px | "
        f"conf={result.confidence:.2f}"
    )

    put_text(vis, line1, (25, 35), color=status_color, scale=0.55)
    put_text(vis, line2, (25, 65), color=status_color, scale=0.50)
    put_text(vis, f"msg: {result.message}", (25, 95), color=(0, 255, 255), scale=0.46)
    put_text(vis,
             "blue=detector | green=global corr | magenta=reprojection | r=reset | q=quit",
             (25, 125), color=(255, 180, 0), scale=0.46)


def draw_debug(vis, result, K, dist, frame_idx, tracker) -> np.ndarray:
    draw_detection_corners(vis, result)
    draw_pose_corners(vis, result)
    draw_reprojection(vis, result, K, dist)
    draw_status(vis, result, frame_idx, tracker)
    return vis


def log_console(frame_idx: int, result, tracker, *, force: bool = False) -> None:
    if not force and result.success and frame_idx % 30 != 0:
        return

    print(
        "[test_tracker]",
        f"frame={frame_idx}",
        f"mode={result.mode.value}",
        f"success={result.success}",
        f"msg={result.message}",
        f"det={len(getattr(result, 'detection_corners', []))}",
        f"pose={len(result.corners)}",
        f"pts={result.num_points}",
        f"inl={result.num_inliers}",
        f"mean={result.mean_reprojection_error_px:.3f}",
        f"pers={len(tracker._persistent_corners)}",
    )


def make_tracker(field_path, marker_json_path, K, dist) -> HydraTracker:
    return HydraTracker(
        field_path=str(field_path),
        marker_json_path=str(marker_json_path),
        K=K,
        dist_coeffs=dist,
        config=TrackerConfig(
            min_points=6,               # 8 war zu streng beim Startup
            min_inliers=5,
            max_mean_reprojection_error_px=4.0,
            max_max_reprojection_error_px=15.0,
            max_lost_frames=8,
            max_translation_jump_mm=120.0,
            max_rotation_jump_deg=45.0,
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
            enable_debug_prints=True,
            log_path="hydramarker_tracker.log",
            log_to_console=False,
            dot_commit_frames=1,               # Warmup nach Bewegung: 1 Frame statt 2
            dot_revoke_frames=5,               # asymmetrisch: langsam revoken
            persistence_max_frames=8,          # mehr Zeit fuer Dot Detector Warmup
        ),
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    field_path = choose_file_qt("Select HydraMarker .field file", "HydraMarker field (*.field)")
    marker_json_path = choose_file_qt("Select marker .json file", "Marker JSON (*.json)")

    pipe, profile = create_realsense_pipeline()
    K_rgb, dist_rgb = realsense_intrinsics_to_cv(profile)

    tracker = make_tracker(field_path, marker_json_path, K_rgb, dist_rgb)

    log_open()

    window_name = "HydraTracker RealSense Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    last_mode: Optional[str] = None
    last_success: Optional[bool] = None
    last_message: Optional[str] = None

    try:
        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_idx += 1
            frame = np.asanyarray(color_frame.get_data())

            t0 = time.perf_counter()
            result = tracker.process_frame(frame)
            wall_ms = (time.perf_counter() - t0) * 1000.0

            # CSV log — every frame
            log_frame(frame_idx, result, wall_ms, tracker)

            # Console log — only on state changes or failures
            mode_changed    = last_mode    != result.mode.value
            success_changed = last_success != bool(result.success)
            message_changed = last_message != result.message
            force_log = mode_changed or success_changed or message_changed or not result.success
            log_console(frame_idx, result, tracker, force=force_log)

            last_mode    = result.mode.value
            last_success = bool(result.success)
            last_message = result.message

            vis = draw_debug(frame.copy(), result, K_rgb, dist_rgb, frame_idx, tracker)
            cv2.imshow(window_name, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                tracker.reset()
                last_mode = last_success = last_message = None
                print("[test_tracker] reset")

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        log_close()


if __name__ == "__main__":
    main()