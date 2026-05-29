"""
Debug-Version des CheckerboardDetector-Tests.

Visualisierung:
    - BLAU:    Recovery/Debug-Corners
    - GRÜN:    Finale Corners
    - GELB:    Recovery-Corners, die final nicht übernommen werden
    - MAGENTA: Cells exakt aus C++ cell.corner_uv

Controls:
    t       Toggle visualization mode
    d       Toggle debug overlay
    SPACE   Save current frame + visualization
    ESC     Exit
"""

import sys
from pathlib import Path
from datetime import datetime
import csv

import cv2
import numpy as np
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication, QFileDialog

import hydramarker_cpp


# ============================================================
# Output
# ============================================================

OUT_DIR = Path("hydramarker_saved_frames")
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Drift logger
# ============================================================

DRIFT_LOG_PATH = OUT_DIR / "drift_log.csv"
_drift_log_file = None
_drift_log_writer = None
_prev_corners: dict = {}   # {(i,j): (u,v)} from previous frame
_frame_idx = 0

def drift_log_open():
    global _drift_log_file, _drift_log_writer
    _drift_log_file = open(DRIFT_LOG_PATH, "w", newline="")
    _drift_log_writer = csv.writer(_drift_log_file)
    _drift_log_writer.writerow([
        "frame", "n_corners", "n_cells",
        "drift_mean_px", "drift_max_px", "drift_p90_px",
        "n_drifting_gt1.5", "n_drifting_gt3", "n_drifting_gt5",
        "square_med_px", "square_min_px",
    ])

def drift_log_update(det, frame_idx: int):
    global _prev_corners, _drift_log_writer

    if det is None or not det.corners:
        _prev_corners = {}
        return

    curr = {}
    for c in det.corners:
        u, v = get_xy(c.uv)
        curr[(int(c.i), int(c.j))] = (u, v)

    drifts = []
    for key, (u, v) in curr.items():
        if key in _prev_corners:
            pu, pv = _prev_corners[key]
            drifts.append(float(np.hypot(u - pu, v - pv)))

    _prev_corners = curr

    if not drifts:
        return

    drifts = np.array(drifts, dtype=np.float32)
    stats = estimate_square_stats(det)

    if _drift_log_writer:
        _drift_log_writer.writerow([
            frame_idx,
            len(det.corners),
            len(det.cells),
            f"{np.mean(drifts):.2f}",
            f"{np.max(drifts):.2f}",
            f"{np.percentile(drifts, 90):.2f}",
            int(np.sum(drifts > 1.5)),
            int(np.sum(drifts > 3.0)),
            int(np.sum(drifts > 5.0)),
            f"{stats['median']:.1f}" if stats else "",
            f"{stats['min']:.1f}" if stats else "",
        ])
        _drift_log_file.flush()

def drift_log_close():
    if _drift_log_file:
        _drift_log_file.close()


# ============================================================
# Helpers
# ============================================================

def select_file(title: str, file_filter: str) -> Path | None:
    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        file_filter,
    )

    return Path(path) if path else None


def get_xy(p):
    if hasattr(p, "x") and hasattr(p, "y"):
        return float(p.x), float(p.y)

    return float(p[0]), float(p[1])


def estimate_square_stats(det):
    if not det or len(det.corners) < 2:
        return None

    corners = []

    for c in det.corners:
        u, v = get_xy(c.uv)
        corners.append((int(c.i), int(c.j), u, v))

    by_idx = {(i, j): (u, v) for i, j, u, v in corners}
    dists = []

    for i, j, u, v in corners:
        for ni, nj in ((i + 1, j), (i, j + 1)):
            if (ni, nj) in by_idx:
                u2, v2 = by_idx[(ni, nj)]
                dists.append(float(np.hypot(u2 - u, v2 - v)))

    if not dists:
        return None

    dists = np.asarray(dists, dtype=np.float32)

    return {
        "median": float(np.median(dists)),
        "mean": float(np.mean(dists)),
        "min": float(np.min(dists)),
        "max": float(np.max(dists)),
        "n": int(len(dists)),
    }


def draw_corners(
    vis: np.ndarray,
    det,
    color=(0, 255, 0),
    radius=4,
    draw_indices=False,
) -> None:
    for corner in det.corners:
        u, v = get_xy(corner.uv)

        cv2.circle(
            vis,
            (int(round(u)), int(round(v))),
            radius,
            color,
            -1,
            lineType=cv2.LINE_AA,
        )

        if draw_indices:
            cv2.putText(
                vis,
                f"{corner.i},{corner.j}",
                (int(round(u)) + 6, int(round(v)) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )


def draw_cells(
    vis: np.ndarray,
    det,
    draw_indices=True,
) -> None:
    for cell in det.cells:
        pts = []

        for p in cell.corner_uv:
            u, v = get_xy(p)
            pts.append((int(round(u)), int(round(v))))

        polygon = np.array(pts, dtype=np.int32)

        cv2.polylines(
            vis,
            [polygon],
            isClosed=True,
            color=(255, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cu, cv_ = get_xy(cell.center_uv)

        cv2.circle(
            vis,
            (int(round(cu)), int(round(cv_))),
            3,
            (255, 0, 255),
            -1,
            lineType=cv2.LINE_AA,
        )

        if draw_indices:
            cv2.putText(
                vis,
                f"{cell.i},{cell.j}",
                (int(round(cu)) + 5, int(round(cv_)) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )


def count_lost_debug_corners(det, debug_det, max_dist_px=10.0):
    if not debug_det or not det:
        return 0

    final_uvs = []

    for c in det.corners:
        u, v = get_xy(c.uv)
        final_uvs.append((u, v))

    lost = 0

    for c in debug_det.corners:
        u, v = get_xy(c.uv)

        found = any(
            abs(u - fu) < max_dist_px and abs(v - fv) < max_dist_px
            for fu, fv in final_uvs
        )

        if not found:
            lost += 1

    return lost


def draw_status(vis, mode_name, det, debug_det, debug_on) -> None:
    normal_n = len(det.corners) if det else 0
    normal_c = len(det.cells) if det else 0
    debug_n = len(debug_det.corners) if debug_det else 0
    debug_c = len(debug_det.cells) if debug_det else 0

    final_stats = estimate_square_stats(det)
    debug_stats = estimate_square_stats(debug_det)

    line1 = f"mode: {mode_name} | t=toggle d=debug SPACE=save ESC=quit"
    line2 = f"final corners: {normal_n} | final cells: {normal_c}"
    line3 = f"recovery/debug corners: {debug_n} | cells: {debug_c} | debug {'ON' if debug_on else 'OFF'}"

    if final_stats is not None:
        line2 += (
            f" | square med: {final_stats['median']:.1f}px"
            f" min: {final_stats['min']:.1f}px"
            f" max: {final_stats['max']:.1f}px"
        )

    if debug_stats is not None:
        line3 += (
            f" | square med: {debug_stats['median']:.1f}px"
            f" min: {debug_stats['min']:.1f}px"
            f" max: {debug_stats['max']:.1f}px"
        )

    if debug_on and debug_det and det:
        lost = count_lost_debug_corners(det, debug_det)
        line3 += f" | lost in final: {lost}"

    for i, line in enumerate([line1, line2, line3]):
        cv2.putText(
            vis,
            line,
            (20, 35 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def save_current_frame(img, vis, det, debug_det, mode_name, debug_on):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    raw_path = OUT_DIR / f"{stamp}_raw.png"
    vis_path = OUT_DIR / f"{stamp}_vis.png"
    npz_path = OUT_DIR / f"{stamp}_data.npz"

    cv2.imwrite(str(raw_path), img)
    cv2.imwrite(str(vis_path), vis)

    final_corners = []
    final_cells = []
    debug_corners = []
    debug_cells = []

    if det:
        for c in det.corners:
            u, v = get_xy(c.uv)
            final_corners.append([u, v, c.i, c.j])

        for cell in det.cells:
            cu, cv_ = get_xy(cell.center_uv)
            final_cells.append([cu, cv_, cell.i, cell.j])

    if debug_det:
        for c in debug_det.corners:
            u, v = get_xy(c.uv)
            debug_corners.append([u, v, c.i, c.j])

        for cell in debug_det.cells:
            cu, cv_ = get_xy(cell.center_uv)
            debug_cells.append([cu, cv_, cell.i, cell.j])

    final_stats = estimate_square_stats(det)
    debug_stats = estimate_square_stats(debug_det)

    np.savez_compressed(
        npz_path,
        raw_image_bgr=img,
        vis_image_bgr=vis,
        final_corners=np.asarray(final_corners, dtype=np.float32),
        final_cells=np.asarray(final_cells, dtype=np.float32),
        debug_corners=np.asarray(debug_corners, dtype=np.float32),
        debug_cells=np.asarray(debug_cells, dtype=np.float32),
        final_square_median_px=np.asarray(
            final_stats["median"] if final_stats is not None else np.nan,
            dtype=np.float32,
        ),
        final_square_min_px=np.asarray(
            final_stats["min"] if final_stats is not None else np.nan,
            dtype=np.float32,
        ),
        final_square_max_px=np.asarray(
            final_stats["max"] if final_stats is not None else np.nan,
            dtype=np.float32,
        ),
        debug_square_median_px=np.asarray(
            debug_stats["median"] if debug_stats is not None else np.nan,
            dtype=np.float32,
        ),
        debug_square_min_px=np.asarray(
            debug_stats["min"] if debug_stats is not None else np.nan,
            dtype=np.float32,
        ),
        debug_square_max_px=np.asarray(
            debug_stats["max"] if debug_stats is not None else np.nan,
            dtype=np.float32,
        ),
        mode_name=np.asarray(mode_name),
        debug_on=np.asarray(debug_on),
    )

    print("Saved:")
    print(f"  {raw_path}")
    print(f"  {vis_path}")
    print(f"  {npz_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    global _frame_idx
    detector = hydramarker_cpp.CheckerboardDetector()
    debug_detector = hydramarker_cpp.CheckerboardDetector()

    drift_log_open()
    print(f"Drift log: {DRIFT_LOG_PATH}")

    pipe = rs.pipeline()
    cfg = rs.config()

    cfg.enable_stream(
        rs.stream.color,
        1920,
        1080,
        rs.format.bgr8,
        30,
    )

    pipe.start(cfg)

    cv2.namedWindow("det", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "det",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    mode = 0

    mode_names = {
        0: "corners",
        1: "cells",
        2: "corners+cells",
    }

    debug_on = False

    try:
        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            vis = img.copy()

            det = detector.detect(img)
            _frame_idx += 1
            drift_log_update(det, _frame_idx)

            debug_det = None

            if debug_on:
                debug_detector.reset_tracking()
                debug_det = debug_detector.detect(img)

            if debug_on and debug_det:
                draw_corners(
                    vis,
                    debug_det,
                    color=(255, 100, 0),
                    radius=3,
                    draw_indices=False,
                )

                if det:
                    final_uvs = []

                    for c in det.corners:
                        u, v = get_xy(c.uv)
                        final_uvs.append((u, v))

                    for c in debug_det.corners:
                        u, v = get_xy(c.uv)

                        found = any(
                            abs(u - fu) < 10 and abs(v - fv) < 10
                            for fu, fv in final_uvs
                        )

                        if not found:
                            cv2.circle(
                                vis,
                                (int(round(u)), int(round(v))),
                                5,
                                (0, 255, 255),
                                2,
                                lineType=cv2.LINE_AA,
                            )

            if det:
                if mode == 0:
                    draw_corners(
                        vis,
                        det,
                        color=(0, 255, 0),
                        radius=4,
                        draw_indices=False,
                    )

                elif mode == 1:
                    draw_cells(
                        vis,
                        det,
                        draw_indices=True,
                    )

                elif mode == 2:
                    draw_cells(
                        vis,
                        det,
                        draw_indices=True,
                    )

                    draw_corners(
                        vis,
                        det,
                        color=(0, 255, 0),
                        radius=4,
                        draw_indices=True,
                    )

            draw_status(
                vis,
                mode_names[mode],
                det,
                debug_det,
                debug_on,
            )

            cv2.imshow("det", vis)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            elif key == ord("t"):
                mode = (mode + 1) % 3

            elif key == ord("d"):
                debug_on = not debug_on
                debug_detector.reset_tracking()

            elif key == ord(" "):
                save_current_frame(
                    img=img,
                    vis=vis,
                    det=det,
                    debug_det=debug_det,
                    mode_name=mode_names[mode],
                    debug_on=debug_on,
                )

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        drift_log_close()
        print(f"Drift log saved to {DRIFT_LOG_PATH}")


if __name__ == "__main__":
    main()