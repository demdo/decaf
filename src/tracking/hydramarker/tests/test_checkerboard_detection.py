"""
Debug-Version des CheckerboardDetector-Tests.

Visualisierung:
    - GRÜN:    Finale Corners
    - MAGENTA: Cells exakt aus C++ cell.corner_uv
    - BLAU:    Recovery/Debug-Corners
    - GELB:    Recovery-Corners, die final nicht übernommen werden

Controls:
    t       Toggle visualization mode
    d       Toggle debug overlay
    l       Toggle flicker diagnostic logging (start/stop)
    SPACE   Save current frame + visualization
    ESC     Exit

Flicker log:
    Eine CSV-Datei pro Logging-Session unter hydramarker_saved_frames/.
    Jede Zeile = ein Corner in einem Frame.
    Fehlende Corners (flackern) sind als Lücken in (frame x (i,j)) sichtbar.
    Analyse-Tipp (pandas):
        df = pd.read_csv("flicker_*.csv")
        # Pivot: Zeilen=frame, Spalten=(i,j), Wert=present (0/1)
        pivot = df.pivot_table(index="frame", columns=["i","j"],
                               values="present", fill_value=0)
        # Flicker: frame N fehlt, N-1 und N+1 vorhanden
        flickering = ((pivot == 0) &
                      (pivot.shift(1) == 1) &
                      (pivot.shift(-1) == 1))
"""

import sys
from pathlib import Path
from datetime import datetime
import csv

import cv2
import numpy as np
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.hydramarker.backend import cpp_impl as hydramarker_cpp


# ============================================================
# Output
# ============================================================

OUT_DIR = Path("hydramarker_saved_frames")
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Flicker diagnostic logger
# ============================================================

_log_active    = False
_log_file      = None
_log_writer    = None
_log_start_ms  = 0.0
_log_frame_idx = 0


def log_start() -> None:
    global _log_active, _log_file, _log_writer, _log_start_ms, _log_frame_idx
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path  = OUT_DIR / f"flicker_{stamp}.csv"
    _log_file   = open(path, "w", newline="", encoding="utf-8")
    _log_writer = csv.writer(_log_file)
    _log_writer.writerow([
        "frame", "timestamp_ms",
        "n_corners", "n_cells", "tracking", "stable",
        "spacing_median", "spacing_min",
        "i", "j", "u", "v", "visibility", "present",
    ])
    _log_start_ms  = datetime.now().timestamp() * 1000.0
    _log_frame_idx = 0
    _log_active    = True
    print(f"[LOG] Started -> {path}")


def log_stop() -> None:
    global _log_active, _log_file
    _log_active = False
    if _log_file:
        _log_file.close()
        _log_file = None
    print(f"[LOG] Stopped - {_log_frame_idx} frames logged")


def log_frame(det, global_frame: int) -> None:
    global _log_frame_idx

    if not _log_active or _log_writer is None:
        return

    now_ms = datetime.now().timestamp() * 1000.0 - _log_start_ms

    n_corners = len(det.corners) if det else 0
    n_cells   = len(det.cells)   if det else 0
    tracking  = int(det.tracking) if det else 0
    stable    = int(det.stable)   if det else 0

    stats          = estimate_square_stats(det)
    spacing_median = f"{stats['median']:.2f}" if stats else "-1"
    spacing_min    = f"{stats['min']:.2f}"    if stats else "-1"

    if det and det.corners:
        for c in det.corners:
            u, v   = get_xy(c.uv)
            vscore = float(getattr(c, "visibility_score", 1.0))
            _log_writer.writerow([
                global_frame,
                f"{now_ms:.1f}",
                n_corners, n_cells, tracking, stable,
                spacing_median, spacing_min,
                int(c.i), int(c.j),
                f"{u:.2f}", f"{v:.2f}",
                f"{vscore:.4f}",
                1,
            ])
    else:
        # No detection — one summary row so the frame is represented.
        _log_writer.writerow([
            global_frame, f"{now_ms:.1f}",
            0, 0, 0, 0,
            spacing_median, spacing_min,
            "", "", "", "", "", 0,
        ])

    _log_file.flush()
    _log_frame_idx += 1


# ============================================================
# Helpers
# ============================================================

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
    dists  = []

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
        "mean":   float(np.mean(dists)),
        "min":    float(np.min(dists)),
        "max":    float(np.max(dists)),
        "n":      int(len(dists)),
    }


def draw_corners(vis, det, color=(0, 255, 0), radius=4,
                 draw_indices=False) -> None:
    for corner in det.corners:
        u, v = get_xy(corner.uv)
        cv2.circle(vis, (int(round(u)), int(round(v))),
                   radius, color, -1, lineType=cv2.LINE_AA)
        if draw_indices:
            cv2.putText(vis, f"{corner.i},{corner.j}",
                        (int(round(u)) + 6, int(round(v)) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
                        cv2.LINE_AA)


def draw_cells(vis, det, draw_indices=True) -> None:
    for cell in det.cells:
        pts = []
        for p in cell.corner_uv:
            u, v = get_xy(p)
            pts.append((int(round(u)), int(round(v))))
        polygon = np.array(pts, dtype=np.int32)
        cv2.polylines(vis, [polygon], isClosed=True,
                      color=(255, 0, 255), thickness=2,
                      lineType=cv2.LINE_AA)
        cu, cv_ = get_xy(cell.center_uv)
        cv2.circle(vis, (int(round(cu)), int(round(cv_))),
                   3, (255, 0, 255), -1, lineType=cv2.LINE_AA)
        if draw_indices:
            cv2.putText(vis, f"{cell.i},{cell.j}",
                        (int(round(cu)) + 5, int(round(cv_)) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 0, 255), 1, cv2.LINE_AA)


def count_lost_debug_corners(det, debug_det, max_dist_px=10.0):
    if not debug_det or not det:
        return 0
    final_uvs = [(get_xy(c.uv)) for c in det.corners]
    lost = 0
    for c in debug_det.corners:
        u, v  = get_xy(c.uv)
        found = any(abs(u - fu) < max_dist_px and abs(v - fv) < max_dist_px
                    for fu, fv in final_uvs)
        if not found:
            lost += 1
    return lost


def draw_status(vis, mode_name, det, debug_det, debug_on) -> None:
    normal_n = len(det.corners) if det else 0
    normal_c = len(det.cells)   if det else 0
    debug_n  = len(debug_det.corners) if debug_det else 0
    debug_c  = len(debug_det.cells)   if debug_det else 0

    final_stats = estimate_square_stats(det)
    debug_stats = estimate_square_stats(debug_det)

    log_indicator = " | [LOG]" if _log_active else ""
    line1 = (f"mode: {mode_name} | t=toggle d=debug "
             f"l=log SPACE=save ESC=quit{log_indicator}")
    line2 = f"final corners: {normal_n} | final cells: {normal_c}"
    line3 = (f"recovery/debug corners: {debug_n} | cells: {debug_c} "
             f"| debug {'ON' if debug_on else 'OFF'}")

    if final_stats is not None:
        line2 += (f" | square med: {final_stats['median']:.1f}px"
                  f" min: {final_stats['min']:.1f}px"
                  f" max: {final_stats['max']:.1f}px")

    if debug_stats is not None:
        line3 += (f" | square med: {debug_stats['median']:.1f}px"
                  f" min: {debug_stats['min']:.1f}px"
                  f" max: {debug_stats['max']:.1f}px")

    if debug_on and debug_det and det:
        lost = count_lost_debug_corners(det, debug_det)
        line3 += f" | lost in final: {lost}"

    for i, line in enumerate([line1, line2, line3]):
        cv2.putText(vis, line, (20, 35 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)


def save_current_frame(img, vis, det, debug_det, mode_name, debug_on):
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    raw_path = OUT_DIR / f"{stamp}_raw.png"
    vis_path = OUT_DIR / f"{stamp}_vis.png"
    cv2.imwrite(str(raw_path), img)
    cv2.imwrite(str(vis_path), vis)
    print(f"Saved: {raw_path}  {vis_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    global _log_frame_idx

    detector       = hydramarker_cpp.CheckerboardDetector()
    debug_detector = hydramarker_cpp.CheckerboardDetector()

    frame_idx = 0

    pipe = rs.pipeline()
    cfg  = rs.config()
    cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    pipe.start(cfg)

    cv2.namedWindow("det", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("det", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    mode       = 0
    mode_names = {0: "corners", 1: "cells", 2: "corners+cells"}
    debug_on   = False

    try:
        while True:
            frames      = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            vis = img.copy()
            frame_idx += 1

            det = detector.detect(img)

            # Flicker log
            log_frame(det, frame_idx)

            # Debug overlay
            debug_det = None
            if debug_on:
                debug_detector.reset_tracking()
                debug_det = debug_detector.detect(img)

            if debug_on and debug_det:
                draw_corners(vis, debug_det, color=(255, 100, 0),
                             radius=3, draw_indices=False)
                if det:
                    final_uvs = [get_xy(c.uv) for c in det.corners]
                    for c in debug_det.corners:
                        u, v  = get_xy(c.uv)
                        found = any(abs(u - fu) < 10 and abs(v - fv) < 10
                                    for fu, fv in final_uvs)
                        if not found:
                            cv2.circle(vis,
                                       (int(round(u)), int(round(v))),
                                       5, (0, 255, 255), 2,
                                       lineType=cv2.LINE_AA)

            if det:
                if mode == 0:
                    draw_corners(vis, det, color=(0, 255, 0),
                                 radius=4, draw_indices=False)
                elif mode == 1:
                    draw_cells(vis, det, draw_indices=True)
                elif mode == 2:
                    draw_cells(vis, det, draw_indices=True)
                    draw_corners(vis, det, color=(0, 255, 0),
                                 radius=4, draw_indices=True)

            draw_status(vis, mode_names[mode], det, debug_det, debug_on)

            # Logging indicator
            if _log_active:
                cv2.circle(vis, (vis.shape[1] - 30, 30), 12,
                           (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(vis, "LOG",
                            (vis.shape[1] - 80, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("det", vis)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == ord("t"):
                mode = (mode + 1) % 3
            elif key == ord("d"):
                debug_on = not debug_on
                debug_detector.reset_tracking()
            elif key == ord("l"):
                if not _log_active:
                    log_start()
                else:
                    log_stop()
            elif key == ord(" "):
                save_current_frame(img, vis, det, debug_det,
                                   mode_names[mode], debug_on)

    finally:
        if _log_active:
            log_stop()
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()