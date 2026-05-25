from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

import hydramarker_cpp


# ============================================================
# Config
# ============================================================

RECORD_SECONDS = 3.0
MAX_KEYFRAMES = 12

OUT_DIR = Path("hydramarker_debug_runs")
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def get_xy(p):
    if hasattr(p, "x") and hasattr(p, "y"):
        return float(p.x), float(p.y)
    return float(p[0]), float(p[1])


def draw_detection(vis, det, color=(0, 255, 0), draw_cells=True):
    if det is None:
        return

    if draw_cells:
        for cell in det.cells:
            pts = []

            for p in cell.corner_uv:
                u, v = get_xy(p)
                pts.append((int(round(u)), int(round(v))))

            if len(pts) == 4:
                cv2.polylines(
                    vis,
                    [np.array(pts, dtype=np.int32)],
                    True,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

    for c in det.corners:
        u, v = get_xy(c.uv)
        cv2.circle(
            vis,
            (int(round(u)), int(round(v))),
            4,
            color,
            -1,
            cv2.LINE_AA,
        )


def put_text(vis, lines):
    for i, line in enumerate(lines):
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


def detection_counts(det):
    if det is None:
        return 0, 0, False
    return len(det.corners), len(det.cells), True


def grab_realsense_frame(pipe):
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        return None

    return np.asanyarray(color_frame.get_data())


def select_keyframe_indices(metrics):
    if len(metrics) == 0:
        return []

    idxs = set()
    n = len(metrics)

    idxs.add(0)
    idxs.add(n - 1)

    recovery_corners = metrics[:, 2]
    recovery_cells = metrics[:, 3]
    live_corners = metrics[:, 5]
    live_cells = metrics[:, 6]

    diff_corners = recovery_corners - live_corners
    diff_cells = recovery_cells - live_cells

    scores = diff_corners + diff_cells
    order = np.argsort(scores)[::-1]

    for i in order[:MAX_KEYFRAMES]:
        idxs.add(int(i))

    idxs = sorted(idxs)

    if len(idxs) > MAX_KEYFRAMES:
        lin = np.linspace(0, len(idxs) - 1, MAX_KEYFRAMES)
        idxs = [idxs[int(round(i))] for i in lin]

    return idxs


def diagnose(metrics):
    if len(metrics) == 0:
        return "No frames recorded."

    recovery_corners = metrics[:, 2]
    recovery_cells = metrics[:, 3]
    live_corners = metrics[:, 5]
    live_cells = metrics[:, 6]

    diff_corners = recovery_corners - live_corners
    diff_cells = recovery_cells - live_cells
    mismatch_score = diff_corners + diff_cells

    worst_idx = int(np.argmax(mismatch_score))
    worst_recovery_idx = int(np.argmin(recovery_corners + recovery_cells))

    lines = []
    lines.append("HydraMarker debug diagnosis")
    lines.append("=" * 60)
    lines.append(f"frames: {len(metrics)}")
    lines.append("")
    lines.append(f"mean recovery corners: {np.mean(recovery_corners):.1f}")
    lines.append(f"mean live corners:     {np.mean(live_corners):.1f}")
    lines.append(f"mean recovery cells:   {np.mean(recovery_cells):.1f}")
    lines.append(f"mean live cells:       {np.mean(live_cells):.1f}")
    lines.append("")
    lines.append(f"max corner gap recovery-live: {np.max(diff_corners):.1f}")
    lines.append(f"max cell gap recovery-live:   {np.max(diff_cells):.1f}")
    lines.append("")
    lines.append(f"worst tracking mismatch frame: {worst_idx}")
    lines.append(
        f"  recovery corners/cells: "
        f"{int(recovery_corners[worst_idx])}/"
        f"{int(recovery_cells[worst_idx])}"
    )
    lines.append(
        f"  live corners/cells:     "
        f"{int(live_corners[worst_idx])}/"
        f"{int(live_cells[worst_idx])}"
    )
    lines.append("")
    lines.append(f"worst recovery frame: {worst_recovery_idx}")
    lines.append(
        f"  recovery corners/cells: "
        f"{int(recovery_corners[worst_recovery_idx])}/"
        f"{int(recovery_cells[worst_recovery_idx])}"
    )

    if np.max(diff_cells) > 20 or np.max(diff_corners) > 20:
        lines.append("")
        lines.append("Likely diagnosis:")
        lines.append("Recovery-only is better than live detect().")
        lines.append("=> likely LK / tracking state / tracking verification problem.")
    elif np.mean(recovery_cells) < 50:
        lines.append("")
        lines.append("Likely diagnosis:")
        lines.append("Recovery-only itself is unstable.")
        lines.append("=> likely corner detection / refinement / lattice / grid problem.")
    else:
        lines.append("")
        lines.append("Likely diagnosis:")
        lines.append("No strong recovery-vs-live mismatch found.")

    return "\n".join(lines)


# ============================================================
# Recording
# ============================================================

def record_debug_run(pipe):
    print("Recording debug run...")

    live_detector = hydramarker_cpp.CheckerboardDetector()
    recovery_detector = hydramarker_cpp.CheckerboardDetector()

    live_detector.reset_tracking()
    recovery_detector.reset_tracking()

    frames_bgr = []
    metrics = []

    t0 = time.time()
    frame_idx = 0

    while True:
        now = time.time()
        elapsed = now - t0

        if elapsed > RECORD_SECONDS:
            break

        frame = grab_realsense_frame(pipe)

        if frame is None:
            continue

        # Recovery-only: reset every frame
        recovery_detector.reset_tracking()
        recovery_det = recovery_detector.detect(frame)
        rec_corners, rec_cells, rec_has = detection_counts(recovery_det)

        # Live path: persistent tracking state
        live_det = live_detector.detect(frame)
        live_corners, live_cells, live_has = detection_counts(live_det)

        tracking_active = bool(live_detector.is_tracking())

        metrics.append([
            frame_idx,
            elapsed,
            rec_corners,
            rec_cells,
            int(rec_has),
            live_corners,
            live_cells,
            int(live_has),
            int(tracking_active),
        ])

        frames_bgr.append(frame.copy())

        # Live display during recording
        vis = frame.copy()

        draw_detection(
            vis,
            recovery_det,
            color=(255, 100, 0),
            draw_cells=False,
        )

        draw_detection(
            vis,
            live_det,
            color=(0, 255, 0),
            draw_cells=True,
        )

        put_text(
            vis,
            [
                "RECORDING DEBUG RUN",
                f"time: {elapsed:.2f}/{RECORD_SECONDS:.2f}s",
                f"recovery: {rec_corners} corners / {rec_cells} cells",
                f"live:     {live_corners} corners / {live_cells} cells",
                f"tracking active: {tracking_active}",
                "blue/orange = recovery-only | green = live tracking",
            ],
        )

        cv2.imshow("det", vis)
        cv2.waitKey(1)

        frame_idx += 1

    metrics = np.asarray(metrics, dtype=np.float32)

    key_idxs = select_keyframe_indices(metrics)
    keyframes = np.asarray([frames_bgr[i] for i in key_idxs], dtype=np.uint8)

    diagnosis = diagnose(metrics)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"hydramarker_debug_{stamp}.npz"

    np.savez_compressed(
        out_path,
        metrics=metrics,
        columns=np.asarray([
            "frame_index",
            "timestamp_s",
            "recovery_corners",
            "recovery_cells",
            "recovery_has_detection",
            "live_corners",
            "live_cells",
            "live_has_detection",
            "tracking_active",
        ]),
        keyframe_indices=np.asarray(key_idxs, dtype=np.int32),
        keyframes_bgr=keyframes,
        diagnosis=np.asarray(diagnosis),
    )

    print("")
    print(diagnosis)
    print("")
    print(f"Saved: {out_path}")

    return out_path


# ============================================================
# Main
# ============================================================

def main():
    detector = hydramarker_cpp.CheckerboardDetector()

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

    print("Controls:")
    print("  SPACE = record 3s debug run as NPZ")
    print("  r     = reset tracking")
    print("  ESC   = quit")

    try:
        while True:
            img = grab_realsense_frame(pipe)

            if img is None:
                continue

            vis = img.copy()

            det = detector.detect(img)
            corners, cells, _ = detection_counts(det)

            draw_detection(
                vis,
                det,
                color=(0, 255, 0),
                draw_cells=True,
            )

            put_text(
                vis,
                [
                    "HydraMarker live NPZ debug",
                    f"tracking active: {detector.is_tracking()}",
                    f"corners: {corners}",
                    f"cells: {cells}",
                    "SPACE: record 3s | r: reset | ESC: quit",
                ],
            )

            cv2.imshow("det", vis)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            elif key == ord("r"):
                detector.reset_tracking()
                print("Tracking reset.")

            elif key == ord(" "):
                record_debug_run(pipe)
                detector.reset_tracking()

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()