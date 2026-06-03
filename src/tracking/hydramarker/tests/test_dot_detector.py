"""
Live DotDetector debug using the current C++ pipeline.

Controls:
    1       checker corners
    2       checker cells
    3       dot states
    4       checker + dot centers
    5       all
    6       score heatmap
    7       ambiguous cells
    8       photometry debug
    SPACE   Save current visualization as PNG
    p       Pause / unpause live update
    ESC     Exit
"""

from pathlib import Path
from datetime import datetime
import csv
import json
import math

import cv2
import numpy as np
import pyrealsense2 as rs

from tracking.hydramarker.backend import cpp_impl as hydramarker_cpp


def round_int(x):
    return int(round(float(x)))


def put_text(img, text, pos, color=(0, 255, 255), scale=0.6, thickness=2):
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


def pxy(p):
    return round_int(p.x), round_int(p.y)


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def score_color(score):
    s = clamp01(score)
    b = int(round(255 * (1.0 - s)))
    r = int(round(255 * s))
    return (b, 0, r)


def cell_pts_np(cell):
    return np.array(
        [
            [cell.corner_uv[0].x, cell.corner_uv[0].y],
            [cell.corner_uv[1].x, cell.corner_uv[1].y],
            [cell.corner_uv[2].x, cell.corner_uv[2].y],
            [cell.corner_uv[3].x, cell.corner_uv[3].y],
        ],
        dtype=np.int32,
    )


def dot_for_cell(dot_lookup, cell):
    return dot_lookup.get((cell.j, cell.i))


def draw_checker_corners(vis, detection):
    for corner in detection.corners:
        u, v = pxy(corner.uv)
        cv2.circle(vis, (u, v), 4, (0, 255, 0), -1, cv2.LINE_AA)
        put_text(vis, f"{corner.i},{corner.j}", (u + 5, v - 5), (0, 255, 0), 0.32, 1)


def draw_checker_cells(vis, detection):
    for cell in detection.cells:
        pts = cell_pts_np(cell)
        cv2.polylines(vis, [pts], True, (0, 180, 255), 1, cv2.LINE_AA)
        cx, cy = pxy(cell.center_uv)
        cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1, cv2.LINE_AA)


def build_dot_lookup(dot_detection):
    if dot_detection is None:
        return {}
    return {(c.row, c.col): c for c in dot_detection.cells}


def draw_dot_cells(vis, checker_detection, dot_detection):
    dot_lookup = build_dot_lookup(dot_detection)

    for cell in checker_detection.cells:
        dot = dot_for_cell(dot_lookup, cell)
        pts = cell_pts_np(cell)

        if dot is None or not dot.valid:
            color = (80, 80, 80)
            label = "missing"
        elif dot.ambiguous:
            color = (0, 255, 255)
            label = f"amb s{dot.score:.2f}"
        elif dot.has_dot:
            color = (255, 0, 255)
            label = f"1 s{dot.score:.2f}"
        else:
            color = (255, 0, 0)
            label = f"0 s{dot.score:.2f}"

        cv2.polylines(vis, [pts], True, color, 2, cv2.LINE_AA)

        cx, cy = pxy(cell.center_uv)

        if dot is not None and dot.valid and dot.has_dot:
            cv2.circle(vis, (cx, cy), 5, color, -1, cv2.LINE_AA)
        elif dot is not None and dot.valid:
            cv2.circle(vis, (cx, cy), 4, color, 1, cv2.LINE_AA)
        else:
            cv2.drawMarker(vis, (cx, cy), color, cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)

        put_text(vis, label, (cx + 5, cy - 5), color, 0.30, 1)


def draw_dot_centers_only(vis, dot_detection):
    if dot_detection is None:
        return

    for dot in dot_detection.cells:
        cx, cy = pxy(dot.center_uv)

        if not dot.valid:
            color = (80, 80, 80)
            cv2.drawMarker(vis, (cx, cy), color, cv2.MARKER_TILTED_CROSS, 10, 2, cv2.LINE_AA)
        elif dot.ambiguous:
            color = (0, 255, 255)
            cv2.circle(vis, (cx, cy), 5, color, 1, cv2.LINE_AA)
        elif dot.has_dot:
            color = (255, 0, 255)
            cv2.circle(vis, (cx, cy), 5, color, -1, cv2.LINE_AA)
        else:
            color = (255, 0, 0)
            cv2.circle(vis, (cx, cy), 4, color, 1, cv2.LINE_AA)


def draw_score_heatmap(vis, checker_detection, dot_detection):
    dot_lookup = build_dot_lookup(dot_detection)
    overlay = vis.copy()

    for cell in checker_detection.cells:
        dot = dot_for_cell(dot_lookup, cell)
        pts = cell_pts_np(cell)

        if dot is None or not dot.valid:
            color = (80, 80, 80)
        else:
            color = score_color(dot.score)

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(vis, [pts], True, color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

    for cell in checker_detection.cells:
        dot = dot_for_cell(dot_lookup, cell)
        if dot is None or not dot.valid:
            continue

        cx, cy = pxy(cell.center_uv)
        put_text(
            vis,
            f"{dot.score:.2f}",
            (cx - 8, cy + 4),
            (255, 255, 255),
            0.30,
            1,
        )


def draw_ambiguous_cells(vis, checker_detection, dot_detection):
    dot_lookup = build_dot_lookup(dot_detection)

    for cell in checker_detection.cells:
        dot = dot_for_cell(dot_lookup, cell)
        pts = cell_pts_np(cell)

        if dot is None or not dot.valid:
            color = (80, 80, 80)
            label = "missing"
            show_label = False
        elif dot.ambiguous:
            color = (0, 255, 255)
            label = f"AMB s={dot.score:.2f}"
            show_label = True
        elif dot.has_dot:
            color = (255, 0, 255)
            label = f"1 s={dot.score:.2f}"
            show_label = False
        else:
            color = (255, 0, 0)
            label = f"0 s={dot.score:.2f}"
            show_label = False

        cv2.polylines(vis, [pts], True, color, 2, cv2.LINE_AA)

        if show_label:
            cx, cy = pxy(cell.center_uv)
            put_text(vis, label, (cx + 4, cy - 4), color, 0.32, 1)


def draw_photometry_debug(vis, checker_detection, dot_detection):
    dot_lookup = build_dot_lookup(dot_detection)

    for cell in checker_detection.cells:
        dot = dot_for_cell(dot_lookup, cell)
        pts = cell_pts_np(cell)

        if dot is None or not dot.valid:
            color = (80, 80, 80)
            label1 = "missing"
            label2 = ""
        else:
            color = score_color(dot.score)
            contrast = abs(float(dot.center_mean) - float(dot.ring_mean))
            label1 = f"s={dot.score:.2f} contrast={contrast:.1f}"
            label2 = (
                f"fg={dot.center_mean:.1f} bg={dot.ring_mean:.1f} "
                f"std={dot.local_std:.1f} pol={dot.polarity}"
            )

        cv2.polylines(vis, [pts], True, color, 2, cv2.LINE_AA)

        if dot is not None and dot.valid and (dot.score > 0.10 or dot.ambiguous):
            cx, cy = pxy(cell.center_uv)
            put_text(vis, label1, (cx + 4, cy - 8), color, 0.28, 1)
            put_text(vis, label2, (cx + 4, cy + 6), color, 0.26, 1)


def dot_stats(dot_detection):
    if dot_detection is None:
        return {"total": 0, "invalid": 0, "empty": 0, "dot": 0, "ambiguous": 0}

    cells = dot_detection.cells

    return {
        "total": len(cells),
        "invalid": sum(not c.valid for c in cells),
        "empty": sum(c.valid and not c.has_dot for c in cells),
        "dot": sum(c.valid and c.has_dot for c in cells),
        "ambiguous": sum(c.valid and c.ambiguous for c in cells),
    }


def draw_info_panel(vis, lines):
    x0, y0 = 25, 25
    line_h = 26
    pad = 14

    width = 1040
    height = pad * 2 + line_h * len(lines)

    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    y = y0 + pad + 18

    for line in lines:
        color = (0, 255, 255)

        if "PAUSED" in line:
            color = (0, 0, 255)
        elif "invalid" in line:
            color = (0, 160, 255)
        elif "ambiguous" in line:
            color = (0, 255, 255)

        put_text(vis, line, (x0 + pad, y), color, 0.55, 1)
        y += line_h


def get_debug_summary(dot_detection):
    if dot_detection is None or not dot_detection.cells:
        return "score avg=0.00 max=0.00 | ambiguous=0 | contrast avg=0.0 max=0.0"

    valid_cells = [c for c in dot_detection.cells if c.valid]

    if not valid_cells:
        return "score avg=0.00 max=0.00 | ambiguous=0 | contrast avg=0.0 max=0.0"

    scores = np.array([c.score for c in valid_cells], dtype=np.float32)
    contrasts = np.array(
        [abs(float(c.center_mean) - float(c.ring_mean)) for c in valid_cells],
        dtype=np.float32,
    )

    ambiguous = int(sum(c.ambiguous for c in valid_cells))

    return (
        f"score avg={float(np.mean(scores)):.2f} max={float(np.max(scores)):.2f} | "
        f"ambiguous={ambiguous} | "
        f"contrast avg={float(np.mean(contrasts)):.1f} max={float(np.max(contrasts)):.1f}"
    )


def point_xy(p):
    return float(p.x), float(p.y)


def polygon_area(points):
    if len(points) < 3:
        return 0.0

    area = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1

    return 0.5 * area


def cell_shape_stats(cell):
    pts = [
        point_xy(cell.corner_uv[0]),
        point_xy(cell.corner_uv[1]),
        point_xy(cell.corner_uv[2]),
        point_xy(cell.corner_uv[3]),
    ]

    edges = []
    for idx in range(4):
        x1, y1 = pts[idx]
        x2, y2 = pts[(idx + 1) % 4]
        edges.append(math.hypot(x2 - x1, y2 - y1))

    d02 = math.hypot(pts[2][0] - pts[0][0], pts[2][1] - pts[0][1])
    d13 = math.hypot(pts[3][0] - pts[1][0], pts[3][1] - pts[1][1])

    min_edge = min(edges) if edges else 0.0
    max_edge = max(edges) if edges else 0.0
    edge_ratio = max_edge / max(min_edge, 1e-6)

    signed_area = polygon_area(pts)
    area = abs(signed_area)

    return {
        "area": area,
        "signed_area": signed_area,
        "min_edge": min_edge,
        "max_edge": max_edge,
        "edge_ratio": edge_ratio,
        "diag_ratio": max(d02, d13) / max(min(d02, d13), 1e-6),
    }


def checker_stats(checker_detection):
    stats = {
        "checker_present": checker_detection is not None,
        "corners": 0,
        "cells": 0,
        "rows": 0,
        "cols": 0,
        "tracking": False,
        "stable": False,
        "min_i": None,
        "max_i": None,
        "min_j": None,
        "max_j": None,
        "duplicate_ids": 0,
        "suspicious_cells": 0,
        "negative_area_cells": 0,
        "cell_edge_ratio_max": 0.0,
        "cell_diag_ratio_max": 0.0,
    }

    if checker_detection is None:
        return stats

    corners = list(checker_detection.corners)
    cells = list(checker_detection.cells)

    stats["corners"] = len(corners)
    stats["cells"] = len(cells)
    stats["rows"] = int(checker_detection.rows)
    stats["cols"] = int(checker_detection.cols)
    stats["tracking"] = bool(checker_detection.tracking)
    stats["stable"] = bool(checker_detection.stable)

    if corners:
        ids = [(int(c.i), int(c.j)) for c in corners]
        is_ = [p[0] for p in ids]
        js = [p[1] for p in ids]
        stats["min_i"] = min(is_)
        stats["max_i"] = max(is_)
        stats["min_j"] = min(js)
        stats["max_j"] = max(js)
        stats["duplicate_ids"] = len(ids) - len(set(ids))

    suspicious = 0
    negative_area = 0
    max_edge_ratio = 0.0
    max_diag_ratio = 0.0

    for cell in cells:
        shape = cell_shape_stats(cell)
        max_edge_ratio = max(max_edge_ratio, shape["edge_ratio"])
        max_diag_ratio = max(max_diag_ratio, shape["diag_ratio"])

        if shape["signed_area"] < 0.0:
            negative_area += 1

        if (
            shape["area"] < 25.0
            or shape["edge_ratio"] > 2.8
            or shape["diag_ratio"] > 2.2
        ):
            suspicious += 1

    stats["suspicious_cells"] = suspicious
    stats["negative_area_cells"] = negative_area
    stats["cell_edge_ratio_max"] = max_edge_ratio
    stats["cell_diag_ratio_max"] = max_diag_ratio

    return stats


def dot_summary_stats(dot_detection):
    stats = dot_stats(dot_detection)

    stats.update(
        {
            "dot_grid_rows": 0,
            "dot_grid_cols": 0,
            "score_avg": 0.0,
            "score_max": 0.0,
            "contrast_avg": 0.0,
            "contrast_max": 0.0,
        }
    )

    if dot_detection is None or not dot_detection.cells:
        return stats

    stats["dot_grid_rows"] = int(dot_detection.rows)
    stats["dot_grid_cols"] = int(dot_detection.cols)

    valid_cells = [c for c in dot_detection.cells if c.valid]
    if not valid_cells:
        return stats

    scores = np.array([c.score for c in valid_cells], dtype=np.float32)
    contrasts = np.array(
        [abs(float(c.center_mean) - float(c.ring_mean)) for c in valid_cells],
        dtype=np.float32,
    )

    stats["score_avg"] = float(np.mean(scores))
    stats["score_max"] = float(np.max(scores))
    stats["contrast_avg"] = float(np.mean(contrasts))
    stats["contrast_max"] = float(np.max(contrasts))

    return stats


class CompactDebugLogger:
    def __init__(self, output_dir, summary_every_n_frames=15, repeat_event_every=30):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary_path = self.output_dir / f"hydramarker_lattice_summary_{ts}.csv"
        self.events_path = self.output_dir / f"hydramarker_lattice_events_{ts}.jsonl"

        self.summary_every_n_frames = int(summary_every_n_frames)
        self.repeat_event_every = int(repeat_event_every)

        self.summary_file = self.summary_path.open("w", newline="", encoding="utf-8")
        self.event_file = self.events_path.open("w", encoding="utf-8")

        self.summary_writer = None
        self.last_signature = None
        self.last_checker_stats = None
        self.event_counts = {}

        print(f"[HydraMarkerDebug] summary log: {self.summary_path}")
        print(f"[HydraMarkerDebug] event log:   {self.events_path}")

    def close(self):
        if self.summary_file:
            self.summary_file.flush()
            self.summary_file.close()
            self.summary_file = None

        if self.event_file:
            self.event_file.flush()
            self.event_file.close()
            self.event_file = None

    def make_signature(self, checker, dots):
        return (
            checker["checker_present"],
            checker["corners"],
            checker["cells"],
            checker["rows"],
            checker["cols"],
            checker["tracking"],
            checker["stable"],
            checker["min_i"],
            checker["max_i"],
            checker["min_j"],
            checker["max_j"],
            checker["duplicate_ids"],
            checker["suspicious_cells"],
            dots["total"],
            dots["invalid"],
            dots["dot"],
            dots["ambiguous"],
        )

    def should_write_summary(self, frame_idx, signature):
        if signature != self.last_signature:
            return True

        return frame_idx % self.summary_every_n_frames == 0

    def write_summary(self, row):
        if self.summary_writer is None:
            self.summary_writer = csv.DictWriter(self.summary_file, fieldnames=list(row.keys()))
            self.summary_writer.writeheader()

        self.summary_writer.writerow(row)
        self.summary_file.flush()

    def write_event(self, frame_idx, level, event_type, message, payload=None):
        payload = payload or {}
        key = (level, event_type, message)

        count = self.event_counts.get(key, 0) + 1
        self.event_counts[key] = count

        if count != 1 and count % self.repeat_event_every != 0:
            return

        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame": int(frame_idx),
            "level": level,
            "type": event_type,
            "repeat_count": count,
            "message": message,
            "payload": payload,
        }

        self.event_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.event_file.flush()

    def detect_events(self, frame_idx, checker, dots):
        if not checker["checker_present"]:
            self.write_event(
                frame_idx,
                "WARN",
                "NO_CHECKER_DETECTION",
                "checker_detector.detect(img) returned None",
            )
            self.last_checker_stats = checker
            return

        if checker["corners"] == 0:
            self.write_event(
                frame_idx,
                "ERROR",
                "ZERO_CORNERS",
                "checker detection exists but contains zero corners",
                checker,
            )

        if checker["rows"] <= 0 or checker["cols"] <= 0:
            self.write_event(
                frame_idx,
                "WARN",
                "INVALID_GRID_DIMENSIONS",
                "grid rows or cols are non-positive",
                checker,
            )

        if checker["duplicate_ids"] > 0:
            self.write_event(
                frame_idx,
                "ERROR",
                "DUPLICATE_GRID_IDS",
                "multiple detected corners share the same (i,j) ID",
                {
                    "duplicate_ids": checker["duplicate_ids"],
                    "corners": checker["corners"],
                    "min_i": checker["min_i"],
                    "max_i": checker["max_i"],
                    "min_j": checker["min_j"],
                    "max_j": checker["max_j"],
                },
            )

        if checker["suspicious_cells"] > 0:
            self.write_event(
                frame_idx,
                "WARN",
                "SUSPICIOUS_CELL_GEOMETRY",
                "one or more cells have extreme shape ratios or tiny area",
                {
                    "suspicious_cells": checker["suspicious_cells"],
                    "negative_area_cells": checker["negative_area_cells"],
                    "cell_edge_ratio_max": checker["cell_edge_ratio_max"],
                    "cell_diag_ratio_max": checker["cell_diag_ratio_max"],
                },
            )

        prev = self.last_checker_stats
        if prev is not None and prev["checker_present"]:
            corner_drop = prev["corners"] - checker["corners"]
            cell_drop = prev["cells"] - checker["cells"]

            if corner_drop >= max(6, int(0.35 * max(prev["corners"], 1))):
                self.write_event(
                    frame_idx,
                    "WARN",
                    "CORNER_COUNT_DROP",
                    "corner count dropped strongly compared to previous frame",
                    {
                        "previous_corners": prev["corners"],
                        "current_corners": checker["corners"],
                        "drop": corner_drop,
                    },
                )

            if cell_drop >= max(6, int(0.35 * max(prev["cells"], 1))):
                self.write_event(
                    frame_idx,
                    "WARN",
                    "CELL_COUNT_DROP",
                    "cell count dropped strongly compared to previous frame",
                    {
                        "previous_cells": prev["cells"],
                        "current_cells": checker["cells"],
                        "drop": cell_drop,
                    },
                )

            previous_dims = (prev["rows"], prev["cols"], prev["min_i"], prev["max_i"], prev["min_j"], prev["max_j"])
            current_dims = (checker["rows"], checker["cols"], checker["min_i"], checker["max_i"], checker["min_j"], checker["max_j"])

            if previous_dims != current_dims and checker["corners"] >= 8 and prev["corners"] >= 8:
                self.write_event(
                    frame_idx,
                    "INFO",
                    "GRID_INDEX_RANGE_CHANGED",
                    "grid dimensions or min/max (i,j) changed",
                    {
                        "previous": {
                            "rows": prev["rows"],
                            "cols": prev["cols"],
                            "min_i": prev["min_i"],
                            "max_i": prev["max_i"],
                            "min_j": prev["min_j"],
                            "max_j": prev["max_j"],
                        },
                        "current": {
                            "rows": checker["rows"],
                            "cols": checker["cols"],
                            "min_i": checker["min_i"],
                            "max_i": checker["max_i"],
                            "min_j": checker["min_j"],
                            "max_j": checker["max_j"],
                        },
                    },
                )

        if dots["total"] > 0 and dots["ambiguous"] >= max(4, int(0.25 * dots["total"])):
            self.write_event(
                frame_idx,
                "INFO",
                "MANY_AMBIGUOUS_DOTS",
                "large fraction of dot cells is ambiguous",
                {
                    "ambiguous": dots["ambiguous"],
                    "total": dots["total"],
                    "score_avg": dots["score_avg"],
                    "score_max": dots["score_max"],
                },
            )

        self.last_checker_stats = checker

    def log_frame(self, frame_idx, checker_detection, dot_detection):
        checker = checker_stats(checker_detection)
        dots = dot_summary_stats(dot_detection)
        signature = self.make_signature(checker, dots)

        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "frame": int(frame_idx),
            **checker,
            **{f"dot_{k}": v for k, v in dots.items()},
        }

        if self.should_write_summary(frame_idx, signature):
            self.write_summary(row)
            self.last_signature = signature

        self.detect_events(frame_idx, checker, dots)



def get_info_lines(mode_name, checker_detection, dot_detection, paused):
    lines = [
        "HydraMarker DotDetector Live Debug",
        f"mode: {mode_name}",
        "1 corners | 2 cells | 3 states | 4 centers | 5 all | 6 scores | 7 ambiguous | 8 photometry | p pause | SPACE save | ESC quit",
    ]

    if paused:
        lines.append("*** PAUSED ***")

    if checker_detection is None:
        lines.append("checker: no detection")
    else:
        lines.append(
            f"checker corners={len(checker_detection.corners)} | "
            f"cells={len(checker_detection.cells)} | "
            f"rows={checker_detection.rows} | cols={checker_detection.cols} | "
            f"tracking={checker_detection.tracking} | "
            f"stable={checker_detection.stable}"
        )

    stats = dot_stats(dot_detection)

    lines.append(
        f"dot cells={stats['total']} | "
        f"invalid={stats['invalid']} | "
        f"empty={stats['empty']} | "
        f"dot={stats['dot']} | "
        f"ambiguous={stats['ambiguous']}"
    )

    if dot_detection is not None:
        lines.append(f"dot grid={dot_detection.cols} x {dot_detection.rows}")
        lines.append(get_debug_summary(dot_detection))

    lines.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return lines


def render(image, checker_detection, dot_detection, mode, paused):
    vis = image.copy()

    mode_names = {
        1: "checker corners",
        2: "checker cells",
        3: "dot states",
        4: "checker + dot centers",
        5: "all",
        6: "score heatmap",
        7: "ambiguous cells",
        8: "photometry debug",
    }

    if checker_detection is not None:
        if mode == 1:
            draw_checker_corners(vis, checker_detection)
        elif mode == 2:
            draw_checker_cells(vis, checker_detection)
        elif mode == 3:
            draw_dot_cells(vis, checker_detection, dot_detection)
        elif mode == 4:
            draw_checker_cells(vis, checker_detection)
            draw_checker_corners(vis, checker_detection)
            draw_dot_centers_only(vis, dot_detection)
        elif mode == 6:
            draw_score_heatmap(vis, checker_detection, dot_detection)
        elif mode == 7:
            draw_ambiguous_cells(vis, checker_detection, dot_detection)
        elif mode == 8:
            draw_photometry_debug(vis, checker_detection, dot_detection)
        else:
            draw_dot_cells(vis, checker_detection, dot_detection)
            draw_checker_corners(vis, checker_detection)

    draw_info_panel(
        vis,
        get_info_lines(
            mode_names.get(mode, "unknown"),
            checker_detection,
            dot_detection,
            paused,
        ),
    )

    return vis


def save_png(vis, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = output_dir / f"dot_detector_live_{ts}.png"

    cv2.imwrite(str(path), vis)
    print(f"Saved: {path}")


def create_dot_detector():
    dot_cfg = hydramarker_cpp.DotDetectorConfig()

    dot_cfg.canonical_size = 80
    dot_cfg.canonical_margin_px = 4.0

    dot_cfg.min_dot_contrast = 8.0
    dot_cfg.strong_dot_contrast = 35.0

    dot_cfg.commit_threshold = 0.45
    dot_cfg.revoke_threshold = 0.20

    dot_cfg.uncertainty_low = 0.20
    dot_cfg.uncertainty_high = 0.45

    dot_cfg.warmup_frames = 1

    return hydramarker_cpp.DotDetector(dot_cfg)


def main():
    checker_detector = hydramarker_cpp.CheckerboardDetector()
    dot_detector = create_dot_detector()

    output_dir = Path(__file__).resolve().parent / "dot_detector_live_snapshots"
    log_dir = Path(__file__).resolve().parent / "dot_detector_live_logs"
    debug_logger = CompactDebugLogger(log_dir)
    frame_idx = 0

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

    window_name = "HydraMarker DotDetector Live Debug"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    mode = 6
    paused = False

    last_img = None
    last_checker_detection = None
    last_dot_detection = None
    last_vis = None

    try:
        while True:
            if not paused:
                frames = pipe.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                img = np.asanyarray(color_frame.get_data())

                checker_detection = checker_detector.detect(img)

                if checker_detection is not None:
                    if len(checker_detection.corners) == 0:
                        dot_detector.reset()
                    dot_detection = dot_detector.detect(img, checker_detection)
                else:
                    dot_detector.reset()
                    dot_detection = None

                debug_logger.log_frame(frame_idx, checker_detection, dot_detection)
                frame_idx += 1

                last_img = img.copy()
                last_checker_detection = checker_detection
                last_dot_detection = dot_detection

            if last_img is None:
                continue

            last_vis = render(
                last_img,
                last_checker_detection,
                last_dot_detection,
                mode,
                paused,
            )

            cv2.imshow(window_name, last_vis)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key in [
                ord("1"),
                ord("2"),
                ord("3"),
                ord("4"),
                ord("5"),
                ord("6"),
                ord("7"),
                ord("8"),
            ]:
                mode = int(chr(key))

            elif key == ord("p"):
                paused = not paused

            elif key == 32 and last_vis is not None:
                save_png(last_vis, output_dir)

    finally:
        debug_logger.close()
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()