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
import json
import time

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
_log_debug_recovery = False
_log_corner_payload = True

TIMING_KEYS = [
    "detect_wall_ms",
    "to_gray_ms",
    "track_total_ms",
    "lk_ms",
    "lk_prev_pyramid_reused_count",
    "tracking_validate_ms",
    "tracking_cull_ms",
    "build_visible_tracked_ms",
    "build_visible_tracked_total_ms",
    "build_visible_spacing_cleanup_ms",
    "grid_build_tracking_ms",
    "update_tracking_state_ms",
    "tracking_local_completion_attempt_ms",
    "tracking_local_completion_reused_recovery_count",
    "tracking_local_completion_corner_detect_ms",
    "tracking_local_completion_refine_ms",
    "tracking_local_completion_detected_candidate_count",
    "tracking_local_completion_guided_seed_count",
    "tracking_local_completion_guided_refine_ms",
    "tracking_local_completion_guided_candidate_count",
    "tracking_local_completion_candidate_count",
    "tracking_local_completion_proposal_count",
    "tracking_local_completion_cell_proposal_count",
    "tracking_local_completion_edge_proposal_count",
    "tracking_local_completion_line_proposal_count",
    "tracking_local_completion_match_count",
    "tracking_local_completion_edge_pair_match_count",
    "tracking_local_completion_guided_match_count",
    "tracking_local_completion_measured_match_count",
    "tracking_local_completion_pending_count",
    "tracking_local_completion_deferred_count",
    "tracking_local_completion_geometry_reject_count",
    "tracking_local_completion_fast_accept_count",
    "tracking_local_completion_added_count",
    "tracking_saddle_snap_count",
    "tracking_saddle_snap_predicted_count",
    "tracking_saddle_snap_error_px",
    "tracking_local_correction_count",
    "tracking_local_correction_error_px",
    "recovery_position_correction_count",
    "recovery_position_correction_error_px",
    "tracking_output_geometry_reject_count",
    "tracking_output_decode_span_guard_count",
    "tracking_output_temporal_reject_count",
    "tracking_output_single_neighbour_hold_count",
    "tracking_output_support_reject_count",
    "tracking_persistent_count",
    "tracking_persistent_missed_count",
    "tracking_persistent_predicted_count",
    "refresh_recovery_call_ms",
    "recovery_total_ms",
    "full_recovery_call_ms",
    "fallback_recovery_call_ms",
    "recovery_crop_ms",
    "recovery_roi_select_ms",
    "recovery_roi_attempt_ms",
    "recovery_roi_area_ratio",
    "recovery_roi_width_px",
    "recovery_roi_height_px",
    "recovery_roi_candidate_only_count",
    "recovery_roi_success_count",
    "recovery_roi_fallback_count",
    "recovery_roi_skipped_count",
    "recovery_roi_retry_select_ms",
    "recovery_roi_retry_attempt_ms",
    "recovery_roi_retry_area_ratio",
    "recovery_roi_retry_width_px",
    "recovery_roi_retry_height_px",
    "recovery_roi_retry_success_count",
    "recovery_roi_retry_fallback_count",
    "recovery_roi_retry_skipped_count",
    "recovery_roi_retry_corner_detect_ms",
    "recovery_roi_retry_refine_ms",
    "recovery_roi_retry_build_best_ms",
    "recovery_full_frame_fallback_ms",
    "recovery_full_frame_fallback_deferred_count",
    "refresh_roi_candidate_only_requested_count",
    "refresh_roi_candidate_only_count",
    "refresh_roi_recovery_fail_count",
    "refresh_roi_align_fail_count",
    "refresh_roi_fail_full_retry_count",
    "refresh_roi_unaligned_reset_count",
    "refresh_full_recovery_unaligned_reset_count",
    "refresh_full_recovery_after_roi_align_fail_deferred_count",
    "refresh_expanded_roi_after_align_fail_ms",
    "refresh_expanded_roi_align_success_count",
    "refresh_expanded_roi_align_fail_count",
    "recovery_expanded_roi_corner_detect_ms",
    "recovery_expanded_roi_refine_ms",
    "recovery_expanded_roi_build_best_ms",
    "refresh_full_recovery_align_success_count",
    "refresh_full_recovery_after_roi_align_fail_ms",
    "fallback_full_recovery_after_roi_align_fail_ms",
    "fallback_full_recovery_after_roi_align_fail_deferred_count",
    "recovery_resize_ms",
    "recovery_raw_count",
    "recovery_refined_count",
    "recovery_quadrant_count",
    "recovery_corner_detect_ms",
    "recovery_roi_corner_detect_ms",
    "recovery_roi_raw_count",
    "recovery_roi_refined_count",
    "recovery_roi_quadrant_count",
    "recovery_roi_refine_ms",
    "recovery_roi_build_best_ms",
    "recovery_full_fallback_corner_detect_ms",
    "recovery_full_fallback_raw_count",
    "recovery_full_fallback_refined_count",
    "recovery_full_fallback_quadrant_count",
    "build_best_total_ms",
    "build_best_subset_pruned_count",
    "corner_detect_total_ms",
    "corner_detect_make_fast_image_ms",
    "corner_detect_fast_gradient_total_ms",
    "corner_detect_fast_gradient_sobel_ms",
    "corner_detect_fast_gradient_blur_ms",
    "corner_detect_fast_gradient_nms_ms",
    "corner_detect_fast_gradient_scan_points_ms",
    "corner_detect_fast_gradient_partial_sort_ms",
    "corner_detect_fast_gftt_good_features_ms",
    "corner_detect_fast_merge_ms",
    "corner_detect_build_variants_ms",
    "corner_detect_variants_clahe_ms",
    "corner_detect_variant_gradient_total_ms",
    "corner_detect_variant_gradient_sobel_ms",
    "corner_detect_variant_gradient_blur_ms",
    "corner_detect_variant_gradient_nms_ms",
    "corner_detect_variant_gradient_scan_points_ms",
    "corner_detect_variant_gradient_partial_sort_ms",
    "corner_detect_variant_gftt_good_features_ms",
    "corner_detect_variant_merge_gradient_ms",
    "corner_detect_variant_merge_gftt_ms",
    "corner_detect_final_gradient_total_ms",
    "recovery_refine_ms",
    "recovery_quadrant_filter_ms",
    "recovery_build_best_ms",
    "recovery_completion_ms",
    "lattice_fit_ms",
    "grid_build_lattice_ms",
]


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
        "seed_span_i", "seed_span_j",
        "seed_cell_span_i", "seed_cell_span_j",
        "seed_max_cell_square",
        "seed_bbox_x", "seed_bbox_y",
        "seed_bbox_w", "seed_bbox_h", "seed_bbox_area_px",
        "seed_corner_density", "seed_cell_density",
        "seed_decodeable_3x3", "seed_score",
        "debug_raw", "debug_refined", "debug_lattice",
        "debug_det_corners", "debug_det_cells",
        *TIMING_KEYS,
        "timing_detail_json",
        "corner_keys", "corner_uvs", "corner_vis", "corner_pred",
    ])
    _log_start_ms  = datetime.now().timestamp() * 1000.0
    _log_frame_idx = 0
    _log_active    = True
    print(f"[LOG] Started -> {path}")


def log_stop() -> None:
    global _log_active, _log_file
    _log_active = False
    if _log_file:
        _log_file.flush()
        _log_file.close()
        _log_file = None
    print(f"[LOG] Stopped - {_log_frame_idx} frames logged")


def toggle_log_debug_recovery() -> None:
    global _log_debug_recovery
    _log_debug_recovery = not _log_debug_recovery
    print(f"[LOG] Recovery-stage logging {'ON' if _log_debug_recovery else 'OFF'}")


def toggle_log_corner_payload() -> None:
    global _log_corner_payload
    _log_corner_payload = not _log_corner_payload
    print(f"[LOG] Corner payload {'ON' if _log_corner_payload else 'OFF'}")


def recovery_counts(dbg):
    if not dbg:
        return 0, 0, 0, 0, 0

    raw_n = len(getattr(dbg, "raw_candidates", []) or [])
    refined_n = sum(
        1 for c in (getattr(dbg, "refined_corners", []) or [])
        if bool(getattr(c, "valid", False))
    )
    lattice = int(bool(getattr(dbg, "has_lattice", False)))
    rec_det = (
        getattr(dbg, "detection", None)
        if getattr(dbg, "has_detection", False)
        else None
    )
    rec_c = len(rec_det.corners) if rec_det else 0
    rec_cells = len(rec_det.cells) if rec_det else 0
    return raw_n, refined_n, lattice, rec_c, rec_cells


def format_timing(timings: dict, key: str) -> str:
    value = timings.get(key)
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return ""


def timing_value(timings: dict, *keys: str) -> float:
    for key in keys:
        value = timings.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def get_checkerboard_timings(detector, detect_wall_ms: float) -> dict:
    timings = {"detect_wall_ms": float(detect_wall_ms)}
    if hasattr(detector, "last_timings_ms"):
        try:
            timings.update({
                str(k): float(v)
                for k, v in detector.last_timings_ms().items()
            })
        except Exception:
            pass
    return timings


def log_frame(det, global_frame: int, timings: dict, debug_recovery=None) -> None:
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
    seed = estimate_seed_geometry(det)
    raw_n, refined_n, lattice, rec_c, rec_cells = recovery_counts(debug_recovery)

    corner_keys = ""
    corner_uvs = ""
    corner_vis = ""
    corner_pred = ""
    if det and det.corners:
        if _log_corner_payload:
            key_parts = []
            uv_parts = []
            vis_parts = []
            pred_parts = []
            for c in det.corners:
                u, v = get_xy(c.uv)
                vscore = float(getattr(c, "visibility_score", 1.0))
                key_parts.append(f"{int(c.i)}:{int(c.j)}")
                uv_parts.append(f"{int(c.i)}:{int(c.j)}:{u:.1f}:{v:.1f}")
                vis_parts.append(f"{int(c.i)}:{int(c.j)}:{vscore:.3f}")
                predicted = 1 if bool(getattr(c, "predicted", False)) else 0
                observed = int(getattr(c, "observed_frames", 0))
                pred_parts.append(f"{int(c.i)}:{int(c.j)}:{predicted}:{observed}")
            corner_keys = ";".join(key_parts)
            corner_uvs = ";".join(uv_parts)
            corner_vis = ";".join(vis_parts)
            corner_pred = ";".join(pred_parts)

    _log_writer.writerow([
        global_frame,
        f"{now_ms:.1f}",
        n_corners, n_cells, tracking, stable,
        spacing_median, spacing_min,
        seed["span_i"], seed["span_j"],
        seed["cell_span_i"], seed["cell_span_j"],
        seed["max_cell_square"],
        f"{seed['bbox_x']:.1f}", f"{seed['bbox_y']:.1f}",
        f"{seed['bbox_w']:.1f}", f"{seed['bbox_h']:.1f}",
        f"{seed['bbox_area_px']:.1f}",
        f"{seed['corner_density']:.3f}",
        f"{seed['cell_density']:.3f}",
        seed["decodeable_3x3"],
        f"{seed['score']:.1f}",
        raw_n, refined_n, lattice, rec_c, rec_cells,
        *[format_timing(timings, key) for key in TIMING_KEYS],
        json.dumps(
            {k: round(float(v), 3) for k, v in sorted(timings.items())},
            ensure_ascii=False,
        ),
        corner_keys, corner_uvs, corner_vis, corner_pred,
    ])

    _log_frame_idx += 1
    if _log_frame_idx % 30 == 0:
        _log_file.flush()

# ============================================================
# Helpers
# ============================================================

def get_xy(p):
    if hasattr(p, "x") and hasattr(p, "y"):
        return float(p.x), float(p.y)
    return float(p[0]), float(p[1])


def draw_points(vis, points, color, radius=3, thickness=-1) -> None:
    for p in points:
        u, v = get_xy(p)
        cv2.circle(vis, (int(round(u)), int(round(v))),
                   radius, color, thickness, lineType=cv2.LINE_AA)


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


def max_contiguous_cell_square(cells) -> int:
    if not cells:
        return 0

    cell_keys = {(int(c.i), int(c.j)) for c in cells}
    best = 0

    for origin_i, origin_j in cell_keys:
        size = 1
        while True:
            ok = True
            for di in range(size):
                for dj in range(size):
                    if (origin_i + di, origin_j + dj) not in cell_keys:
                        ok = False
                        break
                if not ok:
                    break

            if not ok:
                break

            best = max(best, size)
            size += 1

    return best


def estimate_seed_geometry(det) -> dict:
    empty = {
        "span_i": 0,
        "span_j": 0,
        "cell_span_i": 0,
        "cell_span_j": 0,
        "max_cell_square": 0,
        "bbox_x": -1.0,
        "bbox_y": -1.0,
        "bbox_w": 0.0,
        "bbox_h": 0.0,
        "bbox_area_px": 0.0,
        "corner_density": 0.0,
        "cell_density": 0.0,
        "decodeable_3x3": 0,
        "score": 0.0,
    }
    if not det or not det.corners:
        return empty

    corner_is = [int(c.i) for c in det.corners]
    corner_js = [int(c.j) for c in det.corners]
    span_i = max(corner_is) - min(corner_is) + 1
    span_j = max(corner_js) - min(corner_js) + 1

    uvs = np.asarray([get_xy(c.uv) for c in det.corners], dtype=np.float32)
    x0 = float(np.min(uvs[:, 0]))
    y0 = float(np.min(uvs[:, 1]))
    x1 = float(np.max(uvs[:, 0]))
    y1 = float(np.max(uvs[:, 1]))
    bbox_w = max(0.0, x1 - x0)
    bbox_h = max(0.0, y1 - y0)

    if det.cells:
        cell_is = [int(c.i) for c in det.cells]
        cell_js = [int(c.j) for c in det.cells]
        cell_span_i = max(cell_is) - min(cell_is) + 1
        cell_span_j = max(cell_js) - min(cell_js) + 1
    else:
        cell_span_i = 0
        cell_span_j = 0

    max_square = max_contiguous_cell_square(det.cells)
    corner_slots = max(1, span_i * span_j)
    cell_slots = max(1, cell_span_i * cell_span_j)
    corner_density = len(det.corners) / corner_slots
    cell_density = len(det.cells) / cell_slots if det.cells else 0.0
    decodeable = int(max_square >= 3)

    # Diagnostic only: rough 0..100 quality score for comparing cold-start
    # seeds. It intentionally mirrors what we care about for initialization:
    # enough corners, enough cells, a 3x3-decodeable cell patch, and real image
    # extent instead of a tiny local cluster.
    corner_score = min(1.0, len(det.corners) / 60.0)
    cell_score = min(1.0, len(det.cells) / 45.0)
    span_score = min(1.0, min(span_i, span_j) / 6.0)
    square_score = min(1.0, max_square / 4.0)
    extent_score = min(1.0, (bbox_w * bbox_h) / 120000.0)
    score = 100.0 * (
        0.25 * corner_score +
        0.20 * cell_score +
        0.20 * span_score +
        0.25 * square_score +
        0.10 * extent_score
    )

    return {
        "span_i": span_i,
        "span_j": span_j,
        "cell_span_i": cell_span_i,
        "cell_span_j": cell_span_j,
        "max_cell_square": max_square,
        "bbox_x": x0,
        "bbox_y": y0,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "bbox_area_px": bbox_w * bbox_h,
        "corner_density": corner_density,
        "cell_density": cell_density,
        "decodeable_3x3": decodeable,
        "score": score,
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


def final_uvs(det):
    if not det:
        return []
    return [get_xy(c.uv) for c in det.corners]


def draw_recovery_debug(vis, dbg, det) -> None:
    if not dbg:
        return

    raw = list(getattr(dbg, "raw_candidates", []) or [])
    refined = [
        c.uv for c in (getattr(dbg, "refined_corners", []) or [])
        if bool(getattr(c, "valid", False))
    ]

    draw_points(vis, raw, color=(255, 255, 0), radius=2, thickness=1)
    draw_points(vis, refined, color=(0, 165, 255), radius=3, thickness=1)

    if getattr(dbg, "has_detection", False):
        draw_corners(vis, dbg.detection, color=(255, 100, 0),
                     radius=3, draw_indices=False)

        finals = final_uvs(det)
        for c in dbg.detection.corners:
            u, v = get_xy(c.uv)
            found = any(abs(u - fu) < 10 and abs(v - fv) < 10
                        for fu, fv in finals)
            if not found:
                cv2.circle(vis, (int(round(u)), int(round(v))),
                           5, (0, 255, 255), 2, lineType=cv2.LINE_AA)


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


def draw_status(vis, mode_name, det, debug_det, debug_on, timings=None) -> None:
    normal_n = len(det.corners) if det else 0
    normal_c = len(det.cells)   if det else 0
    debug_n  = len(debug_det.corners) if debug_det else 0
    debug_c  = len(debug_det.cells)   if debug_det else 0

    final_stats = estimate_square_stats(det)
    debug_stats = estimate_square_stats(debug_det)

    log_indicator = " | [LOG]" if _log_active else ""
    if _log_active and _log_debug_recovery:
        log_indicator += "[DBG]"
    if _log_active and not _log_corner_payload:
        log_indicator += "[NO-CORNERS]"
    line1 = (f"mode: {mode_name} | t=toggle d=debug "
             f"g=logdbg c=corners l=log SPACE=save ESC=quit{log_indicator}")
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

    lines = [line1, line2, line3]
    if timings:
        lines.append(
            "ms: "
            f"det={float(timings.get('detect_wall_ms', 0.0)):.1f} "
            f"track={float(timings.get('track_total_ms', 0.0)):.1f} "
            f"lk={float(timings.get('lk_ms', 0.0)):.1f} "
            f"val={float(timings.get('tracking_validate_ms', 0.0)):.1f} "
            f"grid={float(timings.get('grid_build_tracking_ms', 0.0)):.1f} "
            f"rec={float(timings.get('recovery_total_ms', 0.0)):.1f}"
        )
        lines.append(
            "corner ms: "
            f"all={timing_value(timings, 'corner_detect_total_ms'):.1f} "
            f"sob={timing_value(timings, 'corner_detect_fast_gradient_sobel_ms', 'corner_detect_variant_gradient_sobel_ms'):.1f} "
            f"blur={timing_value(timings, 'corner_detect_fast_gradient_blur_ms', 'corner_detect_variant_gradient_blur_ms'):.1f} "
            f"nms={timing_value(timings, 'corner_detect_fast_gradient_nms_ms', 'corner_detect_variant_gradient_nms_ms'):.1f} "
            f"scan={timing_value(timings, 'corner_detect_fast_gradient_scan_points_ms', 'corner_detect_variant_gradient_scan_points_ms'):.1f} "
            f"sort={timing_value(timings, 'corner_detect_fast_gradient_partial_sort_ms', 'corner_detect_variant_gradient_partial_sort_ms'):.1f}"
        )

    for i, line in enumerate(lines):
        cv2.putText(vis, line, (20, 35 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)


def draw_recovery_status(vis, dbg) -> None:
    if not dbg:
        return

    raw_n = len(getattr(dbg, "raw_candidates", []) or [])
    refined_n = sum(
        1 for c in (getattr(dbg, "refined_corners", []) or [])
        if bool(getattr(c, "valid", False))
    )
    lattice = "Y" if getattr(dbg, "has_lattice", False) else "N"
    rec_det = (
        getattr(dbg, "detection", None)
        if getattr(dbg, "has_detection", False)
        else None
    )
    rec_c = len(rec_det.corners) if rec_det else 0
    rec_cells = len(rec_det.cells) if rec_det else 0

    line = (
        f"recovery stages: raw={raw_n} refined={refined_n} "
        f"lattice={lattice} det={rec_c}/{rec_cells}"
    )
    cv2.putText(vis, line, (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)


def make_checkerboard_config():
    cfg = hydramarker_cpp.CheckerboardDetectorConfig()
    cfg.refresh_interval_frames = 1
    cfg.det_width = 0
    if hasattr(cfg, "max_undecodeable_tracking_frames"):
        cfg.max_undecodeable_tracking_frames = 12
    if hasattr(cfg, "max_low_corner_frames"):
        cfg.max_low_corner_frames = 12
    if hasattr(cfg, "use_tracking_roi_recovery"):
        cfg.use_tracking_roi_recovery = True
        cfg.tracking_recovery_roi_margin_cells = 3.0
        cfg.tracking_recovery_roi_min_margin_px = 80
        cfg.tracking_recovery_roi_max_area_ratio = 0.85
        if hasattr(cfg, "tracking_recovery_align_fail_full_retry_frames"):
            cfg.tracking_recovery_align_fail_full_retry_frames = 0
            cfg.tracking_recovery_align_fail_roi_margin_multiplier = 2.0
            if hasattr(cfg, "tracking_recovery_roi_fail_retry_margin_multiplier"):
                cfg.tracking_recovery_roi_fail_retry_margin_multiplier = 1.0
            cfg.tracking_recovery_roi_fail_full_retry_frames = 0
            if hasattr(cfg, "tracking_recovery_full_build_interval_frames"):
                cfg.tracking_recovery_full_build_interval_frames = 4
    cfg.recovery_correction_weight = 0.5
    cfg.recovery_correction_max_dist_rel = 0.6
    return cfg


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

    checker_cfg    = make_checkerboard_config()
    detector       = hydramarker_cpp.CheckerboardDetector(checker_cfg)
    debug_detector = hydramarker_cpp.CheckerboardDetector(checker_cfg)

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

            detect_t0 = time.perf_counter()
            det = detector.detect(img)
            detect_wall_ms = (time.perf_counter() - detect_t0) * 1000.0
            checker_timings = get_checkerboard_timings(detector, detect_wall_ms)

            # Debug overlay
            debug_det = None
            debug_recovery = None
            if debug_on or (_log_active and _log_debug_recovery):
                debug_detector.reset_tracking()
                debug_det = debug_detector.detect(img)
                debug_recovery = debug_detector.debug_recovery_stages(img)

            # Flicker/recovery-stage log
            log_frame(det, frame_idx, checker_timings, debug_recovery)

            if debug_on:
                draw_recovery_debug(vis, debug_recovery, det)

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

            draw_status(vis, mode_names[mode], det, debug_det, debug_on, checker_timings)
            if debug_on:
                draw_recovery_status(vis, debug_recovery)

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
            elif key == ord("g"):
                toggle_log_debug_recovery()
            elif key == ord("c"):
                toggle_log_corner_payload()
            elif key == ord("r"):
                detector.reset_tracking()
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
