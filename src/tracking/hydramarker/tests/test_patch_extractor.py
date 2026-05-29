"""
Live PatchExtractor debug using the current C++ pipeline.

Controls:
    1       checker cells
    2       dot states
    3       patch outlines
    4       patch outlines + dot states
    5       selected patch detail
    6       all patches gallery + patch corners
    7       geometry check mode
    g       toggle geometry-valid / geometry-invalid patches in mode 7
    n       next selected patch
    b       previous selected patch
    w       gallery scroll up
    s       gallery scroll down
    p       pause / unpause
    SPACE   save current visualization as PNG
    ESC     exit
"""

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

from PySide6.QtWidgets import QApplication, QFileDialog

import hydramarker_cpp


def round_int(x):
    return int(round(float(x)))


def pxy(p):
    return round_int(p.x), round_int(p.y)


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


def build_checker_cell_lookup(checker_detection):
    if checker_detection is None:
        return {}

    return {(c.i, c.j): c for c in checker_detection.cells}


def build_dot_lookup(dot_detection):
    if dot_detection is None:
        return {}

    return {(c.row, c.col): c for c in dot_detection.cells}


def patch_color(index):
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (180, 80, 255),
        (80, 180, 255),
    ]
    return colors[index % len(colors)]


def geometry_filtered_patches(patches, show_geometry_valid=True):
    return [p for p in patches if bool(p.geometry_valid) == show_geometry_valid]


def draw_checker_cells(vis, checker_detection):
    if checker_detection is None:
        return

    for cell in checker_detection.cells:
        pts = cell_pts_np(cell)
        cv2.polylines(vis, [pts], True, (0, 180, 255), 1, cv2.LINE_AA)
        cx, cy = pxy(cell.center_uv)
        cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1, cv2.LINE_AA)


def draw_dot_states(vis, checker_detection, dot_detection):
    if checker_detection is None or dot_detection is None:
        return

    dot_lookup = build_dot_lookup(dot_detection)

    for cell in checker_detection.cells:
        dot = dot_lookup.get((cell.j, cell.i))
        pts = cell_pts_np(cell)

        if dot is None or not dot.valid:
            color = (80, 80, 80)
            label = "x"
        elif dot.ambiguous:
            color = (0, 255, 255)
            label = "?"
        elif dot.has_dot:
            color = (255, 0, 255)
            label = "1"
        else:
            color = (255, 0, 0)
            label = "0"

        cv2.polylines(vis, [pts], True, color, 1, cv2.LINE_AA)
        cx, cy = pxy(cell.center_uv)
        put_text(vis, label, (cx - 5, cy + 5), color, 0.35, 1)


def patch_cell(checker_cell_lookup, patch_row, patch_col):
    return checker_cell_lookup.get((patch_col, patch_row))


def patch_boundary_points(patch, checker_cell_lookup):
    k = patch.k
    r0 = patch.row
    c0 = patch.col

    tl = patch_cell(checker_cell_lookup, r0, c0)
    tr = patch_cell(checker_cell_lookup, r0, c0 + k - 1)
    br = patch_cell(checker_cell_lookup, r0 + k - 1, c0 + k - 1)
    bl = patch_cell(checker_cell_lookup, r0 + k - 1, c0)

    if tl is None or tr is None or br is None or bl is None:
        return None

    return np.array(
        [
            [tl.corner_uv[0].x, tl.corner_uv[0].y],
            [tr.corner_uv[1].x, tr.corner_uv[1].y],
            [br.corner_uv[2].x, br.corner_uv[2].y],
            [bl.corner_uv[3].x, bl.corner_uv[3].y],
        ],
        dtype=np.int32,
    )


def draw_patch_outlines(
    vis,
    checker_detection,
    patches,
    selected_index=None,
    color_by_geometry=False,
):
    if checker_detection is None:
        return

    checker_cell_lookup = build_checker_cell_lookup(checker_detection)

    for idx, patch in enumerate(patches):
        pts = patch_boundary_points(patch, checker_cell_lookup)
        if pts is None:
            continue

        selected = selected_index is not None and idx == selected_index

        if color_by_geometry:
            color = (0, 255, 0) if patch.geometry_valid else (0, 0, 255)
        else:
            color = (0, 255, 0) if selected else patch_color(idx)

        thickness = 4 if selected else 2

        cv2.polylines(vis, [pts], True, color, thickness, cv2.LINE_AA)

        center = pts.mean(axis=0)
        cx, cy = int(round(center[0])), int(round(center[1]))

        label = str(idx)
        if color_by_geometry:
            label += " G" if patch.geometry_valid else " X"

        put_text(vis, label, (cx - 12, cy + 5), color, 0.35, 1)


def draw_patch_corners(vis, checker_detection, patches, selected_index=None):
    if checker_detection is None:
        return

    checker_cell_lookup = build_checker_cell_lookup(checker_detection)

    for idx, patch in enumerate(patches):
        k = patch.k
        r0 = patch.row
        c0 = patch.col

        selected = selected_index is not None and idx == selected_index
        color = (0, 255, 0) if selected else patch_color(idx)
        radius = 5 if selected else 3
        thickness = -1 if selected else 1

        seen = set()

        for r in range(k):
            for c in range(k):
                cell = patch_cell(checker_cell_lookup, r0 + r, c0 + c)
                if cell is None:
                    continue

                for corner_uv in cell.corner_uv:
                    u, v = pxy(corner_uv)
                    key = (u, v)

                    if key in seen:
                        continue

                    seen.add(key)
                    cv2.circle(vis, (u, v), radius, color, thickness, cv2.LINE_AA)


def draw_selected_patch_detail(vis, patch, x0=25, y0=150, cell_size=42):
    if patch is None:
        return

    k = patch.k
    panel_w = max(k * cell_size + 32, 420)
    panel_h = k * cell_size + 190

    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, vis, 0.35, 0, vis)

    geom_color = (0, 255, 0) if patch.geometry_valid else (0, 0, 255)

    put_text(
        vis,
        f"Selected patch row={patch.row} col={patch.col} k={patch.k}",
        (x0 + 12, y0 + 25),
        (0, 255, 255),
        0.45,
        1,
    )

    put_text(
        vis,
        f"mean_score={patch.mean_score:.3f}",
        (x0 + 12, y0 + 48),
        (0, 255, 255),
        0.45,
        1,
    )

    put_text(
        vis,
        f"geometry_valid={patch.geometry_valid} quality={patch.geometry_quality:.3f}",
        (x0 + 12, y0 + 71),
        geom_color,
        0.45,
        1,
    )

    put_text(
        vis,
        f"area_std={patch.geometry.rel_area_std:.3f} edge_std={patch.geometry.rel_edge_std:.3f}",
        (x0 + 12, y0 + 94),
        geom_color,
        0.45,
        1,
    )

    put_text(
        vis,
        f"cells={patch.geometry.num_valid_cells}/{patch.geometry.num_cells}",
        (x0 + 12, y0 + 117),
        geom_color,
        0.45,
        1,
    )

    put_text(
        vis,
        f"angle=[{patch.geometry.min_cell_angle_deg:.1f},{patch.geometry.max_cell_angle_deg:.1f}]",
        (x0 + 12, y0 + 140),
        geom_color,
        0.45,
        1,
    )

    grid_x0 = x0 + 16
    grid_y0 = y0 + 162

    for r in range(k):
        for c in range(k):
            idx = r * k + c
            bit = int(patch.bits[idx])

            x1 = grid_x0 + c * cell_size
            y1 = grid_y0 + r * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            color = (255, 0, 255) if bit == 1 else (255, 0, 0)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            put_text(vis, str(bit), (x1 + 14, y1 + 28), color, 0.75, 2)


def draw_patch_mini_grid(canvas, patch, x0, y0, cell_size, selected=False):
    k = patch.k

    border_color = (0, 255, 0) if patch.geometry_valid else (0, 0, 255)

    if selected:
        border_color = (0, 255, 255)

    text_color = (0, 255, 255) if selected else (220, 220, 220)

    title = (
        f"#{patch.row},{patch.col} "
        f"g={patch.geometry_quality:.2f} "
        f"{'OK' if patch.geometry_valid else 'BAD'}"
    )

    put_text(canvas, title, (x0, y0 - 5), text_color, 0.32, 1)

    for r in range(k):
        for c in range(k):
            idx = r * k + c
            bit = int(patch.bits[idx])

            x1 = x0 + c * cell_size
            y1 = y0 + r * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            fill = (80, 0, 80) if bit == 1 else (60, 20, 20)
            edge = (255, 0, 255) if bit == 1 else (255, 0, 0)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), edge, 1)
            put_text(canvas, str(bit), (x1 + 6, y1 + cell_size - 7), edge, 0.38, 1)

    cv2.rectangle(
        canvas,
        (x0 - 3, y0 - 22),
        (x0 + k * cell_size + 3, y0 + k * cell_size + 3),
        border_color,
        2 if selected else 1,
    )


def draw_patch_gallery(vis, patches, selected_index, scroll_offset):
    h, w = vis.shape[:2]

    panel_w = 430
    x0 = max(0, w - panel_w)
    y0 = 0

    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.78, vis, 0.22, 0, vis)

    put_text(vis, "All extracted patches", (x0 + 18, 34), (0, 255, 255), 0.62, 2)
    put_text(vis, f"count={len(patches)} | scroll={scroll_offset}", (x0 + 18, 62), (0, 255, 255), 0.45, 1)

    if not patches:
        put_text(vis, "No patches in this view/filter", (x0 + 18, 105), (0, 160, 255), 0.55, 2)
        return

    k = patches[0].k
    cell_size = 18 if k <= 4 else 14
    grid_w = k * cell_size
    tile_w = grid_w + 42
    tile_h = k * cell_size + 44

    cols = max(1, (panel_w - 30) // tile_w)
    rows_visible = max(1, (h - 90) // tile_h)
    max_visible = rows_visible * cols

    start = max(0, min(scroll_offset, max(0, len(patches) - max_visible)))
    end = min(len(patches), start + max_visible)

    for local_idx, patch_idx in enumerate(range(start, end)):
        patch = patches[patch_idx]

        rr = local_idx // cols
        cc = local_idx % cols

        px = x0 + 18 + cc * tile_w
        py = 100 + rr * tile_h

        draw_patch_mini_grid(
            vis,
            patch,
            px,
            py,
            cell_size,
            selected=(patch_idx == selected_index),
        )

        put_text(vis, f"id {patch_idx}", (px, py + k * cell_size + 20), (200, 200, 200), 0.34, 1)


def patch_stats(patches):
    if not patches:
        return "patches=0 | geom valid=0 invalid=0 | score avg=0.00 max=0.00"

    scores = np.array([p.mean_score for p in patches], dtype=np.float32)

    geom_valid = sum(bool(p.geometry_valid) for p in patches)
    geom_invalid = len(patches) - geom_valid

    return (
        f"patches={len(patches)} | "
        f"geom valid={geom_valid} invalid={geom_invalid} | "
        f"score avg={float(np.mean(scores)):.2f} | "
        f"score max={float(np.max(scores)):.2f}"
    )


def draw_info_panel(
    vis,
    mode_name,
    checker_detection,
    dot_detection,
    patches,
    display_patches,
    patch_size,
    selected_index,
    paused,
    show_geometry_valid=True,
):
    lines = [
        "HydraMarker PatchExtractor Live Debug",
        f"mode: {mode_name}",
        f"k={patch_size}",
        "1 cells | 2 dots | 3 patches | 4 patches+dots | 5 selected | 6 gallery | 7 geometry | g toggle geom valid/invalid | n/b select | w/s scroll | p pause | SPACE save | ESC quit",
    ]

    if mode_name == "geometry check":
        lines.append(
            f"geometry filter: {'VALID patches' if show_geometry_valid else 'INVALID patches'}"
        )

    if paused:
        lines.append("*** PAUSED ***")

    if checker_detection is None:
        lines.append("checker: no detection")
    else:
        lines.append(
            f"checker corners={len(checker_detection.corners)} | "
            f"cells={len(checker_detection.cells)} | "
            f"rows={checker_detection.rows} | cols={checker_detection.cols} | "
            f"tracking={checker_detection.tracking} | stable={checker_detection.stable}"
        )

    if dot_detection is None:
        lines.append("dots: none")
    else:
        valid = sum(c.valid for c in dot_detection.cells)
        ambiguous = sum(c.valid and c.ambiguous for c in dot_detection.cells)
        dots = sum(c.valid and c.has_dot for c in dot_detection.cells)

        lines.append(
            f"dot cells={len(dot_detection.cells)} | "
            f"valid={valid} | dots={dots} | ambiguous={ambiguous}"
        )

    lines.append(patch_stats(patches))

    if mode_name == "geometry check":
        lines.append(f"displayed in filter={len(display_patches)}")

    if display_patches and 0 <= selected_index < len(display_patches):
        p = display_patches[selected_index]
        lines.append(
            f"selected={selected_index}/{len(display_patches)-1} | "
            f"row={p.row} col={p.col} score={p.mean_score:.2f} | "
            f"geom={p.geometry_valid} q={p.geometry_quality:.2f}"
        )

    lines.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    x0, y0 = 25, 25
    line_h = 25
    pad = 14
    width = 1440
    height = pad * 2 + line_h * len(lines)

    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    y = y0 + pad + 18

    for line in lines:
        color = (0, 255, 255)

        if "PAUSED" in line:
            color = (0, 0, 255)
        elif "patches=0" in line:
            color = (0, 160, 255)
        elif "INVALID patches" in line:
            color = (0, 0, 255)
        elif "VALID patches" in line:
            color = (0, 255, 0)

        put_text(vis, line, (x0 + pad, y), color, 0.52, 1)
        y += line_h


def render(
    image,
    checker_detection,
    dot_detection,
    patches,
    patch_size,
    mode,
    selected_index,
    gallery_scroll,
    paused,
    show_geometry_valid=True,
):
    vis = image.copy()

    mode_names = {
        1: "checker cells",
        2: "dot states",
        3: "patch outlines",
        4: "patch outlines + dot states",
        5: "selected patch detail",
        6: "all patches gallery + patch corners",
        7: "geometry check",
    }

    display_patches = patches

    if mode == 7:
        display_patches = geometry_filtered_patches(
            patches,
            show_geometry_valid=show_geometry_valid,
        )

        if selected_index >= len(display_patches):
            selected_index = max(0, len(display_patches) - 1)

    if mode == 1:
        draw_checker_cells(vis, checker_detection)

    elif mode == 2:
        draw_dot_states(vis, checker_detection, dot_detection)

    elif mode == 3:
        draw_patch_outlines(vis, checker_detection, display_patches, selected_index)

    elif mode == 4:
        draw_dot_states(vis, checker_detection, dot_detection)
        draw_patch_outlines(vis, checker_detection, display_patches, selected_index)

    elif mode == 5:
        draw_dot_states(vis, checker_detection, dot_detection)
        draw_patch_outlines(vis, checker_detection, display_patches, selected_index)
        if display_patches:
            draw_selected_patch_detail(vis, display_patches[selected_index])

    elif mode == 6:
        draw_dot_states(vis, checker_detection, dot_detection)
        draw_patch_outlines(vis, checker_detection, display_patches, selected_index)
        draw_patch_corners(vis, checker_detection, display_patches, selected_index)
        draw_patch_gallery(vis, display_patches, selected_index, gallery_scroll)

    elif mode == 7:
        draw_dot_states(vis, checker_detection, dot_detection)
        draw_patch_outlines(
            vis,
            checker_detection,
            display_patches,
            selected_index,
            color_by_geometry=True,
        )
        draw_patch_corners(vis, checker_detection, display_patches, selected_index)
        draw_patch_gallery(vis, display_patches, selected_index, gallery_scroll)

    else:
        draw_patch_outlines(vis, checker_detection, display_patches, selected_index)

    draw_info_panel(
        vis,
        mode_names.get(mode, "unknown"),
        checker_detection,
        dot_detection,
        patches,
        display_patches,
        patch_size,
        selected_index if display_patches else -1,
        paused,
        show_geometry_valid=show_geometry_valid,
    )

    return vis


def save_png(vis, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = output_dir / f"patch_extractor_live_{ts}.png"

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
    dot_cfg.uncertainty_low = 0.40
    dot_cfg.uncertainty_high = 0.55
    dot_cfg.warmup_frames = 1

    return hydramarker_cpp.DotDetector(dot_cfg)


def choose_field_file_qt():
    app = QApplication.instance()

    if app is None:
        app = QApplication([])

    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select HydraMarker .field file",
        "",
        "HydraMarker field (*.field);;All files (*.*)",
    )

    if not path:
        raise RuntimeError("No .field file selected.")

    return Path(path)


def clamp_selection(selected_index, gallery_scroll, patches):
    if patches:
        selected_index = max(0, min(selected_index, len(patches) - 1))
        gallery_scroll = max(0, min(gallery_scroll, len(patches) - 1))
    else:
        selected_index = 0
        gallery_scroll = 0

    return selected_index, gallery_scroll


def main():
    field_path = choose_field_file_qt()

    marker_field = hydramarker_cpp.MarkerField.loadFromFile(str(field_path))
    patch_size = marker_field.patchSize()

    print(f"Loaded field: {field_path}")
    print(f"field size: {marker_field.width()} x {marker_field.height()}")
    print(f"patch size k: {patch_size}")

    checker_detector = hydramarker_cpp.CheckerboardDetector()
    dot_detector = create_dot_detector()
    patch_extractor = hydramarker_cpp.PatchExtractor()

    output_dir = Path(__file__).resolve().parent / "patch_extractor_live_snapshots"

    pipe = rs.pipeline()
    cfg = rs.config()

    cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    pipe.start(cfg)

    window_name = "HydraMarker PatchExtractor Live Debug"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    mode = 6
    paused = False
    selected_index = 0
    gallery_scroll = 0
    show_geometry_valid = True

    last_img = None
    last_checker_detection = None
    last_dot_detection = None
    last_patches = []
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
                    dot_detection = dot_detector.detect(img, checker_detection)
                    patches = patch_extractor.extract(dot_detection, patch_size)
                else:
                    dot_detection = None
                    patches = []

                if mode == 7:
                    display_patches = geometry_filtered_patches(
                        patches,
                        show_geometry_valid=show_geometry_valid,
                    )
                else:
                    display_patches = patches

                selected_index, gallery_scroll = clamp_selection(
                    selected_index,
                    gallery_scroll,
                    display_patches,
                )

                last_img = img.copy()
                last_checker_detection = checker_detection
                last_dot_detection = dot_detection
                last_patches = patches

            if last_img is None:
                continue

            if mode == 7:
                display_patches = geometry_filtered_patches(
                    last_patches,
                    show_geometry_valid=show_geometry_valid,
                )
            else:
                display_patches = last_patches

            selected_index, gallery_scroll = clamp_selection(
                selected_index,
                gallery_scroll,
                display_patches,
            )

            last_vis = render(
                last_img,
                last_checker_detection,
                last_dot_detection,
                last_patches,
                patch_size,
                mode,
                selected_index,
                gallery_scroll,
                paused,
                show_geometry_valid,
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
            ]:
                mode = int(chr(key))
                selected_index = 0
                gallery_scroll = 0

            elif key == ord("g"):
                if mode == 7:
                    show_geometry_valid = not show_geometry_valid
                    selected_index = 0
                    gallery_scroll = 0

            elif key == ord("p"):
                paused = not paused

            elif key == ord("n"):
                if mode == 7:
                    current_patches = geometry_filtered_patches(
                        last_patches,
                        show_geometry_valid=show_geometry_valid,
                    )
                else:
                    current_patches = last_patches

                if current_patches:
                    selected_index = (selected_index + 1) % len(current_patches)
                    gallery_scroll = min(gallery_scroll, selected_index)

            elif key == ord("b"):
                if mode == 7:
                    current_patches = geometry_filtered_patches(
                        last_patches,
                        show_geometry_valid=show_geometry_valid,
                    )
                else:
                    current_patches = last_patches

                if current_patches:
                    selected_index = (selected_index - 1) % len(current_patches)
                    gallery_scroll = min(gallery_scroll, selected_index)

            elif key == ord("w"):
                gallery_scroll = max(0, gallery_scroll - 6)

            elif key == ord("s"):
                if mode == 7:
                    current_patches = geometry_filtered_patches(
                        last_patches,
                        show_geometry_valid=show_geometry_valid,
                    )
                else:
                    current_patches = last_patches

                if current_patches:
                    gallery_scroll = min(len(current_patches) - 1, gallery_scroll + 6)

            elif key == 32 and last_vis is not None:
                save_png(last_vis, output_dir)

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()