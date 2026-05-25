import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

import hydramarker_cpp


def select_file(title: str, file_filter: str) -> Path | None:
    app = QApplication.instance() or QApplication(sys.argv)
    path, _ = QFileDialog.getOpenFileName(None, title, "", file_filter)
    if not path:
        return None
    return Path(path)


def round_int(x):
    return int(round(float(x)))


def put_text(img, text, pos, color=(255, 0, 255), scale=0.6, thickness=2):
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


def draw_checker_corners(vis, detection):
    for corner in detection.corners:
        u, v = pxy(corner.uv)
        cv2.circle(vis, (u, v), 4, (0, 255, 0), -1, cv2.LINE_AA)
        put_text(
            vis,
            f"{corner.i},{corner.j}",
            (u + 5, v - 5),
            (0, 255, 0),
            0.32,
            1,
        )


def draw_checker_cells(vis, detection):
    for cell in detection.cells:
        pts = cell_pts_np(cell)
        cv2.polylines(
            vis,
            [pts],
            isClosed=True,
            color=(0, 180, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        cx, cy = pxy(cell.center_uv)
        cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1, cv2.LINE_AA)


def build_dot_lookup(dot_detection):
    lookup = {}
    if dot_detection is None:
        return lookup
    for c in dot_detection.cells:
        lookup[(c.i, c.j)] = c
    return lookup


def draw_dot_cells(vis, checker_detection, dot_detection):
    dot_lookup = build_dot_lookup(dot_detection)

    for cell in checker_detection.cells:
        dot = dot_lookup.get((cell.i, cell.j))
        pts = cell_pts_np(cell)

        if dot is None:
            color = (80, 80, 80)
            value_text = "missing"
        elif dot.value < 0:
            color = (0, 0, 255)       # invalid = red
            value_text = "inv"
        elif dot.value == 0:
            color = (255, 0, 0)       # empty = blue
            value_text = "0"
        else:
            color = (255, 0, 255)     # dot = yellow/cyan-ish
            value_text = "1"

        cv2.polylines(
            vis,
            [pts],
            isClosed=True,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cx, cy = pxy(cell.center_uv)

        if dot is not None and dot.value == 1:
            cv2.circle(vis, (cx, cy), 5, color, -1, cv2.LINE_AA)
        elif dot is not None and dot.value == 0:
            cv2.circle(vis, (cx, cy), 4, color, 1, cv2.LINE_AA)
        else:
            cv2.drawMarker(
                vis,
                (cx, cy),
                color,
                cv2.MARKER_TILTED_CROSS,
                10,
                2,
                cv2.LINE_AA,
            )

        score = dot.score if dot is not None else 0.0

        put_text(
            vis,
            f"{value_text} {score:.0f}",
            (cx + 5, cy - 5),
            color,
            0.32,
            1,
        )


def draw_dot_centers_only(vis, dot_detection):
    if dot_detection is None:
        return

    for dot in dot_detection.cells:
        cx, cy = pxy(dot.center_uv)

        if dot.value < 0:
            color = (0, 0, 255)
            cv2.drawMarker(
                vis,
                (cx, cy),
                color,
                cv2.MARKER_TILTED_CROSS,
                10,
                2,
                cv2.LINE_AA,
            )
        elif dot.value == 0:
            color = (255, 0, 0)
            cv2.circle(vis, (cx, cy), 4, color, 1, cv2.LINE_AA)
        else:
            color = (255, 0, 255)
            cv2.circle(vis, (cx, cy), 5, color, -1, cv2.LINE_AA)


def dot_stats(dot_detection):
    if dot_detection is None:
        return {
            "total": 0,
            "invalid": 0,
            "empty": 0,
            "dot": 0,
        }

    values = [int(c.value) for c in dot_detection.cells]

    return {
        "total": len(values),
        "invalid": sum(v < 0 for v in values),
        "empty": sum(v == 0 for v in values),
        "dot": sum(v == 1 for v in values),
    }


def render(image, checker_detection, dot_detection, mode):
    vis = image.copy()

    if checker_detection is None:
        put_text(vis, "No checkerboard detection", (40, 60), (0, 0, 255))
        return vis

    if mode == 1:
        draw_checker_corners(vis, checker_detection)
        title = "1 checker corners"
    elif mode == 2:
        draw_checker_cells(vis, checker_detection)
        title = "2 checker cells"
    elif mode == 3:
        draw_dot_cells(vis, checker_detection, dot_detection)
        title = "3 dot cells: red invalid | blue empty | yellow dot"
    elif mode == 4:
        draw_checker_cells(vis, checker_detection)
        draw_checker_corners(vis, checker_detection)
        draw_dot_centers_only(vis, dot_detection)
        title = "4 checker + dot centers"
    else:
        draw_dot_cells(vis, checker_detection, dot_detection)
        draw_checker_corners(vis, checker_detection)
        title = "5 all"

    stats = dot_stats(dot_detection)

    overlay = vis.copy()
    cv2.rectangle(overlay, (20, 20), (980, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    put_text(vis, "HydraMarker DotDetector Static Debug", (40, 55))
    put_text(vis, title, (40, 88))
    put_text(
        vis,
        (
            f"checker corners={len(checker_detection.corners)} | "
            f"checker cells={len(checker_detection.cells)} | "
            f"rows={checker_detection.rows} | cols={checker_detection.cols}"
        ),
        (40, 120),
        scale=0.45,
        thickness=1,
    )
    put_text(
        vis,
        (
            f"dot cells={stats['total']} | "
            f"invalid={stats['invalid']} | "
            f"empty={stats['empty']} | "
            f"dot={stats['dot']}"
        ),
        (40, 145),
        scale=0.45,
        thickness=1,
    )
    put_text(
        vis,
        "1 corners | 2 cells | 3 dot states | 4 centers | 5 all | ESC quit",
        (40, 168),
        scale=0.42,
        thickness=1,
    )

    return vis


def print_debug(checker_detection, dot_detection):
    print()
    print("=" * 80)
    print("HYDRAMARKER DOT DETECTOR STATIC DEBUG")
    print("=" * 80)

    if checker_detection is None:
        print("No checkerboard detection.")
        return

    print(f"checker corners: {len(checker_detection.corners)}")
    print(f"checker cells:   {len(checker_detection.cells)}")
    print(f"checker rows:    {checker_detection.rows}")
    print(f"checker cols:    {checker_detection.cols}")
    print(f"tracking:        {checker_detection.tracking}")
    print(f"stable:          {checker_detection.stable}")

    if dot_detection is None:
        print("No dot detection.")
        return

    stats = dot_stats(dot_detection)

    print()
    print("Dot detection:")
    print(f"dot cells:       {stats['total']}")
    print(f"invalid:         {stats['invalid']}")
    print(f"empty:           {stats['empty']}")
    print(f"dot:             {stats['dot']}")
    print(f"grid cols:       {dot_detection.cols}")
    print(f"grid rows:       {dot_detection.rows}")
    print(f"origin i,j:      {dot_detection.origin_i}, {dot_detection.origin_j}")

    scores_valid = [c.score for c in dot_detection.cells if c.value >= 0]

    if scores_valid:
        print()
        print("Valid-cell contrast scores:")
        print(f"min:             {np.min(scores_valid):.2f}")
        print(f"mean:            {np.mean(scores_valid):.2f}")
        print(f"median:          {np.median(scores_valid):.2f}")
        print(f"max:             {np.max(scores_valid):.2f}")


def main():
    image_path = select_file(
        "Select HydraMarker image",
        "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
    )

    if image_path is None:
        print("No image selected.")
        return

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        print("Could not read image.")
        return

    checker = hydramarker_cpp.CheckerboardDetector()

    dot_cfg = hydramarker_cpp.DotDetectorConfig()
    dot_cfg.contrast_threshold = 10.0

    # First values to tune:
    dot_cfg.cell_geometry.min_area_px2 = 25.0
    dot_cfg.cell_geometry.max_opposite_edge_ratio = 1.6
    dot_cfg.cell_geometry.max_diagonal_ratio = 1.7
    dot_cfg.cell_geometry.min_angle_deg = 35.0
    dot_cfg.cell_geometry.max_angle_deg = 145.0
    dot_cfg.cell_geometry.max_opposite_edge_angle_diff_deg = 35.0

    dot_detector = hydramarker_cpp.DotDetector(dot_cfg)

    checker_detection = checker.detect(image)

    if checker_detection is None:
        dot_detection = None
    else:
        dot_detection = dot_detector.detect(image, checker_detection)

    print_debug(checker_detection, dot_detection)

    mode = 3
    vis = render(image, checker_detection, dot_detection, mode)

    window_name = "HydraMarker DotDetector Static Debug"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:
            break

        if key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5")]:
            mode = int(chr(key))
            vis = render(image, checker_detection, dot_detection, mode)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()