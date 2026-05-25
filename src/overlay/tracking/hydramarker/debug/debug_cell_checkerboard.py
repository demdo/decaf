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


def put_text(img, text, pos, color=(0, 255, 255), scale=0.6):
    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_corners(vis, detection):
    for idx, corner in enumerate(detection.corners):
        u, v = corner.uv

        cv2.circle(
            vis,
            (round_int(u), round_int(v)),
            5,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )

        put_text(
            vis,
            f"{corner.i},{corner.j}",
            (round_int(u) + 6, round_int(v) - 6),
            (0, 255, 0),
            0.35,
        )


def draw_cells(vis, detection):
    for cell in detection.cells:
        pts = np.array(
            [
                [cell.corner_uv[0][0], cell.corner_uv[0][1]],
                [cell.corner_uv[1][0], cell.corner_uv[1][1]],
                [cell.corner_uv[2][0], cell.corner_uv[2][1]],
                [cell.corner_uv[3][0], cell.corner_uv[3][1]],
            ],
            dtype=np.int32,
        )

        # Order must be p00 -> p10 -> p11 -> p01
        cv2.polylines(
            vis,
            [pts],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        cx, cy = cell.center_uv

        cv2.circle(
            vis,
            (round_int(cx), round_int(cy)),
            3,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

        put_text(
            vis,
            f"{cell.i},{cell.j}",
            (round_int(cx) + 5, round_int(cy) - 5),
            (0, 180, 255),
            0.35,
        )


def render(image, detection, mode):
    vis = image.copy()

    if detection is None:
        put_text(vis, "No checkerboard detection", (40, 60), (0, 0, 255))
        return vis

    if mode == 1:
        draw_corners(vis, detection)
        title = "1 corners"
    elif mode == 2:
        draw_cells(vis, detection)
        title = "2 cells"
    else:
        draw_cells(vis, detection)
        draw_corners(vis, detection)
        title = "3 corners + cells"

    overlay = vis.copy()
    cv2.rectangle(overlay, (20, 20), (850, 145), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    put_text(vis, "HydraMarker Checkerboard Static Debug", (40, 55))
    put_text(vis, title, (40, 88))
    put_text(
        vis,
        f"corners={len(detection.corners)} | cells={len(detection.cells)} | rows={detection.rows} | cols={detection.cols} | tracking={detection.tracking} | stable={detection.stable}",
        (40, 120),
        scale=0.45,
    )
    put_text(
        vis,
        "1 corners | 2 cells | 3 both | ESC quit",
        (40, 145),
        scale=0.45,
    )

    return vis


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

    detector = hydramarker_cpp.CheckerboardDetector()

    detection = detector.detect(image)

    if detection is None:
        print("No detection.")
    else:
        print()
        print("=" * 80)
        print("HYDRAMARKER CHECKERBOARD DEBUG")
        print("=" * 80)
        print(f"corners:  {len(detection.corners)}")
        print(f"cells:    {len(detection.cells)}")
        print(f"rows:     {detection.rows}")
        print(f"cols:     {detection.cols}")
        print(f"tracking: {detection.tracking}")
        print(f"stable:   {detection.stable}")

    mode = 3
    vis = render(image, detection, mode)

    cv2.namedWindow("HydraMarker Checkerboard Static Debug", cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow("HydraMarker Checkerboard Static Debug", vis)
        key = cv2.waitKey(0) & 0xFF

        if key == 27:
            break

        if key in [ord("1"), ord("2"), ord("3")]:
            mode = int(chr(key))
            vis = render(image, detection, mode)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()