from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

import hydramarker_cpp


def choose_file_qt(title, file_filter):
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    path, _ = QFileDialog.getOpenFileName(None, title, "", file_filter)
    if not path:
        raise RuntimeError(f"No file selected: {title}")

    return Path(path)


def put_text(img, text, pos, color=(0, 255, 255), scale=0.7, thickness=2):
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


def draw_global_patch(marker_img, field, global_row, global_col, k, color=(0, 255, 0)):
    vis = marker_img.copy()

    img_h, img_w = vis.shape[:2]

    field_w = field.width()
    field_h = field.height()

    cell_w = img_w / float(field_w)
    cell_h = img_h / float(field_h)

    x1 = int(round(global_col * cell_w))
    y1 = int(round(global_row * cell_h))
    x2 = int(round((global_col + k) * cell_w))
    y2 = int(round((global_row + k) * cell_h))

    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)

    put_text(
        vis,
        f"example patch row={global_row} col={global_col} k={k}",
        (x1 + 8, max(30, y1 - 10)),
        color,
    )

    for r in range(k + 1):
        y = int(round((global_row + r) * cell_h))
        cv2.line(vis, (x1, y), (x2, y), color, 1, cv2.LINE_AA)

    for c in range(k + 1):
        x = int(round((global_col + c) * cell_w))
        cv2.line(vis, (x, y1), (x, y2), color, 1, cv2.LINE_AA)

    return vis


def main():
    field_path = choose_file_qt(
        "Select HydraMarker .field file",
        "HydraMarker field (*.field);;All files (*.*)",
    )

    marker_img_path = choose_file_qt(
        "Select original marker image",
        "Images (*.png *.jpg *.jpeg *.bmp);;All files (*.*)",
    )

    field = hydramarker_cpp.MarkerField.loadFromFile(str(field_path))
    marker_img = cv2.imread(str(marker_img_path), cv2.IMREAD_COLOR)

    if marker_img is None:
        raise RuntimeError(f"Could not load marker image: {marker_img_path}")

    k = field.patchSize()

    # Beispielposition im globalen Markerfeld
    example_global_row = 2
    example_global_col = 3

    print(f"Loaded field: {field_path}")
    print(f"Loaded marker image: {marker_img_path}")
    print(f"field size: {field.width()} x {field.height()}")
    print(f"patch size k: {k}")
    print(f"example patch: row={example_global_row}, col={example_global_col}")

    vis = draw_global_patch(
        marker_img,
        field,
        example_global_row,
        example_global_col,
        k,
    )

    cv2.namedWindow("debug_patch_decoder_marker_view", cv2.WINDOW_NORMAL)
    cv2.imshow("debug_patch_decoder_marker_view", vis)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()