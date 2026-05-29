from __future__ import annotations

from pathlib import Path
import json
import sys

import cv2
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication, QFileDialog


def choose_file_qt(
    title: str,
    file_filter: str,
) -> Path:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        file_filter,
    )

    if not path:
        raise RuntimeError(f"No file selected: {title}")

    return Path(path)


def main() -> None:
    image_path = choose_file_qt(
        "Select HydraMarker marker image .png",
        "PNG images (*.png)",
    )

    json_path = image_path.with_suffix(".json")

    if not json_path.exists():
        json_path = choose_file_qt(
            "Select matching marker .json",
            "JSON files (*.json)",
        )

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    rows = int(meta["rows"])
    cols = int(meta["cols"])
    cell_px = int(meta["cell_px"])

    #
    # For the generated marker image:
    #
    # rows x cols = cell grid.
    #
    # Outer border corners are not detectable by the current HydraMarker
    # checkerboard/corner logic.
    #
    # Therefore the detectable internal corner grid is:
    #
    #   (rows - 1) x (cols - 1)
    #
    detectable_corner_rows = rows - 1
    detectable_corner_cols = cols - 1

    print()
    print("=" * 70)
    print("HYDRAMARKER MARKER FRAME DEBUG")
    print("=" * 70)
    print(f"image                 : {image_path}")
    print(f"json                  : {json_path}")
    print(f"cell rows x cols      : {rows} x {cols}")
    print(f"detectable corners    : {detectable_corner_rows} x {detectable_corner_cols}")
    print(f"cell_px               : {cell_px}")
    print()
    print("Coordinate convention shown:")
    print("  origin = detectable corner (row=0, col=0)")
    print("  x      = positive global_col")
    print("  y      = positive global_row")
    print()
    print("ID convention:")
    print("  marker_id = row * detectable_corner_cols + col")
    print("=" * 70)
    print()

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_rgb)

    for row in range(detectable_corner_rows):
        for col in range(detectable_corner_cols):
            marker_id = row * detectable_corner_cols + col

            #
            # Internal detectable corners.
            #
            # If the full image has cell-grid corners at:
            #   full_row = 0..rows
            #   full_col = 0..cols
            #
            # then detectable internal corners are:
            #   full_row = 1..rows-1
            #   full_col = 1..cols-1
            #
            x_px = (col + 1) * cell_px
            y_px = (row + 1) * cell_px

            is_origin = row == 0 and col == 0

            ax.scatter(
                x_px,
                y_px,
                c="red" if is_origin else "lime",
                s=140 if is_origin else 35,
                marker="o",
            )

            label = f"{row},{col}\nID {marker_id}"

            ax.text(
                x_px + 6,
                y_px + 6,
                label,
                color="white",
                fontsize=6,
                bbox={
                    "facecolor": "black",
                    "alpha": 0.65,
                    "edgecolor": "none",
                    "pad": 1.0,
                },
            )

    origin_x = cell_px
    origin_y = cell_px

    axis_len = 2.5 * cell_px

    ax.arrow(
        origin_x,
        origin_y,
        axis_len,
        0,
        color="red",
        width=4,
        length_includes_head=True,
    )

    ax.text(
        origin_x + axis_len + 15,
        origin_y,
        "+x / col",
        color="red",
        fontsize=16,
        weight="bold",
    )

    ax.arrow(
        origin_x,
        origin_y,
        0,
        axis_len,
        color="cyan",
        width=4,
        length_includes_head=True,
    )

    ax.text(
        origin_x,
        origin_y + axis_len + 35,
        "+y / row",
        color="cyan",
        fontsize=16,
        weight="bold",
    )

    ax.set_title(
        "HydraMarker detectable corner coordinate system\n"
        "origin = row 0, col 0"
    )

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()