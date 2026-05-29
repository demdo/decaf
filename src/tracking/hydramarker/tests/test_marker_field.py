"""
MarkerField test.

This script verifies the first HydraMarker pipeline component: MarkerField.

The MarkerField stores the global binary marker pattern. A local k x k patch
can later be extracted from the camera image and matched against this global
field. Since the marker can appear in any image rotation, patch matching must
work for all four rotations:

    0°, 90°, 180°, 270°

This test loads a .field file, extracts one known patch from the global field,
and checks whether the Python/C++ MarkerField implementation can find the patch
again under all four rotations.
"""

import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from hydramarker.field import MarkerField


def select_file(title: str, file_filter: str) -> Path | None:
    """
    Open a Qt file dialog and return the selected file path.
    """
    app = QApplication.instance()
    owns_app = app is None

    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        file_filter,
    )

    if owns_app:
        app.quit()

    if not path:
        return None

    return Path(path)


def load_field_from_field_file(path: Path) -> tuple[np.ndarray, int]:
    """
    Load the global binary marker field from a .field file.

    The expected file layout is:

        width height
        width * height binary values
        template_count
        patch_width patch_height
        ...

    For the current tracker, only the global state matrix and the square patch
    size are required.
    """
    tokens = path.read_text().split()
    idx = 0

    width = int(tokens[idx])
    idx += 1

    height = int(tokens[idx])
    idx += 1

    values = list(map(int, tokens[idx:idx + width * height]))
    idx += width * height

    field = np.array(values, dtype=np.uint8).reshape(height, width)

    idx += 1  # Skip template count.

    patch_width = int(tokens[idx])
    idx += 1

    patch_height = int(tokens[idx])
    idx += 1

    assert patch_width == patch_height

    return field, patch_width


def load_patch_from_field_file(
    path: Path,
    x0: int = 0,
    y0: int = 0,
) -> np.ndarray:
    """
    Extract a real k x k patch from the global marker field.

    This simulates the type of local binary patch that will later be produced
    by the DotDetector and PatchExtractor.
    """
    field, patch_size = load_field_from_field_file(path)

    return field[
        y0:y0 + patch_size,
        x0:x0 + patch_size,
    ]


def test_rotations(field: MarkerField, patch: np.ndarray) -> None:
    """
    Test whether the same local patch can be found in all four rotations.
    """
    print("\n=== Rotation test ===")

    rotations = [
        ("0 deg", patch),
        ("90 deg", np.rot90(patch, 1)),
        ("180 deg", np.rot90(patch, 2)),
        ("270 deg", np.rot90(patch, 3)),
    ]

    for name, rotated_patch in rotations:
        print(f"\n--- Testing rotation {name} ---")
        print(rotated_patch)

        matches = field.find_patch(rotated_patch)

        if not matches:
            print("No matches found.")
            continue

        for match in matches:
            print(
                "Match: "
                f"x={match['x']} "
                f"y={match['y']} "
                f"rotation={match['rotation']} deg"
            )


def main() -> None:
    """
    Load a marker field, extract a patch, and verify patch lookup.
    """
    print("=== MarkerField test ===")

    field_path = select_file(
        title="Select HydraMarker field file",
        file_filter="HydraMarker Field (*.field);;All Files (*)",
    )

    if field_path is None:
        print("No file selected.")
        return

    print(f"Selected field file: {field_path}")

    state_matrix, patch_size = load_field_from_field_file(field_path)

    print("\n=== Global state matrix ===")
    print(state_matrix)
    print(f"\nPatch size: {patch_size} x {patch_size}")

    field = MarkerField.from_file(str(field_path))

    patch = load_patch_from_field_file(field_path, x0=0, y0=0)

    print("\n=== Original local patch ===")
    print(patch)

    matches = field.find_patch(patch)

    print("\n=== Matches for original patch ===")
    if not matches:
        print("No matches found.")
    else:
        for match in matches:
            print(match)

    test_rotations(field, patch)


if __name__ == "__main__":
    main()