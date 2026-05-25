"""
HydraMarker planar marker generator.

Creates:
- .field  : binary HydraMarker state matrix + patch shape
- .png    : printable marker image with checkerboard + dots
- .json   : metric marker metadata for 2D-3D correspondences
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    from .field import MarkerField
except ImportError:
    # Allows running directly as:
    # python generate_planar_marker.py
    THIS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = THIS_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from hydramarker.field import MarkerField


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = THIS_DIR / "data"


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def render_marker_image(
    field: np.ndarray,
    cell_px: int,
    dot_radius_rel: float,
) -> np.ndarray:
    """
    Render checkerboard marker with opposite-color dots.

    field[row, col] == 1 -> draw dot in that cell
    field[row, col] == 0 -> no dot

    No border is added.
    """
    field = np.asarray(field, dtype=np.uint8)

    rows, cols = field.shape
    img_h = rows * cell_px
    img_w = cols * cell_px

    img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            y0 = row * cell_px
            x0 = col * cell_px
            y1 = y0 + cell_px
            x1 = x0 + cell_px

            black_cell = ((row + col) % 2 == 0)
            img[y0:y1, x0:x1] = (0, 0, 0) if black_cell else (255, 255, 255)

    dot_radius_px = max(1, int(round(dot_radius_rel * cell_px)))

    for row in range(rows):
        for col in range(cols):
            if int(field[row, col]) != 1:
                continue

            cx = col * cell_px + cell_px // 2
            cy = row * cell_px + cell_px // 2

            black_cell = ((row + col) % 2 == 0)
            dot_color = (255, 255, 255) if black_cell else (0, 0, 0)

            cv2.circle(
                img,
                (cx, cy),
                dot_radius_px,
                dot_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    return img


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------

def save_field_file(path: Path, field: np.ndarray, patch_size: int) -> None:
    """
    Save .field in the format used by the current HydraMarker C++ loader:

    cols rows
    row-major field values
    number_of_tag_shapes
    patch_width patch_height
    row-major patch shape values
    """
    field = np.asarray(field, dtype=np.uint8)

    if field.ndim != 2:
        raise ValueError("field must be a 2D array")

    rows, cols = field.shape

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{cols} {rows}\n")

        for row in range(rows):
            values = [str(int(field[row, col])) for col in range(cols)]
            f.write(" ".join(values) + "\n")

        f.write("1\n")
        f.write(f"{patch_size} {patch_size}\n")

        for _row in range(patch_size):
            values = ["1"] * patch_size
            f.write(" ".join(values) + "\n")


def save_meta_file(
    path: Path,
    rows: int,
    cols: int,
    patch_size: int,
    square_size_cm: float,
    cell_px: int,
    dot_radius_rel: float,
    marker_name: str,
) -> None:
    square_size_mm = square_size_cm * 10.0

    meta = {
        "name": marker_name,
        "rows": rows,
        "cols": cols,
        "patch_size": patch_size,
        "square_size_cm": square_size_cm,
        "square_size_mm": square_size_mm,
        "has_border": False,
        "origin": "top_left",
        "x_axis": "col_positive",
        "y_axis": "row_positive",
        "z_axis": "out_of_plane",
        "corner_rows": rows + 1,
        "corner_cols": cols + 1,
        "cell_px": cell_px,
        "dot_radius_rel": dot_radius_rel,
        "coordinate_mapping": {
            "global_corner_row_col_to_xyz_mm": {
                "X": "col * square_size_mm",
                "Y": "row * square_size_mm",
                "Z": "0",
            }
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def sanitize_marker_name(name: str) -> str:
    name = name.strip()

    if not name:
        return "marker"

    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-", "."):
            safe.append(ch)
        else:
            safe.append("_")

    return "".join(safe)


# -----------------------------------------------------------------------------
# Qt helpers
# -----------------------------------------------------------------------------

def cv_bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape

    qimg = QImage(
        img_rgb.data,
        w,
        h,
        ch * w,
        QImage.Format_RGB888,
    ).copy()

    return QPixmap.fromImage(qimg)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

class GeneratePlanarMarkerUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("HydraMarker - Planar Marker Generator")

        self.current_field: np.ndarray | None = None
        self.current_img: np.ndarray | None = None

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(2, 500)
        self.rows_spin.setValue(20)

        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(2, 500)
        self.cols_spin.setValue(20)

        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(2, 20)
        self.patch_size_spin.setValue(4)

        self.square_size_cm_spin = QDoubleSpinBox()
        self.square_size_cm_spin.setRange(0.001, 1000.0)
        self.square_size_cm_spin.setDecimals(4)
        self.square_size_cm_spin.setSingleStep(0.1)
        self.square_size_cm_spin.setValue(1.25)

        self.cell_px_spin = QSpinBox()
        self.cell_px_spin.setRange(10, 1000)
        self.cell_px_spin.setValue(80)

        self.dot_radius_rel_spin = QDoubleSpinBox()
        self.dot_radius_rel_spin.setRange(0.05, 0.45)
        self.dot_radius_rel_spin.setDecimals(3)
        self.dot_radius_rel_spin.setSingleStep(0.01)
        self.dot_radius_rel_spin.setValue(0.22)

        self.max_ms_spin = QDoubleSpinBox()
        self.max_ms_spin.setRange(100.0, 3600000000.0)
        self.max_ms_spin.setDecimals(0)
        self.max_ms_spin.setSingleStep(1000.0)
        self.max_ms_spin.setValue(60000.0)

        self.max_trial_spin = QSpinBox()
        self.max_trial_spin.setRange(1, 2147483647)
        self.max_trial_spin.setValue(100000)

        self.marker_name_edit = QLineEdit()
        self.marker_name_edit.setText("marker_20x20_p4_12p5mm")

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(str(DEFAULT_DATA_DIR))

        self.browse_button = QPushButton("Browse")
        self.generate_button = QPushButton("Generate Preview")
        self.save_button = QPushButton("Save .field / .png / .json")
        self.save_button.setEnabled(False)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)

        self.preview_label = QLabel("Generate a marker to preview it here.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(720, 720)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet(
            "background-color: #202020; color: #dddddd; border: 1px solid #555;"
        )

        self._build_layout()
        self._connect_signals()

    def _build_layout(self) -> None:
        form = QFormLayout()
        form.addRow("Rows / Cells Y", self.rows_spin)
        form.addRow("Cols / Cells X", self.cols_spin)
        form.addRow("Patch size k", self.patch_size_spin)
        form.addRow("Square size [cm]", self.square_size_cm_spin)
        form.addRow("Cell pixels", self.cell_px_spin)
        form.addRow("Dot radius rel.", self.dot_radius_rel_spin)
        form.addRow("Max generation time [ms]", self.max_ms_spin)
        form.addRow("Max trials", self.max_trial_spin)
        form.addRow("Marker name", self.marker_name_edit)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit)
        output_row.addWidget(self.browse_button)
        form.addRow("Output folder", output_row)

        params_group = QGroupBox("Marker Parameters")
        params_group.setLayout(form)

        button_row = QHBoxLayout()
        button_row.addWidget(self.generate_button)
        button_row.addWidget(self.save_button)

        left_layout = QVBoxLayout()
        left_layout.addWidget(params_group)
        left_layout.addLayout(button_row)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, stretch=0)
        main_layout.addWidget(self.preview_label, stretch=1)

        self.setLayout(main_layout)

    def _connect_signals(self) -> None:
        self.browse_button.clicked.connect(self.on_browse)
        self.generate_button.clicked.connect(self.on_generate)
        self.save_button.clicked.connect(self.on_save)

    def _read_params(self) -> dict:
        rows = int(self.rows_spin.value())
        cols = int(self.cols_spin.value())
        patch_size = int(self.patch_size_spin.value())

        if patch_size > rows or patch_size > cols:
            raise ValueError("Patch size must be <= rows and <= cols.")

        marker_name = sanitize_marker_name(self.marker_name_edit.text())

        return {
            "rows": rows,
            "cols": cols,
            "patch_size": patch_size,
            "square_size_cm": float(self.square_size_cm_spin.value()),
            "cell_px": int(self.cell_px_spin.value()),
            "dot_radius_rel": float(self.dot_radius_rel_spin.value()),
            "max_ms": float(self.max_ms_spin.value()),
            "max_trial": int(self.max_trial_spin.value()),
            "marker_name": marker_name,
            "output_dir": Path(self.output_dir_edit.text()).resolve(),
        }

    def on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self.output_dir_edit.text(),
        )

        if folder:
            self.output_dir_edit.setText(folder)

    def on_generate(self) -> None:
        try:
            params = self._read_params()

            self.status_label.setText("Generating marker field with C++ backend...")
            self.generate_button.setEnabled(False)
            self.save_button.setEnabled(False)
            QApplication.processEvents()

            field = MarkerField.generate_planar(
                rows=params["rows"],
                cols=params["cols"],
                patch_size=params["patch_size"],
                max_ms=params["max_ms"],
                max_trial=params["max_trial"],
                is_print=False,
            )

            field = np.asarray(field, dtype=np.uint8)

            if field.shape != (params["rows"], params["cols"]):
                raise RuntimeError(
                    f"Generated field has unexpected shape {field.shape}, "
                    f"expected {(params['rows'], params['cols'])}."
                )

            unique_values = set(np.unique(field).tolist())
            if not unique_values.issubset({0, 1}):
                raise RuntimeError(
                    f"Generated field contains invalid values: {sorted(unique_values)}"
                )

            img = render_marker_image(
                field=field,
                cell_px=params["cell_px"],
                dot_radius_rel=params["dot_radius_rel"],
            )

            self.current_field = field
            self.current_img = img

            self._update_preview()

            self.status_label.setText(
                "Marker generated successfully.\n"
                f"Field: {params['rows']} x {params['cols']} cells\n"
                f"Patch size: {params['patch_size']} x {params['patch_size']}\n"
                f"Square size: {params['square_size_cm']} cm"
            )

            self.save_button.setEnabled(True)

        except Exception as exc:
            self.current_field = None
            self.current_img = None
            self.save_button.setEnabled(False)

            QMessageBox.critical(
                self,
                "Generation failed",
                str(exc),
            )

            self.status_label.setText("Generation failed.")

        finally:
            self.generate_button.setEnabled(True)

    def _update_preview(self) -> None:
        if self.current_img is None:
            return

        pixmap = cv_bgr_to_qpixmap(self.current_img)

        scaled = pixmap.scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_preview()

    def on_save(self) -> None:
        if self.current_field is None or self.current_img is None:
            QMessageBox.warning(
                self,
                "Nothing to save",
                "Generate a marker first.",
            )
            return

        try:
            params = self._read_params()

            output_dir = params["output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)

            name = params["marker_name"]

            field_path = output_dir / f"{name}.field"
            png_path = output_dir / f"{name}.png"
            json_path = output_dir / f"{name}.json"

            save_field_file(
                path=field_path,
                field=self.current_field,
                patch_size=params["patch_size"],
            )

            ok = cv2.imwrite(str(png_path), self.current_img)
            if not ok:
                raise RuntimeError(f"Could not write PNG file: {png_path}")

            save_meta_file(
                path=json_path,
                rows=params["rows"],
                cols=params["cols"],
                patch_size=params["patch_size"],
                square_size_cm=params["square_size_cm"],
                cell_px=params["cell_px"],
                dot_radius_rel=params["dot_radius_rel"],
                marker_name=name,
            )

            self.status_label.setText(
                "Saved marker files:\n"
                f"{field_path}\n"
                f"{png_path}\n"
                f"{json_path}"
            )

            QMessageBox.information(
                self,
                "Saved",
                "Marker files saved successfully.",
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Save failed",
                str(exc),
            )


def main() -> None:
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    window = GeneratePlanarMarkerUI()
    window.resize(1250, 820)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()