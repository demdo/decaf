"""
HydraMarker planar marker generator.

Creates:
- .field : binary HydraMarker state matrix + patch shape
- .pdf   : printable marker with true physical size
- .json  : metric marker metadata for tracking, SfM alignment and 2D-3D correspondences

User-facing scale is controlled by square_size_mm.
The PDF page size is exactly the marker size in millimeters.
Print the PDF at 100% / actual size.
"""

import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

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
    QCheckBox,
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
    from reportlab.lib.colors import black, white
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
except ImportError as exc:
    raise ImportError(
        "reportlab is required for PDF export. Install it with: pip install reportlab"
    ) from exc

try:
    from .field import MarkerField
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = THIS_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from hydramarker.field import MarkerField


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = THIS_DIR / "data"

DEFAULT_PREVIEW_DPI = 600
DEFAULT_DOT_RADIUS_REL = 0.22
DEFAULT_MAX_MS = 300000.0
DEFAULT_MAX_TRIAL = 1000000


def mm_to_px(size_mm: float, dpi: int) -> int:
    return max(1, int(round(size_mm / 25.4 * float(dpi))))


def format_mm_for_name(value_mm: float) -> str:
    text = f"{value_mm:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def auto_marker_name(rows: int, cols: int, patch_size: int, square_size_mm: float) -> str:
    return f"marker_{rows}x{cols}_p{patch_size}_{format_mm_for_name(square_size_mm)}mm"


def sanitize_marker_name(name: str) -> str:
    name = name.strip()
    if not name:
        return ""

    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def render_marker_image(
    field: np.ndarray,
    cell_px: int,
    dot_radius_rel: float,
    padding_rel: float = 0.0,
) -> np.ndarray:
    """Render the marker field as a BGR image.

    padding_rel controls the checkerboard border rendered around the core field.
    A value of 0.0 means no border; 1.0 means a full cell-width border.
    The border continues the checkerboard pattern but contains no dots, giving
    the saddle detector full contrast on all four sides of every border corner.
    The physical border width is ``padding_rel * cell_px`` pixels.
    """
    field = np.asarray(field, dtype=np.uint8)
    core_rows, core_cols = field.shape

    pad_px = int(round(padding_rel * cell_px))

    # Total grid including one full padding cell on each side (needed to
    # generate the correct checker colour at the border).  We render a grid
    # that is (core + 2) cells wide/tall and then crop to pad_px pixels of
    # the outer ring.
    if pad_px > 0:
        total_rows = core_rows + 2
        total_cols = core_cols + 2
        row_offset = 1  # core starts at row index 1 in the extended grid
        col_offset = 1
    else:
        total_rows = core_rows
        total_cols = core_cols
        row_offset = 0
        col_offset = 0

    img_h = total_rows * cell_px
    img_w = total_cols * cell_px
    img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

    for row in range(total_rows):
        for col in range(total_cols):
            y0 = row * cell_px
            x0 = col * cell_px
            y1 = y0 + cell_px
            x1 = x0 + cell_px
            black_cell = ((row + col) % 2 == 0)
            img[y0:y1, x0:x1] = (0, 0, 0) if black_cell else (255, 255, 255)

    dot_radius_px = max(1, int(round(dot_radius_rel * cell_px)))

    for row in range(core_rows):
        for col in range(core_cols):
            if int(field[row, col]) != 1:
                continue

            grid_row = row + row_offset
            grid_col = col + col_offset
            cx = grid_col * cell_px + cell_px // 2
            cy = grid_row * cell_px + cell_px // 2

            black_cell = ((grid_row + grid_col) % 2 == 0)
            dot_color = (255, 255, 255) if black_cell else (0, 0, 0)

            cv2.circle(
                img,
                (cx, cy),
                dot_radius_px,
                dot_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    # Crop: keep pad_px pixels of the outer ring on each side.
    if pad_px > 0:
        crop_top    = cell_px - pad_px
        crop_left   = cell_px - pad_px
        crop_bottom = img_h - (cell_px - pad_px)
        crop_right  = img_w - (cell_px - pad_px)
        img = img[crop_top:crop_bottom, crop_left:crop_right]

    return img


def cv_bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def write_field_file(path: Path, field: np.ndarray, patch_size: int) -> None:
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


def build_meta(
    rows: int,
    cols: int,
    patch_size: int,
    square_size_mm: float,
    cell_px: int,
    dot_radius_rel: float,
    marker_name: str,
    padding_rel: float = 0.0,
    surface_model: dict | None = None,
) -> dict:
    square_size_cm = square_size_mm / 10.0
    pad_mm = padding_rel * square_size_mm
    # With padding the outermost corner ring is fully visible and reliably
    # detected, so all (rows+1)×(cols+1) corners are detectable and the
    # id_encoding origin sits at (0,0).  Without padding the outer ring is
    # unreliable, so only the inner (rows-1)×(cols-1) corners are used and
    # the origin is offset by (1,1).
    if padding_rel > 0.0:
        detectable_corner_rows = rows + 1
        detectable_corner_cols = cols + 1
        id_origin_row = 0
        id_origin_col = 0
    else:
        detectable_corner_rows = rows - 1
        detectable_corner_cols = cols - 1
        id_origin_row = 1
        id_origin_col = 1

    if detectable_corner_rows < 2 or detectable_corner_cols < 2:
        raise ValueError(
            "Need at least 3x3 cells so that the internal detectable corner grid "
            "contains at least 2x2 corners."
        )

    meta = {
        "name": marker_name,
        "marker_type": "planar",
        "rows": rows,
        "cols": cols,
        "patch_size": patch_size,
        "square_size_cm": square_size_cm,
        "square_size_mm": square_size_mm,
        "padding_rel": padding_rel,
        "padding_mm": round(pad_mm, 6),
        "has_border": pad_mm > 0.0,
        "origin": "top_left",
        "x_axis": "col_positive",
        "y_axis": "row_positive",
        "z_axis": "out_of_plane",
        "corner_rows": rows + 1,
        "corner_cols": cols + 1,
        "detectable_corner_rows": detectable_corner_rows,
        "detectable_corner_cols": detectable_corner_cols,
        "id_encoding": {
            "type": "row_major",
            "id_base": 0,
            "num_cols": detectable_corner_cols,
            "origin_row": id_origin_row,
            "origin_col": id_origin_col,
            "formula": (
                "marker_id = "
                "(origin_row + local_row) * num_cols + "
                "(origin_col + local_col)"
            ),
        },
        "alignment_reference": {
            "origin": {
                "local_row": 0,
                "local_col": 0,
            },
            "x_axis": {
                "local_row": 0,
                "local_col": 1,
            },
            "y_axis": {
                "local_row": 1,
                "local_col": 0,
            },
        },
        "cell_px": cell_px,
        "dot_radius_rel": dot_radius_rel,
        "coordinate_mapping": {
            "detectable_corner_row_col_to_xyz_mm": {
                "X": "local_col * square_size_mm",
                "Y": "local_row * square_size_mm",
                "Z": "0",
            }
        },
    }

    if surface_model:
        meta["surface_model"] = surface_model

    return meta


def write_json_file(path: Path, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_pdf_file(
    path: Path,
    field: np.ndarray,
    square_size_mm: float,
    dot_radius_rel: float,
    padding_rel: float = 0.0,
) -> None:
    """Write the marker as a PDF.

    The PDF page is sized to the visible marker including the padding border.
    padding_rel is the border width as a fraction of square_size_mm (0–1).
    """
    field = np.asarray(field, dtype=np.uint8)

    if field.ndim != 2:
        raise ValueError("field must be a 2D array")

    core_rows, core_cols = field.shape
    pad_mm = padding_rel * square_size_mm

    # Page size = core + 2 * partial padding strip.
    page_width_mm  = core_cols * square_size_mm + 2.0 * pad_mm
    page_height_mm = core_rows * square_size_mm + 2.0 * pad_mm

    cell_pt    = square_size_mm * mm
    pad_pt     = pad_mm * mm
    page_w_pt  = page_width_mm  * mm
    page_h_pt  = page_height_mm * mm
    dot_radius_pt = dot_radius_rel * cell_pt

    # Extended grid dimensions (core + 2 padding cells on each side).
    # We render the full extended grid but only the strip of pad_pt is visible
    # because the page is cropped to page_w_pt x page_h_pt.
    # In reportlab the origin is bottom-left; we offset by pad_pt so that
    # cell (0,0) of the core starts at (pad_pt, page_h_pt - pad_pt - cell_pt).
    total_rows = core_rows + 2
    total_cols = core_cols + 2

    c = canvas.Canvas(str(path), pagesize=(page_w_pt, page_h_pt), pageCompression=0)
    c.setTitle(path.stem)

    for row in range(total_rows):
        for col in range(total_cols):
            # Position relative to page: shift by (pad_pt - cell_pt) so that
            # the extended grid is centred and only pad_pt of the outer ring
            # is visible within the page bounds.
            x0 = (col - 1) * cell_pt + pad_pt
            y0 = page_h_pt - ((row - 1) * cell_pt + pad_pt) - cell_pt
            black_cell = ((row + col) % 2 == 0)
            c.setFillColor(black if black_cell else white)
            c.rect(x0, y0, cell_pt, cell_pt, stroke=0, fill=1)

    for row in range(core_rows):
        for col in range(core_cols):
            if int(field[row, col]) != 1:
                continue

            grid_row = row + 1
            grid_col = col + 1
            cx = (grid_col - 1) * cell_pt + pad_pt + 0.5 * cell_pt
            cy = page_h_pt - ((grid_row - 1) * cell_pt + pad_pt) - 0.5 * cell_pt

            black_cell = ((grid_row + grid_col) % 2 == 0)
            c.setFillColor(white if black_cell else black)
            c.circle(cx, cy, dot_radius_pt, stroke=0, fill=1)

    c.showPage()
    c.save()


def write_temp_then_replace(writer_func, final_path: Path, suffix: str, *args, **kwargs) -> None:
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    with NamedTemporaryFile(
        delete=False,
        dir=str(final_path.parent),
        prefix=f".{final_path.stem}_",
        suffix=suffix,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        writer_func(tmp_path, *args, **kwargs)
        os.replace(tmp_path, final_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def estimate_min_patch_warning(rows: int, cols: int, patch_size: int) -> str | None:
    num_windows = (rows - patch_size + 1) * (cols - patch_size + 1)
    raw_patterns = 2 ** (patch_size * patch_size)

    if patch_size <= 2:
        return (
            "Patch size k is very small. A 2x2 binary patch has too few possible "
            "patterns for a useful marker field. Use k >= 3, preferably k = 4."
        )

    if raw_patterns < 4 * num_windows:
        return (
            "Patch size may be too small for this marker size. Generation may fail "
            "or take a long time. Consider increasing k."
        )

    return None


class GeneratePlanarMarkerUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("HydraMarker - Planar Marker Generator")

        self.current_field: np.ndarray | None = None
        self.current_img: np.ndarray | None = None
        self.current_params: dict | None = None

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(3, 500)
        self.rows_spin.setValue(12)

        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(3, 500)
        self.cols_spin.setValue(20)

        self.patch_size_spin = QSpinBox()
        self.patch_size_spin.setRange(2, 20)
        self.patch_size_spin.setValue(4)

        self.square_size_mm_spin = QDoubleSpinBox()
        self.square_size_mm_spin.setRange(0.1, 1000.0)
        self.square_size_mm_spin.setDecimals(3)
        self.square_size_mm_spin.setSingleStep(0.5)
        self.square_size_mm_spin.setValue(5.0)
        self.square_size_mm_spin.setSuffix(" mm")

        self.padding_rel_spin = QDoubleSpinBox()
        self.padding_rel_spin.setRange(0.0, 1.0)
        self.padding_rel_spin.setDecimals(2)
        self.padding_rel_spin.setSingleStep(0.05)
        self.padding_rel_spin.setValue(0.3)
        self.padding_rel_spin.setToolTip(
            "Border width as a fraction of the cell size (0 = no border, 1 = full cell).\n"
            "The border continues the checkerboard pattern without dots so that\n"
            "edge corners have full saddle contrast on all four sides."
        )

        self.surface_cylinder_checkbox = QCheckBox("Cylinder surface")
        self.surface_cylinder_checkbox.setChecked(False)
        self.surface_cylinder_checkbox.setToolTip(
            "Store cylinder surface metadata for markers mounted on a cylinder."
        )

        self.marker_name_edit = QLineEdit()
        self.marker_name_edit.setPlaceholderText("Leave empty for automatic name")

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(str(DEFAULT_DATA_DIR))

        self.browse_button = QPushButton("Browse")
        self.generate_button = QPushButton("Generate Preview")
        self.save_button = QPushButton("Save .field / .pdf / .json")
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
        self._update_auto_name_placeholder()

    def _build_layout(self) -> None:
        form = QFormLayout()
        form.addRow("Rows / Cells Y", self.rows_spin)
        form.addRow("Cols / Cells X", self.cols_spin)
        form.addRow("Patch size k", self.patch_size_spin)
        form.addRow("Square size [mm]", self.square_size_mm_spin)
        form.addRow("Border padding [0–1]", self.padding_rel_spin)
        form.addRow("Marker name", self.marker_name_edit)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit)
        output_row.addWidget(self.browse_button)
        form.addRow("Output folder", output_row)

        params_group = QGroupBox("Marker Parameters")
        params_group.setLayout(form)

        surface_layout = QVBoxLayout()
        surface_layout.addWidget(self.surface_cylinder_checkbox)

        surface_group = QGroupBox("Surface Model")
        surface_group.setLayout(surface_layout)

        button_row = QHBoxLayout()
        button_row.addWidget(self.generate_button)
        button_row.addWidget(self.save_button)

        left_layout = QVBoxLayout()
        left_layout.addWidget(params_group)
        left_layout.addWidget(surface_group)
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

        self.rows_spin.valueChanged.connect(self.on_params_changed)
        self.cols_spin.valueChanged.connect(self.on_params_changed)
        self.patch_size_spin.valueChanged.connect(self.on_params_changed)
        self.square_size_mm_spin.valueChanged.connect(self.on_params_changed)
        self.padding_rel_spin.valueChanged.connect(self.on_params_changed)
        self.surface_cylinder_checkbox.stateChanged.connect(self.on_params_changed)
        self.marker_name_edit.textChanged.connect(self._update_auto_name_placeholder)

    def _update_auto_name_placeholder(self) -> None:
        rows = int(self.rows_spin.value())
        cols = int(self.cols_spin.value())
        patch_size = int(self.patch_size_spin.value())
        square_size_mm = float(self.square_size_mm_spin.value())
        name = auto_marker_name(rows, cols, patch_size, square_size_mm)
        self.marker_name_edit.setPlaceholderText(name)

    def on_params_changed(self) -> None:
        self._update_auto_name_placeholder()
        self.save_button.setEnabled(False)
        self.current_field = None
        self.current_img = None
        self.current_params = None
        self.preview_label.setText("Parameters changed. Generate a new marker preview.")
        self.preview_label.setPixmap(QPixmap())

    def _read_params(self) -> dict:
        rows = int(self.rows_spin.value())
        cols = int(self.cols_spin.value())
        patch_size = int(self.patch_size_spin.value())
        square_size_mm = float(self.square_size_mm_spin.value())

        if patch_size > rows or patch_size > cols:
            raise ValueError("Patch size must be <= rows and <= cols.")

        if rows < 3 or cols < 3:
            raise ValueError("Rows and cols must be at least 3.")

        if square_size_mm <= 0.0:
            raise ValueError("Square size must be positive.")

        marker_name = sanitize_marker_name(self.marker_name_edit.text())
        if not marker_name:
            marker_name = auto_marker_name(rows, cols, patch_size, square_size_mm)

        padding_rel = float(self.padding_rel_spin.value())
        surface_model = None

        if self.surface_cylinder_checkbox.isChecked():
            surface_model = {
                "type": "cylinder",
                "axis": "row",
                "regularize_columns_z": True,
                "note": "Marker is mounted on a cylinder; SfM regularizes Z per column only.",
            }

        pad_mm = padding_rel * square_size_mm
        cell_px = mm_to_px(square_size_mm, DEFAULT_PREVIEW_DPI)
        width_mm  = cols * square_size_mm + 2.0 * pad_mm
        height_mm = rows * square_size_mm + 2.0 * pad_mm

        return {
            "rows": rows,
            "cols": cols,
            "patch_size": patch_size,
            "square_size_mm": square_size_mm,
            "padding_rel": padding_rel,
            "cell_px": cell_px,
            "dot_radius_rel": DEFAULT_DOT_RADIUS_REL,
            "preview_dpi": DEFAULT_PREVIEW_DPI,
            "width_mm": width_mm,
            "height_mm": height_mm,
            "marker_name": marker_name,
            "surface_model": surface_model,
            "output_dir": Path(self.output_dir_edit.text()).resolve(),
            "max_ms": DEFAULT_MAX_MS,
            "max_trial": DEFAULT_MAX_TRIAL,
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

            warning = estimate_min_patch_warning(
                rows=params["rows"],
                cols=params["cols"],
                patch_size=params["patch_size"],
            )

            status = "Generating marker field with C++ backend..."
            if warning:
                status += "\n\nWarning: " + warning
            self.status_label.setText(status)

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
                    "Generated field contains unresolved or invalid values: "
                    f"{sorted(unique_values)}"
                )

            img = render_marker_image(
                field=field,
                cell_px=params["cell_px"],
                dot_radius_rel=params["dot_radius_rel"],
                padding_rel=params["padding_rel"],
            )

            self.current_field = field
            self.current_img = img
            self.current_params = params

            self._update_preview()

            self.status_label.setText(
                "Marker generated successfully.\n"
                f"Field: {params['rows']} x {params['cols']} cells\n"
                f"Patch size: {params['patch_size']} x {params['patch_size']}\n"
                f"Square size: {params['square_size_mm']:.3f} mm\n"
                f"Border padding: {params['padding_rel']:.2f} × cell = "
                f"{params['padding_rel'] * params['square_size_mm']:.3f} mm\n"
                f"Physical marker size (incl. border): {params['width_mm']:.3f} x "
                f"{params['height_mm']:.3f} mm\n"
                f"Preview raster: {params['cell_px']} px/cell at "
                f"{params['preview_dpi']} dpi\n"
                f"Detectable corner grid: {params['rows'] - 1} x {params['cols'] - 1}\n"
                "Export will save .field, .pdf, and .json. Print the PDF at 100% / actual size."
            )

            self.save_button.setEnabled(True)

        except Exception as exc:
            self.current_field = None
            self.current_img = None
            self.current_params = None
            self.save_button.setEnabled(False)

            QMessageBox.critical(self, "Generation failed", str(exc))
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
        if self.current_field is None or self.current_params is None:
            QMessageBox.warning(self, "Nothing to save", "Generate a marker first.")
            return

        try:
            params = self._read_params()

            if params != self.current_params:
                raise RuntimeError(
                    "Parameters changed after generation. Generate a new preview before saving."
                )

            output_dir = params["output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)

            name = params["marker_name"]
            field_path = output_dir / f"{name}.field"
            pdf_path = output_dir / f"{name}.pdf"
            json_path = output_dir / f"{name}.json"

            meta = build_meta(
                rows=params["rows"],
                cols=params["cols"],
                patch_size=params["patch_size"],
                square_size_mm=params["square_size_mm"],
                cell_px=params["cell_px"],
                dot_radius_rel=params["dot_radius_rel"],
                marker_name=name,
                padding_rel=params["padding_rel"],
                surface_model=params["surface_model"],
            )

            # Write to temporary files first. This prevents partial output if one export fails.
            write_temp_then_replace(
                write_field_file,
                field_path,
                ".field",
                self.current_field,
                params["patch_size"],
            )
            write_temp_then_replace(
                write_pdf_file,
                pdf_path,
                ".pdf",
                self.current_field,
                params["square_size_mm"],
                params["dot_radius_rel"],
                params["padding_rel"],
            )
            write_temp_then_replace(write_json_file, json_path, ".json", meta)

            self.status_label.setText(
                "Saved marker files:\n"
                f"{field_path}\n"
                f"{pdf_path}\n"
                f"{json_path}\n\n"
                "Print the PDF at 100% / actual size."
            )

            QMessageBox.information(
                self,
                "Saved",
                "Marker files saved successfully.\n\nPrint the PDF at 100% / actual size.",
            )

        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))


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
