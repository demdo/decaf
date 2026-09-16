# overlay/gui/pages/page_tool_tracking.py

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from overlay.gui.state import SessionState
from overlay.gui.pages.templates.templ_static_image import StaticImagePage

from tracking.hydramarker.tool_assets import load_tool_assets


def _hide_widget(w: QWidget | None) -> None:
    if w is None:
        return
    try:
        w.setVisible(False)
    except Exception:
        pass
    try:
        w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    except Exception:
        pass
    try:
        w.setMinimumSize(0, 0)
    except Exception:
        pass
    try:
        w.setMaximumSize(0, 0)
    except Exception:
        pass


def _hide_if_exists(obj, attr_name: str) -> None:
    _hide_widget(getattr(obj, attr_name, None))


def _default_ir_calibration_path() -> Path:
    """The IR rig calibration run_tracker also uses by default."""
    import tracking.hydramarker as hm

    return Path(hm.__file__).resolve().parent / "data" / "realsense" / "realsense_ir_calibration.npz"


def _placeholder_image(lines: list[str], size_wh: tuple[int, int]) -> np.ndarray:
    """Grey box in the viewport size with centered status text.

    Placeholder until the tool visualization lands in a later step; after
    loading the box simply shows the loaded state as text.
    """
    w, h = size_wh
    img = np.full((h, w, 3), 64, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 1
    color = (210, 210, 210)
    line_gap = 34

    y0 = h // 2 - (len(lines) - 1) * line_gap // 2
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        org = ((w - tw) // 2, y0 + i * line_gap + th // 2)
        cv2.putText(img, line, org, font, scale, color, thickness, cv2.LINE_AA)

    return img


class ToolTrackingPage(StaticImagePage):
    """
    Mode B — Load tool tracking files

    Guided, sequential selection of the tool file triple (each button is
    enabled once the previous file is chosen):

        1. .field                  detector state matrix
        2. marker .json            metric SfM geometry incl. drill axis
        3. pivot_calibration .npz  tool tip in the marker frame

    Loading and validation go through tracking.hydramarker.tool_assets
    (the single canonical loader). The viewport shows a grey status box
    while files are still missing; once the tool is loaded it is replaced
    by an INTERACTIVE matplotlib 3D figure (rotatable, embedded QtAgg
    canvas in the same slot and size as the static image).
    """

    # 4:3 like the reference pivot figure; the template scales to fit.
    VIEWPORT_SIZE = (640, 480)
    # No instructions on this page: collapse the template's fixed-height
    # instructions slot entirely. Otherwise the empty 120 px slot below the
    # canvas can push the page beyond the available height on smaller
    # screens — which shoves the floating Back/Next bar out of view.
    INSTR_SLOT_H = 0

    def __init__(self, state: SessionState, on_complete_changed, parent=None):
        super().__init__(parent)

        self.state = state
        self.on_complete_changed = on_complete_changed

        # Embedded matplotlib canvas (created on first successful load).
        self._canvas = None

        # Pending selection (paths chosen but maybe not yet complete/valid).
        self._field_path: Path | None = None
        self._marker_json_path: Path | None = None
        self._pivot_npz_path: Path | None = None

        # ======================================================
        # RIGHT: controls — one button per file, used in order
        # ======================================================
        self.btn_field = QPushButton("1. Select .field")
        self.btn_field.clicked.connect(lambda: self._pick_single("field"))
        self.btn_field.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.controls_content.addWidget(self.btn_field)

        self.btn_marker_json = QPushButton("2. Select marker .json")
        self.btn_marker_json.clicked.connect(lambda: self._pick_single("marker_json"))
        self.btn_marker_json.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.controls_content.addWidget(self.btn_marker_json)

        self.btn_pivot = QPushButton("3. Select pivot calibration .npz")
        self.btn_pivot.clicked.connect(lambda: self._pick_single("pivot_npz"))
        self.btn_pivot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.controls_content.addWidget(self.btn_pivot)

        self._ir_default = _default_ir_calibration_path()
        self.chk_ir = QCheckBox("IR refinement (stereo IR)")
        if self._ir_default.is_file():
            self.chk_ir.setChecked(True)
            self.chk_ir.setToolTip(str(self._ir_default))
        else:
            self.chk_ir.setChecked(False)
            self.chk_ir.setEnabled(False)
            self.chk_ir.setToolTip(
                f"IR calibration not found: {self._ir_default}"
            )
        self.chk_ir.toggled.connect(self._on_ir_toggled)
        self.controls_content.addWidget(self.chk_ir)

        self.btn_clear = QPushButton("Clear tool")
        self.btn_clear.setObjectName("SecondaryBtn")
        self.btn_clear.clicked.connect(self.clear_tool)
        self.controls_content.addWidget(self.btn_clear)

        # Word-wrapped notes area (loader warnings); the token-grid layout
        # of set_stats_rows is unsuitable for prose.
        self.lbl_notes = QLabel("")
        self.lbl_notes.setWordWrap(True)
        self.lbl_notes.setStyleSheet("color: #b45309;")  # amber
        self.controls_content.addWidget(self.lbl_notes)

        # Instructions panel stays hidden — the viewport status box carries
        # the guidance text (same treatment as the intrinsics pages). The
        # slot itself is collapsed too (INSTR_SLOT_H = 0 above).
        for name in (
            "instructions_box", "instructions_group", "instructions_panel",
            "instructions_frame", "instructions_label", "instructions_slot",
        ):
            _hide_if_exists(self, name)

        self.refresh()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def is_complete(self) -> bool:
        return self.state.has_tool

    def _viewport_lines(self) -> list[str]:
        tool = self.state.tool_assets
        if tool is not None:
            return [
                f"Tool loaded: {tool.name}",
                f"({tool.marker_type}, {tool.pivot_method or 'pivot'})",
            ]
        if self._field_path is None:
            return ["No tool loaded", "Next: 1. Select .field"]
        if self._marker_json_path is None:
            return ["No tool loaded", "Next: 2. Select marker .json"]
        if self._pivot_npz_path is None:
            return ["No tool loaded", "Next: 3. Select pivot calibration .npz"]
        return ["No tool loaded"]

    def refresh(self) -> None:
        tool = self.state.tool_assets

        # Sequential flow: each button unlocks once the previous file is set.
        self.btn_marker_json.setEnabled(self._field_path is not None)
        self.btn_pivot.setEnabled(self._marker_json_path is not None)

        if tool is not None:
            tip = tool.p_tip_mm
            axis = tool.axis_dir
            # Tool identity is the data folder name (e.g. "PRO6202M") —
            # the marker/json names repeat across tools.
            tool_name = tool.marker_json_path.resolve().parent.name
            rows = [
                ("tool", tool_name),
                ("p_tip [mm]", f"({tip[0]:.2f}, {tip[1]:.2f}, {tip[2]:.2f})"),
                ("axis_dir", f"({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})"),
            ]
            self.lbl_notes.setText("\n".join(tool.warnings))
        else:
            rows = [
                ("tool", "-"),
                ("p_tip [mm]", "-"),
                ("axis_dir", "-"),
            ]
            self.lbl_notes.setText("")

        self.set_stats_rows(rows)

        shown = False
        if tool is not None:
            # Hide the static image BEFORE inserting/showing the canvas:
            # both are fixed-size — if they coexist in the layout for even
            # one pass, the page minimum height doubles and shoves the
            # floating nav bar out of a fullscreen window.
            self.image_label.hide()
            shown = self._show_tool_figure(tool)
        if not shown:
            if self._canvas is not None:
                self._canvas.hide()
            self.image_label.show()
            self.set_image(_placeholder_image(self._viewport_lines(), self.VIEWPORT_SIZE))

    # ------------------------------------------------------------------
    # Interactive 3D figure (replaces the grey box once a tool is loaded)
    # ------------------------------------------------------------------

    def _show_tool_figure(self, tool) -> bool:
        """Build/refresh the embedded 3D figure. Returns False when
        matplotlib is unavailable so the caller keeps the grey box."""
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception:
            return False

        if self._canvas is None:
            fig = Figure(constrained_layout=True)
            self._canvas = FigureCanvasQTAgg(fig)
            # Same left-column slot as the static image (index 1, right
            # after image_label); sized to the template viewport.
            self._left_col.insertWidget(
                1, self._canvas, stretch=0,
                alignment=Qt.AlignLeft | Qt.AlignTop,
            )
            # Widgets created after the floating nav bar stack above it —
            # keep Back/Next on top.
            self._nav_bar.raise_()

        fig = self._canvas.figure
        fig.clear()
        ax = fig.add_subplot(111, projection="3d")
        self._plot_tool_3d(ax, fig, tool)

        self._sync_canvas_size()
        self._canvas.show()
        self._canvas.draw_idle()
        return True

    @staticmethod
    def _plot_tool_3d(ax, fig, tool) -> None:
        """SfM marker corners + origin + drill axis + tip, marker frame [mm].

        Same content as the pivot-calibration reference figure, but without
        title / free-tip elements, and with the z axis inverted so the open
        cylinder face points down.
        """
        pts = [np.zeros(3), tool.p_tip_mm]

        corners = tool.corners_xyz_mm
        if corners is not None and len(corners):
            sc = ax.scatter(
                corners[:, 0], corners[:, 1], corners[:, 2],
                c=np.arange(len(corners)), cmap="viridis", s=25,
                depthshade=True, label="Marker corners (SfM model)",
            )
            fig.colorbar(sc, ax=ax, label="Corner ID", shrink=0.65, pad=0.10)
            pts.append(corners)

        ax.scatter(
            [0.0], [0.0], [0.0],
            marker="*", color="black", s=180, label="Marker origin (0,0,0)",
        )

        # Drill axis through all projected points, slightly extended.
        a0 = tool.axis_point_mm
        d = tool.axis_dir
        all_pts = np.vstack([np.atleast_2d(p) for p in pts])
        t = (all_pts - a0) @ d
        t0, t1 = float(t.min()), float(t.max())
        pad = 0.08 * max(t1 - t0, 1.0)
        line = np.stack([a0 + (t0 - pad) * d, a0 + (t1 + pad) * d])
        ax.plot(
            line[:, 0], line[:, 1], line[:, 2],
            color="navy", linewidth=2.5, label="Drill axis",
        )

        tip = tool.p_tip_mm
        ax.scatter(
            [tip[0]], [tip[1]], [tip[2]],
            marker="*", color="crimson", s=220, label="Drill tip",
        )

        # Marker coordinate system at the origin (like the reference figure).
        axis_len = 20.0
        for vec, color, label in (
            ((axis_len, 0, 0), "red", "+x"),
            ((0, axis_len, 0), "green", "+y"),
            ((0, 0, axis_len), "blue", "+z"),
        ):
            ax.quiver(0, 0, 0, *vec, color=color, linewidth=2.0,
                      arrow_length_ratio=0.25)
            ax.text(vec[0] * 1.25, vec[1] * 1.25, vec[2] * 1.25, label,
                    color=color, fontsize=9)

        # Equal aspect over everything drawn.
        all_pts = np.vstack([all_pts, line])
        center = 0.5 * (all_pts.min(axis=0) + all_pts.max(axis=0))
        half = 0.5 * float((all_pts.max(axis=0) - all_pts.min(axis=0)).max()) * 1.10
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1))

        # Open cylinder face down.
        ax.invert_zaxis()

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        ax.legend(loc="upper left", fontsize=8)

    def _sync_canvas_size(self) -> None:
        # getattr: the base template calls _update_layout_geometry from its
        # own __init__, before this class has set up _canvas.
        if getattr(self, "_canvas", None) is None:
            return
        if self._last_viewport_px is not None:
            w, h = self._last_viewport_px
        else:
            w, h = self.VIEWPORT_SIZE
        self._canvas.setFixedSize(w, h)

    def _update_layout_geometry(self) -> None:
        super()._update_layout_geometry()
        self._sync_canvas_size()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    _PICK_SPECS = {
        "field": ("Select HydraMarker .field", "HydraMarker field (*.field)"),
        "marker_json": ("Select marker .json", "Marker JSON (*.json)"),
        "pivot_npz": ("Select pivot calibration .npz", "NPZ files (*.npz)"),
    }

    def _pick_single(self, slot: str) -> None:
        title, name_filter = self._PICK_SPECS[slot]
        dlg = QFileDialog(self.window(), title)
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter(f"{name_filter};;All Files (*)")
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        if dlg.exec() != QFileDialog.Accepted:
            return
        files = dlg.selectedFiles()
        if not files:
            return
        path = Path(files[0])

        if slot == "field":
            self._field_path = path
        elif slot == "marker_json":
            self._marker_json_path = path
        else:
            self._pivot_npz_path = path

        self._try_load()

    def _try_load(self) -> None:
        if not (self._field_path and self._marker_json_path and self._pivot_npz_path):
            self.state.tool_assets = None
            self.refresh()
            self.on_complete_changed()
            return

        try:
            tool = load_tool_assets(
                self._field_path, self._marker_json_path, self._pivot_npz_path
            )
        except Exception as e:
            self.state.tool_assets = None
            self.refresh()
            self.on_complete_changed()
            QMessageBox.critical(self, "Failed to load tool", str(e))
            return

        self.state.tool_assets = tool
        self._on_ir_toggled(self.chk_ir.isChecked())
        self.refresh()
        self.on_complete_changed()

    def _on_ir_toggled(self, checked: bool) -> None:
        if checked and self._ir_default.is_file():
            self.state.ir_calibration_path = str(self._ir_default)
        else:
            self.state.ir_calibration_path = None
        self.refresh()

    def clear_tool(self) -> None:
        self._field_path = None
        self._marker_json_path = None
        self._pivot_npz_path = None
        self.state.tool_assets = None
        self.refresh()
        self.on_complete_changed()
