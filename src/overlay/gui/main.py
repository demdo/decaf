# overlay/gui/main.py
#
# Wizard runner (Mode A):
#   1) X-ray intrinsics
#   2) Camera intrinsics
#   3) Plane Fitting
#   4) X-ray Marker Selection
#   5) Camera -> X-ray Calibration (PnP)
#
# Uses the per-page template navigation buttons (btn_back / btn_next).
# Sidebar is display-only (wizard gating), no click navigation.
#
# Two modes, chosen in a dialog before the window opens (gui/evaluation.py):
#   Normal      - nothing is written to disk.
#   Evaluation  - every capture is saved into one session folder under a
#                 label from the bar at the top; the calibration files can
#                 be loaded automatically and "New board capture" returns
#                 to step 3 without restarting the GUI.

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
    QPushButton,
)

from overlay.gui.state import SessionState
from overlay.gui.evaluation import (
    EvalBar,
    EvaluationSession,
    StartupChoice,
    StartupDialog,
)
from overlay.gui.pages.page_xray_intrinsics import XrayIntrinsicsPage
from overlay.gui.pages.page_camera_intrinsics import CameraIntrinsicsPage
from overlay.gui.pages.page_plane_fitting import PlaneFittingPage
from overlay.gui.pages.page_xray_marker_selection import XrayMarkerSelectionPage
from overlay.gui.pages.page_camera_to_xray_calibration import CameraToXrayCalibrationPage
from overlay.gui.pages.page_pointer_tool_depth import PointerToolDepthPage
from overlay.gui.pages.page_overlay_preview import OverlayPreviewPage
from overlay.gui.pages.page_tool_tracking import ToolTrackingPage
from overlay.gui.pages.page_overlay_live import OverlayLivePage


QSS = """
QMainWindow { background: #f3f4f6; }
#Sidebar { background: #e9ecef; }
#StepList { background: transparent; border: none; outline: none; }
#StepList::item { padding: 10px 12px; margin: 4px 8px; border-radius: 8px; color: #6c757d; }
#StepList::item:selected { background: #2f80ed; color: white; }
#ContentCard { background: white; border-radius: 14px; }
#ContentTitle { font-size: 18px; font-weight: 600; color: #212529; }
QPushButton { background: #2f80ed; color: white; border: none; padding: 10px 14px; border-radius: 10px; font-weight: 600; }
QPushButton:hover { background: #256ad1; }
QPushButton:disabled { background: #a6c8ff; color: #f8f9fa; }
#NavBar { background: transparent; }
#SecondaryBtn { background: #dee2e6; color: #212529; }
#SecondaryBtn:hover { background: #ced4da; }
#SecondaryBtn:disabled { background: #e9ecef; color: #adb5bd; }
"""


class MainWindow(QMainWindow):
    def __init__(self, choice: StartupChoice | None = None):
        super().__init__()

        self.choice = choice if choice is not None else StartupChoice()

        self.state = SessionState()
        self.state.steps_per_edge = 10

        self.eval_session: EvaluationSession | None = None
        if self.choice.evaluation:
            self.eval_session = EvaluationSession(
                self.choice.eval_root, self.choice.session_name, self.choice.notes
            )
            self.state.eval = self.eval_session

        title = "DeCAF Calibration — Mode A (Wizard)"
        if self.eval_session is not None:
            title += f"  [EVALUATION: {self.eval_session.name}]"
        self.setWindowTitle(title)
        self.resize(1200, 700)

        # ----------------------------
        # Page indices / workflow modes
        # ----------------------------
        self.IDX_XRAY = 0
        self.IDX_CAM_CALIB = 1
        self.IDX_PLANE = 2
        self.IDX_MARKERS = 3
        self.IDX_CAM2XRAY = 4
        self.IDX_PTR_DEPTH = 5
        self.IDX_OVERLAY_PREVIEW = 6
        self.IDX_TOOL_TRACKING = 7
        self.IDX_OVERLAY_LIVE = 8

        self.mode_a_indices = [
            self.IDX_XRAY,
            self.IDX_CAM_CALIB,
            self.IDX_PLANE,
            self.IDX_MARKERS,
            self.IDX_CAM2XRAY,
        ]

        self.mode_b_indices = [
            self.IDX_PTR_DEPTH,
            self.IDX_OVERLAY_PREVIEW,
            self.IDX_TOOL_TRACKING,
            self.IDX_OVERLAY_LIVE,
        ]

        self.current_step = self.IDX_XRAY
        self.mode_b_active = False

        self._normal_content_margins = (16, 16, 16, 16)
        self._normal_root_margins = (16, 16, 16, 16)

        root = QWidget()
        self.root_layout = QHBoxLayout(root)
        self.root_layout.setContentsMargins(*self._normal_root_margins)
        self.root_layout.setSpacing(16)

        # ---------- Sidebar ----------
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(260)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(12)

        logo_label = QLabel("DeCAF")
        logo_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #212529; margin-left: 8px;")
        sidebar_layout.addWidget(logo_label)

        steps_title = QLabel("Calibration (Mode A)")
        steps_title.setStyleSheet("color: #495057; font-weight: 600; margin-left: 8px;")
        sidebar_layout.addWidget(steps_title)

        self.step_list = QListWidget()
        self.step_list.setObjectName("StepList")
        self.step_list.setFocusPolicy(Qt.NoFocus)
        self.step_list.setDisabled(True)
        sidebar_layout.addWidget(self.step_list, stretch=1)

        self.item_xray = QListWidgetItem("1) Load X-ray intrinsics")
        self.item_rgb = QListWidgetItem("2) Load camera intrinsics")
        self.item_plane = QListWidgetItem("3) Plane fitting")
        self.item_markers = QListWidgetItem("4) X-ray marker selection")
        self.item_cam2xray = QListWidgetItem("5) Camera → X-ray (PnP)")

        for it in (self.item_xray, self.item_rgb, self.item_plane, self.item_markers, self.item_cam2xray):
            self.step_list.addItem(it)

        runtime_title = QLabel("Runtime (Mode B)")
        runtime_title.setStyleSheet("color: #495057; font-weight: 600; margin-left: 8px;")
        sidebar_layout.addWidget(runtime_title)

        self.runtime_list = QListWidget()
        self.runtime_list.setObjectName("StepList")
        self.runtime_list.setFocusPolicy(Qt.NoFocus)
        self.runtime_list.setDisabled(True)
        sidebar_layout.addWidget(self.runtime_list)

        self.item_ptr_depth = QListWidgetItem("Pointer Tool Depth")
        self.runtime_list.addItem(self.item_ptr_depth)

        self.item_overlay_preview = QListWidgetItem("Overlay Preview")
        self.runtime_list.addItem(self.item_overlay_preview)

        self.item_tool_tracking = QListWidgetItem("Tool Tracking")
        self.runtime_list.addItem(self.item_tool_tracking)

        self.item_overlay_live = QListWidgetItem("Overlay Live")
        self.runtime_list.addItem(self.item_overlay_live)

        # The QListWidget default size hint is a fixed 256x192 regardless of
        # content — with four entries that clips "Overlay Live" and grows
        # scrollbars. Scrollbars off here; the height is set after the QSS
        # is applied (see below), because the item padding comes from it.
        self.runtime_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.runtime_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.step_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        settings_btn = QPushButton("Settings")
        settings_btn.setEnabled(False)
        sidebar_layout.addWidget(settings_btn)

        # ---------- Content ----------
        self.content_card = QFrame()
        self.content_card.setObjectName("ContentCard")
        self.content_layout = QVBoxLayout(self.content_card)
        self.content_layout.setContentsMargins(*self._normal_content_margins)
        self.content_layout.setSpacing(12)

        self.eval_bar: EvalBar | None = None
        if self.eval_session is not None:
            self.eval_bar = EvalBar(self.eval_session)
            self.eval_bar.btn_new_capture.clicked.connect(self.new_board_capture)
            self.content_layout.addWidget(self.eval_bar)

        self.lbl_step_title = QLabel("")
        self.lbl_step_title.setObjectName("ContentTitle")
        self.content_layout.addWidget(self.lbl_step_title)

        self.pages = QStackedWidget()
        self.content_layout.addWidget(self.pages, stretch=1)

        # Pages
        self.page_xray = XrayIntrinsicsPage(self.state, self.on_complete_changed)
        self.page_cam_calib = CameraIntrinsicsPage(self.state, self.on_complete_changed)
        self.page_plane = PlaneFittingPage(self.state, self.on_complete_changed)
        self.page_markers = XrayMarkerSelectionPage(self.state, self.on_complete_changed)
        self.page_cam2xray = CameraToXrayCalibrationPage(self.state, self.on_complete_changed)
        self.page_ptr_depth = PointerToolDepthPage(self.state, self.on_complete_changed)
        self.page_overlay_preview = OverlayPreviewPage(self.state, self.on_complete_changed)
        self.page_tool_tracking = ToolTrackingPage(self.state, self.on_complete_changed)
        self.page_overlay_live = OverlayLivePage(self.state, self.on_complete_changed)

        self.pages.addWidget(self.page_xray)
        self.pages.addWidget(self.page_cam_calib)
        self.pages.addWidget(self.page_plane)
        self.pages.addWidget(self.page_markers)
        self.pages.addWidget(self.page_cam2xray)
        self.pages.addWidget(self.page_ptr_depth)
        self.pages.addWidget(self.page_overlay_preview)
        self.pages.addWidget(self.page_tool_tracking)
        self.pages.addWidget(self.page_overlay_live)

        self._wire_nav(self.page_xray, idx=self.IDX_XRAY)
        self._wire_nav(self.page_cam_calib, idx=self.IDX_CAM_CALIB)
        self._wire_nav(self.page_plane, idx=self.IDX_PLANE)
        self._wire_nav(self.page_markers, idx=self.IDX_MARKERS)
        self._wire_nav(self.page_cam2xray, idx=self.IDX_CAM2XRAY)
        self._wire_nav(self.page_ptr_depth, idx=self.IDX_PTR_DEPTH)
        self._wire_nav(self.page_overlay_preview, idx=self.IDX_OVERLAY_PREVIEW)
        self._wire_nav(self.page_tool_tracking, idx=self.IDX_TOOL_TRACKING)
        self._wire_nav(self.page_overlay_live, idx=self.IDX_OVERLAY_LIVE)

        self.root_layout.addWidget(self.sidebar)
        self.root_layout.addWidget(self.content_card, stretch=1)
        self.setCentralWidget(root)

        self.setStyleSheet(QSS)

        # Size the runtime list to its four items. Must happen after the QSS
        # is applied: the 10px item padding comes from the stylesheet and
        # only then shows up in sizeHintForRow.
        rows_h = sum(
            self.runtime_list.sizeHintForRow(i)
            for i in range(self.runtime_list.count())
        )
        # + per-item QSS margin (4px top/bottom) + frame + slack
        rows_h += self.runtime_list.count() * 8 + 2 * self.runtime_list.frameWidth() + 4
        self.runtime_list.setFixedHeight(rows_h)

        self.pages.setCurrentIndex(self.IDX_XRAY)
        self._call_page_enter(self.IDX_XRAY)
        self.update_ui()

        if self.eval_session is not None and self.choice.autoload_calibration:
            self._autoload_calibration()

    # ------------------------------------------------------------------
    # Evaluation mode helpers
    # ------------------------------------------------------------------

    def _autoload_calibration(self) -> None:
        """Steps 1 and 2 from the files chosen in the startup dialog; the
        wizard then starts on step 3 (plane fitting)."""
        ok_x = self.page_xray.load_from_path(str(self.choice.xray_calib_path))
        ok_c = self.page_cam_calib.load_from_path(str(self.choice.camera_calib_path))
        if ok_x:
            self.eval_session.record_calibration_file("K_xray", self.choice.xray_calib_path)
        if ok_c:
            self.eval_session.record_calibration_file("K_rgb", self.choice.camera_calib_path)
        if ok_x and ok_c:
            self._switch_to_page(self.IDX_PLANE)

    def new_board_capture(self) -> None:
        """Evaluation mode: next board placement. Keeps the intrinsics,
        clears everything the wizard derives from one board capture and
        returns to step 3, also from Mode B."""
        s = self.state
        s.xray_image = None
        s.xray_image_path = None
        s.xray_points_uv = None
        s.xray_points_confirmed = False
        s.xray_marker_overlay_bgr = None
        s.marker_radius_px = None
        s.xray_lattice_uv = None
        s.xray_lattice_xy_mm = None
        s.xray_points_xyz_c = None
        s.plane_confirmed = False
        s.checkerboard_corners_uv = None
        s.checkerboard_corners_uv_9 = None
        s.checkerboard_corners_confirmed = False
        s.pnp_ransac_threshold_px = None
        s.T_cx = None
        s.T_xc = None
        s.tip_uv_c = None
        s.tip_xyz_c = None
        s.T_tc = None
        s.d_x = None
        s.H_xc = None

        self.page_markers.on_reset_selection()
        self.page_markers.refresh()
        self.page_cam2xray.reset_results()

        self.mode_b_active = False
        if self.current_step != self.IDX_PLANE:
            self._switch_to_page(self.IDX_PLANE)
        # after _switch_to_page / on_enter so the page ends up idle with
        # Start enabled, whatever state the previous capture left it in
        self.page_plane.reset_for_new_capture()
        self.update_ui()

    # ------------------------------------------------------------------
    # Wizard gating
    # ------------------------------------------------------------------

    def step_complete(self, idx: int) -> bool:
        if idx == self.IDX_XRAY:
            return self.state.has_xray_intrinsics
        if idx == self.IDX_CAM_CALIB:
            return self.state.has_rgb_intrinsics
        if idx == self.IDX_PLANE:
            return self.state.has_plane_confirmed
        if idx == self.IDX_MARKERS:
            return self.state.has_xray_points_confirmed
        if idx == self.IDX_CAM2XRAY:
            return self.state.has_cam_to_xray
        if idx == self.IDX_PTR_DEPTH:
            return self.state.has_d_x
        if idx == self.IDX_OVERLAY_PREVIEW:
            return self.state.has_H_xc and self.state.has_xray_image_anatomy
        if idx == self.IDX_TOOL_TRACKING:
            return self.state.has_tool
        if idx == self.IDX_OVERLAY_LIVE:
            # Checked only once the Tool Tracking step before it is done.
            return (
                self.state.has_H_xc
                and self.state.has_xray_image_anatomy
                and self.state.has_tool
            )
        return False

    def can_enter_step(self, idx: int) -> bool:
        if idx not in self.mode_a_indices:
            return False

        step_pos = self.mode_a_indices.index(idx)

        for prev_idx in self.mode_a_indices[:step_pos]:
            if not self.step_complete(prev_idx):
                return False

        return True

    # ------------------------------------------------------------------
    # Navigation wiring
    # ------------------------------------------------------------------

    def _wire_nav(self, page: QWidget, idx: int) -> None:
        if hasattr(page, "btn_back"):
            page.btn_back.clicked.connect(lambda: self._handle_back(idx))
        if hasattr(page, "btn_next"):
            page.btn_next.clicked.connect(lambda: self._handle_next(idx))

    def _handle_back(self, idx: int) -> None:
        if idx in self.mode_a_indices:
            self.go_to(idx - 1)
            return

        if idx in self.mode_b_indices:
            b_pos = self.mode_b_indices.index(idx)
            if b_pos > 0:
                self.go_to(self.mode_b_indices[b_pos - 1])
            return

    def _handle_next(self, idx: int) -> None:
        if idx in self.mode_a_indices:
            if idx == self.IDX_CAM2XRAY:
                self.go_to(self.IDX_PTR_DEPTH)
            else:
                self.go_to(idx + 1)
            return

        if idx in self.mode_b_indices:
            b_pos = self.mode_b_indices.index(idx)
            if b_pos < len(self.mode_b_indices) - 1:
                self.go_to(self.mode_b_indices[b_pos + 1])
            return

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_complete_changed(self):
        self.update_ui()

    def can_enter_mode_b(self) -> bool:
        return self.state.has_xray_to_cam

    # ------------------------------------------------------------------
    # Layout mode
    # ------------------------------------------------------------------

    def _apply_layout_mode(self) -> None:
        is_live = self.current_step == self.IDX_OVERLAY_LIVE

        if is_live:
            self.sidebar.hide()
            self.lbl_step_title.hide()

            self.root_layout.setContentsMargins(0, 0, 0, 0)
            self.root_layout.setSpacing(0)

            self.content_card.setStyleSheet("background: black; border-radius: 0px;")
            self.content_layout.setContentsMargins(0, 0, 0, 0)
            self.content_layout.setSpacing(0)
        else:
            self.sidebar.show()
            self.lbl_step_title.show()

            self.root_layout.setContentsMargins(*self._normal_root_margins)
            self.root_layout.setSpacing(16)

            self.content_card.setStyleSheet("")
            self.content_layout.setContentsMargins(*self._normal_content_margins)
            self.content_layout.setSpacing(12)

    # ------------------------------------------------------------------
    # UI update
    # ------------------------------------------------------------------

    def update_ui(self):
        self.pages.setCurrentIndex(self.current_step)
        self._apply_layout_mode()

        self.step_list.blockSignals(True)
        self.runtime_list.blockSignals(True)

        if self.current_step in self.mode_a_indices:
            self.step_list.setCurrentRow(self.mode_a_indices.index(self.current_step))
            self.runtime_list.clearSelection()
        elif self.current_step in self.mode_b_indices:
            self.step_list.clearSelection()
            self.runtime_list.setCurrentRow(self.mode_b_indices.index(self.current_step))
        else:
            self.step_list.clearSelection()
            self.runtime_list.clearSelection()

        self.step_list.blockSignals(False)
        self.runtime_list.blockSignals(False)

        def _set_item_text(item: QListWidgetItem, done: bool, locked: bool, base: str):
            prefix = "✅ " if done else ("🔒 " if locked else "⬜ ")
            item.setText(prefix + base)

        done_a = [self.step_complete(i) for i in self.mode_a_indices]

        for pos, idx in enumerate(self.mode_a_indices):
            if self.mode_b_active:
                locked = True
            else:
                locked = (not self.can_enter_step(idx)) and (not done_a[pos])

            if idx == self.IDX_XRAY:
                _set_item_text(self.item_xray, done_a[pos], locked, "1) Load X-ray intrinsics")
            elif idx == self.IDX_CAM_CALIB:
                _set_item_text(self.item_rgb, done_a[pos], locked, "2) Load camera intrinsics")
            elif idx == self.IDX_PLANE:
                _set_item_text(self.item_plane, done_a[pos], locked, "3) Plane fitting")
            elif idx == self.IDX_MARKERS:
                _set_item_text(self.item_markers, done_a[pos], locked, "4) X-ray marker selection")
            elif idx == self.IDX_CAM2XRAY:
                _set_item_text(self.item_cam2xray, done_a[pos], locked, "5) Camera → X-ray (PnP)")

        done_b = [self.step_complete(i) for i in self.mode_b_indices]

        for pos, idx in enumerate(self.mode_b_indices):
            locked = not (self.mode_b_active or self.can_enter_mode_b())

            if idx == self.IDX_PTR_DEPTH:
                _set_item_text(self.item_ptr_depth, done_b[pos], locked, "Pointer Tool Depth")
            elif idx == self.IDX_OVERLAY_PREVIEW:
                _set_item_text(self.item_overlay_preview, done_b[pos], locked, "Overlay Preview")
            elif idx == self.IDX_TOOL_TRACKING:
                _set_item_text(self.item_tool_tracking, done_b[pos], locked, "Tool Tracking")
            elif idx == self.IDX_OVERLAY_LIVE:
                _set_item_text(self.item_overlay_live, done_b[pos], locked, "Overlay Live")

        if self.current_step == self.IDX_XRAY:
            self.lbl_step_title.setText("Step 1 — Load X-ray intrinsics")
        elif self.current_step == self.IDX_CAM_CALIB:
            self.lbl_step_title.setText("Step 2 — Load camera intrinsics")
        elif self.current_step == self.IDX_PLANE:
            self.lbl_step_title.setText("Step 3 — Plane fitting")
        elif self.current_step == self.IDX_MARKERS:
            self.lbl_step_title.setText("Step 4 — X-ray marker selection")
        elif self.current_step == self.IDX_CAM2XRAY:
            self.lbl_step_title.setText("Step 5 — Camera → X-ray (PnP)")
        elif self.current_step == self.IDX_PTR_DEPTH:
            self.lbl_step_title.setText("Mode B — Pointer Tool Depth")
        elif self.current_step == self.IDX_OVERLAY_PREVIEW:
            self.lbl_step_title.setText("Mode B — Overlay Preview")
        elif self.current_step == self.IDX_TOOL_TRACKING:
            self.lbl_step_title.setText("Mode B — Tool Tracking")
        elif self.current_step == self.IDX_OVERLAY_LIVE:
            self.lbl_step_title.setText("")
        else:
            self.lbl_step_title.setText("")

        self._update_page_nav_buttons()

    def _update_page_nav_buttons(self) -> None:
        p = self.pages.currentWidget()

        if self.current_step in self.mode_a_indices:
            if hasattr(p, "btn_back"):
                p.btn_back.setEnabled(self.current_step > self.IDX_XRAY)

            if hasattr(p, "btn_next"):
                if self.current_step == self.IDX_CAM2XRAY:
                    allow = self.step_complete(self.current_step) and self.can_enter_mode_b()
                    p.btn_next.setEnabled(bool(allow))
                else:
                    allow = (
                        self.step_complete(self.current_step)
                        and self.can_enter_step(self.current_step + 1)
                    )
                    p.btn_next.setEnabled(bool(allow))
            return

        if self.current_step in self.mode_b_indices:
            b_pos = self.mode_b_indices.index(self.current_step)

            if hasattr(p, "btn_back"):
                p.btn_back.setEnabled(b_pos > 0)

            if hasattr(p, "btn_next"):
                p.btn_next.setEnabled(b_pos < len(self.mode_b_indices) - 1)
            return

    # ------------------------------------------------------------------
    # Step switching
    # ------------------------------------------------------------------

    def go_to(self, idx: int) -> None:
        idx = int(idx)
        idx = max(0, min(self.pages.count() - 1, idx))

        if idx == self.current_step:
            return

        current_in_a = self.current_step in self.mode_a_indices
        current_in_b = self.current_step in self.mode_b_indices
        target_in_a = idx in self.mode_a_indices
        target_in_b = idx in self.mode_b_indices

        if self.mode_b_active and target_in_a:
            return

        if target_in_a:
            if current_in_a and idx > self.current_step:
                if not self.step_complete(self.current_step):
                    return
                if not self.can_enter_step(idx):
                    return

        elif target_in_b:
            if not self.mode_b_active:
                if self.current_step != self.IDX_CAM2XRAY:
                    return
                if not self.step_complete(self.current_step):
                    return
                if not self.can_enter_mode_b():
                    return

                self.mode_b_active = True
            else:
                if not current_in_b:
                    return
        else:
            return

        self._switch_to_page(idx)

    def _call_page_enter(self, idx: int):
        p = self.pages.widget(idx)
        if p and hasattr(p, "on_enter"):
            p.on_enter()

    def _call_page_leave(self, idx: int):
        p = self.pages.widget(idx)
        if p and hasattr(p, "on_leave"):
            p.on_leave()

    def _switch_to_page(self, idx: int) -> None:
        self._call_page_leave(self.current_step)

        self.current_step = idx

        self.pages.setCurrentIndex(self.current_step)
        self._call_page_enter(self.current_step)

        self.update_ui()

    def closeEvent(self, event):
        for i in range(self.pages.count()):
            self._call_page_leave(i)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)

    dlg = StartupDialog()
    if dlg.exec() != StartupDialog.Accepted:
        sys.exit(0)

    win = MainWindow(dlg.choice())
    win.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()