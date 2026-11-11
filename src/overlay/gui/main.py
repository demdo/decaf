# overlay/gui/main.py
#
# Wizard runner (Mode A):
#   1) X-ray intrinsics
#   2) Camera intrinsics
#   3) Plane Fitting
#   4) X-ray Marker Selection
#   5) Camera -> X-ray Calibration (PnP)
#
# Steps 3-5 run once per table height (N_HEIGHTS board placements, title
# counter "(k/3)"); 'Next height' on step 5 returns to step 3, and after the
# last height step 5 pools all placements into T_cx / T_xc. Next to the
# title (steps 3-5): 'Redo height' (current height only) and 'New series'.
#
# Uses the per-page template navigation buttons (btn_back / btn_next).
# Sidebar is display-only (wizard gating), no click navigation.
#
# The dialog before the window opens (gui/evaluation.py) also asks for the
# workflow, in both modes: Single View (one AP X-ray) or Multi-View (lateral
# and AP X-ray, each with a ChArUco board pose) -> state.workflow.
#
# Two modes, chosen in a dialog before the window opens (gui/evaluation.py):
#   Normal      - nothing is written to disk.
#   Evaluation  - every capture is saved into one session folder under a
#                 label from the bar at the top (plus the height, "_h2",
#                 during steps 3-5); the calibration files can be loaded
#                 automatically and "New series" returns to step 3
#                 without restarting the GUI.
#
# Tracking test (Evaluation only, a check box in the start dialog): the
# camera intrinsics (step 2) are loaded from the dialog's file and the window
# opens on Tool Tracking - no X-ray calibration. Tool Tracking -> Overlay Live
# (camera image + drill) and back; Mode A and the other Mode B pages stay
# locked. Should the camera file not load, step 2 stays open to load one by
# hand, and its Next leads to Tool Tracking.

import re
import sys
from pathlib import Path
from PySide6.QtCore import Qt, QTimer
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
    QMessageBox,
    QInputDialog,
)

from overlay.gui.state import WORKFLOW_SINGLE, SessionState
from overlay.gui.xray_link import xray_link
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
from overlay.gui.pages.page_camera_to_xray_calibration import (
    CameraToXrayCalibrationPage,
    N_HEIGHTS,
)
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
        # tool tracking without the X-ray calibration (Evaluation only)
        self.tracking_test = self.eval_session is not None and self.choice.tracking_test
        # Single View or Multi-View (start dialog); the tracking test is single
        self.state.workflow = WORKFLOW_SINGLE if self.tracking_test else self.choice.workflow
        if self.eval_session is not None:
            self.eval_session.set_meta("workflow", self.state.workflow)

        title = "DeCAF Calibration — Mode A (Wizard)"
        if self.tracking_test:
            title = "DeCAF — Tracking test (no X-ray calibration)"
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
        # 1 for the first capture of the current height, 2.. after 'Redo height'
        # (Evaluation file names h2, h2b, h2c, ...)
        self._height_attempt = 1
        # Evaluation: positions in EvaluationSession.used_labels where the
        # current attempt / series started (names what gets discarded)
        self._attempt_mark = 0
        self._series_mark = 0

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

        steps_title = QLabel(
            "Calibration (Mode A) — camera intrinsics only" if self.tracking_test
            else "Calibration (Mode A)"
        )
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

        runtime_title = QLabel(
            "Runtime (Mode B) — tracking test" if self.tracking_test else "Runtime (Mode B)"
        )
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
            self.eval_bar.btn_new_capture.clicked.connect(self.new_series)
            if self.tracking_test:
                # no Cam2X series to start over
                self.eval_bar.set_mode("TRACKING TEST")
            self.content_layout.addWidget(self.eval_bar)

        self.lbl_step_title = QLabel("")
        self.lbl_step_title.setObjectName("ContentTitle")

        # Cam2X series controls next to the title (steps 3-5). 'New series'
        # only in Normal mode: Evaluation mode has it in its bar, where it
        # also works from Mode B.
        self.btn_redo_height = QPushButton("Redo height")
        self.btn_redo_height.setObjectName("SecondaryBtn")
        self.btn_redo_height.setToolTip(
            "Capture a height again from step 3: the current one or a finished one "
            "(e.g. the outlier of the pooled table); the other heights stay."
        )
        self.btn_redo_height.clicked.connect(lambda: self.redo_height())
        self.btn_series_new = QPushButton("New series")
        self.btn_series_new.setObjectName("SecondaryBtn")
        self.btn_series_new.setToolTip("Discard all heights of this series and start again at step 3.")
        self.btn_series_new.clicked.connect(lambda: self.new_series())
        for b in (self.btn_redo_height, self.btn_series_new):
            # NoFocus: SPACE captures / accepts on the pages and must not
            # re-click the button; compact, so the title row keeps its height
            b.setFocusPolicy(Qt.NoFocus)
            b.setStyleSheet("padding: 2px 10px; border-radius: 6px;")

        self.title_row = QWidget()
        title_lay = QHBoxLayout(self.title_row)
        title_lay.setContentsMargins(0, 0, 0, 0)
        title_lay.setSpacing(8)
        title_lay.addWidget(self.lbl_step_title, stretch=1)
        title_lay.addWidget(self.btn_redo_height)
        title_lay.addWidget(self.btn_series_new)
        self.content_layout.addWidget(self.title_row)

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
        # camera image + drill without the X-ray chain only in the tracking test
        self.page_overlay_live.tracking_test = self.tracking_test

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

        self.page_cam2xray.btn_next_height.clicked.connect(self.next_height)
        self.page_plane.on_redo_height = lambda: self.redo_height(confirm=False)
        if self.eval_session is not None:
            self.eval_session.tag_provider = self._eval_tag

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

        # C-arm link: X-rays sent from the C-arm (DICOM over Ethernet) go to
        # the page on screen if it is waiting for one (accept_received_xray);
        # otherwise they stay on disk only (data/xray_received/).
        self.xray_link = xray_link()
        self.xray_link.received.connect(self._on_xray_received)
        self.xray_link.start()

        if self.tracking_test:
            self._start_tracking_test()
        elif self.eval_session is not None and self.choice.autoload_calibration:
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

    def _start_tracking_test(self) -> None:
        """Tracking test: step 2 from the camera file of the startup dialog,
        then straight to Tool Tracking. If the file does not load, step 2
        stays open (its Next leads to Tool Tracking)."""
        ev = self.eval_session
        path = self.choice.camera_calib_path
        ev.set_meta("tracking_test", True)
        if self.page_cam_calib.load_from_path(str(path)):
            self._enter_tracking_test()
            return
        ev.log("tracking_test", "-", {
            "camera_intrinsics": "not loaded",
            "start_file": path.name,
        }, tagged=False)
        self._switch_to_page(self.IDX_CAM_CALIB)

    def _enter_tracking_test(self, by_hand: bool = False) -> None:
        """Records the camera intrinsics the test runs with (the dialog's
        file, or one loaded by hand on step 2), then Tool Tracking."""
        ev = self.eval_session
        path = self.page_cam_calib.loaded_path
        if path is not None:
            ev.record_calibration_file("K_rgb", path)
        numbers = {"camera_intrinsics": Path(path).name if path else "?"}
        if by_hand:
            numbers["loaded"] = "by hand (step 2)"
        ev.log("tracking_test", "-", numbers, tagged=False)
        self.mode_b_active = True
        self._switch_to_page(self.IDX_TOOL_TRACKING)

    def _b_sequence(self) -> list[int]:
        """The Mode B pages Back / Next walk through; the tracking test has
        only Tool Tracking and Overlay Live."""
        if self.tracking_test:
            return [self.IDX_TOOL_TRACKING, self.IDX_OVERLAY_LIVE]
        return self.mode_b_indices

    def _eval_tag(self) -> str:
        """Height tag for Evaluation file names: "h<k>" in steps 3-5."""
        in_series = (
            not self.mode_b_active
            and self.current_step in (self.IDX_PLANE, self.IDX_MARKERS, self.IDX_CAM2XRAY)
        )
        if not in_series:
            return ""
        tag = f"h{self.page_cam2xray.current_height()}"
        if self._height_attempt > 1:
            tag += chr(ord("a") + self._height_attempt - 1)   # h2b, h2c: height redone
        return tag

    def next_height(self) -> None:
        """Step 5 'Next height': keep the solved placement and capture the
        next one (steps 3-5 again)."""
        if not self.page_cam2xray.commit_height():
            return
        self._clear_board_capture()
        self._fresh_attempt()

    def _fresh_attempt(self) -> None:
        """Start capturing the current height: the attempt letter makes its
        Evaluation files share no name with earlier ones (h2, h2b, ...).
        Call on step 3, after _clear_board_capture."""
        self._height_attempt = 1
        ev = self.eval_session
        if ev is not None:
            while (
                self._eval_tag()
                and self._height_attempt < 26
                and any(ev.dir.glob(f"{ev.file_label()}_*"))
            ):
                self._height_attempt += 1
            self._attempt_mark = len(ev.used_labels)
        self.update_ui()

    def _series_labels_since(self, mark: int) -> list[str]:
        """File labels of series captures ("cal_h2b") saved since mark."""
        ev = self.eval_session
        out: list[str] = []
        for lab in ([] if ev is None else ev.used_labels[mark:]):
            if re.search(r"_h\d+[a-z]?$", lab) and lab not in out:
                out.append(lab)
        return out

    def _height_has_data(self) -> bool:
        s = self.state
        return bool(
            s.plane_confirmed
            or s.checkerboard_corners_confirmed
            or s.xray_image is not None
            or self.page_cam2xray.solved_heights() > len(s.cam2x_heights)
        )

    def redo_height(self, confirm: bool = True) -> None:
        """Capture a height again from step 3: the current one (plane fit,
        X-ray, its solve) or, chosen from a list, a finished one - e.g. the
        outlier of the pooled table. The other heights stay. Also brings a
        stuck plane page back to 'Start'."""
        P = self.page_cam2xray
        k = P.current_height()
        drop = None   # index of a finished height to capture again
        if confirm and self.state.cam2x_heights:
            items = P.finished_height_items() + [f"h{k}  (current)"]
            item, ok = QInputDialog.getItem(
                self,
                "Redo height",
                "Capture which height again from step 3? The other finished heights "
                "stay; an unfinished current capture is discarded.",
                items,
                len(items) - 1,
                False,
            )
            if not ok:
                return
            i = items.index(item)
            drop = i if i < len(items) - 1 else None
        elif confirm and self._height_has_data():
            ans = QMessageBox.question(
                self,
                "Redo height",
                f"Discard the plane fit and X-ray of height {k} and capture it again "
                "from step 3? The finished heights stay.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return

        # what is discarded, by the names its files were written under
        # (they stay on disk; the next attempt gets names of its own, h2b)
        current = self._series_labels_since(self._attempt_mark)
        discarded = current
        if drop is not None:
            current_solved = P.solved_heights() > len(self.state.cam2x_heights)
            h = P.drop_height(drop)   # a solved current height stays as finished
            discarded = [h.eval_label or f"h{drop + 1}"] + ([] if current_solved else current)
        ev = self.eval_session
        if ev is not None and discarded:
            ev.log("series", "-", {"discarded_height": ",".join(discarded)}, tagged=False)
        self._clear_board_capture()
        self._fresh_attempt()

    def new_series(self) -> None:
        """Start the Cam2X series over: all heights are discarded. From step
        5 and, in Evaluation mode, from the bar (also from Mode B)."""
        n = self.page_cam2xray.solved_heights()
        if n > 0:
            ans = QMessageBox.question(
                self,
                "New series",
                f"Discard the {n} height(s) captured so far and start again at step 3?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
        ev = self.eval_session
        if ev is not None:
            series = self._series_labels_since(self._series_mark)
            if series:
                ended = sorted({re.sub(r"_h\d+[a-z]?$", "", lab) for lab in series})
                ev.log("series", "-", {
                    "ended_series": ",".join(ended),
                    "heights_solved": n,
                    "pooled": self.state.T_xc is not None,
                }, tagged=False)
            # the new series gets a label nothing was saved under yet (cal ->
            # cal_2, cal1 -> cal2, '' -> capture_2), so its files share no
            # name with earlier captures (also Mode B ones like plane07_depth)
            base = ev.sanitize_label(ev.label)
            if any(ev.dir.glob(f"{base}_*")):
                lab = ev.next_label(base)
                while any(ev.dir.glob(f"{ev.sanitize_label(lab)}_*")):
                    lab = ev.next_label(lab)
                ev.set_label(lab)
        self.state.cam2x_heights.clear()
        self._clear_board_capture()
        self._fresh_attempt()
        if ev is not None:
            self._series_mark = len(ev.used_labels)

    def _clear_board_capture(self) -> None:
        """Keeps the intrinsics and the finished heights, clears everything
        the wizard derives from one board capture and returns to step 3,
        also from Mode B."""
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
            if self.tracking_test:
                # camera image + drill: no H_xc, X-ray or d_x
                return self.state.has_rgb_intrinsics and self.state.has_tool
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
            if self.tracking_test:
                # step 2 is where the tracking test starts
                return
            self.go_to(idx - 1)
            return

        seq = self._b_sequence()
        if idx in seq:
            b_pos = seq.index(idx)
            if b_pos > 0:
                self.go_to(seq[b_pos - 1])
            return

    def _handle_next(self, idx: int) -> None:
        if idx in self.mode_a_indices:
            if self.tracking_test:
                if idx == self.IDX_CAM_CALIB and self.state.has_rgb_intrinsics:
                    self._enter_tracking_test(by_hand=True)
                return
            if idx == self.IDX_CAM2XRAY:
                self.go_to(self.IDX_PTR_DEPTH)
            else:
                self.go_to(idx + 1)
            return

        seq = self._b_sequence()
        if idx in seq:
            b_pos = seq.index(idx)
            if b_pos < len(seq) - 1:
                self.go_to(seq[b_pos + 1])
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
            self.title_row.hide()

            self.root_layout.setContentsMargins(0, 0, 0, 0)
            self.root_layout.setSpacing(0)

            self.content_card.setStyleSheet("background: black; border-radius: 0px;")
            self.content_layout.setContentsMargins(0, 0, 0, 0)
            self.content_layout.setSpacing(0)
        else:
            self.sidebar.show()
            self.title_row.show()

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

        # Steps 3-5 run once per height: counter "(k/3)", and a tick only
        # once the whole series is done (step 5 = pooled pose exists).
        k = f"({self.page_cam2xray.current_height()}/{N_HEIGHTS})"
        # the tracking test has no series: steps 3-5 without the counter
        k_item = "" if self.tracking_test else f" {k}"
        series_done = self.step_complete(self.IDX_CAM2XRAY)

        for pos, idx in enumerate(self.mode_a_indices):
            if self.mode_b_active:
                locked = True
            elif self.tracking_test:
                # only step 2 belongs to the tracking test
                locked = idx != self.IDX_CAM_CALIB
            else:
                locked = (not self.can_enter_step(idx)) and (not done_a[pos])

            if idx == self.IDX_XRAY:
                _set_item_text(self.item_xray, done_a[pos], locked, "1) Load X-ray intrinsics")
            elif idx == self.IDX_CAM_CALIB:
                _set_item_text(self.item_rgb, done_a[pos], locked, "2) Load camera intrinsics")
            elif idx == self.IDX_PLANE:
                _set_item_text(self.item_plane, done_a[pos] and series_done, locked, f"3) Plane fitting{k_item}")
            elif idx == self.IDX_MARKERS:
                _set_item_text(self.item_markers, done_a[pos] and series_done, locked, f"4) X-ray marker selection{k_item}")
            elif idx == self.IDX_CAM2XRAY:
                _set_item_text(self.item_cam2xray, done_a[pos], locked, f"5) Camera → X-ray{k_item}")

        done_b = [self.step_complete(i) for i in self.mode_b_indices]

        for pos, idx in enumerate(self.mode_b_indices):
            if self.tracking_test:
                locked = not self.mode_b_active or idx not in self._b_sequence()
            else:
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
            self.lbl_step_title.setText(
                "Tracking test — Load camera intrinsics" if self.tracking_test
                else "Step 2 — Load camera intrinsics"
            )
        elif self.current_step == self.IDX_PLANE:
            self.lbl_step_title.setText(f"Step 3 — Plane fitting {k}")
        elif self.current_step == self.IDX_MARKERS:
            self.lbl_step_title.setText(f"Step 4 — X-ray marker selection {k}")
        elif self.current_step == self.IDX_CAM2XRAY:
            self.lbl_step_title.setText(f"Step 5 — Camera → X-ray {k}")
        elif self.current_step == self.IDX_PTR_DEPTH:
            self.lbl_step_title.setText("Mode B — Pointer Tool Depth")
        elif self.current_step == self.IDX_OVERLAY_PREVIEW:
            self.lbl_step_title.setText("Mode B — Overlay Preview")
        elif self.current_step == self.IDX_TOOL_TRACKING:
            self.lbl_step_title.setText(
                "Tracking test — Tool Tracking" if self.tracking_test
                else "Mode B — Tool Tracking"
            )
        elif self.current_step == self.IDX_OVERLAY_LIVE:
            self.lbl_step_title.setText("")
        else:
            self.lbl_step_title.setText("")

        in_series = (
            not self.mode_b_active
            and self.current_step in (self.IDX_PLANE, self.IDX_MARKERS, self.IDX_CAM2XRAY)
        )
        self.btn_redo_height.setVisible(in_series)
        self.btn_series_new.setVisible(in_series and self.eval_bar is None)
        self.btn_series_new.setEnabled(self.page_cam2xray.solved_heights() > 0)

        if self.eval_bar is not None:
            self.eval_bar.set_tag(self._eval_tag())

        self._update_page_nav_buttons()

    def _update_page_nav_buttons(self) -> None:
        p = self.pages.currentWidget()

        if self.tracking_test and self.current_step in self.mode_a_indices:
            # step 2, only when the camera file did not load: Next -> Tool Tracking
            if hasattr(p, "btn_back"):
                p.btn_back.setEnabled(False)
            if hasattr(p, "btn_next"):
                p.btn_next.setEnabled(self.state.has_rgb_intrinsics)
            return

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

        seq = self._b_sequence()
        if self.current_step in seq:
            b_pos = seq.index(self.current_step)

            if hasattr(p, "btn_back"):
                p.btn_back.setEnabled(b_pos > 0)

            if hasattr(p, "btn_next"):
                allow = b_pos < len(seq) - 1
                if allow and self.tracking_test:
                    # Overlay Live needs the camera intrinsics and the tool
                    allow = self.step_complete(seq[b_pos + 1])
                p.btn_next.setEnabled(allow)
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
            if self.tracking_test and (
                idx not in self._b_sequence()
                or (idx == self.IDX_OVERLAY_LIVE and not self.step_complete(idx))
            ):
                return
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

    def _on_xray_received(self, path: str) -> None:
        # A dialog is open (file dialog, a question): hand the image over
        # once it is closed, so no page swaps its X-ray under an open question.
        if QApplication.activeModalWidget() is not None:
            QTimer.singleShot(250, lambda: self._on_xray_received(path))
            return
        page = self.pages.widget(self.current_step)
        take = getattr(page, "accept_received_xray", None)
        if take is None or not take(path):
            print(f"[XrayLink] X-ray received, no step waiting for one: {path}")

    def closeEvent(self, event):
        self.xray_link.stop()
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