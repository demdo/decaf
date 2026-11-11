# overlay/gui/pages/page_xray_acquisition.py
"""
Step - X-ray Acquisition (Multi-View only)

Two X-ray views, each with the ChArUco board pose of its C-arm stance
(overlay.tools.view_registration): ① View 2 lateral first, then ② View 1 AP,
the stance the C-arm stays in and tracking runs in. View 2 is optional.

Per button: first the 1 s board pose (BoardPoseCapture; the window starts
again while the C-arm still swings), then the page waits for the X-ray of that
stance (DICOM from the C-arm via MainWindow -> accept_received_xray, or 'From
folder (USB)'). At its arrival one camera frame is checked against the pose:
a median corner shift above BoardPoseConfig.motion_px sets the view with a
warning. ESC / Cancel stops a running view; an earlier capture of it stays
until a new X-ray replaces it.

An X-ray that arrives while no view waits is held ('pending') together with
the board corners of that moment; the next view button takes it right after
its board pose if the board has not moved since.

Retaking View 2 makes View 1 out of date (AP is always last): View 1 is dimmed
and 'Next' stays locked until View 1 is taken again.

View: the camera image (same size as the Overlay Preview) with the ChArUco
corners, green found, grey the predicted missing ones; once both views are set
and nothing runs, the two X-rays side by side instead (View 1 AP left, View 2
lateral right, as the live page), as high as the camera image.

Results: SessionState.xray_views (XrayViewCapture per view); View 1 also goes
to xray_image_anatomy(_path) for the live page. Evaluation mode: per view the
X-ray copy, the board pose npz (BoardPose.save_npz) and a json.

Speed: the board preview runs at most PREVIEW_INTERVAL_S apart, add() only
inside the 1 s window, no copy of the camera frame (the display is a resize).
The X-ray undistortion maps are computed once in a thread at entry; each
X-ray is flipped and undistorted in a thread too.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from overlay.gui.state import VIEW_AP, VIEW_LAT, SessionState, XrayViewCapture
from overlay.gui.pages.templates.templ_base_image import BaseImagePage
from overlay.gui.pages.templates.templ_live_image import LiveImagePage
from overlay.gui.widgets.widget_live_panels import XrayToolPanel
from overlay.gui.widgets.widget_xray_waiting import XrayWaitingOverlay
from overlay.tools.view_registration import (
    PHI_LATERAL_DEG,
    BoardPoseCapture,
    BoardPoseConfig,
    camera_motion,
    jsonable,
)
from overlay.tools.xray_io import read_xray_gray_u8, xray_pixel_md5


PREVIEW_INTERVAL_S = 0.2          # board preview at most 5x/s outside the 1 s window
POSE_DRAW_EVERY = 2               # inside the window: draw every 2nd frame
CAMERA_VIEWPORT = (576, 324)      # as the Overlay Preview
PAIR_VIEWPORT = (648, 324)        # two squares as high as the camera image
UI_REFERENCE_WIDTH = 1920.0       # X-ray panel fonts as on the live page
UI_MIN = 0.72

IDLE, POSE, WAIT = "idle", "pose", "wait"

GREEN, GREY, BLUE, ORANGE, RED = "#2b8a3e", "#6c757d", "#1c64d1", "#d9480f", "#c92a2a"

_VIEWS = {
    VIEW_LAT: dict(card="① Lateral", load="Load X-ray View 2 (Lat)", retake="Retake View 2 (Lat)",
                   name="View 2 (Lat)", stance="lateral", tag="view2_lat", phi=PHI_LATERAL_DEG,
                   panel="X-ray View 2 (Lat)"),
    VIEW_AP: dict(card="② AP", load="Load X-ray View 1 (AP)", retake="Retake View 1 (AP)",
                  name="View 1 (AP)", stance="AP", tag="view1_ap", phi=0.0,
                  panel="X-ray View 1 (AP)"),
}


# ============================================================
# Helpers
# ============================================================

def _corner_rows(detection, n: int) -> np.ndarray:
    """(n, 2) corners by ChArUco id, NaN = not found."""
    rows = np.full((n, 2), np.nan)
    if detection.ids.size:
        rows[detection.ids] = detection.corners
    return rows


def _median_shift(a: Optional[np.ndarray], b: Optional[np.ndarray], min_common: int = 6) -> float:
    """Median corner shift [px] between two (n, 2) corner sets; NaN without
    enough common corners."""
    if a is None or b is None:
        return float("nan")
    common = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
    if int(common.sum()) < min_common:
        return float("nan")
    return float(np.median(np.linalg.norm(a[common] - b[common], axis=1)))


def _hhmm(t: float, seconds: bool = False) -> str:
    return time.strftime("%H:%M:%S" if seconds else "%H:%M", time.localtime(t))


@dataclass
class _Pending:
    """An X-ray that arrived while no view waited for one."""
    image: np.ndarray
    path: str
    source: str
    corners: Optional[np.ndarray]     # board corners in the camera frame at its arrival
    t: float


# ============================================================
# Widgets
# ============================================================

class ViewCard(QFrame):
    """One view: title, one button, status lines. active: light blue with the
    primary button (the running view, or the one to take next)."""

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ViewCard")
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 8)
        v.setSpacing(3)
        t = QLabel(title)
        t.setStyleSheet("font-weight: 700; color: #212529; background: transparent;")
        v.addWidget(t)
        self.button = QPushButton()
        self.button.setMinimumHeight(44)
        self.button.setFocusPolicy(Qt.NoFocus)
        v.addWidget(self.button)
        self._lines_box = QVBoxLayout()
        self._lines_box.setSpacing(1)
        v.addLayout(self._lines_box)
        self._shown: Optional[tuple] = None
        self._active: Optional[bool] = None

    def set(self, button: str, lines: list, active: bool = False, enabled: bool = True) -> None:
        if active != self._active:
            self._active = active
            bg = "#e7f1ff" if active else "#ffffff"
            self.setStyleSheet(f"QFrame#ViewCard {{ background: {bg}; border-radius: 10px; }}")
            self.button.setObjectName("" if active else "SecondaryBtn")
            self.button.style().unpolish(self.button)
            self.button.style().polish(self.button)
        if self.button.text() != button:
            self.button.setText(button)
        self.button.setEnabled(enabled)
        key = tuple(lines)
        if key == self._shown:
            return
        self._shown = key
        while self._lines_box.count():
            w = self._lines_box.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        for text, colour in lines:
            lab = QLabel(text)
            lab.setWordWrap(True)
            lab.setStyleSheet(f"color: {colour}; font-size: 12px; background: transparent;")
            self._lines_box.addWidget(lab)

    def line_texts(self) -> list[str]:
        return [t for t, _ in (self._shown or ())]


class _Dim(QWidget):
    """Darkens an out-of-date X-ray panel and names why."""

    def __init__(self, panel: QWidget):
        super().__init__(panel)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.text = ""
        panel.installEventFilter(self)
        self.setGeometry(panel.rect())
        self.hide()

    def eventFilter(self, obj, ev):
        if obj is self.parent() and ev.type() == QEvent.Resize:
            self.setGeometry(obj.rect())
        return False

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 150))
        f = QFont(self.font())
        f.setPixelSize(max(11, int(self.width() * 0.045)))
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.setPen(QColor(255, 169, 77))
        p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.text)
        p.end()


class XrayPairView(QWidget):
    """Both X-rays side by side over the camera viewport: View 1 (AP) left,
    View 2 (Lat) right, as the X-ray view of the live page."""

    def __init__(self, viewport: QWidget):
        super().__init__(viewport)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background: rgb(8,10,13); border-radius: 12px;")
        self.panels = {VIEW_AP: XrayToolPanel(_VIEWS[VIEW_AP]["panel"], self),
                       VIEW_LAT: XrayToolPanel(_VIEWS[VIEW_LAT]["panel"], self)}
        self.dims = {k: _Dim(p) for k, p in self.panels.items()}
        viewport.installEventFilter(self)
        self.setGeometry(viewport.rect())

    def eventFilter(self, obj, ev):
        if obj is self.parent() and ev.type() == QEvent.Resize:
            self.setGeometry(obj.rect())
        return False

    def showEvent(self, ev):
        super().showEvent(ev)
        self._place()

    def resizeEvent(self, ev):
        self._place()

    def _place(self) -> None:
        w, h = self.width(), self.height()
        side = max(1, min(w // 2, h))
        x0 = (w - 2 * side) // 2
        y0 = (h - side) // 2
        win = self.window()
        ui = max(UI_MIN, min(1.0, (win.width() if win is not None else w) / UI_REFERENCE_WIDTH))
        for i, key in enumerate((VIEW_AP, VIEW_LAT)):
            p = self.panels[key]
            p.setGeometry(x0 + i * side, y0, side, side)
            p.set_ui_scale(ui, int(round(30 * ui)))


class _ViewWaitingPill(XrayWaitingOverlay):
    """The waiting pill of the Overlay Preview, naming the view while the C-arm
    link is ready (otherwise the link's own headline, e.g. offline)."""

    def __init__(self, viewport: QWidget):
        super().__init__(viewport, compact=True)
        self.headline = ""

    def paintEvent(self, ev):
        st = self._status
        if st.ready and self.headline:
            self._status = replace(st, headline=self.headline)
        try:
            super().paintEvent(ev)
        finally:
            self._status = st


# ============================================================
# Page
# ============================================================

class XrayAcquisitionPage(LiveImagePage):
    VIEWPORT_SIZE = CAMERA_VIEWPORT

    # (XrayViewCapture, working image or None) from the preparation thread
    working_ready = Signal(object, object)

    def __init__(self, state: SessionState, on_complete_changed=None, parent=None):
        self.state = state
        self.on_complete_changed = on_complete_changed

        # before super().__init__: the template renders while it builds
        self._frame: Optional[np.ndarray] = None
        self._pair_mode = False
        self._phase = IDLE
        self._active: Optional[str] = None          # view of the running capture
        self._cap: Optional[BoardPoseCapture] = None
        self._cap_key: Optional[tuple] = None
        self._pose = None                           # accepted BoardPose of the running capture
        self._t_pose: Optional[float] = None
        self._preview: Optional[dict] = None
        self._t_preview = 0.0
        self._pending: Optional[_Pending] = None
        self._seq = 0
        # new views from Overlay Live (begin_new_views): View 2 before View 1
        self._lat_first = False
        self._messages: dict = {VIEW_AP: None, VIEW_LAT: None}   # (text, colour) per card
        self._camera_ok = False
        self._draw_count = 0
        self._last_rows: Optional[list] = None
        self._last_instr: Optional[str] = None
        self._pair_shown: dict = {}
        self._motion_key: Optional[tuple] = None
        self._motion: Optional[dict] = None
        self._warm_started = False
        # tests: callable -> (image, frame_index, timestamp_s) or None, instead of the camera
        self.frame_source: Optional[Callable[[], Optional[tuple]]] = None

        super().__init__(parent)

        # The template's width ignores its own root margins (2 x 16 px); with
        # the X-ray pair wider than the camera image the view would run under
        # the right column. RIGHT_COL_W is read again only for that width.
        self.RIGHT_COL_W = BaseImagePage.RIGHT_COL_W + 2 * self.ROOT_MARGIN
        self._last_viewport_px = None

        self.setFocusPolicy(Qt.StrongFocus)
        # the timer polls the camera here (_tick); update_view only draws
        try:
            self._timer.timeout.disconnect(self.update_view)
        except (RuntimeError, TypeError):
            pass
        self._timer.timeout.connect(self._tick)
        self.working_ready.connect(self._on_working_ready)

        self._build_controls()
        self.pair = XrayPairView(self.image_label)
        self.pair.hide()
        self.pill = _ViewWaitingPill(self.image_label)
        self.pill.hide()

        self._refresh()

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def _build_controls(self) -> None:
        self.cards = {VIEW_LAT: ViewCard(_VIEWS[VIEW_LAT]["card"]),
                      VIEW_AP: ViewCard(_VIEWS[VIEW_AP]["card"])}
        self.cards[VIEW_LAT].button.clicked.connect(lambda: self._on_view_button(VIEW_LAT))
        self.cards[VIEW_AP].button.clicked.connect(lambda: self._on_view_button(VIEW_AP))

        self.btn_usb = QPushButton("From folder (USB)")
        self.btn_usb.setObjectName("SecondaryBtn")
        self.btn_usb.setMinimumHeight(36)
        self.btn_usb.setFocusPolicy(Qt.NoFocus)
        self.btn_usb.clicked.connect(self._on_usb)

        # one line for what happened outside the cards (X-ray held, duplicate, ...)
        self.lbl_note = QLabel("")
        self.lbl_note.setWordWrap(True)
        self.lbl_note.hide()

        self.controls_content.addWidget(self.cards[VIEW_LAT])
        self.controls_content.addWidget(self.cards[VIEW_AP])
        self.controls_content.addWidget(self.btn_usb)
        self.controls_content.addWidget(self.lbl_note)

    def _note(self, text: str = "", colour: str = GREY) -> None:
        self.lbl_note.setText(text)
        self.lbl_note.setStyleSheet(f"color: {colour}; font-size: 12px; background: transparent;")
        self.lbl_note.setVisible(bool(text))

    # ---------------------------------------------------------
    # Public (MainWindow, tests)
    # ---------------------------------------------------------

    @property
    def busy(self) -> bool:
        """A view is being captured (board pose or waiting for its X-ray)."""
        return self._phase != IDLE

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def active_view(self) -> Optional[str]:
        return self._active

    @property
    def pair_mode(self) -> bool:
        return self._pair_mode

    def can_continue(self) -> bool:
        """'Next': View 1 (AP) set and newer than View 2, nothing running."""
        return not self.busy and self.state.has_current_view_ap

    def begin_new_views(self) -> None:
        """Overlay Live, 'New views with board' (user 11.11.): the board was off the
        table after the last acquisition and comes back, so a new View 1 needs a new
        View 2 of the same board placement first; View 1 is locked until then. The
        views on screen stay valid meanwhile (their relative pose, which panel 3 of
        Overlay Live uses, no longer depends on the board)."""
        self._lat_first = True
        self._refresh()

    @property
    def lat_first(self) -> bool:
        return self._lat_first

    def start_view(self, view: str) -> None:
        if self.busy or not self._camera_ok:
            return
        if view == VIEW_AP and self._lat_first:
            return
        self._active = view
        self._phase = POSE
        self._pose = None
        self._t_pose = None
        self._messages[view] = None
        self._note()
        self._draw_count = 0
        self._cap.start()
        self._changed()
        self.setFocus()

    def cancel_view(self) -> None:
        if not self.busy:
            return
        view = self._active
        self._cap.cancel()
        self._log_eval(view, {"cancelled": "pose" if self._phase == POSE else "waiting for the X-ray"})
        self._phase = IDLE
        self._active = None
        self._pose = None
        kept = self.state.xray_views.get(view) is not None
        self._messages[view] = (f"Cancelled · the earlier {_VIEWS[view]['name']} stays" if kept
                                else "Cancelled", GREY)
        self._changed()

    # ---------------------------------------------------------
    # Buttons / keys
    # ---------------------------------------------------------

    def _on_view_button(self, view: str) -> None:
        if self.busy:
            if self._active == view:
                self.cancel_view()
            return
        self.start_view(view)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self.busy:
            self.cancel_view()
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_usb(self) -> None:
        if self._phase != WAIT:
            return
        dlg = QFileDialog(self.window(), f"Select the {_VIEWS[self._active]['stance']} X-ray")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.ima);;All Files (*)")
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        if dlg.exec() != QFileDialog.Accepted:
            return
        files = dlg.selectedFiles()
        path = files[0] if files else ""
        if not path or self._phase != WAIT:
            return
        try:
            img = read_xray_gray_u8(path)
        except Exception as e:
            QMessageBox.critical(self, "Load X-ray", f"Could not read the X-ray.\n\n{e}")
            return
        dup = self._duplicate_of(img)
        if dup is not None:
            self._note(f"Same X-ray as {_VIEWS[dup]['name']} · still waiting", ORANGE)
            return
        self._commit(img, path, "USB")

    # ---------------------------------------------------------
    # Received X-rays (MainWindow)
    # ---------------------------------------------------------

    def accept_received_xray(self, path: str) -> bool:
        """An X-ray from the C-arm while this page is shown. Waiting view: it
        is that view's X-ray. Otherwise it is held with the board corners of
        this moment, for the next view button. False: not taken (no camera)."""
        if not self._camera_ok:
            return False
        try:
            img = read_xray_gray_u8(path)
        except Exception as e:
            print(f"[XrayAcquisition] received X-ray not readable: {path}: {e}")
            self._note(f"X-ray not readable: {Path(path).name}", RED)
            return True
        dup = self._duplicate_of(img)
        if dup is not None:
            self._note(f"Same X-ray as {_VIEWS[dup]['name']} · ignored", ORANGE)
            return True
        if self._phase == WAIT:
            self._commit(img, path, "C-arm")
            return True
        self._pending = _Pending(img, str(path), "C-arm", self._board_rows_now(), time.time())
        if self._phase == POSE:
            self._note("X-ray received · it is taken after the board pose if the board has not moved",
                       BLUE)
        else:
            self._note("X-ray received · press the button of its view: it is taken after the "
                       "board pose if the board has not moved", ORANGE)
        return True

    def _duplicate_of(self, img: np.ndarray) -> Optional[str]:
        md5 = xray_pixel_md5(img)
        for view, cap in self.state.xray_views.items():
            if xray_pixel_md5(cap.image) == md5:
                return view
        return None

    # ---------------------------------------------------------
    # Camera tick
    # ---------------------------------------------------------

    def _poll(self) -> Optional[tuple]:
        if self.frame_source is not None:
            return self.frame_source()
        pipe = self.pipeline
        if pipe is None:
            return None
        frames = pipe.poll_for_frames()
        if not frames:
            return None
        cf = frames.get_color_frame()
        if not cf:
            return None
        return self.color_frame_to_bgr(cf), cf.get_frame_number(), cf.get_timestamp() / 1000.0

    def _tick(self) -> None:
        got = self._poll()
        if got is None:
            return
        img, idx, ts = got
        self._frame = img
        if self._phase == POSE:
            st = self._cap.add(img, idx, ts)
            if st == BoardPoseCapture.DONE:
                self._pose_done()
                self.update_view()
                return
            if st == BoardPoseCapture.FAILED:
                self._pose_failed()
                self.update_view()
                return
            self._refresh_cards()
            self._draw_count += 1
            if self._draw_count % POSE_DRAW_EVERY == 0:
                self.update_view()
            return
        if self._pair_mode:
            return          # frames kept for the arrival check only
        now = time.perf_counter()
        if now - self._t_preview >= PREVIEW_INTERVAL_S:
            self._t_preview = now
            self._preview = self._cap.preview(img)
            self._refresh_stats()
        self.update_view()

    def _board_rows_now(self) -> Optional[np.ndarray]:
        if self._frame is None or self._cap is None:
            return None
        return _corner_rows(self._cap.detect(self._frame), len(self._cap.obj))

    # ---------------------------------------------------------
    # Capture steps
    # ---------------------------------------------------------

    def _pose_done(self) -> None:
        self._pose = self._cap.result
        self._t_pose = time.time()
        pend = self._pending
        if pend is not None:
            self._pending = None
            shift = _median_shift(pend.corners, self._pose.arrays.get("mean_corners"))
            limit = self._cap.config.motion_px
            if np.isfinite(shift) and shift <= limit:
                self._commit(pend.image, pend.path, pend.source + " (held)", shift=shift)
                return
            why = ("the board was not in view when it arrived" if not np.isfinite(shift)
                   else f"the board moved {shift:.2f} px since it arrived")
            self._note(f"Held X-ray not used: {why} · take the X-ray again", ORANGE)
            self._log_eval(self._active, {"held_xray_dropped": Path(pend.path).name,
                                          "shift_px": shift})
        self._phase = WAIT
        self._changed()

    def _pose_failed(self) -> None:
        view = self._active
        result = self._cap.result
        self._messages[view] = (result.status_text(), RED)
        self._log_eval(view, {"board_pose": "rejected", "reasons": "; ".join(result.reasons)})
        self._phase = IDLE
        self._active = None
        self._pose = None
        self._changed()

    def _commit(self, img: np.ndarray, path: str, source: str, shift: Optional[float] = None) -> None:
        view = self._active
        pose = self._pose
        if shift is None:
            shift = _median_shift(self._board_rows_now(), pose.arrays.get("mean_corners"))
        self._seq += 1
        cap = XrayViewCapture(
            view=view,
            image=img,
            path=str(path),
            T_cw=np.array(pose.T_cw, dtype=np.float64),
            pose_summary=pose.summary,
            t_acquired=time.time(),
            seq=self._seq,
            phi_deg=float(_VIEWS[view]["phi"]),
            board_shift_px=float(shift),
            pose=pose,
        )
        self.state.xray_views[view] = cap
        if view == VIEW_LAT:
            self._lat_first = False
        if view == VIEW_AP:
            # the live page shows xray_image_anatomy (as after the Overlay Preview)
            self.state.xray_image_anatomy = img
            self.state.xray_image_anatomy_path = str(path)
        self._phase = IDLE
        self._active = None
        self._pose = None
        self._note()
        self._save_eval(cap, source)
        self._prepare_working(cap)
        print(f"[XrayAcquisition] {_VIEWS[view]['name']} set from {source}: {Path(path).name}, "
              f"board shift {shift:.2f} px")
        self._changed()

    # ---------------------------------------------------------
    # X-ray working images (thread)
    # ---------------------------------------------------------

    def _warm_up_remap(self) -> None:
        """The undistortion maps once, in a thread, before the first X-ray
        (several seconds; ~19 s on the laptop's battery). Square detector
        (1024 x 1024); another size only costs this precompute."""
        field = self.state.xray_dist_field
        if field is None or self._warm_started:
            return
        self._warm_started = True
        w = int(field.width)

        def job():
            try:
                field.working_remap(w, w)
            except Exception as e:
                print(f"[XrayAcquisition] undistortion maps not precomputed: {e}")

        threading.Thread(target=job, name="xray-remap-warmup", daemon=True).start()

    def _prepare_working(self, cap: XrayViewCapture) -> None:
        field = self.state.xray_dist_field
        img = cap.image

        def job():
            work = None
            try:
                work = np.ascontiguousarray(cv2.flip(img, 1))
                if field is not None:
                    work = field.undistort_image_working(work)
            except Exception as e:
                print(f"[XrayAcquisition] X-ray not undistorted: {e}")
            try:
                self.working_ready.emit(cap, work)
            except RuntimeError:
                pass        # page already gone

        threading.Thread(target=job, name="xray-working", daemon=True).start()

    def _on_working_ready(self, cap: XrayViewCapture, work) -> None:
        if work is None:
            work = np.ascontiguousarray(cv2.flip(cap.image, 1))
        cap.working = work
        if self.state.xray_views.get(cap.view) is cap:
            self._refresh()

    # ---------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------

    def get_frame(self) -> Optional[np.ndarray]:
        return self._frame

    def update_view(self) -> None:
        frame = self._frame
        if self._pair_mode or frame is None:
            self.clear_view()
            return
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        label = self.image_label
        dpr = float(label.devicePixelRatioF())
        tw = max(1, int(round(label.width() * dpr)))
        th = max(1, int(round(label.height() * dpr)))
        disp = self._resize_cover(frame, tw, th)       # a new image: drawing on it is safe
        self._draw_corners(disp, frame.shape[1], frame.shape[0], dpr)
        pm = self._to_pixmap(disp)
        pm.setDevicePixelRatio(dpr)
        label.setPixmap(pm)

    def _draw_corners(self, disp: np.ndarray, sw: int, sh: int, dpr: float) -> None:
        if self._cap is None:
            return
        th, tw = disp.shape[:2]
        s = max(tw / sw, th / sh)
        ox = (sw * s - tw) / 2.0
        oy = (sh * s - th) / 2.0
        r = max(2, int(round(3 * dpr * min(1.0, tw / (CAMERA_VIEWPORT[0] * dpr)))))
        if self._phase == POSE:
            rows = self._cap.last_corners
            found = np.zeros((0, 2)) if rows is None else rows[np.isfinite(rows[:, 0])]
            missing = np.zeros((0, 2))
        else:
            pv = self._preview
            if pv is None:
                return
            found = pv["detection"].corners
            missing = pv["missing_uv"]
        for u, v in missing:
            x, y = u * s - ox, v * s - oy
            if 0 <= x < tw and 0 <= y < th:
                cv2.circle(disp, (int(x), int(y)), r + 1, (150, 150, 150), max(1, int(dpr)), cv2.LINE_AA)
        for u, v in found:
            cv2.circle(disp, (int(round(u * s - ox)), int(round(v * s - oy))), r, (0, 200, 0), -1, cv2.LINE_AA)

    # ---------------------------------------------------------
    # Refresh
    # ---------------------------------------------------------

    def _changed(self) -> None:
        self._refresh()
        if callable(self.on_complete_changed):
            self.on_complete_changed()

    def _refresh(self) -> None:
        views = self.state.xray_views
        pair = (self._phase == IDLE and self._camera_ok
                and views.get(VIEW_AP) is not None and views.get(VIEW_LAT) is not None)
        if pair != self._pair_mode or self._last_viewport_px is None:
            self._pair_mode = pair
            self._set_viewport(PAIR_VIEWPORT if pair else CAMERA_VIEWPORT)
        self.pair.setVisible(pair)
        if pair:
            self.image_label.clear()
            self._refresh_pair()
        else:
            self._pair_shown = {}
        self.pill.headline = (f"Waiting for the {_VIEWS[self._active]['stance']} X-ray"
                              if self._phase == WAIT else "")
        self.pill.setVisible(self._phase == WAIT)
        self.btn_usb.setEnabled(self._phase == WAIT)
        self._refresh_cards()
        self._refresh_stats()
        self._refresh_instructions()

    def _set_viewport(self, wh: tuple) -> None:
        self.VIEWPORT_SIZE = wh
        self._last_viewport_px = None
        self._update_viewport_size()

    def _motion_now(self) -> Optional[dict]:
        ap, lat = self.state.xray_view_ap, self.state.xray_view_lat
        if ap is None or lat is None:
            return None
        key = (ap.seq, lat.seq)
        if key != self._motion_key:
            self._motion_key = key
            try:
                self._motion = camera_motion(lat.T_cw, ap.T_cw)
            except Exception as e:
                print(f"[XrayAcquisition] camera motion: {e}")
                self._motion = None
        return self._motion

    def _refresh_pair(self) -> None:
        outdated = self.state.view_ap_out_of_date
        motion = self._motion_now()
        for view, panel in self.pair.panels.items():
            cap = self.state.xray_views[view]
            key = (cap.seq, cap.working is not None)
            if self._pair_shown.get(view) != key:
                self._pair_shown[view] = key
                if cap.working is None:
                    panel.set_placeholder(_VIEWS[view]["panel"].upper(), "Preparing the X-ray…")
                else:
                    panel.set_xray(cap.working)
            text = f"Acquired {_hhmm(cap.t_acquired)}"
            if view == VIEW_AP and outdated:
                text += " · out of date"
            if view == VIEW_LAT and motion is not None:
                text += f" · C-arm turned {motion['angle_deg']:.1f}°"
            if np.isfinite(cap.board_shift_px) and cap.board_shift_px > self._shift_limit():
                text += " · board moved"
            panel.set_right_text(text)
            dim = self.pair.dims[view]
            dim.text = "Out of date · retake View 1 (AP)"
            dim.setVisible(view == VIEW_AP and outdated)

    def _shift_limit(self) -> float:
        return self._cap.config.motion_px if self._cap is not None else BoardPoseConfig().motion_px

    def _suggested(self) -> Optional[str]:
        ap, lat = self.state.xray_view_ap, self.state.xray_view_lat
        if self._lat_first or (ap is None and lat is None):
            return VIEW_LAT
        if ap is None or self.state.view_ap_out_of_date:
            return VIEW_AP
        return None

    def _card_state(self, view: str) -> tuple:
        meta = _VIEWS[view]
        cap = self.state.xray_views.get(view)
        if self.busy and self._active == view:
            if self._phase == POSE:
                n, need = self._cap.progress
                lines = [(f"Board pose {n}/{need} · hold the C-arm still" if n
                          else "Board pose · waiting for the board in view", BLUE)]
                if self._cap.restarts:
                    lines.append((f"Started again {self._cap.restarts}× · the C-arm still moves", GREY))
                lines.append(("ESC cancels", GREY))
            else:
                s = self._pose.summary
                lines = [(f"✓ Board pose {_hhmm(self._t_pose, True)} · {s['corners_used']}/"
                          f"{s['corners_total']} corners · {s['rms_px']:.2f} px", GREEN),
                         (f"Take the {meta['stance']} X-ray now; keep the C-arm still.", BLUE)]
                if cap is not None:
                    lines.append((f"The earlier {meta['name']} stays until the new X-ray arrives.", GREY))
            return "Cancel", lines, True, True

        lines: list = []
        lat = self.state.xray_view_lat
        if self._lat_first:
            if view == VIEW_AP:
                lines = [("Locked until View 2 (Lat) is retaken: the board has moved since.", ORANGE)]
                if cap is not None:
                    lines.insert(0, ("✓ X-ray from before (still shown in Live)", GREY))
                return (meta["retake"] if cap is not None else meta["load"]), lines, False, False
            lines = [("Board back on the table? Turn lateral, then retake.", BLUE)]
        elif view == VIEW_AP and self.state.view_ap_out_of_date:
            lines = [("Out of date: older than View 2.", ORANGE), ("Turn back to AP and retake it.", BLUE)]
        elif cap is not None:
            s = cap.pose_summary
            lines = [(f"✓ Board {s['corners_used']}/{s['corners_total']} · {s['rms_px']:.2f} px", GREEN),
                     (f"✓ X-ray {_hhmm(cap.t_acquired, True)}", GREEN)]
            if np.isfinite(cap.board_shift_px) and cap.board_shift_px > self._shift_limit():
                lines.append((f"⚠ Board moved {cap.board_shift_px:.2f} px between board pose and X-ray: "
                              f"retake {meta['name']}", ORANGE))
            elif not np.isfinite(cap.board_shift_px):
                lines.append(("Board not in view at the X-ray: not checked", GREY))
            motion = self._motion_now()
            if view == VIEW_AP and motion is not None:
                lines.append((f"C-arm turned {motion['angle_deg']:.1f}° between the views", GREY))
        elif view == VIEW_LAT:
            lines = [("Turn the C-arm lateral first.", BLUE if self._suggested() == VIEW_LAT else GREY)]
        elif lat is not None:
            lines = [("Turn back to AP, then load View 1.", BLUE)]
        else:
            lines = [("After View 2: turn back to AP. View 2 is optional.", GREY)]
        msg = self._messages.get(view)
        if msg is not None:
            lines.append(msg)
        button = meta["retake"] if cap is not None else meta["load"]
        active = not self.busy and self._suggested() == view
        return button, lines, active, (not self.busy) and self._camera_ok

    def _refresh_cards(self) -> None:
        for view, card in self.cards.items():
            card.set(*self._card_state(view))

    def _stats_rows(self) -> list:
        rows: list = []
        if not self._pair_mode:
            pv = self._preview
            if pv is None or pv["n_corners"] == 0:
                rows.append(("Board", "not found" if pv is not None else "-"))
            else:
                rows.append(("Board", f"{pv['n_corners']}/{pv['n_total']} corners"))
                if np.isfinite(pv["rms_px"]):
                    rows += [("Reprojection", f"{pv['rms_px']:.2f} px"),
                             ("Distance", f"{pv['distance_mm']:.0f} mm"),
                             ("Obliquity", f"{pv['obliquity_deg']:.0f}°")]
        n = len(self.state.xray_views)
        rows.append(("Views", "View 1 out of date" if self.state.view_ap_out_of_date else f"{n} of 2"))
        motion = self._motion_now()
        if motion is not None:
            rows += [("Turned", f"{motion['angle_deg']:.1f}°"),
                     ("Camera moved", f"{float(np.linalg.norm(motion['t_mm'])):.1f} mm")]
        return rows

    def _refresh_stats(self) -> None:
        rows = self._stats_rows()
        if rows != self._last_rows:
            self._last_rows = rows
            self.set_stats_rows(rows)

    def _instructions(self) -> str:
        if not self._has_prerequisites():
            return "The camera intrinsics are missing (Mode A, step 2)."
        if not self._camera_ok:
            return "No camera image: the board pose needs the live camera."
        if self._phase == POSE:
            return (f"Board pose of the {_VIEWS[self._active]['stance']} stance: keep the C-arm still for 1 s.\n"
                    "The window starts again as long as the C-arm still moves. ESC cancels.")
        if self._phase == WAIT:
            return (f"Send the {_VIEWS[self._active]['stance']} X-ray to CONQUEST on the C-arm; "
                    "it is taken over as soon as it arrives.\n"
                    f"Without the C-arm link: 'From folder (USB)'. ESC cancels "
                    f"(an earlier {_VIEWS[self._active]['name']} stays).")
        ap, lat = self.state.xray_view_ap, self.state.xray_view_lat
        # with the tool loaded, 'Next' goes straight back to Overlay Live (MainWindow)
        nxt = "Overlay Live" if self.state.has_tool else "Tool Tracking"
        if self._lat_first:
            return ("New views: put the board back on the table, where the camera sees it in both stances.\n"
                    "1) Turn the C-arm lateral and retake View 2 (Lat)\n"
                    "2) Turn back to AP and retake View 1 (AP); it stays locked until View 2 is new\n"
                    f"3) Remove the board again and click 'Next' to return to {nxt}.")
        if ap is None and lat is None:
            return ("1) Turn the C-arm lateral, click 'Load X-ray View 2 (Lat)', hold it still for 1 s (board pose)\n"
                    "2) Send the lateral X-ray to CONQUEST on the C-arm\n"
                    "3) Turn back to AP, click 'Load X-ray View 1 (AP)', hold still 1 s, send the AP X-ray\n"
                    f"4) Remove the board, click 'Next' ({nxt}). ESC cancels a running view; the board must not move")
        if self.state.view_ap_out_of_date:
            return ("View 2 (Lat) was retaken, so View 1 (AP) is out of date.\n"
                    "Turn back to AP and retake View 1 (AP); 'Next' is locked until then.")
        if ap is None:
            return ("View 2 (Lat) is set. Turn the C-arm back to AP, click 'Load X-ray View 1 (AP)',\n"
                    "hold still 1 s and send the AP X-ray. 'Next' follows once View 1 (AP) is set.")
        if lat is None:
            return ("View 1 (AP) is set; View 2 (Lat) is optional. For View 2: turn lateral, load View 2,\n"
                    "then turn back and retake View 1 (AP is always last).\n"
                    "Then REMOVE THE BOARD from the table: it is only needed for new views.\n"
                    f"Click 'Next' to continue to {nxt}.")
        return ("Both views are set: check the X-rays. Not satisfied? Retake a view (board pose and X-ray again;\n"
                "retaking View 2 (Lat) needs View 1 (AP) again afterwards, AP is always last).\n"
                "Then REMOVE THE BOARD from the table: it is only needed for new views.\n"
                f"Click 'Next' to continue to {nxt}.")

    def _refresh_instructions(self) -> None:
        text = self._instructions()
        if text != self._last_instr:
            self._last_instr = text
            self.instructions_label.setText(text)
            self._update_viewport_size()

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    def _save_eval(self, cap: XrayViewCapture, source: str) -> None:
        ev = self.state.eval
        if ev is None:
            return
        meta = _VIEWS[cap.view]
        tag = meta["tag"]
        try:
            ev.copy_file(f"xray_{tag}", cap.path, page="xray_acquisition")
            npz = ev.path_for(f"board_{tag}", ".npz")
            cap.pose.save_npz(npz)
            motion = self._motion_now()
            data = {
                "view": cap.view,
                "name": meta["name"],
                "source": source,
                "xray": Path(cap.path).name,
                "xray_path": cap.path,
                "seq": cap.seq,
                "t_acquired": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cap.t_acquired)),
                "phi_deg": cap.phi_deg,
                "board_shift_px": cap.board_shift_px,
                "board_shift_limit_px": self._shift_limit(),
                "T_cw_m": cap.T_cw,
                "board_pose": cap.pose_summary,
                "board_npz": npz.name,
            }
            if motion is not None:
                data["camera_motion_lat_to_ap"] = {
                    "angle_deg": motion["angle_deg"], "axis": motion["axis"], "t_mm": motion["t_mm"]}
            s = cap.pose_summary
            summary = {"view": meta["name"], "source": source,
                       "corners": f"{s['corners_used']}/{s['corners_total']}",
                       "rms_px": round(float(s["rms_px"]), 3),
                       "board_shift_px": round(cap.board_shift_px, 3)
                       if np.isfinite(cap.board_shift_px) else "-"}
            if motion is not None:
                summary["turned_deg"] = round(motion["angle_deg"], 2)
            ev.save_json(f"board_{tag}", jsonable(data), page="xray_acquisition", log=True,
                         summary=summary)
        except Exception as e:
            print(f"[WARN] Could not save the X-ray view: {e}")

    def _log_eval(self, view: Optional[str], numbers: dict) -> None:
        ev = self.state.eval
        if ev is None or view is None:
            return
        try:
            ev.log("xray_acquisition", "-", {"view": _VIEWS[view]["name"], **numbers})
        except Exception as e:
            print(f"[WARN] Could not log: {e}")

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    def _has_prerequisites(self) -> bool:
        return self.state.K_rgb is not None and self.state.dist_rgb is not None

    def _ensure_capture(self) -> None:
        K = np.asarray(self.state.K_rgb, dtype=np.float64)
        d = np.asarray(self.state.dist_rgb, dtype=np.float64)
        key = (K.tobytes(), d.tobytes())
        if self._cap is None or key != self._cap_key:
            self._cap = BoardPoseCapture(K, d, BoardPoseConfig(pitch_mm=25.4))
            self._cap_key = key

    def on_enter(self) -> None:
        self._seq = max((c.seq for c in self.state.xray_views.values()), default=0)
        self._phase = IDLE
        self._active = None
        self._camera_ok = False
        if not self._has_prerequisites():
            self.set_viewport_background(active=False)
            self._refresh()
            return
        self._ensure_capture()
        if self.frame_source is None and self.pipeline is None:
            try:
                self.start_realsense(fps=self.FPS, color_size=(1920, 1080), depth_size=None, align_to=None)
            except Exception as e:
                self.stop_realsense()
                self.set_viewport_background(active=False)
                self._refresh()
                QMessageBox.critical(self, "Camera", f"Could not open RealSense camera.\n\n{e}")
                return
        self._camera_ok = True
        self._warm_up_remap()
        self.start_timer(self.FPS)
        self.set_viewport_background(active=True)
        self._refresh()
        self.setFocus()

    def on_leave(self) -> None:
        if self.busy:
            self.cancel_view()
        self._pending = None
        self._note()
        self._camera_ok = False
        super().on_leave()          # timer, camera, view
        self._frame = None
        self._preview = None
