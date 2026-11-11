# overlay/gui/widgets/widget_live_panels.py
"""The three panels of the study's live page and their layout (user study DHS,
docs/user_study/KONZEPT_UserStudy_DHS.md 4). No camera, no tracker: the page, or
gui/design_live_panels.py, feeds them.

    LivePanelsView      layout 1: video large, X-ray view 1 over X-ray view 2 on the right
                        layout 2: only both X-ray views, larger, side by side
                        read-outs and control cards below, the same in both
    CameraPanel         panel 1, a crop of the camera image in native resolution
    XrayToolPanel       panels 2 and 3, the X-ray with the tool
    ReadoutTile         one large number
    PlaneSideView       tip and axis against Π from the side, beside the two numbers

The geometry comes finished from overlay.tools.tool_marks.MarkBuilder, normally
built in the tracking thread; here it is only drawn. What keeps this cheap, each
measured in the running Qt loop next to a C++ load (design_live_panels --bench):

- Own widgets with paintEvent, no QLabel. Everything is drawn in device pixels with
  the painter's matrix cancelled to the identity (scale 1/dpr), so an image blit is
  a plain copy.
- The video is drawn straight from the camera frame, without copying the crop first;
  only the X-ray field, some 6 % of the frame, is copied and blended (CameraPanel).
- The X-ray stays in one prepared image and only the box around the old and new
  marks is repainted: a full 640 x 640 copy alone costs as much as all marks.
- Ruler numbers and tip glyphs are small pre-rendered pixmaps, a sixth of drawText.
- 1 px ticks use Qt's fast stroker, about 1 us each. Lines of 2 px cost by their
  length, in QPainter and cv2 alike; nothing cheaper looked as good.
- The header's text is painted only when it changes: per frame it cost as much as
  the marks. Image formats per decaf-live-performance: Format_RGB32 is Qt's native
  layout.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QLineF, QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygon,
    QRegion,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from overlay.tools.tool_marks import ORIENT_C_ARM, ORIENT_CAMERA, PanelMap, ToolMarks
from overlay.tools.tracking_overlay import TrackingOverlay
from overlay.tools.warp import LiveOverlay, _detect_xray_fov_mask


def _rgb(bgra) -> QColor:
    return QColor(int(bgra[2]), int(bgra[1]), int(bgra[0]))


GREEN = _rgb(TrackingOverlay.AXIS_COLOR)
MAGENTA = _rgb(TrackingOverlay.XRAY_COLOR)
ORANGE = _rgb(TrackingOverlay.MARKED_COLOR)
DIM_MAGENTA = _rgb(TrackingOverlay.XRAY_MARKED_COLOR)

BG_WINDOW = QColor(8, 10, 13)
BG_HEADER = QColor(18, 22, 28)
BG_TILE = QColor(28, 33, 40)
TEXT_STRONG = QColor(235, 238, 242)
TEXT_WEAK = QColor(150, 160, 172)
TEXT_PLACEHOLDER = QColor(120, 128, 140)
LINE_PLACEHOLDER = QColor(60, 66, 76)

FONT_FAMILY = "Segoe UI"

# LivePanelsView.set_view
VIEW_LIVE = 1
VIEW_XRAY = 2

# LivePanelsView.set_workflow: one X-ray view or two (start dialog, 10.11.)
WORKFLOW_SINGLE = "single"
WORKFLOW_MULTI = "multi"

# Status line colours (Set reference plane, Load X-ray: decided 10.11.)
STATUS_NEUTRAL = QColor(190, 198, 208)
STATUS_GOOD = QColor(90, 210, 120)
STATUS_BAD = QColor(240, 90, 90)
STATUS_WARN = QColor(245, 170, 60)

# Without a fresh pose the marks stay this long, then they go: a stale pose must
# never look like a current one (KONZEPT_UserStudy_DHS.md 4.2).
POSE_HOLD_S = 0.25

# Panel side the placeholder's text was chosen for, device pixels.
REFERENCE_SIDE_PX = 640.0


def _font(px: float, bold: bool = False) -> QFont:
    f = QFont(FONT_FAMILY)
    f.setPixelSize(max(1, int(round(px))))
    f.setBold(bold)
    return f


def overlay_in_roi(overlay: Optional[LiveOverlay], roi) -> Optional[LiveOverlay]:
    """A LiveOverlay of the full frame, cut to a crop: its blend_into then works
    on a buffer of the crop's size. Views of the prepared arrays, no pixel copied."""
    if overlay is None:
        return None
    x0, y0, w, h = roi
    oy0, oy1, ox0, ox1 = overlay.box
    a0, a1 = max(ox0, x0), min(ox1, x0 + w)
    b0, b1 = max(oy0, y0), min(oy1, y0 + h)
    if a0 >= a1 or b0 >= b1:
        return None
    xray = overlay.xray_roi[b0 - oy0:b1 - oy0, a0 - ox0:a1 - ox0]
    return LiveOverlay(
        box=(b0 - y0, b1 - y0, a0 - x0, a1 - x0),
        xray_roi=xray,
        mask_roi=overlay.mask_roi[b0 - oy0:b1 - oy0, a0 - ox0:a1 - ox0],
        _scratch=np.empty_like(xray),
    )


# ============================================================
# Small pre-rendered pixmaps
# ============================================================

class _GlyphCache:
    """Ring + cross, pre-rendered at quarter-pixel positions: 16 pixmaps per
    colour and size at most, each costs ~0.1 ms once and ~4 us per use."""

    MAX_ENTRIES = 512

    def __init__(self) -> None:
        self._pixmaps: dict[tuple, tuple[QPixmap, int]] = {}

    def draw(self, p: QPainter, x: float, y: float, colour: QColor, radius: float,
             reach: float, width: float) -> None:
        ix, iy = math.floor(x), math.floor(y)
        fx, fy = int((x - ix) * 4.0), int((y - iy) * 4.0)
        key = (colour.rgb(), round(radius * 4.0), round(reach * 4.0), round(width * 4.0), fx, fy)
        item = self._pixmaps.get(key)
        if item is None:
            item = self._make(colour, radius, reach, width, fx / 4.0 + 0.125, fy / 4.0 + 0.125)
            if len(self._pixmaps) >= self.MAX_ENTRIES:
                self._pixmaps.clear()
            self._pixmaps[key] = item
        pm, c = item
        p.drawPixmap(ix - c, iy - c, pm)

    @staticmethod
    def _make(colour, radius, reach, width, fx, fy) -> tuple[QPixmap, int]:
        c = int(math.ceil(max(radius, reach) + width)) + 2
        pm = QPixmap(2 * c + 1, 2 * c + 1)
        pm.fill(Qt.transparent)
        q = QPainter(pm)
        q.setRenderHint(QPainter.Antialiasing)
        pen = QPen(colour, width)
        pen.setCapStyle(Qt.FlatCap)
        q.setPen(pen)
        q.setBrush(Qt.NoBrush)
        cx, cy = c + fx, c + fy
        q.drawEllipse(QPointF(cx, cy), radius, radius)
        q.drawLine(QPointF(cx - reach, cy), QPointF(cx + reach, cy))
        q.drawLine(QPointF(cx, cy - reach), QPointF(cx, cy + reach))
        q.end()
        return pm, c


class _LabelCache:
    """Ruler numbers, white with a dark edge so they read on any X-ray."""

    def __init__(self) -> None:
        self._pixmaps: dict[tuple, tuple[QPixmap, float, float]] = {}

    def draw(self, p: QPainter, x: float, y: float, text: str, px: float) -> None:
        key = (text, round(px * 2.0))
        item = self._pixmaps.get(key)
        if item is None:
            item = self._make(text, px)
            self._pixmaps[key] = item
        pm, hw, hh = item
        p.drawPixmap(int(round(x - hw)), int(round(y - hh)), pm)

    @staticmethod
    def _make(text: str, px: float) -> tuple[QPixmap, float, float]:
        font = _font(px, bold=True)
        fm = QFontMetricsF(font)
        edge = max(1.0, px / 6.0)
        w = int(math.ceil(fm.horizontalAdvance(text) + 2.0 * edge + 2.0))
        h = int(math.ceil(fm.height() + 2.0 * edge))
        pm = QPixmap(w, h)
        pm.fill(Qt.transparent)
        q = QPainter(pm)
        q.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addText(edge + 1.0, edge + fm.ascent(), font, text)
        q.setPen(QPen(QColor(0, 0, 0, 230), 2.0 * edge))
        q.drawPath(path)
        q.fillPath(path, Qt.white)
        q.end()
        return pm, w / 2.0, h / 2.0


# ============================================================
# Panels
# ============================================================

class _Panel(QWidget):
    """A header band plus an image area. All sizes of the image area are device
    pixels; self._dpr is what one logical pixel is worth."""

    double_clicked = Signal()

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self._title = title
        self._right_text = ""
        self._right_inset = 0
        self._header_h = 30
        self._ui = 1.0
        self._dpr = 1.0
        # device pixels: top-left and size of the image inside the widget
        self._ox = self._oy = self._nw = self._nh = 0
        # profile: None or callable(stage, seconds)
        self.prof: Optional[Callable[[str, float], None]] = None

    def set_ui_scale(self, ui: float, header_h: int) -> None:
        changed = int(header_h) != self._header_h or float(ui) != self._ui
        self._ui = float(ui)
        self._header_h = int(header_h)
        if changed:
            self._place_image()
            self._image_area_changed()
        self.update()

    def set_right_inset(self, px: int) -> None:
        """Keeps the header's right text clear of what lies over this corner."""
        if px != self._right_inset:
            self._right_inset = px
            self.update(0, 0, self.width(), self._header_h)

    def set_right_text(self, text: str) -> None:
        if text != self._right_text:
            self._right_text = text
            self.update(0, 0, self.width(), self._header_h)

    @property
    def mark_scale(self) -> float:
        """Device pixels per pixel of the mark sizes: marks grow with the fonts and
        the pixel density, not with the size of the panel."""
        return self._dpr * self._ui

    def resizeEvent(self, event) -> None:
        self._dpr = float(self.devicePixelRatioF())
        self._place_image()
        self._image_area_changed()

    def _place_image(self) -> None:
        w = int(round(self.width() * self._dpr))
        h = int(round((self.height() - self._header_h) * self._dpr))
        self._nw, self._nh = max(1, w), max(1, h)
        self._ox, self._oy = 0, int(round(self._header_h * self._dpr))

    def _image_area_changed(self) -> None:
        pass

    def mouseDoubleClickEvent(self, event) -> None:
        self.double_clicked.emit()

    def _paint_header(self, p: QPainter, dirty: QRect) -> None:
        if dirty.top() >= self._header_h:
            return
        w = self.width()
        h = self._header_h
        p.fillRect(0, 0, w, h, BG_HEADER)
        ui = self._ui
        p.setFont(_font(16.0 * ui, bold=True))
        p.setPen(TEXT_STRONG)
        p.drawText(QRect(int(10 * ui), 0, w, h), Qt.AlignLeft | Qt.AlignVCenter, self._title)
        if self._right_text:
            p.setFont(_font(13.0 * ui))
            p.setPen(TEXT_WEAK)
            p.drawText(QRect(0, 0, w - int(10 * ui) - self._right_inset, h),
                       Qt.AlignRight | Qt.AlignVCenter, self._right_text)


class CameraPanel(_Panel):
    """Panel 1: a crop of the camera image at one camera pixel per device pixel, around
    the X-ray field. The marks are drawn on top after the crop, so their lines stay sharp.

    show_frame() draws straight from the frame's memory, without copying the crop: only
    the box of the X-ray field is copied, blended and drawn over it. It paints at once
    (repaint, not update), within the caller's tick, so it also takes a camera view that
    dies with the next poll (RealSenseCameraSource.poll). stays_valid=False lets go of the
    frame right after: a later repaint (an expose) then leaves the picture on screen as
    it is and never reads the old memory. The next frame is at most 33 ms away.
    stays_valid=True: the caller keeps the frame alive (a result's frame copy).
    """

    DASH_WIDTH = float(TrackingOverlay.THICKNESS)
    TIP_RADIUS = float(TrackingOverlay.TIP_RADIUS)
    TIP_REACH = float(TrackingOverlay.TIP_CROSS) / 2.0

    # where the crop goes, in camera pixels: the panel changes size or the X-ray
    roi_changed = Signal()

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        self._frame_wh = (1920, 1080)
        self._centre = (960.0, 540.0)
        self.roi = (0, 0, 640, 640)
        self._frame: Optional[np.ndarray] = None    # what _qimg points into
        self._qimg: Optional[QImage] = None
        self._blank = True                  # nothing of this size on screen yet
        self._overlay_full: Optional[LiveOverlay] = None
        # the X-ray field's box: its own buffer, its own blend (_prepare_box)
        self._box_overlay: Optional[LiveOverlay] = None
        self._box_buf: Optional[np.ndarray] = None
        self._box_img: Optional[QImage] = None
        self._box_src = (0, 0, 0, 0)        # y0, y1, x0, x1 in the frame
        self._box_pos = QPoint(0, 0)        # top left in the crop
        self.alpha = 0.5
        self._marks: Optional[ToolMarks] = None
        self.show_camera_view = True
        self.show_xray_view = True
        self._outline: list = []
        self._clip: Optional[QRegion] = None
        self._glyphs = _GlyphCache()
        self._pens: dict[tuple, QPen] = {}

    # ---------------- GUI thread, once ----------------

    def set_frame_size(self, wh) -> None:
        self._frame_wh = (int(wh[0]), int(wh[1]))
        self._image_area_changed()

    def set_centre(self, x: float, y: float) -> None:
        """The camera pixel at the middle of the crop, fixed per X-ray image: a
        crop following the tool would move the whole picture."""
        self._centre = (float(x), float(y))
        self._image_area_changed()

    def set_overlay(self, overlay: Optional[LiveOverlay]) -> None:
        self._overlay_full = overlay
        self._prepare_box()

    def _prepare_box(self) -> None:
        o = overlay_in_roi(self._overlay_full, self.roi)
        if o is None:
            self._box_overlay = self._box_buf = self._box_img = None
            return
        y0, y1, x0, x1 = o.box
        h, w = y1 - y0, x1 - x0
        self._box_overlay = LiveOverlay(box=(0, h, 0, w), xray_roi=o.xray_roi,
                                        mask_roi=o.mask_roi, _scratch=o._scratch)
        self._box_buf = np.empty((h, w, 4), dtype=np.uint8)
        self._box_img = QImage(self._box_buf.data, w, h, 4 * w, QImage.Format_RGB32)
        rx, ry = self.roi[:2]
        self._box_src = (ry + y0, ry + y1, rx + x0, rx + x1)
        self._box_pos = QPoint(x0, y0)

    def set_field_outline(self, outline: list) -> None:
        """Panel pixels (MarkBuilder.field_outline_on_camera): keeps the tip glyph
        inside the field near its edge."""
        self._outline = outline
        self._clip = None

    def _image_area_changed(self) -> None:
        nw, nh = self._nw, self._nh
        fw, fh = self._frame_wh
        w, h = min(nw, fw), min(nh, fh)
        x0 = int(np.clip(round(self._centre[0] - w / 2.0), 0, fw - w))
        y0 = int(np.clip(round(self._centre[1] - h / 2.0), 0, fh - h))
        # letterbox if the panel is larger than the frame
        self._ox = (nw - w) // 2
        self._oy = int(round(self._header_h * self._dpr)) + (nh - h) // 2
        self.roi = (x0, y0, w, h)
        self._blank = True
        self._prepare_box()
        self._clip = None
        self.roi_changed.emit()
        self.update()

    # ---------------- GUI thread, per frame ----------------

    def show_frame(self, frame_bgra: np.ndarray, *, stays_valid: bool = False) -> None:
        """The whole frame, BGRA; set_marks() comes before this. Returns with the
        picture painted, see the class docstring."""
        prof = self.prof
        t0 = time.perf_counter()
        if self._box_overlay is not None:
            y0, y1, x0, x1 = self._box_src
            np.copyto(self._box_buf, frame_bgra[y0:y1, x0:x1])
            t1 = time.perf_counter()
            self._box_overlay.blend_into(self._box_buf, self.alpha)
            if prof:
                prof("p1 box copy", t1 - t0)
                prof("p1 blend", time.perf_counter() - t1)
        h, w = frame_bgra.shape[:2]
        self._frame = frame_bgra
        self._qimg = QImage(frame_bgra.data, w, h, 4 * w, QImage.Format_RGB32)
        self.repaint(0, self._header_h, self.width(), self.height() - self._header_h)
        if not stays_valid:
            self._frame = self._qimg = None

    def set_marks(self, marks: Optional[ToolMarks]) -> None:
        self._marks = marks

    # ---------------- drawing ----------------

    def _pen(self, colour: QColor, width: float) -> QPen:
        key = (colour.rgb(), width)
        pen = self._pens.get(key)
        if pen is None:
            pen = QPen(colour, width)
            pen.setCapStyle(Qt.FlatCap)
            self._pens[key] = pen
        return pen

    def paintEvent(self, event) -> None:
        prof = self.prof
        t0 = time.perf_counter()
        p = QPainter(self)
        dirty = event.rect()
        self._paint_header(p, dirty)
        r = 1.0 / self._dpr
        p.scale(r, r)
        y0 = int(round(self._header_h * self._dpr))
        if self._qimg is None:
            # No frame at this moment: what is on screen stays (WA_OpaquePaintEvent).
            if self._blank:
                p.fillRect(0, y0, self._nw, self._nh, BG_WINDOW)
                self._blank = False
            p.end()
            return
        if self._blank and (self._ox > 0 or self._oy > y0):
            p.fillRect(0, y0, self._nw, self._nh, BG_WINDOW)
        self._blank = False
        p.translate(self._ox, self._oy)
        t1 = time.perf_counter()
        x0, ry, w, h = self.roi
        p.drawImage(QPoint(0, 0), self._qimg, QRect(x0, ry, w, h))
        if self._box_img is not None:
            p.drawImage(self._box_pos, self._box_img)
        t2 = time.perf_counter()
        m = self._marks
        if m is not None:
            # Marks stay on the picture. A letterboxed panel (taller or wider
            # than the frame, e.g. the laptop at dpr 2) is filled only once
            # (WA_OpaquePaintEvent): a tip glyph beyond the crop would be left
            # behind there and pile up (lab 10.11.).
            p.setClipRect(QRect(0, 0, w, h))
            p.setRenderHint(QPainter.Antialiasing)
            self._paint_marks(p, m)
        t3 = time.perf_counter()
        p.end()
        if prof:
            t4 = time.perf_counter()
            prof("p1 painter", (t1 - t0) + (t4 - t3))
            prof("p1 blit", t2 - t1)
            prof("p1 marks", t3 - t2)

    def _paint_marks(self, p: QPainter, m: ToolMarks) -> None:
        cm = m.camera
        k = self.mark_scale
        width = self.DASH_WIDTH * k
        radius, reach = self.TIP_RADIUS * k, self.TIP_REACH * k
        if self.show_camera_view and cm.green_tip is not None:
            colour = ORANGE if m.marked else GREEN
            t0 = time.perf_counter()
            p.setPen(self._pen(colour, width))
            p.drawLines(cm.green_lines)
            t1 = time.perf_counter()
            x, y = cm.green_tip
            self._glyphs.draw(p, x, y, colour, radius, reach, width)
            if self.prof:
                self.prof("p1 . green lines", t1 - t0)
                self.prof("p1 . green glyph", time.perf_counter() - t1)
        if self.show_xray_view and (cm.magenta_lines or cm.magenta_tip is not None):
            colour = DIM_MAGENTA if m.marked else MAGENTA
            if cm.magenta_lines:
                p.setPen(self._pen(colour, width))
                p.drawLines(cm.magenta_lines)
            if cm.magenta_tip is not None:
                x, y = cm.magenta_tip
                if cm.magenta_tip_near_edge and self._outline:
                    # only this close to the edge: its pixels outside the field go
                    if self._clip is None:
                        self._clip = QRegion(QPolygon(
                            [QPoint(int(round(a)), int(round(b))) for a, b in self._outline]
                        ))
                    p.save()
                    p.setClipRegion(self._clip, Qt.IntersectClip)
                    self._glyphs.draw(p, x, y, colour, radius, reach, width)
                    p.restore()
                else:
                    self._glyphs.draw(p, x, y, colour, radius, reach, width)


class XrayToolPanel(_Panel):
    """Panels 2 and 3: one X-ray with the tool, magenta only. Any two planes, AP and
    lateral are only the first use; they differ in the chain from camera to X-ray frame, which is the
    MarkBuilder's business. Without an image it is a placeholder.

    set_marks() once per pose: repaints only the bounding box of the old and new
    marks. tick() per timer: drops a pose older than POSE_HOLD_S.
    """

    LINE_WIDTH = 2.0
    TICK_WIDTH = 1.0
    MAJOR_TICK_WIDTH = 2.0
    TIP_LINE_WIDTH = 2.0
    LABEL_PX = 12.0
    FIELD_GREY = 190.0

    # the panel's map changed (new size or orientation)
    map_changed = Signal()

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        self._working: Optional[np.ndarray] = None
        self._orientation = ORIENT_CAMERA
        self._gain_lut: Optional[np.ndarray] = None
        self._side = 0
        self._bg: Optional[np.ndarray] = None
        self._qimg: Optional[QImage] = None
        self._placeholder = ("", "")
        self.panel_map: Optional[PanelMap] = None
        self._marks: Optional[ToolMarks] = None
        self._xm = None                 # self._marks.xray
        self._old_bbox: Optional[QRect] = None
        self._lost = False              # tracking was on, the pose is gone
        self._t_acquired: Optional[float] = None
        self._acquired_text = ""
        self._glyphs = _GlyphCache()
        self._labels = _LabelCache()
        self._lost_pm: Optional[QPixmap] = None
        self._pens_for = -1.0
        self._pen_line = self._pen_minor = self._pen_major = QPen()

    # ---------------- GUI thread, once per image ----------------

    def set_placeholder(self, title: str, subtitle: str) -> None:
        self._working = None
        self._bg = self._qimg = None
        self._placeholder = (title, subtitle)
        self._marks = self._xm = None
        self._t_acquired = None
        self.panel_map = None
        self.set_right_text("Placeholder")
        self.update()

    def set_xray(self, working_u8: np.ndarray, t_acquired: Optional[float] = None,
                 acquired_text: str = "") -> None:
        """working_u8: the working image (flipped, undistorted), as the overlay uses
        it. Field mask and contrast are computed here, once."""
        working = np.asarray(working_u8)
        if working.ndim != 2 or working.dtype != np.uint8:
            raise ValueError("working image must be 2D uint8")
        mask = _detect_xray_fov_mask(working)
        inside = working[mask > 0]
        lo, hi = np.percentile(inside, (0.5, 99.5)) if inside.size else (0.0, 255.0)
        # the darkest metal black, the bright field a calm grey: magenta and the
        # white labels must stay readable on it
        span = max(1.0, float(hi - lo))
        lut = np.clip((np.arange(256, dtype=np.float32) - float(lo)) * self.FIELD_GREY / span, 0.0, 255.0)
        self._gain_lut = lut.astype(np.uint8)
        # out of the field black: no bright rim from the undistortion
        self._working = cv2.bitwise_and(working, mask)
        self._t_acquired = t_acquired
        self._acquired_text = acquired_text
        self._prepare_background()
        self.update()

    def set_orientation(self, orientation: str) -> None:
        if orientation not in (ORIENT_CAMERA, ORIENT_C_ARM):
            raise ValueError(f"unknown orientation {orientation!r}")
        if orientation != self._orientation:
            self._orientation = orientation
            self._prepare_background()
            self.update()

    @property
    def orientation(self) -> str:
        return self._orientation

    def _image_area_changed(self) -> None:
        self._prepare_background()

    def _prepare_background(self) -> None:
        """One resample per size and orientation, not per frame."""
        side = min(self._nw, self._nh)
        self._side = side
        self._ox = (self._nw - side) // 2
        self._oy = int(round(self._header_h * self._dpr)) + (self._nh - side) // 2
        if self._working is None or side <= 1:
            self._bg = self._qimg = None
            self.panel_map = None
            return
        h, w = self._working.shape[:2]
        flip = 0 if self._orientation == ORIENT_CAMERA else 1
        img = cv2.flip(self._working, flip)
        interp = cv2.INTER_AREA if side < max(w, h) else cv2.INTER_CUBIC
        img = cv2.resize(img, (side, side), interpolation=interp)
        img = cv2.LUT(img, self._gain_lut)
        self._bg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        self._qimg = QImage(self._bg.data, side, side, 4 * side, QImage.Format_RGB32)
        self.panel_map = PanelMap.of(self._orientation, (w, h), side)
        self._old_bbox = None
        self._lost_pm = None
        self.map_changed.emit()

    # ---------------- GUI thread, per pose ----------------

    def set_marks(self, marks: Optional[ToolMarks]) -> None:
        if marks is self._marks:
            return
        self._marks = marks
        self._xm = None if marks is None else marks.xray
        if self._qimg is None:
            return
        new = self._device_box(self._xm)
        old = self._old_bbox
        self._old_bbox = new
        if new is None and old is None:
            return
        box = new if old is None else (old if new is None else old.united(new))
        r = 1.0 / self._dpr
        self.update(int(math.floor(box.x() * r)), int(math.floor(box.y() * r)),
                    int(math.ceil(box.width() * r)) + 2, int(math.ceil(box.height() * r)) + 2)

    def _device_box(self, xm) -> Optional[QRect]:
        """The marks' box in device pixels of the widget, inside the image."""
        if xm is None:
            return None
        x0, y0, x1, y1 = xm.bbox
        side = self._side
        a, b = max(0, int(x0)), max(0, int(y0))
        c, d = min(side, int(x1) + 1), min(side, int(y1) + 1)
        if a >= c or b >= d:
            return None
        return QRect(a + self._ox, b + self._oy, c - a, d - b)

    def set_lost(self, lost: bool) -> None:
        """Tracking is on, but there is no current pose: a visible statement."""
        if lost != self._lost:
            self._lost = lost
            self.update()

    def tick(self, now_wall: float) -> None:
        """Once per second is enough: the age in the header."""
        if self._t_acquired is None:
            return
        age = max(0.0, now_wall - self._t_acquired)
        text = f"{age:.0f} s ago" if age < 120.0 else f"{age / 60.0:.0f} min ago"
        if self._acquired_text:
            text = f"Acquired {self._acquired_text} · {text}"
        self.set_right_text(text)

    # ---------------- drawing ----------------

    def paintEvent(self, event) -> None:
        prof = self.prof
        t0 = time.perf_counter()
        p = QPainter(self)
        dirty = event.rect()
        self._paint_header(p, dirty)
        d = self._dpr
        r = 1.0 / d
        p.scale(r, r)
        if self._qimg is None:
            self._paint_placeholder(p)
            p.end()
            return

        # the dirty part of the prepared image, a plain copy
        side = self._side
        if self._nw != side or self._nh != side:
            p.fillRect(0, int(round(self._header_h * d)), self._nw, self._nh, BG_WINDOW)
        x0 = max(0, int(math.floor(dirty.x() * d)) - self._ox)
        y0 = max(0, int(math.floor(dirty.y() * d)) - self._oy)
        x1 = min(side, int(math.ceil((dirty.x() + dirty.width()) * d)) - self._ox)
        y1 = min(side, int(math.ceil((dirty.y() + dirty.height()) * d)) - self._oy)
        p.translate(self._ox, self._oy)
        t1 = time.perf_counter()
        if x0 < x1 and y0 < y1:
            p.drawImage(QPoint(x0, y0), self._qimg, QRect(x0, y0, x1 - x0, y1 - y0))
        t2 = time.perf_counter()

        xm = self._xm
        if xm is not None:
            p.setRenderHint(QPainter.Antialiasing)
            self._paint_marks(p, xm, self._marks.marked)
        t3 = time.perf_counter()
        if self._lost:
            self._paint_lost(p)
        p.end()
        if prof:
            t4 = time.perf_counter()
            prof("p2 painter", (t1 - t0) + (t4 - t3))
            prof("p2 blit", t2 - t1)
            prof("p2 marks", t3 - t2)

    def _pens(self, colour: QColor, k: float) -> None:
        key = colour.rgb() * 1000.0 + k
        if key == self._pens_for:
            return
        self._pens_for = key

        def pen(width: float) -> QPen:
            q = QPen(colour, width)
            q.setCapStyle(Qt.FlatCap)
            return q

        self._pen_line = pen(self.LINE_WIDTH * k)
        # exactly 1 px is Qt's fast stroker, ~1 us per short tick
        self._pen_minor = pen(max(1.0, round(self.TICK_WIDTH * k)))
        self._pen_major = pen(self.MAJOR_TICK_WIDTH * k)

    def _paint_marks(self, p: QPainter, xm, marked: bool) -> None:
        k = self.mark_scale
        colour = DIM_MAGENTA if marked else MAGENTA
        self._pens(colour, k)
        t0 = time.perf_counter()
        if xm.lines:
            p.setPen(self._pen_line)
            p.drawLines(xm.lines)
        t1 = time.perf_counter()
        if xm.ticks_minor:
            p.setPen(self._pen_minor)
            p.drawLines(xm.ticks_minor)
        if xm.ticks_major:
            p.setPen(self._pen_major)
            p.drawLines(xm.ticks_major)
        t2 = time.perf_counter()
        px = self.LABEL_PX * k
        draw_label = self._labels.draw
        for x, y, text in xm.labels:
            draw_label(p, x, y, text, px)
        t3 = time.perf_counter()
        if xm.tip is not None:
            self._glyphs.draw(p, xm.tip[0], xm.tip[1], colour, xm.tip_radius,
                              xm.tip_radius + 3.0 * k, self.TIP_LINE_WIDTH * k)
        if self.prof:
            self.prof("p2 . lines", t1 - t0)
            self.prof("p2 . ticks", t2 - t1)
            self.prof("p2 . labels", t3 - t2)
            self.prof("p2 . glyph", time.perf_counter() - t3)

    def _paint_lost(self, p: QPainter) -> None:
        k = self.mark_scale
        if self._lost_pm is None:
            self._lost_pm = _LabelCache._make("no tracking", 18.0 * k)[0]
        pm = self._lost_pm
        p.drawPixmap((self._side - pm.width()) // 2, int(12 * k), pm)

    def _paint_placeholder(self, p: QPainter) -> None:
        y0 = int(round(self._header_h * self._dpr))
        p.fillRect(0, y0, self._nw, self._nh, QColor(12, 14, 18))
        side = min(self._nw, self._nh)
        cx, cy = self._nw / 2.0, y0 + self._nh / 2.0
        k = side / REFERENCE_SIDE_PX
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(LINE_PLACEHOLDER, max(1.0, 2.0 * k)))
        p.setBrush(Qt.NoBrush)
        rr = side * 0.47
        p.drawEllipse(QPointF(cx, cy), rr, rr)
        title, sub = self._placeholder
        p.setFont(_font(26.0 * k, bold=True))
        p.setPen(TEXT_PLACEHOLDER)
        p.drawText(QRectF(0, cy - 44.0 * k, self._nw, 40.0 * k), Qt.AlignCenter, title)
        p.setFont(_font(15.0 * k))
        p.setPen(QColor(100, 108, 118))
        p.drawText(QRectF(0, cy, self._nw, 26.0 * k), Qt.AlignCenter, sub)


# ============================================================
# Read-outs
# ============================================================

def readout_int(value: Optional[float], signed: bool = False) -> Optional[str]:
    """A read-out as a whole number, half away from zero (2.5 -> 3, -2.5 -> -3).
    signed: a plus in front of positive values, so the side is clear; never "-0".
    None stays None, the tile shows a dash then."""
    if value is None or not math.isfinite(value):
        return None
    n = int(math.floor(abs(value) + 0.5))
    if value < 0.0 and n:
        return f"-{n}"
    return f"+{n}" if signed and n else str(n)


class ReadoutTile(QWidget):
    """A number one can read from 1-2 m. Painted from small pre-rendered pixmaps: the
    background with the label once per size, each character once. A new value repaints
    only its own line, and an unchanged one nothing at all. No geometry, no text layout."""

    MIN_HEIGHT = 70.0
    LABEL_PX = 13.0
    VALUE_PX = 42.0
    GAP_PX = 7.0
    UNIT_RATIO = 0.48
    WIDEST_VALUE = "-188"

    def __init__(self, label: str, unit: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._label, self._unit = label, unit
        self._value = "–"
        self._ui = 1.0
        self.prof: Optional[Callable[[str, float], None]] = None
        self._bg: Optional[QPixmap] = None
        self._chars: dict[str, tuple[QPixmap, int]] = {}
        self._unit_pm: Optional[QPixmap] = None
        self._x = self._y = self._gap = self._unit_y = 0
        self._dpr = 1.0
        self._value_rect = QRect()      # logical pixels, what a new value repaints

    def set_ui_scale(self, ui: float) -> None:
        self._ui = ui
        self.setMinimumHeight(int(round(self.MIN_HEIGHT * ui)))
        self._bg = None
        self.update()

    def set_value(self, text: Optional[str]) -> None:
        text = "–" if text is None else text
        if text == self._value:
            return
        self._value = text
        if self._bg is None:
            self.update()
        else:
            self.update(self._value_rect)

    def set_label(self, label: str) -> None:
        if label != self._label:
            self._label = label
            self._bg = None
            self.update()

    def resizeEvent(self, event) -> None:
        self._bg = None

    @staticmethod
    def _render(text: str, font: QFont, colour: QColor) -> tuple[QPixmap, int]:
        fm = QFontMetricsF(font)
        w = max(1, int(math.ceil(fm.horizontalAdvance(text))) + 2)
        h = max(1, int(math.ceil(fm.height())))
        pm = QPixmap(w, h)
        pm.fill(Qt.transparent)
        q = QPainter(pm)
        q.setRenderHint(QPainter.TextAntialiasing)
        q.setFont(font)
        q.setPen(colour)
        q.drawText(QPointF(0.0, fm.ascent()), text)
        q.end()
        return pm, int(round(fm.horizontalAdvance(text)))

    def _prepare(self) -> None:
        """Once per size and label, in device pixels."""
        self._dpr = d = float(self.devicePixelRatioF())
        k = self._ui * d
        w, h = max(1, int(round(self.width() * d))), max(1, int(round(self.height() * d)))
        bg = QPixmap(w, h)
        bg.fill(BG_WINDOW)
        q = QPainter(bg)
        q.setRenderHint(QPainter.Antialiasing)
        q.setPen(Qt.NoPen)
        q.setBrush(BG_TILE)
        q.drawRoundedRect(QRectF(0.0, 0.0, w, h), 8.0 * k, 8.0 * k)
        label_font = _font(self.LABEL_PX * k)
        label_fm = QFontMetricsF(label_font)
        self._x = int(12.0 * k)
        q.setFont(label_font)
        q.setPen(TEXT_WEAK)
        q.drawText(QPointF(self._x, int(8.0 * k) + label_fm.ascent()), self._label)
        q.end()
        self._bg = bg

        top = int(8.0 * k + label_fm.height() + 2.0 * k)
        value_px = max(1.0, min(self.VALUE_PX * k, (h - top - 4.0 * k) * 0.95))
        # the widest number that can come, with its unit, still fits
        room = w - 2.0 * self._x
        widest = (QFontMetricsF(_font(value_px, bold=True)).horizontalAdvance(self.WIDEST_VALUE)
                  + self.GAP_PX * k
                  + QFontMetricsF(_font(value_px * self.UNIT_RATIO)).horizontalAdvance(self._unit))
        if widest > room:
            value_px = max(1.0, value_px * room / widest)
        value_font = _font(value_px, bold=True)
        fm = QFontMetricsF(value_font)
        # the digits centred in what the label leaves
        base = top + ((h - top) + fm.capHeight()) / 2.0
        self._y = int(round(base - fm.ascent()))
        self._gap = int(round(self.GAP_PX * k))
        self._chars = {
            c: self._render(c, value_font, TEXT_STRONG) for c in "0123456789.,+-"
        }
        self._chars["–"] = self._render("–", value_font, TEXT_WEAK)
        unit_font = _font(value_px * self.UNIT_RATIO)
        self._unit_pm = self._render(self._unit, unit_font, TEXT_WEAK)[0] if self._unit else None
        unit_fm = QFontMetricsF(unit_font)
        self._unit_y = int(round(base - unit_fm.ascent()))

        # from the top of the digits down, up to the right edge
        self._value_rect = QRect(0, int(math.floor(top / d)), self.width(), self.height())

    def paintEvent(self, event) -> None:
        t0 = time.perf_counter()
        if self._bg is None:
            self._prepare()
        p = QPainter(self)
        d = self._dpr
        p.scale(1.0 / d, 1.0 / d)
        r = event.rect()
        x0, y0 = int(math.floor(r.x() * d)), int(math.floor(r.y() * d))
        x1, y1 = int(math.ceil((r.x() + r.width()) * d)), int(math.ceil((r.y() + r.height()) * d))
        p.drawPixmap(QPoint(x0, y0), self._bg, QRect(x0, y0, x1 - x0, y1 - y0))
        x, y = self._x, self._y
        chars = self._chars
        for c in self._value:
            item = chars.get(c)
            if item is not None:
                p.drawPixmap(x, y, item[0])
                x += item[1]
        if self._unit_pm is not None:
            p.drawPixmap(x + self._gap, self._unit_y, self._unit_pm)
        p.end()
        if self.prof:
            self.prof("tile paint", time.perf_counter() - t0)


def _ray_length(x: float, y: float, dx: float, dy: float,
                x0: float, x1: float, y0: float, y1: float) -> float:
    """How far the ray (x, y) + t (dx, dy), unit direction, runs inside [x0, x1] x [y0, y1]
    from a start inside it."""
    t = math.inf
    if dx > 1e-12:
        t = min(t, (x1 - x) / dx)
    elif dx < -1e-12:
        t = min(t, (x0 - x) / dx)
    if dy > 1e-12:
        t = min(t, (y1 - y) / dy)
    elif dy < -1e-12:
        t = min(t, (y0 - y) / dy)
    return max(0.0, t) if math.isfinite(t) else 0.0


class PlaneSideView(QWidget):
    """Tip and drill axis against the reference plane Π, seen from the side: the cut through
    the drill axis perpendicular to Π. No numbers, the two tiles beside it carry them.

        Π       the horizontal line, fixed
        up      away from the source, the side on which "Tip to plane Π" is positive (+)
        tip     the ring, as in the X-ray panels; its height compressed, linear near Π:
                    y = REACH * h / (|h| + H0_MM)  of the half height
                a tip 1 mm off moves some 5 px, one 8 mm off is half way, one 100 mm off
                is still inside: it never leaves the window, whatever the distance
        solid   ahead of the tip to the window edge: the trajectory, at its true angle to Π
        dashed  behind the tip: the drill. Solid trajectory, dashed drill, as in panel 1
                and the X-ray panels

    The angle is drawn true, the height compressed, so where the solid line crosses Π is
    a hint, not a distance. Fast: tile, label, Π and the signs are one pixmap per size; a
    new pose draws that pixmap, a few lines and the pre-rendered ring, and only if the tip moved by a
    pixel or the axis turned by half a degree. Measured with design_live_panels --bench
    ("plane view paint")."""

    MIN_HEIGHT = 70.0
    LABEL_PX = 13.0
    H0_MM = 8.0         # the tip is half way to the edge at this distance
    REACH = 0.86        # of the half height: the most a tip can take
    TIP_X = 0.60        # of the content width, from the left: the trajectory gets the larger part
    STEP_PX = 1.0       # a new pose repaints only if the tip moved this much ...
    STEP_DEG = 0.5      # ... or the axis turned this much
    PLANE = QColor(120, 150, 190)

    def __init__(self, label: str = "Tip and axis against Π", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._label = label
        self._ui = 1.0
        self._bg: Optional[QPixmap] = None
        self._tip: Optional[tuple] = None    # ring pixmap, its half size in device px
        self._lay: Optional[tuple] = None    # logical px: left, right, top, bottom, y of Π, half height
        self._off: Optional[float] = None
        self._rise: Optional[float] = None
        self._key: Optional[tuple] = None
        self.prof: Optional[Callable[[str, float], None]] = None

    def set_ui_scale(self, ui: float) -> None:
        self._ui = ui
        self.setMinimumHeight(int(round(self.MIN_HEIGHT * ui)))
        self._bg = self._lay = None
        self._key = None if self._off is None else self._key_of(self._off, self._rise)
        self.update()

    def resizeEvent(self, event) -> None:
        self._bg = self._lay = None
        self._key = None if self._off is None else self._key_of(self._off, self._rise)

    @classmethod
    def height_fraction(cls, off_mm: float) -> float:
        """Height of the tip over Π, as a fraction of the half height: -REACH .. REACH."""
        return cls.REACH * off_mm / (abs(off_mm) + cls.H0_MM)

    def _layout(self) -> tuple:
        if self._lay is None:
            k = self._ui
            label_h = QFontMetricsF(_font(self.LABEL_PX * k)).height()
            top = 8.0 * k + label_h + 6.0 * k
            bottom = self.height() - 8.0 * k
            left = 14.0 * k
            right = self.width() - 30.0 * k     # room for Π and the signs
            self._lay = (left, right, top, bottom, 0.5 * (top + bottom), max(1.0, 0.5 * (bottom - top)))
        return self._lay

    def _key_of(self, off_mm: float, rise_deg: float) -> tuple:
        half = self._layout()[5]
        return (int(round(self.height_fraction(off_mm) * half / self.STEP_PX)),
                int(round(rise_deg / self.STEP_DEG)))

    def set_state(self, off_mm: Optional[float], rise_deg: Optional[float]) -> None:
        """Per pose; None, None: no pose, only Π stays."""
        if off_mm is None or rise_deg is None or not (math.isfinite(off_mm) and math.isfinite(rise_deg)):
            key = None
        else:
            key = self._key_of(off_mm, rise_deg)
        if key == self._key:
            return
        self._key = key
        self._off, self._rise = (None, None) if key is None else (off_mm, rise_deg)
        self.update()

    def _prepare(self) -> None:
        """Once per size: tile, label, Π, signs."""
        d = float(self.devicePixelRatioF())
        k = self._ui
        left, right, top, bottom, yc, _ = self._layout()
        bg = QPixmap(max(1, int(round(self.width() * d))), max(1, int(round(self.height() * d))))
        bg.setDevicePixelRatio(d)
        bg.fill(BG_WINDOW)
        q = QPainter(bg)
        q.setRenderHint(QPainter.Antialiasing)
        q.setPen(Qt.NoPen)
        q.setBrush(BG_TILE)
        q.drawRoundedRect(QRectF(0.0, 0.0, self.width(), self.height()), 8.0 * k, 8.0 * k)
        label_font = _font(self.LABEL_PX * k)
        q.setFont(label_font)
        q.setPen(TEXT_WEAK)
        q.drawText(QPointF(12.0 * k, 8.0 * k + QFontMetricsF(label_font).ascent()), self._label)
        q.setPen(QPen(self.PLANE, 2.0 * k))
        q.drawLine(QPointF(left - 4.0 * k, yc), QPointF(right + 4.0 * k, yc))
        sign_font = _font(self.LABEL_PX * 1.15 * k, bold=True)
        fm = QFontMetricsF(sign_font)
        q.setFont(sign_font)
        q.setPen(self.PLANE)
        x = right + 10.0 * k
        q.drawText(QPointF(x, yc + 0.5 * fm.capHeight()), "Π")
        q.setPen(TEXT_WEAK)
        q.drawText(QPointF(x, top + fm.capHeight()), "+")
        q.drawText(QPointF(x, bottom), "−")
        q.end()
        self._bg = bg
        # the tip, ring with a cross as in the X-ray panels, rendered once: a blit per pose
        # instead of an antialiased ellipse and two lines
        r = 6.0 * k
        half_px = int(math.ceil((r + 2.0 * k) * d))
        tip = QPixmap(2 * half_px, 2 * half_px)
        tip.setDevicePixelRatio(d)
        tip.fill(Qt.transparent)
        q = QPainter(tip)
        q.setRenderHint(QPainter.Antialiasing)
        q.setPen(QPen(MAGENTA, 2.0 * k))
        q.setBrush(BG_TILE)
        c = half_px / d
        q.drawEllipse(QPointF(c, c), r, r)
        q.drawLine(QPointF(c - r, c), QPointF(c + r, c))
        q.drawLine(QPointF(c, c - r), QPointF(c, c + r))
        q.end()
        self._tip = (tip, half_px)

    def paintEvent(self, event) -> None:
        t0 = time.perf_counter()
        if self._bg is None:
            self._prepare()
        p = QPainter(self)
        r = event.rect()
        d = self._bg.devicePixelRatio()
        p.drawPixmap(r, self._bg, QRect(int(r.x() * d), int(r.y() * d),
                                        int(math.ceil(r.width() * d)), int(math.ceil(r.height() * d))))
        if self._key is not None:
            left, right, top, bottom, yc, half = self._layout()
            k = self._ui
            w = right - left
            xt = left + self.TIP_X * w
            yt = yc - self.height_fraction(self._off) * half
            a = math.radians(self._rise)
            c, s = math.cos(a), math.sin(a)
            x0, x1 = left - 6.0 * k, right + 6.0 * k
            y0, y1 = top - 2.0 * k, bottom + 2.0 * k
            p.setClipRect(QRectF(x0, y0, x1 - x0, y1 - y0))
            # behind the tip, dashed: the drill, as in panel 1 and the X-ray panels. The dashes
            # as plain short lines, only as far as the window, not antialiased (no visible
            # difference): Qt's dash stroker costs more than the rest together.
            behind = _ray_length(xt, yt, c, -s, x0, x1, y0, y1)
            dash = 4.0 * k
            n = int(behind // (2.0 * dash)) + 1
            dashes = [QLineF(xt + c * i * 2.0 * dash, yt - s * i * 2.0 * dash,
                             xt + c * (i * 2.0 + 1.0) * dash, yt - s * (i * 2.0 + 1.0) * dash)
                      for i in range(1, n)]
            if dashes:
                pen = QPen(MAGENTA, 2.0 * k)
                pen.setCapStyle(Qt.FlatCap)         # dash and gap the same length
                p.setPen(pen)
                p.drawLines(dashes)
            # ahead of the tip, solid to the window edge: the trajectory, antialiased, the line
            # the eye reads the angle and the crossing with Π from
            p.setRenderHint(QPainter.Antialiasing)
            pen = QPen(MAGENTA, 3.0 * k)
            pen.setCapStyle(Qt.FlatCap)
            p.setPen(pen)
            ahead = _ray_length(xt, yt, -c, s, x0, x1, y0, y1)
            p.drawLine(QPointF(xt, yt), QPointF(xt - ahead * c, yt + ahead * s))
            # the tip, pre-rendered, placed on the device pixel grid
            tip, half_px = self._tip
            p.drawPixmap(QPointF((round(xt * d) - half_px) / d, (round(yt * d) - half_px) / d), tip)
        p.end()
        if self.prof:
            self.prof("plane view paint", time.perf_counter() - t0)


class StatusText(QWidget):
    """A line of text that changes rarely: a pre-rendered pixmap per text and colour,
    and unlike a QLabel no layout pass when it changes. setText(text, colour): a colour
    for this text only (the status line's red / green / orange states); without one the
    colour given at construction."""

    MAX_TEXTS = 64

    def __init__(self, text: str, colour: QColor, background: QColor, px: float = 14.0,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._text = text
        self._default_colour = colour
        self._colour, self._background = colour, background
        self._px = px
        self._pixmaps: dict[tuple, QPixmap] = {}
        self.set_ui_scale(1.0)

    def set_ui_scale(self, ui: float) -> None:
        self._ui = ui
        self._pixmaps.clear()
        self.setFixedHeight(int(math.ceil(self._px * ui * 1.5)))
        self.update()

    def text(self) -> str:
        return self._text

    def colour(self) -> QColor:
        return self._colour

    def setText(self, text: str, colour: Optional[QColor] = None) -> None:
        colour = self._default_colour if colour is None else colour
        if text != self._text or colour.rgb() != self._colour.rgb():
            self._text = text
            self._colour = colour
            self.update()

    def paintEvent(self, event) -> None:
        d = float(self.devicePixelRatioF())
        key = (self._text, self._colour.rgb())
        pm = self._pixmaps.get(key)
        if pm is None:
            if len(self._pixmaps) >= self.MAX_TEXTS:
                self._pixmaps.clear()
            pm = ReadoutTile._render(self._text, _font(self._px * self._ui * d), self._colour)[0]
            self._pixmaps[key] = pm
        p = QPainter(self)
        p.scale(1.0 / d, 1.0 / d)
        p.fillRect(0, 0, int(math.ceil(self.width() * d)), int(math.ceil(self.height() * d)), self._background)
        p.drawPixmap(0, 0, pm)
        p.end()


# ============================================================
# View selector
# ============================================================

class _ViewButton(QAbstractButton):
    """One view as a small picture of its panels; no number, the panels have those."""

    def __init__(self, rects, tooltip: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._rects = rects         # (x, y, w, h) as fractions of the inner area
        self.setCheckable(True)
        self.setToolTip(tooltip)
        self.setFocusPolicy(Qt.NoFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        on = self.isChecked()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(46, 112, 200) if on else
                   QColor(54, 62, 72) if self.underMouse() else QColor(42, 48, 56))
        p.drawRoundedRect(r, 3.0, 3.0)
        pad = max(3.0, r.height() * 0.2)
        iw, ih = r.width() - 2.0 * pad, r.height() - 2.0 * pad
        colour = QColor(240, 242, 245) if on else TEXT_WEAK
        for x, y, w, h in self._rects:
            p.fillRect(QRectF(pad + x * iw, pad + y * ih, w * iw, h * ih), colour)
        p.end()


# ============================================================
# Layout
# ============================================================

class LivePanelsView(QWidget):
    """The panels and what lies below them, in one of two views, chosen top right:

        VIEW_LIVE   the video large on the left, X-ray view 1 top right, view 2 bottom right
        VIEW_XRAY   only both X-ray views, larger, side by side

    The area below the images is the same in both, so no control moves. It holds the
    read-outs and the control cards: the widgets of today's rail by the same names
    (slider_alpha, slider_dx, btn_track, chk_show_camera_view, chk_show_xray_view,
    chk_marker_exposure, btn_save, btn_plane), so OverlayLivePage keeps its signals;
    nothing lies on top of an image any more. footer: a slot for more, below all.

    set_workflow (decided 10.11.):

        WORKFLOW_MULTI    the above; card [Load X-ray View 2 (Lat) | Load X-ray View 1 (AP)]
                          [Set reference plane | Snapshot] [From folder (USB)]
        WORKFLOW_SINGLE   no view 2 and no selector: X-ray view 1 square and as high as
                          the window allows on the right, the video beside it, the area
                          below over the full width; card [Load X-ray | From folder (USB)]
                          [Set reference plane | Snapshot]

    Double-click on a panel fills the window with it.
    """

    view_changed = Signal(int)

    REFERENCE_WIDTH = 1920.0
    MIN_UI = 0.72
    HEADER_H = 30.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("livePanelsView")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._ui = -1.0
        self._maximized: Optional[QWidget] = None
        self._view = VIEW_LIVE
        self._workflow = WORKFLOW_MULTI

        self.panel_live = CameraPanel("X-ray and Video Overlay", self)
        self.panel_view1 = XrayToolPanel("X-ray View 1 + Tool Tracking", self)
        self.panel_view2 = XrayToolPanel("X-ray View 2 + Tool Tracking", self)
        self.panel_view2.set_placeholder("X-RAY VIEW 2", "not acquired yet")
        self._panels = (self.panel_live, self.panel_view1, self.panel_view2)
        for panel in self._panels:
            panel.double_clicked.connect(lambda p=panel: self.toggle_maximized(p))

        self.bottom = QWidget(self)
        self.bottom.setObjectName("livePanelsBottom")
        self._build_bottom()
        # its minimum height is known only once the style sheet has reached its widgets
        self.bottom.installEventFilter(self)

        self.view_buttons = {
            VIEW_LIVE: _ViewButton(((0.0, 0.0, 0.62, 1.0), (0.70, 0.0, 0.30, 0.44), (0.70, 0.56, 0.30, 0.44)),
                                   "Video large, both X-ray views on the right (key 1)", self),
            VIEW_XRAY: _ViewButton(((0.0, 0.0, 0.46, 1.0), (0.54, 0.0, 0.46, 1.0)),
                                   "Only both X-ray views, larger (key 2)", self),
        }
        self._view_group = QButtonGroup(self)
        self._view_group.setExclusive(True)
        for view, button in self.view_buttons.items():
            self._view_group.addButton(button)
            button.clicked.connect(lambda _checked=False, v=view: self.set_view(v))
        self.view_buttons[self._view].setChecked(True)

    # ---------------- bottom area ----------------

    def _build_bottom(self) -> None:
        self.tile_off_plane = ReadoutTile("Tip to plane Π", "mm")
        self.tile_inclination = ReadoutTile("Inclination to plane Π", "°")
        # the insertion depth is read off the ruler in the X-ray panels (user 10.11.)
        self._tiles = (self.tile_off_plane, self.tile_inclination)
        # both numbers drawn: tip and axis against Π from the side (supervisor 10.11.)
        self.plane_view = PlaneSideView("Tip and axis against Π")

        tiles = QGridLayout()
        tiles.setContentsMargins(0, 0, 0, 0)
        tiles.setSpacing(8)
        for i, tile in enumerate(self._tiles):
            tiles.addWidget(tile, 0, i)
            tiles.setColumnStretch(i, 2)
        tiles.addWidget(self.plane_view, 0, len(self._tiles))
        tiles.setColumnStretch(len(self._tiles), 3)

        self.lbl_status = StatusText("Tracking off", STATUS_NEUTRAL, BG_WINDOW)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(4)
        left.addLayout(tiles, 1)
        left.addWidget(self.lbl_status)

        # ---------------- the widgets of today's rail ----------------
        self.btn_track = self._button("Stop tracking", primary=True)
        self.chk_marker_exposure = self._check("Marker exposure", False)
        self.lbl_track_status = StatusText("Pose: –", TEXT_WEAK, QColor(20, 24, 30), px=13.0)

        self.lbl_alpha_value = QLabel("0.50")
        self.slider_alpha = self._slider(0, 100, 50)
        self.lbl_dx_value = QLabel("–")
        self.slider_dx = self._slider(-10, 10, 0)
        self.chk_show_camera_view = self._check("Camera view (green)", True)
        self.chk_show_xray_view = self._check("X-ray view (magenta)", True)

        # Set zero and the ruler switch are gone (user 10.11.): the ruler counts from the
        # mask entry. Which buttons show and where: set_workflow.
        self.btn_plane = self._button("Set reference plane")
        self.btn_save = self._button("Snapshot")
        self.btn_load_xray = self._button("Load X-ray View 1 (AP)")
        self.btn_lateral = self._button("Load X-ray View 2 (Lat)")
        self.btn_usb = self._button("From folder (USB)")

        track = QVBoxLayout()
        track.setSpacing(6)
        track.addWidget(self.btn_track)
        track.addWidget(self.chk_marker_exposure)
        track.addWidget(self.lbl_track_status)
        track.addStretch(1)

        overlay = QVBoxLayout()
        overlay.setSpacing(2)
        overlay.addLayout(self._slider_head("Camera weight α", self.lbl_alpha_value))
        overlay.addWidget(self.slider_alpha)
        overlay.addLayout(self._slider_head("Plane Π, d<sub>x</sub>", self.lbl_dx_value))
        overlay.addWidget(self.slider_dx)
        overlay.addWidget(self.chk_show_camera_view)
        overlay.addWidget(self.chk_show_xray_view)
        overlay.addStretch(1)

        acts = QGridLayout()
        acts.setSpacing(6)
        self._acts = acts
        self._arrange_buttons()

        cards = QHBoxLayout()
        cards.setContentsMargins(0, 0, 0, 0)
        cards.setSpacing(8)
        cards.addWidget(self._card("Tracking", track), 3)
        cards.addWidget(self._card("Overlay", overlay), 3)
        cards.addWidget(self._card("Acquisition and measurement", acts), 6)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)
        row.addLayout(left, 41)
        row.addLayout(cards, 59)

        self.footer = QVBoxLayout()
        self.footer.setContentsMargins(0, 0, 0, 0)
        self.footer.setSpacing(0)

        outer = QVBoxLayout(self.bottom)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        outer.addLayout(row, 1)
        outer.addLayout(self.footer, 0)

    def _arrange_buttons(self) -> None:
        """The card "Acquisition and measurement" of the workflow (decided 10.11.)."""
        g = self._acts
        for b in (self.btn_plane, self.btn_save, self.btn_load_xray, self.btn_lateral, self.btn_usb):
            g.removeWidget(b)
        for r in range(4):
            g.setRowStretch(r, 0)
        if self._workflow == WORKFLOW_SINGLE:
            self.btn_load_xray.setText("Load X-ray")
            rows = [[(self.btn_load_xray, 1), (self.btn_usb, 1)],
                    [(self.btn_plane, 1), (self.btn_save, 1)]]
            self.btn_lateral.hide()
        else:
            self.btn_load_xray.setText("Load X-ray View 1 (AP)")
            rows = [[(self.btn_lateral, 1), (self.btn_load_xray, 1)],
                    [(self.btn_plane, 1), (self.btn_save, 1)],
                    [(self.btn_usb, 2)]]
            self.btn_lateral.show()
        for r, row in enumerate(rows):
            c = 0
            for b, span in row:
                g.addWidget(b, r, c, 1, span)
                c += span
        g.setRowStretch(len(rows), 1)

    @property
    def workflow(self) -> str:
        return self._workflow

    def set_workflow(self, workflow: str) -> None:
        """WORKFLOW_SINGLE or WORKFLOW_MULTI: panels, selector and card (class docstring)."""
        if workflow not in (WORKFLOW_SINGLE, WORKFLOW_MULTI):
            raise ValueError(f"unknown workflow {workflow!r}")
        if workflow == self._workflow:
            return
        self._workflow = workflow
        self._arrange_buttons()
        single = workflow == WORKFLOW_SINGLE
        for button in self.view_buttons.values():
            button.setVisible(not single)
        if single:
            self._view = VIEW_LIVE
            self.view_buttons[VIEW_LIVE].setChecked(True)
        self._maximized = None
        self._show_panels()
        self._relayout()

    def _button(self, text: str, primary: bool = False) -> QPushButton:
        b = QPushButton(text)
        b.setObjectName("livePanelsPrimary" if primary else "livePanelsButton")
        b.setFocusPolicy(Qt.NoFocus)
        return b

    def _check(self, text: str, checked: bool) -> QCheckBox:
        c = QCheckBox(text)
        c.setChecked(checked)
        c.setFocusPolicy(Qt.NoFocus)
        return c

    def _slider(self, lo: int, hi: int, value: int) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(value)
        s.setFocusPolicy(Qt.NoFocus)
        return s

    def _slider_head(self, text: str, value_label: QLabel) -> QHBoxLayout:
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        name = QLabel(text)
        name.setObjectName("livePanelsWeak")
        h.addWidget(name)
        h.addStretch(1)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(value_label)
        return h

    def _card(self, title: str, body) -> QFrame:
        f = QFrame()
        f.setObjectName("livePanelsCard")
        f.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(f)
        lay.setSpacing(6)
        name = QLabel(title)
        name.setObjectName("livePanelsCardTitle")
        lay.addWidget(name)
        lay.addLayout(body, 1)
        self.__dict__.setdefault("_card_layouts", []).append(lay)
        return f

    def _apply_style(self, ui: float) -> None:
        def s(v: float) -> int:
            return max(1, int(round(v * ui)))

        self.setStyleSheet(f"""
            QWidget#livePanelsBottom {{ background: rgb(8, 10, 13); }}
            QLabel {{ background: transparent; color: rgb(220, 224, 230); font-size: {s(13)}px; }}
            QLabel#livePanelsWeak {{ color: rgb(150, 160, 172); }}
            QLabel#livePanelsCardTitle {{ color: rgb(200, 206, 214); font-size: {s(14)}px; font-weight: 700; }}
            QFrame#livePanelsCard {{
                background: rgb(20, 24, 30); border: 1px solid rgb(46, 52, 60); border-radius: {s(8)}px;
            }}
            QPushButton#livePanelsButton, QPushButton#livePanelsPrimary {{
                color: rgb(240, 242, 245); border: none; border-radius: {s(6)}px;
                font-size: {s(14)}px; padding: {s(7)}px {s(4)}px;
            }}
            QPushButton#livePanelsButton {{ background: rgb(42, 48, 56); }}
            QPushButton#livePanelsButton:hover {{ background: rgb(54, 62, 72); }}
            QPushButton#livePanelsButton:disabled {{ color: rgb(110, 118, 128); background: rgb(30, 34, 40); }}
            QPushButton#livePanelsPrimary {{ background: rgb(46, 112, 200); }}
            QPushButton#livePanelsPrimary:hover {{ background: rgb(37, 99, 235); }}
            QPushButton#livePanelsPrimary:disabled {{ color: rgb(150, 160, 172); background: rgb(36, 56, 84); }}
            QCheckBox {{ color: rgb(210, 214, 220); font-size: {s(13)}px; spacing: {s(8)}px; background: transparent; }}
            QCheckBox::indicator {{
                width: {s(14)}px; height: {s(14)}px; border: 1px solid rgb(150, 160, 172);
                border-radius: 2px; background: transparent;
            }}
            QCheckBox::indicator:checked {{ background: rgb(46, 112, 200); }}
            QSlider {{ background: transparent; min-height: {s(20)}px; }}
            QSlider::groove:horizontal {{ height: {s(3)}px; background: rgb(70, 78, 88); border-radius: 1px; }}
            QSlider::sub-page:horizontal {{ background: rgb(90, 150, 230); border-radius: 1px; }}
            QSlider::handle:horizontal {{
                width: {s(14)}px; height: {s(14)}px; margin: -{s(6)}px 0; border-radius: {s(7)}px;
                background: rgb(90, 150, 230);
            }}
        """)
        margin = s(10)
        for lay in self.__dict__.get("_card_layouts", []):
            lay.setContentsMargins(margin, s(8), margin, s(8))

    # ---------------- geometry ----------------

    @property
    def view(self) -> int:
        return self._view

    def set_view(self, view: int) -> None:
        """One resampling of the X-rays, once; per frame nothing changes. In VIEW_XRAY
        the page need not compose the video at all."""
        if view not in self.view_buttons:
            raise ValueError(f"unknown view {view!r}")
        self.view_buttons[view].setChecked(True)
        if view == self._view and self._maximized is None:
            return
        self._view = view
        self._maximized = None
        self._show_panels()
        self._relayout()
        self.view_changed.emit(view)

    def toggle_maximized(self, panel: QWidget) -> None:
        self._maximized = None if self._maximized is panel else panel
        self._show_panels()
        self._relayout()

    def _show_panels(self) -> None:
        single = self._workflow == WORKFLOW_SINGLE
        for panel in self._panels:
            if single and panel is self.panel_view2:
                panel.setVisible(False)
            elif self._maximized is not None:
                panel.setVisible(panel is self._maximized)
            else:
                panel.setVisible(self._view == VIEW_LIVE or panel is not self.panel_live)
        self.bottom.setVisible(self._maximized is None)

    def paintEvent(self, event) -> None:
        # the header band runs through, also beside the X-rays of VIEW_XRAY
        p = QPainter(self)
        p.fillRect(self.rect(), BG_WINDOW)
        p.fillRect(0, 0, self.width(), int(round(self.HEADER_H * max(self._ui, self.MIN_UI))), BG_HEADER)
        p.end()

    def eventFilter(self, obj, event) -> bool:
        if obj is self.bottom and event.type() == QEvent.LayoutRequest:
            self._relayout()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:
        self._relayout()

    def _relayout(self) -> None:
        w, h = self.width(), self.height()
        ui = max(self.MIN_UI, min(1.0, w / self.REFERENCE_WIDTH))
        if ui != self._ui:
            self._ui = ui
            self._apply_style(ui)
            for tile in self._tiles:
                tile.set_ui_scale(ui)
            self.plane_view.set_ui_scale(ui)
            self.lbl_status.set_ui_scale(ui)
            self.lbl_track_status.set_ui_scale(ui)
        header = int(round(self.HEADER_H * ui))
        for panel in self._panels:
            panel.set_ui_scale(ui, header)
        self._place_panels(w, h, header)

        # the selector, top right, over whatever panel lies there (none in single)
        bh = max(12, header - int(round(8 * ui)))
        bw = int(round(bh * 1.5))
        gap = int(round(4 * ui))
        x = w - gap - int(round(6 * ui))
        y = (header - bh) // 2
        if self._workflow == WORKFLOW_SINGLE:
            x = w
        else:
            for view in sorted(self.view_buttons, reverse=True):
                button = self.view_buttons[view]
                x -= bw
                button.setGeometry(x, y, bw, bh)
                button.raise_()
                x -= gap
        for panel in self._panels:
            g = panel.geometry()
            panel.set_right_inset(w - x if g.top() < header and g.right() >= x else 0)

    def _place_panels(self, w: int, h: int, header: int) -> None:
        if self._maximized is not None:
            self._maximized.setGeometry(0, 0, w, h)
            return
        bottom_h = self.bottom.minimumSizeHint().height()
        if self._workflow == WORKFLOW_SINGLE:
            # X-ray view 1 as high as the window allows ("big", decided 10.11.)
            large = max(64, h - header - bottom_h)
            side = min(large, w // 2)
            left = w - side
            self.panel_view1.setGeometry(left, 0, side, large + header)
            self.panel_live.setGeometry(0, 0, left, large + header)
            self.bottom.setGeometry(0, large + header, w, h - large - header)
            return
        # the right column of VIEW_LIVE: two X-ray squares over the full height
        column = max(64, min(h // 2 - header, w // 3))
        left = w - column
        large = max(64, h - header - self.bottom.minimumSizeHint().height())

        # the same area below in both views
        self.bottom.setGeometry(0, large + header, left, h - large - header)
        if self._view == VIEW_LIVE:
            self.panel_view1.setGeometry(left, 0, column, column + header)
            self.panel_view2.setGeometry(left, column + header, column, column + header)
            self.panel_live.setGeometry(0, 0, left, large + header)
        else:
            side = max(64, min(w // 2, large))
            x0 = (w - 2 * side) // 2
            self.panel_view1.setGeometry(x0, 0, side, side + header)
            self.panel_view2.setGeometry(x0 + side, 0, side, side + header)
