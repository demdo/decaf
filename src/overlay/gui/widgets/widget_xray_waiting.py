# overlay/gui/widgets/widget_xray_waiting.py
#
# Placeholder over an X-ray viewport while a page waits for an image from
# the C-arm (overlay/gui/xray_link.py): a pulsing dot, one headline, one
# line of detail. It follows the link status by itself and covers its
# parent widget; mouse events go through to the widgets below.
# compact=True: only a small pill (dot + headline) at the bottom, for
# viewports that show live video underneath.

from __future__ import annotations

from PySide6.QtCore import QEvent, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from overlay.gui.xray_link import LinkStatus, xray_link

_COLORS = {
    "listening": QColor(47, 128, 237),     # app blue
    "receiving": QColor(64, 192, 87),
    "offline": QColor(250, 176, 5),
    "busy": QColor(250, 176, 5),
    "unavailable": QColor(134, 142, 150),
}
_PULSE_MS = 1600
_FRAME_MS = 40


class XrayWaitingOverlay(QWidget):
    def __init__(self, viewport: QWidget, compact: bool = False):
        super().__init__(viewport)
        self._compact = compact
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        viewport.installEventFilter(self)
        self.setGeometry(viewport.rect())

        self._phase = 0.0
        self._anim = QTimer(self)
        self._anim.setInterval(_FRAME_MS)
        self._anim.timeout.connect(self._step)

        link = xray_link()
        self._status: LinkStatus = link.status
        link.status_changed.connect(self._on_status)
        self._sync_anim()

    # ------------------------------------------------------------------

    def eventFilter(self, obj, ev):
        if obj is self.parent() and ev.type() == QEvent.Resize:
            self.setGeometry(obj.rect())
        return False

    def showEvent(self, ev):
        super().showEvent(ev)
        self.raise_()
        self._sync_anim()

    def hideEvent(self, ev):
        super().hideEvent(ev)
        self._anim.stop()

    def _on_status(self, st: LinkStatus) -> None:
        self._status = st
        self._sync_anim()
        self.update()

    def _sync_anim(self) -> None:
        if self.isVisible() and self._status.ready:
            if not self._anim.isActive():
                self._anim.start()
        else:
            self._anim.stop()
            self._phase = 0.0

    def _step(self) -> None:
        self._phase = (self._phase + _FRAME_MS / _PULSE_MS) % 1.0
        self.update()

    # ------------------------------------------------------------------

    def paintEvent(self, ev):
        if self._compact:
            self._paint_pill()
            return
        st = self._status
        col = _COLORS.get(st.state, _COLORS["unavailable"])

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        s = min(w, h)
        cx, cy = w / 2.0, h * 0.42
        r0 = max(6.0, s * 0.028)

        # pulse: two rings expanding and fading, half a period apart
        if st.ready:
            for k in (0.0, 0.5):
                t = (self._phase + k) % 1.0
                ring = QColor(col)
                ring.setAlphaF(0.55 * (1.0 - t))
                p.setPen(QPen(ring, max(1.5, s * 0.004)))
                p.setBrush(Qt.NoBrush)
                r = r0 * (1.0 + 3.2 * t)
                p.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawEllipse(QRectF(cx - r0, cy - r0, 2 * r0, 2 * r0))

        # text
        top = cy + r0 * 4.6
        f = QFont(self.font())
        f.setPixelSize(max(12, int(s * 0.036)))
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.setPen(QColor(241, 243, 245))
        line_h = f.pixelSize() * 1.6
        p.drawText(QRectF(0, top, w, line_h), Qt.AlignHCenter | Qt.AlignVCenter, st.headline)

        f2 = QFont(self.font())
        f2.setPixelSize(max(10, int(s * 0.025)))
        p.setFont(f2)
        p.setPen(QColor(173, 181, 189))
        p.drawText(
            QRectF(w * 0.08, top + line_h, w * 0.84, f2.pixelSize() * 3.2),
            Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
            st.detail,
        )

        # address, quiet, at the bottom
        f3 = QFont(self.font())
        f3.setPixelSize(max(9, int(s * 0.021)))
        p.setFont(f3)
        p.setPen(QColor(108, 117, 125))
        p.drawText(
            QRectF(0, h - f3.pixelSize() * 3.0, w, f3.pixelSize() * 2.0),
            Qt.AlignHCenter | Qt.AlignVCenter,
            st.address,
        )
        p.end()

    def _paint_pill(self) -> None:
        st = self._status
        col = _COLORS.get(st.state, _COLORS["unavailable"])
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        f = QFont(self.font())
        f.setPixelSize(max(11, int(min(self.width(), self.height()) * 0.032)))
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        text_w = p.fontMetrics().horizontalAdvance(st.headline)
        h = f.pixelSize() * 2.4
        r0 = f.pixelSize() * 0.38
        pad = h * 0.5
        w = pad + 2 * r0 + pad * 0.6 + text_w + pad
        x = (self.width() - w) / 2.0
        y = self.height() - h - h * 0.6

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(20, 20, 20, 200))
        p.drawRoundedRect(QRectF(x, y, w, h), h / 2, h / 2)

        cx, cy = x + pad + r0, y + h / 2
        if st.ready:
            ring = QColor(col)
            ring.setAlphaF(0.6 * (1.0 - self._phase))
            p.setPen(QPen(ring, 1.5))
            p.setBrush(Qt.NoBrush)
            r = r0 * (1.0 + 1.6 * self._phase)
            p.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawEllipse(QRectF(cx - r0, cy - r0, 2 * r0, 2 * r0))

        p.setPen(QColor(241, 243, 245))
        p.drawText(QRectF(cx + r0 + pad * 0.6, y, text_w + 2, h),
                   Qt.AlignLeft | Qt.AlignVCenter, st.headline)
        p.end()
