# overlay/gui/xray_link.py
#
# The C-arm link of the GUI: one DICOM receiver for the whole application
# (overlay.tools.dicom_receiver), started by MainWindow. It listens on the
# Ethernet address of the direct link to the C-arm; while that address does
# not exist (cable out, adapter off) it retries every few seconds, and it
# notices when the address disappears again.
#
# received(path)        - a complete .dcm on disk, emitted in the GUI thread.
#                         MainWindow hands it to the visible page
#                         (accept_received_xray); no page waiting -> it is
#                         only kept on disk (data/xray_received/<day>/).
# status_changed(LinkStatus) - for the waiting overlay and the sidebar badge.
#
# Override for tests without the C-arm (e.g. tests/send_xray_dicom.py):
#   DECAF_DICOM_HOST=127.0.0.1   DECAF_DICOM_PORT=6000   DECAF_DICOM_AET=...
#   DECAF_DICOM_DIR=<folder for the received files>

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from overlay.tools.dicom_receiver import (
    DEFAULT_AET,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DicomReceiver,
    pynetdicom_missing,
)

RECEIVED_DIR = Path(__file__).resolve().parents[1] / "data" / "xray_received"
RETRY_MS = 3000


@dataclass(frozen=True)
class LinkStatus:
    # "unavailable" | "offline" | "busy" | "listening" | "receiving"
    state: str
    headline: str
    detail: str
    address: str

    @property
    def ready(self) -> bool:
        return self.state in ("listening", "receiving")


def _address_present(host: str) -> bool:
    """Is host an address of this machine right now? (Ethernet plugged in,
    static IP set.)"""
    if host in ("", "0.0.0.0", "127.0.0.1", "localhost"):
        return True
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((host, 0))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _port_taken(host: str, port: int) -> bool:
    """Another listener on host:port (e.g. the stand-alone test receiver).
    On Windows pynetdicom's SO_REUSEADDR would let both bind the port and
    the C-arm's images would end up in either program."""
    try:
        with socket.create_connection((host or "127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


class XrayLink(QObject):
    received = Signal(str)
    status_changed = Signal(object)

    # worker thread -> GUI thread
    _stored = Signal(str)
    _activity = Signal(str, str)

    def __init__(self, host: str, port: int, aet: str, out_dir: Path, parent=None):
        super().__init__(parent)
        self.host, self.port, self.aet = host, int(port), aet
        self.out_dir = Path(out_dir)
        self.last_received: tuple[str, datetime] | None = None

        self._receiver = DicomReceiver(
            self.out_dir,
            host=host,
            port=port,
            aet=aet,
            on_stored=self._stored.emit,
            on_activity=self._activity.emit,
        )
        self._stored.connect(self._on_stored, Qt.QueuedConnection)
        self._activity.connect(self._on_activity, Qt.QueuedConnection)

        self._timer = QTimer(self)
        self._timer.setInterval(RETRY_MS)
        self._timer.timeout.connect(self._tick)

        self._status = self._make("offline")

    # ------------------------------------------------------------------

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def status(self) -> LinkStatus:
        return self._status

    def start(self) -> None:
        self._tick()
        if self._status.state != "unavailable":
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._receiver.stop()

    # ------------------------------------------------------------------

    def _make(self, state: str, detail: str | None = None) -> LinkStatus:
        a = self.address
        headline, default = {
            "unavailable": ("C-arm link not available",
                            "pynetdicom is not installed"),
            "offline": ("No Ethernet link to the C-arm",
                        f"{self.host} is not set on this laptop - check the cable"),
            "busy": ("Port already in use",
                     f"{a} - close the stand-alone DICOM receiver"),
            "listening": ("Waiting for the X-ray",
                          "On the C-arm, send the image to CONQUEST"),
            "receiving": ("Receiving X-ray ...",
                          "Transfer from the C-arm in progress"),
        }[state]
        return LinkStatus(state, headline, detail or default, a)

    def _set(self, state: str, detail: str | None = None) -> None:
        st = self._make(state, detail)
        if st != self._status:
            self._status = st
            self.status_changed.emit(st)

    def _tick(self) -> None:
        missing = pynetdicom_missing()
        if missing:
            self._timer.stop()
            self._set("unavailable", missing)
            return

        if self._receiver.running:
            if not _address_present(self.host):
                self._receiver.stop()
                self._set("offline")
            return

        if not _address_present(self.host):
            self._set("offline")
            return
        if _port_taken(self.host, self.port):
            self._set("busy")
            return
        try:
            self._receiver.start()
        except OSError as e:
            self._set("busy" if getattr(e, "winerror", None) == 10048 else "offline",
                      f"{self.address}: {e.strerror or e}")
            return
        self._set("listening")

    def _on_activity(self, kind: str, peer: str) -> None:
        if not self._receiver.running:
            return
        self._set("receiving" if kind == "open" else "listening")

    def _on_stored(self, path: str) -> None:
        self.last_received = (path, datetime.now())
        self.received.emit(path)


_LINK: XrayLink | None = None


def xray_link() -> XrayLink:
    """The application-wide link (created on first use, started by
    MainWindow). Needs a QApplication."""
    global _LINK
    if _LINK is None:
        _LINK = XrayLink(
            host=os.environ.get("DECAF_DICOM_HOST", DEFAULT_HOST),
            port=int(os.environ.get("DECAF_DICOM_PORT", DEFAULT_PORT)),
            aet=os.environ.get("DECAF_DICOM_AET", DEFAULT_AET),
            out_dir=Path(os.environ.get("DECAF_DICOM_DIR", RECEIVED_DIR)),
        )
    return _LINK
