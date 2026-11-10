# overlay/tools/dicom_receiver.py
#
# DICOM Storage SCP for the X-rays the C-arm pushes over Ethernet (C-STORE
# over TCP). On the C-arm the laptop is the stored network node "CONQUEST"
# (192.168.6.243, port 6000; the C-arm calls as CADISOR20392 from
# 192.168.6.244, 2026-11-10). Every image is first written to disk
# (exclusive .part, flushed, then renamed to .dcm, never overwritten) and
# only then reported through on_stored(path) - the pages load it exactly
# like a file from the USB stick (overlay.tools.xray_io.read_xray_gray_u8).
#
# Qt-free; the GUI side (retry, status, routing to the page) is
# overlay/gui/xray_link.py. Stand-alone test receiver with the same storage
# logic: tests/dicom_storage_scp_standalone.py.

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

LOG = logging.getLogger("overlay.dicom_receiver")

DEFAULT_HOST = "192.168.6.243"
DEFAULT_PORT = 6000
DEFAULT_AET = "DECAF_SCP"


def pynetdicom_missing() -> str | None:
    """None if the receiver can run, else the reason."""
    try:
        import pynetdicom  # noqa: F401
    except ImportError as e:
        return f"pynetdicom missing ({e})"
    return None


class DicomReceiver:
    """C-STORE SCP writing every received image to out_dir/<YYYYMMDD>/.

    on_stored(path): called from a pynetdicom worker thread once the file
        is complete on disk (images with pixel data only).
    on_activity(kind, peer): "open" when an association is accepted,
        "closed" when it is released or aborted (also worker threads).

    Called and calling AE titles are not checked; the server only listens
    on the address of the direct link to the C-arm. Only uncompressed
    transfer syntaxes are offered, so every stored file decodes without
    extra pixel handlers.
    """

    def __init__(
        self,
        out_dir: Path | str,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        aet: str = DEFAULT_AET,
        on_stored: Callable[[str], None] | None = None,
        on_activity: Callable[[str, str], None] | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.host = host
        self.port = int(port)
        self.aet = aet
        self.on_stored = on_stored
        self.on_activity = on_activity
        self._ae = None
        self._server = None

    @property
    def running(self) -> bool:
        return self._server is not None

    # ------------------------------------------------------------------

    def start(self) -> None:
        """Raises OSError if the address cannot be bound."""
        if self._server is not None:
            return
        from pynetdicom import AE, AllStoragePresentationContexts, evt
        from pynetdicom import DEFAULT_TRANSFER_SYNTAXES
        from pynetdicom.sop_class import Verification

        ae = AE(ae_title=self.aet)
        ae.require_called_aet = False
        ae.require_calling_aet = []
        ae.maximum_associations = 5
        ae.acse_timeout = 30
        ae.dimse_timeout = 120
        ae.network_timeout = 120
        for cx in AllStoragePresentationContexts:
            ae.add_supported_context(cx.abstract_syntax, DEFAULT_TRANSFER_SYNTAXES)
        ae.add_supported_context(Verification)

        handlers = [
            (evt.EVT_C_STORE, self._handle_store),
            (evt.EVT_C_ECHO, lambda event: 0x0000),
            (evt.EVT_ACCEPTED, lambda event: self._activity("open", event)),
            (evt.EVT_RELEASED, lambda event: self._activity("closed", event)),
            (evt.EVT_ABORTED, lambda event: self._activity("closed", event)),
        ]
        try:
            server = ae.start_server(
                (self.host, self.port), block=False, evt_handlers=handlers
            )
        except Exception:
            ae.shutdown()
            raise
        self._ae, self._server = ae, server
        LOG.info("listening on %s:%s (AET %s), files to %s",
                 self.host, self.port, self.aet, self.out_dir)

    def stop(self) -> None:
        server, ae = self._server, self._ae
        self._server = self._ae = None
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                LOG.warning("server shutdown failed", exc_info=True)
        if ae is not None:
            try:
                ae.shutdown()
            except Exception:
                LOG.warning("AE shutdown failed", exc_info=True)

    # ------------------------------------------------------------------

    @staticmethod
    def _peer(event) -> str:
        r = event.assoc.requestor
        return f"{r.address}:{r.port}"

    def _activity(self, kind: str, event) -> None:
        if self.on_activity is not None:
            try:
                self.on_activity(kind, self._peer(event))
            except Exception:
                LOG.exception("on_activity failed")

    def _handle_store(self, event) -> int:
        try:
            ds = event.dataset
            ds.file_meta = event.file_meta
            if (str(ds.SOPClassUID) != str(event.request.AffectedSOPClassUID)
                    or str(ds.SOPInstanceUID) != str(event.request.AffectedSOPInstanceUID)):
                LOG.error("C-STORE UID mismatch from %s", self._peer(event))
                return 0xA900
        except Exception:
            LOG.exception("C-STORE invalid dataset from %s", self._peer(event))
            return 0xC210

        # File name from the local clock only: never from patient data or
        # sender-controlled UIDs. The same image sent twice gives two files.
        now = datetime.now()
        day = self.out_dir / now.strftime("%Y%m%d")
        name = now.strftime("%Y%m%d_%H%M%S_%f_") + uuid4().hex[:8]
        pending = day / (name + ".part")
        final = day / (name + ".dcm")
        created = False
        try:
            day.mkdir(parents=True, exist_ok=True)
            with pending.open("xb") as f:
                created = True
                ds.save_as(f, enforce_file_format=True)
                f.flush()
                os.fsync(f.fileno())
            if os.name == "nt":
                os.rename(pending, final)      # fails if final exists
                created = False
            else:
                os.link(pending, final)        # never replaces final
        except OSError:
            LOG.exception("C-STORE storage failure")
            return 0xA700
        except Exception:
            LOG.exception("C-STORE serialisation failure")
            return 0xC210
        finally:
            if created:
                try:
                    pending.unlink()
                except OSError:
                    LOG.warning("could not remove %s", pending, exc_info=True)

        LOG.info("C-STORE %s from %s -> %s", ds.SOPClassUID, self._peer(event), final)
        if "PixelData" in ds and self.on_stored is not None:
            try:
                self.on_stored(str(final))
            except Exception:
                LOG.exception("on_stored failed")
        return 0x0000
