"""Isolated DICOM Storage SCP. Run with Python from the decaf environment.

Defaults: 192.168.6.243:6000; received_dicom/ beside this script.
No dependency on the overlay application. Stop with Ctrl+C.
"""

import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from uuid import uuid4

LOG = logging.getLogger("dicom_scp")


def dependencies():
    try:
        import pydicom
        from pynetdicom import AE, AllStoragePresentationContexts, evt
        from pynetdicom import ALL_TRANSFER_SYNTAXES
        from pynetdicom.sop_class import Verification
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}\nActivate decaf, then run:\n"
            'python -m pip install "pynetdicom>=3,<4" "pydicom>=3,<4"'
        ) from exc
    if int(pydicom.__version__.split('.')[0]) < 3:
        raise SystemExit('Requires pydicom >= 3: python -m pip install "pydicom>=3,<4"')
    return AE, AllStoragePresentationContexts, evt, ALL_TRANSFER_SYNTAXES, Verification


def peer(event):
    requestor = event.assoc.requestor
    return f"{requestor.address}:{requestor.port}"


def log_connection(event):
    LOG.info("%s peer=%s", event.event.name, peer(event))


def log_request(event):
    request = event.assoc.requestor.primitive
    LOG.info("ASSOCIATION requested peer=%s calling_AET=%r called_AET=%r",
             peer(event), request.calling_ae_title, request.called_ae_title)
    for context in event.assoc.requestor.requested_contexts:
        LOG.info("PROPOSED SOP=%s transfer_syntaxes=%s",
                 context.abstract_syntax, list(context.transfer_syntax))


def log_accepted(event):
    log_connection(event)
    for context in event.assoc.accepted_contexts:
        LOG.info("ACCEPTED SOP=%s transfer_syntax=%s",
                 context.abstract_syntax, list(context.transfer_syntax))
    for context in event.assoc.rejected_contexts:
        LOG.warning("REJECTED context SOP=%s result=%s",
                    context.abstract_syntax, context.result)


def handle_echo(event):
    LOG.info("C-ECHO peer=%s", peer(event))
    return 0x0000


def handle_store(event, output):
    """Write exclusively, flush, then publish atomically without replacement."""
    try:
        dataset = event.dataset
        dataset.file_meta = event.file_meta
        if (str(dataset.SOPClassUID) != str(event.request.AffectedSOPClassUID)
                or str(dataset.SOPInstanceUID) != str(event.request.AffectedSOPInstanceUID)):
            LOG.error("C-STORE UID mismatch peer=%s", peer(event))
            return 0xA900
    except Exception:
        LOG.exception("C-STORE invalid dataset peer=%s", peer(event))
        return 0xC210

    # Never use patient data or sender-controlled UIDs as filesystem paths.
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid4().hex
    pending = output / (name + ".part")
    final = output / (name + ".dcm")
    created = False
    try:
        with pending.open("xb") as stream:
            created = True
            dataset.save_as(stream, enforce_file_format=True)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows rename fails if final exists; POSIX rename would overwrite.
        if os.name == "nt":
            os.rename(pending, final)
            created = False
        else:
            os.link(pending, final)
    except OSError:
        LOG.exception("C-STORE storage failure peer=%s", peer(event))
        return 0xA700
    except Exception:
        LOG.exception("C-STORE serialization failure peer=%s", peer(event))
        return 0xC210
    finally:
        if created:
            try:
                pending.unlink()
            except OSError:
                LOG.warning("Could not remove temporary file %s", pending, exc_info=True)
    LOG.info("C-STORE SUCCESS peer=%s SOP=%s instance=%s file=%s",
             peer(event), dataset.SOPClassUID, dataset.SOPInstanceUID, final)
    return 0x0000


def make_ae(output, aet):
    AE, contexts, evt, syntaxes, verification = dependencies()
    ae = AE(ae_title=aet)
    ae.require_called_aet = False
    ae.require_calling_aet = []
    ae.maximum_associations = 5
    ae.acse_timeout = 30
    ae.dimse_timeout = 120
    ae.network_timeout = 120
    for context in contexts:
        ae.add_supported_context(context.abstract_syntax, syntaxes)
    ae.add_supported_context(verification)
    handlers = [(evt.EVT_C_STORE, handle_store, [output]),
                (evt.EVT_C_ECHO, handle_echo),
                (evt.EVT_REQUESTED, log_request),
                (evt.EVT_ACCEPTED, log_accepted)]
    for event in (evt.EVT_CONN_OPEN, evt.EVT_CONN_CLOSE, evt.EVT_REJECTED,
                  evt.EVT_ABORTED, evt.EVT_RELEASED):
        handlers.append((event, log_connection))
    return ae, handlers


def self_test():
    """Synthetic C-ECHO and duplicate C-STORE over localhost, no C-arm needed."""
    from pydicom import Dataset, dcmread
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    from pynetdicom import AE
    from pynetdicom.sop_class import SecondaryCaptureImageStorage, Verification

    with tempfile.TemporaryDirectory(prefix="dicom_scp_test_") as folder:
        output = Path(folder)
        ae, handlers = make_ae(output, "DECAF_TEST")
        server = ae.start_server(("127.0.0.1", 0), block=False, evt_handlers=handlers)
        association = None
        try:
            client = AE(ae_title="TEST_SENDER")
            client.add_requested_context(Verification)
            client.add_requested_context(SecondaryCaptureImageStorage, ExplicitVRLittleEndian)
            association = client.associate("127.0.0.1", server.server_address[1],
                                           ae_title="UNKNOWN_TARGET")
            assert association.is_established, "Association failed"
            assert association.send_c_echo().Status == 0x0000
            ds = Dataset()
            ds.SOPClassUID = SecondaryCaptureImageStorage
            ds.SOPInstanceUID = generate_uid()
            ds.PatientName = "SYNTHETIC^TEST"
            ds.file_meta = FileMetaDataset()
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.Rows = ds.Columns = 1
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = ds.BitsStored = 8
            ds.HighBit = 7
            ds.PixelRepresentation = 0
            ds.PixelData = b"\x2a\x00"
            for _ in range(2):
                assert association.send_c_store(ds).Status == 0x0000
            files = list(output.glob("*.dcm"))
            assert len(files) == 2, "Duplicate reception overwrote a file"
            for file in files:
                saved = dcmread(file)
                assert saved.SOPInstanceUID == ds.SOPInstanceUID
                assert saved.PixelData == ds.PixelData
                assert saved.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
            assert not list(output.glob("*.part"))
            association.release()
            LOG.info("SELF-TEST PASSED: unknown AE title, C-ECHO, two separate readable files")
        finally:
            if association is not None and association.is_established:
                association.abort()
            server.shutdown()
            ae.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.6.243")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--aet", default="DECAF_SCP", help="Local AE title; called titles are not restricted")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "received_dicom")
    parser.add_argument("--debug", action="store_true", help="Detailed pynetdicom negotiation log")
    parser.add_argument("--self-test", action="store_true", help="Synthetic localhost test, then exit")
    args = parser.parse_args()
    dependencies()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("pynetdicom").setLevel(logging.DEBUG if args.debug else logging.INFO)
    if args.self_test:
        self_test()
        return 0
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    logfile = output / ("scp_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex + ".log")
    handler = logging.FileHandler(logfile, mode="x", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    ae, handlers = make_ae(output, args.aet)
    server = None
    try:
        server = ae.start_server((args.host, args.port), block=False, evt_handlers=handlers)
        LOG.info("LISTENING %s:%s local_AET=%r; accepting unknown calling/called AE titles",
                 args.host, args.port, args.aet)
        LOG.info("Output=%s log=%s; stop with Ctrl+C", output, logfile)
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        LOG.info("Stopping")
    except OSError:
        LOG.exception("Cannot listen on %s:%s: check local IP and other listeners", args.host, args.port)
        return 1
    finally:
        if server is not None:
            server.shutdown()
        ae.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
