"""Stands in for the C-arm: sends a DICOM file by C-STORE to the GUI's receiver.

    python -m overlay.tests.send_xray_dicom [file.dcm] [--host H] [--port P]

Without a file: the newest image in tests/received_dicom/. Host and port
default to DECAF_DICOM_HOST / DECAF_DICOM_PORT, else the C-arm link
(192.168.6.243:6000). To test without the cable, start the GUI with
DECAF_DICOM_HOST=127.0.0.1 and send with --host 127.0.0.1.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pydicom import dcmread
from pynetdicom import AE
from pynetdicom.sop_class import Verification

from overlay.tools.dicom_receiver import DEFAULT_HOST, DEFAULT_PORT

SAMPLES = Path(__file__).resolve().parent / "received_dicom"


def send(path: Path, host: str, port: int) -> int:
    ds = dcmread(path)
    ae = AE(ae_title="DECAF_TEST_SND")
    ae.add_requested_context(ds.SOPClassUID, ds.file_meta.TransferSyntaxUID)
    ae.add_requested_context(Verification)
    assoc = ae.associate(host, port, ae_title="CONQUEST")
    if not assoc.is_established:
        print(f"No association with {host}:{port}")
        return 1
    try:
        status = assoc.send_c_store(ds)
        code = int(status.Status) if status else -1
        print(f"C-STORE {path.name} -> {host}:{port}: status 0x{code:04X}")
        return 0 if code == 0 else 1
    finally:
        assoc.release()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", type=Path)
    ap.add_argument("--host", default=os.environ.get("DECAF_DICOM_HOST", DEFAULT_HOST))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("DECAF_DICOM_PORT", DEFAULT_PORT)))
    a = ap.parse_args()
    path = a.file
    if path is None:
        files = sorted(SAMPLES.glob("*.dcm"))
        if not files:
            ap.error(f"no .dcm in {SAMPLES}")
        path = files[-1]
    return send(path, a.host, a.port)


if __name__ == "__main__":
    sys.exit(main())
