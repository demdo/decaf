"""Record one X-ray per C-arm orientation: the image rotation of the intensifier against
the angle (board fixed to the intensifier; thesis 4.2.6 / 5.2.2: +1.73 deg lateral).

    python -m overlay.tests.record_xray_angles                  # 0 30 60 90 0 deg
    python -m overlay.tests.record_xray_angles 0 30 45 60 90 0  # other angles
    python -m overlay.tests.record_xray_angles --local          # 127.0.0.1 (send_xray_dicom)

Turn the C-arm to the angle shown and take the X-ray; it arrives over the C-arm link
(DICOM, as in the GUI) and is saved as <nn>_<angle>deg.dcm in data/XrayAngles/<time>/,
listed in angles.json. Ctrl+C ends. Do not run it next to the GUI (same port).
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import sys
from datetime import datetime
from pathlib import Path

from overlay.tools.dicom_receiver import (
    DEFAULT_AET, DEFAULT_HOST, DEFAULT_PORT, DicomReceiver, pynetdicom_missing,
)

OUT = Path(__file__).resolve().parents[1] / "data" / "XrayAngles"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("angles", nargs="*", type=float, default=[0.0, 30.0, 60.0, 90.0, 0.0])
    ap.add_argument("--local", action="store_true", help="listen on 127.0.0.1, not the C-arm link")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    a = ap.parse_args()
    missing = pynetdicom_missing()
    if missing:
        print(missing)
        return 1

    session = OUT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True)
    arrived: queue.Queue = queue.Queue()
    host = "127.0.0.1" if a.local else DEFAULT_HOST
    rx = DicomReceiver(session / "received", host=host, port=a.port, aet=DEFAULT_AET,
                       on_stored=arrived.put)
    rx.start()
    print(f"Listening on {host}:{a.port}, saving into {session}")

    log = []
    try:
        for i, angle in enumerate(a.angles, 1):
            print(f"[{i}/{len(a.angles)}] Turn the C-arm to {angle:g} deg and take the X-ray ...", flush=True)
            while True:
                try:
                    path = Path(arrived.get(timeout=0.5))   # short waits: Ctrl+C gets through
                    break
                except queue.Empty:
                    pass
            name = f"{i:02d}_{angle:g}deg.dcm"
            shutil.copy2(path, session / name)
            log.append({"index": i, "angle_deg": angle, "file": name, "received": path.name,
                        "time": datetime.now().isoformat(timespec="seconds")})
            (session / "angles.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
            print(f"    saved {name}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        rx.stop()
    print(f"{len(log)} X-ray(s) in {session}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
