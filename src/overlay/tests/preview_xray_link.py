"""Preview of the C-arm link on step 4 (X-ray marker selection).

    python -m overlay.tests.preview_xray_link            # window, link on 127.0.0.1
    python -m overlay.tests.preview_xray_link --carm     # window, real link 192.168.6.243
    python -m overlay.tests.preview_xray_link --shots DIR   # offscreen check + PNGs

Opens the full GUI directly on step 4 (steps 1-3 skipped). In window mode
send an image from a second console, as the C-arm would:

    python -m overlay.tests.send_xray_dicom --host 127.0.0.1

--shots: (a) receiver alone: a sample from tests/received_dicom/ sent by
C-STORE arrives on disk and decodes to the same 8-bit image as the sample
itself; (b) GUI step 4: waiting -> image shown after C-STORE, the same
exposure sent again is dropped, a new one replaces it; (c) no Ethernet;
(d) Overlay Preview with the Bone_New capture as live frame: X-ray ->
frame frozen + overlay. Screenshots into DIR.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(app, cond, timeout_s: float = 10.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def _window(app):
    from overlay.gui.evaluation import StartupChoice
    from overlay.gui.main import QSS, MainWindow

    app.setStyleSheet(QSS)
    win = MainWindow(StartupChoice())
    win._switch_to_page(win.IDX_MARKERS)
    return win


def check_receiver(samples: list[Path]) -> None:
    import numpy as np

    from overlay.tests.send_xray_dicom import send
    from overlay.tools.dicom_receiver import DicomReceiver
    from overlay.tools.xray_io import read_xray_gray_u8

    with tempfile.TemporaryDirectory(prefix="xray_link_") as d:
        got: list[str] = []
        port = _free_port()
        rx = DicomReceiver(d, host="127.0.0.1", port=port, on_stored=got.append)
        rx.start()
        try:
            assert send(samples[0], "127.0.0.1", port) == 0
            assert send(samples[0], "127.0.0.1", port) == 0
        finally:
            rx.stop()
        assert len(got) == 2 and got[0] != got[1], got
        assert not list(Path(d).rglob("*.part"))
        want = read_xray_gray_u8(str(samples[0]))
        for p in got:
            assert np.array_equal(read_xray_gray_u8(p), want), p
    print("(a) receiver: 2 x C-STORE -> 2 files, decode identical to the sample  OK")


def check_gui(samples: list[Path], shots: Path) -> None:
    from PySide6.QtWidgets import QApplication

    from overlay.tests.send_xray_dicom import send
    from overlay.tools.xray_io import read_xray_gray_u8, xray_pixel_md5

    port = int(os.environ["DECAF_DICOM_PORT"])
    app = QApplication.instance() or QApplication(sys.argv)
    win = _window(app)
    win.resize(1400, 820)
    win.show()
    page = win.page_markers
    link = win.xray_link

    assert _wait(app, lambda: link.status.state == "listening"), link.status
    assert page.wait_overlay.isVisible()
    _wait(app, lambda: False, 0.6)                       # let the pulse run
    win.grab().save(str(shots / "xray_link_1_waiting.png"))

    def push(path: Path) -> None:
        threading.Thread(target=send, args=(path, "127.0.0.1", port), daemon=True).start()

    push(samples[0])
    assert _wait(app, lambda: page.state.xray_image is not None), "image not shown"
    md5_first = xray_pixel_md5(page.state.xray_image)
    assert not page.wait_overlay.isVisible()
    _wait(app, lambda: False, 0.3)
    win.grab().save(str(shots / "xray_link_2_received.png"))
    print(f"(b) GUI: C-STORE -> image on step 4 ({Path(page.state.xray_image_path).name})  OK")

    n = len(list(link.out_dir.rglob("*.dcm")))
    push(samples[0])                                      # same exposure again
    assert _wait(app, lambda: len(list(link.out_dir.rglob("*.dcm"))) > n)
    _wait(app, lambda: False, 0.5)
    assert xray_pixel_md5(page.state.xray_image) == md5_first
    print("    same exposure sent twice: dropped silently  OK")

    other = next((s for s in samples[1:]
                  if xray_pixel_md5(read_xray_gray_u8(str(s))) != md5_first), None)
    if other is not None:
        push(other)
        assert _wait(app, lambda: xray_pixel_md5(page.state.xray_image) != md5_first)
        print("    new exposure before the markers are confirmed: replaces the image  OK")

    check_preview(app, win, shots, push)
    win.close()
    app.processEvents()


def check_preview(app, win, shots: Path, push) -> None:
    """(d) Overlay Preview with the Bone_New capture: X-ray during the live
    view -> frame frozen, overlay shown."""
    import numpy as np

    from overlay.tools.xray_io import xray_pixel_md5

    cap_dir = Path(__file__).resolve().parents[1] / "data" / "Evaluation" / "Evaluation_Integrated_System_Bone_New"
    cap_npz = cap_dir / "capture_overlay.npz"
    if not cap_npz.is_file():
        print(f"(d) skipped: {cap_npz} missing")
        return
    cap = np.load(cap_npz, allow_pickle=True)
    st = win.state
    st.K_rgb, st.dist_rgb, st.K_xray = cap["K_rgb"], cap["dist_rgb"], cap["K_xray"]
    st.T_xc, st.T_cx, st.d_x = cap["T_xc"], cap["T_cx"], float(cap["d_x"])
    st.tip_uv_c = cap["tip_uv_c"]
    xray_src = Path(str(cap["xray_image_path"]))
    if not xray_src.is_file():
        xray_src = cap_dir / "2_xray.IMA"

    page = win.page_overlay_preview
    # the page without its camera: the capture's frame stands in for the live view
    win.current_step = win.IDX_OVERLAY_PREVIEW
    win.pages.setCurrentIndex(win.IDX_OVERLAY_PREVIEW)
    win.update_ui()
    page._reset_page_state()
    page._mode = "preview"
    page.set_viewport_background(active=True)
    page._live_color = np.ascontiguousarray(cap["snapshot_rgb_bgr"])
    page._update_controls()
    page.update_view()
    assert page.wait_overlay.isVisible()
    _wait(app, lambda: False, 0.6)
    win.grab().save(str(shots / "xray_link_4_preview_waiting.png"))

    push(xray_src)
    assert _wait(app, lambda: page._overlay_done), "preview overlay not shown"
    assert page._snapshot_taken and not page.wait_overlay.isVisible()
    md5 = xray_pixel_md5(page._xray_gray_u8)
    _wait(app, lambda: False, 0.3)
    win.grab().save(str(shots / "xray_link_5_preview_overlay.png"))
    print(f"(d) Overlay Preview: C-STORE during live view -> frozen + overlay ({xray_src.name})  OK")

    push(xray_src)
    _wait(app, lambda: False, 1.5)
    assert page._overlay_done and xray_pixel_md5(page._xray_gray_u8) == md5
    print("    same exposure again: dropped silently  OK")


def shot_offline(shots: Path) -> None:
    """Separate process state: the link singleton bound to the absent IP."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    win = _window(app)
    win.resize(1400, 820)
    win.show()
    assert _wait(app, lambda: win.xray_link.status.state == "offline"), win.xray_link.status
    win.grab().save(str(shots / "xray_link_3_offline.png"))
    print("(c) no Ethernet: 'No Ethernet link to the C-arm', retries every 3 s  OK")
    win.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carm", action="store_true", help="real link instead of 127.0.0.1")
    ap.add_argument("--shots", type=Path, help="offscreen check, PNGs into this folder")
    ap.add_argument("--offline", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    samples = sorted((Path(__file__).resolve().parent / "received_dicom").glob("*.dcm"))

    if a.shots is not None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("QT_QPA_FONTDIR",
                              os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
        a.shots.mkdir(parents=True, exist_ok=True)
        if a.offline:
            os.environ["DECAF_DICOM_HOST"] = "192.168.6.243"
            shot_offline(a.shots)
            return 0
        if not samples:
            ap.error("no samples in tests/received_dicom")
        check_receiver(samples)
        os.environ["DECAF_DICOM_HOST"] = "127.0.0.1"
        os.environ["DECAF_DICOM_PORT"] = str(_free_port())
        with tempfile.TemporaryDirectory(prefix="xray_link_gui_") as d:
            os.environ["DECAF_DICOM_DIR"] = d
            check_gui(samples, a.shots)
        return 0

    from PySide6.QtWidgets import QApplication

    if not a.carm:
        os.environ.setdefault("DECAF_DICOM_HOST", "127.0.0.1")
    app = QApplication(sys.argv)
    win = _window(app)
    win.resize(1400, 820)
    win.show()
    print(f"C-arm link on {win.xray_link.address}")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
