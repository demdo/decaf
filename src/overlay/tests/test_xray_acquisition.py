"""Checks of the X-ray Acquisition page (Multi-View) and its place in the wizard.

    python -m overlay.tests.test_xray_acquisition

Without a camera: the page reads the lab frames of 10.11.2026
(data/TwoViewBoard, lateral and AP stance) through its frame_source, and
X-rays from data/xray_received. Without those the page checks skip themselves;
(e) needs nothing but this repository.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

import cv2
import numpy as np

OVL = Path(__file__).resolve().parents[1]
LAT_DIR = OVL / "data" / "TwoViewBoard" / "20261110_175550" / "cap02_lateral"
AP_DIR = OVL / "data" / "TwoViewBoard" / "20261110_180020" / "cap02_AP"
XRAY_DIR = OVL / "data" / "xray_received"


class Skip(Exception):
    pass


def _skip(reason: str) -> None:
    raise Skip(reason)


def _app():
    from PySide6.QtWidgets import QApplication
    from overlay.gui.main import QSS
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyleSheet(QSS)
    return app


def _spin(app, seconds: float, until=None) -> bool:
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.002)
    return until is None


class Feed:
    """Camera stand-in: the frames of one folder in order, then the last one
    again (a still C-arm)."""

    _cache: dict = {}

    def __init__(self, folder: Path | None = None, blank: bool = False):
        self.blank = blank
        self.files = [] if blank else sorted(folder.glob("frame_*.png"))
        self.i = 0

    def _img(self, k: int) -> np.ndarray:
        if self.blank:
            return np.full((1080, 1920, 3), 90, np.uint8)
        f = self.files[min(k, len(self.files) - 1)]
        if f not in Feed._cache:
            Feed._cache[f] = cv2.imread(str(f))
        return Feed._cache[f]

    def __call__(self):
        k = self.i
        self.i += 1
        return self._img(k), k, k / 30.0


def _data():
    if not (LAT_DIR.exists() and AP_DIR.exists()):
        _skip("no lab frames data/TwoViewBoard")
    xrays = sorted(XRAY_DIR.glob("*/*.dcm"))
    if len(xrays) < 6:
        _skip("fewer than 6 X-rays in data/xray_received")
    from overlay.gui.evaluation import DEFAULT_CAMERA_CALIB
    if not Path(DEFAULT_CAMERA_CALIB).exists():
        _skip("no camera calibration")
    cal = np.load(DEFAULT_CAMERA_CALIB, allow_pickle=True)
    return cal["K"], cal["dist"], [str(p) for p in xrays[:6]]


def _page(K, dist, ev=None):
    from overlay.gui.state import WORKFLOW_MULTI, SessionState
    from overlay.gui.pages.page_xray_acquisition import XrayAcquisitionPage
    state = SessionState(K_rgb=K, dist_rgb=dist, workflow=WORKFLOW_MULTI, eval=ev)
    calls = []
    page = XrayAcquisitionPage(state, lambda: calls.append(1))
    page.resize(1012, 690)
    page.show()
    return state, page, calls


def _run_until(app, page, phases, max_ticks=400) -> int:
    for n in range(max_ticks):
        if page.phase in phases:
            return n
        page._tick()
        app.processEvents()
    raise AssertionError(f"phase {page.phase}, wanted {phases} within {max_ticks} ticks")


def _take_view(app, page, view, feed, xray) -> None:
    from overlay.gui.pages.page_xray_acquisition import WAIT
    page.frame_source = feed
    page.cards[view].button.click()
    assert page.phase == "pose" and page.active_view == view
    assert not page.btn_usb.isEnabled()
    _run_until(app, page, (WAIT,))
    assert page.pill.isVisible() and page.btn_usb.isEnabled()
    assert page.cards[view].button.text() == "Cancel"
    assert page.accept_received_xray(xray)
    assert page.phase == "idle"


# ------------------------------------------------------------ (a) a full acquisition

def test_page_flow() -> None:
    from overlay.gui.state import VIEW_AP, VIEW_LAT
    from overlay.tools.view_registration import BoardPoseCapture
    from overlay.tools.xray_io import read_xray_gray_u8
    app = _app()
    K, dist, xr = _data()
    state, page, calls = _page(K, dist)
    page.frame_source = Feed(LAT_DIR)
    page.on_enter()
    page.stop_timer()                       # ticks by hand from here
    assert page._camera_ok and not page.pair_mode
    assert page.image_label.width() == 576 and page.image_label.height() == 324, page.image_label.size()
    for _ in range(3):
        page._tick()
    rows = dict(page._last_rows)
    assert rows["Board"] == "34/48 corners", rows
    assert rows["Views"] == "0 of 2"
    assert page.image_label.pixmap() is not None and not page.image_label.pixmap().isNull()
    assert page.cards[VIEW_LAT]._active and not page.cards[VIEW_AP]._active
    assert not page.can_continue()

    # ① lateral: the pose equals BoardPoseCapture over the same frames
    _take_view(app, page, VIEW_LAT, Feed(LAT_DIR), xr[0])
    lat = state.xray_view_lat
    ref = BoardPoseCapture(K, dist)
    ref.start()
    f = Feed(LAT_DIR)
    while ref.add(*f()) == ref.COLLECTING:
        pass
    assert np.array_equal(lat.T_cw, ref.result.T_cw), "pose differs from BoardPoseCapture"
    # a still board: single-frame noise against the 1 s mean, ~0.13 px in the lab
    assert lat.board_shift_px < 0.2, lat.board_shift_px
    assert abs(lat.phi_deg - 1.73) < 1e-9
    assert not page.pair_mode and not page.can_continue()
    assert state.xray_image_anatomy is None             # only View 1 goes there
    assert page.cards[VIEW_AP]._active                   # AP is next
    assert page.cards[VIEW_LAT].button.text() == "Retake View 2 (Lat)"

    # ② AP: both set -> the X-ray pair, Next free
    _take_view(app, page, VIEW_AP, Feed(AP_DIR), xr[1])
    ap = state.xray_view_ap
    assert ap.seq > lat.seq and state.has_current_view_ap and page.can_continue()
    assert page.pair_mode and page.pair.isVisible() and not page.pill.isVisible()
    assert page.image_label.width() == 648 and page.image_label.height() == 324, page.image_label.size()
    rows = dict(page._last_rows)
    assert rows["Views"] == "2 of 2" and rows["Turned"] == "91.3°", rows
    assert abs(float(rows["Camera moved"].split()[0]) - 536.6) < 0.5, rows
    assert np.array_equal(state.xray_image_anatomy, read_xray_gray_u8(xr[1]))
    assert state.xray_image_anatomy_path == xr[1]
    assert _spin(app, 10.0, lambda: ap.working is not None and lat.working is not None), "working images"
    _spin(app, 0.1)
    assert page.pair.panels[VIEW_AP]._qimg is not None and page.pair.panels[VIEW_LAT]._qimg is not None
    assert not page.pair.dims[VIEW_AP].isVisible()

    # the same X-ray again: ignored, nothing held
    assert page.accept_received_xray(xr[1])
    assert page._pending is None and "Same X-ray as View 1 (AP)" in page.lbl_note.text()

    # Retake View 2, cancelled: everything stays
    page.frame_source = Feed(LAT_DIR)
    page.cards[VIEW_LAT].button.click()
    assert not page.pair_mode                           # the camera is back for the retake
    for _ in range(5):
        page._tick()
    page.cancel_view()
    assert state.xray_view_lat is lat and page.pair_mode and page.can_continue()
    assert any("Cancelled" in t for t in page.cards[VIEW_LAT].line_texts())

    # Retake View 2 done: View 1 out of date, Next locked, dimmed
    _take_view(app, page, VIEW_LAT, Feed(LAT_DIR), xr[2])
    assert state.view_ap_out_of_date and not page.can_continue()
    assert page.pair_mode and page.pair.dims[VIEW_AP].isVisible()
    assert page.cards[VIEW_AP]._active
    assert page.cards[VIEW_AP].line_texts()[0].startswith("Out of date")
    assert dict(page._last_rows)["Views"] == "View 1 out of date"

    # an X-ray without a button: held, taken right after the AP board pose
    page.frame_source = Feed(AP_DIR)
    for _ in range(3):
        page._tick()
    assert page.accept_received_xray(xr[3])
    assert page._pending is not None and page.phase == "idle"
    page.cards[VIEW_AP].button.click()
    _run_until(app, page, ("idle",))
    assert state.xray_view_ap.path == xr[3] and page.can_continue()
    assert state.xray_view_ap.board_shift_px < 0.2, state.xray_view_ap.board_shift_px

    # board moved between pose and X-ray: set, with a warning
    page.cards[VIEW_AP].button.click()
    _run_until(app, page, ("wait",))
    page.frame_source = Feed(LAT_DIR)                   # the board is somewhere else now
    page._tick()
    assert page.accept_received_xray(xr[4])
    shift = state.xray_view_ap.board_shift_px
    assert np.isfinite(shift) and shift > 0.3, shift
    assert any(t.startswith("⚠ Board moved") for t in page.cards[VIEW_AP].line_texts())

    # board never in view: the pose fails, the earlier view stays
    before = state.xray_view_ap
    page.frame_source = Feed(blank=True)
    page.cards[VIEW_AP].button.click()
    _run_until(app, page, ("idle",), max_ticks=400)
    assert state.xray_view_ap is before
    assert any("Board pose rejected" in t for t in page.cards[VIEW_AP].line_texts())

    # leaving while a view runs cancels it
    page.frame_source = Feed(AP_DIR)
    page.cards[VIEW_AP].button.click()
    page.on_leave()
    assert page.phase == "idle" and state.xray_view_ap is before and not page._camera_ok
    assert page.accept_received_xray(xr[5]) is False     # no camera: not taken
    assert calls, "on_complete_changed never called"
    page.close()
    print("  (a) acquisition flow ok")


# ------------------------------------------------------------ (b) ESC

def test_escape_key() -> None:
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from overlay.gui.state import VIEW_LAT
    app = _app()
    K, dist, xr = _data()
    state, page, _ = _page(K, dist)
    page.frame_source = Feed(LAT_DIR)
    page.on_enter()
    page.stop_timer()
    page.cards[VIEW_LAT].button.click()
    page._tick()
    page.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    assert page.phase == "idle" and state.xray_view_lat is None
    page.on_leave()
    page.close()
    print("  (b) ESC ok")


# ------------------------------------------------------------ (c) Evaluation files

def test_evaluation_files() -> None:
    import json
    from overlay.gui.evaluation import EvaluationSession
    from overlay.gui.state import VIEW_AP, VIEW_LAT
    app = _app()
    K, dist, xr = _data()
    with tempfile.TemporaryDirectory() as tmp:
        ev = EvaluationSession(Path(tmp), "acq_test")
        ev.set_label("two_view")
        state, page, _ = _page(K, dist, ev)
        page.frame_source = Feed(LAT_DIR)
        page.on_enter()
        page.stop_timer()
        _take_view(app, page, VIEW_LAT, Feed(LAT_DIR), xr[0])
        _take_view(app, page, VIEW_AP, Feed(AP_DIR), xr[1])
        names = sorted(p.name for p in ev.dir.iterdir())
        for want in ("two_view_xray_view2_lat.dcm", "two_view_board_view2_lat.npz",
                     "two_view_board_view2_lat.json", "two_view_xray_view1_ap.dcm",
                     "two_view_board_view1_ap.npz", "two_view_board_view1_ap.json"):
            assert want in names, (want, names)
        js = json.loads((ev.dir / "two_view_board_view1_ap.json").read_text(encoding="utf-8"))
        assert js["view"] == "view1" and abs(js["camera_motion_lat_to_ap"]["angle_deg"] - 91.30) < 0.05, js
        assert js["board_pose"]["accepted"] is True and len(js["T_cw_m"]) == 4
        with np.load(ev.dir / "two_view_board_view2_lat.npz") as npz:   # closed: Windows locks it
            assert np.array_equal(npz["T_cw"], state.xray_view_lat.T_cw)
        log = (ev.dir / "log.md").read_text(encoding="utf-8")
        assert log.count("xray_acquisition") >= 4, log
        page.on_leave()
        page.close()
    print("  (c) evaluation files ok")


# ------------------------------------------------------------ (f) new views from Overlay Live

def test_new_views() -> None:
    """'New views with board' in Overlay Live (11.11.): the board was off the table, so
    View 1 is locked until View 2 is retaken with the board back; then AP last as always."""
    from overlay.gui.state import VIEW_AP, VIEW_LAT
    app = _app()
    K, dist, xr = _data()
    state, page, _ = _page(K, dist)
    page.frame_source = Feed(LAT_DIR)
    page.on_enter()
    page.stop_timer()
    _take_view(app, page, VIEW_LAT, Feed(LAT_DIR), xr[0])
    _take_view(app, page, VIEW_AP, Feed(AP_DIR), xr[1])
    assert "REMOVE THE BOARD" in page.instructions_label.text(), page.instructions_label.text()
    assert "Tool Tracking" in page.instructions_label.text()     # no tool yet
    ap_before = state.xray_view_ap
    page.begin_new_views()
    assert page.lat_first and page.can_continue()        # the views on screen stay valid
    assert "put the board back" in page.instructions_label.text()
    assert not page.cards[VIEW_AP].button.isEnabled() and page.cards[VIEW_LAT].button.isEnabled()
    page.start_view(VIEW_AP)
    assert page.phase == "idle" and state.xray_view_ap is ap_before, "View 1 must wait for View 2"
    _take_view(app, page, VIEW_LAT, Feed(LAT_DIR), xr[2])
    assert not page.lat_first and state.view_ap_out_of_date and not page.can_continue()
    assert page.cards[VIEW_AP].button.isEnabled()
    _take_view(app, page, VIEW_AP, Feed(AP_DIR), xr[3])
    assert page.can_continue() and state.xray_view_ap is not ap_before
    page.on_leave()
    page.close()
    print("  (f) new views from Live: View 1 locked until View 2 is retaken")


# ------------------------------------------------------------ (d) wizard, Multi-View

def test_main_multi() -> None:
    from overlay.gui.evaluation import StartupChoice
    from overlay.gui.main import MainWindow
    from overlay.gui.state import VIEW_AP, VIEW_LAT, XrayViewCapture
    app = _app()
    K, dist, _ = _data()
    win = MainWindow(StartupChoice(workflow="multi"))
    win.xray_link.stop()
    try:
        seq = win._b_sequence()
        assert seq == [win.IDX_PTR_DEPTH, win.IDX_OVERLAY_PREVIEW, win.IDX_XRAY_ACQ,
                       win.IDX_TOOL_TRACKING, win.IDX_OVERLAY_LIVE], seq
        items = [win.runtime_list.item(i).text()[2:] for i in range(win.runtime_list.count())]
        assert items == ["Pointer Tool Depth", "Overlay Preview", "X-ray Acquisition",
                         "Tool Tracking", "Overlay Live"], items
        heads = [lab.text() for lab in win.findChildren(type(win.lbl_step_title))]
        assert "Runtime (Mode B) — Multi-View" in heads
        win.state.K_rgb, win.state.dist_rgb = K, dist
        win.page_xray_acq.frame_source = Feed(AP_DIR)
        win.mode_b_active = True
        win.current_step = win.IDX_OVERLAY_PREVIEW
        win._switch_to_page(win.IDX_XRAY_ACQ)
        page = win.page_xray_acq
        page.stop_timer()
        assert win.lbl_step_title.text() == "Multi-View — X-ray Acquisition"
        assert win.runtime_list.currentRow() == 2
        assert page.btn_back.isEnabled() and not page.btn_next.isEnabled()
        page.cards[VIEW_AP].button.click()          # busy: Back locked too
        assert not page.btn_back.isEnabled() and not page.btn_next.isEnabled()
        page.cancel_view()
        assert page.btn_back.isEnabled()
        # views in the state: Next free once View 1 is newer than View 2
        img = np.zeros((8, 8), np.uint8)
        summ = {"corners_used": 34, "corners_total": 48, "rms_px": 0.2}
        win.state.xray_views[VIEW_LAT] = XrayViewCapture(VIEW_LAT, img, "a", np.eye(4), summ, 0.0, 1)
        win.update_ui()
        assert not page.btn_next.isEnabled() and not win.step_complete(win.IDX_XRAY_ACQ)
        win.state.xray_views[VIEW_AP] = XrayViewCapture(VIEW_AP, img + 1, "b", np.eye(4), summ, 0.0, 2)
        win.update_ui()
        assert page.btn_next.isEnabled() and win.step_complete(win.IDX_XRAY_ACQ)
        # Overlay Live in Multi: View 1 and a tool, no plane from the check (user 11.11.)
        assert win.state.H_xc is None and not win.step_complete(win.IDX_OVERLAY_LIVE)
        win.state.tool_assets = object()
        assert win.step_complete(win.IDX_OVERLAY_LIVE)
        win.state.xray_views[VIEW_LAT] = XrayViewCapture(VIEW_LAT, img + 2, "c", np.eye(4), summ, 0.0, 3)
        win.update_ui()
        assert not page.btn_next.isEnabled(), "View 1 out of date must lock Next"
        assert not win.step_complete(win.IDX_OVERLAY_LIVE)
        # 'Next' straight back to Overlay Live once the tool is loaded (11.11.)
        assert win._next_b_target(win.IDX_XRAY_ACQ) == win.IDX_OVERLAY_LIVE
        win.state.tool_assets = None
        assert win._next_b_target(win.IDX_XRAY_ACQ) == win.IDX_TOOL_TRACKING
        # 'New views with board' in Overlay Live: X-ray Acquisition, View 2 first
        win.page_overlay_live.new_views_requested.emit()
        assert win.current_step == win.IDX_XRAY_ACQ and page.lat_first
        assert not page.cards[VIEW_AP].button.isEnabled()
        page.on_leave()
    finally:
        win.close()
    print("  (d) wizard Multi-View ok")


# ------------------------------------------------------------ (e) wizard, Single View unchanged

def test_main_single() -> None:
    from overlay.gui.evaluation import StartupChoice
    from overlay.gui.main import MainWindow
    _app()
    win = MainWindow(StartupChoice())
    win.xray_link.stop()
    try:
        assert win._b_sequence() == [win.IDX_PTR_DEPTH, win.IDX_OVERLAY_PREVIEW,
                                     win.IDX_TOOL_TRACKING, win.IDX_OVERLAY_LIVE]
        items = [win.runtime_list.item(i).text()[2:] for i in range(win.runtime_list.count())]
        assert items == ["Pointer Tool Depth", "Overlay Preview", "Tool Tracking", "Overlay Live"], items
        heads = [lab.text() for lab in win.findChildren(type(win.lbl_step_title))]
        assert "Runtime (Mode B)" in heads
        for idx, title in ((win.IDX_PTR_DEPTH, "Mode B — Pointer Tool Depth"),
                           (win.IDX_OVERLAY_PREVIEW, "Mode B — Overlay Preview"),
                           (win.IDX_TOOL_TRACKING, "Mode B — Tool Tracking")):
            win.current_step = idx
            win.update_ui()
            assert win.lbl_step_title.text() == title, win.lbl_step_title.text()
    finally:
        win.close()
    print("  (e) wizard Single View unchanged")


def main() -> int:
    tests = (test_main_single, test_main_multi, test_page_flow, test_escape_key, test_evaluation_files,
             test_new_views)
    skipped = 0
    for test in tests:
        try:
            test()
        except Skip as s:
            skipped += 1
            print(f"  SKIPPED {test.__name__}: {s}")
    print("all checks passed" + (f" ({skipped} skipped: data missing)" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
