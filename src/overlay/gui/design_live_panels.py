# overlay/gui/design_live_panels.py
"""Design the live panels of the user study on recorded data. No camera, no tracker.

    python -m overlay.gui.design_live_panels                     B_2, window
    python -m overlay.gui.design_live_panels <snapshot folder> [--tracking <npz>]
    python -m overlay.gui.design_live_panels --grab out.png [--size 1920x1080]
    python -m overlay.gui.design_live_panels --bench 8           numbers, see below

A snapshot folder (00_xray, 01_rgb, snapshot.json) gives the images and the
calibration; <label>_tracking.npz next to it the real jitter of the pose. Sliders
move and tilt the drill virtually, so one sees how the marks behave at the field
edge and on entry. The photo of the drill stays where it was, of course. The band
with the sliders is hidden, so the window looks like the live page; key D shows it.

The panels are the widgets the live page will use (gui/widgets/widget_live_panels.py)
and PoseStream does what the tracking thread will do there: a C++ load, then
MarkBuilder.build. So --bench measures in the running Qt loop, with caches as cold
as on the page, and it runs today's drawing (TrackingOverlay) next to it under the
same load. Offline loops were five times too optimistic here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QLineF, Qt, QThread, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from overlay.calib.calib_xray_to_pointer import extract_depth
from overlay.gui.widgets.widget_live_panels import (
    POSE_HOLD_S,
    VIEW_LIVE,
    VIEW_XRAY,
    LivePanelsView,
    readout_int,
)
from overlay.tools.homography import estimate_plane_induced_homography
from overlay.tools.pointer_depth import PointerDepthEstimator
from overlay.tools.tool_marks import (
    ORIENT_C_ARM,
    ORIENT_CAMERA,
    RULER_FIELD,
    RULER_MODES,
    RULER_SET,
    MarkBuilder,
)
from overlay.tools.tracking_overlay import TrackingOverlay
from overlay.tools.warp import prepare_live_overlay
from overlay.tools.xray_distortion import load_xray_distortion
from overlay.tools.xray_view import XrayField, XrayView

MASTERARBEIT = Path(__file__).resolve().parents[5]
DATA = MASTERARBEIT / "Thesis" / "Plots" / "Evaluation" / "data" / "IntegratedSystem"
K_XRAY_NPZ = Path(__file__).resolve().parents[1] / "data" / "K_xray.npz"

SNAPSHOTS = {
    "B_2 (drill inclined)": DATA / "Evaluation_Integrated_System" / "B_2_live",
    "A_2 (drill in Π)": DATA / "Evaluation_Integrated_System" / "A_2_live",
    "Bone_New (bone)": DATA / "Evaluation_Integrated_System_Bone_New" / "2_live",
}

READOUT_PERIOD_S = 0.2
FPS = 30.0


_REMAP: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _working_image(raw: np.ndarray) -> np.ndarray:
    """As OverlayLivePage._load_xray_from_state, with the maps kept between images:
    computing them takes seconds, applying them a millisecond."""
    h, w = raw.shape[:2]
    maps = _REMAP.get((w, h))
    if maps is None:
        print("Pre-computing the X-ray undistortion once ...", flush=True)
        maps = load_xray_distortion(str(K_XRAY_NPZ)).working_remap(w, h)
        _REMAP[(w, h)] = maps
    return cv2.remap(np.ascontiguousarray(cv2.flip(raw, 1)), maps[0], maps[1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class Snapshot:
    """One recording, read only."""

    def __init__(self, folder: Path, tracking: Optional[Path] = None) -> None:
        self.folder = Path(folder)
        s = json.loads((self.folder / "snapshot.json").read_text())
        self.record = s
        self.K_c = np.array(s["K_rgb"], dtype=np.float64)
        self.dist = np.array(s["dist_rgb"], dtype=np.float64)
        self.K_x = np.array(s["K_xray"], dtype=np.float64)
        self.T_xc = np.array(s["T_xc"], dtype=np.float64)
        self.d_x = float(s["d_x_mm"])
        self.alpha = float(s.get("alpha", 0.5))
        tool = s["tool"]
        self.p_tip = np.array(tool["p_tip_mm"], dtype=np.float64)
        self.axis = np.array(tool["axis_dir"], dtype=np.float64)
        self.points = np.array([tool["tip_cam_mm"], tool["back_cam_mm"]], dtype=np.float64)

        raw = cv2.imread(str(self.folder / "00_xray.png"), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(str(self.folder / "01_rgb.png"))
        if raw is None or rgb is None:
            raise FileNotFoundError(f"no 00_xray.png / 01_rgb.png in {self.folder}")
        self.raw = raw
        self.bgra = cv2.cvtColor(rgb, cv2.COLOR_BGR2BGRA)
        self.working = _working_image(raw)

        # the recording's jitter, moved to the snapshot's pose
        self.stream = self.points[None, :, :]
        path = Path(tracking) if tracking else self._tracking_next_to(self.folder)
        if path is not None and path.exists():
            z = np.load(path)
            tip, back = z["tip_cam_mm"], z["back_cam_mm"]
            ok = np.isfinite(tip).all(axis=1) & np.isfinite(back).all(axis=1)
            if ok.any():
                stream = np.stack([tip[ok], back[ok]], axis=1)
                self.stream = stream - stream.mean(axis=0, keepdims=True) + self.points[None, :, :]
        self.jitter_mm = float(np.linalg.norm(self.stream[:, 0] - self.points[0], axis=1).std())

    @staticmethod
    def _tracking_next_to(folder: Path) -> Optional[Path]:
        name = folder.name
        i = name.find("_live")
        if i <= 0:
            return None
        return folder.parent / f"{name[:i]}_tracking.npz"


class VirtualMove:
    """Sliders of the design band, in the X-ray frame around the tip."""

    def __init__(self, advance=0.0, x=0.0, y=0.0, z=0.0, turn_deg=0.0, tilt_deg=0.0) -> None:
        self.advance, self.x, self.y, self.z = advance, x, y, z
        self.turn_deg, self.tilt_deg = turn_deg, tilt_deg

    def apply(self, points: np.ndarray, view: XrayView) -> np.ndarray:
        """points (2, 3): tip, back in the camera frame. Tilt changes the inclination
        against Pi, turn rotates in Pi, advance runs along the moved axis."""
        P = view.to_xray(points)
        tip, back = P[0], P[1]
        a = tip - back
        length = float(np.linalg.norm(a))
        a = a / length
        if self.tilt_deg:
            # about the axis in Pi that is perpendicular to the drill
            side = np.cross(a, (0.0, 0.0, 1.0))
            n = float(np.linalg.norm(side))
            side = side / n if n > 1e-9 else np.array((1.0, 0.0, 0.0))
            R, _ = cv2.Rodrigues(side * math.radians(self.tilt_deg))
            a = R @ a
        if self.turn_deg:
            R, _ = cv2.Rodrigues(np.array((0.0, 0.0, math.radians(self.turn_deg))))
            a = R @ a
        tip = tip + self.advance * a + np.array((self.x, self.y, self.z))
        back = tip - length * a
        return np.stack([tip, back]) @ view.R.T + view.t_mm


class PoseStream(threading.Thread):
    """What the tracking thread does on the page: the engine (a blur in C++ on all
    cores, without the GIL, as tests/test_xray_view.py) and then MarkBuilder.build.
    The newest marks sit in self.latest; assigning a reference is atomic."""

    def __init__(self, builder: MarkBuilder, load_frame: Optional[np.ndarray]) -> None:
        super().__init__(name="pose-stream", daemon=True)
        self.builder = builder
        self.load_frame = load_frame
        self.latest = None
        self.active = True
        self.jitter = True
        self.stream = None
        self.view: Optional[XrayView] = None
        self.move = VirtualMove()
        self.build_s: list[float] = []
        self._stop_flag = False
        self.points_now: Optional[np.ndarray] = None

    def stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        period = 1.0 / FPS
        t_next = time.perf_counter()
        i = 0
        while not self._stop_flag:
            t_next += period
            delay = t_next - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
            else:
                t_next = time.perf_counter()
            if not self.active or self.stream is None or self.view is None:
                continue
            if self.load_frame is not None:
                cv2.GaussianBlur(self.load_frame, (31, 31), 0)
            stream, view, move = self.stream, self.view, self.move
            points = stream[i % len(stream)] if self.jitter else stream.mean(axis=0)
            i += 1
            points = move.apply(points, view)
            t0 = time.perf_counter()
            marks = self.builder.build(points, t_pose=t0, kind="fused")
            self.build_s.append(time.perf_counter() - t0)
            self.points_now = points
            self.latest = marks


class _Slider(QWidget):
    """A label, a value, a slider; floats by a fixed step."""

    def __init__(self, text: str, unit: str, lo: float, hi: float, step: float, on_change) -> None:
        super().__init__()
        self.unit, self.step = unit, step
        self.name = QLabel(text)
        self.name.setObjectName("livePanelsWeak")
        self.value = QLabel()
        self.value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(round(lo / step)), int(round(hi / step)))
        self.slider.setFocusPolicy(Qt.NoFocus)
        self._on_change = on_change
        self.slider.valueChanged.connect(self._changed)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.name)
        top.addStretch(1)
        top.addWidget(self.value)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(top)
        lay.addWidget(self.slider)
        self._show()

    def _show(self) -> None:
        self.value.setText(f"{self.get():+.0f} {self.unit}")

    def _changed(self, _value: int) -> None:
        self._show()
        self._on_change()

    def get(self) -> float:
        return self.slider.value() * self.step

    def set(self, value: float) -> None:
        self.slider.setValue(int(round(value / self.step)))


class DesignWindow(QWidget):
    """LivePanelsView plus a band with the design sliders, fed from one Snapshot."""

    prof: Optional[dict] = None     # stage -> milliseconds, only under --bench
    _in_frame = False               # panel 1 paints inside _on_frame ("frame logic")
    readout_period_s = READOUT_PERIOD_S

    def __init__(self, snapshot: Snapshot, *, load: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("DeCAF: live panels design")
        self.view = LivePanelsView(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)

        self.snap: Optional[Snapshot] = None
        self.builder = MarkBuilder(snapshot.K_c, snapshot.dist, line_factory=QLineF)
        self._frame: Optional[np.ndarray] = None
        self._overlay_cache: dict[int, object] = {}
        self._dx_offset = 0
        self._base_dx = snapshot.d_x
        self._plane_acquiring = False
        self._plane_new_dx = 0.0
        self._plane_capture: Optional[Path] = None
        self._plane_status = ""         # shown instead of the tracking status
        self._captures: Optional[list] = None
        self._capture_next = 0
        self._pointer: Optional[PointerDepthEstimator] = None
        self._view_x: Optional[XrayView] = None
        self._field: Optional[XrayField] = None
        self._tracking_on = True
        self._t_readout = 0.0
        self._t_age = 0.0
        self._t_start_wall = time.time()
        self._frames = 0
        self._fps = 0.0
        self._t_fps = time.perf_counter()
        self._last_shown = None
        self._compare_today = False
        self._today: Optional[TrackingOverlay] = None
        self._today_buf: Optional[np.ndarray] = None

        self.stream = PoseStream(self.builder, snapshot.bgra.copy() if load else None)
        self._build_design_band()
        self._connect()
        self.setFocusPolicy(Qt.StrongFocus)
        self.view.panel_live.roi_changed.connect(self._on_roi_changed)
        self.view.view_changed.connect(lambda _view: self._on_roi_changed())
        self.view.panel_view1.map_changed.connect(self._on_map_changed)
        self.load_snapshot(snapshot)
        self.stream.start()

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._on_frame)
        self.timer.start(int(1000.0 / FPS))

    # ---------------- design band ----------------

    def _build_design_band(self) -> None:
        band = QFrame()
        band.setObjectName("livePanelsCard")
        band.setAttribute(Qt.WA_StyledBackground, True)
        grid = QGridLayout(band)
        grid.setContentsMargins(10, 6, 10, 6)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)
        title = QLabel("Design: move the drill virtually")
        title.setObjectName("livePanelsCardTitle")
        grid.addWidget(title, 0, 0, 1, 3)

        c = self._on_move_changed
        self.sl_advance = _Slider("Advance along axis", "mm", -150, 150, 1, c)
        self.sl_x = _Slider("Shift x<sub>x</sub>", "mm", -80, 80, 1, c)
        self.sl_y = _Slider("Shift y<sub>x</sub>", "mm", -80, 80, 1, c)
        self.sl_z = _Slider("Depth z<sub>x</sub>", "mm", -80, 80, 1, c)
        self.sl_turn = _Slider("Turn in Π", "°", -180, 180, 1, c)
        self.sl_tilt = _Slider("Tilt against Π", "°", -70, 70, 1, c)
        self._move_sliders = (self.sl_advance, self.sl_x, self.sl_y, self.sl_z, self.sl_turn, self.sl_tilt)
        for i, s in enumerate(self._move_sliders):
            grid.addWidget(s, 1 + i // 3, i % 3)

        self.cmb_snapshot = QComboBox()
        self.cmb_snapshot.setFocusPolicy(Qt.NoFocus)
        self.cmb_snapshot.addItems(list(SNAPSHOTS))
        self.cmb_orientation = QComboBox()
        self.cmb_orientation.setFocusPolicy(Qt.NoFocus)
        self.cmb_orientation.addItem("X-ray as the camera sees it", ORIENT_CAMERA)
        self.cmb_orientation.addItem("X-ray as on the C-arm", ORIENT_C_ARM)
        self.chk_jitter = QCheckBox("Real jitter")
        self.chk_jitter.setChecked(True)
        self.chk_jitter.setFocusPolicy(Qt.NoFocus)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setObjectName("livePanelsButton")
        self.btn_reset.setFocusPolicy(Qt.NoFocus)
        side = QGridLayout()
        side.setSpacing(6)
        side.addWidget(self.cmb_snapshot, 0, 0)
        side.addWidget(self.cmb_orientation, 0, 1)
        side.addWidget(self.chk_jitter, 1, 0)
        side.addWidget(self.btn_reset, 1, 1)
        grid.addLayout(side, 1, 3, 2, 1)
        for col, stretch in enumerate((3, 3, 3, 4)):
            grid.setColumnStretch(col, stretch)
        self.view.footer.addWidget(band)
        # hidden, so the window looks like the live page and panel 1 gets its height (user
        # 10.11.); key D shows it for testing, the sliders work either way
        self.design_band = band
        band.setVisible(False)

    def _connect(self) -> None:
        v = self.view
        v.slider_alpha.valueChanged.connect(self._on_alpha)
        v.slider_dx.valueChanged.connect(self._on_dx)
        v.chk_show_camera_view.toggled.connect(self._on_views)
        v.chk_show_xray_view.toggled.connect(self._on_views)
        v.btn_track.clicked.connect(self._on_track)
        v.btn_plane.clicked.connect(self._on_plane)
        self.cmb_snapshot.currentTextChanged.connect(self._on_snapshot_chosen)
        self.cmb_orientation.currentIndexChanged.connect(self._on_orientation)
        self.chk_jitter.toggled.connect(self._on_jitter)
        self.btn_reset.clicked.connect(self._reset_moves)
        # not in the design tool: the live page has them (OverlayLivePage)
        for b in (v.btn_load_xray, v.btn_lateral, v.btn_save, v.btn_usb):
            b.setEnabled(False)
        v.chk_marker_exposure.setEnabled(False)

    # ---------------- data ----------------

    def load_snapshot(self, snap: Snapshot) -> None:
        self.stream.active = False
        self.snap = snap
        self._frame = snap.bgra
        self._overlay_cache = {}
        self._dx_offset = 0
        self._base_dx = snap.d_x
        self._plane_acquiring = False
        self._plane_status = ""
        self._captures = None
        v = self.view
        self.builder.set_camera(snap.K_c, snap.dist)
        self.builder.set_ruler(RULER_FIELD)

        h, w = snap.bgra.shape[:2]
        self._view_x = XrayView(snap.T_xc, snap.K_x)
        wh = (snap.working.shape[1], snap.working.shape[0])
        ov = TrackingOverlay(snap.K_c, snap.dist, snap.p_tip, snap.axis)
        self._field = XrayField.from_image(
            snap.working, ov.field_inset_px(self._view_x, wh, snap.d_x)
        )
        self.builder.set_xray(self._view_x, self._field, snap.d_x, wh)
        self._today = ov
        ov.set_xray_view(self._view_x, self._field, snap.d_x)

        v.panel_live.set_frame_size((w, h))
        self._centre_crop()     # the crop sits on the field
        v.panel_live.alpha = snap.alpha
        v.slider_alpha.blockSignals(True)
        v.slider_alpha.setValue(int(round(snap.alpha * 100)))
        v.slider_alpha.blockSignals(False)
        v.lbl_alpha_value.setText(f"{snap.alpha:.2f}")
        v.slider_dx.blockSignals(True)
        v.slider_dx.setValue(0)
        v.slider_dx.blockSignals(False)
        v.panel_view1.set_xray(snap.working, self._t_start_wall, time.strftime("%H:%M"))
        self._apply_dx()
        self._on_roi_changed()
        self._on_map_changed()
        self.stream.stream = snap.stream
        self.stream.view = self._view_x
        self.stream.latest = None
        self.stream.active = self._tracking_on

    def _apply_dx(self) -> None:
        snap = self.snap
        d_x = self._base_dx + self._dx_offset
        overlay = self._overlay_cache.get(self._dx_offset)
        if overlay is None:
            H = estimate_plane_induced_homography(
                K_c=snap.K_c, R_xc=snap.T_xc[:3, :3], t_xc=snap.T_xc[:3, 3], K_x=snap.K_x, d_x=d_x,
            )
            h, w = snap.bgra.shape[:2]
            overlay = prepare_live_overlay(
                xray_gray_u8=snap.working, H_xc=H, camera_size_wh=(w, h),
                K_camera=snap.K_c, dist_coeffs=snap.dist,
            )
            self._overlay_cache[self._dx_offset] = overlay
        wh = (snap.working.shape[1], snap.working.shape[0])
        self.builder.set_plane_depth(d_x, wh)
        self.view.panel_live.set_overlay(overlay)
        self.view.panel_live.set_field_outline(self.builder.field_outline_on_camera())
        self.view.lbl_dx_value.setText(f"{d_x:.1f} mm")
        if self._today is not None:
            self._today.set_plane_depth(d_x)

    def _on_roi_changed(self) -> None:
        panel = self.view.panel_live
        self.builder.set_camera_panel(panel.roi if self.view.view == VIEW_LIVE else None,
                                      panel.mark_scale)
        panel.set_field_outline(self.builder.field_outline_on_camera())

    def _on_map_changed(self) -> None:
        panel = self.view.panel_view1
        self.builder.set_xray_panel(panel.panel_map, panel.mark_scale)

    # ---------------- controls ----------------

    def _on_snapshot_chosen(self, name: str) -> None:
        self.load_snapshot(Snapshot(SNAPSHOTS[name]))
        self._reset_moves()

    def _on_alpha(self, value: int) -> None:
        self.view.panel_live.alpha = value / 100.0
        self.view.lbl_alpha_value.setText(f"{value / 100.0:.2f}")

    def _on_dx(self, value: int) -> None:
        self._dx_offset = int(value)
        self._apply_dx()

    def _on_views(self, _checked: bool = False) -> None:
        self.view.panel_live.show_camera_view = self.view.chk_show_camera_view.isChecked()
        self.view.panel_live.show_xray_view = self.view.chk_show_xray_view.isChecked()

    def _on_track(self) -> None:
        self._tracking_on = not self._tracking_on
        self.stream.active = self._tracking_on
        self.view.btn_track.setText("Stop tracking" if self._tracking_on else "Start tracking")
        if self._tracking_on:
            self._plane_status = ""

    # ---------------- reference plane: only a new plane for the X-ray on screen ----------------
    # As OverlayLivePage.start_plane_only, on the recorded pointer frames of this calibration
    # (<label>_depth.npz): the real pointer solve, each press the next frame, the farthest plane
    # first. While acquiring, panel 1 shows that frame.

    def _pointer_captures(self) -> list:
        if self._captures is None:
            snap = self.snap
            found = []
            for path in sorted(snap.folder.parent.glob("*depth.npz")):
                with np.load(path) as z:
                    if np.allclose(z["T_xc"], snap.T_xc):
                        found.append((abs(float(z["d_x_mm"]) - self._base_dx), path))
            found.sort(key=lambda item: -item[0])
            self._captures = [path for _, path in found]
            self._capture_next = 0
        return self._captures

    def _on_plane(self) -> None:
        if self._plane_acquiring:
            self._end_plane("Reference plane: cancelled")
            return
        captures = self._pointer_captures()
        v = self.view
        if not captures:
            self._plane_status = "Reference plane: no recorded pointer frame here"
            v.lbl_status.setText(self._plane_status)
            return
        path = captures[self._capture_next % len(captures)]
        self._capture_next += 1
        snap = self.snap
        with np.load(path) as z:
            rgb = z["rgb_image"]
        frame = cv2.cvtColor(rgb, cv2.COLOR_BGR2BGRA)
        if self._pointer is None:
            self._pointer = PointerDepthEstimator(FPS)
        self._pointer.reset()
        try:
            _raw, res = self._pointer.solve(frame, snap.K_c, snap.dist)
            d_new = float(extract_depth(T_xc=snap.T_xc, T_tc=res.T_4x4).d_x_mm)
        except Exception:
            self._plane_status = f"Reference plane: pointer NOT FOUND in {path.stem}"
            v.lbl_status.setText(self._plane_status)
            return

        # tracker and pointer do not share the frame budget: tracking pauses
        if self._tracking_on:
            self._on_track()
        self._plane_acquiring = True
        self._plane_new_dx = d_new
        self._plane_capture = path
        self._frame = frame
        v.btn_plane.setText("Cancel")
        v.btn_track.setEnabled(False)
        self._plane_status = (f"Pointer found: d_x {d_new:.1f} mm (now {self._base_dx + self._dx_offset:.1f}). "
                              "SPACE accept, ESC cancel")
        v.lbl_status.setText(self._plane_status)

    def _accept_plane(self) -> None:
        if self._plane_acquiring:
            self._set_plane(self._plane_new_dx)
            self._end_plane(f"Reference plane set: d_x {self._base_dx:.1f} mm, same X-ray")

    def _set_plane(self, d_x: float) -> None:
        """A new plane for the X-ray on screen. The image, its field and its age stay."""
        self._base_dx = float(d_x)
        self._dx_offset = 0
        self._overlay_cache = {}
        slider = self.view.slider_dx
        slider.blockSignals(True)
        slider.setValue(0)
        slider.blockSignals(False)
        self._apply_dx()
        # another plane is another place in the camera image: the crop goes with it
        self._centre_crop()

    def _centre_crop(self) -> None:
        h, w = self.snap.bgra.shape[:2]
        self.builder.set_camera_panel((0, 0, w, h))
        outline = np.array(self.builder.field_outline_on_camera())
        cx, cy = outline.mean(axis=0) if len(outline) else (w / 2.0, h / 2.0)
        self.view.panel_live.set_centre(cx, cy)
        self._on_roi_changed()

    def _end_plane(self, status: str) -> None:
        self._plane_acquiring = False
        self._frame = self.snap.bgra
        v = self.view
        v.btn_plane.setText("Set reference plane")
        v.btn_track.setEnabled(True)
        self._plane_status = status
        v.lbl_status.setText(status)

    def _on_orientation(self, _index: int) -> None:
        self.view.panel_view1.set_orientation(self.cmb_orientation.currentData())

    def _on_jitter(self, on: bool) -> None:
        self.stream.jitter = on

    def _on_move_changed(self) -> None:
        self.stream.move = VirtualMove(
            self.sl_advance.get(), self.sl_x.get(), self.sl_y.get(), self.sl_z.get(),
            self.sl_turn.get(), self.sl_tilt.get(),
        )

    def _reset_moves(self) -> None:
        for s in self._move_sliders:
            s.set(0.0)

    def set_zero(self) -> None:
        """Null setzen: the tip now is the zero of ruler and insertion depth. Not on
        the card any more (user 10.11.: the ruler counts from the mask entry); kept
        for --zero and the tests, MarkBuilder still has the mode."""
        m = self.stream.latest
        if m is None:
            return
        self.builder.set_ruler(RULER_SET, m.tip_cam)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if self._plane_acquiring and key in (Qt.Key_Space, Qt.Key_Escape):
            if key == Qt.Key_Space:
                self._accept_plane()
            else:
                self._end_plane("Reference plane: cancelled")
        elif key == Qt.Key_D:
            self.design_band.setVisible(not self.design_band.isVisible())
        elif key == Qt.Key_1:
            self.view.set_view(VIEW_LIVE)
        elif key == Qt.Key_2:
            self.view.set_view(VIEW_XRAY)
        elif key == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif key == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()
        else:
            super().keyPressEvent(event)

    # ---------------- the frame ----------------

    def _stage(self, name: str, seconds: float) -> None:
        if self.prof is not None:
            self.prof[name].append(seconds * 1e3)

    def _on_frame(self) -> None:
        now = time.perf_counter()
        v = self.view
        marks = self.stream.latest
        if not self._tracking_on or (marks is not None and now - marks.t_pose > POSE_HOLD_S):
            marks = None
        # a marks object reaches the panels once, per pose
        fresh = marks is not self._last_shown
        self._last_shown = marks

        # nothing of the video without its panel
        self._in_frame = True
        if v.view == VIEW_LIVE:
            v.panel_live.set_marks(marks)
            # one still picture here: it stays valid
            v.panel_live.show_frame(self._frame, stays_valid=True)
        self._in_frame = False
        v.panel_view1.set_marks(marks)
        v.panel_view1.set_lost(marks is None)
        # per pose, not at the read-out rate: it paints only when the tip or axis moved
        if marks is None:
            v.plane_view.set_state(None, None)
        else:
            v.plane_view.set_state(marks.off_plane_mm, marks.rise_deg)

        if self._compare_today and marks is not None and fresh:
            self._draw_today(self.stream.points_now)

        self._frames += 1
        if now - self._t_readout >= self.readout_period_s:
            self._t_readout = now
            self._update_readouts(marks)
        if now - self._t_age >= 1.0:
            self._t_age = now
            v.panel_view1.tick(time.time())
            dt = now - self._t_fps
            self._fps = self._frames / dt if dt > 0 else 0.0
            self._frames, self._t_fps = 0, now
        self._stage("frame logic", time.perf_counter() - now)

    def _update_readouts(self, marks) -> None:
        v = self.view
        if self._plane_status:
            v.lbl_status.setText(self._plane_status)
        if marks is None:
            v.tile_off_plane.set_value(None)
            v.tile_inclination.set_value(None)
            v.lbl_track_status.setText("Pose: –")
            if not self._plane_status:
                v.lbl_status.setText("Tracking off" if not self._tracking_on else "No tracking")
            return
        # whole numbers (user 10.11.)
        v.tile_off_plane.set_value(readout_int(marks.off_plane_mm, signed=True))
        v.tile_inclination.set_value(readout_int(marks.inclination_deg))
        v.lbl_track_status.setText(f"Pose: {marks.kind or '–'}")
        if not self._plane_status:
            v.lbl_status.setText(f"Tracking active · {marks.kind or '–'} · {self._fps:.0f} fps"
                                 + (" · marker partly occluded" if marks.marked else ""))

    # ---------------- bench ----------------

    def _draw_today(self, points) -> None:
        """Today's path of OverlayLivePage._compose_frame, for comparison, into a
        scratch buffer: full frame copy, geometry and drawing in the GUI thread."""
        ov = self._today
        if self._today_buf is None:
            self._today_buf = np.empty_like(self._frame)
        t0 = time.perf_counter()
        np.copyto(self._today_buf, self._frame)
        t1 = time.perf_counter()
        ov.update(points)
        t2 = time.perf_counter()
        ov.draw_into(self._today_buf, show_camera=True, show_xray=False)
        t3 = time.perf_counter()
        ov.draw_into(self._today_buf, show_camera=False, show_xray=True)
        t4 = time.perf_counter()
        self._stage("today copy full frame", t1 - t0)
        self._stage("today geometry", t2 - t1)
        self._stage("today draw green", t3 - t2)
        self._stage("today draw magenta", t4 - t3)

    def event(self, event) -> bool:
        if self.prof is not None and not self._in_frame and event.type() == QEvent.UpdateRequest:
            t0 = time.perf_counter()
            r = super().event(event)
            self._stage("repaint + flush", time.perf_counter() - t0)
            return r
        return super().event(event)

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.stream.stop()
        super().closeEvent(event)


def _run_bench(app: QApplication, win: DesignWindow, seconds: float) -> None:
    """Three phases in the running loop: warm-up, new path only, new path + today's."""
    win.prof = defaultdict(list)
    prof_all = {}

    def stage(name, dt):
        win.prof[name].append(dt * 1e3)

    win.view.panel_live.prof = stage
    win.view.panel_view1.prof = stage
    for tile in win.view._tiles:
        tile.prof = stage
    win.view.plane_view.prof = stage

    def spin(t: float) -> None:
        t_end = time.perf_counter() + t
        while time.perf_counter() < t_end:
            app.processEvents()
            time.sleep(0.0005)

    spin(1.5)
    win.prof = defaultdict(list)
    win.stream.build_s = []
    spin(seconds)
    prof_all["new"] = (win.prof, list(win.stream.build_s))

    win.prof = defaultdict(list)
    win._compare_today = True
    spin(seconds)
    prof_all["today"] = (win.prof, [])
    win._compare_today = False

    def line(name, a):
        a = np.array(a[5:] if len(a) > 5 else a)
        if a.size:
            print(f"  {name:26s} n={a.size:5d}  median {np.median(a):7.3f}  mean {a.mean():7.3f}  "
                  f"p95 {np.percentile(a, 95):7.3f}  max {a.max():7.3f} ms")

    dpr = win.devicePixelRatioF()
    print(f"platform {QApplication.platformName()}, window {win.width()} x {win.height()}, dpr {dpr}, "
          f"panel {win.view.panel_view1._side} device px, crop {win.view.panel_live.roi}")
    print("--- new path, GUI thread ---")
    prof, builds = prof_all["new"]
    for name in sorted(prof):
        line(name, prof[name])
    n = max(1, len(prof["frame logic"]))
    busy = sum(prof["frame logic"]) + sum(prof["repaint + flush"])
    print(f"  {'GUI thread per frame':26s} mean {busy / n:7.3f} ms (frame logic + repaint outside it)")
    print("--- new path, tracking thread (not the GUI's time) ---")
    line("MarkBuilder.build", [b * 1e3 for b in builds])
    print("--- today, same load ---")
    prof, _ = prof_all["today"]
    for name in sorted(prof):
        if name.startswith("today"):
            line(name, prof[name])


def _parse_size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", nargs="?", default=str(SNAPSHOTS["B_2 (drill inclined)"]))
    ap.add_argument("--tracking", default=None, help="<label>_tracking.npz (default: next to the snapshot)")
    ap.add_argument("--size", default=None, help="window size, e.g. 1920x1080 (default: maximised)")
    ap.add_argument("--grab", default=None, help="save a picture of the window after a moment and exit")
    ap.add_argument("--bench", type=float, default=0.0, help="seconds per phase; prints stage times and exits")
    ap.add_argument("--no-load", action="store_true", help="no stand-in for the tracking engine")
    ap.add_argument("--readout-hz", type=float, default=0.0, help="read-outs per second (default: 5)")
    ap.add_argument("--gui-priority", action="store_true", help="GUI thread above normal priority")
    ap.add_argument("--cv-threads", type=int, default=0, help="cv2.setNumThreads for the stand-in")
    ap.add_argument("--move", default=None, help="advance,x,y,z,turn,tilt: virtual move at the start")
    ap.add_argument("--zero", action="store_true", help="Null setzen before the virtual move")
    ap.add_argument("--ruler", default=None, choices=RULER_MODES)
    ap.add_argument("--view", type=int, default=VIEW_LIVE, choices=(VIEW_LIVE, VIEW_XRAY))
    args = ap.parse_args(argv)

    # the offscreen platform has no font database of its own
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
    if args.cv_threads:
        cv2.setNumThreads(args.cv_threads)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    if args.gui_priority:
        QThread.currentThread().setPriority(QThread.TimeCriticalPriority)
    snap = Snapshot(Path(args.snapshot), Path(args.tracking) if args.tracking else None)
    win = DesignWindow(snap, load=not args.no_load)
    if args.readout_hz > 0.0:
        win.readout_period_s = 1.0 / args.readout_hz
    win.view.set_view(args.view)
    for name, folder in SNAPSHOTS.items():
        if folder.resolve() == snap.folder.resolve():
            win.cmb_snapshot.blockSignals(True)
            win.cmb_snapshot.setCurrentText(name)
            win.cmb_snapshot.blockSignals(False)
    if args.size:
        win.resize(*_parse_size(args.size))
        win.show()
    elif args.grab or args.bench:
        win.resize(1920, 1080)
        win.show()
    else:
        win.showMaximized()

    def spin(t: float) -> None:
        t_end = time.perf_counter() + t
        while time.perf_counter() < t_end:
            app.processEvents()
            time.sleep(0.001)

    if args.zero:
        spin(0.4)
        win.set_zero()
    if args.move:
        values = [float(v) for v in args.move.split(",")]
        for s, value in zip(win._move_sliders, values):
            s.set(value)
    if args.ruler:
        win.builder.set_ruler(args.ruler, win.builder.zero_cam)

    if args.bench:
        _run_bench(app, win, args.bench)
        win.close()
        return 0
    if args.grab:
        spin(1.6)
        win.grab().save(args.grab)
        win.close()
        print(f"saved {args.grab}")
        return 0
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
