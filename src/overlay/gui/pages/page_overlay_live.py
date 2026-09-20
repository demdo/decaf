from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, QRect
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from overlay.gui.state import SessionState
from overlay.tools.homography import estimate_plane_induced_homography
from overlay.tools.tracking_overlay import TrackingOverlay
from overlay.tools.warp import prepare_live_overlay

# The camera is owned by the tracking hardware boundary so the tuned
# RealSense options (auto_exposure_priority=0 for guaranteed fps, emitter
# off for the IR pair, ...) live in exactly one place. The tracker itself
# (HydraTracker, C++ engine) is imported lazily in _start_tracking so the
# GUI starts and the overlay runs even when the tracking build is absent.
from tracking.hydramarker.camera_setup import RealSenseCameraSource
from tracking.hydramarker.config import CameraConfig


# ============================================================
# Frame profiling
# ============================================================

PROFILE_LIVE = True
PROFILE_EVERY = 30          # frames between summary lines

# The profile log is only written in Evaluation mode, into the session
# folder (live_profile.log); in Normal mode nothing is written.
PROFILE_LOG_NAME = "live_profile.log"

# Presentation snapshots (Save snapshot button): one timestamped folder per
# press with pixel-aligned PNG layers of the current live frame. Normal
# mode: overlay/data/snapshots/; Evaluation mode: <label>_live/ in the
# session folder.
SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "data" / "snapshots"


# ============================================================
# Tracking failure log
# ============================================================

# Aggregated dropout causes per tracking run, written as one JSONL record
# when tracking stops. Costs nothing in the live loop: fresh-pose frames do
# no logging work at all, stale frames only update an in-memory dict (and
# take a one-time detail snapshot per distinct cause); the file write
# happens exclusively in _stop_tracking, outside the steady-state path.
# On/off switch: TrackerConfig.failure_log_enabled (set in
# run_tracker.make_live_tracker_config, like every other tracking setting);
# the file itself only exists in Evaluation mode (session folder).
TRACK_FAILURE_LOG_NAME = "tracking_failure_log.jsonl"


class _TrackFailureLog:
    """Dropout causes of one tracking run, deduplicated by (stage, reason).

    note() is only called for stale frames AFTER the first fresh pose
    (acquisition failures are not logged). Repeats of the same cause bump a
    counter instead of producing new entries; the full diagnostic snapshot
    is taken once, at the first occurrence of each cause.
    """

    def __init__(self, path: str):
        self._path: Path | None = Path(path).resolve()
        self._tracker_log = None            # tracking.hydramarker.tracker_log
        self._causes: dict[tuple[str, str], dict] = {}
        self._t0 = 0.0

    def start(self, tracker_log_module) -> None:
        """Arm for a new tracking run (called from _start_tracking)."""
        self._tracker_log = tracker_log_module
        self._causes = {}
        self._t0 = time.perf_counter()

    def note(self, result, frame_idx: int, stale: int) -> None:
        """Record one stale frame. Dict-only work; never raises."""
        try:
            tl = self._tracker_log
            if tl is None:
                return
            if getattr(result, "success", False):
                # Held pose: success without a fresh pose is its own cause,
                # classify_failure would report it as "OK".
                key = ("POSE_HELD", tl.pose_source_value(result))
            else:
                key = tl.classify_failure(result)

            entry = self._causes.get(key)
            if entry is not None:
                entry["count"] += 1
                entry["last_frame"] = frame_idx
                entry["last_t_s"] = round(time.perf_counter() - self._t0, 3)
                return

            # First occurrence: snapshot everything needed to diagnose it.
            self._causes[key] = {
                "stage": key[0],
                "reason": key[1],
                "count": 1,
                "first_frame": frame_idx,
                "last_frame": frame_idx,
                "first_t_s": round(time.perf_counter() - self._t0, 3),
                "last_t_s": round(time.perf_counter() - self._t0, 3),
                "first": {
                    "message": str(getattr(result, "message", "") or ""),
                    "mode": getattr(getattr(result, "mode", None), "value", None),
                    "pose_source": tl.pose_source_value(result),
                    "stale_frames": stale,
                    "num_points": getattr(result, "num_points", None),
                    "num_inliers": getattr(result, "num_inliers", None),
                    "mean_reproj_px": getattr(
                        result, "mean_reprojection_error_px", None
                    ),
                    "max_reproj_px": getattr(
                        result, "max_reprojection_error_px", None
                    ),
                    "n_corners": len(getattr(result, "corners", ()) or ()),
                    "n_detection_corners": len(
                        getattr(result, "detection_corners", ()) or ()
                    ),
                    "debug_counters": dict(
                        getattr(result, "debug_counters", {}) or {}
                    ),
                },
            }
        except Exception:
            pass

    def flush(self, stop_reason: str, total_frames: int) -> None:
        """Write one JSONL record if any dropout was recorded. Never raises."""
        causes, self._causes = self._causes, {}
        if not causes or self._path is None:
            return
        try:
            record = {
                "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "event": "tracking_dropout",
                "stop_reason": stop_reason,
                "frames_total": total_frames,
                "run_duration_s": round(time.perf_counter() - self._t0, 3),
                "causes": sorted(
                    causes.values(), key=lambda c: c["first_frame"]
                ),
            }
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            self._path = None


class _FrameProfile:
    """
    Per-stage timing for the live loop.

    Prints one summary line every PROFILE_EVERY frames with mean and max per
    stage in ms. 'interval' is the wall time between two timer callbacks - if
    that is well above 1000/FPS while the stages are cheap, the timer itself is
    being starved rather than the processing being slow.
    """

    def __init__(self, fps: int, log_path: Path | None):
        self.budget_ms = 1000.0 / float(fps)
        self.log_path = None if log_path is None else Path(log_path).resolve()
        self.reset()

    def emit(self, line: str) -> None:
        """Write one line to the log file. Never raises - the live loop wins."""
        if self.log_path is None:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            self.log_path = None

    def reset(self) -> None:
        self._t: dict[str, list[float]] = defaultdict(list)
        self._n_frames = 0
        self._n_polls = 0
        self._n_empty = 0
        self._t_last_start: float | None = None
        self._t_window0 = time.perf_counter()

    def note_poll(self, empty: bool) -> None:
        self._n_polls += 1
        if empty:
            self._n_empty += 1

    def frame_start(self) -> float:
        now = time.perf_counter()
        if self._t_last_start is not None:
            self._t[" interval"].append((now - self._t_last_start) * 1000.0)
        self._t_last_start = now
        return now

    def add(self, stage: str, seconds: float) -> None:
        self._t[stage].append(seconds * 1000.0)

    def frame_end(self, t0: float) -> None:
        self._t["total"].append((time.perf_counter() - t0) * 1000.0)
        self._n_frames += 1
        if self._n_frames >= PROFILE_EVERY:
            self._report()
            self.reset()

    def _report(self) -> None:
        elapsed = time.perf_counter() - self._t_window0
        fps = self._n_frames / elapsed if elapsed > 0 else 0.0

        parts = []
        for stage in sorted(self._t):
            v = self._t[stage]
            if v:
                parts.append(f"{stage.strip()} {np.mean(v):6.1f}/{np.max(v):6.1f}")

        empty_pct = 100.0 * self._n_empty / max(1, self._n_polls)
        self.emit(
            f"[live] {time.strftime('%H:%M:%S')}  {fps:5.1f} fps "
            f"(budget {self.budget_ms:.1f} ms)  polls empty {empty_pct:4.0f}%  "
            "|  mean/max ms: " + "  ".join(parts)
        )


# ============================================================
# Helpers
# ============================================================

def _draw_point_inplace(
    img: np.ndarray,
    uv: np.ndarray,
    *,
    color=(0, 0, 255, 255),
    radius=8,
    cross_size=20,
    thickness=2,
) -> None:
    """
    Draw the tip directly into the frame buffer.

    The previous version copied the whole 1920x1080 frame first, which cost
    1.9 ms for a marker of a few hundred pixels.
    """
    if uv is None:
        return

    uv = np.asarray(uv, dtype=np.float64).reshape(2)
    if not np.isfinite(uv).all():
        return

    u, v = np.round(uv).astype(int)

    cv2.circle(img, (u, v), radius, color, thickness, cv2.LINE_AA)
    cv2.drawMarker(
        img,
        (u, v),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=cross_size,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )


def _bgra_to_qpixmap(img_bgra: np.ndarray) -> QPixmap:
    """
    Wrap a BGRA buffer without converting or copying anything.

    Format_RGB32 IS Qt's native 32-bit layout on little-endian - BGRA with the
    alpha ignored - so QPixmap.fromImage degrades to a reference and costs
    ~0.01 ms. The neighbouring choices are traps: Format_BGR888 forces a
    per-pixel expansion (1.75 ms) and Format_ARGB32 forces a premultiply
    (1.60 ms). Both are two hundred times more expensive for the same picture.

    The caller must keep img_bgra alive while the pixmap is in use.
    """
    if img_bgra is None:
        return QPixmap()

    h, w = img_bgra.shape[:2]
    qimg = QImage(img_bgra.data, w, h, 4 * w, QImage.Format_RGB32)
    return QPixmap.fromImage(qimg)


# ============================================================
# Professional sidebar — scaled 60%
# ============================================================

class _OverlayControlRail(QFrame):
    """
    Right-side live overlay controls.

    - Starts collapsed.
    - Can be opened/closed repeatedly.
    - Professional dark translucent sidebar.
    - Overall size scaled to 60%.
    """

    SCALE = 0.60

    HANDLE_W = int(round(34 * SCALE))
    PANEL_W = int(round(380 * SCALE))
    PANEL_H = int(round(452 * SCALE))
    # Extra panel height while the tool-tracking card is visible.
    TRACK_EXTRA_H = int(round(150 * SCALE))
    GAP = int(round(10 * SCALE))

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._expanded = False

        self.setObjectName("overlayControlRail")
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_StyledBackground, True)

        # ---------------- Handle ----------------
        self.handle = QToolButton(self)
        self.handle.setObjectName("overlayRailHandle")
        self.handle.setText("‹")
        self.handle.setCursor(Qt.PointingHandCursor)
        self.handle.setToolTip("Show overlay controls")
        self.handle.clicked.connect(self._toggle)

        # ---------------- Panel ----------------
        self.panel = QFrame(self)
        self.panel.setObjectName("overlayRailPanel")
        self.panel.setFrameShape(QFrame.NoFrame)
        self.panel.setAttribute(Qt.WA_StyledBackground, True)

        self.title_label = QLabel("Overlay Controls")
        self.title_label.setObjectName("overlayRailTitle")
        self.title_label.setFixedWidth(self._s(230))

        # ---------------- Alpha ----------------
        self.lbl_alpha = QLabel("Alpha")
        self.lbl_alpha.setObjectName("overlayRailLabel")

        self.lbl_alpha_value = QLabel("0.50")
        self.lbl_alpha_value.setObjectName("overlayRailValue")
        self.lbl_alpha_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        alpha_head = QHBoxLayout()
        alpha_head.setContentsMargins(0, 0, 0, 0)
        alpha_head.setSpacing(self._s(12))
        alpha_head.addWidget(self.lbl_alpha)
        alpha_head.addStretch(1)
        alpha_head.addWidget(self.lbl_alpha_value)

        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setObjectName("overlayRailSlider")
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(50)
        self.slider_alpha.setSingleStep(1)
        self.slider_alpha.setPageStep(10)
        self.slider_alpha.setTracking(True)
        self.slider_alpha.setMinimumHeight(self._s(28))

        # ---------------- d_x ----------------
        self.lbl_dx = QLabel("Plane depth")
        self.lbl_dx.setObjectName("overlayRailLabel")

        self.lbl_dx_value = QLabel("0.0")
        self.lbl_dx_value.setObjectName("overlayRailValue")
        self.lbl_dx_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        dx_head = QHBoxLayout()
        dx_head.setContentsMargins(0, 0, 0, 0)
        dx_head.setSpacing(self._s(12))
        dx_head.addWidget(self.lbl_dx)
        dx_head.addStretch(1)
        dx_head.addWidget(self.lbl_dx_value)

        self.slider_dx = QSlider(Qt.Horizontal)
        self.slider_dx.setObjectName("overlayRailSlider")
        self.slider_dx.setRange(-50, 50)
        self.slider_dx.setValue(0)
        self.slider_dx.setSingleStep(1)
        self.slider_dx.setPageStep(5)
        self.slider_dx.setTracking(True)
        self.slider_dx.setMinimumHeight(self._s(28))

        # ---------------- Tip / axis ----------------
        self.chk_show_tip = QCheckBox("Show pointer tip")
        self.chk_show_tip.setObjectName("overlayRailCheck")
        self.chk_show_tip.setChecked(True)
        self.chk_show_tip.setMinimumHeight(self._s(26))

        self.chk_show_axis = QCheckBox("Show drill axis")
        self.chk_show_axis.setObjectName("overlayRailCheck")
        self.chk_show_axis.setChecked(True)
        self.chk_show_axis.setMinimumHeight(self._s(26))

        # ---------------- Tool tracking ----------------
        # Only shown when a tool is loaded (set_tracking_visible). Tracking
        # is started explicitly via the button; on loss it stops and the
        # button re-arms (no auto re-acquire).
        self.btn_track = QPushButton("Start tracking")
        self.btn_track.setObjectName("overlayRailTrackBtn")
        self.btn_track.setMinimumHeight(self._s(40))

        self.lbl_track_status = QLabel("idle")
        self.lbl_track_status.setObjectName("overlayRailValue")
        self.lbl_track_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # ---------------- Cards ----------------
        alpha_card = QFrame()
        alpha_card.setObjectName("overlayRailGroup")
        alpha_card.setAttribute(Qt.WA_StyledBackground, True)
        alpha_layout = QVBoxLayout(alpha_card)
        alpha_layout.setContentsMargins(
            self._s(18),
            self._s(14),
            self._s(18),
            self._s(14),
        )
        alpha_layout.setSpacing(self._s(10))
        alpha_layout.addLayout(alpha_head)
        alpha_layout.addWidget(self.slider_alpha)

        dx_card = QFrame()
        dx_card.setObjectName("overlayRailGroup")
        dx_card.setAttribute(Qt.WA_StyledBackground, True)
        dx_layout = QVBoxLayout(dx_card)
        dx_layout.setContentsMargins(
            self._s(18),
            self._s(14),
            self._s(18),
            self._s(14),
        )
        dx_layout.setSpacing(self._s(10))
        dx_layout.addLayout(dx_head)
        dx_layout.addWidget(self.slider_dx)

        tip_card = QFrame()
        tip_card.setObjectName("overlayRailGroup")
        tip_card.setAttribute(Qt.WA_StyledBackground, True)
        tip_layout = QVBoxLayout(tip_card)
        tip_layout.setContentsMargins(
            self._s(18),
            self._s(13),
            self._s(18),
            self._s(13),
        )
        tip_layout.setSpacing(self._s(8))
        tip_layout.addWidget(self.chk_show_tip)
        tip_layout.addWidget(self.chk_show_axis)

        # ---------------- Snapshot ----------------
        # Saves pixel-aligned presentation layers of the current frame
        # (see OverlayLivePage._on_save_snapshot).
        self.btn_save = QPushButton("Save snapshot")
        self.btn_save.setObjectName("overlayRailTrackBtn")
        self.btn_save.setMinimumHeight(self._s(40))

        save_card = QFrame()
        save_card.setObjectName("overlayRailGroup")
        save_card.setAttribute(Qt.WA_StyledBackground, True)
        save_layout = QVBoxLayout(save_card)
        save_layout.setContentsMargins(
            self._s(18),
            self._s(13),
            self._s(18),
            self._s(13),
        )
        save_layout.setSpacing(self._s(8))
        save_layout.addWidget(self.btn_save)

        self.track_card = QFrame()
        self.track_card.setObjectName("overlayRailGroup")
        self.track_card.setAttribute(Qt.WA_StyledBackground, True)
        track_layout = QVBoxLayout(self.track_card)
        track_layout.setContentsMargins(
            self._s(18),
            self._s(13),
            self._s(18),
            self._s(13),
        )
        track_layout.setSpacing(self._s(8))
        track_layout.addWidget(self.btn_track)
        track_layout.addWidget(self.lbl_track_status)
        self.track_card.hide()
        self._tracking_visible = False

        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(
            self._s(22),
            self._s(22),
            self._s(22),
            self._s(22),
        )
        panel_layout.setSpacing(self._s(16))
        panel_layout.addWidget(self.title_label, 0, Qt.AlignLeft)
        panel_layout.addWidget(alpha_card)
        panel_layout.addWidget(dx_card)
        panel_layout.addWidget(tip_card)
        panel_layout.addWidget(save_card)
        panel_layout.addWidget(self.track_card)
        panel_layout.addStretch(1)

        self._apply_style()
        self._update_geometry()

    def _s(self, value: float) -> int:
        return max(1, int(round(float(value) * self.SCALE)))

    def set_tracking_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible == self._tracking_visible:
            return
        self._tracking_visible = visible
        self.track_card.setVisible(visible)
        self._update_geometry()

        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "_position_controls"):
            parent._position_controls()

    def _panel_height(self) -> int:
        return self.PANEL_H + (self.TRACK_EXTRA_H if self._tracking_visible else 0)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self.handle.setText("›" if self._expanded else "‹")
        self.handle.setToolTip(
            "Hide overlay controls" if self._expanded else "Show overlay controls"
        )
        self._update_geometry()

        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "_position_controls"):
            parent._position_controls()

    def _update_geometry(self) -> None:
        handle_h = self._s(86)
        handle_y = 0

        if self._expanded:
            panel_h = self._panel_height()
            total_w = self.PANEL_W + self.GAP + self.HANDLE_W
            total_h = panel_h
            self.setFixedSize(total_w, total_h)

            self.panel.show()
            self.panel.setGeometry(0, 0, self.PANEL_W, panel_h)
            self.handle.setGeometry(
                self.PANEL_W + self.GAP,
                handle_y,
                self.HANDLE_W,
                handle_h,
            )
        else:
            self.panel.hide()
            self.setFixedSize(self.HANDLE_W, handle_h)
            self.handle.setGeometry(0, 0, self.HANDLE_W, handle_h)

    def _apply_style(self) -> None:
        border_radius_panel = self._s(22)
        border_radius_handle = self._s(17)
        border_radius_group = self._s(16)
        checkbox_radius = self._s(5)

        font_title = self._s(22)
        font_label = self._s(14)
        font_handle = self._s(28)

        slider_groove_h = self._s(7)
        slider_handle = self._s(18)
        slider_handle_margin = -self._s(6)

        checkbox_size = self._s(18)

        self.setStyleSheet(
            f"""
            QFrame#overlayControlRail {{
                background: transparent;
                border: none;
            }}

            QFrame#overlayRailPanel {{
                background-color: rgba(18, 24, 38, 232);
                border: 1px solid rgba(255, 255, 255, 44);
                border-radius: {border_radius_panel}px;
            }}

            QToolButton#overlayRailHandle {{
                background-color: rgba(18, 24, 38, 226);
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: {border_radius_handle}px;
                color: #f8fafc;
                font-size: {font_handle}px;
                font-weight: 700;
                padding: 0px;
            }}

            QToolButton#overlayRailHandle:hover {{
                background-color: rgba(30, 41, 59, 242);
                border: 1px solid rgba(96, 165, 250, 150);
                color: #ffffff;
            }}

            QLabel#overlayRailTitle {{
                background: transparent;
                color: #f8fafc;
                font-size: {font_title}px;
                font-weight: 800;
                min-height: {self._s(28)}px;
            }}

            QFrame#overlayRailGroup {{
                background-color: rgba(255, 255, 255, 24);
                border: 1px solid rgba(255, 255, 255, 34);
                border-radius: {border_radius_group}px;
            }}

            QLabel#overlayRailLabel {{
                background: transparent;
                color: #e2e8f0;
                font-size: {font_label}px;
                font-weight: 700;
                min-height: {self._s(22)}px;
            }}

            QLabel#overlayRailValue {{
                background: transparent;
                color: #ffffff;
                font-size: {font_label}px;
                font-weight: 800;
                min-width: {self._s(86)}px;
                min-height: {self._s(22)}px;
            }}

            QCheckBox#overlayRailCheck {{
                background: transparent;
                color: #ffffff;
                font-size: {font_label}px;
                font-weight: 700;
                spacing: {self._s(10)}px;
            }}

            QCheckBox#overlayRailCheck::indicator {{
                width: {checkbox_size}px;
                height: {checkbox_size}px;
                border-radius: {checkbox_radius}px;
                border: 1px solid rgba(255, 255, 255, 95);
                background-color: rgba(15, 23, 42, 180);
            }}

            QCheckBox#overlayRailCheck::indicator:checked {{
                background-color: #3b82f6;
                border: 1px solid #93c5fd;
            }}

            QSlider#overlayRailSlider {{
                background: transparent;
                min-height: {self._s(28)}px;
            }}

            QSlider#overlayRailSlider::groove:horizontal {{
                height: {slider_groove_h}px;
                background-color: rgba(148, 163, 184, 80);
                border-radius: {max(1, slider_groove_h // 2)}px;
            }}

            QSlider#overlayRailSlider::sub-page:horizontal {{
                height: {slider_groove_h}px;
                background-color: #3b82f6;
                border-radius: {max(1, slider_groove_h // 2)}px;
            }}

            QSlider#overlayRailSlider::add-page:horizontal {{
                height: {slider_groove_h}px;
                background-color: rgba(148, 163, 184, 80);
                border-radius: {max(1, slider_groove_h // 2)}px;
            }}

            QSlider#overlayRailSlider::handle:horizontal {{
                width: {slider_handle}px;
                height: {slider_handle}px;
                margin: {slider_handle_margin}px 0px;
                border-radius: {max(1, slider_handle // 2)}px;
                background-color: #ffffff;
                border: {self._s(3)}px solid #3b82f6;
            }}

            QSlider#overlayRailSlider::handle:horizontal:hover {{
                border: {self._s(3)}px solid #60a5fa;
            }}

            QPushButton#overlayRailTrackBtn {{
                background-color: #3b82f6;
                color: #ffffff;
                border: none;
                border-radius: {self._s(10)}px;
                font-size: {font_label}px;
                font-weight: 700;
                padding: {self._s(8)}px;
            }}

            QPushButton#overlayRailTrackBtn:hover {{
                background-color: #2563eb;
            }}

            QPushButton#overlayRailTrackBtn:disabled {{
                background-color: rgba(148, 163, 184, 120);
                color: rgba(255, 255, 255, 160);
            }}
            """
        )


# ============================================================
# Page
# ============================================================

class OverlayLivePage(QWidget):
    FPS = 30
    POLL_OVERSAMPLE = 5          # timer ticks per camera frame
    COLOR_SIZE = (1920, 1080)
    ALPHA_SLIDER_MAX = 100
    DX_OFFSET_MIN_MM = -10
    DX_OFFSET_MAX_MM = 10

    # Tool tracking (button-driven: started explicitly, stops on loss, no
    # auto re-acquire). Frame counts at 30 fps; the timeouts match the
    # validated run_tracker values (ACQUIRE_TIMEOUT_FRAMES,
    # TRACKING_STALE_TO_IDLE_FRAMES).
    TRACK_ACQUIRE_TIMEOUT_FRAMES = 90   # 3 s without a first pose -> stop
    TRACK_LOST_STOP_FRAMES = 45         # 1.5 s stale while tracking -> stop
    # Manual IR exposure while the IR pair streams (same default the
    # run_tracker IR path uses; short = less motion blur on the pair).
    TRACK_IR_EXPOSURE_US = 8000.0

    def __init__(self, state: SessionState, on_complete_changed=None, parent=None):
        super().__init__(parent)

        self.state = state
        self.on_complete_changed = on_complete_changed

        self.camera: RealSenseCameraSource | None = None
        self.timer: QTimer | None = None

        # --- tool tracking runtime state ---
        self._tool = None                  # ToolAssets while on the page
        self._tracker = None               # HydraTracker while tracking
        self._tracking_active = False
        self._track_locked = False         # False = acquiring, True = had a pose
        self._track_converged = False      # pose output filter warmup latch
        self._track_frames = 0             # frames since tracking start
        self._track_stale = 0              # consecutive frames without fresh pose
        self._tool_overlay: TrackingOverlay | None = None
        self._last_pose_result = None      # last TrackerResult with a fresh pose
        self._has_fresh_pose = None        # tracker_log.has_fresh_pose (lazy)
        self._tracker_log_module = None    # tracking.hydramarker.tracker_log (lazy)
        # Created in _make_tracker iff TrackerConfig.failure_log_enabled.
        self._track_fail_log: _TrackFailureLog | None = None

        self._live_color: np.ndarray | None = None
        self._display_bgra: np.ndarray | None = None

        self._xray_gray_u8: np.ndarray | None = None
        self._xray_working_u8: np.ndarray | None = None
        self._overlay_cache = None
        self._frame_buf: np.ndarray | None = None
        self._scaled_buf: np.ndarray | None = None

        self._alpha: float = 0.50
        self._show_tip: bool = True
        self._show_axis: bool = True

        self._base_d_x_mm: float | None = None
        self._current_d_x_mm: float | None = None
        self._dx_offset_mm: int = 0

        self._H_xc_lookup: dict[int, np.ndarray] = {}

        # Profiling lines go to a file only in Evaluation mode (see
        # PROFILE_LOG_NAME); the session is attached later, so the profile
        # is created in on_enter.
        self._profile: _FrameProfile | None = None

        self._build_ui()
        self._connect_signals()
        self._load_xray_from_state()
        self._initialize_dx()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _build_ui(self) -> None:
        self.setObjectName("overlayLivePage")
        self.setFocusPolicy(Qt.StrongFocus)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: black;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.controls = _OverlayControlRail(self)

        self.setStyleSheet(
            """
            QWidget#overlayLivePage {
                background: black;
            }
            """
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        self.image_label.setGeometry(self.rect())
        self._position_controls()
        self._update_image_label()

    def _position_controls(self) -> None:
        margin_top = 18
        margin_right = 18
        size = self.controls.size()

        x = self.width() - size.width() - margin_right - 8
        y = margin_top

        self.controls.setGeometry(QRect(x, y, size.width(), size.height()))
        self.controls.raise_()

    def _connect_signals(self) -> None:
        self.controls.slider_alpha.valueChanged.connect(self._on_alpha_changed)
        self.controls.slider_dx.valueChanged.connect(self._on_dx_offset_changed)
        self.controls.chk_show_tip.toggled.connect(self._on_show_tip_toggled)
        self.controls.chk_show_axis.toggled.connect(self._on_show_axis_toggled)
        self.controls.btn_track.clicked.connect(self._on_track_button)
        self.controls.btn_save.clicked.connect(self._on_save_snapshot)

    # --------------------------------------------------------
    # State / prerequisites
    # --------------------------------------------------------

    def _has_prerequisites(self) -> bool:
        return (
            self.state.K_rgb is not None
            and self.state.K_xray is not None
            and self.state.T_xc is not None
            and self.state.d_x is not None
            and self.state.tip_uv_c is not None
        )

    def _load_xray_from_state(self) -> None:
        if self.state.xray_image_anatomy is None:
            self._xray_gray_u8 = None
            self._xray_working_u8 = None
            return

        img = np.asarray(self.state.xray_image_anatomy)

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Keep stored image RAW. It is converted to the X-ray working image space
        # only immediately before warping, so H_xc remains unchanged.
        self._xray_gray_u8 = img

        # Flip and detector undistortion are frame independent, so they happen
        # once here instead of per frame. Same treatment as the preview page:
        # the pose lives in the ideal X-ray frame, so the image must too.
        working = np.ascontiguousarray(cv2.flip(img, 1))
        field = getattr(self.state, "xray_dist_field", None)
        if field is not None:
            working = field.undistort_image_working(working)
        self._xray_working_u8 = working

    def _initialize_dx(self) -> None:
        if self.state.d_x is None:
            self._base_d_x_mm = None
            self._current_d_x_mm = None
            self._dx_offset_mm = 0
            self._H_xc_lookup = {}
            self.controls.lbl_dx_value.setText("-")
            return

        self._base_d_x_mm = float(self.state.d_x)
        self._current_d_x_mm = float(self.state.d_x)
        self._dx_offset_mm = 0

        self.controls.slider_dx.blockSignals(True)
        self.controls.slider_dx.setRange(self.DX_OFFSET_MIN_MM, self.DX_OFFSET_MAX_MM)
        self.controls.slider_dx.setValue(0)
        self.controls.slider_dx.setSingleStep(1)
        self.controls.slider_dx.setPageStep(5)
        self.controls.slider_dx.blockSignals(False)

        self.controls.lbl_dx_value.setText(f"{self._current_d_x_mm:.1f} mm")
        self.controls.lbl_alpha_value.setText(f"{self._alpha:.2f}")

    # --------------------------------------------------------
    # Homography
    # --------------------------------------------------------

    def _precompute_H_xc_lookup(self) -> None:
        self._H_xc_lookup = {}

        if not self._has_prerequisites():
            return

        if self._base_d_x_mm is None:
            return

        T_xc = np.asarray(self.state.T_xc, dtype=np.float64)
        R_xc = T_xc[:3, :3]
        t_xc = T_xc[:3, 3]

        K_c = np.asarray(self.state.K_rgb, dtype=np.float64)
        K_x = np.asarray(self.state.K_xray, dtype=np.float64)

        for dx_offset_mm in range(self.DX_OFFSET_MIN_MM, self.DX_OFFSET_MAX_MM + 1):
            d_x_mm = float(self._base_d_x_mm + dx_offset_mm)

            H_xc = estimate_plane_induced_homography(
                K_c=K_c,
                R_xc=R_xc,
                t_xc=t_xc,
                K_x=K_x,
                d_x=d_x_mm,
            )
            self._H_xc_lookup[int(dx_offset_mm)] = H_xc

    def _get_current_H_xc(self) -> np.ndarray:
        if not self._H_xc_lookup:
            raise RuntimeError("H_xc lookup is empty.")

        if self._dx_offset_mm not in self._H_xc_lookup:
            raise RuntimeError(
                f"No precomputed H_xc for d_x offset {self._dx_offset_mm} mm."
            )

        return self._H_xc_lookup[self._dx_offset_mm]

    def _invalidate_overlay_cache(self) -> None:
        self._overlay_cache = None
        self._frame_buf = None
        self.state.H_xc = None

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    def _compose_frame(self, frame_bgra: np.ndarray) -> np.ndarray:
        """
        Build the display frame in a persistent buffer.

        Nothing here allocates and nothing touches the full frame except the
        one copy of the camera image, which is a plain memcpy.
        """
        prof = self._profile

        if self._frame_buf is None or self._frame_buf.shape != frame_bgra.shape:
            self._frame_buf = np.empty_like(frame_bgra)

        t = time.perf_counter()
        np.copyto(self._frame_buf, frame_bgra)
        if prof:
            prof.add("copy", time.perf_counter() - t)

        if self._xray_gray_u8 is not None:
            H_xc = self._get_current_H_xc()
            self.state.H_xc = H_xc

            if self._overlay_cache is None:
                # H_xc/K_xray are defined in XRAY_WORKING_FLIPPED_UV space, so
                # the RAW X-ray image goes into that space before warping.
                # H_xc itself is never modified.
                t = time.perf_counter()
                h, w = frame_bgra.shape[:2]
                # K_rgb/dist_rgb: the warp routes through the lens model so
                # the X-ray lands where the observed frame shows the anatomy,
                # consistent with the tracker's tip/axis projection.
                self._overlay_cache = prepare_live_overlay(
                    xray_gray_u8=self._xray_working_u8,
                    H_xc=H_xc,
                    camera_size_wh=(w, h),
                    K_camera=self.state.K_rgb,
                    dist_coeffs=self.state.dist_rgb,
                )
                if prof:
                    prof.add("prepare*", time.perf_counter() - t)

            if self._overlay_cache is not None:
                t = time.perf_counter()
                self._overlay_cache.blend_into(self._frame_buf, self._alpha)
                if prof:
                    prof.add("blend", time.perf_counter() - t)

        if self._show_tip:
            t = time.perf_counter()
            _draw_point_inplace(self._frame_buf, self.state.tip_uv_c)
            if prof:
                prof.add("tip", time.perf_counter() - t)

        if self._tool_overlay is not None:
            t = time.perf_counter()
            self._tool_overlay.draw_into(
                self._frame_buf, show_tip=True, show_axis=self._show_axis
            )
            if prof:
                prof.add("drill", time.perf_counter() - t)

        return self._frame_buf

    # --------------------------------------------------------
    # RealSense
    # --------------------------------------------------------

    def _start_realsense(self) -> None:
        if self.camera is not None:
            return

        # bgra8, not bgr8: this is Qt's native 32-bit layout, so the frame
        # goes to the screen without a single per-pixel conversion. The IR
        # pair only streams when a tool is loaded; enabling it also switches
        # the emitter/laser off inside the camera source (speckle would ruin
        # the checkerboard corners) — which is irrelevant here because this
        # page never uses depth.
        with_ir = self._tool is not None
        cfg = CameraConfig(
            width=self.COLOR_SIZE[0],
            height=self.COLOR_SIZE[1],
            fps=self.FPS,
            realsense_color_format="bgra8",
            realsense_frames_queue_size=1,
            realsense_enable_ir=with_ir,
            realsense_ir_exposure_us=self.TRACK_IR_EXPOSURE_US if with_ir else None,
        )
        camera = RealSenseCameraSource(cfg)
        camera.start()
        self.camera = camera

    def _stop_realsense(self) -> None:
        if self.camera is None:
            return

        try:
            self.camera.stop()
        except Exception:
            pass

        self.camera = None

    def _start_timer(self) -> None:
        if self.timer is None:
            self.timer = QTimer(self)
            self.timer.setTimerType(Qt.PreciseTimer)
            self.timer.timeout.connect(self._on_timer)

        # Poll several times per camera period instead of once.
        #
        # With one tick per frame the timer and the camera run at the same
        # nominal rate but drift against each other, so a missed tick costs a
        # whole period - measured 47 ms of interval for 2.9 ms of work, i.e.
        # 21 fps instead of 30. An empty poll costs 0.1 ms, so oversampling is
        # far cheaper than the beat it removes.
        interval_ms = max(1, int(round(1000.0 / (float(self.FPS) * self.POLL_OVERSAMPLE))))
        self.timer.start(interval_ms)

    def _stop_timer(self) -> None:
        if self.timer is not None:
            self.timer.stop()

    def _on_timer(self) -> None:
        if self.camera is None:
            return

        prof = self._profile

        try:
            t_poll = time.perf_counter()
            # Zero-copy views into the SDK buffers, valid until the next
            # poll() — everything below runs within this tick, and the
            # colour image is copied into the persistent frame buffer as
            # the first step of _compose_frame.
            frame = self.camera.poll()
            if frame is None:
                if prof:
                    prof.note_poll(empty=True)
                return

            img = frame.image_bgr  # BGRA (realsense_color_format="bgra8")

            t0 = prof.frame_start() if prof else 0.0
            if prof:
                prof.note_poll(empty=False)
                prof.add("poll", t0 - t_poll)

            if self._tracking_active:
                t = time.perf_counter()
                result = self._tracker.process_frame(
                    img,
                    run_detection=True,
                    ir_left=frame.ir_left,
                    ir_right=frame.ir_right,
                )
                if prof:
                    prof.add("track", time.perf_counter() - t)
                self._update_tracking_state(result)

            self._live_color = img
            self._display_bgra = self._compose_frame(img)
            self._update_image_label()

            if prof:
                prof.frame_end(t0)

        except Exception as e:
            self._stop_tracking("error")
            self._stop_timer()
            self._stop_realsense()
            QMessageBox.critical(
                self,
                "Live Overlay",
                f"Live update failed.\n\n{e}",
            )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    def _update_image_label(self) -> None:
        if self._display_bgra is None:
            return

        prof = self._profile

        # Fit the finished full-resolution frame into the window.
        #
        # The composition above stays at 1920x1080 - only this last step
        # reduces to what the widget can actually show, exactly as
        # pix.scaled() did before. Doing it with cv2 on the BGRA buffer costs
        # 0.5 ms instead of 1.5 ms, because the result is then already in Qt's
        # native layout and QPixmap.fromImage has nothing left to convert.
        t = time.perf_counter()
        src_h, src_w = self._display_bgra.shape[:2]
        lw = max(1, self.image_label.width())
        lh = max(1, self.image_label.height())

        # KeepAspectRatioByExpanding: cover the label, let it clip the overflow
        s = max(lw / src_w, lh / src_h)
        tw, th = max(1, int(round(src_w * s))), max(1, int(round(src_h * s)))

        if self._scaled_buf is None or self._scaled_buf.shape[:2] != (th, tw):
            self._scaled_buf = np.empty((th, tw, 4), dtype=np.uint8)

        cv2.resize(self._display_bgra, (tw, th), dst=self._scaled_buf,
                   interpolation=cv2.INTER_LINEAR)
        if prof:
            prof.add("scale", time.perf_counter() - t)

        t = time.perf_counter()
        pix = _bgra_to_qpixmap(self._scaled_buf)
        if not pix.isNull():
            self.image_label.setPixmap(pix)
        if prof:
            prof.add("qpixmap", time.perf_counter() - t)

    # --------------------------------------------------------
    # Tool tracking (button-driven)
    # --------------------------------------------------------
    #
    # Started explicitly via the rail button. No auto re-acquire: when the
    # marker is not found within the acquire timeout, or the pose goes
    # stale while tracking, tracking stops and the button re-arms.

    def _on_track_button(self) -> None:
        if self._tracking_active:
            self._stop_tracking("stopped")
        else:
            self._start_tracking()

    def _make_tracker(self):
        # Lazy imports: the C++ engine (.pyd) only loads when tracking is
        # actually started, so the overlay works without the tracking build.
        from tracking.hydramarker import tracker_log
        from tracking.hydramarker.run_tracker import (
            make_live_tracker_config,
            make_tracker,
        )

        self._has_fresh_pose = tracker_log.has_fresh_pose
        self._tracker_log_module = tracker_log

        # JSONL run logs (if ever enabled) must land in a fixed absolute
        # location, not in whatever cwd the GUI was started from: the
        # session folder in Evaluation mode, overlay/data/ otherwise.
        ev = self.state.eval
        log_root = (
            ev.dir if ev is not None
            else Path(__file__).resolve().parents[2] / "data"
        )
        tracker_log.set_run_log_dir(str(log_root / "hydramarker_tracker_runs"))

        if self._tracker is not None:
            self._tracker.reset()
            return self._tracker

        config = make_live_tracker_config()
        ir_path = getattr(self.state, "ir_calibration_path", None)
        config.ir_refine_enabled = bool(ir_path)

        if getattr(config, "failure_log_enabled", False):
            ev = self.state.eval
            self._track_fail_log = _TrackFailureLog(
                None if ev is None else str(ev.dir / TRACK_FAILURE_LOG_NAME)
            )
        else:
            self._track_fail_log = None

        tracker = make_tracker(
            self._tool.field_path,
            self._tool.marker_json_path,
            np.asarray(self.state.K_rgb, dtype=np.float64),
            self.state.dist_rgb,
            config,
        )
        if ir_path:
            tracker.set_ir_calibration(ir_path)
        return tracker

    def _start_tracking(self) -> None:
        if self._tracking_active or self._tool is None or self.camera is None:
            return

        try:
            self._tracker = self._make_tracker()
            if self._tool_overlay is None:
                self._tool_overlay = TrackingOverlay(
                    self.state.K_rgb,
                    self.state.dist_rgb,
                    self._tool.p_tip_mm,
                    self._tool.axis_dir,
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Tool Tracking",
                f"Could not start the tracker.\n\n{e}",
            )
            return

        self._tool_overlay.clear()
        if self._track_fail_log is not None:
            self._track_fail_log.start(self._tracker_log_module)
        self._tracking_active = True
        self._track_locked = False
        self._track_converged = False
        self._track_frames = 0
        self._track_stale = 0
        self.controls.btn_track.setText("Stop tracking")
        self.controls.lbl_track_status.setText("acquiring…")

    def _stop_tracking(self, reason: str) -> None:
        self._tracking_active = False
        if self._track_fail_log is not None:
            self._track_fail_log.flush(reason, self._track_frames)
        if self._tool_overlay is not None:
            self._tool_overlay.clear()
        self._last_pose_result = None
        if self._tracker is not None:
            try:
                self._tracker.reset()
            except Exception:
                pass
        self.controls.btn_track.setText("Start tracking")
        self.controls.lbl_track_status.setText(reason)

    def _update_tracking_state(self, result) -> None:
        self._track_frames += 1

        if self._has_fresh_pose(result):
            self._track_stale = 0
            converged = bool(getattr(result, "pose_converged", False))
            if not self._track_locked or converged != self._track_converged:
                self._track_locked = True
                self._track_converged = converged
                self.controls.lbl_track_status.setText(
                    "tracking" if converged else "tracking (settling…)"
                )
            self._tool_overlay.update(result)
            self._last_pose_result = result
            return

        self._track_stale += 1
        self._tool_overlay.clear()
        self._last_pose_result = None

        if not self._track_locked:
            if self._track_frames >= self.TRACK_ACQUIRE_TIMEOUT_FRAMES:
                self._stop_tracking("no marker found")
            return

        # Dropout after a successful lock: aggregate the cause (dedup by
        # stage/reason, counter on repeats). Acquisition failures above are
        # deliberately not logged.
        if self._track_fail_log is not None:
            self._track_fail_log.note(result, self._track_frames, self._track_stale)
        if self._track_stale >= self.TRACK_LOST_STOP_FRAMES:
            self._stop_tracking("lost — press Start")

    # --------------------------------------------------------
    # Controls callbacks
    # --------------------------------------------------------

    def _on_alpha_changed(self, value: int) -> None:
        self._alpha = float(value) / float(self.ALPHA_SLIDER_MAX)
        self.controls.lbl_alpha_value.setText(f"{self._alpha:.2f}")

    def _on_dx_offset_changed(self, value: int) -> None:
        if self._base_d_x_mm is None:
            return

        self._dx_offset_mm = int(value)
        self._current_d_x_mm = float(self._base_d_x_mm + self._dx_offset_mm)

        self.controls.lbl_dx_value.setText(f"{self._current_d_x_mm:.1f} mm")

        self.state.d_x = float(self._current_d_x_mm)
        self.state.H_xc = self._H_xc_lookup.get(self._dx_offset_mm)

        self._overlay_cache = None

    def _on_show_tip_toggled(self, checked: bool) -> None:
        self._show_tip = bool(checked)

    def _on_show_axis_toggled(self, checked: bool) -> None:
        self._show_axis = bool(checked)

    def _on_save_snapshot(self) -> None:
        """Save the current frame as stacked presentation layers.

        One timestamped folder with pixel-aligned, same-size PNGs that can be
        placed on top of each other in PowerPoint and revealed in order:

            00_xray.png               raw anatomical X-ray (reference)
            01_rgb.png                camera image only
            02_rgb_overlay.png        + X-ray overlay at the current alpha
            03_rgb_overlay_drill.png  + drill tip and axis (only while a
                                        tracked pose is on screen)
            snapshot.json             the numbers behind the layers: alpha,
                                        d_x and its offset, H_xc, camera
                                        model, reference-plane point, and
                                        while tracking the projected tip/
                                        axis pixels with the pose they came
                                        from - so a thesis figure can redraw
                                        the marks from the bare camera frame
                                        instead of recovering them from the
                                        burnt-in pixels

        Runs in the click handler between two timer ticks, so the zero-copy
        camera view is stable for the duration; the live loop just skips a
        few frames while the PNGs are written.
        """
        if self._live_color is None:
            QMessageBox.information(
                self, "Save snapshot", "No live image yet — start the camera first."
            )
            return

        try:
            ev = self.state.eval
            if ev is not None:
                out_dir = ev.subdir("live")
            else:
                out_dir = SNAPSHOT_DIR / time.strftime("snapshot_%Y%m%d_%H%M%S")
                out_dir.mkdir(parents=True, exist_ok=True)
            written = []

            if self._xray_gray_u8 is not None:
                cv2.imwrite(str(out_dir / "00_xray.png"), self._xray_gray_u8)
                written.append("00_xray")

            base = self._live_color
            cv2.imwrite(
                str(out_dir / "01_rgb.png"),
                cv2.cvtColor(base, cv2.COLOR_BGRA2BGR),
            )
            written.append("01_rgb")

            # Cumulative composite, built exactly like _compose_frame.
            comp = base.copy()
            if self._overlay_cache is not None:
                self._overlay_cache.blend_into(comp, self._alpha)
                if self._show_tip:
                    _draw_point_inplace(comp, self.state.tip_uv_c)
                cv2.imwrite(
                    str(out_dir / "02_rgb_overlay.png"),
                    cv2.cvtColor(comp, cv2.COLOR_BGRA2BGR),
                )
                written.append("02_rgb_overlay")

            if self._tool_overlay is not None and self._tool_overlay.has_pose:
                self._tool_overlay.draw_into(comp, show_tip=True, show_axis=True)
                cv2.imwrite(
                    str(out_dir / "03_rgb_overlay_drill.png"),
                    cv2.cvtColor(comp, cv2.COLOR_BGRA2BGR),
                )
                written.append("03_rgb_overlay_drill")

            with open(out_dir / "snapshot.json", "w", encoding="utf-8") as f:
                json.dump(self._snapshot_record(out_dir.name), f, indent=2)
            written.append("snapshot.json")

            if ev is not None:
                ev.log("overlay_live", out_dir.name + "/", {
                    "layers": len(written),
                    "d_x_mm": self._current_d_x_mm,
                    "tracking": bool(self._tracking_active),
                })

            QMessageBox.information(
                self,
                "Save snapshot",
                f"Saved {len(written)} layer(s):\n{', '.join(written)}\n\n{out_dir}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Save snapshot", f"Saving failed.\n\n{e}")

    def _snapshot_record(self, name: str) -> dict:
        """Everything a figure needs to rebuild the layers from 01_rgb.png."""

        def arr(v):
            return None if v is None else np.asarray(v, dtype=np.float64).tolist()

        h, w = self._live_color.shape[:2]
        rec = {
            "snapshot": name,
            "frame_wh": [int(w), int(h)],
            "alpha": float(self._alpha),
            "d_x_mm": self._current_d_x_mm,
            "d_x_base_mm": self._base_d_x_mm,
            "d_x_offset_mm": int(self._dx_offset_mm),
            "H_xc": arr(self.state.H_xc),
            "K_rgb": arr(self.state.K_rgb),
            "dist_rgb": arr(self.state.dist_rgb),
            "K_xray": arr(self.state.K_xray),
            "T_xc": arr(self.state.T_xc),
            # the live warp runs through the colour lens (prepare_live_overlay
            # with K_rgb/dist_rgb); recorded so an offline re-render knows
            "warp_through_lens": True,
            "xray_image_path": self.state.xray_image_path,
            "reference_plane_point_uv": arr(self.state.tip_uv_c),
            "reference_plane_point_xyz_c_mm": arr(self.state.tip_xyz_c),
            "show_tip": bool(self._show_tip),
            "show_axis": bool(self._show_axis),
            "tracking_active": bool(self._tracking_active),
            "tool": None,
        }

        if self._tool_overlay is not None and self._tool_overlay.has_pose:
            tool = self._tool_overlay.snapshot_record()
            res = self._last_pose_result
            tool["pose"] = None if res is None else {
                "rvec": arr(res.rvec),
                "tvec": arr(res.tvec),
                "rvec_prefusion": arr(res.rvec_prefusion),
                "tvec_prefusion": arr(res.tvec_prefusion),
                "pose_source": getattr(res.pose_source, "name", str(res.pose_source)),
                "pose_converged": bool(getattr(res, "pose_converged", False)),
                "mean_reprojection_error_px": float(res.mean_reprojection_error_px),
                "num_points": int(res.num_points),
                "num_inliers": int(res.num_inliers),
            }
            if self._tool is not None:
                tool["assets"] = {
                    "name": self._tool.name,
                    "field_path": str(self._tool.field_path),
                    "marker_json_path": str(self._tool.marker_json_path),
                    "pivot_npz_path": str(self._tool.pivot_npz_path),
                }
            rec["tool"] = tool

        return rec

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def on_enter(self) -> None:
        # Tool tracking is optional: without a loaded tool the page behaves
        # exactly as before (no IR streams, no tracker, no rail card).
        self._tool = self.state.tool_assets
        self.controls.set_tracking_visible(self._tool is not None)
        self.controls.btn_track.setText("Start tracking")
        self.controls.lbl_track_status.setText("idle")

        self._load_xray_from_state()
        self._initialize_dx()

        if not self._has_prerequisites():
            QMessageBox.information(
                self,
                "Live Overlay",
                "Missing prerequisites from previous steps.",
            )
            return

        if self._xray_gray_u8 is None:
            QMessageBox.information(
                self,
                "Live Overlay",
                "No anatomical X-ray image available in state.",
            )
            return

        try:
            self._precompute_H_xc_lookup()
            self._invalidate_overlay_cache()

            ev = self.state.eval
            self._profile = (
                _FrameProfile(self.FPS, None if ev is None else ev.dir / PROFILE_LOG_NAME)
                if (PROFILE_LIVE and ev is not None) else None
            )
            if self._profile is not None:
                self._profile.reset()
                h, w = self._xray_gray_u8.shape[:2]
                self._profile.emit(
                    f"[live] === session start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"=== colour {self.COLOR_SIZE[0]}x{self.COLOR_SIZE[1]} @ "
                    f"{self.FPS} fps, x-ray {w}x{h}, "
                    f"{len(self._H_xc_lookup)} precomputed homographies"
                )

            self._start_realsense()
            self._start_timer()
            self.setFocus()

        except Exception as e:
            self._stop_timer()
            self._stop_realsense()
            QMessageBox.critical(
                self,
                "Camera",
                f"Could not open RealSense camera.\n\n{e}",
            )

    def on_leave(self) -> None:
        if self._tracking_active:
            self._stop_tracking("idle")
        self._tracker = None
        self._tool_overlay = None
        self._stop_timer()
        self._stop_realsense()

    def closeEvent(self, event) -> None:
        self.on_leave()
        super().closeEvent(event)