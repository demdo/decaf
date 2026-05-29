# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict

import datetime
from pathlib import Path

import numpy as np
import cv2

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QPushButton,
    QMessageBox,
    QSizePolicy,
    QWidget,
    QCheckBox,
    QHBoxLayout,
    QVBoxLayout,
)

import pyrealsense2 as rs

from overlay.gui.state import SessionState
from overlay.tools import checkerboard_corner_detection as cbd
from overlay.tools import ransac_plane_fitting as rpf
from tracking.pose_filters import PlaneKalmanFilter
from overlay.gui.pages.templates.templ_live_image import LiveImagePage


# ============================================================
# Save config
# ============================================================
SAVE_SNAPSHOTS = False

# ============================================================
# Where debug snapshots are saved
# ============================================================
DEBUG_SNAPSHOT_DIR = Path("debug_snapshots")

# "full"        : ein NPZ pro Run mit allen Keys
# "kalman_only" : nur letzter Run, nur points_xyz_camera_filt + K_xray
SNAPSHOT_MODE: str = "full"


def _draw_extremes(img: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
    out = img.copy()
    for (u, v) in pts_uv:
        cv2.circle(out, (int(round(u)), int(round(v))), 10, (208, 224, 64), -1)
    return out


class PlaneFittingPage(LiveImagePage):
    """
    Step — Plane fitting (checkerboard -> plane fit -> 3D marker grid in camera frame)

    Workflow
    --------
    1. Live RGB-D stream; detect checkerboard continuously.
    2. SPACE -> capture: average n_average_frames depth frames, freeze, ask user to
       confirm corners.
    3. On confirm: run _collect_fits() which performs n_fit_runs independent cycles
       on the same averaged depth map. Each cycle uses a different random seed so
       RANSAC point sampling varies.

    Grid reconstruction
    -------------------
    - always use TL / TR / BL
    - intersect these 3 rays with the fitted plane
    - interpolate full 11x11 grid in 3D

    IMPORTANT:
    - This page MUST NOT add any new SessionState fields dynamically.
    - It ONLY writes fields that already exist in overlay/gui/state.py:
        * xray_points_xyz_c
        * plane_confirmed
        * checkerboard_corners_uv
        * checkerboard_corners_confirmed
    """

    def __init__(self, state: SessionState, on_complete_changed=None, parent=None):
        self.state = state
        self.on_complete_changed = on_complete_changed

        # ---------------- detection ----------------
        self.pattern_size = (3, 3)
        self.det_width = 640

        # ---------------- ROI & sampling ----------------
        self.pad_px = 15
        self.max_points = 5000
        self.z_min, self.z_max = 0.30, 2.0
        self.min_points_for_fit = 800

        # ---------------- RANSAC ----------------
        self.thresh_m = 0.001
        self.ransac_n = 8
        self.iters = 3000
        self.n_stable_runs = 10

        # ---------------- fit repetitions ----------------
        self.n_fit_runs = 10

        # ---------------- local config ----------------
        self.steps_per_edge = 10
        # ---------------- depth averaging ----------------
        self.use_temporal_filter = True
        self.temporal_alpha = 0.1
        self.temporal_delta = 20
        self.n_average_frames = 30

        # ---------------- plane Kalman filter ----------------
        self._plane_kf = PlaneKalmanFilter(
            process_noise=1e-7,
            measurement_noise=1e-4,
            outlier_angle_deg=1.5,
        )

        # ---------------- runtime ----------------
        self._mode = "idle"  # idle | live | frozen
        self._found = False

        self._live_color: np.ndarray | None = None
        self._live_depth = None
        self._live_depth_u16: np.ndarray | None = None
        self._live_depth_vis_bgr: np.ndarray | None = None

        self._snap_color: np.ndarray | None = None
        self._snap_vis: np.ndarray | None = None

        self._snap_depth_avg_raw: np.ndarray | None = None
        self._snap_depth_intrinsics = None
        self._depth_scale_m: float = 1.0

        self._ext_uv: np.ndarray | None = None        # (3,2): TL, TR, BL
        self._corners_full: np.ndarray | None = None  # (9,1,2)
        self._rect: tuple[int, int, int, int] | None = None
        self._seed = 0
        self._show_depth = False

        # RealSense depth tuning
        self.use_tuned_depth_settings = False
        self._depth_sensor: rs.sensor | None = None
        self._depth_prev_settings: dict[rs.option, float] | None = None
        self._temporal_filter = None

        # stats display
        self._stats: list[str] = []
        self._last_stats_rows: list[tuple[str, str]] | None = None

        # snapshot counter
        self._snapshot_counter: int = 0

        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        self._build_controls()

        self.instructions_label.setText(
            "1) Press Start to open RGB-D\n"
            "2) Move checkerboard until FOUND\n"
            "3) Press SPACE to capture\n"
            "4) Confirm corners to run plane fitting"
        )

        self.set_viewport_background(active=False)
        self._mode = "idle"

        if self.state.has_plane_confirmed:
            self._stats = [
                "Inliers: (done)",
                "Mean: (done)",
                "Median: (done)",
                "P95: (done)",
            ]

        self._update_buttons()
        self._update_panels()
        self.update_view()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _build_controls(self) -> None:
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start_clicked)
        self.btn_start.setFocusPolicy(Qt.NoFocus)
        self.btn_start.setMinimumHeight(44)
        self.btn_start.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.btn_start.setMinimumWidth(0)

        self.chk_show_depth = QCheckBox("Show depth image")
        self.chk_show_depth.setChecked(False)
        self.chk_show_depth.toggled.connect(self._on_show_depth_changed)
        self.chk_show_depth.setFocusPolicy(Qt.NoFocus)

        self.chk_depth_tuning = QCheckBox("Depth Tuning")
        self.chk_depth_tuning.setChecked(False)
        self.chk_depth_tuning.toggled.connect(self._on_use_tuning_changed)
        self.chk_depth_tuning.setFocusPolicy(Qt.NoFocus)

        wrap = QWidget()
        wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        row.addWidget(self.btn_start, 0, Qt.AlignTop)

        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(8)
        col.addWidget(self.chk_show_depth)
        col.addWidget(self.chk_depth_tuning)
        col.addStretch(1)

        row.addLayout(col, 1)
        self.controls_content.addWidget(wrap)

    def _update_panels(self) -> None:
        rows = self.stats_rows()
        if rows != self._last_stats_rows:
            self.set_stats_rows(rows)
            self._last_stats_rows = list(rows)

    def stats_rows(self) -> list[tuple[str, str]]:
        if not self._stats:
            return [
                ("Inliers", "-"),
                ("Mean", "-"),
                ("Median", "-"),
                ("P95", "-"),
            ]
        out: list[tuple[str, str]] = []
        for s in self._stats:
            if ":" in s:
                k, v = s.split(":", 1)
                out.append((k.strip(), v.strip()))
            else:
                out.append((s.strip(), ""))
        return out

    # ------------------------------------------------------------------ #
    # Template hooks
    # ------------------------------------------------------------------ #

    def get_frame(self) -> np.ndarray | None:
        if self._mode == "idle":
            return None
        if self._mode == "frozen":
            return self._snap_vis

        if self.pipeline is None or self.align is None:
            if self._show_depth and self._live_depth_vis_bgr is not None:
                return self._live_depth_vis_bgr
            return self._live_color

        frames = self.pipeline.poll_for_frames()
        if not frames:
            if self._show_depth and self._live_depth_vis_bgr is not None:
                return self._live_depth_vis_bgr
            return self._live_color

        frames = self.align.process(frames)
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()

        if df:
            if self._temporal_filter is not None:
                try:
                    df = self._temporal_filter.process(df).as_depth_frame()
                except Exception:
                    pass
            self._live_depth = df
            depth_u16 = np.asanyarray(df.get_data()).astype(np.uint16)
            self._live_depth_u16 = depth_u16
            self._live_depth_vis_bgr = self._depth_u16_to_vis_bgr(depth_u16)

        if cf:
            color = self.color_frame_to_bgr(cf)
            self._live_color = color
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            found, _ = cbd.detect_classic_downscaled(
                gray, self.pattern_size, det_width=self.det_width
            )
            self._found = bool(found)

        self._update_panels()

        if self._show_depth:
            if self._live_depth_vis_bgr is not None:
                depth_bgr = self._live_depth_vis_bgr.copy()
                cv2.putText(
                    depth_bgr, "DEPTH VIEW", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA,
                )
                return depth_bgr
            ref = self._live_color
            if ref is None:
                return None
            h, w = ref.shape[:2]
            return np.zeros((h, w, 3), dtype=np.uint8)

        return self._live_color

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        vis = frame_bgr.copy()
        if self._mode == "live":
            txt = "FOUND (press SPACE)" if self._found else "NOT FOUND"
            col = (0, 255, 0) if self._found else (0, 0, 255)
            cv2.putText(
                vis, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 0), 6, cv2.LINE_AA
            )
            cv2.putText(
                vis, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                col, 2, cv2.LINE_AA
            )
        return vis

    # ------------------------------------------------------------------ #
    # Depth image helpers
    # ------------------------------------------------------------------ #

    def _on_show_depth_changed(self, checked: bool) -> None:
        self._show_depth = bool(checked)
        self.update_view()

    @staticmethod
    def _depth_u16_to_vis_bgr(depth_u16: np.ndarray) -> np.ndarray:
        depth_u16 = np.asarray(depth_u16, dtype=np.uint16)
        nonzero = depth_u16[depth_u16 > 0]
        if nonzero.size == 0:
            return np.zeros((depth_u16.shape[0], depth_u16.shape[1], 3), dtype=np.uint8)
        lo = float(np.percentile(nonzero, 2.0))
        hi = float(np.percentile(nonzero, 98.0))
        if hi <= lo:
            hi = lo + 1.0
        depth_8u = np.clip(
            (depth_u16.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255
        ).astype(np.uint8)
        depth_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
        depth_bgr[depth_u16 == 0] = (0, 0, 0)
        return depth_bgr

    @staticmethod
    def _depth_array_to_vis_bgr(depth_raw: np.ndarray) -> np.ndarray:
        arr = np.asarray(depth_raw)
        if arr.dtype != np.uint16:
            arr = np.nan_to_num(arr, nan=0.0).astype(np.uint16)
        return PlaneFittingPage._depth_u16_to_vis_bgr(arr)

    def _on_use_tuning_changed(self, checked: bool) -> None:
        self.use_tuned_depth_settings = bool(checked)
        if self.pipeline is None:
            return

        if self.use_tuned_depth_settings:
            self._apply_depth_tuning()

        self.update_view()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def on_enter(self) -> None:
        pass

    def on_leave(self) -> None:
        try:
            self._apply_depth_defaults()
        except Exception:
            pass
        super().on_leave()

    # ------------------------------------------------------------------ #
    # RealSense depth tuning
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rs_get(sensor: rs.sensor | None, opt: rs.option) -> float | None:
        try:
            if sensor is not None and sensor.supports(opt):
                return float(sensor.get_option(opt))
        except Exception:
            pass
        return None

    @staticmethod
    def _rs_set(sensor: rs.sensor | None, opt: rs.option, val: float) -> bool:
        try:
            if sensor is not None and sensor.supports(opt):
                sensor.set_option(opt, float(val))
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _rs_range(sensor: rs.sensor | None, opt: rs.option):
        try:
            if sensor is not None and sensor.supports(opt):
                return sensor.get_option_range(opt)
        except Exception:
            pass
        return None

    def _get_depth_sensor(self) -> rs.sensor | None:
        if self.pipeline is None:
            self._depth_sensor = None
            return None
        try:
            prof = self.pipeline.get_active_profile()
            dev = prof.get_device()
            self._depth_sensor = dev.first_depth_sensor()
            return self._depth_sensor
        except Exception:
            self._depth_sensor = None
            return None

    def _snapshot_depth_settings(self, sensor: rs.sensor | None) -> Dict[rs.option, float]:
        opts = [
            rs.option.visual_preset,
            rs.option.emitter_enabled,
            rs.option.laser_power,
            rs.option.enable_auto_exposure,
            rs.option.exposure,
            rs.option.gain,
        ]
        snap: Dict[rs.option, float] = {}
        for o in opts:
            v = self._rs_get(sensor, o)
            if v is not None:
                snap[o] = float(v)
        return snap

    def _apply_depth_defaults(self) -> None:
        sensor = self._get_depth_sensor()
        if sensor is None:
            return
        if self._depth_prev_settings is None:
            self._depth_prev_settings = self._snapshot_depth_settings(sensor)
        self._rs_set(sensor, rs.option.enable_auto_exposure, 1)
        r = self._rs_range(sensor, rs.option.emitter_enabled)
        if r is not None:
            target = 2.0
            if target < r.min or target > r.max:
                target = float(np.clip(1.0, r.min, r.max))
            self._rs_set(sensor, rs.option.emitter_enabled, target)
        r = self._rs_range(sensor, rs.option.laser_power)
        if r is not None:
            mid = 0.5 * (float(r.min) + float(r.max))
            self._rs_set(sensor, rs.option.laser_power, mid)
        r = self._rs_range(sensor, rs.option.visual_preset)
        if r is not None:
            self._rs_set(sensor, rs.option.visual_preset, float(np.clip(0.0, r.min, r.max)))

    def _apply_depth_tuning(self) -> None:
        if not self.use_tuned_depth_settings:
            return
        sensor = self._get_depth_sensor()
        if sensor is None:
            return
        if self._depth_prev_settings is None:
            self._depth_prev_settings = self._snapshot_depth_settings(sensor)
        self._rs_set(sensor, rs.option.emitter_enabled, 1)
        self._rs_set(sensor, rs.option.enable_auto_exposure, 0)
        r = self._rs_range(sensor, rs.option.laser_power)
        if r is not None:
            self._rs_set(sensor, rs.option.laser_power, float(np.clip(150, r.min, r.max)))
        r = self._rs_range(sensor, rs.option.exposure)
        if r is not None:
            self._rs_set(sensor, rs.option.exposure, float(np.clip(4000, r.min, r.max)))
        r = self._rs_range(sensor, rs.option.gain)
        if r is not None:
            self._rs_set(sensor, rs.option.gain, float(np.clip(16, r.min, r.max)))

    def _setup_temporal_filter(self) -> None:
        if not self.use_temporal_filter:
            self._temporal_filter = None
            return
        try:
            tf = rs.temporal_filter()
            tf.set_option(rs.option.filter_smooth_alpha, float(self.temporal_alpha))
            tf.set_option(rs.option.filter_smooth_delta, float(self.temporal_delta))
            self._temporal_filter = tf
        except Exception:
            self._temporal_filter = None

    def _read_depth_scale(self) -> None:
        sensor = self._get_depth_sensor()
        if sensor is None:
            self._depth_scale_m = 1.0
            return
        try:
            self._depth_scale_m = float(sensor.get_depth_scale())
        except Exception:
            self._depth_scale_m = 1.0

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def start_clicked(self) -> None:
        if self.state.K_rgb is None:
            QMessageBox.information(
                self,
                "Plane Fitting",
                "Missing K_rgb. Run Camera Calibration first.",
            )
            return

        try:
            if self.pipeline is None:
                self.start_realsense(
                    fps=self.FPS,
                    color_size=(1920, 1080),
                    depth_size=(1280, 720),
                    align_to="color",
                )

            if self.use_tuned_depth_settings:
                self._apply_depth_tuning()

            self._read_depth_scale()
            self._setup_temporal_filter()

        except Exception as e:
            self.stop_realsense()
            QMessageBox.critical(self, "Camera", f"Could not open RealSense camera.\n\n{e}")
            self._mode = "idle"
            self.set_viewport_background(active=False)
            self._update_buttons()
            self._update_panels()
            self.update_view()
            return

        self.state.xray_points_xyz_c = None
        self.state.plane_confirmed = False
        self.state.checkerboard_corners_uv = None
        self.state.checkerboard_corners_confirmed = False

        self._plane_kf.reset()

        self._stats = []
        self._last_stats_rows = None
        self._snap_color = None
        self._snap_vis = None
        self._snap_depth_avg_raw = None
        self._snap_depth_intrinsics = None
        self._ext_uv = None
        self._corners_full = None
        self._rect = None
        self._seed = 0

        self._mode = "live"
        self.set_viewport_background(active=True)
        self.start_timer(self.FPS)
        self._update_buttons()
        self._update_panels()
        self.setFocus()
        self.update_view()

        if callable(self.on_complete_changed):
            self.on_complete_changed()

    # ------------------------------------------------------------------ #
    # Keyboard
    # ------------------------------------------------------------------ #

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self._mode == "live" and self._found:
                self._capture()
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------ #
    # Depth averaging
    # ------------------------------------------------------------------ #

    def _capture_averaged_depth(self) -> tuple[np.ndarray, object] | None:
        if self.pipeline is None or self.align is None:
            return None

        accumulator = None
        count_map = None
        last_df = None

        for _ in range(self.n_average_frames):
            frames = self.pipeline.wait_for_frames()
            frames = self.align.process(frames)
            df = frames.get_depth_frame()
            if not df:
                continue
            if self._temporal_filter is not None:
                try:
                    df = self._temporal_filter.process(df).as_depth_frame()
                except Exception:
                    pass
            raw = np.asanyarray(df.get_data()).astype(np.float64)
            if accumulator is None:
                accumulator = np.zeros_like(raw, dtype=np.float64)
                count_map = np.zeros_like(raw, dtype=np.int32)
            valid = raw > 0
            accumulator[valid] += raw[valid]
            count_map[valid] += 1
            last_df = df

        if accumulator is None or last_df is None:
            return None

        with np.errstate(invalid="ignore"):
            averaged = np.where(
                count_map > 0,
                accumulator / count_map,
                np.nan,
            ).astype(np.float32)

        return averaged, last_df

    def _sample_pts3d_from_averaged(
        self,
        averaged_raw: np.ndarray,
        intrinsics,
        rect: tuple[int, int, int, int],
        max_points: int,
        z_min: float,
        z_max: float,
        seed: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        umin, vmin, umax, vmax = rect
        roi_w = umax - umin + 1
        roi_h = vmax - vmin + 1
        if roi_w <= 2 or roi_h <= 2:
            return np.empty((0, 3), dtype=np.float64)
        num_pixels = roi_w * roi_h
        idx = rng.choice(num_pixels, size=min(max_points, num_pixels), replace=False)
        us = (idx % roi_w).astype(np.int32) + umin
        vs = (idx // roi_w).astype(np.int32) + vmin
        pts3d = []
        for u, v in zip(us, vs):
            raw = averaged_raw[v, u]
            if np.isnan(raw) or raw <= 0:
                continue
            z = float(raw) * self._depth_scale_m
            if z < z_min or z > z_max:
                continue
            xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], z)
            pts3d.append(xyz)
        return np.asarray(pts3d, dtype=np.float64) if pts3d else np.empty((0, 3), dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Main capture workflow
    # ------------------------------------------------------------------ #

    def _capture(self) -> None:
        if self._live_color is None:
            return

        color = self._live_color.copy()

        avg_out = self._capture_averaged_depth()
        if avg_out is None:
            QMessageBox.warning(self, "Plane Fitting", "Could not capture averaged depth.")
            return

        averaged_raw, last_df = avg_out
        intrinsics = last_df.profile.as_video_stream_profile().intrinsics

        found, corners = cbd.detect_snapshot_full(
            color, pattern_size=self.pattern_size, det_width=self.det_width,
        )
        if (not found) or corners is None:
            return

        ex = cbd.get_extreme_corners_geometric(corners)
        ext_uv = np.array(
            [ex["top_left"], ex["top_right"], ex["bottom_left"]], dtype=np.float64
        )

        preview = color.copy()
        cv2.drawChessboardCorners(preview, self.pattern_size, corners, True)
        preview = _draw_extremes(preview, ext_uv)

        self._mode = "frozen"
        self._snap_color = color
        self._snap_vis = preview
        self._snap_depth_avg_raw = averaged_raw
        self._snap_depth_intrinsics = intrinsics
        self._ext_uv = ext_uv
        self._corners_full = corners
        self.stop_timer()

        self._update_buttons()
        self._update_panels()
        self.update_view()

        ans = QMessageBox.question(
            self,
            "Corner Detection",
            "Satisfied with the detected corners?\n\nYes: run plane fitting\nNo: back to live video",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if ans != QMessageBox.Yes:
            self._seed = 0
            self._mode = "live"
            self._snap_color = None
            self._snap_vis = None
            self._snap_depth_avg_raw = None
            self._snap_depth_intrinsics = None
            self._ext_uv = None
            self._corners_full = None
            self._rect = None
            self._stats = []
            self._last_stats_rows = None
            self.state.checkerboard_corners_uv = None
            self.state.checkerboard_corners_confirmed = False
            self.set_viewport_background(active=True)
            self.start_timer(self.FPS)
            self._update_buttons()
            self._update_panels()
            self.setFocus()
            self.update_view()
            return

        h, w = color.shape[:2]
        self._rect = rpf.rect_from_pts(ext_uv, w, h, self.pad_px)
        self._seed = 0

        self.state.checkerboard_corners_uv = ext_uv
        self.state.checkerboard_corners_confirmed = True

        self._collect_fits()

        self._update_buttons()
        self._update_panels()
        self.setFocus()
        self.update_view()

    # ------------------------------------------------------------------ #
    # n_fit_runs independent cycles on the frozen depth map
    # ------------------------------------------------------------------ #

    def _collect_fits(self) -> None:
        assert self._snap_depth_avg_raw is not None
        assert self._snap_depth_intrinsics is not None
        assert self._rect is not None and self._ext_uv is not None
        assert self._corners_full is not None
        assert self.state.K_rgb is not None

        print(f"[PlaneFitting] Starting {self.n_fit_runs} fit runs on frozen depth map …")
        print(f"[PlaneFitting] SNAPSHOT_MODE = '{SNAPSHOT_MODE}'")

        last_marker_xyz_filtered: np.ndarray | None = None
        last_stats: list[str] = []

        session_planes_raw: list[np.ndarray] = []
        session_planes_filt: list[np.ndarray] = []
        session_points_raw: list[np.ndarray] = []
        session_points_filt: list[np.ndarray] = []
        session_inlier_ratio: list[float] = []
        session_mean_mm: list[float] = []
        session_median_mm: list[float] = []
        session_p95_mm: list[float] = []

        for run_idx in range(self.n_fit_runs):
            seed = self._seed + run_idx

            pts3d = self._sample_pts3d_from_averaged(
                averaged_raw=self._snap_depth_avg_raw,
                intrinsics=self._snap_depth_intrinsics,
                rect=self._rect,
                max_points=self.max_points,
                z_min=self.z_min,
                z_max=self.z_max,
                seed=seed,
            )

            if pts3d.shape[0] < self.min_points_for_fit:
                print(
                    f"[PlaneFitting]   run {run_idx + 1}/{self.n_fit_runs}: "
                    f"not enough points ({pts3d.shape[0]}), skipping."
                )
                continue

            plane_single: np.ndarray | None = None
            try:
                plane_single, _ = rpf.ransac_plane_open3d(
                    pts3d,
                    self.thresh_m,
                    self.ransac_n,
                    self.iters,
                )
                plane_single = np.asarray(plane_single, dtype=np.float64)
            except Exception as e:
                print(f"[PlaneFitting]   run {run_idx + 1}: single RANSAC failed: {e}")

            plane_raw: np.ndarray | None = None
            inliers = np.array([], dtype=np.int64)
            try:
                plane_raw, inliers = rpf.fit_plane_stable(
                    pts3d,
                    distance_threshold=self.thresh_m,
                    ransac_n=self.ransac_n,
                    num_iterations=self.iters,
                    n_runs=self.n_stable_runs,
                )
                plane_raw = np.asarray(plane_raw, dtype=np.float64)
            except Exception as e:
                print(f"[PlaneFitting]   run {run_idx + 1}: fit_plane_stable failed: {e}")
                continue

            plane_filtered = np.asarray(self._plane_kf.update(plane_raw), dtype=np.float64)

            dev = rpf.deviations(pts3d, plane_raw)
            dev_in = dev[inliers] if len(inliers) else dev
            mean = float(np.mean(dev_in))
            med = float(np.median(dev_in))
            p95 = float(np.percentile(dev_in, 95))
            inlier_ratio = len(inliers) / pts3d.shape[0]

            marker_xyz_single = self._plane_to_grid(plane_single)
            marker_xyz_raw = self._plane_to_grid(plane_raw)
            marker_xyz_filtered = self._plane_to_grid(plane_filtered)

            if plane_raw is not None:
                session_planes_raw.append(plane_raw)
            session_planes_filt.append(plane_filtered)
            session_points_raw.append(marker_xyz_raw)
            session_points_filt.append(marker_xyz_filtered)
            session_inlier_ratio.append(inlier_ratio)
            session_mean_mm.append(mean * 1000.0)
            session_median_mm.append(med * 1000.0)
            session_p95_mm.append(p95 * 1000.0)

            if SNAPSHOT_MODE == "full":
                self._save_snapshot(
                    kind="run",
                    run_idx=run_idx,
                    seed=seed,
                    plane_single=plane_single,
                    plane_raw=plane_raw,
                    plane_filtered=plane_filtered,
                    marker_xyz_single=marker_xyz_single,
                    marker_xyz_raw=marker_xyz_raw,
                    marker_xyz_filtered=marker_xyz_filtered,
                    inlier_ratio=inlier_ratio,
                    mean_mm=mean * 1000.0,
                    median_mm=med * 1000.0,
                    p95_mm=p95 * 1000.0,
                )
            elif SNAPSHOT_MODE == "kalman_only" and run_idx == self.n_fit_runs - 1:
                self._save_snapshot(
                    kind="run",
                    run_idx=run_idx,
                    seed=seed,
                    plane_filtered=plane_filtered,
                    marker_xyz_filtered=marker_xyz_filtered,
                )

            last_marker_xyz_filtered = marker_xyz_filtered
            last_stats = [
                f"Inliers: {int(len(inliers))}/{int(pts3d.shape[0])}",
                f"Mean: {mean * 1000.0:.3f} mm",
                f"Median: {med * 1000.0:.3f} mm",
                f"P95: {p95 * 1000.0:.3f} mm",
            ]

            print(
                f"[PlaneFitting]   run {run_idx + 1}/{self.n_fit_runs} — "
                f"inliers {len(inliers)}/{pts3d.shape[0]}, "
                f"mean {mean * 1000:.3f} mm"
            )

        self._seed += self.n_fit_runs

        if last_marker_xyz_filtered is None:
            QMessageBox.warning(
                self, "Plane Fitting", "All fit runs failed — no valid plane found."
            )
            return

        self._save_snapshot(
            kind="session",
            session_planes_raw=session_planes_raw,
            session_planes_filt=session_planes_filt,
            session_points_raw=session_points_raw,
            session_points_filt=session_points_filt,
            session_inlier_ratio=session_inlier_ratio,
            session_mean_mm=session_mean_mm,
            session_median_mm=session_median_mm,
            session_p95_mm=session_p95_mm,
        )

        self.state.xray_points_xyz_c = last_marker_xyz_filtered
        self.state.plane_confirmed = True
        self._stats = last_stats
        self._last_stats_rows = None

        if callable(self.on_complete_changed):
            self.on_complete_changed()

        print(f"[PlaneFitting] Done. state.xray_points_xyz_c set from run {self.n_fit_runs}.")

    # ------------------------------------------------------------------ #
    # Helper: plane -> marker grid
    # ------------------------------------------------------------------ #

    def _plane_to_grid(self, plane: np.ndarray | None) -> np.ndarray:
        if plane is None:
            return np.empty((0, 3), dtype=np.float64)

        try:
            corner_xyz = rpf.intersect_pixels_with_plane(
                self._ext_uv,
                self.state.K_rgb,
                plane,
                None,
            )
            return rpf.interpolate_marker_grid(
                corner_xyz,
                steps_per_edge=int(self.steps_per_edge),
            )

        except Exception as e:
            print(f"[PlaneFitting] _plane_to_grid failed: {e}")
            return np.empty((0, 3), dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Single save function
    # ------------------------------------------------------------------ #

    def _save_snapshot(
        self,
        *,
        kind: str,
        run_idx: int | None = None,
        seed: int | None = None,
        plane_single: np.ndarray | None = None,
        plane_raw: np.ndarray | None = None,
        plane_filtered: np.ndarray | None = None,
        marker_xyz_single: np.ndarray | None = None,
        marker_xyz_raw: np.ndarray | None = None,
        marker_xyz_filtered: np.ndarray | None = None,
        inlier_ratio: float | None = None,
        mean_mm: float | None = None,
        median_mm: float | None = None,
        p95_mm: float | None = None,
        session_planes_raw: list[np.ndarray] | None = None,
        session_planes_filt: list[np.ndarray] | None = None,
        session_points_raw: list[np.ndarray] | None = None,
        session_points_filt: list[np.ndarray] | None = None,
        session_inlier_ratio: list[float] | None = None,
        session_mean_mm: list[float] | None = None,
        session_median_mm: list[float] | None = None,
        session_p95_mm: list[float] | None = None,
    ) -> None:
        if not SAVE_SNAPSHOTS:
            return

        try:
            DEBUG_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            if kind == "run":
                self._snapshot_counter += 1
                fname = (
                    DEBUG_SNAPSHOT_DIR
                    / f"plane_snapshot_{ts}_c{self._snapshot_counter:02d}_r{int(run_idx):02d}.npz"
                )

                arrays: dict[str, np.ndarray] = {}

                if run_idx is not None:
                    arrays["run_index"] = np.array([run_idx], dtype=np.int32)
                if seed is not None:
                    arrays["seed"] = np.array([seed], dtype=np.int32)

                if plane_single is not None:
                    arrays["plane_abcd_single"] = np.asarray(plane_single, dtype=np.float64)
                if plane_raw is not None:
                    arrays["plane_abcd_raw"] = np.asarray(plane_raw, dtype=np.float64)
                if plane_filtered is not None:
                    arrays["plane_abcd_filtered"] = np.asarray(plane_filtered, dtype=np.float64)

                if marker_xyz_single is not None:
                    arrays["points_xyz_camera_single"] = np.asarray(marker_xyz_single, dtype=np.float64)
                if marker_xyz_raw is not None:
                    arrays["points_xyz_camera"] = np.asarray(marker_xyz_raw, dtype=np.float64)
                if marker_xyz_filtered is not None:
                    arrays["points_xyz_camera_filt"] = np.asarray(marker_xyz_filtered, dtype=np.float64)

                if inlier_ratio is not None:
                    arrays["inlier_ratio"] = np.array([inlier_ratio], dtype=np.float64)
                if mean_mm is not None:
                    arrays["mean_mm"] = np.array([mean_mm], dtype=np.float64)
                if median_mm is not None:
                    arrays["median_mm"] = np.array([median_mm], dtype=np.float64)
                if p95_mm is not None:
                    arrays["p95_mm"] = np.array([p95_mm], dtype=np.float64)

                if self._ext_uv is not None:
                    arrays["ext_uv"] = np.asarray(self._ext_uv, dtype=np.float64)

                if self._corners_full is not None:
                    arrays["corners_full_uv"] = np.asarray(self._corners_full, dtype=np.float64)

                if self._snap_color is not None:
                    arrays["rgb_image"] = self._snap_color

                if self._snap_depth_avg_raw is not None:
                    arrays["depth_avg_raw"] = np.asarray(self._snap_depth_avg_raw, dtype=np.float32)

                if self.state.K_rgb is not None:
                    arrays["K_rgb"] = np.asarray(self.state.K_rgb, dtype=np.float64)

                if self.state.dist_rgb is not None:
                    arrays["dist_rgb"] = np.asarray(self.state.dist_rgb, dtype=np.float64)

                K_xray = getattr(self.state, "K_xray", None)
                if K_xray is not None:
                    arrays["K_xray"] = np.asarray(K_xray, dtype=np.float64)

                arrays["kalman_initialized"] = np.array(
                    [1 if self._plane_kf.is_initialized else 0], dtype=np.int8
                )

                np.savez(str(fname), **arrays)
                print(f"[PlaneFitting]   snapshot saved: {fname.name}")
                return

            if kind == "session":
                fname = DEBUG_SNAPSHOT_DIR / f"plane_session_{ts}.npz"

                arrays: dict[str, np.ndarray] = {}
                if session_planes_raw is not None:
                    arrays["planes_raw"] = np.asarray(session_planes_raw, dtype=np.float64)
                if session_planes_filt is not None:
                    arrays["planes_filt"] = np.asarray(session_planes_filt, dtype=np.float64)

                if session_points_raw is not None:
                    arrays["points_raw"] = np.array(session_points_raw, dtype=object)
                if session_points_filt is not None:
                    arrays["points_filt"] = np.array(session_points_filt, dtype=object)

                if session_inlier_ratio is not None:
                    arrays["inlier_ratio"] = np.asarray(session_inlier_ratio, dtype=np.float64)
                if session_mean_mm is not None:
                    arrays["mean_mm"] = np.asarray(session_mean_mm, dtype=np.float64)
                if session_median_mm is not None:
                    arrays["median_mm"] = np.asarray(session_median_mm, dtype=np.float64)
                if session_p95_mm is not None:
                    arrays["p95_mm"] = np.asarray(session_p95_mm, dtype=np.float64)

                if self._ext_uv is not None:
                    arrays["ext_uv"] = np.asarray(self._ext_uv, dtype=np.float64)

                if self._corners_full is not None:
                    arrays["corners_full_uv"] = np.asarray(self._corners_full, dtype=np.float64)

                if self.state.K_rgb is not None:
                    arrays["K_rgb"] = np.asarray(self.state.K_rgb, dtype=np.float64)

                if self.state.dist_rgb is not None:
                    arrays["dist_rgb"] = np.asarray(self.state.dist_rgb, dtype=np.float64)

                K_xray = getattr(self.state, "K_xray", None)
                if K_xray is not None:
                    arrays["K_xray"] = np.asarray(K_xray, dtype=np.float64)

                arrays["n_fit_runs"] = np.array([self.n_fit_runs], dtype=np.int32)
                arrays["n_stable_runs"] = np.array([self.n_stable_runs], dtype=np.int32)

                np.savez(str(fname), **arrays)
                print(f"[PlaneFitting]   session summary saved: {fname.name}")
                return

            print(f"[WARN] Unknown snapshot kind: {kind}")

        except Exception as e:
            print(f"[WARN] Could not save snapshot: {e}")

    # ------------------------------------------------------------------ #
    # Button state
    # ------------------------------------------------------------------ #

    def _update_buttons(self) -> None:
        self.btn_start.setEnabled(self._mode == "idle")
        self.chk_show_depth.setEnabled(self._mode in ("live", "frozen"))
        self.chk_depth_tuning.setEnabled(self.pipeline is None or self._mode == "live")