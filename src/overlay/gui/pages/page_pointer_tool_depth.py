# overlay/gui/pages/page_pointer_tool_depth.py

from __future__ import annotations

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QPushButton,
    QMessageBox,
    QSizePolicy,
    QWidget,
    QHBoxLayout,
)

from dataclasses import replace

from overlay.gui.state import SessionState
from overlay.gui.pages.templates.templ_live_image import LiveImagePage

from overlay.calib.calib_camera_to_pointer import (
    calibrate_camera_to_pointer,
    get_default_pointer_tool_model,
)
from overlay.calib.calib_xray_to_pointer import extract_depth
from tracking.pose_filters import AdaptiveKalmanFilterCV3D


# ============================================================
# Helpers (UI only)
# ============================================================

def _draw_text_box(
    img_bgr: np.ndarray,
    lines: list[str],
    org=(30, 55),
    color=(255, 255, 255),
    line_gap=35,
    font_scale=1.0,
) -> np.ndarray:
    out = img_bgr.copy()
    x, y = org

    for i, t in enumerate(lines):
        yy = y + i * line_gap
        cv2.putText(
            out,
            t,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            t,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA,
        )

    return out


def _draw_point(
    img: np.ndarray,
    uv: np.ndarray,
    *,
    color=(255, 0, 255),
    radius=7,
    cross_size=18,
    thickness=2,
    label: str | None = None,
) -> None:
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

    if label is not None:
        cv2.putText(
            img,
            label,
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _detect_aruco(
    image: np.ndarray,
    aruco_dict,
    detector_params,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    gray = _ensure_gray(image)

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=detector_params,
        )

    return corners, ids


def _draw_aruco_overlay(
    image_bgr: np.ndarray,
    corners: list[np.ndarray],
    ids: np.ndarray | None,
) -> np.ndarray:
    vis = image_bgr.copy()

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

    return vis


# ============================================================
# Page
# ============================================================

class PointerToolDepthPage(LiveImagePage):
    """
    Step — Pointer Tool Depth

    Goal
    ----
    Estimate d_x of the target plane in the X-ray frame by tracking
    the pointer tip.

    Pipeline
    --------
    - calibrate_camera_to_pointer(...) returns the accepted tip pose in camera frame
    - extract_depth(...) computes d_x from T_xc and T_tc
    - accepted results are stored in SessionState

    Stored in SessionState
    ----------------------
    - d_x
    - tip_uv_c
    - tip_xyz_c
    - T_tc
    """

    def __init__(self, state: SessionState, on_complete_changed=None, parent=None):
        self.state = state
        self.on_complete_changed = on_complete_changed

        self.pointer_model = get_default_pointer_tool_model()

        dict_id = getattr(cv2.aruco, self.pointer_model.dictionary_name)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

        if hasattr(cv2.aruco, "DetectorParameters"):
            self.detector_params = cv2.aruco.DetectorParameters()
        else:
            self.detector_params = cv2.aruco.DetectorParameters_create()

        # ---------------- runtime ----------------
        self._mode = "idle"  # idle | preview | tracking | frozen
        self._found = False

        self._live_color: np.ndarray | None = None
        self._frozen_vis: np.ndarray | None = None

        self._last_valid_result = None
        self._last_proposed_result = None

        self._stats: list[str] = []
        self._last_stats_rows: list[tuple[str, str]] | None = None

        # Match the debug script 1:1
        self._tip_kf = AdaptiveKalmanFilterCV3D(
            dt=1.0 / float(self.FPS),
            q_pos_still=1e-4,
            q_vel_still=1e-2,
            r_still=8e-2,
            q_pos_move=5e-3,
            q_vel_move=3e-1,
            r_move=2e-2,
        )

        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        self._build_controls()

        self.instructions_label.setText(
            "1) Live RGB is shown automatically\n"
            "2) Press Start to begin pointer tracking\n"
            "3) Move pointer tool until FOUND\n"
            "4) Press SPACE to freeze current pose\n"
            "5) Confirm if you are satisfied"
        )

        self.set_viewport_background(active=False)

        self._update_buttons()
        self._update_panels()
        self.update_view()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _filter_pointer_result(self, raw_result):
        """
        Apply Kalman filtering to the tip position only.
        Rotation stays raw; it is only used to adapt the filter noise.
        """
        tip_xyz_f_mm = self._tip_kf.filter(
            measurement_mm=raw_result.tip_point_camera_mm,
            rotation_camera=raw_result.rotation,
        ).reshape(3)

        T_tc_f = np.asarray(raw_result.T_4x4, dtype=np.float64).copy()
        T_tc_f[:3, 3] = tip_xyz_f_mm

        tvec_f = tip_xyz_f_mm.reshape(3, 1)
        translation_f = tvec_f.copy()

        K = np.asarray(self.state.K_rgb, dtype=np.float64)
        dist = np.zeros((5, 1), dtype=np.float64)

        tip_uv_f, _ = cv2.projectPoints(
            np.zeros((1, 3), dtype=np.float64),  # tip origin in tip frame
            raw_result.rvec,
            tvec_f,
            K,
            dist,
        )
        tip_uv_f = tip_uv_f.reshape(2)

        return replace(
            raw_result,
            tvec=tvec_f,
            translation=translation_f,
            T_4x4=T_tc_f,
            tip_point_camera_mm=tip_xyz_f_mm,
            tip_uv=tip_uv_f,
        )

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def _build_controls(self) -> None:
        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start_clicked)
        self.btn_start.setFocusPolicy(Qt.NoFocus)
        self.btn_start.setMinimumHeight(44)
        self.btn_start.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.btn_start.setMinimumWidth(0)

        wrap = QWidget()
        wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        row.addWidget(self.btn_start, 0, Qt.AlignTop)
        row.addStretch(1)

        self.controls_content.addWidget(wrap)

    def _update_panels(self) -> None:
        rows = self.stats_rows()
        if rows != self._last_stats_rows:
            self.set_stats_rows(rows)
            self._last_stats_rows = list(rows)

    def stats_rows(self) -> list[tuple[str, str]]:
        if not self._stats:
            return [
                ("Reprojection error", "-"),
                ("d_x", "-"),
            ]

        out: list[tuple[str, str]] = []
        for s in self._stats:
            k, v = s.split(":", 1)
            out.append((k.strip(), v.strip()))
        return out

    # ---------------------------------------------------------
    # Visualization helpers
    # ---------------------------------------------------------

    def _build_tracking_vis(
        self,
        img: np.ndarray,
        result,
        corners: list[np.ndarray],
        ids: np.ndarray | None,
        *,
        state_label: str,
        box_color=(0, 255, 0),
    ) -> np.ndarray:
        vis = _draw_aruco_overlay(img, corners, ids)

        _draw_point(
            vis,
            result.tip_uv,
            color=(0, 0, 255),
            label="tip",
        )

        vis = _draw_text_box(
            vis,
            [state_label],
            color=box_color,
        )
        return vis

    # ---------------------------------------------------------
    # Template hooks
    # ---------------------------------------------------------

    def get_frame(self) -> np.ndarray | None:
        if self._mode == "idle":
            return None

        if self._mode == "frozen":
            return self._frozen_vis

        if self.pipeline is None:
            return self._live_color

        frames = self.pipeline.poll_for_frames()
        if not frames:
            return self._live_color

        cf = frames.get_color_frame()
        if not cf:
            return self._live_color

        img = self.color_frame_to_bgr(cf)
        if img is None:
            return self._live_color

        self._live_color = img.copy()

        # -----------------------------------------------------
        # PREVIEW: only live RGB, no tracking yet
        # -----------------------------------------------------
        if self._mode == "preview":
            self._found = False
            self._last_valid_result = None
            return img

        # -----------------------------------------------------
        # TRACKING: run pointer pose estimation
        # -----------------------------------------------------
        if self._mode == "tracking":
            corners, ids = _detect_aruco(
                image=img,
                aruco_dict=self.aruco_dict,
                detector_params=self.detector_params,
            )

            try:
                raw_result = calibrate_camera_to_pointer(
                    image_bgr=img,
                    camera_intrinsics=self.state.K_rgb,
                    dist_coeffs=None,
                    pointer_model=self.pointer_model,
                    rvec_init=None,
                    tvec_init=None,
                    use_extrinsic_guess=False,
                    pose_method="ippe",
                    refine_with_iterative=True,
                )

                # apply Kalman filtering to tip position / translation
                result = self._filter_pointer_result(raw_result)

                self._last_valid_result = result
                self._found = True

                return self._build_tracking_vis(
                    img,
                    result,
                    corners,
                    ids,
                    state_label="FOUND (press SPACE)",
                    box_color=(0, 255, 0),
                )

            except Exception:
                self._last_valid_result = None
                self._found = False

                vis = img.copy()
                vis = _draw_text_box(
                    vis,
                    ["NOT FOUND"],
                    color=(0, 0, 255),
                )
                return vis

        return img

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        return frame_bgr

    # ---------------------------------------------------------
    # Actions
    # ---------------------------------------------------------

    def start_clicked(self) -> None:
        if self.state.K_rgb is None:
            QMessageBox.information(
                self,
                "Pointer Tool Depth",
                "Missing K_rgb. Run Camera Calibration first.",
            )
            return

        if self.state.T_xc is None:
            QMessageBox.information(
                self,
                "Pointer Tool Depth",
                "Missing T_xc. Complete Mode A first.",
            )
            return

        if self._mode not in ("preview", "idle"):
            return

        # clear accepted session outputs before new attempt
        self.state.d_x = None
        self.state.tip_uv_c = None
        self.state.tip_xyz_c = None
        self.state.T_tc = None
        self.state.H_xc = None

        # reset local runtime for new tracking attempt
        self._stats = []
        self._last_stats_rows = None
        self._frozen_vis = None

        self._last_valid_result = None
        self._last_proposed_result = None
        self._found = False

        # reset Kalman filter
        self._tip_kf.reset()

        self._mode = "tracking"

        if callable(self.on_complete_changed):
            self.on_complete_changed()

        self._update_buttons()
        self._update_panels()
        self.setFocus()
        self.update_view()

    def _confirm_current_pose(self) -> None:
        if self._last_valid_result is None or self._live_color is None:
            return

        self.stop_timer()
        self._last_proposed_result = self._last_valid_result

        frozen_vis = self._build_tracking_vis(
            self._live_color.copy(),
            self._last_valid_result,
            corners=[],
            ids=None,
            state_label="FROZEN",
            box_color=(0, 255, 255),
        )
        self._frozen_vis = frozen_vis
        self.update_view()

        ans = QMessageBox.question(
            self,
            "Confirm pointer pose",
            "Do you want to use this frozen pointer pose?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if ans == QMessageBox.Yes:
            self._accept_current_pose()
            return

        # reject -> back to plain live RGB preview
        self._last_valid_result = None
        self._last_proposed_result = None
        self._found = False
        self._frozen_vis = None
        self._mode = "preview"

        self.start_timer(self.FPS)

        self._update_buttons()
        self._update_panels()
        self.update_view()

    def _accept_current_pose(self) -> None:
        if self._last_proposed_result is None:
            return

        try:
            T_tc = np.asarray(self._last_proposed_result.T_4x4, dtype=np.float64).copy()

            res = extract_depth(
                T_xc=self.state.T_xc,
                T_tc=T_tc,
            )

            self.state.d_x = float(res.d_x_mm)
            self.state.tip_uv_c = self._last_proposed_result.tip_uv.astype(np.float64).copy()
            self.state.tip_xyz_c = (
                self._last_proposed_result.tip_point_camera_mm.astype(np.float64).copy()
            )
            self.state.T_tc = T_tc

            self._stats = [
                f"Reprojection error: {self._last_proposed_result.reproj_mean_px:.3f} px",
                f"d_x: {res.d_x_mm:.3f} mm",
            ]
            self._last_stats_rows = None

            self._mode = "frozen"

            if self._frozen_vis is None and self._live_color is not None:
                self._frozen_vis = self._build_tracking_vis(
                    self._live_color.copy(),
                    self._last_proposed_result,
                    corners=[],
                    ids=None,
                    state_label="ACCEPTED",
                    box_color=(0, 255, 0),
                )

            if callable(self.on_complete_changed):
                self.on_complete_changed()

            self._update_buttons()
            self._update_panels()
            self.update_view()

        except Exception as e:
            QMessageBox.warning(
                self,
                "Pointer Tool Depth",
                f"Depth computation failed.\n\n{e}",
            )

    # ---------------------------------------------------------
    # Keyboard
    # ---------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            if self._mode == "tracking" and self._found and self._last_valid_result is not None:
                self._confirm_current_pose()
            event.accept()
            return

        super().keyPressEvent(event)

    # ---------------------------------------------------------
    # Buttons
    # ---------------------------------------------------------

    def _update_buttons(self) -> None:
        self.btn_start.setEnabled(self._mode in ("idle", "preview"))

    # ---------------------------------------------------------
    # Lifecycle (Wizard)
    # ---------------------------------------------------------

    def on_enter(self) -> None:
        if self.state.K_rgb is None or self.state.T_xc is None:
            self._mode = "idle"
            self.set_viewport_background(active=False)
            self._update_buttons()
            self._update_panels()
            self.update_view()
            return

        try:
            if self.pipeline is None:
                self.start_realsense(
                    fps=self.FPS,
                    color_size=(1920, 1080),
                    depth_size=None,
                    align_to=None,
                )

            self.start_timer(self.FPS)
            self._mode = "preview"
            self.set_viewport_background(active=True)

        except Exception as e:
            self.stop_timer()
            self.stop_realsense()
            self._mode = "idle"
            self.set_viewport_background(active=False)
            QMessageBox.critical(
                self,
                "Camera",
                f"Could not open RealSense camera.\n\n{e}",
            )

        self._update_buttons()
        self._update_panels()
        self.update_view()

    def on_leave(self) -> None:
        super().on_leave()