from __future__ import annotations

import cv2
import numpy as np
import pydicom
from datetime import datetime

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSizePolicy,
    QWidget,
)

from overlay.gui.state import SessionState
from overlay.gui.pages.templates.templ_live_image import LiveImagePage

from overlay.tools.homography import estimate_plane_induced_homography
from overlay.tools.warp import blend_xray_overlay
from overlay.gui.widgets.widget_flow_layout import FlowLayout


# ============================================================
# Save config
# ============================================================

SAVE_OVERLAY_PREVIEW_RESULTS = True


# ============================================================
# Helpers
# ============================================================

def _draw_point(
    img: np.ndarray,
    uv: np.ndarray,
    *,
    color=(0, 0, 255),
    radius=8,
    cross_size=20,
    thickness=2,
) -> np.ndarray:
    out = img.copy()

    uv = np.asarray(uv, dtype=np.float64).reshape(2)
    if not np.isfinite(uv).all():
        return out

    u, v = np.round(uv).astype(int)

    cv2.circle(out, (u, v), radius, color, thickness, cv2.LINE_AA)
    cv2.drawMarker(
        out,
        (u, v),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=cross_size,
        thickness=thickness,
        line_type=cv2.LINE_AA,
    )
    return out


def _flip_xray_raw_to_working(img_gray_u8: np.ndarray) -> np.ndarray:
    """
    Convert RAW X-ray image display space into the X-ray working image space.

    The working UV convention used by K_xray, xray_points_uv, Cam2X and H_xc is:
        u_work = W - 1 - u_raw
        v_work = v_raw

    Therefore the image passed into the warp must be horizontally flipped.
    H_xc itself must NOT be modified.
    """
    img_gray_u8 = np.asarray(img_gray_u8)
    return np.ascontiguousarray(cv2.flip(img_gray_u8, 1))


# ============================================================
# Page
# ============================================================

class OverlayPreviewPage(LiveImagePage):
    """
    Step — Overlay Preview

    Behavior
    --------
    - Live RGB video is shown on this page.
    - SPACE stores a frozen RGB snapshot locally on this page.
    - Then the user can load an anatomical X-ray image.
    - Then the user can compute the overlay.
    - Alpha changes reuse the cached warped X-ray only.
    - Save Image stores the main overlay result data as NPZ.

    X-ray coordinate convention
    ---------------------------
    - Loaded X-ray images are kept in RAW detector image space internally.
    - K_xray, xray_points_uv, Cam2X and H_xc are defined in the X-ray working
      pixel space, where u is horizontally flipped relative to RAW:

          u_work = W - 1 - u_raw
          v_work = v_raw

    - Therefore, the X-ray image is horizontally flipped immediately before
      warping/blending.
    - H_xc is not changed for this visualization flip.

    Important drawing rule
    ----------------------
    The pointer tip is always drawn LAST, i.e. in the foreground above the
    X-ray overlay.
    """

    ALPHA_SLIDER_MAX = 100

    def __init__(self, state: SessionState, on_complete_changed=None, parent=None):
        self.state = state
        self.on_complete_changed = on_complete_changed

        # IMPORTANT:
        # templ_base_image may call get_frame() already during super().__init__()
        self._mode = "idle"   # idle | preview | frozen | overlay

        # -------- local page-only state --------
        self._live_color: np.ndarray | None = None

        self._xray_gray_u8: np.ndarray | None = None
        self._xray_gray_working_u8: np.ndarray | None = None
        self._snapshot_rgb_bgr: np.ndarray | None = None
        self._snapshot_rgb_with_tip_bgr: np.ndarray | None = None

        self._overlay_cache = None
        self._overlay_bgr: np.ndarray | None = None

        self._alpha: float = 0.5
        self._snapshot_taken: bool = False
        self._xray_loaded: bool = False
        self._overlay_done: bool = False

        self._last_stats_rows: list[tuple[str, str]] | None = None

        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        self.instructions_label.setText(
            "1) Live RGB video is shown automatically\n"
            "2) Press SPACE to freeze the current RGB frame\n"
            "3) Click 'Load X-ray Image'\n"
            "4) Click 'Overlay' to warp X-ray onto the frozen RGB frame\n"
            "5) Adjust alpha with the slider\n"
            "6) Click 'Next' to continue to the live overlay page"
        )

        self._build_controls()
        self.set_viewport_background(active=False)

        self._update_controls()
        self._update_panels()
        self.update_view()

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def _build_controls(self) -> None:
        self.btn_load_xray = QPushButton("Load X-ray Image")
        self.btn_load_xray.clicked.connect(self.load_xray_image)
        self.btn_load_xray.setFocusPolicy(Qt.NoFocus)
        self.btn_load_xray.setMinimumHeight(44)
        self.btn_load_xray.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        self.btn_overlay = QPushButton("Overlay")
        self.btn_overlay.clicked.connect(self.overlay_clicked)
        self.btn_overlay.setFocusPolicy(Qt.NoFocus)
        self.btn_overlay.setMinimumHeight(44)
        self.btn_overlay.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        self.btn_save_image = QPushButton("Save Image")
        self.btn_save_image.clicked.connect(self.save_image_clicked)
        self.btn_save_image.setFocusPolicy(Qt.NoFocus)
        self.btn_save_image.setMinimumHeight(44)
        self.btn_save_image.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        row_wrap = QWidget()
        row_wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        row = FlowLayout(row_wrap, spacing=12)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.btn_load_xray)
        row.addWidget(self.btn_overlay)
        row.addWidget(self.btn_save_image)

        self.controls_content.addWidget(row_wrap)

        self.lbl_alpha = QLabel("Alpha 0.500")
        self.controls_content.addWidget(self.lbl_alpha)

        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, self.ALPHA_SLIDER_MAX)
        self.slider_alpha.setValue(int(round(self._alpha * self.ALPHA_SLIDER_MAX)))
        self.slider_alpha.setSingleStep(1)
        self.slider_alpha.setPageStep(10)
        self.slider_alpha.setTracking(True)
        self.slider_alpha.valueChanged.connect(self._on_alpha_changed)
        self.controls_content.addWidget(self.slider_alpha)

    def _update_panels(self) -> None:
        rows = self.stats_rows()
        if rows != self._last_stats_rows:
            self.set_stats_rows(rows)
            self._last_stats_rows = list(rows)

    def stats_rows(self) -> list[tuple[str, str]]:
        return [
            ("Alpha", f"{self._alpha:.3f}" if self._overlay_done else "-"),
            ("d_x", f"{self.state.d_x:.3f} mm" if self.state.d_x is not None else "-"),
        ]

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _has_prerequisites(self) -> bool:
        return (
            self.state.tip_uv_c is not None
            and self.state.K_rgb is not None
            and self.state.K_xray is not None
            and self.state.T_xc is not None
            and self.state.d_x is not None
        )

    def _draw_tip_on_top(self, img: np.ndarray) -> np.ndarray:
        return _draw_point(
            img,
            self.state.tip_uv_c,
            color=(0, 0, 255),
            radius=8,
            cross_size=20,
            thickness=2,
        )

    def _build_snapshot_base(self) -> None:
        self._snapshot_rgb_with_tip_bgr = None

        if self._snapshot_rgb_bgr is None:
            return

        self._snapshot_rgb_with_tip_bgr = self._draw_tip_on_top(self._snapshot_rgb_bgr)

    def _current_display_image(self) -> np.ndarray | None:
        if self._overlay_done and self._overlay_bgr is not None:
            return self._overlay_bgr

        if self._snapshot_taken and self._snapshot_rgb_with_tip_bgr is not None:
            return self._snapshot_rgb_with_tip_bgr

        return self._live_color

    def _update_controls(self) -> None:
        prereq_ok = self._has_prerequisites()

        self.btn_load_xray.setEnabled(prereq_ok and self._snapshot_taken)
        self.btn_overlay.setEnabled(prereq_ok and self._snapshot_taken and self._xray_loaded)
        self.btn_save_image.setEnabled(self._overlay_done and self._overlay_bgr is not None)
        self.slider_alpha.setEnabled(self._overlay_done)

        self.lbl_alpha.setText(f"Alpha {self._alpha:.3f}")

    def _compute_H_xc(self) -> np.ndarray:
        T_xc = np.asarray(self.state.T_xc, dtype=np.float64)

        R_xc = T_xc[:3, :3]
        t_xc = T_xc[:3, 3]

        H_xc = estimate_plane_induced_homography(
            K_c=np.asarray(self.state.K_rgb, dtype=np.float64),
            R_xc=R_xc,
            t_xc=t_xc,
            K_x=np.asarray(self.state.K_xray, dtype=np.float64),
            d_x=float(self.state.d_x),
        )

        return H_xc

    def _reset_overlay_state(self) -> None:
        self._overlay_cache = None
        self._overlay_bgr = None
        self._overlay_done = False
        self._alpha = 0.5

        self.slider_alpha.blockSignals(True)
        self.slider_alpha.setValue(int(round(self._alpha * self.ALPHA_SLIDER_MAX)))
        self.slider_alpha.blockSignals(False)

        self.state.H_xc = None

    def _save_overlay_results(self) -> Path | None:
        if not SAVE_OVERLAY_PREVIEW_RESULTS:
            return None

        if not self._overlay_done or self._overlay_bgr is None:
            return None

        out_dir = Path.cwd()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"overlay_preview_{stamp}.npz"

        xray_image_path = (
            self.state.xray_image_anatomy_path
            if self.state.xray_image_anatomy_path is not None
            else ""
        )

        np.savez(
            out_path,

            # -------------------------------------------------
            # Main result images
            # -------------------------------------------------
            snapshot_rgb_bgr=self._snapshot_rgb_bgr,
            snapshot_rgb_with_tip_bgr=self._snapshot_rgb_with_tip_bgr,
            xray_gray_u8=self._xray_gray_u8,
            xray_gray_working_u8=self._xray_gray_working_u8,
            overlay_bgr=self._overlay_bgr,

            # -------------------------------------------------
            # Main overlay result
            # -------------------------------------------------
            H_xc=self.state.H_xc,
            alpha=np.array(self._alpha, dtype=np.float64),
            d_x=np.array(
                self.state.d_x if self.state.d_x is not None else np.nan,
                dtype=np.float64,
            ),

            # -------------------------------------------------
            # Calibration / pose
            # -------------------------------------------------
            K_rgb=self.state.K_rgb,
            dist_rgb=self.state.dist_rgb,
            K_xray=self.state.K_xray,
            T_cx=self.state.T_cx,
            T_xc=self.state.T_xc,
            T_tc=self.state.T_tc,

            # -------------------------------------------------
            # Pointer
            # -------------------------------------------------
            tip_uv_c=self.state.tip_uv_c,
            tip_xyz_c=self.state.tip_xyz_c,

            # -------------------------------------------------
            # Correspondences from previous steps
            # -------------------------------------------------
            xray_points_uv=self.state.xray_points_uv,
            xray_points_xyz_c=self.state.xray_points_xyz_c,
            checkerboard_corners_uv=self.state.checkerboard_corners_uv,
            checkerboard_corners_uv_9=self.state.checkerboard_corners_uv_9,

            # -------------------------------------------------
            # Metadata / bookkeeping
            # -------------------------------------------------
            xray_image_path=np.array(xray_image_path, dtype=object),
            xray_image_space=np.array("XRAY_RAW", dtype="<U64"),
            xray_warp_image_space=np.array("XRAY_WORKING_FLIPPED_UV", dtype="<U64"),
            xray_working_transform=np.array(
                "u_work = W - 1 - u_raw, v_work = v_raw",
                dtype="<U64",
            ),
            snapshot_taken=np.array(self._snapshot_taken, dtype=bool),
            xray_loaded=np.array(self._xray_loaded, dtype=bool),
            overlay_done=np.array(self._overlay_done, dtype=bool),
        )

        return out_path

    # ---------------------------------------------------------
    # Template hooks
    # ---------------------------------------------------------

    def get_frame(self) -> np.ndarray | None:
        if self._mode == "idle":
            return None

        if self._mode == "preview":
            pipe = getattr(self, "pipeline", None)
            if pipe is None:
                return self._live_color

            frames = pipe.poll_for_frames()
            if not frames:
                return self._live_color

            cf = frames.get_color_frame()
            if not cf:
                return self._live_color

            img = self.color_frame_to_bgr(cf)
            if img is None:
                return self._live_color

            self._live_color = img
            return img

        return self._current_display_image()

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        return frame_bgr

    # ---------------------------------------------------------
    # Keyboard
    # ---------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            if self._mode == "preview":
                self.on_space_pressed()
            event.accept()
            return

        super().keyPressEvent(event)

    def on_space_pressed(self) -> None:
        if not self._has_prerequisites():
            QMessageBox.information(
                self,
                "Overlay Preview",
                "Missing prerequisites from previous steps.",
            )
            return

        if self._live_color is None:
            QMessageBox.information(
                self,
                "Overlay Preview",
                "No live RGB frame available.",
            )
            return

        self._snapshot_rgb_bgr = self._live_color.copy()
        self._snapshot_taken = True

        # new snapshot invalidates previous xray/overlay state
        self._xray_gray_u8 = None
        self._xray_gray_working_u8 = None
        self._xray_loaded = False
        self._build_snapshot_base()
        self._reset_overlay_state()

        self._mode = "frozen"

        self._update_controls()
        self._update_panels()
        self.update_view()

        if callable(self.on_complete_changed):
            self.on_complete_changed()

    # ---------------------------------------------------------
    # Actions
    # ---------------------------------------------------------

    def load_xray_image(self) -> None:
        if not self._has_prerequisites():
            QMessageBox.information(
                self,
                "Overlay Preview",
                "Missing prerequisites from previous steps.",
            )
            return

        if not self._snapshot_taken:
            QMessageBox.information(
                self,
                "Overlay Preview",
                "Please press SPACE first to freeze the RGB image.",
            )
            return

        dlg = QFileDialog(self.window(), "Select X-ray image")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter(
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.ima);;All Files (*)"
        )
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)

        if dlg.exec() != QFileDialog.Accepted:
            return

        files = dlg.selectedFiles()
        path = files[0] if files else ""
        if not path:
            return

        try:
            if path.lower().endswith((".dcm", ".ima")):
                ds = pydicom.dcmread(path)
                img = ds.pixel_array.astype(np.float32)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img = img.astype(np.uint8)
            else:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError("Could not read image.")

            self._xray_gray_u8 = img
            self._xray_gray_working_u8 = None
            self.state.xray_image_anatomy = img.copy()
            self.state.xray_image_anatomy_path = path

            self._overlay_cache = None
            self._overlay_bgr = None
            self._overlay_done = False
            self._xray_loaded = True
            self._alpha = 0.5

            self.slider_alpha.blockSignals(True)
            self.slider_alpha.setValue(int(round(self._alpha * self.ALPHA_SLIDER_MAX)))
            self.slider_alpha.blockSignals(False)

            self.state.H_xc = None
            self._mode = "frozen"

            self._update_controls()
            self._update_panels()
            self.update_view()

            if callable(self.on_complete_changed):
                self.on_complete_changed()

        except Exception as e:
            QMessageBox.critical(self, "Failed to load X-ray image", str(e))

    def overlay_clicked(self) -> None:
        if not self._has_prerequisites():
            QMessageBox.information(
                self,
                "Overlay Preview",
                "Missing prerequisites from previous steps.",
            )
            return

        if self._xray_gray_u8 is None or self._snapshot_rgb_bgr is None:
            QMessageBox.information(
                self,
                "Overlay Preview",
                "Please press SPACE first and load an X-ray image.",
            )
            return

        try:
            H_xc = self._compute_H_xc()
            self.state.H_xc = H_xc

            # H_xc and K_xray are defined in XRAY_WORKING_FLIPPED_UV space.
            # The loaded anatomy image is RAW, so flip the image before warping.
            # Do NOT modify H_xc.
            # The pose was estimated on corrected marker points, so H_xc maps
            # IDEAL X-ray pixels. The image has to be brought into that same
            # frame, otherwise pose and image disagree.
            working = _flip_xray_raw_to_working(self._xray_gray_u8)
            field = self.state.xray_dist_field
            if field is not None:
                working = field.undistort_image_working(working)
            self._xray_gray_working_u8 = working

            # H_xc maps onto IDEAL pinhole pixels; the snapshot is distorted,
            # so the lens model has to be part of the warp.
            out_bgr, cache = blend_xray_overlay(
                camera_bgr=self._snapshot_rgb_bgr,
                xray_gray_u8=self._xray_gray_working_u8,
                H_xc=self.state.H_xc,
                alpha=self._alpha,
                K_camera=self.state.K_rgb,
                dist_coeffs=self.state.dist_rgb,
            )

            out_bgr = self._draw_tip_on_top(out_bgr)

            self._overlay_cache = cache
            self._overlay_bgr = out_bgr
            self._overlay_done = True
            self._mode = "overlay"

            self._update_controls()
            self._update_panels()
            self.update_view()

            if callable(self.on_complete_changed):
                self.on_complete_changed()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Overlay failed",
                str(e),
            )

    def save_image_clicked(self) -> None:
        if not self._overlay_done or self._overlay_bgr is None:
            QMessageBox.information(
                self,
                "Save Overlay Results",
                "Please create the overlay first.",
            )
            return

        try:
            out_path = self._save_overlay_results()

            if out_path is None:
                QMessageBox.information(
                    self,
                    "Save Overlay Results",
                    "Saving is disabled.",
                )
                return

            QMessageBox.information(
                self,
                "Save Overlay Results",
                f"Overlay results saved successfully:\n\n{out_path}",
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Overlay Results",
                f"Failed to save overlay results.\n\n{e}",
            )

    def _on_alpha_changed(self, value: int) -> None:
        self._alpha = float(value) / float(self.ALPHA_SLIDER_MAX)

        if (
            self._overlay_done
            and self._overlay_cache is not None
            and self._snapshot_rgb_bgr is not None
        ):
            try:
                out_bgr = self._overlay_cache.blend(
                    self._snapshot_rgb_bgr,
                    alpha=self._alpha,
                )
                self._overlay_bgr = self._draw_tip_on_top(out_bgr)
            except Exception as e:
                QMessageBox.critical(self, "Alpha update failed", str(e))
                return

        self._update_controls()
        self._update_panels()
        self.update_view()

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    def on_enter(self) -> None:
        if not self._has_prerequisites():
            self._reset_page_state()
            self._mode = "idle"
            self.set_viewport_background(active=False)
            self._update_controls()
            self._update_panels()
            self.update_view()
            return

        try:
            if getattr(self, "pipeline", None) is None:
                self.start_realsense(
                    fps=self.FPS,
                    color_size=(1920, 1080),
                    depth_size=None,
                    align_to=None,
                )

            self._reset_page_state()

            self.start_timer(self.FPS)
            self._mode = "preview"
            self.set_viewport_background(active=True)
            self.setFocus()

        except Exception as e:
            self.stop_timer()
            self.stop_realsense()
            self._reset_page_state()
            self._mode = "idle"
            self.set_viewport_background(active=False)
            QMessageBox.critical(
                self,
                "Camera",
                f"Could not open RealSense camera.\n\n{e}",
            )

        self._update_controls()
        self._update_panels()
        self.update_view()

    def on_leave(self) -> None:
        super().on_leave()

    def _reset_page_state(self) -> None:
        self._xray_gray_u8 = None
        self._xray_gray_working_u8 = None
        self._snapshot_rgb_bgr = None
        self._snapshot_rgb_with_tip_bgr = None

        self._overlay_cache = None
        self._overlay_bgr = None

        self._alpha = 0.5
        self._snapshot_taken = False
        self._xray_loaded = False
        self._overlay_done = False

        self.slider_alpha.blockSignals(True)
        self.slider_alpha.setValue(int(round(self._alpha * self.ALPHA_SLIDER_MAX)))
        self.slider_alpha.blockSignals(False)

        self.state.H_xc = None
        self._last_stats_rows = None
