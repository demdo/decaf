# overlay/gui/pages/page_camera_to_xray_calibration.py

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np

from PySide6.QtWidgets import (
    QPushButton,
    QMessageBox,
    QSizePolicy,
    QCheckBox,
)

from overlay.gui.state import SessionState
from overlay.gui.pages.templates.templ_static_image import StaticImagePage
from overlay.calib.calib_camera_to_xray import calibrate_camera_to_xray
from overlay.gui.widgets.widget_zoom_view import ZoomView


# ============================================================================
# Global save switch
# ============================================================================
SAVE_CAM2X_RESULTS: bool = False

# ============================================================
# Calibration target geometry
# ============================================================
# The target is a single-sided PCB: the checkerboard is printed on the top
# face, while the X-ray-visible feature is the hole in the copper on the
# solder side. The RGB and X-ray sides therefore observe two planes one board
# thickness apart; without this the offset is absorbed into T_cx.
PCB_THICKNESS_MM: float = 1.6


class CameraToXrayCalibrationPage(StaticImagePage):
    """
    Step — Camera → X-ray calibration

    IMPORTANT
    ---------
    - MUST NOT add any new SessionState fields dynamically.
    - SessionState fields used here (already defined in state.py):
        * K_xray
        * K_rgb
        * dist_rgb
        * xray_image
        * xray_points_uv, xray_points_confirmed
        * xray_points_xyz_c           (used for iterative / iterative_ransac / ippe / ippe_handeye)
        * checkerboard_corners_uv     (used for ippe_handeye)
        * checkerboard_corners_confirmed
        * pnp_ransac_threshold_px
        * T_cx, T_xc
        * marker_radius_px
    - All calibration diagnostics/overlays live ONLY in this page as self._... locals.
    """

    def __init__(self, state: SessionState, on_complete_changed, parent=None):
        super().__init__(parent)

        self.state = state
        self.on_complete_changed = on_complete_changed

        # ---------------- local (page-only) cache ----------------
        self._calibration_done: bool = False
        self._pose_method: str = "iterative_ransac"  # default

        self._uv_projected: np.ndarray | None = None
        self._inlier_idx: np.ndarray | None = None
        self._reproj_errors_px: np.ndarray | None = None
        self._reproj_mean_px: float | None = None
        self._reproj_median_px: float | None = None
        self._reproj_p95_in_px: float | None = None
        self._n_points: int | None = None
        self._n_inliers: int | None = None
        self._R_cx: np.ndarray | None = None
        self._t_cx: np.ndarray | None = None

        # ======================================================
        # LEFT: replace template image_label with zoomable view
        # ======================================================
        self.zoom_view = ZoomView(self.image_label)
        self.zoom_view.setGeometry(0, 0, self.image_label.width(), self.image_label.height())
        self.zoom_view.zoom_changed.connect(self._update_zoom_button)
        self.zoom_view.show()

        # ======================================================
        # RIGHT: controls
        # ======================================================
        self.btn_run = QPushButton("Run Calibration")
        self.btn_run.clicked.connect(self.on_run_calibration)

        self.chk_method_ransac = QCheckBox("RANSAC")
        self.chk_method_ippe = QCheckBox("IPPE")
        self.chk_method_ippe_handeye = QCheckBox("IPPE (Handeye)")

        self.chk_method_ransac.setChecked(True)
        self.chk_method_ippe.setChecked(False)
        self.chk_method_ippe_handeye.setChecked(False)

        self.chk_method_ransac.toggled.connect(
            lambda checked: self._on_method_toggled("iterative_ransac", checked)
        )
        self.chk_method_ippe.toggled.connect(
            lambda checked: self._on_method_toggled("ippe", checked)
        )
        self.chk_method_ippe_handeye.toggled.connect(
            lambda checked: self._on_method_toggled("ippe_handeye", checked)
        )

        self.btn_reset_zoom = QPushButton("Reset zoom")
        self.btn_reset_zoom.clicked.connect(self.on_reset_zoom)

        for w in (
            self.btn_run,
            self.chk_method_ransac,
            self.chk_method_ippe,
            self.chk_method_ippe_handeye,
            self.btn_reset_zoom,
        ):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.controls_content.addWidget(w)

        # ======================================================
        # Instructions
        # ======================================================
        self.instructions_label.setText(
            "1) Ensure prerequisites are complete (intrinsics, plane, marker ROI)\n"
            "2) Select calibration method\n"
            "3) Click 'Run Calibration'\n"
            "4) Inspect measured (red) vs projected (cyan) points in the X-ray image\n"
            "Tip: Drag a SQUARE box with LMB to zoom, RMB/double-click to reset"
        )
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.instructions_label.adjustSize()

        self.refresh()

    # ---------------- Completion ----------------

    def is_complete(self) -> bool:
        return self.state.T_cx is not None

    # ---------------- QWidget lifecycle ----------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "zoom_view"):
            self.zoom_view.setGeometry(0, 0, self.image_label.width(), self.image_label.height())
        if hasattr(self, "instructions_label"):
            self.instructions_label.updateGeometry()
            self.instructions_label.adjustSize()

    def on_enter(self) -> None:
        self.refresh()

    # ---------------- Method selection ----------------

    def _on_method_toggled(self, method: str, checked: bool) -> None:
        """
        Keep the three checkboxes mutually exclusive.
        If the active method is unchecked and nothing remains selected,
        revert to RANSAC.
        """
        if checked:
            self._pose_method = method

            self.chk_method_ransac.blockSignals(True)
            self.chk_method_ippe.blockSignals(True)
            self.chk_method_ippe_handeye.blockSignals(True)

            self.chk_method_ransac.setChecked(method == "iterative_ransac")
            self.chk_method_ippe.setChecked(method == "ippe")
            self.chk_method_ippe_handeye.setChecked(method == "ippe_handeye")

            self.chk_method_ransac.blockSignals(False)
            self.chk_method_ippe.blockSignals(False)
            self.chk_method_ippe_handeye.blockSignals(False)

            self.refresh()
            return

        if not (
            self.chk_method_ransac.isChecked()
            or self.chk_method_ippe.isChecked()
            or self.chk_method_ippe_handeye.isChecked()
        ):
            self.chk_method_ransac.blockSignals(True)
            self.chk_method_ransac.setChecked(True)
            self.chk_method_ransac.blockSignals(False)
            self._pose_method = "iterative_ransac"

        self.refresh()

    # ---------------- Stats helpers ----------------

    def _default_stats_rows(self) -> list[tuple[str, str]]:
        return [
            ("Median", "-"),
            ("P95", "-"),
            ("Avg. reprojection error", "-"),
            ("R_cx", "-"),
            ("t_cx", "-"),
        ]

    def _result_stats_rows(self) -> list[tuple[str, str]]:
        med = self._reproj_median_px
        mean = self._reproj_mean_px
        p95 = self._reproj_p95_in_px

        rows: list[tuple[str, str]] = [
            ("Median", "-" if med is None else f"{float(med):.3f} px"),
            ("P95", "-" if p95 is None else f"{float(p95):.3f} px"),
            ("Avg. reprojection error", "-" if mean is None else f"{float(mean):.3f} px"),
        ]

        if self._R_cx is not None:
            R = self._R_cx
            R_lines = "\n".join(
                " ".join(f"{v:+.4f}" for v in row) for row in R
            )
            rows.append(("R_cx", "np.array (3×3)\n" + R_lines))
        else:
            rows.append(("R_cx", "-"))

        if self._t_cx is not None:
            t = self._t_cx.reshape(3)
            rows.append(("t_cx", f"[{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]^T"))
        else:
            rows.append(("t_cx", "-"))

        return rows

    # ---------------- Refresh ----------------

    def refresh(self) -> None:
        # state.xray_image remains RAW.
        # state.xray_points_uv is stored in XRAY_WORKING_FLIPPED_UV.
        # Therefore, only the DISPLAY image is flipped here so that the
        # measured/projected Working-Space UVs are visualized in the
        # same pixel coordinate system used for Cam2X calibration.
        img = self.state.xray_image
        if img is not None:
            img = np.ascontiguousarray(np.fliplr(img))

        if hasattr(self, "instructions_label"):
            self.instructions_label.updateGeometry()
            self.instructions_label.adjustSize()

        if img is None:
            self.set_viewport_background(active=False)
            self.zoom_view.set_image(None)
            self.zoom_view.clear_overlay_data()
            self.zoom_view.reset_zoom()
            self.zoom_view.setEnabled(False)
            self.set_stats_rows(self._default_stats_rows())
            self.btn_run.setEnabled(False)
            self._update_zoom_button()
            return

        self.set_viewport_background(active=True)
        self.zoom_view.set_image(img)

        marker_r = self.state.marker_radius_px
        pick_r = 0.6 * float(marker_r) if marker_r is not None else 20.0

        uv_roi = self.state.xray_points_uv
        uv_meas = self.state.xray_points_uv

        calibration_done = bool(self._calibration_done and self._uv_projected is not None)

        outlier_mask = None
        if calibration_done and self._inlier_idx is not None and uv_meas is not None:
            uv_meas_arr = np.asarray(uv_meas, dtype=np.float64).reshape(-1, 2)
            n = uv_meas_arr.shape[0]
            mask_in = np.zeros(n, dtype=bool)
            idx = np.asarray(self._inlier_idx, dtype=np.int64).reshape(-1)
            idx = idx[(idx >= 0) & (idx < n)]
            mask_in[idx] = True
            outlier_mask = ~mask_in

        if calibration_done:
            self.zoom_view.set_overlay_data(
                uv_roi=uv_roi,
                uv_measured=uv_meas,
                uv_projected=self._uv_projected,
                pick_radius_px=pick_r,
                marker_radius_px=1.2 * marker_r if marker_r is not None else None,
                show_residuals=True,
                outlier_mask=outlier_mask,
            )
            self.zoom_view.setEnabled(True)
        else:
            self.zoom_view.set_overlay_data(
                uv_roi=uv_roi,
                uv_measured=uv_meas,
                uv_projected=None,
                pick_radius_px=pick_r,
                marker_radius_px=1.2 * marker_r if marker_r is not None else None,
                show_residuals=False,
                outlier_mask=None,
            )
            self.zoom_view.reset_zoom()
            self.zoom_view.setEnabled(False)

        miss = self._missing()
        self.btn_run.setEnabled(len(miss) == 0)

        self.set_stats_rows(
            self._result_stats_rows() if calibration_done else self._default_stats_rows()
        )
        self._update_zoom_button()

    # ---------------- Zoom helpers ----------------

    def on_reset_zoom(self) -> None:
        self.zoom_view.reset_zoom()
        self._update_zoom_button()

    def _update_zoom_button(self) -> None:
        zoomed = self.zoom_view.has_zoom()
        self.btn_reset_zoom.setEnabled(bool(self._calibration_done and zoomed))

    # ---------------- Save ----------------

    def _save_results(
        self,
        Kx: np.ndarray,
        uv_meas: np.ndarray,
        xyz_c: np.ndarray | None,
    ) -> None:
        if not SAVE_CAM2X_RESULTS:
            return

        out_dir = Path("debug_cam2x")
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_tag = self._pose_method.replace(" ", "_")
        out_path = out_dir / f"cam2x_calibration_{timestamp}_{method_tag}.npz"

        img = None
        if self.state.xray_image is not None:
            img = np.asarray(self.state.xray_image)

        np.savez(
            out_path,
            pose_method=np.array(self._pose_method),
            ransac_threshold_px=np.array(
                self.state.pnp_ransac_threshold_px
                if self.state.pnp_ransac_threshold_px is not None else np.nan,
                dtype=np.float64,
            ),
            T_cx=np.asarray(self.state.T_cx, dtype=np.float64),
            T_xc=np.asarray(self.state.T_xc, dtype=np.float64),
            R_cx=np.asarray(self._R_cx, dtype=np.float64) if self._R_cx is not None else np.empty((0,)),
            t_cx=np.asarray(self._t_cx, dtype=np.float64) if self._t_cx is not None else np.empty((0,)),
            reproj_errors_px=np.asarray(self._reproj_errors_px, dtype=np.float64)
            if self._reproj_errors_px is not None else np.empty((0,)),
            reproj_mean_px=np.array(
                self._reproj_mean_px if self._reproj_mean_px is not None else np.nan,
                dtype=np.float64,
            ),
            reproj_median_px=np.array(
                self._reproj_median_px if self._reproj_median_px is not None else np.nan,
                dtype=np.float64,
            ),
            reproj_p95_in_px=np.array(
                self._reproj_p95_in_px if self._reproj_p95_in_px is not None else np.nan,
                dtype=np.float64,
            ),
            inlier_idx=np.asarray(self._inlier_idx, dtype=np.int64)
            if self._inlier_idx is not None else np.empty((0,), dtype=np.int64),
            n_inliers=np.array(self._n_inliers if self._n_inliers is not None else -1, dtype=np.int64),
            n_points=np.array(self._n_points if self._n_points is not None else -1, dtype=np.int64),
            K_xray=np.asarray(Kx, dtype=np.float64),
            points_uv_xray=np.asarray(uv_meas, dtype=np.float64),
            points_xyz_camera=np.asarray(xyz_c, dtype=np.float64)
            if xyz_c is not None else np.empty((0, 3)),
            uv_projected=np.asarray(self._uv_projected, dtype=np.float64)
            if self._uv_projected is not None else np.empty((0, 2)),
            xray_image=img if img is not None else np.empty((0,)),
        )

        print(f"[INFO] Saved Camera→X-ray calibration to: {out_path}")

    # ---------------- Action ----------------

    def on_run_calibration(self) -> None:
        miss = self._missing()
        if miss:
            QMessageBox.warning(
                self,
                "Calibration",
                "Cannot run calibration.\n\nMissing: " + ", ".join(miss),
            )
            return

        try:
            Kx = np.asarray(self.state.K_xray, dtype=np.float64)
            uv_meas = np.asarray(self.state.xray_points_uv, dtype=np.float64).reshape(-1, 2)

            # The intensifier distortion cannot be expressed as dist_coeffs
            # (see overlay.tools.xray_distortion), so the marker points are
            # corrected here and the solver sees an ideal pinhole detector.
            field = getattr(self.state, "xray_dist_field", None)
            if field is not None:
                uv_meas = field.undistort_working(uv_meas)

            # Full board lattice for the X-ray-side pose (2026-09-15). Same
            # space and same correction as the patch points; None keeps the
            # legacy patch-only path in the solver.
            lattice_uv = None
            lattice_xy = None
            if self.state.xray_lattice_uv is not None and self.state.xray_lattice_xy_mm is not None:
                lattice_uv = np.asarray(self.state.xray_lattice_uv, dtype=np.float64).reshape(-1, 2)
                if field is not None:
                    lattice_uv = field.undistort_working(lattice_uv)
                lattice_xy = np.asarray(self.state.xray_lattice_xy_mm, dtype=np.float64).reshape(-1, 2)
            xyz_c = np.asarray(self.state.xray_points_xyz_c, dtype=np.float64).reshape(-1, 3)

            ransac_thr = 1.5
            self.state.pnp_ransac_threshold_px = float(ransac_thr)

            if xyz_c.shape[0] != uv_meas.shape[0]:
                QMessageBox.warning(
                    self,
                    "Point count mismatch",
                    f"2D points: {uv_meas.shape[0]}\n"
                    f"3D points: {xyz_c.shape[0]}\n\n"
                    "This step expects the same number of 2D and 3D correspondences.",
                )
                return

            if self._pose_method == "ippe_handeye":
                dist_rgb = None
                if self.state.dist_rgb is not None:
                    dist_rgb = np.asarray(self.state.dist_rgb, dtype=np.float64)

                calib = calibrate_camera_to_xray(
                    K_xray=Kx,
                    points_xyz_camera=xyz_c,
                    points_uv_xray=uv_meas,
                    dist_coeffs=None,
                    dist_coeffs_rgb=dist_rgb,
                    pose_method="ippe_handeye",
                    refine_with_iterative=True,
                    pitch_mm=2.54,
                    checkerboard_corners_uv=np.asarray(
                        self.state.checkerboard_corners_uv, dtype=np.float64
                    ),
                    K_rgb=np.asarray(self.state.K_rgb, dtype=np.float64),
                    steps_per_edge=10,
                    board_thickness_mm=PCB_THICKNESS_MM,
                    xray_lattice_uv=lattice_uv,
                    xray_lattice_xy_mm=lattice_xy,
                )
            else:
                calib = calibrate_camera_to_xray(
                    K_xray=Kx,
                    points_xyz_camera=xyz_c,
                    points_uv_xray=uv_meas,
                    dist_coeffs=None,
                    pose_method=self._pose_method,
                    refine_with_iterative=False,
                    ransac_reprojection_error_px=ransac_thr,
                    ransac_confidence=0.99,
                    ransac_iterations_count=5000,
                )

            # --------- persist ONLY allowed SessionState fields ----------
            self.state.T_cx = calib.T_cx
            self.state.T_xc = calib.T_xc

            # --------- local diagnostics (page-only) ----------
            self._calibration_done = True
            self._n_points = int(uv_meas.shape[0])
            self._R_cx = np.asarray(calib.rotation, dtype=np.float64)
            self._t_cx = np.asarray(calib.translation, dtype=np.float64).reshape(3)

            self._inlier_idx = np.asarray(calib.inlier_idx, dtype=np.int64).reshape(-1)
            self._n_inliers = int(self._inlier_idx.size)

            self._reproj_errors_px = np.asarray(calib.reproj_errors_px, dtype=np.float64).reshape(-1)
            self._reproj_mean_px = float(calib.reproj_mean_px)
            self._reproj_median_px = float(calib.reproj_median_px)

            self._reproj_p95_in_px = None
            if self._reproj_errors_px.size > 0 and self._inlier_idx.size > 0:
                idx = self._inlier_idx.copy()
                idx = idx[(idx >= 0) & (idx < self._reproj_errors_px.size)]
                if idx.size > 0:
                    e_in = self._reproj_errors_px[idx]
                    if e_in.size > 0:
                        self._reproj_p95_in_px = float(np.percentile(e_in, 95))

            self._uv_projected = np.asarray(calib.uv_proj, dtype=np.float64).reshape(-1, 2)

            self._save_results(Kx=Kx, uv_meas=uv_meas, xyz_c=xyz_c)

            self.zoom_view.reset_zoom()
            self.refresh()
            self.on_complete_changed()

        except Exception as e:
            QMessageBox.critical(self, "Calibration failed", str(e))

    # ---------------- Helpers ----------------

    def _missing(self) -> list[str]:
        m: list[str] = []

        if self.state.K_xray is None:
            m.append("K_xray")

        uv = self.state.xray_points_uv
        if uv is None or len(uv) == 0:
            m.append("xray_points_uv")
        if not bool(self.state.xray_points_confirmed):
            m.append("xray_points_confirmed")

        xyz = self.state.xray_points_xyz_c
        if xyz is None or len(xyz) == 0:
            m.append("xray_points_xyz_c")

        if self._pose_method == "ippe_handeye":
            if not self.state.has_checkerboard_corners:
                m.append("checkerboard_corners_uv")
            if self.state.K_rgb is None:
                m.append("K_rgb")

        return m