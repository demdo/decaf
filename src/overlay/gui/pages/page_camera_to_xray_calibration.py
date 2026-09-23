# overlay/gui/pages/page_camera_to_xray_calibration.py

from __future__ import annotations

import hashlib

import numpy as np

from PySide6.QtWidgets import (
    QPushButton,
    QMessageBox,
    QSizePolicy,
)

from overlay.gui.state import SessionState, Cam2XHeight
from overlay.gui.pages.templates.templ_static_image import StaticImagePage
from overlay.calib.calib_camera_to_xray import (
    Cam2XBoardCapture,
    calibrate_camera_to_xray,
    calibrate_camera_to_xray_pooled,
    leave_one_capture_out,
)
from overlay.gui.widgets.widget_zoom_view import ZoomView
from overlay.tools.xray_io import xray_pixel_md5


# ============================================================
# Calibration target geometry
# ============================================================
# The target is a single-sided PCB: the checkerboard is printed on the top
# face, while the X-ray-visible feature is the hole in the copper on the
# solder side. The RGB and X-ray sides therefore observe two planes one board
# thickness apart; without this the offset is absorbed into T_cx.
PCB_THICKNESS_MM: float = 1.6

# ============================================================
# Board placements of one calibration series
# ============================================================
# The board is captured flat at N_HEIGHTS table heights (~0 / +100 / +200 mm;
# camera and C-arm untouched) and all placements go into ONE PnP
# (calibrate_camera_to_xray_pooled). A single flat placement fixes the
# camera-side board tilt only to ~2 deg, and the single-placement
# composition carries that tilt 1:1 into T_cx: overlay drift 3.7 mm mean /
# 7.6 mm max over 220 mm of depth; pooled over three heights 0.33 / 0.77 mm
# on the same pin planes (evaluation 2026-09-22).
N_HEIGHTS: int = 3


def _rotation_deg(T_a: np.ndarray, T_b: np.ndarray) -> float:
    R = np.asarray(T_a, dtype=np.float64)[:3, :3] @ np.asarray(T_b, dtype=np.float64)[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


class CameraToXrayCalibrationPage(StaticImagePage):
    """
    Step — Camera → X-ray calibration, one table height at a time

    For the current height k (steps 3 and 4 done for it) the page solves the
    placement on its own as soon as it is entered (pose_method
    'ippe_handeye': camera-side board pose from the nine corners, X-ray side
    from the hole lattice). 'Next height' keeps it (commit_height, wired in
    main.py) and the wizard returns to step 3; at height N_HEIGHTS all
    placements are pooled into T_cx / T_xc. Only the pooled solve writes
    T_cx / T_xc.

    IMPORTANT
    ---------
    - MUST NOT add any new SessionState fields dynamically.
    - SessionState fields used here (already defined in state.py):
        * K_xray
        * K_rgb
        * dist_rgb
        * xray_dist_field
        * xray_image
        * xray_points_uv, xray_points_confirmed
        * xray_lattice_uv, xray_lattice_xy_mm
        * xray_points_xyz_c           (depth reference of the camera-side pose)
        * checkerboard_corners_uv     (used for ippe_handeye: TL/TR/BL, frame of the patch)
        * checkerboard_corners_uv_9   (used for ippe_handeye: homography of the nine corners)
        * checkerboard_corners_confirmed
        * cam2x_heights               (finished heights; appended by commit_height)
        * T_cx, T_xc                  (pooled result only)
        * marker_radius_px
    - All calibration diagnostics/overlays live ONLY in this page as self._... locals.
    """

    def __init__(self, state: SessionState, on_complete_changed, parent=None):
        super().__init__(parent)

        self.state = state
        self.on_complete_changed = on_complete_changed

        # ---------------- local (page-only) cache ----------------
        # current height: solved, not yet committed to state.cam2x_heights
        self._pending: Cam2XHeight | None = None
        self._pending_key: str | None = None   # fingerprint of the inputs it was solved from
        # last height only: the pooled solve and its leave-one-out check
        self._pooled = None                    # PooledCameraToXrayResult
        self._pooled_K: np.ndarray | None = None  # K_xray it was solved with
        self._loo: list[dict] | None = None

        # current height alone (image overlay + stats)
        self._uv_projected: np.ndarray | None = None
        self._reproj_mean_px: float | None = None
        self._reproj_median_px: float | None = None
        self._reproj_p95_in_px: float | None = None
        self._last_calib = None  # CameraToXrayCalibrationResult of the current height

        # ======================================================
        # LEFT: replace template image_label with zoomable view
        # ======================================================
        self.zoom_view = ZoomView(self.image_label)
        self.zoom_view.setGeometry(0, 0, self.image_label.width(), self.image_label.height())
        self.zoom_view.zoom_changed.connect(self._update_zoom_button)
        self.zoom_view.show()

        # ======================================================
        # RIGHT: controls ('Next height' is wired in main.py; 'Redo height'
        # and 'New series' sit next to the step title, also in steps 3/4)
        # ======================================================
        self.btn_next_height = QPushButton("Next height")
        self.btn_next_height.setToolTip(
            "Keep this height and go back to step 3 for the next one: raise the "
            "table ~100 mm, board flat on it; camera and C-arm stay where they are."
        )

        self.btn_reset_zoom = QPushButton("Reset zoom")
        self.btn_reset_zoom.clicked.connect(self.on_reset_zoom)

        for w in (
            self.btn_next_height,
            self.btn_reset_zoom,
        ):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.controls_content.addWidget(w)

        # ======================================================
        # Instructions
        # ======================================================
        self.instructions_label.setText(
            f"Calibration from {N_HEIGHTS} flat board placements at different table heights "
            "(~0 / +100 / +200 mm). Camera and C-arm must not move during the series.\n"
            "1) Per height: step 3 (plane fitting) and step 4 (X-ray markers); "
            "this page then solves the height by itself\n"
            "2) 'Next height': raise the table ~100 mm, board flat on it, "
            "not touched between plane fitting and X-ray\n"
            f"3) After height {N_HEIGHTS} the pooled pose is set -> 'Next'\n"
            "'Redo height' (next to the title) repeats the current height; "
            "'New series' discards all heights\n"
            "Image: measured (red) vs projected (cyan) points of this height alone\n"
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
        # Solve the current height once per set of inputs: coming back
        # through Back/Next with unchanged steps 3/4 keeps the result.
        if self._inputs_key() != self._pending_key:
            if not self._missing():
                self._solve_height()
            elif self._pending is not None:
                # inputs changed but incomplete: the old solve no longer applies
                self._clear_height_results()
                self.state.T_cx = None
                self.state.T_xc = None
                self.on_complete_changed()
        self.refresh()

    # ---------------- Series ----------------

    def current_height(self) -> int:
        """1-based height the wizard is on (steps 3-5)."""
        return min(len(self.state.cam2x_heights) + 1, N_HEIGHTS)

    def solved_heights(self) -> int:
        """Heights solved so far in this series (finished + current)."""
        return len(self.state.cam2x_heights) + (1 if self._pending is not None else 0)

    def has_height_results(self) -> bool:
        """Anything a 'New series' would discard."""
        return self.solved_heights() > 0

    def finished_height_items(self) -> list[str]:
        """One line per finished height for the 'Redo height' choice, with
        its pooled reprojection error once the pool exists."""
        out = []
        for i, _ in enumerate(self.state.cam2x_heights):
            px = ""
            if self._pooled is not None:
                px = f"  (pooled {self._pooled.reproj_mean_px_per_capture[i]:.2f} px)"
            out.append(f"h{i + 1}{px}")
        return out

    def drop_height(self, i: int) -> Cam2XHeight:
        """Take finished height i out of the series to capture it again; a
        solved current height is kept as finished. The pool does not
        depend on the order of the heights."""
        hs = self.state.cam2x_heights
        if self._pending is not None:
            hs.append(self._pending)
        h = hs.pop(i)
        self._clear_height_results()
        self.state.T_cx = None
        self.state.T_xc = None
        return h

    def commit_height(self) -> bool:
        """Keep the solved current height for the pooled solve. False when
        there is nothing to keep or the series is already complete."""
        if self._pending is None or len(self.state.cam2x_heights) + 1 >= N_HEIGHTS:
            return False
        self.state.cam2x_heights.append(self._pending)
        self._clear_height_results()
        return True

    @staticmethod
    def _md5_of(head: str, arrays) -> str:
        h = hashlib.md5(head.encode())
        for a in arrays:
            if a is None:
                h.update(b"-")
                continue
            a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def _field_arrays(self) -> tuple:
        """The X-ray distortion field by content (a reloaded identical file
        is the same field)."""
        f = getattr(self.state, "xray_dist_field", None)
        if f is None:
            return (None,)
        return (f.coeff, f.centre, np.array([f.degree, f.scale, f.width], dtype=np.float64))

    def _intrinsics_key(self) -> str:
        """What a finished height's capture depends on besides its own
        images: the camera model (board pose T_bc) and the X-ray field
        (corrected lattice). K_xray is left out on purpose: the pooled
        solve reprojects every capture with the current K_xray."""
        s = self.state
        return self._md5_of("intrinsics", (s.K_rgb, s.dist_rgb) + self._field_arrays())

    def _inputs_key(self) -> str:
        """Fingerprint of everything the height solve reads, and of the height."""
        s = self.state
        return self._md5_of(
            f"height{len(s.cam2x_heights)}",
            (
                s.K_xray, s.K_rgb, s.dist_rgb,
                s.xray_points_uv, s.xray_points_xyz_c,
                s.checkerboard_corners_uv, s.checkerboard_corners_uv_9,
                s.xray_lattice_uv, s.xray_lattice_xy_mm,
            ) + self._field_arrays(),
        )

    def _same_K(self, h: Cam2XHeight) -> bool:
        """T_cx_single of h was solved with the K_xray of the pool (else its
        rotation against the pool is not comparable)."""
        return (
            h.K_xray is not None and self._pooled_K is not None
            and np.array_equal(np.asarray(h.K_xray), self._pooled_K)
        )

    @staticmethod
    def _p95_inliers(calib) -> float:
        errors = np.asarray(calib.reproj_errors_px, dtype=np.float64).reshape(-1)
        idx = np.asarray(calib.inlier_idx, dtype=np.int64).reshape(-1)
        idx = idx[(idx >= 0) & (idx < errors.size)]
        return float(np.percentile(errors[idx], 95)) if idx.size > 0 else float("nan")

    # ---------------- Stats helpers ----------------

    def _stats_rows(self) -> list[tuple[str, str]]:
        k = self.current_height()
        med = self._reproj_median_px
        p95 = self._reproj_p95_in_px
        mean = self._reproj_mean_px
        rows: list[tuple[str, str]] = [
            ("Height", f"{k} / {N_HEIGHTS}"),
            # this height alone, as before the series
            ("Median", "-" if med is None else f"{float(med):.3f} px"),
            ("P95", "-" if p95 is None or not np.isfinite(p95) else f"{float(p95):.3f} px"),
            ("Avg. reprojection error", "-" if mean is None else f"{float(mean):.3f} px"),
        ]

        heights = list(self.state.cam2x_heights)
        if self._pending is not None:
            heights.append(self._pending)
        pooled = self._pooled

        if heights:
            # step = board centre moved since the previous height (the table
            # raise, same measure as the step-3 check); px / Δ = pooled
            # reprojection of the height and its single pose against the pool
            lines = [f"{len(heights)} of {N_HEIGHTS}", "h holes step/mm px Δ/deg"]
            for i, h in enumerate(heights):
                n = int(np.asarray(h.capture.lattice_xyz_board_mm).shape[0])
                step = "-"
                if i > 0:
                    d = np.asarray(h.board_centre_c_mm) - np.asarray(heights[i - 1].board_centre_c_mm)
                    step = f"{float(np.linalg.norm(d)):.0f}"
                px = dR = "-"
                if pooled is not None:
                    px = f"{pooled.reproj_mean_px_per_capture[i]:.2f}"
                    if self._same_K(h):
                        dR = f"{_rotation_deg(h.T_cx_single, pooled.T_cx):.2f}"
                lines.append(f"h{i + 1} {n} {step} {px} {dR}")
            rows.append(("Heights", "\n".join(lines)))
        else:
            rows.append(("Heights", "-"))

        if pooled is None:
            rows += [("Pooled", "-"), ("R_cx", "-"), ("t_cx", "-")]
            return rows

        rows.append(("Pooled", f"{pooled.reproj_mean_px:.2f} px mean"))
        if self._loo:
            worst = max(r["rotation_vs_full_deg"] for r in self._loo)
            rows.append(("Leave-one-out", f"≤ {worst:.2f} deg"))
        R = pooled.T_cx[:3, :3]
        R_lines = "\n".join(" ".join(f"{v:+.4f}" for v in row) for row in R)
        rows.append(("R_cx", "np.array (3×3)\n" + R_lines))
        t = pooled.T_cx[:3, 3]
        rows.append(("t_cx", f"[{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]^T"))
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

        k = self.current_height()
        self.btn_next_height.setText(
            f"Next height ({k + 1}/{N_HEIGHTS})" if k < N_HEIGHTS else "Next height"
        )
        self.btn_next_height.setEnabled(self._pending is not None and k < N_HEIGHTS)

        if img is None:
            self.set_viewport_background(active=False)
            self.zoom_view.set_image(None)
            self.zoom_view.clear_overlay_data()
            self.zoom_view.reset_zoom()
            self.zoom_view.setEnabled(False)
            self.set_stats_rows(self._stats_rows())
            self._update_zoom_button()
            return

        self.set_viewport_background(active=True)
        self.zoom_view.set_image(img)

        marker_r = self.state.marker_radius_px
        pick_r = 0.6 * float(marker_r) if marker_r is not None else 20.0

        uv_roi = self.state.xray_points_uv
        uv_meas = self.state.xray_points_uv

        if self._uv_projected is not None:
            self.zoom_view.set_overlay_data(
                uv_roi=uv_roi,
                uv_measured=uv_meas,
                uv_projected=self._uv_projected,
                pick_radius_px=pick_r,
                marker_radius_px=1.2 * marker_r if marker_r is not None else None,
                show_residuals=True,
                outlier_mask=None,
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

        self.set_stats_rows(self._stats_rows())
        self._update_zoom_button()

    # ---------------- Zoom helpers ----------------

    def on_reset_zoom(self) -> None:
        self.zoom_view.reset_zoom()
        self._update_zoom_button()

    def _update_zoom_button(self) -> None:
        zoomed = self.zoom_view.has_zoom()
        self.btn_reset_zoom.setEnabled(bool(self._uv_projected is not None and zoomed))

    # ---------------- Reset ----------------

    def _clear_height_results(self) -> None:
        self._pending = None
        self._pending_key = None
        self._pooled = None
        self._pooled_K = None
        self._loo = None
        self._uv_projected = None
        self._reproj_mean_px = None
        self._reproj_median_px = None
        self._reproj_p95_in_px = None
        self._last_calib = None

    def reset_results(self) -> None:
        """Forget the current height's solve (next height / new series). The
        finished heights in state.cam2x_heights are main.py's business."""
        self._clear_height_results()
        self.zoom_view.reset_zoom()
        self.refresh()

    # ---------------- Evaluation mode ----------------

    def _save_evaluation(
        self,
        *,
        calib,
        height: int,
        Kx: np.ndarray,
        uv_meas: np.ndarray,
        xyz_c: np.ndarray,
        lattice_uv: np.ndarray | None,
        lattice_xy: np.ndarray | None,
    ) -> str:
        """<label>_h<k>_cam2x.npz: the INPUTS of calibrate_camera_to_xray for
        this height (so the solve can be repeated offline: patch only, without
        LM, held-out halves, ...) and its OUTPUTS (this height alone: T_cx,
        T_xc, residuals); <label>_h<k>_cam2x.json: the solver diagnostics the
        console prints - RGB candidates against the depth reference, trust
        region, X-ray lattice refinement. Returns the npz file name."""
        ev = self.state.eval
        pr = getattr(calib, "pose_result", None)
        field = getattr(self.state, "xray_dist_field", None)

        def opt(a, dtype=np.float64):
            return None if a is None else np.asarray(a, dtype=dtype)

        uv_raw = opt(self.state.xray_points_uv)
        lattice_uv_raw = opt(self.state.xray_lattice_uv)

        errors = np.asarray(calib.reproj_errors_px, dtype=np.float64).reshape(-1)
        inlier_idx = np.asarray(calib.inlier_idx, dtype=np.int64).reshape(-1)
        p95_in = self._p95_inliers(calib)

        arrays = dict(
            pose_method=np.array("ippe_handeye"),
            height_index=np.array(int(height), dtype=np.int32),
            n_heights=np.array(int(N_HEIGHTS), dtype=np.int32),
            # ---- inputs (camera side) ----
            points_xyz_camera_m=opt(xyz_c),
            checkerboard_corners_uv=opt(self.state.checkerboard_corners_uv),
            checkerboard_corners_uv_9=opt(self.state.checkerboard_corners_uv_9),
            K_rgb=opt(self.state.K_rgb),
            dist_rgb=opt(self.state.dist_rgb),
            # ---- inputs (X-ray side; working space, u = W-1-u_raw) ----
            K_xray=opt(Kx),
            points_uv_xray=opt(uv_meas),           # field-corrected, as solved
            points_uv_xray_uncorrected=uv_raw,     # before the distortion field
            lattice_uv_xray=opt(lattice_uv),
            lattice_uv_xray_uncorrected=lattice_uv_raw,
            lattice_xy_mm=opt(lattice_xy),
            distortion_field_applied=np.array(field is not None, dtype=bool),
            pitch_mm=np.array(2.54, dtype=np.float64),
            board_thickness_mm=np.array(float(PCB_THICKNESS_MM), dtype=np.float64),
            steps_per_edge=np.array(10, dtype=np.int32),
            # ---- outputs (this height alone) ----
            T_cx=opt(calib.T_cx),
            T_xc=opt(calib.T_xc),
            R_cx=opt(calib.rotation),
            t_cx=np.asarray(calib.translation, dtype=np.float64).reshape(3),
            reproj_errors_px=errors,
            reproj_mean_px=np.array(float(calib.reproj_mean_px)),
            reproj_median_px=np.array(float(calib.reproj_median_px)),
            reproj_p95_in_px=np.array(p95_in),
            inlier_idx=inlier_idx,
            uv_projected=opt(calib.uv_proj),
        )

        diag: dict = {}
        if pr is not None:
            diag = {
                "method": pr.method,
                "candidate_index_rgb": pr.candidate_index_rgb,
                "candidate_index_xray": pr.candidate_index_xray,
                "reproj_mean_px": pr.reproj_mean_px,
                "reproj_median_px": pr.reproj_median_px,
                "reproj_max_px": pr.reproj_max_px,
                "xray_lattice_stats": pr.xray_lattice_stats,
                "rgb_candidates": [
                    {
                        "index": c.candidate_index,
                        "rvec": c.rvec, "tvec": c.tvec,
                        "reproj_mean_px": c.reproj_mean_px,
                        "reproj_max_px": c.reproj_max_px,
                    }
                    for c in (pr.all_candidates_rgb or [])
                ],
                "xray_patch_candidates": [
                    {
                        "index": c.candidate_index,
                        "rvec": c.rvec, "tvec": c.tvec,
                        "reproj_mean_px": c.reproj_mean_px,
                        "reproj_max_px": c.reproj_max_px,
                    }
                    for c in (pr.all_candidates or [])
                ],
                "solver": pr.diagnostics,
            }
            # the solver arrays also go into the npz (T_bc_depth, T_bc, T_bx)
            sd = pr.diagnostics or {}
            for k in ("T_bc_depth", "T_bc", "T_bx"):
                if k in sd:
                    arrays[k] = np.asarray(sd[k], dtype=np.float64)

        summary = {
            "method": "ippe_handeye",
            "height": f"{height}/{N_HEIGHTS}",
            "n": int(uv_meas.shape[0]),
            "reproj_mean_px": float(calib.reproj_mean_px),
        }
        stats = diag.get("xray_lattice_stats") or {}
        if stats:
            summary["lattice_dR_deg"] = stats.get("refined_candidates_dR_deg")
            summary["lattice_mean_px"] = stats.get("lattice_reproj_mean_px")

        try:
            path = ev.save_npz("cam2x", arrays, page="cam2x", summary=summary)
            if diag:
                ev.save_json("cam2x", diag, page="cam2x")
            return path.name
        except Exception as e:
            print(f"[WARN] Could not save Cam2X calibration: {e}")
            return ""

    def _save_pooled_evaluation(self, heights: list[Cam2XHeight], Kx: np.ndarray) -> None:
        """<label>_h3_cam2x_pooled.json (tag of the last height, so a height
        captured again gives a pool of its own name, h3b): the pose the
        wizard sets, the files of the heights it was pooled from (their json
        holds T_bc, pts3d_lattice_mm and lattice_uv, i.e. everything to
        repeat the solve offline), residuals and the leave-one-out check."""
        p = self._pooled
        data = {
            "method": "pooled PnP over the board placements (SQPnP + LM), "
                      "overlay.calib.calib_camera_to_xray.calibrate_camera_to_xray_pooled",
            "units": "T_cx / T_xc translation in metres; T_cx maps camera -> X-ray",
            "n_heights": len(heights),
            "height_labels": [h.eval_label for h in heights],
            "height_files": [h.eval_file for h in heights],
            "K_xray": Kx,
            "T_cx": p.T_cx,
            "T_xc": p.T_xc,
            "rvec": p.rvec.reshape(3),
            "tvec_m": p.tvec.reshape(3),
            "n_points_per_height": list(p.n_points_per_capture),
            "reproj_mean_px": p.reproj_mean_px,
            "reproj_median_px": p.reproj_median_px,
            "reproj_max_px": p.reproj_max_px,
            "reproj_mean_px_per_height": list(p.reproj_mean_px_per_capture),
            "leave_one_out": self._loo,
            "single_heights": [
                {
                    "file": h.eval_file,
                    "T_cx": h.T_cx_single,
                    "K_xray": h.K_xray,
                    "rotation_vs_pooled_deg": (
                        _rotation_deg(h.T_cx_single, p.T_cx) if self._same_K(h) else None
                    ),
                    "board_centre_camera_mm": h.board_centre_c_mm,
                }
                for h in heights
            ],
        }
        summary = {"heights": len(heights), "reproj_mean_px": p.reproj_mean_px}
        if self._loo:
            summary["loo_max_deg"] = max(r["rotation_vs_full_deg"] for r in self._loo)
        try:
            self.state.eval.save_json("cam2x_pooled", data, page="cam2x", log=True, summary=summary)
        except Exception as e:
            print(f"[WARN] Could not save pooled Cam2X calibration: {e}")

    # ---------------- Action ----------------

    def _solve_height(self) -> None:
        """Solve the current height alone; at the last height pool all of
        them and set T_cx / T_xc."""
        height = self.current_height()
        key = self._inputs_key()

        # whatever was there belongs to other inputs now
        self._clear_height_results()
        self.state.T_cx = None
        self.state.T_xc = None

        # The finished heights keep the board poses / corrected lattices of
        # the intrinsics they were solved with; pooling them with a height
        # solved with other intrinsics would mix two calibrations.
        ikey = self._intrinsics_key()
        stale = [i + 1 for i, h in enumerate(self.state.cam2x_heights) if h.intrinsics_key != ikey]
        if stale:
            QMessageBox.warning(
                self,
                "Intrinsics changed",
                f"Height(s) {', '.join(map(str, stale))} of this series were solved with other "
                "camera intrinsics or another X-ray distortion field than the ones loaded now.\n\n"
                "Load the previous files again (steps 1/2), or start a 'New series'.",
            )
            self.refresh()
            self.on_complete_changed()
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
            # space and same correction as the patch points. The pooled solve
            # needs it (Cam2XBoardCapture.from_result).
            lattice_uv = None
            lattice_xy = None
            if self.state.xray_lattice_uv is not None and self.state.xray_lattice_xy_mm is not None:
                lattice_uv = np.asarray(self.state.xray_lattice_uv, dtype=np.float64).reshape(-1, 2)
                if field is not None:
                    lattice_uv = field.undistort_working(lattice_uv)
                lattice_xy = np.asarray(self.state.xray_lattice_xy_mm, dtype=np.float64).reshape(-1, 2)
            xyz_c = np.asarray(self.state.xray_points_xyz_c, dtype=np.float64).reshape(-1, 3)

            if xyz_c.shape[0] != uv_meas.shape[0]:
                QMessageBox.warning(
                    self,
                    "Point count mismatch",
                    f"2D points: {uv_meas.shape[0]}\n"
                    f"3D points: {xyz_c.shape[0]}\n\n"
                    "This step expects the same number of 2D and 3D correspondences.",
                )
                return

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
                checkerboard_corners_uv_9=(
                    None
                    if self.state.checkerboard_corners_uv_9 is None
                    else np.asarray(self.state.checkerboard_corners_uv_9, dtype=np.float64)
                ),
                K_rgb=np.asarray(self.state.K_rgb, dtype=np.float64),
                steps_per_edge=10,
                board_thickness_mm=PCB_THICKNESS_MM,
                xray_lattice_uv=lattice_uv,
                xray_lattice_xy_mm=lattice_xy,
            )
            capture = Cam2XBoardCapture.from_result(calib, label=f"h{height}")

            eval_file = ""
            if self.state.eval is not None:
                eval_file = self._save_evaluation(
                    calib=calib,
                    height=height,
                    Kx=Kx,
                    uv_meas=uv_meas,
                    xyz_c=xyz_c,
                    lattice_uv=lattice_uv,
                    lattice_xy=lattice_xy,
                )

            self._pending = Cam2XHeight(
                capture=capture,
                T_cx_single=np.asarray(calib.T_cx, dtype=np.float64),
                board_centre_c_mm=xyz_c.mean(axis=0) * 1000.0,
                xray_md5=xray_pixel_md5(self.state.xray_image),
                intrinsics_key=ikey,
                K_xray=Kx.copy(),
                eval_label=(self.state.eval.file_label() if self.state.eval is not None else ""),
                eval_file=eval_file,
            )
            self._pending_key = key
            self._uv_projected = np.asarray(calib.uv_proj, dtype=np.float64).reshape(-1, 2)
            self._reproj_mean_px = float(calib.reproj_mean_px)
            self._reproj_median_px = float(calib.reproj_median_px)
            self._reproj_p95_in_px = self._p95_inliers(calib)
            self._last_calib = calib

            if height >= N_HEIGHTS:
                heights = list(self.state.cam2x_heights) + [self._pending]
                captures = [h.capture for h in heights]
                self._pooled = calibrate_camera_to_xray_pooled(Kx, captures)
                self._pooled_K = Kx.copy()
                self._loo = leave_one_capture_out(Kx, captures) if len(captures) >= 3 else None

                # --------- persist ONLY allowed SessionState fields ----------
                self.state.T_cx = self._pooled.T_cx
                self.state.T_xc = self._pooled.T_xc

                if self.state.eval is not None:
                    self._save_pooled_evaluation(heights, Kx)

        except Exception as e:
            self._clear_height_results()
            # same inputs give the same failure: no second solve (and no
            # second set of evaluation files) on the next Back/Next
            self._pending_key = key
            self.state.T_cx = None
            self.state.T_xc = None
            QMessageBox.critical(self, "Calibration failed", str(e))

        finally:
            self.zoom_view.reset_zoom()
            self.refresh()
            self.on_complete_changed()

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

        lat = self.state.xray_lattice_uv
        if lat is None or len(lat) == 0 or self.state.xray_lattice_xy_mm is None:
            m.append("xray_lattice_uv")

        xyz = self.state.xray_points_xyz_c
        if xyz is None or len(xyz) == 0:
            m.append("xray_points_xyz_c")

        if not self.state.has_checkerboard_corners:
            m.append("checkerboard_corners_uv")
        if self.state.K_rgb is None:
            m.append("K_rgb")

        return m
