from __future__ import annotations

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QPushButton, QFileDialog, QMessageBox, QSizePolicy
)

from overlay.gui.state import SessionState
from overlay.gui.pages.templates.templ_static_image import StaticImagePage
from overlay.gui.widgets.widget_xray_marker_selection import XrayMarkerSelectionWidget
from overlay.gui.widgets.widget_xray_waiting import XrayWaitingOverlay

from overlay.tools.blob_detection import HoughCircleParams
from overlay.tools.xray_io import read_xray_gray_u8, xray_pixel_md5
from overlay.tools.xray_marker_selection import (
    run_xray_marker_detection,
    compute_roi_from_grid,
    index_board_lattice,
)



class XrayMarkerSelectionPage(StaticImagePage):
    """
    Step — X-ray marker selection

    IMPORTANT
    ---------
    - MUST NOT add any new SessionState fields dynamically.
    - SessionState fields used here (already defined in state.py):
        * xray_image, xray_image_path
        * xray_points_uv, xray_points_confirmed
        * xray_marker_overlay_bgr
        * marker_radius_px
    - Everything else stays LOCAL to this page:
        * _circles
        * _pick_radius_px

    Coordinate convention in this page
    ----------------------------------
    - state.xray_image stays in RAW coordinates.
    - The marker-selection widget shows the RAW image.
    - Detection runs on the RAW image.
    - self._circles are kept LOCAL in RAW coordinates for interaction.
    - compute_roi_from_grid(...) returns measured UVs directly in XRAY WORKING SPACE:

          u_work = W - 1 - u_raw
          v_work = v_raw

    - state.xray_points_uv receives this returned uv_final directly.
    - The X-ray image itself is NOT flipped here. Image flipping happens later
      when the overlay is rendered/warped.

    Anchor / ordering convention
    ----------------------------
    - The user selects THREE anchors in this order:

        1) TL
        2) TR
        3) BL

      where these roles are defined in CAMERA VIEW semantics, i.e. from the
      top-view / camera-view perspective of the board.

    - compute_roi_from_grid(...) uses these anchors to define an affine ROI model
      for selection and ordering only.

    - No affine regularization or point repositioning is performed.
      The selected points remain measured blob detections.

    Therefore:
    - selection = [TL, TR, BL] in CAMERA VIEW on the RAW image
    - ROI extraction/order = canonical RAW board order
    - stored state.xray_points_uv = transformed XRAY_WORKING_FLIPPED_UV
    """

    def __init__(self, state: SessionState, on_complete_changed, parent=None):
        super().__init__(parent)

        self.state = state
        self.on_complete_changed = on_complete_changed

        # ---------------- local (page-only) runtime ----------------
        self._circles: np.ndarray | None = None       # (N,3), RAW coords
        self._pick_radius_px: float | None = None     # interaction radius
        self._last_roi_uv_raw: np.ndarray | None = None  # local debug/display only
        self._last_lattice: tuple | None = None          # (uv_working, ij, xy_mm, dbg)

        # ======================================================
        # LEFT: replace template image_label with interactive widget
        # ======================================================
        self.marker_widget = XrayMarkerSelectionWidget(self.image_label)
        self.marker_widget.selection_proposed.connect(self.on_selection_proposed)
        self.marker_widget.selection_changed.connect(self._on_selection_changed)

        self.marker_widget.setGeometry(0, 0, self.image_label.width(), self.image_label.height())
        self.marker_widget.show()

        # Shown while no X-ray is loaded: the image arrives from the C-arm
        # by itself (accept_received_xray); the file button is the fallback.
        self.wait_overlay = XrayWaitingOverlay(self.image_label)

        # ======================================================
        # RIGHT: controls
        # ======================================================
        self.btn_load = QPushButton("Load from file (USB)")
        self.btn_load.setObjectName("SecondaryBtn")
        self.btn_load.clicked.connect(self.load_xray_image)

        self.btn_detect = QPushButton("Detect markers")
        self.btn_detect.clicked.connect(self.on_detect_markers)

        self.btn_reset = QPushButton("Reset selection")
        self.btn_reset.clicked.connect(self.on_reset_selection)

        for b in (self.btn_load, self.btn_detect, self.btn_reset):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.controls_content.addWidget(b)

        # ======================================================
        # Instructions
        # ======================================================
        self.instructions_label.setText(
            "1) Take the X-ray and send it to CONQUEST on the C-arm - it appears here\n"
            "   (fallback: 'Load from file (USB)')\n"
            "2) Click 'Detect markers'\n"
            "3) Select 3 anchor markers corresponding to:\n"
            "   TL, TR, BL in CAMERA VIEW\n"
            "\n"
            "IMPORTANT:\n"
            "- The displayed image remains RAW in this page\n"
            "- Select anchors by real board meaning from the top-view / camera-view\n"
            "- compute_roi_from_grid(...) returns canonical RAW UVs\n"
            "- The stored state.xray_points_uv is transformed to XRAY WORKING SPACE:\n"
            "  u_work = W - 1 - u_raw, v_work = v_raw\n"
            "- The X-ray image itself is flipped later during overlay rendering\n"
            "- No semantic j_xray reordering is used anymore"
        )

        self.refresh()

    # ------------------------------------------------------------------
    # Keep marker widget in sync with template viewport size
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.marker_widget.setGeometry(0, 0, self.image_label.width(), self.image_label.height())

    def _on_selection_changed(self):
        self.refresh()

    # ---------------- Completion ----------------

    def is_complete(self) -> bool:
        return self.state.has_xray_points_confirmed

    # ---------------- Refresh ----------------

    def refresh(self):
        img_raw = self.state.xray_image
        self.wait_overlay.setVisible(img_raw is None)

        # --------------------------------------------------
        # Case 1: no image loaded yet
        # --------------------------------------------------
        if img_raw is None:
            self.marker_widget.set_image(None)
            self.marker_widget.set_circles(None)
            self.marker_widget.set_locked(False)

            self.set_stats_rows([
                ("Blobs detected", "-"),
                ("Blobs selected", "-"),
            ])

            self.btn_load.setEnabled(True)
            self.btn_detect.setEnabled(False)
            self.btn_reset.setEnabled(False)
            return

        # --------------------------------------------------
        # Case 2: image loaded — show RAW image
        # --------------------------------------------------
        self.marker_widget.set_image(img_raw)

        if self._circles is not None and self._pick_radius_px is not None:
            self.marker_widget.set_circles(
                np.asarray(self._circles, dtype=np.float64).copy(),
                float(self._pick_radius_px),
            )
        else:
            self.marker_widget.set_circles(None)

        confirmed = bool(self.state.xray_points_confirmed)

        self.btn_load.setEnabled(False)

        already_detected = self._circles is not None
        self.btn_detect.setEnabled(not already_detected)

        selected_idx = self.marker_widget.get_selected_indices()
        self.btn_reset.setEnabled(len(selected_idx) > 0)

        self.marker_widget.set_locked(confirmed)

        if self._circles is None:
            detected_txt = "-"
            selected_txt = "- / 3"
        else:
            n_detected = int(len(self._circles))
            n_selected = len(selected_idx)
            detected_txt = str(n_detected)
            selected_txt = f"{n_selected} / 3"

        self.set_stats_rows([
            ("Blobs detected", detected_txt),
            ("Blobs selected", selected_txt),
        ])

    # ---------------- Image loading ----------------

    def load_xray_image(self):
        dlg = QFileDialog(self.window(), "Select X-ray image")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.ima);;All Files (*)")
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)

        if dlg.exec() != QFileDialog.Accepted:
            return

        files = dlg.selectedFiles()
        path = files[0] if files else ""
        if not path:
            return

        try:
            img = read_xray_gray_u8(path)
        except Exception as e:
            QMessageBox.critical(self, "Failed to load X-ray image", str(e))
            return

        i = self._height_of(img)
        if i is not None:
            QMessageBox.warning(
                self,
                "Same X-ray",
                f"This is the X-ray of height {i + 1} (identical pixel data).\n\n"
                "Every height needs its own X-ray, taken after raising the "
                "table. Please load the new image.",
            )
            return
        self._set_xray(img, path)

    def accept_received_xray(self, path: str) -> bool:
        """An X-ray from the C-arm (MainWindow, while this page is shown).
        Only the image appears; the checks stay silent: until the markers
        are confirmed a new exposure replaces the current one, the same
        exposure again (sent twice, or the X-ray of an earlier height) is
        dropped. False = not taken (markers already confirmed)."""
        if self.state.xray_points_confirmed:
            return False
        try:
            img = read_xray_gray_u8(path)
        except Exception as e:
            print(f"[XrayMarkerSelection] received X-ray not readable: {path}: {e}")
            return True
        md5 = xray_pixel_md5(img)
        cur = self.state.xray_image
        if self._height_of(img) is not None or (cur is not None and xray_pixel_md5(cur) == md5):
            print(f"[XrayMarkerSelection] received X-ray already used, dropped: {path}")
            return True
        self._set_xray(img, path)
        return True

    def _height_of(self, img: np.ndarray) -> int | None:
        """Cam2X series: every height needs its own exposure. The same image
        again (e.g. exported twice) would pair the old hole positions with
        the new camera pose. Index of the height that already has it."""
        md5 = xray_pixel_md5(img)
        for i, h in enumerate(self.state.cam2x_heights):
            if h.xray_md5 == md5:
                return i
        return None

    def _set_xray(self, img: np.ndarray, path: str) -> None:
        self.state.xray_image = img
        self.state.xray_image_path = path

        self.state.xray_points_uv = None
        self.state.xray_points_confirmed = False
        self.state.xray_marker_overlay_bgr = None
        self.state.marker_radius_px = None

        self._circles = None
        self._pick_radius_px = None
        self._last_roi_uv_raw = None

        self.marker_widget.clear_selection()
        self.marker_widget.set_roi_indices([])
        self.marker_widget.set_locked(False)

        self.refresh()
        self.on_complete_changed()

    # ---------------- Marker detection ----------------

    def on_detect_markers(self):
        if self.state.xray_image is None:
            return

        try:
            img_raw = self.state.xray_image

            params = HoughCircleParams(
                min_radius=2,
                max_radius=7,
                dp=1.2,
                minDist=8,
                param1=120,
                param2=9,
                invert=True,
                median_ks=(3, 5),
            )

            res = run_xray_marker_detection(
                img_raw,
                hough_params=params,
                use_clahe=True,
                clahe_clip=2.0,
                clahe_tiles=(12, 12),
                use_mask=False,
            )

            if res.circles is None or len(res.circles) == 0:
                QMessageBox.warning(self, "Marker detection", "No circles detected.")
                return

            circles_raw = np.asarray(res.circles, dtype=np.float64).reshape(-1, 3)

            radii = circles_raw[:, 2]
            finite_r = radii[np.isfinite(radii)]
            if finite_r.size:
                self.state.marker_radius_px = float(np.median(finite_r))
            else:
                self.state.marker_radius_px = None

            if self.state.marker_radius_px is not None:
                pick_radius_px = 0.6 * float(self.state.marker_radius_px)
            else:
                pick_radius_px = 20.0

            self._circles = circles_raw
            self._pick_radius_px = float(pick_radius_px)

            self.state.xray_points_uv = None
            self.state.xray_points_confirmed = False
            self.state.xray_marker_overlay_bgr = None
            self._last_roi_uv_raw = None

            self.marker_widget.clear_selection()
            self.marker_widget.set_roi_indices([])
            self.marker_widget.set_locked(False)

            self.refresh()
            self.on_complete_changed()

        except Exception as e:
            QMessageBox.critical(self, "Marker detection failed", str(e))

    # ---------------- Confirm flow ----------------

    def on_selection_proposed(self, selected_idx):
        if selected_idx is None or len(selected_idx) != 3:
            return

        selected_idx = [int(k) for k in selected_idx]

        if self._circles is None or self._pick_radius_px is None or self.state.xray_image is None:
            self.marker_widget.clear_selection()
            self.marker_widget.set_roi_indices([])
            self.marker_widget.set_locked(False)
            self.refresh()
            self.on_complete_changed()
            return

        try:
            margin_px = 1.1 * float(self._pick_radius_px)

            image_width = int(self.state.xray_image.shape[1])

            roi_uv_final, roi_idx, dbg = compute_roi_from_grid(
                circles=self._circles,
                anchor_idx=selected_idx,
                margin_px=margin_px,
                image_width=image_width,
                gate_tol_pitch=0.40,
                min_steps=2,
            )

            # ------------------------------------------------------------
            # Orientation sanity check (CAMERA VIEW semantics)
            # ------------------------------------------------------------
            if not bool(dbg.get("orientation_ok", True)):
                warning_text = str(
                    dbg.get(
                        "orientation_warning",
                        "Unexpected anchor orientation detected."
                    )
                )

                ux = float(dbg.get("orientation_ux", 0.0))
                uy = float(dbg.get("orientation_uy", 0.0))
                vx = float(dbg.get("orientation_vx", 0.0))
                vy = float(dbg.get("orientation_vy", 0.0))
                cross_z = float(dbg.get("orientation_cross_z", 0.0))

                cond_ux = bool(dbg.get("orientation_cond_ux", False))
                cond_vy = bool(dbg.get("orientation_cond_vy", False))
                cond_cross = bool(dbg.get("orientation_cond_cross", False))

                ans = QMessageBox.warning(
                    self,
                    "Anchor orientation warning",
                    (
                        f"{warning_text}\n\n"
                        "Observed values:\n"
                        f"- (TR - TL) = ({ux:.3f}, {uy:.3f})\n"
                        f"- (BL - TL) = ({vx:.3f}, {vy:.3f})\n"
                        f"- cross_z = {cross_z:.3f}\n\n"
                        "Expected for CAMERA VIEW selection [TL, TR, BL]:\n"
                        f"- x-component of (TR - TL) < 0   -> {'OK' if cond_ux else 'NOT OK'}\n"
                        f"- y-component of (BL - TL) < 0   -> {'OK' if cond_vy else 'NOT OK'}\n"
                        f"- cross_z((TR - TL), (BL - TL)) > 0   -> {'OK' if cond_cross else 'NOT OK'}\n\n"
                        "Do you want to continue anyway?"
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )

                if ans != QMessageBox.Yes:
                    self.state.xray_points_uv = None
                    self.state.xray_points_confirmed = False
                    self.state.xray_marker_overlay_bgr = None

                    self.marker_widget.clear_selection()
                    self.marker_widget.set_roi_indices([])
                    self.marker_widget.set_locked(False)

                    self.refresh()
                    self.on_complete_changed()
                    return

            roi_uv_final = np.asarray(roi_uv_final, dtype=np.float64).reshape(-1, 2)
            roi_idx = np.asarray(roi_idx, dtype=np.int64).reshape(-1)

            # compute_roi_from_grid(...) already returns uv_final in
            # XRAY_WORKING_FLIPPED_UV. Do not transform again here.
            self.state.xray_points_uv = roi_uv_final.astype(float)
            self.state.xray_points_confirmed = True

            # Extend the indexed patch to every detected hole of the board.
            # The X-ray-side pose is refined on this lattice (Cam2X page);
            # see index_board_lattice for why the patch alone is not enough.
            lat_uv, lat_ij, lat_xy, lat_dbg = index_board_lattice(
                np.asarray(self._circles, dtype=np.float64),
                np.asarray(dbg["points_uv_raw"], dtype=np.float64).reshape(-1, 2),
                image_width=int(dbg["image_width"]),
            )
            self.state.xray_lattice_uv = lat_uv.astype(float)
            self.state.xray_lattice_xy_mm = lat_xy.astype(float)
            self._last_lattice = (lat_uv, lat_ij, lat_xy, lat_dbg)
            print(
                f"[XrayMarkerSelection] board lattice: {lat_dbg['n_indexed']} of "
                f"{lat_dbg['n_detections']} detections indexed, j {lat_dbg['j_range']}, "
                f"i {lat_dbg['i_range']}, homography rms {lat_dbg['homography_resid_rms_px']:.2f} px"
            )

            if "debug_uv_raw" in dbg:
                self._last_roi_uv_raw = np.asarray(
                    dbg["debug_uv_raw"], dtype=np.float64
                ).reshape(-1, 2)
            else:
                self._last_roi_uv_raw = None

            self.marker_widget.set_roi_indices(list(roi_idx.tolist()))
            self.marker_widget.set_locked(True)

            self.state.xray_marker_overlay_bgr = self._render_roi_overlay_bgr()

            if self.state.eval is not None:
                self._save_evaluation(
                    selected_idx=selected_idx,
                    roi_idx=roi_idx,
                    dbg=dbg,
                )

            self.refresh()
            self.on_complete_changed()

        except Exception as e:
            QMessageBox.critical(self, "Marker selection failed", str(e))
            self.state.xray_points_uv = None
            self.state.xray_points_confirmed = False
            self.state.xray_marker_overlay_bgr = None
            self.state.xray_lattice_uv = None
            self.state.xray_lattice_xy_mm = None
            self._last_lattice = None
            self._last_roi_uv_raw = None

            self.marker_widget.clear_selection()
            self.marker_widget.set_roi_indices([])
            self.marker_widget.set_locked(False)

            self.refresh()
            self.on_complete_changed()

    # ---------------- Reset ----------------

    def on_reset_selection(self):
        self.state.xray_points_uv = None
        self.state.xray_points_confirmed = False
        self.state.xray_marker_overlay_bgr = None
        self.state.xray_lattice_uv = None
        self.state.xray_lattice_xy_mm = None
        self._last_lattice = None

        self.marker_widget.clear_selection()
        self.marker_widget.set_roi_indices([])
        self.marker_widget.set_locked(False)

        self.refresh()
        self.on_complete_changed()

    # ---------------- Evaluation mode ----------------

    def _save_evaluation(
        self,
        selected_idx: list[int] | np.ndarray,
        roi_idx: np.ndarray,
        dbg: dict,
    ) -> None:
        """<label>_xray.<ext>: copy of the X-ray file; <label>_markers.npz:
        the patch points (working + raw), the full board lattice and every
        Hough detection, i.e. all X-ray-side inputs of Cam2X."""
        if self.state.xray_points_uv is None:
            return

        ev = self.state.eval
        xray_path = self.state.xray_image_path
        try:
            if xray_path:
                ev.copy_file("xray", xray_path, page="marker_selection")
        except Exception as e:
            print(f"[WARN] Could not copy X-ray image: {e}")

        lat = self._last_lattice
        lat_dbg = lat[3] if lat is not None else {}

        arrays = dict(
            # Main points used by the rest of the pipeline:
            # XRAY_WORKING_FLIPPED_UV
            points_uv=np.asarray(self.state.xray_points_uv, dtype=np.float64),

            # Debug / traceability:
            points_uv_working=np.asarray(self.state.xray_points_uv, dtype=np.float64),
            points_uv_raw=np.asarray(
                dbg.get("points_uv_raw", dbg.get("debug_uv_raw", [])),
                dtype=np.float64,
            ).reshape(-1, 2),
            uv_space=np.array("XRAY_WORKING_FLIPPED_UV", dtype="<U64"),
            raw_uv_space=np.array("XRAY_RAW_CANONICAL_BOARD_ORDER", dtype="<U64"),
            uv_transform=np.array("horizontal_flip", dtype="<U32"),
            uv_transform_formula=np.array(
                "u_work = W - 1 - u_raw, v_work = v_raw",
                dtype="<U64",
            ),
            semantic_reordering=np.array(False, dtype=bool),
            no_j_xray_reordering=np.array(True, dtype=bool),
            image_width=np.array(int(dbg.get("image_width", -1)), dtype=np.int32),

            anchor_indices=np.asarray(selected_idx, dtype=np.int64),
            roi_indices=np.asarray(roi_idx, dtype=np.int64),
            # Full board lattice (2026-09-15): all indexed holes, working space,
            # integer lattice coordinates (j, i) and board mm relative to TL.
            lattice_uv_working=(
                np.asarray(self._last_lattice[0], dtype=np.float64)
                if self._last_lattice is not None else np.empty((0, 2))
            ),
            lattice_ij=(
                np.asarray(self._last_lattice[1], dtype=np.int64)
                if self._last_lattice is not None else np.empty((0, 2), dtype=np.int64)
            ),
            lattice_xy_mm=(
                np.asarray(self._last_lattice[2], dtype=np.float64)
                if self._last_lattice is not None else np.empty((0, 2))
            ),
            circles_uvr=(
                np.asarray(self._circles, dtype=np.float64)
                if self._circles is not None
                else np.empty((0, 3), dtype=np.float64)
            ),
            marker_radius_px=(
                np.array(float(self.state.marker_radius_px), dtype=np.float64)
                if self.state.marker_radius_px is not None
                else np.array(np.nan, dtype=np.float64)
            ),
            xray_image_path=np.array(xray_path if xray_path is not None else "", dtype=object),
            orientation_ok=np.array(bool(dbg.get("orientation_ok", True)), dtype=bool),
            orientation_ux=np.array(float(dbg.get("orientation_ux", np.nan)), dtype=np.float64),
            orientation_uy=np.array(float(dbg.get("orientation_uy", np.nan)), dtype=np.float64),
            orientation_vx=np.array(float(dbg.get("orientation_vx", np.nan)), dtype=np.float64),
            orientation_vy=np.array(float(dbg.get("orientation_vy", np.nan)), dtype=np.float64),
            orientation_cross_z=np.array(float(dbg.get("orientation_cross_z", np.nan)), dtype=np.float64),
            orientation_cond_ux=np.array(bool(dbg.get("orientation_cond_ux", False)), dtype=bool),
            orientation_cond_vy=np.array(bool(dbg.get("orientation_cond_vy", False)), dtype=bool),
            orientation_cond_cross=np.array(bool(dbg.get("orientation_cond_cross", False)), dtype=bool),
            margin_px=np.array(1.1 * float(self._pick_radius_px), dtype=np.float64),
            pick_radius_px=np.array(float(self._pick_radius_px), dtype=np.float64),
            xray_image_raw=np.asarray(self.state.xray_image),
            lattice_n_indexed=np.array(int(lat_dbg.get("n_indexed", -1)), dtype=np.int32),
            lattice_n_detections=np.array(int(lat_dbg.get("n_detections", -1)), dtype=np.int32),
            lattice_homography_rms_px=np.array(
                float(lat_dbg.get("homography_resid_rms_px", np.nan)), dtype=np.float64
            ),
        )

        try:
            ev.save_npz(
                "markers",
                arrays,
                page="marker_selection",
                summary={
                    "patch": int(len(self.state.xray_points_uv)),
                    "lattice": int(lat_dbg.get("n_indexed", -1)),
                    "lattice_rms_px": float(lat_dbg.get("homography_resid_rms_px", np.nan)),
                    "orientation_ok": bool(dbg.get("orientation_ok", True)),
                },
            )
        except Exception as e:
            print(f"[WARN] Could not save marker selection: {e}")

    # ---------------- Helpers ----------------

    def _render_roi_overlay_bgr(self) -> np.ndarray | None:
        """
        Render the RAW X-ray image with ROI overlay for display.

        Note
        ----
        state.xray_points_uv is stored in XRAY WORKING SPACE. The page itself
        does not transform these points. For the RAW preview overlay, this uses
        the locally stored RAW ROI points from the selection debug data.
        """
        img_raw = self.state.xray_image
        uv_raw = self._last_roi_uv_raw

        if img_raw is None or uv_raw is None or len(uv_raw) == 0:
            return None

        img8 = img_raw if img_raw.dtype == np.uint8 else np.clip(img_raw, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

        if self._pick_radius_px is None:
            roi_r = 6.0
        else:
            roi_r = max(3.0, 0.35 * float(self._pick_radius_px))

        cross_r = int(round(0.6 * roi_r))

        for (u, v) in np.asarray(uv_raw, dtype=np.float64).reshape(-1, 2):
            if not np.isfinite(u) or not np.isfinite(v):
                continue
            uu, vv = int(round(u)), int(round(v))

            cv2.circle(out, (uu, vv), int(round(roi_r)), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(out, (uu - cross_r, vv), (uu + cross_r, vv), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(out, (uu, vv - cross_r), (uu, vv + cross_r), (0, 0, 255), 2, cv2.LINE_AA)

        return out