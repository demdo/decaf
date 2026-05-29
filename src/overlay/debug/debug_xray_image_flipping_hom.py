# -*- coding: utf-8 -*-
"""
debug_hxc_point_mapping.py

Debug:
Validate whether H_xc maps X-ray marker points correctly onto the RGB board points.

Shows:
1) Original point mapping:
   - green: RGB board grid points
   - red: X-ray marker points projected into RGB with H_xc
   - cyan: residuals

2) Same mapping after subtracting the mean residual:
   This tests whether the error is mainly a constant offset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox

from overlay.tools.homography import estimate_plane_induced_homography
from tracking.pose_solvers import solve_pose


# ============================================================
# Qt helpers
# ============================================================

def _ensure_qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def pick_npz_file(title: str) -> Path | None:
    _ensure_qt_app()
    path, _ = QFileDialog.getOpenFileName(None, title, "", "NPZ files (*.npz);;All files (*.*)")
    return Path(path) if path else None


# ============================================================
# Basic helpers
# ============================================================

def _as_scalar(x, name: str) -> float:
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(f"{name} must be scalar-like, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def _as_xyz(arr: np.ndarray, name: str) -> np.ndarray:
    pts = np.asarray(arr, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N,3), got {pts.shape}")
    return pts


def _as_uv(arr: np.ndarray, name: str) -> np.ndarray:
    pts = np.asarray(arr, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N,2), got {pts.shape}")
    return pts


def _as_mat33(arr: np.ndarray, name: str) -> np.ndarray:
    M = np.asarray(arr, dtype=np.float64)
    if M.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3,3), got {M.shape}")
    return M


def _as_transform44(arr: np.ndarray, name: str) -> np.ndarray:
    T = np.asarray(arr, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4,4), got {T.shape}")
    return T


def _to_uint8_bgr(img: np.ndarray, name: str) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img8 = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    raise ValueError(f"{name} must be grayscale or BGR image, got shape {img.shape}")


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def make_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def compute_T_cx_from_T_bc_T_bx(T_bc_mm: np.ndarray, T_bx_mm: np.ndarray) -> np.ndarray:
    T_cx_mm = np.asarray(T_bx_mm, dtype=np.float64) @ invert_transform(T_bc_mm)
    T_cx_m = T_cx_mm.copy()
    T_cx_m[:3, 3] *= 1e-3
    return T_cx_m


def recompute_dx_from_T_xc(T_xc_m: np.ndarray, T_tc_mm: np.ndarray) -> float:
    T_cx_m = invert_transform(T_xc_m)
    T_cx_mm = T_cx_m.copy()
    T_cx_mm[:3, 3] *= 1e3
    T_tx = T_cx_mm @ T_tc_mm
    return float(T_tx[2, 3])


def format_matrix(M: np.ndarray, decimals: int = 6) -> str:
    return "\n".join(
        "[" + "  ".join(f"{v:+.{decimals}f}" for v in row) + "]" for row in np.asarray(M)
    )


def load_K_xray_from_intrinsics_npz(npz_path: Path) -> np.ndarray:
    data = np.load(str(npz_path), allow_pickle=True)
    for key in ["K_xray", "K", "camera_matrix"]:
        if key in data:
            return _as_mat33(data[key], key)
    raise KeyError(f"No K found in {npz_path.name}. Expected one of: K_xray, K, camera_matrix")


def apply_homography_to_uv(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    uv_h = np.hstack([uv, np.ones((uv.shape[0], 1), dtype=np.float64)])
    proj_h = (H @ uv_h.T).T
    w = proj_h[:, 2]
    out = np.full((uv.shape[0], 2), np.nan, dtype=np.float64)
    valid = np.isfinite(w) & (np.abs(w) > 1e-12)
    out[valid] = proj_h[valid, :2] / w[valid, None]
    return out


def make_flip_x_matrix(width_px: int) -> np.ndarray:
    """Homogeneous transform for horizontal image flip (x' = (w-1)-x)."""
    w1 = float(width_px - 1)
    return np.array([[-1.0, 0.0, w1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def make_flip_x_matrix_from_uv(uv: np.ndarray) -> np.ndarray:
    """Flip around UV span center: x' = (xmin + xmax) - x."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    xmin = float(np.nanmin(uv[:, 0]))
    xmax = float(np.nanmax(uv[:, 0]))
    return np.array([[-1.0, 0.0, xmin + xmax], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def reorder_grid_uv_horizontal(uv: np.ndarray, steps_per_edge: int = 10) -> np.ndarray:
    """Reindex board-grid UV from RAW to canonical by mirroring columns."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    n = steps_per_edge + 1
    if uv.shape[0] != n * n:
        return uv.copy()
    out = np.empty_like(uv)
    for r in range(n):
        for c in range(n):
            out[r * n + c] = uv[r * n + (n - 1 - c)]
    return out


def build_rgb_grid_from_anchors(checkerboard_corners_uv: np.ndarray, *, steps_per_edge: int = 10) -> np.ndarray:
    p_tl, p_tr, p_bl = np.asarray(checkerboard_corners_uv, dtype=np.float64).reshape(3, 2)
    step_x = (p_tr - p_tl) / float(steps_per_edge)
    step_y = (p_bl - p_tl) / float(steps_per_edge)
    return np.array(
        [p_tl + alpha * step_x + beta * step_y for beta in range(steps_per_edge + 1) for alpha in range(steps_per_edge + 1)],
        dtype=np.float64,
    )


def draw_point_debug(camera_bgr: np.ndarray, pts_rgb_ref: np.ndarray, pts_xray_projected: np.ndarray, *, title_text: str) -> np.ndarray:
    out = np.asarray(camera_bgr, dtype=np.uint8).copy()
    pts_rgb_ref = np.asarray(pts_rgb_ref, dtype=np.float64).reshape(-1, 2)
    pts_xray_projected = np.asarray(pts_xray_projected, dtype=np.float64).reshape(-1, 2)
    for p_ref, p_proj in zip(pts_rgb_ref, pts_xray_projected):
        if np.isfinite(p_ref).all() and np.isfinite(p_proj).all():
            cv2.line(out, tuple(np.round(p_ref).astype(int)), tuple(np.round(p_proj).astype(int)), (255, 255, 0), 1, cv2.LINE_AA)
    for p in pts_rgb_ref:
        if np.isfinite(p).all():
            u, v = int(round(p[0])), int(round(p[1]))
            cv2.circle(out, (u, v), 5, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(out, (u, v), 7, (0, 80, 0), 1, cv2.LINE_AA)
    for p in pts_xray_projected:
        if np.isfinite(p).all():
            u, v = int(round(p[0])), int(round(p[1]))
            cv2.circle(out, (u, v), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (u, v), 6, (0, 0, 120), 1, cv2.LINE_AA)
    cv2.putText(out, title_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        "green: RGB grid | red: H_xc * X-ray markers | cyan: residuals",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class OverlayData:
    def __init__(self, npz_path: Path):
        data = np.load(str(npz_path), allow_pickle=True)
        keys = set(data.files)
        required = {"K_rgb", "T_tc", "xray_points_xyz_c", "xray_points_uv", "checkerboard_corners_uv"}
        missing = required - keys
        if missing:
            raise ValueError(f"Missing required keys in overlay NPZ: {sorted(missing)}")
        if "snapshot_rgb_with_tip_bgr" in keys:
            self.camera_bgr = _to_uint8_bgr(data["snapshot_rgb_with_tip_bgr"], "snapshot_rgb_with_tip_bgr")
        elif "snapshot_rgb_bgr" in keys:
            self.camera_bgr = _to_uint8_bgr(data["snapshot_rgb_bgr"], "snapshot_rgb_bgr")
        else:
            raise ValueError("Missing snapshot_rgb_with_tip_bgr / snapshot_rgb_bgr.")
        self.K_rgb = _as_mat33(data["K_rgb"], "K_rgb")
        self.T_tc = _as_transform44(data["T_tc"], "T_tc")
        self.points_xyz_c_m = _as_xyz(data["xray_points_xyz_c"], "xray_points_xyz_c")
        self.points_uv_x = _as_uv(data["xray_points_uv"], "xray_points_uv")
        self.checkerboard_corners_uv = _as_uv(data["checkerboard_corners_uv"], "checkerboard_corners_uv")
        self.xray_shape_hw: tuple[int, int] | None = None
        for xray_key in ("xray_gray_u8", "xray_gray", "snapshot_xray_gray_u8", "snapshot_xray_u8"):
            if xray_key in keys:
                xray_img = np.asarray(data[xray_key])
                if xray_img.ndim == 2:
                    self.xray_shape_hw = (int(xray_img.shape[0]), int(xray_img.shape[1]))
                    break

        if self.xray_shape_hw is None:
            uv_max = np.nanmax(self.points_uv_x, axis=0)
            h = int(np.ceil(max(uv_max[1] + 1.0, 1.0)))
            w = int(np.ceil(max(uv_max[0] + 1.0, 1.0)))
            self.xray_shape_hw = (h, w)
            print(f"[INFO] No X-ray image key found. Using inferred shape from UV points: h={h}, w={w}")
        self.alpha = float(np.clip(_as_scalar(data["alpha"], "alpha"), 0.0, 1.0)) if "alpha" in keys else 0.5
        self.H_xc_saved = _as_mat33(data["H_xc"], "H_xc") if "H_xc" in keys else None
        self.K_xray_saved = _as_mat33(data["K_xray"], "K_xray") if "K_xray" in keys else None


class OverlayWindow(QMainWindow):
    def __init__(self, title: str, overlay_bgr: np.ndarray):
        super().__init__()
        self.overlay_bgr = np.asarray(overlay_bgr, dtype=np.uint8)
        self.setWindowTitle(title)
        self.resize(1400, 1000)
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setMinimumSize(1200, 800)
        self.lbl_image.setStyleSheet("background: #202020; border: 1px solid #505050;")
        self.setCentralWidget(self.lbl_image)
        self._update_image()

    def _update_image(self) -> None:
        pix = bgr_to_qpixmap(self.overlay_bgr)
        self.lbl_image.setPixmap(pix.scaled(self.lbl_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_image()


def show_overlay(*, windows: list[OverlayWindow], title: str, overlay_bgr: np.ndarray) -> None:
    win = OverlayWindow(title, overlay_bgr)
    win.show()
    windows.append(win)


def main() -> int:
    app = _ensure_qt_app()
    overlay_npz_path = pick_npz_file("Select overlay preview NPZ")
    if overlay_npz_path is None:
        return 0
    intrinsics_npz_path = pick_npz_file("Select X-ray intrinsics NPZ")
    if intrinsics_npz_path is None:
        return 0
    try:
        data = OverlayData(overlay_npz_path)
        K_xray = load_K_xray_from_intrinsics_npz(intrinsics_npz_path)
        windows: list[OverlayWindow] = []

        print("\n=== POSE-ONLY CHECK: RAW vs REORDERED X-RAY UV ===")
        pts_uv_reordered = reorder_grid_uv_horizontal(data.points_uv_x, steps_per_edge=10)
        res_handeye_raw = solve_pose(
            object_points_xyz=data.points_xyz_c_m,
            image_points_uv=data.points_uv_x,
            K=K_xray,
            dist_coeffs=None,
            dist_coeffs_rgb=None,
            pose_method="ippe_handeye",
            checkerboard_corners_uv=data.checkerboard_corners_uv,
            K_rgb=data.K_rgb,
            steps_per_edge=10,
            refine_with_iterative=False,
            refine_rgb_iterative=False,
            refine_xray_iterative=False,
        )
        res_handeye_reordered = solve_pose(
            object_points_xyz=data.points_xyz_c_m,
            image_points_uv=pts_uv_reordered,
            K=K_xray,
            dist_coeffs=None,
            dist_coeffs_rgb=None,
            pose_method="ippe_handeye",
            checkerboard_corners_uv=data.checkerboard_corners_uv,
            K_rgb=data.K_rgb,
            steps_per_edge=10,
            refine_with_iterative=False,
            refine_rgb_iterative=False,
            refine_xray_iterative=False,
        )
        print(f"RAW       reproj_mean = {res_handeye_raw.reproj_mean_px:.4f} px")
        print(f"REORDERED reproj_mean = {res_handeye_reordered.reproj_mean_px:.4f} px")

        res_handeye = res_handeye_reordered
        rgb_idx = int(res_handeye.candidate_index_rgb)
        xray_idx = int(res_handeye.candidate_index_xray)
        rgb_cand = res_handeye.all_candidates_rgb[rgb_idx]
        xray_cand = res_handeye.all_candidates[xray_idx]
        T_bc_mm = make_transform(rgb_cand.rvec, rgb_cand.tvec)
        T_bx_mm = make_transform(xray_cand.rvec, xray_cand.tvec)
        T_cx_m = compute_T_cx_from_T_bc_T_bx(T_bc_mm, T_bx_mm)
        T_xc_m = invert_transform(T_cx_m)
        d_x_mm = recompute_dx_from_T_xc(T_xc_m, data.T_tc)
        H_xc = estimate_plane_induced_homography(K_c=data.K_rgb, R_xc=T_xc_m[:3, :3], t_xc=T_xc_m[:3, 3], K_x=K_xray, d_x=d_x_mm)

        pts2d_rgb_ref = build_rgb_grid_from_anchors(data.checkerboard_corners_uv, steps_per_edge=10)
        pts2d_xray_to_rgb = apply_homography_to_uv(H=H_xc, uv=data.points_uv_x)
        residuals = pts2d_xray_to_rgb - pts2d_rgb_ref
        errors = np.linalg.norm(residuals, axis=1)
        finite = np.isfinite(errors)
        pts2d_xray_to_rgb_corr = pts2d_xray_to_rgb.copy()
        if np.any(finite):
            res_mean = np.nanmean(residuals[finite], axis=0)
            pts2d_xray_to_rgb_corr[finite] -= res_mean

        print("\n=== RENDERING-ONLY CHECK: RAW vs FLIPPED X-RAY PIXEL SPACE ===")
        h_x, w_x = data.xray_shape_hw
        F_flip_img = make_flip_x_matrix(w_x)
        F_flip_uv = make_flip_x_matrix_from_uv(data.points_uv_x)
        F_flip = F_flip_uv
        uv_flip = apply_homography_to_uv(F_flip, data.points_uv_x)
        pts_wrong = apply_homography_to_uv(H_xc, uv_flip)        # wrong: flipped image but raw H
        H_xc_for_flipped_source = H_xc @ np.linalg.inv(F_flip)   # correct mapping for flipped source
        pts_right = apply_homography_to_uv(H_xc_for_flipped_source, uv_flip)
        pts_alt = apply_homography_to_uv(H_xc @ F_flip, uv_flip)
        e_wrong = np.linalg.norm(pts_wrong - pts2d_rgb_ref, axis=1)
        e_right = np.linalg.norm(pts_right - pts2d_rgb_ref, axis=1)
        e_alt = np.linalg.norm(pts_alt - pts2d_rgb_ref, axis=1)
        print(f"flip matrix from image width (ref):\n{format_matrix(F_flip_img)}")
        print(f"flip matrix from UV span (used):\n{format_matrix(F_flip_uv)}")
        print(f"mean error wrong (flipped img + raw H): {np.nanmean(e_wrong):.3f} px")
        print(f"mean error right (flipped img + H@F):   {np.nanmean(e_right):.3f} px")
        print(f"mean error alt   (flipped img + H*F):   {np.nanmean(e_alt):.3f} px")

        debug_img = draw_point_debug(data.camera_bgr, pts2d_rgb_ref, pts2d_xray_to_rgb, title_text="DEBUG 1: original H_xc point mapping")
        debug_img_corr = draw_point_debug(data.camera_bgr, pts2d_rgb_ref, pts2d_xray_to_rgb_corr, title_text="DEBUG 2: after subtracting mean residual")
        debug_img_flip_wrong = draw_point_debug(data.camera_bgr, pts2d_rgb_ref, pts_wrong, title_text="DEBUG 3: flipped X-ray + RAW H (wrong)")
        debug_img_flip_right = draw_point_debug(data.camera_bgr, pts2d_rgb_ref, pts_right, title_text="DEBUG 4: flipped X-ray + H@F (correct)")
        show_overlay(windows=windows, title="DEBUG 1 | H_xc maps X-ray markers to RGB board points", overlay_bgr=debug_img)
        show_overlay(windows=windows, title="DEBUG 2 | H_xc point mapping after mean residual correction", overlay_bgr=debug_img_corr)
        show_overlay(windows=windows, title="DEBUG 3 | Flipped X-ray with RAW H (wrong space)", overlay_bgr=debug_img_flip_wrong)
        show_overlay(windows=windows, title="DEBUG 4 | Flipped X-ray with transformed H", overlay_bgr=debug_img_flip_right)
        app._debug_windows = windows
        return app.exec()
    except Exception as e:
        QMessageBox.critical(None, "debug_hxc_point_mapping", str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
