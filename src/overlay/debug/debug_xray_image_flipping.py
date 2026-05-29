# -*- coding: utf-8 -*-
"""
debug_xray_image_flipping.py

Test:
1) saved pose + RAW X-ray image + normal mask detection
2) saved pose + FLIP_LR X-ray image + same H + normal mask detection
3) saved pose + FLIP_LR X-ray image + same H + explicit flipped RAW mask
4) direct IPPE candidate 0 + FLIP_LR X-ray image + same H_direct + explicit flipped RAW mask
5) direct IPPE candidate 1 + FLIP_LR X-ray image + same H_direct + explicit flipped RAW mask
6) IPPE-Handeye selected + BASE K_xray + padded FLIP_LR around laser-cross u0
7) NEW K_xray + IPPE-Handeye selected + padded FLIP_LR around PP / laser-cross u0
8) NEW K_xray + IPPE-Handeye selected + normal FLIP_LR around image center + same H
9) FX=FY laser-cross K_xray + IPPE-Handeye selected + padded FLIP_LR around PP / laser-cross u0
10) IMAGE-CENTER FX=FY K_xray + IPPE-Handeye selected + normal FLIP_LR + same H
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox

from tracking.pose_solvers import solve_pose
from overlay.tools.homography import estimate_plane_induced_homography
from overlay.tools.warp import (
    blend_xray_overlay,
    _detect_xray_fov_mask,
    WarpedOverlay,
)


LASER_CROSS_U = 472.0


# ============================================================
# Qt helpers
# ============================================================

def _ensure_qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def pick_overlay_npz_file() -> Path | None:
    _ensure_qt_app()
    path, _ = QFileDialog.getOpenFileName(
        None,
        "Select overlay preview NPZ",
        "",
        "NPZ files (*.npz);;All files (*.*)",
    )
    return Path(path) if path else None


def pick_intrinsics_npz_file(title: str) -> Path | None:
    _ensure_qt_app()
    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        "NPZ files (*.npz);;All files (*.*)",
    )
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
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"{name} must be grayscale or BGR image, got shape {img.shape}")

    return img


def bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


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
        "[" + "  ".join(f"{v:+.{decimals}f}" for v in row) + "]"
        for row in np.asarray(M)
    )


def _safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def load_K_xray_from_intrinsics_npz(npz_path: Path) -> np.ndarray:
    data = np.load(str(npz_path), allow_pickle=True)

    for key in ["K_xray", "K", "camera_matrix"]:
        if key in data:
            return _as_mat33(data[key], key)

    raise KeyError(
        f"No K found in {npz_path.name}. Expected one of: K_xray, K, camera_matrix"
    )


def pad_and_flip_lr_about_u0(
    img_u8: np.ndarray,
    *,
    u0: float,
    border_value: int = 0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    img_u8 = np.asarray(img_u8)
    if img_u8.ndim != 2:
        raise ValueError(f"Expected 2D image/mask, got {img_u8.shape}")

    h, w = img_u8.shape[:2]

    left_needed = int(np.ceil(float(u0)))
    right_needed = int(np.ceil(float(w - 1) - float(u0)))
    half_width = max(left_needed, right_needed)

    new_w = 2 * half_width + 1
    u0_new = half_width

    pad_left = int(round(float(u0_new) - float(u0)))
    pad_right = int(new_w - w - pad_left)

    if pad_left < 0 or pad_right < 0:
        raise RuntimeError(
            f"Invalid padding: pad_left={pad_left}, pad_right={pad_right}, "
            f"w={w}, u0={u0}"
        )

    padded = cv2.copyMakeBorder(
        img_u8,
        0, 0,
        pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=border_value,
    )

    flipped = cv2.flip(padded, 1)

    T_pad = np.array(
        [
            [1.0, 0.0, float(pad_left)],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return flipped, T_pad, pad_left, pad_right


# ============================================================
# Data container
# ============================================================

class OverlayData:
    def __init__(self, npz_path: Path):
        self.npz_path = Path(npz_path)

        data = np.load(str(npz_path), allow_pickle=True)
        keys = set(data.files)

        required = {
            "xray_gray_u8",
            "K_rgb",
            "K_xray",
            "T_xc",
            "T_cx",
            "T_tc",
            "xray_points_xyz_c",
            "xray_points_uv",
            "checkerboard_corners_uv",
        }
        missing = required - keys
        if missing:
            raise ValueError(f"Missing required keys in overlay NPZ: {sorted(missing)}")

        if "snapshot_rgb_with_tip_bgr" in keys:
            self.camera_bgr = _to_uint8_bgr(
                data["snapshot_rgb_with_tip_bgr"],
                "snapshot_rgb_with_tip_bgr",
            )
        elif "snapshot_rgb_bgr" in keys:
            self.camera_bgr = _to_uint8_bgr(
                data["snapshot_rgb_bgr"],
                "snapshot_rgb_bgr",
            )
        else:
            raise ValueError("Missing snapshot_rgb_with_tip_bgr / snapshot_rgb_bgr.")

        self.xray_gray_u8 = np.asarray(data["xray_gray_u8"])
        if self.xray_gray_u8.ndim != 2:
            raise ValueError(f"xray_gray_u8 must be grayscale, got {self.xray_gray_u8.shape}")
        if self.xray_gray_u8.dtype != np.uint8:
            self.xray_gray_u8 = np.clip(self.xray_gray_u8, 0, 255).astype(np.uint8)

        self.K_rgb = _as_mat33(data["K_rgb"], "K_rgb")
        self.K_xray = _as_mat33(data["K_xray"], "K_xray")

        self.T_xc = _as_transform44(data["T_xc"], "T_xc")
        self.T_cx = _as_transform44(data["T_cx"], "T_cx")
        self.T_tc = _as_transform44(data["T_tc"], "T_tc")

        self.points_xyz_c_m = _as_xyz(data["xray_points_xyz_c"], "xray_points_xyz_c")
        self.points_uv_x = _as_uv(data["xray_points_uv"], "xray_points_uv")
        self.checkerboard_corners_uv = _as_uv(
            data["checkerboard_corners_uv"],
            "checkerboard_corners_uv",
        )

        self.alpha = 0.5
        if "alpha" in keys:
            self.alpha = float(np.clip(_as_scalar(data["alpha"], "alpha"), 0.0, 1.0))

        self.d_x = None
        if "d_x" in keys:
            self.d_x = _as_scalar(data["d_x"], "d_x")

        self.H_xc_saved = None
        if "H_xc" in keys:
            self.H_xc_saved = _as_mat33(data["H_xc"], "H_xc")


# ============================================================
# Overlay helpers
# ============================================================

def compute_H_xc_from_T_xc(
    *,
    data: OverlayData,
    T_xc_m: np.ndarray,
    d_x_mm: float | None = None,
    K_xray: np.ndarray,
) -> tuple[np.ndarray, float]:
    if d_x_mm is None:
        d_x_mm = recompute_dx_from_T_xc(T_xc_m, data.T_tc)

    H_xc = estimate_plane_induced_homography(
        K_c=data.K_rgb,
        R_xc=T_xc_m[:3, :3],
        t_xc=T_xc_m[:3, 3],
        K_x=K_xray,
        d_x=d_x_mm,
    )

    return H_xc, float(d_x_mm)


def make_overlay(
    *,
    data: OverlayData,
    xray_gray_u8: np.ndarray,
    H_xc: np.ndarray,
) -> np.ndarray:
    overlay_bgr, _ = blend_xray_overlay(
        camera_bgr=data.camera_bgr,
        xray_gray_u8=xray_gray_u8,
        H_xc=H_xc,
        alpha=data.alpha,
    )
    return overlay_bgr


def make_overlay_with_explicit_mask(
    *,
    camera_bgr: np.ndarray,
    xray_gray_u8: np.ndarray,
    xray_fov_mask: np.ndarray,
    H_xc: np.ndarray,
    alpha: float,
) -> np.ndarray:
    camera_bgr = np.asarray(camera_bgr)
    xray_gray_u8 = np.asarray(xray_gray_u8)
    xray_fov_mask = np.asarray(xray_fov_mask)

    if camera_bgr.ndim != 3 or camera_bgr.shape[2] != 3:
        raise ValueError(f"camera_bgr must have shape (H,W,3), got {camera_bgr.shape}")
    if camera_bgr.dtype != np.uint8:
        raise ValueError(f"camera_bgr must be uint8, got {camera_bgr.dtype}")

    if xray_gray_u8.ndim != 2:
        raise ValueError(f"xray_gray_u8 must be grayscale, got {xray_gray_u8.shape}")
    if xray_gray_u8.dtype != np.uint8:
        raise ValueError(f"xray_gray_u8 must be uint8, got {xray_gray_u8.dtype}")

    if xray_fov_mask.ndim != 2:
        raise ValueError(f"xray_fov_mask must be grayscale, got {xray_fov_mask.shape}")
    if xray_fov_mask.dtype != np.uint8:
        raise ValueError(f"xray_fov_mask must be uint8, got {xray_fov_mask.dtype}")

    if xray_fov_mask.shape != xray_gray_u8.shape:
        raise ValueError(
            f"mask/image shape mismatch: mask={xray_fov_mask.shape}, image={xray_gray_u8.shape}"
        )

    Hc, Wc = camera_bgr.shape[:2]

    xray_masked = cv2.bitwise_and(
        xray_gray_u8,
        xray_gray_u8,
        mask=xray_fov_mask,
    )

    warped_xray_gray = cv2.warpPerspective(
        xray_masked,
        H_xc,
        dsize=(Wc, Hc),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    overlay_mask = cv2.warpPerspective(
        xray_fov_mask,
        H_xc,
        dsize=(Wc, Hc),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    overlay_mask = (overlay_mask > 0).astype(np.uint8) * 255

    cache = WarpedOverlay(
        warped_xray_gray=warped_xray_gray,
        overlay_mask=overlay_mask,
    )

    return cache.blend(camera_bgr, alpha=alpha)


# ============================================================
# Window
# ============================================================

class OverlayImageWindow(QMainWindow):
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
        pix = pix.scaled(
            self.lbl_image.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.lbl_image.setPixmap(pix)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_image()


def show_overlay(
    *,
    windows: list[OverlayImageWindow],
    title: str,
    overlay_bgr: np.ndarray,
) -> None:
    win = OverlayImageWindow(title, overlay_bgr)
    win.show()
    windows.append(win)


# ============================================================
# Main
# ============================================================

def main() -> int:
    app = _ensure_qt_app()

    overlay_npz_path = pick_overlay_npz_file()
    if overlay_npz_path is None:
        return 0

    intrinsics_npz_path_base = pick_intrinsics_npz_file(
        "Select BASE X-ray intrinsics NPZ for blocks 1-6"
    )
    if intrinsics_npz_path_base is None:
        return 0

    try:
        data = OverlayData(overlay_npz_path)
        K_xray_base = load_K_xray_from_intrinsics_npz(intrinsics_npz_path_base)

        base_name = _safe_name(overlay_npz_path)
        windows: list[OverlayImageWindow] = []

        print("\n" + "=" * 120)
        print("DEBUG XRAY IMAGE FLIPPING")
        print("=" * 120)
        print(f"Overlay NPZ: {overlay_npz_path}")
        print(f"BASE intrinsics NPZ for blocks 1-6: {intrinsics_npz_path_base}")

        print("\nK_xray stored in overlay NPZ =")
        print(format_matrix(data.K_xray))

        print("\nK_xray BASE used for blocks 1-6 =")
        print(format_matrix(K_xray_base))

        print("\nSaved T_cx =")
        print(format_matrix(data.T_cx))

        print("\nSaved T_xc =")
        print(format_matrix(data.T_xc))

        xray_raw = data.xray_gray_u8
        xray_flip_lr = cv2.flip(xray_raw, 1)

        xray_mask_raw = _detect_xray_fov_mask(xray_raw)
        xray_mask_flip_lr = cv2.flip(xray_mask_raw, 1)

        if data.d_x is None:
            d_x_saved = recompute_dx_from_T_xc(data.T_xc, data.T_tc)
            print(f"\nSaved d_x missing in NPZ -> recomputed d_x [mm] = {d_x_saved:+.6f}")
        else:
            d_x_saved = data.d_x
            print(f"\nSaved d_x [mm] = {d_x_saved:+.6f}")

        H_saved, d_x_saved = compute_H_xc_from_T_xc(
            data=data,
            T_xc_m=data.T_xc,
            d_x_mm=d_x_saved,
            K_xray=K_xray_base,
        )

        print("\nH_saved recomputed with BASE K_xray =")
        print(format_matrix(H_saved))

        if data.H_xc_saved is not None:
            print("\nH_xc saved in overlay NPZ =")
            print(format_matrix(data.H_xc_saved))
            print("\nH_saved(BASE K) - H_xc saved in overlay NPZ =")
            print(format_matrix(H_saved - data.H_xc_saved))

        # ========================================================
        # 1) Saved pose: RAW image, normal mask detection
        # ========================================================
        overlay_saved_raw = make_overlay(
            data=data,
            xray_gray_u8=xray_raw,
            H_xc=H_saved,
        )

        print("\n" + "-" * 120)
        print("1) SAVED POSE | BASE K_xray | X-ray RAW | H_saved | normal mask detection")
        print("-" * 120)
        print(f"d_x [mm] = {d_x_saved:+.6f}")

        show_overlay(
            windows=windows,
            title=f"{base_name} | 1 BASE K | Saved pose | RAW | normal mask",
            overlay_bgr=overlay_saved_raw,
        )

        # ========================================================
        # 2) Saved pose: FLIP_LR image, same H, normal mask detection
        # ========================================================
        overlay_saved_flip_same_H = make_overlay(
            data=data,
            xray_gray_u8=xray_flip_lr,
            H_xc=H_saved,
        )

        print("\n" + "-" * 120)
        print("2) SAVED POSE | BASE K_xray | X-ray FLIP_LR | same H_saved | normal mask")
        print("-" * 120)
        print(f"d_x [mm] = {d_x_saved:+.6f}")

        show_overlay(
            windows=windows,
            title=f"{base_name} | 2 BASE K | Saved pose | FLIP_LR | same H | normal mask",
            overlay_bgr=overlay_saved_flip_same_H,
        )

        # ========================================================
        # 3) Saved pose: FLIP_LR image, same H, explicit flipped RAW mask
        # ========================================================
        overlay_saved_flip_explicit_mask = make_overlay_with_explicit_mask(
            camera_bgr=data.camera_bgr,
            xray_gray_u8=xray_flip_lr,
            xray_fov_mask=xray_mask_flip_lr,
            H_xc=H_saved,
            alpha=data.alpha,
        )

        print("\n" + "-" * 120)
        print("3) SAVED POSE | BASE K_xray | FLIP_LR | same H_saved | explicit flipped mask")
        print("-" * 120)
        print(f"d_x [mm] = {d_x_saved:+.6f}")

        show_overlay(
            windows=windows,
            title=f"{base_name} | 3 BASE K | Saved pose | FLIP_LR | same H | explicit flipped mask",
            overlay_bgr=overlay_saved_flip_explicit_mask,
        )

        # ========================================================
        # 4/5) Direct IPPE candidates: FLIP_LR image, same H_direct
        # ========================================================
        print("\n" + "=" * 120)
        print("DIRECT IPPE CANDIDATES | BASE K_xray | X-ray image FLIP_LR | same H_direct")
        print("=" * 120)

        res_direct = solve_pose(
            object_points_xyz=data.points_xyz_c_m,
            image_points_uv=data.points_uv_x,
            K=K_xray_base,
            dist_coeffs=None,
            pose_method="ippe",
            refine_with_iterative=False,
            use_xray_ippe_selection_rule=False,
        )

        if res_direct.all_candidates is None or len(res_direct.all_candidates) != 2:
            raise RuntimeError(
                f"Expected exactly 2 direct IPPE candidates, got "
                f"{0 if res_direct.all_candidates is None else len(res_direct.all_candidates)}."
            )

        for cand_idx, cand in enumerate(res_direct.all_candidates):
            T_cx_direct_m = make_transform(cand.rvec, cand.tvec)
            T_xc_direct_m = invert_transform(T_cx_direct_m)

            H_direct, d_x_direct = compute_H_xc_from_T_xc(
                data=data,
                T_xc_m=T_xc_direct_m,
                d_x_mm=None,
                K_xray=K_xray_base,
            )

            overlay_direct_flip = make_overlay_with_explicit_mask(
                camera_bgr=data.camera_bgr,
                xray_gray_u8=xray_flip_lr,
                xray_fov_mask=xray_mask_flip_lr,
                H_xc=H_direct,
                alpha=data.alpha,
            )

            print("\n" + "-" * 120)
            print(f"DIRECT IPPE candidate {cand_idx} | BASE K | FLIP_LR | same H_direct")
            print("-" * 120)
            print(f"reproj mean [px]    = {cand.reproj_mean_px:.6f}")
            print(f"reproj median [px]  = {cand.reproj_median_px:.6f}")
            print(f"reproj max [px]     = {cand.reproj_max_px:.6f}")

            print("\nT_cx_direct [m] =")
            print(format_matrix(T_cx_direct_m))

            print("\nT_xc_direct [m] =")
            print(format_matrix(T_xc_direct_m))

            print(f"\nd_x_direct [mm] = {d_x_direct:+.6f}")

            print("\nH_direct =")
            print(format_matrix(H_direct))

            show_overlay(
                windows=windows,
                title=f"{base_name} | 4/5 BASE K | Direct IPPE {cand_idx} | FLIP_LR",
                overlay_bgr=overlay_direct_flip,
            )

        # ========================================================
        # 6) IPPE-Handeye selected: padded FLIP_LR around laser-cross u0
        # ========================================================
        print("\n" + "=" * 120)
        print("6) IPPE HANDEYE SELECTED | BASE K_XRAY | padded FLIP_LR around laser-cross u0")
        print("=" * 120)

        xray_flip_laser, T_pad_laser, pad_left, pad_right = pad_and_flip_lr_about_u0(
            xray_raw,
            u0=LASER_CROSS_U,
            border_value=0,
        )

        mask_flip_laser, _, pad_left_m, pad_right_m = pad_and_flip_lr_about_u0(
            xray_mask_raw,
            u0=LASER_CROSS_U,
            border_value=0,
        )

        if pad_left != pad_left_m or pad_right != pad_right_m:
            raise RuntimeError("Image and mask padding do not match.")

        res_handeye = solve_pose(
            object_points_xyz=data.points_xyz_c_m,
            image_points_uv=data.points_uv_x,
            K=K_xray_base,
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

        if res_handeye.all_candidates_rgb is None or len(res_handeye.all_candidates_rgb) != 2:
            raise RuntimeError("Expected exactly 2 RGB IPPE candidates for handeye.")
        if res_handeye.all_candidates is None or len(res_handeye.all_candidates) != 2:
            raise RuntimeError("Expected exactly 2 X-ray IPPE candidates for handeye.")
        if res_handeye.candidate_index_rgb is None:
            raise RuntimeError("No selected RGB candidate index returned.")
        if res_handeye.candidate_index_xray is None:
            raise RuntimeError("No selected X-ray candidate index returned.")

        rgb_idx = int(res_handeye.candidate_index_rgb)
        xray_idx = int(res_handeye.candidate_index_xray)

        rgb_cand = res_handeye.all_candidates_rgb[rgb_idx]
        xray_cand = res_handeye.all_candidates[xray_idx]

        T_bc_mm = make_transform(rgb_cand.rvec, rgb_cand.tvec)
        T_bx_mm = make_transform(xray_cand.rvec, xray_cand.tvec)

        T_cx_handeye_m = compute_T_cx_from_T_bc_T_bx(T_bc_mm, T_bx_mm)
        T_xc_handeye_m = invert_transform(T_cx_handeye_m)

        H_handeye, d_x_handeye = compute_H_xc_from_T_xc(
            data=data,
            T_xc_m=T_xc_handeye_m,
            d_x_mm=None,
            K_xray=K_xray_base,
        )

        H_handeye_padded = H_handeye @ np.linalg.inv(T_pad_laser)

        overlay_handeye_laser_flip = make_overlay_with_explicit_mask(
            camera_bgr=data.camera_bgr,
            xray_gray_u8=xray_flip_laser,
            xray_fov_mask=mask_flip_laser,
            H_xc=H_handeye_padded,
            alpha=data.alpha,
        )

        print(f"selected RGB candidate   = {rgb_idx}")
        print(f"selected X-ray candidate = {xray_idx}")
        print(f"laser u0                 = {LASER_CROSS_U:.3f}")
        print(f"pad_left                 = {pad_left}")
        print(f"pad_right                = {pad_right}")
        print(f"padded image shape       = {xray_flip_laser.shape}")

        print("\nT_bc [mm] =")
        print(format_matrix(T_bc_mm))

        print("\nT_bx [mm] =")
        print(format_matrix(T_bx_mm))

        print("\nT_cx_handeye [m] =")
        print(format_matrix(T_cx_handeye_m))

        print("\nT_xc_handeye [m] =")
        print(format_matrix(T_xc_handeye_m))

        print(f"\nd_x_handeye [mm] = {d_x_handeye:+.6f}")

        print("\nT_pad_laser =")
        print(format_matrix(T_pad_laser))

        print("\nH_handeye =")
        print(format_matrix(H_handeye))

        print("\nH_handeye_padded = H_handeye @ inv(T_pad_laser)")
        print(format_matrix(H_handeye_padded))

        show_overlay(
            windows=windows,
            title=(
                f"{base_name} | 6 BASE K_xray | Handeye selected | "
                f"padded FLIP_LR around laser u={LASER_CROSS_U:.0f}"
            ),
            overlay_bgr=overlay_handeye_laser_flip,
        )

        # ========================================================
        # 7) Load additional NEW intrinsics
        # ========================================================
        print("\n" + "=" * 120)
        print("7) LOAD NEW K_XRAY | recompute handeye + H | LASER/PP FLIP")
        print("=" * 120)

        intrinsics_npz_path_new = pick_intrinsics_npz_file(
            "Select NEW X-ray intrinsics NPZ for block 7 LASER/PP FLIP"
        )
        if intrinsics_npz_path_new is None:
            print("\n[INFO] No additional intrinsics selected. Skipping blocks 7, 8, 9 and 10.")
            app._overlay_windows = windows
            return app.exec()

        K_xray_new = load_K_xray_from_intrinsics_npz(intrinsics_npz_path_new)

        print(f"\nNEW intrinsics NPZ:\n  {intrinsics_npz_path_new}")

        print("\nK_xray BASE used for blocks 1-6 =")
        print(format_matrix(K_xray_base))

        print("\nK_xray NEW used for block 7 =")
        print(format_matrix(K_xray_new))

        print("\nK_xray NEW - K_xray BASE =")
        print(format_matrix(K_xray_new - K_xray_base))

        laser_u_new = float(K_xray_new[0, 2])
        print(f"\nUsing u0 from NEW K_xray principal point: {laser_u_new:.6f}")

        xray_flip_laser_newK, T_pad_laser_newK, pad_left_newK, pad_right_newK = (
            pad_and_flip_lr_about_u0(
                xray_raw,
                u0=laser_u_new,
                border_value=0,
            )
        )

        mask_flip_laser_newK, _, pad_left_m_newK, pad_right_m_newK = (
            pad_and_flip_lr_about_u0(
                xray_mask_raw,
                u0=laser_u_new,
                border_value=0,
            )
        )

        if pad_left_newK != pad_left_m_newK or pad_right_newK != pad_right_m_newK:
            raise RuntimeError("Image and mask padding do not match for new K.")

        res_handeye_newK = solve_pose(
            object_points_xyz=data.points_xyz_c_m,
            image_points_uv=data.points_uv_x,
            K=K_xray_new,
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

        if (
            res_handeye_newK.all_candidates_rgb is None
            or len(res_handeye_newK.all_candidates_rgb) != 2
        ):
            raise RuntimeError("Expected exactly 2 RGB IPPE candidates for new-K handeye.")

        if (
            res_handeye_newK.all_candidates is None
            or len(res_handeye_newK.all_candidates) != 2
        ):
            raise RuntimeError("Expected exactly 2 X-ray IPPE candidates for new-K handeye.")

        if res_handeye_newK.candidate_index_rgb is None:
            raise RuntimeError("No selected RGB candidate index returned for new-K handeye.")

        if res_handeye_newK.candidate_index_xray is None:
            raise RuntimeError("No selected X-ray candidate index returned for new-K handeye.")

        rgb_idx_newK = int(res_handeye_newK.candidate_index_rgb)
        xray_idx_newK = int(res_handeye_newK.candidate_index_xray)

        rgb_cand_newK = res_handeye_newK.all_candidates_rgb[rgb_idx_newK]
        xray_cand_newK = res_handeye_newK.all_candidates[xray_idx_newK]

        T_bc_newK_mm = make_transform(rgb_cand_newK.rvec, rgb_cand_newK.tvec)
        T_bx_newK_mm = make_transform(xray_cand_newK.rvec, xray_cand_newK.tvec)

        T_cx_newK_m = compute_T_cx_from_T_bc_T_bx(T_bc_newK_mm, T_bx_newK_mm)
        T_xc_newK_m = invert_transform(T_cx_newK_m)

        H_newK, d_x_newK = compute_H_xc_from_T_xc(
            data=data,
            T_xc_m=T_xc_newK_m,
            d_x_mm=None,
            K_xray=K_xray_new,
        )

        H_newK_padded = H_newK @ np.linalg.inv(T_pad_laser_newK)

        overlay_newK_laser_flip = make_overlay_with_explicit_mask(
            camera_bgr=data.camera_bgr,
            xray_gray_u8=xray_flip_laser_newK,
            xray_fov_mask=mask_flip_laser_newK,
            H_xc=H_newK_padded,
            alpha=data.alpha,
        )

        print(f"selected RGB candidate   = {rgb_idx_newK}")
        print(f"selected X-ray candidate = {xray_idx_newK}")
        print(f"laser / PP u0            = {laser_u_new:.6f}")
        print(f"pad_left                 = {pad_left_newK}")
        print(f"pad_right                = {pad_right_newK}")
        print(f"padded image shape       = {xray_flip_laser_newK.shape}")

        print("\nT_bc_newK [mm] =")
        print(format_matrix(T_bc_newK_mm))

        print("\nT_bx_newK [mm] =")
        print(format_matrix(T_bx_newK_mm))

        print("\nT_cx_newK [m] =")
        print(format_matrix(T_cx_newK_m))

        print("\nT_xc_newK [m] =")
        print(format_matrix(T_xc_newK_m))

        print(f"\nd_x_newK [mm] = {d_x_newK:+.6f}")

        print("\nT_pad_laser_newK =")
        print(format_matrix(T_pad_laser_newK))

        print("\nH_newK =")
        print(format_matrix(H_newK))

        print("\nH_newK_padded = H_newK @ inv(T_pad_laser_newK)")
        print(format_matrix(H_newK_padded))

        show_overlay(
            windows=windows,
            title=(
                f"{base_name} | 7 NEW K_xray | Handeye selected | "
                f"padded FLIP_LR around PP u={laser_u_new:.0f}"
            ),
            overlay_bgr=overlay_newK_laser_flip,
        )

        # ========================================================
        # 8) Load additional intrinsics
        # ========================================================
        print("\n" + "=" * 120)
        print("8) LOAD NEW K_XRAY | recompute handeye + H | CENTER FLIP | SAME H")
        print("=" * 120)

        intrinsics_npz_path_center = pick_intrinsics_npz_file(
            "Select X-ray intrinsics NPZ for block 8 CENTER FLIP"
        )
        if intrinsics_npz_path_center is None:
            print("\n[INFO] No center-flip intrinsics selected. Skipping blocks 8, 9 and 10.")
            app._overlay_windows = windows
            return app.exec()

        K_xray_center = load_K_xray_from_intrinsics_npz(intrinsics_npz_path_center)

        print(f"\nCENTER-FLIP intrinsics NPZ:\n  {intrinsics_npz_path_center}")

        print("\nK_xray CENTER-FLIP =")
        print(format_matrix(K_xray_center))

        print("\nK_xray CENTER-FLIP - K_xray BASE =")
        print(format_matrix(K_xray_center - K_xray_base))

        xray_flip_center = cv2.flip(xray_raw, 1)
        mask_flip_center = cv2.flip(xray_mask_raw, 1)

        res_handeye_center = solve_pose(
            object_points_xyz=data.points_xyz_c_m,
            image_points_uv=data.points_uv_x,
            K=K_xray_center,
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

        if (
            res_handeye_center.all_candidates_rgb is None
            or len(res_handeye_center.all_candidates_rgb) != 2
        ):
            raise RuntimeError("Expected exactly 2 RGB IPPE candidates for center-flip handeye.")

        if (
            res_handeye_center.all_candidates is None
            or len(res_handeye_center.all_candidates) != 2
        ):
            raise RuntimeError("Expected exactly 2 X-ray IPPE candidates for center-flip handeye.")

        if res_handeye_center.candidate_index_rgb is None:
            raise RuntimeError("No selected RGB candidate index returned for center-flip handeye.")

        if res_handeye_center.candidate_index_xray is None:
            raise RuntimeError("No selected X-ray candidate index returned for center-flip handeye.")

        rgb_idx_center = int(res_handeye_center.candidate_index_rgb)
        xray_idx_center = int(res_handeye_center.candidate_index_xray)

        rgb_cand_center = res_handeye_center.all_candidates_rgb[rgb_idx_center]
        xray_cand_center = res_handeye_center.all_candidates[xray_idx_center]

        T_bc_center_mm = make_transform(rgb_cand_center.rvec, rgb_cand_center.tvec)
        T_bx_center_mm = make_transform(xray_cand_center.rvec, xray_cand_center.tvec)

        T_cx_center_m = compute_T_cx_from_T_bc_T_bx(T_bc_center_mm, T_bx_center_mm)
        T_xc_center_m = invert_transform(T_cx_center_m)

        H_center, d_x_center = compute_H_xc_from_T_xc(
            data=data,
            T_xc_m=T_xc_center_m,
            d_x_mm=None,
            K_xray=K_xray_center,
        )

        overlay_center_flip = make_overlay_with_explicit_mask(
            camera_bgr=data.camera_bgr,
            xray_gray_u8=xray_flip_center,
            xray_fov_mask=mask_flip_center,
            H_xc=H_center,
            alpha=data.alpha,
        )

        print(f"selected RGB candidate   = {rgb_idx_center}")
        print(f"selected X-ray candidate = {xray_idx_center}")
        print("center flip              = cv2.flip(image, 1)")
        print("H usage                  = SAME H_center, no additional flip transform")

        print("\nT_bc_center [mm] =")
        print(format_matrix(T_bc_center_mm))

        print("\nT_bx_center [mm] =")
        print(format_matrix(T_bx_center_mm))

        print("\nT_cx_center [m] =")
        print(format_matrix(T_cx_center_m))

        print("\nT_xc_center [m] =")
        print(format_matrix(T_xc_center_m))

        print(f"\nd_x_center [mm] = {d_x_center:+.6f}")

        print("\nH_center =")
        print(format_matrix(H_center))

        show_overlay(
            windows=windows,
            title=(
                f"{base_name} | 8 CENTER FLIP | Handeye selected | "
                f"normal FLIP_LR | same H"
            ),
            overlay_bgr=overlay_center_flip,
        )

        # ========================================================
        # 9) Load fx=fy laser-cross intrinsics
        # ========================================================
        print("\n" + "=" * 120)
        print("9) LOAD FX=FY + LASER-CROSS K_XRAY | recompute handeye + H | LASER/PP FLIP")
        print("=" * 120)

        intrinsics_npz_path_equal_laser = pick_intrinsics_npz_file(
            "Select fx=fy + laser-cross X-ray intrinsics NPZ for block 9"
        )
        if intrinsics_npz_path_equal_laser is None:
            print("\n[INFO] No fx=fy laser-cross intrinsics selected. Skipping blocks 9 and 10.")
            app._overlay_windows = windows
            return app.exec()

        K_xray_equal_laser = load_K_xray_from_intrinsics_npz(
            intrinsics_npz_path_equal_laser
        )

        print(f"\nFX=FY LASER-CROSS intrinsics NPZ:\n  {intrinsics_npz_path_equal_laser}")

        print("\nK_xray FX=FY LASER-CROSS =")
        print(format_matrix(K_xray_equal_laser))

        print("\nK_xray FX=FY LASER-CROSS - K_xray BASE =")
        print(format_matrix(K_xray_equal_laser - K_xray_base))

        laser_u_equal_laser = float(K_xray_equal_laser[0, 2])
        print(f"\nUsing u0 from FX=FY LASER-CROSS K_xray principal point: {laser_u_equal_laser:.6f}")

        xray_flip_equal_laser, T_pad_equal_laser, pad_left_equal_laser, pad_right_equal_laser = (
            pad_and_flip_lr_about_u0(
                xray_raw,
                u0=laser_u_equal_laser,
                border_value=0,
            )
        )

        mask_flip_equal_laser, _, pad_left_m_equal_laser, pad_right_m_equal_laser = (
            pad_and_flip_lr_about_u0(
                xray_mask_raw,
                u0=laser_u_equal_laser,
                border_value=0,
            )
        )

        if (
            pad_left_equal_laser != pad_left_m_equal_laser
            or pad_right_equal_laser != pad_right_m_equal_laser
        ):
            raise RuntimeError("Image and mask padding do not match for block 9.")

        res_handeye_equal_laser = solve_pose(
            object_points_xyz=data.points_xyz_c_m,
            image_points_uv=data.points_uv_x,
            K=K_xray_equal_laser,
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

        if (
            res_handeye_equal_laser.all_candidates_rgb is None
            or len(res_handeye_equal_laser.all_candidates_rgb) != 2
        ):
            raise RuntimeError("Expected exactly 2 RGB IPPE candidates for block 9.")

        if (
            res_handeye_equal_laser.all_candidates is None
            or len(res_handeye_equal_laser.all_candidates) != 2
        ):
            raise RuntimeError("Expected exactly 2 X-ray IPPE candidates for block 9.")

        if res_handeye_equal_laser.candidate_index_rgb is None:
            raise RuntimeError("No selected RGB candidate index returned for block 9.")

        if res_handeye_equal_laser.candidate_index_xray is None:
            raise RuntimeError("No selected X-ray candidate index returned for block 9.")

        rgb_idx_equal_laser = int(res_handeye_equal_laser.candidate_index_rgb)
        xray_idx_equal_laser = int(res_handeye_equal_laser.candidate_index_xray)

        rgb_cand_equal_laser = res_handeye_equal_laser.all_candidates_rgb[rgb_idx_equal_laser]
        xray_cand_equal_laser = res_handeye_equal_laser.all_candidates[xray_idx_equal_laser]

        T_bc_equal_laser_mm = make_transform(
            rgb_cand_equal_laser.rvec,
            rgb_cand_equal_laser.tvec,
        )
        T_bx_equal_laser_mm = make_transform(
            xray_cand_equal_laser.rvec,
            xray_cand_equal_laser.tvec,
        )

        T_cx_equal_laser_m = compute_T_cx_from_T_bc_T_bx(
            T_bc_equal_laser_mm,
            T_bx_equal_laser_mm,
        )
        T_xc_equal_laser_m = invert_transform(T_cx_equal_laser_m)

        H_equal_laser, d_x_equal_laser = compute_H_xc_from_T_xc(
            data=data,
            T_xc_m=T_xc_equal_laser_m,
            d_x_mm=None,
            K_xray=K_xray_equal_laser,
        )

        H_equal_laser_padded = H_equal_laser @ np.linalg.inv(T_pad_equal_laser)

        overlay_equal_laser_flip = make_overlay_with_explicit_mask(
            camera_bgr=data.camera_bgr,
            xray_gray_u8=xray_flip_equal_laser,
            xray_fov_mask=mask_flip_equal_laser,
            H_xc=H_equal_laser_padded,
            alpha=data.alpha,
        )

        print(f"selected RGB candidate   = {rgb_idx_equal_laser}")
        print(f"selected X-ray candidate = {xray_idx_equal_laser}")
        print(f"laser / PP u0            = {laser_u_equal_laser:.6f}")
        print(f"pad_left                 = {pad_left_equal_laser}")
        print(f"pad_right                = {pad_right_equal_laser}")
        print(f"padded image shape       = {xray_flip_equal_laser.shape}")

        print("\nT_bc_equal_laser [mm] =")
        print(format_matrix(T_bc_equal_laser_mm))

        print("\nT_bx_equal_laser [mm] =")
        print(format_matrix(T_bx_equal_laser_mm))

        print("\nT_cx_equal_laser [m] =")
        print(format_matrix(T_cx_equal_laser_m))

        print("\nT_xc_equal_laser [m] =")
        print(format_matrix(T_xc_equal_laser_m))

        print(f"\nd_x_equal_laser [mm] = {d_x_equal_laser:+.6f}")

        print("\nT_pad_equal_laser =")
        print(format_matrix(T_pad_equal_laser))

        print("\nH_equal_laser =")
        print(format_matrix(H_equal_laser))

        print("\nH_equal_laser_padded = H_equal_laser @ inv(T_pad_equal_laser)")
        print(format_matrix(H_equal_laser_padded))

        show_overlay(
            windows=windows,
            title=(
                f"{base_name} | 9 FX=FY LASER-CROSS K | Handeye selected | "
                f"padded FLIP_LR around PP u={laser_u_equal_laser:.0f}"
            ),
            overlay_bgr=overlay_equal_laser_flip,
        )

        # ========================================================
        # 10) Load image-center fx=fy intrinsics
        #     Recompute IPPE-Handeye, T_cx, T_xc, d_x and H
        #     Normal center flip, but SAME H
        # ========================================================
        print("\n" + "=" * 120)
        print("10) LOAD IMAGE-CENTER FX=FY K_XRAY | recompute handeye + H | NORMAL FLIP | SAME H")
        print("=" * 120)

        intrinsics_npz_path_image_center = pick_intrinsics_npz_file(
            "Select image-center fx=fy X-ray intrinsics NPZ for block 10"
        )
        if intrinsics_npz_path_image_center is None:
            print("\n[INFO] No image-center fx=fy intrinsics selected. Skipping block 10.")
            app._overlay_windows = windows
            return app.exec()

        K_xray_image_center = load_K_xray_from_intrinsics_npz(
            intrinsics_npz_path_image_center
        )

        print(f"\nIMAGE-CENTER FX=FY intrinsics NPZ:\n  {intrinsics_npz_path_image_center}")

        print("\nK_xray IMAGE-CENTER FX=FY =")
        print(format_matrix(K_xray_image_center))

        print("\nK_xray IMAGE-CENTER FX=FY - K_xray BASE =")
        print(format_matrix(K_xray_image_center - K_xray_base))

        # Normal image flip around image center.
        # IMPORTANT: H is NOT modified.
        xray_flip_image_center = cv2.flip(xray_raw, 1)
        mask_flip_image_center = cv2.flip(xray_mask_raw, 1)

        res_handeye_image_center = solve_pose(
            object_points_xyz=data.points_xyz_c_m,
            image_points_uv=data.points_uv_x,
            K=K_xray_image_center,
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

        if (
            res_handeye_image_center.all_candidates_rgb is None
            or len(res_handeye_image_center.all_candidates_rgb) != 2
        ):
            raise RuntimeError("Expected exactly 2 RGB IPPE candidates for block 10.")

        if (
            res_handeye_image_center.all_candidates is None
            or len(res_handeye_image_center.all_candidates) != 2
        ):
            raise RuntimeError("Expected exactly 2 X-ray IPPE candidates for block 10.")

        if res_handeye_image_center.candidate_index_rgb is None:
            raise RuntimeError("No selected RGB candidate index returned for block 10.")

        if res_handeye_image_center.candidate_index_xray is None:
            raise RuntimeError("No selected X-ray candidate index returned for block 10.")

        rgb_idx_image_center = int(res_handeye_image_center.candidate_index_rgb)
        xray_idx_image_center = int(res_handeye_image_center.candidate_index_xray)

        rgb_cand_image_center = res_handeye_image_center.all_candidates_rgb[
            rgb_idx_image_center
        ]
        xray_cand_image_center = res_handeye_image_center.all_candidates[
            xray_idx_image_center
        ]

        T_bc_image_center_mm = make_transform(
            rgb_cand_image_center.rvec,
            rgb_cand_image_center.tvec,
        )
        T_bx_image_center_mm = make_transform(
            xray_cand_image_center.rvec,
            xray_cand_image_center.tvec,
        )

        T_cx_image_center_m = compute_T_cx_from_T_bc_T_bx(
            T_bc_image_center_mm,
            T_bx_image_center_mm,
        )
        T_xc_image_center_m = invert_transform(T_cx_image_center_m)

        H_image_center, d_x_image_center = compute_H_xc_from_T_xc(
            data=data,
            T_xc_m=T_xc_image_center_m,
            d_x_mm=None,
            K_xray=K_xray_image_center,
        )

        overlay_image_center_normal_flip = make_overlay_with_explicit_mask(
            camera_bgr=data.camera_bgr,
            xray_gray_u8=xray_flip_image_center,
            xray_fov_mask=mask_flip_image_center,
            H_xc=H_image_center,  # IMPORTANT: same H, no flip/padding correction
            alpha=data.alpha,
        )

        print(f"selected RGB candidate   = {rgb_idx_image_center}")
        print(f"selected X-ray candidate = {xray_idx_image_center}")
        print("flip mode                = cv2.flip(image, 1)")
        print("H usage                  = SAME H_image_center, no modification")

        print("\nT_bc_image_center [mm] =")
        print(format_matrix(T_bc_image_center_mm))

        print("\nT_bx_image_center [mm] =")
        print(format_matrix(T_bx_image_center_mm))

        print("\nT_cx_image_center [m] =")
        print(format_matrix(T_cx_image_center_m))

        print("\nT_xc_image_center [m] =")
        print(format_matrix(T_xc_image_center_m))

        print(f"\nd_x_image_center [mm] = {d_x_image_center:+.6f}")

        print("\nH_image_center =")
        print(format_matrix(H_image_center))

        show_overlay(
            windows=windows,
            title=(
                f"{base_name} | 10 IMAGE-CENTER FX=FY K | Handeye selected | "
                f"normal FLIP_LR | same H"
            ),
            overlay_bgr=overlay_image_center_normal_flip,
        )

        app._overlay_windows = windows
        return app.exec()

    except Exception as e:
        QMessageBox.critical(None, "debug_xray_image_flipping", str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())