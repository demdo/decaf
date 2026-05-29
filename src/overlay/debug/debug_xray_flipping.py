# -*- coding: utf-8 -*-
"""
debug_xray_flipping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from tracking.pose_solvers import solve_pose
from overlay.tools.homography import build_board_xyz_canonical


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
        "Select overlay preview / overlay problem NPZ",
        "",
        "NPZ files (*.npz);;All files (*.*)",
    )
    return Path(path) if path else None


def pick_intrinsics_npz_file(title: str = "Select X-ray intrinsics NPZ") -> Path | None:
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


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def make_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def format_matrix(M: np.ndarray, decimals: int = 6) -> str:
    rows = []
    for row in np.asarray(M):
        rows.append("[" + "  ".join(f"{v:+.{decimals}f}" for v in row) + "]")
    return "\n".join(rows)


def _safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)

    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def unflip_uv_order_rows(uv: np.ndarray) -> np.ndarray:
    uv = _as_uv(uv, "uv")
    n = uv.shape[0]
    side = int(round(np.sqrt(n)))

    if side * side != n:
        raise ValueError(f"Cannot infer square grid from N={n}.")

    return uv.reshape(side, side, 2)[:, ::-1, :].reshape(-1, 2)


def xy_flip_uv_order_from_raw(uv_saved: np.ndarray) -> np.ndarray:
    uv_saved = _as_uv(uv_saved, "uv_saved")
    n = uv_saved.shape[0]
    side = int(round(np.sqrt(n)))

    if side * side != n:
        raise ValueError(f"Cannot infer square grid from N={n}.")

    raw = uv_saved.reshape(side, side, 2)[:, ::-1, :]
    xy_flipped = raw[::-1, ::-1, :]

    return xy_flipped.reshape(-1, 2)


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
            "xray_points_xyz_c",
            "xray_points_uv",
            "checkerboard_corners_uv",
            "T_tc",
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
            raise ValueError(
                "Overlay NPZ must contain either 'snapshot_rgb_with_tip_bgr' or 'snapshot_rgb_bgr'."
            )

        self.xray_gray_u8 = np.asarray(data["xray_gray_u8"])
        if self.xray_gray_u8.ndim != 2:
            raise ValueError(
                f"xray_gray_u8 must be grayscale, got shape {self.xray_gray_u8.shape}"
            )
        if self.xray_gray_u8.dtype != np.uint8:
            self.xray_gray_u8 = np.clip(self.xray_gray_u8, 0, 255).astype(np.uint8)

        self.K_rgb = _as_mat33(data["K_rgb"], "K_rgb")
        self.points_xyz_c_m = _as_xyz(data["xray_points_xyz_c"], "xray_points_xyz_c")
        self.points_uv_x_saved = _as_uv(data["xray_points_uv"], "xray_points_uv")
        self.checkerboard_corners_uv = _as_uv(
            data["checkerboard_corners_uv"],
            "checkerboard_corners_uv",
        )
        self.T_tc_mm = _as_transform44(data["T_tc"], "T_tc")


def load_intrinsics_npz(npz_path: Path) -> np.ndarray:
    data = np.load(str(npz_path), allow_pickle=True)
    keys = set(data.files)

    for key in ("K_xray", "K_x", "K"):
        if key in keys:
            return _as_mat33(data[key], f"{npz_path.name}:{key}")

    raise ValueError(
        f"{npz_path} does not contain any of the expected keys "
        f"('K_xray', 'K_x', 'K'). Available keys: {sorted(keys)}"
    )


# ============================================================
# Geometry helpers
# ============================================================

def compute_T_cx_from_T_bc_T_bx(T_bc_mm: np.ndarray, T_bx_mm: np.ndarray) -> np.ndarray:
    T_cx_mm = np.asarray(T_bx_mm, dtype=np.float64) @ invert_transform(T_bc_mm)
    T_cx_m = T_cx_mm.copy()
    T_cx_m[:3, 3] *= 1e-3
    return T_cx_m


def _get_uv_for_case(case_name: str, overlay_data: OverlayData) -> np.ndarray:
    if case_name == "SAVED_FLIPPED_UV":
        return overlay_data.points_uv_x_saved

    if case_name == "RAW_UNFLIPPED_UV":
        return unflip_uv_order_rows(overlay_data.points_uv_x_saved)

    if case_name == "RAW_XY_FLIPPED_UV":
        return xy_flip_uv_order_from_raw(overlay_data.points_uv_x_saved)

    raise ValueError(f"Unknown case: {case_name}")


def build_rgb_uv_from_three_anchors(
    checkerboard_corners_uv: np.ndarray,
    *,
    nu: int = 10,
    nv: int = 10,
) -> np.ndarray:
    anchors = _as_uv(checkerboard_corners_uv, "checkerboard_corners_uv")

    if anchors.shape[0] != 3:
        raise ValueError(
            f"Expected exactly 3 RGB reference points, got {anchors.shape[0]}"
        )

    p_tl = anchors[0]
    p_tr = anchors[1]
    p_bl = anchors[2]

    step_u = (p_tr - p_tl) / float(nu)
    step_v = (p_bl - p_tl) / float(nv)

    pts = []
    for i in range(nv + 1):
        for j in range(nu + 1):
            pts.append(p_tl + j * step_u + i * step_v)

    return np.asarray(pts, dtype=np.float64)


def save_rgb_board_correspondences_txt(
    *,
    out_dir: Path,
    overlay_data: OverlayData,
    res_handeye,
    nu: int = 10,
    nv: int = 10,
    pitch_mm: float = 2.54,
) -> None:
    if res_handeye.candidate_index_rgb is None:
        rgb_idx = -1
    else:
        rgb_idx = int(res_handeye.candidate_index_rgb)

    board_xyz_mm = build_board_xyz_canonical(
        nu=nu,
        nv=nv,
        pitch_mm=pitch_mm,
    )

    rgb_uv = build_rgb_uv_from_three_anchors(
        overlay_data.checkerboard_corners_uv,
        nu=nu,
        nv=nv,
    )

    if board_xyz_mm.shape[0] != rgb_uv.shape[0]:
        raise ValueError(
            f"Mismatch: board_xyz={board_xyz_mm.shape}, rgb_uv={rgb_uv.shape}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"debug_rgb_board_correspondences_selected_rgb{rgb_idx}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# RGB board correspondences used for T_bc / RGB IPPE\n")
        f.write(f"# selected_rgb_candidate = {rgb_idx}\n")
        f.write(f"# nu = {nu}\n")
        f.write(f"# nv = {nv}\n")
        f.write(f"# pitch_mm = {pitch_mm:.6f}\n")
        f.write("# board_xyz_mm generated by build_board_xyz_canonical(...)\n")
        f.write("# rgb_uv_px interpolated from checkerboard_corners_uv = [TL, TR, BL]\n")
        f.write("#\n")
        f.write("# idx  i  j  board_x_mm  board_y_mm  board_z_mm  rgb_u_px  rgb_v_px\n")

        ncols = nu + 1

        for k, (p_b, uv) in enumerate(zip(board_xyz_mm, rgb_uv)):
            i = k // ncols
            j = k % ncols

            f.write(
                f"{k:03d}  "
                f"{i:02d}  {j:02d}  "
                f"{p_b[0]:+.6f}  {p_b[1]:+.6f}  {p_b[2]:+.6f}  "
                f"{uv[0]:+.6f}  {uv[1]:+.6f}\n"
            )

    print(f"[SAVE] RGB board correspondences written to: {out_path}")


# ============================================================
# T_bx candidates only
# ============================================================

def print_T_bx_candidates_only(
    *,
    label: str,
    K_xray: np.ndarray,
    points_uv_x: np.ndarray,
) -> None:
    board_xyz_mm = build_board_xyz_canonical(
        nu=10,
        nv=10,
        pitch_mm=2.54,
    )

    points_uv_x = _as_uv(points_uv_x, "points_uv_x")

    if board_xyz_mm.shape[0] != points_uv_x.shape[0]:
        raise ValueError(
            f"{label}: mismatch board_xyz={board_xyz_mm.shape[0]}, uv={points_uv_x.shape[0]}"
        )

    res = solve_pose(
        object_points_xyz=board_xyz_mm,
        image_points_uv=points_uv_x,
        K=K_xray,
        dist_coeffs=None,
        pose_method="ippe",
        refine_with_iterative=False,
        use_xray_ippe_selection_rule=False,
    )

    if res.all_candidates is None or len(res.all_candidates) != 2:
        raise RuntimeError(f"{label}: Expected exactly 2 X-ray IPPE candidates.")

    print("\n" + "=" * 120)
    print(f"T_bx CANDIDATES ONLY | {label}")
    print("=" * 120)

    for idx, cand in enumerate(res.all_candidates):
        T_bx_mm = make_transform(cand.rvec, cand.tvec)
        T_xb_mm = invert_transform(T_bx_mm)

        z_b_in_x = T_bx_mm[:3, 2]
        z_x = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        angle_to_zx = angle_between_deg(z_b_in_x, z_x)
        angle_to_minus_zx = angle_between_deg(z_b_in_x, -z_x)

        print("\n" + "-" * 120)
        print(f"{label} | XRAY candidate {idx}")
        print("-" * 120)
        print(f"reproj mean [px]        = {cand.reproj_mean_px:.6f}")
        print(f"reproj median [px]      = {cand.reproj_median_px:.6f}")
        print(f"reproj max [px]         = {cand.reproj_max_px:.6f}")
        print(f"angle z_b to  z_x [deg] = {angle_to_zx:.6f}")
        print(f"angle z_b to -z_x [deg] = {angle_to_minus_zx:.6f}")

        print("\nT_bx [mm] =")
        print(format_matrix(T_bx_mm))

        print("\nT_xb [mm] =")
        print(format_matrix(T_xb_mm))


# ============================================================
# Main debug case
# ============================================================

def print_pose_case(
    case_name: str,
    overlay_data: OverlayData,
    K_xray: np.ndarray,
) -> None:
    print("\n" + "=" * 120)
    print(f"IPPE HANDEYE CASE: {case_name}")
    print("=" * 120)

    points_uv_x = _get_uv_for_case(case_name, overlay_data)

    print(f"N xyz camera points = {overlay_data.points_xyz_c_m.shape[0]}")
    print(f"N xray uv points    = {points_uv_x.shape[0]}")

    res_handeye = solve_pose(
        object_points_xyz=overlay_data.points_xyz_c_m,
        image_points_uv=points_uv_x,
        K=K_xray,
        dist_coeffs=None,
        dist_coeffs_rgb=None,
        pose_method="ippe_handeye",
        checkerboard_corners_uv=overlay_data.checkerboard_corners_uv,
        K_rgb=overlay_data.K_rgb,
        steps_per_edge=10,
        refine_with_iterative=False,
        refine_rgb_iterative=False,
        refine_xray_iterative=False,
    )

    if res_handeye.all_candidates_rgb is None or len(res_handeye.all_candidates_rgb) != 2:
        raise RuntimeError("Expected exactly 2 RGB IPPE candidates for handeye.")
    if res_handeye.all_candidates is None or len(res_handeye.all_candidates) != 2:
        raise RuntimeError("Expected exactly 2 X-ray IPPE candidates for handeye.")

    save_rgb_board_correspondences_txt(
        out_dir=overlay_data.npz_path.parent,
        overlay_data=overlay_data,
        res_handeye=res_handeye,
        nu=10,
        nv=10,
        pitch_mm=2.54,
    )

    print("\n" + "#" * 120)
    print(f"{case_name} | RGB CANDIDATES")
    print("#" * 120)

    for rgb_idx, rgb_cand in enumerate(res_handeye.all_candidates_rgb):
        T_bc_mm = make_transform(rgb_cand.rvec, rgb_cand.tvec)

        print(f"\nRGB candidate {rgb_idx}")
        print(f"reproj mean [px]    = {rgb_cand.reproj_mean_px:.6f}")
        print(f"reproj median [px]  = {rgb_cand.reproj_median_px:.6f}")
        print(f"reproj max [px]     = {rgb_cand.reproj_max_px:.6f}")
        print("T_bc [mm] =")
        print(format_matrix(T_bc_mm))

    print("\n" + "#" * 120)
    print(f"{case_name} | XRAY CANDIDATES")
    print("#" * 120)

    for xray_idx, xray_cand in enumerate(res_handeye.all_candidates):
        T_bx_mm = make_transform(xray_cand.rvec, xray_cand.tvec)
        T_xb_mm = invert_transform(T_bx_mm)

        z_b_in_x = T_bx_mm[:3, 2]
        z_x = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        angle_to_zx = angle_between_deg(z_b_in_x, z_x)
        angle_to_minus_zx = angle_between_deg(z_b_in_x, -z_x)

        print(f"\nXRAY candidate {xray_idx}")
        print(f"reproj mean [px]        = {xray_cand.reproj_mean_px:.6f}")
        print(f"reproj median [px]      = {xray_cand.reproj_median_px:.6f}")
        print(f"reproj max [px]         = {xray_cand.reproj_max_px:.6f}")
        print(f"angle z_b to  z_x [deg] = {angle_to_zx:.6f}")
        print(f"angle z_b to -z_x [deg] = {angle_to_minus_zx:.6f}")

        print("\nT_bx [mm] =")
        print(format_matrix(T_bx_mm))

        print("\nT_xb [mm] =")
        print(format_matrix(T_xb_mm))

    print("\n" + "#" * 120)
    print(f"{case_name} | ALL T_cx COMBINATIONS")
    print("#" * 120)

    for rgb_idx, rgb_cand in enumerate(res_handeye.all_candidates_rgb):
        T_bc_mm = make_transform(rgb_cand.rvec, rgb_cand.tvec)

        for xray_idx, xray_cand in enumerate(res_handeye.all_candidates):
            T_bx_mm = make_transform(xray_cand.rvec, xray_cand.tvec)
            T_xb_mm = invert_transform(T_bx_mm)

            T_cx_m = compute_T_cx_from_T_bc_T_bx(T_bc_mm, T_bx_mm)
            T_xc_m = invert_transform(T_cx_m)

            T_cx_mm = T_cx_m.copy()
            T_cx_mm[:3, 3] *= 1e3

            T_xc_mm = T_xc_m.copy()
            T_xc_mm[:3, 3] *= 1e3

            z_b_in_x = T_bx_mm[:3, 2]
            z_x = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            angle_to_zx = angle_between_deg(z_b_in_x, z_x)
            angle_to_minus_zx = angle_between_deg(z_b_in_x, -z_x)

            print("\n" + "-" * 120)
            print(f"{case_name} | HANDEYE RGB {rgb_idx} + XRAY {xray_idx}")
            print("-" * 120)

            print(f"RGB reproj mean [px]    = {rgb_cand.reproj_mean_px:.6f}")
            print(f"XRAY reproj mean [px]   = {xray_cand.reproj_mean_px:.6f}")
            print(f"angle z_b to  z_x [deg] = {angle_to_zx:.6f}")
            print(f"angle z_b to -z_x [deg] = {angle_to_minus_zx:.6f}")

            print("\nT_bc [mm] =")
            print(format_matrix(T_bc_mm))

            print("\nT_bx [mm] =")
            print(format_matrix(T_bx_mm))

            print("\nT_xb [mm] =")
            print(format_matrix(T_xb_mm))

            print("\nDerived T_cx [mm] =")
            print(format_matrix(T_cx_mm))

            print("\nDerived T_xc [mm] =")
            print(format_matrix(T_xc_mm))

            print("\nDerived T_cx [m] =")
            print(format_matrix(T_cx_m))

            print("\nDerived T_xc [m] =")
            print(format_matrix(T_xc_m))


# ============================================================
# Main
# ============================================================

def main() -> int:
    _ensure_qt_app()

    overlay_npz_path = pick_overlay_npz_file()
    if overlay_npz_path is None:
        return 0

    intrinsics_path = pick_intrinsics_npz_file("Select CURRENT / OLD X-ray intrinsics NPZ")
    if intrinsics_path is None:
        return 0

    try:
        overlay_data = OverlayData(overlay_npz_path)
        K_xray = load_intrinsics_npz(intrinsics_path)

        print("\n" + "=" * 120)
        print("DEBUG XRAY FLIPPING: IPPE HANDEYE")
        print("=" * 120)
        print(f"Overlay NPZ   : {overlay_npz_path}")
        print(f"Intrinsics NPZ: {intrinsics_path}")
        print(f"Intrinsics    : {_safe_name(intrinsics_path)}")

        print("\nK_xray =")
        print(format_matrix(K_xray))

        print_pose_case(
            case_name="SAVED_FLIPPED_UV",
            overlay_data=overlay_data,
            K_xray=K_xray,
        )

        print_pose_case(
            case_name="RAW_UNFLIPPED_UV",
            overlay_data=overlay_data,
            K_xray=K_xray,
        )

        print_pose_case(
            case_name="RAW_XY_FLIPPED_UV",
            overlay_data=overlay_data,
            K_xray=K_xray,
        )

        print("\n" + "=" * 120)
        print("LOAD NEW XRAY INTRINSICS FOR T_bx ONLY")
        print("=" * 120)

        new_intrinsics_path = pick_intrinsics_npz_file("Select NEW / MOD X-ray intrinsics NPZ")
        if new_intrinsics_path is not None:
            K_xray_new = load_intrinsics_npz(new_intrinsics_path)

            print(f"\nNew intrinsics NPZ: {new_intrinsics_path}")
            print("\nK_xray_new =")
            print(format_matrix(K_xray_new))

            uv_xray_saved = _get_uv_for_case(
                "SAVED_FLIPPED_UV",
                overlay_data,
            )

            uv_xray_raw = _get_uv_for_case(
                "RAW_UNFLIPPED_UV",
                overlay_data,
            )

            print_T_bx_candidates_only(
                label="NEW_INTRINSICS | SAVED_FLIPPED_UV",
                K_xray=K_xray_new,
                points_uv_x=uv_xray_saved,
            )

            print_T_bx_candidates_only(
                label="NEW_INTRINSICS | RAW_UNFLIPPED_UV",
                K_xray=K_xray_new,
                points_uv_x=uv_xray_raw,
            )
        else:
            print("[INFO] No new intrinsics selected. Skipping T_bx-only comparison.")

        print("\n[INFO] Fertig.")
        return 0

    except Exception as e:
        QMessageBox.critical(None, "debug_xray_flipping", str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())