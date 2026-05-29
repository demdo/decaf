# -*- coding: utf-8 -*-
"""
debug_ippe_tl_bl_check.py

Laedt:
    - eine XYZ-NPZ      (enthaelt points_xyz_camera_filt)
    - eine UV-NPZ       (enthaelt uv_xray oder points_uv)
    - eine K_xray-NPZ   (enthaelt K_xray / K / Kx)
    - ein X-ray-Bild als .bmp

Dann:
    1) baut das kanonische Board-KOS auf
    2) berechnet beide IPPE-Kandidaten fuer T_bx
    3) nimmt die Boardpunkte
           TL = idx 0
           BL = idx 110
    4) transformiert sie mit beiden Kandidaten ins X-ray-Frame
    5) projiziert sie mit K_xray ins echte X-ray-Bild
    6) zeichnet sie direkt ins echte X-ray-Bild

Visualisierung
--------------
TL:
    - Kandidat 0 -> gruenes Kreuz
    - Kandidat 1 -> gruener Kreis

BL:
    - Kandidat 0 -> rotes Kreuz
    - Kandidat 1 -> roter Kreis
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.homography import build_board_xyz_canonical
from tracking.pose_solvers import solve_pose


# ============================================================
# Qt
# ============================================================

def _qt() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _pick_one_file(title: str, flt: str) -> Path | None:
    _qt()
    path, _ = QFileDialog.getOpenFileName(None, title, "", flt)
    return Path(path) if path else None


# ============================================================
# Loading
# ============================================================

def _load_xyz_npz(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as z:
        keys = list(z.files)

        if "points_xyz_camera_filt" not in keys:
            raise KeyError(
                "Missing key 'points_xyz_camera_filt' in XYZ NPZ.\n"
                f"Available keys: {keys}"
            )

        xyz = np.asarray(z["points_xyz_camera_filt"], dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(
                f"'points_xyz_camera_filt' has invalid shape {xyz.shape}, expected (N,3)."
            )

        print(f"[INFO] {path.name}: Using XYZ key 'points_xyz_camera_filt'")
        return xyz * 1000.0  # m -> mm


def _load_uv_npz(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as z:
        keys = list(z.files)

        if "uv_xray" in keys:
            uv = np.asarray(z["uv_xray"], dtype=np.float64)
            if uv.ndim == 2 and uv.shape[1] == 2:
                print(f"[INFO] {path.name}: Using UV key 'uv_xray'")
                return uv

        if "points_uv" in keys:
            uv = np.asarray(z["points_uv"], dtype=np.float64)
            if uv.ndim == 2 and uv.shape[1] == 2:
                print(f"[INFO] {path.name}: Using legacy UV key 'points_uv'")
                return uv

        for k in keys:
            arr = np.asarray(z[k], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 2:
                print(f"[INFO] {path.name}: Using fallback UV key '{k}'")
                return arr

        raise KeyError(
            "UV NPZ must contain one (N,2) array, e.g. 'uv_xray' or 'points_uv'.\n"
            f"Available keys: {keys}"
        )


def _load_kxray_npz(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as z:
        keys = list(z.files)

        candidate_keys = ["K_xray", "K", "Kx"]
        for key in candidate_keys:
            if key in keys:
                K = np.asarray(z[key], dtype=np.float64)
                if K.shape == (3, 3):
                    print(f"[INFO] {path.name}: Using K key '{key}'")
                    return K

        # fallback: first 3x3 array
        for key in keys:
            arr = np.asarray(z[key], dtype=np.float64)
            if arr.shape == (3, 3):
                print(f"[INFO] {path.name}: Using fallback K key '{key}'")
                return arr

        raise KeyError(
            "K NPZ must contain a 3x3 matrix, e.g. 'K_xray', 'K', or 'Kx'.\n"
            f"Available keys: {keys}"
        )


def _load_bmp_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read BMP image: {path}")

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 3:
        return img

    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    raise ValueError(f"Unsupported BMP image shape: {img.shape}")


# ============================================================
# Helpers
# ============================================================

def _rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = tvec
    return T


def _transform_points(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    pts_xyz = np.asarray(pts_xyz, dtype=np.float64).reshape(-1, 3)
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ pts_xyz.T).T + t[None, :]


def _project_points_xray(K: np.ndarray, pts_xray_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_xray_xyz, dtype=np.float64).reshape(-1, 3)

    z = pts[:, 2:3]
    if np.any(z <= 0):
        raise ValueError("At least one point has z <= 0 in X-ray frame; projection invalid.")

    uvw = (K @ pts.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv


def _print_T(name: str, T: np.ndarray) -> None:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    print(f"\n{name} =")
    for row in T:
        print("[" + " ".join(f"{v:10.4f}" for v in row) + "]")


def _draw_cross(img: np.ndarray, uv: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    u = int(round(float(uv[0])))
    v = int(round(float(uv[1])))

    size = 10
    cv2.line(img, (u - size, v - size), (u + size, v + size), color, 2, cv2.LINE_AA)
    cv2.line(img, (u - size, v + size), (u + size, v - size), color, 2, cv2.LINE_AA)

    cv2.putText(
        img,
        label,
        (u + 10, v - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_circle(img: np.ndarray, uv: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    u = int(round(float(uv[0])))
    v = int(round(float(uv[1])))

    cv2.circle(img, (u, v), 8, color, 2, lineType=cv2.LINE_AA)

    cv2.putText(
        img,
        label,
        (u + 10, v - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


# ============================================================
# IPPE
# ============================================================

def _solve_ippe_candidates(
    board_xyz: np.ndarray,
    uv_xray: np.ndarray,
    K_xray: np.ndarray,
) -> tuple[dict, dict]:
    res = solve_pose(
        object_points_xyz=board_xyz,
        image_points_uv=uv_xray,
        K=K_xray,
        pose_method="ippe",
        refine_with_iterative=False,
    )

    if res.all_candidates is None or len(res.all_candidates) < 2:
        raise RuntimeError(
            "IPPE did not return both candidates. Check pose_solvers.py / all_candidates."
        )

    cand_map: dict[int, dict] = {}

    for c in res.all_candidates:
        rvec = np.asarray(c.rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(c.tvec, dtype=np.float64).reshape(3)
        R = np.asarray(c.R, dtype=np.float64).reshape(3, 3)

        cand_map[int(c.candidate_index)] = {
            "candidate_index": int(c.candidate_index),
            "rvec": rvec,
            "tvec": tvec,
            "R": R,
            "T_bx": _rvec_tvec_to_T(rvec, tvec),
            "reproj_mean_px": float(c.reproj_mean_px),
            "reproj_median_px": float(c.reproj_median_px),
            "reproj_max_px": float(c.reproj_max_px),
        }

    if 0 not in cand_map or 1 not in cand_map:
        raise RuntimeError(
            f"Expected candidate_index 0 and 1, got: {sorted(cand_map.keys())}"
        )

    return cand_map[0], cand_map[1]


# ============================================================
# Main
# ============================================================

def main() -> None:
    xyz_path = _pick_one_file("XYZ NPZ auswählen", "NumPy NPZ (*.npz)")
    if xyz_path is None:
        print("[INFO] No XYZ NPZ selected.")
        return

    uv_path = _pick_one_file("UV NPZ auswählen", "NumPy NPZ (*.npz)")
    if uv_path is None:
        print("[INFO] No UV NPZ selected.")
        return

    kx_path = _pick_one_file("K_xray NPZ auswählen", "NumPy NPZ (*.npz)")
    if kx_path is None:
        print("[INFO] No K_xray NPZ selected.")
        return

    bmp_path = _pick_one_file("X-ray BMP auswählen", "Bitmap (*.bmp)")
    if bmp_path is None:
        print("[INFO] No BMP selected.")
        return

    print("\n" + "=" * 70)
    print("SELECT FILES")
    print("=" * 70)
    print(f"XYZ: {xyz_path}")
    print(f"UV : {uv_path}")
    print(f"K  : {kx_path}")
    print(f"BMP: {bmp_path}")

    xyz_mm = _load_xyz_npz(xyz_path)
    uv_xray = _load_uv_npz(uv_path)
    K_xray = _load_kxray_npz(kx_path)
    xray_img = _load_bmp_image(bmp_path)

    board_xyz = build_board_xyz_canonical(
        nu=10,
        nv=10,
        pitch_mm=2.54,
    )
    n_board = board_xyz.shape[0]

    if xyz_mm.shape[0] != n_board:
        raise ValueError(f"XYZ has {xyz_mm.shape[0]} points, expected {n_board}.")
    if uv_xray.shape[0] != n_board:
        raise ValueError(f"UV has {uv_xray.shape[0]} points, expected {n_board}.")

    print("\n" + "=" * 70)
    print("INPUT INFO")
    print("=" * 70)
    print(f"Nominal board points : {n_board}")
    print(f"XYZ shape            : {xyz_mm.shape}")
    print(f"UV shape             : {uv_xray.shape}")
    print(f"K shape              : {K_xray.shape}")
    print(f"BMP shape            : {xray_img.shape}")

    cand0, cand1 = _solve_ippe_candidates(board_xyz, uv_xray, K_xray)

    idx_tl = 0
    idx_bl = 110

    p_tl_b = board_xyz[idx_tl]
    p_bl_b = board_xyz[idx_bl]
    pts_ref_b = np.vstack([p_tl_b, p_bl_b])

    print("\n" + "=" * 70)
    print("REFERENCE BOARD POINTS")
    print("=" * 70)
    print(f"TL idx = {idx_tl}   xyz_board = {p_tl_b}")
    print(f"BL idx = {idx_bl}   xyz_board = {p_bl_b}")

    T_bx_0 = cand0["T_bx"]
    pts_x_0 = _transform_points(T_bx_0, pts_ref_b)
    uv_proj_0 = _project_points_xray(K_xray, pts_x_0)

    T_bx_1 = cand1["T_bx"]
    pts_x_1 = _transform_points(T_bx_1, pts_ref_b)
    uv_proj_1 = _project_points_xray(K_xray, pts_x_1)

    print("\n" + "=" * 70)
    print("RESULTS FOR CANDIDATE 0")
    print("=" * 70)
    _print_T("T_bx_0", T_bx_0)
    print(f"TL proj uv         = {uv_proj_0[0]}")
    print(f"BL proj uv         = {uv_proj_0[1]}")
    print(f"u_TL - u_BL        = {uv_proj_0[0,0] - uv_proj_0[1,0]:+.4f}")
    print(f"reproj mean [px]   = {cand0['reproj_mean_px']:.4f}")

    print("\n" + "=" * 70)
    print("RESULTS FOR CANDIDATE 1")
    print("=" * 70)
    _print_T("T_bx_1", T_bx_1)
    print(f"TL proj uv         = {uv_proj_1[0]}")
    print(f"BL proj uv         = {uv_proj_1[1]}")
    print(f"u_TL - u_BL        = {uv_proj_1[0,0] - uv_proj_1[1,0]:+.4f}")
    print(f"reproj mean [px]   = {cand1['reproj_mean_px']:.4f}")

    vis = xray_img.copy()

    for uv in np.asarray(uv_xray, dtype=np.float64):
        u = int(round(float(uv[0])))
        v = int(round(float(uv[1])))
        if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
            cv2.circle(vis, (u, v), 2, (120, 120, 120), -1, lineType=cv2.LINE_AA)

    _draw_cross(vis, uv_proj_0[0], (0, 255, 0), "TL_0")
    _draw_cross(vis, uv_proj_0[1], (0, 0, 255), "BL_0")

    _draw_circle(vis, uv_proj_1[0], (0, 255, 0), "TL_1")
    _draw_circle(vis, uv_proj_1[1], (0, 0, 255), "BL_1")

    info_lines = [
        "Gray = detected UV points",
        "TL: green   |   BL: red",
        "Candidate 0: cross",
        "Candidate 1: circle",
        "q / ESC = quit",
    ]
    for i, txt in enumerate(info_lines):
        cv2.putText(
            vis,
            txt,
            (20, 30 + 28 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    win = "debug_ippe_tl_bl_check"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, vis)

    print("\n[INFO] Close image window with q / ESC.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()