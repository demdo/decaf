from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.homography import build_board_xyz_canonical
from tracking.transforms import (
    rvec_tvec_to_transform,
    invert_transform,
)

from tracking.pose_solvers import (
    normalize_dist_coeffs,
    _build_ippe_candidates,
    _select_ippe_candidate,
    _select_ippe_candidate_rgb,
    _rigid_fit_kabsch,
    _trust_region_from_depth_rms,
)


# ============================================================
# Config
# ============================================================

PITCH_MM = 2.54
STEPS_PER_EDGE = 10

REFINE_RGB_ITERATIVE = True
REFINE_XRAY_ITERATIVE = True


# ============================================================
# Qt file selection
# ============================================================

def select_npz_file(title: str) -> Path | None:
    app = QApplication.instance()
    owns_app = app is None

    if app is None:
        app = QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        "Overlay files (*.npz)",
    )

    if owns_app:
        app.quit()

    if not path:
        return None

    return Path(path)


# ============================================================
# Helpers
# ============================================================

def make_board_points() -> np.ndarray:
    return build_board_xyz_canonical(
        nu=STEPS_PER_EDGE,
        nv=STEPS_PER_EDGE,
        pitch_mm=PITCH_MM,
    )


def make_rgb_grid_from_3_corners(corners_uv: np.ndarray) -> np.ndarray:
    corners_uv = np.asarray(corners_uv, dtype=np.float64).reshape(3, 2)

    p_tl, p_tr, p_bl = corners_uv

    step_x = (p_tr - p_tl) / float(STEPS_PER_EDGE)
    step_y = (p_bl - p_tl) / float(STEPS_PER_EDGE)

    return np.array(
        [
            p_tl + alpha * step_x + beta * step_y
            for beta in range(STEPS_PER_EDGE + 1)
            for alpha in range(STEPS_PER_EDGE + 1)
        ],
        dtype=np.float64,
    )


def refine_pose_iterative(
    obj: np.ndarray,
    img: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ok, rvec_ref, tvec_ref = cv2.solvePnP(
        obj,
        img,
        K,
        dist,
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not ok:
        raise RuntimeError("Refinement failed.")

    return (
        np.asarray(rvec_ref, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec_ref, dtype=np.float64).reshape(3, 1),
    )


def rotation_delta_deg(T1: np.ndarray, T2: np.ndarray) -> float:
    R = T2[:3, :3] @ T1[:3, :3].T
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def z_angle(T: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(T[2, 2]), -1.0, 1.0))))


def print_critical_terms_m(label: str, T1: np.ndarray, T2: np.ndarray) -> None:
    dz = T2[2, 3] - T1[2, 3]

    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    print(f"Pose 1 t_z [m]  = {T1[2, 3]: .12e}")
    print(f"Pose 2 t_z [m]  = {T2[2, 3]: .12e}")
    print(f"delta t_z [m]   = {dz: .12e}")
    print(f"delta t_z [mm]  = {dz * 1000.0: .12e}")

    print()
    print(f"Pose 1 R[2,2]   = {T1[2, 2]: .12e}")
    print(f"Pose 2 R[2,2]   = {T2[2, 2]: .12e}")
    print(f"delta R[2,2]    = {(T2[2, 2] - T1[2, 2]): .12e}")

    print()
    print(f"Pose 1 angle    = {z_angle(T1): .12e}")
    print(f"Pose 2 angle    = {z_angle(T2): .12e}")
    print(f"delta angle     = {(z_angle(T2) - z_angle(T1)): .12e}")

    print()
    print(f"full dR [deg]   = {rotation_delta_deg(T1, T2): .12e}")


def print_pose(label: str, T: np.ndarray) -> None:
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    print(T)


def compose_T_cx(T_bx_mm: np.ndarray, T_bc_mm: np.ndarray) -> np.ndarray:
    T_cx_mm = T_bx_mm @ invert_transform(T_bc_mm)

    T_cx_m = T_cx_mm.copy()
    T_cx_m[:3, 3] *= 1e-3

    return T_cx_m


def load_stored_T_cx(data: np.lib.npyio.NpzFile, file_label: str) -> np.ndarray | None:
    print("\n" + "=" * 70)
    print(f"CHECK STORED POSE IN {file_label}")
    print("=" * 70)

    print("Available keys:")
    for key in data.files:
        print(f"  - {key}")

    if "T_cx" in data.files:
        T = np.asarray(data["T_cx"], dtype=np.float64).reshape(4, 4)
        print("\nLoaded stored T_cx directly.")
        return T

    if "T_xc" in data.files:
        T_xc = np.asarray(data["T_xc"], dtype=np.float64).reshape(4, 4)
        print("\nLoaded stored T_xc and inverted to T_cx.")
        return invert_transform(T_xc)

    print("\nNo stored T_cx or T_xc found.")
    return None


# ============================================================
# RGB side
# ============================================================

def compute_rgb(data: np.lib.npyio.NpzFile) -> dict:
    pts3d = make_board_points()

    cam_xyz = np.asarray(
        data["xray_points_xyz_c"],
        dtype=np.float64,
    ).reshape(-1, 3) * 1000.0

    pts2d = make_rgb_grid_from_3_corners(data["checkerboard_corners_uv"])

    K = np.asarray(data["K_rgb"], dtype=np.float64).reshape(3, 3)

    dist = normalize_dist_coeffs(
        data["dist_rgb"] if "dist_rgb" in data.files else None
    )

    depth = _rigid_fit_kabsch(
        board_xyz_mm=pts3d,
        cam_xyz_mm=cam_xyz,
    )

    T_depth = depth["T"]
    tr = _trust_region_from_depth_rms(depth["rms_mm"])

    cands = _build_ippe_candidates(
        object_points_xyz=pts3d,
        image_points_uv=pts2d,
        K=K,
        dist=dist,
    )

    idx = _select_ippe_candidate_rgb(
        cands,
        T_bc_depth=T_depth,
        gamma_t_mm=tr["gamma_t_mm"],
        gamma_r_deg=tr["gamma_r_deg"],
    )

    rvec = cands[idx].rvec
    tvec = cands[idx].tvec

    if REFINE_RGB_ITERATIVE:
        rvec, tvec = refine_pose_iterative(
            obj=pts3d,
            img=pts2d,
            K=K,
            dist=dist,
            rvec=rvec,
            tvec=tvec,
        )

    T_bc = rvec_tvec_to_transform(rvec, tvec)

    return {
        "T_bc": T_bc,
        "T_bc_depth": T_depth,
        "candidate": int(idx),
        "depth_fit": depth,
    }


# ============================================================
# X-ray side
# ============================================================

def compute_xray(data: np.lib.npyio.NpzFile) -> dict:
    pts3d = make_board_points()

    pts2d = np.asarray(
        data["xray_points_uv"],
        dtype=np.float64,
    ).reshape(-1, 2)

    K = np.asarray(data["K_xray"], dtype=np.float64).reshape(3, 3)
    dist = normalize_dist_coeffs(None)

    cands = _build_ippe_candidates(
        object_points_xyz=pts3d,
        image_points_uv=pts2d,
        K=K,
        dist=dist,
    )

    idx = _select_ippe_candidate(
        cands,
        use_xray_ippe_selection_rule=True,
    )

    rvec = cands[idx].rvec
    tvec = cands[idx].tvec

    if REFINE_XRAY_ITERATIVE:
        rvec, tvec = refine_pose_iterative(
            obj=pts3d,
            img=pts2d,
            K=K,
            dist=dist,
            rvec=rvec,
            tvec=tvec,
        )

    T_bx = rvec_tvec_to_transform(rvec, tvec)

    return {
        "T_bx": T_bx,
        "candidate": int(idx),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    p1 = select_npz_file("Select FIRST overlay file")
    if p1 is None:
        print("No first file selected.")
        return

    p2 = select_npz_file("Select SECOND overlay file")
    if p2 is None:
        print("No second file selected.")
        return

    print("\n" + "=" * 70)
    print("FILES")
    print("=" * 70)
    print("File 1:", p1)
    print("File 2:", p2)
    print(f"REFINE_RGB_ITERATIVE  = {REFINE_RGB_ITERATIVE}")
    print(f"REFINE_XRAY_ITERATIVE = {REFINE_XRAY_ITERATIVE}")

    d1 = np.load(p1, allow_pickle=True)
    d2 = np.load(p2, allow_pickle=True)

    rgb1 = compute_rgb(d1)
    rgb2 = compute_rgb(d2)

    xr1 = compute_xray(d1)
    xr2 = compute_xray(d2)

    T1_cx = compose_T_cx(
        T_bx_mm=xr1["T_bx"],
        T_bc_mm=rgb1["T_bc"],
    )

    T2_cx = compose_T_cx(
        T_bx_mm=xr2["T_bx"],
        T_bc_mm=rgb2["T_bc"],
    )

    print("\n" + "=" * 70)
    print("SELECTED CANDIDATES")
    print("=" * 70)
    print(f"File 1 RGB  candidate = {rgb1['candidate']}")
    print(f"File 2 RGB  candidate = {rgb2['candidate']}")
    print(f"File 1 Xray candidate = {xr1['candidate']}")
    print(f"File 2 Xray candidate = {xr2['candidate']}")

    print_pose("RECOMPUTED NORMAL IPPE-HANDEYE T_cx FILE 1 [m]", T1_cx)
    print_pose("RECOMPUTED NORMAL IPPE-HANDEYE T_cx FILE 2 [m]", T2_cx)

    print_critical_terms_m(
        "RECOMPUTED NORMAL IPPE-HANDEYE T_cx DRIFT FILE 2 vs FILE 1",
        T1_cx,
        T2_cx,
    )

    T1_cx_file = load_stored_T_cx(d1, "FILE 1")
    T2_cx_file = load_stored_T_cx(d2, "FILE 2")

    if T1_cx_file is not None:
        print_pose("STORED T_cx FILE 1 [m]", T1_cx_file)

    if T2_cx_file is not None:
        print_pose("STORED T_cx FILE 2 [m]", T2_cx_file)

    if T1_cx_file is not None and T2_cx_file is not None:
        print_critical_terms_m(
            "STORED T_cx DRIFT FILE 2 vs FILE 1",
            T1_cx_file,
            T2_cx_file,
        )

    if T1_cx_file is not None:
        print_critical_terms_m(
            "FILE 1: STORED T_cx vs RECOMPUTED T_cx",
            T1_cx_file,
            T1_cx,
        )

    if T2_cx_file is not None:
        print_critical_terms_m(
            "FILE 2: STORED T_cx vs RECOMPUTED T_cx",
            T2_cx_file,
            T2_cx,
        )


if __name__ == "__main__":
    main()