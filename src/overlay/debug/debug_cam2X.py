# -*- coding: utf-8 -*-
"""
debug_cam2X_compare_multiview_v3.py

Compare FOUR camera-to-xray calibration routes over MULTIPLE views:

1) Direct RANSAC-PnP:
   xyz_c -> uv_xray with K_xray  =>  T_cx

2) Direct IPPE:
   xyz_c -> uv_xray with K_xray  =>  T_cx

3) Homography (all-depth):
   ALL 121 xyz_c points projected to rgb_uv via K_rgb,
   then homography route  =>  T_cx
   (current production method)

4) Homography (corner-interp):
   Only 3 corner xyz_c points (TL=xyz_c[0], TR=xyz_c[10], BL=xyz_c[110])
   projected to rgb_uv via K_rgb,
   then 121 points interpolated in 2D image space via interpolate_grid_uv,
   then homography route  =>  T_cx

   This isolates depth noise to 3 points only.
   The other 118 points are derived purely from 2D geometry.
   When real corner detection replaces the 3 projected corners,
   depth is eliminated entirely.

Important
---------
- The homography route returns translation in mm -> converted to meters here.
- Grid layout: row-major, TL=index 0, TR=index 10, BL=index 110.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.pose_solvers import solve_pose
from overlay.calib.calib_camera_to_xray_new import calibrate_camera_to_xray


# ============================================================
# Config
# ============================================================

PITCH_MM = 2.54
NROWS = 11
NCOLS = 11

REFINE_WITH_ITERATIVE = True

RANSAC_REPROJ_ERROR_PX = 3.0
RANSAC_CONFIDENCE = 0.99
RANSAC_ITERATIONS = 5000


# ============================================================
# Result container
# ============================================================

@dataclass(frozen=True)
class MethodPoseResult:
    T_cx: np.ndarray
    T_xc: np.ndarray
    reproj_mean_px: float | None
    reproj_median_px: float | None
    reproj_max_px: float | None
    n_inliers: int | None
    method_name: str


@dataclass(frozen=True)
class ViewCompareResult:
    name: str
    pnp_ransac: MethodPoseResult
    ippe: MethodPoseResult
    homography_all: MethodPoseResult
    homography_interp: MethodPoseResult


# ============================================================
# Qt helpers
# ============================================================

def _get_or_create_qapp() -> tuple[QApplication, bool]:
    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True
    return app, created


def _pick_open_files_qt(title: str, file_filter: str, start_dir: str = "") -> list[Path]:
    app, created = _get_or_create_qapp()
    paths, _ = QFileDialog.getOpenFileNames(None, title, start_dir, file_filter)
    if created:
        app.quit()
    return [Path(p.strip()) for p in paths if p.strip()]


def _pick_open_file_qt(title: str, file_filter: str, start_dir: str = "") -> Path | None:
    app, created = _get_or_create_qapp()
    path, _ = QFileDialog.getOpenFileName(None, title, start_dir, file_filter)
    if created:
        app.quit()
    path = path.strip()
    return Path(path) if path else None


# ============================================================
# Load helpers
# ============================================================

def _load_first_existing_array(path: Path, candidate_keys: list[str]) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as npz:
        for key in candidate_keys:
            if key in npz.files:
                arr = np.asarray(npz[key], dtype=np.float64)
                print(f"[INFO] Loaded '{key}' from {path.name} with shape {arr.shape}")
                return arr
        raise KeyError(
            f"{path.name}: none of the keys {candidate_keys} found. "
            f"Available keys: {list(npz.files)}"
        )


def _load_xyz_c(path: Path) -> np.ndarray:
    pts = _load_first_existing_array(
        path,
        ["points_xyz", "xyz_c", "xray_points_xyz_c", "points_xyz_camera", "points_xyz_c"],
    )
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{path.name}: expected shape (N,3), got {pts.shape}")
    return pts


def _load_image_points_uv(path: Path) -> np.ndarray:
    pts = _load_first_existing_array(
        path,
        ["points_uv", "uv", "rgb_points_uv", "xray_points_uv", "image_points_uv"],
    )
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"{path.name}: expected shape (N,2), got {pts.shape}")
    return pts


def _load_K(path: Path) -> np.ndarray:
    K = _load_first_existing_array(path, ["K", "K_rgb", "K_xray", "Kx"])
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"{path.name}: K must have shape (3,3), got {K.shape}")
    return K


# ============================================================
# Geometry helpers
# ============================================================

def _project_xyz_c_to_rgb_uv(xyz_c: np.ndarray, K_rgb: np.ndarray) -> np.ndarray:
    """Project camera-frame 3D points into the RGB image."""
    xyz_c = np.asarray(xyz_c, dtype=np.float64).reshape(-1, 3)
    K_rgb = np.asarray(K_rgb, dtype=np.float64).reshape(3, 3)

    z = xyz_c[:, 2]
    if np.any(z <= 1e-12):
        bad = np.flatnonzero(z <= 1e-12)
        raise ValueError(
            f"Cannot project xyz_c: z <= 0 at indices {bad[:10].tolist()}"
        )

    x = xyz_c[:, 0] / z
    y = xyz_c[:, 1] / z

    u = K_rgb[0, 0] * x + K_rgb[0, 1] * y + K_rgb[0, 2]
    v = K_rgb[1, 1] * y + K_rgb[1, 2]
    return np.column_stack([u, v]).astype(np.float64)


def interpolate_grid_uv(
    uv_TL: np.ndarray,
    uv_TR: np.ndarray,
    uv_BL: np.ndarray,
    nrows: int = 11,
    ncols: int = 11,
) -> np.ndarray:
    """
    Interpolate a full nrows x ncols grid in UV image space from three
    corner UV coordinates. No depth used.

    TL = grid[0, 0], TR = grid[0, ncols-1], BL = grid[nrows-1, 0].
    TL->TR spans (ncols-1) pitch steps, TL->BL spans (nrows-1) pitch steps.

    Returns (nrows*ncols, 2) float64, row-major.
    """
    uv_TL = np.asarray(uv_TL, dtype=np.float64)
    uv_TR = np.asarray(uv_TR, dtype=np.float64)
    uv_BL = np.asarray(uv_BL, dtype=np.float64)

    step_u = (uv_TR - uv_TL) / (ncols - 1)
    step_v = (uv_BL - uv_TL) / (nrows - 1)

    return np.array(
        [uv_TL + j * step_u + i * step_v for i in range(nrows) for j in range(ncols)],
        dtype=np.float64,
    )


def _rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def _invert_transform(T_ab: np.ndarray) -> np.ndarray:
    T_ab = np.asarray(T_ab, dtype=np.float64).reshape(4, 4)
    R, t = T_ab[:3, :3], T_ab[:3, 3]
    T_ba = np.eye(4, dtype=np.float64)
    T_ba[:3, :3] = R.T
    T_ba[:3, 3] = -R.T @ t
    return T_ba


def _rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = np.asarray(R_b, dtype=np.float64) @ np.asarray(R_a, dtype=np.float64).T
    val = float(np.clip(0.5 * (np.trace(R_rel) - 1.0), -1.0, 1.0))
    return float(np.degrees(np.arccos(val)))


def _compare_transforms(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float, np.ndarray]:
    T_a = np.asarray(T_a, dtype=np.float64).reshape(4, 4)
    T_b = np.asarray(T_b, dtype=np.float64).reshape(4, 4)
    rot_deg = _rotation_angle_deg(T_a[:3, :3], T_b[:3, :3])
    dt = T_b[:3, 3] - T_a[:3, 3]
    return rot_deg, float(np.linalg.norm(dt)), dt


# ============================================================
# Print helpers
# ============================================================

def _print_pose(name: str, T: np.ndarray) -> None:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print("R =")
    print(np.array2string(T[:3, :3], precision=6, suppress_small=True))
    print("\nt =")
    print(np.array2string(T[:3, 3], precision=6, suppress_small=True))
    print("\nT =")
    print(np.array2string(T, precision=6, suppress_small=True))


def _print_view_header(i: int, name: str) -> None:
    print("\n" + "#" * 70)
    print(f"VIEW {i + 1}: {name}")
    print("#" * 70)


def _print_method_summary(label: str, res: MethodPoseResult) -> None:
    print(f"\n--- {label} ---")
    if res.reproj_mean_px is not None:
        print(f"inliers              = {res.n_inliers}")
        print(f"reproj mean [px]     = {res.reproj_mean_px:.6f}")
        print(f"reproj median [px]   = {res.reproj_median_px:.6f}")
        print(f"reproj max [px]      = {res.reproj_max_px:.6f}")
    else:
        print("reprojection stats   = n/a")


def _print_pairwise_diff(label: str, rot_deg: float, dt_norm: float, dt: np.ndarray) -> None:
    print(f"\n--- {label} ---")
    print(f"rotation diff [deg]  = {rot_deg:.6f}")
    print(f"translation diff     = {np.array2string(dt, precision=6, suppress_small=True)}")
    print(f"translation norm     = {dt_norm:.6f}")


def _print_stability_block(name: str, Ts: list[np.ndarray]) -> None:
    if not Ts:
        return
    translations = np.array([np.asarray(T)[:3, 3] for T in Ts], dtype=np.float64)
    z_axes = np.array([np.asarray(T)[:3, 2] for T in Ts], dtype=np.float64)

    print("\n" + "=" * 70)
    print(f"STABILITY SUMMARY: {name}")
    print("=" * 70)
    print(f"num views                  = {len(Ts)}")
    print(f"translation mean           = {np.array2string(np.mean(translations, axis=0), precision=6, suppress_small=True)}")
    print(f"translation std            = {np.array2string(np.std(translations, axis=0), precision=6, suppress_small=True)}")
    print(f"translation norm std       = {float(np.std(np.linalg.norm(translations, axis=1))):.6f}")
    print(f"z-axis mean                = {np.array2string(np.mean(z_axes, axis=0), precision=6, suppress_small=True)}")
    print(f"z-axis std                 = {np.array2string(np.std(z_axes, axis=0), precision=6, suppress_small=True)}")

    if len(Ts) >= 2:
        T_ref = np.asarray(Ts[0], dtype=np.float64)
        print("\nRelative to first view:")
        for i, T in enumerate(Ts):
            rot_deg, dt_norm, dt = _compare_transforms(T_ref, T)
            print(
                f"  view {i + 1:2d}: "
                f"rot diff = {rot_deg:9.6f} deg, "
                f"trans diff norm = {dt_norm:9.6f}, "
                f"dt = {np.array2string(dt, precision=6, suppress_small=True)}"
            )


# ============================================================
# Core solvers
# ============================================================

def _run_direct_pose(
    xyz_c: np.ndarray,
    uv_xray: np.ndarray,
    K_xray: np.ndarray,
    *,
    pose_method: str,
) -> MethodPoseResult:
    pose = solve_pose(
        object_points_xyz=xyz_c,
        image_points_uv=uv_xray,
        K=K_xray,
        dist_coeffs=None,
        pose_method=pose_method,
        refine_with_iterative=REFINE_WITH_ITERATIVE,
        ransac_reprojection_error_px=RANSAC_REPROJ_ERROR_PX,
        ransac_confidence=RANSAC_CONFIDENCE,
        ransac_iterations_count=RANSAC_ITERATIONS,
    )
    T_cx = _rvec_tvec_to_transform(pose.rvec, pose.tvec)
    return MethodPoseResult(
        T_cx=T_cx,
        T_xc=_invert_transform(T_cx),
        reproj_mean_px=float(pose.reproj_mean_px),
        reproj_median_px=float(pose.reproj_median_px),
        reproj_max_px=float(pose.reproj_max_px),
        n_inliers=int(len(pose.inlier_idx)),
        method_name=pose_method,
    )


def _run_homography_pose(
    rgb_uv: np.ndarray,
    uv_xray: np.ndarray,
    K_rgb: np.ndarray,
    K_xray: np.ndarray,
    method_name: str,
) -> MethodPoseResult:
    result_h = calibrate_camera_to_xray(
        rgb_points_uv=rgb_uv,
        xray_points_uv=uv_xray,
        K_rgb=K_rgb,
        K_xray=K_xray,
        pitch_mm=PITCH_MM,
        nrows=NROWS,
        ncols=NCOLS,
    )
    T_cx_h = np.asarray(result_h.T_cx, dtype=np.float64).reshape(4, 4).copy()
    T_cx_h[:3, 3] *= 1e-3  # mm -> m
    return MethodPoseResult(
        T_cx=T_cx_h,
        T_xc=_invert_transform(T_cx_h),
        reproj_mean_px=None,
        reproj_median_px=None,
        reproj_max_px=None,
        n_inliers=None,
        method_name=method_name,
    )


def _run_single_view(
    name: str,
    xyz_c: np.ndarray,
    uv_xray: np.ndarray,
    K_rgb: np.ndarray,
    K_xray: np.ndarray,
) -> ViewCompareResult:
    xyz_c = np.asarray(xyz_c, dtype=np.float64).reshape(-1, 3)
    uv_xray = np.asarray(uv_xray, dtype=np.float64).reshape(-1, 2)

    if xyz_c.shape[0] != uv_xray.shape[0]:
        raise ValueError(
            f"{name}: xyz_c and uv_xray must have same number of points, "
            f"got {xyz_c.shape[0]} and {uv_xray.shape[0]}"
        )

    # --- Methods 1 + 2: direct pose ---
    pnp_ransac = _run_direct_pose(xyz_c, uv_xray, K_xray, pose_method="iterative_ransac")
    ippe       = _run_direct_pose(xyz_c, uv_xray, K_xray, pose_method="ippe")

    # --- Method 3: homography, all 121 points projected from xyz_c ---
    rgb_uv_all = _project_xyz_c_to_rgb_uv(xyz_c, K_rgb)
    homography_all = _run_homography_pose(
        rgb_uv_all, uv_xray, K_rgb, K_xray,
        method_name="homography_all_depth",
    )

    # --- Method 4: homography, only 3 corners projected, rest interpolated ---
    # Grid is row-major: TL=index 0, TR=index (NCOLS-1)=10, BL=index (NROWS-1)*NCOLS=110
    uv_TL, uv_TR, uv_BL = (
        _project_xyz_c_to_rgb_uv(xyz_c[[0]],   K_rgb)[0],
        _project_xyz_c_to_rgb_uv(xyz_c[[10]],  K_rgb)[0],
        _project_xyz_c_to_rgb_uv(xyz_c[[110]], K_rgb)[0],
    )
    rgb_uv_interp = interpolate_grid_uv(uv_TL, uv_TR, uv_BL, nrows=NROWS, ncols=NCOLS)
    homography_interp = _run_homography_pose(
        rgb_uv_interp, uv_xray, K_rgb, K_xray,
        method_name="homography_corner_interp",
    )

    # Pixel diff between the two rgb_uv sets (diagnostic)
    diff = np.linalg.norm(rgb_uv_interp - rgb_uv_all, axis=1)
    print(f"[INFO] rgb_uv all-depth vs corner-interp: mean={diff.mean():.3f}px  max={diff.max():.3f}px")

    return ViewCompareResult(
        name=name,
        pnp_ransac=pnp_ransac,
        ippe=ippe,
        homography_all=homography_all,
        homography_interp=homography_interp,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("\nSelect multiple XYZ_c files ...")
    xyz_c_paths = _pick_open_files_qt(
        "XYZ_c NPZ Dateien wählen (mehrere Views)", "NumPy NPZ (*.npz)"
    )
    if not xyz_c_paths:
        print("[INFO] Keine XYZ_c-Dateien gewählt. Abbruch.")
        return

    xray_uv_path = _pick_open_file_qt(
        "X-ray UV NPZ wählen (wird für alle Views verwendet)", "NumPy NPZ (*.npz)"
    )
    if xray_uv_path is None:
        print("[INFO] Keine X-ray-UV-Datei gewählt. Abbruch.")
        return

    K_rgb_path  = _pick_open_file_qt("K_rgb NPZ wählen",  "NumPy NPZ (*.npz)")
    if K_rgb_path is None:
        print("[INFO] Kein K_rgb-NPZ gewählt. Abbruch.")
        return

    K_xray_path = _pick_open_file_qt("K_xray NPZ wählen", "NumPy NPZ (*.npz)")
    if K_xray_path is None:
        print("[INFO] Kein K_xray-NPZ gewählt. Abbruch.")
        return

    xyz_c_paths = sorted(xyz_c_paths, key=lambda p: p.name.lower())

    print("\n" + "=" * 70)
    print("XYZ_c FILES (sorted by filename)")
    print("=" * 70)
    for i, px in enumerate(xyz_c_paths):
        print(f"{i + 1:2d}) {px.name}")

    K_rgb  = _load_K(K_rgb_path)
    K_xray = _load_K(K_xray_path)
    uv_xray = _load_image_points_uv(xray_uv_path)

    print(f"[INFO] uv_xray shape : {uv_xray.shape}")
    print(f"[INFO] grid          : {NROWS}x{NCOLS}, pitch={PITCH_MM}mm")
    print(f"[INFO] corner indices: TL=0, TR={NCOLS-1}, BL={(NROWS-1)*NCOLS}")

    results: list[ViewCompareResult] = []

    for i, xyz_path in enumerate(xyz_c_paths):
        name = xyz_path.stem
        _print_view_header(i, name)

        xyz_c = _load_xyz_c(xyz_path)
        print(f"[INFO] xyz_c shape = {xyz_c.shape}")

        res = _run_single_view(
            name=name, xyz_c=xyz_c, uv_xray=uv_xray, K_rgb=K_rgb, K_xray=K_xray
        )
        results.append(res)

        _print_method_summary("Direct RANSAC-PnP",              res.pnp_ransac)
        _print_method_summary("Direct IPPE",                    res.ippe)
        _print_method_summary("Homography (all 121 projected)", res.homography_all)
        _print_method_summary("Homography (3 corners + interp)",res.homography_interp)

        ref_name = "H-interp"
        ref_T    = res.homography_interp.T_cx
        for label, m in [
            ("RANSAC-PnP", res.pnp_ransac),
            ("IPPE",       res.ippe),
            ("H-all",      res.homography_all),
        ]:
            rot, dt_norm, dt = _compare_transforms(ref_T, m.T_cx)
            _print_pairwise_diff(f"{label} vs {ref_name}", rot, dt_norm, dt)

        _print_pose("T_cx RANSAC-PnP",              res.pnp_ransac.T_cx)
        _print_pose("T_cx IPPE",                    res.ippe.T_cx)
        _print_pose("T_cx Homography all-depth",    res.homography_all.T_cx)
        _print_pose("T_cx Homography corner-interp",res.homography_interp.T_cx)

        for label, T in [
            ("RANSAC", res.pnp_ransac.T_cx),
            ("IPPE",   res.ippe.T_cx),
            ("H-all",  res.homography_all.T_cx),
            ("H-interp", res.homography_interp.T_cx),
        ]:
            z = T[:3, 2]
            print(f"z_c in xray ({label:8s}): {np.array2string(z, precision=4)}  z={z[2]:.4f}")

    _print_stability_block("Direct RANSAC-PnP",               [r.pnp_ransac.T_cx       for r in results])
    _print_stability_block("Direct IPPE",                      [r.ippe.T_cx             for r in results])
    _print_stability_block("Homography (all 121 projected)",   [r.homography_all.T_cx   for r in results])
    _print_stability_block("Homography (3 corners + interp)",  [r.homography_interp.T_cx for r in results])

    print("\nDone.")


if __name__ == "__main__":
    main()