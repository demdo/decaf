# debug_pnp_raw_direct.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2

from tracking.pose_solvers import solve_pose


# ============================================================
# Config
# ============================================================

XYZ_PATH = Path(r"C:\Users\domin\Documents\Studium\Master\Masterarbeit\Projekt\Overlay\src\overlay\tests\debug_plane_fitting_xyz_c10.npz")
UV_PATH  = Path(r"C:\Users\domin\Documents\Studium\Master\Masterarbeit\Projekt\Overlay\src\overlay\tests\uv_debug_raw.npz")
KX_PATH  = Path(r"C:\Users\domin\Documents\Studium\Master\Masterarbeit\Projekt\Overlay\src\overlay\data\Poses\xray_intrinsics.npz")


# ============================================================
# Loaders
# ============================================================

def _load_xyz(path: Path) -> np.ndarray:
    """
    Supports:
    - .npy
    - .npz with keys: points_xyz_camera, points_xyz, xyz
    """
    if path.suffix.lower() == ".npy":
        xyz = np.load(str(path))
    elif path.suffix.lower() == ".npz":
        with np.load(str(path), allow_pickle=False) as npz:
            for key in ("points_xyz_camera", "points_xyz", "xyz"):
                if key in npz.files:
                    xyz = np.asarray(npz[key], dtype=np.float64)
                    break
            else:
                raise KeyError(
                    f"{path.name}: expected one of keys "
                    f"['points_xyz_camera', 'points_xyz', 'xyz'], found {list(npz.files)}"
                )
    else:
        raise ValueError(f"Unsupported xyz file type: {path}")

    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    print(f"Loaded xyz: {xyz.shape}  from {path}")
    return xyz


def _load_uv(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as npz:
        if "points_uv" not in npz.files:
            raise KeyError(f"{path.name}: expected key 'points_uv', found {list(npz.files)}")
        uv = np.asarray(npz["points_uv"], dtype=np.float64).reshape(-1, 2)

    print(f"Loaded uv:  {uv.shape}  from {path}")
    return uv


def _load_kx(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as npz:
        for key in ("K_xray", "Kx", "K"):
            if key in npz.files:
                K = np.asarray(npz[key], dtype=np.float64).reshape(3, 3)
                print(f"Loaded Kx from key '{key}' in {path}")
                return K

    raise KeyError(f"{path.name}: expected one of keys ['K_xray', 'Kx', 'K']")


# ============================================================
# Helpers
# ============================================================

def _rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def _sep(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


# ============================================================
# Main
# ============================================================

def main() -> None:
    _sep("LOAD RAW DATA")
    xyz = _load_xyz(XYZ_PATH)
    uv = _load_uv(UV_PATH)
    Kx = _load_kx(KX_PATH)

    if xyz.shape[0] != uv.shape[0]:
        raise ValueError(f"Point count mismatch: xyz={xyz.shape[0]}  uv={uv.shape[0]}")

    print(f"xyz shape: {xyz.shape}")
    print(f"uv  shape: {uv.shape}")
    print("Kx:")
    print(Kx)

    _sep("DIRECT PnP (RAW / NO FLIP / NO REORDER / NO ROTATION)")

    pose = solve_pose(
        object_points_xyz=xyz,
        image_points_uv=uv,
        K=Kx,
        dist_coeffs=None,
        pose_method="ippe",
        refine_with_iterative=False,
    )

    T_cx = _rvec_tvec_to_transform(pose.rvec, pose.tvec)

    dot_z = float(np.dot(T_cx[:3, 2], np.array([0.0, 0.0, 1.0])))
    tx = float(T_cx[0, 3])
    ty = float(T_cx[1, 3])
    tz = float(T_cx[2, 3])
    tnorm = float(np.linalg.norm(T_cx[:3, 3]))

    _sep("RESULT")
    print(f"reproj mean   = {pose.reproj_mean_px:.4f} px")
    print(f"reproj median = {pose.reproj_median_px:.4f} px")
    print(f"reproj max    = {pose.reproj_max_px:.4f} px")
    #print(f"inliers       = {len(pose.inlier_idx)}/{xyz.shape[0]}")
    print()
    print(f"dot(z_c, z_x) = {dot_z:+.6f}")
    print(f"t_x           = {tx:+.6f} m")
    print(f"t_y           = {ty:+.6f} m")
    print(f"t_z           = {tz:+.6f} m")
    print(f"|t|           = {tnorm:.6f} m")

    print("\nT_cx =")
    for row in T_cx:
        print(" ", np.array2string(row, precision=6, suppress_small=False))
    
    """
    _sep("SAVE")
    out_path = XYZ_PATH.with_name("debug_pnp_raw_direct_result.npz")
    np.savez(
        out_path,
        points_xyz=xyz.astype(np.float64),
        points_uv=uv.astype(np.float64),
        K_x=Kx.astype(np.float64),
        rvec=np.asarray(pose.rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(pose.tvec, dtype=np.float64).reshape(3, 1),
        T_cx=T_cx.astype(np.float64),
        reproj_mean_px=float(pose.reproj_mean_px),
        reproj_median_px=float(pose.reproj_median_px),
        reproj_max_px=float(pose.reproj_max_px),
        inlier_idx=np.asarray(pose.inlier_idx, dtype=np.int64),
        dot_zc_zx=float(dot_z),
        t_x=float(tx),
        t_y=float(ty),
        t_z=float(tz),
        t_norm=float(tnorm),
    )
    print(f"Saved result -> {out_path}")
    """

if __name__ == "__main__":
    main()