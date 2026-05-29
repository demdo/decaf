# -*- coding: utf-8 -*-
"""
debug_pose_axes_in_xray.py

Load:
- xyz npz
- uv npz
- Kx npz
- X-ray image

Then:
- solve pose with solve_pose(...)
- print T and inv(T)
- overlay projected axes in the X-ray image

Axes meaning
------------
The 3D points `xyz` are assumed to be expressed in the RGB camera frame.
Therefore, the plotted axes are the RGB camera-frame axes projected into
the X-ray image using the estimated pose.

Visualization
-------------
- origin: grid center point
- +x_c : red
- +y_c : green
- +z_c : blue
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pydicom

from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.pose_solvers import solve_pose


# ============================================================
# Config
# ============================================================
WINDOW_NAME = "debug_pose_axes_in_xray"
DISPLAY_MAX_W = 1400
DISPLAY_MAX_H = 1000

POSE_METHOD = "ippe"
REFINE_WITH_ITERATIVE = False

# axis length in meters
AXIS_LEN_M = 0.02

# optionally also draw the four grid corner indices
DRAW_CORNER_POINTS = True
CORNER_INDICES = [0, 10, 110, 120]


# ============================================================
# Qt helpers
# ============================================================
def _get_qapp() -> tuple[QApplication, bool]:
    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True
    return app, created


def pick_open_file_qt(title: str, file_filter: str) -> str | None:
    app, created = _get_qapp()

    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        file_filter,
    )

    if created:
        app.quit()

    path = path.strip()
    return path if path else None


# ============================================================
# Loaders
# ============================================================
def load_xyz_npz(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        for key in ("points_xyz_camera", "points_xyz", "xyz"):
            if key in npz.files:
                xyz = np.asarray(npz[key], dtype=np.float64).reshape(-1, 3)
                print(f"[INFO] Loaded xyz from key '{key}'")
                return xyz

    raise KeyError(
        f"{Path(path).name}: expected one of keys "
        f"['points_xyz_camera', 'points_xyz', 'xyz']"
    )


def load_uv_npz(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        if "points_uv" not in npz.files:
            raise KeyError(
                f"{Path(path).name}: expected key 'points_uv', found {list(npz.files)}"
            )
        uv = np.asarray(npz["points_uv"], dtype=np.float64).reshape(-1, 2)
    return uv


def load_k_npz(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as npz:
        for key in ("K_xray", "Kx", "K"):
            if key in npz.files:
                K = np.asarray(npz[key], dtype=np.float64).reshape(3, 3)
                print(f"[INFO] Loaded Kx from key '{key}'")
                return K

    raise KeyError(
        f"{Path(path).name}: expected one of keys ['K_xray', 'Kx', 'K']"
    )


def load_xray_image(path: str) -> np.ndarray:
    path_l = path.lower()

    if path_l.endswith((".dcm", ".ima")):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img.astype(np.uint8)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


# ============================================================
# Math helpers
# ============================================================
def rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def project_points(
    points_xyz: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)

    if dist_coeffs is None:
        dist = np.zeros((5, 1), dtype=np.float64)
    else:
        dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    uv, _ = cv2.projectPoints(
        pts,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        K,
        dist,
    )
    return uv.reshape(-1, 2)


def compute_grid_center(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.shape[0] < 121:
        return np.mean(xyz, axis=0)

    # center of 11x11 grid -> index 60
    return xyz[60].copy()


# ============================================================
# Drawing helpers
# ============================================================
def draw_cross(
    img: np.ndarray,
    u: float,
    v: float,
    color: tuple[int, int, int],
    size: int = 10,
    thickness: int = 2,
) -> None:
    uu = int(round(u))
    vv = int(round(v))
    cv2.line(img, (uu - size, vv), (uu + size, vv), color, thickness, cv2.LINE_AA)
    cv2.line(img, (uu, vv - size), (uu, vv + size), color, thickness, cv2.LINE_AA)


def draw_label(
    img: np.ndarray,
    text: str,
    u: float,
    v: float,
    color: tuple[int, int, int],
    dx: int = 8,
    dy: int = -8,
    scale: float = 0.7,
) -> None:
    x = int(round(u)) + dx
    y = int(round(v)) + dy
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_axes_overlay(
    image_bgr: np.ndarray,
    Kx: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    xyz: np.ndarray,
) -> np.ndarray:
    vis = image_bgr.copy()

    origin = compute_grid_center(xyz)

    pts_axes = np.array(
        [
            origin,
            origin + np.array([AXIS_LEN_M, 0.0, 0.0], dtype=np.float64),
            origin + np.array([0.0, AXIS_LEN_M, 0.0], dtype=np.float64),
            origin + np.array([0.0, 0.0, AXIS_LEN_M], dtype=np.float64),
        ],
        dtype=np.float64,
    )

    uv_axes = project_points(
        points_xyz=pts_axes,
        rvec=rvec,
        tvec=tvec,
        K=Kx,
        dist_coeffs=None,
    )

    p0 = uv_axes[0]
    px = uv_axes[1]
    py = uv_axes[2]
    pz = uv_axes[3]

    # origin
    draw_cross(vis, p0[0], p0[1], color=(255, 255, 255), size=10, thickness=2)
    draw_label(vis, "O", p0[0], p0[1], color=(255, 255, 255))

    # x axis (red)
    cv2.line(
        vis,
        (int(round(p0[0])), int(round(p0[1]))),
        (int(round(px[0])), int(round(px[1]))),
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    draw_cross(vis, px[0], px[1], color=(0, 0, 255), size=8, thickness=2)
    draw_label(vis, "+x_c", px[0], px[1], color=(0, 0, 255))

    # y axis (green)
    cv2.line(
        vis,
        (int(round(p0[0])), int(round(p0[1]))),
        (int(round(py[0])), int(round(py[1]))),
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    draw_cross(vis, py[0], py[1], color=(0, 255, 0), size=8, thickness=2)
    draw_label(vis, "+y_c", py[0], py[1], color=(0, 255, 0))

    # z axis (blue)
    cv2.line(
        vis,
        (int(round(p0[0])), int(round(p0[1]))),
        (int(round(pz[0])), int(round(pz[1]))),
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    draw_cross(vis, pz[0], pz[1], color=(255, 0, 0), size=8, thickness=2)
    draw_label(vis, "+z_c", pz[0], pz[1], color=(255, 0, 0))

    if DRAW_CORNER_POINTS:
        uv_corners = project_points(
            points_xyz=xyz[CORNER_INDICES],
            rvec=rvec,
            tvec=tvec,
            K=Kx,
            dist_coeffs=None,
        )
        for idx, (u, v) in zip(CORNER_INDICES, uv_corners):
            draw_cross(vis, u, v, color=(0, 0, 255), size=8, thickness=2)
            draw_label(vis, str(idx), u, v, color=(0, 0, 255), dx=10, dy=-10, scale=0.6)

    return vis


# ============================================================
# Display helpers
# ============================================================
def make_display_transform(
    img_shape: tuple[int, int],
    max_w: int,
    max_h: int,
) -> tuple[float, int, int, int, int]:
    h, w = img_shape[:2]
    scale = min(max_w / float(w), max_h / float(h))
    if scale <= 0:
        scale = 1.0

    disp_w = max(1, int(round(w * scale)))
    disp_h = max(1, int(round(h * scale)))

    off_x = max(0, (max_w - disp_w) // 2)
    off_y = max(0, (max_h - disp_h) // 2)

    return scale, disp_w, disp_h, off_x, off_y


def render_for_display(
    img_bgr: np.ndarray,
    max_w: int,
    max_h: int,
) -> np.ndarray:
    scale, disp_w, disp_h, off_x, off_y = make_display_transform(img_bgr.shape, max_w, max_h)

    if abs(scale - 1.0) < 1e-12:
        resized = img_bgr.copy()
    else:
        resized = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    canvas[off_y:off_y + disp_h, off_x:off_x + disp_w] = resized
    return canvas


# ============================================================
# Print helpers
# ============================================================
def print_pose_report(title: str, T: np.ndarray) -> None:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    dot_z = float(np.dot(T[:3, 2], np.array([0.0, 0.0, 1.0])))
    tx = float(T[0, 3])
    ty = float(T[1, 3])
    tz = float(T[2, 3])
    tnorm = float(np.linalg.norm(T[:3, 3]))

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"dot(z_c, z_x?) = {dot_z:+.6f}")
    print(f"t_x            = {tx:+.6f} m")
    print(f"t_y            = {ty:+.6f} m")
    print(f"t_z            = {tz:+.6f} m")
    print(f"|t|            = {tnorm:.6f} m")
    print("\nT =")
    for row in T:
        print(" ", np.array2string(row, precision=6, suppress_small=False))


# ============================================================
# Main
# ============================================================
def main() -> None:
    xyz_path = pick_open_file_qt(
        "Load xyz NPZ",
        "NumPy NPZ (*.npz);;All files (*)",
    )
    if not xyz_path:
        print("[INFO] No xyz npz selected. Abort.")
        return

    uv_path = pick_open_file_qt(
        "Load UV NPZ",
        "NumPy NPZ (*.npz);;All files (*)",
    )
    if not uv_path:
        print("[INFO] No uv npz selected. Abort.")
        return

    kx_path = pick_open_file_qt(
        "Load Kx NPZ",
        "NumPy NPZ (*.npz);;All files (*)",
    )
    if not kx_path:
        print("[INFO] No Kx npz selected. Abort.")
        return

    xray_img_path = pick_open_file_qt(
        "Load X-ray image",
        "X-ray / DICOM (*.dcm *.ima *.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)",
    )
    if not xray_img_path:
        print("[INFO] No X-ray image selected. Abort.")
        return

    xyz = load_xyz_npz(xyz_path)
    uv = load_uv_npz(uv_path)
    Kx = load_k_npz(kx_path)
    xray_gray = load_xray_image(xray_img_path)
    xray_bgr = cv2.cvtColor(xray_gray, cv2.COLOR_GRAY2BGR)

    if xyz.shape[0] != uv.shape[0]:
        raise ValueError(f"Point count mismatch: xyz={xyz.shape[0]} uv={uv.shape[0]}")

    print("\n" + "=" * 78)
    print("SOLVE POSE")
    print("=" * 78)

    pose = solve_pose(
        object_points_xyz=xyz,
        image_points_uv=uv,
        K=Kx,
        dist_coeffs=None,
        pose_method=POSE_METHOD,
        refine_with_iterative=REFINE_WITH_ITERATIVE,
        ransac_reprojection_error_px=8.0,
        ransac_confidence=0.99,
        ransac_iterations_count=100,
    )

    print(f"method         = {pose.method}")
    print(f"candidate_idx  = {pose.candidate_index}")
    print(f"refined        = {pose.refined_with_iterative}")
    print(f"reproj mean    = {pose.reproj_mean_px:.4f} px")
    print(f"reproj median  = {pose.reproj_median_px:.4f} px")
    print(f"reproj max     = {pose.reproj_max_px:.4f} px")

    T = rvec_tvec_to_transform(pose.rvec, pose.tvec)
    T_inv = invert_transform(T)

    print_pose_report("POSE AS RETURNED BY solve_pose()", T)
    print_pose_report("INVERSE OF THAT POSE", T_inv)

    vis = draw_axes_overlay(
        image_bgr=xray_bgr,
        Kx=Kx,
        rvec=pose.rvec,
        tvec=pose.tvec,
        xyz=xyz,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow(WINDOW_NAME, render_for_display(vis, DISPLAY_MAX_W, DISPLAY_MAX_H))
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()