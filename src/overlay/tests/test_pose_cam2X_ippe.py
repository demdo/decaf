# -*- coding: utf-8 -*-
"""
Show:
- principal point in original image
- principal point in warped image
- principal ray of both IPPE candidates in original image
- same rays warped with the same H_warp into warped image
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.pose_solvers import solve_pose


# ============================================================
# Qt helpers
# ============================================================

def _qt() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _pick_file(title: str, filt: str) -> Path | None:
    _qt()
    path, _ = QFileDialog.getOpenFileName(None, title, "", filt)
    return Path(path) if path else None


# ============================================================
# Hardcoded K_rgb
# ============================================================

K_rgb = np.array(
    [
        [1360.41301,   0.0,       976.230766],
        [0.0,       1361.74342,   547.474129],
        [0.0,          0.0,         1.0     ],
    ],
    dtype=np.float64,
)


# ============================================================
# IO
# ============================================================

def load_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not load image: {path}")
    return img


def load_xyz(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as z:
        for k in ("points_xyz_camera_filt", "points_xyz_camera"):
            if k in z.files:
                pts = np.asarray(z[k], dtype=np.float64)
                if pts.ndim != 2 or pts.shape[1] != 3:
                    raise RuntimeError(f"XYZ array '{k}' has invalid shape: {pts.shape}")
                if np.mean(np.abs(pts)) < 10.0:
                    pts = pts * 1000.0
                return pts
    raise RuntimeError("No XYZ found in NPZ.")


def load_kx(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as z:
        for key in ("K", "Kx", "K_xray"):
            if key in z.files:
                K = np.asarray(z[key], dtype=np.float64)
                if K.shape != (3, 3):
                    raise RuntimeError(f"{key} has invalid shape: {K.shape}")
                return K
    raise RuntimeError("No key 'K', 'Kx', or 'K_xray' found in NPZ.")


def load_uv_xray(path: Path) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as z:
        if "uv_xray" in z.files:
            uv = np.asarray(z["uv_xray"], dtype=np.float64)
            if uv.ndim == 2 and uv.shape[1] == 2:
                return uv

        for key in z.files:
            arr = np.asarray(z[key], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr

    raise RuntimeError("No (N,2) uv array found in NPZ.")


# ============================================================
# Geometry helpers
# ============================================================

def fit_plane_svd(points_xyz_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_xyz_mm, dtype=np.float64)
    center = np.mean(pts, axis=0)
    A = pts - center
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    n = vt[-1]
    n /= np.linalg.norm(n)
    return n, center


def ensure_plane_faces_camera(n: np.ndarray, center: np.ndarray) -> np.ndarray:
    to_cam = -center
    if float(np.dot(n, to_cam)) < 0.0:
        n = -n
    return n


def project_vector_onto_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.asarray(n, dtype=np.float64).reshape(3)
    n /= np.linalg.norm(n)
    return v - np.dot(v, n) * n


def build_plane_visual_basis(
    n_c: np.ndarray,
    *,
    flip_up: bool = False,
    flip_180: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    x_cam = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    e_x = project_vector_onto_plane(x_cam, n_c)
    if np.linalg.norm(e_x) < 1e-9:
        raise RuntimeError("Projected x_cam onto plane is degenerate.")
    e_x /= np.linalg.norm(e_x)

    e_y = np.cross(n_c, e_x)
    if np.linalg.norm(e_y) < 1e-9:
        raise RuntimeError("Failed to construct e_y.")
    e_y /= np.linalg.norm(e_y)

    if flip_up:
        e_y = -e_y

    if flip_180:
        e_x = -e_x
        e_y = -e_y

    return e_x, e_y


def build_plane_patch_corners_mm(
    center_c: np.ndarray,
    e_x: np.ndarray,
    e_y: np.ndarray,
    out_w_px: int,
    out_h_px: int,
    mm_per_px: float,
) -> np.ndarray:
    half_w_mm = 0.5 * out_w_px * mm_per_px
    half_h_mm = 0.5 * out_h_px * mm_per_px

    p_tl = center_c + half_h_mm * e_y - half_w_mm * e_x
    p_tr = center_c + half_h_mm * e_y + half_w_mm * e_x
    p_br = center_c - half_h_mm * e_y + half_w_mm * e_x
    p_bl = center_c - half_h_mm * e_y - half_w_mm * e_x
    return np.vstack([p_tl, p_tr, p_br, p_bl])


def project_points(K: np.ndarray, P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    uvw = (K @ P.T).T
    return uvw[:, :2] / uvw[:, 2:3]


# ============================================================
# Pose helpers
# ============================================================

def rvec_to_R(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def invert_pose_cx_to_xc(R_cx: np.ndarray, t_cx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_xc = R_cx.T
    t_xc = -R_xc @ np.asarray(t_cx, dtype=np.float64).reshape(3)
    return R_xc, t_xc


def pose_to_dict(rvec, tvec, cand_idx, reproj_mean_px=None) -> dict:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)

    R_cx = rvec_to_R(rvec)
    R_xc, t_xc = invert_pose_cx_to_xc(R_cx, tvec)

    return {
        "candidate_index": int(cand_idx),
        "rvec": rvec,
        "tvec": tvec,
        "R_cx": R_cx,
        "R_xc": R_xc,
        "t_xc": t_xc,
        "reproj_mean_px": None if reproj_mean_px is None else float(reproj_mean_px),
    }


def solve_both_candidates(
    xyz_mm: np.ndarray,
    uv_xray: np.ndarray,
    K_xray: np.ndarray,
) -> tuple[dict | None, dict | None]:
    res = solve_pose(
        object_points_xyz=xyz_mm,
        image_points_uv=uv_xray,
        K=K_xray,
        pose_method="ippe",
        refine_with_iterative=False,
    )

    chosen = pose_to_dict(
        res.rvec,
        res.tvec,
        getattr(res, "candidate_index", 0),
        getattr(res, "reproj_mean_px", None),
    )

    other = None
    if hasattr(res, "all_candidates") and res.all_candidates is not None:
        other_idx = 1 - chosen["candidate_index"]
        if other_idx < len(res.all_candidates):
            c = res.all_candidates[other_idx]
            other = pose_to_dict(
                c.rvec,
                c.tvec,
                other_idx,
                getattr(c, "reproj_mean_px", None),
            )

    if chosen["candidate_index"] == 0:
        cand0, cand1 = chosen, other
    else:
        cand0, cand1 = other, chosen

    return cand0, cand1


# ============================================================
# Principal point + ray helpers
# ============================================================

def get_principal_point_uv(K: np.ndarray) -> np.ndarray:
    return np.array([float(K[0, 2]), float(K[1, 2])], dtype=np.float64)


def warp_uv_with_homography(uv: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.array([[[float(uv[0]), float(uv[1])]]], dtype=np.float32)
    pts_warp = cv2.perspectiveTransform(pts, H)
    return pts_warp[0, 0].astype(np.float64)


def project_point(K: np.ndarray, P_c: np.ndarray) -> np.ndarray | None:
    P = np.asarray(P_c, dtype=np.float64).reshape(3)
    z = float(P[2])
    if z <= 1e-9:
        return None
    uvw = K @ P
    return (uvw[:2] / uvw[2]).astype(np.float64)


def make_xray_principal_ray_points(
    pose: dict,
    *,
    ray_len_mm: float = 1200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ray in camera frame:
        X_c(lambda) = t_xc + lambda * d_c
    with d_c = R_xc[:, 2]
    """
    t_xc = np.asarray(pose["t_xc"], dtype=np.float64).reshape(3)
    d_c = np.asarray(pose["R_xc"], dtype=np.float64)[:, 2].copy()
    d_c /= np.linalg.norm(d_c)

    P0 = t_xc
    P1 = t_xc + ray_len_mm * d_c
    return P0, P1


def draw_pp_marker(
    img: np.ndarray,
    uv: np.ndarray,
    label: str,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    u = int(round(float(uv[0])))
    v = int(round(float(uv[1])))

    cv2.circle(img, (u, v), 7, color, 2, cv2.LINE_AA)
    cv2.drawMarker(
        img,
        (u, v),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        label,
        (u + 10, v - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_ray_segment(
    img: np.ndarray,
    uv0: np.ndarray,
    uv1: np.ndarray,
    color: tuple[int, int, int],
    label: str | None = None,
) -> None:
    p0 = (int(round(float(uv0[0]))), int(round(float(uv0[1]))))
    p1 = (int(round(float(uv1[0]))), int(round(float(uv1[1]))))
    cv2.line(img, p0, p1, color, 2, cv2.LINE_AA)

    if label is not None:
        mx = int(round(0.5 * (p0[0] + p1[0])))
        my = int(round(0.5 * (p0[1] + p1[1])))
        cv2.putText(
            img,
            label,
            (mx + 8, my - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    img_path = _pick_file("Select camera image PNG", "PNG image (*.png)")
    if img_path is None:
        print("No image selected.")
        return

    xyz_path = _pick_file("Select XYZ camera NPZ", "NPZ (*.npz)")
    if xyz_path is None:
        print("No XYZ NPZ selected.")
        return

    kx_path = _pick_file("Select K_xray NPZ", "NPZ (*.npz)")
    if kx_path is None:
        print("No K_xray NPZ selected.")
        return

    uvx_path = _pick_file("Select uv_xray NPZ", "NPZ (*.npz)")
    if uvx_path is None:
        print("No uv_xray NPZ selected.")
        return

    img = load_image_bgr(img_path)
    xyz_mm = load_xyz(xyz_path)
    K_x = load_kx(kx_path)
    uv_xray = load_uv_xray(uvx_path)

    if xyz_mm.shape[0] != uv_xray.shape[0]:
        raise RuntimeError(
            f"Point count mismatch: xyz={xyz_mm.shape[0]}, uv_xray={uv_xray.shape[0]}"
        )

    # --------------------------------------------------------
    # Plane warp exactly as before
    # --------------------------------------------------------
    n_c, plane_center_c = fit_plane_svd(xyz_mm)
    n_c = ensure_plane_faces_camera(n_c, plane_center_c)

    e_x, e_y = build_plane_visual_basis(
        n_c,
        flip_up=False,
        flip_180=False,
    )

    mm_per_px = 2.0
    out_w_px = 700
    out_h_px = 700

    patch_center_c = plane_center_c.copy()

    corners_c = build_plane_patch_corners_mm(
        center_c=patch_center_c,
        e_x=e_x,
        e_y=e_y,
        out_w_px=out_w_px,
        out_h_px=out_h_px,
        mm_per_px=mm_per_px,
    )

    src = project_points(K_rgb, corners_c).astype(np.float32)
    dst = np.array(
        [
            [0, 0],
            [out_w_px - 1, 0],
            [out_w_px - 1, out_h_px - 1],
            [0, out_h_px - 1],
        ],
        dtype=np.float32,
    )

    H_warp = cv2.getPerspectiveTransform(src, dst)
    vis_warp = cv2.warpPerspective(img, H_warp, (out_w_px, out_h_px))

    # --------------------------------------------------------
    # Principal point
    # --------------------------------------------------------
    pp_orig = get_principal_point_uv(K_rgb)
    pp_warp = warp_uv_with_homography(pp_orig, H_warp)

    # --------------------------------------------------------
    # IPPE candidates
    # --------------------------------------------------------
    cand0, cand1 = solve_both_candidates(xyz_mm, uv_xray, K_x)

    # --------------------------------------------------------
    # Rays in original image
    # --------------------------------------------------------
    rays_orig = []

    for pose, color, label in [
        (cand0, (0, 0, 255), "ray0"),
        (cand1, (0, 255, 0), "ray1"),
    ]:
        if pose is None:
            continue

        P0_c, P1_c = make_xray_principal_ray_points(pose, ray_len_mm=1200.0)

        uv0 = project_point(K_rgb, P0_c)
        uv1 = project_point(K_rgb, P1_c)

        if uv0 is None or uv1 is None:
            print(f"[WARN] Could not project {label} because z <= 0.")
            continue

        rays_orig.append((uv0, uv1, color, label))

        print(f"\n{label}:")
        print("  t_xc =", np.array2string(pose['t_xc'], precision=3, suppress_small=False))
        print("  d_c  =", np.array2string(pose['R_xc'][:, 2], precision=6, suppress_small=False))
        print("  uv0  =", np.array2string(uv0, precision=3, suppress_small=False))
        print("  uv1  =", np.array2string(uv1, precision=3, suppress_small=False))

    # --------------------------------------------------------
    # Build previews
    # --------------------------------------------------------
    vis_orig = img.copy()
    cv2.polylines(vis_orig, [np.int32(src)], True, (0, 255, 255), 2, cv2.LINE_AA)
    draw_pp_marker(vis_orig, pp_orig, label="PP orig", color=(255, 255, 255))

    for uv0, uv1, color, label in rays_orig:
        draw_ray_segment(vis_orig, uv0, uv1, color, label)

    cv2.putText(
        vis_orig,
        "View: ORIGINAL  |  t=toggle",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    vis_warp_preview = vis_warp.copy()
    draw_pp_marker(vis_warp_preview, pp_warp, label="PP warp", color=(255, 255, 255))

    for uv0, uv1, color, label in rays_orig:
        uv0_w = warp_uv_with_homography(uv0, H_warp)
        uv1_w = warp_uv_with_homography(uv1, H_warp)
        draw_ray_segment(vis_warp_preview, uv0_w, uv1_w, color, label)

    cv2.putText(
        vis_warp_preview,
        "View: WARPED  |  t=toggle",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------
    show_warp = False
    win = "Principal point + IPPE rays"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow(win, vis_warp_preview if show_warp else vis_orig)
        k = cv2.waitKey(0) & 0xFF

        if k in (27, ord("q")):
            break
        elif k == ord("t"):
            show_warp = not show_warp

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()