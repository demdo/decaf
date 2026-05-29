# -*- coding: utf-8 -*-
"""
debug_pose_cam_xray.py

Live RealSense overlay of the X-ray principal ray for both IPPE candidates.

Behavior
--------
Live mode:
- show live RGB image from RealSense
- draw:
    - RGB principal point
    - candidate 0 ray
    - candidate 1 ray
    - projected source points (optional)

If SPACE is pressed:
- stop live mode
- take the last RAW RGB frame (without any overlays)
- warp only the plane via warpPerspective
- rotate the warped image 90° clockwise
- then re-draw ONLY the intersection points:
    (central X-ray ray) ∩ (plane)

If 's' is pressed in live mode:
- save the last RAW RGB frame
- no lines, no text, no markers

Controls
--------
Live mode:
    q / ESC : quit
    0       : toggle candidate 0
    1       : toggle candidate 1
    p       : toggle source points in live image
    f       : freeze / unfreeze live frame
    SPACE   : switch to top view
    s       : save raw live RGB frame

Warp mode:
    q / ESC : quit
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.pose_solvers import solve_pose


# ============================================================
# Qt helpers
# ============================================================

def _qt() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _pick_npz_file(title: str) -> Path | None:
    _qt()
    path, _ = QFileDialog.getOpenFileName(None, title, "", "NumPy NPZ (*.npz)")
    return Path(path) if path else None


# ============================================================
# Load helpers
# ============================================================

def load_kx(npz_path: Path) -> np.ndarray:
    with np.load(str(npz_path), allow_pickle=False) as z:
        for key in ("K", "Kx", "K_xray"):
            if key in z.files:
                K = np.asarray(z[key], dtype=np.float64)
                if K.shape != (3, 3):
                    raise ValueError(f"{key} has wrong shape: {K.shape}")
                return K
    raise KeyError("No key 'K', 'Kx', or 'K_xray' found.")


def load_xyz_camera_mm(npz_path: Path) -> np.ndarray:
    with np.load(str(npz_path), allow_pickle=False) as z:
        for key in ("points_xyz_camera_filt", "points_xyz_camera"):
            if key in z.files:
                xyz = np.asarray(z[key], dtype=np.float64)
                if xyz.ndim != 2 or xyz.shape[1] != 3:
                    raise ValueError(f"{key} has invalid shape: {xyz.shape}")
                if np.nanmean(np.abs(xyz)) < 10.0:
                    xyz = xyz * 1000.0
                return xyz
    raise KeyError("No key 'points_xyz_camera_filt' or 'points_xyz_camera' found.")


def load_uv_xray(npz_path: Path) -> np.ndarray:
    with np.load(str(npz_path), allow_pickle=False) as z:
        if "uv_xray" in z.files:
            uv = np.asarray(z["uv_xray"], dtype=np.float64)
            if uv.ndim == 2 and uv.shape[1] == 2:
                return uv

        for key in z.files:
            arr = np.asarray(z[key], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr

    raise KeyError("No (N,2) uv array found in NPZ.")


# ============================================================
# Pose helpers
# ============================================================

def rvec_to_R(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def pose_to_dict(rvec, tvec, cand_idx, reproj_mean_px=None) -> dict:
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    return {
        "candidate_index": int(cand_idx),
        "rvec": rvec,
        "tvec": tvec,
        "R_cx": rvec_to_R(rvec),
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


def invert_pose_cx_to_xc(R_cx: np.ndarray, t_cx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_xc = R_cx.T
    t_xc = -R_cx.T @ np.asarray(t_cx, dtype=np.float64).reshape(3)
    return R_xc, t_xc


# ============================================================
# X-ray ray helpers
# ============================================================

def make_principal_ray_x() -> tuple[np.ndarray, np.ndarray]:
    origin_x = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    direction_x = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return origin_x, direction_x


def transform_ray_x_to_c(
    origin_x: np.ndarray,
    direction_x: np.ndarray,
    R_xc: np.ndarray,
    t_xc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origin_c = R_xc @ origin_x + t_xc
    direction_c = R_xc @ direction_x
    direction_c = direction_c / np.linalg.norm(direction_c)
    return origin_c, direction_c


def make_two_points_on_ray(
    origin_c: np.ndarray,
    direction_c: np.ndarray,
    lam1_mm: float,
    lam2_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    p1_c = origin_c + lam1_mm * direction_c
    p2_c = origin_c + lam2_mm * direction_c
    return p1_c, p2_c


# ============================================================
# Plane / top-view helpers
# ============================================================

def fit_plane_svd(points_xyz_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_xyz_mm, dtype=np.float64)
    centroid = np.mean(pts, axis=0)
    A = pts - centroid
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    return normal, centroid


def project_vector_onto_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.asarray(n, dtype=np.float64).reshape(3)
    n = n / np.linalg.norm(n)
    return v - np.dot(v, n) * n


def build_topview_basis_from_camera_axes(n_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    e_right_c : on plane, points to camera +y projected onto plane
    e_back_c  : on plane, points to camera +x projected onto plane
    """
    x_cam = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # front -> back
    y_cam = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # left  -> right

    e_back = project_vector_onto_plane(x_cam, n_c)
    e_right = project_vector_onto_plane(y_cam, n_c)

    if np.linalg.norm(e_back) < 1e-9:
        raise ValueError("Projected x_cam onto plane is degenerate.")
    if np.linalg.norm(e_right) < 1e-9:
        raise ValueError("Projected y_cam onto plane is degenerate.")

    e_back /= np.linalg.norm(e_back)
    e_right = e_right - np.dot(e_right, e_back) * e_back
    e_right /= np.linalg.norm(e_right)
    return e_right, e_back


def project_points(K: np.ndarray, P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    uvw = (K @ P.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def build_plane_patch_corners_mm(
    center_c: np.ndarray,
    e_right_c: np.ndarray,
    e_back_c: np.ndarray,
    out_w_px: int,
    out_h_px: int,
    mm_per_px: float,
) -> np.ndarray:
    half_w_mm = 0.5 * out_w_px * mm_per_px
    half_h_mm = 0.5 * out_h_px * mm_per_px

    p_tl = center_c + half_h_mm * e_back_c - half_w_mm * e_right_c
    p_tr = center_c + half_h_mm * e_back_c + half_w_mm * e_right_c
    p_br = center_c - half_h_mm * e_back_c + half_w_mm * e_right_c
    p_bl = center_c - half_h_mm * e_back_c - half_w_mm * e_right_c
    return np.vstack([p_tl, p_tr, p_br, p_bl])


def camera_point_to_topview_uv_unrotated(
    X_c: np.ndarray,
    center_c: np.ndarray,
    e_right_c: np.ndarray,
    e_back_c: np.ndarray,
    out_w_px: int,
    out_h_px: int,
    mm_per_px: float,
) -> np.ndarray:
    d = np.asarray(X_c, dtype=np.float64).reshape(3) - center_c
    u = (np.dot(d, e_right_c) / mm_per_px) + 0.5 * out_w_px
    v = (-np.dot(d, e_back_c) / mm_per_px) + 0.5 * out_h_px
    return np.array([u, v], dtype=np.float64)


def rotate_uv_90_clockwise(
    uv: np.ndarray,
    w_before: int,
    h_before: int,
) -> np.ndarray:
    u, v = float(uv[0]), float(uv[1])
    return np.array([h_before - 1 - v, u], dtype=np.float64)


def intersect_ray_with_plane(
    source_c: np.ndarray,
    ray_dir_c: np.ndarray,
    plane_normal_c: np.ndarray,
    plane_point_c: np.ndarray,
) -> np.ndarray | None:
    """
    Ray: X = source_c + lambda * ray_dir_c
    Plane: n^T (X - plane_point_c) = 0
    """
    s = np.asarray(source_c, dtype=np.float64).reshape(3)
    d = np.asarray(ray_dir_c, dtype=np.float64).reshape(3)
    n = np.asarray(plane_normal_c, dtype=np.float64).reshape(3)
    p0 = np.asarray(plane_point_c, dtype=np.float64).reshape(3)

    denom = float(np.dot(n, d))
    if abs(denom) < 1e-9:
        return None

    lam = -float(np.dot(n, s - p0)) / denom
    return s + lam * d


def clip_line_to_image(
    uv1: np.ndarray,
    uv2: np.ndarray,
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    p1 = tuple(np.round(uv1).astype(int))
    p2 = tuple(np.round(uv2).astype(int))
    ok, q1, q2 = cv2.clipLine((0, 0, width, height), p1, p2)
    if not ok:
        return None
    return q1, q2


# ============================================================
# Projection / drawing helpers
# ============================================================

def project_point(K: np.ndarray, X_c: np.ndarray) -> np.ndarray | None:
    X_c = np.asarray(X_c, dtype=np.float64).reshape(3)
    if X_c[2] <= 1e-9:
        return None

    x = K @ X_c
    return x[:2] / x[2]


def to_int_pt(uv: np.ndarray) -> tuple[int, int]:
    return tuple(np.round(uv).astype(int))


def draw_cross(
    img: np.ndarray,
    pt: tuple[int, int],
    color: tuple[int, int, int],
    size: int = 10,
    thickness: int = 2,
) -> None:
    x, y = pt
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def draw_circle(
    img: np.ndarray,
    pt: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 7,
    thickness: int = 2,
) -> None:
    cv2.circle(img, pt, radius, color, thickness, cv2.LINE_AA)


def save_raw_frame(frame_bgr: np.ndarray) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path.cwd() / f"debug_pose_cam_xray_raw_{ts}.png"
    ok = cv2.imwrite(str(out_path), frame_bgr)
    if not ok:
        raise RuntimeError(f"Failed to save image to {out_path}")
    return out_path


# ============================================================
# RealSense helpers
# ============================================================

class RealSenseRGB:
    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.profile = None
        self.K_rgb = None

    def start(self) -> None:
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.profile = self.pipeline.start(config)

        for _ in range(30):
            self.pipeline.wait_for_frames()

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Could not get color frame from RealSense during init.")

        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        self.K_rgb = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def get_frame(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Could not get color frame from RealSense.")
        return np.asanyarray(color_frame.get_data()).copy()

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


# ============================================================
# Debug / print helpers
# ============================================================

def print_pose(label: str, pose: dict | None) -> None:
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    if pose is None:
        print("not available")
        return

    print(f"candidate_index = {pose['candidate_index']}")
    print(f"rvec = {pose['rvec']}")
    print(f"tvec [mm] = {pose['tvec']}")
    if pose["reproj_mean_px"] is not None:
        print(f"reproj_mean_px = {pose['reproj_mean_px']:.4f}")
    print("R_cx =")
    print(pose["R_cx"])

    R_xc, t_xc = invert_pose_cx_to_xc(pose["R_cx"], pose["tvec"])
    print("\nR_xc =")
    print(R_xc)
    print("t_xc [mm] =")
    print(t_xc)


def print_candidate_ray_info(name: str, pose: dict | None) -> None:
    if pose is None:
        print(f"[INFO] {name}: not available")
        return

    R_xc, t_xc = invert_pose_cx_to_xc(pose["R_cx"], pose["tvec"])
    origin_x, direction_x = make_principal_ray_x()
    origin_c, direction_c = transform_ray_x_to_c(origin_x, direction_x, R_xc, t_xc)

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)
    print("source_c [mm] =")
    print(origin_c)
    print("principal_ray_c =")
    print(direction_c)


# ============================================================
# Main
# ============================================================

def main() -> None:
    kx_path = _pick_npz_file("Select K_x NPZ")
    if kx_path is None:
        print("No K_x file selected.")
        return

    xyz_path = _pick_npz_file("Select XYZ camera NPZ")
    if xyz_path is None:
        print("No XYZ file selected.")
        return

    uv_path = _pick_npz_file("Select uv_xray NPZ")
    if uv_path is None:
        print("No uv_xray file selected.")
        return

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    K_x = load_kx(kx_path)
    xyz_mm = load_xyz_camera_mm(xyz_path)
    uv_xray = load_uv_xray(uv_path)

    if xyz_mm.shape[0] != uv_xray.shape[0]:
        raise ValueError(
            f"Point count mismatch: xyz={xyz_mm.shape[0]}, uv={uv_xray.shape[0]}"
        )

    print("\n=== K_x ===")
    print(K_x)

    origin_x, direction_x = make_principal_ray_x()

    print("\n=== Principal Ray (X-ray frame) ===")
    print("origin_x    =", origin_x)
    print("direction_x =", direction_x)

    # --------------------------------------------------------
    # Solve both candidates once
    # --------------------------------------------------------
    cand0, cand1 = solve_both_candidates(xyz_mm, uv_xray, K_x)

    print_pose("RESULTS FOR CANDIDATE 0", cand0)
    print_pose("RESULTS FOR CANDIDATE 1", cand1)

    print_candidate_ray_info("candidate 0", cand0)
    print_candidate_ray_info("candidate 1", cand1)

    # --------------------------------------------------------
    # Start live RealSense
    # --------------------------------------------------------
    cam = RealSenseRGB(width=1920, height=1080, fps=30)
    cam.start()
    K_rgb = cam.K_rgb

    print("\n=== K_rgb (from RealSense) ===")
    print(K_rgb)

    # Distances along the ray for live camera-image visualization
    lam1_mm = 500.0
    lam2_mm = 1000.0

    def build_candidate_geom(pose: dict | None) -> dict | None:
        if pose is None:
            return None

        R_xc, t_xc = invert_pose_cx_to_xc(pose["R_cx"], pose["tvec"])
        source_c, ray_dir_c = transform_ray_x_to_c(
            origin_x=origin_x,
            direction_x=direction_x,
            R_xc=R_xc,
            t_xc=t_xc,
        )

        p1_c, p2_c = make_two_points_on_ray(
            origin_c=source_c,
            direction_c=ray_dir_c,
            lam1_mm=lam1_mm,
            lam2_mm=lam2_mm,
        )

        return {
            "source_c": source_c,
            "ray_dir_c": ray_dir_c,
            "p1_c": p1_c,
            "p2_c": p2_c,
            "candidate_index": pose["candidate_index"],
            "reproj_mean_px": pose["reproj_mean_px"],
        }

    geom0 = build_candidate_geom(cand0)
    geom1 = build_candidate_geom(cand1)

    # --------------------------------------------------------
    # Precompute plane basis for top view
    # --------------------------------------------------------
    n_c, center_c = fit_plane_svd(xyz_mm)
    e_right_c, e_back_c = build_topview_basis_from_camera_axes(n_c)

    # Warp configuration
    out_w_px = 900
    out_h_px = 900
    mm_per_px = 1.5

    # --------------------------------------------------------
    # Live loop
    # --------------------------------------------------------
    win = "debug_pose_cam_xray"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    show_candidate_0 = True
    show_candidate_1 = True
    show_source_pts = True
    freeze = False

    last_raw_frame = None
    warped_mode = False
    warped_view = None

    try:
        while True:
            if not warped_mode:
                if not freeze or last_raw_frame is None:
                    frame = cam.get_frame()
                    last_raw_frame = frame.copy()
                else:
                    frame = last_raw_frame.copy()

                vis = frame.copy()
                H_img, W_img = vis.shape[:2]

                # RGB principal point
                cx_rgb = float(K_rgb[0, 2])
                cy_rgb = float(K_rgb[1, 2])
                pp_rgb = (int(round(cx_rgb)), int(round(cy_rgb)))
                draw_cross(vis, pp_rgb, (0, 255, 255), size=14, thickness=2)
                cv2.putText(
                    vis,
                    f"RGB pp ({cx_rgb:.1f}, {cy_rgb:.1f})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                def draw_candidate_live(geom: dict | None, color: tuple[int, int, int], label: str) -> None:
                    if geom is None:
                        return

                    uv1 = project_point(K_rgb, geom["p1_c"])
                    uv2 = project_point(K_rgb, geom["p2_c"])
                    if uv1 is not None and uv2 is not None:
                        clipped = clip_line_to_image(uv1, uv2, W_img, H_img)
                        if clipped is not None:
                            q1, q2 = clipped
                            cv2.line(vis, q1, q2, color, 2, cv2.LINE_AA)
                            draw_cross(vis, q1, color, size=9, thickness=2)
                            draw_cross(vis, q2, color, size=9, thickness=2)

                            mid = ((q1[0] + q2[0]) // 2, (q1[1] + q2[1]) // 2)
                            cv2.putText(
                                vis,
                                label,
                                (mid[0] + 8, mid[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                color,
                                2,
                                cv2.LINE_AA,
                            )

                    if show_source_pts:
                        uv_source = project_point(K_rgb, geom["source_c"])
                        if uv_source is not None:
                            p = to_int_pt(uv_source)
                            if 0 <= p[0] < W_img and 0 <= p[1] < H_img:
                                draw_circle(vis, p, color, radius=8, thickness=2)
                                cv2.putText(
                                    vis,
                                    f"{label} source",
                                    (p[0] + 10, p[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    color,
                                    2,
                                    cv2.LINE_AA,
                                )

                if show_candidate_0:
                    draw_candidate_live(geom0, (0, 255, 0), "candidate 0")
                if show_candidate_1:
                    draw_candidate_live(geom1, (0, 0, 255), "candidate 1")

                cv2.putText(
                    vis,
                    "[q] quit  [0] cand0  [1] cand1  [p] source pts  [f] freeze  [space] top view  [s] save raw",
                    (20, H_img - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    break
                elif key == ord("0"):
                    show_candidate_0 = not show_candidate_0
                elif key == ord("1"):
                    show_candidate_1 = not show_candidate_1
                elif key == ord("p"):
                    show_source_pts = not show_source_pts
                elif key == ord("f"):
                    freeze = not freeze
                elif key == ord("s"):
                    if last_raw_frame is not None:
                        out_path = save_raw_frame(last_raw_frame)
                        print(f"[INFO] Saved raw RGB frame to: {out_path}")
                elif key == 32:  # SPACE
                    if last_raw_frame is None:
                        continue

                    # ------------------------------------------------
                    # 1) warp ONLY the raw RGB frame
                    # ------------------------------------------------
                    corners_c = build_plane_patch_corners_mm(
                        center_c=center_c,
                        e_right_c=e_right_c,
                        e_back_c=e_back_c,
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
                    warp_unrot = cv2.warpPerspective(last_raw_frame, H_warp, (out_w_px, out_h_px))
                    warped_view = cv2.rotate(warp_unrot, cv2.ROTATE_90_CLOCKWISE)

                    H_rot, W_rot = warped_view.shape[:2]

                    # ------------------------------------------------
                    # 2) re-draw only plane-relevant result:
                    #    ray ∩ plane as point
                    # ------------------------------------------------
                    def draw_candidate_hit_topview(
                        img_top: np.ndarray,
                        geom: dict | None,
                        color: tuple[int, int, int],
                        label: str,
                    ) -> None:
                        if geom is None:
                            return

                        hit_c = intersect_ray_with_plane(
                            source_c=geom["source_c"],
                            ray_dir_c=geom["ray_dir_c"],
                            plane_normal_c=n_c,
                            plane_point_c=center_c,
                        )
                        if hit_c is None:
                            return

                        uv_hit_unrot = camera_point_to_topview_uv_unrotated(
                            hit_c, center_c, e_right_c, e_back_c, out_w_px, out_h_px, mm_per_px
                        )
                        uv_hit = rotate_uv_90_clockwise(uv_hit_unrot, out_w_px, out_h_px)
                        p = to_int_pt(uv_hit)

                        if 0 <= p[0] < W_rot and 0 <= p[1] < H_rot:
                            draw_cross(img_top, p, color, size=14, thickness=3)
                            cv2.putText(
                                img_top,
                                f"{label} hit",
                                (p[0] + 12, p[1] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                color,
                                2,
                                cv2.LINE_AA,
                            )

                    if show_candidate_0:
                        draw_candidate_hit_topview(warped_view, geom0, (0, 255, 0), "candidate 0")
                    if show_candidate_1:
                        draw_candidate_hit_topview(warped_view, geom1, (0, 0, 255), "candidate 1")

                    # clean text directly in new warped image
                    cv2.putText(
                        warped_view,
                        "Top view of plane (rotated 90 deg clockwise)",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        warped_view,
                        "Only ray-plane intersection points are drawn here",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        warped_view,
                        "[q] quit",
                        (20, warped_view.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    warped_mode = True

            else:
                cv2.imshow(win, warped_view)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()