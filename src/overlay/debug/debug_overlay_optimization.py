# -*- coding: utf-8 -*-
"""
debug_overlay_optimization.py

Offline re-rendering of a saved overlay preview, as the basis for evaluating
individual pipeline optimizations on the very same capture.

Stage 1 — baseline reproduction (this version)
----------------------------------------------
Reads data/Evaluation/overlay_preview_*.npz and re-renders the overlay through
the exact production path:

    T_xc, d_x  --estimate_plane_induced_homography-->  H_xc
    H_xc       --blend_xray_overlay-->                 overlay
    tip                                                drawn on top

The result is verified against the stored H_xc and overlay_bgr. Both must match
bit for bit; only then is the offline harness a valid reference.

Design
------
Every variant plugs in at exactly ONE point: it returns (T_xc, d_x). The render
path is shared and untouched, so any visible difference between two variants is
attributable to the pose, never to the rendering.

To add an optimization later, write a function

    def variant_xyz(data: EvalData) -> tuple[np.ndarray, float]: ...

and register it in VARIANTS. Nothing else in this file needs to change.

Grid layout
-----------
xray_points_xyz_c / xray_points_uv are row-major 11x11:
TL = index 0, TR = index 10, BL = index 110.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from overlay.tools.homography import (
    estimate_plane_induced_homography,
    build_board_xyz_canonical,
    estimate_homography,
    project_homography,
)
from overlay.tools.checkerboard_corner_detection import get_extreme_corners_geometric
from overlay.tools.ransac_plane_fitting import (
    intersect_pixels_with_plane,
    interpolate_marker_grid,
)
from overlay.tools.warp import blend_xray_overlay, _detect_xray_fov_mask
from overlay.calib.calib_camera_to_xray import calibrate_camera_to_xray
from overlay.gui.pages.page_camera_to_xray_calibration import PCB_THICKNESS_MM

# Private solver internals. The production ippe_handeye path is replicated
# below rather than modified, so this script never changes pipeline behaviour.
from tracking.pose_solvers import (
    normalize_dist_coeffs,
    _rigid_fit_kabsch,
    _trust_region_from_depth_rms,
    _build_ippe_candidates,
    _select_ippe_candidate,
    _select_ippe_candidate_rgb,
    _refine_pose_iterative,
    _make_transform,
)
from tracking.transforms import invert_transform

# Imported rather than re-implemented so the tip marker cannot drift away
# from the production rendering.
from overlay.gui.pages.page_overlay_preview import _draw_point


# ============================================================
# Config
# ============================================================

DEFAULT_EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "Evaluation"

PITCH_MM = 2.54
STEPS_PER_EDGE = 10
NOMINAL_EDGE_MM = PITCH_MM * STEPS_PER_EDGE  # 25.4 mm

# Blend convention (overlay/tools/warp.py):
#     blended = alpha * camera + (1 - alpha) * xray
# so alpha = 1.0 is pure RGB and alpha = 0.0 is pure X-ray.
#
# For judging pin-vs-housing alignment on the PCB header test object:
#   1.00  pure RGB       -> reference: where the plastic housings actually are
#   0.35  balanced       -> housings AND pins both carry contrast; this is the
#                           only level at which the offset is directly readable
#   0.07  X-ray dominant -> pins sharp, housings washed out
EXTRA_ALPHAS: tuple[float, ...] = (1.00, 0.35, 0.07)

# Zoom crop around the warped X-ray FOV, so the offsets are judgeable at all.
ZOOM_SCALE = 3
ZOOM_MARGIN_PX = 20

# d_x sweep, in mm relative to the stored d_x.
#
# H_xc is exact for points on the plane z_x = d_x. The calibration markers are
# coplanar and so are the test object's pins - but d_x comes from the pointer
# tip, i.e. from whichever plane was touched, not from the pin plane. Sweeping
# d_x separates the two possible causes of a residual offset:
#
#   - one common d_x centres ALL headers  -> plane offset only, calibration fine
#   - no single d_x does                  -> genuine calibration error remains
DX_SWEEP_MM: tuple[float, ...] = (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0)
DX_SWEEP_ALPHA = 0.35

IDX_TL = 0
IDX_TR = STEPS_PER_EDGE
IDX_BL = STEPS_PER_EDGE * (STEPS_PER_EDGE + 1)


# ============================================================
# Data container
# ============================================================

@dataclass(frozen=True)
class EvalData:
    """Everything the offline harness needs, loaded from one overlay NPZ."""

    path: Path

    # images
    snapshot_rgb_bgr: np.ndarray
    xray_gray_working_u8: np.ndarray      # exactly as stored
    xray_working_pristine: np.ndarray     # rebuilt from the raw image
    overlay_bgr: np.ndarray

    # production overlay parameters
    H_xc: np.ndarray
    alpha: float
    d_x: float

    # calibration / pose
    K_rgb: np.ndarray
    dist_rgb: np.ndarray | None
    K_xray: np.ndarray
    T_cx: np.ndarray
    T_xc: np.ndarray

    # correspondences
    xray_points_uv: np.ndarray
    xray_points_xyz_c: np.ndarray
    corners_uv_3: np.ndarray
    corners_uv_9: np.ndarray | None

    # pointer
    tip_uv_c: np.ndarray
    tip_xyz_c: np.ndarray          # mm, camera frame


def _optional(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    """Return None for keys that were saved while the state field was None."""
    if key not in data.files:
        return None
    arr = data[key]
    if arr.dtype == object and arr.shape == ():
        return None if arr.item() is None else np.asarray(arr.item(), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def load_eval_npz(path: Path) -> EvalData:
    d = np.load(str(path), allow_pickle=True)

    return EvalData(
        path=path,
        snapshot_rgb_bgr=d["snapshot_rgb_bgr"],
        xray_gray_working_u8=d["xray_gray_working_u8"],
        # A capture made after the X-ray undistortion shipped stores an
        # ALREADY undistorted working image. Variants that undistort would
        # then correct twice, so they start from the raw image instead.
        xray_working_pristine=np.ascontiguousarray(
            cv2.flip(np.asarray(d["xray_gray_u8"]), 1)),
        overlay_bgr=d["overlay_bgr"],
        H_xc=np.asarray(d["H_xc"], dtype=np.float64),
        alpha=float(d["alpha"]),
        d_x=float(d["d_x"]),
        K_rgb=np.asarray(d["K_rgb"], dtype=np.float64),
        dist_rgb=_optional(d, "dist_rgb"),
        K_xray=np.asarray(d["K_xray"], dtype=np.float64),
        T_cx=np.asarray(d["T_cx"], dtype=np.float64),
        T_xc=np.asarray(d["T_xc"], dtype=np.float64),
        xray_points_uv=np.asarray(d["xray_points_uv"], dtype=np.float64).reshape(-1, 2),
        xray_points_xyz_c=np.asarray(d["xray_points_xyz_c"], dtype=np.float64).reshape(-1, 3),
        corners_uv_3=np.asarray(d["checkerboard_corners_uv"], dtype=np.float64).reshape(3, 2),
        corners_uv_9=_optional(d, "checkerboard_corners_uv_9"),
        tip_uv_c=np.asarray(d["tip_uv_c"], dtype=np.float64).reshape(2),
        tip_xyz_c=np.asarray(d["tip_xyz_c"], dtype=np.float64).reshape(3),
    )


def find_latest_npz(directory: Path) -> Path:
    candidates = sorted(directory.glob("overlay_preview_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No overlay_preview_*.npz found in {directory}")
    return candidates[-1]


# ============================================================
# Shared render path — identical for every variant
# ============================================================

def render_overlay(
    data: EvalData,
    T_xc: np.ndarray,
    d_x: float,
    alpha: float,
    *,
    draw_tip: bool = True,
    use_dist: bool = False,
    undistort_xray: bool = False,
    brown_k1: bool = False,
    as_stored: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render one overlay. This mirrors PlaneOverlayPreviewPage exactly:
    _compute_H_xc -> blend_xray_overlay -> _draw_tip_on_top.

    use_dist=True additionally models the RGB lens distortion in the warp.
    H_xc maps X-ray pixels to IDEAL pinhole camera pixels, but the camera
    image is distorted, so the production warp places the X-ray content at
    positions the real camera never images. See warp_with_distortion.

    Returns (overlay_bgr, H_xc, overlay_mask).
    """
    T_xc = np.asarray(T_xc, dtype=np.float64)

    H_xc = estimate_plane_induced_homography(
        K_c=data.K_rgb,
        R_xc=T_xc[:3, :3],
        t_xc=T_xc[:3, 3],
        K_x=data.K_xray,
        d_x=float(d_x),
    )

    xray_img = (data.xray_gray_working_u8 if as_stored
                else data.xray_working_pristine)
    if undistort_xray and XRAY_DIST is not None:
        xray_img = undistort_xray_image_working(xray_img)

    # use_dist selects the production warp's distortion-aware branch
    out_bgr, cache = blend_xray_overlay(
        camera_bgr=data.snapshot_rgb_bgr,
        xray_gray_u8=xray_img,
        H_xc=H_xc,
        alpha=float(alpha),
        K_camera=data.K_rgb if use_dist else None,
        dist_coeffs=data.dist_rgb if use_dist else None,
    )

    if draw_tip:
        out_bgr = _draw_point(
            out_bgr,
            data.tip_uv_c,
            color=(0, 0, 255),
            radius=8,
            cross_size=20,
            thickness=2,
        )

    return out_bgr, H_xc, cache.overlay_mask


def warp_and_blend_distorted(
    data: EvalData, H_xc: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp the X-ray through H_xc AND the RGB distortion model.

    Chain:  xray px --H_xc--> ideal camera px --distort--> observed camera px

    The remap runs backwards: for every observed output pixel, undistort it to
    the ideal pixel, then apply H_xc^-1 to find the X-ray source pixel.
    Everything else (FOV mask, blend) matches overlay/tools/warp.py.
    """
    xray = data.xray_gray_working_u8
    cam = data.snapshot_rgb_bgr
    h, w = cam.shape[:2]

    fov = _detect_xray_fov_mask(xray)
    masked = cv2.bitwise_and(xray, xray, mask=fov)

    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    obs = np.stack([uu.ravel(), vv.ravel()], axis=1).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(obs, data.K_rgb, data.dist_rgb, P=data.K_rgb).reshape(-1, 2)

    q = (np.linalg.inv(H_xc) @ np.hstack([ideal, np.ones((len(ideal), 1))]).T).T
    src = q[:, :2] / q[:, 2:3]
    map_x = np.ascontiguousarray(src[:, 0].reshape(h, w).astype(np.float32))
    map_y = np.ascontiguousarray(src[:, 1].reshape(h, w).astype(np.float32))

    warped = cv2.remap(masked, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.remap(fov, map_x, map_y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = (mask > 0).astype(np.uint8) * 255

    # identical to warp.WarpedOverlay.blend
    out = cam.copy()
    sel = mask > 0
    if np.any(sel):
        xray_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        blended = alpha * cam.astype(np.float32) + (1 - alpha) * xray_bgr.astype(np.float32)
        out[sel] = np.clip(blended, 0, 255).astype(np.uint8)[sel]

    return out, mask


def recompute_grid_with_distortion(data: EvalData) -> np.ndarray:
    """
    Redo the plane-fitting ray/plane intersection with the distortion model.

    _plane_to_grid passes dist_coeffs=None (page_plane_fitting.py:925), so the
    three corner rays are built from a pure pinhole model. The fitted plane
    itself comes from the depth deprojection and is unaffected, and it is
    exactly recoverable from the stored grid because those 121 points are
    coplanar by construction.
    """
    xyz = data.xray_points_xyz_c
    centroid = xyz.mean(axis=0)
    normal = np.linalg.svd(xyz - centroid)[2][2]
    normal = normal / np.linalg.norm(normal)
    plane = np.array([normal[0], normal[1], normal[2],
                      -float(normal @ centroid)], dtype=np.float64)

    corner_xyz = intersect_pixels_with_plane(
        data.corners_uv_3, data.K_rgb, plane, data.dist_rgb
    )
    return interpolate_marker_grid(corner_xyz, steps_per_edge=STEPS_PER_EDGE)


def roi_from_mask(mask: np.ndarray, margin: int = ZOOM_MARGIN_PX) -> tuple[int, int, int, int]:
    """Bounding box (u0, v0, u1, v1) of the warped X-ray FOV, padded by margin."""
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Overlay mask is empty — nothing was warped into the camera view.")

    h, w = mask.shape[:2]
    u0 = max(0, int(xs.min()) - margin)
    v0 = max(0, int(ys.min()) - margin)
    u1 = min(w, int(xs.max()) + margin + 1)
    v1 = min(h, int(ys.max()) + margin + 1)
    return u0, v0, u1, v1


def crop_zoom(img: np.ndarray, roi: tuple[int, int, int, int], scale: int = ZOOM_SCALE) -> np.ndarray:
    u0, v0, u1, v1 = roi
    crop = img[v0:v1, u0:u1]
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


# ============================================================
# Variants — each returns (T_xc, d_x)
# ============================================================

def render_and_save(
    data: EvalData,
    name: str,
    T_xc: np.ndarray,
    d_x: float,
    alphas: tuple[float, ...],
    out_dir: Path,
    roi: tuple[int, int, int, int],
    brown_k1: bool = False,
    as_stored: bool = False,
) -> dict[float, np.ndarray]:
    """
    Render one variant at several blend levels and write full + zoomed PNGs.

    The ROI is passed in rather than derived per variant, so the zoom crops of
    different variants stay pixel-comparable.
    """
    rendered: dict[float, np.ndarray] = {}
    use_dist = name in DISTORTED_WARP
    undist_x = name in UNDISTORT_XRAY

    for a in alphas:
        img, _, _ = render_overlay(data, T_xc, d_x, a, use_dist=use_dist,
                                   undistort_xray=undist_x, brown_k1=brown_k1,
                                   as_stored=as_stored)
        cv2.imwrite(str(out_dir / f"overlay_{name}_alpha{a:.2f}.png"), img)
        cv2.imwrite(str(out_dir / f"zoom_{name}_alpha{a:.2f}.png"), crop_zoom(img, roi))
        rendered[a] = img

    return rendered


def render_dx_sweep(
    data: EvalData,
    T_xc: np.ndarray,
    d_x0: float,
    offsets_mm: tuple[float, ...],
    out_dir: Path,
    roi: tuple[int, int, int, int],
) -> None:
    """
    Render the same calibration at several plane depths.

    Pose and intrinsics stay fixed; only d_x moves. Any difference between the
    frames is therefore pure plane-depth effect.
    """
    sweep_dir = out_dir / "dx_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 68)
    print("d_x sweep")
    print("-" * 68)

    ref = None

    for off in offsets_mm:
        d_x = d_x0 + off
        img, _, _ = render_overlay(data, T_xc, d_x, DX_SWEEP_ALPHA)

        tag = f"{off:+.1f}mm".replace("+", "p").replace("-", "m").replace(".", "_")
        cv2.imwrite(str(sweep_dir / f"zoom_dx_{tag}.png"), crop_zoom(img, roi))

        if off == 0.0:
            ref = img

        # image-space shift the plane change induces, measured on the X-ray FOV
        shift = plane_shift_px(data, T_xc, d_x0, d_x)
        print(f"  d_x = {d_x:8.3f} mm ({off:+5.1f})   "
              f"overlay moves {shift['mean']:6.3f} px  "
              f"[{shift['du']:+.3f} {shift['dv']:+.3f}]")

    if ref is not None:
        cv2.imwrite(str(sweep_dir / "zoom_dx_p0_0mm_reference.png"), crop_zoom(ref, roi))

    print(f"\n  => frames in {sweep_dir}\n")


def plane_shift_px(
    data: EvalData,
    T_xc: np.ndarray,
    d_x_from: float,
    d_x_to: float,
) -> dict:
    """Mean image-space displacement of the warped X-ray content when d_x changes."""
    R_xc = T_xc[:3, :3]
    t_xc = T_xc[:3, 3]

    h, w = data.xray_gray_working_u8.shape[:2]
    su, sv = np.meshgrid(
        np.linspace(0.25 * w, 0.75 * w, 3),
        np.linspace(0.25 * h, 0.75 * h, 3),
    )
    pts = np.stack([su.ravel(), sv.ravel(), np.ones(su.size)], axis=1)

    def project(d_x):
        H = estimate_plane_induced_homography(
            K_c=data.K_rgb, R_xc=R_xc, t_xc=t_xc,
        K_x=(XRAY_K1_K if brown_k1 else data.K_xray), d_x=float(d_x)
        )
        q = (H @ pts.T).T
        return q[:, :2] / q[:, 2:3]

    delta = project(d_x_to) - project(d_x_from)
    return {
        "mean": float(np.linalg.norm(delta, axis=1).mean()),
        "du": float(delta[:, 0].mean()),
        "dv": float(delta[:, 1].mean()),
    }


# ============================================================
# Replication of the production ippe_handeye solver
#
# tracking.pose_solvers._solve_pose_ippe_handeye derives the 121 RGB grid
# points internally, so the interpolation cannot be swapped through the public
# API. The flow is replicated here with the grid injected instead. Production
# code stays untouched; variant "recomputed" proves the replication is exact.
# ============================================================

def rgb_grid_affine_3corners(data: EvalData, steps: int = STEPS_PER_EDGE) -> np.ndarray:
    """What production does: affine interpolation in pixel space from TL/TR/BL."""
    p_tl, p_tr, p_bl = data.corners_uv_3
    step_x = (p_tr - p_tl) / float(steps)
    step_y = (p_bl - p_tl) / float(steps)
    return np.array(
        [p_tl + a * step_x + b * step_y
         for b in range(steps + 1) for a in range(steps + 1)],
        dtype=np.float64,
    )


def order_corners_9(corners_9: np.ndarray) -> np.ndarray:
    """
    Assign board indices (row_c, col_c) to the 9 detected corners.

    overlay.tools.checkerboard_corner_detection._sort_corners_row_major cannot be
    used here: its np.lexsort has v as the primary key, so the secondary u key
    only takes effect on exactly equal v. With subpixel corners that never
    happens, and on a slightly rotated board the within-row order comes out
    scrambled (here: column-mirrored).

    Instead the corners are placed in the affine frame spanned by the geometric
    extremes TL/TR/BL - the same frame the production affine path uses - so the
    indexing is guaranteed consistent with the board points and the X-ray grid.

    Returns (9, 2) ordered row-major: index r*3 + c.
    """
    pts = np.asarray(corners_9, dtype=np.float64).reshape(9, 2)
    ex = get_extreme_corners_geometric(pts.reshape(-1, 1, 2).astype(np.float32))
    p_tl = np.asarray(ex["top_left"], dtype=np.float64).ravel()
    p_tr = np.asarray(ex["top_right"], dtype=np.float64).ravel()
    p_bl = np.asarray(ex["bottom_left"], dtype=np.float64).ravel()

    # 3x3 pattern -> 2 steps between the extremes
    basis = np.column_stack([(p_tr - p_tl) / 2.0, (p_bl - p_tl) / 2.0])
    ab = np.linalg.lstsq(basis, (pts - p_tl).T, rcond=None)[0].T   # (9, 2)
    idx = np.rint(ab).astype(int)

    if idx.min() < 0 or idx.max() > 2:
        raise RuntimeError(f"corner indices out of range:\n{idx}")
    keys = {(int(r), int(c)) for c, r in idx}
    if len(keys) != 9:
        raise RuntimeError(f"corner indexing is not a bijection:\n{idx}")

    out = np.zeros((9, 2), dtype=np.float64)
    for p, (c, r) in zip(pts, idx):
        out[r * 3 + c] = p
    return out


def rgb_grid_homography_9corners(data: EvalData, steps: int = STEPS_PER_EDGE) -> np.ndarray:
    """The improvement: homography through all 9 detected corners."""
    if data.corners_uv_9 is None:
        raise RuntimeError("checkerboard_corners_uv_9 missing from the NPZ.")

    corners = order_corners_9(data.corners_uv_9)
    corner_step = steps // 2                       # 3x3 pattern spans the grid

    board_xy = np.array(
        [[c * corner_step * PITCH_MM, r * corner_step * PITCH_MM]
         for r in range(3) for c in range(3)],
        dtype=np.float64,
    )
    H = estimate_homography(uv_img=corners, XY_grid=board_xy, method="dlt")

    all_xy = np.array(
        [[j * PITCH_MM, i * PITCH_MM]
         for i in range(steps + 1) for j in range(steps + 1)],
        dtype=np.float64,
    )
    return project_homography(H, all_xy)


# ============================================================
# X-ray detector distortion
#
# The image intensifier has pincushion + S-distortion that nothing in the
# pipeline models: _solve_pose_ippe_handeye hardcodes dist_xray = None. The
# calibration hole grid is an exact 2.54 mm lattice, so its undistorted image
# must be a pure homography of that lattice - any deviation IS the distortion.
# ============================================================

DIST_DEGREE = 4
XRAY_WIDTH = 1024

# Brown model for the X-ray side, from estimate_intrinsics_from_homographies
# with radial_model="k1" on the 10 views of session 04202026 (rms 0.6935 px).
# k1 was estimated jointly with f, so the two belong together and the shipped
# f = 4586.31 must NOT be combined with this k1.
#
# A radial model about the image centre is invariant under the RAW<->WORKING
# horizontal flip (u_work = W-1-u_raw maps 512 onto 511), so no space
# conversion is needed here.
# NEGATIVE RESULT - kept so the finding is documented.
#
# This is the only identifiable (f, k1) pair available: debug_hom_intrin's BA
# re-run over the same 12 views of 04222026 with CALIB_FIX_K1 released
# (see debug_xray_intrinsics_radial.py, control reproduces the shipped f to
# 0.003 px). Releasing k1 moves f by 831 px (18 %) for a 5.7 % rms gain - the
# board is planar and the views share similar distance and angulation, so f
# and k1 run along a nearly flat valley and are not separable.
#
# Applied to the overlay it is catastrophic: d_x jumps by ~90 mm and the
# header offsets go to 6-7 px against 2.2-4.3 px for the baseline.
#
# The deeper reason Brown cannot work here: in the gauge that matters the
# field's dominant part is a LINEAR radial term (+1.5 % scale), and Brown's
# model has only r^2, r^4, r^6. Any k1 that matches the shape drags in a wrong
# scale. Use the measured polynomial field instead - variant xray_pose.
XRAY_K1_K = np.array([[5417.9777, 0.0, 512.0],
                      [0.0, 5417.9777, 512.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
XRAY_K1_DIST = np.array([2.533609, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _poly_features(uv: np.ndarray, centre: np.ndarray, scale: float,
                   degree: int = DIST_DEGREE) -> np.ndarray:
    x = (uv[:, 0] - centre[0]) / scale
    y = (uv[:, 1] - centre[1]) / scale
    cols = [x ** i * y ** j
            for d in range(degree + 1) for i in range(d + 1) for j in [d - i]]
    return np.column_stack(cols)


class XrayDistortion:
    """
    Polynomial map from distorted to ideal X-ray pixels.

    Only the forward direction is fitted. The inverse is obtained by fixed
    point iteration on  x = y - poly(x), which is exact to machine precision
    for a displacement field this small - fitting a second polynomial for the
    inverse leaves a round-trip error of several tenths of a pixel, and that
    error goes straight into the image remap.
    """

    def __init__(self, distorted: np.ndarray, ideal: np.ndarray,
                 degree: int = DIST_DEGREE):
        self.centre = np.array([XRAY_WIDTH / 2.0, XRAY_WIDTH / 2.0])
        self.scale = XRAY_WIDTH / 2.0
        self.degree = int(degree)

        A = _poly_features(distorted, self.centre, self.scale, self.degree)
        self.fwd = np.linalg.lstsq(A, ideal - distorted, rcond=None)[0]

        res = distorted + A @ self.fwd - ideal
        self.fit_rms = float(np.sqrt((res ** 2).sum(axis=1).mean()))
        self.n_points = len(distorted)
        self.n_terms = A.shape[1]
        self.raw_rms = float(np.sqrt(((ideal - distorted) ** 2).sum(axis=1).mean()))
        self.raw_max = float(np.sqrt(((ideal - distorted) ** 2).sum(axis=1)).max())

    def undistort(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        return uv + _poly_features(uv, self.centre, self.scale, self.degree) @ self.fwd

    def distort(self, uv: np.ndarray, iters: int = 12) -> np.ndarray:
        target = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        x = target.copy()
        for _ in range(iters):
            x = target - _poly_features(x, self.centre, self.scale, self.degree) @ self.fwd
        return x

    def round_trip_error(self, uv: np.ndarray) -> float:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        return float(np.abs(self.distort(self.undistort(uv)) - uv).max())


def _to_raw(uv_work: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv_work, dtype=np.float64).reshape(-1, 2)
    return np.column_stack([XRAY_WIDTH - 1 - uv[:, 0], uv[:, 1]])


def fit_xray_distortion(eval_dir: Path) -> XrayDistortion | None:
    """
    Build the distortion model from a saved marker-selection run.

    Needs circles_uvr (all detections, RAW coordinates) - enable
    SAVE_XRAY_MARKER_SELECTION to produce it.
    """
    files = sorted(eval_dir.glob("xray_marker_selection_*.npz"))
    src = None
    for f in files:
        d = np.load(str(f), allow_pickle=True)
        if "circles_uvr" in d.files and len(d["circles_uvr"]) > 200:
            src = d
            break
    if src is None:
        print("[INFO] no marker-selection NPZ with circles_uvr found "
              "-> X-ray distortion variants skipped")
        return None

    circles = np.asarray(src["circles_uvr"], dtype=np.float64)[:, :2]
    sel_raw = np.asarray(src["points_uv_raw"], dtype=np.float64).reshape(-1, 2)

    board121 = np.array([[j * PITCH_MM, i * PITCH_MM] for i in range(11) for j in range(11)],
                        dtype=np.float64)
    H, _ = cv2.findHomography(board121.reshape(-1, 1, 2), sel_raw.reshape(-1, 1, 2), 0)

    ok = np.zeros(len(circles), bool)
    lat = np.zeros((len(circles), 2))
    for _ in range(6):
        b = cv2.perspectiveTransform(
            circles.reshape(-1, 1, 2), np.linalg.inv(H)
        ).reshape(-1, 2) / PITCH_MM
        lat = np.rint(b)
        err = np.linalg.norm(b - lat, axis=1)
        ok = np.zeros(len(circles), bool)
        seen = set()
        for t in np.argsort(err):
            if err[t] >= 0.30:
                continue
            key = (int(lat[t, 0]), int(lat[t, 1]))
            if key in seen:
                continue
            seen.add(key)
            ok[t] = True
        H, _ = cv2.findHomography((lat[ok] * PITCH_MM).reshape(-1, 1, 2),
                                  circles[ok].reshape(-1, 1, 2), 0)

    ideal = cv2.perspectiveTransform(
        (lat[ok] * PITCH_MM).reshape(-1, 1, 2), H
    ).reshape(-1, 2)

    # pick the degree by cross-validation so a higher order cannot simply
    # fit the detection noise
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ideal))
    cut = int(0.75 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    print("  X-ray distortion, degree selection (75/25 split):")
    best, best_err = DIST_DEGREE, np.inf
    for deg in (3, 4, 5, 6, 7):
        m = XrayDistortion(circles[ok][tr], ideal[tr], degree=deg)
        r = m.undistort(circles[ok][te]) - ideal[te]
        held = float(np.sqrt((r ** 2).sum(axis=1).mean()))
        print(f"    degree {deg}: {m.n_terms:2d} terms/axis   "
              f"train {m.fit_rms:.3f} px   held-out {held:.3f} px")
        if held < best_err:
            best, best_err = deg, held

    print(f"    -> degree {best}")
    return XrayDistortion(circles[ok], ideal, degree=best)


def solve_ippe_handeye(
    data: EvalData,
    pts2d_rgb: np.ndarray,
    *,
    steps_per_edge: int = STEPS_PER_EDGE,
    refine: bool = True,
    uv_xray_override: np.ndarray | None = None,
    dist_rgb: np.ndarray | None = None,
    cam_xyz_override: np.ndarray | None = None,
    board_offset_mm: float = 0.0,
    K_xray_override: np.ndarray | None = None,
    dist_xray_override: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return T_cx for the given RGB grid. Mirrors the production solver.

    board_offset_mm lifts the RGB-side board plane above the X-ray-side one.
    The calibration target is a 1.6 mm single-sided PCB: the checkerboard is on
    the top face, the X-ray-visible hole is defined by the copper ring on the
    bottom face. Production puts both at Z = 0, so the thickness is absorbed
    into T_cx as a translation along the board normal.
    """
    uv_xray = (data.xray_points_uv if uv_xray_override is None
               else uv_xray_override).reshape(-1, 2)
    cam_xyz = data.xray_points_xyz_c if cam_xyz_override is None else cam_xyz_override
    cam_xyz_mm = np.asarray(cam_xyz, dtype=np.float64).reshape(-1, 3) * 1000.0

    # production passes dist_coeffs_rgb=None even though dist_rgb exists
    dist_xray = normalize_dist_coeffs(dist_xray_override)
    dist_rgb = normalize_dist_coeffs(dist_rgb)
    K_xray = data.K_xray if K_xray_override is None else np.asarray(
        K_xray_override, dtype=np.float64)

    board_xray = build_board_xyz_canonical(
        nu=steps_per_edge, nv=steps_per_edge, pitch_mm=PITCH_MM
    )
    board_mm = board_xray.copy()
    board_mm[:, 2] += float(board_offset_mm)

    depth_fit = _rigid_fit_kabsch(board_mm, cam_xyz_mm)
    tr = _trust_region_from_depth_rms(depth_fit["rms_mm"])

    cand_rgb = _build_ippe_candidates(board_mm, pts2d_rgb, data.K_rgb, dist_rgb)
    i_rgb = _select_ippe_candidate_rgb(
        cand_rgb,
        T_bc_depth=depth_fit["T"],
        gamma_t_mm=tr["gamma_t_mm"],
        gamma_r_deg=tr["gamma_r_deg"],
    )
    rvec_bc = np.asarray(cand_rgb[i_rgb].rvec, dtype=np.float64).reshape(3, 1)
    tvec_bc = np.asarray(cand_rgb[i_rgb].tvec, dtype=np.float64).reshape(3, 1)
    if refine:
        rvec_bc, tvec_bc = _refine_pose_iterative(
            board_mm, pts2d_rgb, data.K_rgb, dist_rgb, rvec_bc, tvec_bc
        )
    T_bc = _make_transform(cv2.Rodrigues(rvec_bc)[0], tvec_bc.ravel())

    cand_x = _build_ippe_candidates(board_mm, uv_xray, K_xray, dist_xray)
    i_x = _select_ippe_candidate(cand_x, use_xray_ippe_selection_rule=True)
    rvec_bx = np.asarray(cand_x[i_x].rvec, dtype=np.float64).reshape(3, 1)
    tvec_bx = np.asarray(cand_x[i_x].tvec, dtype=np.float64).reshape(3, 1)
    if refine:
        rvec_bx, tvec_bx = _refine_pose_iterative(
            board_mm, uv_xray, K_xray, dist_xray, rvec_bx, tvec_bx
        )
    T_bx = _make_transform(cv2.Rodrigues(rvec_bx)[0], tvec_bx.ravel())

    T_cx = T_bx @ invert_transform(T_bc)
    T_cx[:3, 3] *= 1e-3
    return T_cx


def d_x_from_pose(T_cx: np.ndarray, tip_xyz_c_mm: np.ndarray) -> float:
    """
    d_x is the pointer tip depth along the X-ray axis, so it is a function of
    the pose. Changing T_cx without recomputing it would mix two effects.
    """
    tip_m = np.asarray(tip_xyz_c_mm, dtype=np.float64).reshape(3) / 1000.0
    p = T_cx @ np.array([tip_m[0], tip_m[1], tip_m[2], 1.0])
    return float(p[2] * 1000.0)


def _pose_delta(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    dR = T_a[:3, :3] @ T_b[:3, :3].T
    ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))))
    dt = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) * 1000.0)
    return ang, dt


# ============================================================
# Variants — each returns (T_xc, d_x)
# ============================================================

def variant_baseline(data: EvalData) -> tuple[np.ndarray, float]:
    """Production result, taken verbatim from the NPZ."""
    return data.T_xc, data.d_x


def variant_recomputed(data: EvalData) -> tuple[np.ndarray, float]:
    """Control: production inputs through the replicated solver. Must match."""
    T_cx = solve_ippe_handeye(data, rgb_grid_affine_3corners(data))
    return invert_transform(T_cx), d_x_from_pose(T_cx, data.tip_xyz_c)


def variant_corners9(data: EvalData) -> tuple[np.ndarray, float]:
    """Improvement: 9 real corners + homography instead of 3 corners + affine."""
    T_cx = solve_ippe_handeye(data, rgb_grid_homography_9corners(data))
    return invert_transform(T_cx), d_x_from_pose(T_cx, data.tip_xyz_c)


def solve_with_board_offset(data: EvalData, offset_mm: float) -> np.ndarray:
    """
    Same ippe_handeye flow, but the RGB side sits on a plane offset_mm above
    the X-ray side.

    The calibration board is a 1.6 mm PCB: the checkerboard is printed on the
    top face, while the X-ray-visible feature is the hole in the copper on the
    bottom face. build_board_xyz_canonical puts both at Z = 0, so the board
    thickness is absorbed into T_cx as a translation along the board normal.

    offset_mm > 0 places the checkerboard above the copper plane.
    """
    steps = STEPS_PER_EDGE
    uv_xray = data.xray_points_uv.reshape(-1, 2)
    cam_xyz_mm = data.xray_points_xyz_c.reshape(-1, 3) * 1000.0

    dist_xray = normalize_dist_coeffs(None)
    dist_rgb = normalize_dist_coeffs(None)

    board_xray = build_board_xyz_canonical(nu=steps, nv=steps, pitch_mm=PITCH_MM)
    board_rgb = board_xray.copy()
    board_rgb[:, 2] += float(offset_mm)

    pts2d_rgb = rgb_grid_affine_3corners(data)

    depth_fit = _rigid_fit_kabsch(board_rgb, cam_xyz_mm)
    tr = _trust_region_from_depth_rms(depth_fit["rms_mm"])

    cand_rgb = _build_ippe_candidates(board_rgb, pts2d_rgb, data.K_rgb, dist_rgb)
    i_rgb = _select_ippe_candidate_rgb(
        cand_rgb, T_bc_depth=depth_fit["T"],
        gamma_t_mm=tr["gamma_t_mm"], gamma_r_deg=tr["gamma_r_deg"],
    )
    rvec_bc = np.asarray(cand_rgb[i_rgb].rvec, dtype=np.float64).reshape(3, 1)
    tvec_bc = np.asarray(cand_rgb[i_rgb].tvec, dtype=np.float64).reshape(3, 1)
    rvec_bc, tvec_bc = _refine_pose_iterative(
        board_rgb, pts2d_rgb, data.K_rgb, dist_rgb, rvec_bc, tvec_bc
    )
    T_bc = _make_transform(cv2.Rodrigues(rvec_bc)[0], tvec_bc.ravel())

    cand_x = _build_ippe_candidates(board_xray, uv_xray, data.K_xray, dist_xray)
    i_x = _select_ippe_candidate(cand_x, use_xray_ippe_selection_rule=True)
    rvec_bx = np.asarray(cand_x[i_x].rvec, dtype=np.float64).reshape(3, 1)
    tvec_bx = np.asarray(cand_x[i_x].tvec, dtype=np.float64).reshape(3, 1)
    rvec_bx, tvec_bx = _refine_pose_iterative(
        board_xray, uv_xray, data.K_xray, dist_xray, rvec_bx, tvec_bx
    )
    T_bx = _make_transform(cv2.Rodrigues(rvec_bx)[0], tvec_bx.ravel())

    T_cx = T_bx @ invert_transform(T_bc)
    T_cx[:3, 3] *= 1e-3
    return T_cx


def _board_offset_variant(offset_mm: float):
    def fn(data: EvalData) -> tuple[np.ndarray, float]:
        T_cx = solve_with_board_offset(data, offset_mm)
        return invert_transform(T_cx), d_x_from_pose(T_cx, data.tip_xyz_c)
    return fn


XRAY_DIST: XrayDistortion | None = None       # filled in main()


def undistort_xray_points_working(uv_work: np.ndarray) -> np.ndarray:
    """
    Undistort X-ray points given in the WORKING (flipped) space.

    The distortion model lives in RAW detector coordinates - that is where the
    physics is defined - so flip in, correct, flip back.
    """
    ideal_raw = XRAY_DIST.undistort(_to_raw(uv_work))
    return np.column_stack([XRAY_WIDTH - 1 - ideal_raw[:, 0], ideal_raw[:, 1]])


def undistort_xray_image_working(img: np.ndarray) -> np.ndarray:
    """Undistort a WORKING-space X-ray image, so it matches the corrected points."""
    h, w = img.shape[:2]
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    ideal_work = np.stack([uu.ravel(), vv.ravel()], axis=1)
    src_raw = XRAY_DIST.distort(_to_raw(ideal_work))
    src_work = np.column_stack([XRAY_WIDTH - 1 - src_raw[:, 0], src_raw[:, 1]])
    return cv2.remap(
        img,
        np.ascontiguousarray(src_work[:, 0].reshape(h, w).astype(np.float32)),
        np.ascontiguousarray(src_work[:, 1].reshape(h, w).astype(np.float32)),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def variant_xray_pose(data: EvalData) -> tuple[np.ndarray, float]:
    """
    Production, plus the X-ray distortion taken into account in T_bx.

    Only the pose changes: the 121 marker points are undistorted before the
    X-ray-side IPPE. The warped image itself is still the distorted one.
    """
    T_cx = solve_ippe_handeye(
        data,
        rgb_grid_affine_3corners(data),
        dist_rgb=data.dist_rgb,
        cam_xyz_override=recompute_grid_with_distortion(data),
        board_offset_mm=-PCB_THICKNESS_MM,
        uv_xray_override=undistort_xray_points_working(data.xray_points_uv),
    )
    return invert_transform(T_cx), d_x_from_pose(T_cx, data.tip_xyz_c)


def variant_xray_full(data: EvalData) -> tuple[np.ndarray, float]:
    """Same pose as variant_xray_pose; the image is undistorted at render time."""
    return variant_xray_pose(data)


def variant_xray_k1(data: EvalData) -> tuple[np.ndarray, float]:
    """
    Production, but the X-ray side uses the Brown k1 model and its matching f.

    k1 and f were estimated jointly, so both are replaced together. The X-ray
    image is undistorted with the same pair at render time.
    """
    T_cx = solve_ippe_handeye(
        data,
        rgb_grid_affine_3corners(data),
        dist_rgb=data.dist_rgb,
        cam_xyz_override=recompute_grid_with_distortion(data),
        board_offset_mm=-PCB_THICKNESS_MM,
        K_xray_override=XRAY_K1_K,
        dist_xray_override=XRAY_K1_DIST,
    )
    return invert_transform(T_cx), d_x_from_pose(T_cx, data.tip_xyz_c)


def variant_production(data: EvalData) -> tuple[np.ndarray, float]:
    """
    The current production pipeline, end to end, on the stored capture.

    Unlike the other variants this one does NOT use the replicated solver: it
    calls overlay.calib.calibrate_camera_to_xray with exactly the arguments
    page_camera_to_xray_calibration passes, so it reflects the shipped code
    including the dist_rgb pass-through and the PCB thickness. The plane-fitting
    side is reproduced by redoing the ray/plane intersection with dist_rgb, and
    the overlay is rendered through the production distortion-aware warp.
    """
    xyz = recompute_grid_with_distortion(data)

    calib = calibrate_camera_to_xray(
        K_xray=data.K_xray,
        points_xyz_camera=xyz,
        points_uv_xray=data.xray_points_uv,
        dist_coeffs=None,
        dist_coeffs_rgb=data.dist_rgb,
        pose_method="ippe_handeye",
        refine_with_iterative=True,
        pitch_mm=PITCH_MM,
        checkerboard_corners_uv=data.corners_uv_3,
        K_rgb=data.K_rgb,
        steps_per_edge=STEPS_PER_EDGE,
        board_thickness_mm=PCB_THICKNESS_MM,
    )
    return calib.T_xc, d_x_from_pose(calib.T_cx, data.tip_xyz_c)


def variant_all(data: EvalData) -> tuple[np.ndarray, float]:
    """
    Everything at once: 9-corner homography grid, dist_rgb wherever it is
    currently dropped, and the 1.6 mm PCB thickness between the checkerboard
    and the copper plane (physically correct sign: checkerboard above copper,
    i.e. negative along the board +z which points into the table).
    """
    T_cx = solve_ippe_handeye(
        data,
        rgb_grid_homography_9corners(data),
        dist_rgb=data.dist_rgb,
        cam_xyz_override=recompute_grid_with_distortion(data),
        board_offset_mm=-1.6,
    )
    return invert_transform(T_cx), d_x_from_pose(T_cx, data.tip_xyz_c)


def variant_dist_rgb(data: EvalData) -> tuple[np.ndarray, float]:
    """
    Isolated dist_rgb test: apply the RGB distortion model everywhere it is
    currently dropped, while keeping the production 3-corner affine grid.

      1. plane fitting  - distortion-aware ray/plane intersection
      2. cam2X          - dist_coeffs_rgb on the RGB-side IPPE and refinement
      3. overlay warp   - handled at render time via use_dist
    """
    xyz = recompute_grid_with_distortion(data)
    T_cx = solve_ippe_handeye(
        data,
        rgb_grid_affine_3corners(data),
        dist_rgb=data.dist_rgb,
        cam_xyz_override=xyz,
    )
    return invert_transform(T_cx), d_x_from_pose(T_cx, data.tip_xyz_c)


VARIANTS: dict[str, Callable[[EvalData], tuple[np.ndarray, float]]] = {
    "baseline": variant_baseline,
    "recomputed": variant_recomputed,
    "corners9": variant_corners9,
    "dist_rgb": variant_dist_rgb,
    # PCB thickness: checkerboard on top, X-ray-visible hole in the copper on
    # the bottom. 1.6 mm board -> 1.6 mm if single-sided, 0.8 mm if plated.
    # Both signs are computed because the sign follows from the geometry and
    # should be confirmed, not assumed.
    "board_p1.6": _board_offset_variant(+1.6),
    "board_p0.8": _board_offset_variant(+0.8),
    "board_m0.8": _board_offset_variant(-0.8),
    "board_m1.6": _board_offset_variant(-1.6),
    "all": variant_all,
    # the shipped pipeline, through the real calibration entry point
    "production": variant_production,
    # X-ray detector distortion: first in T_bx only, then image as well
    "xray_pose": variant_xray_pose,
    "xray_full": variant_xray_full,
    "xray_k1": variant_xray_k1,
}

# Variants whose overlay is rendered through the distortion-aware RGB warp.
DISTORTED_WARP = {"dist_rgb", "all", "production", "xray_pose", "xray_full", "xray_k1"}

# Variants that additionally undistort the X-ray image before warping.
UNDISTORT_XRAY = {"xray_full"}

# Variants that need the X-ray distortion model and are skipped without it.
NEEDS_XRAY_DIST = {"xray_pose", "xray_full"}


# ============================================================
# Reporting
# ============================================================

def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def side_by_side(left: np.ndarray, right: np.ndarray,
                 left_text: str, right_text: str, gap: int = 12) -> np.ndarray:
    l, r = _label(left, left_text), _label(right, right_text)
    h = max(l.shape[0], r.shape[0])
    canvas = np.full((h, l.shape[1] + gap + r.shape[1], 3), 255, dtype=np.uint8)
    canvas[:l.shape[0], :l.shape[1]] = l
    canvas[:r.shape[0], l.shape[1] + gap:] = r
    return canvas


def render_corner_zoom(snapshot_dir: Path, out_dir: Path, pad: int = 26,
                       scale: int = 10) -> Path | None:
    """
    Zoom onto each detected checkerboard corner from the plane-fitting frame.

    The corners come from the plane-fitting capture, not from the overlay
    snapshot, so this reads the plane snapshot NPZ.
    """
    snaps = sorted(snapshot_dir.glob("plane_snapshot_*.npz"))
    if not snaps:
        print(f"[INFO] no plane_snapshot_*.npz in {snapshot_dir} — skipping corner zoom")
        return None

    s = np.load(str(snaps[0]), allow_pickle=True)
    if "rgb_image" not in s.files or "corners_full_uv" not in s.files:
        print("[INFO] plane snapshot lacks rgb_image/corners_full_uv — skipping corner zoom")
        return None

    img = s["rgb_image"]
    corners = np.asarray(s["corners_full_uv"], dtype=np.float64).reshape(-1, 2)
    ext = np.asarray(s["ext_uv"], dtype=np.float64).reshape(-1, 2) if "ext_uv" in s.files else None

    tiles = []
    for i, (u, v) in enumerate(corners):
        u0, v0 = int(round(u)) - pad, int(round(v)) - pad
        crop = img[max(0, v0):v0 + 2 * pad, max(0, u0):u0 + 2 * pad]
        if crop.size == 0:
            continue
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        # subpixel corner position inside the magnified tile
        cu = (u - u0) * scale
        cv_ = (v - v0) * scale
        cv2.drawMarker(big, (int(round(cu)), int(round(cv_))), (0, 0, 255),
                       cv2.MARKER_CROSS, 40, 1, cv2.LINE_AA)
        cv2.circle(big, (int(round(cu)), int(round(cv_))), 3, (0, 255, 255), -1)
        is_ext = ext is not None and np.min(np.linalg.norm(ext - [u, v], axis=1)) < 1.0
        cv2.putText(big, f"{i}{' EXT' if is_ext else ''}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        tiles.append(big)

    if not tiles:
        return None

    n = int(np.ceil(np.sqrt(len(tiles))))
    th, tw = tiles[0].shape[:2]
    grid = np.full((n * (th + 4), n * (tw + 4), 3), 255, dtype=np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, n)
        grid[r * (th + 4):r * (th + 4) + th, c * (tw + 4):c * (tw + 4) + tw] = t

    path = out_dir / "corner_zoom.png"
    cv2.imwrite(str(path), grid)
    print(f"[INFO] corner zoom ({len(tiles)} corners, {scale}x) -> {path.name}")
    return path


# Acceptance criterion: the X-ray pin row must sit inside the RGB housing.
# Housing width ~2.5 mm at 2.87 px/mm -> +-3.5 px tolerance about its centre.
HOUSING_TOLERANCE_PX = 3.5


def _comps(dark: np.ndarray, min_area: int, k: int):
    m = cv2.morphologyEx(dark.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m, connectivity=8)
    return [(ce[i], np.argwhere(lab == i)[:, ::-1].astype(np.float64))
            for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]


def detect_housings(data: EvalData, roi: tuple[int, int, int, int]) -> list[tuple]:
    """
    Locate the pin-header housings in the RGB snapshot once.

    The RGB image never changes between variants, so the reference structures
    must be detected exactly once - otherwise variants get compared on
    different subsets of headers, which invalidates any rms comparison.

    Returns [(centre_in_full_image, perpendicular_unit_vector), ...].
    """
    u0, v0, u1, v1 = roi
    crop = data.snapshot_rgb_bgr[v0:v1, u0:u1]
    g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    out = []
    for centre, px in _comps(g < np.percentile(g, 12), 300, 7):
        centred = px - px.mean(axis=0)
        axis = np.linalg.svd(centred, full_matrices=False)[2][0]
        axis = axis / np.linalg.norm(axis)
        out.append((centre + np.array([u0, v0]), np.array([-axis[1], axis[0]])))
    return out


def measure_header_offsets(
    data: EvalData,
    T_xc: np.ndarray,
    d_x: float,
    use_dist: bool,
    housings: list[tuple],
    undistort_xray: bool = False,
    brown_k1: bool = False,
    as_stored: bool = False,
) -> np.ndarray:
    """
    For each reference housing, how far the warped X-ray pin row sits from it.

    Only the component PERPENDICULAR to the header axis is used: both
    structures are elongated, so their centroids are poorly constrained along
    the axis but well constrained across it. The sign of that component is
    arbitrary per header (SVD sign), so only magnitudes are comparable.

    Returns one value per housing, NaN where no pin row could be matched.
    """
    T_xc = np.asarray(T_xc, dtype=np.float64)
    H = estimate_plane_induced_homography(
        K_c=data.K_rgb, R_xc=T_xc[:3, :3], t_xc=T_xc[:3, 3],
        K_x=data.K_xray, d_x=float(d_x),
    )

    xray = (data.xray_gray_working_u8 if as_stored
            else data.xray_working_pristine)
    if brown_k1:
        xray = cv2.undistort(xray, XRAY_K1_K, XRAY_K1_DIST, None, XRAY_K1_K)
    elif undistort_xray and XRAY_DIST is not None:
        xray = undistort_xray_image_working(xray)
    h, w = data.snapshot_rgb_bgr.shape[:2]
    fov = _detect_xray_fov_mask(xray)
    masked = cv2.bitwise_and(xray, xray, mask=fov)

    if use_dist:
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        obs = np.stack([uu.ravel(), vv.ravel()], axis=1).reshape(-1, 1, 2)
        ideal = cv2.undistortPoints(obs, data.K_rgb, data.dist_rgb, P=data.K_rgb).reshape(-1, 2)
        q = (np.linalg.inv(H) @ np.hstack([ideal, np.ones((len(ideal), 1))]).T).T
        src = q[:, :2] / q[:, 2:3]
        mx = np.ascontiguousarray(src[:, 0].reshape(h, w).astype(np.float32))
        my = np.ascontiguousarray(src[:, 1].reshape(h, w).astype(np.float32))
        warped = cv2.remap(masked, mx, my, cv2.INTER_LINEAR, borderValue=0)
        mask_c = cv2.remap(fov, mx, my, cv2.INTER_NEAREST, borderValue=0) > 0
    else:
        warped = cv2.warpPerspective(masked, H, (w, h), flags=cv2.INTER_LINEAR)
        mask_c = cv2.warpPerspective(fov, H, (w, h), flags=cv2.INTER_NEAREST) > 0

    mk = cv2.erode(mask_c.astype(np.uint8) * 255, np.ones((31, 31), np.uint8)) > 0
    if not np.any(mk):
        return np.full(len(housings), np.nan)

    xf = warped.astype(np.float32)
    pins = _comps((xf < np.percentile(xf[mk], 4)) & mk, 60, 13)

    out = np.full(len(housings), np.nan)
    for k, (centre, perp) in enumerate(housings):
        near = [p for p in pins if np.linalg.norm(p[0] - centre) < 14]
        if not near:
            continue
        ce_p = max(near, key=lambda p: len(p[1]))[0]
        out[k] = float(np.dot(ce_p - centre, perp))
    return out


def format_offsets(off: np.ndarray, keep: np.ndarray) -> str:
    """Report only headers that could be matched in EVERY variant."""
    v = off[keep]
    if len(v) == 0 or not np.isfinite(v).all():
        return "measurement failed"
    outside = int(np.count_nonzero(np.abs(v) > HOUSING_TOLERANCE_PX))
    return (f"n={len(v)}  rms |perp| {np.sqrt((v ** 2).mean()):5.2f} px  "
            f"max {np.abs(v).max():5.2f}  "
            f"outside +-{HOUSING_TOLERANCE_PX:.1f}px: {outside}/{len(v)}")


def image_diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"identical": False, "note": f"shape mismatch {a.shape} vs {b.shape}"}

    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    n_diff = int(np.count_nonzero(diff.any(axis=2)))

    return {
        "identical": bool(n_diff == 0),
        "max_abs": int(diff.max()),
        "mean_abs": float(diff.mean()),
        "n_pixels_differing": n_diff,
        "pct_pixels_differing": 100.0 * n_diff / (a.shape[0] * a.shape[1]),
    }


def print_capture_summary(data: EvalData) -> None:
    xyz = data.xray_points_xyz_c

    # The 121 grid points are coplanar by construction, so the fitted plane and
    # the corner rays are recoverable from them without the raw depth map.
    centroid = xyz.mean(axis=0)
    _, sv, vt = np.linalg.svd(xyz - centroid)
    normal = vt[2] / np.linalg.norm(vt[2])
    tilt_deg = float(np.degrees(np.arccos(abs(normal[2]))))
    planarity_mm = float(sv[2] / np.sqrt(len(xyz)) * 1000.0)

    edge_tr_mm = float(np.linalg.norm(xyz[IDX_TR] - xyz[IDX_TL]) * 1000.0)
    edge_bl_mm = float(np.linalg.norm(xyz[IDX_BL] - xyz[IDX_TL]) * 1000.0)
    z_mean_mm = float(xyz[:, 2].mean() * 1000.0)

    span_u = float(np.ptp(data.corners_uv_3[:, 0]))
    span_v = float(np.ptp(data.corners_uv_3[:, 1]))

    print("=" * 68)
    print(f"Capture: {data.path.name}")
    print("=" * 68)
    print(f"  working distance (mean z)   : {z_mean_mm:8.2f} mm")
    print(f"  board tilt vs image plane   : {tilt_deg:8.2f} deg")
    print(f"  plane normal (camera frame) : [{normal[0]:+.4f} {normal[1]:+.4f} {normal[2]:+.4f}]")
    print(f"  grid planarity rms          : {planarity_mm:8.2e} mm  (0 = coplanar by construction)")
    print(f"  board image span            : {span_u:8.2f} x {span_v:.2f} px")
    print()
    print(f"  grid edge TL-TR             : {edge_tr_mm:8.4f} mm   "
          f"({100.0 * (edge_tr_mm / NOMINAL_EDGE_MM - 1.0):+.3f} % vs nominal {NOMINAL_EDGE_MM} mm)")
    print(f"  grid edge TL-BL             : {edge_bl_mm:8.4f} mm   "
          f"({100.0 * (edge_bl_mm / NOMINAL_EDGE_MM - 1.0):+.3f} % vs nominal {NOMINAL_EDGE_MM} mm)")
    print(f"  edge-to-edge inconsistency  : {abs(edge_tr_mm - edge_bl_mm):8.4f} mm")
    print()
    print(f"  d_x                         : {data.d_x:8.3f} mm")
    print(f"  alpha                       : {data.alpha:8.3f}")
    print(f"  dist_rgb                    : {'present' if data.dist_rgb is not None else 'MISSING'}")
    print(f"  checkerboard_corners_uv_9   : {'present' if data.corners_uv_9 is not None else 'MISSING'}")
    print()


def verify_baseline(data: EvalData, overlay: np.ndarray, H_xc: np.ndarray) -> bool:
    h_err = float(np.abs(H_xc - data.H_xc).max())
    img = image_diff_stats(overlay, data.overlay_bgr)

    print("-" * 68)
    print("Baseline reproduction")
    print("-" * 68)
    print(f"  max |H_offline - H_saved|   : {h_err:.3e}")

    if img.get("identical"):
        print("  overlay image               : identical (bit for bit)")
    else:
        print(f"  overlay image               : DIFFERS")
        for k, v in img.items():
            if k != "identical":
                print(f"      {k:24s}: {v}")

    ok = (h_err == 0.0) and bool(img.get("identical"))
    print(f"\n  => baseline {'REPRODUCED' if ok else 'NOT reproduced'}\n")
    return ok


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-render a saved overlay offline and compare optimization variants."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help=f"overlay_preview_*.npz (default: newest in {DEFAULT_EVAL_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for rendered PNGs (default: <npz dir>/offline_overlay)",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help=(
            "Comma-separated blend levels to render in addition to the stored one. "
            f"alpha=1 is pure RGB, alpha=0 is pure X-ray. Default: "
            f"{','.join(f'{a:.2f}' for a in EXTRA_ALPHAS)}"
        ),
    )
    parser.add_argument(
        "--no-dx-sweep",
        action="store_true",
        help="Skip the d_x sweep.",
    )
    args = parser.parse_args()

    alphas = (
        EXTRA_ALPHAS
        if args.alphas is None
        else tuple(float(x) for x in args.alphas.split(","))
    )

    npz_path = args.npz if args.npz is not None else find_latest_npz(DEFAULT_EVAL_DIR)
    out_dir = args.out if args.out is not None else npz_path.parent / "offline_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_eval_npz(npz_path)
    print_capture_summary(data)

    global XRAY_DIST
    XRAY_DIST = fit_xray_distortion(npz_path.parent)
    if XRAY_DIST is not None:
        probe = np.stack(np.meshgrid(np.linspace(80, 944, 40),
                                     np.linspace(80, 944, 40)), axis=-1).reshape(-1, 2)
        probe = probe[np.hypot(probe[:, 0] - 512, probe[:, 1] - 512) < 450]
        print(f"  X-ray distortion model: {XRAY_DIST.n_points} holes, "
              f"degree {XRAY_DIST.degree} ({XRAY_DIST.n_terms} terms/axis)")
        print(f"    measured field: rms {XRAY_DIST.raw_rms:.3f} px, "
              f"max {XRAY_DIST.raw_max:.3f} px")
        print(f"    fit residual  : rms {XRAY_DIST.fit_rms:.3f} px")
        print(f"    round trip    : max {XRAY_DIST.round_trip_error(probe):.2e} px\n")

    # ---- baseline first: the harness is only trustworthy if this matches ----
    T_xc_base, d_x_base = VARIANTS["baseline"](data)

    # Captures made before the distortion-aware warp shipped were rendered with
    # the plain projective warp; newer ones were not. Pick whichever branch
    # reproduces the stored overlay, so the baseline stays bit-exact either way.
    base_use_dist = False
    overlay_base, H_base, mask_base = render_overlay(
        data, T_xc_base, d_x_base, data.alpha, use_dist=False, as_stored=True
    )
    if data.dist_rgb is not None and not image_diff_stats(
        overlay_base, data.overlay_bgr
    ).get("identical"):
        alt = render_overlay(data, T_xc_base, d_x_base, data.alpha, use_dist=True,
                              as_stored=True)
        if image_diff_stats(alt[0], data.overlay_bgr).get("identical"):
            overlay_base, H_base, mask_base = alt
            base_use_dist = True

    print(f"  capture rendered with distortion-aware warp: {base_use_dist}")
    if base_use_dist:
        DISTORTED_WARP.add("baseline")
        DISTORTED_WARP.add("recomputed")

    verify_baseline(data, overlay_base, H_base)

    cv2.imwrite(str(out_dir / "overlay_saved.png"), data.overlay_bgr)
    cv2.imwrite(str(out_dir / "overlay_baseline.png"), overlay_base)

    # ROI fixed by the baseline so all variants crop identically
    roi = roi_from_mask(mask_base)

    if not args.no_dx_sweep:
        render_dx_sweep(data, T_xc_base, d_x_base, DX_SWEEP_MM, out_dir, roi)

    # ---- every variant at every requested blend level ----
    print("-" * 68)
    print("Variants")
    print("-" * 68)

    rendered: dict[str, dict[float, np.ndarray]] = {}
    housings = detect_housings(data, roi)
    print(f"  reference housings detected in the RGB snapshot: {len(housings)}\n")

    results: dict[str, tuple] = {}
    for name, fn in VARIANTS.items():
        if name in NEEDS_XRAY_DIST and XRAY_DIST is None:
            continue
        T_xc, d_x = fn(data)
        stored = name in ("baseline", "recomputed")
        rendered[name] = render_and_save(data, name, T_xc, d_x, alphas, out_dir, roi,
                                         brown_k1=(name == "xray_k1"),
                                         as_stored=stored)
        T_cx = invert_transform(T_xc)
        results[name] = (
            _pose_delta(T_cx, data.T_cx),
            d_x - data.d_x,
            measure_header_offsets(data, T_xc, d_x, name in DISTORTED_WARP,
                                   housings, name in UNDISTORT_XRAY,
                                   brown_k1=(name == "xray_k1"),
                                   as_stored=stored),
        )

    # only headers matched by every variant, so the rms values are comparable
    keep = np.all([np.isfinite(r[2]) for r in results.values()], axis=0)

    for name, ((ang, dt), dd, off) in results.items():
        print(f"  {name:12s} dRot {ang:8.5f} deg  dTrans {dt:7.3f} mm  d_x {dd:+7.3f} mm")
        print(f"  {'':12s} {format_offsets(off, keep)}")

    # ---- 1x2 comparison per variant: baseline left, variant right ----
    a_cmp = DX_SWEEP_ALPHA if DX_SWEEP_ALPHA in alphas else alphas[0]
    left_b = crop_zoom(rendered["baseline"][a_cmp], roi)

    for name in rendered:
        if name in ("baseline", "recomputed"):
            continue
        cv2.imwrite(
            str(out_dir / f"compare_{name}.png"),
            side_by_side(left_b, crop_zoom(rendered[name][a_cmp], roi),
                         f"baseline (alpha {a_cmp:.2f})",
                         f"{name} (alpha {a_cmp:.2f})"),
        )

    render_corner_zoom(npz_path.parent / "debug_snapshots", out_dir)

    print(f"\n[INFO] Images written to {out_dir}")
    print(f"[INFO] Blend levels rendered: {', '.join(f'{a:.2f}' for a in alphas)} "
          f"(1.00 = pure RGB, 0.00 = pure X-ray)")


if __name__ == "__main__":
    main()
