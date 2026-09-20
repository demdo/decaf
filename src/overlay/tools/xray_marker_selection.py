# overlay/tools/xray_marker_selection.py
# -*- coding: utf-8 -*-
"""
xray_marker_selection.py

Marker detection + robust ROI extraction from THREE anchor clicks, without any UI
dependencies and without building a circles_grid.

Public API
----------
- run_xray_marker_detection(...)
- compute_roi_from_grid(...)

Design
------
Detection:
    img_gray -> (optional CLAHE) -> (optional detector mask) -> detect_blobs_hough

ROI:
    The user selects three anchors in this exact order:

        1) TL
        2) TR
        3) BL

    IMPORTANT:
    These anchor roles are defined in CAMERA VIEW semantics, i.e. from the
    camera / top-view perspective of the board.

    This function produces a canonical RAW ROI ordering in CAMERA VIEW /
    global board order only.

    It does NOT perform any X-ray semantic reordering. After the canonical RAW
    order has been determined, the measured UVs are transformed into the X-ray
    working pixel space:

        u_work = W - 1 - u_raw
        v_work = v_raw

    The affine ROI model is

        p = p_tl + alpha*u + beta*v

    with:
        u = TR - TL
        v = BL - TL

    This affine model is used only for ROI selection and ordering.
    No affine regularization, grid fitting, or point repositioning is performed.

    Points are kept by:
      (A) affine inclusion: alpha,beta within [0..1] with margin
      (B) lattice proximity gate in PIXEL units using pitch from estimate_pitch_nn

Notes
-----
- No circles_sorted, no circles_grid.
- No affine regularization is applied to the returned ROI points.
- No j_xray = nu - j_cam semantic reordering is performed here.
- estimate_pitch_nn is imported from overlay.tools.blob_detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from overlay.tools.blob_detection import detect_blobs_hough, estimate_pitch_nn


# ============================================================
# Result container (public)
# ============================================================

@dataclass(frozen=True)
class XrayMarkerDetectionResult:
    """
    Result bundle for the marker DETECTION pipeline (no ROI computation).

    Attributes
    ----------
    img_proc : np.ndarray
        Preprocessed image (e.g., CLAHE applied), uint8 (H,W).
    mask : np.ndarray
        Detector mask, uint8 (H,W), values in {0,255}.
    img_masked : np.ndarray
        Masked image fed into blob detection, uint8 (H,W).
    circles : Optional[np.ndarray]
        Circle detections, (N,3) float32 [x,y,r], or None if detection failed.
    """
    img_proc: np.ndarray
    mask: np.ndarray
    img_masked: np.ndarray
    circles: Optional[np.ndarray]


# ============================================================
# Small internal helpers
# ============================================================

def _apply_clahe(
    img_gray_u8: np.ndarray,
    *,
    use_clahe: bool,
    clip_limit: float,
    tile_grid_size: Tuple[int, int],
) -> np.ndarray:
    """Optionally apply CLAHE to a uint8 grayscale image."""
    if not use_clahe:
        return img_gray_u8

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid_size),
    )
    return clahe.apply(img_gray_u8)


def _detector_mask(
    img_gray: np.ndarray,
    *,
    blur_ks: int = 11,
    thr_mode: str = "otsu",
    adaptive_block: int = 51,
    adaptive_C: int = -5,
    close_ks: int = 41,
    close_iter: int = 1,
) -> np.ndarray:
    """
    Create a broad support mask for the detector region.

    Returns
    -------
    mask : np.ndarray
        uint8 (H,W), 255 inside detector ROI, 0 outside.
    """
    if img_gray.ndim != 2:
        raise ValueError("img_gray must be grayscale (H,W).")

    h, w = img_gray.shape

    k = int(blur_ks)
    k = k if (k % 2 == 1) else (k + 1)
    blur = cv2.GaussianBlur(img_gray, (k, k), 0)

    if thr_mode == "adaptive":
        blk = int(adaptive_block)
        blk = blk if (blk % 2 == 1) else (blk + 1)
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blk,
            int(adaptive_C),
        )
    else:
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if float(np.mean(bw)) > 220.0:
        bw = 255 - bw

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.full((h, w), 255, dtype=np.uint8)

    def _touches_border(c: np.ndarray) -> bool:
        xs = c[:, 0, 0]
        ys = c[:, 0, 1]
        return (
            (xs.min() <= 1)
            or (ys.min() <= 1)
            or (xs.max() >= w - 2)
            or (ys.max() >= h - 2)
        )

    nb = [c for c in cnts if not _touches_border(c)]
    candidates = nb if nb else cnts
    largest = max(candidates, key=cv2.contourArea)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)

    kc = int(close_ks)
    if kc > 1:
        kc = kc if (kc % 2 == 1) else (kc + 1)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((kc, kc), np.uint8),
            iterations=int(close_iter),
        )

    return mask


def _solve_alpha_beta(
    P: np.ndarray,
    p0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Solve (alpha,beta) in: P = p0 + alpha*u + beta*v for each P (Nx2).

    Returns
    -------
    AB : np.ndarray
        (N,2) float64 array, columns [alpha, beta].
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)

    A = np.stack([u, v], axis=1)  # 2x2 (columns)
    det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    if abs(det) < 1e-9:
        raise ValueError("Degenerate ROI basis (u and v nearly colinear).")

    Ainv = np.linalg.inv(A)
    D = (np.asarray(P, float) - np.asarray(p0, float)[None, :]).T  # 2xN
    return (Ainv @ D).T


def _lattice_gate_keep(
    *,
    alpha: np.ndarray,
    beta: np.ndarray,
    in_box: np.ndarray,
    Lu: float,
    Lv: float,
    pitch: float,
    nu: int,
    nv: int,
    tol_px: float,
) -> np.ndarray:
    """
    Lattice proximity gate in PIXEL units along u and v.

    au = alpha*|u| should be close to ku*(|u|/nu)
    bv = beta*|v|  should be close to kv*(|v|/nv)
    """
    du = Lu / float(nu)
    dv = Lv / float(nv)

    au = alpha * Lu
    bv = beta * Lv

    ku = np.rint(au / du)
    kv = np.rint(bv / dv)

    ru = np.abs(au - ku * du)
    rv = np.abs(bv - kv * dv)

    on = (ru <= tol_px) & (rv <= tol_px)
    return in_box & on


def _check_anchor_orientation(
    p_tl: np.ndarray,
    p_tr: np.ndarray,
    p_bl: np.ndarray,
) -> tuple[bool, str | None, dict]:
    """
    Check anchor orientation for the current setup.

    The anchors are selected in CAMERA VIEW semantics:

        1) TL
        2) TR
        3) BL

    Hence the provisional affine basis is expected to satisfy:

        u = TR - TL  -> u_x < 0
        v = BL - TL  -> v_y < 0
        cross_z      -> cross_z(u, v) > 0

    with

        cross_z = u_x * v_y - u_y * v_x
    """
    u = p_tr - p_tl
    v = p_bl - p_tl

    ux = float(u[0])
    uy = float(u[1])
    vx = float(v[0])
    vy = float(v[1])

    cross_z = float(ux * vy - uy * vx)

    cond_ux = (ux < 0.0)
    cond_vy = (vy < 0.0)
    cond_cross = (cross_z > 0.0)

    dbg = dict(
        ux=ux,
        uy=uy,
        vx=vx,
        vy=vy,
        cross_z=cross_z,
        cond_ux=cond_ux,
        cond_vy=cond_vy,
        cond_cross=cond_cross,
    )

    if abs(cross_z) < 1e-9:
        return (
            False,
            "The selected anchor markers are nearly collinear.\n\n"
            "This anchor configuration is geometrically unstable and may indicate "
            "an incorrect selection.",
            dbg,
        )

    ok = cond_ux and cond_vy and cond_cross
    if ok:
        return True, None, dbg

    return (
        False,
        "The selected anchors do not match the expected CAMERA VIEW orientation.\n\n"
        "Expected for [TL, TR, BL] selected from the top-view / camera-view:\n"
        "- x-component of (TR - TL) < 0\n"
        "- y-component of (BL - TL) < 0\n"
        "- cross_z((TR - TL), (BL - TL)) > 0\n\n"
        "This may indicate:\n"
        "- anchors selected in the wrong order\n"
        "- incorrect top-view interpretation\n"
        "- inconsistent X-ray image orientation",
        dbg,
    )



def transform_xray_uv_raw_to_working(
    uv_raw: np.ndarray,
    *,
    image_width: int,
) -> np.ndarray:
    """
    RAW X-ray UV -> X-ray working-space UV.

    Horizontal pixel-space flip:
        u_work = W - 1 - u_raw
        v_work = v_raw
    """
    uv_raw = np.asarray(uv_raw, dtype=np.float64).reshape(-1, 2)
    image_width = int(image_width)
    if image_width <= 0:
        raise ValueError("image_width must be positive.")

    uv_work = uv_raw.copy()
    uv_work[:, 0] = float(image_width - 1) - uv_work[:, 0]
    return uv_work


def index_board_lattice(
    circles_uv_raw: np.ndarray,
    patch_uv_raw: np.ndarray,
    *,
    image_width: int,
    pitch_mm: float = 2.54,
    nu: int = 10,
    nv: int = 10,
    tol_pitch: float = 0.30,
    n_iter: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Extend the indexed 11x11 patch to ALL detected holes of the board.

    Added 2026-09-15 for the X-ray side of the camera-to-X-ray calibration.
    The patch (25.4 mm) is what the camera side can see and index; the
    X-ray image, however, shows the whole board (~100 mm, ~1300 holes), and
    only that extent carries enough perspective to make the planar pose
    unambiguous and its tilt measurable (see
    tracking.pose_solvers._solve_pose_ippe_handeye). This function assigns
    every detection its integer lattice coordinates (j, i) in the SAME board
    frame as the patch: origin at the patch's TL hole, j along x_b, i along
    y_b, so that the patch holes keep their indices 0..nu / 0..nv and the
    other holes get indices beyond that range (negative or > 10).

    Method (same bootstrap as debug/export_xray_distortion._indexed_holes,
    which produced the distortion field): a homography from the canonical
    patch board coordinates to the 121 patch detections is used to map all
    detections into board coordinates; each detection within `tol_pitch`
    of an integer site is accepted (one detection per site, the closest
    wins), the homography is re-fitted on the accepted set and the
    assignment repeated `n_iter` times. Only the assignment uses the
    homography; the returned image points are the measured centres.

    Parameters
    ----------
    circles_uv_raw : (N, 2) or (N, 3)
        All detections in RAW detector coordinates (u, v[, r]).
    patch_uv_raw : (121, 2)
        The indexed patch in RAW coordinates, canonical row-major order
        k = (nu+1) * i + j (as compute_roi_from_grid produces it).
    image_width : int
        Detector width, for the RAW -> working-space flip.

    Returns
    -------
    uv_working : (M, 2)
        Accepted detections in XRAY_WORKING_FLIPPED_UV (u = W - 1 - u_raw).
    lattice_ij : (M, 2) int
        Integer lattice coordinates (j, i) per accepted detection.
    board_xy_mm : (M, 2)
        Board coordinates (j * pitch, i * pitch) in mm, z_b = 0 implied.
    dbg : dict
        n_detections, n_indexed, lattice ranges, residual of the last
        homography fit.
    """
    circles = np.asarray(circles_uv_raw, dtype=np.float64).reshape(-1, circles_uv_raw.shape[-1])[:, :2]
    patch = np.asarray(patch_uv_raw, dtype=np.float64).reshape(-1, 2)
    n_expected = (nu + 1) * (nv + 1)
    if patch.shape[0] != n_expected:
        raise ValueError(f"patch_uv_raw must hold {n_expected} points, got {patch.shape[0]}.")
    if circles.shape[0] < n_expected:
        raise ValueError("Fewer detections than patch points.")

    board_patch = np.array(
        [[j * pitch_mm, i * pitch_mm] for i in range(nv + 1) for j in range(nu + 1)],
        dtype=np.float64,
    )
    H, _ = cv2.findHomography(board_patch.reshape(-1, 1, 2), patch.reshape(-1, 1, 2), 0)
    if H is None:
        raise RuntimeError("Bootstrap homography of the patch failed.")

    ok = np.zeros(circles.shape[0], dtype=bool)
    lat = np.zeros((circles.shape[0], 2), dtype=np.int64)
    for _ in range(int(n_iter)):
        b = cv2.perspectiveTransform(
            circles.reshape(-1, 1, 2), np.linalg.inv(H)
        ).reshape(-1, 2) / float(pitch_mm)
        lat_f = np.rint(b)
        err = np.linalg.norm(b - lat_f, axis=1)
        ok[:] = False
        seen: set = set()
        for t in np.argsort(err):
            if err[t] >= tol_pitch:
                break
            key = (int(lat_f[t, 0]), int(lat_f[t, 1]))
            if key in seen:
                continue
            seen.add(key)
            ok[t] = True
        lat = lat_f.astype(np.int64)
        H, _ = cv2.findHomography(
            (lat[ok].astype(np.float64) * pitch_mm).reshape(-1, 1, 2),
            circles[ok].reshape(-1, 1, 2),
            0,
        )
        if H is None:
            raise RuntimeError("Lattice homography re-fit failed.")

    uv_raw_ok = circles[ok]
    lattice_ij = lat[ok]
    board_xy_mm = lattice_ij.astype(np.float64) * float(pitch_mm)
    proj = cv2.perspectiveTransform(board_xy_mm.reshape(-1, 1, 2), H).reshape(-1, 2)
    resid = np.linalg.norm(proj - uv_raw_ok, axis=1)

    dbg = dict(
        n_detections=int(circles.shape[0]),
        n_indexed=int(ok.sum()),
        j_range=(int(lattice_ij[:, 0].min()), int(lattice_ij[:, 0].max())),
        i_range=(int(lattice_ij[:, 1].min()), int(lattice_ij[:, 1].max())),
        homography_resid_rms_px=float(np.sqrt(np.mean(resid ** 2))),
        homography_resid_max_px=float(resid.max()),
    )
    uv_working = transform_xray_uv_raw_to_working(uv_raw_ok, image_width=int(image_width))
    return uv_working, lattice_ij, board_xy_mm, dbg

# ============================================================
# Public API
# ============================================================

def run_xray_marker_detection(
    img_gray: np.ndarray,
    *,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_tiles: Tuple[int, int] = (12, 12),
    use_mask: bool = True,
    blur_ks: int = 11,
    thr_mode: str = "otsu",
    adaptive_block: int = 51,
    adaptive_C: int = -5,
    close_ks: int = 41,
    close_iter: int = 1,
    hough_params=None,
) -> XrayMarkerDetectionResult:
    """
    Detection-only pipeline:
        CLAHE -> detector_mask -> mask image -> detect_blobs_hough
    """
    if img_gray.ndim != 2:
        raise ValueError("img_gray must be grayscale (H,W).")

    img_u8 = (
        img_gray
        if img_gray.dtype == np.uint8
        else np.clip(img_gray, 0, 255).astype(np.uint8)
    )

    img_proc = _apply_clahe(
        img_u8,
        use_clahe=use_clahe,
        clip_limit=clahe_clip,
        tile_grid_size=clahe_tiles,
    )

    if use_mask:
        mask = _detector_mask(
            img_proc,
            blur_ks=blur_ks,
            thr_mode=thr_mode,
            adaptive_block=adaptive_block,
            adaptive_C=adaptive_C,
            close_ks=close_ks,
            close_iter=close_iter,
        )
        img_masked = img_proc.copy()
        img_masked[mask == 0] = 0
    else:
        mask = np.full(img_proc.shape, 255, dtype=np.uint8)
        img_masked = img_proc

    circles = detect_blobs_hough(img_masked, params=hough_params)
    if circles is None or len(circles) == 0:
        return XrayMarkerDetectionResult(
            img_proc=img_proc,
            mask=mask,
            img_masked=img_masked,
            circles=None,
        )

    c = np.asarray(circles, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(c[:, 0]) & np.isfinite(c[:, 1]) & np.isfinite(c[:, 2])
    c = c[finite]
    if c.size == 0:
        c = None

    return XrayMarkerDetectionResult(
        img_proc=img_proc,
        mask=mask,
        img_masked=img_masked,
        circles=c,
    )


def compute_roi_from_grid(
    circles: np.ndarray,
    anchor_idx: Sequence[int],
    *,
    margin_px: float,
    image_width: int,
    gate_tol_pitch: float = 0.40,
    min_steps: int = 2,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute a marker ROI from circle detections using 3 ORDERED anchor indices.

    The three anchors must be provided in this order:
        [TL, TR, BL]
    with roles defined in CAMERA VIEW semantics.

    The function uses an affine parameterization induced by these anchors to:
        - define the ROI support region
        - gate detections by lattice proximity
        - assign a canonical CAMERA VIEW / global board ordering.

    Important
    ---------
    No affine regularization is applied.
    No ideal grid is fitted back onto the detections.
    No X-ray semantic reordering is applied.
    All returned ROI points are measured blob centers transformed into the
    X-ray working pixel space.

    Returns
    -------
    uv_final : np.ndarray
        Canonically ordered ROI points in XRAY_WORKING_FLIPPED_UV, shape (N,2).
        These are measured points only, not regularized points.

    roi_idx : np.ndarray
        Indices of the selected ROI detections in the original circles array,
        in the same canonical order as uv_final.

    dbg : dict
        Debug information for ROI extraction, ordering, and orientation checks.
    """
    c = np.asarray(circles, dtype=np.float64).reshape(-1, 3)
    if c.shape[0] < 3:
        raise ValueError("circles must contain at least 3 entries.")
    if anchor_idx is None or len(anchor_idx) != 3:
        raise ValueError("anchor_idx must contain exactly 3 indices.")

    image_width = int(image_width)
    if image_width <= 0:
        raise ValueError("image_width must be positive.")

    idx = np.asarray(anchor_idx, dtype=int)
    if np.any(idx < 0) or np.any(idx >= c.shape[0]):
        raise ValueError("anchor_idx contains out-of-range indices.")

    xy = c[:, :2]
    if not np.isfinite(xy).all():
        finite = np.isfinite(xy).all(axis=1)
        xy = xy[finite]
        map_back = np.flatnonzero(finite)
    else:
        map_back = None

    if xy.shape[0] < 3:
        raise ValueError("Not enough finite circle centers for ROI computation.")

    if map_back is None:
        idx_eff = idx
    else:
        inv = {int(old): int(new) for new, old in enumerate(map_back)}
        idx_eff = np.array([inv[int(i)] for i in idx], dtype=int)

    p_tl = xy[idx_eff[0], :]
    p_tr = xy[idx_eff[1], :]
    p_bl = xy[idx_eff[2], :]

    orientation_ok, orientation_warning, orientation_dbg = _check_anchor_orientation(
        p_tl, p_tr, p_bl
    )

    u = p_tr - p_tl
    v = p_bl - p_tl

    Lu = float(np.linalg.norm(u))
    Lv = float(np.linalg.norm(v))
    if Lu <= 1e-9 or Lv <= 1e-9:
        raise ValueError("Anchor geometry is degenerate (edges too short).")

    pitch = float(estimate_pitch_nn(xy))
    if (not np.isfinite(pitch)) or pitch <= 1e-6:
        pitch = 10.0

    AB = _solve_alpha_beta(xy, p_tl, u, v)
    alpha = AB[:, 0]
    beta = AB[:, 1]

    tol_px = float(gate_tol_pitch) * float(pitch)
    mu = float((margin_px + tol_px) / (Lu + 1e-12))
    mv = float((margin_px + tol_px) / (Lv + 1e-12))

    in_box = (
        (alpha >= -mu)
        & (alpha <= 1.0 + mu)
        & (beta >= -mv)
        & (beta <= 1.0 + mv)
    )

    nu0 = int(np.clip(np.rint(Lu / pitch), int(min_steps), 10_000))
    nv0 = int(np.clip(np.rint(Lv / pitch), int(min_steps), 10_000))

    best_keep = None
    best = (nu0, nv0, -1)

    for nu in (max(min_steps, nu0 - 1), nu0, nu0 + 1):
        for nv in (max(min_steps, nv0 - 1), nv0, nv0 + 1):
            keep_tmp = _lattice_gate_keep(
                alpha=alpha,
                beta=beta,
                in_box=in_box,
                Lu=Lu,
                Lv=Lv,
                pitch=pitch,
                nu=int(nu),
                nv=int(nv),
                tol_px=tol_px,
            )
            score = int(np.count_nonzero(keep_tmp))
            if score > best[2]:
                best = (int(nu), int(nv), score)
                best_keep = keep_tmp

    keep = best_keep
    if keep is None:
        raise RuntimeError("ROI gating failed unexpectedly.")

    roi_idx_local_all = np.flatnonzero(keep).astype(np.int64)

    alpha_roi = alpha[roi_idx_local_all]
    beta_roi = beta[roi_idx_local_all]

    nu = int(best[0])
    nv = int(best[1])

    # Canonical board/camera-view lattice indices:
    #   i_cam: row index   (TL -> BL)
    #   j_cam: column index(TL -> TR)
    j_cam = np.rint(alpha_roi * nu).astype(np.int32)
    i_cam = np.rint(beta_roi * nv).astype(np.int32)

    j_cam = np.clip(j_cam, 0, nu)
    i_cam = np.clip(i_cam, 0, nv)

    # IMPORTANT:
    # Do NOT compute j_xray = nu - j_cam here anymore.
    # The actual X-ray working-space coordinate transform is performed later
    # by the GUI page using u_work = W - 1 - u_raw.
    order_raw = np.lexsort((j_cam, i_cam))
    roi_idx_local = roi_idx_local_all[order_raw]

    grid_i_raw = i_cam[order_raw].astype(np.int32)
    grid_j_raw = j_cam[order_raw].astype(np.int32)

    uv_raw = xy[roi_idx_local].astype(np.float64)
    uv_final = transform_xray_uv_raw_to_working(uv_raw, image_width=image_width)

    if map_back is None:
        roi_idx = roi_idx_local.astype(np.int64)
    else:
        roi_idx = map_back[roi_idx_local].astype(np.int64)

    dbg = dict(
        pitch=float(pitch),
        Lu=float(Lu),
        Lv=float(Lv),
        nu0=int(nu0),
        nv0=int(nv0),
        nu=int(best[0]),
        nv=int(best[1]),
        mu=float(mu),
        mv=float(mv),
        tol_px=float(tol_px),
        in_box=int(np.count_nonzero(in_box)),
        keep=int(np.count_nonzero(keep)),
        gate_tol_pitch=float(gate_tol_pitch),
        anchor_idx=np.asarray(anchor_idx, dtype=int).tolist(),
        anchor_idx_effective=np.asarray(idx_eff, dtype=int).tolist(),
        anchor_role=["TL", "TR", "BL"],

        grid_i_raw=grid_i_raw.tolist(),
        grid_j_raw=grid_j_raw.tolist(),

        # Backward-compatible aliases.
        # These are now identical to the RAW/canonical indices.
        grid_i_final=grid_i_raw.tolist(),
        grid_j_final=grid_j_raw.tolist(),
        grid_i=grid_i_raw.tolist(),
        grid_j=grid_j_raw.tolist(),

        orientation_ok=bool(orientation_ok),
        orientation_warning=orientation_warning,
        orientation_ux=float(orientation_dbg["ux"]),
        orientation_uy=float(orientation_dbg["uy"]),
        orientation_vx=float(orientation_dbg["vx"]),
        orientation_vy=float(orientation_dbg["vy"]),
        orientation_cross_z=float(orientation_dbg["cross_z"]),
        orientation_cond_ux=bool(orientation_dbg["cond_ux"]),
        orientation_cond_vy=bool(orientation_dbg["cond_vy"]),
        orientation_cond_cross=bool(orientation_dbg["cond_cross"]),

        debug_uv_raw=uv_raw.tolist(),
        debug_uv_raw_final=uv_raw.tolist(),
        debug_uv_final=uv_final.tolist(),
        points_uv_raw=uv_raw.tolist(),
        points_uv_working=uv_final.tolist(),

        semantic_reordering=False,
        no_j_xray_reordering=True,
        uv_space="XRAY_WORKING_FLIPPED_UV",
        raw_uv_space="XRAY_RAW_CANONICAL_BOARD_ORDER",
        uv_transform="horizontal_flip",
        uv_transform_formula="u_work = W - 1 - u_raw, v_work = v_raw",
        image_width=int(image_width),
    )

    return uv_final, roi_idx, dbg
