# -*- coding: utf-8 -*-
"""
calib_camera_to_xray.py

Estimate the rigid camera->xray transform.

calibrate_camera_to_xray(...) solves ONE board placement; all its
pose-estimation methods are implemented inside
tracking.pose_solvers.solve_pose(...).

This wrapper is intentionally thin:
- validate / normalize inputs
- call solve_pose(...) with the requested method
- convert the returned pose to T_cx / T_xc
- package the result

calibrate_camera_to_xray_pooled(...) combines several board placements
(one 'ippe_handeye' result each) into ONE camera->xray transform by a
single PnP over all their lattice points. That PnP (SQPnP + LM) runs
here, not in solve_pose; see the section at the end.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tracking.pose_solvers import solve_pose, PoseSolveResult, normalize_dist_coeffs
from tracking.transforms import invert_transform, rvec_tvec_to_transform


# ============================================================
# Result container
# ============================================================

@dataclass(frozen=True)
class CameraToXrayCalibrationResult:
    """
    Output of ``calibrate_camera_to_xray``.
    """

    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray

    T_cx: np.ndarray
    T_xc: np.ndarray

    inlier_idx: np.ndarray

    uv_proj: np.ndarray
    reproj_errors_px: np.ndarray
    reproj_mean_px: float
    reproj_median_px: float
    reproj_max_px: float

    pose_result: PoseSolveResult


# ============================================================
# Internal helpers
# ============================================================

def _as_uv(arr: np.ndarray, name: str) -> np.ndarray:
    pts = np.asarray(arr, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N,2), got {pts.shape}")
    return pts


def _as_xyz(arr: np.ndarray, name: str) -> np.ndarray:
    pts = np.asarray(arr, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N,3), got {pts.shape}")
    return pts


def _as_K(arr: np.ndarray, name: str) -> np.ndarray:
    K = np.asarray(arr, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3,3), got {K.shape}")
    return K


def _result_from_pose(
    pose_result: PoseSolveResult,
    T_cx: np.ndarray,
) -> CameraToXrayCalibrationResult:
    rotation, _ = cv2.Rodrigues(pose_result.rvec)
    translation = np.asarray(pose_result.tvec, dtype=np.float64).reshape(3, 1)
    T_cx = np.asarray(T_cx, dtype=np.float64)
    T_xc = invert_transform(T_cx)

    return CameraToXrayCalibrationResult(
        rvec=pose_result.rvec,
        tvec=pose_result.tvec,
        rotation=rotation,
        translation=translation,
        T_cx=T_cx,
        T_xc=T_xc,
        inlier_idx=pose_result.inlier_idx,
        uv_proj=pose_result.uv_proj,
        reproj_errors_px=pose_result.reproj_errors_px,
        reproj_mean_px=pose_result.reproj_mean_px,
        reproj_median_px=pose_result.reproj_median_px,
        reproj_max_px=pose_result.reproj_max_px,
        pose_result=pose_result,
    )


# ============================================================
# Public API
# ============================================================

def calibrate_camera_to_xray(
    K_xray: np.ndarray,
    *,
    points_xyz_camera: np.ndarray | None = None,
    points_uv_xray: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    dist_coeffs_rgb: np.ndarray | None = None,
    pose_method: str = "ippe_handeye",
    refine_with_iterative: bool = False,
    refine_rgb_iterative: bool = False,
    pitch_mm: float = 2.54,
    checkerboard_corners_uv: np.ndarray | None = None,
    K_rgb: np.ndarray | None = None,
    steps_per_edge: int = 10,
    board_thickness_mm: float = 0.0,
    xray_lattice_uv: np.ndarray | None = None,
    xray_lattice_xy_mm: np.ndarray | None = None,
    checkerboard_corners_uv_9: np.ndarray | None = None,
) -> CameraToXrayCalibrationResult:
    """
    Estimate the rigid camera->xray transform.

    Parameters
    ----------
    K_xray : (3,3)
        X-ray intrinsic matrix.
    points_xyz_camera : (N,3) or None
        3D points in the camera frame.

        Required for every method ('iterative', 'ippe', 'ippe_handeye').

        For method='ippe_handeye', these must be the reconstructed board
        points in the camera frame (meters). They are used for the
        depth-based RGB-side IPPE disambiguation.
    points_uv_xray : (N,2)
        Corresponding 2D points in the X-ray image.
    dist_coeffs : array-like or None
        Distortion coefficients for the X-ray side.
        For the current DeCAF prototype this is typically None.
    dist_coeffs_rgb : array-like or None
        Distortion coefficients for the RGB camera.
        Only used by pose_method='ippe_handeye'.
    pose_method : str
        One of 'iterative', 'ippe', 'ippe_handeye' (default; the Cam2X page
        uses only this one, and Cam2XBoardCapture.from_result - the input of
        calibrate_camera_to_xray_pooled - accepts only its results).
    checkerboard_corners_uv : (3,2) or None
        The three extreme checkerboard corners [TL, TR, BL] in the RGB
        image. Required for method='ippe_handeye'.
    K_rgb : (3,3) or None
        RGB camera intrinsic matrix. Required for method='ippe_handeye'.
    steps_per_edge : int
        Grid steps per edge for method='ippe_handeye'. Default 10 → 121 points.
    board_thickness_mm : float
        Only used for method='ippe_handeye'. Offset between the checkerboard
        plane seen by the RGB camera and the copper plane seen by the X-ray.
        See tracking.pose_solvers.solve_pose.
    checkerboard_corners_uv_9 : (9,2) or None
        All nine checkerboard corners. With them the RGB-side patch points
        are interpolated by a perspective homography (recommended); without
        them affinely from the three extreme corners, which biases the board
        tilt by ~2 deg. See tracking.pose_solvers._solve_pose_ippe_handeye.
    xray_lattice_uv, xray_lattice_xy_mm : (M,2), (M,2) or None
        Required for pose_method='ippe_handeye': all indexed board holes in
        the X-ray working space (distortion-corrected like points_uv_xray)
        and their board coordinates in mm. The X-ray-side pose is refined
        on the whole board instead of the 121-point patch, which resolves
        the planar two-fold ambiguity by perspective; see
        tracking.pose_solvers._solve_pose_ippe_handeye.

    Notes
    -----
    For pose_method='ippe_handeye', checkerboard_corners_uv, K_rgb,
    points_xyz_camera and the board lattice (xray_lattice_uv,
    xray_lattice_xy_mm) are required. solve_pose(...) internally runs IPPE
    on both the RGB and X-ray side and composes T_cx = T_bx @ inv(T_bc).

    Refinement options
    ------------------
    refine_with_iterative : bool
        For pose_method='ippe_handeye': LM refinement of the RGB-side pose
        T_bc. The X-ray-side pose is always refined on the board lattice.

    refine_rgb_iterative : bool
        Same as refine_with_iterative for the RGB side.

    """
    K_xray = _as_K(K_xray, "K_xray")
    uv = _as_uv(points_uv_xray, "points_uv_xray")

    method = str(pose_method).lower().strip()

    if points_xyz_camera is None:
        raise ValueError(f"pose_method='{method}' requires points_xyz_camera.")

    xyz = _as_xyz(points_xyz_camera, "points_xyz_camera")
    if xyz.shape[0] != uv.shape[0]:
        raise ValueError(
            "3-D and 2-D point counts must match, "
            f"got xyz={xyz.shape[0]} and uv={uv.shape[0]}."
        )

    if method == "ippe_handeye":
        if checkerboard_corners_uv is None:
            raise ValueError("pose_method='ippe_handeye' requires checkerboard_corners_uv.")
        if K_rgb is None:
            raise ValueError("pose_method='ippe_handeye' requires K_rgb.")

    pose_result = solve_pose(
        object_points_xyz=xyz,
        image_points_uv=uv,
        K=K_xray,
        dist_coeffs=dist_coeffs,
        dist_coeffs_rgb=dist_coeffs_rgb,
        pose_method=pose_method,
        refine_with_iterative=refine_with_iterative,
        refine_rgb_iterative=refine_rgb_iterative,
        pitch_mm=pitch_mm,
        checkerboard_corners_uv=checkerboard_corners_uv,
        K_rgb=K_rgb,
        steps_per_edge=steps_per_edge,
        board_thickness_mm=board_thickness_mm,
        xray_lattice_uv=xray_lattice_uv,
        xray_lattice_xy_mm=xray_lattice_xy_mm,
        checkerboard_corners_uv_9=checkerboard_corners_uv_9,
    )

    T_cx = rvec_tvec_to_transform(pose_result.rvec, pose_result.tvec)
    return _result_from_pose(pose_result, T_cx)


# ============================================================
# Pooled calibration over several board placements
# ============================================================
#
# One flat board placement fixes the camera-side board tilt only to ~2 deg
# (a 25.4 mm printed patch seen obliquely), and the single-placement
# composition T_cx = T_bx @ inv(T_bc) carries that tilt 1:1 into T_cx; the
# overlay then drifts with the distance from the board (measured: 3.7 mm
# mean / 7.6 mm max over 220 mm). With the same board captured at several
# table heights, every lattice hole of every capture is put into the camera
# frame by its camera-side pose and all 3-D (camera) / 2-D (X-ray) pairs go
# into ONE PnP. The points then span a volume, the lever arm between the
# heights fixes the rotation, and the camera-side tilt errors only move the
# holes along the (nearly parallel) X-ray viewing direction, where they
# hardly matter (measured on 19 held-out pin planes: 0.33 mm mean / 0.77 mm
# max). This is the pooling of Wang et al. 2016 ("a planar calibration
# board placed at two different sites equals a 3D calibration phantom"),
# solved as a PnP with known intrinsics (Hartley & Zisserman, Ch. 7).

@dataclass(frozen=True)
class Cam2XBoardCapture:
    """
    What one board placement contributes to the pooled calibration.

    T_bc_mm : (4,4)
        Board -> camera pose of this placement, translation in mm (the
        camera-side result of pose_method='ippe_handeye').
    lattice_xyz_board_mm : (M,3)
        Board coordinates of the indexed holes, z = board thickness (the
        X-ray-visible plane), in mm.
    lattice_uv_xray : (M,2)
        The same holes in the X-ray WORKING space, distortion-corrected
        (as passed to calibrate_camera_to_xray).
    label : str
        Free text for reports (e.g. 'h0', 'h100').
    """

    T_bc_mm: np.ndarray
    lattice_xyz_board_mm: np.ndarray
    lattice_uv_xray: np.ndarray
    label: str = ""

    @classmethod
    def from_result(
        cls,
        result: CameraToXrayCalibrationResult,
        label: str = "",
    ) -> "Cam2XBoardCapture":
        """Take the per-capture inputs from an 'ippe_handeye' result."""
        diag = getattr(result.pose_result, "diagnostics", None) or {}
        missing = [k for k in ("T_bc", "pts3d_lattice_mm", "lattice_uv") if k not in diag]
        if missing:
            raise ValueError(
                "Cam2XBoardCapture.from_result needs a pose_method='ippe_handeye' "
                f"result with the board lattice; missing diagnostics: {missing}."
            )
        return cls(
            T_bc_mm=np.asarray(diag["T_bc"], dtype=np.float64),
            lattice_xyz_board_mm=np.asarray(diag["pts3d_lattice_mm"], dtype=np.float64),
            lattice_uv_xray=np.asarray(diag["lattice_uv"], dtype=np.float64),
            label=str(label),
        )

    def points_xyz_camera_mm(self) -> np.ndarray:
        """The holes of this placement in the camera frame (mm)."""
        T = np.asarray(self.T_bc_mm, dtype=np.float64)
        X = _as_xyz(self.lattice_xyz_board_mm, "lattice_xyz_board_mm")
        return X @ T[:3, :3].T + T[:3, 3]


@dataclass(frozen=True)
class PooledCameraToXrayResult:
    """
    Output of ``calibrate_camera_to_xray_pooled``.

    Same conventions as CameraToXrayCalibrationResult: T_cx maps camera ->
    X-ray, T_xc = inv(T_cx), translations in METRES.
    """

    rvec: np.ndarray
    tvec: np.ndarray
    T_cx: np.ndarray
    T_xc: np.ndarray

    capture_labels: tuple[str, ...]
    capture_index: np.ndarray          # (N,) which capture each pooled point came from
    n_points_per_capture: tuple[int, ...]

    uv_proj: np.ndarray                # (N,2) reprojection of all pooled points
    reproj_errors_px: np.ndarray       # (N,)
    reproj_mean_px: float
    reproj_median_px: float
    reproj_max_px: float
    reproj_mean_px_per_capture: tuple[float, ...]


def _pooled_arrays(captures) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    P, uv, idx = [], [], []
    for i, c in enumerate(captures):
        Pc = c.points_xyz_camera_mm()
        uc = _as_uv(c.lattice_uv_xray, "lattice_uv_xray")
        if Pc.shape[0] != uc.shape[0]:
            raise ValueError(
                f"capture {i}: {Pc.shape[0]} lattice points but {uc.shape[0]} X-ray points."
            )
        P.append(Pc); uv.append(uc); idx.append(np.full(Pc.shape[0], i, dtype=np.int64))
    return np.vstack(P), np.vstack(uv), np.concatenate(idx)


def _solve_pooled_pnp(P_mm, uv, K_xray, dist):
    """SQPnP (global, no start value needed) followed by LM refinement."""
    ok, rvec, tvec = cv2.solvePnP(P_mm, uv, K_xray, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError("SQPnP failed on the pooled correspondences.")
    rvec, tvec = cv2.solvePnPRefineLM(P_mm, uv, K_xray, dist, rvec, tvec)
    return np.asarray(rvec, dtype=np.float64).reshape(3, 1), np.asarray(tvec, dtype=np.float64).reshape(3, 1)


def calibrate_camera_to_xray_pooled(
    K_xray: np.ndarray,
    captures,
    *,
    dist_coeffs: np.ndarray | None = None,
) -> PooledCameraToXrayResult:
    """
    Camera -> X-ray transform from several board placements, one PnP.

    Parameters
    ----------
    K_xray : (3,3)
        X-ray intrinsic matrix (working space).
    captures : sequence of Cam2XBoardCapture
        One entry per board placement (e.g. table at 0 / +100 / +200 mm).
        Each contributes its lattice holes, put into the camera frame by its
        own camera-side pose T_bc; the X-ray-side single-placement poses are
        not used.
    dist_coeffs : array-like or None
        X-ray distortion coefficients. None for the DeCAF pipeline, where
        the lattice points are already corrected by the polynomial field.

    Returns
    -------
    PooledCameraToXrayResult
        T_cx / T_xc in metres, like calibrate_camera_to_xray, plus the
        reprojection error over all points and per placement.
    """
    K_xray = _as_K(K_xray, "K_xray")
    captures = list(captures)
    if not captures:
        raise ValueError("calibrate_camera_to_xray_pooled needs at least one capture.")
    P_mm, uv, idx = _pooled_arrays(captures)
    if P_mm.shape[0] < 6:
        raise ValueError(f"Too few pooled points ({P_mm.shape[0]}).")
    dist = normalize_dist_coeffs(dist_coeffs)

    rvec, tvec_mm = _solve_pooled_pnp(P_mm, uv, K_xray, dist)

    uv_proj, _ = cv2.projectPoints(P_mm, rvec, tvec_mm, K_xray, dist)
    uv_proj = uv_proj.reshape(-1, 2)
    err = np.linalg.norm(uv_proj - uv, axis=1)

    tvec_m = tvec_mm * 1e-3                       # mm -> m, as the single-capture result
    T_cx = rvec_tvec_to_transform(rvec, tvec_m)
    T_xc = invert_transform(T_cx)

    return PooledCameraToXrayResult(
        rvec=rvec,
        tvec=tvec_m,
        T_cx=T_cx,
        T_xc=T_xc,
        capture_labels=tuple(str(c.label) for c in captures),
        capture_index=idx,
        n_points_per_capture=tuple(int((idx == i).sum()) for i in range(len(captures))),
        uv_proj=uv_proj,
        reproj_errors_px=err,
        reproj_mean_px=float(err.mean()),
        reproj_median_px=float(np.median(err)),
        reproj_max_px=float(err.max()),
        reproj_mean_px_per_capture=tuple(float(err[idx == i].mean()) for i in range(len(captures))),
    )


def leave_one_capture_out(
    K_xray: np.ndarray,
    captures,
    *,
    dist_coeffs: np.ndarray | None = None,
) -> list[dict]:
    """
    Consistency check of a pooled calibration: for every placement, solve on
    the others and reproject the left-out placement.

    Returns one dict per placement: label, rotation of the reduced solution
    against the full one (deg), and the mean reprojection error of the
    left-out placement under the reduced solution (px). Needs >= 3 captures:
    with two, every reduced problem is a single flat placement and its
    numbers show the single-board tilt, not the consistency of the pool.
    """
    captures = list(captures)
    if len(captures) < 3:
        raise ValueError(
            f"leave_one_capture_out needs at least 3 captures, got {len(captures)}."
        )
    full = calibrate_camera_to_xray_pooled(K_xray, captures, dist_coeffs=dist_coeffs)
    R_full = full.T_cx[:3, :3]
    K_xray = _as_K(K_xray, "K_xray")
    dist = normalize_dist_coeffs(dist_coeffs)
    out = []
    for i, c in enumerate(captures):
        rest = [cc for j, cc in enumerate(captures) if j != i]
        if not rest:
            continue
        red = calibrate_camera_to_xray_pooled(K_xray, rest, dist_coeffs=dist_coeffs)
        P = c.points_xyz_camera_mm()
        uv = _as_uv(c.lattice_uv_xray, "lattice_uv_xray")
        pr, _ = cv2.projectPoints(P, red.rvec, red.tvec * 1e3, K_xray, dist)
        dR = red.T_cx[:3, :3] @ R_full.T
        ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))))
        out.append({
            "label": str(c.label),
            "rotation_vs_full_deg": ang,
            "heldout_reproj_mean_px": float(np.linalg.norm(pr.reshape(-1, 2) - uv, axis=1).mean()),
        })
    return out
