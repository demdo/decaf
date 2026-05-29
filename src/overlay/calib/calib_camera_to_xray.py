# -*- coding: utf-8 -*-
"""
calib_camera_to_xray.py

Estimate the rigid camera->xray transform.

All supported pose-estimation methods are implemented inside
tracking.pose_solvers.solve_pose(...).

This wrapper is intentionally thin:
- validate / normalize inputs
- call solve_pose(...) with the requested method
- convert the returned pose to T_cx / T_xc
- package the result
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tracking.pose_solvers import solve_pose, PoseSolveResult
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

    inliers: np.ndarray | None
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
        inliers=pose_result.inliers,
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
    pose_method: str = "iterative_ransac",
    refine_with_iterative: bool = False,
    refine_rgb_iterative: bool = False,
    refine_xray_iterative: bool = False,
    ransac_reprojection_error_px: float = 3.0,
    ransac_confidence: float = 0.99,
    ransac_iterations_count: int = 5000,
    pitch_mm: float = 2.54,
    checkerboard_corners_uv: np.ndarray | None = None,
    K_rgb: np.ndarray | None = None,
    steps_per_edge: int = 10,
) -> CameraToXrayCalibrationResult:
    """
    Estimate the rigid camera->xray transform.

    Parameters
    ----------
    K_xray : (3,3)
        X-ray intrinsic matrix.
    points_xyz_camera : (N,3) or None
        3D points in the camera frame.

        Required for method='iterative', 'iterative_ransac', 'ippe', and
        'ippe_handeye'.

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
        One of 'iterative', 'iterative_ransac', 'ippe', 'ippe_handeye'.
    checkerboard_corners_uv : (3,2) or None
        The three extreme checkerboard corners [TL, TR, BL] in the RGB
        image. Required for method='ippe_handeye'.
    K_rgb : (3,3) or None
        RGB camera intrinsic matrix. Required for method='ippe_handeye'.
    steps_per_edge : int
        Grid steps per edge for method='ippe_handeye'. Default 10 → 121 points.

    Notes
    -----
    For pose_method='ippe_handeye', checkerboard_corners_uv, K_rgb, and
    points_xyz_camera are required. solve_pose(...) internally runs IPPE
    on both the RGB and X-ray side and composes T_cx = T_bx @ inv(T_bc).

    Refinement options
    ------------------
    refine_with_iterative : bool
        Backward-compatible convenience switch. For pose_method='ippe_handeye',
        this enables iterative refinement on both RGB and X-ray side.

    refine_rgb_iterative : bool
        Only relevant for pose_method='ippe_handeye'. If True, refine T_bc.

    refine_xray_iterative : bool
        Only relevant for pose_method='ippe_handeye'. If True, refine T_bx.
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
        refine_xray_iterative=refine_xray_iterative,
        ransac_reprojection_error_px=ransac_reprojection_error_px,
        ransac_confidence=ransac_confidence,
        ransac_iterations_count=ransac_iterations_count,
        pitch_mm=pitch_mm,
        checkerboard_corners_uv=checkerboard_corners_uv,
        K_rgb=K_rgb,
        steps_per_edge=steps_per_edge,
    )

    T_cx = rvec_tvec_to_transform(pose_result.rvec, pose_result.tvec)
    return _result_from_pose(pose_result, T_cx)