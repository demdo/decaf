# overlay/debug/test_camera_to_xray_handeye.py

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication, QFileDialog

import cv2
import numpy as np

from overlay.calib.calib_camera_to_xray_handeye import (
    calibrate_camera_to_xray_handeye,
    compose_camera_to_xray,
    pose_vectors_to_board_to_xray,
)
from tracking.transforms import (
    make_transform,
    as_transform,
)


# ============================================================
# Config
# ============================================================

BOARD_ROWS = 11
BOARD_COLS = 11
PITCH_MM = 2.54

RANSAC_REPROJECTION_ERROR_PX = 2.0
RANSAC_CONFIDENCE = 0.99
RANSAC_ITERATIONS_COUNT = 5000


# ============================================================
# File dialogs
# ============================================================

def _ensure_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def select_pose_files():
    _ensure_app()

    dlg = QFileDialog()
    dlg.setWindowTitle("Select pose_debug files")
    dlg.setFileMode(QFileDialog.ExistingFiles)
    dlg.setNameFilter("NumPy files (*.npz)")
    dlg.setOption(QFileDialog.DontUseNativeDialog, True)

    if dlg.exec() != QFileDialog.Accepted:
        raise RuntimeError("No pose files selected")

    return [Path(p) for p in dlg.selectedFiles()]


def select_K_file():
    _ensure_app()

    dlg = QFileDialog()
    dlg.setWindowTitle("Select Kx file")
    dlg.setFileMode(QFileDialog.ExistingFile)
    dlg.setNameFilter("NumPy files (*.npz)")
    dlg.setOption(QFileDialog.DontUseNativeDialog, True)

    if dlg.exec() != QFileDialog.Accepted:
        raise RuntimeError("No K file selected")

    return Path(dlg.selectedFiles()[0])


# ============================================================
# Load helpers
# ============================================================

def load_pose(path: Path):
    with np.load(str(path), allow_pickle=False) as npz:
        xyz = np.asarray(npz["points_xyz"], dtype=np.float64)
        uv = np.asarray(npz["points_uv"], dtype=np.float64)

    return xyz, uv


def load_K(path: Path):
    with np.load(str(path), allow_pickle=False) as npz:
        for key in ["Kx", "K", "K_xray"]:
            if key in npz.files:
                return np.asarray(npz[key], dtype=np.float64)

    raise KeyError("No K found in file")


# ============================================================
# Board geometry
# ============================================================

def make_board_points():
    xs = np.arange(BOARD_COLS) * PITCH_MM
    ys = np.arange(BOARD_ROWS) * PITCH_MM

    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ZZ = np.zeros_like(XX)

    pts_mm = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    return pts_mm / 1000.0  # meters


# ============================================================
# Rigid fit (board -> camera)
# ============================================================

def fit_rigid(A, B):
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return make_transform(R, t)


# ============================================================
# Main
# ============================================================

def main():
    np.set_printoptions(precision=6, suppress=True)

    print("=" * 60)
    print("SELECT FILES")
    print("=" * 60)

    pose_files = select_pose_files()
    K_file = select_K_file()

    Kx = load_K(K_file)
    board_points = make_board_points()

    T_bc_list = []
    T_bx_list = []

    print("\n" + "=" * 60)
    print("PER VIEW")
    print("=" * 60)

    for path in pose_files:
        xyz_cam, uv = load_pose(path)

        # --- T_bc ---
        T_bc = fit_rigid(board_points, xyz_cam)

        # --- T_bx via PnP ---
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=board_points,
            imagePoints=uv,
            cameraMatrix=Kx,
            distCoeffs=None,
            iterationsCount=RANSAC_ITERATIONS_COUNT,
            reprojectionError=RANSAC_REPROJECTION_ERROR_PX,
            confidence=RANSAC_CONFIDENCE,
        )

        if not ok:
            raise RuntimeError("PnP failed")

        T_bx = pose_vectors_to_board_to_xray(rvec, tvec)

        # debug
        T_cx_direct = compose_camera_to_xray(T_bc, T_bx)

        print("\n---", path.name)
        print("T_cx_direct:")
        print(T_cx_direct)

        T_bc_list.append(T_bc)
        T_bx_list.append(T_bx)

    print("\n" + "=" * 60)
    print("HAND-EYE")
    print("=" * 60)

    T_cx = calibrate_camera_to_xray_handeye(
        T_bc_list=T_bc_list,
        T_bx_list=T_bx_list,
        method="park",
    )

    print("\nT_cx (camera -> xray):")
    print(T_cx)


if __name__ == "__main__":
    main()