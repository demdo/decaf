# -*- coding: utf-8 -*-
"""
test_warp_paths.py

Checks that the live overlay warp and the preview warp are the same warp.

(a) prepare_live_overlay with the lens model == warp_xray_to_camera with the
    lens model, pixel for pixel inside the live ROI (image and mask).
(b) prepare_live_overlay without a lens model == the plain projective warp
    (the behaviour the live page had before the lens model was added).
(c) with vs. without the lens model the overlay actually moves: zero at the
    principal point, several pixels towards the edge of the frame.

No camera needed. Run from src/:

    python -m overlay.tests.test_warp_paths
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from overlay.tools.warp import prepare_live_overlay, warp_xray_to_camera


CALIB = (
    Path(__file__).resolve().parents[2]
    / "tracking" / "hydramarker" / "data" / "realsense"
    / "realsense_camera_calibration_rational8.npz"
)
CAM_WH = (1920, 1080)


def _synthetic_xray(size: int = 1024) -> np.ndarray:
    """Dark frame, bright circular detector field with a textured interior."""
    img = np.zeros((size, size), np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - size / 2, yy - size / 2)
    field = r < 0.46 * size
    texture = (128 + 60 * np.sin(xx / 17.0) * np.cos(yy / 23.0)).astype(np.uint8)
    img[field] = texture[field]
    return img


def _homography() -> np.ndarray:
    """X-ray px -> camera px: a shrink plus an off-centre shift, mild tilt."""
    return np.array(
        [
            [0.42, 0.02, 480.0],
            [-0.01, 0.41, 260.0],
            [1.5e-5, -1.0e-5, 1.0],
        ],
        dtype=np.float64,
    )


def main() -> None:
    d = np.load(CALIB, allow_pickle=True)
    K = np.asarray(d["K_rgb"], np.float64)
    dist = np.asarray(d["dist_rgb"], np.float64).reshape(-1)

    xray = _synthetic_xray()
    H = _homography()

    # (a) live == preview with the lens model
    live = prepare_live_overlay(xray, H, CAM_WH, K_camera=K, dist_coeffs=dist)
    prev = warp_xray_to_camera(xray, H, CAM_WH, K_camera=K, dist_coeffs=dist)
    assert live is not None
    y0, y1, x0, x1 = live.box
    assert np.array_equal(live.xray_roi[:, :, 0], prev.warped_xray_gray[y0:y1, x0:x1])
    assert np.array_equal(live.mask_roi, prev.overlay_mask[y0:y1, x0:x1])
    # nothing of the mask lies outside the ROI
    outside = prev.overlay_mask.copy()
    outside[y0:y1, x0:x1] = 0
    assert not outside.any()
    print("(a) live == preview with lens model: ok, ROI", live.box)

    # (b) live without the lens model == plain projective warp
    live0 = prepare_live_overlay(xray, H, CAM_WH)
    assert live0 is not None
    from overlay.tools.warp import _detect_xray_fov_mask
    fov = _detect_xray_fov_mask(xray)
    masked = cv2.bitwise_and(xray, xray, mask=fov)
    plain = cv2.warpPerspective(masked, H, CAM_WH, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    plain_mask = cv2.warpPerspective(fov, H, CAM_WH, flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    plain_mask = (plain_mask > 0).astype(np.uint8) * 255
    b0 = live0.box
    assert np.array_equal(live0.xray_roi[:, :, 0], plain[b0[0]:b0[1], b0[2]:b0[3]])
    assert np.array_equal(live0.mask_roi, plain_mask[b0[0]:b0[1], b0[2]:b0[3]])
    print("(b) live without lens model == plain projective warp: ok")

    # (c) the lens model moves the overlay: compare the two full-frame masks
    m_dist = prev.overlay_mask > 0
    m_plain = plain_mask > 0
    diff = np.logical_xor(m_dist, m_plain)
    assert diff.any(), "lens model had no effect"
    # displacement of the field boundary: distance transform of one mask
    # evaluated on the other mask's boundary
    edge_plain = cv2.Canny(plain_mask, 50, 150) > 0
    dt = cv2.distanceTransform((~edge_plain).astype(np.uint8), cv2.DIST_L2, 5)
    edge_dist = cv2.Canny(prev.overlay_mask, 50, 150) > 0
    shifts = dt[edge_dist]
    print(f"(c) boundary shift with vs. without lens model: "
          f"median {np.median(shifts):.2f} px, max {shifts.max():.2f} px")
    assert shifts.max() > 1.0

    # near the principal point the two warps must agree
    cx, cy = int(round(K[0, 2])), int(round(K[1, 2]))
    if m_dist[cy, cx] and m_plain[cy, cx]:
        a = prev.warped_xray_gray[cy - 2:cy + 3, cx - 2:cx + 3].astype(int)
        b = plain[cy - 2:cy + 3, cx - 2:cx + 3].astype(int)
        print(f"    at the principal point: max |diff| = {np.abs(a - b).max()} grey levels")

    print("all checks passed")


if __name__ == "__main__":
    main()
