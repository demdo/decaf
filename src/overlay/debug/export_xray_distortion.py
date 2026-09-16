# -*- coding: utf-8 -*-
"""
export_xray_distortion.py

Write the X-ray detector distortion to a single NPZ, alongside the intrinsics
it belongs to.

Two models are stored, because they are not interchangeable:

  K_xray / dist_xray
      Standard OpenCV pair, drop-in for solvePnP / IPPE / cv2.undistort.
      dist_xray was fitted with K_xray held FIXED at the production value, the
      unknown board pose absorbed by a free homography:

          minimise over (k1, H)   || undistort(p_i; K, k1) - H * X_i ||

      Fitting it against this K is what makes it compatible with the rest of
      the chain - a k1 taken from a calibration that also re-estimated f must
      not be combined with a different f.

  poly_*
      Free-form polynomial field, distorted -> ideal pixels. Fits ~30 % better
      than the radial model because the field is not purely radial, but it
      cannot be expressed as OpenCV distortion coefficients. The inverse is
      obtained by fixed-point iteration, not by a second fit.

Both live in RAW detector coordinates. A radial model about the image centre
is invariant under the RAW<->WORKING horizontal flip; the polynomial is not,
so flip before applying it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from debug_overlay_optimization import fit_xray_distortion, PITCH_MM  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
EVAL = DATA / "Evaluation"
OUT = DATA / "K_xray_dist.npz"

# the production intrinsics this distortion is fitted against
K_XRAY = np.array([[4586.3058, 0.0, 512.0],
                   [0.0, 4586.3058, 512.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)


def _indexed_holes() -> tuple[np.ndarray, np.ndarray]:
    """All detected holes with their integer lattice coordinates."""
    for f in sorted(EVAL.glob("xray_marker_selection_*.npz")):
        d = np.load(str(f), allow_pickle=True)
        if "circles_uvr" in d.files and len(d["circles_uvr"]) > 200:
            break
    else:
        raise FileNotFoundError(
            "no marker-selection NPZ with circles_uvr in " + str(EVAL))

    circles = np.asarray(d["circles_uvr"], dtype=np.float64)[:, :2]
    sel = np.asarray(d["points_uv_raw"], dtype=np.float64).reshape(-1, 2)

    b121 = np.array([[j * PITCH_MM, i * PITCH_MM]
                     for i in range(11) for j in range(11)])
    H, _ = cv2.findHomography(b121.reshape(-1, 1, 2), sel.reshape(-1, 1, 2), 0)

    ok = lat = None
    for _ in range(6):
        b = cv2.perspectiveTransform(
            circles.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2) / PITCH_MM
        lat = np.rint(b)
        err = np.linalg.norm(b - lat, axis=1)
        ok, seen = np.zeros(len(circles), bool), set()
        for t in np.argsort(err):
            if err[t] >= 0.30:
                continue
            key = (int(lat[t, 0]), int(lat[t, 1]))
            if key not in seen:
                seen.add(key)
                ok[t] = True
        H, _ = cv2.findHomography((lat[ok] * PITCH_MM).reshape(-1, 1, 2),
                                  circles[ok].reshape(-1, 1, 2), 0)
    return circles[ok], lat[ok] * PITCH_MM


def fit_k1(pts: np.ndarray, XY: np.ndarray) -> tuple[np.ndarray, float]:
    def residual(kvec):
        d = np.zeros(5)
        d[:len(kvec)] = kvec
        und = cv2.undistortPoints(pts.reshape(-1, 1, 2), K_XRAY, d, P=K_XRAY).reshape(-1, 2)
        H, _ = cv2.findHomography(XY.reshape(-1, 1, 2), und.reshape(-1, 1, 2), 0)
        return (und - cv2.perspectiveTransform(
            XY.reshape(-1, 1, 2), H).reshape(-1, 2)).ravel()

    sol = least_squares(residual, [2.0], method="lm", xtol=1e-12, ftol=1e-12)
    r = residual(sol.x).reshape(-1, 2)
    dist = np.zeros((5, 1))
    dist[0, 0] = float(sol.x[0])
    return dist, float(np.sqrt((r ** 2).sum(axis=1).mean()))


def main() -> None:
    pts, XY = _indexed_holes()
    print(f"holes used: {len(pts)}")

    k1_fit, rms_k1 = fit_k1(pts, XY)
    print(f"k1 would be {k1_fit[0,0]:+.4f} (residual {rms_k1:.3f} px) "
          f"- NOT written, see below")

    poly = fit_xray_distortion(EVAL)
    print(f"polynomial degree {poly.degree}   residual rms {poly.fit_rms:.3f} px")

    # No dist_xray is written.
    #
    # Brown's model has no linear term, but the measured field is dominated by
    # one. Both routes to a k1 therefore fail: fitted from a single view the
    # free homography absorbs the scale (k1 = 2.24 shifts the image 9 px where
    # the distortion is 2 px), and the multi-view BA with K1 released is
    # degenerate for a planar target (f jumps 18 %, d_x off by 90 mm).
    # Both were measured, both make the overlay worse. The pipeline uses poly_*
    # via overlay.tools.xray_distortion and has no Brown path for the X-ray.

    np.savez(
        str(OUT),
        K_xray=K_XRAY,
        # --- the free-form field ---
        poly_coeff=poly.fwd,
        poly_degree=np.array(poly.degree, dtype=np.int32),
        poly_centre=poly.centre,
        poly_scale=np.array(poly.scale, dtype=np.float64),
        poly_direction=np.array("distorted_to_ideal_displacement", dtype="<U64"),
        # --- provenance ---
        uv_space=np.array("XRAY_RAW", dtype="<U64"),
        image_size=np.array([1024, 1024], dtype=np.int32),
        n_holes=np.array(len(pts), dtype=np.int32),
        field_rms_px=np.array(poly.raw_rms),
        field_max_px=np.array(poly.raw_max),
        residual_k1_px=np.array(rms_k1),
        residual_poly_px=np.array(poly.fit_rms),
        residual_k1_would_be=np.array(k1_fit[0, 0]),
        note=np.array(
            "no Brown dist_xray on purpose - it cannot model this field; "
            "use poly_* via overlay.tools.xray_distortion",
            dtype="<U128"),
    )
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
