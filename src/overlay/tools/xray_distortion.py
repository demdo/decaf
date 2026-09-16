# -*- coding: utf-8 -*-
"""
xray_distortion.py

Free-form distortion model for the X-ray image intensifier.

Why not OpenCV distortion coefficients
--------------------------------------
Brown's model has only r^2, r^4, r^6 terms. The measured field of this
intensifier is dominated, in the gauge that matters, by a LINEAR radial term
(~1.5 % scale) that Brown's model structurally cannot express. Every attempt
to fit a k1 therefore drags in a wrong scale:

  - fitted from one planar view with a free homography: the homography absorbs
    the scale, k1 comes out at 2.24 and shifts the image by 9 px at r = 450
    where the real distortion is 2 px
  - fitted by the multi-view BA with K1 released: focal length and k1 are
    degenerate for a planar target, f jumps 18 % for a 5.7 % rms gain and the
    overlay breaks (d_x off by 90 mm)

A polynomial displacement field has the linear term and fits ~30 % better.

Conventions
-----------
The model lives in RAW detector coordinates - that is where the physics is
defined. The pipeline works in the horizontally flipped WORKING space, so the
*_working helpers flip in, correct, and flip back.

The inverse is obtained by fixed point iteration, not by fitting a second
polynomial: a separately fitted inverse leaves a round-trip error of several
tenths of a pixel, which goes straight into the image remap.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _poly_features(uv: np.ndarray, centre: np.ndarray, scale: float,
                   degree: int) -> np.ndarray:
    x = (uv[:, 0] - centre[0]) / scale
    y = (uv[:, 1] - centre[1]) / scale
    cols = [x ** i * y ** j
            for d in range(degree + 1) for i in range(d + 1) for j in [d - i]]
    return np.column_stack(cols)


@dataclass(frozen=True)
class XrayFieldDistortion:
    """Polynomial displacement field, distorted -> ideal, in RAW pixels."""

    coeff: np.ndarray        # (n_terms, 2)
    degree: int
    centre: np.ndarray       # (2,) RAW pixels
    scale: float
    width: int               # detector width, for the RAW <-> WORKING flip

    # ---------------- RAW space ----------------

    def undistort(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        return uv + _poly_features(uv, self.centre, self.scale, self.degree) @ self.coeff

    def distort(self, uv: np.ndarray, iters: int = 12) -> np.ndarray:
        target = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        x = target.copy()
        for _ in range(iters):
            x = target - _poly_features(x, self.centre, self.scale, self.degree) @ self.coeff
        return x

    # ---------------- WORKING space ----------------

    def _to_raw(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        return np.column_stack([self.width - 1 - uv[:, 0], uv[:, 1]])

    def undistort_working(self, uv: np.ndarray) -> np.ndarray:
        ideal = self.undistort(self._to_raw(uv))
        return np.column_stack([self.width - 1 - ideal[:, 0], ideal[:, 1]])

    def undistort_image_working(self, img: np.ndarray) -> np.ndarray:
        """Resample a WORKING-space image into the ideal frame."""
        h, w = img.shape[:2]
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float64),
                             np.arange(h, dtype=np.float64))
        ideal_work = np.stack([uu.ravel(), vv.ravel()], axis=1)
        src_raw = self.distort(self._to_raw(ideal_work))
        src_u = self.width - 1 - src_raw[:, 0]
        return cv2.remap(
            img,
            np.ascontiguousarray(src_u.reshape(h, w).astype(np.float32)),
            np.ascontiguousarray(src_raw[:, 1].reshape(h, w).astype(np.float32)),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    # ---------------- diagnostics ----------------

    def round_trip_error(self, uv: np.ndarray) -> float:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        return float(np.abs(self.distort(self.undistort(uv)) - uv).max())

    def describe(self) -> str:
        return (f"degree {self.degree}, {len(self.coeff)} terms/axis, "
                f"centre ({self.centre[0]:.0f}, {self.centre[1]:.0f})")


def load_xray_distortion(npz_path: str) -> XrayFieldDistortion | None:
    """
    Read the polynomial field from an intrinsics NPZ.

    Returns None when the file carries no field, so callers can simply skip
    the correction and keep their previous behaviour.
    """
    try:
        d = np.load(str(npz_path), allow_pickle=True)
    except Exception:
        return None

    required = ("poly_coeff", "poly_degree", "poly_centre", "poly_scale")
    if not all(k in d.files for k in required):
        return None

    size = d["image_size"] if "image_size" in d.files else np.array([1024, 1024])
    return XrayFieldDistortion(
        coeff=np.asarray(d["poly_coeff"], dtype=np.float64),
        degree=int(d["poly_degree"]),
        centre=np.asarray(d["poly_centre"], dtype=np.float64).reshape(2),
        scale=float(d["poly_scale"]),
        width=int(np.asarray(size).ravel()[0]),
    )
