# overlay/tools/xray_view.py
"""The tracked tool as the X-ray sees it (Thesis 6.2.1). Geometry only, no Qt.

Frames and units, the two traps here:

    T_xc maps X-ray -> camera (the thesis calls this mapping T_cx) and is
    stored in METRES: P_c = R P_x + t. Tool points and d_x are MILLIMETRES.

The origin of the X-ray frame is the source. The reference plane is the one
the live overlay warps the X-ray onto (estimate_plane_induced_homography):

    Pi:  n^T P_x = d_x,   n = (0, 0, 1)

The source ray through a point P_x is { lambda P_x }. Putting it into the plane
equation gives lambda = d_x / (n^T P_x), so with t in mm:

    P_x  = R^T (P_c - t)            the point in the X-ray frame
    uv_x = K_x P_x / z_x            its shadow in the working image (flipped,
                                    undistorted); d_x does not enter
    S_x  = d_x / (n^T P_x) P_x      the ray cut with Pi
    S_c  = R S_x + t                the same point in the camera frame

Projected through the colour lens, S_c lands exactly where the live overlay
shows the X-ray pixel uv_x: S_c is H_xc(d_x) uv_x. A wrong d_x therefore moves
the X-ray content and this view together.

XrayField is the collimated field of one X-ray image, in the working image.
It is convex there; through the colour lens it no longer strictly is.

The live page runs pixel() and XrayField's tests once per pose in the GUI
thread, so they work on plain floats: a dozen small numpy calls cost 0.4 ms
there, against 0.07 ms in an offline loop (cold caches next to the tracker).
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from overlay.tools.warp import _detect_xray_fov_mask

# |t| of a real set-up is about a metre; a value above this is millimetres.
_MAX_T_XC_NORM_M = 10.0


class XrayView:
    """Camera-frame points (mm) seen from the X-ray source."""

    def __init__(self, T_xc_m, K_x) -> None:
        T = np.asarray(T_xc_m, dtype=np.float64).reshape(4, 4)
        t_m = T[:3, 3]
        if float(np.linalg.norm(t_m)) > _MAX_T_XC_NORM_M:
            raise ValueError(
                f"T_xc translation {t_m} is not in metres (see overlay.tools.xray_view)"
            )
        self.R = T[:3, :3].copy()
        self.t_mm = t_m * 1000.0
        self.K_x = np.asarray(K_x, dtype=np.float64).reshape(3, 3).copy()
        if not np.array_equal(self.K_x[2], [0.0, 0.0, 1.0]):
            raise ValueError(f"K_x must end in (0, 0, 1), got {self.K_x[2]}")
        self._K_x_inv = np.linalg.inv(self.K_x)
        self._fx, self._skew, self._cx = self.K_x[0].tolist()
        self._fy, self._cy = self.K_x[1, 1:].tolist()

    def to_xray(self, P_c) -> np.ndarray:
        """(N, 3) P_x in mm: R^T (P_c - t), written for rows."""
        return (np.asarray(P_c, dtype=np.float64).reshape(-1, 3) - self.t_mm) @ self.R

    def uv_from_xray(self, P_x) -> np.ndarray:
        """(N, 2) working-image pixels of X-ray-frame points."""
        p = np.asarray(P_x, dtype=np.float64).reshape(-1, 3) @ self.K_x.T
        return p[:, :2] / p[:, 2:]

    def pixel(self, x: float, y: float, z: float) -> tuple[float, float]:
        """uv_from_xray of one point, on plain floats."""
        return (self._fx * x + self._skew * y) / z + self._cx, self._fy * y / z + self._cy

    def uv_x(self, P_c) -> np.ndarray:
        """(N, 2) working-image pixels of camera-frame points."""
        return self.uv_from_xray(self.to_xray(P_c))

    def on_plane_cam(self, P_c, d_x: float) -> np.ndarray:
        """(N, 3) S_c: each point moved along its source ray onto Pi, mm."""
        P_x = self.to_xray(P_c)
        return (P_x * (float(d_x) / P_x[:, 2:])) @ self.R.T + self.t_mm

    def plane_matrix(self, d_x: float) -> np.ndarray:
        """M with S_c = M (u, v, 1) + t for working-image pixels on Pi:
        S_x = d_x K_x^-1 (u, v, 1)."""
        return float(d_x) * (self.R @ self._K_x_inv)

    def pixels_on_plane_cam(self, uv_x, d_x: float) -> np.ndarray:
        """(N, 3) S_c of working-image pixels."""
        uv = np.asarray(uv_x, dtype=np.float64).reshape(-1, 2)
        return np.c_[uv, np.ones(len(uv))] @ self.plane_matrix(d_x).T + self.t_mm

    def off_plane_mm(self, P_c, d_x: float) -> np.ndarray:
        """(N,) n^T P_x - d_x; positive: farther from the source than Pi."""
        return self.to_xray(P_c)[:, 2] - float(d_x)

    def inclination_deg(self, tip_c, back_c) -> float:
        """Angle between the axis tip-back and Pi: arcsin |n^T R^T a|."""
        P_x = self.to_xray(np.stack([
            np.asarray(tip_c, dtype=np.float64).reshape(3),
            np.asarray(back_c, dtype=np.float64).reshape(3),
        ]))
        return axis_inclination_deg(P_x[0], P_x[1])


def axis_inclination_deg(tip_x, back_x) -> float:
    """Inclination against Pi from two X-ray-frame points."""
    d = np.asarray(back_x, dtype=np.float64) - np.asarray(tip_x, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n <= 0.0:
        return float("nan")
    return math.degrees(math.asin(min(1.0, abs(float(d[2])) / n)))


class XrayField:
    """The convex field of one X-ray image, in working-image pixels."""

    def __init__(self, hull) -> None:
        hull = np.asarray(hull, dtype=np.float64).reshape(-1, 2)
        if len(hull) < 3:
            raise ValueError("a field needs at least three hull points")
        self.hull = hull
        edge = np.roll(hull, -1, axis=0) - hull
        normal = np.c_[edge[:, 1], -edge[:, 0]]
        # inward, whichever way round the hull runs
        flip = ((hull.mean(axis=0) - hull) * normal).sum(axis=1) < 0.0
        normal[flip] *= -1.0
        # n . x >= offset inside, per edge
        self._edges = list(zip(*normal.T.tolist(), (normal * hull).sum(axis=1).tolist()))

    @classmethod
    def from_image(cls, working_u8: np.ndarray, inset_px: float = 0.0,
                   tolerance_px: float = 2.0) -> "XrayField":
        """Hull of the mask the live overlay warps (warp._detect_xray_fov_mask),
        moved inward by inset_px. One image, one call: ~10 ms, not per frame.
        tolerance_px thins the hull (its corners stay, so it only shrinks)."""
        mask = _detect_xray_fov_mask(working_u8)
        if inset_px > 0.0:
            k = 2 * int(math.ceil(inset_px)) + 1
            # outside the image counts as outside the field
            mask = cv2.erode(
                mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                borderType=cv2.BORDER_CONSTANT, borderValue=0,
            )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise RuntimeError("no X-ray field left inside the inset")
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        return cls(cv2.approxPolyDP(hull, float(tolerance_px), True))

    def contains(self, uv) -> bool:
        u, v = float(uv[0]), float(uv[1])
        return all(nx * u + ny * v >= off for nx, ny, off in self._edges)

    def clip_line(self, a, b) -> Optional[tuple[float, float]]:
        """Part of the infinite line a + s (b - a) inside the field: (s_in, s_out),
        None if it misses. One pass for all pieces of a line, they are collinear;
        a itself is inside if s_in <= 0 <= s_out."""
        ax, ay = float(a[0]), float(a[1])
        dx, dy = float(b[0]) - ax, float(b[1]) - ay
        if dx == 0.0 and dy == 0.0:
            return None
        s_in, s_out = -math.inf, math.inf
        for nx, ny, off in self._edges:
            den = nx * dx + ny * dy
            num = off - nx * ax - ny * ay
            if den > 1e-12:
                r = num / den
                if r > s_in:
                    s_in = r
            elif den < -1e-12:
                r = num / den
                if r < s_out:
                    s_out = r
            elif num > 0.0:
                return None     # runs outside, along an edge
        return (s_in, s_out) if s_in <= s_out else None
