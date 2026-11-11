# overlay/tools/linear_light_corners.py
"""Corners of dark squares on white (the pointer's ArUco markers) in LINEAR LIGHT.

Why (Eckentyp-Bias, 14.10.2026; confirmed on the pin planes of 11.11.2026)
---------------------------------------------------------------------------
The colour camera does not output light but a grey value through a tone curve
(gamma 300: darks lifted, brights compressed). An edge blurred by the optics is
symmetric in light, its true position the 50 % point between dark and bright;
after the tone curve the 50 % grey is reached earlier on the dark side, so the
edge moves INTO the dark square. Every ArUco square looks 1.5-3 % smaller, the
pointer pose ~0.4 % too far - and more so further away, where the blur is wider.
On 11.11. that put the reference plane 2.7 mm too shallow at 563 mm and 0.9 mm
too deep at 371 mm (pins against the pointer); in linear light the difference
became constant (+1.2 / +1.6 / +2.0 mm, the tip and pin-head heights of that set-up),
the pin RMS over three planes 0.71 -> 0.41 mm.

Saddle corners (checkerboard X-junctions: Cam2X board, drill label, ChArUco
boards) are immune - the shifts from both sides cancel. Only outline corners,
i.e. the pointer, need this.

Method (K6 "linM" of 14.10., ported 1:1, vectorised)
----------------------------------------------------
Per marker, starting from the production corners: 15 grey profiles across
each edge (18-82 % of its length, -1.6..+1.9 mm across, 0.04 mm steps) in
the undistorted frame, sampled bilinearly in the raw image; each profile goes
through the inverse tone curve; the edge point is where it crosses 50 %
between the dark plateau (-1.5..-0.9 mm) and the bright one (+1.0..+1.6 mm),
nearest to the production edge (within +-0.9 mm). A line through each edge's
points; the corners are where neighbouring lines meet, back in raw pixels.

The tone curve was measured on 06.10.2026 from two exposures of one scene
(7.8 / 3.9 ms; K6 s4_tonecurve) and holds for the colour options the pages
now set at every start (camera_setup, CameraConfig: gamma 300, contrast 50,
...). Changing those options needs a new curve.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

# K6 s4_tonecurve.json (06.10.2026): log h(E) piecewise linear over log(knots),
# slopes exp(par), anchored at 0 at the brightest knot.
TONE_KNOTS = (4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0, 110.0,
              130.0, 150.0, 175.0, 200.0, 225.0, 250.0)
TONE_PAR = (-1.090073108514658, -0.9636206192117543, -1.5476796527114316, -1.3911373952027903,
            -1.7943261738072258, -1.638950952959537, -1.2933608097867357, -1.509691120666697,
            -1.5330403607692167, -1.2604608591573945, -1.4070781529651712, -1.1555109990226442,
            -1.250980053875131, -1.311262876035415, -1.2322627236503447, -1.0467800471681254,
            -1.172272576874403, -1.2837289388182693)

# profile geometry (mm across the edge, fractions along it) as K6
_OFFSETS_MM = np.arange(-1.6, 1.9001, 0.04)
_ALONG = np.linspace(0.18, 0.82, 15)
_DARK = (-1.5, -0.9)
_BRIGHT = (1.0, 1.6)
_SEARCH = (-0.9, 0.9)
_LEVEL = 0.5
_MIN_POINTS = 8          # edge points a line needs

def _lagrange_weights(x: np.ndarray, nodes) -> np.ndarray:
    """(len(x), 3) quadratic Lagrange weights of the three nodes."""
    x0, x1, x2 = nodes
    return np.stack([(x - x1) * (x - x2) / ((x0 - x1) * (x0 - x2)),
                     (x - x0) * (x - x2) / ((x1 - x0) * (x1 - x2)),
                     (x - x0) * (x - x1) / ((x2 - x0) * (x2 - x1))], axis=1)


_LAGRANGE = _lagrange_weights(_OFFSETS_MM, (_OFFSETS_MM[0], 0.5 * (_OFFSETS_MM[0] + _OFFSETS_MM[-1]),
                                            _OFFSETS_MM[-1]))

_m_dark = (_OFFSETS_MM >= _DARK[0]) & (_OFFSETS_MM <= _DARK[1])
_m_bright = (_OFFSETS_MM >= _BRIGHT[0]) & (_OFFSETS_MM <= _BRIGHT[1])
_m_search = (_OFFSETS_MM >= _SEARCH[0]) & (_OFFSETS_MM <= _SEARCH[1])
_o_search = _OFFSETS_MM[_m_search]


def linear_light(grey: np.ndarray) -> np.ndarray:
    """Grey values (any float array, 0..255) -> linear light, up to scale (K6 hlin)."""
    knots = np.asarray(TONE_KNOTS, float)
    y = np.zeros(len(knots))
    for k in range(len(knots) - 2, -1, -1):
        y[k] = y[k + 1] - math.exp(TONE_PAR[k])
    e = np.clip(np.asarray(grey, float), knots[0], knots[-1])
    return np.exp(np.interp(np.log(e), np.log(knots), y))


def _to_ideal(uv: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """raw px -> normalised ideal coordinates."""
    return cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)


def _to_raw(xn: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """normalised ideal coordinates -> raw px."""
    xn = np.asarray(xn, float).reshape(-1, 2)
    uv, _ = cv2.projectPoints(np.c_[xn, np.ones(len(xn))], np.zeros(3), np.zeros(3), K, dist)
    return uv.reshape(-1, 2)


def _crossings(v: np.ndarray) -> np.ndarray:
    """(n, len(_OFFSETS_MM)) linear profiles -> crossing offset (mm) per profile, nan if none
    (K6 crossing, vectorised: the crossing nearest to the production edge)."""
    d = np.median(v[:, _m_dark], axis=1)
    b = np.median(v[:, _m_bright], axis=1)
    ok = (b - d) >= 5e-3 * np.maximum(1.0, np.abs(b))
    w = v[:, _m_search] - (d + _LEVEL * (b - d))[:, None]
    s = np.sign(w)
    change = s[:, :-1] * s[:, 1:] < 0
    o0, o1 = _o_search[:-1][None, :], _o_search[1:][None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = o0 - w[:, :-1] * (o1 - o0) / (w[:, 1:] - w[:, :-1])
    score = np.where(change, np.abs(x), np.inf)
    j = np.argmin(score, axis=1)
    out = x[np.arange(len(x)), j]
    out[~ok | ~np.isfinite(score[np.arange(len(x)), j])] = np.nan
    return out


def refine_quads_linear(gray: np.ndarray, quads: Sequence[np.ndarray], K: np.ndarray,
                        dist: Optional[np.ndarray], side_mm: float) -> list[Optional[np.ndarray]]:
    """Corners of dark squares on a bright surround, in linear light.

    gray     the camera image, 2-D (uint8 or float), raw (distorted) pixels
    quads    per square its 4 corners (4, 2) in raw px, in order around it
             (production detection: ArUco + cornerSubPix)
    K, dist  the colour camera's lens model
    side_mm  the squares' printed side (pointer: marker_length_mm)

    Returns per quad the refined (4, 2) raw px in the same order, or None where
    an edge had fewer than _MIN_POINTS crossings (glare, cut off, ...)."""
    K = np.asarray(K, float)
    dist = np.zeros(5) if dist is None else np.asarray(dist, float).ravel()
    fx = float(K[0, 0])
    img = np.asarray(gray, np.float32)
    nq = len(quads)
    if nq == 0:
        return []
    qi = np.stack([_to_ideal(np.asarray(q, float).reshape(4, 2), K, dist) * fx for q in quads])   # (nq, 4, 2)
    cen = qi.mean(axis=1)                                                                          # (nq, 2)
    a, b = qi, np.roll(qi, -1, axis=1)                                                             # edge j: corner j -> j+1
    side = np.linalg.norm(b - a, axis=2).mean(axis=1)                                             # (nq,)
    ppm = side / float(side_mm)                                                                    # ideal px per mm
    t = (b - a) / np.linalg.norm(b - a, axis=2, keepdims=True)
    n = np.stack([t[..., 1], -t[..., 0]], axis=-1)                                                 # (nq, 4, 2)
    outward = np.einsum("qjk,qjk->qj", n, (a + b) / 2.0 - cen[:, None, :]) < 0
    n[outward] *= -1.0                                                                             # normal points out
    base = a[:, :, None, :] + _ALONG[None, None, :, None] * (b - a)[:, :, None, :]                 # (nq, 4, 15, 2)
    # The lens model only at three points of each profile (its ends and middle), quadratic
    # in between: a profile spans ~10 px, where the distortion is smooth (< 1e-3 px off the
    # exact mapping); projecting all ~63 000 points cost 24 ms.
    nodes = (_OFFSETS_MM[0], 0.5 * (_OFFSETS_MM[0] + _OFFSETS_MM[-1]), _OFFSETS_MM[-1])
    node_pts = np.stack([base + (o * ppm)[:, None, None, None] * n[:, :, None, :] for o in nodes], axis=3)
    node_raw = _to_raw(node_pts.reshape(-1, 2) / fx, K, dist).reshape(node_pts.shape)            # (nq,4,15,3,2)
    raw = np.einsum("ok,qjpkc->qjpoc", _LAGRANGE, node_raw).reshape(-1, 2)
    grey = map_coordinates(img, [raw[:, 1], raw[:, 0]], order=1, mode="nearest")
    lin = linear_light(grey).reshape(nq * 4 * len(_ALONG), len(_OFFSETS_MM))
    x_mm = _crossings(lin).reshape(nq, 4, len(_ALONG))
    edge_pts = base + (x_mm * ppm[:, None, None])[..., None] * n[:, :, None, :]                    # (nq, 4, 15, 2)

    out: list[Optional[np.ndarray]] = []
    for qn in range(nq):
        lines = []
        for j in range(4):
            good = np.isfinite(x_mm[qn, j])
            if good.sum() < _MIN_POINTS:
                lines = None
                break
            p = edge_pts[qn, j][good]
            m = p.mean(axis=0)
            _, _, vt = np.linalg.svd(p - m)
            lines.append((m, vt[0]))
        if lines is None:
            out.append(None)
            continue
        cor = []
        for j in range(4):
            (m1, d1), (m2, d2) = lines[(j - 1) % 4], lines[j]
            s = np.linalg.solve(np.c_[d1, -d2], m2 - m1)
            cor.append(m1 + s[0] * d1)
        out.append(_to_raw(np.asarray(cor) / fx, K, dist))
    return out
