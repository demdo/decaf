# -*- coding: utf-8 -*-
"""
test_xray_view.py

Checks the X-ray view of the drill (overlay.tools.xray_view, the magenta path
of overlay.tools.tracking_overlay, Thesis 6.2.1).

Without data:
(a) geometry: frames, the point on the plane, its pixel = H_xc(d_x) uv_x
(b) XrayField: inside test, line clipping, hull of a field a drill enters
(c) TrackingOverlay on synthetic poses: green untouched by the X-ray view,
    magenta only inside the warped mask (also a glyph at the edge), no glyph
    with the tip outside, forward line ahead of the field, vanishing point,
    marked pose, clear, no shadow at or behind the source

On the recordings of Fig. 4.21 (Thesis/Plots/Evaluation/data/IntegratedSystem);
each of these skips itself when the data is missing:
(d) green pixel-identical to 03_rgb_overlay_drill.png of all snapshots
(e) table of section 2 of docs/user_study/KONZEPT_UserStudy_DHS.md: tip and axis
    of the X-ray view against the drill shadow, inclination; B_2 on the camera
    image: 10.5 deg between the lines, 1.9 px between the tips
(f) magenta inside the warped mask, at d_x and d_x +- 10 mm
(g) slider d_x +- 10 mm: through the real warp a pixel of the X-ray lands on its
    magenta position; magenta stays on the shadow, green does not

The real page, camera and tracker replaced by stubs (B_2: its image, its poses):
(h) switches, slider, a new reference plane, tracking test without magenta,
    snapshot layer 04, frame log, profile log; prints what the X-ray view costs there

Run from src/:

    python -m overlay.tests.test_xray_view [--seconds N]

--seconds: length of the page run in (h), default 6 s. The numbers in (h) come from
the real page code, but not from the camera, the engine or the lab machine's load.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from overlay.tools.homography import estimate_plane_induced_homography
from overlay.tools.tracking_overlay import TrackingOverlay
from overlay.tools.warp import _detect_xray_fov_mask, prepare_live_overlay
from overlay.tools.xray_distortion import load_xray_distortion
from overlay.tools.xray_view import XrayField, XrayView

MASTERARBEIT = Path(__file__).resolve().parents[5]
DATA = MASTERARBEIT / "Thesis" / "Plots" / "Evaluation" / "data" / "IntegratedSystem"
K_XRAY_NPZ = Path(__file__).resolve().parents[1] / "data" / "K_xray.npz"

CASES = (
    ("A", "Evaluation_Integrated_System", "A_2_live"),
    ("B", "Evaluation_Integrated_System", "B_2_live"),
    ("B2", "Evaluation_Integrated_System", "B_2_live_2"),
    ("B3", "Evaluation_Integrated_System", "B_2_live_3"),
    ("Bone", "Evaluation_Integrated_System_Bone_New", "2_live"),
)
# Section 2 of the concept, to the two decimals given there.
WANT_TIP_MM = {"A": (0.61, 0.61), "B": (0.26, 0.47), "Bone": (1.09, 1.09)}
WANT_AXIS_DEG = {"A": (0.23, 0.23), "B": (0.04, 0.11), "Bone": (0.08, 0.08)}
WANT_INCLINATION_DEG = {"A": 1.7, "B": 27.7, "Bone": 0.4}

MAGENTA_ANY = (TrackingOverlay.XRAY_COLOR, TrackingOverlay.XRAY_MARKED_COLOR)


class Skip(Exception):
    pass


def _skip(reason: str) -> None:
    try:
        import pytest
    except ImportError:
        raise Skip(reason)
    pytest.skip(reason)


# ============================================================
# Synthetic set-up
# ============================================================

K_C = np.array([[1380.0, 0.0, 960.0], [0.0, 1380.0, 540.0], [0.0, 0.0, 1.0]])
DIST = np.array([0.12, -0.25, 0.001, -0.001, 0.1])
K_X = np.array([[4586.0, 0.0, 512.0], [0.0, 4586.0, 512.0], [0.0, 0.0, 1.0]])
D_X = 550.0
P_TIP, AXIS = np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


def _T_xc() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.array([0.42, 0.03, 3.1]))
    T[:3, 3] = [0.012, 0.063, 0.956]      # metres
    return T


def _field_image() -> np.ndarray:
    """A bright round field with a dark drill entering from its edge."""
    img = np.zeros((1024, 1024), np.uint8)
    cv2.circle(img, (512, 512), 440, 150, -1)
    cv2.line(img, (512, 40), (512, 420), 5, 22)
    return img


def _overlay(view=None, field=None, d_x=D_X) -> TrackingOverlay:
    ov = TrackingOverlay(K_C, DIST, P_TIP, AXIS)
    if view is not None:
        ov.set_xray_view(view, field, d_x)
    return ov


def _setup():
    view = XrayView(_T_xc(), K_X)
    img = _field_image()
    ov = _overlay()
    field = XrayField.from_image(img, ov.field_inset_px(view, (1024, 1024), D_X))
    return view, img, field


def _points(view, tip_uv, back_uv, tip_z=D_X, back_z=D_X) -> np.ndarray:
    """Camera-frame (tip, back) of two working pixels at given X-ray depths."""
    uv = np.array([tip_uv, back_uv], float)
    rays = np.c_[uv, np.ones(2)] @ np.linalg.inv(K_X).T
    P_x = rays * np.array([[tip_z], [back_z]])
    return P_x @ view.R.T + view.t_mm


def _is(img, colours) -> np.ndarray:
    """Pixels holding exactly one of the colours (anti-aliased edges excluded)."""
    out = np.zeros(img.shape[:2], bool)
    for c in colours:
        out |= (img == np.array(c, np.uint8)).all(axis=2)
    return out


def _drawn(img) -> np.ndarray:
    return img.any(axis=2)


def _warped_mask(work, H, size_wh, K, dist) -> np.ndarray:
    live = prepare_live_overlay(work, H, size_wh, K_camera=K, dist_coeffs=dist)
    mask = np.zeros((size_wh[1], size_wh[0]), bool)
    y0, y1, x0, x1 = live.box
    mask[y0:y1, x0:x1] = live.mask_roi > 0
    return mask


# ============================================================
# (a) geometry
# ============================================================

def test_geometry() -> None:
    view = XrayView(_T_xc(), K_X)
    P_c = np.array([[28.3, -64.3, 423.7], [51.4, -33.0, 392.4]])
    P_x = view.to_xray(P_c)
    assert np.allclose(P_x @ view.R.T + view.t_mm, P_c)

    S_c = view.on_plane_cam(P_c, D_X)
    S_x = view.to_xray(S_c)
    # on the plane, on the ray through the source, same shadow
    assert np.allclose(S_x[:, 2], D_X)
    assert np.allclose(np.cross(P_x, S_x), 0.0, atol=1e-6)
    assert np.allclose(view.uv_x(S_c), view.uv_x(P_c), atol=1e-9)
    assert np.allclose(view.pixels_on_plane_cam(view.uv_x(P_c), D_X), S_c)
    assert np.allclose(view.off_plane_mm(P_c, D_X), P_x[:, 2] - D_X)

    # the same pixel the overlay's homography gives (T_xc in metres, d_x in mm)
    T = _T_xc()
    for d_x in (D_X - 10.0, D_X, D_X + 10.0):
        H = estimate_plane_induced_homography(K_C, T[:3, :3], T[:3, 3], K_X, d_x)
        uv = view.uv_x(P_c)
        q = np.c_[uv, np.ones(2)] @ H.T
        pin = view.on_plane_cam(P_c, d_x) @ K_C.T
        assert np.allclose(q[:, :2] / q[:, 2:], pin[:, :2] / pin[:, 2:], atol=1e-6)

    # inclination: known axis in the X-ray frame
    tip_x, back_x = np.array([0.0, 0.0, 500.0]), np.array([np.cos(0.5), 0.0, 500.0 + np.sin(0.5)])
    tip_c, back_c = np.stack([tip_x, back_x]) @ view.R.T + view.t_mm
    assert abs(view.inclination_deg(tip_c, back_c) - np.degrees(0.5)) < 1e-9

    # millimetres in T_xc are refused
    T_mm = _T_xc()
    T_mm[:3, 3] *= 1000.0
    try:
        XrayView(T_mm, K_X)
    except ValueError:
        pass
    else:
        raise AssertionError("T_xc in mm was accepted")
    print("  (a) geometry: on the plane, on the ray, same shadow, H_xc(d_x), metres enforced")


# ============================================================
# (b) field
# ============================================================

def test_field() -> None:
    f = XrayField([[0, 0], [10, 0], [10, 10], [0, 10]])
    g = XrayField([[0, 0], [0, 10], [10, 10], [10, 0]])     # other way round
    for field in (f, g):
        assert field.contains((5, 5)) and field.contains((0, 0)) and not field.contains((11, 5))
        assert field.clip_line((-5, 5), (15, 5)) == (0.25, 0.75)
        assert field.clip_line((5, 5), (6, 5)) == (-5.0, 5.0)
        assert field.clip_line((-5, 20), (15, 20)) is None      # parallel, outside
        assert field.clip_line((-5, 12), (0, 20)) is None       # misses
        assert field.clip_line((5, 5), (5, 5)) is None          # no line

    img = _field_image()
    hull = XrayField.from_image(img).hull
    # the drill entering from the edge does not cut the field
    assert cv2.pointPolygonTest(hull.astype(np.float32).reshape(-1, 1, 2), (512.0, 200.0), True) > 100
    radius = np.linalg.norm(hull - [512, 512], axis=1)
    assert 430 < radius.min() and radius.max() < 445, (radius.min(), radius.max())
    inset = np.linalg.norm(XrayField.from_image(img, 8.0).hull - [512, 512], axis=1)
    assert np.all(inset < radius.max() - 6.0) and np.all(inset > radius.min() - 12.0)
    print(f"  (b) field: clipping, drill at the edge, inset, {len(hull)} hull points")


# ============================================================
# (c) overlay, synthetic
# ============================================================

def test_overlay_synthetic() -> None:
    view, img, field = _setup()
    size = (1920, 1080)
    T = _T_xc()
    H = estimate_plane_induced_homography(K_C, T[:3, :3], T[:3, 3], K_X, D_X)
    mask = _warped_mask(img, H, size, K_C, DIST)
    blank = lambda: np.zeros((1080, 1920, 4), np.uint8)

    pts = _points(view, (500, 500), (560, 660), D_X - 2.0, D_X + 20.0)

    # no X-ray view: as before, and the green path is unchanged by one
    plain = _overlay()
    plain.update(pts)
    a = blank()
    plain.draw_into(a)
    assert not plain.has_xray_view and plain.snapshot_record()["xray_view"] is None
    ov = _overlay(view, field)
    ov.update(pts)
    assert ov.has_xray_view
    b = blank()
    ov.draw_into(b, show_xray=False)
    assert np.array_equal(a, b)

    # each switch: only its colour
    c = blank()
    ov.draw_into(c, show_camera=False)
    assert _is(c, [TrackingOverlay.XRAY_COLOR]).sum() > 100
    assert not _is(c, [TrackingOverlay.AXIS_COLOR]).any()
    assert not _is(b, [TrackingOverlay.XRAY_COLOR]).any()
    # green first, magenta over it: what the snapshot's layer 04 shows
    d = blank()
    ov.draw_into(d)
    e = b.copy()
    ov.draw_into(e, show_camera=False)
    assert np.array_equal(d, e)
    # both off, or no glyphs
    f = blank()
    ov.draw_into(f, show_camera=False, show_xray=False)
    assert not f.any()
    f = blank()
    ov.draw_into(f, show_camera=False, show_tip=False, show_axis=False)
    assert not f.any()

    # the record is what was drawn
    rec = ov.snapshot_record()["xray_view"]
    assert rec["tip_in_field"] is True
    assert np.allclose(rec["tip_uv_x"], [500, 500]) and np.allclose(rec["back_uv_x"], [560, 660])
    assert abs(rec["off_plane_mm"] + 2.0) < 1e-9
    assert len(rec["segments_px"]) == 1 + (TrackingOverlay.DASH_PIECES + 1) // 2
    json.dumps(ov.snapshot_record())

    # a new d_x moves magenta at once, green not
    green_before = ov.snapshot_record()["tip_uv"]
    ov.set_plane_depth(D_X + 10.0)
    rec2 = ov.snapshot_record()
    assert rec2["tip_uv"] == green_before
    assert rec2["xray_view"]["tip_uv_x"] == rec["tip_uv_x"]     # the shadow does not depend on d_x
    assert np.linalg.norm(np.subtract(rec2["xray_view"]["tip_uv_on_plane"], rec["tip_uv_on_plane"])) > 1.0
    ov.set_plane_depth(D_X)

    # tip outside, aimed at the field: forward line inside it, no glyph
    ov.update(_points(view, (512, 40), (512, -60)))
    rec = ov.snapshot_record()["xray_view"]
    assert rec["tip_in_field"] is False and len(rec["segments_px"]) == 1
    g = blank()
    ov.draw_into(g, show_camera=False)
    assert _drawn(g).any() and not (_drawn(g) & ~mask).any()
    assert np.linalg.norm(np.subtract(rec["tip_uv_on_plane"], np.array(rec["segments_px"][0][0]))) > 5

    # tip outside, aimed away: nothing at all
    ov.update(_points(view, (512, 40), (512, 140)))
    rec = ov.snapshot_record()["xray_view"]
    h = blank()
    ov.draw_into(h, show_camera=False)
    # the shaft runs back into the field here, so only pieces inside are allowed
    assert rec["tip_in_field"] is False and not (_drawn(h) & ~mask).any()
    ov.update(_points(view, (20, 20), (20, -100)))
    h = blank()
    ov.draw_into(h, show_camera=False)
    assert not h.any() and ov.has_xray_view

    # sweep over and across the edge: every magenta pixel in the warped mask,
    # the tip glyph right at the edge included
    rng = np.random.default_rng(1)
    n_edge = 0
    for _ in range(400):
        ang = rng.uniform(0.0, 2.0 * np.pi)
        r = rng.uniform(380.0, 470.0)
        tip = np.array([512.0, 512.0]) + r * np.array([np.cos(ang), np.sin(ang)])
        a2 = rng.uniform(0.0, 2.0 * np.pi)
        back = tip + 160.0 * np.array([np.cos(a2), np.sin(a2)])
        ov.update(_points(view, tip, back, D_X, D_X + rng.uniform(-30, 30)))
        n_edge += bool(ov._xv["tip_near_edge"] and ov._xv["tip_px"] is not None)
        k = blank()
        ov.draw_into(k, show_camera=False)
        assert not (_drawn(k) & ~mask).any(), (tip, back)
    assert n_edge > 20, n_edge

    # the axis almost along the source ray, running away from it: the forward
    # line ends at the vanishing point instead of running through the field
    tip_x = np.array([0.0, 0.0, D_X])
    fwd = np.array([0.03, 0.0, 1.0])
    P_x = np.stack([tip_x, tip_x - 50.0 * fwd / np.linalg.norm(fwd)])
    ov.update(P_x @ view.R.T + view.t_mm)
    vanish = view.uv_from_xray(fwd)[0]
    far = view.pixels_on_plane_cam([vanish], D_X)
    far_px = ov._project(far)[0]
    assert np.linalg.norm(np.subtract(ov._xv["segments"][0][0], far_px)) < 1.0

    # marked: green orange, magenta dimmed, not gone
    ov.update(pts, marked=True)
    m = blank()
    ov.draw_into(m)
    assert _is(m, [TrackingOverlay.MARKED_COLOR]).any()
    assert _is(m, [TrackingOverlay.XRAY_MARKED_COLOR]).any()
    assert not _is(m, [TrackingOverlay.XRAY_COLOR]).any()
    assert ov.snapshot_record()["xray_view"]["colour_bgra"] == list(TrackingOverlay.XRAY_MARKED_COLOR)

    # loss, no shadow, view switched off
    ov.clear()
    assert not ov.has_xray_view and ov.snapshot_record() is None
    near_source = np.array([view.t_mm, view.t_mm + view.R[:, 0] * 10.0])
    ov.update(near_source)
    assert ov.has_pose and not ov.has_xray_view
    ov.update(pts)
    ov.set_xray_view(None)
    assert ov.has_pose and not ov.has_xray_view
    print(f"  (c) overlay: green untouched, switches, record, d_x, field edge ({n_edge} glyphs "
          "at the edge), vanishing point, marked, loss")


# ============================================================
# Recordings of Fig. 4.21
# ============================================================

def _load(sess: str, snap: str) -> SimpleNamespace:
    d = DATA / sess / snap
    s = json.loads((d / "snapshot.json").read_text())
    raw = cv2.imread(str(d / "00_xray.png"), cv2.IMREAD_GRAYSCALE)
    return SimpleNamespace(
        dir=d, s=s, raw=raw,
        K_c=np.array(s["K_rgb"]), dist=np.array(s["dist_rgb"]), K_x=np.array(s["K_xray"]),
        T=np.array(s["T_xc"]), d_x=float(s["d_x_mm"]),
        pts=np.array([s["tool"]["tip_cam_mm"], s["tool"]["back_cam_mm"]]),
        p_tip=np.array(s["tool"]["p_tip_mm"]), axis=np.array(s["tool"]["axis_dir"]),
    )


def _field_distortion():
    if not K_XRAY_NPZ.exists():
        _skip(f"no {K_XRAY_NPZ}")
    return load_xray_distortion(str(K_XRAY_NPZ))


def _need_data() -> None:
    for _, sess, snap in CASES:
        if not (DATA / sess / snap / "snapshot.json").exists():
            _skip(f"no recording {sess}/{snap} under {DATA}")


def _working(rec, distortion) -> np.ndarray:
    """As OverlayLivePage._load_xray_from_state."""
    return distortion.undistort_image_working(np.ascontiguousarray(cv2.flip(rec.raw, 1)))


def _xray_tip(x_u8: np.ndarray) -> np.ndarray:
    """Drill tip in the raw X-ray, as Thesis/Plots/Evaluation/IntegratedSystem/
    is1_drill_projection.py: the largest dark component inside the field, its end
    away from the field edge, sub-pixel by the 50 % crossing along the local axis."""
    img = x_u8.astype(np.float32)
    fov = _detect_xray_fov_mask(x_u8) > 0
    inner = cv2.erode(fov.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    bit_level = np.percentile(img[inner], 1.0)
    bg_level = np.median(img[inner])
    thr = 0.5 * (bit_level + bg_level)
    _, lab, st, _ = cv2.connectedComponentsWithStats(((img < thr) & inner).astype(np.uint8), 8)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    ys, xs = np.nonzero(lab == k)
    pts = np.c_[xs, ys].astype(float)
    c = pts.mean(0)
    w, v = np.linalg.eigh(np.cov((pts - c).T))
    ax = v[:, np.argmax(w)]
    proj = (pts - c) @ ax
    dist_edge = cv2.distanceTransform(fov.astype(np.uint8), cv2.DIST_L2, 5)
    ends = [pts[np.argmin(proj)], pts[np.argmax(proj)]]
    de = [dist_edge[int(e[1]), int(e[0])] for e in ends]
    ax = ax * (-1.0 if de[0] > de[1] else 1.0)
    proj = (pts - c) @ ax
    near = pts[proj > proj.max() - 80]
    c2 = near.mean(0)
    w2, v2 = np.linalg.eigh(np.cov((near - c2).T))
    ax2 = v2[:, np.argmax(w2)]
    if ax2 @ ax < 0:
        ax2 = -ax2
    s = np.arange(-30.0, 40.0, 0.1) + (proj.max() - (c2 - c) @ ax)
    line = c2[None, :] + s[:, None] * ax2[None, :]
    smooth = cv2.GaussianBlur(img, (3, 3), 0)
    pv = cv2.remap(smooth, line[:, 0].astype(np.float32).reshape(-1, 1),
                   line[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
    inside, outside = np.median(pv[s < s[0] + 10]), np.median(pv[s > s[-1] - 10])
    half = 0.5 * (inside + outside)
    i = np.nonzero(pv > half)[0][0]
    t = s[i - 1] + (half - pv[i - 1]) / (pv[i] - pv[i - 1]) * (s[i] - s[i - 1])
    return c2 + t * ax2


def _shadow_axis_deg(rec, view, work) -> float:
    """Projected axis against the centre line of the drill shadow, from cross-sections
    of the shadow every 2 mm along the axis (docs/user_study/scripts/us4_axis.py)."""
    tip_c, back_c = rec.pts
    fwd = (tip_c - back_c) / np.linalg.norm(tip_c - back_c)
    field = XrayField.from_image(work, 0.0, tolerance_px=0.01)
    ss = np.arange(-400.0, 400.25, 0.25)
    uv_all = view.uv_x(tip_c[None, :] + ss[:, None] * fwd[None, :])
    inside = np.array([field.contains(p) for p in uv_all])
    lo = ss[inside][0]

    img = cv2.GaussianBlur(work, (5, 5), 0).astype(np.float32)
    centres = []
    for s in np.arange(-8.0, max(lo + 8.0, -60.0), -2.0):
        uv = view.uv_x(tip_c + s * fwd)[0]
        uv2 = view.uv_x(tip_c + (s + 1.0) * fwd)[0]
        d = (uv2 - uv) / np.linalg.norm(uv2 - uv)
        n = np.array([-d[1], d[0]])
        t = np.arange(-70.0, 70.0, 0.5)
        line = uv[None, :] + t[:, None] * n[None, :]
        p = cv2.remap(img, line[:, 0].astype(np.float32).reshape(-1, 1),
                      line[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
        bg, dk = np.percentile(p, 80), p.min()
        if bg - dk < 25:
            continue
        thr = 0.5 * (bg + dk)
        k = int(np.argmin(p))
        a = b = k
        while a > 0 and p[a - 1] < thr:
            a -= 1
        while b < len(p) - 1 and p[b + 1] < thr:
            b += 1
        centres.append(uv + 0.5 * (t[a] + t[b]) * n)
    centres = np.array(centres)
    c = centres.mean(0)
    w, V = np.linalg.eigh(np.cov((centres - c).T))
    shadow = V[:, np.argmax(w)]
    uv_t, uv_b = view.uv_x(rec.pts)
    a = (uv_b - uv_t) / np.linalg.norm(uv_b - uv_t)
    return float(np.degrees(np.arccos(np.clip(abs(a @ shadow), 0.0, 1.0))))


def _overlay_of(rec, work, d_x=None):
    """The X-ray view as OverlayLivePage._attach_xray_view builds it."""
    view = XrayView(rec.T, rec.K_x)
    ov = TrackingOverlay(rec.K_c, rec.dist, rec.p_tip, rec.axis)
    h, w = work.shape[:2]
    field = XrayField.from_image(work, ov.field_inset_px(view, (w, h), rec.d_x))
    ov.set_xray_view(view, field, rec.d_x if d_x is None else d_x)
    ov.update(rec.pts, marked=bool(rec.s["tool"].get("marked", False)))
    return ov, view


# ------------------------------------------------------------ (d)

def test_green_pixel_identical() -> None:
    _need_data()
    folders = sorted(p.parent for p in DATA.glob("*/*/03_rgb_overlay_drill.png"))
    n = 0
    for d in folders:
        s = json.loads((d / "snapshot.json").read_text())
        tool = s.get("tool")
        if not tool or not (d / "02_rgb_overlay.png").exists():
            continue
        ov = TrackingOverlay(s["K_rgb"], s["dist_rgb"], tool["p_tip_mm"], tool["axis_dir"])
        ov.update(np.array([tool["tip_cam_mm"], tool["back_cam_mm"]]),
                  marked=bool(tool.get("marked", False)))
        img = cv2.cvtColor(cv2.imread(str(d / "02_rgb_overlay.png")), cv2.COLOR_BGR2BGRA)
        ov.draw_into(img, show_camera=True, show_xray=False)
        want = cv2.imread(str(d / "03_rgb_overlay_drill.png"))
        got = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        assert np.array_equal(got, want), f"{d}: {int((got != want).any(axis=2).sum())} pixels differ"
        n += 1
    assert n >= len(CASES), n
    print(f"  (d) green pixel-identical to 03_rgb_overlay_drill.png on {n} snapshots")


# ------------------------------------------------------------ (e)

def test_table_section_2() -> None:
    _need_data()
    distortion = _field_distortion()
    tips, axes, incl = {}, {}, {}
    for key, sess, snap in CASES:
        rec = _load(sess, snap)
        work = _working(rec, distortion)
        ov, view = _overlay_of(rec, work)
        xv = ov.snapshot_record()["xray_view"]
        assert np.allclose(xv["tip_uv_x"], view.uv_x(rec.pts)[0])

        raw_tip = _xray_tip(rec.raw)
        shadow = distortion.undistort_working(
            np.array([[rec.raw.shape[1] - 1 - raw_tip[0], raw_tip[1]]])
        )[0]
        tip_mm = float(np.linalg.norm(np.subtract(xv["tip_uv_x"], shadow))) / (rec.K_x[0, 0] / rec.d_x)
        axis_deg = _shadow_axis_deg(rec, view, work)
        group = key.rstrip("23")
        tips.setdefault(group, []).append(tip_mm)
        axes.setdefault(group, []).append(axis_deg)
        incl.setdefault(group, []).append(xv["inclination_deg"])

        print(f"      {key:5s} tip to shadow {tip_mm:.3f} mm, axis to shadow {axis_deg:.3f} deg, "
              f"inclination {xv['inclination_deg']:.2f} deg")

        if key == "B":
            uv_c = np.array([xv["tip_uv_on_plane"], xv["back_uv_on_plane"]])
            uv_g = np.array([rec.s["tool"]["tip_uv"], rec.s["tool"]["back_uv"]])
            gm, mg = uv_g[1] - uv_g[0], uv_c[1] - uv_c[0]
            angle = abs(np.degrees(np.arctan2(gm[1], gm[0]) - np.arctan2(mg[1], mg[0])))
            assert abs(angle - 10.5) < 0.05, angle
            assert abs(np.linalg.norm(uv_c[0] - uv_g[0]) - 1.9) < 0.05

    for group, (lo, hi) in WANT_TIP_MM.items():
        got = (min(tips[group]), max(tips[group]))
        assert abs(got[0] - lo) < 0.006 and abs(got[1] - hi) < 0.006, (group, got, (lo, hi))
    for group, (lo, hi) in WANT_AXIS_DEG.items():
        got = (min(axes[group]), max(axes[group]))
        assert abs(got[0] - lo) < 0.006 and abs(got[1] - hi) < 0.006, (group, got, (lo, hi))
    for group, want in WANT_INCLINATION_DEG.items():
        assert all(abs(v - want) < 0.5 for v in incl[group]), (group, incl[group], want)
    print("  (e) table of section 2 reproduced (tip, axis, inclination; B_2 10.5 deg, 1.9 px)")


# ------------------------------------------------------------ (f), (g)

def test_mask_and_slider() -> None:
    _need_data()
    distortion = _field_distortion()
    for key, sess, snap in CASES:
        rec = _load(sess, snap)
        work = _working(rec, distortion)
        s = rec.s
        w, h = s["frame_wh"]
        ov, view = _overlay_of(rec, work)
        green = np.array(ov.snapshot_record()["tip_uv"])

        raw_tip = _xray_tip(rec.raw)
        shadow = distortion.undistort_working(
            np.array([[rec.raw.shape[1] - 1 - raw_tip[0], raw_tip[1]]])
        )[0]
        # a point of the X-ray, painted into a copy of the working image
        uv = np.array(ov.snapshot_record()["xray_view"]["tip_uv_x"])
        painted = work.copy()
        cu, cv_ = int(round(uv[0])), int(round(uv[1]))
        painted[cv_ - 8:cv_ + 9, cu - 8:cu + 9] = 255

        to_shadow, to_green = [], []
        for off in (-10.0, 0.0, 10.0):
            d_x = rec.d_x + off
            ov.set_plane_depth(d_x)
            xv = ov.snapshot_record()["xray_view"]
            assert np.allclose(xv["tip_uv_x"], uv)    # X-ray pixel without d_x

            H = estimate_plane_induced_homography(
                rec.K_c, rec.T[:3, :3], rec.T[:3, 3], rec.K_x, d_x)
            live = prepare_live_overlay(work, H, (w, h), K_camera=rec.K_c, dist_coeffs=rec.dist)
            mask = np.zeros((h, w), bool)
            y0, y1, x0, x1 = live.box
            mask[y0:y1, x0:x1] = live.mask_roi > 0
            img = np.zeros((h, w, 4), np.uint8)
            ov.draw_into(img, show_camera=False)
            assert _drawn(img).any(), key
            outside = int((_drawn(img) & ~mask).sum())
            assert outside == 0, (key, off, outside)

            # through the real warp: where the painted patch lands is where
            # the magenta glyph is drawn
            warped = prepare_live_overlay(painted, H, (w, h), K_camera=rec.K_c, dist_coeffs=rec.dist)
            roi = warped.xray_roi[:, :, 0]
            ys, xs = np.nonzero(roi == 255)
            # the patch is the only pure white: it is far from everything else
            centre = np.array([xs.mean() + warped.box[2], ys.mean() + warped.box[0]])
            px_err = float(np.linalg.norm(centre - np.array(xv["tip_uv_on_plane"])))
            assert px_err < 1.2, (key, off, px_err)

            # the shadow as the overlay shows it, through H and the lens
            q = np.array([shadow[0], shadow[1], 1.0]) @ H.T
            ideal = q[:2] / q[2]
            ray = np.array([[(ideal[0] - rec.K_c[0, 2]) / rec.K_c[0, 0],
                             (ideal[1] - rec.K_c[1, 2]) / rec.K_c[1, 1], 1.0]])
            seen, _ = cv2.projectPoints(ray.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), rec.K_c, rec.dist)
            seen = seen.reshape(2)
            to_shadow.append(float(np.linalg.norm(seen - xv["tip_uv_on_plane"])))
            to_green.append(float(np.linalg.norm(seen - green)))

        ov.set_plane_depth(rec.d_x)
        spread_magenta = max(to_shadow) - min(to_shadow)
        spread_green = max(to_green) - min(to_green)
        # magenta rides with the shadow, green does not. The fixed offset of the
        # X-ray view (section 2) only changes with the magnification of the
        # plane on the camera image, about +-4 % over +-10 mm.
        assert spread_magenta < 0.10 * float(np.mean(to_shadow)) + 0.05, (key, to_shadow)
        assert spread_green > 2.0, (key, to_green)
        print(f"      {key:5s} d_x +-10 mm: shadow to magenta changes by {spread_magenta:.3f} px, "
              f"shadow to green by {spread_green:.1f} px")
    print("  (f) magenta inside the warped mask at d_x and d_x +-10 mm")
    print("  (g) slider: an X-ray pixel lands on magenta through the real warp; green leaves the shadow")


# ============================================================
# (h) the real page with stubs for camera and engine
# ============================================================

ACCEPTED_KEY = "cpp_tracker_engine_current_pose_accepted_count"


class _Camera:
    """30 fps of one frame."""

    calibration_info = {}

    def __init__(self, bgra: np.ndarray) -> None:
        self.bgra = bgra
        self.t0 = time.perf_counter()
        self.index = -1

    def poll(self):
        n = int((time.perf_counter() - self.t0) * 30.0)
        if n <= self.index:
            return None
        self.index = n
        return SimpleNamespace(
            image_bgr=self.bgra, ir_left=None, ir_right=None, frame_index=n,
            timestamp_s=time.time(), metadata={"color_time_ms": n * 33.3},
        )

    def stop(self) -> None:
        pass


class _Tracker:
    """Replays the recorded poses of a tracking run. The blur stands in for the
    engine's load: like the engine it runs in C++ on all cores, without the GIL."""

    ir_frame_timing = None

    def __init__(self, rvec: np.ndarray, tvec: np.ndarray) -> None:
        self.config = SimpleNamespace(drawn_pose_mark_below_visible_frac=0.75)
        self.rvec, self.tvec = rvec, tvec
        self.i = 0

    def reset(self) -> None:
        pass

    def process_frame(self, color, *, run_detection=True, ir_left=None, ir_right=None,
                      t_frame=None, **timing):
        cv2.GaussianBlur(color, (31, 31), 0)
        i = self.i % len(self.rvec)
        self.i += 1
        source = SimpleNamespace(value="decode", name="decode")
        return SimpleNamespace(
            success=True, pose_source=source, pose_converged=True, message="",
            mode=SimpleNamespace(value="tracking"), rvec=self.rvec[i], tvec=self.tvec[i],
            rvec_prefusion=None, tvec_prefusion=None, num_points=100, num_inliers=100,
            mean_reprojection_error_px=0.3, visible_cell_fraction=1.0,
            timings_ms={ACCEPTED_KEY: 1.0, "visible_cell_fraction": 1.0},
            debug_counters={}, corners=(), detection_corners=(),
        )


def _make_page(rec, work_raw, distortion, poses, *, tracking_test=False, ev=None):
    from overlay.gui.pages.page_overlay_live import OverlayLivePage
    from overlay.gui.state import SessionState
    from tracking.hydramarker import tracker_log

    st = SessionState()
    st.K_rgb, st.dist_rgb = rec.K_c, rec.dist
    st.K_xray, st.T_xc, st.d_x = rec.K_x, rec.T, rec.d_x
    st.xray_dist_field = distortion
    st.tip_uv_c = np.array(rec.s["reference_plane_point_uv"])
    st.tip_xyz_c = np.array(rec.s["reference_plane_point_xyz_c_mm"])
    st.xray_image_anatomy = work_raw
    st.tool_assets = SimpleNamespace(
        p_tip_mm=rec.p_tip, axis_dir=rec.axis, name="stub", field_path="f.field",
        marker_json_path="m.json", pivot_npz_path="p.npz",
    )
    st.eval = ev
    page = OverlayLivePage(st)
    page.tracking_test = tracking_test
    page._exposure_mode = "auto"
    bgra = cv2.cvtColor(cv2.imread(str(rec.dir / "01_rgb.png")), cv2.COLOR_BGR2BGRA)

    def start_camera():
        page.camera = _Camera(bgra)

    def make_tracker():
        page._has_fresh_pose = tracker_log.has_fresh_pose
        page._tracker_log_module = tracker_log
        page._track_fail_log = None
        return _Tracker(*poses)

    page._start_realsense = start_camera
    page._make_tracker = make_tracker
    page._check_engine_gil = lambda: None
    page.resize(1920, 1080)
    page.show()
    return page


def _spin(app, seconds: float, until=None) -> bool:
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.001)
    return until is None


def test_page(seconds: float = 6.0) -> None:
    _need_data()
    distortion = _field_distortion()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        _skip("no PySide6")
    from overlay.gui.evaluation import EvaluationSession
    from overlay.gui.pages import page_overlay_live as live

    app = QApplication.instance() or QApplication(sys.argv)

    # no modal boxes in a test; an error box is a failure
    errors = []
    info = []
    box_critical, box_information = QMessageBox.critical, QMessageBox.information
    QMessageBox.critical = staticmethod(lambda *a, **k: errors.append(a[2:]))
    QMessageBox.information = staticmethod(lambda *a, **k: info.append(a[2:]))

    rec = _load("Evaluation_Integrated_System", "B_2_live")
    track = DATA / "Evaluation_Integrated_System" / "B_2_tracking.npz"
    if not track.exists():
        _skip(f"no {track}")
    z = np.load(track)
    ok = np.isfinite(z["rvec"]).all(axis=1)
    poses = (z["rvec"][ok], z["tvec"][ok])

    tmp = Path(tempfile.mkdtemp(prefix="xray_view_test_"))
    try:
        ev = EvaluationSession(tmp, "session")
        ev.set_label("test")
        page = _make_page(rec, rec.raw, distortion, poses, ev=ev)
        page.on_enter()
        assert page.camera is not None and not errors, errors
        ctl = page.controls

        ctl.btn_track.click()
        assert page._tracking_active
        assert page._tool_overlay._xray is not None
        assert _spin(app, 5.0, lambda: page._drawn_out is not None), "no pose on screen"
        assert _spin(app, seconds)

        def colours():
            _spin(app, 0.4)
            page._on_timer()
            buf = page._display_bgra
            return (_is(buf, [TrackingOverlay.XRAY_COLOR]).sum(),
                    _is(buf, [TrackingOverlay.AXIS_COLOR]).sum())

        magenta, green = colours()
        assert magenta > 50 and green > 50, (magenta, green)

        ctl.chk_show_xray_view.setChecked(False)
        magenta, green = colours()
        assert magenta == 0 and green > 50, (magenta, green)
        ctl.chk_show_camera_view.setChecked(False)
        magenta, green = colours()
        assert magenta == 0 and green == 0, (magenta, green)
        ctl.chk_show_xray_view.setChecked(True)
        magenta, green = colours()
        assert magenta > 50 and green == 0, (magenta, green)
        ctl.chk_show_camera_view.setChecked(True)

        # slider: the overlay follows at once, the page's d_x with it
        base = page._current_d_x_mm
        ctl.slider_dx.setValue(10)
        assert page._tool_overlay._d_x_mm == base + 10.0
        rec_plus = page._tool_overlay.snapshot_record()["xray_view"]
        assert rec_plus is None or rec_plus["d_x_mm"] == base + 10.0
        ctl.slider_dx.setValue(0)
        assert page._tool_overlay._d_x_mm == base
        _spin(app, 0.3)     # the overlay at the new d_x is built on the next frame

        # snapshot
        ctl.btn_save.click()
        assert _spin(app, 10.0, lambda: ctl.btn_save.isEnabled() and info), "snapshot not written"
        assert not errors, errors
        snap = next(tmp.glob("session/test_live*"))
        for name in ("00_xray", "01_rgb", "02_rgb_overlay", "03_rgb_overlay_drill",
                     "04_rgb_overlay_drill_xrayview"):
            assert (snap / f"{name}.png").exists(), name
        js = json.loads((snap / "snapshot.json").read_text())
        assert js["show_camera_view"] is True and js["show_xray_view"] is True
        xv = js["tool"]["xray_view"]
        assert xv["d_x_mm"] == base and xv["tip_in_field"] in (True, False)
        layer3 = cv2.imread(str(snap / "03_rgb_overlay_drill.png"))
        layer4 = cv2.imread(str(snap / "04_rgb_overlay_drill_xrayview.png"))
        assert not _is(layer3, [TrackingOverlay.XRAY_COLOR[:3]]).any()
        assert _is(layer4, [TrackingOverlay.XRAY_COLOR[:3]]).any()

        # a new reference plane (what _apply_plane does to the page): new X-ray,
        # new d_x; the next tracking start draws with them, not with the old hull
        ctl.btn_track.click()
        old_field = page._tool_overlay._field
        page.state.d_x = base + 5.0
        page.state.xray_image_anatomy = rec.raw.copy()
        page._load_xray_from_state()
        page._initialize_dx()
        page._precompute_H_xc_lookup()
        page._invalidate_overlay_cache()
        assert page._xray_geometry is None
        ctl.btn_track.click()
        assert page._tool_overlay._d_x_mm == base + 5.0
        assert page._tool_overlay._field is not old_field
        assert _spin(app, 5.0, lambda: page._drawn_out is not None), "no pose after the new plane"

        # stop: frame log and profile log of the real page
        ctl.btn_track.click()
        page.on_leave()
        assert _spin(app, 10.0, lambda: list(tmp.glob("session/test_tracking*.npz"))
                     and not any(t.name == "frame-log-writer" for t in _threads())), "no frame log"
        tr = np.load(next(tmp.glob("session/test_tracking*.npz")))
        for key in ("d_x_mm", "tip_x_mm", "tip_uv_x", "back_uv_x", "tip_uv_on_plane",
                    "off_plane_mm", "inclination_deg", "tip_in_field", "xview_update_ms"):
            assert key in tr.files, key
        meta = json.loads(str(tr["meta_json"]))
        assert meta["log_version"] == live.FRAME_LOG_VERSION == 2
        di = np.load(next(tmp.glob("session/test_display*.npz")))
        assert "xview_ms" in di.files

        draw = di["xview_ms"][np.isfinite(di["xview_ms"])]
        up = tr["xview_update_ms"][np.isfinite(tr["xview_update_ms"])]
        drill = di["drill_ms"][np.isfinite(di["drill_ms"])]
        log_lines = (tmp / "session" / live.PROFILE_LOG_NAME).read_text(encoding="utf-8").splitlines()
        with_view = [ln for ln in log_lines if " xview " in ln and "xview_up" in ln]
        assert with_view, "no X-ray view stages in the profile log"
        print(f"      frame log, {len(draw)} shown frames / {len(up)} poses (stub camera + engine):")
        print(f"        magenta drawing {draw.mean():.4f} ms mean, {np.percentile(draw, 95):.4f} p95, {draw.max():.4f} max")
        print(f"        magenta geometry per pose {up.mean():.4f} ms mean, {np.percentile(up, 95):.4f} p95, {up.max():.4f} max")
        print(f"        together {draw.mean() + up.mean():.4f} ms per frame; green drawing for scale {drill.mean():.4f} ms")
        print(f"      profile log: {with_view[-1][with_view[-1].find('mean/max'):]}")

        # the tracking test: no X-ray, no magenta, no switch for it, no layer 04
        info.clear()
        ev2 = EvaluationSession(tmp, "session2")
        ev2.set_label("test")
        page2 = _make_page(rec, rec.raw, distortion, poses, tracking_test=True, ev=ev2)
        page2.on_enter()
        assert page2._tracking_only
        assert not page2.controls.chk_show_xray_view.isVisibleTo(page2.controls)
        page2.controls.btn_track.click()
        assert _spin(app, 5.0, lambda: page2._drawn_out is not None), "no pose on screen"
        assert page2._tool_overlay._xray is None and not page2._tool_overlay.has_xray_view
        _spin(app, 0.3)
        assert not _is(page2._display_bgra, [TrackingOverlay.XRAY_COLOR]).any()
        page2.controls.btn_save.click()
        assert _spin(app, 10.0, lambda: page2.controls.btn_save.isEnabled() and info)
        snap2 = next(tmp.glob("session2/test_live*"))
        assert (snap2 / "03_rgb_overlay_drill.png").exists()
        assert not (snap2 / "04_rgb_overlay_drill_xrayview.png").exists()
        js2 = json.loads((snap2 / "snapshot.json").read_text())
        assert js2["show_xray_view"] is False and js2["tool"]["xray_view"] is None
        page2.controls.btn_track.click()
        page2.on_leave()
        _spin(app, 1.0, lambda: not any(t.name == "frame-log-writer" for t in _threads()))
        assert not errors, errors
        page.close()
        page2.close()
        print("  (h) real page with stubs: switches, slider, snapshot layer 04, frame log, "
              "profile log, tracking test without magenta")
    finally:
        QMessageBox.critical, QMessageBox.information = box_critical, box_information
        shutil.rmtree(tmp, ignore_errors=True)


def _threads():
    import threading
    return threading.enumerate()


# ============================================================
# Main
# ============================================================

def main() -> int:
    seconds = 6.0
    if "--seconds" in sys.argv:
        seconds = float(sys.argv[sys.argv.index("--seconds") + 1])
    tests = [
        test_geometry,
        test_field,
        test_overlay_synthetic,
        test_green_pixel_identical,
        test_table_section_2,
        test_mask_and_slider,
        lambda: test_page(seconds),
    ]
    # outside pytest: a skip is reported, not raised through pytest
    global _skip

    def _print_skip(reason: str) -> None:
        raise Skip(reason)

    _skip = _print_skip
    n_skipped = 0
    for test in tests:
        try:
            test()
        except Skip as s:
            n_skipped += 1
            print(f"  SKIPPED {getattr(test, '__name__', 'test_page')}: {s}")
    print("all checks passed" + (f" ({n_skipped} skipped: data missing)" if n_skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
