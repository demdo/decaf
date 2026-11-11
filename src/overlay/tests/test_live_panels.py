"""Checks of the live panels (user study DHS, docs/user_study/KONZEPT_UserStudy_DHS.md 4).

    python -m overlay.tests.test_live_panels

Without a camera and a tracker. (a)-(c) need nothing but this repository, (d)-(h) the
recordings of Fig. 4.21; without them they skip themselves.
"""

from __future__ import annotations

import math
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

import cv2
import numpy as np

from overlay.tests import test_xray_view as xv
from overlay.tools.tool_marks import (
    ORIENT_C_ARM,
    ORIENT_CAMERA,
    RULER_FIELD,
    RULER_SET,
    TIP_RING_MM,
    CameraModel,
    MarkBuilder,
    PanelMap,
    clip_ray_to_rect,
)
from overlay.tools.tracking_overlay import TrackingOverlay
from overlay.tools.warp import LiveOverlay
from overlay.tools.xray_view import XrayField, XrayView


class Skip(Exception):
    pass


def _skip(reason: str) -> None:
    raise Skip(reason)


# ------------------------------------------------------------ (a)

def test_camera_model() -> None:
    K = np.array([[1356.6, 0.0, 975.9], [0.0, 1357.5, 531.7], [0.0, 0.0, 1.0]])
    for dist in (
        None,
        np.array([0.12, -0.25, 0.001, -0.001, 0.1]),
        np.array([0.1937, 13.31, 1.4e-4, -9.8e-4, 87.6, 0.13, 12.7, 87.98, 0, 0, 0, 0, 0, 0]),
        np.array([0.1, -0.2, 0.001, -0.001, 0.1, 0.1, 0.0, 0.0, 0.002, 0.001, -0.002, 0.001]),
    ):
        cam = CameraModel(K, dist)
        P = np.random.default_rng(1).uniform((-200, -150, 200), (200, 150, 900), (200, 3))
        want, _ = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
        got = np.array([cam.project(*q) for q in P.tolist()])
        assert np.abs(got - want.reshape(-1, 2)).max() < 1e-9
    try:
        CameraModel(K, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("tilted model accepted")
    print("  (a) CameraModel = cv2.projectPoints")


# ------------------------------------------------------------ (b)

def test_panel_map() -> None:
    work = np.zeros((1024, 1024), np.uint8)
    work[200:204, 700:704] = 255        # a bright patch: centre u = 701.5, v = 201.5
    m = PanelMap.of(ORIENT_CAMERA, (1024, 1024), 640)
    img = cv2.resize(cv2.flip(work, 0), (640, 640), interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(img > 0)
    w = img[ys, xs].astype(float)
    # image pixel centres are at +0.5 in painter coordinates
    got = ((xs + 0.5) * w).sum() / w.sum(), ((ys + 0.5) * w).sum() / w.sum()
    want = m.apply(701.5, 201.5)
    assert math.hypot(got[0] - want[0], got[1] - want[1]) < 0.1, (got, want)

    m2 = PanelMap.of(ORIENT_C_ARM, (1024, 1024), 640)
    img = cv2.resize(cv2.flip(work, 1), (640, 640), interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(img > 0)
    w = img[ys, xs].astype(float)
    got = ((xs + 0.5) * w).sum() / w.sum(), ((ys + 0.5) * w).sum() / w.sum()
    want = m2.apply(701.5, 201.5)
    assert math.hypot(got[0] - want[0], got[1] - want[1]) < 0.1, (got, want)

    # the two orientations are 180 degrees apart
    a = m.apply(100.0, 300.0)
    b = m2.apply(100.0, 300.0)
    assert abs(a[0] + b[0] - 640.0) < 1e-9 and abs(a[1] + b[1] - 640.0) < 1e-9
    print("  (b) panel map on the resampled image, both orientations")


# ------------------------------------------------------------ (c)

def _synthetic():
    T = xv._T_xc()
    view = XrayView(T, xv.K_X)
    field = XrayField.from_image(xv._field_image(), 8.0)
    b = MarkBuilder(xv.K_C, xv.DIST)
    b.set_xray(view, field, xv.D_X)
    b.set_xray_panel(PanelMap.of(ORIENT_CAMERA, (1024, 1024), 640))
    return b, view, field


def test_clip_and_ruler_synthetic() -> None:
    # the ray clip
    assert clip_ray_to_rect(10.0, 10.0, 1.0, 0.0, 100.0, 50.0) == (0.0, 90.0)
    assert clip_ray_to_rect(-10.0, 10.0, 1.0, 0.0, 100.0, 50.0) == (10.0, 110.0)
    assert clip_ray_to_rect(-10.0, 10.0, -1.0, 0.0, 100.0, 50.0) is None
    assert clip_ray_to_rect(10.0, 60.0, 1.0, 0.0, 100.0, 50.0) is None

    b, view, field = _synthetic()
    m = b.build(xv._points(view, (512.0, 512.0), (612.0, 612.0)), t_pose=1.5)
    assert m.t_pose == 1.5 and m.xray is not None and m.tip_in_field
    assert abs(m.off_plane_mm) < 1e-6 and abs(m.inclination_deg) < 1e-6
    assert m.depth_mm > 0.0     # tip in the field: it has entered

    # behind the camera and at the source: no marks, never an exception
    assert b.build(np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 10.0]])) is None
    # the axis along a source ray: a point in the image, no ruler
    tip = view.pixels_on_plane_cam([[512.0, 512.0]], xv.D_X)[0]
    back = view.pixels_on_plane_cam([[512.0, 512.0]], xv.D_X + 50.0)[0]
    m = b.build(np.stack([tip, back]))
    assert m is not None and not m.xray.ticks_major and not m.xray.lines and m.tip_in_field
    assert m.depth_mm is None

    # far outside the field: nothing magenta, the numbers stay
    m = b.build(xv._points(view, (-4000.0, 512.0), (-3900.0, 512.0)))
    assert m is not None and not m.tip_in_field
    assert not m.camera.magenta_lines and m.camera.magenta_tip is None
    assert not m.xray.lines and m.xray.tip is None and m.off_plane_mm is not None
    assert m.depth_mm is None and not m.xray.ticks_major

    # no video: no camera marks, the X-ray panel and the numbers stay
    b.set_camera_panel(None)
    m = b.build(xv._points(view, (512.0, 512.0), (612.0, 612.0)))
    assert m.camera.green_tip is None and not m.camera.green_lines
    assert not m.camera.magenta_lines and m.camera.magenta_tip is None
    assert m.xray.lines and m.xray.tip is not None and m.off_plane_mm is not None
    assert b.field_outline_on_camera() == []

    print("  (c) ray clip, degenerate poses")


# ------------------------------------------------------------ recordings

def _case(sess="Evaluation_Integrated_System", snap="B_2_live"):
    if not (xv.DATA / sess / snap / "snapshot.json").exists():
        _skip(f"no recording {sess}/{snap}")
    rec = xv._load(sess, snap)
    distortion = xv._field_distortion()
    return rec, xv._working(rec, distortion)


def _builder_of(rec, work, side=640, orientation=ORIENT_CAMERA, roi=(0, 0, 1920, 1080)):
    view = XrayView(rec.T, rec.K_x)
    ov = TrackingOverlay(rec.K_c, rec.dist, rec.p_tip, rec.axis)
    h, w = work.shape[:2]
    field = XrayField.from_image(work, ov.field_inset_px(view, (w, h), rec.d_x))
    ov.set_xray_view(view, field, rec.d_x)
    ov.update(rec.pts)
    b = MarkBuilder(rec.K_c, rec.dist)
    b.set_camera_panel(roi)
    b.set_xray(view, field, rec.d_x, (w, h))
    m = PanelMap.of(orientation, (w, h), side)
    b.set_xray_panel(m)
    return b, view, field, ov, m


def _mid(q):
    return (q[0] + q[2]) / 2.0, (q[1] + q[3]) / 2.0


# ------------------------------------------------------------ (d)

def test_same_as_chat1() -> None:
    """The geometry of today's drawing, on all snapshots of Fig. 4.21."""
    n = 0
    for _, sess, snap in xv.CASES:
        rec, work = _case(sess, snap)
        b, view, field, ov, m = _builder_of(rec, work)
        marks = b.build(rec.pts)
        cam = marks.camera
        # green: same tip, same dashes (rounded as TrackingOverlay draws them)
        assert np.abs(np.array(cam.green_tip) - 0.5 - ov._tip_uv).max() < 1e-9
        tip, back = ov._tip_uv, ov._back_uv
        for i, q in zip(range(0, ov.DASH_PIECES, 2), cam.green_lines):
            a = tip + (back - tip) * (i / ov.DASH_PIECES)
            assert np.abs(np.array(q[:2]) - 0.5 - a).max() < 1e-9
        # magenta: same segments, same tip
        rec_v = ov.snapshot_record()["xray_view"]
        got = [[[int(round(q[0] - 0.5)), int(round(q[1] - 0.5))],
                [int(round(q[2] - 0.5)), int(round(q[3] - 0.5))]] for q in cam.magenta_lines]
        assert got == rec_v["segments_px"], (sess, snap, got, rec_v["segments_px"])
        assert marks.tip_in_field == rec_v["tip_in_field"]
        if cam.magenta_tip is not None:
            assert np.abs(np.array(cam.magenta_tip) - 0.5 - np.array(rec_v["tip_uv_on_plane"])).max() < 1e-9
        # the numbers
        assert abs(marks.off_plane_mm - rec_v["off_plane_mm"]) < 1e-9
        assert abs(marks.inclination_deg - rec_v["inclination_deg"]) < 1e-9
        # X-ray panel: tip through the map
        uv = view.uv_x(rec.pts[:1])[0]
        want = m.apply(*uv.tolist())
        if marks.xray.tip is not None:
            assert math.hypot(marks.xray.tip[0] - want[0], marks.xray.tip[1] - want[1]) < 1e-9
        n += 1
    print(f"  (d) same geometry as TrackingOverlay on {n} snapshots")


# ------------------------------------------------------------ (e)

def _inverse(m: PanelMap, x: float, y: float):
    # the map is a scale times a turn; both orientations have b = d = 0
    return (x - m.c) / m.a, (y - m.f) / m.e


def test_ruler_real() -> None:
    rec, work = _case()
    b, view, field, _, m = _builder_of(rec, work)
    tip_c, back_c = rec.pts
    fwd = (tip_c - back_c) / np.linalg.norm(tip_c - back_c)

    def where(s):
        return m.apply(*view.uv_x((tip_c + s * fwd)[None, :])[0].tolist())

    # zero set at the tip: every tick at its real place along the 3D axis
    b.set_ruler(RULER_SET, tip_c)
    x = b.build(rec.pts).xray
    assert x.ticks_major and x.ticks_minor and x.labels
    for i, q in enumerate(x.ticks_major):
        want = where(10.0 * i)
        got = _mid(q)
        assert math.hypot(got[0] - want[0], got[1] - want[1]) < 1e-6, (i, got, want)
    for i, q in enumerate(x.ticks_minor):
        want = where(5.0 + 10.0 * i)
        got = _mid(q)
        assert math.hypot(got[0] - want[0], got[1] - want[1]) < 1e-6
    assert [t for _, _, t in x.labels] == [str(10 * i) for i in range(len(x.labels))]
    # the ring is 2 mm at the depth of the tip
    z = view.to_xray(rec.pts[:1])[0, 2]
    assert abs(x.tip_radius - TIP_RING_MM * rec.K_x[0, 0] / z * m.scale) < 1e-9

    # inclined: ticks are closer than for the same axis parallel to Pi
    P = view.to_xray(rec.pts)
    flat = P.copy()
    d = P[1] - P[0]
    flat[1] = P[0] + np.array([d[0], d[1], 0.0]) / np.linalg.norm(d[:2]) * np.linalg.norm(d)
    flat_c = flat @ view.R.T + view.t_mm
    x_flat = b.build(flat_c).xray

    def first_gap(xm):
        a, c = _mid(xm.ticks_major[0]), _mid(xm.ticks_major[1])
        return math.hypot(a[0] - c[0], a[1] - c[1])

    ratio = first_gap(x) / first_gap(x_flat)
    assert 0.8 < ratio < 0.95, ratio    # cos 27.7 = 0.885, plus perspective

    # everything inside the field, in the working image
    for xm in (x, x_flat):
        for q in xm.lines:
            for px, py in ((q[0], q[1]), (q[2], q[3])):
                u, v = _inverse(m, px, py)
                assert field.inner_distance(u, v) > -1e-6

    # Null setzen: the tip 20 mm further reads 20; the zero stays in place along the axis
    b.set_ruler(RULER_SET, tip_c)
    moved = rec.pts + 20.0 * fwd
    marks = b.build(moved)
    assert abs(marks.depth_mm - 20.0) < 1e-9
    x = marks.xray
    labels = {t: (a, c) for a, c, t in x.labels}
    assert "0" in labels and "20" in labels
    # the tick of label 20 is at the moved tip
    tick = _mid(x.ticks_major[2])
    assert math.hypot(tick[0] - x.tip[0], tick[1] - x.tip[1]) < 1e-6
    # backwards: negative depth, the zero is ahead of the tip
    marks = b.build(rec.pts - 10.0 * fwd)
    assert abs(marks.depth_mm + 10.0) < 1e-9

    # ------ zero at the field entry, the standard
    b.set_ruler(RULER_FIELD, None)
    marks = b.build(rec.pts)
    x = marks.xray
    L = marks.depth_mm
    assert 40.0 < L < 90.0, L
    # the first tick is on the field edge and labelled 0, the rest count from there
    assert x.labels[0][2] == "0"
    u, v = _inverse(m, *_mid(x.ticks_major[0]))
    assert abs(field.inner_distance(u, v)) < 1e-6
    entry_c = tip_c - L * fwd
    for i, q in enumerate(x.ticks_major):
        want = m.apply(*view.uv_x((entry_c + 10.0 * i * fwd)[None, :])[0].tolist())
        got = _mid(q)
        assert math.hypot(got[0] - want[0], got[1] - want[1]) < 1e-6, i
    # the closed form: the edge is a plane through the source with normal K_x^T l
    uv_e = view.uv_x(entry_c[None, :])[0]
    edge = min(field._edges, key=lambda e: abs(e[0] * uv_e[0] + e[1] * uv_e[1] - e[2]))
    l = np.array([edge[0], edge[1], -edge[2]])
    P = view.to_xray(rec.pts)
    a_x = (P[0] - P[1]) / np.linalg.norm(P[0] - P[1])
    closed = float(l @ rec.K_x @ P[0]) / float(l @ rec.K_x @ a_x)
    assert abs(closed - L) < 1e-6, (closed, L)
    # ... and it does not depend on d_x
    b.set_plane_depth(rec.d_x + 10.0, work.shape[::-1])
    assert abs(b.build(rec.pts).depth_mm - L) < 1e-9
    b.set_plane_depth(rec.d_x, work.shape[::-1])

    # along the axis the entry stays: the reading follows exactly
    assert abs(b.build(rec.pts + 10.0 * fwd).depth_mm - (L + 10.0)) < 1e-9
    # sideways it moves a little (KONZEPT 4.3: median 0.14, at most 0.40 mm per mm)
    side = np.cross(fwd, view.R[:, 2])
    side /= np.linalg.norm(side)
    assert abs(b.build(rec.pts + side).depth_mm - L) < 0.5
    # not yet in the field: negative, and the zero waits at the edge ahead of the tip
    before = b.build(rec.pts - (L + 20.0) * fwd)
    assert not before.tip_in_field and abs(before.depth_mm + 20.0) < 1e-9
    assert before.xray.labels[0][2] == "0"
    u, v = _inverse(m, *_mid(before.xray.ticks_major[0]))
    assert abs(field.inner_distance(u, v)) < 1e-6
    print(f"  (e) ruler real in 3D (inclined/parallel {ratio:.3f}), Null setzen, field entry "
          f"(B_2: tip {L:.1f} mm past it, closed form agrees)")


# ------------------------------------------------------------ (f)

def test_overlay_in_roi() -> None:
    from overlay.gui.widgets.widget_live_panels import overlay_in_roi
    from overlay.tools.homography import estimate_plane_induced_homography
    from overlay.tools.warp import prepare_live_overlay

    rec, work = _case()
    bgra = cv2.cvtColor(cv2.imread(str(rec.dir / "01_rgb.png")), cv2.COLOR_BGR2BGRA)
    H = estimate_plane_induced_homography(
        K_c=rec.K_c, R_xc=rec.T[:3, :3], t_xc=rec.T[:3, 3], K_x=rec.K_x, d_x=rec.d_x
    )
    full = prepare_live_overlay(work, H, (1920, 1080), K_camera=rec.K_c, dist_coeffs=rec.dist)
    want = bgra.copy()
    full.blend_into(want, 0.3)
    for roi in ((750, 0, 640, 640), (800, 100, 300, 200), (0, 0, 1920, 1080)):
        x0, y0, w, h = roi
        got = bgra[y0:y0 + h, x0:x0 + w].copy()
        o = overlay_in_roi(full, roi)
        o.blend_into(got, 0.3)
        assert np.array_equal(got, want[y0:y0 + h, x0:x0 + w]), roi
    assert overlay_in_roi(full, (0, 900, 100, 100)) is None
    print("  (f) X-ray blend in the crop = crop of the full-frame blend")


# ------------------------------------------------------------ (g)

def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv[:1])


def _pointer_d_x(image, K, dist, T_xc, corner_light: str = "linear") -> float:
    """d_x of one pointer frame through the production solve (PointerDepthEstimator's
    settings), raw: what a single-frame plane search accepts."""
    from overlay.calib.calib_camera_to_pointer import calibrate_camera_to_pointer
    from overlay.calib.calib_xray_to_pointer import extract_depth
    r = calibrate_camera_to_pointer(image_bgr=np.asarray(image), camera_intrinsics=K, dist_coeffs=dist,
                                    pose_method="ippe", refine_with_iterative=True, corner_light=corner_light)
    assert r.corner_light == corner_light, r.corner_light
    return float(extract_depth(T_xc=T_xc, T_tc=r.T_4x4).d_x_mm)


def _qimage_to_rgb(img) -> np.ndarray:
    from PySide6.QtGui import QImage
    img = img.convertToFormat(QImage.Format_RGB32)
    h, w = img.height(), img.width()
    a = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w, :3]
    return a[:, :, ::-1].copy()


def _is_magenta(rgb: np.ndarray) -> np.ndarray:
    return (rgb[:, :, 0] > 200) & (rgb[:, :, 1] < 80) & (rgb[:, :, 2] > 200)


def _spin(app, seconds: float, until=None) -> bool:
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.002)
    return until is None


def test_xray_panel() -> None:
    from PySide6.QtCore import QLineF
    from PySide6.QtWidgets import QApplication
    from overlay.gui.widgets.widget_live_panels import XrayToolPanel

    rec, work = _case()
    app = _app()
    panel = XrayToolPanel("2  X-ray AP")
    panel.set_ui_scale(1.0, 30)
    panel.resize(640, 670)
    panel.show()
    panel.set_xray(work)
    _spin(app, 0.1)
    assert panel.panel_map is not None and panel._side == 640

    b, view, field, _, m = _builder_of(rec, work)
    b._cfg = b._cfg._replace(line_factory=QLineF)
    marks = b.build(rec.pts)

    def on_screen():
        """What the partial repaints left in the backing store, not a fresh render."""
        app.processEvents()
        return _qimage_to_rgb(QApplication.primaryScreen().grabWindow(panel.winId()).toImage())

    before = on_screen()
    assert not _is_magenta(before).any()

    panel.set_marks(marks)
    a = on_screen()
    ys, xs = np.nonzero(_is_magenta(a))
    assert len(xs) > 200
    # the tip mark sits on the tip, in the image area below the header
    tx, ty = marks.xray.tip
    near = np.hypot(xs - tx, ys - 30 - ty) < 25.0
    assert near.sum() > 30

    # outside the field no magenta, in the working image
    for x, y in zip(xs[::7].tolist(), ys[::7].tolist()):
        u, v = _inverse(m, x + 0.5, y - 30 + 0.5)
        assert field.inner_distance(u, v) > -24.0, (x, y)

    # a move: the old marks are gone, exactly the new ones remain
    moved = b.build(rec.pts + np.array([18.0, -9.0, 0.0]))
    panel.set_marks(moved)
    got = on_screen()
    ref = _qimage_to_rgb(panel.grab().toImage())    # fresh render of everything
    assert np.array_equal(got[30:], ref[30:]), int((got[30:] != ref[30:]).sum())

    # none: nothing left at all
    panel.set_marks(None)
    got = on_screen()
    assert np.array_equal(got[30:], before[30:])

    # the other orientation: the tip turns 180 degrees with the image
    panel.set_orientation(ORIENT_C_ARM)
    m2 = panel.panel_map
    b.set_xray_panel(m2)
    x2 = b.build(rec.pts).xray
    assert abs(x2.tip[0] + marks.xray.tip[0] - 640.0) < 1e-6
    assert abs(x2.tip[1] + marks.xray.tip[1] - 640.0) < 1e-6
    panel.set_marks(b.build(rec.pts))
    assert _is_magenta(on_screen()).any()
    panel.close()

    # placeholder: same class, no image
    ph = XrayToolPanel("3")
    ph.set_placeholder("LATERAL", "noch nicht aufgenommen")
    ph.resize(640, 670)
    assert ph.panel_map is None
    ph.set_marks(marks)     # harmless
    _ = ph.grab()
    print("  (g) X-ray panel: partial repaints leave no residue, field, orientation, placeholder")


def test_camera_panel_direct() -> None:
    """Drawn straight from the frame: the same pixels as copy + blend, and a camera view
    is not touched again after show_frame."""
    from PySide6.QtWidgets import QApplication
    from overlay.gui.widgets.widget_live_panels import CameraPanel
    from overlay.tools.homography import estimate_plane_induced_homography
    from overlay.tools.warp import prepare_live_overlay

    rec, work = _case()
    app = _app()
    bgra = cv2.cvtColor(cv2.imread(str(rec.dir / "01_rgb.png")), cv2.COLOR_BGR2BGRA)
    H = estimate_plane_induced_homography(
        K_c=rec.K_c, R_xc=rec.T[:3, :3], t_xc=rec.T[:3, 3], K_x=rec.K_x, d_x=rec.d_x
    )
    full = prepare_live_overlay(work, H, (1920, 1080), K_camera=rec.K_c, dist_coeffs=rec.dist)

    panel = CameraPanel("1")
    panel.set_ui_scale(1.0, 30)
    panel.resize(1000, 630)
    panel.show()
    _spin(app, 0.1)
    panel.set_frame_size((1920, 1080))
    panel.set_centre(1000.0, 330.0)
    panel.set_overlay(full)
    panel.alpha = 0.3
    x0, y0, w, h = panel.roi
    assert (w, h) == (1000, 600)
    want = bgra.copy()
    full.blend_into(want, 0.3)
    want = want[y0:y0 + h, x0:x0 + w, :3][:, :, ::-1]

    # stays valid: a fresh render finds the frame
    panel.show_frame(bgra, stays_valid=True)
    got = _qimage_to_rgb(panel.grab().toImage())[30:]
    assert np.array_equal(got, want), int((got != want).sum())

    def on_screen():
        return _qimage_to_rgb(QApplication.primaryScreen().grabWindow(panel.winId()).toImage())[30:]

    # a camera view: on screen when show_frame returns, and never read again
    other = np.roll(bgra, 40, axis=1)
    view = other.copy()
    panel.show_frame(view)
    assert panel._qimg is None and panel._frame is None
    want2 = other.copy()
    full.blend_into(want2, 0.3)
    want2 = want2[y0:y0 + h, x0:x0 + w, :3][:, :, ::-1]
    view[:] = 0                 # the next poll took the memory
    assert np.array_equal(on_screen(), want2)
    panel.update()
    _spin(app, 0.1)
    assert np.array_equal(on_screen(), want2), "painted from a frame it had let go of"

    # without an overlay: just the crop
    panel.set_overlay(None)
    panel.show_frame(bgra, stays_valid=True)
    got = _qimage_to_rgb(panel.grab().toImage())[30:]
    assert np.array_equal(got, bgra[y0:y0 + h, x0:x0 + w, :3][:, :, ::-1])

    # letterboxed (the laptop at dpr 2: the panel taller than the frame), the tip above the
    # crop: nothing of it may land on the bands, which are filled only once (lab 10.11.)
    from PySide6.QtCore import QLineF
    from overlay.tools.tool_marks import CameraMarks, ToolMarks
    small = np.ascontiguousarray(bgra[:400, :600])
    panel.set_frame_size((600, 400))
    panel.set_centre(300.0, 200.0)
    _spin(app, 0.05)
    lx, ly, lw, lh = panel.roi
    assert (lw, lh) == (600, 400) and panel._oy > 30 and panel._ox > 0     # bands all round
    for tip_y in (-25.0, -10.0, 5.0):
        cm = CameraMarks([QLineF(300.0, 200.0, 300.0, tip_y)], (300.0, tip_y), [], None, False)
        panel.set_marks(ToolMarks(0.0, False, "fused", (0.0, 0.0, 500.0), (0.0, 0.0, 1.0), cm,
                                  None, None, None, None, False, None))
        panel.show_frame(small, stays_valid=True)
    img = _qimage_to_rgb(panel.grab().toImage())
    top, left = panel._oy, panel._ox
    green = (img[:, :, 1] > 150) & (img[:, :, 0] < 120) & (img[:, :, 2] < 120)
    assert not green[30:top].any() and not green[top + lh:].any(), "a mark on the band above / below"
    assert not green[:, :left].any() and not green[:, left + lw:].any(), "a mark on the band beside"
    assert green[top:top + lh, left:left + lw].any(), "the mark inside the picture is gone"
    panel.close()
    print("  (f2) video drawn directly: same pixels as copy + blend, a camera view is let go of; "
          "marks stay off the letterbox bands")


# ------------------------------------------------------------ (h)

def test_design_window() -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from overlay.gui import design_live_panels as dp
    from overlay.gui.widgets.widget_live_panels import POSE_HOLD_S, VIEW_LIVE, VIEW_XRAY

    if not (dp.SNAPSHOTS["B_2 (drill inclined)"] / "snapshot.json").exists():
        _skip("no B_2")
    app = _app()
    snap = dp.Snapshot(dp.SNAPSHOTS["B_2 (drill inclined)"])
    assert snap.stream.shape[0] > 100 and 0.0 < snap.jitter_mm < 1.0
    win = dp.DesignWindow(snap, load=False)
    win.resize(1920, 1080)
    win.show()
    try:
        v = win.view
        assert _spin(app, 5.0, lambda: v.panel_view1._xm is not None), "no marks on panel 2"
        assert v.panel_live._marks is not None
        # video left, AP over lateral right, X-ray squares filling the height
        ap, lat, live = v.panel_view1.geometry(), v.panel_view2.geometry(), v.panel_live.geometry()
        assert ap.width() == 510 and ap.height() == 540 and v.panel_view1._side == 510
        assert lat.top() == ap.bottom() + 1 and lat.bottom() == 1079 and lat.left() == ap.left()
        assert live.left() == 0 and live.right() + 1 == ap.left() and v.bottom.geometry().top() == live.bottom() + 1
        assert v.bottom.geometry().bottom() == 1079 and live.height() > 600
        # the crop is native and contains the X-ray field
        rw, rh = v.panel_live.roi[2:]
        assert rw == live.width() and rh == live.height() - 30
        outline = np.array(win.builder.field_outline_on_camera())
        assert outline[:, 0].min() > 0.0 and outline[:, 0].max() < rw, outline
        assert outline[:, 1].min() > 0.0 and outline[:, 1].max() < rh, outline
        _spin(app, 0.5)
        assert v.tile_off_plane._value != "–" and abs(float(v.tile_inclination._value) - 27.7) < 1.0

        # the virtual move reaches the marks
        tip_before = v.panel_view1._marks.xray.tip
        win.sl_advance.set(-30.0)
        assert _spin(app, 2.0, lambda: v.panel_view1._marks is not None and
                     v.panel_view1._marks.xray.tip != tip_before and
                     math.hypot(v.panel_view1._marks.xray.tip[0] - tip_before[0],
                                v.panel_view1._marks.xray.tip[1] - tip_before[1]) > 50.0)
        win.btn_reset.click()

        # Null setzen, then the depth follows the advance
        _spin(app, 0.3)
        win.set_zero()
        win.sl_advance.set(25.0)
        assert _spin(app, 2.0, lambda: v.panel_view1._marks is not None and
                     v.panel_view1._marks.depth_mm is not None and
                     abs(v.panel_view1._marks.depth_mm - 25.0) < 1.5), "no depth"
        win._reset_moves()

        # a lost pose: within POSE_HOLD_S + a frame the marks are gone and it says so
        v.btn_track.click()
        t0 = time.perf_counter()
        assert _spin(app, 2.0, lambda: v.panel_view1._lost), "still drawn"
        assert time.perf_counter() - t0 <= POSE_HOLD_S + 0.3
        assert v.panel_view1._xm is None and v.panel_live._marks is None
        v.btn_track.click()
        assert _spin(app, 2.0, lambda: not v.panel_view1._lost)

        # view 2: only the two X-rays, larger; the area below does not move
        side_before = v.panel_view1._side
        bottom_before = v.bottom.geometry()
        view_x = win._view_x
        assert v.view_buttons[VIEW_LIVE].isChecked()
        v.view_buttons[VIEW_XRAY].click()
        assert v.view == VIEW_XRAY and v.view_buttons[VIEW_XRAY].isChecked()
        assert not v.view_buttons[VIEW_LIVE].isChecked()
        _spin(app, 0.2)
        assert not v.panel_live.isVisible() and v.panel_view1.isVisible() and v.panel_view2.isVisible()
        ap, lat = v.panel_view1.geometry(), v.panel_view2.geometry()
        assert ap.size() == lat.size() and lat.left() == ap.right() + 1 and lat.top() == ap.top() == 0
        assert v.panel_view1._side > side_before
        assert ap.left() == 1920 - 1 - lat.right()      # centred
        assert v.bottom.geometry() == bottom_before and ap.bottom() + 1 == bottom_before.top()
        # new poses are built for the new panel, and none of the video
        old = v.panel_view1._marks

        def on_new_map():
            m = v.panel_view1._marks
            if m is None or m is old or m.xray is None or m.xray.tip is None:
                return False
            uv = view_x.uv_x(np.array([m.tip_cam]))[0]
            want = v.panel_view1.panel_map.apply(*uv.tolist())
            return (math.hypot(m.xray.tip[0] - want[0], m.xray.tip[1] - want[1]) < 1e-6
                    and m.camera.green_tip is None and not m.camera.magenta_lines)

        assert _spin(app, 2.0, on_new_map), "marks not on the new panel map"
        # the selector stays in the corner, the header text leaves it room
        sel = v.view_buttons[VIEW_LIVE].geometry()
        assert v.view_buttons[VIEW_XRAY].geometry().right() < 1920 and sel.top() >= 0
        assert v.panel_view1._right_inset == 0
        assert (v.panel_view2._right_inset > 0) == (v.panel_view2.geometry().right() >= sel.left())
        v.view_buttons[VIEW_LIVE].click()
        _spin(app, 0.3)
        assert v.view == VIEW_LIVE and v.panel_view1._side == side_before and v.panel_live.isVisible()
        assert v.panel_live.roi[2:] == (rw, rh) and v.bottom.geometry() == bottom_before
        assert v.panel_view1._right_inset > 0 and v.panel_view2._right_inset == 0
        assert _spin(app, 2.0, lambda: v.panel_live._marks is not None
                     and v.panel_live._marks.camera.green_tip is not None), "no camera marks back"

        # Set reference plane on the recorded pointer frames, with the real pointer solve: a new
        # plane, the X-ray stays. Tracking pauses and comes back with its button
        _spin(app, 0.3)
        off_before = v.panel_view1._marks.off_plane_mm
        depth_before = v.panel_view1._marks.depth_mm
        bg_before = v.panel_view1._working.copy()
        d_before = win._base_dx
        v.btn_plane.click()
        assert win._plane_acquiring and v.btn_plane.text() == "Cancel" and not v.btn_track.isEnabled()
        assert not win._tracking_on and win._frame is not snap.bgra
        assert "Pointer found" in v.lbl_status.text()
        with np.load(win._plane_capture) as z:
            # today's pointer solve on that recorded frame (linear-light corners since 11.11.)
            want = _pointer_d_x(z["rgb_image"], snap.K_c, snap.dist, snap.T_xc)
            assert abs(win._plane_new_dx - want) < 1e-3, (win._plane_capture, win._plane_new_dx, want)
            if str(z["pose_method"]) == "ippe":
                # the camera-grey corners give the recorded value back
                old = _pointer_d_x(z["rgb_image"], snap.K_c, snap.dist, snap.T_xc, corner_light="camera")
                assert abs(old - float(z["d_x_raw_mm"])) < 1e-3, (old, float(z["d_x_raw_mm"]))
        QTest.keyClick(win, Qt.Key_Escape)      # cancelled: nothing changed
        assert not win._plane_acquiring and win._base_dx == d_before and win._frame is snap.bgra
        assert v.btn_plane.text() == "Set reference plane" and v.btn_track.isEnabled()
        v.btn_plane.click()                     # the next recorded frame
        d_new = win._plane_new_dx
        assert abs(d_new - d_before) > 100.0, d_new     # the phantom's plane, 132 mm away
        QTest.keyClick(win, Qt.Key_Space)
        assert not win._plane_acquiring and win._base_dx == d_new and win._frame is snap.bgra
        assert "same X-ray" in v.lbl_status.text()
        assert np.array_equal(v.panel_view1._working, bg_before)
        assert v.slider_dx.value() == 0
        assert not win._tracking_on and v.btn_track.text() == "Start tracking"
        # the crop has followed the field onto the new plane
        rw2, rh2 = v.panel_live.roi[2:]
        outline = np.array(win.builder.field_outline_on_camera())
        assert outline[:, 0].min() > 0.0 and outline[:, 0].max() < rw2, outline
        assert outline[:, 1].min() > 0.0 and outline[:, 1].max() < rh2, outline
        v.btn_track.click()
        want_off = off_before + d_before - d_new

        def off_on_new_plane():
            m = v.panel_view1._marks
            return m is not None and abs(m.off_plane_mm - want_off) < 1.0

        assert _spin(app, 2.0, off_on_new_plane), "the tip is not read against the new plane"
        # the X-ray panel does not depend on d_x: the depth reading stays
        assert abs(v.panel_view1._marks.depth_mm - depth_before) < 1.0
        _spin(app, 0.3)
        assert "Tracking active" in v.lbl_status.text()
        win._set_plane(d_before)
        _spin(app, 0.2)

        # a panel fills the window and comes back
        v.toggle_maximized(v.panel_live)
        _spin(app, 0.2)
        assert v.panel_live.roi[2] == 1920 and not v.panel_view1.isVisible()
        v.toggle_maximized(v.panel_live)
        _spin(app, 0.2)
        assert v.panel_live.roi[2:] == (rw, rh) and v.panel_view1.isVisible()

        # other recording, other images
        win.cmb_snapshot.setCurrentText("Bone_New (bone)")
        assert _spin(app, 3.0, lambda: v.panel_view1._xm is not None)
    finally:
        win.close()
    print("  (h) design window: replay, virtual move, Null setzen, lost pose, maximise, other recording")


def test_view_at_start() -> None:
    """The area below does not depend on the view the window starts in."""
    from overlay.gui.widgets.widget_live_panels import VIEW_LIVE, VIEW_XRAY, LivePanelsView

    app = _app()
    boxes = []
    for view in (VIEW_LIVE, VIEW_XRAY):
        v = LivePanelsView()
        v.set_view(view)
        v.resize(1920, 1080)
        v.show()
        _spin(app, 0.2)
        boxes.append(v.bottom.geometry())
        # every tile can hold its number
        assert min(t.height() for t in v._tiles) >= 70
        v.close()
    assert boxes[0] == boxes[1], boxes
    print("  (i) same area below in both views from the start, tiles keep their numbers")


def test_readout_tile() -> None:
    """Only the value line repaints, and it leaves nothing behind; an unchanged value paints
    nothing."""
    from PySide6.QtWidgets import QApplication
    from overlay.gui.widgets.widget_live_panels import LivePanelsView, ReadoutTile

    app = _app()
    v = LivePanelsView()
    # the insertion depth is on the ruler in the X-ray panels, no tile (user 10.11.)
    assert v._tiles == (v.tile_off_plane, v.tile_inclination)
    assert [q._title for q in v._panels] == [
        "X-ray and Video Overlay", "X-ray View 1 + Tool Tracking", "X-ray View 2 + Tool Tracking",
    ]
    for name in ("tile_shaft", "tile_tsd", "tile_tad", "tile_depth"):
        assert not hasattr(v, name)
    v.close()

    tile = ReadoutTile("Tip to plane Π", "mm")
    tile.set_ui_scale(1.0)
    tile.resize(190, 130)
    tile.show()
    _spin(app, 0.1)

    def on_screen():
        app.processEvents()
        return _qimage_to_rgb(QApplication.primaryScreen().grabWindow(tile.winId()).toImage())

    assert tile._value == "–"
    empty = on_screen()
    assert (empty == empty[0, 0]).mean() < 1.0      # the label and the dash are there

    paints = []
    tile.prof = lambda name, dt: paints.append(dt)
    for text in ("60", "-13", "123", "8"):
        tile.set_value(text)
        got = on_screen()
        ref = _qimage_to_rgb(tile.grab().toImage())     # a fresh render of everything
        assert np.array_equal(got, ref), text
    assert not (got == empty).all()
    # the widest number with its unit stays inside the tile
    tile.set_value("-188")
    wide = on_screen()
    bg_row = wide[wide.shape[0] // 2, -3]
    assert (wide[:, -6:] == bg_row).all() or (wide[:, -6:, 0] < 60).all(), "the value runs over the edge"
    tile.set_value("8")
    on_screen()
    n = len(paints)
    tile.set_value("8")
    _spin(app, 0.05)
    assert len(paints) == n, "an unchanged value painted"
    # the value's repaint leaves the label alone
    assert tile._value_rect.top() > 0 and tile._value_rect.height() == tile.height()
    tile.set_value(None)
    assert np.array_equal(on_screen(), empty)
    tile.set_label("Inclination to plane Π")
    assert not np.array_equal(on_screen(), empty)
    tile.close()
    print("  (j) two read-outs; repainting a value leaves nothing behind, an unchanged one paints nothing")


def test_readout_int() -> None:
    """Read-outs as whole numbers, half away from zero, never "-0", sign only where asked."""
    from overlay.gui.widgets.widget_live_panels import readout_int

    assert readout_int(27.7) == "28" and readout_int(27.4) == "27"
    assert readout_int(-1.4, signed=True) == "-1" and readout_int(-1.6, signed=True) == "-2"
    assert readout_int(2.5) == "3" and readout_int(-2.5) == "-3"          # not Python's 2.5 -> 2
    assert readout_int(-0.4, signed=True) == "0" and readout_int(0.4, signed=True) == "0"
    assert readout_int(3.2, signed=True) == "+3" and readout_int(3.2) == "3"
    assert readout_int(None) is None and readout_int(float("nan")) is None
    print("  (j2) read-outs rounded to whole numbers")


def test_plane_side_view() -> None:
    """Tip and axis against Π: the signed rise from the builder, the compressed height that never
    leaves the window, the tip on the side of the sign, and a repaint only when something moved."""
    from PySide6.QtWidgets import QApplication
    from overlay.gui.widgets.widget_live_panels import PlaneSideView

    # the sign of the rise: positive when the drill behind the tip is farther from the source
    rec, work = _case()
    b, view, field, ov, m = _builder_of(rec, work)
    marks = b.build(rec.pts)
    P = view.to_xray(rec.pts)
    assert abs(abs(marks.rise_deg) - marks.inclination_deg) < 1e-9
    assert (marks.rise_deg > 0.0) == (P[1, 2] > P[0, 2])        # B_2: the back is farther, it rises
    flipped = b.build(rec.pts[::-1].copy())                     # tip and back swapped
    assert abs(flipped.rise_deg + marks.rise_deg) < 1e-9

    # the height: linear near Π, half way at H0, inside for any distance, odd
    f = PlaneSideView.height_fraction
    assert f(0.0) == 0.0 and abs(f(PlaneSideView.H0_MM) - 0.5 * PlaneSideView.REACH) < 1e-12
    assert all(abs(f(h)) < PlaneSideView.REACH for h in (1.0, 50.0, 1e3, 1e6))
    assert f(-20.0) == -f(20.0) and f(1.0) < f(2.0) < f(30.0) < f(300.0)

    app = _app()
    w = PlaneSideView()
    w.set_ui_scale(1.0)
    w.resize(330, 140)
    w.show()
    _spin(app, 0.1)

    def tip_row(off):
        w.set_state(off, 0.0)
        app.processEvents()
        img = _qimage_to_rgb(w.grab().toImage())
        rows = np.flatnonzero(_is_magenta(img).any(axis=1))
        return rows.mean()

    above, on, below = tip_row(+5.0), tip_row(0.0), tip_row(-5.0)
    assert above < on < below, (above, on, below)               # up is the positive side
    far_up, far_down = tip_row(+500.0), tip_row(-500.0)
    img = _qimage_to_rgb(w.grab().toImage())
    assert _is_magenta(img).any() and far_up < above and far_down > below   # still in the window

    # solid trajectory ahead of the tip, dashed drill behind it, as in panel 1 and the X-ray panels
    w.set_state(0.0, 0.0)
    app.processEvents()
    img = _qimage_to_rgb(w.grab().toImage())
    dpr = img.shape[1] / w.width()
    left, right, top, bottom, yc, half = w._layout()
    xt = left + w.TIP_X * (right - left)
    row = _is_magenta(img)[int(round(yc * dpr))]
    ring = int(math.ceil(10.0 * dpr))
    ahead = row[int(left * dpr) + 2:int(xt * dpr) - ring]
    behind = row[int(xt * dpr) + ring:int(right * dpr) - 2]
    assert ahead.all(), "the trajectory ahead of the tip is not solid"
    assert 0.3 < behind.mean() < 0.7, behind.mean()            # the drill: dashes and gaps

    paints = []
    w.prof = lambda name, dt: paints.append(dt)
    w.set_state(3.0, 20.0)
    _spin(app, 0.05)
    n = len(paints)
    w.set_state(3.0 + 0.02, 20.0 + 0.1)                        # below a pixel and half a degree
    _spin(app, 0.05)
    assert len(paints) == n, "a pose that did not move painted"
    w.set_state(None, None)
    _spin(app, 0.05)
    assert not _is_magenta(_qimage_to_rgb(w.grab().toImage())).any()   # no pose: only Π
    w.close()
    print("  (j3) plane side view: sign of the rise, compressed height inside the window, solid trajectory "
          "and dashed drill, repaint only on change")


def test_page_plane_only() -> None:
    """The real page, camera and tracker stubbed, the pointer solve real on a recorded pointer
    frame: a new plane for the X-ray on screen, without a file dialog; today's rail button still
    asks for an X-ray."""
    from PySide6.QtWidgets import QMessageBox

    rec, _ = _case()
    distortion = xv._field_distortion()
    z = np.load(xv.DATA / "Evaluation_Integrated_System" / "B_2_tracking.npz")
    ok = np.isfinite(z["rvec"]).all(axis=1)
    with np.load(xv.DATA / "Evaluation_Integrated_System" / "Bone_2_depth.npz") as d:
        pointer_frame = cv2.cvtColor(d["rgb_image"], cv2.COLOR_BGR2BGRA)
    # what today's pointer solve gives on that frame (corners in linear light since 11.11.;
    # the recorded d_x_raw_mm came from the camera-grey corners)
    d_recorded = _pointer_d_x(pointer_frame, rec.K_c, rec.dist, rec.T)
    app = _app()
    box_question = QMessageBox.question
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    try:
        page = xv._make_page(rec, rec.raw, distortion, (z["rvec"][ok], z["tvec"][ok]))
        page.on_enter()
        ctl = page.controls
        asked = []
        page._ask_xray_path = lambda: asked.append(1)    # returns None: cancelled
        ctl.btn_plane.click()
        assert asked and not page._plane_acquiring      # today's rail: unchanged

        ctl.btn_track.click()
        assert page._tracking_active
        assert _spin(app, 5.0, lambda: page._drawn_out is not None), "no pose"
        working, image = page._xray_working_u8, page.state.xray_image_anatomy
        field = page._xray_geometry[1]
        base = page._base_d_x_mm
        n_asked = len(asked)

        page.start_plane_only()
        assert len(asked) == n_asked and page._plane_acquiring
        assert not page._tracking_active and ctl.btn_plane.text() == "Cancel"
        assert "same image" in ctl.lbl_plane_status.text()

        # the camera now sees the recorded pointer frame: the page's own pointer solve finds it
        camera_frame = page.camera.bgra
        page.camera.bgra = pointer_frame
        assert _spin(app, 5.0, lambda: page._plane_res is not None), "pointer not found"
        page._confirm_plane()
        page.camera.bgra = camera_frame

        d_new = page.state.d_x
        assert abs(d_new - d_recorded) < 0.5 and abs(d_new - base) > 100.0, (d_new, d_recorded, base)
        assert not page._plane_acquiring and ctl.btn_plane.text() == "Set reference plane"
        assert page._xray_working_u8 is working and page.state.xray_image_anatomy is image
        assert page._xray_geometry is not None and page._xray_geometry[1] is field
        assert page._base_d_x_mm == d_new and page._current_d_x_mm == d_new and page._planes_set == 1
        assert page._H_xc_lookup

        # tracking comes back with its button, on the new plane and the same field
        ctl.btn_track.click()
        assert page._tool_overlay._d_x_mm == d_new and page._tool_overlay._field is field
        assert _spin(app, 5.0, lambda: page._drawn_out is not None), "no pose after the plane"
        ctl.btn_track.click()

        # cancelled: nothing changes
        page.start_plane_only()
        page.start_plane_only()
        assert not page._plane_acquiring and page.state.d_x == d_new
        page.on_leave()
        page.close()
    finally:
        QMessageBox.question = box_question
    print("  (k) real page: Set reference plane without a file dialog, X-ray and field stay; today's rail unchanged")


def test_page_panels() -> None:
    """The real page with the study's panels, single view (decided 10.11.), camera and tracker
    stubbed: the marks come from the tracking thread into panel 1 and X-ray view 1, the read-outs
    follow; Set reference plane takes SPACE without a question and tracking comes back by itself;
    Load X-ray takes the next C-arm image (also one that came before the button), From folder (USB)
    one from a folder; the snapshot reports in the status line, not in a box."""
    import shutil
    import tempfile
    from pathlib import Path

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QMessageBox

    from overlay.gui.evaluation import EvaluationSession
    from overlay.gui.widgets.widget_live_panels import STATUS_BAD, STATUS_GOOD, STATUS_WARN

    rec, _ = _case()
    distortion = xv._field_distortion()
    z = np.load(xv.DATA / "Evaluation_Integrated_System" / "B_2_tracking.npz")
    ok = np.isfinite(z["rvec"]).all(axis=1)
    with np.load(xv.DATA / "Evaluation_Integrated_System" / "Bone_2_depth.npz") as d:
        pointer_frame = cv2.cvtColor(d["rgb_image"], cv2.COLOR_BGR2BGRA)
    # what today's pointer solve gives on that frame (corners in linear light since 11.11.;
    # the recorded d_x_raw_mm came from the camera-grey corners)
    d_recorded = _pointer_d_x(pointer_frame, rec.K_c, rec.dist, rec.T)
    app = _app()
    errors, info = [], []
    boxes = (QMessageBox.critical, QMessageBox.information, QMessageBox.question)

    def no_question(*_a, **_k):
        raise AssertionError("the panels ask no question")

    QMessageBox.critical = staticmethod(lambda *a, **k: errors.append(a[2:]))
    QMessageBox.information = staticmethod(lambda *a, **k: info.append(a[2:]))
    QMessageBox.question = staticmethod(no_question)
    tmp = Path(tempfile.mkdtemp(prefix="live_panels_page_"))
    try:
        ev = EvaluationSession(tmp, "session")
        ev.set_label("test")
        page = xv._make_page(rec, rec.raw, distortion, (z["rvec"][ok], z["tvec"][ok]),
                             ev=ev, panels=True)
        page.on_enter()
        p = page.panels
        assert page._panels_mode and page.controls is p and not errors, errors
        assert p.isVisible() and not page.image_label.isVisible() and not page._rail.isVisible()
        assert p.panel_view1.isVisible() and not p.panel_view2.isVisible()
        assert not any(b.isVisible() for b in p.view_buttons.values())
        assert [b.text() for b in (p.btn_load_xray, p.btn_usb, p.btn_plane, p.btn_save)] == [
            "Load X-ray", "From folder (USB)", "Set reference plane", "Snapshot"]
        assert not p.btn_lateral.isVisible()
        # single, "big": X-ray view 1 square on the right, the video beside it, the area below wide
        g1, gl, gb = p.panel_view1.geometry(), p.panel_live.geometry(), p.bottom.geometry()
        assert g1.right() == p.width() - 1 and gl.left() == 0 and gl.right() + 1 == g1.left()
        assert gb.width() == p.width() and gb.top() == g1.bottom() + 1
        assert page._overlay_cache is not None and p.panel_live._box_overlay is not None

        # tracking: the marks are built in the tracking thread
        p.btn_track.click()
        assert page._tracking_active and page._track_worker._marks_builder is page._builder
        assert _spin(app, 5.0, lambda: page._drawn_out is not None and page._panel_marks is not None), \
            "no marks"
        m = page._panel_marks
        assert m.xray is not None and m.camera.green_tip is not None
        assert _spin(app, 2.0, lambda: p.tile_off_plane._value != "–"
                     and "Tracking active" in p.lbl_status.text()), p.lbl_status.text()
        assert _spin(app, 2.0, lambda: p.panel_live._marks is not None)
        assert p.panel_view1._xm is not None
        assert _is_magenta(_qimage_to_rgb(p.panel_view1.grab().toImage())).any()

        # Set reference plane: pointer search, SPACE without a question, tracking back by itself
        page.start_plane_only()
        assert page._plane_acquiring and not page._tracking_active and page._resume_tracking
        assert p.btn_plane.text() == "Cancel" and not p.btn_track.isEnabled()
        assert _spin(app, 1.0, lambda: p.lbl_status.colour().rgb() == STATUS_BAD.rgb())
        camera_frame = page.camera.bgra
        page.camera.bgra = pointer_frame
        # the recorded scene has a pointer too: the tip filter swings over from it (a jump of
        # 130 mm, ~3 s) - as a user would, SPACE once the shown d_x has settled
        assert _spin(app, 8.0, lambda: page._plane_d_x_mm is not None
                     and abs(page._plane_d_x_mm - d_recorded) < 0.05), page._plane_d_x_mm
        assert _spin(app, 1.0, lambda: "pointer found" in p.lbl_status.text()
                     and p.lbl_status.colour().rgb() == STATUS_GOOD.rgb()), p.lbl_status.text()
        QTest.keyClick(page, Qt.Key_Space)
        page.camera.bgra = camera_frame
        d_new = page.state.d_x
        assert abs(d_new - d_recorded) < 0.5, (d_new, d_recorded)
        assert not page._plane_acquiring and p.btn_plane.text() == "Set reference plane"
        assert "Plane set" in p.lbl_status.text() and "same X-ray" in p.lbl_status.text()
        assert page._tracking_active and page._builder._cfg.d_x_mm == d_new
        assert _spin(app, 5.0, lambda: page._panel_marks is not None), "no marks after the plane"

        page.start_plane_only()
        QTest.keyClick(page, Qt.Key_Escape)
        assert not page._plane_acquiring and page.state.d_x == d_new and page._tracking_active
        assert "Plane unchanged" in p.lbl_status.text()

        # Load X-ray: an image that came first waits for the button
        new1, new2 = tmp / "new1.png", tmp / "new2.png"
        cv2.imwrite(str(new1), cv2.flip(rec.raw, 0))
        cv2.imwrite(str(new2), cv2.flip(rec.raw, -1))
        working = page._xray_working_u8
        assert page.accept_received_xray(str(new1))
        assert page._xray_pending and page._xray_working_u8 is working
        assert "X-ray received" in p.lbl_status.text() and p.lbl_status.colour().rgb() == STATUS_WARN.rgb()
        p.btn_load_xray.click()
        assert page._xray_working_u8 is not working and page.state.xray_image_anatomy_path == str(new1)
        assert page._xray_pending is None
        # ... and the pointer search for its plane starts by itself (11.11.); ESC keeps the plane
        assert page._plane_acquiring and not page._tracking_active and page._resume_tracking
        assert p.btn_plane.text() == "Cancel" and "New X-ray" in p.lbl_status.text()
        assert "ESC keep plane" in p.lbl_status.text()
        QTest.keyClick(page, Qt.Key_Escape)
        assert not page._plane_acquiring and page.state.d_x == d_new and page._tracking_active
        assert "plane kept" in p.lbl_status.text() and page.state.xray_image_anatomy_path == str(new1)

        # ... the button first: the next C-arm image comes in at once
        p.btn_load_xray.click()
        assert page._xray_waiting and p.btn_load_xray.text() == "Cancel"
        assert page.accept_received_xray(str(new2))
        assert not page._xray_waiting and p.btn_load_xray.text() == "Load X-ray"
        assert page.state.xray_image_anatomy_path == str(new2) and page._plane_acquiring
        QTest.keyClick(page, Qt.Key_Escape)
        assert not page._plane_acquiring and page._tracking_active
        p.btn_load_xray.click()
        assert page.accept_received_xray(str(new2))        # the same image again: nothing changes
        assert "Same X-ray" in p.lbl_status.text() and page.state.xray_image_anatomy_path == str(new2)
        assert not page._plane_acquiring                  # the same image is no new plane
        p.btn_load_xray.click()
        QTest.keyClick(page, Qt.Key_Escape)
        assert not page._xray_waiting and "cancelled" in p.lbl_status.text()

        # From folder (USB): the pointer search starts here too
        page._ask_xray_path = lambda *a, **k: str(new1)
        p.btn_usb.click()
        assert page.state.xray_image_anatomy_path == str(new1) and page._plane_acquiring
        assert list(tmp.glob("session/*xray*")), "the new X-rays are not in the session"

        # Evaluation: the new X-ray and the plane set after it are one plane, one label
        # (the rail's plane button saved both under the label it opened)
        label = ev.sanitize_label(ev.label)
        assert list(tmp.glob(f"session/{label}_xray*")), list(tmp.glob("session/*"))
        page.camera.bgra = pointer_frame
        assert _spin(app, 8.0, lambda: page._plane_d_x_mm is not None
                     and abs(page._plane_d_x_mm - d_recorded) < 0.05), page._plane_d_x_mm
        assert _spin(app, 1.0, lambda: "New X-ray: pointer found" in p.lbl_status.text()), p.lbl_status.text()
        QTest.keyClick(page, Qt.Key_Space)
        page.camera.bgra = camera_frame
        assert "Plane set" in p.lbl_status.text() and "new X-ray" in p.lbl_status.text()
        assert ev.sanitize_label(ev.label) == label and (tmp / "session" / f"{label}_depth.npz").exists(), \
            (ev.label, list(tmp.glob("session/*depth*")))
        assert _spin(app, 5.0, lambda: page._tracking_active and page._panel_marks is not None
                     and page._panel_marks.xray is not None)

        # snapshot: the status line, no box (tracking came back after the plane)
        assert _spin(app, 5.0, lambda: page._tracking_active and page._drawn_out is not None), "no pose"
        n_info = len(info)
        p.btn_save.click()
        assert _spin(app, 10.0, lambda: p.btn_save.isEnabled()
                     and "Snapshot saved" in p.lbl_status.text()), p.lbl_status.text()
        assert len(info) == n_info and not errors, (info, errors)
        snap = next(tmp.glob("session/*_live*"))
        assert (snap / "04_rgb_overlay_drill_xrayview.png").exists()

        p.btn_track.click()
        page.on_leave()
        _spin(app, 2.0, lambda: not any(t.name == "frame-log-writer" for t in xv._threads()))
        assert not errors, errors
        page.close()
    finally:
        QMessageBox.critical, QMessageBox.information, QMessageBox.question = boxes
        shutil.rmtree(tmp, ignore_errors=True)
    print("  (l) real page, panels (single): marks from the tracking thread, read-outs, plane with SPACE "
          "and tracking back, Load X-ray (pending, armed, same image, ESC) and USB start the pointer search, "
          "snapshot in the status line")


# ------------------------------------------------------------ main

def main() -> int:
    tests = (
        test_camera_model, test_panel_map, test_clip_and_ruler_synthetic,
        test_same_as_chat1, test_ruler_real, test_overlay_in_roi, test_camera_panel_direct,
        test_xray_panel, test_design_window,
        test_view_at_start, test_readout_tile, test_readout_int, test_plane_side_view,
        test_page_plane_only, test_page_panels,
    )
    skipped = 0
    for test in tests:
        try:
            test()
        except Skip as s:
            skipped += 1
            print(f"  SKIPPED {test.__name__}: {s}")
    print("all checks passed" + (f" ({skipped} skipped: data missing)" if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
