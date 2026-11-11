# overlay/tools/tool_marks.py
"""Everything the panels draw of one tool pose, computed once per pose. No Qt.

    MarkBuilder.build(points)   tracking thread, per result: tip + back point
                                (camera frame, mm) -> ToolMarks, ready to draw
    panels                      GUI thread, per shown frame: only draw

The GUI thread then does no geometry at all. Everything here runs on plain
floats: a dozen small numpy calls cost 0.4 ms next to the running tracker, a
factor five above an offline loop (overlay.tools.xray_view). configure() swaps
one immutable tuple, so the GUI thread can change the settings while the
tracking thread builds; a build sees either the old or the new ones.

Panel pixels. Marks are given in device pixels of the panel's image area, in
QPainter's continuous coordinates: pixel (0, 0) covers [0, 1) x [0, 1), so an
image pixel with integer coordinate u lies at u + 0.5.

The views (Thesis 6.2.1, overlay.tools.xray_view):

    camera view, green    where the camera sees the tool
    X-ray view, magenta   the tool moved along its source rays onto Pi, seen by
                          the camera; only inside the X-ray field
    X-ray panel           the tool in the working image, uv_x = K_x P_x / z_x;
                          d_x does not enter

The ruler is real in 3D: lengths along the tracked axis, in mm, never in the image.
With a the unit vector tip - back in the X-ray frame, a point s mm ahead of the tip is
P(s) = P_tip + s a, and in the image

    uv(s) = uv_tip + mu(s) v,    mu(s) = s / (z_tip + s a_z),   v constant

so with the back point at s = -L (lam = 1 on the line tip -> back):

    lam(s) = mu(s) / mu(-L)      s(lam) = mu z_tip / (1 - mu a_z),  mu = lam mu(-L)

An inclined axis therefore gets closer ticks by itself. Ruler zero:

    "field" where the axis enters the X-ray field. A field edge with the image line
            l (l^T (u, v, 1) = 0) is a plane through the source with the normal
            K_x^T l, so the axis meets it at P(s_E), s_E = -(l^T K_x P_tip) / (l^T K_x a).
            The tip reads depth = -s_E; negative before it has entered.
    "set"   at the point of "Null setzen", carried along the axis: depth = (P_tip - P_0) . a

Ticks stand at s_zero + k 5 mm, k = 0, 1, ..., with s_zero = -depth. d_x enters neither.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple, Optional

import numpy as np

from overlay.tools.tracking_overlay import TrackingOverlay
from overlay.tools.xray_view import XrayField, XrayView

RULER_FIELD = "field"
RULER_SET = "set"
RULER_MODES = (RULER_FIELD, RULER_SET)

ORIENT_CAMERA = "camera"     # working image flipped vertically: as the camera sees it
ORIENT_C_ARM = "c_arm"       # the raw image: rotated 180 degrees against the camera
ORIENTATIONS = (ORIENT_CAMERA, ORIENT_C_ARM)

# Ring around the tip in the X-ray panel: its radius in mm at the tip's depth,
# so it also shows the tolerance of 2 mm (KONZEPT_UserStudy_DHS.md 4.2).
TIP_RING_MM = 2.0
TICK_STEP_MM = 5.0
_MAX_TICKS = 160
_AXIS_LIMIT_MM = 1000.0
# A field entry farther from the tip than this is no reading: the backward extension
# of an axis that points away from the field also crosses it. Field ~122 mm on Pi.
_MAX_DEPTH_MM = 300.0

_LABELS = [str(i) for i in range(0, 1001, 5)]


class CameraModel:
    """cv2.projectPoints of one point on plain floats, ~5 x cheaper next to the
    tracker. Same model: rational + thin prism (no tilt, no skew, as OpenCV)."""

    def __init__(self, K, dist=None) -> None:
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.fx, self.fy, self.cx, self.cy = (
            float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        )
        k = [0.0] * 14
        if dist is not None:
            for i, c in enumerate(np.asarray(dist, dtype=np.float64).ravel().tolist()):
                k[i] = c
        if k[12] != 0.0 or k[13] != 0.0:
            raise ValueError("tilted sensor model not supported")
        self.k = tuple(k)

    def project(self, X: float, Y: float, Z: float) -> tuple[float, float]:
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, _, _ = self.k
        x, y = X / Z, Y / Z
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        a1 = 2.0 * x * y
        a2 = r2 + 2.0 * x * x
        a3 = r2 + 2.0 * y * y
        c = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6)
        xd = x * c + p1 * a1 + p2 * a2 + s1 * r2 + s2 * r4
        yd = y * c + p1 * a3 + p2 * a1 + s3 * r2 + s4 * r4
        return self.fx * xd + self.cx, self.fy * yd + self.cy


class PanelMap(NamedTuple):
    """Working-image pixel -> panel pixel: x = a u + b v + c, y = d u + e v + f.
    A scale times a whole-numbered turn, so the image is resampled only once."""

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    scale: float

    @classmethod
    def of(cls, orientation: str, image_wh, side_px: int) -> "PanelMap":
        w, h = float(image_wh[0]), float(image_wh[1])
        s = float(side_px) / max(w, h)
        if orientation == ORIENT_CAMERA:
            return cls(s, 0.0, 0.5 * s, 0.0, -s, (h - 0.5) * s, s)
        if orientation == ORIENT_C_ARM:
            return cls(-s, 0.0, (w - 0.5) * s, 0.0, s, 0.5 * s, s)
        raise ValueError(f"unknown orientation {orientation!r}")

    def apply(self, u: float, v: float) -> tuple[float, float]:
        return self.a * u + self.b * v + self.c, self.d * u + self.e * v + self.f


class CameraMarks:
    """Panel 1. Lines are what line_factory made of (x0, y0, x1, y1)."""

    __slots__ = ("green_lines", "green_tip", "magenta_lines", "magenta_tip",
                 "magenta_tip_near_edge")

    def __init__(self, green_lines, green_tip, magenta_lines, magenta_tip, near_edge) -> None:
        self.green_lines = green_lines
        self.green_tip = green_tip
        self.magenta_lines = magenta_lines
        self.magenta_tip = magenta_tip
        self.magenta_tip_near_edge = near_edge


class XrayMarks:
    """An X-ray panel. bbox: (x0, y0, x1, y1) of everything drawn."""

    __slots__ = ("lines", "ticks_minor", "ticks_major", "labels", "tip", "tip_radius", "bbox")

    def __init__(self, lines, ticks_minor, ticks_major, labels, tip, tip_radius, bbox) -> None:
        self.lines = lines
        self.ticks_minor = ticks_minor
        self.ticks_major = ticks_major
        self.labels = labels
        self.tip = tip
        self.tip_radius = tip_radius
        self.bbox = bbox


class ToolMarks:
    """One pose, ready to draw and to read off. xray is None without an X-ray
    or with the tool at or behind the source."""

    __slots__ = ("t_pose", "marked", "kind", "tip_cam", "forward_cam", "camera", "xray",
                 "off_plane_mm", "inclination_deg", "depth_mm", "tip_in_field", "rise_deg")

    def __init__(self, t_pose, marked, kind, tip_cam, forward_cam, camera, xray,
                 off_plane_mm, inclination_deg, depth_mm, tip_in_field, rise_deg=None) -> None:
        # rise_deg: inclination with its sign, positive when the drill behind the tip lies
        # farther from the source than the tip (the shaft rises away from Pi); |rise| = inclination
        self.rise_deg = rise_deg
        self.t_pose = t_pose
        self.marked = marked
        self.kind = kind
        self.tip_cam = tip_cam
        self.forward_cam = forward_cam
        self.camera = camera
        self.xray = xray
        self.off_plane_mm = off_plane_mm
        self.inclination_deg = inclination_deg
        self.depth_mm = depth_mm
        self.tip_in_field = tip_in_field


class _Config(NamedTuple):
    camera: CameraModel
    roi: Optional[tuple]        # x0, y0, w, h of panel 1 in camera pixels, None: no video
    view: Optional[XrayView]
    field: Optional[XrayField]
    R_T: tuple                  # R^T of T_xc, rows, plain floats
    t_mm: tuple
    pixel: Optional[Callable]
    fx_x: float                 # K_x[0, 0]
    plane: tuple                # plane_matrix(d_x), rows
    d_x_mm: float
    near_edge_px: float         # working pixels: the glyph of panel 1 may reach the edge
    xmap: Optional[PanelMap]
    xray_mark_scale: float
    ruler_mode: str
    zero_cam: Optional[tuple]
    line_factory: Callable


def _as_is(*line):
    return line


def _s_of_lam(lam: float, z0: float, z1: float, az: float, length: float) -> float:
    """mm along the axis of the point at lam on the image line tip -> back; beyond a
    vanishing point the limit on that side."""
    mu = lam / (-z1 / length)       # mu(-L) = -L / z_back
    den = 1.0 - mu * az
    if den <= 1e-9:
        return _AXIS_LIMIT_MM if lam < 0.0 else -_AXIS_LIMIT_MM
    return max(-_AXIS_LIMIT_MM, min(_AXIS_LIMIT_MM, mu * z0 / den))


def clip_ray_to_rect(x0, y0, dx, dy, w, h):
    """The part of the ray (x0, y0) + t (dx, dy), t >= 0, inside [0, w] x [0, h]:
    (t_in, t_out) or None."""
    t0, t1 = 0.0, math.inf
    for p, q in ((-dx, x0), (dx, w - x0), (-dy, y0), (dy, h - y0)):
        if p == 0.0:
            if q < 0.0:
                return None
        else:
            r = q / p
            if p < 0.0:
                if r > t0:
                    t0 = r
            elif r < t1:
                t1 = r
    return (t0, t1) if t0 <= t1 and t1 < math.inf else None


class MarkBuilder:
    """Turns poses into ToolMarks. build() may run in another thread than
    configure(); one thread only for build() (the label side has hysteresis)."""

    AXIS_BACK_MM = TrackingOverlay.AXIS_BACK_MM
    DASH_PIECES = TrackingOverlay.DASH_PIECES
    TIP_GLYPH_REACH_PX = TrackingOverlay.TIP_GLYPH_REACH_PX

    def __init__(self, K_c, dist_c, line_factory: Callable = _as_is) -> None:
        self._cfg = _Config(
            camera=CameraModel(K_c, dist_c), roi=(0, 0, 1920, 1080),
            view=None, field=None, R_T=(), t_mm=(), pixel=None, fx_x=0.0, plane=(), d_x_mm=0.0,
            near_edge_px=0.0, xmap=None, xray_mark_scale=1.0, ruler_mode=RULER_FIELD, zero_cam=None,
            line_factory=line_factory,
        )
        self._label_side = 1.0

    # ------------------------------------------------------------------
    # Settings, GUI thread
    # ------------------------------------------------------------------

    def set_camera(self, K_c, dist_c) -> None:
        self._cfg = self._cfg._replace(camera=CameraModel(K_c, dist_c))

    def set_camera_panel(self, roi) -> None:
        """The crop of panel 1 in camera pixels; None: the video is not shown, no
        camera marks are computed."""
        self._cfg = self._cfg._replace(roi=None if roi is None else tuple(int(v) for v in roi))

    def set_xray(self, view: Optional[XrayView], field: Optional[XrayField] = None,
                 d_x_mm: float = 0.0, working_wh=(1024, 1024)) -> None:
        if view is None:
            self._cfg = self._cfg._replace(view=None, field=None, pixel=None, xmap=None)
            return
        if field is None:
            raise ValueError("an X-ray view needs its field")
        self._cfg = self._cfg._replace(
            view=view, field=field,
            R_T=tuple(view.R.T.ravel().tolist()), t_mm=tuple(view.t_mm.tolist()),
            pixel=view.pixel, fx_x=float(view.K_x[0, 0]),
        )
        self.set_plane_depth(d_x_mm, working_wh)

    def set_plane_depth(self, d_x_mm: float, working_wh=(1024, 1024)) -> None:
        cfg = self._cfg
        if cfg.view is None:
            return
        self._cfg = cfg._replace(
            d_x_mm=float(d_x_mm),
            plane=tuple(cfg.view.plane_matrix(d_x_mm).ravel().tolist()),
            near_edge_px=self.TIP_GLYPH_REACH_PX / self._camera_per_working(cfg, d_x_mm, working_wh),
        )

    def _camera_per_working(self, cfg: _Config, d_x_mm: float, working_wh) -> float:
        """Camera pixels per working pixel on Pi; the plane is tilted against the
        camera, the smaller of both directions decides."""
        cx, cy = float(working_wh[0]) / 2.0, float(working_wh[1]) / 2.0
        step = 100.0
        p = cfg.view.pixels_on_plane_cam([[cx, cy], [cx + step, cy], [cx, cy + step]], d_x_mm)
        a = [cfg.camera.project(*q) for q in p.tolist()]
        return min(math.hypot(a[1][0] - a[0][0], a[1][1] - a[0][1]),
                   math.hypot(a[2][0] - a[0][0], a[2][1] - a[0][1])) / step

    def set_xray_panel(self, panel_map: Optional[PanelMap], mark_scale: float = 1.0) -> None:
        """mark_scale: device pixels per pixel of the tick and label sizes."""
        self._cfg = self._cfg._replace(xmap=panel_map, xray_mark_scale=float(mark_scale))

    def set_ruler(self, mode: str, zero_cam=None) -> None:
        if mode not in RULER_MODES:
            raise ValueError(f"unknown ruler mode {mode!r}")
        zero = None if zero_cam is None else tuple(float(v) for v in zero_cam)
        self._cfg = self._cfg._replace(ruler_mode=mode, zero_cam=zero)

    @property
    def ruler_mode(self) -> str:
        return self._cfg.ruler_mode

    @property
    def zero_cam(self) -> Optional[tuple]:
        return self._cfg.zero_cam

    def field_outline_on_camera(self) -> list[tuple[float, float]]:
        """The field as panel 1 shows it, in panel pixels; for clipping the tip
        glyph at the edge. Once per X-ray, d_x and panel size."""
        cfg = self._cfg
        if cfg.view is None or cfg.roi is None:
            return []
        x0, y0 = cfg.roi[0], cfg.roi[1]
        out = []
        for q in cfg.view.pixels_on_plane_cam(cfg.field.hull, cfg.d_x_mm).tolist():
            u, v = cfg.camera.project(*q)
            out.append((u - x0 + 0.5, v - y0 + 0.5))
        return out

    # ------------------------------------------------------------------
    # Per pose, tracking thread
    # ------------------------------------------------------------------

    def build(self, points, *, marked: bool = False, t_pose: float = 0.0,
              kind: Optional[str] = None) -> Optional[ToolMarks]:
        """points: rows tip, back point in the camera frame, mm (ToolPoints)."""
        cfg = self._cfg
        if hasattr(points, "tolist"):
            points = points.tolist()
        (tx, ty, tz), (bx, by, bz) = points
        if tz <= 0.0 or bz <= 0.0:
            return None

        fx, fy, fz = tx - bx, ty - by, tz - bz
        length = math.sqrt(fx * fx + fy * fy + fz * fz)
        if length <= 0.0:
            return None
        fx, fy, fz = fx / length, fy / length, fz / length

        lf = cfg.line_factory
        video = cfg.roi is not None
        green_lines, green_tip = self._green(cfg, tx, ty, tz, bx, by, bz) if video else ([], None)

        depth = None
        if cfg.ruler_mode == RULER_SET and cfg.zero_cam is not None:
            zx, zy, zz = cfg.zero_cam
            depth = (tx - zx) * fx + (ty - zy) * fy + (tz - zz) * fz

        off_plane = inclination = rise = None
        magenta_lines, magenta_tip, near_edge = [], None, False
        xray = None
        tip_in_field = False
        if cfg.view is not None:
            r = cfg.R_T
            t0, t1, t2 = cfg.t_mm
            # P_x = R^T (P_c - t)
            dx, dy, dz = tx - t0, ty - t1, tz - t2
            x0 = r[0] * dx + r[1] * dy + r[2] * dz
            y0 = r[3] * dx + r[4] * dy + r[5] * dz
            z0 = r[6] * dx + r[7] * dy + r[8] * dz
            dx, dy, dz = bx - t0, by - t1, bz - t2
            x1 = r[0] * dx + r[1] * dy + r[2] * dz
            y1 = r[3] * dx + r[4] * dy + r[5] * dz
            z1 = r[6] * dx + r[7] * dy + r[8] * dz
            off_plane = z0 - cfg.d_x_mm
            rise = math.degrees(math.asin(max(-1.0, min(1.0, (z1 - z0) / length))))
            inclination = abs(rise)
            if z0 > 0.0 and z1 > 0.0:
                g = self._axis_in_image(cfg, x0, y0, z0, x1, y1, z1)
                tip_in_field = g[7]
                az = (z0 - z1) / length
                if cfg.ruler_mode == RULER_FIELD and g[5] is not None:
                    # lam_out: where the axis leaves the field on the drill's side
                    s_entry = _s_of_lam(g[5], z0, z1, az, length)
                    if abs(s_entry) <= _MAX_DEPTH_MM:
                        depth = -s_entry
                if video:
                    magenta_lines, magenta_tip, near_edge = self._magenta(cfg, g)
                if cfg.xmap is not None:
                    xray = self._xray_panel(cfg, g, z0, z1, az, length, depth)

        return ToolMarks(
            t_pose, bool(marked), kind, (tx, ty, tz), (fx, fy, fz),
            CameraMarks([lf(*q) for q in green_lines], green_tip,
                        [lf(*q) for q in magenta_lines], magenta_tip, near_edge),
            xray, off_plane, inclination, depth, tip_in_field, rise,
        )

    def _green(self, cfg, tx, ty, tz, bx, by, bz):
        cam = cfg.camera
        x0r, y0r, w, h = cfg.roi
        u, v = cam.project(tx, ty, tz)
        x, y = u - x0r + 0.5, v - y0r + 0.5
        u, v = cam.project(bx, by, bz)
        xb, yb = u - x0r + 0.5, v - y0r + 0.5
        dx, dy = xb - x, yb - y

        lines = []
        pieces = self.DASH_PIECES
        for i in range(0, pieces, 2):
            a, b = i / pieces, (i + 1) / pieces
            lines.append((x + a * dx, y + a * dy, x + b * dx, y + b * dy))
        if dx != 0.0 or dy != 0.0:
            # forward, away from the back point, to the panel edge
            cut = clip_ray_to_rect(x, y, -dx, -dy, float(w), float(h))
            if cut is not None:
                a, b = cut
                lines.append((x - a * dx, y - a * dy, x - b * dx, y - b * dy))
        return lines, (x, y)

    def _axis_in_image(self, cfg, x0, y0, z0, x1, y1, z1):
        """The axis on the line tip + lam (back - tip) of the working image, lam < 0
        is forward. One clip for everything on it, as TrackingOverlay."""
        u0, v0 = cfg.pixel(x0, y0, z0)
        u1, v1 = cfg.pixel(x1, y1, z1)
        du, dv = u1 - u0, v1 - v0
        dd = du * du + dv * dv
        lam_in, lam_out, lam_forward = None, None, None
        if dd > 1e-12:
            inside = cfg.field.clip_line((u0, v0), (u1, v1))
            tip_in = inside is not None and inside[0] <= 0.0 <= inside[1]
            if inside is not None:
                lam_in, lam_out = inside
                lam_forward = lam_in
                # an axis running away from the source ends at its vanishing point
                if z0 - z1 > 1e-9:
                    u, v = cfg.pixel(x0 - x1, y0 - y1, z0 - z1)
                    lam_forward = max(lam_in, ((u - u0) * du + (v - v0) * dv) / dd)
        else:
            tip_in = cfg.field.contains((u0, v0))
        return u0, v0, du, dv, lam_in, lam_out, lam_forward, tip_in

    def _magenta(self, cfg, g):
        u0, v0, du, dv, lam_in, lam_out, lam_forward, tip_in = g
        cam = cfg.camera
        rx, ry = cfg.roi[0], cfg.roi[1]
        m0, m1, m2, m3, m4, m5, m6, m7, m8 = cfg.plane
        t0, t1, t2 = cfg.t_mm

        def panel(lam):
            u, v = u0 + lam * du, v0 + lam * dv
            a, b = cam.project(m0 * u + m1 * v + m2 + t0,
                               m3 * u + m4 * v + m5 + t1,
                               m6 * u + m7 * v + m8 + t2)
            return a - rx + 0.5, b - ry + 0.5

        lines = []
        if lam_in is not None:
            end = min(lam_out, 0.0)
            if lam_forward < end:
                lines.append(panel(lam_forward) + panel(end))
            pieces = self.DASH_PIECES
            for i in range(0, pieces, 2):
                a, b = max(lam_in, i / pieces), min(lam_out, (i + 1) / pieces)
                if a < b:
                    lines.append(panel(a) + panel(b))
        tip = panel(0.0) if tip_in else None
        near_edge = tip_in and cfg.field.inner_distance(u0, v0) <= cfg.near_edge_px
        return lines, tip, near_edge

    def _xray_panel(self, cfg, g, z0, z1, az, length, depth) -> XrayMarks:
        u0, v0, du, dv, lam_in, lam_out, lam_forward, tip_in = g
        a, b, c, d, e, f, scale = cfg.xmap
        k = cfg.xray_mark_scale
        lf = cfg.line_factory

        xt, yt = a * u0 + b * v0 + c, d * u0 + e * v0 + f
        radius = TIP_RING_MM * cfg.fx_x / z0 * scale
        glyph = radius + 8.0 * k
        xs, ys = [xt - glyph, xt + glyph], [yt - glyph, yt + glyph]

        lines = []
        minor, major, labels = [], [], []
        margin = 4.0 * k
        if lam_in is not None:
            end = min(lam_out, 0.0)
            segs = []
            if lam_forward < end:
                segs.append((lam_forward, end))
            pieces = self.DASH_PIECES
            for i in range(0, pieces, 2):
                p, q = max(lam_in, i / pieces), min(lam_out, (i + 1) / pieces)
                if p < q:
                    segs.append((p, q))
            for p, q in segs:
                u, v = u0 + p * du, v0 + p * dv
                xa, ya = a * u + b * v + c, d * u + e * v + f
                u, v = u0 + q * du, v0 + q * dv
                xb, yb = a * u + b * v + c, d * u + e * v + f
                lines.append((xa, ya, xb, yb))
                xs += (xa, xb)
                ys += (ya, yb)

            # forward direction in the panel: -(du, dv) through the linear part
            fx, fy = -(a * du + b * dv), -(d * du + e * dv)
            n = math.hypot(fx, fy)
            if n > 1e-9:
                fx, fy = fx / n, fy / n
                nx, ny = fy, -fx
                # labels on the upper side; turn only well past the horizontal
                # normal, or the jitter of an upright axis would flip them
                if self._label_side * ny > 0.3:
                    self._label_side = -self._label_side
                nx, ny = nx * self._label_side, ny * self._label_side
                self._ruler(cfg, g, z0, z1, az, length, depth, k, nx, ny,
                            minor, major, labels, xs, ys)

        bbox = (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)
        return XrayMarks(
            [lf(*q) for q in lines], [lf(*q) for q in minor], [lf(*q) for q in major],
            labels, (xt, yt) if tip_in else None, radius, bbox,
        )

    def _ruler(self, cfg, g, z0, z1, az, length, depth, k, nx, ny,
               minor, major, labels, xs, ys) -> None:
        if depth is None:
            return
        u0, v0, du, dv, lam_in, lam_out, lam_forward, _ = g
        a, b, c, d, e, f, _scale = cfg.xmap
        lam_per_mu = -z1 / length       # lam = mu / mu(-L), mu(-L) = -L / z_back

        s_back = _s_of_lam(lam_out, z0, z1, az, length)         # mm, smallest visible s
        s_front = _s_of_lam(lam_forward, z0, z1, az, length)    # largest

        s_zero = -depth
        step = TICK_STEP_MM
        first = max(0, int(math.ceil((s_back - s_zero) / step - 1e-9)))

        # how dense do the ticks get here?
        def x_y(s: float):
            w = z0 + s * az
            if w <= 1e-6:
                return None
            lam = s / w * lam_per_mu
            u, v = u0 + lam * du, v0 + lam * dv
            return a * u + b * v + c, d * u + e * v + f

        s_ref = s_zero + first * step
        p, q = x_y(s_ref), x_y(s_ref + step)
        if p is None or q is None:
            return
        spacing = math.hypot(q[0] - p[0], q[1] - p[1]) / k
        with_minor = spacing >= 4.0
        label_every = 2 if spacing >= 11.0 else (4 if spacing >= 5.5 else 10)

        hm, hM, off = 4.0 * k, 7.0 * k, 17.0 * k
        half_w, half_h = 16.0 * k, 11.0 * k     # of a label with three digits
        last = None
        for i in range(first, first + _MAX_TICKS):
            s = s_zero + i * step
            if s > s_front + 1e-9:
                break
            w = z0 + s * az
            if w <= 1e-6:
                break
            lam = s / w * lam_per_mu
            u, v = u0 + lam * du, v0 + lam * dv
            x, y = a * u + b * v + c, d * u + e * v + f
            # towards a vanishing point the ticks run into each other
            if last is not None and abs(x - last[0]) + abs(y - last[1]) < 1.5 * k:
                break
            last = (x, y)
            if i % 2 == 0:
                x0, y0, x1, y1 = x - nx * hM, y - ny * hM, x + nx * hM, y + ny * hM
                major.append((x0, y0, x1, y1))
                xs += (x0, x1)
                ys += (y0, y1)
                if i % label_every == 0:
                    xl, yl = x + nx * off, y + ny * off
                    labels.append((xl, yl, _LABELS[i] if i < len(_LABELS) else str(i * 5)))
                    xs += (xl - half_w, xl + half_w)
                    ys += (yl - half_h, yl + half_h)
            elif with_minor:
                x0, y0, x1, y1 = x - nx * hm, y - ny * hm, x + nx * hm, y + ny * hm
                minor.append((x0, y0, x1, y1))
                xs += (x0, x1)
                ys += (y0, y1)
