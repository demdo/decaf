# overlay/tools/tracking_overlay.py
"""In-place drawing of tracked-tool info into the live frame.

The single place for everything drawn into the live image from a
TrackerResult — future additions (uncertainty, chuck, status glyphs, ...)
belong here, not in the page. The same performance rules as the rest of
the live loop apply: draw directly into the frame buffer, touch only the
glyph pixels, no per-frame allocations beyond a few small arrays.

The camera-frame points come from ToolPoints, which runs in the tracking
thread on every result (overlay.tools.live_tracking.TrackingWorker); per
frame the GUI thread pays one cv2.projectPoints over two points and a
handful of cv2.line calls. Offline that is well under 0.1 ms, on the real
page more: 05.11.2026, page with a stubbed camera and engine, per shown frame
0.14 ms for drawing the camera view, 0.06 ms for drawing the X-ray view and
0.13 ms for its geometry (docs/user_study/LABORCHECK_Ziel1.md).

Two views of the same tool (Thesis 6.2.1):

    camera view, green    the points as the camera sees them
    X-ray view, magenta   the points moved along their source rays onto the
                          reference plane (overlay.tools.xray_view), so they
                          lie on the drill's shadow in the X-ray overlay

update() computes both, per fresh pose; draw_into(show_camera, show_xray)
draws them, green first, so what is on screen is the snapshot's layer 04.
The X-ray view exists only inside the X-ray field: its lines are clipped in
the working image, where the field is convex, and its tip glyph needs the tip
in the field. Without set_xray_view() (tracking test, no X-ray) there is none.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from overlay.tools.xray_view import XrayField, XrayView, axis_inclination_deg

_ZERO3 = np.zeros(3)


def dash_spans(length_px: float, on_px: float, off_px: float, lo: float = 0.0, hi: float = 1.0) -> list:
    """The dashes of a segment length_px long on screen, from its start (lam = 0, the
    tip) to its end (lam = 1): on_px drawn, off_px gap, as (lam_a, lam_b) pairs clipped
    to [lo, hi]. The same rhythm at any distance (user 11.11.2026)."""
    out = []
    if not length_px > 1e-9:
        return out
    on = on_px / length_px
    period = (on_px + off_px) / length_px
    if not period > 0.0:
        return out
    n = int(np.ceil(1.0 / period))
    for i in range(n):
        a, b = i * period, min(1.0, i * period + on)
        a, b = max(lo, a), min(hi, b)
        if a < b:
            out.append((a, b))
    return out


class TrackingOverlay:
    """Projects tip + drill axis of one tracked tool and draws them in place.

    update(points) per fresh pose (TrackOutput.points, from the ToolPoints
    of tool_points()), clear() on loss, draw_into(frame) per composed
    frame. p_tip_mm / axis_dir are marker-frame millimetres.
    """

    # Dashed shaft segment: from the tip toward the marker (the physical
    # drill body side), in millimetres.
    AXIS_BACK_MM = 50.0
    # Its dashes: a fixed length in camera pixels, the rhythm of the live panels
    # (6 / 4 px there times 1.44 on the laptop, one camera pixel per device pixel;
    # user 11.11.2026). TrackingOverlay(..., dash_pieces=n) instead splits it into
    # n pieces (odd -> starts and ends with a drawn dash), as the thesis' recordings
    # were drawn with n = DASH_PIECES.
    DASH_ON_PX = 9.0
    DASH_OFF_PX = 6.0
    DASH_PIECES = 7

    TIP_COLOR = (80, 220, 80, 255)   # BGRA
    AXIS_COLOR = (80, 220, 80, 255)
    # A pose kept under a partial occlusion (ToolPoints.marked): same glyphs
    # in orange.
    MARKED_COLOR = (0, 165, 255, 255)
    THICKNESS = 2
    TIP_RADIUS = 8
    TIP_CROSS = 20

    # X-ray view. Under a marked pose it is dimmed, not hidden: it would
    # otherwise flash on and off with partial occlusions, and the orange
    # camera view already gives the warning.
    XRAY_COLOR = (255, 0, 255, 255)
    XRAY_MARKED_COLOR = (150, 60, 150, 255)
    # Clipping keeps the line centres this many camera pixels inside the field:
    # half the line, its anti-aliased edge, the rounding to whole pixels and
    # the nearest-neighbour warp of the mask.
    FIELD_MARGIN_PX = 3.0
    # Largest distance of a tip-glyph pixel from the tip.
    TIP_GLYPH_REACH_PX = TIP_CROSS // 2 + THICKNESS

    def __init__(self, K, dist, p_tip_mm, axis_dir, dash_pieces: Optional[int] = None) -> None:
        # None: dashes of DASH_ON_PX / DASH_OFF_PX; n: n pieces (the thesis' recordings)
        self.dash_pieces = None if dash_pieces is None else int(dash_pieces)
        self._K = np.asarray(K, dtype=np.float64)
        self._dist = None if dist is None else np.asarray(dist, dtype=np.float64)

        self._p_tip = np.asarray(p_tip_mm, dtype=np.float64).reshape(3)
        axis = np.asarray(axis_dir, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(axis))
        if norm <= 0.0:
            raise ValueError("axis_dir must be non-zero")
        self._axis = axis / norm
        # Marker-side end of the dashed shaft segment.
        self._p_back = self._p_tip + self.AXIS_BACK_MM * self._axis

        self._tip_uv: Optional[np.ndarray] = None
        self._back_uv: Optional[np.ndarray] = None
        # The camera-frame points behind the pixels (mm), kept for the
        # snapshot record; not needed for drawing.
        self._tip_cam: Optional[np.ndarray] = None
        self._back_cam: Optional[np.ndarray] = None
        self._marked = False
        # Line-of-sight shift of the points update() gets: that of the last
        # ToolPoints made by tool_points(), else the class default.
        self._los_offset_mm = float(ToolPoints.LOS_OFFSET_MM)

        # X-ray view: set_xray_view() / set_plane_depth(); _xv is the geometry
        # of the current pose (None: nothing to draw).
        self._xray: Optional[XrayView] = None
        self._field: Optional[XrayField] = None
        self._d_x_mm = 0.0
        self._field_cam: Optional[np.ndarray] = None    # (N, 1, 2) float32
        self._plane_M_T: Optional[np.ndarray] = None    # view.plane_matrix(d_x).T
        self._xv: Optional[dict] = None
        # time the last update() spent on the X-ray view (the page's profile)
        self.xray_update_s = 0.0

    def tool_points(self, los_offset_mm: Optional[float] = None,
                    mark_below_visible_frac: float = 0.0) -> "ToolPoints":
        """A ToolPoints of this tool's tip and back point, for the tracking
        thread; los_offset_mm None: ToolPoints.LOS_OFFSET_MM (the page's);
        mark_below_visible_frac: see ToolPoints.marked. Its line-of-sight
        shift is the one snapshot_record reports from now on."""
        tp = ToolPoints(self._p_tip, self._p_back, los_offset_mm=los_offset_mm,
                        mark_below_visible_frac=mark_below_visible_frac)
        self._los_offset_mm = float(tp.los_offset_mm)
        return tp

    # ------------------------------------------------------------------
    # Per-frame state
    # ------------------------------------------------------------------

    def update(self, points, marked: bool = False) -> None:
        """Recompute the 2D geometry from the camera-frame points of a fresh
        result (TrackOutput.points: rows tip and back point, mm); marked:
        draw in MARKED_COLOR (TrackOutput.points_marked)."""
        if points is None:
            self.clear()
            return
        tip_cam = np.asarray(points[0], dtype=np.float64).reshape(3)
        back_cam = np.asarray(points[1], dtype=np.float64).reshape(3)
        if (
            tip_cam[2] <= 0.0
            or back_cam[2] <= 0.0
        ):
            self.clear()
            return

        uv, _ = cv2.projectPoints(
            np.stack([tip_cam, back_cam]).reshape(-1, 1, 3),
            _ZERO3,
            _ZERO3,
            self._K,
            self._dist,
        )
        uv = uv.reshape(2, 2)
        if not np.isfinite(uv).all():
            self.clear()
            return
        self._tip_uv = uv[0]
        self._back_uv = uv[1]
        self._tip_cam = tip_cam
        self._back_cam = back_cam
        self._marked = bool(marked)
        self._update_xray()

    def clear(self) -> None:
        self._tip_uv = None
        self._back_uv = None
        self._tip_cam = None
        self._back_cam = None
        self._marked = False
        self._xv = None

    @property
    def has_pose(self) -> bool:
        """True while draw_into would actually draw (fresh projected pose)."""
        return self._tip_uv is not None

    @property
    def has_xray_view(self) -> bool:
        """True while there is an X-ray view of the current pose (it may still
        lie outside the field and draw nothing)."""
        return self._xv is not None

    # ------------------------------------------------------------------
    # X-ray view
    # ------------------------------------------------------------------

    def set_xray_view(self, view: Optional[XrayView], field: Optional[XrayField] = None,
                      d_x_mm: Optional[float] = None) -> None:
        """Set the X-ray and its plane depth; view None: no X-ray view. The
        field is XrayField.from_image(working image, field_inset_px(...))."""
        if view is not None and (field is None or d_x_mm is None):
            raise ValueError("an X-ray view needs its field and d_x")
        self._xray = view
        self._field = field if view is not None else None
        self._field_cam = None
        if view is None:
            self._xv = None
            return
        self.set_plane_depth(d_x_mm)

    def set_plane_depth(self, d_x_mm: float) -> None:
        """A new d_x: the current pose is moved at once (the slider)."""
        if self._xray is None:
            return
        self._d_x_mm = float(d_x_mm)
        self._plane_M_T = self._xray.plane_matrix(self._d_x_mm).T
        self._field_cam = self._project(
            self._xray.pixels_on_plane_cam(self._field.hull, self._d_x_mm)
        ).astype(np.float32).reshape(-1, 1, 2)
        self._update_xray()

    def field_inset_px(self, view: XrayView, working_wh, d_x_mm: float) -> float:
        """FIELD_MARGIN_PX in working-image pixels, for XrayField.from_image.
        The plane is tilted against the camera: the smaller scale decides."""
        cx, cy = float(working_wh[0]) / 2.0, float(working_wh[1]) / 2.0
        step = 100.0
        px = self._project(view.pixels_on_plane_cam(
            [[cx, cy], [cx + step, cy], [cx, cy + step]], d_x_mm
        ))
        cam_per_working = min(
            float(np.linalg.norm(px[1] - px[0])), float(np.linalg.norm(px[2] - px[0]))
        ) / step
        return self.FIELD_MARGIN_PX / cam_per_working

    def _project(self, points_cam: np.ndarray) -> np.ndarray:
        """(N, 2) observed camera pixels of camera-frame points."""
        uv, _ = cv2.projectPoints(
            np.asarray(points_cam, dtype=np.float64).reshape(-1, 1, 3),
            _ZERO3, _ZERO3, self._K, self._dist,
        )
        return uv.reshape(-1, 2)

    def _update_xray(self) -> None:
        t0 = time.perf_counter()
        self._xv = self._xray_geometry()
        self.xray_update_s = time.perf_counter() - t0

    def _xray_geometry(self) -> Optional[dict]:
        """Runs per pose in the GUI thread, hence plain floats and few calls
        (see overlay.tools.xray_view). The record's numbers: _xray_record."""
        view = self._xray
        if view is None or self._tip_cam is None:
            return None
        P_x = view.to_xray((self._tip_cam, self._back_cam)).tolist()
        (x0, y0, z0), (x1, y1, z1) = P_x
        if z0 <= 0.0 or z1 <= 0.0:
            return None     # at or behind the source: no shadow
        u0, v0 = view.pixel(x0, y0, z0)
        u1, v1 = view.pixel(x1, y1, z1)

        # Everything drawn lies on the line tip + s (back - tip) of the working
        # image, s < 0 is forward. One clip for all of it.
        du, dv = u1 - u0, v1 - v0
        dd = du * du + dv * dv
        ss = [0.0, 1.0]     # tip, back
        if dd > 1e-12:
            inside = self._field.clip_line((u0, v0), (u1, v1))
            tip_in_field = inside is not None and inside[0] <= 0.0 <= inside[1]
        else:
            inside = None   # the axis along the source ray: a point
            tip_in_field = self._field.contains((u0, v0))
        if inside is not None:
            s_in, s_out = inside
            # An axis running away from the source ends at its vanishing point.
            s_forward = s_in
            if z0 - z1 > 1e-9:
                u, v = view.pixel(x0 - x1, y0 - y1, z0 - z1)
                s_forward = max(s_in, ((u - u0) * du + (v - v0) * dv) / dd)
            end = min(s_out, 0.0)
            if s_forward < end:
                ss += [s_forward, end]
            length = 0.0
            if not self.dash_pieces:
                # the drill on the plane through the camera: nearly linear over its 50 mm
                q = np.array([(u0, v0, 1.0), (u0 + du, v0 + dv, 1.0)]) @ self._plane_M_T + view.t_mm
                pa, pb = self._project(q)
                length = float(np.hypot(*(pb - pa)))
            for a, b in self._dash_spans(length, s_in, s_out):
                ss += [a, b]

        S_c = np.array([(u0 + s * du, v0 + s * dv, 1.0) for s in ss]) @ self._plane_M_T + view.t_mm
        px = self._project(S_c).tolist()
        try:
            ends = [(int(round(u)), int(round(v))) for u, v in px]
        except (ValueError, OverflowError):
            return None     # not finite

        tip_near_edge = tip_in_field and cv2.pointPolygonTest(
            self._field_cam, px[0], True
        ) <= self.TIP_GLYPH_REACH_PX
        return {
            "P_x": P_x,
            "uv_x": ((u0, v0), (u1, v1)),
            "uv_on_plane": px[:2],
            "tip_in_field": tip_in_field,
            "segments": list(zip(ends[2::2], ends[3::2])),
            "tip_px": ends[0] if tip_in_field else None,
            "tip_near_edge": tip_near_edge,
        }

    def snapshot_record(self) -> Optional[dict]:
        """The numbers behind the current drawing, JSON-ready; None without a pose.

        tip_uv/back_uv are the exact float pixels draw_into rounds from,
        tip_cam/back_cam the camera-frame points (mm, ToolPoints) they were
        projected from, p_tip/axis_dir/axis_back_mm the tool geometry,
        los_offset_mm the line-of-sight shift already in tip_cam/back_cam
        (that of the ToolPoints from tool_points(); without one the page's
        ToolPoints.LOS_OFFSET_MM), marked
        whether it is drawn marked (style colour_bgra: the colour used).

        xray_view (None without one): d_x_mm the plane it was computed with,
        tip_x_mm/back_x_mm the points in the X-ray frame, tip_uv_x/back_uv_x
        their working-image pixels, tip_uv_on_plane/back_uv_on_plane the camera
        pixels of their points on the plane (the glyph positions, also outside
        the field), off_plane_mm the tip's n^T P_x - d_x, inclination_deg the
        axis against the plane, tip_in_field whether the tip glyph is drawn,
        segments_px the clipped lines as drawn, colour_bgra their colour.
        """
        if self._tip_uv is None:
            return None
        return {
            "tip_uv": [float(v) for v in self._tip_uv],
            "back_uv": [float(v) for v in self._back_uv],
            "tip_cam_mm": [float(v) for v in self._tip_cam],
            "back_cam_mm": [float(v) for v in self._back_cam],
            "p_tip_mm": [float(v) for v in self._p_tip],
            "axis_dir": [float(v) for v in self._axis],
            "axis_back_mm": float(self.AXIS_BACK_MM),
            "los_offset_mm": self._los_offset_mm,
            "marked": bool(self._marked),
            "style": {
                "tip_radius_px": int(self.TIP_RADIUS),
                "tip_cross_px": int(self.TIP_CROSS),
                "thickness_px": int(self.THICKNESS),
                **({"dash_pieces": int(self.dash_pieces)} if self.dash_pieces
                   else {"dash_px": [float(self.DASH_ON_PX), float(self.DASH_OFF_PX)]}),
                "colour_bgra": [int(c) for c in
                                (self.MARKED_COLOR if self._marked else self.AXIS_COLOR)],
            },
            "xray_view": self._xray_record(),
        }

    def _xray_record(self) -> Optional[dict]:
        xv = self._xv
        if xv is None:
            return None
        return {
            "d_x_mm": self._d_x_mm,
            "tip_x_mm": list(xv["P_x"][0]),
            "back_x_mm": list(xv["P_x"][1]),
            "tip_uv_x": list(xv["uv_x"][0]),
            "back_uv_x": list(xv["uv_x"][1]),
            "tip_uv_on_plane": list(xv["uv_on_plane"][0]),
            "back_uv_on_plane": list(xv["uv_on_plane"][1]),
            "off_plane_mm": xv["P_x"][0][2] - self._d_x_mm,
            "inclination_deg": axis_inclination_deg(xv["P_x"][0], xv["P_x"][1]),
            "tip_in_field": xv["tip_in_field"],
            "segments_px": [[list(a), list(b)] for a, b in xv["segments"]],
            "colour_bgra": [int(c) for c in self._xray_colour()],
        }

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_into(
        self,
        img: np.ndarray,
        *,
        show_camera: bool = True,
        show_xray: bool = True,
        show_tip: bool = True,
        show_axis: bool = True,
    ) -> None:
        """Draw into the frame buffer, in place. No-op without a fresh pose.
        show_camera/show_xray choose the views, show_tip/show_axis the glyphs."""
        if self._tip_uv is None or self._back_uv is None:
            return

        if show_camera:
            tip = self._tip_uv
            back = self._back_uv

            if show_axis:
                self._draw_axis(img, tip, back)
            if show_tip:
                self._draw_tip(img, tip)

        if show_xray and self._xv is not None:
            self._draw_xray(img, show_tip, show_axis)

    def _draw_axis(self, img: np.ndarray, tip: np.ndarray, back: np.ndarray) -> None:
        h, w = img.shape[:2]
        colour = self.MARKED_COLOR if self._marked else self.AXIS_COLOR

        # Solid trajectory line: from the tip onward in the current drill
        # direction (away from the marker), clipped at the image border.
        d = tip - back
        n = float(np.hypot(d[0], d[1]))
        if n > 1e-6:
            far = tip + d * ((float(w) + float(h)) / n)
            ok, p1, p2 = cv2.clipLine(
                (0, 0, w, h),
                (int(round(tip[0])), int(round(tip[1]))),
                (int(round(far[0])), int(round(far[1]))),
            )
            if ok:
                cv2.line(img, p1, p2, colour, self.THICKNESS, cv2.LINE_AA)

        # Dashed shaft segment: tip -> marker side (AXIS_BACK_MM).
        for sa, sb in self._dash_spans(float(np.hypot(*(back - tip))), 0.0, 1.0):
            a = tip + (back - tip) * sa
            b = tip + (back - tip) * sb
            cv2.line(
                img,
                (int(round(a[0])), int(round(a[1]))),
                (int(round(b[0])), int(round(b[1]))),
                colour,
                self.THICKNESS,
                cv2.LINE_AA,
            )

    def _dash_spans(self, length_px: float, lo: float, hi: float) -> list:
        """(lam_a, lam_b) of the drawn dashes on tip (0) -> back (1), clipped to [lo, hi]."""
        n = self.dash_pieces
        if n:
            out = []
            for i in range(0, n, 2):
                a, b = max(lo, i / n), min(hi, (i + 1) / n)
                if a < b:
                    out.append((a, b))
            return out
        return dash_spans(length_px, self.DASH_ON_PX, self.DASH_OFF_PX, lo, hi)

    def _draw_tip(self, img: np.ndarray, tip: np.ndarray) -> None:
        u, v = int(round(tip[0])), int(round(tip[1]))
        colour = self.MARKED_COLOR if self._marked else self.TIP_COLOR
        self._draw_glyph(img, (u, v), colour)

    def _xray_colour(self) -> tuple:
        return self.XRAY_MARKED_COLOR if self._marked else self.XRAY_COLOR

    def _draw_xray(self, img: np.ndarray, show_tip: bool, show_axis: bool) -> None:
        xv = self._xv
        colour = self._xray_colour()
        if show_axis:
            for a, b in xv["segments"]:
                cv2.line(img, a, b, colour, self.THICKNESS, cv2.LINE_AA)
        if not show_tip or xv["tip_px"] is None:
            return
        if not xv["tip_near_edge"]:
            self._draw_glyph(img, xv["tip_px"], colour)
            return

        # The glyph reaches over the field edge: only its pixels inside stay.
        # A patch of ~25 x 25 pixels, and only this close to the edge.
        u, v = xv["tip_px"]
        reach = self.TIP_GLYPH_REACH_PX + 1
        h, w = img.shape[:2]
        x0, x1 = max(0, u - reach), min(w, u + reach + 1)
        y0, y1 = max(0, v - reach), min(h, v + reach + 1)
        if x0 >= x1 or y0 >= y1:
            return
        patch = img[y0:y1, x0:x1]
        saved = patch.copy()
        self._draw_glyph(img, (u, v), colour)
        outside = np.full(patch.shape[:2], 255, dtype=np.uint8)
        polygon = np.rint(self._field_cam).astype(np.int32) - np.array([[[x0, y0]]], np.int32)
        cv2.fillPoly(outside, [polygon], 0)
        cv2.copyTo(saved, outside, patch)

    def _draw_glyph(self, img: np.ndarray, centre: tuple, colour: tuple) -> None:
        cv2.circle(img, centre, self.TIP_RADIUS, colour, self.THICKNESS, cv2.LINE_AA)
        cv2.drawMarker(
            img,
            centre,
            colour,
            markerType=cv2.MARKER_CROSS,
            markerSize=self.TIP_CROSS,
            thickness=self.THICKNESS,
            line_type=cv2.LINE_AA,
        )


class ToolPoints:
    """Tip and axis back point in the camera frame (mm), the points the Live
    page draws, from one TrackerResult. p_tip_mm / p_back_mm are the
    marker-frame points of TrackingOverlay.tool_points().

    Runs in the tracking thread (TrackingWorker) on every result, in
    processing order, with the frame time; the GUI thread only projects the
    two points (TrackingOverlay.update). Stateful, hence reset() whenever
    the tracker is reset.

    The live tracker runs with the output filter off (pose_kf_enabled =
    False): result.rvec/tvec is the unfiltered pose, IR-fused where the
    fusion was applied (timings_ms ir_applied), the RGB-only pose
    otherwise. Live, the fusion is not applied on every frame (no IR pair,
    rejected by its fit gates); drawing the RGB-only pose there would make
    the tip jump by the fused-minus-RGB offset at every switch. So on such
    a frame the RGB-only points are shifted by the camera-frame offset of
    the tip and back point of the last fused frame (fused points minus the
    RGB-only points of rvec_prefusion / tvec_prefusion), for at most
    max_carry_s of frame time after it; without a recent offset the
    RGB-only points are drawn. A fusion the engine rejected for raising the
    RGB reprojection (ir_reproj_gate_reject, near range) drops the offset:
    carrying it would bring back exactly the shift the gate rejected.

    kind after update(): "fused", "carry", "rgb", "gate" (RGB-only, fusion
    rejected by the reprojection gate) or None (no pose). marked after
    update(): the result has points and its visible_cell_fraction
    (timings_ms; the share of the marker cells in view that show checker
    contrast) is below mark_below_visible_frac - a pose the engine kept
    under a partial occlusion (TrackerConfig.drawn_pose_mark_below_visible_frac
    on the Live page; 0 = never).
    """

    # Replays of the live runs: a carried offset is off from the fused one
    # by 0.78 mm (median; p90 2.01) 0.85-1.25 s after its fused frame,
    # dropping it by 0.99 mm (p90 2.09); after longer the p90 is worse than
    # dropping. Max age by frame time, not by processed frames.
    MAX_CARRY_S = 1.0

    # Line-of-sight shift of both drawn points toward the camera along the
    # ray to the marker origin; 0 = off: live draws the raw pose. On the
    # board ground truth the tip lies ~2 mm too far along that ray (a scale
    # error common to the IR-stereo-built marker model and the IR depth);
    # that correction belongs to the evaluation only, not to the live
    # drawing. Until 10.10. 2.27 (RGB-only take mean, 30.09. + 05.10.).
    LOS_OFFSET_MM = 0.0

    def __init__(self, p_tip_mm, p_back_mm, max_carry_s: Optional[float] = None,
                 los_offset_mm: Optional[float] = None,
                 mark_below_visible_frac: float = 0.0) -> None:
        # rows: tip, marker-side end of the dashed shaft segment
        self._p = np.stack([
            np.asarray(p_tip_mm, dtype=np.float64).reshape(3),
            np.asarray(p_back_mm, dtype=np.float64).reshape(3),
        ])
        self.max_carry_s = self.MAX_CARRY_S if max_carry_s is None else float(max_carry_s)
        self.los_offset_mm = self.LOS_OFFSET_MM if los_offset_mm is None else float(los_offset_mm)
        self.mark_below_visible_frac = float(mark_below_visible_frac)
        self.reset()

    def reset(self) -> None:
        self._offset: Optional[np.ndarray] = None   # fused - RGB-only, rows as _p
        self._t_offset = 0.0
        self.kind: Optional[str] = None
        self.marked = False

    def _points(self, rvec, tvec) -> np.ndarray:
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        return self._p @ R.T + np.asarray(tvec, dtype=np.float64).reshape(1, 3)

    def update(self, result, t_frame: float) -> Optional[np.ndarray]:
        """(2, 3) camera-frame points [tip, back] in mm, None without a pose."""
        pts = self._update(result, t_frame)
        frac = float((getattr(result, "timings_ms", None) or {}).get(
            "visible_cell_fraction", float("nan")))
        # NaN (no gate, older engine) compares False: not marked
        self.marked = pts is not None and frac < self.mark_below_visible_frac
        if pts is None or not self.los_offset_mm:
            return pts
        # ray to the marker origin: of the drawn pose, or of the RGB-only
        # pose under the carried offset (pre-fusion pose on fused frames)
        t = None
        if self.kind == "fused":
            t = getattr(result, "tvec_prefusion", None)
        if t is None:
            t = result.tvec
        t = np.asarray(t, dtype=np.float64).reshape(3)
        return pts - self.los_offset_mm * t / max(float(np.linalg.norm(t)), 1e-9)

    def _update(self, result, t_frame: float) -> Optional[np.ndarray]:
        self.kind = None
        rvec = getattr(result, "rvec", None)
        tvec = getattr(result, "tvec", None)
        if rvec is None or tvec is None:
            return None
        pts = self._points(rvec, tvec)
        timings = getattr(result, "timings_ms", None) or {}
        if float(timings.get("ir_applied", 0.0)) > 0.5:
            rp = getattr(result, "rvec_prefusion", None)
            tp = getattr(result, "tvec_prefusion", None)
            if rp is not None and tp is not None:
                self._offset = pts - self._points(rp, tp)
                self._t_offset = float(t_frame)
            self.kind = "fused"
            return pts
        # no fusion this frame: rvec/tvec is the RGB-only pose
        if float(timings.get("ir_reproj_gate_reject", 0.0)) > 0.5:
            self._offset = None
            self.kind = "gate"
            return pts
        if self._offset is not None and float(t_frame) - self._t_offset <= self.max_carry_s:
            self.kind = "carry"
            return pts + self._offset
        self.kind = "rgb"
        return pts
