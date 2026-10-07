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
handful of cv2.line calls — well under 0.1 ms.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

_ZERO3 = np.zeros(3)


class TrackingOverlay:
    """Projects tip + drill axis of one tracked tool and draws them in place.

    update(points) per fresh pose (TrackOutput.points, from the ToolPoints
    of tool_points()), clear() on loss, draw_into(frame) per composed
    frame. p_tip_mm / axis_dir are marker-frame millimetres.
    """

    # Dashed shaft segment: from the tip toward the marker (the physical
    # drill body side), in millimetres.
    AXIS_BACK_MM = 50.0
    # Number of pieces the dashed segment is split into (odd -> starts and
    # ends with a drawn dash).
    DASH_PIECES = 7

    TIP_COLOR = (80, 220, 80, 255)   # BGRA
    AXIS_COLOR = (80, 220, 80, 255)
    # A pose kept under a partial occlusion (ToolPoints.marked): same glyphs
    # in orange.
    MARKED_COLOR = (0, 165, 255, 255)
    THICKNESS = 2
    TIP_RADIUS = 8
    TIP_CROSS = 20

    def __init__(self, K, dist, p_tip_mm, axis_dir) -> None:
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

    def tool_points(self, los_offset_mm: Optional[float] = None,
                    mark_below_visible_frac: float = 0.0) -> "ToolPoints":
        """A ToolPoints of this tool's tip and back point, for the tracking
        thread; los_offset_mm None: ToolPoints.LOS_OFFSET_MM (the page's);
        mark_below_visible_frac: see ToolPoints.marked."""
        return ToolPoints(self._p_tip, self._p_back, los_offset_mm=los_offset_mm,
                          mark_below_visible_frac=mark_below_visible_frac)

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

    def clear(self) -> None:
        self._tip_uv = None
        self._back_uv = None
        self._tip_cam = None
        self._back_cam = None
        self._marked = False

    @property
    def has_pose(self) -> bool:
        """True while draw_into would actually draw (fresh projected pose)."""
        return self._tip_uv is not None

    def snapshot_record(self) -> Optional[dict]:
        """The numbers behind the current drawing, JSON-ready; None without a pose.

        tip_uv/back_uv are the exact float pixels draw_into rounds from,
        tip_cam/back_cam the camera-frame points (mm, ToolPoints) they were
        projected from, p_tip/axis_dir/axis_back_mm the tool geometry,
        los_offset_mm the line-of-sight shift already in tip_cam/back_cam
        (ToolPoints.LOS_OFFSET_MM, the one the page draws with), marked
        whether it is drawn marked (style colour_bgra: the colour used).
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
            "los_offset_mm": float(ToolPoints.LOS_OFFSET_MM),
            "marked": bool(self._marked),
            "style": {
                "tip_radius_px": int(self.TIP_RADIUS),
                "tip_cross_px": int(self.TIP_CROSS),
                "thickness_px": int(self.THICKNESS),
                "dash_pieces": int(self.DASH_PIECES),
                "colour_bgra": [int(c) for c in
                                (self.MARKED_COLOR if self._marked else self.AXIS_COLOR)],
            },
        }

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_into(
        self,
        img: np.ndarray,
        *,
        show_tip: bool = True,
        show_axis: bool = True,
    ) -> None:
        """Draw into the frame buffer, in place. No-op without a fresh pose."""
        if self._tip_uv is None or self._back_uv is None:
            return

        tip = self._tip_uv
        back = self._back_uv

        if show_axis:
            self._draw_axis(img, tip, back)
        if show_tip:
            self._draw_tip(img, tip)

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

        # Dashed shaft segment: tip -> marker side (AXIS_BACK_MM), drawn as
        # alternating pieces.
        pieces = self.DASH_PIECES
        for i in range(0, pieces, 2):
            a = tip + (back - tip) * (i / pieces)
            b = tip + (back - tip) * ((i + 1) / pieces)
            cv2.line(
                img,
                (int(round(a[0])), int(round(a[1]))),
                (int(round(b[0])), int(round(b[1]))),
                colour,
                self.THICKNESS,
                cv2.LINE_AA,
            )

    def _draw_tip(self, img: np.ndarray, tip: np.ndarray) -> None:
        u, v = int(round(tip[0])), int(round(tip[1]))
        colour = self.MARKED_COLOR if self._marked else self.TIP_COLOR
        cv2.circle(img, (u, v), self.TIP_RADIUS, colour, self.THICKNESS, cv2.LINE_AA)
        cv2.drawMarker(
            img,
            (u, v),
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

    # EMPIRICAL depth correction of the drawn points, not a calibration: on
    # the board ground truth (30.09. + 05.10., 15 takes, marker 300-480 mm
    # from the camera) the RGB-only marker pose puts the tip 2.27 mm (take
    # mean) too far along the line of sight to the marker, with no trend over
    # the distance and turning with the camera, not with the label; the
    # cause is open. Both points move this far toward the camera along the
    # ray to the marker origin. Held out (fit without the take / the session):
    # tip error 1.37 / 1.38 mm instead of 2.87 (offset carry), 1.28 / 1.29
    # mm instead of 2.82 (filtered drawing).
    # Re-fit after any change of the marker model, the RGB intrinsics or the
    # tip calibration; < 300 and > 480 mm untested. 0 = off; measurements
    # against that ground truth (tests/test_tip_board_principle.py) run
    # with 0, the value was fitted on it.
    LOS_OFFSET_MM = 2.27

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
