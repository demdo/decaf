# overlay/tools/tracking_overlay.py
"""In-place drawing of tracked-tool info into the live frame.

The single place for everything drawn into the live image from a
TrackerResult — future additions (uncertainty, chuck, status glyphs, ...)
belong here, not in the page. The same performance rules as the rest of
the live loop apply: draw directly into the frame buffer, touch only the
glyph pixels, no per-frame allocations beyond a few small arrays.

Per frame this costs two 3x3 rotations (tip_camera_blend), one
cv2.projectPoints over two points and a handful of cv2.line calls —
well under 0.1 ms.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

_ZERO3 = np.zeros(3)


class TrackingOverlay:
    """Projects tip + drill axis of one tracked tool and draws them in place.

    update(result) per fresh pose, clear() on loss, draw_into(frame) per
    composed frame. All 3D inputs are marker-frame millimetres.
    """

    # Dashed shaft segment: from the tip toward the marker (the physical
    # drill body side), in millimetres.
    AXIS_BACK_MM = 50.0
    # Number of pieces the dashed segment is split into (odd -> starts and
    # ends with a drawn dash).
    DASH_PIECES = 7

    TIP_COLOR = (80, 220, 80, 255)   # BGRA
    AXIS_COLOR = (80, 220, 80, 255)
    THICKNESS = 2
    TIP_RADIUS = 8
    TIP_CROSS = 20

    def __init__(self, K, dist, p_tip_mm, axis_dir) -> None:
        # Lazy import: pulls in the C++ tracker module, which must only
        # load once tracking is actually used.
        from tracking.hydramarker.tracker import tip_camera_blend

        self._blend = tip_camera_blend

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
        # The blended camera-frame points behind the pixels (mm), kept for
        # the snapshot record; not needed for drawing.
        self._tip_cam: Optional[np.ndarray] = None
        self._back_cam: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Per-frame state
    # ------------------------------------------------------------------

    def update(self, result) -> None:
        """Recompute the 2D geometry from a fresh TrackerResult."""
        tip_cam = self._blend(result, self._p_tip)
        back_cam = self._blend(result, self._p_back)
        if (
            tip_cam is None
            or back_cam is None
            or tip_cam[2] <= 0.0
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

    def clear(self) -> None:
        self._tip_uv = None
        self._back_uv = None
        self._tip_cam = None
        self._back_cam = None

    @property
    def has_pose(self) -> bool:
        """True while draw_into would actually draw (fresh projected pose)."""
        return self._tip_uv is not None

    def snapshot_record(self) -> Optional[dict]:
        """The numbers behind the current drawing, JSON-ready; None without a pose.

        tip_uv/back_uv are the exact float pixels draw_into rounds from,
        tip_cam/back_cam the blended camera-frame points (mm) they were
        projected from, p_tip/axis_dir/axis_back_mm the tool geometry.
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
            "style": {
                "tip_radius_px": int(self.TIP_RADIUS),
                "tip_cross_px": int(self.TIP_CROSS),
                "thickness_px": int(self.THICKNESS),
                "dash_pieces": int(self.DASH_PIECES),
                "colour_bgra": [int(c) for c in self.AXIS_COLOR],
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
                cv2.line(img, p1, p2, self.AXIS_COLOR, self.THICKNESS, cv2.LINE_AA)

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
                self.AXIS_COLOR,
                self.THICKNESS,
                cv2.LINE_AA,
            )

    def _draw_tip(self, img: np.ndarray, tip: np.ndarray) -> None:
        u, v = int(round(tip[0])), int(round(tip[1]))
        cv2.circle(img, (u, v), self.TIP_RADIUS, self.TIP_COLOR, self.THICKNESS, cv2.LINE_AA)
        cv2.drawMarker(
            img,
            (u, v),
            self.TIP_COLOR,
            markerType=cv2.MARKER_CROSS,
            markerSize=self.TIP_CROSS,
            thickness=self.THICKNESS,
            line_type=cv2.LINE_AA,
        )
