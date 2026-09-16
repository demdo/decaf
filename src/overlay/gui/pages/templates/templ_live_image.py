# overlay/gui/pages/templates/page_live_image.py

from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QTimer
import pyrealsense2 as rs
import numpy as np

from overlay.gui.pages.templates.templ_base_image import BaseImagePage


class LiveImagePage(BaseImagePage):
    FPS: int = 30

    # all live pages use the same viewport size
    VIEWPORT_SIZE: Tuple[int, int] = (576, 324)
    RENDER_MODE: str = "cover"

    def __init__(self, parent=None):
        super().__init__(parent)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update_view)

        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        
        
    # ---------------- RGB helper ----------------

    def color_frame_to_bgr(self, cf) -> Optional[np.ndarray]:
        if cf is None:
            return None
        return np.asanyarray(cf.get_data())

    # ---------------- Timer ----------------

    def start_timer(self, fps: Optional[int] = None) -> None:
        fps = int(fps or self.FPS)
        fps = max(1, fps)
        self._timer.start(int(1000 / fps))

    def stop_timer(self) -> None:
        self._timer.stop()

    # ---------------- RealSense (single entry point) ----------------

    def start_realsense(
        self,
        *,
        fps: Optional[int] = None,
        color_size: Tuple[int, int] = (1920, 1080),
        depth_size: Optional[Tuple[int, int]] = None,  # None => color-only
        align_to: Optional[str] = None,               # None | "color"
    ) -> rs.pipeline:
        """
        Single unified RealSense start.

        - depth_size=None     -> color only
        - depth_size=(w,h)    -> color + depth
        - align_to="color"    -> self.align aligns frames to COLOR (depth aligned to color)
        """
        fps = int(fps or self.FPS)
        fps = max(1, fps)

        cw, ch = color_size

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, fps)

        if depth_size is not None:
            dw, dh = depth_size
            config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)

        profile = pipeline.start(config)
        self._reduce_latency(profile)

        self.pipeline = pipeline
        self.align = rs.align(rs.stream.color) if align_to == "color" else None
        return pipeline

    def stop_realsense(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None
        self.align = None

    @staticmethod
    def _reduce_latency(profile) -> None:
        try:
            dev = profile.get_device()
            for sensor in dev.query_sensors():
                try:
                    sensor.set_option(rs.option.frames_queue_size, 1)
                except Exception:
                    pass
        except Exception:
            pass

    def on_leave(self) -> None:
        self.stop_timer()
        self.stop_realsense()
        self.set_viewport_background(active=False)
        self.clear_view()
