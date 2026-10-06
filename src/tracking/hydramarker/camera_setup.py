"""Camera setup and frame sources for live HydraMarker workflows.

This module is the hardware boundary for the Python tracking tools. User code
chooses a camera in ``config.py``; this module opens the requested device,
validates the active stream, loads an optional tracking calibration, and returns
frames as OpenCV BGR images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Optional

import cv2
import numpy as np

from tracking.hydramarker.config import CameraConfig


@dataclass
class CameraFrame:
    """One camera frame in the common HydraMarker BGR format."""

    image_bgr: np.ndarray
    frame_index: int
    timestamp_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    # Synchronized global-shutter IR pair (same RealSense frameset as the
    # colour image). None on every backend except RealSense with
    # realsense_enable_ir; consumers must treat them as optional.
    ir_left: np.ndarray | None = None
    ir_right: np.ndarray | None = None


class CameraSource:
    """Base class for live camera sources."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.config.validate_common()
        self.K: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.calibration_info: dict[str, Any] = {}
        self.actual_width = int(config.width)
        self.actual_height = int(config.height)
        self.actual_fps = int(config.fps)
        self.pixel_format = ""
        self._frame_index = 0
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        raise NotImplementedError

    def read(self) -> CameraFrame | None:
        raise NotImplementedError

    def stop(self) -> None:
        self._started = False

    def metadata(self) -> dict[str, Any]:
        meta = {
            "camera_backend": str(self.config.backend).strip().lower(),
            "camera_serial": self.config.serial or "",
            "width": int(self.actual_width),
            "height": int(self.actual_height),
            "fps": int(self.actual_fps),
            "pixel_format": str(self.pixel_format),
        }
        meta.update(dict(self.calibration_info))
        return meta

    def _load_calibration_if_configured(self) -> None:
        path = self.config.calibration_path_obj()
        if path is None:
            self.K = None
            self.dist_coeffs = None
            self.calibration_info = {}
            return

        from tracking.hydramarker.calib import calib_camera

        K, dist, info = calib_camera.load_tracking_calibration_npz(path)
        calib_camera.validate_calibration_image_size(
            info,
            width=int(self.actual_width),
            height=int(self.actual_height),
        )
        info.update(
            {
                "tracker_K": K.tolist(),
                "tracker_dist_coeffs": dist.reshape(-1).tolist(),
            }
        )
        self.K = K
        self.dist_coeffs = dist
        self.calibration_info = info

    def _next_frame_index(self) -> int:
        self._frame_index += 1
        return self._frame_index


class RealSenseCameraSource(CameraSource):
    """Intel RealSense color stream source."""

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        self._rs = None
        self._pipeline = None
        self._profile = None
        self._stereo_sensor_cache = None
        self._ir_exposure_us = None
        self._live_frames = None  # newest poll() frameset (keeps views alive)

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def profile(self):
        return self._profile

    def start(self) -> None:
        rs = _require_realsense()
        self._rs = rs
        pipeline = rs.pipeline()
        cfg = rs.config()
        if self.config.serial:
            cfg.enable_device(str(self.config.serial))
        color_format = str(
            getattr(self.config, "realsense_color_format", "bgr8")
        ).strip().lower()
        cfg.enable_stream(
            rs.stream.color,
            int(self.config.width),
            int(self.config.height),
            rs.format.bgra8 if color_format == "bgra8" else rs.format.bgr8,
            int(self.config.fps),
        )
        self._ir_enabled = bool(getattr(self.config, "realsense_enable_ir", False))
        if self._ir_enabled:
            # global-shutter IR pair for the tracker's IR pose refinement
            # (design B); same frameset as the colour image -> synchronized
            # by the SDK. Emitter is disabled below (the speckle pattern
            # ruins the checkerboard corners).
            ir_w = int(getattr(self.config, "realsense_ir_width", 1280))
            ir_h = int(getattr(self.config, "realsense_ir_height", 720))
            cfg.enable_stream(rs.stream.infrared, 1, ir_w, ir_h,
                              rs.format.y8, int(self.config.fps))
            cfg.enable_stream(rs.stream.infrared, 2, ir_w, ir_h,
                              rs.format.y8, int(self.config.fps))
        profile = pipeline.start(cfg)
        try:
            if self._ir_enabled:
                ir_exposure = getattr(self.config, "realsense_ir_exposure_us", None)
                ir_gain = getattr(self.config, "realsense_ir_gain", None)
                for sensor in profile.get_device().query_sensors():
                    options = [
                        (rs.option.emitter_enabled, 0.0),
                        (rs.option.laser_power, 0.0),
                    ]
                    # optional manual IR exposure (short -> less motion blur on the
                    # global-shutter pair during fast motion)
                    is_depth = False
                    try:
                        is_depth = "stereo" in sensor.get_info(
                            rs.camera_info.name).lower()
                    except Exception:
                        pass
                    if is_depth and ir_exposure is not None:
                        options.append((rs.option.enable_auto_exposure, 0.0))
                        options.append((rs.option.exposure, float(ir_exposure)))
                        if ir_gain is not None:
                            options.append((rs.option.gain, float(ir_gain)))
                    for option, value in options:
                        try:
                            if sensor.supports(option):
                                sensor.set_option(option, value)
                        except Exception:
                            pass
            realsense_details = _configure_realsense_color_sensor(profile, self.config, rs)

            queue_size = getattr(self.config, "realsense_frames_queue_size", None)
            if queue_size is not None:
                for sensor in profile.get_device().query_sensors():
                    try:
                        if sensor.supports(rs.option.frames_queue_size):
                            sensor.set_option(
                                rs.option.frames_queue_size, float(int(queue_size))
                            )
                    except Exception:
                        pass

            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_stream.get_intrinsics()
            self.actual_width = int(getattr(intr, "width", self.config.width))
            self.actual_height = int(getattr(intr, "height", self.config.height))
            self.actual_fps = int(color_stream.fps())
            self.pixel_format = color_format
            _validate_active_stream(self.config, self.actual_width, self.actual_height, self.actual_fps)
        except Exception:
            # self._pipeline is not set yet, so stop() would not reach it:
            # stop here, or the device stays busy until the process ends
            try:
                pipeline.stop()
            except Exception:
                pass
            raise

        self._pipeline = pipeline
        self._profile = profile
        self._started = True
        self._load_calibration_if_configured()

        raw_K = np.array(
            [
                [float(intr.fx), 0.0, float(intr.ppx)],
                [0.0, float(intr.fy), float(intr.ppy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        raw_dist = np.asarray(intr.coeffs[:5], dtype=np.float64).reshape(-1, 1)
        self.calibration_info.update(
            {
                "camera_backend": "realsense",
                "raw_realsense_model": str(getattr(intr, "model", "unknown")),
                "raw_realsense_coeffs": [float(c) for c in intr.coeffs],
                "raw_realsense_K": raw_K.tolist(),
                "raw_realsense_dist_coeffs": raw_dist.reshape(-1).tolist(),
                **realsense_details,
            }
        )

    def _stereo_sensor(self):
        if self._stereo_sensor_cache is not None:
            return self._stereo_sensor_cache
        if self._profile is None or self._rs is None:
            return None
        rs = self._rs
        for sensor in self._profile.get_device().query_sensors():
            try:
                name = sensor.get_info(rs.camera_info.name).lower()
            except Exception:
                name = ""
            if "stereo" in name:
                self._stereo_sensor_cache = sensor
                return sensor
        return None

    def set_ir_exposure(self, exposure_us: float) -> bool:
        """Set a MANUAL IR (stereo module) exposure at runtime (disables IR
        auto-exposure). Driven ONCE by the marker exposure calibration at
        start-up; never called in steady state, so zero per-frame cost."""
        if not getattr(self, "_ir_enabled", False):
            return False
        sensor = self._stereo_sensor()
        if sensor is None:
            return False
        rs = self._rs
        ok = False
        for option, value in ((rs.option.enable_auto_exposure, 0.0),
                              (rs.option.exposure, float(exposure_us))):
            try:
                if sensor.supports(option):
                    sensor.set_option(option, value)
                    ok = True
            except Exception:
                pass
        if ok:
            self._ir_exposure_us = float(exposure_us)
        return ok

    def poll(self) -> CameraFrame | None:
        """Non-blocking read: return the newest complete frameset, or None.

        For GUI frame loops (oversampled QTimer) where blocking
        ``wait_for_frames`` would stall the event loop. Unlike ``read``,
        the returned arrays are ZERO-COPY views into the SDK buffers; the
        source keeps the frameset alive until the next ``poll``/``stop``,
        so the views are valid only until then. Callers must copy (or
        finish using) the data within the current tick — the overlay live
        loop copies into its persistent frame buffer immediately.

        With ``realsense_color_format="bgra8"`` the ``image_bgr`` field
        carries a 4-channel BGRA image (field name kept for compatibility).
        """
        if self._pipeline is None:
            raise RuntimeError("RealSense camera is not started.")
        frames = self._pipeline.poll_for_frames()
        if not frames:
            return None
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        image = np.asanyarray(color_frame.get_data())
        if image.size == 0:
            return None
        ir_left = ir_right = None
        if getattr(self, "_ir_enabled", False):
            irl = frames.get_infrared_frame(1)
            irr = frames.get_infrared_frame(2)
            if irl and irr:
                ir_left = np.asanyarray(irl.get_data())
                ir_right = np.asanyarray(irr.get_data())
        # Keep the composite frameset referenced so the zero-copy views
        # above stay valid until the next poll() replaces it.
        self._live_frames = frames
        return CameraFrame(
            image_bgr=image,
            frame_index=self._next_frame_index(),
            timestamp_s=time.time(),
            metadata={"backend": "realsense"},
            ir_left=ir_left,
            ir_right=ir_right,
        )

    def read(self) -> CameraFrame | None:
        if self._pipeline is None:
            raise RuntimeError("RealSense camera is not started.")
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        image = np.asanyarray(color_frame.get_data()).copy()
        ir_left = ir_right = None
        if getattr(self, "_ir_enabled", False):
            irl = frames.get_infrared_frame(1)
            irr = frames.get_infrared_frame(2)
            if irl and irr:
                ir_left = np.asanyarray(irl.get_data()).copy()
                ir_right = np.asanyarray(irr.get_data()).copy()
        return CameraFrame(
            image_bgr=image,
            frame_index=self._next_frame_index(),
            timestamp_s=time.time(),
            metadata={"backend": "realsense"},
            ir_left=ir_left,
            ir_right=ir_right,
        )

    def stop(self) -> None:
        try:
            if self._pipeline is not None:
                self._pipeline.stop()
        finally:
            self._live_frames = None
            self._pipeline = None
            self._profile = None
            super().stop()


class BaslerCameraSource(CameraSource):
    """Basler pypylon source converted to OpenCV BGR frames."""

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        self._pylon = None
        self._camera = None
        self._converter = None
        self._debayer_code: Optional[int] = None

    @property
    def camera(self):
        return self._camera

    def start(self) -> None:
        pylon = _require_pypylon()
        self._pylon = pylon
        factory = pylon.TlFactory.GetInstance()
        camera = self._create_basler_camera(factory, pylon)
        camera.Open()

        try:
            _set_integer_node(camera.Width, int(self.config.width), "Width")
            _set_integer_node(camera.Height, int(self.config.height), "Height")
            _configure_basler_stream(camera, self.config)

            self.actual_width = int(camera.Width.GetValue())
            self.actual_height = int(camera.Height.GetValue())
            self.actual_fps = _read_basler_fps(camera, fallback=int(self.config.fps))
            self.pixel_format = str(camera.PixelFormat.GetValue()) if hasattr(camera, "PixelFormat") else ""
            self._debayer_code = _basler_debayer_code(self.pixel_format)
            basler_details = {
                "pixel_format": self.pixel_format,
                "exposure_auto": _read_node_value(camera, "ExposureAuto"),
                "exposure_time_us": _read_node_value(camera, "ExposureTime"),
                "gain_auto": _read_node_value(camera, "GainAuto"),
                "gain": _read_node_value(camera, "Gain"),
                "device_link_throughput_limit_mode": _read_node_value(
                    camera,
                    "DeviceLinkThroughputLimitMode",
                ),
                "device_link_throughput_limit": _read_node_value(
                    camera,
                    "DeviceLinkThroughputLimit",
                ),
                "acquisition_frame_rate": _read_node_value(
                    camera,
                    "AcquisitionFrameRate",
                ),
                "resulting_frame_rate": _read_node_value(
                    camera,
                    "ResultingFrameRate",
                    "AcquisitionResultingFrameRate",
                ),
            }
            _validate_active_stream(
                self.config,
                self.actual_width,
                self.actual_height,
                self.actual_fps,
                details=basler_details,
            )

            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            self._camera = camera
            self._converter = converter
            self._started = True
            self._load_calibration_if_configured()
            self.calibration_info.update(
                {
                    "camera_backend": "basler",
                    "camera_model": str(camera.GetDeviceInfo().GetModelName()),
                    "camera_serial": str(camera.GetDeviceInfo().GetSerialNumber()),
                    **basler_details,
                }
            )
        except Exception:
            camera.Close()
            raise

    def read(self) -> CameraFrame | None:
        if self._camera is None or self._converter is None or self._pylon is None:
            raise RuntimeError("Basler camera is not started.")
        grab = self._camera.RetrieveResult(
            5000,
            self._pylon.TimeoutHandling_Return,
        )
        if grab is None:
            return None
        try:
            if not grab.GrabSucceeded():
                return None
            if self._debayer_code is not None:
                raw = np.asarray(grab.GetArray())
                image = cv2.cvtColor(raw, self._debayer_code)
            else:
                converted = self._converter.Convert(grab)
                image = np.asarray(converted.GetArray()).copy()
                image = _ensure_bgr(image)
            return CameraFrame(
                image_bgr=image,
                frame_index=self._next_frame_index(),
                timestamp_s=time.time(),
                metadata={"backend": "basler"},
            )
        finally:
            grab.Release()

    def stop(self) -> None:
        try:
            if self._camera is not None:
                if self._camera.IsGrabbing():
                    self._camera.StopGrabbing()
                if self._camera.IsOpen():
                    self._camera.Close()
        finally:
            self._camera = None
            self._converter = None
            self._debayer_code = None
            super().stop()

    def _create_basler_camera(self, factory, pylon):
        if not self.config.serial:
            return pylon.InstantCamera(factory.CreateFirstDevice())

        serial = str(self.config.serial)
        for device in factory.EnumerateDevices():
            if str(device.GetSerialNumber()) == serial:
                return pylon.InstantCamera(factory.CreateDevice(device))
        raise RuntimeError(f"No Basler camera with serial {serial!r} was found.")


def create_camera_source(config: CameraConfig) -> CameraSource:
    """Create a camera source from a validated user config."""
    config.validate_common()
    backend = str(config.backend).strip().lower()
    if backend == "realsense":
        return RealSenseCameraSource(config)
    if backend == "basler":
        return BaslerCameraSource(config)
    raise ValueError(f"Unsupported camera backend: {config.backend!r}")


def _require_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is required for camera backend='realsense'."
        ) from exc
    return rs


def _require_pypylon():
    try:
        from pypylon import pylon
    except ImportError as exc:
        raise RuntimeError("pypylon is required for camera backend='basler'.") from exc
    return pylon


# Colour sensor options as (rs.option name, CameraConfig field).
# exposure and gain belong to the auto exposure, white_balance to the auto
# white balance: writing one switches its automatic off (librealsense
# auto-disabling controls), and on Windows switching the AE off writes the
# DEFAULT exposure (librealsense mf-uvc.cpp). So the manual values are
# written with their automatic off, the automatic flag after them.
_REALSENSE_COLOR_AUTO_GROUPS = (
    (("enable_auto_exposure", "realsense_enable_auto_exposure"),
     (("exposure", "realsense_exposure"), ("gain", "realsense_gain"))),
    (("enable_auto_white_balance", "realsense_enable_auto_white_balance"),
     (("white_balance", "realsense_white_balance"),)),
)
_REALSENSE_COLOR_OTHER = (
    # 0 = constant frame rate: AE caps the exposure at the frame period
    # and compensates via gain (guaranteed fps even in dim light)
    ("auto_exposure_priority", "realsense_auto_exposure_priority"),
    ("power_line_frequency", "realsense_power_line_frequency"),
    ("brightness", "realsense_brightness"),
    ("contrast", "realsense_contrast"),
    ("saturation", "realsense_saturation"),
    ("hue", "realsense_hue"),
    ("sharpness", "realsense_sharpness"),
    ("gamma", "realsense_gamma"),
    ("backlight_compensation", "realsense_backlight_compensation"),
)
REALSENSE_COLOR_OPTIONS = tuple(
    pair for auto, manual in _REALSENSE_COLOR_AUTO_GROUPS for pair in (auto, *manual)
) + _REALSENSE_COLOR_OTHER

# What start records besides the option values (calibration_info keys).
_REALSENSE_COLOR_RECORD = (
    "realsense_color_ae_roi_px",
    "realsense_color_ae_roi_skipped",
    "realsense_color_ae_roi_error",
    "realsense_color_written",
    "realsense_color_mismatch",
)

# Short names for log lines (Live page profile header and log.md, the
# evaluation script); REALSENSE_COLOR_KEY_VALUES are the ones a header names.
_REALSENSE_COLOR_SHORT = (
    ("realsense_enable_auto_exposure", "AE"),
    ("realsense_auto_exposure_priority", "AE_priority"),
    ("realsense_exposure", "exposure"),
    ("realsense_gain", "gain"),
    ("realsense_power_line_frequency", "PLF"),
    ("realsense_enable_auto_white_balance", "AWB"),
    ("realsense_white_balance", "WB"),
    ("realsense_brightness", "brightness"),
    ("realsense_contrast", "contrast"),
    ("realsense_saturation", "saturation"),
    ("realsense_hue", "hue"),
    ("realsense_sharpness", "sharpness"),
    ("realsense_gamma", "gamma"),
    ("realsense_backlight_compensation", "backlight"),
)
REALSENSE_COLOR_KEY_VALUES = (
    "AE", "AE_priority", "exposure", "gain", "PLF", "AE_ROI", "writes", "mismatch",
)


def realsense_color_readback(info: dict[str, Any]) -> dict[str, Any]:
    """The colour options a started RealSense source runs with.

    info is the source's calibration_info. Returns CameraConfig field ->
    value read back after start, the writes start made, in order
    ("realsense_color_written", "option=value") and the config values the
    device did not take ("realsense_color_mismatch"). Empty for other
    cameras or without a colour sensor.
    """
    if not info.get("realsense_color_sensor_found"):
        return {}
    keys = [name for _, name in REALSENSE_COLOR_OPTIONS] + list(_REALSENSE_COLOR_RECORD)
    return {key: info[key] for key in keys if key in info}


def realsense_color_summary(
    readback: dict[str, Any], keys: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Short name -> value of a realsense_color_readback for log lines
    (whole numbers as int). keys limits it to these short names; "writes"
    is the number of options start wrote, "mismatch" the options the device
    did not take."""
    out: dict[str, Any] = {}
    for name, short in _REALSENSE_COLOR_SHORT:
        if name in readback and (keys is None or short in keys):
            value = float(readback[name])
            out[short] = int(value) if value.is_integer() else value
    if keys is None or "AE_ROI" in keys:
        if readback.get("realsense_color_ae_roi_px") is not None:
            out["AE_ROI"] = ",".join(str(v) for v in readback["realsense_color_ae_roi_px"])
        elif "realsense_color_ae_roi_error" in readback:
            out["AE_ROI"] = "error"
    written = readback.get("realsense_color_written")
    if written is not None and (keys is None or "writes" in keys):
        out["writes"] = len(written)
    mismatch = readback.get("realsense_color_mismatch")
    if mismatch and (keys is None or "mismatch" in keys):
        out["mismatch"] = ",".join(sorted(mismatch))
    return out


def format_realsense_color(
    readback: dict[str, Any], keys: tuple[str, ...] | None = REALSENSE_COLOR_KEY_VALUES
) -> str:
    """One line, "AE=1 AE_priority=0 exposure=156 ..." ("" if empty)."""
    summary = realsense_color_summary(readback, keys)
    return " ".join(f"{k}={v}" for k, v in summary.items())


def _configure_realsense_color_sensor(profile, config: CameraConfig, rs) -> dict[str, Any]:
    """Apply the config's colour options and read all of them back.

    Per automatic (AE, then auto white balance): if a manual value has to
    change, the automatic goes off, the manual values are written, then the
    automatic flag as configured; then the other options. An option whose
    read-back already equals the config value is not written - a device in
    the configured state is left untouched (no AE off/on at start).
    """
    sensor = _find_realsense_color_sensor(profile, rs)
    if sensor is None:
        return {"realsense_color_sensor_found": False}

    written: list[str] = []

    def apply(option_name: str, config_name: str, value) -> None:
        if _realsense_option_equals(sensor, rs, option_name, value):
            return
        _set_realsense_option(sensor, rs, option_name, value, config_name)
        written.append(f"{option_name}={_realsense_option_number(value):g}")

    for (auto_name, auto_config), manual in _REALSENSE_COLOR_AUTO_GROUPS:
        values = [
            (option_name, config_name, getattr(config, config_name, None))
            for option_name, config_name in manual
        ]
        values = [entry for entry in values if entry[2] is not None]
        if any(not _realsense_option_equals(sensor, rs, name, value)
               for name, _, value in values):
            apply(auto_name, auto_config, False)
            for option_name, config_name, value in values:
                apply(option_name, config_name, value)
        auto = getattr(config, auto_config, None)
        if auto is not None:
            apply(auto_name, auto_config, auto)
    for option_name, config_name in _REALSENSE_COLOR_OTHER:
        value = getattr(config, config_name, None)
        if value is not None:
            apply(option_name, config_name, value)
    # after the last AE write (the metering region only matters while it runs)
    roi = getattr(config, "realsense_color_ae_roi", None)
    roi_details: dict[str, Any] = {}
    roi_box = None
    if roi is not None:
        width, height = int(config.width), int(config.height)
        try:
            stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            width, height = int(stream.width()), int(stream.height())
        except Exception:
            pass
        roi_box = _realsense_roi_pixels(roi, width, height)
        roi_details = _apply_realsense_color_ae_roi(sensor, rs, roi_box, written)

    sensor_name = _realsense_sensor_name(sensor, rs)
    details: dict[str, Any] = {
        "realsense_color_sensor_found": True,
        "realsense_color_sensor_name": sensor_name,
        "realsense_color_written": written,
        **roi_details,
    }
    mismatch: dict[str, dict[str, Any]] = {}
    for option_name, config_name in REALSENSE_COLOR_OPTIONS:
        current = _read_realsense_option(sensor, rs, option_name)
        if current is not None:
            details[config_name] = current
        option_range = _read_realsense_option_range(sensor, rs, option_name)
        if option_range is not None:
            details[f"{config_name}_range"] = option_range
        wanted = getattr(config, config_name, None)
        if wanted is not None and current is not None and not _realsense_option_equals(
            sensor, rs, option_name, wanted, current
        ):
            mismatch[config_name] = {
                "config": _realsense_option_number(wanted), "device": current,
            }
    roi_px = roi_details.get("realsense_color_ae_roi_px")
    if (roi_box is not None and roi_px is not None and roi_px != roi_box
            and "realsense_color_ae_roi_skipped" not in roi_details):
        mismatch["realsense_color_ae_roi"] = {"config": roi_box, "device": roi_px}
    if mismatch:
        details["realsense_color_mismatch"] = mismatch
    return details


def _realsense_roi_pixels(roi, width: int, height: int) -> list[int]:
    """(min_x, min_y, max_x, max_y) as fractions -> pixels of the image."""
    x0, y0, x1, y1 = (float(v) for v in roi)
    return [int(round(x0 * width)), int(round(y0 * height)),
            int(round(x1 * width)), int(round(y1 * height))]


def _read_realsense_roi(roi_sensor) -> list[int]:
    region = roi_sensor.get_region_of_interest()
    return [int(region.min_x), int(region.min_y), int(region.max_x), int(region.max_y)]


def _apply_realsense_color_ae_roi(sensor, rs, box: list[int], written: list[str]) -> dict[str, Any]:
    """Set the colour AE metering region (pixels) if the AE is on and the
    region differs; read it back. Never raises: an error is recorded and the
    camera runs with the device's region."""
    out: dict[str, Any] = {}
    try:
        roi_sensor = rs.roi_sensor(sensor)
        if not _realsense_option_equals(sensor, rs, "enable_auto_exposure", True):
            out["realsense_color_ae_roi_skipped"] = "auto exposure off"
        elif _read_realsense_roi(roi_sensor) != box:
            region = rs.region_of_interest()
            region.min_x, region.min_y, region.max_x, region.max_y = box
            roi_sensor.set_region_of_interest(region)
            written.append("ae_roi=" + ",".join(str(v) for v in box))
        out["realsense_color_ae_roi_px"] = _read_realsense_roi(roi_sensor)
    except Exception as exc:
        out["realsense_color_ae_roi_error"] = repr(exc)
    return out


def _find_realsense_color_sensor(profile, rs):
    device = profile.get_device()
    fallback = None
    for sensor in device.query_sensors():
        name = _realsense_sensor_name(sensor, rs).lower()
        if "rgb" in name or "color" in name:
            return sensor
        awb = getattr(rs.option, "enable_auto_white_balance", None)
        if awb is not None:
            try:
                if sensor.supports(awb):
                    fallback = sensor
            except Exception:
                pass
    return fallback


def _realsense_sensor_name(sensor, rs) -> str:
    try:
        return str(sensor.get_info(rs.camera_info.name))
    except Exception:
        return "unknown"


def _realsense_option_number(value) -> float:
    return 1.0 if isinstance(value, bool) and value else 0.0 if isinstance(value, bool) else float(value)


def _realsense_option_equals(sensor, rs, option_name: str, value, current=None) -> bool:
    """True if the option reads back as value (current: a read-back already
    taken). False if it cannot be read."""
    if current is None:
        current = _read_realsense_option(sensor, rs, option_name)
    return current is not None and abs(current - _realsense_option_number(value)) < 1e-6


def _set_realsense_option(sensor, rs, option_name: str, value, config_name: str) -> None:
    option = getattr(rs.option, option_name, None)
    if option is None:
        raise RuntimeError(f"RealSense option {option_name!r} is not available in pyrealsense2.")
    if not sensor.supports(option):
        raise RuntimeError(f"RealSense color sensor does not support option {option_name!r}.")

    numeric = _realsense_option_number(value)
    try:
        option_range = sensor.get_option_range(option)
        minimum = float(option_range.min)
        maximum = float(option_range.max)
    except Exception:
        minimum = maximum = None

    if minimum is not None and maximum is not None and not (minimum <= numeric <= maximum):
        raise RuntimeError(
            f"RealSense {config_name}={value!r} outside supported range "
            f"{minimum:g}..{maximum:g}."
        )
    sensor.set_option(option, numeric)


def _read_realsense_option(sensor, rs, option_name: str):
    option = getattr(rs.option, option_name, None)
    if option is None:
        return None
    try:
        if not sensor.supports(option):
            return None
        return float(sensor.get_option(option))
    except Exception:
        return None


def _read_realsense_option_range(sensor, rs, option_name: str):
    option = getattr(rs.option, option_name, None)
    if option is None:
        return None
    try:
        if not sensor.supports(option):
            return None
        option_range = sensor.get_option_range(option)
        return {
            "min": float(option_range.min),
            "max": float(option_range.max),
            "step": float(option_range.step),
            "default": float(option_range.def_),
        }
    except Exception:
        return None


def _validate_active_stream(
    config: CameraConfig,
    width: int,
    height: int,
    fps: int,
    *,
    details: Optional[dict[str, Any]] = None,
) -> None:
    expected = (int(config.width), int(config.height), int(config.fps))
    actual = (int(width), int(height), int(fps))
    if actual != expected:
        detail_text = ""
        if details:
            detail_items = ", ".join(
                f"{key}={value!r}"
                for key, value in details.items()
                if value is not None
            )
            if detail_items:
                detail_text = f" Active camera details: {detail_items}."
        raise RuntimeError(
            f"Requested {expected[0]}x{expected[1]}@{expected[2]}fps, "
            f"but active camera stream is {actual[0]}x{actual[1]}@{actual[2]}fps."
            f"{detail_text}"
        )


def _set_integer_node(node, value: int, name: str) -> None:
    minimum = int(node.GetMin())
    maximum = int(node.GetMax())
    increment = int(node.GetInc()) if hasattr(node, "GetInc") else 1
    if int(value) < minimum or int(value) > maximum:
        raise RuntimeError(f"Basler {name}={value} outside supported range {minimum}..{maximum}.")
    if increment > 1 and (int(value) - minimum) % increment != 0:
        raise RuntimeError(
            f"Basler {name}={value} does not match required increment {increment} "
            f"from minimum {minimum}."
        )
    node.SetValue(int(value))


def _set_node_if_present(camera, name: str, value, *, strict: bool = False) -> bool:
    if not hasattr(camera, name):
        if strict:
            raise RuntimeError(f"Basler node {name!r} is not available.")
        return False
    node = getattr(camera, name)
    try:
        if hasattr(node, "SetValue"):
            node.SetValue(value)
            return True
    except Exception as exc:
        raise RuntimeError(f"Could not set Basler {name}={value!r}.") from exc
    return False


def _set_node_if_configured(camera, name: str, value) -> bool:
    if value is None:
        return False
    return _set_node_if_present(camera, name, value)


def _configure_basler_stream(camera, config: CameraConfig) -> None:
    # Use raw 8-bit Bayer by default for color dart cameras: it keeps USB
    # bandwidth low, then pypylon converts to OpenCV BGR on the host.
    pixel_format = _resolve_basler_pixel_format(camera, config.pixel_format)
    _set_node_if_configured(camera, "PixelFormat", pixel_format)
    _set_node_if_configured(camera, "ExposureAuto", config.exposure_auto)
    _set_node_if_configured(camera, "ExposureTime", config.exposure_time_us)
    _set_node_if_configured(camera, "GainAuto", config.gain_auto)
    _set_node_if_configured(camera, "Gain", config.gain_db)
    _set_node_if_configured(
        camera,
        "DeviceLinkThroughputLimitMode",
        config.device_link_throughput_limit_mode,
    )
    _set_node_if_configured(
        camera,
        "DeviceLinkThroughputLimit",
        config.device_link_throughput_limit,
    )
    _set_node_if_present(camera, "AcquisitionFrameRateEnable", True)
    _set_node_if_present(camera, "AcquisitionFrameRate", float(config.fps))


def _resolve_basler_pixel_format(camera, configured: Optional[str]) -> Optional[str]:
    if configured is None:
        return None
    value = str(configured).strip()
    if value.lower() != "auto":
        return value
    symbols = _read_enum_symbols(camera, "PixelFormat")
    for candidate in ("BayerRG8", "BayerGB8", "BayerGR8", "BayerBG8", "BGR8", "RGB8", "Mono8"):
        if candidate in symbols:
            return candidate
    raise RuntimeError(
        "Basler pixel_format='auto' did not find a supported 8-bit pixel format. "
        f"Available PixelFormat symbols: {sorted(symbols)!r}."
    )


def _basler_debayer_code(pixel_format: str) -> Optional[int]:
    return {
        "BayerRG8": cv2.COLOR_BayerRG2BGR,
        "BayerGB8": cv2.COLOR_BayerGB2BGR,
        "BayerGR8": cv2.COLOR_BayerGR2BGR,
        "BayerBG8": cv2.COLOR_BayerBG2BGR,
    }.get(str(pixel_format))


def _read_enum_symbols(camera, name: str) -> set[str]:
    if not hasattr(camera, name):
        return set()
    node = getattr(camera, name)
    try:
        if hasattr(node, "GetSymbolics"):
            return {str(symbol) for symbol in node.GetSymbolics()}
    except Exception:
        return set()
    return set()


def _read_node_value(camera, *names: str):
    for name in names:
        if not hasattr(camera, name):
            continue
        node = getattr(camera, name)
        try:
            if hasattr(node, "GetValue"):
                return node.GetValue()
        except Exception:
            continue
    return None


def _read_basler_fps(camera, *, fallback: int) -> int:
    for name in ("ResultingFrameRate", "AcquisitionResultingFrameRate", "AcquisitionFrameRate"):
        if not hasattr(camera, name):
            continue
        node = getattr(camera, name)
        try:
            value = float(node.GetValue())
        except Exception:
            continue
        if np.isfinite(value) and value > 0:
            return int(round(value))
    return int(fallback)


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    raise RuntimeError(f"Unsupported camera frame shape: {arr.shape}")


__all__ = [
    "BaslerCameraSource",
    "CameraFrame",
    "CameraSource",
    "RealSenseCameraSource",
    "create_camera_source",
]
