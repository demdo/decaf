"""Python facade for the native HydraMarker tracker engine.

The tracker implementation lives in the pybind11 extension built from
``hydramarker/cpp``.  This module is intentionally small: it locates the native
extension, converts ``TrackerConfig`` into the matching C++ configuration
object, and exposes ``HydraTracker`` as the Python-facing runtime wrapper.

Python code should treat this file as the only loader for tracker runtime
objects. Detector, decoder, persistence, pose estimation, and hold logic all
execute inside C++; Python only passes configuration in and converts the native
frame result into lightweight Python data containers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import importlib.util
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tracking.hydramarker.config import TrackerConfig


def _load_hydramarker_cpp():
    this_dir = Path(__file__).resolve().parent
    pyd_dir = this_dir / "cpp" / "build" / "Release"

    if not pyd_dir.exists():
        raise ImportError(f"Build directory does not exist: {pyd_dir}")

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(pyd_dir))

    matches = sorted(pyd_dir.glob("hydramarker_cpp*.pyd"))
    if not matches:
        raise ImportError(f"Could not find hydramarker_cpp .pyd in {pyd_dir}")

    pyd_path = matches[0]
    spec = importlib.util.spec_from_file_location("hydramarker_cpp", pyd_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load hydramarker_cpp from {pyd_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hydramarker_cpp = _load_hydramarker_cpp()
hm = hydramarker_cpp


def __getattr__(name):
    return getattr(hydramarker_cpp, name)


def tracker_config_from_python(config=None):
    cpp_config = hydramarker_cpp.TrackerConfig()
    if config is None:
        return cpp_config

    for name in dir(cpp_config):
        if name.startswith("_") or not hasattr(config, name):
            continue
        try:
            setattr(cpp_config, name, getattr(config, name))
        except (AttributeError, TypeError):
            pass
    return cpp_config


def create_tracker_engine(
    field_path: str,
    marker_json_path: str,
    K,
    dist_coeffs=None,
    config=None,
):
    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist_arr = (
        None
        if dist_coeffs is None
        else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    )
    return hydramarker_cpp.TrackerEngine(
        field_path,
        marker_json_path,
        K_arr,
        dist_arr,
        tracker_config_from_python(config),
    )


def create_persistent_matcher(config=None):
    return hydramarker_cpp.PersistentMatcher(tracker_config_from_python(config))


class MarkerFieldCpp:
    def __init__(self, path: str):
        self._field = hydramarker_cpp.MarkerField(path)

    def find_patch(self, patch):
        return [
            {
                "x": match.x,
                "y": match.y,
                "rotation": match.rotation_deg,
            }
            for match in self._field.find_patch(patch)
        ]


def generate_planar_field(
    rows: int,
    cols: int,
    patch_size: int,
    max_ms: float = 60000.0,
    max_trial: int = 100000,
    is_print: bool = False,
) -> np.ndarray:
    field = hydramarker_cpp.generate_planar_field(
        rows=rows,
        cols=cols,
        patch_size=patch_size,
        max_ms=max_ms,
        max_trial=max_trial,
        is_print=is_print,
    )
    return np.asarray(field, dtype=np.uint8)


class TrackerMode(str, Enum):
    """High-level tracker state reported to callers."""

    LOST = "LOST"
    DETECTING = "DETECTING"
    TRACKING = "TRACKING"
    RECOVERING = "RECOVERING"


class PoseSource(str, Enum):
    """Origin of the pose returned for a processed frame."""

    NONE = "none"
    DECODE = "decode"
    PERSISTENT = "persistent"
    FAST_PERSISTENT = "fast_persistent"
    UNCODED_GRID = "uncoded_grid"
    HOLD = "hold"


@dataclass
class DetectedCorner:
    """Frame-local checkerboard corner observed by the C++ detector."""

    local_row: int
    local_col: int
    uv: Tuple[float, float]
    visibility_score: float = 1.0
    observed_frames: int = 0
    predicted: bool = False


@dataclass
class TrackerCorner:
    """Visible marker corner with frame-local and global marker identity."""

    local_row: int
    local_col: int
    global_row: int
    global_col: int
    xyz_mm: Tuple[float, float, float]
    uv: Tuple[float, float]
    votes: int = 0
    visibility_score: float = 1.0
    observed_frames: int = 0
    predicted: bool = False


@dataclass
class TrackerResult:
    """Minimal Python result converted from one C++ TrackerFrameResult."""

    success: bool
    mode: TrackerMode
    message: str = ""

    detection_valid: bool = False
    detection_tracking: bool = False
    detection_stable: bool = False
    detection_corners: List[DetectedCorner] = field(default_factory=list)

    corners: List[TrackerCorner] = field(default_factory=list)
    correspondence_corners: List[TrackerCorner] = field(default_factory=list)

    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    T_marker_camera: Optional[np.ndarray] = None

    mean_reprojection_error_px: float = -1.0
    max_reprojection_error_px: float = -1.0

    num_points: int = 0
    num_inliers: int = 0
    confidence: float = 0.0
    pose_source: PoseSource = PoseSource.NONE

    debug_counters: Dict[str, float] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)


class _PoseState:
    """Read-only mirror of the pose state owned by the C++ engine."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None
        self.T_marker_camera: Optional[np.ndarray] = None

    def set(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        T_marker_camera: Optional[np.ndarray],
    ) -> None:
        self.rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy()
        self.tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy()
        self.T_marker_camera = (
            None
            if T_marker_camera is None
            else np.asarray(T_marker_camera, dtype=np.float64).reshape(4, 4).copy()
        )


class HydraTracker:
    """Expose TrackerEngine through the existing Python tracker API."""

    def __init__(
        self,
        field_path: str,
        marker_json_path: str,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        config: Optional[TrackerConfig] = None,
    ) -> None:
        self.config = config or TrackerConfig()
        self._field_path = str(field_path)
        self._marker_json_path = str(marker_json_path)

        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = (
            np.zeros((0, 1), dtype=np.float64)
            if dist_coeffs is None
            else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        )

        # Exposed for diagnostics/rendering only. Tracking itself is owned by
        # TrackerEngine; these objects are not Python fallback components.
        self.field = hm.MarkerField.loadFromFile(self._field_path)
        self.geometry = hm.MarkerGeometry.load_from_json(self._marker_json_path)

        self.pose_tracker = _PoseState()
        self.mode = TrackerMode.LOST
        self.frame_index = 0
        self.lost_frames = 0
        self._max_pts_seen = 0
        self._last_good_reproj_px = -1.0
        self._last_accepted_rvec: Optional[np.ndarray] = None
        self._last_accepted_tvec: Optional[np.ndarray] = None
        self._last_accepted_T_marker_camera: Optional[np.ndarray] = None
        self._last_accepted_pose_frame = -1
        self._last_debug_counters: Dict[str, float] = {}
        self._persistent_corners: list[object] = []

        try:
            self._engine = create_tracker_engine(
                self._field_path,
                self._marker_json_path,
                self.K,
                self.dist_coeffs,
                self.config,
            )
        except Exception as exc:
            raise RuntimeError(
                "HydraTracker requires the C++ TrackerEngine. Build/load failed."
            ) from exc

    @property
    def rvec(self) -> Optional[np.ndarray]:
        return self.pose_tracker.rvec

    @property
    def tvec(self) -> Optional[np.ndarray]:
        return self.pose_tracker.tvec

    @property
    def T_marker_camera(self) -> Optional[np.ndarray]:
        return self.pose_tracker.T_marker_camera

    @staticmethod
    def _cpp_enum_name(value) -> str:
        name = getattr(value, "name", None)
        if name:
            return str(name).upper()
        text = str(value)
        if "." in text:
            text = text.rsplit(".", 1)[-1]
        return text.upper()

    def _mode_from_cpp(self, value) -> TrackerMode:
        name = self._cpp_enum_name(value)
        return TrackerMode.__members__.get(name, self.mode)

    @staticmethod
    def _pose_source_from_cpp(value) -> PoseSource:
        name = HydraTracker._cpp_enum_name(value)
        return PoseSource.__members__.get(name, PoseSource.NONE)

    @staticmethod
    def _cpp_vector_to_array(values, shape):
        arr = np.asarray(values, dtype=np.float64)
        expected = int(np.prod(shape))
        if arr.size != expected:
            return None
        return arr.reshape(shape).copy()

    @staticmethod
    def _cpp_float_attr(obj, *names: str, default: float = 0.0) -> float:
        for name in names:
            if hasattr(obj, name):
                try:
                    return float(getattr(obj, name))
                except (TypeError, ValueError):
                    return float(default)
        return float(default)

    @staticmethod
    def _cpp_bool_counter(obj, name: str) -> float:
        return 1.0 if bool(getattr(obj, name, False)) else 0.0

    @staticmethod
    def _detected_corners_from_cpp(corners) -> list[DetectedCorner]:
        if corners is None:
            return []
        return [
            DetectedCorner(
                local_row=int(corner.local_row),
                local_col=int(corner.local_col),
                uv=tuple(float(v) for v in corner.uv),
                visibility_score=float(getattr(corner, "visibility_score", 1.0)),
                observed_frames=int(getattr(corner, "observed_frames", 0)),
                predicted=bool(getattr(corner, "predicted", False)),
            )
            for corner in corners
        ]

    @staticmethod
    def _tracker_corners_from_cpp(corners) -> list[TrackerCorner]:
        if corners is None:
            return []
        return [
            TrackerCorner(
                local_row=int(corner.local_row),
                local_col=int(corner.local_col),
                global_row=int(corner.global_row),
                global_col=int(corner.global_col),
                xyz_mm=tuple(float(v) for v in corner.xyz_mm),
                uv=tuple(float(v) for v in corner.uv),
                votes=int(getattr(corner, "votes", 0)),
                visibility_score=float(getattr(corner, "visibility_score", 1.0)),
                observed_frames=int(getattr(corner, "observed_frames", 0)),
                predicted=bool(getattr(corner, "predicted", False)),
            )
            for corner in corners
        ]

    def _tracker_result_from_cpp_engine(self, cpp_result) -> TrackerResult:
        mode = self._mode_from_cpp(getattr(cpp_result, "mode", None))
        pose_source = self._pose_source_from_cpp(
            getattr(cpp_result, "pose_source", None)
        )
        rvec = self._cpp_vector_to_array(getattr(cpp_result, "rvec", []), (3, 1))
        tvec = self._cpp_vector_to_array(getattr(cpp_result, "tvec", []), (3, 1))
        T_marker_camera = self._cpp_vector_to_array(
            getattr(cpp_result, "T_marker_camera", []),
            (4, 4),
        )

        persistent_count = int(getattr(cpp_result, "persistent_count", 0))
        debug_counters = {
            "persistent_count": float(persistent_count),
            "fast_attempted": self._cpp_bool_counter(cpp_result, "fast_attempted"),
            "fast_success": self._cpp_bool_counter(cpp_result, "fast_success"),
            "fast_matches": float(getattr(cpp_result, "fast_matches", 0)),
            "fast_identities": float(persistent_count),
            "fast_current_corners": float(
                getattr(cpp_result, "detection_corner_count", 0)
            ),
            "fast_dense_attempted": self._cpp_bool_counter(
                cpp_result,
                "fast_dense_attempted",
            ),
            "fast_dense_success": self._cpp_bool_counter(
                cpp_result,
                "fast_dense_success",
            ),
            "fast_dense_matches": float(
                getattr(cpp_result, "fast_dense_matches", 0)
            ),
            "fast_dense_median_px": self._cpp_float_attr(
                cpp_result,
                "fast_dense_median_px",
                "fast_dense_median_error_px",
                default=-1.0,
            ),
            "fast_dense_p90_px": self._cpp_float_attr(
                cpp_result,
                "fast_dense_p90_px",
                "fast_dense_p90_error_px",
                default=-1.0,
            ),
            "fast_dense_projected": self._cpp_float_attr(
                cpp_result,
                "fast_dense_projected",
            ),
            "fast_dense_detected": self._cpp_float_attr(
                cpp_result,
                "fast_dense_detected",
            ),
            "fast_dense_rejected_no_projection": self._cpp_float_attr(
                cpp_result,
                "fast_dense_rejected_no_projection",
            ),
            "fast_dense_rejected_far": self._cpp_float_attr(
                cpp_result,
                "fast_dense_rejected_far",
            ),
            "fast_dense_rejected_ambiguous": self._cpp_float_attr(
                cpp_result,
                "fast_dense_rejected_ambiguous",
            ),
            "fast_dense_rejected_non_mutual": self._cpp_float_attr(
                cpp_result,
                "fast_dense_rejected_non_mutual",
            ),
            "fast_dense_image_coverage": self._cpp_float_attr(
                cpp_result,
                "fast_dense_image_coverage",
                default=-1.0,
            ),
            "fast_dense_image_span_u_px": self._cpp_float_attr(
                cpp_result,
                "fast_dense_image_span_u_px",
                default=-1.0,
            ),
            "fast_dense_image_span_v_px": self._cpp_float_attr(
                cpp_result,
                "fast_dense_image_span_v_px",
                default=-1.0,
            ),
            "fast_dense_object_span_mm": self._cpp_float_attr(
                cpp_result,
                "fast_dense_object_span_mm",
                default=-1.0,
            ),
            "fast_dense_distinct_rows": self._cpp_float_attr(
                cpp_result,
                "fast_dense_distinct_rows",
            ),
            "fast_dense_distinct_cols": self._cpp_float_attr(
                cpp_result,
                "fast_dense_distinct_cols",
            ),
            "fast_rejected_no_projection": self._cpp_float_attr(
                cpp_result,
                "fast_rejected_no_projection",
            ),
            "fast_rejected_far": self._cpp_float_attr(
                cpp_result,
                "fast_rejected_far",
            ),
            "fast_rejected_ambiguous": self._cpp_float_attr(
                cpp_result,
                "fast_rejected_ambiguous",
            ),
            "fast_rejected_claimed": self._cpp_float_attr(
                cpp_result,
                "fast_rejected_claimed",
            ),
        }

        timings = {
            str(key): float(value)
            for key, value in dict(getattr(cpp_result, "timings_ms", {}) or {}).items()
        }
        timings["cpp_tracker_engine_count"] = 1.0
        timings["cpp_tracker_engine_current_pose_accepted_count"] = (
            1.0 if bool(getattr(cpp_result, "current_pose_accepted", False)) else 0.0
        )
        timings["cpp_tracker_engine_has_accepted_pose_count"] = (
            1.0 if bool(getattr(cpp_result, "has_accepted_pose", False)) else 0.0
        )

        result = TrackerResult(
            success=bool(getattr(cpp_result, "success", False)),
            mode=mode,
            message=str(getattr(cpp_result, "message", "")),
            detection_valid=bool(getattr(cpp_result, "detection_valid", False)),
            detection_tracking=bool(
                getattr(cpp_result, "detection_tracking", False)
            ),
            detection_stable=bool(getattr(cpp_result, "detection_stable", False)),
            detection_corners=self._detected_corners_from_cpp(
                getattr(cpp_result, "detection_corners", [])
            ),
            corners=self._tracker_corners_from_cpp(
                getattr(cpp_result, "corners", [])
            ),
            correspondence_corners=self._tracker_corners_from_cpp(
                getattr(cpp_result, "correspondence_corners", [])
            ),
            rvec=rvec,
            tvec=tvec,
            T_marker_camera=T_marker_camera,
            mean_reprojection_error_px=float(
                getattr(cpp_result, "mean_reprojection_error_px", -1.0)
            ),
            max_reprojection_error_px=float(
                getattr(cpp_result, "max_reprojection_error_px", -1.0)
            ),
            num_points=int(getattr(cpp_result, "num_points", 0)),
            num_inliers=int(getattr(cpp_result, "num_inliers", 0)),
            confidence=float(getattr(cpp_result, "confidence", 0.0)),
            pose_source=pose_source,
            debug_counters=debug_counters,
            timings_ms=timings,
        )

        self._sync_public_state_from_cpp(
            cpp_result,
            mode,
            debug_counters,
            persistent_count,
        )
        return result

    def _sync_public_state_from_cpp(
        self,
        cpp_result,
        mode: TrackerMode,
        debug_counters: Dict[str, float],
        persistent_count: int,
    ) -> None:
        self.mode = mode
        self.frame_index = int(getattr(cpp_result, "frame_index", self.frame_index))
        self.lost_frames = int(getattr(cpp_result, "lost_frames", self.lost_frames))
        self._last_debug_counters = dict(debug_counters)
        self._persistent_corners = [None] * max(0, int(persistent_count))

        pose_state_rvec = self._cpp_vector_to_array(
            getattr(cpp_result, "pose_tracker_rvec", []),
            (3, 1),
        )
        pose_state_tvec = self._cpp_vector_to_array(
            getattr(cpp_result, "pose_tracker_tvec", []),
            (3, 1),
        )
        pose_state_T = self._cpp_vector_to_array(
            getattr(cpp_result, "pose_tracker_T_marker_camera", []),
            (4, 4),
        )
        if (
            bool(getattr(cpp_result, "pose_tracker_has_pose", False))
            and pose_state_rvec is not None
            and pose_state_tvec is not None
        ):
            self.pose_tracker.set(pose_state_rvec, pose_state_tvec, pose_state_T)
        else:
            self.pose_tracker.reset()

        self._max_pts_seen = int(
            getattr(cpp_result, "max_pts_seen", self._max_pts_seen)
        )
        self._last_good_reproj_px = float(
            getattr(cpp_result, "last_good_reproj_px", self._last_good_reproj_px)
        )

        accepted_rvec = self._cpp_vector_to_array(
            getattr(cpp_result, "accepted_rvec", []),
            (3, 1),
        )
        accepted_tvec = self._cpp_vector_to_array(
            getattr(cpp_result, "accepted_tvec", []),
            (3, 1),
        )
        accepted_T = self._cpp_vector_to_array(
            getattr(cpp_result, "accepted_T_marker_camera", []),
            (4, 4),
        )
        if (
            bool(getattr(cpp_result, "has_accepted_pose", False))
            and accepted_rvec is not None
            and accepted_tvec is not None
        ):
            self._last_accepted_rvec = accepted_rvec.copy()
            self._last_accepted_tvec = accepted_tvec.copy()
            self._last_accepted_T_marker_camera = (
                None if accepted_T is None else accepted_T.copy()
            )
            self._last_accepted_pose_frame = int(
                getattr(cpp_result, "accepted_pose_frame", self.frame_index)
            )
        else:
            self._last_accepted_rvec = None
            self._last_accepted_tvec = None
            self._last_accepted_T_marker_camera = None
            self._last_accepted_pose_frame = -1

    def reset(self) -> None:
        """Reset C++ runtime state and the mirrored public state."""
        self._engine.reset()
        self.pose_tracker.reset()
        self.mode = TrackerMode.LOST
        self.frame_index = 0
        self.lost_frames = 0
        self._max_pts_seen = 0
        self._last_good_reproj_px = -1.0
        self._last_accepted_rvec = None
        self._last_accepted_tvec = None
        self._last_accepted_T_marker_camera = None
        self._last_accepted_pose_frame = -1
        self._last_debug_counters = {}
        self._persistent_corners = []

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        run_detection: bool = True,
    ) -> TrackerResult:
        """Process one camera frame through the C++ TrackerEngine."""
        try:
            cpp_result = self._engine.process_frame(
                np.asarray(frame, dtype=np.uint8),
                bool(run_detection),
            )
        except Exception as exc:
            raise RuntimeError("C++ TrackerEngine.process_frame failed.") from exc

        return self._tracker_result_from_cpp_engine(cpp_result)
