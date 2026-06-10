from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class TrackerMode(str, Enum):
    LOST = "LOST"
    DETECTING = "DETECTING"
    TRACKING = "TRACKING"
    RECOVERING = "RECOVERING"


class PoseSource(str, Enum):
    NONE = "none"
    DECODE = "decode"
    PERSISTENT = "persistent"
    FAST_PERSISTENT = "fast_persistent"
    UNCODED_GRID = "uncoded_grid"
    HOLD = "hold"


@dataclass
class DetectedCorner:
    local_row: int
    local_col: int
    uv: Tuple[float, float]


@dataclass
class TrackerCorner:
    local_row: int
    local_col: int
    global_row: int
    global_col: int
    xyz_mm: Tuple[float, float, float]
    uv: Tuple[float, float]
    votes: int = 0


@dataclass
class FastPathDebug:
    attempted: bool = False
    success: bool = False
    reason: str = ""
    matches: int = 0
    identities: int = 0
    current_corners: int = 0
    used_pose_projection: bool = False
    rejected_no_projection: int = 0
    rejected_far: int = 0
    rejected_ambiguous: int = 0
    rejected_claimed: int = 0
    dense_refine_attempted: bool = False
    dense_refine_success: bool = False
    dense_refine_reason: str = ""
    dense_refine_matches: int = 0
    dense_refine_median_error_px: float = -1.0
    dense_refine_p90_error_px: float = -1.0
    dense_refine_projected: int = 0
    dense_refine_detected: int = 0
    dense_refine_rejected_no_projection: int = 0
    dense_refine_rejected_far: int = 0
    dense_refine_rejected_ambiguous: int = 0
    dense_refine_rejected_non_mutual: int = 0
    dense_refine_image_coverage: float = -1.0
    dense_refine_image_span_u_px: float = -1.0
    dense_refine_image_span_v_px: float = -1.0
    dense_refine_object_span_mm: float = -1.0
    dense_refine_distinct_rows: int = 0
    dense_refine_distinct_cols: int = 0


@dataclass
class TrackerResult:
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
    pnp_method: str = ""
    fast_path_debug: FastPathDebug = field(default_factory=FastPathDebug)
    timings_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class PersistentMatchStats:
    age: int = 0
    identities: int = 0
    current_corners: int = 0
    accepted: int = 0
    used_pose_projection: bool = False
    rejected_no_projection: int = 0
    rejected_far: int = 0
    rejected_ambiguous: int = 0
    rejected_claimed: int = 0


@dataclass
class GeometryCornerCache:
    rows: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    cols: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    xyz_mm: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))


@dataclass
class DenseProjectionMatchStats:
    detected: int = 0
    projected: int = 0
    rejected_no_projection: int = 0
    rejected_far: int = 0
    rejected_ambiguous: int = 0
    rejected_non_mutual: int = 0
    median_error_px: float = float("inf")
    p90_error_px: float = float("inf")
    image_coverage: float = -1.0
    image_span_u_px: float = -1.0
    image_span_v_px: float = -1.0
    object_span_mm: float = -1.0
    distinct_rows: int = 0
    distinct_cols: int = 0


__all__ = [
    "TrackerMode",
    "PoseSource",
    "DetectedCorner",
    "TrackerCorner",
    "FastPathDebug",
    "TrackerResult",
    "PersistentMatchStats",
    "GeometryCornerCache",
    "DenseProjectionMatchStats",
]
