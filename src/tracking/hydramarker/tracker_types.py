"""Shared data structures for HydraMarker tracking results and state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


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
    """Frame-local checkerboard corner observed by the detector."""

    local_row: int
    local_col: int
    uv: Tuple[float, float]


@dataclass
class TrackerCorner:
    """Matched marker corner with frame-local and global marker identity."""

    local_row: int
    local_col: int
    global_row: int
    global_col: int
    xyz_mm: Tuple[float, float, float]
    uv: Tuple[float, float]
    votes: int = 0


@dataclass
class FastPathDebug:
    """Diagnostics for persistent fast-path and dense-refine decisions."""

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
    """Complete public result produced by :class:`HydraTracker` for one frame."""

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
    depth_filter_applied: bool = False
    depth_filter_delta_z_mm: float = 0.0
    depth_filter_raw_z_mm: float = float("nan")
    depth_filter_z_mm: float = float("nan")
    depth_filter_reproj_excess_px: float = float("nan")
    depth_filter_guard_alpha: float = 1.0
    depth_filter_innovation_z_mm: float = 0.0
    depth_filter_innovation_mean_z_mm: float = 0.0
    depth_filter_innovation_cusum_pos_mm: float = 0.0
    depth_filter_innovation_cusum_neg_mm: float = 0.0
    depth_filter_innovation_bias_detected: bool = False
    depth_filter_innovation_bias_direction: int = 0
    depth_filter_innovation_bias_limited: bool = False
    depth_filter_object_z_span_mm: float = float("nan")
    depth_filter_negative_delta_guard_limited: bool = False
    pose_plateau_prior_triggered: bool = False
    pose_plateau_prior_attempted: bool = False
    pose_plateau_prior_applied: bool = False
    pose_plateau_prior_method: str = ""
    pose_plateau_prior_reason: str = ""
    pose_plateau_prior_delta_z_mm: float = float("nan")
    pose_plateau_prior_reproj_excess_px: float = float("nan")
    pose_plateau_prior_max_reproj_excess_px: float = float("nan")
    pose_plateau_prior_iterations: int = 0
    fast_path_debug: FastPathDebug = field(default_factory=FastPathDebug)
    timings_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class PersistentMatchStats:
    """Counters describing how persistent identities matched a current detection."""

    age: int = 0
    identities: int = 0
    current_corners: int = 0
    accepted: int = 0
    used_pose_projection: bool = False
    adaptive_motion_px: float = 0.0
    adaptive_max_dist_px: float = 0.0
    rejected_no_projection: int = 0
    rejected_far: int = 0
    rejected_ambiguous: int = 0
    rejected_claimed: int = 0


@dataclass
class GeometryCornerCache:
    """Vectorized cache of valid marker geometry corners."""

    rows: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    cols: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    xyz_mm: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))


@dataclass
class DenseProjectionMatchStats:
    """Quality metrics for dense projected-corner matching."""

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


GridKey = Tuple[int, int]


@dataclass
class GlobalCornerIdentity:
    """Persistent identity for one global marker corner."""

    global_row: int
    global_col: int

    xyz_mm: Tuple[float, float, float]
    uv: Tuple[float, float]

    votes: int = 0


class IdentityStore:
    """
    Persistent global identity storage.

    IMPORTANT:
    Local checkerboard indices are NOT persistent semantic identities.
    They may change during recovery / lattice refit / tracking updates.

    Therefore:
        - only global IDs are stored persistently
        - local IDs are frame-local only
    """

    def __init__(self) -> None:
        self._items: List[GlobalCornerIdentity] = []

        # Persistent storage ONLY by global key.
        self._by_global: Dict[GridKey, GlobalCornerIdentity] = {}

    def clear(self) -> None:
        self._items.clear()
        self._by_global.clear()

    def empty(self) -> bool:
        return len(self._items) == 0

    def __len__(self) -> int:
        return len(self._items)

    def all(self) -> List[GlobalCornerIdentity]:
        return list(self._items)

    def by_global(self) -> Dict[GridKey, GlobalCornerIdentity]:
        return dict(self._by_global)

    def get_global(
        self,
        global_row: int,
        global_col: int,
    ) -> Optional[GlobalCornerIdentity]:
        return self._by_global.get(
            (int(global_row), int(global_col))
        )

    def has_global(
        self,
        global_row: int,
        global_col: int,
    ) -> bool:
        return (
            int(global_row),
            int(global_col),
        ) in self._by_global

    def global_keys(self) -> set[GridKey]:
        return set(self._by_global.keys())

    def replace(
        self,
        identities: Iterable[GlobalCornerIdentity],
    ) -> None:
        self.clear()
        self.merge(identities)

    def merge(
        self,
        identities: Iterable[GlobalCornerIdentity],
    ) -> None:
        """
        Merge ONLY by global key.

        Never persist local checkerboard indices.
        """

        for p in identities:
            global_key = (
                int(p.global_row),
                int(p.global_col),
            )

            existing = self._by_global.get(global_key)

            if existing is None:
                self._by_global[global_key] = p
                continue

            # Keep higher-vote observation.
            if int(p.votes) >= int(existing.votes):
                self._by_global[global_key] = p

        self._items = list(self._by_global.values())


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
    "GridKey",
    "GlobalCornerIdentity",
    "IdentityStore",
]
