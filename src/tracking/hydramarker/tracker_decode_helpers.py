from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tracking.hydramarker.identity_store import GlobalCornerIdentity
from tracking.hydramarker.map_pose_tracker import PoseTrackPoint
from tracking.hydramarker.tracker_types import (
    DenseProjectionMatchStats,
    DetectedCorner,
    FastPathDebug,
    PersistentMatchStats,
    PoseSource,
    TrackerCorner,
    TrackerMode,
    TrackerResult,
)
from tracking.pose_solvers import make_transform_from_rvec_tvec


class DecodeHelperMixin:
    def _note_decode_topology_failure(self, dots, patches) -> None:
        if len(patches) > 0:
            self._undecodeable_detection_frames = 0
            return

        dot_cells = list(getattr(dots, "cells", []))
        if not dot_cells:
            self._undecodeable_detection_frames = 0
            return

        min_span = max(1, int(self.config.checker_min_tracking_decode_cell_span))
        rows = [int(getattr(c, "row", 0)) for c in dot_cells]
        cols = [int(getattr(c, "col", 0)) for c in dot_cells]
        row_span = max(rows) - min(rows) + 1 if rows else 0
        col_span = max(cols) - min(cols) + 1 if cols else 0

        if row_span >= min_span and col_span >= min_span:
            self._undecodeable_detection_frames = 0
            return

        self._undecodeable_detection_frames += 1
        if (
            self._undecodeable_detection_frames
            > self.config.checker_max_undecodeable_tracking_frames
        ):
            self._force_local_recovery()

    @staticmethod
    def _correspondence_failure_message(num_points: int, corr_result) -> str:
        return (
            f"Too few correspondences: {num_points} "
            f"(patches_used={int(getattr(corr_result, 'decoded_patches_used', 0))}, "
            f"rot_rejected={int(getattr(corr_result, 'decoded_patches_rejected_by_rotation', 0))}, "
            f"assign_total={int(getattr(corr_result, 'assignments_total', 0))}, "
            f"assign_accepted={int(getattr(corr_result, 'assignments_accepted', 0))}, "
            f"conflicted={int(getattr(corr_result, 'assignments_conflicted', 0))}, "
            f"no_geom={int(getattr(corr_result, 'corners_without_geometry', 0))}, "
            f"single_boundary={int(getattr(corr_result, 'single_vote_boundary_corners_accepted', 0))}, "
            f"single_non_boundary_rej={int(getattr(corr_result, 'single_vote_non_boundary_corners_rejected', 0))}, "
            f"rot={int(getattr(corr_result, 'dominant_rotation_deg', -1))}/"
            f"{int(getattr(corr_result, 'dominant_rotation_count', 0))}/"
            f"{int(getattr(corr_result, 'rotation_vote_count', 0))})"
        )

    @staticmethod
    def _decode_failure_message(dots, patches, decoded) -> str:
        dot_cells = list(getattr(dots, "cells", []))
        dot_cell_count = len(dot_cells)
        dot_valid_count = sum(
            1 for c in dot_cells
            if bool(getattr(c, "valid", False))
        )
        dot_ambiguous_count = sum(
            1 for c in dot_cells
            if bool(getattr(c, "ambiguous", False))
        )
        dot_cache_reused_count = sum(
            1 for c in dot_cells
            if bool(getattr(c, "cache_reused", False))
        )
        dot_rows = int(getattr(dots, "rows", 0))
        dot_cols = int(getattr(dots, "cols", 0))

        patch_count = len(patches)
        decoded_count = len(decoded)
        invalid_geometry = sum(
            1 for p in decoded
            if getattr(p, "local", None) is not None
            and getattr(p.local, "valid", False)
            and not getattr(p.local, "geometry_valid", False)
        )
        ambiguous = sum(
            1 for p in decoded
            if getattr(p, "ambiguous", False)
        )
        matched_but_rejected = sum(
            1 for p in decoded
            if int(getattr(p, "num_matches", 0)) > 0 and not getattr(p, "valid", False)
        )

        return (
            "No valid decoded patches "
            f"(cells={dot_cell_count}, valid_cells={dot_valid_count}, "
            f"ambig_cells={dot_ambiguous_count}, cached_cells={dot_cache_reused_count}, "
            f"grid={dot_rows}x{dot_cols}, patches={patch_count}, decoded={decoded_count}, "
            f"bad_geom={invalid_geometry}, ambiguous={ambiguous}, "
            f"matched_rejected={matched_but_rejected})"
        )

