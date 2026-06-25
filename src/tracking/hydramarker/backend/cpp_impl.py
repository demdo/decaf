"""
C++ backend wrapper for HydraMarker.

This module adapts the pybind11 C++ module to the Python interface used by the
rest of the project. The goal is to keep the public Python code independent of
the exact C++ binding details.
"""

from pathlib import Path
import importlib.util
import os

import numpy as np


def _load_hydramarker_cpp():
    this_dir = Path(__file__).resolve().parent
    pyd_dir = this_dir.parent / "cpp" / "build" / "Release"

    if not pyd_dir.exists():
        raise ImportError(f"Build directory does not exist: {pyd_dir}")

    # Required on Windows so that dependent DLLs
    # (OpenCV, protobuf, etc.) can be found.
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(pyd_dir))

    matches = sorted(pyd_dir.glob("hydramarker_cpp*.pyd"))

    if not matches:
        raise ImportError(
            f"Could not find hydramarker_cpp .pyd in {pyd_dir}"
        )

    pyd_path = matches[0]

    spec = importlib.util.spec_from_file_location(
        "hydramarker_cpp",
        pyd_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load hydramarker_cpp from {pyd_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


_hm = _load_hydramarker_cpp()


def __getattr__(name):
    return getattr(_hm, name)


MarkerField = _hm.MarkerField
CheckerboardDetector = _hm.CheckerboardDetector
TrackerConfig = _hm.TrackerConfig
TrackerEngine = _hm.TrackerEngine
TrackerFrameResult = _hm.TrackerFrameResult
TrackerMode = _hm.TrackerMode
PoseSource = _hm.PoseSource
PoseTrackPoint = _hm.PoseTrackPoint
MapPoseResult = _hm.MapPoseResult
MapPoseTrackerConfig = _hm.MapPoseTrackerConfig
MapPoseTracker = _hm.MapPoseTracker
GlobalCornerIdentity = _hm.GlobalCornerIdentity
PersistentTrackerCorner = _hm.TrackerCorner
TrackerCorner = _hm.TrackerCorner
PersistentMatchStats = _hm.PersistentMatchStats
PersistentMatchResult = _hm.PersistentMatchResult
PersistentMatcher = _hm.PersistentMatcher

_MarkerField = _hm.MarkerField
_generate_planar_field = _hm.generate_planar_field


def tracker_config_from_python(config=None):
    """Copy matching Python TrackerConfig fields into the C++ config object."""
    cpp_config = TrackerConfig()
    if config is None:
        return cpp_config

    for name in dir(cpp_config):
        if name.startswith("_"):
            continue
        if not hasattr(config, name):
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
    """Create the experimental C++ TrackerEngine with Python config overrides."""
    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist_arr = (
        None
        if dist_coeffs is None
        else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    )
    return TrackerEngine(
        field_path,
        marker_json_path,
        K_arr,
        dist_arr,
        tracker_config_from_python(config),
    )


def map_pose_config_from_python(config=None):
    """Copy Python pose-tracker settings into the C++ MapPoseTracker config."""
    cpp_config = MapPoseTrackerConfig()
    if config is None:
        return cpp_config

    direct_names = (
        "min_points",
        "min_inliers",
        "ransac_reproj_px",
        "ransac_confidence",
        "ransac_iterations",
        "max_mean_reproj_px",
        "max_max_reproj_px",
        "max_translation_jump_mm",
        "max_rotation_jump_deg",
        "rotation_gate_scale_per_lost_frame",
        "rotation_gate_max_deg",
        "use_pose_prior",
        "refine_with_iterative",
        "use_direct_prior_solver",
        "direct_refine_method",
        "direct_max_mean_reproj_px",
        "direct_max_max_reproj_px",
    )
    for name in direct_names:
        if hasattr(config, name):
            setattr(cpp_config, name, getattr(config, name))

    tracker_config_map = {
        "pnp_ransac_reprojection_px": "ransac_reproj_px",
        "pnp_ransac_confidence": "ransac_confidence",
        "pnp_ransac_iterations": "ransac_iterations",
        "max_mean_reprojection_error_px": "max_mean_reproj_px",
        "max_max_reprojection_error_px": "max_max_reproj_px",
        "pnp_direct_prior_enabled": "use_direct_prior_solver",
        "pnp_direct_refine_method": "direct_refine_method",
        "pnp_direct_max_mean_reprojection_error_px": "direct_max_mean_reproj_px",
        "pnp_direct_max_max_reprojection_error_px": "direct_max_max_reproj_px",
    }
    for py_name, cpp_name in tracker_config_map.items():
        if hasattr(config, py_name):
            setattr(cpp_config, cpp_name, getattr(config, py_name))

    return cpp_config


def pose_track_point_from_python(point):
    """Convert a Python PoseTrackPoint-like object into the C++ point type."""
    cpp_point = PoseTrackPoint()
    cpp_point.global_row = int(point.global_row)
    cpp_point.global_col = int(point.global_col)
    cpp_point.xyz_mm = tuple(float(v) for v in point.xyz_mm)
    cpp_point.uv = tuple(float(v) for v in point.uv)
    cpp_point.votes = int(getattr(point, "votes", 0))
    return cpp_point


def pose_track_points_from_python(points):
    return [pose_track_point_from_python(point) for point in points]


def global_corner_identity_from_python(corner):
    """Convert a Python TrackerCorner-like object into a C++ identity."""
    identity = GlobalCornerIdentity()
    identity.global_row = int(corner.global_row)
    identity.global_col = int(corner.global_col)
    identity.xyz_mm = tuple(float(v) for v in corner.xyz_mm)

    uv_source = getattr(corner, "uv", None)
    if uv_source is None:
        uv_source = getattr(corner, "uv_px")
    identity.uv = tuple(float(v) for v in uv_source)
    identity.votes = int(getattr(corner, "votes", 0))
    return identity


def global_corner_identities_from_python(corners):
    return [global_corner_identity_from_python(corner) for corner in corners]


def create_map_pose_tracker(K, dist_coeffs=None, config=None):
    """Create the isolated C++ MapPoseTracker with Python config overrides."""
    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist_arr = (
        None
        if dist_coeffs is None
        else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    )
    return MapPoseTracker(
        K_arr,
        dist_arr,
        map_pose_config_from_python(config),
    )


def create_persistent_matcher(config=None):
    """Create the isolated C++ PersistentMatcher with Python config overrides."""
    return PersistentMatcher(tracker_config_from_python(config))


class MarkerFieldCpp:
    """
    Thin Python wrapper around the C++ MarkerField implementation.
    """

    def __init__(self, path: str):
        self._mf = _MarkerField(path)

    def find_patch(self, patch):
        return [
            {
                "x": match.x,
                "y": match.y,
                "rotation": match.rotation_deg,
            }
            for match in self._mf.find_patch(patch)
        ]


def generate_planar_field(
    rows: int,
    cols: int,
    patch_size: int,
    max_ms: float = 60000.0,
    max_trial: int = 100000,
    is_print: bool = False,
) -> np.ndarray:
    field = _generate_planar_field(
        rows=rows,
        cols=cols,
        patch_size=patch_size,
        max_ms=max_ms,
        max_trial=max_trial,
        is_print=is_print,
    )

    return np.asarray(field, dtype=np.uint8)
