from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tracking.transforms import (
    as_transform,
    invert_transform,
    extract_translation,
)


# ============================================================
# Helpers
# ============================================================

def _pose_tip_to_xray(
    T_xc: np.ndarray,
    T_tc: np.ndarray,
) -> np.ndarray:
    """
    Compose the transform from tip frame to xray frame.

    Convention
    ----------
    T_ab means: transform from frame a to frame b

    Therefore:
    - T_xc : xray  -> camera
    - T_tc : tip   -> camera

    We need:
    - T_cx : camera -> xray = inv(T_xc)
    - T_tx : tip    -> xray

    Composition:
        T_tx = T_cx @ T_tc
    """
    T_xc = as_transform(T_xc, "T_xc")
    T_tc = as_transform(T_tc, "T_tc")

    T_cx = invert_transform(T_xc)

    # T_xc / T_cx come from camera<->xray calibration in meters,
    # while T_tc comes from pointer calibration in millimeters.
    # Convert the translation part of T_cx to millimeters before composition.
    T_cx = T_cx.copy()
    T_cx[:3, 3] *= 1e3   # m -> mm

    T_tx = T_cx @ T_tc
    return T_tx


# ============================================================
# Public API
# ============================================================

@dataclass(frozen=True)
class ExtractDepthResult:
    T_tx: np.ndarray
    tip_xyz_x_mm: np.ndarray
    d_x_mm: float


def extract_depth(
    T_xc: np.ndarray,
    T_tc: np.ndarray,
) -> ExtractDepthResult:
    """
    Express the pointer tip in the xray frame and extract d_x from its z-coordinate.

    Parameters
    ----------
    T_xc : np.ndarray
        4x4 transform from xray frame to camera frame.
    T_tc : np.ndarray
        4x4 transform from tip frame to camera frame.

    Returns
    -------
    ExtractDepthResult
        T_tx         : tip -> xray
        tip_xyz_x_mm : tip origin expressed in xray coordinates [mm]
        d_x_mm       : z-coordinate of the tip in the xray frame [mm]

    Notes
    -----
    With the convention T_ab = "from a to b" and column-vector transforms,

        p_b = T_ab @ p_a

    the tip pose in xray coordinates is

        T_tx = T_cx @ T_tc
             = inv(T_xc) @ T_tc

    Since T_tx maps tip -> xray, its translation is directly the tip origin
    expressed in the xray frame:

        tip_xyz_x_mm = translation(T_tx)

    Under the project convention, d_x is taken directly as the xray-frame
    z-coordinate of the tip:

        d_x_mm = tip_xyz_x_mm[2]
    """
    T_xc = as_transform(T_xc, "T_xc")
    T_tc = as_transform(T_tc, "T_tc")

    T_tx = _pose_tip_to_xray(T_xc=T_xc, T_tc=T_tc)

    tip_xyz_x_mm = extract_translation(T_tx)
    d_x_mm = float(tip_xyz_x_mm[2])

    return ExtractDepthResult(
        T_tx=T_tx,
        tip_xyz_x_mm=tip_xyz_x_mm,
        d_x_mm=d_x_mm,
    )