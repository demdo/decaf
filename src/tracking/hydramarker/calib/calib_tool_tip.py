"""Tool-tip calibration for HydraMarker via landmark touches on a ChArUco board.

The tip vector ``p_tip`` (drill tip in the marker frame, mm) is estimated from
K touches of known inner ChArUco corners. Per frame the closed-form solution

    p_L     = R_bc @ X_j + t_bc          (touched corner in the camera frame)
    p_tip   = R_tc.T @ (p_L - t_tc)      (tip in the marker frame)

is exact, so the aggregation over frames/touches reduces to robust averaging
(median + 3*MAD outlier filter, then mean over inliers).

Subcommands (run from ``src``):

    python -m tracking.hydramarker.calib.calib_tool_tip selftest
    python -m tracking.hydramarker.calib.calib_tool_tip solve [--dir DIR | SESSION.npz]
    python -m tracking.hydramarker.calib.calib_tool_tip record [--out-dir DIR]
    python -m tracking.hydramarker.calib.calib_tool_tip overlay [--tip-npz FILE]

All touches of one session are stored in a single ``tip_session.npz`` (rewritten
atomically after every touch, so a crash keeps the touches recorded so far), and
``record`` resumes an existing session file automatically.

Conventions (verified against the pipeline):
    - ``T_tc`` marker->camera and ``T_bc`` board->camera, both solvePnP style
      (X_cam = R @ X_obj + t), translations in mm.
    - ChArUco corner id j maps to grid position col = j % 8, row = j // 8
      (board 9x7 squares, DICT_5X5_50). The board frame origin sits at the
      OUTER board corner, so X_j = ((col+1), (row+1), 0) * 25.4 mm; all object
      points are taken from ``board.getChessboardCorners()`` and therefore
      match the pose solver by construction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence


def _ensure_src_on_path() -> None:
    src_root = Path(__file__).resolve().parents[3]
    src = str(src_root)
    if src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()

import cv2
import numpy as np

from tracking.hydramarker.calib import calib_camera
from tracking.hydramarker.calib import calib_checkerboard as ccb


# ---------------------------------------------------------------------------
# Defaults / protocol constants
# ---------------------------------------------------------------------------

# 10 target corners on the 8x6 inner-corner grid (id = row * 8 + col): a ring
# around the board centre plus two central points, all with >= 1 corner margin
# to the border so occlusion by the tool never starves the pose solver on one
# side and the board pose stays interpolated rather than extrapolated.
DEFAULT_TARGET_CORNER_IDS = (10, 13, 22, 30, 37, 34, 25, 17, 19, 28)

DEFAULT_N_FRAMES_PER_TOUCH = 60
DEFAULT_RECORD_TIMEOUT_FRAMES = 300  # ~10 s @ 30 fps without 60 valid frames
DEFAULT_MIN_CHARUCO_CORNERS = 16     # of 48; tool + hand may occlude the rest
DEFAULT_BOARD_RMS_GATE_PX = 2.0
DEFAULT_MARKER_REPROJ_GATE_PX = 2.5

SKIP_FRAMES = 15                     # settling frames dropped per touch
FRAME_MAD_SCALE = 3.5                # per-frame robust gate inside a touch
TOUCH_MAD_SCALE = 3.0                # 3*MAD outlier filter across touches
MIN_FRAMES_USED = 10

SIGMA_NORM_ACCEPT_MM = 0.2
MIN_INLIER_TOUCHES = 8

RECORD_WINDOW_NAME = "HydraMarker Tip-Kalibrierung (Aufnahme)"
OVERLAY_WINDOW_NAME = "HydraMarker Tip-Kalibrierung (Overlay)"
OVERLAY_BOARD_EVERY = 2

TIP_SESSION_FILENAME = "tip_session.npz"
TIP_RESULT_FILENAME = "tip_calibration.npz"


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Math core (shared by selftest / solve / overlay)
# ---------------------------------------------------------------------------


def tip_estimates_from_poses(
    T_tc: np.ndarray,
    T_bc: np.ndarray,
    X_board: np.ndarray,
) -> np.ndarray:
    """Closed-form per-frame tip estimates in the marker frame.

    ``T_tc``/``T_bc`` are (N,4,4) marker->camera / board->camera transforms in
    mm, ``X_board`` the touched corner (3,) in the board frame. Returns (N,3).
    """
    T_tc = np.asarray(T_tc, dtype=np.float64).reshape(-1, 4, 4)
    T_bc = np.asarray(T_bc, dtype=np.float64).reshape(-1, 4, 4)
    X = np.asarray(X_board, dtype=np.float64).reshape(3)

    p_L = np.einsum("nij,j->ni", T_bc[:, :3, :3], X) + T_bc[:, :3, 3]
    R_t = T_tc[:, :3, :3]
    t_t = T_tc[:, :3, 3]
    return np.einsum("nji,nj->ni", R_t, p_L - t_t)


@dataclass(frozen=True)
class TouchSolveResult:
    """Robust tip estimate of one touch plus diagnostics."""

    p_tip: np.ndarray                 # (3,) per-touch median, mm
    p_frames: np.ndarray              # (N_used,3) per-frame estimates, mm
    n_frames_total: int
    n_frames_used: int
    frame_scatter_mm: float           # 1.4826 * MAD of ||p_i - median||
    board_rms_median_px: float
    R_tc_mid: np.ndarray              # (3,3) marker rotation, mid used frame
    R_bc_mid: np.ndarray              # (3,3) board rotation, mid used frame
    corner_id: int = -1
    source: str = ""


def solve_touch(
    T_tc: np.ndarray,
    T_bc: np.ndarray,
    X_board: np.ndarray,
    *,
    board_rms_px: np.ndarray | None = None,
    marker_reproj_px: np.ndarray | None = None,
    skip_frames: int = SKIP_FRAMES,
    mad_scale: float = FRAME_MAD_SCALE,
    corner_id: int = -1,
    source: str = "",
) -> TouchSolveResult:
    """Estimate the tip from one touch: gate frames, then median over frames."""
    T_tc = np.asarray(T_tc, dtype=np.float64).reshape(-1, 4, 4)
    T_bc = np.asarray(T_bc, dtype=np.float64).reshape(-1, 4, 4)
    n_total = int(T_tc.shape[0])
    if T_bc.shape[0] != n_total:
        raise ValueError("T_tc und T_bc müssen gleich viele Frames enthalten.")
    if n_total == 0:
        raise ValueError("Berührung ohne Frames.")

    keep = np.ones(n_total, dtype=bool)
    keep[: min(int(skip_frames), max(0, n_total - MIN_FRAMES_USED))] = False
    keep &= np.isfinite(T_tc).all(axis=(1, 2)) & np.isfinite(T_bc).all(axis=(1, 2))

    def _apply_scalar_gate(values: np.ndarray | None) -> None:
        nonlocal keep
        if values is None:
            return
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if vals.shape[0] != n_total:
            return
        valid = np.isfinite(vals) & (vals >= 0.0)
        if not np.any(valid & keep):
            return
        robust = ccb._robust_scalar_mask(vals, mad_scale)
        candidate = keep & (~valid | robust)
        if int(np.count_nonzero(candidate)) >= MIN_FRAMES_USED:
            keep = candidate

    _apply_scalar_gate(board_rms_px)
    _apply_scalar_gate(marker_reproj_px)

    if int(np.count_nonzero(keep)) < MIN_FRAMES_USED:
        keep = np.isfinite(T_tc).all(axis=(1, 2)) & np.isfinite(T_bc).all(axis=(1, 2))
        if int(np.count_nonzero(keep)) == 0:
            raise ValueError("Berührung enthält keine gültigen Posen.")

    p_all = tip_estimates_from_poses(T_tc, T_bc, X_board)
    p_used = p_all[keep]
    p_med = np.median(p_used, axis=0)
    dist = np.linalg.norm(p_used - p_med, axis=1)
    scatter = 1.4826 * float(np.median(np.abs(dist - np.median(dist)))) if len(dist) > 1 else 0.0

    used_idx = np.flatnonzero(keep)
    mid = int(used_idx[len(used_idx) // 2])
    board_rms_median = float("nan")
    if board_rms_px is not None:
        vals = np.asarray(board_rms_px, dtype=np.float64).reshape(-1)
        if vals.shape[0] == n_total:
            kept_vals = vals[keep]
            kept_vals = kept_vals[np.isfinite(kept_vals) & (kept_vals >= 0.0)]
            if kept_vals.size:
                board_rms_median = float(np.median(kept_vals))

    return TouchSolveResult(
        p_tip=p_med.reshape(3),
        p_frames=p_used.reshape(-1, 3),
        n_frames_total=n_total,
        n_frames_used=int(np.count_nonzero(keep)),
        frame_scatter_mm=scatter,
        board_rms_median_px=board_rms_median,
        R_tc_mid=T_tc[mid, :3, :3].copy(),
        R_bc_mid=T_bc[mid, :3, :3].copy(),
        corner_id=int(corner_id),
        source=str(source),
    )


@dataclass(frozen=True)
class TipAggregation:
    """Final tip estimate across touches."""

    p_tip: np.ndarray            # (3,) mean over inlier touches, mm
    p_tip_median: np.ndarray     # (3,) component-wise median (robust check)
    sigma: np.ndarray            # (4,) (sx, sy, sz, s_norm) via 1.4826*MAD, mm
    p_tip_all: np.ndarray        # (K,3)
    inlier_mask: np.ndarray      # (K,)
    distances_mm: np.ndarray     # (K,) ||p^k - median||

    @property
    def n_touches(self) -> int:
        return int(self.p_tip_all.shape[0])

    @property
    def n_inliers(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def sigma_norm_mm(self) -> float:
        return float(self.sigma[3])

    @property
    def passed(self) -> bool:
        required = min(MIN_INLIER_TOUCHES, self.n_touches)
        return self.sigma_norm_mm <= SIGMA_NORM_ACCEPT_MM and self.n_inliers >= required


def aggregate_touches(
    p_tip_per_touch: np.ndarray,
    *,
    mad_scale: float = TOUCH_MAD_SCALE,
) -> TipAggregation:
    """Median + 3*MAD outlier filter, then mean over inlier touches.

    Because the marker rotation is orthonormal, the joint least-squares
    solution over all touches equals this mean of the closed-form estimates;
    no richer estimator exists without modelling pose bias.
    """
    p_all = np.asarray(p_tip_per_touch, dtype=np.float64).reshape(-1, 3)
    if p_all.shape[0] == 0:
        raise ValueError("Keine Berührungen zu aggregieren.")

    median = np.median(p_all, axis=0)
    sigma_comp = 1.4826 * np.median(np.abs(p_all - median), axis=0)
    sigma_norm = float(np.linalg.norm(sigma_comp))
    dist = np.linalg.norm(p_all - median, axis=1)

    inliers = dist <= mad_scale * sigma_norm if sigma_norm > 0.0 else dist <= 0.0
    if sigma_norm == 0.0:
        inliers = np.ones(p_all.shape[0], dtype=bool)
    if not np.any(inliers):
        inliers = np.ones(p_all.shape[0], dtype=bool)

    p_final = np.mean(p_all[inliers], axis=0)
    sigma = np.array(
        [sigma_comp[0], sigma_comp[1], sigma_comp[2], sigma_norm],
        dtype=np.float64,
    )
    return TipAggregation(
        p_tip=p_final.reshape(3),
        p_tip_median=median.reshape(3),
        sigma=sigma,
        p_tip_all=p_all,
        inlier_mask=np.asarray(inliers, dtype=bool).reshape(-1),
        distances_mm=dist.reshape(-1),
    )


def tilt_deg_for_touch(touch: TouchSolveResult, axis_marker: np.ndarray) -> float:
    """Angle between the board normal and the (proxy) tool axis.

    ``axis_marker`` is a unit direction in the marker frame; without a
    calibrated drill axis we use unit(p_tip) as documented proxy.
    """
    a = np.asarray(axis_marker, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(a))
    if norm <= 1e-12:
        return float("nan")
    c = touch.R_tc_mid @ (a / norm)
    n = touch.R_bc_mid[:, 2]
    dot = float(np.clip(abs(np.dot(n, c)), 0.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


# ---------------------------------------------------------------------------
# NPZ schemas
# ---------------------------------------------------------------------------


def target_corner_object_points_mm(board: Any, corner_ids: Sequence[int]) -> np.ndarray:
    ids = np.asarray(list(corner_ids), dtype=np.int32).reshape(-1)
    return ccb._charuco_object_points_mm(board, ids)


def _atomic_savez(path: Path, **arrays: Any) -> None:
    """Write an NPZ atomically: full file to a temp path, then os.replace.

    Passing an open handle keeps numpy from mangling the ``.tmp`` name into
    ``.tmp.npz``; ``os.replace`` is atomic on Windows and POSIX, so a crash
    mid-write can never corrupt the previously persisted session.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def write_session_npz(
    path: Path,
    touches: Sequence[dict[str, Any]],
    *,
    target_corner_ids: Sequence[int],
    n_frames_requested: int,
    marker_corners_xyz_mm: np.ndarray,
    marker_corner_ids: np.ndarray,
    marker_json_path: str,
    marker_surface_model_json: str,
    field_path: str = "",
    created_utc: str | None = None,
) -> None:
    """Write the whole session to one NPZ (all K touches stacked on axis 0).

    Each touch dict carries the stacked per-frame arrays plus ``X_board``,
    ``corner_id`` and ``n_invalid``. Because every touch collects exactly
    ``n_frames_requested`` valid frames the arrays are rectangular, so the
    per-touch data becomes a clean ``(K, N, ...)`` block.
    """
    K = len(touches)
    if K == 0:
        raise ValueError("write_session_npz braucht mindestens eine Berührung.")

    def stack(key: str, dtype: Any) -> np.ndarray:
        return np.asarray([t[key] for t in touches], dtype=dtype)

    _atomic_savez(
        path,
        T_tc=stack("T_tc", np.float64),
        T_bc=stack("T_bc", np.float64),
        X_board=stack("X_board", np.float64),
        corner_ids=stack("corner_id", np.int32),
        board_rms_px=stack("board_rms_px", np.float64),
        board_n_corners=stack("board_n_corners", np.int32),
        marker_reproj_px=stack("marker_reproj_px", np.float64),
        marker_num_inliers=stack("marker_num_inliers", np.int32),
        timestamps=stack("timestamps", np.float64),
        frame_indices=stack("frame_indices", np.int64),
        n_invalid=stack("n_invalid", np.int32),
        n_touches=np.int32(K),
        n_frames_requested=np.int32(n_frames_requested),
        target_corner_ids=np.asarray(list(target_corner_ids), dtype=np.int32),
        marker_corners_xyz_mm=np.asarray(marker_corners_xyz_mm, dtype=np.float64),
        marker_corner_ids=np.asarray(marker_corner_ids, dtype=np.int64),
        marker_json_path=str(marker_json_path),
        marker_surface_model_json=str(marker_surface_model_json),
        field_path=str(field_path),
        board_squares_xy=np.asarray(
            [calib_camera.SQUARES_X, calib_camera.SQUARES_Y], dtype=np.int32
        ),
        board_square_length_mm=np.float64(calib_camera.SQUARE_LEN_M * 1000.0),
        board_marker_length_mm=np.float64(calib_camera.MARKER_LEN_M * 1000.0),
        board_aruco_dict_id=np.int32(calib_camera.DICT_ID),
        created_utc=str(created_utc or _utc_now_str()),
    )


def load_marker_geometry(marker_json_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Corner cloud (ids, xyz mm, marker frame) + surface model JSON string."""
    with open(marker_json_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    corners = meta.get("corners") or []
    ids = np.asarray([int(c["id"]) for c in corners], dtype=np.int64)
    xyz = np.asarray(
        [np.asarray(c["xyz_mm"], dtype=np.float64).reshape(3) for c in corners],
        dtype=np.float64,
    ).reshape(-1, 3)

    surface_model = meta.get("surface_model")
    surface_json = json.dumps(surface_model) if isinstance(surface_model, dict) else ""
    return ids, xyz, surface_json


# ---------------------------------------------------------------------------
# solve (Modul B)
# ---------------------------------------------------------------------------


def _choose_file_qt(title: str, file_filter: str) -> Path:
    from PySide6.QtWidgets import QApplication, QFileDialog

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    path, _ = QFileDialog.getOpenFileName(None, title, "", file_filter)
    if not path:
        raise RuntimeError(f"Keine Datei ausgewählt: {title}")
    return Path(path)


def _solve_session_file(path: Path) -> tuple[list[TouchSolveResult], dict[str, Any]]:
    """Solve every touch stored in one session NPZ; return touches + geometry."""
    with np.load(path, allow_pickle=False) as data:
        n = int(data["T_tc"].shape[0])
        corner_ids = (
            np.asarray(data["corner_ids"]).reshape(-1)
            if "corner_ids" in data
            else np.full(n, -1, dtype=np.int64)
        )
        touches: list[TouchSolveResult] = []
        for k in range(n):
            touches.append(
                solve_touch(
                    data["T_tc"][k],
                    data["T_bc"][k],
                    data["X_board"][k],
                    board_rms_px=data["board_rms_px"][k] if "board_rms_px" in data else None,
                    marker_reproj_px=(
                        data["marker_reproj_px"][k] if "marker_reproj_px" in data else None
                    ),
                    corner_id=int(corner_ids[k]),
                    source=f"Beruehrung {k + 1:02d}",
                )
            )
        extras = {
            "marker_corners_xyz_mm": (
                np.asarray(data["marker_corners_xyz_mm"], dtype=np.float64)
                if "marker_corners_xyz_mm" in data
                else None
            ),
            "marker_corner_ids": (
                np.asarray(data["marker_corner_ids"], dtype=np.int64)
                if "marker_corner_ids" in data
                else None
            ),
            "marker_surface_model_json": (
                str(data["marker_surface_model_json"])
                if "marker_surface_model_json" in data
                else ""
            ),
            "marker_json_path": (
                str(data["marker_json_path"]) if "marker_json_path" in data else ""
            ),
        }
    return touches, extras


def _set_axes_equal_3d(ax) -> None:
    # Local copy of model.visualization.set_axes_equal (kept import-free so the
    # calibration tool does not pull the SfM stack / SciPy).
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_middle = float(np.mean(x_limits))
    y_middle = float(np.mean(y_limits))
    z_middle = float(np.mean(z_limits))
    radius = 0.5 * max(
        abs(x_limits[1] - x_limits[0]),
        abs(y_limits[1] - y_limits[0]),
        abs(z_limits[1] - z_limits[0]),
    )
    ax.set_xlim3d([x_middle - radius, x_middle + radius])
    ax.set_ylim3d([y_middle - radius, y_middle + radius])
    ax.set_zlim3d([z_middle - radius, z_middle + radius])


def _cylinder_wireframe_points(surface_model_json: str) -> tuple[np.ndarray, ...] | None:
    if not surface_model_json:
        return None
    try:
        surface = json.loads(surface_model_json)
    except (TypeError, ValueError):
        return None
    fitted = surface.get("fitted") if isinstance(surface, dict) else None
    if not isinstance(fitted, dict):
        return None
    try:
        radius = float(fitted["radius_mm"])
        p0 = np.asarray(fitted["axis_point_mm"], dtype=np.float64).reshape(3)
        d = np.asarray(fitted["axis_dir"], dtype=np.float64).reshape(3)
        e1 = np.asarray(fitted["radial_ref_dir"], dtype=np.float64).reshape(3)
        along_min, along_max = (float(v) for v in fitted["axial_range_mm"])
        theta_min, theta_max = (float(v) for v in fitted["angular_range_rad"])
    except (KeyError, TypeError, ValueError):
        return None

    d = d / max(float(np.linalg.norm(d)), 1e-12)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(d, e1)

    along = np.linspace(along_min, along_max, 8)
    theta = np.linspace(theta_min, theta_max, 24)
    A, T = np.meshgrid(along, theta)
    pts = (
        p0[None, None, :]
        + A[..., None] * d[None, None, :]
        + radius * (np.cos(T)[..., None] * e1[None, None, :]
                    + np.sin(T)[..., None] * e2[None, None, :])
    )
    return pts[..., 0], pts[..., 1], pts[..., 2]


def _show_solve_plots(
    touches: list[TouchSolveResult],
    agg: TipAggregation,
    tilt_deg: np.ndarray,
    extras: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    k_idx = np.arange(1, agg.n_touches + 1)
    dev = agg.p_tip_all - agg.p_tip_median[None, :]
    outliers = ~agg.inlier_mask

    fig1 = plt.figure("Tip-Kalibrierung – Statistik", figsize=(13, 9))

    # (1) component deviations per touch
    ax1 = fig1.add_subplot(2, 2, 1)
    for comp, label in enumerate(("x", "y", "z")):
        ax1.plot(k_idx, dev[:, comp], marker="o", label=f"Δ{label}")
    if np.any(outliers):
        for comp in range(3):
            ax1.plot(
                k_idx[outliers], dev[outliers, comp],
                linestyle="none", marker="o",
                markerfacecolor="none", markeredgecolor="red", markersize=11,
            )
    ax1.axhline(0.0, color="gray", linewidth=0.8)
    ax1.set_xlabel("Berührung k")
    ax1.set_ylabel("Abweichung vom Median [mm]")
    ax1.set_title("p_tip^k je Komponente (Ausreißer rot umrandet)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) pooled per-frame jitter
    ax2 = fig1.add_subplot(2, 2, 2)
    pooled = np.concatenate(
        [np.linalg.norm(t.p_frames - t.p_tip[None, :], axis=1) for t in touches]
    )
    ax2.hist(pooled, bins=40, color="steelblue", alpha=0.85)
    ax2.set_xlabel("||p_i − Median der Berührung|| [mm]")
    ax2.set_ylabel("Frames")
    ax2.set_title(
        f"Frame-Jitter innerhalb der Berührungen (Median {np.median(pooled):.3f} mm)"
    )
    ax2.grid(True, alpha=0.3)

    # (3) depth component along the proxy axis vs tilt
    ax3 = fig1.add_subplot(2, 2, 3)
    axis = agg.p_tip / max(float(np.linalg.norm(agg.p_tip)), 1e-12)
    depth = (agg.p_tip_all - agg.p_tip[None, :]) @ axis
    ax3.scatter(tilt_deg[agg.inlier_mask], depth[agg.inlier_mask],
                color="steelblue", label="Inlier")
    if np.any(outliers):
        ax3.scatter(tilt_deg[outliers], depth[outliers], color="red", label="Ausreißer")
    finite = np.isfinite(tilt_deg) & agg.inlier_mask
    if int(np.count_nonzero(finite)) >= 3:
        coeff = np.polyfit(tilt_deg[finite], depth[finite], 1)
        xs = np.linspace(float(np.min(tilt_deg[finite])), float(np.max(tilt_deg[finite])), 20)
        ax3.plot(xs, np.polyval(coeff, xs), "k--",
                 label=f"Trend {coeff[0] * 10.0:.3f} mm/10°")
    ax3.axhline(0.0, color="gray", linewidth=0.8)
    ax3.set_xlabel("Kippwinkel (Proxy-Achse vs. Board-Normale) [°]")
    ax3.set_ylabel("Tiefenkomponente [mm]")
    ax3.set_title("Orientierungsabhängiger Bias-Check")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (4) 3D scatter around the median
    ax4 = fig1.add_subplot(2, 2, 4, projection="3d")
    ax4.scatter(dev[agg.inlier_mask, 0], dev[agg.inlier_mask, 1],
                dev[agg.inlier_mask, 2], color="steelblue", label="Inlier")
    if np.any(outliers):
        ax4.scatter(dev[outliers, 0], dev[outliers, 1], dev[outliers, 2],
                    color="red", label="Ausreißer")
    ax4.scatter([0.0], [0.0], [0.0], color="black", marker="+", s=90, label="Median")
    ax4.set_xlabel("Δx [mm]")
    ax4.set_ylabel("Δy [mm]")
    ax4.set_zlabel("Δz [mm]")
    ax4.set_title("Streuung der p_tip^k um den Median")
    ax4.legend(loc="upper left", fontsize=8)
    _set_axes_equal_3d(ax4)
    fig1.tight_layout()

    # Second window: marker geometry + tip in the marker frame.
    corners = extras.get("marker_corners_xyz_mm")
    corner_ids = extras.get("marker_corner_ids")
    if corners is not None and len(corners) > 0:
        fig2 = plt.figure("Tip-Kalibrierung – 3D-Geometrie (Markerframe)", figsize=(11, 9))
        ax = fig2.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            corners[:, 0], corners[:, 1], corners[:, 2],
            c=corner_ids if corner_ids is not None else None,
            s=18, label="Marker-Ecken (SfM-Modell)",
        )
        if corner_ids is not None:
            fig2.colorbar(scatter, ax=ax, shrink=0.6, label="Ecken-ID")

        wire = _cylinder_wireframe_points(extras.get("marker_surface_model_json", ""))
        if wire is not None:
            ax.plot_wireframe(*wire, color="gray", alpha=0.25, linewidth=0.6)

        centroid = corners.mean(axis=0)
        tip = agg.p_tip
        ax.plot(
            [centroid[0], tip[0]], [centroid[1], tip[1]], [centroid[2], tip[2]],
            "k--", linewidth=1.0,
            label="Verbindungslinie Schwerpunkt→Spitze (keine Bohrerachse)",
        )
        ax.scatter([tip[0]], [tip[1]], [tip[2]], color="red", marker="X", s=140,
                   label=f"p_tip ({tip[0]:.1f}, {tip[1]:.1f}, {tip[2]:.1f}) mm")
        ax.scatter(
            agg.p_tip_all[:, 0], agg.p_tip_all[:, 1], agg.p_tip_all[:, 2],
            color="red", s=8, alpha=0.5,
        )
        dist_mm = float(np.linalg.norm(tip - centroid))
        ax.set_title(
            "Marker-Modell + kalibrierte Spitze "
            f"(Abstand Schwerpunkt→Spitze {dist_mm:.1f} mm)"
        )
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        ax.legend(loc="upper left", fontsize=8)
        _set_axes_equal_3d(ax)
        fig2.tight_layout()
    else:
        print(
            "[calib_tool_tip] Hinweis: keine Marker-Eckenwolke in den NPZs "
            "gefunden – 3D-Geometrie-Plot entfällt."
        )

    plt.show()


def _print_solve_report(
    touches: list[TouchSolveResult],
    agg: TipAggregation,
    tilt_deg: np.ndarray,
) -> None:
    print()
    print("=== Tool-Spitzen-Kalibrierung: Auswertung ===")
    header = (
        f"{'Datei':<22} {'Ecke':>4} {'Frames':>9} "
        f"{'p_tip^k [mm]':<30} {'Streuung':>9} {'Kipp':>6} {'Status':>9}"
    )
    print(header)
    print("-" * len(header))
    for k, touch in enumerate(touches):
        p = touch.p_tip
        status = "INLIER" if agg.inlier_mask[k] else "AUSREISSER"
        tilt = f"{tilt_deg[k]:5.1f}°" if np.isfinite(tilt_deg[k]) else "   – "
        print(
            f"{touch.source:<22} {touch.corner_id:>4} "
            f"{touch.n_frames_used:>4}/{touch.n_frames_total:<4} "
            f"({p[0]:8.3f}, {p[1]:8.3f}, {p[2]:8.3f})   "
            f"{touch.frame_scatter_mm:6.3f}mm {tilt:>6} {status:>9}"
        )
    print("-" * len(header))
    p = agg.p_tip
    m = agg.p_tip_median
    s = agg.sigma
    print(
        f"p_tip (Mittel über {agg.n_inliers} Inlier) = "
        f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) mm, Norm {np.linalg.norm(p):.3f} mm"
    )
    print(f"Median (Kontrolle)                = ({m[0]:.3f}, {m[1]:.3f}, {m[2]:.3f}) mm")
    print(
        f"sigma (1.4826·MAD)                = ({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f}) mm, "
        f"Norm {s[3]:.3f} mm"
    )
    print(f"Berührungen: {agg.n_touches} gesamt, {agg.n_inliers} Inlier")
    required = min(MIN_INLIER_TOUCHES, agg.n_touches)
    verdict = "BESTANDEN" if agg.passed else "NICHT BESTANDEN"
    print(
        f"Abnahme (sigma_norm <= {SIGMA_NORM_ACCEPT_MM:.2f} mm und >= {required} Inlier): "
        f"{verdict}"
    )
    if agg.n_touches < 10:
        print(
            "Hinweis: K < 10 – Kurz-Session; Abnahmekriterium gilt für die "
            "Präzisionssession mit K = 10."
        )
    print()


def run_solve(
    session_path: Path,
    *,
    out_path: Path | None = None,
    show_plots: bool = True,
    notes: str = "",
) -> TipAggregation:
    session_path = Path(session_path)
    touches, extras = _solve_session_file(session_path)
    if not touches:
        raise RuntimeError(f"Session {session_path.name} enthält keine Berührungen.")

    agg = aggregate_touches(np.stack([t.p_tip for t in touches]))
    axis = agg.p_tip / max(float(np.linalg.norm(agg.p_tip)), 1e-12)
    tilt_deg = np.asarray([tilt_deg_for_touch(t, axis) for t in touches])

    _print_solve_report(touches, agg, tilt_deg)

    if out_path is None:
        out_path = session_path.parent / TIP_RESULT_FILENAME
    np.savez(
        out_path,
        p_tip=agg.p_tip,
        p_tip_median=agg.p_tip_median,
        sigma=agg.sigma,
        sigma_norm_mm=np.float64(agg.sigma_norm_mm),
        K=np.int32(agg.n_touches),
        K_inlier=np.int32(agg.n_inliers),
        p_tip_all=agg.p_tip_all,
        inlier_mask=agg.inlier_mask,
        distances_mm=agg.distances_mm,
        corner_ids=np.asarray([t.corner_id for t in touches], dtype=np.int32),
        per_touch_frame_scatter_mm=np.asarray(
            [t.frame_scatter_mm for t in touches], dtype=np.float64
        ),
        per_touch_frames_used=np.asarray(
            [t.n_frames_used for t in touches], dtype=np.int32
        ),
        per_touch_tilt_deg=np.asarray(tilt_deg, dtype=np.float64),
        source_session=str(session_path.name),
        passed=np.bool_(agg.passed),
        marker_json_path=str(extras.get("marker_json_path", "")),
        board_square_length_mm=np.float64(calib_camera.SQUARE_LEN_M * 1000.0),
        board_squares_xy=np.asarray(
            [calib_camera.SQUARES_X, calib_camera.SQUARES_Y], dtype=np.int32
        ),
        board_aruco_dict_id=np.int32(calib_camera.DICT_ID),
        created_utc=_utc_now_str(),
        notes=str(notes),
    )
    print(f"[calib_tool_tip] Ergebnis gespeichert: {out_path}")

    if show_plots:
        _show_solve_plots(touches, agg, tilt_deg, extras)
    return agg


def _resolve_session_path(args: argparse.Namespace) -> Path:
    if args.session:
        return Path(args.session)
    if args.dir:
        candidate = Path(args.dir) / TIP_SESSION_FILENAME
        if not candidate.exists():
            raise RuntimeError(f"{candidate} nicht gefunden.")
        return candidate
    return _choose_file_qt(
        f"{TIP_SESSION_FILENAME} auswählen", "Tip-Session (*.npz)"
    )


def cmd_solve(args: argparse.Namespace) -> int:
    session_path = _resolve_session_path(args)
    agg = run_solve(
        session_path,
        out_path=Path(args.out) if args.out else None,
        show_plots=not args.no_plots,
        notes=args.notes,
    )
    return 0 if agg.passed else 2


# ---------------------------------------------------------------------------
# selftest (Modul D)
# ---------------------------------------------------------------------------


def _random_rotation(rng: np.random.Generator, max_angle_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    angle = np.radians(rng.uniform(0.0, max_angle_deg))
    R, _ = cv2.Rodrigues(axis.reshape(3, 1) * angle)
    return np.asarray(R, dtype=np.float64).reshape(3, 3)


def _make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _synthetic_touch(
    rng: np.random.Generator,
    p_tip_gt: np.ndarray,
    *,
    n_frames: int,
    noise_t_mm: float,
    noise_rot_deg: float,
    landmark_offset_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact synthetic poses for one touch; optional noise and landmark slip.

    ``landmark_offset_mm`` models a slipped tip: the poses are generated for
    the actually touched point ``X_board + slip`` while the nominal corner
    ``X_board`` is returned (and later fed to the solver), so the resulting
    touch estimate is a genuine outlier.
    """
    X_board = np.array(
        [rng.integers(1, 7) * 25.4, rng.integers(1, 5) * 25.4, 0.0],
        dtype=np.float64,
    )
    X_actual = X_board
    if landmark_offset_mm > 0.0:
        slip = rng.normal(size=3)
        slip[2] = 0.0
        slip /= max(float(np.linalg.norm(slip)), 1e-12)
        X_actual = X_board + landmark_offset_mm * slip

    R_bc_base = _random_rotation(rng, 25.0)
    t_bc_base = np.array([rng.uniform(-60, 60), rng.uniform(-60, 60), rng.uniform(380, 650)])
    R_tc_base = _random_rotation(rng, 180.0)

    T_tc = np.empty((n_frames, 4, 4))
    T_bc = np.empty((n_frames, 4, 4))
    for i in range(n_frames):
        R_bc = _random_rotation(rng, 0.02) @ R_bc_base
        t_bc = t_bc_base + rng.normal(scale=0.02, size=3)
        p_L = R_bc @ X_actual + t_bc

        R_tc = _random_rotation(rng, 0.02) @ R_tc_base
        t_tc = p_L - R_tc @ p_tip_gt

        if noise_rot_deg > 0.0:
            R_tc = _random_rotation(rng, noise_rot_deg) @ R_tc
        if noise_t_mm > 0.0:
            t_tc = t_tc + rng.normal(scale=noise_t_mm, size=3)

        T_bc[i] = _make_transform(R_bc, t_bc)
        T_tc[i] = _make_transform(R_tc, t_tc)

    return T_tc, T_bc, X_board


def cmd_selftest(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(int(args.seed))
    p_tip_gt = np.array([5.0, -12.0, 160.0]) + rng.uniform(-3.0, 3.0, size=3)
    n_touches = 10
    n_frames = DEFAULT_N_FRAMES_PER_TOUCH

    print("=== Selbsttest Tool-Spitzen-Kalibrierung ===")
    print(
        f"Ground Truth p_tip = ({p_tip_gt[0]:.4f}, {p_tip_gt[1]:.4f}, "
        f"{p_tip_gt[2]:.4f}) mm, {n_touches} Berührungen à {n_frames} Frames"
    )

    # --- pass 1: noise-free, must reproduce GT to machine precision ---------
    touches: list[TouchSolveResult] = []
    for k in range(n_touches):
        T_tc, T_bc, X_board = _synthetic_touch(
            rng, p_tip_gt, n_frames=n_frames, noise_t_mm=0.0, noise_rot_deg=0.0
        )
        touches.append(
            solve_touch(T_tc, T_bc, X_board, corner_id=k, source=f"synth_{k:02d}")
        )
    agg = aggregate_touches(np.stack([t.p_tip for t in touches]))
    err_clean = float(np.linalg.norm(agg.p_tip - p_tip_gt))
    ok_clean = err_clean < 1e-9
    print(
        f"1) rauschfrei: max. Abweichung {err_clean:.3e} mm "
        f"-> {'OK' if ok_clean else 'FEHLER'}"
    )

    # --- pass 2: noise + one injected outlier touch -------------------------
    outlier_idx = 3
    touches = []
    for k in range(n_touches):
        T_tc, T_bc, X_board = _synthetic_touch(
            rng,
            p_tip_gt,
            n_frames=n_frames,
            noise_t_mm=0.1,
            noise_rot_deg=0.05,
            landmark_offset_mm=3.0 if k == outlier_idx else 0.0,
        )
        touches.append(
            solve_touch(T_tc, T_bc, X_board, corner_id=k, source=f"synth_{k:02d}")
        )
    agg = aggregate_touches(np.stack([t.p_tip for t in touches]))
    err_noisy = float(np.linalg.norm(agg.p_tip - p_tip_gt))
    outlier_removed = not bool(agg.inlier_mask[outlier_idx])
    others_kept = int(np.count_nonzero(agg.inlier_mask)) >= n_touches - 2
    ok_noisy = err_noisy < 0.1 and outlier_removed and others_kept
    print(
        f"2) verrauscht (sigma_t=0.1 mm, sigma_theta=0.05 deg, 1 Ausreisser 3 mm): "
        f"Abweichung {err_noisy:.4f} mm, sigma_norm {agg.sigma_norm_mm:.4f} mm"
    )
    print(
        f"   Ausreißer korrekt entfernt: {'ja' if outlier_removed else 'NEIN'} "
        f"(Inlier {agg.n_inliers}/{agg.n_touches}) -> {'OK' if ok_noisy else 'FEHLER'}"
    )

    if ok_clean and ok_noisy:
        print("Selbsttest BESTANDEN.")
        return 0
    print("Selbsttest NICHT BESTANDEN.")
    return 1


# ---------------------------------------------------------------------------
# record (Modul A)
# ---------------------------------------------------------------------------


def _draw_tracked_corners(vis: np.ndarray, result: Any) -> None:
    """Draw the visible tracked marker corners as green dots (fresh pose)."""
    corners = getattr(result, "corners", None) or []
    for corner in corners:
        uv = getattr(corner, "uv", None)
        if uv is None:
            continue
        u, v = float(uv[0]), float(uv[1])
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        center = (int(round(u)), int(round(v)))
        cv2.circle(vis, center, 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, center, 4, (0, 90, 0), 1, cv2.LINE_AA)


class TipTouchRecorder:
    """Collects marker+board poses per touch inside the run_tracker loop."""

    def __init__(
        self,
        *,
        out_dir: Path,
        target_ids: Sequence[int],
        n_frames: int,
        min_charuco: int,
        board_rms_gate_px: float,
        marker_reproj_gate_px: float,
        timeout_frames: int,
        marker_corner_ids: np.ndarray,
        marker_corners_xyz_mm: np.ndarray,
        marker_json_path: str,
        marker_surface_model_json: str,
        field_path: str = "",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.session_path = self.out_dir / TIP_SESSION_FILENAME
        self.target_ids = [int(i) for i in target_ids]
        self.n_frames = int(n_frames)
        self.min_charuco = int(min_charuco)
        self.board_rms_gate_px = float(board_rms_gate_px)
        self.marker_reproj_gate_px = float(marker_reproj_gate_px)
        self.timeout_frames = int(timeout_frames)

        self.marker_corner_ids = marker_corner_ids
        self.marker_corners_xyz_mm = marker_corners_xyz_mm
        self.marker_json_path = marker_json_path
        self.marker_surface_model_json = marker_surface_model_json
        self.field_path = str(field_path)

        self.board, self.aruco_dict, self.detector_params = calib_camera.make_charuco_board()
        self.target_obj_mm = target_corner_object_points_mm(self.board, self.target_ids)

        self.K: np.ndarray | None = None
        self.dist: np.ndarray | None = None

        self.next_target = 0
        self.recording = False
        self.frames_since_start = 0
        self.n_invalid = 0
        self.buffer: list[dict[str, Any]] = []
        self.touches: list[dict[str, Any]] = []
        self.last_board_pose: ccb.CharucoTableFramePose | None = None
        self.last_status = "Bereit – Tracking mit 's' starten, Aufnahme mit 't'."

    # -- persistence ------------------------------------------------------

    def resume_from_session(self) -> int:
        """Reload an existing session file so a session can be continued.

        Returns the number of recovered touches. Only touches whose corner id
        matches the planned target order are adopted, so a mismatched file is
        ignored rather than silently corrupting the run.
        """
        if not self.session_path.exists():
            return 0
        try:
            with np.load(self.session_path, allow_pickle=False) as data:
                stored_targets = (
                    list(np.asarray(data["target_corner_ids"]).reshape(-1))
                    if "target_corner_ids" in data
                    else []
                )
                if [int(x) for x in stored_targets] != self.target_ids:
                    print(
                        "[calib_tool_tip] Vorhandene Session passt nicht zur Ziel-Ecken-"
                        "Liste – wird ignoriert."
                    )
                    return 0
                n = int(data["T_tc"].shape[0])
                corner_ids = np.asarray(data["corner_ids"]).reshape(-1)
                for k in range(n):
                    self.touches.append(
                        {
                            "T_tc": np.asarray(data["T_tc"][k], dtype=np.float64),
                            "T_bc": np.asarray(data["T_bc"][k], dtype=np.float64),
                            "X_board": np.asarray(data["X_board"][k], dtype=np.float64),
                            "corner_id": int(corner_ids[k]),
                            "board_rms_px": np.asarray(data["board_rms_px"][k]),
                            "board_n_corners": np.asarray(data["board_n_corners"][k]),
                            "marker_reproj_px": np.asarray(data["marker_reproj_px"][k]),
                            "marker_num_inliers": np.asarray(data["marker_num_inliers"][k]),
                            "timestamps": np.asarray(data["timestamps"][k]),
                            "frame_indices": np.asarray(data["frame_indices"][k]),
                            "n_invalid": int(np.asarray(data["n_invalid"]).reshape(-1)[k]),
                        }
                    )
        except (OSError, KeyError, ValueError) as exc:
            print(f"[calib_tool_tip] Konnte Session nicht laden ({exc}) – neue Session.")
            self.touches = []
            return 0
        self.next_target = min(len(self.touches), len(self.target_ids))
        return len(self.touches)

    def _write_session(self) -> None:
        """Persist the whole session to one NPZ (or remove it if now empty)."""
        if not self.touches:
            try:
                self.session_path.unlink()
            except FileNotFoundError:
                pass
            return
        write_session_npz(
            self.session_path,
            self.touches,
            target_corner_ids=self.target_ids,
            n_frames_requested=self.n_frames,
            marker_corners_xyz_mm=self.marker_corners_xyz_mm,
            marker_corner_ids=self.marker_corner_ids,
            marker_json_path=self.marker_json_path,
            marker_surface_model_json=self.marker_surface_model_json,
            field_path=self.field_path,
        )

    # -- camera ---------------------------------------------------------

    def set_camera(self, K: np.ndarray, dist: np.ndarray) -> None:
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    # -- session state ----------------------------------------------------

    @property
    def all_done(self) -> bool:
        return self.next_target >= len(self.target_ids)

    def current_corner_id(self) -> int | None:
        if self.all_done:
            return None
        return self.target_ids[self.next_target]

    def start_touch(self) -> None:
        if self.recording:
            print("[calib_tool_tip] Aufnahme läuft bereits.")
            return
        if self.all_done:
            print("[calib_tool_tip] Alle Ziel-Ecken erledigt – 'u' löscht die letzte.")
            return
        self.recording = True
        self.frames_since_start = 0
        self.n_invalid = 0
        self.buffer = []
        corner = self.current_corner_id()
        print(
            f"[calib_tool_tip] Aufnahme Berührung "
            f"{self.next_target + 1}/{len(self.target_ids)} (Ecke {corner}) gestartet…"
        )

    def undo_last(self) -> None:
        if self.recording:
            self.recording = False
            self.buffer = []
            print("[calib_tool_tip] Laufende Aufnahme verworfen.")
            return
        if not self.touches:
            print("[calib_tool_tip] Nichts zum Löschen vorhanden.")
            return
        self.touches.pop()
        self.next_target = max(0, self.next_target - 1)
        self._write_session()
        print(
            f"[calib_tool_tip] Letzte Berührung verworfen – Ecke "
            f"{self.current_corner_id()} bitte erneut berühren "
            f"({len(self.touches)} in Session)."
        )

    # -- per-frame hook ---------------------------------------------------

    def feed(self, frame_idx: int, frame_bgr: np.ndarray, result: Any) -> None:
        if self.K is None or self.dist is None:
            return

        board_pose, _det = ccb.estimate_charuco_table_frame_pose(
            frame_bgr,
            self.K,
            self.dist,
            board=self.board,
            aruco_dict=self.aruco_dict,
            detector_params=self.detector_params,
            frame_index=int(frame_idx),
            min_charuco_corners=self.min_charuco,
        )
        self.last_board_pose = board_pose

        if not self.recording:
            return

        self.frames_since_start += 1
        valid, reason = self._frame_valid(board_pose, result)
        if valid:
            assert board_pose is not None
            T_tc = ccb.make_transform_from_rvec_tvec(result.rvec, result.tvec)
            T_bc = ccb.make_transform_from_rvec_tvec(
                board_pose.rvec_cb, board_pose.tvec_cb_mm
            )
            self.buffer.append(
                {
                    "T_tc": T_tc,
                    "T_bc": T_bc,
                    "board_rms_px": float(board_pose.rms_px),
                    "board_n_corners": int(board_pose.num_charuco),
                    "marker_reproj_px": float(result.mean_reprojection_error_px),
                    "marker_num_inliers": int(result.num_inliers),
                    "timestamp": time.time(),
                    "frame_index": int(frame_idx),
                }
            )
        else:
            self.n_invalid += 1
            self.last_status = f"Warte auf gültige Frames ({reason})"

        if len(self.buffer) >= self.n_frames:
            self._finish_touch()
        elif self.frames_since_start >= self.timeout_frames:
            self.recording = False
            self.buffer = []
            print(
                f"[calib_tool_tip] ABBRUCH: nach {self.frames_since_start} Frames "
                f"keine {self.n_frames} gültigen Frames (zuletzt: {self.last_status}). "
                "Spitze neu ansetzen und erneut 't' drücken."
            )

    def _frame_valid(self, board_pose, result) -> tuple[bool, str]:
        from tracking.hydramarker import tracker_log

        if board_pose is None:
            return False, "Board nicht erkannt"
        if float(board_pose.rms_px) > self.board_rms_gate_px:
            return False, f"Board-RMS {board_pose.rms_px:.2f}px"
        if not tracker_log.has_fresh_pose(result):
            return False, "Marker-Pose nicht frisch"
        if result.rvec is None or result.tvec is None:
            return False, "Marker-Pose fehlt"
        reproj = float(result.mean_reprojection_error_px)
        if reproj < 0.0 or reproj > self.marker_reproj_gate_px:
            return False, f"Marker-Reproj {reproj:.2f}px"
        return True, ""

    def _finish_touch(self) -> None:
        self.recording = False
        corner_id = int(self.current_corner_id() or -1)
        X_board = self.target_obj_mm[self.next_target]

        idx = self.next_target + 1
        self.touches.append(
            {
                "T_tc": np.stack([b["T_tc"] for b in self.buffer]),
                "T_bc": np.stack([b["T_bc"] for b in self.buffer]),
                "X_board": np.asarray(X_board, dtype=np.float64).reshape(3),
                "corner_id": corner_id,
                "board_rms_px": np.asarray([b["board_rms_px"] for b in self.buffer]),
                "board_n_corners": np.asarray([b["board_n_corners"] for b in self.buffer]),
                "marker_reproj_px": np.asarray([b["marker_reproj_px"] for b in self.buffer]),
                "marker_num_inliers": np.asarray([b["marker_num_inliers"] for b in self.buffer]),
                "timestamps": np.asarray([b["timestamp"] for b in self.buffer]),
                "frame_indices": np.asarray([b["frame_index"] for b in self.buffer]),
                "n_invalid": int(self.n_invalid),
            }
        )
        self._write_session()
        self.buffer = []
        print(
            f"[calib_tool_tip] Berührung {idx}/{len(self.target_ids)} gespeichert "
            f"({self.n_frames} Frames, {self.n_invalid} ungültige übersprungen) "
            f"-> {self.session_path.name} [{len(self.touches)} Berührungen]"
        )
        self.next_target += 1
        if self.all_done:
            print(
                "[calib_tool_tip] Session komplett – Fenster mit 'q' schließen und "
                "'solve' ausführen."
            )

    # -- rendering --------------------------------------------------------

    def render(self, vis: np.ndarray, result: Any) -> np.ndarray:
        if self.K is None or self.dist is None:
            return vis

        board_pose = self.last_board_pose
        if board_pose is not None:
            ccb._draw_axes(vis, self.K, self.dist, board_pose.rvec_cb, board_pose.tvec_cb_mm)
            uv, _ = cv2.projectPoints(
                self.target_obj_mm.reshape(-1, 1, 3),
                board_pose.rvec_cb,
                board_pose.tvec_cb_mm,
                self.K,
                self.dist,
            )
            uv = uv.reshape(-1, 2)
            for i, (u, v) in enumerate(uv):
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                center = (int(round(u)), int(round(v)))
                if i < self.next_target:
                    cv2.circle(vis, center, 9, (0, 200, 0), 2)
                    cv2.line(
                        vis,
                        (center[0] - 4, center[1]),
                        (center[0] + 4, center[1]),
                        (0, 200, 0),
                        2,
                    )
                elif i == self.next_target and not self.all_done:
                    cv2.circle(vis, center, 14, (0, 220, 255), 3)
                    cv2.line(vis, (center[0] - 22, center[1]), (center[0] + 22, center[1]),
                             (0, 220, 255), 1)
                    cv2.line(vis, (center[0], center[1] - 22), (center[0], center[1] + 22),
                             (0, 220, 255), 1)
                    cv2.putText(
                        vis, f"ID {self.target_ids[i]}",
                        (center[0] + 18, center[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA,
                    )
                else:
                    cv2.circle(vis, center, 9, (160, 160, 160), 1)

        from tracking.hydramarker import tracker_log

        marker_fresh = tracker_log.has_fresh_pose(result)
        if marker_fresh:
            _draw_tracked_corners(vis, result)

        n_corners = 0 if board_pose is None else int(board_pose.num_charuco)
        board_rms = float("nan") if board_pose is None else float(board_pose.rms_px)

        if self.all_done:
            headline = "SESSION KOMPLETT - q beendet"
            color = (0, 200, 0)
        elif self.recording:
            headline = (
                f"AUFNAHME Ecke {self.current_corner_id()}: "
                f"{len(self.buffer)}/{self.n_frames} Frames"
            )
            color = (0, 220, 255)
        else:
            headline = (
                f"Beruehrung {self.next_target + 1}/{len(self.target_ids)}: Ecke "
                f"{self.current_corner_id()} ansetzen, dann 't'"
            )
            color = (0, 255, 0) if (marker_fresh and board_pose is not None) else (0, 0, 255)

        lines = [
            headline,
            (
                f"Board: {n_corners} Ecken, RMS "
                f"{board_rms:.2f}px | Marker-Pose: {'OK' if marker_fresh else 'FEHLT'}"
            ),
            "Tasten: t=Aufnahme  u=letzte loeschen  s=Tracking start/stopp  q=Ende",
        ]
        if self.recording and self.n_invalid:
            lines.insert(2, f"uebersprungene ungueltige Frames: {self.n_invalid}")
        return ccb._draw_text_box(vis, lines, color=color)


def _parse_corner_ids(raw: str | None) -> list[int]:
    if not raw:
        return list(DEFAULT_TARGET_CORNER_IDS)
    ids = [int(part) for part in raw.replace(";", ",").split(",") if part.strip()]
    if not ids:
        raise ValueError("Leere Ziel-Ecken-Liste.")
    max_id = (calib_camera.SQUARES_X - 1) * (calib_camera.SQUARES_Y - 1) - 1
    for cid in ids:
        if cid < 0 or cid > max_id:
            raise ValueError(f"Ecken-ID {cid} außerhalb 0..{max_id}.")
    return ids


def cmd_record(args: argparse.Namespace) -> int:
    from tracking.hydramarker import run_tracker as rt
    from tracking.hydramarker.config import LiveTrackerConfig, LoggingConfig

    target_ids = _parse_corner_ids(args.corner_ids)

    field_path = Path(args.field) if args.field else rt.choose_file_qt(
        "HydraMarker .field auswählen", "HydraMarker field (*.field)"
    )
    marker_json_path = Path(args.marker_json) if args.marker_json else rt.choose_file_qt(
        "Marker .json auswählen", "Marker JSON (*.json)"
    )

    marker_corner_ids, marker_corners_xyz, surface_json = load_marker_geometry(
        marker_json_path
    )
    if marker_corners_xyz.size == 0:
        print(
            "[calib_tool_tip] Warnung: Marker-JSON enthält keine 'corners' – "
            "3D-Geometrie-Plot wird später entfallen."
        )

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parent / "output" / f"tip_calib_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[calib_tool_tip] Ausgabeverzeichnis: {out_dir}")

    recorder = TipTouchRecorder(
        out_dir=out_dir,
        target_ids=target_ids,
        n_frames=int(args.n_frames),
        min_charuco=int(args.min_charuco),
        board_rms_gate_px=float(args.board_rms_gate),
        marker_reproj_gate_px=float(args.marker_reproj_gate),
        timeout_frames=int(args.timeout_frames),
        marker_corner_ids=marker_corner_ids,
        marker_corners_xyz_mm=marker_corners_xyz,
        marker_json_path=str(marker_json_path),
        marker_surface_model_json=surface_json,
        field_path=str(field_path),
    )

    resumed = recorder.resume_from_session()
    if resumed:
        print(
            f"[calib_tool_tip] Bestehende Session fortgesetzt: {resumed} Berührungen "
            f"geladen, weiter bei Ecke {recorder.current_corner_id()}."
        )

    def after_camera_ready(camera, K, dist) -> bool:
        recorder.set_camera(K, dist)
        return True

    def on_frame(frame_idx, raw_frame, result, _logging_active) -> None:
        recorder.feed(frame_idx, raw_frame, result)

    def on_key(key, _tracker, _last_result, _frame_idx) -> bool:
        if key == ord("t"):
            recorder.start_touch()
            return True
        if key == ord("u"):
            recorder.undo_last()
            return True
        return False

    def render_frame(vis, result, _tracker, _frame_idx, _logging_active):
        return recorder.render(vis, result)

    config = LiveTrackerConfig(
        tracker=rt.make_live_tracker_config(),
        logging=LoggingConfig(enabled=False, start_mode="disabled"),
        start_tracking_manually=False,
    )
    # None -> run_tracker opens a Qt dialog for the camera calibration .npz.
    config.camera.calibration_path = str(args.calibration) if args.calibration else None

    rt.run_tracker(
        config=config,
        field_path=field_path,
        marker_json_path=marker_json_path,
        window_name=RECORD_WINDOW_NAME,
        console_prefix="[calib_tool_tip]",
        after_camera_ready=after_camera_ready,
        on_frame=on_frame,
        on_key=on_key,
        render_frame=render_frame,
    )

    n = len(recorder.touches)
    print(f"[calib_tool_tip] Session beendet: {n}/{len(target_ids)} Berührungen in {out_dir}")
    if n:
        print(
            "[calib_tool_tip] Auswertung: python -m "
            f"tracking.hydramarker.calib.calib_tool_tip solve --dir \"{out_dir}\""
        )
    return 0


# ---------------------------------------------------------------------------
# overlay (Modul C)
# ---------------------------------------------------------------------------


class TipOverlay:
    """Projects the calibrated tip into the live image + board residual check."""

    def __init__(self, p_tip: np.ndarray, *, min_charuco: int = DEFAULT_MIN_CHARUCO_CORNERS) -> None:
        self.p_tip = np.asarray(p_tip, dtype=np.float64).reshape(3)
        self.min_charuco = int(min_charuco)
        self.board, self.aruco_dict, self.detector_params = calib_camera.make_charuco_board()
        all_ids = np.arange(
            (calib_camera.SQUARES_X - 1) * (calib_camera.SQUARES_Y - 1), dtype=np.int32
        )
        self.all_ids = all_ids
        self.all_corners_mm = ccb._charuco_object_points_mm(self.board, all_ids)
        self.K: np.ndarray | None = None
        self.dist: np.ndarray | None = None
        self.board_pose: ccb.CharucoTableFramePose | None = None
        self.last_residual_mm: float = float("nan")

    def set_camera(self, K: np.ndarray, dist: np.ndarray) -> None:
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    def feed(self, frame_idx: int, frame_bgr: np.ndarray) -> None:
        if self.K is None or int(frame_idx) % OVERLAY_BOARD_EVERY != 0:
            return
        pose, _det = ccb.estimate_charuco_table_frame_pose(
            frame_bgr,
            self.K,
            self.dist,
            board=self.board,
            aruco_dict=self.aruco_dict,
            detector_params=self.detector_params,
            frame_index=int(frame_idx),
            min_charuco_corners=self.min_charuco,
        )
        self.board_pose = pose

    def render(self, vis: np.ndarray, result: Any) -> np.ndarray:
        from tracking.hydramarker import tracker_log

        K = self.K
        dist = self.dist
        if K is None:
            return vis

        lines: list[str] = []
        color = (0, 0, 255)
        if tracker_log.has_fresh_pose(result) and result.rvec is not None:
            rvec = np.asarray(result.rvec, dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(result.tvec, dtype=np.float64).reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            x_tip_cam = (R @ self.p_tip.reshape(3, 1) + tvec).reshape(3)

            uv, _ = cv2.projectPoints(self.p_tip.reshape(1, 1, 3), rvec, tvec, K, dist)
            u, v = uv.reshape(2)
            if np.isfinite(u) and np.isfinite(v):
                c = (int(round(u)), int(round(v)))
                cv2.circle(vis, c, 10, (0, 0, 255), 2)
                cv2.line(vis, (c[0] - 24, c[1]), (c[0] + 24, c[1]), (0, 0, 255), 1)
                cv2.line(vis, (c[0], c[1] - 24), (c[0], c[1] + 24), (0, 0, 255), 1)

            color = (0, 255, 0)
            lines.append(
                f"Tip (Kamera) [mm]: ({x_tip_cam[0]:8.2f}, {x_tip_cam[1]:8.2f}, "
                f"{x_tip_cam[2]:8.2f})"
            )

            board_pose = self.board_pose
            if board_pose is not None:
                R_b = board_pose.R_C_B
                t_b = board_pose.tvec_cb_mm.reshape(3)
                corners_cam = self.all_corners_mm @ R_b.T + t_b
                dists = np.linalg.norm(corners_cam - x_tip_cam, axis=1)
                j = int(np.argmin(dists))
                self.last_residual_mm = float(dists[j])
                lines.append(
                    f"Naechste Board-Ecke ID {int(self.all_ids[j])}: "
                    f"Abstand {dists[j]:.2f} mm"
                )
                uv_c, _ = cv2.projectPoints(
                    self.all_corners_mm[j].reshape(1, 1, 3),
                    board_pose.rvec_cb,
                    board_pose.tvec_cb_mm,
                    K,
                    dist,
                )
                cu, cv_ = uv_c.reshape(2)
                if np.isfinite(cu) and np.isfinite(cv_) and np.isfinite(u) and np.isfinite(v):
                    cv2.line(
                        vis,
                        (int(round(u)), int(round(v))),
                        (int(round(cu)), int(round(cv_))),
                        (255, 200, 0),
                        1,
                    )
        else:
            lines.append("Keine frische Marker-Pose - Marker ins Bild bringen ('s' startet).")

        lines.append("Kontrolle: Spitze auf eine Ecke setzen -> Abstand muss gegen 0 gehen.")
        lines.append("Tasten: s=Tracking start/stopp  r=Reset  q=Ende")
        return ccb._draw_text_box(vis, lines, color=color)


def cmd_overlay(args: argparse.Namespace) -> int:
    from tracking.hydramarker import run_tracker as rt
    from tracking.hydramarker.config import LiveTrackerConfig, LoggingConfig

    tip_path = Path(args.tip_npz) if args.tip_npz else _choose_file_qt(
        "tip_calibration.npz auswählen", "Tip-Kalibrierung (*.npz)"
    )
    with np.load(tip_path, allow_pickle=False) as data:
        p_tip = np.asarray(data["p_tip"], dtype=np.float64).reshape(3)
        sigma_norm = float(data["sigma_norm_mm"]) if "sigma_norm_mm" in data else float("nan")
    print(
        f"[calib_tool_tip] Geladen: p_tip = ({p_tip[0]:.3f}, {p_tip[1]:.3f}, "
        f"{p_tip[2]:.3f}) mm (sigma_norm {sigma_norm:.3f} mm) aus {tip_path.name}"
    )

    field_path = Path(args.field) if args.field else rt.choose_file_qt(
        "HydraMarker .field auswählen", "HydraMarker field (*.field)"
    )
    marker_json_path = Path(args.marker_json) if args.marker_json else rt.choose_file_qt(
        "Marker .json auswählen", "Marker JSON (*.json)"
    )

    overlay = TipOverlay(p_tip)

    def after_camera_ready(camera, K, dist) -> bool:
        overlay.set_camera(K, dist)
        return True

    def on_frame(frame_idx, raw_frame, result, _logging_active) -> None:
        overlay.feed(frame_idx, raw_frame)

    def render_frame(vis, result, _tracker, _frame_idx, _logging_active):
        return overlay.render(vis, result)

    config = LiveTrackerConfig(
        tracker=rt.make_live_tracker_config(),
        logging=LoggingConfig(enabled=False, start_mode="disabled"),
        start_tracking_manually=False,
    )
    # None -> run_tracker opens a Qt dialog for the camera calibration .npz.
    config.camera.calibration_path = str(args.calibration) if args.calibration else None

    rt.run_tracker(
        config=config,
        field_path=field_path,
        marker_json_path=marker_json_path,
        window_name=OVERLAY_WINDOW_NAME,
        console_prefix="[calib_tool_tip]",
        after_camera_ready=after_camera_ready,
        on_frame=on_frame,
        render_frame=render_frame,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="calib_tool_tip",
        description=(
            "Tool-Spitzen-Kalibrierung per Landmark-Berührung auf dem ChArUco-Board "
            "(Ziel: sigma_norm <= 0.2 mm)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_self = sub.add_parser("selftest", help="Synthetischer Selbsttest der Mathematik")
    p_self.add_argument("--seed", type=int, default=7)
    p_self.set_defaults(func=cmd_selftest)

    p_solve = sub.add_parser(
        "solve", help="tip_session.npz auswerten -> tip_calibration.npz"
    )
    p_solve.add_argument("session", nargs="?", help="tip_session.npz")
    p_solve.add_argument("--dir", help=f"Verzeichnis mit {TIP_SESSION_FILENAME}")
    p_solve.add_argument("--out", help="Zielpfad für tip_calibration.npz")
    p_solve.add_argument("--no-plots", action="store_true")
    p_solve.add_argument("--notes", default="", help="Freitext (Bit-ID, Datum, …)")
    p_solve.set_defaults(func=cmd_solve)

    p_rec = sub.add_parser("record", help="Live-Aufnahme der Berührungen")
    p_rec.add_argument("--out-dir", help="Session-Verzeichnis (Default: output/tip_calib_<ts>)")
    p_rec.add_argument("--field", help="HydraMarker .field Datei")
    p_rec.add_argument("--marker-json", help="Marker-Geometrie .json")
    p_rec.add_argument("--calibration", help="Kamera-Kalibrierung .npz")
    p_rec.add_argument("--corner-ids", help=f"CSV, Default {DEFAULT_TARGET_CORNER_IDS}")
    p_rec.add_argument("--n-frames", type=int, default=DEFAULT_N_FRAMES_PER_TOUCH)
    p_rec.add_argument("--min-charuco", type=int, default=DEFAULT_MIN_CHARUCO_CORNERS)
    p_rec.add_argument("--board-rms-gate", type=float, default=DEFAULT_BOARD_RMS_GATE_PX)
    p_rec.add_argument(
        "--marker-reproj-gate", type=float, default=DEFAULT_MARKER_REPROJ_GATE_PX
    )
    p_rec.add_argument("--timeout-frames", type=int, default=DEFAULT_RECORD_TIMEOUT_FRAMES)
    p_rec.set_defaults(func=cmd_record)

    p_ov = sub.add_parser("overlay", help="Live-Verifikation der kalibrierten Spitze")
    p_ov.add_argument("--tip-npz", help="tip_calibration.npz")
    p_ov.add_argument("--field", help="HydraMarker .field Datei")
    p_ov.add_argument("--marker-json", help="Marker-Geometrie .json")
    p_ov.add_argument("--calibration", help="Kamera-Kalibrierung .npz")
    p_ov.set_defaults(func=cmd_overlay)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
