"""Tool-tip (pivot) calibration for HydraMarker.

No options -- running the script IS the calibration:

    python -m tracking.hydramarker.calib.calib_tool_tip

Flow: pick the .field, marker .json and camera .npz via Qt dialogs, then the
live view opens. SPACE starts the recording; seat the tip in the spherical
adapter's socket and tilt/rotate the drill widely (slowly). SPACE again ends
the session and closes the camera. The robust pivot solver then estimates the
tip and a 3D figure (SfM corner cloud + fitted cylinder + tip) is shown.

Method: a spherical adapter keeps its centre p_tip (marker frame) fixed under
rotation, so ``R_i @ p_tip + t_i = c`` for every frame. Stacking
``[R_i | -I] [p_tip; c] = -t_i`` and solving robustly (RANSAC + trimming)
gives p_tip; wide rotation makes all three axes -- including depth along the
tool axis -- well observed.

The marker sits on the drill's cylindrical body, so the fitted cylinder axis
(``surface_model.fitted``) IS the drill axis, and the physical tip must lie ON
it. The final tip is therefore solved CONSTRAINED to that axis
(``p_tip = p0 + s*d``, one axial unknown): the well-fitted axis (long lever,
sub-0.15 mm) supplies the two lateral coordinates, and pivoting only fixes the
depth ``s``. This removes the weakly-observed off-axis DoF that dominates the
monocular pivot error. The robust free solve above still runs -- it supplies
the inlier set and the off-axis diagnostic. Pass ``--free`` for the plain
unconstrained tip.

Outputs (individually in ``calib/``):
    pivot_session.npz       the raw recording (marker->camera poses + K/dist)
    pivot_session_frames/   raw frames (one per logged pose) for the OFFLINE
                            collar/ring calibration off the same sweep
    pivot_calibration.npz   the result (p_tip, sigma, ...), overlay-compatible
    pivot_tip_3d.png        the 3D SfM + tip figure

Live tip tracking / verification lives in ``test_tracker_tool_tip.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import sys
import threading
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


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Math core (shared by selftest / solve / overlay)
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# NPZ schemas
# ---------------------------------------------------------------------------




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
    # Equal ranges only look equal if the 3D box itself is cubic; matplotlib's
    # default box is not, which squashes the round cross-section into an ellipse.
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass  # matplotlib < 3.3 has no set_box_aspect


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


def _drill_axis_from_surface(
    surface_model_json: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (axis_point_mm, unit axis_dir) of the fitted cylinder = drill axis.

    The marker sits on the drill's cylindrical body, so ``surface_model.fitted``
    (from the SfM model) gives the drill axis directly in the marker frame.
    Returns None when the marker JSON carries no fitted cylinder.
    """
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
        p0 = np.asarray(fitted["axis_point_mm"], dtype=np.float64).reshape(3)
        d = np.asarray(fitted["axis_dir"], dtype=np.float64).reshape(3)
    except (KeyError, TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return None
    return p0, d / norm












# ---------------------------------------------------------------------------
# selftest (Modul D)
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# Pivot (Kugel-Adapter) calibration
# ---------------------------------------------------------------------------
#
# A spherical adapter seated in a socket keeps its centre fixed under arbitrary
# rotation. Per frame the marker pose (R_i, t_i) maps the sphere centre p_tip
# (marker frame) to the same camera point c (pivot):
#
#     R_i @ p_tip + t_i = c          for every frame i
#
# Stacking [R_i | -I] [p_tip; c] = -t_i gives a linear system; robust solving
# (RANSAC + trimming) rejects frames where the sphere briefly unseated or the
# pose flipped. Wide rotation (the sphere allows it) makes all three axes of
# p_tip well observed, including depth along the tool axis.

PIVOT_SESSION_FILENAME = "pivot_session.npz"
PIVOT_RESULT_FILENAME = "pivot_calibration.npz"
PIVOT_FRAMES_DIRNAME = "pivot_session_frames"
PIVOT_WINDOW_NAME = "HydraMarker Pivot Calibration (recording)"
PIVOT_DEFAULT_RANSAC_THRESH_MM = 2.0
PIVOT_DEFAULT_RANSAC_ITERS = 300
PIVOT_MIN_FRAMES = 30


@dataclass(frozen=True)
class PivotResult:
    p_tip: np.ndarray            # (3,) sphere centre in the marker frame, mm
    pivot_camera: np.ndarray     # (3,) pivot point in the camera frame, mm
    residual_rms_mm: float       # inlier per-frame residual RMS
    n_frames: int
    n_inliers: int
    inlier_mask: np.ndarray      # (N,) bool
    sigma_axis_mm: np.ndarray    # (3,) per-axis std of p_tip (marker frame)
    weak_dir_marker: np.ndarray  # (3,) least-observed direction (marker frame)
    weak_sigma_mm: float         # std along the weak direction
    cond_number: float           # conditioning of the stacked system
    per_frame_residual_mm: np.ndarray  # (N,)

    @property
    def radius_mm(self) -> float:
        return float(np.linalg.norm(self.p_tip))

    @property
    def sigma_norm_mm(self) -> float:
        return float(np.linalg.norm(self.sigma_axis_mm))


def _pivot_linear_solve(
    T_tc: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Algebraic pivot: least-squares [R_i|-I][p_tip;c]=-t_i on masked frames.

    Returns (p_tip, c, per_frame_residual) where the residual is evaluated on
    ALL frames so it can drive outlier gating.
    """
    T = np.asarray(T_tc, dtype=np.float64).reshape(-1, 4, 4)
    n_all = T.shape[0]
    idx = np.arange(n_all) if mask is None else np.flatnonzero(mask)
    if idx.size < 3:
        raise ValueError("Pivot needs at least 3 valid frames.")

    R = T[idx, :3, :3]
    t = T[idx, :3, 3]
    m = idx.size
    A = np.zeros((3 * m, 6), dtype=np.float64)
    A[:, :3] = R.reshape(3 * m, 3)
    eye_rows = np.tile(-np.eye(3), (m, 1))
    A[:, 3:] = eye_rows
    b = (-t).reshape(3 * m)

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    p_tip = x[:3]
    c = x[3:]

    res_all = np.linalg.norm(
        np.einsum("nij,j->ni", T[:, :3, :3], p_tip) + T[:, :3, 3] - c, axis=1
    )
    return p_tip, c, res_all


def solve_pivot_robust(
    T_tc: np.ndarray,
    *,
    ransac_thresh_mm: float = PIVOT_DEFAULT_RANSAC_THRESH_MM,
    ransac_iters: int = PIVOT_DEFAULT_RANSAC_ITERS,
    sample_size: int = 10,
    seed: int = 0,
) -> PivotResult:
    """Robust pivot: RANSAC for gross rejection, then trimmed refit + covariance."""
    T = np.asarray(T_tc, dtype=np.float64).reshape(-1, 4, 4)
    n = T.shape[0]
    if n < PIVOT_MIN_FRAMES:
        # still solvable, just warn-worthy; caller decides
        pass
    valid = np.isfinite(T).all(axis=(1, 2))
    if int(np.count_nonzero(valid)) < 3:
        raise ValueError("Too few valid pivot frames.")

    rng = np.random.default_rng(int(seed))
    valid_idx = np.flatnonzero(valid)
    s = int(min(sample_size, valid_idx.size))

    best_inliers: np.ndarray | None = None
    best_count = -1
    for _ in range(int(ransac_iters)):
        sample = rng.choice(valid_idx, size=s, replace=False)
        sample_mask = np.zeros(n, dtype=bool)
        sample_mask[sample] = True
        try:
            _p, _c, res = _pivot_linear_solve(T, sample_mask)
        except (ValueError, np.linalg.LinAlgError):
            continue
        inliers = valid & (res <= float(ransac_thresh_mm))
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        best_inliers = valid.copy()

    # Trimmed refit: re-solve on inliers, update the inlier set by robust
    # residual gate, repeat until stable.
    inliers = best_inliers.copy()
    for _ in range(5):
        p_tip, c, res = _pivot_linear_solve(T, inliers)
        keep = valid & (res <= float(ransac_thresh_mm))
        if int(np.count_nonzero(keep)) < 3:
            break
        if np.array_equal(keep, inliers):
            inliers = keep
            break
        inliers = keep

    p_tip, c, res_all = _pivot_linear_solve(T, inliers)
    res_in = res_all[inliers]
    n_in = int(np.count_nonzero(inliers))
    residual_rms = float(np.sqrt(np.mean(res_in * res_in))) if n_in else float("nan")

    # Covariance of [p_tip; c]: sigma^2 (A^T A)^-1 with sigma^2 from residuals.
    R = T[inliers, :3, :3]
    m = R.shape[0]
    A = np.zeros((3 * m, 6), dtype=np.float64)
    A[:, :3] = R.reshape(3 * m, 3)
    A[:, 3:] = np.tile(-np.eye(3), (m, 1))
    ata = A.T @ A
    try:
        ata_inv = np.linalg.inv(ata)
    except np.linalg.LinAlgError:
        ata_inv = np.linalg.pinv(ata)
    dof = max(3 * m - 6, 1)
    sigma2 = float(np.sum(res_in * res_in)) / dof
    cov_p = sigma2 * ata_inv[:3, :3]
    sigma_axis = np.sqrt(np.clip(np.diag(cov_p), 0.0, None))
    evals, evecs = np.linalg.eigh(cov_p)
    weak_sigma = float(np.sqrt(max(evals[-1], 0.0)))
    weak_dir = evecs[:, -1]
    sv = np.linalg.svd(A, compute_uv=False)
    cond = float(sv[0] / max(sv[-1], 1e-12))

    return PivotResult(
        p_tip=p_tip.reshape(3),
        pivot_camera=c.reshape(3),
        residual_rms_mm=residual_rms,
        n_frames=n,
        n_inliers=n_in,
        inlier_mask=inliers,
        sigma_axis_mm=sigma_axis.reshape(3),
        weak_dir_marker=weak_dir.reshape(3),
        weak_sigma_mm=weak_sigma,
        cond_number=cond,
        per_frame_residual_mm=res_all.reshape(-1),
    )


@dataclass(frozen=True)
class AxisConstrainedTip:
    """Pivot tip constrained onto a known axis: p_tip = axis_point + s*axis_dir."""
    p_tip: np.ndarray            # (3,) tip on the axis, marker frame, mm
    axial_pos_mm: float          # s: signed offset from axis_point along axis_dir
    pivot_camera: np.ndarray     # (3,) pivot point in the camera frame, mm
    residual_rms_mm: float       # inlier per-frame residual RMS (constrained fit)
    sigma_axial_mm: float        # std of the single axial DoF
    axis_point: np.ndarray       # (3,) point on the axis (from the model)
    axis_dir: np.ndarray         # (3,) unit axis direction (from the model)
    p_tip_free: np.ndarray       # (3,) the unconstrained tip, for reference
    tip_offset_mm: float         # perpendicular distance of the free tip to axis
    n_inliers: int


def solve_pivot_on_axis(
    T_tc: np.ndarray,
    inlier_mask: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    p_tip_free: np.ndarray,
) -> AxisConstrainedTip:
    """Pivot solve with the tip constrained to a known axis line.

    The marker sits on the drill's cylindrical body, so the physical tip lies on
    the fitted cylinder axis. Substituting ``p_tip = p0 + s*d`` into the pivot
    equation ``R_i p_tip + t_i = c`` leaves a single axial unknown ``s`` and the
    pivot ``c``:  ``[R_i d | -I] [s; c] = -(R_i p0 + t_i)``. The well-fitted axis
    supplies the two lateral tip coordinates (the weakly observed off-axis DoF
    that dominates the monocular pivot error), pivoting only fixes the depth s.

    Runs on the free solve's inlier set. Raises on degenerate input.
    """
    T = np.asarray(T_tc, dtype=np.float64).reshape(-1, 4, 4)
    p0 = np.asarray(axis_point, dtype=np.float64).reshape(3)
    d = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    d = d / max(float(np.linalg.norm(d)), 1e-12)

    idx = np.flatnonzero(np.asarray(inlier_mask, dtype=bool))
    if idx.size < 3:
        raise ValueError("Axis-constrained pivot needs at least 3 inlier frames.")
    R = T[idx, :3, :3]
    t = T[idx, :3, 3]
    m = idx.size

    Rd = np.einsum("nij,j->ni", R, d)      # (m, 3)
    Rp0 = np.einsum("nij,j->ni", R, p0)    # (m, 3)
    A = np.zeros((3 * m, 4), dtype=np.float64)
    A[:, 0] = Rd.reshape(3 * m)
    A[:, 1:] = np.tile(-np.eye(3), (m, 1))
    b = -(Rp0 + t).reshape(3 * m)

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    s = float(x[0])
    c = x[1:4]
    p_tip = p0 + s * d

    res = np.linalg.norm(np.einsum("nij,j->ni", R, p_tip) + t - c, axis=1)
    rms = float(np.sqrt(np.mean(res * res)))

    try:
        ata_inv = np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        ata_inv = np.linalg.pinv(A.T @ A)
    dof = max(3 * m - 4, 1)
    sigma2 = float(np.sum(res * res)) / dof
    sigma_s = float(np.sqrt(max(sigma2 * float(ata_inv[0, 0]), 0.0)))

    pf = np.asarray(p_tip_free, dtype=np.float64).reshape(3)
    foot = p0 + float(np.dot(pf - p0, d)) * d
    tip_offset = float(np.linalg.norm(pf - foot))

    return AxisConstrainedTip(
        p_tip=p_tip.reshape(3),
        axial_pos_mm=s,
        pivot_camera=c.reshape(3),
        residual_rms_mm=rms,
        sigma_axial_mm=sigma_s,
        axis_point=p0.reshape(3),
        axis_dir=d.reshape(3),
        p_tip_free=pf.reshape(3),
        tip_offset_mm=tip_offset,
        n_inliers=int(m),
    )








def _cone_coverage_deg(T_list: Sequence[np.ndarray]) -> float:
    """Max half-angle of the marker z-axis spread seen so far (coverage proxy)."""
    if len(T_list) < 2:
        return 0.0
    z = np.asarray([np.asarray(T)[:3, 2] for T in T_list], dtype=np.float64)
    mean = z.mean(axis=0)
    n = float(np.linalg.norm(mean))
    if n < 1e-9:
        return 180.0
    mean /= n
    angs = np.degrees(np.arccos(np.clip(z @ mean, -1.0, 1.0)))
    return float(np.max(angs))


class PivotRecorder:
    """Continuous marker-pose recorder for a spherical-adapter pivot sweep."""

    def __init__(
        self,
        *,
        out_dir: Path,
        sample_interval_frames: int,
        marker_corner_ids: np.ndarray,
        marker_corners_xyz_mm: np.ndarray,
        marker_json_path: str,
        field_path: str = "",
        save_images: bool = True,
        require_ir: bool = False,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.session_path = self.out_dir / PIVOT_SESSION_FILENAME
        self.sample_interval_frames = max(1, int(sample_interval_frames))
        self.marker_corner_ids = marker_corner_ids
        self.marker_corners_xyz_mm = marker_corners_xyz_mm
        self.marker_json_path = str(marker_json_path)
        self.field_path = str(field_path)

        # Raw frames are saved (synchronised to the logged poses via frame index)
        # so the tool-body ring/edge geometry can be calibrated OFFLINE from the
        # same pivot sweep (the wide tilt is exactly what that joint fit needs).
        # Encoding runs on a background writer thread so the heavy PNG compression
        # never blocks the real-time tracking loop (that caused recording lag).
        self.save_images = bool(save_images)
        self.frames_dir = self.out_dir / PIVOT_FRAMES_DIRNAME
        self._frames_dir_ready = False
        self._img_queue: queue.Queue | None = None
        self._img_thread: threading.Thread | None = None

        self.K: np.ndarray | None = None
        self.dist: np.ndarray | None = None
        self.recording = False
        self.frames: list[dict[str, Any]] = []
        self._last_sample_frame = -10**9
        self._last_solve_count = 0
        self._last_pivot: PivotResult | None = None

        # --ir runs: only log frames whose reported pose actually went through
        # the IR refinement (timings ir_applied). The reference-free QF-IR fusion
        # applies from frame 1 at any orientation, but it still gates off on
        # glare / adverse geometry / a rejected MAP fit - on those frames the
        # reported pose is RGB-only, and mixing that into an "IR" calibration
        # would reintroduce exactly the weak-mode bias the IR channel removes.
        self.require_ir = bool(require_ir)
        self.ir_applied_count = 0
        self.ir_skipped_count = 0
        # Latest IR pair, stashed by the process_frame wrapper (see
        # cmd_calibrate's after_tracker_created) so the same-frame IR images are
        # saved next to the RGB frame for offline analysis / re-pivoting.
        self._last_ir_left: np.ndarray | None = None
        self._last_ir_right: np.ndarray | None = None

    def set_last_ir(self, ir_left, ir_right) -> None:
        self._last_ir_left = ir_left
        self._last_ir_right = ir_right

    def set_camera(self, K: np.ndarray, dist: np.ndarray) -> None:
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    # -- background image writer (keeps PNG encoding off the tracking loop) --
    def _ensure_writer(self) -> None:
        if self._img_thread is None:
            self._img_queue = queue.Queue(maxsize=512)
            self._img_thread = threading.Thread(
                target=self._writer_loop, name="pivot-img-writer", daemon=True
            )
            self._img_thread.start()

    def _writer_loop(self) -> None:
        assert self._img_queue is not None
        while True:
            path, arr = self._img_queue.get()
            try:
                cv2.imwrite(path, arr)
            except Exception:  # never let a bad write kill the writer thread
                pass
            finally:
                self._img_queue.task_done()

    def flush_images(self) -> None:
        """Block until every queued frame has been written to disk."""
        if self._img_queue is not None:
            self._img_queue.join()

    def toggle(self) -> None:
        self.recording = not self.recording
        if self.recording:
            print(
                f"[calib_tool_tip] Pivot recording running - rotate the drill in the socket "
                f"(already {len(self.frames)} frames)."
            )
        else:
            self.flush_images()
            self._write_session()
            print(
                f"[calib_tool_tip] Pivot recording paused at {len(self.frames)} frames "
                f"-> {self.session_path.name}."
            )

    def clear(self) -> None:
        self.flush_images()  # drain pending writes so nothing re-appears after the wipe
        self.frames = []
        self._last_pivot = None
        self._last_solve_count = 0
        try:
            self.session_path.unlink()
        except OSError:
            pass
        # Drop any saved raw frames from the discarded session.
        if self.frames_dir.is_dir():
            for img in self.frames_dir.glob("frame_*.png"):
                try:
                    img.unlink()
                except OSError:
                    pass
        self._frames_dir_ready = False
        print("[calib_tool_tip] Pivot recording reset.")

    def feed(self, frame_idx: int, result: Any, raw_frame: np.ndarray | None = None) -> None:
        from tracking.hydramarker import tracker_log

        if not self.recording or self.K is None:
            return
        if not tracker_log.has_fresh_pose(result) or result.rvec is None:
            return
        if self.require_ir:
            timings = getattr(result, "timings_ms", None) or {}
            if not bool(timings.get("ir_applied", 0.0)):
                # IR gated off for this frame (glare / adverse geometry / MAP
                # reject): do NOT feed the RGB-only pose into the IR pivot solve.
                # The interval counter is not advanced so the next IR-refined
                # frame samples.
                self.ir_skipped_count += 1
                return
            self.ir_applied_count += 1
        if int(frame_idx) - self._last_sample_frame < self.sample_interval_frames:
            return
        self._last_sample_frame = int(frame_idx)
        T_tc = ccb.make_transform_from_rvec_tvec(result.rvec, result.tvec)
        # Log the matched corner observations (marker-frame 3D + image uv) so a
        # constrained bundle adjustment can be run offline (pivot-solve --ba).
        corners = getattr(result, "corners", None) or []
        cxyz = np.asarray([c.xyz_mm for c in corners], dtype=np.float64).reshape(-1, 3)
        cuv = np.asarray([c.uv for c in corners], dtype=np.float64).reshape(-1, 2)
        self.frames.append(
            {
                "T_tc": T_tc, "timestamp": time.time(), "frame_index": int(frame_idx),
                "corners_xyz": cxyz, "corners_uv": cuv,
            }
        )
        # Persist the raw frame under its frame index (implicit pose<->image link).
        # A copy is enqueued (fast memcpy); the PNG encode happens off-loop.
        if self.save_images and raw_frame is not None:
            if not self._frames_dir_ready:
                self.frames_dir.mkdir(parents=True, exist_ok=True)
                for old in self.frames_dir.glob("frame_*.png"):  # wipe stale prior-run frames
                    try:
                        old.unlink()
                    except OSError:
                        pass
                self._ensure_writer()
                self._frames_dir_ready = True
            stem = f"frame_{int(frame_idx):06d}"
            try:
                self._img_queue.put_nowait(
                    (str(self.frames_dir / f"{stem}.png"), raw_frame.copy())
                )
                # Save the same-frame IR pair (global-shutter) alongside the RGB
                # frame so the pivot sweep can be replayed / re-analysed offline
                # exactly like the tip-test --save-frames triplets.
                if self._last_ir_left is not None:
                    self._img_queue.put_nowait(
                        (str(self.frames_dir / f"{stem}_irL.png"),
                         self._last_ir_left.copy()))
                if self._last_ir_right is not None:
                    self._img_queue.put_nowait(
                        (str(self.frames_dir / f"{stem}_irR.png"),
                         self._last_ir_right.copy()))
            except queue.Full:
                # writer fell behind (slow disk): drop this frame's IMAGE only, the
                # pose is still logged. Better a sparse image set than reintroducing lag.
                pass
        if (
            len(self.frames) >= PIVOT_MIN_FRAMES
            and len(self.frames) - self._last_solve_count >= 15
        ):
            self._last_solve_count = len(self.frames)
            try:
                self._last_pivot = solve_pivot_robust(
                    np.stack([f["T_tc"] for f in self.frames])
                )
            except (ValueError, np.linalg.LinAlgError):
                self._last_pivot = None
            self._write_session()

    def _write_session(self) -> None:
        if not self.frames:
            return
        cxyz = [f.get("corners_xyz", np.zeros((0, 3))) for f in self.frames]
        cuv = [f.get("corners_uv", np.zeros((0, 2))) for f in self.frames]
        counts = np.asarray([len(x) for x in cxyz], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        obs_xyz = np.vstack(cxyz) if counts.sum() else np.zeros((0, 3))
        obs_uv = np.vstack(cuv) if counts.sum() else np.zeros((0, 2))
        # Camera intrinsics + frames dir travel with the session so the offline
        # collar/ring calibration can project the SfM axis and measure edges.
        K = self.K if self.K is not None else np.full((3, 3), np.nan)
        dist = self.dist if self.dist is not None else np.zeros((0, 1))
        _atomic_savez(
            self.session_path,
            T_tc=np.stack([f["T_tc"] for f in self.frames]).astype(np.float64),
            timestamps=np.asarray([f["timestamp"] for f in self.frames], dtype=np.float64),
            frame_indices=np.asarray([f["frame_index"] for f in self.frames], dtype=np.int64),
            marker_corners_xyz_mm=np.asarray(self.marker_corners_xyz_mm, dtype=np.float64),
            marker_corner_ids=np.asarray(self.marker_corner_ids, dtype=np.int64),
            obs_corners_xyz_mm=obs_xyz.astype(np.float64),
            obs_corners_uv=obs_uv.astype(np.float64),
            obs_frame_offsets=offsets,
            camera_K=np.asarray(K, dtype=np.float64),
            camera_dist=np.asarray(dist, dtype=np.float64),
            frames_dirname=PIVOT_FRAMES_DIRNAME if self.save_images else "",
            marker_json_path=self.marker_json_path,
            field_path=self.field_path,
            created_utc=_utc_now_str(),
        )

    def render(self, vis: np.ndarray, result: Any) -> np.ndarray:
        from tracking.hydramarker import tracker_log

        fresh = tracker_log.has_fresh_pose(result)
        if fresh:
            _draw_tracked_corners(vis, result)

        coverage = _cone_coverage_deg([f["T_tc"] for f in self.frames])
        if self.recording:
            headline = f"RECORDING - {len(self.frames)} frames collected"
            color = (0, 220, 255)
        else:
            headline = (
                f"PAUSED - SPACE start/stop ({len(self.frames)} frames) | "
                f"Marker pose: {'OK' if fresh else 'MISSING'}"
            )
            color = (0, 255, 0) if fresh else (0, 0, 255)

        lines = [
            headline,
            f"Orientation coverage: {coverage:.0f} deg (goal: rotate as widely as possible)",
        ]
        if self.require_ir:
            total = self.ir_applied_count + self.ir_skipped_count
            rate = 100.0 * self.ir_applied_count / total if total else 0.0
            lines.append(
                f"IR refine: applied {self.ir_applied_count} | skipped {self.ir_skipped_count} "
                f"({rate:.0f}%) - sweep continuously & slowly; IR is reference-free (frame 1)"
            )
        if self._last_pivot is not None:
            pv = self._last_pivot
            lines.append(
                f"live p_tip=({pv.p_tip[0]:.1f},{pv.p_tip[1]:.1f},{pv.p_tip[2]:.1f}) mm "
                f"|r|={pv.radius_mm:.1f}"
            )
            lines.append(
                f"Residual {pv.residual_rms_mm:.2f} mm | inliers {pv.n_inliers}/{pv.n_frames} "
                f"| weakest axis {pv.weak_sigma_mm:.2f} mm"
            )
        lines.append("Keys: SPACE start/stop  c=discard  q=quit")
        return ccb._draw_text_box(vis, lines, color=color)








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








# ---------------------------------------------------------------------------
# tip-log (Phase-1-Diagnose: Live-Tip-Position mitschreiben)
# ---------------------------------------------------------------------------

TIP_LOG_WINDOW_NAME = "HydraMarker Tip-Log (Diagnose)"






CALIB_DIR = Path(__file__).resolve().parent


class _SessionDone(BaseException):
    """Raised inside the live-view key handler to end the recording session.

    It is a BaseException (not Exception) on purpose: run_tracker wraps the key
    callback in ``except Exception``, so this passes straight through, the
    tracker's ``finally`` closes the camera/window, and cmd_calibrate catches it
    to continue with the solver. Keeps the hard stop entirely in this script."""


def _show_pivot_3d(
    result: PivotResult,
    corners_xyz: np.ndarray,
    corner_ids: np.ndarray,
    surface_model_json: str,
    save_path: Path,
    *,
    axis_tip: "AxisConstrainedTip | None" = None,
) -> None:
    """3D figure in the marker frame: SfM corner cloud + fitted cylinder + tip.

    When ``axis_tip`` is given, the drill axis is drawn (cylinder centre -> 10 mm
    past the tip), the constrained tip sits ON it, and the unconstrained tip is
    shown with its perpendicular offset to the axis.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure("Tip Calibration - 3D geometry (marker frame)", figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        corners_xyz[:, 0], corners_xyz[:, 1], corners_xyz[:, 2],
        c=corner_ids if corner_ids is not None and len(corner_ids) else None,
        s=18, label="Marker corners (SfM model)",
    )
    if corner_ids is not None and len(corner_ids):
        fig.colorbar(scatter, ax=ax, shrink=0.6, label="Corner ID")

    wire = _cylinder_wireframe_points(surface_model_json)
    if wire is not None:
        ax.plot_wireframe(*wire, color="gray", alpha=0.25, linewidth=0.6)

    # marker-frame ORIGIN (= corner id 0, the outer grid corner) with a 10 mm
    # axis triad in the OpenCV drawFrameAxes colours (x red, y green, z blue)
    axis_mm = 10.0
    ax.scatter([0.0], [0.0], [0.0], color="black", marker="*", s=260,
               label="Marker origin (0,0,0)")
    for direction, color, label in (
        ((axis_mm, 0.0, 0.0), "red", "+x"),
        ((0.0, axis_mm, 0.0), "green", "+y"),
        ((0.0, 0.0, axis_mm), "blue", "+z"),
    ):
        ax.quiver(0.0, 0.0, 0.0, *direction, color=color,
                  linewidth=2.2, arrow_length_ratio=0.25)
        ax.text(direction[0] * 1.25, direction[1] * 1.25, direction[2] * 1.25,
                label, color=color, fontsize=10, weight="bold")

    tip_free = np.asarray(result.p_tip, dtype=np.float64).reshape(3)
    if axis_tip is not None:
        p0 = np.asarray(axis_tip.axis_point, dtype=np.float64).reshape(3)
        d = np.asarray(axis_tip.axis_dir, dtype=np.float64).reshape(3)
        tip = np.asarray(axis_tip.p_tip, dtype=np.float64).reshape(3)

        # Drill axis: from the cylinder centre along the axis out to 10 mm past
        # the tip (NOT a centre-to-tip chord). The constrained tip sits ON it.
        along = corners_xyz @ d
        along_mid = 0.5 * (float(along.min()) + float(along.max()))
        t_tip = float(np.dot(tip - p0, d))
        sgn = 1.0 if t_tip >= along_mid else -1.0
        a0 = p0 + along_mid * d
        a1 = p0 + (t_tip + 10.0 * sgn) * d
        ax.plot([a0[0], a1[0]], [a0[1], a1[1]], [a0[2], a1[2]],
                color="navy", linewidth=2.2, label="drill axis (cylinder)")
        ax.scatter([tip[0]], [tip[1]], [tip[2]], color="crimson", marker="*", s=240,
                   label=f"tip on axis  |r|={float(np.linalg.norm(tip)):.1f} mm")

        # Where the unconstrained solve landed + its perpendicular offset.
        foot = p0 + float(np.dot(tip_free - p0, d)) * d
        ax.plot([tip_free[0], foot[0]], [tip_free[1], foot[1]], [tip_free[2], foot[2]],
                color="darkorange", linewidth=1.8, linestyle="--",
                label=f"free-tip offset = {axis_tip.tip_offset_mm:.2f} mm")
        ax.scatter([tip_free[0]], [tip_free[1]], [tip_free[2]],
                   color="0.4", marker="x", s=60, label="free tip (unconstrained)")
        title = (
            f"SfM model + tip on drill axis  (constrained residual "
            f"{axis_tip.residual_rms_mm:.2f} mm, free-tip offset "
            f"{axis_tip.tip_offset_mm:.2f} mm)"
        )
    else:
        ax.scatter(
            [tip_free[0]], [tip_free[1]], [tip_free[2]], color="crimson", marker="*", s=220,
            label=f"Tip p_tip  |r|={result.radius_mm:.1f} mm",
        )
        title = (
            f"SfM model + tool tip  (residual {result.residual_rms_mm:.2f} mm, "
            f"sigma_norm {result.sigma_norm_mm:.3f} mm)"
        )
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    _set_axes_equal_3d(ax)
    fig.tight_layout()
    try:
        fig.savefig(save_path, dpi=120)
        print(f"[calib_tool_tip] 3D figure saved: {save_path}")
    except OSError as exc:
        print(f"[calib_tool_tip] could not save 3D figure: {exc}")
    plt.show()


def _finalize_pivot(
    T_tc: np.ndarray,
    marker_corners_xyz: np.ndarray,
    marker_corner_ids: np.ndarray,
    surface_json: str,
    marker_json_path: str,
    out_dir: Path,
    *,
    axis_constrained: bool = True,
) -> tuple[PivotResult, "AxisConstrainedTip | None"]:
    """Solve (robust free + optional axis-constrained), report, save, show figure.

    Shared by the live calibration and the offline re-solve. Writes
    ``pivot_calibration.npz`` and ``pivot_tip_3d.png`` into ``out_dir``.
    """
    T_tc = np.asarray(T_tc, dtype=np.float64).reshape(-1, 4, 4)
    out_dir = Path(out_dir)
    n = T_tc.shape[0]
    print(f"\n[calib_tool_tip] Solving pivot over {n} frames (robust solver) ...")
    result = solve_pivot_robust(T_tc)

    p = result.p_tip
    s = result.sigma_axis_mm
    print("\n===== Tip calibration (free solve) =====")
    print(
        f"p_tip (sphere centre, marker frame) = "
        f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) mm, Radius {result.radius_mm:.3f} mm"
    )
    print(f"sigma per axis = ({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f}) mm, norm {result.sigma_norm_mm:.3f} mm")
    print(
        f"Residual RMS = {result.residual_rms_mm:.3f} mm | inliers "
        f"{result.n_inliers}/{result.n_frames} | condition number {result.cond_number:.1f}"
    )
    wd = result.weak_dir_marker
    print(
        f"Weakest axis: sigma {result.weak_sigma_mm:.3f} mm along "
        f"({wd[0]:+.2f},{wd[1]:+.2f},{wd[2]:+.2f}) [marker frame]"
    )
    if result.weak_sigma_mm > 3.0 * float(np.median(result.sigma_axis_mm) + 1e-9):
        print("  -> WARNING: one direction poorly observed - rotate further / more evenly.")

    # --- Constrain the tip to the known drill (cylinder) axis --------------
    axis = _drill_axis_from_surface(surface_json)
    axis_tip: "AxisConstrainedTip | None" = None
    if axis is None:
        print("[calib_tool_tip] No fitted cylinder axis in the marker JSON - "
              "cannot constrain the tip; keeping the free solve.")
    elif not axis_constrained:
        print("[calib_tool_tip] --free: keeping the UNCONSTRAINED tip "
              "(not projected onto the drill axis).")
    else:
        try:
            axis_tip = solve_pivot_on_axis(
                T_tc, result.inlier_mask, axis[0], axis[1], result.p_tip
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"[calib_tool_tip] axis-constrained solve failed ({exc}); "
                  "keeping the free tip.")

    if axis_tip is not None:
        at = axis_tip
        print("\n----- Tip constrained to the drill axis -----")
        print(f"free tip was {at.tip_offset_mm:.3f} mm off the axis (removed by the constraint).")
        print(
            f"p_tip (on axis, marker frame) = "
            f"({at.p_tip[0]:.3f}, {at.p_tip[1]:.3f}, {at.p_tip[2]:.3f}) mm, "
            f"Radius {float(np.linalg.norm(at.p_tip)):.3f} mm"
        )
        print(
            f"axial position s = {at.axial_pos_mm:.3f} mm (+-{at.sigma_axial_mm:.3f} mm) | "
            f"constrained residual RMS = {at.residual_rms_mm:.3f} mm "
            f"(free {result.residual_rms_mm:.3f} mm)"
        )

    final_tip = axis_tip.p_tip if axis_tip is not None else result.p_tip

    result_path = out_dir / PIVOT_RESULT_FILENAME
    save_kw: dict[str, Any] = dict(
        p_tip=final_tip,
        p_tip_pose_level=final_tip,
        p_tip_free=result.p_tip,
        axis_constrained=np.bool_(axis_tip is not None),
        method="pivot_sphere_axis" if axis_tip is not None else "pivot_sphere",
        sigma_norm_mm=np.float64(
            axis_tip.sigma_axial_mm if axis_tip is not None else result.sigma_norm_mm
        ),
        sigma_axis_mm=result.sigma_axis_mm,
        pivot_camera=(axis_tip.pivot_camera if axis_tip is not None else result.pivot_camera),
        residual_rms_mm=np.float64(
            axis_tip.residual_rms_mm if axis_tip is not None else result.residual_rms_mm
        ),
        residual_rms_free_mm=np.float64(result.residual_rms_mm),
        n_frames=np.int32(result.n_frames),
        n_inliers=np.int32(result.n_inliers),
        inlier_mask=result.inlier_mask,
        weak_dir_marker=result.weak_dir_marker,
        weak_sigma_mm=np.float64(result.weak_sigma_mm),
        cond_number=np.float64(result.cond_number),
        radius_mm=np.float64(float(np.linalg.norm(final_tip))),
        marker_json_path=str(marker_json_path),
        created_utc=_utc_now_str(),
    )
    if axis_tip is not None:
        save_kw.update(
            axis_point_mm=axis_tip.axis_point,
            axis_dir=axis_tip.axis_dir,
            axial_pos_mm=np.float64(axis_tip.axial_pos_mm),
            sigma_axial_mm=np.float64(axis_tip.sigma_axial_mm),
            tip_offset_mm=np.float64(axis_tip.tip_offset_mm),
        )
    np.savez(result_path, **save_kw)
    print(f"\n[calib_tool_tip] Result saved: {result_path}")

    _show_pivot_3d(
        result, marker_corners_xyz, marker_corner_ids, surface_json,
        out_dir / "pivot_tip_3d.png", axis_tip=axis_tip,
    )
    return result, axis_tip


def cmd_calibrate(
    _args: argparse.Namespace,
    enable_ir: bool = False,
    axis_constrained: bool = True,
) -> int:
    """Streamlined default: record a pivot sweep, close the camera on SPACE,
    solve the tip, save everything into calib/, and show the 3D SfM+tip figure.
    No sub-commands or options -- just run the file.

    enable_ir (--ir): calibrate the pivot on the IR-refined pose channel
    (design B). The solver consumes the REPORTED per-frame pose, so with the
    IR stage active P_TIP is estimated in the IR channel's frame - this is the
    re-pivot that absorbs the constant RGB<->IR extrinsic residual. Calibrate
    with --ir when tracking runs with --ir (and without when it does not)."""
    from tracking.hydramarker import run_tracker as rt
    from tracking.hydramarker.config import LiveTrackerConfig, LoggingConfig

    print("[calib_tool_tip] Tip calibration (pivot). Please choose files ...")
    field_path = rt.choose_file_qt(
        "Select HydraMarker .field", "HydraMarker field (*.field)"
    )
    marker_json_path = rt.choose_file_qt(
        "Select marker .json", "Marker JSON (*.json)"
    )
    calibration_path = rt.choose_file_qt(
        "Select camera calibration .npz", "NPZ files (*.npz)"
    )
    marker_corner_ids, marker_corners_xyz, surface_json = load_marker_geometry(
        marker_json_path
    )

    recorder = PivotRecorder(
        out_dir=CALIB_DIR,
        sample_interval_frames=2,
        marker_corner_ids=marker_corner_ids,
        marker_corners_xyz_mm=marker_corners_xyz,
        marker_json_path=str(marker_json_path),
        field_path=str(field_path),
        require_ir=enable_ir,
    )
    def after_camera_ready(camera, K, dist) -> bool:
        recorder.set_camera(K, dist)
        return True

    def on_frame(frame_idx, raw_frame, result, _logging_active) -> None:
        recorder.feed(frame_idx, result, raw_frame)

    def on_key(key, _tracker, _last_result, _frame_idx) -> bool:
        if key == ord(" "):
            recorder.toggle()
            # Second SPACE (recording just turned off with data) hard-ends the
            # live session: raise past run_tracker's ``except Exception`` so its
            # ``finally`` closes the camera, then cmd_calibrate runs the solver.
            if not recorder.recording and recorder.frames:
                raise _SessionDone
            return True
        if key == ord("c"):
            recorder.clear()
            return True
        return False

    def render_frame(vis, result, _tracker, _frame_idx, _logging_active):
        return recorder.render(vis, result)

    def after_tracker_created(tracker) -> None:
        # --ir: capture the per-frame IR pair (process_frame's ir_left/ir_right
        # kwargs) so the recorder can save the same-frame IR images next to the
        # RGB frame. RGB-only runs skip this (no IR to store).
        if not enable_ir:
            return
        _orig_process = tracker.process_frame

        def _process_capture_ir(image_bgr, *args, **kwargs):
            recorder.set_last_ir(kwargs.get("ir_left"), kwargs.get("ir_right"))
            return _orig_process(image_bgr, *args, **kwargs)

        tracker.process_frame = _process_capture_ir

    config = LiveTrackerConfig(
        tracker=rt.make_live_tracker_config(),
        logging=LoggingConfig(enabled=False, start_mode="disabled"),
        start_tracking_manually=False,
    )
    config.camera.calibration_path = str(calibration_path)
    # RAW pose: the robust pivot least-squares wants independent per-frame
    # measurements; the output filter would distort exactly the rotation it uses.
    config.tracker.pose_kf_enabled = False
    if enable_ir:
        # IR refinement stays: it is a per-frame MEASUREMENT (not smoothing),
        # so the pivot solve gets independent, shutter-free poses.
        config.camera.realsense_enable_ir = True
        config.tracker.ir_refine_enabled = True
        print("[calib_tool_tip] IR pose refinement ENABLED - P_TIP will be "
              "calibrated on the IR channel (use --ir tracking with this tip).")
    print(
        "[calib_tool_tip] SPACE starts recording, SPACE stops + closes the "
        "camera and solves.  c = discard.  Tilt the drill widely in the divot (slowly)."
    )

    try:
        rt.run_tracker(
            config=config,
            field_path=field_path,
            marker_json_path=marker_json_path,
            window_name=PIVOT_WINDOW_NAME,
            console_prefix="[calib_tool_tip]",
            after_camera_ready=after_camera_ready,
            after_tracker_created=after_tracker_created,
            on_frame=on_frame,
            on_key=on_key,
            render_frame=render_frame,
        )
    except _SessionDone:
        print("[calib_tool_tip] Recording ended via SPACE, camera closed.")

    # --- camera is closed here; run the solver ---
    recorder.flush_images()  # ensure every queued frame is on disk before we report
    recorder._write_session()
    n = len(recorder.frames)
    if n < PIVOT_MIN_FRAMES:
        print(f"[calib_tool_tip] Only {n} frames (< {PIVOT_MIN_FRAMES}) - no solution.")
        return 1

    if enable_ir:
        total_ir = recorder.ir_applied_count + recorder.ir_skipped_count
        rate = 100.0 * recorder.ir_applied_count / total_ir if total_ir else 0.0
        print(
            f"[calib_tool_tip] IR gate: {recorder.ir_applied_count} frames logged WITH "
            f"IR refinement, {recorder.ir_skipped_count} rejected without ({rate:.0f}% applied)."
        )

    T_tc = np.stack([f["T_tc"] for f in recorder.frames]).astype(np.float64)
    _finalize_pivot(
        T_tc, marker_corners_xyz, marker_corner_ids, surface_json,
        str(marker_json_path), CALIB_DIR, axis_constrained=axis_constrained,
    )

    _fd = CALIB_DIR / PIVOT_FRAMES_DIRNAME
    n_imgs = len(list(_fd.glob("frame_??????.png"))) if _fd.is_dir() else 0
    n_ir = len(list(_fd.glob("frame_*_irL.png"))) if _fd.is_dir() else 0
    print(f"\n[calib_tool_tip] Files in {CALIB_DIR}:")
    print(f"  - {PIVOT_SESSION_FILENAME}  (recording + K/dist)")
    if n_imgs:
        _ir_note = f" + {n_ir} IR pairs" if n_ir else ""
        print(f"  - {PIVOT_FRAMES_DIRNAME}/  ({n_imgs} raw RGB frames{_ir_note} for offline analysis)")
    print(f"  - {PIVOT_RESULT_FILENAME}  (result, overlay-compatible)")
    print("  - pivot_tip_3d.png  (3D figure SfM + tip)")
    return 0


def cmd_solve_offline(
    session_path: str | Path | None = None,
    *,
    axis_constrained: bool = True,
) -> int:
    """Re-solve the pivot tip OFFLINE from a saved ``pivot_session.npz``.

    No camera / recording needed: the stored per-frame poses (T_tc) are re-run
    through the same (by default axis-constrained) solver. The drill axis is
    loaded from the marker JSON referenced by the session (its
    ``surface_model.fitted``); the result NPZ + 3D figure are written next to the
    session file. Pass ``--free`` to skip the axis constraint.
    """
    if session_path is None:
        from tracking.hydramarker import run_tracker as rt
        session_path = rt.choose_file_qt(
            "Select pivot_session.npz", "NPZ files (*.npz)"
        )
    session_path = Path(session_path)
    if not session_path.is_file():
        print(f"[calib_tool_tip] Session not found: {session_path}")
        return 1

    data = np.load(session_path, allow_pickle=True)
    if "T_tc" not in data.files:
        print(f"[calib_tool_tip] {session_path.name} has no 'T_tc' poses.")
        return 1
    T_tc = np.asarray(data["T_tc"], dtype=np.float64).reshape(-1, 4, 4)
    marker_corners_xyz = np.asarray(
        data["marker_corners_xyz_mm"], dtype=np.float64
    ).reshape(-1, 3) if "marker_corners_xyz_mm" in data.files else np.zeros((0, 3))
    marker_corner_ids = np.asarray(
        data["marker_corner_ids"], dtype=np.int64
    ).reshape(-1) if "marker_corner_ids" in data.files else np.zeros((0,), dtype=np.int64)
    marker_json_path = (
        str(data["marker_json_path"]) if "marker_json_path" in data.files else ""
    )

    # The drill axis lives in the marker JSON (surface_model.fitted); load it.
    surface_json = ""
    if marker_json_path and Path(marker_json_path).is_file():
        _ids, _xyz, surface_json = load_marker_geometry(Path(marker_json_path))
    else:
        from tracking.hydramarker import run_tracker as rt
        print("[calib_tool_tip] Marker JSON referenced by the session is missing "
              "- please pick the marker .json (needs surface_model.fitted).")
        marker_json_path = rt.choose_file_qt("Select marker .json", "Marker JSON (*.json)")
        _ids, _xyz, surface_json = load_marker_geometry(Path(marker_json_path))
    # Prefer the marker JSON's corner cloud for the figure if the session lacks it.
    if marker_corners_xyz.shape[0] == 0 and _xyz.shape[0]:
        marker_corners_xyz, marker_corner_ids = _xyz, _ids

    # If the constraint is wanted but this JSON has no fitted axis (e.g. the
    # session referenced the planar marker), let the user point to one that does.
    if axis_constrained and _drill_axis_from_surface(surface_json) is None:
        from tracking.hydramarker import run_tracker as rt
        print("[calib_tool_tip] The session's marker JSON has no surface_model.fitted "
              "axis - pick a marker JSON with the fitted cylinder (or run --free).")
        alt = rt.choose_file_qt("Select marker .json with fitted axis", "Marker JSON (*.json)")
        _ids2, _xyz2, surface_json = load_marker_geometry(Path(alt))
        marker_json_path = str(alt)
        if marker_corners_xyz.shape[0] == 0 and _xyz2.shape[0]:
            marker_corners_xyz, marker_corner_ids = _xyz2, _ids2

    n = T_tc.shape[0]
    print(f"[calib_tool_tip] OFFLINE re-solve from {session_path} ({n} frames).")
    if n < PIVOT_MIN_FRAMES:
        print(f"[calib_tool_tip] Note: only {n} frames (< {PIVOT_MIN_FRAMES}).")

    _finalize_pivot(
        T_tc, marker_corners_xyz, marker_corner_ids, surface_json,
        marker_json_path, session_path.parent, axis_constrained=axis_constrained,
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Running the script IS the tip (pivot) calibration.

    Options:
      --ir       calibrate on the IR-refined pose channel (see cmd_calibrate).
      --offline [pivot_session.npz]
                 skip the camera and re-solve from a saved session (a file
                 dialog opens if no path is given).
      --free     keep the plain unconstrained tip (do NOT project it onto the
                 drill axis). Default: the tip is constrained to the axis.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    axis_constrained = "--free" not in args
    if "--offline" in args:
        i = args.index("--offline")
        nxt = args[i + 1] if i + 1 < len(args) else None
        path = nxt if (nxt and not nxt.startswith("-")) else None
        return int(cmd_solve_offline(path, axis_constrained=axis_constrained))
    return int(
        cmd_calibrate(None, enable_ir="--ir" in args, axis_constrained=axis_constrained)
    )


if __name__ == "__main__":
    raise SystemExit(main())
