# overlay/tools/view_registration.py
"""Further X-ray views through a ChArUco board seen in every C-arm stance
(User Study DHS). Geometry and capture logic only, no Qt, no camera.

Frames and units, as in overlay.tools.xray_view:

    T_ab maps frame b into frame a; w = the board, c = the colour camera.
    T_cw   board -> camera, METRES (the object points are metres)
    T_xc   X-ray -> camera, METRES, the code's name (the thesis calls it T_cx)

Each X-ray view v is taken in its own C-arm stance, and right after it the
board pose of that stance, T_cw_v (1 s of colour frames, BoardPoseCapture).
Tracking runs in the stance the C-arm ends in, whose board pose is T_cw_now.
The T_xc of view v - its X-ray frame mapped into the camera as it is now - is

    T_xc_view = T_cw_now @ inv(T_cw_v) @ T_xc @ Rz(-phi_v)        (metres)

and XrayView(T_xc_view, K_x) draws the tracked tool into that view's image.
For the view taken in the stance the C-arm stays in, T_cw_v = T_cw_now and
phi = 0, so T_xc_view = T_xc. phi_v is the image rotation of the view's X-ray
frame about the beam: +1.73 deg measured for the thesis' lateral stance with
the camera below the image intensifier, not measured elsewhere.

Between view v and now the camera moved by T_cv_cnow = T_cw_v @ inv(T_cw_now)
(Vorgehen_Zwei_Ansichten: T_clca = T_clw inv(T_caw)); camera_motion() gives
its angle, axis and translation for the display.

The board pose (lab test 10.11.2026, tests/charuco_two_view_test.py):
- corners: sub-pixel ArUco corners, ChArUco interpolation with K and dist (as
  the colour intrinsics calibration), then cornerSubPix with a half window of
  10 % of the corner spacing ('adaptive'). The calibration's 25 % window
  reaches into the markers (single corners up to 1.5 px off on synthetic
  images); CharucoDetector alone returns the corners about +0.5 px off in x
  and y. 'adaptive' lies 0.03 px from the truth.
- 1 s window, n_frames frames in a row; the median corner shift against the
  first frame of the window must stay below motion_px, otherwise the window
  starts again (the C-arm still swings after a turn).
- every corner averaged over the frames, then IPPE (both planar solutions),
  each refined with Levenberg-Marquardt; the better one is the pose.
- rejected: too few corners, reprojection too large, ambiguous planar pose,
  C-arm not still, board missing in too many frames.
Measured with this: six 1 s poses in one stance agree to 0.012 mm / 0.003 deg,
the relative pose of 30 lateral x AP pairs to 0.05 mm at a target 430 mm out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

MIN_POSE_CORNERS = 6
PHI_LATERAL_DEG = 1.73

# |t| of a real set-up is about a metre; a value above this is millimetres.
_MAX_T_NORM_M = 10.0


# ============================================================
# Board and settings
# ============================================================

@dataclass(frozen=True)
class BoardSpec:
    """The ChArUco board of the colour-camera calibration (9 x 7 squares,
    48 inner corners, DICT_5X5_50). The pitch actually used for the object
    points is BoardPoseConfig.pitch_mm; the markers scale with it."""
    squares_x: int = 9
    squares_y: int = 7
    pitch_nominal_m: float = 0.0254
    marker_nominal_m: float = 0.01778
    dict_id: int = cv2.aruco.DICT_5X5_50

    @property
    def n_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    def make(self, pitch_m: float):
        dic = cv2.aruco.getPredefinedDictionary(self.dict_id)
        marker_m = self.marker_nominal_m * pitch_m / self.pitch_nominal_m
        board = cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), pitch_m, marker_m, dic)
        return board, dic

    def as_dict(self) -> dict:
        return {"squares_x": self.squares_x, "squares_y": self.squares_y,
                "pitch_nominal_mm": self.pitch_nominal_m * 1e3,
                "marker_nominal_mm": self.marker_nominal_m * 1e3,
                "aruco_dictionary_id": self.dict_id, "n_corners": self.n_corners}


DEFAULT_BOARD = BoardSpec()


@dataclass
class BoardPoseConfig:
    pitch_mm: float = 25.4          # measured square pitch (10.11.: matches nominal)
    n_frames: int = 30              # window, 1 s at 30 fps
    motion_px: float = 0.30         # max median corner shift vs. first frame of the window
    min_corners: int = 30           # the lab position shows 34-35 of 48
    min_presence: float = 0.9       # a corner must be found in this share of frames
    max_rms_px: float = 0.40        # reprojection RMS per axis of the mean corners
    min_ippe_ratio: float = 2.0     # RMS of the 2nd planar solution / best one
    max_flip_deg: float = 2.0       # single-frame pose this far off the mean = flipped
    max_missing_frames: float = 0.1  # share of the window without the board
    max_wait_frames: int = 300      # BoardPoseCapture gives up after this many frames
    subpix: str = "adaptive"        # cornerSubPix window, see SUBPIX_MODES


# ============================================================
# Corner detection
# ============================================================

@dataclass
class Detection:
    ids: np.ndarray             # (n,) int, ChArUco corner ids
    corners: np.ndarray         # (n, 2) float64 [px]
    marker_ids: np.ndarray      # (m,) int
    marker_corners: np.ndarray  # (m, 4, 2) float64 [px]


_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# (share of the median corner spacing, min, max) of the cornerSubPix half
# window. The white rim between a corner and the next marker is 15 % of the
# square (25.4 vs 17.78 mm): 'calib' (the intrinsics calibration's choice)
# reaches into the markers, 'adaptive' stays inside, 'opencv' = no extra step.
SUBPIX_MODES = {"adaptive": (0.10, 3, 7), "calib": (0.25, 5, 15), "opencv": None}


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _subpix_half_window(points: np.ndarray, mode: str = "adaptive") -> int:
    share, lo, hi = SUBPIX_MODES[mode]
    if points.shape[0] < 2:
        return lo
    diff = points[None, :, :] - points[:, None, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(d, np.inf)
    spacing = float(np.median(np.min(d, axis=1)))
    if not np.isfinite(spacing):
        return lo
    return int(np.clip(round(spacing * share), lo, hi))


class CornerDetector:
    """detect_charuco(..., camera_matrix=K, dist_coeffs=dist) of
    tracking/hydramarker/calib/calib_camera.py up to the interpolation, then
    the cornerSubPix window of SUBPIX_MODES. 'calib' is that detection exactly
    (checked by tests/charuco_two_view_test.py --selftest). Images: BGR, BGRA
    or grey."""

    def __init__(self, board, dic, K: np.ndarray, dist: np.ndarray, subpix: str = "adaptive"):
        if subpix not in SUBPIX_MODES:
            raise ValueError(f"subpix must be one of {list(SUBPIX_MODES)}")
        self.subpix = subpix
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco = cv2.aruco.ArucoDetector(dic, p)
        cp = cv2.aruco.CharucoParameters()
        cp.cameraMatrix = np.asarray(K, dtype=np.float64).reshape(3, 3)
        cp.distCoeffs = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
        cp.minMarkers = 1
        self.charuco = cv2.aruco.CharucoDetector(board, charucoParams=cp)

    def detect(self, img: np.ndarray) -> Detection:
        gray = _gray(img)
        mc, mi, _ = self.aruco.detectMarkers(gray)
        empty = Detection(np.zeros(0, int), np.zeros((0, 2)), np.zeros(0, int), np.zeros((0, 4, 2)))
        if mi is None or len(mi) == 0:
            return empty
        marker_ids = mi.ravel().astype(int)
        marker_corners = np.array([c.reshape(4, 2) for c in mc], dtype=np.float64)
        cc, ci, _, _ = self.charuco.detectBoard(gray, markerCorners=mc, markerIds=mi)
        if cc is None or ci is None or len(ci) == 0:
            return Detection(empty.ids, empty.corners, marker_ids, marker_corners)
        pts = cc.reshape(-1, 2).astype(np.float64)
        ids = ci.ravel().astype(int)

        # drop extrapolated corners outside the image (margin 2 px)
        h, w = gray.shape[:2]
        keep = ((pts[:, 0] >= 2.0) & (pts[:, 0] <= w - 3.0)
                & (pts[:, 1] >= 2.0) & (pts[:, 1] <= h - 3.0))
        pts, ids = pts[keep], ids[keep]
        if ids.size == 0:
            return Detection(empty.ids, empty.corners, marker_ids, marker_corners)
        if self.subpix == "opencv":
            return Detection(ids, pts, marker_ids, marker_corners)

        # cornerSubPix, only where the window fits into the image
        half = _subpix_half_window(pts, self.subpix)
        margin = float(half + 2)
        safe = ((pts[:, 0] >= margin) & (pts[:, 0] <= w - 1.0 - margin)
                & (pts[:, 1] >= margin) & (pts[:, 1] <= h - 1.0 - margin))
        if np.any(safe):
            ref = cv2.cornerSubPix(gray, pts[safe].reshape(-1, 1, 2).astype(np.float32),
                                   (half, half), (-1, -1), _SUBPIX_CRITERIA)
            pts[safe] = ref.reshape(-1, 2).astype(np.float64)
        return Detection(ids, pts, marker_ids, marker_corners)


# ============================================================
# Geometry
# ============================================================

_LM_CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-12)


def to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def inv_T_batch(T: np.ndarray) -> np.ndarray:
    R = np.swapaxes(T[..., :3, :3], -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = R
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", R, T[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def rot_z(deg: float) -> np.ndarray:
    """4x4 rotation about z (the beam axis of an X-ray frame)."""
    a = np.radians(deg)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return T


def rotvec(R: np.ndarray) -> np.ndarray:
    rv, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return rv.ravel()


def rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(rotvec(R))))


def mean_rotation(Rs: np.ndarray) -> np.ndarray:
    """Chordal L2 mean of rotation matrices."""
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def project(obj: np.ndarray, T: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    p, _ = cv2.projectPoints(np.asarray(obj, dtype=np.float64), rotvec(T[:3, :3]),
                             T[:3, 3].copy(), K, dist)
    return p.reshape(-1, 2)


def solve_board_pose(obj: np.ndarray, img: np.ndarray, K: np.ndarray, dist: np.ndarray) -> dict:
    """IPPE gives both planar solutions; each is refined with LM. Returns the
    better one (T board -> camera, metres) with its RMS per axis and residuals,
    plus RMS and rotation offset of the other one."""
    obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img, dtype=np.float64).reshape(-1, 2)
    _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    sols = []
    for rv, tv in zip(rvecs, tvecs):
        rv, tv = cv2.solvePnPRefineLM(obj, img, K, dist, rv.copy(), tv.copy(), _LM_CRITERIA)
        R, _ = cv2.Rodrigues(rv)
        T = to_T(R, tv)
        res = project(obj, T, K, dist) - img
        sols.append((float(np.sqrt(np.mean(res ** 2))), T, res))
    sols.sort(key=lambda s: s[0])
    rms, T, res = sols[0]
    out = {"T": T, "rms": rms, "res": res, "rms2": float("nan"), "ratio": float("inf"),
           "angle2_deg": 0.0, "T2": np.full((4, 4), np.nan)}
    if len(sols) > 1:
        rms2, T2, _ = sols[1]
        out.update(rms2=rms2, ratio=rms2 / max(rms, 1e-12),
                   angle2_deg=rot_angle_deg(T2[:3, :3] @ T[:3, :3].T), T2=T2)
    return out


def view_geometry(T: np.ndarray, obj_all: np.ndarray) -> tuple[float, float]:
    """Distance to the board centre [mm] and obliquity [deg] (board normal
    against the line of sight to the centre)."""
    c = T[:3, :3] @ obj_all.mean(axis=0) + T[:3, 3]
    n = T[:3, 2]
    obl = np.degrees(np.arccos(np.clip(abs(n @ c) / np.linalg.norm(c), 0.0, 1.0)))
    return float(np.linalg.norm(c) * 1e3), float(obl)


def _perturb(T: np.ndarray, d: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(d[:3].reshape(3, 1))
    return to_T(R @ T[:3, :3], T[:3, 3] + d[3:])


def pose_jacobian(obj: np.ndarray, T: np.ndarray, K: np.ndarray, dist: np.ndarray,
                  eps: float = 1e-7) -> np.ndarray:
    """d(projection)/d(rotation in the camera frame [rad], translation [m])."""
    J = np.zeros((2 * len(obj), 6))
    for k in range(6):
        d = np.zeros(6)
        d[k] = eps
        J[:, k] = (project(obj, _perturb(T, d), K, dist)
                   - project(obj, _perturb(T, -d), K, dist)).ravel() / (2 * eps)
    return J


def cov_to_std(C: np.ndarray) -> dict:
    s = np.sqrt(np.clip(np.diag(C), 0.0, None))
    return {"rot_deg": np.degrees(s[:3]), "trans_mm": s[3:] * 1e3}


def _check_metres(T: np.ndarray, name: str) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    if float(np.linalg.norm(T[:3, 3])) > _MAX_T_NORM_M:
        raise ValueError(f"{name} translation {T[:3, 3]} is not in metres")
    return T


# ============================================================
# Board pose from a window of frames
# ============================================================

@dataclass
class BoardPose:
    """Result of one 1 s capture. T_cw: board -> camera [m] of the mean
    corners (None without a pose). summary: numbers for display and json
    (jsonable()). arrays: everything to recompute it (np.savez)."""
    T_cw: Optional[np.ndarray]
    accepted: bool
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)

    def save_npz(self, path) -> None:
        np.savez_compressed(path, **self.arrays)

    def status_text(self) -> str:
        """One line for a status bar."""
        s = self.summary
        if not self.accepted:
            return "Board pose rejected: " + "; ".join(self.reasons)
        return (f"Board pose: {s['corners_used']}/{s['corners_total']} corners, "
                f"{s['rms_px']:.2f} px, {s['distance_mm']:.0f} mm, {s['obliquity_deg']:.0f} deg")


def board_object_points(board: BoardSpec, pitch_mm: float):
    """(board, dictionary, (N, 3) corner points [m], marker ids)."""
    b, dic = board.make(pitch_mm / 1e3)
    return b, dic, b.getChessboardCorners().astype(np.float64), b.getIds().ravel().astype(int)


def detect_frames(images: list, detector: CornerDetector, n_corners: int,
                  marker_ids_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(F, N, 2) ChArUco corners and (F, M, 4, 2) ArUco corners, NaN = not found."""
    F, M = len(images), len(marker_ids_all)
    corners = np.full((F, n_corners, 2), np.nan)
    mcorners = np.full((F, M, 4, 2), np.nan)
    slot = {int(m): k for k, m in enumerate(marker_ids_all)}
    for f, img in enumerate(images):
        _fill_rows(detector.detect(img), corners[f], mcorners[f], slot)
    return corners, mcorners


def _fill_rows(d: Detection, corner_row: np.ndarray, marker_row: np.ndarray, slot: dict) -> None:
    if d.ids.size:
        corner_row[d.ids] = d.corners
    for mid, mc in zip(d.marker_ids, d.marker_corners):
        k = slot.get(int(mid))
        if k is not None:
            marker_row[k] = mc


def empty_meta(F: int) -> dict:
    nan = np.full(F, np.nan)
    return {"frame_number": nan.copy(), "timestamp_ms": nan.copy(), "exposure": nan.copy(),
            "gain": nan.copy(), "white_balance": nan.copy()}


def _median_shift(a: np.ndarray, b: np.ndarray) -> float:
    common = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
    if common.sum() < MIN_POSE_CORNERS:
        return float("nan")
    return float(np.median(np.linalg.norm(a[common] - b[common], axis=1)))


def analyze_board_frames(corners: np.ndarray, mcorners: np.ndarray, meta: dict,
                         obj_all: np.ndarray, K: np.ndarray, dist: np.ndarray,
                         config: BoardPoseConfig) -> BoardPose:
    """Board pose of the mean corners of one window, its checks and scatter."""
    F, N = corners.shape[:2]
    found = np.isfinite(corners[..., 0])
    n_per = found.sum(axis=1)
    reasons, warnings = [], []
    arrays: dict[str, Any] = {
        "corners": corners, "marker_corners": mcorners, "n_corners": n_per,
        "n_markers": np.isfinite(mcorners[..., 0, 0]).sum(axis=1),
        "object_points": obj_all, "K": K, "dist": dist,
        **{k: np.asarray(v, dtype=np.float64) for k, v in meta.items()},
    }
    summary: dict[str, Any] = {"frames": F, "corners_total": N,
                               "corners_per_frame_min": int(n_per.min()) if F else 0,
                               "corners_per_frame_max": int(n_per.max()) if F else 0}

    def done(acc: bool, T: Optional[np.ndarray] = None) -> BoardPose:
        summary["accepted"] = acc
        summary["reasons"] = reasons
        summary["warnings"] = warnings
        return BoardPose(T, acc, reasons, warnings, summary, arrays)

    # --- board found and C-arm still ------------------------------------
    frame_ok = n_per >= MIN_POSE_CORNERS
    if not frame_ok.any():
        reasons.append("board not found")
        return done(False)
    missing = int((~frame_ok).sum())
    if missing > config.max_missing_frames * F:
        reasons.append(f"board not found in {missing} of {F} frames")
    elif missing:
        warnings.append(f"board not found in {missing} of {F} frames")

    ref = int(np.argmax(frame_ok))
    motion = np.full(F, np.nan)
    for f in range(F):
        if frame_ok[f]:
            motion[f] = _median_shift(corners[f], corners[ref])
    motion_max = float(np.nanmax(motion))
    arrays["motion_px"] = motion
    summary["motion_max_px"] = motion_max
    if motion_max > config.motion_px:
        reasons.append(f"not still: median corner shift up to {motion_max:.2f} px "
                       f"(limit {config.motion_px:.2f})")

    # --- corners used and their mean -----------------------------------
    presence = found.mean(axis=0)
    used = presence >= config.min_presence
    n_used = int(used.sum())
    arrays["presence"] = presence
    arrays["used_ids"] = np.flatnonzero(used)
    summary["corners_used"] = n_used
    summary["corners_missing_ids"] = [int(i) for i in np.flatnonzero(~used)]
    if n_used < config.min_corners:
        reasons.append(f"only {n_used} of {N} corners (minimum {config.min_corners})")
    elif n_used < N:
        warnings.append(f"only {n_used} of {N} corners, missing ids "
                        f"{summary['corners_missing_ids']}")
    if n_used < MIN_POSE_CORNERS:
        return done(False)

    mean_c = np.full((N, 2), np.nan)
    std_c = np.full((N, 2), np.nan)
    with np.errstate(invalid="ignore"):
        mean_c[used] = np.nanmean(corners[:, used], axis=0)
        if F > 1:
            std_c[used] = np.nanstd(corners[:, used], axis=0, ddof=1)
    noise = float(np.sqrt(np.nanmean(std_c[used] ** 2))) if F > 1 else float("nan")
    arrays["mean_corners"] = mean_c
    arrays["corner_std_px"] = std_c
    summary["corner_noise_px"] = noise

    # --- pose of the mean corners -----------------------------------------
    sol = solve_board_pose(obj_all[used], mean_c[used], K, dist)
    T = sol["T"]
    res = np.full((N, 2), np.nan)
    res[used] = sol["res"]
    arrays.update(T_cw=T, T_cw_second=sol["T2"], residuals=res)
    summary["rms_px"] = sol["rms"]
    summary["max_residual_px"] = float(np.max(np.linalg.norm(sol["res"], axis=1)))
    summary["T_cw"] = T
    summary["ippe"] = {"rms_best_px": sol["rms"], "rms_second_px": sol["rms2"],
                       "ratio": sol["ratio"], "angle_between_deg": sol["angle2_deg"]}
    if sol["rms"] > config.max_rms_px:
        reasons.append(f"reprojection {sol['rms']:.2f} px per axis (limit {config.max_rms_px:.2f})")
    if sol["ratio"] < config.min_ippe_ratio and sol["angle2_deg"] > 0.5:
        reasons.append(f"ambiguous planar pose: second solution {sol['rms2']:.2f} px vs "
                       f"{sol['rms']:.2f} px, {sol['angle2_deg']:.1f} deg apart")
    dist_mm, obl = view_geometry(T, obj_all)
    summary["distance_mm"] = dist_mm
    summary["obliquity_deg"] = obl

    # --- one pose per frame ---------------------------------------------
    Tf = np.full((F, 4, 4), np.nan)
    rms_f = np.full(F, np.nan)
    for f in range(F):
        m = used & found[f]
        if m.sum() >= MIN_POSE_CORNERS:
            s = solve_board_pose(obj_all[m], corners[f, m], K, dist)
            Tf[f], rms_f[f] = s["T"], s["rms"]
    arrays["T_cw_frames"] = Tf
    arrays["rms_frames"] = rms_f
    ok = np.isfinite(rms_f)
    summary["frames_with_pose"] = int(ok.sum())
    if ok.sum() >= 2:
        Rm = mean_rotation(Tf[ok, :3, :3])
        tm = Tf[ok, :3, 3].mean(axis=0)
        drot = np.degrees(np.array([rotvec(R @ Rm.T) for R in Tf[ok, :3, :3]]))
        dang = np.linalg.norm(drot, axis=1)
        dt = (Tf[ok, :3, 3] - tm) * 1e3
        c_w = obj_all.mean(axis=0)
        cen = (np.einsum("fij,j->fi", Tf[ok, :3, :3], c_w) + Tf[ok, :3, 3]) * 1e3
        flips = int((dang > config.max_flip_deg).sum())
        summary["single_frame"] = {
            "rms_px_median": float(np.median(rms_f[ok])),
            "rot_std_deg": drot.std(axis=0, ddof=1),
            "rot_rms_deg": float(np.sqrt(np.mean(dang ** 2))),
            "rot_max_deg": float(dang.max()),
            "trans_std_mm": dt.std(axis=0, ddof=1),
            "trans_max_mm": float(np.linalg.norm(dt, axis=1).max()),
            "centre_std_mm": cen.std(axis=0, ddof=1),
            "flips": flips,
        }
        if flips:
            warnings.append(f"{flips} single-frame poses more than {config.max_flip_deg:.0f} deg "
                            "off the mean (planar flip)")
        summary["mean_pose_vs_pose_mean"] = {
            "rot_deg": rot_angle_deg(T[:3, :3] @ Rm.T),
            "trans_mm": float(np.linalg.norm(T[:3, 3] - tm) * 1e3),
        }

    # --- uncertainty from the PnP Jacobian --------------------------------
    J = pose_jacobian(obj_all[used], T, K, dist)
    JtJi = np.linalg.inv(J.T @ J)
    dof = max(1, 2 * n_used - 6)
    sig_res = float(np.sqrt(np.sum(sol["res"] ** 2) / dof))
    arrays["cov_pose_residual"] = sig_res ** 2 * JtJi
    summary["predicted"] = {"from_residual": cov_to_std(sig_res ** 2 * JtJi)}
    if np.isfinite(noise):
        arrays["cov_pose_single_frame"] = noise ** 2 * JtJi
        summary["predicted"]["single_frame_from_corner_noise"] = cov_to_std(noise ** 2 * JtJi)

    # --- camera bookkeeping -------------------------------------------------
    fn = np.asarray(meta.get("frame_number", []), dtype=np.float64)
    if fn.size > 1 and np.all(np.isfinite(fn)):
        dropped = int(np.sum(np.diff(fn) - 1))
        summary["frames_dropped"] = dropped
        if dropped:
            warnings.append(f"{dropped} camera frames dropped inside the window")
    exp = np.asarray(meta.get("exposure", []), dtype=np.float64)
    if exp.size and np.any(np.isfinite(exp)):
        summary["exposure_range"] = [float(np.nanmin(exp)), float(np.nanmax(exp))]
        if np.nanmax(exp) != np.nanmin(exp):
            warnings.append(f"exposure changed inside the window ({np.nanmin(exp):.0f}-"
                            f"{np.nanmax(exp):.0f})")
    return done(not reasons, T)


# ============================================================
# Frame by frame (for a GUI timer)
# ============================================================

class BoardPoseCapture:
    """The 1 s board pose, fed one camera frame at a time.

        cap = BoardPoseCapture(K_rgb, dist_rgb)
        cap.start()
        ...  every new frame:  state = cap.add(frame.image_bgr, frame.frame_index,
                                                frame.timestamp_s)
        state COLLECTING -> keep feeding; DONE -> cap.result is accepted;
        FAILED -> cap.result.reasons says why (nothing kept).

    A frame whose median corner shift against the first frame of the window
    exceeds motion_px starts the window again from that frame (the C-arm still
    swings); after max_wait_frames frames without a full still window the
    capture fails. add() detects on the full frame, about 10-20 ms at 1920 x
    1080, and keeps no image. preview() serves the live view outside a capture.
    """

    COLLECTING, DONE, FAILED, IDLE = "collecting", "done", "failed", "idle"

    def __init__(self, K: np.ndarray, dist: np.ndarray,
                 config: Optional[BoardPoseConfig] = None, board: BoardSpec = DEFAULT_BOARD):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
        self.config = config or BoardPoseConfig()
        self.board = board
        b, dic, self.obj, self.marker_ids = board_object_points(board, self.config.pitch_mm)
        self.detector = CornerDetector(b, dic, self.K, self.dist, self.config.subpix)
        self._slot = {int(m): k for k, m in enumerate(self.marker_ids)}
        self.state = self.IDLE
        self.result: Optional[BoardPose] = None
        self._reset_window()
        self.frames_seen = 0
        self.restarts = 0

    # ---- one frame, no capture -------------------------------------------
    def detect(self, image: np.ndarray) -> Detection:
        return self.detector.detect(image)

    def preview(self, image: np.ndarray) -> dict:
        """Detection plus a single-frame pose for the live view: corners, the
        predicted position of the missing ones, reprojection, distance,
        obliquity. 'T_cw' is None below 6 corners."""
        d = self.detector.detect(image)
        out = {"detection": d, "n_corners": int(d.ids.size), "n_total": len(self.obj),
               "n_markers": int(d.marker_ids.size), "T_cw": None, "rms_px": float("nan"),
               "distance_mm": float("nan"), "obliquity_deg": float("nan"),
               "missing_uv": np.zeros((0, 2))}
        if d.ids.size >= MIN_POSE_CORNERS:
            s = solve_board_pose(self.obj[d.ids], d.corners, self.K, self.dist)
            out["T_cw"], out["rms_px"] = s["T"], s["rms"]
            out["distance_mm"], out["obliquity_deg"] = view_geometry(s["T"], self.obj)
            missing = np.setdiff1d(np.arange(len(self.obj)), d.ids)
            out["missing_uv"] = project(self.obj[missing], s["T"], self.K, self.dist)
        return out

    # ---- capture -------------------------------------------------------------
    def _reset_window(self) -> None:
        self._corners: list = []
        self._markers: list = []
        self._meta: list = []
        self._ref: Optional[np.ndarray] = None

    def start(self) -> None:
        self._reset_window()
        self.frames_seen = 0
        self.restarts = 0
        self.result = None
        self.state = self.COLLECTING

    def cancel(self) -> None:
        self._reset_window()
        self.state = self.IDLE

    @property
    def progress(self) -> tuple[int, int]:
        """(frames in the current window, frames needed)."""
        return len(self._corners), self.config.n_frames

    @property
    def last_corners(self) -> Optional[np.ndarray]:
        """(N, 2) corners of the newest frame in the window, NaN = not found;
        None before the window has started. For the live view during a
        capture: add() has detected them already."""
        return self._corners[-1] if self._corners else None

    def add(self, image: np.ndarray, frame_index: Optional[int] = None,
            timestamp_s: Optional[float] = None) -> str:
        if self.state != self.COLLECTING:
            return self.state
        self.frames_seen += 1
        crow = np.full((len(self.obj), 2), np.nan)
        mrow = np.full((len(self.marker_ids), 4, 2), np.nan)
        _fill_rows(self.detector.detect(image), crow, mrow, self._slot)
        has_board = int(np.isfinite(crow[:, 0]).sum()) >= MIN_POSE_CORNERS
        if has_board and self._ref is not None:
            shift = _median_shift(crow, self._ref)
            if np.isfinite(shift) and shift > self.config.motion_px:
                self._reset_window()          # still swinging: start again here
                self.restarts += 1
        if self._ref is None:
            if not has_board:                 # the window starts with the board in view
                if self.frames_seen >= self.config.max_wait_frames:
                    self._finish(timed_out=True)
                return self.state
            self._ref = crow
        self._corners.append(crow)
        self._markers.append(mrow)
        self._meta.append((np.nan if frame_index is None else float(frame_index),
                           np.nan if timestamp_s is None else float(timestamp_s) * 1e3))
        if len(self._corners) >= self.config.n_frames:
            self._finish()
        elif self.frames_seen >= self.config.max_wait_frames:
            self._finish(timed_out=True)
        return self.state

    def _finish(self, timed_out: bool = False) -> None:
        # reshape: also (0, N, 2) when the board was never in view
        corners = np.array(self._corners, dtype=np.float64).reshape(-1, len(self.obj), 2)
        markers = np.array(self._markers, dtype=np.float64).reshape(-1, len(self.marker_ids), 4, 2)
        meta = empty_meta(len(corners))
        meta["frame_number"] = np.array([m[0] for m in self._meta], dtype=np.float64)
        meta["timestamp_ms"] = np.array([m[1] for m in self._meta], dtype=np.float64)
        pose = analyze_board_frames(corners, markers, meta, self.obj,
                                    self.K, self.dist, self.config)
        pose.summary["frames_seen"] = self.frames_seen
        pose.summary["window_restarts"] = self.restarts
        if timed_out:
            pose.accepted = False
            pose.summary["accepted"] = False
            pose.reasons.insert(0, f"no still window of {self.config.n_frames} frames within "
                                   f"{self.config.max_wait_frames} frames "
                                   f"({self.restarts} restarts)")
            pose.T_cw = None
        if not pose.accepted:
            pose.T_cw = None
        self.result = pose
        self._reset_window()
        self.state = self.DONE if pose.accepted else self.FAILED


# ============================================================
# Views
# ============================================================

def xray_view_matrix(T_cw_view: np.ndarray, T_cw_now: np.ndarray, T_xc: np.ndarray,
                     phi_deg: float = 0.0) -> np.ndarray:
    """T_xc of a view taken in another stance: its X-ray frame -> the camera
    as it is now, metres, for XrayView(..., K_x).

        T_cw_now @ inv(T_cw_view) @ T_xc @ Rz(-phi)

    T_cw_view: board pose in the stance of that view; T_cw_now: board pose in
    the stance tracking runs in; T_xc as calibrated (X-ray -> camera, metres);
    phi: image rotation of that view (PHI_LATERAL_DEG for the lateral stance)."""
    T_cw_view = _check_metres(T_cw_view, "T_cw_view")
    T_cw_now = _check_metres(T_cw_now, "T_cw_now")
    T_xc = _check_metres(T_xc, "T_xc")
    return T_cw_now @ inv_T(T_cw_view) @ T_xc @ rot_z(-phi_deg)


def camera_motion(T_cw_view: np.ndarray, T_cw_now: np.ndarray,
                  target_now_m: Optional[np.ndarray] = None) -> dict:
    """How the camera moved between the stance of a view and now:
    T = T_cw_view @ inv(T_cw_now) maps the current camera into the camera of
    that view (T_clca in the docs). Angle [deg], axis (the same coordinates in
    both cameras), translation [mm], the view's camera centre seen from now
    [mm]; with a target point (current camera, metres) also where it lies in
    the view's camera [mm]."""
    T = _check_metres(T_cw_view, "T_cw_view") @ inv_T(_check_metres(T_cw_now, "T_cw_now"))
    R, t = T[:3, :3], T[:3, 3]
    rv = rotvec(R)
    ang = float(np.linalg.norm(rv))
    out = {"T": T, "angle_deg": float(np.degrees(ang)),
           "axis": rv / ang if ang > 0 else np.array([0.0, 0.0, 1.0]),
           "t_mm": t * 1e3, "view_camera_in_now_mm": -R.T @ t * 1e3}
    if target_now_m is not None:
        p = np.asarray(target_now_m, dtype=np.float64).reshape(3)
        out["target_now_mm"] = p * 1e3
        out["target_view_mm"] = (R @ p + t) * 1e3
    return out


def camera_motion_scatter(T_cw_frames_view: np.ndarray, T_cw_frames_now: np.ndarray,
                          target_now_m: np.ndarray) -> Optional[dict]:
    """Scatter of camera_motion over all pairs of single-frame poses (arrays
    'T_cw_frames' of two BoardPose results). For the 1 s means it is about
    'mean_factor' times smaller if the frames were independent."""
    if T_cw_frames_view is None or T_cw_frames_now is None:
        return None
    Tv = T_cw_frames_view[np.isfinite(T_cw_frames_view[:, 0, 0])]
    Tn = T_cw_frames_now[np.isfinite(T_cw_frames_now[:, 0, 0])]
    if len(Tv) < 2 or len(Tn) < 2:
        return None
    rel = (Tv[:, None] @ inv_T_batch(Tn)[None, :]).reshape(-1, 4, 4)
    rvs = np.array([rotvec(Rij) for Rij in rel[:, :3, :3]])
    angs = np.degrees(np.linalg.norm(rvs, axis=1))
    axs = rvs / np.linalg.norm(rvs, axis=1, keepdims=True)
    axis = axs.mean(axis=0)
    axis /= np.linalg.norm(axis)
    axis_dev = np.degrees(np.arccos(np.clip(axs @ axis, -1.0, 1.0)))
    tij = rel[:, :3, 3] * 1e3
    p = np.asarray(target_now_m, dtype=np.float64).reshape(3)
    pij = (np.einsum("nij,j->ni", rel[:, :3, :3], p) + rel[:, :3, 3]) * 1e3
    pdev = np.linalg.norm(pij - pij.mean(axis=0), axis=1)
    return {"n": int(len(rel)),
            "angle_std_deg": float(angs.std(ddof=1)),
            "axis_rms_deg": float(np.sqrt(np.mean(axis_dev ** 2))),
            "t_std_mm": tij.std(axis=0, ddof=1),
            "target_rms_mm": float(np.sqrt(np.mean(pdev ** 2))),
            "mean_factor": float(np.sqrt((1.0 / len(Tv) + 1.0 / len(Tn)) / 2.0))}


# ============================================================
# Saving
# ============================================================

def jsonable(x):
    """Numbers, arrays and nested containers as plain json types (NaN -> None)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if np.isfinite(v) else None
    if hasattr(x, "__fspath__"):
        return str(x)
    return x
