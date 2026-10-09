"""Three-camera calibration for the D435i: IR-left, IR-right, and colour.

Purpose (tip-tracking project): the fake-tilt error lives in the weak rotation
mode the colour corners cannot see. A second viewpoint (the IR stereo pair)
observes it directly by triangulation, and the correlated corner bias cancels
as common mode. To transfer the colour-detected marker corners into the IR
pair we need ONE consistent calibration of all three cameras -- the factory
extrinsics combined with our own (rational8) colour intrinsics left an ~8.7 px
transfer error, which this tool removes.

Everything is automatic and needs no arguments (like ``calib_camera``):
    python calib/calib_ir_camera.py

Flow:
  1. pick the existing RGB intrinsics .npz (rational8) in a file dialog -- these
     are held FIXED, never re-estimated.
  2. own RealSense pipeline: colour + infrared 1/2, emitter OFF (the speckle
     pattern would corrupt the checkerboard corners).
  3. capture ChArUco views seen simultaneously by all three cameras.
  4. calibrate each IR camera intrinsically with the plain 5-parameter model
     (standard5), then stereoCalibrate IR-L<->IR-R with the intrinsics fixed.
  5. RGB<->IR-L: ONE joint estimate over all views (every intrinsic and the
     IR-L<->IR-R pair fixed; unknowns = the extrinsic + one board pose per
     view; residuals = the reprojection in all three cameras). The per-view
     PnP relative poses only give the start value and a scatter report.
  6. report the acceptance gates and save the bundle, plus the grabbed views
     (corners and images) in ``<bundle>_views.npz`` next to it.

Offline (no camera): recompute a bundle from a saved views file
    python calib/calib_ir_camera.py --views calib/d435i_ir_calibration_<ts>_views.npz
optionally with ``--ir-from <bundle.npz>``: keep that bundle's IR intrinsics
and IR-L<->IR-R pair and re-estimate only RGB<->IR-L.

Board geometry and the ChArUco detector are reused verbatim from calib_camera.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# --- reuse the exact board + detector + npz loader from calib_camera ----------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from calib_camera import (  # noqa: E402
    SQUARES_X,
    SQUARES_Y,
    SQUARE_LEN_M,
    make_charuco_board,
    detect_charuco,
    _charuco_object_points,
    load_tracking_calibration_npz,
)

IR_WIDTH, IR_HEIGHT, FPS = 1280, 720, 15
TARGET_VIEWS = 25
MIN_SHARED_CORNERS = 8            # per camera pair, per view
MIN_CORNERS_INTRINSIC = 10        # per camera, per view, to use for intrinsics
STILL_MAX_SHIFT_PX = 0.35         # frame-to-frame corner shift to count as "still"
COVER_GRID = (4, 3)               # colour coverage grid (columns, rows) in the preview
COVER_MIN_CORNERS = 3             # grabbed corners in a grid cell to count it as covered
SAVE_VIEW_IMAGES = True           # store the grabbed frames in <bundle>_views.npz (colour as grey)


def choose_file_qt(title: str, file_filter: str) -> Path:
    from PySide6.QtWidgets import QApplication, QFileDialog

    app = QApplication.instance() or QApplication(sys.argv)
    path, _ = QFileDialog.getOpenFileName(None, title, "", file_filter)
    if not path:
        raise RuntimeError(f"No file selected: {title}")
    return Path(path)


# ---------------------------------------------------------------------------
# RealSense three-stream source (colour + IR pair, emitter off)
# ---------------------------------------------------------------------------
class D435iThreeStream:
    def __init__(self, color_w: int, color_h: int, fps: int = FPS) -> None:
        """fps: 15 for the still-based calibration (default); recorders that
        must match the live tracking mode pass 30 (the rolling-shutter readout
        is a property of the configured mode, so recordings for shutter
        analysis MUST use the tracking frame rate)."""
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.infrared, 1, IR_WIDTH, IR_HEIGHT, rs.format.y8, fps)
        cfg.enable_stream(rs.stream.infrared, 2, IR_WIDTH, IR_HEIGHT, rs.format.y8, fps)
        self.profile = self.pipeline.start(cfg)

        # emitter OFF: the projected speckle ruins the checkerboard corners.
        dev = self.profile.get_device()
        if fps >= 30:
            # recorders that must match the live tracking mode also need the
            # CONSTANT frame rate: AE priority off caps the exposure at the
            # frame period (gain compensates) instead of dropping fps in dim
            # light. NO other colour overrides (the sharpness=max lesson).
            for sensor in dev.query_sensors():
                name = ""
                try:
                    name = sensor.get_info(rs.camera_info.name).lower()
                except Exception:
                    pass
                if "rgb" in name or "color" in name:
                    try:
                        if sensor.supports(rs.option.auto_exposure_priority):
                            sensor.set_option(rs.option.auto_exposure_priority, 0)
                    except Exception:
                        pass
                    break
        for sensor in dev.query_sensors():
            if sensor.supports(rs.option.emitter_enabled):
                sensor.set_option(rs.option.emitter_enabled, 0)
            if sensor.supports(rs.option.laser_power):
                try:
                    sensor.set_option(rs.option.laser_power, 0)
                except Exception:
                    pass
            # a fixed, generous exposure keeps the board sharp without the
            # projector; auto-exposure on the IR pair tends to hunt.
            if sensor.supports(rs.option.enable_auto_exposure):
                try:
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
                except Exception:
                    pass

        # native (rectified-off) IR intrinsics for a sanity print only
        irl = self.profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        self.baseline_hint_mm = None
        try:
            irr = self.profile.get_stream(rs.stream.infrared, 2)
            extr = irl.get_extrinsics_to(irr)
            self.baseline_hint_mm = abs(extr.translation[0]) * 1000.0
        except Exception:
            pass

    def read(self):
        frames = self.pipeline.wait_for_frames()
        c = frames.get_color_frame()
        l = frames.get_infrared_frame(1)
        r = frames.get_infrared_frame(2)
        if not c or not l or not r:
            return None
        color = np.asanyarray(c.get_data())
        irl = np.asanyarray(l.get_data())
        irr = np.asanyarray(r.get_data())
        return color, irl, irr

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# per-view detection
# ---------------------------------------------------------------------------
class ViewDet:
    """ChArUco detection in one camera for one view: id -> (x, y) subpixel."""

    def __init__(self, det) -> None:
        self.by_id: dict[int, np.ndarray] = {}
        if det.charuco_corners is not None and det.charuco_ids is not None:
            for cid, corner in zip(
                det.charuco_ids.reshape(-1), det.charuco_corners.reshape(-1, 2)
            ):
                self.by_id[int(cid)] = corner.astype(np.float64)

    @classmethod
    def from_points(cls, uv: np.ndarray) -> "ViewDet":
        """From a saved (n_ids, 2) array; NaN = corner not detected."""
        v = cls.__new__(cls)
        v.by_id = {int(i): np.asarray(uv[i], np.float64)
                   for i in range(len(uv)) if np.isfinite(uv[i]).all()}
        return v

    @property
    def n(self) -> int:
        return len(self.by_id)

    def median_xy(self) -> Optional[np.ndarray]:
        if not self.by_id:
            return None
        return np.median(np.stack(list(self.by_id.values())), axis=0)


def _shared(a: ViewDet, b: ViewDet) -> tuple[list[int], np.ndarray, np.ndarray]:
    ids = sorted(set(a.by_id) & set(b.by_id))
    if not ids:
        return [], np.empty((0, 2)), np.empty((0, 2))
    pa = np.stack([a.by_id[i] for i in ids])
    pb = np.stack([b.by_id[i] for i in ids])
    return ids, pa, pb


def _obj_for_ids(board, ids: list[int]) -> np.ndarray:
    return _charuco_object_points(board, np.asarray(ids, dtype=np.int32)).reshape(-1, 3)


class SavedBoard:
    """Stand-in for the ChArUco board of a saved views file: only the corner
    table (what _charuco_object_points reads)."""

    def __init__(self, obj_xyz: np.ndarray) -> None:
        self._obj = np.asarray(obj_xyz, np.float32).reshape(-1, 3)

    def getChessboardCorners(self) -> np.ndarray:
        return self._obj


# ---------------------------------------------------------------------------
# capture loop
# ---------------------------------------------------------------------------
def _still_shift(prev: Optional[ViewDet], cur: ViewDet) -> float:
    """Median frame-to-frame shift of shared IR-L corners (px). Large = motion."""
    if prev is None:
        return 1e9
    ids = set(prev.by_id) & set(cur.by_id)
    if len(ids) < 4:
        return 1e9
    d = [np.linalg.norm(cur.by_id[i] - prev.by_id[i]) for i in ids]
    return float(np.median(d))


def capture_views(cam: D435iThreeStream, board, aruco_dict, det_params):
    """MANUAL capture: press SPACE only while the board is held STILL (motion
    blur ruins the extrinsics). Returns (views, frames): views = list of
    (color, irl, irr) ViewDet, frames = the grabbed (colour grey, IR-L, IR-R)
    images of the same views."""
    views: list[tuple[ViewDet, ViewDet, ViewDet]] = []
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    prev_dl = None
    win = "calib_ir_camera  [SPACE=grab (hold still!)  c=calibrate  q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("MANUAL capture: hold the board STILL, wait for green 'STILL', press SPACE.")
    print("Vary pose/tilt across the whole view between grabs. Never grab while moving.")

    while True:
        got = cam.read()
        if got is None:
            continue
        color, irl, irr = got
        dc = ViewDet(detect_charuco(color, board, aruco_dict, det_params))
        dl = ViewDet(detect_charuco(irl, board, aruco_dict, det_params))
        dr = ViewDet(detect_charuco(irr, board, aruco_dict, det_params))

        ok_counts = (
            dc.n >= MIN_CORNERS_INTRINSIC
            and dl.n >= MIN_CORNERS_INTRINSIC
            and dr.n >= MIN_CORNERS_INTRINSIC
        )
        shared_lr = len(set(dl.by_id) & set(dr.by_id))
        shared_cl = len(set(dc.by_id) & set(dl.by_id))
        ok_shared = shared_lr >= MIN_SHARED_CORNERS and shared_cl >= MIN_SHARED_CORNERS
        shift = _still_shift(prev_dl, dl)
        still = shift <= STILL_MAX_SHIFT_PX
        prev_dl = dl

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if not (ok_counts and ok_shared):
                print("  ! not grabbed: too few corners / shared corners")
            elif not still:
                print(f"  ! not grabbed: board MOVING ({shift:.2f} px > {STILL_MAX_SHIFT_PX}) -- hold still")
            else:
                views.append((dc, dl, dr))
                if SAVE_VIEW_IMAGES:
                    frames.append((cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), irl.copy(), irr.copy()))
                print(f"  view {len(views):2d}  corners C/L/R={dc.n}/{dl.n}/{dr.n}  "
                      f"shared LR={shared_lr} CL={shared_cl}  shift={shift:.2f}px")

        _draw_mosaic(win, color, irl, irr, dc, dl, dr, len(views),
                     ok_counts, ok_shared, still, shift, views)

        if key == ord("q"):
            views = []
            frames = []
            break
        if key == ord("c"):
            break

    cv2.destroyWindow(win)
    return views, frames


def coverage_cells(views_cam, image_wh, grid=COVER_GRID) -> np.ndarray:
    """Grabbed corners per cell of a grid (rows x columns) over one camera image."""
    counts = np.zeros((grid[1], grid[0]), np.int32)
    w, h = image_wh
    for det in views_cam:
        for p in det.by_id.values():
            cx = min(max(int(p[0] * grid[0] / w), 0), grid[0] - 1)
            cy = min(max(int(p[1] * grid[1] / h), 0), grid[1] - 1)
            counts[cy, cx] += 1
    return counts


def _draw_coverage(small, views_cam, image_wh, grid=None):
    """Grabbed corners of all views (magenta) on a downscaled preview image;
    with grid: the coverage grid (covered cells green, the others red)."""
    sx, sy = small.shape[1] / image_wh[0], small.shape[0] / image_wh[1]
    for det in views_cam:
        for p in det.by_id.values():
            cv2.circle(small, (int(p[0] * sx), int(p[1] * sy)), 2, (255, 0, 255), -1)
    if grid is not None:
        counts = coverage_cells(views_cam, image_wh, grid)
        cw, ch = small.shape[1] / grid[0], small.shape[0] / grid[1]
        for r in range(grid[1]):
            for c in range(grid[0]):
                ok = counts[r, c] >= COVER_MIN_CORNERS
                cv2.rectangle(small, (int(c * cw) + 1, int(r * ch) + 1), (int((c + 1) * cw) - 2, int((r + 1) * ch) - 2),
                              (0, 200, 0) if ok else (0, 0, 200), 1)
    return small


def _draw_mosaic(win, color, irl, irr, dc, dl, dr, n, ok_counts, ok_shared,
                 still=True, shift=0.0, views=()):
    def prep(img, det, label, views_cam=(), grid=None):
        vis = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis = vis.copy()
        for cid, p in det.by_id.items():
            cv2.circle(vis, tuple(np.int32(p)), 3, (0, 255, 0), -1)
        cv2.putText(vis, f"{label}: {det.n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        small = cv2.resize(vis, (640, 360))
        return _draw_coverage(small, views_cam, (img.shape[1], img.shape[0]), grid)

    views = list(views or ())
    top = np.hstack([prep(color, dc, "COLOR", [v[0] for v in views], COVER_GRID),
                     prep(irl, dl, "IR-L", [v[1] for v in views])])
    cover = coverage_cells([v[0] for v in views], (color.shape[1], color.shape[0]))
    n_cov, n_cells = int((cover >= COVER_MIN_CORNERS).sum()), cover.size
    panel = np.zeros((360, 640, 3), np.uint8)
    lines = [
        (f"views grabbed: {n}   colour coverage {n_cov}/{n_cells}",
         (0, 255, 0) if n_cov == n_cells else (255, 255, 255)),
        (f"counts:  {'OK' if ok_counts else 'too few'}", (0, 255, 0) if ok_counts else (0, 0, 255)),
        (f"shared:  {'OK' if ok_shared else 'too few'}", (0, 255, 0) if ok_shared else (0, 0, 255)),
        (f"{'STILL' if still else 'MOVING'}  ({shift:.2f} px)",
         (0, 255, 0) if still else (0, 0, 255)),
        ("SPACE = grab   c = done   q = quit", (200, 200, 200)),
    ]
    for i, (txt, col) in enumerate(lines):
        cv2.putText(panel, txt, (20, 60 + i * 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0 if i == 3 else 0.8, col, 2)
    bot = np.hstack([prep(irr, dr, "IR-R", [v[2] for v in views]), panel])
    cv2.imshow(win, np.vstack([top, bot]))


# ---------------------------------------------------------------------------
# saved views (corners + images): reproducible, re-computable offline
# ---------------------------------------------------------------------------
def save_views(path: Path, views, frames, board, rgb_npz, color_size, ir_size) -> None:
    """All grabbed views in one npz: per camera c/l/r an (n_views, n_ids, 2)
    corner array (NaN = not detected), the board's corner table, and the
    grabbed images (colour as grey) when they were kept."""
    obj = np.asarray(board.getChessboardCorners(), np.float64).reshape(-1, 3)
    data: dict[str, Any] = dict(
        obj_xyz=obj,
        squares_xy=np.asarray([SQUARES_X, SQUARES_Y], np.int32),
        square_len_m=np.asarray(SQUARE_LEN_M),
        color_image_size=np.asarray(color_size, np.int32),
        ir_image_size=np.asarray(ir_size, np.int32),
        rgb_source_npz=str(rgb_npz),
    )
    for k, cam in enumerate("clr"):
        uv = np.full((len(views), len(obj), 2), np.nan)
        for i, v in enumerate(views):
            for cid, p in v[k].by_id.items():
                uv[i, cid] = p
        data[f"uv_{cam}"] = uv
    if frames and len(frames) == len(views):
        data["img_color_grey"] = np.stack([f[0] for f in frames])
        data["img_ir_left"] = np.stack([f[1] for f in frames])
        data["img_ir_right"] = np.stack([f[2] for f in frames])
    np.savez_compressed(path, **data)


def load_views(path: Path):
    """-> (views, board stand-in, info) of a save_views file."""
    with np.load(path, allow_pickle=True) as z:
        uv = {c: np.asarray(z[f"uv_{c}"], np.float64) for c in "clr"}
        board = SavedBoard(z["obj_xyz"])
        info = dict(color_size=tuple(int(x) for x in z["color_image_size"]),
                    ir_size=tuple(int(x) for x in z["ir_image_size"]),
                    rgb_npz=str(z["rgb_source_npz"]))
    views = [tuple(ViewDet.from_points(uv[c][i]) for c in "clr") for i in range(len(uv["c"]))]
    return views, board, info


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def calibrate_intrinsic_standard5(views_idx, board, image_size):
    """Plain 5-parameter (k1,k2,p1,p2,k3) intrinsic calibration for one camera."""
    obj_pts, img_pts = [], []
    for det in views_idx:
        if det.n < MIN_CORNERS_INTRINSIC:
            continue
        ids = sorted(det.by_id)
        img_pts.append(np.stack([det.by_id[i] for i in ids]).astype(np.float32))
        obj_pts.append(_obj_for_ids(board, ids).astype(np.float32))
    flags = 0  # standard5: no rational, no thin-prism, no tilted
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, image_size, None, None, flags=flags
    )
    return rms, K, np.asarray(dist).reshape(-1, 1), len(obj_pts)


def solvepnp_rms(views, K, dist, board) -> float:
    """Per-view solvePnP reprojection RMS -- a DIRECT check that these intrinsics
    match this camera stream. >~1 px means the loaded model is wrong for the
    stream (e.g. wrong camera or resolution)."""
    errs = []
    for det in views:
        ids = sorted(det.by_id)
        if len(ids) < 6:
            continue
        obj = _obj_for_ids(board, ids).astype(np.float32)
        img = np.stack([det.by_id[i] for i in ids]).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        errs.extend(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1).tolist())
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def _avg_rot(Rs: list[np.ndarray]) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.sum(np.stack(Rs), axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def relative_pose_pnp(views_a, views_b, Ka, da, Kb, db, board,
                      rot_tol_deg=0.4, trans_tol_mm=1.5):
    """R, T mapping frame a -> frame b from one PnP per view and camera: the
    START VALUE of the joint estimate and a scatter report. Single-view PnP
    of a flat board scatters in depth and tilt (most at the image border and
    for tilted boards), so views outside rot_tol_deg / trans_tol_mm of the
    median are not wrong views -- they only do not enter this average.
    -> (R, T, n within the tolerance, n views)."""
    Rs, Ts = [], []
    for da_v, db_v in zip(views_a, views_b):
        ids = sorted(set(da_v.by_id) & set(db_v.by_id))
        if len(ids) < MIN_SHARED_CORNERS:
            continue
        obj = _obj_for_ids(board, ids).astype(np.float32)
        ia = np.stack([da_v.by_id[i] for i in ids]).astype(np.float32)
        ib = np.stack([db_v.by_id[i] for i in ids]).astype(np.float32)
        oka, rva, tva = cv2.solvePnP(obj, ia, Ka, da)
        okb, rvb, tvb = cv2.solvePnP(obj, ib, Kb, db)
        if not (oka and okb):
            continue
        Ra, _ = cv2.Rodrigues(rva)
        Rb, _ = cv2.Rodrigues(rvb)
        R_ab = Rb @ Ra.T
        Rs.append(R_ab)
        Ts.append(tvb.reshape(3) - R_ab @ tva.reshape(3))
    if not Rs:
        return np.eye(3), np.zeros(3), 0, 0
    n_all = len(Rs)
    T0 = np.median(np.stack(Ts), axis=0)
    R0 = _avg_rot(Rs)
    keep_R, keep_T = [], []
    for R_ab, t_ab in zip(Rs, Ts):
        ang = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R_ab) - 1) / 2, -1, 1)))
        dt = float(np.linalg.norm(t_ab - T0)) * 1000.0
        if ang <= rot_tol_deg and dt <= trans_tol_mm:
            keep_R.append(R_ab)
            keep_T.append(t_ab)
    if len(keep_R) < 3:                       # tolerances too tight -> fall back
        keep_R, keep_T = Rs, Ts
    R = _avg_rot(keep_R)
    T = np.median(np.stack(keep_T), axis=0)
    return R, T, len(keep_R), n_all


def _project(X, K, d) -> np.ndarray:
    p, _ = cv2.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, d)
    return p.reshape(-1, 2)


def joint_rgb_ir(views_c, views_l, views_r, K_c, d_c, K_l, d_l, K_r, d_r,
                 R_lr, T_lr, board, R0, T0, views_used=None):
    """RGB -> IR-L extrinsic (X_l = R X_c + T) from ALL views at once, like
    stereoCalibrate with every intrinsic fixed: unknowns = the extrinsic and
    one board pose per view (in the colour frame); residuals = the
    reprojection of the board corners in colour, IR-L and IR-R (IR-R through
    the fixed IR-L<->IR-R pair), each camera weighted by its own residual RMS
    (two passes), soft-L1 loss against single bad corners. R0/T0 = start.
    views_used: indices of the views to use (default all with >= 6 colour
    corners). -> (R, T, info)."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    idx = [i for i in (range(len(views_c)) if views_used is None else views_used)
           if views_c[i].n >= 6 and views_l[i].n + views_r[i].n >= 6]
    obs = []                                  # per view: [(cam, obj, img)]
    x0 = [cv2.Rodrigues(np.asarray(R0, np.float64))[0].reshape(3), np.asarray(T0, np.float64).reshape(3)]
    for i in idx:
        per = []
        for cam, det in (("c", views_c[i]), ("l", views_l[i]), ("r", views_r[i])):
            ids = sorted(det.by_id)
            if not ids:
                continue
            per.append((cam, _obj_for_ids(board, ids).astype(np.float64),
                        np.stack([det.by_id[k] for k in ids])))
        obs.append(per)
        _, obj, img = per[0]
        ok, rv, tv = cv2.solvePnP(obj, img, K_c, d_c)
        x0 += [rv.reshape(3), tv.reshape(3)]
    x0 = np.concatenate(x0)
    if not obs:
        return np.asarray(R0), np.asarray(T0).reshape(3), dict(n_views=0)
    cams = {"c": (K_c, d_c), "l": (K_l, d_l), "r": (K_r, d_r)}
    weight = {"c": 1.0, "l": 1.0, "r": 1.0}

    def residuals(x):
        Re = cv2.Rodrigues(x[:3])[0]
        te = x[3:6]
        out = []
        for j, per in enumerate(obs):
            Rv = cv2.Rodrigues(x[6 + 6 * j:9 + 6 * j])[0]
            tv = x[9 + 6 * j:12 + 6 * j]
            for cam, obj, img in per:
                X = obj @ Rv.T + tv
                if cam != "c":
                    X = X @ Re.T + te
                if cam == "r":
                    X = X @ R_lr.T + T_lr.reshape(3)
                out.append(((_project(X, *cams[cam]) - img) * weight[cam]).ravel())
        return np.concatenate(out)

    sparsity = lil_matrix((sum(2 * len(o[2]) for per in obs for o in per), len(x0)), dtype=int)
    row = 0
    for j, per in enumerate(obs):
        for cam, obj, _img in per:
            sparsity[row:row + 2 * len(obj), 6 + 6 * j:12 + 6 * j] = 1
            if cam != "c":
                sparsity[row:row + 2 * len(obj), :6] = 1
            row += 2 * len(obj)
    x = x0
    rms = {}
    for _ in range(2):
        sol = least_squares(residuals, x, jac_sparsity=sparsity, loss="soft_l1",
                            f_scale=1.0, x_scale="jac", max_nfev=200)
        x = sol.x
        res = residuals(x)
        row, per_cam = 0, {"c": [], "l": [], "r": []}
        for per in obs:
            for cam, obj, _img in per:
                per_cam[cam].append(res[row:row + 2 * len(obj)] / weight[cam])
                row += 2 * len(obj)
        rms = {c: float(np.sqrt(np.mean(np.concatenate(v) ** 2))) if v else float("nan")
               for c, v in per_cam.items()}
        weight = {c: 1.0 / max(rms[c], 0.02) if np.isfinite(rms[c]) else 1.0 for c in rms}
    return (cv2.Rodrigues(x[:3])[0], x[3:6].copy(),
            dict(n_views=len(obs), rms_px=rms, converged=bool(sol.success)))


def joint_holdout_transfer(views_c, views_l, views_r, K_c, d_c, K_l, d_l, K_r, d_r,
                           R_lr, T_lr, board, R_all, T_all, folds=5):
    """Held-out check of the joint estimate: views split into `folds` groups;
    each group's transfer residual is measured with the extrinsic estimated
    WITHOUT it (the IR intrinsics and the IR pair stay those of all views,
    so this checks the extrinsic only). R_all/T_all must be the joint
    estimate of all views: each fold's estimate starts from it and its
    shift is measured against it. -> (median px over all held-out corners,
    max fold shift of the extrinsic vs the all-view one in deg / mm)."""
    n = len(views_c)
    if n < 2 * folds:
        return float("nan"), float("nan"), float("nan")
    errs, dang, dtr = [], [], []
    for f in range(folds):
        test = list(range(f, n, folds))
        train = [i for i in range(n) if i not in test]
        R, T, _ = joint_rgb_ir(views_c, views_l, views_r, K_c, d_c, K_l, d_l, K_r, d_r,
                               R_lr, T_lr, board, R_all, T_all, views_used=train)
        for i in test:
            e = _transfer_errors(views_c[i], views_l[i], K_c, d_c, K_l, d_l, R, T, board)
            errs.extend(e)
        dang.append(np.degrees(np.linalg.norm(cv2.Rodrigues(R @ np.asarray(R_all).T)[0])))
        dtr.append(float(np.linalg.norm(T - np.asarray(T_all).reshape(3))) * 1000.0)
    return (float(np.median(errs)) if errs else float("nan"), float(max(dang)), float(max(dtr)))


def stereo_fixed(views_a, views_b, Ka, da, Kb, db, board, image_size):
    """stereoCalibrate with both intrinsics fixed -> R, T (b expressed in a)."""
    obj_pts, ipa, ipb = [], [], []
    for da_v, db_v in zip(views_a, views_b):
        ids, pa, pb = _shared(da_v, db_v)
        if len(ids) < MIN_SHARED_CORNERS:
            continue
        obj_pts.append(_obj_for_ids(board, ids).astype(np.float32))
        ipa.append(pa.astype(np.float32))
        ipb.append(pb.astype(np.float32))
    flags = cv2.CALIB_FIX_INTRINSIC
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, *_rest, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, ipa, ipb, Ka, da, Kb, db, image_size, flags=flags, criteria=crit
    )
    return rms, R, np.asarray(T).reshape(3), F, len(obj_pts)


def _fundamental(K_a, K_b, R, T) -> np.ndarray:
    """F of a fixed pair (b expressed in a: X_b = R X_a + T)."""
    t = np.asarray(T, np.float64).reshape(3)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return np.linalg.inv(K_b).T @ tx @ R @ np.linalg.inv(K_a)


def _epipolar_residual(views_a, views_b, F) -> float:
    """Mean symmetric point-to-epipolar-line distance |dv| over shared corners."""
    res = []
    for da_v, db_v in zip(views_a, views_b):
        ids, pa, pb = _shared(da_v, db_v)
        if not ids:
            continue
        pa_h = np.hstack([pa, np.ones((len(pa), 1))])
        pb_h = np.hstack([pb, np.ones((len(pb), 1))])
        la = (F @ pa_h.T).T          # epilines in b for points in a
        lb = (F.T @ pb_h.T).T        # epilines in a for points in b
        da_ = np.abs(np.sum(la * pb_h, 1)) / np.linalg.norm(la[:, :2], axis=1)
        db_ = np.abs(np.sum(lb * pa_h, 1)) / np.linalg.norm(lb[:, :2], axis=1)
        res.extend(da_.tolist()); res.extend(db_.tolist())
    return float(np.mean(res)) if res else float("nan")


def _transfer_errors(dc_v, dl_v, K_rgb, d_rgb, K_l, d_l, R, T, board) -> list:
    """Transfer error (px) of one view: solvePnP in RGB, the same 3D corners
    projected into IR-L via (R, T), compared to the IR-L detections."""
    ids, pc, pl = _shared(dc_v, dl_v)
    if len(ids) < MIN_SHARED_CORNERS:
        return []
    obj = _obj_for_ids(board, ids).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, pc.astype(np.float32), K_rgb, d_rgb)
    if not ok:
        return []
    Rc, _ = cv2.Rodrigues(rvec)
    pts_l = (R @ (Rc @ obj.T + tvec.reshape(3, 1)) + np.asarray(T).reshape(3, 1)).T
    proj, _ = cv2.projectPoints(pts_l.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                K_l, d_l)
    return np.linalg.norm(proj.reshape(-1, 2) - pl, axis=1).tolist()


def _transfer_residual(views_c, views_l, K_rgb, d_rgb, K_l, d_l, R, T, board) -> float:
    """RGB->IR-L transfer error: for each view solvePnP in RGB, project the same
    3D corners into IR-L via (R,T), compare to IR-L detections. THE number that
    was ~8.7 px with the factory extrinsics."""
    errs = []
    for dc_v, dl_v in zip(views_c, views_l):
        errs.extend(_transfer_errors(dc_v, dl_v, K_rgb, d_rgb, K_l, d_l, R, T, board))
    if not errs:
        return float("nan"), float("nan")
    return float(np.mean(errs)), float(np.median(errs))


def _scale_check(views_l, views_r, K_l, d_l, K_r, d_r, R, T, board) -> float:
    """Signed scale error of the IR pair: per view the triangulated corners are
    fitted to the board model with a similarity transform; the median of
    (scale - 1) over the views. (A median of |pairwise distance error| would
    measure the corner noise instead: 0.12 px noise alone reads ~0.4 %.)"""
    P1 = K_l @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K_r @ np.hstack([R, T.reshape(3, 1)])
    scales = []
    for dl_v, dr_v in zip(views_l, views_r):
        ids, pl, pr = _shared(dl_v, dr_v)
        if len(ids) < 6:
            continue
        pl_u = cv2.undistortPoints(pl.reshape(-1, 1, 2), K_l, d_l, P=K_l).reshape(-1, 2)
        pr_u = cv2.undistortPoints(pr.reshape(-1, 1, 2), K_r, d_r, P=K_r).reshape(-1, 2)
        X = cv2.triangulatePoints(P1, P2, pl_u.T, pr_u.T)
        X = (X[:3] / X[3]).T
        obj = _obj_for_ids(board, ids).astype(np.float64)
        A, B = obj - obj.mean(0), X - X.mean(0)
        U, S, Vt = np.linalg.svd(B.T @ A)
        sign = np.sign(np.linalg.det(U @ Vt))
        scales.append(float((S[0] + S[1] + sign * S[2]) / (A ** 2).sum()))
    return float(np.median(scales) - 1.0) if scales else float("nan")


def calibrate_views(views, board, K_rgb, d_rgb, color_size, ir_size, ir_from=None):
    """Everything after the capture: IR intrinsics, IR stereo pair, joint
    RGB<->IR-L, gates. ir_from: an existing bundle (dict of arrays) whose IR
    intrinsics and IR-L<->IR-R pair are kept instead of re-estimated.
    -> (bundle dict for np.savez, all gates passed)."""
    vc = [v[0] for v in views]
    vl = [v[1] for v in views]
    vr = [v[2] for v in views]

    # DIAGNOSTIC: does the loaded RGB model actually match the colour stream?
    rgb_reproj = solvepnp_rms(vc, K_rgb, d_rgb, board)
    print(f"  [diag] RGB solvePnP reproj RMS = {rgb_reproj:.3f} px  "
          f"({'intrinsics MATCH the colour stream' if rgb_reproj < 1.0 else 'WRONG intrinsics for this stream!'})")

    if ir_from is None:
        rms_l, K_l, d_l, n_l = calibrate_intrinsic_standard5(vl, board, ir_size)
        rms_r, K_r, d_r, n_r = calibrate_intrinsic_standard5(vr, board, ir_size)
        print(f"  IR-L intrinsics (standard5): RMS {rms_l:.3f}px  ({n_l} views)")
        print(f"  IR-R intrinsics (standard5): RMS {rms_r:.3f}px  ({n_r} views)")
        rms_lr, R_lr, T_lr, F_lr, n_lr = stereo_fixed(vl, vr, K_l, d_l, K_r, d_r, board, ir_size)
        print(f"  stereo IR-L<->IR-R: RMS {rms_lr:.3f}px  ({n_lr} views)")
    else:
        K_l, d_l = np.asarray(ir_from["K_ir_left"]), np.asarray(ir_from["dist_ir_left"]).reshape(-1, 1)
        K_r, d_r = np.asarray(ir_from["K_ir_right"]), np.asarray(ir_from["dist_ir_right"]).reshape(-1, 1)
        R_lr, T_lr = np.asarray(ir_from["R_left_right"]), np.asarray(ir_from["T_left_right"]).reshape(3)
        F_lr = _fundamental(K_l, K_r, R_lr, T_lr)
        rms_l = solvepnp_rms(vl, K_l, d_l, board)
        rms_r = solvepnp_rms(vr, K_r, d_r, board)
        print(f"  IR intrinsics + IR-L<->IR-R pair KEPT from the given bundle "
              f"(solvePnP RMS here: IR-L {rms_l:.3f}px, IR-R {rms_r:.3f}px)")
    baseline_mm = float(np.linalg.norm(T_lr)) * 1000.0
    print(f"  IR baseline {baseline_mm:.2f} mm")

    # RGB<->IR-L: per-view PnP pairs -> start value + scatter report; then the
    # joint estimate over all views (the result).
    R_pv, T_pv, n_in, n_all = relative_pose_pnp(vc, vl, K_rgb, d_rgb, K_l, d_l, board)
    print(f"  RGB<->IR-L per-view PnP: {n_in}/{n_all} views within 0.4 deg / 1.5 mm of their "
          f"median (single-view PnP scatter, largest for border / tilted views; report only)")
    R_cl, T_cl, jinfo = joint_rgb_ir(vc, vl, vr, K_rgb, d_rgb, K_l, d_l, K_r, d_r,
                                     R_lr, T_lr, board, R_pv, T_pv)
    d_rot = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R_cl @ R_pv.T)[0])))
    d_tr = float(np.linalg.norm(T_cl - T_pv)) * 1000.0
    rp = jinfo.get("rms_px", {})
    print(f"  RGB<->IR-L joint ({jinfo.get('n_views', 0)} views): reproj RMS colour "
          f"{rp.get('c', float('nan')):.3f} / IR-L {rp.get('l', float('nan')):.3f} / IR-R "
          f"{rp.get('r', float('nan')):.3f} px; vs per-view average {d_rot:.3f} deg, {d_tr:.2f} mm")
    ho_px, ho_deg, ho_mm = joint_holdout_transfer(vc, vl, vr, K_rgb, d_rgb, K_l, d_l, K_r, d_r,
                                                  R_lr, T_lr, board, R_cl, T_cl)
    print(f"  held-out (5 folds, extrinsic only): transfer median {ho_px:.3f} px; extrinsic moves "
          f"by at most {ho_deg:.3f} deg / {ho_mm:.2f} mm when a fold is left out")

    # ---- acceptance gates ----
    epi_lr = _epipolar_residual(vl, vr, F_lr)
    transfer_mean, transfer_med = _transfer_residual(
        vc, vl, K_rgb, d_rgb, K_l, d_l, R_cl, T_cl, board)
    scale_rel = _scale_check(vl, vr, K_l, d_l, K_r, d_r, R_lr, T_lr, board)

    print("\n== acceptance gates ==")
    gates = [
        ("RGB intrinsics fit < 1.0 px", rgb_reproj, rgb_reproj < 1.0, f"{rgb_reproj:.3f} px"),
        ("IR-L reproj RMS   < 0.35 px", rms_l, rms_l < 0.35, f"{rms_l:.3f} px"),
        ("IR-R reproj RMS   < 0.35 px", rms_r, rms_r < 0.35, f"{rms_r:.3f} px"),
        ("epipolar |dv| L-R < 0.25 px", epi_lr, epi_lr < 0.25, f"{epi_lr:.3f} px"),
        ("scale error |s-1| < 0.50 %", scale_rel, abs(scale_rel) < 0.005, f"{scale_rel*100:+.2f} %"),
        ("RGB->IR transfer(med) < 0.50 px", transfer_med, transfer_med < 0.50,
         f"{transfer_med:.3f} px (mean {transfer_mean:.3f})"),
        ("held-out transfer(med) < 0.50 px", ho_px, bool(ho_px < 0.50) or not np.isfinite(ho_px),
         f"{ho_px:.3f} px" if np.isfinite(ho_px) else "n/a (< 10 views)"),
        ("baseline ~ 50 mm", baseline_mm, abs(baseline_mm - 50.0) < 3.0, f"{baseline_mm:.2f} mm"),
    ]
    all_pass = True
    for label, _val, ok, shown in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:32s} -> {shown}")
        all_pass = all_pass and ok
    print(f"\n  OVERALL: {'PASS' if all_pass else 'CHECK FAILED GATES'}")

    bundle = dict(
        K_rgb=K_rgb, dist_rgb=d_rgb.reshape(-1),
        K_ir_left=K_l, dist_ir_left=d_l.reshape(-1),
        K_ir_right=K_r, dist_ir_right=d_r.reshape(-1),
        R_left_right=R_lr, T_left_right=T_lr,      # IR-R expressed in IR-L frame
        R_rgb_left=R_cl, T_rgb_left=T_cl,          # IR-L expressed in RGB frame
        R_rgb_left_per_view=R_pv, T_rgb_left_per_view=T_pv,
        rgb_left_method=np.asarray("joint"),
        joint_rms_px=np.asarray([rp.get(c, np.nan) for c in "clr"]),
        joint_n_views=np.asarray(jinfo.get("n_views", 0)),
        holdout_transfer_med_px=np.asarray(ho_px),
        color_image_size=np.asarray(color_size, np.int32),
        ir_image_size=np.asarray(ir_size, np.int32),
        baseline_mm=np.asarray(baseline_mm),
        square_len_m=np.asarray(SQUARE_LEN_M),
        squares_xy=np.asarray([SQUARES_X, SQUARES_Y], np.int32),
        gates_pass=np.asarray(all_pass),
    )
    return bundle, all_pass


def _arg(argv: list, name: str) -> Optional[str]:
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    print("== calib_ir_camera :: D435i three-camera (IR-L, IR-R, RGB) ==")
    views_path = _arg(argv, "--views")
    ir_from_path = _arg(argv, "--ir-from")
    frames = []
    if views_path:
        # offline: recompute from saved views (no camera)
        views, board, vinfo = load_views(Path(views_path))
        rgb_npz = Path(_arg(argv, "--rgb") or vinfo["rgb_npz"])
        if not rgb_npz.exists():
            rgb_npz = choose_file_qt("Select RGB intrinsics .npz (rational8)", "NPZ (*.npz)")
        K_rgb, d_rgb, info = load_tracking_calibration_npz(rgb_npz)
        color_w, color_h = vinfo["color_size"]
        ir_size = vinfo["ir_size"]
        print(f"  {len(views)} saved views from {views_path}; RGB intrinsics {rgb_npz.name}, held FIXED")
        out_dir = Path(_arg(argv, "--out-dir") or Path(views_path).parent)
    else:
        rgb_npz = choose_file_qt("Select RGB intrinsics .npz (rational8)", "NPZ (*.npz)")
        K_rgb, d_rgb, info = load_tracking_calibration_npz(rgb_npz)
        color_size = info.get("calibration_image_size")
        if color_size is None:
            color_w, color_h = 1280, 720
            print("  ! NPZ has no image size; assuming colour 1280x720. Verify!")
        else:
            color_w, color_h = int(color_size[0]), int(color_size[1])
        print(f"  RGB intrinsics loaded ({d_rgb.size}-coeff model) @ {color_w}x{color_h}, held FIXED")

        board, aruco_dict, det_params = make_charuco_board()
        cam = D435iThreeStream(color_w, color_h)
        if cam.baseline_hint_mm:
            print(f"  factory IR baseline hint: {cam.baseline_hint_mm:.1f} mm")

        try:
            views, frames = capture_views(cam, board, aruco_dict, det_params)
        finally:
            cam.stop()
        ir_size = (IR_WIDTH, IR_HEIGHT)
        out_dir = _HERE

    if len(views) < 8:
        print(f"FAIL: only {len(views)} views captured (need >= ~8).")
        return 1
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"d435i_ir_calibration_{ts}.npz"
    if not views_path:
        # the grabbed views first: kept even if the calibration below fails
        views_out = out.with_name(out.stem + "_views.npz")
        save_views(views_out, views, frames, board, rgb_npz, (color_w, color_h), ir_size)
        print(f"  views saved -> {views_out}")
    print(f"\ncalibrating from {len(views)} views ...")

    ir_from = None
    if ir_from_path:
        with np.load(ir_from_path, allow_pickle=True) as z:
            ir_from = {k: z[k] for k in z.files}
    bundle, all_pass = calibrate_views(views, board, K_rgb, d_rgb, (color_w, color_h), ir_size, ir_from)

    # ---- save ----
    np.savez(
        out,
        **bundle,
        rgb_source_npz=str(rgb_npz),
        views_npz=str(views_path or out.with_name(out.stem + "_views.npz")),
        ir_from_npz=str(ir_from_path or ""),
        created_at=ts,
    )
    print(f"\nsaved -> {out}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
