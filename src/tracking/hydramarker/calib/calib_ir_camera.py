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
     (standard5), then stereoCalibrate IR-L<->IR-R and RGB<->IR-L with the
     intrinsics fixed.
  5. report the acceptance gates and save the bundle.

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
    blur ruins the extrinsics). Returns list of (color, irl, irr) ViewDet."""
    views: list[tuple[ViewDet, ViewDet, ViewDet]] = []
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
                print(f"  view {len(views):2d}  corners C/L/R={dc.n}/{dl.n}/{dr.n}  "
                      f"shared LR={shared_lr} CL={shared_cl}  shift={shift:.2f}px")

        _draw_mosaic(win, color, irl, irr, dc, dl, dr, len(views),
                     ok_counts, ok_shared, still, shift)

        if key == ord("q"):
            views = []
            break
        if key == ord("c"):
            break

    cv2.destroyWindow(win)
    return views


def _draw_mosaic(win, color, irl, irr, dc, dl, dr, n, ok_counts, ok_shared,
                 still=True, shift=0.0):
    def prep(img, det, label):
        vis = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vis = vis.copy()
        for cid, p in det.by_id.items():
            cv2.circle(vis, tuple(np.int32(p)), 3, (0, 255, 0), -1)
        cv2.putText(vis, f"{label}: {det.n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return cv2.resize(vis, (640, 360))

    top = np.hstack([prep(color, dc, "COLOR"), prep(irl, dl, "IR-L")])
    panel = np.zeros((360, 640, 3), np.uint8)
    lines = [
        (f"views grabbed: {n}", (255, 255, 255)),
        (f"counts:  {'OK' if ok_counts else 'too few'}", (0, 255, 0) if ok_counts else (0, 0, 255)),
        (f"shared:  {'OK' if ok_shared else 'too few'}", (0, 255, 0) if ok_shared else (0, 0, 255)),
        (f"{'STILL' if still else 'MOVING'}  ({shift:.2f} px)",
         (0, 255, 0) if still else (0, 0, 255)),
        ("SPACE = grab   c = done   q = quit", (200, 200, 200)),
    ]
    for i, (txt, col) in enumerate(lines):
        cv2.putText(panel, txt, (20, 60 + i * 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0 if i == 3 else 0.8, col, 2)
    bot = np.hstack([prep(irr, dr, "IR-R"), panel])
    cv2.imshow(win, np.vstack([top, bot]))


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
    """R, T mapping frame a -> frame b. The rig is RIGID, so the per-view
    relative pose must be identical across views; views that deviate (motion
    blur, non-simultaneous frames) are rejected, then the consistent ones are
    averaged. Robust to different resolutions / distortion models."""
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
    # reject views deviating from the consensus (rigid rig => should be tiny)
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


def _transfer_residual(views_c, views_l, K_rgb, d_rgb, K_l, d_l, R, T, board) -> float:
    """RGB->IR-L transfer error: for each view solvePnP in RGB, project the same
    3D corners into IR-L via (R,T), compare to IR-L detections. THE number that
    was ~8.7 px with the factory extrinsics."""
    errs = []
    for dc_v, dl_v in zip(views_c, views_l):
        ids, pc, pl = _shared(dc_v, dl_v)
        if len(ids) < MIN_SHARED_CORNERS:
            continue
        obj = _obj_for_ids(board, ids).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, pc.astype(np.float32), K_rgb, d_rgb)
        if not ok:
            continue
        Rc, _ = cv2.Rodrigues(rvec)
        pts_l = (R @ (Rc @ obj.T + tvec.reshape(3, 1)) + T.reshape(3, 1)).T
        proj, _ = cv2.projectPoints(pts_l.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                    K_l, d_l)
        proj = proj.reshape(-1, 2)
        errs.extend(np.linalg.norm(proj - pl, axis=1).tolist())
    if not errs:
        return float("nan"), float("nan")
    return float(np.mean(errs)), float(np.median(errs))


def _scale_check(views_l, views_r, K_l, d_l, K_r, d_r, R, T, board) -> float:
    """Triangulate shared corners and compare mean neighbour spacing to the known
    square size (relative error)."""
    P1 = K_l @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K_r @ np.hstack([R, T.reshape(3, 1)])
    rel = []
    for dl_v, dr_v in zip(views_l, views_r):
        ids, pl, pr = _shared(dl_v, dr_v)
        if len(ids) < 4:
            continue
        pl_u = cv2.undistortPoints(pl.reshape(-1, 1, 2), K_l, d_l, P=K_l).reshape(-1, 2)
        pr_u = cv2.undistortPoints(pr.reshape(-1, 1, 2), K_r, d_r, P=K_r).reshape(-1, 2)
        X = cv2.triangulatePoints(P1, P2, pl_u.T, pr_u.T)
        X = (X[:3] / X[3]).T
        obj = _obj_for_ids(board, ids)
        # compare pairwise distances of triangulated vs model corners
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dm = np.linalg.norm(obj[i] - obj[j])
                if dm < 1e-6:
                    continue
                dt = np.linalg.norm(X[i] - X[j])
                rel.append(abs(dt - dm) / dm)
    return float(np.median(rel)) if rel else float("nan")


def main() -> int:
    print("== calib_ir_camera :: D435i three-camera (IR-L, IR-R, RGB) ==")
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
        views = capture_views(cam, board, aruco_dict, det_params)
    finally:
        cam.stop()

    if len(views) < 8:
        print(f"FAIL: only {len(views)} views captured (need >= ~8).")
        return 1
    print(f"\ncalibrating from {len(views)} views ...")

    vc = [v[0] for v in views]
    vl = [v[1] for v in views]
    vr = [v[2] for v in views]
    ir_size = (IR_WIDTH, IR_HEIGHT)
    color_size_t = (color_w, color_h)

    # DIAGNOSTIC: does the loaded RGB model actually match the colour stream?
    rgb_reproj = solvepnp_rms(vc, K_rgb, d_rgb, board)
    print(f"  [diag] RGB solvePnP reproj RMS = {rgb_reproj:.3f} px  "
          f"({'intrinsics MATCH the colour stream' if rgb_reproj < 1.0 else 'WRONG intrinsics for this stream!'})")

    rms_l, K_l, d_l, n_l = calibrate_intrinsic_standard5(vl, board, ir_size)
    rms_r, K_r, d_r, n_r = calibrate_intrinsic_standard5(vr, board, ir_size)
    print(f"  IR-L intrinsics (standard5): RMS {rms_l:.3f}px  ({n_l} views)")
    print(f"  IR-R intrinsics (standard5): RMS {rms_r:.3f}px  ({n_r} views)")

    rms_lr, R_lr, T_lr, F_lr, n_lr = stereo_fixed(vl, vr, K_l, d_l, K_r, d_r, board, ir_size)
    baseline_mm = float(np.linalg.norm(T_lr)) * 1000.0
    print(f"  stereo IR-L<->IR-R: RMS {rms_lr:.3f}px  baseline {baseline_mm:.2f} mm  ({n_lr} views)")

    # RGB<->IR-L via robust per-view relative pose (handles the different
    # resolution / 14-coeff-vs-5-coeff models that break stereoCalibrate, and
    # rejects motion-blurred / non-simultaneous views).
    R_cl, T_cl, n_kept, n_all = relative_pose_pnp(vc, vl, K_rgb, d_rgb, K_l, d_l, board)
    print(f"  RGB<->IR-L (relative-pose): {n_kept}/{n_all} views consistent (rest rejected as blurred/moving)")

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
        ("scale error       < 0.50 %", scale_rel, scale_rel < 0.005, f"{scale_rel*100:.2f} %"),
        ("RGB->IR transfer(med) < 0.50 px", transfer_med, transfer_med < 0.50,
         f"{transfer_med:.3f} px (mean {transfer_mean:.3f})"),
        ("baseline ~ 50 mm", baseline_mm, abs(baseline_mm - 50.0) < 3.0, f"{baseline_mm:.2f} mm"),
    ]
    all_pass = True
    for label, _val, ok, shown in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:30s} -> {shown}")
        all_pass = all_pass and ok
    print(f"\n  OVERALL: {'PASS' if all_pass else 'CHECK FAILED GATES'}")

    # ---- save ----
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = _HERE / f"d435i_ir_calibration_{ts}.npz"
    np.savez(
        out,
        K_rgb=K_rgb, dist_rgb=d_rgb.reshape(-1),
        K_ir_left=K_l, dist_ir_left=d_l.reshape(-1),
        K_ir_right=K_r, dist_ir_right=d_r.reshape(-1),
        R_left_right=R_lr, T_left_right=T_lr,      # IR-R expressed in IR-L frame
        R_rgb_left=R_cl, T_rgb_left=T_cl,          # IR-L expressed in RGB frame
        color_image_size=np.asarray(color_size_t, np.int32),
        ir_image_size=np.asarray(ir_size, np.int32),
        baseline_mm=np.asarray(baseline_mm),
        square_len_m=np.asarray(SQUARE_LEN_M),
        squares_xy=np.asarray([SQUARES_X, SQUARES_Y], np.int32),
        rgb_source_npz=str(rgb_npz),
        gates_pass=np.asarray(all_pass),
        created_at=ts,
    )
    print(f"\nsaved -> {out}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
