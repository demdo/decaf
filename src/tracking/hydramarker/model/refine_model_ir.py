"""IR model refinement post-pass (optional, RealSense-IR only).

Refines the SfM marker model with the reference-free IR stereo measurement:
per observation still, the pose comes from solvePnP on the stored
correspondences, the marker corners are measured in BOTH IR views
(quadratic-form + synthetic saddle, via the tracker engine), triangulated
over the metric baseline and compared against the model. The per-corner 3D
median residual (marker frame) corrects the corner coordinates — exactly
the radial/depth information the monocular RGB SfM cannot observe.

The RGB SfM pipeline itself is untouched; without IR frames next to the
observations this stage simply refuses to run, so the model pipeline stays
fully usable with any camera.

usage:
  python refine_model_ir.py FIELD MODEL.json OBS.npz FRAMES_DIR \
         IR_CALIB.npz CAM_CALIB.npz [--min-obs 40] [--depth-scale 1.00197]

output: MODEL_ir3d.json (versioned copy; the input model is never touched)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tracking.hydramarker.run_tracker import make_tracker  # noqa: E402
from tracking.hydramarker.config import TrackerConfig  # noqa: E402
from tracking.hydramarker.model.observations import (  # noqa: E402
    load_observations_npz,
)


def load_cam_calib(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    K = None
    dist = None
    for k in ("K", "camera_matrix", "matrix", "K_rgb"):
        if k in z.files:
            K = np.asarray(z[k], float).reshape(3, 3)
            break
    for k in ("dist", "dist_coeffs", "distortion", "dist_rgb"):
        if k in z.files:
            dist = np.asarray(z[k], float).reshape(-1, 1)
            break
    if K is None:
        raise SystemExit(f"no camera matrix in {path}")
    if dist is None:
        dist = np.zeros((5, 1))
    return K, dist


def fit_cylinder(X: np.ndarray, p0, d0, r0, iters=30):
    """Least-squares cylinder refit (axis point/dir/radius), GN with the
    previous fit as initialization. Returns (p, d, r, rms)."""
    p = np.asarray(p0, float).copy()
    d = np.asarray(d0, float)
    d = d / np.linalg.norm(d)
    r = float(r0)

    def residuals(p, d, r):
        w = X - p
        along = w @ d
        rad = w - np.outer(along, d)
        return np.linalg.norm(rad, axis=1) - r

    for _ in range(iters):
        # numeric Jacobian over 5 free params: p (2 dof orthogonal to d),
        # d (2 dof), r
        b1 = np.array([d[1] - d[2], d[2] - d[0], d[0] - d[1]])
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(d, b1)
        base = residuals(p, d, r)
        J = np.zeros((len(base), 5))
        eps = 1e-5
        for j, dp in enumerate((b1, b2)):
            J[:, j] = (residuals(p + eps * dp, d, r) - base) / eps
        for j, dd in enumerate((b1, b2)):
            d2 = d + eps * dd
            d2 /= np.linalg.norm(d2)
            J[:, 2 + j] = (residuals(p, d2, r) - base) / eps
        J[:, 4] = (residuals(p, d, r + eps) - base) / eps
        try:
            step = np.linalg.solve(J.T @ J + 1e-9 * np.eye(5), -(J.T @ base))
        except np.linalg.LinAlgError:
            break
        p = p + step[0] * b1 + step[1] * b2
        d = d + step[2] * b1 + step[3] * b2
        d /= np.linalg.norm(d)
        r += step[4]
        if np.linalg.norm(step) < 1e-9:
            break
    rms = float(np.sqrt(np.mean(residuals(p, d, r) ** 2)))
    return p, d, r, rms


def _choose_file_qt(title: str, file_filter: str) -> str:
    from PySide6.QtWidgets import QApplication, QFileDialog  # type: ignore

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    path, _ = QFileDialog.getOpenFileName(None, title, "", file_filter)
    if not path:
        raise SystemExit(f"Keine Datei gewaehlt: {title}")
    return path


def _choose_dir_qt(title: str) -> str:
    from PySide6.QtWidgets import QApplication, QFileDialog  # type: ignore

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    path = QFileDialog.getExistingDirectory(None, title, "")
    if not path:
        raise SystemExit(f"Kein Ordner gewaehlt: {title}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("field", nargs="?")
    ap.add_argument("model_json", nargs="?")
    ap.add_argument("observations_npz", nargs="?")
    ap.add_argument("frames_dir", nargs="?")
    ap.add_argument("ir_calib_npz", nargs="?")
    ap.add_argument("cam_calib_npz", nargs="?")
    ap.add_argument("--min-obs", type=int, default=40)
    ap.add_argument("--depth-scale", type=float, default=1.00197)
    ap.add_argument("--max-reproj-px", type=float, default=1.0)
    ap.add_argument("--epipolar-max-dv-px", type=float, default=2.5)
    args = ap.parse_args()

    # Ohne CLI-Argumente: dieselben Qt-Dialoge wie Recorder/Pipeline.
    if args.field is None:
        args.field = _choose_file_qt(
            "HydraMarker .field waehlen", "HydraMarker field (*.field)")
    if args.model_json is None:
        args.model_json = _choose_file_qt(
            "SfM-Modell .json waehlen (Basis, wird NICHT veraendert)",
            "Marker JSON (*.json)")
    if args.observations_npz is None:
        args.observations_npz = _choose_file_qt(
            "Observations .npz waehlen", "NPZ (*.npz)")
    if args.frames_dir is None:
        args.frames_dir = _choose_dir_qt(
            "Frames-Ordner waehlen (mit frame_*_irL/irR.png)")
    if args.ir_calib_npz is None:
        args.ir_calib_npz = _choose_file_qt(
            "IR-Rig-Kalibrierung .npz waehlen (realsense_ir_calibration)",
            "NPZ (*.npz)")
    if args.cam_calib_npz is None:
        args.cam_calib_npz = _choose_file_qt(
            "RGB-Kamera-Kalibrierung .npz waehlen (z.B. rational8)",
            "NPZ (*.npz)")

    frames_dir = Path(args.frames_dir)
    if not list(frames_dir.glob("frame_*_irL.png")):
        raise SystemExit(
            "no frame_*_irL.png in the frames dir — this optional stage "
            "needs the IR pair dumped by the recorder (RealSense with IR); "
            "the RGB model pipeline is complete without it.")

    g = json.loads(Path(args.model_json).read_text(encoding="utf-8"))
    num_cols = int(g["id_encoding"]["num_cols"])
    corners = {int(c["id"]): np.asarray(c["xyz_mm"], float)
               for c in g["corners"]}
    sm = g["surface_model"]["fitted"]

    ircal = np.load(args.ir_calib_npz, allow_pickle=True)
    R_rl = np.asarray(ircal["R_rgb_left"], float).reshape(3, 3)
    t_rl = np.asarray(ircal["T_rgb_left"], float).ravel() * 1000.0
    R_lr = np.asarray(ircal["R_left_right"], float).reshape(3, 3)
    t_lr = np.asarray(ircal["T_left_right"], float).ravel() * 1000.0
    K_L = np.asarray(ircal["K_ir_left"], float).reshape(3, 3)
    K_R = np.asarray(ircal["K_ir_right"], float).reshape(3, 3)
    d_L = np.asarray(ircal["dist_ir_left"], float).ravel()
    d_R = np.asarray(ircal["dist_ir_right"], float).ravel()

    K_cam, dist_cam = load_cam_calib(Path(args.cam_calib_npz))

    # Engine instance only for its configured reference-free IR measurement
    # (surface model + full-pattern template + IR intrinsics).
    cfg = TrackerConfig()
    cfg.ir_refine_enabled = True
    cfg.ir_corner_method = "quadratic_form"
    tracker = make_tracker(args.field, args.model_json, K_cam, dist_cam,
                           config=cfg)
    tracker.set_ir_calibration(args.ir_calib_npz)
    eng = tracker._engine

    frames = load_observations_npz(Path(args.observations_npz))
    print(f"[refine-ir] frames {len(frames)}")

    frame_clouds: list[tuple[np.ndarray, np.ndarray]] = []
    used_frames = 0
    for fr in frames:
        img_l = frames_dir / f"frame_{fr.frame_id:06d}_irL.png"
        img_r = frames_dir / f"frame_{fr.frame_id:06d}_irR.png"
        if not (img_l.exists() and img_r.exists()):
            continue
        ids = sorted(i for i in fr.observations if i in corners)
        if len(ids) < 12:
            continue
        obj = np.array([corners[i] for i in ids])
        uv = np.array([fr.observations[i].uv for i in ids])
        ok, rvec, tvec = cv2.solvePnP(
            obj.reshape(-1, 1, 3), uv.reshape(-1, 1, 2), K_cam, dist_cam,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec,
                                    K_cam, dist_cam)
        reproj = float(np.mean(np.linalg.norm(
            proj.reshape(-1, 2) - uv, axis=1)))
        if reproj > args.max_reproj_px:
            continue

        R_f = cv2.Rodrigues(rvec)[0]
        t_f = tvec.reshape(3)
        R_A = R_rl @ R_f
        t_A = R_rl @ t_f + t_rl
        R_B = R_lr @ R_A
        t_B = R_lr @ t_A + t_lr

        irl = cv2.imread(str(img_l), 0)
        irr = cv2.imread(str(img_r), 0)
        if irl is None or irr is None:
            continue

        seedL, _ = cv2.projectPoints(obj.reshape(-1, 1, 3),
                                     cv2.Rodrigues(R_A)[0],
                                     t_A.reshape(3, 1), K_L,
                                     d_L.reshape(-1, 1))
        seedR, _ = cv2.projectPoints(obj.reshape(-1, 1, 3),
                                     cv2.Rodrigues(R_B)[0],
                                     t_B.reshape(3, 1), K_R,
                                     d_R.reshape(-1, 1))
        seedL = seedL.reshape(-1, 2)
        seedR = seedR.reshape(-1, 2)

        rvec_a = cv2.Rodrigues(R_A)[0].reshape(3)
        rvec_b = cv2.Rodrigues(R_B)[0].reshape(3)
        mL = eng.measure_ir_view_qf(
            irl, False, [float(x) for x in rvec_a],
            [float(x) for x in t_A], [float(x) for x in obj.ravel()],
            [float(x) for x in seedL.ravel()])
        mR = eng.measure_ir_view_qf(
            irr, True, [float(x) for x in rvec_b],
            [float(x) for x in t_B], [float(x) for x in obj.ravel()],
            [float(x) for x in seedR.ravel()])

        used_frames += 1
        # Triangulate the surviving pairs (pose-INDEPENDENT stereo cloud in
        # the IR-left frame). The pose per frame is then a rigid Kabsch
        # fit of the model onto this cloud - the RGB solvePnP pose only
        # seeded the measurement. Using the RGB pose directly here leaks
        # its monocular DEPTH error into every residual, and over the
        # orientation sweep that medians into a fake radial inflation
        # (first version: radius 14.35 -> 15.25, |dX| ~1 mm).
        cloud_ids = []
        cloud_pts = []
        for j, cid in enumerate(ids):
            uL, vL, okL, qL = mL[j]
            uR, vR, okR, qR = mR[j]
            if not (okL and okR):
                continue
            if abs(vL - vR) > args.epipolar_max_dv_px:
                continue
            nl = cv2.undistortPoints(
                np.array([[[uL, vL]]], np.float64), K_L,
                d_L.reshape(-1, 1)).reshape(2)
            nr = cv2.undistortPoints(
                np.array([[[uR, vR]]], np.float64), K_R,
                d_R.reshape(-1, 1)).reshape(2)
            vLr = np.array([nl[0], nl[1], 1.0])
            vLr /= np.linalg.norm(vLr)
            vRr = R_lr.T @ np.array([nr[0], nr[1], 1.0])
            vRr /= np.linalg.norm(vRr)
            oR = -R_lr.T @ t_lr
            aa = vLr @ vLr
            bb = vLr @ vRr
            cc = vRr @ vRr
            dd = vLr @ (-oR)
            ee = vRr @ (-oR)
            den = aa * cc - bb * bb
            if abs(den) < 1e-12:
                continue
            s = (bb * ee - cc * dd) / den
            t2 = (aa * ee - bb * dd) / den
            P = 0.5 * ((s * vLr) + (oR + t2 * vRr)) * args.depth_scale
            cloud_ids.append(cid)
            cloud_pts.append(P)
        if len(cloud_ids) >= 10:
            frame_clouds.append((np.array(cloud_ids), np.array(cloud_pts)))

    print(f"[refine-ir] used frames {used_frames}, frames with cloud "
          f"{len(frame_clouds)}")

    def kabsch(A, B):
        """Rigid B ~ R @ A + t (least squares)."""
        ca = A.mean(0)
        cb = B.mean(0)
        H = (A - ca).T @ (B - cb)
        U, S, Vt = np.linalg.svd(H)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        return R, cb - R @ ca

    # Alternate: per-frame rigid pose (trimmed Kabsch, model -> cloud)
    # <-> per-corner 3D median residual; rigid gauge re-alignment after
    # each corner update keeps the marker frame convention.
    corrected = dict(corners)
    report = []
    for it in range(3):
        pool = {}
        pool_frame = {}
        for f_idx, (cids, pts) in enumerate(frame_clouds):
            M = np.array([corrected[c] for c in cids])
            keep = np.ones(len(cids), bool)
            R_p = None
            t_p = None
            for _ in range(3):  # trimmed rigid fit
                if keep.sum() < 8:
                    break
                R_p, t_p = kabsch(M[keep], pts[keep])
                r = np.linalg.norm(pts - (M @ R_p.T + t_p), axis=1)
                thr = max(1.0, 3.0 * 1.4826 * np.median(np.abs(
                    r[keep] - np.median(r[keep]))) + np.median(r[keep]))
                keep = r < thr
            if R_p is None or keep.sum() < 8:
                continue
            resid = (pts - (M @ R_p.T + t_p)) @ R_p  # -> marker frame
            for c, dr, k in zip(cids, resid, keep):
                if not k:
                    continue
                pool.setdefault(int(c), []).append(dr)
                pool_frame.setdefault(int(c), []).append(f_idx)
        # per-corner update
        report = []
        upd = {}
        for cid, res in pool.items():
            if len(res) < args.min_obs:
                continue
            Rm = np.array(res)
            med = np.median(Rm, axis=0)
            fids = np.array(pool_frame[cid])
            me = np.median(Rm[fids % 2 == 0], axis=0) if (
                fids % 2 == 0).sum() >= 10 else med
            mo = np.median(Rm[fids % 2 == 1], axis=0) if (
                fids % 2 == 1).sum() >= 10 else med
            upd[cid] = med
            report.append((cid, len(res), float(np.linalg.norm(med)),
                           float(np.linalg.norm(me - mo))))
        if not upd:
            break
        for cid, med in upd.items():
            corrected[cid] = corrected[cid] + med
        # gauge: rigid re-alignment of the corrected set onto the ORIGINAL
        # corners (no scale! genuine shape/radius changes must survive).
        ids_all = sorted(corrected)
        A = np.array([corrected[c] for c in ids_all])
        B = np.array([corners[c] for c in ids_all])
        R_g, t_g = kabsch(A, B)
        for c in ids_all:
            corrected[c] = R_g @ corrected[c] + t_g
        dm = np.median([r[2] for r in report])
        print(f"[refine-ir] iter {it + 1}: korrigierte Ecken {len(report)}, "
              f"|d| med {dm:.3f} mm")
        if dm < 0.02:
            break
    # report vs ORIGINAL corners
    report = [(c, n, float(np.linalg.norm(corrected[c] - corners[c])), s)
              for (c, n, _, s) in report]

    if not report:
        raise SystemExit("[refine-ir] no corner reached min-obs — nothing "
                         "to write")
    dmag = np.array([r[2] for r in report])
    splits = np.array([r[3] for r in report])
    print(f"[refine-ir] corrected {len(report)} corners | |dX| med "
          f"{np.median(dmag):.3f} max {dmag.max():.3f} mm | odd/even split "
          f"med {np.median(splits):.3f} max {splits.max():.3f} mm")
    if np.median(splits) > 0.15:
        print("[refine-ir] WARNUNG: odd/even split > 0.15 mm — Korrektur "
              "nicht stabil, Ergebnis pruefen!")

    # surface refit on the corrected corners
    X = np.array([corrected[i] for i in sorted(corrected)])
    p, d, r, rms = fit_cylinder(
        X, np.asarray(sm["axis_point_mm"], float),
        np.asarray(sm["axis_dir"], float), float(sm["radius_mm"]))
    print(f"[refine-ir] cylinder refit: radius {sm['radius_mm']:.3f} -> "
          f"{r:.3f} mm | fit_rms {sm.get('fit_rms_mm', float('nan')):.3f} "
          f"-> {rms:.3f} mm")

    out = json.loads(Path(args.model_json).read_text(encoding="utf-8"))
    for c in out["corners"]:
        c["xyz_mm"] = [float(x) for x in corrected[int(c["id"])]]
    out["surface_model"]["fitted"]["axis_point_mm"] = [float(x) for x in p]
    out["surface_model"]["fitted"]["axis_dir"] = [float(x) for x in d]
    out["surface_model"]["fitted"]["radius_mm"] = float(r)
    out["surface_model"]["fitted"]["fit_rms_mm"] = float(rms)
    out["ir_refinement"] = {
        "source_observations": str(args.observations_npz),
        "used_frames": used_frames,
        "corrected_corners": len(report),
        "median_dx_mm": float(np.median(dmag)),
        "odd_even_split_med_mm": float(np.median(splits)),
        "depth_scale": args.depth_scale,
    }
    out_path = Path(args.model_json).with_name(
        Path(args.model_json).stem + "_ir3d.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[refine-ir] geschrieben: {out_path}")


if __name__ == "__main__":
    main()
