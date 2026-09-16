# -*- coding: utf-8 -*-
"""
debug_xray_intrinsics_radial.py

Re-run the X-ray intrinsics BA with the radial term enabled.

Why this script exists
----------------------
The shipped K_xray was produced by debug_hom_intrin.run_ba_center_pp_equal_f
with CALIB_FIX_K1..K6 set, i.e. distortion pinned to zero. The image
intensifier does have pincushion distortion, so that zero was absorbed into
the focal length.

Focal length and radial distortion are DEGENERATE for a single planar view -
a radial expansion looks like a longer focal length. They only become
separable across several views at different orientations, which is exactly
what calibrateCamera exploits. Fitting k1 from one view (with a free
homography for the unknown board pose) therefore yields a coefficient whose
scale part is arbitrary; applying such a k1 to an image shifts it by far more
than the distortion it was meant to remove.

So the only valid route is this one: the same BA, same views, same flags,
with K1 (and optionally K2) released. f and k1 then come out as a matched
pair and must be used together.

Control
-------
The run with distortion pinned to zero must reproduce the shipped values
exactly, otherwise the replication is not faithful and nothing below means
anything.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# The session behind K_xray_ba_center_pp_equal_f, identified by re-running the
# BA over every correspondence folder: this is the only one that reproduces the
# shipped f = 4586.3058 / rms = 0.740042.
CORR = Path(r"C:\Users\domin\Documents\Studium\Master\Masterarbeit\Bilder\Overlay"
            r"\Calibration\X_ray_Images\04222026\Correspondences")
OUT = Path(__file__).resolve().parents[1] / "data" / "K_xray_ba_center_pp_equal_f_k1.npz"

IMAGE_SIZE = (1024, 1024)
INITIAL_F_PX = 4650.0

# reference values of the shipped file, for the control run
REF_F = 4586.3058
REF_RMS = 0.74004234


def load_views() -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    files = sorted(CORR.glob("VIEW*_corr.npz"),
                   key=lambda p: int(p.stem.replace("VIEW", "").replace("_corr", "")))
    obj, img, names = [], [], []
    for f in files:
        d = np.load(str(f))
        XY = np.asarray(d["XY"], dtype=np.float64).reshape(-1, 2)
        uv = np.asarray(d["uv"], dtype=np.float64).reshape(-1, 2)
        obj.append(np.hstack([XY, np.zeros((len(XY), 1))]).astype(np.float32).reshape(-1, 1, 3))
        img.append(uv.astype(np.float32).reshape(-1, 1, 2))
        names.append(f.stem.replace("_corr", ""))
    return obj, img, names


def run_ba(obj, img, *, free_k1: bool = False, free_k2: bool = False):
    """debug_hom_intrin.run_ba_center_pp_equal_f, with K1/K2 optionally released."""
    w, h = IMAGE_SIZE
    K_init = np.array([[INITIAL_F_PX, 0.0, w / 2.0],
                       [0.0, INITIAL_F_PX, h / 2.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_init = np.zeros((8, 1), dtype=np.float64)

    flags = (cv2.CALIB_USE_INTRINSIC_GUESS
             | cv2.CALIB_FIX_PRINCIPAL_POINT
             | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST
             | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4
             | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
             | cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_FIX_TAUX_TAUY)
    if not free_k1:
        flags |= cv2.CALIB_FIX_K1
    if not free_k2:
        flags |= cv2.CALIB_FIX_K2

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 500, 1e-12)

    rms, K, dist, _, _ = cv2.calibrateCamera(
        objectPoints=obj, imagePoints=img, imageSize=IMAGE_SIZE,
        cameraMatrix=K_init, distCoeffs=dist_init,
        flags=flags, criteria=criteria,
    )
    return float(rms), np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64).ravel()


def main() -> None:
    obj, img, names = load_views()
    print(f"views: {len(names)} ({', '.join(names)}), {len(obj[0])} points each\n")

    print("CONTROL - distortion pinned to zero, as the shipped file was made")
    rms0, K0, d0 = run_ba(obj, img)
    print(f"  f   = {K0[0,0]:.4f}   (shipped {REF_F:.4f}, delta {K0[0,0]-REF_F:+.4f})")
    print(f"  rms = {rms0:.6f}      (shipped {REF_RMS:.6f}, delta {rms0-REF_RMS:+.6f})")
    # tolerance covers the float32 conversion of the image points; the shipped
    # run agrees to 0.003 px in f and 7e-5 px in rms
    ok = abs(K0[0, 0] - REF_F) < 0.01 and abs(rms0 - REF_RMS) < 1e-3
    print(f"  -> replication {'EXACT' if ok else 'DOES NOT MATCH - stop here'}\n")
    if not ok:
        return

    print("k1 released")
    rms1, K1, d1 = run_ba(obj, img, free_k1=True)
    print(f"  f   = {K1[0,0]:.4f}   ({K1[0,0]-K0[0,0]:+.4f} vs the zero-distortion fit)")
    print(f"  k1  = {d1[0]:+.6f}")
    print(f"  rms = {rms1:.6f}   ({100*(rms1/rms0-1):+.2f} %)\n")

    print("k1 and k2 released")
    rms2, K2, d2 = run_ba(obj, img, free_k1=True, free_k2=True)
    print(f"  f   = {K2[0,0]:.4f}")
    print(f"  k1  = {d2[0]:+.6f}   k2 = {d2[1]:+.6f}")
    print(f"  rms = {rms2:.6f}   ({100*(rms2/rms0-1):+.2f} %)\n")

    # How much does this pair actually move the image? The whole point is that
    # it must be comparable to the measured field (rms 2.0 px, max 6.0 px) and
    # not dominated by a scale term.
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    print("  image displacement of the k1 pair (undistort):")
    for r in (100, 200, 300, 400, 450):
        p = np.column_stack([512 + r * np.cos(th), 512 + r * np.sin(th)])
        u = cv2.undistortPoints(p.reshape(-1, 1, 2), K1,
                                np.array([d1[0], 0, 0, 0, 0]), P=K1).reshape(-1, 2)
        print(f"    r = {r:3d} px : {np.linalg.norm(u - p, axis=1).mean():6.2f} px")

    dist_out = np.zeros((5, 1), dtype=np.float64)
    dist_out[0, 0] = d1[0]

    np.savez(
        str(OUT),
        K=K1, K_x=K1, K_xray=K1,
        dist=dist_out, dist_x=dist_out, dist_xray=dist_out,
        rms=np.array(rms1),
        image_size=np.array(IMAGE_SIZE, dtype=np.int32),
        width=np.array(IMAGE_SIZE[0], dtype=np.int32),
        height=np.array(IMAGE_SIZE[1], dtype=np.int32),
        method=np.array("BA_CENTER_PP_EQUAL_F_K1", dtype="<U64"),
        uv_space=np.array("XRAY_WORKING_FLIPPED_UV", dtype="<U64"),
        uv_transform=np.array("horizontal_flip", dtype="<U32"),
        initial_f_px=np.array(INITIAL_F_PX),
        fixed_principal_point=np.array(True, dtype=bool),
        cx_fixed=np.array(IMAGE_SIZE[0] / 2.0),
        cy_fixed=np.array(IMAGE_SIZE[1] / 2.0),
        equal_focal_lengths=np.array(True, dtype=bool),
        distortion_fixed_zero=np.array(False, dtype=bool),
        used_view_names=np.array(names, dtype=object),
        note=np.array("f and k1 are a matched pair - use both or neither",
                      dtype="<U128"),
    )
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
