from __future__ import annotations

"""
debug_plane_3d.py

Live-Debug-Skript für die Plane-Fitting -> 3D-Grid Pipeline.
Es bildet die Logik der GUI-Page `page_plane_fitting.py` so eng wie möglich nach.

Workflow
--------
1) RealSense RGB-D Stream starten
2) Checkerboard live detektieren
3) SPACE drücken, wenn FOUND
4) Es werden N Depth-Frames gemittelt
5) Auf dem Freeze-Bild werden die 3 Extrempunkte bestimmt (TL, TR, BL)
6) Daraus wird exakt wie auf der Page gerechnet:
      - ROI
      - 3D-Punkte sampeln
      - fit_plane_stable(...)
      - intersect_corners_with_plane(...)
      - interpolate_marker_grid(...)
7) Es wird gespeichert:
      - TXT mit 11x11 Punkten im Kameraframe
      - PNG mit Principal Point + 11x11 Bildpunkten + 3 Extrempunkten
      - NPZ mit allen Debug-Daten
"""

import argparse
import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from overlay.tools import checkerboard_corner_detection as cbd
from overlay.tools import ransac_plane_fitting as rpf
from tracking.pose_filters import PlaneKalmanFilter


# ============================================================
# Defaults möglichst nah an page_plane_fitting.py
# ============================================================
PATTERN_SIZE = (3, 3)
DET_WIDTH = 640
PAD_PX = 15
MAX_POINTS = 5000
Z_MIN = 0.30
Z_MAX = 2.0
MIN_POINTS_FOR_FIT = 800
THRESH_M = 0.001
RANSAC_N = 8
ITERS = 3000
N_STABLE_RUNS = 10
N_AVERAGE_FRAMES = 30
FPS = 30
STEPS_PER_EDGE = 10
USE_TEMPORAL_FILTER = True
TEMPORAL_ALPHA = 0.1
TEMPORAL_DELTA = 20


# ============================================================
# Hardcoded K_rgb
# ============================================================
K_RGB_HARDCODED = np.array(
    [
        [1360.41301, 0.0, 976.230766],
        [0.0, 1361.74342, 547.474129],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


# ============================================================
# Helpers
# ============================================================
def load_K_rgb(npz_path: Path | None) -> np.ndarray:
    """If a path is given, load K_rgb from NPZ, else use hardcoded."""
    if npz_path is None:
        return K_RGB_HARDCODED.copy()

    data = np.load(str(npz_path))
    if "K_rgb" not in data:
        raise KeyError(f"Missing key 'K_rgb' in {npz_path}")

    K = np.asarray(data["K_rgb"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("K_rgb must have shape (3, 3)")
    return K


def make_output_dir(base: Path) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"debug_plane_3d_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def draw_extremes(img: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
    out = img.copy()
    for (u, v) in pts_uv:
        cv2.circle(out, (int(round(u)), int(round(v))), 10, (208, 224, 64), -1)
    return out


def intersect_ray_with_plane(
    uv: np.ndarray,
    rgb_intrinsics: np.ndarray,
    plane_model: np.ndarray,
) -> np.ndarray:
    """
    Intersect a single pixel ray with the fitted plane.

    uv: shape (2,) -> [u, v]
    returns: shape (3,) -> [X, Y, Z] in camera frame
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(2)
    K = np.asarray(rgb_intrinsics, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError("rgb_intrinsics must be a 3x3 matrix.")

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    a, b, c, d = [float(x) for x in plane_model]

    u, v = float(uv[0]), float(uv[1])
    x = (u - cx) / fx
    y = (v - cy) / fy

    denom = a * x + b * y + c
    if abs(denom) < 1e-12:
        raise ValueError("Ray is parallel to the fitted plane.")

    z = -d / denom
    return np.array([x * z, y * z, z], dtype=np.float64)


def draw_grid_and_pp(
    img: np.ndarray,
    K_rgb: np.ndarray,
    ext_uv: np.ndarray,
    grid_uv: np.ndarray,
    corners_full: np.ndarray | None = None,
    principal_axis_dbg: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    out = img.copy()

    cx = float(K_rgb[0, 2])
    cy = float(K_rgb[1, 2])
    pp = (int(round(cx)), int(round(cy)))

    # Principal point
    cv2.circle(out, pp, 10, (0, 0, 255), -1)
    cv2.circle(out, pp, 16, (255, 255, 255), 2)
    cv2.putText(
        out,
        "PP",
        (pp[0] + 12, pp[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # Crosshair through principal point
    h, w = out.shape[:2]
    cv2.line(out, (0, pp[1]), (w - 1, pp[1]), (0, 0, 255), 1)
    cv2.line(out, (pp[0], 0), (pp[0], h - 1), (0, 0, 255), 1)

    # Image direction arrows
    cv2.arrowedLine(out, pp, (pp[0] + 120, pp[1]), (255, 255, 255), 2, tipLength=0.08)
    cv2.putText(
        out,
        "+u / image right",
        (pp[0] + 125, pp[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.arrowedLine(out, pp, (pp[0], pp[1] + 120), (255, 255, 255), 2, tipLength=0.08)
    cv2.putText(
        out,
        "+v / image down",
        (pp[0] + 5, pp[1] + 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Optional: all detected 3x3 checkerboard corners with indices
    if corners_full is not None:
        cf = np.asarray(corners_full, dtype=np.float64).reshape(-1, 2)
        for i, (u, v) in enumerate(cf):
            cv2.circle(out, (int(round(u)), int(round(v))), 5, (255, 255, 0), -1)
            cv2.putText(
                out,
                str(i),
                (int(round(u)) + 5, int(round(v)) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    # Grid points
    for idx, (u, v) in enumerate(grid_uv):
        if np.isnan(u) or np.isnan(v):
            continue
        cv2.circle(out, (int(round(u)), int(round(v))), 4, (0, 255, 255), -1)
        if idx in {0, 10, 60, 110, 120}:
            cv2.putText(
                out,
                str(idx),
                (int(round(u)) + 5, int(round(v)) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Extrempunkte TL / TR / BL
    labels = ["TL", "TR", "BL"]
    cols = [(0, 255, 0), (255, 0, 0), (0, 165, 255)]
    for label, col, (u, v) in zip(labels, cols, ext_uv):
        cv2.circle(out, (int(round(u)), int(round(v))), 9, col, 2)
        cv2.putText(
            out,
            label,
            (int(round(u)) + 8, int(round(v)) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            col,
            2,
            cv2.LINE_AA,
        )

    # Principal-axis debug points
    if principal_axis_dbg is not None:
        dbg_uv = np.asarray(principal_axis_dbg["uv"], dtype=np.float64)
        dbg_labels = list(principal_axis_dbg["labels"])
        dbg_cols = {
            "PP": (0, 0, 255),
            "PP+u": (255, 255, 255),
            "PP-u": (180, 180, 180),
            "PP+v": (255, 255, 255),
            "PP-v": (180, 180, 180),
        }
        for label, (u, v) in zip(dbg_labels, dbg_uv):
            col = dbg_cols.get(label, (255, 255, 255))
            cv2.circle(out, (int(round(u)), int(round(v))), 6, col, 2)
            cv2.putText(
                out,
                label,
                (int(round(u)) + 6, int(round(v)) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                col,
                1,
                cv2.LINE_AA,
            )

    return out


def project_points(K: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    uv = []
    for X, Y, Z in xyz:
        if Z <= 1e-12:
            uv.append([np.nan, np.nan])
            continue
        u = K[0, 0] * (X / Z) + K[0, 2]
        v = K[1, 1] * (Y / Z) + K[1, 2]
        uv.append([u, v])
    return np.asarray(uv, dtype=np.float64)


def principal_axis_debug_points(
    K: np.ndarray,
    plane_model: np.ndarray,
    du_px: float = 100.0,
    dv_px: float = 100.0,
) -> dict[str, np.ndarray]:
    """
    Construct 5 synthetic image points around the principal point and intersect
    their rays with the fitted plane. This shows directly which 3D axis changes
    when moving in +u / +v in the image.
    """
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    pts_uv = np.array(
        [
            [cx, cy],
            [cx + du_px, cy],
            [cx - du_px, cy],
            [cx, cy + dv_px],
            [cx, cy - dv_px],
        ],
        dtype=np.float64,
    )
    pts_xyz = np.array(
        [intersect_ray_with_plane(uv, K, plane_model) for uv in pts_uv],
        dtype=np.float64,
    )
    return {
        "uv": pts_uv,
        "xyz": pts_xyz,
        "labels": np.array(["PP", "PP+u", "PP-u", "PP+v", "PP-v"], dtype=object),
    }


def depth_u16_to_vis_bgr(depth_u16: np.ndarray) -> np.ndarray:
    depth_u16 = np.asarray(depth_u16, dtype=np.uint16)
    nonzero = depth_u16[depth_u16 > 0]
    if nonzero.size == 0:
        return np.zeros((depth_u16.shape[0], depth_u16.shape[1], 3), dtype=np.uint8)
    lo = float(np.percentile(nonzero, 2.0))
    hi = float(np.percentile(nonzero, 98.0))
    if hi <= lo:
        hi = lo + 1.0
    depth_8u = np.clip(
        (depth_u16.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255
    ).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    depth_bgr[depth_u16 == 0] = (0, 0, 0)
    return depth_bgr


def capture_averaged_depth(
    pipeline: rs.pipeline,
    align: rs.align,
    temporal_filter,
    n_average_frames: int,
):
    accumulator = None
    count_map = None
    last_df = None
    last_color = None

    for _ in range(n_average_frames):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not df:
            continue
        if temporal_filter is not None:
            try:
                df = temporal_filter.process(df).as_depth_frame()
            except Exception:
                pass
        raw = np.asanyarray(df.get_data()).astype(np.float64)
        if accumulator is None:
            accumulator = np.zeros_like(raw, dtype=np.float64)
            count_map = np.zeros_like(raw, dtype=np.int32)
        valid = raw > 0
        accumulator[valid] += raw[valid]
        count_map[valid] += 1
        last_df = df
        if cf:
            last_color = np.asanyarray(cf.get_data()).copy()

    if accumulator is None or last_df is None:
        return None, None, None

    with np.errstate(invalid="ignore"):
        averaged = np.where(count_map > 0, accumulator / count_map, np.nan).astype(np.float32)

    return averaged, last_df, last_color


def sample_pts3d_from_averaged(
    averaged_raw: np.ndarray,
    intrinsics,
    rect: tuple[int, int, int, int],
    max_points: int,
    z_min: float,
    z_max: float,
    seed: int,
    depth_scale_m: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    umin, vmin, umax, vmax = rect
    roi_w = umax - umin + 1
    roi_h = vmax - vmin + 1
    if roi_w <= 2 or roi_h <= 2:
        return np.empty((0, 3), dtype=np.float64)

    num_pixels = roi_w * roi_h
    idx = rng.choice(num_pixels, size=min(max_points, num_pixels), replace=False)
    us = (idx % roi_w).astype(np.int32) + umin
    vs = (idx // roi_w).astype(np.int32) + vmin

    pts3d = []
    for u, v in zip(us, vs):
        raw = averaged_raw[v, u]
        if np.isnan(raw) or raw <= 0:
            continue
        z = float(raw) * depth_scale_m
        if z < z_min or z > z_max:
            continue
        xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], z)
        pts3d.append(xyz)

    return np.asarray(pts3d, dtype=np.float64) if pts3d else np.empty((0, 3), dtype=np.float64)


def save_txt(
    txt_path: Path,
    points_xyz_camera: np.ndarray,
    points_uv: np.ndarray,
    K_rgb: np.ndarray,
    ext_uv: np.ndarray,
    corner_xyz: np.ndarray,
    plane_model: np.ndarray,
    principal_axis_dbg: dict[str, np.ndarray],
) -> None:
    cx = float(K_rgb[0, 2])
    cy = float(K_rgb[1, 2])

    dbg_uv = np.asarray(principal_axis_dbg["uv"], dtype=np.float64)
    dbg_xyz = np.asarray(principal_axis_dbg["xyz"], dtype=np.float64)
    dbg_labels = list(principal_axis_dbg["labels"])

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("DEBUG PLANE 3D\n")
        f.write("=" * 90 + "\n\n")

        f.write("K_rgb\n")
        f.write(str(K_rgb) + "\n\n")

        f.write(f"Principal point: cx={cx:.6f}, cy={cy:.6f}\n\n")

        f.write("Plane model [a b c d]\n")
        f.write(np.array2string(np.asarray(plane_model, dtype=np.float64), precision=9) + "\n\n")

        f.write("Extreme corners UV [TL, TR, BL]\n")
        for i, (u, v) in enumerate(ext_uv):
            f.write(f"  {i}: u={u:.6f}, v={v:.6f}, du={u-cx:.6f}, dv={v-cy:.6f}\n")
        f.write("\n")

        f.write("Extreme corners XYZ [TL, TR, BL]\n")
        for i, (X, Y, Z) in enumerate(corner_xyz):
            f.write(f"  {i}: X={X*1000:.6f} mm, Y={Y*1000:.6f} mm, Z={Z*1000:.6f} mm\n")
        f.write("\n")

        f.write("Principal-axis debug points\n")
        for label, uv, xyz in zip(dbg_labels, dbg_uv, dbg_xyz):
            u, v = uv
            X, Y, Z = xyz * 1000.0
            f.write(
                f"  {label}: u={u:.3f}, v={v:.3f}, du={u-cx:.3f}, dv={v-cy:.3f} || "
                f"X={X:.3f} mm, Y={Y:.3f} mm, Z={Z:.3f} mm\n"
            )
        f.write("\n")

        f.write("Axis-delta interpretation from principal point\n")
        pp_xyz = dbg_xyz[0]
        for label, xyz in zip(dbg_labels[1:], dbg_xyz[1:]):
            dxyz = (xyz - pp_xyz) * 1000.0
            f.write(
                f"  {label} - PP = dX={dxyz[0]:+.3f} mm, "
                f"dY={dxyz[1]:+.3f} mm, dZ={dxyz[2]:+.3f} mm\n"
            )
        f.write("\n")

        f.write("IDX | Xc [mm]   Yc [mm]   Zc [mm] || u [px]   v [px]   du=u-cx   dv=v-cy\n")
        f.write("-" * 110 + "\n")
        for idx, (p, uv) in enumerate(zip(points_xyz_camera, points_uv)):
            X, Y, Z = p * 1000.0
            u, v = uv
            f.write(
                f"{idx:3d} | "
                f"{X:9.3f} {Y:9.3f} {Z:9.3f} || "
                f"{u:8.3f} {v:8.3f} {u-cx:9.3f} {v-cy:9.3f}\n"
            )

        mins = np.min(points_xyz_camera, axis=0) * 1000.0
        maxs = np.max(points_xyz_camera, axis=0) * 1000.0
        means = np.mean(points_xyz_camera, axis=0) * 1000.0

        f.write("\nPOINTS XYZ stats [mm]\n")
        f.write(f"  min  = {mins}\n")
        f.write(f"  max  = {maxs}\n")
        f.write(f"  mean = {means}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k-rgb-npz",
        type=Path,
        default=None,
        help="Optional: NPZ mit K_rgb (wenn nicht gesetzt, wird hardcoded K_rgb verwendet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug_plane_outputs"),
        help="Basis-Outputordner",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed für das ROI-Sampling",
    )
    args = parser.parse_args()

    K_rgb = load_K_rgb(args.k_rgb_npz)
    out_dir = make_output_dir(args.output_dir)

    plane_kf = PlaneKalmanFilter(
        process_noise=1e-7,
        measurement_noise=1e-4,
        outlier_angle_deg=1.5,
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_m = float(depth_sensor.get_depth_scale())

    temporal_filter = None
    if USE_TEMPORAL_FILTER:
        try:
            temporal_filter = rs.temporal_filter()
            temporal_filter.set_option(rs.option.filter_smooth_alpha, float(TEMPORAL_ALPHA))
            temporal_filter.set_option(rs.option.filter_smooth_delta, float(TEMPORAL_DELTA))
        except Exception:
            temporal_filter = None

    print("\nDEBUG PLANE 3D gestartet")
    print("- Bewege das Checkerboard bis FOUND erscheint")
    print("- SPACE = Capture")
    print("- ESC   = Quit\n")

    try:
        live_color = None
        found_live = False

        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()

            if df and temporal_filter is not None:
                try:
                    df = temporal_filter.process(df).as_depth_frame()
                except Exception:
                    pass

            if cf:
                live_color = np.asanyarray(cf.get_data()).copy()
                gray = cv2.cvtColor(live_color, cv2.COLOR_BGR2GRAY)
                found, _ = cbd.detect_classic_downscaled(
                    gray,
                    PATTERN_SIZE,
                    det_width=DET_WIDTH,
                )
                found_live = bool(found)

            if live_color is None:
                continue

            vis = live_color.copy()
            txt = "FOUND (SPACE)" if found_live else "NOT FOUND"
            col = (0, 255, 0) if found_live else (0, 0, 255)
            cv2.putText(vis, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(vis, txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 2, cv2.LINE_AA)
            cv2.imshow("debug_plane_3d", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("Abbruch durch ESC.")
                break

            if key == 32 and found_live:  # SPACE
                print("\n[Capture] Sammle gemittelte Depth-Frames ...")
                averaged_raw, last_df, snap_color = capture_averaged_depth(
                    pipeline,
                    align,
                    temporal_filter,
                    N_AVERAGE_FRAMES,
                )
                if averaged_raw is None or last_df is None or snap_color is None:
                    print("[ERR] Averaged depth capture fehlgeschlagen.")
                    continue

                found, corners = cbd.detect_snapshot_full(
                    snap_color,
                    pattern_size=PATTERN_SIZE,
                    det_width=DET_WIDTH,
                )
                if (not found) or corners is None:
                    print("[ERR] Checkerboard snapshot detection fehlgeschlagen.")
                    continue

                ex = cbd.get_extreme_corners_geometric(corners)
                ext_uv = np.array(
                    [ex["top_left"], ex["top_right"], ex["bottom_left"]],
                    dtype=np.float64,
                )

                preview = snap_color.copy()
                cv2.drawChessboardCorners(preview, PATTERN_SIZE, corners, True)
                preview = draw_extremes(preview, ext_uv)
                cv2.imshow("debug_plane_3d_freeze", preview)
                cv2.waitKey(150)

                h, w = snap_color.shape[:2]
                rect = rpf.rect_from_pts(ext_uv, w, h, PAD_PX)
                intrinsics = last_df.profile.as_video_stream_profile().intrinsics

                pts3d = sample_pts3d_from_averaged(
                    averaged_raw=averaged_raw,
                    intrinsics=intrinsics,
                    rect=rect,
                    max_points=MAX_POINTS,
                    z_min=Z_MIN,
                    z_max=Z_MAX,
                    seed=args.seed,
                    depth_scale_m=depth_scale_m,
                )
                print(f"[Plane] Sampled points: {pts3d.shape[0]}")
                if pts3d.shape[0] < MIN_POINTS_FOR_FIT:
                    print(f"[ERR] Nicht genug Punkte für Plane Fit: {pts3d.shape[0]}")
                    continue

                plane_raw, inliers = rpf.fit_plane_stable(
                    pts3d,
                    distance_threshold=THRESH_M,
                    ransac_n=RANSAC_N,
                    num_iterations=ITERS,
                    n_runs=N_STABLE_RUNS,
                )
                plane_filtered = plane_kf.update(plane_raw)

                dev = rpf.deviations(pts3d, plane_raw)
                dev_in = dev[inliers] if len(inliers) else dev
                mean = float(np.mean(dev_in))
                med = float(np.median(dev_in))
                p95 = float(np.percentile(dev_in, 95))
                print(
                    f"[Plane] inliers {len(inliers)}/{pts3d.shape[0]} | "
                    f"mean={mean*1000:.3f} mm | median={med*1000:.3f} mm | p95={p95*1000:.3f} mm"
                )

                corner_xyz = rpf.intersect_corners_with_plane(ext_uv, K_rgb, plane_filtered)
                points_xyz_camera = rpf.interpolate_marker_grid(corner_xyz, steps_per_edge=STEPS_PER_EDGE)
                points_uv = project_points(K_rgb, points_xyz_camera)
                principal_axis_dbg = principal_axis_debug_points(K_rgb, plane_filtered)

                overlay = draw_grid_and_pp(
                    snap_color,
                    K_rgb,
                    ext_uv,
                    points_uv,
                    corners_full=corners,
                    principal_axis_dbg=principal_axis_dbg,
                )
                depth_vis = depth_u16_to_vis_bgr(np.nan_to_num(averaged_raw, nan=0.0).astype(np.uint16))

                txt_path = out_dir / "debug_plane_3d_points.txt"
                img_path = out_dir / "debug_plane_3d_overlay.png"
                freeze_path = out_dir / "debug_plane_3d_freeze.png"
                depth_path = out_dir / "debug_plane_3d_depth.png"
                npz_path = out_dir / "debug_plane_3d_data.npz"

                save_txt(
                    txt_path=txt_path,
                    points_xyz_camera=points_xyz_camera,
                    points_uv=points_uv,
                    K_rgb=K_rgb,
                    ext_uv=ext_uv,
                    corner_xyz=corner_xyz,
                    plane_model=plane_filtered,
                    principal_axis_dbg=principal_axis_dbg,
                )

                cv2.imwrite(str(img_path), overlay)
                cv2.imwrite(str(freeze_path), preview)
                cv2.imwrite(str(depth_path), depth_vis)

                np.savez(
                    str(npz_path),
                    K_rgb=K_rgb,
                    ext_uv=ext_uv,
                    corner_xyz=corner_xyz,
                    plane_raw=plane_raw,
                    plane_filtered=plane_filtered,
                    points_xyz_camera=points_xyz_camera,
                    points_uv=points_uv,
                    principal_axis_dbg_uv=principal_axis_dbg["uv"],
                    principal_axis_dbg_xyz=principal_axis_dbg["xyz"],
                    rgb_image=snap_color,
                    depth_avg_raw=averaged_raw,
                    sampled_pts3d=pts3d,
                    inliers=inliers,
                    rect=np.asarray(rect, dtype=np.int32),
                )

                print(f"\n[OK] Dateien gespeichert in: {out_dir}")
                print(f"  - {txt_path.name}")
                print(f"  - {img_path.name}")
                print(f"  - {freeze_path.name}")
                print(f"  - {depth_path.name}")
                print(f"  - {npz_path.name}")

                cv2.imshow("debug_plane_3d_result", overlay)
                print("\nErgebnisfenster geöffnet. Taste drücken zum Beenden.")
                cv2.waitKey(0)
                break

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()