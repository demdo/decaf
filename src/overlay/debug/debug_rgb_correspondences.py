# -*- coding: utf-8 -*-
"""
debug_rgb_correspondences.py

Live RealSense debug for the RGB-side board pose T_bc.

What it does
------------
- Opens RealSense RGB-D live
- Runs the SAME checkerboard detection as in the project
- On SPACE:
    1) captures averaged depth
    2) detects checkerboard on the frozen RGB frame
    3) computes TL / TR / BL
    4) runs the SAME multi-run plane fitting logic as the project
       and uses the FINAL FILTERED board points in camera frame
    5) computes T_bc^depth from those filtered 121 board points
    6) computes RGB IPPE candidates from the affine 121 RGB grid
    7) applies the SAME trust-region selection as in the project
    8) shows:
         - captured RGB correspondences (121)
         - reprojection from final T_bc
    9) prints trust-region stats for BOTH RGB IPPE candidates

Controls
--------
SPACE : capture and run debug
ESC   : quit

Important
---------
- Put your CALIBRATED K_rgb below. Do NOT use ad-hoc stream intrinsics for the
  board reconstruction / pose logic if you want parity with the project.
- This script is ONLY for T_bc (board -> camera), not T_cx.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs

from overlay.tools import checkerboard_corner_detection as cbd
from overlay.tools import ransac_plane_fitting as rpf
from overlay.tools.homography import build_board_xyz_canonical
from tracking.pose_filters import PlaneKalmanFilter

from tracking.pose_solvers import (
    _build_ippe_candidates,
    _export_ippe_candidate_result,
    _make_transform,
    _rigid_fit_kabsch,
    _score_candidate_against_depth,
    _select_ippe_candidate_rgb,
    _trust_region_from_depth_rms,
    normalize_dist_coeffs,
)


# ============================================================
# USER CONFIG
# ============================================================

# IMPORTANT: use your CALIBRATED RGB intrinsics here
K_RGB_CALIB = np.array([
    [1360.41, 0.0, 976.23],
    [0.0, 1361.74, 547.47],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

PATTERN_SIZE = (3, 3)
DET_WIDTH = 640

STEPS_PER_EDGE = 10
PITCH_MM = 2.54

PAD_PX = 15
MAX_POINTS = 5000
Z_MIN_M = 0.30
Z_MAX_M = 2.0
MIN_POINTS_FOR_FIT = 800

THRESH_M = 0.001
RANSAC_N = 8
ITERS = 3000
N_STABLE_RUNS = 10
N_FIT_RUNS = 10

N_AVERAGE_FRAMES = 30

COLOR_SIZE = (1920, 1080)
DEPTH_SIZE = (1280, 720)
FPS = 30

USE_TEMPORAL_FILTER = False
TEMPORAL_ALPHA = 0.1
TEMPORAL_DELTA = 20.0


# ============================================================
# Result containers
# ============================================================

@dataclass(frozen=True)
class CandidateTrustStats:
    candidate_index: int
    reproj_mean_px: float
    reproj_median_px: float
    reproj_max_px: float
    delta_t_mm: float
    delta_r_deg: float
    score: float
    feasible: bool
    selected: bool


@dataclass(frozen=True)
class DebugTbcResult:
    T_bc_depth: np.ndarray
    T_bc_rgb: np.ndarray
    candidate_index_rgb: int
    depth_fit_rms_mm: float
    gamma_t_mm: float
    gamma_r_deg: float
    all_candidates_rgb: list
    candidate_trust_stats: list[CandidateTrustStats]
    points_xyz_camera_filtered_m: np.ndarray
    points_uv_rgb_121: np.ndarray


# ============================================================
# Formatting helpers
# ============================================================

def fmt_vec(v: np.ndarray) -> str:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{x:+.6f}" for x in v) + "]"


def fmt_T(T: np.ndarray, indent: str = "    ") -> str:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    lines = []
    for row in T:
        lines.append(indent + "  ".join(f"{x:+.6f}" for x in row))
    return "\n".join(lines)


def rotation_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def compare_transforms(T_ref: np.ndarray, T_test: np.ndarray) -> dict:
    T_ref = np.asarray(T_ref, dtype=np.float64).reshape(4, 4)
    T_test = np.asarray(T_test, dtype=np.float64).reshape(4, 4)

    R_ref = T_ref[:3, :3]
    t_ref = T_ref[:3, 3]

    R_test = T_test[:3, :3]
    t_test = T_test[:3, 3]

    R_rel = R_ref.T @ R_test
    delta_R_deg = rotation_angle_deg(R_rel)
    delta_t_mm = float(np.linalg.norm(t_test - t_ref))

    return {
        "delta_R_deg": delta_R_deg,
        "delta_t_mm": delta_t_mm,
    }


# ============================================================
# RealSense helpers
# ============================================================

def start_realsense() -> tuple[rs.pipeline, rs.align, float, rs.temporal_filter | None]:
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, COLOR_SIZE[0], COLOR_SIZE[1], rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_SIZE[0], DEPTH_SIZE[1], rs.format.z16, FPS)

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

    return pipeline, align, depth_scale_m, temporal_filter


def capture_averaged_depth(
    pipeline: rs.pipeline,
    align: rs.align,
    temporal_filter: rs.temporal_filter | None,
    n_average_frames: int,
) -> tuple[np.ndarray, object] | None:
    accumulator = None
    count_map = None
    last_df = None

    for _ in range(n_average_frames):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
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

    if accumulator is None or last_df is None:
        return None

    with np.errstate(invalid="ignore"):
        averaged = np.where(
            count_map > 0,
            accumulator / count_map,
            np.nan,
        ).astype(np.float32)

    return averaged, last_df


# ============================================================
# Geometry helpers
# ============================================================

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

        z_m = float(raw) * depth_scale_m
        if z_m < z_min or z_m > z_max:
            continue

        xyz = rs.rs2_deproject_pixel_to_point(
            intrinsics,
            [float(u), float(v)],
            z_m,
        )
        pts3d.append(xyz)

    if not pts3d:
        return np.empty((0, 3), dtype=np.float64)

    return np.asarray(pts3d, dtype=np.float64)


def plane_to_grid_xyz_camera_filtered(
    ext_uv: np.ndarray,
    plane_abcd: np.ndarray,
    K_rgb: np.ndarray,
    steps_per_edge: int,
) -> np.ndarray:
    corner_xyz = rpf.intersect_corners_with_plane(
        ext_uv,
        K_rgb,
        plane_abcd,
    )
    return rpf.interpolate_marker_grid(
        corner_xyz,
        steps_per_edge=int(steps_per_edge),
    )


def build_rgb_grid_from_extremes(
    ext_uv: np.ndarray,
    steps_per_edge: int,
) -> np.ndarray:
    ext_uv = np.asarray(ext_uv, dtype=np.float64).reshape(3, 2)
    p_tl, p_tr, p_bl = ext_uv

    step_x = (p_tr - p_tl) / float(steps_per_edge)
    step_y = (p_bl - p_tl) / float(steps_per_edge)

    pts2d_rgb = np.array(
        [
            p_tl + alpha * step_x + beta * step_y
            for beta in range(steps_per_edge + 1)
            for alpha in range(steps_per_edge + 1)
        ],
        dtype=np.float64,
    )

    return pts2d_rgb


def project_points(
    object_points_xyz: np.ndarray,
    T_bc: np.ndarray,
    K_rgb: np.ndarray,
) -> np.ndarray:
    object_points_xyz = np.asarray(object_points_xyz, dtype=np.float64).reshape(-1, 3)
    T_bc = np.asarray(T_bc, dtype=np.float64).reshape(4, 4)
    K_rgb = np.asarray(K_rgb, dtype=np.float64).reshape(3, 3)

    R = T_bc[:3, :3]
    t = T_bc[:3, 3].reshape(3, 1)

    rvec, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(
        object_points_xyz,
        rvec,
        t,
        K_rgb,
        np.zeros((5, 1), dtype=np.float64),
    )
    return uv.reshape(-1, 2)


# ============================================================
# Exact project-like filtered plane path
# ============================================================

def compute_filtered_board_points_like_project(
    averaged_raw: np.ndarray,
    intrinsics,
    depth_scale_m: float,
    ext_uv: np.ndarray,
    K_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Mimics the important project logic:
    - multiple fit runs
    - fit_plane_stable on each run
    - PlaneKalmanFilter across runs
    - final board points from the filtered plane of the final run
    """
    h = int(averaged_raw.shape[0])
    w = int(averaged_raw.shape[1])
    rect = rpf.rect_from_pts(ext_uv, w, h, PAD_PX)

    plane_kf = PlaneKalmanFilter(
        process_noise=1e-7,
        measurement_noise=1e-4,
        outlier_angle_deg=1.5,
    )

    last_marker_xyz_filtered: np.ndarray | None = None
    last_plane_filtered: np.ndarray | None = None
    last_stats: dict | None = None

    for run_idx in range(N_FIT_RUNS):
        pts3d = sample_pts3d_from_averaged(
            averaged_raw=averaged_raw,
            intrinsics=intrinsics,
            rect=rect,
            max_points=MAX_POINTS,
            z_min=Z_MIN_M,
            z_max=Z_MAX_M,
            seed=run_idx,
            depth_scale_m=depth_scale_m,
        )

        if pts3d.shape[0] < MIN_POINTS_FOR_FIT:
            print(f"[run {run_idx}] not enough points: {pts3d.shape[0]}")
            continue

        plane_raw, inliers = rpf.fit_plane_stable(
            pts3d,
            distance_threshold=THRESH_M,
            ransac_n=RANSAC_N,
            num_iterations=ITERS,
            n_runs=N_STABLE_RUNS,
        )
        plane_raw = np.asarray(plane_raw, dtype=np.float64)
        plane_filtered = np.asarray(plane_kf.update(plane_raw), dtype=np.float64)

        marker_xyz_filtered = plane_to_grid_xyz_camera_filtered(
            ext_uv=ext_uv,
            plane_abcd=plane_filtered,
            K_rgb=K_rgb,
            steps_per_edge=STEPS_PER_EDGE,
        )

        dev = rpf.deviations(pts3d, plane_raw)
        dev_in = dev[inliers] if len(inliers) else dev

        last_marker_xyz_filtered = marker_xyz_filtered
        last_plane_filtered = plane_filtered
        last_stats = {
            "run_idx": run_idx,
            "inliers": int(len(inliers)),
            "num_pts": int(pts3d.shape[0]),
            "mean_mm": float(np.mean(dev_in) * 1000.0),
            "median_mm": float(np.median(dev_in) * 1000.0),
            "p95_mm": float(np.percentile(dev_in, 95) * 1000.0),
        }

    if last_marker_xyz_filtered is None or last_plane_filtered is None or last_stats is None:
        raise RuntimeError("All filtered plane-fit runs failed.")

    return last_marker_xyz_filtered, last_plane_filtered, last_stats


# ============================================================
# Pose computation for T_bc
# ============================================================

def compute_t_bc_exact_current_logic(
    ext_uv: np.ndarray,
    points_xyz_camera_filtered_m: np.ndarray,
    K_rgb: np.ndarray,
    steps_per_edge: int,
    pitch_mm: float,
) -> DebugTbcResult:
    dist_zero = normalize_dist_coeffs(None)

    pts3d_board_mm = build_board_xyz_canonical(
        nu=int(steps_per_edge),
        nv=int(steps_per_edge),
        pitch_mm=float(pitch_mm),
    )

    pts2d_rgb = build_rgb_grid_from_extremes(
        ext_uv=ext_uv,
        steps_per_edge=steps_per_edge,
    )

    cam_points_xyz_mm = np.asarray(points_xyz_camera_filtered_m, dtype=np.float64).reshape(-1, 3) * 1000.0

    depth_fit = _rigid_fit_kabsch(
        board_xyz_mm=pts3d_board_mm,
        cam_xyz_mm=cam_points_xyz_mm,
    )
    T_bc_depth = depth_fit["T"]

    tr = _trust_region_from_depth_rms(depth_fit["rms_mm"])

    candidates_rgb = _build_ippe_candidates(
        object_points_xyz=pts3d_board_mm,
        image_points_uv=pts2d_rgb,
        K=K_rgb,
        dist=dist_zero,
    )

    best_idx_rgb = _select_ippe_candidate_rgb(
        candidates_rgb,
        T_bc_depth=T_bc_depth,
        gamma_t_mm=tr["gamma_t_mm"],
        gamma_r_deg=tr["gamma_r_deg"],
    )

    best_rgb = candidates_rgb[best_idx_rgb]
    T_bc_rgb = _make_transform(best_rgb.R, best_rgb.tvec.ravel())

    all_candidates_rgb = [
        _export_ippe_candidate_result(
            candidate=c,
            object_points_xyz=pts3d_board_mm,
            image_points_uv=pts2d_rgb,
            K=K_rgb,
            dist=dist_zero,
            candidate_index=i,
        )
        for i, c in enumerate(candidates_rgb)
    ]

    candidate_trust_stats: list[CandidateTrustStats] = []
    for cand_res in all_candidates_rgb:
        T_cand = _make_transform(cand_res.R, cand_res.tvec.ravel())

        trust_stats = _score_candidate_against_depth(
            T_depth=T_bc_depth,
            T_ippe=T_cand,
            gamma_t_mm=tr["gamma_t_mm"],
            gamma_r_deg=tr["gamma_r_deg"],
        )

        candidate_trust_stats.append(
            CandidateTrustStats(
                candidate_index=int(cand_res.candidate_index),
                reproj_mean_px=float(cand_res.reproj_mean_px),
                reproj_median_px=float(cand_res.reproj_median_px),
                reproj_max_px=float(cand_res.reproj_max_px),
                delta_t_mm=float(trust_stats["delta_t_mm"]),
                delta_r_deg=float(trust_stats["delta_r_deg"]),
                score=float(trust_stats["score"]),
                feasible=bool(trust_stats["feasible"]),
                selected=bool(cand_res.candidate_index == best_idx_rgb),
            )
        )

    return DebugTbcResult(
        T_bc_depth=T_bc_depth,
        T_bc_rgb=T_bc_rgb,
        candidate_index_rgb=int(best_idx_rgb),
        depth_fit_rms_mm=float(depth_fit["rms_mm"]),
        gamma_t_mm=float(tr["gamma_t_mm"]),
        gamma_r_deg=float(tr["gamma_r_deg"]),
        all_candidates_rgb=all_candidates_rgb,
        candidate_trust_stats=candidate_trust_stats,
        points_xyz_camera_filtered_m=np.asarray(points_xyz_camera_filtered_m, dtype=np.float64).reshape(-1, 3),
        points_uv_rgb_121=np.asarray(pts2d_rgb, dtype=np.float64).reshape(-1, 2),
    )


# ============================================================
# Drawing helpers
# ============================================================

def draw_extremes(img: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
    out = img.copy()
    labels = ["TL", "TR", "BL"]

    for k, (u, v) in enumerate(np.asarray(pts_uv, dtype=np.float64).reshape(-1, 2)):
        uu = int(round(u))
        vv = int(round(v))
        cv2.circle(out, (uu, vv), 8, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (uu, vv), 10, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            out,
            labels[k],
            (uu + 10, vv - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return out


def draw_numbered_points(
    img: np.ndarray,
    pts_uv: np.ndarray,
    *,
    point_color=(0, 255, 0),
    text_color=(0, 255, 255),
    radius_px: int = 3,
    font_scale: float = 0.32,
    thickness: int = 1,
    title: str | None = None,
) -> np.ndarray:
    out = img.copy()

    if title:
        cv2.putText(
            out,
            title,
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for idx, p in enumerate(np.asarray(pts_uv, dtype=np.float64).reshape(-1, 2)):
        u = int(round(float(p[0])))
        v = int(round(float(p[1])))

        cv2.circle(out, (u, v), radius_px + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (u, v), radius_px, point_color, -1, cv2.LINE_AA)

        cv2.putText(
            out,
            str(idx),
            (u + 5, v - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            str(idx),
            (u + 5, v - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    return out


# ============================================================
# Main capture / debug routine
# ============================================================

def process_capture(
    pipeline: rs.pipeline,
    align: rs.align,
    temporal_filter: rs.temporal_filter | None,
    depth_scale_m: float,
    color_bgr: np.ndarray,
) -> None:
    print("\n" + "=" * 80)
    print("CAPTURE")
    print("=" * 80)

    avg_out = capture_averaged_depth(
        pipeline=pipeline,
        align=align,
        temporal_filter=temporal_filter,
        n_average_frames=N_AVERAGE_FRAMES,
    )
    if avg_out is None:
        print("[ERROR] Could not capture averaged depth.")
        return

    averaged_raw, last_df = avg_out
    intrinsics = last_df.profile.as_video_stream_profile().intrinsics
    K_rgb = K_RGB_CALIB.copy()

    found, corners = cbd.detect_snapshot_full(
        color_bgr,
        pattern_size=PATTERN_SIZE,
        det_width=DET_WIDTH,
    )
    if not found or corners is None:
        print("[ERROR] Checkerboard detection failed on snapshot.")
        return

    ex = cbd.get_extreme_corners_geometric(corners)
    ext_uv = np.array(
        [ex["top_left"], ex["top_right"], ex["bottom_left"]],
        dtype=np.float64,
    )

    print(f"ext_uv shape                : {ext_uv.shape}")
    print(f"TL / TR / BL                :\n{ext_uv}")
    print(f"K_rgb (calibrated) =\n{K_rgb}")

    try:
        points_xyz_camera_filtered_m, plane_filtered, plane_stats = compute_filtered_board_points_like_project(
            averaged_raw=averaged_raw,
            intrinsics=intrinsics,
            depth_scale_m=depth_scale_m,
            ext_uv=ext_uv,
            K_rgb=K_rgb,
        )
    except Exception as e:
        print(f"[ERROR] Filtered plane path failed: {e}")
        return

    print("\nFinal filtered plane result")
    print("-" * 80)
    print(f"plane_abcd_filtered         : {fmt_vec(plane_filtered)}")
    print(f"final run index             : {plane_stats['run_idx']}")
    print(f"inliers                     : {plane_stats['inliers']}/{plane_stats['num_pts']}")
    print(f"mean dev [mm]               : {plane_stats['mean_mm']:.6f}")
    print(f"median dev [mm]             : {plane_stats['median_mm']:.6f}")
    print(f"p95 dev [mm]                : {plane_stats['p95_mm']:.6f}")

    tbc_res = compute_t_bc_exact_current_logic(
        ext_uv=ext_uv,
        points_xyz_camera_filtered_m=points_xyz_camera_filtered_m,
        K_rgb=K_rgb,
        steps_per_edge=STEPS_PER_EDGE,
        pitch_mm=PITCH_MM,
    )

    print("\nT_bc selection")
    print("-" * 80)
    print(f"candidate_index_rgb         : {tbc_res.candidate_index_rgb}")
    print(f"depth_fit_rms_mm            : {tbc_res.depth_fit_rms_mm:.6f}")
    print(f"gamma_t_mm                  : {tbc_res.gamma_t_mm:.6f}")
    print(f"gamma_r_deg                 : {tbc_res.gamma_r_deg:.6f}")

    print("\nT_bc_depth =")
    print(fmt_T(tbc_res.T_bc_depth))
    print("\nT_bc_rgb =")
    print(fmt_T(tbc_res.T_bc_rgb))

    cmp = compare_transforms(tbc_res.T_bc_depth, tbc_res.T_bc_rgb)
    print("\nDepth vs selected RGB T_bc")
    print("-" * 80)
    print(f"delta_R [deg]               : {cmp['delta_R_deg']:.6f}")
    print(f"delta_t [mm]                : {cmp['delta_t_mm']:.6f}")

    print("\nRGB IPPE candidates (trust-region stats)")
    print("-" * 80)
    for stats in tbc_res.candidate_trust_stats:
        print(f"candidate {stats.candidate_index}:")
        print(f"    selected         : {stats.selected}")
        print(f"    reproj mean [px] : {stats.reproj_mean_px:.6f}")
        print(f"    reproj median[px]: {stats.reproj_median_px:.6f}")
        print(f"    reproj max [px]  : {stats.reproj_max_px:.6f}")
        print(f"    delta_t [mm]     : {stats.delta_t_mm:.6f}")
        print(f"    delta_R [deg]    : {stats.delta_r_deg:.6f}")
        print(f"    score            : {stats.score:.6f}")
        print(f"    feasible         : {stats.feasible}")
        print()

    pts3d_board_mm = build_board_xyz_canonical(
        nu=STEPS_PER_EDGE,
        nv=STEPS_PER_EDGE,
        pitch_mm=PITCH_MM,
    )

    uv_reproj = project_points(
        object_points_xyz=pts3d_board_mm,
        T_bc=tbc_res.T_bc_rgb,
        K_rgb=K_rgb,
    )

    base_detect = color_bgr.copy()
    cv2.drawChessboardCorners(base_detect, PATTERN_SIZE, corners, True)
    base_detect = draw_extremes(base_detect, ext_uv)

    vis_interp = draw_numbered_points(
        base_detect,
        tbc_res.points_uv_rgb_121,
        point_color=(0, 255, 0),
        text_color=(0, 255, 255),
        radius_px=3,
        font_scale=0.32,
        title="Captured RGB correspondences (121)",
    )

    vis_reproj = draw_numbered_points(
        base_detect,
        uv_reproj,
        point_color=(255, 0, 0),
        text_color=(0, 255, 255),
        radius_px=3,
        font_scale=0.32,
        title="Reprojection from final T_bc",
    )

    cv2.namedWindow("RGB correspondences (121)", cv2.WINDOW_NORMAL)
    cv2.imshow("RGB correspondences (121)", vis_interp)

    cv2.namedWindow("Reprojection from T_bc", cv2.WINDOW_NORMAL)
    cv2.imshow("Reprojection from T_bc", vis_reproj)

    print("\nPress any key in one of the windows to continue...")
    cv2.waitKey(0)

    cv2.destroyWindow("RGB correspondences (121)")
    cv2.destroyWindow("Reprojection from T_bc")


# ============================================================
# Live loop
# ============================================================

def main() -> None:
    try:
        pipeline, align, depth_scale_m, temporal_filter = start_realsense()
    except Exception as e:
        print(f"[ERROR] Could not start RealSense: {e}")
        sys.exit(1)

    print("debug_rgb_correspondences")
    print("SPACE = capture and debug T_bc")
    print("ESC   = quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            cf = frames.get_color_frame()
            if not cf:
                continue

            color_bgr = np.asanyarray(cf.get_data())
            gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)

            found_live, _ = cbd.detect_classic_downscaled(
                gray,
                PATTERN_SIZE,
                det_width=DET_WIDTH,
            )

            vis_live = color_bgr.copy()
            txt = "FOUND (SPACE)" if found_live else "NOT FOUND"
            col = (0, 255, 0) if found_live else (0, 0, 255)

            cv2.putText(
                vis_live,
                txt,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis_live,
                txt,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                col,
                2,
                cv2.LINE_AA,
            )

            cv2.namedWindow("live_rgb", cv2.WINDOW_NORMAL)
            cv2.imshow("live_rgb", vis_live)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key == 32 and found_live:
                process_capture(
                    pipeline=pipeline,
                    align=align,
                    temporal_filter=temporal_filter,
                    depth_scale_m=depth_scale_m,
                    color_bgr=color_bgr.copy(),
                )

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()