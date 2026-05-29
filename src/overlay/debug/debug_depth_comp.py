# -*- coding: utf-8 -*-
"""
debug_depth_eval.py

Evaluation script for pointer-tool depth d_x using only:

    1) iterative + refinement
    2) IPPE + refinement + Kalman filtering on the 3D tip position in camera coordinates

Behavior:
- SPACE: set reference, clear old samples, reset KF, start recording
- q/ESC: save recorded data to disk and quit

Saved output:
- one .npz file containing all recorded arrays + metadata
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from overlay.calib.calib_camera_to_pointer import calibrate_camera_to_pointer
from overlay.calib.calib_xray_to_pointer import extract_depth
from tracking.pose_filters import AdaptiveKalmanFilterCV3D
from tracking.transforms import invert_transform


# ============================================================
# Global config
# ============================================================

SAVE_RESULTS = True
SAVE_DIR = Path("debug_outputs/pointer_depth_eval")

K_RGB = np.array(
    [
        [1.36041301e03, 0.0, 9.76230766e02],
        [0.0, 1.36041301e03, 5.48233972e02],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

T_CX = np.array(
    [
        [0.99826878, -0.03444292, -0.04767723, 0.00271865],
        [-0.02366857, -0.97731641, 0.21045765, -0.10833699],
        [-0.05384452, -0.20896485, -0.97643968, 1.09274608],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


# ============================================================
# Helpers
# ============================================================

def draw_text_lines(
    img: np.ndarray,
    lines: list[str],
    org: tuple[int, int] = (30, 40),
    line_step: int = 32,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.8,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    out = img.copy()
    x0, y0 = org

    for i, line in enumerate(lines):
        y = y0 + i * line_step
        cv2.putText(
            out,
            line,
            (x0, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return out


def draw_tip_marker(
    img: np.ndarray,
    tip_uv: np.ndarray | None,
    *,
    color: tuple[int, int, int],
    marker_type: int,
    label: str,
) -> None:
    if tip_uv is None:
        return

    tip_uv = np.asarray(tip_uv, dtype=np.float64).reshape(2)
    if not np.isfinite(tip_uv).all():
        return

    u, v = np.round(tip_uv).astype(int)

    cv2.drawMarker(
        img,
        (int(u), int(v)),
        color,
        markerType=marker_type,
        markerSize=26,
        thickness=2,
        line_type=cv2.LINE_AA,
    )

    cv2.putText(
        img,
        label,
        (int(u) + 10, int(v) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def rotation_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return float(np.degrees(theta))


def project_tip_uv_from_camera_xyz(
    tip_xyz_c_mm: np.ndarray,
    K_rgb: np.ndarray,
) -> np.ndarray:
    point_xyz = np.asarray(tip_xyz_c_mm, dtype=np.float64).reshape(1, 3)
    K_rgb = np.asarray(K_rgb, dtype=np.float64).reshape(3, 3)
    dist = np.zeros((5, 1), dtype=np.float64)

    uv, _ = cv2.projectPoints(
        point_xyz,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        K_rgb,
        dist,
    )
    return uv.reshape(2)


def save_recording(
    *,
    save_dir: Path,
    times_s: list[float],
    frame_idx_list: list[int],
    angle_deg_list: list[float],
    iter_tip_xyz_c_list: list[list[float]],
    iter_tip_uv_list: list[list[float]],
    iter_dx_list: list[float],
    iter_reproj_mean_list: list[float],
    ippe_kf_tip_xyz_c_list: list[list[float]],
    ippe_kf_tip_uv_list: list[list[float]],
    ippe_kf_dx_list: list[float],
    ippe_reproj_mean_list: list[float],
    ippe_reproj_median_list: list[float],
    ippe_reproj_max_list: list[float],
    ippe_solution_index_list: list[int],
    n_markers_list: list[int],
    kf_motion_score_list: list[float],
    kf_tip_step_mm_list: list[float],
    kf_rot_step_deg_list: list[float],
    K_rgb: np.ndarray,
    T_cx: np.ndarray,
    T_xc: np.ndarray,
) -> Path | None:
    if len(times_s) == 0:
        print("[SAVE] No samples recorded. Nothing saved.")
        return None

    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = save_dir / f"pointer_depth_eval_{timestamp}.npz"

    np.savez_compressed(
        out_path,
        times_s=np.asarray(times_s, dtype=np.float64),
        frame_idx=np.asarray(frame_idx_list, dtype=np.int32),
        angle_deg=np.asarray(angle_deg_list, dtype=np.float64),

        iter_tip_xyz_c_mm=np.asarray(iter_tip_xyz_c_list, dtype=np.float64),
        iter_tip_uv=np.asarray(iter_tip_uv_list, dtype=np.float64),
        iter_d_x_mm=np.asarray(iter_dx_list, dtype=np.float64),
        iter_reproj_mean_px=np.asarray(iter_reproj_mean_list, dtype=np.float64),

        ippe_kf_tip_xyz_c_mm=np.asarray(ippe_kf_tip_xyz_c_list, dtype=np.float64),
        ippe_kf_tip_uv=np.asarray(ippe_kf_tip_uv_list, dtype=np.float64),
        ippe_kf_d_x_mm=np.asarray(ippe_kf_dx_list, dtype=np.float64),

        ippe_reproj_mean_px=np.asarray(ippe_reproj_mean_list, dtype=np.float64),
        ippe_reproj_median_px=np.asarray(ippe_reproj_median_list, dtype=np.float64),
        ippe_reproj_max_px=np.asarray(ippe_reproj_max_list, dtype=np.float64),
        ippe_solution_index=np.asarray(ippe_solution_index_list, dtype=np.int32),
        n_markers=np.asarray(n_markers_list, dtype=np.int32),

        kf_motion_score=np.asarray(kf_motion_score_list, dtype=np.float64),
        kf_tip_step_mm=np.asarray(kf_tip_step_mm_list, dtype=np.float64),
        kf_rot_step_deg=np.asarray(kf_rot_step_deg_list, dtype=np.float64),

        K_rgb=np.asarray(K_rgb, dtype=np.float64),
        T_cx=np.asarray(T_cx, dtype=np.float64),
        T_xc=np.asarray(T_xc, dtype=np.float64),
    )

    print(f"[SAVE] Saved recording to: {out_path}")
    return out_path


# ============================================================
# Main
# ============================================================

def main() -> None:
    K_rgb = K_RGB.copy()
    T_cx = T_CX.copy()
    T_xc = invert_transform(T_cx)

    print("=" * 80)
    print("HARDCODED SETUP")
    print("=" * 80)
    print("K_rgb:")
    print(K_rgb)
    print()
    print("T_cx:")
    print(T_cx)
    print()
    print("T_xc = inv(T_cx):")
    print(T_xc)
    print()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    print("Starting RealSense pipeline...")
    pipeline.start(config)
    print("Pipeline started.")
    print()

    prev_rvec_iter: np.ndarray | None = None
    prev_tvec_iter: np.ndarray | None = None
    R_ref_iter: np.ndarray | None = None
    recording = False

    dt = 1.0 / 30.0
    kf_ippe_tip = AdaptiveKalmanFilterCV3D(
        dt=dt,
        q_pos_still=1e-4,
        q_vel_still=1e-2,
        r_still=8e-2,
        q_pos_move=5e-3,
        q_vel_move=3e-1,
        r_move=2e-2,
    )

    frame_idx = 0
    t0_record: float | None = None

    # --------------------------------------------------
    # Buffers to save
    # --------------------------------------------------
    times_s: list[float] = []
    frame_idx_list: list[int] = []
    angle_deg_list: list[float] = []

    iter_tip_xyz_c_list: list[list[float]] = []
    iter_tip_uv_list: list[list[float]] = []
    iter_dx_list: list[float] = []
    iter_reproj_mean_list: list[float] = []

    ippe_kf_tip_xyz_c_list: list[list[float]] = []
    ippe_kf_tip_uv_list: list[list[float]] = []
    ippe_kf_dx_list: list[float] = []

    ippe_reproj_mean_list: list[float] = []
    ippe_reproj_median_list: list[float] = []
    ippe_reproj_max_list: list[float] = []
    ippe_solution_index_list: list[int] = []
    n_markers_list: list[int] = []

    kf_motion_score_list: list[float] = []
    kf_tip_step_mm_list: list[float] = []
    kf_rot_step_deg_list: list[float] = []

    print("Controls:")
    print("  SPACE : set reference + clear old samples + reset KF + start recording")
    print("  q/ESC : save recording + quit")
    print()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_idx += 1
            img_bgr = np.asanyarray(color_frame.get_data())

            iter_error = None
            ippe_error = None

            result_iter = None
            depth_iter = None

            result_ippe = None
            depth_ippe = None

            current_angle_deg = None
            current_dx_iter_mm = None
            current_dx_ippe_kf_mm = None

            current_motion_score_ippe_kf = None
            current_tip_step_mm = None
            current_rot_step_deg = None

            tip_uv_iter = None
            tip_uv_ippe_kf = None
            tip_ippe_kf_xyz_c = None

            # --------------------------------------------------
            # ITERATIVE + REFINEMENT
            # --------------------------------------------------
            try:
                use_guess = (prev_rvec_iter is not None and prev_tvec_iter is not None)

                result_iter = calibrate_camera_to_pointer(
                    image_bgr=img_bgr,
                    camera_intrinsics=K_rgb,
                    dist_coeffs=None,
                    rvec_init=prev_rvec_iter,
                    tvec_init=prev_tvec_iter,
                    use_extrinsic_guess=use_guess,
                    pose_method="iterative",
                    refine_with_iterative=True,
                )

                prev_rvec_iter = result_iter.rvec.copy()
                prev_tvec_iter = result_iter.tvec.copy()

                depth_iter = extract_depth(
                    T_xc=T_xc,
                    T_tc=result_iter.T_4x4,
                )

                current_dx_iter_mm = float(depth_iter.d_x_mm)
                tip_uv_iter = np.asarray(result_iter.tip_uv, dtype=np.float64).reshape(2)

            except Exception as e:
                iter_error = str(e)
                prev_rvec_iter = None
                prev_tvec_iter = None

            # --------------------------------------------------
            # IPPE + REFINEMENT
            # --------------------------------------------------
            try:
                result_ippe = calibrate_camera_to_pointer(
                    image_bgr=img_bgr,
                    camera_intrinsics=K_rgb,
                    dist_coeffs=None,
                    rvec_init=None,
                    tvec_init=None,
                    use_extrinsic_guess=False,
                    pose_method="ippe",
                    refine_with_iterative=True,
                )

                tip_ippe_cam_filt = kf_ippe_tip.filter(
                    measurement_mm=result_ippe.tip_point_camera_mm,
                    rotation_camera=result_ippe.rotation,
                )

                tip_ippe_kf_xyz_c = np.asarray(
                    tip_ippe_cam_filt,
                    dtype=np.float64,
                ).reshape(3)

                T_tc_ippe_kf = np.asarray(result_ippe.T_4x4, dtype=np.float64).copy()
                T_tc_ippe_kf[:3, 3] = tip_ippe_kf_xyz_c.reshape(3)

                depth_ippe = extract_depth(
                    T_xc=T_xc,
                    T_tc=T_tc_ippe_kf,
                )
                current_dx_ippe_kf_mm = float(depth_ippe.d_x_mm)

                tip_uv_ippe_kf = project_tip_uv_from_camera_xyz(
                    tip_ippe_kf_xyz_c,
                    K_rgb,
                )

                current_motion_score_ippe_kf = getattr(kf_ippe_tip, "last_motion_score", None)
                current_tip_step_mm = getattr(kf_ippe_tip, "last_tip_step_mm", None)
                current_rot_step_deg = getattr(kf_ippe_tip, "last_rot_step_deg", None)

            except Exception as e:
                ippe_error = str(e)

            # --------------------------------------------------
            # Angle relative to iterative reference
            # --------------------------------------------------
            if result_iter is not None:
                R_iter = result_iter.T_4x4[:3, :3]
                if R_ref_iter is None:
                    current_angle_deg = 0.0
                else:
                    R_rel = R_ref_iter.T @ R_iter
                    current_angle_deg = rotation_angle_deg(R_rel)

            # --------------------------------------------------
            # Recording
            # --------------------------------------------------
            valid_for_record = (
                recording
                and result_iter is not None
                and depth_iter is not None
                and result_ippe is not None
                and tip_ippe_kf_xyz_c is not None
                and current_dx_ippe_kf_mm is not None
                and tip_uv_ippe_kf is not None
                and current_angle_deg is not None
            )

            if valid_for_record:
                t_now = time.perf_counter()
                if t0_record is None:
                    t0_record = t_now

                times_s.append(float(t_now - t0_record))
                frame_idx_list.append(int(frame_idx))
                angle_deg_list.append(float(current_angle_deg))

                tip_iter_xyz_c = np.asarray(
                    result_iter.tip_point_camera_mm,
                    dtype=np.float64,
                ).reshape(3)
                iter_tip_xyz_c_list.append(tip_iter_xyz_c.tolist())
                iter_tip_uv_list.append(np.asarray(result_iter.tip_uv, dtype=np.float64).reshape(2).tolist())
                iter_dx_list.append(float(depth_iter.d_x_mm))
                iter_reproj_mean_list.append(float(result_iter.reproj_mean_px))

                ippe_kf_tip_xyz_c_list.append(tip_ippe_kf_xyz_c.tolist())
                ippe_kf_tip_uv_list.append(np.asarray(tip_uv_ippe_kf, dtype=np.float64).reshape(2).tolist())
                ippe_kf_dx_list.append(float(current_dx_ippe_kf_mm))

                ippe_reproj_mean_list.append(float(result_ippe.reproj_mean_px))
                ippe_reproj_median_list.append(float(result_ippe.reproj_median_px))
                ippe_reproj_max_list.append(float(result_ippe.reproj_max_px))
                ippe_solution_index_list.append(int(result_ippe.ippe_solution_index))
                n_markers_list.append(int(len(result_ippe.marker_ids_used)))

                kf_motion_score_list.append(float(current_motion_score_ippe_kf) if current_motion_score_ippe_kf is not None else np.nan)
                kf_tip_step_mm_list.append(float(current_tip_step_mm) if current_tip_step_mm is not None else np.nan)
                kf_rot_step_deg_list.append(float(current_rot_step_deg) if current_rot_step_deg is not None else np.nan)

            # --------------------------------------------------
            # Visualization
            # --------------------------------------------------
            if tip_uv_iter is not None:
                draw_tip_marker(
                    img_bgr,
                    tip_uv_iter,
                    color=(0, 0, 255),
                    marker_type=cv2.MARKER_CROSS,
                    label="iter",
                )

            if tip_uv_ippe_kf is not None:
                draw_tip_marker(
                    img_bgr,
                    tip_uv_ippe_kf,
                    color=(255, 0, 255),
                    marker_type=cv2.MARKER_CROSS,
                    label="ippe_kf",
                )

            status_lines = [
                f"frame: {frame_idx}",
                f"recording: {'ON' if recording else 'OFF'}",
                f"saved samples: {len(times_s)}",
                f"angle rel. reference: {current_angle_deg:.2f} deg" if current_angle_deg is not None else "angle rel. reference: n/a",
            ]

            if result_iter is not None and current_dx_iter_mm is not None:
                status_lines.append(
                    f"iter+refine      d_x {current_dx_iter_mm:.2f} mm   reproj {result_iter.reproj_mean_px:.2f}px"
                )
            else:
                status_lines.append(f"iter+refine      FAILED {iter_error}")

            if result_ippe is not None and current_dx_ippe_kf_mm is not None:
                status_lines.append(
                    f"ippe+refine+kf   d_x {current_dx_ippe_kf_mm:.2f} mm   reproj {result_ippe.reproj_mean_px:.2f}px   idx {result_ippe.ippe_solution_index}"
                )
            else:
                status_lines.append(f"ippe+refine+kf   FAILED {ippe_error}")

            if current_motion_score_ippe_kf is not None:
                status_lines.append(
                    f"kf motion {current_motion_score_ippe_kf:.2f}   tip_step {current_tip_step_mm:.3f} mm   rot_step {current_rot_step_deg:.3f} deg"
                )
            else:
                status_lines.append("kf motion n/a")

            status_lines.append("SPACE = start recording   q/ESC = save and quit")

            vis = draw_text_lines(img_bgr, status_lines)
            cv2.imshow("debug_depth_eval", vis)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                if SAVE_RESULTS:
                    save_recording(
                        save_dir=SAVE_DIR,
                        times_s=times_s,
                        frame_idx_list=frame_idx_list,
                        angle_deg_list=angle_deg_list,
                        iter_tip_xyz_c_list=iter_tip_xyz_c_list,
                        iter_tip_uv_list=iter_tip_uv_list,
                        iter_dx_list=iter_dx_list,
                        iter_reproj_mean_list=iter_reproj_mean_list,
                        ippe_kf_tip_xyz_c_list=ippe_kf_tip_xyz_c_list,
                        ippe_kf_tip_uv_list=ippe_kf_tip_uv_list,
                        ippe_kf_dx_list=ippe_kf_dx_list,
                        ippe_reproj_mean_list=ippe_reproj_mean_list,
                        ippe_reproj_median_list=ippe_reproj_median_list,
                        ippe_reproj_max_list=ippe_reproj_max_list,
                        ippe_solution_index_list=ippe_solution_index_list,
                        n_markers_list=n_markers_list,
                        kf_motion_score_list=kf_motion_score_list,
                        kf_tip_step_mm_list=kf_tip_step_mm_list,
                        kf_rot_step_deg_list=kf_rot_step_deg_list,
                        K_rgb=K_rgb,
                        T_cx=T_cx,
                        T_xc=T_xc,
                    )
                break

            elif key == ord(" "):
                if result_iter is None:
                    print("[SPACE] cannot start recording: iterative tracking invalid.")
                    continue

                R_ref_iter = result_iter.T_4x4[:3, :3].copy()

                times_s.clear()
                frame_idx_list.clear()
                angle_deg_list.clear()

                iter_tip_xyz_c_list.clear()
                iter_tip_uv_list.clear()
                iter_dx_list.clear()
                iter_reproj_mean_list.clear()

                ippe_kf_tip_xyz_c_list.clear()
                ippe_kf_tip_uv_list.clear()
                ippe_kf_dx_list.clear()

                ippe_reproj_mean_list.clear()
                ippe_reproj_median_list.clear()
                ippe_reproj_max_list.clear()
                ippe_solution_index_list.clear()
                n_markers_list.clear()

                kf_motion_score_list.clear()
                kf_tip_step_mm_list.clear()
                kf_rot_step_deg_list.clear()

                kf_ippe_tip.reset()
                t0_record = None
                recording = True

                print("[SPACE] reference set, buffers cleared, KF reset, recording started.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()