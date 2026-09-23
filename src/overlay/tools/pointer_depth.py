# overlay/tools/pointer_depth.py
#
# Pointer tip -> reference-plane depth d_x, shared by the Pointer Tool Depth
# page and the "Set reference plane" button of the live overlay page.
#
# One place for the pose solve, the Kalman filter on the tip and the arrays
# of the <label>_depth.npz evaluation capture, so a plane set on the live
# page is measured and stored exactly like one set on the wizard page.

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from overlay.calib.calib_camera_to_pointer import (
    calibrate_camera_to_pointer,
    get_default_pointer_tool_model,
)
from overlay.calib.calib_xray_to_pointer import extract_depth
from tracking.pose_filters import AdaptiveKalmanFilterCV3D


class PointerDepthEstimator:
    """
    Per-frame pointer solve with a Kalman filter on the tip position.

    solve(img) returns (raw, filtered) CameraToPointerResult or raises when
    the pointer is not found. The filter state carries over between calls,
    so reset() before every new measurement.
    """

    def __init__(self, fps: float):
        self.pointer_model = get_default_pointer_tool_model()
        # Match the debug script 1:1
        self._tip_kf = AdaptiveKalmanFilterCV3D(
            dt=1.0 / float(fps),
            q_pos_still=1e-4,
            q_vel_still=1e-2,
            r_still=8e-2,
            q_pos_move=5e-3,
            q_vel_move=3e-1,
            r_move=2e-2,
        )

    def reset(self) -> None:
        self._tip_kf.reset()

    def solve(self, image: np.ndarray, K_rgb, dist_rgb):
        """image: BGR, BGRA or gray (the ArUco detection converts itself)."""
        # dist_rgb: the ArUco corners are observed in the distorted frame;
        # solving with the lens model gives T_cp against the ideal camera,
        # the frame T_xc is defined in.
        raw = calibrate_camera_to_pointer(
            image_bgr=image,
            camera_intrinsics=K_rgb,
            dist_coeffs=dist_rgb,
            pointer_model=self.pointer_model,
            rvec_init=None,
            tvec_init=None,
            use_extrinsic_guess=False,
            pose_method="ippe",
            refine_with_iterative=True,
        )
        return raw, self._filter(raw, K_rgb, dist_rgb)

    def _filter(self, raw_result, K_rgb, dist_rgb):
        """
        Apply Kalman filtering to the tip position only.
        Rotation stays raw; it is only used to adapt the filter noise.
        """
        tip_xyz_f_mm = self._tip_kf.filter(
            measurement_mm=raw_result.tip_point_camera_mm,
            rotation_camera=raw_result.rotation,
        ).reshape(3)

        T_tc_f = np.asarray(raw_result.T_4x4, dtype=np.float64).copy()
        T_tc_f[:3, 3] = tip_xyz_f_mm

        tvec_f = tip_xyz_f_mm.reshape(3, 1)

        # Project WITH the lens model: tip_uv_c is drawn on the observed
        # (distorted) frame, like the tracker's tip and the warped X-ray.
        tip_uv_f, _ = cv2.projectPoints(
            np.zeros((1, 3), dtype=np.float64),  # tip origin in tip frame
            raw_result.rvec,
            tvec_f,
            np.asarray(K_rgb, dtype=np.float64),
            dist_rgb,
        )

        return replace(
            raw_result,
            tvec=tvec_f,
            translation=tvec_f.copy(),
            T_4x4=T_tc_f,
            tip_point_camera_mm=tip_xyz_f_mm,
            tip_uv=tip_uv_f.reshape(2),
        )


def depth_capture_arrays(
    *,
    rgb_image: np.ndarray | None,
    K_rgb,
    dist_rgb,
    T_xc,
    result,
    raw_result,
    d_x_mm: float,
) -> tuple[dict, dict]:
    """
    Arrays and log summary of a <label>_depth.npz capture: the accepted
    pointer frame with everything behind d_x - detected corners and ids (so
    the pose can be re-solved offline, e.g. without the lens model), the raw
    and the filtered tip, T_tc, T_xc, K_rgb, dist_rgb.
    """
    res, raw = result, raw_result

    def opt(a, dtype=np.float64):
        return None if a is None else np.asarray(a, dtype=dtype)

    arrays = dict(
        rgb_image=rgb_image,
        K_rgb=opt(K_rgb),
        dist_rgb=opt(dist_rgb),
        T_xc=opt(T_xc),
        # pointer solve (raw = unfiltered, as it left calibrate_camera_to_pointer)
        marker_ids_used=opt(getattr(res, "marker_ids_used", None), np.int64),
        object_points_xyz_mm=opt(getattr(res, "object_points_xyz", None)),
        image_points_uv=opt(getattr(res, "image_points_uv", None)),
        rvec=opt(getattr(res, "rvec", None)),
        tvec_raw=opt(getattr(raw, "tvec", None)),
        tvec_filtered=opt(getattr(res, "tvec", None)),
        reproj_errors_px=opt(getattr(res, "reproj_errors_px", None)),
        reproj_mean_px=np.array(float(getattr(res, "reproj_mean_px", np.nan))),
        pose_method=np.array(str(getattr(res, "pose_method", ""))),
        tip_xyz_c_raw_mm=opt(getattr(raw, "tip_point_camera_mm", None)),
        tip_xyz_c_filtered_mm=opt(getattr(res, "tip_point_camera_mm", None)),
        tip_uv_c=opt(getattr(res, "tip_uv", None)),
        T_tc=opt(getattr(res, "T_4x4", None)),
        T_tc_raw=opt(getattr(raw, "T_4x4", None)),
        d_x_mm=np.array(float(d_x_mm)),
    )
    d_x_raw = np.nan
    try:
        if raw is not None:
            d_x_raw = float(extract_depth(T_xc=T_xc, T_tc=raw.T_4x4).d_x_mm)
    except Exception:
        pass
    arrays["d_x_raw_mm"] = np.array(d_x_raw)

    summary = {
        "d_x_mm": float(d_x_mm),
        "d_x_raw_mm": d_x_raw,
        "reproj_px": float(getattr(res, "reproj_mean_px", np.nan)),
    }
    return arrays, summary
