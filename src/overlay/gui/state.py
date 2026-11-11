# overlay/gui/state.py

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# Workflow from the start dialog (same strings as LivePanelsView.set_workflow):
# one AP X-ray, or lateral and AP X-ray each with a ChArUco board pose.
WORKFLOW_SINGLE = "single"
WORKFLOW_MULTI = "multi"


@dataclass(frozen=True)
class Cam2XHeight:
    """One finished board placement of the Cam2X series (see
    SessionState.cam2x_heights)."""

    # overlay.calib.calib_camera_to_xray.Cam2XBoardCapture (kept as object to
    # avoid a hard import here, same pattern as xray_dist_field below)
    capture: object
    T_cx_single: np.ndarray        # (4,4) this placement solved alone, metres (diagnostic)
    board_centre_c_mm: np.ndarray  # (3,) patch centre in the camera frame (step 3 check)
    xray_md5: str                  # overlay.tools.xray_io.xray_pixel_md5 (step 4 check)
    intrinsics_key: str = ""       # K_rgb, dist_rgb, X-ray field it was solved with
    K_xray: Optional[np.ndarray] = None  # K_xray of T_cx_single (not part of the capture)
    eval_label: str = ""           # file label of its captures in Evaluation mode ("cal_h2")
    eval_file: str = ""            # its <label>_cam2x.npz in Evaluation mode


@dataclass
class SessionState:
    # --- X-ray intrinsics ---
    K_xray: Optional[np.ndarray] = None

    # --- Camera calibration ---
    K_rgb: Optional[np.ndarray] = None
    dist_rgb: Optional[np.ndarray] = None
    # overlay.tools.xray_distortion.XrayFieldDistortion, or None.
    # There is deliberately no Brown-style dist_xray: those coefficients cannot
    # represent this detector's field, see that module.
    xray_dist_field: Optional[object] = None

    # --- X-ray image for calibration / marker selection ---
    xray_image: Optional[np.ndarray] = None
    xray_image_path: Optional[str] = None


    # --- X-ray marker selection (interactive -> confirmed) ---
    xray_points_uv: Optional[np.ndarray] = None
    xray_points_confirmed: bool = False
    xray_marker_overlay_bgr: Optional[np.ndarray] = None
    marker_radius_px: Optional[float] = None

    # --- Full board lattice for the X-ray-side pose (2026-09-15) ---
    # All indexed holes of the board, XRAY_WORKING_FLIPPED_UV, and their
    # board coordinates in mm relative to the patch origin (TL hole). The
    # Cam2X page refines the X-ray pose on these instead of the 121-point
    # patch; see tools.xray_marker_selection.index_board_lattice.
    xray_lattice_uv: Optional[np.ndarray] = None       # (M, 2)
    xray_lattice_xy_mm: Optional[np.ndarray] = None    # (M, 2)

    # --- Plane fitting result (interactive -> confirmed) ---
    xray_points_xyz_c: Optional[np.ndarray] = None
    plane_confirmed: bool = False

    # --- RGB checkerboard corners (for IPPE-based T_cx composition) ---
    checkerboard_corners_uv: Optional[np.ndarray] = None  # (3, 2): TL, TR, BL in RGB pixels
    checkerboard_corners_confirmed: bool = False

    # --- Full 3x3 checkerboard corners in RGB ---
    checkerboard_corners_uv_9: Optional[np.ndarray] = None  # (9, 2)

    # --- Cam2X over several board placements (table heights, 2026-09-22) ---
    # One Cam2XHeight per finished height, in capture order. The Cam2X page
    # pools them with the current height into T_cx / T_xc once all heights
    # are in; steps 3 and 4 check a new capture against them.
    cam2x_heights: list = field(default_factory=list)

    # --- Final transformations (homogeneous 4x4) ---
    # Convention:
    #   X_x = T_cx @ X_c   (Camera -> X-ray)
    #   X_c = T_xc @ X_x   (X-ray -> Camera)
    T_cx: Optional[np.ndarray] = None  # Camera -> X-ray
    T_xc: Optional[np.ndarray] = None  # X-ray -> Camera

    # --- Accepted pointer-tip snapshot from camera-to-pointer measurement ---
    tip_uv_c: Optional[np.ndarray] = None   # (2,) tip position in RGB image [px]
    tip_xyz_c: Optional[np.ndarray] = None  # (3,) tip position in camera frame [mm]

    # --- Pointer pose diagnostics (for debugging d_x issues) ---
    T_tc: Optional[np.ndarray] = None  # tip -> camera

    # --- Distance from X-ray source to target plane used in plane-induced homography ---
    d_x: Optional[float] = None

    # --- Plane-induced homography for X-ray -> camera overlay ---
    H_xc: Optional[np.ndarray] = None
    
    # --- Anatomical X-ray image for preview / live overlay ---
    xray_image_anatomy: Optional[np.ndarray] = None
    xray_image_anatomy_path: Optional[str] = None

    # --- Tool tracking (optional; loaded on the Tool Tracking page) ---
    # tracking.hydramarker.tool_assets.ToolAssets (kept as object to avoid a
    # hard import here, same pattern as xray_dist_field above).
    tool_assets: Optional[object] = None
    # IR rig calibration npz for stereo-IR pose refinement, or None (RGB only).
    ir_calibration_path: Optional[str] = None

    # --- Evaluation mode ---
    # overlay.gui.evaluation.EvaluationSession, or None in Normal mode. The
    # pages write to disk only through this object (kept as object to avoid
    # a Qt import here).
    eval: Optional[object] = None

    # --- Workflow (start dialog): WORKFLOW_SINGLE or WORKFLOW_MULTI ---
    workflow: str = WORKFLOW_SINGLE

    # ----------------
    # Convenience flags
    # ----------------
    @property
    def is_multi_view(self) -> bool:
        return self.workflow == WORKFLOW_MULTI

    @property
    def has_rgb_intrinsics(self) -> bool:
        return self.K_rgb is not None

    @property
    def has_xray_intrinsics(self) -> bool:
        return self.K_xray is not None

    @property
    def has_xray_image(self) -> bool:
        return self.xray_image is not None

    @property
    def has_xray_image_anatomy(self) -> bool:
        return self.xray_image_anatomy is not None

    @property
    def has_xray_points(self) -> bool:
        return self.xray_points_uv is not None and len(self.xray_points_uv) > 0

    @property
    def has_xray_points_confirmed(self) -> bool:
        return self.xray_points_confirmed and self.has_xray_points

    @property
    def has_xray_points_xyz(self) -> bool:
        return self.xray_points_xyz_c is not None and len(self.xray_points_xyz_c) > 0

    @property
    def has_plane_confirmed(self) -> bool:
        return self.plane_confirmed and self.has_xray_points_xyz

    @property
    def has_checkerboard_corners(self) -> bool:
        return self.checkerboard_corners_uv is not None and self.checkerboard_corners_confirmed

    @property
    def has_cam_to_xray(self) -> bool:
        return self.T_cx is not None

    @property
    def has_xray_to_cam(self) -> bool:
        return self.T_xc is not None

    @property
    def has_d_x(self) -> bool:
        return self.d_x is not None

    @property
    def has_H_xc(self) -> bool:
        return self.H_xc is not None

    @property
    def has_tool(self) -> bool:
        return self.tool_assets is not None