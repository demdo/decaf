# -*- coding: utf-8 -*-
"""
debug_xray_correspondences.py
=============================

Debug-Skript für globale Regularisierung der X-ray-Korrespondenzen.

Ablauf
------
1) X-ray RAW-Bild via Qt wählen
2) Intrinsics-NPZ wählen
3) Marker-Detection auf RAW
4) 3 Anchor-Punkte direkt auf RAW wählen
5) ROI direkt in RAW-Koordinaten berechnen
6) Auf Basis der diskreten Grid-Indizes:
   - raw ROI-Punkte
   - affine regularisierte Punkte
   - optional homography-regularisierte Punkte
7) Für jede Variante T_bx über tracking.pose_solvers.solve_pose(...)
   mit pose_method="ippe" und use_xray_ippe_selection_rule=True schätzen
8) Reprojection-Fehler, Pose-Differenzen und Visualisierung vergleichen

Wichtige Anzeige
----------------
Es werden SEPARATE Fenster geöffnet für:
- RAW points
- AFFINE points
- HOMOGRAPHY points (optional)
- RAW vs AFFINE displacement
- RAW vs HOMOGRAPHY displacement (optional)
- Reprojection RAW
- Reprojection AFFINE
- Reprojection HOMOGRAPHY (optional)

Steuerung
---------
Linksklick  = Anchor wählen
R           = Reset
Q / ESC     = Beenden
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
import pydicom
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.blob_detection import HoughCircleParams
from overlay.tools.xray_marker_selection import (
    run_xray_marker_detection,
    compute_roi_from_grid,
)
from overlay.tools.homography import build_board_xyz_canonical
from tracking.pose_solvers import solve_pose


# ============================================================
# Konfiguration
# ============================================================

OUT_NPZ_DEFAULT = (
    r"C:\Users\domin\Documents\Studium\Master\Masterarbeit\Projekt\Data\debug_xray_correspondences.npz"
)

HOUGH_PARAMS = HoughCircleParams(
    min_radius=2,
    max_radius=7,
    dp=1.2,
    minDist=8,
    param1=120,
    param2=9,
    invert=True,
    median_ks=(3, 5),
)

DISPLAY_MAX_W = 1200
DISPLAY_MAX_H = 1000

PITCH_MM = 2.54
STEPS_PER_EDGE = 10  # 11x11 -> 121 Punkte
USE_HOMOGRAPHY_MODEL = True

RAW_COLOR = (0, 255, 255)       # gelb
AFF_COLOR = (255, 255, 0)       # cyan
H_COLOR = (0, 255, 0)           # grün


# ============================================================
# Qt helpers
# ============================================================

def pick_open_file_qt(title: str, filt: str) -> str | None:
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    path, _ = QFileDialog.getOpenFileName(None, title, "", filt)

    if created_app:
        app.quit()

    path = path.strip()
    return path if path else None


def pick_save_file_qt(default_path: str) -> str | None:
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    path, _ = QFileDialog.getSaveFileName(
        None,
        "Debug-NPZ speichern unter",
        default_path,
        "NumPy NPZ (*.npz)",
    )

    if created_app:
        app.quit()

    path = path.strip()
    if not path:
        return None
    if not path.lower().endswith(".npz"):
        path += ".npz"
    return path


# ============================================================
# IO helpers
# ============================================================

def load_xray(path: str) -> np.ndarray:
    if path.lower().endswith((".dcm", ".ima")):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img.astype(np.uint8)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {path}")
    return img


def load_intrinsics_npz(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)

    for key in ("K_xray", "K_x", "K"):
        if key in data:
            return np.asarray(data[key], dtype=np.float64).reshape(3, 3)

    raise KeyError(
        f"Intrinsics-NPZ muss einen der Keys 'K_xray', 'K_x' oder 'K' enthalten. "
        f"Vorhandene Keys: {list(data.keys())}"
    )


# ============================================================
# Geometrie / Anzeige
# ============================================================

def nearest_circle_index(circles: np.ndarray, u_click: float, v_click: float) -> int | None:
    if circles is None or len(circles) == 0:
        return None

    xy = circles[:, :2].astype(np.float64)
    finite = np.isfinite(xy).all(axis=1)
    if not np.any(finite):
        return None

    xyf = xy[finite]
    idx_map = np.flatnonzero(finite)

    d2 = (xyf[:, 0] - float(u_click)) ** 2 + (xyf[:, 1] - float(v_click)) ** 2
    return int(idx_map[np.argmin(d2)])


def make_display_transform(img_shape: tuple[int, int], max_w: int, max_h: int):
    h, w = img_shape[:2]
    scale = min(max_w / float(w), max_h / float(h))
    scale = max(scale, 1e-12)

    disp_w = max(1, int(round(w * scale)))
    disp_h = max(1, int(round(h * scale)))

    off_x = max(0, (max_w - disp_w) // 2)
    off_y = max(0, (max_h - disp_h) // 2)

    return scale, disp_w, disp_h, off_x, off_y


def render_for_display(img_bgr: np.ndarray, max_w: int, max_h: int):
    scale, disp_w, disp_h, off_x, off_y = make_display_transform(img_bgr.shape, max_w, max_h)

    if abs(scale - 1.0) < 1e-12:
        resized = img_bgr.copy()
    else:
        resized = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    canvas[off_y:off_y + disp_h, off_x:off_x + disp_w] = resized
    return canvas, scale, off_x, off_y


def map_display_to_raw(x_disp: int, y_disp: int, raw_shape, scale: float, off_x: int, off_y: int):
    h, w = raw_shape[:2]

    x_rel = x_disp - off_x
    y_rel = y_disp - off_y

    disp_w = int(round(w * scale))
    disp_h = int(round(h * scale))

    if x_rel < 0 or y_rel < 0 or x_rel >= disp_w or y_rel >= disp_h:
        return None

    u = x_rel / scale
    v = y_rel / scale
    return float(u), float(v)


def draw_cross(img, u, v, color, size=8, thickness=1):
    uu = int(round(float(u)))
    vv = int(round(float(v)))
    cv2.line(img, (uu - size, vv), (uu + size, vv), color, thickness, cv2.LINE_AA)
    cv2.line(img, (uu, vv - size), (uu, vv + size), color, thickness, cv2.LINE_AA)


# ============================================================
# Grid-Regularisierung
# ============================================================

@dataclass(frozen=True)
class AffineGridModel:
    p0: np.ndarray  # (2,)
    u: np.ndarray   # (2,)
    v: np.ndarray   # (2,)

    def predict(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        i = np.asarray(i, dtype=np.float64).reshape(-1)
        j = np.asarray(j, dtype=np.float64).reshape(-1)
        return self.p0[None, :] + j[:, None] * self.u[None, :] + i[:, None] * self.v[None, :]


def fit_affine_grid(points_uv: np.ndarray, grid_i: np.ndarray, grid_j: np.ndarray) -> AffineGridModel:
    uv = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    i = np.asarray(grid_i, dtype=np.float64).reshape(-1)
    j = np.asarray(grid_j, dtype=np.float64).reshape(-1)

    if not (len(uv) == len(i) == len(j)):
        raise ValueError("points_uv, grid_i, grid_j must have same length.")

    n = len(uv)
    A = np.zeros((2 * n, 6), dtype=np.float64)
    b = np.zeros((2 * n,), dtype=np.float64)

    # x = p0x + j*ux + i*vx
    A[0::2, 0] = 1.0
    A[0::2, 2] = j
    A[0::2, 4] = i
    b[0::2] = uv[:, 0]

    # y = p0y + j*uy + i*vy
    A[1::2, 1] = 1.0
    A[1::2, 3] = j
    A[1::2, 5] = i
    b[1::2] = uv[:, 1]

    x, *_ = np.linalg.lstsq(A, b, rcond=None)

    return AffineGridModel(
        p0=np.array([x[0], x[1]], dtype=np.float64),
        u=np.array([x[2], x[3]], dtype=np.float64),
        v=np.array([x[4], x[5]], dtype=np.float64),
    )


def fit_homography_grid(points_uv: np.ndarray, grid_i: np.ndarray, grid_j: np.ndarray) -> np.ndarray:
    src = np.stack([grid_j, grid_i], axis=1).astype(np.float64)
    dst = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("Homography-Fit fehlgeschlagen.")
    return H


def apply_homography_grid(H: np.ndarray, grid_i: np.ndarray, grid_j: np.ndarray) -> np.ndarray:
    src = np.stack([grid_j, grid_i], axis=1).astype(np.float64).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(src, np.asarray(H, dtype=np.float64))
    return dst.reshape(-1, 2)


# ============================================================
# Pose / Fehler
# ============================================================

def transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def rotation_angle_deg(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def relative_pose_metrics(T_ref: np.ndarray, T_test: np.ndarray) -> dict:
    R_ref = T_ref[:3, :3]
    t_ref = T_ref[:3, 3]
    R_test = T_test[:3, :3]
    t_test = T_test[:3, 3]

    R_rel = R_ref.T @ R_test
    return {
        "delta_R_deg": rotation_angle_deg(R_rel),
        "delta_t_mm": float(np.linalg.norm(t_test - t_ref)),
    }


def solve_T_bx(points_uv: np.ndarray, K_xray: np.ndarray):
    pts3d_board_mm = build_board_xyz_canonical(
        nu=STEPS_PER_EDGE,
        nv=STEPS_PER_EDGE,
        pitch_mm=PITCH_MM,
    )

    pose = solve_pose(
        object_points_xyz=pts3d_board_mm,
        image_points_uv=np.asarray(points_uv, dtype=np.float64).reshape(-1, 2),
        K=np.asarray(K_xray, dtype=np.float64).reshape(3, 3),
        dist_coeffs=None,
        pose_method="ippe",
        refine_with_iterative=True,
        use_xray_ippe_selection_rule=True,
    )

    T_bx = transform_from_rvec_tvec(pose.rvec, pose.tvec)
    return pose, T_bx


# ============================================================
# Visualisierung
# ============================================================

def draw_points(img_gray: np.ndarray, points_uv: np.ndarray, color, radius=4, labels=None, title: str | None = None):
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for k, (u, v) in enumerate(points_uv):
        cv2.circle(vis, (int(round(u)), int(round(v))), radius, color, -1, cv2.LINE_AA)
        if labels is not None:
            cv2.putText(
                vis,
                str(labels[k]),
                (int(round(u)) + 3, int(round(v)) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
                cv2.LINE_AA,
            )

    if title is not None:
        cv2.putText(vis, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


def draw_displacements(img_gray: np.ndarray, uv_a: np.ndarray, uv_b: np.ndarray, title: str | None = None):
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for p, q in zip(uv_a, uv_b):
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 3, RAW_COLOR, -1, cv2.LINE_AA)
        cv2.circle(vis, (int(round(q[0])), int(round(q[1]))), 3, AFF_COLOR, -1, cv2.LINE_AA)
        cv2.line(
            vis,
            (int(round(p[0])), int(round(p[1]))),
            (int(round(q[0])), int(round(q[1]))),
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    if title is not None:
        cv2.putText(vis, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


def draw_reprojection(img_gray: np.ndarray, uv_obs: np.ndarray, uv_proj: np.ndarray, title: str | None = None):
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for p, q in zip(uv_obs, uv_proj):
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 3, (0, 255, 255), -1, cv2.LINE_AA)
        draw_cross(vis, q[0], q[1], color=(0, 0, 255), size=6, thickness=1)
        cv2.line(
            vis,
            (int(round(p[0])), int(round(p[1]))),
            (int(round(q[0])), int(round(q[1]))),
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    if title is not None:
        cv2.putText(vis, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


def close_debug_windows():
    names = [
        "RAW points",
        "AFFINE points",
        "HOMOGRAPHY points",
        "RAW vs AFFINE displacement",
        "RAW vs HOMOGRAPHY displacement",
        "Reprojection RAW",
        "Reprojection AFFINE",
        "Reprojection HOMOGRAPHY",
    ]
    for name in names:
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass


def show_debug_windows(
    img_raw: np.ndarray,
    roi_uv_raw: np.ndarray,
    roi_uv_aff: np.ndarray,
    roi_uv_h: np.ndarray | None,
    pose_raw,
    pose_aff,
    pose_h,
):
    vis_raw = draw_points(img_raw, roi_uv_raw, RAW_COLOR, title="RAW points")
    cv2.imshow("RAW points", vis_raw)

    vis_aff = draw_points(img_raw, roi_uv_aff, AFF_COLOR, title="AFFINE points")
    cv2.imshow("AFFINE points", vis_aff)

    vis_disp_aff = draw_displacements(
        img_raw,
        roi_uv_raw,
        roi_uv_aff,
        title="RAW vs AFFINE displacement",
    )
    cv2.imshow("RAW vs AFFINE displacement", vis_disp_aff)

    vis_repr_raw = draw_reprojection(
        img_raw,
        roi_uv_raw,
        pose_raw.uv_proj,
        title="Reprojection RAW",
    )
    cv2.imshow("Reprojection RAW", vis_repr_raw)

    vis_repr_aff = draw_reprojection(
        img_raw,
        roi_uv_aff,
        pose_aff.uv_proj,
        title="Reprojection AFFINE",
    )
    cv2.imshow("Reprojection AFFINE", vis_repr_aff)

    if roi_uv_h is not None and pose_h is not None:
        vis_h = draw_points(img_raw, roi_uv_h, H_COLOR, title="HOMOGRAPHY points")
        cv2.imshow("HOMOGRAPHY points", vis_h)

        vis_disp_h = draw_displacements(
            img_raw,
            roi_uv_raw,
            roi_uv_h,
            title="RAW vs HOMOGRAPHY displacement",
        )
        cv2.imshow("RAW vs HOMOGRAPHY displacement", vis_disp_h)

        vis_repr_h = draw_reprojection(
            img_raw,
            roi_uv_h,
            pose_h.uv_proj,
            title="Reprojection HOMOGRAPHY",
        )
        cv2.imshow("Reprojection HOMOGRAPHY", vis_repr_h)
    else:
        for name in [
            "HOMOGRAPHY points",
            "RAW vs HOMOGRAPHY displacement",
            "Reprojection HOMOGRAPHY",
        ]:
            try:
                cv2.destroyWindow(name)
            except cv2.error:
                pass


def arrange_debug_windows():
    try:
        cv2.moveWindow("RAW points", 40, 40)
        cv2.moveWindow("AFFINE points", 720, 40)
        cv2.moveWindow("HOMOGRAPHY points", 1400, 40)

        cv2.moveWindow("RAW vs AFFINE displacement", 40, 640)
        cv2.moveWindow("RAW vs HOMOGRAPHY displacement", 720, 640)

        cv2.moveWindow("Reprojection RAW", 1400, 640)
        cv2.moveWindow("Reprojection AFFINE", 2080, 40)
        cv2.moveWindow("Reprojection HOMOGRAPHY", 2080, 640)
    except cv2.error:
        pass


# ============================================================
# State drawing for main window
# ============================================================

def draw_main_state(
    img_gray: np.ndarray,
    circles: np.ndarray,
    anchors: list[int],
    roi_uv_raw: np.ndarray | None,
    *,
    title: str,
) -> np.ndarray:
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for (x, y, r) in circles:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(
                vis,
                (int(round(x)), int(round(y))),
                max(2, int(round(r))),
                (120, 120, 120),
                1,
            )

    anchor_labels = ["TL", "TR", "BL"]
    for k, idx in enumerate(anchors):
        x, y, r = circles[idx]
        label = anchor_labels[k]
        cv2.circle(
            vis,
            (int(round(x)), int(round(y))),
            max(4, int(round(r)) + 2),
            (0, 140, 255),
            2,
        )
        cv2.putText(
            vis,
            label,
            (int(round(x)) + 6, int(round(y)) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 140, 255),
            2,
            cv2.LINE_AA,
        )

    if roi_uv_raw is not None:
        for u, v in roi_uv_raw:
            cv2.circle(vis, (int(round(u)), int(round(v))), 4, RAW_COLOR, -1, cv2.LINE_AA)

    cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        f"Anchors: {len(anchors)}/3  |  LMB=select  R=reset  Q=quit",
        (10, vis.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return vis


# ============================================================
# Main
# ============================================================

def main():
    xray_path = pick_open_file_qt(
        "X-ray Bild wählen",
        "X-ray / DICOM (*.dcm *.ima *.png *.jpg *.jpeg *.tif *.tiff *.bmp);;Alle Dateien (*)",
    )
    if not xray_path:
        print("[INFO] Kein X-ray Bild gewählt. Abbruch.")
        return

    intr_path = pick_open_file_qt(
        "Intrinsics-NPZ wählen",
        "NumPy archive (*.npz);;Alle Dateien (*)",
    )
    if not intr_path:
        print("[INFO] Kein Intrinsics-NPZ gewählt. Abbruch.")
        return

    out_path = pick_save_file_qt(OUT_NPZ_DEFAULT)
    if not out_path:
        print("[INFO] Kein Speicherpfad gewählt. Abbruch.")
        return

    img_raw = load_xray(xray_path)
    K_xray = load_intrinsics_npz(intr_path)

    print(f"Lade X-ray: {xray_path}")
    print(f"Lade K_xray: {intr_path}")

    res = run_xray_marker_detection(
        img_raw,
        hough_params=HOUGH_PARAMS,
        use_clahe=True,
        clahe_clip=2.0,
        clahe_tiles=(12, 12),
        use_mask=False,
    )

    if res.circles is None or len(res.circles) == 0:
        print("[ERR] Keine Kreise detektiert.")
        sys.exit(1)

    circles_raw = np.asarray(res.circles, dtype=np.float64).reshape(-1, 3)
    print(f"Detektiert: {len(circles_raw)} Kreise")

    radii = circles_raw[:, 2]
    finite_r = radii[np.isfinite(radii)]
    pick_r = 0.6 * float(np.median(finite_r)) if finite_r.size else 20.0

    win_title = f"debug_xray_correspondences - {Path(xray_path).name}"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    anchors: list[int] = []

    roi_uv_raw: np.ndarray | None = None
    roi_uv_aff: np.ndarray | None = None
    roi_uv_h: np.ndarray | None = None
    dbg: dict | None = None

    pose_raw = None
    pose_aff = None
    pose_h = None
    T_bx_raw = None
    T_bx_aff = None
    T_bx_h = None

    display_state = {"scale": 1.0, "off_x": 0, "off_y": 0}

    def on_click(event, x, y, flags, param):
        nonlocal anchors
        nonlocal roi_uv_raw, roi_uv_aff, roi_uv_h, dbg
        nonlocal pose_raw, pose_aff, pose_h
        nonlocal T_bx_raw, T_bx_aff, T_bx_h

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(anchors) >= 3:
            return

        mapped = map_display_to_raw(
            x_disp=x,
            y_disp=y,
            raw_shape=img_raw.shape,
            scale=float(display_state["scale"]),
            off_x=int(display_state["off_x"]),
            off_y=int(display_state["off_y"]),
        )
        if mapped is None:
            return

        u_raw, v_raw = mapped
        nearest = nearest_circle_index(circles_raw, u_raw, v_raw)
        if nearest is None:
            return

        d = np.linalg.norm(circles_raw[nearest, :2] - np.array([u_raw, v_raw], dtype=np.float64))
        if d > pick_r * 3.0:
            return
        if nearest in anchors:
            return

        anchors.append(nearest)
        print(
            f"Anchor {len(anchors)}: idx={nearest}  "
            f"uv_raw=({circles_raw[nearest,0]:.1f}, {circles_raw[nearest,1]:.1f})"
        )

        if len(anchors) == 3:
            try:
                roi_uv_raw_, roi_idx, dbg_ = compute_roi_from_grid(
                    circles=circles_raw,
                    anchor_idx=anchors,
                    margin_px=1.1 * pick_r,
                    gate_tol_pitch=0.40,
                    min_steps=2,
                )

                roi_uv_raw = np.asarray(roi_uv_raw_, dtype=np.float64).reshape(-1, 2)
                dbg = dbg_

                grid_i = np.asarray(dbg["grid_i"], dtype=np.int32)
                grid_j = np.asarray(dbg["grid_j"], dtype=np.int32)

                aff = fit_affine_grid(roi_uv_raw, grid_i, grid_j)
                roi_uv_aff = aff.predict(grid_i, grid_j)

                if USE_HOMOGRAPHY_MODEL:
                    H = fit_homography_grid(roi_uv_raw, grid_i, grid_j)
                    roi_uv_h = apply_homography_grid(H, grid_i, grid_j)
                else:
                    H = None
                    roi_uv_h = None

                pose_raw, T_bx_raw = solve_T_bx(roi_uv_raw, K_xray)
                pose_aff, T_bx_aff = solve_T_bx(roi_uv_aff, K_xray)
                cmp_aff = relative_pose_metrics(T_bx_raw, T_bx_aff)

                if roi_uv_h is not None:
                    pose_h, T_bx_h = solve_T_bx(roi_uv_h, K_xray)
                    cmp_h = relative_pose_metrics(T_bx_raw, T_bx_h)
                else:
                    pose_h = None
                    T_bx_h = None
                    cmp_h = None

                d_aff = np.linalg.norm(roi_uv_aff - roi_uv_raw, axis=1)

                print("\n" + "=" * 80)
                print("DEBUG: X-RAY CORRESPONDENCE REGULARIZATION")
                print("=" * 80)
                print(f"N ROI points                  : {len(roi_uv_raw)}")
                print(f"nu / nv                       : {dbg['nu']} / {dbg['nv']}")
                print(f"pitch [px]                    : {dbg['pitch']:.6f}")

                print("\n[1] RAW -> AFFINE")
                print(f"mean displacement [px]        : {np.mean(d_aff):.6f}")
                print(f"median displacement [px]      : {np.median(d_aff):.6f}")
                print(f"max displacement [px]         : {np.max(d_aff):.6f}")

                print("\n[2] T_bx from RAW")
                print(f"candidate_index               : {pose_raw.candidate_index}")
                print(f"reproj mean [px]              : {pose_raw.reproj_mean_px:.6f}")
                print(f"reproj median [px]            : {pose_raw.reproj_median_px:.6f}")
                print(f"reproj max [px]               : {pose_raw.reproj_max_px:.6f}")
                print(f"t_bx [mm]                     : {pose_raw.tvec.reshape(-1)}")

                print("\n[3] T_bx from AFFINE")
                print(f"candidate_index               : {pose_aff.candidate_index}")
                print(f"reproj mean [px]              : {pose_aff.reproj_mean_px:.6f}")
                print(f"reproj median [px]            : {pose_aff.reproj_median_px:.6f}")
                print(f"reproj max [px]               : {pose_aff.reproj_max_px:.6f}")
                print(f"t_bx [mm]                     : {pose_aff.tvec.reshape(-1)}")
                print(f"delta_t vs raw [mm]           : {cmp_aff['delta_t_mm']:.6f}")
                print(f"delta_R vs raw [deg]          : {cmp_aff['delta_R_deg']:.6f}")

                if roi_uv_h is not None and pose_h is not None:
                    d_h = np.linalg.norm(roi_uv_h - roi_uv_raw, axis=1)

                    print("\n[4] RAW -> HOMOGRAPHY")
                    print(f"mean displacement [px]        : {np.mean(d_h):.6f}")
                    print(f"median displacement [px]      : {np.median(d_h):.6f}")
                    print(f"max displacement [px]         : {np.max(d_h):.6f}")

                    print("\n[5] T_bx from HOMOGRAPHY")
                    print(f"candidate_index               : {pose_h.candidate_index}")
                    print(f"reproj mean [px]              : {pose_h.reproj_mean_px:.6f}")
                    print(f"reproj median [px]            : {pose_h.reproj_median_px:.6f}")
                    print(f"reproj max [px]               : {pose_h.reproj_max_px:.6f}")
                    print(f"t_bx [mm]                     : {pose_h.tvec.reshape(-1)}")
                    print(f"delta_t vs raw [mm]           : {cmp_h['delta_t_mm']:.6f}")
                    print(f"delta_R vs raw [deg]          : {cmp_h['delta_R_deg']:.6f}")

                save_dict = dict(
                    xray_path=str(xray_path),
                    intrinsics_path=str(intr_path),
                    K_xray=K_xray,
                    points_uv_raw=roi_uv_raw,
                    points_uv_affine=roi_uv_aff,
                    grid_i=np.asarray(dbg["grid_i"], dtype=np.int32),
                    grid_j=np.asarray(dbg["grid_j"], dtype=np.int32),
                    T_bx_raw=T_bx_raw,
                    T_bx_affine=T_bx_aff,
                    rvec_raw=pose_raw.rvec,
                    tvec_raw=pose_raw.tvec,
                    rvec_affine=pose_aff.rvec,
                    tvec_affine=pose_aff.tvec,
                    reproj_err_raw=pose_raw.reproj_errors_px,
                    reproj_err_affine=pose_aff.reproj_errors_px,
                    uv_proj_raw=pose_raw.uv_proj,
                    uv_proj_affine=pose_aff.uv_proj,
                )

                if roi_uv_h is not None and pose_h is not None:
                    save_dict["points_uv_homography"] = roi_uv_h
                    save_dict["T_bx_homography"] = T_bx_h
                    save_dict["rvec_homography"] = pose_h.rvec
                    save_dict["tvec_homography"] = pose_h.tvec
                    save_dict["reproj_err_homography"] = pose_h.reproj_errors_px
                    save_dict["uv_proj_homography"] = pose_h.uv_proj

                np.savez(out_path, **save_dict)
                print(f"\n[OK] saved debug result -> {out_path}")

                show_debug_windows(
                    img_raw=img_raw,
                    roi_uv_raw=roi_uv_raw,
                    roi_uv_aff=roi_uv_aff,
                    roi_uv_h=roi_uv_h,
                    pose_raw=pose_raw,
                    pose_aff=pose_aff,
                    pose_h=pose_h,
                )
                arrange_debug_windows()

            except Exception as e:
                print(f"[ERR] Debug computation failed: {e}")
                roi_uv_raw = None
                roi_uv_aff = None
                roi_uv_h = None
                dbg = None
                pose_raw = None
                pose_aff = None
                pose_h = None
                T_bx_raw = None
                T_bx_aff = None
                T_bx_h = None
                close_debug_windows()

    cv2.setMouseCallback(win_title, on_click)

    print("\nLinksklick = 3 Anchors wählen | R = Reset | Q/ESC = Quit\n")

    while True:
        vis = draw_main_state(
            img_raw,
            circles_raw,
            anchors,
            roi_uv_raw,
            title="RAW view",
        )

        vis_disp, scale, off_x, off_y = render_for_display(vis, DISPLAY_MAX_W, DISPLAY_MAX_H)
        display_state["scale"] = scale
        display_state["off_x"] = off_x
        display_state["off_y"] = off_y

        cv2.imshow(win_title, vis_disp)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break

        if key in (ord("r"), ord("R")):
            anchors = []
            roi_uv_raw = None
            roi_uv_aff = None
            roi_uv_h = None
            dbg = None
            pose_raw = None
            pose_aff = None
            pose_h = None
            T_bx_raw = None
            T_bx_aff = None
            T_bx_h = None
            close_debug_windows()
            print("→ Reset.\n")

    close_debug_windows()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()