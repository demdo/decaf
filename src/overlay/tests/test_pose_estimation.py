# -*- coding: utf-8 -*-
"""
pose_estimation_drift_comparison.py

Compare pose drift across repeated snapshots of the SAME board position for
different pose estimation methods:

    1) RANSAC-PnP
    2) IPPE
    3) Homography-based pose composition

Assumptions
-----------
- All snapshots show the same physical board pose.
- The same uv_xray correspondences are used for all snapshots
  (or optionally one uv_xray per snapshot if provided that way).
- K_rgb and K_xray are fixed and taken from the FIRST loaded snapshot.
- points_xyz_camera are camera-frame 3D points stored in METRES in the NPZ,
  converted here to mm.

Per method, the FIRST VALID pose is used as the reference pose.
All later poses are expressed as residual transforms relative to this reference.

Residuals shown:
- translation drift: Δtx, Δty, Δtz [mm]
- rotation drift:    Δrx, Δry, Δrz [deg]
  where Δr is the Rodrigues vector of the residual rotation.

NPZ per snapshot should contain at least:
    points_xyz_camera : (121, 3)
    K_xray            : (3, 3)
    K_rgb             : (3, 3)
    corners_uv        : (9, 2)
    rgb_image         : (H, W, 3) optional but expected in prior workflow
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.pose_solvers import solve_pose
from overlay.tools.checkerboard_corner_detection import interpolate_grid_uv
from overlay.tools.homography import (
    estimate_homography_dlt,
    decompose_homography_to_pose,
)
from tracking.transforms import invert_transform

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
RANSAC_REPROJ_PX  = 3.0
RANSAC_CONFIDENCE = 0.99
RANSAC_ITERATIONS = 5000

PITCH_MM = 2.54
NROWS    = 11
NCOLS    = 11

METHODS = ("ransac", "ippe", "homography")
METHOD_LABELS = {
    "ransac": "RANSAC-PnP",
    "ippe": "IPPE",
    "homography": "Homography",
}


# ══════════════════════════════════════════════════════════
# Qt / file I/O
# ══════════════════════════════════════════════════════════

def _qt() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _pick(title: str) -> list[Path]:
    _qt()
    ps, _ = QFileDialog.getOpenFileNames(None, title, "", "NumPy NPZ (*.npz)")
    return sorted([Path(p) for p in ps if p], key=lambda p: p.name.lower())


def _load_snaps(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        try:
            with np.load(str(p), allow_pickle=False) as z:
                out.append({
                    "name":      p.stem,
                    "xyz_mm":    np.asarray(z["points_xyz_camera"], dtype=np.float64) * 1000.0,
                    "K_xray":    np.asarray(z["K_xray"], dtype=np.float64),
                    "K_rgb":     np.asarray(z["K_rgb"], dtype=np.float64),
                    "corners":   np.asarray(z["corners_uv"], dtype=np.float64),
                    "rgb_image": np.asarray(z["rgb_image"], dtype=np.uint8),
                })
            print(f"  [OK]   {p.name}")
        except Exception as e:
            print(f"  [SKIP] {p.name}: {e}")
    return out


def _load_uv_xray(paths: list[Path], n: int) -> list[np.ndarray | None]:
    """
    Flexible loader:
    - supports one NPZ containing one (121,2) uv_xray for all snapshots
    - supports one NPZ containing multiple arrays
    - supports one NPZ per snapshot
    """
    result: list[np.ndarray | None] = [None] * n
    if not paths:
        return result

    if len(paths) == 1:
        with np.load(str(paths[0]), allow_pickle=False) as z:
            keys = list(z.files)

            if "uv_xray" in keys:
                arr = np.asarray(z["uv_xray"], dtype=np.float64)
                if arr.ndim == 3:
                    for i in range(min(n, arr.shape[0])):
                        result[i] = arr[i]
                    return result
                if arr.ndim == 2 and arr.shape[1] == 2:
                    for i in range(n):
                        result[i] = arr
                    return result

            snap_keys = sorted(k for k in keys if k.startswith("snap_"))
            if snap_keys:
                for i, sk in enumerate(snap_keys[:n]):
                    result[i] = np.asarray(z[sk], dtype=np.float64)
                return result

            for k in keys:
                arr = np.asarray(z[k], dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    for i in range(n):
                        result[i] = arr
                    return result
        return result

    for i, p in enumerate(paths[:n]):
        try:
            with np.load(str(p), allow_pickle=False) as z:
                keys = list(z.files)
                if "uv_xray" in keys:
                    result[i] = np.asarray(z["uv_xray"], dtype=np.float64)
                else:
                    for k in keys:
                        arr = np.asarray(z[k], dtype=np.float64)
                        if arr.ndim == 2 and arr.shape[1] == 2:
                            result[i] = arr
                            break
        except Exception as e:
            print(f"  [SKIP uv_xray] {p.name}: {e}")
    return result


# ══════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════

def _rvec_tvec_to_T(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def _reproj_errors(xyz_mm, uv_obs, rvec, tvec, K) -> np.ndarray:
    pts, _ = cv2.projectPoints(
        xyz_mm.reshape(-1, 3).astype(np.float64),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        K.astype(np.float64),
        np.zeros((5, 1), dtype=np.float64),
    )
    return np.linalg.norm(pts.reshape(-1, 2) - uv_obs.reshape(-1, 2), axis=1)


def _rotmat_to_rvec_deg(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return np.rad2deg(rvec.reshape(3))


def _safe_norm_rows(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    return np.linalg.norm(arr, axis=1)


# ══════════════════════════════════════════════════════════
# Board-point grid (object coordinates, mm)
# ══════════════════════════════════════════════════════════

def _board_xyz_mm(
    nrows: int = NROWS,
    ncols: int = NCOLS,
    pitch_mm: float = PITCH_MM,
) -> np.ndarray:
    pts = []
    for r in range(nrows):
        for c in range(ncols):
            pts.append([c * pitch_mm, r * pitch_mm, 0.0])
    return np.array(pts, dtype=np.float64)


# ══════════════════════════════════════════════════════════
# Homography evaluation
# ══════════════════════════════════════════════════════════

def _eval_homography_snapshot(snap: dict, uv_xray: np.ndarray, K_rgb: np.ndarray, K_xray: np.ndarray) -> dict | None:
    """
    Compute T_cx via:
        T_bc : board -> camera  from RGB homography
        T_bx : board -> xray    from X-ray homography
        T_cx = T_bx @ inv(T_bc)

    Returns same-style dict as the PnP methods.
    """
    try:
        uv_rgb_121 = interpolate_grid_uv(
            snap["corners"],
            nrows=NROWS,
            ncols=NCOLS,
            pitch_mm=PITCH_MM,
            corner_step=5,
        )  # (121,2)

        board_xyz = _board_xyz_mm()
        board_xy = board_xyz[:, :2]
        uv_x = np.asarray(uv_xray, dtype=np.float64)

        H_rgb = estimate_homography_dlt(
            uv_img=uv_rgb_121,
            XY_grid=board_xy,
        )
        R_bc, t_bc, T_bc = decompose_homography_to_pose(H_rgb, K_rgb)

        H_xray = estimate_homography_dlt(
            uv_img=uv_x,
            XY_grid=board_xy,
        )
        R_bx, t_bx, T_bx = decompose_homography_to_pose(H_xray, K_xray)

        T_cx = T_bx @ invert_transform(T_bc)
        R_cx = T_cx[:3, :3]
        t_cx = T_cx[:3, 3].copy()
        rvec_cx, _ = cv2.Rodrigues(R_cx)

        # Optional sanity reprojection: project camera-frame 3D points into X-ray
        xyz_mm = snap["xyz_mm"]
        errs = _reproj_errors(xyz_mm, uv_x, rvec_cx.reshape(3), t_cx.reshape(3), K_xray)

        return {
            "T_cx":      T_cx,
            "rvec":      rvec_cx.reshape(3),
            "tvec":      t_cx.reshape(3),
            "mean_px":   float(np.mean(errs)),
            "median_px": float(np.median(errs)),
            "max_px":    float(np.max(errs)),
        }
    except Exception as e:
        print(f"      [HOMOGRAPHY ERROR] {e}")
        return None


# ══════════════════════════════════════════════════════════
# Per-snapshot evaluation
# ══════════════════════════════════════════════════════════

def _eval_snapshot(snap: dict, uv_xray: np.ndarray, K_rgb: np.ndarray, K_xray: np.ndarray) -> dict:
    """
    Returns dict with keys:
        'ransac', 'ippe', 'homography'
    """
    xyz = snap["xyz_mm"]
    uv  = np.asarray(uv_xray, dtype=np.float64)
    result = {}

    # ── RANSAC-PnP ─────────────────────────────────────────
    try:
        r = solve_pose(
            object_points_xyz=xyz,
            image_points_uv=uv,
            K=K_xray,
            pose_method="iterative_ransac",
            refine_with_iterative=True,
            ransac_reprojection_error_px=RANSAC_REPROJ_PX,
            ransac_confidence=RANSAC_CONFIDENCE,
            ransac_iterations_count=RANSAC_ITERATIONS,
        )
        T_cx = _rvec_tvec_to_T(r.rvec, r.tvec)
        errs = _reproj_errors(xyz, uv, r.rvec, r.tvec, K_xray)
        result["ransac"] = {
            "T_cx":         T_cx,
            "rvec":         np.asarray(r.rvec, dtype=np.float64).reshape(3),
            "tvec":         np.asarray(r.tvec, dtype=np.float64).reshape(3),
            "mean_px":      float(np.mean(errs)),
            "median_px":    float(np.median(errs)),
            "max_px":       float(np.max(errs)),
            "inlier_ratio": len(r.inlier_idx) / len(uv) if hasattr(r, "inlier_idx") and r.inlier_idx is not None else np.nan,
        }
    except Exception as e:
        print(f"      [RANSAC ERROR] {e}")
        result["ransac"] = None

    # ── IPPE ───────────────────────────────────────────────
    try:
        r = solve_pose(
            object_points_xyz=xyz,
            image_points_uv=uv,
            K=K_xray,
            pose_method="ippe",
            refine_with_iterative=True,
        )
        T_cx = _rvec_tvec_to_T(r.rvec, r.tvec)
        errs = _reproj_errors(xyz, uv, r.rvec, r.tvec, K_xray)
        result["ippe"] = {
            "T_cx":      T_cx,
            "rvec":      np.asarray(r.rvec, dtype=np.float64).reshape(3),
            "tvec":      np.asarray(r.tvec, dtype=np.float64).reshape(3),
            "mean_px":   float(np.mean(errs)),
            "median_px": float(np.median(errs)),
            "max_px":    float(np.max(errs)),
        }
    except Exception as e:
        print(f"      [IPPE ERROR] {e}")
        result["ippe"] = None

    # ── Homography ─────────────────────────────────────────
    result["homography"] = _eval_homography_snapshot(snap, uv, K_rgb, K_xray)

    return result


# ══════════════════════════════════════════════════════════
# Pretty-print helpers
# ══════════════════════════════════════════════════════════

def _print_T(T: np.ndarray, indent: str = "      ") -> None:
    for row in T:
        print(indent + "  ".join(f"{v:+12.6f}" for v in row))


def _print_result(snap_name: str, idx: int, res: dict) -> None:
    sep = "─" * 76
    print(f"\n{sep}")
    print(f"  Snapshot [{idx+1:02d}]  {snap_name}")
    print(sep)

    for method in METHODS:
        label = METHOD_LABELS[method]
        r = res.get(method)
        if r is None:
            print(f"\n  {label:12s}  →  FAILED")
            continue

        rvec = r["rvec"]
        tvec = r["tvec"]
        print(f"\n  {label}")
        print(f"    rvec  (Rodrigues) [rad]: [{rvec[0]:+.6f}  {rvec[1]:+.6f}  {rvec[2]:+.6f}]")
        print(f"    tvec              [mm]:  [{tvec[0]:+.3f}  {tvec[1]:+.3f}  {tvec[2]:+.3f}]")
        tail = ""
        if method == "ransac" and np.isfinite(r.get("inlier_ratio", np.nan)):
            tail = f"  inliers={r['inlier_ratio']*100:.1f}%"
        print(f"    Reprojection  mean={r['mean_px']:.3f}px  median={r['median_px']:.3f}px  max={r['max_px']:.3f}px{tail}")
        print("    T_cx [4×4]:")
        _print_T(r["T_cx"])


# ══════════════════════════════════════════════════════════
# Drift analysis
# ══════════════════════════════════════════════════════════

def _collect_method_entries(all_results: list[dict | None], snaps: list[dict], method: str) -> list[dict]:
    """
    Returns list of valid entries for one method:
        {
            "snap_idx": int,
            "snap_name": str,
            "T_cx": ...,
            "rvec": ...,
            "tvec": ...,
        }
    """
    entries = []
    for i, (snap, res) in enumerate(zip(snaps, all_results)):
        if res is None:
            continue
        rr = res.get(method)
        if rr is None:
            continue
        entries.append({
            "snap_idx": i,
            "snap_name": snap["name"],
            "T_cx": rr["T_cx"],
            "rvec": rr["rvec"],
            "tvec": rr["tvec"],
            "mean_px": rr["mean_px"],
            "median_px": rr["median_px"],
            "max_px": rr["max_px"],
        })
    return entries


def _compute_drift_per_method(all_results: list[dict | None], snaps: list[dict]) -> dict:
    """
    For each method:
    - use first valid pose as reference
    - compute residual transform:
          T_err = T_i @ inv(T_ref)
    - derive:
          dt_mm    = residual translation [mm]
          dr_deg   = Rodrigues residual rotation [deg]
          dt_norm  = ||dt||
          dr_norm  = ||dr||
    """
    drift = {}

    for method in METHODS:
        entries = _collect_method_entries(all_results, snaps, method)
        if not entries:
            drift[method] = None
            continue

        ref = entries[0]
        T_ref = ref["T_cx"]
        T_ref_inv = invert_transform(T_ref)

        dt_list = []
        dr_list = []
        snap_names = []
        snap_indices = []

        for e in entries:
            T_i = e["T_cx"]
            T_err = T_i @ T_ref_inv

            dt = T_err[:3, 3].copy()
            dr_deg = _rotmat_to_rvec_deg(T_err[:3, :3])

            dt_list.append(dt)
            dr_list.append(dr_deg)
            snap_names.append(e["snap_name"])
            snap_indices.append(e["snap_idx"])

        dt_arr = np.asarray(dt_list, dtype=np.float64)
        dr_arr = np.asarray(dr_list, dtype=np.float64)

        drift[method] = {
            "reference_snapshot_name": ref["snap_name"],
            "reference_snapshot_index": ref["snap_idx"],
            "dt_mm": dt_arr,
            "dr_deg": dr_arr,
            "dt_norm_mm": _safe_norm_rows(dt_arr),
            "dr_norm_deg": _safe_norm_rows(dr_arr),
            "snap_names": snap_names,
            "snap_indices": snap_indices,
        }

    return drift


def _print_drift_summary(drift: dict) -> None:
    print(f"\n{'═'*90}")
    print("  Drift Summary  (reference = first valid pose of each method)")
    print(f"{'═'*90}")

    for method in METHODS:
        d = drift.get(method)
        label = METHOD_LABELS[method]

        if d is None:
            print(f"  {label:12s}  no valid results")
            continue

        dt = d["dt_mm"]
        dr = d["dr_deg"]
        dt_norm = d["dt_norm_mm"]
        dr_norm = d["dr_norm_deg"]

        print(f"\n  {label}")
        print(f"    reference snapshot : {d['reference_snapshot_name']}")
        print(f"    used poses         : {len(dt)}")
        print(f"    median |Δt| [mm]   : {np.median(dt_norm):.4f}")
        print(f"    max    |Δt| [mm]   : {np.max(dt_norm):.4f}")
        print(f"    median |Δr| [deg]  : {np.median(dr_norm):.4f}")
        print(f"    max    |Δr| [deg]  : {np.max(dr_norm):.4f}")
        print(f"    std Δtx,Δty,Δtz    : ({np.std(dt[:,0]):.4f}, {np.std(dt[:,1]):.4f}, {np.std(dt[:,2]):.4f}) mm")
        print(f"    std Δrx,Δry,Δrz    : ({np.std(dr[:,0]):.4f}, {np.std(dr[:,1]):.4f}, {np.std(dr[:,2]):.4f}) deg")
    print()


# ══════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════

def _grouped_boxplot(ax, grouped_data: list[list[np.ndarray]], method_labels: list[str], group_labels: list[str], ylabel: str, title: str) -> None:
    """
    grouped_data structure:
        len(grouped_data) == number of component groups
        grouped_data[g][m] = 1D array for group g and method m

    Example:
        grouped_data = [
            [dtx_ransac, dtx_ippe, dtx_homo],
            [dty_ransac, dty_ippe, dty_homo],
            [dtz_ransac, dtz_ippe, dtz_homo],
        ]
    """
    n_groups = len(grouped_data)
    n_methods = len(method_labels)

    centers = np.arange(1, n_groups + 1, dtype=np.float64)
    offsets = np.linspace(-0.25, 0.25, n_methods)
    width = 0.22

    for m in range(n_methods):
        data_m = [np.asarray(grouped_data[g][m], dtype=np.float64) for g in range(n_groups)]
        positions = centers + offsets[m]
        bp = ax.boxplot(
            data_m,
            positions=positions,
            widths=width,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
        )
        # moderate fixed styling so methods are visually separated
        facecolors = ["#5DA5DA", "#60BD68", "#F17CB0"]
        col = facecolors[m % len(facecolors)]

        for patch in bp["boxes"]:
            patch.set(facecolor=col, alpha=0.65, edgecolor="black", linewidth=1.0)
        for key in ("whiskers", "caps", "medians"):
            for item in bp[key]:
                item.set(color="black", linewidth=1.0)

    ax.set_xticks(centers)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    legend_handles = []
    facecolors = ["#5DA5DA", "#60BD68", "#F17CB0"]
    for i, label in enumerate(method_labels):
        handle = plt.Rectangle((0, 0), 1, 1, facecolor=facecolors[i % len(facecolors)], edgecolor="black", alpha=0.65)
        legend_handles.append(handle)
    ax.legend(legend_handles, method_labels, loc="best")


def _plot_translation_drift(drift: dict, out_dir: Path) -> Path:
    method_order = [m for m in METHODS if drift.get(m) is not None]
    method_labels = [METHOD_LABELS[m] for m in method_order]

    grouped_data = [
        [drift[m]["dt_mm"][:, 0] for m in method_order],
        [drift[m]["dt_mm"][:, 1] for m in method_order],
        [drift[m]["dt_mm"][:, 2] for m in method_order],
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    _grouped_boxplot(
        ax=ax,
        grouped_data=grouped_data,
        method_labels=method_labels,
        group_labels=[r"$\Delta t_x$", r"$\Delta t_y$", r"$\Delta t_z$"],
        ylabel="Translation residual [mm]",
        title="Pose Drift Comparison — Translation Residuals",
    )
    plt.tight_layout()

    out_path = out_dir / "pose_drift_translation_boxplot.png"
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    print(f"[SAVE] {out_path}")
    plt.show()
    return out_path


def _plot_rotation_drift(drift: dict, out_dir: Path) -> Path:
    method_order = [m for m in METHODS if drift.get(m) is not None]
    method_labels = [METHOD_LABELS[m] for m in method_order]

    grouped_data = [
        [drift[m]["dr_deg"][:, 0] for m in method_order],
        [drift[m]["dr_deg"][:, 1] for m in method_order],
        [drift[m]["dr_deg"][:, 2] for m in method_order],
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    _grouped_boxplot(
        ax=ax,
        grouped_data=grouped_data,
        method_labels=method_labels,
        group_labels=[r"$\Delta r_x$", r"$\Delta r_y$", r"$\Delta r_z$"],
        ylabel="Rotation residual [deg]",
        title="Pose Drift Comparison — Rotation Residuals",
    )
    plt.tight_layout()

    out_path = out_dir / "pose_drift_rotation_boxplot.png"
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    print(f"[SAVE] {out_path}")
    plt.show()
    return out_path


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main() -> None:
    # ── Load snapshots ─────────────────────────────────────
    snap_paths = _pick("Select plane_fitting_snapshot NPZ files")
    if not snap_paths:
        print("[INFO] No snapshots selected.")
        return

    print(f"\n[INFO] Loading {len(snap_paths)} snapshot(s)…")
    snaps = _load_snaps(snap_paths)
    if not snaps:
        print("[ERROR] No valid snapshots.")
        return

    # ── Load uv_xray ───────────────────────────────────────
    print("\n[INFO] Select uv_xray NPZ file…")
    uv_xray_list = _load_uv_xray(_pick("Select uv_xray NPZ"), len(snaps))
    loaded = sum(1 for v in uv_xray_list if v is not None)
    print(f"[INFO] uv_xray ready for {loaded}/{len(snaps)} snapshot(s).")

    # Use the FIRST snapshot intrinsics for ALL snapshots/methods
    K_xray = snaps[0]["K_xray"]
    K_rgb  = snaps[0]["K_rgb"]

    print(f"[INFO] Fixed intrinsics from first snapshot:")
    print(f"       K_xray  fx={K_xray[0,0]:.3f}  fy={K_xray[1,1]:.3f}  cx={K_xray[0,2]:.3f}  cy={K_xray[1,2]:.3f}")
    print(f"       K_rgb   fx={K_rgb[0,0]:.3f}   fy={K_rgb[1,1]:.3f}   cx={K_rgb[0,2]:.3f}   cy={K_rgb[1,2]:.3f}")

    out_dir = snap_paths[0].parent
    print(f"[INFO] Output directory: {out_dir}\n")

    # ── Evaluate per snapshot ─────────────────────────────
    all_results: list[dict | None] = []

    for i, (snap, uv) in enumerate(zip(snaps, uv_xray_list)):
        if uv is None:
            print(f"  [SKIP] Snapshot {i+1}: no uv_xray data.")
            all_results.append(None)
            continue

        try:
            res = _eval_snapshot(snap, uv, K_rgb=K_rgb, K_xray=K_xray)
            all_results.append(res)
            _print_result(snap["name"], i, res)
        except Exception as e:
            print(f"  [ERROR] Snapshot {i+1}: {e}")
            all_results.append(None)

    # ── Drift analysis ─────────────────────────────────────
    drift = _compute_drift_per_method(all_results, snaps)
    _print_drift_summary(drift)

    # ── Plots ──────────────────────────────────────────────
    has_any = any(drift.get(m) is not None for m in METHODS)
    if not has_any:
        print("[ERROR] No valid drift results available.")
        return

    _plot_translation_drift(drift, out_dir)
    _plot_rotation_drift(drift, out_dir)


if __name__ == "__main__":
    main()