# -*- coding: utf-8 -*-
"""
test_drift_pose.py

Vergleicht die Drift von T_cx ueber mehrere Snapshots fuer drei Methoden:

    1) iterative_ransac
    2) ippe
    3) homography

Eingabe:
    - mehrere Snapshot-NPZ-Dateien
    - eine oder mehrere uv_xray-NPZ-Dateien

Pro Snapshot wird bevorzugt verwendet:
    points_xyz_camera_filt
    points_xyz_camera
    points_xyz_camera_single

Zusaetzlich benoetigt:
    K_xray

Auswertung:
    - Translation std (tx, ty, tz)
    - Rotations-std (approximiert ueber Winkel gegen Referenz-R)
    - Frame-to-frame |dt|
    - Frame-to-frame dR
    - Mean reprojection error pro Snapshot
    - tz pro Snapshot

Hinweise:
    - iterative_ransac und ippe werden ueber solve_pose(...) aus
      tracking.pose_solvers ausgefuehrt.
    - homography ebenfalls zentral ueber solve_pose(...).
    - Fuer homography ist refine_with_iterative standardmaessig deaktiviert,
      damit der reine Homography-Ansatz sichtbar bleibt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.pose_solvers import solve_pose


# ============================================================
# Konfiguration
# ============================================================

_XYZ_KEYS_PREFERRED = (
    "points_xyz_camera_filt",
    "points_xyz_camera",
    "points_xyz_camera_single",
)

_METHODS = (
    "iterative_ransac",
    "ippe",
    "homography",
)

_METHOD_LABELS = {
    "iterative_ransac": "ransac iterative",
    "ippe":             "ippe",
    "homography":       "homography",
}

_METHOD_COLORS = {
    "iterative_ransac": "#4C8BB5",
    "ippe":             "#E07B3F",
    "homography":       "#5BAB6E",
}

_METHOD_MARKERS = {
    "iterative_ransac": "o",
    "ippe":             "s",
    "homography":       "^",
}

_RANSAC_REPROJ_ERROR_PX = 8.0
_RANSAC_CONFIDENCE = 0.99
_RANSAC_ITERATIONS = 100

_REFINE_RANSAC_WITH_ITERATIVE = True
_REFINE_IPPE_WITH_ITERATIVE = True
_REFINE_HOMOGRAPHY_WITH_ITERATIVE = False

_BOARD_PITCH_MM = 2.54
_BOARD_NROWS = 11
_BOARD_NCOLS = 11


# ============================================================
# Qt helpers
# ============================================================

def _qt() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def _pick_npz_files(title: str) -> list[Path]:
    _qt()
    paths, _ = QFileDialog.getOpenFileNames(None, title, "", "NumPy NPZ (*.npz)")
    return sorted([Path(p) for p in paths if p], key=lambda p: p.name.lower())


# ============================================================
# Load helpers
# ============================================================

def _first_existing_xyz_key(keys: list[str]) -> str | None:
    for k in _XYZ_KEYS_PREFERRED:
        if k in keys:
            return k
    return None


def _load_snapshots(paths: list[Path]) -> list[dict]:
    snaps: list[dict] = []

    for p in paths:
        try:
            with np.load(str(p), allow_pickle=False) as z:
                keys = list(z.files)

                if "K_xray" not in keys:
                    print(f"[SKIP] {p.name}: missing key 'K_xray'")
                    continue

                xyz_key = _first_existing_xyz_key(keys)
                if xyz_key is None:
                    print(
                        f"[SKIP] {p.name}: none of "
                        f"{list(_XYZ_KEYS_PREFERRED)} found"
                    )
                    continue

                xyz_m = np.asarray(z[xyz_key], dtype=np.float64)
                K_xray = np.asarray(z["K_xray"], dtype=np.float64)

                if xyz_m.ndim != 2 or xyz_m.shape[1] != 3:
                    print(f"[SKIP] {p.name}: {xyz_key} shape {xyz_m.shape}")
                    continue

                if K_xray.shape != (3, 3):
                    print(f"[SKIP] {p.name}: K_xray shape {K_xray.shape}")
                    continue

                snaps.append({
                    "path": p,
                    "name": p.stem,
                    "xyz_mm": xyz_m * 1000.0,
                    "xyz_key": xyz_key,
                    "K_xray": K_xray,
                })

        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")

    return snaps


def _first_uv_array_from_npz(path: Path) -> np.ndarray | None:
    try:
        with np.load(str(path), allow_pickle=False) as z:
            if "uv_xray" in z.files:
                arr = np.asarray(z["uv_xray"], dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    return arr

            for k in z.files:
                arr = np.asarray(z[k], dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    return arr
    except Exception:
        pass

    return None


def _load_uv_xray(paths: list[Path], n_snapshots: int) -> list[np.ndarray | None]:
    result: list[np.ndarray | None] = [None] * n_snapshots

    if not paths:
        return result

    if len(paths) == 1:
        p = paths[0]
        try:
            with np.load(str(p), allow_pickle=False) as z:
                keys = list(z.files)

                if "uv_xray" in keys:
                    arr = np.asarray(z["uv_xray"], dtype=np.float64)

                    if arr.ndim == 2 and arr.shape[1] == 2:
                        for i in range(n_snapshots):
                            result[i] = arr
                        return result

                    if arr.ndim == 3 and arr.shape[2] == 2:
                        for i in range(min(n_snapshots, arr.shape[0])):
                            result[i] = arr[i]
                        return result

                snap_keys = sorted(k for k in keys if k.startswith("snap_"))
                if snap_keys:
                    for i, k in enumerate(snap_keys[:n_snapshots]):
                        arr = np.asarray(z[k], dtype=np.float64)
                        if arr.ndim == 2 and arr.shape[1] == 2:
                            result[i] = arr
                    return result

                for k in keys:
                    arr = np.asarray(z[k], dtype=np.float64)
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        for i in range(n_snapshots):
                            result[i] = arr
                        return result

        except Exception as e:
            print(f"[WARN] {p.name}: {e}")

        return result

    for i, p in enumerate(paths[:n_snapshots]):
        arr = _first_uv_array_from_npz(p)
        if arr is not None:
            result[i] = arr
        else:
            print(f"[SKIP uv_xray] {p.name}: no (N,2) array found")

    return result


# ============================================================
# Pose helpers
# ============================================================

def _pose_result_to_T(result) -> np.ndarray:
    rvec = np.asarray(result.rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(result.tvec, dtype=np.float64).reshape(3, 1)

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = tvec
    return T


def _rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    cos_theta = np.clip(0.5 * (np.trace(R_a.T @ R_b) - 1.0), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _frame_to_frame(T_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    Ts = np.asarray(T_list, dtype=np.float64)
    t_all = Ts[:, :3, 3]
    R_all = Ts[:, :3, :3]

    dt_mm = np.linalg.norm(np.diff(t_all, axis=0), axis=1)
    dR_deg = np.array(
        [_rotation_angle_deg(R_all[i], R_all[i + 1]) for i in range(len(R_all) - 1)],
        dtype=np.float64,
    )
    return dt_mm, dR_deg


def _std_metrics(T_list: list[np.ndarray]) -> dict[str, float]:
    Ts = np.asarray(T_list, dtype=np.float64)
    t_all = Ts[:, :3, 3]
    R_all = Ts[:, :3, :3]

    R_ref = R_all[0]
    angles = np.array([_rotation_angle_deg(R_ref, R) for R in R_all], dtype=np.float64)

    return {
        "std_tx": float(np.std(t_all[:, 0])),
        "std_ty": float(np.std(t_all[:, 1])),
        "std_tz": float(np.std(t_all[:, 2])),
        "std_dR": float(np.std(angles)),
    }


# ============================================================
# Console output
# ============================================================

def _print_summary(
    method: str,
    names: list[str],
    xyz_keys: list[str],
    T_list: list[np.ndarray],
    reproj_mean_list: list[float],
) -> None:
    Ts = np.asarray(T_list, dtype=np.float64)
    t_all = Ts[:, :3, 3]

    m = _std_metrics(T_list)

    print("\n" + "=" * 78)
    print(f"T_cx  [{_METHOD_LABELS[method]}]")
    print("=" * 78)

    print(
        f"\nTranslation std [mm]:  "
        f"tx={m['std_tx']:.3f}  ty={m['std_ty']:.3f}  tz={m['std_tz']:.3f}"
    )
    print(f"Rotation    std [deg]: dR={m['std_dR']:.4f}")

    if len(T_list) >= 2:
        dt_mm, dR_deg = _frame_to_frame(T_list)
        print(
            f"\nFrame-to-frame |dt|:  mean={np.mean(dt_mm):.3f} mm   "
            f"max={np.max(dt_mm):.3f} mm"
        )
        print(
            f"Frame-to-frame  dR:   mean={np.mean(dR_deg):.4f} deg  "
            f"max={np.max(dR_deg):.4f} deg"
        )

    print(
        f"\nReprojection mean [px]: "
        f"mean={np.mean(reproj_mean_list):.4f}  max={np.max(reproj_mean_list):.4f}"
    )

    print("\nPer snapshot:")
    for name, xyz_key, t, reproj in zip(names, xyz_keys, t_all, reproj_mean_list):
        print(
            f"  {name}  ({xyz_key}): "
            f"tx={t[0]:+9.3f}  ty={t[1]:+9.3f}  tz={t[2]:+9.3f}   "
            f"reproj={reproj:.4f}px"
        )


# ============================================================
# Plot
# ============================================================

def _show_plots(results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle(
        "Drift comparison of T_cx across snapshots\n"
        "Methods: ransac iterative vs ippe vs homography",
        fontsize=13,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Oben links: std tx/ty/tz
    # --------------------------------------------------------
    ax_t_std = axes[0, 0]
    std_labels = ["tx", "ty", "tz"]
    x_std = np.arange(len(std_labels))
    bar_width = 0.22
    n_methods = len(_METHODS)

    for i, method in enumerate(_METHODS):
        T_list = results.get(method, {}).get("T_list", [])
        if not T_list:
            continue

        m = _std_metrics(T_list)
        offset = (i - (n_methods - 1) / 2) * bar_width

        ax_t_std.bar(
            x_std + offset,
            [m["std_tx"], m["std_ty"], m["std_tz"]],
            width=bar_width,
            color=_METHOD_COLORS[method],
            label=_METHOD_LABELS[method],
            alpha=0.85,
        )

    ax_t_std.set_xticks(x_std)
    ax_t_std.set_xticklabels(std_labels)
    ax_t_std.set_ylabel("std [mm]")
    ax_t_std.set_title("Translation std across snapshots")
    ax_t_std.grid(axis="y", alpha=0.35, linestyle="--")
    ax_t_std.legend(fontsize=9)

    # --------------------------------------------------------
    # Oben rechts: std dR
    # --------------------------------------------------------
    ax_r_std = axes[0, 1]

    for i, method in enumerate(_METHODS):
        T_list = results.get(method, {}).get("T_list", [])
        if not T_list:
            continue

        m = _std_metrics(T_list)
        offset = (i - (n_methods - 1) / 2) * bar_width

        ax_r_std.bar(
            [0 + offset],
            [m["std_dR"]],
            width=bar_width,
            color=_METHOD_COLORS[method],
            label=_METHOD_LABELS[method],
            alpha=0.85,
        )

    ax_r_std.set_xticks([0])
    ax_r_std.set_xticklabels(["dR"])
    ax_r_std.set_ylabel("std [deg]")
    ax_r_std.set_title("Rotation std across snapshots")
    ax_r_std.grid(axis="y", alpha=0.35, linestyle="--")
    ax_r_std.legend(fontsize=9)

    # --------------------------------------------------------
    # Mitte links: frame-to-frame |dt|
    # --------------------------------------------------------
    ax_dt = axes[1, 0]

    for method in _METHODS:
        T_list = results.get(method, {}).get("T_list", [])
        if len(T_list) < 2:
            continue

        dt_mm, _ = _frame_to_frame(T_list)
        xs = np.arange(1, len(dt_mm) + 1)

        ax_dt.plot(
            xs,
            dt_mm,
            marker=_METHOD_MARKERS[method],
            color=_METHOD_COLORS[method],
            label=_METHOD_LABELS[method],
            linewidth=1.8,
            markersize=6,
        )

    max_steps = max(
        (len(v.get("T_list", [])) - 1 for v in results.values() if len(v.get("T_list", [])) > 1),
        default=1,
    )
    xs_ref = np.arange(1, max_steps + 1)

    ax_dt.set_xticks(xs_ref)
    ax_dt.set_xticklabels([f"{i-1}→{i}" for i in xs_ref], rotation=30, fontsize=8)
    ax_dt.set_xlabel("Snapshot transition")
    ax_dt.set_ylabel("|dt| [mm]")
    ax_dt.set_title("Frame-to-frame translation jump")
    ax_dt.grid(alpha=0.35, linestyle="--")
    ax_dt.legend(fontsize=9)

    # --------------------------------------------------------
    # Mitte rechts: frame-to-frame dR
    # --------------------------------------------------------
    ax_dR = axes[1, 1]

    for method in _METHODS:
        T_list = results.get(method, {}).get("T_list", [])
        if len(T_list) < 2:
            continue

        _, dR_deg = _frame_to_frame(T_list)
        xs = np.arange(1, len(dR_deg) + 1)

        ax_dR.plot(
            xs,
            dR_deg,
            marker=_METHOD_MARKERS[method],
            color=_METHOD_COLORS[method],
            label=_METHOD_LABELS[method],
            linewidth=1.8,
            markersize=6,
        )

    ax_dR.set_xticks(xs_ref)
    ax_dR.set_xticklabels([f"{i-1}→{i}" for i in xs_ref], rotation=30, fontsize=8)
    ax_dR.set_xlabel("Snapshot transition")
    ax_dR.set_ylabel("dR [deg]")
    ax_dR.set_title("Frame-to-frame rotation jump")
    ax_dR.grid(alpha=0.35, linestyle="--")
    ax_dR.legend(fontsize=9)

    # --------------------------------------------------------
    # Unten links: mean reprojection error pro Snapshot
    # --------------------------------------------------------
    ax_repr = axes[2, 0]

    for method in _METHODS:
        reproj = results.get(method, {}).get("reproj_mean_list", [])
        if not reproj:
            continue

        xs = np.arange(len(reproj))
        ax_repr.plot(
            xs,
            reproj,
            marker=_METHOD_MARKERS[method],
            color=_METHOD_COLORS[method],
            label=_METHOD_LABELS[method],
            linewidth=1.8,
            markersize=6,
        )

    max_len = max((len(v.get("reproj_mean_list", [])) for v in results.values()), default=1)
    ax_repr.set_xticks(np.arange(max_len))
    ax_repr.set_xlabel("Snapshot index")
    ax_repr.set_ylabel("mean reproj [px]")
    ax_repr.set_title("Mean reprojection error per snapshot")
    ax_repr.grid(alpha=0.35, linestyle="--")
    ax_repr.legend(fontsize=9)

    # --------------------------------------------------------
    # Unten rechts: tz pro Snapshot
    # --------------------------------------------------------
    ax_tz = axes[2, 1]

    for method in _METHODS:
        T_list = results.get(method, {}).get("T_list", [])
        if not T_list:
            continue

        tz = np.asarray(T_list, dtype=np.float64)[:, 2, 3]
        xs = np.arange(len(tz))

        ax_tz.plot(
            xs,
            tz,
            marker=_METHOD_MARKERS[method],
            color=_METHOD_COLORS[method],
            label=_METHOD_LABELS[method],
            linewidth=1.8,
            markersize=6,
        )

    max_len_tz = max((len(v.get("T_list", [])) for v in results.values()), default=1)
    ax_tz.set_xticks(np.arange(max_len_tz))
    ax_tz.set_xlabel("Snapshot index")
    ax_tz.set_ylabel("tz [mm]")
    ax_tz.set_title("tz per snapshot")
    ax_tz.grid(alpha=0.35, linestyle="--")
    ax_tz.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main() -> None:
    snap_paths = _pick_npz_files("Snapshot NPZ-Dateien auswaehlen")
    if not snap_paths:
        print("[INFO] Keine Snapshots ausgewaehlt.")
        return

    snaps = _load_snapshots(snap_paths)
    if not snaps:
        print("[ERROR] Keine gueltigen Snapshots.")
        return

    uv_paths = _pick_npz_files("uv_xray NPZ-Datei(en) auswaehlen")
    uv_list = _load_uv_xray(uv_paths, len(snaps))

    expected_n = _BOARD_NROWS * _BOARD_NCOLS
    print(
        f"\n[INFO] Erwartete Punktzahl: {expected_n} "
        f"({_BOARD_NROWS}x{_BOARD_NCOLS}, pitch={_BOARD_PITCH_MM:.2f} mm)"
    )

    results: dict[str, dict] = {
        method: {
            "names": [],
            "xyz_keys": [],
            "T_list": [],
            "reproj_mean_list": [],
        }
        for method in _METHODS
    }

    for snap, uv in zip(snaps, uv_list):
        name = snap["name"]
        xyz_mm = snap["xyz_mm"]
        xyz_key = snap["xyz_key"]
        K_xray = snap["K_xray"]

        print("\n" + "-" * 78)
        print(f"[SNAPSHOT] {name}")
        print("-" * 78)
        print(f"  xyz source key : {xyz_key}")
        print(f"  xyz shape      : {xyz_mm.shape}")

        if uv is None:
            print("  [SKIP] kein uv_xray gefunden.")
            continue

        uv = np.asarray(uv, dtype=np.float64)
        print(f"  uv shape       : {uv.shape}")

        if xyz_mm.shape[0] != uv.shape[0]:
            print(
                f"  [SKIP] Punktzahl passt nicht: "
                f"xyz={xyz_mm.shape[0]}  uv={uv.shape[0]}"
            )
            continue

        if xyz_mm.shape[0] != expected_n:
            print(
                f"  [WARN] Unerwartete Punktzahl: {xyz_mm.shape[0]} "
                f"(erwartet {expected_n})"
            )

        for method in _METHODS:
            try:
                if method == "iterative_ransac":
                    result = solve_pose(
                        object_points_xyz=xyz_mm,
                        image_points_uv=uv,
                        K=K_xray,
                        dist_coeffs=None,
                        pose_method="iterative_ransac",
                        refine_with_iterative=_REFINE_RANSAC_WITH_ITERATIVE,
                        ransac_reprojection_error_px=_RANSAC_REPROJ_ERROR_PX,
                        ransac_confidence=_RANSAC_CONFIDENCE,
                        ransac_iterations_count=_RANSAC_ITERATIONS,
                        pitch_mm=_BOARD_PITCH_MM,
                        nrows=_BOARD_NROWS,
                        ncols=_BOARD_NCOLS,
                    )

                elif method == "ippe":
                    result = solve_pose(
                        object_points_xyz=xyz_mm,
                        image_points_uv=uv,
                        K=K_xray,
                        dist_coeffs=None,
                        pose_method="ippe",
                        refine_with_iterative=_REFINE_IPPE_WITH_ITERATIVE,
                        pitch_mm=_BOARD_PITCH_MM,
                        nrows=_BOARD_NROWS,
                        ncols=_BOARD_NCOLS,
                    )

                elif method == "homography":
                    result = solve_pose(
                        object_points_xyz=xyz_mm,
                        image_points_uv=uv,
                        K=K_xray,
                        dist_coeffs=None,
                        pose_method="homography",
                        refine_with_iterative=_REFINE_HOMOGRAPHY_WITH_ITERATIVE,
                        pitch_mm=_BOARD_PITCH_MM,
                        nrows=_BOARD_NROWS,
                        ncols=_BOARD_NCOLS,
                    )

                else:
                    raise ValueError(f"Unknown method: {method}")

                T_cx = _pose_result_to_T(result)

                results[method]["names"].append(name)
                results[method]["xyz_keys"].append(xyz_key)
                results[method]["T_list"].append(T_cx)
                results[method]["reproj_mean_list"].append(float(result.reproj_mean_px))

                t = T_cx[:3, 3]
                print(
                    f"  [{_METHOD_LABELS[method]:>16}] "
                    f"tx={t[0]:+9.3f}  ty={t[1]:+9.3f}  tz={t[2]:+9.3f}   "
                    f"reproj={result.reproj_mean_px:.4f}px"
                )

            except Exception as e:
                print(f"  [ERROR {method}] {e}")

    any_result = False
    for method in _METHODS:
        if results[method]["T_list"]:
            _print_summary(
                method=method,
                names=results[method]["names"],
                xyz_keys=results[method]["xyz_keys"],
                T_list=results[method]["T_list"],
                reproj_mean_list=results[method]["reproj_mean_list"],
            )
            any_result = True

    if not any_result:
        print("[ERROR] Keine gueltigen Posen berechnet.")
        return

    _show_plots(results)
    print("\n[INFO] Fertig.")


if __name__ == "__main__":
    main()