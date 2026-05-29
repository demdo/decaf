# -*- coding: utf-8 -*-
"""
debug_ippe_filtered.py

Laedt fuer mehrere plane_fitting_snapshot_*.npz Dateien die
Kalman-gefilterten 3D-Punkte (points_xyz_camera_filt) und berechnet
daraus die beiden IPPE-Kandidaten-Posen (ohne iterative refinement).

Ausgabe pro Snapshot:
    - Kandidat 0 und Kandidat 1:
        * T_cx  = camera -> xray
        * T_xc  = xray -> camera
        * tvec, rvec, reproj-Fehler
    - Reproj-Delta zwischen den beiden Kandidaten

Summary am Ende:
    - std tx/ty/tz und mean reproj fuer den gewaehlten Kandidaten

Benoetigte Snapshot-Keys:
    points_xyz_camera_filt  (N,3)  [m]
    K_xray                  (3,3)

uv_xray aus separater NPZ-Datei.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2
from PySide6.QtWidgets import QApplication, QFileDialog

from tracking.pose_solvers import solve_pose


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

def _load_snapshots(paths: list[Path]) -> list[dict]:
    snaps: list[dict] = []
    for p in paths:
        try:
            with np.load(str(p), allow_pickle=False) as z:
                keys = list(z.files)

                missing = [k for k in ("points_xyz_camera_filt", "K_xray") if k not in keys]
                if missing:
                    print(f"[SKIP] {p.name}: missing keys {missing}")
                    continue

                xyz_m = np.asarray(z["points_xyz_camera_filt"], dtype=np.float64)
                K_xray = np.asarray(z["K_xray"], dtype=np.float64)

                if xyz_m.ndim != 2 or xyz_m.shape[1] != 3:
                    print(f"[SKIP] {p.name}: points_xyz_camera_filt shape {xyz_m.shape}")
                    continue
                if K_xray.shape != (3, 3):
                    print(f"[SKIP] {p.name}: K_xray shape {K_xray.shape}")
                    continue

                snaps.append({
                    "name": p.stem,
                    "xyz_mm": xyz_m * 1000.0,
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

def _rvec_to_R(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def _invert_pose_cx_to_xc(R_cx: np.ndarray, t_cx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    T_cx: camera -> xray
        x_x = R_cx * x_c + t_cx

    Invert to:
    T_xc: xray -> camera
        x_c = R_xc * x_x + t_xc
    """
    R_cx = np.asarray(R_cx, dtype=np.float64).reshape(3, 3)
    t_cx = np.asarray(t_cx, dtype=np.float64).reshape(3)

    R_xc = R_cx.T
    t_xc = -R_xc @ t_cx
    return R_xc, t_xc


def _solve_both_candidates(
    xyz_mm: np.ndarray,
    uv_xray: np.ndarray,
    K_xray: np.ndarray,
) -> tuple[dict | None, dict | None, dict | None]:
    """
    Liefert:
        cand0, cand1, chosen

    cand0 / cand1 werden so sortiert, dass der candidate_index passt.
    chosen entspricht genau dem Kandidaten, den solve_pose als Ergebnis
    zurueckgegeben hat.
    """
    res = solve_pose(
        object_points_xyz=xyz_mm,
        image_points_uv=uv_xray,
        K=K_xray,
        pose_method="ippe",
        refine_with_iterative=False,
    )

    def _to_dict(rvec, tvec, reproj_mean, reproj_median, reproj_max, cand_idx):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        R_cx = _rvec_to_R(rvec)
        R_xc, t_xc = _invert_pose_cx_to_xc(R_cx, tvec)

        return {
            "candidate_index": int(cand_idx),
            "rvec": rvec,
            "tvec": tvec,
            "R_cx": R_cx,
            "R_xc": R_xc,
            "t_xc": t_xc,
            "reproj_mean_px": float(reproj_mean),
            "reproj_median_px": float(reproj_median),
            "reproj_max_px": float(reproj_max),
        }

    chosen = _to_dict(
        res.rvec,
        res.tvec,
        res.reproj_mean_px,
        res.reproj_median_px,
        res.reproj_max_px,
        res.candidate_index,
    )

    other = None
    if hasattr(res, "all_candidates") and res.all_candidates is not None:
        other_idx = 1 - res.candidate_index
        if other_idx < len(res.all_candidates):
            c = res.all_candidates[other_idx]
            other = _to_dict(
                c.rvec,
                c.tvec,
                c.reproj_mean_px,
                c.reproj_median_px,
                c.reproj_max_px,
                other_idx,
            )

    if chosen["candidate_index"] == 0:
        cand0, cand1 = chosen, other
    else:
        cand0, cand1 = other, chosen

    return cand0, cand1, chosen


# ============================================================
# Output
# ============================================================

def _print_matrix_block(label: str, M: np.ndarray) -> None:
    print(f"    {label} =")
    for row in M:
        print("      [" + "  ".join(f"{v:+10.5f}" for v in row) + "]")


def _print_candidate(label: str, c: dict | None) -> None:
    if c is None:
        print(f"  {label}: nicht verfuegbar")
        return

    tv = c["tvec"]
    rv = c["rvec"]
    txc = c["t_xc"]

    print(f"  {label} (Kandidat {c['candidate_index']}):")

    print("    --- T_cx (camera -> xray) ---")
    print(f"    tvec [mm]   : tx={tv[0]:+9.2f}  ty={tv[1]:+9.2f}  tz={tv[2]:+9.2f}")
    print(f"    rvec [rad]  : [{rv[0]:+.4f}  {rv[1]:+.4f}  {rv[2]:+.4f}]")
    print(f"    R_cx[2,2]   : {c['R_cx'][2,2]:+.4f}")

    print("    --- T_xc (xray -> camera) ---")
    print(f"    t_xc [mm]   : x={txc[0]:+9.2f}  y={txc[1]:+9.2f}  z={txc[2]:+9.2f}")
    print(f"    R_xc[2,2]   : {c['R_xc'][2,2]:+.4f}")

    print(
        f"    reproj [px] : "
        f"mean={c['reproj_mean_px']:.4f}  "
        f"median={c['reproj_median_px']:.4f}  "
        f"max={c['reproj_max_px']:.4f}"
    )

    _print_matrix_block("R_cx", c["R_cx"])
    _print_matrix_block("R_xc", c["R_xc"])


def _print_delta(cand0: dict | None, cand1: dict | None) -> None:
    if cand0 is None or cand1 is None:
        return

    dt_cx = cand1["tvec"] - cand0["tvec"]
    dt_xc = cand1["t_xc"] - cand0["t_xc"]

    print("  --- Delta Kandidat 1 - Kandidat 0 ---")
    print(
        f"    Delta t_cx [mm] : "
        f"dtx={dt_cx[0]:+9.2f}  dty={dt_cx[1]:+9.2f}  dtz={dt_cx[2]:+9.2f}"
    )
    print(
        f"    Delta t_xc [mm] : "
        f"dx ={dt_xc[0]:+9.2f}  dy ={dt_xc[1]:+9.2f}  dz ={dt_xc[2]:+9.2f}"
    )
    print(
        f"    Delta reproj [px] : "
        f"{cand1['reproj_mean_px'] - cand0['reproj_mean_px']:+.4f}"
    )


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        return

    chosen_list = [r["chosen"] for r in rows if r["chosen"] is not None]
    if not chosen_list:
        return

    t_cx_all = np.array([c["tvec"] for c in chosen_list], dtype=np.float64)
    t_xc_all = np.array([c["t_xc"] for c in chosen_list], dtype=np.float64)
    reproj_all = np.array([c["reproj_mean_px"] for c in chosen_list], dtype=np.float64)

    print("\n" + "=" * 70)
    print("SUMMARY  (gewaehlter Kandidat laut solve_pose)")
    print("=" * 70)

    print("\n  T_cx (camera -> xray)")
    print(f"    mean tx [mm]      : {np.mean(t_cx_all[:, 0]):+.3f}    std : {np.std(t_cx_all[:, 0]):.3f}")
    print(f"    mean ty [mm]      : {np.mean(t_cx_all[:, 1]):+.3f}    std : {np.std(t_cx_all[:, 1]):.3f}")
    print(f"    mean tz [mm]      : {np.mean(t_cx_all[:, 2]):+.3f}    std : {np.std(t_cx_all[:, 2]):.3f}")

    print("\n  T_xc (xray -> camera)")
    print(f"    mean x [mm]       : {np.mean(t_xc_all[:, 0]):+.3f}    std : {np.std(t_xc_all[:, 0]):.3f}")
    print(f"    mean y [mm]       : {np.mean(t_xc_all[:, 1]):+.3f}    std : {np.std(t_xc_all[:, 1]):.3f}")
    print(f"    mean z [mm]       : {np.mean(t_xc_all[:, 2]):+.3f}    std : {np.std(t_xc_all[:, 2]):.3f}")

    print(f"\n  mean reproj [px]    : {np.mean(reproj_all):.4f}")
    print(f"  max  reproj [px]    : {np.max(reproj_all):.4f}")


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

    uv_paths = _pick_npz_files("uv_xray NPZ auswaehlen")
    uv_list = _load_uv_xray(uv_paths, len(snaps))

    rows: list[dict] = []

    for snap, uv in zip(snaps, uv_list):
        if uv is None:
            print(f"\n[SKIP] {snap['name']}: kein uv_xray gefunden.")
            continue

        xyz_mm = snap["xyz_mm"]
        K_xray = snap["K_xray"]

        if xyz_mm.shape[0] != uv.shape[0]:
            print(
                f"\n[SKIP] {snap['name']}: mismatch "
                f"xyz={xyz_mm.shape[0]} Punkte, uv={uv.shape[0]} Punkte."
            )
            continue

        try:
            cand0, cand1, chosen = _solve_both_candidates(xyz_mm, uv, K_xray)

            rows.append({
                "name": snap["name"],
                "cand0": cand0,
                "cand1": cand1,
                "chosen": chosen,
            })

            print("\n" + "=" * 70)
            print(f"  {snap['name']}")
            print("=" * 70)
            _print_candidate("Kandidat 0", cand0)
            _print_candidate("Kandidat 1", cand1)
            _print_delta(cand0, cand1)

            if chosen is not None:
                print(f"  -> solve_pose gewaehlt: Kandidat {chosen['candidate_index']}")

        except Exception as e:
            print(f"\n[ERROR] {snap['name']}: {e}")

    _print_summary(rows)
    print("\n[INFO] Fertig.")


if __name__ == "__main__":
    main()