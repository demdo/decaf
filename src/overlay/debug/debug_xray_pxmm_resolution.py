from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog

from overlay.tools.blob_detection import estimate_pitch_nn


# ============================================================
# Config
# ============================================================

REAL_PITCH_MM = 2.54
SDD_MM = 980.0


# ============================================================
# Qt helpers
# ============================================================

def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _select_npz(title: str) -> Path:
    _get_app()
    path, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        "NPZ files (*.npz);;All files (*)",
        options=QFileDialog.DontUseNativeDialog,
    )
    if not path:
        raise RuntimeError("No file selected.")
    return Path(path)


# ============================================================
# Load UV
# ============================================================

def load_uv(npz_path: Path) -> tuple[np.ndarray, str]:
    data = np.load(npz_path, allow_pickle=True)

    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr.astype(np.float64), key

    raise RuntimeError("No Nx2 uv array found in NPZ.")


# ============================================================
# Main
# ============================================================

def main() -> None:
    npz_path = _select_npz("Select UV NPZ (e.g. uv_debug_raw.npz)")

    uv, key = load_uv(npz_path)

    if len(uv) < 10:
        raise RuntimeError("Too few points for reliable pitch estimation.")

    pitch_px = float(estimate_pitch_nn(uv))

    px_per_mm = pitch_px / REAL_PITCH_MM
    mm_per_px = 1.0 / px_per_mm

    f_phys = SDD_MM * px_per_mm

    nn_dists = []
    for i in range(len(uv)):
        d = np.linalg.norm(uv - uv[i], axis=1)
        d[i] = np.inf
        nn_dists.append(np.min(d))
    nn_dists = np.asarray(nn_dists)

    print("\n============================================================")
    print("DEBUG X-RAY DETECTOR RESOLUTION (UV)")
    print("============================================================")

    print(f"file              = {npz_path}")
    print(f"uv key            = {key}")
    print(f"n points          = {len(uv)}")

    print("\nNearest neighbor distances")
    print("--------------------------")
    print(f"mean [px]         = {np.mean(nn_dists):.6f}")
    print(f"median [px]       = {np.median(nn_dists):.6f}")
    print(f"std [px]          = {np.std(nn_dists):.6f}")
    print(f"min [px]          = {np.min(nn_dists):.6f}")
    print(f"max [px]          = {np.max(nn_dists):.6f}")

    print("\nResolution")
    print("----------")
    print(f"pitch_px          = {pitch_px:.6f}")
    print(f"px/mm             = {px_per_mm:.6f}")
    print(f"mm/px             = {mm_per_px:.6f}")

    print("\nPhysical focal length")
    print("---------------------")
    print(f"f_phys [px]       = {f_phys:.6f}")

    print("\nSanity expectation")
    print("------------------")
    print("Expected range if correct (calibrated f = 4586.3 px):")
    print("  px/mm ≈ 4.5 – 4.9")
    print("  f ≈ 4400 – 4800 px")


if __name__ == "__main__":
    main()