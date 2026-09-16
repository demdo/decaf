# overlay/gui/pages/page_camera_intrinsics.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from PySide6.QtWidgets import (
    QPushButton,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
    QWidget,
)

from overlay.gui.state import SessionState
from overlay.gui.pages.templates.templ_static_image import StaticImagePage


def _hide_widget(w: QWidget | None) -> None:
    if w is None:
        return
    try:
        w.setVisible(False)
    except Exception:
        pass
    try:
        w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    except Exception:
        pass
    try:
        w.setMinimumSize(0, 0)
    except Exception:
        pass
    try:
        w.setMaximumSize(0, 0)
    except Exception:
        pass


def _hide_if_exists(obj, attr_name: str) -> None:
    _hide_widget(getattr(obj, attr_name, None))


# OpenCV distortion order: k1 k2 p1 p2 [k3 [k4 k5 k6 [s1 s2 s3 s4 [taux tauy]]]]
_DIST_NAMES = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
               "s1", "s2", "s3", "s4", "taux", "tauy"]

# Coefficients per displayed row (keeps the stats box from growing too wide).
_DIST_PER_ROW = 5


def _dist_names(n: int) -> list[str]:
    if n <= len(_DIST_NAMES):
        return _DIST_NAMES[:n]
    return [f"c{i}" for i in range(n)]


def _num_populated(d: np.ndarray) -> int:
    """
    Number of leading coefficients up to (and including) the last non-zero one.

    OpenCV always returns a full-length vector and zero-pads the coefficients
    that the chosen distortion model does not use (e.g. rational8 leaves
    s1..s4 and taux/tauy at zero in a 14-element vector).
    """
    nz = np.flatnonzero(d)
    if nz.size == 0:
        return d.size
    return int(nz[-1]) + 1


def _format_dist(dist: np.ndarray) -> str:
    """
    Build the multi-line stats value for dist_rgb.

    Only the populated coefficients are shown; the zero-padded tail is hidden.
    The header keeps the full vector length.

    set_stats_rows() renders the first line next to the label and lays out the
    remaining lines in a grid (one cell per whitespace-separated token), so
    alternating name/value lines end up column-aligned:

        dist_rgb: np.array (1×5)
        k1      k2     p1     p2     k3
        -0.12    0.23   0.00  -0.00   0.01
    """
    d = np.asarray(dist, dtype=float).reshape(-1)
    names = _dist_names(d.size)

    n = _num_populated(d)

    lines = [f"np.array (1×{d.size})"]
    for start in range(0, n, _DIST_PER_ROW):
        stop = min(start + _DIST_PER_ROW, n)
        chunk_names = names[start:stop]
        chunk_vals = d[start:stop]
        lines.append(" ".join(chunk_names))
        lines.append(" ".join(f"{v:.2f}" for v in chunk_vals))

    return "\n".join(lines)


class CameraIntrinsicsPage(StaticImagePage):
    """
    Step — Load camera intrinsics (K_rgb)

    Special-case page (same layout as XrayIntrinsicsPage):
      - NO image/viewport on the left
      - NO instructions box
      - stats only:
          K_rgb / dist_rgb: -                       (before)
          K_rgb:    np.array (3×3)                  (after load)
          dist_rgb: np.array (1×N), named columns

    dist_rgb is only shown when the loaded file contains distortion
    coefficients; otherwise it stays at "-".
    """

    def __init__(self, state: SessionState, on_complete_changed, parent=None):
        super().__init__(parent)

        self.state = state
        self.on_complete_changed = on_complete_changed

        # ======================================================
        # RIGHT: controls
        # ======================================================
        self.btn_load = QPushButton("Load camera intrinsics")
        self.btn_load.clicked.connect(self.load_intrinsics)
        self.btn_load.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.controls_content.addWidget(self.btn_load)

        # ======================================================
        # SPECIAL-CASE UI: hide instructions panel completely
        # ======================================================
        _hide_if_exists(self, "instructions_box")
        _hide_if_exists(self, "instructions_group")
        _hide_if_exists(self, "instructions_panel")
        _hide_if_exists(self, "instructions_frame")
        _hide_if_exists(self, "instructions_label")

        # ======================================================
        # SPECIAL-CASE UI: hide the entire image/viewport area
        # ======================================================
        _hide_if_exists(self, "image_label")
        _hide_if_exists(self, "viewport")
        _hide_if_exists(self, "viewport_widget")
        _hide_if_exists(self, "viewport_frame")
        _hide_if_exists(self, "viewport_container")
        _hide_if_exists(self, "image_container")
        _hide_if_exists(self, "image_frame")
        _hide_if_exists(self, "left_panel")
        _hide_if_exists(self, "left_widget")
        _hide_if_exists(self, "workspace_widget")

        try:
            self.set_viewport_background(active=False)
        except Exception:
            pass
        try:
            self.set_image(None)
        except Exception:
            pass

        self.refresh()

    def is_complete(self) -> bool:
        return getattr(self.state, "K_rgb", None) is not None

    def refresh(self) -> None:
        if self.is_complete():
            K = self.state.K_rgb

            lines = [
                " ".join(f"{v:.2f}" for v in row)
                for row in K
            ]

            K_str = "\n".join(["np.array (3×3)"] + lines)

            dist = getattr(self.state, "dist_rgb", None)
            dist_str = "-" if dist is None else _format_dist(dist)

            self.set_stats_rows([
                ("K_rgb", K_str),
                ("dist_rgb", dist_str),
            ])
        else:
            self.set_stats_rows([
                ("K_rgb", "-"),
                ("dist_rgb", "-"),
            ])

        try:
            self.set_viewport_background(active=False)
        except Exception:
            pass
        try:
            self.set_image(None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _as_dist(arr) -> np.ndarray | None:
        try:
            d = np.array(arr, dtype=float).reshape(-1)
        except Exception:
            return None
        if d.size == 0:
            return None
        return d

    def _load_from_json(self, path: str) -> tuple[np.ndarray, np.ndarray | None]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dist = None

        if isinstance(data, dict):
            K_raw = None
            for key in ("K_rgb", "K", "camera_matrix", "intrinsics"):
                if key in data:
                    K_raw = data[key]
                    break
            if K_raw is None:
                raise ValueError(
                    "JSON object contains no intrinsics matrix. "
                    "Expected one of: K_rgb, K, camera_matrix, intrinsics."
                )
            for key in ("dist_rgb", "dist", "dist_coeffs", "distCoeffs"):
                if key in data:
                    dist = self._as_dist(data[key])
                    break
        else:
            K_raw = data

        K = np.array(K_raw, dtype=float)
        if K.shape != (3, 3):
            raise ValueError("Intrinsics matrix in JSON must be 3×3.")
        return K, dist

    def _load_from_npz(self, path: str) -> tuple[np.ndarray, np.ndarray | None]:
        npz = np.load(path, allow_pickle=True)

        dist = None
        for key in ("dist_rgb", "dist", "dist_coeffs", "distCoeffs"):
            if key in npz.files:
                dist = self._as_dist(npz[key])
                break

        # 1) First try common expected keys
        preferred_keys = ["K_rgb", "K"]
        for key in preferred_keys:
            if key in npz.files:
                K = np.array(npz[key], dtype=float)
                if K.shape == (3, 3):
                    return K, dist
                raise ValueError(
                    f"Array '{key}' found in NPZ, but has shape {K.shape} instead of (3, 3)."
                )

        # 2) Otherwise search all arrays for 3x3 candidates
        candidates: list[tuple[str, np.ndarray]] = []
        for key in npz.files:
            arr = np.array(npz[key], dtype=float)
            if arr.shape == (3, 3):
                candidates.append((key, arr))

        if len(candidates) == 1:
            return candidates[0][1], dist

        if len(candidates) > 1:
            keys = ", ".join(key for key, _ in candidates)
            raise ValueError(
                "NPZ contains multiple 3×3 arrays. "
                f"Could not decide which one is K_rgb. Candidates: {keys}"
            )

        available = ", ".join(npz.files) if npz.files else "(no arrays)"
        raise ValueError(
            "No 3×3 intrinsics matrix found in NPZ. "
            f"Available arrays: {available}"
        )

    def load_intrinsics(self) -> None:
        dlg = QFileDialog(self.window(), "Select camera intrinsics file")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter(
            "Supported Files (*.json *.npz);;JSON Files (*.json);;NPZ Files (*.npz);;All Files (*)"
        )

        dlg.setOption(QFileDialog.DontUseNativeDialog, True)

        if dlg.exec() != QFileDialog.Accepted:
            return

        files = dlg.selectedFiles()
        path = files[0] if files else ""
        if not path:
            return

        try:
            suffix = Path(path).suffix.lower()

            if suffix == ".json":
                K, dist = self._load_from_json(path)
            elif suffix == ".npz":
                K, dist = self._load_from_npz(path)
            else:
                raise ValueError(
                    "Unsupported file type. Please select a .json or .npz file."
                )

            self.state.K_rgb = K
            self.state.dist_rgb = dist

            self.refresh()
            self.on_complete_changed()

        except Exception as e:
            QMessageBox.critical(self, "Failed to load camera intrinsics", str(e))
