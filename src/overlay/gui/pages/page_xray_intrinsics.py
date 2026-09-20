# overlay/gui/pages/page_xray_intrinsics.py

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

from overlay.tools.xray_distortion import load_xray_distortion


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


class XrayIntrinsicsPage(StaticImagePage):
    """
    Step — Load X-ray intrinsics (K_xray)

    Special-case page:
      - NO image/viewport on the left
      - NO instructions box
      - stats only:
          K_xray: -                (before)
          K_xray: np.array (3×3)   (after load)
    """

    def __init__(self, state: SessionState, on_complete_changed, parent=None):
        super().__init__(parent)

        self.state = state
        self.on_complete_changed = on_complete_changed

        # ======================================================
        # RIGHT: controls
        # ======================================================
        self.btn_load = QPushButton("Load X-ray intrinsics")
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
        return getattr(self.state, "K_xray", None) is not None

    def refresh(self) -> None:
        if self.is_complete():
            K = self.state.K_xray

            lines = [
                " ".join(f"{v:.2f}" for v in row)
                for row in K
            ]

            K_str = "\n".join(["np.array (3×3)"] + lines)

            field = getattr(self.state, "xray_dist_field", None)

            self.set_stats_rows([
                ("K_xray", K_str),
                ("dist_xray", "-" if field is None else field.describe()),
            ])
        else:
            self.set_stats_rows([
                ("K_xray", "-"),
                ("dist_xray", "-"),
            ])

        try:
            self.set_viewport_background(active=False)
        except Exception:
            pass
        try:
            self.set_image(None)
        except Exception:
            pass

    def _load_K_from_json(self, path: str) -> np.ndarray:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        K = np.array(data, dtype=float)
        if K.shape != (3, 3):
            raise ValueError("Intrinsics matrix in JSON must be 3×3.")
        return K

    def _load_K_from_npz(self, path: str) -> np.ndarray:
        npz = np.load(path, allow_pickle=True)

        # 1) First try common expected keys
        preferred_keys = ["K_xray", "K"]
        for key in preferred_keys:
            if key in npz.files:
                K = np.array(npz[key], dtype=float)
                if K.shape == (3, 3):
                    return K
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
            return candidates[0][1]

        if len(candidates) > 1:
            keys = ", ".join(key for key, _ in candidates)
            raise ValueError(
                "NPZ contains multiple 3×3 arrays. "
                f"Could not decide which one is K_xray. Candidates: {keys}"
            )

        available = ", ".join(npz.files) if npz.files else "(no arrays)"
        raise ValueError(
            "No 3×3 intrinsics matrix found in NPZ. "
            f"Available arrays: {available}"
        )

    def load_intrinsics(self) -> None:
        dlg = QFileDialog(self.window(), "Select X-ray intrinsics file")
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

        self.load_from_path(path)

    def load_from_path(self, path: str) -> bool:
        """Load K_xray (+ distortion field) from a file; also used by the
        evaluation-mode auto-load. Returns True on success."""
        try:
            suffix = Path(path).suffix.lower()

            field = None
            if suffix == ".json":
                K = self._load_K_from_json(path)
            elif suffix == ".npz":
                K = self._load_K_from_npz(path)
                field = load_xray_distortion(path)
            else:
                raise ValueError(
                    "Unsupported file type. Please select a .json or .npz file."
                )

            self.state.K_xray = K
            self.state.xray_dist_field = field

            self.refresh()
            self.on_complete_changed()
            return True

        except Exception as e:
            QMessageBox.critical(self, "Failed to load X-ray intrinsics", str(e))
            return False