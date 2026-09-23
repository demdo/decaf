# overlay/gui/evaluation.py
#
# Evaluation mode: everything the wizard writes to disk goes through one
# EvaluationSession. In Normal mode the pages write nothing (state.eval is
# None); in Evaluation mode every capture lands in one session folder as
# <label>_<kind>.<ext> with a line in log.md. The label comes from the bar
# at the top of the window, so a file always says which board placement or
# test it belongs to. While a Cam2X series runs (steps 3-5) the main window
# appends the height, <label>_h1_<kind> .. <label>_h3_<kind> (a height
# captured again: h2b, h2c, ...); the pooled result of the series is
# <label>_h3_cam2x_pooled.json (tag of the last height).
#
# Folder:
#   overlay/data/Evaluation/<session>/
#     session.json      mode, calibration files (path + md5), notes
#     log.md            one row per saved file: time, page, label, file, numbers
#     <label>_plane.npz, <label>_xray.IMA, <label>_markers.npz,
#     <label>_cam2x.npz + .json, <label>_depth.npz, <label>_overlay.npz, ...

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# Default locations (absolute, independent of the working directory)
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = SRC_DIR / "overlay" / "data" / "Evaluation"
DEFAULT_CAMERA_CALIB = (
    SRC_DIR / "tracking" / "hydramarker" / "data" / "realsense"
    / "realsense_camera_calibration_rational8.npz"
)
DEFAULT_XRAY_CALIB = SRC_DIR / "overlay" / "data" / "K_xray.npz"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_jsonable(v):
    """numpy -> plain python for json.dump; unknown objects -> str."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


# ============================================================
# Startup choice
# ============================================================

@dataclass
class StartupChoice:
    evaluation: bool = False
    session_name: str = ""
    eval_root: Path = DEFAULT_EVAL_ROOT
    autoload_calibration: bool = True
    camera_calib_path: Path = DEFAULT_CAMERA_CALIB
    xray_calib_path: Path = DEFAULT_XRAY_CALIB
    notes: str = ""


# ============================================================
# Session
# ============================================================

class EvaluationSession:
    """
    One folder per evaluation session; all pages save through it.

    label       current label from the bar ("cal1", "raised25", "h10_on", ...)
    save_npz    <label>_<kind>.npz + log row
    save_json   <label>_<kind>.json (human-readable diagnostics)
    copy_file   <label>_<kind><suffix of src> (e.g. the X-ray IMA)
    """

    def __init__(self, root: Path, name: str, notes: str = ""):
        self.name = name
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.label = ""
        self.last_saved: Path | None = None
        self.on_saved: Callable[[Path], None] | None = None
        # set by the EvalBar so a page can change the label (set_label)
        self.on_label_set: Callable[[str], None] | None = None
        # set by the main window: the height of a Cam2X series ("h1"..) while
        # steps 3-5 run, "" otherwise; appended to the label in file names
        self.tag_provider: Callable[[], str] | None = None
        # every file label a file was saved under, in order (the main window
        # names discarded / ended series from it, whatever the bar shows now)
        self.used_labels: list[str] = []
        self._session_json = self.dir / "session.json"
        self._log = self.dir / "log.md"
        self._meta: dict = {
            "session": name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": notes,
            "calibration_files": {},
        }
        if not self._session_json.exists():
            self._write_session_json()
        if not self._log.exists():
            self._log.write_text(
                f"# Evaluation log — {name}\n\n"
                "| time | page | label | file | numbers |\n"
                "|---|---|---|---|---|\n",
                encoding="utf-8",
            )

    # ---------------- session.json ----------------

    def _write_session_json(self) -> None:
        with open(self._session_json, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(self._meta), f, indent=2)

    def record_calibration_file(self, role: str, path: Path | str) -> None:
        p = Path(path)
        self._meta["calibration_files"][role] = {
            "path": str(p),
            "md5": _md5(p) if p.exists() else None,
        }
        self._write_session_json()

    # ---------------- naming ----------------

    @staticmethod
    def sanitize_label(text: str) -> str:
        s = re.sub(r"[^A-Za-z0-9_\-]+", "_", text.strip()).strip("_")
        return s or "capture"

    def set_label(self, text: str) -> None:
        """Change the label from a page; the bar's field follows."""
        self.label = text
        if self.on_label_set is not None:
            self.on_label_set(text)

    @staticmethod
    def next_label(label: str) -> str:
        """plane03 -> plane04 (width kept), h10_on -> h10_on_2, "" -> "2"."""
        m = re.match(r"^(.*?)(\d+)$", label.strip())
        if m:
            head, num = m.groups()
            return f"{head}{int(num) + 1:0{len(num)}d}"
        return f"{label.strip()}_2" if label.strip() else "2"

    def file_label(self, tagged: bool = True) -> str:
        """The label as it goes into file names: sanitized, plus the
        series height ("cal_h2") unless tagged is False."""
        base = self.sanitize_label(self.label)
        tag = self.tag_provider() if (tagged and self.tag_provider is not None) else ""
        return f"{base}_{tag}" if tag else base

    def path_for(self, kind: str, suffix: str = ".npz", tagged: bool = True) -> Path:
        label = self.file_label(tagged)
        self.used_labels.append(label)
        base = f"{label}_{kind}"
        p = self.dir / f"{base}{suffix}"
        n = 2
        while p.exists():
            p = self.dir / f"{base}_{n}{suffix}"
            n += 1
        return p

    # ---------------- saving ----------------

    def _done(self, path: Path, page: str, summary: dict | None, tagged: bool = True) -> Path:
        self.last_saved = path
        self.log(page, path.name, summary or {}, tagged=tagged)
        print(f"[EVAL] saved {path}")
        if self.on_saved is not None:
            self.on_saved(path)
        return path

    def save_npz(
        self,
        kind: str,
        arrays: dict,
        *,
        page: str = "",
        summary: dict | None = None,
    ) -> Path:
        path = self.path_for(kind, ".npz")
        clean = {k: v for k, v in arrays.items() if v is not None}
        np.savez(str(path), **clean)
        return self._done(path, page, summary)

    def save_json(
        self,
        kind: str,
        data: dict,
        *,
        page: str = "",
        log: bool = False,
        summary: dict | None = None,
        tagged: bool = True,
    ) -> Path:
        path = self.path_for(kind, ".json", tagged=tagged)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(data), f, indent=2)
        if log:
            return self._done(path, page, summary, tagged=tagged)
        self.last_saved = path
        return path

    def copy_file(self, kind: str, src: Path | str, *, page: str = "") -> Path:
        src = Path(src)
        path = self.path_for(kind, src.suffix)
        shutil.copy2(src, path)
        return self._done(path, page, {"source": src.name})

    def subdir(self, kind: str) -> Path:
        """A folder <label>_<kind>/ (live snapshot layers)."""
        p = self.path_for(kind, "")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def log(self, page: str, filename: str, summary: dict, tagged: bool = True) -> None:
        nums = ", ".join(f"{k}={_fmt(v)}" for k, v in summary.items())
        row = (
            f"| {time.strftime('%H:%M:%S')} | {page} | "
            f"{self.file_label(tagged)} | {filename} | {nums} |\n"
        )
        try:
            with open(self._log, "a", encoding="utf-8") as f:
                f.write(row)
        except Exception as e:
            print(f"[EVAL] could not write log: {e}")


# ============================================================
# Startup dialog
# ============================================================

class StartupDialog(QDialog):
    """Normal | Evaluation, shown before the main window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DeCAF — start")
        self.setMinimumWidth(640)

        self.rb_normal = QRadioButton("Normal — nothing is written to disk")
        self.rb_eval = QRadioButton("Evaluation — every capture is saved to a session folder")
        self.rb_normal.setChecked(True)

        self.ed_session = QLineEdit(time.strftime("%Y%m%d_"))
        self.ed_root = QLineEdit(str(DEFAULT_EVAL_ROOT))
        self.btn_root = QPushButton("…")
        self.btn_root.setObjectName("SecondaryBtn")
        self.btn_root.setFixedWidth(36)
        self.btn_root.clicked.connect(self._browse_root)

        self.chk_autoload = QCheckBox("Load calibration files automatically (steps 1 and 2)")
        self.chk_autoload.setChecked(True)
        self.ed_cam = QLineEdit(str(DEFAULT_CAMERA_CALIB))
        self.ed_xray = QLineEdit(str(DEFAULT_XRAY_CALIB))
        self.btn_cam = QPushButton("…")
        self.btn_xray = QPushButton("…")
        for b in (self.btn_cam, self.btn_xray):
            b.setObjectName("SecondaryBtn")
            b.setFixedWidth(36)
        self.btn_cam.clicked.connect(lambda: self._browse_file(self.ed_cam, "Camera calibration"))
        self.btn_xray.clicked.connect(lambda: self._browse_file(self.ed_xray, "X-ray intrinsics"))

        self.ed_notes = QLineEdit()
        self.ed_notes.setPlaceholderText("notes for session.json (camera position, table height, ...)")

        self.eval_box = QWidget()
        g = QGridLayout(self.eval_box)
        g.setContentsMargins(24, 4, 0, 4)
        g.addWidget(QLabel("Session name"), 0, 0)
        g.addWidget(self.ed_session, 0, 1, 1, 2)
        g.addWidget(QLabel("Output folder"), 1, 0)
        g.addWidget(self.ed_root, 1, 1)
        g.addWidget(self.btn_root, 1, 2)
        g.addWidget(self.chk_autoload, 2, 0, 1, 3)
        g.addWidget(QLabel("Camera"), 3, 0)
        g.addWidget(self.ed_cam, 3, 1)
        g.addWidget(self.btn_cam, 3, 2)
        g.addWidget(QLabel("X-ray"), 4, 0)
        g.addWidget(self.ed_xray, 4, 1)
        g.addWidget(self.btn_xray, 4, 2)
        g.addWidget(QLabel("Notes"), 5, 0)
        g.addWidget(self.ed_notes, 5, 1, 1, 2)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        title = QLabel("DeCAF")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        lay.addWidget(title)
        lay.addWidget(self.rb_normal)
        lay.addWidget(self.rb_eval)
        lay.addWidget(self.eval_box)
        lay.addWidget(buttons)

        self.rb_eval.toggled.connect(self._sync)
        self.chk_autoload.toggled.connect(self._sync)
        self._sync()

    def _sync(self) -> None:
        ev = self.rb_eval.isChecked()
        self.eval_box.setEnabled(ev)
        auto = self.chk_autoload.isChecked()
        for w in (self.ed_cam, self.ed_xray, self.btn_cam, self.btn_xray):
            w.setEnabled(ev and auto)

    def _browse_root(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.ed_root.text())
        if d:
            self.ed_root.setText(d)

    def _browse_file(self, edit: QLineEdit, title: str) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, title, str(Path(edit.text()).parent), "NPZ / JSON (*.npz *.json);;All Files (*)"
        )
        if p:
            edit.setText(p)

    def choice(self) -> StartupChoice:
        return StartupChoice(
            evaluation=self.rb_eval.isChecked(),
            session_name=EvaluationSession.sanitize_label(self.ed_session.text()),
            eval_root=Path(self.ed_root.text()),
            autoload_calibration=self.chk_autoload.isChecked(),
            camera_calib_path=Path(self.ed_cam.text()),
            xray_calib_path=Path(self.ed_xray.text()),
            notes=self.ed_notes.text(),
        )


# ============================================================
# Bar at the top of the main window (Evaluation mode only)
# ============================================================

class EvalBar(QFrame):
    """
    session name · label field (+ series height) · last saved file · "New series".

    The label is read by the pages at save time through session.label; the
    bar keeps it in sync on every keystroke.
    """

    def __init__(self, session: EvaluationSession, parent=None):
        super().__init__(parent)
        self.session = session
        self.setObjectName("EvalBar")
        self.setStyleSheet(
            "#EvalBar { background: #fff3cd; border: 1px solid #ffe69c; border-radius: 10px; }"
            "#EvalBar QLabel { color: #664d03; }"
            "#EvalBar QLineEdit { background: white; border: 1px solid #ffda6a; "
            "border-radius: 6px; padding: 4px 8px; min-width: 160px; }"
        )

        self.lbl_session = QLabel(f"EVALUATION · {session.name}")
        self.lbl_session.setStyleSheet("font-weight: 700;")
        self.ed_label = QLineEdit()
        self.ed_label.setPlaceholderText("label, e.g. cal1 / raised25 / h10_on")
        self.ed_label.textChanged.connect(self._on_label)
        self.lbl_tag = QLabel("")
        self.lbl_tag.setToolTip("Height of the Cam2X series, appended to the label in file names")
        self.lbl_tag.hide()
        self.lbl_last = QLabel("nothing saved yet")
        self.btn_new_capture = QPushButton("New series")
        # NoFocus: SPACE captures / accepts on the pages and must not re-click it
        self.btn_new_capture.setFocusPolicy(Qt.NoFocus)
        self.btn_new_capture.setToolTip(
            "Back to step 3 (plane fitting) and start the Cam2X series again; "
            "intrinsics stay loaded, all heights and everything after step 2 are cleared."
        )

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(12)
        lay.addWidget(self.lbl_session)
        lay.addWidget(QLabel("Label"))
        lay.addWidget(self.ed_label)
        lay.addWidget(self.lbl_tag)
        lay.addWidget(self.lbl_last, stretch=1)
        lay.addWidget(self.btn_new_capture)

        session.on_saved = self._on_saved
        session.on_label_set = self.ed_label.setText

    def _on_label(self, text: str) -> None:
        self.session.label = text

    def _on_saved(self, path: Path) -> None:
        self.lbl_last.setText(f"saved: {path.name}")

    def set_tag(self, tag: str) -> None:
        """Show the series height the file names get ("_h2"), or nothing."""
        self.lbl_tag.setText(f"_{tag}" if tag else "")
        self.lbl_tag.setVisible(bool(tag))
