# -*- coding: utf-8 -*-
"""
test_page_tool_tracking.py

Standalone viewer for the "Tool Tracking" wizard page (Mode B).

Launches the REAL MainWindow and jumps straight to the Tool Tracking step,
so the page appears exactly as in the final wizard — same QSS, content card,
sidebar and nav buttons — without walking through Mode A first. No camera
and no tracker are touched; this page only loads files.

usage:
  python test_page_tool_tracking.py [--fullscreen] [--tool DIR]

  --fullscreen   show like the real app (showFullScreen)
  --tool DIR     preload a tool folder (e.g.
                 tracking/hydramarker/data/drill/PRO6202M) to see the
                 loaded state without clicking through the file dialogs
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from overlay.gui.main import MainWindow


def _preload_tool(win: MainWindow, tool_dir: str) -> None:
    """Fill the page's file slots from a tool folder, as 'Select tool folder'
    would, so the loaded state is visible immediately."""
    from tracking.hydramarker.tool_assets import suggest_tool_files

    sug = suggest_tool_files(Path(tool_dir))
    page = win.page_tool_tracking
    page._field_path = sug["field"]
    page._marker_json_path = sug["marker_json"]
    page._pivot_npz_path = sug["pivot_npz"]
    page._try_load()

    if not win.state.has_tool:
        print(f"[test_page_tool_tracking] no complete tool triple in {tool_dir}: {sug}")


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()

    # Jump straight to the Tool Tracking page. Mode B is one-way in the real
    # wizard, so flip the flag and switch directly — this bypasses the gating
    # that would otherwise require a completed Mode A.
    win.mode_b_active = True
    win._switch_to_page(win.IDX_TOOL_TRACKING)

    if "--tool" in sys.argv:
        tool_dir = sys.argv[sys.argv.index("--tool") + 1]
        _preload_tool(win, tool_dir)
        win.update_ui()

    if "--fullscreen" in sys.argv:
        win.showFullScreen()
    else:
        win.resize(1200, 700)
        win.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
