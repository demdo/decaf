"""Manual RealSense entry point for the shared HydraMarker live tracker.

This file intentionally contains no tracking logic.  It exists so the legacy
manual command can still be launched from the tests directory while the actual
camera loop, ``HydraTracker`` setup, and logging behavior live in
``hydramarker.debug.live_tracker_runner``.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    src_root = Path(__file__).resolve().parents[3]
    src = str(src_root)
    if src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()


def main() -> None:
    from tracking.hydramarker.debug.live_tracker_runner import run_live_tracker

    run_live_tracker()


if __name__ == "__main__":
    main()
