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
