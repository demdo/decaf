"""
C++ backend wrapper for HydraMarker.

This module adapts the pybind11 C++ module to the Python interface used by the
rest of the project. The goal is to keep the public Python code independent of
the exact C++ binding details.
"""

from pathlib import Path
import importlib.util

import numpy as np


def _load_hydramarker_cpp():
    this_dir = Path(__file__).resolve().parent
    pyd_dir = this_dir.parent / "cpp" / "build" / "Release"

    matches = list(pyd_dir.glob("hydramarker_cpp*.pyd"))

    if not matches:
        raise ImportError(f"Could not find hydramarker_cpp .pyd in {pyd_dir}")

    pyd_path = matches[0]

    spec = importlib.util.spec_from_file_location(
        "hydramarker_cpp",
        pyd_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load hydramarker_cpp from {pyd_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


_hm = _load_hydramarker_cpp()


def __getattr__(name):
    return getattr(_hm, name)


MarkerField = _hm.MarkerField
CheckerboardDetector = _hm.CheckerboardDetector

_MarkerField = _hm.MarkerField
_generate_planar_field = _hm.generate_planar_field


class MarkerFieldCpp:
    """
    Thin Python wrapper around the C++ MarkerField implementation.
    """

    def __init__(self, path: str):
        self._mf = _MarkerField(path)

    def find_patch(self, patch):
        return [
            {
                "x": match.x,
                "y": match.y,
                "rotation": match.rotation_deg,
            }
            for match in self._mf.find_patch(patch)
        ]


def generate_planar_field(
    rows: int,
    cols: int,
    patch_size: int,
    max_ms: float = 60000.0,
    max_trial: int = 100000,
    is_print: bool = False,
) -> np.ndarray:
    field = _generate_planar_field(
        rows=rows,
        cols=cols,
        patch_size=patch_size,
        max_ms=max_ms,
        max_trial=max_trial,
        is_print=is_print,
    )

    return np.asarray(field, dtype=np.uint8)