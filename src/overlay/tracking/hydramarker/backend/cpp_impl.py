"""
C++ backend wrapper for HydraMarker.

This module adapts the pybind11 C++ module to the Python interface used by the
rest of the project. The goal is to keep the public Python code independent of
the exact C++ binding details.
"""

import numpy as np

from hydramarker_cpp import (
    MarkerField as _MarkerField,
    generate_planar_field as _generate_planar_field,
)


class MarkerFieldCpp:
    """
    Thin Python wrapper around the C++ MarkerField implementation.
    """

    def __init__(self, path: str):
        """
        Load the C++ MarkerField from a .field file.

        Parameters
        ----------
        path:
            Path to the HydraMarker .field file.
        """
        self._mf = _MarkerField(path)

    def find_patch(self, patch):
        """
        Forward patch lookup to the C++ implementation.

        Parameters
        ----------
        patch:
            Flattened k*k binary patch as a Python list.

        Returns
        -------
        list[dict]
            Converted C++ PatchMatch objects as Python dictionaries.
        """
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
    """
    Generate a planar HydraMarker field using the C++ backend.

    Parameters
    ----------
    rows:
        Number of marker rows (cells).

    cols:
        Number of marker cols (cells).

    patch_size:
        k for k x k patches.

    max_ms:
        Maximum generation time in milliseconds.

    max_trial:
        Maximum number of generation trials.

    is_print:
        Whether to print generation progress.

    Returns
    -------
    np.ndarray
        Binary uint8 array with shape (rows, cols).
    """
    field = _generate_planar_field(
        rows=rows,
        cols=cols,
        patch_size=patch_size,
        max_ms=max_ms,
        max_trial=max_trial,
        is_print=is_print,
    )

    return np.asarray(field, dtype=np.uint8)