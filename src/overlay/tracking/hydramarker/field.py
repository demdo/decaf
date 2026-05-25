"""
High-level Python wrapper for the HydraMarker marker field.

The MarkerField stores the global binary HydraMarker pattern. Later in the
pipeline, a local binary patch extracted from the image is matched against this
global field to recover the patch position and orientation.

This class intentionally hides the backend implementation. At the moment, the
actual lookup is performed by the C++ backend through pybind11.
"""

import numpy as np

from .backend.cpp_impl import (
    MarkerFieldCpp,
    generate_planar_field,
)


class MarkerField:
    """
    Backend-independent Python interface for marker field lookup.
    """

    def __init__(self, backend):
        """
        Store the backend implementation.

        Parameters
        ----------
        backend:
            Object that implements find_patch(...).
        """
        self.backend = backend

    @classmethod
    def from_file(cls, path: str):
        """
        Load a marker field from a .field file.

        Parameters
        ----------
        path:
            Path to the HydraMarker .field file.

        Returns
        -------
        MarkerField
            Python wrapper around the C++ MarkerField backend.
        """
        return cls(MarkerFieldCpp(path))

    @staticmethod
    def generate_planar(
        rows: int,
        cols: int,
        patch_size: int,
        max_ms: float = 60000.0,
        max_trial: int = 100000,
        is_print: bool = False,
    ) -> np.ndarray:
        """
        Generate a planar HydraMarker field.

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
            Binary uint8 marker field.
        """
        return generate_planar_field(
            rows=rows,
            cols=cols,
            patch_size=patch_size,
            max_ms=max_ms,
            max_trial=max_trial,
            is_print=is_print,
        )

    def find_patch(self, patch: np.ndarray):
        """
        Find a local binary patch in the global marker field.

        Parameters
        ----------
        patch:
            k x k binary numpy array. Values are expected to be 0 or 1.

        Returns
        -------
        list[dict]
            One dictionary per match:
                {
                    "x": global x index,
                    "y": global y index,
                    "rotation": rotation in degrees
                }
        """
        patch = patch.astype(np.uint8).flatten().tolist()
        return self.backend.find_patch(patch)