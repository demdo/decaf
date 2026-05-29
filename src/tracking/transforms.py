from __future__ import annotations

import cv2
import numpy as np


def as_transform(T: np.ndarray, name: str = "T") -> np.ndarray:
    """
    Validate and normalize a rigid 4x4 homogeneous transform.

    Parameters
    ----------
    T : np.ndarray
        Transform to validate.
    name : str, optional
        Variable name used in error messages.

    Returns
    -------
    np.ndarray
        Float64 transform of shape (4, 4).
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4,4), got {T.shape}")
    return T


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform from rotation and translation.

    Convention
    ----------
    For T_ab, points are mapped as

        x_b = T_ab @ x_a_h

    Parameters
    ----------
    rotation : np.ndarray
        Rotation matrix of shape (3, 3).
    translation : np.ndarray
        Translation vector of shape (3,), (3,1), or compatible.

    Returns
    -------
    np.ndarray
        Homogeneous transform of shape (4, 4).
    """
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float64).reshape(3, 1)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Invert a rigid 4x4 homogeneous transform.

    Parameters
    ----------
    T : np.ndarray
        Rigid transform of shape (4, 4).

    Returns
    -------
    np.ndarray
        Inverse transform of shape (4, 4).
    """
    T = as_transform(T, "T")

    R = T[:3, :3]
    t = T[:3, 3:4]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3:4] = -R.T @ t
    return T_inv


def transform_points(T: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    """
    Apply a rigid 4x4 homogeneous transform to one or more 3D points.

    Parameters
    ----------
    T : np.ndarray
        Homogeneous transform of shape (4, 4).
    points_xyz : np.ndarray
        3D point(s) of shape (3,) or (N, 3).

    Returns
    -------
    np.ndarray
        Transformed point(s) with the same outer shape:
        - input (3,)   -> output (3,)
        - input (N, 3) -> output (N, 3)
    """
    T = as_transform(T, "T")
    points_xyz = np.asarray(points_xyz, dtype=np.float64)

    squeeze_output = False
    if points_xyz.ndim == 1:
        if points_xyz.shape[0] != 3:
            raise ValueError(
                f"points_xyz must have shape (3,) or (N,3), got {points_xyz.shape}"
            )
        points_xyz = points_xyz.reshape(1, 3)
        squeeze_output = True
    elif points_xyz.ndim == 2:
        if points_xyz.shape[1] != 3:
            raise ValueError(
                f"points_xyz must have shape (3,) or (N,3), got {points_xyz.shape}"
            )
    else:
        raise ValueError(
            f"points_xyz must have shape (3,) or (N,3), got {points_xyz.shape}"
        )

    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_xyz, ones], axis=1)
    points_t_h = (T @ points_h.T).T
    points_t = points_t_h[:, :3]

    if squeeze_output:
        return points_t[0]
    return points_t


def extract_translation(T: np.ndarray) -> np.ndarray:
    """
    Return the translation vector of a 4x4 transform.

    Parameters
    ----------
    T : np.ndarray
        Homogeneous transform of shape (4, 4).

    Returns
    -------
    np.ndarray
        Translation vector of shape (3,).
    """
    T = as_transform(T, "T")
    return T[:3, 3].astype(np.float64).copy()


def extract_translation_z(T: np.ndarray) -> float:
    """
    Return the z-component of the translation of a 4x4 transform.

    Parameters
    ----------
    T : np.ndarray
        Homogeneous transform of shape (4, 4).

    Returns
    -------
    float
        z-translation.
    """
    T = as_transform(T, "T")
    return float(T[2, 3])


def transform_to_rvec_tvec(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a rigid 4x4 transform to OpenCV pose vectors.

    Parameters
    ----------
    T : np.ndarray
        Homogeneous transform of shape (4, 4).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (rvec, tvec), both shaped (3,1).
    """
    T = as_transform(T, "T")
    R = T[:3, :3]
    t = T[:3, 3:4]

    rvec, _ = cv2.Rodrigues(R)
    return rvec.astype(np.float64), t.astype(np.float64)


def rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV pose vectors to a rigid 4x4 transform.

    Parameters
    ----------
    rvec : np.ndarray
        Rotation vector of shape (3,) or (3,1).
    tvec : np.ndarray
        Translation vector of shape (3,) or (3,1).

    Returns
    -------
    np.ndarray
        Homogeneous transform of shape (4, 4).
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    R, _ = cv2.Rodrigues(rvec)
    return make_transform(R, tvec)


__all__ = [
    "as_transform",
    "make_transform",
    "invert_transform",
    "transform_points",
    "extract_translation",
    "extract_translation_z",
    "transform_to_rvec_tvec",
    "rvec_tvec_to_transform",
]