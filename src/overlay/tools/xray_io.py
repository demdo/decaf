# overlay/tools/xray_io.py
#
# Reading an anatomical X-ray file into the raw 8-bit grey image the overlay
# pages keep in SessionState.xray_image_anatomy (flip and undistortion
# happen later, right before warping).

from __future__ import annotations

import hashlib

import cv2
import numpy as np
import pydicom


def read_xray_gray_u8(path: str) -> np.ndarray:
    """DICOM (.dcm/.ima) min-max scaled to uint8; other images as greyscale."""
    if path.lower().endswith((".dcm", ".ima")):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Could not read image.")
    return img


def xray_pixel_md5(img: np.ndarray) -> str:
    """Fingerprint of an X-ray's pixel data (shape, dtype, bytes). The same
    exposure gives the same value however it was exported or named; two
    exposures never do (quantum noise)."""
    a = np.ascontiguousarray(img)
    h = hashlib.md5(f"{a.shape}{a.dtype}".encode())
    h.update(a.tobytes())
    return h.hexdigest()
