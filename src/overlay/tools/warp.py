# overlay/tools/warp.py

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ============================================================
# Validation helpers
# ============================================================

def _as_homography(H: np.ndarray, name: str = "H_xc") -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3,3), got {H.shape}")
    if not np.isfinite(H).all():
        raise ValueError(f"{name} contains non-finite values.")
    return H


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    return alpha


# ============================================================
# X-ray FOV mask detection (SEGMENTATION-BASED)
# ============================================================

def _smooth_image(img_u8: np.ndarray, ksize: int = 9) -> np.ndarray:
    if img_u8.ndim != 2:
        raise ValueError(f"img_u8 must be grayscale, got {img_u8.shape}")
    if img_u8.dtype != np.uint8:
        raise ValueError(f"img_u8 must be uint8, got {img_u8.dtype}")
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd.")
    return cv2.GaussianBlur(img_u8, (ksize, ksize), 0)


def _threshold_fov_region(img_u8: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(
        img_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return mask


def _largest_connected_component(mask_u8: np.ndarray) -> np.ndarray:
    mask_bin = (mask_u8 > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin,
        connectivity=8,
    )

    if num_labels <= 1:
        raise RuntimeError("No foreground component found.")

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    out = np.zeros_like(mask_u8)
    out[labels == largest_label] = 255
    return out


def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    h, w = mask_u8.shape
    flood = mask_u8.copy()

    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask_u8, flood_inv)


def _morph_cleanup(mask_u8: np.ndarray) -> np.ndarray:
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    return mask


def _largest_inscribed_circle_mask(mask_u8: np.ndarray, margin_px: int = 0) -> np.ndarray:
    """
    Convert a binary foreground mask into the largest inscribed circle mask.

    Parameters
    ----------
    mask_u8 : (H,W) uint8
        Binary foreground mask, expected values 0 / 255.

    margin_px : int
        Optional safety margin subtracted from the radius.

    Returns
    -------
    circle_mask_u8 : (H,W) uint8
        Binary mask of the largest circle fully inside the foreground region.
    """
    mask_u8 = np.asarray(mask_u8)
    if mask_u8.ndim != 2:
        raise ValueError("mask_u8 must be 2D")
    if mask_u8.dtype != np.uint8:
        raise ValueError("mask_u8 must be uint8")

    fg = (mask_u8 > 0).astype(np.uint8) * 255
    if not np.any(fg):
        raise RuntimeError("Foreground mask is empty.")

    dist = cv2.distanceTransform(fg, distanceType=cv2.DIST_L2, maskSize=5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)

    radius = int(np.floor(max_val)) - int(margin_px)
    if radius <= 0:
        raise RuntimeError("Largest inscribed circle radius is non-positive.")

    circle_mask = np.zeros_like(mask_u8)
    cv2.circle(circle_mask, max_loc, radius, 255, thickness=cv2.FILLED)
    return circle_mask


def _detect_xray_fov_mask(xray_gray_u8: np.ndarray) -> np.ndarray:
    """
    Detect visible X-ray FOV region via segmentation.
    """
    xray_gray_u8 = np.asarray(xray_gray_u8)
    if xray_gray_u8.ndim != 2:
        raise ValueError("xray must be grayscale")
    if xray_gray_u8.dtype != np.uint8:
        raise ValueError("xray must be uint8")

    blur = _smooth_image(xray_gray_u8)
    raw = _threshold_fov_region(blur)
    largest = _largest_connected_component(raw)
    filled = _fill_holes(largest)
    final = _morph_cleanup(filled)
    
    return final
    
    #circle = _largest_inscribed_circle_mask(final, margin_px=2)
    #return circle


# ============================================================
# Warped overlay cache
# ============================================================

@dataclass
class WarpedOverlay:
    warped_xray_gray: np.ndarray
    overlay_mask: np.ndarray

    def blend(self, camera_bgr: np.ndarray, alpha: float) -> np.ndarray:
        alpha = _validate_alpha(alpha)

        out = camera_bgr.copy()
        mask = self.overlay_mask > 0

        if not np.any(mask):
            return out

        xray_bgr = cv2.cvtColor(self.warped_xray_gray, cv2.COLOR_GRAY2BGR)

        cam_f = camera_bgr.astype(np.float32)
        xray_f = xray_bgr.astype(np.float32)

        blended = alpha * cam_f + (1 - alpha) * xray_f
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        out[mask] = blended[mask]
        return out


# ============================================================
# Live overlay - prepared once, then only the covered pixels
# ============================================================

@dataclass
class LiveOverlay:
    """
    Overlay prepared for repeated blending into a BGRA frame.

    Everything that does not change between frames is done once: the warp, the
    FOV mask, the colour expansion, and the bounding box of the covered area.
    Per frame only the pixels the overlay actually covers are touched - on a
    1920x1080 frame that is about 6 % of the image.

    Deliberately separate from WarpedOverlay: that one stays byte-for-byte as
    it was so the preview page and the offline analyses keep reproducing.
    """

    box: tuple[int, int, int, int]      # y0, y1, x0, x1
    xray_roi: np.ndarray                # (h, w, 4) BGRA
    mask_roi: np.ndarray                # (h, w) uint8, 0/255
    _scratch: np.ndarray

    def blend_into(self, dst_bgra: np.ndarray, alpha: float) -> None:
        """Blend in place. dst_bgra is modified, nothing is allocated."""
        alpha = _validate_alpha(alpha)
        y0, y1, x0, x1 = self.box
        roi = dst_bgra[y0:y1, x0:x1]
        cv2.addWeighted(roi, alpha, self.xray_roi, 1.0 - alpha, 0.0, dst=self._scratch)
        cv2.copyTo(self._scratch, self.mask_roi, roi)


def prepare_live_overlay(
    xray_gray_u8: np.ndarray,
    H_xc: np.ndarray,
    camera_size_wh: tuple[int, int],
) -> LiveOverlay | None:
    """
    Build a LiveOverlay. Returns None when the warp lands outside the frame.

    Uses the same FOV detection and warp as warp_xray_to_camera, so the result
    is visually identical to the preview path.
    """
    xray_gray_u8 = np.asarray(xray_gray_u8)
    H_xc = _as_homography(H_xc, name="H_xc")
    Wc, Hc = int(camera_size_wh[0]), int(camera_size_wh[1])

    fov = _detect_xray_fov_mask(xray_gray_u8)
    masked = cv2.bitwise_and(xray_gray_u8, xray_gray_u8, mask=fov)

    warped = cv2.warpPerspective(masked, H_xc, (Wc, Hc), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpPerspective(fov, H_xc, (Wc, Hc), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = (mask > 0).astype(np.uint8) * 255

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    xray_roi = cv2.cvtColor(warped[y0:y1, x0:x1], cv2.COLOR_GRAY2BGRA)
    xray_roi[:, :, 3] = 255

    return LiveOverlay(
        box=(y0, y1, x0, x1),
        xray_roi=np.ascontiguousarray(xray_roi),
        mask_roi=np.ascontiguousarray(mask[y0:y1, x0:x1]),
        _scratch=np.empty_like(xray_roi),
    )


# ============================================================
# Public API
# ============================================================

def _distortion_aware_maps(
    H_xc: np.ndarray,
    K_camera: np.ndarray,
    dist_coeffs: np.ndarray,
    camera_size_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build remap tables for  xray px --H_xc--> ideal px --distort--> observed px.

    H_xc maps X-ray pixels to IDEAL pinhole camera pixels, but the camera image
    is distorted. Warping with H_xc alone therefore places the X-ray content
    where a distortion-free camera would have imaged it.

    The remap runs backwards: for every observed output pixel, undistort it to
    the ideal pixel, then apply H_xc^-1 to find the X-ray source pixel.
    """
    Wc, Hc = int(camera_size_wh[0]), int(camera_size_wh[1])

    uu, vv = np.meshgrid(
        np.arange(Wc, dtype=np.float32),
        np.arange(Hc, dtype=np.float32),
    )
    observed = np.stack([uu.ravel(), vv.ravel()], axis=1).reshape(-1, 1, 2)

    ideal = cv2.undistortPoints(
        observed, K_camera, dist_coeffs, P=K_camera
    ).reshape(-1, 2)

    q = (np.linalg.inv(H_xc) @ np.hstack([ideal, np.ones((len(ideal), 1))]).T).T
    src = q[:, :2] / q[:, 2:3]

    map_x = np.ascontiguousarray(src[:, 0].reshape(Hc, Wc).astype(np.float32))
    map_y = np.ascontiguousarray(src[:, 1].reshape(Hc, Wc).astype(np.float32))
    return map_x, map_y


def warp_xray_to_camera(
    xray_gray_u8: np.ndarray,
    H_xc: np.ndarray,
    camera_size_wh: tuple[int, int],
    *,
    K_camera: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
) -> WarpedOverlay:
    """
    Warp a grayscale X-ray image and its detected FOV mask into camera coordinates.

    Parameters
    ----------
    xray_gray_u8 : (Hx,Wx) uint8
        Grayscale X-ray image.

    H_xc : (3,3) float64
        Homography mapping X-ray pixels to camera pixels:

            u_c ~ H_xc * u_x

    camera_size_wh : (Wc,Hc)
        Output size in camera image coordinates.

    K_camera, dist_coeffs : optional
        RGB intrinsics and distortion coefficients. If both are given and the
        coefficients are non-zero, the lens distortion is modelled in the warp
        (see _distortion_aware_maps). Otherwise the plain projective warp is
        used, which assumes an undistorted camera.

    Returns
    -------
    cache : WarpedOverlay
        Cached warped X-ray image and warped overlay mask.
    """
    xray_gray_u8 = np.asarray(xray_gray_u8)
    if xray_gray_u8.ndim != 2:
        raise ValueError(
            f"xray_gray_u8 must be grayscale, got shape {xray_gray_u8.shape}"
        )
    if xray_gray_u8.dtype != np.uint8:
        raise ValueError(f"xray_gray_u8 must be uint8, got {xray_gray_u8.dtype}")

    H_xc = _as_homography(H_xc, name="H_xc")

    if len(camera_size_wh) != 2:
        raise ValueError(f"camera_size_wh must be (W,H), got {camera_size_wh}")

    Wc = int(camera_size_wh[0])
    Hc = int(camera_size_wh[1])
    if Wc <= 0 or Hc <= 0:
        raise ValueError(f"camera_size_wh must be positive, got {camera_size_wh}")

    # detect valid X-ray FOV region
    xray_fov_mask = _detect_xray_fov_mask(xray_gray_u8)

    # restrict X-ray to valid FOV only
    xray_masked = cv2.bitwise_and(xray_gray_u8, xray_gray_u8, mask=xray_fov_mask)

    use_distortion = (
        K_camera is not None
        and dist_coeffs is not None
        and np.any(np.asarray(dist_coeffs, dtype=np.float64))
    )

    if use_distortion:
        map_x, map_y = _distortion_aware_maps(
            H_xc,
            np.asarray(K_camera, dtype=np.float64).reshape(3, 3),
            np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
            (Wc, Hc),
        )
        warped_xray_gray = cv2.remap(
            xray_masked, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        overlay_mask = cv2.remap(
            xray_fov_mask, map_x, map_y, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    else:
        # warp masked X-ray image into camera image coordinates
        warped_xray_gray = cv2.warpPerspective(
            xray_masked,
            H_xc,
            dsize=(Wc, Hc),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # warp FOV mask -> actual overlay region in camera image
        overlay_mask = cv2.warpPerspective(
            xray_fov_mask,
            H_xc,
            dsize=(Wc, Hc),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    overlay_mask = (overlay_mask > 0).astype(np.uint8) * 255

    return WarpedOverlay(
        warped_xray_gray=warped_xray_gray,
        overlay_mask=overlay_mask,
    )


def blend_xray_overlay(
    camera_bgr: np.ndarray,
    xray_gray_u8: np.ndarray,
    H_xc: np.ndarray,
    alpha: float,
    *,
    K_camera: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
) -> tuple[np.ndarray, WarpedOverlay]:
    """
    Convenience function:
    - detect X-ray FOV mask
    - warp grayscale X-ray image and FOV mask into camera coordinates
    - blend with camera image

    After the first call, the returned WarpedOverlay cache can be reused
    for cheap alpha-only updates.

    Parameters
    ----------
    camera_bgr : (Hc,Wc,3) uint8
        Camera image.

    xray_gray_u8 : (Hx,Wx) uint8
        Grayscale X-ray image.

    H_xc : (3,3) float64
        Homography mapping X-ray pixels to camera pixels.

    alpha : float
        Blend parameter in [0,1].

    Returns
    -------
    out_bgr : (Hc,Wc,3) uint8
        Blended image.

    cache : WarpedOverlay
        Cached warped X-ray image and overlay mask.
    """
    alpha = _validate_alpha(alpha)

    camera_bgr = np.asarray(camera_bgr)
    if camera_bgr.ndim != 3 or camera_bgr.shape[2] != 3:
        raise ValueError(
            f"camera_bgr must have shape (H,W,3), got {camera_bgr.shape}"
        )
    if camera_bgr.dtype != np.uint8:
        raise ValueError(f"camera_bgr must be uint8, got {camera_bgr.dtype}")

    Hc, Wc = camera_bgr.shape[:2]

    cache = warp_xray_to_camera(
        xray_gray_u8=xray_gray_u8,
        H_xc=H_xc,
        camera_size_wh=(Wc, Hc),
        K_camera=K_camera,
        dist_coeffs=dist_coeffs,
    )

    out_bgr = cache.blend(camera_bgr, alpha=alpha)
    return out_bgr, cache


