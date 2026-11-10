"""Display the C-arm image received on 2026-11-10; no overlay imports.

Run in decaf: python show_received_dicom.py
Optional: --window 1 (second stored window), --check (no GUI).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut


IMAGE_NAME = "20261110T195107_516456Z_ae99dc64d6c04906a61d4d5283d2e3fc.dcm"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=0,
                        help="Stored window index (0, 1, ...; default: 0)")
    parser.add_argument("--check", action="store_true",
                        help="Decode and render without opening a window")
    args = parser.parse_args()
    path = Path(__file__).resolve().parent / "received_dicom" / IMAGE_NAME
    if not path.is_file():
        parser.error(f"DICOM file not found: {path}")
    ds = pydicom.dcmread(path)
    windows = ds.get("WindowCenter")
    count = len(windows) if isinstance(windows, pydicom.multival.MultiValue) else 1
    if not 0 <= args.window < count:
        parser.error(f"Window index must be between 0 and {count - 1}")
    pixels = ds.pixel_array
    if int(ds.get("NumberOfFrames", 1)) > 1:
        pixels = pixels[0]
    if pixels.ndim != 2:
        parser.error(f"Expected a grayscale image, received shape {pixels.shape}")
    image = apply_modality_lut(pixels, ds)
    image = apply_voi_lut(image, ds, index=args.window, prefer_lut=False)
    cmap = "gray_r" if ds.PhotometricInterpretation == "MONOCHROME1" else "gray"
    if args.check:
        plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(9, 9))
    # Preserve the full window range rather than stretching just the present pixels.
    limits = {}
    if windows is not None and "WindowWidth" in ds and "VOILUTSequence" not in ds:
        signed = int(ds.get("PixelRepresentation", 0))
        bits = int(ds.BitsStored)
        low = -(2 ** (bits - 1)) if signed else 0
        high = 2 ** (bits - signed) - 1
        if "ModalityLUTSequence" in ds:
            low, high = 0, 2 ** int(ds.ModalityLUTSequence[0].LUTDescriptor[2]) - 1
        else:
            slope = float(ds.get("RescaleSlope", 1))
            intercept = float(ds.get("RescaleIntercept", 0))
            low, high = low * slope + intercept, high * slope + intercept
        limits = dict(vmin=min(low, high), vmax=max(low, high))
    ax.imshow(image, cmap=cmap, interpolation="nearest", aspect="equal", **limits)
    ax.set_title(f"C-Arm DICOM | {pixels.shape[1]} x {pixels.shape[0]} | Window {args.window + 1}/{count}")
    ax.set_axis_off()
    fig.tight_layout()
    print(f"Loaded: {path}")
    print(f"Pixels: {pixels.shape}; raw range: {np.min(pixels)} .. {np.max(pixels)}")
    print(f"Window {args.window + 1}/{count}; {ds.PhotometricInterpretation}")
    if args.check:
        fig.canvas.draw()
        plt.close(fig)
        print("OK: decoded and rendered successfully.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
