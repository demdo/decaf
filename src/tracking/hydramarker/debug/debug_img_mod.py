from pathlib import Path
import sys

import cv2
import numpy as np

from PySide6.QtWidgets import QApplication, QFileDialog


# ============================================================
# USE EXISTING QApplication IF ALREADY RUNNING
# ============================================================

app = QApplication.instance()

if app is None:
    app = QApplication(sys.argv)


# ============================================================
# QT FILE DIALOG
# ============================================================

img_path, _ = QFileDialog.getOpenFileName(
    None,
    "Select HydraMarker Image",
    "",
    "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
)

if not img_path:
    raise RuntimeError("No image selected")


img_path = Path(img_path)


# ============================================================
# OUTPUT FOLDER
# ============================================================

out_dir = img_path.parent / "debug_img_mod"
out_dir.mkdir(exist_ok=True)


# ============================================================
# LOAD IMAGE
# ============================================================

img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise RuntimeError(f"Could not load image:\n{img_path}")


# ============================================================
# HELPERS
# ============================================================

def save(name, image):
    out_path = out_dir / name
    cv2.imwrite(str(out_path), image)
    print("saved:", out_path)


def add_gaussian_noise(image, sigma):
    noise = np.random.normal(
        0,
        sigma,
        image.shape
    ).astype(np.float32)

    out = image.astype(np.float32) + noise

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gradient(image, strength=0.4):
    h, w = image.shape

    x = np.linspace(0.0, 1.0, w, dtype=np.float32)

    gradient = (
        1.0 - strength +
        strength * x
    )

    gradient = np.tile(
        gradient[None, :],
        (h, 1)
    )

    out = image.astype(np.float32) * gradient

    return np.clip(out, 0, 255).astype(np.uint8)


def darken(image, factor):
    out = image.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================
# SAVE ORIGINAL
# ============================================================

save("00_original.png", img)


# ============================================================
# BLUR
# ============================================================

blur_1 = cv2.GaussianBlur(img, (5, 5), 1.0)
blur_2 = cv2.GaussianBlur(img, (9, 9), 2.0)
blur_3 = cv2.GaussianBlur(img, (15, 15), 4.0)

save("01_blur_light.png", blur_1)
save("02_blur_medium.png", blur_2)
save("03_blur_strong.png", blur_3)


# ============================================================
# DARKER
# ============================================================

dark_1 = darken(img, 0.8)
dark_2 = darken(img, 0.6)
dark_3 = darken(img, 0.4)

save("04_dark_light.png", dark_1)
save("05_dark_medium.png", dark_2)
save("06_dark_strong.png", dark_3)


# ============================================================
# GRADIENT
# ============================================================

grad_1 = apply_gradient(img, 0.3)
grad_2 = apply_gradient(img, 0.5)
grad_3 = apply_gradient(img, 0.7)

save("07_gradient_light.png", grad_1)
save("08_gradient_medium.png", grad_2)
save("09_gradient_strong.png", grad_3)


# ============================================================
# NOISE
# ============================================================

noise_1 = add_gaussian_noise(img, 5)
noise_2 = add_gaussian_noise(img, 10)
noise_3 = add_gaussian_noise(img, 20)

save("10_noise_light.png", noise_1)
save("11_noise_medium.png", noise_2)
save("12_noise_strong.png", noise_3)


# ============================================================
# COMBINED REALISTIC
# ============================================================

combo_1 = darken(img, 0.75)
combo_1 = cv2.GaussianBlur(combo_1, (9, 9), 2.0)
combo_1 = apply_gradient(combo_1, 0.4)
combo_1 = add_gaussian_noise(combo_1, 8)

save("13_combo_realistic_1.png", combo_1)


combo_2 = darken(img, 0.6)
combo_2 = cv2.GaussianBlur(combo_2, (15, 15), 4.0)
combo_2 = apply_gradient(combo_2, 0.6)
combo_2 = add_gaussian_noise(combo_2, 15)

save("14_combo_realistic_2.png", combo_2)


print()
print("========================================")
print("DONE")
print("saved to:")
print(out_dir)
print("========================================")